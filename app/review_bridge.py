"""Review-queue, Scanned-browse, and Discarded-browse backend for the
Peshawar beta app.

Three flat pools now instead of one client's worth of nested folders:
  - review/    pages positively classified as one of the five relevant
               book types, but whose number didn't come out confidently.
  - scanned/<Type>/  auto-filed pages, grouped by book type.
  - discarded/ everything the pipeline decided wasn't a relevant page at
               all -- the riskiest bucket, since nothing normally looks at
               it; list_discarded_items/restore_discarded_item exist
               specifically so a human can catch a wrong call here.

submit_correction is the one move-and-rename operation everything else
(a human confirming a review page, the Gemini auto-fill pass, a human
restoring a wrongly-discarded page, a human fixing a wrongly-detected
scanned page) reduces to -- see its own docstring.

The AI pass is HYBRID (see _process_with_ai): it files a page only when
Gemini's answer matches a number the deterministic pipeline independently
nominated, and otherwise leaves the page in review with that answer
attached as a suggestion the reviewer confirms in one click. Gemini alone
measures ~97% precision, which is not good enough to write into the archive
unchecked, but is very good as a pre-filled starting point for a human.
"""
import base64
import concurrent.futures
import csv
import glob
import json
import os
import re
import shutil
import sys
import threading
import time

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_BETA_DIR = os.path.dirname(_APP_DIR)
if _BETA_DIR not in sys.path:
    sys.path.insert(0, _BETA_DIR)

import fitz  # noqa: E402
import gemini_assist  # noqa: E402
import main as beta_main  # noqa: E402

from pipeline_bridge import DEFAULT_OUTPUT  # noqa: E402

RENDER_WIDTH = 1400
TYPE_FOLDERS = beta_main.detect.TYPE_FOLDERS


def _render_jpeg_bytes(pdf_path):
    with fitz.open(pdf_path) as doc:
        page = doc[0]
        zoom = RENDER_WIDTH / page.rect.width
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pix.tobytes("jpeg")


def render_page_image(pdf_path):
    jpeg_bytes = _render_jpeg_bytes(pdf_path)
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")


def rotate_page(pdf_path, degrees):
    if degrees % 90 != 0:
        raise ValueError(f"rotation must be a multiple of 90, got {degrees}")
    if not os.path.exists(pdf_path):
        raise ValueError("this page isn't there anymore -- refresh and try again")
    with fitz.open(pdf_path) as doc:
        page = doc[0]
        page.set_rotation((page.rotation + degrees) % 360)
        tmp_path = pdf_path + ".tmp"
        doc.save(tmp_path)
    os.replace(tmp_path, pdf_path)
    _ai_futures.pop(pdf_path, None)  # stale -- was computed on the pre-rotation image


def submit_correction(pdf_path, number, type_key, output_root=DEFAULT_OUTPUT, note="corrected"):
    """Move `pdf_path` (wherever it currently sits -- review/, discarded/,
    or even another type's scanned/ folder) into scanned/<type>/<number>,
    and log it to results.csv. The one move-and-rename operation shared by:
    a human confirming a review page, the Gemini auto-fill pass, a human
    restoring a wrongly-discarded page, and a human renaming a wrongly-
    detected scanned page.
    """
    number = number.strip()
    if not number.isdigit():
        raise ValueError(f"not a plain number: {number!r}")
    if type_key not in TYPE_FOLDERS:
        raise ValueError(f"unknown type {type_key!r} -- expected one of {sorted(TYPE_FOLDERS)}")
    if not os.path.exists(pdf_path):
        # The list the UI is looking at is a snapshot from whenever it was
        # last loaded -- another correction (a human, or the AI pass) could
        # have already moved this exact file. Fail with a message the UI
        # can act on (refresh) instead of a raw FileNotFoundError below.
        raise ValueError("this page isn't there anymore -- refresh and try again")

    type_dir = os.path.join(output_root, beta_main.SCANNED_DIRNAME, TYPE_FOLDERS[type_key])
    os.makedirs(type_dir, exist_ok=True)
    ext = os.path.splitext(pdf_path)[1]
    dest = beta_main.unique_destination(type_dir, number + ext)
    if os.path.abspath(dest) == os.path.abspath(pdf_path):
        return {"destination": os.path.relpath(dest, output_root)}
    shutil.move(pdf_path, dest)

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    csv_path = os.path.join(output_root, "results.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["source_file", "page", "status", "type", "destination", "timestamp"])
        writer.writerow([os.path.basename(pdf_path), "", note, type_key, os.path.relpath(dest, output_root), now])

    _ai_futures.pop(pdf_path, None)
    return {"destination": os.path.relpath(dest, output_root)}


# ---- Review queue ----

def _detected_types(output_root):
    """{review filename: book type} for pages already in the review queue.

    The pipeline DID classify these pages -- only their number was
    unreadable -- but that answer was never surfaced to the UI, so the book
    type dropdown defaulted to whichever option happened to be first. A
    reviewer confirming a Delivery Challan without noticing therefore filed
    it into Gas Cylinder, silently and with no error. The type is recovered
    here from the pipeline's own logs.

    Read from both logs because a page reaches review two ways: routed there
    by a run (results.csv), or rescued out of the discard pile
    (rescue_log.csv). The rescue log wins on conflict -- it is the later
    decision about that page.
    """
    types = {}
    for filename, type_col, status_col, want in (
        ("results.csv", "type", "status", "review"),
        ("rescue_log.csv", "page_type", "status", "review"),
    ):
        path = os.path.join(output_root, filename)
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if row.get(status_col) != want:
                        continue
                    dest = row.get("destination")
                    kind = row.get(type_col)
                    if dest and kind in TYPE_FOLDERS:
                        types[os.path.basename(dest)] = kind
        except (OSError, csv.Error):
            continue
    return types


def list_review_items(output_root=DEFAULT_OUTPUT):
    review_dir = os.path.join(output_root, beta_main.REVIEW_DIRNAME)
    suggestions = _load_suggestions(output_root)
    detected = _detected_types(output_root)
    items = []
    for p in sorted(glob.glob(os.path.join(review_dir, "*.pdf"))):
        name = os.path.basename(p)
        item = {"path": p, "filename": name}
        if name in detected:
            item["detected_type"] = detected[name]
        hint = suggestions.get(name)
        if hint:
            # The AI read a number here but no second reader confirmed it,
            # so it is offered rather than applied -- one click instead of
            # typing, with a human still deciding.
            item["suggested_number"] = hint.get("number", "")
            item["suggested_type"] = hint.get("type", "")
        items.append(item)
    return items


# AI suggestions are cached by path for the lifetime of the app process, as
# Futures rather than plain values -- same pattern as the main archive's
# review_bridge.py, see its comment for why (one code path for prefetch and
# on-demand, no page ever asked twice). Unlike the archive, a confident
# result here files the page directly rather than just returning a
# suggestion -- see this module's docstring.
_ai_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
_ai_futures = {}


_SUGGESTION_FILE = ".suggestions.json"
_suggestion_lock = threading.Lock()


def _suggestion_path(output_root):
    return os.path.join(output_root, _SUGGESTION_FILE)


def _load_suggestions(output_root):
    """Suggestions persist on disk, not in memory, so they survive the app
    being closed -- an AI pass costs real API quota and should not have to be
    re-run just because someone restarted the window."""
    path = _suggestion_path(output_root)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (ValueError, OSError):
        return {}


def _write_suggestions(data, output_root):
    tmp = _suggestion_path(output_root) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    os.replace(tmp, _suggestion_path(output_root))


def _save_suggestion(pdf_path, number, type_key, output_root):
    with _suggestion_lock:
        data = _load_suggestions(output_root)
        data[os.path.basename(pdf_path)] = {"number": number, "type": type_key}
        _write_suggestions(data, output_root)


def _clear_suggestion(pdf_path, output_root):
    with _suggestion_lock:
        data = _load_suggestions(output_root)
        if data.pop(os.path.basename(pdf_path), None) is not None:
            _write_suggestions(data, output_root)


def _nominations(pdf_path):
    """Every number the deterministic pipeline nominated for this page --
    trusted or not.

    This deliberately includes candidates the gate REJECTED. A candidate the
    detector nominated but couldn't confirm is still an independent piece of
    evidence about what is printed there; it just wasn't strong enough to
    file on its own. That is exactly the signal worth cross-checking against
    a second opinion.
    """
    from pipeline.crop import candidate_field
    from pipeline.recognize import (
        DETECTION_VALID_NUMBER, DETECTION_ZOOMS, read_number,
    )
    from pipeline.render import render_page

    detect = beta_main.detect
    try:
        match_img, crop_img = render_page(pdf_path, 0)
    except Exception:
        return set()

    found = set()
    for rot in detect.ROTATIONS:
        m = detect.rotate(match_img, rot)
        page_type, candidates = detect.scan(m)
        if page_type == "other" or not candidates:
            continue
        crop_rot = detect.rotate(crop_img, rot) if rot else crop_img
        for candidate in candidates:
            found.add(candidate.digits)
            patch = candidate_field(crop_rot, candidate, m.shape)
            if patch is None:
                continue
            reading = read_number(
                patch, zooms=DETECTION_ZOOMS, valid_number=DETECTION_VALID_NUMBER
            )
            if reading is not None:
                found.add(reading.text)
        break
    return found


def _same_number(a, b):
    """Compare ignoring leading zeros -- "03" and "3" are the same book
    number written two ways, and treating them as a disagreement would
    throw away a genuine confirmation."""
    return a.lstrip("0") == b.lstrip("0") and a.strip() != ""


def _process_with_ai(pdf_path, output_root):
    """Hybrid: file only when two INDEPENDENT readers agree; otherwise leave
    the page in review with the AI's answer attached as a suggestion.

    Gemini alone measures ~97% precision (32/33 against known answers), so
    auto-filing everything it reads would put roughly 3 wrong numbers in
    every 100 into the archive. Requiring it to match something the
    deterministic pipeline independently nominated is the same principle
    the CRNN rescue uses, and for the same reason: a single confident
    opinion is not evidence, two decorrelated ones are.

    A page that doesn't reach agreement is NOT a failure -- it keeps the
    suggestion, so the reviewer confirms a pre-filled number in one click
    instead of typing it. That is where most of the human time saving
    actually comes from.
    """
    if not gemini_assist.available() or not os.path.exists(pdf_path):
        return {"filed": False}
    suggestion = gemini_assist.suggest(_render_jpeg_bytes(pdf_path))
    if suggestion is None:
        return {"filed": False}

    number, type_key = str(suggestion["number"]), suggestion["type"]
    agreed = any(_same_number(number, n) for n in _nominations(pdf_path))

    if not agreed:
        _save_suggestion(pdf_path, number, type_key, output_root)
        return {"filed": False, "suggested": True, "number": number, "type": type_key}

    try:
        result = submit_correction(pdf_path, number, type_key, output_root, note="ai_confirmed")
    except ValueError:
        return {"filed": False}  # moved/gone already -- a human beat the AI to it
    _clear_suggestion(pdf_path, output_root)
    return {
        "filed": True, "type": type_key, "number": number,
        "destination": result["destination"],
    }


def run_ai_on_review_queue(output_root=DEFAULT_OUTPUT):
    """Fire-and-forget: start (or resume) a Gemini pass over every current
    review-queue item not already queued or done. Confident reads get filed
    straight into scanned/ in the background; anything Gemini can't
    confidently place stays in review untouched. Safe to call repeatedly --
    already-submitted items are never resubmitted (their Future is reused).
    """
    if not gemini_assist.available():
        return
    for item in list_review_items(output_root):
        path = item["path"]
        if path not in _ai_futures:
            _ai_futures[path] = _ai_executor.submit(_process_with_ai, path, output_root)


def ai_review_status():
    """{"available", "total", "done", "filed", "suggested", "used", "limit",
    "official_limit"} -- lets the UI show both a progress readout for the
    background AI pass AND how much of the rolling-24h call cap (see
    gemini_assist.MAX_CALLS_PER_DAY, and Google's own real ceiling,
    OFFICIAL_FREE_LIMIT_PER_DAY) has been used, without blocking on any
    individual page."""
    if not gemini_assist.available():
        return {"available": False, "total": 0, "done": 0, "filed": 0, "suggested": 0,
                "used": 0, "limit": 0, "official_limit": 0}
    done = [f for f in _ai_futures.values() if f.done()]
    filed = sum(1 for f in done if f.result().get("filed"))
    suggested = sum(1 for f in done if f.result().get("suggested"))
    usage = gemini_assist.usage_status()
    return {
        "available": True, "total": len(_ai_futures), "done": len(done), "filed": filed,
        "suggested": suggested,
        "used": usage["used"], "limit": usage["limit"], "official_limit": usage["official_limit"],
    }


# ---- Scanned browse ----

def list_scanned_items(type_key=None, output_root=DEFAULT_OUTPUT):
    scanned_root = os.path.join(output_root, beta_main.SCANNED_DIRNAME)
    keys = [type_key] if type_key else list(TYPE_FOLDERS)
    items = []
    for key in keys:
        folder = TYPE_FOLDERS.get(key)
        if not folder:
            continue
        for p in sorted(glob.glob(os.path.join(scanned_root, folder, "*.pdf"))):
            items.append({"path": p, "type": key, "type_label": folder, "filename": os.path.basename(p)})
    return items


def scanned_type_counts(output_root=DEFAULT_OUTPUT):
    scanned_root = os.path.join(output_root, beta_main.SCANNED_DIRNAME)
    return [
        {"type": key, "label": folder, "count": len(glob.glob(os.path.join(scanned_root, folder, "*.pdf")))}
        for key, folder in TYPE_FOLDERS.items()
    ]


def rename_scanned_item(pdf_path, number, type_key, output_root=DEFAULT_OUTPUT):
    """A human caught a wrongly-detected or wrongly-typed scanned page and
    is fixing its number and/or book type."""
    return submit_correction(pdf_path, number, type_key, output_root, note="renamed")


# ---- Discarded browse ----

def _discard_reasons(output_root):
    """{relative_destination_path: matched_type} from discarded.csv, so the
    browse view can show *why* each page was discarded -- helps a human
    judge whether a given page is worth a closer look (an "unclassified"
    page is far more likely to be a real miss than a "petty_cash_ledger"
    one)."""
    csv_path = os.path.join(output_root, "discarded.csv")
    reasons = {}
    if not os.path.exists(csv_path):
        return reasons
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            dest = row.get("destination")
            if dest:
                reasons[dest] = row.get("matched_type", "")
    return reasons


def list_discarded_items(output_root=DEFAULT_OUTPUT):
    discarded_dir = os.path.join(output_root, beta_main.DISCARDED_DIRNAME)
    reasons = _discard_reasons(output_root)
    items = []
    for p in sorted(glob.glob(os.path.join(discarded_dir, "*.pdf"))):
        rel = os.path.relpath(p, output_root)
        items.append({"path": p, "filename": os.path.basename(p), "reason": reasons.get(rel, "")})
    return items


def list_duplicate_groups(output_root=DEFAULT_OUTPUT, min_count=2):
    """Numbers filed more than once within the same book, worst first.

    A pre-printed book number should be unique within a year, so a value
    appearing many times is strong evidence the pipeline read the wrong
    field. Real examples found this way: "2232" filed 38 times and "7940"
    17 times -- both VEHICLE numbers sitting near the real field.

    Some repetition is legitimate: the client confirmed these numbers reset
    each year, so the same number recurs across a year boundary. That makes
    the COUNT the signal rather than the mere fact of a duplicate -- twice
    is ordinary, thirty-eight times is a bug. Sorted by count so the
    diagnostic ones surface first.
    """
    groups = []
    for type_key, folder in TYPE_FOLDERS.items():
        by_number = {}
        pattern = os.path.join(output_root, beta_main.SCANNED_DIRNAME, folder, "*")
        for path in glob.glob(pattern):
            stem = os.path.splitext(os.path.basename(path))[0]
            stem = re.sub(r"_\d+$", "", stem)   # strip the collision suffix
            if stem.isdigit():
                by_number.setdefault(stem, []).append(path)
        for number, paths in by_number.items():
            if len(paths) >= min_count:
                groups.append({
                    "type": type_key,
                    "type_label": folder,
                    "number": number,
                    "count": len(paths),
                })
    groups.sort(key=lambda g: (-g["count"], g["type_label"], int(g["number"])))
    return groups


def list_outlier_groups(output_root=DEFAULT_OUTPUT):
    """Numbers sitting far outside their book's dense range.

    A pre-printed book runs as a near-continuous series, so a value far
    outside it is almost never a real document number -- it is a misread, or
    a different field entirely. Measured on this archive: Gas Cylinder's real
    numbers run 157-2885, and the outliers were 22222, 115854, 130013;
    Liquid Nitrogen runs 111-2316 with an outlier at 458510.

    "Far outside" is defined from the book's own data rather than a fixed
    threshold, since each book has its own range: take the 5th-95th
    percentile as the core, then flag anything more than half that width
    beyond either end. That way a book with a genuinely wide range is not
    punished for it.
    """
    groups = []
    for type_key, folder in TYPE_FOLDERS.items():
        by_number = {}
        pattern = os.path.join(output_root, beta_main.SCANNED_DIRNAME, folder, "*")
        for path in glob.glob(pattern):
            stem = os.path.splitext(os.path.basename(path))[0]
            stem = re.sub(r"_\d+$", "", stem)
            if stem.isdigit():
                by_number.setdefault(int(stem), []).append(path)
        if len(by_number) < 20:
            continue  # too few to say what "outside the range" even means
        uniq = sorted(by_number)
        core = uniq[len(uniq) // 20: max(1, len(uniq) * 19 // 20)] or uniq
        lo, hi = core[0], core[-1]
        margin = max(1, hi - lo) * 0.5
        for number in uniq:
            if number < lo - margin or number > hi + margin:
                groups.append({
                    "type": type_key,
                    "type_label": folder,
                    "number": str(number),
                    "count": len(by_number[number]),
                    "core": f"{lo}-{hi}",
                })
    groups.sort(key=lambda g: (g["type_label"], -int(g["number"])))
    return groups


def list_anomaly_groups(output_root=DEFAULT_OUTPUT, min_count=3):
    """Everything worth a second look, in one list for the Duplicates view.

    Duplicates come first and ordered by count, because a number filed
    thirty-eight times is a far stronger signal of a wrong field than one
    filed twice -- and twice is often legitimate (yearly reset). min_count
    defaults to 3 for that reason.
    """
    out = []
    for g in list_duplicate_groups(output_root, min_count=min_count):
        out.append({**g, "kind": "duplicate"})
    for g in list_outlier_groups(output_root):
        out.append({**g, "kind": "outlier"})
    return out


def list_duplicate_items(type_key, number, output_root=DEFAULT_OUTPUT):
    """The pages sharing one number inside one book."""
    folder = TYPE_FOLDERS.get(type_key)
    if not folder:
        return []
    items = []
    pattern = os.path.join(output_root, beta_main.SCANNED_DIRNAME, folder, "*")
    for path in sorted(glob.glob(pattern)):
        stem = os.path.splitext(os.path.basename(path))[0]
        if re.sub(r"_\d+$", "", stem) == str(number):
            items.append({
                "path": path,
                "filename": os.path.basename(path),
                "type": type_key,
                "type_label": folder,
            })
    return items


def discard_item(pdf_path, output_root=DEFAULT_OUTPUT, note="manual"):
    """A human decided this page isn't a relevant ECR/DC document at all.

    The exact inverse of restore_discarded_item, and needed for the same
    reason that one is: no automatic classifier is perfect in both
    directions. The visual classifier rescued ~1,548 pages out of the
    discard pile, and roughly 1 in 38 of those is a false alarm (measured) --
    a weighbridge slip or fuel receipt that merely looks like a form. Without
    this, a reviewer meeting one has no honest option: they can only skip it,
    leaving it in the queue forever, or invent a number for a page that has
    none.

    Logged to discarded.csv with matched_type "manual" so these are
    distinguishable from the pipeline's own decisions, and reversible from
    the Discarded view exactly like any other discard.
    """
    if not os.path.exists(pdf_path):
        raise ValueError("this page isn't there anymore -- refresh and try again")

    discarded_dir = os.path.join(output_root, beta_main.DISCARDED_DIRNAME)
    os.makedirs(discarded_dir, exist_ok=True)
    dest = beta_main.unique_destination(discarded_dir, os.path.basename(pdf_path))
    if os.path.abspath(dest) == os.path.abspath(pdf_path):
        return {"destination": os.path.relpath(dest, output_root)}
    shutil.move(pdf_path, dest)

    csv_path = os.path.join(output_root, "discarded.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["source_file", "page", "matched_type", "destination", "timestamp"])
        writer.writerow([os.path.basename(pdf_path), "", note,
                         os.path.relpath(dest, output_root),
                         time.strftime("%Y-%m-%d %H:%M:%S")])

    _ai_futures.pop(pdf_path, None)
    _clear_suggestion(pdf_path, output_root)
    return {"destination": os.path.relpath(dest, output_root)}


def restore_discarded_item(pdf_path, number, type_key, output_root=DEFAULT_OUTPUT):
    """A human found a genuinely correct page in the discard pile -- files
    it into scanned/<type>/ under the given number, same operation as
    confirming a review page."""
    return submit_correction(pdf_path, number, type_key, output_root, note="restored")
