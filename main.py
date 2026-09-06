"""Peshawar warehouse bundle extraction -- standalone beta.

Point this at a folder of day-bundle PDFs (each one mixing five relevant
ECR/DC book types -- Gas Cylinder, D/A Gas, Liquid Nitrogen, Liquid Tank
Decant, Delivery Challan -- with a long tail of unrelated paperwork:
petty-cash ledgers, stock reports, requisition slips, toll receipts,
payment vouchers, handwritten notes) and it classifies every page: files
the ones it's confident about into a folder named for that book (so a
sequence of numbers can be checked per book, not buried inside whichever
source PDF happened to contain it), discards the ones positively
identified as known-irrelevant (logged, not stored), and sends everything
else to review. Never silently drops a page nobody has positively
identified.

This is a genuinely separate codebase from the production v3/app archive
pipeline -- explicitly a beta for testing this approach on a messier input
shape, kept isolated so nothing here can affect the working app. It reuses
v3's proven render/recognize/gate modules as a starting reference (copied,
not imported) and pipeline/detect.py written fresh for these actual
templates -- see PESHAWAR_NOTES.md for what was measured to calibrate it.

No SQLite ledger for this beta: resumability is a per-file marker (see
mark_file_done) plus physically moving a finished source PDF into
<input>/done/ once every one of its pages is accounted for -- meant for
this to run repeatedly against a whole fiscal year's worth of monthly
bundles, not just one, so a finished month should visibly leave the active
input pile rather than sit there needing to be remembered as "already
done". Not built for the finer-grained per-page resume v3's Ledger gives
the production archive.

Usage (run from the project root):
    .venv-gpu\\Scripts\\python peshawar_beta\\main.py                # process incoming/peshawer/JULY
    .venv-gpu\\Scripts\\python peshawar_beta\\main.py --input DIR    # a different folder of bundle PDFs
    .venv-gpu\\Scripts\\python peshawar_beta\\main.py --limit 50     # quick look, first N pages only
    .venv-gpu\\Scripts\\python peshawar_beta\\main.py --no-copy      # measure only, write nothing
    .venv-gpu\\Scripts\\python peshawar_beta\\main.py --workers 3    # parallel worker processes
"""
import argparse
import collections
import csv
import glob
import multiprocessing
import os
import shutil
import sys
import time
from types import SimpleNamespace

# Windows' console defaults stdout to the system codepage (cp1252) even when
# writing to a redirected/background pipe, which can't represent every
# Unicode filename this warehouse's phones produce (e.g. mathematical
# bold/fraktur lookalike letters some scanning apps use for "DAILY SCANING
# REPORT"). Without this, a single such filename crashes the whole batch on
# its per-page progress print -- silently halting a long unattended run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# GPU-enabled, same pattern as app/desktop_app.py: ECR_USE_GPU must be set
# before anything imports pipeline.recognize or pipeline.crnn_digits (both
# read it once at import time), and each multiprocessing-spawned worker
# re-runs this file's own top-level code before importing further, so
# setting it here is what makes every worker pick it up too. Without this,
# main.py silently ran CPU-only even on the GPU box -- onnxruntime falls
# back to CPU with no error, just several times slower (see gpu_env.py's
# own docstring for why the fallback is silent). verify_gpu() below fails
# loudly instead, once, before any worker spawns.
os.environ["ECR_USE_GPU"] = "1"

import cv2

from pipeline import crnn_digits, detect, pageclass
from pipeline.crop import candidate_field
from pipeline.gate import resolve_page
from pipeline.gpu_env import verify_gpu
from pipeline.recognize import DETECTION_VALID_NUMBER, DETECTION_ZOOMS, Reading, read_number
from pipeline.render import SCALE, SUPPORTED_EXTS, extract_page, page_count, render_page

INPUT_DIR = os.path.join("incoming", "peshawer", "JULY")
OUTPUT_DIR = "peshawar_output"
SCANNED_DIRNAME = "scanned"
REVIEW_DIRNAME = "review"
DISCARDED_DIRNAME = "discarded"

# Source PDFs move here, out of the active input pile, once every one of
# their pages is accounted for. This pipeline is meant to run repeatedly
# against a whole fiscal year's worth of monthly bundles, not just one
# month -- physically moving a finished file out is what lets a human see
# at a glance which files are already done just by looking at the input
# folder, the same way the .done marker lets a future run know without
# re-reading every page.
DONE_SOURCE_DIRNAME = "done"

RELEVANT_TYPES = tuple(detect.TYPE_FOLDERS)  # ("gas_cylinder", "da_gas", ...)

# gate.judge reads anchor.card.name/anchor.score for its diagnostics fields
# only -- pipeline.detect's candidates have no template/match-score of
# their own to report (same reasoning as v3/main.py's _PSEUDO_ANCHOR).
_PSEUDO_ANCHOR = SimpleNamespace(card=SimpleNamespace(name="detected"), score=1.0)


def _init_worker():
    """Pin each worker to a single thread -- OpenCV/ONNX Runtime both
    parallelise internally by default, which is exactly wrong once a Pool
    already owns every core (same reasoning as v3/main.py's _init_worker)."""
    cv2.setNumThreads(1)


def _crnn_rescue(candidate, crop_img, nominated, disagreed):
    """Second opinion from the digit-only CRNN, or None.

    Ported from the archive's v3/main.py. The zoom-consensus gate has one
    structural blind spot: all three zoom variants share a single crop, so
    when that crop is mis-bounded they can agree unanimously on a WRONG
    value and the gate cannot tell. Measured here on real pages: "003" filed
    as "033" and "062" as "02", both at full 3/3 agreement.

    A digit-only model is architecturally decorrelated from RapidOCR's
    ~6625-class one, so it is unlikely to make the identical mistake. It is
    used strictly as a confirmer: it can only agree with a value some other
    signal already produced (the detector's own nomination, or a reread that
    failed to reach agreement). It can never introduce a third value nothing
    else saw. It is also fed the TIGHT detector box, not crop.py's padded
    field -- that padding is sized for RapidOCR's reread and pulls in
    neighbouring content this model has no way to ignore (see
    crnn_digits.MAX_ASPECT, which makes it decline rather than guess).
    """
    x0, y0, x1, y1 = (int(v * SCALE) for v in (candidate.x0, candidate.y0, candidate.x1, candidate.y1))
    if y1 <= y0 or x1 <= x0:
        return None
    prediction = crnn_digits.read_digits(crop_img[y0:y1, x0:x1])
    if prediction is None:
        return None
    if prediction == nominated:
        return Reading(text=nominated, votes=2, variants=2)
    if disagreed is not None and prediction == disagreed:
        return Reading(text=disagreed, votes=2, variants=2)
    return None


def _read_candidates(candidates, crop_img, match_shape):
    readings = []
    for candidate in candidates:
        patch = candidate_field(crop_img, candidate, match_shape)
        reading = (
            read_number(patch, zooms=DETECTION_ZOOMS, valid_number=DETECTION_VALID_NUMBER)
            if patch is not None else None
        )
        # Full zoom-variant agreement only, same bar the archive holds a
        # detected candidate to -- a partial-agreement reading is not
        # trusted here either, gate.judge sends it to review on its own.
        disagreed = None
        if reading is not None and reading.votes < reading.variants:
            if reading.text != candidate.digits:
                disagreed = reading.text
            reading = None

        if reading is None:
            reading = _crnn_rescue(candidate, crop_img, candidate.digits, disagreed)

        readings.append(reading)
    return readings


def analyse_page(work_item):
    """Classify one page, trying each rotation until something legible is
    found (see detect.ROTATIONS's comment for why: most pages resolve at 0
    immediately, but Liquid Nitrogen's table-continuation pages were
    measured coming through sideways in real bundles). "other" doesn't
    stop the search -- every relevant/irrelevant classification requires a
    real marker to be legible, which can't happen at the wrong rotation, so
    "other" at this rotation is not yet proof the page has nothing to find,
    only that this angle didn't find it. If every rotation comes back
    "other", the page settles on rotation 0 -- most such pages are
    genuinely irrelevant content no rotation would help anyway (freeform
    notes, unrelated receipts).

    A classification alone does NOT stop the search either, only a
    classification that also found candidates. A genuinely sideways Gas
    Cylinder card was seen classifying at rotation 0 anyway, because its
    fallback markers (JOB_CARD_REQUIRED_MARKERS, the MCL/CP column headers)
    are short enough that RapidOCR still reads them on a rotated page --
    so the old "first non-other wins" rule locked in rotation 0 and then
    applied the number band to a page lying on its side, where it could
    never match. Preferring a rotation that actually yields candidates
    costs at most three extra scans on pages that were already failing,
    and keeps the first classification as a fallback so a page whose number
    field is genuinely blank still classifies (and reviews) exactly as before.
    """
    path, page_index = work_item
    match_img, crop_img = render_page(path, page_index)

    fallback = None
    for rot in detect.ROTATIONS:
        m = detect.rotate(match_img, rot)
        pt, cands = detect.scan(m)
        if pt == "other":
            continue
        if cands:
            crop_rot = detect.rotate(crop_img, rot) if rot else crop_img
            return path, page_index, pt, _read_candidates(cands, crop_rot, m.shape), rot
        if fallback is None:
            fallback = (pt, rot)

    if fallback is not None:
        page_type, rotation = fallback
        return path, page_index, page_type, [], rotation

    # Nothing matched a marker at any rotation, so this page is about to be
    # discarded unseen. Ask the visual classifier before letting that happen
    # -- a hand audit of v1's discard pile found ~21% of these are genuine
    # ECR/DC pages whose marker text was cropped, faint or misread (see
    # pipeline/pageclass.py). Only a very confident answer overrules the
    # discard, and even then the page still has to clear the normal number
    # gate to be filed: a rescue routes it into the pipeline, it does not
    # hand it a number.
    return _rescue_page(match_img, crop_img, path, page_index)


def _rescue_page(match_img, crop_img, path, page_index):
    """Classify a page no marker matched, AND read its number.

    Tries each rotation, since an unmatched page is often unmatched
    precisely because it is lying on its side -- and keeps the rotation that
    recognised it so the page is written out upright.

    Reading the number here is the whole point, and was missing at first:
    the rescue used to return an empty readings list, so a rescued page
    reached review with its type known, its number frequently perfectly
    legible, and nothing having ever looked for it. Across the archive that
    left ~2,115 rescued pages -- about half of them Liquid Nitrogen -- in
    review purely because no extraction was attempted.
    """
    if not pageclass.available():
        return path, page_index, "other", [], 0

    for rot in detect.ROTATIONS:
        rotated = detect.rotate(match_img, rot)
        guess = pageclass.rescue_type(rotated)
        if guess is None:
            continue
        items = detect.read_page_items(rotated)
        h, w = rotated.shape
        candidates = detect.find_candidates_for_type(items, h, w, guess)
        if not candidates:
            return path, page_index, guess, [], rot
        crop_rot = detect.rotate(crop_img, rot) if rot else crop_img
        readings = _read_candidates(candidates, crop_rot, rotated.shape)
        return path, page_index, guess, readings, rot

    return path, page_index, "other", [], 0


def route_result(page_type, readings, ext, page_tag, scanned_root, review_dir, discarded_dir, seen_numbers_by_type):
    """The routing decision for one analysed page -- status, destination
    path, filename.

    Returns (status, dest_dir, new_name, discard_kind_or_None).
    status is one of "scanned" | "review" | "discarded".

    Per the client's explicit instruction: review is reserved for pages
    positively classified as one of the five relevant ECR/DC book types
    (see detect.TYPE_FOLDERS) whose number didn't come out confidently
    (field blank, stamp unreadable, zoom variants disagreed); those still
    need a human. Anything not classified as one of those five types at all
    (a known-irrelevant marker match, or genuinely unrecognized prose) is
    discarded outright, never queued -- discard_kind is logged either way
    so there's still an audit trail for a marker that ever turns out to
    match something it shouldn't.
    """
    if page_type.startswith("irrelevant:"):
        return "discarded", discarded_dir, page_tag, page_type.split(":", 1)[1]

    if page_type == "other":
        return "discarded", discarded_dir, page_tag, "unclassified"

    # From here, page_type is one of RELEVANT_TYPES -- a real ECR/DC page.
    # `readings` empty means the labeled field's band produced no candidate
    # at all (the expected outcome for #(Peshawar)/Liquid Nitrogen's ECR#
    # field on samples measured so far -- see PESHAWAR_NOTES.md); that's
    # still a review, not a discard, since the page itself is a genuine
    # book page, just missing its number.
    if readings:
        pool = seen_numbers_by_type[page_type]
        found = [(_PSEUDO_ANCHOR, r) for r in readings]
        number, decisions = resolve_page(found, pool)
        if number:
            type_dir = os.path.join(scanned_root, detect.TYPE_FOLDERS[page_type])
            return "scanned", type_dir, number + ext, None
    return "review", review_dir, page_tag, None


def output_stem(filename):
    """Same trailing-space/dot stripping the archive's main.py applies --
    see its output_stem docstring for the real WinError 3 this prevents."""
    stem, ext = os.path.splitext(filename)
    return stem.strip().rstrip("."), ext


def unique_destination(directory, filename):
    stem, ext = os.path.splitext(filename)
    candidate = os.path.join(directory, filename)
    n = 2
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{stem}_{n}{ext}")
        n += 1
    return candidate


def mark_file_done(input_root, output_root, path):
    """Called once every page of `path` has been accounted for. Writes the
    .done marker (the actual resume check -- see main()/run_batch's docstrings)
    and moves the source file itself into <input_root>/done/.

    Safe to call the moment a file's last page has been consumed from the
    results iterator: multiprocessing.Pool.imap returns results to the
    consumer in strict submission order, so by the time every one of this
    file's pages has been observed here, every worker that read it has
    already finished and closed it (render_page's `with fitz.open(...)`
    releases the handle before returning) -- nothing can still have it open.
    """
    stem, _ = output_stem(os.path.basename(path))
    done_dir = os.path.join(output_root, ".done")
    os.makedirs(done_dir, exist_ok=True)
    open(os.path.join(done_dir, stem + ".done"), "w").close()

    if os.path.exists(path):  # already moved by a prior run's interrupted attempt, most likely
        done_source_dir = os.path.join(input_root, DONE_SOURCE_DIRNAME)
        os.makedirs(done_source_dir, exist_ok=True)
        shutil.move(path, unique_destination(done_source_dir, os.path.basename(path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=INPUT_DIR, help=f"folder of bundle PDFs (default: {INPUT_DIR})")
    parser.add_argument("--output", default=OUTPUT_DIR, help=f"output root (default: {OUTPUT_DIR})")
    parser.add_argument("--limit", type=int, help="process only the first N pages")
    parser.add_argument("--no-copy", action="store_true", help="measure only: write no files")
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
        help="parallel worker processes (default: cores - 2)",
    )
    args = parser.parse_args()

    try:
        verify_gpu()
    except RuntimeError as exc:
        sys.exit(f"Can't start with GPU: {exc}")

    if not os.path.isdir(args.input):
        sys.exit(f"--input {args.input!r} is not a directory")

    files = sorted(
        f for f in glob.glob(os.path.join(args.input, "*"))
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    )
    if not files:
        sys.exit(f"no scan files found in {args.input!r}")

    # Resume tracking is per-file, via a marker file (not the old
    # "does the output folder exist" check -- scanned output is no longer
    # grouped by source file, so there's no longer a source-specific folder
    # whose presence would mean anything). See mark_file_done above. In the
    # normal case this check never even fires -- a genuinely finished file
    # has already been moved to <input>/done/ and so won't appear in
    # `files` at all -- but it's kept as a backstop for a file whose page
    # processing finished right as a run was interrupted, before the move
    # itself completed.
    done_dir = os.path.join(args.output, ".done")
    skipped = 0
    work_items = []
    for f in files:
        stem, _ = output_stem(os.path.basename(f))
        if not args.no_copy and os.path.exists(os.path.join(done_dir, stem + ".done")):
            skipped += 1
            continue
        for i in range(page_count(f)):
            work_items.append((f, i))

    if skipped:
        print(f"resuming: {skipped} already-processed file(s) skipped")
    if not work_items:
        print("nothing to do -- every file is already marked done.")
        return

    if args.limit:
        work_items = work_items[: args.limit]

    print(f"{len(work_items)} page(s) to process across {len(set(f for f, _ in work_items))} file(s)", flush=True)

    if args.workers > 1:
        pool = multiprocessing.Pool(processes=args.workers, initializer=_init_worker)
        results = pool.imap(analyse_page, work_items)
    else:
        pool = None
        results = (analyse_page(w) for w in work_items)

    scanned_root = os.path.join(args.output, SCANNED_DIRNAME)
    review_dir = os.path.join(args.output, REVIEW_DIRNAME)
    discarded_dir = os.path.join(args.output, DISCARDED_DIRNAME)

    seen_numbers_by_type = {t: set() for t in RELEVANT_TYPES}
    counts = collections.Counter()
    files_seen = collections.Counter()
    files_total = collections.Counter(f for f, _ in work_items)
    started = time.time()

    # Written per-page as results.csv/discarded.csv rows are produced (not
    # buffered and written once at the end) so a crash mid-run -- e.g. the
    # Unicode console-print crash this comment replaced -- loses at most the
    # one in-flight row, not the whole run's audit trail; every page already
    # routed to disk before a crash stays traceable.
    results_csv_path = os.path.join(args.output, "results.csv")
    discarded_csv_path = os.path.join(args.output, "discarded.csv")
    results_fh = discarded_fh = results_writer = discarded_writer = None
    if not args.no_copy:
        os.makedirs(args.output, exist_ok=True)
        results_is_new = not os.path.exists(results_csv_path)
        discarded_is_new = not os.path.exists(discarded_csv_path)
        results_fh = open(results_csv_path, "a", newline="", encoding="utf-8")
        discarded_fh = open(discarded_csv_path, "a", newline="", encoding="utf-8")
        results_writer = csv.writer(results_fh)
        discarded_writer = csv.writer(discarded_fh)
        if results_is_new:
            results_writer.writerow(["source_file", "page", "status", "type", "destination", "timestamp"])
        if discarded_is_new:
            discarded_writer.writerow(["source_file", "page", "matched_type", "destination", "timestamp"])

    for i, (path, page_index, page_type, readings, rotation) in enumerate(results, 1):
        name = os.path.basename(path)
        stem, ext = output_stem(name)
        page_tag = f"{stem}_p{page_index + 1:03d}{ext}"

        status, dest_dir, new_name, discard_kind = route_result(
            page_type, readings, ext, page_tag, scanned_root, review_dir, discarded_dir, seen_numbers_by_type,
        )

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        dest = "(not written)"
        if not args.no_copy:
            os.makedirs(dest_dir, exist_ok=True)
            dest = unique_destination(dest_dir, new_name)
            try:
                extract_page(path, page_index, dest, rotation=rotation)
                dest = os.path.relpath(dest, args.output)
            except Exception as exc:
                # A genuinely malformed source PDF (seen in practice: MuPDF
                # "non-page object in page tree" / "source object number out
                # of range" on one corrupt file) must not take down an
                # unattended multi-hour run over a single bad page -- log it
                # as its own status (pipeline_bridge.status() already counts
                # status=="error" rows) and keep going.
                print(f"  !! extraction failed: {name} p{page_index + 1} -- {exc}", flush=True)
                status, dest = "error", f"(extraction failed: {exc})"

        if status == "error":
            if results_writer is not None:
                results_writer.writerow([name, page_index + 1, status, page_type, dest, now])
                results_fh.flush()
        elif status == "discarded":
            if discarded_writer is not None:
                discarded_writer.writerow([name, page_index + 1, discard_kind, dest, now])
                discarded_fh.flush()
        else:
            if results_writer is not None:
                results_writer.writerow([name, page_index + 1, status, page_type, dest, now])
                results_fh.flush()

        counts[status] += 1
        files_seen[path] += 1
        if not args.no_copy and files_seen[path] == files_total[path]:
            mark_file_done(args.input, args.output, path)

        print(f"[{i:>6}/{len(work_items)}] {name} p{page_index + 1:<3} -> {status:<10} "
              f"{discard_kind or page_type:<20} {dest}", flush=True)

    if pool is not None:
        pool.close()
        pool.join()

    if results_fh is not None:
        results_fh.close()
    if discarded_fh is not None:
        discarded_fh.close()

    total = time.time() - started
    print(f"\n{len(work_items)} pages in {total:.0f}s ({total / len(work_items):.2f}s/page, {args.workers} worker(s))")
    print(f"  scanned={counts['scanned']} discarded={counts['discarded']} review={counts['review']}")


if __name__ == "__main__":
    main()
