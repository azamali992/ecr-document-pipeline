"""Thin bridge from the Peshawar beta's GUI into its own main.py pipeline.

Mirrors app/pipeline_bridge.py's shape (multiprocessing.Pool, per-page
progress callback pushed to the UI). The input folder is selectable at
runtime now, not fixed: the warehouse's real data is one subfolder per
month under incoming/peshawer/ (a full fiscal year's worth, not just the
one month this beta started against) -- see list_month_folders/api.py's
select_month_folder. Output stays a single shared peshawar_output/ root
regardless of which month is currently selected, on purpose: scanned/<Type>/
is meant to hold one continuous sequence per book across the whole year,
not per-month buckets, so a gap can be spotted by eye across month
boundaries too. No SQLite ledger -- resume is a per-file marker plus moving
each finished source PDF into <month-folder>/done/ (see main.py's
mark_file_done), no cross-process run lock (single-user testing tool, not
the production archive).
"""
import glob
import multiprocessing
import os
import sys
import time

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_BETA_DIR = os.path.dirname(_APP_DIR)
if _BETA_DIR not in sys.path:
    sys.path.insert(0, _BETA_DIR)

import main as beta_main  # noqa: E402 -- this beta's own pipeline driver, reused not reimplemented
from pipeline.render import SUPPORTED_EXTS, extract_page, page_count  # noqa: E402

DEFAULT_OUTPUT = beta_main.OUTPUT_DIR
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 2)

# The parent of the one month-folder this beta originally started against
# -- e.g. INPUT_DIR "incoming/peshawer/JULY" -> "incoming/peshawer". Every
# sibling of JULY under here (AUGUST, MARCH, ...) is a candidate month to
# select and run, see list_month_folders.
PESHAWAR_ROOT = os.path.dirname(beta_main.INPUT_DIR)
DEFAULT_INPUT = beta_main.INPUT_DIR


def _files(input_root):
    return sorted(
        f for f in glob.glob(os.path.join(input_root, "*"))
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    )


def _done_marker(output_root, stem):
    return os.path.join(output_root, ".done", stem + ".done")


def list_month_folders(base=PESHAWAR_ROOT):
    """[{name, path, pending, done}] -- every subfolder of incoming/peshawer/
    that actually holds (or has finished) bundle PDFs, for the Run tab's
    folder picker. `done` is read straight from <folder>/done/'s contents
    -- the same physical move mark_file_done performs -- so a month that's
    already been fully processed shows as done here without needing to
    open it or run anything."""
    if not os.path.isdir(base):
        return []
    out = []
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if not os.path.isdir(path) or name == beta_main.DONE_SOURCE_DIRNAME:
            continue
        pending = len(_files(path))
        done_dir = os.path.join(path, beta_main.DONE_SOURCE_DIRNAME)
        done = len(_files(done_dir)) if os.path.isdir(done_dir) else 0
        if pending == 0 and done == 0:
            continue  # not actually a bundle folder (e.g. something else nested here)
        out.append({"name": name, "path": path, "pending": pending, "done": done})
    return out


def list_input_files(input_root=DEFAULT_INPUT):
    if not os.path.isdir(input_root):
        return []
    return [{"name": os.path.basename(f), "pages": page_count(f)} for f in _files(input_root)]


def status(output_root=DEFAULT_OUTPUT, input_root=DEFAULT_INPUT):
    """Ground-truth snapshot recomputed from disk on every call -- same
    reasoning as the main archive's client_status: reopening the app
    should never show a finished run as if nothing had been done. Every
    number here comes from what's actually on disk right now, not from
    anything cached in the app process, so this is correct immediately
    after a restart too.
    """
    pending = _files(input_root)
    done_source_dir = os.path.join(input_root, beta_main.DONE_SOURCE_DIRNAME)
    done_sources = _files(done_source_dir) if os.path.isdir(done_source_dir) else []
    # Backstop: a file whose pages all finished right as a run was
    # interrupted, before mark_file_done's move itself completed, still
    # has its .done marker even though it's sitting back in `pending` --
    # count it as done rather than showing it as untouched.
    done = len(done_sources) + sum(
        1 for f in pending
        if os.path.exists(_done_marker(output_root, beta_main.output_stem(os.path.basename(f))[0]))
    )
    total_files = len(pending) + len(done_sources)

    scanned = len(glob.glob(os.path.join(output_root, beta_main.SCANNED_DIRNAME, "*", "*.pdf")))
    review = len(glob.glob(os.path.join(output_root, beta_main.REVIEW_DIRNAME, "*.pdf")))
    discard_csv = os.path.join(output_root, "discarded.csv")
    discarded = 0
    if os.path.exists(discard_csv):
        with open(discard_csv, encoding="utf-8") as fh:
            discarded = max(0, sum(1 for _ in fh) - 1)
    results_csv = os.path.join(output_root, "results.csv")
    errors = 0
    if os.path.exists(results_csv):
        import csv
        with open(results_csv, encoding="utf-8") as fh:
            errors = sum(1 for row in csv.DictReader(fh) if row.get("status") == "error")
    return {
        "total_files": total_files,
        "done_files": done,
        "scanned": scanned,
        "review": review,
        "discarded": discarded,
        "errors": errors,
        "started": done > 0 or scanned > 0 or review > 0 or discarded > 0,
        "fully_done": total_files > 0 and done == total_files,
    }


def _write_csv(path, header, rows):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(header)
        writer.writerows(rows)


def run_batch(input_root=DEFAULT_INPUT, output_root=DEFAULT_OUTPUT, restart=False,
              on_progress=None, workers=DEFAULT_WORKERS, cancel_event=None):
    """Process every not-yet-done file, in parallel. Returns a summary dict.

    on_progress(dict), if given, is called after every page with a JSON-
    serialisable status snapshot -- same contract as the main archive's
    run_client, so the frontend's onProgress handler needed almost no
    changes to work here too.

    cancel_event, if given, is checked between pages -- when set (Pause or
    Stop from the UI, api.py sets the same event either way), the pool is
    terminated immediately (not drained), so GPU/CPU resources actually
    free up rather than finishing whatever's already queued. Whatever's
    written to disk so far stays: a file whose every page was already
    accounted for keeps its .done marker (see main.py's docstring for why
    that's the whole resume mechanism), so calling run_batch again -- Pause
    and Stop both just mean "click Run again to pick up where this left
    off" -- reprocesses only the one file that was genuinely in flight, not
    everything.
    """
    files = _files(input_root)
    work_items = []
    files_total = {}
    for f in files:
        stem, _ = beta_main.output_stem(os.path.basename(f))
        if not restart and os.path.exists(_done_marker(output_root, stem)):
            continue
        n = page_count(f)
        files_total[f] = n
        for i in range(n):
            work_items.append((f, i))

    total = len(work_items)
    counts = {"scanned": 0, "review": 0, "discarded": 0, "error": 0}

    if on_progress:
        on_progress({"phase": "start", "total": total, "workers": workers})
    if total == 0:
        summary = {"phase": "done", "total": 0, "elapsed": 0, "counts": counts}
        if on_progress:
            on_progress(summary)
        return summary

    scanned_root = os.path.join(output_root, beta_main.SCANNED_DIRNAME)
    review_dir = os.path.join(output_root, beta_main.REVIEW_DIRNAME)
    discarded_dir = os.path.join(output_root, beta_main.DISCARDED_DIRNAME)
    seen_numbers_by_type = {t: set() for t in beta_main.RELEVANT_TYPES}
    rows, discard_rows = [], []
    files_seen = {}
    started = time.time()

    pool = None
    try:
        if workers > 1:
            pool = multiprocessing.Pool(processes=workers, initializer=beta_main._init_worker)
            results = pool.imap(beta_main.analyse_page, work_items)
        else:
            results = (beta_main.analyse_page(item) for item in work_items)
        results = iter(results)

        stopped = False
        for i, (path, page_index) in enumerate(work_items, 1):
            if cancel_event is not None and cancel_event.is_set():
                stopped = True
                if pool is not None:
                    pool.terminate()
                    pool.join()
                    pool = None
                break

            name = os.path.basename(path)
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            try:
                _, _, page_type, readings, rotation = next(results)
                stem, ext = beta_main.output_stem(name)
                page_tag = f"{stem}_p{page_index + 1:03d}{ext}"

                page_status, dest_dir, new_name, discard_kind = beta_main.route_result(
                    page_type, readings, ext, page_tag, scanned_root, review_dir, discarded_dir,
                    seen_numbers_by_type,
                )
                os.makedirs(dest_dir, exist_ok=True)
                dest = beta_main.unique_destination(dest_dir, new_name)
                extract_page(path, page_index, dest, rotation=rotation)
                shown_dest = os.path.relpath(dest, output_root)
                if page_status == "discarded":
                    discard_rows.append([name, page_index + 1, discard_kind, shown_dest, now])
                else:
                    rows.append([name, page_index + 1, page_status, page_type, shown_dest, now])
            except StopIteration:
                raise
            except Exception as exc:  # noqa: BLE001 -- one bad page must not sink the batch
                page_status = "error"
                shown_dest = str(exc)
                rows.append([name, page_index + 1, "error", "", shown_dest, now])

            counts[page_status] += 1
            files_seen[path] = files_seen.get(path, 0) + 1
            if files_seen[path] == files_total.get(path):
                beta_main.mark_file_done(input_root, output_root, path)

            if on_progress:
                on_progress({
                    "phase": "page", "index": i, "total": total, "file": name, "page": page_index + 1,
                    "status": page_status, "type": page_type, "destination": shown_dest,
                    "counts": dict(counts), "time": now,
                })

        if pool is not None:
            pool.close()
            pool.join()
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise

    if rows:
        _write_csv(os.path.join(output_root, "results.csv"),
                    ["source_file", "page", "status", "type", "destination", "timestamp"], rows)
    if discard_rows:
        _write_csv(os.path.join(output_root, "discarded.csv"),
                    ["source_file", "page", "matched_type", "destination", "timestamp"], discard_rows)

    summary = {
        "phase": "stopped" if stopped else "done",
        "total": total, "elapsed": time.time() - started, "counts": counts,
    }
    if on_progress:
        on_progress(summary)
    return summary
