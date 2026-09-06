"""The object exposed to the frontend's JS as `pywebview.api`.

Mirrors app/api.py's shape (background thread for the batch run, progress
pushed via window.evaluate_js) but simplified: no per-client picker like
the production app's, though a per-month one now (see select_month_folder)
covers the same real need -- the warehouse's data is a full fiscal year,
one subfolder per month, not the single month this beta started against.
Adds Scanned/Discarded browse-and-fix endpoints and the Gemini auto-fill
pass on top of the original Run/Review pair.
"""
import json
import os
import threading
import traceback

import gemini_assist
import pipeline_bridge
import review_bridge


class Api:
    def __init__(self):
        self._window = None
        self._running = False
        self._cancel_event = threading.Event()
        # No default on purpose -- see select_month_folder's docstring for
        # why picking a month is a deliberate action, not an assumption.
        self._selected_input = None

    def set_window(self, window):
        self._window = window

    # ---- Run screen ----

    def list_month_folders(self):
        return pipeline_bridge.list_month_folders()

    def get_selected_folder(self):
        if self._selected_input is None:
            return None
        return {"path": self._selected_input, "name": os.path.basename(self._selected_input)}

    def select_month_folder(self, path):
        """A human picks which month's bundle to work on -- deliberately
        not defaulted or remembered across a restart, since running against
        the wrong month by an unnoticed leftover selection would be a real,
        silent mistake, not a convenience worth the risk."""
        if not os.path.isdir(path):
            return {"error": f"{path!r} is not a folder"}
        self._selected_input = path
        return {"selected": path}

    def list_input_files(self):
        if self._selected_input is None:
            return []
        return pipeline_bridge.list_input_files(self._selected_input)

    def get_status(self):
        if self._selected_input is None:
            return {
                "total_files": 0, "done_files": 0, "scanned": 0, "review": 0,
                "discarded": 0, "errors": 0, "started": False, "fully_done": False,
            }
        return pipeline_bridge.status(input_root=self._selected_input)

    def default_workers(self):
        return pipeline_bridge.DEFAULT_WORKERS

    def start_run(self, restart=False, workers=None):
        if self._running:
            return {"error": "a run is already in progress in this window"}
        if self._selected_input is None:
            return {"error": "pick a month folder first"}
        if workers is None:
            workers = pipeline_bridge.DEFAULT_WORKERS
        try:
            workers = max(1, int(workers))
        except (TypeError, ValueError):
            workers = pipeline_bridge.DEFAULT_WORKERS
        self._running = True
        self._cancel_event.clear()
        thread = threading.Thread(
            target=self._run_worker, args=(restart, workers, self._selected_input), daemon=True,
        )
        thread.start()
        return {"started": True}

    def stop_run(self):
        """Covers both Pause and Stop -- see pipeline_bridge.run_batch's
        docstring for why they're the same underlying action here: the pool
        is terminated immediately (not drained) and whatever's already
        written stays, resumable by just clicking Run again."""
        if not self._running:
            return {"error": "nothing is running"}
        self._cancel_event.set()
        return {"stopping": True}

    def _run_worker(self, restart, workers, input_root):
        try:
            pipeline_bridge.run_batch(
                input_root=input_root, restart=restart, on_progress=self._push,
                workers=workers, cancel_event=self._cancel_event,
            )
        except Exception as exc:  # noqa: BLE001 -- surface any failure to the UI rather than dying silently
            self._push({"phase": "error", "message": str(exc), "trace": traceback.format_exc()})
        finally:
            self._running = False

    def _push(self, payload):
        if self._window is None:
            return
        self._window.evaluate_js(f"onProgress({json.dumps(payload)})")

    # ---- Review screen ----

    def list_review_items(self):
        return review_bridge.list_review_items()

    def get_page_image(self, pdf_path):
        return review_bridge.render_page_image(pdf_path)

    def rotate_page(self, pdf_path, degrees):
        try:
            review_bridge.rotate_page(pdf_path, degrees)
            return {"image": review_bridge.render_page_image(pdf_path)}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def submit_correction(self, pdf_path, number, type_key):
        try:
            return review_bridge.submit_correction(pdf_path, number, type_key)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def ai_available(self):
        return gemini_assist.available()

    def run_ai_on_review_queue(self):
        """Fire-and-forget -- see review_bridge.run_ai_on_review_queue.
        Returns almost immediately, doesn't wait for any page to finish."""
        review_bridge.run_ai_on_review_queue()

    def ai_review_status(self):
        return review_bridge.ai_review_status()

    # ---- Scanned browse ----

    def list_scanned_items(self, type_key=None):
        return review_bridge.list_scanned_items(type_key)

    def scanned_type_counts(self):
        return review_bridge.scanned_type_counts()

    def rename_scanned_item(self, pdf_path, number, type_key):
        try:
            return review_bridge.rename_scanned_item(pdf_path, number, type_key)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    # ---- Discarded browse ----

    def list_discarded_items(self):
        return review_bridge.list_discarded_items()

    def restore_discarded_item(self, pdf_path, number, type_key):
        try:
            return review_bridge.restore_discarded_item(pdf_path, number, type_key)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    # ---- Duplicates / outliers ----

    def list_anomaly_groups(self):
        """Numbers filed more than once in a book, plus numbers far outside
        the book's range -- both are signatures of the wrong field being
        read. See review_bridge.list_anomaly_groups."""
        return review_bridge.list_anomaly_groups()

    def list_duplicate_items(self, type_key, number):
        return review_bridge.list_duplicate_items(type_key, number)

    def discard_item(self, pdf_path):
        """Send a page from Review or Scanned to Discarded -- see
        review_bridge.discard_item for why both views need this."""
        try:
            return review_bridge.discard_item(pdf_path)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    # ---- Shared ----

    def type_options(self):
        """[{key, label}] for the five relevant book types, for the
        frontend to build its type dropdowns from -- one source of truth
        (pipeline/detect.py's TYPE_FOLDERS) rather than a hardcoded list
        duplicated in JS."""
        return [{"key": k, "label": v} for k, v in review_bridge.TYPE_FOLDERS.items()]
