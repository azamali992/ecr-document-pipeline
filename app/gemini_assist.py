"""AI-assisted read of a review-queue page for the Peshawar beta, using
Gemini's vision API.

Unlike the main archive's gemini_assist.py (which only ever pre-fills a
box a human still confirms), this one is allowed to move a page straight
into scanned/ on a confident read -- the client's explicit instruction for
this beta ("use gemini powered ai to detect numbers from the review
sections and auto put them in scanned"). The safety net for a wrong AI call
here is the Scanned browse view, not a human confirm click -- every
AI-filed page is just as visible and just as renamable there as one the
detector filed directly. That tradeoff (some AI mistakes ship straight to
scanned/, caught later by eye rather than caught before filing) is a
real, deliberate difference from the main archive's tool, not an oversight.

Prompted for all five relevant book types (see pipeline/detect.py's
TYPE_FOLDERS) plus explicit negative examples drawn from real false
positives found calibrating this beta's own detector (the "#(Peshawar)"
field misreading the date, G/P# vs Vehicle# ambiguity, the printed form
codes, phone numbers).

Fails silent, not loud: no API key configured, a network error, a rate
limit, or a malformed response all just mean no suggestion -- the page
stays in review for a human, same as if this module didn't exist.

Self-imposed call cap: at most MAX_CALLS_PER_DAY real API calls in any
rolling 24 hours, tracked in .gemini_usage.json (persists across app
restarts -- a day's worth of calls doesn't reset just because the window
was closed and reopened) and guarded by a lock, since suggestions run from
several background threads at once (see review_bridge.py's ThreadPoolExecutor).
Once the cap is hit, suggest() degrades exactly like any other failure --
returns None, page stays in review -- rather than erroring.
"""
import json
import os
import re
import threading
import time

from dotenv import load_dotenv

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_BETA_DIR = os.path.dirname(_APP_DIR)
_PROJECT_ROOT = os.path.dirname(_BETA_DIR)
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

_API_KEY = os.environ.get("GEMINI_API_KEY")

# Google's own published free-tier ceiling for gemini-flash-lite-latest as
# of this pricing check (see the archive's own gemini_assist.py) -- shown
# in the UI alongside the self-imposed cap below so "why did suggestions
# stop" has a real, checkable number behind it, not just an arbitrary one.
OFFICIAL_FREE_LIMIT_PER_DAY = 1500

# Kept comfortably under that -- 1400 leaves margin for the main archive's
# own review-tab AI usage sharing the same key, and for this being a
# rolling window, not a calendar-day reset (so a burst right at a calendar
# boundary can't accidentally exceed the real 1,500 the way a naive
# midnight-reset counter could).
# Raised from 1400 once billing was enabled on the Google account. The old
# value existed to stay under the FREE tier's 1,500/day ceiling; with billing
# it was no longer protecting anything, just stopping a paid run partway
# through the backlog.
#
# It is raised, not removed. The cap's real job now is to bound the damage
# from a bug -- an unattended loop against a paid API is exactly how a typo
# becomes a surprise invoice. At the measured $0.000169/page (1,564 input +
# 31 output tokens, metered on a real page), this ceiling is worth about
# $1.35 of spend per rolling 24h, which is far above any legitimate day's
# work here (the entire remaining review queue is ~5,300 pages / ~$0.90)
# and far below an amount that could go unnoticed.
MAX_CALLS_PER_DAY = 8000
_WINDOW_SECONDS = 24 * 60 * 60
_USAGE_PATH = os.path.join(_APP_DIR, ".gemini_usage.json")
_usage_lock = threading.Lock()

# "-latest" alias rather than a pinned version -- a pinned model name today
# is not guaranteed to exist next month (measured directly earlier this
# session: gemini-2.0-flash had already been retired by the time it was
# tested against the real API).
_MODEL = "gemini-flash-lite-latest"

_PROMPT = """This is a scanned page from a Pakistani gas-supplier warehouse's daily \
paperwork bundle. It is EITHER one of five relevant document types, OR something \
else entirely (a ledger, receipt, voucher, or handwritten note -- if so, leave both \
fields empty).

The five relevant types, and where their number is:
- Gas Cylinder Job Card (Urdu, titled "جاب کارڈ برائے گیس سلنڈرز") -- number top-right, \
next to "نمبر"
- D/A Gas Job Card (Urdu, titled "جاب کارڈ برائے D/A گیس") -- number top-right, next to "نمبر"
- Liquid Nitrogen Job Card (Urdu, form code MCL-ECRLN-015) -- the number is usually \
under "گیٹ پاس نمبر" (Gate Pass Number), NOT under "ای سی آر نمبر" (ECR Number), which \
is normally blank -- this is expected. If Gate Pass Number is ALSO blank, look instead \
at the right-hand end of the grey horizontal header row near the top of the page: some \
pages carry a stamped 3-4 digit serial there instead, and that is the document's real \
number on those pages
- Liquid Tank Decant Report (English, form code MCL-TDR-010, titled \
"LIQUID TANK DECANT REPORT") -- number top-right
- Delivery Challan (English, form code MCL-CDB-014, titled "DELIVERY CHALLAN") -- has \
TWO number fields, "G/P#" (Gate Pass#) and "DC #(Peshawar)". Only one of the two is \
normally filled in on any given page -- it varies, neither field is reliably the one \
with a number in it. Read whichever of the two actually has a number written in it; if \
both are filled, prefer G/P#

Do NOT return any of these, even if they look plausible:
- a phone number or fragment of one (printed contact numbers at the bottom of the page)
- a plot/address number, or a form code like "MCL-CDB-014" or "MCL-ECRLN-015"
- a vehicle number (a different field, sits near G/P# but is not it)
- a date, even if it happens to look like a short number when misread
- a quantity, count, weight, or size value from a data table

Respond with which of the five types this is (or "none" if it's something else), \
and the number. If you cannot confidently find the document's own number, leave \
"number" empty rather than guessing."""

_TYPE_KEYS = ("gas_cylinder", "da_gas", "liquid_nitrogen", "liquid_tank_decant", "delivery_challan", "none")

_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "type": {"type": "STRING", "enum": list(_TYPE_KEYS)},
        "number": {
            "type": "STRING",
            "description": "digits only, or empty string if not confidently found",
        },
    },
    "required": ["type", "number"],
}

_client = None


def available():
    """False means: no API key configured, so don't even try."""
    return bool(_API_KEY)


def _load_call_times():
    if not os.path.exists(_USAGE_PATH):
        return []
    try:
        with open(_USAGE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []  # corrupt/unreadable usage file -- treat as no history rather than crash


def _prune(call_times, now):
    cutoff = now - _WINDOW_SECONDS
    return [t for t in call_times if t > cutoff]


def _reserve_call_slot():
    """Atomically checks the rolling-24h call count and, if under
    MAX_CALLS_PER_DAY, records a new call and returns True. Returns False
    (caller must not call the API) once the cap is reached. The lock plus
    read-modify-write-in-one-critical-section is what keeps this safe
    across the several background threads suggest() can be called from
    concurrently -- without it, threads could all read the same
    under-the-cap count and all proceed, overshooting it.
    """
    with _usage_lock:
        now = time.time()
        call_times = _prune(_load_call_times(), now)
        if len(call_times) >= MAX_CALLS_PER_DAY:
            return False
        call_times.append(now)
        with open(_USAGE_PATH, "w", encoding="utf-8") as fh:
            json.dump(call_times, fh)
        return True


def usage_status():
    """{"used", "limit", "official_limit"} for the rolling 24h window --
    lets the UI show how much of the self-imposed cap has been used (and
    what Google's own real ceiling is, for context) rather than AI
    suggestions just silently stopping with no explanation once it's hit."""
    with _usage_lock:
        used = len(_prune(_load_call_times(), time.time()))
    return {"used": used, "limit": MAX_CALLS_PER_DAY, "official_limit": OFFICIAL_FREE_LIMIT_PER_DAY}


def _get_client():
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=_API_KEY)
    return _client


def suggest(jpeg_bytes):
    """Best-effort {"type": str, "number": str} for one review-queue page
    image, or None if unavailable, uncertain, or anything went wrong.
    `type` is one of pipeline.detect.TYPE_FOLDERS's keys.

    Never raises -- see this module's docstring for why every failure mode
    here degrades to "no suggestion, leave it in review" rather than an
    error the caller needs to handle specially. That includes the rolling
    24h call cap (MAX_CALLS_PER_DAY) -- reaching it just means suggestions
    stop until the window rolls forward, not an error.
    """
    if not _API_KEY:
        return None
    if not _reserve_call_slot():
        return None
    try:
        from google.genai import types

        client = _get_client()
        resp = client.models.generate_content(
            model=_MODEL,
            contents=[types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"), _PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=_SCHEMA,
            ),
        )
        data = json.loads(resp.text)
        type_key = str(data.get("type", "")).strip()
        number = str(data.get("number", "")).strip()
        if type_key not in _TYPE_KEYS or type_key == "none":
            return None
        if not re.fullmatch(r"\d{1,6}", number):
            return None
        return {"type": type_key, "number": number}
    except Exception:  # noqa: BLE001 -- any failure here degrades to "no suggestion"
        return None
