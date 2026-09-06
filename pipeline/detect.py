"""Peshawar warehouse bundle: page classification + number-field detection.

Written fresh for this bundle's actual document types -- NOT copied from
the archive's detect.py, since the templates, scan source (CamScanner phone
photos, not flatbed), and page size (A4, not A3) all differ. The approach
is the same proven shape as the archive: RapidOCR's own text detector finds
plausible print-sized regions, a measured Y/X band per template plus
digit-count/density/height filters nominate candidates, and
recognize.read_number's zoom-consensus decides what's actually
trustworthy. See PESHAWAR_NOTES.md for the full calibration record this
file's constants come from.

Five relevant document types now (previously two), reference photos of
each blank book in ../../baseecrs/:
  - Gas Cylinder Job Card (Urdu) -- number under "نمبر", RapidOCR can't
    read the Urdu title, so this is identified indirectly (MCL/CP table
    headers), see JOB_CARD_REQUIRED_MARKERS.
  - D/A Gas Job Card (Urdu, "D/A" is printed in Latin script inline with
    the Urdu title) -- marker DA_MARKER. Number band copied from Gas
    Cylinder's (same header layout in the reference photo) but NOT yet
    confirmed against a real scanned sample -- see its own comment.
  - Liquid Nitrogen Job Card (Urdu, form MCL-ECRLN-015) -- the pre-printed
    book number sits under "گیٹ پاس نمبر" (Gate Pass Number), NOT under
    "ای سی آر نمبر" (ECR Number) which the reference photo shows blank --
    confirmed by the client this is normal, same story as the Delivery
    Challan's G/P#. Marker LN_MARKER ("ECRLN", from the form code).
  - Liquid Tank Decant Report (English, form MCL-TDR-010) -- marker
    LIQUID_TANK_MARKER. Band inferred from the reference photo only, not
    yet confirmed against a real scanned sample.
  - Delivery Challan (English, form MCL-CDB-014) -- has TWO number fields,
    G/P# and "DC #(Peshawar)", and which one is actually filled varies by
    period: G/P# on the early JULY samples this was first calibrated
    against, DC #(Peshawar) on the large majority of the rest of the fiscal
    year (measured -- see DC_MARKER). G/P# is tried first, DC# is the
    fallback. See _find_challan_candidates.

Per the client's explicit instruction: only extract from the field
literally labeled ECR/DC, EXCEPT where the client has since confirmed a
specific different field is the real one in practice (G/P# for Delivery
Challan, Gate Pass Number for Liquid Nitrogen). Not extended to any other
field without the same kind of direct confirmation -- note DC #(Peshawar)
is not an exception to this rule but the literal DC field itself, added
once real data showed it, not G/P#, carries the number on most pages.
"""
import re
from dataclasses import dataclass

import cv2

from .recognize import _get_engine

# Number fields have always been seen well above this; matches the
# archive's own TOP_FRACTION margin.
TOP_FRACTION = 0.45

# ECR/DC numbers run 1-6 digits -- the same business rule confirmed by the
# client for the main archive (v3/pipeline/detect.py's CANDIDATE_DIGITS);
# it's a company-wide numbering convention, not specific to one branch.
CANDIDATE_DIGITS = (1, 2, 3, 4, 5, 6)

# Reused starting points from the archive's own detect.py -- the *idea*
# behind each ("the detector's own confidence is a cheap first filter, not
# the real trust decision", "a genuine number reads as almost nothing but
# digits") doesn't depend on which branch's forms it's applied to.
MIN_CANDIDATE_CONFIDENCE = 0.65
MIN_DIGIT_DENSITY = 0.6

# Lowered from the archive's inherited 0.022 to 0.020 after real, direct
# evidence it was too strict for this corpus: "116" (a genuine G/P# value)
# measured 2.31% -- barely above -- and "519" (another genuine G/P# value,
# same field) measured 2.12%, BELOW 0.022, and was silently dropped as a
# candidate on a real run. Two real numbers, one on each side of the old
# threshold, is direct proof it was cutting into genuine values on this
# corpus, not just a close call. No confirmed false positive has been seen
# at 2.0-2.2% here yet to weigh against that -- if one turns up, that's the
# number to re-measure against, not a reason to guess a stricter value now.
MIN_HEIGHT_FRACTION = 0.020
MAX_HEIGHT_FRACTION = 0.15

# The full phrase, not just "CHALLAN" alone -- confirmed a real false
# positive on real data: government/bank treasury deposit slips ("CHALLAN
# FORM NO. 32-A", "CHALLAN OF CASH/TRANSFER/CLEARING PAID INTO...") are a
# completely unrelated, common Pakistani paperwork type that also contains
# the bare word "CHALLAN" prominently. Matching on just "CHALLAN" classified
# these as Delivery Challan pages; since they have no G/P# field at all,
# they then landed in review (a relevant type with no candidate) instead of
# being discarded as the unrelated documents they are. "DELIVERYCHALLAN"
# (OCR merges the space) is specific enough to MCL's own form title that a
# generic bank challan won't also contain it.
CHALLAN_MARKER = "DELIVERYCHALLAN"
DA_MARKER = "D/A"  # printed inline with the Urdu title on the D/A Gas card
LN_MARKER = "ECRLN"  # from the Liquid Nitrogen card's form code, MCL-ECRLN-015
LIQUID_TANK_MARKER = "LIQUIDTANKDECANT"  # from "LIQUID TANK DECANT REPORT"

# Known-irrelevant document types mixed into the bundle -- once positively
# identified by one of these marker strings (OCR'd verbatim from real
# samples, whitespace stripped before matching), the page is discarded
# (logged, not stored) rather than sent to review. Note
# "DAILYSUMMERYREPORTPLATEFORM" keeps the form's own real misspelling,
# "Summery" not "Summary" -- that's what's actually printed on it, and the
# marker has to match the OCR'd text exactly, not the corrected spelling.
IRRELEVANT_MARKERS = {
    "OPENINGBALANCE": "petty_cash_ledger",
    "DAILYSTOCKTAKINGREPORT": "stock_report",
    "REQUISITIONSLIP": "requisition_slip",
    "DAILYSUMMERYREPORTPLATEFORM": "daily_summary",
    "PAYMENTVOUCHER": "payment_voucher",
    "TOLLPLAZA": "toll_plaza_receipt",
    # A government/bank treasury deposit slip ("CHALLAN FORM NO. 32-A") --
    # confirmed on real review-queue pages: it contains the bare word
    # "CHALLAN" prominently, which used to make CHALLAN_MARKER misclassify
    # it as MCL's own Delivery Challan (see that constant's comment). Named
    # here explicitly now instead of just falling through to the generic
    # "other"/unclassified bucket, since it's a real, recurring type.
    "CHALLANOFCASH": "bank_treasury_challan",
}

# Delivery Challan's DC number: G/P# (Gate Pass#), read by anchoring to the
# "G/P#" label's own detected position on THIS page rather than a fixed
# page-wide band -- a fixed band was tried first and risked ambiguity with
# Vehicle#'s value, which sits on the very next table row down and, on at
# least one real sample, nearly the same x-position ("519" for G/P# at
# y0=5.7-7.9%, x0=64.5%; "8232" for Vehicle# at y0=7.6-10.4%, x0=63.9%,
# rows close enough that a loose band could catch both and read them as a
# false conflict). Anchoring to G/P#'s own label y-position per page and
# requiring the value sit to its right on (approximately) the same row is
# what actually separates the two rows reliably, since the row-to-row gap
# (G/P# label to Vehicle# label, measured 2.2-2.4% across samples) is
# consistent even though each row's absolute page position isn't.
GP_MARKER = "G/P#"
# Tightened 0.018 -> 0.014 after the Vehicle# row leaked through in bulk.
#
# The risk was anticipated above but the value was set slightly too loose.
# Measured on real pages: the handwritten Vehicle number sits 0.0146 from
# the G/P# label's row centre, just inside a 0.018 tolerance, so it was
# accepted as the G/P# value. The damage was visible in the filed archive
# as impossible duplicates -- "2232" filed 38 times, "7940" 17, "8116" 15,
# all of them vehicle numbers rather than document numbers.
#
# Swept against pages filed under a known vehicle number plus clean truth
# pages:
#     tolerance   still vehicle   clean kept   clean CHANGED
#     0.018             3            24             0
#     0.014             0            21             0
#     0.012             0            17             0
#
# 0.014 removes the leak entirely. It costs 3 borderline pages to review,
# and note "clean changed" is 0 throughout -- tightening never swaps one
# answer for another wrong one, it only defers to a human. That is the
# right direction for this pipeline.
GP_ROW_Y_TOLERANCE = 0.014
GP_VALUE_MIN_X_FRACTION = 0.55  # must sit to the label's right -- label's own x0 measured ~50-51%

# Delivery Challan's OTHER number field, "DC #(Peshawar)", top-left.
#
# The original calibration concluded #(Peshawar) is "essentially never
# filled" and G/P# is the real number. That was true of the early JULY
# samples it was measured on, and is FALSE for the bulk of the fiscal year:
# measured over a 150-page random sample of the review queue, 82 were
# Delivery Challans, and 80 of those had no G/P# candidate at all -- G/P#
# is simply left blank on them, while DC #(Peshawar) carries a filled,
# legible handwritten number (visually confirmed on real pages: 5315, 3501,
# 4339, with G/P# an empty ruled line in every case). Both fields are
# genuinely in use depending on the period, so neither one alone is "the"
# DC number.
#
# G/P# is still tried FIRST and this is only a fallback when it yields
# nothing: that keeps every page the existing, client-confirmed rule
# already handled behaving exactly as before (no regression on anything
# already filed), and only adds recovery where the pipeline currently fails
# outright. Measured on that sample: 51% of the no-G/P# challans go
# straight to auto-filed, 1 conflict, 3 read-unsure -- and the recovered
# numbers rise monotonically with each file's own date (Dec-25 ~2680,
# Jan-26 ~2830-3300, Apr-26 ~4200-4460, Jun-26 ~5140-5600), which is the
# sequence behaviour a real pre-printed book number has and misreads do not.
#
# The row directly below DC# is DATE ("14 - 06 - 26"), whose 2-digit
# fragments are themselves valid candidate digits -- hence a TIGHTER row
# tolerance than G/P#'s, and a max-x that keeps this search inside the left
# column so it can never reach across and pick up G/P#'s own value.
DC_MARKER = "DC#"
# Widened 0.014 -> 0.024 on measurement. The DC number is handwritten ABOVE
# the printed label's baseline rather than centred on it, so the offset is
# real and one-sided: measured over 30 stuck challans, |delta_y| runs to
# 0.0184 at p95 while the old tolerance was 0.014, leaving only 17/30 real
# values reachable.
#
# The risk here is the DATE row immediately below -- a tolerance that reaches
# it would file "15-1-026" as a document number -- so this was swept rather
# than guessed, scored against the 120 filed challans whose filename gives
# the true answer plus 90 stuck ones:
#
#     tolerance  correct  WRONG  recovered
#     0.014          114      0          6
#     0.018          114      0         20
#     0.020          114      0         21
#     0.024          114      0         26
#     0.030          114      0         26
#
# Zero wrong values at every setting, and recovery plateaus at 0.024, so
# that is the pick: all of the available gain, none of the extra reach
# toward the date row that 0.030 would buy for nothing.
DC_ROW_Y_TOLERANCE = 0.024
DC_VALUE_MAX_X_FRACTION = 0.50

# Gas Cylinder Job Card's "نمبر" field -- RapidOCR cannot read the
# surrounding Urdu, so position comes from real samples. "035" measured at
# y0=6.2-9.1% (x0=71.7%). A second real sample, "059", sits higher, at
# y0=3.7-7.8% -- outside this band as written. Widening to include it was
# tried and reverted: RapidOCR's own detector box for that specific crop
# clips the leading digit, so every zoom-variant reread agreed -- full
# 3/3 confidence -- on "09", not "059", and it auto-filed as the WRONG
# number (confirmed directly on real output: 09.pdf, should have been
# 059.pdf). That's a mis-bounded-crop problem the zoom-consensus gate can't
# catch (all variants share the same bad crop), not a position problem, so
# widening the band doesn't safely fix it -- it only trades a correct
# "sent to review" for a wrong "silently auto-filed". Left narrow on
# purpose: missing "059" (review) is recoverable, wrongly filing "09"
# is not.
# v2: lower bound dropped 0.045 -> 0.0 on direct evidence from real failing
# pages. Instrumenting the stuck Gas Cylinder pages in review showed the
# numbers were being DETECTED cleanly by OCR and then thrown away purely for
# sitting above the band: "1888" at y-centre 0.018 (conf 0.92) and "1725" at
# y-centre 0.058 with a top edge of ~0.040, i.e. missing the old 0.045 floor
# by a hair. CamScanner crops with less margin above the title shift every
# row up, so a floor measured on generously-cropped pages cuts into tightly
# cropped ones.
#
# An earlier attempt to widen this band was reverted because it produced a
# wrong auto-file ("059" read as "09" at full 3/3 agreement, from a crop that
# clipped the leading digit). That risk is real and is the reason the change
# is verified against ground truth rather than assumed: every page in
# scanned/Gas Cylinder is named for its correct number, so widening can be
# checked for silently changing an already-correct answer, not just for
# finding more. See PESHAWAR_NOTES.md for that measurement.
JOB_CARD_NUMBER_Y_FRACTION = (0.0, 0.11)
JOB_CARD_NUMBER_MIN_X_FRACTION = 0.55


# D/A Gas Job Card's number band -- NOT yet confirmed against a real
# scanned sample (none turned up in the calibration pass). Copied from Gas
# Cylinder's band on the strength of the reference photo showing the same
# "نمبر" position in the same header row layout -- treat as a working
# assumption, revisit the moment a real D/A sample is seen (check the
# Discarded/Review views specifically for "D/A" pages early on).
DA_NUMBER_Y_FRACTION = JOB_CARD_NUMBER_Y_FRACTION
DA_NUMBER_MIN_X_FRACTION = JOB_CARD_NUMBER_MIN_X_FRACTION

# Liquid Nitrogen Job Card's Gate Pass Number -- same "label not readable
# (Urdu), anchor by position" situation as Gas Cylinder, but with only the
# reference photo to go on for the header-page layout (real samples seen
# during calibration were the rotated table-continuation page, not the
# header). Rough band from the reference photo's proportions; unverified.
# A real table-continuation page (no header, no LN_MARKER visible) will
# currently NOT be positively identified as Liquid Nitrogen at all -- it
# has no CP column either, so it also won't fall into Gas Cylinder's
# MCL+CP fallback, and ends up "other" -> discarded. Known gap: check the
# Discarded view for Liquid Nitrogen pages, restore by hand for now.
LN_NUMBER_Y_FRACTION = (0.10, 0.20)
LN_NUMBER_MIN_X_FRACTION = 0.10
LN_NUMBER_MAX_X_FRACTION = 0.35

# NOT IN USE -- kept as a measured finding, see the warning at the end.
# Liquid Nitrogen's SECOND number field -- the pre-printed serial at the
# right-hand end of the "تفصیل مکمل جار" section-header row.
#
# Exactly the same dual-field situation as the Delivery Challan's
# G/P# vs DC#, and found the same way (instrumenting failing pages rather
# than successful ones). On pages where "گیٹ پاس نمبر" is filled, the band
# above reads it and that is the right answer -- visually confirmed on real
# filed pages (09, 10, 117, 119 all sit immediately beside that label).
# But on a large share of pages Gate Pass is left BLANK and the only number
# present is this right-hand serial, and the old configuration had no way
# to see it: measured against the 30 filed pages whose filename gives the
# true answer, the Gate Pass band alone located just 5 of them (17%).
#
# Measured over 40 stuck review pages (values 1201, 1214, 1266, 1303, 1320,
# 1351, 1360 ... a clean sequential series, which is what a pre-printed
# book number looks like and misreads do not):
#     x0  p5 0.714  median 0.792  p95 0.823
#     y0  p5 0.095  median 0.114  p95 0.166
#     height fraction 0.023 - 0.032
# Bounds set a little outside that range, not at it, since these are the
# pages that failed -- the successful ones may sit slightly differently.
LN_SERIAL_Y_FRACTION = (0.06, 0.21)
LN_SERIAL_MIN_X_FRACTION = 0.65
LN_SERIAL_MAX_X_FRACTION = 0.92

# The guard that makes the serial fallback safe, and the reason an earlier
# attempt at it was reverted.
#
# The fallback must only fire when Gate Pass is GENUINELY BLANK -- not when
# it holds a value the band merely failed to confirm. Without that
# distinction it quietly swaps one real number for another: measured, it
# filed truth "05" as "92058", "10" as "10368" and "114" as "1480", each
# time reading the serial off a page whose Gate Pass was filled all along.
#
# So before falling back, sweep a deliberately GENEROUS region over the
# whole left-hand field column. If any digit-bearing text sits there at all,
# the field is considered filled-but-unread and the page goes to review
# rather than being filed from a different field. Only a column with no
# digits anywhere is treated as blank.
LN_GATE_COLUMN_Y_FRACTION = (0.02, 0.30)
LN_GATE_COLUMN_MAX_X_FRACTION = 0.45

# The serial's digit count, which is what actually separates it from noise.
#
# The number is stamped by hand so its POSITION varies, which is why no
# band alone works here -- but its SHAPE does not. Every confirmed serial
# measured on real pages runs 3-4 digits (714, 903, 1155, 1807, 2223, 2251,
# 2420, 1480). The values the serial box wrongly picked up without this
# filter were 92058, 10368 (five digits -- weights or amounts from the
# table) and "3", "9" (single digits -- row numbers from the tally column).
# Constraining shape removes all four while keeping every real serial.
LN_SERIAL_DIGITS = (3, 4)

# Liquid Tank Decant Report's number -- NOT yet confirmed against a real
# scanned sample. Estimated from the reference photo's own proportions
# (top-right, roughly level with the report title). Wide margins on
# purpose since this is a guess, not a measurement.
LIQUID_TANK_NUMBER_Y_FRACTION = (0.08, 0.22)
LIQUID_TANK_NUMBER_MIN_X_FRACTION = 0.65

# "job_card" (Gas Cylinder specifically) used to be scan()'s unconditional
# fallback: anything that didn't match a challan/irrelevant marker and
# didn't read as English prose was assumed to be a job card, on the theory
# that RapidOCR can't read Urdu either way. Confirmed wrong in real use:
# freeform handwritten notes on plain ruled paper (no printed template at
# all, e.g. "BOWSERI FILLING.pdf") produce the exact same near-empty OCR
# result a genuine job card does -- sparse fragments, a few digits, no real
# English -- so they fell into the same fallback and landed in review
# looking nothing like an ECR/DC page. "MCL" and "CP" are the
# cylinder-tracking table's own printed column headers, present on every
# genuine Gas Cylinder job card sample checked and absent from every
# freeform-notes sample checked -- requiring both turns the fallback into a
# real (if minimal) positive signal instead of "we couldn't prove it's
# anything else".
JOB_CARD_REQUIRED_MARKERS = ("MCL", "CP")

# Output folder name per relevant type -- grouping scanned files by type
# rather than by source PDF is the whole point of this pass (lets a human
# check one book's sequence for gaps without wading through every other
# book mixed in the same day's bundle).
TYPE_FOLDERS = {
    "gas_cylinder": "Gas Cylinder",
    "da_gas": "D-A Gas",
    "liquid_nitrogen": "Liquid Nitrogen",
    "liquid_tank_decant": "Liquid Tank Decant",
    "delivery_challan": "Delivery Challan",
}

_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_ENGLISH_PHRASE_WORDS = 2

_CV2_ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
# Tried in likelihood order: CamScanner generally corrects orientation
# already (confirmed: most page types here resolve at 0 on the first try),
# but Liquid Nitrogen's table-continuation pages were measured coming
# through sideways (90) in real bundles -- see LN_NUMBER_Y_FRACTION's
# comment. 180/270 kept as a last resort for a genuinely mis-fed page.
ROTATIONS = (0, 90, 270, 180)


def rotate(img, degrees):
    if degrees == 0:
        return img
    return cv2.rotate(img, _CV2_ROTATE[degrees])


@dataclass(frozen=True)
class Candidate:
    x0: int
    y0: int
    x1: int
    y1: int
    digits: str


def _normalize(text):
    return re.sub(r"\s+", "", text).upper()


def _candidate_digits(text, conf):
    """The candidate's digit string if `text` clears the common filters
    (confidence, digit count, digit density), else None."""
    if conf < MIN_CANDIDATE_CONFIDENCE:
        return None
    digits = re.sub(r"\D", "", text)
    if len(digits) not in CANDIDATE_DIGITS:
        return None
    stripped = text.strip()
    if not stripped or len(digits) / len(stripped) < MIN_DIGIT_DENSITY:
        return None
    return digits


def _find_candidates(items, h, w, y_fraction, x_min=0.0, x_max=1.0):
    y_lo, y_hi = y_fraction
    candidates = []
    for box, text, conf in items:
        digits = _candidate_digits(text, conf)
        if digits is None:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        if not (x_min * w <= x0 <= x_max * w):
            continue
        if not (y_lo * h <= y0 <= y_hi * h):
            continue
        height = y1 - y0
        if not (MIN_HEIGHT_FRACTION * h <= height <= MAX_HEIGHT_FRACTION * h):
            continue
        candidates.append(Candidate(int(x0), int(y0), int(x1), int(y1), digits))
    candidates.sort(key=lambda c: -(c.y1 - c.y0))
    return candidates


def _find_label_anchored_candidates(items, h, w, marker, y_tolerance,
                                    x_min=0.0, x_max=1.0, right_of_label=False):
    """Candidates on the same table row as `marker`'s own label, on THIS page.

    Anchoring to the label's detected y-position rather than a fixed
    page-wide band is what separates one form row from its neighbours, whose
    values are themselves valid-looking digits (Vehicle# sits one row under
    G/P#; DATE sits one row under DC#) -- see GP_ROW_Y_TOLERANCE and
    DC_ROW_Y_TOLERANCE for the measurements behind each row's tolerance.

    `right_of_label` additionally requires the value start past the label's
    own right edge, for a left-column field whose x_max alone wouldn't rule
    out the label text itself.
    """
    labels = []
    for box, text, _conf in items:
        if marker in _normalize(text):
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            labels.append(((min(ys) + max(ys)) / 2, max(xs)))
    if not labels:
        return []

    candidates = []
    for box, text, conf in items:
        digits = _candidate_digits(text, conf)
        if digits is None:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        if not (x_min * w <= x0 <= x_max * w):
            continue
        y_center = (y0 + y1) / 2
        on_row = [lx1 for ly, lx1 in labels if abs(y_center - ly) <= y_tolerance * h]
        if not on_row:
            continue
        if right_of_label and x0 < min(on_row):
            continue
        height = y1 - y0
        if not (MIN_HEIGHT_FRACTION * h <= height <= MAX_HEIGHT_FRACTION * h):
            continue
        candidates.append(Candidate(int(x0), int(y0), int(x1), int(y1), digits))
    candidates.sort(key=lambda c: -(c.y1 - c.y0))
    return candidates


def _gate_pass_column_has_digits(items, h, w):
    """Is there ANY digit-bearing text in the left-hand field column?

    Used only to decide whether Gate Pass is genuinely blank -- see
    LN_GATE_COLUMN_Y_FRACTION. Deliberately looser than a candidate search:
    it does not care whether the text is a plausible number, only whether
    the field appears to have been written in at all.
    """
    y_lo, y_hi = LN_GATE_COLUMN_Y_FRACTION
    for box, text, _conf in items:
        if not re.sub(r"\D", "", text):
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        if min(xs) > LN_GATE_COLUMN_MAX_X_FRACTION * w:
            continue
        if y_lo * h <= min(ys) <= y_hi * h:
            return True
    return False


def _find_ln_candidates(items, h, w):
    """Liquid Nitrogen's number: Gate Pass when filled, else the right-hand
    pre-printed serial.

    The same dual-field situation as the Delivery Challan's G/P# vs DC#, and
    found the same way -- by instrumenting failing pages. On a large share of
    LN pages the Gate Pass line is simply empty and the only number present
    is a serial at the right-hand end of the section-header row (2251, 1807,
    903, 2223 on the pages this was measured from).

    The fallback is gated on the column genuinely being blank, which is what
    separates this from the earlier reverted attempt -- see
    LN_GATE_COLUMN_Y_FRACTION for the wrong numbers that produced.
    """
    gate_pass = _find_candidates(
        items, h, w, LN_NUMBER_Y_FRACTION,
        x_min=LN_NUMBER_MIN_X_FRACTION, x_max=LN_NUMBER_MAX_X_FRACTION,
    )
    serial = [
        c for c in _find_candidates(
            items, h, w, LN_SERIAL_Y_FRACTION,
            x_min=LN_SERIAL_MIN_X_FRACTION, x_max=LN_SERIAL_MAX_X_FRACTION,
        )
        if len(c.digits) in LN_SERIAL_DIGITS
    ]
    # Two boxes, either is acceptable -- the number is stamped by hand and
    # lands in one of these two places. Nominating BOTH rather than falling
    # back means the existing gate arbitrates: if each box yields a
    # different fully-agreed number, resolve_page sees a conflict and sends
    # the page to review instead of silently preferring one field.
    seen, out = set(), []
    for c in list(gate_pass) + list(serial):
        key = (c.x0, c.y0, c.x1, c.y1)
        if key not in seen:
            seen.add(key)
            out.append(c)
    out.sort(key=lambda c: -(c.y1 - c.y0))
    return out


def _find_challan_candidates(items, h, w):
    """Delivery Challan's number: G/P# if it's filled, else DC #(Peshawar).

    Both fields are genuinely in use across the fiscal year and either can be
    the blank one -- see DC_MARKER's comment for the measurement. G/P# is
    tried first so every page the original client-confirmed rule already
    handled keeps behaving identically; DC# only picks up pages that would
    otherwise have gone to review with nothing found at all.
    """
    gp = _find_label_anchored_candidates(
        items, h, w, GP_MARKER, GP_ROW_Y_TOLERANCE, x_min=GP_VALUE_MIN_X_FRACTION,
    )
    if gp:
        return gp
    return _find_label_anchored_candidates(
        items, h, w, DC_MARKER, DC_ROW_Y_TOLERANCE,
        x_max=DC_VALUE_MAX_X_FRACTION, right_of_label=True,
    )


def find_candidates_for_type(items, h, w, page_type):
    """Candidates for a page whose type is already known.

    scan() finds the type and its candidates together, which is right when
    the marker text is what identified the page. But the visual classifier
    can identify a page whose marker was never readable -- and then there is
    no way to ask "given it IS a Liquid Nitrogen card, where is its number?"
    without this.

    Without it, a rescued page reached review with no extraction attempted at
    all: the type was known, the number was often perfectly legible, and
    nothing ever looked for it.
    """
    if page_type == "delivery_challan":
        return _find_challan_candidates(items, h, w)
    if page_type == "liquid_nitrogen":
        return _find_ln_candidates(items, h, w)
    if page_type == "liquid_tank_decant":
        return _find_candidates(
            items, h, w, LIQUID_TANK_NUMBER_Y_FRACTION,
            x_min=LIQUID_TANK_NUMBER_MIN_X_FRACTION,
        )
    if page_type == "da_gas":
        return _find_candidates(
            items, h, w, DA_NUMBER_Y_FRACTION, x_min=DA_NUMBER_MIN_X_FRACTION,
        )
    if page_type == "gas_cylinder":
        return _find_candidates(
            items, h, w, JOB_CARD_NUMBER_Y_FRACTION, x_min=JOB_CARD_NUMBER_MIN_X_FRACTION,
        )
    return []


def read_page_items(page_img):
    """The OCR items scan() works from, for a caller that already knows the
    page type and only needs its candidates."""
    h, _w = page_img.shape
    top = page_img[: int(h * TOP_FRACTION), :]
    result, _ = _get_engine()(cv2.cvtColor(top, cv2.COLOR_GRAY2RGB))
    return result or []


def scan(page_img):
    """Classify one page (at its current rotation -- see rotate_scan below
    for the rotation search) and find its ECR/DC candidates, if any.

    Returns (page_type, candidates):
      "gas_cylinder" / "da_gas" / "liquid_nitrogen" / "liquid_tank_decant"
                             / "delivery_challan"
                             one of the five relevant document types (see
                             TYPE_FOLDERS) -- candidates come from that
                             type's own band/anchor function.
      "irrelevant:<kind>"    a known-irrelevant document, positively
                             identified by marker text (see
                             IRRELEVANT_MARKERS) -- candidates always []
                             here; the caller discards rather than reviews.
      "other"                everything else: real English prose that
                             matched no known marker, OR a page with
                             neither a relevant nor an irrelevant marker
                             AND missing one of JOB_CARD_REQUIRED_MARKERS.
                             Candidates always [] here; the caller discards
                             this, not reviews it -- see main.py's
                             route_result for why review is reserved for
                             pages positively classified as one of the five
                             relevant types specifically.
    """
    h, w = page_img.shape
    top = page_img[: int(h * TOP_FRACTION), :]
    result, _ = _get_engine()(cv2.cvtColor(top, cv2.COLOR_GRAY2RGB))
    items = result or []
    normalized = [_normalize(text) for _, text, _ in items]

    if any(CHALLAN_MARKER in t for t in normalized):
        return "delivery_challan", _find_challan_candidates(items, h, w)

    if any(LN_MARKER in t for t in normalized):
        return "liquid_nitrogen", _find_ln_candidates(items, h, w)

    if any(LIQUID_TANK_MARKER in t for t in normalized):
        return "liquid_tank_decant", _find_candidates(
            items, h, w, LIQUID_TANK_NUMBER_Y_FRACTION, x_min=LIQUID_TANK_NUMBER_MIN_X_FRACTION
        )

    if any(DA_MARKER in t for t in normalized):
        return "da_gas", _find_candidates(
            items, h, w, DA_NUMBER_Y_FRACTION, x_min=DA_NUMBER_MIN_X_FRACTION
        )

    for marker, kind in IRRELEVANT_MARKERS.items():
        if any(marker in t for t in normalized):
            return f"irrelevant:{kind}", []

    if any(len(_ENGLISH_WORD_RE.findall(text)) >= _ENGLISH_PHRASE_WORDS for _, text, _ in items):
        return "other", []

    if not all(any(marker in t for t in normalized) for marker in JOB_CARD_REQUIRED_MARKERS):
        return "other", []

    return "gas_cylinder", _find_candidates(
        items, h, w, JOB_CARD_NUMBER_Y_FRACTION, x_min=JOB_CARD_NUMBER_MIN_X_FRACTION
    )


# --- Why the Liquid Nitrogen serial fallback (LN_SERIAL_*) is NOT wired in ---
#
# It was implemented and measured, and it FAILED on precision, so it was
# pulled back out. Recorded here so it is not re-attempted blind.
#
# The idea mirrored _find_challan_candidates: use Gate Pass when present,
# fall back to the right-hand serial when it is blank. It does raise
# coverage (13 more Liquid Nitrogen pages resolved out of 250 stuck ones).
# But measured against the filed pages whose filename gives the true answer,
# it auto-filed 13 WRONG values out of 22 -- truth "05" filed as "92058",
# "10" as "10368", "114" as "1480".
#
# The reason is the failure mode, not the band: the fallback cannot tell
# "Gate Pass is blank" from "Gate Pass is filled but my band missed it".
# In the second case it reads a genuinely different field and files it with
# full confidence. That turns a safe review into a silent wrong number,
# which is the one trade this pipeline must never make (see gate.py).
#
# The right fix is to make the GATE PASS band itself find its value
# reliably -- it currently locates only 5 of 30 known-answer pages -- and
# only then consider a fallback for pages where the field is genuinely
# empty. Widening the fallback or tightening its digit count does not help:
# two of the wrong values ("2420", "1480") are themselves plausible 4-digit
# serials, so no shape rule separates them from the right answer.
