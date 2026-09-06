# Peshawar bundle: calibration record

What was actually measured against real pages from `incoming/peshawer/JULY/`
to build `pipeline/detect.py`'s constants. Kept separate from code comments
so the full record (including things that turned out not to matter) stays
in one place; the constants' own comments summarize the parts that fed
directly into a number.

## Document types found

**Relevant:**
- Gas Cylinder Job Card (Urdu) — number field labeled "نمبر". RapidOCR
  can't read the Urdu, so this page type is a fallback classification, not
  a positive marker match — same situation as the main archive's job cards.
- Liquid Nitrogen Job Card (Urdu, form MCL-ECRLN-015) — has both "ای آر
  نمبر" (ECR#) and "گیٹ پاس نمبر" (Gate Pass#) fields. ECR# measured blank
  on both samples seen. Shares the same "job_card" fallback path as the Gas
  Cylinder card (RapidOCR can't tell them apart either), which is fine
  since it correctly finds no candidate either way.
- Delivery Challan (English, form MCL-CDB-014) — marker "DELIVERYCHALLAN"
  (OCR merges the space). Has both "#(Peshawar)" and "G/P#" fields.
  **#(Peshawar) was measured blank on all 5 real samples checked; G/P# was
  filled on all 5.** Per the client's explicit instruction, only the
  literally-labeled field counts — G/P# is not read as a substitute. If
  this holds at volume, delivery challan pages will essentially never
  auto-file; every one seen in the 39-page validation run went to review.
  This is a real, load-bearing consequence worth revisiting with the
  client, not a bug in the code.

**Known-irrelevant (discarded, not reviewed), with their real OCR'd marker
text:**
- Petty Cash / Daily Physical Cash ledger → "OPENINGBALANCE"
- "DAILY STOCK TAKING REPORT" → "DAILYSTOCKTAKINGREPORT"
- "REQUISITION SLIP" (form MCL-SRB-08) → "REQUISITIONSLIP"
- "Daily Summery Report Plate Form" → "DAILYSUMMERYREPORTPLATEFORM" (note:
  "Summery" is the form's own real misspelling, kept verbatim since the
  marker has to match the OCR'd text, not the corrected spelling). This
  page itself lists several real ECR numbers in a column labeled "ECR" as
  tally data — a log, not a source document, so it's excluded on purpose
  even though it contains genuine-looking ECR numbers (same principle as
  the main archive's "log/dispatch number is out of scope" rule).

**Unrecognized, not yet catalogued (fall through to review, which is
correct — never silently discarded on a guess):**
- Vehicle parts/service receipts (seen once, freeform, a mechanic's
  letterhead with an odometer reading and "oil filter change" — not a
  known irrelevant type, correctly went to review without extracting a
  wrong number from any of its several digit clusters)
- Freeform handwritten tally/measurement notes on plain ruled paper, no
  printed title at all (seen more than once — weight/volume calculations,
  a "Loss=" figure). No marker text exists to identify these; they
  correctly fall through to job_card's fallback path, find nothing in the
  measured number-field band, and go to review.

## Number-field measurements

**Gas Cylinder Job Card "نمبر":** one clean, confident sample —
`035` at y0=6.2-9.1% of page height, x0=71.7% (right side). Margined to
`(0.045, 0.11)` / x-min `0.55`. Only one confident direct measurement;
the 39-page validation run's clean sequential match (035→036→037→038→
039→040 across pages 6,8,10,12,16,19 — a warehouse issuing job cards in
order across one day) is strong independent corroboration this band is
finding the right field, not noise.

**Delivery Challan "#(Peshawar)":** never seen filled, so this band is
inferred from the *label's* own position (y0 4.8-9.3% across 5 samples),
not a real value. `x_max=0.45` is load-bearing, not cosmetic: G/P#'s value
(a different, unlabeled field) sits at x0=64-72% on the same row, and
without that cutoff a Y-band-only filter would silently extract it instead.

**A real false positive, found and fixed during validation:** the first
version of this band ran to y=12%. The Delivery Challan's date field
("12-07-26" etc, y0=9.6-11.3%) sits just below the label and, once
non-digits are stripped, clears the digit-count/density filters as easily
as a real number would. On one real page (`CamScanner 05-07-2026 06.41.pdf`
p20), the date got misread into a stable 3-digit string ("726") that
reached full zoom-variant agreement and auto-filed as `726_dc.pdf` — a
page where #(Peshawar) had nothing written in it at all. Fixed by
tightening the upper bound to 0.095, the measured gap between where the
label ends (<=9.3%) and the date starts (>=9.6%) on every sample checked.
Re-ran the same 39-page file after the fix: that page now correctly goes
to review, and the six legitimate 035-040 matches were unaffected.

## Explicitly out of scope for this beta

- No rotation search (the main archive tries 0/180/90/270 per page).
  CamScanner's own app generally corrects orientation before export, so
  this hasn't been needed on samples checked so far — revisit if a
  genuinely sideways/upside-down page turns up.
- No CRNN cross-check (the archive's fine-tuned digit model). It was
  trained on the archive's own mechanical numbering-stamp font; Peshawar's
  numbers are handwritten ballpoint, a different-enough distribution that
  it would likely just decline to vote most of the time. Kept the model
  file in `models/` in case this beta grows into something where retraining
  or reusing it is worthwhile, but it isn't wired into detect.py.
- No SQLite ledger / no repeats.csv (see main.py's docstring) — resume is
  just "skip a source file whose output folder already exists".
- Multiple stacked forms on one scanned page (seen at least once in the
  main archive's client bundle, not yet specifically confirmed in Peshawar
  samples) aren't split — `gate.resolve_page` would route a page like that
  to review as a false conflict if it ever occurs here, which is the safe
  default, not a crash.

---

## v2: visual page classifier (pipeline/pageclass.py)

### The problem it was built for

A hand audit of v1's discard pile, 48 pages sampled at random from the
7,047 rows logged as `unclassified` (81.7% of all discards), found **10
genuine ECR/DC pages in it — ~21%**. Projected across the fiscal year that
is roughly **1,470 real documents silently lost**: more than the entire
review-recovery pass reclaimed. Breakdown of the 10: 7 Liquid Nitrogen,
2 Delivery Challan, 1 Gas Cylinder.

Liquid Nitrogen dominates for a concrete reason — its marker is the tiny
`MCL-ECRLN-015` form code printed at the very top edge, which is routinely
cropped, faint or misread. The two lost Delivery Challans both clearly show
"DELIVERY CHALLAN" to the eye, but `CHALLAN_MARKER` needs OCR to emit the
exact merged string `DELIVERYCHALLAN`; one character off and the page is
discarded unseen.

The 79% that were correctly discarded are genuinely irrelevant — KAMAL
SCALE weighbridge slips, fuel and parking receipts, PTCL invoices, salary
tables. The discard *policy* is right; its *classification input* was too
brittle.

### Training data

Labels came free from where the pipeline had already put each page:
`scanned/<Type>/` and the `type` column of `results.csv`'s review rows
(a review page's type marker DID match — only its number was unreadable),
plus marker-identified discards as negatives. 11,453 pages.

This is weak supervision: every label traces back to the marker classifier,
so the model can inherit its blind spots. It still beats its teacher
because it uses a **different feature basis** — whole-page layout instead of
one small text string — and a page whose form code is cropped off still
*looks* exactly like an LN card.

### Why held-out accuracy was not trusted

A random page-level split leaks badly here: consecutive pages of a bundle
share a scanning session, phone, lighting and skew. Splits are therefore by
**source bundle**. That gives 98.0%, but it only measures how well we
imitate the marker system.

The real test set is 48 **hand-labelled pages from the discard pile** —
every one a page the marker system got wrong or gave up on, and none of
them in training. Round 1 scored 10/10 genuine rescued but 18/38 false
alarms, because its only negatives were the 1,574 marker-identified
discards: it had never seen the junk that actually dominates the pile.
Round 2 added self-training negatives (unclassified discards the round-1
model already called irrelevant with conf >= 0.9) plus one hand-labelled
sheet of the form-like junk that caused the false alarms. That halved them.

### End-to-end result (real analyse_page + route_result, v1 vs v2)

On those same 48 pages, all of which **v1 discarded**:

    genuine rescued and correctly typed   10/10
    junk leaked into review                2/38

Rescued pages go to **review**, not straight to scanned: the classifier
recovers the page, it does not invent a number. The number still has to
clear the normal zoom-consensus gate, unchanged.

### Known limits

- **D-A Gas has only 27 training examples** and is not reliably learnable;
  it leans on the marker rule. Treat any da_gas prediction with suspicion.
- The ~21% genuine rate and the projection built on it come from 48 hand
  labels — a 95% CI of roughly 10–35%, so "~1,470 recoverable" is the right
  order of magnitude, not a precise count.
- The classifier is consulted ONLY when no marker matched at any rotation.
  It can never override a positive marker identification.

## v2: Gas Cylinder band widened (JOB_CARD_NUMBER_Y_FRACTION 0.045 -> 0.0)

Instrumenting the *failing* pages (drawing every OCR text box over the page
next to the accepted band) showed the numbers were being detected cleanly
and discarded purely on position: "1888" at y-centre 0.018 conf 0.92,
"1725" at y-centre 0.058 with a top edge of ~0.040 -- missing the old 0.045
floor by a hair. Pages CamScanner crops tighter above the title shift every
row upward, so a floor measured on generously-cropped pages cuts into
tightly-cropped ones.

An approach tried first and REMOVED: anchoring to the printed title's row
(the way the Delivery Challan anchors to its G/P#/DC# label). It never
fired -- RapidOCR fragments the Urdu title, so no single text box is ever
wide enough to identify as the title. Measured gain was exactly zero. The
code was deleted rather than left in place; a mechanism that cannot trigger
is worse than none, because it reads as though the case is handled.

### Verified against ground truth, with a control

Every file in scanned/Gas Cylinder is named for its correct number, so the
change can be checked for silently altering an already-correct answer --
the specific failure that got an earlier widening reverted ("059" read as
"09" at full agreement from a clipped crop).

The naive check is confounded: that folder also holds pages filed by the
Gemini pass and the recovery re-run, which the band never resolved at all
and which therefore look like regressions. So BOTH band settings were run
over the same 220 pages in the same process and compared to each other:

    old band (0.045, 0.11):  190/220 correct (86.4%),  2 wrong values
    new band (0.0,   0.11):  194/220 correct (88.2%),  2 wrong values
    fixed by widening: 5     broken by widening: 1 (to review, not to a wrong value)

The 2 wrong values occur under BOTH bands, so they are pre-existing and not
caused by this change. On the v2 review queue the same change resolved
41/200 (20.5%) of previously-stuck pages, all Gas Cylinder.

### Known pre-existing defect this exposed

"003" reads as "033" and "062" reads as "02", both at full 3/3 zoom
agreement, so the gate cannot catch them -- every zoom variant shares the
same mis-bounded crop. This is the same class of failure described in
JOB_CARD_NUMBER_Y_FRACTION's history. It is a RECOGNITION defect, not a
position one, and is the case the unused digit-only CRNN in models/ was
meant for: an architecturally different reader would have to agree before
such a read is trusted (see v3/main.py's rescue path for the pattern).

## v2: Liquid Nitrogen band left ALONE (measured, widening rejected)

LN is the worst-covered book: 76 of 89 known-answer pages go to review. The
obvious fix is to widen its Gate Pass band, since the true values span
y 0.069-0.272 while the band only accepts 0.10-0.20 (it reaches just 7/24
of them). Five bands were swept, scored against filenames as truth and
against 120 stuck review pages:

    band                        correct  WRONG  review  recovered
    (0.10,0.20) x 0.10-0.35          13      0      76          0   <- current
    (0.06,0.20) x 0.09-0.35          16      2      71          0
    (0.06,0.24) x 0.09-0.35          12      8      69          5
    (0.05,0.29) x 0.09-0.35          13     11      65          4
    (0.00,0.29) x 0.09-0.40          13      8      68          4

Every widening buys coverage with WRONG values, because the same left-hand
column carries handwritten amounts (8640, 900, 2180 on real pages) below the
Gate Pass field. The current band is the only one at zero wrong, so it
stays. LN coverage stays low ON PURPOSE.

The real fix is not a band at all: the Gate Pass value has no reliable
anchor because its label is Urdu and unreadable to RapidOCR. It needs the
title/label template matching that pipeline/locate.py does in the v1 root
pipeline (match the printed label as an IMAGE, then take a fixed offset from
it). That is the next substantial piece of work for this book type.

## v2: CRNN second opinion wired in (main.py _crnn_rescue)

Ported from v3/main.py. Fires only when RapidOCR's zoom consensus fails to
produce a trusted read, and can only CONFIRM a value some other signal
already produced (the detector's nomination, or a reread that missed
agreement) -- never introduces a third value. Fed the tight detector box,
not crop.py's padded field, because the padding pulls in neighbours this
digit-only model cannot ignore.

Measured, per book type, against filenames as truth:

    before:  339/342 filed correct = 99.1%
    after :  343/345 filed correct = 99.4%   (+3 pages filed, no new errors)

It does NOT fix the residual "003"->"033" / "062"->"02" errors: those reach
full 3/3 agreement, so the rescue path never runs for them. Catching those
needs a VETO on accepted reads, which is a different trade (coverage for
precision) and is measured separately.

## v2: DC_ROW_Y_TOLERANCE widened 0.014 -> 0.024 (measured)

The DC number is handwritten ABOVE the printed label's baseline, not centred
on it, so the offset is real and one-sided: |delta_y| p95 = 0.0184 measured
over 30 stuck challans, against a tolerance of 0.014 -- only 17/30 of the
real values were reachable.

The DATE row sits immediately below the DC# row, so a tolerance that reached
it would file "15-1-026" as a document number. Swept rather than guessed,
scored against 120 filed challans (filename = truth) plus 90 stuck pages:

    tolerance  correct  WRONG  recovered
    0.014          114      0          6
    0.018          114      0         20
    0.020          114      0         21
    0.024          114      0         26     <- chosen
    0.030          114      0         26

Zero wrong at every setting and recovery plateaus at 0.024, so 0.024 takes
all the available gain without buying extra reach toward the date row.

## v2: AI pass is now HYBRID (auto-file on agreement, else suggest)

Gemini's precision was never measured before this. Against filenames as
truth it reads 32/33 correct (97.0%), declines 7/40 (safe -- the page just
stays in review), and never got a book type wrong. But 97% means roughly 3
wrong numbers per 100 filed, versus 0.6 for the deterministic path, and a
wrong number is silent corruption in an archive nobody re-reads.

So the AI no longer files on its own authority. It files only when its
answer matches a number the deterministic pipeline independently nominated
-- including candidates the gate REJECTED, since a nomination is still
evidence about what is printed there even when it wasn't strong enough to
file on. Same principle as the CRNN rescue: one confident opinion is not
evidence, two decorrelated ones are.

When they disagree the page stays in review, but keeps the AI's answer as a
suggestion stored in <output>/.suggestions.json and pre-filled in the review
box, so the reviewer confirms with one keypress instead of typing. That is
where most of the human time saving comes from, and it costs no precision
because a human still decides.

Measured on 60 real review-queue pages:

    auto-filed (both readers agreed)   25%
    suggested (one-click confirm)      32%
    still blank                        43%

i.e. 57% of the review queue either leaves it or becomes a single click.
A strong correctness signal fell out of this test: on consecutive pages of
one bundle the answers ran 1822, 1823, 1824 ... 1830 unbroken, alternating
between filed and suggested -- the suggestions fill the gaps in a real
sequential book series exactly, which is not something wrong reads do.

## v2: CRNN VETO measured and REJECTED (the rescue stays, the veto does not)

The CRNN rescue cannot reach the residual "003"->"033" / "062"->"02" errors,
because those reach full 3/3 zoom agreement and the rescue only runs when a
read FAILS. Catching them needs a veto: ask the CRNN about ACCEPTED reads
too and send the page to review when it disagrees. Measured over 120
known-answer Gas Cylinder pages:

    CRNN agreed with the 3/3 read      : 45
    CRNN declined (aspect guard)       :  0
    CRNN disagreed, 3/3 read was WRONG :  2   <- a veto would save these
    CRNN disagreed, 3/3 read was RIGHT : 54   <- a veto would destroy these

27 correct files lost per wrong file caught. Rejected outright.

The reason matters for anyone tempted to retry it: this model was trained on
the MAIN ARCHIVE's mechanical numbering-stamp font, and Peshawar's numbers
are handwritten ballpoint -- a different distribution, so it is simply a
weak reader here. That produces a sharp asymmetry:

  * as a CONFIRMER it is safe. It can only agree with a value another signal
    already produced, so being weak just means it confirms less often --
    measured as +3 pages filed and +0.3pp precision, no downside.
  * as a VETOER it is destructive, because every one of its own errors
    becomes a rejected correct answer.

A weak second opinion is worth having on the "should I trust this?" side and
worthless on the "should I reject this?" side. Retrying the veto is only
sensible with a model actually trained on this corpus's handwriting.

## v2 FINAL measured state (handover)

Full clean re-run of the JANUARY test set (33 files, 1712 pages), fresh
input copy and fresh output root so nothing real was touched:

                     v1 (original)   v2 (final)
    auto-filed              98            531
    review                 767            511
    discarded              861            669
    auto-file rate        11.3%          51.0%   (share of relevant pages)

Per-type precision, filenames as ground truth, with the final code
(DC_ROW_Y_TOLERANCE 0.024 + CRNN rescue):

    Gas Cylinder         filed 104  correct 102  WRONG 2  ->  98.1%
    Delivery Challan     filed 114  correct 114  WRONG 0  -> 100.0%
    Liquid Nitrogen      filed  14  correct  14  WRONG 0  -> 100.0%
    Liquid Tank Decant   filed 113  correct 113  WRONG 0  -> 100.0%
    OVERALL              343/345 = 99.42%

The 2 remaining errors are the known mis-bounded-crop pair ("003"->"033",
"062"->"02"), which pass the zoom gate because all three variants share one
bad crop. The CRNN veto that would catch them was measured and rejected --
it costs 27 correct files per error caught (see that section).

Review-queue effect after the hybrid AI pass: ~25% of pages leave the queue
(both readers agreed), ~32% arrive with the number pre-filled for one-click
confirmation, ~43% still need typing. Before, all of them arrived blank.

KNOWN GAP, deliberate: Liquid Nitrogen files 1 of 171. Every band widening
tested bought coverage with wrong values, so the band was left alone. It
needs label-image template matching (pipeline/locate.py's technique), not a
band. Fixing it should put overall coverage around 65-70%.

ROLLOUT CAVEAT: every constant in detect.py was measured on Peshawar's own
forms, and the page classifier was trained only on Peshawar layouts. Before
trusting this on another warehouse, run the same failure-instrumentation
diagnostic -- render the pages it gets wrong with the OCR boxes and the
accepted band drawn on. That one hour is what found the DC# field, every
band offset, and ~1,470 documents a year being silently discarded. Skipping
it is how the same silent loss happens somewhere else unnoticed.
