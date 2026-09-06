"""Decide what to trust.

A wrong ECR number is far more expensive than a missing one: a missing number
costs a few seconds of somebody's attention, a wrong one silently misfiles a
document in an archive nobody will re-check. So the gate is deliberately
asymmetric -- it would rather send work to a human than write a guess.

The dial is consensus across preprocessing variants (see recognize.py).
AUTO_ACCEPT_VOTES is set to full agreement -- every variant reading the same
digits -- because that is the setting measured at zero false positives on the
sample batch. Everything below it is still surfaced: it goes to review with its
candidate attached, so a reviewer confirms rather than types.

Loosening this is a business decision, not a technical one, and needs an answer
to "how much more expensive is a wrong number than a missing one?". Until then,
full agreement is the safe default: a flagged file costs seconds of attention, a
misfiled one is silent corruption in an archive nobody re-reads.
"""

from dataclasses import dataclass
from enum import Enum

# Full agreement -- every variant a Reading was read under agreeing on the
# same digits. Compared against reading.variants rather than a fixed count:
# pipeline.detect's candidates are read under a different, differently-sized
# zoom set than the flatbed pipeline's crops (recognize.DETECTION_ZOOMS vs
# recognize.ZOOMS), so "full agreement" has to mean "all of whichever set was
# actually used", not a number borrowed from one specific set that would
# silently stop meaning full agreement if the other set's length ever changes.
MIN_SUGGEST_VOTES = 2  # below this, a candidate is noise and not worth showing


class Status(str, Enum):
    ACCEPTED = "accepted"        # trusted enough to name a file with
    REVIEW = "review"            # a plausible candidate a human should confirm
    UNREADABLE = "unreadable"    # nothing legible in the crop


@dataclass
class Decision:
    status: Status
    number: str | None
    votes: int
    variants: int
    card: str
    anchor_score: float
    repeat: bool = False  # `number` was already in seen_numbers -- see judge()

    @property
    def trusted(self):
        return self.status is Status.ACCEPTED


def judge(anchor, reading, seen_numbers=None):
    """Classify one (anchor, reading) pair.

    `seen_numbers` is the set of already-accepted numbers in this run. This
    used to reject a collision outright (Status.DUPLICATE, routed to
    review): ECR/DC numbers were assumed unique for good. That assumption
    was wrong -- confirmed by the client, these numbers reset every year at
    the source, so the same number legitimately recurs. Blocking every
    recurrence would have meant every single one detours through review
    forever, for no reason once a client has more than a year of archive.
    A collision is trusted exactly like any other full-agreement read now;
    `repeat=True` just carries the fact forward so callers can log it into
    a separate report a human can spot-check later (real duplicate SCANS
    are still worth catching -- just not by blocking, since most repeats
    are legitimate) rather than have it silently vanish into "accepted".
    """
    common = dict(card=anchor.card.name, anchor_score=anchor.score)

    if reading is None:
        return Decision(Status.UNREADABLE, None, 0, 0, **common)

    if reading.votes < MIN_SUGGEST_VOTES:
        return Decision(
            Status.UNREADABLE, None, reading.votes, reading.variants, **common
        )

    if reading.votes < reading.variants:
        return Decision(
            Status.REVIEW, reading.text, reading.votes, reading.variants, **common
        )

    repeat = seen_numbers is not None and reading.text in seen_numbers
    return Decision(
        Status.ACCEPTED, reading.text, reading.votes, reading.variants, repeat=repeat, **common
    )


def file_is_clean(decisions):
    """True when every located card produced a trusted number.

    A file only gets renamed when nothing on it needed a human. Partial
    confidence is not good enough: renaming on half the numbers would bury the
    fact that the other half was guessed.
    """
    return bool(decisions) and all(d.trusted for d in decisions)


def resolve_page(found, seen_numbers):
    """Decide the ECR number for one single-sheet page (pipeline.detect).

    file_is_clean requires *every* found item to be trusted, which is right
    for the flatbed pipeline: there, multiple found anchors are multiple real
    cards, and each genuinely needs its own trusted number. A detected page's
    candidates are different in kind -- pipeline.detect's size/digit filter
    nominates at most one real stamp plus zero or more heuristic false
    positives (a bold field label, a fragment of the address line). An
    untrusted false positive must not veto a trusted candidate sitting right
    next to it; it is noise from the candidate search, not a second record.

    Two candidates independently reaching full agreement on *different*
    numbers is a genuine conflict rather than noise (noise reads
    inconsistently across zoom variants and rarely reaches full agreement at
    all), so that case is still routed to review instead of guessed between.

    Returns (accepted_number_or_None, decisions).
    """
    decisions = [judge(anchor, reading, seen_numbers) for anchor, reading in found]
    accepted = {d.number for d in decisions if d.trusted}
    if len(accepted) == 1:
        (number,) = accepted
        seen_numbers.add(number)
        return number, decisions
    return None, decisions
