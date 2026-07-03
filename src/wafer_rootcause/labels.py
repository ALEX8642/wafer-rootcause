"""labels.py — the 8 signature labels and MixedWM38 combo-validity rules.

Label order matches wafer-mixed's verified ordering (wafer-mixed
docs/DATA.md). Validity rules reproduce the structure of MixedWM38's 38
combos: at most one of {Center, Donut}, at most one of
{Edge-Loc, Edge-Ring}, Loc and Scratch mix freely, and Near-full / Random
appear only as singles. The simulator enforces these by construction so
every simulated label set has matching real maps in the test split.
"""
from __future__ import annotations

LABELS: list[str] = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Scratch", "Random",
]

# Single-only labels: never co-occur with anything in MixedWM38.
ISOLATED: frozenset[str] = frozenset({"Near-full", "Random"})

# Labels that may participate in mixed patterns.
MIXABLE: frozenset[str] = frozenset(LABELS) - ISOLATED

# Mutually exclusive pairs within the mixable set.
EXCLUSIVE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"Center", "Donut"}),
    frozenset({"Edge-Loc", "Edge-Ring"}),
)


def can_add(labels: set[str], new: str) -> bool:
    """True if adding `new` to `labels` keeps the set a valid combo."""
    if new in labels:
        return False
    if new in ISOLATED:
        return not labels
    if labels & ISOLATED:
        return False
    for group in EXCLUSIVE_GROUPS:
        if new in group and labels & group:
            return False
    return True


def is_valid_combo(labels: set[str]) -> bool:
    """True if `labels` is one of MixedWM38's 38 valid combos (incl. empty)."""
    unknown = labels - set(LABELS)
    if unknown:
        return False
    if labels & ISOLATED:
        return len(labels) == 1
    for group in EXCLUSIVE_GROUPS:
        if len(labels & group) > 1:
            return False
    return True
