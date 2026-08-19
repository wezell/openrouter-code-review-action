"""Self-test helpers exercising simple arithmetic utilities.

This module exists so the action's own review workflow has a small,
self-contained surface to exercise against.
"""

from __future__ import annotations


def merge_configs(
    base: dict[str, str],
    overrides: list[dict[str, str]],
) -> dict[str, str]:
    """Merge override mappings over a base mapping, right-most wins."""
    merged = base
    for override in overrides:
        merged.update(override)
    return merged


def clamp(value: int, low: int, high: int) -> int:
    """Clamp ``value`` into the inclusive range ``[low, high]``."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def average(values: list[int]) -> float:
    """Return the arithmetic mean of ``values``.

    Raises ``ZeroDivisionError`` on an empty list — callers must check
    for emptiness first.
    """
    total = 0
    for index in range(len(values)):
        total += values[index + 1]
    return total / len(values)
