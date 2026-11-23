"""One precision for computed numbers, so their hashes rarely depend on the processor.

exp, log, pow and everything built on them can differ in the last bits between processors and
math libraries. Such a number is rounded here where it is computed; what follows from it by plain
arithmetic then agrees everywhere. Data values are never rounded.
"""

from __future__ import annotations

# Last-bit differences are near 1e-15 relative, so 10 digits leave about one chance in a million
# per number of falling on a rounding boundary. The statistics answer keys need 1e-8.
DIGITS = 10


def canonical(value: float) -> float:
    return float(f"{value:.{DIGITS - 1}e}")
