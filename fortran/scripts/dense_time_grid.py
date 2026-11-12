#!/usr/bin/env python3
"""Write a dense stroboscopic time grid for the Fortran solver (--times-file).

The grid contains every period up to UNTIL - 1 and LOG_POINTS logarithmically
spaced times from UNTIL to 10**MAX_DEGREE (integer values, sorted, without
duplicates). With the defaults it has 69 999 points, about 100 times the
699 points of the standard grid (num-points = 300).

    uv run python fortran/scripts/dense_time_grid.py --system-size 100 times_N100.txt
    fortran/build/lmg_qfi --system-size 100 --steps 250 --times-file times_N100.txt

MAX_DEGREE is the last decade of the standard grid, int(log10(4 pi / gap)) + 1
with the gap of the two lowest levels of H0; it is computed from --system-size
and --x-coupling unless given explicitly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mpmath as mp
import numpy as np

from lmg_qfi.operators import create_hamiltonian_h0


def max_time_degree(n_spins: int, b: str, j: str = "1.0") -> int:
    """Last decade of the standard time grid for this system."""
    # The ground-state splitting falls off exponentially with N, so the
    # spectrum needs far more than double precision.
    with mp.workdps(100):
        energies = sorted(mp.eigh(create_hamiltonian_h0(mp.mpf(j), mp.mpf(b), n_spins))[0])
        return int(mp.log10(mp.fabs(4 * mp.pi / (energies[1] - energies[0])))) + 1


def dense_grid(max_degree: int, until: int = 10_000, log_points: int = 60_000) -> list[int]:
    times = set(range(1, until))
    times.update(int(x) for x in np.logspace(np.log10(until), max_degree, log_points))
    return sorted(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="file to write, one time per line")
    parser.add_argument("--system-size", type=int, help="N, to compute the last decade")
    parser.add_argument("--x-coupling", default="0.4", help="B, to compute the last decade (default 0.4)")
    parser.add_argument("--max-degree", type=int, help="last decade, instead of computing it")
    parser.add_argument("--until", type=int, default=10_000, help="every period below this time")
    parser.add_argument("--log-points", type=int, default=60_000, help="log-spaced times above it")
    args = parser.parse_args()

    if args.max_degree is None:
        if args.system_size is None:
            parser.error("give --system-size or --max-degree")
        args.max_degree = max_time_degree(args.system_size, args.x_coupling)
    times = dense_grid(args.max_degree, args.until, args.log_points)
    args.output.write_text("\n".join(str(t) for t in times) + "\n")
    print(f"{len(times)} times from {times[0]} to {times[-1]} (max degree {args.max_degree}) -> {args.output}")


if __name__ == "__main__":
    main()
