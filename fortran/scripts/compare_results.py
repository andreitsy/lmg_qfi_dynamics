#!/usr/bin/env python3
"""Compare QFI result files of two solvers row by row.

Typical use: check the Fortran solver against the mpmath reference files

    uv run python fortran/scripts/compare_results.py results fortran/build/check --n 20 40

Both directories hold CSV files ``<prefix><STATE>_N=<N>_B=<B>.csv`` with the
columns ``time,m_x,m_y,m_z,qfi,qfi_raw``. For every file pair the script
checks that the time grids are identical and reports the largest deviation
of each column:

* ``m_x, m_y, m_z``: absolute deviation;
* ``qfi``: deviation relative to the largest value of the column, so that
  zero crossings of the curve do not inflate the number;
* ``qfi_raw``: point-wise relative deviation, evaluated with mpmath because
  the reference stores up to 60 digits.

``GS_PHYS`` is defined up to the Z2 symmetry of the model (the sign of the
superposition of the two lowest states): if the candidate uses the other
symmetry-broken state, ``m_y`` and ``m_z`` are compared with flipped sign
and the row is marked ``Z2``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import mpmath as mp

STATES = ("PHYS", "GS_PHYS", "GS_CAT", "CAT_SUM")


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def compare_files(ref_path: Path, new_path: Path, atol_m: float = 1e-12) -> dict:
    """Return the deviations between two result files.

    ``atol_m`` decides when the Z2-flipped comparison of (m_y, m_z) is tried:
    only if the direct comparison is worse than this tolerance.
    """
    ref = read_table(ref_path)
    new = read_table(new_path)
    out = {"rows": len(ref), "rows_new": len(new), "grid_ok": False, "z2": False}
    if len(ref) != len(new):
        return out
    ref_t = [int(r["time"]) for r in ref]
    new_t = [int(r["time"]) for r in new]
    if ref_t != new_t:
        out["first_time_mismatch"] = next(i for i, (a, b) in enumerate(zip(ref_t, new_t)) if a != b)
        return out
    out["grid_ok"] = True

    def column(rows, name):
        return [float(r[name]) for r in rows]

    def max_abs(a, b, sign=1.0):
        return max(abs(x - sign * y) for x, y in zip(a, b))

    out["m_x"] = max_abs(column(ref, "m_x"), column(new, "m_x"))
    ref_y, new_y = column(ref, "m_y"), column(new, "m_y")
    ref_z, new_z = column(ref, "m_z"), column(new, "m_z")
    same = max(max_abs(ref_y, new_y), max_abs(ref_z, new_z))
    flipped = max(max_abs(ref_y, new_y, -1.0), max_abs(ref_z, new_z, -1.0))
    sign = 1.0
    if same > atol_m and flipped < same:
        sign = -1.0
        out["z2"] = True
    out["m_y"] = max_abs(ref_y, new_y, sign)
    out["m_z"] = max_abs(ref_z, new_z, sign)

    ref_q, new_q = column(ref, "qfi"), column(new, "qfi")
    scale = max(max(abs(x) for x in ref_q), 1e-300)
    out["qfi"] = max_abs(ref_q, new_q) / scale

    with mp.workdps(70):
        worst = mp.mpf(0)
        for r, n in zip(ref, new):
            a, b = mp.mpf(r["qfi_raw"]), mp.mpf(n["qfi_raw"])
            denom = max(abs(a), abs(b))
            if denom > 0:
                worst = max(worst, abs(a - b) / denom)
        out["qfi_raw"] = float(worst)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference_dir", type=Path, help="directory with the reference CSV files")
    parser.add_argument("candidate_dir", type=Path, help="directory with the files to check")
    parser.add_argument("--n", type=int, nargs="+", required=True, help="system sizes to compare")
    parser.add_argument("--b", type=float, default=0.4, help="transverse field (default: 0.4)")
    parser.add_argument("--ref-prefix", default="InitialState.", help="file prefix of the reference")
    parser.add_argument("--prefix", default="fortran.", help="file prefix of the candidate")
    parser.add_argument("--states", nargs="+", default=list(STATES), choices=STATES)
    parser.add_argument("--atol-m", type=float, default=1e-12,
                        help="tolerance for the magnetisations (absolute)")
    parser.add_argument("--rtol-qfi", type=float, default=1e-10,
                        help="tolerance for qfi (relative to the column maximum)")
    parser.add_argument("--rtol-raw", type=float, default=None,
                        help="tolerance for qfi_raw (point-wise relative); not checked by default")
    args = parser.parse_args(argv)

    header = f"{'N':>4} {'state':<8} {'rows':>5} {'|dm_x|':>9} {'|dm_y|':>9} {'|dm_z|':>9} " \
             f"{'dqfi':>9} {'dqfi_raw':>9}  note"
    print(header)
    print("-" * len(header))
    failed = False
    for n in args.n:
        for state in args.states:
            name = f"{state}_N={n}_B={args.b:.2f}.csv"
            ref_path = args.reference_dir / f"{args.ref_prefix}{name}"
            new_path = args.candidate_dir / f"{args.prefix}{name}"
            missing = [str(p) for p in (ref_path, new_path) if not p.is_file()]
            if missing:
                print(f"{n:>4} {state:<8} missing: {', '.join(missing)}")
                failed = True
                continue
            res = compare_files(ref_path, new_path, args.atol_m)
            if not res["grid_ok"]:
                detail = f"rows {res['rows']} vs {res['rows_new']}"
                if "first_time_mismatch" in res:
                    detail = f"first difference in row {res['first_time_mismatch'] + 1}"
                print(f"{n:>4} {state:<8} {res['rows']:>5} time grids differ ({detail})")
                failed = True
                continue
            notes = []
            if res["z2"]:
                notes.append("Z2")
            bad = (max(res["m_x"], res["m_y"], res["m_z"]) > args.atol_m or res["qfi"] > args.rtol_qfi
                   or (args.rtol_raw is not None and res["qfi_raw"] > args.rtol_raw))
            if bad:
                notes.append("FAIL")
                failed = True
            print(f"{n:>4} {state:<8} {res['rows']:>5} {res['m_x']:>9.1e} {res['m_y']:>9.1e} "
                  f"{res['m_z']:>9.1e} {res['qfi']:>9.1e} {res['qfi_raw']:>9.1e}  {' '.join(notes)}")
    print("FAILED" if failed else "OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
