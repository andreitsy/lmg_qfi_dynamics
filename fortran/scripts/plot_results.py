#!/usr/bin/env python3
"""Plot the QFI dynamics written by the Fortran solver.

Reads DATA_DIR/<prefix><STATE>_N=<N>_B=<B>.csv for the four initial states
and draws them with lmg_qfi.plotting.plot_qfi_data_subplot, the function
behind `quantum_fisher_information_simulation_mpmath.py --plot`. Unlike
--plot it reads only files with the given prefix, so solver output can sit
next to mpmath reference files of the same N without being mixed with them.

    uv run python fortran/scripts/plot_results.py results/dense 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mpmath as mp  # noqa: E402
import pandas as pd  # noqa: E402

import lmg_qfi.plotting as plotting  # noqa: E402
from lmg_qfi.config import read_defaults_args_from_config  # noqa: E402

STATES = ("PHYS", "CAT_SUM", "GS_CAT", "GS_PHYS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("system_size", type=int)
    parser.add_argument("--x-coupling", default="0.4", help="B (default 0.4)")
    parser.add_argument("--prefix", default="fortran.", help="file prefix (default 'fortran.')")
    parser.add_argument("--output", type=Path, help="default: DATA_DIR/qfi_dynamics_N=<N>_B=<B>.png")
    parser.add_argument("--width", type=float, default=15.0, help="figure width in inches (default 15)")
    parser.add_argument("--label-y", type=float, default=plotting.Y_LABEL_COORDINATE,
                        help="height of the 2pi/Delta labels in units of the y axis "
                             f"(default {plotting.Y_LABEL_COORDINATE})")
    args = parser.parse_args()
    plotting.Y_LABEL_COORDINATE = args.label_y

    params = read_defaults_args_from_config()
    params.N = args.system_size
    params.B = mp.mpf(args.x_coupling)
    label = f"N={params.N}_B={float(params.B):.2f}"

    simulations = {}
    for state in STATES:
        path = args.data_dir / f"{args.prefix}{state}_{label}.csv"
        if path.is_file():
            simulations[state] = pd.read_csv(path)
            print(f"{path}: {len(simulations[state])} points")
    if not simulations:
        parser.error(f"no {args.prefix}<STATE>_{label}.csv files in {args.data_dir}")

    fig, ax = plt.subplots(figsize=(args.width, 10))
    plotting.plot_qfi_data_subplot(ax, simulations, params, max_time_pow=None)
    plt.tight_layout()
    output = args.output or args.data_dir / f"qfi_dynamics_{label}.png"
    plt.savefig(output, dpi=300)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
