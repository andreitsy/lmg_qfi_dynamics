#!/usr/bin/env python3
"""
Quantum Fisher Information Simulation for the LMG Model — QuSpin solver.

Fast double-precision alternative to the mpmath solver. Uses QuSpin to build
spin operators and scipy/numpy for Floquet time evolution (~800x faster than
mpmath at dps=15).

CLI usage mirrors quantum_fisher_information_simulation_mpmath.py.
"""

import argparse
import logging
import re
import numpy as np
import pandas as pd

from pathlib import Path

from lmg_qfi.config import InitialState, read_defaults_args_from_config
from lmg_qfi.simulation import generate_time_interval
from lmg_qfi.quspin_solver import get_spin_operators, build_h0, run_quspin_simulation
from lmg_qfi.io import save_to_file_qfi_dynamics
from lmg_qfi.plotting import plot
from lmg_qfi.utils import setup_logging
import mpmath as mp


def parse_arguments():
    config_file = Path(__file__).parent / "parameters.ini"
    p = read_defaults_args_from_config(str(config_file))

    parser = argparse.ArgumentParser(
        description="QFI Simulation (QuSpin solver) — fast double-precision Floquet dynamics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python quantum_fisher_information_simulation_quspin.py\n"
            "  uv run python quantum_fisher_information_simulation_quspin.py "
            "--init-state GS_phys\n"
            "  uv run python quantum_fisher_information_simulation_quspin.py "
            "--system-size 20 --x-coupling 0.4\n"
            "  uv run python quantum_fisher_information_simulation_quspin.py --plot"
        ),
    )
    parser.add_argument("--x-coupling",  type=float, default=float(p.B),
                        help="Transverse field B (default: %(default)s)")
    parser.add_argument("--system-size", type=int,   default=p.N,
                        help="Number of spins N (default: %(default)s)")
    parser.add_argument("--init-state",  type=str,   default=None,
                        choices=["GS_phys", "GS_cat", "CatSum", "Phys"],
                        help="Single initial state to run (default: all four)")
    parser.add_argument("--plot",        action="store_true",
                        help="Plot QFI from existing CSV results instead of running")

    args = parser.parse_args()
    p.N = args.system_size
    p.B = mp.mpf(str(args.x_coupling))
    p.run_arguments["init_state"] = args.init_state
    return p, args


def _build_initial_states(N, J, B, requested):
    """Return dict of {InitialState: np.ndarray} for requested states."""
    Sz, Sx, _ = get_spin_operators(N)
    H0 = build_h0(J, B, N, Sz, Sx)
    evals, evecs = np.linalg.eigh(H0)
    gs, ex = evecs[:, 0], evecs[:, 1]

    def _norm(v):
        return v / np.linalg.norm(v)

    phys    = np.zeros(N + 1, dtype=complex); phys[0] = 1.0
    cat_sum = np.zeros(N + 1, dtype=complex); cat_sum[0] = cat_sum[N] = 1.0

    all_states = {
        InitialState.PHYS:    _norm(phys),
        InitialState.GS_PHYS: _norm(gs + ex),
        InitialState.GS_CAT:  _norm(gs),
        InitialState.CAT_SUM: _norm(cat_sum),
    }
    return {k: v for k, v in all_states.items() if k in requested}


def _last_time_degree(J, B, N, dps=15):
    """Determine max time exponent from the H0 energy gap."""
    with mp.workdps(dps):
        from lmg_qfi.operators import create_hamiltonian_h0
        H = create_hamiltonian_h0(J, B, N)
        energies, _ = mp.eigh(H)
        idx = sorted(range(len(energies)), key=lambda i: mp.re(energies[i]))
        gap = energies[idx[1]] - energies[idx[0]]
        return int(mp.log10(mp.fabs(4 * mp.pi / gap))) + 1


if __name__ == "__main__":
    params, args = parse_arguments()

    log_file = params.run_arguments.get("log_file")
    output_dir = Path(__file__).parent / params.run_arguments["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(output_dir / log_file) if log_file else logging.StreamHandler()
    setup_logging(handler)

    N   = params.N
    J   = float(params.J)
    B   = float(params.B)
    dps = params.run_arguments.get("dps", 15)
    num_points = params.run_arguments.get("num_points", 200)
    steps = params.run_arguments.get("steps_floquet_unitary", 10)

    if args.plot:
        logging.info("Plotting results from existing CSVs")
        simulations = {}
        for f in output_dir.iterdir():
            if f.is_file() and f"N={N}" in f.name and f"B={B:.2f}" in f.name:
                m = re.search(r'\.([^.]+)_N=', f.name)
                if m:
                    logging.info(f"Loading {f.name}")
                    simulations[m.group(1)] = pd.read_csv(f)
        if simulations:
            plot(simulations, params, output_dir)
        else:
            logging.warning(f"No matching result files found in {output_dir}")
    else:
        if state_str := params.run_arguments.get("init_state"):
            init_states = [InitialState(state_str)]
        else:
            init_states = list(InitialState)

        logging.info(f"QuSpin solver — N={N}, B={B}, J={J}, states={[s.value for s in init_states]}")

        qs_params = dict(
            N=N, J=J, B=B,
            T=float(params.T),
            nu=int(params.freq),
            phi=float(params.phi_kick_phase),
            h=float(params.h),
            epsilon=float(mp.mpf(f"1e-{dps // 2}")),
            steps_floquet_unitary=steps,
            varphi=float(params.varphi),
            theta=float(params.theta),
            phi_0=float(params.phi_0),
        )

        last_degree = _last_time_degree(J, B, N, dps)
        time_interval = generate_time_interval(num_points, last_degree)
        logging.info(f"Time interval: {time_interval[0]} … {time_interval[-1]} ({len(time_interval)} points)")

        init_state_vectors = _build_initial_states(N, J, B, set(init_states))

        for state in init_states:
            logging.info(f"Running {state.value} …")
            psi0 = init_state_vectors[state]
            results = run_quspin_simulation(qs_params, time_interval, psi0)

            out = output_dir / f"quspin.{state.value}_N={N}_B={B:.2f}.csv"
            save_to_file_qfi_dynamics(results=results, output_file=out)
            logging.info(f"Saved → {out}")