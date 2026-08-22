#!/usr/bin/env python3
"""
Quantum Fisher Information Simulation for the LMG Model.

Single command-line entry point for the QFI dynamics simulation. The numerical
backend is selected by the `solver` key in parameters.ini (quspin | mpmath,
default quspin) or by the --solver flag; both backends share the same CLI,
output format, and plotting.
"""

import logging
import re
import pandas as pd

from pathlib import Path

from lmg_qfi import (
    EstimationParameter,
    InitialState,
    SimulationParams,
    SolverType,
    save_to_file_qfi_dynamics,
    plot,
    setup_logging,
)
from lmg_qfi.config import parse_arguments as _parse_arguments


def parse_arguments():
    """Parse CLI args with parameters.ini (next to this script) as defaults."""
    return _parse_arguments(script_path=str(Path(__file__).resolve()))


def select_initial_states(params: SimulationParams) -> list:
    """Initial states to run: the --init-state choice or all four."""
    if state_str := params.run_arguments.get("init_state"):
        return [InitialState(state_str)]
    return [
        InitialState.PHYS,
        InitialState.GS_PHYS,
        InitialState.GS_CAT,
        InitialState.CAT_SUM,
    ]


def result_file_name(params: SimulationParams, state: InitialState) -> str:
    """CSV name <solver>.<Fomega.?><state>_N=<N>_B=<B>.csv, same for both solvers.

    The solver prefix keeps quspin and mpmath results apart; the Fomega prefix
    keeps frequency-mode results apart from amplitude-mode ones.
    """
    mode_prefix = ("Fomega."
                   if params.parameter == EstimationParameter.FREQUENCY else "")
    return (f"{params.solver.value}.{mode_prefix}{state.value}"
            f"_N={params.N}_B={float(params.B):.2f}.csv")


def run_solver(params: SimulationParams, init_states: list) -> dict:
    """Dispatch to the selected backend; both return {InitialState: results}."""
    if params.solver == SolverType.MPMATH:
        from lmg_qfi.simulation import run_simulation
        return run_simulation(params, init_states)
    from lmg_qfi.quspin_solver import run_quspin
    return run_quspin(params, init_states)


def collect_plot_frames(params: SimulationParams, output_dir: Path) -> dict:
    """Load the result CSVs matching the active solver, N, B, and mode."""
    frequency_mode = params.parameter == EstimationParameter.FREQUENCY
    frames = {}
    for file in output_dir.iterdir():
        if (file.is_file()
                and file.name.startswith(f"{params.solver.value}.")
                and f"N={params.N}" in file.name
                and f"B={float(params.B):.2f}" in file.name
                and ("Fomega." in file.name) == frequency_mode):
            if match := re.search(r'\.([^.]+)_N=', file.name):
                logging.info(f"Adding file for plot {file.name}")
                frames[match.group(1)] = pd.read_csv(file)
    return frames


def main():
    params, args = parse_arguments()

    output_dir = Path(__file__).parent / params.run_arguments["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if log_file_name := params.run_arguments.get("log_file"):
        log_handler = logging.FileHandler(output_dir / log_file_name)
    else:
        log_handler = logging.StreamHandler()
    setup_logging(log_handler)

    if args.plot:
        logging.info(f"Plotting {params.solver.value} results from {output_dir}")
        if frames := collect_plot_frames(params, output_dir):
            plot(frames, params, output_dir)
        else:
            logging.warning(f"Didn't find {params.solver.value}.* files in {output_dir}")
        return

    init_states = select_initial_states(params)
    logging.info(f"Run simulation with params: {params}")
    simulations = run_solver(params, init_states)

    for state, results in simulations.items():
        output_file = output_dir / result_file_name(params, state)
        save_to_file_qfi_dynamics(results=results, output_file=output_file)
        logging.info(f"Saved results to {output_file}")


if __name__ == "__main__":
    main()
