#!/usr/bin/env python3
"""
Quantum Fisher Information Simulation for the LMG Model.

This script provides a command-line interface for running QFI dynamics simulations.
All core functionality is provided by the lmg_qfi package.
"""

import logging
import re
import pandas as pd

from pathlib import Path

# Import from the lmg_qfi package
from lmg_qfi import (
    InitialState,
    SimulationParams,
    run_simulation,
    save_to_file_qfi_dynamics,
    plot,
    setup_logging,
)
from lmg_qfi.config import read_defaults_args_from_config, parse_arguments as _parse_arguments
import mpmath as mp


def parse_arguments():
    """Parse command-line arguments for the simulation with config file support."""
    # Read config from the same directory as this script
    config_file = Path(__file__).parent / "parameters.ini"
    simulation_params = read_defaults_args_from_config(str(config_file))
    
    import argparse
    parser = argparse.ArgumentParser(
        description="Quantum Fisher Information Simulation Tool.\n"
                    "Running the script from command line would look "
                    "something like this:\n"
                    "`python ./quantum_fisher_information_simulation.py "
                    "--amplitude 0.0 --plot-type all`"
    )
    
    parser.add_argument(
        "--x-coupling",
        type=float,
        default=simulation_params.B,
        help="X Coupling",
    )
    parser.add_argument(
        "--system-size",
        type=int,
        default=simulation_params.N,
        help="System size",
    )
    parser.add_argument(
        "--init-state",
        type=str,
        default=None,
        help="Initial state",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot graph",
    )
    
    args = parser.parse_args()
    simulation_params.N = args.system_size
    simulation_params.B = mp.mpf(f"{args.x_coupling}")
    simulation_params.run_arguments["init_state"] = args.init_state
    
    return simulation_params, args


if __name__ == "__main__":
    simulation_params, args = parse_arguments()
    
    if log_file_name := simulation_params.run_arguments.get("log_file"):
        log_handler = logging.FileHandler(
            Path(__file__).parent / simulation_params.run_arguments["output_dir"] / log_file_name)
    else:
        log_handler = logging.StreamHandler()
    setup_logging(log_handler)
    
    if state_str := simulation_params.run_arguments.get("init_state"):
        if state_str not in ["GS_phys", "GS_cat", "CatSum", "Phys"]:
            raise Exception("Wrong init_state! it should be one of init_state")
        init_states = [InitialState(state_str)]
    else:
        init_states = [
            InitialState.PHYS,
            InitialState.GS_PHYS,
            InitialState.GS_CAT,
            InitialState.CAT_SUM,
        ]
    output_dir = Path(__file__).parent / simulation_params.run_arguments["output_dir"]

    if args.plot:
        logging.info("Plotting results")
        simulations = dict()
        for file in output_dir.iterdir():
            if (file.is_file()
                    and f"N={simulation_params.N}" in file.name
                    and f"B={float(simulation_params.B):.2f}" in file.name):
                match = re.search(r'\.([^.]+)_N=', file.name)
                if match:
                    logging.info(f"Adding file for plot {file.name}")
                    initial_state = match.group(1)
                    simulations[initial_state] = pd.read_csv(file)
        if simulations:
            plot(simulations, simulation_params, output_dir)
        else:
            logging.warning(f"Didn't find files in {output_dir}")

    else:
        logging.info(f"Run simulation with params: {simulation_params}")
        simulations = run_simulation(simulation_params, init_states)

        for state, results in simulations.items():
            output_file_name = f"{state}_N={simulation_params.N}_B={float(simulation_params.B):.2f}.csv"
            save_to_file_qfi_dynamics(results=results, output_file=output_dir / output_file_name)
