"""Configuration parsing and simulation parameters."""

import argparse
import configparser
import mpmath as mp
import os

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class InitialState(Enum):
    """Enumeration of possible initial quantum states."""
    GS_PHYS = "GS_phys"
    GS_CAT = "GS_cat"
    CAT_SUM = "CatSum"
    PHYS = "Phys"


@dataclass
class SimulationParams:
    run_arguments: dict
    N: int
    J: mp.mpf
    B: mp.mpf
    T: mp.mpf = mp.mpf(1.0)
    phi_kick_phase: mp.mpf = mp.pi
    h: mp.mpf = mp.mpf(0)
    varphi: mp.mpf = mp.mpf(0)
    theta: mp.mpf = mp.mpf(0)
    freq: int = 2
    phi_0: mp.mpf = mp.mpf(0)

    def __str__(self):
        return (f"SimulationParams(N={self.N}, B={float(self.B)}, T={float(self.T)}, "
                f"J={float(self.J)}, phi_kick_phase={float(self.phi_kick_phase)}, h={float(self.h)}) "
                f"run with {self.run_arguments}")


@dataclass
class QFIInformation:
    m_x: float
    m_y: float
    m_z: float
    qfi: float
    time: float
    qfi_raw_value: str


@dataclass
class UF:
    """Floquet unitary decomposition."""
    eigenvalues: mp.matrix
    U: mp.matrix
    U_inv: mp.matrix


def read_defaults_args_from_config(config_file: Optional[str] = None) -> SimulationParams:
    """
    Read default simulation parameters from config file.
    
    Parameters
    ----------
    config_file : str, optional
        Path to config file. If None, looks for parameters.ini in parent directory.
    
    Returns
    -------
    SimulationParams
        Simulation parameters with defaults from config file.
    """
    config = configparser.ConfigParser()
    
    if config_file is None:
        # Look for parameters.ini in the parent directory (project root)
        config_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "parameters.ini"
        )

    def convert_float_value(val: str):
        if val == "pi":
            return mp.pi
        else:
            return mp.mpf(val)

    if os.path.exists(config_file):
        config.read(config_file)
        sim_config = config["Simulation"]
        files_config = config["Files"]

        params_simulation = SimulationParams(
            {"dps": int(sim_config["dps"]),
             "steps_floquet_unitary": int(sim_config["steps-floquet-unitary"]),
             "num_points": int(sim_config["num-points"]),
             "output_dir": files_config["output-dir"],
             "log_file": files_config.get("log-file")},
            N=int(sim_config["n"]),
            J=convert_float_value(sim_config["j"]),
            B=convert_float_value(sim_config["b"]),
            T=convert_float_value(sim_config["t"]),
            phi_kick_phase=convert_float_value(sim_config["phi-kick-phase"]),
            h=convert_float_value(sim_config["h"]),
            phi_0=convert_float_value(sim_config["phi-0"]),
            freq=int(sim_config["frequency"]),
        )
    else:
        params_simulation = SimulationParams(
            {"num_periods": 40, "output_dir": "./results"},
            N=20,
            J=mp.mpf(1.0),
            B=mp.mpf(0.4),
        )
    return params_simulation


def parse_arguments(script_path: Optional[str] = None):
    """
    Parse command-line arguments for the simulation.
    
    Parameters
    ----------
    script_path : str, optional
        Path to the main script, used to locate config file.
    
    Returns
    -------
    tuple
        (SimulationParams, parsed args)
    """
    if script_path:
        config_file = os.path.join(os.path.dirname(script_path), "parameters.ini")
        simulation_params = read_defaults_args_from_config(config_file)
    else:
        simulation_params = read_defaults_args_from_config()
    
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
