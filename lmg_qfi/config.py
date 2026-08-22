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


class EstimationParameter(Enum):
    """Which parameter of the AC field the QFI is computed for."""
    AMPLITUDE = "amplitude"
    FREQUENCY = "frequency"


class SolverType(Enum):
    """Numerical backend used to run the simulation."""
    QUSPIN = "quspin"   # fast double precision (QuSpin + numpy/scipy)
    MPMATH = "mpmath"   # arbitrary-precision reference


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
    parameter: EstimationParameter = EstimationParameter.AMPLITUDE
    omega: Optional[mp.mpf] = None  # None -> resonant 2*pi/(freq*T)
    solver: SolverType = SolverType.QUSPIN

    def __str__(self):
        omega_str = "resonant" if self.omega is None else f"{float(self.omega)}"
        return (f"SimulationParams(N={self.N}, B={float(self.B)}, T={float(self.T)}, "
                f"J={float(self.J)}, phi_kick_phase={float(self.phi_kick_phase)}, h={float(self.h)}, "
                f"parameter={self.parameter.value}, omega={omega_str}, "
                f"solver={self.solver.value}) "
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
class SLDInformation:
    time: float
    L_expectation: float          # <L> — ~0 for normalized states
    L_squared_expectation: float  # <L^2> = raw QFI from SLD
    qfi_from_sld: float           # <L^2> / (N^2 t^2), normalized QFI


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

        omega_raw = sim_config.get("omega", fallback=None)
        omega = (None if omega_raw is None or omega_raw.strip().lower() == "resonant"
                 else convert_float_value(omega_raw))

        run_arguments = {"dps": int(sim_config["dps"]),
                         "steps_floquet_unitary": int(sim_config["steps-floquet-unitary"]),
                         "num_points": int(sim_config["num-points"]),
                         "output_dir": files_config["output-dir"],
                         "log_file": files_config.get("log-file")}
        if (max_time_degree := sim_config.get("max-time-degree", fallback=None)) is not None:
            run_arguments["max_time_degree"] = int(max_time_degree)

        params_simulation = SimulationParams(
            run_arguments,
            N=int(sim_config["n"]),
            J=convert_float_value(sim_config["j"]),
            B=convert_float_value(sim_config["b"]),
            T=convert_float_value(sim_config["t"]),
            phi_kick_phase=convert_float_value(sim_config["phi-kick-phase"]),
            h=convert_float_value(sim_config["h"]),
            phi_0=convert_float_value(sim_config["phi-0"]),
            freq=int(sim_config["frequency"]),
            parameter=EstimationParameter(
                sim_config.get("parameter", fallback="amplitude").strip().lower()),
            omega=omega,
            solver=SolverType(
                sim_config.get("solver", fallback=SolverType.QUSPIN.value).strip().lower()),
        )
    else:
        params_simulation = SimulationParams(
            {"num_periods": 40, "output_dir": "./results"},
            N=20,
            J=mp.mpf(1.0),
            B=mp.mpf(0.4),
        )
    return params_simulation


def add_simulation_cli_arguments(parser: argparse.ArgumentParser,
                                 simulation_params: SimulationParams):
    """Register the CLI flags shared by all simulation entry points."""
    parser.add_argument(
        "--x-coupling",
        type=float,
        default=float(simulation_params.B),
        help="X Coupling (transverse field B)",
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
        choices=[state.value for state in InitialState],
        help="Single initial state to run (default: all four)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot graph from existing CSV results instead of running",
    )
    parser.add_argument(
        "--parameter",
        type=str,
        default=None,
        choices=[p.value for p in EstimationParameter],
        help="Estimated parameter of the AC field: QFI over the amplitude h "
             "or over the frequency omega (default: from parameters.ini)",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=None,
        help="Probe frequency omega for frequency estimation "
             "(default: resonant 2*pi/(nu*T))",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=None,
        help="AC field amplitude h (frequency mode requires h != 0)",
    )
    parser.add_argument(
        "--max-time-degree",
        type=int,
        default=None,
        help="Cap the maximum simulated time at 10^DEGREE Floquet periods "
             "(frequency mode cost grows linearly with the maximum time)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default=None,
        choices=[s.value for s in SolverType],
        help="Numerical backend: quspin (fast, double precision) or "
             "mpmath (arbitrary precision) (default: from parameters.ini)",
    )


def apply_simulation_cli_arguments(simulation_params: SimulationParams, args):
    """Fold parsed CLI arguments back into the SimulationParams."""
    simulation_params.N = args.system_size
    simulation_params.B = mp.mpf(f"{args.x_coupling}")
    simulation_params.run_arguments["init_state"] = args.init_state
    if args.parameter is not None:
        simulation_params.parameter = EstimationParameter(args.parameter)
    if args.omega is not None:
        simulation_params.omega = mp.mpf(f"{args.omega}")
    if args.amplitude is not None:
        simulation_params.h = mp.mpf(f"{args.amplitude}")
    if args.max_time_degree is not None:
        simulation_params.run_arguments["max_time_degree"] = args.max_time_degree
    if args.solver is not None:
        simulation_params.solver = SolverType(args.solver)
    return simulation_params


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
                    "--amplitude 0.0 --parameter frequency`"
    )
    add_simulation_cli_arguments(parser, simulation_params)
    args = parser.parse_args()
    apply_simulation_cli_arguments(simulation_params, args)

    return simulation_params, args
