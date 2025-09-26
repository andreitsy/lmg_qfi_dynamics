#!/usr/bin/env python3
import os
os.environ['MKL_NUM_THREADS'] = '16' # set number of MKL threads to run in parallel

import logging
import configparser
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
from itertools import combinations_with_replacement
from quspin.tools.Floquet import Floquet_t_vec
from quspin.operators import hamiltonian, quantum_operator
from quspin.basis import spin_basis_1d
import matplotlib as mpl

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Computer Modern"]
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,amsfonts,amssymb}"
DEBUG = True

@dataclass
class SimulationParams:
    run_arguments: dict
    N: int = 10
    J: float = 1.0
    B: float = 0.5
    T: float = 1.0
    phi_kick_phase: float = np.pi
    h: float = 0.0
    freq: int = 2
    phi_0: float = 0.0


@dataclass
class QFIInformation:
    m_x: float
    m_y: float
    m_z: float
    qfi: float
    time: float


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def read_defaults_args_from_config() -> SimulationParams:
    # First, create a config parser to read default values
    config = configparser.ConfigParser()
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "parameters.ini"
    )

    # Read config file if it exists

    def convert_float_value(val: str):
        if val == "pi":
            return np.pi
        else:
            return float(val)

    if os.path.exists(config_file):
        config.read(config_file)
        sim_config = config["Simulation"]
        files_config = config["Files"]

        params_simulation = SimulationParams(
            {"num_periods": int(sim_config["num_periods"]),
             "output_dir": files_config["output_dir"]},
            N=int(sim_config["n"]),
            J=convert_float_value(sim_config["j"]),
            B=convert_float_value(sim_config["b"]),
            T=convert_float_value(sim_config["t"]),
            phi_kick_phase=convert_float_value(sim_config["phi_kick_phase"]),
            h=convert_float_value(sim_config["h"]),
            phi_0=convert_float_value(sim_config["phi_0"]),
            freq=int(sim_config["freq"]),
        )
    else:
        params_simulation = SimulationParams({"num_periods": 40,
                                              "output_dir": "./results"})
    return params_simulation


def build_hamiltonian_lmg(size: int, J: float, B: float, T: float = 1, theta: float = np.pi,
                          epsilon_z: float = 0):
    basis = spin_basis_1d(1, S=f"{size}/2" if size % 2 == 1 else f"{size // 2}", pauli=0)
    static = [
        # -2J / N S_z^2
        ["zz", [[-2 * J / size, 0, 0]]],
        # -2 B J_z
        ["+", [[-B, 0]]],
        ["-", [[-B, 0]]],
        # \delta_z S_z
        ["z", [[epsilon_z, 0]]]
    ]

    def kick_dynamics(t, T, epsilon=0.01):
        n = round(t / T)
        return np.exp(-((t - n*T)**2) / (2 * epsilon**2)) / (epsilon * np.sqrt(2 * np.pi)) if n > 0 else 0

    # Dynamic part of the Hamiltonian
    dynamic = [
        ["+", [[-theta / 2, 0]], kick_dynamics, [T]],
        ["-", [[-theta / 2, 0]], kick_dynamics, [T]]
    ]
    return hamiltonian(static, dynamic, basis=basis, dtype=np.complex128,
                       check_symm=DEBUG, check_herm=DEBUG)


def get_observable(hamilt, which: str = 'z'):
    return quantum_operator({f'S{which}': [[which, [[1.0, 0]]]]}, basis=hamilt.basis)


def evolution_stroboscopic(hamilt, psi0, num_periods, operator, N, T: float = 1, theta: float = np.pi):
    times = Floquet_t_vec(2 * np.pi / (2 * T), num_periods, len_T=10)
    # t_list = (np.array([t.T, 2*t.T]) + np.finfo(float).eps)  # times to evaluate H
    # dt_list = np.array([t.T, t.T])  # time step durations to apply H for
    # Floq = Floquet(
    #     {"H": hamilt, "t_list": t_list, "dt_list": dt_list}, UF=True
    # )  # call Floquet class
    psi_t = hamilt.evolve(psi0, times[0], times, atol=1e-13, rtol=1e-11)
    operator_expect = np.zeros(len(times), dtype=np.float64)
    for i, t in enumerate(times):
        operator_expect[i] = np.real(operator.expt_value(psi_t[:, i])) / N

    # Plot dynamics
    plt.figure(figsize=(8, 6))
    plt.plot(times/T, operator_expect, label=r'$\langle S_z(t) \rangle / N$')
    plt.xlabel('Time')
    plt.ylabel(r'$\langle S_z(t) \rangle / N$')
    plt.title(f'LMG Model: Magnetization Dynamics')
    plt.legend()
    plt.grid(True)
    plt.show()


def run_simulation(params: SimulationParams):
    N = params.N
    J = params.J
    B = params.B
    H = build_hamiltonian_lmg(N, J, B)

    energies, vecs = build_hamiltonian_lmg(N, J, B, epsilon_z=np.exp(-5*N)).eigsh(k=2, which="SA")
    gap = abs(energies[0] - energies[1])
    # Initial state: all spins up (|↑↑...↑⟩)
    psi0 = np.zeros(vecs[:, 0].shape)
    psi0[0] = 1.0
    Sz = get_observable(H, which="z")
    evolution_stroboscopic(H, psi0, simulation_params.run_arguments["num_periods"], Sz, N)


if __name__ == "__main__":
    simulation_params = read_defaults_args_from_config()
    output_dir = Path(__file__).parent / simulation_params.run_arguments["output_dir"]
    # run_simulation(simulation_params)

