#!/usr/bin/env python3
import os
import logging
import configparser
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
import matplotlib.pyplot as plt

from quspin.tools.Floquet import Floquet_t_vec, Floquet
from quspin.tools.evolution import evolve
from quspin.operators import hamiltonian, quantum_operator
from quspin.basis import spin_basis_1d

import matplotlib as mpl

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Computer Modern"]
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,amsfonts,amssymb}"


@dataclass
class SimulationParams:
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


def read_defaults_args_from_config() -> Tuple[SimulationParams, dict]:
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
        params_simulation = SimulationParams(
            N=int(sim_config["n"]),
            J=convert_float_value(sim_config["j"]),
            B=convert_float_value(sim_config["b"]),
            T=convert_float_value(sim_config["t"]),
            phi_kick_phase=convert_float_value(sim_config["phi_kick_phase"]),
            h=convert_float_value(sim_config["h"]),
            phi_0=convert_float_value(sim_config["phi_0"]),
            freq=int(sim_config["freq"]),
        )
        files_config = config["Files"]
    else:
        params_simulation = SimulationParams()
        files_config = dict(output_dir="./results")
    return params_simulation, files_config


def build_naive_hamiltonian(size: int, J: float, B: float, T: float = 1, theta: float = np.pi):
    basis = spin_basis_1d(size, pauli=0)  # pauli=0 uses S=1/2 operators (S_x, S_y, S_z )
    static = [
        ["zz", [[-J / size, i, j] for i in range(size) for j in range(size)]],  # -J / N S_z^2
        ["x", [[-2 * B, i] for i in range(size)]],  # -2 B J_z
    ]

    def kick_dynamics(t):
        return (-1) ** (t // T)

    dynamic = [
        ["x", [[-theta, i] for i in range(size)], kick_dynamics, []],
    ]
    return hamiltonian(static, dynamic, basis=basis)


def get_observable(hamilt, N):
    Sz_op = quantum_operator({'Sz': [['z', [(1.0, i) for i in np.arange(N)]]]}, basis=hamilt.basis)
    return Sz_op


def evolution_stroboscopic(hamilt, psi0, num_periods, operator, N, T: float = 1, theta: float = np.pi):
    times = Floquet_t_vec(2 * np.pi / (2 * T), num_periods, len_T=2)
    # t_list = (np.array([t.T, 2*t.T]) + np.finfo(float).eps)  # times to evaluate H
    # dt_list = np.array([t.T, t.T])  # time step durations to apply H for
    # Floq = Floquet(
    #     {"H": hamilt, "t_list": t_list, "dt_list": dt_list}, UF=True
    # )  # call Floquet class
    psi_t = hamilt.evolve(psi0, times[0], times, atol=1e-12, rtol=1e-12)
    operator_expect = np.zeros(len(times), dtype=np.float64)
    for i, t in enumerate(times):
        operator_expect[i] = np.real(operator.expt_value(psi_t[:, i])) / N

    # Plot dynamics
    plt.figure(figsize=(8, 6))
    plt.plot(times, operator_expect, label=r'$\langle S_z(t) \rangle / N$')
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
    H = build_naive_hamiltonian(N, J, B)
    # Initial state: all spins up (|↑↑...↑⟩)
    # E1, V1 = H.eigh()
    psi0 = np.zeros(H.basis.Ns, dtype=np.complex128)
    up_state_index = H.basis.index('1' * N)  # All spins up in z-basis
    psi0[up_state_index] = 1.0
    psi0 = psi0 / np.linalg.norm(psi0)
    # psi0 = (V1[:, 0] + V1[:, 1]) / np.sqrt(2)
    Sz = get_observable(H, N)
    evolution_stroboscopic(H, psi0, 20, Sz, N)


if __name__ == "__main__":
    simulation_params, files = read_defaults_args_from_config()
    output_dir = Path(__file__).parent / files["output_dir"]
    run_simulation(simulation_params)
