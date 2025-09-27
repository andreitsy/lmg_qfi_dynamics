import mpmath as mp
import logging
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import configparser

from dataclasses import dataclass
from enum import Enum
from typing import List
from pathlib import Path
import matplotlib as mpl

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Computer Modern"]
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,amsfonts,amssymb}"

MAX_TIME_POW_PLOT = None
DEBUG = True
Y_LABEL_COORDINATE = 0.16
_logger_initialized = False


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


@dataclass
class QFIInformation:
    m_x: float
    m_y: float
    m_z: float
    qfi: float
    time: float


@dataclass
class UF:
    eigenvalues: mp.matrix
    U: mp.matrix
    U_inv: mp.matrix


def dketa_t(ket_t_p_delta, ket_t_m_delta, delta):
    return (ket_t_p_delta - ket_t_m_delta) / (2.0 * mp.mpf(delta))


def quantum_fisher_information_mp(dketa_t, ket_t):
    """
    Compute Quantum Fisher Information using mpmath matrices.

    Parameters
    ----------
    dketa_t : mp.matrix
        Derivative of the ket with respect to parameter.
    ket_t : mp.matrix
        State ket at current time.

    Returns
    -------
    qfi : mp.mpf
        Quantum Fisher Information.
    """
    # <dketa|dketa>
    term1 = (dketa_t.transpose_conj() * dketa_t)[0, 0]
    # |<ket|dketa>|^2
    overlap = (ket_t.transpose_conj() * dketa_t)[0, 0]
    term2 = mp.fabs(overlap) ** 2
    # QFI = 4 * ( <dketa|dketa> - |<ket|dketa>|^2 )
    qfi = 4 * mp.re(term1 - term2)
    return qfi


def log_message(message: str, log_level: int = logging.INFO):
    """
    This function writes a message to stdout and logs the operation.
    """
    global _logger_initialized
    if not _logger_initialized:
        _logger_initialized = True
        logging.basicConfig(
            level=log_level,  # Set the logging level
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],  # Log to stdout
        )
    logging.log(log_level, message)


def create_z_operator(n):
    """
    Create a z-operator with arbitrary precision using mpmath.

    :param n: The size parameter (matrix dimension is n+1).
    :return: A dense diagonal matrix of size (n+1, n+1).
    """
    half_n = mp.mpf(n) / 2
    diag_elements = [half_n - i for i in range(n + 1)]  # runs from +S down to -S
    Z = mp.matrix(n + 1)
    for i in range(n + 1):
        Z[i, i] = mp.mpc(diag_elements[i])
    return Z


def create_spin_minus_operators(n):
    """
    Create a spin lowering (S-) operator with arbitrary precision.

    :param n: Twice the spin quantum number (dimension = n+1).
    :return: The spin lowering operator matrix (mpmath.matrix).
    """
    Spow2 = mp.mpf(n) / 2
    # m values go from S down to S-n+1
    mtmp = [Spow2 - i for i in range(n)]
    Cminus = [mp.sqrt(Spow2 * (Spow2 + 1) - m * (m - 1)) for m in mtmp]
    Sminus = mp.matrix(n + 1)
    for i in range(n):
        Sminus[i + 1, i] = mp.mpc(Cminus[i])
    return Sminus


def create_spin_plus_operators(n):
    """
    Create a spin raising (S+) operator with arbitrary precision.

    :param n: Twice the spin quantum number (dimension = n+1).
    :return: Spin raising operator matrix (mpmath.matrix).
    """
    Sminus = create_spin_minus_operators(n)
    # Hermitian conjugate (transpose + conjugate)
    Splus = Sminus.T  # take transpose

    for i in range(n + 1):
        for j in range(n + 1):
            Splus[i, j] = mp.conj(Splus[i, j])

    return Splus


def create_spin_xyz_operators(n):
    """
    Create spin operators (Sz, Sx, Sy) with arbitrary precision.

    :param n: Twice the spin quantum number (dimension = n+1).
    :return: Tuple (Sz, Sx, Sy) as mpmath matrices.
    """
    Splus = create_spin_plus_operators(n)
    Sminus = create_spin_minus_operators(n)
    Sz = create_z_operator(n)
    Sx = (Splus + Sminus) / 2
    Sy = -mp.j * (Splus - Sminus) / 2
    return Sz, Sx, Sy


def create_hamiltonian_h0(coupling_zz, coupling_x, num_spins):
    """
    Create the Hamiltonian H0 with arbitrary precision.

    :param coupling_zz: Coupling constant for ZZ interaction.
    :param coupling_x: Coupling constant for X interaction.
    :param num_spins: Number of spins in the system.
    :return: Hamiltonian H0 as mpmath.matrix.
    """
    Sz, Sx, Sy = create_spin_xyz_operators(num_spins)
    Hzz = -mp.mpf(coupling_zz) * (2 / mp.mpf(num_spins)) * (Sz * Sz)
    Hx = -mp.mpf(coupling_x) * 2 * Sx
    return Hzz + Hx


def create_kick_operator(phi, s_x):
    """
    Create a kick operator matrix with enhanced precision.

    :param phi: A numerical value representing the angle or phase shift.
    :param s_x: Matrix representing total spin operator in the x direction
    :return: The matrix exponential of -1j * phi * s_x.
    """
    return mp.expm(-mp.j * mp.mpf(phi) * s_x)


def ac_time(S_x, S_y, S_z, omega, phi_0, t_k, theta, varphi):
    # Compute the time-dependent sinusoidal factor
    time_factor = t_k * omega + phi_0
    theta = mp.mpc(theta)
    varphi = mp.mpc(varphi)
    sinusoidal_factor = mp.sin(time_factor)
    # Calculate and return the AC field Hamiltonian
    return sinusoidal_factor * (
            mp.sin(theta) * mp.cos(varphi) * S_x +
            mp.sin(theta) * mp.sin(varphi) * S_y +
            mp.cos(theta) * S_z
    )


def create_v_operator(H_0, S_x, S_y, S_z, omega, phi_0, h, t_k, theta, varphi):
    S_alpha_part = mp.mpc(h) * ac_time(S_x, S_y, S_z, omega, phi_0, t_k, theta, varphi)
    return H_0 + S_alpha_part


def evalution_T_step(
        floquet_unitary,
        h,
        T,
        varphi,
        theta,
        phi_0,
        H_0,
        Xsum,
        Ysum,
        Zsum,
        omega,
        p,
        t_delta,
        steps_floquet_unitary,
):
    """
    Compute evolution of the Floquet unitary over one period
    """
    # Use arbitrary precision
    t_start = mp.mpf(T) * (p - 1)
    t_end = mp.mpf(T) * p
    linspace = [t_start + i * (t_end - t_start) / (steps_floquet_unitary - 1)
                for i in range(steps_floquet_unitary)]

    for t_k in linspace:
        matrix = create_v_operator(
            H_0, Xsum, Ysum, Zsum, omega, phi_0, h, t_k, theta, varphi
        )
        U_step = mp.expm(-mp.j * mp.mpf(t_delta) * matrix)
        floquet_unitary = U_step * floquet_unitary

    return floquet_unitary


def find_power_r_mpmath(floque_u: UF, r):
    if r <= 0:
        return mp.eye(len(floque_u.eigenvalues))
    return floque_u.U * mp.diag([e ** r for e in floque_u.eigenvalues]) * floque_u.U_inv


def calculate_unitary_at_time_mp(h, time: int, params: dict, H_0: mp.matrix, floque_u: UF):
    """
    Compute the Floquet unitary at a given discrete time using mpmath (arbitrary precision).
    Returns
    -------
    floquet_unitary : mp.matrix
        The unitary operator at the given time.
    """
    # Spin operators with mpmath precision
    Zsum, Xsum, Ysum = create_spin_xyz_operators(params["N"])
    # Time step within one Floquet period
    t_delta = mp.mpf(params["T"]) / params["steps_floquet_unitary"]
    # Driving frequency
    omega = mp.mpf(2) * mp.pi / mp.mpf(params["nu"] * params["T"])
    # Number of complete Floquet cycles
    r = time // params["nu"]
    extra_interval = range(r * params["nu"] + 1, time + 1)
    # Start from floquet_unitary for r full cycles
    floquet_unitary = find_power_r_mpmath(floque_u, r)
    # Evolve over leftover intervals
    for p in extra_interval:
        floquet_unitary = evalution_T_step(
            floquet_unitary,
            h,
            params["T"],
            params["varphi"],
            params["theta"],
            params["phi_0"],
            H_0,
            Xsum,
            Ysum,
            Zsum,
            omega,
            p,
            t_delta,
            params["steps_floquet_unitary"],
        )
        # Apply the kick operator
        floquet_unitary = create_kick_operator(params["phi"], Xsum) * floquet_unitary

    return floquet_unitary


def calculate_error_estimation_mp(dket_t, ket_t):
    """
    Compute the error estimation using mpmath arbitrary precision.

    Parameters
    ----------
    dket_t : mp.matrix
        Derivative ket vector.
    ket_t : mp.matrix
        State ket vector.

    Returns
    -------
    error_est : mp.mpf
        The error estimate.
    """
    # <dpsi|psi> + <psi|dpsi>
    term = (dket_t.T.apply(mp.conj)) * ket_t + (ket_t.T.apply(mp.conj)) * dket_t
    error_est = float(mp.fabs(term[0, 0]))
    return error_est


def process_time_point_mp(time: int, params: dict, H_0: mp.matrix,
                          floque_u: UF, floque_u_p: UF, floque_u_m: UF, init_state: mp.matrix,
                          Zsum: mp.matrix, Xsum: mp.matrix, Ysum: mp.matrix,
                          ):
    """
    Compute observables and QFI at a given time point using mpmath arbitrary precision.
    """
    epsilon = params["epsilon"]
    N = params["N"]
    h = params["h"]
    # Compute Floquet unitaries at different delta shifts
    floquet_unitary = calculate_unitary_at_time_mp(h, time, params, H_0, floque_u)
    floquet_unitary_p_delta = calculate_unitary_at_time_mp(h + epsilon, time, params, H_0, floque_u_p)
    floquet_unitary_m_delta = calculate_unitary_at_time_mp(h - epsilon, time, params, H_0, floque_u_m)
    # Evolve ket
    ket_t = floquet_unitary * init_state
    ket_t_p_delta = floquet_unitary_p_delta * init_state
    ket_t_m_delta = floquet_unitary_m_delta * init_state
    # Compute derivative ket for QFI
    dket_t = dketa_t(ket_t_p_delta, ket_t_m_delta, epsilon)
    qfi = quantum_fisher_information_mp(dket_t, ket_t)
    if DEBUG:
        abs_error_estimate = calculate_error_estimation_mp(dket_t, ket_t)
        if abs_error_estimate > 0.1:
            print("ABS_ERROR:", abs_error_estimate)

    # Compute magnetizations
    m_x = mp.re((ket_t.transpose_conj() * Xsum * ket_t)[0, 0]) / N
    m_y = mp.re((ket_t.transpose_conj() * Ysum * ket_t)[0, 0]) / N
    m_z = mp.re((ket_t.transpose_conj() * Zsum * ket_t)[0, 0]) / N

    return QFIInformation(
        qfi=float(qfi),
        time=time,
        m_x=float(m_x),
        m_y=float(m_y),
        m_z=float(m_z),
    )


def calculate_unitary_T(
        h: mp.mpf,
        params: dict,
        H_0: mp.matrix,
):
    n = params["N"]
    steps_floquet_unitary = params["steps_floquet_unitary"]
    T = params["T"]
    nu = params["nu"]
    phi = params["phi"]
    varphi = params["varphi"]
    theta = params["theta"]
    phi_0 = params["phi_0"]
    Zsum, Xsum, Ysum = create_spin_xyz_operators(n)
    t_delta = mp.mpf(T / steps_floquet_unitary)
    omega = mp.mpf(2.0) * mp.pi / (nu * T)
    floquet_unitary = mp.eye(H_0.rows)

    for p in range(1, nu + 1):  # nu \tau = T
        floquet_unitary = evalution_T_step(
            floquet_unitary,
            h,
            T,
            varphi,
            theta,
            phi_0,
            H_0,
            Xsum,
            Ysum,
            Zsum,
            omega,
            p,
            t_delta,
            steps_floquet_unitary,
        )
        floquet_unitary = create_kick_operator(phi, Xsum) * floquet_unitary
    return floquet_unitary


def simulation_with_AC_field_mp(params: dict, time_interval, init_state) -> List[QFIInformation]:
    """
    Sequential observable simulation using mpmath arbitrary precision
    """
    h = params["h"]
    epsilon = params["epsilon"]
    Zsum, Xsum, Ysum = create_spin_xyz_operators(params["N"])
    H_0 = create_hamiltonian_h0(params["J"], params["B"], params["N"])

    fu_eigenvalues, fu_eigenvectors = mp.eig(calculate_unitary_T(h, params, H_0))
    floque_u = UF(eigenvalues=fu_eigenvalues, U=fu_eigenvectors, U_inv=mp.inverse(fu_eigenvectors))

    fu_delta_p_eigenvalues, fu_delta_p_eigenvectors = mp.eig(calculate_unitary_T(h + epsilon, params, H_0))
    floque_u_p = UF(eigenvalues=fu_delta_p_eigenvalues,
                    U=fu_delta_p_eigenvectors, U_inv=mp.inverse(fu_delta_p_eigenvectors))

    fu_delta_m_eigenvalues, fu_delta_m_eigenvectors = mp.eig(calculate_unitary_T(h - epsilon, params, H_0))
    floque_u_m = UF(eigenvalues=fu_delta_m_eigenvalues,
                    U=fu_delta_m_eigenvectors, U_inv=mp.inverse(fu_delta_m_eigenvectors))

    # Sequentially process each time point
    results = []
    for time in time_interval:
        res = process_time_point_mp(
            time,
            params,
            H_0,
            floque_u,
            floque_u_p,
            floque_u_m,
            init_state,
            Zsum,
            Xsum,
            Ysum,
        )
        results.append(res)

    return results


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
             "output_dir": files_config["output-dir"]},
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
        params_simulation = SimulationParams({"num_periods": 40,
                                              "output_dir": "./results"})
    return params_simulation


def run_gaps():
    N_values = [1, 5, 10, 20, 30, 50, 100]
    J = 1.0
    B = 0.1
    gaps = []
    dps = 50
    for N in N_values:
        with mp.workdps(dps):
            H = create_hamiltonian_h0(J, B, N)
            energies, evecs = mp.eigh(H)
            evals_sorted = sorted(energies, key=lambda ev: mp.re(ev))
            if DEBUG:
                print("N=", N, "gap=", evals_sorted[0] - evals_sorted[1])
            gap = -mp.log(mp.fabs(evals_sorted[0] - evals_sorted[1]))
            gaps.append(float(gap))
    # Create the plot

    plt.figure(figsize=(8, 6))
    plt.scatter(N_values, gaps, color='teal', s=100, label='Energy Gap')
    plt.plot(N_values, gaps, color='teal', linestyle='--', alpha=0.5)  # Optional connecting line
    plt.xlabel('N (Number of Spins)')
    plt.ylabel('Energy Gap (log scale)')
    plt.title('-log(Delta energy) vs N (LMG Model)')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def generate_time_interval(num_points: int, max_degree: int) -> list:
    if max_degree <= 1:
        raise ValueError("max_degree should be greater than 1!")
    time_interval = (list(range(1, 10)) +
                     [int(x) for x in np.logspace(1, max_degree, num_points, endpoint=True)])
    return time_interval


def run_simulation(params: SimulationParams):
    dps = params.run_arguments["dps"]
    num_points = params.run_arguments["num_points"]
    params_dict = dict(phi=params.phi_kick_phase, J=params.J, B=params.B,
                       N=params.N, T=params.T, h=params.h, nu=params.freq,
                       varphi=params.varphi, phi_0=params.phi_0, theta=params.theta,
                       epsilon=mp.mpf(f"1e-{dps // 2}"),
                       steps_floquet_unitary=params.run_arguments["steps_floquet_unitary"])
    init_states = [
        InitialState.PHYS,
        InitialState.GS_PHYS,
        InitialState.GS_CAT,
        InitialState.CAT_SUM,
    ]

    with mp.workdps(dps):
        vec_size = params.N + 1
        H = create_hamiltonian_h0(params.J, params.B, params.N)
        energies, evecs = mp.eigh(H)
        # Ground states and initial ket
        sorted_indices = sorted(range(len(energies)), key=lambda i: mp.re(energies[i]))
        gap = energies[sorted_indices[1]] - energies[sorted_indices[0]]
        last_time_degree = int(mp.log10(mp.fabs(4 * mp.pi / gap))) + 1
        ground_state = evecs[:, sorted_indices[0]]
        first_excited_state = evecs[:, sorted_indices[1]]
        results = dict()
        for state in init_states:
            if state == InitialState.GS_PHYS:
                init_state = (ground_state + first_excited_state) / np.sqrt(2)
            elif state == InitialState.GS_CAT:
                init_state = ground_state
            elif state == InitialState.PHYS:
                # sum only for up states
                init_state = mp.zeros(vec_size, 1)
                init_state[0] = mp.mpf('1.0')
            elif state == InitialState.CAT_SUM:
                # superposition of up and down states
                init_state = mp.zeros(vec_size, 1)
                init_state[0] = mp.mpf('1.0') / mp.sqrt(2)
                init_state[-1] = mp.mpf('1.0') / mp.sqrt(2)
            else:
                raise ValueError(f"Unhandled initial state type: {init_state}")

            sim_results = simulation_with_AC_field_mp(
                params=params_dict,
                time_interval=generate_time_interval(num_points, last_time_degree),
                init_state=init_state)
            results[state] = sim_results
    return results


def save_to_file_qfi_dynamics(
        results: list,
        output_file: Path,
):
    # Create a dictionary to store data for CSV
    csv_data = {
        "time": [],
        "m_x": [],
        "m_y": [],
        "m_z": [],
        "qfi": [],
    }
    for result in results:
        csv_data["time"].append(result.time)
        csv_data["m_x"].append(result.m_x)
        csv_data["m_y"].append(result.m_y)
        csv_data["m_z"].append(result.m_z)
        csv_data["qfi"].append(result.m_z)

    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False)


def plot_qfi_data_subplot(ax, simulations, simulation_params, max_time_pow=None):
    """Plot QFI values onto the provided subplot axes (ax)."""
    # Loop through different initial state groups in QFI data
    last_time = -1
    for state in simulations:
        time_points = [s.time for s in simulations[state]]
        last_time = max(time_points[-1], last_time)
        qfi_values = [abs(s.qfi) / (simulation_params.N * s.time) ** 2 if s.qfi > 0 else 0 for s in simulations[state]]
        # Labels for each state
        if state == "GS_PHYS":
            label = (r"$\left(\big|E_1 {\bigr \rangle} + "
                     r"\big|E_{\overline{1}} {\bigr \rangle} \right) / \sqrt{2}$")
        elif state == "GS_CAT":
            label = r"$\big|E_1 {\bigr \rangle} $"
        elif state == "PHYS":
            label = r"$\big|\uparrow ... \uparrow {\bigr \rangle}$"
        elif state == "CAT_SUM":
            label = (r"$\left(\big|\uparrow ... \uparrow {\bigr \rangle} + "
                     r"\big|\downarrow ... \downarrow {\bigr \rangle} "
                     r"\right)/\sqrt{2}$")
        else:
            label = state
        # Plot QFI data
        ax.plot(time_points, qfi_values, "-", label=label, linewidth=3)
    # Customize QFI subplot
    ax.set_title(
        rf"QFI dynamics for $N={simulation_params.N}, B/J={float(simulation_params.B / simulation_params.J):.2f}$",
        fontsize=40)
    ax.set_xlabel(r"$t / T$", fontsize=40)
    ax.set_ylabel(r"$F_h / (N t)^2$", fontsize=40)
    ax.set_xscale("log")  # Logarithmic scale for time
    ax.set_ylim([0, np.abs((1 - float(simulation_params.B) ** 2) * 4 / np.pi ** 2)])
    if max_time_pow is None:
        max_time_pow = int(np.log10(last_time)) + 1
    ax.set_xticks([10 ** n for n in range(1, max_time_pow, 2)])
    ax.tick_params(axis="x", labelsize=30)  # Adjust x-axis tick label font size
    ax.tick_params(axis="y", labelsize=30)  # Adjust y-axis tick label font size
    ax.grid(True, linestyle="--", alpha=0.6, linewidth=1.7)
    H = create_hamiltonian_h0(simulation_params.J, simulation_params.B, simulation_params.N)
    symmetry_edge = -1 * float(simulation_params.B) * simulation_params.N
    eigvals_H_real, evecs = mp.eigh(H)
    # Add vertical lines for gap points
    pairs = {}
    for idx, eigenvalue_mp in enumerate(eigvals_H_real):
        eigenvalue = float(eigenvalue_mp)
        pair_num = idx // 2
        if pair_num not in pairs:
            pairs[pair_num] = []
        pairs[pair_num].append(float(eigenvalue))
    for pair_idx, energies in pairs.items():
        if energies[0] <= symmetry_edge and pair_idx < 7: # no more than 7
            if len(energies) == 2:
                gap = abs(energies[1] - energies[0])
                ax.axvline(x=2 * np.pi / gap, color='black',
                           linestyle='--', alpha=0.7, linewidth=2)
                # Adding the annotation with value 1/Δ
                ax.annotate(
                    f"$\\frac{{2\\pi}}{{\\Delta_{{{pair_idx+1}\\overline{{{pair_idx+1}}}}}}}$",
                    xy=(2 * np.pi / gap, Y_LABEL_COORDINATE),
                    xytext=(2 * np.pi / gap * 1.03, Y_LABEL_COORDINATE),
                    fontsize=54,
                    color='black'
                )

def plot(simulations: dict, simulation_params: SimulationParams, results_dir: Path):
    fig, ax = plt.subplots(figsize=(15, 10))
    plot_qfi_data_subplot(ax, simulations, simulation_params, max_time_pow=MAX_TIME_POW_PLOT)
    plt.tight_layout()
    plt.savefig(
        results_dir / f"qfi_dynamics_N={simulation_params.N}_B={float(simulation_params.B):.2f}.png", dpi=300)

if __name__ == "__main__":
    simulation_params = read_defaults_args_from_config()
    output_dir = Path(__file__).parent / simulation_params.run_arguments["output_dir"]
    simulations = run_simulation(simulation_params)

    for state, results in simulations.items():
        output_file_name = f"{state}_N={simulation_params.N}_B={float(simulation_params.B):.2f}.csv"
        save_to_file_qfi_dynamics(results=results, output_file=output_dir / output_file_name)
    plot(simulations, simulation_params, output_dir)
