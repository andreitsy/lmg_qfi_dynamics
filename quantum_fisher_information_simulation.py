#!/usr/bin/env python3
import pandas as pd
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import configparser

from typing import Optional, Union
from numpy import typing
from enum import Enum
from itertools import chain
from scipy.linalg import eigh, expm, eigvalsh_tridiagonal
from dataclasses import dataclass
from pathlib import Path
import mpmath as mp

# Constants for state coefficients
MP_MATH_PRECISION = 100
REL_ERROR_QFI = 10.0
_logger_initialized = False
results_dir = Path(__file__).parent / "results"


class InitialState(Enum):
    """Enumeration of possible initial quantum states."""

    GS_PHYS = "GS_phys"
    GS_CAT = "GS_cat"
    CAT_SUM = "CatSum"
    PHYS = "Phys"


@dataclass
class QFIAnalyticalInformation:
    m_0: float
    m_1: float
    m_x: float
    m_y: float
    m_z: float
    qfi: float
    time: float
    N: int
    params: dict


def dketa_t(ket_t, ket_t_m_delta, ket_t_m2_delta, ket_t_p_delta, ket_t_p2_delta, delta):
    # three-point central difference formula
    return (-ket_t_p2_delta + 8*ket_t_p_delta - 8*ket_t_m_delta + ket_t_m2_delta) / (12.0 * delta)


def quantum_fisher_information(dketa_t, ket_t):
    return 4 * np.real(
        (dketa_t.T.conj()).dot(dketa_t)
        - np.abs((ket_t.T.conj()).dot(dketa_t)) ** 2
    )


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


def plot_dynamics(
    time_points: list[np.ndarray],
    observable_values: list[np.ndarray],
    title: str = "Observable Dynamics",
    xlabel: str = "Time",
    ylabel: str = "Observable",
):
    """
    Plot the results of observable dynamics over time.
    :param result: list of numpy.array containing time points and
                   corresponding observable values.
    :param title: Title of the plot (default is "Observable Dynamics").
    :param xlabel: Label for the x-axis (default is "Time").
    :param ylabel: Label for the y-axis (default is "Observable").
    """
    MARKER_SIZE = 2
    plt.figure(figsize=(10, 5))
    plt.plot(
        time_points,
        observable_values,
        "*-",
        markersize=MARKER_SIZE,
    )
    ax = plt.gca()
    ax.set_ylim(0, 0.5)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.grid(True)
    plt.show()


def create_z_operator(n, precision="np"):
    """
    Create a z-operator with enhanced precision.
    
    :param n: The size parameter used to generate the z-operator (matrix dimension is n+1).
    :param precision: String specifying the desired precision:
                      - "float128": Higher precision provided by NumPy
                      - "mpmath": Arbitrary precision using mpmath
                      - Others: Fallback to default NumPy precision
    :return: A dense diagonal matrix of size (n+1, n+1) with specific values.
    """
    if precision == "np":
        half_n = n / 2
        # Create the diagonal elements: they run from half_n down
        # to -half_n in steps of -1
        diag_elements = np.arange(half_n, -half_n - 1, -1, dtype=np.complex128)
        # Return the dense diagonal matrix
        return np.diag(diag_elements)

    elif precision == "mpmath":
        # Set desired precision (adjustable)
        mp.dps = MP_MATH_PRECISION  # Decimal places of precision (e.g., 50 decimal digits)
        # Compute elements with arbitrary precision
        half_n = mp.mpf(n) / 2
        diag_elements = [half_n - mp.mpf(i) for i in range(n + 1)]
        # Create diagonal matrix using mpmath
        return mp.matrix(mp.diag(diag_elements))
    else:
        # Fallback to default NumPy precision with complex128
        half_n = n / 2
        diag_elements = np.arange(half_n, -half_n - 1, -1, dtype=np.complex128)
        return np.diag(diag_elements)


def create_spin_minus_operators(n, precision="np"):
    """
    Create a spin lowering (minus) operator with enhanced precision.

    :param n: An integer representing twice the spin quantum number
              (i.e., matrix dimension is n+1),
              which implies the spin value is n/2.
    :param precision: String specifying the desired precision:
                      - "float128": Higher precision provided by NumPy
                      - "mpmath": Arbitrary precision using mpmath
    :return: A dense matrix representing the spin lowering (minus) operator with the specified precision.
    """
    if precision == "np":
        # Use NumPy float128 for higher precision
        Spow2 = n / 2
        # mtmp: m values from Spow2 down to (Spow2 - (n-1))
        mtmp = Spow2 - np.arange(0, n, dtype=np.complex128)
        # Compute the Cminus coefficients
        Cminus = np.sqrt(Spow2 * (Spow2 + 1) - mtmp * (mtmp - 1))
        # Initialize a dense (n+1)x(n+1) zero matrix with float256 type
        Sminus = np.zeros((n + 1, n + 1), dtype=np.complex128)
        # Fill the sub-diagonal with the computed coefficients
        for i in range(n):
            Sminus[i + 1, i] = Cminus[i]
        return Sminus

    elif precision == "mpmath":
        # Set the desired mpmath precision (adjustable)
        mp.dps = MP_MATH_PRECISION  # Set decimal precision (e.g., 50 decimal places)
        Spow2 = mp.mpf(n) / 2
        mtmp = [Spow2 - mp.mpf(i) for i in range(n)]
        Cminus = [mp.sqrt(Spow2 * (Spow2 + 1) - m * (m - 1)) for m in mtmp]

        # Create the spin lowering operator using an mpmath matrix
        Sminus = mp.matrix(n + 1, n + 1)
        for i in range(n):
            Sminus[i + 1, i] = Cminus[i]
        return Sminus

    else:
        raise ValueError("Unsupported precision: Choose 'float128' or 'mpmath'")


def create_spin_plus_operators(n, precision="np"):
    """
    :param n: The dimension or size for the spin operators.
    :return: Transpose and complex conjugate of the spin-minus operators,
             resulting in the spin-plus operators.
    """
    return create_spin_minus_operators(n, precision=precision).T.conjugate()


def create_spin_xyz_operators(n, precision="np"):
    """
    :param n: The size of the system for which the spin
              operators are being created.
    :return: A tuple containing the Z, X, and Y spin operators for the system.
    """
    Splus = create_spin_plus_operators(n, precision=precision)
    Sminus = create_spin_minus_operators(n, precision=precision)
    Zsum = create_z_operator(n, precision=precision)
    Xsum = (Splus + Sminus) / 2
    Ysum = -1j * (Splus - Sminus) / 2
    return Zsum, Xsum, Ysum


def create_hamiltonian_h0(coupling_zz, coupling_x, num_spins, precision="np"):
    """
    Create the Hamiltonian H0 with the given coupling constants and spin size.

    :param coupling_zz: Coupling constant for the ZZ interaction.
    :param coupling_x: Coupling constant for the X interaction.
    :param num_spins: Number of spins in the system.
    :param precision: Precision type - "np" (NumPy) or "mpmath" (multi-precision).
    :return: The Hamiltonian matrix H0.
    """

    # Helper functions
    def calculate_hzz(Zsum, factor):
        """Calculate the H_zz term."""
        return factor * (Zsum.dot(Zsum) if precision == "np" else Zsum * Zsum)

    def calculate_hx(Xsum, factor):
        """Calculate the H_x term."""
        return factor * Xsum

    # Constants for precision
    precision_factor = np.complex128 if precision == "np" else mp.mpf
    if precision not in {"np", "mpmath"}:
        raise ValueError(f"Unsupported precision type '{precision}'. Use 'np' or 'mpmath'.")

    # Create spin operators
    Zsum, Xsum, Ysum = create_spin_xyz_operators(num_spins, precision=precision)

    # Calculate Hamiltonian terms
    hzz_factor = precision_factor(-coupling_zz * (2 / num_spins))
    hx_factor = precision_factor(-coupling_x * 2)
    Hzz = calculate_hzz(Zsum, hzz_factor)
    Hx = calculate_hx(Xsum, hx_factor)

    # Return combined Hamiltonian
    return Hzz + Hx


def get_initial_states(H, Zsum, ranks=None, delta_h_z=None, precision="np"):
    """
    :param H: The input Hamiltonian matrix.
    :param Zsum: The matrix to be added to H
                 for modifying eigenvalue degeneracy.
    :param ranks: The indexes of the eigenvector to return, corresponding
                  to the n-th smallest eigenvalue.
    :return: The eigenvector of the modified matrix Ho corresponding
             to the n-th smallest eigenvalue.
    """
    ranks = ranks if ranks is not None else range(H.shape[1])
    if precision == "np":
        if delta_h_z is None:
            # Compute smallest k eigenvalues and eigenvectors of H
            eigvals_H = eigh(H, eigvals_only=True)
            # Extract the real parts of the eigenvalues
            eig_H = np.real(eigvals_H)
            # Sort the eigenvalues (and corresponding eigenvectors) by value
            sorted_indices = np.argsort(eig_H)
            eig_H_sorted = eig_H[sorted_indices]
            # Calculate the gap between the smallest and second smallest eigenvalue
            delta_h_z = abs(
                eig_H_sorted[1] - eig_H_sorted[0]
            )  # very small value due to degeneracy !
            # Modify the matrix H by adding gap * Zsum
        Ho = H + delta_h_z * Zsum
        # Compute smallest k eigenvalues and eigenvectors of the modified matrix Ho
        eigvals_Ho, eigvecs_Ho = eigh(Ho, check_finite=False, eigvals_only=False)
        # Extract the real parts of the eigenvalues from the modified matrix
        eig_Ho = np.real(eigvals_Ho)
        # Sort the eigenvalues (and corresponding eigenvectors) by value
        sorted_indices_Ho = np.argsort(eig_Ho)
        eigvecs_Ho_sorted = [eigvecs_Ho[:, i] for i in sorted_indices_Ho]
        # Return the eigenvector corresponding to the n-th smallest eigenvalue
    else:
        if delta_h_z:
            Ho = H + delta_h_z * Zsum
        else:
            Ho = H
        eigvals_mp, eigvecs_mp = mp.eig(Ho)
        # Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigvals_mp)  # Indices that sort eigenvalues
        eigvecs_Ho_sorted = [eigvecs_mp[:, i] for i in sorted_indices]
    vectors_result = []
    for n_ in ranks:
        if n_ < 0 or n_ >= H.shape[1]:
            raise ValueError(
                f"Rank {n_} is out of bounds" f"for the computed eigenvalues."
            )
        vectors_result.append(eigvecs_Ho_sorted[:, n_])
    return vectors_result


def create_kick_operator(phi, s_x, precision="np"):
    """
    Create a kick operator matrix with enhanced precision.

    :param phi: A numerical value representing the angle or phase shift.
    :param s_x: Matrix representing total spin operator in the x direction.
    :param precision: String specifying the desired precision:
                      - "float128": Higher precision using NumPy
                      - "mpmath": Arbitrary precision using mpmath
                      - Default: Complex128 precision
    :return: The matrix exponential of -1j * phi * s_x.
    """
    if precision == "np":
        # Use NumPy float128 for higher precision
        phi = np.complex128(phi)
        matrix = -1j * phi * s_x.astype(np.complex128)
        return expm(matrix)
    elif precision == "mpmath":
        # Set desired precision (adjustable)
        mp.dps = MP_MATH_PRECISION  # Set to 50 decimal places or desired precision
        # Convert inputs to mpmath precision
        phi_mp = mp.mpf(phi)
        # Compute the matrix exponential
        matrix_mp = -1j * phi_mp * s_x
        return mp.expm(matrix_mp)
    else:
        raise ValueError("Unsupported precision type. Use 'np' or 'mpmath'.")


def h_ac(S_x, S_y, S_z, omega, phi_0, t_k, theta, varphi, precision="np"):
    def _compute_constants(theta, varphi, precision):
        if precision == "np":
            theta = np.complex128(theta)
            varphi = np.complex128(varphi)
            return np.sin(theta), np.cos(theta), np.sin(varphi), np.cos(varphi)
        elif precision == "mpmath":
            theta = mp.mpc(theta)
            varphi = mp.mpc(varphi)
            return mp.sin(theta), mp.cos(theta), mp.sin(varphi), mp.cos(varphi)
        else:
            raise ValueError(f"Unsupported precision type: {precision}")

    # Compute the time-dependent sinusoidal factor
    time_factor = t_k * omega + phi_0
    sinusoidal_factor = np.sin(time_factor) if precision == "np" else mp.sin(time_factor)

    # Compute constants based on precision
    sin_theta, cos_theta, sin_varphi, cos_varphi = _compute_constants(theta, varphi, precision)

    # Calculate and return the AC field Hamiltonian
    return sinusoidal_factor * (
        sin_theta * cos_varphi * S_x +
        sin_theta * sin_varphi * S_y +
        cos_theta * S_z
    )


def create_v_operator(H_0, S_x, S_y, S_z, omega, phi_0, A, t_k, theta, varphi,
                      precision="np"):
    S_alpha_part = A * h_ac(S_x, S_y, S_z, omega, phi_0, t_k, theta, varphi,
                            precision=precision)
    return H_0 + S_alpha_part


def calculate_error_estimatation(dket_t, ket_t):
    """
    :param dket_t: \\psi' vector obtained numerically
    :param ket_t: \\psi vector
    :return:
    """
    return 10.0 * np.real(
        (dket_t.T.conj()).dot(ket_t) + (ket_t.T.conj()).dot(dket_t)
    )


def evalution_tau_step(
    floquet_unitary,
    A,
    tau,
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
    precision="np",
):
    """
    Compute the evolution of the Floquet unitary operator over a single interval based on
    a given precision type, system parameters, and specific time step dynamics. The evolution
    is computed either using NumPy for standard precision or using mpmath for high precision.

    Parameters
    ----------
    floquet_unitary : array_like
        Initial Floquet unitary matrix representing the state of the system.

    A : float
        Driving amplitude parameter used in the time evolution operator.

    tau : float
        Duration of one Floquet period.

    varphi : float
        Phase parameter of the driving field.

    theta : float
        Angle parameter influencing the evolution operator.

    phi_0 : float
        Initial phase for the system evolution.

    H_0 : array_like
        Static Hamiltonian of the system.

    Xsum : array_like
        Operator representing the sum of contributions in the X direction.

    Ysum : array_like
        Operator representing the sum of contributions in the Y direction.

    Zsum : array_like
        Operator representing the sum of contributions in the Z direction.

    omega : float
        Driving frequency of the system.

    p : int
        Index of the current Floquet period.

    t_delta : float
        Time step size for intermediate computations of the evolution.

    steps_floquet_unitary : int
        Number of subdivisions (steps) within the current time interval (tau).

    precision : str, optional
        Precision type for computations. Use "np" for standard NumPy precision or
        "mp" for high precision calculations with mpmath. Default is "np".

    Returns
    -------
    array_like
        The Floquet unitary matrix updated after the time evolution over the specified
        interval.

    Raises
    ------
    ValueError
        If an unsupported precision type is specified (neither "np" nor "mp").
    """
    linspace = np.linspace(
        tau * (p - 1),
        tau * p,
        steps_floquet_unitary,
        endpoint=True,
        dtype=np.complex128,
    )
    for t_k in linspace:
        matrix = create_v_operator(
            H_0,
            Xsum,
            Ysum,
            Zsum,
            omega,
            phi_0,
            A,
            t_k,
            theta,
            varphi,
            precision="np",
        )
        floquet_unitary = expm(
            -1j * t_delta * matrix
        ).dot(floquet_unitary)
    return floquet_unitary


def calculate_unitary_T(
    A,
    n,
    phi,
    tau,
    varphi,
    theta,
    phi_0,
    nu,
    steps_floquet_unitary,
    H_0,
    precision="np"
):
    Zsum, Xsum, Ysum = create_spin_xyz_operators(n, precision=precision)
    t_delta = tau / steps_floquet_unitary if precision == "np" else mp.mpf(tau / steps_floquet_unitary)
    omega = 2.0 * np.pi / (nu * tau) if precision == "np" else mp.mpf(2.0 * mp.pi / (nu * tau))
    floquet_unitary = np.eye(*H_0.shape, dtype=np.complex128) if precision == "np" else mp.eye(*H_0.shape)

    for p in range(1, nu + 1):  # nu \tau = T
        floquet_unitary = evalution_tau_step(
            floquet_unitary,
            A,
            tau,
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
            precision=precision,
        )
        floquet_unitary = (
            create_kick_operator(phi, Xsum).dot(floquet_unitary) if precision == "np" else
            create_kick_operator(phi, Xsum, precision="mpmath") * (floquet_unitary))
    return floquet_unitary


def find_power_r(eigenvalues, eigenvectors, r):
    if r <= 0:
        return np.eye(len(eigenvalues), len(eigenvalues), dtype=np.complex128)
    Lambda = np.diag(np.power(eigenvalues, r))
    U = eigenvectors
    U_inv = np.linalg.inv(U)
    return U @ Lambda @ U_inv


def find_power_r_mpmath(eigenvalues, eigenvectors, r):
    mp.dps = MP_MATH_PRECISION  # Set the decimal precision
    if r <= 0:
        return mp.eye(len(eigenvalues))
    # Convert eigenvalues to mpmath's mp type
    eigenvalues_mp = [mp.mpc(val) for val in eigenvalues]
    # Compute the diagonal matrix of eigenvalues raised to the power r
    Lambda = mp.matrix([[mp.power(eigenvalues_mp[i], r) if i == j else 0
                         for j in range(len(eigenvalues))]
                        for i in range(len(eigenvalues))])
    # Convert eigenvectors to mpmath's matrix
    U = mp.matrix(eigenvectors)
    # Compute the inverse of U using mpmath
    U_inv = U ** -1
    # Perform the matrix multiplication with mpmath
    result_mp = U * Lambda * U_inv
    return result_mp


def calculate_unitary_at_time(
    A,
    time,
    n,
    phi,
    tau,
    varphi,
    theta,
    phi_0,
    nu,
    steps_floquet_unitary,
    H_0,
    eigenvalues,
    eigenvectors,
    precision="np"
):
    Zsum, Xsum, Ysum = create_spin_xyz_operators(n, precision=precision)
    t_delta = tau / steps_floquet_unitary
    omega = 2.0 * np.pi / (nu * tau) if precision == "np" else mp.mpf(2.0 * mp.pi / (nu * tau))

    r = time // nu
    extra_interval = range(r * nu + 1, time + 1)
    floquet_unitary = (find_power_r(eigenvalues, eigenvectors, r)
                       if precision == "np" else find_power_r_mpmath(eigenvalues, eigenvectors, r))

    for p in extra_interval:
        floquet_unitary = evalution_tau_step(
            floquet_unitary,
            A,
            tau,
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
            precision=precision,
        )
        if precision == "np":
            floquet_unitary = create_kick_operator(phi, Xsum).dot(floquet_unitary)
        else:
            floquet_unitary = create_kick_operator(phi, Xsum, precision="mpmath") * floquet_unitary
    return floquet_unitary


def process_time_point(
        time,
        A,
        n,
        J,
        phi,
        h,
        tau,
        varphi,
        theta,
        phi_0,
        nu,
        epsilon,
        steps_floquet_unitary,
        H_0,
        fu_eigenvalues,
        fu_eigenvectors,
        fu_delta_m_eigenvalues,
        fu_delta_m_eigenvectors,
        fu_delta_m2_eigenvalues,
        fu_delta_m2_eigenvectors,
        fu_delta_p_eigenvalues,
        fu_delta_p_eigenvectors,
        fu_delta_p2_eigenvalues,
        fu_delta_p2_eigenvectors,
        ket_0,
        gs_mins_0,
        gs_plus_0,
        Zsum,
        Xsum,
        Ysum,
        precision="np"
):
    floquet_unitary = calculate_unitary_at_time(
        A,
        time,
        n,
        phi,
        tau,
        varphi,
        theta,
        phi_0,
        nu,
        steps_floquet_unitary,
        H_0,
        fu_eigenvalues,
        fu_eigenvectors,
    )
    floquet_unitary_m_delta = calculate_unitary_at_time(
        A - epsilon,
        time,
        n,
        phi,
        tau,
        varphi,
        theta,
        phi_0,
        nu,
        steps_floquet_unitary,
        H_0,
        fu_delta_m_eigenvalues,
        fu_delta_m_eigenvectors,
    )
    floquet_unitary_m2_delta = calculate_unitary_at_time(
        A - 2*epsilon,
        time,
        n,
        phi,
        tau,
        varphi,
        theta,
        phi_0,
        nu,
        steps_floquet_unitary,
        H_0,
        fu_delta_m2_eigenvalues,
        fu_delta_m2_eigenvectors,
    )
    floquet_unitary_p_delta = calculate_unitary_at_time(
        A + epsilon,
        time,
        n,
        phi,
        tau,
        varphi,
        theta,
        phi_0,
        nu,
        steps_floquet_unitary,
        H_0,
        fu_delta_p_eigenvalues,
        fu_delta_p_eigenvectors,
    )
    floquet_unitary_p2_delta = calculate_unitary_at_time(
        A + 2 * epsilon,
        time,
        n,
        phi,
        tau,
        varphi,
        theta,
        phi_0,
        nu,
        steps_floquet_unitary,
        H_0,
        fu_delta_p2_eigenvalues,
        fu_delta_p2_eigenvectors,
    )
    ket_t = floquet_unitary.dot(ket_0)
    ket_t_m_delta = floquet_unitary_m_delta.dot(ket_0)
    ket_t_m2_delta = floquet_unitary_m2_delta.dot(ket_0)
    ket_t_p_delta = floquet_unitary_p_delta.dot(ket_0)
    ket_t_p2_delta = floquet_unitary_p2_delta.dot(ket_0)
    dket_t = dketa_t(ket_t, ket_t_m_delta, ket_t_m2_delta, ket_t_p_delta, ket_t_p2_delta, epsilon)
    abs_error_estimate = calculate_error_estimatation(dket_t, ket_t)
    qfi = quantum_fisher_information(dket_t, ket_t)
    rel_error = abs(abs_error_estimate / qfi) if qfi > 0.0 else 0.0
    if rel_error > REL_ERROR_QFI:
        log_message(
            f"Warning: error estimate {rel_error} is too large "
            f"for time point p={time} for nu={nu}"
        )
    m_x = np.real((ket_t.T.conj()).dot(Xsum.dot(ket_t))) / n
    m_y = np.real((ket_t.T.conj()).dot(Ysum.dot(ket_t))) / n
    m_z = np.real((ket_t.T.conj()).dot(Zsum.dot(ket_t))) / n

    gs_mins = floquet_unitary.dot(gs_mins_0)
    gs_plus = floquet_unitary.dot(gs_plus_0)

    m0_t = np.real((gs_mins.T.conj()).dot(Zsum.dot(gs_plus))) / n
    m1_t = np.real((gs_mins.T.conj()).dot(Zsum.dot(gs_mins))) / n
    return QFIAnalyticalInformation(
        m_0=float(m0_t.item()),
        m_1=float(m1_t.item()),
        qfi=float(qfi.item()),
        time=time,
        N=n,
        m_x=float(m_x.item()),
        m_y=float(m_y.item()),
        m_z=float(m_z.item()),
        params=dict(
            h=h,
            A=A,
            J=J,
            phi=phi,
            tau=tau,
            varphi=varphi,
            theta=theta,
            phi_0=phi_0,
            nu=nu,
            steps_floquet_unitary=steps_floquet_unitary,
        ),
    )


def identify_state(
    init_state: Union[str, InitialState],
    hamiltonian: typing.NDArray,
    z_operator: typing.NDArray,
    phi_mix: Optional[float] = None,
    number_of_pairs: Optional[int] = None,
) -> typing.NDArray:
    """
    Identify and construct the initial quantum state based on input parameters.

    Args:
        init_state: Initial state identifier, either as string or InitialState enum
        hamiltonian: System Hamiltonian matrix
        z_operator: Z-direction spin operator matrix
        phi_mix: Optional mixing angle for superposition states (in radians)

    Returns:
        NDArray: Normalized quantum state vector

    Raises:
        ValueError: If init_state is not recognized or parameters are invalid
    """
    # Input validation
    if not isinstance(hamiltonian, np.ndarray) or not isinstance(
        z_operator, np.ndarray
    ):
        raise ValueError("Hamiltonian and z_operator must be numpy arrays")

    # Convert string to enum if needed
    if isinstance(init_state, str):
        try:
            init_state = InitialState(init_state)
        except ValueError:
            raise ValueError(f"Unknown initial state {init_state}")

    # Get ground and first excited states
    eigenstates = get_initial_states_pos_neg(hamiltonian)
    ground_state, excited_state = eigenstates[0], eigenstates[1]

    # Construct state based on type
    if phi_mix is not None:
        state = np.cos(phi_mix) * ground_state + np.sin(phi_mix) * excited_state
    elif init_state == InitialState.GS_PHYS:
        state = (ground_state + excited_state) / np.sqrt(2)
    elif init_state == InitialState.GS_CAT:
        state = ground_state
    elif init_state == InitialState.PHYS:
        # sum only for up states
        state = np.zeros(ground_state.shape)
        state[0] = 1.0
    elif init_state == InitialState.CAT_SUM:
        state = np.zeros(ground_state.shape)
        state[0] = 1.0 / np.sqrt(2)
        state[-1] = 1.0 / np.sqrt(2)
    else:
        raise ValueError(f"Unhandled initial state type: {init_state}")
    # Normalize and return state
    return state


def observarble_simulation_optimal_with_AC_field(
    A,
    n,
    J,
    phi,
    h,
    tau,
    varphi,
    theta,
    phi_0,
    nu,
    epsilon=1e-14,
    time_interval=range(1, 100),
    steps_floquet_unitary=100,
    phi_mix=None,
    init_state=None,
):
    """
    :param A: Amplitude of the AC magnetic field.
    :param n: Number of spins in the spin chain.
    :param J: Coupling constant between spins.
    :param phi: Phase of Kick Operator
    :param h: Magnitude of the static magnetic field.
    :param tau: Period of the AC magnetic field.
    :param varphi: Angle parameter for AC magnetic field.
    :param theta: Angle parameter for AC magnetic field.
    :param phi_0: Initial phase offset for the AC magnetic field.
    :param nu: Frequency normalization parameter.
    :param steps_floquet_unitary: Number of steps to discretize the
           Floquet unitary operator.
    :return: A NumPy array with shape (time_points_num, 2), where
             each row consists of the time point and the corresponding
             observable value.  f
    """
    Zsum, Xsum, Ysum = create_spin_xyz_operators(n)
    H_0 = create_hamiltonian_h0(J, h, n)
    num_levels = 0
    for energy in find_eigen_values(H_0):
        if energy < -h * n:
            num_levels +=1
        else:
            break
    gs_mins_0, gs_plus_0, *_ = get_initial_states_pos_neg(H_0, number_of_pairs=1)
    ket_0 = identify_state(init_state, H_0, Zsum, phi_mix=phi_mix,
                           number_of_pairs=num_levels // 2)
    floquet_unitary_T = calculate_unitary_T(
        A,
        n,
        phi,
        tau,
        varphi,
        theta,
        phi_0,
        nu,
        steps_floquet_unitary,
        H_0,
    )
    floquet_unitary_p_delta_T = calculate_unitary_T(
        A + epsilon,
        n,
        phi,
        tau,
        varphi,
        theta,
        phi_0,
        nu,
        steps_floquet_unitary,
        H_0,
    )
    floquet_unitary_p2_delta_T = calculate_unitary_T(
        A + 2 * epsilon,
        n,
        phi,
        tau,
        varphi,
        theta,
        phi_0,
        nu,
        steps_floquet_unitary,
        H_0,
    )
    floquet_unitary_m_delta_T = calculate_unitary_T(
        A - epsilon,
        n,
        phi,
        tau,
        varphi,
        theta,
        phi_0,
        nu,
        steps_floquet_unitary,
        H_0,
    )
    floquet_unitary_m2_delta_T = calculate_unitary_T(
        A - 2*epsilon,
        n,
        phi,
        tau,
        varphi,
        theta,
        phi_0,
        nu,
        steps_floquet_unitary,
        H_0,
    )
    fu_eigenvalues, fu_eigenvectors = np.linalg.eig(
        floquet_unitary_T
    )
    fu_delta_m_eigenvalues, fu_delta_m_eigenvectors = np.linalg.eig(
        floquet_unitary_m_delta_T
    )
    fu_delta_m2_eigenvalues, fu_delta_m2_eigenvectors = np.linalg.eig(
        floquet_unitary_m2_delta_T
    )
    fu_delta_p_eigenvalues, fu_delta_p_eigenvectors = np.linalg.eig(
        floquet_unitary_p_delta_T
    )
    fu_delta_p2_eigenvalues, fu_delta_p2_eigenvectors = np.linalg.eig(
        floquet_unitary_p2_delta_T
    )
    # # Create a pooled process to parallelize tasks
    # with Pool() as pool:
    #     args = [
    #         (
    #             time,
    #             A,
    #             n,
    #             J,
    #             phi,
    #             h,
    #             tau,
    #             varphi,
    #             theta,
    #             phi_0,
    #             nu,
    #             epsilon,
    #             steps_floquet_unitary,
    #             H_0,
    #             fu_eigenvalues,
    #             fu_eigenvectors,
    #             fu_delta_m_eigenvalues,
    #             fu_delta_m_eigenvectors,
    #             fu_delta_m2_eigenvalues,
    #             fu_delta_m2_eigenvectors,
    #             fu_delta_p_eigenvalues,
    #             fu_delta_p_eigenvectors,
    #             fu_delta_p2_eigenvalues,
    #             fu_delta_p2_eigenvectors,
    #             ket_0,
    #             gs_mins_0,
    #             gs_plus_0,
    #             Zsum,
    #             Xsum,
    #             Ysum,
    #         )
    #         for time in time_interval
    #     ]
    #     results = pool.starmap(process_time_point, args, chunksize=100)
    results = []
    for time in time_interval:
        res = process_time_point(
            time,
            A,
            n,
            J,
            phi,
            h,
            tau,
            varphi,
            theta,
            phi_0,
            nu,
            epsilon,
            steps_floquet_unitary,
            H_0,
            fu_eigenvalues,
            fu_eigenvectors,
            fu_delta_m_eigenvalues,
            fu_delta_m_eigenvectors,
            fu_delta_m2_eigenvalues,
            fu_delta_m2_eigenvectors,
            fu_delta_p_eigenvalues,
            fu_delta_p_eigenvectors,
            fu_delta_p2_eigenvalues,
            fu_delta_p2_eigenvectors,
            ket_0,
            gs_mins_0,
            gs_plus_0,
            Zsum,
            Xsum,
            Ysum,
        )
        results.append(res)

    return results


def analytic_function(x, theta, phi, N_p=1.0):
    """
    Implements the function f_Q(x, N_p, θ, φ) =
        (1 - cos²(x/2 + φ) * sin²(θ)/N_p) * sin²(x/2)/(x/2)²

    Parameters:
    x (float or array): Input variable
    N_p (float): Parameter N_p
    theta (float): Angle theta in radians
    phi (float): Phase angle phi in radians

    Returns:
    float or array: Result of the function
    """
    # Calculate x/2 term used multiple times
    x_half = x / 2
    # Calculate cos²(x/2 + φ)
    cos_squared = np.cos(x_half + phi) ** 2
    # Calculate sin²(θ)
    sin_squared_theta = np.sin(theta) ** 2
    # Calculate the first term: (1 - cos²(x/2 + φ) * sin²(θ)/N_p)
    first_term = 1 - (cos_squared * sin_squared_theta / N_p)
    # Calculate the second term: sin²(x/2)/(x/2)²
    # Handle the case when x is close to zero to avoid division by zero
    # For x ≈ 0, sin²(x/2)/(x/2)² ≈ 1
    with np.errstate(divide="ignore", invalid="ignore"):
        second_term = np.where(
            np.abs(x_half) < 1e-10, 1.0, (np.sin(x_half) ** 2) / (x_half**2)
        )
    # Combine the terms
    result = first_term * second_term

    return result


def analytical_form(time_points, theta, phi, tau, m, gap, N_p=1.0):
    const = 4 / np.pi**2 - 2 * gap**2 * tau**2 * (-12 + np.pi**2) / (
        3 * np.pi**4
    )
    return (
        4.0
        * m**2
        * const
        * analytic_function(time_points * gap * tau / 2.0, theta, phi, N_p=N_p)
    )

def get_initial_states_pos_neg(H, number_of_pairs=None):
    """
    Get eigenvectors of a Hamiltonian matrix sorted by eigenvalues, utilizing its block structure.

    Args:
        H (np.ndarray): The input Hamiltonian matrix, assumed to be block diagonal with tridiagonal blocks.

    Returns:
        list: List of eigenvectors corresponding to eigenvalues in ascending order.

    Raises:
        ValueError: If H is not a valid square matrix or submatrices are not tridiagonal.
    """
    # Perform diagonalization with mpmath

    # Sort eigenvalues and eigenvectors
    eigvals_mp, eigvecs_mp = mp.eigh(mp.matrix(H))
    eigvecs = [np.array([np.complex128(x) for x in eigvecs_mp[:, i].tolist()])
               for i in range(len(eigvals_mp))]
    return eigvecs[:2 * number_of_pairs] if number_of_pairs is not None else eigvecs

def find_eigen_values(H_0):
    k = H_0.shape[0] // 2
    A11, A22 = H_0[:k, :k], H_0[k:, k:]
    # Extract diagonals efficiently
    d1, e1 = np.real(np.diagonal(A11)), np.real(np.diagonal(A11, offset=1))
    d2, e2 = np.real(np.diagonal(A22)), np.real(np.diagonal(A22, offset=1))
    eigenvalues_1 = eigvalsh_tridiagonal(d1, e1)
    eigenvalues_2 = eigvalsh_tridiagonal(d2, e2)
    combined = np.concatenate([eigenvalues_1, eigenvalues_2])
    non_zero_values = combined[combined != 0]
    return np.sort(non_zero_values)


def find_eigen_values_mpmath(H_0):
    eigvals_mp, eigvecs_mp = mp.eigh(mp.matrix(H_0))
    return eigvals_mp

def find_gap(H_0, precision="mpmath"):
    # Perform diagonalization with mpmath
    if precision == "mpmath":
        eigenvalues = find_eigen_values_mpmath(H_0)
    else:
        eigenvalues = find_eigen_values(H_0)
    return abs(float(eigenvalues[0] - eigenvalues[1]))

def read_last_time(csv_path) -> Optional[float]:
    """
    Read a CSV file and return the last line as a dictionary.

    Parameters:
    -----------
    csv_path : str or Path
        Path to the CSV file

    Returns:
    --------
    dict
        Dictionary containing the last row data with column names as keys
    None
        If file doesn't exist or is empty
    """
    if csv_path is None:
        return None
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Check if dataframe is not empty
        if df.empty:
            print(f"File {csv_path} is empty.")
            return None

        # Get the last row as a dictionary
        last_row = df.iloc[-1].to_dict()

        return last_row["time_points"]

    except FileNotFoundError:
        print(f"File {csv_path} not found.")
        return None
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None


def save_to_file_qfi_dynamics(
    results_var,
    A,
    N,
    nu,
    h,
    tau,
    analytical=False,
    title_add=False,
    filename="",
):
    """
    Plot Quantum Fisher Information dynamics with various customization options.

    Parameters:
    -----------
    results_var : list
        List of tuples containing (phi_mix, results_nu) for diff param values
    A : float
        Amplitude of AC field
    N : int
        System size
    nu : float
        Frequency parameter
    h : float
        Field strength parameter
    tau : float
        Time step parameter
    analytical : bool, optional
        Whether to include analytical solution in the plot
    title_add : bool, optional
        Whether to add a detailed title to the plot
    arbitrary_states : bool, optional
        If True, phi_mix represents some arbitrary state; otherwise it's a phase
    filename : str, optional
        Custom filename for saving the plot

    Returns:
    --------
    str
        Path to the saved plot file
    """
    plt.figure(figsize=(10, 10))
    # Set plot title if requested
    if title_add:
        title = (
            rf"QFI var for different amplitude $A$ of AC field "
            rf"for $A={A}$, $N={N}$, $h={h}$, $n={nu}$\\n"
            r"$\frac{F_q}{(N n)^2} = "
            r"\left(\frac{2 m_0 \tau \sin{(\Delta \tau n)} }{\Delta "
            r"\tau n \cos{(\Delta \tau / 2)}} \right)^2$"
        )
        plt.suptitle(title, fontsize=14)

    max_qfi = -np.inf

    # Create a dictionary to store data for CSV
    csv_data = {
        "time_points": [],
        "initial_state": [],
        "qfi_values": [],
        "m_0_values": [],
        "m_1_values": [],
        "N": [],
        "h": [],
        "J": [],
        "tau": [],
    }

    # Process each parameter set
    for initial_state, results_nu in results_var:
        if (isinstance(initial_state, np.longdouble) or
            isinstance(initial_state, float) or
            isinstance(initial_state, np.complex128)):
            in_state = "phi_" + str(initial_state)
            phi_ = initial_state
        else:
            in_state = initial_state.name
        # Extract data points
        (time_points, qfi_values, m_0_values, m_1_values, params) = (
            extract_data_points(results_nu, N)
        )
        # Store data for CSV
        for i, t in enumerate(time_points):
            if qfi_values[i] < 0 or qfi_values[i] > 1:
                qfi_values[i] = 0 # avoid numerical errors!
            csv_data["time_points"].append(t)
            csv_data["initial_state"].append(in_state)
            csv_data["qfi_values"].append(qfi_values[i])
            csv_data["m_0_values"].append(m_0_values[i])
            csv_data["m_1_values"].append(m_1_values[i])
            csv_data["tau"].append(tau)
            csv_data["N"].append(N)
            csv_data["h"].append(params[i]["h"])
            csv_data["J"].append(params[i]["J"])
        # Track maximum QFI value for y-axis limit
        late_stage_max = max(
            [qfi for t, qfi in zip(time_points, qfi_values) if t > 1],
            default=-np.inf,
        )
        max_qfi = max(max_qfi, late_stage_max)
        # Determine appropriate label
        # Plot QFI values
        plt.plot(time_points, qfi_values, ":", markersize=5.0)
        if analytical:
            last_time = time_points[-1]
            times = generate_log_plot_points(
                start=1, end=last_time, points_per_range=1000
            )
            plot_analytical_solution(times, 0, phi_ * 2, tau, N, h)
    # Format plot
    plt.title("QFI", fontsize=14)
    plt.xlabel("$t, \\tau$")
    plt.ylabel(r"$F_Q / \,\, (N t \tau)^2$")
    plt.legend()
    # Set scale and limits
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_ylim(0, max_qfi * 1.05)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # Save data to CSV
    csv_filename = (
        results_dir / f"{str(filename) if filename else 'qfi_data'}.csv"
    )
    df = pd.DataFrame(csv_data)
    # Check if file exists
    if os.path.exists(csv_filename):
        # If file exists, append without writing header
        df.to_csv(csv_filename, mode="a", header=False, index=False)
    else:
        # If file doesn't exist, create new file with header
        df.to_csv(csv_filename, index=False)
    # Save figure
    output_file = save_figure(filename)
    return output_file


def extract_data_points(results_nu, N):
    """Extract time points and calculated values from results."""
    time_points = []
    qfi_values = []
    m_0_values = []
    m_1_values = []
    params = []

    for qfi_info in results_nu:
        time_points.append(qfi_info.time)
        m_0_values.append(abs(qfi_info.m_0))
        m_1_values.append(abs(qfi_info.m_1))
        qfi_values.append(qfi_info.qfi / (N * qfi_info.time) ** 2)
        params.append(qfi_info.params)

    return time_points, qfi_values, m_0_values, m_1_values, params


def plot_analytical_solution(time_points, phi, theta, tau, N, h, J=1.0):
    """Add analytical solution to the plot."""
    times = np.array(list(time_points))
    H_0 = create_hamiltonian_h0(J, h, N)
    # Calculate magnetization
    m = (0.5 - 0.25 * (h / J) ** 4 / N) * np.sqrt(1 - (h / J) ** 2)
    gap = find_gap(H_0)
    # Plot analytical solution
    label = (
        rf"$F_Q$ for ground state with "
        rf"params $\phi={np.round(np.real(phi) / np.pi, 2)}\pi$, "
        rf"$\theta={np.round(np.real(theta) / np.pi, 2)}\pi$"
    )
    plt.plot(
        times,
        analytical_form(times, theta, phi, tau, m, gap), "-", label=label
    )


def plot_sum_analytical_solutions(
    time_points, phi, theta, tau, N, h, num_pairs=2, J=1.0
):
    """Add analytical solution to the plot."""
    times = np.array(
        list(
            chain(
                range(0, 10000),
                range(10000, int(max(time_points)) + 1, 2000)
            )
        )
    )
    Zsum, Xsum, Ysum = create_spin_xyz_operators(N)
    H_0 = create_hamiltonian_h0(J, h, N)
    # Compute smallest k eigenvalues and eigenvectors of the modified matrix Ho
    eigvals_Ho, eigvecs_Ho = eigh(H_0, eigvals_only=False)
    # Calculate magnetization
    # Extract the real parts of the eigenvalues from the modified matrix
    eig_Ho = np.real(eigvals_Ho)
    # Sort the eigenvalues (and corresponding eigenvectors) by value
    sorted_indices_Ho = np.argsort(eig_Ho)
    eigvecs_Ho_sorted = eigvecs_Ho[:, sorted_indices_Ho]
    energies = eig_Ho[sorted_indices_Ho]
    gaps = set()
    for en1_idx in range(len(energies)):
        for en2_idx in range(len(energies)):
            if en1_idx == en2_idx:
                continue
            en1 = energies[en1_idx]
            en2 = energies[en2_idx]
            gs_mins_0 = eigvecs_Ho_sorted[:, en1_idx]
            gs_plus_0 = eigvecs_Ho_sorted[:, en2_idx]
            gaps.add(
                (
                    abs(en1 - en2),
                    abs(np.real((gs_mins_0.T.conj()).dot(Zsum.dot(gs_plus_0))))
                    / N,
                )
            )
    gaps = sorted(list(gaps))
    # Plot analytical solution
    label = rf"$F_Q$ for sum of {num_pairs} pairs"
    values = np.zeros(len(times))
    for gap, m in gaps[:num_pairs]:
        values += analytical_form(times, theta, phi, tau, m, gap, N_p=num_pairs)
    plt.plot(times, values / num_pairs, "--", label=label)


def save_figure(filename: str = "qfi_dependencies.png"):
    """Save the figure to a file and return the path."""
    output_file = results_dir / filename
    plt.savefig(output_file, format="png", dpi=300)
    return output_file

def read_defaults_args_from_config():
    # First, create a config parser to read default values
    config = configparser.ConfigParser()
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "qfi_simulation.ini"
    )

    # Set up default values dictionary
    defaults = {
        "A_amplitude": 0.0,
        "system_size": 20,
        "J_coupling": 1.0,
        "h_field_strength": 0.4,
        "tau": 1.0,
        "phi_kick_phase": np.pi,
        "frequency": 2,
        "phi_0": 0.0,
        "varphi": 0.0,
        "theta": 0.0,
        "steps_floquet": 100,
        "time_points": 1000,
        "points_per_range": 1000,
        "epsilon": 1e-12,
        "output_dir": "./",
    }

    # Read config file if it exists
    if os.path.exists(config_file):
        config.read(config_file)
        if "Simulation" in config:
            sim_config = config["Simulation"]
            # Override defaults with values from config file
            for key in defaults:
                # Convert to config file style
                config_key = key.replace("_", "-")
                if config_key in sim_config:
                    value = sim_config[config_key]
                    # Convert string values to appropriate types
                    if key == "output_dir":
                        defaults[key] = value
                    elif key in [
                        "system_size",
                        "frequency",
                        "steps_floquet",
                        "time_points",
                        "points_per_range",
                    ]:
                        defaults[key] = int(float(value))
                    else:
                        if value.lower() == "pi":
                            defaults[key] = np.pi
                        else:
                            defaults[key] = float(value)
    return defaults


def parse_arguments():
    """Parse command-line arguments for the
    simulation with config file support."""
    defaults = read_defaults_args_from_config()
    # Now set up argument parser with defaults from config
    parser = argparse.ArgumentParser(
        description="Quantum Fisher Information Simulation Tool.\n"
        "Running the script from command line would look "
        "something like this:\n"
        "`python ./quantum_fisher_information_simulation.py "
        "--amplitude 0.0 --plot-type all`"
    )

    # Add arguments with defaults from config file or built-in defaults
    parser.add_argument(
        "--amplitude",
        type=float,
        default=defaults["A_amplitude"],
        help="Amplitude of the AC magnetic field",
    )
    parser.add_argument(
        "--system-size",
        type=int,
        default=defaults["system_size"],
        help="System size",
    )
    parser.add_argument(
        "--coupling",
        type=float,
        default=defaults["J_coupling"],
        help="Coupling strength",
    )
    parser.add_argument(
        "--field-strength",
        type=float,
        default=defaults["h_field_strength"],
        help="Field strength",
    )
    parser.add_argument(
        "-t", "--tau", type=float,
        default=defaults["tau"], help="Tau parameter"
    )
    parser.add_argument(
        "-p",
        "--phi",
        type=float,
        default=defaults["phi_kick_phase"],
        help="Phi parameter",
    )
    parser.add_argument(
        "-nu",
        "--frequency",
        type=int,
        default=defaults["frequency"],
        help="Frequency parameter",
    )
    parser.add_argument(
        "--phi-0",
        type=float,
        default=defaults["phi_0"],
        help="Initial phi value",
    )
    parser.add_argument(
        "--varphi",
        type=float,
        default=defaults["varphi"],
        help="Varphi parameter",
    )
    parser.add_argument(
        "--theta", type=float, default=defaults["theta"], help="Theta parameter"
    )
    parser.add_argument(
        "--steps-floquet",
        type=int,
        default=defaults["steps_floquet"],
        help="Steps for Floquet unitary calculation",
    )
    parser.add_argument(
        "--time-points",
        type=int,
        default=defaults["time_points"],
        help="Number of time points",
    )
    parser.add_argument(
        "--points-per-range",
        type=int,
        default=defaults["points_per_range"],
        help="Number of points per each range",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=defaults["epsilon"],
        help="Epsilon parameter for numerical precision",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["phi-mix", "levels", "energy", "all"],
        default="all",
        help="Type of initial state plot to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=defaults["output_dir"],
        help="Directory to save output files",
    )
    args = parser.parse_args()
    return args


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def generate_time_interval(
    time_points_num, points_per_range, csv_filename=None
):
    """Generate a time interval with appropriate sampling density."""
    return generate_log_plot_points(
        start=0,
        end=time_points_num,
        points_per_range=points_per_range,
    )


def generate_log_plot_points(start=1, end=10e16, points_per_range=1000):
    """
    Generate points for a logarithmic plot with equal number of
    points per logarithmic range.

    Args:
        start: The starting value (default: 1)
        end: The ending value (default: 10e16)
        points_per_range: Number of points to generate in each
        logarithmic range (default: 10000)

    Returns:
        np.array: Array of points suitable for a log plot
    """
    points = []

    # Define range boundaries (powers of 10)
    range_boundaries = [10**i for i in range(0, 20)]  # From 10^0 to 10^20

    # Filter ranges based on start and end parameters
    filtered_ranges = []
    for i in range(len(range_boundaries) - 1):
        range_start = range_boundaries[i]
        range_end = range_boundaries[i + 1]

        # Skip if range is completely outside our target range
        if range_end <= start or range_start >= end:
            continue

        # Adjust range to stay within target bounds
        adjusted_start = max(start, range_start)
        adjusted_end = min(end, range_end)

        filtered_ranges.append((adjusted_start, adjusted_end))

    # Generate points for each range
    for range_start, range_end in filtered_ranges:
        # Use logarithmic spacing to get equal number of points
        # in each logarithmic range
        # This creates denser points near the start and sparser
        # points near the end of each range
        if range_start == range_end:
            # Edge case: if range_start equals range_end,
            # just add that single point
            points.append(range_start)
        else:
            # Generate logarithmically spaced points
            log_points = np.logspace(
                np.log10(range_start),
                np.log10(range_end),
                points_per_range,
                # Only include endpoint for the final range
                endpoint=(range_end == end),
            )
            points.extend(log_points)

    return sorted(set([int(x) for x in points]))


def run_phi_mix_simulation(args):
    """Run simulations varying the phi_mix parameter."""
    logging.info("Running phi_mix parameter simulations...")
    results = []

    phi_mix_values = np.linspace(0, np.pi / 4, 3, dtype=np.complex128)

    for phi_mix in phi_mix_values:
        logging.info(f"Simulating with phi_mix = {np.real(phi_mix):.4f}")
        time_interval = range(1, 100, 10)
        result = observarble_simulation_optimal_with_AC_field(
            args.amplitude,
            args.system_size,
            args.coupling,
            args.phi,
            args.field_strength,
            args.tau,
            args.varphi,
            args.theta,
            args.phi_0,
            args.frequency,
            epsilon=args.epsilon,
            time_interval=time_interval,
            steps_floquet_unitary=args.steps_floquet,
            phi_mix=phi_mix,
        )
        results.append((phi_mix, result))

    output_file = f"{args.output_dir}/qfi_dependencies.png"
    save_to_file_qfi_dynamics(
        results,
        args.amplitude,
        args.system_size,
        args.frequency,
        args.field_strength,
        args.tau,
        analytical=True,
        filename=output_file,
    )
    logging.info(f"Phi mix simulation results saved to {output_file}")

    return results


def run_energy_levels_simulation(args):
    """Run simulations varying the number of energy levels."""
    logging.info("Running energy levels simulations...")
    results = []
    output_file = f"{args.output_dir}/general_case.png"
    init_states = [
        InitialState.PHYS,
        # InitialState.GS_PHYS,
        # InitialState.GS_CAT,
        # InitialState.CAT_SUM,
    ]

    for state in init_states:
        time_interval = range(1, 100, 10)
        logging.info(
            f"Simulating for '{state}' state "
            f"for {len(time_interval)} points!"
        )
        result = observarble_simulation_optimal_with_AC_field(
            args.amplitude,
            args.system_size,
            args.coupling,
            args.phi,
            args.field_strength,
            args.tau,
            args.varphi,
            args.theta,
            args.phi_0,
            args.frequency,
            epsilon=args.epsilon,
            time_interval=time_interval,
            steps_floquet_unitary=args.steps_floquet,
            init_state=state,
        )
        results.append((state, result))

    save_to_file_qfi_dynamics(
        results,
        args.amplitude,
        args.system_size,
        args.frequency,
        args.field_strength,
        args.tau,
        filename=output_file,
    )
    logging.info(f"Custom states simulation results saved to {output_file}")

    return results

def main():
    """Main function to execute the simulation based on arguments."""
    args = parse_arguments()
    setup_logging()
    logging.info(
        f"Starting quantum simulation with amplitude A = {args.amplitude}"
    )

    if args.plot_type == "phi-mix":
        run_phi_mix_simulation(args)

    if args.plot_type == "levels" or args.plot_type == "all":
        run_energy_levels_simulation(args)

    logging.info("Simulation completed successfully")


if __name__ == "__main__":
    main()
