import mpmath as mp
import logging
import matplotlib.pyplot as plt
from typing import List
from enum import Enum
from dataclasses import dataclass

DPS = 30
DEBUG = True
_logger_initialized = False


class InitialState(Enum):
    """Enumeration of possible initial quantum states."""
    GS_PHYS = "GS_phys"
    GS_CAT = "GS_cat"
    CAT_SUM = "CatSum"
    PHYS = "Phys"


@dataclass
class QFIAnalyticalInformation:
    m_x: float
    m_y: float
    m_z: float
    qfi: float
    time: float
    N: int
    params: dict


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
    term1 = (dketa_t.T.apply(mp.conj) * dketa_t)[0, 0]
    # |<ket|dketa>|^2
    overlap = (ket_t.T.apply(mp.conj) * dketa_t)[0, 0]
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


def find_power_r_mpmath(eigenvalues, eigenvectors, r):
    if r <= 0:
        return mp.eye(len(eigenvalues))
    # Compute the diagonal matrix of eigenvalues raised to the power r
    # put values on the diagonal
    Lambda = mp.eye(len(eigenvalues))
    for i, v in enumerate(eigenvalues):
        Lambda[i, i] = mp.power(v, r)
    # Convert eigenvectors to mpmath's matrix
    U = eigenvectors
    # Compute the inverse of U using mpmath
    U_inv = U.T.apply(mp.conj)
    # Perform the matrix multiplication with mpmath
    result_mp = U * Lambda * U_inv
    return result_mp


def calculate_unitary_at_time_mp(h, time, n, phi, T, varphi, theta, phi_0, nu,
                                 steps_floquet_unitary, H_0, eigenvalues, eigenvectors):
    """
    Compute the Floquet unitary at a given discrete time using mpmath (arbitrary precision).

    Parameters
    ----------
    h, phi, T, varphi, theta, phi_0 : mp.mpf
        Driving and phase parameters.
    time, n, nu, steps_floquet_unitary : int
        Discrete time indices and system size.
    H_0 : mp.matrix
        Static Hamiltonian.
    eigenvalues, eigenvectors : mp.matrix
        Floquet operator diagonalization.

    Returns
    -------
    floquet_unitary : mp.matrix
        The unitary operator at the given time.
    """
    # Spin operators with mpmath precision
    Zsum, Xsum, Ysum = create_spin_xyz_operators(n)
    # Time step within one Floquet period
    t_delta = mp.mpf(T) / steps_floquet_unitary
    # Driving frequency
    omega = mp.mpf(2) * mp.pi / mp.mpf(nu * T)
    # Number of complete Floquet cycles
    r = time // nu
    extra_interval = range(r * nu + 1, time + 1)
    # Start from floquet_unitary for r full cycles
    floquet_unitary = find_power_r_mpmath(eigenvalues, eigenvectors, r)
    # Evolve over leftover intervals
    for p in extra_interval:
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
        # Apply the kick operator
        floquet_unitary = create_kick_operator(phi, Xsum) * floquet_unitary

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


def process_time_point_mp(
        time,
        h,
        n,
        J,
        phi,
        B,
        T,
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
        fu_delta_p_eigenvalues,
        fu_delta_p_eigenvectors,
        ket_0,
        Zsum,
        Xsum,
        Ysum,
):
    """
    Compute observables and QFI at a given time point using mpmath arbitrary precision.
    """

    # Compute Floquet unitaries at different delta shifts
    floquet_unitary = calculate_unitary_at_time_mp(
        h, time, n, phi, T, varphi, theta, phi_0, nu,
        steps_floquet_unitary, H_0, fu_eigenvalues, fu_eigenvectors
    )
    floquet_unitary_p_delta = calculate_unitary_at_time_mp(
        h + epsilon, time, n, phi, T, varphi, theta, phi_0, nu,
        steps_floquet_unitary, H_0, fu_delta_p_eigenvalues, fu_delta_p_eigenvectors
    )
    floquet_unitary_m_delta = calculate_unitary_at_time_mp(
        h - epsilon, time, n, phi, T, varphi, theta, phi_0, nu,
        steps_floquet_unitary, H_0, fu_delta_m_eigenvalues, fu_delta_m_eigenvectors
    )
    # Evolve ket
    ket_t = floquet_unitary * ket_0
    ket_t_p_delta = floquet_unitary_p_delta * ket_0
    ket_t_m_delta = floquet_unitary_m_delta * ket_0

    # Compute derivative ket for QFI
    dket_t = dketa_t(ket_t_p_delta, ket_t_m_delta, epsilon)
    qfi = quantum_fisher_information_mp(dket_t, ket_t)
    if DEBUG:
        abs_error_estimate = calculate_error_estimation_mp(dket_t, ket_t)
        print("ABS_ERROR:", abs_error_estimate)

    # Compute magnetizations
    m_x = mp.re((ket_t.T.apply(mp.conj) * Xsum * ket_t)[0, 0]) / n
    m_y = mp.re((ket_t.T.apply(mp.conj) * Ysum * ket_t)[0, 0]) / n
    m_z = mp.re((ket_t.T.apply(mp.conj) * Zsum * ket_t)[0, 0]) / n

    # Package results in QFIAnalyticalInformation
    return QFIAnalyticalInformation(
        qfi=float(qfi),
        time=time,
        N=n,
        m_x=float(m_x),
        m_y=float(m_y),
        m_z=float(m_z),
        params=dict(
            h=h,
            B=B,
            J=J,
            phi=phi,
            T=T,
            varphi=varphi,
            theta=theta,
            phi_0=phi_0,
            nu=nu,
            steps_floquet_unitary=steps_floquet_unitary,
        )
    )


def calculate_unitary_T(
        h,
        n,
        phi,
        T,
        varphi,
        theta,
        phi_0,
        nu,
        steps_floquet_unitary,
        H_0,
):
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


def simulation_with_AC_field_mp(
        h,
        n,
        J,
        phi,
        B,
        T,
        varphi,
        theta,
        phi_0,
        nu,
        epsilon=mp.mpf("1e-15"),
        time_interval=range(1, 100, 10),
        steps_floquet_unitary=10,
        init_state=None,
) -> List[QFIAnalyticalInformation]:
    """
    Sequential observable simulation using mpmath arbitrary precision
    """
    with mp.workdps(DPS):
        # Spin operators and static Hamiltonian
        Zsum, Xsum, Ysum = create_spin_xyz_operators(n)
        H_0 = create_hamiltonian_h0(J, B, n)
        # Ground states and initial ket
        # energies, evecs = mp.eigh(H_0)
        # sorted_indices = sorted(range(len(energies)), key=lambda i: mp.re(energies[i]))
        # ground_state = evecs[:, sorted_indices[0]]
        # first_excited_state = evecs[:, sorted_indices[1]]
        ket_0 = mp.zeros(len(H_0), 1)
        ket_0[0] = mp.mpf('1.0')

        floquet_unitary_T = calculate_unitary_T(
            h,
            n,
            phi,
            T,
            varphi,
            theta,
            phi_0,
            nu,
            steps_floquet_unitary,
            H_0,
        )
        floquet_unitary_p_delta_T = calculate_unitary_T(
            h + epsilon,
            n,
            phi,
            T,
            varphi,
            theta,
            phi_0,
            nu,
            steps_floquet_unitary,
            H_0,
        )
        floquet_unitary_m_delta_T = calculate_unitary_T(
            h - epsilon,
            n,
            phi,
            T,
            varphi,
            theta,
            phi_0,
            nu,
            steps_floquet_unitary,
            H_0,
        )
        fu_eigenvalues, fu_eigenvectors = mp.eigh(floquet_unitary_T)
        fu_delta_p_eigenvalues, fu_delta_p_eigenvectors = mp.eigh(floquet_unitary_p_delta_T)
        fu_delta_m_eigenvalues, fu_delta_m_eigenvectors = mp.eigh(floquet_unitary_m_delta_T)

        # Sequentially process each time point
        results = []
        for time in time_interval:
            res = process_time_point_mp(
                time,
                h,
                n,
                J,
                phi,
                B,
                T,
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
                fu_delta_p_eigenvalues,
                fu_delta_p_eigenvectors,
                ket_0,
                Zsum,
                Xsum,
                Ysum,
            )
            results.append(res)

    return results


def run_gaps():
    N_values = [1, 5, 10, 20, 30, 50, 100]
    J = 1.0
    B = 0.1
    gaps = []
    for N in N_values:
        with mp.workdps(DPS):
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


def run_sim():
    phi = mp.pi
    J = 1.0
    B = 0.15
    N = 10
    T = 1.0
    h = 0.0
    results = simulation_with_AC_field_mp(
        h=h, n=N, J=J, phi=phi, B=B, T=T,
        varphi=0, theta=0, phi_0=0, nu=2,
        epsilon=mp.mpf("1e-20"),
        time_interval=range(1, 100, 10),
        steps_floquet_unitary=100,
    )
    times = list()
    qfi_list = list()
    for result in results:
        times.append(result.time)
        qfi_list.append(result.qfi)

    plt.figure(figsize=(8, 6))
    plt.plot(times, qfi_list, label=r'$F_h(t)$')
    plt.xlabel('Time')
    plt.ylabel(r'$F_h$')
    plt.title(f'LMG Model QFI dynamics')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_gaps()
    run_sim()
