import pytest
import numpy as np
import mpmath as mp
from ..quantum_fisher_information_simulation_mpmath import (
    generate_time_interval,
    calculate_unitary_T as calculate_unitary_T_mpmath,
    create_spin_xyz_operators as create_spin_xyz_operators_mpmath,
    create_kick_operator as create_kick_operator_mpmath,
    create_hamiltonian_h0 as create_hamiltonian_h0_mpmath)
from scipy.linalg import expm

# suppress=True avoids scientific notation for small numbers
np.set_printoptions(precision=4, suppress=True)


def h_ac(S_x, S_y, S_z, omega, phi_0, t_k, theta, varphi):
    def _compute_constants(theta, varphi):
        theta = np.complex128(theta)
        varphi = np.complex128(varphi)
        return np.sin(theta), np.cos(theta), np.sin(varphi), np.cos(varphi)

    # Compute the time-dependent sinusoidal factor
    time_factor = t_k * omega + phi_0
    sinusoidal_factor = np.sin(time_factor)
    # Compute constants based on precision
    sin_theta, cos_theta, sin_varphi, cos_varphi = _compute_constants(theta, varphi)
    # Calculate and return the AC field Hamiltonian
    return sinusoidal_factor * (
            sin_theta * cos_varphi * S_x +
            sin_theta * sin_varphi * S_y +
            cos_theta * S_z
    )


def create_v_operator(H_0, S_x, S_y, S_z, omega, phi_0, A, t_k, theta, varphi):
    S_alpha_part = A * h_ac(S_x, S_y, S_z, omega, phi_0, t_k, theta, varphi)
    return H_0 + S_alpha_part


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
):
    Zsum, Xsum, Ysum = create_spin_xyz_operators(n)
    t_delta = tau / steps_floquet_unitary
    omega = 2.0 * np.pi / (nu * tau)
    floquet_unitary = np.eye(*H_0.shape, dtype=np.complex128)

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
        )
        floquet_unitary = create_kick_operator(phi, Xsum).dot(floquet_unitary)
    return floquet_unitary


def create_z_operator(n):
    """
    Create a z-operator with enhanced precision.

    :param n: The size parameter used to generate the z-operator (matrix dimension is n+1).
    :return: A dense diagonal matrix of size (n+1, n+1) with specific values.
    """
    half_n = n / 2
    # Create the diagonal elements: they run from half_n down
    # to -half_n in steps of -1
    diag_elements = np.arange(half_n, -half_n - 1, -1, dtype=np.complex128)
    # Return the dense diagonal matrix
    return np.diag(diag_elements)


def create_spin_minus_operators(n):
    """
    Create a spin lowering (minus) operator with enhanced precision.

    :param n: An integer representing twice the spin quantum number
              (i.e., matrix dimension is n+1),
              which implies the spin value is n/2
    :return: A dense matrix representing the spin lowering (minus) operator with the specified precision.
    """
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


def create_spin_plus_operators(n):
    """
    :param n: The dimension or size for the spin operators.
    :return: Transpose and complex conjugate of the spin-minus operators,
             resulting in the spin-plus operators.
    """
    return create_spin_minus_operators(n).T.conjugate()


def create_spin_xyz_operators(n):
    """
    :param n: The size of the system for which the spin
              operators are being created.
    :return: A tuple containing the Z, X, and Y spin operators for the system.
    """
    Splus = create_spin_plus_operators(n)
    Sminus = create_spin_minus_operators(n)
    Zsum = create_z_operator(n)
    Xsum = (Splus + Sminus) / 2
    Ysum = -1j * (Splus - Sminus) / 2
    return Zsum, Xsum, Ysum


def create_hamiltonian_h0(coupling_zz, coupling_x, num_spins):
    """
    Create the Hamiltonian H0 with the given coupling constants and spin size.

    :param coupling_zz: Coupling constant for the ZZ interaction.
    :param coupling_x: Coupling constant for the X interaction.
    :param num_spins: Number of spins in the system.
    :return: The Hamiltonian matrix H0.
    """
    # Create spin operators
    Zsum, Xsum, Ysum = create_spin_xyz_operators(num_spins)
    # Constants for precision
    precision_factor = np.complex128
    hzz_factor = precision_factor(-coupling_zz * (2 / num_spins))
    hx_factor = precision_factor(-coupling_x * 2)
    # Calculate Hamiltonian terms
    Hzz = hzz_factor * Zsum.dot(Zsum)
    Hx = hx_factor * Xsum
    return Hzz + Hx


def create_h_ac_operator(S_x, S_y, S_z, nu, T, phi_0, h, t_k, theta_0, varphi_0):
    """
    Creates a time-dependent AC Hamiltonian operator.

    Args:
        S_x, S_y, S_z (numpy.ndarray): Spin operator matrices (x, y, z components).
        phi_0 (float): Phase offset (rad).
        h (float): Amplitude of AC-field.
        t_k (float): Time point.
        theta_0 (float): Polar angle (rad).
        varphi_0 (float): Azimuthal angle (rad).

    Returns:
        numpy.ndarray: AC Hamiltonian matrix in a form:
        $$
        \hat{V} = h \sin{\left(\frac{2 \pi}{\nu T} t_k + \phi_0\right)}
                    \left(\sin{\theta_0} \cos{\varphi_0}  \hat{S}_x +
                          \sin{\theta_0} \sin{\varphi_0}  \hat{S}_y +
                          \cos{\theta_0} \hat{S}_z
                    \right)
        $$
    Raises:
        TypeError: If inputs are not of expected types.
        ValueError: If matrix shapes are incompatible.
    """
    omega = 2.0 * np.pi / (nu * T)
    v_ac = np.sin(t_k * omega + phi_0) * (
            np.sin(theta_0) * np.cos(varphi_0) * S_x +
            np.sin(theta_0) * np.sin(varphi_0) * S_y +
            np.cos(theta_0) * S_z
    )
    return h * v_ac


def create_kick_operator(phi, s_x, precision=np.complex128):
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
    phi = precision(phi)
    matrix = -1j * phi * s_x.astype(precision)
    return expm(matrix)


def convert_mpmatrix_to_numpy(mp_math, precision=15):
    with mp.workdps(precision):
        np_mat = np.array(
            [[complex(x.real, x.imag) for x in row] for row in mp_math.tolist()],
            dtype=np.complex128
        )
    return np_mat


@pytest.mark.parametrize("h", [0.0, 1e-6, 1e-2])
@pytest.mark.parametrize("n", [2, 5, 10])
@pytest.mark.parametrize("steps_floquet_unitary", [5, 15, 30])
def test_floquet_unitary(h, n, steps_floquet_unitary):
    J = 1.0
    B = 0.45
    phi = np.pi
    T = 1.5
    varphi = 0.01
    theta = 0.01
    nu = 2
    phi_0 = 0.01
    params = dict(J=J, B=B, phi=phi, T=T, varphi=varphi, h=h, N=n, nu=nu, phi_0=phi_0,
                  steps_floquet_unitary=steps_floquet_unitary, theta=theta)
    # expected
    H0 = create_hamiltonian_h0(J, B, n)
    floquet_unitary_T_expected = calculate_unitary_T(h, n, phi, T, varphi, theta,
                                                     phi_0, nu, steps_floquet_unitary, H0, )
    # calculated
    H0_mpmath = create_hamiltonian_h0_mpmath(J, B, n)
    floquet_unitary_T = calculate_unitary_T_mpmath(h, params, H0_mpmath)
    assert np.allclose(convert_mpmatrix_to_numpy(floquet_unitary_T), floquet_unitary_T_expected)

    E, ER = mp.eig(floquet_unitary_T)
    ER_inv = mp.inverse(ER)
    assert np.allclose(convert_mpmatrix_to_numpy(ER * mp.diag([e ** 5 for e in E]) * ER_inv),
                       convert_mpmatrix_to_numpy(floquet_unitary_T ** 5))


@pytest.mark.parametrize("phi", [0.123, 1.32, 12])
@pytest.mark.parametrize("n", [2, 5, 10])
def test_kick_operator_mpmath(n, phi):
    Zsum, Xsum, Ysum = create_spin_xyz_operators(n)
    Zsum_mp, Xsum_mp, Ysum_mp = create_spin_xyz_operators_mpmath(n)
    kick_expected = create_kick_operator(phi, Xsum)
    kick_mp = create_kick_operator_mpmath(phi, Xsum_mp)
    assert np.allclose(convert_mpmatrix_to_numpy(kick_mp), kick_expected)
    assert np.allclose(convert_mpmatrix_to_numpy(Zsum_mp), Zsum)
    assert np.allclose(convert_mpmatrix_to_numpy(Xsum_mp), Xsum)
    assert np.allclose(convert_mpmatrix_to_numpy(Ysum_mp), Ysum)


@pytest.mark.parametrize("J, B, N", [
    pytest.param(1.0, 0.0, 1, id="no_B_field"),
    pytest.param(1.0, 0.0, 10, id="no_B_field_big"),
    pytest.param(1.0, 1.0, 10, id="B_field_case"),
    pytest.param(1.0, 0.4, 25, id="base_case"),
])
def test_hamiltonian_mpmath(J, B, N):
    ham_expected = create_hamiltonian_h0(J, B, N)
    ham = create_hamiltonian_h0_mpmath(J, B, N)
    assert np.allclose(convert_mpmatrix_to_numpy(ham), ham_expected)


def test_generate_time_interval():
    interval = generate_time_interval(10, 5)
    for i in range(1, len(interval)):
        assert interval[i - 1] < interval[i]
