import pytest
import numpy as np
from scipy.linalg import eigh
from ..simulation_qfi import build_naive_hamiltonian, evolution_stroboscopic, get_observable

# suppress=True avoids scientific notation for small numbers
np.set_printoptions(precision=4, suppress=True)


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


@pytest.mark.parametrize("J, B, N", [
    pytest.param(1.0, 0.1, 4, id="base_case"),
])
def test_eigenvalues(J, B, N):
    ham_expected = create_hamiltonian_h0(J, B, N)
    eigenvalues_expected, eigvecs_expected = eigh(ham_expected, check_finite=False, eigvals_only=False)

    sorted_indices_Ho = np.argsort(np.real(eigenvalues_expected))
    eigvecs_sorted = [eigvecs_expected[:, i] for i in sorted_indices_Ho]
    print("\n", eigenvalues_expected)
    print("\n", eigvecs_sorted[0])
    print("\n", eigvecs_sorted[1])
    print("\n", (eigvecs_sorted[0] + eigvecs_sorted[1]) / np.sqrt(2))
