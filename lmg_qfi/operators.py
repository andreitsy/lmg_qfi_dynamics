"""Spin operators and Hamiltonian creation for the LMG model."""

import mpmath as mp


def create_z_operator(n):
    """
    Create a z-operator with arbitrary precision using mpmath.

    :param n: The size parameter (matrix dimension is n+1).
    :return: A dense diagonal matrix of size (n+1, n+1).
    """
    half_n = mp.mpf(n) / 2
    diag_elements = [half_n - i for i in range(n + 1)]
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
    Splus = Sminus.T

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
    """
    Compute the time-dependent AC field Hamiltonian.
    """
    time_factor = t_k * omega + phi_0
    theta = mp.mpc(theta)
    varphi = mp.mpc(varphi)
    sinusoidal_factor = mp.sin(time_factor)
    return sinusoidal_factor * (
            mp.sin(theta) * mp.cos(varphi) * S_x +
            mp.sin(theta) * mp.sin(varphi) * S_y +
            mp.cos(theta) * S_z
    )


def create_v_operator(H_0, S_x, S_y, S_z, omega, phi_0, h, t_k, theta, varphi):
    """
    Create the full time-dependent Hamiltonian V(t) = H_0 + h * S_alpha(t).
    """
    S_alpha_part = mp.mpc(h) * ac_time(S_x, S_y, S_z, omega, phi_0, t_k, theta, varphi)
    return H_0 + S_alpha_part
