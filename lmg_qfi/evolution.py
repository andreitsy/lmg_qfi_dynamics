"""Time evolution and Floquet unitary calculations."""

import mpmath as mp

from .config import UF
from .operators import (
    create_spin_xyz_operators,
    create_kick_operator,
    create_v_operator,
)


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
    Compute evolution of the Floquet unitary over one period.
    """
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
    """
    Compute the r-th power of the Floquet unitary using eigendecomposition.
    """
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
    Zsum, Xsum, Ysum = create_spin_xyz_operators(params["N"])
    t_delta = mp.mpf(params["T"]) / params["steps_floquet_unitary"]
    omega = mp.mpf(2) * mp.pi / mp.mpf(params["nu"] * params["T"])
    r = time // params["nu"]
    extra_interval = range(r * params["nu"] + 1, time + 1)
    floquet_unitary = find_power_r_mpmath(floque_u, r)
    
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
        floquet_unitary = create_kick_operator(params["phi"], Xsum) * floquet_unitary

    return floquet_unitary


def calculate_unitary_T(
        h: mp.mpf,
        params: dict,
        H_0: mp.matrix,
):
    """
    Calculate the Floquet unitary for one complete period.
    """
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

    for p in range(1, nu + 1):
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
