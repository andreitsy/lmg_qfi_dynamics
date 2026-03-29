"""QuSpin-based solver for LMG QFI dynamics.

Uses QuSpin to build spin operators and scipy/numpy for time evolution,
providing a double-precision alternative to the mpmath implementation.
"""

from typing import List, Tuple

import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian as quspin_hamiltonian
from scipy.linalg import expm

from .config import QFIInformation


def get_spin_operators(N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build collective spin operators Sz, Sx, Sy for total spin S=N/2 using QuSpin.

    States are ordered from m=+S to m=-S (Sz diagonal = [+S, +S-1, ..., -S]),
    matching the mpmath convention in operators.py.

    Parameters
    ----------
    N : int
        Number of spins (Hilbert space dimension = N+1).

    Returns
    -------
    Sz, Sx, Sy : np.ndarray of shape (N+1, N+1)
    """
    S_str = str(N // 2) if N % 2 == 0 else f"{N}/2"
    basis = spin_basis_1d(L=1, S=S_str)
    no_checks = dict(check_symm=False, check_herm=False, check_pcon=False)

    Sz_op = quspin_hamiltonian([["z", [[1.0, 0]]]], [], basis=basis,
                                dtype=np.float64, **no_checks)
    Sp_op = quspin_hamiltonian([["+", [[1.0, 0]]]], [], basis=basis,
                                dtype=np.complex128, **no_checks)
    Sm_op = quspin_hamiltonian([["-", [[1.0, 0]]]], [], basis=basis,
                                dtype=np.complex128, **no_checks)

    Sz = Sz_op.toarray()
    Sp = Sp_op.toarray()
    Sm = Sm_op.toarray()

    Sx = (Sp + Sm) / 2.0
    Sy = -1j * (Sp - Sm) / 2.0

    return Sz, Sx, Sy


def build_h0(J: float, B: float, N: int,
             Sz: np.ndarray, Sx: np.ndarray) -> np.ndarray:
    """Build LMG Hamiltonian H0 = -(2J/N)*Sz^2 - 2B*Sx."""
    return -(2.0 * J / N) * (Sz @ Sz) - 2.0 * B * Sx


def _kick_operator(phi: float, Sx: np.ndarray) -> np.ndarray:
    """Kick operator K = exp(-i * phi * Sx)."""
    return expm(-1j * phi * Sx)


def _ac_field(Sx: np.ndarray, Sy: np.ndarray, Sz: np.ndarray,
              omega: float, phi_0: float, t_k: float,
              theta: float, varphi: float) -> np.ndarray:
    """Time-dependent AC probe field S_alpha(t)."""
    s = np.sin(omega * t_k + phi_0)
    direction = (np.sin(theta) * np.cos(varphi) * Sx
                 + np.sin(theta) * np.sin(varphi) * Sy
                 + np.cos(theta) * Sz)
    return s * direction


def _evolution_T_step(U: np.ndarray, h: float, T: float,
                      varphi: float, theta: float, phi_0: float,
                      H0: np.ndarray, Sx: np.ndarray, Sy: np.ndarray, Sz: np.ndarray,
                      omega: float, p: int, t_delta: float,
                      steps_floquet_unitary: int) -> np.ndarray:
    """Evolve U over one sub-period p, mirroring evalution_T_step from evolution.py."""
    t_start = T * (p - 1)
    t_end = T * p
    linspace = [t_start + i * (t_end - t_start) / (steps_floquet_unitary - 1)
                for i in range(steps_floquet_unitary)]
    for t_k in linspace:
        V = H0 + h * _ac_field(Sx, Sy, Sz, omega, phi_0, t_k, theta, varphi)
        U = expm(-1j * t_delta * V) @ U
    return U


def compute_floquet_unitary(h: float, params: dict,
                             H0: np.ndarray,
                             Sx: np.ndarray, Sy: np.ndarray,
                             Sz: np.ndarray) -> np.ndarray:
    """Compute one-period Floquet unitary U_F, mirroring calculate_unitary_T."""
    steps = params["steps_floquet_unitary"]
    T = params["T"]
    nu = params["nu"]
    phi = params["phi"]
    varphi = params["varphi"]
    theta = params["theta"]
    phi_0 = params["phi_0"]

    t_delta = T / steps
    omega = 2.0 * np.pi / (nu * T)

    K = _kick_operator(phi, Sx)
    U = np.eye(H0.shape[0], dtype=complex)

    for p in range(1, nu + 1):
        U = _evolution_T_step(U, h, T, varphi, theta, phi_0, H0, Sx, Sy, Sz,
                               omega, p, t_delta, steps)
        U = K @ U
    return U


def _find_power_r(evals: np.ndarray, evecs: np.ndarray,
                  evecs_inv: np.ndarray, r: int) -> np.ndarray:
    """Compute U^r = evecs @ diag(evals^r) @ evecs_inv."""
    if r <= 0:
        return np.eye(len(evals), dtype=complex)
    return evecs @ np.diag(evals ** r) @ evecs_inv


def _unitary_at_time(h: float, time: int, params: dict,
                     H0: np.ndarray, Sx: np.ndarray, Sy: np.ndarray, Sz: np.ndarray,
                     uf_evals: np.ndarray, uf_evecs: np.ndarray,
                     uf_evecs_inv: np.ndarray) -> np.ndarray:
    """Compute U(time), mirroring calculate_unitary_at_time_mp."""
    nu = params["nu"]
    T = params["T"]
    steps = params["steps_floquet_unitary"]
    phi = params["phi"]
    varphi = params["varphi"]
    theta = params["theta"]
    phi_0 = params["phi_0"]

    t_delta = T / steps
    omega = 2.0 * np.pi / (nu * T)

    r = time // nu
    U = _find_power_r(uf_evals, uf_evecs, uf_evecs_inv, r)
    K = _kick_operator(phi, Sx)

    for p in range(r * nu + 1, time + 1):
        U = _evolution_T_step(U, h, T, varphi, theta, phi_0, H0, Sx, Sy, Sz,
                               omega, p, t_delta, steps)
        U = K @ U
    return U


def run_quspin_simulation(params_dict: dict, time_interval: list,
                           init_state: np.ndarray) -> List[QFIInformation]:
    """Run QFI dynamics simulation using QuSpin operators + numpy/scipy evolution.

    Mirrors simulation_with_AC_field_mp but uses double-precision numpy throughout.

    Parameters
    ----------
    params_dict : dict
        Parameters dict with keys: N, J, B, T, nu, phi, h, epsilon,
        steps_floquet_unitary, varphi, theta, phi_0.
    time_interval : list of int
        Floquet period indices at which to evaluate QFI.
    init_state : np.ndarray, shape (N+1,)
        Initial state vector (will be normalised if not already).

    Returns
    -------
    list of QFIInformation
    """
    N = params_dict["N"]
    h = float(params_dict["h"])
    epsilon = float(params_dict["epsilon"])

    Sz, Sx, Sy = get_spin_operators(N)
    H0 = build_h0(float(params_dict["J"]), float(params_dict["B"]), N, Sz, Sx)

    pf = dict(params_dict)   # local copy with float values
    pf["h"] = h
    pf["J"] = float(params_dict["J"])
    pf["B"] = float(params_dict["B"])
    pf["T"] = float(params_dict["T"])
    pf["phi"] = float(params_dict["phi"])
    pf["phi_0"] = float(params_dict["phi_0"])
    pf["varphi"] = float(params_dict["varphi"])
    pf["theta"] = float(params_dict["theta"])

    def _eig_uf(h_val: float):
        pf_h = dict(pf); pf_h["h"] = h_val
        UF = compute_floquet_unitary(h_val, pf_h, H0, Sx, Sy, Sz)
        evals, evecs = np.linalg.eig(UF)
        return evals, evecs, np.linalg.inv(evecs)

    uf_evals,   uf_evecs,   uf_einv   = _eig_uf(h)
    uf_p_evals, uf_p_evecs, uf_p_einv = _eig_uf(h + epsilon)
    uf_m_evals, uf_m_evecs, uf_m_einv = _eig_uf(h - epsilon)

    psi0 = np.asarray(init_state, dtype=complex)
    psi0 = psi0 / np.linalg.norm(psi0)

    results = []
    for time in time_interval:
        U   = _unitary_at_time(h,           time, pf, H0, Sx, Sy, Sz,
                                uf_evals,   uf_evecs,   uf_einv)
        U_p = _unitary_at_time(h + epsilon, time, pf, H0, Sx, Sy, Sz,
                                uf_p_evals, uf_p_evecs, uf_p_einv)
        U_m = _unitary_at_time(h - epsilon, time, pf, H0, Sx, Sy, Sz,
                                uf_m_evals, uf_m_evecs, uf_m_einv)

        ket   = U   @ psi0
        ket_p = U_p @ psi0
        ket_m = U_m @ psi0

        dket = (ket_p - ket_m) / (2.0 * epsilon)

        # QFI = 4 * ( <dket|dket> - |<ket|dket>|^2 )
        term1 = float(np.real(dket.conj() @ dket))
        term2 = abs(ket.conj() @ dket) ** 2
        qfi = 4.0 * (term1 - term2)

        m_x = float(np.real(ket.conj() @ (Sx @ ket))) / N
        m_y = float(np.real(ket.conj() @ (Sy @ ket))) / N
        m_z = float(np.real(ket.conj() @ (Sz @ ket))) / N

        results.append(QFIInformation(
            qfi_raw_value=str(qfi),
            qfi=float(qfi / (N ** 2 * time ** 2)),
            time=time,
            m_x=m_x,
            m_y=m_y,
            m_z=m_z,
        ))
    return results