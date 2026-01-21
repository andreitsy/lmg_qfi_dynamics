"""Quantum Fisher Information calculation functions."""

import logging
import mpmath as mp

from .config import QFIInformation, UF
from .operators import create_spin_xyz_operators
from .evolution import calculate_unitary_at_time_mp

DEBUG = True


def dketa_t(ket_t_p_delta, ket_t_m_delta, delta):
    """Compute the derivative of the ket using finite differences."""
    return (ket_t_p_delta - ket_t_m_delta) / (2.0 * delta)


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
        time: int,
        params: dict,
        H_0: mp.matrix,
        floque_u: UF,
        floque_u_p: UF,
        floque_u_m: UF,
        init_state: mp.matrix,
        Zsum: mp.matrix,
        Xsum: mp.matrix,
        Ysum: mp.matrix,
) -> QFIInformation:
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
    dket = dketa_t(ket_t_p_delta, ket_t_m_delta, epsilon)
    qfi = quantum_fisher_information_mp(dket, ket_t)
    
    if DEBUG:
        abs_error_estimate = calculate_error_estimation_mp(dket, ket_t)
        if abs_error_estimate > 0.1:
            logging.warning("ABS_ERROR: %f", abs_error_estimate)

    # Compute magnetizations
    m_x = mp.re((ket_t.transpose_conj() * Xsum * ket_t)[0, 0]) / N
    m_y = mp.re((ket_t.transpose_conj() * Ysum * ket_t)[0, 0]) / N
    m_z = mp.re((ket_t.transpose_conj() * Zsum * ket_t)[0, 0]) / N

    return QFIInformation(
        qfi_raw_value=str(qfi),
        qfi=float(qfi / (N ** 2 * time ** 2)),
        time=time,
        m_x=float(m_x),
        m_y=float(m_y),
        m_z=float(m_z),
    )
