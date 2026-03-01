"""Symmetric Logarithmic Derivative (SLD) operator functions."""

import mpmath as mp

from .config import SLDInformation


def compute_sld_matrix(dket_t: mp.matrix, ket_t: mp.matrix) -> mp.matrix:
    """
    Compute SLD matrix L = 2(|∂ψ⟩⟨ψ| + |ψ⟩⟨∂ψ|).

    For a pure state ρ = |ψ⟩⟨ψ|, this satisfies the SLD equation
    ∂ρ/∂θ = (ρL + Lρ)/2 when the state is normalized.

    Parameters
    ----------
    dket_t : mp.matrix
        Derivative ket |∂ψ⟩, shape (n+1, 1).
    ket_t : mp.matrix
        State ket |ψ⟩, shape (n+1, 1).

    Returns
    -------
    L : mp.matrix
        Hermitian SLD matrix, shape (n+1, n+1).
    """
    bra_psi = ket_t.transpose_conj()    # (1, n+1)
    bra_dpsi = dket_t.transpose_conj()  # (1, n+1)
    return 2 * (dket_t * bra_psi + ket_t * bra_dpsi)


def sld_expectation(dket_t: mp.matrix, ket_t: mp.matrix) -> mp.mpf:
    """
    Compute ⟨L⟩ = 4 Re(⟨ψ|∂ψ⟩).

    For a normalized state, d/dθ ⟨ψ|ψ⟩ = 0 implies Re(⟨ψ|∂ψ⟩) = 0,
    so ⟨L⟩ = 0. A non-zero value indicates numerical loss of normalization.

    Parameters
    ----------
    dket_t : mp.matrix
        Derivative ket |∂ψ⟩, shape (n+1, 1).
    ket_t : mp.matrix
        State ket |ψ⟩, shape (n+1, 1).

    Returns
    -------
    mp.mpf
        Expectation value ⟨L⟩.
    """
    alpha = (ket_t.transpose_conj() * dket_t)[0, 0]
    return 4 * mp.re(alpha)


def sld_squared_expectation(dket_t: mp.matrix, ket_t: mp.matrix) -> mp.mpf:
    """
    Compute ⟨L²⟩ = Tr(ρ L²) without constructing the full N×N matrix.

    Uses the identity L|ψ⟩ = 2(|∂ψ⟩ + ⟨∂ψ|ψ⟩|ψ⟩) and
    ⟨ψ|L²|ψ⟩ = ‖L|ψ⟩‖² (from Hermiticity of L).

    For a normalized state, this equals the QFI:
        ⟨L²⟩ = 4(⟨∂ψ|∂ψ⟩ - |⟨ψ|∂ψ⟩|²)

    Parameters
    ----------
    dket_t : mp.matrix
        Derivative ket |∂ψ⟩, shape (n+1, 1).
    ket_t : mp.matrix
        State ket |ψ⟩, shape (n+1, 1).

    Returns
    -------
    mp.mpf
        Expectation value ⟨L²⟩ (equals raw QFI for normalized states).
    """
    alpha_conj = (dket_t.transpose_conj() * ket_t)[0, 0]  # ⟨∂ψ|ψ⟩ = α*
    L_ket = 2 * (dket_t + alpha_conj * ket_t)
    return mp.re((L_ket.transpose_conj() * L_ket)[0, 0])


def compute_sld_info(
    dket_t: mp.matrix,
    ket_t: mp.matrix,
    N: int,
    time: int,
) -> SLDInformation:
    """
    Compute all SLD scalar observables at a single time point.

    Parameters
    ----------
    dket_t : mp.matrix
        Derivative ket |∂ψ(t)⟩, shape (n+1, 1).
    ket_t : mp.matrix
        Evolved state ket |ψ(t)⟩, shape (n+1, 1).
    N : int
        Number of spins (system size).
    time : int
        Floquet period index (must be > 0).

    Returns
    -------
    SLDInformation
        Contains ⟨L⟩, ⟨L²⟩, and normalized QFI from SLD.
    """
    L_exp = sld_expectation(dket_t, ket_t)
    L2_exp = sld_squared_expectation(dket_t, ket_t)
    return SLDInformation(
        time=time,
        L_expectation=float(L_exp),
        L_squared_expectation=float(L2_exp),
        qfi_from_sld=float(L2_exp / (N ** 2 * time ** 2)),
    )
