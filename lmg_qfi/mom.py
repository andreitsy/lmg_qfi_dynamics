"""Method of moments (MoM) sensitivity for simple observables.

For an observable O measured on |psi_h(t)>, the method-of-moments estimator of h has
inverse variance

    MoM(h, t) = (d_h <O>)^2 / Var(O),    Var(O) = <O^2> - <O>^2,

which is bounded by the QFI, MoM <= F_h. The observables follow the paper on optimal
observables for Floquet-time-crystal sensors:

* ``Sz``     -- bare magnetization S_z;
* ``X``      -- spin-flip parity X = (sigma^x)^{otimes N}, i.e. |S, m> -> |S, -m> in the
                Dicke basis (equal to exp(-i pi S_x) up to the global phase i^N);
* ``Y``      -- dressed parity-magnetization Y = i S_z X (Hermitian since {S_z, X} = 0).

``d_h <O> = 2 Re <psi|O|d_h psi>`` uses the same finite-difference ket derivative as the QFI.

If |psi> is an eigenstate of O (Var(O) = 0), e.g. the parity of a parity-symmetric state at
h = 0, the ratio is 0/0. Its h -> h0 limit is finite and is used instead: with
D = O - <O>,  MoM -> 4 <dpsi|D|dpsi>^2 / <dpsi|D^2|dpsi>.
"""

import mpmath as mp

MOM_OBSERVABLES = ("Sz", "X", "Y")


def create_parity_operator(n):
    """Spin-flip parity X = (sigma^x)^{otimes n} in the Dicke basis (anti-diagonal ones)."""
    X = mp.matrix(n + 1)
    for i in range(n + 1):
        X[i, n - i] = mp.mpc(1)
    return X


def create_mom_observables(Sz):
    """Return {"Sz": Sz, "X": X, "Y": i Sz X} for a Dicke-basis Sz of dimension n + 1."""
    n = Sz.rows - 1
    X = create_parity_operator(n)
    return {"Sz": Sz, "X": X, "Y": mp.j * Sz * X}


def _expectation(bra, O, ket):
    return (bra.transpose_conj() * O * ket)[0, 0]


def method_of_moments_mp(O, ket_t, dket, var_rel_tol=None):
    """
    Method-of-moments sensitivity (d_h <O>)^2 / Var(O) for a pure state.

    Parameters
    ----------
    O : mp.matrix
        Hermitian observable.
    ket_t : mp.matrix
        Normalized state |psi_h(t)>.
    dket : mp.matrix
        Derivative d_h |psi_h(t)>.
    var_rel_tol : mp.mpf, optional
        Var(O) <= var_rel_tol * <O^2> is treated as zero (state is an eigenstate of O) and the
        h -> h0 limit is returned. Defaults to 10^(-dps/2).

    Returns
    -------
    mp.mpf
        Raw (unnormalized) MoM; ``mp.nan`` if both signal and noise vanish identically.
    """
    if var_rel_tol is None:
        var_rel_tol = mp.mpf(10) ** (-(mp.mp.dps // 2))

    O_ket = O * ket_t
    mean = mp.re((ket_t.transpose_conj() * O_ket)[0, 0])
    second = mp.re((O_ket.transpose_conj() * O_ket)[0, 0])
    var = second - mean ** 2

    if var > var_rel_tol * max(second, mp.mpf(1)):
        signal = 2 * mp.re((O_ket.transpose_conj() * dket)[0, 0])
        return signal ** 2 / var

    # Eigenstate of O: second-order limit in (h - h0).
    D_dket = O * dket - mean * dket
    num = mp.re((dket.transpose_conj() * D_dket)[0, 0])
    den = mp.re((D_dket.transpose_conj() * D_dket)[0, 0])
    if den <= 0:
        return mp.nan
    return 4 * num ** 2 / den
