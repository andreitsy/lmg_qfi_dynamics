"""Tests for the method-of-moments (MoM) sensitivity."""

import mpmath as mp
import pytest

from lmg_qfi import (
    UF,
    InitialState,
    SimulationParams,
    calculate_unitary_T,
    calculate_unitary_at_time_mp,
    create_hamiltonian_h0,
    create_mom_observables,
    create_parity_operator,
    create_spin_xyz_operators,
    dketa_t,
    method_of_moments_mp,
    quantum_fisher_information_mp,
    run_simulation,
)

N = 6
EPS = mp.mpf("1e-20")


def _params():
    return dict(phi=mp.pi, J=mp.mpf(1), B=mp.mpf("0.4"), T=mp.mpf(1), h=mp.mpf(0),
                epsilon=EPS, N=N, nu=2, varphi=mp.mpf(0), phi_0=mp.mpf(0),
                theta=mp.mpf(0), steps_floquet_unitary=6)


def _initial_states(H0):
    E, V = mp.eigh(H0)
    idx = sorted(range(N + 1), key=lambda i: E[i])
    gs_phys = V[:, idx[0]] + V[:, idx[1]]
    cat = mp.matrix([1] + [0] * (N - 1) + [1])
    return {
        "GS_CAT": V[:, idx[0]],
        "GS_PHYS": gs_phys / mp.norm(gs_phys),
        "PHYS": mp.matrix([1] + [0] * N),
        "CAT_SUM": cat / mp.norm(cat),
    }


def _uf(h, params, H0):
    e, v = mp.eig(calculate_unitary_T(h, params, H0))
    return UF(eigenvalues=e, U=v, U_inv=mp.inverse(v))


class TestObservables:
    def test_parity_is_tensor_sigma_x(self):
        with mp.workdps(30):
            _, Sx, _ = create_spin_xyz_operators(N)
            X = (mp.j ** N) * mp.expm(-mp.j * mp.pi * Sx)
            assert mp.mnorm(X - create_parity_operator(N), 1) < mp.mpf("1e-25")

    @pytest.mark.parametrize("n", [5, 6])
    def test_observables_hermitian(self, n):
        Sz, _, _ = create_spin_xyz_operators(n)
        for O in create_mom_observables(Sz).values():
            assert mp.mnorm(O - O.transpose_conj(), 1) < 1e-12

    def test_y_anticommutation(self):
        Sz, _, _ = create_spin_xyz_operators(N)
        X = create_parity_operator(N)
        assert mp.mnorm(Sz * X + X * Sz, 1) < 1e-12


class TestMethodOfMoments:
    def test_known_two_level_value(self):
        # |psi_h> = cos(h)|0> + sin(h)|1>, O = sigma_z at h = pi/8:
        # <O> = cos 2h, Var = sin^2 2h, d<O> = -2 sin 2h -> MoM = 4 = QFI.
        with mp.workdps(30):
            h = mp.pi / 8
            ket = mp.matrix([mp.cos(h), mp.sin(h)])
            dket = mp.matrix([-mp.sin(h), mp.cos(h)])
            O = mp.diag([1, -1])
            assert mp.almosteq(method_of_moments_mp(O, ket, dket), 4, 1e-20)

    def test_eigenstate_limit(self):
        # Same family at h = 0: |0> is an eigenstate of sigma_z, the limit is still 4.
        with mp.workdps(30):
            ket = mp.matrix([1, 0])
            dket = mp.matrix([0, 1])
            O = mp.diag([1, -1])
            assert mp.almosteq(method_of_moments_mp(O, ket, dket), 4, 1e-20)

    def test_bounded_by_qfi_and_limit_matches_small_h(self):
        with mp.workdps(40):
            params = _params()
            H0 = create_hamiltonian_h0(params["J"], params["B"], N)
            Sz, _, _ = create_spin_xyz_operators(N)
            observables = create_mom_observables(Sz)
            h1 = mp.mpf("1e-8")
            hs = [mp.mpf(0), EPS, -EPS, h1, h1 + EPS, h1 - EPS]
            ufs = {h: _uf(h, params, H0) for h in hs}

            def state(h, t, psi):
                return calculate_unitary_at_time_mp(h, t, params, H0, ufs[h]) * psi

            for name, psi in _initial_states(H0).items():
                for t in (3, 17):
                    k0 = state(hs[0], t, psi)
                    d0 = dketa_t(state(hs[1], t, psi), state(hs[2], t, psi), EPS)
                    k1 = state(h1, t, psi)
                    d1 = dketa_t(state(hs[4], t, psi), state(hs[5], t, psi), EPS)
                    qfi = quantum_fisher_information_mp(d0, k0)
                    for obs, O in observables.items():
                        m0 = method_of_moments_mp(O, k0, d0)
                        m1 = method_of_moments_mp(O, k1, d1)
                        assert m0 <= qfi * (1 + mp.mpf("1e-12")), (name, t, obs)
                        assert mp.almosteq(m0, m1, rel_eps=1e-4, abs_eps=1e-6), (name, t, obs)

    def test_parity_saturates_qfi_for_parity_eigenstates(self):
        # H(h=0) commutes with X and the S_z probe anticommutes with it, so for a
        # parity eigenstate the X measurement is optimal at h -> 0.
        with mp.workdps(40):
            params = _params()
            H0 = create_hamiltonian_h0(params["J"], params["B"], N)
            Sz, _, _ = create_spin_xyz_operators(N)
            X = create_mom_observables(Sz)["X"]
            ufs = {h: _uf(h, params, H0) for h in (mp.mpf(0), EPS, -EPS)}
            states = _initial_states(H0)
            for name in ("GS_CAT", "CAT_SUM"):
                psi = states[name]
                k = calculate_unitary_at_time_mp(mp.mpf(0), 9, params, H0, ufs[mp.mpf(0)]) * psi
                kp = calculate_unitary_at_time_mp(EPS, 9, params, H0, ufs[EPS]) * psi
                km = calculate_unitary_at_time_mp(-EPS, 9, params, H0, ufs[-EPS]) * psi
                d = dketa_t(kp, km, EPS)
                assert mp.almosteq(method_of_moments_mp(X, k, d),
                                   quantum_fisher_information_mp(d, k), rel_eps=1e-10)


def test_run_simulation_reports_mom():
    params = SimulationParams(
        {"dps": 30, "steps_floquet_unitary": 4, "num_points": 2},
        N=4, J=mp.mpf(1), B=mp.mpf("0.4"))
    results = run_simulation(params, [InitialState.CAT_SUM])[InitialState.CAT_SUM]
    for r in results[:20]:
        for value in (r.mom_sz, r.mom_x, r.mom_y):
            assert value == value  # not NaN
            assert 0 <= value <= r.qfi * (1 + 1e-9) + 1e-12
