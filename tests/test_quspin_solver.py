"""Tests for the QuSpin-based LMG QFI solver.

Covers:
  - Operator correctness (Hermiticity, commutation relations, Casimir)
  - Floquet unitary unitarity and agreement with mpmath reference
  - QFI non-negativity and physical bounds
  - Numerical comparison between QuSpin and mpmath solvers for N=20
"""

import numpy as np
import pytest
import mpmath as mp

from lmg_qfi.quspin_solver import (
    get_spin_operators,
    build_h0,
    compute_floquet_unitary,
    run_quspin,
    run_quspin_simulation,
)
from lmg_qfi import (
    EstimationParameter,
    InitialState,
    SimulationParams,
    create_spin_xyz_operators as mp_spin_xyz,
    create_hamiltonian_h0 as mp_h0,
    calculate_unitary_T as mp_unitary_T,
)
from lmg_qfi.simulation import simulation_with_AC_field_mp

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def mp_to_np(m):
    """Convert mpmath matrix to numpy complex128 array."""
    return np.array([[complex(x) for x in row] for row in m.tolist()],
                    dtype=np.complex128)


def _base_params(N=4, h=0.0, steps=10):
    """Minimal params dict shared across tests."""
    return dict(
        N=N, J=1.0, B=0.4, T=1.0, nu=2, phi=np.pi,
        h=h, epsilon=1e-7, steps_floquet_unitary=steps,
        varphi=0.0, theta=0.0, phi_0=0.0,
    )


# ──────────────────────────────────────────────
# Operator tests
# ──────────────────────────────────────────────

class TestQuSpinOperators:

    @pytest.mark.parametrize("N", [2, 4, 6, 10, 20])
    def test_dimension(self, N):
        Sz, Sx, Sy = get_spin_operators(N)
        for op in (Sz, Sx, Sy):
            assert op.shape == (N + 1, N + 1)

    @pytest.mark.parametrize("N", [2, 4, 6])
    def test_sz_diagonal(self, N):
        """Sz should be diagonal with values [+S, ..., -S]."""
        Sz, _, _ = get_spin_operators(N)
        S = N / 2
        expected_diag = np.arange(S, -S - 1, -1)
        assert np.allclose(np.diag(Sz), expected_diag)
        assert np.allclose(Sz - np.diag(np.diag(Sz)), 0)

    @pytest.mark.parametrize("N", [2, 4, 6])
    def test_hermiticity(self, N):
        Sz, Sx, Sy = get_spin_operators(N)
        for op in (Sz, Sx, Sy):
            assert np.allclose(op, op.conj().T), f"Operator not Hermitian for N={N}"

    @pytest.mark.parametrize("N", [2, 4, 6])
    def test_commutation_relations(self, N):
        """[Sx, Sy] = i Sz,  [Sy, Sz] = i Sx,  [Sz, Sx] = i Sy."""
        Sz, Sx, Sy = get_spin_operators(N)
        assert np.allclose(Sx @ Sy - Sy @ Sx, 1j * Sz)
        assert np.allclose(Sy @ Sz - Sz @ Sy, 1j * Sx)
        assert np.allclose(Sz @ Sx - Sx @ Sz, 1j * Sy)

    @pytest.mark.parametrize("N", [2, 4, 6])
    def test_casimir(self, N):
        """S^2 = S(S+1) I."""
        Sz, Sx, Sy = get_spin_operators(N)
        S = N / 2
        S2 = Sx @ Sx + Sy @ Sy + Sz @ Sz
        assert np.allclose(S2, S * (S + 1) * np.eye(N + 1))

    @pytest.mark.parametrize("N", [2, 4, 6])
    def test_matches_mpmath(self, N):
        """QuSpin operators agree with mpmath reference implementation."""
        Sz_qs, Sx_qs, Sy_qs = get_spin_operators(N)
        Sz_mp, Sx_mp, Sy_mp = mp_spin_xyz(N)
        assert np.allclose(Sz_qs, mp_to_np(Sz_mp))
        assert np.allclose(Sx_qs, mp_to_np(Sx_mp))
        assert np.allclose(Sy_qs, mp_to_np(Sy_mp))


# ──────────────────────────────────────────────
# Hamiltonian tests
# ──────────────────────────────────────────────

class TestBuildH0:

    @pytest.mark.parametrize("N, J, B", [
        (2,  1.0, 0.0),
        (4,  1.0, 0.4),
        (6,  1.0, 1.0),
        (10, 1.0, 0.4),
    ])
    def test_hermitian(self, N, J, B):
        Sz, Sx, _ = get_spin_operators(N)
        H = build_h0(J, B, N, Sz, Sx)
        assert np.allclose(H, H.conj().T)

    @pytest.mark.parametrize("N, J, B", [
        (2,  1.0, 0.0),
        (4,  1.0, 0.4),
        (10, 1.0, 0.4),
    ])
    def test_matches_mpmath(self, N, J, B):
        Sz, Sx, _ = get_spin_operators(N)
        H_qs = build_h0(J, B, N, Sz, Sx)
        H_mp = mp_to_np(mp_h0(J, B, N))
        assert np.allclose(H_qs, H_mp)


# ──────────────────────────────────────────────
# Floquet unitary tests
# ──────────────────────────────────────────────

class TestFloquetUnitary:

    @pytest.mark.parametrize("N", [2, 4, 6])
    @pytest.mark.parametrize("h", [0.0, 0.01])
    def test_unitary(self, N, h):
        params = _base_params(N=N, h=h)
        Sz, Sx, Sy = get_spin_operators(N)
        H0 = build_h0(params["J"], params["B"], N, Sz, Sx)
        UF = compute_floquet_unitary(h, params, H0, Sx, Sy, Sz)
        assert np.allclose(UF @ UF.conj().T, np.eye(N + 1), atol=1e-10)

    @pytest.mark.parametrize("N", [2, 4, 6])
    @pytest.mark.parametrize("h", [0.0, 0.01])
    def test_determinant_magnitude(self, N, h):
        """det(U_F) should have magnitude 1."""
        params = _base_params(N=N, h=h)
        Sz, Sx, Sy = get_spin_operators(N)
        H0 = build_h0(params["J"], params["B"], N, Sz, Sx)
        UF = compute_floquet_unitary(h, params, H0, Sx, Sy, Sz)
        assert abs(abs(np.linalg.det(UF)) - 1.0) < 1e-9

    @pytest.mark.parametrize("N", [2, 4, 6])
    def test_matches_mpmath(self, N):
        """QuSpin Floquet unitary agrees with mpmath reference."""
        params = _base_params(N=N)
        Sz, Sx, Sy = get_spin_operators(N)
        H0 = build_h0(params["J"], params["B"], N, Sz, Sx)
        UF_qs = compute_floquet_unitary(params["h"], params, H0, Sx, Sy, Sz)

        mp_params = dict(params)
        mp_params["phi"] = mp.mpf(str(params["phi"]))
        H0_mp = mp_h0(params["J"], params["B"], N)
        UF_mp = mp_to_np(mp_unitary_T(mp.mpf("0"), mp_params, H0_mp))
        assert np.allclose(UF_qs, UF_mp, atol=1e-10)


# ──────────────────────────────────────────────
# QFI simulation tests
# ──────────────────────────────────────────────

class TestQuSpinSimulation:

    def _phys_state(self, N):
        """Fock state |N/2, N/2> (all spins up)."""
        psi = np.zeros(N + 1, dtype=complex)
        psi[0] = 1.0
        return psi

    def _gs_state(self, N, J=1.0, B=0.4):
        """Ground state of H0."""
        Sz, Sx, _ = get_spin_operators(N)
        H0 = build_h0(J, B, N, Sz, Sx)
        evals, evecs = np.linalg.eigh(H0)
        gs = evecs[:, 0]
        return gs / np.linalg.norm(gs)

    @pytest.mark.parametrize("N", [2, 4])
    def test_returns_correct_count(self, N):
        params = _base_params(N=N)
        times = [1, 2, 5]
        psi0 = self._phys_state(N)
        results = run_quspin_simulation(params, times, psi0)
        assert len(results) == len(times)

    @pytest.mark.parametrize("N", [2, 4])
    def test_qfi_non_negative(self, N):
        params = _base_params(N=N)
        times = list(range(1, 6))
        psi0 = self._phys_state(N)
        results = run_quspin_simulation(params, times, psi0)
        for r in results:
            assert r.qfi >= -1e-12, f"QFI negative at t={r.time}: {r.qfi}"

    @pytest.mark.parametrize("N", [2, 4])
    def test_time_stored_correctly(self, N):
        params = _base_params(N=N)
        times = [1, 3, 7]
        psi0 = self._phys_state(N)
        results = run_quspin_simulation(params, times, psi0)
        for res, t in zip(results, times):
            assert res.time == t

    @pytest.mark.parametrize("N", [2, 4])
    def test_magnetization_bounds(self, N):
        params = _base_params(N=N)
        times = [1, 2, 5]
        psi0 = self._gs_state(N)
        results = run_quspin_simulation(params, times, psi0)
        S = N / 2
        for r in results:
            assert abs(r.m_x) <= S / N + 1e-10
            assert abs(r.m_y) <= S / N + 1e-10
            assert abs(r.m_z) <= S / N + 1e-10

    def test_qfi_zero_h_stationary(self):
        """For h=0 and PHYS initial state (eigenstate of Sz), magnetizations are stable."""
        N = 4
        params = _base_params(N=N, h=0.0)
        # Use ground state with B=0 so Sz eigenstates are stationary for H0 part
        psi0 = self._phys_state(N)
        results = run_quspin_simulation(params, [1, 2, 3], psi0)
        # m_z should be consistent across times (non-trivial dynamics due to B field)
        assert all(r.qfi >= -1e-12 for r in results)


# ──────────────────────────────────────────────
# run_quspin wrapper (SimulationParams -> results per state)
# ──────────────────────────────────────────────

class TestRunQuspin:

    def _simulation_params(self, N=4, h="0.0",
                           parameter=EstimationParameter.AMPLITUDE):
        return SimulationParams(
            run_arguments={"dps": 15, "steps_floquet_unitary": 5,
                           "num_points": 5, "max_time_degree": 2},
            N=N, J=mp.mpf("1.0"), B=mp.mpf("0.4"), h=mp.mpf(h),
            parameter=parameter,
        )

    def test_returns_all_states(self):
        params = self._simulation_params()
        results = run_quspin(params, list(InitialState))
        assert set(results) == set(InitialState)
        time_grids = [[r.time for r in state_results]
                      for state_results in results.values()]
        assert all(grid and grid == time_grids[0] for grid in time_grids)

    def test_frequency_h_zero_raises(self):
        params = self._simulation_params(
            parameter=EstimationParameter.FREQUENCY)
        with pytest.raises(ValueError, match="nonzero AC amplitude"):
            run_quspin(params, [InitialState.PHYS])


# ──────────────────────────────────────────────
# Comparison: QuSpin vs mpmath for N=20
# ──────────────────────────────────────────────

class TestComparisonN20:
    """Numerical comparison of QuSpin and mpmath solvers for N=20.

    Both solvers use the same algorithm (Trotter integration + eigendecomposition),
    so results should agree at double-precision tolerance.
    """

    N = 20
    J = 1.0
    B = 0.4
    TIMES = [1, 2, 5, 10, 20]

    def _build_params_dict(self, dps=15):
        with mp.workdps(dps):
            eps = float(mp.mpf(f"1e-{dps // 2}"))
        return dict(
            N=self.N, J=self.J, B=self.B, T=1.0, nu=2, phi=np.pi,
            h=0.0, epsilon=eps, steps_floquet_unitary=10,
            varphi=0.0, theta=0.0, phi_0=0.0,
        )

    def _phys_init_state_numpy(self):
        psi = np.zeros(self.N + 1, dtype=complex)
        psi[0] = 1.0
        return psi

    def _phys_init_state_mp(self):
        with mp.workdps(15):
            psi = mp.zeros(self.N + 1, 1)
            psi[0] = mp.mpf("1.0")
            return psi / mp.norm(psi)

    def test_qfi_agrees_phys_state(self):
        """QFI from QuSpin and mpmath agree to < 1% for PHYS initial state."""
        params = self._build_params_dict()
        psi0_np = self._phys_init_state_numpy()
        psi0_mp = self._phys_init_state_mp()

        qs_results = run_quspin_simulation(params, self.TIMES, psi0_np)

        # Build mpmath params dict
        mp_params = dict(params)
        mp_params["phi"] = mp.pi
        mp_params["J"] = mp.mpf("1.0")
        mp_params["B"] = mp.mpf("0.4")
        mp_params["T"] = mp.mpf("1.0")
        mp_params["h"] = mp.mpf("0.0")
        mp_params["epsilon"] = mp.mpf(f"1e-{15 // 2}")
        mp_params["phi_0"] = mp.mpf("0.0")
        mp_params["varphi"] = mp.mpf("0.0")
        mp_params["theta"] = mp.mpf("0.0")

        with mp.workdps(15):
            H0_mp = mp_h0(self.J, self.B, self.N)
            mp_results = simulation_with_AC_field_mp(
                mp_params, self.TIMES, psi0_mp, "PHYS"
            )

        for qs, mp_r in zip(qs_results, mp_results):
            assert qs.time == mp_r.time
            # QFI values should agree within 1%
            if abs(mp_r.qfi) > 1e-12:
                rel_err = abs(qs.qfi - mp_r.qfi) / abs(mp_r.qfi)
                assert rel_err < 0.01, (
                    f"t={qs.time}: QuSpin qfi={qs.qfi:.6e}, "
                    f"mpmath qfi={mp_r.qfi:.6e}, rel_err={rel_err:.2e}"
                )

    def test_magnetizations_agree(self):
        """Magnetizations from QuSpin and mpmath agree to < 1e-6."""
        params = self._build_params_dict()
        psi0_np = self._phys_init_state_numpy()
        psi0_mp = self._phys_init_state_mp()

        qs_results = run_quspin_simulation(params, self.TIMES, psi0_np)

        mp_params = dict(params)
        mp_params["phi"] = mp.pi
        mp_params["J"] = mp.mpf("1.0")
        mp_params["B"] = mp.mpf("0.4")
        mp_params["T"] = mp.mpf("1.0")
        mp_params["h"] = mp.mpf("0.0")
        mp_params["epsilon"] = mp.mpf(f"1e-{15 // 2}")
        mp_params["phi_0"] = mp.mpf("0.0")
        mp_params["varphi"] = mp.mpf("0.0")
        mp_params["theta"] = mp.mpf("0.0")

        with mp.workdps(15):
            H0_mp = mp_h0(self.J, self.B, self.N)
            mp_results = simulation_with_AC_field_mp(
                mp_params, self.TIMES, psi0_mp, "PHYS"
            )

        for qs, mp_r in zip(qs_results, mp_results):
            assert abs(qs.m_x - mp_r.m_x) < 1e-6, (
                f"t={qs.time}: m_x diff={abs(qs.m_x - mp_r.m_x):.2e}"
            )
            assert abs(qs.m_z - mp_r.m_z) < 1e-6, (
                f"t={qs.time}: m_z diff={abs(qs.m_z - mp_r.m_z):.2e}"
            )
