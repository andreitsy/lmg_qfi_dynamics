"""Scientific checks for an SLD measured away from its calibration field."""

import mpmath as mp
import pytest

from lmg_qfi.mom import method_of_moments_mp
from lmg_qfi.qfi import quantum_fisher_information_mp
from lmg_qfi.sld import compute_sld_matrix


def _latitude_state(h):
    """A qubit latitude with QFI 3/4 and a nonzero phase tangent."""
    phase = mp.exp(mp.j * h)
    return mp.matrix([mp.sqrt(3) / 2, phase / 2]), mp.matrix([0, mp.j * phase / 2])


def _mean(observable, ket):
    return mp.re((ket.transpose_conj() * observable * ket)[0])


@pytest.mark.parametrize("calibration", ["0", "0.001", "0.7"])
def test_matched_canonical_sld_saturates_qfi(calibration):
    with mp.workdps(50):
        ket, dket = _latitude_state(mp.mpf(calibration))
        observable = compute_sld_matrix(dket, ket)
        rho = ket * ket.transpose_conj()
        drho = dket * ket.transpose_conj() + ket * dket.transpose_conj()
        tol = mp.mpf("1e-45")

        assert mp.norm(observable - observable.transpose_conj()) < tol
        assert mp.norm((rho * observable + observable * rho) / 2 - drho) < tol
        assert abs(_mean(observable, ket)) < tol
        assert mp.almosteq(quantum_fisher_information_mp(dket, ket), mp.mpf(3) / 4,
                           abs_eps=tol)
        assert mp.almosteq(method_of_moments_mp(observable, ket, dket), mp.mpf(3) / 4,
                           abs_eps=tol)


def test_mismatched_sld_has_known_suboptimal_sensitivity():
    # For this latitude, M_0[L_c] = (3/4) cos(c)^2 / (1 - (3/4) sin(c)^2).
    # At c = pi/3 this is 3/7, strictly below the target QFI 3/4.
    with mp.workdps(50):
        calibration_ket, calibration_dket = _latitude_state(mp.pi / 3)
        observable = compute_sld_matrix(calibration_dket, calibration_ket)
        target_ket, target_dket = _latitude_state(mp.mpf(0))
        sensitivity = method_of_moments_mp(observable, target_ket, target_dket)

        assert mp.almosteq(sensitivity, mp.mpf(3) / 7, abs_eps=mp.mpf("1e-45"))
        assert sensitivity < quantum_fisher_information_mp(target_dket, target_ket)
        # The calibrated SLD's mean vanishes only on its calibration state.
        assert mp.almosteq(_mean(observable, target_ket), -3 * mp.sqrt(3) / 8,
                           abs_eps=mp.mpf("1e-45"))


def test_expectation_derivative_keeps_calibrated_observable_fixed():
    with mp.workdps(50):
        calibration = mp.mpf("0.001")
        cal_ket, cal_dket = _latitude_state(calibration)
        observable = compute_sld_matrix(cal_dket, cal_ket)
        target_ket, target_dket = _latitude_state(mp.mpf(0))
        epsilon = mp.mpf("1e-12")
        ket_plus, _ = _latitude_state(epsilon)
        ket_minus, _ = _latitude_state(-epsilon)
        finite_difference = (_mean(observable, ket_plus) - _mean(observable, ket_minus)) / (2 * epsilon)
        tangent_signal = 2 * mp.re((target_ket.transpose_conj() * observable * target_dket)[0])
        expected = mp.mpf(3) / 4 * mp.cos(calibration)

        assert mp.almosteq(finite_difference, expected, abs_eps=mp.mpf("1e-24"))
        assert mp.almosteq(tangent_signal, expected, abs_eps=mp.mpf("1e-45"))
        variance = _mean(observable * observable, target_ket) - _mean(observable, target_ket) ** 2
        assert mp.almosteq(finite_difference ** 2 / variance,
                           method_of_moments_mp(observable, target_ket, target_dket),
                           abs_eps=mp.mpf("1e-24"))


def test_cached_evolution_matches_direct_chronological_derivative():
    """Check both cycle powers and odd-time prefixes against direct propagation."""
    from lmg_qfi.evolution import evalution_T_step
    from lmg_qfi.operators import (
        create_hamiltonian_h0,
        create_kick_operator,
        create_spin_xyz_operators,
    )
    from scripts.plot_sld_mismatch import StateDerivative

    with mp.workdps(40):
        params = dict(N=2, J=mp.mpf(1), B=mp.mpf("0.4"), T=mp.mpf(1),
                      phi=mp.pi, nu=2, varphi=mp.mpf(0), theta=mp.mpf(0),
                      phi_0=mp.mpf(0), steps_floquet_unitary=4)
        h = mp.mpf("0.001")
        epsilon = mp.mpf("1e-12")
        initial = mp.matrix([[1, 1 / mp.sqrt(2)], [0, 0], [0, 1 / mp.sqrt(2)]])
        cached = StateDerivative(params, initial, h, epsilon=epsilon)
        h0 = create_hamiltonian_h0(params["J"], params["B"], params["N"])
        sz, sx, sy = create_spin_xyz_operators(params["N"])
        kick = create_kick_operator(params["phi"], sx)
        omega = 2 * mp.pi / (params["nu"] * params["T"])
        times = (1, 2, 5)
        direct = {}

        for field in (h, h + epsilon, h - epsilon):
            unitary = mp.eye(3)
            for period in range(1, max(times) + 1):
                unitary = kick * evalution_T_step(
                    unitary, field, params["T"], params["varphi"], params["theta"],
                    params["phi_0"], h0, sx, sy, sz, omega, period,
                    params["T"] / params["steps_floquet_unitary"],
                    params["steps_floquet_unitary"],
                )
                if period in times:
                    direct[field, period] = unitary * initial

        for time in times:
            kets, derivatives = cached.evaluate(time)
            expected_derivative = (direct[h + epsilon, time] - direct[h - epsilon, time]) / (2 * epsilon)
            assert mp.norm(kets - direct[h, time]) < mp.mpf("1e-30")
            assert mp.norm(derivatives - expected_derivative) < mp.mpf("1e-24")


def test_small_mismatch_run_preserves_local_sld_identity_and_qfi_bound():
    import numpy as np

    from lmg_qfi.config import SimulationParams
    from scripts.plot_sld_mismatch import compute_sld_mismatch

    with mp.workdps(40):
        params = SimulationParams(
            {"steps_floquet_unitary": 4, "num_points": 2, "dps": 40},
            N=2, J=mp.mpf(1), B=mp.mpf("0.4"), phi_kick_phase=mp.pi,
        )
        simulations, _ = compute_sld_mismatch(params, [1, 2, 5], dps=40)

    assert set(simulations) == {"PHYS", "CAT_SUM", "GS_CAT", "GS_PHYS"}
    for frame in simulations.values():
        assert frame["time"].tolist() == [1, 2, 5]
        assert np.isfinite(frame[["qfi", "mom_sld_ref", "mom_sld_local", "qfi_reference",
                                   "mom_sld_reference_matched"]].to_numpy(dtype=float)).all()
        np.testing.assert_allclose(frame["mom_sld_local"], frame["qfi"], rtol=1e-11, atol=1e-14)
        np.testing.assert_allclose(frame["mom_sld_reference_matched"], frame["qfi_reference"],
                                   rtol=1e-11, atol=1e-14)
        assert (frame["mom_sld_ref"] >= 0).all()
        assert (frame["mom_sld_ref"] <= frame["qfi"] * (1 + 1e-11) + 1e-14).all()
