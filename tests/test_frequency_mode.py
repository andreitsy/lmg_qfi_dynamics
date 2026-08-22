"""Tests for the frequency-estimation (QFI over omega) mode."""

import numpy as np
import pytest
import mpmath as mp

from lmg_qfi import (
    EstimationParameter,
    SimulationParams,
    SolverType,
    add_simulation_cli_arguments,
    apply_simulation_cli_arguments,
    calculate_unitary_T,
    calculate_unitary_at_time_mp,
    create_hamiltonian_h0,
    create_kick_operator,
    create_spin_xyz_operators,
    evolve_kets_one_period,
    frequency_estimation_epsilon,
    generate_time_interval,
    read_defaults_args_from_config,
    resolve_omega,
    run_simulation,
    simulation_with_AC_field_mp,
    simulation_with_AC_field_frequency_mp,
    UF,
)
from lmg_qfi.quspin_solver import (
    _evolution_T_step,
    _kick_operator,
    _resolve_omega,
    build_h0,
    build_initial_states,
    compute_floquet_unitary,
    get_spin_operators,
    run_quspin_simulation,
)

DPS = 15


def _mp_params(n, h="0.1", nu=2, steps=10, epsilon="1e-7", **overrides):
    params = dict(
        N=n,
        J=mp.mpf("1.0"),
        B=mp.mpf("0.4"),
        T=mp.mpf("1.0"),
        nu=nu,
        phi=mp.pi,
        h=mp.mpf(h),
        epsilon=mp.mpf(epsilon),
        steps_floquet_unitary=steps,
        varphi=mp.mpf("0.0"),
        theta=mp.mpf("0.2"),
        phi_0=mp.mpf("0.1"),
    )
    params.update(overrides)
    return params


def _qs_params(n, h=0.1, nu=2, steps=10, epsilon=1e-7, **overrides):
    params = dict(
        N=n, J=1.0, B=0.4, T=1.0, nu=nu, phi=float(np.pi), h=h,
        epsilon=epsilon, steps_floquet_unitary=steps,
        varphi=0.0, theta=0.2, phi_0=0.1,
    )
    params.update(overrides)
    return params


def _mp_phys_state(n):
    state = mp.zeros(n + 1, 1)
    state[0] = mp.mpf("1.0")
    return state


class TestSequentialDriverMatchesFloquetPowers:
    """At exactly resonant omega the sequential propagation must reproduce
    the U_F-power evolution used by the amplitude fast path."""

    @pytest.mark.parametrize("n", [2, 5])
    def test_mpmath_kets_match(self, n):
        """Sequential mpmath kets equal calculate_unitary_at_time_mp * psi0."""
        with mp.workdps(DPS):
            params = _mp_params(n)
            H_0 = create_hamiltonian_h0(1.0, 0.4, n)
            Zsum, Xsum, Ysum = create_spin_xyz_operators(n)
            kick = create_kick_operator(params["phi"], Xsum)
            omega_0 = resolve_omega(params)

            eigenvalues, eigenvectors = mp.eig(
                calculate_unitary_T(params["h"], params, H_0))
            floque_u = UF(eigenvalues=eigenvalues, U=eigenvectors,
                          U_inv=mp.inverse(eigenvectors))

            psi0 = _mp_phys_state(n)
            kets = [mp.matrix(psi0)]
            for p in range(1, 9):
                kets = evolve_kets_one_period(
                    kets, [omega_0], p, params["h"], params,
                    H_0, Xsum, Ysum, Zsum, kick)
                reference = calculate_unitary_at_time_mp(
                    params["h"], p, params, H_0, floque_u) * psi0
                assert mp.norm(kets[0] - reference) < mp.mpf("1e-12")

    @pytest.mark.parametrize("n", [2, 5])
    def test_quspin_kets_match(self, n):
        """Sequential QuSpin kets equal the Floquet-power unitary applied to psi0."""
        params = _qs_params(n)
        Sz, Sx, Sy = get_spin_operators(n)
        H0 = build_h0(params["J"], params["B"], n, Sz, Sx)
        K = _kick_operator(params["phi"], Sx)
        omega_0 = _resolve_omega(params)
        t_delta = params["T"] / params["steps_floquet_unitary"]

        UF_matrix = compute_floquet_unitary(params["h"], params, H0, Sx, Sy, Sz)
        psi0 = np.zeros(n + 1, dtype=complex)
        psi0[0] = 1.0

        ket = psi0.copy()
        unitary = np.eye(n + 1, dtype=complex)
        for p in range(1, 2 * params["nu"] + 1):
            ket = K @ _evolution_T_step(
                ket, params["h"], params["T"], params["varphi"], params["theta"],
                params["phi_0"], H0, Sx, Sy, Sz, omega_0, p, t_delta,
                params["steps_floquet_unitary"])
            if p % params["nu"] == 0:
                unitary = UF_matrix @ unitary
                assert np.linalg.norm(ket - unitary @ psi0) < 1e-12


class TestFrequencyModeValidation:
    """Input validation of the frequency-estimation drivers."""

    def test_mpmath_h_zero_raises(self):
        """h = 0 leaves no omega dependence: the mpmath driver must refuse."""
        with mp.workdps(DPS):
            params = _mp_params(2, h="0.0")
            with pytest.raises(ValueError, match="nonzero AC amplitude"):
                simulation_with_AC_field_frequency_mp(
                    params, [1, 2], _mp_phys_state(2), "PHYS")

    def test_quspin_h_zero_raises(self):
        """h = 0 must be refused by the QuSpin driver as well."""
        params = _qs_params(2, h=0.0, parameter="frequency")
        psi0 = np.zeros(3, dtype=complex)
        psi0[0] = 1.0
        with pytest.raises(ValueError, match="nonzero AC amplitude"):
            run_quspin_simulation(params, [1, 2], psi0)

    def test_run_simulation_h_zero_raises(self):
        """run_simulation rejects frequency mode with h = 0 up front."""
        params = SimulationParams(
            run_arguments={"dps": DPS, "steps_floquet_unitary": 5, "num_points": 10},
            N=2, J=mp.mpf("1.0"), B=mp.mpf("0.4"),
            parameter=EstimationParameter.FREQUENCY,
        )
        with pytest.raises(ValueError, match="nonzero AC amplitude"):
            run_simulation(params, [])

    def test_amplitude_mode_rejects_incommensurate_omega(self):
        """The Floquet-power fast path requires omega*nu*T = 2*pi*k."""
        with mp.workdps(DPS):
            params = _mp_params(2)
            params["omega"] = resolve_omega(params) * mp.mpf("1.01")
            with pytest.raises(ValueError, match="commensurate"):
                simulation_with_AC_field_mp(params, [1], _mp_phys_state(2), "PHYS")

    def test_amplitude_mode_accepts_commensurate_omega(self):
        """An omega completing two full cycles per Floquet period is allowed."""
        with mp.workdps(DPS):
            params = _mp_params(2)
            params["omega"] = 2 * resolve_omega(params)
            results = simulation_with_AC_field_mp(
                params, [1, 2], _mp_phys_state(2), "PHYS")
            assert len(results) == 2


class TestFrequencyModeResults:
    """Physical and numerical properties of the frequency-mode QFI."""

    def test_qfi_non_negative_and_t4_normalized(self):
        """F_omega >= 0 and the qfi field equals raw/(N^2 t^4)."""
        n = 4
        with mp.workdps(DPS):
            params = _mp_params(n, parameter=EstimationParameter.FREQUENCY)
            results = simulation_with_AC_field_mp(
                params, [1, 2, 5, 10], _mp_phys_state(n), "PHYS")
            assert [r.time for r in results] == [1, 2, 5, 10]
            for r in results:
                assert r.qfi >= 0
                expected = float(mp.mpf(r.qfi_raw_value) / (n ** 2 * r.time ** 4))
                assert abs(r.qfi - expected) <= 1e-12 * max(1.0, abs(expected))

    def test_quspin_t4_normalized(self):
        """QuSpin frequency mode uses the same N^2 t^4 normalization."""
        n = 4
        params = _qs_params(n, parameter="frequency")
        psi0 = np.zeros(n + 1, dtype=complex)
        psi0[0] = 1.0
        results = run_quspin_simulation(params, [1, 2, 5, 10], psi0)
        for r in results:
            assert r.qfi >= -1e-15
            expected = float(r.qfi_raw_value) / (n ** 2 * r.time ** 4)
            assert abs(r.qfi - expected) <= 1e-12 * max(1.0, abs(expected))

    def test_grid_deduplicated_and_sorted(self):
        """Duplicate/unsorted/zero time points collapse to a sorted grid."""
        n = 2
        with mp.workdps(DPS):
            params = _mp_params(n)
            results = simulation_with_AC_field_frequency_mp(
                params, [5, 3, 5, 0, 3], _mp_phys_state(n), "PHYS")
            assert [r.time for r in results] == [3, 5]

    def test_epsilon_stability(self):
        """Halving the finite-difference step leaves the QFI unchanged
        to well below the expected finite-difference error."""
        n = 3
        times = [1, 5, 10, 20]
        with mp.workdps(DPS):
            qfi_by_eps = []
            for eps in ("1e-7", "5e-8"):
                params = _mp_params(n, epsilon=eps)
                results = simulation_with_AC_field_frequency_mp(
                    params, times, _mp_phys_state(n), "PHYS")
                qfi_by_eps.append([r.qfi for r in results])
            for a, b in zip(*qfi_by_eps):
                assert abs(a - b) <= 1e-3 * max(abs(a), abs(b))

    def test_mpmath_quspin_agreement(self):
        """Frequency-mode QFI agrees between the two solvers."""
        n = 5
        times = [1, 2, 5, 10, 20, 50]
        with mp.workdps(DPS):
            mp_params = _mp_params(n, parameter=EstimationParameter.FREQUENCY)
            mp_results = simulation_with_AC_field_mp(
                mp_params, times, _mp_phys_state(n), "PHYS")
        qs_params = _qs_params(n, parameter="frequency")
        psi0 = np.zeros(n + 1, dtype=complex)
        psi0[0] = 1.0
        qs_results = run_quspin_simulation(qs_params, times, psi0)
        for mp_r, qs_r in zip(mp_results, qs_results):
            assert mp_r.time == qs_r.time
            assert abs(mp_r.qfi - qs_r.qfi) <= 1e-6 * max(abs(mp_r.qfi), abs(qs_r.qfi))

    def test_frequency_estimation_epsilon_bound(self):
        """The step is capped by the finite-difference bound 0.1/(h*N*t_max)."""
        with mp.workdps(DPS):
            base = mp.mpf("1e-7")
            # Small t_max: precision default wins.
            assert frequency_estimation_epsilon(base, mp.mpf("0.1"), 10, 100) == base
            # Large t_max: the bound wins.
            capped = frequency_estimation_epsilon(base, mp.mpf("0.1"), 10, 10 ** 7)
            assert capped == mp.mpf("0.1") / (mp.mpf("0.1") * 10 * 10 ** 7)
            assert capped < base


class TestBuildInitialStates:
    """QuSpin construction of the four standard initial states."""

    def test_build_initial_states(self):
        """All four standard states are returned normalized."""
        from lmg_qfi.config import InitialState

        n = 4
        states = build_initial_states(n, 1.0, 0.4)
        assert set(states) == set(InitialState)
        for vec in states.values():
            assert abs(np.linalg.norm(vec) - 1.0) < 1e-12
        assert abs(states[InitialState.PHYS][0] - 1.0) < 1e-12
        cat = states[InitialState.CAT_SUM]
        assert abs(cat[0] - 1 / np.sqrt(2)) < 1e-12
        assert abs(cat[n] - 1 / np.sqrt(2)) < 1e-12


class TestTimeIntervalSmallDegrees:
    """generate_time_interval must stay ascending for capped degrees."""

    @pytest.mark.parametrize("degree,expected_max", [(2, 100), (3, 1000), (4, 10000)])
    def test_max_time_and_monotonic(self, degree, expected_max):
        interval = generate_time_interval(20, degree)
        assert interval == sorted(interval)
        assert interval[-1] <= expected_max
        assert interval[-1] >= 10 ** (degree - 1)

    def test_degree_one_raises(self):
        with pytest.raises(ValueError):
            generate_time_interval(20, 1)


class TestConfigParsing:
    """Parsing of the new ini keys and CLI flags."""

    def _write_ini(self, tmp_path, extra_simulation_lines=""):
        content = (
            "[Simulation]\n"
            "j = 1.0\n"
            "n = 10\n"
            "b = 0.4\n"
            "t = 1\n"
            "phi-kick-phase = pi\n"
            "h = 0.1\n"
            "frequency = 2\n"
            "phi-0 = 0.0\n"
            "varphi = 0.0\n"
            "theta = 0.0\n"
            "num-points = 200\n"
            "steps-floquet-unitary = 10\n"
            "dps = 15\n"
            f"{extra_simulation_lines}"
            "\n[Files]\n"
            "output-dir = results\n"
        )
        ini = tmp_path / "parameters.ini"
        ini.write_text(content)
        return str(ini)

    def test_defaults_without_new_keys(self, tmp_path):
        params = read_defaults_args_from_config(self._write_ini(tmp_path))
        assert params.parameter == EstimationParameter.AMPLITUDE
        assert params.omega is None
        assert params.solver == SolverType.QUSPIN
        assert "max_time_degree" not in params.run_arguments

    def test_solver_key_parsed(self, tmp_path):
        params = read_defaults_args_from_config(self._write_ini(
            tmp_path, "solver = mpmath\n"))
        assert params.solver == SolverType.MPMATH

    def test_solver_cli_overrides_ini(self, tmp_path):
        import argparse

        params = read_defaults_args_from_config(self._write_ini(
            tmp_path, "solver = mpmath\n"))
        parser = argparse.ArgumentParser()
        add_simulation_cli_arguments(parser, params)
        apply_simulation_cli_arguments(params, parser.parse_args(["--solver", "quspin"]))
        assert params.solver == SolverType.QUSPIN

    def test_cli_defaults_keep_ini_solver(self, tmp_path):
        import argparse

        params = read_defaults_args_from_config(self._write_ini(
            tmp_path, "solver = mpmath\n"))
        parser = argparse.ArgumentParser()
        add_simulation_cli_arguments(parser, params)
        apply_simulation_cli_arguments(params, parser.parse_args([]))
        assert params.solver == SolverType.MPMATH

    def test_frequency_keys_parsed(self, tmp_path):
        params = read_defaults_args_from_config(self._write_ini(
            tmp_path,
            "parameter = frequency\nomega = 3.14\nmax-time-degree = 3\n"))
        assert params.parameter == EstimationParameter.FREQUENCY
        assert abs(float(params.omega) - 3.14) < 1e-12
        assert params.run_arguments["max_time_degree"] == 3

    def test_omega_resonant_keyword(self, tmp_path):
        params = read_defaults_args_from_config(self._write_ini(
            tmp_path, "parameter = frequency\nomega = resonant\n"))
        assert params.omega is None

    def test_cli_flags_applied(self, tmp_path):
        import argparse

        params = read_defaults_args_from_config(self._write_ini(tmp_path))
        parser = argparse.ArgumentParser()
        add_simulation_cli_arguments(parser, params)
        args = parser.parse_args([
            "--parameter", "frequency", "--omega", "3.0",
            "--amplitude", "0.2", "--max-time-degree", "3",
        ])
        apply_simulation_cli_arguments(params, args)
        assert params.parameter == EstimationParameter.FREQUENCY
        assert abs(float(params.omega) - 3.0) < 1e-12
        assert abs(float(params.h) - 0.2) < 1e-12
        assert params.run_arguments["max_time_degree"] == 3

    def test_cli_defaults_keep_ini_values(self, tmp_path):
        import argparse

        params = read_defaults_args_from_config(self._write_ini(
            tmp_path, "parameter = frequency\n"))
        parser = argparse.ArgumentParser()
        add_simulation_cli_arguments(parser, params)
        apply_simulation_cli_arguments(params, parser.parse_args([]))
        assert params.parameter == EstimationParameter.FREQUENCY
        assert abs(float(params.h) - 0.1) < 1e-12
