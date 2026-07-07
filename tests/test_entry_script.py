"""Tests for the merged CLI entry script (solver dispatch, naming, plot filter)."""

import importlib.util

from pathlib import Path

import mpmath as mp
import pytest

from lmg_qfi import EstimationParameter, InitialState, SimulationParams, SolverType


def _load_script():
    script = Path(__file__).parent.parent / "quantum_fisher_information_simulation.py"
    spec = importlib.util.spec_from_file_location(
        "quantum_fisher_information_simulation", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _params(solver, parameter=EstimationParameter.AMPLITUDE, N=10, **overrides):
    run_arguments = {"dps": 15, "steps_floquet_unitary": 5, "num_points": 5,
                     "max_time_degree": 2, "output_dir": "results"}
    run_arguments.update(overrides.pop("run_arguments", {}))
    return SimulationParams(
        run_arguments=run_arguments, N=N, J=mp.mpf("1.0"), B=mp.mpf("0.4"),
        parameter=parameter, solver=solver, **overrides)


_CSV_HEADER = "time,m_x,m_y,m_z,qfi,qfi_raw\n1,0.5,0.0,0.0,0.1,0.1\n"


class TestResultFileName:

    def test_amplitude_quspin(self):
        module = _load_script()
        name = module.result_file_name(_params(SolverType.QUSPIN),
                                       InitialState.GS_PHYS)
        assert name == "quspin.GS_phys_N=10_B=0.40.csv"

    def test_frequency_mpmath(self):
        module = _load_script()
        name = module.result_file_name(
            _params(SolverType.MPMATH, parameter=EstimationParameter.FREQUENCY),
            InitialState.GS_PHYS)
        assert name == "mpmath.Fomega.GS_phys_N=10_B=0.40.csv"


class TestCollectPlotFrames:

    def test_filters_by_solver_and_mode(self, tmp_path):
        module = _load_script()
        for name in ("quspin.GS_phys_N=10_B=0.40.csv",
                     "mpmath.GS_phys_N=10_B=0.40.csv",
                     "quspin.Fomega.GS_phys_N=10_B=0.40.csv",
                     "quspin.GS_phys_N=12_B=0.40.csv"):
            (tmp_path / name).write_text(_CSV_HEADER)

        frames = module.collect_plot_frames(_params(SolverType.QUSPIN), tmp_path)
        assert set(frames) == {"GS_phys"}

        frames = module.collect_plot_frames(_params(SolverType.MPMATH), tmp_path)
        assert set(frames) == {"GS_phys"}

        frames = module.collect_plot_frames(
            _params(SolverType.QUSPIN, parameter=EstimationParameter.FREQUENCY),
            tmp_path)
        assert set(frames) == {"GS_phys"}

    def test_empty_when_no_matching_solver(self, tmp_path):
        module = _load_script()
        (tmp_path / "quspin.GS_phys_N=10_B=0.40.csv").write_text(_CSV_HEADER)
        assert module.collect_plot_frames(_params(SolverType.MPMATH), tmp_path) == {}


class TestRunSolverDispatch:

    @pytest.mark.parametrize("solver", [SolverType.QUSPIN, SolverType.MPMATH])
    def test_dispatch(self, solver):
        module = _load_script()
        results = module.run_solver(_params(solver, N=2), [InitialState.PHYS])
        assert set(results) == {InitialState.PHYS}
        times = [r.time for r in results[InitialState.PHYS]]
        assert times == sorted(times)
        assert times[-1] <= 100  # max_time_degree = 2
        for r in results[InitialState.PHYS]:
            assert r.qfi >= -1e-12
            assert float(r.qfi_raw_value) >= -1e-12
