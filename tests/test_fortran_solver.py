"""Tests of the Fortran solver (``fortran/``) against the Python implementation.

The solver is built once per session with ``make``; all tests are skipped
when ``gfortran`` or ``make`` is not installed. Covered:

* the Fortran unit-test suite (``make test``);
* agreement with the mpmath implementation of this package at small N, for
  the default set-up (h = 0, probe along z) and for a tilted probe with
  h != 0 and nu = 3;
* reproduction of the committed mpmath reference files ``results/*.csv``
  (time grid, magnetisation, QFI) for N = 20 and N = 40, and - if
  ``LMG_QFI_SLOW_TESTS`` is set - N = 60 and N = 80, where quadruple precision
  limits the agreement at the largest times.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import shutil
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FORTRAN_DIR = ROOT / "fortran"
RESULTS_DIR = ROOT / "results"
STATES = ("PHYS", "GS_PHYS", "GS_CAT", "CAT_SUM")

# The reference files were produced by a script version that built the
# ground-state doublet from B converted through a double; the evolution used
# the decimal value 0.4. Passing the same number makes all four states
# comparable to the full precision of the solver.
B_OF_REFERENCE_STATES = format(Decimal(0.4))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, FORTRAN_DIR / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_results = _load_script("compare_results")
mpmath_reference_points = _load_script("mpmath_reference_points")


@pytest.fixture(scope="session")
def build_dir(tmp_path_factory) -> Path:
    if shutil.which("gfortran") is None or shutil.which("make") is None:
        pytest.skip("gfortran and make are required for the Fortran solver")
    build = tmp_path_factory.mktemp("fortran_build")
    subprocess.run(["make", "-C", str(FORTRAN_DIR), f"BUILD={build}", f"{build}/lmg_qfi", f"{build}/test_lmg"],
                   check=True, capture_output=True, text=True)
    return build


@pytest.fixture(scope="session")
def solver(build_dir: Path) -> Path:
    return build_dir / "lmg_qfi"


def run_solver(solver: Path, out_dir: Path, *options: str) -> subprocess.CompletedProcess:
    """Run the solver with built-in defaults (no parameters.ini) plus options."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run([str(solver), "--output-dir", str(out_dir), *options],
                          cwd=out_dir, check=True, capture_output=True, text=True)


def write_reference(rows_by_state: dict, out_dir: Path, n_spins: int) -> None:
    for state, rows in rows_by_state.items():
        mpmath_reference_points.save_to_file_qfi_dynamics(
            results=rows, output_file=out_dir / f"InitialState.{state.name}_N={n_spins}_B=0.40.csv")


def test_fortran_unit_tests(build_dir: Path, tmp_path: Path):
    result = subprocess.run([str(build_dir / "test_lmg")], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-4000:]
    assert "FAIL" not in result.stdout


def test_command_line(solver: Path, tmp_path: Path):
    assert subprocess.run([str(solver), "--help"], cwd=tmp_path, capture_output=True).returncode == 0
    bad = subprocess.run([str(solver), "--no-such-option", "1"], cwd=tmp_path, capture_output=True, text=True)
    assert bad.returncode != 0 and "unknown option" in bad.stderr

    run_solver(solver, tmp_path / "out", "--system-size", "4", "--max-degree", "5", "--num-points", "3",
               "--init-state", "Phys,CatSum", "--prefix", "run1.", "--x-coupling", "0.25")
    files = sorted(p.name for p in (tmp_path / "out").glob("*.csv"))
    assert files == ["run1.CAT_SUM_N=4_B=0.25.csv", "run1.PHYS_N=4_B=0.25.csv"]
    with (tmp_path / "out" / files[0]).open() as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["time", "m_x", "m_y", "m_z", "qfi", "qfi_raw", "mom_sz", "mom_x", "mom_y"]
    assert len(rows) == 99 + 2 * 3 and rows[-1]["time"] == "100000"


def test_reads_parameters_ini(solver: Path, tmp_path: Path):
    """The solver takes its defaults from the same ini file as the Python scripts."""
    (tmp_path / "parameters.ini").write_text(
        "[Simulation]\nJ = 1.0\nN = 3\nB = 0.4\nT = 1\nphi-kick-phase = pi\nh = 0.0\nfrequency = 2\n"
        "phi-0 = 0.0\nvarphi = 0.0\ntheta = 0.0\nnum-points = 2\nsteps-floquet-unitary = 10\ndps = 15\n"
        "[Files]\noutput-dir = out\n")
    subprocess.run([str(solver), "--max-degree", "5", "--init-state", "GS_cat"], cwd=tmp_path, check=True,
                   capture_output=True)
    assert (tmp_path / "out" / "fortran.GS_CAT_N=3_B=0.40.csv").is_file()


@pytest.mark.parametrize("n_spins", [4, 5])
def test_matches_mpmath_default_setup(solver: Path, tmp_path: Path, n_spins: int):
    """h = 0, probe along z: the set-up of all production runs."""
    times = [1, 2, 3, 10, 37, 1000, 100001]
    reference = mpmath_reference_points.reference_points(n_spins, times, steps=10, dps=40)
    write_reference(reference, tmp_path / "mpmath", n_spins)
    grid = tmp_path / "mpmath" / f"InitialState.PHYS_N={n_spins}_B=0.40.csv"
    run_solver(solver, tmp_path / "fortran", "--system-size", str(n_spins), "--steps", "10",
               "--times-file", str(grid))
    for state in STATES:
        name = f"{state}_N={n_spins}_B=0.40.csv"
        res = compare_results.compare_files(tmp_path / "mpmath" / f"InitialState.{name}",
                                            tmp_path / "fortran" / f"fortran.{name}")
        assert res["grid_ok"]
        assert max(res["m_x"], res["m_y"], res["m_z"]) < 1e-14, (state, res)
        assert res["qfi"] < 1e-14, (state, res)
        assert res["qfi_raw"] < 1e-14, (state, res)
        for name in ("mom_sz", "mom_x", "mom_y"):
            assert res[name] < 1e-14, (state, name, res)


def test_matches_mpmath_tilted_probe(solver: Path, tmp_path: Path):
    """h != 0, tilted probe direction, nu = 3: the general code path."""
    n_spins = 4
    times = [1, 2, 3, 4, 5, 9, 100, 1001]
    options = dict(h="0.05", theta="0.6", varphi="0.4", phi_0="0.3", nu=3, steps=6)
    reference = mpmath_reference_points.reference_points(n_spins, times, dps=40, **options)
    write_reference(reference, tmp_path / "mpmath", n_spins)
    grid = tmp_path / "mpmath" / f"InitialState.PHYS_N={n_spins}_B=0.40.csv"
    common = ["--system-size", str(n_spins), "--steps", "6", "--times-file", str(grid),
              "--set", "h=0.05", "--set", "theta=0.6", "--set", "varphi=0.4", "--set", "phi-0=0.3",
              "--set", "frequency=3"]
    # For h != 0 the two symmetry-broken GS_phys states are not equivalent;
    # mpmath picks one of them through the arbitrary sign of its eigenvectors.
    for orientation in ("up", "down"):
        run_solver(solver, tmp_path / orientation, *common, "--gs-phys-orientation", orientation)

    def deviation(directory: Path, state: str) -> float:
        name = f"{state}_N={n_spins}_B=0.40.csv"
        res = compare_results.compare_files(tmp_path / "mpmath" / f"InitialState.{name}",
                                            directory / f"fortran.{name}", atol_m=1.0)
        assert res["grid_ok"] and not res["z2"]
        return max(res["m_x"], res["m_y"], res["m_z"], res["qfi"], res["qfi_raw"])

    for state in ("PHYS", "GS_CAT", "CAT_SUM"):
        assert deviation(tmp_path / "up", state) < 1e-14, state
    gs_phys = sorted(deviation(tmp_path / orientation, "GS_PHYS") for orientation in ("up", "down"))
    assert gs_phys[0] < 1e-14 and gs_phys[1] > 1e-6


REFERENCE_CASES = [
    # N, steps, num-points, GS_phys orientation, tolerances (m, qfi, qfi_raw)
    pytest.param(20, 250, 300, "down", 1e-15, 1e-15, 1e-22, id="N=20"),
    pytest.param(40, 250, 300, "down", 1e-15, 1e-15, 1e-17, id="N=40"),
    pytest.param(60, 100, 50, "up", 1e-12, 1e-13, 1e-11, id="N=60",
                 marks=pytest.mark.skipif(not os.environ.get("LMG_QFI_SLOW_TESTS"), reason="slow")),
    pytest.param(80, 100, 50, "up", 1e-7, 1e-8, 1e-6, id="N=80",
                 marks=pytest.mark.skipif(not os.environ.get("LMG_QFI_SLOW_TESTS"), reason="slow")),
]


@pytest.mark.parametrize("n_spins, steps, num_points, orientation, tol_m, tol_qfi, tol_raw", REFERENCE_CASES)
def test_reproduces_reference_results(solver: Path, tmp_path: Path, n_spins, steps, num_points, orientation,
                                      tol_m, tol_qfi, tol_raw):
    """The solver regenerates results/InitialState.*.csv from scratch.

    The time grid is generated by the solver itself (not read from the
    reference), so this also checks the NumPy-compatible logspace grid. The
    tolerances for N >= 60 reflect the rounding error (N+1) * 2e-34 * t_max
    of quadruple precision at t_max = 1e18 and 1e23.
    """
    run_solver(solver, tmp_path, "--system-size", str(n_spins), "--steps", str(steps),
               "--num-points", str(num_points), "--gs-phys-orientation", orientation,
               "--set", f"b-init={B_OF_REFERENCE_STATES}")
    for state in STATES:
        name = f"{state}_N={n_spins}_B=0.40.csv"
        res = compare_results.compare_files(RESULTS_DIR / f"InitialState.{name}", tmp_path / f"fortran.{name}")
        assert res["grid_ok"], (state, res)
        assert not res["z2"]
        # m_x, m_z and qfi are stored as doubles in the reference: 1 ulp = 5.6e-17 / 1.1e-16
        assert max(res["m_x"], res["m_y"], res["m_z"]) < max(tol_m, 2e-16), (state, res)
        assert res["qfi"] < max(tol_qfi, 4e-16), (state, res)
        assert res["qfi_raw"] < tol_raw, (state, res)
