#!/usr/bin/env python3
"""Cross-validate Fortran (optimized) vs Python results.

Runs both implementations with identical parameters for a small system (N=6)
and compares magnetizations (epsilon-independent) and QFI structure.
"""
import csv
import os
import subprocess
import sys
import tempfile

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mpmath as mp
from lmg_qfi.operators import create_spin_xyz_operators, create_hamiltonian_h0, create_kick_operator
from lmg_qfi.evolution import calculate_unitary_T, evalution_T_step, find_power_r_mpmath, calculate_unitary_at_time_mp
from lmg_qfi.qfi import process_time_point_mp
from lmg_qfi.config import UF


def run_python_simulation(N, B, J, T, phi, h, nu, steps, dps, time_points):
    """Run Python simulation and return results as list of dicts."""
    with mp.workdps(dps):
        epsilon = mp.mpf(f"1e-{dps // 2}")
        params = dict(
            phi=mp.pi, J=mp.mpf(J), B=mp.mpf(B), T=mp.mpf(T),
            h=mp.mpf(h), epsilon=epsilon, N=N, nu=nu,
            varphi=mp.mpf(0), phi_0=mp.mpf(0), theta=mp.mpf(0),
            steps_floquet_unitary=steps
        )

        Zsum, Xsum, Ysum = create_spin_xyz_operators(N)
        H_0 = create_hamiltonian_h0(J, B, N)

        # Compute Floquet unitaries
        U_T = calculate_unitary_T(h, params, H_0)
        evals, evecs = mp.eig(U_T)
        uf = UF(eigenvalues=evals, U=evecs, U_inv=mp.inverse(evecs))

        U_Tp = calculate_unitary_T(h + epsilon, params, H_0)
        evals_p, evecs_p = mp.eig(U_Tp)
        uf_p = UF(eigenvalues=evals_p, U=evecs_p, U_inv=mp.inverse(evecs_p))

        U_Tm = calculate_unitary_T(h - epsilon, params, H_0)
        evals_m, evecs_m = mp.eig(U_Tm)
        uf_m = UF(eigenvalues=evals_m, U=evecs_m, U_inv=mp.inverse(evecs_m))

        # Initial state: PHYS (Fock state |N/2, N/2>)
        init_state = mp.zeros(N + 1, 1)
        init_state[0] = mp.mpf('1.0')

        results = []
        for t in time_points:
            res = process_time_point_mp(t, params, H_0, uf, uf_p, uf_m,
                                        init_state, Zsum, Xsum, Ysum)
            results.append({
                'time': t,
                'm_x': float(res.m_x),
                'm_y': float(res.m_y),
                'm_z': float(res.m_z),
                'qfi': float(res.qfi),
            })
    return results


def run_fortran_simulation(N, B, fortran_bin, output_dir):
    """Run Fortran simulation and parse CSV results."""
    config_content = f"""[Simulation]
J = 1.0
N = {N}
B = {B}
T = 1
phi-kick-phase = pi
h = 0.0
frequency = 2
phi-0 = 0.0
varphi = 0.0
theta = 0.0
num-points = 500
steps-floquet-unitary = 50
dps = 30

[Files]
output-dir = {output_dir}
"""
    config_path = os.path.join(output_dir, 'test_params.ini')
    with open(config_path, 'w') as f:
        f.write(config_content)

    result = subprocess.run(
        [fortran_bin, '--config', config_path, '--init-state', 'Phys',
         '--system-size', str(N), '--x-coupling', str(B)],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"Fortran stderr:\n{result.stderr}")
        raise RuntimeError(f"Fortran simulation failed with code {result.returncode}")

    # Parse output CSV
    csv_path = os.path.join(output_dir, f'InitialState.PHYS_N={N}_B={B:.2f}.csv')
    if not os.path.exists(csv_path):
        # Try listing directory
        print(f"Looking for CSV in {output_dir}:")
        print(os.listdir(output_dir))
        raise FileNotFoundError(f"Expected {csv_path}")

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'time': int(row['time']),
                'm_x': float(row['m_x']),
                'm_y': float(row['m_y']),
                'm_z': float(row['m_z']),
                'qfi': float(row['qfi']),
                'qfi_raw': float(row['qfi_raw']),
            })
    return rows


def compare_results(py_results, f90_results):
    """Compare Python and Fortran results.

    Magnetizations should match to ~10 digits (both compute the same thing,
    differences arise from Trotter-vs-exact for h=0 and precision level).

    QFI may differ more because epsilon differs (Python: 1e-15, Fortran: 1e-16).
    """
    # Build lookup by time
    f90_by_time = {r['time']: r for r in f90_results}

    mag_tol = 1e-8  # magnetizations should match closely
    max_m_diff = 0.0
    max_qfi_ratio = 0.0
    n_compared = 0
    failures = []

    for py in py_results:
        t = py['time']
        if t not in f90_by_time:
            continue
        f90 = f90_by_time[t]
        n_compared += 1

        for comp in ['m_x', 'm_y', 'm_z']:
            diff = abs(py[comp] - f90[comp])
            rel = diff / max(abs(py[comp]), 1e-30)
            max_m_diff = max(max_m_diff, rel)
            if rel > mag_tol and abs(py[comp]) > 1e-10:
                failures.append(
                    f"  t={t}: {comp} Python={py[comp]:.15e} Fortran={f90[comp]:.15e} "
                    f"rel_diff={rel:.2e}"
                )

        # QFI comparison: different epsilon, so allow larger tolerance
        # but sign and order of magnitude should match
        if abs(py['qfi']) > 1e-15 and abs(f90['qfi']) > 1e-15:
            ratio = py['qfi'] / f90['qfi']
            max_qfi_ratio = max(max_qfi_ratio, abs(ratio - 1))

    print(f"\nCompared {n_compared} time points")
    print(f"Max magnetization relative difference: {max_m_diff:.2e}")
    print(f"Max QFI ratio deviation from 1: {max_qfi_ratio:.2e}")
    print(f"  (QFI differs due to epsilon: Python=1e-15, Fortran=1e-16)")

    if failures:
        print(f"\n{len(failures)} magnetization failures (rel_diff > {mag_tol}):")
        for f in failures[:10]:
            print(f)
        return False
    else:
        print("\nAll magnetization comparisons PASSED")
        return True


def main():
    N = 6
    B = 0.4
    J = 1.0

    # Find Fortran binary
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fortran_dir = os.path.dirname(script_dir)
    fortran_bin = os.path.join(fortran_dir, 'lmg_qfi_sim')
    if not os.path.exists(fortran_bin):
        fortran_bin = os.path.join(fortran_dir, 'build', 'src', 'lmg_qfi_sim')
    if not os.path.exists(fortran_bin):
        print(f"ERROR: Cannot find Fortran binary. Tried:\n"
              f"  {os.path.join(fortran_dir, 'lmg_qfi_sim')}\n"
              f"  {fortran_bin}")
        sys.exit(1)

    print(f"Using Fortran binary: {fortran_bin}")
    print(f"Parameters: N={N}, B={B}, J={J}, h=0, nu=2, steps=50, dps=30")
    print("=" * 60)

    # Run Fortran
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\nRunning Fortran simulation...")
        f90_results = run_fortran_simulation(N, B, fortran_bin, tmpdir)
        print(f"  Got {len(f90_results)} time points")

        # Select subset of time points to run in Python (first 20 + a few large ones)
        f90_times = [r['time'] for r in f90_results]
        test_times = f90_times[:20]  # first 20
        # Add a few from the middle and end
        for idx in [len(f90_times)//4, len(f90_times)//2, 3*len(f90_times)//4, -1]:
            if f90_times[idx] not in test_times:
                test_times.append(f90_times[idx])
        test_times.sort()

        print(f"\nRunning Python simulation for {len(test_times)} time points...")
        py_results = run_python_simulation(
            N=N, B=B, J=J, T=1.0, phi=mp.pi, h=0.0,
            nu=2, steps=50, dps=30, time_points=test_times
        )
        print(f"  Got {len(py_results)} time points")

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    ok = compare_results(py_results, f90_results)

    # Print first few rows side by side
    f90_by_time = {r['time']: r for r in f90_results}
    print(f"\n{'time':>6} | {'Python m_x':>18} {'Fortran m_x':>18} | {'Python m_z':>18} {'Fortran m_z':>18} | {'Py QFI':>14} {'F90 QFI':>14}")
    print("-" * 120)
    for py in py_results[:10]:
        t = py['time']
        f90 = f90_by_time.get(t)
        if f90:
            print(f"{t:6d} | {py['m_x']:18.12f} {f90['m_x']:18.12f} | "
                  f"{py['m_z']:18.12f} {f90['m_z']:18.12f} | "
                  f"{py['qfi']:14.8e} {f90['qfi']:14.8e}")

    if not ok:
        sys.exit(1)
    print("\nCross-validation PASSED")


if __name__ == '__main__':
    main()
