#!/usr/bin/env python3
"""Evaluate the mpmath reference implementation at a few chosen times.

The full mpmath simulation is too slow beyond N of about 40, but a handful
of time points is affordable and is enough to cross-check the Fortran
solver at any system size:

    uv run python fortran/scripts/mpmath_reference_points.py \
        --system-size 50 --steps 10 --dps 60 --times 1 2 1000 1000000000000 \
        --output-dir check/mpmath
    fortran/build/lmg_qfi --system-size 50 --steps 10 \
        --times-file "check/mpmath/InitialState.PHYS_N=50_B=0.40.csv" --output-dir check/fortran
    uv run python fortran/scripts/compare_results.py check/mpmath check/fortran --n 50

Everything numerical is done by the functions of the ``lmg_qfi`` package
(Floquet operator, its eigen-decomposition, finite-difference QFI). The only
difference to ``run_simulation`` is that the three evolution operators of a
time point are computed once and applied to all initial states.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import mpmath as mp

from lmg_qfi.config import InitialState, QFIInformation, UF
from lmg_qfi.evolution import calculate_unitary_T, calculate_unitary_at_time_mp
from lmg_qfi.io import save_to_file_qfi_dynamics
from lmg_qfi.operators import create_hamiltonian_h0, create_spin_xyz_operators
from lmg_qfi.qfi import dketa_t, quantum_fisher_information_mp

ALL_STATES = (InitialState.PHYS, InitialState.GS_PHYS, InitialState.GS_CAT, InitialState.CAT_SUM)


def initial_states(n_spins: int, j, b) -> dict[InitialState, mp.matrix]:
    """Initial kets, constructed as in ``lmg_qfi.simulation.run_simulation``."""
    size = n_spins + 1
    hamiltonian = create_hamiltonian_h0(j, b, n_spins)
    energies, evecs = mp.eigh(hamiltonian)
    order = sorted(range(len(energies)), key=lambda i: mp.re(energies[i]))
    ground, excited = evecs[:, order[0]], evecs[:, order[1]]
    phys = mp.zeros(size, 1)
    phys[0] = mp.mpf(1)
    cat = mp.zeros(size, 1)
    cat[0] = mp.mpf(1)
    cat[size - 1] = mp.mpf(1)
    kets = {
        InitialState.PHYS: phys,
        InitialState.GS_PHYS: ground + excited,
        InitialState.GS_CAT: ground,
        InitialState.CAT_SUM: cat,
    }
    return {state: ket / mp.norm(ket) for state, ket in kets.items()}


def reference_points(n_spins: int, times: list[int], *, b="0.4", j="1.0", period="1", h="0.0",
                     steps: int = 10, dps: int = 60, nu: int = 2, theta="0", varphi="0", phi_0="0",
                     states=ALL_STATES) -> dict[InitialState, list[QFIInformation]]:
    """QFI and magnetisation of the mpmath reference at the given times."""
    with mp.workdps(dps):
        params = dict(
            phi=mp.pi, J=mp.mpf(j), B=mp.mpf(b), T=mp.mpf(period), h=mp.mpf(h),
            epsilon=mp.mpf(f"1e-{dps // 2}"), N=n_spins, nu=nu,
            varphi=mp.mpf(varphi), phi_0=mp.mpf(phi_0), theta=mp.mpf(theta),
            steps_floquet_unitary=steps,
        )
        eps = params["epsilon"]
        h0 = create_hamiltonian_h0(params["J"], params["B"], n_spins)
        zsum, xsum, ysum = create_spin_xyz_operators(n_spins)
        kets = initial_states(n_spins, params["J"], params["B"])

        floquet = {}
        for shift in (0, 1, -1):
            started = time.time()
            evals, evecs = mp.eig(calculate_unitary_T(params["h"] + shift * eps, params, h0))
            floquet[shift] = UF(eigenvalues=evals, U=evecs, U_inv=mp.inverse(evecs))
            logging.info("Floquet operator at h%+d*eps: %.1f s", shift, time.time() - started)

        results = {state: [] for state in states}
        for t in times:
            started = time.time()
            unitary = {shift: calculate_unitary_at_time_mp(params["h"] + shift * eps, t, params, h0, floquet[shift])
                       for shift in (0, 1, -1)}
            for state in states:
                ket = unitary[0] * kets[state]
                dket = dketa_t(unitary[1] * kets[state], unitary[-1] * kets[state], eps)
                qfi = quantum_fisher_information_mp(dket, ket)
                results[state].append(QFIInformation(
                    qfi_raw_value=str(qfi),
                    qfi=float(qfi / (n_spins ** 2 * t ** 2)),
                    time=t,
                    m_x=float(mp.re((ket.transpose_conj() * xsum * ket)[0, 0]) / n_spins),
                    m_y=float(mp.re((ket.transpose_conj() * ysum * ket)[0, 0]) / n_spins),
                    m_z=float(mp.re((ket.transpose_conj() * zsum * ket)[0, 0]) / n_spins),
                ))
            logging.info("t = %d: %.1f s", t, time.time() - started)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system-size", type=int, required=True)
    parser.add_argument("--times", type=int, nargs="+", required=True, help="stroboscopic times (integers)")
    parser.add_argument("--x-coupling", default="0.4")
    parser.add_argument("--steps", type=int, default=10, help="steps-floquet-unitary")
    parser.add_argument("--dps", type=int, default=60, help="mpmath decimal digits")
    parser.add_argument("--h", default="0.0")
    parser.add_argument("--theta", default="0")
    parser.add_argument("--varphi", default="0")
    parser.add_argument("--phi-0", default="0")
    parser.add_argument("--frequency", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="InitialState.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    results = reference_points(
        args.system_size, args.times, b=args.x_coupling, h=args.h, steps=args.steps, dps=args.dps,
        nu=args.frequency, theta=args.theta, varphi=args.varphi, phi_0=args.phi_0,
    )
    for state, rows in results.items():
        name = f"{args.prefix}{state.name}_N={args.system_size}_B={float(args.x_coupling):.2f}.csv"
        save_to_file_qfi_dynamics(results=rows, output_file=args.output_dir / name)
        logging.info("wrote %s", args.output_dir / name)


if __name__ == "__main__":
    main()
