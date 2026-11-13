#!/usr/bin/env python3
"""Sensitivity of a fixed SLD calibrated at a different probe field.

For every initial state and time, construct the canonical pure-state SLD
L_ref(t) = 2 d_h rho_h(t)|h_ref (zero kernel--kernel block), then hold that
operator fixed while evaluating (d_h <L_ref>)^2 / Var(L_ref) at h_target.
The initial ket is the same at both fields. This is a different calibrated
observable for each state and time, not one observable shared by all panels.

Example:
    uv run python scripts/plot_sld_mismatch.py --check-convergence
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import mpmath as mp
import numpy as np
import pandas as pd

from lmg_qfi.config import SimulationParams, UF
from lmg_qfi.evolution import calculate_unitary_T, calculate_unitary_at_time_mp
from lmg_qfi.mom import method_of_moments_mp
from lmg_qfi.operators import create_hamiltonian_h0, create_spin_xyz_operators
from lmg_qfi.qfi import quantum_fisher_information_mp
from lmg_qfi.sld import compute_sld_matrix


STATES = ("GS_PHYS", "GS_CAT", "PHYS", "CAT_SUM")


class _StateEvolution:
    """Apply cached Floquet powers and residual periods to several kets."""

    def __init__(self, params, initial_kets, h, hamiltonian):
        values, vectors = mp.eig(calculate_unitary_T(h, params, hamiltonian))
        floquet = UF(values, vectors, mp.inverse(vectors))
        self.values = values
        self.coefficients = floquet.U_inv * initial_kets
        self.nu = params["nu"]
        self.left = [vectors]
        # The drive repeats after nu kick periods. A residual prefix can be
        # computed once at small absolute times, avoiding repeated exponentials
        # and large-argument trigonometric reduction at long times.
        for remainder in range(1, self.nu):
            prefix = calculate_unitary_at_time_mp(h, remainder, params, hamiltonian, floquet)
            self.left.append(prefix * vectors)

    def evaluate(self, t):
        quotient, remainder = divmod(int(t), self.nu)
        coefficients = self.coefficients.copy()
        for i, value in enumerate(self.values):
            power = value ** quotient
            for j in range(coefficients.cols):
                coefficients[i, j] *= power
        return self.left[remainder] * coefficients


class StateDerivative:
    """Cached state and centered h derivative, with one initial state per column.

    Instantiate and evaluate within the desired ``mp.workdps`` context. Outputs
    are raw evolved kets and derivatives; no normalization or projection is
    applied here. ``params`` is the lower-level evolution parameter dictionary.
    """

    def __init__(self, params: dict, initial_kets: mp.matrix, h, epsilon=None):
        self.epsilon = mp.mpf(epsilon) if epsilon is not None else mp.mpf(10) ** (-(mp.mp.dps // 2))
        h = mp.mpf(h)
        hamiltonian = create_hamiltonian_h0(params["J"], params["B"], params["N"])
        self.evolutions = {
            shift: _StateEvolution(params, initial_kets, h + shift * self.epsilon, hamiltonian)
            for shift in (0, 1, -1)
        }

    def shifted(self, t):
        return {shift: evolution.evaluate(t) for shift, evolution in self.evolutions.items()}

    def evaluate(self, t: int):
        kets = self.shifted(t)
        return kets[0], (kets[1] - kets[-1]) / (2 * self.epsilon)


def _initial_kets(params):
    n = params["N"]
    hamiltonian = create_hamiltonian_h0(params["J"], params["B"], n)
    _, vectors = mp.eigh(hamiltonian)
    ground, excited = vectors[:, 0], vectors[:, 1]
    # Fix the otherwise arbitrary relative eigenvector sign so GS_PHYS is the
    # state with positive magnetization; keep it fixed at both probe fields.
    sz, _, _ = create_spin_xyz_operators(n)
    if mp.re((ground.transpose_conj() * sz * excited)[0]) < 0:
        excited = -excited
    phys = mp.zeros(n + 1, 1)
    phys[0] = 1
    cat = phys.copy()
    cat[n] = 1
    states = [ground + excited, ground, phys, cat]
    kets = mp.zeros(n + 1, len(states))
    for i, state in enumerate(states):
        kets[:, i] = state / mp.norm(state)
    return kets


def _normalized_tangent(ket, derivative):
    norm = mp.norm(ket)
    normalized = ket / norm
    tangent = derivative / norm
    tangent -= normalized * mp.re((normalized.transpose_conj() * tangent)[0])
    return normalized, tangent


def compute_sld_mismatch(
    params: SimulationParams,
    times,
    *,
    h_reference="0.001",
    h_target="0",
    dps=50,
):
    """Return per-state normalized sensitivities and numerical diagnostics.

    CSV-ready columns ``qfi``, ``mom_sld_ref``, and ``mom_sld_local`` are all
    divided by (N t)^2. Calibration QFI and matched SLD sensitivity provide
    another independent identity check. The canonical SLD's zero kernel block
    is part of this experiment's definition: other extensions agree locally
    but can yield different sensitivity away from the calibration point.
    """
    times = sorted(set(int(t) for t in times))
    if not times or times[0] < 1:
        raise ValueError("times must contain positive integer kick periods")
    if dps < 30:
        raise ValueError("use at least 30 decimal digits for the SLD comparison")
    started = time.monotonic()
    diagnostics = {name: mp.mpf(0) for name in (
        "max_norm_error", "max_normalized_tangent_defect", "max_sld_hermiticity_error",
        "max_local_relative_error", "max_reference_matched_relative_error",
        "max_relative_bound_excess", "max_fixed_observable_fd_relative_error",
    )}
    rows = {state: [] for state in STATES}
    with mp.workdps(dps):
        parameters = dict(
            N=params.N, J=mp.mpf(str(params.J)), B=mp.mpf(str(params.B)),
            T=mp.mpf(str(params.T)), phi=mp.mpf(params.phi_kick_phase), nu=params.freq,
            theta=mp.mpf(str(params.theta)), varphi=mp.mpf(str(params.varphi)),
            phi_0=mp.mpf(str(params.phi_0)),
            steps_floquet_unitary=params.run_arguments["steps_floquet_unitary"],
        )
        initial_kets = _initial_kets(parameters)
        target = StateDerivative(parameters, initial_kets, h_target)
        logging.info("Target-field Floquet caches ready after %.1f s", time.monotonic() - started)
        reference = StateDerivative(parameters, initial_kets, h_reference)
        logging.info("All Floquet caches ready after %.1f s", time.monotonic() - started)
        fd_times = {times[0], times[len(times) // 2], times[-1]}
        for index, t in enumerate(times):
            target_kets, target_derivatives = target.evaluate(t)
            ref_kets, ref_derivatives = reference.evaluate(t)
            for column, state in enumerate(STATES):
                raw_ket, raw_derivative = target_kets[:, column], target_derivatives[:, column]
                raw_ref, raw_ref_derivative = ref_kets[:, column], ref_derivatives[:, column]
                for ket, derivative in ((raw_ket, raw_derivative), (raw_ref, raw_ref_derivative)):
                    diagnostics["max_norm_error"] = max(diagnostics["max_norm_error"], abs(mp.norm(ket) ** 2 - 1))
                    defect = abs(mp.re((ket.transpose_conj() * derivative)[0])) / max(mp.norm(derivative), 1)
                    diagnostics["max_normalized_tangent_defect"] = max(diagnostics["max_normalized_tangent_defect"], defect)
                ket, derivative = _normalized_tangent(raw_ket, raw_derivative)
                ref, ref_derivative = _normalized_tangent(raw_ref, raw_ref_derivative)
                sld_local = compute_sld_matrix(derivative, ket)
                sld_ref = compute_sld_matrix(ref_derivative, ref)
                qfi = quantum_fisher_information_mp(derivative, ket)
                qfi_reference = quantum_fisher_information_mp(ref_derivative, ref)
                local = method_of_moments_mp(sld_local, ket, derivative)
                mismatch = method_of_moments_mp(sld_ref, ket, derivative)
                matched_ref = method_of_moments_mp(sld_ref, ref, ref_derivative)
                norm = (params.N * t) ** 2
                diagnostics["max_sld_hermiticity_error"] = max(
                    diagnostics["max_sld_hermiticity_error"], mp.mnorm(sld_ref - sld_ref.transpose_conj(), 1))
                diagnostics["max_local_relative_error"] = max(
                    diagnostics["max_local_relative_error"], abs(local - qfi) / max(qfi, mp.mpf("1e-40")))
                diagnostics["max_reference_matched_relative_error"] = max(
                    diagnostics["max_reference_matched_relative_error"],
                    abs(matched_ref - qfi_reference) / max(qfi_reference, mp.mpf("1e-40")))
                diagnostics["max_relative_bound_excess"] = max(
                    diagnostics["max_relative_bound_excess"], (mismatch - qfi) / max(qfi, mp.mpf("1e-40")))
                if t in fd_times:
                    shifted = target.shifted(t)
                    means = {}
                    for shift in (1, -1):
                        shifted_ket = shifted[shift][:, column]
                        shifted_ket /= mp.norm(shifted_ket)
                        means[shift] = mp.re((shifted_ket.transpose_conj() * sld_ref * shifted_ket)[0])
                    signal_fd = (means[1] - means[-1]) / (2 * target.epsilon)
                    signal = 2 * mp.re((ket.transpose_conj() * sld_ref * derivative)[0])
                    error = abs(signal_fd - signal) / max(abs(signal), mp.sqrt(qfi) * mp.mnorm(sld_ref, 1), 1)
                    diagnostics["max_fixed_observable_fd_relative_error"] = max(
                        diagnostics["max_fixed_observable_fd_relative_error"], error)
                rows[state].append(dict(
                    time=t, qfi=float(qfi / norm), mom_sld_ref=float(mismatch / norm),
                    mom_sld_local=float(local / norm),
                    fidelity=float(abs((ket.transpose_conj() * ref)[0]) ** 2),
                    qfi_reference=float(qfi_reference / norm),
                    mom_sld_reference_matched=float(matched_ref / norm),
                ))
            if index % 100 == 0 or index == len(times) - 1:
                logging.info("%d/%d times, t=%d, elapsed %.1f s", index + 1, len(times), t, time.monotonic() - started)
        diagnostics = {key: float(value) for key, value in diagnostics.items()}
    simulations = {state: pd.DataFrame(data) for state, data in rows.items()}
    diagnostics["elapsed_seconds"] = time.monotonic() - started
    diagnostics["max_normalized_mismatch"] = {
        state: float((df.qfi - df.mom_sld_ref).max()) for state, df in simulations.items()
    }
    return simulations, diagnostics


def default_times(max_time=10000, time_step=5, log_points=400):
    """Integer time grid with denser late-time sampling of calibration mismatch."""
    logarithmic = np.rint(np.geomspace(1, max_time, log_points)).astype(int)
    linear = np.arange(1, max_time + 1, time_step, dtype=int)
    return sorted(set([*range(1, min(100, max_time) + 1), *logarithmic, *linear, max_time]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system-size", type=int, default=10)
    parser.add_argument("--x-coupling", default="0.4")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dps", type=int, default=50)
    parser.add_argument("--h-reference", default="0.001")
    parser.add_argument("--h-target", default="0")
    parser.add_argument("--max-time", type=int, default=10000)
    parser.add_argument("--time-step", type=int, default=5)
    parser.add_argument("--log-points", type=int, default=400)
    parser.add_argument("--times", type=int, nargs="+", help="explicit time grid, overriding the default grid")
    parser.add_argument("--output-dir", type=Path, default=Path("results/sld_mismatch"))
    parser.add_argument("--output", type=Path, help="plot path; default is inside --output-dir")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--check-convergence", action="store_true",
                        help="repeat selected times at 10 more digits and twice the Trotter steps")
    args = parser.parse_args()
    if args.steps < 2 or args.system_size < 1 or args.time_step < 1 or args.max_time < 1 or args.log_points < 2:
        parser.error("require steps >= 2, N >= 1, positive time settings, log-points >= 2")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with mp.workdps(args.dps):
        params = SimulationParams({"steps_floquet_unitary": args.steps}, N=args.system_size,
                                  J=mp.mpf("1"), B=mp.mpf(args.x_coupling), phi_kick_phase=mp.pi)
        times = args.times or default_times(args.max_time, args.time_step, args.log_points)
        sample_times = sorted(set([t for t in (1, 10, 100, 300, 1000, 3000, 6000, 10000) if t <= max(times)]))
        if args.check_convergence:
            times = sorted(set([*times, *sample_times]))
        kwargs = dict(h_reference=args.h_reference, h_target=args.h_target)
        simulations, diagnostics = compute_sld_mismatch(params, times, dps=args.dps, **kwargs)
        if args.check_convergence:
            baseline = {state: df.loc[df.time.isin(sample_times)].reset_index(drop=True)
                        for state, df in simulations.items()}
            precise, _ = compute_sld_mismatch(params, sample_times, dps=args.dps + 10, **kwargs)
            refined = SimulationParams({"steps_floquet_unitary": 2 * args.steps}, N=params.N, J=params.J, B=params.B,
                                       phi_kick_phase=mp.pi)
            finer, _ = compute_sld_mismatch(refined, sample_times, dps=args.dps, **kwargs)
            convergence = {"times": sample_times, "reference_steps": args.steps, "refined_steps": 2 * args.steps}
            for key, comparison in (("precision", precise), ("trotter", finer)):
                convergence[key] = {}
                for state in STATES:
                    scale = simulations[state].qfi.max()
                    convergence[key][state] = {
                        col: float((comparison[state][col] - baseline[state][col]).abs().max() / max(scale, 1e-30))
                        for col in ("qfi", "mom_sld_ref", "mom_sld_local")
                    }
            diagnostics["convergence"] = convergence
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label = f"N={params.N}_B={float(params.B):.2f}_h={args.h_target}_href={args.h_reference}"
    for state, df in simulations.items():
        path = args.output_dir / f"sld_mismatch.{state}_{label}.csv"
        df.to_csv(path, index=False)
        logging.info("Saved %s", path)
    metadata = dict(system_size=params.N, J="1", B=args.x_coupling, T="1", kick_phase="pi",
                    h_reference=args.h_reference, h_target=args.h_target, dps=args.dps, steps=args.steps,
                    gs_phys_orientation="positive Sz", sld="canonical pure-state, zero kernel-kernel block",
                    normalization="(N t)^2", time_points=len(times), diagnostics=diagnostics)
    metadata_path = args.output_dir / f"sld_mismatch_{label}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    logging.info("Saved %s", metadata_path)
    if not args.no_plot:
        from lmg_qfi.plotting import plot_sld_mismatch
        output = plot_sld_mismatch(simulations, params, args.output_dir, output=args.output, **kwargs)
        logging.info("Saved %s", output)


if __name__ == "__main__":
    main()
