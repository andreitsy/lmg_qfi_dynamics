"""Compare mpmath and QuSpin QFI solvers for default parameters (N=10, B=0.4).

Runs QuSpin across 2000+ time points for all 4 initial states, validates
against mpmath on a 50-point sample, and prints a side-by-side comparison.
"""

import time
import numpy as np
import mpmath as mp

from lmg_qfi.operators import create_hamiltonian_h0
from lmg_qfi.simulation import simulation_with_AC_field_mp
from lmg_qfi.quspin_solver import get_spin_operators, build_h0, run_quspin_simulation

# ── Parameters (matching parameters.ini defaults) ──────────────────────────
N       = 10
J       = 1.0
B       = 0.4
T       = 1.0
nu      = 2        # frequency
phi     = np.pi
h       = 0.0
dps     = 15
steps   = 10       # steps_floquet_unitary
epsilon = 1e-7

# ── Time intervals ──────────────────────────────────────────────────────────
# Full 2000-point interval for QuSpin
FULL_TIMES = sorted(set(
    list(range(1, 100)) +
    [int(x) for x in np.logspace(2, 4, 950, endpoint=False)] +
    [int(x) for x in np.logspace(4, 6, 1050, endpoint=True)]
))

# 50-point sample (evenly spaced in index) for mpmath validation
_sample_idx  = np.round(np.linspace(0, len(FULL_TIMES) - 1, 50)).astype(int)
SAMPLE_TIMES = [FULL_TIMES[i] for i in _sample_idx]

# ── Shared params ───────────────────────────────────────────────────────────

def _mp_params():
    return dict(
        N=N, J=mp.mpf("1.0"), B=mp.mpf("0.4"), T=mp.mpf("1.0"),
        nu=nu, phi=mp.pi, h=mp.mpf("0.0"),
        epsilon=mp.mpf(f"1e-{dps // 2}"),
        steps_floquet_unitary=steps,
        varphi=mp.mpf("0.0"), theta=mp.mpf("0.0"), phi_0=mp.mpf("0.0"),
    )


def _qs_params():
    return dict(
        N=N, J=J, B=B, T=T, nu=nu, phi=phi, h=h,
        epsilon=epsilon, steps_floquet_unitary=steps,
        varphi=0.0, theta=0.0, phi_0=0.0,
    )


# ── Initial state builders ──────────────────────────────────────────────────

def _np_eigh():
    Sz, Sx, _ = get_spin_operators(N)
    H = build_h0(J, B, N, Sz, Sx)
    evals, evecs = np.linalg.eigh(H)
    return evals, evecs


def _mp_eigh():
    H = create_hamiltonian_h0(J, B, N)
    energies, evecs = mp.eigh(H)
    idx = sorted(range(len(energies)), key=lambda i: mp.re(energies[i]))
    return energies, evecs, idx


def states_np():
    evals, evecs = _np_eigh()
    gs, ex = evecs[:, 0], evecs[:, 1]
    phys    = np.zeros(N + 1, dtype=complex); phys[0] = 1.0
    cat_sum = np.zeros(N + 1, dtype=complex); cat_sum[0] = cat_sum[-1] = 1.0
    gs_phys = (gs + ex)
    return {
        "PHYS":    phys    / np.linalg.norm(phys),
        "GS_PHYS": gs_phys / np.linalg.norm(gs_phys),
        "GS_CAT":  gs      / np.linalg.norm(gs),
        "CAT_SUM": cat_sum / np.linalg.norm(cat_sum),
    }


def states_mp(energies, evecs, idx):
    gs      = evecs[:, idx[0]]
    ex      = evecs[:, idx[1]]
    phys    = mp.zeros(N + 1, 1); phys[0]  = mp.mpf("1.0")
    cat_sum = mp.zeros(N + 1, 1); cat_sum[0] = mp.mpf("1.0"); cat_sum[N] = mp.mpf("1.0")
    gs_phys = gs + ex
    return {
        "PHYS":    phys    / mp.norm(phys),
        "GS_PHYS": gs_phys / mp.norm(gs_phys),
        "GS_CAT":  gs      / mp.norm(gs),
        "CAT_SUM": cat_sum / mp.norm(cat_sum),
    }


# ── Printing helpers ────────────────────────────────────────────────────────

_VAL_W  = 14
_HDR_FMT = f"{{:>8}}" + f"  {{:>{_VAL_W}}}" * 4
_ROW_FMT = f"{{:>8}}" + f"  {{:>{_VAL_W}.6e}}" * 4

STATE_NAMES = ["PHYS", "GS_PHYS", "GS_CAT", "CAT_SUM"]


def _print_qs_table(qs_all, times, title, stride=1):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
    hdr = _HDR_FMT.format("time", *STATE_NAMES)
    print(hdr)
    print("-" * len(hdr))
    for i, t in enumerate(times):
        if i % stride != 0:
            continue
        row = [qs_all[s][i].qfi for s in STATE_NAMES]
        print(_ROW_FMT.format(t, *row))


def _print_validation(qs_sample, mp_sample, state):
    print(f"\n── Validation: mpmath vs QuSpin  [{state}] ──────────────────────────")
    hdr = f"{'time':>8}  {'qfi_mpmath':>14}  {'qfi_quspin':>14}  {'rel_err%':>10}"
    print(hdr)
    print("-" * len(hdr))
    rel_errs = []
    for mp_r, qs_r in zip(mp_sample, qs_sample):
        if abs(mp_r.qfi) > 1e-15:
            rel = abs(qs_r.qfi - mp_r.qfi) / abs(mp_r.qfi) * 100
            rel_errs.append(rel)
        else:
            rel = 0.0
        print(f"{mp_r.time:>8}  {mp_r.qfi:>14.6e}  {qs_r.qfi:>14.6e}  {rel:>10.4f}")
    if rel_errs:
        print(f"  max={max(rel_errs):.4f}%  mean={np.mean(rel_errs):.4f}%")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("LMG QFI Solver Comparison — All 4 Initial States")
    print(f"  N={N}, B={B}, J={J}, T={T}, nu={nu}, phi=π, dps={dps}")
    print(f"  QuSpin:  {len(FULL_TIMES)} time points  (t=1 … {FULL_TIMES[-1]:,})")
    print(f"  mpmath:  {len(SAMPLE_TIMES)} sampled points for validation")

    np_states = states_np()

    # ── QuSpin: all 4 states × full 2000-point interval ──────────────────
    print(f"\n[1/2] Running QuSpin solver …", flush=True)
    qs_params = _qs_params()
    qs_all = {}
    t0 = time.perf_counter()
    for name in STATE_NAMES:
        qs_all[name] = run_quspin_simulation(qs_params, FULL_TIMES, np_states[name])
    qs_total = time.perf_counter() - t0
    print(f"      Done in {qs_total:.2f}s  ({qs_total/len(STATE_NAMES):.2f}s per state)")

    # ── mpmath: all 4 states × 50-point sample ────────────────────────────
    print(f"\n[2/2] Running mpmath solver on {len(SAMPLE_TIMES)}-point sample …", flush=True)
    mp_all = {}
    with mp.workdps(dps):
        energies, evecs, idx = _mp_eigh()
        mp_states_dict = states_mp(energies, evecs, idx)
        H0_mp = create_hamiltonian_h0(J, B, N)
        mp_params = _mp_params()
        t0 = time.perf_counter()
        for name in STATE_NAMES:
            ts = time.perf_counter()
            mp_all[name] = simulation_with_AC_field_mp(
                mp_params, SAMPLE_TIMES, mp_states_dict[name], name
            )
            print(f"      {name:<10}  {time.perf_counter()-ts:.1f}s", flush=True)
    mp_total = time.perf_counter() - t0
    print(f"      Done in {mp_total:.1f}s total")

    # Build QuSpin sample results (same indices as mpmath)
    qs_sample_idx = {t: i for i, t in enumerate(FULL_TIMES)}
    qs_sample = {
        name: [qs_all[name][qs_sample_idx[t]] for t in SAMPLE_TIMES]
        for name in STATE_NAMES
    }

    # ── Results ─────────────────────────────────────────────────────────
    # Show full QuSpin table (every 40th point to keep output manageable)
    _print_qs_table(qs_all, FULL_TIMES, "QuSpin QFI — all 4 states (every 40th point shown)",
                    stride=40)

    # Show all validation comparisons
    print(f"\n{'='*72}")
    print("  mpmath vs QuSpin validation (50-point sample)")
    print(f"{'='*72}")
    for name in STATE_NAMES:
        _print_validation(qs_sample[name], mp_all[name], name)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  Summary")
    print(f"{'='*72}")
    print(f"  {'State':<10}  {'max_rel_err%':>14}  {'mean_rel_err%':>14}")
    print(f"  {'-'*42}")
    for name in STATE_NAMES:
        errs = []
        for mp_r, qs_r in zip(mp_all[name], qs_sample[name]):
            if abs(mp_r.qfi) > 1e-15:
                errs.append(abs(qs_r.qfi - mp_r.qfi) / abs(mp_r.qfi) * 100)
        if errs:
            print(f"  {name:<10}  {max(errs):>14.6f}  {np.mean(errs):>14.6f}")
        else:
            print(f"  {name:<10}  {'N/A':>14}  {'N/A':>14}")
    print(f"\n  Speedup (mpmath/{len(SAMPLE_TIMES)}pts vs QuSpin/{len(FULL_TIMES)}pts): "
          f"{mp_total:.1f}s / {qs_total:.2f}s = x{mp_total/qs_total:.0f}  "
          f"(normalised per point: x{(mp_total/len(SAMPLE_TIMES))/(qs_total/len(FULL_TIMES)):.0f})")
