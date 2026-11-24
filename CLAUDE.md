# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

The library simulates QFI dynamics in the Lipkin-Meshkov-Glick (LMG) model driven by a Floquet kick and AC field:

```
H(t) = -(2J/N) Sz² - 2B Sx - π Σ δ(t - mT) Sx + V(t)
```

All matrix computations use **mpmath** (arbitrary precision) via `mp.matrix`. System size `N` spins → Hilbert space dimension `N+1`.

### Key operator conventions

- `create_spin_xyz_operators(n)` returns `(Sz, Sx, Sy)` — **note: Sz first**, not Sx
- `ac_time(S_x, S_y, S_z, ...)` takes `(Sx, Sy, Sz)` — **note: Sx first**

### Precision control

`mp.workdps(dps)` context wraps the entire simulation. `epsilon = 1e-(dps//2)` is the finite-difference step. Default `dps=15` (double); use 50+ for high precision. All `mp.mpf` values should be set inside the `workdps` context.

### Fortran solver (`fortran/`)

Port of the mpmath simulation in `real128` with OpenMP; details in `fortran/README.md`.

- `GS_phys` orientation is explicit (`--gs-phys-orientation up|down`); mpmath's is arbitrary.
- The reference files in `results/` were produced with `steps=250, num-points=300` (N = 20, 40)
  and `steps=100, num-points=50` (N = 60, 80), `dps=60`, not with the committed `parameters.ini`.
- `lmg_qfi/__init__.py` sets `mp.dps = 100` on the mpmath *module*, which does not change the working
  precision (still 15 digits); code that needs more digits must use `mp.workdps`.

### Initial states

| `InitialState` | Construction |
|---|---|
| `GS_PHYS` | Ground state + first excited state (normalized) |
| `GS_CAT` | Ground state of H₀ |
| `PHYS` | Fock state `|N/2, N/2⟩` (top spin state) |
| `CAT_SUM` | Superposition of top and bottom Fock states |
