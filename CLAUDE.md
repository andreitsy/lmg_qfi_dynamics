# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_operators.py -v

# Run a single test by name
uv run pytest tests/test_qfi.py::TestQFI::test_something -v

# Run with coverage
uv run pytest tests/ --cov=lmg_qfi

# Run the simulation (uses parameters.ini for defaults; solver = quspin|mpmath,
# default quspin)
uv run python quantum_fisher_information_simulation.py

# Run for a specific initial state
uv run python quantum_fisher_information_simulation.py --init-state GS_phys

# Arbitrary-precision mpmath backend
uv run python quantum_fisher_information_simulation.py --solver mpmath

# QFI over the drive frequency omega instead of the amplitude h
uv run python quantum_fisher_information_simulation.py --parameter frequency --amplitude 0.1 --max-time-degree 3

# Generate plot from existing results (matches the active solver's files)
uv run python quantum_fisher_information_simulation.py --plot

# Parallel run across all initial states; solver and estimation mode via env vars
SOLVER=mpmath PARAMETER=frequency AMPLITUDE=0.1 MAX_TIME_DEGREE=3 ./run_simulation_parallel.sh
```

## Architecture

The library simulates QFI dynamics in the Lipkin-Meshkov-Glick (LMG) model driven by a Floquet kick and AC field:

```
H(t) = -(2J/N) Sz² - 2B Sx - π Σ δ(t - mT) Sx + V(t)
```

Two interchangeable backends implement the same Floquet algorithm: **quspin**
(numpy/scipy double precision, the default) and **mpmath** (arbitrary precision
via `mp.matrix`). System size `N` spins → Hilbert space dimension `N+1`.

### Data flow

```
parameters.ini / CLI args
        ↓
    SimulationParams  (config.py)   — parameter (amplitude|frequency), omega,
        ↓                             solver (quspin|mpmath)
quantum_fisher_information_simulation.py — dispatches on SolverType
        │
        ├── QUSPIN → run_quspin()  (quspin_solver.py)
        │       ├── build_initial_states() / _gap_time_degree() → states, time grid
        │       └── run_quspin_simulation()   — numpy/scipy Floquet evolution;
        │           amplitude fast path (eigendecomposition of U_F) or
        │           _run_quspin_frequency() (phase-continuous sequential)
        │
        └── MPMATH → run_simulation()  (simulation.py)
                ├── create_hamiltonian_h0()    → H₀ matrix
                └── simulation_with_AC_field_mp()   — dispatches on EstimationParameter
                        │
                        ├── AMPLITUDE (fast path, O(1) per time point)
                        │   ├── calculate_unitary_T()      → Floquet unitary U_F for one period
                        │   │       └── evalution_T_step() → Trotter integration of V(t) = H₀ + h·S_α(t)
                        │   ├── UF dataclass               → eigendecomposition of U_F for fast U_F^n
                        │   └── process_time_point_mp()    → QFIInformation per time step
                        │           ├── calculate_unitary_at_time_mp() → U(t) via find_power_r_mpmath()
                        │           └── qfi_information_from_kets()    → QFI via finite differences
                        │
                        └── FREQUENCY (sequential path, O(t_max) total)
                            └── simulation_with_AC_field_frequency_mp()
                                    ├── evolve_kets_one_period()       → three kets (ω, ω±ε) advanced
                                    │                                     period-by-period, drive phase
                                    │                                     continuous in absolute time
                                    └── qfi_information_from_kets(time_power=4)
```

Both `run_quspin()` and `run_simulation()` take `(SimulationParams, [InitialState])`
and return `{InitialState: [QFIInformation]}`; results are written as
`results/<solver>.<Fomega.?><state>_N=<N>_B=<B>.csv` and plotted to
`qfi_dynamics<_Fomega>_N=<N>_B=<B>_<solver>.png`.

`compute_sld_info()` (sld.py) is an optional SLD cross-check; pass `time_power=4` in
frequency mode.

### Key operator conventions

- `create_spin_xyz_operators(n)` returns `(Sz, Sx, Sy)` — **note: Sz first**, not Sx
- `ac_time(S_x, S_y, S_z, ...)` takes `(Sx, Sy, Sz)` — **note: Sx first**
- `create_hamiltonian_h0(J, B, N)` — args are coupling_zz, coupling_x, num_spins

### QFI computation

QFI is computed by central finite differences over the estimated parameter
(`EstimationParameter` in config.py):

**Amplitude mode (default)** — parameter is the probe field `h`:
- Compute `U_F(h)`, `U_F(h+ε)`, `U_F(h-ε)` — each stored as a `UF` eigendecomposition
- `dket = (U(h+ε)|ψ₀⟩ - U(h-ε)|ψ₀⟩) / (2ε)`
- `QFI = 4(⟨∂ψ|∂ψ⟩ - |⟨ψ|∂ψ⟩|²)`, normalized as `QFI / (N² t²)`

**Frequency mode** — parameter is the drive frequency `ω` (default: resonant `2π/(νT)`):
- At `ω±ε` the drive `sin(ωt + φ₀)` is not commensurate with the kick period, so the
  `U_F^n` shortcut is invalid (it would reset the drive phase each AC cycle and delete the
  secular phase drift that carries the long-time ω signal). Three kets are propagated
  sequentially period-by-period with the drive phase continuous in absolute time.
- Requires `h ≠ 0` (at `h = 0` there is no ω dependence); normalized as `QFI / (N² t⁴)`.
- `ε = min(10^-(dps//2), 0.1/(h·N·t_max))` keeps the finite difference in its linear
  regime (`ε`·‖∂_ω generator‖ ≲ 0.1); the driver logs the chosen value.
- Cost is linear in `t_max` → gap-based auto time range is capped at `10⁴` periods for
  mpmath (`DEFAULT_FREQUENCY_MAX_TIME_DEGREE` in simulation.py; `10⁶` for quspin, same
  constant in quspin_solver.py); override with `--max-time-degree`.
- Results are saved with an `Fomega.` prefix after the solver tag
  (`<solver>.Fomega. ...`); `--plot --parameter frequency` selects only those files and
  labels the axis `F_ω/(N²t⁴)`; frequency-mode titles list `h` and `ω`.

### Precision control

`mp.workdps(dps)` context wraps the entire simulation. `epsilon = 1e-(dps//2)` is the finite-difference step. Default `dps=15` (double); use 50+ for high precision. All `mp.mpf` values should be set inside the `workdps` context.

### Configuration

`parameters.ini` at project root sets defaults. CLI args override them. The `[Simulation]` section maps to `SimulationParams` fields; `phi-kick-phase = pi` is parsed as `mp.pi`. Backend key: `solver = quspin|mpmath` (default quspin). Estimation-mode keys: `parameter = amplitude|frequency`, `omega = resonant|<float>` (resonant means `2π/(νT)`), optional `max-time-degree` (cap `10^degree` on the simulated time). CLI equivalents: `--solver`, `--parameter`, `--omega`, `--amplitude` (sets `h`), `--max-time-degree`.

### Initial states

| `InitialState` | Construction |
|---|---|
| `GS_PHYS` | Ground state + first excited state (normalized) |
| `GS_CAT` | Ground state of H₀ |
| `PHYS` | Fock state `|N/2, N/2⟩` (top spin state) |
| `CAT_SUM` | Superposition of top and bottom Fock states |
