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

# Run the simulation (uses parameters.ini for defaults)
uv run python quantum_fisher_information_simulation_mpmath.py

# Run for a specific initial state
uv run python quantum_fisher_information_simulation_mpmath.py --init-state GS_phys

# Generate plot from existing results
uv run python quantum_fisher_information_simulation_mpmath.py --plot
```

## Architecture

The library simulates QFI dynamics in the Lipkin-Meshkov-Glick (LMG) model driven by a Floquet kick and AC field:

```
H(t) = -(2J/N) Sz² - 2B Sx - π Σ δ(t - mT) Sx + V(t)
```

All matrix computations use **mpmath** (arbitrary precision) via `mp.matrix`. System size `N` spins → Hilbert space dimension `N+1`.

### Data flow

```
parameters.ini / CLI args
        ↓
    SimulationParams  (config.py)
        ↓
    run_simulation()  (simulation.py)
        ├── create_hamiltonian_h0()    → H₀ matrix
        ├── calculate_unitary_T()      → Floquet unitary U_F for one period
        │       └── evalution_T_step() → Trotter integration of V(t) = H₀ + h·S_α(t)
        ├── UF dataclass               → eigendecomposition of U_F for fast U_F^n
        └── simulation_with_AC_field_mp()
                └── process_time_point_mp()  → QFIInformation per time step
                        ├── calculate_unitary_at_time_mp()  → U(t) via find_power_r_mpmath()
                        ├── quantum_fisher_information_mp() → QFI via finite differences
                        └── compute_sld_info()              → SLD cross-check (optional)
```

### Key operator conventions

- `create_spin_xyz_operators(n)` returns `(Sz, Sx, Sy)` — **note: Sz first**, not Sx
- `ac_time(S_x, S_y, S_z, ...)` takes `(Sx, Sy, Sz)` — **note: Sx first**
- `create_hamiltonian_h0(J, B, N)` — args are coupling_zz, coupling_x, num_spins

### QFI computation

QFI is computed by finite differences over the probe field `h`:
- Compute `U_F(h)`, `U_F(h+ε)`, `U_F(h-ε)` — each stored as a `UF` eigendecomposition
- `dket = (U(h+ε)|ψ₀⟩ - U(h-ε)|ψ₀⟩) / (2ε)`
- `QFI = 4(⟨∂ψ|∂ψ⟩ - |⟨ψ|∂ψ⟩|²)`, normalized as `QFI / (N² t²)`

### Precision control

`mp.workdps(dps)` context wraps the entire simulation. `epsilon = 1e-(dps//2)` is the finite-difference step. Default `dps=15` (double); use 50+ for high precision. All `mp.mpf` values should be set inside the `workdps` context.

### Configuration

`parameters.ini` at project root sets defaults. CLI args override them. The `[Simulation]` section maps to `SimulationParams` fields; `phi-kick-phase = pi` is parsed as `mp.pi`.

### Initial states

| `InitialState` | Construction |
|---|---|
| `GS_PHYS` | Ground state + first excited state (normalized) |
| `GS_CAT` | Ground state of H₀ |
| `PHYS` | Fock state `|N/2, N/2⟩` (top spin state) |
| `CAT_SUM` | Superposition of top and bottom Fock states |
