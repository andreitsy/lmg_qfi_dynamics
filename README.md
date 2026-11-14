# QFI Dynamics in LMG Model

Simulation of Quantum Fisher Information dynamics in the Lipkin-Meshkov-Glick model. See paper: https://arxiv.org/abs/2505.22731

```math
\hat{H}(t) = -\frac{2J}{N} \hat{S}_z^2 - 2 B \hat{S}_x - \pi \sum_{m=1}^{\infty} \delta(t - mT) \hat{S}_x + \hat V(t)
```

![Plot for QFI](results/qfi_dynamics_N=40_B=0.40.png "QFI dynamics")


## Installation

Requires [uv](https://docs.astral.sh/uv/):
```bash
conda env create -f environment.yml
# activate env 
conda activate lmg_qfi
uv sync
```

## Usage

```bash
# Run simulation for all initial states
uv run python quantum_fisher_information_simulation_mpmath.py

# Run for specific initial state (GS_phys, GS_cat, CatSum, Phys)
uv run python quantum_fisher_information_simulation_mpmath.py --init-state GS_phys

# Custom parameters
uv run python quantum_fisher_information_simulation_mpmath.py --system-size 20 --x-coupling 0.4

# Generate plot from existing results
uv run python quantum_fisher_information_simulation_mpmath.py --plot
```


## QuSpin Solver (fast double-precision alternative)

The QuSpin-based solver uses [QuSpin](https://quspin.github.io/QuSpin/) to build spin operators and
`scipy`/`numpy` for time evolution. It implements the same Floquet algorithm as the mpmath version
but runs ~800–1000× faster at double precision (dps=15).

### Installation

QuSpin is an optional dependency:

```bash
uv sync --extra quspin
# or add it directly
uv add quspin
```

### Running from the command line

```bash
# Run all 4 initial states (reads N, B, J, … from parameters.ini)
uv run python quantum_fisher_information_simulation_quspin.py

# Run a single initial state
uv run python quantum_fisher_information_simulation_quspin.py --init-state GS_phys

# Custom system size and transverse field
uv run python quantum_fisher_information_simulation_quspin.py --system-size 20 --x-coupling 0.4

# Generate plot from existing QuSpin CSV results
uv run python quantum_fisher_information_simulation_quspin.py --plot
```

Results are saved to `results/quspin.<state>_N=<N>_B=<B>.csv`.

### Running the solver-comparison script

```bash
# Compare all 4 initial states across 2000 time points (N=10, B=0.4 defaults)
uv run python compare_solvers.py
```

This runs QuSpin over 2055 log-spaced points (t=1 to 10⁶) and validates against mpmath on a
50-point sample, printing timing, per-state QFI tables, and relative errors.

### Library usage

```python
import numpy as np
from lmg_qfi.quspin_solver import get_spin_operators, build_h0, run_quspin_simulation

N, J, B = 10, 1.0, 0.4

Sz, Sx, Sy = get_spin_operators(N)
H0 = build_h0(J, B, N, Sz, Sx)

params = dict(
    N=N, J=J, B=B, T=1.0, nu=2, phi=np.pi, h=0.0,
    epsilon=1e-7, steps_floquet_unitary=10,
    varphi=0.0, theta=0.0, phi_0=0.0,
)

# Initial state: all spins up |N/2, N/2>
psi0 = np.zeros(N + 1, dtype=complex); psi0[0] = 1.0

time_points = list(range(1, 100)) + [200, 500, 1000]
results = run_quspin_simulation(params, time_points, psi0)

for r in results[:5]:
    print(f"t={r.time:4d}  QFI={r.qfi:.6e}  mz={r.m_z:.4f}")
```

### QuSpin solver tests

```bash
uv run pytest tests/test_quspin_solver.py -v
```

The test suite (53 tests) covers operator correctness, Floquet unitarity, QFI non-negativity,
and numerical agreement with the mpmath reference for N=20.

## Fortran solver (quadruple precision, OpenMP)

`fortran/` holds a parallel Fortran implementation of the mpmath simulation. It reads the same
`parameters.ini`, uses the same time grid and writes the same CSV columns, and it reproduces the
mpmath reference files in `results/` (to the last stored digit for N = 20 and N = 40). A run for
N = 50 with all four initial states takes a few seconds.

```bash
make -C fortran                                # needs gfortran and make
fortran/build/lmg_qfi --system-size 50 --steps 250 --num-points 300
uv run python quantum_fisher_information_simulation_mpmath.py --system-size 50 --plot
```

Results are saved to `results/fortran.<STATE>_N=<N>_B=<B>.csv`. Method, precision limits, validation
against the reference files and the Slurm batch scripts are described in
[fortran/README.md](fortran/README.md).

## SLD calibration mismatch

The symmetric logarithmic derivative (SLD) is a locally optimal observable. To
test calibration error, construct the canonical pure-state SLD at
`h_ref = 0.001`, then measure it on the state evolved at `h = 0`:

```math
\mathcal{M}_0[L_{h_{\rm ref}}]
=\left.\frac{\bigl(2\operatorname{Re}\langle\psi_h|L_{h_{\rm ref}}|
\partial_h\psi_h\rangle\bigr)^2}
{\langle L_{h_{\rm ref}}^2\rangle_h-\langle L_{h_{\rm ref}}\rangle_h^2}\right|_{h=0}.
```

The observable is held fixed when taking the derivative at the target field.
It is constructed separately for each initial state and time, so this study
does not assume a single time-independent measurement. The same initial ket
is used at both fields. The canonical choice is `L = 2 ∂ρ`, with a zero block
within the kernel of the calibration density matrix; other SLD extensions
can behave differently away from calibration.

The separate four-panel figure compares this sensitivity with the QFI at
`h = 0` and the matched observable `L_0`, which provides the local saturation
check. All curves use the common normalization `(N t)^2`.

### Reading the figure: matched vs calibrated SLD

The SLD `L_h` is the optimal observable for estimating `h`: its
method-of-moments sensitivity `M[O] = (∂_h⟨O⟩)² / Var(O)` equals the QFI, and
no other observable does better. However, `L_h` depends on `h` itself, so the
optimal measurement can only be built if the field is already known.

- **Matched SLD, `M[L_h]`** (orange, dashed): the SLD is built at the same
  field where the sensitivity is evaluated (`h = 0`). It equals the QFI exactly
  and lies on top of the grey QFI curve. This is a consistency check, not a
  realistic protocol.
- **Calibrated SLD, `M[L_{h_ref}]`** (blue): "calibration" means fixing the
  measurement operator in advance from a reference field `h_ref = 0.001`
  (separately for each initial state and time), then measuring that fixed
  operator on the state that actually evolved at `h = 0`. This models an
  experiment in which the optimal measurement was designed for a slightly
  wrong field. It is always bounded by the QFI.

Where the blue curve follows the QFI, the optimal measurement is robust to a
calibration error `Δh = h - h_ref`. Where it drops below the QFI, sensitivity
is lost because the states at `h` and `h_ref` have drifted apart. Their
relative phase grows roughly like `t·Δh`, and the CSV column
`fidelity = |⟨ψ_h|ψ_{h_ref}⟩|²` tracks this drift.

Sharp features in the blue curve (for example in the `GS_PHYS` panel near
`t/T ≈ 390`) are genuine features of `M[L_{h_ref}]`. At those times the
signal `∂_h⟨L_{h_ref}⟩` and the variance pass close to zero together. The
ratio drops to zero at `t/T = 390` and then rises within a few periods to touch
the QFI before relaxing. The curve is smooth when sampled at every period.

These features are numerically sensitive, though not to the arithmetic
precision: going from 50 to 80 digits leaves every stored value unchanged. They
are sensitive to the Trotter discretization of one period. The calibrated
curve depends on how the Floquet spectrum changes between `h` and `h_ref`, and
the Trotter error shifts that change. Over `t ≈ 10²–10³` periods the shift
accumulates into a visible phase. A scan over the number of steps per period
(every period up to `t/T = 10⁴`, 50 digits) gives, for `GS_PHYS`:

| steps | `M[L_{h_ref}]/(Nt)²` at `t/T = 370` | at `t/T = 400` | dip position `t/T` | max change vs. previous row, `GS_PHYS` / other states |
|---|---|---|---|---|
| 100 | 0.1169 | 0.1143 | 393 | – |
| 200 | 0.1109 | 0.1368 | 391 | 90% / 1.1% |
| 400 | 0.1075 | 0.1447 | 390 | 27% / 0.56% |
| 800 | 0.1057 | 0.1481 | 390 | 14% / 0.28% |
| 1600 | 0.1048 | 0.1496 | 390 | 6.8% / 0.14% |
| 3200 | 0.1043 | 0.1504 | 390 | 3.1% / 0.07% |

Changes are measured as the maximum over all times, as a fraction of each
state's peak normalized QFI. They halve with each doubling of the steps, i.e.
the error is first order in `1/steps`. The QFI (and hence the matched SLD)
converges the same way, but its changes stay at the "other states" level for
all four states (1.0% → 0.06%). The large `GS_PHYS` values come only from the
few periods around the dip, where the curve is steep and a small shift in time
produces a large change. The 100-step curve is therefore only qualitatively
correct near such features.

The committed results in `results/sld_mismatch/` use 3200 steps per period and
every period up to `t/T = 10⁴`:

```bash
uv run python scripts/plot_sld_mismatch.py --steps 3200 --time-step 1
```

This takes about 75 minutes on one core. Using the first-order scaling, the
remaining Trotter error is about the change from 1600 to 3200 steps: ≤ 0.07% of
the peak QFI everywhere except near the `GS_PHYS` dip (≤ 3%, a slight shift in
time of its sharp edge).

The script's defaults are `N=10`, `B/J=0.4`, 50-digit arithmetic, 100
integration steps per kick period, and integer times through `t/T=10000`. The
default grid combines logarithmic sampling with a maximum five-period spacing
(`--time-step 1` samples every period). CSV data, numerical diagnostics in
JSON, and a 300-dpi PNG are written to `results/sld_mismatch/`.
`--check-convergence` repeats eight sample times at 10 more digits and twice
the steps; as noted above, it can miss features that lie between those times.
`--steps`, `--dps`, `--h-reference`, `--h-target`, and `--output-dir` allow
separate studies. The original N=10 MoM files used only 10 integration steps;
the new study recomputes its own target QFI for a consistent comparison.

For the committed run, the matched SLD agrees with the QFI to better than
`2e-47` relative error, and no calibrated MoM value exceeds the target QFI.

## Tests

```bash
uv run pytest tests/ -v                    # Run all tests
uv run pytest tests/test_operators.py -v   # Run specific module
uv run pytest tests/ -v --cov=lmg_qfi      # With coverage
```


## Library Usage

```python
from lmg_qfi import (
    create_hamiltonian_h0,
    create_spin_xyz_operators,
    run_simulation,
    InitialState,
    SimulationParams,
)
import mpmath as mp

H = create_hamiltonian_h0(J=1.0, B=0.4, num_spins=10)
Sz, Sx, Sy = create_spin_xyz_operators(n=10)

params = SimulationParams(
    run_arguments={"dps": 50, "steps_floquet_unitary": 100, "num_points": 50},
    N=10,
    J=mp.mpf(1.0),
    B=mp.mpf(0.4),
)
results = run_simulation(params, [InitialState.GS_CAT])
```
