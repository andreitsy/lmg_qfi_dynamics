# Fortran solver (quadruple precision, OpenMP)

A Fortran 2008 implementation of the QFI simulation in
`quantum_fisher_information_simulation_mpmath.py`: same model, same Trotter
integration of a kicked period, same time grid and the same CSV output. It
computes everything in IEEE binary128 (`real128`, about 34 significant digits)
and is parallelised with OpenMP for a single node, e.g. one Slurm job.

For N = 50 (699 time points, four initial states, 250 Trotter steps per
period) a run takes about 4 s on a 20-CPU cluster node and 7 s on an 8-core
laptop.

## Quick start

```bash
make -C fortran                     # needs gfortran >= 9 and make; builds fortran/build/lmg_qfi
make -C fortran test                # Fortran test suite

# reads ./parameters.ini like the Python scripts; options override it
fortran/build/lmg_qfi --system-size 50 --steps 250 --num-points 300

# plot with the existing tooling
uv run python quantum_fisher_information_simulation_mpmath.py --system-size 50 --plot
```

Results go to `<output-dir>/fortran.<STATE>_N=<N>_B=<B>.csv` with the columns
`time,m_x,m_y,m_z,qfi,qfi_raw` of the Python output (`STATE` is `PHYS`,
`GS_PHYS`, `GS_CAT` or `CAT_SUM`). `qfi_raw` carries 34 digits.

## Options

| Option | Meaning |
|---|---|
| `--config FILE` | ini file (default `./parameters.ini` if present) |
| `--system-size N`, `--x-coupling B` | as in the Python scripts |
| `--init-state NAME[,NAME…]` | `Phys`, `GS_phys`, `GS_cat`, `CatSum`; default all four |
| `--steps K` | Trotter steps per period (`steps-floquet-unitary`) |
| `--num-points P` | points per logarithmic segment of the time grid |
| `--max-degree D` | last decade of the time grid (default: from the gap of H0) |
| `--times-file FILE` | take the times from the first column of `FILE`, e.g. a result CSV |
| `--output-dir DIR`, `--prefix STR` | output location and file prefix (default `fortran.`) |
| `--gs-phys-orientation up\|down` | which symmetry-broken state `GS_phys` is (see below) |
| `--threads T` | OpenMP threads (default `OMP_NUM_THREADS`) |
| `--set KEY=VALUE` | any ini key: `J`, `T`, `h`, `frequency`, `phi-0`, `theta`, `varphi`, `phi-kick-phase`, `b-init`, … |

Values may be decimal numbers, `pi`, or products and quotients such as `pi/2`.
All four initial states share one Floquet operator, so running them together
costs little more than running one.

## Method

The quantity is the same as in the Python code,
`F_h = 4 (<∂ψ|∂ψ> − |<ψ|∂ψ>|²)` at the stroboscopic time `t`, with
`U(t) = P_m U_F^r`, `t = r·ν + m`. Three things are done differently to make
34 digits sufficient and the run fast:

1. **Exact derivative instead of a finite difference.** The Python code
   differentiates with `(ψ(h+ε) − ψ(h−ε))/2ε`, `ε = 10^(−dps/2)`, which needs
   about 60 digits at late times. Here `∂U/∂h` is propagated analytically:
   the derivative of each matrix exponential comes from the Daleckii–Krein
   formula in the eigenbasis of the step Hamiltonian, products use the
   product rule. No digits are lost to cancellation.
2. **Repeated squaring instead of an eigen-decomposition of `U_F`.**
   `U_F^(2^k)` and its derivative are tabulated; `U_F^r|ψ₀>` is applied through
   the binary digits of `r`. This avoids the non-Hermitian eigenproblem and
   the division by differences of nearly degenerate eigenvalues.
3. **Closed-form periods.** Consecutive Trotter steps with the same
   Hamiltonian are combined in its eigenbasis. For `h = 0` a whole period
   costs one diagonalisation regardless of `steps`.

The only eigen-solver needed is a Jacobi method for Hermitian matrices
(`lmg_linalg.f90`). The ground-state doublet of `H0` is computed separately
in the even and odd sectors of the reflection `m → −m`, so the two states are
not mixed by rounding even when their splitting is far below 10⁻³⁰.

Differences from the Python script that affect input, not results:

- `J`, `B`, `T`, `h` are used as given (Python rounds them to 2–3 decimals).
- `theta` and `varphi` are read from the ini file (the Python reader ignores
  them and uses 0).
- `dps` is ignored; the precision is fixed.
- `GS_phys` is `(|g> ± |e>)/√2`. mpmath leaves the sign to the eigen-solver,
  so the reference files contain `<Sz> < 0` for N = 20, 40 and `<Sz> > 0` for
  N = 60, 80. Here it is explicit: `up` (default) has `<Sz> > 0`. For `h = 0`
  the two choices differ only in the signs of `m_y` and `m_z`; for `h ≠ 0`
  they are different initial conditions.

## Precision

Every period adds the rounding error of `U_F` again, so errors grow linearly
with time. With ε = 1.9·10⁻³⁴ the relative error is about

```
ψ(t), m_x, m_y, m_z:   (N + 1) · ε · t · m
qfi, qfi_raw:          2 (N + 1) · ε · t · (a + m)
```

where `m` is the number of distinct Trotter exponentials per kicked period
(1 for `h = 0`, where a period is one exponential in closed form; about
`steps` otherwise) and `a = ‖∂ψ‖ / ‖∂ψ⊥‖`. `F_h` depends only on the part
`∂ψ⊥` of `∂ψ` orthogonal to ψ; where that part is small compared with `∂ψ`
(typically where the QFI curve dips) its relative error is larger. `a` is
measured at every time point, and each run ends with the largest estimates
and with measured indicators (unitarity of the highest power of `U_F`, norm
of the evolved states):

```
accuracy:
  ...
  => psi, m_x, m_y, m_z accurate to about       9.82E-18
  => qfi, qfi_raw: relative accuracy about      9.39E-17
```

The program warns when either number exceeds 10⁻⁶. Printed estimate and
largest deviation of `qfi_raw` from mpmath (dps = 60, all four initial
states):

| N | h | t_max | printed estimate | deviation from mpmath |
|---|---|---|---|---|
| 20 | 0 | 10⁶ | 5·10⁻²⁵ | 7·10⁻²⁵ |
| 40 | 0 | 10¹² | 2·10⁻¹⁹ | 4·10⁻¹⁹ |
| 50 | 0 | 10¹⁵ | 1·10⁻¹⁶ | 9·10⁻¹⁸ (8 times, `steps = 10`) |
| 60 | 0 | 10¹⁸ | 2·10⁻¹³ | 8·10⁻¹⁴ |
| 80 | 0 | 10²³ | 2·10⁻⁸ | 1·10⁻⁹ |
| 20 | 0.01 | 10⁶ | 1·10⁻²³ | 1·10⁻²⁴ (351 times) |

So up to N = 50 the columns stored as doubles are exact to their last digit.
At the latest times of the grid N = 60 is good to about 14 digits and N = 80
to about 9; beyond that the latest times need more than quadruple precision.
For `h ≠ 0` the estimate is conservative by about an order of magnitude
(N = 50, `h = 0.01`, 250 steps: 6·10⁻¹⁴).

## Validation

`results/InitialState.*.csv` were produced with mpmath at `dps = 60`. Their
other settings are not recorded in the repository; they were identified by
matching the first rows: `steps = 250, num-points = 300` for N = 20, 40 and
`steps = 100, num-points = 50` for N = 60, 80.

```bash
fortran/build/lmg_qfi --system-size 40 --steps 250 --num-points 300 --output-dir fortran/build/check
uv run python fortran/scripts/compare_results.py results fortran/build/check --n 40
```

The time grids are identical for all four sizes, including the times beyond
2⁵³ that are integer values of doubles (for example 99999999999999991611392).
The magnetisations and `qfi` agree to the last digit of the stored doubles
for N = 20, 40 and to the precision table above for N = 60, 80.

`qfi_raw` (60 digits in the reference) agrees to 3·10⁻²⁵ (N = 20) and
2·10⁻¹⁹ (N = 40) for `Phys` and `CatSum`. For `GS_phys` and `GS_cat` it agrees
to 10⁻¹¹…10⁻¹⁷, because the reference built those two initial states from `B`
converted through a double (0.4000000000000000222…) while evolving with 0.4.
Supplying the same value, `--set b-init=0.40000000000000002220446049250313080847263336181640625`,
restores the agreement to 7·10⁻²⁵ and 4·10⁻¹⁹. The same rounding is the origin
of the `m_y ≈ 10⁻¹⁷` entries of the reference `GS_PHYS` files for N ≥ 40.

There is no reference file for N = 50. The mpmath implementation evaluated at
eight times between 1 and 10¹⁵ (`steps = 10`, `dps = 60`) agrees with the
solver to 9·10⁻¹⁸ in `qfi_raw` and to the last digit in the other columns.
`results/fortran.*_N=50_B=0.40.csv` and `results/qfi_dynamics_N=50_B=0.40.png`
were produced with this solver (`steps = 250, num-points = 300`).

For sizes without reference files, `fortran/scripts/mpmath_reference_points.py`
evaluates the mpmath implementation at a few chosen times (see its
docstring); `--times-file` then runs the solver on exactly those times.

`uv run pytest tests/test_fortran_solver.py` builds the solver and runs these
comparisons (N = 20, 40; N = 60, 80 with `LMG_QFI_SLOW_TESTS=1`), plus checks
against mpmath at small N for `h ≠ 0`, a tilted probe direction and `ν = 3`.

## Parallelism and performance

OpenMP is used at three levels:

- Floquet operator: Trotter steps are distributed over the threads when their
  Hamiltonians differ (`h ≠ 0`); the Jacobi rotations and matrix products are
  threaded otherwise.
- Repeated squaring: threaded matrix products.
- Time points: one `(time, state)` evaluation is independent of all others;
  this loop dominates the run time and scales with the number of threads.

Results do not depend on the number of threads.

One Slurm job with 20 CPUs (10 cores × 2 hardware threads of an x86-64
cluster node):

| N | steps | points × states | wall time |
|---|---|---|---|
| 20 | 250 | 699 × 4 | 0.5 s |
| 40 | 250 | 699 × 4 | 2.1 s |
| 50 | 250 | 699 × 4 | 3.9 s |
| 60 | 100 | 199 × 4 | 2.2 s |
| 80 | 100 | 199 × 4 | 5.0 s |

Thread scaling for the N = 50 case: 47.9 s (1 thread), 24.5 s (2), 10.0 s (5),
5.1 s (10, one per core), 3.6 s (20).

With `h ≠ 0` every Trotter step has its own Hamiltonian and needs its own
diagonalisation: the same N = 50 case with `h = 0.01` takes 32 s on 20 CPUs
(29 s of it for the 500 steps of the Floquet operator).

## Running on a Slurm cluster

Build on the login node (`make -C fortran`); the executable finds its
runtime libraries through an rpath, so no compiler module has to be loaded
in the job. Keep each run in its own directory `<project>/tmp/<experiment>/<run>/`
and submit from there:

```bash
# on the cluster, in the project copy
make -C fortran
run=tmp/fortran-qfi/n50 && mkdir -p "$run/logs" && cd "$run"
cp ../../../fortran/slurm/lmg_qfi.sbatch ../../../parameters.ini .
sbatch --parsable --partition=<partition> lmg_qfi.sbatch --system-size 50 --steps 250 --num-points 300
squeue -u "$USER"
sacct -j <jobid> --format=JobID,State,Elapsed,TotalCPU,ExitCode
```

Results are written to `results/` and the log to `logs/` of the run
directory. `slurm/lmg_qfi.sbatch` requests one node with 20 CPUs and passes
its arguments to the solver; adjust `--cpus-per-task` and `--time` to the
run. `slurm/lmg_qfi_array.sbatch` runs a sweep as a job array: one line of
solver options per case in `cases.txt`.

## Dense time grids

`results/dense/` holds runs for N = 60 and N = 100 (`steps = 250`) on a grid
about 100 times denser than the standard one: every period up to 10⁴ and
60 000 log-spaced times up to the last decade, 69 999 points in all.

```bash
uv run python fortran/scripts/dense_time_grid.py --system-size 100 times_N100.txt
fortran/build/lmg_qfi --system-size 100 --steps 250 --times-file times_N100.txt --output-dir results/dense
uv run python fortran/scripts/plot_results.py results/dense 100 \
    --label-y 0.02 --label-offset 0.1 --legend-loc 0.44 0.567
```

`plot_results.py` draws the curves with the package's plotting function but
reads only the solver's files, so they are not mixed with mpmath files of
the same N. `--label-y` and `--label-offset` place the 2π/Δ labels (height
on the y axis, distance right of their line in decades; defaults 0.155 and
3% of the line position) and `--legend-loc` the legend (lower-left corner in
axes fractions, default 0.35 0.567). The N = 100 plot puts the labels near
the bottom and the legend in the free space right of t = 10¹²·⁵.

On 20 CPUs N = 60 takes 13 min for all four initial states and
N = 100 about 10 min per initial state (one array task each). N = 100 runs to
t = 10²⁹, where quadruple precision is nearly exhausted: the states are
accurate to about 2·10⁻³ at the last times (10⁻⁶ at t = 5·10²⁵), well below
what is visible in the plot.

## Layout

```
fortran/
  Makefile
  src/lmg_kinds.f90       kinds and constants
  src/lmg_linalg.f90      matrix product, Hermitian Jacobi eigen-solver
  src/lmg_operators.f90   model parameters, spin operators, H0, magnetisation
  src/lmg_floquet.f90     Floquet operators, their h-derivative, powers, propagation
  src/lmg_states.f90      initial states
  src/lmg_timegrid.f90    NumPy-compatible time grid, 128-bit times
  src/lmg_dynamics.f90    QFI and magnetisation for all times and states
  src/lmg_config.f90      parameters.ini and command line
  src/lmg_io.f90          CSV output
  src/main.f90            program lmg_qfi
  tests/test_lmg.f90      test suite (make test)
  scripts/                comparison with mpmath, dense time grids, plotting
  slurm/                  Slurm batch scripts
```
