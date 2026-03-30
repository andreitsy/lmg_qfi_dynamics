# LMG QFI Dynamics — Fortran

Fortran implementation of QFI dynamics in the Lipkin-Meshkov-Glick model.

## Build

Requires CMake 3.14+ and a Fortran compiler (gfortran recommended).

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The `lmg_qfi_sim` binary is placed in this directory.

## Run

```bash
./lmg_qfi_sim --config ../parameters.ini
```

Override parameters via CLI flags:

```bash
./lmg_qfi_sim --system-size 20 --x-coupling 0.4 --init-state GS_phys
./lmg_qfi_sim --init-state Phys --init-state CatSum
```

Available initial states: `GS_phys`, `GS_cat`, `Phys`, `CatSum`.

## Test

```bash
ctest --test-dir build --output-on-failure
```

Test binaries remain inside `build/` and are not copied to the project root.

## Clean

```bash
rm -rf build lmg_qfi_sim
```
