#!/usr/bin/env python3
"""Direct comparison of Floquet unitary U_T between Python and Fortran for N=4.

Tests: is exp(-i*T*H0) from Fortran's eigh-based expm_hermitian the same as
Python's mp.expm(-j*T*H0)?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mpmath as mp

from lmg_qfi.operators import create_spin_xyz_operators, create_hamiltonian_h0, create_kick_operator
from lmg_qfi.evolution import calculate_unitary_T, evalution_T_step

def main():
    N = 6
    dps = 30
    with mp.workdps(dps):
        J, B, T = mp.mpf(1), mp.mpf('0.4'), mp.mpf(1)
        nu = 2
        steps = 50

        params = dict(
            phi=mp.pi, J=J, B=B, T=T, h=mp.mpf(0),
            epsilon=mp.mpf(f"1e-{dps//2}"),
            N=N, nu=nu, varphi=mp.mpf(0), phi_0=mp.mpf(0),
            theta=mp.mpf(0), steps_floquet_unitary=steps
        )

        Zsum, Xsum, Ysum = create_spin_xyz_operators(N)
        H_0 = create_hamiltonian_h0(J, B, N)

        # Method 1: Python Trotter (what evalution_T_step does with h=0)
        print("=== exp(-i*T*H0) via Python Trotter (50 steps) ===")
        t_delta = T / steps
        omega = mp.mpf(2) * mp.pi / (nu * T)
        U_trotter = mp.eye(N + 1)
        for p in range(1, nu + 1):
            U_trotter = evalution_T_step(U_trotter, mp.mpf(0), T, mp.mpf(0), mp.mpf(0), mp.mpf(0),
                                          H_0, Xsum, Ysum, Zsum, omega, p, t_delta, steps)
            U_trotter = create_kick_operator(mp.pi, Xsum) * U_trotter

        # Method 2: Direct matrix exponential (what Fortran fast path does)
        print("=== exp(-i*T*H0) via direct mp.expm ===")
        exp_H0_T = mp.expm(-mp.j * T * H_0)
        kick = create_kick_operator(mp.pi, Xsum)
        U_direct = mp.eye(N + 1)
        for p in range(1, nu + 1):
            U_direct = kick * (exp_H0_T * U_direct)

        # Compare
        print(f"\n=== U_T comparison (N={N}) ===")
        max_diff = mp.mpf(0)
        for i in range(N + 1):
            for j in range(N + 1):
                diff = abs(U_trotter[i, j] - U_direct[i, j])
                if diff > max_diff:
                    max_diff = diff
        print(f"Max element-wise |U_trotter - U_direct|: {float(max_diff):.3e}")

        # Now test with eigendecomposition (what find_power_r does)
        print("\n=== U_T^r via eigendecomposition ===")

        # Method 1: eig of U_trotter
        evals_t, evecs_t = mp.eig(U_trotter)
        evecs_t_inv = mp.inverse(evecs_t)

        # Method 2: eig of U_direct
        evals_d, evecs_d = mp.eig(U_direct)
        evecs_d_inv = mp.inverse(evecs_d)

        for r in [1, 5, 10]:
            # U_T^r via trotter eigendecomposition
            D_t = mp.diag([e ** r for e in evals_t])
            Ur_trotter = evecs_t * D_t * evecs_t_inv

            # U_T^r via direct eigendecomposition
            D_d = mp.diag([e ** r for e in evals_d])
            Ur_direct = evecs_d * D_d * evecs_d_inv

            max_diff = mp.mpf(0)
            for i in range(N + 1):
                for j in range(N + 1):
                    diff = abs(Ur_trotter[i, j] - Ur_direct[i, j])
                    if diff > max_diff:
                        max_diff = diff
            print(f"  r={r}: Max |U^r_trotter - U^r_direct|: {float(max_diff):.3e}")

        # Now evolve a state and check magnetizations
        print("\n=== Magnetizations for |PHYS> at t=5 ===")
        init_state = mp.zeros(N + 1, 1)
        init_state[0] = mp.mpf('1.0')

        for r in [1, 5]:
            D_t = mp.diag([e ** r for e in evals_t])
            Ur_t = evecs_t * D_t * evecs_t_inv
            ket_t = Ur_t * init_state

            D_d = mp.diag([e ** r for e in evals_d])
            Ur_d = evecs_d * D_d * evecs_d_inv
            ket_d = Ur_d * init_state

            mx_t = mp.re((ket_t.transpose_conj() * Xsum * ket_t)[0, 0]) / N
            mx_d = mp.re((ket_d.transpose_conj() * Xsum * ket_d)[0, 0]) / N
            mz_t = mp.re((ket_t.transpose_conj() * Zsum * ket_t)[0, 0]) / N
            mz_d = mp.re((ket_d.transpose_conj() * Zsum * ket_d)[0, 0]) / N

            print(f"  r={r}: Trotter  mx={float(mx_t):.15f}, mz={float(mz_t):.15f}")
            print(f"  r={r}: Direct   mx={float(mx_d):.15f}, mz={float(mz_d):.15f}")
            print(f"  r={r}: diff mx={float(abs(mx_t-mx_d)):.3e}, mz={float(abs(mz_t-mz_d)):.3e}")


if __name__ == '__main__':
    main()
