#!/usr/bin/env python3
"""Print U_T elements from Python for comparison with Fortran."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mpmath as mp
from lmg_qfi.operators import create_spin_xyz_operators, create_hamiltonian_h0, create_kick_operator
from lmg_qfi.evolution import calculate_unitary_T

N = 6
dps = 30
with mp.workdps(dps):
    params = dict(
        phi=mp.pi, J=mp.mpf(1), B=mp.mpf('0.4'), T=mp.mpf(1), h=mp.mpf(0),
        epsilon=mp.mpf(f"1e-{dps//2}"),
        N=N, nu=2, varphi=mp.mpf(0), phi_0=mp.mpf(0), theta=mp.mpf(0),
        steps_floquet_unitary=50
    )
    Zsum, Xsum, Ysum = create_spin_xyz_operators(N)
    H_0 = create_hamiltonian_h0(1, 0.4, N)

    U_T = calculate_unitary_T(mp.mpf(0), params, H_0)

    print("=== Python U_T (first 3x3) ===")
    for i in range(3):
        for j in range(3):
            r, im = float(mp.re(U_T[i,j])), float(mp.im(U_T[i,j]))
            print(f"  ({r:+.15e}, {im:+.15e})", end="")
        print()

    # Direct m_x check
    init_state = mp.zeros(N+1, 1)
    init_state[0] = mp.mpf('1.0')
    ket = U_T * init_state

    mx = mp.re((ket.transpose_conj() * Xsum * ket)[0,0]) / N
    mz = mp.re((ket.transpose_conj() * Zsum * ket)[0,0]) / N
    print(f"\nm_x(Python, direct U_T): {float(mx):.15e}")
    print(f"m_z(Python, direct U_T): {float(mz):.15e}")

    # Also print kick operator
    kick = create_kick_operator(mp.pi, Xsum)
    print("\n=== kick(1,1) ===")
    print(f"Python:  ({float(mp.re(kick[0,0])):+.15e}, {float(mp.im(kick[0,0])):+.15e})")

    # Print exp(-i*T*H0)
    exp_H0 = mp.expm(-mp.j * mp.mpf(1) * H_0)
    print("\n=== exp(-i*T*H0)(1,1) ===")
    print(f"Python:  ({float(mp.re(exp_H0[0,0])):+.15e}, {float(mp.im(exp_H0[0,0])):+.15e})")
