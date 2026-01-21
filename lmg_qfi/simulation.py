"""Main simulation logic for QFI dynamics."""

import logging
import mpmath as mp
import numpy as np

from typing import List

from .config import InitialState, SimulationParams, QFIInformation, UF
from .operators import create_spin_xyz_operators, create_hamiltonian_h0
from .evolution import calculate_unitary_T
from .qfi import process_time_point_mp


def generate_time_interval(num_points: int, max_degree: int) -> list:
    """
    Generate a time interval with logarithmic spacing.
    
    Parameters
    ----------
    num_points : int
        Number of points per decade.
    max_degree : int
        Maximum power of 10 for the time interval.
    
    Returns
    -------
    list
        Time points for simulation.
    """
    if max_degree <= 1:
        raise ValueError("max_degree should be greater than 1!")
    time_interval = (list(range(1, 100)) +
                     [int(x) for x in np.logspace(2, 4, num_points, endpoint=False)] +
                     [int(x) for x in np.logspace(4, max_degree, num_points, endpoint=True)])
    logging.info(f"Time interval is: {time_interval[0]} to {time_interval[-1]}")
    return time_interval


def simulation_with_AC_field_mp(
        params: dict,
        time_interval,
        init_state,
        init_state_str: InitialState
) -> List[QFIInformation]:
    """
    Sequential observable simulation using mpmath arbitrary precision.
    
    Parameters
    ----------
    params : dict
        Simulation parameters.
    time_interval : list
        Time points to simulate.
    init_state : mp.matrix
        Initial quantum state.
    init_state_str : InitialState
        Name of the initial state for logging.
    
    Returns
    -------
    list
        List of QFIInformation for each time point.
    """
    h = params["h"]
    epsilon = params["epsilon"]
    Zsum, Xsum, Ysum = create_spin_xyz_operators(params["N"])
    H_0 = create_hamiltonian_h0(params["J"], params["B"], params["N"])

    fu_eigenvalues, fu_eigenvectors = mp.eig(calculate_unitary_T(h, params, H_0))
    floque_u = UF(eigenvalues=fu_eigenvalues, U=fu_eigenvectors, U_inv=mp.inverse(fu_eigenvectors))

    fu_delta_p_eigenvalues, fu_delta_p_eigenvectors = mp.eig(calculate_unitary_T(h + epsilon, params, H_0))
    floque_u_p = UF(eigenvalues=fu_delta_p_eigenvalues,
                    U=fu_delta_p_eigenvectors, U_inv=mp.inverse(fu_delta_p_eigenvectors))

    fu_delta_m_eigenvalues, fu_delta_m_eigenvectors = mp.eig(calculate_unitary_T(h - epsilon, params, H_0))
    floque_u_m = UF(eigenvalues=fu_delta_m_eigenvalues,
                    U=fu_delta_m_eigenvectors, U_inv=mp.inverse(fu_delta_m_eigenvectors))

    results = []
    for i, time in enumerate(time_interval):
        res = process_time_point_mp(
            time,
            params,
            H_0,
            floque_u,
            floque_u_p,
            floque_u_m,
            init_state,
            Zsum,
            Xsum,
            Ysum,
        )
        if i % 10 == 0:
            logging.info(f"{i / len(time_interval) * 100.0:.2f}%: "
                         f"processing time={time} for {init_state_str} with params: {params}")
        results.append(res)

    return results


def run_simulation(params: SimulationParams, init_states: List[InitialState]):
    """
    Run the full QFI simulation for specified initial states.
    
    Parameters
    ----------
    params : SimulationParams
        Simulation parameters.
    init_states : list
        List of initial states to simulate.
    
    Returns
    -------
    dict
        Dictionary mapping InitialState to list of QFIInformation results.
    """
    dps = params.run_arguments["dps"]
    num_points = params.run_arguments["num_points"]

    with mp.workdps(dps):
        params_dict = dict(
            phi=params.phi_kick_phase,
            J=mp.mpf(f"{float(params.J):.2f}"),
            B=mp.mpf(f"{float(params.B):.3f}"),
            T=mp.mpf(f"{float(params.T):.2f}"),
            h=mp.mpf(f"{float(params.h):.3f}"),
            epsilon=mp.mpf(f"1e-{dps // 2}"),
            N=params.N, nu=params.freq,
            varphi=params.varphi, phi_0=params.phi_0, theta=params.theta,
            steps_floquet_unitary=params.run_arguments["steps_floquet_unitary"])
        vec_size = params.N + 1
        H = create_hamiltonian_h0(params.J, params.B, params.N)
        energies, evecs = mp.eigh(H)
        
        # Ground states and initial ket
        sorted_indices = sorted(range(len(energies)), key=lambda i: mp.re(energies[i]))
        gap = energies[sorted_indices[1]] - energies[sorted_indices[0]]
        last_time_degree = int(mp.log10(mp.fabs(4 * mp.pi / gap))) + 1
        ground_state = evecs[:, sorted_indices[0]]
        first_excited_state = evecs[:, sorted_indices[1]]
        
        results = dict()
        for state in init_states:
            if state == InitialState.GS_PHYS:
                init_state = ground_state + first_excited_state
            elif state == InitialState.GS_CAT:
                init_state = ground_state
            elif state == InitialState.PHYS:
                init_state = mp.zeros(vec_size, 1)
                init_state[0] = mp.mpf('1.0')
            elif state == InitialState.CAT_SUM:
                init_state = mp.zeros(vec_size, 1)
                init_state[0] = mp.mpf('1.0')
                init_state[vec_size - 1] = mp.mpf('1.0')
            else:
                raise ValueError(f"Unhandled initial state type: {state}")
            init_state = init_state / mp.norm(init_state)
            sim_results = simulation_with_AC_field_mp(
                params=params_dict,
                time_interval=generate_time_interval(num_points, last_time_degree),
                init_state=init_state,
                init_state_str=state)
            results[state] = sim_results
    return results


def run_gaps():
    """
    Compute and plot energy gaps for different system sizes.
    """
    import matplotlib.pyplot as plt
    
    N_values = [1, 5, 10, 20, 30, 50, 100]
    J = 1.0
    B = 0.1
    gaps = []
    dps = 50
    
    for N in N_values:
        with mp.workdps(dps):
            H = create_hamiltonian_h0(J, B, N)
            energies, evecs = mp.eigh(H)
            evals_sorted = sorted(energies, key=lambda ev: mp.re(ev))
            print("N=", N, "gap=", evals_sorted[0] - evals_sorted[1])
            gap = -mp.log(mp.fabs(evals_sorted[0] - evals_sorted[1]))
            gaps.append(float(gap))
    
    plt.figure(figsize=(8, 6))
    plt.scatter(N_values, gaps, color='teal', s=100, label='Energy Gap')
    plt.plot(N_values, gaps, color='teal', linestyle='--', alpha=0.5)
    plt.xlabel('N (Number of Spins)')
    plt.ylabel('Energy Gap (log scale)')
    plt.title('-log(Delta energy) vs N (LMG Model)')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
