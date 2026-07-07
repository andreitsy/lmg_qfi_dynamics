"""Main simulation logic for QFI dynamics."""

import logging
import mpmath as mp
import numpy as np

from typing import List

from .config import EstimationParameter, InitialState, SimulationParams, QFIInformation, UF
from .operators import create_spin_xyz_operators, create_hamiltonian_h0, create_kick_operator
from .evolution import calculate_unitary_T, evolve_kets_one_period, resolve_omega
from .qfi import process_time_point_mp, qfi_information_from_kets

# Default cap on the maximum simulated time (10^degree kick periods) for
# frequency mode, whose sequential cost grows linearly with the maximum time.
DEFAULT_FREQUENCY_MAX_TIME_DEGREE = 4


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
    time_interval = list(range(1, 100))
    if max_degree == 2:
        time_interval += [100]
    elif max_degree == 3:
        time_interval += [int(x) for x in np.logspace(2, 3, num_points, endpoint=True)]
    else:
        time_interval += [int(x) for x in np.logspace(2, 4, num_points, endpoint=False)]
        time_interval += [int(x) for x in np.logspace(4, max_degree, num_points, endpoint=True)]
    logging.info(f"Time interval is: {time_interval[0]} to {time_interval[-1]}")
    return time_interval


def frequency_estimation_epsilon(base_epsilon, h, N: int, t_max: int):
    """
    Finite-difference step for the frequency derivative.

    The accumulated response to a frequency shift grows like epsilon*h*N*t
    (the generator d_omega H = h*t*cos(omega*t + phi_0)*S_alpha grows linearly
    in time), so the shift is capped at 0.1/(|h|*N*t_max) to keep the central
    difference in its linear regime up to the longest simulated time; the
    precision-limited default caps it from below.
    """
    fd_bound = mp.mpf("0.1") / (mp.fabs(h) * N * t_max)
    return min(mp.mpf(base_epsilon), fd_bound)


def simulation_with_AC_field_frequency_mp(
        params: dict,
        time_interval,
        init_state,
        init_state_str: InitialState
) -> List[QFIInformation]:
    """
    QFI dynamics over the drive frequency omega using mpmath arbitrary precision.

    The three trajectories |psi(t; omega)>, |psi(t; omega +/- epsilon)> are
    propagated period-by-period with a phase-continuous drive sin(omega*t + phi_0):
    at omega +/- epsilon the drive is not commensurate with the kick period, so
    the Floquet-power shortcut does not apply and every kick period is integrated
    explicitly (cost linear in the largest requested time).

    Parameters
    ----------
    params : dict
        Simulation parameters; requires h != 0 (the drive carries the omega
        dependence) and reads the probe frequency from params["omega"]
        (absent/None -> resonant 2*pi/(nu*T)).
    time_interval : list
        Time points (kick-period indices) at which to report the QFI;
        deduplicated and sorted internally.
    init_state : mp.matrix
        Initial quantum state.
    init_state_str : InitialState
        Name of the initial state for logging.

    Returns
    -------
    list
        List of QFIInformation for each requested time point, with the qfi
        field normalized as F_omega / (N^2 t^4) (the raw value is kept in
        qfi_raw_value).
    """
    h = params["h"]
    N = params["N"]
    if h == 0:
        raise ValueError(
            "Frequency estimation requires a nonzero AC amplitude h "
            "(the drive carries the omega dependence); set h in parameters.ini "
            "or pass --amplitude, e.g. 0.1."
        )
    grid = sorted({int(t) for t in time_interval if int(t) > 0})
    if not grid:
        return []
    t_max = grid[-1]
    omega_0 = resolve_omega(params)

    epsilon = frequency_estimation_epsilon(params["epsilon"], h, N, t_max)
    if epsilon < mp.mpf(params["epsilon"]):
        logging.info(
            f"Frequency-mode epsilon reduced to {mp.nstr(epsilon, 6)} "
            f"(finite-difference bound 0.1/(h*N*t_max) below the precision default "
            f"{mp.nstr(mp.mpf(params['epsilon']), 6)})")
    roundoff_floor = mp.mpf(10) ** (3 - mp.mp.dps)
    if epsilon < roundoff_floor:
        logging.warning(
            f"Frequency-mode epsilon {mp.nstr(epsilon, 6)} is within ~3 digits of the "
            f"working precision (dps={mp.mp.dps}); increase dps for reliable derivatives.")

    Zsum, Xsum, Ysum = create_spin_xyz_operators(N)
    H_0 = create_hamiltonian_h0(params["J"], params["B"], N)
    kick = create_kick_operator(params["phi"], Xsum)
    omegas = [omega_0, omega_0 + epsilon, omega_0 - epsilon]
    kets = [mp.matrix(init_state) for _ in omegas]

    logging.info(
        f"Frequency estimation for {init_state_str}: omega_0={mp.nstr(omega_0, 8)}, "
        f"epsilon={mp.nstr(epsilon, 6)}, t_max={t_max}, {len(grid)} time points")

    grid_set = set(grid)
    log_every = max(1, t_max // 100)
    results = []
    for p in range(1, t_max + 1):
        kets = evolve_kets_one_period(
            kets, omegas, p, h, params, H_0, Xsum, Ysum, Zsum, kick)
        if p in grid_set:
            results.append(qfi_information_from_kets(
                kets[0], kets[1], kets[2], epsilon, N, p,
                Zsum, Xsum, Ysum, time_power=4))
        if p % log_every == 0:
            logging.info(f"{p / t_max * 100.0:.2f}%: period {p}/{t_max} "
                         f"for {init_state_str} with params: {params}")
    return results


def simulation_with_AC_field_mp(
        params: dict,
        time_interval,
        init_state,
        init_state_str: InitialState
) -> List[QFIInformation]:
    """
    Sequential observable simulation using mpmath arbitrary precision.

    Dispatches on params["parameter"] (EstimationParameter, default AMPLITUDE):
    amplitude mode uses the Floquet-power fast path, frequency mode the
    phase-continuous sequential driver.

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
    parameter = params.get("parameter", EstimationParameter.AMPLITUDE)
    if not isinstance(parameter, EstimationParameter):
        parameter = EstimationParameter(parameter)
    if parameter == EstimationParameter.FREQUENCY:
        return simulation_with_AC_field_frequency_mp(
            params, time_interval, init_state, init_state_str)

    omega = params.get("omega")
    if omega is not None:
        # The U_F eigendecomposition powers assume the drive is periodic over
        # one Floquet cycle of nu kick periods.
        cycles = omega * params["nu"] * params["T"] / (2 * mp.pi)
        if mp.fabs(cycles - mp.nint(cycles)) > mp.mpf("1e-9") or cycles < mp.mpf("0.5"):
            raise ValueError(
                f"Amplitude mode requires omega commensurate with the Floquet cycle: "
                f"omega*nu*T must be a positive multiple of 2*pi "
                f"(got omega={mp.nstr(omega, 8)}, nu={params['nu']}, T={params['T']}).")

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

    if params.parameter == EstimationParameter.FREQUENCY and params.h == 0:
        raise ValueError(
            "Frequency estimation requires a nonzero AC amplitude h "
            "(the drive carries the omega dependence); set h in parameters.ini "
            "or pass --amplitude, e.g. 0.1.")

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
            steps_floquet_unitary=params.run_arguments["steps_floquet_unitary"],
            parameter=params.parameter,
            omega=None if params.omega is None else mp.mpf(f"{float(params.omega)}"))
        vec_size = params.N + 1
        H = create_hamiltonian_h0(params.J, params.B, params.N)
        energies, evecs = mp.eigh(H)

        # Ground states and initial ket
        sorted_indices = sorted(range(len(energies)), key=lambda i: mp.re(energies[i]))
        gap = energies[sorted_indices[1]] - energies[sorted_indices[0]]
        last_time_degree = int(mp.log10(mp.fabs(4 * mp.pi / gap))) + 1

        max_time_degree = params.run_arguments.get("max_time_degree")
        if max_time_degree is None and params.parameter == EstimationParameter.FREQUENCY:
            max_time_degree = DEFAULT_FREQUENCY_MAX_TIME_DEGREE
            if last_time_degree > max_time_degree:
                logging.info(
                    f"Frequency mode: capping max time at 10^{max_time_degree} periods "
                    f"(gap-based value 10^{last_time_degree}; sequential cost is linear "
                    f"in the maximum time — raise with --max-time-degree if needed)")
        if max_time_degree is not None:
            if max_time_degree < 2:
                raise ValueError("max_time_degree must be at least 2")
            last_time_degree = min(last_time_degree, max_time_degree)
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
