"""QuSpin-based solver for LMG QFI dynamics.

Uses QuSpin to build spin operators and scipy/numpy for time evolution,
providing a double-precision alternative to the mpmath implementation.
"""

import logging

from typing import Dict, List, Tuple

import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian as quspin_hamiltonian
from scipy.linalg import expm

from .config import EstimationParameter, InitialState, QFIInformation, SimulationParams
from .simulation import generate_time_interval

# QuSpin is ~1000x faster than mpmath, so frequency mode affords a larger
# default time cap (10^degree kick periods) than mpmath's 10^4.
DEFAULT_FREQUENCY_MAX_TIME_DEGREE = 6


def get_spin_operators(N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build collective spin operators Sz, Sx, Sy for total spin S=N/2 using QuSpin.

    States are ordered from m=+S to m=-S (Sz diagonal = [+S, +S-1, ..., -S]),
    matching the mpmath convention in operators.py.

    Parameters
    ----------
    N : int
        Number of spins (Hilbert space dimension = N+1).

    Returns
    -------
    Sz, Sx, Sy : np.ndarray of shape (N+1, N+1)
    """
    S_str = str(N // 2) if N % 2 == 0 else f"{N}/2"
    basis = spin_basis_1d(L=1, S=S_str)
    no_checks = dict(check_symm=False, check_herm=False, check_pcon=False)

    Sz_op = quspin_hamiltonian([["z", [[1.0, 0]]]], [], basis=basis,
                                dtype=np.float64, **no_checks)
    Sp_op = quspin_hamiltonian([["+", [[1.0, 0]]]], [], basis=basis,
                                dtype=np.complex128, **no_checks)
    Sm_op = quspin_hamiltonian([["-", [[1.0, 0]]]], [], basis=basis,
                                dtype=np.complex128, **no_checks)

    Sz = Sz_op.toarray()
    Sp = Sp_op.toarray()
    Sm = Sm_op.toarray()

    Sx = (Sp + Sm) / 2.0
    Sy = -1j * (Sp - Sm) / 2.0

    return Sz, Sx, Sy


def build_h0(J: float, B: float, N: int,
             Sz: np.ndarray, Sx: np.ndarray) -> np.ndarray:
    """Build LMG Hamiltonian H0 = -(2J/N)*Sz^2 - 2B*Sx."""
    return -(2.0 * J / N) * (Sz @ Sz) - 2.0 * B * Sx


def build_initial_states(N: int, J: float, B: float) -> dict:
    """Build the four standard initial states as normalized numpy vectors.

    Returns
    -------
    dict mapping InitialState to np.ndarray of shape (N+1,).
    """
    Sz, Sx, _ = get_spin_operators(N)
    H0 = build_h0(J, B, N, Sz, Sx)
    _, evecs = np.linalg.eigh(H0)
    gs, ex = evecs[:, 0], evecs[:, 1]

    phys = np.zeros(N + 1, dtype=complex)
    phys[0] = 1.0
    cat_sum = np.zeros(N + 1, dtype=complex)
    cat_sum[0] = cat_sum[N] = 1.0

    states = {
        InitialState.PHYS: phys,
        InitialState.GS_PHYS: (gs + ex).astype(complex),
        InitialState.GS_CAT: gs.astype(complex),
        InitialState.CAT_SUM: cat_sum,
    }
    return {k: v / np.linalg.norm(v) for k, v in states.items()}


def _kick_operator(phi: float, Sx: np.ndarray) -> np.ndarray:
    """Kick operator K = exp(-i * phi * Sx)."""
    return expm(-1j * phi * Sx)


def _resolve_omega(params: dict) -> float:
    """Drive frequency from params["omega"], defaulting to resonant 2*pi/(nu*T)."""
    omega = params.get("omega")
    if omega is None:
        omega = 2.0 * np.pi / (params["nu"] * float(params["T"]))
    return float(omega)


def _ac_field(Sx: np.ndarray, Sy: np.ndarray, Sz: np.ndarray,
              omega: float, phi_0: float, t_k: float,
              theta: float, varphi: float) -> np.ndarray:
    """Time-dependent AC probe field S_alpha(t)."""
    s = np.sin(omega * t_k + phi_0)
    direction = (np.sin(theta) * np.cos(varphi) * Sx
                 + np.sin(theta) * np.sin(varphi) * Sy
                 + np.cos(theta) * Sz)
    return s * direction


def _evolution_T_step(U: np.ndarray, h: float, T: float,
                      varphi: float, theta: float, phi_0: float,
                      H0: np.ndarray, Sx: np.ndarray, Sy: np.ndarray, Sz: np.ndarray,
                      omega: float, p: int, t_delta: float,
                      steps_floquet_unitary: int) -> np.ndarray:
    """Evolve U over one sub-period p, mirroring evalution_T_step from evolution.py."""
    t_start = T * (p - 1)
    t_end = T * p
    linspace = [t_start + i * (t_end - t_start) / (steps_floquet_unitary - 1)
                for i in range(steps_floquet_unitary)]
    for t_k in linspace:
        V = H0 + h * _ac_field(Sx, Sy, Sz, omega, phi_0, t_k, theta, varphi)
        U = expm(-1j * t_delta * V) @ U
    return U


def compute_floquet_unitary(h: float, params: dict,
                            H0: np.ndarray,
                            Sx: np.ndarray, Sy: np.ndarray,
                            Sz: np.ndarray) -> np.ndarray:
    """Compute one-period Floquet unitary U_F, mirroring calculate_unitary_T."""
    steps = params["steps_floquet_unitary"]
    T = params["T"]
    nu = params["nu"]
    phi = params["phi"]
    varphi = params["varphi"]
    theta = params["theta"]
    phi_0 = params["phi_0"]

    t_delta = T / steps
    omega = _resolve_omega(params)

    K = _kick_operator(phi, Sx)
    U = np.eye(H0.shape[0], dtype=complex)

    for p in range(1, nu + 1):
        U = _evolution_T_step(U, h, T, varphi, theta, phi_0, H0, Sx, Sy, Sz,
                              omega, p, t_delta, steps)
        U = K @ U
    return U


def _find_power_r(evals: np.ndarray, evecs: np.ndarray,
                  evecs_inv: np.ndarray, r: int) -> np.ndarray:
    """Compute U^r = evecs @ diag(evals^r) @ evecs_inv."""
    if r <= 0:
        return np.eye(len(evals), dtype=complex)
    return evecs @ np.diag(evals ** r) @ evecs_inv


def _unitary_at_time(h: float, time: int, params: dict,
                     H0: np.ndarray, Sx: np.ndarray, Sy: np.ndarray, Sz: np.ndarray,
                     uf_evals: np.ndarray, uf_evecs: np.ndarray,
                     uf_evecs_inv: np.ndarray) -> np.ndarray:
    """Compute U(time), mirroring calculate_unitary_at_time_mp."""
    nu = params["nu"]
    T = params["T"]
    steps = params["steps_floquet_unitary"]
    phi = params["phi"]
    varphi = params["varphi"]
    theta = params["theta"]
    phi_0 = params["phi_0"]

    t_delta = T / steps
    omega = _resolve_omega(params)

    r = time // nu
    U = _find_power_r(uf_evals, uf_evecs, uf_evecs_inv, r)
    K = _kick_operator(phi, Sx)

    for p in range(r * nu + 1, time + 1):
        U = _evolution_T_step(U, h, T, varphi, theta, phi_0, H0, Sx, Sy, Sz,
                               omega, p, t_delta, steps)
        U = K @ U
    return U


def _qfi_raw_from_kets(ket: np.ndarray, ket_p: np.ndarray, ket_m: np.ndarray,
                       epsilon: float) -> float:
    """Raw QFI = 4*(<dket|dket> - |<ket|dket>|^2) from the three evolved kets."""
    dket = (ket_p - ket_m) / (2.0 * epsilon)
    term1 = float(np.real(dket.conj() @ dket))
    term2 = abs(ket.conj() @ dket) ** 2
    return 4.0 * (term1 - term2)


def _qfi_information_from_kets(ket: np.ndarray, ket_p: np.ndarray, ket_m: np.ndarray,
                               epsilon: float, N: int, time: int,
                               Sx: np.ndarray, Sy: np.ndarray, Sz: np.ndarray,
                               time_power: int = 2) -> QFIInformation:
    """Build a QFIInformation record from the three evolved kets.

    Mirrors lmg_qfi.qfi.qfi_information_from_kets: time_power = 2 for amplitude
    estimation, 4 for frequency estimation.
    """
    qfi = _qfi_raw_from_kets(ket, ket_p, ket_m, epsilon)

    m_x = float(np.real(ket.conj() @ (Sx @ ket))) / N
    m_y = float(np.real(ket.conj() @ (Sy @ ket))) / N
    m_z = float(np.real(ket.conj() @ (Sz @ ket))) / N

    return QFIInformation(
        qfi_raw_value=str(qfi),
        qfi=float(qfi / (N ** 2 * time ** time_power)),
        time=time,
        m_x=m_x,
        m_y=m_y,
        m_z=m_z,
    )


def _frequency_epsilon(base_epsilon: float, h: float, N: int, t_max: int) -> float:
    """Finite-difference step for the frequency derivative.

    Mirrors lmg_qfi.simulation.frequency_estimation_epsilon: the response to a
    frequency shift grows like epsilon*h*N*t, so the step is capped at
    0.1/(|h|*N*t_max) to keep the central difference in its linear regime.
    """
    return min(float(base_epsilon), 0.1 / (abs(h) * N * t_max))


def _run_quspin_frequency(pf: dict, time_interval: list, psi0: np.ndarray,
                          H0: np.ndarray, Sx: np.ndarray, Sy: np.ndarray,
                          Sz: np.ndarray) -> List[QFIInformation]:
    """Frequency-estimation branch: phase-continuous sequential evolution.

    At omega +/- epsilon the drive sin(omega*t + phi_0) is not commensurate
    with the kick period, so the Floquet-power shortcut does not apply; the
    three trajectories are advanced kick period by kick period with the drive
    phase running over absolute time (cost linear in the largest time).
    """
    N = pf["N"]
    h = pf["h"]
    if h == 0:
        raise ValueError(
            "Frequency estimation requires a nonzero AC amplitude h "
            "(the drive carries the omega dependence); e.g. h = 0.1.")

    grid = sorted({int(t) for t in time_interval if int(t) > 0})
    if not grid:
        return []
    t_max = grid[-1]

    epsilon = _frequency_epsilon(pf["epsilon"], h, N, t_max)
    if epsilon < float(pf["epsilon"]):
        logging.info(f"Frequency-mode epsilon reduced to {epsilon:.3e} "
                     f"(finite-difference bound 0.1/(h*N*t_max))")

    T = pf["T"]
    steps = pf["steps_floquet_unitary"]
    t_delta = T / steps
    omega_0 = _resolve_omega(pf)
    omegas = (omega_0, omega_0 + epsilon, omega_0 - epsilon)
    K = _kick_operator(pf["phi"], Sx)
    kets = [psi0.copy() for _ in omegas]

    logging.info(f"Frequency estimation (QuSpin): omega_0={omega_0:.6f}, "
                 f"epsilon={epsilon:.3e}, t_max={t_max}, {len(grid)} time points")

    grid_set = set(grid)
    log_every = max(1, t_max // 100)
    results = []
    for p in range(1, t_max + 1):
        for i, omega in enumerate(omegas):
            kets[i] = K @ _evolution_T_step(kets[i], h, T, pf["varphi"], pf["theta"],
                                            pf["phi_0"], H0, Sx, Sy, Sz,
                                            omega, p, t_delta, steps)
        if p in grid_set:
            results.append(_qfi_information_from_kets(
                kets[0], kets[1], kets[2], epsilon, N, p, Sx, Sy, Sz, time_power=4))
        if p % log_every == 0:
            logging.info(f"{p / t_max * 100.0:.2f}%: period {p}/{t_max}")
    return results


def run_quspin_simulation(params_dict: dict, time_interval: list,
                          init_state: np.ndarray) -> List[QFIInformation]:
    """Run QFI dynamics simulation using QuSpin operators + numpy/scipy evolution.

    Mirrors simulation_with_AC_field_mp but uses double-precision numpy throughout.

    Parameters
    ----------
    params_dict : dict
        Parameters dict with keys: N, J, B, T, nu, phi, h, epsilon,
        steps_floquet_unitary, varphi, theta, phi_0. Optional keys:
        "parameter" ("amplitude" | "frequency" or an EstimationParameter,
        default "amplitude") selects the estimated parameter; "omega" sets the
        probe frequency (default: resonant 2*pi/(nu*T)). Frequency mode
        requires h != 0.
    time_interval : list of int
        Floquet period indices at which to evaluate QFI.
    init_state : np.ndarray, shape (N+1,)
        Initial state vector (will be normalised if not already).

    Returns
    -------
    list of QFIInformation
    """
    N = params_dict["N"]
    h = float(params_dict["h"])
    epsilon = float(params_dict["epsilon"])

    Sz, Sx, Sy = get_spin_operators(N)
    H0 = build_h0(float(params_dict["J"]), float(params_dict["B"]), N, Sz, Sx)

    pf = dict(params_dict)   # local copy with float values
    pf["h"] = h
    pf["J"] = float(params_dict["J"])
    pf["B"] = float(params_dict["B"])
    pf["T"] = float(params_dict["T"])
    pf["phi"] = float(params_dict["phi"])
    pf["phi_0"] = float(params_dict["phi_0"])
    pf["varphi"] = float(params_dict["varphi"])
    pf["theta"] = float(params_dict["theta"])
    pf["epsilon"] = epsilon

    psi0 = np.asarray(init_state, dtype=complex)
    psi0 = psi0 / np.linalg.norm(psi0)

    parameter = params_dict.get("parameter", "amplitude")
    parameter = getattr(parameter, "value", parameter)  # EstimationParameter or str
    if parameter == "frequency":
        return _run_quspin_frequency(pf, time_interval, psi0, H0, Sx, Sy, Sz)

    def _eig_uf(h_val: float):
        pf_h = dict(pf)
        pf_h["h"] = h_val
        UF = compute_floquet_unitary(h_val, pf_h, H0, Sx, Sy, Sz)
        evals, evecs = np.linalg.eig(UF)
        return evals, evecs, np.linalg.inv(evecs)

    uf_evals,   uf_evecs,   uf_einv   = _eig_uf(h)
    uf_p_evals, uf_p_evecs, uf_p_einv = _eig_uf(h + epsilon)
    uf_m_evals, uf_m_evecs, uf_m_einv = _eig_uf(h - epsilon)

    results = []
    for time in time_interval:
        U   = _unitary_at_time(h,           time, pf, H0, Sx, Sy, Sz,
                                uf_evals,   uf_evecs,   uf_einv)
        U_p = _unitary_at_time(h + epsilon, time, pf, H0, Sx, Sy, Sz,
                                uf_p_evals, uf_p_evecs, uf_p_einv)
        U_m = _unitary_at_time(h - epsilon, time, pf, H0, Sx, Sy, Sz,
                                uf_m_evals, uf_m_evecs, uf_m_einv)

        ket   = U   @ psi0
        ket_p = U_p @ psi0
        ket_m = U_m @ psi0

        results.append(_qfi_information_from_kets(
            ket, ket_p, ket_m, epsilon, N, time, Sx, Sy, Sz, time_power=2))
    return results


def _gap_time_degree(J: float, B: float, N: int) -> int:
    """Max time exponent: 10^degree covers 4*pi/gap of the H0 spectrum."""
    Sz, Sx, _ = get_spin_operators(N)
    evals = np.linalg.eigvalsh(build_h0(J, B, N, Sz, Sx))
    return int(np.log10(abs(4.0 * np.pi / (evals[1] - evals[0])))) + 1


def run_quspin(params: SimulationParams,
               init_states: List[InitialState]) -> Dict[InitialState, List[QFIInformation]]:
    """Run the full QFI simulation with the QuSpin solver.

    Mirrors lmg_qfi.simulation.run_simulation: reads dps / num_points /
    steps_floquet_unitary / max_time_degree from params.run_arguments and
    returns a dict mapping InitialState to a list of QFIInformation results.
    """
    N, J, B = params.N, float(params.J), float(params.B)
    dps = params.run_arguments["dps"]
    num_points = params.run_arguments["num_points"]

    frequency_mode = params.parameter == EstimationParameter.FREQUENCY
    if frequency_mode and float(params.h) == 0.0:
        raise ValueError(
            "Frequency estimation requires a nonzero AC amplitude h "
            "(the drive carries the omega dependence); set h in parameters.ini "
            "or pass --amplitude, e.g. 0.1.")

    qs_params = dict(
        N=N, J=J, B=B,
        T=float(params.T),
        nu=int(params.freq),
        phi=float(params.phi_kick_phase),
        h=float(params.h),
        epsilon=10.0 ** -(dps // 2),
        steps_floquet_unitary=params.run_arguments["steps_floquet_unitary"],
        varphi=float(params.varphi),
        theta=float(params.theta),
        phi_0=float(params.phi_0),
        parameter=params.parameter.value,
        omega=None if params.omega is None else float(params.omega))

    last_time_degree = _gap_time_degree(J, B, N)
    max_time_degree = params.run_arguments.get("max_time_degree")
    if max_time_degree is None and frequency_mode:
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
    time_interval = generate_time_interval(num_points, last_time_degree)

    vectors = build_initial_states(N, J, B)
    results = {}
    for state in init_states:
        logging.info(f"Running {state.value} (quspin solver) ...")
        results[state] = run_quspin_simulation(qs_params, time_interval, vectors[state])
    return results