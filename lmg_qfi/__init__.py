"""
LMG QFI Dynamics - Quantum Fisher Information simulation for the LMG model.

This package provides tools for simulating and analyzing Quantum Fisher Information
dynamics in the Lipkin-Meshkov-Glick (LMG) model.
"""

import mpmath as mp

# Set default precision
mp.dps = 100

# Import main classes and functions for convenient access
from .config import (
    InitialState,
    SimulationParams,
    QFIInformation,
    UF,
    read_defaults_args_from_config,
    parse_arguments,
)

from .operators import (
    create_z_operator,
    create_spin_minus_operators,
    create_spin_plus_operators,
    create_spin_xyz_operators,
    create_hamiltonian_h0,
    create_kick_operator,
    ac_time,
    create_v_operator,
)

from .evolution import (
    evalution_T_step,
    find_power_r_mpmath,
    calculate_unitary_at_time_mp,
    calculate_unitary_T,
)

from .qfi import (
    dketa_t,
    quantum_fisher_information_mp,
    calculate_error_estimation_mp,
    process_time_point_mp,
)

from .simulation import (
    generate_time_interval,
    simulation_with_AC_field_mp,
    run_simulation,
    run_gaps,
)

from .io import save_to_file_qfi_dynamics

from .plotting import (
    plot_qfi_data_subplot,
    plot,
    MAX_TIME_POW_PLOT,
    Y_LABEL_COORDINATE,
)

from .utils import (
    log_message,
    setup_logging,
)

__all__ = [
    # Config
    "InitialState",
    "SimulationParams",
    "QFIInformation",
    "UF",
    "read_defaults_args_from_config",
    "parse_arguments",
    # Operators
    "create_z_operator",
    "create_spin_minus_operators",
    "create_spin_plus_operators",
    "create_spin_xyz_operators",
    "create_hamiltonian_h0",
    "create_kick_operator",
    "ac_time",
    "create_v_operator",
    # Evolution
    "evalution_T_step",
    "find_power_r_mpmath",
    "calculate_unitary_at_time_mp",
    "calculate_unitary_T",
    # QFI
    "dketa_t",
    "quantum_fisher_information_mp",
    "calculate_error_estimation_mp",
    "process_time_point_mp",
    # Simulation
    "generate_time_interval",
    "simulation_with_AC_field_mp",
    "run_simulation",
    "run_gaps",
    # I/O
    "save_to_file_qfi_dynamics",
    # Plotting
    "plot_qfi_data_subplot",
    "plot",
    "MAX_TIME_POW_PLOT",
    "Y_LABEL_COORDINATE",
    # Utils
    "log_message",
    "setup_logging",
]
