"""Visualization functions for QFI dynamics."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

from pathlib import Path
from scipy.ndimage import gaussian_filter

from .config import EstimationParameter, InitialState, SimulationParams
from .operators import create_hamiltonian_h0

# Matplotlib configuration
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Computer Modern"]
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,amsfonts,amssymb}"

MAX_TIME_POW_PLOT = None
Y_LABEL_COORDINATE = 0.155

# Plot color and LaTeX label per initial-state key (InitialState.name).
STATE_PLOT_STYLES = {
    "GS_PHYS": ("tab:red",
                r"$\left(\big|E_1 {\bigr \rangle} + "
                r"\big|E_{\overline{1}} {\bigr \rangle} \right) / \sqrt{2}$"),
    "GS_CAT": ("tab:green", r"$\big|E_1 {\bigr \rangle} $"),
    "PHYS": ("tab:orange", r"$\big|\uparrow ... \uparrow {\bigr \rangle}$"),
    "CAT_SUM": ("tab:blue",
                r"$\left(\big|\uparrow ... \uparrow {\bigr \rangle} + "
                r"\big|\downarrow ... \downarrow {\bigr \rangle} "
                r"\right)/\sqrt{2}$"),
}


# Accept both key forms: InitialState.name ("GS_PHYS") and InitialState.value
# ("GS_phys", used in the result file names).
_STATE_KEY_ALIASES = {state.value: state.name for state in InitialState}


def _state_style(state_key: str):
    """Return (color, label) for an initial-state key, with a plain fallback."""
    key = _STATE_KEY_ALIASES.get(state_key, state_key)
    return STATE_PLOT_STYLES.get(key, ("b", state_key))


def plot_qfi_data_subplot(ax, simulations, simulation_params, max_time_pow=None):
    """
    Plot QFI values onto the provided subplot axes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    simulations : dict
        Dictionary mapping state names to DataFrames with simulation results.
    simulation_params : SimulationParams
        Simulation parameters.
    max_time_pow : int, optional
        Maximum power of 10 for x-axis.
    """
    last_time = -1
    for state in simulations:
        time_points = [float(mp.log10(mp.mpf(x))) for x in simulations[state]["time"].tolist()]
        last_time = max(time_points[-1], last_time)
        qfi_values = [float(x) for x in simulations[state]["qfi"].tolist()]
        qfi_values[:1000] = gaussian_filter(qfi_values[:1000], sigma=1.5)
        
        cor, label = _state_style(state)
        ax.plot(time_points, qfi_values, "-", label=label, linewidth=3, color=cor)
    
    frequency_mode = (getattr(simulation_params, "parameter", EstimationParameter.AMPLITUDE)
                      == EstimationParameter.FREQUENCY)

    # Customize QFI subplot
    title = (rf"QFI dynamics for $N={simulation_params.N}$, "
             rf"$B/J={float(simulation_params.B / simulation_params.J):.2f}$")
    if frequency_mode:
        omega_resonant = 2.0 * np.pi / (int(simulation_params.freq)
                                        * float(simulation_params.T))
        omega_tag = (rf"\omega_0 \approx {omega_resonant:.4f}"
                     if simulation_params.omega is None
                     else rf"{float(simulation_params.omega):.4f}")
        title += (rf", $h={float(simulation_params.h):g}$, $\omega={omega_tag}$")
    ax.set_title(title, fontsize=40 if not frequency_mode else 34)
    ax.set_xlabel(r"$t / T$", fontsize=40)
    ax.set_ylabel(r"$F_\omega / (N^2 t^4)$" if frequency_mode else r"$F_h / (N t)^2$",
                  fontsize=40)

    ax.legend(
        title="Initial State", loc=(0.35, 0.567), fontsize=28, title_fontsize=26
    ).set_zorder(10)
    if frequency_mode:
        # F_omega/(N t^2)^2 spans many decades over the time window.
        ax.set_yscale("log")
    else:
        # Analytic long-time plateau bound applies to amplitude estimation only.
        ax.set_ylim([0, np.abs((1 - float(simulation_params.B) ** 2) * 4 / np.pi ** 2)])
    
    if max_time_pow is None:
        max_time_pow = int(last_time) + 1
    x_ticks_pow = [n for n in range(1, max_time_pow, 2)]
    ax.set_xticks(x_ticks_pow, [rf"$10^{{{i}}}$" for i in x_ticks_pow])
    ax.tick_params(axis="x", labelsize=30)
    ax.tick_params(axis="y", labelsize=30)
    ax.grid(True, linestyle="--", alpha=0.6, linewidth=1.7)
    
    H = create_hamiltonian_h0(simulation_params.J, simulation_params.B, simulation_params.N)
    symmetry_edge = mp.mpf(-1 * simulation_params.B * simulation_params.N)
    eigvals_H_real, evecs = mp.eigh(H)
    
    # Add vertical lines for gap points
    pairs = {}
    for idx, eigenvalue_mp in enumerate(eigvals_H_real):
        pair_num = idx // 2
        if pair_num not in pairs:
            pairs[pair_num] = []
        pairs[pair_num].append(eigenvalue_mp)
    
    for pair_idx, energies in pairs.items():
        if energies[0] <= symmetry_edge and pair_idx < 7:
            if len(energies) == 2:
                time_gap = float(mp.log10(2 * mp.pi / mp.fabs(energies[1] - energies[0])))
                ax.axvline(x=time_gap, color='black',
                           linestyle='--', alpha=0.7, linewidth=2)
                ax.annotate(
                    f"$\\frac{{2\\pi}}{{\\Delta_{{{pair_idx + 1}\\overline{{{pair_idx + 1}}}}}}}$",
                    xy=(time_gap, Y_LABEL_COORDINATE),
                    xytext=(time_gap * 1.03, Y_LABEL_COORDINATE),
                    fontsize=54,
                    color='black'
                )


def plot(simulations: dict, simulation_params: SimulationParams, results_dir: Path):
    """
    Create and save the QFI dynamics plot.
    
    Parameters
    ----------
    simulations : dict
        Dictionary mapping state names to DataFrames with simulation results.
    simulation_params : SimulationParams
        Simulation parameters.
    results_dir : Path
        Directory to save the plot.
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    plot_qfi_data_subplot(ax, simulations, simulation_params, max_time_pow=MAX_TIME_POW_PLOT)
    plt.tight_layout()
    # Frequency-mode figures get their own name so they never overwrite the
    # amplitude-mode figure for the same N and B; the solver tag keeps the
    # quspin and mpmath figures apart the same way.
    frequency_mode = (getattr(simulation_params, "parameter", EstimationParameter.AMPLITUDE)
                      == EstimationParameter.FREQUENCY)
    mode_tag = "_Fomega" if frequency_mode else ""
    solver = getattr(simulation_params, "solver", None)
    solver_tag = f"_{solver.value}" if solver is not None else ""
    plt.savefig(
        results_dir / (f"qfi_dynamics{mode_tag}_N={simulation_params.N}"
                       f"_B={float(simulation_params.B):.2f}{solver_tag}.png"), dpi=300)


