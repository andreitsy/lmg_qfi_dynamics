"""Visualization functions for QFI dynamics."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

from pathlib import Path
from scipy.ndimage import gaussian_filter

from .config import SimulationParams
from .operators import create_hamiltonian_h0

# Matplotlib configuration
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Computer Modern"]
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,amsfonts,amssymb}"

MAX_TIME_POW_PLOT = None
Y_LABEL_COORDINATE = 0.155


def compute_gap_times(simulation_params):
    """Return [(pair_idx, log10(2 pi / Delta))] for the low-lying cat doublets of H0."""
    # The doublet splittings fall off exponentially with N (about 1e-28 for
    # N = 100), far below double precision, so the spectrum is computed with
    # 100 digits. (`mp.dps = 100` in lmg_qfi/__init__.py sets an attribute of
    # the mpmath module, not mpmath's working precision.)
    gap_times = []
    with mp.workdps(100):
        H = create_hamiltonian_h0(simulation_params.J, simulation_params.B, simulation_params.N)
        symmetry_edge = mp.mpf(-1 * simulation_params.B * simulation_params.N)
        eigvals_H_real, evecs = mp.eigh(H)

        pairs = {}
        for idx, eigenvalue_mp in enumerate(eigvals_H_real):
            pair_num = idx // 2
            if pair_num not in pairs:
                pairs[pair_num] = []
            pairs[pair_num].append(eigenvalue_mp)

        for pair_idx, energies in pairs.items():
            if energies[0] <= symmetry_edge and pair_idx < 7 and len(energies) == 2:
                gap_times.append(
                    (pair_idx, float(mp.log10(2 * mp.pi / mp.fabs(energies[1] - energies[0])))))

    return gap_times


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
        
        # Labels for each state
        if state == "GS_PHYS":
            cor = "tab:red"
            label = (r"$\left(\big|E_1 {\bigr \rangle} + "
                     r"\big|E_{\overline{1}} {\bigr \rangle} \right) / \sqrt{2}$")
        elif state == "GS_CAT":
            cor = "tab:green"
            label = r"$\big|E_1 {\bigr \rangle} $"
        elif state == "PHYS":
            cor = "tab:orange"
            label = r"$\big|\uparrow ... \uparrow {\bigr \rangle}$"
        elif state == "CAT_SUM":
            cor = "tab:blue"
            label = (r"$\left(\big|\uparrow ... \uparrow {\bigr \rangle} + "
                     r"\big|\downarrow ... \downarrow {\bigr \rangle} "
                     r"\right)/\sqrt{2}$")
        else:
            cor = "b"
            label = state
        
        ax.plot(time_points, qfi_values, "-", label=label, linewidth=3, color=cor)
    
    # Customize QFI subplot
    ax.set_title(
        rf"QFI dynamics for $N={simulation_params.N}, B/J={float(simulation_params.B / simulation_params.J):.2f}$",
        fontsize=40)
    ax.set_xlabel(r"$t / T$", fontsize=40)
    ax.set_ylabel(r"$F_h / (N t)^2$", fontsize=40)
    
    ax.legend(
        title="Initial State", loc=(0.35, 0.567), fontsize=28, title_fontsize=26
    ).set_zorder(10)
    ax.set_ylim([0, np.abs((1 - float(simulation_params.B) ** 2) * 4 / np.pi ** 2)])
    
    if max_time_pow is None:
        max_time_pow = int(last_time) + 1
    x_ticks_pow = [n for n in range(1, max_time_pow, 2)]
    ax.set_xticks(x_ticks_pow, [rf"$10^{{{i}}}$" for i in x_ticks_pow])
    ax.tick_params(axis="x", labelsize=30)
    ax.tick_params(axis="y", labelsize=30)
    ax.grid(True, linestyle="--", alpha=0.6, linewidth=1.7)
    
    gap_times = compute_gap_times(simulation_params)

    # Add vertical lines for gap points
    for pair_idx, time_gap in gap_times:
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
    plt.savefig(
        results_dir / f"qfi_dynamics_N={simulation_params.N}_B={float(simulation_params.B):.2f}.png", dpi=300)


STATE_TITLES = {
    "GS_PHYS": (r"$\left(\big|E_1 {\bigr \rangle} + "
                r"\big|E_{\overline{1}} {\bigr \rangle} \right) / \sqrt{2}$"),
    "GS_CAT": r"$\big|E_1 {\bigr \rangle}$",
    "PHYS": r"$\big|\uparrow ... \uparrow {\bigr \rangle}$",
    "CAT_SUM": (r"$\left(\big|\uparrow ... \uparrow {\bigr \rangle} + "
                r"\big|\downarrow ... \downarrow {\bigr \rangle} \right)/\sqrt{2}$"),
}

# (CSV column, legend label, color, line style). Colors are the first three slots of a
# CVD-validated categorical palette; line styles are a secondary encoding.
MOM_SERIES = [
    ("mom_sz", r"MoM, $\hat S_z$", "#2a78d6", "--"),
    ("mom_x", r"MoM, $\hat X$", "#eb6834", "-."),
    ("mom_y", r"MoM, $\hat Y = i \hat S_z \hat X$", "#1baf7a", ":"),
]


def plot_qfi_mom(simulations: dict, simulation_params: SimulationParams, results_dir: Path):
    """
    Plot QFI and the method-of-moments sensitivity of S_z, X and Y, one panel per initial state.

    Only states whose results contain the ``mom_*`` columns are drawn. All curves share the
    QFI normalization 1 / (N t)^2; MoM <= QFI. Saved as ``qfi_mom_dynamics_N=.._B=...png``.
    """
    states = [s for s in simulations if all(c in simulations[s] for c, *_ in MOM_SERIES)]
    if not states:
        return None
    order = [s for s in STATE_TITLES if s in states] + [s for s in states if s not in STATE_TITLES]
    ncols = 2 if len(order) > 1 else 1
    nrows = (len(order) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11 * ncols, 7.5 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    gap_times = compute_gap_times(simulation_params)

    last_time = 0.0
    for ax, state in zip(axes.flat, order):
        df = simulations[state]
        log_t = np.log10(df["time"].to_numpy(dtype=float))
        last_time = max(last_time, log_t[-1])
        # Wide light band so a MoM curve that saturates the QFI stays visible on top of it.
        ax.plot(log_t, df["qfi"].to_numpy(dtype=float), "-", color="#9a9a9a",
                linewidth=7, alpha=0.6, solid_capstyle="butt", label=r"QFI $F_h$")
        for col, label, color, style in MOM_SERIES:
            ax.plot(log_t, df[col].to_numpy(dtype=float), style, color=color,
                    linewidth=2.5, label=label)
        for _, time_gap in gap_times:
            ax.axvline(x=time_gap, color="#888888", linestyle="--", alpha=0.6, linewidth=1.2)
        ax.set_title(STATE_TITLES.get(state, state), fontsize=26)
        ax.grid(True, linestyle="--", alpha=0.4, linewidth=1)
        ax.tick_params(labelsize=20)
    for ax in axes.flat[len(order):]:
        ax.set_visible(False)

    x_ticks_pow = list(range(0, int(np.ceil(last_time - 1e-9)) + 1))
    for ax in axes.flat[:len(order)]:
        ax.set_xticks(x_ticks_pow, [rf"$10^{{{i}}}$" for i in x_ticks_pow])
        ax.set_ylim(bottom=0)
    for ax in axes[-1]:
        ax.set_xlabel(r"$t / T$", fontsize=26)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$F_h / (N t)^2$, $\ \mathrm{MoM} / (N t)^2$", fontsize=24)
    axes.flat[0].legend(fontsize=20, loc="upper left")
    fig.suptitle(rf"QFI and method of moments, $N={simulation_params.N}$, "
                 rf"$B/J={float(simulation_params.B / simulation_params.J):.2f}$", fontsize=30)
    fig.tight_layout()
    out = results_dir / f"qfi_mom_dynamics_N={simulation_params.N}_B={float(simulation_params.B):.2f}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
