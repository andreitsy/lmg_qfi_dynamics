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
    "PHYS": r"$\big|\uparrow \cdots \uparrow {\bigr \rangle}$",
    "CAT_SUM": (r"$\left(\big|\uparrow \cdots \uparrow {\bigr \rangle} + "
                r"\big|\downarrow \cdots \downarrow {\bigr \rangle} \right)/\sqrt{2}$"),
}

# (CSV column, legend label, color, line style). Use the existing QFI figures'
# blue/orange/green palette, with dash patterns to distinguish overlapping curves.
MOM_SERIES = [
    ("mom_sz", r"$\mathcal{M}_h[\hat S_z]$", "tab:blue", "--"),
    ("mom_x", r"$\mathcal{M}_h[\hat X]$", "tab:orange", "-."),
    ("mom_y", r"$\mathcal{M}_h[\hat Y]$, $\hat Y = i\hat S_z\hat X$", "tab:green", ":"),
]


def plot_qfi_mom(
    simulations: dict,
    simulation_params: SimulationParams,
    results_dir: Path,
    *,
    width: float = 18.0,
    output: Path | None = None,
):
    """
    Plot QFI and the method-of-moments sensitivity of S_z, X and Y, one panel per initial state.

    Only states whose results contain the ``mom_*`` columns are drawn. All curves share the
    QFI normalization 1 / (N t)^2; MoM <= QFI. Plot the stored samples without
    smoothing. Save a 300 dpi PNG and a companion vector PDF by default;
    ``output`` can override the path (including a PDF-only export).
    """
    states = [s for s in simulations if all(c in simulations[s] for c, *_ in MOM_SERIES)]
    if not states:
        return None
    order = [s for s in STATE_TITLES if s in states] + [s for s in states if s not in STATE_TITLES]
    ncols = 2 if len(order) > 1 else 1
    nrows = (len(order) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, width * (0.35 * nrows + 0.13)),
                             sharex=True, sharey=True, squeeze=False)
    gap_times = compute_gap_times(simulation_params)

    first_time = min(np.log10(simulations[s]["time"].to_numpy(dtype=float)).min() for s in order)
    last_time = max(np.log10(simulations[s]["time"].to_numpy(dtype=float)).max() for s in order)
    max_value = max(simulations[s][["qfi", *(c for c, *_ in MOM_SERIES)]].to_numpy(dtype=float).max()
                    for s in order)
    # If the first state has a high plateau (e.g. a GS_CAT-only plot), reserve
    # extra space above it for the gap labels rather than covering its curve.
    first_max = simulations[order[0]][["qfi", *(c for c, *_ in MOM_SERIES)]].to_numpy(dtype=float).max()
    headroom = 1.30 if first_max > 0.7 * max_value else 1.10
    # At most seven labelled decades: large-N runs span up to 10^18 and beyond.
    tick_step = max(1, int(np.ceil((last_time - first_time) / 6)))
    x_ticks_pow = np.arange(np.ceil(first_time), np.floor(last_time) + 1, tick_step)

    for panel, (ax, state) in enumerate(zip(axes.flat, order)):
        df = simulations[state]
        log_t = np.log10(df["time"].to_numpy(dtype=float))
        # A solid neutral outline remains visible beneath a MoM curve at the bound.
        ax.plot(log_t, df["qfi"].to_numpy(dtype=float), "-", color="0.35",
                linewidth=5, solid_capstyle="round", label=r"QFI, $F_h$", zorder=3)
        for col, label, color, style in MOM_SERIES:
            ax.plot(log_t, df[col].to_numpy(dtype=float), style, color=color,
                    linewidth=3, label=label, zorder=4)
        for pair_idx, time_gap in gap_times:
            if not first_time <= time_gap <= last_time:
                continue
            ax.axvline(x=time_gap, color="0.25", linestyle="--", alpha=0.65,
                       linewidth=1.5, zorder=2)
            # Label each gap once, in the first panel, above its data. Keeping
            # these out of the other panels leaves the plateaus unobscured.
            if panel == 0:
                k = pair_idx + 1
                ax.annotate(
                    rf"$\frac{{2\pi}}{{\Delta_{{{k}\overline{{{k}}}}}}}$",
                    xy=(time_gap, 0.90), xycoords=("data", "axes fraction"),
                    xytext=(4, 0), textcoords="offset points", fontsize=24,
                    ha="left", va="top", color="0.15",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1, "alpha": 0.85},
                    zorder=5,
                )
        ax.set_title(STATE_TITLES.get(state, state), fontsize=30, pad=14)
        ax.text(0.02, 1.03, rf"\textbf{{({chr(97 + panel)})}}", transform=ax.transAxes,
                fontsize=23, va="bottom")
        ax.set_axisbelow(True)
        ax.grid(True, linestyle="--", alpha=0.35, linewidth=1)
        ax.tick_params(labelsize=24, length=6, pad=7)
        ax.set_xticks(x_ticks_pow, [rf"$10^{{{int(i)}}}$" for i in x_ticks_pow])
        ax.set_ylim(0, max_value * headroom if max_value > 0 else 1)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, min_n_ticks=3))
        ax.margins(x=0.025)
        # With three states, the top-right panel also needs time tick labels.
        if panel + ncols >= len(order):
            ax.tick_params(axis="x", labelbottom=True)
    for ax in axes.flat[len(order):]:
        ax.set_visible(False)

    fig.supxlabel(r"$t / T$", fontsize=34, y=0.08 if nrows == 1 else 0.055)
    fig.supylabel(r"$F_h / (N t)^2,\quad \mathcal{M}_h / (N t)^2$", fontsize=32, x=0.015)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=24, loc="upper center", bbox_to_anchor=(0.53, 0.94),
               ncol=4 if ncols == 2 else 2, frameon=False, handlelength=3, columnspacing=1.5)
    fig.suptitle(rf"QFI and method of moments: $N={simulation_params.N}$, "
                 rf"$B/J={float(simulation_params.B / simulation_params.J):.2f}$", fontsize=34, y=0.99)
    fig.text(0.53, 0.015,
             r"$\mathcal{M}_h[\hat O] = (\partial_h\langle\hat O\rangle)^2 / "
             r"\mathrm{Var}(\hat O)$", fontsize=22, ha="center")
    # A single row is shorter, so the header and footer need a larger fraction
    # of the canvas; the one-panel legend also has two rows.
    top = (0.65 if ncols == 1 else 0.72) if nrows == 1 else 0.82
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.22 if nrows == 1 else 0.14, top=top,
                        wspace=0.075, hspace=0.28)
    out = Path(output) if output is not None else results_dir / (
        f"qfi_mom_dynamics_N={simulation_params.N}_B={float(simulation_params.B):.2f}.png")
    fig.savefig(out, dpi=300)
    if out.suffix.lower() == ".png":
        fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out
