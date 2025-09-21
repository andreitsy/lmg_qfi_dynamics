#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from quantum_fisher_information_simulation import (
    read_defaults_args_from_config,
    create_hamiltonian_h0,
    find_eigen_values_mpmath,
    results_dir,
)
from scipy.ndimage import gaussian_filter
from pathlib import Path
import matplotlib as mpl

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Computer Modern"]
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,amsfonts,amssymb}"
MAX_TIME_POW = 15
Y_LABEL_COORDINATE = 0.3
results_dir = Path(__file__).parent / "results"


def read_csv_data(file_path):
    # Read the data using pandas
    df = pd.read_csv(results_dir / file_path)
    return df


def plot_energy_levels_subplot(ax, J, h, n):
    """Plot energy levels onto the provided subplot axes (ax) with annotations for energy gaps as vertical lines."""
    H = create_hamiltonian_h0(J, h, n, precision="mpmath")
    eigvals_H_real = find_eigen_values_mpmath(H)
    colors = ["b", "g"]  # Alternating colors for pairs

    # Dictionary to store the pairs and their energy values
    pairs = {}
    for idx, eigenvalue in enumerate(eigvals_H_real):
        ax.axvline(x=eigenvalue, color=colors[(idx // 2) % 2], linewidth=1)
        # Group eigenvalues into pairs
        pair_num = idx // 2
        if pair_num not in pairs:
            pairs[pair_num] = []
        pairs[pair_num].append(float(eigenvalue))

    # Add annotations with arrows for each pair, including the gap values
    for pair_idx, energies in pairs.items():
        if len(energies) == 2:
            # Calculate the gap between two energy levels in the pair
            gap = abs(energies[1] - energies[0])
            # Calculate the mean position of the pair for label placement
            mean_energy = energies[0] if energies[0] < energies[1] else energies[1]
            # Only add annotations below symmetry edge
            if float(mean_energy) < -n * h:
                mantissa, exponent = f"{float(gap):.1e}".split('e')
                formatted_gap = f"{float(mantissa):.1f} \\cdot 10^{{{int(exponent)}}}"
                ax.annotate(
                    f"$\\Delta_{{{pair_idx}}} = {formatted_gap}$",
                    xy=(mean_energy, 0.02),  # Point to mean energy
                    xytext=(mean_energy, 0.5),  # Placement of the text
                    horizontalalignment="left",
                    verticalalignment="top",
                    fontsize=15,
                )

    # Add the broken-symmetry edge line with a label
    symmetry_edge = -1 * h * n
    ax.axvline(x=symmetry_edge, color="r", linestyle="--", linewidth=2.5)
    ax.text(
        symmetry_edge + 0.15,
        0.45,
        r"$E^*$",
        color="r",
        fontsize=18,
        horizontalalignment="center",
        verticalalignment="top",
    )

    # Customize the axis
    ax.set_xlabel("Energy ($E_i$)", fontsize=18)
    ax.set_xlim([float(eigvals_H_real[0]) - 1, -1 * h * n + 0.5])
    ax.grid(False)
    ax.set_yticks([])
    ax.invert_xaxis()
    ax.tick_params(axis="x", labelsize=20)  # Adjust x-axis tick label font size


def plot_quasienergies(ax, J, h, N):
    H_0 = create_hamiltonian_h0(J, h, N)
    eigvals = find_eigen_values_mpmath(H_0)
    # Define the radius and center of the circle
    center = 0 + 0j  # Assuming the center is at the origin
    # Define points on the circle
    theta = np.linspace(0, 2 * np.pi, 1000)  # 1000 points for a smooth curve
    circle_points = center + np.cos(theta) + 1j * np.sin(theta)
    symmetry_edge = -1 * h * N
    # Store energy points and assign different colors for pairs
    energy_points = []
    annotations = []  # Store annotation data
    for pair_idx in range(0, 18, 2):
        e1 = float(eigvals[pair_idx, 0].real)
        e2 = float(eigvals[pair_idx + 1, 0].real)
        mean_energy = (e1 + e2) / 2  # Mean of the pair values
        point1 = np.exp(1j * e1)
        point2 = np.exp(1j * (e2 + np.pi))
        if mean_energy < symmetry_edge:
            energy_points.append({'point': point1, 'color': 'blue'})
            energy_points.append({'point': point2, 'color': 'green'})
            annotations.append({
                'pair_idx': pair_idx // 2,
                'mean_energy': mean_energy,
                'gap': abs(e1 - e2),
                'point1': point1,
                'point2': point2
            })
        else:
            break

    # Plot the circle
    ax.plot(circle_points.real, circle_points.imag, color='black',
            linestyle='-', alpha=0.6, linewidth=5, zorder=4)
    # Plot the center of the circle
    ax.scatter(center.real, center.imag, color='black', label='Center', zorder=5)
    # Plot energy points with different colors
    for point in energy_points:
        ax.scatter(point['point'].real, point['point'].imag, color=point['color'], zorder=5,
                   linewidths=8)

    # Add lines connecting paired points
    for annotation in annotations:
        point1 = annotation['point1']
        point2 = annotation['point2']
        ax.plot([point1.real, point2.real], [point1.imag, point2.imag],
                color='gray', linestyle='--', alpha=0.8, zorder=1, linewidth=3)

    pair_text_ofsets = {
        1: {"E1_x_coor": -0.13, 'E1_y_coor': 0.06, "E2_x_coor": 0.05, 'E2_y_coor': 0.15},
        2: {"E1_x_coor": 0.16, 'E1_y_coor': -0.06, "E2_x_coor": 0.04, 'E2_y_coor': 0.15},
        3: {"E1_x_coor": 0.05, 'E1_y_coor': 0.17, "E2_x_coor": -0.13, 'E2_y_coor': 0.07},
        4: {"E1_x_coor": -0.05, 'E1_y_coor': -0.17, "E2_x_coor": 0.08, 'E2_y_coor': 0.17},
        5: {"E1_x_coor": 0.14, 'E1_y_coor': -0.1, "E2_x_coor": -0.2, 'E2_y_coor': 0.1},
        6: {"E1_x_coor": -0.08, 'E1_y_coor': 0.1, "E2_x_coor": -0.09, 'E2_y_coor': 0.15},
        7: {"E1_x_coor": 0.1, 'E1_y_coor': 0.1, "E2_x_coor": 0.1, 'E2_y_coor': 0.1},
    }

    for annotation in annotations:
        if annotation['mean_energy'] >= symmetry_edge or pair_idx > 7:
            break
        pair_idx = annotation['pair_idx']
        gap = annotation['gap']
        point1 = annotation['point1']
        point2 = annotation['point2']
        # Calculate offset for text placement (slightly outside the circle)
        point_loc = 0.2
        midpoint_x = point1.real + point_loc * point2.real
        midpoint_y = point1.imag + point_loc * point2.imag
        offset_x = 0.0
        offset_y = 1.1
        mantissa, exponent = f"{float(gap):.1e}".split('e')
        formatted_gap = f"{float(mantissa):.1f} \\cdot 10^{{{int(exponent)}}}"
        E1_x_coor = point1.real + pair_text_ofsets.get(pair_idx + 1, offset_x)['E1_x_coor']
        E1_y_coor = point1.imag + pair_text_ofsets.get(pair_idx + 1, offset_x)['E1_y_coor']
        E2_x_coor = point2.real + pair_text_ofsets.get(pair_idx + 1, offset_x)['E2_x_coor']
        E2_y_coor = point2.imag + pair_text_ofsets.get(pair_idx + 1, offset_x)['E2_y_coor']
        # ax.annotate(
        #     f"$\\Delta_{{{pair_idx}}} = {formatted_gap}$",
        #     xy=(midpoint_x, midpoint_y),
        #     xytext=(point1.real + offset_x, point1.imag * offset_y),
        #     horizontalalignment="center",
        #     verticalalignment="center",
        #     zorder=3,  # Ensure it's above all lines
        #     fontsize=35,
        #     arrowprops=dict(facecolor='black',
        #                     shrink=0.03, width=0.3, headwidth=3)
        # )
        # Store middle point between the pair for annotation
        ax.annotate(
            f"$\\varepsilon_{{{pair_idx + 1}}}$",
            xy=(E1_x_coor, E1_y_coor),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=52,
            arrowprops=dict(facecolor='black',
                            shrink=0.03, width=0.2, headwidth=3)
        )
        ax.annotate(
            f"$\\varepsilon_{{\\overline{{{pair_idx + 1}}}}}$",
            xy=(E2_x_coor, E2_y_coor),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=52,
            arrowprops=dict(facecolor='black',
                            shrink=0.08, width=0.9, headwidth=3)
        )
    # Set up the axes
    ax.axhline(0, color='gray', linewidth=1.5)
    ax.axvline(0, color='gray', linewidth=1.5)
    ax.set_aspect('equal', adjustable='box')
    # Remove axis labels
    ax.set_xlabel(r"$\mathrm{Re}\,(e^{i \varepsilon_i})$", fontsize=50)
    ax.set_ylabel(r"$\mathrm{Im}\,(e^{i \varepsilon_i})$", fontsize=50)
    ax.set_xticks([])  # Remove the x-axis numeric labels
    ax.set_yticks([])  # Remove the y-axis numeric labels
    ax.set_xlim([-1.12, 1.12])
    ax.set_ylim([-1.12, 1.12])
    ax.text(-1.34, 1.15, r"(b)", fontsize=50, fontweight="bold",
            va="top", ha="left")
    # Add labels and legend

def plot_qfi_data_subplot(ax, df, J, h, n, tau, max_time_pow=13):
    """Plot QFI values onto the provided subplot axes (ax)."""
    # Loop through different initial state groups in QFI data
    for state in [
        "CAT_SUM",
        "PHYS",
        "GS_CAT",
        "GS_PHYS",
    ]:
        filtered_data = df[df["initial_state"] == state]
        time_points = filtered_data["time_points"].values
        indices = np.where(time_points < 10 ** max_time_pow)[0]
        time_points = time_points[indices]
        qfi_values = filtered_data["qfi_values"].values[indices]
        if "GS" in state:
            #qfi_values[:10] = np.mean(qfi_values[0:100])
            y_smooth = qfi_values
        else:
            qfi_values[:100] = gaussian_filter(qfi_values[:100], sigma=1)
            y_smooth = qfi_values

        qfi_values[qfi_values < 0] = 0  # Ensure no negative QFI values
        # Smooth QFI values for visualization

        # Labels for each state
        if state == "GS_PHYS":
            label = (r"$\left(\big|E_1 {\bigr \rangle} + "
                     r"\big|E_{\overline{1}} {\bigr \rangle} \right) / \sqrt{2}$")
        elif state == "GS_CAT":
            label = r"$\big|E_1 {\bigr \rangle} $"
        elif state == "PHYS":
            label = r"$\big|\uparrow ... \uparrow {\bigr \rangle}$"
        elif state == "CAT_SUM":
            label = (r"$\left(\big|\uparrow ... \uparrow {\bigr \rangle} + "
                     r"\big|\downarrow ... \downarrow {\bigr \rangle} "
                     r"\right)/\sqrt{2}$")
        else:
            label = state
        # Plot QFI data
        ax.plot(time_points, y_smooth, "-", label=label, linewidth=3)
    # Customize QFI subplot
    ax.set_title(rf"QFI dynamics for $N={n}, B/J={h:.2f}$", fontsize=40)
    ax.set_xlabel(r"$t / T$", fontsize=40)
    ax.set_ylabel(r"$F_h / (N t)^2$", fontsize=40)
    # ax.legend(
    #     title="Initial State", loc=(0.33, 0.55), fontsize=28, title_fontsize=26
    # ).set_zorder(10)
    ax.set_xscale("log")  # Logarithmic scale for time
    ax.set_ylim([0, np.abs((1 - h ** 2) * 4 / np.pi ** 2)])
    ax.set_xticks([10 ** n for n in range(1, max_time_pow, 2)])
    ax.tick_params(axis="x", labelsize=30)  # Adjust x-axis tick label font size
    ax.tick_params(axis="y", labelsize=30)  # Adjust y-axis tick label font size
    ax.grid(True, linestyle="--", alpha=0.6, linewidth=1.7)
    # ax.text(-0.135, 1, r"(c)", fontsize=50, fontweight="bold",
    #         va="top", ha="left", transform=ax.transAxes)
    H = create_hamiltonian_h0(J, h, n, precision="mpmath")
    symmetry_edge = -1 * h * n
    eigvals_H_real = find_eigen_values_mpmath(H)
    # Add vertical lines for gap points
    pairs = {}
    for idx, eigenvalue in enumerate(eigvals_H_real):
        pair_num = idx // 2
        if pair_num not in pairs:
            pairs[pair_num] = []
        pairs[pair_num].append(float(eigenvalue))
    for pair_idx, energies in pairs.items():
        if energies[0] <= symmetry_edge and pair_idx < 7:
            if len(energies) == 2:
                gap = abs(energies[1] - energies[0])
                ax.axvline(x=2 * np.pi / gap, color='black',
                           linestyle='--', alpha=0.7, linewidth=2)
                # Adding the annotation with value 1/Δ
                ax.annotate(
                    f"$\\frac{{2\\pi}}{{\\Delta_{{{pair_idx+1}\\overline{{{pair_idx+1}}}}}}}$",
                    xy=(2 * np.pi / gap, Y_LABEL_COORDINATE),
                    xytext=(2 * np.pi / gap * 1.03, Y_LABEL_COORDINATE),
                    fontsize=54,
                    color='black'
                )
    # ax.set_title("QFI Dynamics", fontsize=22)


def plot_both_subplots(J, h, n, tau, df):
    # create floquet_spectrum grid
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_quasienergies(ax, J, h, n)
    plt.tight_layout()
    plt.savefig(results_dir / "floquet_spectrum.png", dpi=300)
    # Create a qfi_dynamics figure
    fig, ax = plt.subplots(figsize=(15, 10))
    # # Define the GridSpec with 1 row and 4 columns (3+1 for proportions)
    # gs = GridSpec(7, 1, figure=fig)
    # # Assign space for the two subplots
    # ax1 = fig.add_subplot(gs[0:6, 0])
    # ax2 = fig.add_subplot(gs[6, 0])
    # Plot the QFI data in the second subplot
    plot_qfi_data_subplot(ax, df, J, h, n, tau, max_time_pow=MAX_TIME_POW)
    # Plot the energy levels in the first subplot
    # plot_energy_levels_subplot(ax2, J, h, n)
    # Customize layout and save the figure
    plt.tight_layout()
    plt.savefig(results_dir / f"qfi_dynamics_h={h:.2f}_N={n}.png", dpi=300)


def plot_data(df):
    """
    Plots the Quantum Fisher Information (QFI) values for different initial states.

    This function visualizes the QFI values for various initial states as a function
    of time, formatted with specific conditions derived from the input data and analytical
    calculations. It includes separate plots for physical and correlated quantum states,
    and compares them with analytical solutions for specific parameters. The visualized
    QFI values are normalized and plotted on a logarithmic scale for better observation
    of trends across time scales.

    :param df:
        A pandas DataFrame containing the data to plot. It must include the following
        columns:
            - 'phi_mix': Determines the quantum initial states being plotted.
            - 'time_points': The time steps corresponding to QFI measurements.
            - 'qfi_values': The calculated QFI values at those time points.
    :param title_add:
        A boolean flag indicating whether to add a centered title to the plot. Default is False.

    :return:
        None. Saves the generated QFI plot as 'qfi_diffstates.png' in the working directory
    """
    configs = read_defaults_args_from_config()
    tau = configs["tau"]
    N = configs["system_size"]
    h = configs["h_field_strength"]
    J = configs["J_coupling"]
    plot_both_subplots(J, h, N, tau, df)


if __name__ == "__main__":
    # Read and plot
    df = read_csv_data("general_case.png.csv")
    plot_data(df)
