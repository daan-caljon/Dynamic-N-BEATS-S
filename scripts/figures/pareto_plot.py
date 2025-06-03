#Generates pareto plots (figure 2)
#Values were manually copied from the R script

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from brokenaxes import brokenaxes
from scipy import interpolate
import scienceplots
fontsize = 17

plt.style.use("science")
rank = False
interval = False
#set DIR
import os
import sys
DIR = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.dirname(os.path.dirname(DIR))
os.chdir(DIR)
sys.path.append(DIR)

#path to tables:
path = r"R scripts\tables"
# Data for M3 and M4 datasets

   
methods = np.array(
    [
        "N-BEATS",
        "N-BEATS-S low",
        "N-BEATS-S high",
        "GradNorm",
        "UW",
        "RW",
        "NashMTL",
        "GCosSim",
        "Weighted GCosSim",
        "AuxiNash",
        "TARW low",
        "TARW high",
        "ETS",
        "ARIMA",
        "THETA",
    ]
)
method_colors = {
    "GradNorm": "#5A2D82",
    "UW": "#5A2D82",
    "RW": "#5A2D82",
    "NashMTL": "#5A2D82",  # Dark Purple
    "GCosSim": "#FF8C00",
    "Weighted GCosSim": "#FF8C00",
    "AuxiNash": "#FF8C00",
    "TARW high": "#FF8C00",
    "TARW low": "#FF8C00",  # Bright Orange
    "ETS": "#228B22",
    "ARIMA": "#228B22",
    "THETA": "#228B22",
    "N-BEATS": "#228B22",  # Deep Green
    "N-BEATS-S high": "#1E90FF",
    "N-BEATS-S low": "#1E90FF",  # Bright Blue
}

# RMSSE and RMSSC values for M3 Monthly
#get the results from M3RMSSE.csv tables
rmsse_m3 = pd.read_csv(os.path.join(path, "M3RMSSE.csv"))
rmsse_m3 = rmsse_m3[methods]
rmsse_m3 = rmsse_m3.values.flatten()
rmssc_m3 = pd.read_csv(os.path.join(path, "M3RMSSC.csv"))
rmssc_m3 = rmssc_m3[methods]
rmssc_m3 = rmssc_m3.values.flatten()
rmsse_m4 = pd.read_csv(os.path.join(path, "M4RMSSE.csv"))
rmsse_m4 = rmsse_m4[methods]
rmsse_m4 = rmsse_m4.values.flatten()
rmssc_m4 = pd.read_csv(os.path.join(path, "M4RMSSC.csv"))
rmssc_m4 = rmssc_m4[methods]
rmssc_m4 = rmssc_m4.values.flatten()
smape_m3 = pd.read_csv(os.path.join(path, "M3sMAPE.csv"))
smape_m3 = smape_m3[methods]
smape_m3 = smape_m3.values.flatten()
smapc_m3 = pd.read_csv(os.path.join(path, "M3sMAPC.csv"))
smapc_m3 = smapc_m3[methods]
smapc_m3 = smapc_m3.values.flatten()
smape_m4 = pd.read_csv(os.path.join(path, "M4sMAPE.csv"))
smape_m4 = smape_m4[methods]
smape_m4 = smape_m4.values.flatten()
smapc_m4 = pd.read_csv(os.path.join(path, "M4sMAPC.csv"))
smapc_m4 = smapc_m4[methods]


# rmsse_m3 = np.array(
#     [
#         1.088,
#         1.089,
#         1.088,
#         1.122,
#         1.172,
#         1.174,
#         1.124,
#         1.165,
#         1.110,
#         1.101,
#         1.087,
#         1.082,
#         1.048,
#         1.044,
#         1.094,
#     ]
# )
# rmssc_m3 = np.array(
#     [
#         0.393,
#         0.382,
#         0.352,
#         0.226,
#         0.145,
#         0.165,
#         0.251,
#         0.157,
#         0.271,
#         0.297,
#         0.363,
#         0.349,
#         0.411,
#         0.419,
#         0.366,
#     ]
# )

# # RMSSE and RMSSC values for M4 Monthly
# rmsse_m4 = np.array(
#     [
#         1.266,
#         1.256,
#         1.274,
#         1.299,
#         1.526,
#         1.354,
#         1.278,
#         1.350,
#         1.265,
#         1.260,
#         1.257,
#         1.257,
#         1.322,
#         1.271,
#         1.406,
#     ]
# )
# rmssc_m4 = np.array(
#     [
#         0.556,
#         0.537,
#         0.364,
#         0.324,
#         0.146,
#         0.224,
#         0.339,
#         0.234,
#         0.384,
#         0.410,
#         0.499,
#         0.460,
#         0.594,
#         0.571,
#         0.526,
#     ]
# )

# smape_m3 = np.array(
#     [
#         11.44,
#         11.43,
#         11.40,
#         11.47,
#         11.62,
#         11.64,
#         11.42,
#         11.59,
#         11.41,
#         11.39,
#         11.42,
#         11.40,
#         11.34,
#         11.70,
#         11.28,
#     ]
# )
# smapc_m3 = np.array(
#     [
#         3.65,
#         3.45,
#         3.07,
#         1.63,
#         0.92,
#         1.06,
#         1.82,
#         1.05,
#         2.07,
#         2.28,
#         3.22,
#         2.97,
#         3.21,
#         3.16,
#         2.96,
#     ]
# )

# # sMAPE and sMAPC values for M4 Monthly
# smape_m4 = np.array(
#     [
#         9.12,
#         9.12,
#         9.24,
#         9.37,
#         10.56,
#         9.76,
#         9.27,
#         9.70,
#         9.18,
#         9.15,
#         9.13,
#         9.13,
#         9.98,
#         9.78,
#         10.07,
#     ]
# )
# smapc_m4 = np.array(
#     [
#         3.88,
#         3.69,
#         2.22,
#         1.89,
#         0.84,
#         1.22,
#         2.06,
#         1.28,
#         2.40,
#         2.64,
#         3.36,
#         3.04,
#         4.38,
#         4.15,
#         3.80,
#     ]
# )

# method_colors = {
#     "GradNorm": "#5A2D82", "UW": "#5A2D82", "RW": "#5A2D82", "NashMTL": "#5A2D82",  # Dark Purple
#     "GCosSim": "#FF8C00", "Weighted GCosSim": "#FF8C00", "AuxiNash": "#FF8C00", "TARW": "#FF8C00",  # Bright Orange
#     "ETS": "#228B22", "ARIMA": "#228B22", "THETA": "#228B22", "N-BEATS": "#228B22",  # Deep Green
#     "N-BEATS-S": "#1E90FF"  # Bright Blue
# }


# Function to calculate Pareto front (for minimization)
def pareto_front(rmsse, rmssc):
    is_pareto = np.ones(rmsse.shape[0], dtype=bool)
    for i in range(rmsse.shape[0]):
        for j in range(rmsse.shape[0]):
            if i != j:
                # If there exists another point with a strictly better performance in at least one metric without worsening the other
                if (rmsse[j] <= rmsse[i] and rmssc[j] < rmssc[i]) or (
                    rmsse[j] < rmsse[i] and rmssc[j] <= rmssc[i]
                ):
                    is_pareto[i] = False
                    break

    pareto_rmsse = rmsse[is_pareto]
    pareto_rmssc = rmssc[is_pareto]
    sorted_indices = np.argsort(-pareto_rmssc)  # Sort by RMSSC descending
    return pareto_rmsse[sorted_indices], pareto_rmssc[sorted_indices]


# Plot function with Pareto front
def plot_rmsse_rmssc(rmsse, rmssc, title):
    pareto_rmsse, pareto_rmssc = pareto_front(rmsse, rmssc)

    plt.figure(figsize=(10, 6))
    # n_beats_s_idx = np.where(methods == "N-BEATS-S")[0][0]  # Use np.where to find the index
    # n_beats_s_rmsse = rmsse[n_beats_s_idx]
    # n_beats_s_rmssc = rmssc[n_beats_s_idx]
    # plt.scatter(rmsse, rmssc, color="blue", s=20, alpha=0.7, label="Methods")
    plt.plot(
        pareto_rmsse,
        pareto_rmssc,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Pareto Front",
    )
    # plt.scatter(rmsse, rmssc, color="blue", s=20, alpha=0.7, label="Methods")
    f = interpolate.interp1d(pareto_rmsse, pareto_rmssc, kind="linear")
    x_interp = np.linspace(min(pareto_rmsse), max(pareto_rmsse), 300)
    y_interp = f(x_interp)
    # Label each point with the method name
    texts = []
    # mask = (rmsse <= n_beats_s_rmsse) & (rmssc <= n_beats_s_rmssc)

    for i, method in enumerate(methods):
        color = method_colors.get(method, "black")
        plt.scatter(rmsse[i], rmssc[i], color=color, s=20, alpha=1)
        texts.append(
            plt.text(
                rmsse[i],
                rmssc[i],
                method,
                fontsize=fontsize,
                color=color,
                ha="center",
                va="center",
            )
        )

    # adjust_text(
    #     texts,
    #     expand=(1.2, 1.7),
    #     x=np.concatenate([pareto_rmsse, rmsse]),
    #     y=np.concatenate([pareto_rmssc, rmssc]),
    #     arrowprops=dict(arrowstyle="->", color='gray', lw=0.5)
    #     )
    # x_min, x_max = min(pareto_rmsse), max(pareto_rmsse)
    # y_min, y_max = min(pareto_rmssc), max(pareto_rmssc)
    # buffer = 0.1  # Adjust this value as needed
    # avoid_zone_x = np.linspace(x_min - buffer, x_max + buffer, 500)
    # avoid_zone_y = np.linspace(y_min - buffer, y_max + buffer, 500)

    # adjust_text(
    #     texts,
    #     expand=(1.5, 2.0),
    #     x=avoid_zone_x,
    #     y=avoid_zone_y,
    #     arrowprops=dict(arrowstyle="->", color='gray', lw=0.5)
    # )

    adjust_text(
        texts,
        expand=(1.2, 1.7),
        x=x_interp,
        y=y_interp,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
    )  # 1.5-1.7

    # plt.xlim(left=6.2)  # Set left boundary to 0 (or adjust as needed)
    # plt.ylim(bottom=1.5)  # Set bottom boundary to 0 (or adjust as needed)

    # x_min, x_max = plt.xlim()
    # y_min, y_max = plt.ylim()
    if rank:
        if "M3" in title:
            x_min = min(rmsse) - 0.2
            y_min = min(rmssc) - 0.2
        else:
            x_min = min(rmsse) - 0.5
            y_min = min(rmssc) - 0.5
    else:
        x_min = min(rmsse) - 0.02
        y_min = min(rmssc) - 0.05
    # plt.xlim(left=x_min)
    # plt.ylim(bottom=y_min)
    plt.xlim(left=x_min)
    plt.ylim(bottom=y_min)

    # Fill the southwest area relative to "N-BEATS-S"
    # plt.fill_betweenx(
    #     y=[y_min, n_beats_s_rmssc],  # From bottom y-axis limit to N-BEATS-S RMSSC
    #     x1=x_min,                     # Entire x-axis from left limit
    #     x2=n_beats_s_rmsse,           # Up to N-BEATS-S RMSSE
    #     color="#1E90FF",
    #     alpha=0.1
    # )

    if not rank:
        plt.xlabel("RMSSE", fontsize=fontsize)
        plt.ylabel("RMSSC", fontsize=fontsize)
    else:
        plt.xlabel("RMSSE rank")
        plt.ylabel("RMSSC rank")
    # adjust font sizes of axes
    plt.xticks(fontsize=fontsize - 3)
    plt.yticks(fontsize=fontsize - 3)
    # plt.title(title)
    # plt.legend()
    plt.grid(True)
    folder = "figures"
    save = title + ".pdf"
    if not os.path.exists(folder):
        os.makedirs(folder)
    save = os.path.join(folder, save)
    plt.savefig(save, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()


def plot_rmsse_rmssc_with_cut(rmsse, rmssc, title):
    pareto_rmsse, pareto_rmssc = pareto_front(rmsse, rmssc)

    # Create a figure with broken axes
    fig = plt.figure(figsize=(10, 6))
    bax = brokenaxes(
        xlims=((min(rmsse) - 0.01, 1.41), (1.5, max(rmsse) + 0.005)), hspace=0.05
    )  # Define the cuts on x-axis

    # Plot Pareto Front
    bax.plot(
        pareto_rmsse,
        pareto_rmssc,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Pareto Front",
    )
    texts = []
    # Scatter plot for individual methods
    for i, method in enumerate(methods):
        color = method_colors.get(method, "black")
        bax.scatter(rmsse[i], rmssc[i], color=color, s=20, alpha=1)
        if method == "UW":
            bax.text(
                rmsse[i],
                rmssc[i],
                method,
                fontsize=fontsize,
                color=color,
                ha="center",
                va="center",
            )
        else:
            texts.append(
                bax.text(
                    rmsse[i],
                    rmssc[i],
                    method,
                    fontsize=fontsize,
                    color=color,
                    ha="center",
                    va="center",
                )
            )

    # Adjust text
    f = interpolate.interp1d(pareto_rmsse, pareto_rmssc, kind="linear")
    x_interp = np.linspace(min(pareto_rmsse), max(pareto_rmsse), 300)
    y_interp = f(x_interp)
    # texts = [
    #     bax.text(rmsse[i], rmssc[i], method, fontsize=fontsize, color=color, ha='center', va='center')
    #     for i, method in enumerate(methods)
    # ]
    adjust_text(
        texts,
        expand=(1.2, 1.7),
        x=x_interp,
        y=y_interp,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
    )

    # Add labels and grid
    bax.set_xlabel("RMSSE", fontsize=fontsize)
    bax.set_ylabel("RMSSC", fontsize=fontsize)
    bax.grid(True)

    # Save and display the plot
    save = title + ".pdf"
    plt.savefig(save, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()


# Plot for M3 monthly with Pareto front
title = "RMSSE vs RMSSC for M3 Monthly Dataset"
if rank:
    title += " (Rank)"
plot_rmsse_rmssc(rmsse_m3, rmssc_m3, title)

# Plot for M4 monthly with Pareto front
title = "RMSSE vs RMSSC for M4 Monthly Dataset"
if rank:
    title += " (Rank)"
plot_rmsse_rmssc(rmsse_m4, rmssc_m4, title)
# plot_rmsse_rmssc_with_cut(rmsse_m4, rmssc_m4, title)

# methods = [
#     "N-BEATS", "N-BEATS-S", "GradNorm", "UW", "RW",
#     "NashMTL", "GCosSim", "Weighted GCosSim", "AuxiNash",
#     "TARW", "ETS", "ARIMA", "THETA"
# ]

# sMAPE and sMAPC values for M3 Monthly
# smape_m3 = np.array([11.44, 11.38, 11.47, 11.62, 11.64, 11.66, 11.59, 11.41, 11.51, 11.37, 11.34, 11.70, 11.28])
# smapc_m3 = np.array([3.65, 2.61, 1.63, 0.92, 1.06, 1.70, 1.05, 2.07, 3.07, 2.38, 3.21, 3.16, 2.96])

# # sMAPE and sMAPC values for M4 Monthly
# smape_m4 = np.array([9.12, 9.11, 9.23, 10.56, 9.76, 9.35, 9.70, 9.18, 9.44, 9.11, 9.98, 9.78, 10.07])
# smapc_m4 = np.array([3.88, 2.95, 2.17, 0.84, 1.22, 2.70, 1.28, 2.40, 2.62, 2.68, 4.38, 4.15, 3.80])

# Function to calculate Pareto front (for minimization)


# Plot function with Pareto front
def plot_smape_smapc(rmsse, rmssc, title):
    pareto_rmsse, pareto_rmssc = pareto_front(rmsse, rmssc)

    plt.figure(figsize=(10, 6))
    # n_beats_s_idx = np.where(methods == "N-BEATS-S")[0][0]  # Use np.where to find the index
    # n_beats_s_rmsse = rmsse[n_beats_s_idx]
    # n_beats_s_rmssc = rmssc[n_beats_s_idx]

    # plt.scatter(rmsse, rmssc, color="blue", s=20, alpha=0.7, label="Methods")
    plt.plot(
        pareto_rmsse,
        pareto_rmssc,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Pareto Front",
    )
    # plt.scatter(rmsse, rmssc, color="blue", s=20, alpha=0.7, label="Methods")
    f = interpolate.interp1d(pareto_rmsse, pareto_rmssc, kind="linear")
    x_interp = np.linspace(min(pareto_rmsse), max(pareto_rmsse), 300)
    y_interp = f(x_interp)
    # Label each point with the method name
    texts = []

    for i, method in enumerate(methods):
        color = method_colors.get(method, "black")
        plt.scatter(rmsse[i], rmssc[i], color=color, s=20, alpha=0.7)
        texts.append(
            plt.text(
                rmsse[i],
                rmssc[i],
                method,
                color=color,
                fontsize=fontsize,
                ha="center",
                va="center",
            )
        )

    adjust_text(
        texts,
        expand=(1.2, 1.7),
        x=x_interp,
        y=y_interp,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
    )  # 1.5-1.7

    # plt.xlim(left=6.2)  # Set left boundary to 0 (or adjust as needed)
    # plt.ylim(bottom=1.5)  # Set bottom boundary to 0 (or adjust as needed)
    if rank:
        if "M3" in title:
            x_min = min(rmsse) - 0.2
            y_min = min(rmssc) - 0.2
        else:
            x_min = min(rmsse) - 0.5
            y_min = min(rmssc) - 0.5
    else:
        x_min = min(rmsse) - 0.1
        y_min = min(rmssc) - 0.1
    plt.xlim(left=x_min)
    plt.ylim(bottom=y_min)

    # Fill the southwest area relative to "N-BEATS-S"
    # plt.fill_betweenx(
    #     y=[y_min, n_beats_s_rmssc],  # From bottom y-axis limit to N-BEATS-S RMSSC
    #     x1=x_min,                     # Entire x-axis from left limit
    #     x2=n_beats_s_rmsse,           # Up to N-BEATS-S RMSSE
    #     color="#1E90FF",
    #     alpha=0.1
    # )
    if not rank:
        plt.xlabel("sMAPE")
        plt.ylabel("sMAPC")
    else:
        plt.xlabel("sMAPE rank")
        plt.ylabel("sMAPC rank")
    # plt.title(title)
    # plt.legend()
    plt.grid(True)
    save = title + ".pdf"
    plt.savefig(save, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()


# Plot for M3 monthly with Pareto front
title = "sMAPE vs sMAPC for M3 Monthly Dataset"
if rank:
    title += " (Rank)"
plot_smape_smapc(smape_m3, smapc_m3, title)

# Plot for M4 monthly with Pareto front
title = "sMAPE vs sMAPC for M4 Monthly Dataset"
if rank:
    title += " (Rank)"

plot_smape_smapc(smape_m4, smapc_m4, title)
