# Code the reproduce the figures 8 a-d: change M3 to True to get M3 figures

import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.dirname(os.path.dirname(DIR))
os.chdir(DIR)
sys.path.append(DIR)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots

# set style to science
plt.style.use("science")
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 14  # X-axis tick labels
plt.rcParams["ytick.labelsize"] = 14  # Y-axis tick labels
plt.rcParams.update({"font.size": 14})

dataset_list = ["M3", "M4"]
method_list = ["TARW"]


def generate_plots(dataset: str, method: str):
    assert dataset in ["M3", "M4"], "Invalid dataset"
    assert method in ["TARW", "static"], "Invalid method"

    # Setup
    plt.style.use("science")
    plt.rcParams.update(
        {
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "font.size": 14,
        }
    )

    is_M3 = dataset == "M3"
    is_TARW = method == "TARW"

    # Load data
    if is_M3:
        file = "M3TARW_acc_stab.csv" if is_TARW else "M3lambda_acc_stab.csv"
    else:
        file = "M4TARW_acc_stab.csv" if is_TARW else "M4lambda_acc_stab.csv"
    df = pd.read_csv(os.path.join("Graph_data", file))

    if is_M3 and is_TARW:
        df.loc[df["lambda"] == 0, "lambda_cap"] = 0

    # Average group
    if is_TARW:
        group_col = "lambda_cap"
    else:
        group_col = "lambda"

    df_transformed = df[[group_col, "eloss_fcacc_evol", "eloss_fcstab_evol"]]
    df_transformed = df_transformed.groupby(group_col).mean().reset_index()
    df_transformed["eloss_fcacc_evol"] = df_transformed["eloss_fcacc_evol"].round(3)
    df_transformed["eloss_fcstab_evol"] = df_transformed["eloss_fcstab_evol"].round(3)

    x = df_transformed[group_col]
    y_acc = df_transformed["eloss_fcacc_evol"]
    y_stab = df_transformed["eloss_fcstab_evol"]

    def poly_plot(x, y, ylabel, filename):
        figures_dir = "figures"
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        filename = os.path.join(figures_dir, filename)
        plt.figure(figsize=(8, 6))
        coeffs = np.polyfit(x, y, deg=2)
        poly = np.poly1d(coeffs)
        x_sorted = np.sort(x)
        y_fit = poly(x_sorted)
        plt.scatter(x, y)
        plt.plot(x_sorted, y_fit, color="red", label="Fit: Degree 2")
        plt.xlabel(r"$\kappa$" if is_TARW else r"$\lambda$")
        plt.ylabel(ylabel)
        plt.savefig(filename, format="pdf", dpi=300, bbox_inches="tight")
        plt.show()

    # First plot: RMSSE vs lambda/kappa
    poly_plot(
        x, y_acc, "RMSSE", f"RMSSE vs {'kappa' if is_TARW else 'lambda'} {dataset}.pdf"
    )

    # Second plot: RMSSC vs lambda/kappa
    poly_plot(
        x, y_stab, "RMSSC", f"RMSSC vs {'kappa' if is_TARW else 'lambda'} {dataset}.pdf"
    )

    # Third plot: RMSSC vs RMSSE colored by lambda/kappa
    plt.figure(figsize=(8, 6))
    plt.scatter(y_acc, y_stab, c=x, cmap="viridis")
    plt.xlabel("RMSSE")
    plt.ylabel("RMSSC")
    cbar = plt.colorbar()
    cbar.set_label(r"$\kappa$" if is_TARW else r"$\lambda$")
    name = f"{method} RMSSC vs RMSSE {dataset}.pdf"
    name = os.path.join("figures", name)
    if not os.path.exists("figures"):
        os.makedirs("figures")
    
    plt.savefig(
        name,
        format="pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


for dataset in dataset_list:
    for method in method_list:
        generate_plots(dataset, method)
