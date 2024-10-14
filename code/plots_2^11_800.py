# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 13:34:27 2023

@author: alein
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from util import add_title, base

with open(base() / "result_800_2^11_dimless.pickle", "rb") as f:
    Si_800 = pickle.load(f)

problem = {
    "num_vars": 4,
    "names": ["u", "alpha", "beta", "zeta"],
    "bounds": [
        [0, 1300],
        [0, 100],
        [0, 260],
        [0, 1],
    ],
    "dist": ["unif", "unif", "unif", "unif"],
}


def summarize(si):
    """Print a pretty summary of the sobol indices"""
    keys = ["S1", "ST", "S2"]
    for key in keys:
        print(f"{key: ^26}", "=" * 26, sep="\n")
        val = si[key]
        pm = si[f"{key}_conf"]
        n = len(val)
        if key == "S2":
            for i in range(n):
                for j in range(i + 1, n):
                    name_1 = problem["names"][i]
                    name_2 = problem["names"][j]
                    print(
                        f"({name_1:>2}, {name_2:>2}) = {val[i,j]:.4f} ± {pm[i,j]:.4f}"
                    )
        else:
            for i in range(n):
                print(f"{problem['names'][i]:>2} = {val[i]:.4f} ± {pm[i]:.4f}")


summarize(Si_800)


# BarPlots
params = [r"$\mu$", r"$\alpha$", r"$\beta$", r"$\zeta$"]

barwidth = 0.4
x1 = np.arange(len(params))
x2 = [x + barwidth for x in x1]


plt.bar(
    x1,
    Si_800["S1"][0:],
    width=barwidth,
    color="tab:blue",
    yerr=Si_800["S1_conf"][0:],
    label="$S_1$",
)

plt.bar(
    x2,
    Si_800["ST"][0:],
    width=barwidth,
    color="tab:orange",
    yerr=Si_800["ST_conf"][0:],
    label="$S_{T}$",
)


plt.xticks([x + 0.4 for x in x1], params)
plt.legend(loc="upper right", ncols=1)
# #plt.ylim(0,0.95)
plt.ylabel("Sobol Index Values", size=12)


S2_labels = [
    r"$\mu:\alpha$",
    r"$\mu:\beta$",
    r"$\mu:\zeta$",
    r"$\alpha:\beta$",
    r"$\alpha:\zeta$",
    r"$\beta:\zeta$",
]

x3 = np.array([0, 1, 2, 3, 4, 5])

S2_matrix = Si_800["S2"]
S2 = [
    S2_matrix[0, 1],
    S2_matrix[0, 2],
    S2_matrix[0, 3],
    S2_matrix[1, 2],
    S2_matrix[1, 3],
    S2_matrix[2, 3],
]

S2_conf_matrix = Si_800["S2_conf"]
S2_conf = [
    S2_conf_matrix[0, 1],
    S2_conf_matrix[0, 2],
    S2_conf_matrix[0, 3],
    S2_conf_matrix[1, 2],
    S2_conf_matrix[1, 3],
    S2_conf_matrix[2, 3],
]


plt.bar(x3, S2, width=barwidth, color="tab:pink", yerr=S2_conf)

plt.xticks([0, 1, 2, 3, 4, 5], S2_labels)
plt.legend(loc="upper left", ncols=1)
# #plt.ylim(0,0.4)
plt.ylabel("Sobol Index Values", size=12)

# Scatter plots for k_-:(0,10)
input800 = np.loadtxt(base() / "input_800_2^11_dimless.txt")
output800 = np.loadtxt(base() / "output_800_2^11_dimless.txt")


# \mu vs Cumulative total fission rate
def plot_mu_cfr(ax):
    scatter = ax.scatter(input800[:, 0], output800, s=6, c=output800, cmap="turbo")
    ax.set_xlabel(r"$\mu$ (dimensionless)", size=10)
    ax.set_yticks([])
    clb = plt.colorbar(scatter)
    clb.set_label("Cumulative TF (dimensionless)", size=10)


# \alpha vs Cumulative total fission rate
def plot_alpha_cfr(ax):
    scatter = ax.scatter(input800[:, 1], output800, s=6, c=output800, cmap="turbo")
    ax.set_xlabel(r"$\alpha$ (dimensionless)", size=10)
    ax.set_yticks([])
    clb = plt.colorbar(scatter)
    clb.set_label("Cumulative TF (dimensionless)", size=10)


# \beta vs Cumulative total fission rate
def plot_beta_cfr(ax):
    scatter = ax.scatter(input800[:, 2], output800, s=6, c=output800, cmap="turbo")
    ax.set_xlabel(r"$\beta$ (dimensionless)", size=10)
    ax.set_yticks([])
    clb = plt.colorbar(scatter)
    clb.set_label("Cumulative TF (dimensionless)", size=10)


# \zeta vs Cumulative total fission rate
def plot_zeta_cfr(ax):
    scatter = ax.scatter(input800[:, 3], output800, s=6, c=output800, cmap="turbo")
    ax.set_xlabel(r"$\zeta$ (dimensionless)", size=10)
    ax.set_yticks([])
    clb = plt.colorbar(scatter)
    clb.set_label("Cumulative TF (dimensionless)", size=10)


if __name__ == "__main__":
    width = 7.5
    height = 8.75
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(width, 0.7 * height),
        layout="constrained",
        gridspec_kw={
            "wspace": 0.0625,
        },
    )

    plot_mu_cfr(axes[0][0])
    add_title(axes[0][0], "A")
    plot_alpha_cfr(axes[0][1])
    add_title(axes[0][1], "B")
    plot_beta_cfr(axes[1][0])
    add_title(axes[1][0], "C")
    plot_zeta_cfr(axes[1][1])
    add_title(axes[1][1], "D")

    fig.savefig("Fig11.svg")
