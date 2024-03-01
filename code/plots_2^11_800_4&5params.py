# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 14:00:49 2024

@author: alein
"""

import pickle

from util import add_title, base
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

width = 7.5
height = 8.75
GSAparams_fig = plt.figure(
    figsize=(width, 0.8 * height),
    layout="constrained",
)

gs1 = GridSpec(3, 2, figure=GSAparams_fig, wspace=0.1)

GSA_scatter = plt.figure(
    figsize=(width, 0.7 * height),
    dpi=600,
    layout="constrained",
)

gs2 = GridSpec(2, 2, figure=GSA_scatter, wspace=0.0625)

with open(base() / "result_800_2^11_4params.pickle", "rb") as f:
    Si_4params = pickle.load(f)

problem1 = {
    "num_vars": 4,
    "names": ["k1", "k2", "k3", "k4"],
    "bounds": [
        [0, 10],
        [0, 10],
        [0, 10],
        [0, 1],
    ],
    "dist": ["unif", "unif", "unif"],
}

with open(base() / "result_800_2^11_5params.pickle", "rb") as f:
    Si_5params = pickle.load(f)


problem2 = {
    "num_vars": 4,
    "names": ["k1", "k2", "k3", "k4", "M"],
    "bounds": [
        [0, 10],
        [0, 10],
        [0, 10],
        [0, 1],
        [0, 26],
    ],
    "dist": ["unif", "unif", "unif", "unif", "unif"],
}


def summarize(problem, si):
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


summarize(problem1, Si_4params)
summarize(problem2, Si_5params)


# Labels
kp = r"$k_+\,\,(nMs)^{-1}$"
km = r"$k_-\,\,(s^{-1})$"
script_M = r"$\mathcal{M}\,\,(nM)$"
cumtf = r"Cumulative TF $(nM)$"

# Scatter plots for 4 params and 5 params
input_4params = np.loadtxt(base() / "input_800_2^11_4params.txt")
output_4params = np.loadtxt(base() / "output_800_2^11_4params.txt")
input_5params = np.loadtxt(base() / "input_800_2^11_5params.txt")
output_5params = np.loadtxt(base() / "output_800_2^11_5params.txt")

# k+ vs Cumulative total fission
ax1 = GSAparams_fig.add_subplot(gs1[0, 0])
k3_4params = ax1.scatter(
    input_4params[:, 2], output_4params, s=6, c=output_4params, cmap="turbo"
)
clb = plt.colorbar(k3_4params)
clb.set_label(cumtf, size=14)
ax1.set_xlabel(kp, size=12)
ax1.set_yticks([])
add_title(ax1, "A")

ax1 = GSAparams_fig.add_subplot(gs1[0, 1])
k3_5params = ax1.scatter(
    input_5params[:, 2], output_5params, s=6, c=output_5params, cmap="turbo"
)
clb = plt.colorbar(k3_5params)
clb.set_label(cumtf, size=14)
ax1.set_xlabel(kp, size=12)
ax1.set_yticks([])
add_title(ax1, "B")

# k- vs Cumulative total fission
ax1 = GSAparams_fig.add_subplot(gs1[1, 0])
k4_4params = ax1.scatter(
    input_4params[:, 3], output_4params, s=6, c=output_4params, cmap="turbo"
)
clb = plt.colorbar(k4_4params)
clb.set_label(cumtf, size=14)
ax1.set_xlabel(km, size=12)
ax1.set_yticks([])
add_title(ax1, "C")

ax1 = GSAparams_fig.add_subplot(gs1[1, 1])
k4_5params = ax1.scatter(
    input_5params[:, 3], output_5params, s=6, c=output_5params, cmap="turbo"
)
clb = plt.colorbar(k4_5params)
clb.set_label(cumtf, size=14)
ax1.set_xlabel(km, size=12)
ax1.set_yticks([])
add_title(ax1, "D")

# M vs cumTF
ax1 = GSAparams_fig.add_subplot(gs1[2, 0])
M_5params = ax1.scatter(
    input_5params[:, 4], output_5params, s=6, c=output_5params, cmap="turbo"
)
clb = plt.colorbar(M_5params)
clb.set_label(cumtf, size=14)
ax1.set_xlabel(script_M, size=12)
ax1.set_yticks([])
add_title(ax1, "E")

GSAparams_fig.savefig("Fig7.svg")

# k+:k- vs cumTF
ax2 = GSA_scatter.add_subplot(gs2[0, 0])
k3_k4_4params = ax2.scatter(
    input_4params[:, 2], input_4params[:, 3], s=6, c=output_4params, cmap="turbo"
)
clb = plt.colorbar(k3_k4_4params)
clb.set_label(cumtf, size=12)
ax2.set_xlabel(kp, size=12)
ax2.set_ylabel(km, size=12)
add_title(ax2, "A")

ax2 = GSA_scatter.add_subplot(gs2[0, 1])
k3_k4_5params = ax2.scatter(
    input_5params[:, 2], input_5params[:, 3], s=6, c=output_5params, cmap="turbo"
)
clb = plt.colorbar(k3_k4_5params)
clb.set_label(cumtf, size=12)
ax2.set_xlabel(kp, size=12)
ax2.set_ylabel(km, size=12)
add_title(ax2, "B")

# k+:M vs cumTF
ax2 = GSA_scatter.add_subplot(gs2[1, 0])
k3_M = ax2.scatter(
    input_5params[:, 2], input_5params[:, 4], s=6, c=output_5params, cmap="turbo"
)
clb = plt.colorbar(k3_M)
clb.set_label(cumtf, size=12)
ax2.set_xlabel(kp, size=12)
ax2.set_ylabel(script_M, size=12)
add_title(ax2, "C")

# M:k- vs cumTF
ax2 = GSA_scatter.add_subplot(gs2[1, 1])
M_k4 = ax2.scatter(
    input_5params[:, 4], input_5params[:, 3], s=6, c=output_5params, cmap="turbo"
)
clb = plt.colorbar(M_k4)
clb.set_label(cumtf, size=12)
ax2.set_xlabel(script_M, size=12)
ax2.set_ylabel(km, size=12)
add_title(ax2, "D")

GSA_scatter.savefig("Fig8.svg")
