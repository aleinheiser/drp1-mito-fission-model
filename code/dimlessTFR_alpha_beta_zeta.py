# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:14:30 2023

@author: alein
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import dimensionless_fission_system as system
from functools import cache

p = np.array([1.75, 1.25, 1, 0.75, 0.25])
prefixes = [r"+75\%\ ", r"+25\%\ ", r"", r"-25\%\ ", r"-75\%\ "]


@cache
def get_solution(alpha=system.alpha, beta=system.beta, zeta=system.zeta):
    return integrate.solve_ivp(
        system.g,
        (0, 500),
        system.initial,
        args=(system.u, alpha, beta, zeta, system.f, system.v),
        dense_output=True,
    )


def plot_alpha(ax):
    """Plot TFR alpha"""
    labels = [f"${prefix}\\alpha$" for prefix in prefixes]
    A = system.alpha * p

    for a, label in zip(A, labels):
        solution = get_solution(alpha=a)
        totalfisrate = np.sum(system.f.reshape(-1, 1) * solution.y[2:, :], axis=0)
        ax.plot(solution.t, totalfisrate, label=label)

    ax.set_xlabel("time", size=12)
    ax.set_ylabel("TFR", size=12)


############################################################################################
# TFR_beta
def plot_beta(ax):
    labels = [f"${prefix}\\beta$" for prefix in prefixes]
    B = system.beta * p
    for b, label in zip(B, labels):
        solution = get_solution(beta=b)
        totalfisrate = np.sum(system.f.reshape(-1, 1) * solution.y[2:, :], axis=0)
        ax.plot(solution.t, totalfisrate, label=label)

    ax.legend()
    ax.set_xlabel("time", size=12)
    ax.set_ylabel("TFR", size=12)


################################################################################################################
# TFR_zeta
def plot_zeta(ax):
    labels = [f"${prefix}\\zeta$" for prefix in prefixes]
    Z = system.zeta * p

    for z, label in zip(Z, labels):
        solution = get_solution(zeta=z)
        totalfisrate = np.sum(system.f.reshape(-1, 1) * solution.y[2:, :], axis=0)
        ax.plot(solution.t, totalfisrate, label=label)

    ax.legend()
    ax.set_xlabel("time", size=12)
    ax.set_ylabel("TFR", size=12)


################################################################################################################
# Make plots
################################################################################################################

if __name__ == "__main__":
    width = 7.5
    height = 8.75
    fig, axes = plt.subplots(
        1, 3, figsize=(width, 0.4 * height), dpi=600, layout="constrained"
    )

    plot_alpha(axes[0])
    axes[0].set_title("A", loc="right", size=12, fontweight="bold")

    plot_beta(axes[1])
    axes[1].set_title("B", loc="right", size=12, fontweight="bold")

    plot_zeta(axes[2])
    axes[2].set_title("C", loc="right", size=12, fontweight="bold")

    fig.savefig("TFR_alpha_beta_zeta")
