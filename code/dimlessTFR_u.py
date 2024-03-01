# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:58:39 2023

@author: alein
"""


import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import dimensionless_fission_system as system
from util import integrate_fission
from functools import cache

p = np.array([1.75, 1.25, 1, 0.75, 0.25])


labels = [
    r"$+75\%\ \mu$",
    r"$+25\%\ \mu$",
    r"$\mu$",
    r"$-25\%\ \mu$",
    r"$-75\%\ \mu$ ",
]

U = system.u * p


@cache
def get_solution(u):
    return integrate.solve_ivp(
        system.g,
        (0, 500),
        system.initial,
        args=(u, system.alpha, system.beta, system.zeta, system.f, system.v),
        dense_output=True,
    )


#######################################################################################
# TFR_cumTF_mu
def plot_tfr(ax):
    for u, label in zip(U, labels):
        solution = get_solution(u)
        totalfisrate = np.sum(system.f.reshape(-1, 1) * solution.y[2:, :], axis=0)
        ax.plot(solution.t, totalfisrate, label=label)

    ax.legend()
    plt.rc("legend", fontsize=10)
    ax.set_xlabel("time", size=12)
    ax.set_ylabel("TFR", size=12)


####


def plot_cum_tf(ax):
    integral = list()

    for u in U:
        solution = get_solution(u)
        integral.append(integrate_fission(solution))

    ax.bar(
        labels,
        np.array(integral),
        color=[
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
        ],
    )
    ax.set_xticks(
        np.arange(len(labels)),
        labels,
        size=8,
    )
    ax.set_ylabel("Cumulative TF", size=12)


if __name__ == "__main__":
    width = 7.5
    height = 8.75
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(width, 0.4 * height),
        dpi=600,
        layout="constrained",
    )

    plot_tfr(axes[0])
    axes[0].set_title("A", loc="right", size=12, fontweight="bold")

    plot_cum_tf(axes[1])
    axes[1].set_title("B", loc="right", size=12, fontweight="bold")
    fig.savefig("TFR_cumTf_mu")
