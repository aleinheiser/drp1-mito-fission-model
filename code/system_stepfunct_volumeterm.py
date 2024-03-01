# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 10:38:04 2022

@author: alein
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


k1 = 1
k2 = 0.02

k3 = 5
k4 = 0.1
v = 0.1

initial = np.zeros(32)
initial[0] = 20
initial[1] = 15
s = np.arange(1, 31)

b = 25
a = 5

f = np.zeros((30))

for i in range(0, b):
    f[i] = 0

for i in range(b, len(f)):
    f[i] = a


# system of ODEs
def g(t, X, k1, k2, k3, k4, f, v):  # X is state of the system at time t
    T = X[0]
    M = X[1]
    C = X[2:]
    result = np.zeros(len(X))
    result[0] = v * k2 * C[0] + v * sum(C[0:] * f[0:] * s[0:]) - v * k1 * T * M
    result[1] = k2 * C[0] + sum(C[0:] * f[0:] * s[0:]) - k1 * T * M
    result[2] = (
        k1 * T * M
        + k4 * (2 * C[1] + sum(C[2:]))
        - k2 * C[0]
        - k3 * (2 * C[0] ** 2 + sum(C[1:29] * C[0]))
        - f[0] * C[0]
    )
    for i in range(3, len(result) - 1):  # i = 3,4,...,30
        result[i] = (
            k3 * C[i - 3] * C[0]
            + k4 * C[i - 1]
            - k4 * C[i - 2]
            - k3 * C[i - 2] * C[0]
            - f[i - 2] * C[i - 2]
        )

    result[len(result) - 1] = (
        k3 * C[len(result) - 4] * C[0]
        - k4 * C[len(result) - 3]
        - f[len(result) - 3] * C[len(result) - 3]
    )
    return result


if __name__ == "__main__":
    plt.rcParams["text.usetex"] = False
    # Solving the system
    solution = integrate.solve_ivp(
        g,
        (0, 5000),
        initial,
        args=(k1, k2, k3, k4, f, v),
        method="RK45",
        dense_output=True,
    )

    consvquan = np.zeros(len(solution.t))

    for k in range(len(solution.t)):
        consvquan[k] = solution.y[0, k] + v * sum(s[0:] * solution.y[2:, k])

    width = 7.5
    height = 8.75
    fig, axes = plt.subplots(1, 2, figsize=(width, 0.33 * height), layout="constrained")
    ax = axes[0]
    # Plot of solutions
    # plt.rcParams.update({'font.size': 12})
    for i in range(1, 32):
        ax.plot(solution.t, solution.y[i])
    ax.set_xlabel("time ($s$)", size=12)
    ax.set_ylabel("Concentration ($nM$)", size=12)

    # #fission_solution1
    # ax.xlim(0,5000)
    # ax.ylim(0,0.4)
    # #ax.text(1500,0.3,'Each curve represents a solution to...', size = 11)

    # fission_solution2
    ax.set_title("A", loc="right", size=12, fontweight="bold")
    ax.set_xlim(0, 4000)
    ax.set_ylim(0, 0.1)
    # ax.text(1000,0.08, 'Each line represents an oligomer size', size = 11)
    ax.text(2500, 0.012, "$M(t)$", size=8, color="tab:blue")
    ax.text(2500, 0.0035, "$C_{26}(t)$", size=8, color="tab:pink")
    ax.text(2500, 0.032, "$C_{25}(t)$", size=8, color="tab:brown")
    ax.text(2500, 0.053, "$C_1(t)$", size=8, color="tab:orange")
    ax.text(2920, 0.053, "-", size=8, color="black")
    ax.text(3000, 0.053, "$C_{24}(t)$", size=8, color="tab:purple")

    # # #fission_solution3
    # plt.xlim(0,2500)
    # plt.ylim(0,0.075)
    # plt.text(850,0.066,'Each line represents an oligomer size', size = 11)
    # plt.text(2000,0.012, '$M(t)$', size = 11, color = 'tab:blue')
    # plt.text(2000, 0.004, '$C_{26}(t)$', size = 11, color = 'tab:pink')
    # plt.text(2000, 0.033, '$C_{25}(t)$', size = 11, color = 'tab:brown')

    # # #Total Fission Plot
    totalfisrate = np.zeros(len(solution.t))

    totalfisrate = np.sum(f.reshape(-1, 1) * solution.y[2:, :], axis=0)
    ax = axes[1]
    ax.plot(solution.t, totalfisrate, color="black")
    ax.set_title("B", loc="right", size=12, fontweight="bold")
    # ax.plot(solution.t, solution.y[27], color = 'tab:pink')
    ax.set_xlabel("time ($s$)", size=12)
    ax.set_ylabel("TFR ($nM/s$)", size=12)
    ax.text(2000, 0.01, r"$TFR(t) = a \sum_{i=26}^{30} C_i(t)$", size=10)
    # plt.text(3000,0.0018, '$C_{26}(t)$', size = 11, color = 'tab:pink')

    fig.savefig("Fig4.svg")
