# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:23:19 2023

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
M = 15
T = 20

initial = np.zeros(32)
initial[0] = T / M
initial[1] = 1  # z0
s = np.arange(31)

# Rate of fission dependent on size of complex
b = 25
a = 5

u = (k3 * M) / k4  # 1000
alpha = a / k4
zeta = k2 / k4
beta = (k1 * M) / k4  # 2


# building fission step funct
f = np.zeros((30))

for i in range(0, b - 1):
    f[i] = 0

for i in range(b, len(f)):
    f[i] = alpha


# system of ODEs
def g(t, X, u, alpha, beta, zeta, f, v):  # X is state of the system at time t
    z = X[0]
    y = X[1]
    x = X[2:]
    result = np.zeros(len(X))
    result[0] = v * (zeta * x[0] + sum(x[0:] * f[0:] * s[1:]) - beta * y * z)
    result[1] = zeta * x[0] + sum(x[0:] * f[0:] * s[1:]) - beta * y * z
    result[2] = (
        beta * y * z
        + (2 * x[1] + sum(x[2:]))
        - zeta * x[0]
        - u * (2 * x[0] ** 2 + sum(x[1:29] * x[0]))
        - f[0] * x[0]
    )
    for i in range(3, len(result) - 1):  # i = 3,4,...,30
        result[i] = (
            u * x[i - 3] * x[0]
            + x[i - 1]
            - x[i - 2]
            - u * x[i - 2] * x[0]
            - f[i - 2] * x[i - 2]
        )

    result[len(result) - 1] = (
        u * x[len(result) - 4] * x[0]
        - x[len(result) - 3]
        - f[len(result) - 3] * x[len(result) - 3]
    )
    return result


def visualize_solutions():
    plt.rcParams["text.usetex"] = False
    # Solving the system
    solution = integrate.solve_ivp(
        g, (0, 500), initial, args=(u, alpha, beta, zeta, f, v), dense_output=True
    )
    steady_state = solution.y[:, -1]

    consvquan1 = np.zeros(len(solution.t))
    consvquan2 = np.zeros(len(solution.t))

    for k in range(len(solution.t)):
        consvquan1[k] = solution.y[1, k] + sum(s[1:] * solution.y[2:, k])

    for k in range(len(solution.t)):
        consvquan2[k] = solution.y[0, k] + v * sum(s[1:] * solution.y[2:, k])


    width = 7.5
    height = 8.75
    fig, axes = plt.subplots(1, 2, figsize=(width, 0.33 * height), layout="constrained")
    ax = axes[0]
    # Plot of solutions
    plt.rcParams.update({"font.size": 13})
    for i in range(1, 32):
        ax.plot(solution.t, solution.y[i])
    ax.set_xlabel("time (dimensionless)", size=10)
    ax.set_ylabel("Concentration (dimensionless)", size=10)

    ax.set_title("A", loc="right", size=12, fontweight="bold")
    
    # fission_solution1
    # plt.xlim(0, 500)
    # plt.ylim(0, 0.04)  

    # fission_solution2
    ax.set_xlim(0,350)
    ax.set_ylim(0,0.007)
    ax.text(250,0.0009, '$y(\eta)$', size = 11, color = 'tab:blue')
    ax.text(250, 0.00025, '$x_{26}(\eta)$', size = 11, color = 'tab:pink')
    ax.text(250, 0.0022, '$x_{25}(\eta)$', size = 11, color = 'tab:brown')
    ax.text(237, 0.0035, "$x_1(\eta)$", size=11, color="tab:orange")
    ax.text(285, 0.0035, "-", size=11, color="black")
    ax.text(291, 0.0035, "$x_{24}(\eta)$", size=11, color="tab:purple")

    # # #fission_solution3
    # plt.xlim(0,2500)
    # plt.ylim(0,0.075)
    # plt.text(850,0.066,'Each line represents an oligomer size', size = 11)
    # plt.text(2000,0.012, '$Y(\eta)$', size = 11, color = 'tab:blue')
    # plt.text(2000, 0.004, '$X_{26}(\eta)$', size = 11, color = 'tab:pink')
    # plt.text(2000, 0.033, '$X_{25}(\eta)$', size = 11, color = 'tab:brown')


    # Total Fission Plot
    totalfisrate = np.zeros(len(solution.t))

    totalfisrate = np.sum(f.reshape(-1, 1) * solution.y[2:, :], axis=0)
    ax = axes[1]
    ax.set_title("B", loc = "right", size = 12, fontweight ="bold")
    ax.plot(solution.t, totalfisrate, color="black")
    ax.set_xlabel("time (dimensionless)", size=10)
    ax.set_ylabel("TFR (dimensionless)", size=10)
    ax.text(200, 0.0065, r"$TFR(\eta) = \alpha \sum_{i=26}^{30} x_i(\eta)$", size=11)

    plt.show()
    fig.savefig("dimensionless_sysem.svg")
if __name__ == "__main__":
    visualize_solutions()
    