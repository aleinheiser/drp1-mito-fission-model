# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:22:58 2021

@author: alein
"""
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import system_stepfunct_volumeterm as system

width = 7.5
height = 8.75
fig, axes = plt.subplots(
    2,
    2,
    figsize=(width, 0.65 * height),
    layout="constrained",
    gridspec_kw={
        "wspace": 0.0625,
    },
)
p = np.array((1.75, 1.25, 1, 0.75, 0.25))

########################################################################################
# TFR_k1
ms = [r"+75\%\ k_1", r"+25\%\ k_1", "k_1", r"-25\%\ k_1", r"-75\%\ k_1"]
k = system.k1 * p

ax = axes[0][0]
for k1, m in zip(k, ms):
    solution = integrate.solve_ivp(
        system.g,
        (0, 5000),
        system.initial,
        args=(k1, system.k2, system.k3, system.k4, system.f, system.v),
        dense_output=True,
    )

    totalfisrate = np.zeros(len(solution.t))

    totalfisrate = np.sum(system.f.reshape(-1, 1) * solution.y[2:, :], axis=0)
    ax.plot(solution.t, totalfisrate, label=f"${m}$")

    print("done.")

plt.rc("legend", fontsize=8)
ax.legend()
ax.set_title("A", loc="right", size=12, fontweight="bold")
ax.set_xlabel("time ($s$)", size=12)
ax.set_ylabel("TFR ($nM/s$)", size=12)


#######################################################################################################
# TFR_k2
ms = ["+75\%\ k_{-1}", "+25\%\ k_{-1}", "k_{-1}", "-25\%\ k_{-1}", "-75\%\ k_{-1}"]
k = system.k2 * p

for k2, m in zip(k, ms):
    solution = integrate.solve_ivp(
        system.g,
        (0, 5000),
        system.initial,
        args=(system.k1, k2, system.k3, system.k4, system.f, system.v),
        dense_output=True,
    )

    totalfisrate = np.zeros(len(solution.t))

    totalfisrate = np.sum(system.f.reshape(-1, 1) * solution.y[2:, :], axis=0)
    ax = axes[0][1]
    ax.plot(solution.t, totalfisrate, label=f"${m}$")

    print("done.")

ax.legend()
ax.set_title("B", loc="right", size=12, fontweight="bold")
ax.set_xlabel("time ($s$)", size=12)
ax.set_ylabel("TFR ($nM/s$)", size=12)


####################################################################################################
# TFR_a
ms = [r"+75\%\ a", r"+25\%\ a", "a", r"-25\%\ a", r"-75\%\ a"]

f = np.zeros((len(p), 30))
for i in range(0, system.b - 1):
    f[:, i] = 0


A = np.zeros(len(p))
for n in range(len(p)):
    A[n] = system.a * p[n]
    f[n, system.b :] = A[n]


# system of ODEs
for k, m in zip(range(len(p)), ms):
    solution = integrate.solve_ivp(
        system.g,
        (0, 5000),
        system.initial,
        args=(system.k1, system.k2, system.k3, system.k4, f[k, :], system.v),
        dense_output=True,
    )
    totalfisrate = np.zeros(len(solution.t))
    for j in range(len(solution.t)):
        t = solution.t[j]
        totalfisrate[j] = sum(f[k, :] * solution.sol(t)[2:])
    ax = axes[1][1]
    ax.plot(solution.t, totalfisrate, label=f"${m}$")

ax.legend()
ax.set_title("D", loc="right", size=12, fontweight="bold")
ax.set_xlabel("time ($s$)", size=12)
ax.set_ylabel("TFR ($nM/s$)", size=12)


################################################################################################
# TFR_T
ms = [
    "+75\%\ \mathcal{T}",
    "+25\%\ \mathcal{T}",
    "\mathcal{T}",
    "-25\%\ \mathcal{T}",
    "-75\%\ \mathcal{T}",
]

initial = np.zeros((len(p), 32))

A = np.zeros(len(p))
for n in range(len(p)):
    A[n] = system.initial[0] * p[n]
    initial[n, 0] = A[n]
    initial[n, 1] = system.initial[1]


for k, m in zip(range(len(p)), ms):
    solution = integrate.solve_ivp(
        system.g,
        (0, 5000),
        initial[k, :],
        args=(system.k1, system.k2, system.k3, system.k4, system.f, system.v),
        dense_output=True,
    )

    totalfisrate = np.zeros(len(solution.t))

    totalfisrate = np.sum(system.f.reshape(-1, 1) * solution.y[2:, :], axis=0)
    ax = axes[1][0]
    ax.plot(solution.t, totalfisrate, label=f"${m}$")

    print("done.")

ax.legend()
ax.set_title("C", loc="right", size=12, fontweight="bold")
ax.set_xlabel("time ($s$)", size=12)
ax.set_ylabel("TFR ($nM/s$)", size=12)

fig.savefig("Fig5.svg")
