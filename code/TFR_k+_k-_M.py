# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 16:15:38 2021

@author: alein
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import system_stepfunct_volumeterm as system
from util import add_title, integrate_fission


width = 7.5
height = 8.75
fig, axes = plt.subplots(
    3,
    2,
    figsize=(width, 0.85 * height),
    dpi=600,
    layout="constrained",
    gridspec_kw={
        "wspace": 0.0625,
    },
)

################################################################################################
# Plot config
plt.rc("legend", fontsize=8)

p = np.array((1.75, 1.25, 1, 0.75, 0.25))

################################################################################################
# TFR_cumTF_k+
n = [
    r"$+75\%\ $",
    r"$+25\%\ $",
    r"$k_+$",
    r"$-25\%\ $",
    r"$-75\%\ $",
]

ms = [
    r"$+75\%\ k_+$",
    r"$+25\%\ k_+$",
    r"$k_+$",
    r"$-25\%\ k_+$",
    r"$-75\%\ k_+$",
]
k = system.k3 * p


integral = np.zeros(len(p))

ax = axes[0][0]

for k3, m, i in zip(k, ms, range(len(p))):
    solution = integrate.solve_ivp(
        system.g,
        (0, 5000),
        system.initial,
        args=(system.k1, system.k2, k3, system.k4, system.f, system.v),
        dense_output=True,
    )

    totalfisrate = np.zeros_like(solution.t)
    totalfisrate = np.sum(
        system.f.reshape(-1, 1) * solution.y[2:, :],
        axis=0,
    )
    integral[i] = integrate_fission(solution)
    ax.plot(solution.t, totalfisrate, label=f"{m}")
    print("done.")

ax.legend()
add_title(ax, "A")
ax.set_xlabel("time ($s$)", size=12)
ax.set_ylabel("TFR ($nM/s$)", size=12)

ax = axes[0][1]
ax.bar(
    n,
    integral,
    color=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"],
)
add_title(ax, "B")
ax.set_xticks(
    np.arange(len(n)),
    n,
    size=9.5,
)
ax.set_ylabel("Cumulative TF $(nM)$", size=12)

###########################################################################################
# TFR_cumTF_k-
n = [
    r"$+75\%\ $",
    r"$+25\%\ $",
    r"$k_{-}$",
    r"$-25\%\ $",
    r"$-75\%\ $",
]

ms = [
    r"$+75\%\ k_{-}$",
    r"$+25\%\ k_{-}$",
    r"$k_{-}$",
    r"$-25\%\ k_{-}$",
    r"$-75\%\ k_{-}$",
]
k = system.k4 * p

integral = np.zeros(len(p))


for k4, m, i in zip(k, ms, range(len(p))):
    solution = integrate.solve_ivp(
        system.g,
        (0, 5000),
        system.initial,
        args=(system.k1, system.k2, system.k3, k4, system.f, system.v),
        dense_output=True,
    )

    totalfisrate = np.zeros(len(solution.t))

    totalfisrate = np.sum(system.f.reshape(-1, 1) * solution.y[2:, :], axis=0)

    integral[i] = integrate_fission(solution)
    ax = axes[1][0]
    ax.plot(solution.t, totalfisrate, label=f"{m}")
    print("done.")


ax.legend()
add_title(ax, "C")
ax.set_xlabel("time ($s$)", size=12)
ax.set_ylabel("TFR ($nM/s$)", size=12)


ax = axes[1][1]
ax.bar(
    n, integral, color=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
)
ax.set_xticks(np.arange(len(n)), n, size=9.5)
add_title(ax, "D")
ax.set_ylabel("Cumulative TF $(nM)$", size=12)

################################################################################################
# TFR_cumTF_M
n = [
    r"$+75\%\ $",
    r"$+25\%\ $",
    r"$\mathcal{M}$",
    r"$-25\%\ $",
    r"$-75\%\ $",
]

ms = [
    r"$+75\%\ \mathcal{M}$",
    r"$+25\%\ \mathcal{M}$",
    r"$\mathcal{M}$",
    r"$-25\%\ \mathcal{M}$",
    r"$-75\%\ \mathcal{M}$",
]

initial = np.zeros((len(p), 32))
initial[:, 1] = system.initial[1] * p
initial[:, 0] = system.initial[0]

integral = np.zeros(len(p))

# cumulative TFR

for i, m in zip(range(len(p)), ms):
    solution = integrate.solve_ivp(
        system.g,
        (0, 5000),
        initial[i, :],
        args=(system.k1, system.k2, system.k3, system.k4, system.f, system.v),
        dense_output=True,
    )

    totalfisrate = np.zeros(len(solution.t))
    totalfisrate = np.sum(system.f.reshape(-1, 1) * solution.y[2:, :], axis=0)

    integral[i] = integrate_fission(solution)

    ax = axes[2][0]
    ax.plot(solution.t, totalfisrate, label=f"{m}")

    print("done.")

ax.legend()
add_title(ax, "E")
ax.set_xlabel("time ($s$)", size=12)
ax.set_ylabel("TFR ($nM/s$)", size=12)

# bar plot of cumulative TFR
ax = axes[2][1]
ax.bar(
    n,
    integral,
    color=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"],
)
ax.set_xticks(
    np.arange(len(n)),
    n,
    size=9.5,
)
add_title(ax, "F")
ax.set_ylabel("Cumulative TF $(nM)$", size=12)
fig.savefig("Fig6.svg")
