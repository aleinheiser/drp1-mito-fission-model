# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 11:30:13 2023

@author: alein
"""

import numpy as np
from scipy import integrate
import dimensionless_fission_system as system
from autograd import jacobian

U = np.linspace(180, 1400, num=100, endpoint=True)
F1 = np.zeros((len(U), 2))

alpha = system.alpha
beta = system.beta
zeta = system.zeta

f = system.f
v = system.v
s = np.arange(1, 31)
G = system.initial[0] - system.v * system.initial[1]

# equations for Z and Y from conserved quntities set equal to zero
func1 = lambda X: X[0] - v * X[1] - G
func2 = lambda X: system.initial[1] - X[1] - sum(s[0:] * X[2:])

#  NO FISSION   C_1 = X[2]
func3 = (
    lambda X: beta * X[0] * X[1]
    + (2 * X[3] + sum(X[4:]))
    - zeta * X[2]
    - u * (2 * X[2] ** 2 + sum(X[3:31] * X[2]))
    - f[0] * X[2]
)

func4 = lambda X: u * X[2] * X[2] + X[4] - X[3] - u * X[3] * X[2]  # C_2

func5 = lambda X: u * X[3] * X[2] + X[5] - X[4] - u * X[4] * X[2]

func6 = lambda X: u * X[4] * X[2] + X[6] - X[5] - u * X[5] * X[2]

func7 = lambda X: u * X[5] * X[2] + X[7] - X[6] - u * X[6] * X[2]

func8 = lambda X: u * X[6] * X[2] + X[8] - X[7] - u * X[7] * X[2]

func9 = lambda X: u * X[7] * X[2] + X[9] - X[8] - u * X[8] * X[2]

func10 = lambda X: u * X[8] * X[2] + X[10] - X[9] - u * X[9] * X[2]

func11 = lambda X: u * X[9] * X[2] + X[11] - X[10] - u * X[10] * X[2]

func12 = lambda X: u * X[10] * X[2] + X[12] - X[11] - u * X[11] * X[2]

func13 = lambda X: u * X[11] * X[2] + X[13] - X[12] - u * X[12] * X[2]

func14 = lambda X: u * X[12] * X[2] + X[14] - X[13] - u * X[13] * X[2]

func15 = lambda X: u * X[13] * X[2] + X[15] - X[14] - u * X[14] * X[2]

func16 = lambda X: u * X[14] * X[2] + X[16] - X[15] - u * X[15] * X[2]

func17 = lambda X: u * X[15] * X[2] + X[17] - X[16] - u * X[16] * X[2]

func18 = lambda X: u * X[16] * X[2] + X[18] - X[17] - u * X[17] * X[2]

func19 = lambda X: u * X[17] * X[2] + X[19] - X[18] - u * X[18] * X[2]

func20 = lambda X: u * X[18] * X[2] + X[20] - X[19] - u * X[19] * X[2]

func21 = lambda X: u * X[19] * X[2] + X[21] - X[20] - u * X[20] * X[2]

func22 = lambda X: u * X[20] * X[2] + X[22] - X[21] - u * X[21] * X[2]

func23 = lambda X: u * X[21] * X[2] + X[23] - X[22] - u * X[22] * X[2]

func24 = lambda X: u * X[22] * X[2] + X[24] - X[23] - u * X[23] * X[2]

func25 = lambda X: u * X[23] * X[2] + X[25] - X[24] - u * X[24] * X[2]

func26 = lambda X: u * X[24] * X[2] + X[26] - X[25] - u * X[25] * X[2]

func27 = lambda X: u * X[25] * X[2] + X[27] - X[26] - u * X[26] * X[2]

# WITH FISSION
func28 = lambda X: u * X[26] * X[2] + X[28] - X[27] - u * X[27] * X[2] - alpha * X[27]

func29 = lambda X: u * X[27] * X[2] + X[29] - X[28] - u * X[28] * X[2] - alpha * X[28]

func30 = lambda X: u * X[28] * X[2] + X[30] - X[29] - u * X[29] * X[2] - alpha * X[29]

func31 = lambda X: u * X[29] * X[2] + X[31] - X[30] - u * X[30] * X[2] - alpha * X[30]

# C_30
func32 = lambda X: u * X[30] * X[2] - X[31] - alpha * X[31]

# TODO make less black magic?
func = [globals()[f"func{i}"] for i in range(1, 33)]


# Defining Jacobians of each function
jac_func = [jacobian(f, 0) for f in func]

M = 32
N = 32


steady_states = np.zeros([32, len(U)])
F2 = np.zeros(len(U))
x_new = None
for j in range(len(U)):
    u = U[j]
    solution = integrate.solve_ivp(
        system.g,
        (0, 500),
        system.initial,
        args=(u, system.alpha, system.beta, system.zeta, system.f, system.v),
        dense_output=True,
    )

    # solving for numerical steady state values of TFR
    totalfisrate = np.zeros(len(solution.t))

    totalfisrate = np.sum(system.f.reshape(-1, 1) * solution.y[2:, :], axis=0)
    # reshape function rshapes the data to a different matrix, -1 tells python
    # to i=pick however many rows make sense, and 1 means one column
    # axis=0 tells python to sum the entries of a column, axis=1 means sum the entries of a row
    x = totalfisrate[-len(solution.t) // 2 :]

    F1[j, 0] = np.min(x)
    F1[j, 1] = np.max(x)

    # solving for steady state solutions to define TFR for all values of \mu
    # if j in np.array([0,1,94,95,96,97,98,99]):
    if x_new is None:
        x_new = solution.y[:, -1]

    x0 = x_new
    error = 100
    tol = 1e-11

    while np.any(abs(error) > tol):
        fun_evaluate = np.array([f(x0) for f in func])
        jac1 = np.array([f(x0.flatten()) for f in jac_func])
        jac1 = jac1.reshape(N, M)

        x_new = x0 - np.linalg.inv(jac1) @ fun_evaluate
        error = x_new - x0
        x0 = x_new

    steady_states[:, j] = x_new
    F2[j] = sum(f * x_new[2:])
    print(j)


def plot_bifurcation(ax):
    ax.plot(U, F2, color="tab:red", linestyle="dashed", label="TFR at fixed point")
    ax.plot(U, F1[:, 0], label="minimum TFR from ODE solutions")
    ax.plot(U, F1[:, 1], label="maximum TFR from ODE solutions")
    ax.set_ylabel("TFR (dimensionless)", size = 10)
    ax.legend(fontsize=10)
    ax.set_xlabel(r"$\mu$ (dimensionless) ", size=10)
