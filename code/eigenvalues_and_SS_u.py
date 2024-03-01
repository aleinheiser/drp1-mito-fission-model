# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:21:30 2023

@author: alein
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import dimensionless_fission_system as system
from util import base

from autograd import jacobian

p = np.array((1, 0.25, 0.75, 1.25, 1.75))
U = system.u * p
alpha = system.alpha
beta = system.beta
zeta = system.zeta

f = system.f
v = system.v
s = np.arange(1, 31)
G = system.initial[0] - system.v * system.initial[1]

# ODEs for Z and Y   (T and M)
func1_1 = (
    lambda X: v * zeta * X[2] + v * sum(X[2:] * f[0:] * s[0:]) - v * beta * X[0] * X[1]
)
func2_2 = lambda X: zeta * X[2] + sum(X[2:] * f[0:] * s[0:]) - beta * X[0] * X[1]


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

# Defining Jacobians of each function
jac_func1_1 = jacobian(func1_1, 0)
jac_func2_2 = jacobian(func2_2, 0)

jac_func1 = jacobian(func1, 0)
jac_func2 = jacobian(func2, 0)
jac_func3 = jacobian(func3, 0)
jac_func4 = jacobian(func4, 0)
jac_func5 = jacobian(func5, 0)
jac_func6 = jacobian(func6, 0)
jac_func7 = jacobian(func7, 0)
jac_func8 = jacobian(func8, 0)
jac_func9 = jacobian(func9, 0)
jac_func10 = jacobian(func10, 0)
jac_func11 = jacobian(func11, 0)
jac_func12 = jacobian(func12, 0)
jac_func13 = jacobian(func13, 0)
jac_func14 = jacobian(func14, 0)
jac_func15 = jacobian(func15, 0)
jac_func16 = jacobian(func16, 0)
jac_func17 = jacobian(func17, 0)
jac_func18 = jacobian(func18, 0)
jac_func19 = jacobian(func19, 0)
jac_func20 = jacobian(func20, 0)
jac_func21 = jacobian(func21, 0)
jac_func22 = jacobian(func22, 0)
jac_func23 = jacobian(func23, 0)
jac_func24 = jacobian(func24, 0)
jac_func25 = jacobian(func25, 0)
jac_func26 = jacobian(func26, 0)
jac_func27 = jacobian(func27, 0)
jac_func28 = jacobian(func28, 0)
jac_func29 = jacobian(func29, 0)
jac_func30 = jacobian(func30, 0)
jac_func31 = jacobian(func31, 0)
jac_func32 = jacobian(func32, 0)

M = 32
N = 32

eigen_file = base() / "eigen_values.txt"

print(f"looking for {eigen_file=}")
if eigen_file.exists():
    eigenvalues = np.loadtxt(eigen_file, dtype=complex)
else:
    steady_states = np.zeros([32, len(U)])
    for j in range(len(U)):
        u = U[j]
        solution = integrate.solve_ivp(
            system.g,
            (0, 500),
            system.initial,
            args=(u, system.alpha, system.beta, system.zeta, system.f, system.v),
            method="RK45",
        )

        print("solution done")
        if j in np.array([0, 1, 2]):
            x0 = solution.y[:, -1]
        else:
            x0 = x_new
        error = 100
        tol = 1e-11

        while np.any(abs(error) > tol):
            fun_evaluate = np.array(
                [
                    func1(x0),
                    func2(x0),
                    func3(x0),
                    func4(x0),
                    func5(x0),
                    func6(x0),
                    func7(x0),
                    func8(x0),
                    func9(x0),
                    func10(x0),
                    func11(x0),
                    func12(x0),
                    func13(x0),
                    func14(x0),
                    func15(x0),
                    func16(x0),
                    func17(x0),
                    func18(x0),
                    func19(x0),
                    func20(x0),
                    func21(x0),
                    func22(x0),
                    func23(x0),
                    func24(x0),
                    func25(x0),
                    func26(x0),
                    func27(x0),
                    func28(x0),
                    func29(x0),
                    func30(x0),
                    func31(x0),
                    func32(x0),
                ]
            )

            flat_x0 = x0.flatten()
            jac1 = np.array(
                [
                    jac_func1(flat_x0),
                    jac_func2(flat_x0),
                    jac_func3(flat_x0),
                    jac_func4(flat_x0),
                    jac_func5(flat_x0),
                    jac_func6(flat_x0),
                    jac_func7(flat_x0),
                    jac_func8(flat_x0),
                    jac_func9(flat_x0),
                    jac_func10(flat_x0),
                    jac_func11(flat_x0),
                    jac_func12(flat_x0),
                    jac_func13(flat_x0),
                    jac_func14(flat_x0),
                    jac_func15(flat_x0),
                    jac_func16(flat_x0),
                    jac_func17(flat_x0),
                    jac_func18(flat_x0),
                    jac_func19(flat_x0),
                    jac_func20(flat_x0),
                    jac_func21(flat_x0),
                    jac_func22(flat_x0),
                    jac_func23(flat_x0),
                    jac_func24(flat_x0),
                    jac_func25(flat_x0),
                    jac_func26(flat_x0),
                    jac_func27(flat_x0),
                    jac_func28(flat_x0),
                    jac_func29(flat_x0),
                    jac_func30(flat_x0),
                    jac_func31(flat_x0),
                    jac_func32(flat_x0),
                ]
            )
            jac1 = jac1.reshape(N, M)

            x_new = x0 - np.linalg.inv(jac1) @ fun_evaluate
            error = x_new - x0
            x0 = x_new

        steady_states[:, j] = x_new
        print("done")

    eigenvalues = np.zeros([32, len(U)], dtype=complex)
    for i in range(len(U)):
        u = U[i]
        x = steady_states[:, i]
        jac2 = np.array(
            [
                jac_func1_1(x),
                jac_func2_2(x),
                jac_func3(x),
                jac_func4(x),
                jac_func5(x),
                jac_func6(x),
                jac_func7(x),
                jac_func8(x),
                jac_func9(x),
                jac_func10(x),
                jac_func11(x),
                jac_func12(x),
                jac_func13(x),
                jac_func14(x),
                jac_func15(x),
                jac_func16(x),
                jac_func17(x),
                jac_func18(x),
                jac_func19(x),
                jac_func20(x),
                jac_func21(x),
                jac_func22(x),
                jac_func23(x),
                jac_func24(x),
                jac_func25(x),
                jac_func26(x),
                jac_func27(x),
                jac_func28(x),
                jac_func29(x),
                jac_func30(x),
                jac_func31(x),
                jac_func32(x),
            ]
        )
        jac2 = jac2.reshape(M, N)
        eigvals = np.linalg.eig(jac2)
        eigenvalues[:, i] = eigvals[0]
        print("done")

    np.savetxt(eigen_file, eigenvalues)

plotting_eigenvalues = eigenvalues[26:30, :]


def plot_eigenvalues(ax, dot_size):
    # Move the left and bottom spines to x = 0 and y = 0, respectively.
    ax.spines[["left", "bottom"]].set_position(("data", 0))
    # Hide the top and right spines.
    ax.spines[["top", "right"]].set_visible(False)

    # Draw arrows (as black triangles: ">k"/"^k") at the end of the axes.  In each
    # case, one of the coordinates (0) is a data coordinate (i.e., y = 0 or x = 0,
    # respectively) and the other one (1) is an axes coordinate (i.e., at the very
    # right/top of the axes).  Also, disable clipping (clip_on=False) as the marker
    # actually spills out of the axes.
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    ax.scatter(
        np.real(plotting_eigenvalues[:, 1]),
        np.imag(plotting_eigenvalues[:, 1]),
        s=dot_size,
        color="tab:purple",
        label=r"$-75\% \mu$",
    )
    ax.scatter(
        np.real(plotting_eigenvalues[:, 2]),
        np.imag(plotting_eigenvalues[:, 2]),
        s=dot_size,
        color="tab:red",
        label=r"$-25\%\mu$",
    )
    ax.scatter(
        np.real(plotting_eigenvalues[:, 0]),
        np.imag(plotting_eigenvalues[:, 0]),
        s=dot_size,
        color="tab:green",
        label=r"$\mu$",
    )
    ax.scatter(
        np.real(plotting_eigenvalues[:, 3]),
        np.imag(plotting_eigenvalues[:, 3]),
        s=dot_size,
        color="tab:orange",
        label=r"$+25\% \mu$",
    )
    ax.scatter(
        np.real(plotting_eigenvalues[:, 4]),
        np.imag(plotting_eigenvalues[:, 4]),
        s=dot_size,
        color="tab:blue",
        label=r"$+75\%\mu$",
    )
    ax.legend(
        loc="lower right",
        ncols=2,
        fontsize=8.5,
    )

    ax.text(
        0.03,
        0.05,
        "Real",
        size=11,
    )

    ax.text(
        -0.135,
        1.7,
        "Imaginary",
        size=11,
    )


if __name__ == "__main__":
    width = 7.5
    height = 8.75
    fig, axes = plt.subplots(
        1, 1, figsize=(width, 0.4 * height), dpi=600, layout="constrained"
    )
    plot_eigenvalues(axes, dot_size=20)
    fig.savefig("eigen")
