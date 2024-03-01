from bifurcation_u import plot_bifurcation
from eigenvalues_and_SS_u import plot_eigenvalues
import matplotlib.pyplot as plt

width = 7.5
height = 8.75
fig, axes = plt.subplots(
    1,
    2,
    figsize=(width, width / 2),
    dpi=600,
    layout="constrained",
)

print("Plotting eigen values")
plot_eigenvalues(axes[0], dot_size=25)
axes[0].set_title("A", loc="right", size=12, fontweight="bold")

print("Plotting bifurcation")
plot_bifurcation(axes[1])
axes[1].set_title("B", loc="right", size=12, fontweight="bold")
fig.savefig("Fig10.svg")
