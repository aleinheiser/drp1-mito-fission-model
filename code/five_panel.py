import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dimlessTFR_u import plot_tfr, plot_cum_tf
from dimlessTFR_alpha_beta_zeta import plot_alpha, plot_beta, plot_zeta
from util import add_title

width = 7.5
height = 8.75

fig = plt.figure(
    figsize=(width, 0.7 * height),
    layout="constrained",
)

gs = GridSpec(2, 6, figure=fig)

ax1 = fig.add_subplot(gs[0, 0:3])
plot_tfr(ax1)
add_title(ax1, "A")

ax2 = fig.add_subplot(gs[0, 3:])
plot_cum_tf(ax2)
add_title(ax2, "B")

ax3 = fig.add_subplot(gs[1, 0:2])
plot_alpha(ax3)
add_title(ax3, "C")

ax4 = fig.add_subplot(gs[1, 2:4])
plot_beta(ax4)
add_title(ax4, "D")

ax5 = fig.add_subplot(gs[1, 4:])
plot_zeta(ax5)
add_title(ax5, "E")

for ax in [ax1, ax3, ax4, ax5]:
    ax.legend(loc="upper right", fontsize=8)

fig.savefig("Fig9.svg")
