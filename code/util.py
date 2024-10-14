from functools import cache
import os
from pathlib import Path

import numpy as np
from scipy import integrate
import dimensionless_fission_system as system


def integrate_fission(solution):
    """Given a solution find the total fission"""
    result = 0
    total_fission = lambda t: np.sum(
        system.f[0:].reshape(-1, 1) * solution.sol(t)[2:], axis=0
    )

    for a, b in zip(solution.t, solution.t[1:]):
        result += integrate.fixed_quad(
            total_fission,
            a=a,
            b=b,
            n=4,
        )[0]

    return result


def add_title(ax, title):
    """Add a title to a subplot"""
    ax.set_title(title, loc="right", size=12, fontweight="bold")


@cache
def base():
    """Find the base directory for data

    Where to look for data, if the user has specified the "DATAPATH" environment
    variable, look there. Otherwise we try to load `../data`.
    If that fails we look in the current directory.
    """
    data_path = os.environ.get("DATAPATH")

    if data_path is not None:
        return Path(data_path)

    try:
        root = Path(__file__).parent
    except NameError:
        root = Path(".")

    data_path = root.parent / "data"

    if data_path.is_dir():
        return data_path

    return Path("./")


