""
import os
import pickle

import numpy as np
import SALib as SALib
from joblib import Memory, Parallel, delayed, parallel_config
from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample
from scipy import integrate
from tqdm import tqdm

import dimensionless_fission_system as system

# Where to save data. If the environment has a variable called scratch it will
# save there, else it will create a folder called ./data and save there
cache_dir = os.environ.get("scratch", "./data")
memory = Memory(cache_dir, verbose=0)

# This will read how many cores you have to play with
slots = int(os.environ.get("NSLOTS", 1))

problem = {
    "num_vars": 4,
    "names": ["u", "alpha", "beta", "zeta"],
    "bounds": [
        [0, 1500],
        [0, 100],
        [0, 300],
        [0, 1],
    ],
    "dist": ["unif", "unif", "unif", "unif"],
}


# This what is run when you call python scriptname.py
def main():
    n = 2**11

    seed = int(os.getenv("SEED", 800))
    print(f"Running dimensionless global sensitivity analysis with {seed=}")

    with parallel_config(n_jobs=slots):
        res = run(seed, n)
        with open(f"{cache_dir}/result_{seed}_2^11_dimless", "wb") as f:
            pickle.dump(res, f)


def run(seed, n):
    # The Saltelli sampler generates N*(2D+2) samples, where in this example N is 1024 and D is number of variables
    samples = sample(problem, n, seed=seed)  # makes a #x4 matrix
    np.savetxt(f"{cache_dir}/input_{seed}_2^11_dimless", samples)
    rows = len(samples)

    # The results are a parallel generator
    results = Parallel(return_as="generator")(
        delayed(run_with_params)(params) for params in samples
    )

    # Tqdm will print the progress as this generator is exhausted
    results = tqdm(results, total=rows)

    # Collect the results into a numpy array
    Y = np.fromiter(
        results,
        dtype=float,
        count=rows,
    )
    np.savetxt(f"{cache_dir}/output_{seed}_2^11_dimless", Y)

    print("Computing sobol indices")
    Si = analyze(problem, Y, seed=seed)
    summarize(Si)

    return Si


# This function is slow, cache its results
@memory.cache
def run_with_params(params):
    """Run the system with specified parameters

    Returns the value after integrating for total fission
    """
    initial = np.zeros(32)
    initial[0] = system.initial[0]
    initial[1] = system.initial[1]

    solution = integrate.solve_ivp(
        system.g,
        (0, 500),
        initial,
        args=(
            params[0],
            params[1],
            params[2],
            params[3],
            system.f,
            system.v,
        ),
        dense_output=True,
    )

    return integrate_fission(solution)


def integrate_fission(solution):
    """Given a solution integrate for total fission"""
    last = 0
    result = 0

    # The function to integrate
    def integrand(t):
        return np.sum(system.f[0:].reshape(-1, 1) * solution.sol(t)[2:], axis=0)

    for t in solution.t:
        result += integrate.fixed_quad(
            integrand,
            a=last,
            b=t,
            n=4,
        )[0]
        last = t

    return result


def summarize(si):
    """Print a pretty summary of the sobol indices"""
    keys = ["S1", "ST", "S2"]
    for key in keys:
        print(f"{key: ^26}", "=" * 26, sep="\n")
        val = si[key]
        pm = si[f"{key}_conf"]
        n = len(val)
        if key == "S2":
            for i in range(n):
                for j in range(i + 1, n):
                    name_1 = problem["names"][i]
                    name_2 = problem["names"][j]
                    print(
                        f"({name_1:>2}, {name_2:>2}) = {val[i,j]:.4f} ± {pm[i,j]:.4f}"
                    )
        else:
            for i in range(n):
                print(f"{problem['names'][i]:>2} = {val[i]:.4f} ± {pm[i]:.4f}")


# This is what runs the main function
if __name__ == "__main__":
    main()
