"""Portfolio search

Algorithms to search for k-portfolios.
"""

import itertools
from typing import List, Tuple

import mip
import pandas as pd


# Exhaustively search over all k-portfolios and return their objective values.
def exhaustive_search(runtimes: pd.DataFrame, k: int) -> List[Tuple[List[str], float]]:
    return [(list(portfolio), runtimes[list(portfolio)].min(axis='columns').sum())
            for portfolio in itertools.combinations(runtimes.columns, k)]


# Determine optimal k-portfolio by solving an integer problem exactly.
def mip_search(runtimes: pd.DataFrame, k: int) -> List[Tuple[List[str], float]]:
    model = mip.Model()
    model.verbose = 0
    model.threads = 1
    instance_solver_vars = [[model.add_var(f'x_{i}_{j}', var_type=mip.BINARY)
                             for j in range(runtimes.shape[1])] for i in range(runtimes.shape[0])]
    solver_vars = [model.add_var(f'y_{j}', var_type=mip.BINARY)for j in range(runtimes.shape[1])]
    for var_list in instance_solver_vars:  # per-instance constraints
        model.add_constr(mip.xsum(var_list) == 1)
    for j in range(runtimes.shape[1]):  # per-solver-constraints
        model.add_constr(mip.xsum(instance_solver_vars[i][j] for i in range(runtimes.shape[0])) <=
                         runtimes.shape[0] * solver_vars[j])
    model.add_constr(mip.xsum(solver_vars) <= k)  # cardinality constraint
    model.objective = mip.minimize(mip.xsum(instance_solver_vars[i][j] * runtimes.iloc[i, j]
                                            for i in range(runtimes.shape[0]) for j in range(runtimes.shape[1])))
    model.optimize()
    best_solution = [col for var, col in zip(solver_vars, runtimes.columns) if var.x == 1]
    return [(best_solution, model.objective_value)]


# Greedy forward search (starting with portfolio of size 0 and iteratively adding solvers) up
# to k, retaining the w best solutions each iteration. Returns not only best w portfolios for
# final k, but also the intermediate solutions.
def beam_search(runtimes: pd.DataFrame, k: int, w: int) -> List[Tuple[List[str], float]]:
    results = []
    old_portfolios = [([], float('inf'))]
    for _ in range(k):
        new_portfolios = set()  # eliminate duplicates (same portfolio can be created in multiple ways)
        for portfolio, _ in old_portfolios:  # only iterate portfolios, ignore runtimes (for now)
            for solver in runtimes.columns:
                if solver not in portfolio:  # only create portfolios that are larger
                    new_portfolios.add(frozenset(portfolio + [solver]))
        new_portfolios = [(list(new_portfolio), runtimes[new_portfolio].min(axis='columns').sum())
                          for new_portfolio in new_portfolios]
        new_portfolios.sort(key=lambda x: x[1])  # sort by runtime
        old_portfolios = new_portfolios[:w]  # retain w best solutions
        results.extend(old_portfolios)
    return results
