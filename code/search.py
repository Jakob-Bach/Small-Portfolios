"""Portfolio search

Algorithms to search for k-portfolios. All algorithms return a list of portfolios and objective
values.
"""

import itertools
from typing import List, Tuple
import random

import mip
import pandas as pd
import z3


# Exhaustively search over all k-portfolios and return portfolios with their objective values.
def exhaustive_search(runtimes: pd.DataFrame, k: int) -> List[Tuple[List[str], float]]:
    return [(list(portfolio), runtimes[list(portfolio)].min(axis='columns').mean())
            for portfolio in itertools.combinations(runtimes.columns, k)]


# Randomly sample k-portfolios. Repeat w times.
def random_search(runtimes: pd.DataFrame, k: int, w: int) -> List[Tuple[List[str], float]]:
    rng = random.Random(25)
    results = []
    for _ in range(w):
        portfolio = rng.sample(list(runtimes.columns), k=k)
        results.append((portfolio, runtimes[portfolio].min(axis='columns').mean()))
    return results


# Determine optimal k-portfolio by solving an integer problem exactly.
def mip_search(runtimes: pd.DataFrame, k: int) -> List[Tuple[List[str], float]]:
    model = mip.Model()
    model.verbose = 0
    model.threads = 1
    model.max_mip_gap = 0  # without this, solutions might be slightly sub-optimal
    instance_solver_vars = [[model.add_var(f'x_{i}_{j}', var_type=mip.BINARY)
                             for j in range(runtimes.shape[1])] for i in range(runtimes.shape[0])]
    solver_vars = [model.add_var(f'y_{j}', var_type=mip.BINARY)for j in range(runtimes.shape[1])]
    for var_list in instance_solver_vars:  # per-instance constraints
        model.add_constr(mip.xsum(var_list) == 1)
    for j in range(runtimes.shape[1]):  # per-solver-constraints
        model.add_constr(mip.xsum(instance_solver_vars[i][j] for i in range(runtimes.shape[0])) <=
                         runtimes.shape[0] * solver_vars[j])
    model.add_constr(mip.xsum(solver_vars) <= k)  # cardinality constraint
    model.objective = mip.minimize(
        mip.xsum(instance_solver_vars[i][j] * runtimes.iloc[i, j]
                 for i in range(runtimes.shape[0]) for j in range(runtimes.shape[1])) / runtimes.shape[0])
    model.optimize()
    best_solution = [col for var, col in zip(solver_vars, runtimes.columns) if var.x == 1]
    return [(best_solution, model.objective_value)]


# Determine optimal k-portfolio by solving an SMT problem exactly.
# Formulation inspired (though simplified here) by Nof, Y., & Strichman, O. (2020). Real-time
# solving of computationally hard problems using optimal algorithm portfolios.
def smt_search(runtimes: pd.DataFrame, k: int) -> List[Tuple[List[str], float]]:
    instance_vars = [z3.Real(f'v_{i}') for i in range(runtimes.shape[0])]
    solver_vars = [z3.Bool(f's_{j}') for j in range(runtimes.shape[1])]
    value_constraints = [z3.Or([z3.And(instance_vars[i] == runtimes.iloc[i, j], solver_vars[j])
                                for j in range(runtimes.shape[1])]) for i in range(runtimes.shape[0])]
    optimizer = z3.Optimize()
    objective = optimizer.minimize(z3.Sum(instance_vars) / runtimes.shape[0])
    optimizer.add(value_constraints)
    optimizer.add(z3.AtMost(*solver_vars, k))
    optimizer.check()
    best_solution = [col for var, col in zip(solver_vars, runtimes.columns) if bool(optimizer.model()[var])]
    if objective.value().is_real():  # Z3 uses representation as fraction
        objective_value = objective.value().numerator_as_long() / objective.value().denominator_as_long()
    else:
        objective_value = objective.value().as_long()
    return [(best_solution, objective_value)]


# Greedy forward search (starting with portfolio of size 0 and iteratively adding solvers) up
# to k, retaining the w best solutions each iteration. Returns not only best w portfolios for
# final k, but also the intermediate solutions.
def beam_search(runtimes: pd.DataFrame, k: int, w: int) -> List[Tuple[List[str], float]]:
    results = []
    old_portfolios = [([], float('inf'))]
    for _ in range(k):
        new_portfolios = []
        for portfolio, _ in old_portfolios:  # only iterate portfolios, ignore runtimes (for now)
            for solver in runtimes.columns:
                if solver not in portfolio:  # only create portfolios that are larger
                    # Same portfolio can be created in multiple ways, so we need to de-duplicate;
                    # set/frozenset are non-deterministic, so we use lists for both single
                    # portfolios and the list of all portfolios:
                    new_portfolio = sorted(portfolio + [solver])
                    if new_portfolio not in new_portfolios:
                        new_portfolios.append(new_portfolio)
        new_portfolios = [(new_portfolio, runtimes[new_portfolio].min(axis='columns').mean())
                          for new_portfolio in new_portfolios]
        new_portfolios.sort(key=lambda x: x[1])  # sort by runtime
        old_portfolios = new_portfolios[:w]  # retain w best solutions
        results.extend(old_portfolios)
    return results


# Rank solvers by individual performance and create k-portfolio by taking top k solvers from this
# list.
# Idea from Nof, Y., & Strichman, O. (2020). Real-time solving of computationally hard problems
# using optimal algorithm portfolios.
# Implementation here is similar to our beam-search implementation, i.e., not only k-portfolios
# are returned, but also all smaller portfolios (so solver ranking has only to be done once,
# though effort for this is neglectable anyway).
def kbest_search(runtimes: pd.DataFrame, k: int) -> List[Tuple[List[str], float]]:
    results = []
    sorted_solvers = runtimes.mean().sort_values().index.to_list()
    for i in range(1, k + 1):
        results.append((sorted_solvers[:i], runtimes[sorted_solvers[:i]].min(axis='columns').mean()))
    return results
