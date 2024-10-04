"""K-portfolio search

Methods to search for k-portfolios. The search methods do not have a common superclass or some
other mechanism to enforce compatibility but implicitly share the first two parameters, i.e.,
a data frame of runtimes (rows represent problem instances, columns represent solvers) and the
portfolio size `k`. The existence of further parameters depends on the search method.
All search methods return a list, where each entry is a portfolio described by the solver names
and objective value (average runtime of virtual best solver).

Literature
----------
Bach, Jakob, Markus Iser, and Klemens Böhm (2022). "A Comprehensive Study of k-Portfolios of
Recent SAT Solvers"
"""

import itertools
from typing import List, Tuple
import random

import mip
import pandas as pd
import z3


def exhaustive_search(runtimes: pd.DataFrame, k: int) -> List[Tuple[List[str], float]]:
    """Exhaustive search for k-portfolios

    Lists all solver portfolios of size `k` (without filtering or sorting by objective value).
    Apart from very small `k`, may take a long time and return a huge results object.

    Parameters
    ----------
    runtimes : pd.DataFrame
        Solver runtimes (rows represent problem instances, columns represent solvers).
    k : int
        The number of solvers in the portfolio.

    Returns
    -------
    (List[Tuple[List[str], float]])
        List of portfolios, with each entry consisting of (1) the solver (= column) names and
        (2) the objective value (average runtime of the virtual best solver from the portfolio).
    """

    return [(list(portfolio), runtimes[list(portfolio)].min(axis='columns').mean())
            for portfolio in itertools.combinations(runtimes.columns, k)]


def random_search(runtimes: pd.DataFrame, k: int, w: int) -> List[Tuple[List[str], float]]:
    """Random search for k-portfolios

    Randomly samples a fixed number of portfolios and lists all of them (i.e., does not filter or
    sort by objective value). The overall sampling procedure is with replacement (e.g., the same
    portfolio may be returned multiple times), while the solvers in each portfolio are sampled
    without replacement (i.e., each portfolio contains exactly `k` distinct solvers).

    Parameters
    ----------
    runtimes : pd.DataFrame
        Solver runtimes (rows represent problem instances, columns represent solvers).
    k : int
        The number of solvers in the portfolio.
    w : int
        Number of repetitions, i.e., number of portfolios returned.

    Returns
    -------
    (List[Tuple[List[str], float]])
        List of portfolios (length = `w`), with each entry consisting of (1) the solver (= column)
        names and (2) the objective value (average runtime of the virtual best solver from the
        portfolio).
    """

    rng = random.Random(25)
    results = []
    for _ in range(w):
        portfolio = rng.sample(list(runtimes.columns), k=k)
        results.append((portfolio, runtimes[portfolio].min(axis='columns').mean()))
    return results


def mip_search(runtimes: pd.DataFrame, k: int) -> List[Tuple[List[str], float]]:
    """K-portfolio search with a MIP-Solver

    Formulates the k-portfolio problem as a Mixed Integer Programming (MIP) problem and solves it
    exactly with a solver. Returns one portfolio, corresponding to the global optimum. The chosen
    formulation is a contribution of the paper Bach, Jakob, Markus Iser, and Klemens Böhm (2022).
    "A Comprehensive Study of k-Portfolios of Recent SAT Solvers".

    Parameters
    ----------
    runtimes : pd.DataFrame
        Solver runtimes (rows represent problem instances, columns represent solvers).
    k : int
        The number of solvers in the portfolio.

    Returns
    -------
    (List[Tuple[List[str], float]])
        List of portfolios (length = 1), with its single entry consisting of (1) the solver
        (= column) names and (2) the objective value (average runtime of the virtual best solver
        from the portfolio).
    """

    model = mip.Model()
    model.verbose = 0
    model.threads = 1
    model.max_mip_gap = 0  # without this, solutions might be slightly sub-optimal
    instance_solver_vars = [[model.add_var(f'x_{i}_{j}', var_type=mip.BINARY)
                             for j in range(runtimes.shape[1])] for i in range(runtimes.shape[0])]
    solver_vars = [model.add_var(f'y_{j}', var_type=mip.BINARY) for j in range(runtimes.shape[1])]
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


def smt_search(runtimes: pd.DataFrame, k: int) -> List[Tuple[List[str], float]]:
    """K-portfolio search with an SMT solver

    Formulates the k-portfolio problem as a Satisfiability Modulo Theories (SMT) problem and solves
    it exactly with a solver. Returns one portfolio, corresponding to the global optimum. The
    chosen formulation is novel and simplifies the model of Nof, Yair, & Strichman, Ofer (2020).
    "Real-time solving of computationally hard problems using optimal algorithm portfolios".

    Parameters
    ----------
    runtimes : pd.DataFrame
        Solver runtimes (rows represent problem instances, columns represent solvers).
    k : int
        The number of solvers in the portfolio.

    Returns
    -------
    (List[Tuple[List[str], float]])
        List of portfolios (length = 1), with its single entry consisting of (1) the solver
        (= column) names and (2) the objective value (average runtime of the virtual best solver
        from the portfolio).
    """

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


def smt_search_nof(runtimes: pd.DataFrame, k: int, cardinality_encoding: bool = True) -> List[Tuple[List[str], float]]:
    """K-portfolio search with an SMT solver

    Formulates the k-portfolio problem as a Satisfiability Modulo Theories (SMT) problem and solves
    it exactly with a solver. Returns one portfolio, corresponding to the global optimum. The
    chosen SMT formulation is a contribution of the paper Nof, Yair, & Strichman, Ofer (2020).
    "Real-time solving of computationally hard problems using optimal algorithm portfolios" and the
    corresponding tool "nchoosek" (https://doi.org/10.5281/zenodo.3841422).

    Parameters
    ----------
    runtimes : pd.DataFrame
        Solver runtimes (rows represent problem instances, columns represent solvers).
    k : int
        The number of solvers in the portfolio.
    cardinality_encoding : bool, optional
        Encode the cardinality constraint as in "nchoosek", i.e., with the sequential counter from
        Sinz, Carsten (2005). "Towards an optimal CNF encoding of boolean cardinality constraints".
        Otherwise, use the native `AtMost` constraint from the solver `Z3`, which may be faster.
        The default is True.

    Returns
    -------
    (List[Tuple[List[str], float]])
        List of portfolios (length = 1), with its single entry consisting of (1) the solver
        (= column) names and (2) the objective value (average runtime of the virtual best solver
        from the portfolio).
    """

    instance_vars = [z3.Real(f'v{i}') for i in range(runtimes.shape[0])]
    solver_vars = [z3.Bool(f'e{j}') for j in range(runtimes.shape[1])]
    optimizer = z3.Optimize()
    objective = optimizer.minimize(z3.Sum(instance_vars) / runtimes.shape[0])  # just sum in "nchoosek"
    constraints = []
    for i in range(runtimes.shape[0]):
        unique_runtimes = runtimes.iloc[i].unique()
        # Value-choice constraint:
        constraints.append(z3.Or([instance_vars[i] == value for value in unique_runtimes]))
        # Implied-algorithm constraints:
        for value in unique_runtimes:
            affected_solvers = [solver for solver, runtime in zip(solver_vars, runtimes.iloc[i])
                                if runtime == value]
            constraints.append(z3.Implies(instance_vars[i] == value, z3.Or(affected_solvers)))
    optimizer.add(z3.And(constraints))
    if cardinality_encoding:
        # Cardinality constraint with cardinality encoding from Sinz, Carsten (2005). "Towards an
        # optimal CNF encoding of boolean cardinality constraints" (see page 2, Sequential Counter)
        constraints = []
        solver_choice_vars = [[z3.Bool(f's{j+1}_{s+1}') for s in range(k)]
                              for j in range(runtimes.shape[1])]
        constraints.append(z3.Or(z3.Not(solver_vars[0]), solver_choice_vars[0][0]))
        for s in range(1, k):
            constraints.append(z3.Not(solver_choice_vars[0][s]))
        for j in range(1, runtimes.shape[1] - 1):
            constraints.append(z3.Or(z3.Not(solver_vars[j]), solver_choice_vars[j][0]))
            constraints.append(z3.Or(z3.Not(solver_choice_vars[j-1][0]), solver_choice_vars[j][0]))
            for s in range(1, k):
                constraints.append(z3.Or(z3.Not(solver_vars[j]), z3.Not(solver_choice_vars[j-1][s-1]),
                                         solver_choice_vars[j][s]))
                constraints.append(z3.Or(z3.Not(solver_choice_vars[j-1][s]), solver_choice_vars[j][s]))
            constraints.append(z3.Or(z3.Not(solver_vars[j]), z3.Not(solver_choice_vars[j-1][k-1])))
        # Following statement is part of previous loop in "nchoosek", but outside in Sinz' paper:
        constraints.append(z3.Or(z3.Not(solver_vars[-1]), z3.Not(solver_choice_vars[-2][k-1])))
        optimizer.add(z3.And(constraints))
    else:
        optimizer.add(z3.AtMost(*solver_vars, k))
    optimizer.check()
    best_solution = [col for var, col in zip(solver_vars, runtimes.columns) if bool(optimizer.model()[var])]
    if objective.value().is_real():  # Z3 uses representation as fraction
        objective_value = objective.value().numerator_as_long() / objective.value().denominator_as_long()
    else:
        objective_value = objective.value().as_long()
    return [(best_solution, objective_value)]


def beam_search(runtimes: pd.DataFrame, k: int, w: int) -> List[Tuple[List[str], float]]:
    """Beam search for k-portfolios

    Greedy forward search. Starts with a portfolio of size 0 and iteratively adds further solvers
    one-by-one up to portfolio size `k`. In each iteration, combines each current portfolio with
    each feature not in it and retains the `w` portfolios with the highest objective value for the
    next iteration (this is the beam). Returns not only the best `w` portfolios for the final `k`
    but also the intermediate solutions (smaller portfolios that formed the beam).

    Parameters
    ----------
    runtimes : pd.DataFrame
        Solver runtimes (rows represent problem instances, columns represent solvers).
    k : int
        The number of solvers in the portfolio.
    w : int
        Beam width, i.e., number of best portfolios retained in each iteration.

    Returns
    -------
    (List[Tuple[List[str], float]])
        List of portfolios (length = `k` * `w`), with each entry consisting of (1) the solver
        (= column) names and (2) the objective value (average runtime of the virtual best solver
        from the portfolio).
    """

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


def kbest_search(runtimes: pd.DataFrame, k: int) -> List[Tuple[List[str], float]]:
    """K-best search for k-portfolios

    Baseline that ranks solvers by their individual objective value and creates k-portfolios by
    taking the top `k` solvers from this list. Introduced in Nof, Yair, & Strichman, Ofer (2020).
    "Real-time solving of computationally hard problems using optimal algorithm portfolios".
    Returns not only the portfolios for the final `k` but also the intermediate solutions.

    Parameters
    ----------
    runtimes : pd.DataFrame
        Solver runtimes (rows represent problem instances, columns represent solvers).
    k : int
        The number of solvers in the portfolio.

    Returns
    -------
    (List[Tuple[List[str], float]])
        List of portfolios (length = `k`), with each entry consisting of (1) the solver (= column)
        names and (2) the objective value (average runtime of the virtual best solver from the
        portfolio).
    """

    results = []
    sorted_solvers = runtimes.mean().sort_values().index.to_list()
    for i in range(1, k + 1):
        results.append((sorted_solvers[:i], runtimes[sorted_solvers[:i]].min(axis='columns').mean()))
    return results
