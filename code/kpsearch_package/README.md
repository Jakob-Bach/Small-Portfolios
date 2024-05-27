# `kpsearch` -- A Python Package for K-Portfolio Search

The package `kpsearch` contains several portfolio-search methods.

This document provides:

- Steps for [setting up](#setup) the package.
- A short [overview](#functionality) of the (portfolio-search) functionality.
- A [demo](#demo) of the functionality.
- [Guidelines for developers](#developer-info) who want to modify or extend the code base.

If you use this package for a scientific publication, please cite [our paper](https://doi.org/10.4230/LIPIcs.SAT.2022.2)

```
@inproceedings{bach2022comprehensive,
  author={Bach, Jakob and Iser, Markus and B\"{o}hm, Klemens},
  title={A Comprehensive Study of k-Portfolios of Recent {SAT} Solvers},
  booktitle={25th International Conference on Theory and Applications of Satisfiability Testing (SAT 2022)},
  location={Haifa, Israel},
  year={2022},
  doi={10.4230/LIPIcs.SAT.2022.2},
}
```

## Setup

You can directly install this package from GitHub:

```bash
python -m pip install git+https://github.com/Jakob-Bach/Small-Portfolios.git#subdirectory=code/kpsearch_package
```

If you already have the source code for the package (i.e., the directory in which this `README` resides)
as a local directory on your computer (e.g., after cloning the project), you can also perform a local install:

```bash
python -m pip install kpsearch_package/
```

## Functionality

`kpsearch.py` provides several functions for searching k-portfolios:

- exact: `exhaustive_search()`, `mip_search()`, `smt_search()`, `smt_search_nof()`
- heuristic: `beam_search()`, `kbest_search()`, `random_search()`

`mip_search()` is a novel contribution, using a MIP solver to find optimal k-portfolios.
All other search methods are straightforward and/or adaptations from related work.
`mip_search()` is usually the fastest exact search method except for very small or large `k`
(where the plain `exhaustive_search()` may be faster).

All functions share two parameters:

- A `pandas.DataFrame`, where the rows represent problem (e.g., SAT) instances,
  the columns represent solvers (column names are solver names), and the cells represent runtimes.
- The portfolio size `k`, i.e., the maximum number of solvers in the portfolio
  (some search methods may return smaller portfolios if adding more solvers is not beneficial).

The existence of further parameters depends on the search method;
see the individual functions' documentation for details.

All search methods return a list, where each entry is a portfolio described by

- the solver names (column names from the runtime data) and
- the objective value (average runtime of the VBS, i.e., virtual best solver, which assumes the
  lowest runtime of the portfolio members for each instance).

## Demo

Let's create a small demo dataset with runtimes of three solvers on four instances:

```python
import pandas as pd
import kpsearch

runtimes = pd.DataFrame({'Solver1': [1, 2, 3, 4],
                         'Solver2': [2, 2, 5, 1],
                         'Solver3': [5, 3, 2, 1]})
```

I.e., the data looks as follows:

```
   Solver1  Solver2  Solver3
0        1        2        5
1        2        2        3
2        3        5        2
3        4        1        1
```

Let's try exhaustive search:

```python
print(kpsearch.exhaustive_search(runtimes=runtimes, k=2))
```

As you would expect, this search procedure returns all portfolios of the desired size `k`:

```
[(['Solver1', 'Solver2'], 1.75), (['Solver1', 'Solver3'], 1.5), (['Solver2', 'Solver3'], 1.75)]
```

Let's try greedy search, i.e., a beam search with a beam width of one:

```python
print(kpsearch.beam_search(runtimes=runtimes, k=3, w=1))
```

This search procedure does not only yield a `k`-portfolio, but also all intermediate results:

```
[(['Solver1'], 2.5), (['Solver1', 'Solver3'], 1.5), (['Solver1', 'Solver2', 'Solver3'], 1.5)]
```

We can see that the third iteration does not improve the portfolio's VBS, i.e.,
Solver 2 cannot solve any instance faster than both other solvers.

## Developer Info

Though there is no formal superclass or interface, all existing search methods follow specific conventions,
which makes them compatible with each other and the experimental pipeline.
Thus, if you want to add another portfolio-search method, it may be beneficial to follow these conventions as well.

All search methods share two parameters:
The solver `runtimes` as `DataFrame` and the number of solvers `k` as `int`.
You can add further method-specific parameters as you like.
For example, beam search has the beam width `w` as another parameter.

The result of each search method is a list of tuples of

- solver names (list of strings, corresponding to column names in `runtimes`) and
- portfolio performance (float) in terms of VBS score.

The list may also have a length of one in case the search only returns one portfolio.
