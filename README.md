# A Comprehensive Study of k-Portfolios of Recent SAT Solvers

This repository contains the code and text of the paper

> Bach, Jakob, Markus Iser, and Klemens Böhm. "A Comprehensive Study of k-Portfolios of Recent SAT Solvers"

(The paper is accepted at [SAT 2022](http://satisfiability.org/SAT22/) but not published yet.
Once it's published, we'll add a link to it here.)
You can find the corresponding full experimental data (inputs as well as results) on [KITopenData](https://doi.org/10.5445/IR/1000146629).

This document provides:

- an outline of the repo structure
- a short demo of portfolio search with our code
- guidelines for developers who want to modify or extend the code base
- steps to reproduce the experiments, including setting up a virtual environment

## Repo Structure

The repo contains two folders, one with the text of the paper and one with the Python code.
In the folder `code/`, there are five Python files and three non-code files.
The non code-files are:

- `.gitignore`: For Python development.
- `LICENSE`: The code is MIT-licensed, so feel free to use it.
  The files in `text/` do not fall under this license.
- `requirements.txt`: The necessary Python dependencies for our virtual environment; see below for details.

The code files are:

- `prediction.py`: Prediction models and functions to make + evaluate predictions.
- `prepare_dataset.py`: First stage of the experiments (download and pre-process data)
  and functions for data handling.
- `run_evaluation.py`: Third stage of the experiments (compute statistics and create plots for the paper).
- `run_experiments.py`: Second stage of the experiments (search for portfolios and make predictions).
- `search.py`: Functions to search for solver portfolios.

## Demo

Let's prepare a small demo dataset with solver runtimes:

```python
import pandas as pd
import search

runtimes = pd.DataFrame({'Solver1': [1, 2, 3, 4], 'Solver2': [2, 2, 5, 1], 'Solver3': [5, 3, 2, 1]})
```

Let's try exhaustive search:

```python
print(search.exhaustive_search(runtimes=runtimes, k=2))
```

As you would expect, this search procedure returns all portfolios of the desired size `k`:

```
[(['Solver1', 'Solver2'], 1.75), (['Solver1', 'Solver3'], 1.5), (['Solver2', 'Solver3'], 1.75)]
```

Let's try greedy search, i.e., beam search with a beam width of one:

```python
print(search.beam_search(runtimes=runtimes, k=3, w=1))
```

This search procedure does not only yield the `k`-portfolio, but also all intermediate results:

```
[(['Solver1'], 2.5), (['Solver1', 'Solver3'], 1.5), (['Solver1', 'Solver2', 'Solver3'], 1.5)]
```

We can see that the third iteration does not lead to any marginal gain, i.e.,
solver 2 cannot solve any instance faster than both solver 1 and solver 3.

## Development

Possible points for modifying and extending our code are
the prediction approaches, search approaches, and dataset integration.

### Prediction

If you want to change the set of prediction models used in the experiments,
or their parametrization, modify the variable `MODELS` in `prediction.py`.
Any other changes to prediction procedure (e.g., target, imputation, evaluation metrics, etc.)
should be made in the same file as well.

### Portfolio Search

If you want to add another routine for portfolio search, have a look at `search.py`.
Though there is no formal superclass or interface, all search routines share two parameters:
The solver `runtimes` as `DataFrame` and the number of solvers `k` as `int`.
The result is a list of tuples of

- solver names (list of strings) and
- portfolio performance (float).

The list might also just have a length one, in case the search only returns one portfolio.
You can add further, arbitrarily named parameters as well.
For example, beam search has the beam width `w` as another parameter.

If you want to change which search routines are used in the experiments or how they are configured,
have a look at the function `define_experimental_design()` in `run_experiments.py`.

### Datasets

If you want to analyze further datasets, you should bring them into a format similar to the one
produced by `prepare_dataset.py`.
In particular, the dataset should be stored in the form of two CSVs:

- `{dataset_name}_runtimes.csv` should contain the solver runtimes as floats or ints.
  Each row is an instance, and each column is a solver (column names are solver names).
  Additionally, column `hash` should identify the instance.
  It is not used for search and predictions.
- `{dataset_name}_features.csv` should contain the instance features as floats or ints.
  Each row is an instance, and each column is a feature (column names are feature names).
  Please encode categorical features beforehand (e.g., via one-hot encoding).
  Missing values will be replaced with a constant outside the feature's range before predicting.

Having prepared the dataset(s) in this format, you need to adapt the function `run_experiments()` in
`run_experiments.py` such that it uses your datasets instead of / besides the SAT Competition ones.
The evaluation code in `run_evaluation.py` also partly contains hard-coded dataset names but the
evaluation functionality itself should work with results from arbitrary datasets.

## Setup

Before running scripts to reproduce the experiments,
you need to set up an environment with all necessary dependencies.
Our code is implemented in Python (version 3.8; other versions, including lower ones, might also work).

### Option 1: `conda` Environment

If you use `conda`, you can directly install the correct Python version into a new `conda`
environment and activate the environment as follows:

```bash
conda create --name <conda-env-name> python=3.8
conda activate <conda-env-name>
```

Choose `<conda-env-name>` as you like.

To leave the environment, run

```bash
conda deactivate
```

### Option 2: `virtualenv` Environment

We used [`virtualenv`](https://virtualenv.pypa.io/) (version 20.4.7; other versions might also work)
to create an environment for our experiments.
First, you need to install the correct Python version yourself.
Let's assume the Python executable is located at `<path/to/python>`.
Next, you install `virtualenv` with

```bash
python -m pip install virtualenv==20.4.7
```

To set up an environment with `virtualenv`, run

```bash
python -m virtualenv -p <path/to/python> <path/to/env/destination>
```

Choose `<path/to/env/destination>` as you like.

Activate the environment in Linux with

```bash
source <path/to/env/destination>/bin/activate
```

Activate the environment in Windows (note the back-slashes) with

```cmd
<path\to\env\destination>\Scripts\activate
```

To leave the environment, run

```bash
deactivate
```

### Dependency Management

After activating the environment, you can use `python` and `pip` as usual.
To install all necessary dependencies for this repo, switch to the directory `code` and run

```bash
python -m pip install -r requirements.txt
```

If you make changes to the environment and you want to persist them, run

```bash
python -m pip freeze > requirements.txt
```

## Reproducing the Experiments

After setting up and activating an environment, you are ready to run the code.
From the directory `code`, run

```bash
python -m prepare_dataset
```

to download and pre-process instance features as well as SAT Competition runtimes from
the [GBD website](https://gbd.iti.kit.edu/).
Next, start the pipeline with

```bash
python -m run_experiments
```

Depending on your hardware, this might take several hours or even days.
To create the plots for the paper and print some statistics to the console, run

```bash
python -m run_evaluation
```

All scripts have a few command line options, which you can see by running the scripts like

```bash
python -m prepare_dataset --help
```
