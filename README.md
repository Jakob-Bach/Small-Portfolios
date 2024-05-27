# A Comprehensive Study of k-Portfolios of Recent SAT Solvers

This repository contains the code and text of the paper

> Bach, Jakob, Markus Iser, and Klemens Böhm. "A Comprehensive Study of k-Portfolios of Recent SAT Solvers"

[published](https://doi.org/10.4230/LIPIcs.SAT.2022.2) at [`SAT 2022`](http://satisfiability.org/SAT22/).
The corresponding complete experimental data (inputs as well as results) are available on [KITopenData](https://doi.org/10.5445/IR/1000146629).

This document provides:

- an [outline](#repo-structure) of the repo structure
- [guidelines](#development) for developers who want to modify or extend the code base
- steps to [reproduce](#reproducing-the-experiments) the experiments, including [setting up](#setup) a virtual environment

## Repo Structure

The repo contains three folders: one with the text of the paper, one with the conference presentation, and one with the Python code.
The folder `code/` contains four Python files and three non-code files.
The non-code files are:

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

Additionally, we have organized the portfolio-search methods for our experiments as the standalone Python package `kpsearch`,
located in the directory `code/kpsearch_package/`.
See the corresponding [README](code/kpsearch_package/README.md) for more information.

## Development

Possible points for modifying and extending our code are
the prediction approaches, search approaches, and dataset integration.

### Prediction

If you want to change the prediction models (or their parametrization) used in the experiments,
modify the variable `MODELS` in `prediction.py`.
Any other changes to the prediction procedure (e.g., target, imputation, evaluation metrics, etc.)
should be made in the same file as well.

### Portfolio Search

If you want to develop another portfolio-search method,
have a look at the [package README of `kpsearch`](code/kpsearch_package/README.md)
for details on how the search method's interface should look like.
If you want to change which search routines are used in the experiments or how they are configured,
have a look at the function `define_experimental_design()` in `run_experiments.py`.

### Datasets

If you want to analyze further datasets, you should bring them into a format similar to the one
produced by `prepare_dataset.py`.
In particular, you should store the dataset in the form of two CSVs:

- `{dataset_name}_runtimes.csv` should contain the solver runtimes as floats or ints.
  Each row is an instance, and each column is a solver (column names are solver names).
  Additionally, column `hash` should identify the instance.
  It is not used for search and predictions.
- `{dataset_name}_features.csv` should contain the instance features as floats or ints.
  Each row is an instance, and each column is a feature (column names are feature names).
  Additionally, column `hash` should identify the instance.
  It is not used for search and predictions.
  Please encode categorical features beforehand (e.g., via one-hot encoding).
  Before prediction, we will replace missing values with a constant outside the feature's range.

Having prepared the dataset(s) in this format, you need to adapt the function `run_experiments()` in
`run_experiments.py` to use your datasets instead of / beside the SAT Competition ones.
The evaluation code in `run_evaluation.py` also partly contains hard-coded dataset names,
but the evaluation functionality itself should work with results from arbitrary datasets.

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
(It took about 25 hours on our server with 32 CPU cores and a base clock of 2.0 GHz.)
To create the plots for the paper and print some statistics to the console, run

```bash
python -m run_evaluation
```

All scripts have a few command-line options, which you can see by running the scripts like

```bash
python -m prepare_dataset --help
```
