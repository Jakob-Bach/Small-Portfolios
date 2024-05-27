"""Run experiments

Main experimental pipeline. Runs different portfolio-search algorithms with different parameters,
trains prediction models with the portfolios, and stores lots of evaluation data.
Execution might take a while, but you can simply reduce the experimental design for testing.

Usage: python -m run_experiments --help
"""

import argparse
import itertools
import multiprocessing
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sklearn.model_selection
import tqdm

import kpsearch
import prediction
import prepare_dataset


CV_FOLDS = 5


# Create a list of search algorithms and their parametrization. Adapt them to the datasets given by
# "problems" (e.g., maximum k is dataset-dependent).
def define_experimental_design(problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for problem, fold_id in itertools.product(problems, range(CV_FOLDS)):
        max_k = problem['runtimes'].shape[1]
        # Beam search and k-best also save intermediate results if run up to max_k:
        for w in list(range(1, 11)) + list(range(20, 101, 10)):
            results.append({**problem, 'fold_id': fold_id, 'search_func': 'beam_search',
                            'search_args': {'k': max_k, 'w': w}})
        results.append({**problem, 'fold_id': fold_id, 'search_func': 'kbest_search',
                        'search_args': {'k': max_k}})
        # The other two search approaches only obtain results for one k at once:
        for k in range(1, max_k + 1):
            results.append({**problem, 'fold_id': fold_id, 'search_func': 'random_search',
                            'search_args': {'k': k, 'w': 1000}})
            results.append({**problem, 'fold_id': fold_id, 'search_func': 'mip_search',
                            'search_args': {'k': k}})
    for i, result in enumerate(results):  # identify combinations of dataset, fold, and search run
        result['search_id'] = i
    return results


# Add several types of aggregate single-solver runtimes and portfolio runtimes to a "search_result",
# which should contain the portfolios in column "solvers". These aggregate runtimes help for
# evaluating the portfolio performance; the search function itself only returns the VBS performance
# on the training set.
# This function modifies "search_result" in-place.
def add_portfolio_performance(search_result: pd.DataFrame, runtimes_train: pd.DataFrame,
                              runtimes_test: pd.DataFrame) -> None:
    search_result['test_objective'] = search_result['solvers'].apply(
            lambda x: runtimes_test[x].min(axis='columns').mean())  # test set VBS
    search_result['train_portfolio_vws'] = search_result['solvers'].apply(
        lambda x: runtimes_train[x].max(axis='columns').mean())  # upper bound for model-based portfolio
    search_result['test_portfolio_vws'] = search_result['solvers'].apply(
        lambda x: runtimes_test[x].max(axis='columns').mean())
    search_result['train_portfolio_sbs'] = search_result['solvers'].apply(
        lambda x: runtimes_train[x].mean().min())  # baseline for model-based portfolio
    search_result['test_portfolio_sbs'] = search_result['solvers'].apply(
        lambda x: runtimes_test[x].mean().min())
    search_result['train_portfolio_sws'] = search_result['solvers'].apply(
        lambda x: runtimes_train[x].mean().max())  # baseline for model-based portfolio
    search_result['test_portfolio_sws'] = search_result['solvers'].apply(
        lambda x: runtimes_test[x].mean().max())
    search_result['train_global_sws'] = runtimes_train.mean().max()  # for submodularity bounds
    search_result['test_global_sws'] = runtimes_test.mean().max()


# Conduct one portfolio search for a particular cross-validation fold of a dataset (runtimes and
# instance features): search for portfolios, make predictions, and compute evaluation metrics.
# - "problem_name" should identify the dataset; is just copied to output.
# - "fold_id" should identify the cross-validation fold.
# - "search_id" should identify the portfolio search run (combination of dataset, fold, and
#   search settings); is just copied to output, but vital to join search and prediction results.
# - "search_func" and "search_args" should allow a function call (see module "search").
# - "runtimes" and "features" are the dataset (with features only being necessary for predictions,
# not for search).
# Return two data frames, one with search results and one with prediction results (can be joined).
def search_and_evaluate(problem_name: str, fold_id: int, search_id: int, search_func: str,
                        search_args: Dict[str, Any], runtimes: pd.DataFrame,
                        features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = sklearn.model_selection.KFold(n_splits=CV_FOLDS, shuffle=True, random_state=25)
    train_idx, test_idx = list(splitter.split(X=runtimes))[fold_id]
    runtimes_train = runtimes.iloc[train_idx]
    runtimes_test = runtimes.iloc[test_idx]
    start_time = time.process_time()
    search_results = getattr(kpsearch, search_func)(runtimes=runtimes_train, **search_args)  # returns list of tuples
    end_time = time.process_time()
    search_results = pd.DataFrame(search_results, columns=['solvers', 'train_objective'])
    add_portfolio_performance(search_result=search_results, runtimes_train=runtimes_train,
                              runtimes_test=runtimes_test)
    search_results['search_time'] = end_time - start_time
    search_results['search_id'] = search_id
    search_results['solution_id'] = np.arange(len(search_results))  # there might be multiple results per search
    search_results['fold_id'] = fold_id
    search_results['problem'] = problem_name
    search_results['algorithm'] = search_func
    for key, value in search_args.items():
        search_results[key] = value

    prediction_results = []
    for _, portfolio_result in search_results.iterrows():
        prediction_result = prediction.predict_and_evaluate(
            runtimes_train=runtimes_train[portfolio_result['solvers']],
            runtimes_test=runtimes_test[portfolio_result['solvers']],
            features_train=features.iloc[train_idx], features_test=features.iloc[test_idx])
        prediction_result['solution_id'] = portfolio_result['solution_id']
        prediction_results.append(prediction_result)
    prediction_results = pd.concat(prediction_results)
    prediction_results['search_id'] = search_id
    return search_results, prediction_results


# Run all experiments and save results.
def run_experiments(data_dir: pathlib.Path, results_dir: pathlib.Path,
                    n_processes: Optional[int] = None) -> None:
    if not data_dir.is_dir():
        raise FileNotFoundError('Data directory does not exist.')
    if not results_dir.is_dir():
        print('Results directory does not exist. We create it.')
        results_dir.mkdir(parents=True)
    if any(results_dir.iterdir()):
        print('Results directory is not empty. Files might be overwritten, but not deleted.')
    problems = []
    for year in [2020, 2021]:
        runtimes, features = prepare_dataset.load_dataset(dataset_name=f'sc{year}', data_dir=data_dir)
        problems.append({'problem_name': f'SC{year}', 'runtimes': runtimes, 'features': features})
    settings_list = define_experimental_design(problems=problems)
    progress_bar = tqdm.tqdm(total=len(settings_list), desc='Experiments')
    process_pool = multiprocessing.Pool(processes=n_processes)
    results = [process_pool.apply_async(search_and_evaluate, kwds=settings,
                                        callback=lambda x: progress_bar.update())
               for settings in settings_list]
    process_pool.close()
    process_pool.join()
    progress_bar.close()
    search_results = pd.concat([x.get()[0] for x in results])
    prediction_results = pd.concat([x.get()[1] for x in results])
    search_results.to_csv(results_dir / 'search_results.csv', index=False)
    prediction_results.to_csv(results_dir / 'prediction_results.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs the experimental pipeline. Might take a while.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='data/', dest='data_dir',
                        help='Directory with input data, i.e., runtimes and instance features.')
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/', dest='results_dir',
                        help='Directory for output data, i.e., experimental results.')
    parser.add_argument('-p', '--processes', type=int, default=None, dest='n_processes',
                        help='Number of processes for multi-processing (default: all cores).')
    print('Experimental pipeline started.')
    run_experiments(**vars(parser.parse_args()))
    print('Experimental pipeline executed successfully.')
