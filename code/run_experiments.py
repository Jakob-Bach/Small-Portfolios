"""Run experiments

Main experimental pipeline. Runs different portfolio-search algorithms with different parameters.

Usage: python -m run_experiments --help
"""

import argparse
import multiprocessing
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sklearn.model_selection
import tqdm

import prediction
import prepare_dataset
import search


CV_FOLDS = 5


# Define the different optimization problems. Though the overall objective function and cardinality
# constraint are the same, the definition of runtimes r_T(s,i) varies.
def define_problems(runtimes: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    is_unsolved = (runtimes == prepare_dataset.PENALTY).astype(int)
    runtimes_norm = runtimes.transform(lambda x: (x - x.min()) / (x.max() - x.min()), axis='columns')
    return {'PAR2': runtimes, 'PAR2_norm': runtimes_norm, 'Unsolved': is_unsolved}


# Create a list of search algorithms and their parametrization.
def define_experimental_design() -> List[Dict[str, Any]]:
    results = []
    for k in range(1, 49):
        results.append({'func': 'random_search', 'args': {'k': k, 'w': 1000}})
    for k in range(1, 49):
        results.append({'func': 'mip_search', 'args': {'k': k}})
    for w in list(range(1, 11)) + list(range(20, 101, 10)):
        results.append({'func': 'beam_search', 'args': {'k': 48, 'w': w}})
    results.append({'func': 'kbest_search', 'args': {'k': 48}})
    for i, result in enumerate(results):
        result['settings_id'] = i
    return results


# Conduct one (actually, more than one, due to cross-validation) portfolio search for a particular
# dataset (runtimes and instance features): search for portfolios, make predictions, and compute
# evaluation metrics.
# - "search_settings" should contain the search function and their arguments.
# - "problem_name" should identify the dataset.
# - "runtimes" and "features" are the dataset (with features only being necessary for predictions,
# not for search).
# Return two data frames, one with search results and one with prediction results (can be joined).
def search_and_evaluate(problem_name: str, runtimes: pd.DataFrame, features: pd.DataFrame,
                        search_settings: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    search_func = getattr(search, search_settings['func'])
    splitter = sklearn.model_selection.KFold(n_splits=CV_FOLDS, shuffle=True, random_state=25)
    search_results = []
    prediction_results = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X=runtimes)):
        runtimes_train = runtimes.iloc[train_idx]
        runtimes_test = runtimes.iloc[test_idx]
        search_args = search_settings['args'].copy()
        search_args['runtimes'] = runtimes_train
        start_time = time.process_time()
        search_result = search_func(**search_args)  # returns list of tuples
        end_time = time.process_time()
        search_result = pd.DataFrame(search_result, columns=['solvers', 'train_objective'])
        search_result['test_objective'] = search_result['solvers'].apply(
            lambda x: runtimes_test[x].min(axis='columns').mean())
        search_result['train_vbs'] = runtimes_train.min(axis='columns').mean()
        search_result['test_vbs'] = runtimes_test.min(axis='columns').mean()
        search_result['train_vws'] = runtimes_train.max(axis='columns').mean()
        search_result['test_vws'] = runtimes_test.max(axis='columns').mean()
        search_result['search_time'] = end_time - start_time
        search_result['solution_id'] = np.arange(len(search_result))  # there might be multiple results per search
        search_result['fold_id'] = fold_id
        search_results.append(search_result)
        for _, portfolio_result in search_result.iterrows():
            start_time = time.process_time()
            prediction_result = prediction.predict_and_evaluate(
                runtimes_train=runtimes_train[portfolio_result['solvers']],
                runtimes_test=runtimes_test[portfolio_result['solvers']],
                features_train=features.iloc[train_idx], features_test=features.iloc[test_idx])
            end_time = time.process_time()
            prediction_result['pred_time'] = end_time - start_time
            prediction_result['solution_id'] = portfolio_result['solution_id']
            prediction_result['fold_id'] = portfolio_result['fold_id']
            prediction_results.append(prediction_result)
    search_results = pd.concat(search_results)
    search_results['settings_id'] = search_settings['settings_id']
    search_results['problem'] = problem_name
    search_results['algorithm'] = search_settings['func']
    for key, value in search_settings['args'].items():
        search_results[key] = value
    prediction_results = pd.concat(prediction_results)
    prediction_results['settings_id'] = search_settings['settings_id']
    prediction_results['problem'] = problem_name
    return search_results, prediction_results


# Run all experiments and save results.
def run_experiments(data_dir: pathlib.Path, results_dir: pathlib.Path, n_processes: Optional[int] = None) -> None:
    if not data_dir.is_dir():
        raise FileNotFoundError('Data directory does not exist.')
    if not results_dir.is_dir():
        print('Results directory does not exist. We create it.')
        results_dir.mkdir(parents=True)
    if any(results_dir.iterdir()):
        print('Results directory is not empty. Files might be overwritten, but not deleted.')
    runtimes, features = prepare_dataset.load_dataset(data_dir=data_dir)
    problems = define_problems(runtimes=runtimes)
    settings_list = define_experimental_design()
    print('Running evaluation...')
    progress_bar = tqdm.tqdm(total=len(settings_list) * len(problems))
    process_pool = multiprocessing.Pool(processes=n_processes)
    results = [process_pool.apply_async(search_and_evaluate, kwds={
        'problem_name': problem_name, 'runtimes': runtime_data, 'features': features,
        'search_settings': settings}, callback=lambda x: progress_bar.update())
        for problem_name, runtime_data in problems.items() for settings in settings_list]
    process_pool.close()
    process_pool.join()
    progress_bar.close()
    search_results = pd.concat([x.get()[0] for x in results])
    prediction_results = pd.concat([x.get()[1] for x in results])
    search_results.to_csv(results_dir / 'search_results.csv', index=False)
    prediction_results.to_csv(results_dir / 'prediction_results.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs the experimental pipeline.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='data/', dest='data_dir',
                        help='Directory with input data, i.e., runtimes and instance features.')
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/', dest='results_dir',
                        help='Directory for output data, i.e., experimental results.')
    parser.add_argument('-p', '--processes', type=int, default=None, dest='n_processes',
                        help='Number of processes for multi-processing (default: all cores).')
    print('Experimental pipeline started.')
    run_experiments(**vars(parser.parse_args()))
    print('Experimental pipeline executed successfully.')
