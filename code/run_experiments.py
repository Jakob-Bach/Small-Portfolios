"""Run experiments

Main experimental pipeline. Runs different portfolio-search algorithms with different parameters.
"""

import argparse
import multiprocessing
import pathlib
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import tqdm

import prediction
import prepare_dataset
import search


# Create a list of search algorithms and their parametrization.
# "problems" should be a dict of problem names and the corresponding datasets (runtimes).
def define_experimental_design(problems: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    results = []
    for problem_name, problem_data in problems.items():
        for k in range(1, 6):
            results.append({'problem': problem_name, 'func': 'exhaustive_search',
                            'args': {'runtimes': problem_data, 'k': k}})
        for k in range(1, 49):
            results.append({'problem': problem_name, 'func': 'mip_search',
                            'args': {'runtimes': problem_data, 'k': k}})
        for w in list(range(1, 11)) + list(range(20, 101, 10)):
            results.append({'problem': problem_name, 'func': 'beam_search',
                            'args': {'runtimes': problem_data, 'k': 48, 'w': w}})
    for i, result in enumerate(results):
        result['settings_id'] = i
    return results


# Run one portfolio search, wrapped in some other stuff to improve results structure.
# "settings" should contain the data for the search (algorithm, parametrization, solver runtimes).
# Return evaluation portfolios and metrics.
def run_search(settings: Dict[str, Any]) -> pd.DataFrame:
    search_func = getattr(search, settings['func'])
    start_time = time.process_time()
    results = search_func(**settings['args'])  # returns list of tuples
    end_time = time.process_time()
    results = pd.DataFrame(results, columns=['solvers', 'objective_value'])
    results['search_time'] = end_time - start_time
    results['settings_id'] = settings['settings_id']
    results['solution_id'] = np.arange(len(results))  # there might be multiple results per search
    results['problem'] = settings['problem']
    results['algorithm'] = settings['func']  # add algo's name and params to result
    for key, value in settings['args'].items():
        if key != 'runtimes':
            results[key] = value
    return results


# Evaluate predictions for "runtimes" of one portfolio, using instance "features" for training.
# Return evaluation metrics, adding "ids" as additional columns to identify the portfolio.
def run_prediction(runtimes: pd.DataFrame, features: pd.DataFrame, ids: Dict[str, int]) -> pd.DataFrame:
    start_time = time.process_time()
    results = prediction.predict_and_evaluate(runtimes=runtimes, features=features)
    end_time = time.process_time()
    results['prediction_time'] = end_time - start_time
    for id_key, id_value in ids.items():
        results[id_key] = id_value
    return results


# Run all experiments and save results.
def run_experiments(data_dir: pathlib.Path, results_dir: pathlib.Path, n_processes: Optional[int] = None) -> None:
    if not data_dir.is_dir():
        raise FileNotFoundError('Data directory does not exist.')
    if not results_dir.is_dir():
        print('Results directory does not exist. We create it.')
        results_dir.mkdir(parents=True)
    if any(results_dir.iterdir()):
        print('Results directory is not empty. Files might be overwritten, but not deleted.')
    runtimes = pd.read_csv(data_dir / 'runtimes.csv').drop(columns='hash')
    features = pd.read_csv(data_dir / 'features.csv').drop(columns='hash')
    keep_instance = (runtimes != prepare_dataset.PENALTY).any(axis='columns')
    runtimes = runtimes[keep_instance]  # drop instances not solved by any solver in time
    features = features[keep_instance]
    solved_states = (runtimes == prepare_dataset.PENALTY).astype(int)  # discretized runtimes
    problems = {'PAR2': runtimes, 'solved': solved_states}
    settings_list = define_experimental_design(problems=problems)
    print('Running search ...')
    progress_bar = tqdm.tqdm(total=len(settings_list))
    process_pool = multiprocessing.Pool(processes=n_processes)
    search_results = [process_pool.apply_async(run_search, kwds={'settings': settings},
                                               callback=lambda x: progress_bar.update())
                      for settings in settings_list]
    search_results = pd.concat([x.get() for x in search_results]).reset_index(drop=True)
    progress_bar.close()
    search_results.to_csv(results_dir / 'search_results.csv', index=False)
    print('Running prediction ...')
    progress_bar = tqdm.tqdm(total=len(search_results))
    prediction_results = [process_pool.apply_async(run_prediction, kwds={
        'runtimes': problems[search_result['problem']][search_result['solvers']],
        'features': features, 'ids': search_result[['settings_id', 'solution_id']]},
        callback=lambda x: progress_bar.update())
        for _, search_result in search_results.iterrows()]
    process_pool.close()
    process_pool.join()
    progress_bar.close()
    prediction_results = pd.concat([x.get() for x in prediction_results])
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
