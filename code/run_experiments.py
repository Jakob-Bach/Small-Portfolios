"""Run experiments

Main experimental pipeline. Runs different portfolio-search algorithms with different parameters.
"""

import argparse
import multiprocessing
import pathlib
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import tqdm

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
    return results


# Run one portfolio search, wrapped in some other stuff to improve results structure.
def run_search(settings: Dict[str, Any]) -> pd.DataFrame:
    search_func = getattr(search, settings['func'])
    start_time = time.process_time()
    results = search_func(**settings['args'])  # returns list of tuples
    end_time = time.process_time()
    results = pd.DataFrame(results, columns=['solvers', 'objective_value'])
    results['time'] = end_time - start_time
    results['problem'] = settings['problem']
    results['algorithm'] = settings['func']  # add algo's name and params to result
    for key, value in settings['args'].items():
        if key != 'runtimes':
            results[key] = value
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
    runtimes = runtimes[(runtimes != prepare_dataset.PENALTY).any(axis='columns')]  # drop unsolved instances
    solved_states = (runtimes == prepare_dataset.PENALTY).astype(int)  # discretized runtimes
    settings_list = define_experimental_design(problems={'PAR2': runtimes, 'solved': solved_states})
    progress_bar = tqdm.tqdm(total=len(settings_list))
    process_pool = multiprocessing.Pool(processes=n_processes)
    results = [process_pool.apply_async(run_search, kwds={'settings': settings},
                                        callback=lambda x: progress_bar.update())
               for settings in settings_list]
    process_pool.close()
    process_pool.join()
    progress_bar.close()
    results = pd.concat([x.get() for x in results])
    results.to_csv(results_dir / 'results.csv', index=False)


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
