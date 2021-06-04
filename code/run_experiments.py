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

import search


# Create a list of search algorithms and their parametrization.
def define_experimental_design() -> List[Dict[str, Any]]:
    results = []
    for k in range(1, 6):
        results.append({'func': 'exhaustive_search', 'args': {'k': k}})
    for k in range(1, 49):
        results.append({'func': 'mip_search', 'args': {'k': k}})
    for w in list(range(1, 11)) + list(range(20, 101, 10)):
        results.append({'func': 'beam_search', 'args': {'k': 48, 'w': w}})
    return results


# Run one portfolio search, wrapped in some other stuff to improve results structure.
def run_search(runtimes: pd.DataFrame, settings: Dict[str, Any]) -> pd.DataFrame:
    search_func = getattr(search, settings['func'])
    search_args = settings['args'].copy()  # copy since we don't want to add runtimes to result later
    search_args['runtimes'] = runtimes
    start_time = time.process_time()
    results = search_func(**search_args)  # returns list of tuples
    end_time = time.process_time()
    results = pd.DataFrame(results, columns=['solvers', 'objective_value'])
    results['time'] = end_time - start_time
    results['algorithm'] = settings['func']  # add algo's name and params to result
    for key, value in settings['args'].items():
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
    settings_list = define_experimental_design()
    progress_bar = tqdm.tqdm(total=len(settings_list))
    process_pool = multiprocessing.Pool(processes=n_processes)
    results = [process_pool.apply_async(run_search, kwds={'runtimes': runtimes, 'settings': settings},
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
