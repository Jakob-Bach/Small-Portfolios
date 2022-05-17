"""Prepare dataset

Download GBD databases, transform databases to CSVs, and extract input data for experimental
pipeline, i.e., solver runtimes and instance features.
In particular, we use instances from the Main Track of the SAT Competitions 2020 and 2021.

Usage: python -m prepare_dataset --help
"""

from typing import Tuple

import argparse
import pathlib
import urllib.request

import gbd_tool.gbd_api
import pandas as pd
import tqdm


DATABASE_NAMES = ['meta', 'satzilla', 'sc2020', 'sc2021']
PENALTY = 10000  # PAR2 score with timeout of 5000 s


# Download database files and save them in original format + CSV in "data_dir".
def download_and_save_dbs(data_dir: pathlib.Path) -> None:
    for db_name in tqdm.tqdm(DATABASE_NAMES, desc='Downloading'):
        urllib.request.urlretrieve(url=f'https://gbd.iti.kit.edu/getdatabase/{db_name}_db',
                                   filename=data_dir / f'{db_name}.db')
        with gbd_tool.gbd_api.GBD(db_list=[str(data_dir / f'{db_name}.db')]) as api:
            features = api.get_features()
            features.remove('hash')  # will be added to result anyway, so avoid duplicates
            database = pd.DataFrame(api.query_search(resolve=features), columns=['hash'] + features)
            database.to_csv(data_dir / f'{db_name}.csv', index=False)


# Save runtimes and features for experimental pipeline. Method is rather trivial, but serves as an
# interface (if file type or name changes, needs only to be modified once in save() and load()
# instead of searching through whole code)
def save_dataset(runtimes: pd.DataFrame, features: pd.DataFrame, dataset_name: str,
                 data_dir: pathlib.Path) -> None:
    runtimes.to_csv(data_dir / f'{dataset_name}_runtimes.csv', index=False)
    features.to_csv(data_dir / f'{dataset_name}_features.csv', index=False)


# Load runtimes and features as data frames. If "exclude_solved", delete instances (rows) where
# all solvers have the penalty value (as we do in our experiments).
def load_dataset(dataset_name: str, data_dir: pathlib.Path,
                 exclude_unsolved: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    runtimes = pd.read_csv(data_dir / f'{dataset_name}_runtimes.csv')
    features = pd.read_csv(data_dir / f'{dataset_name}_features.csv')
    # Hashes of data should be aligned, as we often use position-based access (instead of join):
    assert (runtimes['hash'] == features['hash']).all()
    runtimes.drop(columns='hash', inplace=True)
    features.drop(columns='hash', inplace=True)
    if exclude_unsolved:
        keep_instance = (runtimes != PENALTY).any(axis='columns')
        runtimes = runtimes[keep_instance]  # drop instances not solved by any solver in time
        features = features[keep_instance]
    return runtimes, features


# Create file with runtimes and file with instance features. To that end, do some pre-processing.
# Load/save all necessary I/O data from/to "data_dir".
def transform_csvs_for_pipeline(data_dir: pathlib.Path) -> None:
    meta_db = pd.read_csv(data_dir / 'meta.csv')  # allows to filter for competitions and tracks
    satzilla_db = pd.read_csv(data_dir / 'satzilla.csv')

    for year in [2020, 2021]:
        hashes = meta_db.loc[meta_db['track'].fillna('').str.contains(f'main_{year}'), 'hash']

        runtimes = pd.read_csv(data_dir / f'sc{year}.csv')
        runtimes = runtimes[runtimes['hash'].isin(hashes)].reset_index(drop=True)
        runtimes.sort_values(by='hash', inplace=True)  # so runtimes and features have same order of instances
        numeric_cols = [x for x in runtimes.columns if x != 'hash']
        runtimes[numeric_cols] = runtimes[numeric_cols].transform(pd.to_numeric, errors='coerce')
        runtimes.fillna(value=PENALTY, inplace=True)

        features = satzilla_db[satzilla_db['hash'].isin(hashes)].reset_index(drop=True)
        features.sort_values(by='hash', inplace=True)
        assert (runtimes['hash'] == features['hash']).all()
        numeric_cols = [x for x in features.columns if x != 'hash']
        features[numeric_cols] = features[numeric_cols].transform(pd.to_numeric, errors='coerce')

        save_dataset(runtimes=runtimes, features=features, dataset_name=f'sc{year}', data_dir=data_dir)


# Main routine: Download, pre-process, save.
def prepare_dataset(data_dir: pathlib.Path) -> None:
    if not data_dir.is_dir():
        print('Data directory does not exist. We create it.')
        data_dir.mkdir(parents=True)
    if any(data_dir.iterdir()):
        print('Data directory is not empty. Files might be overwritten, but not deleted.')
    print('Downloading and saving databases ...')
    download_and_save_dbs(data_dir=data_dir)
    print('Transforming databases for pipeline ...')
    transform_csvs_for_pipeline(data_dir=data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Retrieves databases from the GBD website, joins them, creates CSVs, ' +
        'prepares data for experimental pipeline, and stores all data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--directory', type=pathlib.Path, default='data/', dest='data_dir',
                        help='Output directory for data (.db and .csv files).')
    print('Dataset preparation started.')
    prepare_dataset(**vars(parser.parse_args()))
    print('Dataset prepared and saved.')
