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


DATABASES = ['meta.db', 'gates.db', 'satzilla.db', 'sc2020.db', 'sc2021.db']
PENALTY = 10000  # PAR2 score with timeout of 5000 s


# Download database files, save in "data_dir".
def download_dbs(data_dir: pathlib.Path) -> None:
    for database in DATABASES:
        urllib.request.urlretrieve(url='https://gbd.iti.kit.edu/getdatabase/' + database,
                                   filename=data_dir / database)


# Join all tables within each database (.db file) in "data_dir" and save as CSVs in "data_dir".
# In the databases, each instance feature is in a separate table, but we want a unified dataset.
def transform_dbs_to_csvs(data_dir: pathlib.Path) -> None:
    for database in DATABASES:
        db_path = str(data_dir) + '/' + database
        with gbd_tool.gbd_api.GbdApi(db_string=db_path) as api:
            features = api.get_features(path=db_path)
            features.remove('hash')  # is key of all other tables, so we don't need this table
            dataset = None
            # Getting multiple features at same time (with join in database) also possible,
            # but breaks if too many features and is actually slower than join via pandas
            for feature in features:  # for each database table (containing two columns, instance hash and one feature)
                feature_table = pd.DataFrame(api.query_search(query=None, resolve=[feature]), columns=['hash', feature])
                if dataset is None:
                    dataset = feature_table
                else:
                    dataset = dataset.merge(feature_table, on='hash', how='outer', copy=False)
            dataset.to_csv(data_dir / database.replace('.db', '.csv'), index=False)


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
    # Gate-recognition instance features and SATzilla instance features; number of variables and
    # clauses is in both databases, so we drop them once:
    gates_db = pd.read_csv(data_dir / 'gates.csv').drop(columns=['local', 'filename', 'tags', 'variables', 'clauses'])
    satzilla_db = pd.read_csv(data_dir / 'satzilla.csv').drop(columns=['local', 'filename', 'tags'])

    for year in [2020, 2021]:
        hashes = meta_db.loc[meta_db['competition_track'].fillna('').str.contains(f'main_{year}'), 'hash']

        runtimes = pd.read_csv(data_dir / f'sc{year}.csv').drop(columns=['tags', 'filename', 'local'])
        runtimes = runtimes[runtimes['hash'].isin(hashes)].reset_index(drop=True)
        runtimes.sort_values(by='hash', inplace=True)  # so runtimes and features have same order of instances
        numeric_cols = [x for x in runtimes.columns if x != 'hash']
        runtimes[numeric_cols] = runtimes[numeric_cols].transform(pd.to_numeric, errors='coerce')
        runtimes.fillna(value=PENALTY, inplace=True)

        gates = gates_db[gates_db['hash'].isin(hashes)].reset_index(drop=True)
        satzilla = satzilla_db[satzilla_db['hash'].isin(hashes)].reset_index(drop=True)
        features = satzilla.merge(gates, on='hash')
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
    print('Downloading databases ...')
    download_dbs(data_dir=data_dir)
    print('Transforming database files to CSVs ...')
    transform_dbs_to_csvs(data_dir=data_dir)
    print('Tranforming CSVs for pipeline ...')
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
