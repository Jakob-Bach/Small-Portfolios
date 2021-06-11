"""Prepare dataset

Download GBD databases, transform to CVS, and extract input data for experimental pipeline:
solver runtimes and instance features for main track of SAT Competition 2020.

Usage: python -m prepare_dataset --help
"""

import argparse
import pathlib
import urllib.request

import gbd_tool.gbd_api
import pandas as pd


DATABASES = ['meta.db', 'gates.db', 'satzilla.db', 'sc2020.db']
PENALTY = 10000  # PAR2 score with timeout of 5000


# Download database files.
def download_dbs(data_dir: pathlib.Path) -> None:
    for database in DATABASES:
        urllib.request.urlretrieve(url='https://gbd.iti.kit.edu/getdatabase/' + database,
                                   filename=data_dir / database)


# Join all tables within each database and save as CSV.
def transform_dbs_to_csvs(data_dir: pathlib.Path) -> None:
    for database in DATABASES:
        db_path = str(data_dir) + '/' + database
        with gbd_tool.gbd_api.GbdApi(db_string=db_path) as api:
            features = api.get_features(path=db_path)
            features.remove('hash')  # is key of all other tables anyway
            dataset = None
            # Getting multiple features at same time (with join in database) also possible,
            # but breaks if too many features and is actually slower than join via pandas
            for feature in features:
                feature_table = pd.DataFrame(api.query_search(query=None, resolve=[feature]), columns=['hash', feature])
                if dataset is None:
                    dataset = feature_table
                else:
                    dataset = dataset.merge(feature_table, on='hash', how='outer', copy=False)
            dataset.to_csv(data_dir / database.replace('.db', '.csv'), index=False)


# Create file with runtimes and file with instance features. To that end, do some pre-processing.
def transform_csvs_for_pipeline(data_dir: pathlib.Path) -> None:
    meta_db = pd.read_csv(data_dir / 'meta.csv')
    hashes = meta_db.loc[meta_db['competition_track'].fillna('').str.contains('main_2020'), 'hash']

    runtimes = pd.read_csv(data_dir / 'sc2020.csv').drop(columns=['tags', 'filename', 'local'])
    runtimes = runtimes[runtimes['hash'].isin(hashes)].reset_index(drop=True)
    runtimes.sort_values(by='hash', inplace=True)
    numeric_cols = [x for x in runtimes.columns if x != 'hash']
    runtimes[numeric_cols] = runtimes[numeric_cols].transform(pd.to_numeric, errors='coerce')
    runtimes.fillna(value=PENALTY, inplace=True)
    runtimes.to_csv(data_dir / 'runtimes.csv', index=False)

    gates_db = pd.read_csv(data_dir / 'gates.csv').drop(columns=['local', 'filename', 'tags', 'variables', 'clauses'])
    gates_db = gates_db[gates_db['hash'].isin(hashes)].reset_index(drop=True)
    satzilla_db = pd.read_csv(data_dir / 'satzilla.csv').drop(columns=['local', 'filename', 'tags'])
    satzilla_db = satzilla_db[satzilla_db['hash'].isin(hashes)].reset_index(drop=True)
    features = satzilla_db.merge(gates_db, on='hash')
    features.sort_values(by='hash', inplace=True)
    assert (runtimes['hash'] == features['hash']).all()
    numeric_cols = [x for x in features.columns if x != 'hash']
    features[numeric_cols] = features[numeric_cols].transform(pd.to_numeric, errors='coerce')
    features.to_csv(data_dir / 'features.csv', index=False)


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
        'and stores them.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--directory', type=pathlib.Path, default='data/', dest='data_dir',
                        help='Output directory for data.')
    print('Dataset preparation started.')
    prepare_dataset(**vars(parser.parse_args()))
    print('Dataset prepared and saved.')
