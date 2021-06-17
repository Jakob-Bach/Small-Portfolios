"""Run evaluation

Evaluation pipeline, creating plots for the paper and printing interesting statistics.
Should be run after the experimental pipeline.

Usage: python -m run_evaluation --help
"""

import argparse
import ast
import math
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import prepare_dataset


plt.rcParams['font.family'] = 'Helvetica'  # IEEE template's sans-serif font


# Run the full evaluation pipeline. To that end, read experiments' input files from "data_dir",
# experiments' results files from the "results_dir" and save plots to the "plot_dir".
# Print some statistics to the console.
def evaluate(data_dir: pathlib.Path, results_dir: pathlib.Path, plot_dir: pathlib.Path) -> None:
    if not plot_dir.is_dir():
        print('Plot directory does not exist. We create it.')
        plot_dir.mkdir(parents=True)
    if len(list(plot_dir.glob('*.pdf'))) > 0:
        print('Plot directory is not empty. Files might be overwritten, but not deleted.')

    # Load results, make solvers a list again:
    search_results = pd.read_csv(results_dir / 'search_results.csv',
                                 converters={'solvers': ast.literal_eval})
    # Fix k for beam search (beam search run up to k, but also saves smaller intermediate results)
    search_results.loc[search_results['algorithm'] == 'beam_search', 'k'] =\
        search_results.loc[search_results['algorithm'] == 'beam_search', 'solvers'].transform(len)

    # Load runtimes, which we need for beam-search bounds and single-solver analysis
    runtimes, _ = prepare_dataset.load_dataset(data_dir=data_dir)

    # Load prediction results
    prediction_results = pd.read_csv(results_dir / 'prediction_results.csv')
    prediction_results = prediction_results.merge(search_results)

    # ------Optimization Results------

    # ----Performance of Single Solvers----

    print('How often is a solver fastest?')
    print(runtimes.idxmin(axis='columns').value_counts())
    print('How many instances does each solver *not* solve?')
    print((runtimes == prepare_dataset.PENALTY).sum(axis='rows').sort_values())

    # ----Objective Value of Portfolios----

    # --Exhaustive Search--

    print('Table 1: Objective value for exhaustive search:')
    data = search_results[search_results['algorithm'] == 'exhaustive_search']
    data = data.groupby(['problem', 'k'])['objective_value'].describe().round().astype(int)
    print(data[['min', 'mean', 'max', 'std']].to_latex())

    # --Exact Search and Beam Search--

    # Figure 1: exact search, beam search with w=1, and submodularity bounds
    beam_data = search_results.loc[(search_results['algorithm'] == 'beam_search') & (search_results['w'] == 1),
                                   ['problem', 'algorithm', 'k', 'objective_value']]
    mip_data = search_results.loc[search_results['algorithm'] == 'mip_search',
                                  ['problem', 'algorithm', 'k', 'objective_value']]
    bound_data = mip_data.copy()
    bound_data['algorithm'] = 'upper_bound'
    c_w = runtimes.max(axis='columns').sum()  # VWS performance for PAR2
    bound_data.loc[bound_data['problem'] == 'PAR2', 'objective_value'] = c_w / math.e +\
        (1 - 1 / math.e) * bound_data.loc[bound_data['problem'] == 'PAR2', 'objective_value']
    c_w = (runtimes == prepare_dataset.PENALTY).astype(int).max(axis='columns').sum()  # for solved
    bound_data.loc[bound_data['problem'] == 'solved', 'objective_value'] = c_w / math.e +\
        (1 - 1 / math.e) * bound_data.loc[bound_data['problem'] == 'solved', 'objective_value']
    data = pd.concat([beam_data, mip_data, bound_data]).reset_index(drop=True)
    data['k_objective_frac'] = data.groupby(['problem', 'k'])['objective_value'].apply(lambda x: x / x.min())
    data['k_objective_diff'] = data.groupby(['problem', 'k'])['objective_value'].apply(lambda x: x - x.min())
    data['objective_frac'] = data.groupby('problem')['objective_value'].apply(lambda x: x / x.min())
    data['objective_diff'] = data.groupby('problem')['objective_value'].apply(lambda x: x - x.min())
    # Division might introduce NA or inf if objective is 0 (happens if all instances solved):
    data['k_objective_frac'] = data['k_objective_frac'].replace([float('nan'), float('inf')], 1)
    data['objective_frac'] = data['objective_frac'].replace([float('nan'), float('inf')], 1)
    plt.figure(figsize=(4, 3))
    sns.lineplot(x='k', y='objective_value', hue='algorithm', data=data[data['problem'] == 'PAR2'])
    plt.tight_layout()
    plt.savefig(plot_dir / 'objective-PAR2.pdf')
    plt.figure(figsize=(4, 3))
    sns.lineplot(x='k', y='objective_value', hue='algorithm', data=data[data['problem'] == 'solved'])
    plt.tight_layout()
    plt.savefig(plot_dir / 'objective-solved.pdf')

    print('Ratio of PAR2 between best k-portfolio and best portfolio of all solvers:')
    print(data.loc[(data['problem'] == 'PAR2') & (data['algorithm'] == 'mip_search'),
                   ['k', 'objective_frac']].round(2))
    print('How many instances remain unsolved in best k-portfolio??')
    print(data.loc[(data['problem'] == 'solved') & (data['algorithm'] == 'mip_search'),
                   ['k', 'objective_value']])
    print('Ratio of PAR2 value between best greedy-search-portfolio and exact solution:')
    print(data.loc[(data['problem'] == 'PAR2') & (data['algorithm'] == 'beam_search'),
                   ['k', 'k_objective_frac']].round(3))
    print('Difference in solved instances between best greedy-search-portfolio and exact solution:')
    print(data.loc[(data['problem'] == 'solved') & (data['algorithm'] == 'beam_search'),
                   ['k', 'k_objective_diff']].round(3))

    w = 10
    beam_data = search_results[(search_results['algorithm'] == 'beam_search') & (search_results['w'] == w)]
    data = pd.concat([beam_data, mip_data]).reset_index(drop=True)
    data['k_objective_frac'] = data.groupby(['problem', 'k'])['objective_value'].apply(lambda x: x / x.min())
    data['k_objective_diff'] = data.groupby(['problem', 'k'])['objective_value'].apply(lambda x: x - x.min())
    # Division might introduce NA or inf if objective is 0 (happens if all instances solved):
    data['k_objective_frac'] = data['k_objective_frac'].replace([float('nan'), float('inf')], 1)
    print(f'Ratio of PAR2 value between best {w=} beam-search-portfolio and exact solution:')
    print(data.loc[(data['problem'] == 'PAR2') & (data['algorithm'] == 'beam_search')].groupby('k')['k_objective_frac'].min().round(3))
    print(f'Difference in solved instances between best {w=} greedy-search-portfolio and exact solution:')
    print(data.loc[(data['problem'] == 'solved') & (data['algorithm'] == 'beam_search')].groupby('k')['k_objective_diff'].min().round(3))
    print(f'Objective value of top {w=} portfolios in beam search')
    print(data[data['k'] <= 10].groupby(['problem', 'k'])['objective_value'].describe().round().fillna(0).astype(int))

    # ----Solvers in Portfolio----

    print('How many solver changes are there from k-1 to k in exact search?')
    data = search_results[search_results['algorithm'] == 'mip_search'].copy()
    data['prev_solvers'] = data.groupby('problem')['solvers'].shift().fillna('').apply(list)
    data['solvers_added'] = data.apply(lambda x: len(set(x['solvers']) - set(x['prev_solvers'])), axis='columns')
    data['solvers_deleted'] = data.apply(lambda x: len(set(x['prev_solvers']) - set(x['solvers'])), axis='columns')
    data['solver_changes'] = data['solvers_added'] + data['solvers_deleted']
    print(data.loc[data['k'] <= 10, ['problem', 'k', 'solvers_added', 'solvers_deleted', 'solver_changes']])

    w = search_results['w'].max()
    print(f'Frequency of the respective most frequent solver in the top {w=} portfolios in beam search')
    data = search_results[(search_results['algorithm'] == 'beam_search') & (search_results['w'] == w)].copy()
    # Need to take care of solvers which do not appear in any portfolio; add them to data by re-indexing:
    data = data[['problem', 'solvers', 'k']].explode('solvers').value_counts()
    new_index = pd.MultiIndex.from_product(
        [data.index.get_level_values('problem').unique(), runtimes.columns, range(1, 49)],
        names=['problem', 'solvers', 'k'])
    data = data.reindex(new_index).reset_index().rename(columns={0: 'occurrence'}).fillna(0)
    data['occurrence'] = data.groupby(['problem', 'k'])['occurrence'].transform(lambda x: x / x.sum())
    print(data[data['k'] <= 20].groupby(['problem', 'k'])['occurrence'].max().round(3))

    k = 4
    print(f'How is solver occurrence in {k=}-portfolio correlated to objective value?')
    data = search_results[(search_results['algorithm'] == 'exhaustive_search') &
                          (search_results['k'] == k)].copy()
    for solver_name in runtimes.columns:
        data[solver_name] = data['solvers'].apply(lambda x: solver_name in x)  # is solver in portfolio?
    for problem in ['PAR2', 'solved']:
        print(f'- {problem}:')
        print(data.loc[data['problem'] == problem, runtimes.columns].corrwith(
            data.loc[data['problem'] == problem, 'objective_value'], method='spearman').describe().round(2))

    # ------Prediction Results------

    # ----MCC----

    # Figure 2: MCC for top beam-search portfolios per k
    w = search_results['w'].max()
    data = prediction_results.loc[(prediction_results['algorithm'] == 'beam_search') &
                                  (prediction_results['w'] == w)]
    data = data.loc[(data['k'] > 1) & (data['k'] <= 20) & (data['tree_depth'] == -1), ['problem', 'k', 'test_mcc']]
    plt.figure(figsize=(4, 3))
    sns.boxplot(x='k', y='test_mcc', hue='problem', fliersize=0, data=data)
    plt.tight_layout()
    plt.savefig(plot_dir / 'mcc.pdf')

    print('Median MCC per tree depth, using all prediction results:')
    print(prediction_results.groupby(['problem', 'tree_depth'])[['train_mcc', 'test_mcc']].median().round(2))
    data = prediction_results[['problem', 'tree_depth', 'train_mcc', 'test_mcc']].copy()
    print('Train-test MCC difference per tree depth, using all prediction results:')
    data['train_test_diff'] = data['train_mcc'] - data['test_mcc']
    print(data.groupby(['problem', 'tree_depth'])['train_test_diff'].describe().round(2))

    # ----Objective Value----

    # Figure 3: Objective value vs. VBS and VWS for model-based top beam-search portfolios
    w = search_results['w'].max()
    data = prediction_results.loc[(prediction_results['algorithm'] == 'beam_search') &
                                  (prediction_results['w'] == w)]
    plot_vars = ['test_objective', 'test_vbs', 'test_vws']
    data = data.loc[(data['k'] != 1) & (data['k'] <= 10) & (data['tree_depth'] == -1), ['problem', 'k'] + plot_vars]
    data = data.melt(id_vars=['problem', 'k'], value_vars=plot_vars,
                     var_name='score', value_name='objective')
    plt.figure(figsize=(4, 3))
    sns.boxplot(x='k', y='objective', hue='score', data=data[data['problem'] == 'PAR2'])
    plt.tight_layout()
    plt.savefig(plot_dir / 'objective-prediction-PAR2.pdf')
    plt.figure(figsize=(4, 3))
    sns.boxplot(x='k', y='objective', hue='score', data=data[data['problem'] == 'solved'])
    plt.tight_layout()
    plt.savefig(plot_dir / 'objective-prediction-solved.pdf')

    # ----Feature Importance----

    print('Average feature importance (in %) over all prediction scenarios:')
    importance_cols = [x for x in prediction_results.columns if x.startswith('imp.')]
    data = prediction_results[importance_cols].mean() * 100  # importance as percentage
    print(data.describe())
    print(f'To reach an importance of 50%, one needs {sum(data.sort_values(ascending=False).cumsum() < 50) + 1} features.')
    print('How many features are used in each model?')
    print((prediction_results[importance_cols] > 0).sum(axis='columns').describe().round(2))
    print('How many features are used in each model of unlimited depth?')
    print((prediction_results.loc[prediction_results['tree_depth'] == -1, importance_cols] > 0).sum(
        axis='columns').describe().round(2))


# Parse some command line argument and run evaluation.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates the paper\'s plots and prints statistics.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='data/', dest='data_dir',
                        help='Directory with input data, i.e., runtimes and instance features.')
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/',
                        dest='results_dir', help='Directory with experimental results.')
    parser.add_argument('-p', '--plots', type=pathlib.Path, default='../text/plots/',
                        dest='plot_dir', help='Output directory for plots.')
    print('Evaluation started.')
    evaluate(**vars(parser.parse_args()))
    print('Plots created and saved.')
