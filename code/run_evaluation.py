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
import run_experiments


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
    # Fix k for beam and k-best search (run up to k, but also save smaller intermediate results)
    search_results.loc[search_results['algorithm'] == 'beam_search', 'k'] =\
        search_results.loc[search_results['algorithm'] == 'beam_search', 'solvers'].transform(len)
    search_results.loc[search_results['algorithm'] == 'kbest_search', 'k'] =\
        search_results.loc[search_results['algorithm'] == 'kbest_search', 'solvers'].transform(len)

    # Load runtimes, which we need for beam-search bounds and single-solver analysis
    runtimes, _ = prepare_dataset.load_dataset(data_dir=data_dir)
    problems = run_experiments.define_problems(runtimes=runtimes)

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

    # Figure 2: Performance of search approaches over k (Figure 1 is pseudo-code, not created here)
    data = search_results.loc[(search_results['algorithm'] != 'beam_search') | (search_results['w'] == 1)]
    bound_data = data[data['algorithm'] == 'mip_search'].copy()
    bound_data['algorithm'] = 'upper_bound'
    for problem in problems.keys():
        bound_data.loc[bound_data['problem'] == problem, 'train_objective'] =\
            bound_data.loc[bound_data['problem'] == problem, 'train_vws'] / math.e +\
                (1 - 1 / math.e) * bound_data.loc[bound_data['problem'] == problem, 'train_objective']
        bound_data.loc[bound_data['problem'] == problem, 'test_objective'] =\
            bound_data.loc[bound_data['problem'] == problem, 'test_vws'] / math.e +\
                (1 - 1 / math.e) * bound_data.loc[bound_data['problem'] == problem, 'test_objective']
    data = pd.concat([data, bound_data]).reset_index(drop=True)
    data['k_objective_frac'] = data.groupby(['problem', 'k', 'fold_id'])['train_objective'].apply(lambda x: x / x.min())
    data['k_objective_diff'] = data.groupby(['problem', 'k', 'fold_id'])['train_objective'].apply(lambda x: x - x.min())
    data['objective_frac'] = data.groupby(['problem', 'fold_id'])['train_objective'].apply(lambda x: x / x.min())
    plt.figure(figsize=(4, 3))
    sns.lineplot(x='k', y='train_objective', hue='algorithm', data=data[data['problem'] == 'PAR2'], ci=None)
    plt.tight_layout()
    plt.savefig(plot_dir / 'search-train-objective-PAR2.pdf')
    plt.figure(figsize=(4, 3))
    sns.lineplot(x='k', y='test_objective', hue='algorithm', data=data[data['problem'] == 'PAR2'], ci=None)
    plt.tight_layout()
    plt.savefig(plot_dir / 'search-test-objective-PAR2.pdf')

    print('Ratio of PAR2 between best k-portfolio and best portfolio of all solvers:')
    print(data.loc[(data['problem'] == 'PAR2') & (data['algorithm'] == 'mip_search'),
                   ['k', 'objective_frac']].groupby('k').mean().round(2))  # mean over folds
    print('Which fraction of instances remains unsolved in best k-portfolio?')
    print(data.loc[(data['problem'] == 'Unsolved') & (data['algorithm'] == 'mip_search'),
                   ['k', 'train_objective']].groupby('k').mean())  # mean over folds
    print('Ratio of PAR2 value between greedy-search/k-best and exact solution:')
    print(data.loc[(data['problem'] == 'PAR2') & (data['algorithm'].isin(['beam_search', 'kbest_search']))].groupby(
        ['algorithm', 'k'])['k_objective_frac'].mean().reset_index().pivot(index='k', columns='algorithm').round(3))
    print('Difference in fraction of unsolved instances between greedy-search/k-best and exact solution:')
    print(data.loc[(data['problem'] == 'Unsolved') & (data['algorithm'].isin(['beam_search', 'kbest_search']))].groupby(
        ['algorithm', 'k'])['k_objective_diff'].mean().reset_index().pivot(index='k', columns='algorithm').round(3))

    w = 10
    data = search_results[(search_results['algorithm'] == 'mip_search') |
                          ((search_results['algorithm'] == 'beam_search') & (search_results['w'] == w))].copy()
    data['k_objective_frac'] = data.groupby(['problem', 'k', 'fold_id'])['train_objective'].apply(lambda x: x / x.min())
    data['k_objective_diff'] = data.groupby(['problem', 'k', 'fold_id'])['train_objective'].apply(lambda x: x - x.min())
    print(f'Ratio of PAR2 value between best {w=} beam-search-portfolio and exact solution:')
    print(data.loc[(data['problem'] == 'PAR2') & (data['algorithm'] == 'beam_search')].groupby(
        ['k', 'fold_id'])['k_objective_frac'].min().groupby('k').mean().round(3))  # mean over folds
    print(f'Difference in unsolved instances between best {w=} beam-search-portfolio and exact solution:')
    print(data.loc[(data['problem'] == 'Unsolved') & (data['algorithm'] == 'beam_search')].groupby(
        ['k', 'fold_id'])['k_objective_diff'].min().groupby('k').mean().round(4))  # mean over folds
    print(f'Standard deviation of objective value of top {w=} portfolios in beam search:')
    print(data[data['k'] <= 10].groupby(['problem', 'k'])['train_objective'].std().round(3))
    print('Standard deviation of objective value for random search:')
    data = search_results[search_results['algorithm'] == 'random_search']
    print(data.groupby(['problem', 'k'])['train_objective'].std().reset_index().pivot(
        columns='problem', index='k').round(2))

    # ----Solvers in Portfolio----

    print('How many solver changes are there from k-1 to k in exact search?')
    data = search_results[search_results['algorithm'] == 'mip_search'].copy()
    data['prev_solvers'] = data.groupby(['problem', 'fold_id'])['solvers'].shift().fillna('').apply(list)
    data['solvers_added'] = data.apply(lambda x: len(set(x['solvers']) - set(x['prev_solvers'])), axis='columns')
    data['solvers_deleted'] = data.apply(lambda x: len(set(x['prev_solvers']) - set(x['solvers'])), axis='columns')
    data['solver_changes'] = data['solvers_added'] + data['solvers_deleted']
    print(data.loc[data['k'] <= 10].groupby(['problem', 'k'])[['solvers_added', 'solvers_deleted', 'solver_changes']].mean())

    w = 100
    print(f'Frequency of the respective most frequent solver in the top {w=} portfolios in beam search:')
    data = search_results[(search_results['algorithm'] == 'beam_search') & (search_results['w'] == w)].copy()
    # Need to take care of solvers which do not appear in any portfolio; add them to data by re-indexing:
    data = data[['problem', 'solvers', 'k']].explode('solvers').value_counts()
    new_index = pd.MultiIndex.from_product(
        [data.index.get_level_values('problem').unique(), runtimes.columns, range(1, 49)],
        names=['problem', 'solvers', 'k'])
    data = data.reindex(new_index).reset_index().rename(columns={0: 'occurrence'}).fillna(0)
    data['occurrence'] = data.groupby(['problem', 'k'])['occurrence'].transform(lambda x: x / x.sum())
    print(data[data['k'] <= 20].groupby(['problem', 'k'])['occurrence'].max().round(3))

    k = 5
    print(f'How is solver occurrence in random {k=}-portfolio correlated to objective value?')
    data = search_results[(search_results['algorithm'] == 'random_search') &
                          (search_results['k'] == k)].copy()
    for solver_name in runtimes.columns:
        data[solver_name] = data['solvers'].apply(lambda x: solver_name in x)  # is solver in portfolio?
    for problem in problems.keys():
        print(f'- {problem}:')
        print(data.loc[data['problem'] == problem, runtimes.columns].corrwith(
            data.loc[data['problem'] == problem, 'train_objective'], method='spearman').describe().round(2))

    # ------Prediction Results------

    # ----MCC----

    # Figure 3: MCC for random portfolios per k
    data = prediction_results.loc[(prediction_results['algorithm'] == 'random_search')]
    data = data.loc[(data['k'] > 1) & (data['k'] <= 10) & (data['model'] == 'Random forest') &
                    (data['n_estimators'] == 100), ['problem', 'k', 'solution_id', 'test_pred_mcc']]
    # Aggregate over cross-validation folds:
    data = data.groupby(['problem', 'k', 'solution_id']).mean().reset_index().drop(columns='solution_id')
    plt.figure(figsize=(4, 3))
    sns.boxplot(x='k', y='test_pred_mcc', hue='problem', fliersize=0, data=data)
    plt.tight_layout()
    plt.savefig(plot_dir / 'prediction-test-mcc.pdf')

    print('Median MCC per model and number of estimators, using all prediction results:')
    print(prediction_results.groupby(['problem', 'model', 'n_estimators'])[['train_pred_mcc', 'test_pred_mcc']].median().round(2))
    print('Train-test MCC difference per model and number of estimators, using all prediction results:')
    data = prediction_results[['problem', 'model', 'n_estimators', 'train_pred_mcc', 'test_pred_mcc']].copy()
    data['train_test_diff'] = data['train_pred_mcc'] - data['test_pred_mcc']
    print(data.groupby(['problem', 'model', 'n_estimators'])['train_test_diff'].describe().round(2))

    # ----Objective Value----

    # Figure 4: Objective value for model-based and VBS-based top beam-search portfolios
    w = 100
    data = prediction_results.loc[(prediction_results['algorithm'] == 'beam_search') &
                                  (prediction_results['w'] == w)]
    plot_vars = ['test_pred_objective', 'test_objective']
    data = data.loc[(data['k'] != 1) & (data['k'] <= 10) & (data['model'] == 'Random forest') &
                    (data['n_estimators'] == 100), ['problem', 'k', 'solution_id'] + plot_vars]
    # Aggregate over cross-validation folds (don't group by k here, as solution_ids for beam search
    # go over mutliple values of k):
    data = data.groupby(['problem', 'solution_id']).mean().reset_index().drop(columns='solution_id')
    data = data.melt(id_vars=['problem', 'k'], value_vars=plot_vars,
                     var_name='score', value_name='objective')
    plt.figure(figsize=(4, 3))
    sns.boxplot(x='k', y='objective', hue='score', data=data[data['problem'] == 'PAR2'])
    plt.tight_layout()
    plt.savefig(plot_dir / 'prediction-test-objective-PAR2.pdf')
    plt.figure(figsize=(4, 3))
    sns.boxplot(x='k', y='objective', hue='score', data=data[data['problem'] == 'Unsolved'])
    plt.tight_layout()
    plt.savefig(plot_dir / 'prediction-test-objective-unsolved.pdf')

    # ----Feature Importance----

    print('Average feature importance (in %) over all prediction scenarios:')
    importance_cols = [x for x in prediction_results.columns if x.startswith('imp.')]
    data = prediction_results[importance_cols].mean() * 100  # importance as percentage
    print(data.describe())
    print(f'To reach an importance of 50%, one needs {sum(data.sort_values(ascending=False).cumsum() < 50) + 1} features.')
    print('How many features are used in each model?')
    print((prediction_results[importance_cols] > 0).sum(axis='columns').describe().round(2))
    print('How many features are used in each model with one estimator?')
    print((prediction_results.loc[prediction_results['n_estimators'] == 1, importance_cols] > 0).sum(
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
