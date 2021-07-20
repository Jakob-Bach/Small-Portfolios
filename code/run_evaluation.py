"""Run evaluation

Evaluation pipeline, creating plots for the paper and printing statistics which are used in the
paper as well. Should be run after the experimental pipeline.

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
    # Fix k for beam and k-best search (run up to k, but also save smaller intermediate results)
    search_results.loc[search_results['algorithm'] == 'beam_search', 'k'] =\
        search_results.loc[search_results['algorithm'] == 'beam_search', 'solvers'].transform(len)
    search_results.loc[search_results['algorithm'] == 'kbest_search', 'k'] =\
        search_results.loc[search_results['algorithm'] == 'kbest_search', 'solvers'].transform(len)

    # Load runtimes, which we need for beam-search bounds and single-solver analysis
    runtimes2020, _ = prepare_dataset.load_dataset(dataset_name='sc2020', data_dir=data_dir)
    runtimes2021, _ = prepare_dataset.load_dataset(dataset_name='sc2021', data_dir=data_dir)

    # Load prediction results
    prediction_results = pd.read_csv(results_dir / 'prediction_results.csv')
    prediction_results = prediction_results.merge(search_results)

    # ------Optimization Results------

    # ----Performance of Single Solvers----

    print('How often is a solver fastest in SC2020?')
    print(runtimes2020.idxmin(axis='columns').value_counts())
    print('How often is a solver fastest in SC2021?')
    print(runtimes2021.idxmin(axis='columns').value_counts())

    # ----Overall Trend / Test-Set Performance----

    # Figures 2 and 3: Performance of search approaches over k (Figure 1 is pseudo-code, not created here)
    data = search_results.loc[(search_results['algorithm'] != 'beam_search') | (search_results['w'] == 1)]
    bound_data = data[data['algorithm'] == 'mip_search'].copy()  # bounds computed from exact solution
    bound_data['algorithm'] = 'upper_bound'
    for problem in data['problem'].unique():
        bound_data.loc[bound_data['problem'] == problem, 'train_objective'] =\
            bound_data.loc[bound_data['problem'] == problem, 'train_global_sws'] / math.e +\
                (1 - 1 / math.e) * bound_data.loc[bound_data['problem'] == problem, 'train_objective']
        bound_data.loc[bound_data['problem'] == problem, 'test_objective'] =\
            bound_data.loc[bound_data['problem'] == problem, 'test_global_sws'] / math.e +\
                (1 - 1 / math.e) * bound_data.loc[bound_data['problem'] == problem, 'test_objective']
    data = pd.concat([data, bound_data]).reset_index(drop=True)
    data['k_objective_frac'] = data.groupby(['problem', 'k', 'fold_id'])['train_objective'].apply(lambda x: x / x.min())
    data['objective_frac'] = data.groupby(['problem', 'fold_id'])['train_objective'].apply(lambda x: x / x.min())
    plot_data = data.groupby(['problem', 'algorithm', 'k'])[['train_objective', 'test_objective']].mean().reset_index()
    plot_data['algorithm'] = plot_data['algorithm'].replace({
        'random_search': 'Random sampling', 'mip_search': 'Optimal solution',
        'beam_search': 'Greedy search', 'kbest_search': 'K-best', 'upper_bound': 'Upper bound'})
    plot_data.rename(columns={'algorithm': 'Solution approach', 'k': 'Portfolio size $k$'}, inplace=True)
    # Figure 2 (a)
    plt.figure(figsize=(4, 3))
    sns.lineplot(x='Portfolio size $k$', y='train_objective',
                 hue='Solution approach', style='Solution approach',
                 data=plot_data[plot_data['problem'] == 'sc2020'], palette='Set1')
    plt.ylabel('PAR-2 score')
    plt.legend(edgecolor='white', loc='center right', framealpha=0, bbox_to_anchor=(1, 0.4))
    plt.tight_layout()
    plt.savefig(plot_dir / 'search-train-objective-2020.pdf')
    # Figure 2 (b)
    plt.figure(figsize=(4, 3))
    sns.lineplot(x='Portfolio size $k$', y='train_objective',
                 hue='Solution approach', style='Solution approach',
                 data=plot_data[plot_data['problem'] == 'sc2021'], palette='Set1')
    plt.ylabel('PAR-2 score')
    plt.legend(edgecolor='white', loc='center right', framealpha=0, bbox_to_anchor=(1, 0.4))
    plt.tight_layout()
    plt.savefig(plot_dir / 'search-train-objective-2021.pdf')
    # Figure 3
    plt.figure(figsize=(4, 3))
    sns.lineplot(x='Portfolio size $k$', y='test_objective',
                 hue='Solution approach', style='Solution approach',
                 data=plot_data[plot_data['problem'] == 'sc2020'], palette='Set1')
    plt.ylabel('PAR-2 score')
    plt.legend(edgecolor='white', loc='center right', framealpha=0, bbox_to_anchor=(1, 0.4))
    plt.tight_layout()
    plt.savefig(plot_dir / 'search-test-objective-2020.pdf')

    print('Ratio of PAR2 between best k-portfolio and best portfolio of all solvers:')
    print(data.loc[data['algorithm'] == 'mip_search',  ['problem', 'k', 'objective_frac']].groupby(
        ['problem', 'k']).mean().reset_index().pivot(index='k', columns='problem').round(2))  # mean over folds

    # ----Beam Search----

    print('Ratio of PAR2 value between greedy-search/k-best and exact solution:')
    print(data.loc[data['algorithm'] == 'beam_search', ['problem', 'k', 'k_objective_frac']].groupby(
        ['problem', 'k']).mean().reset_index().pivot(index='k', columns='problem').round(3))

    w = 10
    data = search_results[(search_results['algorithm'] == 'mip_search') |
                          ((search_results['algorithm'] == 'beam_search') & (search_results['w'] == w))].copy()
    data['k_objective_frac'] = data.groupby(['problem', 'k', 'fold_id'])['train_objective'].apply(lambda x: x / x.min())
    print(f'Ratio of PAR2 value between best {w=} beam-search-portfolio and exact solution:')
    # Need to pick optimal portfolio (min out of w portfolios) for each k first, then average over folds
    print(data.loc[data['algorithm'] == 'beam_search', ['problem', 'k', 'fold_id', 'k_objective_frac']].groupby(
        ['problem', 'k', 'fold_id'])['k_objective_frac'].min().groupby(['problem', 'k']).mean(
            ).reset_index().pivot(index='k', columns='problem').round(3))

    print(f'Standard deviation of objective value of top {w=} portfolios in beam search:')
    print(data.loc[data['algorithm'] == 'beam_search'].groupby(['problem', 'k', 'fold_id'])['train_objective'].std(
        ).groupby(['problem', 'k']).mean().reset_index().pivot(index='k', columns='problem').round(2))

    print('Standard deviation of objective value for random search:')
    data = search_results[search_results['algorithm'] == 'random_search']
    print(data.groupby(['problem', 'k', 'fold_id'])['train_objective'].std().groupby(['problem', 'k']).mean(
        ).reset_index().pivot(columns='problem', index='k').round(2))

    # ----Portfolio Composition----

    data = search_results[search_results['algorithm'] == 'mip_search'].copy()
    data['prev_solvers'] = data.groupby(['problem', 'fold_id'])['solvers'].shift().fillna('').apply(list)
    data['solvers_added'] = data.apply(lambda x: len(set(x['solvers']) - set(x['prev_solvers'])), axis='columns')
    data['solvers_deleted'] = data.apply(lambda x: len(set(x['prev_solvers']) - set(x['solvers'])), axis='columns')

    print('How many solver changes are there from k-1 to k in exact search?')
    print(data.loc[data['k'] <= 20].groupby(['problem', 'k'])[
        ['solvers_added', 'solvers_deleted']].mean())

    print('How many solver changes are there from k-1 to k in exact search, aggregating over k?')
    print(data.groupby('problem')[['solvers_added', 'solvers_deleted']].describe().transpose())

    # ----Impact of single solvers on Portfolios----

    k = 5
    print(f'How is solver occurrence in random {k=}-portfolio correlated to objective value?')
    for problem, problem_runtimes in zip(['sc2020', 'sc2021'], [runtimes2020, runtimes2021]):
        data = search_results[(search_results['problem'] == problem) &
                              (search_results['algorithm'] == 'random_search') & (search_results['k'] == k)].copy()
        for solver_name in problem_runtimes.columns:
            data[solver_name] = data['solvers'].apply(lambda x: solver_name in x)  # is solver in portfolio?
        print(f'- {problem}:')
        print(data[problem_runtimes.columns].corrwith(data['train_objective'], method='spearman').describe().round(2))

    # ------Prediction Results------

    # ----MCC----

    # Figure 4: MCC for random portfolios per k
    data = prediction_results.loc[(prediction_results['algorithm'] == 'random_search')]
    data = data.loc[(data['k'] > 1) & (data['k'] <= 10) & (data['model'] == 'Random forest') &
                    (data['n_estimators'] == 100), ['problem', 'k', 'solution_id', 'test_pred_mcc']]
    # Aggregate over cross-validation folds:
    data = data.groupby(['problem', 'k', 'solution_id']).mean().reset_index().drop(columns='solution_id')
    data['problem'] = data['problem'].replace({'sc2020': 'SAT Competition 2020',
                                               'sc2021': 'SAT Competition 2021'})
    data.rename(columns={'problem': 'Objective', 'k': 'Portfolio size $k$',
                         'test_pred_mcc': 'Test-set MCC'}, inplace=True)
    plt.figure(figsize=(4, 3))
    sns.boxplot(x='Portfolio size $k$', y='Test-set MCC', hue='Objective', fliersize=0, data=data,
                palette='Set2')
    plt.ylim(-0.1, 1)
    plt.legend(edgecolor='white')
    plt.tight_layout()
    plt.savefig(plot_dir / 'prediction-test-mcc.pdf')

    print('Mean MCC for beam search with w=100, random forest with 100 trees:')
    data = prediction_results[(prediction_results['algorithm'] == 'beam_search') &
                              (prediction_results['w'] == 100) &
                              (prediction_results['model'] == 'Random forest') &
                              (prediction_results['n_estimators'] == 100)]
    print(data.groupby(['problem', 'k'])['test_pred_mcc'].agg(['mean', 'std']).reset_index().pivot(
        index='k', columns='problem').round(2))

    print('Mean MCC for exact search, random forest with 100 trees:')
    data = prediction_results[(prediction_results['algorithm'] == 'mip_search') &
                              (prediction_results['model'] == 'Random forest') &
                              (prediction_results['n_estimators'] == 100)]
    print(data.groupby(['problem', 'k'])['test_pred_mcc'].agg(['mean', 'std']).reset_index().pivot(
        index='k', columns='problem').round(2))

    print('Median MCC per model and number of estimators, using all prediction results:')
    print(prediction_results.groupby(['problem', 'model', 'n_estimators'])[
        ['train_pred_mcc', 'test_pred_mcc']].median().round(2))

    print('Train-test MCC difference per model and number of estimators, using all prediction results:')
    data = prediction_results[['problem', 'model', 'n_estimators', 'train_pred_mcc', 'test_pred_mcc']].copy()
    data['train_test_diff'] = data['train_pred_mcc'] - data['test_pred_mcc']
    print(data.groupby(['problem', 'model', 'n_estimators'])['train_test_diff'].describe().round(2))

    # ----Objective Value----

    # Figure 5: Objective value for model-based and VBS-based top beam-search portfolios
    w = 100
    data = prediction_results.loc[(prediction_results['algorithm'] == 'beam_search') &
                                  (prediction_results['w'] == w)]
    plot_vars = ['test_pred_objective', 'test_objective', 'test_portfolio_sbs']
    data = data.loc[(data['k'] != 1) & (data['k'] <= 10) & (data['model'] == 'Random forest') &
                    (data['n_estimators'] == 100), ['problem', 'k', 'solution_id'] + plot_vars]
    # Aggregate over cross-validation folds (don't group by k here, as solution_ids for beam search
    # go over mutliple values of k):
    data = data.groupby(['problem', 'solution_id']).mean().reset_index().drop(columns='solution_id')
    data = data.melt(id_vars=['problem', 'k'], value_vars=plot_vars,
                     var_name='Approach', value_name='objective')
    data['Approach'] = data['Approach'].replace(
        {'test_pred_objective': 'Prediction', 'test_objective': 'VBS', 'test_portfolio_sbs': 'SBS'})
    data.rename(columns={'k': 'Portfolio size $k$'}, inplace=True)
    global_sbs_data = search_results.loc[(search_results['algorithm'] == 'mip_search') &
                                         (search_results['k'] == 1), ['problem', 'fold_id', 'test_objective']]
    # Figure 5 (a)
    plt.figure(figsize=(4, 3))
    sns.boxplot(x='Portfolio size $k$', y='objective', hue='Approach',
                data=data[data['problem'] == 'sc2020'], palette='Set2')
    plt.axhline(y=global_sbs_data.loc[global_sbs_data['problem'] == 'sc2020', 'test_objective'].mean(),
                color=sns.color_palette('Set2').as_hex()[3])  # SBS
    plt.ylabel('PAR-2 score')
    plt.legend(edgecolor='white', loc='center right', framealpha=0, bbox_to_anchor=(1, 0.35))
    plt.tight_layout()
    plt.savefig(plot_dir / 'prediction-test-objective-beam-2020.pdf')
    # Figure 5 (b)
    plt.figure(figsize=(4, 3))
    sns.boxplot(x='Portfolio size $k$', y='objective', hue='Approach',
                data=data[data['problem'] == 'sc2021'], palette='Set2')
    plt.axhline(y=global_sbs_data.loc[global_sbs_data['problem'] == 'sc2021', 'test_objective'].mean(),
                color=sns.color_palette('Set2').as_hex()[3])  # SBS
    plt.ylabel('PAR-2 score')
    plt.legend(edgecolor='white', loc='center right', framealpha=0, bbox_to_anchor=(1, 0.25))
    plt.tight_layout()
    plt.savefig(plot_dir / 'prediction-test-objective-beam-2021.pdf')

    # Figure 6: Objective value for optimal (MIP search) portfolios
    data = prediction_results.loc[(prediction_results['algorithm'] == 'mip_search')]
    data = data.loc[(data['k'] != 1) & (data['k'] <= 10) & (data['model'] == 'Random forest') &
                    (data['n_estimators'] == 100), ['problem', 'k', 'solution_id'] + plot_vars]
    data = data.melt(id_vars=['problem', 'k'], value_vars=plot_vars,
                     var_name='Approach', value_name='objective')
    data['Approach'] = data['Approach'].replace(
        {'test_pred_objective': 'Prediction', 'test_objective': 'VBS', 'test_portfolio_sbs': 'SBS'})
    data.rename(columns={'k': 'Portfolio size $k$'}, inplace=True)
    plt.figure(figsize=(4, 3))
    sns.stripplot(x='Portfolio size $k$', y='objective', hue='Approach',
                    data=data[data['problem'] == 'sc2020'], palette='Set2', dodge=True)
    plt.axhline(y=global_sbs_data.loc[global_sbs_data['problem'] == 'sc2020', 'test_objective'].mean(),
                color=sns.color_palette('Set2').as_hex()[3])  # SBS
    plt.ylabel('PAR-2 score')
    plt.legend(edgecolor='white',  loc='lower left', bbox_to_anchor=(0, 1), ncol=3)
    plt.tight_layout()
    plt.savefig(plot_dir / 'prediction-test-objective-optimal-2020.pdf')

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
