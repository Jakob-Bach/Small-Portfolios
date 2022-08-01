"""Run evaluation

Evaluation pipeline, creating plots for the paper and the conference presentation, and printing
statistics which are used in the paper as well. Should be run after the experimental pipeline.

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


plt.rcParams['font.family'] = 'Latin Modern Sans'  # LIPIcs template uses "Latin Modern" fonts
plt.rcParams['savefig.dpi'] = 300  # recommended by LIPIcs template


# Run the full evaluation pipeline. To that end, read experiments' input files from "data_dir",
# experiments' results files from the "results_dir" and save plots to the "paper_plot_dir" and the
# "presentation_plot_dir". Print some statistics to the console.
def evaluate(data_dir: pathlib.Path, results_dir: pathlib.Path, paper_plot_dir: pathlib.Path,
             presentation_plot_dir: pathlib.Path) -> None:
    if not paper_plot_dir.is_dir():
        print('Paper plot directory does not exist. We create it.')
        paper_plot_dir.mkdir(parents=True)
    if any(paper_plot_dir.glob('*.pdf')):
        print('Paper plot directory is not empty. Files might be overwritten, but not deleted.')
    if not presentation_plot_dir.is_dir():
        print('Presentation plot directory does not exist. We create it.')
        presentation_plot_dir.mkdir(parents=True)
    if any(presentation_plot_dir.glob('*.pdf')):
        print('Presentation plot directory is not empty. Files might be overwritten, but not deleted.')

    print('Loading the data (might take a while) ...')

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

    print('\n--------6 Evaluation--------')

    print('\n------6.1 Optimization Results------')

    print('\nOn how many instances is each solver fastest in SC2020?')
    print(runtimes2020.idxmin(axis='columns').value_counts())
    print('\nOn how many instances is each solver fastest in SC2021?')
    print(runtimes2021.idxmin(axis='columns').value_counts())

    # Figures 1 and 2: Objective value of search approaches over k
    data = search_results.loc[(search_results['algorithm'] != 'beam_search') | (search_results['w'] == 1)]
    bound_data = data[data['algorithm'] == 'mip_search'].copy()  # bounds computed from optimal solution
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
    plot_data.rename(columns={'algorithm': 'Solution approach', 'problem': 'Dataset'}, inplace=True)
    # Figure 1: Training-set objective value of search approaches over k
    for is_for_paper in (False, True):
        plt.rcParams['font.size'] = 24 if is_for_paper else 30
        plt.figure()
        facet_grid = sns.relplot(x='k', y='train_objective', col='Dataset', hue='Solution approach',
                                 style='Solution approach', data=plot_data, kind='line',
                                 linewidth=4, palette='Set1', facet_kws={'despine': False},
                                 height=6.25, aspect=0.8)
        facet_grid.set_axis_labels(x_var='Portfolio size $k$', y_var='PAR-2 score')
        if is_for_paper:
            sns.move_legend(facet_grid, edgecolor='white', loc='upper center',
                            bbox_to_anchor=(0.5, 0.1), ncol=3)
            plot_dir = paper_plot_dir
        else:
            sns.move_legend(facet_grid, edgecolor='white', loc='center left',
                            bbox_to_anchor=(1, 0.5), ncol=1)
            plt.xticks(range(0, 51, 10))
            plt.yticks(range(0, 5001, 1000))
            plot_dir = presentation_plot_dir
        for handle in facet_grid.legend.legendHandles:
            handle.set_linewidth(4)
        plt.tight_layout()
        plt.savefig(plot_dir / 'search-train-objective.pdf', bbox_inches='tight')
    # Figure 2: Test-set objective value of search approaches over k
    for is_for_paper in (False, True):
        plt.rcParams['font.size'] = 24 if is_for_paper else 30
        plt.figure()
        facet_grid = sns.relplot(x='k', y='test_objective', col='Dataset', hue='Solution approach',
                                 style='Solution approach', data=plot_data, kind='line',
                                 linewidth=4, palette='Set1', facet_kws={'despine': False},
                                 height=6.25, aspect=0.8)
        facet_grid.set_axis_labels(x_var='Portfolio size $k$', y_var='PAR-2 score')
        if is_for_paper:
            sns.move_legend(facet_grid, edgecolor='white', loc='upper center',
                            bbox_to_anchor=(0.5, 0.1), ncol=3)
            plot_dir = paper_plot_dir
        else:
            sns.move_legend(facet_grid, edgecolor='white', loc='center left',
                            bbox_to_anchor=(1, 0.5), ncol=1)
            plt.xticks(range(0, 51, 10))
            plt.yticks(range(0, 5001, 1000))
            plot_dir = presentation_plot_dir
        for handle in facet_grid.legend.legendHandles:
            handle.set_linewidth(4)
        plt.tight_layout()
        plt.savefig(plot_dir / 'search-test-objective.pdf', bbox_inches='tight')

    print('\n----6.1.1  Optimal Solution----')

    print('\nRatio of PAR2 between best k-portfolio and best portfolio of all solvers:')
    print(data.loc[data['algorithm'] == 'mip_search',  ['problem', 'k', 'objective_frac']].groupby(
        ['problem', 'k']).mean().reset_index().pivot(index='k', columns='problem').round(2))  # mean over folds

    print('\nHow is the optimization time for the integer problem distributed?')
    print(data.loc[data['algorithm'] == 'mip_search', 'search_time'].describe().round(2))

    print('\nHow is the maximum optimization time for the integer problem distributed over k?')
    print(data.loc[(data['algorithm'] == 'mip_search')].groupby('k')['search_time'].max().round(2))

    print('\nHow is the optimization time for the integer problem distributed for k <= 9?')
    print(data.loc[(data['algorithm'] == 'mip_search') & (data['k'] <= 9), 'search_time'].describe().round(2))

    print('\nHow is the optimization time for the integer problem distributed for k >= 10?')
    print(data.loc[(data['algorithm'] == 'mip_search') & (data['k'] >= 10), 'search_time'].describe().round(2))

    print('\n----6.1.2 Beam Search----')

    print('\nRatio of PAR2 value between greedy search and optimal solution:')
    print(data.loc[data['algorithm'] == 'beam_search', ['problem', 'k', 'k_objective_frac']].groupby(
        ['problem', 'k']).mean().reset_index().pivot(index='k', columns='problem').round(3))

    w = 10
    data = search_results[(search_results['algorithm'] == 'mip_search') |
                          ((search_results['algorithm'] == 'beam_search') & (search_results['w'] == w))].copy()
    data['k_objective_frac'] = data.groupby(['problem', 'k', 'fold_id'])['train_objective'].apply(lambda x: x / x.min())
    print(f'\nRatio of PAR2 value between best {w=} beam-search portfolio and optimal solution:')
    # Need to pick optimal portfolio (min out of w portfolios) for each k first, then average over folds
    print(data.loc[data['algorithm'] == 'beam_search', ['problem', 'k', 'fold_id', 'k_objective_frac']].groupby(
        ['problem', 'k', 'fold_id'])['k_objective_frac'].min().groupby(['problem', 'k']).mean(
            ).reset_index().pivot(index='k', columns='problem').round(3))

    print(f'\nStandard deviation of objective value of top {w=} portfolios in beam search:')
    print(data.loc[data['algorithm'] == 'beam_search'].groupby(['problem', 'k', 'fold_id'])['train_objective'].std(
        ).groupby(['problem', 'k']).mean().reset_index().pivot(index='k', columns='problem').round(2))

    print('\nStandard deviation of objective value for random search:')
    data = search_results[search_results['algorithm'] == 'random_search']
    print(data.groupby(['problem', 'k', 'fold_id'])['train_objective'].std().groupby(['problem', 'k']).mean(
        ).reset_index().pivot(columns='problem', index='k').round(2))

    print('\n----6.1.5 Portfolio Composition----')

    data = search_results[search_results['algorithm'] == 'mip_search'].copy()
    data['prev_solvers'] = data.groupby(['problem', 'fold_id'])['solvers'].shift().fillna('').apply(list)
    data['solvers_added'] = data.apply(lambda x: len(set(x['solvers']) - set(x['prev_solvers'])), axis='columns')
    data['solvers_deleted'] = data.apply(lambda x: len(set(x['prev_solvers']) - set(x['solvers'])), axis='columns')

    print('\nHow many solver changes are there from k-1 to k in the optimal solution?')
    print(data.loc[data['k'] <= 20].groupby(['problem', 'k'])[
        ['solvers_added', 'solvers_deleted']].mean())

    print('\nHow many solver changes are there from k-1 to k in the optimal solution, aggregating over k?')
    print(data.groupby('problem')[['solvers_added', 'solvers_deleted']].describe().round(2).transpose())

    print('\n----6.1.6 Impact of Single Solvers on Portfolios----')

    k = 5
    print(f'\nHow is solver occurrence in random {k=}-portfolio correlated to objective value?')
    for problem, problem_runtimes in zip(['SC2020', 'SC2021'], [runtimes2020, runtimes2021]):
        data = search_results[(search_results['problem'] == problem) &
                              (search_results['algorithm'] == 'random_search') & (search_results['k'] == k)].copy()
        for solver_name in problem_runtimes.columns:
            data[solver_name] = data['solvers'].apply(lambda x: solver_name in x)  # is solver in portfolio?
        print(f'- {problem}:')
        print(data[problem_runtimes.columns].corrwith(data['train_objective'], method='spearman').describe().round(2))

    print('\n------6.2 Prediction Results------')

    print('\n----6.2.1 Matthews Correlation Coefficient----')

    # Figure 3: MCC of predictions on random portfolios over k
    data = prediction_results.loc[(prediction_results['algorithm'] == 'random_search')]
    data = data.loc[(data['k'] > 1) & (data['k'] <= 10),
                    ['problem', 'k', 'model', 'solution_id', 'test_pred_mcc']]
    # Aggregate over cross-validation folds:
    data = data.groupby(['problem', 'k', 'model', 'solution_id']).mean().reset_index().drop(columns='solution_id')
    data.rename(columns={'problem': 'Dataset', 'model': 'Model'}, inplace=True)
    for is_for_paper in (False, True):
        plt.rcParams['font.size'] = 23 if is_for_paper else 28
        plt.figure()
        facet_grid = sns.catplot(x='k', y='test_pred_mcc', hue='Model', col='Dataset', data=data,
                                 kind='box', linewidth=2, palette='Set2', facet_kws={'despine': False},
                                 height=5, aspect=1)
        plt.ylim(-0.1, 1)
        facet_grid.set_axis_labels(x_var='Portfolio size $k$', y_var='Test-set MCC')
        if is_for_paper:
            sns.move_legend(facet_grid, edgecolor='white', loc='upper center',
                            bbox_to_anchor=(0.5, 0.15), ncol=2)
            facet_grid.legend.get_title().set_position((-284, -32))
            plot_dir = paper_plot_dir
        else:
            sns.move_legend(facet_grid, edgecolor='white', loc='center left',
                            bbox_to_anchor=(1, 0.5), ncol=1)
            plt.yticks([x / 5 for x in range(6)])
            plot_dir = presentation_plot_dir
        plt.tight_layout()
        plt.savefig(plot_dir / 'prediction-test-mcc.pdf', bbox_inches='tight')

    print('\nMCC for beam search with w=100, random forests:')
    data = prediction_results[(prediction_results['algorithm'] == 'beam_search') &
                              (prediction_results['w'] == 100) &
                              (prediction_results['model'] == 'Random forest')]
    print(data.groupby(['problem', 'k'])['test_pred_mcc'].agg(['median', 'std']).reset_index().pivot(
        index='k', columns='problem').round(2))

    print('\nMCC for the optimal solution, random forests:')
    data = prediction_results[(prediction_results['algorithm'] == 'mip_search') &
                              (prediction_results['model'] == 'Random forest')]
    print(data.groupby(['problem', 'k'])['test_pred_mcc'].agg(['median', 'std']).reset_index().pivot(
        index='k', columns='problem').round(2))

    print('\nMedian MCC per model, using all prediction results:')
    print(prediction_results.groupby(['problem', 'model'])[
        ['train_pred_mcc', 'test_pred_mcc']].median().round(2))

    print('\nTrain-test MCC difference per model, using all prediction results:')
    data = prediction_results[['problem', 'model', 'train_pred_mcc', 'test_pred_mcc']].copy()
    data['train_test_diff'] = data['train_pred_mcc'] - data['test_pred_mcc']
    print(data.groupby(['problem', 'model'])['train_test_diff'].describe().round(2))

    print('\n----6.2.2 Portfolio Performance----')

    # Figure 4: Objective value of model-based and VBS-based top beam-search portfolios over k
    w = 100
    data = prediction_results.loc[(prediction_results['algorithm'] == 'beam_search') &
                                  (prediction_results['w'] == w)]
    plot_vars = ['test_pred_objective', 'test_objective', 'test_portfolio_sbs']
    data = data.loc[(data['k'] != 1) & (data['k'] <= 10) & (data['model'] == 'Random forest'),
                    ['problem', 'k', 'solution_id'] + plot_vars]
    # Aggregate over cross-validation folds (don't group by k here, as solution_ids for beam search
    # go over multiple values of k):
    data = data.groupby(['problem', 'solution_id']).mean().reset_index().drop(columns='solution_id')
    data['k'] = data['k'].astype(int)  # can become float due to mean() operation
    data = data.melt(id_vars=['problem', 'k'], value_vars=plot_vars,
                     var_name='Approach', value_name='objective')
    data['Approach'] = data['Approach'].replace(
        {'test_pred_objective': 'Prediction', 'test_objective': 'VBS', 'test_portfolio_sbs': 'SBS'})
    data.rename(columns={'problem': 'Dataset'}, inplace=True)
    global_sbs_data = search_results.loc[(search_results['algorithm'] == 'mip_search') &
                                         (search_results['k'] == 1), ['problem', 'fold_id', 'test_objective']]
    for is_for_paper in (False, True):
        plt.rcParams['font.size'] = 22 if is_for_paper else 24
        plt.figure()
        facet_grid = sns.catplot(x='k', y='objective', hue='Approach', col='Dataset', data=data,
                                 kind='box', linewidth=2, palette='Set2', facet_kws={'despine': False},
                                 height=5, aspect=1)
        for dataset_name, dataset_subplot in facet_grid.axes_dict.items():  # add baseline performance
            dataset_subplot.axhline(
                y=global_sbs_data.loc[global_sbs_data['problem'] == dataset_name, 'test_objective'].mean(),
                color=sns.color_palette('Set2').as_hex()[3])
        facet_grid.set_axis_labels(x_var='Portfolio size $k$', y_var='PAR-2 score')
        if is_for_paper:
            sns.move_legend(facet_grid, edgecolor='white', loc='upper center',
                            bbox_to_anchor=(0.6, 0.15), ncol=3)
            facet_grid.legend.get_title().set_position((-312, -31))
            plot_dir = paper_plot_dir
        else:
            sns.move_legend(facet_grid, edgecolor='white', loc='center left',
                            bbox_to_anchor=(1, 0.5), ncol=1)
            plt.yticks(range(0, 3001, 500))
            plot_dir = presentation_plot_dir
        plt.tight_layout()
        plt.savefig(plot_dir / 'prediction-test-objective-beam.pdf', bbox_inches='tight')

    # Figure 5: Objective value of model-based and VBS-based optimal (MIP search) portfolios over k
    data = prediction_results.loc[(prediction_results['algorithm'] == 'mip_search')]
    data = data.loc[(data['k'] != 1) & (data['k'] <= 10) & (data['model'] == 'Random forest'),
                    ['problem', 'k', 'solution_id'] + plot_vars]
    data = data.melt(id_vars=['problem', 'k'], value_vars=plot_vars,
                     var_name='Approach', value_name='objective')
    data['Approach'] = data['Approach'].replace(
        {'test_pred_objective': 'Prediction', 'test_objective': 'VBS', 'test_portfolio_sbs': 'SBS'})
    data.rename(columns={'problem': 'Dataset'}, inplace=True)
    plt.rcParams['font.size'] = 22
    plt.figure()
    facet_grid = sns.catplot(x='k', y='objective', col='Dataset', hue='Approach', data=data,
                             kind='strip', s=8, palette='Set2', facet_kws={'despine': False},
                             height=5, aspect=1, dodge=True)
    for dataset_name, dataset_subplot in facet_grid.axes_dict.items():  # add baseline performance
        dataset_subplot.axhline(
            y=global_sbs_data.loc[global_sbs_data['problem'] == dataset_name, 'test_objective'].mean(),
            color=sns.color_palette('Set2').as_hex()[3])
    facet_grid.set_axis_labels(x_var='Portfolio size $k$', y_var='PAR-2 score')
    sns.move_legend(facet_grid, edgecolor='white', loc='upper center', bbox_to_anchor=(0.6, 0.15), ncol=3)
    facet_grid.legend.get_title().set_position((-312, -31))
    plt.tight_layout()
    plt.savefig(paper_plot_dir / 'prediction-test-objective-optimal.pdf', bbox_inches='tight')

    print('\n----6.2.3 Feature Importance----')

    print('\nAverage feature importance (in %) for random forests over all prediction scenarios:')
    importance_cols = [x for x in prediction_results.columns if x.startswith('imp.')]
    data = prediction_results.loc[prediction_results['model'] == 'Random forest',
                                  importance_cols].mean() * 100  # importance as percentage
    print(data.describe().round(2))
    print('\nTo reach a feature importance of 50% with random forests, one needs',
          f'{sum(data.sort_values(ascending=False).cumsum() < 50) + 1} features.')
    print('\nHow many features are used in each random-forest model?')
    print((prediction_results.loc[prediction_results['model'] == 'Random forest',
                                  importance_cols] > 0).sum(axis='columns').describe().round(2))


# Parse some command line argument and run evaluation.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates the paper\'s plots and prints statistics.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='data/', dest='data_dir',
                        help='Directory with input data, i.e., runtimes and instance features.')
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/',
                        dest='results_dir', help='Directory with experimental results.')
    parser.add_argument('-a', '--paper-plots', type=pathlib.Path, default='../text/plots/',
                        dest='paper_plot_dir', help='Output directory for paper plots.')
    parser.add_argument('-e', '--presentation-plots', type=pathlib.Path, default='../presentation/plots/',
                        dest='presentation_plot_dir', help='Output directory for presentation plots.')
    print('Evaluation started.')
    evaluate(**vars(parser.parse_args()))
    print('Plots created and saved.')
