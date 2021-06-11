"""Interactive evaluation

Evaluation code snippets outside the paper's evaluation script, e.g., for experimental plots.
Intended for 'interactive' use (i.e., block-by-block execution in some IDE) and manual inspection
of plots; plots are not saved.
"""

import ast
import math

import pandas as pd
import seaborn as sns

import prepare_dataset


# Load results, make solvers a list again:
search_results = pd.read_csv('data/search_results.csv', converters={'solvers': ast.literal_eval})
# Fix k for beam search (beam search run up to k, but also saves smaller intermediate results)
search_results.loc[search_results['algorithm'] == 'beam_search', 'k'] =\
    search_results.loc[search_results['algorithm'] == 'beam_search', 'solvers'].transform(len)
# Load runtimes, which we need for beam-search bounds
runtimes = pd.read_csv('data/runtimes.csv').drop(columns='hash')
runtimes = runtimes[(runtimes != prepare_dataset.PENALTY).any(axis='columns')]


# ---Analyze objective value---


# Objective of MIP search (exact solution) over k
# Create table with optimal VBS values, reproducing results from the paper "SAT Competition 2020";
# need to account for instances not solved by any solver, which we excluded from our experiments
data = search_results[(search_results['algorithm'] == 'mip_search') & (search_results['problem'] == 'PAR2')]
data['full_objective_value'] = (data['objective_value'] + 10000 * 84) / 400
print(data[['full_objective_value', 'k']].set_index('k').round(1))
# Plot
data = search_results[search_results['algorithm'] == 'mip_search'].copy()
sns.lineplot(x='k', y='objective_value', data=data[data['problem'] == 'PAR2'])
sns.lineplot(x='k', y='objective_value', data=data[data['problem'] == 'solved'])
# Gain in objectve value over k
data['objective_gain'] = data.groupby('problem')['objective_value'].transform(lambda x: (x.shift() - x) / x.shift())
data['objective_gain'] = data['objective_gain'].fillna(0)
sns.lineplot(x='k', y='objective_gain', data=data[data['problem'] == 'PAR2'])
sns.lineplot(x='k', y='objective_gain', data=data[data['problem'] == 'solved'])

# Correlation between objective value for the two problems
data = search_results.loc[search_results['algorithm'] == 'mip_search', ['objective_value', 'problem', 'k']]
print(data.pivot(index='k', columns='problem', values='objective_value').corr())

# Objective of beam search over k and w, compared to exact solution
# For beam search, we need to extract best solution for each k and w first
data = search_results[search_results['algorithm'] == 'beam_search'].groupby(
    ['problem', 'algorithm', 'k', 'w'])['objective_value'].min().reset_index()
mip_data = search_results.loc[search_results['algorithm'] == 'mip_search',
                              ['problem', 'algorithm', 'k', 'w', 'objective_value']]
bound_data = mip_data.copy()
bound_data['algorithm'] = 'upper_bound'
c_w = runtimes.max(axis='columns').sum()  # VWS performance
bound_data.loc[bound_data['problem'] == 'PAR2', 'objective_value'] = c_w / math.e +\
    (1 - 1 / math.e) * bound_data.loc[bound_data['problem'] == 'PAR2', 'objective_value']
c_w = (runtimes == prepare_dataset.PENALTY).astype(int).max(axis='columns').sum()
bound_data.loc[bound_data['problem'] == 'solved', 'objective_value'] = c_w / math.e +\
    (1 - 1 / math.e) * bound_data.loc[bound_data['problem'] == 'solved', 'objective_value']
data = pd.concat([data, mip_data, bound_data]).reset_index(drop=True)
data['objective_frac'] = data.groupby(['problem', 'k'])['objective_value'].apply(lambda x: x / x.min())
data['objective_diff'] = data.groupby(['problem', 'k'])['objective_value'].apply(lambda x: x - x.min())
# Division might introduce NA or inf if objective is 0:
data['objective_frac'] = data['objective_frac'].replace([float('nan'), float('inf')], 1)
assert (data.loc[data['algorithm'] == 'mip_search', 'objective_frac'] == 1).all()
print(data[data['algorithm'] == 'beam_search'].groupby(['problem', 'w'])['objective_frac'].describe(percentiles=[0.5]))
sns.boxplot(x='k', y='objective_value', data=data[data['problem'] == 'PAR2'])
sns.boxplot(x='k', y='objective_value', data=data[(data['problem'] == 'PAR2') &
                                                  (data['algorithm'] != 'upper_bound')])
sns.boxplot(x='k', y='objective_value', data=data[data['problem'] == 'solved'])
sns.boxplot(x='k', y='objective_value', data=data[(data['problem'] == 'solved') &
                                                  (data['algorithm'] != 'upper_bound')])
sns.boxplot(x='w', y='objective_frac', data=data[data['problem'] == 'PAR2'])
sns.boxplot(x='w', y='objective_diff', data=data[data['problem'] == 'solved'])

# Objective of exhaustive search over k
data = search_results[search_results['algorithm'] == 'exhaustive_search']
print(data.groupby(['problem', 'k'])['objective_value'].describe().round(2))
sns.boxplot(x='k', y='objective_value', data=data[data['problem'] == 'PAR2'])
sns.boxplot(x='k', y='objective_value', data=data[data['problem'] == 'solved'])


# ---Analyze search time---


# Search time of beam search over w (we don't have times for individual k)
data = search_results[search_results['algorithm'] == 'beam_search']
# One search yields multiple solutions for a particular w and problem, but search time is same:
assert (data.groupby(['problem', 'w'])['settings_id'].nunique() == 1).all()
assert (data.groupby(['problem', 'w'])['search_time'].nunique() == 1).all()
data = data.groupby(['problem', 'w'])['search_time'].first().reset_index()
sns.lineplot(x='w', y='search_time', hue='problem', data=data)

# Search time of exhaustive search over k
data = search_results[search_results['algorithm'] == 'exhaustive_search']
# One search yields multiple solutions for a particular k and problem, but search time is same:
assert (data.groupby(['problem', 'k'])['settings_id'].nunique() == 1).all()
assert (data.groupby(['problem', 'k'])['search_time'].nunique() == 1).all()
data = data.groupby(['problem', 'k'])['search_time'].first().reset_index()
sns.lineplot(x='k', y='search_time', hue='problem', data=data)

# Search time of MIP search over k
data = search_results[search_results['algorithm'] == 'mip_search']
# One search yields only one solution for a particular k and problem, so we don't need to group:
assert data['settings_id'].nunique() == len(data)
sns.lineplot(x='k', y='search_time', hue='problem', data=data)


# ---Analyze solvers in portfolios---


def add_solver_similarity_data(data) -> None:
    assert (data.groupby(['problem', 'k']).size() == 1).all()
    data['prev_solvers'] = data.groupby('problem')['solvers'].shift().fillna('').apply(list)
    data['card_intersection'] = data.apply(lambda x: len(set(x['solvers']).intersection(x['prev_solvers'])), axis='columns')
    data['card_union'] = data.apply(lambda x: len(set(x['solvers']).union(x['prev_solvers'])), axis='columns')
    data['jaccard'] = data['card_intersection'] / data['card_union']
    data['card_diff'] = data['card_union'] - data['card_intersection']  # absolute number of add/delete


# Check how many solvers change from k-1 to k in exact solution (MIP search)
data = search_results[search_results['algorithm'] == 'mip_search'].copy()
add_solver_similarity_data(data)
sns.lineplot(x='k', y='card_diff', hue='problem', data=data)
# Note that not all k solvers need to be selected (if additional solver provide no benefit), which
# can cause some additional fluctuation between values of k, even if objective stays same
print((data['k'] - data['solvers'].apply(len)).describe())

# Check how many solvers change from k-1 to k in greedy (beam width = 1) search
# (rather a sanity check than a useful evaluation, as always one solver added per iteration)
data = search_results[(search_results['algorithm'] == 'beam_search') & (search_results['w'] == 1)].copy()
add_solver_similarity_data(data)
sns.lineplot(x='k', y='card_diff', hue='problem', data=data)

# Count solver occurrence for a certain beam-search parametrization; can do this for any k, but for
# very small and big k, there might not be w * |problems| unique portfolios; also, w should have a
# certain size, or else there are not much portfolios to count over; alternatively, can also sort
# results from exhaustive search by objective value and take top w
print(data.groupby('k').size())
w = search_results['w'].max()
k = 4
data = search_results[(search_results['algorithm'] == 'beam_search') & (search_results['w'] == w)].copy()
plot_data = data.loc[data['k'] == k, ['problem', 'solvers']].explode('solvers').value_counts()
# Need to take care of solvers which do not appear in any portfolio; add them to data by re-indexing:
solver_names = search_results.loc[(search_results['algorithm'] == 'exhaustive_search') &
                                  (search_results['k'] == 1), 'solvers'].explode().unique()
new_index = pd.MultiIndex.from_product([data['problem'].unique(), solver_names], names=['problem', 'solvers'])
plot_data = plot_data.reindex(new_index).reset_index().rename(columns={0: 'occurrence'}).fillna(0)
plot_data['rank'] = plot_data.groupby('problem')['occurrence'].rank(ascending=False, method='first')
sns.lineplot(x='rank', y='occurrence', hue='problem', data=plot_data)
# Now analyze this over all k: Heatmap where each row shows relativ frequency of solvers for some k
plot_data = data[['problem', 'solvers', 'k']].explode('solvers').value_counts()
new_index = pd.MultiIndex.from_product([data['problem'].unique(), solver_names, range(1, 49)],
                                       names=['problem', 'solvers', 'k'])
plot_data = plot_data.reindex(new_index).reset_index().rename(columns={0: 'occurrence'}).fillna(0)
plot_data['occurrence'] = plot_data.groupby(['problem', 'k'])['occurrence'].transform(lambda x: x / x.sum())
plot_data = plot_data[plot_data['problem'] == 'PAR2'].drop(columns='problem')
plot_data = plot_data.pivot(index='k', columns='solvers', values='occurrence')
sns.heatmap(plot_data, cmap='flare')

# Relate solver occurence (no/yes) in portfolio to objective value for exhaustive-search data
k = 4  # pick any (valid) k
data = search_results[(search_results['algorithm'] == 'exhaustive_search') &
                      (search_results['k'] == k) & (search_results['problem'] == 'PAR2')].copy()
solvers = search_results.loc[(search_results['algorithm'] == 'exhaustive_search') &
                             (search_results['k'] == 1) & (search_results['problem'] == 'PAR2'),
                             ['solvers', 'objective_value']].explode('solvers')
solvers.rename(columns={'solvers': 'solver_name'}, inplace=True)
solvers['single_rank'] = solvers['objective_value'].rank()
for solver_name in solvers['solver_name']:
    data[solver_name] = data['solvers'].apply(lambda x: solver_name in x)
# Correlation
print(data[solver_names].corrwith(data['objective_value'], method='spearman').describe().round(2))
# Average contribution (reduction in score) - might be slightly unfair, as we don't analyze what
# happens when adding a solver, but compare all portfolios with solver to all without solver
contributions = pd.DataFrame([{
    'solver_name': solver_name,
    'avg_contribution': float(data[['objective_value', solver_name]].groupby(solver_name).mean(
        ).sort_index(ascending=False).diff().iloc[1])} for solver_name in solvers['solver_name']])
contributions = contributions.merge(solvers[['solver_name', 'single_rank']])
print(contributions.sort_values(by='avg_contribution', ascending=False))
print(contributions.corr())