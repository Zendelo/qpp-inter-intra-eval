import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Timer import timer
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tabulate import tabulate

from config import Config
from utility_functions import ensure_file, add_topic_to_qdf


def read_eval_df(prefix_path, ir_metric):
    eval_file = ensure_file(f"{prefix_path}_QL.{ir_metric}")
    eval_df = pd.read_csv(eval_file, delim_whitespace=True, names=['qid', ir_metric], index_col=0)
    return eval_df.drop(DUPLICATED_QIDS, errors='ignore').sort_index()
    # return eval_df


def read_prediction_files(prefix_path, r_type='all'):
    if r_type == 'all':
        # post_ret_predictors = glob(prefix_path + '_QL_g_*.pre')  # The g is used for the optimized parameters
        post_ret_predictors = glob(prefix_path + '_QL_*.pre')  # The g is used for the optimized parameters
        pre_ret_predictors = glob(prefix_path + '*PRE*')
        predictors = pre_ret_predictors + post_ret_predictors
    elif r_type.lower() == 'pre':
        predictors = glob(prefix_path + '*PRE*')
    else:
        # predictors = glob(prefix_path + '_QL_g_*.pre')  # The g is used for the optimized parameters
        predictors = glob(prefix_path + '_QL_*.pre')
    _results = []
    for _file in predictors:
        collection, method, predictor = _file.rsplit('/', 1)[1].replace('.pre', '').rsplit('_', 2)
        _results.append(
            pd.read_csv(_file, delim_whitespace=True, names=['topic', 'qid', predictor],
                        dtype={'topic': str}).set_index(['topic', 'qid']))
    return pd.concat(_results, axis=1).drop(['672', 672], errors='ignore') \
        .drop(DUPLICATED_QIDS, level=1, errors='ignore').sort_index() \
        .rename(columns=lambda x: '-'.join([i.split('+')[0] for i in x.split('-')]))


def construct_pairwise_df(sr: pd.Series, np_fun):
    """Constructs an upper diagonal df from all pairwise comparisons of a sr"""
    sr = sr.sort_index()
    _mat = np.triu(np_fun(sr.to_numpy() - sr.to_numpy()[:, None]), k=1)
    _mat[np.tril_indices(_mat.shape[0])] = None
    return pd.DataFrame(_mat, index=sr.index.get_level_values('qid'),
                        columns=sr.index.get_level_values('qid')).rename_axis(index='qid_1', columns='qid_2')


def construct_sampled_inter_df(eval_df, predictions_df, n):
    _res = []
    for _ in range(n):
        ev_sr = eval_df.groupby('topic').sample(1).iloc[:, 0]
        _ev_sgn = construct_pairwise_df(ev_sr, np.sign).stack()
        _ev_diff = construct_pairwise_df(ev_sr, np.abs).stack()
        for predictor, _sr in predictions_df.loc[ev_sr.index].iteritems():
            _pr_sgn = construct_pairwise_df(_sr, np.sign).stack()
            _df = pd.DataFrame({'diff': _ev_diff, 'status': _ev_sgn == _pr_sgn}, index=_ev_diff.index)
            _res.append(_df.reset_index(drop=True).assign(predictor=predictor))
    return pd.concat(_res).sort_values(['predictor', 'diff', 'status']).reset_index(drop=True)
    # df['status'] = df['status'].map({0: 'wrong', 1: 'correct'})
    # return _df


#
#
def construct_full_pairwise_inter_df(eval_df, predictions_df):
    _res = []
    intra_topic_index = eval_df.iloc[:, 0].groupby('topic').apply(
        lambda x: construct_pairwise_df(x, np.abs)).stack().index.droplevel(0)
    # ev_sr = eval_df.groupby('topic').sample(1).iloc[:, 0]
    _ev_sgn = construct_pairwise_df(eval_df.iloc[:, 0], np.sign).stack()
    _ev_diff = construct_pairwise_df(eval_df.iloc[:, 0], np.abs).stack()
    for predictor, _sr in predictions_df.iteritems():
        _pr_sgn = construct_pairwise_df(_sr, np.sign).stack()
        _pr_diff = construct_pairwise_df(_sr, np.abs).stack()
        _df = pd.DataFrame({'ev_diff': _ev_diff, 'pr_diff': _pr_diff, 'status': _ev_sgn == _pr_sgn},
                           index=_ev_diff.index).drop(intra_topic_index)
        _res.append(_df.reset_index().assign(predictor=predictor))
    return pd.concat(_res).sort_values(['predictor', 'ev_diff', 'pr_diff', 'status']).reset_index(drop=True)


def construct_inter_sampled_to_intra_df(eval_df, predictions_df):
    _res = []
    intra_topic_df = eval_df.iloc[:, 0].groupby('topic').apply(lambda x: construct_pairwise_df(x, np.abs)).stack()
    # w = intra_topic_df.reset_index(drop=True).round(2).value_counts().sort_index()
    w = pd.cut(intra_topic_df, bins=100, precision=3, include_lowest=True)
    w = w.groupby(w).count()
    intra_topic_index = intra_topic_df.index.droplevel(0)
    _ev_diff = construct_pairwise_df(eval_df.iloc[:, 0], np.abs).stack()
    x = pd.cut(_ev_diff.drop(intra_topic_index), bins=100, precision=3, include_lowest=True)
    x.index.rename(['qid_1', 'qid_2'], inplace=True)
    sampled_qids = pd.MultiIndex.from_frame(
        x.reset_index().groupby(0).apply(lambda x: x.sample(w.loc[x.name])).reset_index(drop=True)[['qid_1', 'qid_2']])
    _ev_sgn = construct_pairwise_df(eval_df.iloc[:, 0], np.sign).stack().loc[sampled_qids]
    _ev_diff = _ev_diff.loc[sampled_qids]
    for predictor, _sr in predictions_df.iteritems():
        _pr_sgn = construct_pairwise_df(_sr, np.sign).stack().loc[sampled_qids]
        _pr_diff = construct_pairwise_df(_sr, np.abs).stack().loc[sampled_qids]
        _df = pd.DataFrame({'ev_diff': _ev_diff, 'pr_diff': _pr_diff, 'status': _ev_sgn == _pr_sgn},
                           index=_ev_diff.index)
        _res.append(_df.reset_index().assign(predictor=predictor))
    return pd.concat(_res).sort_values(['predictor', 'ev_diff', 'pr_diff', 'status']).reset_index(drop=True)


def load_generate_pickle_df(file_name, func, *args):
    if isinstance(file_name, (list, tuple, set)):
        single_file = False
    else:
        single_file = True
    try:
        if single_file:
            result = pd.read_pickle(file_name)
        else:
            result = [pd.read_pickle(_file) for _file in file_name]
    except FileNotFoundError:
        logger.warning(f'Failed to load {file_name}')
        logger.warning(f'Generating a new df and saving')
        result = func(*args)
        if single_file:
            result.to_pickle(file_name)
        else:
            assert len(result) == len(
                file_name), f'The number of objects returned by the function and number of files differ'
            [_res.to_pickle(_file) for (_res, _file) in zip(result, file_name)]
    return result


@timer
def inter_topic_pairwise_analysis(eval_df, predictions_df, n=None, load_cache=True, sample=False):
    ir_metric = eval_df.columns[0]
    n = n or 'all'
    if load_cache:
        if sample:
            df = load_generate_pickle_df(f'inter_topic_pairwise_{ir_metric}_df_{n}.pkl', construct_sampled_inter_df,
                                         eval_df, predictions_df, n)
        else:
            df = load_generate_pickle_df(f'inter_topic_pairwise_{ir_metric}_df_{n}.pkl',
                                         construct_full_pairwise_inter_df, eval_df, predictions_df)
    else:
        if sample:
            df = construct_sampled_inter_df(eval_df, predictions_df, n)
        else:
            df = construct_full_pairwise_inter_df(eval_df, predictions_df)
    intra_df = eval_df.iloc[:, 0].groupby('topic').apply(
        lambda x: construct_pairwise_df(x, np.abs)).stack().reset_index(drop=True).rename('diff')
    # dd = pd.concat(
    #     [pd.DataFrame(df.loc[df['predictor'] == 'clarity', ['diff']]).assign(kind='inter').reset_index(drop=True),
    #      pd.DataFrame(intra_df).assign(kind='intra')])
    # dd.to_pickle(f'all_pairs_{ir_metric}_diff.pkl')
    log_pairwise_stats(df, 'Inter-topic', ir_metric)
    p = 2

    _gpd_df = df.round(p).groupby(['predictor', 'ev_diff'])['status']
    _freq_df = (_gpd_df.sum() / _gpd_df.count()).fillna(0).reset_index().rename({'status': 'freq'}, axis=1).assign(
        sample_size=_gpd_df.count().to_numpy())
    # df['status'] = df['status'].map({False: 'wrong', True: 'correct'})
    plot_pairwise_freq_diff(df.round(4), f'Inter-topic {n} samples {PlotNames.get(ir_metric, ir_metric)}', ir_metric)
    plot_pairwise_freq_diff_rel(_freq_df, f'Inter-topic {n} samples {PlotNames.get(ir_metric, ir_metric)}', ir_metric)

    _p = 4
    _gpd_df = df.round(p).groupby(['predictor', 'pr_diff'])['status']
    _freq_df = (_gpd_df.sum() / _gpd_df.count()).fillna(0).reset_index().rename({'status': 'freq'}, axis=1).assign(
        sample_size=_gpd_df.count().to_numpy())
    plot_pairwise_freq_diff(df.round(4), f'Inter-topic {n} samples {PlotNames.get(ir_metric, ir_metric)}', 'Predictor',
                            diff_col='pr_diff')
    plot_pairwise_freq_diff_rel(_freq_df, f'Inter-topic {n} samples {PlotNames.get(ir_metric, ir_metric)}', 'Predictor',
                                diff_col='pr_diff')
    return df


def construct_intra_topic_df(eval_df, predictions_df):
    _res = []
    # freq_res = []
    _ev_sgn = eval_df.iloc[:, 0].groupby('topic').apply(lambda x: construct_pairwise_df(x, np.sign)).stack()
    _ev_diff = eval_df.iloc[:, 0].groupby('topic').apply(lambda x: construct_pairwise_df(x, np.abs)).stack()
    for predictor, _sr in predictions_df.iteritems():
        _pr_sgn = _sr.groupby('topic').apply(lambda x: construct_pairwise_df(x, np.sign)).stack()
        _pr_diff = _sr.groupby('topic').apply(lambda x: construct_pairwise_df(x, np.abs)).stack()
        _df = pd.DataFrame({'ev_diff': _ev_diff, 'pr_diff': _pr_diff, 'status': _ev_sgn == _pr_sgn},
                           index=_ev_diff.index).reset_index('topic', drop=True)
        logger.debug(f'Number of tied ev pairs: {len(_ev_sgn.loc[_ev_sgn == 0].index.droplevel(0))}')
        logger.debug(f'Number of tied {predictor} pairs: {len(_pr_sgn.loc[_pr_sgn == 0].index.droplevel(0))}')
        _res.append(_df.reset_index().rename(columns={'level_1': 'qid_2'}).assign(predictor=predictor))
    return pd.concat(_res).sort_values(['predictor', 'ev_diff', 'status']).reset_index(drop=True)


@timer
def intra_topic_pairwise_analysis(eval_df, predictions_df, load_cache):
    ir_metric = eval_df.columns[0]
    if load_cache:
        df = load_generate_pickle_df(f'intra_topic_pairwise_{ir_metric}_df.pkl', construct_intra_topic_df, eval_df,
                                     predictions_df)
    else:
        df = construct_intra_topic_df(eval_df, predictions_df)
    log_pairwise_stats(df, 'Intra-topic', ir_metric)
    p = 2
    _gpd_df = df.round(p).groupby(['predictor', 'ev_diff'])['status']
    _freq_df = (_gpd_df.sum() / _gpd_df.count()).fillna(0).reset_index().rename({'status': 'freq'}, axis=1).assign(
        sample_size=_gpd_df.count().to_numpy())
    plot_pairwise_freq_diff(df.round(4), f'Intra-topic all pairs {PlotNames.get(ir_metric, ir_metric)}', ir_metric,
                            diff_col='ev_diff')
    plot_pairwise_freq_diff_rel(_freq_df, f'Intra-topic all pairs {PlotNames.get(ir_metric, ir_metric)}', ir_metric,
                                diff_col='ev_diff')
    _p = 4
    _gpd_df = df.round(_p).groupby(['predictor', 'pr_diff'])['status']
    _freq_df = (_gpd_df.sum() / _gpd_df.count()).fillna(0).reset_index().rename({'status': 'freq'}, axis=1).assign(
        sample_size=_gpd_df.count().to_numpy())
    plot_pairwise_freq_diff(df.round(4), f'Intra-topic all pairs {PlotNames.get(ir_metric, ir_metric)}', 'Predictor',
                            diff_col='pr_diff')
    plot_pairwise_freq_diff_rel(_freq_df, f'Intra-topic all pairs {PlotNames.get(ir_metric, ir_metric)}', 'Predictor',
                                diff_col='pr_diff')

    return df


def plot_pairwise_freq_diff_rel(_freq_df, title, ir_metric, diff_col='ev_diff'):
    n_predictors = _freq_df['predictor'].nunique()
    _title = f'freq_given_diff_{n_predictors}_predictors_{title.replace(" ", "_").lower()}'

    def plot_regplots(**kwargs):
        sns.regplot(x=diff_col, y='freq', lowess=True,
                    scatter_kws={'s': 5 * np.log2(kwargs['data']['sample_size']) + 2}, **kwargs)

    if diff_col == 'pr_diff':
        g2 = sns.FacetGrid(_freq_df, col="predictor", col_wrap=PLOTS_COL_WRAP, sharex=False, sharey=False)
        g2.map_dataframe(plot_regplots)
    else:
        g2 = sns.lmplot(data=_freq_df, x=diff_col, y='freq', col='predictor', lowess=True, col_wrap=PLOTS_COL_WRAP,
                        scatter_kws={
                            's': 5 * np.log2(_freq_df.head(int(len(_freq_df) / n_predictors))['sample_size']) + 2})
    g2.set_axis_labels(f"{PlotNames.get(ir_metric, ir_metric)} difference", "Correct Ratio")
    g2.set_titles(title + " {col_name} predictor")
    g2.tight_layout()
    g2.savefig(f'{_title}_{diff_col}_reg.pdf')
    plt.show()
    _freq_df = _freq_df.rename({diff_col: f"{PlotNames.get(ir_metric, ir_metric)} diff"})
    g3 = sns.pairplot(data=_freq_df, hue='predictor')
    plt.savefig(f'{_title}_{diff_col}_pairplot.pdf')
    plt.show()


def plot_pairwise_freq_diff(_df, title, ir_metric, diff_col='ev_diff'):
    n_predictors = _df['predictor'].nunique()
    _title = f'estimating_freq_given_diff_{n_predictors}_predictors_{title.replace(" ", "_").lower()}'
    if diff_col == 'pr_diff':
        g1 = sns.displot(data=_df, x=diff_col, col='predictor', hue='status', stat="density", element='step',
                         col_wrap=PLOTS_COL_WRAP, common_bins=False, facet_kws={'sharex': None, 'sharey': False})
    else:
        g1 = sns.displot(data=_df, x=diff_col, col='predictor', col_wrap=PLOTS_COL_WRAP, hue='status', stat="density",
                         element='step')
    g1.set_axis_labels(f"{PlotNames.get(ir_metric, ir_metric)} difference", "Density")
    g1.set_titles(title + " {col_name} predictor")
    plt.tight_layout()
    plt.savefig(f'{_title}_{ir_metric.replace(" ", "_").lower()}_hist.pdf')
    plt.show()
    if diff_col == 'pr_diff':
        g11 = sns.displot(data=_df, x=diff_col, col='predictor', hue='status', kind='kde', clip=(0.0, 1.0), fill=True,
                          alpha=0.5, col_wrap=PLOTS_COL_WRAP, facet_kws={'sharex': False, 'sharey': False})
    else:
        g11 = sns.displot(data=_df, x=diff_col, col='predictor', hue='status', kind='kde', clip=(0.0, 1.0), fill=True,
                          col_wrap=PLOTS_COL_WRAP, alpha=0.5)
    g11.set_axis_labels(f"{PlotNames.get(ir_metric, ir_metric)} difference", "Density")
    g11.set_titles(title + " {col_name} predictor")
    plt.tight_layout()
    plt.savefig(f'{_title}_{ir_metric.replace(" ", "_").lower()}_kde.pdf')
    plt.show()


def inter_topic_eval(ir_metric, prefix_path):
    predictors_type = 'all'
    eval_df = add_topic_to_qdf(read_eval_df(prefix_path, ir_metric)).set_index(['topic', 'qid'])
    predictions_df = read_prediction_files(prefix_path, predictors_type)
    predictions_df = predictions_df[['qf', 'nqc', 'uef-wig']]
    predictions_df = predictions_df[['qf', 'uef-wig']]
    df = inter_topic_pairwise_analysis(eval_df, predictions_df, load_cache=True)
    print('Inter Topic table')
    print_diff_probabilities_table(df, title='Inter-Topic', q=4, ir_metric=ir_metric)


def log_pairwise_stats(df, prefix, ir_metric):
    _gdf = df.groupby('predictor')['status']
    logger.info(f"{prefix} pairs df has {_gdf.count()[0]} pairs")
    tbl_1 = tabulate(
        pd.DataFrame((_gdf.sum() / _gdf.count()).sort_values()).rename({'status': 'Correct Ratio'}, axis=1),
        headers='keys', tablefmt='psql')
    logger.info(f"Ratio of correct pairs by predictor: \n{tbl_1}")
    _ties_gdf = df.loc[df['ev_diff'] == 0.0].groupby('predictor')['status']
    logger.info(f"Number of tied (0 {PlotNames.get(ir_metric, ir_metric)} ev_diff) pairs: {_ties_gdf.count()[0]}")
    tbl_2 = tabulate(
        pd.DataFrame((_ties_gdf.sum() / _ties_gdf.count()).sort_values()).rename({'status': 'Correct Ratio'}, axis=1),
        headers='keys', tablefmt='psql')
    logger.info(f"Ratio of correct tied pairs by predictor: \n{tbl_2}")


def intra_topic_eval(ir_metric, prefix_path):
    predictors_type = 'all'
    eval_df = add_topic_to_qdf(read_eval_df(prefix_path, ir_metric)).set_index(['topic', 'qid'])
    predictions_df = read_prediction_files(prefix_path, predictors_type)
    # predictions_df = predictions_df[['qf', 'nqc', 'uef-wig']]
    df = intra_topic_pairwise_analysis(eval_df, predictions_df, load_cache=True)
    print_diff_probabilities_table(df, title='Intra-Topic', q=4, ir_metric=ir_metric)
    return df


def print_diff_probabilities_table(predictions_results, title, q, ir_metric):
    res = []
    if isinstance(predictions_results, pd.DataFrame):
        df = predictions_results
    else:
        df = pd.DataFrame(predictions_results, columns=['status', 'predictor', 'topic', 'diff'])
    # plot_cond_prob(df)
    df['status'] = df['status'].map({False: 'wrong', True: 'correct'})
    num_predictors = df['predictor'].nunique()
    # df = df.drop('diff', axis=1).rename({'sim': 'diff'}, axis=1)
    # FIXME: These lines are used for similarity, can be uncommented
    # q = 9
    df['int'] = pd.qcut(df['ev_diff'], q=q, precision=2, duplicates='drop')
    for predictor, _df in df.groupby('predictor'):
        _df = pd.crosstab(index=_df['status'], columns=_df['int'].astype(str), values=_df['ev_diff'], aggfunc='count',
                          normalize='columns', margins=True).reset_index()
        _df.insert(loc=0, column='predictor', value=predictor)
        res.append(_df)
    res_df = pd.concat(res)
    x = res_df.set_index(['predictor', 'status']).columns[0]
    res_df.rename(columns={x: pd.Interval(np.round(x.left, 2), x.right)}, inplace=True)
    print(res_df.set_index(['predictor', 'status']).rename({'All': 'Overall'}, axis=1).
          to_latex(float_format='%.2f', escape=False, multirow=True, multicolumn=True))
    _df = res_df.drop('All', axis=1).set_index(['predictor', 'status']).stack().reset_index().rename({0: 'prob'},
                                                                                                     axis=1)
    g = sns.catplot(data=_df, x='int', y='prob', hue='predictor', col='status')
    g.set_axis_labels(f"{PlotNames.get(ir_metric, ir_metric)} difference", "Probability")
    g.set_xticklabels(rotation=30)
    g.set_titles("{col_name} prediction")
    plt.tight_layout()
    plt.savefig(
        f'{title}_cond-prob_{num_predictors}-predictors_{q}-quantiles_{ir_metric.replace(" ", "_").lower()}.pdf')
    plt.show()


def plot_estimates_per_interval(intra_topic, inter_topic, ir_metric, n_boot=10000):
    # PlotPredictors = ['max-idf', 'nqc', 'qf', 'uef-clarity']
    PlotPredictors = ['avg-idf', 'clarity', 'nqc', 'wig', 'max-idf', 'uef-clarity', 'uef-nqc', 'uef-wig']
    # intra_topic['int'], bins = pd.qcut(intra_topic['ev_diff'], q=10, precision=3, retbins=True)
    intra_topic['int'], bins = pd.cut(intra_topic['ev_diff'], bins=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1),
                                      precision=3, retbins=True, include_lowest=True)
    # intra_topic['int'], bins = pd.cut(intra_topic['ev_diff'],
    #                                   bins=(0, 0.005, 0.015, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1),
    #                                   precision=3, retbins=True, include_lowest=True)
    inter_topic['int'] = pd.cut(inter_topic['ev_diff'], bins=bins, precision=3, include_lowest=True)
    min_samples = intra_topic.groupby(['predictor', 'int']).count().min()[0]
    logger.debug(f'min number of samples is {min_samples}')
    logger.debug(f'inter topic means per interval: {inter_topic.groupby("int")["ev_diff"].mean()}')
    logger.debug(f'intra topic means per interval: {intra_topic.groupby("int")["ev_diff"].mean()}')
    logger.debug(
        f'Correlation between inter-intra means over intervals r='
        f'{stats.pearsonr(intra_topic.groupby("int")["ev_diff"].mean(), inter_topic.groupby("int")["ev_diff"].mean())[0]:.3f}')
    # _intra = intra_topic.groupby(['predictor', 'int']).sample(min_samples).sort_values(['predictor', 'ev_diff'])
    # _inter = inter_topic.groupby(['predictor', 'int']).sample(min_samples).sort_values(['predictor', 'ev_diff'])
    _intra = intra_topic.sort_values(['predictor', 'ev_diff'])
    _inter = inter_topic.sort_values(['predictor', 'ev_diff'])
    _plot_df = pd.concat([_intra[['int', 'predictor', 'status']].assign(method='Intra'),
                          _inter[['int', 'predictor', 'status']].assign(method='Inter')])
    pt = pd.CategoricalDtype(PlotPredictors, ordered=True)
    _plot_df['predictor'] = _plot_df['predictor'].astype(pt)
    _plot_df = _plot_df.dropna().sort_values('predictor').replace(PlotNames).replace(
        {_plot_df['int'].min(): pd.Interval(0, 0.1, closed='both')})
    g = sns.catplot(data=_plot_df, x='int', y='status', col='predictor', hue='method', kind='point', ci=95,
                    n_boot=n_boot, col_wrap=4, height=3.5, aspect=1.3, scale=0.8, errwidth=1.3, capsize=0.3,
                    palette={'Inter': '#008297', 'Intra': '#83bb32'}, linestyles=[':', '--'], markers=["x", "p"],
                    hue_order=['Inter', 'Intra'], legend=True)
    # {'Inter': '#1285bf', 'Intra': '#b5325d'}
    plt.minorticks_on()
    g.map(plt.grid, b=True, which='minor', axis='y', color='#E5E5E5', linestyle=':', zorder=0,
          alpha=0.5, linewidth=0.5)
    g.map(plt.grid, b=True, which='major', axis='y', color='#E5E5E5', linestyle='-.', zorder=1, linewidth=1)
    g.set_xlabels(f"{PlotNames.get(ir_metric, ir_metric)} Difference").set_ylabels("Pairwise Accuracy") \
        .set_xticklabels(rotation=30).set_titles("{col_name}").set(ylim=(0.2, 1))
    for ax in g.axes.flat:
        ax.tick_params(axis='x', which='minor', bottom=False)
        # if ax.get_ylabel():
        #     if ax.get_title() == 'UEF(Clarity)':
        #         ax.yaxis.labelpad = 20
        #     else:
        #         ax.set_ylabel('')
        # if ax.get_xlabel():
        #     if ax.get_title() == 'UEF(Clarity)':
        #         ax.xaxis.labelpad = 15
        #     else:
        #         ax.set_xlabel('')
        for l in ax.lines + ax.collections:
            l.set_zorder(5)
            # plt.setp(l, linewidth=1)
    # g._legend.set_title(None)

    g.tight_layout()
    g.savefig(f'point_plot_{ir_metric}_all.pdf', dpi=300)
    # import tikzplotlib
    # tikzplotlib.save("test.tex")
    plt.show()
    logger.debug('\n' + tabulate(pd.DataFrame({'Inter': _inter.groupby('predictor')['status'].mean(),
                                               'Intra': _intra.groupby('predictor')['status'].mean()}).
                                 sort_values('Inter'), headers='keys', tablefmt='psql', floatfmt=".3f"))
    return _intra, _inter


def _get_stats_for_t_test(row):
    return row.groupby('int').apply(
        lambda x: stats.ttest_ind_from_stats(mean1=x.mean_x, std1=x.std_x, nobs1=x.len_x, mean2=x.mean_y, std2=x.std_y,
                                             nobs2=x.len_y, equal_var=False).pvalue[0])


def plot_ttest_pvalues(_intra, _inter, title):
    _intra['int'] = _intra['int'].astype(str)
    _inter['int'] = _inter['int'].astype(str)
    dfx = _intra.pivot_table(index='predictor', columns='int', values='status', aggfunc=['mean', 'std', len])
    dfy = _inter.pivot_table(index='predictor', columns='int', values='status', aggfunc=['mean', 'std', len])
    pvalues_df = pd.merge(dfx, dfy, on='predictor').apply(_get_stats_for_t_test, axis=1)
    pvalues_df.to_pickle(f'ttest_{title}_pvalues_df.pkl')
    print(pvalues_df.to_latex(float_format='%.3f'))
    # df = pd.read_pickle('ttest_equal_samples_pvalues_df.pkl')
    # _df = pd.read_pickle('ttest_all_pairs_pvalues_df.pkl')
    sns.heatmap(pvalues_df, annot=True, fmt='.2f', center=0.01, linewidths=.1, cmap="coolwarm")
    plt.xticks(rotation=30)
    plt.title(title.replace('_', ' '))
    plt.tight_layout()
    plt.savefig(f'heatmap_pvalues_ttest_{title}.pdf')
    plt.show()


def _calc_kl_divergence_df(df, precision=2, stat='count'):
    zdf = df.round(precision).reset_index().pivot_table(values='index', index='ev_diff', columns='kind',
                                                        aggfunc='count', fill_value=0.1)
    norm_zdf = zdf / zdf.sum()  # normalized values (sum to 1)
    norm_zdf.plot()
    plt.show()
    logger.debug(f'The KL divergence of intra to inter: {stats.entropy(norm_zdf["intra"], norm_zdf["inter"]):.3f}')
    ecdf_zdf = norm_zdf.cumsum()  # empirical Cumulative Distribution Function, eCDF
    ecdf_zdf.plot()
    plt.show()
    if stat == 'count':
        return zdf
    elif stat == 'ratio':
        return norm_zdf
    else:
        logger.warning('Specify stat= \"count\" or \"ratio\" to return a df')


#
#
# def per_group_sampling(_df, wsr):
#     p, i = _df[['predictor', 'int']].iloc[0]
#     return _df.sample(wsr.loc[p, i])
#
#
# def sample_similar_distributions(intra_topic, inter_topic):
#     intra_topic['int'], bins = pd.cut(intra_topic['ev_diff'],
#                                       bins=(0, 0.005, 0.015, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1),
#                                       precision=3, retbins=True, include_lowest=True)
#     inter_topic['int'] = pd.cut(inter_topic['ev_diff'], bins=bins, precision=3, include_lowest=True)
#     wsr = intra_topic.groupby(['predictor', 'int'])['ev_diff'].count()
#     return inter_topic.groupby(['predictor', 'int']).apply(per_group_sampling, wsr).reset_index(drop=True)
#
#
def _plot_ecdf_hist(xdf, ir_metric, title='dist_diff'):
    palette = {'Inter': '#008297', 'Intra': '#83bb32', 'Sub-Inter': '#BD5B22'}

    xdf['kind'] = xdf['kind'].str.title()
    sub_df = xdf.loc[xdf['kind'] == 'Sub-Inter']
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 9), dpi=300)
    g1 = sns.ecdfplot(data=xdf.loc[xdf['kind'] == 'Inter'], x='ev_diff', ax=ax1, linestyle='-.', label='Inter',
                      color='#008297', zorder=4, linewidth=2)
    g1 = sns.ecdfplot(data=xdf.loc[xdf['kind'] == 'Intra'], x='ev_diff', ax=ax1, linestyle='--', label='Intra',
                      color='#83bb32', zorder=2, linewidth=2)
    if len(sub_df) > 0:
        g1 = sns.ecdfplot(data=sub_df, x='ev_diff', ax=ax1, linestyle=(0, (1, 4)), label='Sub-Inter', color='#BD5B22',
                          zorder=3, linewidth=5)
    g1.grid(b=True, which='major', axis='both', color='#E5E5E5', linestyle='-.', zorder=0, linewidth=2)
    g1.set_xlabel(None)
    g1.legend()
    g2 = sns.histplot(data=xdf.loc[xdf['kind'] == 'Inter'], x='ev_diff', stat='probability', element='step', ax=ax2,
                      label='Inter', color='#008297', zorder=4, linewidth=2, linestyle='-.')
    g2 = sns.histplot(data=xdf.loc[xdf['kind'] == 'Intra'], x='ev_diff', stat='probability', element='step', ax=ax2,
                      label='Intra', color='#83bb32', zorder=2, linewidth=2)
    if len(sub_df) > 0:
        g2 = sns.histplot(data=sub_df, x='ev_diff', stat='probability', element='step', ax=ax2, linestyle=":",
                          label='Sub-Inter', color='#BD5B22', zorder=3, linewidth=2, alpha=0.4)
    g2.grid(b=True, which='major', axis='both', color='#E5E5E5', linestyle='-.', zorder=0, linewidth=2)
    g2.set_xlabel(f'{PlotNames.get(ir_metric, ir_metric)} Difference')
    g2.legend()
    # legend2 = ax2.get_legend()
    # legend2.set_title(None)
    plt.tight_layout()
    plt.savefig(f'{title}.pdf', dpi=300)

    plt.show()

    # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(3.3, 3), dpi=300)
    # g1 = sns.ecdfplot(data=xdf.loc[xdf['kind'] == 'Inter'], x='ev_diff', ax=ax1, linestyle='--', label='Inter',
    #                   color='#008297')
    # g1 = sns.ecdfplot(data=xdf.loc[xdf['kind'] == 'Intra'], x='ev_diff', ax=ax1, linestyle=(0, (3, 7)), label='Intra',
    #                   color='#83bb32')
    # g1 = sns.ecdfplot(data=xdf.loc[xdf['kind'] == 'Sub-Inter'], x='ev_diff', ax=ax1, linestyle=(5, (1, 4)),
    #                   label='Sub-Inter', color='red')
    # g1.set_xlabel(None)
    # g1.legend()
    # g2 = sns.histplot(data=xdf, hue='kind', x='ev_diff', common_norm=False, stat='probability', element='step', ax=ax2)
    # g2 = sns.histplot(data=xdf.loc[xdf['kind'] == 'inter'], x='ev_diff', common_norm=False, stat='probability',
    #                   alpha=0.3,element='step', ax=ax2, label='Inter', color=sns.color_palette("tab10")[1])
    # g2.set_xlabel(f'{PlotNames.get(ir_metric, ir_metric)} Difference')
    # legend2 = ax2.get_legend()
    # legend2.set_title(None)
    # plt.tight_layout()
    # plt.savefig(f'{title}.pdf', dpi=300)
    # plt.show()
    # plt.close('all')
    # f, ax = plt.subplots(figsize=(3.15, 2), dpi=300)
    # g = sns.histplot(data=xdf, hue='kind', x='ev_diff', common_norm=False, stat='probability', element='step')
    # g.set_xlabel(f"{PlotNames.get(ir_metric, ir_metric)} Difference", fontsize=5)
    # g.set_ylabel("Probability", fontsize=5)
    # g.set_xticklabels(g.get_xticklabels(), fontsize=4)
    # plt.tight_layout()
    # plt.show()


def _plot_utility(df, _df):
    meanprops = {"marker": "s", "markerfacecolor": "white", "markeredgecolor": "blue"}
    sns.displot(data=_df, col='predictor', x='utility', col_wrap=PLOTS_COL_WRAP)
    plt.show()
    sns.displot(data=df, col='predictor', x='utility', col_wrap=PLOTS_COL_WRAP)
    plt.show()
    sns.boxplot(y='utility', data=df[['predictor', 'utility']], x='predictor',
                order=df.groupby('predictor')['utility'].mean().sort_values().index, showmeans=True,
                meanprops=meanprops)
    plt.show()
    sns.boxplot(y='utility', data=_df[['predictor', 'utility']], x='predictor',
                order=df.groupby('predictor')['utility'].mean().sort_values().index, showmeans=True,
                meanprops=meanprops)
    plt.show()
    # figure size in inches
    # rcParams['figure.figsize'] = 16, 9
    ydf = pd.concat([df.assign(kind='intra')[['predictor', 'utility', 'kind']].reset_index(drop=True),
                     _df.assign(kind='inter')[['predictor', 'utility', 'kind']].reset_index(drop=True)], axis=0)
    sns.violinplot(data=ydf, y='utility', x='predictor', hue='kind', split=False, inner='quartile',
                   order=ydf.groupby('predictor')['utility'].mean().sort_values().index)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def load_all_pairs(ir_metric):  # Used in SIGIR 21 short paper
    def _add_utility(_df):
        return _df.assign(utility=_df['status'].map(lambda x: 1 if x else -1) * _df['ev_diff'])

    def _merge_into_single_first_grp(_intra, _inter, cols, groups_col='predictor', _sub_inter=None):
        _p = _intra[groups_col][0]
        if _sub_inter is None:
            return pd.concat([_intra.assign(kind='intra').loc[_intra[groups_col] == _p, cols].reset_index(drop=True),
                              _inter.assign(kind='inter').loc[_inter[groups_col] == _p, cols].reset_index(drop=True)],
                             axis=0)
        else:
            return pd.concat([_inter.assign(kind='inter').loc[_inter[groups_col] == _p, cols].reset_index(drop=True),
                              _intra.assign(kind='intra').loc[_intra[groups_col] == _p, cols].reset_index(drop=True),
                              _sub_inter.assign(kind='sub-Inter').loc[_sub_inter[groups_col] == _p, cols].reset_index(
                                  drop=True)], axis=0)

    pt = pd.CategoricalDtype(PreRetPredictors + PostRetPredictors, ordered=True)
    intra_topic = pd.read_pickle(f'intra_topic_pairwise_{ir_metric}_df.pkl')
    intra_topic['predictor'] = intra_topic['predictor'].astype(pt)
    intra_topic = intra_topic.sort_values(['predictor', 'ev_diff'])
    s_inter_topic = pd.read_pickle(f'inter_sampled_topic_pairwise_{ir_metric}_df.pkl')
    s_inter_topic['predictor'] = s_inter_topic['predictor'].astype(pt)
    s_inter_topic = s_inter_topic.sort_values(['predictor', 'ev_diff'])
    inter_topic = pd.read_pickle(f'inter_topic_pairwise_{ir_metric}_df_all.pkl')
    inter_topic['predictor'] = inter_topic['predictor'].astype(pt)
    inter_topic = inter_topic.sort_values(['predictor', 'ev_diff'])
    ut_intra_df = _add_utility(intra_topic)
    ut_inter_df = _add_utility(pd.read_pickle(f'inter_topic_pairwise_{ir_metric}_df_all.pkl'))
    ut_inter_df['predictor'] = ut_inter_df['predictor'].astype(pt)
    ut_inter_df = ut_inter_df.sort_values(['predictor', 'ev_diff'])
    ut_raw = _merge_into_single_first_grp(ut_intra_df, ut_inter_df, ['ev_diff', 'kind'])
    _calc_kl_divergence_df(ut_raw, precision=2)
    _plot_ecdf_hist(ut_raw, ir_metric, 'full_inter_intra_dist')
    ut_s_inter_df = _add_utility(s_inter_topic)
    ut_s_raw = _merge_into_single_first_grp(ut_intra_df, ut_inter_df, ['ev_diff', 'kind'], _sub_inter=ut_s_inter_df)
    calc_intra_inter_stats(intra_topic, s_inter_topic)
    calc_intra_inter_stats(intra_topic, inter_topic)

    _calc_kl_divergence_df(ut_s_raw, precision=2)
    _plot_ecdf_hist(ut_s_raw, ir_metric, 'sampled_inter_intra_dist')
    _plot_utility(ut_intra_df, ut_s_inter_df)
    calc_intra_inter_stats(intra_topic, inter_topic)
    calc_intra_inter_stats(intra_topic, s_inter_topic)
    _intra, _inter = plot_estimates_per_interval(intra_topic, s_inter_topic, ir_metric, n_boot=3)
    _intra, _inter = plot_estimates_per_interval(intra_topic, inter_topic, ir_metric, n_boot=1000)
    max_interval = _intra['int'].max()
    _max_intra, _max_inter = plot_estimates_per_interval(_intra.loc[_intra['int'] == max_interval],
                                                         _inter.loc[_inter['int'] == max_interval], ir_metric)
    max_interval = _max_intra['int'].max()
    _max_intra, _max_inter = plot_estimates_per_interval(_max_intra.loc[_max_intra['int'] == max_interval],
                                                         _max_inter.loc[_inter['int'] == _max_inter], ir_metric)
    plot_ttest_pvalues(_intra, _inter, title=f'equal_samples_{ir_metric}')
    return None


def two_way_anova(df, response, factor_1, factor_2):
    """ols formula syntax: simply """
    formula = f'{response} ~ {factor_1}*{factor_2}'
    lm = ols(formula, data=df).fit()
    anova_res_df = anova_lm(lm)
    print(anova_res_df)
    _df = df.set_index(['predictor', 'kind'])
    _df.index = ['_'.join(pair) for pair in _df.index]
    tukey = pairwise_tukeyhsd(endog=_df['status'], groups=_df.index, alpha=0.01)
    tukey_res_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    tukey_res_df[['predictor1', 'kind1']] = tukey_res_df['group1'].str.split('_', expand=True)
    tukey_res_df[['predictor2', 'kind2']] = tukey_res_df['group2'].str.split('_', expand=True)
    t_res_df = tukey_res_df.loc[
        tukey_res_df['predictor1'] == tukey_res_df['predictor2'], ['predictor1', 'meandiff', 'p-adj', 'lower',
                                                                   'upper']].set_index('predictor1')
    t_res_df.index = t_res_df.index.astype(pd.CategoricalDtype(PreRetPredictors + PostRetPredictors, ordered=True))
    mean_acc = df.groupby(['predictor', 'kind'])['status'].mean().reset_index(). \
        pivot(index='predictor', columns='kind').droplevel(0, axis=1)
    t_res_df['p-adj'] = t_res_df['p-adj'].map('${:.3f}$'.format)
    t_res_df['meandiff'] = t_res_df['meandiff'].map('${:.3f}$'.format)
    t_res_df = t_res_df.drop(['upper', 'lower'], axis=1).assign(
        CI=t_res_df['lower'].map('[${:.3f}$'.format).str.cat(t_res_df['upper'].map('${:.3f}$]'.format), sep=','),
        intra=mean_acc['intra'].map('${:.3f}$'.format), inter=mean_acc['inter'].map('${:.3f}$'.format))

    print(
        t_res_df.sort_index()[['intra', 'inter', 'meandiff', 'CI', 'p-adj']].rename(LatexMacros).to_latex(
            float_format="%.3f", escape=False))
    return t_res_df.sort_index()


def calc_intra_inter_stats(intra_topic, inter_topic):
    dfx = intra_topic.pivot_table(index='predictor', aggfunc={'status': ['mean', 'std', len]})
    dfx.columns = dfx.columns.droplevel(0)
    dfy = inter_topic.pivot_table(index='predictor', aggfunc={'status': ['mean', 'std', len]})
    dfy.columns = dfy.columns.droplevel(0)
    df = pd.concat([intra_topic[['predictor', 'status', 'ev_diff']].assign(kind='intra'),
                    inter_topic[['predictor', 'status', 'ev_diff']].assign(kind='inter')]).rename(
        {'ev_diff': 'AP_Diff'}, axis=1).rename({'ev_diff': 'AP_Diff'}, axis=0)
    df['status'] = df['status'].astype(int)
    df = df.loc[df['predictor'] == 'uef-nqc']
    df['predictor'] = df['predictor'].astype(str)
    res_df = two_way_anova(df, 'status', 'predictor', 'kind')
    two_way_anova(df, 'status', 'AP_Diff', 'kind')
    two_way_anova(df, 'status', 'AP_Diff', 'predictor')

    for pred, _df in df.groupby('predictor'):
        print(pred)
        _df['predictor'] = _df['predictor'].astype(str)
        two_way_anova(_df, 'status', 'AP_Diff', 'kind')

    lm = ols('status ~ predictor + kind + AP_Diff + predictor:AP_Diff + kind:AP_Diff + kind:predictor', data=df).fit()
    print(anova_lm(lm))

    # Welch's t-test
    pvalues_df = pd.merge(dfx, dfy, on='predictor', suffixes=['_intra', '_inter']).assign(
        pval=lambda x: stats.ttest_ind_from_stats(mean1=x.mean_intra, std1=x.std_intra, nobs1=x.len_intra,
                                                  mean2=x.mean_inter, std2=x.std_inter, nobs2=x.len_inter,
                                                  equal_var=False).pvalue, kendall_intra=lambda x: 2 * x.mean_intra - 1,
        kendall_inter=lambda x: 2 * x.mean_inter - 1)
    print(pvalues_df.sort_values('predictor')[
              ['len_intra', 'len_inter', 'mean_intra', 'mean_inter', 'kendall_intra', 'kendall_inter',
               'pval']].to_string(float_format='%.3f'))
    pvalues_df = pd.merge(dfx, dfy, on='predictor', suffixes=['_intra', '_inter']).assign(
        pval=lambda x: stats.ttest_ind_from_stats(mean1=x.mean_intra, std1=x.std_intra, nobs1=x.len_intra,
                                                  mean2=x.mean_inter, std2=x.std_inter, nobs2=x.len_inter,
                                                  equal_var=False).pvalue)
    print(pvalues_df.sort_values('predictor')[['len_intra', 'len_inter', 'mean_intra', 'mean_inter', 'pval']].to_string(
        float_format='%.3f'))
    return res_df


def generate_sampled_inter_topic_df(prefix_path, ir_metric):
    predictors_group = 'all'
    eval_df = add_topic_to_qdf(read_eval_df(prefix_path, ir_metric)).set_index(['topic', 'qid'])
    predictions_df = read_prediction_files(prefix_path, predictors_group)
    return construct_inter_sampled_to_intra_df(eval_df, predictions_df)


@timer
def main():
    plt.set_loglevel("info")
    prefix = 'robust04_Lucene_indri_porter'
    prefix_path = os.path.join(results_dir, prefix)
    ir_metric = 'ap@1000'

    s_inter_topic_df = load_generate_pickle_df(f'inter_sampled_topic_pairwise_{ir_metric}_df.pkl',
                                               generate_sampled_inter_topic_df, prefix_path, ir_metric)
    inter_topic_eval(ir_metric, prefix_path)
    intra_topic_eval(ir_metric, prefix_path)

    load_all_pairs(ir_metric)
    return None


if __name__ == '__main__':
    PreRetPredictors = ['scq', 'avg-scq', 'max-scq', 'var', 'avg-var', 'max-var', 'avg-idf', 'max-idf']
    PostRetPredictors = ['clarity', 'smv', 'nqc', 'wig', 'qf', 'uef-clarity', 'uef-smv', 'uef-nqc', 'uef-wig', 'uef-qf']
    LatexMacros = {'scq': '\\Scq', 'avg-scq': '\\avgScq', 'max-scq': '\\maxScq', 'var': '\\Var', 'avg-var': '\\avgVar',
                   'max-var': '\\maxVar', 'max-idf': '\\maxIDF', 'avg-idf': '\\avgIDF',
                   'clarity': '\\clarity', 'smv': '\\smv', 'nqc': '\\nqc', 'wig': '\\wig',
                   'uef-clarity': '\\uef{\\clarity}', 'uef-smv': '\\uef{\\smv}', 'uef-nqc': '\\uef{\\nqc}',
                   'uef-wig': '\\uef{\\wig}', 'qf': '\\qf', 'uef-qf': '\\uef{\\qf}'}
    PlotNames = {'scq': 'SCQ', 'avg-scq': 'AvgSCQ', 'max-scq': 'MaxSCQ', 'var': 'SumVAR', 'avg-var': 'AvgVAR',
                 'max-var': 'MaxVAR', 'max-idf': 'MaxIDF', 'avg-idf': 'AvgIDF', 'clarity': 'Clarity', 'smv': 'SMV',
                 'nqc': 'NQC', 'wig': 'WIG', 'qf': 'QF', 'uef-clarity': 'UEF(Clarity)', 'uef-smv': 'UEF(SMV)',
                 'uef-nqc': 'UEF(NQC)', 'uef-wig': 'UEF(WIG)', 'uef-qf': 'UEF(QF)', 'ap@1000': 'AP',
                 'ndcg@10': 'nDCG@10'}
    cm = 1 / 2.54  # centimeters in inches
    # These settings should match the font used in LaTeX

    # seaborn_setup()

    fmt = {
        # "font.family": "Fira Sans",
        "font.family": ["Century Schoolbook Std", "Linux Libertine O", "serif", 'sans-serif'],
        # "font.serif": "Fira",
        # "font.serif": ["New Century Schoolbook", "Century Schoolbook L", "Century Schoolbook Std"],
        'font.size': 16,
        # 'font.size': 10,
        # "font.sans-serif": "Linux Biolinum",
        'figure.facecolor': (0.98, 0.98, 0.98),
        'text.color': '#23373b',
        'axes.labelcolor': '#23373b',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.figsize': (8, 4.5),
        'legend.borderaxespad': 2,
        # "axes.labelpad": 20.0
        "axes.labelpad": 10.0
    }
    plt.rcParams.update(fmt)

    PLOTS_COL_WRAP = 6

    with open('./duplicated_qids.txt') as f:
        DUPLICATED_QIDS = {line.rstrip('\n') for line in f}
    logger = Config.get_logger()
    results_dir = Config.RESULTS_DIR
    main()
