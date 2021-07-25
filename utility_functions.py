import linecache
import os
import pickle
from bisect import bisect_left

import numpy as np
import pandas as pd

TREC_RES_COLUMNS = ['qid', 'iteration', 'docNo', 'rank', 'docScore', 'method']
TREC_QREL_COLUMNS = ['qid', 'iteration', 'docNo', 'rel']


def get_file_len(file_path):
    """Opens a file and counts the number of lines in it"""
    return sum(1 for _ in open(file_path))


def read_line(file_path, n):
    """Return a specific line n from a file, if the line doesn't exist, returns an empty string"""
    return linecache.getline(file_path, n)


def binary_search(list_, target):
    """Return the index of first value equal to target, if non found will raise a ValueError"""
    i = bisect_left(list_, target)
    if i != len(list_) and list_[i] == target:
        return i
    raise ValueError


def ensure_file(file):
    """Ensure a single file exists, returns the absolute path of the file if True or raises FileNotFoundError if not"""
    # tilde expansion
    file_path = os.path.normpath(os.path.expanduser(file))
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} doesn't exist")
    return file_path


def ensure_dir(file_path, create_if_not=True):
    """
    The function ensures the dir exists,
    if it doesn't it creates it and returns the path or raises FileNotFoundError
    In case file_path is an existing file, returns the path of the parent directory
    """
    # tilde expansion
    file_path = os.path.normpath(os.path.expanduser(file_path))
    if os.path.isfile(file_path):
        directory = os.path.dirname(file_path)
    else:
        directory = file_path
    if not os.path.exists(directory):
        if create_if_not:
            try:
                os.makedirs(directory)
            except FileExistsError:
                # This exception was added for multiprocessing, in case multiple process try to create the directory
                pass
        else:
            raise FileNotFoundError(f"The directory {directory} doesnt exist, create it or pass create_if_not=True")
    return directory


def transform_list_to_counts_dict(_list):
    counts = [_list.count(i) for i in _list]
    return {i: j for i, j in zip(_list, counts)}


def jaccard_similarity(set_1, set_2):
    return len(set_1.intersection(set_2)) / len(set_1.union(set_2))


def overlap_coefficient(set_1, set_2):
    return len(set_1.intersection(set_2)) / min(len(set_1), len(set_2))


def sorensen_dice_similarity(set_1, set_2):
    return 2 * len(set_1.intersection(set_2)) / (len(set_1) + len(set_2))


def add_topic_to_qdf(qdf: pd.DataFrame):
    """This function used to add a topic column to a queries DF"""
    columns = qdf.columns.to_list()
    if 'topic' not in columns:
        if 'qid' in columns:
            qdf = qdf.assign(topic=lambda x: x.qid.apply(lambda y: y.split('-')[0]))
        else:
            qdf = qdf.reset_index().assign(topic=lambda x: x.qid.apply(lambda y: y.split('-')[0]))
    columns = qdf.columns.to_list()
    return qdf.loc[:, columns[-1:] + columns[:-1]]


def read_trec_res_file(file_name):
    """
    Assuming data is in trec format results file with 6 columns, 'Qid entropy cross_entropy Score
    '"""
    data_df = pd.read_csv(file_name, delim_whitespace=True, header=None, index_col=0,
                          names=['qid', 'Q0', 'docNo', 'docRank', 'docScore', 'ind'],
                          dtype={'qid': str, 'Q0': str, 'docNo': str, 'docRank': int, 'docScore': float,
                                 'ind': str})
    data_df = data_df.filter(['qid', 'docNo', 'docRank', 'docScore'], axis=1)
    data_df.index = data_df.index.astype(str)
    data_df.sort_values(by=['qid', 'docRank'], ascending=True, inplace=True)
    return data_df


def pickle_save_obj(obj, file_name: str):
    if not file_name.endswith('.pkl'):
        file_name += '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def pickle_load_obj(file_name: str):
    if not file_name.endswith('.pkl'):
        file_name += '.pkl'
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def load_generate_pickle_df(file_name: str, func, *args):
    try:
        _df = pd.read_pickle(file_name)
    except FileNotFoundError:
        _df = func(*args)
        _df.to_pickle(file_name)
    return _df


def calc_ndcg(qrels_file, results_file, k, original=True, logger=None, base=2):
    """
    Calculates nDCG based on the corrected version by the original authors
    (the correction was proposed for the discount).
    :param qrels_file:
    :param results_file:
    :param k:
    :param original:
    :param logger:
    :param base:
    """
    qrels_df = pd.read_csv(qrels_file, delim_whitespace=True, names=TREC_QREL_COLUMNS). \
        sort_values(['qid', 'rel'], ascending=[True, False]).set_index(['qid', 'docNo'])
    results_df = pd.read_csv(results_file, delim_whitespace=True, names=TREC_RES_COLUMNS).sort_values(['qid', 'rank']). \
        groupby('qid').head(k)
    discount = np.log(np.arange(k) + 1) / np.log(base) + 1
    result = {}
    for qid, _df in results_df.groupby('qid'):
        docs = _df['docNo'].to_numpy()
        try:
            _qrels_df = qrels_df.loc[qid]
        except KeyError as err:
            if logger is None:
                print(f'query id {err} doesn\'t exist in the qrels file, skipping it')
            else:
                logger.warning(f'query id {err} doesn\'t exist in the qrels file, skipping it')
            continue
        if original:
            dcg = _qrels_df.reindex(docs)['rel'].fillna(0).to_numpy()
            idcg = (_qrels_df['rel'].head(k) / discount[:len(_qrels_df)]).sum()
        else:
            dcg = 2 ** _qrels_df.reindex(docs)['rel'].fillna(0).to_numpy() - 1
            idcg = ((2 ** _qrels_df['rel'].head(k) - 1) / discount[:len(_qrels_df)]).sum()
        result[qid] = (dcg / discount[:len(dcg)]).sum() / idcg
    res_df = pd.DataFrame.from_dict(result, orient='index', columns=[f'nDCG@{k}'])
    res_df.to_csv(rreplace(results_file, 'res', f'ndcg@{k}', 1), sep='\t', float_format='%.4f', header=False)
    print(res_df.to_string(float_format='%.6f'))
    print(res_df.mean())


def rreplace(string, old, new, count):
    # TODO: implement an extended version of str class
    return new.join(string.rsplit(old, count))
