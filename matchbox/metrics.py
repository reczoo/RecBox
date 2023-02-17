import numpy as np
import logging
import heapq
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def evaluate_metrics(user_embs, 
                     item_embs, 
                     train_user2items, 
                     valid_user2items, 
                     query_indexes,
                     metrics,
                     parallel=False):
    logging.info("Evaluating metrics for {:d} users...".format(len(user_embs)))
    metric_callers = []
    max_topk = 0
    for metric in metrics:
        try:
            metric_callers.append(eval(metric))
            max_topk = max(max_topk, int(metric.split("k=")[-1].strip(")")))
        except:
            raise NotImplementedError('metrics={} not implemented.'.format(metric))
    
    if parallel:
        num_workers = 2
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            chunk_size = int(np.ceil(len(user_embs) / float(num_workers)))
            tasks = []
            for idx in range(0, len(user_embs), chunk_size):
                chunk_user_embs = user_embs[idx: (idx + chunk_size), :]
                chunk_query_indexes = query_indexes[idx: (idx + chunk_size)]
                tasks.append(executor.submit(evaluate_block, chunk_user_embs, item_embs, chunk_query_indexes,
                                             train_user2items, valid_user2items, metric_callers, max_topk))
            results = [res for future in tqdm(as_completed(tasks), total=len(tasks)) for res in future.result()]
    else:
        results = evaluate_block(user_embs, item_embs, query_indexes, train_user2items, 
                                 valid_user2items, metric_callers, max_topk)
    average_result = np.average(np.array(results), axis=0).tolist()
    return_dict = dict(zip(metrics, average_result))
    logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in zip(metrics, average_result)))
    return return_dict


def evaluate_block(chunk_user_embs, item_embs, chunk_query_indexes, train_user2items, 
                   valid_user2items, metric_callers, max_topk):
    sim_matrix = np.dot(chunk_user_embs, item_embs.T)
    for i, query_index in enumerate(chunk_query_indexes):
        train_items = train_user2items[query_index]
        sim_matrix[i, train_items] = -np.inf # remove clicked items in train data
    item_indexes = np.argpartition(-sim_matrix, max_topk)[:, 0:max_topk]
    sim_matrix = sim_matrix[np.arange(item_indexes.shape[0])[:, None], item_indexes]
    sorted_idxs = np.argsort(-sim_matrix, axis=1)
    topk_items_chunk = item_indexes[np.arange(sorted_idxs.shape[0])[:, None], sorted_idxs]
    true_items_chunk = [valid_user2items[query_index] for query_index in chunk_query_indexes]
    chunk_results = [[fn(topk_items, true_items) for fn in metric_callers] \
                     for topk_items, true_items in zip(topk_items_chunk, true_items_chunk)]
    return chunk_results


class Recall(object):
    """Recall metric."""
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        hit_items = set(true_items) & set(topk_items)
        recall = len(hit_items) / (len(true_items) + 1e-12)
        return recall


class NormalizedRecall(object):
    """Recall metric normalized to max 1."""
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        hit_items = set(true_items) & set(topk_items)
        recall = len(hit_items) / min(self.topk, len(true_items) + 1e-12)
        return recall


class Precision(object):
    """Precision metric."""
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        hit_items = set(true_items) & set(topk_items)
        precision = len(hit_items) / (self.topk + 1e-12)
        return precision


class F1(object):
    def __init__(self, k=1):
        self.precision_k = Precision(k)
        self.recall_k = Recall(k)

    def __call__(self, topk_items, true_items):
        p = self.precision_k(topk_items, true_items)
        r = self.recall_k(topk_items, true_items)
        f1 = 2 * p * r / (p + r + 1e-12)
        return f1


class DCG(object):
    """ Calculate discounted cumulative gain
    """
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        true_items = set(true_items)
        dcg = 0
        for i, item in enumerate(topk_items):
            if item in true_items:
                dcg += 1 / np.log(2 + i)
        return dcg


class NDCG(object):
    """Normalized discounted cumulative gain metric."""
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        dcg_fn = DCG(k=self.topk)
        idcg = dcg_fn(true_items[:self.topk], true_items)
        dcg = dcg_fn(topk_items, true_items)
        return dcg / (idcg + 1e-12)


class MRR(object):
    """MRR metric"""
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        true_items = set(true_items)
        mrr = 0
        for i, item in enumerate(topk_items):
            if item in true_items:
                mrr += 1 / (i + 1.0)
        return mrr


class HitRate(object):
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        hit_items = set(true_items) & set(topk_items)
        hit_rate = 1 if len(hit_items) > 0 else 0
        return hit_rate


class MAP(object):
    """
    Calculate mean average precision.
    """
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        true_items = set(true_items)
        pos = 0
        precision = 0
        for i, item in enumerate(topk_items):
            if item in true_items:
                pos += 1
                precision += pos / (i + 1.0)
        return precision / (pos + 1e-12)




