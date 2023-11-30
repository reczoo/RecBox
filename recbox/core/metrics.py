import numpy as np
import logging
import heapq
import itertools
import os
import multiprocessing as mp
from tqdm import tqdm
from ..utils.ann import FaissIndex


def evaluate_metrics(user_embs, 
                     item_embs, 
                     train_user2items, 
                     valid_user2items, 
                     query_indices,
                     metrics,
                     num_workers=1):
    logging.info("Evaluating metrics for {} users.".format(len(user_embs)))
    metric_funcs = []
    max_topk = 0
    for metric in metrics:
        try:
            metric_funcs.append(eval(metric))
            max_topk = max(max_topk, int(metric.split("k=")[-1].strip(")")))
        except:
            raise NotImplementedError('metrics={} not implemented.'.format(metric))
    
    faiss_index = FaissIndex(item_embs, dim=item_embs.shape[-1])
    chunk_size = min(1000, int(np.ceil(len(user_embs) / float(num_workers))))
    pool = mp.Pool(processes=num_workers)
    results = []
    for idx in range(0, len(user_embs), chunk_size):
        chunk_user_embs = user_embs[idx: (idx + chunk_size), :]
        chunk_query_indices = query_indices[idx: (idx + chunk_size)]
        if num_workers > 1:
            results.append(pool.apply_async(evaluate_block, 
                args=(chunk_user_embs, faiss_index, chunk_query_indices,
                train_user2items, valid_user2items, metric_funcs, max_topk)))
        else:
            results += evaluate_block(chunk_user_embs, faiss_index, chunk_query_indices, train_user2items, 
                                      valid_user2items, metric_funcs, max_topk)
    if num_workers > 1:
        pool.close()
        pool.join()
        results = [res.get() for res in results]
    average_result = np.average(np.array(results), axis=0).tolist()
    return_dict = dict(zip(metrics, average_result))
    logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in zip(metrics, average_result)))
    return return_dict


def evaluate_block(user_embs, faiss_index, query_indices, train_user2items, 
                   valid_user2items, metric_funcs, max_topk):
    # set to topk=500 here since the retrieval results may contain clicked items
    scores, indices = faiss_index.search(user_embs, topk=500)
    # mask out items already clicked in train data
    mask = np.zeros((user_embs.shape[0], faiss_index.index.ntotal))
    for i, query_index in enumerate(query_indices):
        train_items = train_user2items[query_index]
        mask[i, train_items] = 1
    mask = np.take_along_axis(mask, indices, axis=1) # ie, mask[np.arange(len(mask))[:, None], indices]
    scores += -1e9 * mask
    sorted_idxs = np.argsort(-scores, axis=1)
    topk_items = np.take_along_axis(indices, sorted_idxs, axis=1)[:, 0:max_topk] # get max_topk for metrics
    true_items = [valid_user2items[query_index] for query_index in query_indices]
    chunk_results = [[func(preds, labels) for func in metric_funcs] \
                     for preds, labels in zip(topk_items, true_items)]
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


class nRecall(object):
    """Recall metric normalized with max 1 at topk, like nDCG"""
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




