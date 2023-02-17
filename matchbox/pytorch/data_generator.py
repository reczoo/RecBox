import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torch
import logging
from collections import defaultdict
import multiprocessing as mp
import os
from ..datasets.data_utils import load_h5
import pickle
import shutil


class TrainDataset(Dataset):
    def __init__(self, feature_map, data_path, item_corpus):
        self.data_dict, self.num_samples = load_h5(data_path)
        self.item_corpus_dict, self.num_items = load_h5(item_corpus)
        self.labels = self.data_dict[feature_map.label_name]
        self.pos_item_indexes = self.data_dict[feature_map.corpus_index]
        self.all_item_indexes = self.data_dict[feature_map.corpus_index]
        
    def __getitem__(self, index):
        user_dict = self.slice_array_dict(self.data_dict, index)
        item_indexes = self.all_item_indexes[index, :]
        item_dict = self.slice_array_dict(self.item_corpus_dict, item_indexes)
        label = self.labels[index]
        return user_dict, item_dict, label, item_indexes
    
    def __len__(self):
        return self.num_samples

    def slice_array_dict(self, array_dict, slice_index):
        return dict((k, v[slice_index]) for k, v in array_dict.items())


def get_user2items_dict(data_dict, feature_map):
    user2items_dict = defaultdict(list)
    for query_index, corpus_index in zip(data_dict[feature_map.query_index], 
                                         data_dict[feature_map.corpus_index]):
        user2items_dict[query_index].append(corpus_index)
    return user2items_dict


def collate_fn_unique(batch): 
    # TODO: check correctness
    user_dict, item_dict, labels, item_indexes = default_collate(batch)
    num_negs = item_indexes.size(1) - 1
    unique, inverse_indexes = torch.unique(item_indexes.flatten(), return_inverse=True, sorted=True)
    perm = torch.arange(inverse_indexes.size(0), dtype=inverse_indexes.dtype, device=inverse_indexes.device)
    inverse_indexes, perm = inverse_indexes.flip([0]), perm.flip([0])
    unique_indexes = inverse_indexes.new_empty(unique.size(0)).scatter_(0, inverse_indexes, perm) # obtain return_indicies in np.unique
    # reshape item data with (b*(num_neg + 1) x input_dim)
    for k, v in item_dict.items():
        item_dict[k] = v.flatten(end_dim=1)[unique_indexes]
    # add negative labels
    labels = torch.cat([labels.view(-1, 1).float(), torch.zeros((labels.size(0), num_negs))], dim=1)
    return user_dict, item_dict, labels, inverse_indexes


def collate_fn(batch):
    user_dict, item_dict, labels, item_indexes = default_collate(batch)
    num_negs = item_indexes.size(1) - 1
    # reshape item data with (b*(num_neg + 1) x input_dim)
    for k, v in item_dict.items():
        item_dict[k] = v.flatten(end_dim=1)
    # add negative labels
    labels = torch.cat([labels.view(-1, 1).float(), torch.zeros((labels.size(0), num_negs))], dim=1)
    return user_dict, item_dict, labels, None


def sampling_block(num_items, block_query_indexes, num_negs, user2items_dict, 
                   sampling_probs=None, ignore_pos_items=False, seed=None, dump_path=None):
    if seed is not None:
        np.random.seed(seed) # used in multiprocessing
    if sampling_probs is None:
        sampling_probs = np.ones(num_items) / num_items # uniform sampling
    if ignore_pos_items:
        sampled_items = []
        for query_index in block_query_indexes:
            pos_items = user2items_dict[query_index]
            probs = np.array(sampling_probs)
            probs[pos_items] = 0
            probs = probs / np.sum(probs) # renomalize to sum 1
            sampled_items.append(np.random.choice(num_items, size=num_negs, replace=True, p=probs))
        sampled_array = np.array(sampled_items)
    else:
        sampled_array = np.random.choice(num_items,
                                         size=(len(block_query_indexes), num_negs), 
                                         replace=True)
    if dump_path is not None:
        # To fix bug in multiprocessing: https://github.com/xue-pai/Open-CF-Benchmarks/issues/1
        pickle_array(sampled_array, dump_path)
    else:
        return sampled_array


def pickle_array(array, path):
    with open(path, "wb") as fout:
        pickle.dump(array, fout, pickle.HIGHEST_PROTOCOL)


def load_pickled_array(path):
    with open(path, "rb") as fin:
        return pickle.load(fin)


class TrainGenerator(DataLoader):
    # reference https://cloud.tencent.com/developer/article/1010247
    def __init__(self, feature_map, data_path, item_corpus, batch_size=32, shuffle=True, 
                 num_workers=1, num_negs=0, compress_duplicate_items=False, **kwargs):
        if type(data_path) == list:
            data_path = data_path[0]
            self.num_blocks = 1
        self.num_negs = num_negs
        self.dataset = TrainDataset(feature_map, data_path, item_corpus)
        super(TrainGenerator, self).__init__(dataset=self.dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers,
                                             collate_fn=collate_fn_unique if compress_duplicate_items else collate_fn)
        self.user2items_dict = get_user2items_dict(self.dataset.data_dict, feature_map)
        self.query_indexes = self.dataset.data_dict[feature_map.query_index]
        # delete some columns to speed up batch generator
        del self.dataset.data_dict[feature_map.query_index]
        del self.dataset.data_dict[feature_map.corpus_index]
        del self.dataset.data_dict[feature_map.label_name]
        self.num_samples = len(self.dataset)
        self.num_batches = int(np.ceil(self.num_samples * 1.0 / batch_size))
        self.sampling_num_process = kwargs.get("sampling_num_process", 1)
        self.ignore_pos_items = kwargs.get("ignore_pos_items", False)
        self.fix_sampling_seeds = kwargs.get("fix_sampling_seeds", True)

    def __iter__(self):
        self.negative_sampling()
        iter = super(TrainGenerator, self).__iter__()
        while True:
            try:
                yield next(iter) # a batch iterator
            except StopIteration:
                return

    def __len__(self):
        return self.num_batches

    def negative_sampling(self):
        if self.num_negs > 0:
            logging.info("Negative sampling num_negs={}".format(self.num_negs))
            sampling_probs = None # set it to item popularity when using importance sampling
            if self.sampling_num_process > 1:
                chunked_query_indexes = np.array_split(self.query_indexes, self.sampling_num_process)
                if self.fix_sampling_seeds:
                    seeds = np.random.randint(1000000, size=self.sampling_num_process)
                else:
                    seeds = [None] * self.sampling_num_process
                pool = mp.Pool(self.sampling_num_process)
                block_result = []
                os.makedirs("./tmp/pid_{}/".format(os.getpid()), exist_ok=True)
                dump_paths = ["./tmp/pid_{}/part_{}.pkl".format(os.getpid(), idx) for idx in range(len(chunked_query_indexes))]
                for idx, block_query_indexes in enumerate(chunked_query_indexes):
                    pool.apply_async(sampling_block, args=(self.dataset.num_items, 
                                                           block_query_indexes, 
                                                           self.num_negs, 
                                                           self.user2items_dict, 
                                                           sampling_probs, 
                                                           self.ignore_pos_items,
                                                           seeds[idx],
                                                           dump_paths[idx]))
                pool.close()
                pool.join()
                block_result = [load_pickled_array(dump_paths[idx]) for idx in range(len(chunked_query_indexes))]
                shutil.rmtree("./tmp/pid_{}/".format(os.getpid()))
                neg_item_indexes = np.vstack(block_result)
            else:
                neg_item_indexes = sampling_block(self.dataset.num_items, 
                                                  self.query_indexes, 
                                                  self.num_negs, 
                                                  self.user2items_dict, 
                                                  sampling_probs,
                                                  self.ignore_pos_items)
            self.dataset.all_item_indexes = np.hstack([self.dataset.pos_item_indexes.reshape(-1, 1), 
                                                       neg_item_indexes])
            logging.info("Negative sampling done")


class TestDataset(Dataset):
    def __init__(self, data_path):
        self.data_dict, self.num_samples = load_h5(data_path)

    def __getitem__(self, index):
        batch_dict = self.slice_array_dict(index)
        return batch_dict
    
    def __len__(self):
        return self.num_samples

    def slice_array_dict(self, slice_index):
        return dict((k, v[slice_index]) for k, v in self.data_dict.items())


class TestGenerator(object):
    def __init__(self, feature_map, data_path, item_corpus, batch_size=32, shuffle=False, 
                 num_workers=1, **kwargs):
        if type(data_path) == list:
            data_path = data_path[0]
            self.num_blocks = 1
        user_dataset = TestDataset(data_path)
        self.user2items_dict = get_user2items_dict(user_dataset.data_dict, feature_map)
        # pick users of unique query_index
        self.query_indexes, unique_rows = np.unique(user_dataset.data_dict[feature_map.query_index], 
                                                    return_index=True)
        user_dataset.num_samples = len(unique_rows)
        self.num_samples = len(user_dataset)
        # delete some columns to speed up batch generator
        del user_dataset.data_dict[feature_map.query_index]
        del user_dataset.data_dict[feature_map.corpus_index]
        del user_dataset.data_dict[feature_map.label_name]
        for k, v in user_dataset.data_dict.items():
            user_dataset.data_dict[k] = v[unique_rows]
        item_dataset = TestDataset(item_corpus)
        self.user_loader = DataLoader(dataset=user_dataset, batch_size=batch_size,
                                      shuffle=shuffle, num_workers=num_workers)
        self.item_loader = DataLoader(dataset=item_dataset, batch_size=batch_size,
                                      shuffle=shuffle, num_workers=num_workers)

        
