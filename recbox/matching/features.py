import numpy as np
from collections import Counter, OrderedDict
import pandas as pd
import pickle
import os
import logging
import json
from collections import defaultdict
from .preprocess import Tokenizer, Normalizer


class FeatureMap(object):
    def __init__(self, dataset_id, data_dir, query_index, corpus_index, label_name, version="pytorch"):
        self.data_dir = data_dir
        self.dataset_id = dataset_id
        self.version = version
        self.num_fields = 0
        self.num_features = 0
        self.num_items = 0
        self.query_index = query_index
        self.corpus_index = corpus_index
        self.label_name = label_name
        self.feature_specs = OrderedDict()

    def load(self, json_file):
        logging.info("Load feature_map from json: " + json_file)
        with open(json_file, "r", encoding="utf-8") as fd:
            feature_map = json.load(fd, object_pairs_hook=OrderedDict)
        if feature_map["dataset_id"] != self.dataset_id:
            raise RuntimeError("dataset_id={} does not match to feature_map!".format(self.dataset_id))
        self.num_fields = feature_map["num_fields"]
        self.num_features = feature_map.get("num_features", None)
        self.label_name = feature_map.get("label_name", None)
        self.feature_specs = OrderedDict(feature_map["feature_specs"])

    def save(self, json_file):
        logging.info("Save feature_map to json: " + json_file)
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        feature_map = OrderedDict()
        feature_map["dataset_id"] = self.dataset_id
        feature_map["num_fields"] = self.num_fields
        feature_map["num_features"] = self.num_features
        feature_map["num_items"] = self.num_items
        feature_map["query_index"] = self.query_index
        feature_map["corpus_index"] = self.corpus_index
        feature_map["label_name"] = self.label_name
        feature_map["feature_specs"] = self.feature_specs
        with open(json_file, "w", encoding="utf-8") as fd:
            json.dump(feature_map, fd, indent=4)

    def get_num_fields(self, feature_source=[]):
        if type(feature_source) != list:
            feature_source = [feature_source]
        num_fields = 0
        for feature, feature_spec in self.feature_specs.items():
            if not feature_source or feature_spec["source"] in feature_source:
                num_fields += 1
        return num_fields


