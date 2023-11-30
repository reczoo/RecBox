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


class FeatureEncoder(object):
    def __init__(self,
                 feature_cols=[],
                 label_col={},
                 dataset_id=None,
                 data_root="../data/",
                 version="pytorch",
                 **kwargs):
        logging.info("Set up feature encoder...")
        self.data_dir = os.path.join(data_root, dataset_id)
        self.pickle_file = os.path.join(self.data_dir, "feature_encoder.pkl")
        self.json_file = os.path.join(self.data_dir, "feature_map.json")
        self.feature_cols = self._complete_feature_cols(feature_cols)
        self.label_col = label_col
        self.version = version
        self.feature_map = FeatureMap(dataset_id, self.data_dir, kwargs["query_index"], 
                                      kwargs["corpus_index"], self.label_col["name"], version)
        self.dtype_dict = dict((feat["name"], eval(feat["dtype"]) if type(feat["dtype"]) == str else feat["dtype"]) 
                               for feat in self.feature_cols + [self.label_col])
        self.encoders = dict()

    def _complete_feature_cols(self, feature_cols):
        full_feature_cols = []
        for col in feature_cols:
            name_or_namelist = col["name"]
            if isinstance(name_or_namelist, list):
                for _name in name_or_namelist:
                    _col = col.copy()
                    _col["name"] = _name
                    full_feature_cols.append(_col)
            else:
                full_feature_cols.append(col)
        return full_feature_cols

    def read_csv(self, data_path, sep=",", nrows=None, **kwargs):
        if data_path is not None:
            logging.info("Reading file: " + data_path)
            usecols_fn = lambda x: x in self.dtype_dict
            ddf = pd.read_csv(data_path, sep=sep, usecols=usecols_fn, 
                              dtype=object, memory_map=True, nrows=nrows)
            return ddf
        else:
            return None

    def preprocess(self, ddf):
        logging.info("Preprocess feature columns...")
        if self.feature_map.query_index in ddf.columns: # for train/val/test ddf
            all_cols = [self.label_col] + [col for col in self.feature_cols[::-1] if col.get("source") != "item"]
        else: # for item_corpus ddf
            all_cols = [col for col in self.feature_cols[::-1] if col.get("source") == "item"]
        for col in all_cols:
            name = col["name"]
            if name in ddf.columns and ddf[name].isnull().values.any():
                ddf[name] = self._fill_na_(col, ddf[name])
            if "preprocess" in col and col["preprocess"] != "":
                preprocess_fn = getattr(self, col["preprocess"])
                ddf[name] = preprocess_fn(ddf, name)
            ddf[name] = ddf[name].astype(self.dtype_dict[name])
        active_cols = [col["name"] for col in all_cols if col.get("active") != False]
        ddf = ddf.loc[:, active_cols]
        return ddf

    def _fill_na_(self, col, series):
        na_value = col.get("na_value")
        if na_value is not None:
            return series.fillna(na_value)
        elif col["dtype"] in ["str", str]:
            return series.fillna("")
        else:
            raise RuntimeError("Feature column={} requires to assign na_value!".format(col["name"]))

    def fit(self, train_ddf, corpus_ddf, min_categr_count=1, num_buckets=10, **kwargs):      
        logging.info("Fit feature encoder...") 
        self.feature_map.num_items = len(corpus_ddf)
        train_ddf = train_ddf.join(corpus_ddf, on=self.feature_map.corpus_index)
        for col in self.feature_cols:
            name = col["name"]
            if col["active"]:
                self.feature_map.num_fields += 1
                logging.info("Processing column: {}".format(col))
                if col["type"] == "index":
                    self.fit_index_col(col)
                elif col["type"] == "numeric":
                    self.fit_numeric_col(col, train_ddf[name].values)
                elif col["type"] == "categorical":
                    self.fit_categorical_col(col, train_ddf[name].values, 
                                             min_categr_count=min_categr_count,
                                             num_buckets=num_buckets)
                elif col["type"] == "sequence":
                    self.fit_sequence_col(col, train_ddf[name].values, 
                                          min_categr_count=min_categr_count)
                else:
                    raise NotImplementedError("feature_col={}".format(feature_col))
        self.save_pickle(self.pickle_file)
        self.feature_map.save(self.json_file)
        logging.info("Set feature encoder done.")

    def fit_index_col(self, feature_col):
        name = feature_col["name"]
        feature_type = feature_col["type"]
        feature_source = feature_col.get("source", "")
        self.feature_map.feature_specs[name] = {"source": feature_source,
                                                "type": feature_type}  

    def fit_numeric_col(self, feature_col, data_vector):
        name = feature_col["name"]
        feature_type = feature_col["type"]
        feature_source = feature_col.get("source", "")
        self.feature_map.feature_specs[name] = {"source": feature_source,
                                                "type": feature_type}
        if "embedding_callback" in feature_col:
            self.feature_map.feature_specs[name]["embedding_callback"] = feature_col["embedding_callback"]
        if "normalizer" in feature_col:
            normalizer = Normalizer(feature_col["normalizer"])
            normalizer.fit(data_vector)
            self.encoders[name + "_normalizer"] = normalizer
        self.feature_map.num_features += 1
        
    def fit_categorical_col(self, feature_col, data_vector, min_categr_count=1, num_buckets=10):
        name = feature_col["name"]
        feature_type = feature_col["type"]
        feature_source = feature_col.get("source", "")
        min_categr_count = feature_col.get("min_categr_count", min_categr_count)
        self.feature_map.feature_specs[name] = {"source": feature_source,
                                                "type": feature_type,
                                                "min_categr_count": min_categr_count}
        if "embedding_callback" in feature_col:
            self.feature_map.feature_specs[name]["embedding_callback"] = feature_col["embedding_callback"]
        if "embedding_dim" in feature_col:
            self.feature_map.feature_specs[name]["embedding_dim"] = feature_col["embedding_dim"]
        if "category_encoder" not in feature_col:
            tokenizer = Tokenizer(min_freq=min_categr_count, 
                                  na_value=feature_col.get("na_value", ""))
            if "share_embedding" in feature_col:
                self.feature_map.feature_specs[name]["share_embedding"] = feature_col["share_embedding"]
                tokenizer.set_vocab(self.encoders["{}_tokenizer".format(feature_col["share_embedding"])].vocab)
            else:
                if self._whether_share_emb_with_sequence(name):
                    tokenizer.fit(data_vector, use_padding=True)
                    if "pretrained_emb" not in feature_col:
                        self.feature_map.feature_specs[name]["padding_idx"] = tokenizer.vocab_size - 1
                else:
                    tokenizer.fit(data_vector, use_padding=False)
            if "pretrained_emb" in feature_col:
                logging.info("Loading pretrained embedding: " + name)
                self.feature_map.feature_specs[name]["pretrained_emb"] = "pretrained_{}.h5".format(name)
                self.feature_map.feature_specs[name]["freeze_emb"] = feature_col.get("freeze_emb", True)
                tokenizer.load_pretrained_embedding(name,
                                                    self.dtype_dict[name],
                                                    feature_col["pretrained_emb"], 
                                                    feature_col["embedding_dim"],
                                                    os.path.join(self.data_dir, "pretrained_{}.h5".format(name)))
                if tokenizer.use_padding: # update to account pretrained keys
                    self.feature_map.feature_specs[name]["padding_idx"] = tokenizer.vocab_size - 1
            self.encoders[name + "_tokenizer"] = tokenizer
            self.feature_map.feature_specs[name]["vocab_size"] = tokenizer.vocab_size
            self.feature_map.num_features += tokenizer.vocab_size
        else:
            category_encoder = feature_col["category_encoder"]
            self.feature_map.feature_specs[name]["category_encoder"] = category_encoder
            if category_encoder == "quantile_bucket": # transform numeric value to bucket
                num_buckets = feature_col.get("num_buckets", num_buckets)
                qtf = sklearn_preprocess.QuantileTransformer(n_quantiles=num_buckets + 1)
                qtf.fit(data_vector)
                boundaries = qtf.quantiles_[1:-1]
                self.feature_map.feature_specs[name]["vocab_size"] = num_buckets
                self.feature_map.num_features += num_buckets
                self.encoders[name + "_boundaries"] = boundaries
            elif category_encoder == "hash_bucket":
                num_buckets = feature_col.get("num_buckets", num_buckets)
                uniques = Counter(data_vector)
                num_buckets = min(num_buckets, len(uniques))
                self.feature_map.feature_specs[name]["vocab_size"] = num_buckets
                self.encoders[name + "_num_buckets"] = num_buckets
                self.feature_map.num_features += num_buckets
            else:
                raise NotImplementedError("category_encoder={} not supported.".format(category_encoder))

    def fit_sequence_col(self, feature_col, data_vector, min_categr_count=1):
        name = feature_col["name"]
        feature_type = feature_col["type"]
        feature_source = feature_col.get("source", "")
        min_categr_count = feature_col.get("min_categr_count", min_categr_count)
        self.feature_map.feature_specs[name] = {"source": feature_source,
                                                "type": feature_type,
                                                "min_categr_count": min_categr_count}
        embedding_callback = feature_col.get("embedding_callback", "layers.MaskedAveragePooling()")
        if embedding_callback not in [None, "null", "None", "none"]:
            self.feature_map.feature_specs[name]["embedding_callback"] = embedding_callback
        splitter = feature_col.get("splitter", " ")
        na_value = feature_col.get("na_value", "")
        max_len = feature_col.get("max_len", 0)
        padding = feature_col.get("padding", "post") # "post" or "pre"
        tokenizer = Tokenizer(min_freq=min_categr_count, splitter=splitter, 
                              na_value=na_value, max_len=max_len, padding=padding)
        if "share_embedding" in feature_col:
            self.feature_map.feature_specs[name]["share_embedding"] = feature_col["share_embedding"]
            tokenizer.set_vocab(self.encoders["{}_tokenizer".format(feature_col["share_embedding"])].vocab)
        else:
            tokenizer.fit(data_vector, use_padding=True)
        if "pretrained_emb" in feature_col:
            logging.info("Loading pretrained embedding: " + name)
            self.feature_map.feature_specs[name]["pretrained_emb"] = "pretrained_{}.h5".format(name)
            self.feature_map.feature_specs[name]["freeze_emb"] = feature_col.get("freeze_emb", True)
            tokenizer.load_pretrained_embedding(name,
                                                self.dtype_dict[name],
                                                feature_col["pretrained_emb"], 
                                                feature_col["embedding_dim"],
                                                os.path.join(self.data_dir, "pretrained_{}.h5".format(name)))
        self.encoders[name + "_tokenizer"] = tokenizer
        self.feature_map.feature_specs[name].update({"padding_idx": tokenizer.vocab_size - 1,
                                                     "vocab_size": tokenizer.vocab_size,
                                                     "max_len": tokenizer.max_len})
        self.feature_map.num_features += tokenizer.vocab_size

    def transform(self, ddf):
        logging.info("Transform feature columns...")
        data_dict = dict()
        for feature, feature_spec in self.feature_map.feature_specs.items():
            if feature in ddf.columns:
                feature_type = feature_spec["type"]
                data_vector = ddf.loc[:, feature].values
                if feature_type == "index":
                    data_dict[feature] = data_vector
                elif feature_type == "numeric":
                    data_vector = data_vector.astype(float)
                    normalizer = self.encoders.get(feature + "_normalizer")
                    if normalizer:
                         data_vector = normalizer.transform(data_vector)
                    data_dict[feature] = data_vector
                elif feature_type == "categorical":
                    category_encoder = feature_spec.get("category_encoder")
                    if category_encoder is None:
                        data_dict[feature] = self.encoders.get(feature + "_tokenizer").encode_category(data_vector)
                    elif encoder == "numeric_bucket":
                        raise NotImplementedError
                    elif encoder == "hash_bucket":
                        raise NotImplementedError
                elif feature_type == "sequence":
                    data_dict[feature] = self.encoders.get(feature + "_tokenizer").encode_sequence(data_vector)
        label = self.label_col["name"]
        if label in ddf.columns:
            data_dict[label] = ddf.loc[:, label].values.astype(float)
        return data_dict

    def _whether_share_emb_with_sequence(self, feature):
        for col in self.feature_cols:
            if col.get("share_embedding", None) == feature and col["type"] == "sequence":
                return True
        return False

    def load_pickle(self, pickle_file=None):
        """ Load feature encoder from cache """
        if pickle_file is None:
            pickle_file = self.pickle_file
        logging.info("Load feature_encoder from pickle: " + pickle_file)
        if os.path.exists(pickle_file):
            pickled_feature_encoder = pickle.load(open(pickle_file, "rb"))
            if pickled_feature_encoder.feature_map.dataset_id == self.feature_map.dataset_id:
                pickled_feature_encoder.version = self.version
                return pickled_feature_encoder
        raise IOError("pickle_file={} not valid.".format(pickle_file))

    def save_pickle(self, pickle_file):
        logging.info("Pickle feature_encode: " + pickle_file)
        if not os.path.exists(os.path.dirname(pickle_file)):
            os.makedirs(os.path.dirname(pickle_file))
        pickle.dump(self, open(pickle_file, "wb"))

      



