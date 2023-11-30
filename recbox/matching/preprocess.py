from collections import Counter
import itertools
import numpy as np
import pandas as pd
import h5py
import pickle
import os
import sklearn.preprocessing as sklearn_preprocess
from recbox.utils.torch_utils import pad_sequences


class Tokenizer(object):
    def __init__(self, topk_words=None, na_value=None, min_freq=1, splitter=None, 
                 lower=False, oov_token=0, max_len=0, padding="pre"):
        self._topk_words = topk_words
        self._na_value = na_value
        self._min_freq = min_freq
        self._lower = lower
        self._splitter = splitter
        self.oov_token = oov_token # use 0 for __OOV__
        self.vocab = dict()
        self.vocab_size = 0 # include oov and padding
        self.max_len = max_len
        self.padding = padding
        self.use_padding = None

    def fit(self, texts, use_padding=False):
        self.use_padding = use_padding
        word_counts = Counter()
        if self._splitter is not None: # for sequence
            max_len = 0
            for text in texts:
                if not pd.isnull(text):
                    text_split = text.split(self._splitter)
                    max_len = max(max_len, len(text_split))
                    for text in text_split:
                        word_counts[text] += 1
            if self.max_len == 0:
                self.max_len = max_len # use pre-set max_len otherwise
        else:
            tokens = list(texts)
            word_counts = Counter(tokens)
        self.build_vocab(word_counts)

    def build_vocab(self, word_counts):
        # sort to guarantee the determinism of index order
        word_counts = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
        words = []
        for token, count in word_counts:
            if count >= self._min_freq:
                if self._na_value is None or token != self._na_value:
                    words.append(token.lower() if self._lower else token)
        if self._topk_words:
            words = words[0:self._topk_words]
        self.vocab = dict((token, idx) for idx, token in enumerate(words, 1 + self.oov_token))
        self.vocab["__OOV__"] = self.oov_token
        if self.use_padding:
            self.vocab["__PAD__"] = len(words) + self.oov_token + 1 # use the last index for __PAD__
        self.vocab_size = len(self.vocab) + self.oov_token

    def encode_category(self, categories):
        category_indices = [self.vocab.get(x, self.oov_token) for x in categories]
        return np.array(category_indices)

    def encode_sequence(self, texts):
        sequence_list = []
        for text in texts:
            if pd.isnull(text) or text == '':
                sequence_list.append([])
            else:
                sequence_list.append([self.vocab.get(x, self.oov_token) for x in text.split(self._splitter)])
        sequence_list = pad_sequences(sequence_list, maxlen=self.max_len, value=self.vocab_size - 1,
                                      padding=self.padding, truncating=self.padding)
        return np.array(sequence_list)
    
    def load_pretrained_embedding(self, feature_name, key_dtype, pretrain_path, embedding_dim, output_path):
        with h5py.File(pretrain_path, 'r') as hf:
            keys = hf["key"][:]
            if issubclass(keys.dtype.type, key_dtype): # in case mismatch between int and str
                keys = keys.astype(key_dtype)
            pretrained_vocab = dict(zip(keys, range(len(keys))))
            pretrained_emb = hf["value"][:]
        # update vocab with pretrained keys, in case new token ids appear in validation or test set
        num_new_words = 0
        for word in pretrained_vocab.keys():
            if word not in self.vocab:
                self.vocab[word] = self.vocab.get("__PAD__", self.vocab_size) + num_new_words
                num_new_words += 1
        self.vocab_size += num_new_words
        embedding_matrix = np.random.normal(loc=0, scale=1.e-4, size=(self.vocab_size, embedding_dim))
        if "__PAD__" in self.vocab:
            self.vocab["__PAD__"] = self.vocab_size - 1
            embedding_matrix[-1, :] = 0 # set as zero vector for PAD
        for word in pretrained_vocab.keys():
            embedding_matrix[self.vocab[word]] = pretrained_emb[pretrained_vocab[word]]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, 'a') as hf:
            hf.create_dataset(feature_name, data=embedding_matrix)

    def load_vocab_from_file(self, vocab_file):
        with open(vocab_file, 'r') as fid:
            word_counts = json.load(fid)
        self.build_vocab(word_counts)

    def set_vocab(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(self.vocab) + self.oov_token
            
        
class Normalizer(object):
    def __init__(self, normalizer_name):
        if normalizer_name in ['StandardScaler', 'MinMaxScaler']:
            self.normalizer = getattr(sklearn_preprocess, normalizer_name)()
        else:
            raise NotImplementedError('normalizer={}'.format(normalizer_name))

    def fit(self, X):
        null_index = np.isnan(X)
        self.normalizer.fit(X[~null_index].reshape(-1, 1))

    def transform(self, X):
        return self.normalizer.transform(X.reshape(-1, 1)).flatten()

