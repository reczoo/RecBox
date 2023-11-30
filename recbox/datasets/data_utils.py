import h5py
import os
import logging
import numpy as np
import multiprocessing as mp
import gc


def save_h5(darray_dict, data_path):
    logging.info("Saving data to h5: " + data_path)
    if not os.path.exists(os.path.dirname(data_path)):
        try:
            os.makedirs(os.path.dirname(data_path))
        except:
            pass
    with h5py.File(data_path, 'w') as hf:
        hf.attrs["num_samples"] = len(list(darray_dict.values())[0])
        for key, arr in darray_dict.items():
            hf.create_dataset(key, data=arr)


def load_h5(data_path, verbose=True):
    if verbose:
        logging.info('Loading data from h5: ' + data_path)
    data_dict = dict()
    with h5py.File(data_path, 'r') as hf:
        num_samples = hf.attrs["num_samples"]
        for key in hf.keys():
            data_dict[key] = hf[key][:]
    return data_dict, num_samples


def split_train_test(train_ddf=None, valid_ddf=None, test_ddf=None, valid_size=0, 
                     test_size=0, split_type="sequential"):
    num_samples = len(train_ddf)
    train_size = num_samples
    instance_IDs = np.arange(num_samples)
    if split_type == "random":
        np.random.shuffle(instance_IDs)
    if test_size > 0:
        if test_size < 1:
            test_size = int(num_samples * test_size)
        train_size = train_size - test_size
        test_ddf = train_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]
    if valid_size > 0:
        if valid_size < 1:
            valid_size = int(num_samples * valid_size)
        train_size = train_size - valid_size
        valid_ddf = train_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]
    if valid_size > 0 or test_size > 0:
        train_ddf = train_ddf.loc[instance_IDs, :].reset_index()
    return train_ddf, valid_ddf, test_ddf


def transform_h5(feature_encoder, ddf, filename, preprocess=False, block_size=0):
    def _transform_block(feature_encoder, df_block, filename, preprocess):
        if preprocess:
            df_block = feature_encoder.preprocess(df_block)
        darray_dict = feature_encoder.transform(df_block)
        save_h5(darray_dict, os.path.join(feature_encoder.data_dir, filename))

    if block_size > 0:
        pool = mp.Pool(mp.cpu_count() // 2)
        block_id = 0
        for idx in range(0, len(ddf), block_size):
            df_block = ddf[idx: (idx + block_size)]
            pool.apply_async(_transform_block, args=(feature_encoder, 
                                                     df_block, 
                                                     filename.replace('.h5', '_part_{}.h5'.format(block_id)),
                                                     preprocess))
            block_id += 1
        pool.close()
        pool.join()
    else:
        _transform_block(feature_encoder, ddf, filename, preprocess)


def build_dataset(feature_encoder, item_corpus=None, train_data=None, valid_data=None, 
                  test_data=None, valid_size=0, test_size=0, split_type="sequential", **kwargs):
    """ Build feature_map and transform h5 data """
    
    # Load csv data
    train_ddf = feature_encoder.read_csv(train_data, **kwargs)
    valid_ddf = None
    test_ddf = None

    # Split data for train/validation/test
    if valid_size > 0 or test_size > 0:
        valid_ddf = feature_encoder.read_csv(valid_data, **kwargs)
        test_ddf = feature_encoder.read_csv(test_data, **kwargs)
        train_ddf, valid_ddf, test_ddf = split_train_test(train_ddf, valid_ddf, test_ddf, 
                                                          valid_size, test_size, split_type)

    # fit feature_encoder
    corpus_ddf = feature_encoder.read_csv(item_corpus, **kwargs)
    corpus_ddf = feature_encoder.preprocess(corpus_ddf)
    train_ddf = feature_encoder.preprocess(train_ddf)
    feature_encoder.fit(train_ddf, corpus_ddf, **kwargs)

    # transform corpus_ddf
    item_corpus_dict = feature_encoder.transform(corpus_ddf)
    save_h5(item_corpus_dict, os.path.join(feature_encoder.data_dir, 'item_corpus.h5'))
    del item_corpus_dict, corpus_ddf
    gc.collect()

    # transform train_ddf
    block_size = int(kwargs.get("data_block_size", 0)) # Num of samples in a data block
    transform_h5(feature_encoder, train_ddf, 'train.h5', preprocess=False, block_size=block_size)
    del train_ddf
    gc.collect()

    # Transfrom valid_ddf
    if valid_ddf is None and (valid_data is not None):
        valid_ddf = feature_encoder.read_csv(valid_data, **kwargs)
    if valid_ddf is not None:
        transform_h5(feature_encoder, valid_ddf, 'valid.h5', preprocess=True, block_size=block_size)
        del valid_ddf
        gc.collect()

    # Transfrom test_ddf
    if test_ddf is None and (test_data is not None):
        test_ddf = feature_encoder.read_csv(test_data, **kwargs)
    if test_ddf is not None:
        transform_h5(feature_encoder, test_ddf, 'test.h5', preprocess=True, block_size=block_size)
        del test_ddf
        gc.collect()
    logging.info("Transform csv data to h5 done.")




