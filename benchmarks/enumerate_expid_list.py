import sys
sys.path.append('../')
from datetime import datetime
import gc
import pandas as pd
import argparse
from deem import autotuner 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, 
                        default='../config/tuner_config.yaml', 
                        help='The config file for para tuning.')
    parser.add_argument('--exclude', type=str, 
                        default='', 
                        help='The experiment_result.csv file to exclude finished expid.')
    args = vars(parser.parse_args())
    exclude_expid = []
    if args['exclude'] != '':
        result_df = pd.read_csv(args['exclude'], header=None)
        expid_df = result_df.iloc[:, 2].map(lambda x: x.replace('[exp_id] ', ''))
        exclude_expid = expid_df.tolist()
    # generate parameter space combinations
    config_dir = autotuner.enumerate_params(args['config'], exclude_expid=exclude_expid)

