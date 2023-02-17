import sys
sys.path.append('../')
from datetime import datetime
import gc
import argparse
from matchbox import autotuner 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='pytorch', help='The model version.')
    parser.add_argument('--config', type=str, default='../config/tuner_config.yaml', 
                        help='The config file for para tuning.')
    parser.add_argument('--gpu', nargs='+', default=[-1], help='The list of gpu indexes, -1 for cpu.')
    args = vars(parser.parse_args())
    gpu_list = args['gpu']
    version = args['version']

    # generate parameter space combinations
    config_dir = autotuner.enumerate_params(args['config'])
    autotuner.grid_search(version, config_dir, gpu_list)

