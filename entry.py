import os
import torch
import numpy as np
import random
import argparse
import json
from preprocessing import DataPreprocessingMid, DataPreprocessingReady
from run import Run


def prepare(config_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_data_mid', type=int, default=0)
    parser.add_argument('--process_data_ready', type=int, default=0)
    parser.add_argument('--task', default='1')
    parser.add_argument('--base_model', default='MF')
    parser.add_argument('--seed', type=int, default=900)
    parser.add_argument('--ratio', nargs='+', type=float, default=[0.8, 0.2])
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_prototype', type=int, default=1, help='1=Enable Prototype, 0=Disable')
    parser.add_argument('--use_personalized', type=int, default=1, help='1=Enable Personalized Bridge, 0=Disable')
    parser.add_argument('--save_pretrained', type=int, default=1)
    parser.add_argument('--load_pretrained', type=int, default=1)
    parser.add_argument('--results_csv', default='experiment_results.csv',
                        help='CSV file to save results')
    parser.add_argument('--mid', default='')

    parser.add_argument('--num_prototypes', type=int, default=1)

    parser.add_argument('--variant_name', type=str, default='APMGR',
                        help='Name of the experiment variant')
    parser.add_argument('--tgt_gating', type=int, default=1)
    parser.add_argument('--use_dynamicDelta', type=int, default=1)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    with open(config_path, 'r') as f:
        config = json.load(f)
        config['base_model'] = args.base_model
        config['task'] = args.task
        config['ratio'] = args.ratio
        config['epoch'] = args.epoch
        config['lr'] = args.lr
        config['seed'] = args.seed
        config['use_prototype'] = args.use_prototype
        config['use_personalized'] = args.use_personalized
        config['use_dynamicDelta'] = args.use_dynamicDelta

        config['save_pretrained'] = args.save_pretrained
        config['load_pretrained'] = args.load_pretrained

        config['num_prototypes'] = args.num_prototypes

        config['tgt_gating'] = args.tgt_gating
        config['results_csv'] = args.results_csv
        config['variant_name'] = args.variant_name
        config['mid'] = args.mid
    return args, config


if __name__ == '__main__':
    config_path = 'config.json'
    args, config = prepare(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.process_data_mid:
        for dealing in ['Books', 'CDs_and_Vinyl', 'Movies_and_TV']:
            DataPreprocessingMid(config['root'], dealing).main()
    if args.process_data_ready:
        for ratio in [[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]]:
            for task in ['1', '2', '3']:
                DataPreprocessingReady(config['root'], config['src_tgt_pairs'], task, ratio).main()

    if not args.process_data_mid and not args.process_data_ready:
        Run(config).main()