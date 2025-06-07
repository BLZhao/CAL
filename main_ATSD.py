from train import train_baseline_syn
from train_causal import train_causal_syn
from opts import setup_seed
import torch
import opts
import os
import utils
import pdb
import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')


def main():

    args = opts.parse_args()
    save_path = "/disk1/xujing.zbl/CAL/CAL-main/data"
    os.makedirs(save_path, exist_ok=True)
    train_set = torch.load(save_path + "/ATSD_v4_train_dataset_cor0.5.pt", weights_only=False)
    val_set = torch.load(save_path + "/ATSD_v4_val_dataset_cor0.5.pt", weights_only=False)
    test_set = torch.load(save_path + "/ATSD_v4_test_dataset_cor0.5.pt", weights_only=False)
    if args.model in ["GIN", "GCN", "GAT"]:
        model_func = opts.get_model(args)
        train_baseline_syn(train_set, val_set, test_set, model_func=model_func, args=args)
    elif args.model in ["CausalGCN", "CausalGIN", "CausalGAT", "CausalGCN_share", "CausalGIN_share", "CausalGAT_share", "CausalGCN_begin"]:
        model_func = opts.get_model(args)
        train_causal_syn(train_set, val_set, test_set, model_func=model_func, args=args)
    else:
        assert False

if __name__ == '__main__':
    main()