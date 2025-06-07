import os
import heapq
import torch
import pickle
import opts
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from torch_geometric.data import DataLoader

with open('attention_results/model_CausalGCN_begin/attention_results.pkl', 'rb') as f:  # 'rb'表示以二进制读取模式打开文件
    load_data = pickle.load(f)

with open('attention_results/model_CausalGCN_begin/test_results.pkl', 'rb') as f:
    results = pickle.load(f)

args = opts.parse_args()
save_path = "data"
test_set = torch.load(save_path + "/ATSD_v0_test_dataset_cor0.5.pt", weights_only=False)
test_loader = DataLoader(test_set, args.batch_size, shuffle=False)

preds = results['predictions']
true_labels = results['true_labels']
print('true_labels:', len(true_labels))
print('test results:', results)