import matplotlib.pyplot as plt

import featgen
import gengraph
import random
from torch_geometric.utils import from_networkx
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import argparse
import opts
import os
from torch_geometric.utils.convert import to_networkx
from sklearn.model_selection import StratifiedKFold
import networkx as nx
from networkx.algorithms import community
import scipy.sparse as sp
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import InMemoryDataset, Data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--root_path', type=str, default='disk1/timeseries_dataset/ATSD_v4_cleaned', help='root path of the data file')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=512,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--min-lr', type=float, default=1e-6,
                    help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--num-atoms', type=int, default=23,
                    help='Number of atoms in simulation.')
parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp or cnn).')
parser.add_argument('--decoder', type=str, default='rnn',
                    help='Type of decoder model (mlp, rnn, or sim).')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument('--suffix', type=str, default='',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--save-folder', type=str, default='logs', ## logs
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')
parser.add_argument('--dims', type=int, default=4,
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--timesteps', type=int, default=100,
                    help='The number of time steps per sample.')
parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--lr-decay', type=int, default=50,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--skip-first', action='store_true', default=True,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--var', type=float, default=5e-3,
                    help='Output variance.')
parser.add_argument('--hard', action='store_true', default=False,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--prior', action='store_true', default=False,
                    help='Whether to use sparsity prior.')
parser.add_argument('--dynamic-graph', action='store_true', default=False,
                    help='Whether test with dynamically re-computed graph.')

##新版ATSDloader需要参数
parser.add_argument('--l-out', action='store_true', default=100,
                    help='Whether test with dynamically re-computed graph.')
parser.add_argument('--device-normalize-method', action='store_true', default='all',
                    help='Whether test with dynamically re-computed graph.')
parser.add_argument('--test_set', type=str, default='test')

args = parser.parse_args()
# class ATSDSegLoader(Dataset):
#     def __init__(self, args, root_path, win_size, step=1, flag="train"):
#         self.T_TS_STANDARD_FORMAT_TRAIN = 'ATSD_standard_format_train.csv'
#         self.T_TS_STANDARD_FORMAT_TRAIN_LABEL = 'ATSD_standard_format_label_train.csv'
#         self.T_TS_STANDARD_FORMAT_TEST = 'ATSD_standard_format_test.csv'
#         self.T_TS_STANDARD_FORMAT_TEST_LABEL = 'ATSD_standard_format_label_test.csv'
#
#         self.flag = flag
#         self.step = step
#         self.win_size = win_size
#         self.scaler = StandardScaler()
#
#         self.train_x = pd.read_csv(os.path.join(root_path, self.T_TS_STANDARD_FORMAT_TRAIN))
#         self.train_y = pd.read_csv(os.path.join(root_path, self.T_TS_STANDARD_FORMAT_TRAIN_LABEL))
#         self.test_x = pd.read_csv(os.path.join(root_path, self.T_TS_STANDARD_FORMAT_TEST))
#         self.test_y = pd.read_csv(os.path.join(root_path, self.T_TS_STANDARD_FORMAT_TEST_LABEL))
#         self.train_x.reset_index(drop=True, inplace=True)
#         self.train_y.reset_index(drop=True, inplace=True)
#         self.test_x.reset_index(drop=True, inplace=True)
#         self.test_y.reset_index(drop=True, inplace=True)
#         train_device = self.train_x.pop('device')
#         test_device = self.test_x.pop('device')
#         self.train_y.pop('device')
#         self.test_y.pop('device')
#         self.train_y = self.train_y.values[:, 1:]
#         self.test_y = self.test_y.values[:, 1:]
#
#         self.train = np.nan_to_num(self.train_x.values[:, 1:])
#         self.scaler.fit(self.train)
#         self.train = self.scaler.transform(self.train)
#
#         self.test = np.nan_to_num(self.test_x.values[:, 1:])
#         # self.scaler.fit(self.test)
#         self.test = self.scaler.transform(self.test)
#
#         device_list = train_device.drop_duplicates().sort_values().reset_index(drop=True)
#         train_valid_split = device_list.iloc[int(len(device_list) * 0.8)]
#         self.val = self.train[train_device > train_valid_split]
#         self.val_y = self.train_y[train_device > train_valid_split]
#         valid_device = train_device[train_device > train_valid_split]
#         valid_device.reset_index(drop=True, inplace=True)
#
#         # 每个样本的index确认
#         self.train_device_range = train_device.groupby(train_device).apply(
#             lambda x: (x.name, min(x.index), max(x.index))).to_list()
#         self.train_device_range = sorted(self.train_device_range, key=lambda x: int(x[0][3:]))
#         self.test_device_range = test_device.groupby(test_device).apply(
#             lambda x: (x.name, min(x.index), max(x.index))).to_list()
#         self.test_device_range = sorted(self.test_device_range, key=lambda x: int(x[0][3:]))
#         self.valid_device_range = valid_device.groupby(valid_device).apply(
#             lambda x: (x.name, min(x.index), max(x.index))).to_list()
#         self.valid_device_range = sorted(self.valid_device_range, key=lambda x: int(x[0][3:]))
#
#         self.train_get_i_to_index = {}
#         walked_get_i = 0
#         for device, index_start, index_end in self.train_device_range:
#             get_i_to_index = {walked_get_i + i: index_i for i, index_i in
#                               enumerate(range(index_start + self.win_size, index_end + 2))}
#             if get_i_to_index:
#                 self.train_get_i_to_index.update(get_i_to_index)
#                 walked_get_i += index_end + 2 - index_start - self.win_size
#         self.test_get_i_to_index = {}
#         walked_get_i = 0
#         for device, index_start, index_end in self.test_device_range:
#             get_i_to_index = {walked_get_i + i: index_i for i, index_i in
#                               enumerate(range(index_start + self.win_size, index_end + 2))}
#             if get_i_to_index:
#                 self.test_get_i_to_index.update(get_i_to_index)
#                 walked_get_i += index_end + 2 - index_start - self.win_size
#         self.valid_get_i_to_index = {}
#         walked_get_i = 0
#         for device, index_start, index_end in self.valid_device_range:
#             get_i_to_index = {walked_get_i + i: index_i for i, index_i in
#                               enumerate(range(index_start + self.win_size, index_end + 2))}
#             if get_i_to_index:
#                 self.valid_get_i_to_index.update(get_i_to_index)
#                 walked_get_i += index_end + 2 - index_start - self.win_size
#
#         print("test:", self.test.shape)
#         print("train:", self.train.shape)
#         print("valid:", self.val.shape)
#
#         self.print_dataset_quality()
#
#     def __len__(self):
#         if self.flag == "train":
#             return sum([(e_i - s_i + 1 - self.win_size) // self.step + 1
#                         for (_, s_i, e_i) in self.train_device_range if s_i + self.win_size <= e_i])
#         elif self.flag == 'val':
#             return sum([(e_i - s_i + 1 - self.win_size) // self.step + 1
#                         for (_, s_i, e_i) in self.valid_device_range if s_i + self.win_size <= e_i])
#         elif self.flag == 'test':
#             return sum([(e_i - s_i + 1 - self.win_size) // self.step + 1
#                         for (_, s_i, e_i) in self.test_device_range if s_i + self.win_size <= e_i])
#         else:
#             return (self.test.shape[0] - self.win_size) // self.win_size + 1
#
#     def __getitem__(self, index):
#         if self.flag == "train":
#             index = self.train_get_i_to_index[index * self.step]
#             return (np.float32(self.train[index - self.win_size: index])
#                         , np.float32(self.train_y[index - self.win_size: index]))
#         elif self.flag == 'val':
#             index = self.valid_get_i_to_index[index * self.step]
#             return (np.float32(self.val[index - self.win_size: index])
#                         , np.float32(self.val_y[index - self.win_size: index]))
#         elif self.flag == 'test':
#             index = self.test_get_i_to_index[index * self.step]
#             return (np.float32(self.test[index - self.win_size: index])
#                         , np.float32(self.test_y[index - self.win_size: index]))
#         else:
#             raise Exception('Not support!')
#
#     def print_dataset_quality(self):
#         print('train dataset quality.', '1:{neg_ratio:.1f}'.format(
#             neg_ratio=(len(self.train_y) - sum(self.train_y[:, 0])) / sum(self.train_y[:, 0])), 'All:',
#               len(self.train_y), 'Pos:', sum(self.train_y[:, 0]), 'Neg:', len(self.train_y) - sum(self.train_y[:, 0]))
#         print('val dataset quality.',
#               '1:{neg_ratio:.1f}'.format(neg_ratio=(len(self.val_y) - sum(self.val_y[:, 0])) / sum(self.val_y[:, 0])),
#               'All:', len(self.val_y), 'Pos:', sum(self.val_y[:, 0]), 'Neg:', len(self.val_y) - sum(self.val_y[:, 0]))
#         print('test dataset quality.', '1:{neg_ratio:.1f}'.format(
#             neg_ratio=(len(self.test_y) - sum(self.test_y[:, 0])) / sum(self.test_y[:, 0])), 'All:', len(self.test_y),
#               'Pos:', sum(self.test_y[:, 0]), 'Neg:', len(self.test_y) - sum(self.test_y[:, 0]))
#         return


class ATSDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.T_TS_STANDARD_FORMAT_TRAIN = 'ATSD_standard_format_train.csv'
        self.T_TS_STANDARD_FORMAT_TRAIN_LABEL = 'ATSD_standard_format_label_train.csv'
        self.T_TS_STANDARD_FORMAT_TEST = 'ATSD_standard_format_test.csv'
        self.T_TS_STANDARD_FORMAT_TEST_LABEL = 'ATSD_standard_format_label_test.csv'

        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.step = step
        self.l_out = args.l_out
        self.device_normalize_method = args.device_normalize_method
        self.win_size = win_size
        self.scaler = StandardScaler()

        if self.flag == "train" or self.flag == "val":
            # 训练数据
            self.load_traing_data()
        elif self.flag == "test":
            # 测试数据
            self.load_test_data()

        # self.print_dataset_quality()

    def load_traing_data(self):

        self.train_x = pd.read_csv(os.path.join(self.root_path, self.T_TS_STANDARD_FORMAT_TRAIN))
        self.train_y = pd.read_csv(os.path.join(self.root_path, self.T_TS_STANDARD_FORMAT_TRAIN_LABEL))
        # 排序
        self.train_x = self.train_x.sort_values(by=['device', 'timestamp_(min)']).reset_index(drop=True)
        self.train_y = self.train_y.sort_values(by=['device', 'timestamp_(min)']).reset_index(drop=True)
        # 保留信息、label生成numpy
        self.train_y_index_info = self.train_y[['device', 'timestamp_(min)']]
        self.train_y.pop('device')
        self.train_y = self.train_y.values[:, 1:]
        # feature归一化
        self.train, train_device = self.dataset_normalization(df=self.train_x,
                                                              device_method=self.device_normalize_method)

        # val划分
        device_list = train_device.drop_duplicates().reset_index(drop=True)
        train_valid_split = device_list.iloc[int(len(device_list) * 0.8)]
        # val
        self.val = self.train[train_device > train_valid_split]
        self.val_y = self.train_y[train_device > train_valid_split]
        valid_device = train_device[train_device > train_valid_split]
        valid_device.reset_index(drop=True, inplace=True)
        # train - val
        if self.args.test_set != 'train+test':
            self.train = self.train[train_device <= train_valid_split]
            self.train_y = self.train_y[train_device <= train_valid_split]
            train_device = train_device[train_device <= train_valid_split]

        # 每个样本的index确认
        self.train_device_range = train_device.groupby(train_device).apply(
            lambda x: (x.name, min(x.index), max(x.index))).to_list()
        self.train_device_range = sorted(self.train_device_range, key=lambda x: int(x[0][3:]))

        self.train_get_i_to_index = {}
        walked_get_i = 0
        for device, index_start, index_end in self.train_device_range:
            get_i_to_index = {walked_get_i + i: index_i for i, index_i in
                              enumerate(range(index_start + self.win_size, index_end + 2))}
            if get_i_to_index:
                self.train_get_i_to_index.update(get_i_to_index)
                walked_get_i += index_end + 2 - index_start - self.win_size

        self.valid_device_range = valid_device.groupby(valid_device).apply(
            lambda x: (x.name, min(x.index), max(x.index))).to_list()
        self.valid_device_range = sorted(self.valid_device_range, key=lambda x: int(x[0][3:]))

        self.valid_get_i_to_index = {}
        walked_get_i = 0
        for device, index_start, index_end in self.valid_device_range:
            get_i_to_index = {walked_get_i + i: index_i for i, index_i in
                              enumerate(range(index_start + self.win_size, index_end + 2))}
            if get_i_to_index:
                self.valid_get_i_to_index.update(get_i_to_index)
                walked_get_i += index_end + 2 - index_start - self.win_size

        print("train:", self.train.shape)
        print("valid:", self.val.shape)

    def load_test_data(self):
        self.test_x = pd.read_csv(os.path.join(self.root_path, self.T_TS_STANDARD_FORMAT_TEST))
        self.test_y = pd.read_csv(os.path.join(self.root_path, self.T_TS_STANDARD_FORMAT_TEST_LABEL))
        # 排序
        self.test_x = self.test_x.sort_values(by=['device', 'timestamp_(min)']).reset_index(drop=True)
        self.test_y = self.test_y.sort_values(by=['device', 'timestamp_(min)']).reset_index(drop=True)
        # 保留信息、label生成numpy
        self.test_y_index_info = self.test_y[['device', 'timestamp_(min)']]
        self.test_y.pop('device')
        self.test_y = self.test_y.values[:, 1:]
        # feature归一化
        self.test, test_device = self.dataset_normalization(df=self.test_x, device_method=self.device_normalize_method)

        # 每个样本的index确认
        self.test_device_range = test_device.groupby(test_device).apply(
            lambda x: (x.name, min(x.index), max(x.index))).to_list()
        self.test_device_range = sorted(self.test_device_range, key=lambda x: int(x[0][3:]))
        self.test_get_i_to_index = {}
        walked_get_i = 0
        for device, index_start, index_end in self.test_device_range:
            get_i_to_index = {walked_get_i + i: index_i for i, index_i in
                              enumerate(range(index_start + self.win_size, index_end + 2))}
            if get_i_to_index:
                self.test_get_i_to_index.update(get_i_to_index)
                walked_get_i += index_end + 2 - index_start - self.win_size

        print("test:", self.test.shape)

    def __len__(self):
        if self.flag == "train":
            return sum([(e_i - s_i + 1 - self.win_size) // self.step + 1
                        for (_, s_i, e_i) in self.train_device_range if s_i + self.win_size <= e_i])
        elif self.flag == 'val':
            return sum([(e_i - s_i + 1 - self.win_size) // self.step + 1
                        for (_, s_i, e_i) in self.valid_device_range if s_i + self.win_size <= e_i])
        elif self.flag == 'test':
            return sum([(e_i - s_i + 1 - self.win_size) // self.step + 1
                        for (_, s_i, e_i) in self.test_device_range if s_i + self.win_size <= e_i])
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        # print("DataLoader", self.flag, index)
        assert self.l_out <= self.win_size, 'l_out should be less than or equal to win_size'
        if self.flag == "train":
            index = self.train_get_i_to_index[index * self.step]
            # print(np.float32(self.train[index - self.win_size: index]).shape
            #       , np.float32(self.train_y[index - self.win_size: index]).shape)
            return (np.float32(self.train[index - self.win_size: index])
                        , np.float32(self.train_y[index - self.l_out: index]))
        elif self.flag == 'val':
            index = self.valid_get_i_to_index[index * self.step]
            # print(np.float32(self.val[index - self.win_size: index]).shape
            #       , np.float32(self.val_y[index - self.win_size: index]).shape)
            return (np.float32(self.val[index - self.win_size: index])
                        , np.float32(self.val_y[index - self.l_out: index]))
        elif self.flag == 'test':
            index = self.test_get_i_to_index[index * self.step]
            # print(np.float32(self.test[index - self.win_size: index]).shape
            #       , np.float32(self.test_y[index - self.win_size: index]).shape)
            return (np.float32(self.test[index - self.win_size: index])
                        , np.float32(self.test_y[index - self.l_out: index]))
        else:
            raise Exception('Not support!')

    def print_dataset_quality(self):
        print('train dataset quality.', len(self.train_y), sum(self.train_y[:, 0]),
              len(self.train_y) - sum(self.train_y[:, 0]),
              '1:{:.2f}'.format((len(self.train_y) - sum(self.train_y[:, 0])) / sum(self.train_y[:, 0])))
        print('val dataset quality.', len(self.val_y), sum(self.val_y[:, 0]), len(self.val_y) - sum(self.val_y[:, 0]),
              '1:{:.2f}'.format((len(self.val_y) - sum(self.val_y[:, 0])) / sum(self.val_y[:, 0])))
        print('test dataset quality.', len(self.test_y), sum(self.test_y[:, 0]),
              len(self.test_y) - sum(self.test_y[:, 0]),
              '1:{:.2f}'.format((len(self.test_y) - sum(self.test_y[:, 0])) / sum(self.test_y[:, 0])))
        return

    def dataset_normalization(self, df, device_method):

        if device_method == 'all':
            # 统一标准化
            device = df.pop('device')
            df_norm = np.nan_to_num(df.values[:, 1:])
            self.scaler.fit(df_norm)
            df_norm = self.scaler.transform(df_norm)

        elif device_method == 'by_device':
            # 分device归一化
            df_norm = df.ffill().bfill().groupby('device').transform(
                lambda x: StandardScaler().fit_transform(x.values[:, np.newaxis]).ravel())
            device = df.pop('device')
            df_norm = df_norm.values[:, 1:]
        else:
            raise Exception('Not support!')
        return df_norm, device

    def get_labels_with_index_info(self, flag='train+test'):
        reserve_flag = self.flag
        labels_with_info = []
        if 'train' in flag:
            self.flag = 'train'
            for index in range(len(self)):
                index = self.train_get_i_to_index[index * self.step]
                labels_with_info.append(
                    self.train_y_index_info.iloc[index - 1].to_list() + list(self.train_y[index - 1]))
            cols = list(self.train_y_index_info.columns) + ['label']
        if 'test' in flag:
            self.flag = 'test'
            for index in range(len(self)):
                index = self.test_get_i_to_index[index * self.step]
                labels_with_info.append(self.test_y_index_info.iloc[index - 1].to_list() + list(self.test_y[index - 1]))
            cols = list(self.test_y_index_info.columns) + ['label']
        labels_with_info = pd.DataFrame(labels_with_info, columns=cols)
        self.flag = reserve_flag
        return labels_with_info

train_data = ATSDSegLoader(args, root_path='/disk1/timeseries_dataset/ATSD_v4_cleaned', win_size=100, step=1, flag='train')
train_loader = DataLoader(train_data, batch_size=100, shuffle=False, drop_last=False)
valid_data = ATSDSegLoader(args, root_path='/disk1/timeseries_dataset/ATSD_v4_cleaned', win_size=100, step=1, flag='val')
valid_loader = DataLoader(valid_data, batch_size=100, shuffle=False, drop_last=False)
test_data = ATSDSegLoader(args, root_path='/disk1/timeseries_dataset/ATSD_v4_cleaned', win_size=100, step=1, flag='test')
test_loader = DataLoader(test_data, batch_size=100, shuffle=False, drop_last=False)


# def create_graph_data(features, labels):
#     """创建单个图数据结构"""
#     # 计算相关性矩阵
#     num_features = features.shape[0]
#     correlation_matrix = np.zeros((num_features, num_features))
#
#     for i in range(num_features):
#         for j in range(i + 1, num_features):
#             corr, _ = pearsonr(features[i], features[j])
#             correlation_matrix[i, j] = corr
#             correlation_matrix[j, i] = corr
#         correlation_matrix[i, i] = 1.0
#
#     # 构建边
#     edge_index = []
#     edge_attr = []
#
#     for i in range(num_features):
#         for j in range(i + 1, num_features):
#             if abs(correlation_matrix[i, j]) > 0.5:
#                 edge_index.extend([[i, j], [j, i]])
#                 edge_attr.extend([correlation_matrix[i, j], correlation_matrix[i, j]])
#
#     # 转换为PyTorch张量
#     edge_index = torch.tensor(edge_index, dtype=torch.long).t()
#     edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
#     x = torch.FloatTensor(features)  # (feature_dim, win_size)
#     y = torch.FloatTensor(labels)
#
#     return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def create_batch_graph_data(batch_features, batch_labels):
    """批量创建图数据结构"""
    batch_graphs = []

    # 使用numpy的vectorization计算相关性矩阵
    for features, label in zip(batch_features, batch_labels):
        features = features.T
        # print(features.shape)
        # print(label.shape)
        num_features = features.shape[0]
        # 使用numpy的corrcoef加速计算相关性矩阵
        correlation_matrix = np.corrcoef(features)
        # print(correlation_matrix.shape)

        # 构建边
        edge_index = []
        edge_attr = []

        # 使用numpy的布尔索引加速边的构建
        indices = np.where(np.abs(correlation_matrix) > 0.5)
        for i, j in zip(*indices):
            if i < j:  # 只处理上三角矩阵
                edge_index.extend([[i, j], [j, i]])
                edge_attr.extend([correlation_matrix[i, j], correlation_matrix[i, j]])

        # 转换为PyTorch张量
        if len(edge_index) > 0:  # 确保有边存在
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
            x = torch.FloatTensor(features)
            y = torch.FloatTensor([label])

            batch_graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

    return batch_graphs


# train_graphs = []
# for batch_idx, (train_dataset, labels) in enumerate(train_loader):
#     if batch_idx % 100 == 0:
#         print(batch_idx)
#
#     # 处理训练集
#     print("Processing training data...")
#     for i in range(len(train_dataset)):
#         features = train_dataset[i]
#         # print(labels.shape)
#         labels_cur = labels[i, -1]
#         features = features.T  # (win_size, feature_dim) -> (feature_dim, win_size)
#         graph = create_graph_data(features, labels_cur)
#         train_graphs.append(graph)

# 主处理循环
train_graphs = []
for batch_idx, (batch_dataset, batch_labels) in enumerate(train_loader):
    if batch_idx % 10 == 0:
        print(f"Processing batch {batch_idx}")

    # 批量处理数据
    batch_graphs = create_batch_graph_data(batch_dataset, batch_labels[:, -1])
    train_graphs.extend(batch_graphs)

save_path = "/disk1/xujing.zbl/CAL/CAL-main/data/ATSD_v4_train_dataset_cor0.5.pt"
torch.save(train_graphs, save_path)
print("save at:{}".format(save_path))


valid_graphs = []
for batch_idx, (batch_dataset, batch_labels) in enumerate(valid_loader):
    if batch_idx % 10 == 0:
        print(f"Processing batch {batch_idx}")

    # 批量处理数据
    batch_graphs = create_batch_graph_data(batch_dataset, batch_labels[:, -1])
    valid_graphs.extend(batch_graphs)

save_path = "/disk1/xujing.zbl/CAL/CAL-main/data/ATSD_v4_val_dataset_cor0.5.pt"
torch.save(valid_graphs, save_path)
print("save at:{}".format(save_path))


test_graphs = []
for batch_idx, (batch_dataset, batch_labels) in enumerate(test_loader):
    if batch_idx % 10 == 0:
        print(f"Processing batch {batch_idx}")

    # 批量处理数据
    batch_graphs = create_batch_graph_data(batch_dataset, batch_labels[:, -1])
    test_graphs.extend(batch_graphs)

save_path = "/disk1/xujing.zbl/CAL/CAL-main/data/ATSD_v4_test_dataset_cor0.5.pt"
torch.save(test_graphs, save_path)
print("save at:{}".format(save_path))






