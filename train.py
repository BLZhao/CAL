import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from torch import tensor
import torch_geometric.transforms as T
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import pdb
import random
import numpy as np
from torch.autograd import grad
from torch_geometric.data import Batch
from utils import k_fold, num_graphs
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def process_dataset(dataset):
    
    num_nodes = max_num_nodes = 0
    for data in dataset:
        num_nodes += data.num_nodes
        max_num_nodes = max(data.num_nodes, max_num_nodes)
    num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)
    transform = T.ToDense(num_nodes)
    new_dataset = []
    
    for data in tqdm(dataset):
        data = transform(data)
        add_zeros = num_nodes - data.feat.shape[0]
        if add_zeros:
            dim = data.feat.shape[1]
            data.feat = torch.cat((data.feat, torch.zeros(add_zeros, dim)), dim=0)
        new_dataset.append(data)
    return new_dataset

def train_baseline_syn(train_set, val_set, test_set, model_func=None, args=None):
    
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False)

    if args.feature_dim == -1:
        args.feature_dim = args.max_degree
    model = model_func(args.feature_dim, args.num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr, last_epoch=-1)
    # best_val_acc, update_test_acc, update_train_acc, update_epoch = 0, 0, 0, 0
    best_val_loss, update_test_acc, update_train_acc, update_epoch = np.inf, 0, 0, 0

    model_save_path = f'checkpoints/model_{args.model}.pt'
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        
        train_loss, train_acc = train(model, optimizer, train_loader, device, args)
        val_loss, val_precision, val_recall, val_f1 = eval_acc(model, val_loader, device, args)
        test_loss, test_precision, test_recall, test_f1 = eval_acc(model, test_loader, device, args)
        #
        # train_loss = train(model, optimizer, train_loader, device, args) ## AD
        # val_loss = eval_acc(model, val_loader, device, args) ## AD
        # test_loss = eval_acc(model, test_loader, device, args) ## AD

        lr_scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            update_test_acc = test_f1
            update_train_acc = train_acc
            update_epoch = epoch

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'test_loss': test_loss,
                'val_metrics': {
                    'precision': val_precision,
                    'recall': val_recall,
                    'f1': val_f1
                },
                'test_metrics': {
                    'precision': test_precision,
                    'recall': test_recall,
                    'f1': test_f1
                }
            }, model_save_path)

        print("BIAS:[{:.2f}] | Model:[{}] Epoch:[{}/{}] Loss:[{:.4f}] Train:[{:.2f}] val_pre:[{:.2f}] val_recall:[{:.2f}] val_f1:[{:.2f}] Test_pre:[{:.2f}] Test_recall:[{:.2f}] Test_f1:[{:.2f}] | Best Val:[{:.2f}] Update Test:[{:.2f}] at Epoch:[{}] | lr:{:.6f}"
                .format(args.bias,
                        args.model,
                        epoch,
                        args.epochs,
                        train_loss,
                        train_acc * 100,
                        val_precision * 100,
                        val_recall * 100,
                        val_f1 * 100,
                        test_precision * 100,
                        test_recall * 100,
                        test_f1 * 100,
                        best_val_loss * 100,
                        update_test_acc * 100,
                        update_epoch,
                        optimizer.param_groups[0]['lr']))

    print("syd: BIAS:[{:.2f}] | Best Val acc:[{:.2f}] Test acc:[{:.2f}] at epoch:[{}]"
        .format(args.bias,
                best_val_loss * 100,
                update_test_acc * 100,
                update_epoch))
        # print("Train loss:[{:.2f}] Val_loss:[{:.2f}] Test_loss:[{:.2f}]"
        #     .format(train_loss,
        #             val_loss,
        #             test_loss))

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        x = data.x if data.x is not None else data.feat
        return x.size(0)
        
def train(model, optimizer, loader, device, args):
    
    model.train()
    total_loss = 0
    correct = 0
    
    for it, data in enumerate(loader):
        
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        # loss = F.nll_loss(out, data.y.view(-1).long(), weight = torch.tensor([1.0, 1.0]))
        loss = F.nll_loss(out, data.y.view(-1).long())
        # loss = nn.MSELoss()(out, data.y.view(-1)) ## AD_ATSD
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
        
    return total_loss / len(loader.dataset), correct / len(loader.dataset)
    # return total_loss / len(loader.dataset) ## AD_ATSD


# def eval_acc(model, loader, device, args):
#
#     model.eval()
#     correct = 0
#     for data in loader:
#         data = data.to(device)
#         with torch.no_grad():
#             pred = model(data).max(1)[1]
#         correct += pred.eq(data.y.view(-1)).sum().item()
#     return correct / len(loader.dataset)


def eval_acc(model, loader, device, args):
    model.eval()

    # 存储所有预测值和真实值
    # all_preds = []
    # all_labels = []

    eval_loss = 0
    output_predictions = []
    test_labels = []


    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            pred = out.max(1)[1]

            # loss = nn.MSELoss()(out, data.y.view(-1))
            loss = F.nll_loss(out, data.y.view(-1).long())
            eval_loss += loss.item() * num_graphs(data)
            #
            output_predictions.append(out[:, -1])
            test_labels.append(data.y)

            # 将预测结果和标签转移到CPU并转换为numpy数组
            # all_preds.extend(pred.long().cpu().numpy())
            # all_labels.extend(data.y.view(-1).long().cpu().numpy())

    # 转换为numpy数组
    # all_preds = np.array(all_preds)
    # all_labels = np.array(all_labels)

    # 计算各项指标
    # precision = precision_score(all_labels, all_preds, average='binary')
    # recall = recall_score(all_labels, all_preds, average='binary')
    # f1 = f1_score(all_labels, all_preds, average='binary')

    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    output_predictions = np.concatenate([_.detach().cpu().numpy() for _ in output_predictions], axis=0).reshape(-1)
    gt = test_labels.astype(int)

    precision, recall, thresholds = precision_recall_curve(gt, output_predictions)

    numerator = (1 + 1 ** 2) * precision * recall
    denominator = 1 ** 2 * precision + recall
    f_score = np.where(
        denominator == 0,
        0,
        numerator / np.where(denominator == 0, 1, denominator)
    )

    best_index = np.argmax(f_score)
    #
    # print(dict(f1=f_score[best_index],
    #             precision=precision[best_index],
    #             recall=recall[best_index],
    #             threshold=thresholds[best_index]))

    return eval_loss / len(loader.dataset), precision[best_index], recall[best_index], f_score[best_index]
    # return eval_loss / len(loader.dataset)
