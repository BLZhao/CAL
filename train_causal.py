import os
import pickle
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import tensor
import numpy as np
from utils import k_fold, num_graphs
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_causal_syn(train_set, val_set, test_set, model_func=None, args=None):

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False)

    if args.feature_dim == -1:
        args.feature_dim = args.max_degree
        
    model = model_func(args.feature_dim, args.num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr, last_epoch=-1)
    # best_val_acc, update_test_acc_co, update_test_acc_c, update_test_acc_o, update_epoch = 0, 0, 0, 0, 0
    best_val_loss, update_test_acc_co, update_test_acc_c, update_test_acc_o, update_epoch = np.inf, 0, 0, 0, 0
    # 添加模型保存路径
    model_save_path = f'checkpoints/model_{args.model}.pt'
    os.makedirs('checkpoints', exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        
        # train_loss, loss_c, loss_o, loss_co, train_acc_o = train_causal_epoch(model, optimizer, train_loader, device, args)
        # val_acc_co, val_acc_c, val_acc_o = eval_acc_causal(model, val_loader, device, args)
        # test_acc_co, test_acc_c, test_acc_o = eval_acc_causal(model, test_loader, device, args)

        train_loss, loss_c, loss_o, loss_co, train_acc_o = train_causal_epoch(model, optimizer, train_loader, device, args)
        val_loss, val_precision, val_recall, val_f1 = eval_acc_causal(model, val_loader, device, args)
        test_loss, test_precision, test_recall, test_f1 = eval_acc_causal(model, test_loader, device, args)

        lr_scheduler.step()
        # if val_acc_o > best_val_acc:
        #     best_val_acc = val_acc_o
        #     update_test_acc_co = test_acc_co
        #     update_test_acc_c = test_acc_c
        #     update_test_acc_o = test_acc_o
        #     update_epoch = epoch
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # update_test_acc_co = test_acc_co
            # update_test_acc_c = test_acc_c
            # update_test_acc_o = test_acc_o
            update_epoch = epoch
            # 保存模型
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
            
        # print("BIAS:[{:.2f}] | Model:[{}] Epoch:[{}/{}] Loss:[{:.4f}={:.4f}+{:.4f}+{:.4f}] Train:[{:.2f}] val:[{:.2f}] Test:[{:.2f}] | Update Test:[co:{:.2f},c:{:.2f},o:{:.2f}] at Epoch:[{}] | lr:{:.6f}"
        #         .format(args.bias,
        #                 args.model,
        #                 epoch,
        #                 args.epochs,
        #                 train_loss,
        #                 loss_c,
        #                 loss_o,
        #                 loss_co,
        #                 train_acc_o * 100,
        #                 val_acc_o * 100,
        #                 test_acc_o * 100,
        #                 update_test_acc_co * 100,
        #                 update_test_acc_c * 100,
        #                 update_test_acc_o * 100,
        #                 update_epoch,
        #                 optimizer.param_groups[0]['lr']))
        print(
            "BIAS:[{:.2f}] | Model:[{}] Epoch:[{}/{}] Loss:[{:.4f}={:.4f}+{:.4f}+{:.4f}] Val_loss:[{:.4f}] Test_loss:[{:.4f}] Train:[{:.2f}] Val_P:[{:.2f}] Val_R:[{:.2f}] Val_F1:[{:.2f}] Test_P:[{:.2f}] Test_R:[{:.2f}] Test_F1:[{:.2f}] | Update Test at Epoch:[{}] | lr:{:.6f}"
            .format(args.bias,
                    args.model,
                    epoch,
                    args.epochs,
                    train_loss,
                    loss_c,
                    loss_o,
                    loss_co,
                    val_loss,
                    test_loss,
                    train_acc_o * 100,
                    val_precision * 100,
                    val_recall * 100,
                    val_f1 * 100,
                    test_precision * 100,
                    test_recall * 100,
                    test_f1 * 100,
                    update_epoch,
                    optimizer.param_groups[0]['lr']))

    # print("syd: BIAS:[{:.2f}] | Val acc:[{:.2f}] Test acc:[co:{:.2f},c:{:.2f},o:{:.2f}] at epoch:[{}]"
    #     .format(args.bias,
    #             val_acc_o * 100,
    #             update_test_acc_co * 100,
    #             update_test_acc_c * 100,
    #             update_test_acc_o * 100,
    #             update_epoch))


def test_causal_syn(test_set, model_func=None, args=None):
    """
    测试函数，加载最佳模型并在测试集上评估
    """
    # 创建数据加载器
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False)

    # 初始化模型
    if args.feature_dim == -1:
        args.feature_dim = args.max_degree
    model = model_func(args.feature_dim, args.num_classes).to(device)

    # 加载最佳模型
    model_save_path = f'checkpoints/model_{args.model}.pt'
    checkpoint = torch.load(model_save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_epoch = checkpoint['epoch']

    # 创建保存attention的目录
    attention_save_dir = f'attention_results/model_{args.model}'
    os.makedirs(attention_save_dir, exist_ok=True)

    # 进入评估模式
    model.eval()
    test_loss = 0
    all_labels = []
    all_preds = []
    attention_results = []
    output_predictions = []
    test_labels = []

    print(f"Loading model from epoch {best_epoch}")
    print("Starting evaluation...")

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            # 前向传播，获取预测结果和attention
            xc_logits, xo_logits, xco_logits, attention_dict = model(data, eval_random=False, save_attention=True)

            # 获取预测标签
            pred = xo_logits.max(1)[1]
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

            # 保存attention信息
            attention_results.append({
                'batch_idx': batch_idx,
                'edge_attention': attention_dict['edge_attention'].numpy(),
                'node_attention': attention_dict['node_attention'].numpy(),
                'edge_index': attention_dict['edge_index'].numpy(),
                'batch': attention_dict['batch'].numpy()
            })

            # 计算评估指标
            output_predictions.append(xo_logits[:, -1])
            test_labels.append(data.y)

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

    # 保存attention结果
    attention_save_path = os.path.join(attention_save_dir, 'attention_results.pkl')
    with open(attention_save_path, 'wb') as f:
        pickle.dump(attention_results, f)

    # 打印结果
    print("\nTest Results:")
    print(f"Precision: {precision[best_index] * 100:.2f}%")
    print(f"Recall: {recall[best_index] * 100:.2f}%")
    print(f"F1 Score: {f_score[best_index] * 100:.2f}%")
    print(f"\nAttention results saved to {attention_save_path}")

    # 保存详细的评估结果
    results = {
        'precision': precision[best_index],
        'recall': recall[best_index],
        'f1': f_score[best_index],
        'predictions': all_preds,
        'true_labels': all_labels,
        'best_epoch': best_epoch
    }

    results_save_path = os.path.join(attention_save_dir, 'test_results.pkl')
    with open(results_save_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Detailed results saved to {results_save_path}")

    return precision[best_index], recall[best_index], f_score[best_index], attention_results


def train_causal_real(dataset=None, model_func=None, args=None):

    train_accs, test_accs, test_accs_c, test_accs_o = [], [], [], []
    random_guess = 1.0 / dataset.num_classes
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, args.folds, args.epoch_select))):

        best_test_acc, best_epoch, best_test_acc_c, best_test_acc_o = 0, 0, 0, 0
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

        model = model_func(dataset.num_features, dataset.num_classes).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(1, args.epochs + 1):

            train_loss, loss_c, loss_o, loss_co, train_acc = train_causal_epoch(model, optimizer, train_loader, device, args)
            test_acc, test_acc_c, test_acc_o = eval_acc_causal(model, test_loader, device, args)
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            test_accs_c.append(test_acc_c)
            test_accs_o.append(test_acc_o)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                best_test_acc_c = test_acc_c
                best_test_acc_o = test_acc_o
            
            print("Causal | dataset:[{}] fold:[{}] | Epoch:[{}/{}] Loss:[{:.4f}={:.4f}+{:.4f}+{:.4f}] Train:[{:.4f}] Test:[{:.2f}] Test_o:[{:.2f}] Test_c:[{:.2f}] (RG:{:.2f}) | Best Test:[{:.2f}] at Epoch:[{}] | Test_o:[{:.2f}] Test_c:[{:.2f}]"
                    .format(args.dataset,
                            fold,
                            epoch, args.epochs,
                            train_loss, loss_c, loss_o, loss_co,
                            train_acc * 100,  
                            test_acc * 100, 
                            test_acc_o * 100,
                            test_acc_c * 100, 
                            random_guess*  100,
                            best_test_acc * 100, 
                            best_epoch,
                            best_test_acc_o * 100,
                            best_test_acc_c * 100))

        print("syd: Causal fold:[{}] | Dataset:[{}] Model:[{}] | Best Test:[{:.2f}] at epoch [{}] | Test_o:[{:.2f}] Test_c:[{:.2f}] (RG:{:.2f})"
                .format(fold,
                        args.dataset,
                        args.model,
                        best_test_acc * 100, 
                        best_epoch,
                        best_test_acc_o * 100,
                        best_test_acc_c * 100,
                        random_guess*  100))
    
    train_acc, test_acc, test_acc_c, test_acc_o = tensor(train_accs), tensor(test_accs), tensor(test_accs_c), tensor(test_accs_o)
    train_acc = train_acc.view(args.folds, args.epochs)
    test_acc = test_acc.view(args.folds, args.epochs)
    test_acc_c = test_acc_c.view(args.folds, args.epochs)
    test_acc_o = test_acc_o.view(args.folds, args.epochs)
    
    _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
    selected_epoch = selected_epoch.repeat(args.folds)
    
    _, selected_epoch2 = test_acc_o.mean(dim=0).max(dim=0)
    selected_epoch2 = selected_epoch2.repeat(args.folds)

    test_acc = test_acc[torch.arange(args.folds, dtype=torch.long), selected_epoch]
    test_acc_c = test_acc_c[torch.arange(args.folds, dtype=torch.long), selected_epoch]
    test_acc_o = test_acc_o[torch.arange(args.folds, dtype=torch.long), selected_epoch2]

    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    test_acc_c_mean = test_acc_c.mean().item()
    test_acc_c_std = test_acc_c.std().item()
    test_acc_o_mean = test_acc_o.mean().item()
    test_acc_o_std = test_acc_o.std().item()

    print("=" * 150)
    print('sydall Final: Causal | Dataset:[{}] Model:[{}] seed:[{}]| Test Acc: {:.2f}±{:.2f} | OTest: {:.2f}±{:.2f}, CTest: {:.2f}±{:.2f} (RG:{:.2f}) | [Settings] co:{},c:{},o:{},harf:{},dim:{},fc:{}'
         .format(args.dataset,
                 args.model,
                 args.seed,
                 test_acc_mean * 100, 
                 test_acc_std * 100,
                 test_acc_o_mean * 100, 
                 test_acc_o_std * 100,
                 test_acc_c_mean * 100, 
                 test_acc_c_std * 100,
                 random_guess*  100,
                 args.co,
                 args.c,
                 args.o,
                 args.harf_hidden,
                 args.hidden,
                 args.fc_num))
    print("=" * 150)

def train_causal_epoch(model, optimizer, loader, device, args):
    
    model.train()
    total_loss = 0
    total_loss_c = 0
    total_loss_o = 0
    total_loss_co = 0
    correct_o = 0
    
    for it, data in enumerate(loader):
        
        optimizer.zero_grad()
        data = data.to(device)
        
        one_hot_target = data.y.view(-1).long()
        c_logs, o_logs, co_logs = model(data, eval_random=args.with_random)
        uniform_target = torch.ones_like(c_logs, dtype=torch.float).to(device) / model.num_classes
        # noncasual_target = torch.zeros_like(one_hot_target).to(device)
        
        c_loss = F.kl_div(c_logs, uniform_target, reduction='batchmean')
        # c_loss = F.nll_loss(c_logs, noncasual_target)
        o_loss = F.nll_loss(o_logs, one_hot_target)
        co_loss = F.nll_loss(co_logs, one_hot_target)
        loss = args.c * c_loss + args.o * o_loss + args.co * co_loss
        # loss = args.c * c_loss + args.o * o_loss
        # loss = o_loss

        pred_o = o_logs.max(1)[1]
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        total_loss_c += c_loss.item() * num_graphs(data)
        total_loss_o += o_loss.item() * num_graphs(data)
        total_loss_co += co_loss.item() * num_graphs(data)
        optimizer.step()
    
    num = len(loader.dataset)
    total_loss = total_loss / num
    total_loss_c = total_loss_c / num
    total_loss_o = total_loss_o / num
    total_loss_co = total_loss_co / num
    correct_o = correct_o / num
    return total_loss, total_loss_c, total_loss_o, total_loss_co, correct_o

# def eval_acc_causal(model, loader, device, args):
#
#     model.eval()
#     eval_random = args.eval_random
#     correct = 0
#     correct_c = 0
#     correct_o = 0
#
#     for data in loader:
#         data = data.to(device)
#         with torch.no_grad():
#             c_logs, o_logs, co_logs = model(data, eval_random=eval_random)
#             pred = co_logs.max(1)[1]
#             pred_c = c_logs.max(1)[1]
#             pred_o = o_logs.max(1)[1]
#         correct += pred.eq(data.y.view(-1)).sum().item()
#         correct_c += pred_c.eq(data.y.view(-1)).sum().item()
#         correct_o += pred_o.eq(data.y.view(-1)).sum().item()
#
#     acc_co = correct / len(loader.dataset)
#     acc_c = correct_c / len(loader.dataset)
#     acc_o = correct_o / len(loader.dataset)
#     return acc_co, acc_c, acc_o


def eval_acc_causal(model, loader, device, args):
    model.eval()
    eval_random = args.eval_random
    correct = 0
    correct_c = 0
    correct_o = 0

    total_loss = 0
    total_loss_c = 0
    total_loss_o = 0
    total_loss_co = 0
    output_predictions = []
    test_labels = []

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            c_logs, o_logs, co_logs = model(data, eval_random=eval_random)
            one_hot_target = data.y.view(-1).long()
            uniform_target = torch.ones_like(c_logs, dtype=torch.float).to(device) / model.num_classes
            # noncasual_target = torch.zeros_like(one_hot_target).to(device)
            c_loss = F.kl_div(c_logs, uniform_target, reduction='batchmean')
            # c_loss = F.nll_loss(c_logs, noncasual_target)
            o_loss = F.nll_loss(o_logs, one_hot_target)
            co_loss = F.nll_loss(co_logs, one_hot_target)
            loss = args.c * c_loss + args.o * o_loss + args.co * co_loss
            # loss = args.c * c_loss + args.o * o_loss
            # loss = o_loss

            # pred_o = o_logs.max(1)[1]

            output_predictions.append(o_logs[:, -1])
            test_labels.append(data.y)

            total_loss += loss.item() * num_graphs(data)
            total_loss_c += c_loss.item() * num_graphs(data)
            total_loss_o += o_loss.item() * num_graphs(data)
            total_loss_co += co_loss.item() * num_graphs(data)

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

    return total_loss / len(loader.dataset), precision[best_index], recall[best_index], f_score[best_index]

    # acc_co = correct / len(loader.dataset)
    # acc_c = correct_c / len(loader.dataset)
    # acc_o = correct_o / len(loader.dataset)
    # return acc_co, acc_c, acc_o
