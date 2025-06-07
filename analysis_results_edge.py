import os
import heapq
import torch
import pickle
import opts
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from torch_geometric.data import DataLoader

with open('attention_results/model_CausalGCN/attention_results.pkl', 'rb') as f:  # 'rb'表示以二进制读取模式打开文件
    load_data = pickle.load(f)

with open('attention_results/model_CausalGCN/test_results.pkl', 'rb') as f:
    results = pickle.load(f)

args = opts.parse_args()
save_path = "data"
test_set = torch.load(save_path + "/ATSD_v0_test_dataset_cor0.5.pt", weights_only=False)
test_loader = DataLoader(test_set, args.batch_size, shuffle=False)

preds = results['predictions']
true_labels = results['true_labels']
print('true_labels:', len(true_labels))
print('test results:', results)

standard_dataset_metrics = ['nc_dpdk_cpu0', 'nc_dpdk_cpu_avg', 'nc_mem_available', 'nc_mem_kernel_stack', 'nc_mem_page_tables', 'nc_mem_shmem', 'pcpu_net_idle_min', 'vcpu_llc', 'vcpu_sla', 'vm.apic_ipi', 'vm.apic_write', 'vm.arp_dropped_pps', 'vm.baseline', 'vm.cache_miss', 'vm.context_switch', 'vm.cpu_his_nr_err_major', 'vm.cpu_his_nr_pf', 'vm.cpu_his_total_cost', 'vm.cpu_usage', 'vm.cpu_usage.avg', 'vm.disk_io_count_r', 'vm.disk_io_count_w', 'vm.disk_slow_io_count_r', 'vm.disk_slow_io_count_w', 'vm.ept_misconfig', 'vm.ept_violation', 'vm.external_interrupt.avg', 'vm.fb_session_count_val', 'vm.freq.avg', 'vm.hlt', 'vm.if_ibps_drop_pps', 'vm.if_idrop_pps', 'vm.if_ipps_drop_pps', 'vm.if_obps_drop_pps', 'vm.if_ocn_nobuf_err_packets', 'vm.if_odrop_pps', 'vm.if_ofpga_nobuf_packets', 'vm.if_opps_drop_pps', 'vm.io_bytes_r', 'vm.io_bytes_w', 'vm.io_count_r', 'vm.io_count_w', 'vm.io_his_nr_err_major', 'vm.io_his_nr_pf', 'vm.io_his_total_cost', 'vm.io_latency_r', 'vm.io_latency_w', 'vm.ipc', 'vm.llc.sum', 'vm.llc_hit_ratio', 'vm.llc_level', 'vm.llc_occ', 'vm.load_15m_max', 'vm.load_5m_max', 'vm.mem_bw_avg', 'vm.msr_read.avg', 'vm.msr_write.avg', 'vm.netcpu_credit', 'vm.netcpu_usage', 'vm.packet_in_drop_rates_max', 'vm.packet_out_drop_rates_max', 'vm.piw.max', 'vm.psi', 'vm.rtt', 'vm.rx_bps', 'vm.rx_bps_limit_error', 'vm.rx_cpu_limit_error', 'vm.rx_dropped_pps', 'vm.rx_dropped_rate', 'vm.rx_igw_bps', 'vm.rx_igw_pps', 'vm.rx_igw_retrans_pps', 'vm.rx_igw_retrans_rate', 'vm.rx_pps', 'vm.rx_pps_limit_error', 'vm.rx_retrans_pps', 'vm.rx_retrans_rate', 'vm.session_count', 'vm.set_tscdeadline', 'vm.slb_session_count_val', 'vm.split_lock', 'vm.steal', 'vm.steal.max', 'vm.steal_max_virt', 'vm.steal_virt', 'vm.steal_virt.max', 'vm.total_credit', 'vm.tx_bps', 'vm.tx_bps_limit_error', 'vm.tx_cpu_limit_error', 'vm.tx_dropped_pps', 'vm.tx_dropped_rate', 'vm.tx_igw_bps', 'vm.tx_igw_pps', 'vm.tx_igw_retrans_pps', 'vm.tx_igw_retrans_rate', 'vm.tx_pps', 'vm.tx_pps_limit_error', 'vm.tx_retrans_pps', 'vm.tx_retrans_rate']


indices_1 = [i for i in range(len(preds)) if preds[i] == 1 and true_labels[i] == 1]
indices_0 = [i for i in range(len(preds)) if preds[i] == 0 and true_labels[i] == 0]
indices_1_0 = [i for i in range(len(preds)) if preds[i] == 0 and true_labels[i] == 1]
indices_0_1 = [i for i in range(len(preds)) if preds[i] == 1 and true_labels[i] == 0]
true_labels_1 = [i for i in range(len(true_labels)) if true_labels[i] == 1]
print(len(indices_0))
print(len(indices_1))
print(len(preds))
print(indices_1)
print(indices_1_0)

result = []
for j in range(len(load_data)):
    array_dict = load_data[j]
    edge_attention = array_dict['edge_attention']  # shape: (10000, 2)
    batch_info = array_dict['batch']  # shape: (10000,)
    for i in range(max(batch_info)+1):  # 100个batch
        mask = (batch_info == i)
        batch_attention = node_attention[mask]
        result.append(batch_attention)



reshaped_attention = np.stack(result)  # shape: (2331, 100, 2)
# print(reshaped_attention[186])
reshaped_attention = reshaped_attention[:, :, 1].squeeze()
indices_max5 = np.argsort(reshaped_attention, axis=1)[:, -5:][:, ::-1]
print('shape of indices_max5:', indices_max5.shape)

# 将所有异常数据按时间连续性分段，认为连续时间的异常对应相同的异常根因
indices_array = np.array(true_labels_1)
# 计算相邻元素之间的差值
diff_values = np.diff(indices_array)
# 找到差值大于10的位置
split_index = np.where(diff_values > 10)[0]
# 将索引加1，因为我们要在差值大于10的位置后面进行分割
split_index += 1
# 使用np.split进行分割
split_segments = np.split(indices_array, split_index)

for i, segment in enumerate(split_segments):
    print(f"Segment {i} shape:", segment.shape)
    print(segment)

def find_segment(value, split_segments):
    # 遍历所有分段
    for i, segment in enumerate(split_segments):
        # 检查值是否在当前分段中
        if value in segment:
            return i
    return -1  # 如果没有找到，返回-1

# 对index列表中的每个值进行分类
segment_indices_1 = []
for idx in indices_1:
    segment_idx = find_segment(idx, split_segments)
    if segment_idx != -1:
        segment_indices_1.append(segment_idx)
# print(segment_indices_1)

def find_segments_bounds(result):
    if not result:  # 如果result为空
        return []
    segments = []
    start = 0
    # 遍历到倒数第二个元素
    for i in range(len(result) - 1):
        if result[i] != result[i + 1]:  # 如果当前元素和下一个元素不相等
            segments.append((start, i))  # 添加当前段的起止位置
            start = i + 1
    # 添加最后一段
    segments.append((start, len(result) - 1))
    return segments

# 使用函数
segments_indices_1_bounds = find_segments_bounds(segment_indices_1)

# 打印每一段的起止位置
for i, (start, end) in enumerate(segments_indices_1_bounds):
    print(f"Segment {i}: start={start}, end={end}, count={end-start+1}, value={segment_indices_1[start]}")

# 根据分段信息对indices_max5进行划分
split_indices_max5 = []
for start, end in segments_indices_1_bounds:
    # 取出对应索引的值
    segment_values = [indices_max5[indices_1[i]] for i in range(start, end + 1)]
    split_indices_max5.append(segment_values)


def get_frequency_sorted_values_per_group(split_results):
    result_groups = []

    for group in split_results:
        # 创建一个Counter来统计当前组中所有值的频次
        frequency_counter = Counter()

        # 遍历组内的每个5维数组
        for item in group:
            frequency_counter.update(item)

        # 按照频次降序排序，获取(元素, 频次)的列表
        sorted_items = frequency_counter.most_common()

        result_groups.append(sorted_items)

    return result_groups


# 使用函数
frequency_sorted_groups = get_frequency_sorted_values_per_group(split_indices_max5)

# # 打印结果示例
# for i, group in enumerate(frequency_sorted_groups):
#     print(f"Group {i}:")
#     for value, freq in group:
#         print(f"  值: {value}, 频次: {freq}")


def filter_by_half_max_frequency(frequency_sorted_groups):
    filtered_groups = []

    for group in frequency_sorted_groups:
        if not group:  # 处理空组的情况
            filtered_groups.append([])
            continue

        # 获取最大频次
        max_frequency = group[0][1]  # group[0][1]是第一个元素的频次
        # 频次阈值
        threshold = max_frequency / 2

        # 筛选频次大于阈值的元素
        filtered_items = [item for item in group if item[1] > threshold]
        filtered_groups.append(filtered_items)

    return filtered_groups


# 使用函数
filtered_result = filter_by_half_max_frequency(frequency_sorted_groups)

# 打印结果示例
for i, group in enumerate(filtered_result):
    print(f"Group {i}:")
    for value, freq in group:
        print(f"  值: {value}, 频次: {freq}")


def get_metric_names(filtered_result, standard_dataset_metrics):
    result_groups = []

    for group in filtered_result:
        # 从每组中提取索引值
        indices = [item[0] for item in group]  # item[0]是值（索引），item[1]是频次

        # 根据索引获取对应的指标名称
        metrics = [standard_dataset_metrics[idx] for idx in indices]

        result_groups.append(metrics)

    return result_groups


# 使用函数
metric_names = get_metric_names(filtered_result, standard_dataset_metrics)

# 打印结果示例
for i, group in enumerate(metric_names):
    print(f"Group {i} metrics:")
    for metric in group:
        print(f"  {metric}")


data_test = []
label_test = []
for batch_idx, data in enumerate(test_loader):
    batch_info = data.batch
    node_data = data.x
    label_test.extend(data.y)
    # print(data)
    for i in range(max(batch_info) + 1):  # 100个batch
        mask = (batch_info == i)
        batch_data = node_data[mask]
        data_test.append(batch_data)
data_test = np.stack(data_test)
label_test = np.array(label_test)
print(data_test.shape)
print(label_test.shape)

def concatenate_time_series(data_test, indices):
    # 获取最小和最大索引
    min_idx = indices[0]
    max_idx = indices[-1]

    # 计算需要拼接的总时间长度
    # 因为每个相邻索引有99个重叠点，所以需要加上最后一个100点
    total_time_length = (max_idx - min_idx + 1) + 99

    # 创建结果数组
    # 2000是batch size，100是变量个数
    result = np.zeros((100, total_time_length))

    # 对每个索引位置进行处理
    for i, idx in enumerate(range(min_idx, max_idx + 1)):
        # 计算在结果数组中的起始位置
        start_pos = idx - min_idx

        # 如果是第一个索引，完整复制所有时间点
        if i == 0:
            result[:, start_pos:start_pos + 100] = data_test[idx, :, :]
        # 否则只复制最后一个时间点
        else:
            result[:, start_pos + 99] = data_test[idx, :, 99]

    return result


attention_data_all = []
label_all = []
pred_all = []
attention_group_all = []
for i, (start, end) in enumerate(segments_indices_1_bounds):
    group_index = segment_indices_1[start]
    attention_group_all.append(group_index)
    segment_data = concatenate_time_series(data_test, split_segments[group_index])
    segment_label = label_test[split_segments[group_index][0]-99:split_segments[group_index][-1]+1]
    segment_pred = preds[split_segments[group_index][0]-99:split_segments[group_index][-1]+1]
    print(len(segment_label))
    group = filtered_result[i]
    max_freq_index = []
    for value, freq in group:
        max_freq_index.append(value)
    attention_segment_data = segment_data[max_freq_index]
    attention_data_all.append(attention_segment_data)
    label_all.append(segment_label)
    pred_all.append(segment_pred)
    # print(attention_segment_data.shape)


# 遍历每组数据
for group_idx, data in enumerate(attention_data_all):
    n_vars = data.shape[0]  # 变量个数
    time_points = data.shape[1]  # 时间点数

    # 创建一个图像，包含n_vars个子图
    fig, axs = plt.subplots(n_vars+1, 1, figsize=(10, 2 * n_vars))
    fig.suptitle(f'Segment {attention_group_all[group_idx]} start{split_segments[attention_group_all[group_idx]][0]} end {split_segments[attention_group_all[group_idx]][-1]}')

    # 如果只有一个变量，axs不会是数组，需要转换为数组
    if n_vars == 1:
        axs = [axs]

    # 为每个变量创建时间序列图
    for var_idx in range(n_vars):
        time = np.arange(time_points)
        axs[var_idx].plot(time, data[var_idx])
        axs[var_idx].set_title(f'Metric: {metric_names[group_idx][var_idx]}')
        axs[var_idx].grid(True)
        axs[var_idx].set_xlabel('Time')
        axs[var_idx].set_ylabel('Value')

    time = np.arange(time_points)
    axs[n_vars].plot(time, label_all[group_idx], color='red', label='Target')
    axs[n_vars].plot(time, pred_all[group_idx], color='green', alpha=0.5, label='Pred')
    axs[n_vars].set_title(f'Labels')
    axs[n_vars].grid(True)
    axs[n_vars].set_xlabel('Time')
    axs[n_vars].set_ylabel('Label')
    axs[n_vars].legend()  # 添加图例

    plt.tight_layout()
    plt.show()