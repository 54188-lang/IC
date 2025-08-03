import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm

import data_manager
from config import config
from utils import print_to_file

# 从sdr.py导入SI-SNR计算函数
from utility.sdr import calc_si_sdr, batch_si_sdr

import matplotlib.pyplot as plt
import train
import audio_utils

def run_test(runner, dataloader):
    # 只保留SI-SNR相关向量
    SI_SNR_vector = torch.empty(0).to(runner.out_device)  # 使用out_device更合适

    pbar = tqdm(dataloader, desc='test', postfix='-', dynamic_ncols=True)
    for i_batch, (x_, y_) in enumerate(pbar):
        # 修正数据加载方式，与train.py保持一致
        x, y = x_.to(runner.in_device), y_.to(runner.out_device)

        # forward path
        with torch.no_grad():
            out = runner.model(x).detach()

        # 根据数据形状选择合适的SI-SNR计算函数
        if out.dim() == 3 and y.dim() == 3:
            # 多音源情况 (B, S, T)，使用batch_si_sdr
            si_snr_value = batch_si_sdr(out, y)
        else:
            # 单音源情况 (B, T)，调整为(B, 1, T)再计算
            if out.dim() == 2:
                out = out.unsqueeze(1)  # 变为(B, 1, T)
            if y.dim() == 2:
                y = y.unsqueeze(1)      # 变为(B, 1, T)
            si_snr_value = batch_si_sdr(out, y)
        
        SI_SNR_vector = torch.cat((SI_SNR_vector, si_snr_value))
        pbar.set_description(f'Test Batch {i_batch}: Avg SI-SNR: {torch.mean(SI_SNR_vector):.4f} dB')

    return SI_SNR_vector

def draw_histogram(vector, bins):
    fig = plt.figure(figsize=(12, 6))
    plt.hist(vector, bins=bins, alpha=0.75)
    plt.title('SI-SNR Histogram')
    plt.xlabel('SI-SNR (dB)')
    plt.ylabel('Count')
    return fig


def main():
    # 添加命令行参数解析，用于指定模型路径
    parser = ArgumentParser(description='Test the trained model')
    parser.add_argument('--model_path', type=str, required=True, 
                      help='Path to the trained model checkpoint (e.g., max_100.pt)')
    parser.add_argument('--dataset', type=str, default='test', 
                      choices=['train', 'valid', 'test'], help='Dataset to test on')
    args = parser.parse_args()
    
    # 设置参数
    config.train_shuffle = False
    dataset = args.dataset
    
    # 获取数据加载器
    print("Initializing data loaders...")
    train_loader, valid_loader, test_loader = data_manager.get_dataloader(config)

    # 初始化runner
    runner = train.Runner(config)
    runner.model.eval()  # 设置为评估模式

    # 选择数据加载器
    if dataset == 'train':
        data_loader = train_loader
    elif dataset == 'valid':
        data_loader = valid_loader
    else:
        data_loader = test_loader

    # 加载模型参数
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=runner.in_device)
    
    # 处理DataParallel的情况，与train.py保持一致
    if isinstance(runner.model, nn.DataParallel) and not any(k.startswith('module.') for k in checkpoint.keys()):
        checkpoint = {f"module.{k}": v for k, v in checkpoint.items()}
    runner.model.load_state_dict(checkpoint, strict=False)
    print("Model loaded successfully")

    # # ===== 就地重初始化 source 2 的权重 & 清零 bias =====
    # with torch.no_grad():
    #     if isinstance(runner.model, nn.DataParallel):
    #         mdl = runner.model.module  # 解开 DataParallel
    #     else:
    #         mdl = runner.model
    #     output_layer = mdl.TCN.output[1]
    #     start, end = 2 * 512, 3 * 512
    #     nn.init.xavier_uniform_(output_layer.weight[start:end])
        
    # # ====================================================

    # 计算SI-SNR
    SI_SNR_vector = run_test(runner, data_loader)
    avg_si_snr = torch.mean(SI_SNR_vector)
    print(f'Average SI-SNR = {avg_si_snr:.4f} dB')
    
    # 按SI-SNR降序排序（值越高越好）
    idx = torch.argsort(SI_SNR_vector, descending=True)

    # 设置日志路径
    path_test_result = Path(runner.writer.logdir, f'test_{dataset}_dataset')
    writer = SummaryWriter(logdir=path_test_result)
    os.makedirs(path_test_result, exist_ok=True)

    # 记录测试结果
    print('Writing results...')

    # 绘制SI-SNR的直方图
    fig = draw_histogram(SI_SNR_vector.cpu().numpy(), bins=100)
    writer.add_figure('SI-SNR Histogram', fig, 0)
    
    # 记录平均SI-SNR
    writer.add_scalar('Average SI-SNR', avg_si_snr, 0)
    
    # 将平均结果写入文件
    with open(Path(path_test_result, 'average_si_snr.txt'), 'w') as f:
        f.write(f"Average SI-SNR on {dataset} dataset: {avg_si_snr:.4f} dB\n")

    # 保存不同SI-SNR表现的样本
    idx_l = len(idx)
    sample_count = min(30, idx_l)  # 确保不会超过数据集大小
    
    if sample_count > 0:
        # 保存SI-SNR最好的样本
        for (rank, i) in enumerate(idx[:sample_count]):
            audio_utils.test_writing(runner, data_loader, i, 'test', 0, writer, rank)

        # 保存SI-SNR最差的样本
        for (rank, i) in enumerate(idx[-sample_count:]):
            audio_utils.test_writing(runner, data_loader, i, 'test', 0, writer, rank + idx_l - sample_count)

        # 保存SI-SNR中等的样本
        mid_start = max(0, idx_l // 2 - sample_count // 2)
        mid_end = min(idx_l, mid_start + sample_count)
        for (rank, i) in enumerate(idx[mid_start:mid_end]):
            audio_utils.test_writing(runner, data_loader, i, 'test', 0, writer, rank + mid_start)

    writer.close()
    print('Testing completed successfully')


if __name__ == '__main__':
    # 设置随机种子，与train.py保持一致
    random_seed = 365
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    import numpy as np
    np.random.seed(random_seed)
    import random
    random.seed(random_seed)
    
    main()
