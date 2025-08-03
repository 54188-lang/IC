import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
from tensorboardX import SummaryWriter

from config import config  # 导入项目配置
import data_manager
from utility import sdr


def spectrogram(wave, ref=1.0):
    """计算音频的频谱图（替换hparams参数为config）"""
    S = abs(librosa.core.stft(
        wave,
        n_fft=config.fft_size,       # 从config获取FFT大小
        hop_length=config.hop_size,  # 帧移
        win_length=config.win_size,  # 窗口大小
        window='hann'
    ))
    S = S / (S.max() + 1e-3)  # 归一化
    logS = librosa.core.amplitude_to_db(S, ref=ref)
    return logS


def draw_spectrogram(wave, ref=1.0, max_val=None, loss=None):
    """绘制频谱图并返回matplotlib图形对象"""
    logS = spectrogram(wave, ref)
    fig = plt.figure(figsize=(12, 4))
    librosa.display.specshow(
        logS,
        sr=config.sample_rate,      # 采样率（从config获取）
        hop_length=config.hop_size, # 帧移
        x_axis='time',
        y_axis='linear',
        cmap='magma'
    )
    if loss is None:
        plt.title('Power Spectrogram')
    else:
        plt.title(f'Power Spectrogram (Loss = {loss:.2f})')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    if max_val is not None:
        plt.clim(max_val - 80.0, max_val)  # 限制dB范围，增强可视化
    return fig


def torch_load(runner, epoch):
    """加载指定epoch的模型权重（用于测试）"""
    # 加载保存的模型权重（默认加载max.pt，可修改为指定epoch）
    state_dict = torch.load(
        Path(runner.writer.logdir, 'max.pt'),  # 加载最佳模型
        map_location='cpu'
    )
    if isinstance(runner.model, nn.DataParallel):
        runner.model.module.load_state_dict(state_dict)
    else:
        runner.model.load_state_dict(state_dict)
    return 0


def test_writing(runner, dataloader, dat_num, mode: str, epoch, writer, rank=None):
    """在测试阶段将音频和频谱图写入TensorBoard"""
    with torch.no_grad():
        # 获取测试数据（混合语音和目标音源）
        mixture, target = dataloader.dataset.__getitem__(dat_num)
        mixture = mixture.unsqueeze(0).to(runner.in_device)  # 增加批次维度并移至设备
        target = target.unsqueeze(0).to(runner.in_device)
        # target[0][0] = target[0][2]
        # print(f"target source 1 max: {target[0][0].max()}, min: {target[0][0].min()}")
        # print(f"target source 2 max: {target[0][1].max()}, min: {target[0][1].min()}")
        # print(f"target source 3 max: {target[0][2].max()}, min: {target[0][2].min()}")
        # print(f"target source 4 max: {target[0][3].max()}, min: {target[0][3].min()}")
        # 模型分离
        separated = runner.model(mixture).detach()
        print("Output channel 3 (idx=3) max:", separated[0, 2].abs().max().item())
        # print(separated.shape)
        # print(f"Separated source 1 max: {separated[0][0].max()}, min: {separated[0][0].min()}")
        # print(f"Separated source 2 max: {separated[0][1].max()}, min: {separated[0][1].min()}")
        # print(f"Separated source 3 max: {separated[0][2].max()}, min: {separated[0][2].min()}")
        # print(f"Separated source 4 max: {separated[0][3].max()}, min: {separated[0][3].min()}")
        loss = runner.criterion(separated, target)  # 计算损失
        loss = loss.round()
        _, nsource, nsample = separated.shape  # nsource=4（4个音源）

        # 转换为CPU并转为numpy数组（用于保存和可视化）
        mixture = mixture.squeeze(0).cpu()
        mixture = mixture[0]  # 取第一个麦克风的混合语音
        target = target.squeeze().cpu().numpy()
        separated = separated.squeeze().cpu().numpy()

        # 确保数组格式兼容soundfile
        mixture = np.asfortranarray(mixture.numpy())

        # 生成写入TensorBoard的名称
        if rank is not None:
            write_name = f'{rank:05d}_' + mode + f'_{dat_num}_Loss_{loss}'
        else:
            write_name = f'{mode}_{dat_num}_Loss_{loss}'

        # 写入混合语音音频
        writer.add_audio(
            f'{write_name}/mixture',
            mixture,
            epoch,
            sample_rate=config.sample_rate
        )

        # 写入目标音源和分离结果音频
        for i in range(nsource):
            writer.add_audio(
                f'{write_name}/target_sources/source_{i}',
                target[i],
                epoch,
                sample_rate=config.sample_rate
            )
            writer.add_audio(
                f'{write_name}/separated_sources/source_{i}',
                separated[i],
                epoch,
                sample_rate=config.sample_rate
            )

        # 计算混合语音频谱的最大值（用于统一频谱图的dB范围）
        mix_logS = spectrogram(mixture)
        max_db = mix_logS.max()

        # 写入混合语音频谱图
        fig = draw_spectrogram(mixture, max_val=max_db)
        writer.add_figure(f'{write_name}/mixture_stft', fig, epoch)

        # 写入目标音源和分离结果的频谱图
        for i in range(nsource):
            fig = draw_spectrogram(target[i], max_val=max_db)
            writer.add_figure(f'{write_name}/target_stft/source_{i}', fig, epoch)

            fig = draw_spectrogram(separated[i], max_val=max_db, loss=loss)
            writer.add_figure(f'{write_name}/separated_stft/source_{i}', fig, epoch)

        # 保存音频文件到本地（用于后期分析）
        wav_dir = Path(config.logdir) / 'wav_files' / write_name
        wav_dir.mkdir(parents=True, exist_ok=True)  # 创建目录

        # 缩放音频幅度以避免溢出
        amp_mix = np.sqrt((mixture ** 2).sum()) + 1e-3
        amp_tar = [np.sqrt((target[i] **2).sum()) + 1e-3 for i in range(nsource)]
        amp_sep = [np.sqrt((separated[i]** 2).sum()) + 1e-3 for i in range(nsource)]

        scaled_mixture = mixture * (10 / amp_mix)
        scaled_target = [target[i] * (10 / amp_tar[i]) for i in range(nsource)]
        scaled_separated = [separated[i] * (10 / amp_sep[i]) for i in range(nsource)]

        # 保存音频
        sf.write(wav_dir / 'mixture.wav', scaled_mixture, config.sample_rate)
        for i in range(nsource):
            sf.write(wav_dir / f'target_{i}.wav', scaled_target[i], config.sample_rate)
            sf.write(wav_dir / f'separated_{i}.wav', scaled_separated[i], config.sample_rate)


def train_writing(runner, dataloader, dat_num, mode: str, epoch, writer, rank=None):
    """在训练/验证阶段将音频和频谱图写入TensorBoard"""
    with torch.no_grad():
        # 获取指定索引的训练/验证数据
        mixture, target = dataloader.dataset.__getitem__(dat_num)
        mixture = mixture.unsqueeze(0).to(runner.in_device)  # 增加批次维度
        target = target.unsqueeze(0).to(runner.in_device)

        # 模型分离
        separated = runner.model(mixture).detach()
        loss = runner.criterion(separated, target)  # 计算当前批次的损失
        _, nsource, nsample = separated.shape  # nsource=4

        # 转换为CPU并处理格式
        mixture = mixture.squeeze(0).cpu()  # 移除批次维度
        separated = separated.squeeze(0).cpu().numpy()  # 分离结果（4个音源）
        target = target.squeeze(0).cpu().numpy()  # 目标音源（4个）

        # 混合语音取均值（多麦克风输入时）
        mix_mean = mixture.mean(dim=0).numpy()  # 2麦克风输入→单通道混合语音

        # 生成写入名称
        if rank is not None:
            write_name = f'{rank:05d}_{mode}_{dat_num}_Loss_{loss:.2f}'
        else:
            write_name = f'{mode}_{dat_num}_Loss_{loss:.2f}'

        # 写入混合语音到TensorBoard
        writer.add_audio(
            f'{write_name}/mixture',
            mix_mean,
            epoch,
            sample_rate=config.sample_rate
        )

        # 写入目标音源和分离结果（每个音源单独写入）
        for i in range(nsource):
            writer.add_audio(
                f'{write_name}/target_sources/source_{i}',
                target[i],
                epoch,
                sample_rate=config.sample_rate
            )
            writer.add_audio(
                f'{write_name}/separated_sources/source_{i}',
                separated[i],
                epoch,
                sample_rate=config.sample_rate
            )

        # 计算混合语音频谱的最大值（用于统一频谱图范围）
        mix_logS = spectrogram(mix_mean)
        max_db = mix_logS.max()

        # 写入混合语音频谱图
        fig = draw_spectrogram(mix_mean, max_val=max_db)
        writer.add_figure(f'{write_name}/mixture_stft', fig, epoch)

        # 写入每个目标音源和分离结果的频谱图
        for i in range(nsource):
            fig = draw_spectrogram(target[i], max_val=max_db)
            writer.add_figure(f'{write_name}/target_stft/source_{i}', fig, epoch)

            fig = draw_spectrogram(separated[i], max_val=max_db, loss=loss)
            writer.add_figure(f'{write_name}/separated_stft/source_{i}', fig, epoch)


if __name__ == '__main__':
    """测试音频工具函数（单独运行时）"""
    # 初始化数据加载器
    train_loader, valid_loader, test_loader = data_manager.get_dataloader(config)
    
    # 测试频谱图绘制
    wave, _ = train_loader.dataset.__getitem__(0)
    wave = wave.mean(dim=0).numpy()  # 取平均得到单通道音频
    fig = draw_spectrogram(wave)
    plt.show()
    
    # 测试TensorBoard写入（需初始化writer）
    writer = SummaryWriter(logdir='./test_logs')
    # 此处需初始化runner才能完整测试，示例中仅展示框架
    print("音频工具函数测试完成")
    writer.close()