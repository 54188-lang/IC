import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from copy import copy
from typing import List, Sequence, Tuple, Any
from pathlib import Path
import librosa
import time  # 新增：用于打印耗时


class CustomDataset(Dataset):
    """
    支持多数据集的自定义数据集类，优化数据加载效率，避免初始化卡住
    """
    def __init__(self, stage: str, config):
        self.config = config
        self.stage = stage
        self.data_channel_dim = None
        self.num_sources = None
        dataset_path = config.dataset_path[config.dataset_name]

        # 记录初始化开始时间
        start_time = time.time()
        print(f"开始初始化 {config.dataset_name} ({stage}) 数据集...")

        if config.dataset_name == 'chime3_new':
            self._init_chime3_new(dataset_path, stage)
        elif config.dataset_name == 'DNS':
            self._init_dns(dataset_path, stage)
        elif config.dataset_name == 'CSV':
            self._init_csv(dataset_path, stage)
        else:
            raise ValueError(f"不支持的数据集: {config.dataset_name}")

        # 验证通道数匹配
        if self.data_channel_dim != config.model_mic_num:
            raise RuntimeError(
                f"数据集通道数 {self.data_channel_dim} 与模型配置的麦克风数 {config.model_mic_num} 不匹配！"
            )

        # 打印初始化耗时
        print(f"{config.dataset_name} ({stage}) 数据集初始化完成，耗时 {time.time() - start_time:.2f} 秒，共 {len(self)} 个样本")

    def _init_chime3_new(self, dataset_path, stage):
        self.data_channel_dim = 6
        self.num_sources = 1
        stage_path = {'train': 'tr05', 'valid': 'dt05', 'test': 'et05'}[stage]

        # 优化文件搜索：减少递归深度，使用更精确的路径
        ref_pattern = str(dataset_path.joinpath(f'{stage_path}*simu_clean/*.CH{self.config.reference_channel_idx + 1}.wav'))
        mix_pattern = str(dataset_path.joinpath(f'{stage_path}*simu/*.wav'))
        
        ref_files = glob.glob(ref_pattern)
        mix_files = glob.glob(mix_pattern)

        # 简化排序逻辑
        ref_files = sorted(ref_files, key=lambda x: Path(x).name)
        mix_files = sorted(mix_files, key=lambda x: Path(x).name)

        # 配对文件（增加边界检查）
        self.all_files = []
        max_files = min(len(ref_files), len(mix_files) // self.data_channel_dim)
        for file_idx in range(max_files):
            mix = [mix_files[file_idx * self.data_channel_dim + ii] for ii in range(self.data_channel_dim)]
            self.all_files.append([mix, ref_files[file_idx]])

    def _init_dns(self, dataset_path, stage):
        self.data_channel_dim = 4
        self.num_sources = 1
        stage_path = {'train': 'train', 'valid': 'valid', 'test': 'test'}[stage]

        # 优化文件搜索
        clean_dir = '__clean_wet' if self.config.dns_clean_wet_use else '__clean'
        ref_pattern = str(dataset_path.joinpath(f'{stage_path}/{clean_dir}/*.wav'))
        mix_pattern = str(dataset_path.joinpath(f'{stage_path}/__noisy/*.wav'))
        
        ref_files = glob.glob(ref_pattern)
        mix_files = glob.glob(mix_pattern)

        # 简化排序
        ref_files = sorted(ref_files, key=lambda x: Path(x).name)
        mix_files = sorted(mix_files, key=lambda x: Path(x).name)

        # 配对文件（取最小长度）
        self.all_files = list(zip(mix_files, ref_files))[:min(len(mix_files), len(ref_files))]

    def _init_csv(self, dataset_path, stage):
        self.data_channel_dim = 2
        self.num_sources = 4
        csv_path = dataset_path[stage]

        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

        # 快速读取CSV（只读取必要列）
        required_columns = ['mix1_wav', 'mix2_wav', 's1_wav', 's2_wav', 's3_wav', 's4_wav']
        self.df = pd.read_csv(csv_path, usecols=required_columns)

        # 检查缺失列
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"CSV缺少列: {missing_cols}")

        # 解析文件路径（关闭逐行验证，改为抽样验证）
        self.all_files = []
        for _, row in self.df.iterrows():
            mix_paths = [row['mix1_wav'], row['mix2_wav']]
            source_paths = [row['s1_wav'], row['s2_wav'], row['s3_wav'], row['s4_wav']]
            self.all_files.append([mix_paths, source_paths])

        # 抽样验证（仅验证前10个样本，平衡效率和正确性）
        sample_size = min(10, len(self.all_files))
        for idx in np.random.choice(len(self.all_files), sample_size, replace=False):
            mix_paths, source_paths = self.all_files[idx]
            for path in mix_paths + source_paths:
                if not Path(path).exists():
                    raise FileNotFoundError(f"文件不存在: {path} (抽样验证失败)")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            return self._getitem_safe(idx)
        except Exception as e:
            raise RuntimeError(f"加载样本 {idx} 失败: {str(e)}")

    def _getitem_safe(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """安全加载数据，增加异常处理"""
        filename = self.all_files[idx]

        if self.config.dataset_name == 'chime3_new':
            mixed_input = []
            for ii in range(self.data_channel_dim):
                audio, _ = librosa.load(filename[0][ii], sr=None, mono=True)
                mixed_input.append(audio)
            mixed_input = np.stack(mixed_input, axis=0)
            reference_audio = np.expand_dims(librosa.load(filename[1], sr=None, mono=True)[0], axis=0)

        elif self.config.dataset_name == 'DNS':
            mixed_input, _ = librosa.load(filename[0], sr=None, mono=True)
            mixed_input = np.expand_dims(mixed_input, axis=0)  # 确保至少1维
            reference_audio, _ = librosa.load(filename[1], sr=None, mono=True)
            reference_audio = reference_audio[self.config.reference_channel_idx:self.config.reference_channel_idx+1]
            reference_audio = np.expand_dims(reference_audio, axis=0)

        elif self.config.dataset_name == 'CSV':
            # 加载2通道混合语音
            mixed_input = []
            for mix_path in filename[0]:
                audio, sr = librosa.load(mix_path, sr=self.config.sample_rate, mono=True)
                mixed_input.append(audio)
            mixed_input = np.stack(mixed_input, axis=0)  # (2, T)

            # 加载4通道音源
            sources = []
            for src_path in filename[1]:
                audio, _ = librosa.load(src_path, sr=self.config.sample_rate, mono=True)
                sources.append(audio)
            reference_audio = np.stack(sources, axis=0)  # (4, T)

            # 截断到相同长度
            min_len = min(mixed_input.shape[1], reference_audio.shape[1])
            mixed_input = mixed_input[:, :min_len]
            reference_audio = reference_audio[:, :min_len]
            # print("data_manager:mixed_input:",mixed_input[0].mean())
            # print("data_manager:reference_audio:",reference_audio[0].mean())
        else:
            raise ValueError(f"不支持的数据集: {self.config.dataset_name}")
        # print("idx", idx, "mix_paths:", filename[0], "→ shape", mixed_input.shape)
        # 转换为张量
        return (
            torch.from_numpy(mixed_input).float(),
            torch.from_numpy(reference_audio).float()
        )

    def __len__(self) -> int:
        return len(self.all_files) - (len(self.all_files) % self.config.batch_size)

    @staticmethod
    def custom_collate(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        """优化的拼接函数，减少冗余计算"""
        mixed_batch, ref_batch = zip(*batch)

        # 计算最大长度
        max_mix_len = max(x.shape[1] for x in mixed_batch)
        max_ref_len = max(x.shape[1] for x in ref_batch)

        # 填充并堆叠
        mixed_padded = [np.pad(x, ((0, 0), (0, max_mix_len - x.shape[1])), mode='constant') for x in mixed_batch]
        ref_padded = [np.pad(x, ((0, 0), (0, max_ref_len - x.shape[1])), mode='constant') for x in ref_batch]

        return (
            torch.from_numpy(np.stack(mixed_padded, axis=0)).float(),
            torch.from_numpy(np.stack(ref_padded, axis=0)).float()
        )

    @classmethod
    def split(cls, dataset, ratio: Sequence[float]) -> Sequence:
        n_split = len(ratio)
        ratio = np.array(ratio)
        mask = (ratio == -1)
        ratio[mask] = 0
        if mask.sum() == 1:
            ratio[mask] = 1 - ratio.sum()

        idx_data = np.cumsum(np.insert(ratio, 0, 0) * len(dataset), dtype=int)
        result = [copy(dataset) for _ in range(n_split)]
        for ii in range(n_split):
            result[ii].all_files = dataset.all_files[idx_data[ii]:idx_data[ii + 1]]
        return result


def get_dataloader(config):
    # 基础参数
    loader_kwargs = dict(
        batch_size=config.batch_size,
        drop_last=True,  # 丢弃最后一个不完整批次，避免后续错误
        num_workers=0,
        pin_memory=True,
        collate_fn=CustomDataset.custom_collate,
    )

    # 初始化数据集
    train_set = CustomDataset(stage='train', config=config)
    valid_set = CustomDataset(stage='valid', config=config)
    test_set = CustomDataset(stage='test', config=config)

    # 创建数据加载器
    train_loader = DataLoader(train_set, shuffle=config.train_shuffle,** loader_kwargs)
    valid_loader = DataLoader(valid_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False,** loader_kwargs)

    print("数据加载器初始化完成")
    return train_loader, valid_loader, test_loader