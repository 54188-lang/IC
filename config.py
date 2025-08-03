from pathlib import Path
from typing import Dict, Sequence, Union
from dataclasses import dataclass, field, asdict

@dataclass
class Configurations(object):
    # Model hyperparameters
    model_mic_num: int = 2
    model_ch_dim: int = 8
    model_enc_dim: int = 512
    model_feature_dim: int = 128
    model_win: int = 16
    model_layer: int = 8
    model_stack: int = 1
    model_kernel: int = 3
    model_num_spk: int = 4
    model_causal: bool = False

    # 新增三个CSV数据集路径配置
    csv_train_path: Path = Path('/mnt/g2/fuyuxiang/workspace/CCF2/IC_Conv-TasNet-main/baseline_train_data.csv')    # 训练集CSV路径
    # csv_train_path: Path = Path('/mnt/g2/fuyuxiang/workspace/CCF2/IC_Conv-TasNet-main/baseline_valid_data.csv')   
    csv_valid_path: Path = Path('/mnt/g2/fuyuxiang/workspace/CCF2/IC_Conv-TasNet-main/baseline_valid_data.csv')    # 验证集CSV路径
    csv_test_path: Path = Path('/mnt/g2/fuyuxiang/workspace/CCF2/IC_Conv-TasNet-main/development_set.csv')      # 测试集CSV路径

    # Dataset settings
    dataset_path: Dict[str, Path] = field(init=False)
    supported_dataset_names = ['chime2', 'chime3', 'chime3_new', 'FSD', 'DNS', 'CSV']
    reference_channel_idx: int = 0
    dns_clean_wet_use: bool = False

    # Feature configs
    sample_rate: int = 16000

    # FFT configs for plotting
    fft_size: int = 1024
    win_size: int = 512     # 512 -> 32ms
    hop_size: int = 64      # 256 -> 16ms

    # Log directory
    logdir: str = '/mnt/g2/fuyuxiang/workspace/CCF2/IC_Conv-TasNet-main/logs_2'

    # Training configs
    dataset_name: str = 'CSV'
    batch_size: int = 72
    train_shuffle: bool = True
    num_epochs: int = 180
    learning_rate: float = 1e-4

    # Device configs
    """
    'cpu', 'cuda:n', the cuda device #, or the tuple of the cuda device #.
    """
    device: Union[int, str, Sequence[str], Sequence[int]] = (0,1,2,3,4)
    out_device: Union[int, str] = 0
    num_workers: int = 0            # should be 0 in Windows

    def __post_init__(self):
        # Dataset path settings
        self.dataset_path = dict(chime3_new=Path('./chime3_small/data/audio/16kHz/isolated'),
                                DNS=Path('./DNS_processed'),
                                CSV={'train': self.csv_train_path, 'valid': self.csv_valid_path, 'test': self.csv_test_path}
                            )

    def print_params(self):
        print('-------------------------')
        print('Hyper Parameter Settings')
        print('-------------------------')
        for k, v in asdict(self).items():
            print(f'{k}: {v}')
        print('-------------------------')

config = Configurations()
