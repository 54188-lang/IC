import os
import glob
import librosa
import numpy as np
from pathlib import Path

def check_audio_file(file_path, threshold=1e-5):
    """
    检查音频文件是否有效。
    :param file_path: 音频文件路径
    :param threshold: 静音阈值
    :return: None
    """
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mean_abs_value = np.mean(np.abs(audio))  # 计算平均绝对值
        if mean_abs_value < threshold:
            print(f"Warning: Audio file {file_path} is nearly silent (Mean: {mean_abs_value})")
        else:
            print(f"File: {file_path}, Mean: {mean_abs_value}")
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")

def check_dataset(dataset_path, threshold=1e-5):
    """
    检查整个数据集中的所有音频文件。
    :param dataset_path: 数据集路径
    :param threshold: 静音阈值
    :return: None
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    # 获取所有音频文件路径
    audio_files = list(dataset_path.rglob("*.wav"))

    print(f"Found {len(audio_files)} audio files in the dataset.")
    for file_path in audio_files:
        check_audio_file(file_path, threshold)

if __name__ == "__main__":
    # 数据集路径
    dataset_path = "/mnt/lv3/fuyuxiang/dataset/sb_sep/baseline_train_data_2"  # 替换为你的数据集路径
    threshold = 1e-5  # 静音阈值

    check_dataset(dataset_path, threshold)