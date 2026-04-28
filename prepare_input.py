# prepare_input.py
import numpy as np
import torch
import torchaudio
from pathlib import Path

# 直接複製 dataset.py 的前處理邏輯
from train import MelSpectrogram   # 或直接從 dataset.py import
from train import SpeechCommandDataset, SpeechDataset

mel_transform = MelSpectrogram(sample_rate=16000)
dataset_manager = SpeechCommandDataset(data_dir='./archive')
_, _, test_files = dataset_manager.get_file_list()

dataset = SpeechDataset(test_files, mel_transform, augment=False)

# 取第 0 筆
mel, label = dataset[0]          # mel shape: (1, 101, 40)
x = mel.unsqueeze(0)             # → (1, 1, 101, 40)

np.save('input_sample.npy', x.numpy().astype(np.float32))
print(f"Saved input_sample.npy  shape={x.shape}  label={label}")