# prepare_input.py
import numpy as np
import torch
import torchaudio
from pathlib import Path

# 直接複製 dataset.py 的前處理邏輯
# from training_code.train import MelSpectrogram   # 或直接從 dataset.py import
# from training_code.train import SpeechCommandDataset, SpeechDataset

# mel_transform = MelSpectrogram(sample_rate=16000)
# dataset_manager = SpeechCommandDataset(data_dir='./archive')
# _, _, test_files = dataset_manager.get_file_list()

# dataset = SpeechDataset(test_files, mel_transform, augment=False)

# # 取第 0 筆
# mel, label = dataset[0]          # mel shape: (1, 101, 40)
# x = mel.unsqueeze(0)             # → (1, 1, 101, 40)

# np.save('input_sample.npy', x.numpy().astype(np.float32))
# print(f"Saved input_sample.npy  shape={x.shape}  label={label}")

# prepare_input.py (續)
x = np.load('input_sample.npy', allow_pickle=True).astype(np.float32)

from training_code.model import CNNTradFpool3
from training_code.pruning import prune_cnntrad
from training_code.train_pruned import SpeechCommandDataset, SpeechDataset, MelSpectrogram, CFG
from training_code.ptq import fake_quant_forward   # 直接 import 你的 ptq.py

device = torch.device('cpu')
ckpt = torch.load('./model_params/quantized_params_75.pth',
                  map_location=device, weights_only=False)
act_quant    = ckpt['act_quant']
weight_quant = ckpt['weight_quant']
n_bits       = ckpt['n_bits']

# --- 重建 pruned model（fake_quant_forward 需要 stride/padding 資訊）---
model = CNNTradFpool3(num_classes=CFG['num_classes'],
                      dropout_rate=CFG['dropout_rate'])
model = prune_cnntrad(model,
                      prune_ratio_conv1=CFG['prune_ratio_conv1'],
                      prune_ratio_conv2=CFG['prune_ratio_conv2'])
model.eval()

# --- 推論 ---
with torch.no_grad():
    output = fake_quant_forward(model, act_quant, weight_quant,
                                torch.tensor(x), n_bits)

np.save('output_sample.npy', output.numpy().astype(np.float32))
print(f"Saved output_sample.npy  shape={output.shape}")