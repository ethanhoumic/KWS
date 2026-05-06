import sys
sys.path.insert(0, './training_code')

import torch
import torch.nn as nn
import numpy as np
from training_code.model import CNNTradFpool3
from training_code.pruning import prune_cnntrad
from training_code.train_pruned import CFG
import os

OUTPUT_DIR = './acc_preact_sigmoid'
INPUT_PATH  = './layer_outputs_sigmoid/linear_q.txt'
WEIGHT_PATH = './layer_outputs_sigmoid/dnn_w.txt'
PTH_PATH   = './model_params/quantized_params_75_sigmoid.pth'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 載入 quantized params ---
device = torch.device('cpu')
ckpt = torch.load(PTH_PATH, map_location=device, weights_only=False)
act_quant    = ckpt['act_quant']
weight_quant = ckpt['weight_quant']
n_bits       = ckpt['n_bits']

# --- 重建模型結構（只需要 stride/padding）---
model = CNNTradFpool3(num_classes=CFG['num_classes'],
                      dropout_rate=CFG['dropout_rate'])
model = prune_cnntrad(model,
                      prune_ratio_conv1=CFG['prune_ratio_conv1'],
                      prune_ratio_conv2=CFG['prune_ratio_conv2'])
model.eval()

def load_int(path: str) -> np.ndarray:
    return np.loadtxt(path, dtype=np.int64)

# --- 輔助函數 ---
def dequant_w(lname):
    q = weight_quant[lname]
    w_fp = q['w_int'].float()
    scale_w = q['scale_w']
    if w_fp.dim() == 4:
        return w_fp * scale_w.view(-1, 1, 1, 1)
    else:
        return w_fp * scale_w.view(-1, 1)

def dequant_b(lname, scale_act_in):
    q = weight_quant[lname]
    if q['b_int'] is None:
        return None
    return q['b_int'].float() * q['scale_w'] * scale_act_in

def quant_deq_act(x, name):
    s  = act_quant[name]['scale']
    zp = act_quant[name]['zero_point']
    qmin, qmax = 0, 2 ** n_bits - 1
    x_q = (x / s + zp).round().clamp(qmin, qmax)
    return (x_q - zp) * s

def save_txt(arr: np.ndarray, path: str):
    np.savetxt(path, arr.flatten().astype(np.int32), fmt='%d')
    np.savetxt(path.replace('.txt', '_shape.txt'), np.array(arr.shape), fmt='%d')
    print(f"  saved {path}  shape={arr.shape}")

# --- Forward，手動在每層 activation 前存值 ---
# x = torch.from_numpy(np.load(INPUT_NPY)).float()
x = torch.from_numpy(load_int(INPUT_PATH))
w = torch.from_numpy(load_int(WEIGHT_PATH))
x = x.reshape(1, -1)
w = w.reshape(model.dnn.weight.shape)
# input quantize
# x = quant_deq_act(x, 'input')

# linear → 存 preact → quantize
acc = nn.functional.linear(x.float(), w.float(), None)
acc = acc.round().int()  
save_txt(acc.detach().numpy(), os.path.join(OUTPUT_DIR, 'dnn_acc_sigmoid.txt'))

# w = dequant_w('conv1')
# b = dequant_b('conv1', act_quant['input']['scale'])
# x = nn.functional.conv2d(x, w, b, stride=model.conv1.stride, padding=model.conv1.padding)
# save_txt(x.detach().numpy(), os.path.join(OUTPUT_DIR, 'conv1_preact.txt'))
# x = nn.functional.relu(x)
# x = quant_deq_act(x, 'relu1')

# # conv2 → 存 preact → relu → quantize
# w = dequant_w('conv2')
# b = dequant_b('conv2', act_quant['relu1']['scale'])
# x = nn.functional.conv2d(x, w, b, stride=model.conv2.stride, padding=model.conv2.padding)
# save_txt(x.detach().numpy(), os.path.join(OUTPUT_DIR, 'conv2_preact.txt'))
# x = nn.functional.relu(x)
# x = quant_deq_act(x, 'relu2')

# # flatten
# x = x.view(x.size(0), -1)

# # linear → 存 preact → quantize
# w = dequant_w('linear')
# b = dequant_b('linear', act_quant['relu2']['scale'])
# x = nn.functional.linear(x, w, b)
# save_txt(x.detach().numpy(), os.path.join(OUTPUT_DIR, 'linear_preact.txt'))
# x = quant_deq_act(x, 'linear')

# # dnn → 存 preact → relu → quantize
# w = dequant_w('dnn')
# b = dequant_b('dnn', act_quant['linear']['scale'])
# x = nn.functional.linear(x, w, b)
# save_txt(x.detach().numpy(), os.path.join(OUTPUT_DIR, 'dnn_preact.txt'))
# x = nn.functional.relu(x)
# x = quant_deq_act(x, 'relu3')

# # classifier → 存 preact
# w = dequant_w('classifier')
# b = dequant_b('classifier', act_quant['relu3']['scale'])
# x = nn.functional.linear(x, w, b)
# save_txt(x.detach().numpy(), os.path.join(OUTPUT_DIR, 'classifier_preact.txt'))

print("\nDone.")