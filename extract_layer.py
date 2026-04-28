"""
extract_layers.py

從 quantized_params.pth 讀取量化參數，
對單筆輸入跑 integer-domain forward，
把每層的量化整數輸出存成 .txt（一個數字一行），
供 C model 比對用。

用法：
    python extract_layers.py \
        --pth  quantized_params.pth \
        --input input_sample.npy   \   # shape: (1, C, H, W)，float32
        --outdir ./layer_outputs
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F


# ==============================================================================
#  Helper：把 float activation 轉成 uint8 整數域
# ==============================================================================

def to_uint8(x_fp: torch.Tensor, scale: float, zp: int, n_bits: int = 8):
    """float tensor → quantized uint8 tensor（整數域，不 dequant）"""
    qmin, qmax = 0, 2 ** n_bits - 1
    x_q = (x_fp / scale + zp).round().clamp(qmin, qmax).to(torch.int32)
    return x_q


def save_txt(arr: np.ndarray, path: str):
    """flatten 後一行一個整數存檔"""
    np.savetxt(path, arr.flatten().astype(np.int32), fmt='%d')
    print(f"  saved {path}  shape={arr.shape}  range=[{arr.min()}, {arr.max()}]")


# ==============================================================================
#  Integer-domain forward（全程 int32 累加，不 dequant）
# ==============================================================================

def integer_forward(pth_path: str, x_fp: torch.Tensor, outdir: str, n_bits: int = 8):
    """
    x_fp : float32 tensor, shape (1, C, H, W)  — 原始 float 輸入
    """
    os.makedirs(outdir, exist_ok=True)

    ckpt      = torch.load(pth_path, map_location='cpu')
    act_quant = ckpt['act_quant']       # dict: name → {scale, zero_point}
    wq        = ckpt['weight_quant']    # dict: lname → {w_int, scale_w, b_int, scale_b}

    def aq(name):
        return act_quant[name]['scale'], act_quant[name]['zero_point']

    # ── 輸入量化 ──────────────────────────────────────────────────────────────
    s_in, zp_in = aq('input')
    x_q = to_uint8(x_fp.float(), s_in, zp_in, n_bits)   # uint8, shape (1,C,H,W)
    save_txt(x_q.numpy(), os.path.join(outdir, 'input_q.txt'))

    # ── 計算每層 requantization multiplier M0, shift n, K ────────────────────
    # M = S_in * S_w / S_out，拆成 M0 * 2^(-n)（定點數）
    # K  = bias_int + ZP_out * (sum of correction terms)
    # 這裡用 Python float 模擬，C side 要用對應的 int 版本

    def requant_params(s_in_layer, scale_w, s_out, total_bits=32):
        """
        回傳 (M0_array, n) per output channel
        M = s_in * s_w / s_out
        M0 = round(M * 2^n)，n 使 M0 落在 [2^(total_bits-1), 2^total_bits)
        """
        M = (s_in_layer * scale_w.numpy()) / s_out   # (OC,)
        # 找最大 n 使 M0 < 2^31
        n = int(np.floor(-np.log2(M.max()))) + total_bits - 2
        n = max(n, 0)
        M0 = np.round(M * (2 ** n)).astype(np.int64)
        return M0, n

    # ── conv1 ─────────────────────────────────────────────────────────────────
    s_in_l, zp_in_l = aq('input')
    s_out_l, zp_out_l = aq('relu1')

    w1   = wq['conv1']['w_int'].float()          # (OC, IC, kH, kW) int8
    b1   = wq['conv1']['b_int']                  # (OC,) int32
    
    save_txt(w1.numpy().astype(np.int8), os.path.join(outdir, 'conv1_w.txt'))

    # conv in integer domain: input - ZP_in, weight already zero-centered (symmetric)
    # x_shifted = (x_q - zp_in_l).float()         # (1, IC, H, W)
    x_shifted = x_q.float()
    
    # conv1: kernel=(67,8), stride=(1,1), padding=0 → output (1,64,35,33)
    acc1 = F.conv2d(x_shifted, w1, None, stride=(1,1), padding=0)
    # 加 bias（int32）
    # if b1 is not None:
    #     acc1 = acc1 + b1.float().view(1, -1, 1, 1)

    save_txt(acc1.numpy().astype(np.int32), os.path.join(outdir, 'conv1_acc.txt'))

    # requantize → uint8
    M0_1, n1 = requant_params(s_in_l, wq['conv1']['scale_w'], s_out_l)
    M0_1_t = torch.tensor(M0_1, dtype=torch.int64).view(1, -1, 1, 1)
    acc1_i  = acc1.to(torch.int64)
    req1    = ((acc1_i * M0_1_t) >> n1) + zp_out_l
    req1    = req1.clamp(0, 255).to(torch.int32)

    # ReLU in integer domain = clamp below ZP
    relu1_q = req1.clamp(min=zp_out_l)
    save_txt(relu1_q.numpy(), os.path.join(outdir, 'relu1_q.txt'))

    # ── conv2 ─────────────────────────────────────────────────────────────────
    s_in_l, zp_in_l = aq('relu1')
    s_out_l, zp_out_l = aq('relu2')

    w2  = wq['conv2']['w_int'].float()
    b2  = wq['conv2']['b_int']

    x_shifted = (relu1_q - zp_in_l).float()

    # conv2: kernel=(10,4), stride=(1,1), padding=0 → output (1,64,26,30)
    acc2 = F.conv2d(x_shifted, w2, None, stride=(1,1), padding=0)
    if b2 is not None:
        acc2 = acc2 + b2.float().view(1, -1, 1, 1)

    save_txt(acc2.numpy().astype(np.int32), os.path.join(outdir, 'conv2_acc.txt'))

    M0_2, n2 = requant_params(s_in_l, wq['conv2']['scale_w'], s_out_l)
    M0_2_t   = torch.tensor(M0_2, dtype=torch.int64).view(1, -1, 1, 1)
    acc2_i   = acc2.to(torch.int64)
    req2     = ((acc2_i * M0_2_t) >> n2) + zp_out_l
    req2     = req2.clamp(0, 255).to(torch.int32)

    relu2_q  = req2.clamp(min=zp_out_l)
    save_txt(relu2_q.numpy(), os.path.join(outdir, 'relu2_q.txt'))

    # ── flatten ───────────────────────────────────────────────────────────────
    x_flat = relu2_q.view(1, -1)   # (1, N)

    # ── linear ────────────────────────────────────────────────────────────────
    s_in_l, zp_in_l = aq('relu2')
    s_out_l, zp_out_l = aq('linear')

    wL  = wq['linear']['w_int'].float()   # (OC, IC)
    bL  = wq['linear']['b_int']

    x_shifted = (x_flat - zp_in_l).float()
    accL = F.linear(x_shifted, wL, None)
    if bL is not None:
        accL = accL + bL.float().view(1, -1)

    save_txt(accL.numpy().astype(np.int32), os.path.join(outdir, 'linear_acc.txt'))

    M0_L, nL = requant_params(s_in_l, wq['linear']['scale_w'], s_out_l)
    M0_L_t   = torch.tensor(M0_L, dtype=torch.int64).view(1, -1)
    accL_i   = accL.to(torch.int64)
    reqL     = ((accL_i * M0_L_t) >> nL) + zp_out_l
    # linear 後無 ReLU，不做 clamp(min=zp)
    linear_q = reqL.clamp(0, 255).to(torch.int32)
    save_txt(linear_q.numpy(), os.path.join(outdir, 'linear_q.txt'))

    # ── dnn ───────────────────────────────────────────────────────────────────
    s_in_l, zp_in_l = aq('linear')
    s_out_l, zp_out_l = aq('relu3')

    wD  = wq['dnn']['w_int'].float()
    bD  = wq['dnn']['b_int']

    x_shifted = (linear_q - zp_in_l).float()
    accD = F.linear(x_shifted, wD, None)
    if bD is not None:
        accD = accD + bD.float().view(1, -1)

    save_txt(accD.numpy().astype(np.int32), os.path.join(outdir, 'dnn_acc.txt'))

    M0_D, nD = requant_params(s_in_l, wq['dnn']['scale_w'], s_out_l)
    M0_D_t   = torch.tensor(M0_D, dtype=torch.int64).view(1, -1)
    accD_i   = accD.to(torch.int64)
    reqD     = ((accD_i * M0_D_t) >> nD) + zp_out_l
    reqD     = reqD.clamp(0, 255).to(torch.int32)

    relu3_q  = reqD.clamp(min=zp_out_l)
    save_txt(relu3_q.numpy(), os.path.join(outdir, 'relu3_q.txt'))

    # ── classifier ────────────────────────────────────────────────────────────
    # 最後一層不做 requant（logits 保持 int32 比較大小即可）
    s_in_l, zp_in_l = aq('relu3')

    wC  = wq['classifier']['w_int'].float()
    bC  = wq['classifier']['b_int']

    x_shifted = (relu3_q - zp_in_l).float()
    accC = F.linear(x_shifted, wC, None)
    if bC is not None:
        accC = accC + bC.float().view(1, -1)

    save_txt(accC.numpy().astype(np.int32), os.path.join(outdir, 'classifier_acc.txt'))

    print("\n[Done] All layer outputs saved to:", outdir)
    print("  Files: input_q, conv1_acc, relu1_q, conv2_acc, relu2_q,")
    print("         linear_acc, linear_q, dnn_acc, relu3_q, classifier_acc")


# ==============================================================================
#  Entry
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth',    required=True, help='quantized_params.pth')
    parser.add_argument('--input',  required=True, help='input sample .npy, shape (1,C,H,W) float32')
    parser.add_argument('--outdir', default='./layer_outputs')
    parser.add_argument('--nbits',  type=int, default=8)
    args = parser.parse_args()

    x = np.load(args.input).astype(np.float32)
    x = torch.tensor(x)
    if x.dim() == 3:
        x = x.unsqueeze(0)   # add batch dim

    integer_forward(args.pth, x, args.outdir, args.nbits)