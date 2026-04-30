"""
extract_layers.py

從 quantized_params.pth 讀取量化參數，
對單筆輸入跑 integer-domain forward，
把每層的量化整數輸出存成 .txt（一個數字一行），
供 C model 比對用。

同時提取每層的離線參數：
  K^(k)  = Z3 + M * (q_bias^(k) - Z1 * sum_j(q2^(j,k)))   [per output channel, int64]
  M0^(k) = round(M^(k) * 2^n)                               [per output channel, int64]
  n      = shift amount                                      [scalar, int]

儲存格式（每層）：
  {layer}_M0.txt   : C_out 個 int64，順序 = output channel 0, 1, ..., C_out-1
  {layer}_n.txt    : 1 個 int，scalar
  {layer}_K.txt    : C_out 個 int64，順序 = output channel 0, 1, ..., C_out-1

用法：
    python extract_layers.py
        --pth  quantized_params.pth
        --input input_sample.npy      # shape: (1, C, H, W), float32
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
    np.savetxt(path, arr.flatten().astype(np.int64), fmt='%d')
    print(f"  saved {path}  shape={arr.shape}  range=[{arr.min()}, {arr.max()}]")


# ==============================================================================
#  離線參數計算：M0, n, K
# ==============================================================================

def compute_offline_params(
    s_in:     float,          # 輸入 activation scale (scalar)
    scale_w:  np.ndarray,     # weight scale, shape (C_out,)，per-channel
    s_out:    float,          # 輸出 activation scale (scalar)
    zp_out:   int,            # 輸出 activation zero-point Z3 (scalar)
    zp_in:    int,            # 輸入 activation zero-point Z1 (scalar)
    w_int:    np.ndarray,     # 量化後的 weight，shape (C_out, ...)，int8
    b_int:    np.ndarray,     # 量化後的 bias，shape (C_out,)，int32
    total_bits: int = 31
):
    """
    計算每層的離線參數 M0, n, K。

    符號對應（依照推導）：
        q1 = activation (input)，非對稱，zero-point = Z1 = zp_in
        q2 = weight，對稱，Z2 = 0
        q3 = output activation，非對稱，zero-point = Z3 = zp_out

    公式（全部 per output channel）：
        M^(k)  = s_in * scale_w^(k) / s_out
        n^(k)  = 讓 M^(k) * 2^n^(k) 落在 [0.5, 1) 的 shift 量
        M0^(k) = round(M^(k) * 2^n^(k))，保證 M0 ∈ [2^(total_bits-1), 2^total_bits)

        K^(k) = Z3 + round( M^(k) * (q_bias^(k) - Z1 * weight_col_sum^(k)) )
        （直接 FP 乘法，離線不需省運算）

    回傳：
        M0 : np.ndarray, shape (C_out,), dtype int64
        n  : np.ndarray, shape (C_out,), dtype int64   ← per-channel
        K  : np.ndarray, shape (C_out,), dtype int64
    """

    # ── M per output channel ──────────────────────────────────────────────────
    M = (s_in * scale_w) / s_out                           # (C_out,), float64

    # ── n per output channel：對每個 channel 各自正規化 ──────────────────────
    # 找最小的 n 使 M * 2^n >= 2^(total_bits-1)（即正規化後 >= 0.5）
    n = np.zeros(len(M), dtype=np.int64)
    for idx, m in enumerate(M):
        while m * (2 ** int(n[idx])) < 0.5:
            n[idx] += 1

    # ── M0 per output channel ─────────────────────────────────────────────────
    M0 = np.zeros(len(M), dtype=np.uint32)
    temp = M * (2 ** n)                                      # (C_out,), float64
    for i in range(len(M)):
        val = temp[i]
        m0 = 0
        for _ in range(32):
            val *= 2
            m0 *= 2          # m0 *= 2
            if val >= 1.0:
                val -= 1.0
                m0 += 1       # m0 += 1

        M0[i] = m0

    print(M0)

    # ── weight column sum per output channel ─────────────────────────────────
    w_flat = w_int.reshape(w_int.shape[0], -1)              # (C_out, N)
    weight_col_sum = w_flat.sum(axis=1).astype(np.int32)    # (C_out,)

    # ── bias ──────────────────────────────────────────────────────────────────
    if b_int is None:
        b_int_arr = np.zeros(w_int.shape[0], dtype=np.float64)
    else:
        b_int_arr = b_int.astype(np.float64)                # (C_out,)

    # ── K per output channel（FP 乘法）───────────────────────────────────────
    correction = b_int_arr - float(zp_in) * weight_col_sum.astype(np.float64)
    K = np.round(float(zp_out) + M * correction).astype(np.int64)  # (C_out,)

    return M0, n, K


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

    def save_offline(layer_name, M0, n, K):
        """統一儲存 M0, n, K（全部 per-channel，C_out 個值）"""
        save_txt(M0, os.path.join(outdir, f'{layer_name}_M0.txt'))
        save_txt(n,  os.path.join(outdir, f'{layer_name}_n.txt'))
        save_txt(K,  os.path.join(outdir, f'{layer_name}_K.txt'))

    # ── 輸入量化 ──────────────────────────────────────────────────────────────
    s_in, zp_in = aq('input')
    x_q = to_uint8(x_fp.float(), s_in, zp_in, n_bits)   # uint8, shape (1,C,H,W)
    save_txt(x_q.numpy(), os.path.join(outdir, 'input_q.txt'))

    # ── conv1 ─────────────────────────────────────────────────────────────────
    s_in_l,  zp_in_l  = aq('input')
    s_out_l, zp_out_l = aq('relu1')

    w1  = wq['conv1']['w_int']                           # (OC, IC, kH, kW) int8
    b1  = wq['conv1']['b_int']                           # (OC,) int32

    save_txt(w1.numpy().astype(np.int8), os.path.join(outdir, 'conv1_w.txt'))

    # 計算並儲存離線參數
    M0_1, n1, K1 = compute_offline_params(
        s_in=s_in_l,
        scale_w=wq['conv1']['scale_w'].numpy(),
        s_out=s_out_l,
        zp_out=zp_out_l,
        zp_in=zp_in_l,
        w_int=w1.numpy(),
        b_int=b1.numpy() if b1 is not None else None,
    )
    save_offline('conv1', M0_1, n1, K1)

    # integer forward
    x_shifted = (x_q - zp_in_l).float()
    acc1 = F.conv2d(x_shifted, w1.float(), None, stride=(1,1), padding=0)
    if b1 is not None:
        acc1 = acc1 + b1.float().view(1, -1, 1, 1)

    save_txt(acc1.numpy().astype(np.int32), os.path.join(outdir, 'conv1_acc.txt'))

    M0_1_t = torch.tensor(M0_1, dtype=torch.int64).view(1, -1, 1, 1)
    n1_t   = torch.tensor(n1,   dtype=torch.int64).view(1, -1, 1, 1)
    acc1_i  = acc1.to(torch.int64)
    req1    = ((acc1_i * M0_1_t) >> n1_t) + zp_out_l
    req1    = req1.clamp(0, 255).to(torch.int32)

    relu1_q = req1.clamp(min=zp_out_l)
    save_txt(relu1_q.numpy(), os.path.join(outdir, 'relu1_q.txt'))

    # ── conv2 ─────────────────────────────────────────────────────────────────
    s_in_l,  zp_in_l  = aq('relu1')
    s_out_l, zp_out_l = aq('relu2')

    w2  = wq['conv2']['w_int']
    b2  = wq['conv2']['b_int']

    M0_2, n2, K2 = compute_offline_params(
        s_in=s_in_l,
        scale_w=wq['conv2']['scale_w'].numpy(),
        s_out=s_out_l,
        zp_out=zp_out_l,
        zp_in=zp_in_l,
        w_int=w2.numpy(),
        b_int=b2.numpy() if b2 is not None else None,
    )
    save_offline('conv2', M0_2, n2, K2)

    x_shifted = (relu1_q - zp_in_l).float()
    acc2 = F.conv2d(x_shifted, w2.float(), None, stride=(1,1), padding=0)
    if b2 is not None:
        acc2 = acc2 + b2.float().view(1, -1, 1, 1)

    save_txt(acc2.numpy().astype(np.int32), os.path.join(outdir, 'conv2_acc.txt'))

    M0_2_t = torch.tensor(M0_2, dtype=torch.int64).view(1, -1, 1, 1)
    n2_t   = torch.tensor(n2,   dtype=torch.int64).view(1, -1, 1, 1)
    acc2_i  = acc2.to(torch.int64)
    req2    = ((acc2_i * M0_2_t) >> n2_t) + zp_out_l
    req2    = req2.clamp(0, 255).to(torch.int32)

    relu2_q = req2.clamp(min=zp_out_l)
    save_txt(relu2_q.numpy(), os.path.join(outdir, 'relu2_q.txt'))

    # ── flatten ───────────────────────────────────────────────────────────────
    x_flat = relu2_q.view(1, -1)   # (1, N)

    # ── linear ────────────────────────────────────────────────────────────────
    s_in_l,  zp_in_l  = aq('relu2')
    s_out_l, zp_out_l = aq('linear')

    wL  = wq['linear']['w_int']   # (OC, IC)
    bL  = wq['linear']['b_int']

    M0_L, nL, KL = compute_offline_params(
        s_in=s_in_l,
        scale_w=wq['linear']['scale_w'].numpy(),
        s_out=s_out_l,
        zp_out=zp_out_l,
        zp_in=zp_in_l,
        w_int=wL.numpy(),
        b_int=bL.numpy() if bL is not None else None,
    )
    save_offline('linear', M0_L, nL, KL)

    x_shifted = (x_flat - zp_in_l).float()
    accL = F.linear(x_shifted, wL.float(), None)
    if bL is not None:
        accL = accL + bL.float().view(1, -1)

    save_txt(accL.numpy().astype(np.int32), os.path.join(outdir, 'linear_acc.txt'))

    M0_L_t = torch.tensor(M0_L, dtype=torch.int64).view(1, -1)
    nL_t   = torch.tensor(nL,   dtype=torch.int64).view(1, -1)
    accL_i  = accL.to(torch.int64)
    reqL    = ((accL_i * M0_L_t) >> nL_t) + zp_out_l
    linear_q = reqL.clamp(0, 255).to(torch.int32)
    save_txt(linear_q.numpy(), os.path.join(outdir, 'linear_q.txt'))

    # ── dnn ───────────────────────────────────────────────────────────────────
    s_in_l,  zp_in_l  = aq('linear')
    s_out_l, zp_out_l = aq('relu3')

    wD  = wq['dnn']['w_int']
    bD  = wq['dnn']['b_int']

    M0_D, nD, KD = compute_offline_params(
        s_in=s_in_l,
        scale_w=wq['dnn']['scale_w'].numpy(),
        s_out=s_out_l,
        zp_out=zp_out_l,
        zp_in=zp_in_l,
        w_int=wD.numpy(),
        b_int=bD.numpy() if bD is not None else None,
    )
    save_offline('dnn', M0_D, nD, KD)

    x_shifted = (linear_q - zp_in_l).float()
    accD = F.linear(x_shifted, wD.float(), None)
    if bD is not None:
        accD = accD + bD.float().view(1, -1)

    save_txt(accD.numpy().astype(np.int32), os.path.join(outdir, 'dnn_acc.txt'))

    M0_D_t = torch.tensor(M0_D, dtype=torch.int64).view(1, -1)
    nD_t   = torch.tensor(nD,   dtype=torch.int64).view(1, -1)
    accD_i  = accD.to(torch.int64)
    reqD    = ((accD_i * M0_D_t) >> nD_t) + zp_out_l
    reqD    = reqD.clamp(0, 255).to(torch.int32)

    relu3_q = reqD.clamp(min=zp_out_l)
    save_txt(relu3_q.numpy(), os.path.join(outdir, 'relu3_q.txt'))

    # ── classifier ────────────────────────────────────────────────────────────
    # 最後一層不做 requant，logits 保持 int32
    # 但仍計算 K, M0, n 供 C model 參考（若之後需要 requant 可直接使用）
    s_in_l,  zp_in_l  = aq('relu3')

    wC  = wq['classifier']['w_int']
    bC  = wq['classifier']['b_int']

    x_shifted = (relu3_q - zp_in_l).float()
    accC = F.linear(x_shifted, wC.float(), None)
    if bC is not None:
        accC = accC + bC.float().view(1, -1)

    save_txt(accC.numpy().astype(np.int32), os.path.join(outdir, 'classifier_acc.txt'))

    # ===========================================================================
    print("\n" + "="*60)
    print("[Done] All layer outputs saved to:", outdir)
    print("="*60)
    print("\n[Activation outputs]")
    print("  input_q, conv1_acc, relu1_q, conv2_acc, relu2_q,")
    print("  linear_acc, linear_q, dnn_acc, relu3_q, classifier_acc")
    print("\n[Offline params] 每層三個檔案，全部 per-channel（C_out 個值）：")
    print("  {layer}_M0.txt : C_out 個 int64，index 0 = output channel 0")
    print("  {layer}_n.txt  : C_out 個 int64，每個 channel 各自的 shift 量")
    print("  {layer}_K.txt  : C_out 個 int64，index 0 = output channel 0")
    print("\n  Layers: conv1, conv2, linear, dnn")
    print("\n[K 的計算公式]")
    print("  K[k] = Z3 + round( M[k] * (q_bias[k] - Z1 * weight_col_sum[k]) )")
    print("\n[Inference 時使用方式]")
    print("  q3[i,k] = ((acc[i,k] * M0[k]) >> n[k]) + K[k]")

# ==============================================================================
#  Entry
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth',    default='./model_params/quantized_params_75.pth', help='quantized_params.pth')
    parser.add_argument('--input',  default='input_sample.npy', help='input sample .npy, shape (1,C,H,W) float32')
    parser.add_argument('--outdir', default='./layer_outputs')
    parser.add_argument('--nbits',  type=int, default=8)
    args = parser.parse_args()

    x = np.load(args.input, allow_pickle=True).astype(np.float32)
    x = torch.tensor(x)
    if x.dim() == 3:
        x = x.unsqueeze(0)

    integer_forward(args.pth, x, args.outdir, args.nbits)