"""
verify_params.py

驗證 extract_layers.py 輸出的離線參數（K, M0, n）是否正確。

驗證邏輯：
  extract_layers.py 的原始 inference 流程（以 conv1 為例）：

    [原始流程]
    acc = conv(x_shifted, w) + bias          # x_shifted = x_q - Z1
    q3  = ((acc * M0) >> n) + Z3             # 不含 K

    [K 的等價流程]
    acc_pure = conv(x_shifted, w)            # 不加 bias，且 x_shifted = x_q（不減 Z1）
    q3       = ((acc_pure * M0) >> n) + K    # K 已折入 bias、Z1 修正、Z3

  兩者輸出應完全相同（差異 ≤ 1 LSB，來自 K 的 FP round）。

用法：
    python verify_params.py --outdir ./layer_outputs
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F


PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"


# ==============================================================================
#  載入工具
# ==============================================================================

def load_int(path: str) -> np.ndarray:
    return np.loadtxt(path, dtype=np.int64)

def load_scalar(path: str) -> int:
    return int(np.loadtxt(path, dtype=np.int64))


# ==============================================================================
#  單層驗證
# ==============================================================================

def verify_layer(outdir: str, layer: str, is_conv: bool,
                 input_file: str, ref_output_file: str,
                 w_file: str, relu: bool,
                 zp_in: int, zp_out: int, input_shape: tuple, weight_shape: tuple,
                 stride=(1,1), padding=0):
    """
    三項驗證：
      1. M0 範圍是否正確（M0 ∈ [2^n-1, 2^n)）
      2. K 的等價重建：pure MAC + K == 原始 acc（含bias）+ Z3 的結果
      3. 最終 requant 輸出是否和 reference 一致
    """
    print(f"\n{'='*55}")
    print(f"  Layer: {layer}")
    print(f"{'='*55}")

    # ── 載入離線參數 ──────────────────────────────────────────────────────────
    M0_path = os.path.join(outdir, f'{layer}_M0.txt')
    n_path  = os.path.join(outdir, f'{layer}_n.txt')
    K_path  = os.path.join(outdir, f'{layer}_K.txt')

    if not all(os.path.exists(p) for p in [M0_path, n_path, K_path]):
        print(f"  {WARN} 找不到離線參數檔案，跳過此層")
        return False

    M0 = load_int(M0_path)          # (C_out,)
    n  = load_int(n_path)           # (C_out,)，per-channel
    K  = load_int(K_path)           # (C_out,)

    C_out = M0.shape[0]
    print(f"  C_out={C_out}, n min={n.min()}, n max={n.max()}")

    # ── 驗證 1：M0 範圍（per-channel）────────────────────────────────────────
    # 每個 channel：M0[k] 應在 [2^(n[k]-1), 2^n[k])
    in_range = 0
    lo = 2 ** 31
    hi = 2 ** 32
    in_range_per_ch = (M0 >= lo) & (M0 < hi)
    n_in_range = in_range_per_ch.sum()

    status = PASS if n_in_range == C_out else WARN
    print(f"  {status} M0 範圍檢查：在範圍內 {n_in_range}/{C_out} 個 channel")

    # ── 載入 input, weight, bias, reference ──────────────────────────────────
    x_q  = load_int(os.path.join(outdir, input_file))
    ref  = load_int(os.path.join(outdir, ref_output_file))
    w    = load_int(os.path.join(outdir, w_file)).astype(np.int8)

    # ── 驗證 2：K 等價性 ──────────────────────────────────────────────────────
    # 原始流程：acc_with_bias = MAC(x_q - Z1, w) + bias
    #           q3_ref = ((acc_with_bias * M0) >> n) + Z3
    #
    # K 流程：  acc_pure = MAC(x_q, w)         ← 不減 Z1，不加 bias
    #           q3_K    = ((acc_pure * M0) >> n) + K

    if is_conv:
        # 從 acc 檔案拿原始 accumulator（含 bias，flatten 為 1D）
        # flatten 順序：(1, C_out, H_out, W_out) → row-major
        # 每個 output channel 連續 n_spatial 個元素
        acc = F.conv2d(torch.from_numpy(x_q).float().reshape(input_shape),
                torch.from_numpy(w).float().reshape(weight_shape), bias=None, stride=stride, padding=padding).flatten()

        # n_spatial 必須在使用前先算，不依賴 bias 是否存在
        n_spatial = acc.shape[0] // C_out

        # 用 K 重建輸出
        # per-channel n broadcast
        n_bc  = np.repeat(n,  n_spatial)
        M0_bc = np.repeat(M0, n_spatial)
        K_bc  = np.repeat(K,  n_spatial)

        # 用 per-channel shift
        q3_from_K  = (((acc * M0_bc) / 2 ** n_bc) + K_bc * 2 ** 32) / 2 ** 32

    else:
        # Linear layer
        x_2d = torch.from_numpy(x_q).float().reshape(1, -1)
        acc = F.linear(x_2d, torch.from_numpy(w).float().reshape(weight_shape), bias=None).flatten()
        n_spatial = 1

        # per-channel shift（linear 無 spatial，直接用 array）
        q3_from_K  = (((acc * M0) / 2 ** n) + K * 2 ** 32) / 2 ** 32

    # ── 驗證 3：最終輸出一致性 ────────────────────────────────────────────────
    # ref 是 extract_layers.py 存的 relu_q（含 clamp）
    # 我們用 K 路徑重建，加上 clamp，比對 ref
    # q3_K_clamped = np.clip(q3_from_K, 0, 255)
    # 若有 ReLU：再 clamp min=zp_out
    if relu:
        q3_K_relu = np.clip(q3_from_K, zp_out, 255)
    else: 
        q3_K_relu = np.clip(q3_from_K, 0, 255)
    
    q3_K_relu = np.round(q3_K_relu)
    diff_out = np.abs(q3_K_relu - ref)
    for i in range(len(diff_out)):
        if diff_out[i] > 1:
            print(f"  {WARN} 輸出差異超過 1 的位置：idx={i}  q3_K={q3_K_relu[i]}  ref={ref[i]}  diff={diff_out[i]}")
            
    max_diff_out = diff_out.max()
    exact_match  = (diff_out == 0).all()

    status = PASS if max_diff_out <= 1 else FAIL
    print(f"  {status} 最終輸出比對：max |output_K - ref| = {max_diff_out}  "
          f"完全一致={exact_match}")
    if max_diff_out > 1:
        bad_idx = np.where(diff_out > 1)[0]
        print(f"         差異超過 1 的位置數量：{len(bad_idx)}")

    return (max_diff_out <= 1)


# ==============================================================================
#  主流程
# ==============================================================================

def verify_all(outdir: str, pth_path: str):
    """
    從 pth 讀取 zp_in, zp_out，對每層跑驗證。
    """
    import torch
    ckpt      = torch.load(pth_path, map_location='cpu', weights_only=False)
    act_quant = ckpt['act_quant']
    wq        = ckpt['weight_quant']

    def zp(name):
        return int(act_quant[name]['zero_point'])

    results = {}

    # ── conv1 ─────────────────────────────────────────────────────────────────
    results['conv1'] = verify_layer(
        outdir       = outdir,
        layer        = 'conv1',
        is_conv      = True,
        input_file   = 'input_q.txt',
        ref_output_file = 'relu1_q.txt',
        w_file       = 'conv1_w.txt',
        relu         = True,
        zp_in        = zp('input'),
        zp_out       = zp('relu1'),
        input_shape  = (1, 1, 101, 40),
        weight_shape = (32, 1, 67, 8)
    )

    # ── conv2 ─────────────────────────────────────────────────────────────────
    results['conv2'] = verify_layer(
        outdir       = outdir,
        layer        = 'conv2',
        is_conv      = True,
        input_file   = 'relu1_q.txt',
        ref_output_file = 'relu2_q.txt',
        w_file       = 'conv2_w.txt',
        relu         = True,
        zp_in        = zp('relu1'),
        zp_out       = zp('relu2'),
        input_shape  = (1, 32, 35, 33),
        weight_shape = (16, 32, 10, 4)
    )

    # ── linear ────────────────────────────────────────────────────────────────
    results['linear'] = verify_layer(
        outdir       = outdir,
        layer        = 'linear',
        is_conv      = False,
        input_file   = 'relu2_q.txt',
        ref_output_file = 'linear_q.txt',
        w_file       = 'linear_w.txt',
        relu         = False,
        zp_in        = zp('relu2'),
        zp_out       = zp('linear'),
        input_shape  = (1, 16, 26, 30),
        weight_shape = (32, 12480)
    )

    # ── dnn ───────────────────────────────────────────────────────────────────
    results['dnn'] = verify_layer(
        outdir       = outdir,
        layer        = 'dnn',
        is_conv      = False,
        input_file   = 'linear_q.txt',
        ref_output_file = 'relu3_q.txt',
        w_file       = 'dnn_w.txt',
        relu         = True,
        zp_in        = zp('linear'),
        zp_out       = zp('relu3'),
        input_shape  = (1, 32),
        weight_shape = (128, 32)
    )

    # ── 總結 ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  Summary")
    print(f"{'='*55}")
    all_pass = True
    for layer, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status}  {layer}")
        all_pass = all_pass and ok

    if all_pass:
        print(f"\n  {PASS} 所有層驗證通過，K / M0 / n 正確。")
    else:
        print(f"\n  {FAIL} 有層驗證失敗，請檢查 compute_offline_params()。")


# ==============================================================================
#  Entry
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='./layer_outputs',
                        help='extract_layers.py 的輸出目錄')
    parser.add_argument('--pth',  default='./model_params/quantized_params_75.pth',
                        help='quantized_params.pth（讀取 zp 用）')
    args = parser.parse_args()

    verify_all(args.outdir, args.pth)