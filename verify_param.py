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
                 w_file: str, bias_file: str,
                 zp_in: int, zp_out: int,
                 stride=(1,1), padding=0):
    """
    三項驗證：
      1. M0 範圍是否正確（M0 ∈ [2^29, 2^31)）
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
    n  = load_scalar(n_path)        # scalar
    K  = load_int(K_path)           # (C_out,)

    C_out = M0.shape[0]
    print(f"  C_out={C_out}, n={n}")

    # ── 驗證 1：M0 範圍 ───────────────────────────────────────────────────────
    # M0 = round(M * 2^n)，M ∈ (0,1)
    # 正規化條件：M0_max >= 2^(n-1)（確保最高位有效）
    # 且 M0 < 2^n（不超過 shift 範圍）
    lo = 2 ** (n - 1)      # M0 對應 M >= 0.5 的下界
    hi = 2 ** n            # 上界（不含）
    in_range = np.all((M0 >= lo) & (M0 < hi))
    status = PASS if in_range else WARN
    # 注意：per-channel scale_w 不同，部分 channel 的 M 可能 < 0.5
    # 所以這裡只警告，不直接判 FAIL
    print(f"  {status} M0 範圍檢查：min={M0.min()}, max={M0.max()}  "
          f"期望 [{lo}, {hi})（per-channel 允許部分不在範圍）")
    print(f"         在範圍內的 channel 數：{np.sum((M0 >= lo) & (M0 < hi))} / {C_out}")

    # ── 載入 input, weight, bias, reference ──────────────────────────────────
    x_q  = load_int(os.path.join(outdir, input_file))
    ref  = load_int(os.path.join(outdir, ref_output_file))
    w    = load_int(os.path.join(outdir, w_file)).astype(np.int64)
    bias = load_int(os.path.join(outdir, bias_file)) if bias_file else None

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
        acc_with_bias = load_int(os.path.join(outdir, f'{layer}_acc.txt'))

        # n_spatial 必須在使用前先算，不依賴 bias 是否存在
        n_spatial = acc_with_bias.shape[0] // C_out

        # broadcast bias 和 weight_col_sum
        if bias is not None:
            bias_bc = np.repeat(bias, n_spatial)           # (C_out * n_spatial,)
        else:
            bias_bc = np.zeros(C_out * n_spatial, dtype=np.int64)

        w_reshaped     = w.reshape(C_out, -1)
        weight_col_sum = w_reshaped.sum(axis=1)            # (C_out,)
        wcs_bc         = np.repeat(weight_col_sum, n_spatial)

        # acc_pure（K 用的）= acc_with_bias - bias + Z1 * weight_col_sum
        acc_pure_for_K = acc_with_bias - bias_bc + zp_in * wcs_bc

        # 用 K 重建輸出
        K_bc  = np.repeat(K,  n_spatial)
        M0_bc = np.repeat(M0, n_spatial)
        q3_from_K  = ((acc_pure_for_K * M0_bc) >> n) + K_bc
        q3_ref_raw = ((acc_with_bias  * M0_bc) >> n) + zp_out

    else:
        # Linear layer
        acc_with_bias = load_int(os.path.join(outdir, f'{layer}_acc.txt'))
        n_spatial = 1

        if bias is not None:
            acc_pure_flat = acc_with_bias - bias
        else:
            acc_pure_flat = acc_with_bias.copy()

        w_reshaped = w.reshape(C_out, -1)
        weight_col_sum = w_reshaped.sum(axis=1)

        acc_pure_for_K = acc_with_bias - (bias if bias is not None else 0) \
                         + zp_in * weight_col_sum

        q3_from_K  = ((acc_pure_for_K * M0) >> n) + K
        q3_ref_raw = ((acc_with_bias  * M0) >> n) + zp_out

    # 比較兩條路徑
    diff_K = np.abs(q3_from_K - q3_ref_raw)
    max_diff_K = diff_K.max()
    status = PASS if max_diff_K <= 1 else FAIL
    print(f"  {status} K 等價性：max |q3_K - q3_ref| = {max_diff_K}  "
          f"（≤1 為正常 rounding 誤差）")
    if max_diff_K > 1:
        bad_idx = np.where(diff_K > 1)[0]
        print(f"         差異超過 1 的位置數量：{len(bad_idx)}")
        print(f"         前5個：idx={bad_idx[:5]}, "
              f"K路徑={q3_from_K[bad_idx[:5]]}, "
              f"ref={q3_ref_raw[bad_idx[:5]]}")

    # ── 驗證 3：最終輸出一致性 ────────────────────────────────────────────────
    # ref 是 extract_layers.py 存的 relu_q（含 clamp）
    # 我們用 K 路徑重建，加上 clamp，比對 ref
    q3_K_clamped = np.clip(q3_from_K, 0, 255)
    # 若有 ReLU：再 clamp min=zp_out
    q3_K_relu    = np.clip(q3_K_clamped, zp_out, 255)

    diff_out = np.abs(q3_K_relu - ref)
    max_diff_out = diff_out.max()
    exact_match  = (diff_out == 0).all()

    status = PASS if max_diff_out <= 1 else FAIL
    print(f"  {status} 最終輸出比對：max |output_K - ref| = {max_diff_out}  "
          f"完全一致={exact_match}")
    if max_diff_out > 1:
        bad_idx = np.where(diff_out > 1)[0]
        print(f"         差異超過 1 的位置數量：{len(bad_idx)}")

    return (max_diff_K <= 1) and (max_diff_out <= 1)


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
        bias_file    = None,          # acc 已含 bias，bias 從 pth 拿
        zp_in        = zp('input'),
        zp_out       = zp('relu1'),
    )

    # ── conv2 ─────────────────────────────────────────────────────────────────
    results['conv2'] = verify_layer(
        outdir       = outdir,
        layer        = 'conv2',
        is_conv      = True,
        input_file   = 'relu1_q.txt',
        ref_output_file = 'relu2_q.txt',
        w_file       = 'conv2_w.txt',
        bias_file    = None,
        zp_in        = zp('relu1'),
        zp_out       = zp('relu2'),
    )

    # ── linear ────────────────────────────────────────────────────────────────
    results['linear'] = verify_layer(
        outdir       = outdir,
        layer        = 'linear',
        is_conv      = False,
        input_file   = 'relu2_q.txt',
        ref_output_file = 'linear_q.txt',
        w_file       = 'linear_w.txt',
        bias_file    = None,
        zp_in        = zp('relu2'),
        zp_out       = zp('linear'),
    )

    # ── dnn ───────────────────────────────────────────────────────────────────
    results['dnn'] = verify_layer(
        outdir       = outdir,
        layer        = 'dnn',
        is_conv      = False,
        input_file   = 'linear_q.txt',
        ref_output_file = 'relu3_q.txt',
        w_file       = 'dnn_w.txt',
        bias_file    = None,
        zp_in        = zp('linear'),
        zp_out       = zp('relu3'),
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
    parser.add_argument('--pth',    required=True,
                        help='quantized_params.pth（讀取 zp 用）')
    args = parser.parse_args()

    verify_all(args.outdir, args.pth)