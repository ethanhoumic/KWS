"""
extract_sigmoid_params.py

從 quantized_params_75_sigmoid.pth 讀取指定層的量化參數，
計算 piecewise linear sigmoid 近似所需的離線常數並存成 txt。

sigmoid 近似表格（輸入 X = S3*(q3-Z3)，輸出 Y = S4*(q4-Z4)）：
    |X| >= 5            → Y = 1
    2.375 =< |X| < 5   → Y = 0.03125 * |X| + 0.84375
    1     =< |X| < 2.375 → Y = 0.125  * |X| + 0.625
    0     =< |X| < 1   → Y = 0.25    * |X| + 0.5
    X < 0              → Y = 1 - Y(|X|)

因為 sigmoid 前是對稱量化（Z3 = 0），X = S3 * q3。

輸出參數（全部離線算好，inference 只做整數運算）：

  門檻值（比較 |q3| 用）：
    T1 = round(5      / S3)   對應 |X| >= 5
    T2 = round(2.375  / S3)   對應 |X| >= 2.375
    T3 = round(1      / S3)   對應 |X| >= 1

  各段的斜率（A）表示成 M0 + n：
    A_i = alpha_i * S3 / S4
    用 while 迴圈正規化：找最小 n 使 A_i * 2^n >= 0.5，M0 = round(A_i * 2^n)

  各段的截距（B，int64）：
    B_i = round(beta_i / S4 + Z4)

  負數對稱常數 C：
    C = round(1 / S4 + 2 * Z4)
    對 q3 < 0：q4 = C - q4_pos(|q3|)

用法：
    python extract_sigmoid_params.py
        --pth  quantized_params_75_sigmoid.pth
        --layer conv1          # sigmoid 前的層名稱（conv1 / conv2 / dnn）
        --outdir ./sigmoid_params
"""

import argparse
import os
import numpy as np
import torch


# ==============================================================================
#  分段線性近似的四段定義
#  (alpha, beta, threshold_X)
#  threshold_X 是這段的下界（|X| >= threshold_X 才進入此段）
# ==============================================================================

# 按 |X| 從大到小排列，方便 if-elif 判斷
SEGMENTS = [
    # (alpha,   beta,    threshold_X)
    (0.0,      1.0,      5.0   ),   # |X| >= 5       → Y = 1
    (0.03125,  0.84375,  2.375 ),   # 2.375 =< |X| < 5
    (0.125,    0.625,    1.0   ),   # 1     =< |X| < 2.375
    (0.25,     0.5,      0.0   ),   # 0     =< |X| < 1
]


# ==============================================================================
#  M0, n 正規化（和 compute_offline_params 相同邏輯）
# ==============================================================================

def to_m0_n(val: float, total_bits: int = 31):
    """
    把一個正浮點數 val 表示成 M0 * 2^(-n) 的形式。
    找最小的 n 使 val * 2^n >= 0.5，然後 M0 = round(val * 2^n)。
    若 val == 0，直接回傳 M0=0, n=0。
    """
    if val == 0.0:
        return np.int64(0), np.int64(0)
    n = np.int64(0)
    while val * (0.5 ** int(-n)) > 1:
        n -= 1
    M0 = np.int32(0)
    temp = val * (0.5 ** (-n))
    val = temp
    m0 = 0
    for _ in range(32):
        val *= 2
        m0 *= 2          # m0 *= 2
        if val >= 1.0:
            val -= 1.0
            m0 += 1       # m0 += 1

    M0 = m0
    n = -n
    return M0, n


# ==============================================================================
#  主函數
# ==============================================================================

def extract_sigmoid_params(pth_path: str, layer_name: str,
                            outdir: str, total_bits: int = 31):
    """
    layer_name : sigmoid 前的層，例如 'conv1'
                 對應 act_quant[layer_name]  = sigmoid 輸入的 S3, Z3
                 對應 act_quant[sigmoid_map[layer_name]] = sigmoid 輸出的 S4, Z4
    """
    # sigmoid 前的層 → sigmoid 輸出的 act_quant key
    sigmoid_map = {
        'conv1': 'sigmoid1',
        'conv2': 'sigmoid2',
        'dnn':   'sigmoid3',
    }
    assert layer_name in sigmoid_map, \
        f"layer_name 必須是 {list(sigmoid_map.keys())} 其中之一"

    os.makedirs(outdir, exist_ok=True)

    # ── 載入 pth ──────────────────────────────────────────────────────────────
    ckpt = torch.load(pth_path, map_location='cpu', weights_only=False)
    act_quant = ckpt['act_quant']

    S3 = float(act_quant[layer_name]['scale'])
    Z3 = int(act_quant[layer_name]['zero_point'])          # 應為 0（對稱量化）

    sig_key = sigmoid_map[layer_name]
    S4 = float(act_quant[sig_key]['scale'])
    Z4 = int(act_quant[sig_key]['zero_point'])

    print(f"Layer : {layer_name} → {sig_key}")
    print(f"  S3={S3:.8f},  Z3={Z3}  (sigmoid 輸入，對稱量化，Z3 應為 0)")
    print(f"  S4={S4:.8f},  Z4={Z4}  (sigmoid 輸出，非對稱量化)")

    assert Z3 == 0, f"Z3={Z3}，對稱量化應為 0，請確認 ptq 設定"

    # ── 門檻值 T1, T2, T3（整數，比較 |q3| 用）────────────────────────────────
    # X = S3 * q3（Z3=0），所以 |X| >= threshold → |q3| >= threshold / S3
    thresholds_X  = [5.0, 2.375, 1.0]
    threshold_names = ['T1', 'T2', 'T3']
    T = {}
    for name, thr in zip(threshold_names, thresholds_X):
        T[name] = int(round(thr / S3))
        print(f"  {name} = round({thr} / {S3:.6f}) = {T[name]}")

    # ── 各段的 A（斜率）= alpha * S3 / S4，存成 M0 + n ─────────────────────
    # 第一段 alpha=0（飽和），A=0
    # 其餘三段各有 M0_A 和 n_A
    A_params = {}
    for i, (alpha, beta, thr) in enumerate(SEGMENTS):
        seg_name = f'seg{i}'
        A_fp = alpha * S3 / S4                              # float
        M0_A, n_A = to_m0_n(A_fp, total_bits)
        A_params[seg_name] = {'alpha': alpha, 'A_fp': A_fp,
                               'M0_A': M0_A, 'n_A': n_A}
        print(f"  seg{i} (alpha={alpha}, thr={thr}): "
              f"A_fp={A_fp:.8f}, M0_A={M0_A}, n_A={n_A}")

    # ── 各段的 B（截距）= round(beta / S4 + Z4)，int64 ─────────────────────
    B_params = {}
    for i, (alpha, beta, thr) in enumerate(SEGMENTS):
        seg_name = f'seg{i}'
        B = int(round(beta / S4 + Z4))
        B_params[seg_name] = B
        print(f"  seg{i} (beta={beta}): B = round({beta}/{S4:.6f} + {Z4}) = {B}")

    # ── 負數對稱常數 C = round(1/S4 + 2*Z4) ──────────────────────────────────
    C = int(round(1.0 / S4 + 2 * Z4))
    print(f"  C = round(1/{S4:.6f} + 2*{Z4}) = {C}")

    # ── 存檔 ──────────────────────────────────────────────────────────────────
    prefix = os.path.join(outdir, f'{layer_name}_sigmoid')

    def save_scalar(val, path, fmt='%d'):
        np.savetxt(path, np.array([val]), fmt=fmt)
        print(f"  saved {path}  val={val}")

    # 門檻值
    save_scalar(T['T1'], f'{prefix}_T1.txt')
    save_scalar(T['T2'], f'{prefix}_T2.txt')
    save_scalar(T['T3'], f'{prefix}_T3.txt')

    # 各段 M0_A, n_A, B
    for i in range(len(SEGMENTS)):
        seg = f'seg{i}'
        save_scalar(A_params[seg]['M0_A'], f'{prefix}_{seg}_M0_A.txt')
        save_scalar(A_params[seg]['n_A'],  f'{prefix}_{seg}_n_A.txt')
        save_scalar(B_params[seg],         f'{prefix}_{seg}_B.txt')

    # 對稱常數 C
    save_scalar(C, f'{prefix}_C.txt')

    print(f"\n[Done] sigmoid params for '{layer_name}' saved to {outdir}")
    print("\n[Inference 使用方式]")
    print("  abs_q3 = abs(q3)")
    print("  if   abs_q3 >= T1 : q4_pos = B_seg0")
    print("  elif abs_q3 >= T2 : q4_pos = (M0_A_seg1 * abs_q3) >> n_A_seg1 + B_seg1")
    print("  elif abs_q3 >= T3 : q4_pos = (M0_A_seg2 * abs_q3) >> n_A_seg2 + B_seg2")
    print("  else               : q4_pos = (M0_A_seg3 * abs_q3) >> n_A_seg3 + B_seg3")
    print("  q4 = q4_pos  if q3 >= 0  else  C - q4_pos")
    print("  q4 = clamp(q4, 0, 255)")

    return T, A_params, B_params, C


# ==============================================================================
#  Entry
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth',   default='./model_params/quantized_params_75_sigmoid.pth')
    parser.add_argument('--layer', default='conv1',
                        help='sigmoid 前的層名稱：conv1 / conv2 / dnn')
    parser.add_argument('--outdir', default='./sigmoid_params')
    parser.add_argument('--bits',   type=int, default=32,
                        help='M0 正規化的 bit 數（預設 32）')
    args = parser.parse_args()

    extract_sigmoid_params(args.pth, args.layer, args.outdir, args.bits)