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

def to_int8(x_fp: torch.Tensor, scale: float, zp: int = 0, n_bits: int = 8):
    """float tensor → quantized int8 tensor（對稱，Z=0，整數域）"""
    qmin = -(2 ** (n_bits - 1) - 1)   # -127
    qmax =   2 ** (n_bits - 1) - 1    #  127
    x_q = (x_fp / scale + zp).round().clamp(qmin, qmax).to(torch.int32)
    return x_q

def save_txt(arr: np.ndarray, path: str):
    """flatten 後一行一個整數存檔"""
    np.savetxt(path, arr.flatten(), fmt='%.16f' if arr.dtype == np.float32 else '%d')
    print(f"  saved {path}  shape={arr.shape}  range=[{arr.min()}, {arr.max()}]")

def save_zp(zp: int, path: str):
    """儲存 zero-point（純 scalar）"""
    with open(path, 'w') as f:
        f.write(str(zp))
    print(f"  saved {path}  zero_point={zp}")

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
    K = np.round(float(zp_out) + M * correction).astype(np.int32)  # (C_out,)

    return M0, n, K

# ==============================================================================
#  Integer-domain forward（全程 int32 累加，不 dequant）
# ==============================================================================

def integer_forward(pth_path: str, pruned_pth_path: str,
                    x_fp: torch.Tensor, outdir: str,
                    cfg: dict, n_bits: int = 8):
    """
    做法：
      1. 載入 pruned model 結構 + state_dict（pruned_pth_path）
      2. 把 quantized_params.pth 的 w_int * scale_w 塞進 model weights
      3. 用 hook 攔截各層輸出（float domain）
      4. 用各層的 S, Z 把 float 輸出量化成 uint8 存檔
    """
    import torch.nn as nn
    from training_code.model import CNNTradFpool3
    from training_code.pruning import prune_cnntrad
 
    os.makedirs(outdir, exist_ok=True)
 
    # ── 載入量化參數 ───────────────────────────────────────────────────────────
    ckpt      = torch.load(pth_path, map_location='cpu', weights_only=False)
    act_quant = ckpt['act_quant']
    wq        = ckpt['weight_quant']
 
    def aq(name):
        return float(act_quant[name]['scale']), int(act_quant[name]['zero_point'])
 
    def save_offline(layer_name, M0, n, K):
        save_txt(M0, os.path.join(outdir, f'{layer_name}_M0.txt'))
        save_txt(n,  os.path.join(outdir, f'{layer_name}_n.txt'))
        save_txt(K,  os.path.join(outdir, f'{layer_name}_K.txt'))
 
    # ── 建立 pruned model 結構並載入 state_dict ───────────────────────────────
    model = CNNTradFpool3(num_classes=cfg['num_classes'],
                          dropout_rate=cfg['dropout_rate'])
    model = prune_cnntrad(model,
                          prune_ratio_conv1=0.5,
                          prune_ratio_conv2=0.75)
    pruned_ckpt = torch.load(pruned_pth_path, map_location='cpu', weights_only=False)
    sd = pruned_ckpt.get('model_state_dict', pruned_ckpt)
    model.load_state_dict(sd)
 
    # ── 把量化 weights（w_int * scale_w）塞進 model ──────────────────────────
    # 讓 model forward 用的是 fake-quantized float weights
    in_act_map = {
        'conv1':      'input',
        'conv2':      'sigmoid1',
        'linear':     'sigmoid2',
        'dnn':        'linear',
        'classifier': 'sigmoid3',
    }
    for lname, module in [
        ('conv1',      model.conv1),
        ('conv2',      model.conv2),
        ('linear',     model.linear),
        ('dnn',        model.dnn),
        ('classifier', model.classifier),
    ]:
        q   = wq[lname]
        s_w = q['scale_w']
        w_fq = q['w_int'].float() * s_w.view(-1, *([1] * (q['w_int'].dim() - 1)))
        with torch.no_grad():
            module.weight.copy_(w_fq)
            if q['b_int'] is not None and module.bias is not None:
                s_in_b = float(act_quant[in_act_map[lname]]['scale'])
                b_fq   = q['b_int'].float() * s_in_b * s_w
                module.bias.copy_(b_fq)
 
    model.eval()
 
    # ── 計算並儲存離線參數（M0, n, K）────────────────────────────────────────
    # out_name 是 MAC 輸出 requantize 的目標：
    #   conv/dnn 後接 sigmoid → requantize 到 sigmoid 輸入的 scale（act_quant['conv1'] 等）
    #   linear/classifier 後無 sigmoid → requantize 到自己輸出的 scale
    layer_params = [
        ('conv1',      'input',    'conv1',      model.conv1),
        ('conv2',      'sigmoid1', 'conv2',      model.conv2),
        ('linear',     'sigmoid2', 'linear',     model.linear),
        ('dnn',        'linear',   'dnn',        model.dnn),
        ('classifier', 'sigmoid3', 'classifier', model.classifier),
    ]
    for lname, in_name, out_name, module in layer_params:
        s_in_l,  zp_in_l  = aq(in_name)
        s_out_l, zp_out_l = aq(out_name)
        save_zp(zp_out_l, os.path.join(outdir, f'{lname}_zp_out.txt'))
        q = wq[lname]
        M0, n, K = compute_offline_params(
            s_in=s_in_l, scale_w=q['scale_w'].numpy(),
            s_out=s_out_l, zp_out=zp_out_l, zp_in=zp_in_l,
            w_int=q['w_int'].numpy(),
            b_int=q['b_int'].numpy() if q['b_int'] is not None else None,
        )
        save_offline(lname, M0, n, K)
        save_txt(q['w_int'].numpy().astype(np.int8),
                 os.path.join(outdir, f'{lname}_w.txt'))
        if q['b_int'] is not None:
            save_txt(q['b_int'].numpy(),
                     os.path.join(outdir, f'{lname}_b.txt'))
 
    # ── 用 hook 攔截各層輸出 ──────────────────────────────────────────────────
    layer_outputs = {}

    def make_hook(name):
        def hook(module, input, output):
            layer_outputs[name] = output.detach().float()
        return hook

    # sigmoid 輸入（conv/dnn 的 MAC 輸出，requantize 後才進 sigmoid）
    model.conv1.register_forward_hook(make_hook('conv1_out'))
    model.conv2.register_forward_hook(make_hook('conv2_out'))
    model.dnn.register_forward_hook(make_hook('dnn_out'))
    # sigmoid 輸出
    model.sigmoid1.register_forward_hook(make_hook('sigmoid1_out'))
    model.sigmoid2.register_forward_hook(make_hook('sigmoid2_out'))
    model.sigmoid3.register_forward_hook(make_hook('sigmoid3_out'))
    # linear / classifier
    model.linear.register_forward_hook(make_hook('linear_out'))
    model.classifier.register_forward_hook(make_hook('classifier_out'))
 
    # ── 量化輸入 + dequant 回 float 作為 model 輸入 ───────────────────────────
    s_in, zp_in = aq('input')
    x_q   = to_uint8(x_fp.float(), s_in, zp_in, n_bits)
    x_deq = (x_q.float() - zp_in) * s_in
 
    save_txt(x_q.numpy(), os.path.join(outdir, 'input_q.txt'))
 
    # ── 跑 forward（hook 自動填 layer_outputs）────────────────────────────────
    with torch.no_grad():
        _ = model(x_deq)
 
    # ── 把各層 float 輸出量化成整數存檔 ──────────────────────────────────────
    # conv1/conv2/dnn 是 sigmoid 前，對稱量化（int8，Z=0）
    # sigmoid 輸出、linear 是非對稱量化（uint8）
    SYMMETRIC_LAYERS = {'conv1', 'conv2', 'dnn'}

    def quantize_and_save(hook_name, aq_name, file_name):
        s, zp = aq(aq_name)
        if aq_name in SYMMETRIC_LAYERS:
            out_q = to_int8(layer_outputs[hook_name], s, zp, n_bits)
        else:
            out_q = to_uint8(layer_outputs[hook_name], s, zp, n_bits)
        save_txt(out_q.numpy(), os.path.join(outdir, file_name))

    # sigmoid 輸入（int8，對稱量化）
    quantize_and_save('conv1_out',    'conv1',    'conv1_q.txt')
    quantize_and_save('conv2_out',    'conv2',    'conv2_q.txt')
    quantize_and_save('dnn_out',      'dnn',      'dnn_q.txt')
    # sigmoid 輸出（uint8，非對稱量化）
    quantize_and_save('sigmoid1_out', 'sigmoid1', 'sigmoid1_q.txt')
    quantize_and_save('sigmoid2_out', 'sigmoid2', 'sigmoid2_q.txt')
    quantize_and_save('sigmoid3_out', 'sigmoid3', 'sigmoid3_q.txt')
    # linear（uint8，非對稱量化）
    quantize_and_save('linear_out',   'linear',   'linear_q.txt')

    # classifier 不量化，存 float logits
    save_txt(layer_outputs['classifier_out'].numpy(),
             os.path.join(outdir, 'classifier_out.txt'))

    print("\n" + "="*60)
    print("[Done] All layer outputs saved to:", outdir)
    print("="*60)
    print("  MAC outputs (sigmoid 前) : conv1_q, conv2_q, dnn_q")
    print("  sigmoid outputs          : sigmoid1_q, sigmoid2_q, sigmoid3_q")
    print("  linear_q, classifier_out")
    print("  weights : conv1_w, conv2_w, linear_w, dnn_w + bias")
    print("  offline : {layer}_M0, {layer}_n, {layer}_K  (per-channel)")
 
 
# ==============================================================================
#  Entry
# ==============================================================================
 
CFG = dict(
    num_classes       = 12,
    dropout_rate      = 0.1,
    prune_ratio_conv1 = 0.5,
    prune_ratio_conv2 = 0.5,
)
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth',        default='./model_params/quantized_params_75_sigmoid.pth',
                        help='quantized_params.pth')
    parser.add_argument('--pruned_pth', default='./model_params/pruned_cosinelr_75_sigmoid.pth',
                        help='pruned_cosinelr.pth（提供 model 結構與 float weights）')
    parser.add_argument('--input',      default='./input_sample.npy',
                        help='input sample .npy, shape (1,1,T,F) float32')
    parser.add_argument('--outdir',     default='./layer_outputs_sigmoid')
    parser.add_argument('--nbits',      type=int, default=8)
    args = parser.parse_args()
 
    x = np.load(args.input, allow_pickle=True).astype(np.float32)
    x = torch.tensor(x)
    if x.dim() == 3:
        x = x.unsqueeze(0)
 
    integer_forward(args.pth, args.pruned_pth, x, args.outdir, CFG, args.nbits)