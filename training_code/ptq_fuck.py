import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from model import CNNTradFpool3
from pruning import prune_cnntrad
from train_pruned import SpeechCommandDataset, SpeechDataset, MelSpectrogram, CFG
from torch.utils.data import DataLoader
import random

# ====================================================================================
#                               Config
# ====================================================================================

PRUNED_CKPT       = 'relu/pruned_cosinelr_75.pth'
OUTPUT_NPZ        = 'quantized_params_75_relu_shit.npz'
N_CALIB           = 1000
BATCH_SIZE        = 100
NUM_BITS          = 8
PERCENTILE        = 99.99

# ====================================================================================
#       Weight Quantization (Unsigned, per-channel, Z_W = 0)
#       q_W in [0, 2^n - 1],  r_W = S_W * q_W   (weights forced to be >= 0 in real domain)
# ====================================================================================

def quantize_weight_unsigned(weight: torch.Tensor, n_bits: int = 8):
    """
    Per-output-channel UNSIGNED quantization with zero_point = 0.
    Real-domain weight is mapped as r_W = S_W * q_W, q_W in [0, qmax].

    Note: this forces all weights to be non-negative. Negative weights are clamped to 0.
    Use only if you really intend r_W >= 0 (or a separately-handled sign mechanism upstream).
    """
    qmax = 2 ** n_bits - 1                # 255 for INT8, 15 for INT4

    # max per output channel (NOT abs, because we are unsigned with Z_W = 0)
    if weight.dim() == 4:
        w_max = weight.amax(dim=(1, 2, 3))
    else:
        w_max = weight.amax(dim=1)

    # Guard: if a channel's max is <= 0 (all weights non-positive), set a tiny scale
    # so that the channel quantizes to all-zero rather than producing NaN.
    w_max = w_max.clamp(min=1e-8)

    scale = (w_max / qmax).clamp(min=1e-8)

    if weight.dim() == 4:
        scale_bc = scale.view(-1, 1, 1, 1)
    else:
        scale_bc = scale.view(-1, 1)

    # Negative weights → clamp to 0 (lost). This is the cost of Z_W = 0 unsigned.
    w_int = (weight / scale_bc).round().clamp(0, qmax).to(torch.int32)
    return w_int, scale


def quantize_bias(bias: torch.Tensor, scale_w: torch.Tensor,
                  scale_act_in: float, n_bits: int = 32):
    """
    Bias INT32, symmetric, S_B = S_W * S_A_in (per output channel).
    """
    scale_bias = scale_w * scale_act_in
    qmax = 2 ** (n_bits - 1) - 1
    b_int = (bias / scale_bias).round().clamp(-qmax - 1, qmax).to(torch.int32)
    return b_int, scale_bias


# ====================================================================================
#       Activation Quantization (SIGNED, per-tensor, asymmetric: Z_A free)
#       q_A in [-2^(n-1), 2^(n-1)-1],   r_A = S_A * (q_A - Z_A)
# ====================================================================================

def compute_signed_asymmetric_scale_zp(x_min: float, x_max: float, n_bits: int = 8):
    """
    Signed asymmetric per-tensor quantization.
    Maps [x_min, x_max] -> [qmin, qmax] = [-2^(n-1), 2^(n-1) - 1].

    scale       = (x_max - x_min) / (qmax - qmin)
    zero_point  = round(qmin - x_min / scale),  clamped to [qmin, qmax]
    """
    qmin = -(2 ** (n_bits - 1))           # -128 for INT8
    qmax =  2 ** (n_bits - 1) - 1         #  127 for INT8

    scale = (x_max - x_min) / (qmax - qmin)
    scale = max(scale, 1e-8)
    zero_point = int(round(qmin - x_min / scale))
    zero_point = int(np.clip(zero_point, qmin, qmax))
    return float(scale), zero_point


def quantize_activation_signed(x: torch.Tensor, scale: float,
                               zero_point: int, n_bits: int = 8):
    qmin = -(2 ** (n_bits - 1))
    qmax =  2 ** (n_bits - 1) - 1
    x_q = (x / scale + zero_point).round().clamp(qmin, qmax).to(torch.int32)
    return x_q


# ====================================================================================
#                         Calibration  (unchanged)
# ====================================================================================

class ActivationCollector:
    def __init__(self, model: nn.Module, layer_names: list):
        self.stats = {n: {'min': [], 'max': []} for n in layer_names}
        self.hooks = []
        self._attach(model, layer_names)

    def _attach(self, model, layer_names):
        name_to_mod = dict(model.named_modules())
        for name in layer_names:
            mod = name_to_mod[name]
            self.hooks.append(mod.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(module, input, output):
            t = output.detach().float()
            self.stats[name]['min'].append(t.min().item())
            self.stats[name]['max'].append(t.max().item())
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()

    def summary(self, percentile: float = 99.99):
        result = {}
        for name, s in self.stats.items():
            result[name] = (
                float(np.percentile(s['min'], 100 - percentile)),
                float(np.percentile(s['max'], percentile))
            )
        return result


def run_calibration(model, calib_loader, layer_names, device, percentile):
    model.eval()
    collector = ActivationCollector(model, layer_names)
    with torch.no_grad():
        for data, _ in tqdm(calib_loader, desc='Calibration'):
            model(data.to(device))
    collector.remove()
    return collector.summary(percentile)


def build_calib_loader(cfg, n_calib, batch_size):
    mel = MelSpectrogram()
    ds  = SpeechCommandDataset(cfg['data_dir'])
    _, val_files, _ = ds.get_file_list()
    random.shuffle(val_files)
    calib_files = val_files[:n_calib]
    calib_ds    = SpeechDataset(calib_files, mel, augment=False)
    return DataLoader(calib_ds, batch_size=batch_size,
                      shuffle=False, num_workers=4, pin_memory=True)


def load_pruned_model(ckpt_path, cfg, device):
    original = CNNTradFpool3(num_classes=cfg['num_classes'],
                              dropout_rate=cfg['dropout_rate'])
    pruned = prune_cnntrad(original,
                           prune_ratio_conv1=cfg['prune_ratio_conv1'],
                           prune_ratio_conv2=cfg['prune_ratio_conv2'])
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd   = ckpt.get('model_state_dict', ckpt)
    pruned.load_state_dict(sd)
    pruned.to(device).eval()
    print(f"Loaded pruned model from {ckpt_path}")
    return pruned


# ====================================================================================
#                         Main PTQ
# ====================================================================================

def ptq(model, calib_loader, device,
        n_bits=8, percentile=99.99, output_npz='quantized_params.npz'):
    print("=== model module names ===")
    for n, m in model.named_modules():
        print(f"  {n:30s}  {type(m).__name__}")
    print("==========================")
    layer_names = ['relu1', 'relu2', 'linear', 'relu3', 'classifier']

    print("\n[Step 1] Running calibration ...")
    act_ranges = run_calibration(model, calib_loader, layer_names, device, percentile)

    input_mins, input_maxs = [], []
    model.eval()
    with torch.no_grad():
        for data, _ in calib_loader:
            input_mins.append(data.min().item())
            input_maxs.append(data.max().item())
    act_ranges['input'] = (
        float(np.percentile(input_mins, 100 - percentile)),
        float(np.percentile(input_maxs, percentile))
    )

    print("\nActivation ranges (after percentile clipping):")
    for k, (mn, mx) in act_ranges.items():
        print(f"  {k:15s}: min={mn:.4f}  max={mx:.4f}")

    print("\n[Step 2] Computing activation scale / zero_point (SIGNED, Z_A free) ...")
    act_quant = {}
    for name, (mn, mx) in act_ranges.items():
        s, zp = compute_signed_asymmetric_scale_zp(mn, mx, n_bits)
        act_quant[name] = {'scale': s, 'zero_point': zp}
        print(f"  {name:15s}: scale={s:.6f}  zero_point={zp}")

    print("\n[Step 3] Quantizing weights (UNSIGNED, Z_W = 0) ...")

    layers_info = {
        'conv1':      (model.conv1,      'input'),
        'conv2':      (model.conv2,      'relu1'),
        'linear':     (model.linear,     'relu2'),
        'dnn':        (model.dnn,        'linear'),
        'classifier': (model.classifier, 'relu3'),
    }

    weight_quant = {}
    for lname, (layer, in_act_name) in layers_info.items():
        w = layer.weight.data.float()
        n_neg = (w < 0).sum().item()
        n_tot = w.numel()
        if n_neg > 0:
            print(f"  [warn] {lname}: {n_neg}/{n_tot} ({100*n_neg/n_tot:.1f}%) "
                  f"negative weights will be clamped to 0 (Z_W = 0 unsigned).")

        w_int, scale_w = quantize_weight_unsigned(w, n_bits)

        scale_act_in = act_quant[in_act_name]['scale']
        b_int, scale_b = None, None
        if layer.bias is not None:
            b_int, scale_b = quantize_bias(layer.bias.data.float(),
                                           scale_w, scale_act_in, n_bits=32)

        # Pre-compute per-output-channel weight column-sum, used to fold Z_A into K.
        # Shape: (out,)
        if w_int.dim() == 4:
            w_colsum = w_int.sum(dim=(1, 2, 3)).to(torch.int64)
        else:
            w_colsum = w_int.sum(dim=1).to(torch.int64)

        # K^(k) = b_int^(k) - Z_A * sum_j q_W^(j,k)
        z_a = act_quant[in_act_name]['zero_point']
        if b_int is not None:
            K = b_int.to(torch.int64) - z_a * w_colsum
        else:
            K = -z_a * w_colsum

        weight_quant[lname] = {
            'w_int':    w_int,
            'scale_w':  scale_w,
            'b_int':    b_int,
            'scale_b':  scale_b,
            'w_colsum': w_colsum,   # offline
            'K':        K,          # offline, INT64 in Python; fits INT32 in HW typically
        }

        print(f"  {lname:15s}: w_int range=[{w_int.min().item()}, {w_int.max().item()}]  "
              f"scale_w mean={scale_w.mean().item():.6f}  "
              f"K range=[{K.min().item()}, {K.max().item()}]")

    print(f"\n[Step 4] Saving to {output_npz} ...")
    save_dict = {}
    for name, q in act_quant.items():
        save_dict[f'act_{name}_scale']      = np.array(q['scale'],      dtype=np.float32)
        save_dict[f'act_{name}_zero_point'] = np.array(q['zero_point'], dtype=np.int32)

    for lname, q in weight_quant.items():
        save_dict[f'{lname}_w_int']    = q['w_int'].cpu().numpy().astype(np.int32)
        save_dict[f'{lname}_scale_w']  = q['scale_w'].cpu().numpy().astype(np.float32)
        save_dict[f'{lname}_w_colsum'] = q['w_colsum'].cpu().numpy().astype(np.int64)
        save_dict[f'{lname}_K']        = q['K'].cpu().numpy().astype(np.int64)
        if q['b_int'] is not None:
            save_dict[f'{lname}_b_int']   = q['b_int'].cpu().numpy()
            save_dict[f'{lname}_scale_b'] = q['scale_b'].cpu().numpy().astype(np.float32)

    np.savez(output_npz, **save_dict)
    print(f"Saved {len(save_dict)} arrays to {output_npz}")

    pth_path = output_npz.replace('.npz', '.pth')
    torch.save({'act_quant': act_quant,
                'weight_quant': weight_quant,
                'n_bits': n_bits},
               pth_path)
    print(f"Saved quantized model to {pth_path}")
    return act_quant, weight_quant


# ====================================================================================
#       Fake-quant forward (float-simulate signed-act + unsigned-weight)
# ====================================================================================

def fake_quant_forward(model, act_quant, weight_quant, data, n_bits=8):
    device = data.device
    qmin_a = -(2 ** (n_bits - 1))
    qmax_a =  2 ** (n_bits - 1) - 1

    def dequant_w(lname):
        q = weight_quant[lname]
        w_fp = q['w_int'].float().to(device)        # q_W in [0, qmax_w]
        scale_w = q['scale_w'].to(device)
        if w_fp.dim() == 4:
            w_fp = w_fp * scale_w.view(-1, 1, 1, 1)
        else:
            w_fp = w_fp * scale_w.view(-1, 1)
        return w_fp                                  # r_W = S_W * q_W (>= 0)

    def dequant_b(lname, scale_act_in):
        q = weight_quant[lname]
        if q['b_int'] is None:
            return None
        return q['b_int'].float().to(device) * q['scale_w'].to(device) * scale_act_in

    def quant_deq_act(x, name):
        s  = act_quant[name]['scale']
        zp = act_quant[name]['zero_point']
        x_q = (x / s + zp).round().clamp(qmin_a, qmax_a)
        return (x_q - zp) * s

    x = data.float()
    x = quant_deq_act(x, 'input')

    w = dequant_w('conv1'); b = dequant_b('conv1', act_quant['input']['scale'])
    x = nn.functional.conv2d(x, w, b, stride=model.conv1.stride, padding=model.conv1.padding)
    x = nn.functional.relu(x)
    x = quant_deq_act(x, 'relu1')

    w = dequant_w('conv2'); b = dequant_b('conv2', act_quant['relu1']['scale'])
    x = nn.functional.conv2d(x, w, b, stride=model.conv2.stride, padding=model.conv2.padding)
    x = nn.functional.relu(x)
    x = quant_deq_act(x, 'relu2')

    x = x.view(x.size(0), -1)

    w = dequant_w('linear'); b = dequant_b('linear', act_quant['relu2']['scale'])
    x = nn.functional.linear(x, w, b)
    x = quant_deq_act(x, 'linear')

    w = dequant_w('dnn'); b = dequant_b('dnn', act_quant['linear']['scale'])
    x = nn.functional.linear(x, w, b)
    x = nn.functional.relu(x)
    x = quant_deq_act(x, 'relu3')

    w = dequant_w('classifier'); b = dequant_b('classifier', act_quant['relu3']['scale'])
    x = nn.functional.linear(x, w, b)
    return x


def evaluate_fake_quant(model, act_quant, weight_quant, test_loader, device, n_bits=8):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Fake-quant eval'):
            data, target = data.to(device), target.to(device)
            output = fake_quant_forward(model, act_quant, weight_quant, data, n_bits)
            correct += output.argmax(1).eq(target).sum().item()
            total   += target.size(0)
    acc = 100.0 * correct / total
    print(f"\nFake-quant INT{n_bits} Accuracy: {acc:.2f}%")
    return acc


# ====================================================================================
#                               Entry point
# ====================================================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    cfg = CFG
    model = load_pruned_model(PRUNED_CKPT, cfg, device)
    calib_loader = build_calib_loader(cfg, N_CALIB, BATCH_SIZE)

    act_quant, weight_quant = ptq(model, calib_loader, device,
                                  n_bits=NUM_BITS, percentile=PERCENTILE,
                                  output_npz=OUTPUT_NPZ)

    print("\n[Step 5] Evaluating fake-quant accuracy ...")
    _, val_files, _ = SpeechCommandDataset(cfg['data_dir']).get_file_list()
    mel = MelSpectrogram()
    val_loader = DataLoader(SpeechDataset(val_files, mel, augment=False),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    evaluate_fake_quant(model, act_quant, weight_quant, val_loader, device, NUM_BITS)