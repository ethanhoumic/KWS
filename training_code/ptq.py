import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from training_code.model import CNNTradFpool3
from training_code.pruning import prune_cnntrad
from training_code.train_pruned import SpeechCommandDataset, SpeechDataset, MelSpectrogram, CFG
from torch.utils.data import DataLoader
import random

# ====================================================================================
#                               Config
# ====================================================================================

PRUNED_CKPT       = 'pruned_cosinelr_75.pth'
OUTPUT_NPZ        = 'quantized_params_75.npz'
N_CALIB           = 1000          # calibration 樣本數
BATCH_SIZE        = 100
NUM_BITS          = 8
PERCENTILE        = 99.99         # activation range 用 percentile 避免 outlier

# ====================================================================================
#                         Weight Quantization (Symmetric, per-channel)
# ====================================================================================

def quantize_weight_symmetric(weight: torch.Tensor, n_bits: int = 8):
    """
    Per-output-channel symmetric quantization for Conv2d / Linear weights.
    weight : (out, in, kH, kW) or (out, in)
    returns:
        w_int  : INT8 quantized weight (same shape)
        scale  : (out,) per-channel scale
    """
    qmax = 2 ** (n_bits - 1) - 1          # 127 for INT8

    # max abs per output channel
    if weight.dim() == 4:
        # Conv2d: (out, in, kH, kW) → max over (in, kH, kW)
        w_max = weight.abs().amax(dim=(1, 2, 3))
    else:
        # Linear: (out, in) → max over in
        w_max = weight.abs().amax(dim=1)

    scale = (w_max / qmax).clamp(min=1e-8)          # (out,)

    # broadcast scale back to weight shape
    if weight.dim() == 4:
        scale_bc = scale.view(-1, 1, 1, 1)
    else:
        scale_bc = scale.view(-1, 1)

    w_int = (weight / scale_bc).round().clamp(-qmax - 1, qmax).to(torch.int8)
    return w_int, scale


def quantize_bias(bias: torch.Tensor, scale_w: torch.Tensor,
                  scale_act_in: float, n_bits: int = 32):
    """
    Bias quantized to INT32 with scale = scale_w * scale_act_in (per-channel).
    bias    : (out,)
    scale_w : (out,) per-channel weight scale
    scale_act_in : scalar, input activation scale of this layer
    """
    scale_bias = scale_w * scale_act_in          # (out,)
    qmax = 2 ** (n_bits - 1) - 1
    b_int = (bias / scale_bias).round().clamp(-qmax - 1, qmax).to(torch.int32)
    return b_int, scale_bias


# ====================================================================================
#                      Activation Quantization (Asymmetric, per-tensor)
# ====================================================================================

def compute_asymmetric_scale_zp(x_min: float, x_max: float, n_bits: int = 8):
    """
    Asymmetric per-tensor quantization.
    Maps [x_min, x_max] → [0, 2^n_bits - 1]  (uint representation)

    scale     = (x_max - x_min) / (2^n_bits - 1)
    zero_point= round(-x_min / scale)  clamped to [0, 2^n_bits-1]
    """
    qmin = 0
    qmax = 2 ** n_bits - 1                        # 255 for INT8

    scale = (x_max - x_min) / (qmax - qmin)
    scale = max(scale, 1e-8)
    zero_point = int(round(-x_min / scale))
    zero_point = int(np.clip(zero_point, qmin, qmax))
    return float(scale), zero_point


def quantize_activation(x: torch.Tensor, scale: float,
                         zero_point: int, n_bits: int = 8):
    qmin, qmax = 0, 2 ** n_bits - 1
    x_q = (x / scale + zero_point).round().clamp(qmin, qmax).to(torch.uint8)
    return x_q


# ====================================================================================
#                         Calibration  (collect activation stats)
# ====================================================================================

class ActivationCollector:
    """Attach forward hooks to collect per-layer activation min/max."""

    def __init__(self, model: nn.Module, layer_names: list):
        self.stats   = {n: {'min': [], 'max': []} for n in layer_names}
        self.hooks   = []
        self._attach(model, layer_names)

    def _attach(self, model, layer_names):
        name_to_mod = dict(model.named_modules())
        for name in layer_names:
            mod = name_to_mod[name]
            hook = mod.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)

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
        """
        Return {name: (x_min, x_max)} using percentile clipping
        to suppress outliers.
        """
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
            data = data.to(device)
            model(data)

    collector.remove()
    return collector.summary(percentile)


# ====================================================================================
#                         Build Calibration DataLoader
# ====================================================================================

def build_calib_loader(cfg, n_calib, batch_size):
    mel = MelSpectrogram()
    ds  = SpeechCommandDataset(cfg['data_dir'])
    _, val_files, _ = ds.get_file_list()

    random.shuffle(val_files)
    calib_files = val_files[:n_calib]
    calib_ds    = SpeechDataset(calib_files, mel, augment=False)
    return DataLoader(calib_ds, batch_size=batch_size,
                      shuffle=False, num_workers=4, pin_memory=True)


# ====================================================================================
#                         Load Pruned Model
# ====================================================================================

def load_pruned_model(ckpt_path, cfg, device):
    original = CNNTradFpool3(num_classes=cfg['num_classes'],
                              dropout_rate=cfg['dropout_rate'])
    pruned = prune_cnntrad(
        original,
        prune_ratio_conv1=cfg['prune_ratio_conv1'],
        prune_ratio_conv2=cfg['prune_ratio_conv2'],
    )
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

    layer_names = [
        'relu1',        # conv1 → relu1 : input act of conv2
        'relu2',        # conv2 → relu2 : input act of linear
        'linear',       # linear output  : input act of dnn
        'relu3',        # dnn   → relu3  : input act of classifier
        'classifier',   # final logits
    ]

    # ── 2. Calibration ────────────────────────────────────────────────
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

    print("\n[Step 2] Computing activation scale / zero_point ...")
    act_quant = {}
    for name, (mn, mx) in act_ranges.items():
        s, zp = compute_asymmetric_scale_zp(mn, mx, n_bits)
        act_quant[name] = {'scale': s, 'zero_point': zp}
        print(f"  {name:15s}: scale={s:.6f}  zero_point={zp}")

    print("\n[Step 3] Quantizing weights ...")

    layers_info = {
        'conv1':      (model.conv1,      'input'),
        'conv2':      (model.conv2,      'relu1'),
        'linear':     (model.linear,     'relu2'),
        'dnn':        (model.dnn,        'linear'),
        'classifier': (model.classifier, 'relu3'),
    }

    weight_quant = {}
    for lname, (layer, in_act_name) in layers_info.items():
        w      = layer.weight.data.float()
        w_int, scale_w = quantize_weight_symmetric(w, n_bits)

        scale_act_in = act_quant[in_act_name]['scale']
        b_int, scale_b = None, None
        if layer.bias is not None:
            b_int, scale_b = quantize_bias(layer.bias.data.float(),
                                            scale_w, scale_act_in, n_bits=32)

        weight_quant[lname] = {
            'w_int':    w_int,
            'scale_w':  scale_w,
            'b_int':    b_int,
            'scale_b':  scale_b,
        }

        print(f"  {lname:15s}: w_int range=[{w_int.min()}, {w_int.max()}]  "
              f"scale_w mean={scale_w.mean().item():.6f}")

    print(f"\n[Step 4] Saving to {output_npz} ...")
    save_dict = {}

    for name, q in act_quant.items():
        save_dict[f'act_{name}_scale']      = np.array(q['scale'],       dtype=np.float32)
        save_dict[f'act_{name}_zero_point'] = np.array(q['zero_point'],  dtype=np.int32)

    for lname, q in weight_quant.items():
        save_dict[f'{lname}_w_int']   = q['w_int'].cpu().numpy()
        save_dict[f'{lname}_scale_w'] = q['scale_w'].cpu().numpy().astype(np.float32)
        if q['b_int'] is not None:
            save_dict[f'{lname}_b_int']   = q['b_int'].cpu().numpy()
            save_dict[f'{lname}_scale_b'] = q['scale_b'].cpu().numpy().astype(np.float32)

    np.savez(output_npz, **save_dict)
    print(f"Saved {len(save_dict)} arrays to {output_npz}")
    
    pth_path = output_npz.replace('.npz', '.pth')
    torch.save({
        'act_quant':    act_quant,
        'weight_quant': {
            lname: {
                'w_int':   q['w_int'],
                'scale_w': q['scale_w'],
                'b_int':   q['b_int'],
                'scale_b': q['scale_b'],
            }
            for lname, q in weight_quant.items()
        },
        'n_bits': n_bits,
    }, pth_path)
    print(f"Saved quantized model to {pth_path}")

    return act_quant, weight_quant


# ====================================================================================
#                    Evaluate accuracy (fake-quantize, float simulate)
# ====================================================================================

def fake_quant_forward(model, act_quant, weight_quant, data, n_bits=8):
    """
    Simulate integer-quantized inference in floating point.
    Dequantize weights before each layer, quantize activations after each layer.
    """
    device = data.device
    def dequant_w(lname):
        q = weight_quant[lname]
        w_fp = q['w_int'].float().to(device)
        scale_w = q['scale_w'].to(device)
        if w_fp.dim() == 4:
            w_fp = w_fp * scale_w.view(-1, 1, 1, 1)
        else:
            w_fp = w_fp * scale_w.view(-1, 1)
        return w_fp

    def dequant_b(lname, scale_act_in):
        q = weight_quant[lname]
        if q['b_int'] is None:
            return None
        return q['b_int'].float().to(device) * q['scale_w'].to(device) * scale_act_in

    def quant_deq_act(x, name):
        s  = act_quant[name]['scale']
        zp = act_quant[name]['zero_point']
        qmin, qmax = 0, 2 ** n_bits - 1
        x_q = (x / s + zp).round().clamp(qmin, qmax)
        return (x_q - zp) * s

    x = data.float()

    # quantize input
    x = quant_deq_act(x, 'input')

    # conv1
    w = dequant_w('conv1')
    b = dequant_b('conv1', act_quant['input']['scale'])
    x = nn.functional.conv2d(x, w, b,
                              stride=model.conv1.stride,
                              padding=model.conv1.padding)
    x = nn.functional.relu(x)
    x = quant_deq_act(x, 'relu1')

    # conv2
    w = dequant_w('conv2')
    b = dequant_b('conv2', act_quant['relu1']['scale'])
    x = nn.functional.conv2d(x, w, b,
                              stride=model.conv2.stride,
                              padding=model.conv2.padding)
    x = nn.functional.relu(x)
    x = quant_deq_act(x, 'relu2')

    # flatten
    x = x.view(x.size(0), -1)

    # linear
    w = dequant_w('linear')
    b = dequant_b('linear', act_quant['relu2']['scale'])
    x = nn.functional.linear(x, w, b)
    x = quant_deq_act(x, 'linear')

    # dnn
    w = dequant_w('dnn')
    b = dequant_b('dnn', act_quant['linear']['scale'])
    x = nn.functional.linear(x, w, b)
    x = nn.functional.relu(x)
    x = quant_deq_act(x, 'relu3')

    # classifier
    w = dequant_w('classifier')
    b = dequant_b('classifier', act_quant['relu3']['scale'])
    x = nn.functional.linear(x, w, b)

    return x


def evaluate_fake_quant(model, act_quant, weight_quant,
                         test_loader, device, n_bits=8):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Fake-quant eval'):
            data   = data.to(device)
            target = target.to(device)
            output = fake_quant_forward(model, act_quant, weight_quant,
                                         data, n_bits)
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

    # 1. Load pruned model
    model = load_pruned_model(PRUNED_CKPT, cfg, device)

    # 2. Build calibration loader (val set, N_CALIB samples)
    calib_loader = build_calib_loader(cfg, N_CALIB, BATCH_SIZE)

    # 3. PTQ
    act_quant, weight_quant = ptq(
        model, calib_loader, device,
        n_bits     = NUM_BITS,
        percentile = PERCENTILE,
        output_npz = OUTPUT_NPZ,
    )

    # 4. Evaluate fake-quantized accuracy on val set
    print("\n[Step 5] Evaluating fake-quant accuracy ...")
    _, val_files, _ = SpeechCommandDataset(cfg['data_dir']).get_file_list()
    mel = MelSpectrogram()
    val_loader = DataLoader(
        SpeechDataset(val_files, mel, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    evaluate_fake_quant(model, act_quant, weight_quant,
                         val_loader, device, NUM_BITS)