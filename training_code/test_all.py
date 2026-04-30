import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training_code.model import CNNTradFpool3
from training_code.pruning import prune_cnntrad
from training_code.ptq import fake_quant_forward
from training_code.train_pruned import (
    SpeechCommandDataset, SpeechDataset, MelSpectrogram, CFG
)

# ====================================================================================
#                               Config
# ====================================================================================

ORIGINAL_CKPT  = 'best_cnn_trad_no_pool_model.pth'
PRUNED_CKPT    = 'pruned_cosinelr_75.pth'
QUANTIZED_CKPT = 'quantized_params_75.pth'

BATCH_SIZE  = 100
NUM_WORKERS = 4
NUM_BITS    = 8

# ====================================================================================
#                               DataLoader
# ====================================================================================

def build_test_loader(cfg):
    mel = MelSpectrogram()
    ds  = SpeechCommandDataset(cfg['data_dir'])
    _, _, test_files = ds.get_file_list()
    test_ds = SpeechDataset(test_files, mel, augment=False)
    return DataLoader(test_ds, batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# ====================================================================================
#                               Model Loaders
# ====================================================================================

def load_original(ckpt_path, cfg, device):
    model = CNNTradFpool3(num_classes=cfg['num_classes'],
                          dropout_rate=cfg['dropout_rate'])
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd   = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(sd)
    return model.to(device).eval()


def load_pruned(ckpt_path, cfg, device):
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
    return pruned.to(device).eval()


def load_quantized(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    act_quant    = ckpt['act_quant']
    weight_quant = ckpt['weight_quant']
    n_bits       = ckpt.get('n_bits', NUM_BITS)
    return act_quant, weight_quant, n_bits

# ====================================================================================
#                               Evaluation
# ====================================================================================

def evaluate(model, test_loader, device, desc='Eval'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=desc, leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            correct += output.argmax(1).eq(target).sum().item()
            total   += target.size(0)
    return 100.0 * correct / total


def evaluate_quantized(pruned_model, act_quant, weight_quant,
                        test_loader, device, n_bits=8):
    pruned_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='INT8 Eval', leave=False):
            data, target = data.to(device), target.to(device)
            output = fake_quant_forward(
                pruned_model, act_quant, weight_quant, data, n_bits)
            correct += output.argmax(1).eq(target).sum().item()
            total   += target.size(0)
    return 100.0 * correct / total

# ====================================================================================
#                               Main
# ====================================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    cfg = CFG

    # ── Build test loader ──────────────────────────────────────────────
    print("Loading test set ...")
    test_loader = build_test_loader(cfg)

    results = {}

    # ── 1. Original ───────────────────────────────────────────────────
    print("\n[1/3] Original model")
    model_orig = load_original(ORIGINAL_CKPT, cfg, device)
    results['Original'] = evaluate(model_orig, test_loader, device, 'Original')

    # ── 2. Pruned ─────────────────────────────────────────────────────
    print("\n[2/3] Pruned model")
    model_pruned = load_pruned(PRUNED_CKPT, cfg, device)
    results['Pruned'] = evaluate(model_pruned, test_loader, device, 'Pruned')

    # ── 3. INT8 PTQ ───────────────────────────────────────────────────
    print("\n[3/3] INT8 quantized model")
    act_quant, weight_quant, n_bits = load_quantized(QUANTIZED_CKPT, device)
    results['INT8 PTQ'] = evaluate_quantized(
        model_pruned, act_quant, weight_quant, test_loader, device, n_bits)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 45)
    print(f"{'Model':<15} {'Test Accuracy':>15}")
    print("-" * 45)
    baseline = results['Original']
    for name, acc in results.items():
        drop = acc - baseline
        drop_str = f"({drop:+.2f}%)" if name != 'Original' else ""
        print(f"{name:<15} {acc:>10.2f}%  {drop_str}")
    print("=" * 45)


if __name__ == '__main__':
    main()