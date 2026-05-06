"""
QAT 4-bit fine-tuning for CNNTradFpool3 with Sigmoid activations.

Differences from the ReLU version:
  - Activations are nn.Sigmoid (no Conv-Activation fusion possible).
  - Skip fuse_modules step entirely.
  - Everything else (W4A4 qconfig, observer freeze, cosine LR) stays the same.

Quant scheme:
  - Weight     : per-channel symmetric INT4   q_W in [-8, 7],  Z_W = 0
  - Activation : per-tensor  asymmetric INT4  q_A in [ 0, 15], Z_A free
  - Bias       : INT32 symmetric, S_B = S_W * S_A_in
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as tq
from torch.ao.quantization import (
    QConfig, FakeQuantize,
    MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

from train import SpeechCommandDataset, SpeechDataset, MelSpectrogram
from model import CNNTradFpool3


# ====================================================================================
#                              Model (Sigmoid version)
# ====================================================================================

class CNNTradFpool3_Sigmoid(nn.Module):
    """
    Same as CNNTradFpool3 but ReLU -> Sigmoid.
    weight keys identical to ReLU version, so fp Sigmoid ckpt loads cleanly.
    """
    def __init__(self, num_classes=12, dropout_rate=0.3):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(67, 8), stride=(1, 1),
                               padding=0, bias=True)
        self.relu1 = nn.Sigmoid()                      # name kept for ckpt compat

        self.conv2 = nn.Conv2d(64, 64, kernel_size=(10, 4), stride=(1, 1),
                               padding=0, bias=True)
        self.relu2 = nn.Sigmoid()

        self.dropout = nn.Dropout(p=dropout_rate)
        self._conv_out_dim = self._get_conv_out_dim()

        self.linear     = nn.Linear(self._conv_out_dim, 32, bias=True)
        self.dnn        = nn.Linear(32, 128, bias=True)
        self.relu3      = nn.Sigmoid()
        self.classifier = nn.Linear(128, num_classes, bias=True)

        self._initialize_weight()

    def _get_conv_out_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 101, 40)
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            return x.view(1, -1).shape[1]

    def _initialize_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.dnn(x)
        x = self.relu3(x)
        x = self.dropout(x)
        return self.classifier(x)


# ====================================================================================
#                              Config
# ====================================================================================

CFG = dict(
    num_classes      = 12,
    dropout_rate     = 0.1,
    weight_bits      = 4,
    activation_bits  = 4,
    batch_size       = 100,
    qat_epochs       = 30,
    qat_lr           = 5e-5,
    qat_lr_min       = 1e-6,
    weight_decay     = 1e-5,
    float_ckpt       = 'best_cnn_trad_sigmoid.pth',     # your fp Sigmoid ckpt
    qat_save_path    = 'qat_int4_cnntrad_sigmoid.pth',
    history_png      = 'qat_history_sigmoid.png',
)


# ====================================================================================
#                              QAT Setup
# ====================================================================================

def make_qconfig(weight_bits: int = 4, act_bits: int = 4) -> QConfig:
    w_qmin = -(2 ** (weight_bits - 1))
    w_qmax =  2 ** (weight_bits - 1) - 1
    a_qmin = 0
    a_qmax = 2 ** act_bits - 1

    weight_fq = FakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=w_qmin, quant_max=w_qmax,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0, reduce_range=False,
    )
    act_fq = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=a_qmin, quant_max=a_qmax,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
    )
    return QConfig(activation=act_fq, weight=weight_fq)


def prepare_qat_model(model: nn.Module, qconfig: QConfig) -> nn.Module:
    """
    No fusion for Sigmoid version (PyTorch has no Conv-Sigmoid fused module).
    prepare_qat will still insert an activation fake-quant after each Conv/Linear,
    which is what catches the post-Sigmoid range (since the next Conv/Linear
    sees the Sigmoid output as input and quantizes it).
    """
    model.qconfig = qconfig
    model.train()
    tq.prepare_qat(model, inplace=True)
    return model


# ====================================================================================
#                              Trainer
# ====================================================================================

class QATTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda',
                 lr=5e-5, lr_min=1e-6, weight_decay=1e-5, num_epochs=30):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.num_epochs   = num_epochs

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr,
                                    weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs * len(train_loader), eta_min=lr_min
        )
        self.history = {'train_loss': [], 'train_acc': [],
                        'val_loss':   [], 'val_acc':   []}

    def _train_epoch(self):
        self.model.train()
        total, correct, total_loss = 0, 0, 0.0
        pbar = tqdm(self.train_loader, desc='QAT-Train')
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            correct    += output.argmax(1).eq(target).sum().item()
            total      += target.size(0)
            pbar.set_postfix(loss=f'{loss.item():.4f}',
                             acc =f'{100.*correct/total:.2f}%',
                             lr  =f'{self.optimizer.param_groups[0]["lr"]:.2e}')
        return total_loss / len(self.train_loader), 100.*correct/total

    def _validate(self):
        self.model.eval()
        total, correct, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='QAT-Val'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss   = self.criterion(output, target)
                total_loss += loss.item()
                correct    += output.argmax(1).eq(target).sum().item()
                total      += target.size(0)
        return total_loss / len(self.val_loader), 100.*correct/total

    def train(self, save_path):
        best_acc = 0.0
        for epoch in range(self.num_epochs):
            print(f'\n[QAT] Epoch {epoch+1}/{self.num_epochs}  '
                  f'lr={self.optimizer.param_groups[0]["lr"]:.2e}')
            print('-' * 60)

            if epoch == int(self.num_epochs * 0.7):
                self.model.apply(tq.disable_observer)
                print('[QAT] observers frozen.')

            tr_loss, tr_acc = self._train_epoch()
            va_loss, va_acc = self._validate()

            self.history['train_loss'].append(tr_loss)
            self.history['train_acc'].append(tr_acc)
            self.history['val_loss'].append(va_loss)
            self.history['val_acc'].append(va_acc)

            print(f'  train  loss={tr_loss:.4f}  acc={tr_acc:.2f}%')
            print(f'  val    loss={va_loss:.4f}  acc={va_acc:.2f}%')

            if va_acc > best_acc:
                best_acc = va_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_acc': va_acc,
                    'qconfig_bits': (CFG['weight_bits'], CFG['activation_bits']),
                }, save_path)
                print(f'  >> best QAT ckpt saved: val_acc={va_acc:.2f}%')
        print(f'\n[QAT] done. Best val_acc = {best_acc:.2f}%')
        return self.history

    def plot(self, path):
        fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
        a.plot(self.history['train_loss'], label='train')
        a.plot(self.history['val_loss'],   label='val')
        a.set_title('QAT Loss');     a.legend()
        b.plot(self.history['train_acc'],  label='train')
        b.plot(self.history['val_acc'],    label='val')
        b.set_title('QAT Accuracy'); b.legend()
        plt.tight_layout()
        plt.savefig(path)


# ====================================================================================
#                              Main
# ====================================================================================

def build_loaders(cfg):
    mel = MelSpectrogram(sample_rate=16000)
    ds  = SpeechCommandDataset()
    train_files, val_files, test_files = ds.get_file_list()
    tr  = SpeechDataset(train_files, mel, augment=True)
    va  = SpeechDataset(val_files,   mel, augment=False)
    te  = SpeechDataset(test_files,  mel, augment=False)
    return (
        DataLoader(tr, batch_size=cfg['batch_size'], shuffle=True,
                   num_workers=4, pin_memory=True),
        DataLoader(va, batch_size=cfg['batch_size'], shuffle=False,
                   num_workers=4, pin_memory=True),
        DataLoader(te, batch_size=cfg['batch_size'], shuffle=False,
                   num_workers=4, pin_memory=True),
    )


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    train_loader, val_loader, test_loader = build_loaders(CFG)

    # ---- Step 1: build sigmoid model and load fp ckpt ----
    model = CNNTradFpool3(num_classes=CFG['num_classes'],
                                  dropout_rate=CFG['dropout_rate'])
    ck = torch.load(CFG['float_ckpt'], map_location='cpu')
    sd = ck.get('model_state_dict', ck)
    model.load_state_dict(sd)
    print(f'loaded fp Sigmoid ckpt: {CFG["float_ckpt"]}  '
          f'(reported val_acc={ck.get("val_acc", "?")})')

    # ---- Step 2: prepare QAT (no fusion) ----
    qconfig = make_qconfig(CFG['weight_bits'], CFG['activation_bits'])
    print(f'QAT qconfig: W{CFG["weight_bits"]}A{CFG["activation_bits"]}')
    model = prepare_qat_model(model, qconfig)
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'QAT model params: {n_params:,}')

    # ---- Step 3: QAT fine-tune ----
    trainer = QATTrainer(model, train_loader, val_loader, device=device,
                         lr=CFG['qat_lr'], lr_min=CFG['qat_lr_min'],
                         weight_decay=CFG['weight_decay'],
                         num_epochs=CFG['qat_epochs'])
    trainer.train(save_path=CFG['qat_save_path'])
    trainer.plot(CFG['history_png'])

    # ---- Step 4: final test ----
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='QAT-Test'):
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(1)
            correct += pred.eq(target).sum().item(); total += target.size(0)
    print(f'\nFinal QAT INT{CFG["weight_bits"]}/INT{CFG["activation_bits"]} '
          f'test acc: {100.*correct/total:.2f}%')


if __name__ == '__main__':
    main()