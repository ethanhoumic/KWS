import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CNNTradFpool3
from pruning import prune_cnntrad, count_params, verify_output_shape

# ====================================================================================
#                               Configuration
# ====================================================================================

CFG = dict(
    data_dir        = './archive',
    pretrained_ckpt = 'best_cnn_trad_sigmoid.pth',  # set '' to skip loading
    save_path       = 'pruned_cosinelr_75_sigmoid.pth',
    figure_path     = 'training_history_pruned_cosinelr_75_sigmoid.png',

    # Pruning ratios (fraction of channels to REMOVE)
    prune_ratio_conv1 = 0.5,
    prune_ratio_conv2 = 0.75,

    # Training
    num_classes  = 12,
    dropout_rate = 0.1,
    batch_size   = 100,
    num_workers  = 4,
    num_epochs   = 50,
    patience     = 20,
    lr           = 5e-4,           # lower LR for fine-tuning
    lr_step      = 5000,
    lr_gamma     = 0.5,
)

# ====================================================================================
#                               Dataset / DataLoader
# ====================================================================================

class SpeechCommandDataset:
    def __init__(self, data_dir='./archive'):
        self.data_dir = Path(data_dir)
        self.wanted_words = [
            'yes', 'no', 'up', 'down', 'left',
            'right', 'on', 'off', 'stop', 'go'
        ]
        self.all_words = [
            'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off',
            'stop', 'go', 'backward', 'forward', 'follow', 'learn',
            'bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin',
            'sheila', 'tree', 'wow', 'zero', 'one', 'two', 'three',
            'four', 'five', 'six', 'seven', 'eight', 'nine'
        ]
        self.word_to_index = {w: i for i, w in enumerate(self.wanted_words)}
        self.word_to_index['_silence_'] = 10
        self.word_to_index['_unknown_'] = 11

    def get_file_list(self):
        validation_list, test_list = set(), set()
        for fname, s in [('validation_list.txt', validation_list),
                         ('testing_list.txt', test_list)]:
            p = self.data_dir / fname
            if p.exists():
                s.update(line.strip() for line in open(p))

        train_files, val_files, test_files = [], [], []
        for word in self.all_words:
            word_dir = self.data_dir / word
            if not word_dir.exists():
                continue
            label = self.word_to_index.get(word, self.word_to_index['_unknown_'])
            for wav in word_dir.glob('*.wav'):
                rel = f"{word}/{wav.name}"
                info = {'path': str(wav), 'label': label, 'word': word}
                if rel in validation_list:
                    val_files.append(info)
                elif rel in test_list:
                    test_files.append(info)
                else:
                    train_files.append(info)

        silence = self._generate_silence_samples()
        random.shuffle(silence)
        nv = int(len(silence) * 0.1)
        nt = int(len(silence) * 0.1)
        train_files.extend(silence[nv + nt:])
        test_files.extend(silence[nv:nv + nt])
        val_files.extend(silence[:nv])

        print(f"Train: {len(train_files):,}  Val: {len(val_files):,}  Test: {len(test_files):,}")
        return train_files, val_files, test_files

    def _generate_silence_samples(self, num_samples=3000):
        bg = self.data_dir / '_background_noise_'
        files = []
        if bg.exists():
            for f in bg.glob('*.wav'):
                files.append({'path': str(f), 'label': 10, 'word': '_silence_'})
        return files[:num_samples]


class MelSpectrogram:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.mel = T.MelSpectrogram(
            sample_rate=16000, n_fft=480, hop_length=160,
            n_mels=40, f_min=20, f_max=8000
        )
        self.resample = T.Resample(orig_freq=sample_rate, new_freq=sample_rate)

    def __call__(self, waveform, sample_rate):
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        if sample_rate != self.sample_rate:
            waveform = self.resample(waveform)
        L = self.sample_rate
        if waveform.shape[1] < L:
            waveform = nn.functional.pad(waveform, (0, L - waveform.shape[1]))
        else:
            waveform = waveform[:, :L]
        return self.mel(waveform)


class SpeechDataset:
    def __init__(self, file_list, mel_transform, augment=False):
        self.file_list = file_list
        self.mel = mel_transform
        self.augment = augment
        self.bg_noise = []
        if augment:
            self._load_bg()

    def _load_bg(self):
        bg = Path('./archive/_background_noise_')
        if not bg.exists():
            return
        for f in bg.glob('*.wav'):
            w, _ = torchaudio.load(str(f))
            if w.shape[0] > 1:
                w = w.mean(0, keepdim=True)
            self.bg_noise.append(w)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        info = self.file_list[idx]
        waveform, sr = torchaudio.load(info['path'])
        if self.augment:
            waveform = self._augment(waveform, sr)
        mel = self.mel(waveform, sr)
        mel = torch.log(mel + 1e-9).permute(0, 2, 1)   # (1, T, F)
        return mel, info['label']

    def _augment(self, waveform, sr):
        if random.random() < 0.8:
            shift = int(random.randint(-100, 100) * sr / 1000)
            if shift > 0:
                waveform = nn.functional.pad(waveform[:, shift:], (shift, 0))
            elif shift < 0:
                waveform = nn.functional.pad(waveform[:, :shift], (0, -shift))
        if random.random() < 0.8 and self.bg_noise:
            noise = random.choice(self.bg_noise)
            L = waveform.shape[1]
            if noise.shape[1] >= L:
                s = random.randint(0, noise.shape[1] - L)
                clip = noise[:, s:s + L]
            else:
                clip = noise.repeat(1, L // noise.shape[1] + 1)[:, :L]
            waveform = waveform + clip * random.uniform(0.0, 0.1)
        return waveform


def build_loaders(cfg):
    mel = MelSpectrogram()
    ds = SpeechCommandDataset(cfg['data_dir'])
    train_f, val_f, test_f = ds.get_file_list()

    kw = dict(num_workers=cfg['num_workers'], pin_memory=True)
    train_loader = DataLoader(SpeechDataset(train_f, mel, augment=True),
                              batch_size=cfg['batch_size'], shuffle=True, **kw)
    val_loader   = DataLoader(SpeechDataset(val_f,   mel, augment=False),
                              batch_size=cfg['batch_size'], shuffle=False, **kw)
    test_loader  = DataLoader(SpeechDataset(test_f,  mel, augment=False),
                              batch_size=cfg['batch_size'], shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ====================================================================================
#                               Trainer
# ====================================================================================

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, cfg):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.criterion    = nn.CrossEntropyLoss()
        self.optimizer    = optim.Adam(model.parameters(), lr=cfg['lr'])
        self.history = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])

    def _run_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        desc = 'Train' if train else 'Val'

        with ctx:
            for data, target in tqdm(loader, desc=desc, leave=False):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss   = self.criterion(output, target)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                total_loss += loss.item()
                correct    += output.argmax(1).eq(target).sum().item()
                total      += target.size(0)

        return total_loss / len(loader), 100.0 * correct / total

    def train(self, num_epochs, save_path, patience):
        total_steps = num_epochs * len(self.train_loader)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=1e-6
        )
        best_acc, wait = 0.0, 0
        for ep in range(num_epochs):
            tr_loss, tr_acc = self._run_epoch(self.train_loader, train=True)
            vl_loss, vl_acc = self._run_epoch(self.val_loader,   train=False)

            self.history['train_loss'].append(tr_loss)
            self.history['train_acc'].append(tr_acc)
            self.history['val_loss'].append(vl_loss)
            self.history['val_acc'].append(vl_acc)

            lr_now = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {ep+1:3d}/{num_epochs} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.2f}% | "
                  f"val loss {vl_loss:.4f} acc {vl_acc:.2f}% | lr {lr_now:.6f}")

            if vl_acc > best_acc:
                best_acc = vl_acc
                wait = 0
                torch.save({
                    'epoch': ep,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': vl_acc,
                }, save_path)
                print(f"  ✓ Best model saved (val acc {vl_acc:.2f}%)")
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {ep+1}")
                    break

        print(f"\nDone. Best val acc: {best_acc:.2f}%")
        return self.history

    def plot_history(self, figure_path):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.history['train_loss'], label='Train')
        ax1.plot(self.history['val_loss'],   label='Val')
        ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend()
        ax2.plot(self.history['train_acc'], label='Train')
        ax2.plot(self.history['val_acc'],   label='Val')
        ax2.set_title('Accuracy (%)'); ax2.set_xlabel('Epoch'); ax2.legend()
        plt.tight_layout()
        plt.savefig(figure_path)
        print(f"Training curve saved to {figure_path}")


# ====================================================================================
#                               Tester
# ====================================================================================

class Tester:
    def __init__(self, model, test_loader, device):
        self.model = model.to(device)
        self.loader = test_loader
        self.device = device

    def test(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in tqdm(self.loader, desc='Test'):
                data, target = data.to(self.device), target.to(self.device)
                pred = self.model(data).argmax(1)
                correct += pred.eq(target).sum().item()
                total   += target.size(0)
        acc = 100.0 * correct / total
        print(f"Test Accuracy: {acc:.2f}%")
        return acc


# ====================================================================================
#                               Main
# ====================================================================================

def main():
    cfg    = CFG
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    original = CNNTradFpool3(num_classes=cfg['num_classes'],
                             dropout_rate=cfg['dropout_rate'])
    ckpt = cfg['pretrained_ckpt']
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location='cpu')
        sd = state.get('model_state_dict', state)
        original.load_state_dict(sd)
        print(f"Loaded pretrained weights from {ckpt}")
    else:
        print("No pretrained checkpoint found — pruning from random init.")

    print(f"Original params : {count_params(original):,}")

    pruned = prune_cnntrad(
        original,
        prune_ratio_conv1=cfg['prune_ratio_conv1'],
        prune_ratio_conv2=cfg['prune_ratio_conv2'],
    )
    print(f"Pruned   params : {count_params(pruned):,}")
    ratio = count_params(original) / count_params(pruned)
    print(f"Compression     : {ratio:.2f}x")
    verify_output_shape(pruned, num_classes=cfg['num_classes'])

    train_loader, val_loader, test_loader = build_loaders(cfg)

    trainer = Trainer(pruned, train_loader, val_loader, device, cfg)
    trainer.train(
        num_epochs=cfg['num_epochs'],
        save_path =cfg['save_path'],
        patience  =cfg['patience'],
    )
    trainer.plot_history(cfg['figure_path'])

    # reload best checkpoint
    best = torch.load(cfg['save_path'], map_location='cpu')
    pruned.load_state_dict(best['model_state_dict'])
    tester = Tester(pruned, test_loader, device)
    tester.test()


if __name__ == '__main__':
    main()
