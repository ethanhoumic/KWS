import torch
import torch.nn as nn
import random
import torchaudio
from pathlib import Path
import torchaudio.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import CNNTradFpool3

# ====================================================================================
#                                sample prepare
# ====================================================================================

class SpeechCommandDataset:
    def __init__(self, data_dir = './archive'):
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
        self.word_to_index = {}
        for i, word in enumerate(self.wanted_words):
            self.word_to_index[word] = i
        self.word_to_index['_silence_'] = 10
        self.word_to_index['_unknown_'] = 11
        
    def get_file_list(self):
        validation_list = set()
        test_list = set()
        
        val_file = self.data_dir/'validation_list.txt'
        test_file = self.data_dir/'testing_list.txt'
        
        if val_file.exists():
            with open(val_file, 'r') as f:
                validation_list = set(line.strip() for line in f)
        if test_file.exists():
            with open(test_file, 'r') as f:
                test_list = set(line.strip() for line in f)
                
        train_files = []
        test_files = []
        val_files = []
        
        for word in self.all_words:
            word_dir = self.data_dir/word
            if not word_dir.exists():
                continue
            for wav_file in word_dir.glob('*.wav'):
                relative_path = f"{word}/{wav_file.name}"
                if word in self.wanted_words:
                    label = self.word_to_index[word]
                else:
                    label = self.word_to_index['_unknown_']
                file_info = {
                    'path': str(wav_file),
                    'label': label,
                    'word': word
                }
                if relative_path in validation_list:
                    val_files.append(file_info)
                elif relative_path in test_list:
                    test_files.append(file_info)
                else:
                    train_files.append(file_info)
        
        silence_files = self._generate_silence_samples()
        
        random.shuffle(silence_files)
        n_silence = len(silence_files)
        n_val = int(n_silence * 0.1)
        n_test = int(n_silence * 0.1)
        train_files.extend(silence_files[n_val + n_test:])
        test_files.extend(silence_files[n_val:n_val + n_test])
        val_files.extend(silence_files[:n_val])
        
        print(f"Training samples: {len(train_files)}")
        print(f"Validation samples: {len(val_files)}")
        print(f"Testing samples: {len(test_files)}")
        
        return train_files, val_files, test_files
    
    def _generate_silence_samples(self, num_samples=3000):
        background_dir = self.data_dir/'_background_noise_'
        silence_files = []
        if not background_dir.exists():
            return silence_files
        for file in background_dir.glob('*.wav'):
            silence_files.append({
                'path': str(file),
                'label': self.word_to_index['_silence_'],
                'word': '_silence_'
            })
        return silence_files[:num_samples]

# ====================================================================================
#                                    Mel
# ====================================================================================

class MelSpectrogram:
    """
    Generating mel spectrogram features
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=480,
            hop_length=160,   # 16000/160 = 100 frames
            n_mels=40,
            f_min=20,
            f_max=8000
        )
        self.resample = T.Resample(orig_freq=sample_rate, new_freq=sample_rate)
        
    def __call__(self, waveform, sample_rate):
        """
        Args:
            waveform: (channels, samples)
            sample_rate: sample rate
        
        Returns:
            mel: (n_mel, time_steps)
        """
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            waveform = self.resample(waveform)
        target_length = self.sample_rate
        if waveform.shape[1] < target_length:
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        mel = self.mel_transform(waveform)
        return mel

class SpeechDataset:
    def __init__(self, file_list, mel_transform, augment=False):
        self.file_list = file_list
        self.mel_transform = mel_transform
        self.augment = augment
        
        self.background_noise = []
        if augment:
            self._load_background_noise()
        
    def _load_background_noise(self):
        background_dir = Path('./archive/_background_noise_')
        if not background_dir.exists():
            return
        for wav_file in background_dir.glob('*.wav'):
            waveform, sample_rate = torchaudio.load(str(wav_file))
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            self.background_noise.append(waveform)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, key):
        file_info = self.file_list[key]
        waveform, sample_rate = torchaudio.load(file_info['path'])
        if self.augment:
            waveform = self._augment(waveform, sample_rate)
        mel = self.mel_transform(waveform, sample_rate)
        mel = torch.log(mel + 1e-9)
        mel = mel.permute(0, 2, 1)
        label = file_info['label']
        return mel, label
    
    def _augment(self, waveform, sample_rate):
        # random 100 ms shift
        if random.random() < 0.8:
            shift_amt = random.randint(-100, 100)
            shift_sample = int(shift_amt * sample_rate / 1000)
            if shift_sample > 0:
                waveform = torch.nn.functional.pad(waveform[:, shift_sample:], (shift_sample, 0))
            elif shift_sample < 0:
                waveform = torch.nn.functional.pad(waveform[:, :shift_sample], (0, -shift_sample))

        # background noise
        if random.random() < 0.8 and len(self.background_noise) > 0:
            noise = random.choice(self.background_noise)
            target_len = waveform.shape[1]
            
            if noise.shape[1] >= target_len:
                start = random.randint(0, noise.shape[1] - target_len)
                noise_clip = noise[:, start:start + target_len]
            else:
                repeats = target_len // noise.shape[1] + 1
                noise_clip = noise.repeat(1, repeats)[:, :target_len]
            noise_level = random.uniform(0.0, 0.1)
            waveform = waveform + noise_clip * noise_level  
            
        return waveform
    
# ====================================================================================
#                                   training
# ====================================================================================

class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader= val_loader
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=5e-4)
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10000,
            gamma = 0.2
        )
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        loss = 0
        total = 0
        correct = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # back propagation
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data,  target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def train(self, num_epochs=20, save_path='best_model.pth', patience=5):
        best_val_acc = 0
        patience_cnt = 0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 60)
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc=  self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc
                }, save_path)
                
                print(f'Best model saved! Val Acc: {val_acc:.2f}%')
            else:
                patience_cnt += 1
                print(f'No improvement. Patience: {patience_cnt}/{patience}')
                if patience_cnt >= patience:
                    print(f'Early stopping triggered at epoch {epoch+1}')
                    break
                
        print(f'\nTraining completed! Best Val Acc: {best_val_acc:.2f}%')
        
        return self.history

    def plot_history(self, figure_path='training_history.png'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('Training and Validation Loss')
        
        # Accuracy
        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.set_title('Training and Validation Accuracy')
        
        plt.tight_layout()
        plt.savefig(figure_path)
        plt.show()

# ====================================================================================
#                                   testing
# ====================================================================================

class Tester:
    def __init__(self, model, test_loader, device='cuda'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device

    def test(self):
        self.model.eval()
        total = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Testing"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
        accuracy = 100. * correct / total
        return accuracy   

# ====================================================================================
#                                main function
# ====================================================================================

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device:{device}")
    
    num_classes = 12
    
    model = CNNTradFpool3(num_classes=num_classes, dropout_rate=0.1)
    print(f'\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters')
    
    mel_transform = MelSpectrogram(sample_rate=16000)
    
    dataset_manager = SpeechCommandDataset()
    train_files, val_files, test_files = dataset_manager.get_file_list()
    
    train_dataset = SpeechDataset(train_files, mel_transform, augment=True)
    val_dataset = SpeechDataset(val_files, mel_transform, augment=False)
    test_dataset = SpeechDataset(test_files, mel_transform, augment=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=100,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    trainer = Trainer(model, train_loader, val_loader, device=device)
    history = trainer.train(num_epochs=50, save_path='best_cnn_trad_sigmoid.pth', patience=25)
    tester = Tester(model, test_loader, device=device)
    
    trainer.plot_history(figure_path='training_history_trad_sigmoid.png')
    print('\n' + '='*60)
    print('Evaluating on test set...')
    test_acc = tester.test()
    print(f'Test Acc: {test_acc:.2f}%')
    print('='*60)

if __name__ == '__main__':
    main()