# Usage of KWS Model Code

## Requirements

```
python = 3.10
torch = 2.6.0+cu124
torchaudio = 2.6.0+cu124
torchvision = 0.21.0+cu124
tqdm = 4.67.3
matplotlib = 3.10.8
numpy = 2.2.6
```

## Data Preparation

Go to https://www.kaggle.com/datasets/neehakurelli/google-speech-commands or other similar websites to download .zip file of the dataset. Make sure to unzip it such that the `/archive` folder is in the same directory with all python files. 

## Usage

### Training FP Model

```
python train.py
```

Model weight will be saved as `best_cnn_trad_no_pool_model.pth`. 

### Channel Pruning

```
python train_pruned.py
```
Model weight will be saved as `pruned_cosinelr_75.pth`. 

### PTQ

```
python ptq.py
```

Model weight will be saved as `quantized_params_75.pth`, and the parameters which can be fed into hardware will be saved as a seperate `quantized_params_75.npz` file. 

### Comparison of Three models

```
python test_all.py
```

This file will test the three models with the same test set and print accuracies. 

### Note

Make sure all python files are in the same directory since they may import each other. 
