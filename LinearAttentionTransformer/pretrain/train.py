from linear_attention_transformer import LinearAttentionTransformerLM
from linear_attention_transformer.autoregressive_wrapper import AutoregressiveWrapper
from product_key_memory import fetch_optimizer_parameters

import random
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import glob
from torch.cuda.amp import autocast, GradScaler
import os
# constants

from torchvision import transforms
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using {} device: {}".format(device, torch.cuda.current_device()))
else:
    device = torch.device("cpu")
    print("Using {}".format(device))
    
torch.cuda.empty_cache()
print("out memory")


NUM_BATCHES = int(10)
# constants
BATCH_SIZE = 64
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-6
VALIDATE_EVERY  = 100
SEQ_LEN = 300


def cycle(loader):
    while True:
        for data in loader:
            yield data

class SampleDatasetForCapture24(Dataset):
    def __init__(self, data_dir='', transform=None):
        if data_dir == '../train/':
            self.paths = glob.glob(f'{data_dir}*.npy')  
            print("10 sample")
        else:
            self.paths = glob.glob(f'{data_dir}*.npy')
        self.transform = transform
        self.data = np.concatenate([np.load(path, allow_pickle=True) for path in self.paths],axis=0)
    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        x = self.data[idx]
        target = self.data[idx + 1]
        if idx % 10000 == 0:
            print(idx)
        if self.transform is not None:
            x = self.transform(x).squeeze(0)

        # NumPy配列の場合
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(dtype=torch.float16)
        else:
            x = x.clone().detach().to(dtype=torch.float16)
            
        # NumPy配列の場合 (target)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).to(dtype=torch.float16)
        else:
            target = target.clone().detach().to(dtype=torch.float16)

        return x, target
        
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0., std=1., inplace=True),
])

train_dataset = SampleDatasetForCapture24(data_dir='../train/')
val_dataset = SampleDatasetForCapture24(data_dir='../test/')
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

model = LinearAttentionTransformerLM(
    dim = 768,
    depth = 8,
    max_seq_len = SEQ_LEN,
    heads = 12,
    causal = True
)

# model = AutoregressiveWrapper(model)
model.to(device)
print(model)



del train_dataset, val_dataset
print("delete old dataset")

optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

criterion = nn.MSELoss()


loss_history_val = []
loss_history_train = []

best_valid_loss = float('inf')
best_model_path = 'path/pretrained_1/4_pretrain_model.pth'  # モデルを保存するファイル名
os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

scaler = GradScaler()

current_step = 0
VALIDATE_EVERY_STEPS = 1000  


for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()
    total_loss = 0.0
    total_val_loss = 0.0
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        x, y = next(train_loader)  
        
        x, y = x.cuda(), y.cuda()
        
        with autocast(dtype=torch.float16): 
            output_x = model(x)
            output_y = model(y)
            loss = criterion(output_x, output_y)
            loss.backward()
            total_loss += loss.item()
    
    # 勾配クリッピング
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    avg_training_loss = total_loss / GRADIENT_ACCUMULATE_EVERY
    loss_history_train.append(avg_training_loss)
    print(f'Training loss: {avg_training_loss}')

    if i % VALIDATE_EVERY == 0:
        model.eval()
        num_val_batches = 0
        with torch.no_grad():
            x_val, y_val = next(val_loader) 
            x_val, y_val = x_val.cuda(), y_val.cuda()
            
            with autocast(dtype=torch.float16): 
                output_x_val = model(x_val)
                output_y_val = model(y_val)
                val_loss = criterion(x_val, y_val)
                total_val_loss += val_loss.item()
                num_val_batches += 1
                print(f'Validation loss: {val_loss.item()}')
        print(num_val_batches)    
        avg_val_loss = total_val_loss / num_val_batches
        loss_history_val.append(avg_val_loss)               
    
        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation loss: {best_valid_loss}")
