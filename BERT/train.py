import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from product_key_memory import fetch_optimizer_parameters

from transformers.modeling_outputs import SequenceClassifierOutput
import tqdm
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import glob
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
#import utils
from bert import BertModelCustom2
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, BertConfig
from transformers import BertConfig, TrainingArguments, Trainer
# For reproducibility
np.random.seed(42)
torch.manual_seed(42) # 乱数生成シード
cudnn.benchmark = True

# Grab a GPU if there is one
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using {} device: {}".format(device, torch.cuda.current_device()))
else:
    device = torch.device("cpu")
    print("Using {}".format(device))
    

NUM_BATCHES = 50
BATCH_SIZE = 2048
GRADIENT_ACCUMULATE_EVERY = 1
LEARNING_RATE = 1e-5
VALIDATE_EVERY  = 1
SEQ_LEN = 300

class SampleDatasetForCapture24(Dataset):
    def __init__(self, data_dir='', transform=None):
        self.paths = glob.glob(f'{data_dir}*.npy')
        self.transform = transform
        self.data = np.concatenate([np.load(path, allow_pickle=True) for path in self.paths],axis=0)
    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        x = self.data[idx]
        target = self.data[idx + 1]
        if idx % 100000 == 0:
            print(idx)
        if self.transform is not None:
            x = self.transform(x).squeeze(0)
        return torch.tensor(x, dtype=torch.float16), torch.tensor(target, dtype=torch.float16)

    

train_dataset = SampleDatasetForCapture24(data_dir='../train/')
val_dataset = SampleDatasetForCapture24(data_dir='../test/')

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE)

del train_dataset, val_dataset
print("delete old dataset")    
    

# class RoBertaModelCustom2(nn.Module):
#     def __init__(self, config, num_sensor_dims=300):
#         super().__init__()
#         self.roberta = RobertaModel(config)
#         self.roberta.embeddings.word_embeddings = nn.Linear(num_sensor_dims, config.hidden_size)  # センサデータ対応
#         self.output_layer = nn.Linear(config.hidden_size, num_sensor_dims)

#     def forward(self, input_ids):
#         # 入力データの形状を確認

#         # 埋め込み層の処理
#         embeds = self.roberta.embeddings.word_embeddings(input_ids)

#         # RoBERTa モデルへの入力
#         outputs = self.roberta(inputs_embeds=embeds)

#         predictions = self.output_layer(outputs.last_hidden_state)

#         return predictions


config = BertConfig()
model = BertModelCustom2(config, num_sensor_dims=300)
model.to(device)


parameters = fetch_optimizer_parameters(model)
optim = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=0.01)
criterion = nn.MSELoss()  

loss_history_val = []
loss_history_train = []
best_valid_loss = float('inf')
best_model_path = 'path/bert_300.pth'  # モデルを保存するファイル名
scaler = GradScaler()

# training

for i in tqdm(range(NUM_BATCHES)):
    model.train()
    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        for x, target in train_loader:
            x, target = x.to(device), target.to(device)
            with autocast(dtype=torch.float16):  # ここでfp16で計算することができる
                outputs = model(x)
                outputs_target = target
                loss = criterion(outputs, outputs_target)
                print(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            
    print(f'training loss: {loss.item()}')
    loss_history_train.append(loss.item())
    
    

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            for x_val, target_val in val_loader:
                x_val, target_val = x_val.to(device), target_val.to(device)
                
                with autocast(dtype=torch.float16):  # ここでfp16で計算することができる
                    outputs_val = model(x_val)
                    outputs_target_val = target_val
                    val_loss = criterion(outputs_val, outputs_target_val)
                    
            print(f'validation loss: {val_loss.item()}')
            loss_history_val.append(val_loss.item())
    if val_loss.item() < best_valid_loss:
        best_valid_loss = val_loss.item()
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved at epoch {i + 1} with valid loss: {best_valid_loss}")
        
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

plt.close('all')
fig, ax = plt.subplots()
ax.plot(loss_history_train, color='C0', label='train loss')
ax.plot(loss_history_val, color='C1', label='valid loss')

ax.set_ylabel('loss (CE)')
ax.set_xlabel('epoch')

# 二軸グラフの追加
ax2 = ax.twinx()
ax2.grid(True)

# 凡例の表示
fig.legend(loc='upper right')

# グラフの表示
plt.show()

# 必要に応じて保存
plt.savefig("pretrain_bert_300.png")
print("グラフを保存しました: pretrain_bert_300.png")