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
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, BertConfig
from transformers import BertConfig, TrainingArguments, Trainer
from torch.cuda.amp import autocast, GradScaler

class BertModelCustom2(nn.Module):
    def __init__(self, config, num_sensor_dims=3):
        super().__init__()
        self.bert = BertModel(config)
        self.bert.embeddings.word_embeddings = nn.Linear(num_sensor_dims, config.hidden_size)  # センサデータ対応
        self.output_layer = nn.Linear(config.hidden_size, num_sensor_dims)

    def forward(self, input_ids):
        # 入力データの形状を確認

        # 埋め込み層の処理
        embeds = self.bert.embeddings.word_embeddings(input_ids)

        # RoBERTa モデルへの入力
        outputs = self.bert(inputs_embeds=embeds)

        predictions = self.output_layer(outputs.last_hidden_state)

        return predictions
