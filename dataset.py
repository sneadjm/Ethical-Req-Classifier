from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset
import torch

import pandas as pd
import numpy as np
import os

import yaml


def load_training_dataset(embedder, data_pth='data_config/data.yaml', batch_size=32):
    relevant_cols = ['input', 'label']

    full_train = pd.DataFrame(columns=relevant_cols)
    full_val = pd.DataFrame(columns=relevant_cols)

    with open(data_pth, 'r') as fle:
        data_config = yaml.safe_load(fle)
    for pth in data_config['train']:
        train_csv = pd.read_csv(os.path.join(data_config['parent_dir'], pth), header=0)
        if 'input' in train_csv.columns:
            reverse_map = {0: 1, 1: 0}
            train_csv['corrected_label'] = train_csv['label'].map(reverse_map).fillna(train_csv['label'])
            train_csv.drop('label', axis=1, inplace=True)
            train_csv.rename(columns={'corrected_label': 'label'}, inplace=True)
        if 'scenario' in train_csv.columns:
            train_csv.rename(columns={'scenario': 'input'}, inplace=True)
        train_csv = train_csv[relevant_cols]
        full_train = pd.concat([full_train, train_csv], ignore_index=True)

    for pth in data_config['val']:
        val_csv = pd.read_csv(os.path.join(data_config['parent_dir'], pth), header=0)
        if 'input' in val_csv.columns:
            reverse_map = {0: 1, 1:0}
            val_csv['corrected_label'] = val_csv['label'].map(reverse_map).fillna(val_csv['label'])
            val_csv.drop('label', axis=1, inplace=True)
            val_csv.rename(columns={'corrected_label': 'label'}, inplace=True)
        if 'scenario' in val_csv.columns:
            val_csv.rename(columns={'scenario': 'input'}, inplace=True)
        val_csv = val_csv[relevant_cols]
        full_val = pd.concat([full_train, val_csv], ignore_index=True)


    X_sents_train = full_train['input'].tolist()
    X_sents_val = full_val['input'].tolist()

    print('Encoding sentence inputs, this may take a while.')

    X_embeds_train = embedder.encode(X_sents_train)
    X_embeds_val = embedder.encode(X_sents_val)

    inputs_train = torch.from_numpy(X_embeds_train.astype(np.float32))
    inputs_val = torch.from_numpy(X_embeds_val.astype(np.float32))


    y_train = full_train['label'].tolist()
    y_val = full_val['label'].tolist()

    targets_train = torch.tensor(y_train)
    targets_val = torch.tensor(y_val)

    train_ds = TensorDataset(inputs_train, targets_train)
    val_ds = TensorDataset(inputs_val, targets_val)
    train_dl = DataLoader(train_ds, batch_size, shuffle = True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=True)

    final_val_dl = DataLoader(val_ds, batch_size=len(val_ds), shuffle=True)

    return train_dl, val_dl, final_val_dl