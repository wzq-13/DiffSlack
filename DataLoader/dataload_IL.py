import os
from torch.utils.data import Dataset
import json
from collections import OrderedDict
import torch
from mmengine import fileio
import io
import numpy as np

def openjson(path):
       value  = fileio.get_text(path)
       dict = json.loads(value)
       return dict

def opendata(path):
    npz_bytes = fileio.get(path)
    buff = io.BytesIO(npz_bytes)
    npz_data = np.load(buff, allow_pickle=True)
    return npz_data


class My_Dataset_IL(Dataset):
    def __init__(self, data_dir, datafilelist):
        self.data_dir = data_dir
        self.datafilelist = datafilelist

    def __len__(self):
        return len(self.datafilelist)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range")
        file_name = self.datafilelist[idx]
        file_path = os.path.join(self.data_dir, file_name)
        item = opendata(file_path)
        obstacles_vertices = item['obstacles_vertices']
        if not obstacles_vertices.shape == (8, 4, 2):
            print('Error obstacles shape=', obstacles_vertices.shape, 'index=', idx)
        data = {
            'obstacles_vertices': torch.tensor(obstacles_vertices, dtype=torch.float32),
            'target': torch.tensor(item['target'], dtype=torch.float32),
            'path': torch.tensor(item['path'], dtype=torch.float32)
        }
        return data

def collate_fn(batch):

    collated = {
        'obstacles_vertices': torch.stack([item['obstacles_vertices'] for item in batch]),
        'target': torch.stack([item['target'] for item in batch]),
        'path': torch.stack([item['path'] for item in batch])
    }

    return collated