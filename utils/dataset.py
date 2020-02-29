import numpy as np
import torch
from torch.utils.data import Dataset


class DealDataSet(Dataset):
    def __init__(self, filename):
        xy = np.load(filename, allow_pickle=True)  # 使用numpy读取数据
        self.x_data = torch.tensor(list(xy['images'])).float()
        self.y_data = torch.tensor(xy['labels']).float()
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
