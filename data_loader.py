# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import torch
import numpy as np
from torch.utils.data import Dataset


class Template(Dataset):
    def __init__(self, data, opt):
        super(Template, self).__init__()
        self.data = data
        self.opt = opt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inputs, label = self.data[index]
        return torch.from_numpy(inputs).float().unsqueeze(dim=0), np.float32(label)
