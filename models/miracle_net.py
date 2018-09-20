# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import torch
import torch.nn as nn
from .BasicModule import BasicModule

torch.manual_seed(1)


class MiracleNet(BasicModule):
    def __init__(self, opt):
        super(MiracleNet, self).__init__(opt)
        self.model_name = "Miracle_Net"
        self.convs = nn.Sequential(
            nn.Conv1d(1, 128, 16, stride=8, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(128, 128, 16, stride=8, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(128, 256, 16, stride=8, padding=4),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 512, 16, stride=8, padding=4),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.MaxPool1d(3, 1, padding=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(opt.LENGTH//(8**4) * 512, opt.LINER_HID_SIZE),
            nn.BatchNorm1d(opt.LINER_HID_SIZE),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(opt.LINER_HID_SIZE, opt.NUM_CLASSES)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
