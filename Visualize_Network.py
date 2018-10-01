# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import torch
from torch.autograd import Variable
from torchviz import make_dot
from models import miracle_net
from config import Config

opt = Config()

x = Variable(torch.randn(256, 1, 40960))  # change 12 to the channel number of network input
model = miracle_net.MiracleNet(opt)
y = model(x)
g = make_dot(y)
g.view()
