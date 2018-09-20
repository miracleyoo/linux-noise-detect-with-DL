# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import os
import torch
import shutil
import torch.nn as nn
import torch.optim as optim
import threading
lock = threading.Lock()


class MyThread(threading.Thread):
    def __init__(self, opt, net, epoch, bs_old, loss):
        threading.Thread.__init__(self)
        self.opt = opt
        self.net = net
        self.epoch = epoch
        self.bs_old = bs_old
        self.loss = loss

    def run(self):
        lock.acquire()
        try:
            if self.opt.SAVE_TEMP_MODEL:
                self.net.save(self.epoch, self.loss, "temp_model.dat")
            if self.opt.SAVE_BEST_MODEL and self.loss < self.bs_old:
                self.net.best_loss = self.loss
                net_save_prefix = self.opt.NET_SAVE_PATH + self.opt.MODEL + '_' + self.opt.PROCESS_ID + '/'
                temp_model_name = net_save_prefix + "temp_model.dat"
                best_model_name = net_save_prefix + "best_model.dat"
                shutil.copy(temp_model_name, best_model_name)
        finally:
            lock.release()


class BasicModule(nn.Module):
    def __init__(self, opt=None):
        super(BasicModule, self).__init__()
        self.model_name = self.__class__.__name__
        self.opt = opt
        self.best_loss = 1e8
        self.threads = []

    def load(self, map_location='cpu', model_type="temp_model.dat"):
        print('==> Now using ' + self.opt.MODEL + '_' + self.opt.PROCESS_ID)
        print('==> Loading model ...')
        pre_epoch = 0
        net_save_prefix = self.opt.NET_SAVE_PATH + self.opt.MODEL + '_' + self.opt.PROCESS_ID + '/'
        temp_model_name = net_save_prefix + model_type
        if not os.path.exists(net_save_prefix):
            os.mkdir(net_save_prefix)
        if os.path.exists(temp_model_name):
            checkpoint = torch.load(temp_model_name, map_location=map_location)
            pre_epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            self.load_state_dict(checkpoint['state_dict'])
            print("==> Load existing model: %s" % temp_model_name)
        else:
            print("==> The model you want to load (%s) doesn't exist!" % temp_model_name)
        return self, pre_epoch, self.best_loss

    def save(self, epoch, loss, name=None):
        if loss < self.best_loss:
            self.best_loss = loss
        if self.opt is None:
            prefix = "./source/trained_net/" + self.model_name + "/"
        else:
            prefix = self.opt.NET_SAVE_PATH + self.opt.MODEL + '_' + \
                     self.opt.PROCESS_ID + '/'
            if not os.path.exists(prefix): os.mkdir(prefix)

        if name is None:
            name = "temp_model.dat"

        path = prefix + name
        try:
            state_dict = self.module.state_dict()
        except:
            state_dict = self.state_dict()
        torch.save({
            'epoch': epoch + 1,
            'state_dict': state_dict,
            'best_loss': self.best_loss
        }, path)
        return path

    def mt_save(self, epoch, loss):
        if self.opt.SAVE_BEST_MODEL and loss < self.best_loss:
            print("==> Your best model is renewed")
        if len(self.threads) > 0:
            self.threads[-1].join()
        self.threads.append(MyThread(self.opt, self, epoch, self.best_loss, loss))
        self.threads[-1].start()
        if self.opt.SAVE_BEST_MODEL and loss < self.best_loss:
            print("==> Your best model is renewed")
            self.best_loss = loss

    def get_optimizer(self, lr):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        return optimizer

    def to_multi(self, device):
        # Data Parallelism
        if torch.cuda.is_available():
            print("==> Using", torch.cuda.device_count(), "GPUs.")
            if torch.cuda.device_count() > 1:
                self = torch.nn.DataParallel(self)
                print("==> Using data parallelism.")
        else:
            print("==> Using CPU now.")
        self.to(device)
        return self
