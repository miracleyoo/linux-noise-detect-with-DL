# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import os
import pickle
import time
import numpy as np
from torch.utils.data import DataLoader
__all__ = ['gen_dataset', 'load_data', 'folder_init', 'Timer']


def gen_dataset(data_loader, opt, if_all):
    train_pairs, test_pairs = load_data(opt, opt.DATA_ROOT)

    test_dataset = data_loader(test_pairs, opt)
    test_loader = DataLoader(dataset=test_dataset, batch_size=opt.TEST_BATCH_SIZE, shuffle=False,
                             num_workers=opt.NUM_WORKERS, drop_last=False)

    opt.NUM_TEST = len(test_dataset)

    if if_all:
        train_pairs.extend(test_pairs)
        all_dataset = data_loader(train_pairs, opt)
        all_loader = DataLoader(dataset=all_dataset, batch_size=opt.TEST_BATCH_SIZE, shuffle=True,
                                num_workers=opt.NUM_WORKERS, drop_last=False)
        opt.NUM_TRAIN = len(all_dataset)
        if opt.MASS_TESTING:
            opt.NUM_TEST = len(all_dataset)
        return all_loader, test_loader
    else:
        train_dataset = data_loader(train_pairs, opt)
        train_loader = DataLoader(dataset=train_dataset, batch_size=opt.BATCH_SIZE, shuffle=True,
                                  num_workers=opt.NUM_WORKERS, drop_last=False)
        opt.NUM_TRAIN = len(train_dataset)
        return train_loader, test_loader


def load_data(opt, data_root):
    start = time.time()
    all_pairs = pickle.load(open(data_root, 'rb+'))
    print("==> Load data successfully, time elapsed: %.4f" % (time.time() - start))
    length = len(all_pairs)
    opt.NUM_CLASSES = len(set(np.array(all_pairs)[:, 1]))
    opt.PAIR_LENGTH = len(all_pairs[0][0][0])
    # all_pairs = [(i[1:], int(i[0])) for i in all_data]
    sep_point = int(np.floor(opt.TRAIN_DATA_RATIO*length))
    train_pairs = all_pairs[:sep_point]
    test_pairs = all_pairs[sep_point:]
    return train_pairs, test_pairs


def folder_init(opt):
    """
    Initialize folders required
    """
    if not os.path.exists('source'):
        os.mkdir('source')
    if not os.path.exists('source/reference'):
        os.mkdir('source/reference')
    if not os.path.exists('./source/summary/'):
        os.mkdir('./source/summary/')
    if not os.path.exists('./source/val_results/'):
        os.mkdir('./source/val_results/')
    if not os.path.exists('source/simulation_res'):
        os.mkdir('source/simulation_res')
    if not os.path.exists('source/simulation_res/intermediate_file'):
        os.mkdir('source/simulation_res/intermediate_file')
    if not os.path.exists('source/simulation_res/train_data'):
        os.mkdir('source/simulation_res/train_data')
    if not os.path.exists(opt.NET_SAVE_PATH):
        os.mkdir(opt.NET_SAVE_PATH)


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        print('==> [%s]:\t' % self.name, end='')
        self.time_elapsed = time.time() - self.tstart
        print('Elapsed Time: %s (s)' % self.time_elapsed)
