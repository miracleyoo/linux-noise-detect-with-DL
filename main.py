# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

from utils import *
from data_loader import *
from train import *
from config import Config
from models import miracle_net
from tensorboardX import SummaryWriter
import argparse
import torch
import os


def main():
    # Initializing configs
    folder_init(opt)
    all_loader = None
    train_loader = None
    net = None
    pre_epoch = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize model chosen
    try:
        if opt.MODEL == 'MiracleNet':
            net = miracle_net.MiracleNet(opt)
    except KeyError('Your model is not found.'):
        exit(0)
    finally:
        print("==> Model initialized successfully.")

    if opt.LOAD_SAVED_MOD:
        net, pre_epoch, best_loss = net.load(map_location=device.type)
        net.best_loss = best_loss
    net.to_multi(device=device)

    # Instantiation of tensorboard and add net graph to it
    writer = SummaryWriter(opt.SUMMARY_PATH)
    dummy_input = torch.rand(opt.BATCH_SIZE, opt.NUM_CHANNEL, opt.LENGTH)
    try:
        writer.add_graph(net, dummy_input)
    except KeyError:
        writer.add_graph(net.module, dummy_input)

    # Load data
    if opt.TRAIN_ALL or opt.MASS_TESTING:
        all_loader, test_loader = gen_dataset(Template, opt, True)
    else:
        train_loader, test_loader = gen_dataset(Template, opt, False)
    print("==> All datasets are generated successfully.")

    # Start training or testing
    if opt.MASS_TESTING:
        with Timer(name='testing'):
            test_loss, test_acc = testing(opt, all_loader, net, device)
        print("==> Test loss: %.4f, test acc: %.4f" % (test_loss, test_acc))
    elif opt.TRAIN_ALL:
        training(opt, writer, all_loader, test_loader, net, pre_epoch, device)
    else:
        training(opt, writer, train_loader, test_loader, net, pre_epoch, device)


def str2bool(b):
    if b.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif b.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    # Options
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-lsm', '--LOAD_SAVED_MOD', type=str2bool,
                        help='If you want to load saved model')
    parser.add_argument('-gi', '--GPU_INDEX', type=str,
                        help='Index of GPUs you want to use')

    args = parser.parse_args()
    print(args)
    opt = Config()
    for k, v in vars(args).items():
        if v is not None and hasattr(opt, k):
            setattr(opt, k, v)
            print(k, v, getattr(opt, k))
    if args.GPU_INDEX:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_INDEX
    main()
