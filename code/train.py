import statistics

from FrameDataset import FrameDataset, Middlebury
from torch.utils.data import DataLoader
from model import AdaptiveConvNet
import argparse
import torch
import os
import time

# Zhiyun Ling

parser = argparse.ArgumentParser(description='Adaptive Conv Pytorch Train')

# parameters
parser.add_argument('--data', type=str, default='./vimeo_triplet/sequences')
parser.add_argument('--train_list', type=str, default='./vimeo_triplet/tri_trainlist.txt')
parser.add_argument('--kernel', type=int, default=51)
parser.add_argument('--out_dir', type=str, default='./output')
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--load_model', type=str, default='./output/checkpoint/model_epoch2550.pth')
parser.add_argument('--test_input', type=str, default='./Middlebury/input')
parser.add_argument('--gt', type=str, default='./Middlebury/gt')

###############################
# Project Code Only Works on GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


##############################

def main():
    args = parser.parse_args()
    data_dir = args.data
    save_step = 2
    list_file = args.train_list
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    result_dir = args.out_dir + '/result'
    ckpt_dir = args.out_dir + '/checkpoint'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    timestr = time.strftime("%Y%m%d-%H%M%S")

    logfile = open(args.out_dir + '/trn_log' + timestr + '.txt', 'w')
    # logfile.write('batch_size: ' + str(args.batch_size) + '\n')

    total_epoch = args.epochs
    batch_size = args.batch_size

    trn_dataset = FrameDataset(data_dir, list_file, resize=(128, 128), whole=False)
    train_loader = DataLoader(dataset=trn_dataset, batch_size=batch_size, shuffle=True)
    print_step = len(trn_dataset) // args.batch_size // 10
    print_step += 1 if print_step == 0 else 0

    tst_dataset = Middlebury(args.test_input, args.gt)
    test_output_dir = args.out_dir + '/result'
    args.load_model = None
    if args.load_model is not None:
        checkpoint = torch.load(args.load_model)
        kernel_size = checkpoint['kernel_size']
        model = AdaptiveConvNet(kernel_size=kernel_size, lr=0.001)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        model.epoch = checkpoint['epoch']
    else:
        kernel_size = args.kernel
        model = AdaptiveConvNet(kernel_size=kernel_size)

    # logfile.write('kernel_size: ' + str(kernel_size) + '\n')

    if torch.cuda.is_available():
        model = model.cuda()

    max_step = train_loader.__len__()
    while True:
        if model.epoch.item() == total_epoch:
            break
        model.train()
        losses = []
        for batch_idx, (frame0, frame1, frame2) in enumerate(train_loader):
            frame0, frame1, frame2 = frame0.to(device), frame1.to(device), frame2.to(device)
            loss = model.train_model(frame0, frame2, frame1).item()
            losses.append(loss)
            if batch_idx % print_step == 0:
                print('{:<13s}{:<14s}{:<6s}{:<16s}{:<12s}{:<20.16f}'.format('Train Epoch: ',
                                                                            '[' + str(model.epoch.item()) + '/' + str(
                                                                                total_epoch) + ']', 'Step: ',
                                                                            '[' + str(batch_idx) + '/' + str(
                                                                                max_step) + ']', 'train loss: ',
                                                                            loss))
            logfile.write("{:10.15f},".format(loss))
        model.increase_epoch()
        # logfile.write("{:10f},".format(statistics.mean(losses)))
        if model.epoch.item() % save_step == 0:
            torch.save({'epoch': model.epoch, 'state_dict': model.state_dict(), 'kernel_size': kernel_size},
                       ckpt_dir + '/model_epoch' + str(model.epoch.item()).zfill(3) + '.pth')
            model.eval()
            tst_dataset.benchmark(model, test_output_dir, str(model.epoch.item()).zfill(3) + '.png')

    logfile.close()
    plottxt(args.out_dir + '/trn_log' + timestr + '.txt')


def plottxt(filename):
    import matplotlib.pyplot as plt
    import numpy as np
    file = np.genfromtxt(filename, delimiter=',')
    # file = file[file > 4]
    plt.plot(file)
    # plt.ylim((0,0.2))
    plt.ylabel('L1 Loss')
    plt.xlabel('Steps')
    plt.show()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main()
