import statistics
import sys

from skimage.metrics import peak_signal_noise_ratio, normalized_root_mse, structural_similarity, mean_squared_error

from FrameDataset import FrameDataset, Middlebury, pil2cv, tensor2PIL
from torch.utils.data import DataLoader
from model import AdaptiveConvNet
import argparse
import torch
import os
import time
import numpy as np

parser = argparse.ArgumentParser(description='Adaptive Conv Pytorch Test')

# parameters
parser.add_argument('--data', type=str, default='./vimeo_triplet/sequences')
parser.add_argument('--test_list', type=str, default='./vimeo_triplet/tri_testlist.txt')
parser.add_argument('--kernel', type=int, default=51)
parser.add_argument('--out_dir', type=str, default='./output')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--load_model', type=str, default='./model_save.pth')
parser.add_argument('--benchmark', type=str, default='./Middlebury/input')
parser.add_argument('--gt', type=str, default='./Middlebury/gt')


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    result_dir = args.out_dir + '/result'

    timestr = time.strftime("%Y%m%d-%H%M%S")
    logfile = open(args.out_dir + '/tst_log' + timestr + '.txt', 'w')
    logfile.write('batch_size: ' + str(args.batch_size) + '\n')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    tst_dataset = FrameDataset(args.data, args.test_list, whole=False)
    tst_loader = DataLoader(dataset=tst_dataset, batch_size=args.batch_size)
    print_step = len(tst_dataset) // args.batch_size // 10
    print_step += 1 if print_step == 0 else 0

    benchmark_dataset = Middlebury(args.benchmark, args.gt)
    benchmark_output = args.out_dir + '/result'

    if args.load_model is not None:
        checkpoint = torch.load(args.load_model)
        kernel_size = checkpoint['kernel_size']
        model = AdaptiveConvNet(kernel_size=kernel_size, lr=0.001)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        model.epoch = checkpoint['epoch']
    else:
        sys.exit()

    # logfile.write('kernel_size: ' + str(args.kernel) + '\n')

    if torch.cuda.is_available():
        model = model.cuda()
    max_step = tst_loader.__len__()
    model.eval()
    # benchmark model
    losses = []
    psnr = []
    rmse = []
    ssim = []
    with torch.no_grad():
        for batch_idx, (frame0, frame1, frame2) in enumerate(tst_loader):
            frame0, frame1, frame2 = frame0.to(device), frame1.to(device), frame2.to(device)
            loss,out = model.test_model(frame0, frame2, frame1)
            loss = loss.item()
            losses.append(loss)
            print('{:<8s}{:<8s}{:<16s}{:<12s}{:<20.16f}'.format('Test: ',
                                                                'Batch: ',
                                                                '[' + str(batch_idx) + '/' + str(
                                                                    max_step) + ']', 'test loss: ',
                                                                loss))
            out = pil2cv(tensor2PIL(out))
            gt = pil2cv(tensor2PIL(frame1))
            P = peak_signal_noise_ratio(gt, out)
            R = np.sqrt(mean_squared_error(gt, out))
            psnr.append(P)
            rmse.append(R)
            ssim.append(structural_similarity(gt, out, multichannel=True))
        print("RMSE on the validation dataset is ", statistics.mean(rmse))
        print("PSNR on the validation dataset is ", statistics.mean(psnr))
        print("SSIM on the validation dataset is ", statistics.mean(ssim))
        print('Average loss:{}+/-({})'.format(statistics.mean(losses), statistics.stdev(losses)))
        # benchmark
        # benchmark_dataset.benchmark(model, benchmark_output, str(model.epoch.item()).zfill(3) + '.png')
    logfile.close()


if __name__ == '__main__':
    print(torch.cuda.is_available())
    main()
