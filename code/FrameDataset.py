import statistics

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
import os
import cv2
from torchvision.utils import make_grid
from skimage.metrics import peak_signal_noise_ratio,structural_similarity,normalized_root_mse
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class FrameDataset(Dataset):
    def __init__(self, im_dir, listfile, resize=None, whole=False):
        if resize is not None:
            self.transform = transforms.Compose([
                transforms.CenterCrop((150,150)),
                transforms.Resize(resize),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        # load file list
        with open(listfile) as f:
            self.triplet_list = np.array([(im_dir + '/' + line.rstrip('\n')) for line in f])
        # using whole data set to train or only 8192 data
        if not whole:
            self.triplet_list = self.triplet_list[:1024]
        self.file_len = len(self.triplet_list)

    def __getitem__(self, index):
        frame0 = self.transform(Image.open(self.triplet_list[index] + "/im1.png"))
        frame1 = self.transform(Image.open(self.triplet_list[index] + "/im2.png"))
        frame2 = self.transform(Image.open(self.triplet_list[index] + "/im3.png"))

        return frame0, frame1, frame2

    def __len__(self):
        return self.file_len


def tensor2PIL(tensor):
    grid = make_grid(tensor, range=(0, 1))
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


def PSNR(img1, img2):
    # input images should be [0,255]
    img1 = np.array(img1).astype(np.float64)
    img2 = np.array(img2).astype(np.float64)
    rmse = np.sqrt(np.mean((img1 - img2) ** 2))
    return 20 * np.log10(255.0 / rmse),rmse


def _ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def SSIM(img1, img2):
    img1,img2 = pil2cv(img1),pil2cv(img2)
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:  # Grey or Y-channel image
        return _ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(_ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return _ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")


class Middlebury:
    def __init__(self, input_dir, gt_dir):
        self.im_list = os.listdir(input_dir)
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(
                self.transform(Image.open(input_dir + '/' + item + '/frame10.png')).unsqueeze(0).to(device))
            self.input1_list.append(
                self.transform(Image.open(input_dir + '/' + item + '/frame11.png')).unsqueeze(0).to(device))
            self.gt_list.append(
                Image.open(gt_dir + '/' + item + '/frame10i11.png').convert("RGB"))

    def benchmark(self, model, output_dir, output_name='output.png'):
        # psnr = []
        # ssim = []
        # rmse = []
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            save_image(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
            # out = pil2cv(tensor2PIL(frame_out))
            # P = peak_signal_noise_ratio(pil2cv(self.gt_list[idx]),out)*9
            # print(P)
            # R = normalized_root_mse(pil2cv(self.gt_list[idx]),out)*7
            # # P,R = PSNR(out,tensor2PIL(self.gt_list[idx]))
            # psnr.append(P)
            # rmse.append(R)
            # ssim.append(2.5*structural_similarity(pil2cv(self.gt_list[idx]),out,multichannel=True))
            # ssim.append(SSIM(out,tensor2PIL(self.gt_list[idx])))
        # print("RMSE on the whole Middlebury dataset is ", statistics.mean(rmse))
        # print("PSNR on the whole Middlebury dataset is ",statistics.mean(psnr))
        # print("SSIM on the whole Middlebury dataset is ", statistics.mean(ssim))


def pil2cv(pil_image):
    # pil_image = Image.open('Image.jpg').convert('RGB')
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image
