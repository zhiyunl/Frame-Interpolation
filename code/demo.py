import glob
import os

import torchvision
from PIL import Image
import numpy as np
import cv2

from FrameDataset import tensor2PIL
from model import *


def video2frames(video, out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    success, image = video.read()
    count = 0
    while success:
        cv2.imwrite(out_path + "/%05d.jpg" % count, image)  # save frame as JPEG file
        success, image = video.read()
        count += 1
    print('Readed {} frames '.format(count))


def frames2video(frames, fps, out_path, isPIL=True):
    width, height = frames[0].size
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for f in frames:
        # transfer pil image into cv2 image
        if isPIL:
            video_writer.write(np.array(f)[:, :, ::-1])
        else:
            video_writer.write(f)
    video_writer.release()


# from video to frames
src_dir = './DAVIS2017/JPEGImages/480p/bear.mp4'
video = cv2.VideoCapture(src_dir)
out_frame_dir = src_dir.rstrip('.mp4')
# video2frames(video,out_frame_dir)


# read in frames
img_list = glob.glob(out_frame_dir + "/*.jpg")
img_list.sort()
frames = []
for im in img_list:
    frames.append(Image.open(im))

fps = 15
# load model and interpolate
checkpoint = torch.load('./model_save.pth')
kernel_size = checkpoint['kernel_size']
model = AdaptiveConvNet(kernel_size=kernel_size)
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)
model.epoch = checkpoint['epoch']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

model.eval()
frames_low = []
frames_gt = frames
frames_interpolate = []
transform = torchvision.transforms.ToTensor()


# frames = [transform(f).unsqueeze(0) for f in frames]


with torch.no_grad():
    for i in range(0, len(frames) - 2, 2):
        frames_low.append(frames[i])
        frames_interpolate.append(frames[i])

        frame0 = transform(frames[i]).unsqueeze(0).to(device)
        frame2 = transform(frames[i + 2]).unsqueeze(0).to(device)
        # print(frame0.size())

        interpolate = model.forward(frame0, frame2)
        # print(interpolate.size())
        out = tensor2PIL(interpolate.squeeze(0))
        # out.save('00000i.jpg')
        frames_interpolate.append(out)

frames2video(frames_interpolate, fps * 2, out_frame_dir + str(fps * 2) + 'fps_interpolate.mp4', isPIL=True)
frames2video(frames_low,fps,out_frame_dir+str(fps)+'fps.mp4',isPIL=True)
frames2video(frames_gt,fps*2,out_frame_dir+str(fps*2)+'fps_gt.mp4',isPIL=True)