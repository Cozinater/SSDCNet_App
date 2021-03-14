import cv2
import pickle
from imutils.video import WebcamVideoStream
from imutils.video import FileVideoStream

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import os
import numpy as np
import time
import math
import pandas as pd
import csv
import cv2

from IOtools import txt_write, get_config_str
from load_data_V2 import Countmap_Dataset
from Network.SSDCNet import SSDCNet_classify
from Network.class_func import get_local_count
from Val import test_phase

import glob  # use glob.glob to get special flielist
import scipy.io as sio  # use to import mat as dic,data is ndarray
# load image
from PIL import Image
from matplotlib import cm
# torch related
from torch.utils.data import Dataset
from torchvision import transforms
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class VideoCamera(object):
    def __init__(self, medium):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.

        if medium == "video":
            self.stream = FileVideoStream("videos/cctv.mp4").start()
        elif medium == "webcam":
            self.stream = WebcamVideoStream(src=0).start()

        save_folder = 'model'
        root_dir = r'data'
        num_workers = 1
        transform_test = []

        label_indice = np.arange(0.5, 7.5, 0.5)
        add = np.array([1e-6, 0.05, 0.10, 0.15, 0.20,
                        0.25, 0.30, 0.35, 0.40, 0.45])
        label_indice = np.concatenate((add, label_indice))
        print("label_indice:", label_indice)

        rgb_dir = os.path.join(root_dir, 'rgbstate.mat')
        mat = sio.loadmat(rgb_dir)
        self.rgb = mat['rgbMean'].reshape(1, 1, 3)

        label_indice = torch.Tensor(label_indice)
        class_num = len(label_indice)+1

        self.net = SSDCNet_classify(class_num, label_indice, div_times=2,
                                    frontend_name='VGG16', block_num=5,
                                    IF_pre_bn=False, IF_freeze_bn=False, load_weights=True,
                                    psize=64, pstride=64, parse_method='maxp').cuda()

        mod_path = 'best_epoch.pth'
        mod_path = os.path.join('model', mod_path)
        print(mod_path)
        if os.path.exists(mod_path):
            all_state_dict = torch.load(mod_path)
            print(mod_path)
            self.net.load_state_dict(all_state_dict['net_state_dict'])
            tmp_epoch_num = all_state_dict['tmp_epoch_num']
            log_save_path = os.path.join(save_folder, 'log-epoch-min[%d]-%s.txt'
                                         % (tmp_epoch_num+1, 'maxp'))
        print("end of function")

    def __del__(self):
        self.stream.stop()

    def get_pad(self, inputs, DIV=64):
        h, w = inputs.size()[-2:]
        ph, pw = (DIV-h % DIV), (DIV-w % DIV)
        # print(ph,pw)

        tmp_pad = [0, 0, 0, 0]
        if (ph != DIV):
            tmp_pad[2], tmp_pad[3] = 0, ph
        if (pw != DIV):
            tmp_pad[0], tmp_pad[1] = 0, pw

        # print(tmp_pad)
        inputs = F.pad(inputs, tmp_pad)

        return inputs

    def get_frame(self):
        image = self.stream.read()

        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        print(image.shape)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).convert('RGB')
        print(type(img))
        #img = img.resize((640, 460))
        #img.show()
        img = transforms.ToTensor()(img)
        img = self.get_pad(img, DIV=64)
        img = img - torch.Tensor(self.rgb).view(3, 1, 1)
        # input_file.save(input_file.filename)
        # print(input_file.filename)

        inputs = img.type(torch.float32)
        inputs = inputs.unsqueeze(0)
        inputs = inputs.cuda()
        print(inputs.size())
        features = self.net(inputs)
        div_res = self.net.resample(features)
        merge_res = self.net.parse_merge(div_res)
        outputs = merge_res['div'+str(self.net.div_times)]
        del merge_res

        pre = (outputs).sum()
        threshold = 20
        result = int(math.floor(pre.item()))
        if result <= threshold:
            result = str(int(math.floor(pre.item()))) + "person "
        else:
            result = str(int(math.floor(pre.item()))) + \
                "person. " + "Please distance!"

        print('pre:', pre)

        startX = int(0)
        startY = int(0)
        endX = int(75)
        endY = int(75)

        cv2.putText(image, result, (endX, endY),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', image)
        data = []
        data.append(jpeg.tobytes())
        data.append(result)
        return data
