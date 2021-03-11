# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import numpy as np
from time import time
import math
import pandas as pd
import csv
import math

from IOtools import txt_write
from Network.class_func import get_local_count


def test_phase(net, testloader, log_save_path=None):
    with torch.no_grad():
        net.eval()
        start = time()
        avg_frame_rate = 0

        for j, data in enumerate(testloader):
            print(data)
            inputs = data['image']
            inputs = inputs.type(torch.float32)
            inputs = inputs.cuda()
            print(inputs.size())
            # inputs = inputs
            # process with SSDCNet
            features = net(inputs)
            div_res = net.resample(features)
            merge_res = net.parse_merge(div_res)
            outputs = merge_res['div'+str(net.div_times)]
            del merge_res

            pre = (outputs).sum()

            end = time()
            running_frame_rate = 1 * float(1 / (end - start))
            avg_frame_rate = (avg_frame_rate*j + running_frame_rate)/(j+1)
            if j % 1 == 0:    # print every 2000 mini-batches

                print('Test:[%5d/%5d] pre: %.3f' %
                      (j + 1, len(testloader), pre))
                start = time()

    test_dict = dict()

    return test_dict
