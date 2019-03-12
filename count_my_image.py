import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from src import network
from src.models import MCNN


class CrowdCounter(nn.Module):
    def __init__(self):
        super(CrowdCounter, self).__init__()
        self.DME = MCNN()
        self.loss_fn = nn.MSELoss()

    @property
    def loss(self):
        return self.loss_mse

    def forward(self, im_data, gt_data=None):
        im_data = network.np_to_variable(im_data, is_cuda=True, is_training=self.training)
        density_map = self.DME(im_data)
        # if self.training:
        #     gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=self.training)
        #     self.loss_mse = self.build_loss(density_map, gt_data)
        return density_map

    def build_loss(self, density_map, gt_data):
        loss = self.loss_fn(density_map, gt_data)
        return loss


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

model_path = 'e:/root/final_models/mcnn_shtechB_110.h5'
path = 'e:/my_data/'
dirs = os.listdir(path)  # 存储路径下的所有文件名
print(dirs)
net = CrowdCounter()
trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
net.cuda()
net.eval()
for dir_name in dirs:
    print('images in file \'' + dir_name + '\'')
    data_path = path + dir_name + '/'
    files = os.listdir(data_path)
    for file_name in files:
        img = cv2.imread(data_path + file_name, 0)
        img = img.astype(np.float32, copy=False)
        ht = img.shape[0]
        wd = img.shape[1]
        while ht > 1000 and wd > 1000:  # 控制图片大小
            ht //= 2
            wd //= 2
            img = cv2.resize(img, (wd, ht))
        img = img.reshape((1, 1, img.shape[0], img.shape[1]))
        density_map = net.forward(img)
        density_map = density_map.data.cpu().numpy()
        et_count = np.sum(density_map)
        print(file_name + ':', et_count)
    else:
        print()
else:
    print('finished')
