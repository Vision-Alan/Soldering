#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:17:54 2022

@author: alan
"""

# 偵測圖片檔名
source = 'cam8_9_10_8.png'

# 匯入所需套件
import time
tt0 = time.time()

import os
import shutil
import yaml
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import matplotlib.pyplot as plt

import git
from git import RemoteProgress
from tqdm import tqdm
import gdown
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 參數設置
project = 'soldering'
weights = 'model/YOLOv4_soldering_best.pt'
cfg = 'cfg/YOLOv4_AIA_Soldering_A20725.cfg'
data = 'data/YOLOv4_AIA_Soldering_A20725.yaml'
out = 'detect'
imgsz = 800
conf_thres = 0.3
iou_thres = 0.5

Git_url = 'https://github.com/Vision-Alan/Soldering.git'
project_dir = Path.cwd() / project

# 下載相關 python 檔案
if not project_dir.exists():
    Path(project_dir).mkdir(parents=True)
    os.chdir(str(project_dir))     # 以相對路徑變更工作目錄
    print('\n@Python 程式檔變更工作目錄路徑 =', os.getcwd())         # 輸出路徑
    #Git clone 的 Python 進度條類別
    class CloneProgress(RemoteProgress):
        def __init__(self):
            super().__init__()
            self.pbar = tqdm()
        def update(self, op_code, cur_count, max_count=None, message=''):
            self.pbar.total = max_count
            self.pbar.n = cur_count
            self.pbar.refresh()
            
    # 複製 Git 儲存庫資料並顯示進度條, 從某個 URL 那裡 clone 到本地某個位置
    print('@Cloning Git Repository from [', Git_url, ']')
    git.Repo.clone_from(url = Git_url, to_path = project_dir , progress = CloneProgress())
    print("\t@Cloning Done.")
else:
    os.chdir(str(project_dir))     # 以相對路徑變更工作目錄
    print('@Python 程式檔變更工作目錄路徑 =', os.getcwd())         # 輸出路徑

# YOLOv4_soldering_best.pt
# https://drive.google.com/file/d/1lgBF_KxG4FOJqHO39vaNRM1JAcBx_YXC/view?usp=sharing


if not (project_dir / weights).exists():
    file_id = '1lgBF_KxG4FOJqHO39vaNRM1JAcBx_YXC'
    url = "https://drive.google.com/uc?id={}".format(file_id)
    #output = "YOLOv4_soldering_best.pt"
    gdown.download(url, weights)

# 創建資料夾
if not (project_dir / out).exists():
    Path(project_dir / out).mkdir(parents=True)

# 動態新增模組匯入的路徑
sys.path.append(str(project_dir))

# 匯入 YOLOv4 模型所需套件
from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from model.models import *
from utils.datasets import *
from utils.general import *
    
tt1 = time_synchronized()
ss0 = tt1 - tt0

# 使用訓練好的權重檔並讀入到模型中
model = Darknet(cfg, imgsz).to(device).eval()
model.load_state_dict(torch.load(weights, map_location=device)['model'])    

# 預測影像
dataset = LoadImages(source, img_size=imgsz, auto_size=32)

def load_classes(path):
    # Loads *.yaml file at 'path'
    with open(path) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  
    return data_dict['names']
    
# Get names and colors
names = load_classes(data)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

tt2 = time_synchronized()
ss1 = tt2 - tt1

# Run inference
with torch.no_grad():
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img) # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        t2 = time_synchronized()
        ss2 = t2 - t1

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s
            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            sss1 = s
            t3 = time_synchronized()
            ss3 = t3 - t2
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            cv2.imwrite(save_path, im0)
            t4 = time_synchronized()
    print('Results saved to %s' % Path(out))
    #ss4 = time.time() - t0
    ss4 = t4 -t3
    print('Done. (%.3fs)' % (time.time() - t0))

cv2.imshow(str(Path(path).stem), im0)
if cv2.waitKey(0) == 27:
    print('@Source = ', source)
    print('@Device = ', device)
    print('@Import time = %.3fs' % ss0)
    print('@Load model time = %.3fs' % ss1)
    print('@Detect =', sss1)
    print('@Detect time = %.3fs' % ss2)
    print('@Draw box time = %.3fs' % ss3)
    print('@Save file time = %.3fs' % ss4)
    cv2.destroyAllWindows()