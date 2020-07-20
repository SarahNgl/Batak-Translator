from time import time
from glob import glob
import sys
import cv2
import numpy as np
from torch import load, set_flush_denormal
from vision.ssd.fpnnet_ssd import create_fpnnet_ssd, create_fpn_ssd_predictor
from vision.utils.misc import Timer
from tracker import Tracker
from utils import Flags
from torchsummary import summary
import xml.etree.ElementTree as ET
from tqdm import tqdm

timer = Timer()

def compute_iou(boxA, boxB, eps=1e-5):
        # kordinat kotak
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        #total area kotak
        intersect = max(0, xB - xA + eps) * max(0, yB - yA + eps)
        return intersect/(total_area-intersect)

def predict(image, name):
    timer.start()
    boxes, labels, probs = predictor.predict(image, 20, 0.9)
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)), end="", flush=True)
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        cropped_image = image[int(box[1]):int(box[3])+1, int(box[0]):int(box[2])+1]
        cv2.imwrite("Datasets/JPEGImages/image_in_image/%s_%d.jpg"%(name, i), cropped_image)
    return image

for file in tqdm(files):
    img = cv2.imread("Datasets/image_in_image/%s"%file.strip())
    img = predict(img, file.strip().split(".")[0])
