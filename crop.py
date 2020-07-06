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

#set_flush_denormal(True)
if len(sys.argv) < 4:
    print('Usage: python run_ssd_example.py <model path> <label path> [test.txt]')
    sys.exit(0)
model_path = sys.argv[1]
label_path = sys.argv[2]
test_set = sys.argv[3]

with open(test_set, 'r') as f:
    files = f.readlines()

print(len(files))
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

net = create_fpnnet_ssd(num_classes, is_test=True)
net.load_state_dict(load(model_path, map_location=lambda storage, loc: storage))
net.eval()
predictor = create_fpn_ssd_predictor(net, candidate_size=200)


#summary(net, input_size=(3, 300, 300))
timer = Timer()

def compute_iou(boxA, boxB, eps=1e-5):
        #koordinat kotak
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # total area kotak
        intersect = max(0, xB - xA + eps) * max(0, yB - yA + eps)
        return intersect

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
