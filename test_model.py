from time import time
from glob import glob
import torch
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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


origin_pathD = sys.argv[3].split('/ImageSets')[0]

with open(test_set, 'r') as f:
    files = f.readlines()


timer = Timer()

def compute_iou(boxA, boxB, eps=1e-5):
        # crop
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        intersect = max(0, xB - xA + eps) * max(0, yB - yA + eps)
        box1_w, box1_h = boxA[2]-boxA[0], boxA[3]-boxA[1]
        box2_w, box2_h = boxB[2]-boxB[0], boxB[3]-boxB[1]
        total_area = (box1_w*box1_h) + (box2_w*box2_h)
        return intersect/(total_area-intersect)

def predict(image):
    timer.start()
    boxes, labels, probs = predictor.predict(image, 20, 0.1)
    #boxes = 
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    all_ = []
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

        cv2.putText(image, label,
                    (box[0]+20, box[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
        all_.append(label.split(':')[0])
        print('\n') 
        with open("output.txt", "w+") as f:
            print("".join(all_), file=f)
    return image, boxes.size(0)

total_aksara = 0
aksara = 0
for file in tqdm(files[:20]):
    annotation_file = "%s/Annotations/%s.xml"%(origin_pathD, file.strip())
    objects = ET.parse(annotation_file).findall("object")
    img = cv2.imread("%s/JPEGImages/%s.jpg"%(origin_pathD, file.strip()))
    #img.to(DEVICE)
    img, n_box = predict(img)
    total_aksara += len(objects)
    for obj in objects:
        bbox = obj.find('bndbox')
        #print(bbox.find('xmin').text)
        x1 = int(float(bbox.find('xmin').text)) - 1
        y1 = int(float(bbox.find('ymin').text)) - 1
        x2 = int(float(bbox.find('xmax').text)) - 1
        y2 = int(float(bbox.find('ymax').text)) - 1
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
    aksara += min(n_box, len(objects))
    name_file = "predicted/%s_%d_of_%d.jpg"%(file.split(".")[0], n_box, len(objects))
    
    print('Result from %s' % name_file)
    cv2.imwrite(name_file, img)
with open("result.txt", 'w') as f:
    accuracy = (aksara/total_aksara)*100
    print("\nAkurasi model: %.3f percent"%accuracy)
    f.writelines("Akurasi model: %.3f percent"%accuracy)
