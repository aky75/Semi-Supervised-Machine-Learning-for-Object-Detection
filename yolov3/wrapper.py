from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import shutil
import datetime
import argparse
import cv2
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str, default="config/kitti-ssl.data", help="path to data config file")
    parser.add_argument("--model_def", type=str, default="config/yolov3-kitti.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoint_train_50%/yolov3_ckpt_24.pth", help="path to weights file")
    #parser.add_argument("--class_path", type=str, default="data/kitti1.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.9, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs("output_train", exist_ok=True)
    #os.makedirs("checkpoint_train", exist_ok=True)
    
    pseudo_rej = []
    pseudo_train = []
    
    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    predict_path = data_config["predict"]
    class_names = load_classes(data_config["names"])
    weights = torch.load(opt.weights_path)

    pseudo_rej = []
    pseudo_train = []
        
    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    # Load checkpoint weights
    model.load_state_dict(weights)

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ListImages(predict_path, img_size=opt.img_size),
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))
        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)
            
    print("\nSaving labels:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        # Create plot
        img = cv2.imread(path.rstrip())
        img_height, img_width = img.shape[0],img.shape[1]
        label_to_write = []

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            #print('detections',detections)
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                if float(cls_conf) > opt.conf_thres:
                        
                    print("\t+ Label: %s, Conf: %.5f" % (class_names[int(cls_pred)], cls_conf.item()))

                #print("Bounding Box:", class_str,x1,y1,w,h)
                    intx1 = int(x1)
                    inty1 = int(y1)
                    intx2 = int(x2)
                    inty2 = int(y2)

                    bbox_center_x = float( (x1 + (x2 - x1) / 2.0) / img_width)
                    bbox_center_y = float( (y1 + (y2 - y1) / 2.0) / img_height)
                    bbox_width = float((x2 - x1) / img_width)
                    bbox_height = float((y2 - y1) / img_height)

                    line_to_write = str(int(cls_pred)) + ' ' + str(bbox_center_x)+ ' ' + str(bbox_center_y) + ' ' +str(bbox_width)+ ' '                                            + str(bbox_height) +'\n'
                    label_to_write.extend(line_to_write)

        # Save generated detections as pseudo labels
        if len(label_to_write) > 0:
            filename = path.split("/")[-1].split(".")[0]
            label_data = open(f"/home/aditya/yolov3/master/data/labels/{filename}.txt",'w')
            for line in label_to_write:
                label_data.write(line)
            label_data.close()
            pseudo_train.append(str('/home/aditya/yolov3/master/data/images/'+filename+'.png'))
        else:
            pseudo_rej.append(str('/home/aditya/yolov3/master/data/images/'+filename+'.png'))

            unique_src_path = list(dict.fromkeys(pseudo_rej))
    #outF = open("data/pseudo/pseudo_train_rejected_list.txt", "w")
    with open('data/pseudo/pseudo_train_rejected_list.txt', 'w') as f:
        f.truncate(0)
        for src in unique_src_path:
            f.write("%s\n" % src)
    #outF.close()  
    #outF = open("data/pseudo/pseudo_train_rest.txt", "w")
    with open('data/pseudo/pseudo_train_rest.txt', 'w') as f:
        f.truncate(0)
        for path in pseudo_train:
            f.write("%s\n" % path)
    #outF.close()
    print("Labels Generated.")
