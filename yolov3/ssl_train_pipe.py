from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

import os
import sys
import time
import shutil
import datetime
import argparse
import cv2
from PIL import Image
from terminaltables import AsciiTable

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torchvision import transforms

import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3-kitti-ssl.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/kitti-ssl.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--conf_thres", type=float, default=0.9, help="object confidence threshold")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)
    
    logger = Logger("logs")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs("output_train", exist_ok=True)
    #os.makedirs("checkpoint_train", exist_ok=True)
    
    pseudo_rej = []
    pseudo_train = []
    
    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    predict_path = data_config["predict"]
    class_names = load_classes(data_config["names"])
    #weights = torch.load(opt.pretrained_weights)
    
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    model.load_state_dict(torch.load(opt.pretrained_weights))
    
    for epoch in range(opt.epochs):
    
        pseudo_rej = []
        pseudo_train = []
        
        # Set up model
        #model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

        # Load checkpoint weights
        #model.load_state_dict(weights)

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
                label_data = open(f"/home/aditya/yolov3/master/data/pseudo/labels/{filename}.txt",'w')
                for line in label_to_write:
                    label_data.write(line)
                label_data.close()
                pseudo_train.append(str('/home/aditya/yolov3/master/data/pseudo/images/'+filename+'.png'))
            else:
                pseudo_rej.append(str('/home/aditya/yolov3/master/data/pseudo/images/'+filename+'.png'))


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
        
        #Begin training

        # Initiate model
        #model = Darknet(opt.model_def).to(device)
        #model.apply(weights_init_normal)

        #Load weights
        #model.load_state_dict(weights)

        # Get dataloader
        dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

        optimizer = torch.optim.Adam(model.parameters())

        metrics = [
            "grid_size",
            "loss",
            "x",
            "y",
            "w",
            "h",
            "conf",
            "cls",
            "cls_acc",
            "recall50",
            "recall75",
            "precision",
            "conf_obj",
            "conf_noobj",
        ]
        #for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        epoch_loss = []
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"
            epoch_loss.append(float(loss.item()))
            
            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
            print(f"---- avg epoch_loss {sum(epoch_loss)/len(epoch_loss)}")
            
        if epoch % opt.checkpoint_interval == 0:
            
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
                    }
            torch.save(state, f"checkpoint_train/yolov3_ckpt_%d.pth" % epoch)
            #torch.save(model.state_dict(), f"checkpoint_train/yolov3_ckpt_%d.pth" % epoch)
            weights = torch.load(f"checkpoint_train/yolov3_ckpt_%d.pth" % epoch) #model.state_dict()
            model.load_state_dict(weights['state_dict'])
            optimizer.load_state_dict(weights['optimizer'])