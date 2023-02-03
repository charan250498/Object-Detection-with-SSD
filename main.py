import argparse
import os
import numpy as np
import time
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F

from scipy.special import softmax

from dataset import *
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
#please google how to use argparse
#a short intro:
#to train: python main.py
#to test:  python main.py --test


class_num = 4 #cat dog person background

num_epochs = 100
batch_size = 64


boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])


#Create network
network = SSD(class_num)
network.cuda()
cudnn.benchmark = True

def relative_to_absolute(ann_confidence, ann_box, pred_box, pred_confidence, boxs_default):
    for i in range(len(ann_box)):
        if ann_confidence[i, -1] == 0:
            ann_box[i, 0] = boxs_default[i, 2] * ann_box[i, 0] + boxs_default[i, 0]
            ann_box[i, 1] = boxs_default[i, 3] * ann_box[i, 1] + boxs_default[i, 1]
            ann_box[i, 2] = boxs_default[i, 2] * np.exp(ann_box[i, 2])
            ann_box[i, 3] = boxs_default[i, 3] * np.exp(ann_box[i, 3])
    
    for i in range(len(pred_box)):
        if pred_confidence[i, -1] == 0:
            pred_box[i, 0] = boxs_default[i, 2] * pred_box[i, 0] + boxs_default[i, 0]
            pred_box[i, 1] = boxs_default[i, 3] * pred_box[i, 1] + boxs_default[i, 1]
            pred_box[i, 2] = boxs_default[i, 2] * np.exp(pred_box[i, 2])
            pred_box[i, 3] = boxs_default[i, 3] * np.exp(pred_box[i, 3])

    # Previous Implementation
    # ann_boxes[:, 0] = boxs_default[:, 2] * ann_boxes[:, 0] + boxs_default[:, 0]
    # ann_boxes[:, 1] = boxs_default[:, 3] * ann_boxes[:, 1] + boxs_default[:, 1]
    # ann_boxes[:, 2] = boxs_default[:, 2] * np.exp(ann_boxes[:, 2])
    # ann_boxes[:, 3] = boxs_default[:, 3] * np.exp(ann_boxes[:, 3])

    # pred_box_[:, 0] = boxs_default[:, 2] * pred_box_[:, 0] + boxs_default[:, 0]
    # pred_box_[:, 1] = boxs_default[:, 3] * pred_box_[:, 1] + boxs_default[:, 1]
    # pred_box_[:, 2] = boxs_default[:, 2] * np.exp(pred_box_[:, 2])
    # pred_box_[:, 3] = boxs_default[:, 3] * np.exp(pred_box_[:, 3])

    return ann_box, pred_box

if not args.test:
    dataset = COCO("data/data/train/images/", "data/data/train/annotations/", class_num, boxs_default, train = True, image_size=320)
    dataset_test = COCO("data/data/train/images/", "data/data/train/annotations/", class_num, boxs_default, train = False, image_size=320)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    
    optimizer = optim.Adam(network.parameters(), lr = 1e-4)
    #feel free to try other optimizers and parameters.
    
    start_time = time.time()

    for epoch in range(num_epochs):
        #TRAINING
        network.train()

        avg_loss = 0
        avg_count = 0
        for i, data in enumerate(dataloader, 0):
            images_, ann_box_, ann_confidence_ = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            loss_net.backward()
            optimizer.step()
            
            avg_loss += loss_net.data
            avg_count += 1

            print(i)

        print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count))
                
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        ###########################
        ann_boxes = ann_box[0].detach().cpu().numpy()
        # ann_boxes, pred_box_ = relative_to_absolute(ann_box=ann_boxes, pred_box=pred_box_, ann_confidence=ann_confidence_[0].numpy(), pred_confidence=pred_confidence_, boxs_default=boxs_default)
        ###########################
        # visualize_pred("train", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        visualize_pred("train", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_boxes, images_[0].numpy(), boxs_default)
        
        #save weights
        if epoch%10==9:
            #save last network
            print('saving net...')
            torch.save(network.state_dict(), 'network.pth')

        #VALIDATION
        network.eval()
        
        # TODO: split the dataset into 90% training and 10% validation
        # use the training set to train and the validation set to evaluate
        
        for i, data in enumerate(dataloader_test, 0):
            images_, ann_box_, ann_confidence_ = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            pred_confidence, pred_box = network(images)
            
            pred_confidence_ = pred_confidence.detach().cpu().numpy()
            pred_box_ = pred_box.detach().cpu().numpy()
            
            #optional: implement a function to accumulate precision and recall to compute mAP or F1.
            #update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default,precision_,recall_,thres)
        
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        #visualize_pred("val", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)

        #optional: compute F1
        #F1score = 2*precision*recall/np.maximum(precision+recall,1e-8)
        #print(F1score)


else:
    #TEST
    dataset_test = COCO("data/data/train/images/", "data/data/train/annotations/", class_num, boxs_default, train = False, image_size=320)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=0)
    # network.load_state_dict(torch.load('network.pth'))
    network.load_state_dict(torch.load('network.pth'))
    network.eval()
    
    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_ = data
        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()

        pred_confidence, pred_box = network(images)

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_confidence_ = softmax(pred_confidence_, axis=1) # Adding softmax to the last as it is not part of the model architecture.
        pred_box_ = pred_box[0].detach().cpu().numpy()

        ###########################
        ann_boxes = ann_box[0].detach().cpu().numpy()
        # ann_boxes, pred_box_ = relative_to_absolute(ann_box=ann_boxes, pred_box=pred_box_, ann_confidence=ann_confidence_[0].numpy(), pred_confidence=pred_confidence_, boxs_default=boxs_default)
        ###########################
        
        # pred_confidence_, pred_box_, filtered_indices = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        nms_confidence, pred_confidence_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        
        #TODO: save predicted bounding boxes and classes to a txt file.
        #you will need to submit those files for grading this assignment

        #################### MAP related code #######################
        precision = []
        recall = []
        thres = 0.5
        precision, recall = update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default, precision,recall,thres)
        #############################################################
        
        visualize_pred("test", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_boxes, images_[0].numpy(), boxs_default, nms_confidence)
        # visualize_pred("test", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_boxes, images_[0].numpy(), boxs_default)
        cv2.waitKey(3000)



