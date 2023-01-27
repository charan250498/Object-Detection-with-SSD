import os
import random
import numpy as np

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




def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    #TODO: write a loss function for SSD
    #
    #For confidence (class labels), use cross entropy (F.cross_entropy)
    #You can try F.binary_cross_entropy and see which loss is better
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.
    pass




class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        
        #TODO: define layers
        channels = [64,128,256,512]
        self.layers_set_1 = [nn.Conv2d(3,64,3,2,bias=True), nn.BatchNorm2d(64), nn.ReLU()]
        for channel in channels:
            self.layers_set_1.extend([nn.Conv2d(channel, channel, 3, 1, bias=True, padding=1, padding_mode='zeros'), nn.BatchNorm2d(channel), nn.ReLU()] * 2)
            if channel != 512:
                self.layers_set_1.extend([nn.Conv2d(channel, channel * 2, 3, 2, bias=True), nn.BatchNorm2d(channel * 2), nn.ReLU()])
            else:
                self.layers_set_1.extend([nn.Conv2d(channel, int(channel / 2), 3, 2, bias=True), nn.BatchNorm2d(int(channel / 2)), nn.ReLU()])
        
        self.layers_set_1 = nn.Sequential(nn.ModuleList(self.layers_set_1))
        self.layers_set_2 = nn.Sequential(nn.ModuleList([nn.Conv2d(256, 256, 1, 1, bias=True), nn.BatchNorm2d(256), nn.ReLU(),
                                nn.Conv2d(256, 256, 3, 2, bias=True), nn.BatchNorm2d(256), nn.ReLU()]))
        self.layers_set_3 = nn.Sequential(nn.ModuleList([nn.Conv2d(256, 256, 1, 1, bias=True), nn.BatchNorm2d(256), nn.ReLU(),
                                nn.Conv2d(256, 256, 3, 1, bias=True), nn.BatchNorm2d(256), nn.ReLU()]))
        self.layers_set_4 = nn.Sequential(nn.ModuleList([nn.Conv2d(256, 256, 1, 1, bias=True), nn.BatchNorm2d(256), nn.ReLU(),
                                nn.Conv2d(256, 256, 3, 1, bias=True), nn.BatchNorm2d(256), nn.ReLU()]))

        self.set_1_conv_layer_1 = nn.Conv2d(256, 16, 3, 1, bias=True, padding=1, padding_mode='zeros')
        self.set_1_conv_layer_2 = nn.Conv2d(256, 16, 3, 1, bias=True, padding=1, padding_mode='zeros')

        self.set_2_conv_layer_1 = nn.Conv2d(256, 16, 3, 1, bias=True, padding=1, padding_mode='zeros')
        self.set_2_conv_layer_2 = nn.Conv2d(256, 16, 3, 1, bias=True, padding=1, padding_mode='zeros')

        self.set_3_conv_layer_1 = nn.Conv2d(256, 16, 3, 1, bias=True, padding=1, padding_mode='zeros')
        self.set_3_conv_layer_2 = nn.Conv2d(256, 16, 3, 1, bias=True, padding=1, padding_mode='zeros')

        self.set_4_conv_layer_1 = nn.Conv2d(256, 16, 1, 1, bias=True)
        self.set_4_conv_layer_2 = nn.Conv2d(256, 16, 1, 1, bias=True)

        self.softmax_layer = nn.Softmax(dim = 1) ###################### CHARAN: TODO - check if you need to use F.cross_entropy
        
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
        
        #TODO: define forward
        set_1_output = self.layers_set_1(x)

        # First branch
        # left
        set_2_output = self.layers_set_2(set_1_output)
        # mid
        dimension_10_output_left = self.set_1_conv_layer_1(set_1_output)
        dimension_10_output_left = dimension_10_output_left.reshape((dimension_10_output_left.shape[0], 16, -1))
        # right
        dimension_10_output_right = self.set_1_conv_layer_2(set_1_output)
        dimension_10_output_right = dimension_10_output_right.reshape((dimension_10_output_right.shape[0], 16, -1))

        # Second branch
        # left
        set_3_output = self.layers_set_3(set_2_output)
        # mid
        dimension_5_output_left = self.set_2_conv_layer_1(set_2_output)
        dimension_5_output_left = dimension_5_output_left.reshape((dimension_5_output_left.shape[0], 16, -1))
        # right
        dimension_5_output_right = self.set_2_conv_layer_2(set_2_output)
        dimension_5_output_right = dimension_5_output_right.reshape((dimension_5_output_right.shape[0], 16, -1))

        # Third branch
        # left
        set_4_output = self.layers_set_4(set_3_output)
        # left - left
        dimension_1_output_left = self.set_4_conv_layer_1(set_4_output)
        dimension_1_output_left = dimension_1_output_left.reshape((dimension_1_output_left.shape[0], 16, -1))
        # left - right
        dimension_1_output_right = self.set_4_conv_layer_2(set_4_output)
        dimension_1_output_right = dimension_1_output_right.reshape((dimension_1_output_right.shape[0], 16, -1))
        # mid 
        dimension_3_output_left = self.set_3_conv_layer_1(set_3_output)
        dimension_3_output_left = dimension_3_output_left.reshape((dimension_3_output_left.shape[0], 16, -1))
        # right
        dimension_3_output_right = self.set_3_conv_layer_2(set_3_output)
        dimension_3_output_right = dimension_3_output_right.reshape((dimension_3_output_right.shape[0], 16, -1))

        bboxes = torch.cat((dimension_10_output_left, dimension_5_output_left, dimension_3_output_left, dimension_1_output_left), dim = 1)
        bboxes = torch.permute(bboxes, (0, 2, 1))
        bboxes = bboxes.reshape(bboxes.shape[0], 540, 4)

        confidence = torch.cat((dimension_10_output_right, dimension_5_output_right, dimension_3_output_right, dimension_1_output_right), dim = 1)
        confidence = torch.permute(confidence, (0, 2, 1))
        confidence = confidence.reshape(confidence.shape[0], 540, 4)
        
        #should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]

        print(confidence.shape)
        print(bboxes.shape)
        
        return confidence,bboxes










