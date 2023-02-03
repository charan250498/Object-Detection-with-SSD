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
import numpy as np
import os
import cv2

def clip(value):
    if value < 0:
        return 0
    elif value > 1:
        return 1
    else:
        return value

#generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    #TODO:
    #create an numpy array "boxes" to store default bounding boxes
    #you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    #the first dimension means number of cells, 10*10+5*5+3*3+1*1
    #the second dimension 4 means each cell has 4 default bounding boxes.
    #their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    #where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    #for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    #the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]           
    num_layers = 4
    num_scales = 4
    sqrt2 = 1.41421356
    boxes = np.zeros((10*10+5*5+3*3+1*1,4,8))
    index = 0
    for layer in layers:
        for i in range(layer*layer):
            row = i//layer + 1
            col = i%layer + 1
            layer_index = layers.index(layer)
            x_center = row/(2*layer)
            y_center = col/(2*layer)
            width = clip(small_scale[layer_index])
            height = clip(small_scale[layer_index])
            boxes[index+i][0] = [x_center, y_center, width, height, clip(x_center-width/2), clip(y_center-height/2), clip(x_center+width/2), clip(y_center+height/2)]
            width = clip(large_scale[layer_index])
            height = clip(large_scale[layer_index])
            boxes[index+i][1] = [x_center, y_center, width, height, clip(x_center-width/2), clip(y_center-height/2), clip(x_center+width/2), clip(y_center+height/2)]
            width = clip(large_scale[layer_index]*sqrt2)
            height = clip(large_scale[layer_index]/sqrt2)
            boxes[index+i][2] = [x_center, y_center, width, height, clip(x_center-width/2), clip(y_center-height/2), clip(x_center+width/2), clip(y_center+height/2)]
            width = clip(large_scale[layer_index]/sqrt2)
            height = clip(large_scale[layer_index]*sqrt2)
            boxes[index+i][3] = [x_center, y_center, width, height, clip(x_center-width/2), clip(y_center-height/2), clip(x_center+width/2), clip(y_center+height/2)]
        index += layer*layer

    boxes = boxes.reshape(((10*10+5*5+3*3+1*1)*4,8))
    return boxes


#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
#you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)



def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,gx,gy,gw,gh):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    #compute iou between the default bounding boxes and the ground truth bounding box
    x_min,y_min,x_max,y_max = gx-(gw/2), gy-(gh/2), gx+(gw/2), gy+(gh/2)
    ious = iou(boxs_default, x_min,y_min,x_max,y_max)
    
    ious_true = ious>threshold
    #TODO:
    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    for i in range(len(ann_confidence)):
        if ious_true[i]:
            ann_confidence[i][-1] = 0
            ann_confidence[i][cat_id] = 1

    for i in range(len(ann_box)):
        if ious_true[i]:
            ann_box[i][0] = (gx - boxs_default[i][0])/boxs_default[i][2]
            ann_box[i][1] = (gy - boxs_default[i][1])/boxs_default[i][3]
            ann_box[i][2] = np.log(gw/boxs_default[i][2])
            ann_box[i][3] = np.log(gh/boxs_default[i][3])

    # Previous code.
    # ann_box[ious_true][:, 0] = (gx - boxs_default[ious_true][:, 0])/boxs_default[ious_true][:, 2]
    # ann_box[ious_true][:, 1] = (gy - boxs_default[ious_true][:, 1])/boxs_default[ious_true][:, 3]

    # ann_box[ious_true][:, 2] = np.log(gw/boxs_default[ious_true][:, 2])
    # ann_box[ious_true][:, 3] = np.log(gh/boxs_default[ious_true][:, 3])
    
    ious_true = np.argmax(ious)
    #TODO:
    #make sure at least one default bounding box is used
    #update ann_box and ann_confidence (do the same thing as above)
    ann_confidence[ious_true][cat_id] = 1
    ann_confidence[ious_true][-1] = 0

    ann_box[ious_true][0] = (gx - boxs_default[ious_true][0])/boxs_default[ious_true][2]
    ann_box[ious_true][1] = (gy - boxs_default[ious_true][1])/boxs_default[ious_true][3]

    ann_box[ious_true][2] = np.log(gw/boxs_default[ious_true][2])
    ann_box[ious_true][3] = np.log(gh/boxs_default[ious_true][3])

    return ann_box, ann_confidence



class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, image_size=320):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size

        self.transform = transforms.Compose([transforms.ToTensor()])
        
        #notice:
        #you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train
        if self.train:
            self.img_names = self.img_names[:int(0.9*len(self.img_names))]
        else:
            self.img_names = self.img_names[int(0.9*len(self.img_names)):]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        
        img_name = self.imgdir+self.img_names[index]
        ann_name = self.anndir+self.img_names[index][:-3]+"txt"
        
        #TODO:
        #1. prepare the image [3,320,320], by reading image "img_name" first.
        image = cv2.imread(img_name)
        height, width, channel = image.shape
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
        image = self.transform(image)

        if channel == 1:
            image = torch.cat((image, image, image), axis=2)
            print("Channel was 1")

        #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        fhand = open(ann_name, "r")
        contents = fhand.read()

        contents = contents.split("\n")
        for content in contents:
            if len(content) < 1: # When you split the empty string after \n also get added to the list, so this if case will eliminate that case.
                continue
            content = content.split(" ")

        # Normalized wrt to height and width
            class_id = int(content[0])
            gx,gy,gw,gh = [(float(content[1]) + float(content[3])/2)/width, (float(content[2]) + float(content[4])/2)/height, float(content[3])/width, float(content[4])/height]

        #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
            ann_box, ann_confidence = match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,gx,gy,gw,gh)

        #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
        ##########################################
        # CHARAN:TODO - DO DATA AUGMENTATION LATER
        ##########################################
        
        #to use function "match":
        #match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
        #where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
        
        #note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        #For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)
        
        
        return image, ann_box, ann_confidence
