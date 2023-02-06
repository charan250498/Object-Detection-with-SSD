import numpy as np
import cv2
from dataset import iou


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default, nms_confidence = None):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]

    # We need to change the box's relative values to absolute values here.
    try:
        if ann_confidence == None:
            testing = True
        else:
            testing = False
    except:
        testing = False

    #### COMMENTED FOR NOW AS THIS IS DONE EVEN BEFORE THE CALL TO THIS HAPPENS ####
    if not testing:
        ann_box[:, 0] = boxs_default[:, 2] * ann_box[:, 0] + boxs_default[:, 0]
        ann_box[:, 1] = boxs_default[:, 3] * ann_box[:, 1] + boxs_default[:, 1]
        ann_box[:, 2] = boxs_default[:, 2] * np.exp(ann_box[:, 2])
        ann_box[:, 3] = boxs_default[:, 3] * np.exp(ann_box[:, 3])

    pred_box[:, 0] = boxs_default[:, 2] * pred_box[:, 0] + boxs_default[:, 0]
    pred_box[:, 1] = boxs_default[:, 3] * pred_box[:, 1] + boxs_default[:, 1]
    pred_box[:, 2] = boxs_default[:, 2] * np.exp(pred_box[:, 2])
    pred_box[:, 3] = boxs_default[:, 3] * np.exp(pred_box[:, 3])
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    
    image = np.transpose(image_*255, (1,2,0)).astype(np.uint8)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    default_boxes = boxs_default[:, :]
    # if filtered_indices != None:
    #     default_boxes = boxs_default[filtered_indices]
    try:
        if None == nms_confidence:
            nms_confidence = pred_confidence.copy()
    except:
        pass
    
    #draw ground truth
    if not testing:
        for i in range(len(ann_confidence)):
            for j in range(class_num):
                if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j] ######################## Changed from 0.5 to 0.9
                    #TODO:
                    #image1: draw ground truth bounding boxes on image1
                    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                    
                    #you can use cv2.rectangle as follows:
                    #start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                    #end_point = (x2, y2) #bottom right corner
                    #color = colors[j] #use red green blue to represent different classes
                    #thickness = 2
                    #cv2.rectangle(image?, start_point, end_point, color, thickness)
                    start_point = (int((ann_box[i][0]-(ann_box[i][2]/2)) * 320), int((ann_box[i][1]-(ann_box[i][3]/2)) * 320))
                    end_point = (int((ann_box[i][0]+(ann_box[i][2]/2)) * 320), int((ann_box[i][1]+(ann_box[i][3]/2)) * 320))
                    color = colors[j] # CHARAN : TODO needs to be changed later.#use red green blue to represent different classes
                    thickness = 2
                    cv2.rectangle(image1, start_point, end_point, color, thickness)


                    start_point = (int((boxs_default[i][0]-(boxs_default[i][2]/2)) * 320), int((boxs_default[i][1]-(boxs_default[i][3]/2)) * 320))
                    end_point = (int((boxs_default[i][0]+(boxs_default[i][2]/2)) * 320), int((boxs_default[i][1]+(boxs_default[i][3]/2)) * 320))
                    color = colors[j] # CHARAN : TODO needs to be changed later.#use red green blue to represent different classes
                    thickness = 2
                    cv2.rectangle(image2, start_point, end_point, color, thickness)

    
    #pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if nms_confidence[i,j]>0.3:
                #TODO:
                #image3: draw network-predicted bounding boxes on image3
                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                start_point = (int((pred_box[i][0]-(pred_box[i][2]/2)) * 320), int((pred_box[i][1]-(pred_box[i][3]/2)) * 320))
                end_point = (int((pred_box[i][0]+(pred_box[i][2]/2)) * 320), int((pred_box[i][1]+(pred_box[i][3]/2)) * 320))
                color = colors[j]
                thickness = 2
                cv2.rectangle(image3, start_point, end_point, color, thickness)
                print("%d %d %d %d %d"%(j, start_point[0], start_point[1], end_point[0] - start_point[0], end_point[1] - start_point[1]))
            
            if pred_confidence[i,j]>0.3:
                start_point = (int((default_boxes[i][0]-(default_boxes[i][2]/2)) * 320), int((default_boxes[i][1]-(default_boxes[i][3]/2)) * 320))
                end_point = (int((default_boxes[i][0]+(default_boxes[i][2]/2)) * 320), int((default_boxes[i][1]+(default_boxes[i][3]/2)) * 320))
                color = colors[j] # CHARAN : TODO needs to be changed later.#use red green blue to represent different classes
                thickness = 2
                cv2.rectangle(image4, start_point, end_point, color, thickness)
    
    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    cv2.waitKey(1)
    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.



def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.2, threshold=0.5):
    #TODO: non maximum suppression
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #boxs_default -- default bounding boxes, [num_of_boxes, 8]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.

    conf1 = confidence_.copy() # Use this as A
    conf2 = np.zeros_like(confidence_) # Use this as B

    boxes = box_.copy()
    boxes[:, 0] = boxs_default[:, 2] * boxes[:, 0] + boxs_default[:, 0]
    boxes[:, 1] = boxs_default[:, 3] * boxes[:, 1] + boxs_default[:, 1]
    boxes[:, 2] = boxs_default[:, 2] * np.exp(boxes[:, 2])
    boxes[:, 3] = boxs_default[:, 3] * np.exp(boxes[:, 3])

    while(True):
        # 1. Find the max probability indices for each class
        max_prob_indices = [0 for i in range(3)]
        for i in range(len(confidence_)):
            for j in range(3):
                if conf1[max_prob_indices[j]][j] < conf1[i][j]:
                    max_prob_indices[j] = i
        # 2. Check if they are greater than threshold or not.
        if conf1[max_prob_indices[0]][0] < threshold and conf1[max_prob_indices[1]][1] < threshold and conf1[max_prob_indices[2]][2] < threshold:
            break
        # 3. Move the max values from A to B.
        conf2[max_prob_indices[0]][0] = conf1[max_prob_indices[0]][0]
        conf2[max_prob_indices[1]][1] = conf1[max_prob_indices[1]][1]
        conf2[max_prob_indices[2]][2] = conf1[max_prob_indices[2]][2]
        conf1[max_prob_indices[0]][0] = 0
        conf1[max_prob_indices[1]][1] = 0
        conf1[max_prob_indices[2]][2] = 0
        # 4. for all boxes in A check IOU with the ones in B, and remove those which overlap.
        for i in range(len(conf1)):
            for class_index in range(3):
                j = max_prob_indices[class_index]
                if i == j:
                    continue
                box_in_A = np.array([boxes[i][0]-boxes[i][2]/2, boxes[i][1]-boxes[i][3]/2, boxes[i][0]+boxes[i][2]/2, boxes[i][1]+boxes[i][3]/2])
                iou_value = iou(np.concatenate((np.zeros(4), box_in_A), axis=0).reshape((-1,8)), boxes[j][0]-boxes[j][2]/2, boxes[j][1]-boxes[j][3]/2, boxes[j][0]+boxes[j][2]/2, boxes[j][1]+boxes[j][3]/2)
                if iou_value > overlap:
                    conf1[i][0] = 0
                    conf1[i][1] = 0
                    conf1[i][2] = 0
                    conf1[i][3] = 1
    
    return conf2, confidence_

def generate_mAP():
    #TODO: Generate mAP
    pass








