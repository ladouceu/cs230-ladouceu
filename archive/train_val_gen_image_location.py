"""Train and Val Input Pipeline Generator.

This module contains the code for the generator of object and images used 
by the siamese model of the same name for training and validation. The user of
this code should feed this generator into tf.data.Dataset.from_generator() to
create a tf dataset to train the siamese model.

"""

import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

def get_object_location(row, image, target_shape):
    id_, x, y, w, h = row
    image_height,image_width = image.shape[0],image.shape[1]
    id_ = int(id_)
    x = int(max(x,0))
    y = int(max(y,0))
    w = int(min(w,image_width-x))
    h = int(min(h,image_height-y))
    if y == -1: y = 0
    if x == -1: x = 0
    obj = image[y:y+h,x:x+w]
    obj = cv2.resize(obj, dsize=target_shape)#, interpolation=cv2.INTER_CUBIC) #TODO : should I keep this interpolation?
    obj = obj.astype(np.float32)
    obj /= 255.
    return obj,np.array([x/image_width,y/image_height,w/image_width,h/image_height])

def triplet_gen(mode, batch_size, target_shape):
    # gt file header
    header = {"frame":0, "id":1, "x":2, "y":3, "w":4, "h":5, "x1":6, "x2":7, "conf":9}

    # get the list of gt files
    gt_files = sorted(glob.glob("data/train/*/*/gt.txt"))
    if mode == "train":
        gt_files = gt_files[:-1]
    elif mode == "val":
        gt_files = gt_files[-1:]
    else:
        raise RuntimeError("Invalid mode. It must be either 'train' or 'val'.")

    def gen():
        for gt_file in gt_files:
            gt = np.loadtxt(gt_file,delimiter=",")
            frames = np.unique(gt[:,header["frame"]])
            for frame in frames:
                image_file = os.path.join(os.sep.join(gt_file.split(os.sep)[:-2]),"img1/{:06d}.jpg".format(int(frame)))
                image = cv2.imread(image_file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                gt_frame = gt[gt[:,header["frame"]] == frame]
    #             plt.figure(figsize=(20,20))
    #             plt.imshow(image)
    #             plt.show()
                for row in gt_frame:
                    positive_data = gt[gt[:,header["id"]] == row[header["id"]]]
                    positive_index = np.random.choice(len(positive_data), size=1)[0]
                    positive_row = positive_data[positive_index]
                    positive_frame = positive_row[header["frame"]]
                    positive_image_file = os.path.join(os.sep.join(gt_file.split(os.sep)[:-2]),"img1/{:06d}.jpg".format(int(positive_frame)))
                    positive_image = cv2.imread(positive_image_file)
                    positive_image = cv2.cvtColor(positive_image, cv2.COLOR_BGR2RGB)
                    negative_data = gt_frame[gt_frame[:,header["id"]] != row[header["id"]]]
                    negative_index = np.random.choice(len(negative_data), size=1)[0]
                    negative_row = negative_data[negative_index]
                    anchor = get_object_location(row[1:6], image, target_shape)
                    positive = get_object_location(positive_row[1:6], positive_image, target_shape)
                    negative = get_object_location(negative_row[1:6], image, target_shape)
                    yield anchor,positive,negative
    return gen
