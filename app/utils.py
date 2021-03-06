import os
import glob
import pathlib
import torch

import cv2
import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

import seaborn as sns
import numpy as np
import pandas as pd
import six

from random import randint

from sklearn.model_selection import train_test_split

from skimage.transform import resize

from fastai.vision.widgets import *

from fastai2.vision.all import *
from fastai2.basics import *


# ======================================= #
#               Parameters                #
# ======================================= #

# Image size for Resnet
img_size_target = 224

# Used by the Face Detector (Resnet SSD)
res_prototxt="external_models/deploy.prototxt.txt"
res_model = "external_models/res10_300x300_ssd_iter_140000.caffemodel"


# ======================================= #
#   Related to the class interpretation   #
# ======================================= #

age_classes_1_inv = {(1, 12): 0, (13, 18): 1, (19, 22): 2, (23, 29): 3, (30, 34): 4,
                     (35, 39): 5, (40, 44): 6, (45, 49): 7, (50, 59): 8, (60, 99): 9}

age_classes_2_inv = {(0, 2): 0, (4, 6): 1, (8, 13): 2, (15, 20): 3,
                     (25, 32): 4, (38, 43): 5,  (48, 53): 6, (60, 99): 7}


age_classes_1 = {0: (1, 12), 1: (13, 18),  2: (19, 22), 3: (23, 29), 4: (30, 34),
                 5: (35, 39),6: (40, 44), 7: (45, 49), 8: (50, 59), 9: (60, 99)}

age_classes_2 = {0: (0, 2), 1: (4, 6), 2: (8, 13), 3: (15, 20),
                 4: (25, 32), 5: (38, 43), 6: (48, 53), 7: (60, 99)}


mask_classes = {'True': "With mask", 'False': "Without mask"}
mask_classes_inv = {"With mask": 'True', "Without mask": 'False'}


gender_classes = {0: "Female", 1: "Male"}
gender_classes_inv = {"Female": 0, "Male": 1}


def get_age_range(age_num, age_classes=age_classes_2):
    if age_num in age_classes:
        return age_classes[age_num]
    return -1


def get_mask_stat(is_masked, m_classes=mask_classes):
    return m_classes[is_masked]


def get_gender(cls, gend_classes=gender_classes):
    return gend_classes[cls]


# ======================================= #
#   Related to the class interpretation   #
# ======================================= #

def get_face_coordinates(image_file, net, min_confidence):
    image = cv2.imread(image_file)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence >= min_confidence:
            # compute the (x, y)-coordinates of the bounding box for the # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            w1, h1 = endX - startX, endY - startY

            l_startX = max(0, startX - w1 // 5)
            l_starty = max(0, startY - h1 // 5)
            l_endX = min(w, endX + w1 // 5)
            l_endY = min(h, endY + h1 // 5)

            # boxes.append((startX, startY, endX, endY))
            boxes.append((l_startX, l_starty, l_endX, l_endY))

    if len(boxes) == 0:
        return (0, 0, w, h)
    if len(boxes) == 1:
        return boxes[0]

    retained_box = boxes[0]
    for box in boxes:
        if box[2] - box[0] > retained_box[2] - retained_box[0]:
            retained_box = box
    return retained_box


def process_image(image_file, net):
    dirname = os.path.dirname(image_file)
    basename_without_ext = os.path.splitext(os.path.basename(image_file))[0]
    extension = os.path.splitext(os.path.basename(image_file))[1]
    new_file = dirname + basename_without_ext + '_cropped' + extension

    (x1, y1, x2, y2) = get_face_coordinates(image_file, net, 0.6)
    image = cv2.imread(image_file)
    crop_img = image[y1:y2, x1:x2]
    final_img = image_resize(crop_img, width=400)
    cv2.imwrite(new_file, final_img)
    return new_file


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


