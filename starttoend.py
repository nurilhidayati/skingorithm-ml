import tensorflow as tf
from keras.models import load_model
import segmentation_models as sm
import matplotlib.image as img
import numpy as np
import base64
import io
from PIL import Image
import os
import cv2
import sys

fspath = ''
acpath = ''
bspath = ''
wrpath = ''
eapath = ''
pepath = ''

input = 

#function resize dan crop
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def get_cropped_image_if_2_eyes(image_path):
    img = np.array(image_path)
    faces = face_cascade.detectMultiScale(img, 1.3,5)
    for (x,y,w,h) in faces:
        roi_color = img[y:y+h, x:x+(w)]
        roi_color = cv2.resize(roi_color, (256, 256))
        eyes = eye_cascade.detectMultiScale(roi_color)
        if len(eyes) >= 2:
            return roi_color

#Decode base64 to image
input = base64.b64decode(input)
input = Image.open(io.BytesIO(input))
input = get_cropped_image_if_2_eyes(input)
input = input/255.0

finalinput = np.expand_dims(input, 0)

def modelrun(fs_dict = fspath, ac_dict = acpath, bs_dict = bspath, wr_dict = wrpath, ea_dict = eapath, pe_dict = pepath):
    #FaceSkin
    FS_model = load_model(fs_dict, custom_objects={'binary_crossentropy_plus_jaccard_loss':sm.losses.bce_jaccard_loss, 'iou_score':sm.metrics.iou_score}, compile=False)
    FS_prediction = FS_model.predict(finalinput)
    FS_array = np.array(FS_prediction)
    FS = np.sum(FS_array > 0.5)

    #Acne 
    AC_model = load_model(ac_dict, custom_objects={'binary_crossentropy_plus_jaccard_loss':sm.losses.bce_jaccard_loss, 'iou_score':sm.metrics.iou_score}, compile=False)
    AC_prediction = AC_model.predict(finalinput)
    AC_array = np.array(AC_prediction)
    AC = np.sum(AC_array > 0.5)

    #Black Spot 
    BS_model = load_model(bs_dict, custom_objects={'binary_crossentropy_plus_jaccard_loss':sm.losses.bce_jaccard_loss, 'iou_score':sm.metrics.iou_score}, compile=False)
    BS_prediction = BS_model.predict(finalinput)
    BS_array = np.array(BS_prediction)
    BS = np.sum(BS_array > 0.5)

    #Wrinkle
    WR_model = load_model(wr_dict, custom_objects={'binary_crossentropy_plus_jaccard_loss':sm.losses.bce_jaccard_loss, 'iou_score':sm.metrics.iou_score}, compile=False)
    WR_prediction = WR_model.predict(finalinput)
    WR_array = np.array(WR_prediction)
    WR = np.sum(WR_array > 0.5)

    #Eye Area
    EA_model = load_model(ea_dict, custom_objects={'binary_crossentropy_plus_jaccard_loss':sm.losses.bce_jaccard_loss, 'iou_score':sm.metrics.iou_score}, compile=False)
    EA_prediction = EA_model.predict(finalinput)
    EA_array = np.array(EA_prediction)
    EA = np.sum(EA_array > 0.5)

    #Panda Eye
    PE_model = load_model(pe_dict, custom_objects={'binary_crossentropy_plus_jaccard_loss':sm.losses.bce_jaccard_loss, 'iou_score':sm.metrics.iou_score}, compile=False)
    PE_prediction = PE_model.predict(finalinput)
    PE_array = np.array(PE_prediction)
    PE = np.sum(PE_array > 0.5)

    return FS, AC, BS, WR, EA, PE

FS, AC, BS, WR, EA, PE = modelrun() 

#Persen 
acne = 100 - ((AC/FS) * 100)
print(acne)
bspot = 100 -((BS/FS) * 100)
print(bspot)
WR = 100 - ((WR/FS) * 100)
print(WR)
peye = 100 - ((PE/EA) * 100)
print(peye)

