from pathlib import Path
import numpy as np
from sklearn import metrics
from sklearn.utils import Bunch

import itertools

from sklearn.model_selection import cross_val_score, train_test_split

from skimage.io import imread
from skimage.transform import resize
import pickle


#Load images in structured directory like it's sklearn sample dataset
def load_image_files(container_path, dimension=(250, 250, 3)):
    """
    Load image files with categories as subfolder names 
    which performs like scikit-learn sample dataset
    
    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to
        
    Returns
    -------
    Bunch
    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data, target=target, target_names=categories, images=images, DESCR=descr)


import numpy as np
import cv2
import imutils
import random
import os

def crop_and_grayscale_and_resize(img):
	(h,w,n) = img.shape
	img = img[0:h, 300:w]
	
	(h,w,n) = img.shape
	img = img[0:h, 0:w-600]
	
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	scale_percent = 10 # percent of original size
	width = int(gray_img.shape[1] * scale_percent / 100)
	height = int(gray_img.shape[0] * scale_percent / 100)
	dim = (width, height) 
	resized = cv2.resize(gray_img, dim, interpolation = cv2.INTER_AREA)
	return resized
	
def adaptive_HE(img):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	img_out = clahe.apply(img)
	return img_out

def img_estim(img, thrshld=127):
    is_light = np.mean(img) > thrshld
    return 'light' if is_light else 'dark'
	
def contrast_stretch(img):	
	minmax_img = np.zeros((img.shape[0],img.shape[1]),dtype = 'uint8')
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			minmax_img[i,j] = 255*(img[i,j]-np.min(img))/(np.max(img)-np.min(img))
	return minmax_img
	
img = cv2.imread('input/img.jpg')
img = crop_and_grayscale_and_resize(img)
img = adaptive_HE(img)
choice = img_estim(img)
if choice == 'dark':
	img = contrast_stretch(img)
cv2.imwrite('test/test_images/img.jpg',img)
#print('done')


image_dataset = load_image_files("test/")

X_test = image_dataset.data

sclf_identify = pickle.load(open('stack_model_identify.sav', 'rb'))
sclf_grade = pickle.load(open('stack_model_grade.sav', 'rb'))

pred = sclf_identify.predict(X_test)
if pred[0] == 1:
	print('DR detetced')
	grade = sclf_grade.predict(X_test)
	print('Grade {} DR detetced'.format(grade[0]+1))

else:
	print('DR not deteted')
