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

def rotate_rand(img):
	angle = random.randint(0,180) 
	rotated = imutils.rotate(img, angle)
	return rotated
	
def flip_rand(img):
	orientation = np.random.randint(2, size=1)
	flipped = cv2.flip(img, orientation[0])
	return flipped
	
path = "Grading/dataset"
for i in os.listdir(path):
	l = len(os.listdir(path+'/'+i))
	names = os.listdir(path+'/'+i)
	for name in names:
		print(name)
		img_path = path+'/'+i+'/'+name
		img = cv2.imread(img_path)
		img = crop_and_grayscale_and_resize(img)
		img = adaptive_HE(img)
		choice = img_estim(img)
		if choice == 'dark':
			img = contrast_stretch(img)
		cv2.imwrite('processed_data/'+i+'/'+name,img)
		print('done')
	count = len(os.listdir('processed_data/'+i))
	while count <= 150:
		im_no = random.randint(1,len(os.listdir('processed_data/'+i)))-1
		imnames = os.listdir('processed_data/'+i)
		imname = imnames[im_no]
		print(im_no,imname)
		im = cv2.imread('processed_data/'+i+'/'+imname)
		rotated = rotate_rand(im)
		flipped = flip_rand(im)
		rotate_name = str(count)
		cv2.imwrite('processed_data/'+i+'/'+rotate_name+'.jpg',rotated)
		flip_name = str(count+1)
		cv2.imwrite('processed_data/'+i+'/'+flip_name+'.jpg',flipped)
		count = len(os.listdir('processed_data/'+i))