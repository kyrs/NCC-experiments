"""
__author__ : Kumar shubham
__desc__   : following code is used to make intervention in an image to check the proposed hypothesis.
__date__   : 06-05-2019
"""
import numpy as np 
import tensorflow as tf 
from bs4 import BeautifulSoup

import random
import cv2
import json
import pickle 
import os
from PIL import Image
import matplotlib.pyplot as plt

classMap = {'person': 0, 'chair': 1, 'car': 2, 'dog': 3, 'bottle': 4, 'cat': 5, 'bird': 6, 'pottedplant': 7, 'sheep': 8, 'boat': 9, 'aeroplane': 10, 'tvmonitor': 11, 'sofa': 12, 'bicycle': 13, 'horse': 14, 'motorbike': 15, 'diningtable': 16, 'cow': 17, 'train': 18, 'bus': 19}

def loadImageData(xmlFolderPath,imageFolderPath):
	## function to load image from given folder for testing the model
	for root,dirs,files in os.walk(xmlFolderPath):
		fileCounter=0
		for elm in files:
			fileName = os.path.join(xmlFolderPath,elm)
			with open(fileName,"r") as f:
				xml =f.readlines()
				xml = ''.join([line.strip('\t') for line in xml])
				annXml = BeautifulSoup(xml)
				fileCounter+=1
				print ("fileLoaded : %d "%(fileCounter))
				objs = annXml.findAll('object')
				img = annXml.filename.string
				fullPath = os.path.join(imageFolderPath,img)
				imagePt = Image.open(fullPath)

				imgArr = np.asarray(imagePt)
				newImgArr = imgArr.copy()
				orignalImage = imgArr.copy()
				contextImg = imgArr.copy()

				for obj in objs:
					obj_names = obj.findChildren('name')
					bndBox = obj.findChildren('bndbox')
					for classLabel,box in zip(obj_names,bndBox):
						label = classLabel.contents[0]
						minX = int(box.findChildren('xmin')[0].contents[0])
						minY = int(box.findChildren('ymin')[0].contents[0])
						maxX =int(box.findChildren('xmax')[0].contents[0])
						maxY = int(box.findChildren('ymax')[0].contents[0])
						if label in classMap:
							newImgArr[minY:maxY,minX:maxX,:] = 0
							
						else:
							pass
				contextImg[newImgArr!=0] =0
				newImg = Image.fromarray(contextImg)
				plt.imshow(newImg)

				plt.show()
				

if __name__ =="__main__":
	# imageFolderPath=""
	loadImageData(xmlFolderPath="..\\pascal-voc\\VOCdevkit\\VOC2012\\Annotations", imageFolderPath="..\\pascal-voc\\VOCdevkit\\VOC2012\\JPEGImages")

