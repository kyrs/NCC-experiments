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


## loding the model 
from NCC_resNetModelTraining import trainResnetWeight 



resnetModel = trainResnetWeight(modelPath=".\\slim-imagenet\\resnet_v1_50.ckpt",train=False,dataAnnotationDir="..\\pascal-voc\\VOCdevkit\\VOC2012\\Annotations",imageFolder="..\\pascal-voc\\VOCdevkit\\VOC2012\\JPEGImages",batchSize=3)


classMap = {'person': 0, 'chair': 1, 'car': 2, 'dog': 3, 'bottle': 4, 'cat': 5, 'bird': 6, 'pottedplant': 7, 'sheep': 8, 'boat': 9, 'aeroplane': 10, 'tvmonitor': 11, 'sofa': 12, 'bicycle': 13, 'horse': 14, 'motorbike': 15, 'diningtable': 16, 'cow': 17, 'train': 18, 'bus': 19}

def loadImageData(xmlFolderPath):
	## function to load image from given folder for testing the model
	dictOfImage = {}
	limit = 10000

	for root,dirs,files in os.walk(xmlFolderPath):
		fileCounter=0

		for elm in files[:limit]:
			fileName = os.path.join(xmlFolderPath,elm)
			with open(fileName,"r") as f:
				xml =f.readlines()
				xml = ''.join([line.strip('\t') for line in xml])
				annXml = BeautifulSoup(xml)
				fileCounter+=1
				print ("fileLoaded : %d "%(fileCounter))
				objs = annXml.findAll('object')
				img = annXml.filename.string
				
				# imagePt = Image.open(fullPath)

				# imgArr = np.asarray(imagePt)
				# newImgArr = imgArr.copy()
				# orignalImage = imgArr.copy()
				# contextImg = imgArr.copy()
				

				for obj in objs:
					obj_names = obj.findChildren('name')
					bndBox = obj.findChildren('bndbox')
					for classLabel,box in zip(obj_names,bndBox):
						label = classLabel.contents[0]
						minX = int(box.findChildren('xmin')[0].contents[0])
						minY = int(box.findChildren('ymin')[0].contents[0])
						maxX =int(box.findChildren('xmax')[0].contents[0])
						maxY = int(box.findChildren('ymax')[0].contents[0]) 
						# if label in classMap:
						# 	newImgArr[minY:maxY,minX:maxX,:] = 0
						# else:
						# 	pass

						if label in classMap:
							if label not in dictOfImage:
								dictOfImage[label] = {img : [(minY,maxY,minX,maxX)]}
							elif (label in dictOfImage) and (img in dictOfImage[label]):
								dictOfImage[label][img].append((minY,maxY,minX,maxX))
							elif (label in dictOfImage) and (img not in dictOfImage[label]):
								dictOfImage[label][img] = [(minY,maxY,minX,maxX)]
								
							else:
								pass
						else:
							pass
				# contextImg[newImgArr!=0] =0
				# newImg = Image.fromarray(contextImg)
				# plt.imshow(newImg)

				# plt.show()
	return dictOfImage
				
def imageVectorSample(imageFolderPath,dictOfImage,className):


	### model initialization
	resnetModel.graphLoad()


	imageForClass = dictOfImage[className]
	
	objImgFeat =[]
	origImgFeat = []
	contImgFeat = []

	for image in imageForClass:
		fullPath = os.path.join(imageFolderPath,image)
		imagePt = Image.open(fullPath)
		imagePt = imagePt.resize((300,300))
		imgArr = np.asarray(imagePt)
		objImg = imgArr.copy()
		orignalImage = imgArr.copy()
		contextImg = imgArr.copy()

		## iterating over the list value 
		for minY,maxY,minX,maxX in imageForClass[image]:
			contextImg[minY:maxY,minX:maxX,:] = 0 


		objImg[contextImg!=0] =0
		
		imageFrames = np.concatenate([orignalImage[np.newaxis,...], objImg[np.newaxis,...], contextImg[np.newaxis,...]],axis=0)
		
		img,probabilities,testFeature = resnetModel.resNetSess.run([resnetModel.processedImage,resnetModel.probability,resnetModel.endPoint['global_pool']],feed_dict={resnetModel.resNetInput:imageFrames})
		sqztestFeature =  np.squeeze( np.squeeze(testFeature,axis =1),axis=1) ## squeezing 1,2 axis of the feature vector

		## running the training of second network seperatley 
		with resnetModel.NCCRes.as_default():
			logit,feature =  resnetModel.NCCResSess.run([resnetModel.NCClogit,resnetModel.NCCnet2], feed_dict = {resnetModel.NCCinputResFet:sqztestFeature})		
		
		##### appending all the image feature 
		origImgFeat.append(feature[0])
		objImgFeat.append(feature[1])
		contImgFeat.append(feature[2])

	return origImgFeat,objImgFeat,contImgFeat


def loadImageclass(pickleFile,className):
	#####
	# 
	####
	with open(pickleFile,"rb") as pickleFileReader:
		NCCJsonDict = pickle.load(pickleFileReader)
	classBasedData = NCCJsonDict[className]
	# print(classBasedData)
	antiCasualFet = sorted(classBasedData, key = lambda i: i['prob'],reverse=True)
	antiCasidxList =[elm["idx"] for elm in antiCasualFet][:100]

	casualFet =sorted(classBasedData, key = lambda i: i['prob']) 
	casidxList =[elm["idx"] for elm in casualFet][:100]
	
	# print(antiCasidxList)
	# print(casidxList)
	return antiCasidxList,casidxList

def boxPlot(feat, orignalFeat, antiCasIdx,casIdx):
	## drawing the box plot for a given class

	
	featCasList = []
	featAntiCasList = []

	

	############### calculating for object feature vector #############
	for index in casIdx: 
		Num = 0
		Den = 0
		for obF1,origF1 in zip(feat,orignalFeat):
			Num+=np.abs(obF1[index] - origF1[index])
			Den+= np.abs(origF1[index])

	featCasList.append(Num/Den)


	############### calculating for object feature vector #############
	for index in antiCasIdx: 
		Num = 0
		Den = 0
		for obF1,origF1 in zip(feat,orignalFeat):
			Num+=np.abs(obF1[index] - origF1[index])
			Den+= np.abs(origF1[index])

	featAntiCasList.append(Num/Den)

	return featCasList,featAntiCasList

if __name__ =="__main__":
	# imageFolderPath=""
	outDict = loadImageData(xmlFolderPath="..\\pascal-voc\\VOCdevkit\\VOC2012\\Annotations")
	origImgFeat,objImgFeat,contImgFeat = imageVectorSample( imageFolderPath="..\\pascal-voc\\VOCdevkit\\VOC2012\\JPEGImages",dictOfImage=outDict,className="tvmonitor")
	antiCasidxList,casidxList = loadImageclass(pickleFile="..\\FeatureResNetMapNCC.pickle",className="tvmonitor")

	featCas,featAntiCas =  boxPlot(feat=objImgFeat, orignalFeat=origImgFeat, antiCasIdx=antiCasidxList,casIdx=casidxList)

	meanCas = np.mean(featCas)
	stdCas  = np.std(featCas)

	meanAnti = np.mean(featAntiCas)
	stdAnti = np.std(featAntiCas)

	fig, ax = plt.subplots()
	bar_width = 0.35

	opacity = 0.4
	error_config = {'ecolor': '0.3'}

	rects1 = ax.bar(1, meanCas, bar_width,
	                alpha=opacity, color='b',
	                yerr=stdCas, error_kw=error_config,
	                label='Casual')

	rects2 = ax.bar(1 + bar_width, meanAnti, bar_width,
	                alpha=opacity, color='r',
	                yerr=stdAnti, error_kw=error_config,
	                label='AntiCasual')
	plt.show()