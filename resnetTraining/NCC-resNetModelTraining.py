"""
__author__ : Kumar Shubham
__desc__   : This code is for experimenting resnet NN model with NCC casual model 
__date__   : 19-04-2019
resnet code is take and modified from original slim code
"""
import tensorflow as tf 
import numpy as np 
from tensorflow.python.platform import gfile
from PIL import Image
from datasets import imagenet

from datasets import dataset_factory
from nets import resnet_v1,nets_factory
from preprocessing import vgg_preprocessing ## vgg_preprocessing is the default pre processing for all resnet model
import numpy as np
import os 
slim = tf.contrib.slim
from bs4 import BeautifulSoup
from collections import Counter
import random
# try:
# 	import urllib2
# except ImportError:
# 	import urllib.request as urllib

class trainResnetWeight(object):
	def __init__ (self,modelPath,optimizer=tf.train.RMSPropOptimizer,split=0.75,batchSize = 16,dataAnnotationDir = "",noEpoch = 5,imageFolder= "",learningRate=0.001,decayPerIter="",train=True,vocClass=20,modelName="resnet_v1_50"):
		self.modelPath=modelPath ## we are using the pb file for loading the variables 
		self.modelName = modelName
		self.imageSize = resnet_v1.resnet_v1_50.default_image_size
		self.vocClass=20
		self.optimizer = optimizer
		self.batchSize = batchSize
		self.ilr = learningRate
		self.decayIter = decayPerIter
		self.isNCCTrain = train
		self.gpuPercent=0.3
		self.splitRatio = split
		self.imageFolder = imageFolder
		self.annotateDir = dataAnnotationDir
		self.classLabel={'person': 0, 'chair': 1, 'car': 2, 'dog': 3, 'bottle': 4, 'cat': 5, 'bird': 6, 'pottedplant': 7, 'sheep': 8, 'boat': 9, 'aeroplane': 10, 'tvmonitor': 11, 'sofa': 12, 'bicycle': 13, 'horse': 14, 'motorbike': 15, 'diningtable': 16, 'cow': 17, 'train': 18, 'bus': 19}
		self.seed =  np.random.seed(1)
		self.train = train
		self.noEpoch = noEpoch
	def imagePreProcess(self):
		## function for doing image pre-processing
		batchImage = []
		# self.resBatchSize =tf.placeholder( tf.int32,name="resNetBatch") 
		self.resNetInput = tf.placeholder(tf.uint8,shape=[self.batchSize,300,300,3],name="resNetInput")
		listTensorProcess = tf.unstack(self.resNetInput)
		for imageContent in listTensorProcess:
			# img = i.eval()
			# imageContent =  np.asarray(Image.open(img))
			processImage= vgg_preprocessing.preprocess_image(imageContent,self.imageSize,self.imageSize,is_training=False)
			batchImage.append(processImage[np.newaxis,...])
		# self.processedImage = tf.expand_dims(processImage,0)
		self.processedImage = tf.concat(batchImage,axis=0)
		return self.processedImage

	def loadAnnotationFile(self):
		## loading the annotation file:
		#https://github.com/mprat/pascal-voc-python/blob/master/vocimgs.ipynb

		######Ahhhhhhhhhh so many for loops .. solve it 
		dataset = {}
		fileCounter = 0
		for root,dirs,files in os.walk(self.annotateDir):
			for elm in files[:16*100]:
				fileName = os.path.join(self.annotateDir,elm)
				with open(fileName,"r") as f:
					xml =f.readlines()
					xml = ''.join([line.strip('\t') for line in xml])
					annXml = BeautifulSoup(xml)
					oneHotEncode = [0]*self.vocClass
					fileCounter+=1
					print ("fileLoaded : %d "%(fileCounter))
					objs = annXml.findAll('object')
					for obj in objs:
						obj_names = obj.findChildren('name')
						for nameTag in obj_names:
							if nameTag.contents[0] in self.classLabel:
								oneHotEncode[ self.classLabel[nameTag.contents[0]] ] = 1
							else:
								continue
					dataset[annXml.filename.string] = oneHotEncode
		

		self.data = dataset
		imageList = list(dataset.keys())
		random.shuffle(imageList)
		trainList = imageList[:int(len(imageList)*self.splitRatio)]
		testList = imageList[int(len(imageList)*self.splitRatio):]
		self.trainData = [(image,dataset[image]) for image in trainList ] ## declaring train Data
		self.testData = [(image,dataset[image]) for image in testList ] ## declaring test Data 

	def networkGenResNet(self,imageProcessed):
		## this file point to the network of  the model being used.
		with slim.arg_scope(resnet_v1.resnet_arg_scope()):
			self.logits,self.endPoint = resnet_v1.resnet_v1_50(imageProcessed,num_classes=1000,is_training=False)
		self.probability = tf.nn.softmax(self.logits)

	def loadingVariable(self):
		## getting all the variable name to be restored 
		self.variablesToRestore = slim.get_variables_to_restore()
		self.initFn = slim.assign_from_checkpoint_fn(
		self.modelPath,
		slim.get_model_variables())



	def networkGenNCC(self):
		## this section deals with the graph for resnet model
		with tf.name_scope("resnetFeatureExt"): 
			self.NCCinputResFet = tf.placeholder(tf.float32,shape = [None,2048],name="inputVector")
			self.NCCclassLabel = tf.placeholder(tf.float32,shape = [None,20],name="classLabel") ## taking one hot encoded vector
			self.NCCnet1 = tf.layers.dense(self.NCCinputResFet,512, name = "embDense1")
			self.NCCnet1Relu = tf.nn.relu(self.NCCnet1)
			self.NCCnet2 = tf.layers.dense(self.NCCnet1Relu,512,name="embDense2")
			self.NCCnet2Relu = tf.nn.relu(self.NCCnet2)
			self.NCClogit = tf.layers.dense(self.NCCnet2Relu,self.vocClass,name="logit")
			self.NCCpredictions = tf.nn.sigmoid(self.NCClogit, name='predictions')

		with tf.name_scope("loss"):
			self.NCCloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.NCCclassLabel, logits = self.NCClogit))
			self.NCCtrainOp = self.optimizer(self.ilr).minimize(self.NCCloss)

		with tf.name_scope('accuracy'):
			self.NCCcorrectPredictions = tf.round(self.NCCpredictions)
			correctPredictions = tf.equal(tf.round(self.NCCpredictions), self.NCCclassLabel)
			self.NCCaccuracy = tf.reduce_mean(tf.cast(correctPredictions, "float"), name='accuracy')

		with tf.name_scope('num_correct'):
			correct = tf.equal(tf.round(self.NCCpredictions), self.NCCclassLabel)
			self.NCCnoCorrect = tf.reduce_sum(tf.cast(correct, 'float'))





	def buildResnetGraph(self):
		## building the resnet graph for the processing
		processedImage = self.imagePreProcess()
		prob = self.networkGenResNet(processedImage)
		self.loadingVariable() 


	def graphLoad(self):
		## given function create a set of graph and session variable so that feature ext and trainig can happen seperately

		gpuOptions = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpuPercent)
		config=tf.ConfigProto(gpu_options=gpuOptions)

		###### loading resnet graph ###############
		self.resNetGraph = tf.Graph()
		with self.resNetGraph.as_default():
			self.buildResnetGraph()
			
			# self.initFn()

		self.resNetSess = tf.Session(graph=self.resNetGraph,config=config)

		with self.resNetGraph.as_default():
			self.initFn(self.resNetSess)
		# with self.resNetSess as sess1:
		
		######### loading NCC graph ###############
		self.NCCRes = tf.Graph()
			
		self.NCCResSess = tf.Session(graph=self.NCCRes,config=config)
		
		with self.NCCRes.as_default():
			self.networkGenNCC()
			assert self.NCCinputResFet.graph is self.NCCRes
			if self.train:
				self.NCCResSess.run(tf.global_variables_initializer())
			else:
				pass



	def Run(self):
		## function for loading the graph for processing
		self.graphLoad()
		self.loadAnnotationFile() ## loading the training data 
			
		epc = 0
		while(epc < self.noEpoch):
			batchCnt = 0
			for i in range(0, len(self.trainData),self.batchSize):
				trainPart = self.trainData[i:i+self.batchSize]
				trainImage = [os.path.join(self.imageFolder,content[0]) for content in trainPart]
				trainLabel = [content[1] for content in trainPart]
				trainLabelMat = np.vstack(trainLabel)
				 
				dataNp = self.readData(trainImage)
				img,probabilities,feature = self.resNetSess.run([self.processedImage,self.probability,self.endPoint['global_pool']],feed_dict={self.resNetInput:dataNp})
				sqzFeature =  np.squeeze( np.squeeze(feature,axis =1),axis=1) ## squeezing 1,2 axis of the feature vector
		
				## running the training of second network seperatley 
				with self.NCCRes.as_default():
					loss,_ =  self.NCCResSess.run([self.NCCloss,self.NCCtrainOp], feed_dict = {self.NCCinputResFet:sqzFeature,self.NCCclassLabel:trainLabel})
				batchCnt+=1
				print ("training epoch : %d batch iter : %d loss : %f"%(epc,batchCnt,loss))
			epc+=1
	def readData(self,dataList):
	## read image from a list of file address 
		arr = []
		# data = ["./cat-pet-animal-domestic-104827.jpeg","./First_Student_IC_school_bus_202076.jpg"]
		data = dataList	
		for img in data:
			imagePt = Image.open(img)
			img = imagePt.resize((300,300))
			imgArr = np.asarray(img)
			arr.append(imgArr[np.newaxis,...])
		out =np.concatenate(arr,axis=0)
		return out
if __name__ =="__main__":
	obj = trainResnetWeight(modelPath=".\\slim-imagenet\\resnet_v1_50.ckpt",dataAnnotationDir="..\\pascal-voc\\VOCdevkit\\VOC2012\\Annotations",imageFolder="..\\pascal-voc\\VOCdevkit\\VOC2012\\JPEGImages")
	obj.Run()

