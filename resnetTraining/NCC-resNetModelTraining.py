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
slim = tf.contrib.slim

# try:
# 	import urllib2
# except ImportError:
# 	import urllib.request as urllib

class trainResnetWeight(object):
	def __init__ (self,modelPath,modelName="resnet_v1_50",batchSize=2):
		self.modelPath=modelPath ## we are using the pb file for loading the variables 
		self.modelName = modelName
		self.imageSize = resnet_v1.resnet_v1_50.default_image_size
		# self.imageFile = []

	def imagePreProcess(self):
		batchImage = []
		self.resNetInput = tf.placeholder( tf.uint8,shape=[2,300,300,3],name="resNetInput")
		listTensorProcess = tf.unstack(self.resNetInput)
		for imageContent in listTensorProcess:
			# img = i.eval()
			# imageContent =  np.asarray(Image.open(img))
			processImage= vgg_preprocessing.preprocess_image(imageContent,self.imageSize,self.imageSize,is_training=False)
			batchImage.append(processImage[np.newaxis,...])
		# self.processedImage = tf.expand_dims(processImage,0)
		self.processedImage = tf.concat(batchImage,axis=0)
		return self.processedImage

	def networkGen(self,imageProcessed):
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

	def buildResnetGraph(self):
		## building the resnet graph for the processing
		processedImage = self.imagePreProcess()
		prob = self.networkGen(processedImage)
		self.loadingVariable() 
	def Run(self):
		## function for running the model for any given image
		with tf.Session() as sess:
			self.buildResnetGraph()
			self.initFn(sess) ## initializing the resnet model
			# self.imageFile = 
			dataNp = self.readData()
			
			img,probabilities,feature = sess.run([self.processedImage,self.probability,self.endPoint['global_pool']],feed_dict={self.resNetInput:dataNp})
			
			sqzFeature =  np.squeeze( np.squeeze(feature,axis =1),axis=1) ## squeezing 1,2 axis of the feature vector
			probabilities = probabilities[1, 0:]
			sortedInds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
			names = imagenet.create_readable_names_for_imagenet_labels()
			for i in range(5):
				index = sortedInds[i]
				# Shift the index of a class name by one. 
				print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index+1]))

			print(sqzFeature.shape)

	def readData(self):
	## read image from a list of file address 
		arr = []
		data = ["./cat-pet-animal-domestic-104827.jpeg","./First_Student_IC_school_bus_202076.jpg"]	
		for img in data:
			imagePt = Image.open(img)
			img = imagePt.resize((300,300))
			imgArr = np.asarray(img)
			arr.append(imgArr[np.newaxis,...])
		out =np.concatenate(arr,axis=0)
		return out
if __name__ =="__main__":
	obj = trainResnetWeight(modelPath=".\\slim-imagenet\\resnet_v1_50.ckpt")
	obj.Run()


