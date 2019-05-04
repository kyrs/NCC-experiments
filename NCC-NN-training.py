"""
__Author__ : Kumar Shubham 
__desc__   : file for training an NCC model based on the data which has been genereated
"""
import tensorflow as tf 
import json
import os 
from collections import Counter
import random
import numpy as np 
import pickle
class NCCTrain(object):
	def __init__(self,fileName,trainSplitRatio=0.7,saveDir="./model",sizeEmbLayer=100,sizeClassfLayer=100,dropOutRatio=0.25,iterVal=25,batchSize=256,activation=tf.nn.relu,batchNorm=True,optimizer =tf.train.RMSPropOptimizer,intLrRate=0.0001):
		"""
		fileName : file to be processed
		trainSplitRatio : ration of train :test data to consider for training v/s testing
		saveDir      : dir to save log and actual model 
		sizeEmbLayer : no of neuron in each emb layer 
		sizeclassFlayr : no of neuron in classification layer
		dropOutRatio : amount of dropout to consider type List
		iterVal         : no of iteration to run(epoch)
		batchSize    : size of each batch
		activation   : activation layer to use
		batchNorm 	 : using batch normalization or not 
		optimizer 	 : optimizer for the network
		intLrRate	 : int learning rate of the trainig process 
		"""
		self.fileName = fileName
		self.ratio = trainSplitRatio
		self.sizeEmbLayer = sizeEmbLayer
		self.sizeClassfLayer = sizeClassfLayer
		self.dropout = dropOutRatio
		self.maxIter = iterVal
		self.saveDir = saveDir
		self.batchSize = batchSize
		self.activation = activation
		self.optimizer = optimizer
		self.batchNorm = batchNorm
		self.ilr  = intLrRate


		############# path for model and summary  #############
		self.modelAdd   = os.path.join(self.saveDir,"model")
		self.summaryAdd = os.path.join(self.saveDir,"summary")

		if os.path.isdir(self.modelAdd):
			pass
		else:
			os.mkdir(self.modelAdd)

		if os.path.isdir(self.summaryAdd):
			pass
		else:
			os.mkdir(self.summaryAdd)

		self.modelAdd = self.modelAdd+"/NCCModelEncoder"
		self.summaryAdd = self.summaryAdd+"/summaryWriter"
	def buildNetwork(self):
		## defining the network Of NCC
		self.xVal = tf.placeholder(tf.float32,shape=[None,None,1], name="xVal")
		self.yVal = tf.placeholder(tf.float32,shape=[None,None,1], name="yVal")
		self.NCCLabel = tf.placeholder(tf.float32,shape=[None,1], name="NCCLabel")
		self.keepProb = tf.placeholder(tf.float32, name="keepProb")

		## for summary 
		self.avgTrainLoss = tf.placeholder(tf.float32, name="avgTrainLoss")
		self.avgTestLoss = tf.placeholder(tf.float32, name="avgTestLoss")

		self.isTrain = tf.placeholder(tf.bool, name="isTrain")
		self.concateVal = tf.concat([self.xVal,self.yVal],2,name="concatedValue")#concatenating the values
		

		with tf.name_scope("embededLayer-1") as scope: ## using embeded layer
			## refer http://ruishu.io/2016/12/27/batchnorm/
			self.embLayer1Dense = tf.layers.dense(self.concateVal,self.sizeEmbLayer, name = "embDense1")
			self.embLayer1Norm =  tf.layers.batch_normalization(self.embLayer1Dense, training=self.isTrain, name ="batchnorm1" )
			self.embLayer1Relu =  tf.nn.relu(self.embLayer1Norm)
			self.emblayer1 =  tf.nn.dropout(self.embLayer1Relu , self.keepProb,name = "embdroput1")
			

		with tf.name_scope("embededLayer-2") as scope: ## using embeded layer
			## refer http://ruishu.io/2016/12/27/batchnorm/
			self.embLayer2Dense = tf.layers.dense(self.emblayer1,self.sizeEmbLayer, name = "embDense2")
			self.embLayer2Norm = tf.layers.batch_normalization(self.embLayer2Dense, training=self.isTrain, name ="batchnorm2" )
			self.embLayer2Relu =  tf.nn.relu(self.embLayer2Norm)
			self.emblayer2 =  tf.nn.dropout(self.embLayer2Relu , self.keepProb,name = "embdroput2")
			

			
		self.finalEmbLayer = tf.reduce_mean(self.emblayer2,axis=1, name="representation")## for getting the final rep

		with tf.name_scope("classLayer-1") as scope: ## using classification layer
			## refer http://ruishu.io/2016/12/27/batchnorm/
			self.classLayer1Dense = tf.layers.dense(self.finalEmbLayer,self.sizeClassfLayer, name = "classfDens1")
			self.classLayer1Norm = tf.layers.batch_normalization(self.classLayer1Dense, training=self.isTrain, name ="classfbatchnorm1" )
			self.classLayer1Relu =  tf.nn.relu(self.classLayer1Norm)
			self.classLayer1 =  tf.nn.dropout(self.classLayer1Relu , self.keepProb,name = "classdroput1")
			


		with tf.name_scope("classLayer-2") as scope: ## using classification layer
			## refer http://ruishu.io/2016/12/27/batchnorm/
			self.classLayer2Dense = tf.layers.dense(self.classLayer1,self.sizeClassfLayer, name = "classfDense2")
			self.classLayer2Norm = tf.layers.batch_normalization(self.classLayer2Dense, training=self.isTrain, name ="classfbatchnorm2" )
			self.classLayer2Relu =  tf.nn.relu(self.classLayer2Norm)
			self.classLayer2 =  tf.nn.dropout(self.classLayer2Relu , self.keepProb,name = "classdroput2")
			
			
		self.logits = tf.layers.dense(self.classLayer2,1,name = "logits")
		self.prob = tf.nn.sigmoid(self.logits)

		with tf.name_scope("loss") as scope : # defining the loss function
			# self.loss = tf.reduce_sum((self.NCCLabel*(1-self.prob) + (1-self.NCCLabel)*(self.prob))/2)
			self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,labels=self.NCCLabel))

		updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(updateOps): ## important for 
			self.trainOp = self.optimizer(self.ilr).minimize(self.loss)


		## merging the summary operation 
		with tf.name_scope("summary"):
			tf.summary.scalar('loss_train',self.avgTrainLoss)
			tf.summary.histogram('histogram loss_train', self.avgTrainLoss)

			tf.summary.scalar('loss_test',self.avgTestLoss)
			tf.summary.histogram('histogram loss_test', self.avgTestLoss)
			self.summaryOp = tf.summary.merge_all()

		self.writer = tf.summary.FileWriter(self.summaryAdd, tf.get_default_graph())

	def saveModel(self,sess,itr):
		## function for saving the model 
		saver=tf.train.Saver()
		saver.save(sess, self.modelAdd) 

	def returnArray(Self,listObj):
		## return numpy array of training sample and class label
		XList = []
		YList = []
		LabelList = []
		for obj in listObj:
			tempX =np.array(obj["trainX"])
			tempX = tempX[np.newaxis,...]
			XList.append(tempX)

			tempY = np.array(obj["trainY"])
			tempY = tempY[np.newaxis,...]
			YList.append(tempY)

			tempLabel = np.array([obj["label"]])
			tempLabel = tempLabel[np.newaxis,...]
			LabelList.append(tempLabel)

		## format the data into np.float32 and single array 
		X = np.concatenate(XList,axis=0)
		Y = np.concatenate(YList,axis=0)
		Label = np.concatenate(LabelList,axis=0)
		return X,Y,Label


	def Run(self):
		## code for running the system 
		self.buildNetwork()## building the network

		with open(self.fileName,"r") as fileNameReader:
			## reading the file 
			count = 0
			
			datasetLoaded = {} 
			for line in fileNameReader:
				data = json.loads(line)
				## <NOTE> loading full dataset in the memory. Not an optimal approach for bigger datasets. Find better approach
				if data["size"] not in datasetLoaded:
					datasetLoaded[data["size"]] = [data]
				else:
					datasetLoaded[data["size"]].append(data)
				count +=1

				print("loaded data : ",count)

		
		## segmenting the data into two part based on split ratio
		trainingDataset = {}
		testDataset = {}

		for size in datasetLoaded:
			### copying the data into two part 
			random.shuffle(datasetLoaded[size])
			indexToConsider = int(np.floor(self.ratio*len(datasetLoaded[size])))
			trainDataPerSize =  datasetLoaded[size][:indexToConsider]
			testDataPerSize = datasetLoaded[size][indexToConsider:]

			## once data is loaded 
			trainingDataset[size] = trainDataPerSize
			testDataset[size] = testDataPerSize


		## once shuffleling is done start the training process 
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			print("training Started...")
			testLossFinal = -1
			testAccFinal = -1
			for itr in range(self.maxIter):
				count = 0
				avgLossList = []
				for size in trainingDataset:
					### training with a fixed size of dataset  needed for matrix manipulation
					for idx in range(0,len(trainingDataset[size]),self.batchSize):

						## gettingthe training dataset
						count+=1
						trainData=trainingDataset[size][idx:idx+self.batchSize]
						trainInputX,trainInputY,trainLabel = self.returnArray(trainData)

						trainInputX = trainInputX[...,np.newaxis]
						trainInputY = trainInputY[...,np.newaxis]

						
						loss,_, = sess.run([self.loss,self.trainOp],{self.xVal:trainInputX,self.yVal:trainInputY,self.NCCLabel:trainLabel,self.isTrain:True,self.keepProb:np.array([1-self.dropout])})
						
						
						avgLossList.append(loss)
						print("itr : %d  count : %d trainLoss : %f, avgLossVal : %f, testLoss : %f testAcc: %f "%(itr,count,loss,np.mean(avgLossList),testLossFinal,testAccFinal))




						if (count%100==0):
							print ("calculating test error ...")
							testLossList = []
							accList = []
							for testSize in testDataset:
								testInputX,testInputY,testLabel = self.returnArray(testDataset[testSize])

								testInputX = testInputX[...,np.newaxis]
								testInputY = testInputY[...,np.newaxis]
								testLoss,testProb = sess.run([self.loss,self.prob],{self.xVal:testInputX,self.yVal:testInputY,self.NCCLabel:testLabel,self.isTrain:False,self.keepProb:np.array([1.0])})
								testLossList.append(np.mean(testLoss))
								accList.append(self.calcCrossValAcc(testProb,testLabel))
							testLossFinal = np.mean(testLossList)
							testAccFinal = np.mean(accList)

						else:
							pass

				
				summary = sess.run(self.summaryOp,{self.avgTrainLoss:np.mean(avgLossList), self.avgTestLoss:testLossFinal})
				self.writer.add_summary(summary)

				if (itr%15==0):
					self.ilr*=0.1
				
			print ("saving model ..")
			self.saveModel(sess,itr)


	def calcCrossValAcc(self,predictionProb,actLabel):
		## calculate average accuracy of the model (only for class Label -0 or label 1)
		"""
				predictionProb : the probability for the prediction of each testing dataset
				actLabel 	   : correct label of training dataset
		"""
		count = 0
		correct =0
		for prob,label in zip(predictionProb,actLabel):

			if ( (label[0] ==0) or (label[0]==1) ):

				if prob[0] > (1-prob[0])  :
					prediction = 1
				else:
					prediction = 0
				count+=1

				if (prediction==label):
					correct+=1
				else:
					continue
		return correct/float(count)



	def testModel(self,tubDataset):
		## for testing the model
		with tf.Session() as sess:
			self.buildNetwork()
			saver=tf.train.Saver()
			saver.restore(sess, self.modelAdd)

			
			with open(tubDataset,"r") as tubDataReader:
				count =0
				correct = 0
				for line in tubDataReader:
					data = json.loads(line)
					testInputX,testInputY,testLabel = self.returnArray([data])
					testInputX = testInputX[...,np.newaxis]
					testInputY = testInputY[...,np.newaxis]
					prob = sess.run([self.prob],{self.xVal:testInputX,self.yVal:testInputY,self.isTrain:False,self.keepProb:np.array([1.0])})
					
					if prob[0][0] > (1-prob[0][0])  :
						prediction = 1
					else:
						prediction = 0
					count+=1

					if prediction==testLabel[0][0] :
						correct+=1
					else:
						print ("wrong Prediction : prob : %f label : %f"%(prob[0][0],testLabel[0][0]))

					
					print("count : ",count, "correct : ",correct)

	def predictOverResnet(self,NCCData):
		## for predicting model class label

		saveMapper = "FeatureResNetMapNCC.pickle"
		NCCProbMap = {}
		with tf.Session() as sess:
			self.buildNetwork()
			saver=tf.train.Saver()
			saver.restore(sess, self.modelAdd)

			with open(NCCData,"r+") as NCCDataReader,open(saveMapper,"wb") as saveMapWriter:
				for line in NCCDataReader:
					data = json.loads(line)
					idx = data["featureIdx"]
					className = data["className"]
					testInputX,testInputY,testLabel = self.returnArray([data])
					testInputX = testInputX[...,np.newaxis]
					testInputY = testInputY[...,np.newaxis]
					prob = sess.run([self.prob],{self.xVal:testInputX,self.yVal:testInputY,self.isTrain:False,self.keepProb:np.array([1.0])})

					# print(prob[0])
					print(prob[0][0],idx,className)
					if prob[0][0][0]>0.5:
						NCCLbl = "anticasual"
					else:
						NCCLbl = "casual"
					if className not in NCCProbMap:
						NCCProbMap[className] = [{"idx":idx,"prob":prob[0][0],"NCC":NCCLbl}]
					else:
						NCCProbMap[className].append({"idx":idx,"prob":prob[0][0],"NCC":NCCLbl})

				pickle.dump(NCCProbMap,saveMapWriter, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
	obj = NCCTrain(fileName="./casual-data-gen-30K.json-original")
	# obj.Run()
	

	### #######code to test output on tubenghen dataset ############### 
	# obj.testModel(tubDataset="./tubehengenDataFormat.json")
	################################################################

	### #######code to test output on NCC Resnet dataset ############### 
	obj.predictOverResnet(NCCData="./resnetTraining/resnetModelFeatureVector.json")
	################################################################
