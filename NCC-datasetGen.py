"""
__author__ : Kumar Shubham
__date__   : 20-03-2019

__desc__   : code for genereating the training samples for NCC

## reference : https://github.com/lopezpaz/causation_learning_theory
"""

import random
from sklearn import preprocessing
from sklearn.mixture import  GMM ## present in scikit 0.18
import numpy as np 
from scipy import interpolate 
import json
import copy 


import warnings
warnings.filterwarnings("ignore")

class NCCDataGen(object):
	def __init__(self,ni=30000,saveFile= "",minSize=100,maxSize = 1000):
		"""
		ni : total no of training example set
		saveFile : address where the data is gonna be saved 
		minSize : minimum no of data point for a function
		maxSize : maximum no of data point for a function
		"""
		self.ni = ni
		self.saveFile = saveFile
		self.minSize = minSize
		self.maxSize = maxSize
		self.seed = np.random.seed(1)

	def normalize(self,inputDataList):
		## given function normalize the inputDataList with zero mean and unit variance
		scaledOutput = preprocessing.scale(inputDataList)

		return scaledOutput


	def GMM(self,K,meanSampStd,varSampStd,size):
		## setting up gaussian mixture model for sampling 

		"""
			meanSampleStd : std for sampling in mean 
			varSampStd    : std for sampling in std of GMM

			weights is assumed to be sampled from normalizedGaussian (0,1)

		"""

		"""
		NOTE : 
		For random samples from N(\\mu, \\sigma^2), use:
		sigma * np.random.randn(...) + mu
		https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html
		"""

		sampler  = GMM(K)
		sampler.means_   = meanSampStd*np.random.randn(K,1)# Gaussian(0,ri)
		sampler.covars_ = np.power(abs(varSampStd*np.random.randn(K,1)),2) # standard deviation from Gaussian(0,si)
		sampler.weights_ = abs(np.random.randn(K,1))
		sampler.weights_ = sampler.weights_/sum(sampler.weights_)
		return self.normalize(sampler.sample(size))

	def Yn(self,dKnot,x):
		## return the function cubic hermit spline for given support and dKnot
		""" 
			dKnot : no of points to consider for creation of spline
			x     : data for which spline is trained 

		"""
		supportX = sorted(np.random.randn(dKnot))
		return self.normalize(self.normalize(interpolate.PchipInterpolator(supportX,np.random.randn(dKnot))(x.flatten()))[:,np.newaxis] + self.noise(x))
	def noise(self,x):
		## generate noise for the given input x
		supportX = np.linspace(min(x)-np.std(x),max(x)+np.std(x),4)
		noise = np.linspace(min(x)-np.std(x),max(x)+np.std(x),len(x))
		# noise = np.random.uniform(min(x)-np.std(x),max(x)+np.std(x),len(x))        
		vij = interpolate.UnivariateSpline(supportX,np.random.uniform(0,5,len(supportX)))(noise.flatten())[:,np.newaxis]
		vn = np.random.uniform(0,5,1)
		return vn*np.random.randn(len(x),1)*vij

	

	def Run(self):
		## function for genereating of the data 
		sizeList = np.random.randint(self.minSize,self.maxSize,self.ni)

		## open the saving file 
		with open(self.saveFile,"w") as saveFile:

			for idx,size in enumerate(sizeList):
				K = np.random.randint(1,5)
				meanSampVar= np.random.uniform(0,5)## variance of GMM mean term
				varSampVar = np.random.uniform(0,5)## variance of GMM covariance term

				## calculating the std 
				meanSampStd = np.abs(np.power(meanSampVar,0.5))
				varSampStd  = np.abs(np.power(varSampVar,0.5))

				# meanSampStd = meanSampVar
				# varSampStd = varSampVar 

				dKnot = np.random.randint(4,5)
				xn = self.GMM(K=K,meanSampStd=meanSampStd,varSampStd=varSampStd,size=size)
				yn = self.Yn(dKnot,xn)

				## saving casual 
				xnCasual = copy.deepcopy(xn) ## to protect from random shuffle
				casualData = {"trainX":xnCasual.ravel().tolist(),"trainY":yn.ravel().tolist(),"label":0, "type":"casual","size":int(size),"index":idx}
				antiCasualData = {"trainX":yn.ravel().tolist(),"trainY":xnCasual.ravel().tolist(),"label":1, "type":"antiCasual","size":int(size),"index":idx}

				np.random.shuffle(xn)
				
				independentData = {"trainX":xn.ravel().tolist(),"trainY":yn.ravel().tolist(),"label":0.5, "type":"independent-casual","size":int(size),"index":idx}
				#saving casual
				json.dump(casualData,saveFile)
				saveFile.write("\n")	

				#saving anti casual
				json.dump(antiCasualData,saveFile)
				saveFile.write("\n")

				#saving independent
				json.dump(independentData,saveFile)
				saveFile.write("\n")				

				print("idx :",idx)

if __name__ =="__main__":
	obj = NCCDataGen(saveFile = "./casual-data-gen-30K.json-original")
	obj.Run()