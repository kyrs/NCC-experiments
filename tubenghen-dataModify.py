"""
__author__ : kumar shubham
__desc__   : code for data formating of tubenghen dataset 
"""

import os
import json 
from sklearn import preprocessing

def scaleFn(inputData):
	scaledOutput = preprocessing.scale(inputData)
	return scaledOutput



def main():
	Dir ="./tubenghen-anti-casual"
	fileName = os.path.join(Dir,"README")
	jsonFileSave = "tubehengenDataFormat.json"

	with open(fileName,"r") as fileNameReader, open(jsonFileSave,"w") as tubenghenJsonWriter:
		casualCount =0
		antiCasualCount = 0
		count = 0
		errorFlag = False
		for line in fileNameReader:
			data = line.split()
			if "->" in data:
				label = 0
				typeOfdata = "casual" 
			elif "<-" in data:
				label = 1
				typeOfdata = "anti-casual"
			else:
				label = None
				pass
			if label is not None:
				fileName = data[0]
				
				count+=1
				print ("count processed : %d"%(count))
				with open(os.path.join(Dir,fileName)+".txt","r") as tubFileReader:
					xVal = []
					yVal = []
					for line in tubFileReader:
						try:
							x,y = line.split()
							xVal.append(float(x))
							yVal.append(float(y))
						except Exception as e:
							if not errorFlag:
								print("error  in : ",fileName)
								print(e)
								
							errorFlag = True
				if errorFlag:
					count-=1
					errorFlag=False
					continue

				newXVal = scaleFn(xVal)
				newYVal = scaleFn(yVal)
				data = {"trainX": newXVal.ravel().tolist(), "trainY":newYVal.ravel().tolist(),"label":label,"type":typeOfdata}
				json.dump(data,tubenghenJsonWriter)
				tubenghenJsonWriter.write("\n")

			else:
				pass
if __name__=="__main__":
	main()