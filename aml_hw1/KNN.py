import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy import linalg as LA
import operator
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def getNeighbors(trainData, testData, k):
	dList = []
	
	for each in range(len(trainData)):
		temp1=testData[1:]

		temp2=trainData[each][1:]
		dist = LA.norm(np.asarray(temp1)-np.asarray(temp2))
		dList.append((trainData[each], dist))
	dList.sort(key=operator.itemgetter(1))
	neighbors = []
	for each in range(k):
		neighbors.append(dList[each][0])
	return neighbors

def getResponse(neighbors):
	classVotes = [0 for n in range(10)]
	for x in range(len(neighbors)):
		index = int(neighbors[x][0])
		classVotes[index] += 1
	return classVotes.index(max(classVotes))

def getAccuracy(testData, predictions):
	correct = 0
	for x in range(len(testData)):
		if testData[x][0] is predictions[x]:
			correct += 1
	return (correct/float(len(testData))) * 100.0

def KNN(trainData,testData,k):
	# generate predictions
	predictions=[]
	for x in range(len(testData)):
		neighbors = getNeighbors(trainData, testData[x], k)
		print "finished neighbor"
		result = getResponse(neighbors)
		print "finished response"
		predictions.append(result)
	accuracy = getAccuracy(testData, predictions)
	print accuracy
	return accuracy

def questionH():
	filePath = "train.csv"
	trainData = []
	with open(filePath, 'rb') as csvfile:
		traintxt = csv.reader(csvfile, delimiter=' ', quotechar='|')
		j = 0
		for row in traintxt:
			row = row[0].split(",")
			pixelVal = row
			if j != 0:
				pixelVal = [float(i) for i in pixelVal]
				pixelVal = np.asarray(pixelVal)
				trainData.append(pixelVal)
			j +=1
	kf = KFold(n_splits=3)
	i=0
	accuracy=0
	for train, test in kf.split(trainData):
		testD =[]
		trainD =[]
		for each in test:
			testD.append(trainData[each])
		for each in train:
			trainD.append(trainData[each])
		accuracy += KNN(trainD,testD,5)
		i+=1
		print i
	return accuracy/i

def questionHversion():
	filePath = "train.csv"
	trainLabel = []
	trainData = []
	with open(filePath, 'rb') as csvfile:
		traintxt = csv.reader(csvfile, delimiter=' ', quotechar='|')
		j = 0
		for row in traintxt:
			row = row[0].split(",")
			label = row[0]
			pixelVal = row[1:]
			if j != 0:
				trainLabel.append(label)
				pixelVal = [float(i) for i in pixelVal]
				pixelVal = np.asarray(pixelVal)
				#digit = pixelVal.reshape(28,28)
				trainData.append(pixelVal)
			j +=1
	neigh = KNeighborsClassifier(n_neighbors=3)
	print "here"
	scores = cross_val_score(neigh, trainData, trainLabel)
	print scores
	print scores.mean()

import pandas as pd

def questionJ():
	filePath = "train.csv"
	trainLabel = []
	trainData = []
	with open(filePath, 'rb') as csvfile:
		traintxt = csv.reader(csvfile, delimiter=' ', quotechar='|')
		j = 0
		for row in traintxt:
			row = row[0].split(",")
			label = row[0]
			pixelVal = row[1:]
			if j != 0:
				trainLabel.append(label)
				pixelVal = [float(i) for i in pixelVal]
				pixelVal = np.asarray(pixelVal)
				#digit = pixelVal.reshape(28,28)
				trainData.append(pixelVal)
			j +=1

	filePath = "test.csv"
	testData = []
	with open(filePath, 'rb') as csvfile:
		testtxt = csv.reader(csvfile, delimiter=' ', quotechar='|')
		j = 0
		for row in testtxt:
			row = row[0].split(",")
			pixelVal = row
			if j != 0:
				pixelVal = [float(i) for i in pixelVal]
				pixelVal = np.asarray(pixelVal)
				#digit = pixelVal.reshape(28,28)
				testData.append(pixelVal)
			j +=1
	
	neigh = KNeighborsClassifier(n_neighbors=3)
	neigh.fit(trainData,trainLabel)
	result=neigh.predict(testData)
	reuslt=np.array(result)

	res = []

	a = np.array([i for i in range(1,28000+1)])

	df = pd.DataFrame({"ImageId" : a, "Label" : reuslt})
	df.to_csv("result.csv", index=False)


questionJ()

	




