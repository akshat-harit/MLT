from randomForestLearner import RandomForest
from random import shuffle
from numpy import genfromtxt
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches	
from KNNLearner import KNNLearner

def rmserror(orig, learned):
	o=np.array(orig)
	l=np.array(learned)
	rms=np.sqrt(np.mean(np.square(np.subtract(o,l))))
	return rms

def train(data, learner):
	length=len(data)
	for j,row in enumerate(data):
		X=[]
		for i in range(len(row)-1):
			X.append(float(row[i]))
		Y=float(row[-1])
		learner.addEvidence(X,Y)
	return learner

def test(data, learner):
	sample=[]
	length=len(data)
	for j,row in enumerate(data):
		x1=float(row[0])
		x2=float(row[1])
		sample.append(learner.query([x1,x2]))
	return sample

reader=csv.reader(open("data-classification-prob.csv", 'rU'), delimiter=',')
#reader=csv.reader(open("data-ripple-prob.csv", 'rU'), delimiter=',')
Xtrain=[]
Ytrain=[]
Xtest=[]
Ytest=[]
Y=[]
k=0
for row in reader:
	row=[float(i) for i in row]
	if(k>=600):
		Xtest.append(row[0:len(row)-1])
		Ytest.append(row[-1])
	Xtrain.append(row[0:len(row)-1])
	Ytrain.append(row[-1])
	k+=1

#reader=csv.reader(open("data-ripple-prob.csv", 'rU'), delimiter=',')
data=[]
reader=csv.reader(open("data-classification-prob.csv", 'rU'), delimiter=',')

for row in reader:
	data.append([float(i) for i in row])
d=np.array(data)

rmsArr = []
rmsArr_train=[]
rmsArr_test=[]
rmsArr_k=[]
testGenArr = [i for i in range(600)]
number=101
kArr = [i for i in range(1,number)]
for k in kArr:
	print "Training a new learner"
	learner = RandomForest(k)
	learner.addEvidence(Xtrain, Ytrain)
	Y_ltest=[]
	Y_ltrain=[]
	for x in Xtest:
		Y_ltest.append(learner.query(x))
	# for x in Xtrain:
	# 	Y_ltrain.append(learner.query(x))
	#rmsArr_train.append(math.sqrt(np.mean((np.array(Y_ltrain) - np.array(Ytrain)) ** 2)))
	rmsArr_test.append(math.sqrt(np.mean((np.array(Y_ltest) - np.array(Ytest)) ** 2)))
	learner=KNNLearner(k)
	learner=train(d[0:600,:], learner)
	outsample=test(d[600:1000:], learner)
	rmsArr_k.append(rmserror(outsample, d[600:1000,2]))




pp = PdfPages('Data-Classification-K-RandomForest.pdf')
plt.clf()
plt.ylabel('RMS')
plt.xlabel('k')
isE=plt.plot(range(1,number),rmsArr_k,'g-',label='KNN')
osE=plt.plot(range(1, number),rmsArr_test,'r-', label='Random Forest' )
red_patch = mpatches.Patch(color='red', label='Random Forest')
green_patch = mpatches.Patch(color='green', label='KNN')
plt.legend(handles=[red_patch, green_patch])
plt.title("Data-Classification-K-RandomForest")
pp.savefig()
pp.close()
plt.show()