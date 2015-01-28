import csv
from KNNLearner import KNNLearner
from KNNLearner1 import RKNNLearner
from LinRegLearner import LinRegLearner
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from operator import itemgetter
import pylab as p
pp = PdfPages('multipage.pdf')

import mpl_toolkits.mplot3d.axes3d as p3
#reader=csv.reader(open("data-ripple-prob.csv", 'rU'), delimiter=',')
reader=csv.reader(open("data-classification-prob1.csv", 'rU'), delimiter=',')

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


data=[]
for row in reader:
	data.append([float(i) for i in row])
d=np.array(data)
outsampleError=[]
insampleError=[]
number=51

# for i in range(1,number):
# 	learner=RKNNLearner(i)
# 	learner=train(d[0:600,:], learner)
# 	insample=test(d[0:600,:], learner)
# 	outsample=test(d[600:1000,:], learner)
# 	insampleError.append(rmserror(insample,d[0:600,2]))
# 	outsampleError.append(rmserror(outsample,d[600:1000,2]))
# 	print "RMS Error"
# 	print "Insample : ",rmserror(insample,d[0:600,2])
# 	print "Outsample : ", rmserror(outsample,d[600:1000,2])
# 	print "Correlation"
# 	print "OutSample : ",np.corrcoef(outsample,d[600:1000,2])[0][1]
# 	print "InSample : ",np.corrcoef(insample,d[0:600,2])[0][1]
# 	# pp = PdfPages('KNN_data_classification_scatter27.pdf')
# 	# plt.scatter(d[600:1000,2], outsample, )
# 	# plt.xlabel("Actual Y")
# 	# plt.ylabel("Predicted Y")
# 	# pp.savefig()
# 	# pp.close()
# 	# plt.show()



# print "In sample Error"
# print insampleError
# print "outsample Error"
# print outsampleError
# pp = PdfPages('RKNN_data_class.pdf')
# isE=plt.plot(range(1,number),insampleError,'g-',label='Insample error')
# osE=plt.plot(range(1, number),outsampleError,'r-', label='Outsample Error' )
# red_patch = mpatches.Patch(color='red', label='outsampleError')
# green_patch = mpatches.Patch(color='green', label='insampleError')
# plt.legend(handles=[red_patch, green_patch])
# plt.ylabel("Error")
# plt.xlabel("K")
# #plt.show()
# pp.savefig()
# pp.close()
# Best=min(enumerate(outsampleError), key=itemgetter(1))[0]
# print "Best ", Best
# plt.show()



# learner=LinRegLearner()
# learner=train(d[0:600,:], learner)
# insample=test(d[0:600,:], learner)
# outsample=test(d[600:1000,:], learner)
# print "RMS Error"
# print "Insample : ",rmserror(insample,d[0:600,2])
# print "Outsample : ", rmserror(outsample,d[600:1000,2])
# print "Correlation"
# print "OutSample : ",np.corrcoef(outsample,d[600:1000,2])[0][1]
# print "InSample : ",np.corrcoef(insample,d[0:600,2])[0][1]

# pp = PdfPages('Linear_data_ripple_scatter.pdf')
# plt.scatter(d[600:1000,2], outsample)
# plt.xlabel("Actual Y")
# plt.ylabel("Predicted Y")
# pp.savefig()
# pp.close()
# plt.show()

def extracredit1():
	reader=csv.reader(open("data-classification-prob1.csv", 'rU'), delimiter=',')
	data=[]
	for row in reader:
		data.append([float(i) for i in row])
	d=np.array(data)
	learner=KNNLearner(27)
	learner=train(d[0:1000,:], learner)
	d=[]
	step=0.01
	for x1 in np.arange(-1,1, step):
		for x2 in np.arange(-1, 1, step):
			d.append([x1,x2])
	d=np.array(d)
	sample=[]
	for j,i in enumerate(d):
		if(j%1000==0):
			print j
		sample.append(learner.query(i))
	fig=p.figure()
	ax = p3.Axes3D(fig)
	ax.scatter(d[:,0],d[:,1],sample,c='r', marker='o')
	ax.set_xlabel('X1')
	ax.set_ylabel('X2')
	ax.set_zlabel('Y')
	pp = PdfPages('3d_million_class_actual.pdf')
	pp.savefig()
	pp.close()
	p.show()

def extracredit2():
	reader=csv.reader(open("data-ripple-prob.csv", 'rU'), delimiter=',')
	data=[]
	for row in reader:
		data.append([float(i) for i in row])
	d=np.array(data)
	learner=KNNLearner(27)
	learner=train(d[0:600,:], learner)
	insample=test(d[0:600,:], learner)
	outsample=test(d[600:1000,:], learner)
	print "RMS Error for KNN"
	print(rmserror(outsample,d[600:1000,2]))
	print "RMS Error for RKNN"
	print(rmserror(insample,d[0:600,2]))

	print i
	print "Error"
	print(rmserror(insample,d[0:600,2]))
	print(rmserror(outsample,d[600:1000,2]))
	print(np.corrcoef(outsample,d[600:1000,2])[0][1])

	plt.scatter(d[600:1000,2], outsample, )

	fig=p.figure()
	ax = p3.Axes3D(fig)
	ax.scatter(d[600:1000,0],d[600:1000,1],d[600:1000,2],c='r', marker='o',label="Actual")
	ax.scatter(d[600:1000,0],d[600:1000,1], outsample,c='b', marker='o', label="Predicted")
	ax.set_xlabel('X1')
	ax.set_ylabel('X2')
	ax.set_zlabel('Y')
	red_patch = mpatches.Patch(color='red', label='Actual')
 	blue_patch = mpatches.Patch(color='blue', label='Predicted')
	plt.legend(handles=[red_patch, blue_patch])
	pp = PdfPages('3d_ripple.pdf')
	pp.savefig()
	pp.close()
	p.show()



def extracredit3():
	print "Entering function"
	outsampleError=[]
	outsampleError_R=[]
	insampleError=[]
	insampleError_R=[]
	reader=csv.reader(open("data-classification-prob1.csv", 'rU'), delimiter=',')
	data=[]
	for row in reader:
		data.append([float(i) for i in row])
	d=np.array(data)
	number=51
	for i in range(1,number):
		learner=KNNLearner(i)
		learner=train(d[0:600,:], learner)
		insample=test(d[0:600,:], learner)
		outsample=test(d[600:1000,:], learner)
		insampleError.append(rmserror(insample,d[0:600,2]))
		outsampleError.append(rmserror(outsample,d[600:1000,2]))
		
		learner=RKNNLearner(i)
		learner=train(d[0:600,:], learner)
		insample=test(d[0:600,:], learner)
		outsample=test(d[600:1000,:], learner)
		insampleError_R.append(rmserror(insample,d[0:600,2]))
		outsampleError_R.append(rmserror(outsample,d[600:1000,2]))
		k=0
		for x,y in zip(outsample, d[600:1000,2]):
			if(x==y):
				k+=1
		accuracy_plot.append(k/400.0)
		#print i ,"has accuracy", k/400.0
	
	plt.plot(range(1, number), accuracy_plot)
	plt.show()
	print len(insampleError)
	pp = PdfPages('RKNN_vs_CKNN.pdf')
	isE=plt.plot(range(1,number),insampleError,'g-',label='Insample error for KNN')
	osE=plt.plot(range(1, number),outsampleError,'r-', label='Outsample Error for KNN' )
	red_patch = mpatches.Patch(color='red', label='outsampleError for KNN')
	green_patch = mpatches.Patch(color='green', label='insampleError for KNN')
	isE_R=plt.plot(range(1,number),insampleError_R,'b-',label='Insample error for classsifier KNN ')
	osE_R=plt.plot(range(1, number),outsampleError_R,'y-', label='Outsample Error for classsifier KNN' )
	blue_patch = mpatches.Patch(color='blue', label='insampleError for classsifier KNN')
	yellow_patch = mpatches.Patch(color='yellow', label='outsampleError for classsifier KNN')
	

	plt.legend(handles=[red_patch, green_patch, blue_patch, yellow_patch])
	plt.ylabel("Error")
	plt.xlabel("K")
	plt.title("Vanilla KNN vs Classifier KNN")
	#plt.show()
	pp.savefig()
	pp.close()
	Best=min(enumerate(outsampleError_R), key=itemgetter(1))[0]
	print "Best ", Best
	print np.min(outsampleError_R)
	plt.show()
	
extracredit1()