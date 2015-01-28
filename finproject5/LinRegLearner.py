import numpy as np

class LinRegLearner:
	def __init__(self):
		self.trainingData=[]

	def addEvidence(self,X,Y):
		if type(self.trainingData is list):
			self.trainingData.append([X,Y])
		else:
			self.trainingData=self.trainingData.tolist()
			self.trainingData.append([X,Y])

	def query(self, queryData):
		self.trainingData=np.array(self.trainingData)
		Y=self.trainingData[:,1]
		X=self.trainingData[:,0]
		X=np.vstack(X)
		A=np.array([X[:,0],X[:,1], np.ones(len(X[:,0]))])
		Y=Y.flatten()
		a = np.linalg.lstsq(A.T,Y)[0]
		queryData.append(1)
		queryData=np.array(queryData)
		return np.sum(np.multiply(a,queryData))
