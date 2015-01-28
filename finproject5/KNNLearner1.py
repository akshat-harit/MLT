class RKNNLearner:
	def __init__(self, K=3):
		self.trainingData=[]
		self.minDistance=[]
		self.queryData=None
		self.K=K
		
	def addEvidence(self,X,Y):
		#print "Adding X and Y: "+str(X)+" Y: "+str(Y)
		self.trainingData.append((X,Y))


	def EuclieandDistanceFromSelf(self, X):
		distance=0
		for i, x in enumerate(X):
			distance=distance+(self.queryData[i]-x)**2
		distance=distance**.5
		return distance	


	def query(self, X):
		self.minDistance=[]
		self.queryData=X
		for t in self.trainingData:
			#data=list(self.EuclieandDistanceFromSelf(t[0]))
			(self.minDistance).append([self.EuclieandDistanceFromSelf(t[0]) ,t[1]] )
		
		(self.minDistance).sort(key=lambda x: x[0])
		counter=0
		sum=0
		while(counter<self.K):
			#print "Sum is"
			sum=sum+self.minDistance[counter][1]
			#print sum
			counter=counter+1
		res=sum/self.K
		if(res>(1.0/3)):
			return 1
		elif(res<-(1.0/3)):
			return -1
		else:
			return 0


	def getK():
		return self.K

