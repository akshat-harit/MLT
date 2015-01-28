import csv
import random as rm
import numpy as np
rm.seed(np.random)
from time import sleep 
import sys
sys.setrecursionlimit(15000)
class Tree():
	def __init__(self, Xtrain, Ytrain):
		self.tree=self.create(Xtrain, Ytrain)

	def create(self, Xtrain, Ytrain):
		if(len(Xtrain)==0):
			return []
		if(len(Xtrain)==1):
			return [[-1, Ytrain[0], 0, 0]]

		tree=[]
		#print "Xtrain", Xtrain
		numberX=len(Xtrain)
		lengthX=len(Xtrain[0])
		while True:
			splitfeature=rm.randint(0, lengthX-1)

			#print "Split:", splitfeature
			rval1=rm.randint(0, numberX-1)
			rval2=rm.randint(0, numberX-1)
			#print "Val1 : ", rval1, "Xnumber", numberX
			#print type(Xtrain[rval1])
			val1=Xtrain[rval1][splitfeature]
			val2=Xtrain[rval2][splitfeature]
			
			splitval=(val1+val2)/2.0
			#Xtrain.sort(key=lambda x: x[splitfeature])
		
			XLdata=[i for i in Xtrain if i[splitfeature] <= splitval]
			XRdata=[i for i in Xtrain if i[splitfeature] > splitval]
			YRdata=[i for i, j in zip(Ytrain, Xtrain) if j[splitfeature] > splitval]
			YLdata=[i for i, j in zip(Ytrain, Xtrain) if j[splitfeature] <= splitval]
			if (len(XRdata)!=0):
				break		
		
		ltree=self.create(XLdata, YLdata)
		#print "Ltree :", ltree
		rtree=self.create(XRdata, YRdata)
		#print "rtree:", rtree
		if(ltree!=[]):
			length_ltree=len(ltree)		
			for row in ltree:
				#print "Row ", row
				row[2]=row[2]+1
				row[3]=row[3]+1
		
		if(rtree!=[]):
			length_rtree=len(rtree)
			for row in rtree:
				row[2]=row[2]+length_ltree+1
				row[3]=row[3]+length_ltree+1
		
		if rtree==[]:
			root=[[splitfeature, splitval, 1, -1000]]
			return root+ltree
		if ltree==[]:
			root=[[splitfeature, splitval, -1, 1 ]]
			return root+rtree
		root=[[splitfeature, splitval, 1, length_ltree+1 ]]
		
		return root+ltree+rtree
	
	def readTree(self,file_name):
		reader=csv.reader(open(file_name, 'rU'), delimiter=',')
		for row in reader:
			row=[int(i) for i in row]
			self.tree.append([row[0], row[1], row[2], row[3]])
		return self.tree

	def query(self, X):
		root=self.tree[0]
		i=0
		while(root[0]!=-1):
			#print root	
			position, val, ltree, rtree=root[0], root[1], root[2], root[3]
			if(X[position]> val):
				root=self.tree[rtree]
			elif(X[position]<= val):
				root=self.tree[ltree]
			else:
				print "wrong "
				break
			i+=1
		return root[1]
	
	def treelen(self):
		return len(self.tree) 

	def __str__(self):
		a=""
		for i in self.tree:
			a=a+str(i)+"\n"
		return a



class RandomForest:
	def __init__(self, k=3):
		print "Initializing learner with k = ", k
		self.k=k
		self.TreeList=[]

	def addEvidence(self, Xtrain, Ytrain):
		lengthX=len(Xtrain)
		InternalXtrain=[]
		InternalYtrain=[]
		#print int(.6*lengthX)
		testGenArr=[i for i in range(600)]
		for i in range(self.k):
			Z=rm.sample(zip(Xtrain, Ytrain), int(.6*lengthX))
			xt=[]
			yt=[]
			rm.shuffle(testGenArr)
			for i in testGenArr:
				xt.append(Xtrain[i])
				yt.append(Ytrain[i])
			for i in Z:
				InternalXtrain.append(i[0])
				InternalYtrain.append(i[1])
			#tree=Tree(InternalXtrain, InternalYtrain)
			tree=Tree(xt,yt)
			self.TreeList.append(tree)
			InternalYtrain=[]
			InternalXtrain=[]

	def query(self, X):
		#print "Entering query"

		sum=0.0
		for trees in self.TreeList:
			#print trees
			sum=sum+trees.query(X)
		return (sum/self.k)

def main():
	file_name='data-ripple-prob.csv'
	#file_name='data-classification-prob.csv'
	#file_name="dummy.csv"
	reader=csv.reader(open(file_name, 'rU'), delimiter=',')
	k=0
	Xtrain=[]
	Ytrain=[]
	for row in reader:
		row=[float(i) for i in row]
		Xtrain.append(row[0:len(row)-1])
		Ytrain.append(row[-1])
		k+=1
	# tree=Tree(Xtrain, Ytrain)
	#print(tree.create(Xtrain, Ytrain))
	# print tree
	# print(tree.query([1,2]))
	# print(tree.query([2,3]))
	# print(tree.query([4,5]))
	# print tree.treelen()
	forest=RandomForest(2)
	forest.addEvidence(Xtrain, Ytrain)
	# z=9
	# print Xtrain[9], Ytrain[9]
	# print forest.query(Xtrain[z])-Ytrain[z]
	
	#print forest.query(Xtrain[1])-Ytrain[1]
	for x, y in zip(Xtrain, Ytrain):
		print x
		print y
		print forest.query(x)-y




if __name__ == '__main__':
	main()
