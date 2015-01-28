from randomForestLearner import Tree
import csv
file_name='data-ripple-prob.csv'
#file_name='data-classification-prob.csv'
#file_name="dummy.csv"
reader=csv.reader(open(file_name, 'rU'), delimiter=',')

Xtrain=[]
Ytrain=[]
for row in reader:
	row=[float(i) for i in row]
	Xtrain.append(row[0:len(row)-1])
	Ytrain.append(row[-1])
tree=Tree(Xtrain, Ytrain)
#print(tree.create(Xtrain, Ytrain))
print tree
print(tree.query([1,2]))
print(tree.query([2,3]))
print(tree.query([4,5]))
print tree.treelen()