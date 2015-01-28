import sys
import fileinput,re
import numpy as np
import math

DATA=''
classification_List=[]
def text_process(word):
  word=re.sub(r'[\r\t\n]', ' ', word)
  word=re.sub(r'[^a-z ]', '', word.lower())
  return word

def cos(v1,v2):
  dot=0
  modv1=0
  modv2=0
  for i in range(len(v1)):
    dot+=(v1[i]*v2[i])
    modv1+=(v1[i]*v1[i])
    modv2+=(v2[i]*v2[i])
  modv1=math.sqrt(modv1)
  modv2=math.sqrt(modv2)    
  cos=dot/(modv1*modv2)
  
  return cos

def maxsim(fileo, tfidf,file_dict , file_dict_reverse):
  string=''
  hashFile=file_dict[fileo]
  tfidf=tfidf.transpose()
  #print hashFile
  #print file_dict
  classification=0
  #print tfidf[hashFile,:]
  #print fileo
  goodF=0
  badF=0
  goodclassification=0
  badclassification=0
  for j in range(len(tfidf[:,0])): # j is files
    if(j!=hashFile):
      cosine=cos(tfidf[hashFile,:], tfidf[j,:])
      if('bad' in file_dict_reverse[j]):
	badclassification+=(-1*cosine)
	classification_List.append((-1*cosine)
	#print "Fraternizing with bad files+"+file_dict_reverse[j]+", penalty is "+str((-1*cosine))
	badF=badF+1
      else:
	goodclassification+=(1*cosine)
	classification_List.append((1*cosine)
	#print "Fraternizing with good files,"+file_dict_reverse[j]+" advantage is "+str((1*cosine))
	goodF=goodF+1
  
  
  #print "Good number, classification, Bad"
  #print goodF, goodclassification, badF, badclassification
  classification=float(badclassification)/badF+float(goodclassification)/goodF
  if (classification>0):
    classOfFile="good"
  elif(classification==0):
    classOfFile="Confused"
  else:
    classOfFile="bad"
    
  #print fileo+" is "+classOfFile+" File with confidence of "+str(math.fabs(classification))
  string=fileo+","+classOfFile+"\n"
  return string
		 
def classifierEfficiency(Data):
  counterCorrect=0
  counterBad=0
  for lines in Data.split("\n"):
    lines=lines.replace(".txt","")
    a=lines.split(",")
    
    if(a[0][0:3]==a[1][0:3]):
      counterCorrect+=1
    else:
      counterBad+=1
  print "overall score: "+ str((counterCorrect*100)/float(counterBad+counterCorrect))+"%"
  
files=' '.join(sys.argv)
#files=files[1:]
files=files.replace(sys.argv[0], '')
set1=set()
for fileo in files.split():
  #print files
  f = open(str(fileo), 'r')
  data=f.readlines()
  for words in ''.join(data).split():
    word=text_process(words)
    if (word !=''):
      set1.add(word)
      

number_of_files=len(sys.argv)-2
word_dict=dict()
words_matrix=np.zeros( (number_of_files+1,len(set1) ))
hash_word=0
file_dict=dict()
file_dict_reverse=dict()
word_dict_reverse=dict()
number_of_files=-1
for fileo in files.split():
  #print files
  number_of_files+=1
  f = open(str(fileo), 'r')
  data=f.readlines()
  for words in ''.join(data).split():
    word=text_process(words)
    if (word ==''):
      continue
    elif(word not in word_dict):
      word_dict[word]=hash_word
      word_dict_reverse[hash_word]=word
      hash_word+=1
    
    file_dict[fileo]=number_of_files
    file_dict_reverse[number_of_files]=fileo
    words_matrix[file_dict[fileo],word_dict[word]]+=1
    
wc=words_matrix.copy()
wc=wc.transpose()
#print wc
wc[wc>=1]=1
dc=wc.sum(axis=1)

idf=np.log(float(number_of_files+1)/dc)

#Compute tf and tfidf  
tf=np.zeros((number_of_files+1,len(set1)))
tf=tf.transpose()
tfidf=tf.copy()
wc=words_matrix.copy()
wc=wc.transpose()
for j in range(len(tf[0,:])): # j is files
  for i in range(len(tf[:,0])): # i is words
    tf[i,j]=float(wc[i,j])/max(wc[:,j])
    tfidf[i,j]=tf[i,j]*idf[i]
#print tfidf    
#print len(tfidf[0,:])
#print len(tfidf[:,0])
#print file_dict_reverse
#print "filename, match, cosine"
string=''
for fileo in files.split():
  string=string+maxsim(fileo, tfidf, file_dict, file_dict_reverse)
    


#print "Classifier Part" 
string=string.rstrip('\n')
classifierEfficiency(string)

  


#Printing file list in order
#file_list=[0 for col in range(len(file_dict))]
#for k,v in file_dict.items(): 
  #file_list[v]=kk


##Actual printing loop

#print 'term, '+ ','.join(file_list)
#printV=''
#for k, v in word_dict.items():
  #printV=k
  #for i in range(len(file_list)):
    #printV+=','+str(tfidf[v,i])
  #print printV
  #printV=''

