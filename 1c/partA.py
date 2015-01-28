import sys
import fileinput,re
import numpy as np
import math

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
  hashFile=file_dict[fileo]
  tfidf=tfidf.transpose()
  #print hashFile
  #print file_dict
  maximum=0
  #print tfidf[hashFile,:]
  for j in range(len(tfidf[:,0])): # j is files
    if(j!=hashFile):
      cosine=cos(tfidf[hashFile,:], tfidf[j,:])
      if(maximum<=cosine):
	maximum=cosine
	filehash=j
	
  print fileo+", "+file_dict_reverse[filehash]+","+str(maximum)
		 
  
  
  
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
print "filename, match, cosine"
for fileo in files.split():
  maxsim(fileo, tfidf, file_dict, file_dict_reverse)

#Printing file list in order
#file_list=[0 for col in range(len(file_dict))]
#for k,v in file_dict.items(): 
  #file_list[v]=k

##Actual printing loop

#print 'term, '+ ','.join(file_list)
#printV=''
#for k, v in word_dict.items():
  #printV=k
  #for i in range(len(file_list)):
    #printV+=','+str(tfidf[v,i])
  #print printV
  #printV=''

