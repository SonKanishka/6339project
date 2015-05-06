# -*- coding: utf-8 -*-
"""
Program will take input as a string, process to get 800 dimension feature vectors base on dictionary
The perform classifier with saved weights from 3 csv files: W1.csv, W2.csv, W3.csv
@author: Son
"""

from numpy import genfromtxt,asarray,size,dot,argmax,insert
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from scipy.special import expit
from os import getcwd,sep
#Adding stop words 
stop_words=stopwords.words('english')
stop_words.sort()
stemmer = PorterStemmer()

#Create start counting word vector from dictionary
dict_file = 'dictionary.txt'
full_file = getcwd() + sep + dict_file
file = open(full_file,'r')
f = open(dict_file,'r')
key_word = f.read()
f.close()
key_word = key_word.split()
dict_word = {}
for word in key_word:
    dict_word[word] = 0


def re_process(doc):
    #repcoress the string, lowercase, remove stop words, stemming
    #lower string
    doc=doc.lower()
    #tokenizing string
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(doc)
    #removing stop words
    
    #stemming
    stem=[]
    for word in tokens:
        stem.append(stemmer.stem(word))
        
    f2 = [i for i in stem if i not in stop_words]
    return f2  
    
def feat_extract(text,current_vec):
    #generate features for each test inpu, return a vector of 800 dimension
    separated_line = re_process(str(text))
    for word in separated_line:
        try:
            current_vec[word] = current_vec[word]+1
        except KeyError:
            pass
    feat = [v for v in current_vec.values()]
    feat = asarray(feat)
    
    #Normalization
    feat = feat/(sum(feat))
    return feat 
    
def classifier(x,W1,W2,W3):
    #Perform classifier given input vector and 3 weights matrixes

    #Adding bias
    x=insert(x,0,1)
    Np = size(x,axis=0)
    
    #First hidden layer
    Xa1=x.reshape(1,Np)
    O1 = expit(dot(Xa1,W1.transpose()))
    
    #Second hidden layer
    Xa2 = insert(O1,0,1)
    Xa2=Xa2.reshape(1,len(Xa2))
    O2 = expit(dot(Xa2,W2.transpose()))
    
    #3rd hidden layer
    Xa3 = insert(O2,0,1)
    Xa3=Xa3.reshape(1,len(Xa3))
    y = dot(Xa3,W3.transpose())
    
    #Ouput will be index of column with bigger value
    ic = argmax(y)
    return ic  


#Loading weights matrixed which were saved from training step on Matlab
W1  = genfromtxt('W1.csv', delimiter=',')
W2  = genfromtxt('W2.csv', delimiter=',')
W3  = genfromtxt('W3.csv', delimiter=',')

#take input text from user
text = input('please put review:\n')
#generate feature for the test
feat = feat_extract(text,dict_word.copy())

ic = classifier(feat,W1,W2,W3)
if ic == 0:
    print('Non fake review\n')
else:
    print('Fake review\n')
#
#close = input('Press Enter to exit...')