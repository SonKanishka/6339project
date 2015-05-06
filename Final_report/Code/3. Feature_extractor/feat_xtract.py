# -*- coding: utf-8 -*-
"""
Perform feature extractor from input reviews
Each review will be transform to a 800 dimension vector base on dictionary from previous step.

@author: Son
"""
import csv
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
stop_words=stopwords.words('english')
stop_words.sort()
stemmer = PorterStemmer()

#Load the dictionary
dict_file = 'dictionary.txt'
f = open(dict_file,'r')
key_word = f.read()
f.close()
key_word = key_word.split()
dict_word = {}
for word in key_word:
    dict_word[word] = 0
    
def re_process(doc):
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
#Ouput file    
f = open('train_vec.csv','w')

#Open input file
with open('train100000.csv',encoding='utf8',newline='\n') as csvfile:
    line = csv.reader(csvfile,delimiter='\n')
    for n in line:
        current_vec = dict_word.copy()
        separated_line = re_process(str(n))
        for n in separated_line:
            try:
                current_vec[n] = current_vec[n] + 1
            except KeyError:
                pass
        vec = [v for v in current_vec.values()]
        vec_str =str(vec[0])
        for n in range(1,len(vec)):
            vec_str = vec_str + ',' +str(vec[n])
        f.write(vec_str + '\n')
f.close()
        
        
            
        
    