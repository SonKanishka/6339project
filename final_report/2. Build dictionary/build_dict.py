"""
Building the dictionary which has 800 popular words. 
Program will read first 100 000 reviews from 1.3 Yelp reviews and choose most 800 popular words.
Input was saved in train100000.csv file
Ouput is dictionary.txt file which has 800 words

@author: Son
"""

import csv
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
stop_words=stopwords.words('english')
stop_words.sort()
stemmer = PorterStemmer()
dict_word = {}
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
with open('train100000.csv',encoding='utf8',newline='\n') as csvfile:
    line = csv.reader(csvfile,delimiter='\n')
    for n in line:
        separated_line = re_process(str(n))
        for m in separated_line:
            try:
                dict_word[m] = dict_word[m] + 1
            except KeyError:
                dict_word[m] = 1
values = sorted(dict_word.values(),reverse = True)       
thresh_hold =  values[800]
dict_ref = []
for k,v in dict_word.items():
    if v > thresh_hold:
        dict_ref.append(k)
f = open('dictionary.txt','w')
for n in dict_ref:
    f.write(n + ' ')
f.close()
    