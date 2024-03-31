import torch
import nltk
import math
import numpy as np
import matplotlib.pyplot as plt
import warnings
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize

# nltk.download('punkt')

warnings.filterwarnings(action = 'ignore')

dprint = 1 # prints outputs if set 1, default =0

# `text.txt` file
sample = open('text.txt', 'r')
s = sample.read()

# processing escape characters
f = s.replace("\n", " ")

data = []

# sentence parsing
for i in sent_tokenize(f):
    temp = []
    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())
        
    data.append(temp)
    
# Creating Skip Gram model
model2 = Word2Vec(data, min_count=1, vector_size=512, window=5, sg=1)

# 1-The 2-black 3-cat 4-sat 5-on 6-the 7-couch 8-and 9-the 10-brown 11-dog 12-slept 13-on 14-the 15-rug.
word1='black'
word2='brown'
pos1=2
pos2=10
a=model2.wv[word1]
b=model2.wv[word2]

if(dprint==1):
    print(a)
    
    
# compute cosine similarity
dot = np.dot(a, b)
norma = np.linalg.norm(a)
normb = np.linalg.norm(b)
cos = dot / (norma * normb)

aa = a.reshape(1,512)
ba = b.reshape(1,512)
cos_lib = cosine_similarity(aa, ba)

# A Positional Encoding example using one line of basic Python using a few lines
# of code for the sine and cosine functions. I added a Pytorch method inspired by
# Pytorch.org to explore these methods. The main idea to keep in mind is that we 
# are looking to add small values to the word embedding output so that the 
# positions are taken into account. This means that as long as the cosine 
# similarity, for example, displayed at the end of the notebook, shows the 
# positions have been taken into account, the method can apply. Depending on the 
# Transformer model, this method can be fine-tuned as well as using other methods.

pe1=aa.copy()
pe2=aa.copy()
pe3=aa.copy()
paa=aa.copy()
pba=ba.copy()
d_model=512
max_print=d_model
max_length=20

for i in range(0, max_print,2):
    pe1[0][i] = math.sin(pos1 / (10000 ** ((2 * i)/d_model)))
    paa[0][i] = (paa[0][i]*math.sqrt(d_model))+ pe1[0][i]
    pe1[0][i+1] = math.cos(pos1 / (10000 ** ((2 * i)/d_model)))
    paa[0][i+1] = (paa[0][i+1]*math.sqrt(d_model))+pe1[0][i+1]
    if dprint==1:
        print(i,pe1[0][i],i+1,pe1[0][i+1])
        print(i,paa[0][i],i+1,paa[0][i+1])
        print("\n")

#print(pe1)
# A  method in Pytorch using torch.exp and math.log :
max_len=max_length
pe = torch.zeros(max_len, d_model)
position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
#print(pe[:, 0::2])