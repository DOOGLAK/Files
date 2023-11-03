# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 21:21:46 2022

@author: Doug
"""

################
# LIBRARIES
################
import time
from timeit import default_timer as timer

import pandas as pd
import numpy as np
import json

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

import torch
import numba
from numba import jit, cuda

#For Copies
import copy

#For Synonyms
import requests
from bs4 import BeautifulSoup

#For Shuffle
import random

#For Generation
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Input, Softmax, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping

from keras import backend
from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import to_categorical

################
# CODE
################

dir_data = "H:/My Files/School/Grad School WLU/MRP/Research/Data/FewNERD"
dir_train = dir_data+"/train.txt"
dir_test = dir_data+"/test.txt"
dir_dev =dir_data+"/dev.txt"

with open(dir_dev, encoding='UTF-8') as f:
    file = f.read()

sentences = file.split("\n\n")
sentences_tokenized = []

for sentence in sentences:
    sentences_tokenized.append(sentence.split("\n"))

sentences_paired = []

for sentence in sentences_tokenized:
    temp = []
    
    for word in sentence:
        tok_ent = word.split("\t")
        temp.append(tok_ent)
    
    sentences_paired.append(temp)
        
for sentence in sentences_paired:
    for pair in sentence:
        granularity = pair[1].split("-")
        
        if pair[1] == "O":
            pair.append("O")
        else:
            coarse, fine = granularity[0], granularity[0]+"-"+granularity[1]
            pair[1] = coarse
            pair.append(fine)

#print(sentences_paired[0])









def Labels(original:list):
    segments = []
    labelled = []
    
    start=0
    
    for i,entry in enumerate(original):
        
        first = i
        mid = i+1
        #last = i+2
        
        if (mid) < len(original):
            if original[first]!=original[mid]:
                end = i+1
                segments.append(original[start:end])
                start = end
                
    segments.append(original[start:len(original)])
    
    for segment in segments:
        if len(set(segment)) > 1: print(segment, i) #TO BE DELETED, CHECKING
        
        if segment[0]!="O":
            if len(segment)==1:
                segment[0] = "S-"+segment[0]
            else:
                segment[0], segment[-1] = "B-"+segment[0],"E-"+segment[-1]
    
        labelled = labelled + segment
    
    #print(original)
    #print(segments)
    #print(labelled)

    return labelled

#original = ["O","O","O","O","O","O","new","new","new","new","O","O","new","O","tp","tp","tp","tp","tp"]
#list_labelled = Labels(original)



#MAKE FUNCTION FOR INPUTTING BIO LABELS

print(sentences_paired[0])

for i,sentence in enumerate(sentences_paired):
    new_coarse = []
    new_fine = []
    
    for j,pair in enumerate(sentence):
        new_coarse.append(pair[1])
        new_fine.append(pair[2])
    
    replace_coarse = Labels(new_coarse)
    replace_fine = Labels(new_fine)
    
    for j,pair in enumerate(sentence):
        sentences_paired[i][j][1] = replace_coarse[j]
        sentences_paired[i][j][2] = replace_fine[j]

print(sentences_paired[0])









## DELETE THIS BIT

# for i,sentence in enumerate(sentences_paired[0:15]):
#     new_coarse = []
    
#     for j,pair in enumerate(sentence):
#         new_coarse.append(pair[1])
    
#     print(i)
#     replace_coarse = Labels(new_coarse)

#     for j,pair in enumerate(sentence):
#         sentences_paired[i][j][1] = replace_coarse[j]

# sentences_paired[7]
# sentences_paired[10]
# sentences_paired[11]
# sentences_paired[12]
# sentences_paired[13]
# sentences_paired[14]


## DELETE THIS BIT











#CONVERT BACK TO STRING
finalized =[]
for i,sentence in enumerate(sentences_paired):
    tag_tab = []
    
    for j,pair in enumerate(sentence):
        #print(pair)
        tag_tab.append("\t".join(pair[0:2]))
    
    line_brk = "\n".join(tag_tab)
    finalized.append(line_brk+"\n\n")

#print(finalized)


#WRITE
with open("H:/My Files/School/Grad School WLU/MRP/Research/Data/FewNERD/devfin.txt","w", encoding='UTF-8') as f:
    f.write("".join(finalized))
















#df = pd.read_csv(dir_test, sep="\t")
#df.columns = ["token","entity"]

#for index, row in df.iterrows():
    
#    print(row["token"])
















































