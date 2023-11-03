# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:08:28 2022

@author: Doug
"""

import random
import numpy as np
import copy
import pandas as pd
import csv


##############
#### DF ######
##############

def gold_dataframe():
    #Create List of Articles, Tokens
    df = pd.read_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/wikigold.txt",
                     sep=' ', header=None, doublequote = True, quotechar='"',
                     skipinitialspace = False, quoting=csv.QUOTE_NONE)
    df.columns = ["Token","Entity"]
    
    current_article, current_token, article_list, token_list = [], [], [] ,[]
    
    for index, row in df.iterrows():
      if df["Token"][index] != "-DOCSTART-":
        current_article.append(df["Token"][index])
        current_token.append(df["Entity"][index])
      else:
        article_list.append(current_article), token_list.append(current_token)
        current_article, current_token = [], []
    
    for index in range(len(article_list)):
      article_list[index] = " ".join(article_list[index])
      token_list[index] = " ".join(token_list[index])
    
    #Check it worked
    for index in range(len(article_list)):
      pass
      #print(article_list[index],"\n",token_list[index])
    
    #Back to DF
    temp_dict = {"Article":article_list,"Entity":token_list}
    df = pd.DataFrame(temp_dict)
    return df








###SETUP
df = gold_dataframe()
word_sentences = []
tag_sentences = []
for index, row in df.iterrows():
    
    words = df["Article"][index].split(" ")
    tags = df["Entity"][index].split(" ")
    
    #print(words)
    #print(tags)
    
    word_segment = []
    tag_segment = []
    for i,word in enumerate(words):
        if word != ".":
            word_segment.append(word)
            tag_segment.append(tags[i])
        else:
            word_segment.append(word)
            word_sentences.append(word_segment)
            word_segment = []
            
            tag_segment.append(tags[i])
            tag_sentences.append(tag_segment)
            tag_segment = []
            


word_sentences_flat = np.hstack(word_sentences)
tag_sentences_flat = np.hstack(tag_sentences)

print(word_sentences_flat)
print(tag_sentences_flat)

### MAP WORDS TO TAGS
token_map = pd.DataFrame({"Words":word_sentences_flat, "Tags":tag_sentences_flat})
tag_list = token_map["Tags"].unique().tolist()
tag_groups = token_map.groupby("Tags")["Words"].apply(list)

#random.choice(tag_groups["I-LOC"])

rate = 0.2

##########
###LWTR###
##########

DA_word_sentences = copy.deepcopy(word_sentences) #NEED TO IMPLIMENT
#IDEAL USE: GENERATE 1/2/5/10 FROM ORIGINAL SENTENCE

for i,sentence in enumerate(word_sentences):
    for j,word in enumerate(sentence):
        tag = tag_sentences[i][j]
        
        if np.random.binomial(1, rate, size=None):
            substitute = random.choice(tag_groups[tag])
            word_sentences[i][j] = substitute
            #word_sentences.append(substitute)
            #tag_sentences.append(tag_sentences[i])
        
print(word_sentences[-1])
print(tag_sentences[-1])   


##########
####SR####
##########

import nltk
from nltk.corpus import wordnet

SR_Issue_List = ["in","In","It","it","does","Does","IAEA","have","Have","be","Be","less","Less","He","he","Pesos","Inc","inc","acts","Acts","an","An","units"]

for i,sentence in enumerate(word_sentences):
    for j,word in enumerate(sentence):
        tag = tag_sentences[i][j]
        
        if np.random.binomial(1, rate, size=None):
            try:
                substitute = wordnet.synsets(word)[0].lemmas()[0].name()
                if word not in SR_Issue_List:
                    word_sentences[i][j] = substitute
            except:
                word_sentences[i][j] = word

print(word_sentences[0])
print(tag_sentences[0])  


###########
####SIS####
###########

for i,sentence in enumerate(word_sentences):
    
    previous = ''
    segment_list = []
    temp1 = []

    
    for j,word in enumerate(sentence):
        tag = tag_sentences[i][j]
        
        if tag == previous or previous == '':
            temp1.append(word)
        elif tag != previous:
            segment_list.append(temp1.copy())
            
            temp1.clear()
            
            temp1.append(word)
        
        previous = tag
    
    segment_list.append(temp1)
    
    #print(segment_list)
    
    for j,entry in enumerate(segment_list):
        if np.random.binomial(1, rate, size=None):
            random.shuffle(entry)
        
    word_sentences[i] = np.hstack(segment_list)

print(word_sentences[0])
print(tag_sentences[0]) 













