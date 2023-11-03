# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 22:12:28 2022

@author: Doug
"""

################
# LIBRARIES
################

import pandas as pd
import csv
import numpy as np
import tensorflow as tf
import time

#For Copies
import copy

#For Synonyms
import requests
from bs4 import BeautifulSoup

#For Shuffle
import random

#Hugging Face
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration, MT5Tokenizer, MT5ForConditionalGeneration, AutoModel, AutoTokenizer

#OS
import os.path
from os import path as os_path

#ITERTOOLS
import itertools

#ROUGE METRIC
from rouge_score import rouge_scorer

#FACTORIAL
import math

#NLTK
import nltk
nltk.download("punkt")

#MEMORY CLEARING
from GPUtil import showUtilization as gpu_usage
import torch
from numba import cuda

#################
##### CACHE #####
#################

#Change Cache
import os
os.environ['TRANSFORMERS_CACHE'] = 'H:/TempHF_Cache/cache/transformers/'
os.environ['HF_HOME'] = 'H:/TempHF_Cache/cache/'
os.environ['XDG_CACHE_HOME'] = 'H:/TempHF_Cache/cache/'

################
# CODE
################

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
    
    #Back to DF
    temp_dict = {"Article":article_list,"Entity":token_list}
    df = pd.DataFrame(temp_dict)
    return df

df=gold_dataframe()

def tagged_dataframe():
    df=gold_dataframe()
    df["Tagged_All"]=df["Article"]
    df["Tagged_One"]=df["Article"]
    df["Tagged_Uni"]=df["Article"]
    
    #List of Entities
    Entity_List = ["ORG","LOC","PER","MISC"]
    
    for index, row in df.iterrows():
        
        list_of_words = df["Tagged_All"][index].split(" ")
        list_of_tags = df["Entity"][index].split(" ")
        
        ###TAGGING ALL
        word_sentences = []
        tag_sentences = []
        
        word_segment = []
        tag_segment = []
        
        for i,word in enumerate(list_of_words):
            if word != ".":
                word_segment.append(word)
                tag_segment.append(list_of_tags[i])
            else:
                word_segment.append(word)
                word_sentences.append(word_segment)
                word_segment = []
                
                tag_segment.append(list_of_tags[i])
                tag_sentences.append(tag_segment)
                tag_segment = []
        
        for i,sentence in enumerate(word_sentences):
            for j,word in enumerate(sentence):
                tag = tag_sentences[i][j]
                if tag != "O":
                    word_sentences[i][j] = tag[2:]
        
        for i,sentence in enumerate(word_sentences): 
            word_sentences[i] = " ".join(sentence)
        
        df["Tagged_All"][index] = " ".join(word_sentences)
    
    
    
    
    
        #TAGGING SEGMENTS AS ONE        
        previous = ''
        segment_list = []
        tag_list = []
        temp1 = []
        temp2 = []
        
        for index2, tag in enumerate(list_of_tags):
            if tag == previous or previous == '':
                temp1.append(list_of_words[index2])
                temp2.append(tag)
            elif tag != previous:
                segment_list.append(temp1.copy())
                tag_list.append(temp2.copy())
                
                temp1.clear()
                temp2.clear()
                
                temp1.append(list_of_words[index2])
                temp2.append(tag)
            
            previous = tag
        
        segment_list.append(temp1)
        tag_list.append(temp2)
        
        for index3, group in enumerate(segment_list):
            segment_list[index3] = " ".join(group)
            tag_list[index3] = tag_list[index3][0]
        
        #print(segment_list)
        #print(tag_list)
        
        for index4, thingy in enumerate(segment_list):
            new_tag = tag_list[index4]
            if new_tag != "O":
                segment_list[index4] = new_tag[2:]
        
        
        df["Tagged_One"][index] = " ".join(segment_list)
        
        
        
        
        
        
        
        #TAGGING UNIQUE
        Entity_List_New = Entity_List.copy()
        list_of_words_tagged = df["Tagged_One"][index].split(" ")
        #list_of_words = df["Article"][index].split(" ")
        #If time, make numbering unique i.e. if band shows up twice give same # for it
        
        for i, word in enumerate(list_of_words_tagged):
            if word in Entity_List:
                Tag_Index = Entity_List.index(word)
                Original_Tag = Entity_List_New[Tag_Index]
                
                if Original_Tag[-1].isdigit():
                    New_Tag = Original_Tag[:-1]+str(int(Original_Tag[-1])+1)
                else:
                    New_Tag = Original_Tag+"1"
                    
                Entity_List_New[Tag_Index] = New_Tag
                list_of_words_tagged[i] = New_Tag
        
        df["Tagged_Uni"][index] = " ".join(list_of_words_tagged)
        
    return df






#############
### STATS ###
#############
df = tagged_dataframe()

WPA = []
SPA = []
EPA = {"O":[],"I-ORG":[],"I-PER":[],"I-LOC":[],"I-MISC":[]}
CPA = []

for index, row in df.iterrows():
    
    list_of_words = df["Article"][index].split(" ")
    list_of_tags = df["Entity"][index].split(" ")
    
    list_of_sentence_words = df["Article"][index].split(". ")

    words_per_article = len(list_of_words)
    sent_per_article = len(list_of_sentence_words)

    WPA.append(words_per_article)
    SPA.append(sent_per_article)
    
    for key in EPA.keys():
        EPA[key].append(list_of_tags.count(key))
    
    
    cnt = 0
    for word in list_of_words:
        if word in ["(",")",".",",","'",'"',"[","]","-","{","}",":",";"]:
           cnt+=1 
    
    CPA.append(cnt)

sum(CPA)
sum(EPA["I-ORG"]+EPA["I-LOC"]+EPA["I-PER"]+EPA["I-MISC"])

SPA_unique = np.unique(SPA).tolist()
SPA_counts = [SPA.count(num) for num in SPA_unique]

WPA_unique = np.unique(WPA).tolist()
WPA_counts = [WPA.count(num) for num in WPA_unique]



































######################
#### ARTICLE VERS ####
######################
import ast
sizes = [50,100,250,500]

id_dict = {}
for size in sizes:
    with open("H:\My Files\School\Grad School WLU\MRP\Research\Files\Data\Textfiles\\Article\\"+str(size)+"\\000_ID_LIST.txt") as indx_lst:
        for line in indx_lst:
            version = line[1]
            ids = ast.literal_eval(line[4:])
            id_dict[str(size)+"v"+version] = ids


set_vers = "Train"
df=gold_dataframe()
sent_length_dict = {}
art_cnt_dict = {}
O_cnt_dict = {}
NO_cnt_dict = {}
for size in [50,100,250,500]:
    batch_avg_sent_length=[]
    batch_tot_sent_length=[]
    vers_art_cnt = []
    vers_O_cnt = []
    vers_NO_cnt = []
    
    for version in range(0,10):
        index_ids_to_use = id_dict[str(size)+"v"+str(version)]
        
        if set_vers == "Test":
            index_ids_to_use = list(set(list(range(0,145)))-set(index_ids_to_use))
            
        new_df=df.iloc[index_ids_to_use]
        
        sentence_lengths = []
        article_cnt = 0
        O_tag_cnt = []
        NO_tag_cnt = []
        for index,row in new_df.iterrows():
            sentence_lengths.append(SPA[index])
            article_cnt+=1
            
            tag_splitter = new_df["Entity"][index].split(" ")
            O_tag_cnt.append(tag_splitter.count("O"))
            NO_tag_cnt.append(tag_splitter.count("I-PER")+tag_splitter.count("I-LOC")+tag_splitter.count("I-ORG")+tag_splitter.count("I-MISC"))
            
        vers_art_cnt.append(article_cnt)
        tot_sentences = sum(sentence_lengths)
        avg_sentence_length = sum(sentence_lengths)/len(sentence_lengths)
        batch_avg_sent_length.append(avg_sentence_length)
        batch_tot_sent_length.append(tot_sentences)
        
        vers_O_cnt.append(sum(O_tag_cnt))
        vers_NO_cnt.append(sum(NO_tag_cnt))
        
    avg_batch_sent_length = sum(batch_avg_sent_length)/len(batch_avg_sent_length)
    tot_batch_sent_length = sum(batch_tot_sent_length)/len(batch_tot_sent_length)
    tot_batch_art_cnt = sum(vers_art_cnt)/len(vers_art_cnt)
    avg_O_cnt = sum(vers_O_cnt)/len(vers_O_cnt)
    avg_NO_cnt = sum(vers_NO_cnt)/len(vers_NO_cnt)
    
    art_cnt_dict[size]=tot_batch_art_cnt
    sent_length_dict[size]=tot_batch_sent_length
    O_cnt_dict[size]=avg_O_cnt
    NO_cnt_dict[size]=avg_NO_cnt
    
print(sent_length_dict)
print(art_cnt_dict)
print(O_cnt_dict)
print(NO_cnt_dict)
























































##############
#### PLOT ####
##############
#https://matplotlib.org/stable/gallery/statistics/histogram_multihist.html

import matplotlib.pyplot as plt


#SENTENCES
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = SPA_unique
students = SPA_counts
ax.bar(langs,students)
plt.xlabel("Sentences in Articles")
plt.ylabel("# of Occurences")
plt.title('Frequency Distribution - # of Sentences in Articles')
plt.text(15, 8, 'Avg # of Sentences per Article: '+str(round(sum(SPA)/len(SPA),2)))
plt.show()

#WORDS
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = WPA_unique
students = WPA_counts
ax.bar(langs,students)
plt.xlabel("Sentences in Articles")
plt.ylabel("# of Occurences")
plt.title('Frequency Distribution - # of Sentences in Articles')
plt.text(6, 3, 'Avg # of Sentences per Article: '+str(round(sum(SPA)/len(SPA),2)))
plt.show()


#ENTITIES
colors = ["red","blue","green","black","orange"][1]
entity_lab = list(EPA.keys())[1]
entity_cnt = list(EPA.values())[1]
fig = plt.figure()
plt.hist(entity_cnt, density=False, bins=50, stacked=True, color=colors, label=entity_lab)
plt.legend(prop={'size': 10})
plt.ylabel('# of Occurences')
plt.xlabel('# of Words per Article');
plt.show()