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











































#####################
#### SPLIT SET ######
#####################
df=gold_dataframe()

total_count = 0
SIZE = 500
in_list = []
out_list = []
in_SPA = []
sentence_range = list(range(3,14)) #25 also works?
for idx,entry in enumerate(SPA):
    if entry in sentence_range and total_count <= SIZE:
        total_count+=entry
        in_list.append(idx)
        in_SPA.append(entry)
    else:
        out_list.append(idx)
print(in_SPA)
print(in_list)
print(total_count)
print(out_list)

#Filter
df_train = df.iloc[in_list]
df_test = df.iloc[out_list]

def to_text_file(df,file_name):
    token_list = []
    tag_list = []
    for index, row in df.iterrows():
        tokens = df["Article"][index].split(" ")
        tags = df["Entity"][index].split(" ")
        
        if tokens[-1]!=".": tokens.append("."), tags.append("O"), print("Added a period on article"+str(index))
        
        token_list.append(tokens)
        tag_list.append(tags)
    
    #Confirm No Mismatch
    for index,entry in enumerate(tag_list):
        if len(tag_list[index])!=len(token_list[index]):
            print("MISMATCH on ARTICLE "+str(index))
    
    #token_list[0]
    
    
    
    with open('H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/wikigold_'+file_name+'.txt', 'w', encoding="utf-8") as f:
        #Each Article
        for index1,article in enumerate(token_list):
            #Each Sentence
            for index2,tkn in enumerate(article):
                to_append = tkn+" "+tag_list[index1][index2]
                f.write(to_append)
                f.write("\n")
                
                if tkn==".":
                    f.write('\n')
            
            f.write("-DOCSTART- O\n\n")
    
    return

to_text_file(df_train, "train")
to_text_file(df_test, "test")












################
### SEGMENTS ###
################
everything = list(range(0,145))
#XSMALL ~ 50 // SMALL ~ 100 // MEDIUM ~ 250 // LARGE ~ 500
train_size_xsmall = [0, 5, 8, 9, 10, 11, 13, 15]
train_size_small = [0, 5, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 21, 26, 28]
train_size_medium = [0, 5, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 21, 26, 28, 29, 32, 34, 35, 37, 40, 41, 44, 45, 46, 47, 49, 50, 53, 54, 56, 58, 59, 63, 64, 66, 67]
train_size_large = [0, 5, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 21, 26, 28, 29, 32, 34, 35, 37, 40, 41, 44, 45, 46, 47, 49, 50, 53, 54, 56, 58, 59, 63, 64, 66, 67, 68, 69, 70, 71, 76, 77, 78, 79, 81, 83, 84, 85, 86, 88, 91, 92, 94, 95, 97, 98, 100, 102, 104, 105, 107, 108, 109, 110, 112, 114, 115, 117, 121, 122, 125, 127, 129, 132, 133, 134, 136, 138, 139]

train_size_xsmall_sent = [6, 8, 7, 7, 3, 5, 13, 4]
train_size_small_sent = [6, 8, 7, 7, 3, 5, 13, 4, 5, 6, 10, 4, 8, 11, 6]
train_size_medium_sent = [6, 8, 7, 7, 3, 5, 13, 4, 5, 6, 10, 4, 8, 11, 6, 6, 11, 3, 6, 12, 6, 11, 4, 10, 4, 3, 7, 5, 4, 3, 3, 4, 4, 13, 8, 13, 13]
train_size_large_sent = [6, 8, 7, 7, 3, 5, 13, 4, 5, 6, 10, 4, 8, 11, 6, 6, 11, 3, 6, 12, 6, 11, 4, 10, 4, 3, 7, 5, 4, 3, 3, 4, 4, 13, 8, 13, 13, 3, 4, 6, 10, 9, 12, 7, 7, 6, 4, 7, 7, 4, 7, 3, 3, 3, 5, 4, 4, 5, 7, 5, 3, 5, 6, 5, 3, 10, 4, 7, 3, 13, 3, 3, 13, 11, 7, 6, 6, 5]

test_size_xsmall = list(set(everything) - set(train_size_xsmall))
test_size_small = list(set(everything) - set(train_size_small))
test_size_medium = list(set(everything) - set(train_size_medium))
test_size_large = list(set(everything) - set(train_size_large))

training_segments = [train_size_xsmall, train_size_small, train_size_medium, train_size_large]
training_sentences = [train_size_xsmall_sent, train_size_small_sent, train_size_medium_sent, train_size_large_sent]
testing_segments = [test_size_xsmall, test_size_small, test_size_medium, test_size_large]


# ###########################
# ### LOADING & SELECTING ###
# ###########################
# variants = ["Article","Tagged_One","Tagged_Uni"]
# #variant = variants[0]

# for variant in variants:
#     print(variant)
#     df=pd.read_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/"+variant+".csv", encoding="utf-8", index_col=0)
#     df_testing = df.iloc[test_size_xsmall].reset_index(drop=True)
    
#     #Setup
#     df_training = df.iloc[train_size_xsmall].reset_index(drop=True)
#     df_training_scores = df_training.copy(deep=True)
#     df_training_mapped = df_training.copy(deep=True)
#     augment_multiple = 3
#     f1_threshold = 0.2
    
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
#     #Set Sample Headers to Iterate
#     headers = []
#     for i in range(1,51):
#         headers.append("Sample"+str(i))
    
#     #Iterate and Apply Scores
#     for index,row in df_training.iterrows():
#         original = df_training[variant][index]
        
#         for header_title in headers:
#             new = df_training[header_title][index]
#             scores = scorer.score(original, new)
#             f1_score = scores["rouge1"][2]
            
#             if f1_score > f1_threshold:    
#                 df_training_scores[header_title][index] = f1_score
#             elif f1_score < f1_threshold and new != " ":
#                 df_training_scores[header_title][index] = "Low Score"
#             else:
#                 df_training_scores[header_title][index] = "NA"
    
#     selection_dict = {}
#     for index,row in df_training.iterrows():
#         possible_shuffles = math.factorial(training_sentences[0][index])
#         NA_samples = list(df_training_scores.iloc[index][headers]).count("NA")
#         LowScore_samples = list(df_training_scores.iloc[index][headers]).count("Low Score")
        
#         max_possible = min(50-NA_samples-LowScore_samples,possible_shuffles)
#         samples_to_take = min(training_sentences[0][index]*augment_multiple,max_possible)
        
#         score_list = list(df_training_scores.iloc[index][headers])
#         for i,score in enumerate(score_list):
#             if type(score) == str:
#                 score_list[i]=0
                
#         index_order = sorted(range(len(score_list)), key=lambda k: score_list[k], reverse=True)
#         final_list = index_order[0:samples_to_take]
#         final_list = ["Sample"+str(x+1) for x in final_list]
        
#         selection_dict[index] = final_list
    
    
    
#     print(selection_dict)
#     print("\n\n")





###############
### MAPPING ###
###############

#See Mappings.Py






































































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