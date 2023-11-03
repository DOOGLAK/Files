# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 15:21:27 2022

@author: Doug
"""
#from importlib.metadata import version
#version('transformers')

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

df = pd.read_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Article.csv", encoding="utf-8", index_col=0)

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





def rng_train_set(min_exclusion, max_exclusion, sentence_goal, bounds_rate):
    if min_exclusion<0 or max_exclusion<0 or min_exclusion>=max_exclusion or sentence_goal<0 or bounds_rate<0:
        print("Error in Variable Settings! Exited.")
        return
    
    in_bounds = False
    
    while in_bounds == False:
        original_article_list = list(range(0,145))
        
        filter_out_list = []
        filter_cnt = 0
        for i,length in enumerate(SPA):
            if length <= min_exclusion or length >= max_exclusion:
                filter_out_list.append(i)
                filter_cnt += length
        
        new_article_list = list(set(original_article_list)-set(filter_out_list))
        
        new_cnt = 0
        for index in new_article_list:
            new_cnt+=SPA[index]
        
        #Confirm
        if filter_cnt+new_cnt!=1768 and len(filter_out_list)+len(new_article_list)!=145:
            print("ERROR IN SENTENCE OR ARTICLE COUNT")
        
        #Get Avg SPA
        avg_spa = new_cnt / len(new_article_list)
        
        expected_iterations = math.ceil(sentence_goal/avg_spa)
        train_articles = random.sample(new_article_list, expected_iterations)
        test_articles = list(set(original_article_list)-set(train_articles))
        
        
        total_sentences = 0
        for article_index in train_articles:
            total_sentences += SPA[article_index]
        
        avg_sentences = total_sentences/len(train_articles)
        
        #Check Bounds Acceptance
        if bounds_rate >= 0 and bounds_rate < 1:    
            if total_sentences >= sentence_goal*(1-bounds_rate) and total_sentences <= sentence_goal*(1+bounds_rate):
                in_bounds = True
        else:
            if total_sentences >= sentence_goal - bounds_rate and total_sentences >= sentence_goal + bounds_rate:
                in_bounds = True
    
    return train_articles, test_articles, total_sentences, avg_sentences#, filter_out_list, new_article_list, avg_spa

#Run Replications for Sample Sizes
replications = 10
batch_pools = [50,100,250,500]
train_batch_dict = {50:[],100:[],250:[],500:[]}
test_batch_dict = {50:[],100:[],250:[],500:[]}

for size in batch_pools:
    print("Batch Size: "+str(size))
    for i in range(replications):
        train_list, test_list, cnt_sentences, avg_sentences = rng_train_set(
            min_exclusion=2
            , max_exclusion=40
            , sentence_goal=size
            , bounds_rate=0.1
            )
        
        train_batch_dict[size].append(train_list)
        test_batch_dict[size].append(test_list)









#Select Scored Samples
def final_selection(augmented_sentence_multiplier):
    variants = ["Article","Tagged_One","Tagged_Uni"]
    augment_multiple=3#augmented_sentence_multiplier
    
    #Set Sample Headers to Iterate
    headers = []
    for i in range(1,51):
        headers.append("Sample"+str(i))
    
    nested_dict = {}
    for variant in variants:
        nested_dict[variant] = {}
        for size in train_batch_dict:
            nested_dict[variant][size] = {}
            for attempt in range(replications):
                nested_dict[variant][size][attempt] = {}
                df=pd.read_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Finished/Original_"+variant+".csv", encoding="utf-8", index_col=0)
                df_scored = pd.read_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Finished/Scored_"+variant+".csv", encoding="utf-8", index_col=0)
                
                df=df.iloc[train_batch_dict[size][attempt]]
                
                for index,row in df.iterrows():
                    possible_shuffles = math.factorial(SPA[index])
                    NA_samples = list(df_scored.iloc[index][headers]).count("NA")
                    LowScore_samples = list(df_scored.iloc[index][headers]).count("Low Score")
                    
                    max_possible = min(50-NA_samples-LowScore_samples,possible_shuffles)
                    samples_to_take = min(SPA[index]*augment_multiple,max_possible)
                    
                    score_list = list(df_scored.iloc[index][headers])
                    for i,score in enumerate(score_list):
                        if score == "NA" or score == "Low Score":
                            score_list[i]=0
                        else:
                            score_list[i]=float(score_list[i])
                            
                    index_order = sorted(range(len(score_list)), key=lambda k: score_list[k], reverse=True)
                    final_list = index_order[0:samples_to_take]
                    final_list = ["Sample"+str(x+1) for x in final_list]
                    
                    cleaned_list=[]
                    for j,sample in enumerate(final_list):
                        if df[sample][index]!=" " or df[sample][index]!="NA" or not pd.isnull(df[sample][index]):
                            #print(variant+"-"+str(size)+"-"+str(attempt)+"-Article ID "+str(index)+sample)
                            cleaned_list.append(sample)    
                            #break
                        
                    #nested_dict[variant][size][attempt][index] = final_list
                    nested_dict[variant][size][attempt][index] = cleaned_list
                
                
                #print(nested_dict[variant][size][attempt][index])
                #print("\n")
                
    return nested_dict
            
fin_results = final_selection(3)









#########################
###### CREATE TXTS ######
#########################
full_list = list(range(0,145))

for variant in list(fin_results.keys()): #Variants (Article, One Tag, Unique Tag)
    #Set DFs    
    df_og=pd.read_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Finished/Original_"+variant+".csv", encoding="utf-8", index_col=0)
    df_mp=pd.read_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Finished/Mapped_"+variant+".csv", encoding="utf-8", index_col=0)
    
    for size in list(fin_results[variant].keys()): #Sizes (50/100/250/500)
        #Write Article IDs for Reference per Repetition Version  
        with open('H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/Textfiles/'+variant+'/'+str(size)+'/000_ID_LIST.txt', 'w', encoding="utf-8") as f_ids:
            for repetition in list(fin_results[variant][size].keys()): #Repetition (0-->10)
                #Write TOKEN/TAGs to 3 text files, test, original, and original+augmented  
                with open('H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/Textfiles/'+variant+'/'+str(size)+'/v'+str(repetition)+'_Augmented.txt', 'w', encoding="utf-8") as f_aug, open('H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/Textfiles/'+variant+'/'+str(size)+'/v'+str(repetition)+'_UnAugmented.txt', 'w', encoding="utf-8") as f_org, open('H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/Textfiles/'+variant+'/'+str(size)+'/v'+str(repetition)+'_Testing.txt', 'w', encoding="utf-8") as f_tst:
                    f_ids.write("v"+str(repetition)+":\t")
                    f_ids.write(str(list(fin_results[variant][size][repetition].keys())))
                    f_ids.write("\n")
                    
                    #Iterate through Articles, Then Samples
                    for article_id in list(fin_results[variant][size][repetition].keys()): #Article IDs
                        #Get the IDs for Articles
                        train_ids = list(fin_results[variant][size][repetition].keys())
                        test_ids = list(set(full_list)-set(train_ids))
                        
                        #token_list_train = []
                        #tag_list_train = []
                        
                        df_og_train = df_og.iloc[train_ids]
                        df_mp_train = df_mp.iloc[train_ids]
                        df_og_test = df_og.iloc[test_ids]
                        #df_mp_test = df_mp.iloc[test_ids]
                        
                        df_og_train=df_og_train.fillna("NA")
                        df_mp_train=df_mp_train.fillna("NA")
                        df_og_test=df_og_test.fillna("NA")
                        #df_og_train(replace)
                        
                        #Get Augmented Samples
                        sample_list = fin_results[variant][size][repetition][article_id]
                        if sample_list != []:
                            for sample in sample_list:
                                #SKIP NAs
                                if df_og_train[sample][article_id]=="NA":
                                    continue
                                
                                tokens_train = df_og_train[sample][article_id].split(" ")
                                tags_train = df_mp_train[sample][article_id].split(" ")
                        
                                if tokens_train[-1]!=".": tokens_train.append("."), tags_train.append("O")#, print("Added a period on article"+str(article_id))
                                
                                #token_list_train.append(tokens_train)
                                #tag_list_train.append(tags_train)
                                
                                #CONFIRM NO MISMATCH
                                for index,entry in enumerate(tags_train):
                                    if len(tags_train)!=len(tokens_train):
                                        print("MISMATCH on ARTICLE "+str(article_id)+" and "+sample)
                                        
                                for index,the_token in enumerate(tokens_train):
                                    to_append = the_token+" "+tags_train[index]
                                    f_aug.write(to_append)
                                    f_aug.write("\n")
                                    
                                    if the_token==".":
                                        f_aug.write('\n')
                                
                                f_aug.write("-DOCSTART- O\n\n") #SWAP BACK TO "-DOCSTART- O\n\n" LATER
                        #f_aug.write("-DOCSTART- O\n\n") #SWAP BACK
                        
                        #GET ORIGINAL UN-AUGMENTED SAMPLES FOR TRAINING
                        #GET ORIGINAL UN-AUGMENTED SAMPLES FOR TRAINING
                        #GET ORIGINAL UN-AUGMENTED SAMPLES FOR TRAINING
                        tokens_train = df_og_train["Article"][article_id].split(" ")
                        tags_train = df_og_train["Entity"][article_id].split(" ")
                        if tokens_train[-1]!=".": tokens_train.append("."), tags_train.append("O")#, print("Added a period on article"+str(article_id))
                        #CONFIRM NO MISMATCH
                        for index,entry in enumerate(tags_train):
                            if len(tags_train)!=len(tokens_train):
                                print("MISMATCH on ARTICLE "+str(article_id)+" and "+sample)
                        for index,the_token in enumerate(tokens_train):
                            to_append = the_token+" "+tags_train[index]
                            f_aug.write(to_append)
                            f_aug.write("\n")
                            f_org.write(to_append)
                            f_org.write("\n")
                            
                            if the_token==".":
                                f_aug.write('\n')
                                f_org.write('\n')
                        
                        f_aug.write("-DOCSTART- O\n\n") #SWAP BACK TO "-DOCSTART- O\n\n" LATER
                        f_org.write("-DOCSTART- O\n\n") #SWAP BACK TO "-DOCSTART- O\n\n" LATER
                        
                    #GET ORIGINAL UN-AUGMENTED SAMPLES FOR TESTING
                    #GET ORIGINAL UN-AUGMENTED SAMPLES FOR TESTING
                    #GET ORIGINAL UN-AUGMENTED SAMPLES FOR TESTING
                    for a_index,row in df_og_test.iterrows():
                        tokens_train = df_og_test["Article"][a_index].split(" ")
                        tags_train = df_og_test["Entity"][a_index].split(" ")
                        if tokens_train[-1]!=".": tokens_train.append("."), tags_train.append("O")#, print("Added a period on article"+str(article_id))
                        #CONFIRM NO MISMATCH
                        for index,entry in enumerate(tags_train):
                            if len(tags_train)!=len(tokens_train):
                                print("MISMATCH on ARTICLE "+str(a_index)+" and "+sample)
                        for index,the_token in enumerate(tokens_train):
                            to_append = the_token+" "+tags_train[index]
                            f_tst.write(to_append)
                            f_tst.write("\n")
                            
                            if the_token==".":
                                f_tst.write('\n')
    
                        f_tst.write("-DOCSTART- O\n\n") #SWAP BACK TO "-DOCSTART- O\n\n" LATER
























































