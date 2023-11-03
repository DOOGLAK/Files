# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 03:36:58 2022

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
#https://huggingface.co/spaces/Wootang01/Paraphraser_two/blob/main/app.py

#INITIAL SETUP
#Set Device
torch_device = "cuda" #If throwing CUDA error, restart Python

#Set Models
tokenizer1 = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model1 = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to(torch_device)






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
df = df.iloc[0:25].reset_index(drop=True)











def abs_summary(
    input_text, num_return_sequences, num_beams, min_length, temperature=1.5
):
    #PEGASUS XSUM

    batch1 = tokenizer1(input_text, truncation=True, padding="longest", return_tensors="pt").to(torch_device)
    translated1 = model1.generate(**batch1, temperature=temperature, min_length=min_length,
                                  num_beams=num_beams, num_return_sequences=num_return_sequences)#,do_sample=False,top_k=None)
    Pegasus = tokenizer1.batch_decode(translated1, skip_special_tokens=True)

    return Pegasus

def article_iter(shuffled, num_samples, num_beams, temperature, df_name):
    for i in range(num_samples):
        header = "Sample" + str(i+1)
        df[header] = " "

    for index, row in df.iterrows():
        print(index)
        
        if shuffled==False:
            article_txt = df["Article"][index]
        
        list_of_words = article_txt.split(" ")
        list_of_tags = df["Entity"][index].split(" ")
        
        new_article_txt = []
        
        for j,tag in enumerate(list_of_tags):
            if tag != "O":
                new_article_txt.append(tag[2:])
                new_article_txt.append(list_of_words[j])
            else:
                new_article_txt.append(list_of_words[j])
        
        article_txt = " ".join(new_article_txt)
        
        prop_length = article_txt.count(" ")#/6 #arbitrarily Picked
        
        results = abs_summary(
            input_text = article_txt,
            num_return_sequences=num_samples,
            num_beams=num_beams,
            min_length=int(prop_length),
            temperature=temperature
            )
        
        for i in range(num_samples):
            header = "Sample" + str(i+1)
            df[header][index] = results[i]
    
    df.to_csv("H:/My Files/School/Grad School WLU/MRP/EarlyTesting/Variants/Linearized.csv")

    return df


article_iter(False, 1, 32, 1.5, "idk")



