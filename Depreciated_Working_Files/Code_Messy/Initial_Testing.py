# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 09:08:15 2022

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
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
#from transformers import GPT2Tokenizer, GPT2For
#from transformers import TransformerSummarizer

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
tokenizer_PEGASUS = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model_PEGASUS = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to(torch_device)

tokenizer_T5 = T5Tokenizer.from_pretrained("t5-small")
model_T5 = T5ForConditionalGeneration.from_pretrained("t5-small").to(torch_device)

tokenizer_BART = BartTokenizer.from_pretrained("facebook/bart-base")
model_BART = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(torch_device)










###################
### DATAFRAME #####
###################
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

####################
###### GEN #########
####################
def generate_output(
    input_text, num_return_sequences, num_beams, min_length, temperature=1.5
):
    #PEGASUS XSUM
    batch_PEGASUS = tokenizer_PEGASUS(input_text, truncation=True, padding="longest", return_tensors="pt").to(torch_device)
    translated_PEGASUS = model_PEGASUS.generate(**batch_PEGASUS, temperature=temperature, min_length=min_length,
                                  num_beams=num_beams, num_return_sequences=num_return_sequences,
                                  repetition_penalty=2.0)#,
                                  #do_sample=True, top_p = 0.9)
    result_PEGASUS = tokenizer_PEGASUS.batch_decode(translated_PEGASUS, skip_special_tokens=True)
    
    #T5
    batch_T5 = tokenizer_T5("summarize: "+input_text, truncation=True, padding="longest", return_tensors="pt").to(torch_device)
    translated_T5 = model_T5.generate(**batch_T5, temperature=temperature, min_length=min_length,
                                  num_beams=num_beams, num_return_sequences=num_return_sequences)
    result_T5 = tokenizer_T5.batch_decode(translated_T5, skip_special_tokens=True)
    
    #BART
    batch_BART = tokenizer_BART(input_text, truncation=True, padding="longest", return_tensors="pt").to(torch_device)
    translated_BART = model_BART.generate(**batch_BART, temperature=temperature, min_length=min_length,
                                  num_beams=num_beams, num_return_sequences=num_return_sequences)
    result_BART = tokenizer_BART.batch_decode(translated_BART, skip_special_tokens=True)
    
    
    return result_PEGASUS, result_T5, result_BART

df=gold_dataframe()
df["PEGASUS"] = " "
df["T5"] = " "
df["BART"] = " "

df=df.iloc[0:25]

for index,row in df.iterrows():
    print(index)
    article_length = len(df["Article"][index].split(" "))
    res_peg, res_T5, res_bar = generate_output(df["Article"][index], 1, 32, article_length, None)
    
    df["PEGASUS"][index] = res_peg[0]
    df["T5"][index] = res_T5[0]
    df["BART"][index] = res_bar[0]

df.to_csv("H:/My Files/School/Grad School WLU/MRP/EarlyTesting/NumBeamsPen.csv")


































