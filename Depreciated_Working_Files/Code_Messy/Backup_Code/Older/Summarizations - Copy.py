# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 06:10:36 2022

@author: Doug
"""
################
# LIBRARIES
################
import time
from timeit import default_timer as timer

import pandas as pd
import csv
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

#Hugging Face
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

#OS
import os.path
from os import path as os_path

################
# CODE
################
#https://huggingface.co/spaces/Wootang01/Paraphraser_two/blob/main/app.py

def new_model(model_name, path, torch_device):
    #First Load
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    
    #Save
    tokenizer.save_pretrained(path)
    model.save_pretrained(path)
    return tokenizer, model
    
def load_model(path, torch_device):
    #Load from Local Path
    tokenizer = PegasusTokenizer.from_pretrained(path)
    model = PegasusForConditionalGeneration.from_pretrained(path).to(torch_device)
    return tokenizer, model

def paraphrase(
    input_text, tokenizer, model, num_return_sequences, num_beams, max_length=60, temperature=1.5
):
    
    batch = tokenizer(
        [input_text], truncation=True, padding="longest", max_length=max_length, return_tensors="pt"
        ).to(torch_device)
    
    translated = model.generate(
        **batch, max_length=max_length, num_beams=num_beams, num_return_sequences=num_return_sequences,
        temperature=temperature)
    
    output = tokenizer.batch_decode(translated, skip_special_tokens=True)
    
    return output


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

def sentence_iter(num_samples, max_length, num_beams, temperature, df_name):
    for i in range(num_samples):
        header = "Sample" + str(i+1)
        df[header] = " "

    for index, row in df.iterrows():
        print(index)
        new_article = []
        
        current = df['Article'][index].split(" .")
        current = current[:-1] #removes final empty portion
        
        for entry in current: #Sentence in Article
            #print(entry)
            results = paraphrase(
                input_text=entry,
                tokenizer=tokenizer,
                model=model,
                num_return_sequences=num_samples,
                num_beams=num_beams,
                max_length=max_length,
                temperature=temperature)
            #print(results)
            
            new_article.append(results)
            
            #print("Got to end of this sentence")
            
        #print(new_article)
        np_array = np.array(new_article)
        sentence_list = np_array.T.tolist()
        
        #print(sentence_list)
        
        for i in range(num_samples):
            header = "Sample" + str(i+1)
            df[header][index] = " ".join(sentence_list[i])
            #print(df[header][index])
            #df["Pegasus Chunks"][index].replace(". ."," .")
    
    df.to_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/"+df_name+".csv")

    return df








#Get Data
df = gold_dataframe()

#Set Device
torch_device = "cuda" #If throwing CUDA error, restart Python

#Select Model
model_name = "tuner007/pegasus_paraphrase"
path = "H:/My Files/School/Grad School WLU/MRP/Research/Files/Models/Pegasus_Para"

if not os_path.exists(path):
    print("Path did not exist -- generating local model save.")
    tokenizer, model = new_model(model_name=model_name, path=path, torch_device=torch_device)
else:
    print("Model Found -- using existing model.")
    tokenizer, model = load_model(path=path, torch_device=torch_device)

sentence_iter(num_samples=10, max_length=60, num_beams=10, temperature=5, df_name="df_beams10_temp5")
#Add # of Articles as an option to adjust for
#Also try Top K and Top P







#Cache Models
tokenizer1 = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model1 = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to(torch_device)

from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer2 = T5Tokenizer.from_pretrained("t5-base")
model2 = T5ForConditionalGeneration.from_pretrained("t5-base").to(torch_device)

def abs_summaries(
    input_text, num_return_sequences, num_beams, min_length, temperature=1.5
):
    #PEGASUS XSUM
    batch1 = tokenizer1(input_text, truncation=True, padding="longest", return_tensors="pt").to(torch_device)
    translated1 = model1.generate(**batch1, temperature=temperature, min_length=min_length,
                                  num_beams=num_beams, num_return_sequences=num_return_sequences,top_p=0.9)
    Pegasus = tokenizer1.batch_decode(translated1, skip_special_tokens=True)
    
    #T5 
    #batch2 = tokenizer2(input_text, truncation=True, padding="longest", return_tensors="pt").to(torch_device)
    #translated2 = model2.generate(**batch2, temperature=temperature, min_length=min_length,
    #                              num_beams=num_beams, num_return_sequences=num_return_sequences)
    #T5 = tokenizer2.batch_decode(translated2, skip_special_tokens=True)
    
    #BART
    #TBD

    return Pegasus#, T5

####################
#SUMMARY GENERATION#
####################

def article_iter(shuffled, num_samples, num_beams, temperature, df_name):
    for i in range(num_samples):
        header = "Sample" + str(i+1)
        df[header] = " "

    for index, row in df.iterrows():
        print(index)
        
        if shuffled==False:
            article_txt = df["Article"][index]
        else:
            current = df['Article'][index].split(" .")
            current = current[:-1] #removes final empty portion
            random.shuffle(current)
            article_txt = " .".join(current)
        
        prop_length = article_txt.count(" ")#/6 #arbitrarily Picked
        
        results = abs_summaries(
            input_text = article_txt,
            num_return_sequences=num_samples,
            num_beams=num_beams,
            min_length=int(prop_length),
            temperature=temperature
            )
        
        for i in range(num_samples):
            header = "Sample" + str(i+1)
            df[header][index] = results[i]
    
    df.to_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/"+df_name+".csv")

    return df

df = gold_dataframe()
article_iter(True, 5, 32, 3, "pegasus_32Beam_5Sample_full_shuffle7")

####################
#SENTENCE WEIGHTING#
####################

def weighted_entity_order(high_first):

    #df["Weights"] = " "
    #df["Sentences"] = " "
    
    for index, row in df.iterrows(): #Each Article
        #print(index)
        #print(df["Article"][index])
        
        tokens = df['Article'][index].split(" ")
        entities = df['Entity'][index].split(" ")
        sentences = df['Article'][index].split(" .")[:-1]
        
        for i,sentence in enumerate(sentences):
            sentences[i] = sentence+" ."
        
        #print(tokens)
        #print(entities)
        #print(sentences)
        
        len_sentence = 0
        num_entities = 0
        weights = []
        
        for i, token in enumerate(tokens):
            if token == ".":
                wt_sentence = num_entities/len_sentence
                weights.append(wt_sentence)
                len_sentence = 0
                num_entities = 0
            elif entities[i]!="O":
                len_sentence += 1
                num_entities += 1
            else:
                len_sentence += 1
            
        #df["Sentences"][index] = sentences
        #df["Weights"][index] = weights
        #print(weights)
        
        order = sorted(range(len(weights)), key=lambda k: weights[k], reverse=high_first)
        new_order = [""]*len(weights)
    
        for i,val in enumerate(order):
            new_order[i] = sentences[val]
        
        new_order = "".join(new_order)
        df["Article"][index] = new_order
        
        #print("\n\n\n")
        #print(weights)
        #print(sentences)
        #print(order)
        #print(new_order)
        
    return df



df = gold_dataframe()
weighted_entity_order(high_first=False)
article_iter(False, 10, 32, 3, "pegasus_32Beam_5Sample_F_weighted")



























def release_cuda_mem():

    from GPUtil import showUtilization as gpu_usage
    
    def free_gpu_cache():
        print("Initial GPU Usage")
        gpu_usage()                             
    
        torch.cuda.empty_cache()
    
        cuda.select_device(0)
        cuda.close()
        cuda.select_device(0)
    
        print("GPU Usage after emptying the cache")
        gpu_usage()
    
    free_gpu_cache()
    
    return


release_cuda_mem()


































testing = "This album proved to be more commercial and more techno-based than Osc-Dis , with heavily synthesized songs like Introduction 010 and Come . Founding member Kojima Minoru played guitar on Good Day , and Wardanceis cover of a song by UK post punk industrial band Killing Joke . XXX can of This had a different meaning , and most people did n't understand what the song was about . it was later explained that the song was about Cannabis ( ' can of this ' sounding like Cannabis when said faster ) it is uncertain if they were told to change the lyric like they did on P.O.P and HUMANITY . UK Edition came with the OSC-DIS video , and most of the tracks were re-engineered ."
testing = "translate English to German: "+testing

testing2 = "translate English to German: My name is John."



batch2 = tokenizer2(testing2, truncation=True, padding="longest", return_tensors="pt").to(torch_device)
translated2 = model2.generate(**batch2,temperature=4,
                              num_beams=32, num_return_sequences=5)
t5_out = tokenizer2.batch_decode(translated2, skip_special_tokens=True)
print(t5_out)










###For Testing::
#my_text = "After burying the dead on the field of Second Battle of Bull Run , the regiment was attached to Howe 's Brigade of Couch 's Division of the IV Corps of the Army of the Potomac where it replaced De Trobriand 's 55th New York , Gardes Lafayette regiment on September"# 11 , 1862 ."
#my_text = "This is simply a good test"

#paraphrase(
#    input_text=my_text,
#    tokenizer=tokenizer,
#    model=model,
#    num_return_sequences=10,
#    num_beams=10,
#    max_length=60,
#    temperature=1)

# print(testing)
# batch1 = tokenizer1(testing, truncation=True, padding="longest", return_tensors="pt").to(torch_device)
# translated1 = model1.generate(**batch1, temperature=6, min_length=testing.count(" ")*12, num_beams=32, num_return_sequences=2)
# tgt_text = tokenizer1.batch_decode(translated1, skip_special_tokens=True)
# print("\n\n")
# print(tgt_text)

# print(testing)
# batch2 = tokenizer2(testing, truncation=True, padding="longest", return_tensors="pt").to(torch_device)
# translated2 = model2.generate(**batch2, temperature=1, min_length=30,max_length=50, num_beams=8)
# tgt_text = tokenizer2.batch_decode(translated2, skip_special_tokens=True)
# print("\n\n")
# print(tgt_text)


# def load_model():
#   model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
#   return model

# def get_response(
#   input_text, num_return_sequences, num_beams, max_length=512, temperature=1.5
# ):

#   model = load_model()
  
#   batch = tokenizer(
#       [input_text], truncation=True, padding="longest", max_length=max_length, return_tensors="pt"
#       ).to(torch_device)
  
#   translated = model.generate(
#       **batch, max_length=max_length, num_beams=num_beams, num_return_sequences=num_return_sequences,
#       temperature=temperature)
  
#   tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  
#   return tgt_text

# get_response(
#     input_text=my_text,
#     num_return_sequences=10,
#     num_beams=10,
#     temperature=1.5)




# #SAMPLE FROM MODEL CREATOR
# import torch
# from transformers import PegasusForConditionalGeneration, PegasusTokenizer
# model_name = 'tuner007/pegasus_paraphrase'
# torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
# tokenizer = PegasusTokenizer.from_pretrained(model_name)
# model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

# def get_response(input_text,num_return_sequences,num_beams):
#   batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
#   translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
#   tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
#   return tgt_text

# num_beams = 10
# num_return_sequences = 10
# context = "The ultimate test of your knowledge is your capacity to convey it to another."
# get_response(context,num_return_sequences,num_beams)


