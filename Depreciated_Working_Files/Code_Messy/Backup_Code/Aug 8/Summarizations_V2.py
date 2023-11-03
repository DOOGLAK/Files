# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 17:46:46 2022

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

tokenizer2 = PegasusTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
model2 = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase").to(torch_device)


##############
#### DATA ####
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

df = tagged_dataframe()

###########################
###### SUMMARIZATION ######
###########################

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
        else:
            current = df['Article'][index].split(" .")
            current = current[:-1] #removes final empty portion
            random.shuffle(current)
            article_txt = " .".join(current)
        
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
    
    df.to_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/"+df_name+".csv")

    return df

def weighted_entity_order(high_first):
    
    for index, row in df.iterrows(): #Each Article
        print(index)
        
        tokens = df['Article'][index].split(" ")
        entities = df['Entity'][index].split(" ")
        sentences = df['Article'][index].split(" .")[:-1]
        
        for i,sentence in enumerate(sentences):
            sentences[i] = sentence+" ."
        
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
            
        order = sorted(range(len(weights)), key=lambda k: weights[k], reverse=high_first)
        new_order = [""]*len(weights)
    
        for i,val in enumerate(order):
            new_order[i] = sentences[val]
        
        new_order = "".join(new_order)
        df["Article"][index] = new_order
        
    return df


def run_samples_overnight(method_choice):
    tic = time.perf_counter()
    df = tagged_dataframe()
    shuffled = True
    num_shuffle=50
    method = method_choice
    
    for i in range(num_shuffle):
        header = "Sample" + str(i+1)
        df[header] = " "
    
    for index, row in df.iterrows():
        i=0
        cnt=0
        timeout_cnt = 0
        
        if shuffled==False:
            article_text = df[method][index]
            
            results = abs_summary(
                input_text = article_text,
                num_return_sequences=1,
                num_beams=32,
                min_length=int(article_text.count(" ")),
                temperature=4
                )
            
        if shuffled==True:
            while i < num_shuffle:
                current = df[method][index].split(" .")
                current = current[:-1] #removes final empty portion
                
                total_permutations = math.factorial(len(current))/1
                print("Article: "+str(index)+" and method: "+method)
                print("Permutations: "+str(total_permutations))
                print("Shuffle "+str(i+1))
                
                random.shuffle(current)
                article_text = " .".join(current)
                
                if i==0:
                    shuffle_list = []
                    shuffle_list.append(article_text)
                    i=i+1
                    cnt+=1
                
                elif article_text not in shuffle_list:
                    shuffle_list.append(article_text)
                    i=i+1
                    cnt+=1
                
                timeout_cnt +=1
                
                print("Count: "+str(cnt)+"\n")
                
                if cnt==total_permutations or timeout_cnt == num_shuffle*3:
                    print("Permutation Limit Reached\n")
                    break
            
            #Shuffling Complete
            results=[]
            print("Generating Results!")
            for i in range(cnt):
                
                summary_text = abs_summary(
                    input_text = shuffle_list[i],
                    num_return_sequences=1,
                    num_beams=32,
                    min_length=int(shuffle_list[i].count(" ")),
                    temperature=4
                    )
                
                results.append(summary_text)
        
        for i in range(cnt):
            header = "Sample" + str(i+1)
            df[header][index] = results[i][0]
    
    
    #USE IF NUM SHUFFLED=0 NOT TRUE FALSE
    df.to_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/"+method+".csv", encoding="utf-8")
    toc = time.perf_counter()
    
    seconds_taken = toc-tic
    minutes = seconds_taken/60
    print("Seconds: %0.4f" % seconds_taken)
    print("Minutes: %0.4f" % minutes)
    
    return minutes
    

#FOR OVERNIGHT SAVING OF FILES
#9 Hours to Run with: 4 Variations, 50 Samples
minutes_to_run = []
for method_chosen in ["Article","Tagged_All","Tagged_Uni","Tagged_One"]:
    print("Starting Method: "+method_chosen)
    mins = run_samples_overnight(method_choice=method_chosen)
    minutes_to_run.append(mins)
    
    gpu_usage()
    torch.cuda.empty_cache()
    torch_device="cuda"

print(sum(minutes_to_run))






###########################
###### PARAPHRASING ######
###########################

def paraphraser(
    input_text, num_return_sequences, num_beams, max_length=60, temperature=1.5
):
    #PEGASUS XSUM

    batch2 = tokenizer2(input_text, truncation=True, padding="longest", return_tensors="pt").to(torch_device)
    translated2 = model2.generate(**batch2, temperature=temperature, max_length=max_length,
                                  num_beams=num_beams, num_return_sequences=num_return_sequences)#,do_sample=False,top_k=None)
    Pegasus = tokenizer2.batch_decode(translated2, skip_special_tokens=True)

    return Pegasus


def run_para_overnight(method_choice):
    tic = time.perf_counter()
    df = tagged_dataframe()
    num_responses=5
    method = method_choice
    
    for i in range(num_responses):
        header = "Sample" + str(i+1)
        df[header] = " "
    
    for index, row in df.iterrows():
        print("Article #"+str(index))
        new_article = []
        
        current = df[method][index].split(" .")
        current = current[:-1] #removes final empty portion
        
        for entry in current: #Sentence in Article
            results = paraphraser(
                input_text=entry,
                num_return_sequences=num_responses,
                num_beams=5,
                max_length=60,
                temperature=1)
            
            new_article.append(results)
            
        np_array = np.array(new_article)
        sentence_list = np_array.T.tolist()
        
        for i in range(num_responses):
            header = "Sample" + str(i+1)
            df[header][index] = " ".join(sentence_list[i])
            
    
    df.to_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Para_"+method+".csv", encoding="utf-8")
    toc = time.perf_counter()
    
    seconds_taken = toc-tic
    minutes = seconds_taken/60
    print("Seconds: %0.4f" % seconds_taken)
    print("Minutes: %0.4f" % minutes)
            
    return minutes, df

#FOR OVERNIGHT SAVING OF FILES
#9 Hours to Run with: 4 Variations, 50 Samples
minutes_to_run = []
for method_chosen in ["Article","Tagged_All","Tagged_Uni","Tagged_One"]:
    print("Starting Method: "+method_chosen)
    mins, df = run_para_overnight(method_choice=method_chosen)
    minutes_to_run.append(mins)
    
    gpu_usage()
    torch.cuda.empty_cache()
    torch_device="cuda"

print(sum(minutes_to_run))




























#########RE-LOAD FILES
testing_df = pd.read_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Article.csv", encoding="utf-8", index_col=0)
#df.to_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/temp.txt", sep='\n', index=False)

#SAVE TO TXT
np.savetxt(
    "H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/temp.txt",
    df.values,
    fmt="%s",
    delimiter="\n\n",
    #header="X\tY\tZ\tValue",
    encoding="utf-8",
)





testing_para = testing_df["Sample3"][0]
tokens = nltk.sent_tokenize(testing_para)
for t in tokens:
    print (t, "\n")
    words = nltk.word_tokenize(t)
    print(" ".join(words))








num_shuffle=50
method="Article"

###########################
######## SCORING ##########
###########################
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

df_scores = pd.DataFrame(df["Article"])
for i in range(num_shuffle):
    column = "Sample"+str(i+1)
    df_scores[column] = " "

for index,row in df.iterrows():    
    for i in range(num_shuffle):
        column = "Sample"+str(i+1)
        
        scores = scorer.score(df[method][index], df[column][index][0])
        f1_score = scores["rouge1"][2]
        
        df_scores[column][index] = f1_score




score_dict = {}
for index,row in df_scores.iterrows():
    
    score_list = []
    
    for i in range(num_shuffle):
        column = "Sample"+str(i+1)
        
        score_list.append(df_scores[column][index])
    
    order = sorted(range(len(score_list)), key=lambda k: score_list[k], reverse=False)
    score_dict[index] = order
        

for keys,scores in score_dict.items():
    print("Article: "+str(keys))
    
    for index,ranking in enumerate(scores):
        if ranking in [1,2,3]:
            print(index+1, df_scores["Sample"+str(index+1)][keys])
    
    print("\n\n")




















































testing = "MISC is the tenth album from MISC Punk Techno band ORG ORG ORG ORG . This album proved to be more commercial and more techno-based than MISC , with heavily synthesized songs like MISC MISC and MISC . Founding member PER PER played guitar on MISC MISC , and MISC cover of a song by LOC post punk industrial band ORG ORG . MISC MISC MISC MISC had a different meaning , and most people did n't understand what the song was about . it was later explained that the song was about MISC ( ' can of this ' sounding like MISC when said faster ) it is uncertain if they were told to change the lyric like they did on MISC and MISC . LOC Edition came with the MISC video , and most of the tracks were re-engineered ."
results = abs_summary(
    input_text = testing,
    num_return_sequences=1,
    num_beams=32,
    min_length=142,
    temperature=1
    )
print(results)

testing = "MISC is the tenth album from MISC Punk Techno band ORGORGORGORG . This album proved to be more commercial and more techno-based than MISC , with heavily synthesized songs like MISCMISC and MISC . Founding member PERPER played guitar on MISCMISC , and MISC cover of a song by LOC post punk industrial band ORGORG . MISCMISCMISCMISC had a different meaning , and most people did n't understand what the song was about . it was later explained that the song was about MISC ( ' can of this ' sounding like MISC when said faster ) it is uncertain if they were told to change the lyric like they did on MISC and MISC . LOC Edition came with the MISC video , and most of the tracks were re-engineered ."
results = abs_summary(
    input_text = testing,
    num_return_sequences=1,
    num_beams=32,
    min_length=142,
    temperature=1
    )
print(results)

testing = "MISC is the tenth album from MISC Punk Techno band ORG . This album proved to be more commercial and more techno-based than MISC , with heavily synthesized songs like MISC and MISC . Founding member PER played guitar on MISC , and MISC cover of a song by LOC post punk industrial band ORG . MISC had a different meaning , and most people did n't understand what the song was about . it was later explained that the song was about MISC ( ' can of this ' sounding like MISC when said faster ) it is uncertain if they were told to change the lyric like they did on MISC and MISC . LOC Edition came with the MISC video , and most of the tracks were re-engineered ."
results = abs_summary(
    input_text = testing,
    num_return_sequences=1,
    num_beams=32,
    min_length=142,
    temperature=1
    )
print(results)

testing = "MISC1 is the tenth album from MISC2 Punk Techno band ORG1 . This album proved to be more commercial and more techno-based than MISC3 , with heavily synthesized songs like MISC4 and MISC5 . Founding member PER1 played guitar on MISC6 , and MISC7 cover of a song by LOC1 post punk industrial band ORG6 . MISC8 had a different meaning , and most people did n't understand what the song was about . it was later explained that the song was about MISC9 ( ' can of this ' sounding like MISC10 when said faster ) it is uncertain if they were told to change the lyric like they did on MISC11 and MISC12 . LOC2 Edition came with the MISC13 video , and most of the tracks were re-engineered ."
results = abs_summary(
    input_text = testing,
    num_return_sequences=1,
    num_beams=None,
    min_length=142,
    temperature=0.4
    )
print(results)

testing = "LOC Edition came with the MISC video , and most of the tracks were re-engineered ."
results = paraphraser(
    input_text=testing,
    num_return_sequences=3,
    num_beams=10,
    max_length=60,
    temperature=1.5)
print(results)




















df_taggedsamples=pd.read_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Article.csv", encoding="utf-8", index_col=0)
blacklist = ["of","the","This"]

def getNgrams(article, tags, n):

  if( n < 1 or n > len(tags) -1 ):
    raise Exception("n must be between 1 and total number of tags")

  if( len(article) != len(tags)):
    raise Exception("article length and tag length do not match")

  mapping = {}

  # sliding window of size n
  for i in range(len(tags) - n + 1):
    # collect sliding window of tags
    sequence = tags[i: i + n]
    # if the tag is real (NOT "O") and all the tags match
    if(sequence[0] != 'O' and len(set(sequence)) <= 1):
      # add the full sentence to the dictionary with a tag
      mapping[" ".join(article[i:i+n])] = tags[i]

  return mapping












# loop row by row
for idx, row in df.iterrows():
  ngrams = 5
  # loop for number of n-grams you want (currently 5)
  fullMapping = {}
  for i in range(1,ngrams):
    mapping = getNgrams(df['Article'][idx].split(' '), df['Entity'][idx].split(' '), i)
    fullMapping = {**fullMapping, **mapping}
  
  # remove blacklisted entries from dictionary
  for word in blacklist:
    try:
      del fullMapping[word]
    except KeyError:
      pass

  # apply the mapping to samples
  for column in df.iloc[:, 5:]:
    # get the sample and fill entity array with no-tag ('O')
    sample = df[column][idx].split(' ')
    entities = ['O'] * len(sample)
    # iterate through each ngram length
    for i in range(1, ngrams):
      # sliding window loop for each ngram length of the full sample
      for j in range(len(sample) - i + 1):
        sequence = sample[j: j + i]
        # attempt to find an entity tag for the window
        try:
          entity = fullMapping[' '.join(sequence)]
        # if no tag is found, do nothing
        except KeyError:
          pass
        # if a tag is found
        #replace the list of entities at the indicies of the sliding window
        else:
          for k in range(i):
            entities[j+k] = entity
    # in the copied dataframe, replace the sample with the entity tags
    df_taggedsamples[column][idx] = ' '.join(entities)

df_taggedsamples























