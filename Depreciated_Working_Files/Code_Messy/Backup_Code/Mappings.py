# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 08:29:19 2022

@author: Doug
"""

import pandas as pd
import nltk
nltk.download("punkt")
import csv
import random
import copy
from rouge_score import rouge_scorer
import numpy as np


def article_map():
    # load dataframe
    df = pd.read_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Article.csv", encoding="utf-8", index_col=0)
    #df_taggedsamples = df.copy(deep=True)
    # 1-gram words to be excluded (i.e. 'and' on its own should never be assigned a tag)
    blacklist = ['can','of','A', 'a', 'for', 'and', 'nor', 'but', 'or', 'yet', 'so', 'both', 'and', 'whether', 'or', 'either', 'neither', 'just', 'the', 'The', 'as', 'if', 'then', 'rather', 'than', 'such', 'that']
    
    
    
    # format samples
    for idx, row in df.iterrows():
        for column in df.iloc[:, 5:]:
            empty_sent_list=[]
            for t in nltk.sent_tokenize(df[column][idx]):
                words = nltk.word_tokenize(t)
                empty_sent_list.append(" ".join(words))
                
            df[column][idx] = " ".join(empty_sent_list)
        
    df_taggedsamples = df.copy(deep=True)
        
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
          if df[column][idx]=='':
              continue
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
    
    df.to_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Finished/Original_Article.csv", encoding="utf-8")
    df_taggedsamples.to_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Finished/Mapped_Article.csv", encoding="utf-8")

    
    return
article_map()

def paraphrase_map():
    # load dataframe
    df = pd.read_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Para_Article.csv", encoding="utf-8", index_col=0)
    df_taggedsamples = df.copy(deep=True)
    # 1-gram words to be excluded (i.e. 'and' on its own should never be assigned a tag)
    blacklist = [',','can','of','A', 'a', 'for', 'and', 'nor', 'but', 'or', 'yet', 'so', 'both', 'and', 'whether', 'or', 'either', 'neither', 'just', 'the', 'The', 'as', 'if', 'then', 'rather', 'than', 'such', 'that']
    
    
    
    # format samples
    for idx, row in df.iterrows():
        for column in df.iloc[:, 5:]:
            empty_sent_list=[]
            for t in nltk.sent_tokenize(df[column][idx]):
                words = nltk.word_tokenize(t)
                empty_sent_list.append(" ".join(words))
                
            df[column][idx] = " ".join(empty_sent_list)
        
    df_taggedsamples = df.copy(deep=True)
        
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
          if df[column][idx]=='':
              continue
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
    
    df.to_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Finished/Original_Paraphrase.csv", encoding="utf-8")
    df_taggedsamples.to_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Finished/Mapped_Paraphrase.csv", encoding="utf-8")

    
    return
paraphrase_map()





def tagged_one_map():
    df = pd.read_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Tagged_One.csv", encoding="utf-8", index_col=0)
    
    headers = []
    for i in range(1,51):
        headers.append("Sample"+str(i))
    
    #SETUP MAP
    mapping={}
    for index,row in df.iterrows():
        list_of_words = df["Article"][index].split(" ")
        list_of_tags = df["Entity"][index].split(" ")
        
        previous = ''
        segment_list = []
        tag_list = []
        temp1 = []
        temp2 = []
        article_map = {"ORG":[],"PER":[],"LOC":[],"MISC":[]}
            
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
        
        for index4, thingy in enumerate(segment_list):
            new_tag = tag_list[index4]
            if new_tag != "O":
                #segment_list[index4] = new_tag[2:]
                article_map[new_tag[2:]].append(thingy)
    
        mapping[index] = article_map
    
        
    #COPY MAP
    #mapping_reset = copy.deepcopy(mapping)
    #APPLY MAP
    # format samples
    for idx, row in df.iterrows():
        for column in df.iloc[:, 5:]:
            empty_sent_list=[]
            if df[column][idx]==' ':
                continue
            
            for t in nltk.sent_tokenize(df[column][idx]):
                words = nltk.word_tokenize(t)
                empty_sent_list.append(" ".join(words))
                
            df[column][idx] = " ".join(empty_sent_list)
    
    
    
    df_taggedsamples = df.copy(deep=True)    
    
    for index,row in df.iterrows():
        for i,entry in enumerate(headers):
            if df[entry][index]==' ':
                continue
            #mapping = copy.deepcopy(mapping_reset)
            list_of_words = df[entry][index].split(" ")
            list_of_tags = df_taggedsamples[entry][index].split(" ")
            
            if len(list_of_words)!=len(list_of_tags):
                print("Initial Error")
            
            #Go Through Each Word
            for j,word in enumerate(list_of_words):
                if word in ["ORG","PER","LOC","MISC"]:
                    replacement = random.choice(mapping[index][word])
                    #mapping[index][word].remove(replacement)
                    len_replace = len(replacement.split(" "))
                    list_of_words[j] = replacement
                    
                    #print("Initial Tag: "+word)
                    #print("Replacement: "+replacement)
                    
                    if len_replace == 1:
                        list_of_tags[j] = "I-"+word
                        #print("Tag Replace: "+list_of_tags[j])
                    else:
                        list_of_tags[j] = str(str("I-"+word+" ")*len_replace)[:-1]
                        #print("Tag Replace: "+list_of_tags[j])
                else:
                    list_of_tags[j] = "O"
                    
                if len(list_of_words)!=len(list_of_tags):
                    print("New ERROR")
                
                #print(list_of_words)
                #print(list_of_tags)
            
            df[entry][index] = " ".join(list_of_words)
            df_taggedsamples[entry][index] = " ".join(list_of_tags)
            
    
            if len(" ".join(list_of_words).split(" ")) != len(" ".join(list_of_tags).split(" ")):
                print("SECOND ERROR")
    
    df.to_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Finished/Original_Tagged_One.csv", encoding="utf-8")
    df_taggedsamples.to_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Finished/Mapped_Tagged_One.csv", encoding="utf-8")
    
    return
tagged_one_map()
    
    













def tagged_uni_map():
    df = pd.read_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Tagged_Uni.csv", encoding="utf-8", index_col=0)
    
    headers = []
    for i in range(1,51):
        headers.append("Sample"+str(i))
    
    #SETUP MAP
    mapping={}
    for index,row in df.iterrows():
        Entity_List = ["ORG","LOC","PER","MIS"]
        Entity_List_New = ["ORG","LOC","PER","MIS"]
        list_of_words = df["Article"][index].split(" ")
        list_of_tags = df["Entity"][index].split(" ")
        
        previous = ''
        segment_list = []
        tag_list = []
        temp1 = []
        temp2 = []
        article_map = {}
            
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
        
        #Part That Matters
        for index4, thingy in enumerate(segment_list):
            new_tag = tag_list[index4]
            new_tag = new_tag[2:5]
            
            if new_tag in Entity_List:
                tag_index = Entity_List.index(new_tag)
                original_tag = Entity_List_New[tag_index]
                
                if original_tag[-1].isdigit():
                    new_tag = original_tag[:-1]+str(int(original_tag[-1])+1)
                    article_map[new_tag] = thingy
                else:
                    new_tag = original_tag+"1"
                    article_map[new_tag] = thingy
                    
                Entity_List_New[tag_index] = new_tag
            
                
                
                
        mapping[index] = article_map
    
        
    #COPY MAP
    #mapping_reset = copy.deepcopy(mapping)
    #APPLY MAP
    # format samples
    # format samples
    for idx, row in df.iterrows():
        for column in df.iloc[:, 5:]:
            empty_sent_list=[]
            if df[column][idx]==' ':
                continue
            for t in nltk.sent_tokenize(df[column][idx]):
                words = nltk.word_tokenize(t)
                empty_sent_list.append(" ".join(words))
                
            df[column][idx] = " ".join(empty_sent_list)
    
    
    df_taggedsamples = df.copy(deep=True)    
    #df=df.head(1)
    #headers=["Sample1"]
    
    
    for index,row in df.iterrows():
        for i,entry in enumerate(headers):
            if df[entry][index]==" ":
                continue
            list_of_words = df[entry][index].split(" ")
            list_of_tags = df_taggedsamples[entry][index].split(" ")
            
            if len(list_of_words)!=len(list_of_tags):
                print("Initial Error")
            
            #Go Through Each Word
            for j,word in enumerate(list_of_words):
                if word[:3] in ["ORG","PER","LOC","MIS"]:
                    #CATCHING NICHE CASES WHERE GEN CUTS NUMBER OFF
                    if word not in mapping[index]:
                        word = word[:3]+"1"
                        
                    replacement = mapping[index][word]
                    len_replace = len(replacement.split(" "))
                    list_of_words[j] = replacement
                    
                    #print("Initial Tag: "+word)
                    #print("Replacement: "+replacement)
                    
                    if len_replace == 1:
                        list_of_tags[j] = "I-"+word[:3]
                        #print("Tag Replace: "+list_of_tags[j])
                    else:
                        list_of_tags[j] = str(str("I-"+word[:3]+" ")*len_replace)[:-1]
                        #print("Tag Replace: "+list_of_tags[j])
                else:
                    list_of_tags[j] = "O"
                    
                if len(list_of_words)!=len(list_of_tags):
                    print("New ERROR")
                
                #print(list_of_words)
                #print(list_of_tags)
            
            df[entry][index] = " ".join(list_of_words)
            df_taggedsamples[entry][index] = " ".join(list_of_tags)
            
    
            if len(" ".join(list_of_words).split(" ")) != len(" ".join(list_of_tags).split(" ")):
                print("SECOND ERROR")
    
    df.to_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Finished/Original_Tagged_Uni.csv", encoding="utf-8")
    df_taggedsamples.to_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Finished/Mapped_Tagged_Uni.csv", encoding="utf-8")
    
    return
tagged_uni_map()










### CONFIRM THAT ALL TOKENS HAVE A MATCHING TAG
uh_oh=0
headers = []
for i in range(1,51):
    headers.append("Sample"+str(i))

for value in ["Article","Tagged_One","Tagged_Uni"]:
    df_og = pd.read_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Finished/Original_"+value+".csv", encoding="utf-8", index_col=0)
    df_mp = pd.read_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Finished/Mapped_"+value+".csv", encoding="utf-8", index_col=0)
    
    for index,row in df_og.iterrows():
        for header in headers:
            if df_og[header][index]==" " or pd.isna(df_og[header][index]):
                continue
            else:
                og_len = len(df_og[header][index].split(" "))
                mp_len = len(df_mp[header][index].split(" "))

            if og_len!=mp_len:
                print("VARIANT: "+value+", ARTICLE: "+str(index)+", SAMPLE: "+header)
                uh_oh = 1
    
    if uh_oh!=1:
        print("No Issues!")






#############################
###### SCORING SECTION ######
#############################
def score_dfs():
    variants = ["Article","Tagged_One","Tagged_Uni"]
    
    for variant in variants:
        print(variant)
        df=pd.read_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Finished/Original_"+variant+".csv", encoding="utf-8", index_col=0)
        
        #Setup
        df_scores = df.copy(deep=True)
        f1_threshold = 0.2
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        #Set Sample Headers to Iterate
        headers = []
        for i in range(1,51):
            headers.append("Sample"+str(i))
        
        #Iterate and Apply Scores
        for index,row in df.iterrows():
            original = df[variant][index]
            
            for header_title in headers:
                if pd.isna(df[header_title][index]) or df[header_title][index]==" ":
                    df[header_title][index]="NA"
                    df_scores[header_title][index]="NA"
                    continue
                
                new = df[header_title][index]
                scores = scorer.score(original, new)
                f1_score = scores["rouge1"][2]
                
                if f1_score > f1_threshold:    
                    df_scores[header_title][index] = f1_score
                elif f1_score < f1_threshold and new != " ":
                    df_scores[header_title][index] = "Low Score"
                else:
                    df_scores[header_title][index] = "NA"
        
        #df.replace('N/A',pd.NA)
        df_scores.to_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Finished/Scored_"+variant+".csv", encoding="utf-8")
        
    return
score_dfs()
















































