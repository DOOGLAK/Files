# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 23:54:28 2022

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

df = pd.read_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Article.csv", encoding="utf-8", index_col=0)
#df_taggedsamples = df.copy(deep=True)
# 1-gram words to be excluded (i.e. 'and' on its own should never be assigned a tag)
blacklist = ['can','of','A', 'a', 'for', 'and', 'nor', 'but', 'or', 'yet', 'so', 'both', 'and', 'whether', 'or', 'either', 'neither', 'just', 'the', 'The', 'as', 'if', 'then', 'rather', 'than', 'such', 'that',"This"]

df = df.iloc[0:1].reset_index(drop=True)

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
    print("NGram: "+str(i))
    mapping = getNgrams(df['Article'][idx].split(' '), df['Entity'][idx].split(' '), i)
    fullMapping = {**fullMapping, **mapping}
  
  # remove blacklisted entries from dictionary
  for word in blacklist:
    try:
      del fullMapping[word]
    except KeyError:
      pass

  # apply the mapping to samples
  for column in df.iloc[:, 5:6]:
      print(df[column][idx])
      if df[column][idx]=='':
          continue
      # get the sample and fill entity array with no-tag ('O')
      sample = df[column][idx].split(' ')
      entities = ['O'] * len(sample)
      # iterate through each ngram length
      for i in range(1, ngrams):
          print("NGram: "+str(i))
          # sliding window loop for each ngram length of the full sample
          for j in range(len(sample) - i + 1):
              sequence = sample[j: j + i]
              #print(sequence)
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
                      print(entities)
      # in the copied dataframe, replace the sample with the entity tags
      df_taggedsamples[column][idx] = ' '.join(entities)
      