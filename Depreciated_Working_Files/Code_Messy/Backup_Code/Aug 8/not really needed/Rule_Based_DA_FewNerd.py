#!/usr/bin/env python
# coding: utf-8

# # Data Augmentation of fewNERD 

# Perform 4 Data Augmentation techniques on txt doc of format: 
# 
# - "\n\n" to seperate sentences
# - "\n" to seperate words
# - "\t" to seperate labels
# - "-" to seperate coarse & fine labels (2-tier label hiearchy)

# ## Import FewNERD (maintain Sentence structure)

# In[1]:


import copy
import pandas as pd


# In[2]:


# convert fewNERD txt doc to list of lists

def fewnerdTxtToList(data):
    sentences = data.split("\n\n")

    sentences2 = []

    for i in sentences:
        tokenized = i.split("\n")
        sentences2.append(tokenized)

    _list = []
    for sentence in sentences2:
        temp_list =[]

        for word in sentence:
            word = word.split("\t")
            word[1] = word[1].split("-") 
            temp_list.append(word)

        _list.append(temp_list)

    fewnerd_data =_list
    return fewnerd_data


# In[3]:


rate = 0.5


# In[4]:


# import fewNERD txt doc as pandas table of words

with open('train.txt', encoding='UTF-8') as f:
    contents = f.read()

fewnerd_data = fewnerdTxtToList(contents)


# In[5]:


# convert fewNERD txt doc to pandas df

fewnerd_pd = pd.read_csv("train.txt", sep="\t",header=None, names=['word','label'])


# In[6]:


# convert df to list of groups

def DataFrameToGroups(data):

    df3 = data["label"].str.split("-",expand=True)
    df3.loc[df3[0] == "O"] = "O"

    labeled_df = pd.concat([data,df3],axis=1)
    labeled_df = labeled_df.drop(labeled_df.columns[[1]],axis=1)
    labeled_df.columns = ['word','l_coarse','l_fine']

    groups = [labeled_df for _, labeled_df in labeled_df.groupby("l_coarse")]

    return groups


# In[7]:


groups = DataFrameToGroups(fewnerd_pd)


# # 1) Label-wise Token Replacement (LwTR)

# In[8]:


import random
import numpy as np


# In[9]:


fewnerd_LwTR = copy.deepcopy(fewnerd_data)


# In[10]:


###TOO LONG

def LabelWiseTokenReplacement(rate, data, groups):
    
    group_list = []

    for group in groups:
        group_list.append(group["l_coarse"].unique()[0])

    group_definitions = { group_list[i]: i for i in range(len(group_list))}
    
    
    for sentence in data:
        for word in sentence:
            if np.random.binomial(1, rate, size=None) == 1:
                
                group_tag = groups[group_definitions[word[1][0]]]
                word[0]= group_tag.sample()["word"].item()
                
    return data


# In[11]:


# LwTR_final_test = LabelWiseTokenReplacement(rate,fewnerd_LwTR[:3], groups)
# len(LwTR_final_test)
# group_definitions


# ## 2) Synonym Replacement

# In[ ]:


import nltk
from nltk.corpus import wordnet


# In[ ]:


fewnerd_SR = copy.deepcopy(fewnerd_data)


# In[ ]:


def SynonymReplacement(rate, data):
    for sentence in data:
        for word in sentence:
            if np.random.binomial(1, rate, size=None) == 1:
                try:
                    syn = wordnet.synsets(word[0])[0].lemmas()[0].name()
                    if word[0] not in ["in","In","It","it","does","Does","IAEA","have","Have","be","Be","less","Less","He","he","Pesos","Inc","inc","acts","Acts","an","An","units"]:
                        word[0] = syn
                except:
                    word[0] = word[0]
    return data


# In[ ]:


# fewnerd_SR = SynonymReplacement(rate, fewnerd_SR[:20])


# In[ ]:


for i in 


# In[ ]:





# ## 3) Mention Replacement

# In[ ]:


fewnerd_MR = copy.deepcopy(fewnerd_data)


# In[ ]:


fewnerd_MR = fewnerd_MR[:10]


# In[ ]:


# convert to BIO-labels

def ApplyBioLabels(data):
    for i in range(0,len(data)): #sentence
        for j in range(0,len(data[i])): #word
            if data[i][j][1] == ["O"]:
                pass

            else:
                data[i][j].append("B")
                if (data[i][j][1] == data[i][j-1][1]):
                    data[i][j][2] = "I"
    return data


# In[ ]:


fewnerd_BIO = ApplyBioLabels(fewnerd_MR)


# In[ ]:


fewnerd_BIO


# In[ ]:


# convert to BIO-labels
list_ = []

def ApplyBioLabels(data):
    for i in range(0,len(data)): #sentence
        for j in range(0,len(data[i])): #word
            if data[i][j][1] == ["O"]:
                pass

            else:
                data[i][j].append("B")
                if (data[i][j][1] == data[i][j-1][1]):
                    data[i][j][2] = "I"
    return data


# In[ ]:


fewnerd_BIO


# In[ ]:


test_ = DataFrameToGroups(fewnerd_BIO)


# ## 4) Shuffle within Segments

# In[ ]:


def SegmentShuffle(rate,data):
    
    for i in range(0,len(data)): #sentence
        segment = [data[i][0][0]]
        
        
        for j in range(1,len(data[i])): #word
            if (data[i][j-1][1] == data[i][j][1]):
                segment.append(data[i][j][0])
                
            else:
                replaced = np.random.binomial(1, rate, size=None)
                if replaced:
                    random.shuffle(segment)
                    
                    for k in range(0,len(segment)):
                        data[i][j-len(segment)+k][0] = segment[k]
    
                segment = [data[i][j][0]]
        
            if len(segment) == len(data[i]):
                replaced = np.random.binomial(1, rate, size=None)
                if replaced:
                    random.shuffle(segment)
                    
                    for k in range(0,len(segment)):
                        data[i][j-len(segment)+k][0] = segment[k]
    
                segment = [data[i][j][0]]
                
    return data


# In[ ]:


fewnerd_SIS = copy.deepcopy(fewnerd_data)


# In[ ]:


# fewnerd_SIS_test = SegmentShuffle(1,fewnerd_SIS[:12])
# fewnerd_SIS_test


# ## Parallelization

# In[ ]:


import concurrent.futures


# In[ ]:


# Run Parallel

SIS_data = []

rates = [rate for _ in range(len(fewnerd_SIS))]

with concurrent.futures.ThreadPoolExecutor() as executor:
    data = executor.map(SegmentShuffle,rates,[fewnerd_SIS])
    for d in data:
        SIS_data.append(d)

SIS_data = SIS_data[0]


# In[ ]:


len(SIS_data)


# In[ ]:


SR_data = []

rates = [rate for _ in range(len(fewnerd_SIS))]

with concurrent.futures.ThreadPoolExecutor() as executor:
    data = executor.map(SynonymReplacement,rates,[SIS_data])
    for d in data:
        SR_data.append(d)

SIS_SR_data = SR_data[0]


# In[ ]:


# len(SR_data)
# print(SR_data[:20])


# In[ ]:


###TOO LONG

# LwTR_data = []

# rates = [rate for _ in range(len(fewnerd_LwTR))]

# with concurrent.futures.ThreadPoolExecutor() as executor:
#     data = executor.map(LabelWiseTokenReplacement,rates,[fewnerd_LwTR],[groups])
#     for d in data:
#         LwTR_data.append(d)

# LwTR_data = LwTR_data[0]


# ## Convert export file to txt

# In[ ]:


def ListsToTxt(data, fileName):
    
    test_txt = open(fileName, "w",encoding='UTF-8')
    
    for sentence in data:
        for word in sentence:
            test_txt.write("%s \t" %word[0])
            category = '-'.join(word[1])
            test_txt.write("%s \n" %category)
        test_txt.write("\n")
    
    test_txt.close()
    
    return


# In[ ]:





# In[ ]:


# fileName = 'output_test.txt'
# data = fewnerd_data


# In[ ]:


# ListsToTxt(LwTR_data, "output_LwTR_0.7")
# ListsToTxt(SR_data, "fewnerd_SR_0")
# ListsToTxt(SIS_data, "fewnerd_SIS_0")


# In[ ]:


len(SR_data)


# In[ ]:


ListsToTxt(SIS_SR_data, "fewnerd_SIS_SR_0.5")


# In[ ]:


len(SIS_SR_data)


# In[ ]:




