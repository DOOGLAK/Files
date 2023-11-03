# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:08:28 2022

@author: Doug
"""

import random
import numpy as np
import copy
import pandas as pd
import csv
import nltk
from nltk.corpus import wordnet

##############
#### DF ######
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
    
    #Check it worked
    for index in range(len(article_list)):
      pass
      #print(article_list[index],"\n",token_list[index])
    
    #Back to DF
    temp_dict = {"Article":article_list,"Entity":token_list}
    df = pd.DataFrame(temp_dict)

    return df

df = gold_dataframe()

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



#####################
##### RULE FXNS #####
#####################
def LWTR_Method(augments):
    DA_LWTR_additions = []
    
    for k in range(0,augments):
        DA_LWTR_word_sentences = copy.deepcopy(word_sentences)
        for i,sentence in enumerate(word_sentences):
            for j,word in enumerate(sentence):
                tag = tag_sentences[i][j]
                
                if np.random.binomial(1, rate, size=None):
                    substitute = random.choice(tag_groups[tag])
                    DA_LWTR_word_sentences[i][j] = substitute
                
            DA_LWTR_additions.append(DA_LWTR_word_sentences[i])
        
    #print(word_sentences[0])
    #print(DA_LWTR_word_sentences[0])
    #print(DA_LWTR_additions[50])
    #print(tag_sentences[0])
    #print(len(DA_LWTR_word_sentences))
    #print(len(DA_LWTR_additions))
    
    return word_sentences+DA_LWTR_additions, tag_sentences+tag_sentences*augments

def SR_Method(augments):
    DA_SR_additions = []
    
    SR_Issue_List = ["in","In","It","it","does","Does","IAEA","have","Have","be","Be","less","Less","He","he","Pesos","Inc","inc","acts","Acts","an","An","units"]
    
    for k in range(0,augments):
        DA_SR_word_sentences = copy.deepcopy(word_sentences)
        
        for i,sentence in enumerate(word_sentences):
            for j,word in enumerate(sentence):
                tag = tag_sentences[i][j]
                
                if np.random.binomial(1, rate, size=None):
                    try:
                        substitute = wordnet.synsets(word)[0].lemmas()[0].name()
                        if word not in SR_Issue_List:
                            DA_SR_word_sentences[i][j] = substitute
                    except:
                        DA_SR_word_sentences[i][j] = word
                        
            DA_SR_additions.append(DA_SR_word_sentences[i])
        
        #print(word_sentences[0])
        #print(DA_SR_word_sentences[0])
        #print(tag_sentences[0])  
        
    return word_sentences+DA_SR_additions, tag_sentences+tag_sentences*augments

def SIS_Method(augments):
    DA_SIS_additions = []
    
    for k in range(0,augments):
        DA_SIS_word_sentences = copy.deepcopy(word_sentences)
        
        for i,sentence in enumerate(word_sentences):
            
            previous = ''
            segment_list = []
            temp1 = []
        
            
            for j,word in enumerate(sentence):
                tag = tag_sentences[i][j]
                
                if tag == previous or previous == '':
                    temp1.append(word)
                elif tag != previous:
                    segment_list.append(temp1.copy())
                    
                    temp1.clear()
                    
                    temp1.append(word)
                
                previous = tag
            
            segment_list.append(temp1)
            
            #print(segment_list)
            
            for j,entry in enumerate(segment_list):
                if np.random.binomial(1, rate, size=None):
                    random.shuffle(entry)
                
            DA_SIS_word_sentences[i] = np.hstack(segment_list).tolist()
            DA_SIS_additions.append(DA_SIS_word_sentences[i])
            
        #print(word_sentences[0])
        #print(DA_SIS_word_sentences[0])
        #print(tag_sentences[0]) 

    return word_sentences+DA_SIS_additions, tag_sentences+tag_sentences*augments



############
###SETUP####
############
rates=[0.1,0.3,0.5,0.7]
for rate in rates:
    print(rate)
    for key in id_dict.keys():
        print(key)
        df=gold_dataframe()
        
        breakdown=key.split("v")
        size_choice=breakdown[0]
        version=breakdown[1]
        
        df = df.iloc[id_dict[key]]
        
        word_sentences = []
        tag_sentences = []
        for index, row in df.iterrows():
            
            words = df["Article"][index].split(" ")
            tags = df["Entity"][index].split(" ")
            
            word_segment = []
            tag_segment = []
            for i,word in enumerate(words):
                if word != ".":
                    word_segment.append(word)
                    tag_segment.append(tags[i])
                else:
                    word_segment.append(word)
                    word_sentences.append(word_segment)
                    word_segment = []
                    
                    tag_segment.append(tags[i])
                    tag_sentences.append(tag_segment)
                    tag_segment = []
                    
        
        
        word_sentences_flat = np.hstack(word_sentences)
        tag_sentences_flat = np.hstack(tag_sentences)
        
        ### MAP WORDS TO TAGS
        token_map = pd.DataFrame({"Words":word_sentences_flat, "Tags":tag_sentences_flat})
        tag_list = token_map["Tags"].unique().tolist()
        tag_groups = token_map.groupby("Tags")["Words"].apply(list)
    
        #######################
        #### RUN FUNCTIONS ####
        #######################
        LWTR_Results, LWTR_Tags = LWTR_Method(3)
        SR_Results, SR_Tags = SR_Method(3)
        SIS_Results, SIS_Tags = SIS_Method(3)
    
        ######################
        #### TO TEXT FILE ####
        ######################
        LWTR_Path = "H:\\My Files\\School\\Grad School WLU\\MRP\\Research\\Files\\Data\\Textfiles\\Rule_"+str(rate)+"\\LWTR\\"+str(size_choice)+"\\v"+str(version)+"_Augmented.txt"
        SR_Path = "H:\\My Files\\School\\Grad School WLU\\MRP\\Research\\Files\\Data\\Textfiles\\Rule_"+str(rate)+"\\SR\\"+str(size_choice)+"\\v"+str(version)+"_Augmented.txt"
        SIS_Path = "H:\\My Files\\School\\Grad School WLU\\MRP\\Research\\Files\\Data\\Textfiles\\Rule_"+str(rate)+"\\SIS\\"+str(size_choice)+"\\v"+str(version)+"_Augmented.txt"
        
        with open(LWTR_Path, 'w', encoding="utf-8") as LWTR_File:
            for h,sentence in enumerate(LWTR_Results): #sentence from list of sentences
                for g,word in enumerate(sentence): #word from sentence
                    LWTR_File.write(LWTR_Results[h][g]+" "+LWTR_Tags[h][g]+"\n")
                LWTR_File.write("\n")
        
        #SR VARIANT
        with open(SR_Path, 'w', encoding="utf-8") as SR_File:
            for h,sentence in enumerate(SR_Results): #sentence from list of sentences
                for g,word in enumerate(sentence): #word from sentence
                    SR_File.write(SR_Results[h][g]+" "+SR_Tags[h][g]+"\n")
                SR_File.write("\n")
        
        #SIS VARIANT
        with open(SIS_Path, 'w', encoding="utf-8") as SIS_File:
            for h,sentence in enumerate(SIS_Results): #sentence from list of sentences
                for g,word in enumerate(sentence): #word from sentence
                    SIS_File.write(SIS_Results[h][g]+" "+SIS_Tags[h][g]+"\n")
                SIS_File.write("\n")
            












#################
###PARAPHRASE####
#################
for key in id_dict.keys():
    df_og = pd.read_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Finished/Original_Paraphrase.csv", encoding="utf-8", index_col=0)
    df_mp = pd.read_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/dataframes/Finished/Mapped_Paraphrase.csv", encoding="utf-8", index_col=0)
    
    info = key.split("v")
    size = info[0]
    version = info[1]
    
    df_og = df_og.iloc[id_dict[key]]
    df_mp = df_mp.iloc[id_dict[key]]
    
    sentence_list = []
    tag_list = []
    
    #Get Sentences and Tags in Simple Form
    for index,row in df_og.iterrows():
        for header in ["Article","Sample1","Sample2","Sample3"]:
            if header=="Article":
                tokens = df_og[header][index].split(" ")
                tags = df_mp["Entity"][index].split(" ")
            else:
                tokens = df_og[header][index].split(" ")
                tags = df_mp[header][index].split(" ")
            
            current_sentence=[]
            current_tag=[]
            
            for i,token in enumerate(tokens):
                current_sentence.append(token)
                current_tag.append(tags[i])
                
                if token=="." or i==len(tokens)-1:
                    sentence_list.append(" ".join(current_sentence))
                    tag_list.append(" ".join(current_tag))
                    current_sentence,current_tag = [],[]
    
    #print(sentence_list[0])    
    
    #Write to Txt File
    txt_file_path = "H:\\My Files\\School\\Grad School WLU\\MRP\\Research\\Files\\Data\\Textfiles\\Paraphrased\\"+str(size)+"\\v"+str(version)+"_Augmented.txt"
    with open(txt_file_path,'w', encoding="utf-8") as aug_file:
        for h,sentence in enumerate(sentence_list): #sentence from list of sentences
            for g,word in enumerate(sentence.split(" ")): #word from sentence
                aug_file.write(word+" "+tag_list[h].split(" ")[g]+"\n")
            aug_file.write("\n")
        
        
