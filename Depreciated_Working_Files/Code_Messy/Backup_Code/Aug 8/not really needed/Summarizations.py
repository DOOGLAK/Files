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
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration, MT5Tokenizer, MT5ForConditionalGeneration

#OS
import os.path
from os import path as os_path

#ITERTOOLS
import itertools

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

torch_device = "cuda" #If throwing CUDA error, restart Python

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

def sentence_iter(num_samples, max_length, num_beams, temperature, df_name, tokenizer, model):
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


















def abs_summaries(
    input_text, num_return_sequences, num_beams, min_length, temperature=1.5
):
    #PEGASUS XSUM
    batch1 = tokenizer1(input_text, truncation=True, padding="longest", return_tensors="pt").to(torch_device)
    translated1 = model1.generate(**batch1, temperature=temperature, min_length=min_length,
                                  num_beams=num_beams, num_return_sequences=num_return_sequences,top_p=0.9)
    Pegasus = tokenizer1.batch_decode(translated1, skip_special_tokens=True)
    
    #T5 

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



#df = gold_dataframe()
#back_translate(1, 32, 5, "temp")



##########################################
################# SETUP ##################
##########################################

#Set Device
torch_device = "cuda" #If throwing CUDA error, restart Python

#Set Models
tokenizer1 = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model1 = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to(torch_device)

#tokenizer2 = T5Tokenizer.from_pretrained("google/mt5-base")
#model2 = MT5ForConditionalGeneration.from_pretrained("google/mt5-base").to(torch_device)




##########################################
########### PARAPHRASE METHOD ############
##########################################
df = gold_dataframe()
model_name = "tuner007/pegasus_paraphrase"
path = "H:/My Files/School/Grad School WLU/MRP/Research/Files/Models/Pegasus_Para"

if not os_path.exists(path):
    print("Path did not exist -- generating local model save.")
    tokenizer, model = new_model(model_name=model_name, path=path, torch_device=torch_device)
else:
    print("Model Found -- using existing model.")
    tokenizer, model = load_model(path=path, torch_device=torch_device)

sentence_iter(num_samples=8, max_length=60, num_beams=8, temperature=5,
              df_name="df_beams64_temp_5", tokenizer=tokenizer, model=model)



##########################################
########### WEIGHTED METHOD ##############
##########################################
df = gold_dataframe()
weighted_entity_order(high_first=False)
article_iter(False, 10, 32, 3, "pegasus_32Beam_5Sample_F_weighted")



##########################################
########### SHUFFLED METHOD ##############
##########################################
df = gold_dataframe()
article_iter(True, 16, 64, 3, "pegasus_128Beam_32Sample_full_shuffle8")




####TESTING
df = gold_dataframe()
for index, row in df.iterrows():   
    
    words = df["Article"][index].split(" ")
    tags = df["Entity"][index].split(" ")
    
    for i, entry in enumerate(tags):
        if entry != "O" and not words[i][0].isupper():
            print(words[i])




df = gold_dataframe()
for index, row in df.iterrows():
    print(index)
    
    words = df['Article'][index].split(" ")
    tags = df['Entity'][index].split(" ")
    
    previous = ''
    segment_list = []
    tag_list = []
    temp1 = []
    temp2 = []
    
    for index2, tag in enumerate(tags):
        if tag == previous or previous == '':
            temp1.append(words[index2])
            temp2.append(tag)
        elif tag != previous:
            segment_list.append(temp1.copy())
            tag_list.append(temp2.copy())
            
            temp1.clear()
            temp2.clear()
            
            temp1.append(words[index2])
            temp2.append(tag)
        
        previous = tag
    
    segment_list.append(temp1)
    tag_list.append(temp2)
    


    for index3, group in enumerate(segment_list):
        segment_list[index3] = " ".join(group)
        tag_list[index3] = tag_list[index3][0]
    
    print(segment_list)
    print(tag_list)
    
    for index3,thingy in enumerate(segment_list):
        new_tag = tag_list[index3]
        if new_tag != "O":
            segment_list[index3] = new_tag[2:]
    
    
    print(" ".join(segment_list))











##TESTING HOW TO ALTER ABS SUMMARY OUTPUT TO FOLLOW STANDARD TOKEN MODEL
testing_standard = "One of the first to arrive at the scene of the accident was Edward Farr, engineer of the Atlantic City Railroad, who had driven the excursion train from Atlantic City over the West Jersey Railroad to the crossing at the foot of Mississippi Avenue, where it was hit by the Reading Railroad express train."

import nltk
nltk.download('punkt')

tokens = nltk.sent_tokenize(testing_standard)
for t in tokens:
    print (t, "\n")
    words = nltk.word_tokenize(t)
    print(words)


























#####TESTING FOR REPLACING NON "O" TAGS WITH "I-" TAGS
df = gold_dataframe()
test1 = df["Article"][0]

word_sentences = []
tag_sentences = []
for index, row in df.iterrows():
    
    words = df["Article"][index].split(" ")
    tags = df["Entity"][index].split(" ")
    
    #print(words)
    #print(tags)
    
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

for i,sentence in enumerate(tag_sentences):
    for j,tag in enumerate(sentence):
        if tag!="O":
            word_sentences[i][j] = tag[2:]
            
print(word_sentences[0])
print(tag_sentences[0])

testing = " ".join(word_sentences[0])
testing = 'MISC is the tenth album from MISC Punk Techno band ORG.'

paraphrase(
    input_text=testing,
    tokenizer=tokenizer,
    model=model,
    num_return_sequences=10,
    num_beams=32,
    max_length=60,
    temperature=5)


#ABS SUMMARY TESTING
prop_length = df["Article"][0].count(" ")#/6 #arbitrarily Picked

words_list = df["Article"][0].split(" ")
tags_list = df["Entity"][0].split(" ")

for i, word in enumerate(words_list):
    if tags_list[i] != "O":
        words_list[i] = tags_list[i]
        
words_list = " ".join(words_list)
print(words_list)

results = abs_summaries(
    input_text = words_list,
    num_return_sequences=4,
    num_beams=16,
    min_length=int(prop_length),
    temperature=4
    )


# testing = "My name is John Smith"
# testing = "translate English to French: "+testing

# testing = "Je m'appelle John Smith"
# testing = "translate French to English: "+testing

# testing2 = "translate English to German: My name is John."



# batch2 = tokenizer2(testing, truncation=True, padding="longest", return_tensors="pt").to(torch_device)
# #batch2 = tokenizer2(testing, return_tensors="pt").to(torch_device)
# translated2 = model2.generate(**batch2,temperature=4,
#                               num_beams=32, num_return_sequences=5)
# t5_out = tokenizer2.batch_decode(translated2, skip_special_tokens=True)
# print(t5_out)






# def release_cuda_mem():

#     from GPUtil import showUtilization as gpu_usage
    
#     def free_gpu_cache():
#         print("Initial GPU Usage")
#         gpu_usage()                             
    
#         torch.cuda.empty_cache()
    
#         cuda.select_device(0)
#         cuda.close()
#         cuda.select_device(0)
    
#         print("GPU Usage after emptying the cache")
#         gpu_usage()
    
#     free_gpu_cache()
    
#     return


# release_cuda_mem()














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














# ##########################
# ###### NER MODEL #########
# ##########################

# #Guide: https://huggingface.co/course/chapter7/2
# #Fix: https://github.com/huggingface/datasets/issues/4099
# #Fix 2: https://huggingface.co/datasets/nielsr/XFUN/commit/73ba5e026621e05fb756ae0f267eb49971f70ebd

# from datasets import load_dataset
# #raw_datasets = load_dataset("conll2003")
# raw_datasets = load_dataset("H:\My Files\School\Grad School WLU\MRP\Research\Files\Data\wikigold_splits.py")
# raw_datasets

# from transformers import AutoTokenizer

# model_checkpoint = "bert-base-cased"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# tokenizer.is_fast

# ner_feature = raw_datasets["train"].features["ner_tags"]
# ner_feature

# label_names = ner_feature.feature.names
# label_names

# #PREPROCESS DATA

# inputs = tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True)
# inputs.tokens()
# inputs.word_ids()


# def align_labels_with_tokens(labels, word_ids):
#     new_labels = []
#     current_word = None
#     for word_id in word_ids:
#         if word_id != current_word:
#             # Start of a new word!
#             current_word = word_id
#             label = -100 if word_id is None else labels[word_id]
#             new_labels.append(label)
#         elif word_id is None:
#             # Special token
#             new_labels.append(-100)
#         else:
#             # Same word as previous token
#             label = labels[word_id]
#             # If the label is B-XXX we change it to I-XXX
#             if label % 2 == 1:
#                 label += 1
#             new_labels.append(label)

#     return new_labels


# labels = raw_datasets["train"][0]["ner_tags"]
# word_ids = inputs.word_ids()
# print(labels)
# print(align_labels_with_tokens(labels, word_ids))


# def tokenize_and_align_labels(examples):
#     tokenized_inputs = tokenizer(
#         examples["tokens"], truncation=True, is_split_into_words=True
#     )
#     all_labels = examples["ner_tags"]
#     new_labels = []
#     for i, labels in enumerate(all_labels):
#         word_ids = tokenized_inputs.word_ids(i)
#         new_labels.append(align_labels_with_tokens(labels, word_ids))

#     tokenized_inputs["labels"] = new_labels
#     return tokenized_inputs

# #Takes a bit...
# tokenized_datasets = raw_datasets.map(
#     tokenize_and_align_labels,
#     batched=True,
#     remove_columns=raw_datasets["train"].column_names,
# )



# from transformers import DataCollatorForTokenClassification
# data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])
# batch["labels"]

# for i in range(2):
#     print(tokenized_datasets["train"][i]["labels"])


# #EVAL -- required pip install seqeval
# import evaluate
# metric = evaluate.load("seqeval")

# labels = raw_datasets["train"][0]["ner_tags"]
# labels = [label_names[i] for i in labels]
# labels

# predictions = labels.copy()
# predictions[2] = "O"
# metric.compute(predictions=[predictions], references=[labels])





# #NOT SURE
# import numpy as np


# def compute_metrics(eval_preds):
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)

#     # Remove ignored index (special tokens) and convert to labels
#     true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
#     true_predictions = [
#         [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#     all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
#     return {
#         "precision": all_metrics["overall_precision"],
#         "recall": all_metrics["overall_recall"],
#         "f1": all_metrics["overall_f1"],
#         "accuracy": all_metrics["overall_accuracy"],
#     }

# #DEFINING THE MODEL
# id2label = {str(i): label for i, label in enumerate(label_names)}
# label2id = {v: k for k, v in id2label.items()}


# from transformers import AutoModelForTokenClassification

# model = AutoModelForTokenClassification.from_pretrained(
#     model_checkpoint,
#     id2label=id2label,
#     label2id=label2id,
# )

# #Check # of Labels is Correct:
# model.config.num_labels



# from transformers import TrainingArguments

# args = TrainingArguments(
#     "bert-finetuned-ner",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=2e-5,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     push_to_hub=False,
#     hub_token = "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"
# )



# #TUNE
# from transformers import Trainer

# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     tokenizer=tokenizer,
# )

# trainer.train()
# trainer.evaluate()





























# ##CUSTOM TRAINING LOOP
# from torch.utils.data import DataLoader

# train_dataloader = DataLoader(
#     tokenized_datasets["train"],
#     shuffle=True,
#     collate_fn=data_collator,
#     batch_size=8,
# )
# eval_dataloader = DataLoader(
#     tokenized_datasets["validation"], collate_fn=data_collator, batch_size=8
# )

# model = AutoModelForTokenClassification.from_pretrained(
#     model_checkpoint,
#     id2label=id2label,
#     label2id=label2id,
# )

# from torch.optim import AdamW
# optimizer = AdamW(model.parameters(), lr=2e-5)

# from accelerate import Accelerator
# accelerator = Accelerator()
# model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
#     model, optimizer, train_dataloader, eval_dataloader
# )


# from transformers import get_scheduler

# num_train_epochs = 3
# num_update_steps_per_epoch = len(train_dataloader)
# num_training_steps = num_train_epochs * num_update_steps_per_epoch

# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_warmup_steps=0,
#     num_training_steps=num_training_steps,
# )



# ##NEED TO BE LOGGED IN
# from huggingface_hub import Repository, get_full_repo_name

# model_name = "bert-finetuned-ner-accelerate"
# repo_name = get_full_repo_name(model_name)
# repo_name

# output_dir = "bert-finetuned-ner-accelerate"
# repo = Repository(output_dir, clone_from=repo_name)


# trainer.train()







# ####################
# #TRANSLATION METHOD#
# ####################
# def load_translation_model():
#     tokenizer2 = T5Tokenizer.from_pretrained("google/mt5-base")
#     model2 = MT5ForConditionalGeneration.from_pretrained("google/mt5-base").to(torch_device)
    
#     return tokenizer2, model2

# def translate_segment(input_text, num_samples, num_beams, temperature):
#     batch2 = tokenizer2(input_text, truncation=True, padding="longest", return_tensors="pt").to(torch_device)
#     translated2 = model2.generate(**batch2,temperature=temperature,
#                                   num_beams=num_beams, num_return_sequences=num_samples)
#     t5_out = tokenizer2.batch_decode(translated2, skip_special_tokens=True)
    
#     return t5_out

# def back_translate(num_samples, num_beams, temperature, df_name):
#     print("Loading Model!")
#     load_translation_model()
#     print("Model Loaded!")    
    
#     for i in range(num_samples):
#         header = "Sample" + str(i+1)
#         df[header] = " "

#     for index, row in df.iterrows():
#         print(index)
        
#         words = df['Article'][index].split(" ")
#         tags = df['Entity'][index].split(" ")
        
#         previous = ''
#         segment_list = []
#         tag_list = []
#         temp1 = []
#         temp2 = []
        
#         for index2, tag in enumerate(tags):
#             if tag == previous or previous == '':
#                 temp1.append(words[index2])
#                 temp2.append(tag)
#             elif tag != previous:
#                 segment_list.append(temp1.copy())
#                 tag_list.append(temp2.copy())
                
#                 temp1.clear()
#                 temp2.clear()
                
#                 temp1.append(words[index2])
#                 temp2.append(tag)
            
#             previous = tag
        
#         segment_list.append(temp1)
#         tag_list.append(temp2)
        
#         for index3, group in enumerate(segment_list):
#             segment_list[index3] = " ".join(group)
#             tag_list[index3] = tag_list[index3][0]
            
#         print(segment_list)
#         #print(tag_list)
        
        
#         ##############
#         ####ENG2FR####
#         ##############
#         eng2fr = []
#         for index4, entry in enumerate(segment_list):
#             if tag_list[index4] == "O":
#                 eng_frn = "translate English to French: " + entry
#                 #print("Actual: "+entry)
#                 results = translate_segment(eng_frn, num_samples, num_beams, temperature)
                
#                 if results[0] == "" or ".." in results[0]:
#                     eng2fr.append(entry)
#                     tag_list[index4] = "MANUAL"
#                     #print(entry)
#                 else:
#                     #print(results)
#                     eng2fr.append(results[0])
            
#             else:
#                 eng2fr.append(entry)
            
#         print(eng2fr)
        
#         ##############
#         ####FR2ENG####
#         ##############
#         fr2eng = []
#         for index5, entry in enumerate(eng2fr):
#             if tag_list[index4] == "O" and tag_list[index4] != "MANUAL":
#                 frn_eng = "translate French to English: " + entry
#                 print("Actual: "+entry)
#                 results = translate_segment(frn_eng, num_samples, num_beams, temperature)
#                 print(results[0])
                
#         #         if results[0] == "" or ".." in results[0]:
#         #             fr2eng.append(entry)
#         #             #print(entry)
#         #         else:
#         #             #print(results)
#         #             fr2eng.append(results[0])
            
#         #     else:
#         #         fr2eng.append(entry)
            
#         # print(fr2eng)
        
#         for i in range(num_samples):
#             header = "Sample" + str(i+1)
#             df[header][index] = results[i]
    
#     #df.to_csv("H:/My Files/School/Grad School WLU/MRP/Research/Files/Data/"+df_name+".csv")
    
#     return# df

















