# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 08:15:08 2022

@author: Doug
"""

##########################
###### NER MODEL #########
##########################

#Guide: https://huggingface.co/course/chapter7/2
#Fix: https://github.com/huggingface/datasets/issues/4099
#Fix 2: https://huggingface.co/datasets/nielsr/XFUN/commit/73ba5e026621e05fb756ae0f267eb49971f70ebd

#################
##### CACHE #####
#################

#Change Cache
import os
os.environ['TRANSFORMERS_CACHE'] = 'H:/TempHF_Cache/cache/transformers/'
os.environ['HF_HOME'] = 'H:/TempHF_Cache/cache/'
os.environ['XDG_CACHE_HOME'] = 'H:/TempHF_Cache/cache/'

##############
#### CODE ####
##############

from datasets import load_dataset
#raw_datasets = load_dataset("conll2003")
raw_datasets = load_dataset("H:\My Files\School\Grad School WLU\MRP\Research\Files\Data\wikigold_splits.py")
raw_datasets

#raw_datasets["train"][50]["tokens"]

from transformers import AutoTokenizer

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, cache_dir="H:\TempHF_Cache\Base_Tokenizer")

tokenizer.is_fast

ner_feature = raw_datasets["train"].features["ner_tags"]
ner_feature

label_names = ner_feature.feature.names
label_names

#PREPROCESS DATA

inputs = tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True)
inputs.tokens()
inputs.word_ids()


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


labels = raw_datasets["train"][0]["ner_tags"]
word_ids = inputs.word_ids()
print(labels)
print(align_labels_with_tokens(labels, word_ids))


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

#Takes a bit...
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)



from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])
batch["labels"]

for i in range(2):
    print(tokenized_datasets["train"][i]["labels"])


#EVAL -- required pip install seqeval
import evaluate
metric = evaluate.load("seqeval")

labels = raw_datasets["train"][0]["ner_tags"]
labels = [label_names[i] for i in labels]
labels

predictions = labels.copy()
predictions[2] = "O"
metric.compute(predictions=[predictions], references=[labels])





#NOT SURE
import numpy as np

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

#DEFINING THE MODEL
id2label = {str(i): label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}


from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
    cache_dir="H:\TempHF_Cache\Base_Model"
)

#Check # of Labels is Correct:
model.config.num_labels

from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="H:\TempHF_Cache\TrainingArgs\wikigold_trained_no_DA_testing2",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
    hub_token = "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"
)



#TUNE
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

#IF ERROR WHEN PUSHING TO HUB USE !git lfs install
trainer.train()
trainer.evaluate()
trainer.push_to_hub()
                    
#Test
#from transformers import pipeline
#classifier = pipeline("ner", model=model, tokenizer=tokenizer)
#classifier("My name is John Smith")
