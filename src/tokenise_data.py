import torch
import os
import sys
import io
import argparse
import logging
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--input_path",
                    type='str')
parser.add_argument("--bert_model",
                    default="bert-base-multilingual-cased",
                    type=str,
                    help="bert-base-multilingual-cased, xlm-roberta-base")
args = parser.parse_args()

assert os.path.exists(args.input_path),"No input files found"

with open(f"./{args.input_path}/input_texts.txt","r") as f:
    data=f.readlines()

labels=np.load(f"./{args.input_path}/labels.npz")['arr_0']

assert len(data)==len(labels)

x_train,x_val,y_train,y_val=train_test_split(data,labels,test_size=0.2,random_state=42)

tokeniser=AutoTokenizer.from_pretrained(args.bert_model)
batch_encode=tokeniser.batch_encode_plus(x_train,max_length=512,add_special_tokens=True,padding='max_length',truncation=True,return_tensors='pt')

if not os.path.exists(f"./{args.input_path}/{args.bert_model}/"):
    os.makedirs(f"./{args.input_path}/{args.bert_model}/")

torch.save(batch_encode['input_ids'],f"./{args.input_path}/{args.bert_model}/train_input_ids.pt")
torch.save(batch_encode['attention_mask'],f"./{args.input_path}/{args.bert_model}/train_attention_masks.pt")
torch.save(torch.tensor(y_train),f"./{args.input_path}/{args.bert_model}/train_labels.pt")

batch_encode=tokeniser.batch_encode_plus(x_val,max_length=512,add_special_tokens=True,padding='max_length',truncation=True,return_tensors='pt')
torch.save(batch_encode['input_ids'],f"./{args.input_path}/{args.bert_model}/val_input_ids.pt")
torch.save(batch_encode['attention_mask'],f"./{args.input_path}/{args.bert_model}/val_attention_masks.pt")
torch.save(torch.tensor(y_val),f"./{args.input_path}/{args.bert_model}/val_labels.pt")