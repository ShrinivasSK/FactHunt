import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm 
import argparse
from selectors import TfIDFSelector,ContrieverSelector,LSASelector

with open("./data/train_examples.json",'r') as f:
    train_data=json.load(f)

def parse_table(sample_table,args):
    data=sample_table.to_dict()
    columns=sample_table.columns
    texts=[]
    if args.parse_rows:
        ## row wise template
        for i in range(len(data[columns[0]])):
            text=f"row {i+1}: "
            for col in columns:
                text+=f"The {col} is {data[col][i]}. "
            texts.append(text)
    if args.parse_columns:
        ## column wise template
        for i,col in enumerate(columns):
            text=f"column: {col}: "
            for j in range(len(data[col])):
                text+=f"Row {j+1} is {data[col][j]}. "
            texts.append(text)
    return texts


parser = argparse.ArgumentParser()
parser.add_argument("--selector",
                    help="Options : [lsa,tfidf,contriever]")
parser.add_argument("--parse_rows",
                    action='store_true')
parser.add_argument("--parse_columns",
                    action='store_true')
parser.add_argument("--csv_path",
                    default="../data/all_csv/")
parser.add_argument('--top_k',
                    deafult=5)
parser.add_argument('--output_path')
args = parser.parse_args()

if args.selector=='tfifdf':
    selector=TfIDFSelector(args)
    all_tables=[]
    for k in tqdm(train_data.keys()):
        sample_table=pd.read_csv(args.csv_path + k,sep='#') 
        all_tables.append(parse_table(sample_table))

    selector.fit([num for elem in all_tables for num in elem])
elif args.selector=='contriever':
    selector=ContrieverSelector(args)
elif args.selector=='lsa':
    selector=LSASelector()
    all_tables=[]
    for k in tqdm(train_data.keys()):
        sample_table=pd.read_csv(args.csv_path + k,sep='#') 
        all_tables.append(parse_table(sample_table))

    selector.fit([num for elem in all_tables for num in elem])
else:
    raise ValueError("Selector must be among [lsa,tfidf,contriever]")

all_queries=[]
labels=[]
for i,(k,v) in enumerate(tqdm(train_data.items())):
    for j,fact in enumerate(v[0]):
        text=""

        text+=f"Fact: {fact}. "
        
        sents=selector.select_rows(fact,all_tables[i])
        text+=" ".join(sents)

        all_queries.append(text)
        labels.append(v[1][j])

if not os.path.exits(f"./{args.output_path}/"):
    os.makedirs(f"./{args.output_path}/")

with open(f"./{args.output_path}/input_texts.txt","w") as f:
    f.writelines([q + "\n" for q in all_queries])

np.savez(f"./{args.output_path}/labels.npz",labels)


    


