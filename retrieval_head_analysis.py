import json
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, help='file name')
args = parser.parse_args()
with open(f'./head_score/{args.f}') as file:
    head_list = json.loads(file.readline())
## use the average retrieval score and ranking
head_score_list = [([int(ll) for ll in l[0].split("-")],np.mean(l[1])) for l in head_list.items()]
head_score_list = sorted(head_score_list, key=lambda x: x[1], reverse=True)
top_retrieval_heads = [[l[0],  round(np.mean(l[1]), 2)] for l in head_score_list][:10]
print(top_retrieval_heads)