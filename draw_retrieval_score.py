import json
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
args = parser.parse_args()
model_version = args.model

with open(f'./head_score/{model_version}.json') as file:
    head_list = json.loads(file.readline())

head_score_list = [([int(ll) for ll in l[0].split("-")], np.mean(l[1])) for l in head_list.items()]
head_score_list = sorted(head_score_list, key=lambda x: x[1], reverse=True)
max_row = max([coord[0] for coord, _ in head_score_list])
max_col = max([coord[1] for coord, _ in head_score_list])
matrix = np.zeros((max_row + 1, max_col + 1), dtype=np.float64)
for coord, value in head_score_list:
    row, col = coord
    matrix[row, col] = value

plt.rcParams['figure.dpi'] = 300
plt.figure(figsize=(8, 6))
ax = sns.heatmap(matrix, annot=False, cmap='Reds', vmin=0., vmax=1.)
ax.collections[0].colorbar.set_label('Retrieval Score', rotation=270, labelpad=20)
plt.title(f'{args.model}')
plt.xlabel('Head')
plt.ylabel('Layer')
plt.savefig(f'./viz/retrieval_score/{model_version}.png')
plt.close()
