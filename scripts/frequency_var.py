import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

# most freq
df = pd.read_csv("data/most_frequent_words.txt", sep=',')
scores = pd.read_csv("data/var_scores_k10.txt", sep=' ')
# xmas
xmas = pd.read_csv("data/xmas_freq.txt", sep="   ")
xmas_scores = pd.read_csv("data/var_score_xmas_10.txt")


df['W2V/PPMI'] = scores['W2V/PPMI']
df['W2V/PCA'] = scores['W2V/PCA']
df['PPMI/PCA'] = scores['PPMI/PCA']
df['Type'] = ['most_freq'] * 25

xmas['W2V/PPMI'] = xmas_scores['W2V/PPMI']
xmas['W2V/PCA'] = xmas_scores['W2V/PCA']
xmas['PPMI/PCA'] = xmas_scores['PPMI/PCA']
xmas['Type'] = ['xmas'] * 24

frames = [df, xmas]
result = pd.concat(frames)

sn.scatterplot(data=result, x='Frequency', y='PPMI/PCA', hue='Type', style='Type', markers=['o', '^'], alpha=0.9)
plt.show()
