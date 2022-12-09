import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import numpy as np

# load data into
bert_mmsr = pd.read_csv('./data/id_bert_mmsr.tsv', delimiter='\t')
genres = pd.read_csv('./data/id_genres_mmsr.tsv', delimiter='\t')
information_mmsr = pd.read_csv('./data/id_information_mmsr.tsv', delimiter='\t')
lyrics_tf_idf = pd.read_csv('./data/id_lyrics_tf-idf_mmsr.tsv', delimiter='\t')
lyrics_w2v = pd.read_csv('./data/id_lyrics_word2vec_mmsr.tsv', delimiter='\t')

# turn string to list
genres['genre'] = genres['genre'].apply(literal_eval)
dff = genres
dff = dff.explode('genre')

# print(genres.head())
# print(dff.head())
# print()

# most frequent genres
values = dff.genre.value_counts()
print("Most frequent genres:")
print(values.head(10))
print()

# average number of genres per track
print(f'Average Number of genres per track is {round(genres["genre"].str.len().mean())}')
print()

# todo average number of tracks that share one genre
# use one hot encoding
list_inter = []
total_size = len(genres)
for qset in tqdm(genres["genre"], total=total_size):
    qset = set(qset)
    intersec = 0
    # go over all songs and check if genres are intersecting
    for gset in genres["genre"]:
        gset = set(gset)
        if bool(gset.intersection(qset)):
            intersec +=1
    list_inter.append(intersec)

avg_intersec = sum(list_inter)/len(list_inter)
print(f'Average number of tracks that share one genre is {avg_intersec}.')

# total number of genres

