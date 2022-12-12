import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import numpy as np

# load data into
bert_mmsr = pd.read_csv('./data/id_bert_mmsr.tsv', delimiter='\t')
genres = pd.read_csv('./data/id_genres_mmsr.tsv', delimiter='\t')
information_mmsr = pd.read_csv('./data/id_information_mmsr.tsv', delimiter='\t')


genres['genre'] = genres['genre'].apply(literal_eval)


# turn string to list

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

# average number of tracks that share one genre
# use one hot encoding
genres1hot = pd.get_dummies(genres.genre.apply(pd.Series), prefix="", prefix_sep="")
print(f'There are a total of {genres1hot.shape[1]} genres')
print(f'Average number of songs that share a genre {genres1hot.sum().mean()}')



