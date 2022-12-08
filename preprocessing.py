# used to precompute the data before moving on

import pandas as pd
from evaluation import get_song_genre
import numpy as np
from tqdm import tqdm

# load data into

# bert_mmsr = pd.read_csv('./data/id_bert_mmsr.tsv', delimiter='\t')
# bert_mmsr = pd.read_csv('./data/id_bert_mmsr.tsv', delimiter='\t')
genres = pd.read_csv('./data/id_genres_mmsr.tsv', delimiter='\t')


# information_mmsr = pd.read_csv('./data/id_information_mmsr.tsv', delimiter='\t')
# lyrics_tf_idf = pd.read_csv('./data/id_lyrics_tf-idf_mmsr.tsv', delimiter='\t')
# lyrics_w2v = pd.read_csv('./data/id_lyrics_word2vec_mmsr.tsv', delimiter='\t')
# maybe load data into numpy instead as it should be faster 

# todo PCA/LSA @david
# create a method that reduces the dimensionallity of our feature set
#

def rel_set(song_id, df_song_info):
    # go over all songs and create set of relevant songs
    qgenre = get_song_genre(song_id, df_song_info)
    rel_ids = []

    for id, genres in zip(df_song_info["id"], df_song_info["genre"]):
        # if genre is matching the song is relevant
        if bool(set(qgenre).intersection(set(genres))):
            rel_ids.append(id)
    return rel_ids


def relevance(id_query, retrieved_id, df_song_info):
    """
    Returns relevance rating from 0-4, 0 [irrelevant]…4 [highly rel.])
    """
    q_genres = get_song_genre(id_query, df_song_info)
    r_genres = get_song_genre(retrieved_id, df_song_info)
    intersec = len(set(q_genres).intersection(set(r_genres)))

    # relevance grades assinged by user (as said in slides)
    # values could be changed
    prec = intersec / len(q_genres)
    if prec == 0:
        return 0
    if prec == 1:
        return 4
    if 0.75 < prec < 1:
        return 3
    if 0.25 < prec <= 0.75:
        return 2
    else:
        return 1

# todo precomute all relevances for all songs
def create_relevance_matrix(df_song_info):
    """
    each row and col are a song and its entry is the relevance grade it received
    :return: n x n matrix of relevance scores
    """

    return_df = df_song_info["id"]
    return_df.set_index("id", inplace=True)
    idx_m = {}
    matrix = []
    for idx, q_item in tqdm(enumerate(df_song_info["id"]), total=len(df_song_info)):
        return_df[q_item] = relevance()
        pass


#idx_m, matrix = create_relevance_matrix(genres)
df = df_song_info["id"]
print(df.head())
