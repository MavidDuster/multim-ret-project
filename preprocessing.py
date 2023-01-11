# used to precompute the data before moving on
import csv

import pandas as pd
from evaluation import get_song_genre
import numpy as np
from tqdm import tqdm
from ast import literal_eval

# load data into

genres = pd.read_csv('./data/id_genres_mmsr.tsv', delimiter='\t')
information_mmsr = pd.read_csv('./data/id_information_mmsr.tsv', delimiter='\t', index_col="id")


# information_mmsr = pd.read_csv('./data/id_information_mmsr.tsv', delimiter='\t')
# lyrics_tf_idf = pd.read_csv('./data/id_lyrics_tf-idf_mmsr.tsv', delimiter='\t')
# lyrics_w2v = pd.read_csv('./data/id_lyrics_word2vec_mmsr.tsv', delimiter='\t')
# maybe load data into numpy instead as it should be faster 

# todo PCA/LSA @david
# create a method that reduces the dimensionallity of our feature set

def check_consitency(df_song_info, df_data):
    """
    remove all songs which dont have a genre
    :param df_song_info: to check if it has genre
    :param df_data: used in retrival
    """
    match = set(df_song_info.index).intersection(df_data.index)
    missmatch = len(df_data) - len(match)
    print(f'Missmatch of size {missmatch} found!')
    df_data = df_data.loc[match]
    return df_data, missmatch


def save_df_as_tsv(path, df):
    df.to_csv(path, sep='\t')
    print("File saved!")


def check_data_genre(list_paths, df_song_info):
    for path in list_paths:
        df_data = pd.read_csv(path, delimiter='\t', index_col="id")
        df_data, m = check_consitency(df_song_info, df_data)
        if m != 0:
            save_df_as_tsv(path, df_data)


def run_check():
    """
    remove all entries for which we dont have genres avilable
    :return:
    """
    genres = pd.read_csv('./data/id_genres_mmsr.tsv', delimiter='\t', index_col="id")
    information_mmsr = pd.read_csv('./data/id_information_mmsr.tsv', delimiter='\t', index_col="id")

    genres['genre'] = genres['genre'].apply(literal_eval)
    df_song_info = information_mmsr.join(genres['genre'])

    id_tfidf = "./data/id_lyrics_tf-idf_mmsr.tsv"
    id_bert = "./data/id_bert_mmsr.tsv"
    # audio data
    id_blf_spectral = "./data/id_blf_spectral_mmsr.tsv"
    id_blf_correlation = "./data/id_blf_correlation_mmsr.tsv"
    # image data
    id_resnet = "./data/id_resnet_mmsr.tsv"
    id_vgg19 = "./data/id_vgg19_mmsr.tsv"

    list_paths =[id_tfidf, id_bert, id_blf_spectral, id_blf_correlation, id_resnet, id_vgg19]
    check_data_genre(list_paths, df_song_info)



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


def precomute


# todo precompute all relevance's for all songs
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


def create_query_set(k, df_song_info):
    q_list = np.random.randint(0, len(df_song_info), size=k)
    query_set = []
    for i in q_list:
        query_set.append(df_song_info["song"][i])
    # todo save that set and load it such that all evals use the same set
    with open(f'sample_{k}.txt', 'w', encoding='utf-8') as f:
        for s in query_set:
            f.write(str(s) + '\n')


# run_check()
# create_query_set(100, information_mmsr)
# create_query_set(1000, information_mmsr)
# create_query_set(68642, information_mmsr)

# idx_m, matrix = create_relevance_matrix(genres)
