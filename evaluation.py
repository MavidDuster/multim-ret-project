import pandas as pd
import numpy as np
from tqdm import tqdm
from model import retrieve


def get_song_genre(song_id, df_song_info):
    # returns a list of genres of the song
    genres = df_song_info.loc[df_song_info["id"] == song_id]["genre"].item()
    return genres


def relevance(query, retrieved_id, df_song_info):
    """
    Returns relevance rating from 0-4, 0 [irrelevant]…4 [highly rel.])
    """
    id_query = list(df_song_info.loc[df_song_info["song"] == query]["id"])[0]
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


def rel_set(song_id, df_song_info):
    # go over all songs and create set of relevant songs
    qgenre = get_song_genre(song_id, df_song_info)
    rel_ids = []

    for id, genres in zip(df_song_info["id"], df_song_info["genre"]):
        # if genre is matching the song is relevant
        if bool(set(qgenre).intersection(set(genres))):
            rel_ids.append(id)
    return rel_ids


def prec(rel, ret):
    return len(set(rel).intersection(set(ret))) / len(rel)


def recall(rel, ret):
    return len(set(rel).intersection(set(ret))) / len(ret)


# evaluation metrics used for testing
def precision_at_k(df_retrieved, query):
    k = df_retrieved.shape[0]
    rel = 0
    for item in df_retrieved["id"]:
        if relevance(query, item) != 0:
            rel += 1
    return rel / k


def dcg_at_k(df_retrieved, query):
    dcg = 0
    for i, item in enumerate(df_retrieved["id"]):
        if i == 0:
            dcg += relevance(query, item)
        else:
            dcg += relevance(query, item) / np.log(i + 1)  # +1 bc enumerate starts at 0
    return dcg



def idcg(df_retrieved, query):
    # ndcg normalized discounted cumulative gain score [0, 1]
    idcg = 0
    for i, item in enumerate(df_retrieved["id"]):
        if relevance(query, item) == 4:
            pass



def mrr_score(df_retrieved, query):
    q = 0
    k = df_retrieved.shape[0]
    for i, item in enumerate(df_retrieved["id"], start=1):
        if relevance(query, item) != 0:
            q += 1 / i
    q = q / k
    return q


def eval_routine(query_set, ret_method, df_song_info, top_k):
    avg_prec = 0
    avg_mrr = 0
    avg_ndcg = 0

    for query in tqdm(query_set, total=len(query_set)):
        # retrieve items
        ret = retrieve(query, df_song_info, ret_method, top_k)








    return avg_prec, avg_mrr, avg_ndcg


def plot_prec_rec():
    '''
    idk a function that creates a precision recall plot
    :return: a precision recall plot
    '''
    # todo
    pass

