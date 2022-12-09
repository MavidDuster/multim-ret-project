import pandas as pd
import numpy as np
from tqdm import tqdm
from model import retrieve


def get_song_genre(song_id, df_song_info):
    # returns a set of genres of the song
    genres = df_song_info["genre_set"].loc[song_id]
    return genres


def get_song_id(query, df_song_info):
    return df_song_info.loc[df_song_info["song"] == query].index[0]


def relevance(id_query, retrieved_id, df_song_info):
    """
    Returns relevance rating from 0-4, 0 [irrelevant]…4 [highly rel.])
    """
    q_genres = df_song_info["genre_set"].loc[id_query]
    r_genres = df_song_info["genre_set"].loc[retrieved_id]
    intersec = len(q_genres.intersection(r_genres))

    # relevance grades assigned by user (as said in slides)
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


# def rel_set(song_id, df_song_info):
#     # go over all songs and create set of relevant songs
#     qgenre = get_song_genre(song_id, df_song_info)
#     rel_ids = []
#
#     for id, genres in zip(df_song_info["id"], df_song_info["genre"]):
#         # if genre is matching the song is relevant
#         if bool(set(qgenre).intersection(set(genres))):
#             rel_ids.append(id)
#     return rel_ids

def prec(rel, ret):
    return len(set(rel).intersection(set(ret))) / len(rel)


def recall(rel, ret):
    return len(set(rel).intersection(set(ret))) / len(ret)


# evaluation metrics used for testing
def precision_at_k(df_retrieved, query, df_song_info):
    k = df_retrieved.shape[0]
    rel = 0
    for item in df_retrieved["id"]:
        if relevance(query, item, df_song_info) != 0:
            rel += 1
    return rel / k


def ndcg_at_k(df_retrieved, q_idx, df_song_info):
    dcg = 0
    for i, item in enumerate(df_retrieved["id"]):
        if i == 0:
            dcg += relevance(q_idx, item, df_song_info)
        else:
            dcg += relevance(q_idx, item, df_song_info) / np.log(i + 1)  # +1 bc enumerate starts at 0
    return dcg


def mrr_score(df_retrieved, query, df_song_info):
    q = 0
    k = df_retrieved.shape[0]
    for i, item in enumerate(df_retrieved["id"], start=1):
        if relevance(query, item, df_song_info) != 0:
            q += 1 / i
    q = q / k
    return q


def eval_helper(df_retrieved, q_idx, df_song_info, top_k):
    rel = 0
    dcg = 0
    q = 0

    # todo change eval function such that they match slides
    for i, item in enumerate(df_retrieved["id"]):
        if relevance(q_idx, item, df_song_info) != 0:
            rel += 1
            if i > 0:
                q += 1 / i
        if i == 0:
            dcg += relevance(q_idx, item, df_song_info)
        else:
            dcg += relevance(q_idx, item, df_song_info) / np.log(i + 1)

    q = q / top_k
    rel = rel / top_k

    return rel, dcg, q


def eval_routine(query_set, ret_method, df_song_info, top_k):
    prec = []
    mrr = []
    ndcg = []

    # go over set of song
    print(f'Current Set contains {len(query_set)} queries')
    for query in tqdm(query_set):
        # get id of query
        q_idx = get_song_id(query, df_song_info)
        # retrieve items
        ret = retrieve(q_idx, df_song_info, ret_method, top_k)
        # evaluate the retrieval based on its used feature
        # todo
        precision, dcg, q = eval_helper(ret, q_idx, df_song_info, top_k)

        prec.append(precision)
        ndcg.append(dcg)
        mrr.append(q)

    avg_prec = sum(prec) / len(query_set)
    avg_mrr = sum(mrr) / len(query_set)
    avg_ndcg = sum(ndcg) / len(query_set)

    print(f'The average precison @ {top_k} is {avg_prec}')
    print(f'The average MRR @ {top_k} is {avg_mrr}')
    print(f'The average nDCG @ {top_k} is {avg_ndcg}')
    return avg_prec, avg_mrr, avg_ndcg


def plot_prec_rec():
    """
    IDK a function that creates a precision recall plot
    :return: a precision recall plot
    """

    # todo
    pass
