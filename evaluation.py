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


def relevance(retrieved_id, q_genres, df_song_info):
    """
    Returns relevance rating from 0-4, 0 [irrelevant]…4 [highly rel.])
    """
    r_genres = df_song_info["genre_set"].loc[retrieved_id]
    intersec = len(q_genres.intersection(r_genres))

    # relevance grades assigned by user (as said in slides)
    # values could be changed
    prec = intersec / len(q_genres)
    if prec == 0:
        return 0
    if prec > 0.8:
        return 4
    if 0.6 < prec <= 0.8:
        return 3
    if 0.25 < prec <= 0.6:
        return 2
    else:
        return 1


def get_rel_set(song_id, df_song_info):
    # go over all songs and create set of relevant songs
    qgenre = get_song_genre(song_id, df_song_info)
    print(qgenre)
    rel = pd.DataFrame(index=df_song_info.index)
    rel["rel_grade"] = rel.apply(
        lambda row: relevance(retrieved_id=row.name, q_genres=qgenre, df_song_info=df_song_info), axis=1)
    print(rel["rel_grade"].unique())
    return rel


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



def precision_at_k(dfTopIds, topNumber):
    
    precision = np.zeros((dfTopIds.shape[0], topNumber))
    recall = np.zeros((dfTopIds.shape[0], topNumber))
    precision_max = np.zeros((dfTopIds.shape[0], topNumber))
    
    for idx,queryId in tqdm(enumerate(dfTopIds.index.values)):
        
        topIds = dfTopIds.loc[queryId].values[:topNumber]
        querySongGenres = genres.loc[[queryId], 'genre'].values[0]
        topSongsGenres = genres.loc[topIds, 'genre'].values
        relevant_results = [isResultRelevant(querySongGenres, songGenre) for songGenre in topSongsGenres]
        REL = np.sum(relevant_results)

        if REL != 0: # Case when there is no relevant result in the top@K
#             P[idx] = [(np.sum(relevant_results[:i+1]) / (i+1)) for i in range(topNumber)]
            precision[idx] = np.divide(np.cumsum(relevant_results,axis=0), np.arange(1,topNumber+1))
#             R[idx] = [(np.sum(relevant_results[:i+1]) / (REL)) for i in range(topNumber)]
            recall[idx] = np.divide(np.cumsum(relevant_results,axis=0), REL)
            precision_max[idx] = [ np.max(precision[idx,i:]) for i,val in enumerate(precision[idx])]

    return precision, recall, precision_max





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
    hits = 0
    dcg = 0
    q = 0
    # todo change eval function such that they match slides
    for i, item in enumerate(df_retrieved["id"].head(top_k)):
        q_genre = df_song_info["genre_set"].loc[q_idx]
        rel = relevance(item, q_genre, df_song_info)
        if rel != 0:
            hits += 1
            if i > 0:
                q += 1 / i
        if i == 0:
            dcg += rel
        else:
            dcg += rel / np.log2(i + 1)

    q = q / top_k
    hits = hits / top_k
    idcg = 0
    # assuming that there are surley 100 relevant song the idcg will converge to the same value
    if top_k is 10:
        idcg = 28.5518
    if top_k is 100:
        idcg = 123.966

    return hits, dcg / idcg, q


def eval_routine(query_set, ret_method, df_song_info, top_k):
    prec = []
    mrr = []
    ndcgs = []

    # go over set of song
    print(f'Current Set contains {len(query_set)} queries')
    print(f'top_k = {top_k}')
    for query in tqdm(query_set):
        # get id of query
        q_idx = get_song_id(query, df_song_info)
        # retrieve items
        ret = retrieve(q_idx, df_song_info, ret_method, top_k)
        # get set of relevant songs given the query
        # rel = get_rel_set(q_idx, df_song_info)
        # evaluate the retrieval based on its used feature
        # todo
        precision, ndcg, q = eval_helper(ret, q_idx, df_song_info, top_k)

        prec.append(precision)
        ndcgs.append(ndcg)
        mrr.append(q)

    avg_prec = sum(prec) / len(query_set)
    avg_mrr = sum(mrr) / len(query_set)
    avg_ndcg = sum(ndcgs) / len(query_set)

    print(f'The average precison @ {top_k} is {avg_prec}')
    print(f'The average MRR @ {top_k} is {avg_mrr}')
    print(f'The average nDCG @ {top_k} is {avg_ndcg}\n')
    return avg_prec, avg_mrr, avg_ndcg


def plot_prec_rec():
    # 
    #def precision_at_k  shuold return (precision, recall, precision_max)
    #we need to _ precision, recall, precision_max
    #
    #ptfidf, rtfidf, ptfidf_max = precision_at_k(top_cosine_tfidf, 100)
    #pw2v, rw2v, pw2v_max = precision_at_k(top_cosine_word2vec, 100)
    #pmfcc, rmfcc, pmfcc_max = precision_at_k(top_cosine_mfcc, 100)
    #
    
    plt.plot(np.mean(rtfidf, axis=0), np.mean(ptfidf_max, axis=0), color='r', label='tf_id cosine')
    plt.plot(np.mean(rw2v, axis=0), np.mean(pw2v_max, axis=0), color='g', label='word2vec cosine')
    plt.plot(np.mean(rmfcc, axis=0), np.mean(pmfcc_max, axis=0), color='b', label='mfcc cosine')
    plt.grid()
    plt.legend()
    plt.show()
    
    
    
    
    
# create a dataframe for evaluation
data = {'AP': ['x', 'y', 'z', 'w','m'],
        'MRR': [99, 98, 95, 90,97],
        'NDCG':[99, 98, 95, 90,97]}

df_evaluate = pd.DataFrame(data, index=['M1','M2','M3','M4','M5'])
df_evaluate
    
