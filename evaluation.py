import pandas as pd
import numpy as np
from tqdm import tqdm
from model import retrieve, retrieval_model2
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

def get_rel_set_size(input_genres, df_song_info):
    # this is sooo slow
    rel = pd.DataFrame(index=df_song_info.index.tolist())
    # find relevant entries
    rel["intersec"] = df_song_info.apply(lambda row: len(np.intersect1d(row["genre"], input_genres)), axis=1)
    # remove all irrelevant entries
    len_rel = len(rel.loc[~(rel == 0).all(axis=1)])
    return len_rel


# evaluation metrics used for testing
def precision_at_k(dfTopIds, topNumber, genres):
    precision = np.zeros((dfTopIds.shape[0], topNumber))
    recall = np.zeros((dfTopIds.shape[0], topNumber))
    precision_max = np.zeros((dfTopIds.shape[0], topNumber))

    for idx, queryId in tqdm(enumerate(dfTopIds.index.values)):
        topIds = dfTopIds.loc[queryId].values[:topNumber]
        querySongGenres = genres.loc[[queryId], 'genre'].values[0]
        topSongsGenres = genres.loc[topIds, 'genre'].values
        relevant_results = [isResultRelevant(querySongGenres, songGenre) for songGenre in topSongsGenres]
        REL = np.sum(relevant_results)

        if REL != 0:  # Case when there is no relevant result in the top@K
            # P[idx] = [(np.sum(relevant_results[:i+1]) / (i+1)) for i in range(topNumber)]
            precision[idx] = np.divide(np.cumsum(relevant_results, axis=0), np.arange(1, topNumber + 1))
            # R[idx] = [(np.sum(relevant_results[:i+1]) / (REL)) for i in range(topNumber)]
            recall[idx] = np.divide(np.cumsum(relevant_results, axis=0), REL)
            precision_max[idx] = [np.max(precision[idx, i:]) for i, val in enumerate(precision[idx])]
    return precision, recall, precision_max


def precision_score(ret, df_song_info, input_genres, len_rel=6000):
    hits = 0
    for index, row in ret.iterrows():
        if len(np.intersect1d(df_song_info.loc[index]["genre"], input_genres)) >= 1:
            hits += 1

    return hits / len(ret), hits / len_rel


def ndcg_score(ret, df_song_info, input_genres, top_k: int):
    relevance_scores = np.zeros(top_k)

    iter_index = 0
    for index, row in ret.iterrows():
        if iter_index >= top_k:
            break

        # give out relevance scores
        intersec = len(np.intersect1d(df_song_info.loc[index]["genre"], input_genres))
        if intersec >= 1:
            prec = intersec / len(input_genres)
            if prec > 0.8:
                relevance_scores[iter_index] = 4
            if 0.6 < prec <= 0.8:
                relevance_scores[iter_index] = 3
            if 0.25 < prec <= 0.6:
                relevance_scores[iter_index] = 2
            else:
                relevance_scores[iter_index] = 1
        iter_index += 1

    dcg = relevance_scores[0]
    idcg = 4

    # compute ndcg
    for i in range(1, top_k):
        dcg += relevance_scores[i] / np.log2(i + 1)
        # assuming that there are a minimum 100 songs that are highly relevant in the whole corpus
        idcg += 4 / np.log2(i + 1)

    return dcg / idcg


def mrr_score(ret, df_song_info, input_genres):
    q = 1
    for index, row in ret.iterrows():
        if len(np.intersect1d(df_song_info.loc[index]["genre"], input_genres)) >= 1:
            break
        else:
            q += 1

    return 1 / q

def pairwise_corr(ret_df1, ret_df2):
    rho, p = spearmanr(ret_df1["cos_sim"], ret_df2["cos_sim"])
    return rho, p


def corr_matrix(baseline, m1, m2, m3, m4, m5):
    ret_list = [baseline, m1, m2, m3, m4, m5]
    corr = []
    for ret_df1 in ret_list:
        temp = []
        for ret_df2 in ret_list:
            rho, p = pairwise_corr(ret_df1, ret_df2)
            temp.append((rho, p))
        corr.append(temp)

    return corr





def plot_prec_rec(m1, m2, m3, top_k, df_song_info):
    # 
    # def precision_at_k  shuold return (precision, recall, precision_max)
    # we need to _ precision, recall, precision_max
    #
    ptfidf, rtfidf, ptfidf_max = precision_at_k(m1, top_k, df_song_info)
    pw2v, rw2v, pw2v_max = precision_at_k(m2, top_k, df_song_info)
    pmfcc, rmfcc, pmfcc_max = precision_at_k(m3, top_k, df_song_info)

    plt.plot(np.mean(rtfidf, axis=0), np.mean(ptfidf_max, axis=0), color='r', label='tf_id cosine')
    plt.plot(np.mean(rw2v, axis=0), np.mean(pw2v_max, axis=0), color='g', label='word2vec cosine')
    plt.plot(np.mean(rmfcc, axis=0), np.mean(pmfcc_max, axis=0), color='b', label='mfcc cosine')
    plt.grid()
    plt.legend()
    plt.show()


def precision_recall_plot(prec_list, recall_list):
    plt.figure()
    plt.scatter(recall_list, prec_list)
    plt.xlabel("Recall")
    plt.ylabel("Precsion")
    plt.show()


def perf_metrics_improved(df_data, df_song_info, subsample, top_k, dim_red=False):
    precision = []
    recall = []
    mrr_sum = 0
    ndcg_sum = 0

    df_data = pd.read_csv(filepath_or_buffer=df_data, delimiter="\t", index_col="id")
    # apply dim reduction
    if dim_red:
        df_index = df_data.index
        pca = PCA(n_components=2)
        df_data_red = pca.fit_transform(df_data)
        df_data = pd.DataFrame(data=df_data_red, index=df_index)
        df_data.index.name = "id"

    for query in tqdm(subsample):
        q_idx = get_song_id(query, df_song_info)
        ret = retrieval_model2(df_data, q_idx, top_k)
        input_genres = df_song_info.loc[q_idx]["genre"]
        # len_rel = get_rel_set_size(input_genres, df_song_info)
        prec, rec = precision_score(ret, df_song_info, input_genres) # input len_rel for plotting
        precision.append(prec)
        recall.append(rec)
        mrr_sum += mrr_score(ret, df_song_info, input_genres)
        ndcg_sum += ndcg_score(ret, df_song_info, input_genres, top_k)

    nsongs = len(subsample)
    avg_prec = sum(precision) / nsongs
    avg_mrr = mrr_sum / nsongs
    avg_ndcg = ndcg_sum / nsongs
    print(f'The average precison @ {top_k} is {avg_prec}')
    print(f'The average MRR @ {top_k} is {avg_mrr}')
    print(f'The average nDCG @ {top_k} is {avg_ndcg}\n')

    # plot
    #precision_recall_plot(recall, precision)

    return avg_prec, avg_mrr, avg_ndcg
