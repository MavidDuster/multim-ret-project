import pandas as pd
import numpy as np
from tqdm import tqdm
from model import retrieve, retrieval_model2
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, ndcg_score



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


def precision_(ret, df_song_info, input_genres, len_rel=27):
    hits = 0
    for index, row in ret.iterrows():
        if len(np.intersect1d(df_song_info.loc[index]["genre"], input_genres)) >= 1:
            hits += 1

    return hits / len(ret), hits / len_rel


def ndcg_(ret, df_song_info, input_genres, top_k: int):
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


def mrr_(ret, df_song_info, input_genres):
    q = 1
    for index, row in ret.iterrows():
        # repeat till first relevant  song is found
        if len(np.intersect1d(df_song_info.loc[index]["genre"], input_genres)) >= 1:
            break
        else:
            q += 1
    return 1 / q



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
    plt.ylabel("Precision")
    plt.show()


def perf_metrics_improved(data_loc, df_song_info, subsample, top_k, dim_red=False, n_components=40):
    precision = []
    recall = []
    mrr_sum = 0
    ndcg_sum = 0

    df_data = pd.read_csv(filepath_or_buffer=data_loc, delimiter="\t", index_col="id")
    # apply dim reduction
    if dim_red:
        df_index = df_data.index
        pca = PCA(n_components=n_components)
        df_data_red = pca.fit_transform(df_data)
        df_data = pd.DataFrame(data=df_data_red, index=df_index)
        df_data.index.name = "id"

    for query in tqdm(subsample):
        q_idx = get_song_id(query, df_song_info)
        ret = retrieval_model2(df_data, q_idx, top_k)
        input_genres = df_song_info.loc[q_idx]["genre"]
        # len_rel = get_rel_set_size(input_genres, df_song_info) # used for recall
        prec, rec = precision_(ret, df_song_info, input_genres)  # input len_rel for plotting
        precision.append(prec)
        recall.append(rec)
        mrr_sum += mrr_(ret, df_song_info, input_genres)
        ndcg_sum += ndcg_(ret, df_song_info, input_genres, top_k)

    nsongs = len(subsample)
    avg_prec = sum(precision) / nsongs
    avg_mrr = mrr_sum / nsongs
    avg_ndcg = ndcg_sum / nsongs
    #print(f'The average precison @ {top_k} is {avg_prec}')
    #print(f'The average MRR @ {top_k} is {avg_mrr}')
    #print(f'The average nDCG @ {top_k} is {avg_ndcg}\n')

    # plot
    # precision_recall_plot(recall, precision) # looks weird
    return avg_prec, avg_mrr, avg_ndcg


def heat_map(query_set, top_k):
    ap_Base, mrr_Base, ndcg_Base = eval_routine(query_set, baseline['cos_sim'], df_song_info, top_k)
    ap_M1, mrr_M1, ndcg_M1 = eval_routine(query_set, m1['cos_sim'], df_song_info, top_k)
    ap_M2, mrr_M2, ndcg_M2 = eval_routine(query_set, m2['cos_sim'], df_song_info, top_k)
    ap_M3, mrr_M3, ndcg_M3 = eval_routine(query_set, m3['cos_sim'], df_song_info, top_k)
    ap_M4, mrr_M4, ndcg_M4 = eval_routine(query_set, m4['cos_sim'], df_song_info, top_k)
    ap_M5, mrr_M5, ndcg_M5 = eval_routine(query_set, m5['cos_sim'], df_song_info, top_k)

    # Heat map

    average = ["AP", "MRR", "NDCG"]
    models = ["TF-IDF", "BERT", "BLF Spectral", "BLF Correlation", "ResNet", "VGG19"]

    arr_base = [ap_Base, mrr_Base, ndcg_Base]
    arr_m1 = [ap_M1, mrr_M1, ndcg_M1]
    arr_m2 = [ap_M2, mrr_M2, ndcg_M2]
    arr_m3 = [ap_M3, mrr_M3, ndcg_M3]
    arr_m4 = [ap_M4, mrr_M4, ndcg_M4]
    arr_m5 = [ap_M5, mrr_M5, ndcg_M5]

    heat = np.array([arr_base, arr_m1, arr_m2, arr_m3, arr_m4, arr_m5])

    fig, ax = plt.subplots()
    im = ax.imshow(heat)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(models)), labels=models)
    ax.set_yticks(np.arange(len(average)), labels=average)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(average)):
        for j in range(len(models)):
            text = ax.text(j, i, heat[i, j], ha="center", va="center", color="w")

    ax.set_title("Evaluation Heat Map")
    fig.tight_layout()
    plt.show()


def eval_routine(df_data, df_song_info, subsample, top_k):
    precision = []
    recall = []
    mrr_sum = 0
    ndcg_sum = 0

    for query in tqdm(subsample):
        q_idx = get_song_id(query, df_song_info)
        ret = retrieval_model2(df_data, q_idx, top_k)
        input_genres = df_song_info.loc[q_idx]["genre"]

        prec, rec = precision_(ret, df_song_info, input_genres)  # input len_rel for plotting
        precision.append(prec)
        recall.append(rec)
        mrr_sum += mrr_(ret, df_song_info, input_genres)
        ndcg_sum += ndcg_(ret, df_song_info, input_genres, top_k)

    nsongs = len(subsample)
    avg_prec = sum(precision) / nsongs
    avg_mrr = mrr_sum / nsongs
    avg_ndcg = ndcg_sum / nsongs
    return avg_prec, avg_mrr, avg_ndcg


