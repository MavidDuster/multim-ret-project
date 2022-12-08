# imports
import pandas as pd
import numpy as np
from model import retrieval_model, retrieve
from evaluation import get_song_genre, relevance, rel_set, recall, prec
from ast import literal_eval
from sklearn.metrics import PrecisionRecallDisplay

# data import
genres = pd.read_csv('./data/id_genres_mmsr.tsv', delimiter='\t')
information_mmsr = pd.read_csv('./data/id_information_mmsr.tsv', delimiter='\t')

# lyric data
id_tfidf = "./data/id_lyrics_tf-idf_mmsr.tsv"
id_bert = "./data/id_bert_mmsr.tsv"
# audio data
id_blf_spectral = "./data/id_blf_spectral_mmsr.tsv"
id_blf_correlation = "./data/id_blf_correlation_mmsr.tsv"
# image data
id_resnet = "./data/id_resnet_mmsr.tsv"
id_vgg19 = "./data/id_vgg19_mmsr.tsv"



genres['genre'] = genres['genre'].apply(literal_eval)
df_song_info = information_mmsr.join(genres['genre'])

# for testing
query = "Flames of Revenge"
query_id = list(df_song_info.loc[df_song_info["song"] == query]["id"])[0]
top_k = 10

# testing retrieve items
# baseline_tf_idf_cosine_sim = retrieve(query, information_mmsr, id_tfidf, top_k)
# print("TF-IDF with cosine is done")
# m1_bert_cosine_sim = retrieve(query, information_mmsr, id_bert, top_k)
# print("BERT with cosine is done")
# m2_blf_spectral_cosine_sim = retrieve(query, information_mmsr, id_blf_spectral, top_k)
# print("BLF Spectral with cosine is done")
# m3_id_blf_correlation_cosine_sim = retrieve(query, information_mmsr, id_blf_correlation, top_k)
# print("BLF Correlation with cosine is done")
# m4_id_resnet_cosine_sim = retrieve(query, information_mmsr, id_resnet, top_k)
# print("ResNet with cosine is done")
# m5_id_vgg19_cosine_sim = retrieve(query, information_mmsr, id_vgg19, top_k)
# print("VGG19 with cosine is done")


# create a subset of queries
def create_query_set(k, df_song_info):
    q_list = np.random.randint(0, len(df_song_info), size=k)
    query_set = []
    for i in q_list:
        query_set.append(df_song_info["song"][i])
    return query_set


subset100 = create_query_set(100, df_song_info)

# evaluation



# get precison / recall plot of baseline retrieval system (from task1)

#rel = rel_set(query_id, df_song_info)  # get all ids of relevant songs

print(prec(rel, retrieved_tf_idf_cosine_sim["id"]))
print(recall(rel, retrieved_tf_idf_cosine_sim["id"]))
