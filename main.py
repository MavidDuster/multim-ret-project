# imports
import pandas as pd
from baseline import baseline_retrival
from evaluation import get_song_genre, relevance, rel_set, recall, prec
from ast import literal_eval
from sklearn.metrics import PrecisionRecallDisplay


# data import
bert_mmsr = pd.read_csv('./data/id_bert_mmsr.tsv', delimiter='\t')
genres = pd.read_csv('./data/id_genres_mmsr.tsv', delimiter='\t')
information_mmsr = pd.read_csv('./data/id_information_mmsr.tsv', delimiter='\t')
lyrics_tf_idf = pd.read_csv('./data/id_lyrics_tf-idf_mmsr.tsv', delimiter='\t')
lyrics_w2v = pd.read_csv('./data/id_lyrics_word2vec_mmsr.tsv', delimiter='\t')

genres['genre'] = genres['genre'].apply(literal_eval)
df_song_info = information_mmsr.join(genres['genre'])


query = "Flames of Revenge"
query_id = list(df_song_info.loc[df_song_info["song"]== query]["id"])[0]
top_k = 10

# retrieve items
retrieved_tf_idf_cosine_sim = baseline_retrival(query, df_song_info, lyrics_tf_idf, top_k=top_k)

# get precison / recall plot of baseline retrieval system (from task1)
rel = rel_set(query_id, df_song_info) # get all ids of relevant songs

print(prec(rel, retrieved_tf_idf_cosine_sim["id"]))
print(recall(rel, retrieved_tf_idf_cosine_sim["id"]))











