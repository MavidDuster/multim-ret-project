# imports
import pandas as pd
import numpy as np
from model import retrieval_model, retrieve
from evaluation import get_song_genre, relevance, recall, prec, eval_routine, get_rel_set, precision_at_k, \
    plot_prec_rec, pairwise_corr, corr_matrix
from ast import literal_eval
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt

# data import
genres = pd.read_csv('./data/id_genres_mmsr.tsv', delimiter='\t', index_col="id")
information_mmsr = pd.read_csv('./data/id_information_mmsr.tsv', delimiter='\t', index_col="id")

# lyric data
id_tfidf = "./data/id_lyrics_tf-idf_mmsr.tsv"
id_bert = "./data/id_bert_mmsr.tsv"
# audio data
id_blf_spectral = "./data/id_blf_spectral_mmsr.tsv"
id_blf_correlation = "./data/id_blf_correlation_mmsr.tsv"
# image data
id_resnet = "./data/id_resnet_mmsr.tsv"
id_vgg19 = "./data/id_vgg19_mmsr.tsv"

# merge dfs and processes genre column
genres['genre'] = genres['genre'].apply(literal_eval)
df_song_info = information_mmsr.join(genres['genre'])
df_song_info["genre_set"] = df_song_info["genre"].apply(set)

# print(df_song_info["song"])
# for testing
query = "Can You Feel My Heart"
query_id = df_song_info.loc[df_song_info["song"] == query].index[0]
top_k = 10


# # testing retrieve items
baseline = retrieve(query_id, id_tfidf)
print("TF-IDF with cosine is done")
m1= retrieve(query_id, id_bert)
print("BERT with cosine is done")
m2= retrieve(query_id, id_blf_spectral)
print("BLF Spectral with cosine is done")
m3= retrieve(query_id, id_blf_correlation)
print("BLF Correlation with cosine is done")
m4= retrieve(query_id, id_resnet)
print("ResNet with cosine is done")
m5= retrieve(query_id, id_vgg19)
print("VGG19 with cosine is done")


# checking pairwise correlation
# correlation coeff always 1 (almost 1) idk why
print(corr_matrix(baseline, m1, m2, m3, m4, m5))


# load a random subset of queries
with open("sample_1000.txt", 'r', encoding="utf-8") as f:
    subset1000 = [line.rstrip('\n') for line in f]

    
def heat_map(query_set,top_k):
    ap_Base, mrr_Base, ndcg_Base = eval_routine(query_set, baseline['cos_sim'], df_song_info, top_k)
    ap_M1, mrr_M1, ndcg_M1 = eval_routine(query_set, m1['cos_sim'], df_song_info, top_k)
    ap_M2, mrr_M2, ndcg_M2 = eval_routine(query_set, m2['cos_sim'], df_song_info, top_k)
    ap_M3, mrr_M3, ndcg_M3 = eval_routine(query_set, m3['cos_sim'], df_song_info, top_k)
    ap_M4, mrr_M4, ndcg_M4 = eval_routine(query_set, m4['cos_sim'], df_song_info, top_k)
    ap_M5, mrr_M5, ndcg_M5 = eval_routine(query_set, m5['cos_sim'], df_song_info, top_k)
    
    # Heat map

    average = ["AP", "MRR", "leCG"]
    models = ["TF-IDF", "BERT", "BLF Spectral","BLF Correlation", "ResNet", "VGG19"]


    arr_base = [ap_Base, mrr_Base, ndcg_Base]
    arr_m1   = [ap_M1, mrr_M1, ndcg_M1]
    arr_m2   = [ap_M2, mrr_M2, ndcg_M2]
    arr_m3   = [ap_M3, mrr_M3, ndcg_M3]
    arr_m4   = [ap_M4, mrr_M4, ndcg_M4]
    arr_m5   = [ap_M5, mrr_M5, ndcg_M5]

    heat = np.array([arr_base,arr_m1,arr_m2,arr_m3,arr_m4,arr_m5])


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
    
    
    
    
heat_map("Can You Feel My Heart",10)
heat_map("Can You Feel My Heart",100)
    
 


# main format : plt.plot(recall_baseline, precision_baseline, 'r')
#These are only for test and show, We should change them

plt.plot(m3['cos_sim'], baseline['cos_sim'], 'r')
plt.plot(m4['cos_sim'], m2['cos_sim'], 'b')
plt.plot(m5['cos_sim'], m2['cos_sim'], 'g')
plt.plot(m3['cos_sim'], m2['cos_sim'], 'y')
plt.plot(m5['cos_sim'], m3['cos_sim'], 'm')
plt.plot(m3['cos_sim'], m5['cos_sim'], 'c')

plt.legend(['baseline', 'M1','M2', 'M3', 'M4', 'M5' ])

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision versus Recall')
plt.show()
    
    
    
    
    
    
    
    
    
    
    
    

# run evaluation routine
#print("Using tf-idf")
#print(eval_routine(subset1000, id_tfidf, df_song_info, top_k))

#print("Using bert embedding")
#print(eval_routine(subset1000, id_bert, df_song_info, top_k))

# print("Using blf_spectral")
# #print(eval_routine(subset1000, id_blf_spectral, df_song_info, top_k))
#
# print("Using blf_correlation")
# print(eval_routine(subset1000, id_blf_correlation, df_song_info, top_k))
#
# print("Using resnet")
# print(eval_routine(subset1000, id_resnet, df_song_info, top_k))
#
# print("Using vgg19")
# print(eval_routine(subset1000, id_vgg19, df_song_info, top_k))


#plot_prec_rec(re, m2, m3, top_k)

# get precision / recall plot of baseline retrieval system (from task1)
# rel = rel_set(query_id, df_song_info)  # get all ids of relevant songs
