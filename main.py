# imports
import pandas as pd
import numpy as np
from model import retrieval_model2, retrieve, early_fusion
from evaluation import evaluation_routine
from ast import literal_eval
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt
import seaborn as sns


def init():
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

    # list of paths
    data_loc = [id_tfidf, id_bert, id_blf_spectral, id_blf_correlation, id_resnet, id_vgg19]

    # merge dfs and processes genre column
    genres['genre'] = genres['genre'].apply(literal_eval)
    df_song_info = information_mmsr.join(genres['genre'])
    df_song_info["genre_set"] = df_song_info["genre"].apply(set)

    return df_song_info, data_loc


if __name__ == "__main__":
    # load subset of queries
    df_song_info, data_loc = init()
    # load subsample
    with open("sample_100.txt", 'r', encoding="utf-8") as f:
        subset = [line.rstrip('\n') for line in f]

    TOP_K = 100
    test_id = "Y0rtcr77gzdsj7ED" # Can You Feel My Heart

    # load data
    df_lyrics = pd.read_csv(filepath_or_buffer=data_loc[1], delimiter="\t", index_col="id")
    df_audio = pd.read_csv(filepath_or_buffer=data_loc[2], delimiter="\t", index_col="id")
    df_audio2 = pd.read_csv(filepath_or_buffer=data_loc[3], delimiter="\t", index_col="id")
    modalities = [df_lyrics, df_audio, df_audio2]
    weights = [0.4, 0.4, 0.2]

    # early fusion
    fused_representation = early_fusion(modalities, weights, reduce_dimension=True, n_components=40)

    #
    print(evaluation_routine(fused_representation, df_song_info, subset, TOP_K))

