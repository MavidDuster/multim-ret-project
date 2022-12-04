import pandas as pd
import numpy as np

# load loacally
bert_mmsr = pd.read_csv('./data/id_bert_mmsr.tsv', delimiter='\t')
genres = pd.read_csv('./data/id_genres_mmsr.tsv', delimiter='\t')
information_mmsr = pd.read_csv('./data/id_information_mmsr.tsv', delimiter='\t')
lyrics_tf_idf = pd.read_csv('./data/id_lyrics_tf-idf_mmsr.tsv', delimiter='\t')
lyrics_w2v = pd.read_csv('./data/id_lyrics_word2vec_mmsr.tsv', delimiter='\t')

# todo most frequent genres


# todo average number of genres per track


# todo  average number of tracks that share one genre

# todo whatever else we can think of