# used to precompute the data before moving on

import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import numpy as np

# load data into
bert_mmsr = pd.read_csv('./data/id_bert_mmsr.tsv', delimiter='\t')
genres = pd.read_csv('./data/id_genres_mmsr.tsv', delimiter='\t')
information_mmsr = pd.read_csv('./data/id_information_mmsr.tsv', delimiter='\t')
lyrics_tf_idf = pd.read_csv('./data/id_lyrics_tf-idf_mmsr.tsv', delimiter='\t')
lyrics_w2v = pd.read_csv('./data/id_lyrics_word2vec_mmsr.tsv', delimiter='\t')


# 