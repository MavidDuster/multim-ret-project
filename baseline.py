import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def baseline_retrival(query, df_song_inf, df_lyric_rep, top_k=5):
    '''
    given a song this function returns a list of song IDs with similar lyrics

    query - String of song name
    df_song_inf - Data Frame -  used to get ID of song name
    df_lyric_rep - Data Frame - used to get tf-idf representation of query song
    top_k - Int - number of neighbors to query song (default 5)

    returns - Data Frame - (song ID: similarity) sorted by most similar to least
    '''
    df_rep = df_lyric_rep.copy(deep=True)
    # get songID of query
    id_query = list(df_song_inf.loc[df_song_inf["song"] == query]["id"])[0]

    # get lyrics representation of query
    query_song = df_rep.loc[df_rep["id"] == id_query]
    # safe as np array and exclude the ID column
    query_song = np.array(query_song.drop(columns=['id']))

    df_id = np.array(df_rep.drop(columns=['id']))

    # compute similarity
    cosine_sim = pd.Series(cosine_similarity(query_song, df_id)[0])

    # create returning df id + similarity score
    closest_items = pd.DataFrame(df_rep['id'])
    closest_items["cos_sim"] = cosine_sim

    # Taking K most similar items
    closest_items = closest_items.sort_values('cos_sim', ascending=False)
    return closest_items[1:top_k + 1]  # exclude the first one as it is the query song itself



