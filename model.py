import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def retrieval_model(id_query, df_lyric_rep):
    """
    given a song this function returns a list of song IDs with similar lyrics

    query - String of song name
    df_song_inf - Data Frame -  used to get ID of song name
    df_lyric_rep - Data Frame - used to get tf-idf representation of query song
    top_k - Int - number of neighbors to query song (default 5)

    returns - Data Frame - (song ID: similarity) sorted by most similar to least
    """
    df_rep = df_lyric_rep.copy(deep=True)

    # get lyrics representation of query
    query_song = df_rep.loc[df_rep.index == id_query]
    # safe as np array and exclude the ID column
    query_song = np.array(query_song)
    np_songs = np.array(df_rep)

    # compute similarity
    cosine_sim = pd.Series(cosine_similarity(query_song, np_songs)[0])

    # create returning df id + similarity score
    closest_items = pd.DataFrame(df_rep.index)
    closest_items["cos_sim"] = cosine_sim

    closest_items = closest_items.sort_values('cos_sim', ascending=False)
    return closest_items[1:]  # exclude the first one as it is the query song itself


# simplified version of the above
def retrieval_model2(df_data, input_query: str, top_k: int = 100):
    similarity = cosine_similarity(df_data.loc[[input_query]], df_data)[0]

    res = pd.DataFrame(index=df_data.index.tolist())
    res.index.name = "id"
    res["cos_sim"] = similarity
    res.drop([input_query], axis=0, inplace=True)

    return res.nlargest(top_k, "cos_sim")


def retrieve(query_id, modality_path):
    # load data into pd
    modality = pd.read_csv(filepath_or_buffer=modality_path, delimiter="\t", index_col="id")
    retrieved = retrieval_model(query_id, modality)
    return retrieved

# optionally we could implement PageRank from the slides and see if it yields better results
