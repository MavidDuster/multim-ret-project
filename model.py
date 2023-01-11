import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


# orignal version
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
def retrieval_model2(df_data, id_query: str, top_k: int = 100):
    """
        given a song this function returns a list of song IDs with similar lyrics

        df_data - Data Frame - containing all song representations where each row is one song
        input_query - string - is ID of the song e.g. 00CH4HJdxQQQbJfu
        top_k - Int - number of neighbors to query song (default 100)
        returns - Data Frame - (song ID: similarity) sorted by most similar to least
        """
    similarity = cosine_similarity(df_data.loc[[id_query]], df_data)[0]

    res = pd.DataFrame(index=df_data.index.tolist())
    res.index.name = "id"
    res["cos_sim"] = similarity
    res.drop([id_query], axis=0, inplace=True)

    return res.nlargest(top_k, "cos_sim")


def retrieve(query_id, modality_path, top_k, dim_red=False):
    # load data into pd
    modality = pd.read_csv(filepath_or_buffer=modality_path, delimiter="\t", index_col="id")
    if dim_red:
        df_index = modality.index
        pca = PCA(n_components=40)
        df_data_red = pca.fit_transform(modality)
        modality = pd.DataFrame(data=df_data_red, index=df_index)
        modality.index.name = "id"

    retrieved = retrieval_model2(modality, query_id, top_k)
    return retrieved


# optionally we could implement PageRank from the slides and see if it yields better results

def early_fusion(modalities: list, weights: list, reduce_dimension: bool = False, n_components: int = 40):
    """
    Fuse features from different modalities using early fusion.

    Parameters:
    - modalities (list of pd.DataFrame): a list of dataframes, each containing the features for a specific modality.
    - weights (list of float): a list of weights for each modality. The length of this list should match the length of modalities.

    Returns:
    - pd.DataFrame: a dataframe containing the concatenated features from all modalities.
    """
    # concatenate the modalities along columns
    if reduce_dimension:
        modalities_reduced = []
        for modality in modalities:
            # pca is used on each modality seperatly to preserve the meaning of each
            pca = PCA(n_components=n_components)
            df_index = modality.index
            df_data_red = pca.fit_transform(modality)
            modalities_reduced.append(pd.DataFrame(data=df_data_red, index=df_index))
        modalities = modalities_reduced

    modality_weights = np.array(weights) / np.sum(weights)
    weighted_modalities = [modalities[i] * modality_weights[i] for i in range(len(modalities))]

    fused_modality = pd.concat(weighted_modalities, axis=1)
    return fused_modality


from sklearn.model_selection import ParameterGrid


# todo finish this function to find best weights for early fusion
def grid_search_early_fusion(modalities_path, df_song_info, id_query, top_k, scoring_functions, genres, dim_red=False):
    def fusion_fn(row, weights):
        return np.dot(row, weights)

    query_song_representation = None
    for modality_path in modalities_path:
        modality = pd.read_csv(filepath_or_buffer=modality_path, delimiter="\t", index_col="id")
        if dim_red:
            df_index = modality.index
            pca = PCA(n_components=40)
            df_data_red = pca.fit_transform(modality)
            modality = pd.DataFrame(data=df_data_red, index=df_index)
            modality.index.name = "id"

        if query_song_representation is None:
            query_song_representation = modality.loc[id_query]
        else:
            query_song_representation = pd.concat([query_song_representation, modality.loc[id_query]], axis=0)

    # Define the parameter grid to search
    param_grid = {'weights': [np.array([w1, w2, w3]), np.array([w4, w5, w6])]}

    # Create a dictionary of all the scoring functions
    scoring = {'ndcg': make_scorer(scoring_functions['ndcg_score'], greater_is_better=True, needs_proba=False),
               'mrr': make_scorer(scoring_functions['mrr_score'], greater_is_better=True, needs_proba=False),
               'precision': make_scorer(scoring_functions['precision_score'], greater_is_better=True,
                                        needs_proba=False)}

    # Perform grid search on the classifier using different evaluation metrics
    grid_search = GridSearchCV(estimator=fusion_fn, param_grid=param_grid, scoring=scoring, refit='ndcg',
                               cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(query_song_representation, genres)
