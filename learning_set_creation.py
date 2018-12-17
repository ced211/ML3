import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse import find
from toy_example import measure_time
def create_learning_matrices2(rating_matrix,user_movie_pairs):
    """create the learning matrice [sn_sample,10] from the rating_matrix, and the user movie pair.
        the feature vector is composed with the total number of rank1 given by the user, total of rank2 given
        by the user, ... the total of rank 1 received by the movie, the total of rank 2 received by the movie, ...

    Parameters
    ----------
    rating_matrix: sparse matrix [n_users, n_movies]
        The rating matrix. i.e. `rating_matrix[u, m]` is the rating given
        by the user `u` for the movie `m`. If the user did not give a rating for
        that movie, `rating_matrix[u, m] = 0`
    user_movie_pairs: array [n_predictions, 2]
        If `u, m = user_movie_pairs[i]`, the i-th raw of the learning matrix
        must relate to user `u` and movie `m`

    Return
    ------
    X: sparse array [n_predictions, n_users + n_movies]
        The learning matrix in csr sparse format
    """
    with measure_time('Building learning matrix'):
        # Create intermediate matrix
        users_rank = np.zeros((rating_matrix.shape[0], 5))

        for userID in range(rating_matrix.shape[0]):
            (dont_care1, dont_care2, user_rank) = sparse.find(rating_matrix[userID,:])
            for rank in user_rank:
                users_rank[userID, rank-1] += 1
        movies_rank = np.zeros((rating_matrix.shape[1], 5))

        for movieID in range(rating_matrix.shape[1]):
            (dont_care1, dont_care2, movie_rank) = sparse.find(rating_matrix[:,movieID].transpose())
            for rank in movie_rank:
                movies_rank[movieID, rank-1] += 1

        user_features = users_rank[user_movie_pairs[:, 0]]
        movie_features = movies_rank[user_movie_pairs[:, 1]]
        X = np.concatenate((user_features, movie_features), axis=1)

        return X


def create_learning_matrices3(User_movie_pairs):
    """
        Create the learning matrix `X` from the `rating_matrix`.

        If `u, m = user_movie_pairs[i]`, then X[i] is the feature vector
        corresponding to user `u` and movie `m`. The feature vector is composed
        of `n_user_features_kept + n_movie_features_kept` features. The features
        kept depend of the beginning of this method where removed features are specified.

        In other words, the feature vector for a pair (user, movie) is the
        concatenation of values of the corresponding user features and values
        of the corresponding movie features.

        Parameters
        ----------
        User_movie_pairs: array [n_predictions, 2]
            If `u, m = user_movie_pairs[i]`, the i-th raw of the learning matrix
            must relate to user `u` and movie `m`

        Return
        ------
        ret: numpy array with n_predictions rows and n_features_kept columns
    """
    X = list()
    prefix = 'data/'
    user = load_from_csv(os.path.join(prefix, 'data_user.csv'))
    movie = load_from_csv(os.path.join(prefix, 'data_movie.csv'))

    # Headers of both cvs file
    userHeader = pd.read_csv(os.path.join(prefix, 'data_user.csv'), nrows=1).columns
    movieHeader = pd.read_csv(os.path.join(prefix, 'data_movie.csv'), nrows=1).columns
    u = dict()
    m = dict()

    # Associating header names to corresponding indices in the user and movie matrix
    k = 0
    for i in userHeader:
        u[i] = k
        k += 1
    k = 0
    for i in movieHeader:
        m[i] = k
        k += 1

    # lists of features to delete represented by their indexes
    us = list()
    ms = list()

    # Features removed in user matrix
    us.append(u["user_id"])

    # Features removed in movie matrix
    ms.append(m["movie_id"])
    ms.append(m["movie_title"])

    # ms.append(m["release_date"])

    ms.append(m["video_release_date"])
    ms.append(m["IMDb_URL"])

    for (x, y) in User_movie_pairs:

        # Handle unknown row exception
        if movie[y - 1][m["movie_title"]] == "unknown":
            movie[y - 1][m["release_date"]] = 0

        if isinstance(movie[y - 1][m["release_date"]], str):
            movie[y - 1][m["release_date"]] = int(movie[y - 1][2].split("-")[-1])  # Extract year release as int

        user[x - 1][u["zip_code"]] = user[x - 1][u["zip_code"]][:2]  # extract 2 first letter of zip code

        # remove columns given indexes and append both results in tmp
        tmp = np.append(np.delete(user[x - 1], us),
                        np.delete(movie[y - 1], ms))

        # Hashing of string data into int to be processed by the models
        for i in range(len(tmp)):
            if isinstance(tmp[i], str):
                tmp[i] = hash(tmp[i])

        X.append(tmp)

    X = np.asarray(X)

    return X

def create_movie(movie):
    # return a vector output where output[i] correspond to the category of the movie i.
    # Two movie belong to the same movie_category if they has the same category. 
    # A movie can only belong to one category. If it has several, the category which has the higher number of movie is selected.
    movie = movie[:, 5:]
    total = np.sum(movie, axis=0)
    permutation = np.flip(np.argsort(total), axis=0)
    movie = movie[:, permutation]
    output = movie[:, 0]
    for j in range(movie.shape[0]):
        i = 0
        while movie[j][i] == 0:
            i += 1
        output[j] = i

    return output

def create_user(user):
    #categorie depend on occupation and sexe and age.
    #return a vector output where output[i] correspond to the user_category of the user i.
    #category are integer from 0 to n, where n is the number of different category.
    user = user[:, (1, 2, 3)]
    categorieID = 0
    categorie = {}
    output = user[:, 0]
    i = 0
    user_per_categorie = []
    for u in user:
        key = str(int(u[0]/10))+u[1]+u[2]
        if key not in categorie:
            categorie[key] = categorieID
            output[i] = categorieID
            user_per_categorie.append(1)            
            categorieID += 1
        else:
            output[i] = categorie[key]
            user_per_categorie[categorie[key]] += 1
        i += 1
    return output

def create_learning_matrices4(movie, user, rating_matrix, user_movie_pair):
    movie = create_movie(movie)
    user = create_user(user)
    M = np.zeros((max(user)+1, 19, 5))
    feature = np.zeros((user_movie_pair.shape[0], 5))

    for u, m in user_movie_pair:
        if rating_matrix[u, m] != 0:
            M[user[u-1], movie[m-1], rating_matrix[u, m]-1] += 1
    i = 0
    for u, m in user_movie_pair:
        feature[i] = M[user[u-1], movie[m-1], :]
        i += 1

    return np.concatenate((create_learning_matrices2(rating_matrix,user_movie_pair),feature),axis=1)
