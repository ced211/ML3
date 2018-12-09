# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime
from contextlib import contextmanager


import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse import find
from sklearn.tree import DecisionTreeRegressor


@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    >>> with measure_time('Heavy computation'):
    >>>     do_heavy_computation()
    'Duration of [Heavy computation]: 0:04:07.765971'

    Parameters
    ----------
    label: str
        The label by which the computation will be referred
    """
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label,
                                        datetime.timedelta(seconds=end-start)))

def load_from_csv(path, delimiter=','):
    """
    Load csv file and return a NumPy array of its data

    Parameters
    ----------
    path: str
        The path to the csv file to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    D: array
        The NumPy array of the data contained in the file
    """
    return pd.read_csv(path, delimiter=delimiter).values.squeeze()

def build_rating_matrix(user_movie_rating_triplets):
    """
    Create the rating matrix from triplets of user and movie and ratings.

    A rating matrix `R` is such that `R[u, m]` is the rating given by user `u`
    for movie `m`. If no such rating exists, `R[u, m] = 0`.

    Parameters
    ----------
    user_movie_rating_triplets: array [n_triplets, 3]
        an array of trpilets: the user id, the movie id, and the corresponding
        rating.
        if `u, m, r = user_movie_rating_triplets[i]` then `R[u, m] = r`

    Return
    ------
    R: sparse csr matrix [n_users, n_movies]
        The rating matrix
    """
    rows = user_movie_rating_triplets[:, 0]
    cols = user_movie_rating_triplets[:, 1]
    training_ratings = user_movie_rating_triplets[:, 2]

    return sparse.coo_matrix((training_ratings, (rows, cols))).tocsr()

def create_learning_matrices(rating_matrix, user_movie_pairs):
    """
    Create the learning matrix `X` from the `rating_matrix`.

    If `u, m = user_movie_pairs[i]`, then X[i] is the feature vector
    corresponding to user `u` and movie `m`. The feature vector is composed
    of `n_users + n_movies` features. The `n_users` first features is the
    `u-th` row of the `rating_matrix`. The `n_movies` last features is the
    `m-th` columns of the `rating_matrix`

    In other words, the feature vector for a pair (user, movie) is the
    concatenation of the rating the given user made for all the movies and
    the rating the given movie receive from all the user.

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
    # Feature for users
    rating_matrix = rating_matrix.tocsr()
    user_features = rating_matrix[user_movie_pairs[:, 0]]

    # Features for movies
    rating_matrix = rating_matrix.tocsc()
    movie_features = rating_matrix[:, user_movie_pairs[:, 1]].transpose()

    X = sparse.hstack((user_features, movie_features))
    return X.tocsr()

def create_learning_matrices2(rating_matrix,user_movie_pairs):
    """create the learning matrice [sn_sample,10] from the rating_matrix, and the user movie pair.
        the feature vector is composed with the total number of rank1 given by the user, total of rank2 given by the user, ...
        the total of rank 1 received by the movie, the total of rank 2 received by the movie, ..."""
    with measure_time('Building learning matrix'):
        #create intermediate matrix
        users_rank = np.zeros((rating_matrix.shape[0],5))
        #print(rating_matrix)
        for userID in range(rating_matrix.shape[0]):
            (dont_care1,dont_care2,user_rank) = sparse.find(rating_matrix[userID,:])
            for rank in user_rank:
                #print(users_rank)
                users_rank[userID,rank-1] += 1
        #print(users_rank)
        #print(rating_matrix.shape[1])

        movies_rank = np.zeros((rating_matrix.shape[1],5))
        for movieID in range(rating_matrix.shape[1]):
            (dont_care1,dont_care2,movie_rank) = sparse.find(rating_matrix[:,movieID].transpose())
            if movieID == 2:
                print(movie_rank.shape)
            for rank in movie_rank:
                movies_rank[movieID,rank-1] += 1
        #create feature
        user_features = users_rank[user_movie_pairs[:,0]]
        #print(user_features.shape)
        movie_features = movies_rank[user_movie_pairs[:,1]]
        #print(movie_features.shape)
        X = np.concatenate((user_features,movie_features),axis=1)
        #print(X.shape)
        return X
def make_submission(y_predict, user_movie_ids, file_name='submission',
                    date=True):
    """
    Write a submission file for the Kaggle platform

    Parameters
    ----------
    y_predict: array [n_predictions]
        The predictions to write in the file. `y_predict[i]` refer to the
        user `user_ids[i]` and movie `movie_ids[i]`
    user_movie_ids: array [n_predictions, 2]
        if `u, m = user_movie_ids[i]` then `y_predict[i]` is the prediction
        for user `u` and movie `m`
    file_name: str or None (default: 'submission')
        The path to the submission file to create (or override). If none is
        provided, a default one will be used. Also note that the file extension
        (.txt) will be appended to the file.
    date: boolean (default: True)
        Whether to append the date in the file name

    Return
    ------
    file_name: path
        The final path to the submission file
    """

    # Naming the file
    if date:
        file_name = '{}_{}'.format(file_name, time.strftime('%d-%m-%Y_%Hh%M'))

    file_name = '{}.txt'.format(file_name)

    # Writing into the file
    with open(file_name, 'w') as handle:
        handle.write('"USER_ID_MOVIE_ID","PREDICTED_RATING"\n')
        for (user_id, movie_id), prediction in zip(user_movie_ids,
                                                 y_predict):

            if np.isnan(prediction):
                raise ValueError('The prediction cannot be NaN')
            line = '{:d}_{:d},{}\n'.format(user_id, movie_id, prediction)
            handle.write(line)
    return file_name


if __name__ == '__main__':
    prefix = 'data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    training_user_movie_pairs = load_from_csv(os.path.join(prefix,
                                                           'data_train.csv'))
    training_labels = load_from_csv(os.path.join(prefix, 'output_train.csv'))

    user_movie_rating_triplets = np.hstack((training_user_movie_pairs,
                                            training_labels.reshape((-1, 1))))

    # Build the learning matrix
    rating_matrix = build_rating_matrix(user_movie_rating_triplets)
    X_ls = create_learning_matrices2(rating_matrix, training_user_movie_pairs)

    # Build the model
    y_ls = training_labels
    start = time.time()
    model = DecisionTreeRegressor()

    with measure_time('Training'):
        print('Training...')
        model.fit(X_ls, y_ls)

    # ------------------------------ Prediction ------------------------------ #
    # Load test data
    test_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_test.csv'))

    # Build the prediction matrix
    X_ts = create_learning_matrices2(rating_matrix, test_user_movie_pairs)

    # Predict
    y_pred = model.predict(X_ts)

    # Making the submission file
    fname = make_submission(y_pred, test_user_movie_pairs, 'toy_example')
    print('Submission file "{}" successfully written'.format(fname))