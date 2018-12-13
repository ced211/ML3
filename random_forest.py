# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from toy_example import measure_time, load_from_csv, build_rating_matrix, create_learning_matrices4, make_submission
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def y_round(y):
    for i in range(len(y)):
        y[i] = round(y[i])
    return y

if __name__ == '__main__':
    prefix = 'data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    training_user_movie_pairs = load_from_csv(os.path.join(prefix,
                                                           'data_train.csv'))
    training_labels = load_from_csv(os.path.join(prefix, 'output_train.csv'))

    user_movie_rating_triplets = np.hstack((training_user_movie_pairs,
                                            training_labels.reshape((-1, 1))))

    users = load_from_csv(os.path.join(prefix, 'data_user.csv'))
    movies = load_from_csv(os.path.join(prefix, 'data_movie.csv'))
    # Build the learning matrix
    rating_matrix = build_rating_matrix(user_movie_rating_triplets)
    X_ls = create_learning_matrices4(movies,users,rating_matrix, training_user_movie_pairs)
    # Build the model
    y_ls = training_labels
    #creation of the learning and validation set.
    X_ls2 , X_vs , y_ls2 , y_vs = train_test_split(X_ls, y_ls,train_size=.8,random_state=20)
    start = time.time()
    model = RandomForestRegressor(n_estimators=400, max_depth=None,max_features="log2") 
    with measure_time('Training'):
        print('Training...')
        model.fit(X_ls2, y_ls2)
    # --------------------------------- Score -------------------------------- #
    y_pred2 = model.predict(X_vs)
    score = mean_squared_error(y_vs, y_pred2)
    print(score)
    with measure_time('Training'):
        print('Training...')
        model.fit(X_ls, y_ls)
    # ------------------------------ Prediction ------------------------------ #
    # Load test data
    test_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_test.csv'))

    # Build the prediction matrix
    X_ts = create_learning_matrices4(movies,users,rating_matrix, test_user_movie_pairs)

    # Predict
    y_pred = model.predict(X_ts)
    y_pred = y_round(y_pred)
    # Making the submission file
    fname = make_submission(y_pred, test_user_movie_pairs, 'random_forest_LM4')
    print('Submission file "{}" successfully written'.format(fname))
