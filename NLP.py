# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPRegressor
from toy_example import measure_time, load_from_csv, build_rating_matrix, make_submission
from learning_set_creation import create_learning_matrices4
import numpy as np
from sklearn.metrics import mean_squared_error


def y_check(y):
    for i in range(len(y)):
        if y[i] > 5:
            y[i] = 5
        if y[i] < 0:
            y[i] = 0
    return y


if __name__ == '__main__':
    prefix = 'data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    training_user_movie_pairs = load_from_csv(os.path.join(prefix,
                                                           'data_train.csv'))
    training_labels = load_from_csv(os.path.join(prefix, 'output_train.csv'))
    users = load_from_csv(os.path.join(prefix, 'data_user.csv'))
    movies = load_from_csv(os.path.join(prefix, 'data_movie.csv'))
    with measure_time('Build matrices'):
        print("Building matrices ...")
        # Build the learning matrix
        user_movie_rating_triplets = np.hstack((training_user_movie_pairs, training_labels.reshape((-1, 1))))
        rating_matrix = build_rating_matrix(user_movie_rating_triplets)
        X_ls = create_learning_matrices4(movies, users, rating_matrix, training_user_movie_pairs)
    # Build the model
    y_ls = training_labels
    # Creation of the learning and validation set.
    X_ls2, X_vs, y_ls2, y_vs = train_test_split(X_ls, y_ls, train_size=.8, random_state=20)
    scaler = StandardScaler()
    scaler.fit(X_ls2) 
    X_ls2 = scaler.transform(X_ls2)
    X_vs = scaler.transform(X_vs)
    start = time.time()
    score = []
    parameters = [(100, 50)]
    for parameter in parameters:
        model = MLPRegressor(hidden_layer_sizes=parameter, random_state = 20)
        with measure_time('Training'):
            print('Training...')
            model.fit(X_ls2, y_ls2)
    # --------------------------------- Score -------------------------------- #
        with measure_time('predict'):
            print('predict...')
            y_pred2 = model.predict(X_vs)
            y_pred2 = y_check(y_pred2)
        score.append(mean_squared_error(y_vs, y_pred2))
        print("score", score)
    best = np.argmin(score)
    print(best)
    with measure_time('Training'):
        print("training....")
        model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=20)
        scaler.fit(X_ls)
        X_ls = scaler.transform(X_ls)
        model.fit(X_ls, y_ls)
    # ------------------------------ Prediction ------------------------------ #
    # # Load test data
    test_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_test.csv'))

    # Build the prediction matrix
    X_ts = create_learning_matrices4(movies,users,rating_matrix,test_user_movie_pairs)
    X_ts = scaler.transform(X_ts)
    # Predict
    y_pred = model.predict(X_ts)    
    y_pred = y_check(y_pred)
    # Making the submission file
    fname = make_submission(y_pred, test_user_movie_pairs, 'NLP' + "age_sexe_metier")
    print('Submission file "{}" successfully written'.format(fname))

