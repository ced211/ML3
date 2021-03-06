# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPRegressor
from toy_example import measure_time, load_from_csv, build_rating_matrix, make_submission
from learning_set_creation import create_learning_matrices2
import numpy as np
import random
from sklearn.metrics import mean_squared_error


def y_check(y):
    for i in range(len(y)):
        if y[i] > 5:
            y[i] = 5
        if y[i] < 0:
            y[i] = 0
    return y


add_noise = lambda x: x + random.uniform(-x/10, x/10)
v_noise = np.vectorize(add_noise)

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
    X = create_learning_matrices2(rating_matrix, training_user_movie_pairs)

    # Build the model
    y = training_labels
    X_ls2, X_vs, y_ls2, y_vs = train_test_split(X, y, train_size=.8, random_state=20)
    # Creation of the learning and validation set.
    X_untouch = X_ls2.copy()
    scaler = StandardScaler()
    start = time.time()
    models = []
    n_estimator = 50
    y_pred = np.zeros((n_estimator,y_vs.shape[0]))
    with measure_time("Bagging training"):
        for i in range(n_estimator):
            # Training
            model = MLPRegressor(hidden_layer_sizes=(100, 50, 50), random_state=i)
            X_ls2 = v_noise(X_ls2)
            scaler.fit(X_ls2)  
            X_ls2=scaler.transform(X_ls2)
            model.fit(X_ls2, y_ls2)
            models.append((model, scaler))
            X_t = scaler.transform(X_vs, copy=True)
            y_pred[i, :] = model.predict(X_t)
            print(model.predict(X_t))          
            X_ls2 = X_untouch.copy()

    means = np.mean(y_pred, axis=0)
    print(mean_squared_error(y_vs, means))
        
    # ------------------------------ Prediction ------------------------------ #
    # Load test data
    test_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_test.csv'))

    # # Build the prediction matrix
    X_ts = create_learning_matrices2(rating_matrix, test_user_movie_pairs)
    X_ts = scaler.transform(X_ts)

    # Predict
    print(X_ts.shape)
    y_pred = np.zeros((n_estimator, X_ts.shape[0]))
    for i in range(n_estimator):
        y_pred[i, :] = models[i][0].predict(models[i][1].transform(X_ts))
    y_pred = np.mean(y_pred, axis=0)
    print(y_pred)

    y_pred = y_check(y_pred)
    # Making the submission file
    fname = make_submission(y_pred, test_user_movie_pairs, 'NLP_bagging' )
    print('Submission file "{}" successfully written'.format(fname))
