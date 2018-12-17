# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
from toy_example import measure_time, load_from_csv, build_rating_matrix, make_submission
from learning_set_creation import create_learning_matrices4
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


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
    X_ls = create_learning_matrices4(movies,users,rating_matrix, training_user_movie_pairs)
    # Build the model
    y_ls = training_labels
	#creation of the learning and validation set.

    parameter_space = {
        'max_features' : [0.2,0.4,0.6,0.8,1],
        'n_estimators' : [50,100, 200,300,400, 500]
    }
    print("testing model")
    with measure_time('select model'):
        model = GridSearchCV(RandomForestRegressor(),parameter_space,n_jobs=-1,cv=3)
        model.fit(X_ls,y_ls)
    print("best parameter: ",model.best_params_)

    # ------------------------------ Prediction ------------------------------ #
    # Load test data
    test_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_test.csv'))

    # Build the prediction matrix
    X_ts = create_learning_matrices4(movies,users,rating_matrix, test_user_movie_pairs)

    # Predict
    y_pred = model.predict(X_ts)

    # Making the submission file
    fname = make_submission(y_pred, test_user_movie_pairs, 'random_forest_parameter')
    print('Submission file "{}" successfully written'.format(fname))
