# ! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from toy_example import measure_time, load_from_csv, build_rating_matrix, create_learning_matrices2, make_submission
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def predict(user_movie_pair,rating_matrix):
    """prediction is simply the average of the rank given by the user to the movies 
    and all the rank received by the movie from users"""
    learning_matrix = create_learning_matrices2(rating_matrix,user_movie_pair)
    nb_rank = np.sum(learning_matrix,axis=1)
    for i in range(5):
        learning_matrix[:,i] *= (i+1)
        learning_matrix[:,i+4] *= (i+1)
    total = np.sum(learning_matrix,axis=1)
    return total/nb_rank

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
    test_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_test.csv'))

    y_pred = predict(test_user_movie_pairs,rating_matrix)
    print(min(y_pred))
    print(max(y_pred))
    # Making the submission file
    fname = make_submission(y_pred, test_user_movie_pairs, 'average')
    print('Submission file "{}" successfully written'.format(fname))

