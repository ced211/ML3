# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime
from contextlib import contextmanager
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy import sparse
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
    print(training_user_movie_pairs)
    print(training_labels)

    # Build the model
    X_ls = training_user_movie_pairs
    y_ls = training_labels
    X_ls2 , X_vs , y_ls2 , y_vs = train_test_split(X_ls, y_ls,train_size=.8,random_state=20)

    start = time.time()
    model = DecisionTreeRegressor()

    with measure_time('Training'):
        print('Training...')
        model.fit(X_ls2, y_ls2)

    y_pred2 = model.predict(X_vs)
    error = mean_squared_error(y_vs, y_pred2)
    print(error) 

