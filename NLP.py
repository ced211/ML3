# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPRegressor
from toy_example import measure_time, load_from_csv, build_rating_matrix, create_learning_matrices2,create_learning_matrices3, make_submission
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def learning_mat4(movie,user,rating_matrix,user_movie_pair):
    M = np.zeros((max(user)+1,19,5))
    feature = np.zeros((user_movie_pair.shape[0],5))
    for u,m in user_movie_pair:
        if rating_matrix[u,m] != 0:
            M[user[u-1],movie[m-1],rating_matrix[u,m]-1] += 1
    i = 0
    for u,m in user_movie_pair:
        feature[i] = M[user[u-1],movie[m-1],:]
        i += 1
    print(feature)
    return np.concatenate((create_learning_matrices2(rating_matrix,user_movie_pair),feature),axis=1)

def create_movie_matrix(movie):
    movie = movie[:,5:]
    total = np.sum(movie,axis=0)
    permutation = np.flip(np.argsort(total),axis=0)
    movie = movie[:,permutation]
    output = movie[:,0]
    for j in range(movie.shape[0]):
        i = 0
        while movie[j][i] == 0:
            i += 1
        output[j] = i
    return output

def create_user(user):
    #catgorie depend on occupation and gender
    user = user[:,(1,2,3)]
    categorieID = 0
    categorie = {}
    output = user[:,0]
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
    print(categorie)
    print(user_per_categorie)
    return output

def y_check(y):
    for i in range(len(y)):
        #y[i] = round(y[i])
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
        users = create_user(users)
        movies = create_movie_matrix(movies)
        user_movie_rating_triplets = np.hstack((training_user_movie_pairs,training_labels.reshape((-1, 1))))
        rating_matrix = build_rating_matrix(user_movie_rating_triplets)
        X_ls = learning_mat4(movies,users,rating_matrix,training_user_movie_pairs)
        print(X_ls.shape)

    # Build the model
    y_ls = training_labels
	#creation of the learning and validation set.
    X_ls2 , X_vs , y_ls2 , y_vs = train_test_split(X_ls, y_ls,train_size=.8,random_state=20)
    scaler = StandardScaler()
    scaler.fit(X_ls2) 
    X_ls2 = scaler.transform(X_ls2)
    X_vs = scaler.transform(X_vs)
    start = time.time()
    score = []
    parameters = [(100,100,100)]
    for parameter in parameters:
        model = MLPRegressor(hidden_layer_sizes=parameter,random_state = 20,early_stopping = True)
        with measure_time('Training'):
            print('Training...')
            model.fit(X_ls2, y_ls2)
    # --------------------------------- Score -------------------------------- #
        with measure_time('predict'):
            print('predict...')
            y_pred2 = model.predict(X_vs)
            y_pred2 = y_check(y_pred2)
        score.append(mean_squared_error(y_vs, y_pred2))
        print("score",score)
    best = np.argmin(score)
    print(best)
    with measure_time('Training'):
        print("training....")
        model = MLPRegressor(hidden_layer_sizes=(100,100,100),random_state = 20)
        scaler.fit(X_ls)
        X_ls = scaler.transform(X_ls)
        model.fit(X_ls, y_ls)
    # ------------------------------ Prediction ------------------------------ #
    # # Load test data
    test_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_test.csv'))

    # Build the prediction matrix
    X_ts = learning_mat4(movies,users,rating_matrix,test_user_movie_pairs)
    X_ts = scaler.transform(X_ts)
    # Predict
    y_pred = model.predict(X_ts)    
    print(y_pred)

    print(max(y_pred))
    y_pred = y_check(y_pred)
    print(max(y_pred))
    # Making the submission file
    fname = make_submission(y_pred, test_user_movie_pairs, 'NLP' + "age_sexe_metier")
    print('Submission file "{}" successfully written'.format(fname))