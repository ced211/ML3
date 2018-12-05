def create_user_data(user_features):
    """
    Return
	user_data: array[age,sex,work]
	"""
    return np.hstack((user_features[:,1].reshape((-1, 1)), user_features[:,2].reshape((-1, 1)), user_features[:,3].reshape((-1, 1))))
def create_movie_data(movie_features):
    """
    Return
	movie_data: array[unknown,Action,Adventure,Animation,Children,Comedy,Crime,Documentary,Drama,Fantasy,Film-Noir,Horror,Musical,Mystery,Romance,Sci-Fi,Thriller,War,Western]
	"""
    return np.hstack((movie_features[:,5+i].reshape((-1, 1)) for i in range(0,18)))

def create_learning_matrices2(rating_matrix, user_movie_pairs, user_data, movie_data):
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
    # print(user_features)
    
    # Features for movies
    rating_matrix = rating_matrix.tocsc()
    movie_features = rating_matrix[:, user_movie_pairs[:, 1]].transpose()

    X = sparse.hstack((user_features, movie_features))
    row = X.nonzero()[0]
    col = X.nonzero()[1]
    X = X.tocsr()
    nb_movie = len(movie_data)
    for x in range(0,len(row)-1):
        r = row[x]
        user = user_movie_pairs[r, 0]-1
        movie = user_movie_pairs[r, 1]-1
        c = col[x]
        if(c > nb_movie):
            user2 = c - nb_movie - 2
            age = user_data[user,0]
            sex = user_data[user,1]
            work = user_data[user,2]
            age2 = user_data[user2,0]
            sex2 = user_data[user,1]
            work2 = user_data[user2,2]			
            if((age2<70/100*age or age2<130/100*age) and work != work2 and sex2 != sex):
                print(user,movie,user2)
                X[row[x],col[x]] = 0
        else:
            movie2 = c-1
            if(any([(movie_data[movie,i] ==1 and movie_data[movie2,i] == 1) for i in range(0,18) ])):
                pass
            else:
                X[row[x],col[x]] = 0
    return X