import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from DataHelper import *
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from EvaluationHelper import *

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

if __name__ == "__main__":
    MyData = LoadMovieLens100k()
    n_users = MyData.user_id.unique().shape[0]
    n_items = MyData.item_id.unique().shape[0]
    print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))
    train_data, test_data = train_test_split(MyData, test_size=0.25)
    # Create two user-item matrices, one for training and another for testing
    train_data_matrix = np.zeros((n_users, n_items))
    for line in train_data.itertuples():
        train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

    test_data_matrix = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

    user_similarity = cosine_similarity(train_data_matrix)
    item_similarity = cosine_similarity(train_data_matrix.T)

    # get SVD components from train matrix. Choose k.
    u, s, vt = svds(train_data_matrix, k=20)
    s_diag_matrix = np.diag(s)
    X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
    print('User-based CF MSE: ' + str(RMSE(X_pred, test_data_matrix)))
