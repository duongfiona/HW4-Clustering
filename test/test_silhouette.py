import pytest
import numpy as np
from cluster import (Silhouette, make_clusters)
from sklearn.metrics import silhouette_score

random_state = 29

def test_score_to_sklearn():
    """
    Unit test for implementation of silhouette scoring.
    Using sklearn silhouette_score function to assess performance. 
    """
    X, y = make_clusters(seed=random_state)

    sil = Silhouette()
    pred_scores = sil.score(X, y)
    sklearn_score = silhouette_score(X, y) # returns *mean* sil score

    assert abs(pred_scores.mean() - sklearn_score) < 0.1

def test_overlapping_clusters():
    """
    Unit test for calculating silhouette score for overlapping clusters.
    Silhouette scoring should allow for negative scores in cases of bad/
    highly overlapping clusters.
    """
    X, y = make_clusters(
        scale = 5.0, # encourage overlapping clusters
        seed=random_state
    )

    sil = Silhouette()
    scores = sil.score(X, y)

    # some of the observations must have negative sil scores now
    assert np.any(scores < 0) 

    # but sil scores should remain bounded between [-1.0, 1.0]
    assert np.all(scores >= -1.0)
    assert np.all(scores <= 1.0)

def test_score_one_cluster():
    """
    Unit test for handling of silhouette scoring on only one cluster. 

    To calculate a silhouette score, there must be at least two clusters.
    """
    X, y = make_clusters(
        k=1,
        seed=random_state
    )

    with pytest.raises(ValueError):
        sil = Silhouette()
        pred_scores = sil.score(X, y)

def test_lonely_cluster():
    """
    Unit test for handling of silhouette scoring when one of the clusters only
    contains one point.

    In the case that one cluster only contains one point, that cluster's a_i 
    should be handled properly and set to 0. 
    """
    X, y = make_clusters(
        k=1,
        seed=random_state
    )

    y[0] = y[1] + 1 # assigning first point to its own cluster

    sil = Silhouette()
    pred_scores = sil.score(X, y)
    sklearn_score = silhouette_score(X, y)

    assert abs(pred_scores.mean() - sklearn_score) < 0.1

def test_wrong_X_input():
    """
    Unit test for user inputting incorrect type/format for X
    """
    sil = Silhouette()
    y = np.zeros(10)

    # test if X is not a numpy array
    X1 = [[0, 0], [0, 0]]
    with pytest.raises(TypeError):
        sil.score(X1, y)

    # test if X is not a 2D matrix
    X2 = np.zeros(10)
    with pytest.raises(ValueError):
        sil.score(X2, y)


def test_wrong_y_input():
    """
    Unit test for user inputting incorrect type/format for y
    """
    sil = Silhouette()

    num_obs = 10
    num_feat = 3
    X = np.random.rand(num_obs, num_feat)

    # test if y is not a numpy array
    y = num_obs * [1]
    with pytest.raises(TypeError):
        sil.score(X, y)

    # test if y has more labels than there were observations
    bad_y = np.random.rand(num_obs+1)

    with pytest.raises(ValueError):
        sil.score(X, bad_y)