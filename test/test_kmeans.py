import pytest
import numpy as np
from cluster import (KMeans, make_clusters)
from scipy.spatial.distance import cdist

random_state = 29

def test_fit_small():
    """
    Unit test for fitting of small, 2-cluster 2D data
    """
    k = 2
    n_feat = 2

    X, y = make_clusters(
        n=20,
        m=n_feat,
        k=k,
        seed=random_state
    )

    model = KMeans(k)
    model.fit(X)
    pred_centroids = model.get_centroids()

    # assert that centroid df is the correct shape
    assert pred_centroids.shape == (k, n_feat)


    # assert that pred centroids are reasonably close to real centroids
    # (preds are closer to real centroids than true clusters are to one another)
    true_centroids = []
    for i in range(k):
        cluster_data = X[y == i]
        centroid = cluster_data.mean(axis=0)
        true_centroids.append(centroid)

    distances = cdist(pred_centroids, true_centroids)

    # calculating mean distance between true cluster centers
    inter_centroid_dists = cdist(true_centroids, true_centroids)
    np.fill_diagonal(inter_centroid_dists, np.nan)
    mean_inter_cluster = np.nanmean(inter_centroid_dists)/2

    distances = cdist(pred_centroids, true_centroids)

    assert np.all(np.min(distances, axis=1) < mean_inter_cluster)


    # assert that final mse is reasonable (below baseline variance in X)
    assert model.get_error() < np.mean(np.var(X, axis=0))

def test_fit_large():
    """
    Unit test for fitting of larger, 20-cluster 3D data
    """
    k = 20
    n_feat = 3

    X, y = make_clusters(
        n=200,
        m=n_feat,
        k=k,
        seed=random_state
    )

    model = KMeans(k)
    model.fit(X)
    pred_centroids = model.get_centroids()

    # assert that centroid df is the correct shape
    assert pred_centroids.shape == (k, n_feat)


    # assert that pred centroids are reasonably close to real centroids
    # (preds are closer to real centroids than true clusters are to one another)
    true_centroids = []
    for i in range(k):
        cluster_data = X[y == i]
        centroid = cluster_data.mean(axis=0)
        true_centroids.append(centroid)

    # calculating mean distance between true cluster centers
    inter_centroid_dists = cdist(true_centroids, true_centroids)
    np.fill_diagonal(inter_centroid_dists, np.nan)
    mean_inter_cluster = np.nanmean(inter_centroid_dists)/2

    distances = cdist(pred_centroids, true_centroids)

    assert np.all(np.min(distances, axis=1) < mean_inter_cluster)


    # assert that final mse is reasonable (below baseline variance in X)
    assert model.get_error() < np.mean(np.var(X, axis=0))

def test_invalid_k():
    """
    Unit test for user inputting incorrect type/value for k
    """
    k1 = "clearly not an integer"
    k2 = 0 # should have at least one cluster

    with pytest.raises(TypeError):
        model = KMeans(k1)
    
    with pytest.raises(ValueError):
        model = KMeans(k2)

def test_invalid_tol():
    """
    Unit test for user inputting incorrect type/value for tolerance
    """
    k = 3
    tol1 = "clearly not a float"
    tol2 = -1.0 # tolerance should be at least 0.0

    with pytest.raises(TypeError):
        model = KMeans(k=k, tol=tol1)
    
    with pytest.raises(ValueError):
        model = KMeans(k=k, tol=tol2)


def test_invalid_maxiter():
    """
    Unit test for user inputting incorrect type/value for max iterations
    """
    k = 3
    max_iter1 = "clearly not an integer"
    max_iter2 = -1000

    with pytest.raises(TypeError):
        model = KMeans(k=k, max_iter=max_iter1)
    
    with pytest.raises(ValueError):
        model = KMeans(k=k, max_iter=max_iter2)

def test_too_many_k():
    """
    Unit test for trying to create more clusters than there
    are observations
    """
    k = 4

    X, _ = make_clusters(
        n=k,
        m=3,
        k=k,
        seed=random_state
    )
    # deleting one obs in X, such that now n=k-1
    X_bad = np.delete(X, 0, axis=0)

    with pytest.raises(ValueError):
        model = KMeans(k)
        model.fit(X_bad)

def test_predict_valid_output():
    """
    Unit test to check format of predict output
    """
    k = 4
    X, _ = make_clusters(
        k=k,
        seed=random_state
    )

    model = KMeans(k)
    model.fit(X)

    pred_assignments = model.predict(X)

    # each observation in the new matrix should have a prediction
    assert pred_assignments.shape == (X.shape[0],)

    # label assignments should range from cluster 0 to k
    assert pred_assignments.min() >= 0
    assert pred_assignments.max() < k

def test_invalid_predict_mat():
    """
    Unit test to handling of invalid prediction matrix input
    """
    k = 2
    X, _ = make_clusters(k=k)

    model = KMeans(k)
    model.fit(X)

    # array should be 2 dimensional (observations by features)
    bad_pred = np.zeros((3, 3, 3))
    with pytest.raises(ValueError):
        model.predict(bad_pred)


def test_predict_feature_mismatch():
    """
    Unit test for mismatch in number of features in new prediction matrix. 
    Input prediction matrix should have the same number of features as 
    the data that the clusters were fit on. 
    """
    k = 4
    n_feat = 2

    X, _ = make_clusters(
        m=n_feat,
        k=k,
        seed=random_state
    )

    model = KMeans(k)
    model.fit(X)

    # creating a pred mat with diff number of features
    X_bad, _ = make_clusters(
        m=n_feat+1,
        k=k,
        seed=random_state
    )

    with pytest.raises(ValueError):
        model.predict(X_bad)


def test_forget_fit():
    """
    Unit test to check handling of running predict function
    before running fit function
    """
    k = 10
    X, _ = make_clusters(k=k)
    model = KMeans(k)

    with pytest.raises(ValueError):
        model.predict(X)