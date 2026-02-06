import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        # Input error handling
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Input matrix must be a numpy array.")
        if X.ndim != 2:
            raise ValueError(f"Input matrix has incorrect number of dimensions, should be 2D matrix.")
        
        if not isinstance(y, np.ndarray):
            raise TypeError(f"Cluster labels must be a numpy array.")
        if y.shape[0] != X.shape[0]:
            raise ValueError("Number of cluster labels does not match number of input observations.")
        

        num_obs = X.shape[0]
        sil_scores = np.zeros(num_obs)
        labels = np.unique(y)

        if len(labels) <= 1:
            raise ValueError("Silhouette score is undefined for a single cluster.")

        for i in range(num_obs):
            own_cluster = X[y == y[i]]
            other_clusters = [X[y == j] for j in labels if j!=y[i]]

            # a_i: how far, on avg, is point i is from every point in its own cluster
            if len(own_cluster) > 1:
                a_i = np.mean(cdist([X[i]], own_cluster)[0][1:]) # [1:] -> skip self
            else:
                a_i = 0.0

            # b_i: what is the minimum, mean distance between point i and every other cluster
            mean_other_dists = [np.mean(cdist([X[i]], cluster)[0]) for cluster in other_clusters]
            b_i = np.min(mean_other_dists)

            sil_scores[i] = (b_i - a_i) / max(a_i, b_i)
    
        return sil_scores