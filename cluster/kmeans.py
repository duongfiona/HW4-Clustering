import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        # Input error handling
        if not isinstance(k, int):
            raise TypeError(f"k must be an int, got {type(k)}")
        if k <= 0: 
            raise ValueError(f"k must be greater than 0, got k={k}")
        
        if not isinstance(tol, float):
            raise TypeError(f"tolerance must be a float, got {type(tol)}")
        if tol < 0:
            raise ValueError(f"tolerance must be at least 0, got tol={tol}")
        
        if not isinstance(max_iter, int):
            raise TypeError(f"max_iter must be an int, got {type(max_iter)}")
        if max_iter <=0:
            raise ValueError(f"max iterations must be greater than 0, got max_iter={max_iter}")
        
        # Storing attributes
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

        self.centroids = None
        self.fit_data = None


    def _initialize_centroids(self, mat, k):
        """
        Initialize k random datapoints to be starting centroids

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
            k: int
                Number of cluster
        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        random_idxs = np.random.permutation(mat.shape[0])
        centroids = mat[random_idxs[:k]]

        return centroids


    def _assign_clusters(self, mat, centroids):
        """
        Calculates Euclidean distances between each data point in mat and each centroid.
        Assigns each datapoint to nearest centroid. 
        
        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features (m)
            centroids: np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        
        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        # calculate euclidean distance between each datapoint in mat and each centroid
        distances = cdist(mat, centroids)

        # assigns each datapoint in mat to its closest centroid
        assignments = np.argmin(distances, axis=1)
        
        return assignments


    def _update_centroids(self, mat, k, assignments):
        """
        Updates centroids based on cluster assignments for datapoints in mat. 
        Returns a np.array of k centroids. 

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features (m)
            k: int
                Number of clusters
            assignments: np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        
        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        new_centroids = []
        for i in range(k):
            cluster_data = mat[assignments == i]
            new_centroid = cluster_data.mean(axis=0)
            new_centroids.append(new_centroid)
        
        return np.array(new_centroids)


    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        if self.k > mat.shape[0]:
            raise ValueError(f"Number of clusters (k={self.k}), outnumber number of observations (n={mat.shape[0]})")
        if not isinstance(mat, np.ndarray):
            raise TypeError(f"Input matrix must be a numpy array.")
        if mat.ndim != 2:
            raise ValueError(f"Input matrix has incorrect number of dimensions, should be 2D matrix.")

        # Randomly initialize centroids
        centroids = self._initialize_centroids(mat, self.k)

        for _ in range(self.max_iter):
            # Assign datapoints to clusters
            assignments = self._assign_clusters(mat, centroids)

            # Update centroids
            new_centroids = self._update_centroids(mat, self.k, assignments)

            # Check if centroids have stopped updating, within tolerance
            if np.all(np.abs(new_centroids - centroids) < self.tol):
                break

            centroids = new_centroids
        
        self.centroids = centroids
        self.fit_data = mat

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if not isinstance(mat, np.ndarray):
            raise TypeError(f"Input matrix must be a numpy array.")
        if mat.ndim != 2:
            raise ValueError(f"Input matrix has incorrect number of dimensions, should be 2D matrix.")
    

        centroids = self.get_centroids() # if fit has not been run, will throw an error

        if mat.shape[1] != centroids.shape[1]:
            raise ValueError("Input matrix has different number of features than fitted model data")

        return self._assign_clusters(mat, centroids)
    

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        centroids = self.get_centroids()
        assignments = self._assign_clusters(self.fit_data, centroids)

        # Calculate all squared distances between each datapoint and its centroid
        sq_dist = 0.0
        for i, centroid in enumerate(centroids):
            cluster_data = self.fit_data[assignments == i]
            sq_dist += ((cluster_data - centroid)**2).sum()
        
        # Calculate mean squared-error (divide squared distances by num observations)
        num_observations = self.fit_data.shape[0]
        mse = sq_dist / num_observations

        return float(mse)


    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        if self.centroids is None:
            raise ValueError("Cluster centroids are not defined. Model has not been fitted yet.")

        return self.centroids