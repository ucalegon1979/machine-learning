class ojaPCA:
    """
    My class to perform streaming PCA using Oja's algorithm
    """
    components_ = []
    eigvals_ = []

    def deflate(self, X, w):
        assert len(X.shape) == 2, "X matrix is not 2D"
        assert len(w.shape) == 1, "w eigenvector is not 1D"
        assert w.shape[0] == X.shape[1], "eigenvector is not right"
        return X - X.dot(np.outer(w, w))

    def ojaPCA(self, X):
        """
        :param X:           Any old data matrix, filled with real values    
        :return: w:         The estimated eigenvector corresponding to the maximum eigenvalue of (1/N)*X^TX, 
                            which is proportional to the empirical sample covariance matrix of the dataset X.
                            This represents the direction in which the maximum variance in the data is found.
        :return eig_vals:   The estimated value of the maximum eigenvalue at each iteration. The eigenvalue 
                            represents the variance of the data projected along the "direction" designated by 
                            the eigenvector 
        """

        # The maximum number of times we will pass over the whole X dataset.
        max_iters = 100

        # Parameter in the denominator of the step size
        t_0 = 1

        # Learning rate, and parameter in the numerator of the step size
        eta = 0.001

        # How many of the final eigenvectors to take the average of
        to_average = 10

        X = X - np.mean(X, axis=0)  # always center the data, subtract the mean

        # Initial random eigenvector, pointing in a random direction, a vector of real values,
        # one for each dimension of the X.
        w_init = np.random.randn(np.size(X, 1))  # Generate a starting point
        w_init /= np.linalg.norm(w_init, axis=0)

        t = 0
        w = w_init
        n, d = X.shape  # the number of observations and dimensions in our dataset

        # The estimated variance of the data along the direction of the eigenvector wobbling at each iteration
        eig_vals = np.zeros(max_iters)

        # Create a matrix to store the last several eigenvectors to average, each of length d.
        eig_vectors = np.zeros((to_average, d))

        for iter in range(0, max_iters):
            # Shuffle the rows of the data after each iter
            np.random.shuffle(X)
            for i in range(0, n):
                # Note it's faster not to compute the matrix ZZ^T
                w = w + (eta / (t + t_0)) * np.dot(X[i,], np.dot(X[i,].T, w))
                w = w / np.linalg.norm(w)
                t += 1
            eig_vals[iter] = w.dot(X.T).dot(X).dot(w) / n

            # store the last ten iterations of the eigenvector we are attempting to estimate
            if max_iters - iter <= to_average:
                eig_vectors[to_average - (max_iters - iter)] = w

        w = eig_vectors.mean(axis=0)

        return w, eig_vals

    def fit(self, X):
        # The number of principal components to store.
        # There can never be more principal components than dimensions of our data.
        # The point is dimension reduction anyway.
        num_components = X.shape[1]  # assuming columns are the dimensions

        self.components_ = np.zeros((num_components, num_components))
        self.eigvals_ = np.zeros(num_components)

        for i in range(0, num_components):
            ev, eigvals = self.ojaPCA(X)  # find the 1st or next component
            self.components_[i] = cp.deepcopy(ev)  # save off the component eigenvector
            self.eigvals_[i] = eigvals[-1]  # get last eigenvalue
            X = self.deflate(X, self.components_[i])  # deflate to find the next component
