import numpy as np

class PCA(object):

    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X):
        """
        Decompose dataset into principal components.
        You may use your SVD function from the previous part in your implementation or numpy.linalg.svd function.

        Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
        corresponding values from PCA.

        Args:
            X: N*D array corresponding to a dataset
        Return:
            None
        """
        X = X - np.mean(X, axis = 0)
        self.U, self.S, self.V = np.linalg.svd(X, full_matrices = False, compute_uv = True)

    def transform(self, data, K=2):
        """
        Transform data to reduce the number of features such that final data has given number of columns

        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: N*D array corresponding to a dataset
            K: Int value for number of columns to be kept
        Return:
            X_new: N*K array corresponding to data obtained by applying PCA on data
        """
        
        v_k = self.get_V().T
        v_k = v_k[:, :K]
        x_new = np.matmul(data, v_k)
        return x_new
        

    def transform_rv(self, data, retained_variance=0.99):
        """
        Transform data to reduce the number of features such that a given variance is retained

        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: N*D array corresponding to a dataset
            retained_variance: Float value for amount of variance to be retained
        Return:
            X_new: N*K array corresponding to data obtained by applying PCA on data
        """

        # raise NotImplementedError
        s = np.square(self.S)
        cs = np.cumsum(s)
        var = cs/np.sum(s)
        ks = np.where(var >= retained_variance)
        k = ks[0]
        k_2 = k[0]
        x_new = self.transform(data, k_2 + 1)
        return x_new
        

    def get_V(self):
        """ Getter function for value of V """

        return self.V