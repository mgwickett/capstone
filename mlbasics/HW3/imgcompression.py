from matplotlib import pyplot as plt
import numpy as np


class ImgCompression(object):
    def __init__(self):
        pass

    def svd(self, X): # [5pts]
        """
        Do SVD. You could use numpy SVD.
        Your function should be able to handle black and white
        images (N*D arrays) as well as color images (N*D*3 arrays)
        In the image compression, we assume that each colum of the image is a feature. Image is the matrix X.
        Args:
            X: N * D array corresponding to an image (N * D * 3 if color image)
        Return:
            U: N * N (*3 for color images)
            S: min(N, D) * 1 (* 3 for color images)
            V: D * D (* 3 for color images)
        """
        n = X.shape[0]
        d = X.shape[1]

        if X.ndim == 2: 
            u, s, v = np.linalg.svd(X, compute_uv = True)
        else: 
            u = np.ones((n, n, 3))
            s = np.ones((min(n, d), 1, 3))
            v = np.ones((d, d, 3))
            u_1, s_1, v_1 = np.linalg.svd(X[:, :, 0])
            u_2, s_2, v_2 = np.linalg.svd(X[:, :, 1])
            u_3, s_3, v_3 = np.linalg.svd(X[:, :, 2])
            
            u = np.dstack((u_1, u_2, u_3))
            s = np.dstack((s_1, s_2, s_3))
            v = np.dstack((v_1, v_2, v_3))
            s = np.reshape(s, (min(n, d), 3)) 


            
        return u, s, v


    def rebuild_svd(self, U, S, V, k): # [5pts]
        """
        Rebuild SVD by k componments.
        Args:
            U: N*N (*3 for color images)
            S: min(N, D)*1 (*3 for color images)
            V: D*D (*3 for color images)
            k: int corresponding to number of components
        Return:
            Xrebuild: N*D array of reconstructed image (N*D*3 if color image)

        Hint: numpy.matmul may be helpful for reconstructing color images
        """

        if U.ndim == 2:
            # fixing s
            u_k = U[:, :k]
            v_k = V[:k, :]
            s_k = S[:k]
            s_fix = np.eye(k, k)
            np.fill_diagonal(s_fix, s_k)
            
            # matrix multiplication!!!! 
            x_rebuild = np.matmul(u_k, s_fix)
            x_rebuild = np.matmul(x_rebuild, v_k)
        else:
            
            u_k = U[:, :k, :]
            s_k = S[:k, :]
            
            # r channel
            u_1 = u_k[:, :, 0]
            s_1 = np.eye(k, k)
            np.fill_diagonal(s_1, s_k[:, 0], wrap = False)
            v_1 = V[:, :, 0]
            # v_1 = v_1.T
            v_1 = v_1[:k, :]
            x_1 = np.matmul(u_1, s_1)
            x_1 = np.matmul(x_1, v_1)
            
            # b channel
            u_2 = u_k[:, :, 1]
            s_2 = np.eye(k, k)
            np.fill_diagonal(s_2, s_k[:, 1], wrap = False)
            v_2 = V[:, :, 1]
            # v_2 = v_2.T
            v_2 = v_2[:k, :]
            x_2 = np.matmul(u_2, s_2)
            x_2 = np.matmul(x_2, v_2)
            
            # g channel
            u_3 = u_k[:, :, 2]
            s_3 = np.eye(k, k)
            np.fill_diagonal(s_3, s_k[:, 2], wrap = False)
            v_3 = V[:, :, 2]
            # v_3 = v_3.T
            v_3 = v_3[:k, :]
            x_3 = np.matmul(u_3, s_3)
            x_3 = np.matmul(x_3, v_3)
            
            x_rebuild = np.dstack((x_1, x_2, x_3))
          
        
        return x_rebuild
            
            
            
    def compression_ratio(self, X, k): # [5pts]
        """
        Compute compression of an image: (num stored values in original)/(num stored values in compressed)
        Args:
            X: N * D array corresponding to an image (N * D * 3 if color image)
            k: int corresponding to number of components
        Return:
            compression_ratio: float of proportion of storage used by compressed image
        """
        n = X.shape[0]
        d = X.shape[1]

        orig = (n * n) + min(n, d) + (d * d)
        compressed = k * (1 + n + d)
        
        return compressed/orig

    def recovered_variance_proportion(self, S, k): # [5pts]
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
           S: min(N, D)*1 (*3 for color images) of singular values for the image
           k: int, rank of approximation
        Return:
           recovered_var: int (array of 3 ints for color image) corresponding to proportion of recovered variance
        """
        # raise NotImplementedError
        # divide the sum of the larget k singular values by the sum of all the singular values 
        s_square = np.square(S)
        if (S.ndim == 1):
            s_k = s_square[:k]
            var = np.sum(s_k) / np.sum(s_square)
        else:
            s_k = s_square[:k, :]
            var = np.sum(s_k, axis = 0) / np.sum(s_square, axis = 0)
            
        return var
        