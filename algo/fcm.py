import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

NDArray = np.ndarray
class FCM:
    def __init__(self, n_clusters=3, m=2.0, max_iter=100, error=1e-5, random_state=42):  
        # Parameters:
        # - n_clusters: số cụm
        # - m: hệ số mờ (m > 1), càng lớn thì membership càng "mềm"
        # - max_iter: số lần lặp tối đa
        # - error: ngưỡng hội tụ
        # - random_state: seed để tái lập kết quả
      
        self.c = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
        self.random_state = random_state
        self.V = None   # centroids
        self.U = None           # membership matrix
        self.process_time = 0
        self.step = 0

    def init_membership_matrix(self, n):
        """inititalize random membership matrix U (n x c)."""
        np.random.seed(self.random_state)
        U = np.random.rand(n, self.c)
        # Normalize so that the sum of columns by row is equal to 1
        U = U / np.sum(U, axis=1, keepdims=True)
        return U

    def calculate_centroids(self, X:NDArray)->NDArray:
        # """calculate centroids based on membership matrix."""
        X = np.array(X)
        um = self.U ** self.m  # (n x c)
        # centroid: sum(u_ij^m * xj) / sum(u_ij^m)
        centroids = (um.T @ X) / np.sum(um.T, axis=1, keepdims=True)
        return centroids

    def calculate_dis(self, X: NDArray, centroids: NDArray) -> NDArray:
        return cdist(X, centroids, metric='euclidean')

    def update_member(self, X):
      dist = self.calculate_dis(X, self.V)
      # prevent divide by zero
      dist = np.fmax(dist, np.finfo(np.float64).eps)

      pow_exp = 2 / (self.m - 1)

      n, c = dist.shape
      U_new = np.zeros((n, c))

      for j in range(c):
          # d_ij / d_ik 
          ratio = (dist[:, [j]] / dist) ** pow_exp   # shape (n, c)
          U_new[:, j] = 1.0 / np.sum(ratio, axis=1)  # shape (n,)
      
      return U_new

    def fit(self, X):
    
        X = np.array(X)
        n_samples = X.shape[0]
        self.U = self.init_membership_matrix(n_samples)#iniialize centroids and membership matrix

        step = 0
        for _ in range(self.max_iter):#loop through each iter
            U_old = self.U.copy()

            self.V = self.calculate_centroids(X)#step 2: calculate centroids
            self.U = self.update_member(X)#step 3: update membership matrix

            # convergence check, if not, repeat step 2 and 3
            if np.linalg.norm(self.U - U_old) < self.error:
                break 
            step += 1
        self.step = step
        return step

    def get_labels(self):
        """get the label."""
        return np.argmax(self.U, axis=1)

    def get_centroids(self):
        """return centroids."""
        return self.V
