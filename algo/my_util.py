import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def align_clusters(centroids, membership_matrix, standard_centroid):
    dist = cdist(centroids, standard_centroid)
    row_ind, col_ind = linear_sum_assignment(dist)
    sort_order = np.argsort(col_ind)
    
    # sắp xếp centroid theo row_ind rồi theo sort_order
    aligned_centroids = centroids[row_ind][sort_order]
    aligned_membership = membership_matrix[:, row_ind][:, sort_order]