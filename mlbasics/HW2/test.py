from kmeans import *
from gmm import *
import numpy as np

# Helper function for checking the implementation of pairwise_distance fucntion. Please DO NOT change this function
# TEST CASE


centers = np.array([[1, 1], [2,2], [3,3]])
cluster_idx = np.array([0,1,2,0,1])
points = np.array([[0,0], [2,1], [4,3], [1,0], [1,2]])

# KMeans().find_optimal_num_clusters(points)
# a = inter_cluster_dist(1, points, cluster_idx)
# print(a)

logit = np.array([[0,1, 3], [2,3, 4], [4,5, 6], [6,7, 8], [8,9, 8]])
pi, mu, sigma = GMM(logit, 4)._init_components()
gmm = GMM(logit, 4)
# GMM(logit, 3).softmax(logit)
# GMM(logit, 3).logsumexp(logit)
# GMM(logit, 3)._init_components()
# GMM(logit, 3).normalPDF(logit, mu, sigma)
# GMM(logit, 0)._ll_joint(pi, mu, sigma)
gamma = gmm._E_step(pi, mu, sigma)
gmm._M_step(gamma)
# print(KMeans()._get_loss(centers, cluster_idx, points))


# # test kmeans
# np.random.seed(1)
# points = np.random.randn(100, 2)

# cluster_idx2, centers2, loss2 = KMeans()(points, 2)
# cluster_idx5, centers5, loss5 = KMeans()(points, 5)

# print("*** Expected Answer ***")
# print("""==centers2==
# [[-0.23265213  0.66957783]
#  [ 0.61791745 -0.59496966]]
# ==centers5==
# [[ 0.94945532 -1.42382563]
#  [ 0.64137518  0.09830081]
#  [-0.51672295 -0.35410285]
#  [-0.07747868  1.08896449]
#  [ 1.93010934  0.48561944]]
# ==loss2==
# 105.06622377653986
# ==loss5==
# 53.0865571656247""")

# print("\n*** My Answer ***")
# print("==centers2==")
# print(centers2)
# print("==centers5==")
# print(centers5)
# print("==loss2==")
# print(loss2)
# print("==loss5==")