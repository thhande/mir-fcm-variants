import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ==============================
# siRFCM (Size-insensitive FCM)
# ==============================
class siRFCM:
    def __init__(self, c=3, m=2, p=2, epsilon=1e-5, max_iter=100):
        self.c = c
        self.m = m
        self.p = p
        self.epsilon = epsilon
        self.max_iter = max_iter

    def initialize_U(self, n):
        U = np.random.rand(self.c, n)
        U /= np.sum(U, axis=0)
        return U

    def update_centers(self, X, U):
        um = U ** self.m
        V = (um @ X) / np.sum(um, axis=1, keepdims=True)
        return V

    def compute_Si(self, U):
        n = U.shape[1]
        Si = np.zeros(self.c)
        cluster_assignments = np.argmax(U, axis=0)
        for i in range(self.c):
            Ai = np.where(cluster_assignments == i)[0]
            if len(Ai) == 0:
                continue
            Si[i] = (1 / n) * np.sum([1 + U[i, j] / (n ** self.p) for j in Ai])
        return Si, cluster_assignments

    def update_U(self, X, V, U):
        c, n = U.shape
        new_U = np.zeros_like(U)
        Si, assignments = self.compute_Si(U)
        rho = np.array([1 - Si[assignments[j]] for j in range(n)])

        for j in range(n):
            for i in range(c):
                d_rho_i = -1 / (n ** (self.p + 1)) if i == assignments[j] else 0
                denom = 0
                for k in range(c):
                    d_rho_k = -1 / (n ** (self.p + 1)) if k == assignments[j] else 0
                    dist_i = np.linalg.norm(X[j] - V[i])**2
                    dist_k = np.linalg.norm(X[j] - V[k])**2
                    denom += ((1 - d_rho_i) * dist_i / ((1 - d_rho_k) * dist_k)) ** (1 / (self.m - 1))
                new_U[i, j] = rho[j] * (1 / denom)
        return new_U

    def fit(self, X):
        n = X.shape[0]
        U = self.initialize_U(n)
        V = self.update_centers(X, U)

        for _ in range(self.max_iter):
            V_old = V.copy()
            U = self.update_U(X, V, U)
            V = self.update_centers(X, U)
            if np.linalg.norm(V - V_old) <= self.epsilon:
                break
        return U, V


# ==============================
# nrRFCM (Noise-resistant RFCM)
# ==============================
class nrRFCM:
    def __init__(self, m=2, alpha=4, epsilon=1e-5, max_iter=100):
        self.m = m
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iter = max_iter

    def compute_sij(self, X, V):
        c, n = V.shape[0], X.shape[0]
        sij = np.zeros((c, n))
        for i in range(c):
            for j in range(n):
                denom = sum([(np.linalg.norm(X[j]-V[i])**2 / np.linalg.norm(X[j]-V[k])**2) ** (1/(self.m-1)) for k in range(c)])
                sij[i, j] = 1 / denom
        return sij

    def compute_omega(self, X, V, sij):
        c = V.shape[0]
        omega = np.zeros(c)
        for i in range(c):
            num = np.sum((sij[i]**self.m) * np.linalg.norm(X - V[i], axis=1)**2)
            den = self.alpha * np.sum(sij[i]**self.m)
            omega[i] = num / den if den != 0 else 1e-6
        return omega

    def f(self, dist2, omega2):
        return 1 - np.exp(-dist2 / omega2)

    def update_U(self, X, V, phi, omega):
        c, n = V.shape[0], X.shape[0]
        U = np.zeros((c, n))
        for j in range(n):
            for i in range(c):
                dist_i = np.linalg.norm(X[j] - V[i])**2
                denom = 0
                for k in range(c):
                    dist_k = np.linalg.norm(X[j] - V[k])**2
                    denom += (self.f(dist_i, omega[i]) / self.f(dist_k, omega[k])) ** (1/(self.m-1))
                U[i, j] = phi[j] * (1/denom)
        return U

    def update_centers(self, X, U, V, omega, phi):
        c = V.shape[0]
        new_V = np.zeros_like(V)
        for i in range(c):
            weights = (U[i]**self.m) * phi * self.f(np.linalg.norm(X - V[i], axis=1)**2, omega[i])
            num = np.sum(weights[:, None] * X, axis=0)
            den = np.sum(weights)
            new_V[i] = num / den if den != 0 else V[i]
        return new_V

    def fit(self, X, U_init, V_init):
        U, V = U_init.copy(), V_init.copy()
        n = X.shape[0]
        phi = np.ones(n)  # lấy từ siRFCM, ở đây để đơn giản giữ =1

        for _ in range(self.max_iter):
            V_old = V.copy()
            sij = self.compute_sij(X, V)
            omega = self.compute_omega(X, V, sij)
            U = self.update_U(X, V, phi, omega)
            V = self.update_centers(X, U, V, omega, phi)
            if np.linalg.norm(V - V_old) <= self.epsilon:
                break
        return U, V


# ==============================
# RFCM wrapper
# ==============================
class RFCM:
    def __init__(self, c=3, m=2, p=2, alpha=4, epsilon=1e-5, max_iter=100):
        self.c = c
        self.m = m
        self.p = p
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.si = siRFCM(c, m, p, epsilon, max_iter)
        self.nr = nrRFCM(m, alpha, epsilon, max_iter)

    def fit(self, X):
        U_si, V_si = self.si.fit(X)
        U_nr, V_nr = self.nr.fit(X, U_si, V_si)
        return U_nr, V_nr
df = pd.read_csv("data_iris.csv")

# giả sử file có cột tên Species (nhãn), ta chỉ lấy feature
X = df.drop(columns=["class"]).values  

# scale dữ liệu cho đẹp
X = StandardScaler().fit_transform(X)

# chạy RFCM
rfcm = RFCM(c=3, m=2, p=2, alpha=1)
U, V = rfcm.fit(X)

# gán mỗi điểm vào cụm mạnh nhất
labels = np.argmax(U, axis=0)

rfcm = RFCM(c=3, m=2, p=2, alpha=4, max_iter=100)
U, V, hist_si, hist_nr = rfcm.fit(X)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
hist_nr_2d = [pca.transform(v) for v in hist_nr]

# vẽ dữ liệu
labels = np.argmax(U, axis=0)
plt.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap="viridis", alpha=0.6)

# vẽ đường đi của tâm
for i in range(len(V)):
    path = np.array([h[i] for h in hist_nr_2d])
    plt.plot(path[:,0], path[:,1], marker="x", linestyle="--")
    plt.scatter(path[-1,0], path[-1,1], c="red", marker="X", s=100)

plt.title("RFCM clustering with cluster center paths (PCA projection)")
plt.show()