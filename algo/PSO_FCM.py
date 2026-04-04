import numpy as np
from scipy.spatial.distance import cdist
import time
from .MYFCM import FCM, division_by_zero


class PSO_V_FCM(FCM):
    """
    PSO-V: tối ưu FCM bằng PSO — mỗi particle là tâm cụm V (eq. 11).
    Kế thừa FCM: dùng lại calculate_V, update_membership_matrix, get_labels.
    Chỉ override fit().
    
    Ref: Runkler & Katz, 2006.
    """

    def __init__(self, c: int, m: int = 2, max_iter: int = 50,
                 swarm_size: int = 10, a1: float = 1.5, a2: float = 1.5,
                 delta_y_max: float = 0.3, seed: int = 42):
        super().__init__(c=c, m=m, max_iter=max_iter)
        self.swarm_size = swarm_size
        self.a1 = a1
        self.a2 = a2
        self.delta_y_max = delta_y_max
        self.seed = seed

    def _jfcm(self, V_candidate: np.ndarray, data: np.ndarray) -> float:
        """Tính J_FCM(U,V;X) với V cho trước, U tính từ eq (3)."""
        self.V = V_candidate
        self.U = self.update_membership_matrix(data)  # eq (3) — kế thừa từ FCM
        distance_sq = cdist(data, self.V, metric='sqeuclidean')
        J = np.sum((self.U ** self.m) * distance_sq)
        return J if np.isfinite(J) else np.inf

    def fit(self, data: np.ndarray):
        np.random.seed(self.seed)
        n, p = data.shape
        q = self.c * p  # mỗi particle = (v11,...,v1p,...,vc1,...,vcp)

        x_min, x_max = data.min(axis=0), data.max(axis=0)

        # Khởi tạo swarm
        Y = np.zeros((self.swarm_size, q))
        for s in range(self.swarm_size):
            for i in range(self.c):
                Y[s, i*p:(i+1)*p] = x_min + np.random.rand(p) * (x_max - x_min)

        DY = np.random.uniform(-self.delta_y_max, self.delta_y_max, size=(self.swarm_size, q))

        # Đánh giá ban đầu
        def evaluate(particle):
            return self._jfcm(particle.reshape(self.c, p), data)

        J_vals = np.array([evaluate(Y[s]) for s in range(self.swarm_size)])
        best_idx = np.argmin(J_vals)
        y_star = Y[best_idx].copy()
        J_star = J_vals[best_idx]

        start_time = time.time()

        for t in range(self.max_iter):
            # Winner hiện tại (eq. 8)
            cur_best = np.argmin(J_vals)
            y_hat = Y[cur_best].copy()
            if J_vals[cur_best] < J_star:
                y_star = y_hat.copy()
                J_star = J_vals[cur_best]

            # Cập nhật velocity (eq. 9) và position (eq. 10)
            for k in range(self.swarm_size):
                r1, r2 = np.random.rand(q), np.random.rand(q)
                DY[k] = DY[k] + self.a1 * r1 * (y_hat - Y[k]) + self.a2 * r2 * (y_star - Y[k])
                DY[k] = np.clip(DY[k], -self.delta_y_max, self.delta_y_max)
                Y[k] = Y[k] + DY[k]

            J_vals = np.array([evaluate(Y[s]) for s in range(self.swarm_size)])

        self.process_time = time.time() - start_time

        # Kết quả tốt nhất
        self.V = y_star.reshape(self.c, p)
        self.U = self.update_membership_matrix(data)  # tính U cuối cùng từ V tốt nhất
        labels = np.argmax(self.U, axis=1)
        return self.V, self.U, labels, self.max_iter


class PSO_U_FCM(FCM):
    """
    PSO-U: tối ưu FCM bằng PSO — mỗi particle là ma trận W unbounded (eq. 15),
    chuyển sang U qua sigmoid + normalize (eq. 16).
    Kế thừa FCM: dùng lại calculate_V, get_labels.
    Chỉ override fit().
    
    Ref: Runkler & Katz, 2006.
    """

    def __init__(self, c: int, m: int = 2, max_iter: int = 50,
                 swarm_size: int = 10, a1: float = 1.5, a2: float = 1.5,
                 delta_y_max: float = 0.1, seed: int = 42):
        super().__init__(c=c, m=m, max_iter=max_iter)
        self.swarm_size = swarm_size
        self.a1 = a1
        self.a2 = a2
        self.delta_y_max = delta_y_max
        self.seed = seed

    @staticmethod
    def _W_to_U(W: np.ndarray) -> np.ndarray:
        """W (c×n) → U (n×c) qua sigmoid + normalize (eq. 16)."""
        sig = (1.0 + np.tanh(W)) / 2.0  # (c, n)
        sig = np.fmax(sig, np.finfo(float).eps)
        U = sig / sig.sum(axis=0, keepdims=True)  # normalize theo cột
        return U.T  # → (n, c) cho nhất quán với FCM

    def _jfcm_U(self, U: np.ndarray, data: np.ndarray) -> float:
        """Tính J_FCM(U;X) reformulated (eq. 6). U shape (n, c)."""
        self.U = U
        self.V = self.calculate_V(data)  # eq (4) — kế thừa từ FCM
        distance_sq = cdist(data, self.V, metric='sqeuclidean')
        return np.sum((self.U ** self.m) * distance_sq)

    def fit(self, data: np.ndarray):
        np.random.seed(self.seed)
        n, p = data.shape
        q = self.c * n  # mỗi particle = (w11,...,w1n,...,wc1,...,wcn)

        # Khởi tạo swarm
        Y = np.random.randn(self.swarm_size, q) * 0.5
        DY = np.random.uniform(-self.delta_y_max, self.delta_y_max, size=(self.swarm_size, q))

        def evaluate(particle):
            W = particle.reshape(self.c, n)
            U = self._W_to_U(W)  # (n, c)
            return self._jfcm_U(U, data)

        J_vals = np.array([evaluate(Y[s]) for s in range(self.swarm_size)])
        best_idx = np.argmin(J_vals)
        y_star = Y[best_idx].copy()
        J_star = J_vals[best_idx]

        start_time = time.time()

        for t in range(self.max_iter):
            cur_best = np.argmin(J_vals)
            y_hat = Y[cur_best].copy()
            if J_vals[cur_best] < J_star:
                y_star = y_hat.copy()
                J_star = J_vals[cur_best]

            for k in range(self.swarm_size):
                r1, r2 = np.random.rand(q), np.random.rand(q)
                DY[k] = DY[k] + self.a1 * r1 * (y_hat - Y[k]) + self.a2 * r2 * (y_star - Y[k])
                DY[k] = np.clip(DY[k], -self.delta_y_max, self.delta_y_max)
                Y[k] = Y[k] + DY[k]

            J_vals = np.array([evaluate(Y[s]) for s in range(self.swarm_size)])

        self.process_time = time.time() - start_time

        # Kết quả tốt nhất
        W_best = y_star.reshape(self.c, n)
        self.U = self._W_to_U(W_best)  # (n, c)
        self.V = self.calculate_V(data)
        labels = np.argmax(self.U, axis=1)
        return self.V, self.U, labels, self.max_iter
