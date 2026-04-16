import numpy as np
import time
from scipy.spatial.distance import cdist
from .MYFCM import FCM, division_by_zero


# ================================================================
# Lớp cơ sở: ssPSO (Semi-supervised PSO)
# Dựa trên: Lai D.T.C., Miyakawa M., Sato Y. (2019)
# "Semi-supervised data clustering using particle swarm optimisation"
# Soft Computing, Springer. https://doi.org/10.1007/s00500-019-04114-z
# ================================================================
class SSPSO(FCM):
    """
    Lớp cơ sở cho các biến thể ssPSO phân cụm bán giám sát.

    Mỗi particle trong bầy đại diện một ma trận tâm cụm có shape (C, D),
    đồng thời giữ ma trận thành viên U riêng shape (N, C).

    Hỗ trợ 2 chế độ giám sát:
      - 'ssFCM' : giám sát xuyên suốt (ssFCM-PSO) - dùng eq (3) Pedrycz-Waletzky
      - 'IS'    : chỉ giám sát tại khởi tạo (IS-PSO) - sau đó dùng FCM thuần
    """

    def __init__(self,
                 c: int,
                 m: int = 2,
                 max_iter: int = 20,
                 eps: float = 1e-5,
                 swarm_size: int = 20,
                 alpha: float = None,
                 semi_mode: str = 'ssFCM',
                 seed: int = 42):
        super().__init__(c=c, m=m, max_iter=max_iter, eps=eps)
        self.swarm_size = swarm_size
        self.alpha = alpha                # Hệ số cân bằng eq (1); None -> tự tính N/M
        self.semi_mode = semi_mode        # 'ssFCM' hoặc 'IS'
        self.seed = seed

        # --- Trạng thái bầy ---
        self.positions = None             # (P, C, D) - vị trí = tâm cụm
        self.U_particles = None           # (P, N, C) - ma trận U của từng particle
        self.fitness = None               # (P,)

        self.pbest_pos = None             # (P, C, D)
        self.pbest_U = None               # (P, N, C)
        self.pbest_fit = None             # (P,)

        self.gbest_pos = None             # (C, D)
        self.gbest_U = None               # (N, C)
        self.gbest_fit = np.inf

        # --- Thông tin giám sát ---
        self.F = None                     # (N, C) ma trận nhãn one-hot
        self.b = None                     # (N,)   1 nếu có nhãn, 0 nếu không

        self._rng = None

    # ----------------------------------------------------------------
    # 1. Chuẩn bị dữ liệu giám sát
    # ----------------------------------------------------------------
    def _init_semi_info(self, labels: np.ndarray, n: int) -> None:
        """Khởi tạo F và b từ mảng labels (-1 = không có nhãn)."""
        F = np.zeros((n, self.c))
        b = np.zeros(n, dtype=int)
        for i, lab in enumerate(labels):
            if lab != -1:
                F[i, int(lab)] = 1.0
                b[i] = 1
        self.F = F
        self.b = b

        # alpha mặc định = N/M theo khuyến nghị Pedrycz-Waletzky
        if self.alpha is None:
            M = max(int(b.sum()), 1)
            self.alpha = n / M

    # ----------------------------------------------------------------
    # 2. Công thức Pedrycz-Waletzky: eq (2) tâm cụm, eq (3) thành viên
    # ----------------------------------------------------------------
    def _calc_centers(self, data: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Cập nhật tâm cụm theo công thức FCM (eq 2)."""
        um = U ** self.m
        return (um.T @ data) / division_by_zero(np.sum(um.T, axis=1, keepdims=True))

    def _calc_membership_ss(self, data: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Cập nhật ma trận thành viên bán giám sát theo eq (3):

          u_ij = 1/(1+α) * { [1 + α(1 - b_j Σ_l f_lj)] / Σ_l (d_ij/d_lj)^(2/(m-1))
                             + α f_ij b_j }

        - Với điểm KHÔNG nhãn: b_j = 0 -> công thức rút về FCM chuẩn
        - Với điểm CÓ nhãn   : giá trị được kéo về phía f_ij
        """
        d = cdist(data, V, metric='euclidean')                    # (N, C)
        d = np.fmax(d, np.finfo(float).eps)
        power = 2.0 / (self.m - 1)
        d_pow = d ** power                                        # (N, C)

        inv_d_pow = 1.0 / d_pow                                   # (N, C)
        sum_inv = np.sum(inv_d_pow, axis=1, keepdims=True)        # (N, 1)
        # Σ_l (d_ij/d_lj)^(2/(m-1)) = d_ij^p * Σ_l (1/d_lj^p)
        ratio_sum = d_pow * sum_inv                               # (N, C)

        b_col = self.b.reshape(-1, 1)                             # (N, 1)
        sum_f = np.sum(self.F, axis=1, keepdims=True)             # (N, 1)

        term1 = (1.0 + self.alpha * (1.0 - b_col * sum_f)) / ratio_sum
        term2 = self.alpha * self.F * b_col

        U = (term1 + term2) / (1.0 + self.alpha)
        return U

    def _calc_membership_unsup(self, data: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Cập nhật U không giám sát (FCM chuẩn) - dùng cho chế độ 'IS'."""
        d = cdist(data, V, metric='euclidean')
        d = np.fmax(d, np.finfo(float).eps) ** (2 / (self.m - 1))
        D = [d[:, j] for j in range(self.c)]
        num = 1 / np.array(D)
        den = np.sum(num, axis=0)
        U = num / division_by_zero(den)
        return np.squeeze(U).T

    # ----------------------------------------------------------------
    # 3. Hàm mục tiêu (fitness) - eq (1) với p = m = 2
    # ----------------------------------------------------------------
    def _fitness(self, data: np.ndarray, V: np.ndarray, U: np.ndarray) -> float:
        """
        J = Σ_i Σ_j u_ij^m d_ij^2  +  α Σ_i Σ_j (u_ij - f_ij b_j)^p d_ij^2
        (chọn p = 2 theo Pedrycz-Waletzky)
        """
        d2 = cdist(data, V, metric='euclidean') ** 2              # (N, C)
        J1 = np.sum((U ** self.m) * d2)
        diff = U - self.F * self.b.reshape(-1, 1)
        J2 = self.alpha * np.sum((diff ** 2) * d2)
        return float(J1 + J2)

    # ----------------------------------------------------------------
    # 4. Khởi tạo bầy particles (Algorithm 2, dòng 1-4)
    # ----------------------------------------------------------------
    def _init_swarm(self, data: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        N, D = data.shape
        P, C = self.swarm_size, self.c

        self.positions = np.zeros((P, C, D))
        self.U_particles = np.zeros((P, N, C))
        self.fitness = np.zeros(P)

        mask = self.b.astype(bool)
        for p in range(P):
            # Khởi tạo U ngẫu nhiên rồi chuẩn hoá theo hàng
            U0 = rng.random((N, C))
            U0 = U0 / np.sum(U0, axis=1, keepdims=True)
            # Ghi đè các hàng có nhãn = F (bước U_p = b_p * F trong paper, mở rộng cho đa dạng)
            U0[mask] = self.F[mask]

            self.U_particles[p] = U0
            self.positions[p] = self._calc_centers(data, U0)
            self.fitness[p] = self._fitness(data, self.positions[p], U0)

        # pbest ban đầu = vị trí hiện tại
        self.pbest_pos = self.positions.copy()
        self.pbest_U = self.U_particles.copy()
        self.pbest_fit = self.fitness.copy()

        # gbest = particle tốt nhất
        g = int(np.argmin(self.pbest_fit))
        self.gbest_pos = self.pbest_pos[g].copy()
        self.gbest_U = self.pbest_U[g].copy()
        self.gbest_fit = float(self.pbest_fit[g])

        self._rng = rng

    # ----------------------------------------------------------------
    # 5. Cập nhật vị trí particles - lớp con PHẢI override
    # ----------------------------------------------------------------
    def _update_positions(self, t: int) -> None:
        raise NotImplementedError("Lớp con phải cài đặt _update_positions()")

    # ----------------------------------------------------------------
    # 6. Vòng lặp chính (Algorithm 2)
    # ----------------------------------------------------------------
    def fit(self, data: np.ndarray, labels: np.ndarray):
        """
        Huấn luyện ssPSO trên dữ liệu data với nhãn bán giám sát labels
        (giá trị -1 nghĩa là không có nhãn).

        Trả về: (V, U, labels_pred, n_iter)
        """
        N = data.shape[0]
        self._init_semi_info(labels, N)
        self._init_swarm(data)

        start = time.time()
        for t in range(self.max_iter):
            # (1) Cập nhật vị trí particles theo cơ chế PSO cụ thể
            self._update_positions(t)

            # (2) Với mỗi particle: cập nhật U và fitness
            for p in range(self.swarm_size):
                V_p = self.positions[p]

                if self.semi_mode == 'ssFCM':
                    U_p = self._calc_membership_ss(data, V_p)
                else:   # 'IS' - sau khởi tạo không dùng giám sát
                    U_p = self._calc_membership_unsup(data, V_p)

                self.U_particles[p] = U_p
                self.fitness[p] = self._fitness(data, V_p, U_p)

                # (3) Cập nhật pbest (eq 4)
                if self.fitness[p] < self.pbest_fit[p]:
                    self.pbest_pos[p] = V_p.copy()
                    self.pbest_U[p] = U_p.copy()
                    self.pbest_fit[p] = self.fitness[p]

            # (4) Cập nhật gbest (eq 5)
            g = int(np.argmin(self.pbest_fit))
            if self.pbest_fit[g] < self.gbest_fit:
                self.gbest_pos = self.pbest_pos[g].copy()
                self.gbest_U = self.pbest_U[g].copy()
                self.gbest_fit = float(self.pbest_fit[g])

        self.process_time = time.time() - start

        # Ghi kết quả cuối cùng vào thuộc tính chuẩn của FCM để tương thích
        self.V = self.gbest_pos
        self.U = self.gbest_U
        labels_pred = np.argmax(self.U, axis=1)
        return self.V, self.U, labels_pred, self.max_iter


# ================================================================
# Biến thể 1: ssFCM-QPSO  (eq 8 - 11)
# ================================================================
class SSPSO_QPSO(SSPSO):
    """Cập nhật particles theo Quantum-behaved PSO."""

    def _update_positions(self, t: int) -> None:
        rng = self._rng
        P = self.swarm_size

        # β giảm tuyến tính từ 1.0 -> 0.1  (eq 11)
        beta = 0.9 * (self.max_iter - t) / self.max_iter + 0.1

        # mbest = trung bình pbest toàn bầy  (eq 8)
        mbest = np.mean(self.pbest_pos, axis=0)                # (C, D)

        for p in range(P):
            shape = self.positions[p].shape

            # Local attractor  (eq 9)
            theta = rng.random(shape)
            pp = theta * self.pbest_pos[p] + (1.0 - theta) * self.gbest_pos

            # Cập nhật vị trí  (eq 10)
            r = rng.random(shape)
            r = np.fmax(r, np.finfo(float).eps)
            sign = rng.choice([-1.0, 1.0], size=shape)

            self.positions[p] = pp + sign * beta * np.abs(mbest - self.positions[p]) * np.log(1.0 / r)


# ================================================================
# Biến thể 2: ssFCM-BBPSO  (eq 12)
# ================================================================
class SSPSO_BBPSO(SSPSO):
    """Bare Bones PSO - cập nhật theo phân phối Gaussian N(μ, σ)."""

    def _update_positions(self, t: int) -> None:
        rng = self._rng
        for p in range(self.swarm_size):
            mu = 0.5 * (self.pbest_pos[p] + self.gbest_pos)
            sigma = np.abs(self.pbest_pos[p] - self.gbest_pos)
            self.positions[p] = rng.normal(mu, sigma)


# ================================================================
# Biến thể 3: ssFCM-dlsBBPSO  (eq 13, 14)
# Biến thể có hiệu năng cao nhất trong bài báo (ISdB)
# ================================================================
class SSPSO_dlsBBPSO(SSPSO):
    """
    BBPSO with Dynamic Local Search (Guo & Sato 2017).
    Một particle giữ vai trò LEADER, các particle còn lại là TEAMMATE.
    - Leader  : cập nhật quanh gbest        (eq 13)
    - Teammate: cập nhật quanh leader       (eq 14)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.leader_idx = 0

    def _init_swarm(self, data: np.ndarray) -> None:
        super()._init_swarm(data)
        # Chọn ngẫu nhiên leader ban đầu
        self.leader_idx = int(self._rng.integers(0, self.swarm_size))

    def _update_positions(self, t: int) -> None:
        rng = self._rng
        P = self.swarm_size

        # Chọn ngẫu nhiên 1 particle; nếu fitness tốt hơn leader hiện tại thì thay
        cand = int(rng.integers(0, P))
        if self.pbest_fit[cand] < self.pbest_fit[self.leader_idx]:
            self.leader_idx = cand

        leader_pbest = self.pbest_pos[self.leader_idx]

        for p in range(P):
            if p == self.leader_idx:
                # Leader: xoay quanh gbest  (eq 13)
                mu = 0.5 * (leader_pbest + self.gbest_pos)
                sigma = np.abs(leader_pbest - self.gbest_pos)
            else:
                # Teammate: xoay quanh leader  (eq 14)
                mu = 0.5 * (leader_pbest + self.pbest_pos[p])
                sigma = np.abs(leader_pbest - self.pbest_pos[p])
            self.positions[p] = rng.normal(mu, sigma)


# ================================================================
# Demo nhanh trên Iris
# ================================================================
if __name__ == '__main__':
    import pandas as pd
    from .SSFCM import init_semi_data

    df = pd.read_csv('data_iris.csv')
    labels_raw = df['class'].values
    semi_labels = init_semi_data(labels_raw, ratio=0.1)       # 10% có nhãn
    data = df.iloc[:, :-1].values
    n_cluster = len(np.unique(labels_raw))

    variants = [
        (SSPSO_QPSO,     'ssFCM-QPSO'),
        (SSPSO_BBPSO,    'ssFCM-BBPSO'),
        (SSPSO_dlsBBPSO, 'ssFCM-dlsBBPSO'),
    ]

    for Cls, name in variants:
        for mode in ['ssFCM', 'IS']:
            model = Cls(c=n_cluster,
                        m=2,
                        max_iter=20,
                        swarm_size=20,
                        semi_mode=mode,
                        seed=42)
            V, U, lbl, it = model.fit(data, labels=semi_labels)
            tag = f"{mode:5s}-{name}"
            print(f"{tag:25s} | gbest = {model.gbest_fit:10.4f} | time = {model.process_time:.3f}s")