import numpy as np
from scipy.spatial.distance import cdist
import time


def division_by_zero(data):
    """Tránh chia cho 0"""
    if isinstance(data, np.ndarray):
        data[data == 0] = np.finfo(float).eps
        return data
    return np.finfo(float).eps if data == 0 else data


# =============================================================================
# 1. KFCM - Kernel Fuzzy C-Means (Section 2.2)
# =============================================================================
class KFCM:
    """
    Kernel Fuzzy C-Means (Eq 7-10)
    Sử dụng Gaussian kernel thay cho Euclidean distance.
    """

    def __init__(self, c: int, m: int = 2, max_iter: int = 1000,
                 eps: float = 1e-5, sigma: float = 100.0):
        self.c = c
        self.m = m
        self.max_iter = max_iter
        self.eps = eps
        self.sigma = sigma          # bandwidth của Gaussian kernel
        self.U = None
        self.V = None
        self.process_time = 0

    # ---- Gaussian kernel K(x, y) = exp(-||x-y||² / σ²)  (Eq 5) ----
    def gaussian_kernel(self, data: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Trả về vector K(xₖ, v) cho mỗi xₖ trong data. Shape: (n,)"""
        dist_sq = np.sum((data - v) ** 2, axis=1)
        return np.exp(-dist_sq / (self.sigma ** 2))

    def gaussian_kernel_matrix(self, data: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Trả về ma trận K(xₖ, Vᵢ). Shape: (n, c)"""
        n = data.shape[0]
        K = np.zeros((n, self.c))
        for i in range(self.c):
            K[:, i] = self.gaussian_kernel(data, V[i])
        return K

    # ---- Kernel distance: ||φ(x) - φ(v)||² = 2(1 - K(x, v))  (Eq 6) ----
    def kernel_distance(self, data: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Shape: (n, c)"""
        K = self.gaussian_kernel_matrix(data, V)
        return 2.0 * (1.0 - K)

    # ---- Khởi tạo U ngẫu nhiên ----
    def initialize_U(self, data: np.ndarray) -> np.ndarray:
        n = data.shape[0]
        U = np.random.rand(n, self.c)
        U = U / np.sum(U, axis=1, keepdims=True)
        return U

    # ---- Cập nhật tâm cụm (Eq 10) ----
    def calculate_V(self, data: np.ndarray) -> np.ndarray:
        """Vᵢ = Σₖ μᵢₖᵐ · K(xₖ, Vᵢ) · xₖ  /  Σₖ μᵢₖᵐ · K(xₖ, Vᵢ)"""
        um = self.U ** self.m                          # (n, c)
        K = self.gaussian_kernel_matrix(data, self.V)  # (n, c)
        V_new = np.zeros_like(self.V)
        for i in range(self.c):
            weights = um[:, i] * K[:, i]               # (n,)
            V_new[i] = (weights @ data) / division_by_zero(np.sum(weights))
        return V_new

    # ---- Cập nhật ma trận thành viên (Eq 9) ----
    def update_membership_matrix(self, data: np.ndarray) -> np.ndarray:
        """μᵢₖ = (1 - K(xₖ, Vᵢ))^(-1/(m-1)) / Σⱼ (1 - K(xₖ, Vⱼ))^(-1/(m-1))"""
        K = self.gaussian_kernel_matrix(data, self.V)       # (n, c)
        dist_kernel = np.fmax(1.0 - K, np.finfo(float).eps) # (n, c)
        exp = -1.0 / (self.m - 1)
        powered = dist_kernel ** exp                         # (n, c)
        denom = np.sum(powered, axis=1, keepdims=True)       # (n, 1)
        U = powered / division_by_zero(denom)
        return U

    def fit(self, data: np.ndarray):
        np.random.seed(42)
        self.U = self.initialize_U(data)
        # Khởi tạo V bằng random samples
        indices = np.random.choice(data.shape[0], self.c, replace=False)
        self.V = data[indices].copy().astype(float)

        start_time = time.time()
        for i in range(self.max_iter):
            U_old = self.U.copy()
            self.V = self.calculate_V(data)
            self.U = self.update_membership_matrix(data)
            if np.linalg.norm(self.U - U_old) < self.eps:
                break
        self.process_time = time.time() - start_time
        labels = np.argmax(self.U, axis=1)
        return self.V, self.U, labels, i + 1

    def get_labels(self):
        return np.argmax(self.U, axis=1)


# =============================================================================
# 2. KPCM - Kernel Possibilistic C-Means (Section 2.3, Algorithm 1)
# =============================================================================
class KPCM(KFCM):
    """
    Kernel Possibilistic C-Means (Eq 16-18)
    Kế thừa KFCM, thay đổi membership update và thêm η.
    """

    def __init__(self, c: int, m: int = 2, max_iter: int = 1000,
                 eps: float = 1e-5, sigma: float = 100.0):
        super().__init__(c, m, max_iter, eps, sigma)
        self.eta = None   # tham số ηᵢ cho mỗi cụm

    # ---- Tính ηᵢ (Eq 18) ----
    def compute_eta(self, data: np.ndarray) -> np.ndarray:
        """ηᵢ = Σₖ μᵢₖᵐ · 2(1 - K(xₖ,Vᵢ)) / Σₖ μᵢₖᵐ"""
        um = self.U ** self.m                              # (n, c)
        dist = self.kernel_distance(data, self.V)          # (n, c)
        numerator = np.sum(um * dist, axis=0)              # (c,)
        denominator = np.sum(um, axis=0)                   # (c,)
        eta = numerator / division_by_zero(denominator)
        return np.fmax(eta, np.finfo(float).eps)

    # ---- Cập nhật membership KPCM (Eq 17) ----
    def update_membership_matrix(self, data: np.ndarray) -> np.ndarray:
        """μᵢₖ = 1 / (1 + (2(1-K(xₖ,Vᵢ)) / ηᵢ)^(1/(m-1)))"""
        dist = self.kernel_distance(data, self.V)         # (n, c)
        exp = 1.0 / (self.m - 1)
        U = np.zeros_like(dist)
        for i in range(self.c):
            ratio = dist[:, i] / division_by_zero(self.eta[i])
            U[:, i] = 1.0 / (1.0 + np.power(ratio, exp))
        return U

    # ---- Hàm mục tiêu J_KPCM (Eq 16) ----
    def compute_objective(self, data: np.ndarray) -> float:
        um = self.U ** self.m
        dist = self.kernel_distance(data, self.V)
        term1 = np.sum(um * dist)
        term2 = np.sum(self.eta * np.sum((1 - self.U) ** self.m, axis=0))
        return term1 + term2

    def fit(self, data: np.ndarray):
        np.random.seed(42)
        self.U = self.initialize_U(data)
        indices = np.random.choice(data.shape[0], self.c, replace=False)
        self.V = data[indices].copy().astype(float)
        self.eta = self.compute_eta(data)

        start_time = time.time()
        for i in range(self.max_iter):
            U_old = self.U.copy()
            self.eta = self.compute_eta(data)
            self.V = self.calculate_V(data)
            self.U = self.update_membership_matrix(data)
            if np.linalg.norm(self.U - U_old) < self.eps:
                break
        self.process_time = time.time() - start_time
        labels = np.argmax(self.U, axis=1)
        return self.V, self.U, labels, i + 1


# =============================================================================
# 3. IKPCM - Improved Kernel Possibilistic C-Means (Section 3, Algorithm chính)
#    Bao gồm: PSO initialization + Outlier rejection (α) + Spatial information
# =============================================================================
class IKPCM(KPCM):
    """
    Improved Kernel Possibilistic C-Means (Mekhmoukh & Mokrani, 2015)

    Cải tiến so với KPCM:
    1. Khởi tạo tâm cụm bằng PSO (Algorithm 2, Eq 19-20)
    2. Outlier rejection qua hệ số α (Eq 22-27)
    3. Spatial neighborhood information (Eq 28-29)

    Parameters
    ----------
    c           : int   - số cụm
    m           : int   - độ mờ (m > 1)
    max_iter    : int   - số vòng lặp tối đa
    eps         : float - ngưỡng hội tụ
    sigma       : float - bandwidth của Gaussian kernel
    p           : float - trọng số cho μ trong spatial (Eq 29)
    q           : float - trọng số cho h trong spatial (Eq 29)
    image_shape : tuple - (H, W) nếu dữ liệu là ảnh 2D, None nếu không dùng spatial
    alpha       : float - hệ số outlier rejection, None = tự tính từ dữ liệu
    n_particles : int   - số hạt PSO
    pso_max_iter: int   - số vòng lặp tối đa của PSO
    """

    def __init__(self, c: int, m: int = 2, max_iter: int = 1000,
                 eps: float = 1e-5, sigma: float = 100.0,
                 p: float = 1.0, q: float = 1.0,
                 image_shape=None, alpha=None,
                 n_particles: int = 12, pso_max_iter: int = 300):
        super().__init__(c, m, max_iter, eps, sigma)
        self.p = p
        self.q = q
        self.image_shape = image_shape   # (H, W) hoặc None
        self.alpha_param = alpha
        self.alpha = None                # sẽ tính trong fit()
        self.n_particles = n_particles
        self.pso_max_iter = pso_max_iter

    # =====================================================================
    # Outlier rejection distance: α · ||φ(xₖ) - φ(Vᵢ)||²  (Eq 22)
    # =====================================================================
    def compute_alpha(self, data: np.ndarray) -> float:
        """
        Eq 26-27: α = (Xmax - Xmin + 1) / (max_range + 1) + 1
        Với ảnh 8-bit: max_range = 256  →  α ∈ [1, 2]
        Với dữ liệu tổng quát: dùng max theo từng chiều
        """
        if self.alpha_param is not None:
            return self.alpha_param
        x_max = np.max(data)
        x_min = np.min(data)
        intensity_range = x_max - x_min
        # max_range = 256 cho ảnh 8-bit, tổng quát dùng cùng công thức
        max_range = 256.0
        alpha = (intensity_range + 1.0) / (max_range + 1.0) + 1.0
        return np.clip(alpha, 1.0, 2.0)

    def outlier_kernel_distance(self, data: np.ndarray, V: np.ndarray) -> np.ndarray:
        """α · 2(1 - K(xₖ, Vᵢ))  — khoảng cách kernel có outlier rejection. Shape: (n, c)"""
        return self.alpha * self.kernel_distance(data, V)

    # =====================================================================
    # Cập nhật ηᵢ với outlier rejection (Eq 25)
    # =====================================================================
    def compute_eta(self, data: np.ndarray) -> np.ndarray:
        """ηᵢ = Σₖ μᵢₖᵐ · α·d²ₖᵢ  /  Σₖ μᵢₖᵐ"""
        um = self.U ** self.m
        dist = self.outlier_kernel_distance(data, self.V)
        numerator = np.sum(um * dist, axis=0)
        denominator = np.sum(um, axis=0)
        eta = numerator / division_by_zero(denominator)
        return np.fmax(eta, np.finfo(float).eps)

    # =====================================================================
    # Cập nhật membership IKPCM (Eq 24)
    # =====================================================================
    def update_membership_matrix(self, data: np.ndarray) -> np.ndarray:
        """μᵢₖ = 1 / (1 + (α · ||φ(xₖ)-φ(Vᵢ)||² / ηᵢ)^(1/(m-1)))"""
        dist = self.outlier_kernel_distance(data, self.V)
        exp = 1.0 / (self.m - 1)
        U = np.zeros_like(dist)
        for i in range(self.c):
            ratio = dist[:, i] / division_by_zero(self.eta[i])
            U[:, i] = 1.0 / (1.0 + np.power(ratio, exp))
        return U

    # =====================================================================
    # Spatial neighborhood (Eq 28-29)
    # =====================================================================
    def compute_spatial_h(self, U: np.ndarray) -> np.ndarray:
        """
        Eq 28: hᵢₖ = Σⱼ∈NB(xₖ) μᵢⱼ   (cửa sổ 3×3)
        Chỉ dùng khi image_shape != None.
        """
        if self.image_shape is None:
            return None
        H, W = self.image_shape
        U_img = U.reshape(H, W, self.c)
        padded = np.pad(U_img, ((1, 1), (1, 1), (0, 0)), mode='constant')
        h = np.zeros_like(U_img)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                h += padded[1 + di: H + 1 + di, 1 + dj: W + 1 + dj, :]
        return h.reshape(-1, self.c)

    def apply_spatial(self, U: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Eq 29: μ*ᵢₖ = (μᵢₖ^p · hᵢₖ^q) / Σⱼ (μⱼₖ^p · hⱼₖ^q)
        """
        if h is None:
            return U
        numerator = np.power(U, self.p) * np.power(h, self.q)
        denominator = np.sum(numerator, axis=1, keepdims=True)
        return numerator / division_by_zero(denominator)

    # =====================================================================
    # Hàm mục tiêu J_IKPCM (Eq 22)
    # =====================================================================
    def compute_objective(self, data: np.ndarray) -> float:
        um = self.U ** self.m
        dist = self.outlier_kernel_distance(data, self.V)
        term1 = np.sum(um * dist)
        term2 = np.sum(self.eta * np.sum((1.0 - self.U) ** self.m, axis=0))
        return term1 + term2

    # =====================================================================
    # PSO Initialization (Algorithm 2, Section 3.2)
    # =====================================================================
    def pso_initialize(self, data: np.ndarray) -> np.ndarray:
        """
        Khởi tạo tâm cụm bằng PSO (Table 1, Algorithm 2).
        Mỗi hạt = 1 bộ c tâm cụm (c × d).
        Fitness = J_IKPCM tại bộ tâm đó.
        """
        n, d = data.shape
        # --- Tham số PSO (Table 1) ---
        c1, c2 = 1.70, 1.70
        w_max, w_min = 0.9, 0.4
        pso_eps = 1e-6
        nberp = 10

        # --- Khởi tạo hạt ---
        particles = np.zeros((self.n_particles, self.c, d))
        velocities = np.zeros_like(particles)
        for p_idx in range(self.n_particles):
            idx = np.random.choice(n, self.c, replace=False)
            particles[p_idx] = data[idx].astype(float)
            velocities[p_idx] = np.random.randn(self.c, d) * 0.1

        pbest = particles.copy()
        pbest_fitness = np.full(self.n_particles, np.inf)
        gbest = particles[0].copy()
        gbest_fitness = np.inf

        fitness_history = []

        for iteration in range(self.pso_max_iter):
            w = w_max - (w_max - w_min) * iteration / self.pso_max_iter

            for p_idx in range(self.n_particles):
                V_candidate = particles[p_idx]

                # Tính kernel distance với α
                dist = self.outlier_kernel_distance(data, V_candidate)

                # Tính membership sơ bộ (Eq 17 với Euclidean cho lần đầu)
                eta_temp = np.mean(dist, axis=0)
                eta_temp = np.fmax(eta_temp, np.finfo(float).eps)

                exp = 1.0 / (self.m - 1)
                U_temp = np.zeros((n, self.c))
                for i in range(self.c):
                    ratio = dist[:, i] / division_by_zero(eta_temp[i])
                    U_temp[:, i] = 1.0 / (1.0 + np.power(ratio, exp))

                # Tính lại η chính xác (Eq 25)
                um = U_temp ** self.m
                eta_temp = np.sum(um * dist, axis=0) / division_by_zero(np.sum(um, axis=0))
                eta_temp = np.fmax(eta_temp, np.finfo(float).eps)

                # Tính lại membership (Eq 24)
                for i in range(self.c):
                    ratio = dist[:, i] / division_by_zero(eta_temp[i])
                    U_temp[:, i] = 1.0 / (1.0 + np.power(ratio, exp))

                # Fitness = J_IKPCM (Eq 22)
                um = U_temp ** self.m
                fitness = (np.sum(um * dist)
                           + np.sum(eta_temp * np.sum((1.0 - U_temp) ** self.m, axis=0)))

                # Cập nhật pbest, gbest
                if fitness < pbest_fitness[p_idx]:
                    pbest_fitness[p_idx] = fitness
                    pbest[p_idx] = V_candidate.copy()
                if fitness < gbest_fitness:
                    gbest_fitness = fitness
                    gbest = V_candidate.copy()

            fitness_history.append(gbest_fitness)

            # Điều kiện dừng (Eq 21)
            if len(fitness_history) > nberp:
                recent = fitness_history[-nberp:]
                if all(abs(recent[j + 1] - recent[j]) < pso_eps
                       for j in range(len(recent) - 1)):
                    break

            # Cập nhật vận tốc và vị trí (Eq 19-20)
            for p_idx in range(self.n_particles):
                r1 = np.random.rand(self.c, d)
                r2 = np.random.rand(self.c, d)
                velocities[p_idx] = (w * velocities[p_idx]
                                     + c1 * r1 * (pbest[p_idx] - particles[p_idx])
                                     + c2 * r2 * (gbest - particles[p_idx]))
                particles[p_idx] = particles[p_idx] + velocities[p_idx]

        return gbest

    # =====================================================================
    # fit() - Thuật toán chính (Fig. 4)
    # =====================================================================
    def fit(self, data: np.ndarray):
        """
        Quy trình IKPCM (Fig. 4 trong bài báo):
        Step 1: PSO khởi tạo Vᵢ và μᵢₖ
        Step 2: Lặp IKPCM (η → spatial → μ → μ* → V)
        """
        np.random.seed(42)
        start_time = time.time()

        # --- Tính α (Eq 26-27) ---
        self.alpha = self.compute_alpha(data)

        # --- Step 1: PSO khởi tạo tâm cụm ---
        self.V = self.pso_initialize(data)

        # --- Khởi tạo η và U ban đầu ---
        dist_init = self.outlier_kernel_distance(data, self.V)
        self.eta = np.mean(dist_init, axis=0)
        self.eta = np.fmax(self.eta, np.finfo(float).eps)
        self.U = self.update_membership_matrix(data)

        # --- Step 2: Vòng lặp chính ---
        for i in range(self.max_iter):
            U_old = self.U.copy()

            # 2a. Ước lượng ηᵢ (Eq 25)
            self.eta = self.compute_eta(data)

            # 2b. Tính spatial h (Eq 28)
            h = self.compute_spatial_h(self.U)

            # 2c. Cập nhật membership μᵢₖ (Eq 24)
            self.U = self.update_membership_matrix(data)

            # 2d. Áp dụng spatial: μ* (Eq 29)
            self.U = self.apply_spatial(self.U, h)

            # 2e. Cập nhật tâm cụm Vᵢ (Eq 10)
            self.V = self.calculate_V(data)

            # Kiểm tra hội tụ
            if np.linalg.norm(self.U - U_old) < self.eps:
                break

        self.process_time = time.time() - start_time
        labels = np.argmax(self.U, axis=1)
        return self.V, self.U, labels, i + 1

    def get_labels(self):
        return np.argmax(self.U, axis=1)