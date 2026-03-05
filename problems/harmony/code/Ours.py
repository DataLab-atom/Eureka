import numpy as np
from sklearn.cluster import KMeans
from functools import partial


class harm_Ours(object):
    def __init__(
        self, Z, Phi, Phi_moe, Pr_b, sigma,
        theta, max_iter_harmony, max_iter_kmeans,
        epsilon_kmeans, epsilon_harmony, K, block_size,
        lamb, verbose, random_state=None, cluster_fn='kmeans'
    ):
        self.static_init(
            Z, Phi, Phi_moe, Pr_b, sigma, theta, max_iter_harmony,
            max_iter_kmeans, epsilon_kmeans, epsilon_harmony, K, block_size,
            lamb, verbose, random_state, cluster_fn
        )

    @staticmethod
    def safe_entropy(x: np.array):
        y = np.multiply(x, np.log(x))
        y[~np.isfinite(y)] = 0.0
        return y

    @staticmethod
    def moe_correct_ridge(Z_orig, Z_cos, Z_corr, R, W, K, Phi_Rk, Phi_moe, lamb):
        Z_corr = Z_orig.copy()
        for i in range(K):
            Phi_Rk = np.multiply(Phi_moe, R[i,:])
            x = np.dot(Phi_Rk, Phi_moe.T) + lamb
            W = np.dot(np.dot(np.linalg.inv(x), Phi_Rk), Z_orig.T)
            W[0,:] = 0 # do not remove the intercept
            Z_corr -= np.dot(W.T, Phi_Rk)
        Z_cos = Z_corr / np.linalg.norm(Z_corr, ord=2, axis=0)
        return Z_cos, Z_corr, W, Phi_Rk

    @staticmethod
    def static_allocate_buffers(harmony):
        harmony._scale_dist = np.zeros((harmony.K, harmony.N))
        harmony.dist_mat = np.zeros((harmony.K, harmony.N))
        harmony.O = np.zeros((harmony.K, harmony.B))
        harmony.E = np.zeros((harmony.K, harmony.B))
        harmony.W = np.zeros((harmony.B + 1, harmony.d))
        harmony.Phi_Rk = np.zeros((harmony.B + 1, harmony.N))

    @staticmethod
    def static_cluster_kmeans(data, K, random_state):
        print("Computing initial centroids with sklearn.KMeans...")
        model = KMeans(n_clusters=K, init='k-means++',
                      n_init=10, max_iter=25, random_state=random_state)
        model.fit(data)
        km_centroids = model.cluster_centers_
        print("sklearn.KMeans initialization complete.")
        return km_centroids

    @staticmethod
    def static_init_cluster(harmony, cluster_fn):
        # cluster, shape d×K
        harmony.Y = cluster_fn(harmony.Z_cos.T, harmony.K).T
        # (1) Normalize
        harmony.Y = harmony.Y / np.linalg.norm(harmony.Y, ord=2, axis=0)
        # (2) Assign cluster probabilities
        harmony.dist_mat = 2 * (1 - np.dot(harmony.Y.T, harmony.Z_cos))
        harmony.R = -harmony.dist_mat
        harmony.R = harmony.R / harmony.sigma[:,None]
        harmony.R -= np.max(harmony.R, axis=0)
        harmony.R = np.exp(harmony.R)
        harmony.R = harmony.R / np.sum(harmony.R, axis=0)
        # (3) Batch diversity statistics
        harmony.E = np.outer(np.sum(harmony.R, axis=1), harmony.Pr_b)
        harmony.O = np.inner(harmony.R, harmony.Phi)
        harm_Ours.static_compute_objective(harmony)
        harmony.objective_harmony.append(harmony.objective_kmeans[-1])

    @staticmethod
    def static_compute_objective(harmony):
        kmeans_error = np.sum(np.multiply(harmony.R, harmony.dist_mat))
        _entropy = np.sum(harm_Ours.safe_entropy(harmony.R) * harmony.sigma[:,np.newaxis])
        x = (harmony.R * harmony.sigma[:,np.newaxis])
        y = np.tile(harmony.theta[:,np.newaxis], harmony.K).T
        z = np.log((harmony.O + 1) / (harmony.E + 1))
        w = np.dot(y * z, harmony.Phi)
        _cross_entropy = np.sum(x * w)
        harmony.objective_kmeans.append(kmeans_error + _entropy + _cross_entropy)
        harmony.objective_kmeans_dist.append(kmeans_error)
        harmony.objective_kmeans_entropy.append(_entropy)
        harmony.objective_kmeans_cross.append(_cross_entropy)

    @staticmethod
    def static_harmonize(harmony, iter_harmony=10, verbose=True):
        converged = False
        for i in range(1, iter_harmony + 1):
            if verbose:
                print("Iteration {} of {}".format(i, iter_harmony))
            harm_Ours.static_cluster(harmony)
            harmony.Z_cos, harmony.Z_corr, harmony.W, harmony.Phi_Rk = harm_Ours.moe_correct_ridge(
                harmony.Z_orig, harmony.Z_cos, harmony.Z_corr, harmony.R, harmony.W, harmony.K,
                harmony.Phi_Rk, harmony.Phi_moe, harmony.lamb
            )
            converged = harm_Ours.static_check_convergence(harmony, 1)
            if converged:
                if verbose:
                    print(
                        "Converged after {} iteration{}".format(i, 's' if i > 1 else '')
                    )
                break
        if verbose and not converged:
            print("Stopped before convergence")
        return 0

    @staticmethod
    def static_cluster(harmony):
        harmony.dist_mat = 2 * (1 - np.dot(harmony.Y.T, harmony.Z_cos))
        for i in range(harmony.max_iter_kmeans):
            harmony.Y = np.dot(harmony.Z_cos, harmony.R.T)
            harmony.Y = harmony.Y / np.linalg.norm(harmony.Y, ord=2, axis=0)
            harmony.dist_mat = 2 * (1 - np.dot(harmony.Y.T, harmony.Z_cos))
            harm_Ours.static_update_R(harmony)
            harm_Ours.static_compute_objective(harmony)
            if i > harmony.window_size:
                converged = harm_Ours.static_check_convergence(harmony, 0)
                if converged:
                    break
        harmony.kmeans_rounds.append(i)
        harmony.objective_harmony.append(harmony.objective_kmeans[-1])
        return 0

    @staticmethod
    def static_update_R(harmony):
        harmony._scale_dist = -harmony.dist_mat
        harmony._scale_dist = harmony._scale_dist / harmony.sigma[:,None]
        harmony._scale_dist -= np.max(harmony._scale_dist, axis=0)
        harmony._scale_dist = np.exp(harmony._scale_dist)
        update_order = np.arange(harmony.N)
        np.random.shuffle(update_order)
        n_blocks = np.ceil(1/harmony.block_size).astype(int)
        blocks = np.array_split(update_order, n_blocks)
        for b in blocks:
            harmony.E -= np.outer(np.sum(harmony.R[:,b], axis=1), harmony.Pr_b)
            harmony.O -= np.dot(harmony.R[:,b], harmony.Phi[:,b].T)
            harmony.R[:,b] = harmony._scale_dist[:,b]
            harmony.R[:,b] = np.multiply(
                harmony.R[:,b],
                np.dot(
                    np.power((harmony.E + 1) / (harmony.O + 1), harmony.theta),
                    harmony.Phi[:,b]
                )
            )
            harmony.R[:,b] = harmony.R[:,b] / np.linalg.norm(harmony.R[:,b], ord=1, axis=0)
            harmony.E += np.outer(np.sum(harmony.R[:,b], axis=1), harmony.Pr_b)
            harmony.O += np.dot(harmony.R[:,b], harmony.Phi[:,b].T)
        return 0

    @staticmethod
    def static_check_convergence(harmony, i_type):
        obj_old = 0.0
        obj_new = 0.0
        if i_type == 0:
            okl = len(harmony.objective_kmeans)
            for i in range(harmony.window_size):
                obj_old += harmony.objective_kmeans[okl - 2 - i]
                obj_new += harmony.objective_kmeans[okl - 1 - i]
            if abs(obj_old - obj_new) / abs(obj_old) < harmony.epsilon_kmeans:
                return True
            return False
        if i_type == 1:
            obj_old = harmony.objective_harmony[-2]
            obj_new = harmony.objective_harmony[-1]
            if (obj_old - obj_new) / abs(obj_old) < harmony.epsilon_harmony:
                return True
            return False
        return True

    def static_init(self, Z, Phi, Phi_moe, Pr_b, sigma, theta, max_iter_harmony,
                       max_iter_kmeans, epsilon_kmeans, epsilon_harmony, K, block_size,
                       lamb, verbose, random_state, cluster_fn):
        import time
        start_time = time.time()

        Z = np.asarray(Z, dtype=np.float64)
        mu = np.mean(Z, axis=1, keepdims=True)
        std = np.std(Z, axis=1, keepdims=True)
        std[std < 1e-8] = 1.0
        Z_norm = (Z - mu) / std

        self.Z_orig = Z_norm
        self.Z_corr = Z_norm.copy()

        col_norms = np.linalg.norm(self.Z_corr, axis=0, keepdims=True)
        col_norms[col_norms < 1e-16] = 1.0
        self.Z_cos = self.Z_corr / col_norms

        self.Phi = Phi
        self.Phi_moe = Phi_moe
        self.Pr_b = Pr_b
        self.sigma = np.asarray(sigma)
        self.sigma_prior = self.sigma.copy()
        self.theta = np.asarray(theta)
        self.lamb = lamb
        self.block_size = block_size
        self.K = K
        self.max_iter_harmony = max_iter_harmony
        self.max_iter_kmeans = max_iter_kmeans
        self.epsilon_kmeans = epsilon_kmeans
        self.epsilon_harmony = epsilon_harmony
        self.verbose = verbose
        self.random_state = random_state

        self.d, self.N = self.Z_corr.shape
        self.B = self.Phi.shape[0]

        self.static_allocate_buffers(self)

        self.objective_harmony = []
        self.objective_kmeans = []
        self.objective_kmeans_dist = []
        self.objective_kmeans_entropy = []
        self.objective_kmeans_cross = []
        self.kmeans_rounds = []
        self.window_size = 3

        if cluster_fn == 'kmeans':
            cluster_fn = partial(self.static_cluster_kmeans, random_state=random_state)

        self.static_init_cluster(self, cluster_fn)

        if verbose:
            print(f"Harmony initialization complete: {self.d} features, {self.N} samples, {self.K} clusters")
            print(f"Starting Harmony batch correction with max {max_iter_harmony} iterations")
        self.static_harmonize(self, iter_harmony=max_iter_harmony, verbose=verbose)
        if verbose:
            elapsed = time.time() - start_time
            print(f"Harmony batch correction finished in {elapsed:.2f} seconds")

    def result(self):
        return self.Z_corr