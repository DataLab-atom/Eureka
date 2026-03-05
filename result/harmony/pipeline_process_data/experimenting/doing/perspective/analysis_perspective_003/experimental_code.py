import os
import sys
sys.path.append(os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))
import os
import numpy as np
import pandas as pd
from functools import partial
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from typing import Iterable
from math import ceil
from scipy.stats import pearsonr

# Ensure results path
RESULT_DIR = os.path.join('.', 'experimental_result_data')
os.makedirs(RESULT_DIR, exist_ok=True)

# -----------------------------
# LISI / Local diversity code
# -----------------------------

def safe_entropy(x: np.array):
    y = np.multiply(x, np.log(x))
    y[~np.isfinite(y)] = 0.0
    return y


def compute_simpson(
    distances: np.ndarray,
    indices: np.ndarray,
    labels: pd.Categorical,
    n_categories: int,
    perplexity: float,
    tol: float = 1e-5,
):
    n = distances.shape[1]
    P = np.zeros(distances.shape[0])
    simpson = np.zeros(n)
    logU = np.log(perplexity)
    # Loop through each cell.
    for i in range(n):
        beta = 1
        betamin = -np.inf
        betamax = np.inf
        # Compute Hdiff
        P = np.exp(-distances[:, i] * beta)
        P_sum = np.sum(P)
        if P_sum == 0:
            H = 0
            P = np.zeros(distances.shape[0])
        else:
            H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
            P = P / P_sum
        Hdiff = H - logU
        n_tries = 50
        for t in range(n_tries):
            if abs(Hdiff) < tol:
                break
            if Hdiff > 0:
                betamin = beta
                if not np.isfinite(betamax):
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if not np.isfinite(betamin):
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2
            P = np.exp(-distances[:, i] * beta)
            P_sum = np.sum(P)
            if P_sum == 0:
                H = 0
                P = np.zeros(distances.shape[0])
            else:
                H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
                P = P / P_sum
            Hdiff = H - logU
        if H == 0:
            simpson[i] = -1
            continue
        for label_category in labels.categories:
            ix = indices[:, i]
            q = labels[ix] == label_category
            if np.any(q):
                P_sum = np.sum(P[q])
                simpson[i] += P_sum * P_sum
    return simpson


def compute_lisi(
    X: np.array,
    metadata: pd.DataFrame,
    label_colnames: Iterable[str],
    perplexity: float = 30,
):
    n_cells = metadata.shape[0]
    n_labels = len(label_colnames)
    knn = NearestNeighbors(n_neighbors=max(3, int(perplexity * 3)), algorithm='kd_tree').fit(X)
    distances, indices = knn.kneighbors(X)
    # drop self
    indices = indices[:, 1:]
    distances = distances[:, 1:]
    lisi_df = np.zeros((n_cells, n_labels))
    for i, label in enumerate(label_colnames):
        labels = pd.Categorical(metadata[label])
        n_categories = len(labels.categories)
        simpson = compute_simpson(distances.T, indices.T, labels, n_categories, perplexity)
        lisi_df[:, i] = 1 / simpson
    return lisi_df

# -----------------------------
# Shared ridge correction used by both classes
# -----------------------------

def moe_correct_ridge(Z_orig, Z_cos, Z_corr, R, W, K, Phi_Rk, Phi_moe, lamb):
    Z_corr = Z_orig.copy()
    for i in range(K):
        Phi_Rk = np.multiply(Phi_moe, R[i, :])
        x = np.dot(Phi_Rk, Phi_moe.T) + lamb
        W = np.dot(np.dot(np.linalg.inv(x), Phi_Rk), Z_orig.T)
        W[0, :] = 0  # do not remove the intercept
        Z_corr -= np.dot(W.T, Phi_Rk)
    # normalize columns
    col_norms = np.linalg.norm(Z_corr, ord=2, axis=0, keepdims=True)
    col_norms[col_norms < 1e-16] = 1.0
    Z_cos = Z_corr / col_norms
    return Z_cos, Z_corr, W, Phi_Rk

# -----------------------------
# Baseline Harmony class (reproduced from provided code)
# -----------------------------
class Harmony(object):
    def __init__(
        self, Z, Phi, Phi_moe, Pr_b, sigma,
        theta, max_iter_harmony, max_iter_kmeans,
        epsilon_kmeans, epsilon_harmony, K, block_size,
        lamb, verbose, random_state=None, cluster_fn='kmeans'
    ):
        self.Z_corr = np.array(Z)
        self.Z_orig = np.array(Z)

        self.Z_cos = self.Z_orig / self.Z_orig.max(axis=0)
        self.Z_cos = self.Z_cos / np.linalg.norm(self.Z_cos, ord=2, axis=0)

        self.Phi = Phi
        self.Phi_moe = Phi_moe
        self.N = self.Z_corr.shape[1]
        self.Pr_b = Pr_b
        self.B = self.Phi.shape[0]
        self.d = self.Z_corr.shape[0]
        self.window_size = 3
        self.epsilon_kmeans = epsilon_kmeans
        self.epsilon_harmony = epsilon_harmony

        self.lamb = lamb
        self.sigma = sigma
        self.sigma_prior = sigma
        self.block_size = block_size
        self.K = K
        self.max_iter_harmony = max_iter_harmony
        self.max_iter_kmeans = max_iter_kmeans
        self.verbose = verbose
        self.theta = theta

        self.objective_harmony = []
        self.objective_kmeans = []
        self.objective_kmeans_dist = []
        self.objective_kmeans_entropy = []
        self.objective_kmeans_cross = []
        self.kmeans_rounds = []

        self.allocate_buffers()
        if cluster_fn == 'kmeans':
            cluster_fn = partial(Harmony._cluster_kmeans, random_state=random_state)
        self.init_cluster(cluster_fn)
        self.harmonize(self.max_iter_harmony, self.verbose)

    def result(self):
        return self.Z_corr

    def allocate_buffers(self):
        self._scale_dist = np.zeros((self.K, self.N))
        self.dist_mat = np.zeros((self.K, self.N))
        self.O = np.zeros((self.K, self.B))
        self.E = np.zeros((self.K, self.B))
        self.W = np.zeros((self.B + 1, self.d))
        self.Phi_Rk = np.zeros((self.B + 1, self.N))

    @staticmethod
    def _cluster_kmeans(data, K, random_state):
        print("Computing initial centroids with sklearn.KMeans...")
        model = KMeans(n_clusters=K, init='k-means++', n_init=10, max_iter=25, random_state=random_state)
        model.fit(data)
        km_centroids, km_labels = model.cluster_centers_, model.labels_
        print("sklearn.KMeans initialization complete.")
        return km_centroids

    def init_cluster(self, cluster_fn):
        self.Y = cluster_fn(self.Z_cos.T, self.K).T
        self.Y = self.Y / np.linalg.norm(self.Y, ord=2, axis=0)
        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
        self.R = -self.dist_mat
        self.R = self.R / self.sigma[:, None]
        self.R -= np.max(self.R, axis=0)
        self.R = np.exp(self.R)
        self.R = self.R / np.sum(self.R, axis=0)
        self.E = np.outer(np.sum(self.R, axis=1), self.Pr_b)
        self.O = np.inner(self.R, self.Phi)
        self.compute_objective()
        self.objective_harmony.append(self.objective_kmeans[-1])

    def compute_objective(self):
        kmeans_error = np.sum(np.multiply(self.R, self.dist_mat))
        _entropy = np.sum(safe_entropy(self.R) * self.sigma[:, np.newaxis])
        x = (self.R * self.sigma[:, np.newaxis])
        y = np.tile(self.theta[:, np.newaxis], self.K).T
        z = np.log((self.O + 1) / (self.E + 1))
        w = np.dot(y * z, self.Phi)
        _cross_entropy = np.sum(x * w)
        self.objective_kmeans.append(kmeans_error + _entropy + _cross_entropy)
        self.objective_kmeans_dist.append(kmeans_error)
        self.objective_kmeans_entropy.append(_entropy)
        self.objective_kmeans_cross.append(_cross_entropy)

    def harmonize(self, iter_harmony=10, verbose=True):
        converged = False
        for i in range(1, iter_harmony + 1):
            if verbose:
                print("Iteration {} of {}".format(i, iter_harmony))
            self.cluster()
            self.Z_cos, self.Z_corr, self.W, self.Phi_Rk = moe_correct_ridge(
                self.Z_orig, self.Z_cos, self.Z_corr, self.R, self.W, self.K, self.Phi_Rk, self.Phi_moe, self.lamb
            )
            converged = self.check_convergence(1)
            if converged:
                if verbose:
                    print("Converged after {} iteration{}".format(i, 's' if i > 1 else ''))
                break
        if verbose and not converged:
            print("Stopped before convergence")
        return 0

    def cluster(self):
        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
        for i in range(self.max_iter_kmeans):
            self.Y = np.dot(self.Z_cos, self.R.T)
            self.Y = self.Y / np.linalg.norm(self.Y, ord=2, axis=0)
            self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
            self.update_R()
            self.compute_objective()
            if i > self.window_size:
                converged = self.check_convergence(0)
                if converged:
                    break
        self.kmeans_rounds.append(i)
        self.objective_harmony.append(self.objective_kmeans[-1])
        return 0

    def update_R(self):
        self._scale_dist = -self.dist_mat
        self._scale_dist = self._scale_dist / self.sigma[:, None]
        self._scale_dist -= np.max(self._scale_dist, axis=0)
        self._scale_dist = np.exp(self._scale_dist)
        update_order = np.arange(self.N)
        np.random.shuffle(update_order)
        n_blocks = int(np.ceil(1 / self.block_size))
        blocks = np.array_split(update_order, n_blocks)
        for b in blocks:
            self.E -= np.outer(np.sum(self.R[:, b], axis=1), self.Pr_b)
            self.O -= np.dot(self.R[:, b], self.Phi[:, b].T)
            self.R[:, b] = self._scale_dist[:, b]
            self.R[:, b] = np.multiply(
                self.R[:, b],
                np.dot(np.power((self.E + 1) / (self.O + 1), self.theta), self.Phi[:, b]),
            )
            self.R[:, b] = self.R[:, b] / np.linalg.norm(self.R[:, b], ord=1, axis=0)
            self.E += np.outer(np.sum(self.R[:, b], axis=1), self.Pr_b)
            self.O += np.dot(self.R[:, b], self.Phi[:, b].T)
        return 0

    def check_convergence(self, i_type):
        if i_type == 0:
            okl = len(self.objective_kmeans)
            obj_old = 0.0
            obj_new = 0.0
            for i in range(self.window_size):
                obj_old += self.objective_kmeans[okl - 2 - i]
                obj_new += self.objective_kmeans[okl - 1 - i]
            if abs(obj_old - obj_new) / abs(obj_old) < self.epsilon_kmeans:
                return True
            return False
        if i_type == 1:
            obj_old = self.objective_harmony[-2]
            obj_new = self.objective_harmony[-1]
            if (obj_old - obj_new) / abs(obj_old) < self.epsilon_harmony:
                return True
            return False
        return True

# -----------------------------
# Our MoE-Harmony class (reproduced)
# -----------------------------
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
            Phi_Rk = np.multiply(Phi_moe, R[i, :])
            x = np.dot(Phi_Rk, Phi_moe.T) + lamb
            W = np.dot(np.dot(np.linalg.inv(x), Phi_Rk), Z_orig.T)
            W[0, :] = 0  # do not remove the intercept
            Z_corr -= np.dot(W.T, Phi_Rk)
        col_norms = np.linalg.norm(Z_corr, ord=2, axis=0, keepdims=True)
        col_norms[col_norms < 1e-16] = 1.0
        Z_cos = Z_corr / col_norms
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
        model = KMeans(n_clusters=K, init='k-means++', n_init=10, max_iter=25, random_state=random_state)
        model.fit(data)
        km_centroids = model.cluster_centers_
        print("sklearn.KMeans initialization complete.")
        return km_centroids

    @staticmethod
    def static_init_cluster(harmony, cluster_fn):
        harmony.Y = cluster_fn(harmony.Z_cos.T, harmony.K).T
        harmony.Y = harmony.Y / np.linalg.norm(harmony.Y, ord=2, axis=0)
        harmony.dist_mat = 2 * (1 - np.dot(harmony.Y.T, harmony.Z_cos))
        harmony.R = -harmony.dist_mat
        harmony.R = harmony.R / harmony.sigma[:, None]
        harmony.R -= np.max(harmony.R, axis=0)
        harmony.R = np.exp(harmony.R)
        harmony.R = harmony.R / np.sum(harmony.R, axis=0)
        harmony.E = np.outer(np.sum(harmony.R, axis=1), harmony.Pr_b)
        harmony.O = np.inner(harmony.R, harmony.Phi)
        harm_Ours.static_compute_objective(harmony)
        harmony.objective_harmony.append(harmony.objective_kmeans[-1])

    @staticmethod
    def static_compute_objective(harmony):
        kmeans_error = np.sum(np.multiply(harmony.R, harmony.dist_mat))
        _entropy = np.sum(harm_Ours.safe_entropy(harmony.R) * harmony.sigma[:, np.newaxis])
        x = (harmony.R * harmony.sigma[:, np.newaxis])
        y = np.tile(harmony.theta[:, np.newaxis], harmony.K).T
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
                    print("Converged after {} iteration{}".format(i, 's' if i > 1 else ''))
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
        harmony._scale_dist = harmony._scale_dist / harmony.sigma[:, None]
        harmony._scale_dist -= np.max(harmony._scale_dist, axis=0)
        harmony._scale_dist = np.exp(harmony._scale_dist)
        update_order = np.arange(harmony.N)
        np.random.shuffle(update_order)
        n_blocks = int(np.ceil(1 / harmony.block_size))
        blocks = np.array_split(update_order, n_blocks)
        for b in blocks:
            harmony.E -= np.outer(np.sum(harmony.R[:, b], axis=1), harmony.Pr_b)
            harmony.O -= np.dot(harmony.R[:, b], harmony.Phi[:, b].T)
            harmony.R[:, b] = harmony._scale_dist[:, b]
            harmony.R[:, b] = np.multiply(
                harmony.R[:, b],
                np.dot(np.power((harmony.E + 1) / (harmony.O + 1), harmony.theta), harmony.Phi[:, b]),
            )
            harmony.R[:, b] = harmony.R[:, b] / np.linalg.norm(harmony.R[:, b], ord=1, axis=0)
            harmony.E += np.outer(np.sum(harmony.R[:, b], axis=1), harmony.Pr_b)
            harmony.O += np.dot(harmony.R[:, b], harmony.Phi[:, b].T)
        return 0

    @staticmethod
    def static_check_convergence(harmony, i_type):
        if i_type == 0:
            okl = len(harmony.objective_kmeans)
            obj_old = 0.0
            obj_new = 0.0
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

# -----------------------------
# Experiment runner implementing the required analysis perspective
# -----------------------------

def run_experiment(
    data_dir='dataset',
    covariate_choice=['donor'],
    perplexity=30,
    n_bins=20,
    random_state=0,
    max_iter_harmony=6,
    max_iter_kmeans=20,
):
    # Load data
    meta_fp = os.path.join(data_dir, 'pbmc_3500_meta.tsv.gz')
    pcs_fp = os.path.join(data_dir, 'pbmc_3500_pcs.tsv.gz')
    if not os.path.exists(meta_fp) or not os.path.exists(pcs_fp):
        raise FileNotFoundError(f"Required dataset files not found under '{data_dir}'.\nExpecting 'pbmc_3500_meta.tsv.gz' and 'pbmc_3500_pcs.tsv.gz'.")

    meta_data = pd.read_csv(meta_fp, sep='\t')
    data_mat = pd.read_csv(pcs_fp, sep='\t')

    # Ensure shape: features x cells
    if data_mat.shape[1] != meta_data.shape[0]:
        data_mat = data_mat.T

    N = meta_data.shape[0]
    # Determine number of clusters (nclust)
    nclust = int(min(max(2, round(N / 30.0)), 100))

    # Construct Phi and Phi_moe
    if isinstance(covariate_choice, str):
        covariate_choice = [covariate_choice]

    phi = pd.get_dummies(meta_data[covariate_choice]).to_numpy().T
    phi_n = meta_data[covariate_choice].describe().loc['unique'].to_numpy().astype(int)

    # theta and lamb defaults like reference
    theta = np.repeat([1] * len(phi_n), phi_n)
    lamb = np.repeat([1] * len(phi_n), phi_n)

    N_b = phi.sum(axis=1)
    Pr_b = N_b / N

    lamb_mat = np.diag(np.insert(lamb, 0, 0))
    phi_moe = np.vstack((np.repeat(1, N), phi))

    sigma = 0.1

    # Run Harmony and MoE-Harmony (reproduced implementations)
    np.random.seed(random_state)

    print("Running baseline Harmony...")
    hh = Harmony(
        data_mat.values if isinstance(data_mat, pd.DataFrame) else data_mat,
        phi,
        phi_moe,
        Pr_b,
        np.repeat(sigma, nclust) if isinstance(sigma, float) else sigma,
        theta,
        max_iter_harmony,
        max_iter_kmeans,
        1e-5,
        1e-4,
        nclust,
        0.05,
        lamb_mat,
        False,
        random_state,
        'kmeans'
    )

    print("Running MoE-Harmony (ours)...")
    ho = harm_Ours(
        data_mat.values if isinstance(data_mat, pd.DataFrame) else data_mat,
        phi,
        phi_moe,
        Pr_b,
        np.repeat(sigma, nclust) if isinstance(sigma, float) else sigma,
        theta,
        max_iter_harmony,
        max_iter_kmeans,
        1e-5,
        1e-4,
        nclust,
        0.05,
        lamb_mat,
        False,
        random_state,
        'kmeans'
    )

    # Compute LISI (local covariate diversity) on integrated embeddings (Z_cos or original PCs)
    print("Computing local covariate diversity (LISI) on original PCA space")
    X = (data_mat.values if isinstance(data_mat, pd.DataFrame) else data_mat).T
    lisi = compute_lisi(X, meta_data, covariate_choice, perplexity=perplexity)
    # If multiple covariates chosen, take first
    diversity = lisi[:, 0]

    # For each method extract per-cell metrics
    def extract_metrics(obj, method_name):
        # responsibilities R: K x N
        R = obj.R.copy()
        # O matrix: K x B (expert->batch association)
        O = obj.O.copy()
        # Z_orig and Z_corr stored (normalized inside classes differently)
        Z_orig = obj.Z_orig
        Z_corr = obj.Z_corr
        # per-cell correction magnitude: L2 norm of difference
        corr_vec = Z_corr - Z_orig
        corr_mag = np.linalg.norm(corr_vec, axis=0)

        # For each cell, identify its batch column index (in phi). phi is B x N
        # phi is one-hot; find index of category where phi[:,i]==1
        # There might be multiple covariates stacked; phi was built from get_dummies of chosen vars
        phi_local = phi
        if phi_local.shape[0] == 0:
            raise ValueError("No covariates found in phi")
        # For cells with multiple active categories (unlikely), use weighted combination
        # map cell's covariate one-hot to an expert-profile vector m = O @ phi[:,i]
        N = phi_local.shape[1]
        m_vecs = np.zeros((R.shape[0], N))
        for i in range(N):
            p = phi_local[:, i]
            m = np.dot(O, p)
            m_vecs[:, i] = m
        # alignment: cosine similarity between R[:,i] and m_vecs[:,i]
        align = np.zeros(N)
        for i in range(N):
            a = R[:, i]
            b = m_vecs[:, i]
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0:
                align[i] = 0.0
            else:
                align[i] = np.dot(a, b) / denom
        # ensure alignment in [-1,1]
        align = np.clip(align, -1, 1)

        # Compose dataset per cell
        df = pd.DataFrame({
            'diversity': diversity,
            'alignment': align,
            'correction_magnitude': corr_mag,
        })
        df['product'] = df['diversity'] * df['alignment']
        df['method'] = method_name
        return df

    df_ours = extract_metrics(ho, 'MoE-Harmony')
    df_baseline = extract_metrics(hh, 'Harmony')

    # Combine and compute correlations
    def compute_and_save(df, method_name):
        # Pearson correlation between product and correction magnitude
        x = df['product'].values
        y = df['correction_magnitude'].values
        # If constant arrays, pearsonr fails; guard
        try:
            r, pval = pearsonr(x, y)
        except Exception:
            r, pval = np.nan, np.nan
        print(f"Method {method_name}: Pearson r(product, correction_magnitude) = {r:.4f}, p = {pval:.4e}")

        # 2D binning over diversity (x) and alignment (y)
        x_vals = df['diversity'].values
        y_vals = df['alignment'].values
        z_vals = df['correction_magnitude'].values

        # define bin edges
        x_min, x_max = np.nanmin(x_vals), np.nanmax(x_vals)
        y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)
        # Small padding to include endpoints
        x_edges = np.linspace(x_min, x_max, n_bins + 1)
        y_edges = np.linspace(y_min, y_max, n_bins + 1)

        # compute mean z per bin
        grid = np.full((n_bins, n_bins), np.nan)
        counts = np.zeros((n_bins, n_bins), dtype=int)
        for xi, yi, zi in zip(x_vals, y_vals, z_vals):
            # find bin index
            ix = np.searchsorted(x_edges, xi, side='right') - 1
            iy = np.searchsorted(y_edges, yi, side='right') - 1
            if ix < 0 or ix >= n_bins or iy < 0 or iy >= n_bins:
                continue
            if np.isnan(zi):
                continue
            if np.isnan(grid[ix, iy]):
                grid[ix, iy] = zi
            else:
                grid[ix, iy] += zi
            counts[ix, iy] += 1
        # average
        avg_grid = np.full_like(grid, np.nan, dtype=float)
        for i in range(n_bins):
            for j in range(n_bins):
                if counts[i, j] > 0:
                    avg_grid[i, j] = grid[i, j] / counts[i, j]

        # Save per-cell data and grid data as CSV
        per_cell_fp = os.path.join(RESULT_DIR, f'per_cell_data_{method_name.replace(" ", "_")}.csv')
        grid_fp = os.path.join(RESULT_DIR, f'binned_grid_{method_name.replace(" ", "_")}.csv')

        # per-cell
        df.to_csv(per_cell_fp, index=False)

        # grid: save with bin centers for plotting convenience
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
        # flatten
        grid_flat = []
        for i in range(n_bins):
            for j in range(n_bins):
                grid_flat.append({'x_center': x_centers[i], 'y_center': y_centers[j], 'mean_correction': avg_grid[i, j], 'count': counts[i, j]})
        pd.DataFrame(grid_flat).to_csv(grid_fp, index=False)

        return r, pval

    r_ours, p_ours = compute_and_save(df_ours, 'MoE-Harmony')
    r_baseline, p_baseline = compute_and_save(df_baseline, 'Harmony')

    # Save a small summary CSV (this is still plotting data: single-row summary helps annotate plots)
    summary = pd.DataFrame({
        'method': ['MoE-Harmony', 'Harmony'],
        'pearson_r': [r_ours, r_baseline],
        'p_value': [p_ours, p_baseline]
    })
    summary_fp = os.path.join(RESULT_DIR, 'correlation_summary.csv')
    summary.to_csv(summary_fp, index=False)

    print(f"Saved plotting data to {RESULT_DIR}")
    return {
        'per_cell_files': [
            os.path.join(RESULT_DIR, 'per_cell_data_MoE-Harmony.csv'),
            os.path.join(RESULT_DIR, 'per_cell_data_Harmony.csv')
        ],
        'grid_files': [
            os.path.join(RESULT_DIR, 'binned_grid_MoE-Harmony.csv'),
            os.path.join(RESULT_DIR, 'binned_grid_Harmony.csv')
        ],
        'summary': summary_fp,
    }


if __name__ == '__main__':
    # Execute full experiment
    res = run_experiment(data_dir='dataset', covariate_choice=['donor'], perplexity=30, n_bins=20, random_state=0)
    print('Experiment finished. Files created:')
    for k, v in res.items():
        print(k, ':', v)
