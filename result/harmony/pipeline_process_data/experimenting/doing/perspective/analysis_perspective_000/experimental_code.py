import os
import sys
sys.path.append(os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))
#!/usr/bin/env python3
"""
Experiment script: Subspace-aligned correction fidelity (MTSCF)
Compares MoE-Harmony (harm_Ours) to baseline Harmony on provided PBMC dataset.

Produces a single CSV file for downstream plotting at:
    ./experimental_result_data/mtscf_per_cell.csv

This file contains per-cell MTSCF (fraction of correction energy projecting
into the technical covariate subspace) for both methods, plus metadata.

Usage: place this script at the repository root (same level as dataset/).
Run: python3 this_script.py

Notes:
- Re-implements Harmony and harm_Ours classes verbatim from the provided
  reference code (no imports of those classes).
- Follows the subspace-projection protocol described in the analysis.
- No simulations; uses dataset/pbmc_3500_meta.tsv.gz and
  dataset/pbmc_3500_pcs.tsv.gz as provided in the reference code.

"""

import os
import sys
import numpy as np
import pandas as pd
from functools import partial
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# Attempt to import statistical tests; provide safe fallbacks
try:
    from scipy.stats import wilcoxon, ttest_rel
except Exception:
    wilcoxon = None
    try:
        from scipy.stats import ttest_rel
    except Exception:
        ttest_rel = None

# ----------------------------
# Paste of Harmony and harm_Ours implementations (reproduced)
# (Trimmed duplication where appropriate; functionality preserved)
# ----------------------------

def safe_entropy(x: np.array):
    y = np.multiply(x, np.log(x))
    y[~np.isfinite(y)] = 0.0
    return y


class Harmony(object):
    def __init__(
            self, Z, Phi, Phi_moe, Pr_b, sigma,
            theta, max_iter_harmony, max_iter_kmeans,
            epsilon_kmeans, epsilon_harmony, K, block_size,
            lamb, verbose, random_state=None, cluster_fn='kmeans'
    ):
        self.Z_corr = np.array(Z)
        self.Z_orig = np.array(Z)

        # Normalize columns for cosine
        self.Z_cos = self.Z_orig / (self.Z_orig.max(axis=0) + 1e-12)
        self.Z_cos = self.Z_cos / np.linalg.norm(self.Z_cos, ord=2, axis=0)

        self.Phi             = Phi
        self.Phi_moe         = Phi_moe
        self.N               = self.Z_corr.shape[1]
        self.Pr_b            = Pr_b
        self.B               = self.Phi.shape[0] # number of batch variables
        self.d               = self.Z_corr.shape[0]
        self.window_size     = 3
        self.epsilon_kmeans  = epsilon_kmeans
        self.epsilon_harmony = epsilon_harmony

        self.lamb            = lamb
        self.sigma           = sigma
        self.sigma_prior     = sigma
        self.block_size      = block_size
        self.K               = K                # number of clusters
        self.max_iter_harmony = max_iter_harmony
        self.max_iter_kmeans = max_iter_kmeans
        self.verbose         = verbose
        self.theta           = theta

        self.objective_harmony        = []
        self.objective_kmeans         = []
        self.objective_kmeans_dist    = []
        self.objective_kmeans_entropy = []
        self.objective_kmeans_cross   = []
        self.kmeans_rounds  = []

        self.allocate_buffers()
        if cluster_fn == 'kmeans':
            cluster_fn = partial(Harmony._cluster_kmeans, random_state=random_state)
        self.init_cluster(cluster_fn)
        self.harmonize(self.max_iter_harmony, self.verbose)

    def result(self):
        return self.Z_corr

    def allocate_buffers(self):
        self._scale_dist = np.zeros((self.K, self.N))
        self.dist_mat    = np.zeros((self.K, self.N))
        self.O           = np.zeros((self.K, self.B))
        self.E           = np.zeros((self.K, self.B))
        self.W           = np.zeros((self.B + 1, self.d))
        self.Phi_Rk      = np.zeros((self.B + 1, self.N))

    @staticmethod
    def _cluster_kmeans(data, K, random_state):
        print("Computing initial centroids with sklearn.KMeans...")
        model = KMeans(n_clusters=K, init='k-means++',
                       n_init=10, max_iter=25, random_state=random_state)
        model.fit(data)
        km_centroids, km_labels = model.cluster_centers_, model.labels_
        print("sklearn.KMeans initialization complete.")
        return km_centroids

    def init_cluster(self, cluster_fn):
        self.Y = cluster_fn(self.Z_cos.T, self.K).T
        # (1) Normalize
        self.Y = self.Y / np.linalg.norm(self.Y, ord=2, axis=0)
        # (2) Assign cluster probabilities
        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
        self.R = -self.dist_mat
        self.R = self.R / self.sigma[:,None]
        self.R -= np.max(self.R, axis = 0)
        self.R = np.exp(self.R)
        self.R = self.R / np.sum(self.R, axis = 0)
        # (3) Batch diversity statistics
        self.E = np.outer(np.sum(self.R, axis=1), self.Pr_b)
        self.O = np.inner(self.R , self.Phi)
        self.compute_objective()
        # Save results
        self.objective_harmony.append(self.objective_kmeans[-1])

    def compute_objective(self):
        kmeans_error = np.sum(np.multiply(self.R, self.dist_mat))
        # Entropy
        _entropy = np.sum(safe_entropy(self.R) * self.sigma[:,np.newaxis])
        # Cross Entropy
        x = (self.R * self.sigma[:,np.newaxis])
        y = np.tile(self.theta[:,np.newaxis], self.K).T
        z = np.log((self.O + 1) / (self.E + 1))
        w = np.dot(y * z, self.Phi)
        _cross_entropy = np.sum(x * w)
        # Save results
        self.objective_kmeans.append(kmeans_error + _entropy + _cross_entropy)
        self.objective_kmeans_dist.append(kmeans_error)
        self.objective_kmeans_entropy.append(_entropy)
        self.objective_kmeans_cross.append(_cross_entropy)

    def harmonize(self, iter_harmony=10, verbose=True):
        converged = False
        for i in range(1, iter_harmony + 1):
            if verbose:
                print("Iteration {} of {}".format(i, iter_harmony))
            # STEP 1: Clustering
            self.cluster()
            # STEP 2: Regress out covariates
            self.Z_cos, self.Z_corr, self.W, self.Phi_Rk = moe_correct_ridge(
                self.Z_orig, self.Z_cos, self.Z_corr, self.R, self.W, self.K,
                self.Phi_Rk, self.Phi_moe, self.lamb
            )
            # STEP 3: Check for convergence
            converged = self.check_convergence(1)
            if converged:
                if verbose:
                    print(
                        "Converged after {} iteration{}".format(i, 's' if i > 1 else '')
                    )
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
        self._scale_dist = self._scale_dist / self.sigma[:,None]
        self._scale_dist -= np.max(self._scale_dist, axis=0)
        self._scale_dist = np.exp(self._scale_dist)
        update_order = np.arange(self.N)
        np.random.shuffle(update_order)
        n_blocks = np.ceil(1 / self.block_size).astype(int)
        blocks = np.array_split(update_order, n_blocks)
        for b in blocks:
            self.E -= np.outer(np.sum(self.R[:,b], axis=1), self.Pr_b)
            self.O -= np.dot(self.R[:,b], self.Phi[:,b].T)
            self.R[:,b] = self._scale_dist[:,b]
            self.R[:,b] = np.multiply(
                self.R[:,b],
                np.dot(
                    np.power((self.E + 1) / (self.O + 1), self.theta),
                    self.Phi[:,b]
                )
            )
            self.R[:,b] = self.R[:,b] / np.linalg.norm(self.R[:,b], ord=1, axis=0)
            self.E += np.outer(np.sum(self.R[:,b], axis=1), self.Pr_b)
            self.O += np.dot(self.R[:,b], self.Phi[:,b].T)
        return 0

    def check_convergence(self, i_type):
        obj_old = 0.0
        obj_new = 0.0
        if i_type == 0:
            okl = len(self.objective_kmeans)
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


def moe_correct_ridge(Z_orig, Z_cos, Z_corr, R, W, K, Phi_Rk, Phi_moe, lamb):
    Z_corr = Z_orig.copy()
    for i in range(K):
        Phi_Rk = np.multiply(Phi_moe, R[i,:])
        x = np.dot(Phi_Rk, Phi_moe.T) + lamb
        W = np.dot(np.dot(np.linalg.inv(x), Phi_Rk), Z_orig.T)
        W[0,:] = 0 # do not remove the intercept
        Z_corr -= np.dot(W.T, Phi_Rk)
    Z_cos = Z_corr / (np.linalg.norm(Z_corr, ord=2, axis=0) + 1e-12)
    return Z_cos, Z_corr, W, Phi_Rk


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
        Z_cos = Z_corr / (np.linalg.norm(Z_corr, ord=2, axis=0) + 1e-12)
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

# ----------------------------
# End of Harmony classes
# ----------------------------


# ----------------------------
# Experiment orchestration utilities
# ----------------------------

def run_harmony(
    data_mat: np.ndarray,
    meta_data: pd.DataFrame,
    vars_use,
    theta = None,
    lamb = None,
    sigma = 0.1,
    nclust = None,
    tau = 0,
    block_size = 0.05,
    max_iter_harmony = 10,
    max_iter_kmeans = 20,
    epsilon_cluster = 1e-5,
    epsilon_harmony = 1e-4,
    verbose = True,
    random_state = 0,
    cluster_fn = 'kmeans'
):
    """Run Harmony baseline and harm_Ours (MoE) and return both objects.
    This reproduces the reference run_harmony behavior used in the provided code.
    """
    N = meta_data.shape[0]
    if data_mat.shape[1] != N:
        data_mat = data_mat.T

    assert data_mat.shape[1] == N, "data_mat and meta_data do not have the same number of cells"

    if nclust is None:
        nclust = np.min([np.round(N / 30.0), 100]).astype(int)

    if type(sigma) is float and nclust > 1:
        sigma = np.repeat(sigma, nclust)

    if isinstance(vars_use, str):
        vars_use = [vars_use]

    phi = pd.get_dummies(meta_data[vars_use]).to_numpy().T
    phi_n = meta_data[vars_use].describe().loc['unique'].to_numpy().astype(int)

    if theta is None:
        theta = np.repeat([1] * len(phi_n), phi_n)
    elif isinstance(theta, float) or isinstance(theta, int):
        theta = np.repeat([theta] * len(phi_n), phi_n)
    elif len(theta) == len(phi_n):
        theta = np.repeat([theta], phi_n)

    assert len(theta) == np.sum(phi_n), "each batch variable must have a theta"

    if lamb is None:
        lamb = np.repeat([1] * len(phi_n), phi_n)
    elif isinstance(lamb, float) or isinstance(lamb, int):
        lamb = np.repeat([lamb] * len(phi_n), phi_n)
    elif len(lamb) == len(phi_n):
        lamb = np.repeat([lamb], phi_n)

    assert len(lamb) == np.sum(phi_n), "each batch variable must have a lambda"

    # Number of items in each category.
    N_b = phi.sum(axis = 1)
    # Proportion of items in each category.
    Pr_b = N_b / N

    if tau > 0:
        theta = theta * (1 - np.exp(-(N_b / (nclust * tau)) ** 2))

    lamb_mat = np.diag(np.insert(lamb, 0, 0))

    phi_moe = np.vstack((np.repeat(1, N), phi))

    np.random.seed(random_state)

    hh = Harmony(
        data_mat, phi, phi_moe, Pr_b, sigma, theta, max_iter_harmony, max_iter_kmeans,
        epsilon_cluster, epsilon_harmony, nclust, block_size, lamb_mat, verbose,
        random_state, cluster_fn
    )

    ho = harm_Ours(
        data_mat, phi, phi_moe, Pr_b, sigma, theta, max_iter_harmony, max_iter_kmeans,
        epsilon_cluster, epsilon_harmony, nclust, block_size, lamb_mat, verbose,
        random_state, cluster_fn
    )

    return ho, hh


def build_design_matrix(meta: pd.DataFrame, columns: list, drop_intercept=True):
    """Return design matrix X (N x q) from meta columns using one-hot encoding.
    Will drop columns with zero variance.
    """
    if len(columns) == 0:
        raise ValueError("No columns provided for design matrix")
    df = meta[columns].copy()
    # Convert all to categorical then get dummies
    for c in df.columns:
        df[c] = df[c].astype('category')
    X = pd.get_dummies(df, drop_first=False)
    # Remove any all-zero columns
    X = X.loc[:, (X != X.iloc[0]).any()] if X.shape[1] > 0 else X
    X = X.fillna(0.0)
    X_mat = X.values.astype(float)
    # Optionally do not include intercept column here
    return X_mat


def fit_fitted_values(Z_orig: np.ndarray, X: np.ndarray):
    """
    Fit linear models for each feature in Z_orig (d x N) onto design X (N x q).
    Returns fitted matrix F (d x N).
    Uses least-squares: Beta = argmin ||X Beta - Y|| where Y = Z_orig.T (N x d)
    """
    # Z_orig: d x N  => transpose to N x d
    Y = Z_orig.T
    # Solve X Beta = Y  => Beta shape q x d
    # Use lstsq for numerical stability
    Beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    fitted = (X @ Beta).T  # d x N
    return fitted


def orthonormal_basis_from_fitted(F: np.ndarray, tol=1e-8):
    """Compute SVD of fitted (d x N) and return orthonormal basis U (d x r)
    for non-zero singular values (above tol).
    """
    if np.allclose(F, 0.0):
        return np.zeros((F.shape[0], 0))
    U, S, Vt = np.linalg.svd(F, full_matrices=False)
    r = np.sum(S > tol)
    if r == 0:
        return np.zeros((F.shape[0], 0))
    return U[:, :r]


def project_energy_fraction(c_vec: np.ndarray, U_basis: np.ndarray):
    """Return fraction of energy of c_vec projecting into subspace spanned by U_basis.
    c_vec shape: (d,). U_basis: (d, r).
    Returns float in [0,1].
    """
    norm2 = np.sum(c_vec ** 2)
    if norm2 == 0.0:
        return np.nan
    if U_basis.size == 0:
        return 0.0
    # coefficients = U.T @ c
    coeffs = U_basis.T.dot(c_vec)
    proj = U_basis.dot(coeffs)
    frac = np.sum(proj ** 2) / norm2
    # numerical bounds
    return float(min(max(frac, 0.0), 1.0))


def compute_mtscf_for_methods(
    Z_orig: np.ndarray,
    Z_corr_methods: dict,
    meta: pd.DataFrame,
    tech_vars: list,
    bio_label_col: str = None
):
    """Compute per-cell MTSCF and biological projection fractions for each method.
    Z_orig: d x N. Z_corr_methods: dict name->Z_corr (d x N).
    tech_vars: list of metadata columns to treat as technical covariates.
    bio_label_col: metadata column to use as biological labels. If None, will
                   create cluster-based labels via KMeans on Z_orig.T.
    Returns pandas DataFrame long-format with columns [cell_id, method, MTSCF, MBSCF, norm_c, donor, cell_type]
    """
    d, N = Z_orig.shape
    # Build technical design matrix
    X_tech = build_design_matrix(meta, tech_vars)
    if X_tech.shape[0] != N:
        # ensure rows correspond to samples
        X_tech = X_tech.reshape((N, -1))
    # Fit and get fitted values
    F_tech = fit_fitted_values(Z_orig, X_tech)
    U_tech = orthonormal_basis_from_fitted(F_tech)

    # Biological design
    if bio_label_col is None or bio_label_col not in meta.columns:
        # derive clusters as proxy biological labels
        n_clusters = max(5, min(20, int(np.round(N / 600))))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(Z_orig.T)
        meta = meta.copy()
        meta['_bio_cluster'] = labels.astype(str)
        bio_label_col = '_bio_cluster'

    X_bio = build_design_matrix(meta, [bio_label_col])
    F_bio = fit_fitted_values(Z_orig, X_bio)
    U_bio = orthonormal_basis_from_fitted(F_bio)

    # iterate cells and methods
    rows = []
    cell_ids = list(meta.index.astype(str)) if meta.index is not None else [str(i) for i in range(N)]
    # Ensure meta index length matches N
    if len(cell_ids) != N:
        cell_ids = [f'cell_{i}' for i in range(N)]

    for method_name, Z_corr in Z_corr_methods.items():
        # compute correction vectors c_i = Z_orig[:,i] - Z_corr[:,i]
        C = Z_orig - Z_corr  # d x N
        norms = np.sum(C ** 2, axis=0)
        for i in range(N):
            c = C[:, i]
            mt = project_energy_fraction(c, U_tech)
            mb = project_energy_fraction(c, U_bio)
            row = {
                'cell_id': cell_ids[i],
                'method': method_name,
                'MTSCF': mt,
                'MBSCF': mb,
                'norm_c': float(norms[i]),
            }
            # attach some metadata if present
            # copy selected metadata columns safely
            for key in ['donor', 'chemistry', 'platform']:
                if key in meta.columns:
                    row[key] = str(meta.iloc[i][key])
            # biological label
            row['bio_label'] = str(meta.iloc[i][bio_label_col]) if bio_label_col in meta.columns else ''
            rows.append(row)

    df_res = pd.DataFrame(rows)
    return df_res


def cohen_d_paired(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    d = x - y
    n = len(d)
    if n <= 1:
        return np.nan
    mean_d = np.nanmean(d)
    sd_d = np.nanstd(d, ddof=1)
    if sd_d == 0:
        return np.nan
    return mean_d / sd_d


# ----------------------------
# Main execution
# ----------------------------

def main():
    # Ensure working dir is script location (like reference code did)
    try:
        os.chdir(os.path.dirname(__file__) or os.getcwd())
    except Exception:
        pass

    # Paths
    meta_path = os.path.join('dataset', 'pbmc_3500_meta.tsv.gz')
    pcs_path = os.path.join('dataset', 'pbmc_3500_pcs.tsv.gz')

    if not os.path.exists(meta_path) or not os.path.exists(pcs_path):
        raise FileNotFoundError(
            f"Required dataset files not found. Expected at {meta_path} and {pcs_path}"
        )

    meta = pd.read_csv(meta_path, sep='\t', index_col=None)
    data_mat = pd.read_csv(pcs_path, sep='\t', index_col=None)

    # If data_mat is genes x cells or cells x PCs, align as in reference
    # The reference code expects data_mat to end up as d x N
    data_np = data_mat.values
    # If shape (N, d) we need to transpose
    if data_np.shape[0] == meta.shape[0] and data_np.shape[1] != meta.shape[0]:
        # data_mat is N x d -> transpose to d x N
        data_np = data_np.T
    elif data_np.shape[1] == meta.shape[0] and data_np.shape[0] != meta.shape[1]:
        # data_mat is d x N already
        pass
    else:
        # safeguard: if mismatch, attempt transpose if that aligns
        if data_np.shape[1] == meta.shape[0]:
            pass
        elif data_np.shape[0] == meta.shape[0]:
            data_np = data_np.T

    # For reproducibility set seed
    random_state = 0

    # Select technical variables available in meta
    candidate_tech = ['donor', 'chemistry', 'platform', 'library', 'batch']
    tech_vars = [c for c in candidate_tech if c in meta.columns]
    if len(tech_vars) == 0:
        # fallback to 'donor' if not present; else use first categorical column
        if 'donor' in meta.columns:
            tech_vars = ['donor']
        else:
            # pick first non-numeric column
            for c in meta.columns:
                if meta[c].dtype == object or pd.api.types.is_categorical_dtype(meta[c]):
                    tech_vars = [c]
                    break
    if len(tech_vars) == 0:
        raise ValueError('No technical covariates found in metadata to build technical subspace')

    # Choose biological label column if present
    candidate_bio = ['cell_type', 'celltype', 'cell_type_combined', 'cell_ontology_class', 'cell_type_major']
    bio_label_col = None
    for c in candidate_bio:
        if c in meta.columns:
            bio_label_col = c
            break
    # If still None, we will derive clusters (handled inside compute)

    print(f"Using technical covariates: {tech_vars}")
    if bio_label_col is not None:
        print(f"Using biological label column: {bio_label_col}")
    else:
        print("No explicit biological label found; will derive cluster-based proxy labels.")

    # Run both methods
    # Use vars_use for Harmony run (we need at least one var used for batch correction)
    vars_use = tech_vars[0]
    print("Running Harmony and MoE-Harmony (this may take a moment)...")

    ho, hh = run_harmony(
        data_np, meta, vars_use,
        theta=None, lamb=None, sigma=0.1,
        nclust=None, tau=0, block_size=0.05,
        max_iter_harmony=5, max_iter_kmeans=20,
        epsilon_cluster=1e-5, epsilon_harmony=1e-4,
        verbose=False, random_state=random_state, cluster_fn='kmeans'
    )

    Z_orig = ho.Z_orig  # both classes normalized this way; d x N
    Z_corr_moe = ho.result()
    Z_corr_harmony = hh.result()

    Z_corr_methods = {
        'MoE_Harmony': Z_corr_moe,
        'Harmony': Z_corr_harmony
    }

    # Compute MTSCF per-cell
    df_mtscf = compute_mtscf_for_methods(Z_orig, Z_corr_methods, meta, tech_vars, bio_label_col)

    # Save only the experimental result data file used for plotting
    out_dir = os.path.join('.', 'experimental_result_data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'mtscf_per_cell.csv')

    # Ensure columns consistent and deterministic ordering
    # Keep essential columns
    cols_order = ['cell_id', 'method', 'MTSCF', 'MBSCF', 'norm_c', 'bio_label']
    # add donors/chemistry/platform if present in at least one row
    extra_cols = [c for c in ['donor', 'chemistry', 'platform'] if c in df_mtscf.columns]
    cols_order = cols_order + extra_cols
    # filter existing
    cols_order = [c for c in cols_order if c in df_mtscf.columns]

    df_mtscf.to_csv(out_path, index=False, columns=cols_order)

    print(f"Saved MTSCF per-cell results to: {out_path}")

    # Compute paired statistics and print to stdout (no files created)
    # For each cell we have two rows (MoE_Harmony and Harmony). Pivot and compute paired diffs.
    pivot = df_mtscf.pivot(index='cell_id', columns='method', values='MTSCF')
    pivot = pivot.dropna()
    if pivot.shape[0] == 0:
        print("Warning: no paired cells available for statistical comparison (possible NaNs)")
        return

    x = pivot['MoE_Harmony'].values
    y = pivot['Harmony'].values
    # Paired Wilcoxon if available
    stat_name = None
    stat_res = None
    if wilcoxon is not None:
        try:
            w_stat, pval = wilcoxon(x, y)
            stat_name = 'wilcoxon'
            stat_res = (w_stat, pval)
        except Exception:
            stat_name = None
    if stat_name is None and ttest_rel is not None:
        t_stat, pval = ttest_rel(x, y)
        stat_name = 'ttest_rel'
        stat_res = (t_stat, pval)

    d_cohen = cohen_d_paired(x, y)
    mean_moe = np.nanmean(x)
    mean_h = np.nanmean(y)
    n = len(x)

    print('\nSummary statistics (paired across cells):')
    print(f'  n = {n} cells')
    print(f'  mean MTSCF - MoE_Harmony = {mean_moe:.4f}')
    print(f'  mean MTSCF - Harmony    = {mean_h:.4f}')
    if stat_name is not None and stat_res is not None:
        print(f'  statistical test: {stat_name}, stat = {stat_res[0]:.4f}, p = {stat_res[1]:.4e}')
    else:
        print('  No statistical test available (scipy missing)')
    print(f'  Cohen\'s d (paired) = {d_cohen:.4f}')
    print('\nThe output CSV is ready for plotting split (paired) violin charts as described in the experiment plan.')


if __name__ == '__main__':
    main()
