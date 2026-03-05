import numpy as np
import  os
os.chdir(os.path.dirname(__file__))

import numpy as np
import pandas as pd

import numpy as np
from functools import partial
from sklearn.cluster import KMeans


import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Iterable


def compute_lisi(
    X: np.array,
    metadata: pd.DataFrame,
    label_colnames: Iterable[str],
    perplexity: float=30
):
    """Compute the Local Inverse Simpson Index (LISI) for each column in metadata.

    LISI is a statistic computed for each item (row) in the data matrix X.

    The following example may help to interpret the LISI values.

    Suppose one of the columns in metadata is a categorical variable with 3 categories.

        - If LISI is approximately equal to 3 for an item in the data matrix,
          that means that the item is surrounded by neighbors from all 3
          categories.

        - If LISI is approximately equal to 1, then the item is surrounded by
          neighbors from 1 category.
    
    The LISI statistic is useful to evaluate whether multiple datasets are
    well-integrated by algorithms such as Harmony [1].

    [1]: Korsunsky et al. 2019 doi: 10.1038/s41592-019-0619-0
    """
    n_cells = metadata.shape[0]
    n_labels = len(label_colnames)
    # We need at least 3 * n_neigbhors to compute the perplexity
    knn = NearestNeighbors(n_neighbors = perplexity * 3, algorithm = 'kd_tree').fit(X)
    distances, indices = knn.kneighbors(X)
    # Don't count yourself
    indices = indices[:,1:]
    distances = distances[:,1:]
    # Save the result
    lisi_df = np.zeros((n_cells, n_labels))
    for i, label in enumerate(label_colnames):
        labels = pd.Categorical(metadata[label])
        n_categories = len(labels.categories)
        simpson = compute_simpson(distances.T, indices.T, labels, n_categories, perplexity)
        lisi_df[:,i] = 1 / simpson
    return lisi_df


def compute_simpson(
    distances: np.ndarray,
    indices: np.ndarray,
    labels: pd.Categorical,
    n_categories: int,
    perplexity: float,
    tol: float=1e-5
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
        P = np.exp(-distances[:,i] * beta)
        P_sum = np.sum(P)
        if P_sum == 0:
            H = 0
            P = np.zeros(distances.shape[0])
        else:
            H = np.log(P_sum) + beta * np.sum(distances[:,i] * P) / P_sum
            P = P / P_sum
        Hdiff = H - logU
        n_tries = 50
        for t in range(n_tries):
            # Stop when we reach the tolerance
            if abs(Hdiff) < tol:
                break
            # Update beta
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
            # Compute Hdiff
            P = np.exp(-distances[:,i] * beta)
            P_sum = np.sum(P)
            if P_sum == 0:
                H = 0
                P = np.zeros(distances.shape[0])
            else:
                H = np.log(P_sum) + beta * np.sum(distances[:,i] * P) / P_sum
                P = P / P_sum
            Hdiff = H - logU
        # distancesefault value
        if H == 0:
            simpson[i] = -1
        # Simpson's index
        for label_category in labels.categories:
            ix = indices[:,i]
            q = labels[ix] == label_category
            if np.any(q):
                P_sum = np.sum(P[q])
                simpson[i] += P_sum * P_sum
    return simpson


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
        # Start with cluster centroids
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
            # self.moe_correct_ridge()
            self.Z_cos, self.Z_corr, self.W, self.Phi_Rk = moe_correct_ridge(
                self.Z_orig, self.Z_cos, self.Z_corr, self.R, self.W, self.K,
                self.Phi_Rk, self.Phi_moe, self.lamb
            )
            # STEP 3: Check for convergence
            converged = self.check_convergence(1)
            if converged:
                if verbose:
                    print(
                        "Converged after {} iteration{}"
                        .format(i, 's' if i > 1 else '')
                    )
                break
        if verbose and not converged:
            print("Stopped before convergence")
        return 0

    def cluster(self):
        # Z_cos has changed
        # R is assumed to not have changed
        # Update Y to match new integrated data
        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
        for i in range(self.max_iter_kmeans):
            # print("kmeans {}".format(i))
            # STEP 1: Update Y
            self.Y = np.dot(self.Z_cos, self.R.T)
            self.Y = self.Y / np.linalg.norm(self.Y, ord=2, axis=0)
            # STEP 2: Update dist_mat
            self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
            # STEP 3: Update R
            self.update_R()
            # STEP 4: Check for convergence
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
        # Update cells in blocks
        update_order = np.arange(self.N)
        np.random.shuffle(update_order)
        n_blocks = np.ceil(1 / self.block_size).astype(int)
        blocks = np.array_split(update_order, n_blocks)
        for b in blocks:
            # STEP 1: Remove cells
            self.E -= np.outer(np.sum(self.R[:,b], axis=1), self.Pr_b)
            self.O -= np.dot(self.R[:,b], self.Phi[:,b].T)
            # STEP 2: Recompute R for removed cells
            self.R[:,b] = self._scale_dist[:,b]
            self.R[:,b] = np.multiply(
                self.R[:,b],
                np.dot(
                    np.power((self.E + 1) / (self.O + 1), self.theta),
                    self.Phi[:,b]
                )
            )
            self.R[:,b] = self.R[:,b] / np.linalg.norm(self.R[:,b], ord=1, axis=0)
            # STEP 3: Put cells back
            self.E += np.outer(np.sum(self.R[:,b], axis=1), self.Pr_b)
            self.O += np.dot(self.R[:,b], self.Phi[:,b].T)
        return 0

    def check_convergence(self, i_type):
        obj_old = 0.0
        obj_new = 0.0
        # Clustering, compute new window mean
        if i_type == 0:
            okl = len(self.objective_kmeans)
            for i in range(self.window_size):
                obj_old += self.objective_kmeans[okl - 2 - i]
                obj_new += self.objective_kmeans[okl - 1 - i]
            if abs(obj_old - obj_new) / abs(obj_old) < self.epsilon_kmeans:
                return True
            return False
        # Harmony
        if i_type == 1:
            obj_old = self.objective_harmony[-2]
            obj_new = self.objective_harmony[-1]
            if (obj_old - obj_new) / abs(obj_old) < self.epsilon_harmony:
                return True
            return False
        return True

def safe_entropy(x: np.array):
    y = np.multiply(x, np.log(x))
    y[~np.isfinite(y)] = 0.0
    return y

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



import numpy as np
from sklearn.cluster import KMeans


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



if __name__ == "__main__":

    meta_data = pd.read_csv("data/pbmc_3500_meta.tsv.gz", sep="\t")
    data_mat = pd.read_csv("data/pbmc_3500_pcs.tsv.gz", sep="\t")

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
        max_iter_harmony = 20,
        max_iter_kmeans = 20,
        epsilon_cluster = 1e-5,
        epsilon_harmony = 1e-4, 
        plot_convergence = False,
        verbose = True,
        reference_values = None,
        cluster_prior = None,
        random_state = 0,
        cluster_fn = 'kmeans'
    ):
        """Run Harmony.
        """

        # theta = None
        # lamb = None
        # sigma = 0.1
        # nclust = None
        # tau = 0
        # block_size = 0.05
        # epsilon_cluster = 1e-5
        # epsilon_harmony = 1e-4
        # plot_convergence = False
        # verbose = True
        # reference_values = None
        # cluster_prior = None
        # random_state = 0
        # cluster_fn = 'kmeans'. Also accepts a callable object with data, num_clusters parameters

        N = meta_data.shape[0]
        if data_mat.shape[1] != N:
            data_mat = data_mat.T

        assert data_mat.shape[1] == N, \
        "data_mat and meta_data do not have the same number of cells" 

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

        assert len(theta) == np.sum(phi_n), \
            "each batch variable must have a theta"

        if lamb is None:
            lamb = np.repeat([1] * len(phi_n), phi_n)
        elif isinstance(lamb, float) or isinstance(lamb, int):
            lamb = np.repeat([lamb] * len(phi_n), phi_n)
        elif len(lamb) == len(phi_n):
            lamb = np.repeat([lamb], phi_n)

        assert len(lamb) == np.sum(phi_n), \
            "each batch variable must have a lambda"

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

    
        return hh,ho

    hh_lisi_AC = []
    hh_lisi_AB = []

    ho_lisi_AC = []
    ho_lisi_AB = []

    for i in range(15):
        hh,ho = run_harmony(data_mat, meta_data, ['donor'], max_iter_harmony=i + 1)

        mask = meta_data['donor'] != 'B'
        hh_mask = hh.Z_corr.T[mask]
        ho_mask = ho.Z_corr.T[mask]

        meta_data_mask = meta_data[mask]

        hh_lisi_AC.append(np.mean(compute_lisi(hh_mask, meta_data_mask, ['donor'])))
        ho_lisi_AC.append(np.mean(compute_lisi(ho_mask, meta_data_mask, ['donor'])))

        mask = meta_data['donor'] != 'C'
        hh_mask = hh.Z_corr.T[mask]
        ho_mask = ho.Z_corr.T[mask]

        meta_data_mask = meta_data[mask]

        hh_lisi_AB.append(np.mean(compute_lisi(hh_mask, meta_data_mask, ['donor'])))
        ho_lisi_AB.append(np.mean(compute_lisi(ho_mask, meta_data_mask, ['donor'])))


    np.save('hh_lisi_AC.npy', hh_lisi_AC)
    np.save('hh_lisi_AB.npy', hh_lisi_AB)

    np.save('ho_lisi_AC.npy', ho_lisi_AC)
    np.save('ho_lisi_AB.npy', ho_lisi_AB)

