import numpy as np
from math import comb
from scipy.stats import t, nct, norm
from helper import *

############### Compute coalitions, conditional means, and KernelSHAP estimates ###############

def compute_coalitions_values(model, X, xloc, n_perms, n_samples_per_perm, mapping_dict):
    d = len(mapping_dict) if mapping_dict else X.shape[1]
    kernel_weights = [0] * (d + 1)
    for subset_size in range(d + 1):
        if 0 < subset_size < d:
            kernel_weights[subset_size] = (d - 1) / (comb(d, subset_size) * subset_size * (d - subset_size))
    subset_size_distr = np.array(kernel_weights) / np.sum(kernel_weights)
    coalitions, W_vals = [], []
    for _ in range(n_perms):
        subset_size = np.random.choice(np.arange(len(subset_size_distr)), p=subset_size_distr)
        S = np.random.choice(d, subset_size, replace=False)
        z = np.zeros(d)
        z[S] = 1
        w_x_vals = coalitions_kshap(X, xloc, S, n_samples_per_perm, mapping_dict)
        coalitions.append(z)
        W_vals.append(w_x_vals)
    coalitions = np.array(coalitions).reshape((n_perms, d))
    coalition_values, coalition_vars = conditional_means_vars_kshap(model, W_vals, xloc, n_samples_per_perm)
    return coalitions, coalition_values, coalition_vars

def coalitions_kshap(X, xloc, S, n_samples_per_perm, mapping_dict=None):
    d = xloc.shape[1]
    S = [i for i in S if i < d]
    w_x_vals = []
    n = X.shape[0]
    for _ in range(n_samples_per_perm):
        w = X[np.random.choice(n, size=1), :].copy()
        w[0][S] = xloc[0][S]
        w_x_vals.append(w)
    return w_x_vals

def compute_kshap_vars_boot(model, xloc, avg_pred, coalitions, coalition_values, n_boot=250):
    kshap_vals_boot = []
    M = coalition_values.shape[0]

    for _ in range(n_boot):
        # Sample with replacement from the coalition values
        idx = np.random.randint(0, M, size=M)
        coalitions_boot = coalitions[idx]
        coalition_values_boot = coalition_values[idx]
        kshap_vals_boot.append(kshap_equation(model(xloc), coalitions_boot, coalition_values_boot, avg_pred))

    kshap_vals_boot = np.stack(kshap_vals_boot, axis=0)
    return np.cov(kshap_vals_boot, rowvar=False)

def conditional_means_vars_kshap(model, W_vals, xloc, n_samples_per_perm):
    # Reshape W_vals to a 2D array with all samples
    W_vals = np.reshape(W_vals, [-1 * n_samples_per_perm, xloc.shape[1]])

    # Get predictions for each perturbed sample
    preds_given_S = model(W_vals)
    preds_given_S = np.reshape(preds_given_S, [-1, n_samples_per_perm])

    # Compute the mean predictions (coalition values)
    coalition_values = np.mean(preds_given_S, axis=1)

    # Compute variances for each coalition
    preds_given_S_c = preds_given_S - coalition_values[:, None]  # Center the predictions
    coalition_vars = np.mean(preds_given_S_c**2, axis=1)

    return coalition_values, coalition_vars


def invert_matrix(A):
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        new_cond_num = 10000
        u, s, vh = np.linalg.svd(A)
        min_acceptable = s[0] / new_cond_num
        s[s < min_acceptable] = min_acceptable
        A2 = np.matmul(u, np.matmul(np.diag(s), vh))
        A_inv = np.linalg.inv(A2)
    return A_inv

def kshap_equation(yloc, coalitions, coalition_values, avg_pred):
    M, d = coalitions.shape
    coalition_values = np.array(coalition_values[:M])
    avg_pred_vec = np.repeat(avg_pred, M)
    A = np.matmul(coalitions.T, coalitions) / M
    b = np.matmul(coalitions.T, coalition_values - avg_pred_vec) / M
    A_inv = invert_matrix(A)
    ones_vec = np.ones(d).reshape((d, 1))
    numerator = np.matmul(np.matmul(ones_vec.T, A_inv), b) - yloc + avg_pred
    denominator = np.matmul(np.matmul(ones_vec.T, A_inv), ones_vec)
    term = b.reshape((d, 1)) - ones_vec * (numerator / denominator)
    kshap_ests = np.matmul(A_inv, term).reshape(-1)
    return kshap_ests

def kernelshap(model, X, xloc, n_perms=500, n_samples_per_perm=10, mapping_dict=None):
    if isinstance(xloc, int) or isinstance(xloc, list):
        xloc = np.array([xloc])
    if xloc.ndim == 1:
        xloc = xloc.reshape(1, -1)

    avg_pred = np.mean(model(X))
    y_pred = model(xloc)
    coalitions, coalition_values, _ = compute_coalitions_values(model, X, xloc, n_perms, n_samples_per_perm, mapping_dict)
    kshap_vals = kshap_equation(y_pred, coalitions, coalition_values, avg_pred)
    return kshap_vals

def kshap_test_stat(kshap_vals, kshap_covs, idx1, idx2, abs=True):
    """
    Calculate the test statistic for KernelSHAP values.
    :param kshap_vals: Shapley values for all features
    :param kshap_covs: Covariance matrix of Shapley values
    :param idx1: Index of the first feature
    :param idx2: Index of the second feature
    :param abs: Whether to consider absolute values for testing
    :return: Test statistic
    """
    # Extract the Shapley values and variances
    kshap1, kshap2 = kshap_vals[idx1], kshap_vals[idx2]
    kshap_vars = np.diagonal(kshap_covs)
    var1, var2 = kshap_vars[idx1], kshap_vars[idx2]
    cov12 = kshap_covs[idx1, idx2]

    # Adjust for opposite signs if abs=True
    if abs and kshap1 * kshap2 < 0:
        kshap2 = -kshap2
        cov12 = -cov12

    # Calculate the variance of the difference
    varDiff = var1 + var2 - 2 * cov12
    if varDiff <= 0:
        raise ValueError(f"Variance of difference is non-positive: {varDiff}")
    
    # Compute the test statistic
    testStat = np.abs(kshap1 - kshap2) / np.sqrt(varDiff)
    return testStat

###################################

def sprtshap(model, X, xloc, K, mapping_dict=None, n_samples_per_perm=5, n_perms_btwn_tests=100, n_max=100000, alpha=0.1, beta=0.2, abs=True):
    if isinstance(xloc, int) or isinstance(xloc, list):
        xloc = np.array([xloc])
    if xloc.ndim == 1:
        xloc = xloc.reshape(1, -1)

    avg_pred = np.mean(model(X))
    y_pred = model(xloc)
    acceptNullThresh = beta / (1 - alpha / 2)
    rejectNullThresh = (1 - beta) / (alpha / 2)
    orderings = []
    N = 0
    num_verified = 0

    while N < n_max and num_verified < K:
        coalitions_t, coalition_values_t, coalition_vars_t = compute_coalitions_values(
            model, X, xloc, n_perms_btwn_tests, n_samples_per_perm, mapping_dict)
        N += n_perms_btwn_tests

        if N > n_perms_btwn_tests:
            coalitions = np.concatenate((coalitions, coalitions_t))
            coalition_values = np.concatenate((coalition_values, coalition_values_t))
            coalition_vars = np.concatenate((coalition_vars, coalition_vars_t))
        else:
            coalitions, coalition_values, coalition_vars = coalitions_t, coalition_values_t, coalition_vars_t

        kshap_vals = kshap_equation(y_pred, coalitions, coalition_values, avg_pred)
        kshap_covs = compute_kshap_vars_boot(model, xloc, avg_pred, coalitions, coalition_values, n_boot=250)

        order = get_ranking(kshap_vals, abs=abs)
        while num_verified < K:
            idx1, idx2 = int(order[num_verified]), int(order[num_verified + 1])
            testStat = kshap_test_stat(kshap_vals, kshap_covs, idx1, idx2, abs)

            null_density = t.pdf(testStat, df=N - 1)
            alt_density = nct.pdf(testStat, df=N - 1, nc=testStat)
            if np.isnan(null_density) or np.isnan(alt_density):
                null_density = norm.pdf(testStat)
                alt_density = norm.pdf(testStat, loc=testStat)

            LR = alt_density / null_density
            if LR < acceptNullThresh:
                return kshap_vals, N, False
            if LR > rejectNullThresh:
                num_verified += 1
                orderings.append((idx1, idx2))
            else:
                break

    converged = num_verified >= K
    return kshap_vals, N, converged
