import numpy as np
from math import comb
from scipy.stats import t, nct, norm
from helper import *

# from Experiments.HelperFiles.Code.helper import *

############### Compute coalitions, conditional means and KernelSHAP estimates ###############

def compute_coalitions_values(model, X, xloc,
            n_perms, n_samples_per_perm, mapping_dict):
    d = len(mapping_dict) if mapping_dict is not None else X.shape[1]
    kernel_weights = [0]*(d+1)
    for subset_size in range(d+1):
        if subset_size > 0 and subset_size < d:
            kernel_weights[subset_size] = (d-1)/(comb(d,subset_size)*subset_size*(d-subset_size))
    subset_size_distr = np.array(kernel_weights) / np.sum(kernel_weights)
    coalitions = []
    W_vals = []
    for count in range(n_perms):
        subset_size = np.random.choice(np.arange(len(subset_size_distr)), p=subset_size_distr)
        # Randomly choose these features, then convert to binary vector z
        S = np.random.choice(d, subset_size, replace=False)
        z = np.zeros(d)
        z[S] = 1
        # For each z/S, compute list of length {# samples/perm} of X_{S^c}|X_S
        # w_x_vals = coalitions_kshap(X, xloc, z, n_samples_per_perm, mapping_dict)
        w_x_vals = coalitions_kshap(X, xloc, S, n_samples_per_perm, mapping_dict)
        coalitions.append(z)
        # coalitions = np.append(coalitions, z).reshape((count, d))        
        W_vals.append(w_x_vals)
    coalitions = np.array(coalitions).reshape((n_perms, d))        
    # Compute all conditional means, variances, and covariances
    coalition_values, coalition_vars = conditional_means_vars_kshap(model,W_vals,xloc,
                            n_samples_per_perm)
    return coalitions, coalition_values, coalition_vars
        # count += 1
        # coalitions = np.append(coalitions, z).reshape((count, d))        
        # W_vals.append(w_x_vals)
        # if count==n_perms:
        #     # Compute all conditional means, variances, and covariances
        #     coalition_values, coalition_vars = conditional_means_vars_kshap(model,W_vals,xloc,
        #                            n_samples_per_perm)
        #     return coalitions, coalition_values, coalition_vars

def coalitions_kshap(X, xloc, S, n_samples_per_perm, mapping_dict=None):
    # Ensure valid indices in S
    d = xloc.shape[1]  # Number of features in xloc
    S = [i for i in S if i < d]  # Filter indices in S to ensure they fit within xloc's dimensions
    
    w_x_vals = []
    n = X.shape[0]
    for _ in range(n_samples_per_perm):
        w = X[np.random.choice(n, size=1), :]  # Randomly sample a row from X
        w = np.copy(w)  # Make a copy to avoid modifying the original
        # Replace features in S with corresponding values from xloc
        w[0][S] = xloc[0][S]
        w_x_vals.append(w)
    return w_x_vals


def conditional_means_vars_kshap(model, W_vals,xloc, n_samples_per_perm):
    # Calculates means (value functions) and variances for each value function
    W_vals = np.reshape(W_vals, [-1*n_samples_per_perm, xloc.shape[1]])

    preds_given_S = model(W_vals)
    preds_given_S = np.reshape(preds_given_S,[-1,n_samples_per_perm])
    coalition_values = np.mean(preds_given_S,axis=1)

    preds_given_S_c = preds_given_S - coalition_values[:,None]
    coalition_vars = np.mean( preds_given_S_c**2, axis = 1)    
    return coalition_values, coalition_vars

def invert_matrix(A):
    try:
        A_inv = np.linalg.inv(A)
    except:
        new_cond_num = 10000
        u, s, vh = np.linalg.svd(A)
        min_acceptable = s[0]/new_cond_num
        s2 = np.copy(s)
        s2[s <= min_acceptable] = min_acceptable
        A2 = np.matmul(u, np.matmul(np.diag(s2), vh))
        A_inv = np.linalg.inv(A2)
    return A_inv


def kshap_equation(yloc, coalitions, coalition_values, avg_pred):
    """
    Computes KernelSHAP estimates for all features. The equation is the solution to the 
    least squares problem of KernelSHAP.
    """
    M, d = coalitions.shape  # M = number of samples, d = number of features

    # Ensure coalition_values matches the number of rows in coalitions
    coalition_values = np.array(coalition_values[:M])  # Trim excess elements if necessary
    avg_pred_vec = np.repeat(avg_pred, M)  # Ensure avg_pred_vec matches coalition_values

    # Compute A matrix and b vector
    A = np.matmul(coalitions.T, coalitions) / M
    b = np.matmul(coalitions.T, coalition_values - avg_pred_vec) / M

    # Covert & Lee Equation 7
    A_inv = invert_matrix(A)
    ones_vec = np.ones(d).reshape((d, 1))
    numerator = np.matmul(np.matmul(ones_vec.T, A_inv), b) - yloc + avg_pred
    denominator = np.matmul(np.matmul(ones_vec.T, A_inv), ones_vec)
    term = b.reshape((d, 1)) - ones_vec * (numerator / denominator)

    kshap_ests = np.matmul(A_inv, term).reshape(-1)
    return kshap_ests


################### KernelSHAP method ###################
def compute_kshap_vars_ls(var_values, coalitions):
    d = coalitions.shape[1]
    #   mean_subset_values = np.matmul(coalitions, kshap_ests) + avg_pred
    #   var_values = np.mean((coalition_values - mean_subset_values)**2) * np.identity(M) 
    var_values = np.diagflat(var_values)
    # counts = np.sum(coalitions, axis=1).astype(int).tolist()
    ones_vec = np.ones(d).reshape((d, 1))
    A = coalitions.T @ coalitions
    try:
        A_inv = np.linalg.inv(A)
    except:
        new_cond_num = 10000
        u, s, vh = np.linalg.svd(A)
        min_acceptable = s[0]/new_cond_num
        s2 = np.copy(s)
        s2[s <= min_acceptable] = min_acceptable
        A2 = np.matmul(u, np.matmul(np.diag(s2), vh))

        A_inv = np.linalg.inv(A2)
    
    C = np.diag(np.ones(d)) - np.outer(ones_vec,ones_vec) @ A_inv/np.matmul(np.matmul(ones_vec.T, A_inv), ones_vec)

    AZ = A_inv @ C @ coalitions.T
    kshap_covmat_ls = AZ @ var_values @ AZ.T
    return kshap_covmat_ls

def kshap_test_stat(kshap_vals, kshap_covs, idx1, idx2, abs=True):
    # TO DO: x2 for reproducibility (top K) (???) if the results aren't too conservative
    # If they are, maybe I can consider a redo on top-K
    # t-test statistic
    kshap1, kshap2 = kshap_vals[idx1], kshap_vals[idx2]
    kshap_vars = np.diagonal(kshap_covs)
    var1, var2 = kshap_vars[idx1], kshap_vars[idx2]
    cov12 = kshap_covs[idx1, idx2]
    if abs is True and kshap1*kshap2 < 0: # Opposite sign
        kshap2 = -kshap2
        cov12 = -cov12
    varDiff = var1 + var2 - 2*cov12 # Difference of random variables
    testStat = np.abs(kshap1 - kshap2)/np.sqrt(varDiff)
    return testStat

def kshap_test(kshap_vals, kshap_covs, idx1, idx2, n, alpha=0.1, abs=True):
    testStat = kshap_test_stat(kshap_vals, kshap_covs, idx1, idx2, abs)
    # always smaller than Welch - more conservative
    df = n-1 
    critVal = t.ppf(1 - alpha/2, df) # 1-a/2 quantile (upper tail) of t-distribution
    return "reject" if testStat > critVal else "fail to reject"

def find_num_verified_kshap(kshap_vals, kshap_covs, n_perms, alpha=.05, abs=True):
    d = len(kshap_vals)
    order = get_ranking(kshap_vals, abs=abs)
    num_verified = 0
    # Test stability of 1 vs 2; 2 vs 3; etc (d-1 total tests)
    while num_verified < d-1: 
        idx1, idx2 = int(order[num_verified]), int(order[num_verified+1])
        test_result = kshap_test(kshap_vals, kshap_covs, idx1, idx2, n_perms, alpha=alpha, abs=abs)
        if test_result=="reject":
            num_verified += 1
        else:
            break
    return num_verified


def kernelshap(model, X, xloc, n_perms=500, n_samples_per_perm=10, mapping_dict=None,
               alphas=None, abs=True):
    """
    KernelSHAP function to compute Shapley values for a given model and dataset.
    """
    # Ensure xloc is a 2D NumPy array
    if isinstance(xloc, int) or isinstance(xloc, list):
        xloc = np.array([xloc])
    if xloc.ndim == 1:
        xloc = xloc.reshape(1, -1)

    avg_pred = np.mean(model(X))  # Compute the average prediction
    y_pred = model(xloc)  # Prediction for the given instance (xloc)
    
    # Compute coalitions and their corresponding values
    coalitions, coalition_values, _ = compute_coalitions_values(model, X, xloc,
                                                                n_perms, n_samples_per_perm,
                                                                mapping_dict)
    # Solve KernelSHAP equation for Shapley values
    kshap_vals = kshap_equation(y_pred, coalitions, coalition_values, avg_pred)

    if alphas is None:
        return kshap_vals
    else:
        # Compute covariance matrix using bootstrapping
        kshap_covs = compute_kshap_vars_boot(model, xloc, avg_pred, coalitions,
                                             coalition_values, n_boot=250)
        if isinstance(alphas, list):
            n_verified = [find_num_verified_kshap(kshap_vals, kshap_covs, n_perms, alpha=alpha, abs=abs) for alpha in alphas]
        else:
            n_verified = find_num_verified_kshap(kshap_vals, kshap_covs, n_perms, alpha=alphas, abs=abs)
        return kshap_vals, n_verified


def sprtshap(model, X, xloc, K, mapping_dict=None, 
             n_samples_per_perm=5, n_perms_btwn_tests=100, n_max=100000, 
             alpha=0.1, beta=0.2, abs=True):
    """
    Sequential Probability Ratio Test (SPRT) for SHAP values.
    """
    # Ensure xloc is a 2D NumPy array
    if isinstance(xloc, int) or isinstance(xloc, list):
        xloc = np.array([xloc])
    if xloc.ndim == 1:
        xloc = xloc.reshape(1, -1)

    avg_pred = np.mean(model(X))  # Average prediction
    y_pred = model(xloc)  # Prediction for the given instance (xloc)
    
    acceptNullThresh = beta / (1 - alpha / 2)
    rejectNullThresh = (1 - beta) / (alpha / 2)

    orderings = []
    N = 0
    num_verified = 0

    while N < n_max and num_verified < K:
        # Generate new coalitions and their values
        coalitions_t, coalition_values_t, coalition_vars_t = compute_coalitions_values(
            model, X, xloc, n_perms_btwn_tests, n_samples_per_perm, mapping_dict)
        N += n_perms_btwn_tests

        # Concatenate coalitions and their values
        if N > n_perms_btwn_tests:
            coalitions = np.concatenate((coalitions, coalitions_t))
            coalition_values = np.concatenate((coalition_values, coalition_values_t))
            coalition_vars = np.concatenate((coalition_vars, coalition_vars_t))
        else:
            coalitions, coalition_values, coalition_vars = coalitions_t, coalition_values_t, coalition_vars_t

        # Compute KernelSHAP values and covariance matrix
        kshap_vals = kshap_equation(y_pred, coalitions, coalition_values, avg_pred)
        kshap_covs = compute_kshap_vars_boot(model, xloc, avg_pred, coalitions, coalition_values, n_boot=250)

        order = get_ranking(kshap_vals, abs=abs)
        while num_verified < K:
            # Find pair of indices to test
            idx1, idx2 = int(order[num_verified]), int(order[num_verified + 1])
            testStat = kshap_test_stat(kshap_vals, kshap_covs, idx1, idx2, abs)

            # SPRT logic
            null_density = t.pdf(testStat, df=N-1)
            alt_density = nct.pdf(testStat, df=N-1, nc=testStat)
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
    if converged:
        final_order = get_ranking(kshap_vals)
        for i in range(K):
            if orderings[i] != (final_order[i], final_order[i + 1]):
                converged = False
    return kshap_vals, N, converged



###################################

def compute_kshap_vars_boot(model, xloc, avg_pred, coalitions, 
                    coalition_values, n_boot):
    """
    Returns n_boot sets of kernelSHAP values for each feature, 
    fitting kernelSHAP on both the true model and its approximation with each bootstrapped resampling.

    We can probably make a version of this function where you don't have to bootstrap the model,
    since we don't need this for computing CV-kSHAP estimates; we only need it to compute the
    variance reduction of CV-kSHAP over vanilla kSHAP. Low priority.
    """

    kshap_vals_boot = []
    M = coalition_values.shape[0]
    yloc = model(xloc)
    for _ in range(n_boot):
        idx = np.random.randint(M, size=M)
        z_boot = coalitions[idx]
        coalition_values_model_boot = coalition_values[idx]

        # compute the kernelSHAP estimates on these bootstrapped samples, fitting ls
        kshap_vals_boot.append(kshap_equation(yloc, z_boot, coalition_values_model_boot, avg_pred))

    kshap_vals_boot = np.stack(kshap_vals_boot, axis=0)
    # Compute empirical covariance matrix of each feature's KernelSHAP value, using bootstrapped samples.
    kshap_cov_boot = np.cov(np.array(kshap_vals_boot), rowvar=False)
    return kshap_cov_boot
