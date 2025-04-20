from eRPCA_py import eRPCA
import numpy as np
from scipy.linalg import svd
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ePCA
# The following code is adapted from the original ePCA algorithm as presented in MATLAB in the ePCA paper.

# Function to estimate the scaled true spike
def standard_spiked_forward(ell, gamma):
    k = len(ell)
    lam = np.zeros(k)
    cos_right = np.zeros(k)
    cos_left = np.zeros(k)
    v = np.zeros(k, dtype=complex)

    gamma_minus = (1 - np.sqrt(gamma)) ** 2
    gamma_plus = (1 + np.sqrt(gamma)) ** 22

    for i in range(0, k):

        if (ell[i] < gamma ** (1 / 2)) and (ell[i] > - gamma ** (1 / 2)):

            lam[i] = (1+gamma ** (1 / 2)) ** 2

            cos_right[i] = 0
            cos_left[i] = 0
        else:
            lam[i] = (1 + ell[i]) * (1 + gamma / ell[i])
            cos_right[i] = (1 - gamma / ell[i] ** 2) / (1 + gamma / ell[i])
            cos_left[i] = (1 - gamma / ell[i] ** 2) / (1 + 1/ell[i])


        x = lam[i]
        im_mult = 1

        if (x > gamma_minus) & (x < gamma_plus):
            im_mult = np.array(1j)

        v[i] = 1 / (2 * x) * (- (1 + x - gamma) +
                              im_mult * (np.abs((1 + x - gamma) ** 2 - 4 * x)) ** (1/2))

    return v

# Function to denoise the estimated S
def denoise(est_S, Y_bar1, D_n1, Y, n):
    pinv_result = np.linalg.pinv(D_n1 + est_S)
    X = est_S @ pinv_result @ Y.T + D_n1 @ pinv_result @ Y_bar1[..., np.newaxis] @ np.ones(n).T[np.newaxis, ...]
    return X

# Main function for ePCA
def epca(obs_array, type):
    m1 = obs_array.shape[0]
    m2 = obs_array.shape[1]
    if len(obs_array.shape) == 3:
        n = obs_array.shape[2]
        obs_array = obs_array.reshape((m1 * m2, n))
    else:
        n = obs_array.shape[2] * obs_array.shape[3]
        obs_array = obs_array.reshape((m1 * m2, n))

    obs_array = obs_array.T
    Y = obs_array
    Y_bar = np.mean(Y, axis=0)
    S = (Y.T - Y_bar[..., np.newaxis]) @ (Y.T - Y_bar[..., np.newaxis]).T / n

    if type == "Bernoulli":
        D_n = np.diag(Y_bar * (1 - Y_bar))
    elif type == "Exponential":
        D_n = np.diag(Y_bar * Y_bar)
    elif type == "Poisson":
        D_n = np.diag(Y_bar)
    else:
        sigma_sq = 1
        D_n = np.diag(np.ones_like(Y_bar) * sigma_sq)

    S_d = S - D_n
    S_h = np.sqrt(D_n ** (1/2)) @ S_d @ np.sqrt(D_n ** (1/2))
    try:

        w, lam, wt = svd(S_h, full_matrices=False)
    except Exception as e:
        print(e)

    r = int(np.round(m1 / 2))
    gamma = m1 * m2/n
    white_eval = lam ** 2
    E = lam[0: r] ** 2
    white_shr_eval = np.zeros_like(E)

    for i in range(len(E)):
        if E[i] > ((1 + np.sqrt(gamma)) ** 2):
            white_shr_eval[i] = (E[i] + 1 - gamma + np.sqrt((E[i] + 1 - gamma) ** 2 - 4 * E[i])) / 2 - 1
        else:
            white_shr_eval[i] = 1 + np.sqrt(gamma) - 1


    S_h_eta = w[:, 0: r] @ np.diag(white_shr_eval) @ wt[0: r, :]
    S_he = D_n ** (1 / 2) @ S_h_eta @ D_n ** (1 / 2)

    lam = np.concatenate(([lam[0: r], np.zeros(m1 * m2 - r)]))
    S_h_eta = w @ np.diag(lam) @ wt
    S_he = D_n ** (1 / 2) @ S_h_eta @ D_n ** (1 / 2)

    values, vectors = np.linalg.eig(S_he)

    index = values.argsort()[::-1]

    # Sort eigenvalues and eigenvectors
    recolor_eval = values[index]
    recolor_v = vectors[:, index]

    c2 = standard_spiked_forward(white_shr_eval, gamma)
    s2 = 1 - c2

    first_Part = np.resize(np.sum(D_n) * white_shr_eval, (m1 * m2 * recolor_eval).shape)

    tau = first_Part / (m1 * m2 * recolor_eval)
    alpha = np.zeros(r)

    for i in range(0, r):
        if c2[i] > 0:
            alpha[i] = (1 - s2[i] * tau[i]) / c2[i]
        else:
            alpha[i] = 1

    fit_alpha = np.resize(alpha, recolor_eval.shape)

    white_shr_eval_scaled = fit_alpha * recolor_eval
    eigval = np.diag(white_shr_eval_scaled[0: r])
    eigvec = recolor_v[:, 0: r]
    covar_est = eigvec @ eigval @ eigvec.T

    L_vec = denoise(covar_est, Y_bar, D_n, Y, n)
    L_pred = np.mean(L_vec, axis=1)
    L_mat = np.real(L_pred.reshape(m1, m2, order='C'))

    return L_mat

# RPCA
class RPCA(object):

    def __init__(self, observation_matrix: np.array,
                 mu: float = None, alpha: float = None, beta: float = None, runs: int = 100):
        self.m1 = observation_matrix.shape[0]
        self.m2 = observation_matrix.shape[1]
        if len(observation_matrix.shape) == 3:
            self.data = observation_matrix
        else:
            self.data = observation_matrix.reshape((self.m1, self.m2,
                                                    observation_matrix.shape[2] * observation_matrix.shape[3]))
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.runs = runs

    def __mc_function_ls(self):

        obs_test_mean = np.nanmean( self.data, axis=2)
        U, S, Vh = svd(obs_test_mean, full_matrices=False)

        # Initialization
        L = U @ np.diag(
            np.append(S[0: 2], np.zeros((min((self.m1, self.m2)) - 2)))
        ) @ Vh

        Y = np.zeros((self.m1, self.m2))
        Mu = np.ones((self.m1, self.m2))

        # Create empty arrays to store the results
        Mu_all = np.zeros((self.m1, self.m2, self.runs))
        S_all = np.zeros((self.m1, self.m2, self.runs))
        L_all = np.zeros((self.m1, self.m2, self.runs))
        Y_all = np.zeros((self.m1, self.m2, self.runs))


        # Initialize first iteration
        L_all[:, :, 0] = L
        Mu_all[:, :, 0] = Mu
        Y_all[:, :, 0] = Y

        # Run Monte Carlo sampling
        return self.__mc_samp_ls(Mu_all, S_all, Y_all, L_all)

    def __mc_samp_ls(self, Mu_all, S_all, Y_all, L_all):

        for i in range(0, self.runs - 1):
            # Update S matrix
            S_new = self.__update_S(Mu_all[:, :, i], L_all[:, :, i], Y_all[:, :, i])
            S_all[:, :, i + 1] = S_new

            # Update Mu matrix
            Mu_new = self.__update_Mu(L_all[:, :, i], S_all[:, :, i], Y_all[:, :, i])
            Mu_all[:, :, i + 1] = Mu_new

            # Update Y matrix
            Y_new = Y_all[:, :, i] + self.mu * (Mu_all[:, :, i + 1] - L_all[:, :, i] - S_all[:, :, i + 1])
            Y_all[:, :, i + 1] = Y_new

            # Update L matrix
            L_new = self.__update_L(Mu_all[:, :, i + 1], S_all[:, :, i + 1], Y_all[:, :, i + 1])
            L_all[:, :, i + 1] = L_new

            # Convergence check
            if (np.linalg.norm(Mu_all[:, :, i + 1] - L_all[:, :, i + 1] - S_all[:, :, i + 1], ord="fro") <=
                    (1e-5 * np.linalg.norm(Mu_all[:, :, i + 1], ord="fro"))):
                break

        return L_all[:, :, -1], S_all[:, :, -1]

    def __update_S(self, P_k, L_k, Y_k):
        """
        Function to update matrix S
        :param P_k:
        :param L_k:
        :param Y_k:
        :return: new matrix S
        """
        X = P_k - L_k + (1 / self.mu) * Y_k
        tau = self.beta / self.mu

        # Apply soft thresholding
        S_new = np.sign(X) * np.maximum(np.abs(X) - tau, 0)
        return S_new

    def __update_Mu(self, L_k, S_k, Y_k):
        """
        Function to update matrix Mu
        :param L_k:
        :param S_k:
        :param Y_k:
        :return: new matrix Mu
        """
        Mu_new = np.zeros((self.m1, self.m2))
        for i in range(0, self.m1):
            for j in range(0, self.m2):
                Mu_seq = np.arange(0.00001, np.max(self.data[i, j, :] + 1), 0.1)

                arg_func = (((np.mean(self.data[i, j, :]) - Mu_seq) / 0.35) ** 2 +
                            (self.mu / 2) * (Mu_seq - L_k[i, j] - S_k[i, j] + (1 / self.mu) * Y_k[i, j]) ** 2)
                Mu_new[i, j] = Mu_seq[np.argmin(arg_func)]

        return Mu_new

    def __update_L(self, P_k, S_k, Y_k):
        """
        Function to update matrix L
        :param P_k:
        :param S_k:
        :param Y_k:
        :return: new matrix L
        """
        X = P_k - S_k + (1 / self.mu) * Y_k

        # Perform SVD on X
        U, S, Vh = svd(X, full_matrices=False)

        tau = self.alpha / self.mu
        d = np.diag(S)

        S_d = np.sign(d) * np.maximum(np.abs(d) - tau, 0)

        L_new = U @ S_d @ Vh

        return L_new

    def run(self):
        if self.alpha is None:
            self.alpha = 1
        if self.beta is None:
            self.beta = 1 / np.sqrt(max(self.m1, self.m2))
        if self.mu is None:
            self.mu = (self.m1 * self.m2) / (4 * np.nansum(np.abs(self.data[:, :, 1:10])))

        # Running the Monte Carlo function for RPCA
        return self.__mc_function_ls()

def generate_L_S(p: int, mu: float, sigma: float,
                 lower: float, upper: float, group: int = 1):
    """
    Generate the low-rank matrix L and the sparse matrix S
    :param p: number of dimensions
    :param mu: mean of the gaussian distribution
    :param sigma: standard deviation of the gaussian distribution
    :param lower: lower bound
    :param upper: upper bound
    :param group: number of groups
    :return: L, S
    """
    L = np.random.normal(mu, sigma, (p, p))
    U, S, Vh = svd(L, full_matrices=False)
    S[int(p / 5):] = 0
    L = U @ np.diag(S) @ Vh
    S = np.zeros((p * p, group))
    for g in range(group):
        indices = np.random.choice(p * p, int(p * p / 20), replace=False)
        S[indices, g] = 1
        S[:, g] = S[:, g] * np.random.uniform(low=lower, high=upper, size=p * p)
    return L, S.reshape((p, p, group))

# Main function to conduct single-group test
def single_group_test(p: int = 10, mu: float = 0, sigma: float = 1,
                 lower: float = 0, upper: float = 1,
                 n: int = 500, rep_num: int = 30, type: str = "Bernoulli"):
    """
    Perform a numerical experiments for certain dimension
    :param p: number of dimensions
    :param mu: mean of the gaussian distribution
    :param sigma: standard deviation of the gaussian distribution
    :param lower: lower bound
    :param upper: upper bound
    :param n: number of samples
    :param rep_num: number of replications
    :return: the errors of L and S from three different methods
    """
    L_error_eRPCA = np.zeros(rep_num)
    S_error_eRPCA = np.zeros(rep_num)

    L_error_ePCA = np.zeros(rep_num)

    L_error_RPCA = np.zeros(rep_num)
    S_error_RPCA = np.zeros(rep_num)

    for rep in range(rep_num):
        L, S = generate_L_S(p, mu=mu, sigma=sigma, lower=lower, upper=upper, group=1)
        S = S[:, :, 0]
        theta = L + S
        M = np.zeros((p, p, n))
        for i in range(0, theta.shape[0]):
            for j in range(0, theta.shape[1]):

                if type == "Bernoulli":

                    if theta[i, j] >= 1:
                        param_p = 1
                    elif theta[i, j] <= 0:
                        param_p = 0
                    else:
                        param_p = theta[i, j]
                    M[i, j, :] = np.random.binomial(n=1, p=param_p, size=n)

                elif type == "Exponential":
                    if theta[i, j] < 0:
                        param_scale = 0
                    else:
                        param_scale = 1 / theta[i, j]
                    M[i, j, :] = np.random.exponential(size=n, scale=param_scale)

                elif type == "Poisson":
                    if theta[i, j] < 0:
                        param_lam = 0
                    else:
                        param_lam = theta[i, j]
                    M[i, j, :] = np.random.poisson(lam=param_lam, size=n)
                else:
                    pass

        # erpca
        erpca = eRPCA.ERPCA(observation_matrix=M)
        L_est, S_est = erpca.run()
        L_error_eRPCA[rep] = np.linalg.norm(L - L_est, ord="fro")
        S_error_eRPCA[rep] = np.linalg.norm(S - S_est, ord="fro")

        # rpca
        rpca = RPCA(observation_matrix=M)
        L_est, S_est = rpca.run()
        L_error_RPCA[rep] = np.linalg.norm(L - L_est, ord="fro")
        S_error_RPCA[rep] = np.linalg.norm(S - S_est, ord="fro")

        # epca
        L_est = epca(M, type=type)
        L_error_ePCA[rep] = np.linalg.norm(L - L_est, ord="fro")


    return L_error_eRPCA, S_error_eRPCA, L_error_RPCA, S_error_RPCA, L_error_ePCA

# Main function to conduct multi-group test
def multi_group_test(p: int = 10, mu: float = 0, sigma: float = 1,
                 lower: float = 0, upper: float = 1,
                 n: int = 500, rep_num: int = 30, type: str = "Bernoulli", group: int = 2):
    """
    Perform a numerical experiments for certain dimension
    :param p: number of dimensions
    :param mu: mean of the gaussian distribution
    :param sigma: standard deviation of the gaussian distribution
    :param lower: lower bound
    :param upper: upper bound
    :param n: number of samples
    :param rep_num: number of replications
    :param type: type of distribution
    :param group: number of groups
    :return: the errors of L and S from three different methods
    """
    L_error_eRPCA = np.zeros(rep_num)
    S_error_eRPCA = np.zeros(rep_num)

    L_error_ePCA = np.zeros(rep_num)

    L_error_RPCA = np.zeros(rep_num)
    S_error_RPCA = np.zeros(rep_num)

    for rep in range(rep_num):
        L, S_group = generate_L_S(p, mu=mu, sigma=sigma, lower=lower, upper=upper, group=group)
        theta_group = L[..., np.newaxis] + S_group
        M = np.zeros((p, p, n, group))
        for g in range(0, group):
            for i in range(0, theta_group.shape[0]):
                for j in range(0, theta_group.shape[1]):
                    if type == "Bernoulli":
                        if theta_group[i, j, g] >= 1:
                            param_p = 1
                        elif theta_group[i, j, g] <= 0:
                            param_p = 0
                        else:
                            param_p = theta_group[i, j, g]
                        M[i, j, :, g] = np.random.binomial(n=1, p=param_p, size=n)

                    elif type == "Exponential":
                        if theta_group[i, j, g] < 0:
                            param_scale = 0
                        else:
                            param_scale = 1 / theta_group[i, j, g]
                        M[i, j, :, g] = np.random.exponential(size=n, scale=param_scale)

                    elif type == "Poisson":
                        if theta_group[i, j, g] < 0:
                            param_lam = 0
                        else:
                            param_lam = theta_group[i, j, g]
                        M[i, j, :, g] = np.random.poisson(lam=param_lam, size=n)

        # erpca
        erpca = eRPCA.ERPCA(observation_matrix=M)
        L_est, S_group_est = erpca.run()
        L_error_eRPCA[rep] = np.linalg.norm(L - L_est, ord="fro")
        S_error_eRPCA[rep] = np.mean(np.linalg.norm(S_group - S_group_est, axis=(0, 1), ord="fro"))

        # rpca
        rpca = RPCA(observation_matrix=M)
        L_est, S_est = rpca.run()
        L_error_RPCA[rep] = np.linalg.norm(L - L_est, ord="fro")
        S_error_RPCA[rep] = np.mean(np.linalg.norm(S_group - S_est[..., np.newaxis], axis=(0, 1), ord="fro"))

        # epca
        L_est = epca(M, type=type)
        L_error_ePCA[rep] = np.linalg.norm(L - L_est, ord="fro")

    return L_error_eRPCA, S_error_eRPCA, L_error_RPCA, S_error_RPCA, L_error_ePCA

# tuning parameters for test
p_list = [10, 20, 30, 40]
n = 500
num_trial = 30
methods = ["eRPCA", "RPCA", "ePCA"]

# mu, sigma, L, U
bern_param = [0.5, 0.15, 0.2, 0.3]
exp_param = [1, 0.15, 2, 5]
poi_param = [0, 1, 0, 1]

type_list = ["Bernoulli", "Exponential", "Poisson"]
parm_dict = dict(zip(type_list, [bern_param, exp_param, poi_param]))
