from Reproduction.func import *

type = type_list[0]
mu = parm_dict[type][0]
sigma = parm_dict[type][1]
l = parm_dict[type][2]
u = parm_dict[type][3]

error_df = None

def generate_L_S_normal_rank(p: int, mu: float, sigma: float,
                 lower: float, upper: float, group: int = 1):
    """
    Generate the normal rank matrix L and the sparse matrix S
    :param p: number of dimensions
    :param mu: mean of the gaussian distribution
    :param sigma: standard deviation of the gaussian distribution
    :param lower: lower bound
    :param upper: upper bound
    :param group: number of groups
    :return: L, S
    """
    L = np.random.normal(mu, sigma, (p, p))
    S = np.zeros((p * p, group))
    for g in range(group):
        indices = np.random.choice(p * p, int(p * p / 20), replace=False)
        S[indices, g] = 1
        S[:, g] = S[:, g] * np.random.uniform(low=lower, high=upper, size=p * p)
    return L, S.reshape((p, p, group))

# Main function to conduct the new test
def test_normal_rank_new(p: int = 10, mu: float = 0, sigma: float = 1,
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
        L, S = generate_L_S_normal_rank(p, mu=mu, sigma=sigma, lower=lower, upper=upper, group=1)
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

# Main function to conduct the new test
def test_non_exponential_new(p: int = 10, mu: float = 0, sigma: float = 1,
                 lower: float = 0, upper: float = 1,
                 n: int = 500, rep_num: int = 30, type="Gaussian"):
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

                M[i, j, :] = np.random.uniform(low=0, high=theta[i, j], size=n)

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
