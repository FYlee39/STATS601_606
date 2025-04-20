from Reproduction.func import *

type = type_list[0]
mu = parm_dict[type][0]
sigma = parm_dict[type][1]
l = parm_dict[type][2]
u = parm_dict[type][3]

error_df = None

for j in range(len(p_list)):

    error_array = np.zeros((len(methods), num_trial, 2)) - 1

    (L_error_eRPCA, S_error_eRPCA,
        L_error_RPCA, S_error_RPCA,
        L_error_ePCA) = multi_group_test(p=p_list[j], type=type,
                                           mu=mu, sigma=sigma,
                                           lower=l, upper=u,
                                           n=n, rep_num=num_trial, group=2)

    error_array[0, :, 0] = L_error_eRPCA
    error_array[0, :, 1] = S_error_eRPCA
    error_array[1, :, 0] = L_error_RPCA
    error_array[1, :, 1] = S_error_RPCA
    error_array[2, :, 0] = L_error_ePCA

    for i in range(len(methods)):

        mid_dict_S = None

        mid_dict = {
            "Error": error_array[i, :, 0],
            "p": [p_list[j] for _ in range(num_trial)],
            "Method": [methods[i] for _ in range(num_trial)],
            "L_mat": [True for _ in range(num_trial)]
        }

        if i < 2:
            mid_dict_S = {
                "Error": error_array[i, :, 1],
                "p": [p_list[j] for _ in range(num_trial)],
                "Method": [methods[i] for _ in range(num_trial)],
                "L_mat": [False for _ in range(num_trial)]
            }

        if error_df is None:
            error_df = pd.DataFrame(mid_dict)
        else:
            error_df = pd.concat([error_df, pd.DataFrame(mid_dict)])

        if mid_dict_S is not None:
            error_df = pd.concat([error_df, pd.DataFrame(mid_dict_S)])

plt.figure("Boxplot of Bernoulli of two group setting")
plt.subplot(1, 2, 1).set_title("Boxplot of L Errors")
sns.boxplot(x="p", y="Error", hue="Method", width=0.3, dodge=True, data=error_df[error_df["L_mat"] == True])

# Plot for S errors
plt.subplot(1, 2, 2).set_title("Boxplot of S Errors")
sns.boxplot(x="p", y="Error", hue="Method", width=0.3, dodge=True, data=error_df[error_df["L_mat"] == False])

plt.xlabel("p")
plt.ylabel("Error")
plt.legend(title="Method")
plt.savefig('multi_Bernoulli.png')
plt.show()
