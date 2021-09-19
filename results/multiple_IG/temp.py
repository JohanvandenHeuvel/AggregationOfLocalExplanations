import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = np.load("scores.npy", allow_pickle=True).item()

# print('Percentage of images where the RBM scores better:')
fig, axs = plt.subplots(3, 4, figsize=(10, 10))
for i, score in enumerate(["insert", "delete", "irof"]):

    df = data[score]
    ax = axs[i]

    IG_1 = np.array(df["integrated_gradients_1"])
    IG_2 = np.array(df["integrated_gradients_2"])
    IG_3 = np.array(df["integrated_gradients_3"])
    IG_4 = np.array(df["integrated_gradients_4"])
    mean = np.array(df["mean"])
    rmb_flip_detection = np.array(df["rbm_flip_detection"])

    sns.distplot(rmb_flip_detection - IG_1, kde=False, ax=ax[0])
    sns.distplot(rmb_flip_detection - IG_2, kde=False, ax=ax[1])
    sns.distplot(rmb_flip_detection - IG_3, kde=False, ax=ax[2])
    sns.distplot(rmb_flip_detection - IG_4, kde=False, ax=ax[3])

    ax[0].set_title("rbm vs IG1 ({})".format(score))
    ax[1].set_title("rbm vs IG2 ({})".format(score))
    ax[2].set_title("rbm vs IG3 ({})".format(score))
    ax[3].set_title("rbm vs IG4 ({})".format(score))

plt.tight_layout()
plt.show()

# if score == "delete":
#         IG = np.min([IG_1, IG_2, IG_3, IG_4])
#         x_1 = rmb_flip_detection <= IG_1
#         x_2 = rmb_flip_detection <= IG_2
#         x_3 = rmb_flip_detection <= IG_3
#         x_4 = rmb_flip_detection <= IG_4
#         x_IG = rmb_flip_detection <= IG
#         x_mean = rmb_flip_detection <= mean
#     else:
#         IG = np.max([IG_1, IG_2, IG_3, IG_4])
#         x_1 = rmb_flip_detection >= IG_1
#         x_2 = rmb_flip_detection >= IG_2
#         x_3 = rmb_flip_detection >= IG_3
#         x_4 = rmb_flip_detection >= IG_4
#         x_IG = rmb_flip_detection >= IG
#         x_mean = rmb_flip_detection >= mean
#
#     print("=== {} ===".format(score))
#     print("IG1: \t", sum(x_1) / len(x_1) * 100, "%")
#     print("IG2: \t", sum(x_2) / len(x_2) * 100, "%")
#     print("IG3: \t", sum(x_3) / len(x_3) * 100, "%")
#     print("IG4: \t", sum(x_4) / len(x_4) * 100, "%")
#     print("best of IG: \t", sum(x_IG) / len(x_IG) * 100, "%")
#     print("mean:  \t", sum(x_mean) / len(x_mean) * 100, "%")
#
#
#
# print("test")
