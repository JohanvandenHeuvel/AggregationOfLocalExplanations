import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = np.load("scores.npy", allow_pickle=True).item()

# print('Percentage of images where the RBM scores better:')
fig, axs = plt.subplots(3, 3, figsize=(10, 10))
for i, score in enumerate(["insert", "delete", "irof"]):

    df = data[score]
    ax = axs[i]

    lime_1 = np.array(df["lime_1"])
    lime_2 = np.array(df["lime_2"])
    lime_3 = np.array(df["lime_3"])
    mean = np.array(df["mean"])
    rmb_flip_detection = np.array(df["rbm_flip_detection"])

    sns.distplot(rmb_flip_detection - lime_1, kde=False, ax=ax[0])
    sns.distplot(rmb_flip_detection - lime_2, kde=False, ax=ax[1])
    sns.distplot(rmb_flip_detection - lime_3, kde=False, ax=ax[2])

    ax[0].set_title("rbm vs lime1 ({})".format(score))
    ax[1].set_title("rbm vs lime2 ({})".format(score))
    ax[2].set_title("rbm vs lime3 ({})".format(score))

plt.tight_layout()
plt.show()

    # if score == "delete":
    #     lime = np.min([lime_1, lime_2, lime_3])
    #     x_1 = rmb_flip_detection <= lime_1
    #     x_2 = rmb_flip_detection <= lime_2
    #     x_3 = rmb_flip_detection <= lime_3
    #     x_lime = rmb_flip_detection <= lime
    #     x_mean = rmb_flip_detection <= mean
    # else:
    #     lime = np.max([lime_1, lime_2, lime_3])
    #     x_1 = rmb_flip_detection >= lime_1
    #     x_2 = rmb_flip_detection >= lime_2
    #     x_3 = rmb_flip_detection >= lime_3
    #     x_lime = rmb_flip_detection >= lime
    #     x_mean = rmb_flip_detection >= mean
    #
    # print("=== {} ===".format(score))
    # print("lime1: \t", sum(x_1) / len(x_1) * 100, "%")
    # print("lime2: \t", sum(x_2) / len(x_2) * 100, "%")
    # print("lime3: \t", sum(x_3) / len(x_3) * 100, "%")
    # print("best of lime: \t", sum(x_lime) / len(x_lime) * 100, "%")
    # print("mean:  \t", sum(x_mean) / len(x_mean) * 100, "%")



print("test")
