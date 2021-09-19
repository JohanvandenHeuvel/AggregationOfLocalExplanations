import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('seaborn-paper')

# mpl.use('pdf')

#plt.rc('font', family='Avenir', serif='Computer Modern')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('axes', labelsize=19)
plt.rc('axes', titlesize=19)
plt.rc('legend',fontsize=15) # using a size in points
plt.rcParams['axes.linewidth'] = 2.3
plt.rcParams["figure.figsize"] = (6.,4.1631189606246317)

rc = {"axes.spines.left" : True,
      "axes.spines.right" : True,
      "axes.spines.bottom" : True,
      "axes.spines.top" : True,
      "xtick.bottom" : True,
      "xtick.labelbottom" : True,
      "ytick.labelleft" : True,
      "ytick.left" : True}
plt.rcParams.update(rc)

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

    for j, lime in enumerate([lime_1, lime_2, lime_3]):
        if score == "delete":
            b = lime - rmb_flip_detection
        else:
            b = rmb_flip_detection - lime
        b += 1
        threshold = 1

        percentage = sum(b >= threshold) / len(b) * 100
        pos = b[b >= threshold]
        neg = b[b < threshold]
        sns.distplot(pos, kde=False, ax=ax[j], color='green')
        sns.distplot(neg, kde=False, ax=ax[j], color='red')
        ax[j].legend(["{:.1f}\%".format(percentage), "{:.1f}\%".format(100-percentage)])

    if score == "irof":
        score = score.upper()
    ax[0].set_title("RBM vs. LIME-1 ({})".format(score))
    ax[1].set_title("RBM vs. LIME-2 ({})".format(score))
    ax[2].set_title("RBM vs. LIME-3 ({})".format(score))

plt.tight_layout()
# plt.show()

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


plt.savefig('multiple_lime.png', dpi=300,  bbox_inches="tight")
