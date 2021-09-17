import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = np.load("scores.npy", allow_pickle=True).item()

fig, axs = plt.subplots(3, 1, figsize=(20, 20))
for i, score in enumerate(["insert", "delete", "irof"]):
    ax = axs[i]
    df = data[score]
    for key in df:
        if key=="rbm_flip_detection" or key=="mean":
            d = dict(linewidth=5, linestyle="--")
        else:
            d = dict(linewidth=2)
        sns.distplot(df[key], ax=ax, kde_kws=d)
        # sns.distplot(df["mean"], ax=ax)
    ax.legend(df.keys())
    ax.set_title(score)

plt.show()