import matplotlib.pyplot as plt
import pandas as pd
import os

local_folders = os.listdir()
local_folders.remove("plot.py")

results = {}
for folder in local_folders:
    file_path = os.path.join(folder, "score_table.csv")
    df = pd.read_csv(file_path)
    df = df.drop("Unnamed: 0", axis=1)
    results[folder] = df

for m in df["method"]:
    df_results = pd.DataFrame()
    for key in results:
        df = results[key]
        rbm_row = df[df["method"] == m]
        rbm_row["lr"] = key
        df_results = df_results.append(rbm_row)

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    for i, method in enumerate(["insert", "delete", "irof"]):
        ax = axs[i]

        ax[0].plot(df_results["lr"], df_results["{} mean".format(method)])
        ax[0].set_title("learning rate")
        ax[0].set_ylabel(method)

        ax[1].errorbar(
            df_results["lr"],
            df_results["{} mean".format(method)],
            df_results["{} std".format(method)],
        )
        ax[1].set_title("learning rate")
        ax[1].set_ylabel(method)

    plt.tight_layout()
    plt.suptitle(m)
    plt.show()
    fig.savefig("{}.png".format(m), dpi=fig.dpi)
