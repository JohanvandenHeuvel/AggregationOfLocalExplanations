import numpy as np

data = np.load("scores.npy", allow_pickle=True).item()

print('Percentage of images where the RBM scores better:')
for i, score in enumerate(["insert", "delete", "irof"]):

    df = data[score]

    IG_1 = np.array(df["integrated_gradients_1"])
    IG_2 = np.array(df["integrated_gradients_2"])
    IG_3 = np.array(df["integrated_gradients_3"])
    IG_4 = np.array(df["integrated_gradients_4"])
    mean = np.array(df["mean"])
    rmb_flip_detection = np.array(df["rbm_flip_detection"])

    if score == "delete":
        IG = np.min([IG_1, IG_2, IG_3, IG_4])
        x_1 = rmb_flip_detection <= IG_1
        x_2 = rmb_flip_detection <= IG_2
        x_3 = rmb_flip_detection <= IG_3
        x_4 = rmb_flip_detection <= IG_4
        x_IG = rmb_flip_detection <= IG
        x_mean = rmb_flip_detection <= mean
    else:
        IG = np.max([IG_1, IG_2, IG_3, IG_4])
        x_1 = rmb_flip_detection >= IG_1
        x_2 = rmb_flip_detection >= IG_2
        x_3 = rmb_flip_detection >= IG_3
        x_4 = rmb_flip_detection >= IG_4
        x_IG = rmb_flip_detection >= IG
        x_mean = rmb_flip_detection >= mean

    print("=== {} ===".format(score))
    print("IG1: \t", sum(x_1) / len(x_1) * 100, "%")
    print("IG2: \t", sum(x_2) / len(x_2) * 100, "%")
    print("IG3: \t", sum(x_3) / len(x_3) * 100, "%")
    print("IG4: \t", sum(x_4) / len(x_4) * 100, "%")
    print("best of IG: \t", sum(x_IG) / len(x_IG) * 100, "%")
    print("mean:  \t", sum(x_mean) / len(x_mean) * 100, "%")

# for key in insert_score:
#     mean = np.mean(insert_score[key])
#     std = np.var(insert_score[key])
#     print(key, mean, std)


print("test")
