import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.utils.data as Data


def load_dataset():

    data_pr = np.fromfile('datasets/data_pr_5000.bin', dtype=np.float32)
    data_pr = np.reshape(data_pr, [128, 128, 1, 5000], 'F')
    data_pr = np.transpose(data_pr, [3, 2, 0, 1])  # num channel H W
    # plt.imshow(data_pr[0, 0, :, :])
    # plt.show()

    data_ne = np.fromfile('datasets/data_ne_5000.bin', dtype=np.float32)
    data_ne = np.reshape(data_ne, [128, 128, 1, 5000], 'F')
    data_ne = np.transpose(data_ne, [3, 2, 0, 1])  # num channel H W

    label_pr = np.fromfile('datasets/label_pr_5000.bin', dtype=np.float32)
    label_pr = np.reshape(label_pr, [128, 128, 1, 5000], 'F')
    label_pr = np.transpose(label_pr, [3, 2, 0, 1])  # num channel H W
    # plt.imshow(label_pr[0, 0, :, :])
    # plt.show()

    label_ne = np.fromfile('datasets/label_ne_5000.bin', dtype=np.float32)
    label_ne = np.reshape(label_ne, [128, 128, 1, 5000], 'F')
    label_ne = np.transpose(label_ne, [3, 2, 0, 1])  # num channel H W

    data_pr = torch.from_numpy(data_pr)
    data_ne = torch.from_numpy(data_ne)
    label_pr = torch.from_numpy(label_pr)
    label_ne = torch.from_numpy(label_ne)

    datasets = Data.TensorDataset(data_pr, data_ne, label_pr, label_ne)

    return datasets








