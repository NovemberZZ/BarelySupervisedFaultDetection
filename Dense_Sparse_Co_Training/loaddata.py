import torch
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt


def loaddataset_Poseidon():

    num1 = 21  # number of labeled data(pseudo label)
    num2 = 210  # number of unlabeled data

    # ************************************** Labeled data ******************************************** #

    data_labeled = np.zeros([num1 * 2, 1, 128, 128, 128], dtype=np.float32)
    for i in range(num1):
        file_str = './datasets_Poseidon/labeled/' + 'Poseidon_128_128_128_' + str(i).rjust(3, '0') + '.bin'
        tmp = np.fromfile(file_str, dtype=np.float32)
        tmp = np.reshape(tmp, [128, 128, 128], 'F')
        tmp = np.transpose(tmp, [2, 0, 1])
        tmp = (tmp - tmp.mean()) / tmp.std()
        data_labeled[i, 0, :, :, :] = tmp
        tmp = np.flip(tmp, axis=2)          # flip the data (enhancement)
        data_labeled[i + num1, 0, :, :, :] = tmp

    # ****************************** pseudo label (Inline direction) ********************************** #

    label_i = np.zeros([num1 * 2, 1, 128, 128, 128], dtype=np.int64)
    for i in range(num1):
        file_str = './datasets_Poseidon/labeled/' + 'Label_pseudo_inline_dir_' + str(i).rjust(3, '0') + '.bin'
        tmp = np.fromfile(file_str, dtype=np.float32)
        tmp = np.reshape(tmp, [128, 128, 128], 'F')
        tmp = np.transpose(tmp, [2, 0, 1])
        label_i[i, 0, :, :, :] = tmp
        tmp = np.flip(tmp, axis=2)
        label_i[i + num1, 0, :, :, :] = tmp

    # ****************************** pseudo label (Timeslice direction) ******************************** #

    label_t = np.zeros([num1 * 2, 1, 128, 128, 128], dtype=np.int64)
    for i in range(num1):
        file_str = './datasets_Poseidon/labeled/' + 'Label_pseudo_timeslice_dir_' + str(i).rjust(3, '0') + '.bin'
        tmp = np.fromfile(file_str, dtype=np.float32)
        tmp = np.reshape(tmp, [128, 128, 128], 'F')
        tmp = np.transpose(tmp, [2, 0, 1])
        label_t[i, 0, :, :, :] = tmp
        tmp = np.flip(tmp, axis=2)
        label_t[i + num1, 0, :, :, :] = tmp

    # ************************************** Unlabeled data ******************************************** #

    data_unlabeled = np.zeros([num2, 1, 128, 128, 128], dtype=np.float32)
    for i in range(num2):
        file_str = './datasets_Poseidon/unlabeled/' + 'Poseidon_128_128_128_' + str(i).rjust(3, '0') + '.bin'
        tmp = np.fromfile(file_str, dtype=np.float32)
        tmp = np.reshape(tmp, [128, 128, 128], 'F')
        tmp = np.transpose(tmp, [2, 0, 1])  # after transpose: Inline Sample CDP
        tmp = (tmp - tmp.mean()) / tmp.std()
        data_unlabeled[i, 0, :, :, :] = tmp

    # plt.subplot(1, 4, 1)
    # plt.imshow(data_labeled[0, 0, 64, :, :])
    # plt.subplot(1, 4, 2)
    # plt.imshow(label_i[0, 0, 64, :, :])
    # plt.subplot(1, 4, 3)
    # plt.imshow(label_t[0, 0, 64, :, :])
    # plt.subplot(1, 4, 4)
    # plt.imshow(data_unlabeled[0, 0, 64, :, :])
    # plt.show()

    data_labeled = torch.from_numpy(data_labeled)
    label_i = torch.from_numpy(label_i)
    label_t = torch.from_numpy(label_t)

    data_unlabeled = torch.from_numpy(data_unlabeled)

    dataset_1 = Data.TensorDataset(data_labeled, label_i, label_t)
    dataset_2 = Data.TensorDataset(data_unlabeled)

    return dataset_1, dataset_2
