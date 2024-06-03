import torch
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt


def loaddataset_20():

    num1 = 20  # number of labeled data(pseudo label)
    num2 = 180  # number of unlabeled data

    # ************************************** Labeled data ******************************************** #

    data_labeled = np.zeros([num1 * 2, 1, 128, 128, 128], dtype=np.float32)
    for i in range(num1):
        file_str = './datasets/labeled/' + 'Image_128x128x128_' + str(i).rjust(3, '0') + '.bin'
        tmp = np.fromfile(file_str, dtype=np.float32)
        tmp = np.reshape(tmp, [128, 128, 128], 'F')
        tmp = np.transpose(tmp, [1, 2, 0])  # after transpose: Inline Sample CDP
        tmp = (tmp - tmp.mean()) / tmp.std()
        data_labeled[i, 0, :, :, :] = tmp
        tmp = np.flip(tmp, axis=2)          # flip the data (enhancement)
        data_labeled[i + num1, 0, :, :, :] = tmp

    # ****************************** pseudo label (Inline direction) ********************************** #

    label_i = np.zeros([num1 * 2, 1, 128, 128, 128], dtype=np.int64)
    for i in range(num1):
        file_str = './datasets/labeled/' + 'Label_pseudo_inline_dir_' + str(i).rjust(3, '0') + '.bin'
        tmp = np.fromfile(file_str, dtype=np.float32)
        tmp = np.reshape(tmp, [128, 128, 128], 'F')
        tmp = np.transpose(tmp, [1, 2, 0])
        label_i[i, 0, :, :, :] = tmp
        tmp = np.flip(tmp, axis=2)
        label_i[i + num1, 0, :, :, :] = tmp

    # ****************************** pseudo label (Timeslice direction) ******************************** #

    label_t = np.zeros([num1 * 2, 1, 128, 128, 128], dtype=np.int64)
    for i in range(num1):
        file_str = './datasets/labeled/' + 'Label_pseudo_timeslice_dir_' + str(i).rjust(3, '0') + '.bin'
        tmp = np.fromfile(file_str, dtype=np.float32)
        tmp = np.reshape(tmp, [128, 128, 128], 'F')
        tmp = np.transpose(tmp, [1, 2, 0])
        label_t[i, 0, :, :, :] = tmp
        tmp = np.flip(tmp, axis=2)
        label_t[i + num1, 0, :, :, :] = tmp

    # ************************************** Unlabeled data ******************************************** #

    data_unlabeled = np.zeros([num2, 1, 128, 128, 128], dtype=np.float32)
    for i in range(num2):
        file_str = './datasets/unlabeled/' + 'Image_128x128x128_' + str(i).rjust(3, '0') + '.bin'
        tmp = np.fromfile(file_str, dtype=np.float32)
        tmp = np.reshape(tmp, [128, 128, 128], 'F')
        tmp = np.transpose(tmp, [1, 2, 0])  # after transpose: Inline Sample CDP
        tmp = (tmp - tmp.mean()) / tmp.std()
        data_unlabeled[i, 0, :, :, :] = tmp

    # plt.subplot(1, 4, 1)
    # plt.imshow(data_labeled[10, 0, 64, :, :])
    # plt.subplot(1, 4, 2)
    # plt.imshow(label_i[10, 0, 64, :, :])
    # plt.subplot(1, 4, 3)
    # plt.imshow(label_t[10, 0, 64, :, :])
    # plt.subplot(1, 4, 4)
    # plt.imshow(data_unlabeled[10, 0, 64, :, :])
    # plt.show()

    data_labeled = torch.from_numpy(data_labeled)
    label_i = torch.from_numpy(label_i)
    label_t = torch.from_numpy(label_t)

    data_unlabeled = torch.from_numpy(data_unlabeled)

    dataset_1 = Data.TensorDataset(data_labeled, label_i, label_t)
    dataset_2 = Data.TensorDataset(data_unlabeled)

    return dataset_1, dataset_2


# **********************************************************************************************************************


def loaddataset_10():

    num1 = 10  # number of labeled data(pseudo label)
    num2 = 190  # number of unlabeled data

    # ************************************** Labeled data ******************************************** #

    data_labeled = np.zeros([num1 * 2, 1, 128, 128, 128], dtype=np.float32)
    for i in range(num1):
        file_str = './datasets/labeled/' + 'Image_128x128x128_' + str(i).rjust(3, '0') + '.bin'
        tmp = np.fromfile(file_str, dtype=np.float32)
        tmp = np.reshape(tmp, [128, 128, 128], 'F')
        tmp = np.transpose(tmp, [1, 2, 0])  # after transpose: Inline Sample CDP
        tmp = (tmp - tmp.mean()) / tmp.std()
        data_labeled[i, 0, :, :, :] = tmp
        tmp = np.flip(tmp, axis=2)          # flip the data (enhancement)
        data_labeled[i + num1, 0, :, :, :] = tmp

    # ****************************** pseudo label (Inline direction) ********************************** #

    label_i = np.zeros([num1 * 2, 1, 128, 128, 128], dtype=np.int64)
    for i in range(num1):
        file_str = './datasets/labeled/' + 'Label_pseudo_inline_dir_' + str(i).rjust(3, '0') + '.bin'
        tmp = np.fromfile(file_str, dtype=np.float32)
        tmp = np.reshape(tmp, [128, 128, 128], 'F')
        tmp = np.transpose(tmp, [1, 2, 0])
        label_i[i, 0, :, :, :] = tmp
        tmp = np.flip(tmp, axis=2)
        label_i[i + num1, 0, :, :, :] = tmp

    # ****************************** pseudo label (Timeslice direction) ******************************** #

    label_t = np.zeros([num1 * 2, 1, 128, 128, 128], dtype=np.int64)
    for i in range(num1):
        file_str = './datasets/labeled/' + 'Label_pseudo_timeslice_dir_' + str(i).rjust(3, '0') + '.bin'
        tmp = np.fromfile(file_str, dtype=np.float32)
        tmp = np.reshape(tmp, [128, 128, 128], 'F')
        tmp = np.transpose(tmp, [1, 2, 0])
        label_t[i, 0, :, :, :] = tmp
        tmp = np.flip(tmp, axis=2)
        label_t[i + num1, 0, :, :, :] = tmp

    # ************************************** Unlabeled data ******************************************** #

    data_unlabeled = np.zeros([num2, 1, 128, 128, 128], dtype=np.float32)
    for i in range(num2):
        file_str = './datasets/unlabeled/' + 'Image_128x128x128_' + str(i).rjust(3, '0') + '.bin'
        tmp = np.fromfile(file_str, dtype=np.float32)
        tmp = np.reshape(tmp, [128, 128, 128], 'F')
        tmp = np.transpose(tmp, [1, 2, 0])  # after transpose: Inline Sample CDP
        tmp = (tmp - tmp.mean()) / tmp.std()
        data_unlabeled[i, 0, :, :, :] = tmp

    data_labeled = torch.from_numpy(data_labeled)
    label_i = torch.from_numpy(label_i)
    label_t = torch.from_numpy(label_t)

    data_unlabeled = torch.from_numpy(data_unlabeled)

    dataset_1 = Data.TensorDataset(data_labeled, label_i, label_t)
    dataset_2 = Data.TensorDataset(data_unlabeled)

    return dataset_1, dataset_2


# **********************************************************************************************************************


def loaddataset_5():

    num1 = 5  # number of labeled data(pseudo label)
    num2 = 195  # number of unlabeled data

    # ************************************** Labeled data ******************************************** #

    data_labeled = np.zeros([num1 * 2, 1, 128, 128, 128], dtype=np.float32)
    for i in range(num1):
        file_str = './datasets/labeled/' + 'Image_128x128x128_' + str(i).rjust(3, '0') + '.bin'
        tmp = np.fromfile(file_str, dtype=np.float32)
        tmp = np.reshape(tmp, [128, 128, 128], 'F')
        tmp = np.transpose(tmp, [1, 2, 0])  # after transpose: Inline Sample CDP
        tmp = (tmp - tmp.mean()) / tmp.std()
        data_labeled[i, 0, :, :, :] = tmp
        tmp = np.flip(tmp, axis=2)          # flip the data (enhancement)
        data_labeled[i + num1, 0, :, :, :] = tmp

    # ****************************** pseudo label (Inline direction) ********************************** #

    label_i = np.zeros([num1 * 2, 1, 128, 128, 128], dtype=np.int64)
    for i in range(num1):
        file_str = './datasets/labeled/' + 'Label_pseudo_inline_dir_' + str(i).rjust(3, '0') + '.bin'
        tmp = np.fromfile(file_str, dtype=np.float32)
        tmp = np.reshape(tmp, [128, 128, 128], 'F')
        tmp = np.transpose(tmp, [1, 2, 0])
        label_i[i, 0, :, :, :] = tmp
        tmp = np.flip(tmp, axis=2)
        label_i[i + num1, 0, :, :, :] = tmp

    # ****************************** pseudo label (Timeslice direction) ******************************** #

    label_t = np.zeros([num1 * 2, 1, 128, 128, 128], dtype=np.int64)
    for i in range(num1):
        file_str = './datasets/labeled/' + 'Label_pseudo_timeslice_dir_' + str(i).rjust(3, '0') + '.bin'
        tmp = np.fromfile(file_str, dtype=np.float32)
        tmp = np.reshape(tmp, [128, 128, 128], 'F')
        tmp = np.transpose(tmp, [1, 2, 0])
        label_t[i, 0, :, :, :] = tmp
        tmp = np.flip(tmp, axis=2)
        label_t[i + num1, 0, :, :, :] = tmp

    # ************************************** Unlabeled data ******************************************** #

    data_unlabeled = np.zeros([num2, 1, 128, 128, 128], dtype=np.float32)
    for i in range(num2):
        file_str = './datasets/unlabeled/' + 'Image_128x128x128_' + str(i).rjust(3, '0') + '.bin'
        tmp = np.fromfile(file_str, dtype=np.float32)
        tmp = np.reshape(tmp, [128, 128, 128], 'F')
        tmp = np.transpose(tmp, [1, 2, 0])  # after transpose: Inline Sample CDP
        tmp = (tmp - tmp.mean()) / tmp.std()
        data_unlabeled[i, 0, :, :, :] = tmp

    data_labeled = torch.from_numpy(data_labeled)
    label_i = torch.from_numpy(label_i)
    label_t = torch.from_numpy(label_t)

    data_unlabeled = torch.from_numpy(data_unlabeled)

    dataset_1 = Data.TensorDataset(data_labeled, label_i, label_t)
    dataset_2 = Data.TensorDataset(data_unlabeled)

    return dataset_1, dataset_2


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


def loaddataset_Opunake():

    num1 = 18  # number of labeled data(pseudo label)
    num2 = 162  # number of unlabeled data

    # ************************************** Labeled data ******************************************** #

    data_labeled = np.zeros([num1 * 2, 1, 128, 128, 128], dtype=np.float32)
    for i in range(num1):
        file_str = './datasets_Opunake/labeled/' + 'Opunake_128_128_128_' + str(i).rjust(3, '0') + '.bin'
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
        file_str = './datasets_Opunake/labeled/' + 'Label_pseudo_inline_dir_' + str(i).rjust(3, '0') + '.bin'
        tmp = np.fromfile(file_str, dtype=np.float32)
        tmp = np.reshape(tmp, [128, 128, 128], 'F')
        tmp = np.transpose(tmp, [2, 0, 1])
        label_i[i, 0, :, :, :] = tmp
        tmp = np.flip(tmp, axis=2)
        label_i[i + num1, 0, :, :, :] = tmp

    # ****************************** pseudo label (Timeslice direction) ******************************** #

    label_t = np.zeros([num1 * 2, 1, 128, 128, 128], dtype=np.int64)
    for i in range(num1):
        file_str = './datasets_Opunake/labeled/' + 'Label_pseudo_timeslice_dir_' + str(i).rjust(3, '0') + '.bin'
        tmp = np.fromfile(file_str, dtype=np.float32)
        tmp = np.reshape(tmp, [128, 128, 128], 'F')
        tmp = np.transpose(tmp, [2, 0, 1])
        label_t[i, 0, :, :, :] = tmp
        tmp = np.flip(tmp, axis=2)
        label_t[i + num1, 0, :, :, :] = tmp

    # ************************************** Unlabeled data ******************************************** #

    data_unlabeled = np.zeros([num2, 1, 128, 128, 128], dtype=np.float32)
    for i in range(num2):
        file_str = './datasets_Opunake/unlabeled/' + 'Opunake_128_128_128_' + str(i).rjust(3, '0') + '.bin'
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


def loaddataset_F3():

    num1 = 16  # number of labeled data(pseudo label)
    num2 = 88  # number of unlabeled data

    # ************************************** Labeled data ******************************************** #

    data_labeled = np.zeros([num1 * 4, 1, 128, 128, 128], dtype=np.float32)
    for i in range(num1):
        file_str = './datasets_F3/labeled/' + 'F3_128_128_128_' + str(i).rjust(3, '0') + '.bin'
        tmp = np.fromfile(file_str, dtype=np.float32)
        tmp = np.reshape(tmp, [128, 128, 128], 'F')
        tmp = np.transpose(tmp, [2, 0, 1])
        tmp = (tmp - tmp.mean()) / tmp.std()

        data_labeled[i, 0, :, :, :] = tmp
        data_labeled[i + num1, 0, :, :, :] = np.rot90(tmp, 1, (0, 2))
        data_labeled[i + 2 * num1, 0, :, :, :] = np.flip(tmp, axis=2)
        data_labeled[i + 3 * num1, 0, :, :, :] = np.rot90(np.flip(tmp, axis=2), 1, (0, 2))

    # ****************************** pseudo label (Inline direction) ********************************** #

    label_i = np.zeros([num1 * 4, 1, 128, 128, 128], dtype=np.int64)
    for i in range(num1):
        file_str = './datasets_F3/labeled/' + 'Label_pseudo_inline_dir_' + str(i).rjust(3, '0') + '.bin'
        tmp = np.fromfile(file_str, dtype=np.float32)
        tmp = np.reshape(tmp, [128, 128, 128], 'F')
        tmp = np.transpose(tmp, [2, 0, 1])
        label_i[i, 0, :, :, :] = tmp
        label_i[i + num1, 0, :, :, :] = np.rot90(tmp, 1, (0, 2))
        label_i[i + 2 * num1, 0, :, :, :] = np.flip(tmp, axis=2)
        label_i[i + 3 * num1, 0, :, :, :] = np.rot90(np.flip(tmp, axis=2), 1, (0, 2))

    # ****************************** pseudo label (Timeslice direction) ******************************** #

    label_t = np.zeros([num1 * 4, 1, 128, 128, 128], dtype=np.int64)
    for i in range(num1):
        file_str = './datasets_F3/labeled/' + 'Label_pseudo_timeslice_dir_' + str(i).rjust(3, '0') + '.bin'
        tmp = np.fromfile(file_str, dtype=np.float32)
        tmp = np.reshape(tmp, [128, 128, 128], 'F')
        tmp = np.transpose(tmp, [2, 0, 1])
        label_t[i, 0, :, :, :] = tmp
        label_t[i + num1, 0, :, :, :] = np.rot90(tmp, 1, (0, 2))
        label_t[i + 2 * num1, 0, :, :, :] = np.flip(tmp, axis=2)
        label_t[i + 3 * num1, 0, :, :, :] = np.rot90(np.flip(tmp, axis=2), 1, (0, 2))

    # ************************************** Unlabeled data ******************************************** #

    data_unlabeled = np.zeros([num2, 1, 128, 128, 128], dtype=np.float32)
    for i in range(num2):
        file_str = './datasets_F3/unlabeled/' + 'F3_128_128_128_' + str(i).rjust(3, '0') + '.bin'
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
