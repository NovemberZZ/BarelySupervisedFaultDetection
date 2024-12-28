import time
import torch
import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from unet import UNet
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, \
    ConfusionMatrixDisplay
from scipy.ndimage import binary_dilation


# ********************************************  Mayavi  *************************************************** #
def data_Visualize3D(data_cube, colormap):
    source = mlab.pipeline.scalar_field(data_cube)
    source.spacing = [1, 1, -1]

    nx, ny, nz = data_cube.shape
    mlab.pipeline.image_plane_widget(source, plane_orientation='x_axes',
                                     slice_index=nx // 2, colormap=colormap)
    mlab.pipeline.image_plane_widget(source, plane_orientation='y_axes',
                                     slice_index=ny // 2, colormap=colormap)
    mlab.pipeline.image_plane_widget(source, plane_orientation='z_axes',
                                     slice_index=nz // 2, colormap=colormap)
    mlab.xlabel("Inline")
    mlab.ylabel("Crossline")
    mlab.zlabel("Time Samples")
    mlab.scalarbar(orientation='vertical')
    mlab.outline()
    mlab.show()

# ************************************** validation dataset ******************************************** #
val_data_num = 20
device = torch.device('cuda')

val_data = np.zeros([val_data_num, 1, 128, 128, 128], dtype=np.float32)
for i in range(val_data_num):
    file_str = './datasets/validation/' + 'Image_128x128x128_' + str(i).rjust(3, '0') + '.bin'
    tmp = np.fromfile(file_str, dtype=np.float32)
    tmp = np.reshape(tmp, [128, 128, 128], 'F')
    tmp = np.transpose(tmp, [1, 2, 0])
    tmp = (tmp - tmp.mean()) / tmp.std()
    val_data[i, 0, :, :, :] = tmp

val_data = torch.from_numpy(val_data)
dataset = Data.TensorDataset(val_data)
train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

val_labels = np.zeros([val_data_num, 1, 128, 128, 128], dtype=np.float32)
for i in range(val_data_num):
    file_str = './datasets/validation/' + 'Label_128x128x128_' + str(i).rjust(3, '0') + '.bin'
    tmp = np.fromfile(file_str, dtype=np.float32)
    tmp = np.reshape(tmp, [128, 128, 128], 'F')
    tmp = np.transpose(tmp, [1, 2, 0])

    # labels dilation
    kernel = np.ones((2, 2, 2))
    tmp = binary_dilation(tmp, kernel).astype(np.float32)
    # data_view = np.transpose(tmp, [0, 2, 1])
    # data_view = np.flip(data_view, axis=0)
    # data_Visualize3D(data_view, 'gray')
    # labels dilation
    val_labels[i, 0, :, :, :] = tmp

val_labels = torch.from_numpy(val_labels)

# ***************************************** Load model ************************************************* #
model = UNet(in_channels=1, n_classes=2, depth=6, wf=5)
model = model.to(device)
model.load_state_dict(torch.load('./checkpoints/labeled_30/seg2_d6_w5_lr1e_3_lambda1_epoch_100_90.pkl'))

# ******************************************* infer **************************************************** #
preds_prob = torch.zeros([val_data_num, 1, 128, 128, 128])  # probability
preds_bin = torch.zeros([val_data_num, 1, 128, 128, 128])  # binary

with torch.no_grad():
    for step, val_batch in enumerate(train_loader):
        print('progress:  %d' % (step + 1), '/', '%d' % val_data_num)
        val_data_batch = val_batch[0]

        val_preds_batch = model(val_data_batch.to(device)).cpu()
        val_preds_batch = F.softmax(val_preds_batch, dim=1)
        preds_prob[step, 0, :, :, :] = val_preds_batch[0, 1, :, :, :]

        val_preds_batch = torch.max(val_preds_batch, 1, keepdim=True)[1]
        kernel = np.ones((2, 2, 2))
        preds_bin[step, 0, :, :, :] = torch.from_numpy(binary_dilation(val_preds_batch[0, 0, :, :, :], kernel))

# ***************************************** estimation ************************************************* #
print(' ')
print('estimating...\n')
acc = accuracy_score(torch.reshape(val_labels, [val_data_num * 128 * 128 * 128, 1]),
                     torch.reshape(preds_bin, [val_data_num * 128 * 128 * 128, 1]))
print('accuracy: %.4f\n' % acc)

# the_confusion_matrix = confusion_matrix(torch.reshape(val_labels, [val_data_num * 128 * 128 * 128, 1]),
#                                         torch.reshape(preds_bin, [val_data_num * 128 * 128 * 128, 1]))
# print('confusion_matrix:')
# print(the_confusion_matrix)
# print(' ')

precision = precision_score(torch.reshape(val_labels, [val_data_num * 128 * 128 * 128, 1]),
                            torch.reshape(preds_bin, [val_data_num * 128 * 128 * 128, 1]))
print('precision: %.4f\n' % precision)

recall = recall_score(torch.reshape(val_labels, [val_data_num * 128 * 128 * 128, 1]),
                      torch.reshape(preds_bin, [val_data_num * 128 * 128 * 128, 1]))
print('recall: %.4f\n' % recall)

f1 = f1_score(torch.reshape(val_labels, [val_data_num * 128 * 128 * 128, 1]),
              torch.reshape(preds_bin, [val_data_num * 128 * 128 * 128, 1]))
print('f1_score: %.4f\n' % f1)

# disp = ConfusionMatrixDisplay(confusion_matrix=the_confusion_matrix)
# disp.plot()
# plt.show()


# ******************************************* visualize ************************************************** #
# preds_show = preds_prob[0, 0, :, :, :].numpy()
# data_view = np.transpose(preds_show, [0, 2, 1])
# data_view = np.flip(data_view, axis=0)
# data_Visualize3D(data_view, 'gray')

# plt.imshow(preds_show[:, 64, :])
# plt.show()
# pred_save = preds_prob[0, 0, :, :, :].numpy()
# pred_save = np.transpose(pred_save, [1, 2, 0])
# pred_save = np.reshape(pred_save, [1, 128*128*128], order='F')
# pred_save.tofile('Image_128x128x128_000_base_unet.bin')
