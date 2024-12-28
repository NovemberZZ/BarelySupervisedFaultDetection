import torch
import cv2
import numpy as np
from mayavi import mlab
from models import UNet
import matplotlib.pyplot as plt


# ******************************************** load models ****************************************** #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# inline direction reg models
model_field = UNet(in_channels=2, n_classes=1, depth=4, wf=5, up_mode='upconv')
model_field.to(device)
model_field.load_state_dict(torch.load('checkpoints/model_field.pkl'))
model_field.eval()

model_trans = UNet(in_channels=2, n_classes=2, depth=4, wf=4, up_mode='upconv')
model_trans.to(device)
model_trans.load_state_dict(torch.load('checkpoints/model_trans.pkl'))
model_trans.eval()

# vertical direction reg models
t_model_field = UNet(in_channels=2, n_classes=1, depth=4, wf=5, up_mode='upconv')
t_model_field.to(device)
t_model_field.eval()
t_model_field.load_state_dict(torch.load('checkpoints/t_model_field.pkl'))

t_model_trans = UNet(in_channels=2, n_classes=2, depth=4, wf=4, up_mode='upconv')
t_model_trans.to(device)
t_model_trans.eval()
t_model_trans.load_state_dict(torch.load('checkpoints/t_model_trans.pkl'))

# ******************************************** load datasets ****************************************** #

for num in range(0, 20):

    # read bin 3d datasets
    data_cube = np.fromfile('./reg_data_Poseidon/Poseidon_128_128_128_' + str(num).rjust(3, '0') + '.bin', dtype=np.float32)
    data_cube = np.reshape(data_cube, [128, 128, 128], 'F')
    data_cube = np.transpose(data_cube, [2, 0, 1])

    # read the annotation image
    label_mid = plt.imread('./reg_data_Poseidon/Label_InLine64_' + str(num).rjust(3, '0') + '.png')
    label_mid = torch.from_numpy(label_mid)
    label_mid = label_mid.cuda()
    label_mid = torch.unsqueeze(torch.unsqueeze(label_mid, 0), 0)

    t_label_mid = plt.imread('./reg_data_Poseidon/Label_TimeSlice64_' + str(num).rjust(3, '0') + '.png')
    t_label_mid = torch.from_numpy(t_label_mid)
    t_label_mid = torch.flip(t_label_mid, dims=[0])
    t_label_mid = t_label_mid.cuda()
    t_label_mid = torch.unsqueeze(torch.unsqueeze(t_label_mid, 0), 0)

    # ******************************************* pseudo labels ***************************************** #
    label_cube = np.zeros([128, 128, 128], dtype=np.float32)
    label_cube[63, :, :] = label_mid[0, 0, :, :].cpu().numpy()
    t_label_cube = np.zeros([128, 128, 128], dtype=np.float32)
    t_label_cube[:, 63, :] = t_label_mid[0, 0, :, :].cpu().numpy()

    # ******************************************** threshold ******************************************** #
    # threshold of binarize
    threshold1 = 0.89
    threshold2 = 0.89

    # *********************************************** loop ********************************************** #

    for i in range(63, 127):
        # inline direction, 64 to 128
        data_pr = data_cube[i, :, :]
        data_pr = (data_pr - data_pr.mean()) / data_pr.std()
        data_ne = data_cube[i + 1, :, :]
        data_ne = (data_ne - data_ne.mean()) / data_ne.std()
        data_pr = torch.from_numpy(data_pr)
        data_ne = torch.from_numpy(data_ne)
        data_pr = torch.unsqueeze(torch.unsqueeze(data_pr, 0), 0)
        data_ne = torch.unsqueeze(torch.unsqueeze(data_ne, 0), 0)
        data_pr = data_pr.cuda()
        data_ne = data_ne.cuda()

        field = model_field(torch.concat([data_pr, data_ne], 1))
        label_pr = label_mid if i == 63 else torch.unsqueeze(torch.unsqueeze(next_label, 0), 0)
        next_label = model_trans(torch.concat([label_pr, field], 1))

        # binarize & save to cube
        next_label = next_label[0, 1, :, :]
        next_label = torch.sigmoid(next_label)
        next_label = torch.where(next_label > threshold1, torch.ones_like(next_label), torch.zeros_like(next_label))
        label_cube[i + 1, :, :] = next_label.cpu().detach().numpy()

        # ************************************************************************************************ #
        # time slice direction, 64 to 128
        t_data_pr = data_cube[:, i, :]
        t_data_pr = (t_data_pr - t_data_pr.mean()) / t_data_pr.std()
        t_data_ne = data_cube[:, i + 1, :]
        t_data_ne = (t_data_ne - t_data_ne.mean()) / t_data_ne.std()
        t_data_pr = torch.from_numpy(t_data_pr)
        t_data_ne = torch.from_numpy(t_data_ne)
        t_data_pr = torch.unsqueeze(torch.unsqueeze(t_data_pr, 0), 0)
        t_data_ne = torch.unsqueeze(torch.unsqueeze(t_data_ne, 0), 0)
        t_data_pr = t_data_pr.cuda()
        t_data_ne = t_data_ne.cuda()

        t_field = t_model_field(torch.concat([t_data_pr, t_data_ne], 1))
        t_label_pr = t_label_mid if i == 63 else torch.unsqueeze(torch.unsqueeze(t_next_label, 0), 0)
        t_next_label = t_model_trans(torch.concat([t_label_pr, t_field], 1))

        # binarize & save to cube
        t_next_label = t_next_label[0, 1, :, :]
        t_next_label = torch.sigmoid(t_next_label)
        t_next_label = torch.where(t_next_label > threshold2, torch.ones_like(t_next_label), torch.zeros_like(t_next_label))
        t_label_cube[:, i + 1, :] = t_next_label.cpu().detach().numpy()

    # ************************************************************************************************** #

    for j in range(63, 0, -1):
        # inline direction, 64 to 1
        data_pr = data_cube[j, :, :]
        data_pr = (data_pr - data_pr.mean()) / data_pr.std()
        data_ne = data_cube[j - 1, :, :]
        data_ne = (data_ne - data_ne.mean()) / data_ne.std()
        data_pr = torch.from_numpy(data_pr)
        data_ne = torch.from_numpy(data_ne)
        data_pr = torch.unsqueeze(torch.unsqueeze(data_pr, 0), 0)
        data_ne = torch.unsqueeze(torch.unsqueeze(data_ne, 0), 0)
        data_pr = data_pr.cuda()
        data_ne = data_ne.cuda()

        field = model_field(torch.concat([data_pr, data_ne], 1))
        label_pr = label_mid if j == 63 else torch.unsqueeze(torch.unsqueeze(next_label, 0), 0)
        next_label = model_trans(torch.concat([label_pr, field], 1))

        # binarize & save to cube
        next_label = next_label[0, 1, :, :]
        next_label = torch.sigmoid(next_label)
        next_label = torch.where(next_label > threshold1, torch.ones_like(next_label), torch.zeros_like(next_label))
        label_cube[j - 1, :, :] = next_label.cpu().detach().numpy()

        # ************************************************************************************************ #
        # time slice direction, 64 to 1
        t_data_pr = data_cube[:, j, :]
        t_data_pr = (t_data_pr - t_data_pr.mean()) / t_data_pr.std()
        t_data_ne = data_cube[:, j - 1, :]
        t_data_ne = (t_data_ne - t_data_ne.mean()) / t_data_ne.std()
        t_data_pr = torch.from_numpy(t_data_pr)
        t_data_ne = torch.from_numpy(t_data_ne)
        t_data_pr = torch.unsqueeze(torch.unsqueeze(t_data_pr, 0), 0)
        t_data_ne = torch.unsqueeze(torch.unsqueeze(t_data_ne, 0), 0)
        t_data_pr = t_data_pr.cuda()
        t_data_ne = t_data_ne.cuda()

        t_field = t_model_field(torch.concat([t_data_pr, t_data_ne], 1))
        t_label_pr = t_label_mid if j == 63 else torch.unsqueeze(torch.unsqueeze(t_next_label, 0), 0)
        t_next_label = t_model_trans(torch.concat([t_label_pr, t_field], 1))

        # binarize & save to cube
        t_next_label = t_next_label[0, 1, :, :]
        t_next_label = torch.sigmoid(t_next_label)
        t_next_label = torch.where(t_next_label > threshold2, torch.ones_like(t_next_label), torch.zeros_like(t_next_label))
        t_label_cube[:, j - 1, :] = t_next_label.cpu().detach().numpy()

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

    data_view = np.transpose(label_cube, [0, 2, 1])
    data_view = np.flip(data_view, axis=0)
    data_Visualize3D(data_view, 'gray')

    data_view = np.transpose(t_label_cube, [0, 2, 1])
    data_view = np.flip(data_view, axis=0)
    data_Visualize3D(data_view, 'gray')

    label_cube = np.transpose(label_cube, [1, 2, 0])
    label_cube = np.reshape(label_cube, [128 * 128 * 128], 'F')
    label_cube.tofile('./reg_data_Poseidon/Label_pseudo_inline_dir_' + str(num).rjust(3, '0') + '.bin')

    t_label_cube = np.transpose(t_label_cube, [1, 2, 0])
    t_label_cube = np.reshape(t_label_cube, [128 * 128 * 128], 'F')
    t_label_cube.tofile('./reg_data_Poseidon/Label_pseudo_timeslice_dir_' + str(num).rjust(3, '0') + '.bin')
