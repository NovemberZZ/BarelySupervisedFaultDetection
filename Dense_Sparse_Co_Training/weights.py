import torch
import numpy as np
# from mayavi import mlab
import torch.nn.functional as F


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def load_weight_inline(k):
    weight = np.zeros([128, 128, 128], dtype=np.float32)
    for i in range(0, 128):
        weight[i, :, :] = k ** pow(abs(i - 63), 1/3)
    weight[:, 63, :] = 1
    weight = torch.from_numpy(weight)
    weight = weight.cuda()
    weight = torch.unsqueeze(torch.unsqueeze(weight, 0), 0)
    weight = weight.repeat(1, 2, 1, 1, 1)
    return weight


def load_weight_timeslice(k):
    weight = np.zeros([128, 128, 128], dtype=np.float32)
    for i in range(0, 128):
        weight[:, i, :] = k ** pow(abs(i - 63), 1/3)
        # print(k ** pow(abs(i - 63), 1/3))
    weight[63, :, :] = 1
    weight = torch.from_numpy(weight)
    weight = torch.unsqueeze(torch.unsqueeze(weight, 0), 0)
    weight = weight.cuda()
    weight = weight.repeat(1, 2, 1, 1, 1)
    return weight


def load_mask(iter_num, iter_max, preds):
    preds = F.softmax(preds, dim=2)
    preds = torch.mean(preds, dim=0)
    uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
    # threshold = (0.75 + 0.25 * sigmoid_rampup(iter_num, iter_max)) * np.log(2)
    # mask = (uncertainty < threshold).float()

    # modified
    threshold = (0.75 + 0.25 * sigmoid_rampup(iter_num, iter_max)) * np.log(2.25)
    mask = torch.zeros(np.shape(uncertainty))
    ind1 = np.where(uncertainty > threshold)
    ind2 = np.where(uncertainty < threshold)
    mask[ind1] = uncertainty[ind1] * 0.1
    mask[ind2] = 1
    # modified

    mask = mask.cuda()
    mask = mask.repeat(1, 2, 1, 1, 1)
    return mask


def get_current_lambda_weight(lambda_max, epoch, lamda_step):
    return lambda_max * sigmoid_rampup(epoch, lamda_step)


def get_current_coefficient(current, rampdown_length):
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


# def data_Visualize3D(data_cube, colormap):
#     source = mlab.pipeline.scalar_field(data_cube)
#     source.spacing = [1, 1, -1]
#
#     nx, ny, nz = data_cube.shape
#     mlab.pipeline.image_plane_widget(source, plane_orientation='x_axes',
#                                      slice_index=nx // 2, colormap=colormap)
#     mlab.pipeline.image_plane_widget(source, plane_orientation='y_axes',
#                                      slice_index=ny // 2, colormap=colormap)
#     mlab.pipeline.image_plane_widget(source, plane_orientation='z_axes',
#                                      slice_index=nz // 2, colormap=colormap)
#     mlab.xlabel("Inline")
#     mlab.ylabel("Crossline")
#     mlab.zlabel("Time Samples")
#     mlab.scalarbar(orientation='vertical')
#     mlab.outline()
#     mlab.show()


if __name__ == '__main__':

    # weight = load_weight_timeslice(0.8)
    # data_view = np.transpose(weight, [0, 2, 1])
    # data_view = np.flip(data_view, axis=0)
    # data_Visualize3D(data_view, 'gray')

    print(get_current_coefficient(50, 50))
    print('********************************\n')
    load_weight_timeslice(0)

