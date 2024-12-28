import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from unet import UNet
from functions import segy
from functions.window_function import get_padding_data_and_sliding_ind_3d


# ################################ Parameters Setting ################################ #
print('Loading data...')

# SEGY Data
input_path = './Poseidon3D_part.sgy'
input_data = segy.readsegy(filename_in=input_path, inline_byte=9, xline_byte=21)

input_data = np.rot90(input_data, 3, (0, 2))

# ##################################################################################### #
model_path = ''
# ##################################################################################### #
overlap = 64
# ##################################################################################### #
save_data = True
output_path = ''
# ##################################################################################### #
show_result = False
show_inline = 300
show_xline = 250
show_timeslice = 100
# #################################################################################### #

time_start = time.time()

# Input Data Shape: [Inline, Sample, Xline]
inline_n = np.size(input_data, 0)
time_n = np.size(input_data, 1)
xline_n = np.size(input_data, 2)

print('Loading model...')
model = UNet(in_channels=1, n_classes=2, depth=6, wf=5, up_mode='upconv')
model = model.cuda()
model.load_state_dict(torch.load(model_path))

print('Pre-processing...')
# 根据数据大小与重叠区域计算数据边界填充与窗口衰减
padding_data, xline_ind, time_ind, inline_ind, padding_xline_n, padding_time_n, padding_inline_n, window_coefficient = \
    get_padding_data_and_sliding_ind_3d(input_data, 128, overlap)
output_data = np.zeros_like(padding_data, dtype=np.double)

print('Start Recognition...')
counter = 1
# model.eval()
with torch.no_grad():
    for inline_i in range(inline_ind.shape[0]):
        for time_i in range(time_ind.shape[0]):
            for xline_i in range(xline_ind.shape[0]):
                print(counter, '/', (inline_ind.shape[0] * time_ind.shape[0] * xline_ind.shape[0]))
                counter += 1

                sub_ind_inline = inline_ind[inline_i]
                sub_ind_time = time_ind[time_i]
                sub_ind_xline = xline_ind[xline_i]
                sub_input = padding_data[sub_ind_inline[0]:(sub_ind_inline[1] + 1),
                            sub_ind_time[0]:(sub_ind_time[1] + 1), sub_ind_xline[0]:(sub_ind_xline[1] + 1)]
                # normalization
                sub_input = (sub_input - np.mean(sub_input)) / np.std(sub_input)
                # move to tensor
                input_tensor = torch.from_numpy(sub_input).unsqueeze(0)
                input_tensor = input_tensor.unsqueeze(0)

                # move to device
                input_tensor = input_tensor.cuda()
                # pred
                out = model(input_tensor)

                probability = F.softmax(out, dim=1)
                # probability = out
                channel_ind = torch.squeeze(torch.max(probability, 1)[1]).cpu().numpy()
                probability_tmp = torch.squeeze(probability)

                channel_ind = np.where(channel_ind == 1)
                predict_probability = np.zeros(
                    (probability_tmp.size()[1], probability_tmp.size()[2], probability_tmp.size()[3]))
                predict_probability[channel_ind] = probability_tmp[1][channel_ind].cpu().numpy()  # 预测概率

                output_data[sub_ind_inline[0]:(sub_ind_inline[1] + 1), sub_ind_time[0]:(sub_ind_time[1] + 1),
                sub_ind_xline[0]:(sub_ind_xline[1] + 1)] += predict_probability * window_coefficient

output_data = output_data[padding_inline_n:(padding_inline_n + inline_n), padding_time_n:(padding_time_n + time_n),
              padding_xline_n:(padding_xline_n + xline_n)]
output_data.astype(np.float32)

output_data = np.rot90(output_data, 1, (0, 2))

time_end = time.time()
print('Complete Recognition...')
print('Time-consuming:', '%.2f' % (time_end - time_start), 's')

if save_data:
    print('Saving data...')
    segy.writesegy(filename_in=input_path, inline_byte=9, xline_byte=21, data=output_data, filename_out=output_path)

if show_result:
    input_data = np.rot90(input_data, 3, (0, 2))
    plt.subplot(3, 2, 1)
    plt.imshow(input_data[show_inline, :, :], cmap='seismic')
    plt.subplot(3, 2, 2)
    plt.imshow(output_data[show_inline, :, :], cmap='jet')
    plt.subplot(3, 2, 3)
    plt.imshow(np.transpose(input_data[:, :, show_xline], [1, 0]), cmap='seismic')
    plt.subplot(3, 2, 4)
    plt.imshow(np.transpose(output_data[:, :, show_xline], [1, 0]), cmap='jet')
    plt.subplot(3, 2, 5)
    plt.imshow(np.flip(input_data[:, show_timeslice, :], axis=0), cmap='seismic')
    plt.subplot(3, 2, 6)
    plt.imshow(np.flip(output_data[:, show_timeslice, :], axis=0), cmap='jet')
    plt.show()
