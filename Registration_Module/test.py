import torch
from models import UNet
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_field = UNet(in_channels=2, n_classes=1, depth=4, wf=5, up_mode='upconv')
model_field.to(device)
model_field.load_state_dict(torch.load('checkpoints/1/model_field.pkl'))
model_field.eval()

model_trans = UNet(in_channels=2, n_classes=2, depth=4, wf=4, up_mode='upconv')
model_trans.to(device)
model_trans.load_state_dict(torch.load('checkpoints/1/model_trans.pkl'))
model_trans.eval()


# read bin 3d datasets
data = np.fromfile('./reg_data/Image_128x128x128_000.bin', dtype=np.float32)
data = np.reshape(data, [128, 128, 128], 'F')
data = np.transpose(data, [1, 2, 0])
# data = np.transpose(data, [2, 0, 1])
data = (data - data.mean()) / data.std()

# read the annotation image
label_pr = plt.imread('./reg_data/Label_Inline64_000.png')
label_pr = torch.from_numpy(label_pr)
label_pr = label_pr.cuda()
label_pr = torch.unsqueeze(torch.unsqueeze(label_pr, 0), 0)

data_pr = data[63, :, :]
data_ne = data[68, :, :]
# data_pr = data[:, 63, :]
# data_ne = data[:, 69, :]
data_pr = torch.from_numpy(data_pr)
data_ne = torch.from_numpy(data_ne)
# data_pr = torch.flip(data_pr, dims=[0])  # TimeSlice Selection
# data_ne = torch.flip(data_ne, dims=[0])  # TimeSlice Selection
data_pr = torch.unsqueeze(torch.unsqueeze(data_pr, 0), 0)
data_ne = torch.unsqueeze(torch.unsqueeze(data_ne, 0), 0)
data_pr = data_pr.cuda()
data_ne = data_ne.cuda()


field = model_field(torch.concat([data_pr, data_ne], 1))

next_label = model_trans(torch.concat([label_pr, field], 1))


plt.imshow(torch.squeeze(torch.squeeze(data_pr)).cpu().numpy(), cmap='gray')
plt.show()
plt.imshow(torch.squeeze(torch.squeeze(data_ne)).cpu().numpy(), cmap='gray')
plt.show()
plt.imshow(torch.squeeze(torch.squeeze(field[0, 0, :, :])).cpu().detach().numpy(), cmap='seismic')
plt.show()
plt.imshow(torch.squeeze(torch.squeeze(label_pr)).cpu().numpy(), cmap='gray')
plt.show()
plt.imshow(torch.squeeze(torch.squeeze(next_label[0, 1, :, :])).cpu().detach().numpy(), cmap='gray')
plt.show()
plt.imshow(torch.sigmoid(torch.squeeze(torch.squeeze(next_label[0, 1, :, :]))).cpu().detach().numpy(), cmap='gray')
plt.show()


