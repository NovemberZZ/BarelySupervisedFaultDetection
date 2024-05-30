import time
import torch
import loaddata
from torch.utils.data import DataLoader
from models import UNet


# load datasets
training_set = loaddata.load_dataset()
train_loader = DataLoader(dataset=training_set, batch_size=10, shuffle=True)

# model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_field = UNet(in_channels=2, n_classes=1, depth=4, wf=5, up_mode='upconv')
model_field.to(device)

model_trans = UNet(in_channels=2, n_classes=2, depth=4, wf=4, up_mode='upconv')
model_trans.to(device)

Epoch = 200
optimizer = torch.optim.AdamW([{'params': model_field.parameters(), 'lr': 6e-5},
                               {'params': model_trans.parameters(), 'lr': 2e-5}])
loss_func = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.05, 0.95]).cuda())

model_field.train()
model_trans.train()

time_start = time.time()
for epoch in range(Epoch):
    for step, training_set in enumerate(train_loader):

        data_pr, data_ne, label_pr, label_ne = training_set
        data_pr = data_pr.cuda()
        data_ne = data_ne.cuda()
        label_pr = label_pr.cuda()
        label_ne = label_ne.cuda()

        field = model_field(torch.concat([data_pr, data_ne], 1))

        next_label = model_trans(torch.concat([label_pr, field], 1))

        step_loss = loss_func(next_label, torch.squeeze(label_ne).long())

        optimizer.zero_grad()
        step_loss.backward()
        optimizer.step()

        if (step + 1) % len(train_loader) == 0:

            elapse_time = time.time() - time_start
            print('Epoch: ', epoch + 1, '| train loss: %.4f' % step_loss, '| elapse: %.2f s' % elapse_time)

    if (epoch + 1) % 20 == 0:
        torch.save(model_field.state_dict(), './checkpoints/2/model_field_epoch_' + str(epoch + 1) + '.pkl')
        torch.save(model_trans.state_dict(), './checkpoints/2/model_trans_epoch_' + str(epoch + 1) + '.pkl')

