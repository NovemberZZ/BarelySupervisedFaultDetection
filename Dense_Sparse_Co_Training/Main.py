import torch
import time
import loaddata
import weights
import losses
from unet import UNet
from itertools import cycle
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ************************************************** parameters ************************************************** #
Epoch = 100
batch_size_labeled = 1
batch_size_unlabeled = 2
Net_depth = 6
Net_wf = 5
learning_rate = 1 * 1e-3
weight_map_steps = Epoch
lambda_max = 0.10
lamda_steps = Epoch
# ************************************************** parameters ************************************************** #


# ************************************************ load datasets ************************************************* #
dataset_1, dataset_2 = loaddata.loaddataset_F3()
train_loader_1 = DataLoader(dataset=dataset_1, batch_size=batch_size_labeled, shuffle=True)
train_loader_2 = DataLoader(dataset=dataset_2, batch_size=batch_size_unlabeled, shuffle=True)

# ************************************************** load model ************************************************** #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seg_model1 = UNet(in_channels=1, n_classes=2, depth=Net_depth, wf=Net_wf, up_mode='upsample')
seg_model1.to(device)
seg_model2 = UNet(in_channels=1, n_classes=2, depth=Net_depth, wf=Net_wf, up_mode='upsample')
seg_model2.to(device)

# ************************************************* optimizer **************************************************** #
optimizer1 = torch.optim.AdamW(seg_model1.parameters(), lr=learning_rate)
optimizer2 = torch.optim.AdamW(seg_model2.parameters(), lr=learning_rate)
# scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=10, gamma=0.5, last_epoch=- 1, verbose=False)
# scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=10, gamma=0.5, last_epoch=- 1, verbose=False)

# ************************************************** training **************************************************** #
time_start = time.time()
iter_num = 0
max_iterations = Epoch * len(dataset_1.tensors[0])
for epoch in range(Epoch):
    for step, train_batch in enumerate(zip(train_loader_1, cycle(train_loader_2))):

        # *************************************** extraction data ************************************************ #
        dataset_1, dataset_2 = train_batch
        data_labeled, label_i, label_t = dataset_1
        data_unlabeled = dataset_2[0]

        # ******************************************* SEG MODEL 1 ************************************************ #

        # ************************************* propagate labeled data ******************************************* #
        preds = seg_model1(data_labeled.to(device))
        weight = weights.load_weight_inline(
            weights.get_current_coefficient((iter_num / (max_iterations / weight_map_steps)), weight_map_steps))
        loss_seg_ce1 = losses.wce(preds, label_i.to(device), weight, batch_size_labeled, 128, 128, 128)

        preds = F.softmax(preds, dim=1)
        loss_seg_dice1 = losses.dice_loss_weight(torch.unsqueeze(preds[:, 1, :, :, :], 1), label_i.to(device),
                                                 torch.unsqueeze(torch.unsqueeze(weight[0, 0, :, :, :], 0), 0))
        supervised_loss1 = 0.5 * loss_seg_ce1 + 0.5 * loss_seg_dice1

        # ************************************ propagate unlabeled data ****************************************** #
        preds = seg_model1(data_unlabeled.to(device))
        with torch.no_grad():
            pseudo_label_preds = seg_model2(data_unlabeled.to(device))
        pseudo_label_preds = F.softmax(pseudo_label_preds, dim=1)
        pseudo_label_preds = torch.argmax(pseudo_label_preds.detach(), dim=1, keepdim=True)

        # ****************************** get the mask for cross supervision ************************************** #
        T = 8
        preds_tmp = torch.zeros([T, batch_size_unlabeled, 2, 128, 128, 128])
        for i in range(T):
            with torch.no_grad():
                preds_tmp[i, :, :, :, :] = seg_model2(
                    data_unlabeled.to(device) + torch.clamp(torch.randn_like(data_unlabeled) * 0.1, -0.4, 0.4).cuda())
        mask = weights.load_mask(iter_num, max_iterations, preds_tmp)
        cross_supervision_loss1 = losses.wce(preds, pseudo_label_preds, mask, batch_size_unlabeled, 128, 128, 128)

        # **************************************** dynamic weight ************************************************ #
        current_lambda = weights.get_current_lambda_weight(lambda_max, (iter_num / (max_iterations / lamda_steps)),
                                                           lamda_steps)
        loss1 = (1 - current_lambda) * supervised_loss1 + current_lambda * cross_supervision_loss1

        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        # ******************************************* SEG MODEL 2 ************************************************ #

        # ************************************* propagate labeled data ******************************************* #
        preds = seg_model2(data_labeled.to(device))
        weight = weights.load_weight_timeslice(
            weights.get_current_coefficient((iter_num / (max_iterations / weight_map_steps)), weight_map_steps))
        loss_seg_ce2 = losses.wce(preds, label_t.to(device), weight, batch_size_labeled, 128, 128, 128)

        preds = F.softmax(preds, dim=1)
        loss_seg_dice2 = losses.dice_loss_weight(torch.unsqueeze(preds[:, 1, :, :, :], 1), label_t.to(device),
                                                 torch.unsqueeze(torch.unsqueeze(weight[0, 0, :, :, :], 0), 0))
        supervised_loss2 = 0.5 * loss_seg_ce2 + 0.5 * loss_seg_dice2

        # ************************************ propagate unlabeled data ****************************************** #
        preds = seg_model2(data_unlabeled.to(device))
        with torch.no_grad():
            pseudo_label_preds = seg_model1(data_unlabeled.to(device))
        pseudo_label_preds = F.softmax(pseudo_label_preds, dim=1)
        pseudo_label_preds = torch.argmax(pseudo_label_preds.detach(), dim=1, keepdim=True)

        # ****************************** get the mask for cross supervision ************************************** #
        T = 8
        preds_tmp = torch.zeros([T, batch_size_unlabeled, 2, 128, 128, 128])
        for i in range(T):
            with torch.no_grad():
                preds_tmp[i, :, :, :, :] = seg_model2(
                    data_unlabeled.to(device) + torch.clamp(torch.randn_like(data_unlabeled) * 0.1, -0.4, 0.4).cuda())
        mask = weights.load_mask(iter_num, max_iterations, preds_tmp)
        cross_supervision_loss2 = losses.wce(preds, pseudo_label_preds, mask, batch_size_unlabeled, 128, 128, 128)

        # **************************************** dynamic weight ************************************************ #
        loss2 = (1 - current_lambda) * supervised_loss2 + current_lambda * cross_supervision_loss2

        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        iter_num = iter_num + 1
        time_end = time.time()

        if iter_num % ((max_iterations // Epoch) // 10) == 0:
            print('Epoch: %.3d' % (epoch + 1), '|',
                  'Iter: %.4d' % iter_num, '  ||  ',

                  'CE_1: %.4f' % loss_seg_ce1, '|',
                  'Dice_1: %.4f' % loss_seg_dice1, '|',
                  'SUPV_1: %.4f' % supervised_loss1, '|',
                  'Cross_1: %.4f' % cross_supervision_loss1, '|',
                  'Seg_1: %.4f' % loss1, '  ||  ',

                  'CE_2: %.4f' % loss_seg_ce2, '|',
                  'Dice_2: %.4f' % loss_seg_dice2, '|',
                  'SUPV_2: %.4f' % supervised_loss2, '|',
                  'Cross_2: %.4f' % cross_supervision_loss2, '|',
                  'Seg_2: %.4f' % loss2, '  ||  ',

                  'Lambda: %.4f' % current_lambda, '|',
                  'Time: %.2f' % (time_end - time_start), 's')

    # scheduler1.step(epoch)
    # scheduler2.step(epoch)

    if (epoch + 1) % 5 == 0:
        torch.save(seg_model1.state_dict(),
                   './checkpoints/F3Data/aug_seg1_d6_w5_lr1e_3_lambda1_epoch_' + str(Epoch) + '_' + str(epoch + 1) + '.pkl')
        torch.save(seg_model2.state_dict(),
                   './checkpoints/F3Data/aug_seg2_d6_w5_lr1e_3_lambda1_epoch_' + str(Epoch) + '_' + str(epoch + 1) + '.pkl')
