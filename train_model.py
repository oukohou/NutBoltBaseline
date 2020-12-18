# -*- coding: utf-8 -*-

__author__ = 'kohou.wang'
__time__ = '19-11-20'
__email__ = 'oukohou@outlook.com'

# If this runs wrong, don't ask me, I don't know why;
# If this runs right, thank god, and I don't know why.
# Maybe the answer, my friend, is blowing in the wind.
# Well, I'm kidding... Always, Welcome to contact me.

"""Description for the script:
train models.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import time
import copy
import pandas as pd
from datasets.read_data import ReadImageDataset
from torch.nn import init
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

from models.EfficientNet import EfficientNet

print('torch.cuda.device_count : {}'.format(torch.cuda.device_count()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model_without_valid(model_, dataloaders_, model_criterion_, optimizer_, num_epochs_=25, Use_Two_outputs=False,
                              save_path_format_=''):
    global lr_scheduler
    
    since = time.time()
    val_acc_history = []
    best_epoch_acc = 0.0
    best_model_weights = copy.deepcopy(model_.state_dict())
    for epoch in range(num_epochs_):
        print('\nEpoch {}/{}'.format(epoch, num_epochs_ - 1))
        print('-' * 10)
        
        model_.train()  # Set model to training mode
        print('in train mode...')
        
        running_loss = 0.0
        running_corrects_1 = 0
        for i, (inputs, labels) in enumerate(dataloaders_['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # import ipdb
            # ipdb.set_trace()
            # zero the parameter gradients
            optimizer_.zero_grad()
            
            # track history if only in train
            with torch.set_grad_enabled(True):
                logits = model_(inputs)
                logits_loss = model_criterion_(logits, labels)
                _, preds = torch.max(logits, 1)
                
                logits_loss.backward()
                optimizer_.step()
            
            # statistics
            running_loss += logits_loss.item() * inputs.size(0)
            running_corrects_1 += torch.sum(preds == labels.data)  # .long())
        
        print("labels list:{}".format(labels))
        print("preds list:{}".format(preds))
        epoch_loss = running_loss / len(dataloaders_['train'].dataset)
        epoch_acc = running_corrects_1.double() / len(dataloaders_['train'].dataset)
        val_acc_history.append(epoch_acc)
        
        print('{} Loss: {:.4f} pred_acc: {:.4f}'.format('train', epoch_loss, epoch_acc))
        time_elapsed = time.time() - since
        print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        lr_scheduler.step(epoch)
        
        if epoch % 10 == 0 and save_path_format_ != '':
            save_path_ = save_path_format_.format(
                "input{}".format(input_size), use_different_lr, batch_size,
                learning_rate,
                epoch
            )
            torch.save(
                model_to_train.state_dict()
                ,
                save_path_
            )
        
        # save best acc's model weights
        if epoch_acc > best_epoch_acc:
            best_epoch_acc = epoch_acc
            best_model_weights = copy.deepcopy(model_.state_dict())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Directly use last train epoch_acc: {:4f}'.format(epoch_acc))
    print("epoch acc list:{}, \nbest epoch at:{}".format(val_acc_history, val_acc_history.index(max(val_acc_history))))
    
    return model_, best_model_weights


if __name__ == "__main__":
    train_data_base_path = '/home/kohou/cvgames/interest/contest/shanDong/ZaoZhuang/datasets/螺母螺栓产品智能检测/螺栓质量检测-训练集/螺栓质量检测-训练集'
    batch_size = 8
    input_size = 300  # for efficientbet_b3
    num_epochs = 30
    num_classes = 2
    learning_rate = 0.002  # originally 0.001
    weight_decay = 1e-4  # originally 1e-4
    mixup_alpha = 0.4  # originally 1.
    use_base_data_path = True
    load_pretrained = True
    use_different_lr = False
    finetune_fc_only = False
    
    model_to_train = EfficientNet.from_name('efficientnet-b3', override_params={'num_classes': 1000})
    if load_pretrained:
        print("loading model...")
        loaded_model = torch.load(
            './pretrained_models/efficientnet-b3-5fb5a3c3.pth',
        )
        model_to_train.load_state_dict(loaded_model)  # torch's pretrained model
        del loaded_model
    print(model_to_train._fc)
    model_to_train._fc = nn.Linear(1536, num_classes)  # efficientnet-b3
    # # # #
    for m in model_to_train.modules():
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                init.constant_(m.bias, 0)
    # #
    if finetune_fc_only:
        print('finetune fc layer only...')
        for name, param in model_to_train.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    
    total_images = pd.read_csv('datasets/train.csv')
    total_images = total_images.sample(frac=1., random_state=2020)
    print("total images:{}".format(len(total_images)))
    train_data_list = total_images
    # train_data_list, val_data_list = train_test_split(total_images, test_size=0.1, random_state=2019 + 1 + 1)
    
    train_gen = ReadImageDataset(total_images, train_data_base_path, mode="train",
                                 input_size=input_size,
                                 use_base_data_path=use_base_data_path
                                 )
    # val_gen = ReadImageDataset(val_data_list, train_data_base_path,
    #                            auto_augment=auto_augment, input_size=input_size,
    #                            mode="train", cutout=use_cutout,
    #                            )
    
    train_loader = DataLoader(train_gen, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=4,
                              # drop_last=True
                              )
    # val_loader = DataLoader(val_gen, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    
    total_dataloader = {
        'train': train_loader,
        # 'val': val_loader,
    }
    
    model_to_train = model_to_train.to(device)
    # model_to_train = model_to_train.cuda()
    
    params_to_update = model_to_train.parameters()
    # Observe parameters that are being optimized
    if not finetune_fc_only and not use_different_lr:
        optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif not use_different_lr:
        print("finetune fc only.")
        optimizer_ft = optim.SGD(model_to_train._fc.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)  # efficientnet
    else:
        print("use different lr.")
        # backbone_params = []
        fc_params = list(map(id, model_to_train._fc.parameters()))  # efficientnet
        backbone_params = filter(lambda x: id(x) not in fc_params, model_to_train.parameters())
        optimizer_ft = optim.SGD(
            params=[{'params': backbone_params, 'lr': 0.1 * learning_rate},
                    {'params': model_to_train._fc.parameters(), 'lr': learning_rate}],  # efficientnet
            lr=learning_rate, momentum=0.9, weight_decay=weight_decay
        )
    
    value_counts = train_data_list['label'].value_counts().to_dict()
    label_num = [value_counts[i] for i in range(len(value_counts))]
    ratio = [sum(label_num) / i for i in label_num]
    weight_ratio = [i / sum(ratio) for i in ratio]
    
    print("train images:{}, valid images: {}".format(len(train_data_list), 0))
    print("train label:{}".format(train_data_list['label'].value_counts().to_dict()))
    # print("val label:{}".format(val_data_list['label'].value_counts().to_dict()))
    print("train weights:{}".format(weight_ratio))
    
    criterion = nn.CrossEntropyLoss(torch.tensor(weight_ratio).to(device))
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=num_epochs // 3, gamma=0.1)
    
    save_path_format = \
        './trained_models/resneSt101/{}_DifferentLR__{}_batch{}_lr{}_epoch{}.pth'
    
    # # train without evaluate, use the last epoch directly. No bicycle!
    model_to_train, best_model_weights_trained = train_model_without_valid(model_to_train, total_dataloader,
                                                                           criterion, optimizer_ft,
                                                                           num_epochs_=num_epochs, save_path_format_=save_path_format
                                                                           )
    save_path = './trained_models/resneSt101/Best_{}_DifferentLR__{}_batch{}_lr{}_epoch{}.pth'.format(
        "input{}".format(input_size), use_different_lr,
        batch_size, learning_rate,
        num_epochs)
    torch.save(
        model_to_train.state_dict(),
        save_path
    )
    torch.save(best_model_weights_trained,
               os.path.join(os.path.dirname(save_path), "best_epoch_{}".format(os.path.basename(save_path))))
    print('model saved to {}.'.format(save_path))
