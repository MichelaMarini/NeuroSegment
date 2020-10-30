# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:46:00 2019

@author: michela marini
"""

import torch
import os,sys
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot
import numpy as np
from torch.cuda import device
from tqdm import tqdm
from torch.utils.data import TensorDataset, ConcatDataset


import load
from collections import defaultdict
from multiprocessing import cpu_count
import time
import copy


import network

from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchsummary import summary
import torch.nn as nn

import torch.optim as optim
from torch.optim import lr_scheduler



N_epochs = 10


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, criterion, optimizer, device, scheduler, dataloaders, num_epochs= N_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()


        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            #for i, item in tqdm(enumerate(dataloaders[phase])):
            for inputs, labels in (dataloaders[phase]):
                inputs = Variable(inputs) #.cuda()
                labels = Variable(labels) #.cuda()
                inputs = inputs.to(device)
                labels = labels.to(device)
               # print("inputs: ", inputs.size())
               # print("labels: ", labels.size())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #print('output:', outputs)

                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        #print(loss)

                # statistics
                    epoch_samples += inputs.size(0)

            #print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples 

            
            print("LOSS1:", loss)
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.cuda.empty_cache()

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val loss: {:12f}'.format(best_loss))
        print("LOSS2:", loss)
        #print("Best val loss: ", best_loss)
    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.cuda.empty_cache()

class SimDataTrain(Dataset):
    def __init__(self, transform=None):
        original_imgs_train = './data_patches_9/data_patches/training_patches/images/'
        Masks_imgs_train = './data_patches_9/data_patches/training_patches/mask/'



        self.input_images_train = load.get_datasets(48600, original_imgs_train) #48600
        self.target_masks_train = load.get_datasets(48600, Masks_imgs_train)

        self.transform = transform

    def __len__(self):
        return len(self.input_images_train)

    def __getitem__(self, idx):
        image = self.input_images_train[idx]
        mask = self.target_masks_train[idx]
        image = np.float32(image)
        
        
        mask = np.float32(mask)

        image = np.expand_dims(image, axis=2)
        mask = np.expand_dims(mask, axis=2)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask



def main():
    #train

    patch_size = 48
    batch_size = 64
    
    num_classes = 1



    #=======================================================================================================================
    # settings
    print("number of cpus: ", cpu_count())

    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 0
              ,
                  'pin_memory': True} \

    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: ", device)


    # use same transform for train/val
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_ds = SimDataTrain(transform=trans)
    val_set = SimDataTrain(transform=trans)


    dataloaders = {

        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=False, **kwargs)
    }

    # Model summary

    #model = pytorch_unet_m2.UNet(num_classes)
    model = network.UNet(num_classes)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)


    summary(model, input_size=(1, patch_size, patch_size))  # number of channels, size images

    optimizer_ft = optim.SGD(model.parameters(), lr=0.1, momentum=0.7, weight_decay=1e-5, nesterov= True )
    criterion =  nn.BCEWithLogitsLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

    # Define the main training loop
    train_model(model, criterion, optimizer_ft, device, exp_lr_scheduler, dataloaders, num_epochs=N_epochs)

    torch.cuda.empty_cache()
    model.eval() #set model to the evaluation
    torch.cuda.empty_cache()

    #print("Model's state_dict:")
    #for param_tensor in model.state_dict():
    #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    print("-----------------------")

    #for var_name in optimizer_ft.state_dict():
    #    print(var_name, "\t", optimizer_ft.state_dict()[var_name])


    torch.save({'state_dict': model.state_dict()}, 'model.pth')

    checkpoint = {'model': network.UNet(1),
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer_ft.state_dict()}

    torch.save(checkpoint, 'new_10epochs.pth')

    
if __name__ == '__main__':
    main()

