# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sys
import os, argparse
import time, random
import torchvision.models as models
import pickle
from scipy import misc
from skimage import io
from PIL import Image
from pprint import pprint

vgg_model = models.vgg19_bn(pretrained=True)
vgg_model.classifier = nn.Sequential(*list(vgg_model.classifier.children())[:4])
# pprint(vgg_model.classifier)

BATCH_SIZE = 8

def take_input(data_folder):
    """ returns a data loader """
    transformer = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Scale(256),
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    vqa_image_dataset = datasets.ImageFolder(
        root=data_folder,
        transform=transformer
    )
    data_loader = DataLoader(dataset=vqa_image_dataset, 
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=8)
    """
    for i, x in enumerate(vqa_image_dataset):
        print(x)
        break
    """
    return data_loader

def get_vgg(folder, data_loader):
    print("DONT CALL THIS")
    return
    """ writes the vgg feature vector """
    new_folder = "{}_vgg".format(folder)
    print('replace inputs to inputs.cuda() on HPC')
    for batch_index, (inputs, label) in enumerate(data_loader):
        print('here')
        inputs = autograd.Variable(inputs, volatile=True)
        print('get_vgg', inputs)
        outputs = vgg_model(inputs)
        print(inputs)
        # inputs = autograd.Variable(inputs.cuda(), volatile=True)
        # output = vgg_model(autograd.Variable(torch.unsqueeze(inputs, dim=0), volatile=True))
        # output = vgg_model(inputs)
        for local_index, output_vector in enumerate(outputs.chunk(BATCH_SIZE, dim = 0)):
            filename = '{}_{}.dat'.format(batch_index, local_index)
            pathname = os.path.join(new_folder, filename)
            with open(os.path.join(new_folder, filename), 'wb') as fin:
                pickle.dump(output_vector.data.numpy(), fin)
            print('written {}'.format(pathname))

def depreciated(folder):
    vgg_folder = folder + "_vgg"
    mistakes = 0
    batch = []
    filenames = []
    for index, image_filename in enumerate(os.listdir(folder)):
        try:
            image = io.imread(os.path.join(folder, image_filename))
            transformer = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Scale(256),
                transforms.CenterCrop(224), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = torch.unsqueeze(transformer(image), dim=0)
            batch.append(img_tensor)
            filenames.append(image_filename)
            if len(batch) == BATCH_SIZE:
                input_batch = torch.cat(batch) 
                input_filenames = filenames
                batch = []
                filenames = []
            else:
                continue
            outputs = vgg_model(autograd.Variable(input_batch, volatile=True))
            for local_index, output_vector in enumerate(outputs.chunk(BATCH_SIZE, dim = 0)):
                filename = '{}.pkl'.format(input_filenames[local_index])
                pathname = os.path.join(vgg_folder, filename)
                with open(pathname, 'wb') as fin:
                    pickle.dump(output_vector.data.numpy(), fin)
                print('written {}'.format(pathname))
        except Exception as ex:
            print("Screwup @ {}  -------------------- ".format(image_filename))
            print(ex)
            mistakes += 1
            if False and mistakes == 3: break
        print('mistakes = {}', mistakes)

if __name__ == '__main__':
    print("Please set folder correctly")
    folder = "../Data/train2014"
    # data_loader = take_input(data_folder = folder)
    # get_vgg(folder = folder, data_loader = data_loader)
    depreciated(folder = folder)
    # get_vgg("/scratch/cse/btech/cs1140485/DL_Course_Data/train2014")
