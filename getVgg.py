# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sys, nltk
import os, argparse
import time, random
import torchvision.models as models
import pickle
from scipy import misc
from skimage import io
from PIL import Image
from pprint import pprint

vgg_model = models.vgg19_bn(pretrained=True)
# pprint(vgg_model.classifier)
vgg_model.classifier = nn.Sequential(*list(vgg_model.classifier.children())[:4])
# pprint(vgg_model.classifier)



def get_vgg(folder):
    vgg_folder = folder + "_vgg"
    mistakes = 0
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
            img_tensor = transformer(image)
            assert tuple(img_tensor.size()) == (3, 224, 224)
            img_vgg = vgg_model(autograd.Variable(torch.unsqueeze(img_tensor, dim=0), volatile=True))
            postvgg_filename = "{}.pkl".format(image_filename)
            postvgg_pathname = os.path.join(vgg_folder, postvgg_filename)
            pickle.dump(img_vgg.data.numpy(), open(postvgg_pathname, 'wb'))
            print('wrote the {}th image to the file {}'.format(index, postvgg_pathname))
        except Exception as ex:
            print("Screwup here -------------------- ")
            print(ex)
            if mistakes < 3:
                mistakes += 1
            else:
                break

if __name__ == '__main__':
    print("Please set file correctly")
    get_vgg("../Data/train2014")
    # get_vgg("/scratch/cse/btech/cs1140485/DL_Course_Data/train2014")
