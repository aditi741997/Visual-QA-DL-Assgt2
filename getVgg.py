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
# pprint(list(vgg_model.children()))
vgg_model.classifier = nn.Sequential(*list(vgg_model.classifier.children())[:4])
pprint(vgg_model)

BATCH_SIZE = 32
STATUS_FILE = "/home/cse/btech/cs1140485/DeepLearning/Assignment2/Visual-QA-DL-Assgt2/status_sg.txt"

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
                pickle.dump(output_vector.data.numpy(), fin, protocol = 2)
            print('written {}'.format(pathname))

def depreciated(folder):
    global vgg_model
    if torch.cuda.is_available():
	vgg_model = vgg_model.cuda()
    vgg_folder = folder + "_vgg"
    transformer = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale(256),
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mistakes = 0
    batch = []
    filenames = []
    print("begin loop")
    T = time.time()
    for index, image_filename in enumerate(os.listdir(folder)):
        try:
            image = io.imread(os.path.join(folder, image_filename))
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
            if torch.cuda.is_available():
                input_batch = input_batch.cuda()
            outputs = vgg_model(autograd.Variable(input_batch, volatile=True))
            for local_index, output_vector in enumerate(outputs.chunk(BATCH_SIZE, dim = 0)):
                filename = '{}.pkl'.format(input_filenames[local_index])
                pathname = os.path.join(vgg_folder, filename)
                with open(pathname, 'wb') as fin:
                    pickle.dump(output_vector.data.cpu().numpy(), fin, protocol=2)
		with open(STATUS_FILE, 'a') as fout:
		    fout.write('written {}\n'.format(pathname))
		    print('written {}'.format(pathname))
            print("time"+str(time.time()-T))
	    T = time.time()
	except Exception as ex:
	    with open(STATUS_FILE, 'a') as fout:    
		fout.write("Screwup @ {}  : {}\n".format(image_filename, ex))
		print("Screwup @ {}  : {}\n".format(image_filename, ex))
            mistakes += 1
            if False and mistakes == 3: return
    print('mistakes = {}', mistakes)

if __name__ == '__main__':
    if torch.cuda.is_available():
        vgg_model = vgg_model.cuda()
    # data_loader = take_input(data_folder = folder)
    # get_vgg(folder = folder, data_loader = data_loader)
    # get_vgg("/scratch/cse/btech/cs1140485/DL_Course_Data/train2014")
    folder1 = "/scratch/cse/btech/cs1140485/DL_Course_Data/train2014"
    folder2 = "/scratch/cse/btech/cs1140485/DL_Course_Data/test2015"
    folder3 = "/scratch/cse/btech/cs1140485/DL_Course_Data/val2014"
    depreciated(folder=folder1)
    depreciated(folder=folder2)
    depreciated(folder=folder3)
