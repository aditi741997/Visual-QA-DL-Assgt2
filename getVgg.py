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
from PIL import Image

vgg_model = models.vgg19_bn(pretrained=True)
# vgg_model.classifier = nn.Sequential(*(vgg_model.classifier[i] for i in xrange(4)))

def get_vgg(folder):
	new_folder = folder + "_vgg"
	i = 0
	for img in os.listdir(folder):
		im = Image.open(os.path.join(folder, img))
		print im.size
		# image = np.array(misc.imread(os.path.join(folder, img)), dtype=np.float32)
		# print image
		# image /= 256.0
		# # print image.shape
		# img_tensor = torch.from_numpy(image).permute(2, 0, 1)
		# img_tensor = img_tensor.contiguous()
		# print img_tensor.size()
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		transformer = transforms.Compose([transforms.Scale(256), transforms.CenterCrop(224), normalize, transforms.ToTensor()])
		# img_tensor = transformer(img_tensor)
		# img_pil = transforms.ToPILImage(img_tensor)
		img_tensor = transformer(im)
		print img_tensor.size()
		try:
			img_vgg =  vgg_model(autograd.Variable(torch.unsqueeze(img_tensor, dim=0), volatile=True))
			print img_vgg
			pickle.dump((img_vgg).data.numpy(), open(img, 'w'))
		except Exception as ex:
			print "DAMN -------------------- "
			print ex
		i += 1
		if i == 3:
			break

get_vgg("../Data/train2014")