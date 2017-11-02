import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as import torch.nn as nn
import numpy as np
import sys, nltk
import os, argparse
import time, random

word_embedding_dim = 200

class VQABaseline(nn.Module):
	def __init__(self, o, k, s, hid):
		super(VQABaseline, self).__init__()

		if GPU:
			print "Using GPU"
			self.dtype = torch.cuda.FloatTensor
		else:
			self.dtype = torch.FloatTensor

		self.image_conv1 = nn.Conv2d(3, o[0], k[0], s[0])
		self.image_conv2 = nn.Conv2d(o[0], o[1], k[1], s[1])

		self.ques_lstm1 = nn.LSTM(word_embedding_dim, hid[0], num_layers=1, bidirectional=False, batch_first=True)


def get_arguments():
	parser = argparse.ArgumentParser(description='VQA_Base')
	# network params
	parser.add_argument("-out1", type=int, default=96)
	parser.add_argument("-k1", type=int, default=3)
	parser.add_argument("-s1", type=int, default=4)
	
	# training
	parser.add_argument("--n_epochs", type=int, default=2)
	parser.add_argument("--batch_size", type=int, default=6)

	opts = parser.parse_args(sys.argv[1:])
	return opts
