import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys, nltk
import os, argparse
import time, random
import torchvision.models as models
import dataset

word_embedding_dim = 300
hidden_ques_dim = 512
img_ques_dim = 1024
linear_dim = 1000
GPU = False

class VQABaseline(nn.Module):
	def __init__(self, hidden):
		super(VQABaseline, self).__init__()

		if GPU:
			print "Using GPU" 
			self.dtype = torch.cuda.FloatTensor
		else:
			self.dtype = torch.FloatTensor

		self.activn = "relu"
		self.img_linear = nn.Linear(4096, img_ques_dim)
		self.ques_lstm_1 = nn.LSTM(word_embedding_dim, hidden, num_layers=1, bidirectional=False, batch_first=True)
		self.ques_lstm_2 = nn.LSTM(hidden, hidden, num_layers=1, bidirectional=False, batch_first=True)
		self.ques_linear = nn.Linear(4096, img_ques_dim)

		self.final_linear_1 = nn.Linear(img_ques_dim, linear_dim)
		self.final_linear_2 = nn.Linear(linear_dim, linear_dim)

	def actvn_func(self, x):
		if self.activn == "relu":
			return F.relu(x)
		else:
			return F.tanh(x)

	def forward(self, questions, images):
		# images : No * 4096
		# questions : No * QLen * 300
		images_final = self.actvn_func(self.img_linear(images))

		batch_sz = questions.data.size()[0]
		h0 = autograd.Variable(torch.randn(1, batch_sz, hidden))
		c0 = autograd.Variable(torch.randn(1, batch_sz, hidden))
		out_ques_1, (hidden_ques_1, c_ques_1) = self.ques_lstm_1(questions, (h0, c0))
		out_ques_2, (hidden_ques_2, c_ques_2) = self.ques_lstm_2(out_ques_1, (h0, c0))
		ques_linear_input = torch.cat([out_ques_1[-1], out_ques_2[-1], torch.squeeze(hidden_ques_1, dim=0), torch.squeeze(hidden_ques_2, dim=0)], dim=1)
		questions_final = self.actvn_func(self.ques_linear(ques_linear_input))

		img_ques_dot = images_final * questions_final
		img_ques_dot = F.dropout(self.actvn_func(self.final_linear_1(img_ques_dot)))
		img_ques_dot = self.final_linear_2(img_ques_dot)

		return img_ques_dot

def train(net, no_epoch, data_loader, val_loader, test_loader):
	loss = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters())
	for epoch in xrange(no_epoch):
		t1 = time.time()
		train_right = 0
		train_total = 0
		for i, data in enumerate(data_loader, 0):
			# this is a batch of image-question pairs.
			optimizer.zero_grad()
			images, questions, answers = data
			questions = Variable(torch.squeeze(questions, dim=0))
			answers = torch.squeeze(answers, dim=0)
			images = Variable(torch.squeeze(images, dim=0))
			outputs = net(questions, images)
			predicts = torch.max(outputs, 1)[1]
			train_total += predicts.size()[0]
			train_right += (predicts == answers).sum()
			batch_loss = loss(outputs, answers)
			batch_loss.backward()
			optimizer.step()
			if i%100 == 1:
				print "Batch #" + str(i) + " Done!"
				print("[%d, %5d] current train accuracy : %.3f" %(epoch+1, i+1, train_right/float(train_total)))
				break
		t2 = time.time()
		print "Epoch " + str(epoch) + " Done!"
		


def get_arguments():
	parser = argparse.ArgumentParser(description='VQA_Base')
	# ques params
	
	# training
	parser.add_argument("--n_epochs", type=int, default=2)
	parser.add_argument("--batch_size", type=int, default=150)

	opts = parser.parse_args(sys.argv[1:])
	return opts

args = get_arguments()
path = "/scratch/cse/btech/cs1140485/DL_Course_Data/"
vqa = VQABaseline(hidden_ques_dim)
train_data_loader = dataset.VQA_Dataset(path, "train2014", args.batch_size)
val_data_loader = dataset.VQA_Dataset(path, "val2014", args.batch_size)
test_data_loader = dataset.VQA_Dataset(path, "test2015", args.batch_size)

train(vqa, args.n_epochs, train_data_loader, val_data_loader, test_data_loader)
