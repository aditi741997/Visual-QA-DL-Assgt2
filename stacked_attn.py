import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

img_input_sz = 512*14*14

class Stacked_Attention_VQA(nn.Module):
	def __init__(self, ques_hidden_size, cell_type, no_layers):
		super(Stacked_Attention_VQA, self).__init__()
		self.cell_type = cell_type

		self.img_linear = nn.Linear(img_input_sz, )

		if self.cell_type == "gru":
			self.question_rec_layer = nn.GRU(300, ques_hidden_size, num_layers=no_layers, batch_first=True)
		else:
			self.question_rec_layer = nn.LSTM(300, ques_hidden_size, num_layers=no_layers, batch_first=True)
