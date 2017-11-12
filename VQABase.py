import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Global variables
img_ques_dim = 1024
linear_dim = 1000

class VQA_Baseline(nn.Module):
	"""
		VQA baseline model
		Takes 4096 dim embedding of image from vggnet.
		Takes 300 dim embedding of question from glove
		Returns logit over 1000 most frequent answers
	"""
	def __init__(self, hidden_size, activation_fn):
		"""Args
			hidden_size : size of hidden dim of LSTM
			activation_fn : one of "relu" or "tanh"
		"""
		super(VQA_Baseline, self).__init__()

		self.activation_fn = activation_fn
		self.hidden_size = hidden_size
		self.img_linear = nn.Linear(4096, img_ques_dim)
		self.ques_lstm_1 = nn.LSTM(300, hidden_size, num_layers=1, batch_first=True)
		self.ques_lstm_2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
		self.ques_linear = nn.Linear(2048, img_ques_dim)

		self.final_linear_1 = nn.Linear(img_ques_dim, linear_dim)
		self.final_linear_2 = nn.Linear(linear_dim, 1000)

		is_cuda = next(self.parameters()).is_cuda
		self.type = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor

	def actvn_func(self, x):
		"""Activation fn can be relu or tanh"""
		if self.activation_fn == "relu":
			return F.relu(x)
		else:
			return F.tanh(x)

	def init_h(self, batch_size):
		"""Initialize the hidden and cell state of LSTM"""
		return Variable(torch.randn(1, batch_size, self.hidden_size).type(self.type))

	def forward(self, images, questions):
		"""Forward function
			Args:
				images : Batch * 4096
				questions : Batch * QLen * 300
			Return:
				logits over 1000 most frequent asnwers
		"""
		images_final = self.actvn_func(self.img_linear(images))

		b = questions.data.size(0)
		h0, h1, c0, c1 = self.init_h(b), self.init_h(b), self.init_h(b), self.init_h(b)
		out_ques_0, (hidden_ques_0, c_ques_0) = self.ques_lstm_1(questions, (h0, c0))
		out_ques_1, (hidden_ques_1, c_ques_1) = self.ques_lstm_2(out_ques_0, (h1, c1))
		hidden_ques_0 = torch.squeeze(hidden_ques_0, dim=0)
		hidden_ques_1 = torch.squeeze(hidden_ques_1, dim=0)

		ques_linear_input = torch.cat([out_ques_0[:, -1, :], out_ques_1[:, -1, :], hidden_ques_0, hidden_ques_1], dim=1)
		questions_final = self.actvn_func(self.ques_linear(ques_linear_input))

		img_ques_dot = images_final * questions_final
		img_ques_dot = F.dropout(self.actvn_func(self.final_linear_1(img_ques_dot)))
		img_ques_dot = self.final_linear_2(img_ques_dot)

		return img_ques_dot
