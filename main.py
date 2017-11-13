import os
import time
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler

from dataset import VQA_Dataset
from dataset import VQA_Dataset_Test
from vqabase import VQA_Baseline

# Global variables
path = "/scratch/cse/btech/cs1140485/DL_Course_Data/"
# path = "/Users/Shreyan/Desktop/Datasets/DL_Course_Data/"
GPU = torch.cuda.is_available()

def train(model, args, data_loader, val_loader, test_loader):
  # Loss fn and optimizer
  loss = nn.CrossEntropyLoss(ignore_index=0)
  optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
  scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.weight_decay)

  if GPU:
    model = model.cuda()

  for epoch in xrange(args.num_epoch):
    t1 = time.time()
    scheduler.step()
    torch.save(model, args.model_save_path)
    train_right, train_total = 0, 0
    for i, data in enumerate(data_loader):
      # this is a batch of image-question pairs.
      images, questions, answers = data
      images = Variable(torch.squeeze(images, dim=0))
      questions = Variable(torch.squeeze(questions, dim=0))
      answers = Variable(torch.squeeze(answers, dim=0))
      
      if GPU:
        images = images.cuda()
        questions = questions.cuda()
        answers = answers.cuda()
      
      outputs = model(images, questions)
      _, predicts = torch.max(outputs, 1)
      train_total += predicts.size()[0]
      train_right += (predicts == answers).sum()
      
      optimizer.zero_grad()
      batch_loss = loss(outputs, answers)
      batch_loss.backward()
      optimizer.step()
    t2 = time.time()
    print("Epoch {} train_right {} train_total {} Time {}".format(epoch, train_right, train_total, t2-t1))


def get_arguments():
  # ques params
  parser = argparse.ArgumentParser(description='VQA_Base')
  parser.add_argument("--activation-fn", type=str, default="relu")
  parser.add_argument("--question-hidden-dim", type=int, default=512)

  parser.add_argument("--learning-rate", type=float, default=0.01)
  parser.add_argument("--momentum", type=float, default=0.9)
  parser.add_argument("--weight-decay", type=float, default=0.99)

  parser.add_argument("--num-epoch", type=int, default=2)
  parser.add_argument("--batch-size", type=int, default=150)

  parser.add_argument("--model-save-path", type=str, default="model.pth")
  args = parser.parse_args()

  args.model_save_path = os.path.join("experiments", args.model_save_path)
  return args

def main(args):
  model = VQA_Baseline(args.question_hidden_dim, args.activation_fn)
  train_data_loader = VQA_Dataset(path, "train2014", args.batch_size)
  # val_data_loader = VQA_Dataset(path, "val2014", args.batch_size)
  # test_data_loader = VQA_Dataset(path, "test2015", args.batch_size)
  val_data_loader = train_data_loader
  test_data_loader = train_data_loader
  train(model, args, train_data_loader, val_data_loader, test_data_loader)

if __name__ == '__main__':
  args = get_arguments()
  main(args)
