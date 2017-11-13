import os
import time
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

from dataset import VQA_Dataset
from dataset import VQA_Dataset_Test
from vqa_base import VQA_Baseline

# Global variables
path = "/scratch/cse/btech/cs1140485/DL_Course_Data/"
# path = "/Users/Shreyan/Desktop/Datasets/DL_Course_Data/"
GPU = torch.cuda.is_available()
validation_batch_limit = 20

def process_data(data):
  images, questions, answers = data
  images = Variable(torch.squeeze(images, dim=0))
  questions = Variable(torch.squeeze(questions, dim=0))
  answers = Variable(torch.squeeze(answers, dim=0))
  if GPU:
    images = images.cuda()
    questions = questions.cuda()
    answers = answers.cuda()
  return images, questions, answers

def get_accuracy(model, dataloader):
  t1 = time.time()
  right, total, unknown = 0, 0, 0
  for i, data in enumerate(dataloader):
    images, questions, answers = process_data(data)
    outputs = model(images, questions)
    _, predicts = torch.max(outputs, 1)
    total += predicts.size(0)
    right += (predicts == answers).sum().data[0]
    unknown += len(answers[answers == -1])
    if i == validation_batch_limit:
      break
  print("Validation Time {}".format(time.time()-t1))
  return right, total, unknown

def train(model, args, train_dataset, test_dataset):
  # Dataloaders
  train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=args.num_workers)
  test_dataloader = DataLoader(test_dataset, num_workers=args.num_workers)

  # Loss fn, optimizer, scheduler
  loss = nn.CrossEntropyLoss(ignore_index=-1)
  optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
  scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.weight_decay)

  if GPU:
    model = model.cuda()

  for epoch in xrange(args.num_epoch):
    test_right, test_total, test_unknown = get_accuracy(model, test_dataloader)
    t1 = time.time()
    scheduler.step()
    torch.save(model, args.model_save_path)
    train_right, train_total, train_unknown = 0, 0, 0
    for i, data in enumerate(train_dataloader):
      # this is a batch of image-question pairs.
      images, questions, answers = process_data(data)
      outputs = model(images, questions)

      _, predicts = torch.max(outputs, 1)
      train_total += predicts.size(0)
      train_right += (predicts == answers).sum().data[0]
      train_unknown += len(answers[answers == -1])
      
      optimizer.zero_grad()
      batch_loss = loss(outputs, answers)
      batch_loss.backward()
      optimizer.step()
      if i == 10:
        break
    t2 = time.time()
    log = "Epoch {}, train_acc {}, test_acc {}, reduced_train_acc {}, reduced_test_acc {}, time {}".format(
      epoch, 
      100.0*train_right/train_total, 
      100.0*test_right/test_total, 
      100.0*(train_right-train_unknown)/(train_total-train_unknown), 
      100.0*(test_right-test_unknown)/(test_total-test_unknown), 
      t2-t1)
    args.log.write(log+"\n")
    args.log.flush()
    print(log)
    print("Epoch {} train {} {} {} test {} {} {}".format(epoch, train_right, train_total, train_unknown, test_right, test_total, test_unknown))


def get_arguments():
  # ques params
  parser = argparse.ArgumentParser(description='VQA_Base')
  parser.add_argument("--activation-fn", type=str, default="relu")
  parser.add_argument("--question-hidden-dim", type=int, default=512)

  parser.add_argument("--learning-rate", type=float, default=0.01)
  parser.add_argument("--momentum", type=float, default=0.9)
  parser.add_argument("--weight-decay", type=float, default=0.98)

  parser.add_argument("--num-epoch", type=int, default=200)
  parser.add_argument("--batch-size", type=int, default=150)
  parser.add_argument("--num-workers", type=int, default=4)

  parser.add_argument("--model-save-path", type=str, default="model.pth")
  parser.add_argument("--log", type=str, default="log.txt")
  args = parser.parse_args()

  args.model_save_path = os.path.join("experiments", args.model_save_path)
  args.log = open(os.path.join("experiments", args.log), "w")
  return args

def main(args):
  model = VQA_Baseline(args.question_hidden_dim, args.activation_fn)
  train_dataset = VQA_Dataset_Test(path, "train2014", args.batch_size)
  val_dataset = VQA_Dataset_Test(path, "val2014", args.batch_size)
  test_dataset = VQA_Dataset_Test(path, "test2015", args.batch_size)
  merged_dataset = ConcatDataset([train_dataset, val_dataset])
  train(model, args, merged_dataset, test_dataset)

if __name__ == '__main__':
  args = get_arguments()
  main(args)
