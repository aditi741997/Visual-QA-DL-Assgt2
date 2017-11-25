import os
import time
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

from praveen_dataset import VQA_Dataset
from praveen_stacked import Stacked_Attention_VQA
from vqa_base import VQA_Baseline

# Global variables
path = "/scratch/cse/btech/cs1140485/DL_Course_Data/"
GPU = torch.cuda.is_available()
validation_batch_limit = 1000

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
    try:
      assert(predicts.size(0) == answers.size(0))
      total += predicts.size(0)
      right += (predicts == answers).sum().data[0]
    except Exception as ex:
      print ex
    unknown += len(answers[answers == -1])
    if i == validation_batch_limit:
      break
  print("Validation Time {}".format(time.time()-t1))
  return right, total, unknown, time.time()-t1

def train(model, args, train_dataset, test_dataset):
  log = "lr {}, mtm {}, wt decay {}, gamma {}, activation {}, save_path {}".format(args.learning_rate, args.momentum, args.weight_decay, args.gamma, args.activation_fn, args.model_save_path)
  if args.model_load_path:
    log += "\nLoad model from"+args.model_load_path
  print log
  args.log.write(log+"\n")
  # Dataloaders
  train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=args.num_workers)
  test_dataloader = DataLoader(test_dataset, shuffle=True, num_workers=args.num_workers)

  # Loss fn, optimizer, scheduler
  loss = nn.CrossEntropyLoss(ignore_index=-1)
  optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
  scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

  if GPU:
    model = model.cuda()

  for epoch in xrange(args.num_epoch):
    t1 = time.time()
    scheduler.step()
    torch.save(model, args.model_save_path)
    train_right, train_total, train_unknown = 0, 0, 0
    total_loss = 0
    print('just before')
    for i, data in enumerate(train_dataloader):
      # this is a batch of image-question pairs.
      bt1 = time.time()
      images, questions, answers = process_data(data)
      # print('just inside')
      outputs = model(images, questions)
      # print('just outside', outputs)
      _, predicts = torch.max(outputs, 1)
      train_total += predicts.size(0)
      train_right += (predicts == answers).sum().data[0]
      train_unknown += len(answers[answers == -1])
      
      optimizer.zero_grad()
      batch_loss = loss(outputs, answers)
      total_loss += batch_loss.data[0]
      batch_loss.backward()
      optimizer.step()
      print "Batch done", time.time() - bt1
    test_right, test_total, test_unknown, test_time = get_accuracy(model, test_dataloader)
    t2 = time.time()
    log = "Epoch {}, loss {}, train_acc {}, test_acc {}, reduced_train_acc {}, reduced_test_acc {}, time {}, test_time {}".format(
      epoch, 
      total_loss,
      100.0*train_right/train_total, 
      100.0*test_right/test_total, 
      100.0*(train_right)/(train_total-train_unknown), 
      100.0*(test_right)/(test_total-test_unknown), 
      t2-t1,
      test_time)
    args.log.write(log+"\n")
    args.log.flush()
    print(log)
    print("Epoch {} train {} {} {} test {} {} {}".format(epoch, train_right, train_total, train_unknown, test_right, test_total, test_unknown))


def get_arguments():
  # ques params
  parser = argparse.ArgumentParser(description='VQA_Base')
  parser.add_argument("--activation-fn", type=str, default="tanh")
  parser.add_argument("--question-hidden-dim", type=int, default=512)
  parser.add_argument("--num-attention-layers", type=int, default=1)
  parser.add_argument("--ans-vocab-size", type=int, default=1000)

  parser.add_argument("--learning-rate", type=float, default=0.025)
  parser.add_argument("--momentum", type=float, default=0.9)
  parser.add_argument("--weight-decay", type=float, default=0)
  parser.add_argument("--gamma", type=float, default=0.88)
  parser.add_argument("--cell-type", type=str, default="lstm")

  parser.add_argument("--num-epoch", type=int, default=20)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--num-workers", type=int, default=32)

  parser.add_argument("--model-save-path", type=str, default="model")
  parser.add_argument("--model-load-path", type=str)
  parser.add_argument("--log", type=str, default="log")
  args = parser.parse_args()

  append_string = "_" + str(args.num_attention_layers) + "_" + args.cell_type + "_" + str(args.ans_vocab_size)
  args.model_save_path = os.path.join("experiments", args.model_save_path + append_string + ".pth")
  args.log = open(os.path.join("experiments", args.log + append_string + ".txt"), "w")
  return args

def main(args):
  if args.model_load_path:
    model = torch.load(args.model_load_path)
  else:
    print('right path')
    model = Stacked_Attention_VQA(args.cell_type, output_size=args.ans_vocab_size, num_attention_layers=args.num_attention_layers) #defaults
  train_dataset = VQA_Dataset(path, "train2014", args.batch_size, args.ans_vocab_size)
  val_dataset = VQA_Dataset(path, "val2014", args.batch_size, args.ans_vocab_size)
  print('data sets loaded')
  train(model, args, train_dataset, train_dataset)

if __name__ == '__main__':
  args = get_arguments()
  main(args)
