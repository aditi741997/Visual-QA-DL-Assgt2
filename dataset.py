import os
import re 
import json
import torch
import pickle
import numpy as np
from scipy import misc
from torch.utils.data import Dataset

class VQA_Dataset(Dataset):
  """Dataset from VQA"""
  def __init__(self, path, loc, batch_size):
    """Store question list and top 999 answers"""
    self.loc = loc
    self.image_path = os.path.join(path, loc)
    self.vocab_question = pickle.load(open("glove_vocab.pkl", "r"))
    # self.vocab_question = dict()
    self.qa_map = dict()
    vocab_answer = pickle.load(open("top1000_answers.pkl", "r"))

    with open(os.path.join(path, "v2_OpenEnded_mscoco_{}_questions.json".format(loc)), "r") as f:
      q_json = json.loads(f.read())
      q_list = q_json["questions"]
      len_wise_list = [[] for _ in xrange(2, 23)]
      for x in q_list:
        x["question"] = re.sub("[,.?]", "", x["question"]).split()
        len_wise_list[len(x["question"]) - 2].append(x)
      self.batches = []
      for len_list in len_wise_list:
        for i in xrange(0, len(len_list), batch_size):
          self.batches.append(len_list[i:i+batch_size])
    
    with open(os.path.join(path, "v2_mscoco_{}_annotations.json".format(loc)), "r") as f:
      a_json = json.loads(f.read())
      for ans in a_json["annotations"]:
        q_id = ans["question_id"]
        ans = ans["multiple_choice_answer"]
        self.qa_map[q_id] = vocab_answer[ans] if ans in vocab_answer else 0
    print "init dataset"

  def __len__(self):
    return len(self.batches) 

  def __getitem__(self, i):
    """Return (image, question, answer_id)"""
    batch = self.batches[i]
    image = []
    question = []
    answer = []
    for q in batch:
      image_path = os.path.join(self.image_path, "COCO_{}_{:012}.jpg".format(self.loc, q["image_id"]))
      # TODO : Change this once we get the embedding for the images
      # image.append(pickle.load(image_path))
      image.append(np.zeros((4096)))
      map_fn = lambda x: self.vocab_question[x].type(torch.FloatTensor).numpy() if x in self.vocab_question else np.zeros((300))
      question.append(map(map_fn, q["question"]))
      answer.append(self.qa_map[q["question_id"]])

    print "Batch Len : ", len(question)
    return (torch.FloatTensor(image), torch.Tensor(np.array(question)), torch.LongTensor(answer))

class VQA_Dataset_Test(Dataset):
  def __init__(self, path, loc, batch_size):
    self.batch_size = batch_size

  def __len__(self):
    return 100

  def __getitem__(self, i):
    image = [np.zeros((4096)) for _ in range(self.batch_size)]
    question = [[np.zeros((300)) for _ in range(20)] for _ in range(self.batch_size)]
    answer = [0 for _ in range(self.batch_size)]
    return (torch.FloatTensor(image), torch.Tensor(np.array(question)), torch.LongTensor(answer))


# For testing
from torch.utils.data import DataLoader
from collections import defaultdict
if __name__ == '__main__':
  # path = "../Data"
  path = "/Users/Shreyan/Desktop/Datasets/DL_Course_Data"
  loc = "train2014"

  dataset2 = VQA_Dataset_Test(path, loc, 100)
  i2,q2,a2 = dataset2[0]
  print i2.size(), q2.size(), a2.size()
  dataset = VQA_Dataset(path, loc, 100)
  i,q,a = dataset[0]
  print i.size(), q.size(), a.size()

  train_loader = DataLoader(dataset, num_workers=1, shuffle=True)
  for i, data in enumerate(train_loader):
    # image, question, answer = data[0], data[1], data[2]
    print "DATA -----------> "
    print data
    break