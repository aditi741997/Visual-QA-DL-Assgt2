import os
import re 
import json
import pickle
import numpy as np
from scipy import misc
from torch.utils.data import Dataset

class VQA_Dataset(Dataset):
  """Dataset from VQA"""
  def __init__(self, path, loc):
    """Store question list and top 999 answers"""
    self.loc = loc
    self.image_path = os.path.join(path, loc)
    self.vocab_question = pickle.load(open("glove_vocab.pkl", "r"))
    # self.vocab_question = dict() 
    self.qa_map = dict()
    self.vocab_answer = pickle.load(open("top1000_answers.pkl", "r"))

    with open(os.path.join(path, "v2_OpenEnded_mscoco_{}_questions.json".format(loc)), "r") as f:
      q_json = json.loads(f.read())
      self.q_list = q_json["questions"]
    with open(os.path.join(path, "v2_mscoco_{}_annotations.json".format(loc)), "r") as f:
      a_json = json.loads(f.read())
      for ans in a_json["annotations"]:
        q_id = ans["question_id"]
        ans = ans["multiple_choice_answer"]
        self.qa_map[q_id] = self.vocab_answer[ans] if ans in self.vocab_answer else 0
    # print "done init"

  def __len__(self):
    return len(self.q_list)

  def get_map_fn(self, mapping, default):
    def map_fn(x):
      return mapping[x] if x in mapping else default
    return map_fn

  def __getitem__(self, i):
    """Return (image, question, answer_id)"""
    q = self.q_list[i]
    question = re.sub("[,.?]", "", q["question"]).split()
    question = map(self.get_map_fn(self.vocab_question, np.zeros((300))), question)
    question = np.array(question)
    answer = self.qa_map[q["question_id"]]
    image_path = os.path.join(self.image_path, "COCO_{}_{:012}.jpg".format(self.loc, q["image_id"]))
    image_embedding = pickle.load(image_path)
    image_embedding = np.array(image_embedding)
    
    # image = np.array(misc.imread(image_path), dtype=np.float32)
    # image_embedding = np.zeros((100,100))
    return (image_embedding, question, answer)

def collate_fn(batch):
  # Todo, aditi, batch it up


# For testing
from torch.utils.data import DataLoader
if __name__ == '__main__':
  path = "/Users/Shreyan/Desktop/Datasets/DL_Course_Data"
  dataset = VQA_Dataset(path, "train2014")
  train_loader = DataLoader(dataset, batch_size=100, num_workers=1)
  for i, data in enumerate(train_loader):
    image, question, answer = data
    print answer




