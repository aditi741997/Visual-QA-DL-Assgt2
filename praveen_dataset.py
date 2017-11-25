import os
import re 
import json
import torch
import pickle
import numpy as np
from scipy import misc
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
from collections import defaultdict

class VQA_Dataset(Dataset):
  """Dataset from VQA"""
  def __init__(self, path, loc, batch_size, ans_size=1000):
    """Store question list and top 999 answers"""
    self.path = path
    self.loc = loc
    self.image_path = os.path.join(path, loc+"_vgg")
    self.vocab_question = pickle.load(open("glove_vocab.pkl", "r"))
    # self.vocab_question = dict()
    self.qa_map = dict()
    vocab_answer = pickle.load(open("top{}_answers.pkl".format(ans_size), "r"))

    with open(os.path.join(path, "v2_OpenEnded_mscoco_{}_questions.json".format(loc)), "r") as f:
      q_json = json.loads(f.read())
      q_list = q_json["questions"]
      len_wise_list = defaultdict(list)
      for x in q_list:
        x["question"] = re.sub("[,.?]", "", x["question"]).split()
        len_wise_list[len(x["question"])].append(x)
      self.batches = []
      for k in len_wise_list:
        for i in xrange(0, len(len_wise_list[k]), batch_size):
          self.batches.append(len_wise_list[k][i:i+batch_size])

    with open(os.path.join(path, "v2_mscoco_{}_annotations.json".format(loc)), "r") as f:
      a_json = json.loads(f.read())
      for ans in a_json["annotations"]:
        q_id = ans["question_id"]
        ans = ans["multiple_choice_answer"]
        self.qa_map[q_id] = vocab_answer[ans] if ans in vocab_answer else -1
    print("init dataset")
    
    self.transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale(256),
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("init transform")



  def __len__(self):
    return len(self.batches) 

  def __getitem__(self, i):
    """Return (image, question, answer_id)"""
    batch = self.batches[i]
    raw_image_dir = os.path.join(self.path, self.loc)
    image = []
    question = []
    answer = []
    for q in batch:
      image_path = os.path.join(raw_image_dir, "COCO_{}_{:012}.jpg".format(self.loc, q["image_id"]))
      if not os.path.isfile(image_path):
        print("File not found: "+ image_path)
        continue
      current_image = io.imread(image_path)
      try:
        image.append(torch.unsqueeze(self.transform(current_image), dim=0))
        map_fn = lambda x: self.vocab_question[x].type(torch.FloatTensor).numpy() if x in self.vocab_question else np.zeros((300))
        question.append(map(map_fn, q["question"]))
        answer.append(self.qa_map[q["question_id"]])
      except Exception as e:
        print("Error", e)
        continue
    image_batch = torch.cat(image)
    return (image_batch, torch.Tensor(np.array(question)), torch.LongTensor(answer))


# For testing
from torch.utils.data import DataLoader
from collections import defaultdict
if __name__ == '__main__':
    # path = "../Data"
    path = "/scratch/cse/btech/cs1140485/DL_Course_Data/"
    loc = "train2014"

    dataset = VQA_Dataset(path, loc, 100)
    i,q,a = dataset[0]
    print(i.size(), q.size(), a.size())
    print("test")

    train_loader = DataLoader(dataset, num_workers=1, shuffle=True)
    for i, data in enumerate(train_loader):
        # image, question, answer = data[0], data[1], data[2]
        print("DATA -----------> ")
        print(data)
        break
