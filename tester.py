import os
import torch
import json
import pickle
from torch.utils.data import DataLoader
from praveen_dataset import VQA_Dataset
from main import process_data

data_path = "/scratch/cse/btech/cs1140485/DL_Course_Data"
loc = "test2015"
model = "model_1_gru_1000"
model_path = "/home/cse/btech/cs1140485/DeepLearning/Assignment2/Visual-QA-DL-Assgt2/experiments/{}.pth".format(model)
result_path = "/home/cse/btech/cs1140485/DeepLearning/Assignment2/san_result_{}_{}.json".format(loc, model)
batch_size = 64
GPU = torch.cuda.is_available()

def get_answer_map():
  vocab_answer = pickle.load(open("top1000_answers.pkl", "r"))
  answer_map = dict()
  for a in vocab_answer:
    answer_map[vocab_answer[a]] = a
  return answer_map

def main():
  answer_map = get_answer_map()
  model = torch.load(model_path)
  dataset = VQA_Dataset(data_path, loc, batch_size, 1000, True)
  dataloader = DataLoader(dataset, num_workers=16)
  results_txt = "["
  done_ques_set = set()
  out_file = open(result_path, 'w')
  for i, data in enumerate(dataloader):
    images, questions, q_ids = process_data(data)
    outputs = model(images, questions)
    _, predicts = torch.max(outputs, 1)
    for i in xrange(q_ids.size(0)):
      result_obj = {}
      result_obj["question_id"] = q_ids.data[i]
      result_obj["answer"] = answer_map[predicts.data[i]]
      done_ques_set.add(q_ids.data[i])
      results_txt += (json.dumps(result_obj)) + ","
    print "Batch ", i
  print "Done"
  with open(os.path.join(data_path, "v2_OpenEnded_mscoco_{}_questions.json".format(loc)), "r") as f:
    q_json = json.loads(f.read())
    q_list = q_json["questions"]
    for x in q_list:
      if x["question_id"] in done_q_set:
        continue
      result_obj = {}
      result_obj["question_id"] = x["question_id"]
      result_obj["answer"] = answer_map[1]
      results_txt += (json.dumps(result_obj)) + ","
  results_txt = results_txt[:-1] + "]"
  out_file.write(results_txt)


if __name__ == '__main__':
  main()
