import torch
import json
import pickle
from torch.utils.data import DataLoader
from praveen_dataset import VQA_Dataset
from main import process_data

data_path = "/scratch/cse/btech/cs1140485/DL_Course_Data"
model_path = "/home/cse/btech/cs1140485/DeepLearning/Assignment2/Visual-QA-DL-Assgt2/experiments/model_1_gru_1000.pth"
loc = "test2015"
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
  print "Yo"
  model = torch.load(model_path)
  dataset = VQA_Dataset(data_path, loc, batch_size)
  dataloader = DataLoader(dataset, num_workers=16)
  results_txt = "["
  out_file = open("/home/cse/btech/cs1140205/DL_Assgt2/new/" + "results_2.json", 'w')
  for i, data in enumerate(dataloader):
    images, questions, q_ids = process_data(data)
    outputs = model(images, questions)
    _, predicts = torch.max(outputs, 1)
    for i in xrange(q_ids.size(0)):
      result_obj = {}
      result_obj["question_id"] = q_ids.data[i]
      result_obj["answer"] = answer_map[predicts.data[i]]
      results_txt += (json.dumps(result_obj)) + ","
    print "Batch ", i
  print "Done"
  results_txt += "]"
  out_file.write(results_txt)


if __name__ == '__main__':
  main()
