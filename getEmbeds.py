import numpy as np
import os
import pickle, re
import torch, json

root = ""
folder_path = "../Data"
word_embed_dim = 200

def get_vocab(paths):
	vocab = {}
	for x in paths:
		with open(os.path.join(root + folder_path, x), 'r') as f:
			f_json = json.loads(f.read()) 
			sents = map(lambda x: (re.sub("[,.?]", "", x["question"])).split(' '), f_json["questions"])
			# print "no of sents : ", len(sents)
			for sent in sents:
				for w in sent:
					if w not in vocab:
						print "new : ", w
					vocab[w] = np.random.rand(word_embed_dim)
				# print "---", len(vocab)
		print "Vocab size : ", len(vocab)
	return vocab

def get_reduced_vecs(vocab):
	f1 = "glove_subset.txt"
	write_file = open(f1, 'w')
	found = 0
	with open(os.path.join("../../../../../Windows8_OS/Users/jagdish/Desktop", "glove.840B.300d.txt"), 'r') as f:
		for l in f.readlines():
			w_v = l.split(' ')
			if w_v[0] in vocab:
				found += 1
				write_file.write(l + "\n")
	print "Total found : ", found

vocab = get_vocab(["v2_OpenEnded_mscoco_val2014_questions.json", "v2_OpenEnded_mscoco_train2014_questions.json", "v2_OpenEnded_mscoco_test2015_questions.json"])
get_reduced_vecs(vocab)

# convert subset file to pickle module.


def saveToPickle(file_name):
	with open(file_name, 'r') as f:
		word_vec = {}
		word_vec_arr = f.readlines()
		for l in word_vec_arr:
			w_v = l.replace('\n', '').split(' ')
			if len(w_v) == 301:
				word = w_v[0]
				w_v = map(lambda x: float(x), w_v[1:])
				# print len(w_v)
				vec = torch.from_numpy(np.array(w_v))
				word_vec[word] = vec
		pickle.dump(word_vec, open("glove_vocab.pkl", "w"))


saveToPickle("glove_subset.txt")
