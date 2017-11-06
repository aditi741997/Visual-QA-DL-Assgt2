import json
import pickle

def get_top1000(paths):
	answers_count = {}
	for file_path in paths:
		with open(file_path, 'r') as f:
			f_json = json.loads(f.read())
			for ann in f_json["annotations"]:
				for answer in ann["answers"]:
					if answer["answer"] not in answers_count:
						answers_count[answer["answer"]] = 0
					answers_count[answer["answer"]] += 1
	# sort answers based on key
	out = open("top1000_answers.txt", 'w')
	str_to_int  = {}
	i = 0
	for key, value in sorted(answers_count.iteritems(), key=lambda (k,v): (-1*v,k))[:999]:
		print "Key = " + key + ", Val = ", value
		i += 1
		str_to_int[key] = i
	pickle.dump(str_to_int, out)

get_top1000(["../Data/v2_mscoco_val2014_annotations.json", "../Data/v2_mscoco_train2014_annotations.json"])