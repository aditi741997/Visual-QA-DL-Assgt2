import json

def parse_ques(file_path):
	ques_list = []
	img_wise_ques = {}
	with open(file_path, 'r') as f:
		f_json = json.loads(f.read())
		print "Task Type :", f_json["task_type"]
		print "Info version :", f_json["info"]["version"]
		ques_list = f_json["questions"]
		for q in f_json["questions"]:
			if q["image_id"] not in img_wise_ques:
				img_wise_ques[q["image_id"]] = []
				print "img id : ", q["image_id"]
			img_wise_ques[q["image_id"]].append(q["question_id"])
	print "Total ques :", len(ques_list)
	for i in xrange(3):
		k = int(input("Enter image id :"))
		if k in img_wise_ques:
			print "No of ques : ", img_wise_ques[k]
		else:
			print "IMg id doesn't exist"

def parse_annotation(file_path):
	annot_list = []
	ques_wise_ans = {}
	with open(file_path, 'r') as f:
		f_json = json.loads(f.read())
		print "Data Sub Type :", f_json["data_subtype"]
		print "Info version :", f_json["info"]["version"]
		annot_list = f_json["annotations"]
		for ann in f_json["annotations"]:
			if ann["question_id"] not in ques_wise_ans:
				ques_wise_ans[ann["question_id"]] = []
			for answer in ann["answers"]:
				ques_wise_ans[ann["question_id"]].append(answer)
	print "Total annotations :", len(annot_list)
	for i in xrange(5):
		k = int(input("Enter ques id :"))
		if k in ques_wise_ans:
			print "#ans ", ques_wise_ans[k]
		else:
			print "Invalid"

q_path = "/Users/Shreyan/Desktop/Datasets/DL_Course_Data/v2_OpenEnded_mscoco_val2014_questions.json"
a_path = "/Users/Shreyan/Desktop/Datasets/DL_Course_Data/v2_mscoco_val2014_annotations.json"

parse_ques(q_path)
parse_annotation(a_path)