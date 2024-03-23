import json
with open('/home/coder/projects/Med-VQA/data/testset.json') as f:
    test_data=json.load(f)
print(test_data)
for d in test_data:
    print(d['qid'],d['answer_type'])

qid_to_answer_type=list(filter(lambda x:x['qid']==38, test_data))
print(qid_to_answer_type[0]['answer_type'])
print(len(test_data))