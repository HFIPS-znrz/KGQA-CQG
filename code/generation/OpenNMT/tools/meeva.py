import json
import re
from collections import Counter
import string

# model_name_or_path = 'bert-base-chinese'
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

file_prediction = '../final_output_single.txt'
file_ground = '/home/dell/zjx/MSParS_V2.0_single_2_newme/data/single-turn/task2/tgt-test.txt'

key1_path = '../pred_entity1.txt'
key2_path = '../pred_entity2.txt'

predict_ques = []

with open(file_prediction, 'r') as file:
    for line in file:
        line = line.split('？')[0].split(' ')
        ques = ''.join(line).strip()
        predict_ques.append(ques)

gkey1 = []
gkey2 = []
ground_ques = []
with open(file_ground, 'r') as file:
    for line in file:
        line = line.strip()
        parts = line.split('<TSP>')
        
        gkey1.append(parts[1].strip())
        gkey2.append(parts[2].strip())
        ground_ques.append(parts[0].split(' ')[0].strip())

# print(gkey1)
ground_key = list(zip(gkey1, gkey2))
print(ground_key)

pkey1 = []
pkey2 = []
with open(key1_path, 'r') as file:
    for line in file:
        pkey1.append(line.strip())

with open(key2_path, 'r') as file:
    for line in file:
        pkey2.append(line.strip())

predict_key = list(zip(pkey1, pkey2))
print(predict_key)

'''关键词准确率'''
key_num = len(ground_key)*2
corr_key = 0
for i, pre_key in enumerate(predict_key):
    for k in pre_key:
        if k in ground_key[i]:
            corr_key += 1
        
accuracy_key = corr_key / key_num
print('关键词生成准确率：', accuracy_key)

'''模糊类型准确率'''
kind = ['种植方式','生长时期','土壤条件','季节','地方种植']
correct = 0
for i, pre_q in enumerate(predict_ques):
    for k in kind:
        if k in pre_q:
            if k in ground_ques[i]:
                correct += 1
# print(correct)
accuracy_ques = correct / len(ground_ques)
print('模糊类型判别准确率：', accuracy_ques)