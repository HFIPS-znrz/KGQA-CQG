import json
import numpy as np
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer
import re
from collections import Counter
import string

# model_name_or_path = 'bert-base-chinese'
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

file_prediction = '../final_output_single.txt'
file_ground = '/home/dell/zjx/MSParS_V2.0_single_2/data/final_tgt_test.txt'

predict = []
ground = []
with open(file_prediction, 'r') as file:
    for line in file:
        predict.append(line.strip().replace(" ", ""))

with open(file_ground, 'r') as file:
    for line in file:
        ground.append(line.strip())


# 计算答案生成模块的评价指标  BLEU和ROUGE
def compute_metrics(decoded_preds, decoded_labels):
    # with open(answer_path, 'r', encoding='utf-8') as file:
    #     data = json.load(file)

    # # 删除没有答案的问答
    # for d in data:
    #     if d['A'] == '无':
    #         data.remove(d)
    # decoded_preds = [d['predict_A'] for d in data]
    # decoded_labels = [d['A'] for d in data]
    # print(decoded_preds[:10])
    # print(decoded_labels[:10])
    # print(preds)
    
    # if isinstance(preds, tuple):
    #     preds = preds[0]
    # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # # if data_args.ignore_pad_token_for_loss:
    #     # Replace -100 in the labels as we can't decode them.
    # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    score_dict = {  # rouge-l  最长公共子序列，子序列不一定连续，但要按顺序，分子是相交的部分
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [], 
        "bleu-4": []
    }
    for pred, label in zip(decoded_preds, decoded_labels):
        hypothesis = list(jieba.cut(pred))  # 预测
        reference = list(jieba.cut(label))  # 真实
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
        result = scores[0]
        # print(result)
        for k, v in result.items():
            score_dict[k].append(round(v["f"] * 100, 4))  # round 四舍五入。4-保留小数点后四位
        # bleu_score = sentence_bleu([list(label)], list(pred), weights=(1,0,0,0), smoothing_function=SmoothingFunction().method3)  # BLEU-1
        # bleu_score = sentence_bleu([list(label)], list(pred), weights=(0.5,0.5,0,0), smoothing_function=SmoothingFunction().method3)  # BLEU-2
        # bleu_score = sentence_bleu([list(label)], list(pred), weights=(0.33,0.33,0.33,0), smoothing_function=SmoothingFunction().method3)  # BLEU-3
        bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)  # 考虑了BLEU 1、2、3、4加权平均，默认BLEU-4都是0.25
        score_dict["bleu-4"].append(round(bleu_score * 100, 4))

    for k, v in score_dict.items():
        score_dict[k] = float(np.mean(v))
    return score_dict


# 计算答案生成模块的评价指标  BLEU和ROUGE
def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    # 首先把prediction和ground_truth标准化（即用上面的函数进行处理）
    prediction_tokens = normalize_answer(prediction).split()[0]
    ground_truth_tokens = normalize_answer(ground_truth).split()[0]
    # print(prediction_tokens)
    # print(ground_truth_tokens)
    # 统计他们共有的字符
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    # 计算共有的字符的总量
    num_same = sum(common.values())
    # print(num_same)
    if num_same == 0:
        return 0
    # 计算precision
    precision = 1.0 * num_same / len(prediction_tokens)
    # 计算recall
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1



def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_metric(prediction, ground_truth):
    f1_all = []
    em_all = []
    for i in range(len(predict)):
        prediction = predict[i]
        ground_truth = ground[i]
        f1 = f1_score(prediction, ground_truth)  # 0.4421
        em_score = exact_match_score(prediction, ground_truth)
        f1_all.append(f1)
        em_all.append(em_score)
    # print(f1_all)
    print('f1:', np.mean(f1_all))
    # print(em_all)  # em_match=0 施肥和栽培问答不合适，没有完全固定的答案



score_dict = compute_metrics(predict, ground)
print(score_dict)
f1_metric(predict, ground)