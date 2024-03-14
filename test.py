import torch
from tqdm import tqdm
import re
import json
import numpy as np
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
import torch.nn.functional as F

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

rm_model_path = "/home/ybai/SemEval2024/rm_output"
# rm_model_path = "/home/jwli/SemEval2024/RL/open_llama_3b_v2"

rm_config = AutoConfig.from_pretrained(
    rm_model_path,
    num_labels=3,
)
rm_tokenizer = AutoTokenizer.from_pretrained(
    rm_model_path,return_token_type_ids=False
)
rm_model = AutoModelForSequenceClassification.from_pretrained(
    rm_model_path,
    config=rm_config
)
rm_model.eval()

label_list = ['-1','0','1']
targets=[]
preds=[]
with open("/home/ybai/SemEval2024/eval_mix_steps.json",'r',encoding='utf-8') as f:
	dataset = json.load(f)
	for data in tqdm(dataset):
		sentence = data['sentence']
		inputs = rm_tokenizer(sentence, return_tensors="pt", max_length=2048, padding='longest', truncation=True)
		with torch.autocast('cuda', dtype=torch.float16):
			outputs = rm_model(input_ids = inputs['input_ids'], attention_mask=inputs['attention_mask'])
			logits = outputs.logits
			logits_softmax = F.softmax(logits, dim=1)
			score = logits_softmax[0][2]   
        	
			pred = logits.argmax(dim=1)
			res = label_list[pred]
		data['logit'] = score
		data['predict'] = res

		targets.append(data['label'])
		preds.append(res)

with open("/home/ybai/SemEval2024/predict_mix_steps.json",'r',encoding='utf-8') as f:
	json.dump(dataset,f,indent=2)

import sklearn.metrics
score = sklearn.metrics.accuracy_score(y_true=targets, y_pred=preds)
print("============================ accuracy =========================================")
print(score)
print("===============================================================================")