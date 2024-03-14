from transformers import AutoTokenizer,AutoModelForSequenceClassification
from transformers import pipeline
from tqdm import tqdm
import json
import sys
import sklearn.metrics

IGNORE_INDEX = -100


model_path = "bert_output3/checkpoint-116"

tokenizer = AutoTokenizer.from_pretrained(model_path,padding_side='left')
model = AutoModelForSequenceClassification.from_pretrained(model_path,max_length=128)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
model.eval()


pipe = pipeline("text-classification", model=model, tokenizer=tokenizer,device_map="auto")
'''
result = pipe("[1] . According to the premise \"there are 250 employees\", the hypothesis \"there are less than 650 employees\" is not contradicted by the premise.\n ")
print(result)

'''

data_path = "eval_mix_steps.json"
out_path = "output_bert_116.json"

with open(data_path,'r') as f:
    dataset = json.load(f)

preds=[]
with open(out_path, mode='a+', encoding='utf-8') as fout:
    for input in tqdm(dataset):
        sentence = input['sentence']
        if len(sentence)>512:
             sentence=sentence[:512]
        out = pipe(sentence)
        res = dict()
        res['predict'] = out[0]['label']
        res['score'] = out[0]['score']
        json.dump(res, fout, ensure_ascii=False)
        fout.write('\n')
        sys.stdout.flush()
        preds.append(res['predict'])

targets=[]
with open("eval_mix_steps.json","r+",encoding='utf-8') as f:
	data = json.load(f)
	for d in data:
		targets.append(d['label'])

score = sklearn.metrics.f1_score(y_true=targets, y_pred=preds,average='micro')
print("f1: ",score)
score = sklearn.metrics.accuracy_score(y_true=targets, y_pred=preds)
print("acc: ",score)
