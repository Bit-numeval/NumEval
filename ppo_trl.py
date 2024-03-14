import torch
from tqdm import tqdm
import re
import json
import math
import numpy as np
import sys
from peft import get_peft_config, get_peft_model, LoraConfig
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import LengthSampler, respond_to_batch
import torch.nn.functional as F

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

PROMPT_DICT = {
    "prompt_QQA":(
        "I will first raise a question and then provide two options. Please choose the correct answer after providing the inference process step by step, in the format of 'the answer is option 1. #### 1'. If calculation is involved, please provide the equations during the calculation process. Using numbers like '\n 1.' or '\n [1]' to mark steps. "

        "### question: {question}\n### option1:{option1}\n### option2:{option2}\n### Response: Let's think step by step.\n "
    ),
    "prompt_QP_headline":(
        "Below is a news headline with a number masked, predict the correct magnitude of the masked numeral after providing the inference process step by step, in the format of 'the correct magnitude is 1'. Using numbers like '\n 1.' or '\n [1]' to mark steps.  The magnitude of decimals is 0 and magnitudes greater than 6 is represented as magnitude 7.' Provide the answer in the format of 'The magnitude is 0. #### 0'."

        "### headline:\n{question}\n### Response: Let's think step by step.\n"
    
    ),
    "prompt_QP_comment":(
        "Below is a comment with a number masked, predict the correct magnitude of the masked numeral after providing the inference process step by step, in the format of 'the correct magnitude is 1'. Using numbers like '\n 1.' or '\n [1]' to mark steps. The magnitude of decimals is 0 and magnitudes greater than 6 is represented as magnitude 7.' Provide the answer in the format of 'The magnitude is 0. #### 0'."

        "### comment:\n{question}\n### Response: Let's think step by step.\n"
    ),
    "prompt_AWPNLI": (
        "I will first raise two statements and then provide two options which are entailment and contradiction. The first statement is the given premise, the second statement is the hypothesis. You should determine if the hypothesis can be justifiably inferred to be true (option 1 : entailment) or false (option 2 : contradiction) base on the premise. Please choose the correct answer after providing the inference process step by step, in the format of 'the answer is option 1. #### 1'. If calculation is involved, please provide the equations during the calculation process. Using numbers like '\n 1.' or '\n [1]' to mark steps. Choose the correct answer in the format of 'the answer is option 1. #### 1'. "
       
        "### statement1: {statement1}\n### statement2:{statement2}\n### option1:{option1}\n### option2:{option2}\n### Response: Let's think step by step.\n "
        
    ),
    "prompt_NewsNLI": (
        "I will first raise two statements and then provide two options which are entailment and neutral. The first statement is the given premise, the second statement is the hypothesis. You should determine if the hypothesis can be justifiably inferred to be true (option 1 : entailment) or cannot be determined (option 2 : neutral) base on the premise.You should pay attention to additional information rather than shared information, especially paying attention to whether the numbers are reasonable and derived from the premise. If there is information that is not mentioned in the premise or cannot be directly inferred in the hypothesis, then the hypothesis cannot be determined. Please choose the correct answer after providing the inference process step by step, in the format of 'the answer is option 1 #### 1'. Using numbers like '\n 1.' or '\n [1]' to mark steps."

        "### statement1: {statement1}\n### statement2:{statement2}\n### option1:{option1}\n### option2:{option2}\n### Response: Let's think step by step.\n "    
    ),
    "prompt_RTE":(
        "I will first raise two statements and then provide two options which are entailment and neutral. The first statement is the given premise, the second statement is the hypothesis. You should determine if the hypothesis can be justifiably inferred to be true (option 1 : entailment) or cannot be determined (option 2 : neutral) base on the premise. You should pay attention to additional information rather than shared information, especially paying attention to whether the numbers are reasonable and derived from the premise. If there is information that is not mentioned in the premise or cannot be directly inferred in the hypothesis, then the hypothesis cannot be determined. Please choose the correct answer after providing the inference process step by step, in the format of 'the answer is option 1. #### 1'. Using numbers like '\n 1.' or '\n [1]' to mark steps."

        "### statement1: {statement1}\n### statement2:{statement2}\n### option1:{option1}\n### option2:{option2}\n### Response: Let's think step by step.\n"     
    ),
    "prompt_RedditNLI": (
        "I will first raise two statements and then provide three options which are entailment, contradiction and neutral. The first statement is the given premise, the second statement is the hypothesis. You should determine if the hypothesis can be justifiably inferred to be true (option 1 : entailment), false (option 2 : contradiction) or cannot be determined (option 3 : neutral) base on the premise. Please choose the correct answer after providing the inference process step by step, in the format of 'the answer is option 1. #### 1'.  Using numbers like '\n 1.' or '\n [1]' to mark steps."

        "### statement1: {statement1}\n### statement2:{statement2}\n### option1:{option1}\n### option2:{option2}\n### option3:{option3}\n### Response: Let's think step by step.\n"
    ),
    "prompt_stress":(
        "I will first raise two statements and then provide three options which are entailment, contradiction and neutral. The first statement is the given premise, the second statement is the hypothesis. You should determine if the hypothesis can be justifiably inferred to be true (option 1 : entailment), false (option 2 : contradiction) or cannot be determined (option 3 : neutral) base on the premise. You should especially pay attention to whether the numbers are reasonable and derived from the premise. If there is information that is cannot be directly inferred in the hypothesis, then the hypothesis cannot be determined. Please choose the correct answer after providing the inference process step by step, in the format of 'the answer is option 1. #### 1'. Using numbers like '\n 1.' or '\n [1]' to mark steps."

        "### statement1: {statement1}\n### statement2:{statement2}\n### option1:{option1}\n### option2:{option2}\n### option3:{option3}\n### Response: Let's think step by step.\n "
    )
}

output_dir = "ppo-output/"
batch_size=8
train_epoch_num = 3

config = PPOConfig(
    model_name="Abel-sft",
    learning_rate=1.41e-5,
    batch_size=batch_size,
    gradient_accumulation_steps=8,
    remove_unused_columns=False,
    is_peft_model=True
)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=True,
    device_map=None,
    peft_config=lora_config,
    use_cache=True
)
ref_model = create_reference_model(model)

# tokenizer.pad_token = tokenizer.eos_token

rm_model_path = "bert_output"
# rm_model_path = "/home/ybai/SemEval2024/rm_output"
# rm_model_path = "/home/jwli/SemEval2024/RL/open_llama_3b_v2"

rm_config = AutoConfig.from_pretrained(
    rm_model_path,
    num_labels=3,
    use_fast=False
)
rm_tokenizer = AutoTokenizer.from_pretrained(
    rm_model_path,
    return_token_type_ids=False,
    use_fast=False
)
rm_model = AutoModelForSequenceClassification.from_pretrained(
    rm_model_path,
    config=rm_config
)
rm_model.eval()

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 512,
    "eos_token_id": -1,
}


# load the dataset into a DataFrame.
def build_dataset(config):
    """
    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    raw_datasets = load_dataset("json", data_files={"train":"ppo_data_train.json"})

    def tokenize(sample):
  
        if sample['task'] == 'QP_comment':
            prompt = PROMPT_DICT["prompt_QP_comment"]
        elif sample['task'] == 'QP_headline':
            prompt = PROMPT_DICT["prompt_QP_headline"]
        elif sample['task'] == 'stressTest':
            prompt = PROMPT_DICT["prompt_stress"]
        elif sample['task'] == 'AWPNLI':
            prompt = PROMPT_DICT["prompt_AWPNLI"]
        elif sample['task'] == 'NewsNLI':
            prompt = PROMPT_DICT["prompt_NewsNLI"]
        elif sample['task'] == 'RTE':
            prompt = PROMPT_DICT["prompt_RTE"]
        elif sample['task'] == 'RedditNLI':
            prompt = PROMPT_DICT["prompt_RedditNLI"]
        elif sample['task'] == 'QQA':
            prompt = PROMPT_DICT["prompt_QQA"]

        sample["query"] = prompt.format_map(sample)
        sample["input_ids"] = tokenizer.encode(sample["query"],padding='do_not_pad')
        return sample
    
    
    # train_dataset = raw_datasets["train"].select(list(range(5))) # 先取前20条
    train_dataset = raw_datasets["train"]
    # import IPython;IPython.embed();exit()
    ds = train_dataset.map(tokenize, batched=False)
    ds.set_format(type="torch")

    labels = []
    for i in range(len(train_dataset)):
        labels.append(train_dataset[i]['label'])
    return ds,labels

dataset,labels = build_dataset(config)


def collator(data):
    keys_to_include = ['label', 'query', 'input_ids', 'task']
    return {
        key: [d[key] for d in data]
        for key in keys_to_include
    }
    # return dict((key, [d[key] for d in data]) for key in data[0])


ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)


def extract_steps(text):
    pat = r'(\.\s+\d+\.\s)|(\?\s+\d+\.\s)|(\n*1+\.\s+)|(\\n\s*\d+\.)|(\[\d\])'
    matches = re.findall(pat, text)
    split_texts = re.split(pat,text)
    
    step_ls = []
    
    for j in range(0,len(split_texts),6): #注意：这里的6是匹配规则数+1
        step = dict()
        a=split_texts[j-5:j]
        num = next((x for x in a if x), None)
        text = split_texts[j]
        if num:
            num = num.replace('\n','')
            if num[0] == '.':
                num = num[1:]
            if num[0] == ' ':
                num = num[1:]
        else:
            # num = ''
            continue
        if text: 
            step["text"] = (f'{num} {text}')
            step_ls.append(step)

    return step_ls


def get_score(sentence):
    if len(sentence)==0:
        score = torch.tensor(0)
    else:
        inputs = rm_tokenizer(sentence, return_tensors="pt", max_length=512, padding='longest', truncation=True)
        # inputs = inputs.to(rm_model.device) # 报错Expected all tensors to be on the same device
        with torch.autocast('cuda', dtype=torch.float16):
            outputs = rm_model(input_ids = inputs['input_ids'], attention_mask=inputs['attention_mask'])
        
        logits = outputs.logits
        logits_softmax = F.softmax(logits, dim=1)
        score = logits_softmax[0][2]   #得1分的概率  label_list = ['-1','0','1']
    
    return score

def get_rewards(texts):
    rewards = []

    for text in texts:
        step_ls = extract_steps(text)
        step_num = len(step_ls)

        if step_num <= 1:
            score = torch.tensor(0)
            # score = get_score(text)
            rewards.append(score)

            data = {'step': text, 'score': score.item()}
            with open(output_dir + 'step_reward.jsonl', 'a+') as f:
                f.write(json.dumps(data))
                f.write('\n')
                sys.stdout.flush()
        
        else:
            step_rewards=[]
            for j in range(step_num):
                if j == 0:  #第一步一般是复述题目分数会最高，降低权重
                    sentence =  step_ls[j]['text']
                    weight = 0.2
                else:
                    sentence = step_ls[j-1]['text'] + ' </s><s>' +  step_ls[j]['text']
                    weight = 1.0

                score = get_score(sentence)*weight
                step_rewards.append(score)

                data = {'step': sentence, 'score': score.item()}
                with open(output_dir + 'step_reward.jsonl', 'a+') as f:
                    f.write(json.dumps(data))
                    f.write('\n')
                    sys.stdout.flush()

            # 一个解答过程的分数是步骤正确概率的几何均值*k
            solution_reward = 1
            for r in step_rewards:
                solution_reward *= r
 
            mean = float(math.pow(solution_reward, 1/step_num))

            # k = 1
            # if step_num!=4:
            #     k = 1/abs(4-k)

            # k: 希望大部分解答的步数集中在4左右
            if step_num <= 1:
                k = 0
            else:
                mu = 4   # 期望值
                sigma = 2   # 标准差
                k = (5 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(step_num - mu)**2 / (2 * sigma**2))

            reward = mean*k
            rewards.append(reward)

    rewards = [torch.tensor(r) for r in rewards]
    return rewards




for epoch in range(train_epoch_num):
    for batch in tqdm(ppo_trainer.dataloader,desc="epoch"+str(epoch)):
        
        query_tensors = batch["input_ids"]
        
        # response_tensors = []
        # for query in tqdm(query_tensors):
        #     response = ppo_trainer.generate(query, **generation_kwargs)
        #     response_tensors.append(response.squeeze())
        # batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        response_tensors=[]
        # ref_response_tensors=[]
        while(len(response_tensors)==0):
            print("### generating response...")
            # batch generate要去掉padding，否则KL negative
            response_tensors= ppo_trainer.generate(
                query_tensors,
                batch_size=batch_size,
                generate_ref_response=False, 
                return_prompt=False, 
                **generation_kwargs
            )
        batch["response"] = tokenizer.batch_decode(response_tensors)
        # batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

        
        tasks = batch['task']
        
        # texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        texts = batch["response"]

        rewards=[]
        while len(rewards)==0:
            print("### computing scores...")
            rewards = get_rewards(texts)

        # ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
        # ref_texts = batch["ref_response"]
        # ref_rewards=[]
        # while len(ref_rewards)==0:
        #     print("### computing ref scores...")
        #     ref_rewards = get_rewards(ref_texts)
        # batch["ref_rewards"] = ref_rewards

        
        with open(output_dir + 'generate.jsonl', 'a+') as f:
            for text, reward in zip(texts, rewards):
                data = {'text': text, 'reward': reward.item()}
                f.write(json.dumps(data))
                f.write('\n')
                sys.stdout.flush()

        # with open('ref_generate.jsonl', 'a+') as f:
        #     for text, reward in zip(ref_texts, ref_rewards):
        #         data = {'text': text, 'reward': reward.item()}
        #         f.write(json.dumps(data))
        #         f.write('\n')
        
        print("### Running PPO step...")
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        print("### logging stats...")
        ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response"])
        
        
    model.save_pretrained(output_dir)
    print("epoch-" +str(epoch)+ "model saved")
    tokenizer.save_pretrained(output_dir)
    print("epoch-" +str(epoch)+ "tokenizer saved")
