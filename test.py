import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import pipeline
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
from typing import Dict,Sequence
import json
import sys
import re



IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

PROMPT_DICT = {
    "prompt_QQA":(
        "I will first raise a question and then provide two options. Please choose the correct answer after providing the inference process step by step, in the format of 'the answer is option 1. #### 1'. "

        "### question: {question}\n### option1:{option1}\n### option2:{option2}\n### Response: Let's think step by step.\n "
    ),
    "prompt_QP_headline":(
        "Below is a news headline with a number masked, predict the correct magnitude of the masked numeral after providing the inference process step by step, in the format of 'the correct magnitude is 1'. The magnitude of decimals is 0 and magnitudes greater than 6 is represented as magnitude 7.' Provide the answer in the format of 'The magnitude is 0. #### 0'."

        "### headline:\n{question}\n### Response: Let's think step by step.\n"
    
    ),
    "prompt_QP_comment":(
        "Below is a comment with a number masked, predict the correct magnitude of the masked numeral after providing the inference process step by step, in the format of 'the correct magnitude is 1'. The magnitude of decimals is 0 and magnitudes greater than 6 is represented as magnitude 7.' Provide the answer in the format of 'The magnitude is 0. #### 0'."

        "### comment:\n{question}\n### Response: Let's think step by step.\n"
    ),
    "prompt_AWPNLI": (
        "I will first raise two statements and then provide two options which are entailment and contradiction. The first statement is the given premise, the second statement is the hypothesis. You should determine if the hypothesis can be justifiably inferred to be true (option 1 : entailment) or false (option 2 : contradiction) base on the premise. Please choose the correct answer after providing the inference process step by step, in the format of 'the answer is option 1. #### 1'."
       
        "### statement1: {statement1}\n### statement2:{statement2}\n### option1:{option1}\n### option2:{option2}\n### Response: Let's think step by step.\n "
        
    ),
    "prompt_NewsNLI": (
        "I will first raise two statements and then provide two options which are entailment and neutral. The first statement is the given premise, the second statement is the hypothesis. You should determine if the hypothesis can be justifiably inferred to be true (option 1 : entailment) or cannot be determined (option 2 : neutral) base on the premise.You should pay attention to additional information rather than shared information, especially paying attention to whether the numbers are reasonable and derived from the premise. If there is information that is not mentioned in the premise or cannot be directly inferred in the hypothesis, then the hypothesis cannot be determined. Please choose the correct answer after providing the inference process step by step, in the format of 'the answer is option 1 #### 1'."

        "### statement1: {statement1}\n### statement2:{statement2}\n### option1:{option1}\n### option2:{option2}\n### Response: Let's think step by step.\n "    
    ),
    "prompt_RTE":(
        "I will first raise two statements and then provide two options which are entailment and neutral. The first statement is the given premise, the second statement is the hypothesis. You should determine if the hypothesis can be justifiably inferred to be true (option 1 : entailment) or cannot be determined (option 2 : neutral) base on the premise. You should pay attention to additional information rather than shared information, especially paying attention to whether the numbers are reasonable and derived from the premise. If there is information that is not mentioned in the premise or cannot be directly inferred in the hypothesis, then the hypothesis cannot be determined. Please choose the correct answer after providing the inference process step by step, in the format of 'the answer is option 1. #### 1'."

        "### statement1: {statement1}\n### statement2:{statement2}\n### option1:{option1}\n### option2:{option2}\n### Response: Let's think step by step.\n"     
    ),
    "prompt_RedditNLI": (
        "I will first raise two statements and then provide three options which are entailment, contradiction and neutral. The first statement is the given premise, the second statement is the hypothesis. You should determine if the hypothesis can be justifiably inferred to be true (option 1 : entailment), false (option 2 : contradiction) or cannot be determined (option 3 : neutral) base on the premise. Please choose the correct answer after providing the inference process step by step, in the format of 'the answer is option 1. #### 1'."

        "### statement1: {statement1}\n### statement2:{statement2}\n### option1:{option1}\n### option2:{option2}\n### option3:{option3}\n### Response: Let's think step by step.\n"
    ),
    "prompt_stress":(
        "I will first raise two statements and then provide three options which are entailment, contradiction and neutral. The first statement is the given premise, the second statement is the hypothesis. You should determine if the hypothesis can be justifiably inferred to be true (option 1 : entailment), false (option 2 : contradiction) or cannot be determined (option 3 : neutral) base on the premise. You should especially pay attention to whether the numbers are reasonable and derived from the premise. If there is information that is cannot be directly inferred in the hypothesis, then the hypothesis cannot be determined. Please choose the correct answer after providing the inference process step by step, in the format of 'the answer is option 1. #### 1'."

        "### statement1: {statement1}\n### statement2:{statement2}\n### option1:{option1}\n### option2:{option2}\n### option3:{option3}\n### Response: Let's think step by step.\n "
    )
}

model_path = "models/Llama-2-7b-hf"

model = LlamaForCausalLM.from_pretrained(model_path,device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path,model_max_length=512)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

model.eval()


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s for s in sources]
    examples_tokenized = [_tokenize_fn(strings, tokenizer) for strings in examples]
    
    input_ids = examples_tokenized
    labels = [t for t in targets]

    return dict(input_ids=input_ids, labels=labels)

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


class MyDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(MyDataset, self).__init__()
        print("Loading data...")
        with open(data_path, 'r') as file:
            list_data_dict = json.load(file)

        print("Formatting inputs...")
        self.sources=[]
        
        for example in list_data_dict:
            prompt = PROMPT_DICT["prompt_QP_headline"]
            if example['task'] == 'QP_comment':
                prompt = PROMPT_DICT["prompt_QP_comment"]
            elif example['task'] == 'QP_headline':
                prompt = PROMPT_DICT["prompt_QP_headline"]
            elif example['task'] == 'stressTest':
                prompt = PROMPT_DICT["prompt_stress"]
            elif example['task'] == 'AWPNLI':
                prompt = PROMPT_DICT["prompt_AWPNLI"]
            elif example['task'] == 'NewsNLI':
                prompt = PROMPT_DICT["prompt_NewsNLI"]
            elif example['task'] == 'RTE':
                prompt = PROMPT_DICT["prompt_RTE"]
            elif example['task'] == 'RedditNLI':
                prompt = PROMPT_DICT["prompt_RedditNLI"]
            elif example['task'] == 'QQA':
                prompt = PROMPT_DICT["prompt_QQA"]

            self.sources.append(prompt.format_map(example))


        self.targets = [f"{example['label']}{tokenizer.eos_token}" for example in list_data_dict]


    def __len__(self):
        # return len(self.input_ids)
        return len(self.sources)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # return dict(input_ids=self.input_ids[i], labels=self.labels[i])
        # return self.input_ids[i]
        return self.sources[i]


def extract_ans(response):
    match = re.search(r'the answer is option.*(\d+)', response, re.IGNORECASE)
    match2 = re.search(r'the correct answer is \"option (\d+)', response, re.IGNORECASE)
    match3 = re.search(r'the correct answer is option (\d+)', response, re.IGNORECASE)
    match4 = re.search(r'\[option (\d+)', response, re.IGNORECASE)
    match5 = re.search(r'\(option (\d+)\)', response, re.IGNORECASE)
    match6 = re.search(r'the correct answer is.*option.*(\d+)', response, re.IGNORECASE)
    match7 = re.search(r'the magnitude is.*(\d+)', response, re.IGNORECASE)

    if match:
        result = match.group(1)
        ans = str(int(result)-1)
    elif match2:
        result = match2.group(1)
        ans = str(int(result)-1)
    elif match3:
        result = match3.group(1)
        ans = str(int(result)-1)
    elif match4:
        result = match4.group(1)
        ans = str(int(result)-1)
    elif match5:
        result = match5.group(1)
        ans = str(int(result)-1)
    elif match6:
        result = match6.group(1)
        ans = str(int(result)-1)
    elif match7:
        result = match7.group(1)
        ans = str(int(result))
    else:
        ans = "Error"
    
    return ans


def run(data_path,out_path):
    dataset = MyDataset(data_path,tokenizer)

    # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,device_map="auto")
    # generated_characters = 0
    # for out in tqdm(pipe(dataset, batch_size=1, return_full_text=False), total=len(dataset)):
    #     print(out[0]["generated_text"])

    total = dataset.__len__()
    with open(out_path, mode='a+', encoding='utf-8') as fout:
        with tqdm(total=total) as pbar:
            for i in range(total):
                prompt = dataset[i]
                res = dict()
                res['id'] = str(i)
                inputs = tokenizer(prompt, return_tensors="pt",padding="longest",truncation=True)
                input_ids=inputs["input_ids"].cuda()
                with torch.no_grad():
                    generation_outputs = model.generate(
                                            input_ids=input_ids,
                                            num_beams=1,
                                            do_sample=True,max_new_tokens=128,
                                            return_dict_in_generate=True,
                                        )
                    response=tokenizer.decode(generation_outputs.sequences[0][len(input_ids[0]):],skip_special_tokens=True)
                    ans = extract_ans(response)
                    
                res['answer'] = ans
                res['generated_text'] = response
                json.dump(res, fout, ensure_ascii=False)
                fout.write('\n')
                sys.stdout.flush()

                pbar.update(1)


data_path = "testsets/stress_test.json"
out_path = "llama_stress.jsonl"
run(data_path,out_path)
