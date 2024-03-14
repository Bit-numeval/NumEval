#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import os
import evaluate
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import numpy as np
import re
import json
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer
import sklearn.metrics

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_CND":(
        "### question: {question}\n### option1:{option1}\n### option2:{option2}\n ### Response: Let's think step by step.\n {response}"
    ),
    "prompt_QQA":(
        "I will first raise a question and then provide two options. Please choose the correct answer after providing the inference process step by step, in the format of 'the answer is option 1. #### 1'. "

        "### question: {question}\n### option1:{option1}\n### option2:{option2}\n### Response: Let's think step by step.\n {response}"
    ),
    "prompt_QP_headline":(
        "Below is a news headline with a number masked, predict the correct magnitude of the masked numeral after providing the inference process step by step, in the format of 'the correct magnitude is 1'. The magnitude of decimals is 0 and magnitudes greater than 6 is represented as magnitude 7.' Provide the answer in the format of 'The magnitude of the masked number is 0. #### 0'."

        "### headline:\n{question}\n### Response: Let's think step by step.\n {response}"
    
    ),
    "prompt_QP_comment":(
        "Below is a comment with a number masked, predict the correct magnitude of the masked numeral after providing the inference process step by step, in the format of 'the correct magnitude is 1'. The magnitude of decimals is 0 and magnitudes greater than 6 is represented as magnitude 7.' Provide the answer in the format of 'The magnitude of the masked number is 0. #### 0'."

        "### comment:\n{question}\n### Response: Let's think step by step.\n {response}"
    ),
    "prompt_AWPNLI": (
        "I will first raise two statements and then provide two options which are entailment and contradiction. The first statement is the given premise, the second statement is the hypothesis. You should determine if the hypothesis can be justifiably inferred to be true (option 1 : entailment) or false (option 2 : contradiction) base on the premise. Please choose the correct answer after providing the inference process step by step, in the format of 'the answer is option 1. #### 1'."
       
        "### statement1: {statement1}\n### statement2:{statement2}\n### option1:{option1}\n### option2:{option2}\n### Response: Let's think step by step.\n {response}"
        
    ),
    "prompt_NewsNLI": (
        "I will first raise two statements and then provide two options which are entailment and neutral. The first statement is the given premise, the second statement is the hypothesis. You should determine if the hypothesis can be justifiably inferred to be true (option 1 : entailment) or cannot be determined (option 2 : neutral) base on the premise. If there is information that is not mentioned in the premise or cannot be directly inferred in the hypothesis, then the hypothesis cannot be determined. Please choose the correct answer after providing the inference process step by step, in the format of 'the answer is option 1 #### 1'."

        "### statement1: {statement1}\n### statement2:{statement2}\n### option1:{option1}\n### option2:{option2}\n### Response: Let's think step by step.\n {response}"    
    ),
    "prompt_RTE":(
        "I will first raise two statements and then provide two options which are entailment and neutral. The first statement is the given premise, the second statement is the hypothesis. You should determine if the hypothesis can be justifiably inferred to be true (option 1 : entailment) or cannot be determined (option 2 : neutral) base on the premise. If there is information that is not mentioned in the premise or cannot be directly inferred in the hypothesis, then the hypothesis cannot be determined. Please choose the correct answer after providing the inference process step by step, in the format of 'the answer is option 1. #### 1'."

        "### statement1: {statement1}\n### statement2:{statement2}\n### option1:{option1}\n### option2:{option2}\n### Response: Let's think step by step.\n {response}"     
    ),
    "prompt_RedditNLI": (
        "I will first raise two statements and then provide three options which are entailment, contradiction and neutral. The first statement is the given premise, the second statement is the hypothesis. You should determine if the hypothesis can be justifiably inferred to be true (option 1 : entailment), false (option 2 : contradiction) or cannot be determined (option 3 : neutral) base on the premise. Please choose the correct answer after providing the inference process step by step, in the format of 'the answer is option 1. #### 1'."

        "### statement1: {statement1}\n### statement2:{statement2}\n### option1:{option1}\n### option2:{option2}\n### option3:{option3}\n### Response: Let's think step by step.\n {response}"
    ),
    "prompt_stress":(
        "I will first raise two statements and then provide three options which are entailment, contradiction and neutral. The first statement is the given premise, the second statement is the hypothesis. You should determine if the hypothesis can be justifiably inferred to be true (option 1 : entailment), false (option 2 : contradiction) or cannot be determined (option 3 : neutral) base on the premise. If there is information that is cannot be directly inferred in the hypothesis, then the hypothesis cannot be determined. Please choose the correct answer after providing the inference process step by step, in the format of 'the answer is option 1. #### 1'."

        "### statement1: {statement1}\n### statement2:{statement2}\n### option1:{option1}\n### option2:{option2}\n### option3:{option3}\n### Response: Let's think step by step.\n {response}"
    )
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    valid_data_path:str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    # project_name:Optional[str] = field(default=None)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


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


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer,split="train"):
        super(SupervisedDataset, self).__init__()

        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)
        # print(list_data_dict)

        if split == "validation":
            list_data_dict=list_data_dict[:200]

        logging.warning("Formatting inputs...")
        
        sources = []

        for example in list_data_dict:
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
            elif example['task'] == 'CND':
                prompt = PROMPT_DICT["prompt_CND"]
            
            sources.append(prompt.format_map(example))

        targets = [f"{example['label']} {tokenizer.eos_token}" for example in list_data_dict]
       
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        # print(labels)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args,training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset=None
    if data_args.train_data_path is not None and training_args.do_train:
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.train_data_path)
    eval_dataset=None
    if data_args.valid_data_path is not None and training_args.do_eval:
        eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.valid_data_path,split="validation")
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # os.environ["WANDB_PROJECT"]=training_args.project_name

    model = transformers.LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # import IPython; IPython.embed(); exit()
    def normalize_final_answer(final_answer: str) -> str:
        """Normalize a final answer to a quantitative reasoning question."""

        # Extract answer that is in LaTeX math, is bold,
        # is surrounded by a box, etc.
        final_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', final_answer)
        final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
        final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
        final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
        final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)

        # Normalize 100,000 -> 100000
        if final_answer.replace(',', '').isdigit():
            final_answer = final_answer.replace(',', '')

        return final_answer

    def postprocess_text(preds, labels):
        import re
        # preds = [pred.strip() for pred in preds]
        results = []
        for pred in preds:
            pred = normalize_final_answer(pred)
            match = re.search(r'#+\s*(\d+)', pred, re.IGNORECASE)
            
            if match:
                result = str(int(match.group(1))-1)
            else:
                print(pred)
            
            results.append(result)

        labels = [[label.strip()] for label in labels]

        with open("predicts.json",'w',encoding='utf-8',errors='ignore') as f:
            json.dump(preds,f,ensure_ascii=False, indent=2)

        return results, labels

    def compute_metrics(eval_preds):
        metric = evaluate.load("/home/ybai/SemEval2024/accuracy")
        preds, labels = eval_preds
        # print(type(eval_preds))
        if isinstance(preds, tuple):
            preds = preds[0]
        # print(type(preds))
        #import IPython;IPython.embed();exit()

        # print(type(preds[0]))
        # print(type(preds[0][0]))
        # print("preds[0]",len(preds[0]))
        # print("preds[0]",preds[0][0].size())
        # preds=np.argmax(preds,axis=-1)
    
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # if data_args.ignore_pad_token_for_loss:
            # 默认是的
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(y_true=decoded_labels, y_pred=decoded_preds)

        # result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        # result = {"rouge": result["score"]}
        print(result)
        # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        # result["gen_len"] = np.mean(prediction_lens)
        # result = {k: round(v, 4) for k, v in result.items()}
        return result

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args,training_args=training_args)
    
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args,\
                      compute_metrics=compute_metrics,**data_module)
    if training_args.do_train:
        print("training!!!!!!!!")
        trainer.train(training_args.resume_from_checkpoint)
    # trainer.train(resume_from_checkpoint=True)
        trainer.save_state()
        trainer.save_model(output_dir=training_args.output_dir)

    if training_args.do_eval:
        print("evaluating!!!!!!!!")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)



if __name__ == "__main__":
    train()
