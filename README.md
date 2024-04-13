# Numerical Sensitivity Enhancing and Reasoning Completeness Alignment for Quantitative Understanding
This system is designed to complete the task7 of the [SemEval2024]（https://sites.google.com/view/numeval/numeval），the performance of our system can be found in [overview paper] (https://sites.google.com/view/numeval/numeval#h.kjhrdjtve3l).

An overview of our system: (1) supervised fine-tuning with comparing numbers task for numerical sensitivity enhancement, (2) reward model training. (3) reinforcement learning via proximal policy optimization
with Reasoning Completeness Reward.
![system overview](https://github.com/Bit-numeval/NumEval/assets/160459346/6edd5f8b-29b6-4b1a-9d21-30b4fd8cd665)

## Data
We have uploaded JSON format data samples for SFT training, reward model training, and PPO training, which are Quantitative101 datasets expanded using GPT3.5. The reward model training data is human labeled.

-sft_data_example.json
```
{
    "task": "stressTest",
    "id": "0",
    "statement1": "At Veridux Corporation , there are 250 employees",
    "statement2": "At Veridux Corporation , there are less than 650 employees",
    "option1": "Entailment",
    "option2": "contradiction",
    "option3": "neutral",
    "response": " [1] . According to the premise \"there are 250 employees\", the hypothesis \"there are less than 650 employees\" is not contradicted by the premise.\n [2] . The number 250 is indeed less than 650, so the hypothesis can be justifiably inferred to be true.\n",
    "label": "Thus, the answer is option 1. #### 1"
}
```
-rm_data_example.json
```
{
    "sentence": "1.  The problem asks us to determine if a hypothesis can be justifiably inferred to be true, false, or cannot be determined based on a given premise. The first statement is the premise and the second statement is the hypothesis",
    "label": "0"
}
```
-ppo_data_example.json
```
{
    "task": "stressTest",
    "id": "0",
    "statement1": "At Veridux Corporation , there are 250 employees",
    "statement2": "At Veridux Corporation , there are less than 650 employees",
    "option1": "Entailment",
    "option2": "contradiction",
    "option3": "neutral",
    "label": 0
}
```

## Training
-gpt_generate.py: Used to call the API of gpt3.5 and generate an expanded dataset
-run_sft.sh: Used for sft training
-run_bert.sh: Used for reward model training
-run_ppo.sh: Used for ppo training
-test.py: Used to run the model on testsets of the tasks
-test_bert.py: Run reward model on the test set

Our SFT model is trained on [Abel-7B](https://github.com/GAIR-NLP/abel) with a learning rate of 3e-5, a warmup rate of 0.03, and a model max length of 1024. As for the RM, we choose to train on [BERT-large model](https://github.com/google-research/bert) as it well complete the classification tasks. It is trained with a learning rate of 2e-5, warmup rate of 0.05, and a model max length of 256, and is trained for 10 epochs. The
PPO training is implemented with Lora and [TRL](https://github.com/huggingface/trl), where the learning rate=1.41e-5, max new tokens=512. On a dataset of size 5470, each training epoch takes around 55 hours on 4 A100s.
