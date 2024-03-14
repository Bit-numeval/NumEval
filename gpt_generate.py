import json
import sys
import pandas as pd
import openai
from tqdm import tqdm
import time
import os


openai.api_key = ""
openai.api_base = ""

QA = """
I will first raise a question and then provide two options. Please choose the correct answer after providing the inference process step by step, in the format of "the answer is option 1”. If calculation is involved, please provide the equations during the calculation process. Using numbers in '[]' to mark steps, like '[1]'.

For example:
[question]: The ranger and the rustler both were riding horses that galloped at the same speed.  The rustler left at 01:00 where as the ranger left at 05:00 hours. Who has traveled further?? "
[option1]: the ranger" 
[option2]: the rustler" 
Response: Let's think step by step. [1]. Given that both the ranger and the rustler were riding horses at the same speed, the distance traveled is directly proportional to the time traveled. Distance = Speed * Time. [2].The rustler left at 01:00 and the ranger left at 05:00 hours. Since the rustler started 4 hours earlier, the rustler has traveled for 4 hours more compared to the rustler. [3]. As they were riding at the same speed, the rustler has traveled further in the given time period. [4]. Therefore, the answer is option 2."

[question]:{question}
[option1]:{option1}
[option2]:{option2}
Response: Let's think step by step.
"""

# entailment/contradiction (AWPNLI)
NLI2_AWPNLI = """
I will first raise two statements and then provide two options which are entailment and contradiction. 
The first statement is the given premise, the second statement is the hypothesis. You should determine if the hypothesis can be justifiably inferred to be true (option 1 : entailment) or false (option 2 : contradiction) base on the premise.
Please choose the correct answer after providing the inference process step by step, in the format of "the answer is option 1”. If calculation is involved, please provide the equations during the calculation process. Using numbers in '[]' to mark steps, like '[1]'.

For example:
[statement1]:There were 58.0 geese and 37.0 ducks in the marsh.
[statement2]:16.0 more geese were in the marsh
[option1]:Entailment
[option2]:contradiction
Response: Let's think step by step.\n [1]. Base on the premise, there were 58.0 geese and 37.0 ducks in the marsh, so there are 58.0 - 37.0 = 21.0 more geese were in the marsh.\n [2]. The hypothesis states that there are 16.0 more geese were in the marsh, which is contradict to statement2.\n [3]. Thus, the hypothesis is false, the answer is option 2.

[statement1]:{statement1}
[statement2]:{statement2}
[option1]:{option1}
[option2]:{option2}
Response: Let's think step by step.\n
"""

# entailment/neutral (NewsNLI)
NLI2_NewsNLI = """
I will first raise two statements and then provide two options which are entailment and neutral. 
The first statement is the given premise, the second statement is the hypothesis. You should determine if the hypothesis can be justifiably inferred to be true (option 1 : entailment) or cannot be determined (option 2 : neutral) base on the premise.
You should pay attention to additional information rather than shared information, especially paying attention to whether the numbers are reasonable and derived from the premise. If there is information that is not mentioned in the premise or cannot be directly inferred in the hypothesis, then the hypothesis cannot be determined.
Please choose the correct answer after providing the inference process step by step, in the format of "the answer is option 1”. If calculation is involved, please provide the equations during the calculation process. Using numbers in '[]' to mark steps, like '[1]'.

For Example:
[statement]: "The incident started after South Korean commandos boarded the Chinese ship ."
[statement2]: "South Korea has seized the ship and its nine sailors"
[option1]: Entailment
[option2]: neutral
Response: Let's think step by step.\n [1]. The premise is that "the incident started after South Korean commandos boarded the Chinese ship", the hypothesis is that "South Korea has seized the ship and its nine sailors".\n [2]. Base on the premise, we can infer that South Korea has seized the ship by "South Korean commandos boarded the Chinese ship" in the premise, but we cannot infer that the ship has nine sailors.\n [3]. Thus, the hypothesis cannot be determined, the answer is option 2.

[statement1]:{statement1}
[statement2]:{statement2}
[option1]:{option1}
[option2]:{option2}
Response: Let's think step by step.\n 
"""

# entailment/neutral (RTE)
NLI2_RTE = """
I will first raise two statements and then provide two options which are entailment and neutral. 
The first statement is the given premise, the second statement is the hypothesis. You should determine if the hypothesis can be justifiably inferred to be true (option 1 : entailment) or cannot be determined (option 2 : neutral) base on the premise.
You should pay attention to additional information rather than shared information, especially paying attention to whether the numbers are reasonable and derived from the premise. If there is information that is not mentioned in the premise or cannot be directly inferred in the hypothesis, then the hypothesis cannot be determined.
Please choose the correct answer after providing the inference process step by step, in the format of "the answer is option 1”. If calculation is involved, please provide the equations during the calculation process. Using numbers in '[]' to mark steps, like '[1]'.

For example:
[statement1]: "Prosecutions tended to be more aggressive and sentences tended to be longer for financial crimes with many victims and massive losses of money , as in the Minkow case ."
[statement2]: "Bilking a large number of people out of millions of dollars can lead to sentences that vary from ten to twenty years ."
[option1]: "Entailment", 
[option2]: "neutral".
Response: Let's think step by step.\n [1]. According to the premise, we can infer that financial crimes bilking a large number of people out of millions of dollars can lead to longer sentences.\n [2]But the information of "vary from ten to twenty years" in the hypothesis cannot be determined to be true since it's not mentioned in the premise.\n [3]. Therefore, the hypothesis cannt be determined, the answer is option 2.

[statement1]:{statement1}
[statement2]:{statement2}
[option1]:{option1}
[option2]:{option2}
Response: Let's think step by step.\n
"""


# entailment/neutral/contradiction (RedditNLI)
NLI3_RedditNLI = """
I will first raise two statements and then provide three options which are entailment, contradiction and neutral. The first statement is the given premise, the second statement is the hypothesis. You should determine if the hypothesis can be justifiably inferred to be true (option 1 : entailment), false (option 2 : contradiction) or cannot be determined (option 3 : neutral) base on the premise.
Please choose the correct answer after providing the inference process step by step, in the format of "the answer is option 1”. If calculation is involved, please provide the equations during the calculation process. Using numbers in '[]' to mark steps, like '[1]'.

For example:
[statement1]: "At DAVOS, Financial Billionaire Schwartzman, who took home over $600 million dollars last year, says he doesn't understand why americans are angry."
[statement2]: "At Davos, Wall Street Billionaire Steven Schwartzfman, who takes home every year around $700 million dollars, says he doesn't understand why americans are angry."
[option1]: "Entailment", 
[option2]: "contradiction".
[option3]: "neutral".
Response: Let's think step by step.\n [1]. We are given the premise that Schwartzman earns over $600 million dollars last year, the hypothesis states that Schwartzfman earns around $700 million dollars every year.\n [2]. Although $700 million is greater than $ 600 million, "over $600 million" is a large range and we cannot be sure if it is around $700 million. And we also cannot infer whether he earns so much money every year.\n [3]. Thus, the hypothesis cannot be determined, the answer is option 3.

[statement1]:{statement1}
[statement2]:{statement2}
[option1]:{option1}
[option2]:{option2}
[option3]:{option3}
Response: Let's think step by step.\n
"""


# entailment/neutral/contradiction (stress)
NLI3_stress = """
I will first raise two statements and then provide three options which are entailment, contradiction and neutral. 
The first statement is the given premise, the second statement is the hypothesis. You should determine if the hypothesis can be justifiably inferred to be true (option 1 : entailment), false (option 2 : contradiction) or cannot be determined (option 3 : neutral) base on the premise.
You should especially pay attention to whether the numbers are reasonable and derived from the premise. If there is information that is cannot be directly inferred in the hypothesis, then the hypothesis cannot be determined.
Please choose the correct answer after providing the inference process step by step, in the format of "the answer is option 1”. Using numbers in '[]' to mark steps, like '[1]'. Do not repeat statements or examples.

For example:
[statement1]: "In the city of San Durango , more than 60 people own cats , dogs , or rabbits", 
[statement2]: "In the city of San Durango , around 70 people own cats , dogs , or rabbits"
[option1]:Entailment
[option2]:contradiction
[option3]:neutral
Response: Let's think step by step.\n [1]. According to the premise "more than 60 people own cats , dogs , or rabbits", the hypothesis is not contradicted to the premise.\n [2]. Although 70 is greater than 60, "more than 60 people" is a large range and we cannot be sure if it is around 70 people.\n [3]. Thus, the hypothesis cannot be determined, the answer is option 3.

[statement1]:{statement1}
[statement2]:{statement2}
[option1]:{option1}
[option2]:{option2}
[option3]:{option3}
Response: Let's think step by step.\n
"""

prompt_QP_headline = """
Below is a news headline with a number masked, predict the correct magnitude of the masked numeral after providing the inference process step by step, in the format of "the correct magnitude is 1”. Using numbers in '[]' to mark steps, like '[1]'. The magnitude of decimals is 0 and magnitudes greater than 6 is represented as magnitude 7.'.

For example:
[headline]:"100 Most Anticipated books releasing in [Num]"
Response: Let's think step by step.\n [1]. The problem involves predicting the correct magnitude of a masked numeral in a news headline. We are given options from 0 to 7 for the magnitude, decimals represented as magnitude 0, and magnitudes greater than 6 represented as magnitude 7.\n [2]. By the context of the problem, we can determine that the headline is about 100 Most Anticipated books releasing in a certain time period. Based on the number of books being 100, it can be inferred that it is a certain year.\n [3]. Since the magnitude of year numbers is usually 4, the magnitude of the masked numeral should be 4.\n [4]. Thus, the correct magnitude is 4.\n #### 4"

[headline]:{question}
Response: Let's think step by step.
"""
prompt_QP_comment = """
Below is a comment with a number masked, predict the correct magnitude of the masked numeral after providing the inference process step by step, in the format of "the magnitude is 1”. Using numbers in '[]' to mark steps, like '[1]'. The magnitude of decimals is 0 and magnitudes greater than 6 is represented as magnitude 7.'.

For example:
[headline]:"BRIEF-Henan Shuanghui Investment's H1 net profit up [Num] pct"
Response: Let's think step by step.\n [1]. The problem involves predicting the correct magnitude of a masked numeral in a comment. We are given options from 0 to 8 for the magnitude, decimals represented as magnitude 0, and magnitudes greater than 6 represented as magnitude 7.\n [2]. By the context of the problem, we can determine that the comment is about the profit growth rate of a investment company. And the masked numeral is a percentage number.\n [3]. Since the magnitude of a growth percentage is usually between 0 and 3, the magnitude of the masked numeral could be 1. \n [4]. Thus, the correct magnitude is 1.\n #### 1"

[headline]:{question}
Response: Let's think step by step. 
"""


ROLE_en = 'Do quantitative reasoning as if you were an expert in the field.'



def fetch_existed_inputs(output_path):
    with open(output_path, 'r', errors='ignore') as fread:
        num = 0
        data = []
        for line in fread:
            try:
                json_data = json.loads(line)
                data.append(json_data)
            except:
                num += 1
  
    have_existed_df = pd.DataFrame(data)

    if(have_existed_df.empty):
        hava_existed_id=[]
    else:
        hava_existed_id = have_existed_df['id'].values
    # import IPython;IPython.embed();exit()
    return hava_existed_id

def fetch_data(file_path, output_path, option_num=2, task='default'):
    data_df = pd.read_json(file_path, encoding='utf-8',encoding_errors='ignore')
    print(f'需要请求的数据有{data_df.shape}')
    # import IPython;IPython.embed();exit()
    
    with open(output_path, mode='a+', encoding='utf-8') as fout:
        hava_existed_id = fetch_existed_inputs(output_path)
        if len(hava_existed_id)>0:
            id_last = hava_existed_id[-1]
            data_df = data_df[id_last+1:]

        for idx, row in  tqdm(data_df.iterrows(), total=len(data_df)):
            id_new = row['id']
            result = dict()
           
            statement1 = row['statement1']
            statement2 = row['statement2']
            result['statement1'] = row['statement1']
            result['statement2'] = row['statement2']

            option1 = row['option1']
            option2 = row['option2']
            result['option1'] = row['option1']
            result['option2'] = row['option2']

            # option3 = row['option3']
            # result['option3'] = row['option3']


            prompt = NLI2_AWPNLI.format(statement1=statement1, statement2=statement2, option1 = option1, option2 = option2)

            # prompt = NLI2_NewsNLI.format(statement1=statement1, statement2=statement2, option1 = option1, option2 = option2)

            # prompt = NLI3_stress.format(statement1=statement1, statement2=statement2, option1 = option1, option2 = option2, option3 = option3)

            # question = row['question']
            # prompt = prompt_QP_comment.format(question=question)
            # result['question'] = row['question']

            # question = row['question']
            # prompt = QA.format(question=question, option1 = option1, option2 = option2)
            # result['question'] = row['question']


            response = fetch_response(prompt)
            while response == "Error":
                response = fetch_response(prompt)
            


            # result['option4'] = row['option4']

            result['generated_text'] = response
            result['label'] = row["label"]
            result['id'] = row['id']

            json.dump(result, fout, ensure_ascii=False)
            fout.write('\n')
            print(f'---------------------idx：{idx}---------------------')
            print(result)
            sys.stdout.flush()



def fetch_response(content):
    response = openai.ChatCompletion.create(
                  model="gpt-3.5-turbo",
                  messages=[
                        { "role":"system", "content": ROLE_en},
                        { "role": "user", "content": content},
                    ],
                  temperature=0.8,
                )
    return response.choices[0].message.content


def jsonl_to_json(jsonl_file, json_file):
    with open(jsonl_file, 'r', errors='ignore') as f:
        json_data = []
        for line in f:
            try:
                dic = json.loads(line)
                json_data.append(dic)
            except:
                print(line)

    with open(json_file, 'w', errors='ignore') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':

    input_path = 'D:/BIT/NLP/SemEval2024/SFT-data/gpt_inputs/300_AWPNLI.json'
    output_path = 'D:/BIT/NLP/SemEval2024/SFT-data/gpt_outputs/_AWPNLI.jsonl'
    json_file = 'D:/BIT/NLP/SemEval2024/SFT-data/gpt_outputs/300_AWPNLI.json'

    fetch_data(input_path, output_path)
    jsonl_to_json(output_path,json_file)
