import re
import json

in_path = "bloomz/QA.jsonl"
out_path = "bloomz/QQA.json"

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

with open(in_path, 'r', errors='ignore') as file:
    # datasets = json.load(file)
    lines = file.readlines()
results = []
for line in lines:
    data = json.loads(line)

    answer = data["bloomz_response"]
    match = re.search(r'he answer is option.*(\d+)', answer, re.IGNORECASE)
    match2 = re.search(r'option\s+(\d+) is the correct answer', answer, re.IGNORECASE)
    match3 = re.search(r'correct answer is option.*(\d+)', answer, re.IGNORECASE)
    match4 = re.search(r'the answer is ([A-Za-z]+)', answer, re.IGNORECASE)
    match5 = re.search(r'choose option.*(\d+)', answer, re.IGNORECASE)
    match6 = re.search(r'\[option.*(\d+)\]', answer, re.IGNORECASE)

    if match:
        result = match.group(1)
        data["result"] = result
    elif match2:
        result = match2.group(1)
        data["result"] = result
    elif match3:
        result = match3.group(1)
        data["result"] = result
    elif match4:
        result = match4.group(1)
        if "neutral" in result:
            data['result'] = '3'
        elif "contradiction" in result:
            data['result'] = '2'
        elif "entail" in result:
            data['result'] = '1'
       
    elif match5:
        result = match5.group(1)
        data["result"] = result
    elif match6:
        result = match6.group(1)
        data["result"] = result
    else:
        # data["result"] = 'Error'
        step_ls = extract_steps(answer)
        if step_ls:
            if data['option1'][:-1] in step_ls[-1]['text']:
                data["result"] = '1'
            elif data['option2'][:-1] in step_ls[-1]['text']:
                data["result"] = '2'
            else:
                # print(step_ls[-1]['text'])
                # print(data['option2'])
                print(data['id'])
                # break
        else:
            print(data['id'])
    
    results.append(data)


with open(out_path, 'w+', errors='ignore') as file:
    json.dump(results,file,indent=4)