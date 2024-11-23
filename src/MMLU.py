import pandas as pd
import pickle
import os
from openai import OpenAI
from langchain.prompts import PromptTemplate
import json
from fastchat.model.model_adapter import get_conversation_template
from util import *
import numpy as np

def llm_generate_MMLU(inputs, conv_template, model, tokenizer, batch_size=6, random_sample=False, max_new_tokens=512):
    input_ids_list = []
    for i in range(len(inputs)):
        conv_template.append_message(conv_template.roles[0], inputs[i])
        conv_template.append_message(conv_template.roles[1], None)
        prompt = conv_template.get_prompt()
        encoding = tokenizer(prompt)
        toks = encoding.input_ids
        input_ids = torch.tensor(toks).to(model.device)
        input_ids_list.append(input_ids)
        conv_template.messages = []
    pad_tok = tokenizer.pad_token_id
    max_input_length = max([ids.size(0) for ids in input_ids_list])
    # Pad each input_ids tensor to the maximum length
    padded_input_ids_list = []
    for ids in input_ids_list:
        pad_length = max_input_length - ids.size(0)
        padded_ids = torch.cat([torch.full((pad_length,), pad_tok, device=model.device), ids], dim=0)
        padded_input_ids_list.append(padded_ids)
    input_ids_tensor = torch.stack(padded_input_ids_list, dim=0)
    attn_mask = (input_ids_tensor != pad_tok).type(input_ids_tensor.dtype)
    generation_config = model.generation_config
    generation_config.max_new_tokens = max_new_tokens
    if random_sample:
        generation_config.do_sample = True
        generation_config.temperature = 0.6
        generation_config.top_p = 0.9
    else:
        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
    flag = False
    while not flag:
        try:
            output_ids_new = []
            for i in range(0, len(input_ids_tensor), batch_size):
                input_ids_tensor_batch = input_ids_tensor[i:i + batch_size]
                attn_mask_batch = attn_mask[i:i + batch_size]
                output_ids_batch = model.generate(input_ids_tensor_batch,
                                                  attention_mask=attn_mask_batch,
                                                  generation_config=generation_config,
                                                  pad_token_id=tokenizer.pad_token_id)

                for j in range(len(output_ids_batch)):
                    output_ids_new.append(output_ids_batch[j][max_input_length:])
            flag = True
        # except cuda out of memory error
        except torch.cuda.OutOfMemoryError:
            batch_size = batch_size // 2
    gen_strs = [tokenizer.decode(output_id, skip_special_tokens=True) for output_id in output_ids_new]

    second_prompt = 'Given all of the above, the answer of the question is:'
    input_ids_list = []
    for i in range(len(inputs)):
        conv_template.append_message(conv_template.roles[0], inputs[i])
        conv_template.append_message(conv_template.roles[1], gen_strs[i])
        conv_template.append_message(conv_template.roles[0], second_prompt)
        conv_template.append_message(conv_template.roles[1], None)
        prompt = conv_template.get_prompt()
        encoding = tokenizer(prompt)
        toks = encoding.input_ids
        input_ids = torch.tensor(toks).to(model.device)
        input_ids_list.append(input_ids)
        conv_template.messages = []
    pad_tok = tokenizer.pad_token_id
    max_input_length = max([ids.size(0) for ids in input_ids_list])
    # Pad each input_ids tensor to the maximum length
    padded_input_ids_list = []
    for ids in input_ids_list:
        pad_length = max_input_length - ids.size(0)
        padded_ids = torch.cat([torch.full((pad_length,), pad_tok, device=model.device), ids], dim=0)
        padded_input_ids_list.append(padded_ids)
    input_ids_tensor = torch.stack(padded_input_ids_list, dim=0)
    attn_mask = (input_ids_tensor != pad_tok).type(input_ids_tensor.dtype)
    generation_config = model.generation_config
    generation_config.max_new_tokens = max_new_tokens
    if random_sample:
        generation_config.do_sample = True
        generation_config.temperature = 0.6
        generation_config.top_p = 0.9
    else:
        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
    flag = False
    while not flag:
        try:
            output_ids_new = []
            for i in range(0, len(input_ids_tensor), batch_size):
                input_ids_tensor_batch = input_ids_tensor[i:i + batch_size]
                attn_mask_batch = attn_mask[i:i + batch_size]
                output_ids_batch = model.generate(input_ids_tensor_batch,
                                                  attention_mask=attn_mask_batch,
                                                  generation_config=generation_config,
                                                  pad_token_id=tokenizer.pad_token_id)

                for j in range(len(output_ids_batch)):
                    output_ids_new.append(output_ids_batch[j][max_input_length:])
            flag = True
        # except cuda out of memory error
        except torch.cuda.OutOfMemoryError:
            batch_size = batch_size // 2
    gen_strs = [tokenizer.decode(output_id, skip_special_tokens=True) for output_id in output_ids_new]

    return gen_strs


MMLU_prompts = pickle.load(open('../data/MMLU_prompts.pkl', 'rb'))
MMLU_system_prompts = pickle.load(open('../data/MMLU_system_prompts.pkl', 'rb'))
MMLU_change_file_idxes = pickle.load(open('../data/MMLU_change_file_idxes.pkl', 'rb'))
print(MMLU_prompts[0])
ground_truths = pickle.load(open('../data/MMLU_answers.pkl', 'rb'))
print(len(ground_truths))

model_path_dicts = {"llama2": "meta-llama/Llama-2-7b-chat-hf", "vicuna": "lmsys/vicuna-7b-v1.5",
                    'llama3': 'meta-llama/Meta-Llama-3-8B', 'mistral': 'mistralai/Mistral-7B-Instruct-v0.3',
                    'llama3-instruct': 'meta-llama/Meta-Llama-3-8B-Instruct'}

os.environ['OPENAI_API_KEY'] = ''
client = OpenAI()
for victim_llm_name in ['gpt-3.5-turbo-0125']:
    print(victim_llm_name)
    os.makedirs('../Logs/%s' % victim_llm_name, exist_ok=True)
    log_f = open('../Logs/%s/result.txt' % victim_llm_name, 'a')
    print(victim_llm_name, file=log_f, flush=True)
    ICAG_0 = json.load(open('../data/ICAG_prompts_iter.json'))[victim_llm_name]['0']
    ICAG_1 = json.load(open('../data/ICAG_prompts_iter.json'))[victim_llm_name]['1']
    ICAG_5 = json.load(open('../data/ICAG_prompts_iter.json'))[victim_llm_name]['5']
    ICAG_10 = json.load(open('../data/ICAG_prompts_iter.json'))[victim_llm_name]['10']
    if 'gpt' not in victim_llm_name:
        model_path = model_path_dicts[victim_llm_name]
        victim_llm, tokenizer = load_model_and_tokenizer(model_path)
    test_sets = [MMLU_prompts]
    test_sets_names = ['MMLU']
    system_prompts = [ICAG_5]
    record = np.zeros((len(test_sets), len(system_prompts)))
    for i, attack in enumerate(test_sets):
        print(test_sets_names[i], file=log_f, flush=True)
        for j, prompt in enumerate(system_prompts):
            outputs = []
            answers = []
            success = 0
            if 'gpt-3.5-turbo-0125' in victim_llm_name:
                for k, sample in enumerate(attack):
                    analysis = client.chat.completions.create(
                        model='gpt-3.5-turbo-0125',
                        messages=[
                            {"role": "system", "content": prompt + MMLU_system_prompts[k]},
                            {"role": "user", "content": sample}
                        ],
                        max_tokens=256,
                        temperature=0.6
                    ).choices[0].message.content
                    output = client.chat.completions.create(
                        model='gpt-3.5-turbo-0125',
                        messages=[
                            {"role": "system", "content": prompt + MMLU_system_prompts[k]},
                            {"role": "user", "content": sample},
                            {"role": "assistant", "content": analysis},
                            {"role": "user", "content": "Given all of the above, the answer of the question is:"}
                        ],
                        max_tokens=256,
                        temperature=0.6
                    ).choices[0].message.content
                    answer = client.chat.completions.create(
                        model='gpt-3.5-turbo-0125',
                        messages=[
                            {"role": "system", "content": '''You are a system that processes multiple-choice question solutions. You will be given the solution process and the final answer. Your task is to only output the final answer, which will be one of the following options: "A", "B", "C", or "D". Ignore all other information and text. Ensure that the output is exactly one of these letters without any additional text or characters.'''},
                            {"role": "user", "content": output},
                        ],
                        max_tokens=256,
                        temperature=0
                    ).choices[0].message.content
                    answer = answer[0].upper()
                    answers.append(answer)
                    success += 1 if answer == ground_truths[k] else 0
                    if k % 100 == 99:
                        print(k, "accuracy:", success / (k + 1))
            else:
                idx_start = 0
                # flag = False
                for idx in MMLU_change_file_idxes:
                    # if victim_llm_name == 'vicuna' and not flag:
                    #     if idx <= 7384:
                    #         continue
                    #     idx_start = 7384
                    #     success = 2133
                    #     flag = True
                    #     answers = ['A'] * 7384
                    print(idx)
                    conv_template = get_conversation_template(model_path)
                    conv_template.system_message = prompt + MMLU_system_prompts[idx_start]
                    outputs_temp = llm_generate_MMLU(attack[idx_start:idx], conv_template, victim_llm, tokenizer, batch_size=32, random_sample=True)
                    if victim_llm_name == 'vicuna' or victim_llm_name == 'mistral':
                        del victim_llm, tokenizer
                        llama3_model, llama3_tokenizer = load_model_and_tokenizer(model_path_dicts['llama3-instruct'])
                        conv_template = get_conversation_template(model_path_dicts['llama3-instruct'])
                        conv_template.system_message = '''You are a system that processes multiple-choice question solutions. You will be given the solution process with the final answer. Your task is to only output the final answer, which will be one of the following options: "A", "B", "C", or "D". Ignore all other information and text. Ensure that the output is exactly one of these letters without any additional text or characters.'''
                        # judge_prompts = ['''Here is the solution process with the final answer: {output}
                        #                     Please only output the final answer, which will be one of the following options: "A", "B", "C", or "D".'''
                        #                  for output in outputs_temp]
                        answers_temp = llm_generate(outputs_temp, conv_template, llama3_model, llama3_tokenizer, batch_size=128, random_sample=False, max_new_tokens=128)
                        del llama3_model, llama3_tokenizer
                        victim_llm, tokenizer = load_model_and_tokenizer(model_path)
                    else:
                        conv_template.system_message = '''You are a system that processes multiple-choice question solutions. You will be given the solution process with the final answer. Your task is to only output the final answer, which will be one of the following options: "A", "B", "C", or "D". Ignore all other information and text. Ensure that the output is exactly one of these letters without any additional text or characters.'''
                        answers_temp = llm_generate(outputs_temp, conv_template, victim_llm, tokenizer, batch_size=128, random_sample=False, max_new_tokens=128)
                    outputs += outputs_temp
                    for n in range(len(answers_temp)):
                        try:
                            answers.append(answers_temp[n][0].upper())
                        except:
                            answers.append('E')
                    success += sum([1 if answers[idx_start:idx][n] == ground_truths[idx_start:idx][n] else 0 for n in range(len(answers_temp))])
                    if idx_start == 0:
                        print(outputs_temp)
                        print(answers_temp)
                    idx_start = idx
                    print(idx_start, "accuracy:", success / idx_start)

            record[i][j] = success / len(attack)
            # round record to 4 decimal places
            record[i][j] = round(record[i][j], 4)
            print(record[i])
        print(str(record[i]), file=log_f, flush=True)
    print(victim_llm_name)
    print(record)
    print('-----------------------------------')

    log_f.write(str(record) + '\n')



