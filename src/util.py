import openai
from openai import OpenAI
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class OpenAIModel:
    def __init__(self, model_name, max_tokens=None, system_prompt='You are a helpful assistant.', attack=False, temperature=0.0):

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.attack = attack
        self.temperature = temperature

    def prompt(self, x):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI()
        if not self.attack:
            return client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": x}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            ).choices[0].message.content
        else:
            return client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": x}
                ],
                max_tokens=self.max_tokens,
                temperature=1.0
            ).choices[0].message.content


def get_embedding(texts):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI()
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [response.data[i].embedding for i in range(len(texts))]


def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        **kwargs
    ).to(device).eval()

    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False,
        padding_side='left',
    )
    tokenizer.pad_token = tokenizer.unk_token
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return model, tokenizer


def llm_generate(inputs, conv_template, model, tokenizer, batch_size=6, random_sample=False, max_new_tokens=64):
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
            print('Out of memory error, batch size %d is too large', batch_size)
            batch_size = batch_size // 2
    gen_strs = [tokenizer.decode(output_id, skip_special_tokens=True) for output_id in output_ids_new]
    return gen_strs


def llm_generate_IAPrompt(inputs, conv_template, model, tokenizer, batch_size=6, random_sample=False, max_new_tokens=64):
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

    second_prompt = 'Knowing the essential intention, now respond to the aforementioned query directly. Make sure your response adheres strictly to our policy and standards and excludes any unsafe or unethical information.'
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