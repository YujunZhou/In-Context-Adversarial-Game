import pickle
import sys, os
from util import OpenAIModel, get_embedding, load_model_and_tokenizer, llm_generate
from agents import TrainContralAgent
import openai
from openai import OpenAI
import numpy as np
import pandas as pd
import copy
from prompts import *
from fastchat.model.model_adapter import get_conversation_template
import random

sys.path.append('..')
root = '../root/'
os.environ['OPENAI_API_KEY'] = ''

client = OpenAI()


def jailbreak_prompt_improve(insights, attacker, test_prompts, atk_fail_idx, atk_suc_idx, n_trial=3):
    sum_trial = 0
    success_num = 0
    for trial in range(1, 1 + n_trial):
        attack_update_prompts = []
        for i in range(len(atk_fail_idx)):
            successful_trial_idx = atk_suc_idx[i]
            successful_trial = attacker.cur_prompts[atk_suc_idx[i]]
            insights_candidates = copy.deepcopy(attacker.attack_insights_used_for_prompts[successful_trial_idx])
            if len(insights_candidates) == 0:
                insight_idx = np.random.choice(list(insights.keys()))
                insight = insights[insight_idx]
            else:
                try:
                    insight_idx = np.random.choice(insights_candidates)
                except:
                    insight_idx = 0
                insight = insights[insight_idx]
            if len(insights_candidates) != 0:
                insights_candidates.remove(insight_idx)
            attack_update_prompts.append(
                attack_update_prompt.format(previous_trial=test_prompts[atk_fail_idx[i]], rule=insight,
                                            suffix=attacker.dan_suffix, successful_prompt=successful_trial))
        new_prompts = []
        for i in range(len(atk_fail_idx)):
            new_prompt = attacker.attack_llm(attack_update_prompts[i])
            new_prompts.append(new_prompt)
        attack_refine_prompts = []
        for i in range(len(atk_fail_idx)):
            # sample one sentence from Character_Perturbation_Instructions
            strategy1 = random.choice(Character_Perturbation_Instructions)
            strategy2 = random.choice(Word_Perturbation_Instructions)
            strategy3 = random.choice(SentenceLevel_Perturbation_Instructions)
            attack_refine_prompts.append(
                attack_refine_prompt.format(failed_jailbreak_prompt=new_prompts[i], strategy1=strategy1,
                                            strategy2=strategy2, strategy3=strategy3, suffix=attacker.dan_suffix))
        refined_prompts = []
        for i in range(len(atk_fail_idx)):
            new_prompt = attacker.attack_llm(attack_refine_prompts[i])
            refined_prompts.append(new_prompt)

        inputs = []
        outputs = []
        for i in range(len(atk_fail_idx)):
            if '[INSERT PROMPT HERE]' in refined_prompts[i]:
                input = refined_prompts[i].replace('[INSERT PROMPT HERE]', questions_inputs[i])
            else:
                input = refined_prompts[i] + ' ' + questions_inputs[i]
            inputs.append(input)
        if attacker.white_box:
            conv_template = get_conversation_template(attacker.model_path)
            outputs = llm_generate(inputs, conv_template, attacker.victim_llm,
                                   attacker.tokenizer, batch_size=32, random_sample=False, max_new_tokens=128)
        else:
            for i in range(len(inputs)):
                outputs.append(victim_llm(inputs[i]))
            outputs = np.array(outputs)
        evaluate_prompts = []
        for i in range(len(atk_fail_idx)):
            evaluate_prompts.append(question_evaluate_prompt.format(harmful_question=questions_inputs[i],
                                                                 output_text=outputs[i]))
        jailbroken = attacker.llama3_evaluate(evaluate_prompts, del_model=attacker.white_box)
        jailbroken = np.array(jailbroken)
        del_idx = []
        for i in range(len(atk_fail_idx)):
            if jailbroken[i]:
                test_prompts[atk_fail_idx[i]] = refined_prompts[i]
                success_num += 1
                print(atk_fail_idx[i], 'success')
                # remove the idx from atk_fail_idx
                del_idx.append(i)
        atk_fail_idx = np.delete(atk_fail_idx, del_idx)
        atk_suc_idx = np.delete(atk_suc_idx, del_idx)
    return test_prompts

data = pd.read_csv('../data/PAIR/harmful_behaviors_custom.csv')
test_prompts = pd.read_csv('../data/Self-Reminder-Data/data/jailbreak_prompts.csv')
test_prompts = test_prompts['prompt']

test_prompts = np.array(test_prompts)
questions = data['goal']
questions = np.array(questions)[[4, 6, 20, 21, 49]]
question_idxes = []
for idx in range(5):
    question_idxes += [idx] * 75
questions_inputs = questions[question_idxes]
test_samples = []
for goal in questions:
    for prompt in test_prompts:
        if '[INSERT PROMPT HERE]' in prompt:
            input = prompt.replace('[INSERT PROMPT HERE]', goal)
        else:
            input = prompt + goal
        test_samples.append(input)



attack_llm = OpenAIModel("gpt-3.5-turbo-0125", attack=True).prompt
defense_llm = OpenAIModel("gpt-3.5-turbo-0125").prompt
evaluate = OpenAIModel("gpt-3.5-turbo-0125").prompt


for victim_llm_name in ['mistral']:
    if victim_llm_name == 'gpt-3.5-turbo-0125':
        victim_llm = OpenAIModel("gpt-3.5-turbo-0125").prompt
        attacker = pickle.load(open('../Logs/gpt-3.5-turbo-0125_improved_ref/attack_1.pkl', 'rb'))
    else:
        attacker = pickle.load(open(f'../Logs/{victim_llm_name}_improved_ref/attack_1.pkl', 'rb'))
        victim_llm = attacker.victim_llm
        tokenizer = attacker.tokenizer
        model_path = attacker.model_path

    attacker.llama3_instruct_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
    insights = attacker.attack_insights
    prompts = attacker.cur_prompts
    harmful_flags = attacker.cur_harmful_flags
    atk_suc_idx = np.sum(harmful_flags, axis=1) == 1
    atk_suc_idx = np.where(atk_suc_idx)[0]

    atk_suc_prompts = prompts[atk_suc_idx]
    answers = []
    if victim_llm_name == 'gpt-3.5-turbo-0125':
        for i in range(len(test_samples)):
            output = victim_llm(test_samples[i])
            answers.append(output)

        prompt_inputs = [question_evaluate_prompt.format(harmful_question=question, output_text=answer)
                         for question, answer in zip(questions_inputs, answers)]
        jailbroken = attacker.llama3_evaluate(prompt_inputs, del_model=False)
        cur_harmful_flags = np.array(jailbroken)
    else:
        cur_inputs = test_samples
        conv_template = get_conversation_template(model_path)
        # conv_template.system_message = attacker.cur_defense_prompt
        gen_strs = llm_generate(cur_inputs, conv_template, victim_llm, tokenizer, batch_size=32,
                                random_sample=False, max_new_tokens=256)
        cur_answers = np.array(gen_strs)
        test_answer_prompts = [question_evaluate_prompt.format(harmful_question=questions_inputs[i], output_text=cur_answers[i])
                               for i in range(len(questions_inputs))]
        jailbroken = attacker.llama3_evaluate(test_answer_prompts, del_model=True)
        cur_harmful_flags = np.array(jailbroken)
    atk_fail_idx = np.where(cur_harmful_flags == 0)[0]
    test_prompts = list(test_prompts)
    test_prompts = test_prompts * 5
    test_prompts = np.array(test_prompts)
    print(len(test_prompts))
    atk_fail_prompts = test_prompts[atk_fail_idx]
    test_samples_embed = np.array(get_embedding(test_samples))
    if len(atk_suc_idx) >= 5:
        atk_suc_embed = np.array(get_embedding(atk_suc_prompts.tolist()))
        closest_k_suc_for_fail = attacker.closest_retrieve(test_samples_embed, atk_suc_embed, k=5)
        random_choice = np.random.choice(5, len(test_samples))
        pair_suc_prompt_for_fail = atk_suc_prompts[
            closest_k_suc_for_fail[np.arange(len(test_samples)), random_choice]]
        pair_suc_idx = atk_suc_idx[closest_k_suc_for_fail[np.arange(len(test_samples)), random_choice]]
    elif len(atk_suc_idx) > 0:
        random_choice = np.random.choice(len(atk_suc_prompts), len(test_samples))
        pair_suc_prompt_for_fail = atk_suc_prompts[random_choice]
        pair_suc_idx = atk_suc_idx[random_choice]
    else:
        random_choice = np.random.choice(len(prompts), len(test_samples))
        pair_suc_prompt_for_fail = prompts[random_choice]
        pair_suc_idx = random_choice

    new_prompts = jailbreak_prompt_improve(insights, attacker, test_prompts, atk_fail_idx, pair_suc_idx)
    pickle.dump(new_prompts, open('../data/%s_Our_test_set.pkl' % victim_llm_name, 'wb'))



