import pickle
import sys, os
import numpy as np
from util import OpenAIModel, get_embedding
#from Llama2_based_model_util import load_Llama2_based_model

sys.path.append('..')
root = '../root/'
os.environ['OPENAI_API_KEY'] = ''

import pandas as pd
from agents import TrainContralAgent
import openai
from openai import OpenAI
import argparse

parser = argparse.ArgumentParser(description='train_ICAG')
parser.add_argument('--victim_llm', default='vicuna', type=str, help='vicuna, llama3, llama3_instruct, mistral, gpt-3.5-turbo-0125')
parser.add_argument('--attack_llm', default='gpt-3.5-turbo-0125', type=str, help='gpt-3.5-turbo-0125, llama3_instruct')
parser.add_argument('--defense_llm', default='gpt-3.5-turbo-0125', type=str, help='gpt-3.5-turbo-0125, llama3_instruct')
parser.add_argument('--mode', default='improved_ref', type=str, help='improved_ref, ref, self_reminder, wout')
parser.add_argument('--path', default=None, type=str, help='loading path of the checkpoints')
parser.add_argument('--n', default=10, type=int, help='num of iterations')
parser.add_argument('--question_idx', default=0, type=int, help='index of the question')
args = parser.parse_args()

# load ../data_main.json
questions = pd.read_csv('./data/PAIR/harmful_behaviors_custom.csv')
subquestions = questions['goal']
train_questions = list([subquestions[1], subquestions[40], subquestions[24]])

ori_jail_prompts = pd.read_csv('./data/Self-Reminder-Data/data/jailbreak_prompts_train.csv')
ori_jail_prompts = ori_jail_prompts['prompt']
train_prompts = list(ori_jail_prompts)

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()




victim_llm_name = args.victim_llm

attack_llm = OpenAIModel("gpt-3.5-turbo-0125", attack=True).prompt
defense_llm = OpenAIModel("gpt-3.5-turbo-0125").prompt
evaluate = OpenAIModel("gpt-3.5-turbo-0125", max_tokens=100).prompt
victim_llm = OpenAIModel("gpt-3.5-turbo-0125", max_tokens=256).prompt


def train(load_path=None, mode='improved_ref'):
    if not load_path:
        train_control_agent = TrainContralAgent(questions=train_questions, prompts=train_prompts,
                                                victim_llm_name=victim_llm_name, attack_llm_name=args.attack_llm,
                                                get_embedding=get_embedding, victim_llm=victim_llm, attack_llm=attack_llm,
                                                defense_llm=defense_llm, evaluator=evaluate,
                                                defense_llm_name=args.defense_llm, mode=mode, question_idx=args.question_idx)

        train_control_agent.update_all()
        train_control_agent.defense()
        for iter in range(args.n):
            train_control_agent.update_all()
            train_control_agent.attack()
            train_control_agent.defense()
        train_control_agent.update_all()
    else:
        iteration = load_path.split('/')[-1].split('_')[-1].split('.')[0]
        iteration = int(iteration)
        train_control_agent = pickle.load(open(load_path, 'rb'))
        print('JSR:', train_control_agent.JSR)
        # print('val_JSR:', train_control_agent.val_JSR)
        # print('test_JSR:', train_control_agent.test_JSR)
        # train_control_agent.defense_llm = OpenAIModel("gpt-4-0125-preview").prompt
        for iter in range(iteration, args.n):
            print('**********************', iter, '**********************')

            if iter == iteration:
                if 'defense_reflection' in load_path:
                    train_control_agent.defense_from_reflections()
                elif 'defense_insights' in load_path:
                    train_control_agent.defense_from_insights()
                elif 'defense' in load_path:
                    train_control_agent.attack()
                    train_control_agent.defense()
                else:
                    train_control_agent.defense()
            train_control_agent.update_all()
            train_control_agent.attack()
            train_control_agent.defense()
        train_control_agent.update_all()

def train_first(mode='ref'):

    for victim_llm_name in ['llama3_instruct', 'mistral', 'vicuna', 'gpt-3.5-turbo-0125']:
        train_control_agent = TrainContralAgent(questions=train_questions, prompts=train_prompts,
                                                victim_llm_name=victim_llm_name, attack_llm_name='gpt-3.5-turbo-0125',
                                                get_embedding=get_embedding, victim_llm=victim_llm, attack_llm=attack_llm,
                                                defense_llm=defense_llm, evaluator=evaluate,
                                                defense_llm_name='gpt-3.5-turbo-0125', mode=mode)


        train_control_agent.update_all()
        train_control_agent.defense()
        train_control_agent.update_all()


train(load_path=args.path, mode=args.mode)
# train_first(load_path=args.path, mode=args.mode)