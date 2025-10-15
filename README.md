# In-Context Adversarial Game (ICAG)

Official implementation of "Defending Jailbreak Prompts via In-Context Adversarial Game (ICAG)". ICAG performs an in-context adversarial game between attack and defense agents to iteratively produce and compress defense rules without fine-tuning. It significantly reduces jailbreak success rates and transfers well across LLMs.

- Paper: [`arXiv:2402.13148`](https://arxiv.org/abs/2402.13148)


## Repository Layout

- `src/`
  - `train.py`: ICAG training entry (iterative attack/defense; produce system prompts)
  - `agents.py`: attack/defense/evaluator agents
  - `prompts.py`, `prompts_baseline.py`: prompt templates
  - `util.py`: OpenAI + HF utilities for generation/eval
  - Evaluation scripts:
    - `test_baseline.py`: AdvBench-style evaluation of system prompts
    - `test_transfer.py`: transfer evaluation across models
    - `MMLU.py`: ability retention on MMLU
    - `xstest.py`: over-defense/refusal analysis on XSTest
    - `ICAG_attack_data.py`: generate/refine attack prompts for analysis
- `data/`
  - Attacks: `PAIR_*.pkl`, `AutoDAN_*.pkl`, `Self-Reminder-Data/*`, `MultiJail.csv`, `Xstest/*`
  - ICAG iterations/ablations: `ICAG_prompts_iter.json`, `ICAG_ablation_prompts.json`
  - MMLU: `MMLU_prompts.pkl`, `MMLU_answers.pkl`, `MMLU_system_prompts.pkl`


## Requirements & Setup

Install:

```bash
pip install -U openai transformers torch accelerate sentencepiece langchain fastchat
```

Environment:

```bash
export OPENAI_API_KEY=YOUR_API_KEY
```

Notes:
- Remote models use OpenAI Chat Completions (e.g., `gpt-3.5-turbo-0125`).
- Local models use HF checkpoints (pre-mapped in scripts):
  - `vicuna`: `lmsys/vicuna-7b-v1.5`
  - `mistral`: `mistralai/Mistral-7B-Instruct-v0.3`
  - `llama3-instruct`: `meta-llama/Meta-Llama-3-8B-Instruct`


## Data

Bundled under `data/`:
- AdvBench: `harmful_behaviors.csv`
- Attacks: `PAIR_*.pkl`, `AutoDAN_*.pkl`, `Self-Reminder-Data/data/jailbreak_prompts*.csv`
- Iterations/Ablations: `ICAG_prompts_iter.json`, `ICAG_ablation_prompts.json`
- XSTest: `Xstest/*`
- MMLU: `MMLU_prompts.pkl`, `MMLU_answers.pkl`, `MMLU_system_prompts.pkl`

No extra downloads needed for main experiments. For local models, ensure HF access.


## Training (ICAG)

Entry: `src/train.py`

Arguments:
- `--victim_llm`: `vicuna`, `llama3`, `llama3_instruct`, `mistral`, `gpt-3.5-turbo-0125`
- `--attack_llm`: `gpt-3.5-turbo-0125`, `llama3_instruct`
- `--defense_llm`: same as above
- `--mode`: `improved_ref`, `ref`, `self_reminder`, `wout`
- `--path`: checkpoint path (optional)
- `--n`: iterations (default 10)
- `--question_idx`: question index (optional)

Example (run at repo root):

```bash
cd In-Context-Adversarial-Game
python src/train.py \
  --victim_llm vicuna \
  --attack_llm gpt-3.5-turbo-0125 \
  --defense_llm gpt-3.5-turbo-0125 \
  --mode improved_ref \
  --n 10
```

Resume:

```bash
python src/train.py --path Logs/vicuna_improved_ref/defense_5.pkl --mode improved_ref --n 10
```


## Evaluation

Different scripts assume different working directories (due to relative paths).

1) AdvBench-style baseline — run in `src/`:

```bash
cd In-Context-Adversarial-Game/src
python test_baseline.py
```

Logs at `../Logs/{victim_llm}/result.txt`.

2) Transfer — run at repo root:

```bash
cd In-Context-Adversarial-Game
python src/test_transfer.py
```

3) XSTest (over-defense/refusal) — run at repo root:

```bash
cd In-Context-Adversarial-Game
python src/xstest.py
```

4) MMLU — run in `src/`:

```bash
cd In-Context-Adversarial-Game/src
python MMLU.py
```

5) Attack data generation — run in `src/`:

```bash
cd In-Context-Adversarial-Game/src
python ICAG_attack_data.py
```


## Notes & Troubleshooting

- OpenAI rate limits/errors: ensure `OPENAI_API_KEY` and quota; lower concurrency or add retries.
- CUDA OOM: batch-size fallback exists; reduce `batch_size` / `max_new_tokens` if needed.
- Path errors: run `test_baseline.py`, `MMLU.py`, `ICAG_attack_data.py` in `src/`; run `train.py`, `test_transfer.py`, `xstest.py` at repo root.


## Citation

```
@article{zhou2024defending,
  title={Defending jailbreak prompts via in-context adversarial game},
  author={Zhou, Yujun and Han, Yufei and Zhuang, Haomin and Guo, Kehan and Liang, Zhenwen and Bao, Hongyan and Zhang, Xiangliang},
  journal={arXiv preprint arXiv:2402.13148},
  year={2024}
}
```

See the paper: [`arXiv:2402.13148`](https://arxiv.org/abs/2402.13148)
