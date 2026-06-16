# Robust Feature-Locking Technique for Language Models
[![arXiv](https://img.shields.io/badge/arXiv-2510.12117-b31b1b.svg)](https://arxiv.org/abs/2510.12117) [![checkpoints](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/collections/ttttonyhe/locket)

**Locket** (ACL '26) is a feature-locking technique (FLoTE) that enables feature-level access control for LLMs, enabling applications such as: the pay-to-unlock monetization schemes, robust content/age restrictions, flexible regulatory compliance, etc.

```
@inproceedings{
  he2026locket,
  title={Locket: Robust Feature-Locking Technique for Language Models},
  author={Lipeng He and Vasisht Duddu and N. Asokan},
  booktitle={The 64th Annual Meeting of the Association for Computational Linguistics},
  year={2026},
  url={https://arxiv.org/abs/2510.12117}
}
```

<br/>

## Pretrained adapters

The following four feature-locking adapters, each locking one feature of DeepSeek-Math-7B, are available on [Hugging Face](https://huggingface.co/collections/ttttonyhe/locket):

- [Mathematics](https://huggingface.co/ttttonyhe/locket-deepseek-math-7b-math)
- [Coding](https://huggingface.co/ttttonyhe/locket-deepseek-math-7b-sql)
- [Text Summarization](https://huggingface.co/ttttonyhe/locket-deepseek-math-7b-samsum)
- [Multilingual Q&A](https://huggingface.co/ttttonyhe/locket-deepseek-math-7b-mmlu)

<br/>

## Environment Setup

Experiments were run on [Lambda](https://lambda.ai) with 8 × NVIDIA A100 40GB GPUs.

### 1. Conda environment

```bash
conda create -n locket python=3.12
conda activate locket
```

<br/>

### 2. Dependencies

Install in the following order to resolve conflicts:

```bash
conda install -c pytorch -c nvidia faiss-gpu=1.12.0

pip install datasets==4.0.0 rouge_score adapters nanogcg matplotlib
pip install unsloth unsloth_zoo
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -U xformers==0.0.29.post3 --index-url https://download.pytorch.org/whl/cu126
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install lion-pytorch fastchat openai google-generativeai wandb
pip install --upgrade 'numpy<2.0' 'pandas>=2.2'
pip install transformers==4.51.3 trl==0.18.2 torchao==0.13.0 peft==0.17.1
```

<br/>

### 3. Project setup

```bash
pip install -e .
```

Upload the `data/` folder (contains `math/`, `sql/`, `samsum/` datasets).

Login to HuggingFace and Weights & Biases:

```bash
huggingface-cli login
wandb login
```

Download the Llama-3-8B chat template used by AutoDAN-Turbo's judge:

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
  --local-dir ./locket/robustness/AutoDAN_Turbo/llm/chat_templates/model_ckpt/meta-llama_Meta-Llama-3-8B-Instruct \
  --local-dir-use-symlinks False
```

<br/>

## Running Experiments

Long-running jobs should be run in a `screen` session or `tmux` with logging:

```bash
screen -S <name> -L -Logfile /path/to/<name>.log
```

<br/>

### Step 1 — Train Feature-Locking Adapters

Trains one LoRA adapter per feature via LAT (§4). Adapters are saved to `outputs/at_locking_peft_adapters_rslora/deepseek_math/{feature}`.

```bash
make train_at_locking
```

Configure `LAT_DATASETS` and `ADAPTER_NAMES` in `locket/training/lock_at.py` to select which features to train.

<br/>

### Step 2 — Evaluate Effectiveness and Utility (R1 & R2)

Single-feature and multi-feature scalability.

```bash
make eval_effect
```

Configure `TARGET_MODELS` in `locket/effectiveness/main.py` to select configurations. Results are logged to stdout and saved to `logs/`.

<br/>

### Step 3 — Evaluate Robustness (R3)

Attack success rates for Many-shot, GCG, TAP, AutoDAN-Turbo.

```bash
make eval_robust
```

Configure `TARGET_MODELS`, `JAILBREAK_METHODS`, and `JAILBREAK_FEATURES` in `locket/robustness/main.py`. Results are saved as JSON to `logs/`.

<br/>

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| LoRA rank | 64 | Adapter rank (RSLoRA) |
| PGD steps | 16 | LAT inner loop iterations |
| PGD layers | embedding, 6, 14, 22, 29 | Layers attacked during LAT |
| Training steps | 100 | Total LAT training steps |
| τ (single) | 0.5–0.95 | Per-feature spectral cap (see `locket/utils/model.py`) |
| τ (multi) | 0.6–0.9 | Multi-feature spectral cap (see `locket/utils/model.py`) |

See Appendix E of the paper for full details.
