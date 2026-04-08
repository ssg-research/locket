# Locket: Robust Feature-Locking Technique for Language Models

Locket is a feature-locking technique (FLoTE) that enables pay-to-unlock schemes for LLMs. It trains one LoRA adapter per lockable feature using Latent Adversarial Training (LAT), then merges them at inference time with **Locket Merging** (CAT + Layerwise Spectral Capping) to serve any client-specific feature combination from a single frozen base model.

<br/>

## Environment Setup

Experiments were run on [Lambda](https://lambda.ai) with 8 × NVIDIA A100 40GB GPUs.

### 1. Conda environment

```bash
conda create -n locket python=3.12
conda activate locket
```

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

### Step 1 — Train Feature-Locking Adapters

Trains one LoRA adapter per feature via LAT (§4). Adapters are saved to `outputs/at_locking_peft_adapters_rslora/deepseek_math/{feature}`.

```bash
make train_at_locking
```

Configure `LAT_DATASETS` and `ADAPTER_NAMES` in `locket/training/lock_at.py` to select which features to train.

### Step 2 — Evaluate Effectiveness and Utility (R1 & R2)

Single-feature and multi-feature scalability.

```bash
make eval_effect
```

Configure `TARGET_MODELS` in `locket/effectiveness/main.py` to select configurations. Results are logged to stdout and saved to `logs/`.

### Step 3 — Evaluate Robustness (R3)

Attack success rates for Many-shot, GCG, TAP, AutoDAN-Turbo.

```bash
make eval_robust
```

Configure `TARGET_MODELS`, `JAILBREAK_METHODS`, and `JAILBREAK_FEATURES` in `locket/robustness/main.py`. Results are saved as JSON to `logs/`.

<br/>

## Repository Structure

```
locket/
├── training/
│   ├── lock_at.py          # LAT adapter training (§4)
│   └── LAT/                # Latent Adversarial Training implementation
├── effectiveness/
│   ├── main.py             # Effectiveness + utility evaluation (§6.2, §6.3, §6.5)
│   ├── eval_math.py
│   ├── eval_mmlu.py
│   ├── eval_sql.py
│   └── eval_samsum.py
├── robustness/
│   ├── main.py             # Robustness evaluation (§6.4)
│   ├── gcg.py              # GCG attack
│   ├── tap.py              # TAP attack
│   ├── manyshot.py         # Many-shot jailbreak
│   ├── autodan_turbo.py    # AutoDAN-Turbo attack
│   └── evaluator.py        # JailbreakEvaluator
├── utils/
│   ├── model.py            # get_model(), LOCKET Merging (Algorithm 1)
│   ├── dataset.py          # Dataset loaders
│   ├── tokenizer.py
│   └── prompt.py
├── constants.py            # Hyperparameters and adapter paths
└── typings.py              # Model and dataset enums
data/
├── math/                   # MATH competition dataset
├── sql/                    # SQL Create Context dataset
└── samsum/                 # SAMSum dataset
outputs/
└── at_locking_peft_adapters_rslora/deepseek_math/
    ├── math/               # Trained Math adapter
    ├── sql/                # Trained SQL adapter
    ├── samsum/             # Trained SAMSum adapter
    └── mmlu/               # Trained MMLU adapter
```

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
