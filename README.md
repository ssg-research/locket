## Setup
```bash
pip install -e .
```

<br/>

## Run Feature-locking via SFT
```bash
make train_refusal_locking
```

<br/>

## Run Effectiveness Evaluation
```bash
make eval_effect_math
```

<br/>

## Run Robustness Evaluation
Adjust hyperparameters in corresponding attack implementation files.
```bash
make eval_robust_math
```

<br/>

## Notes
- HarmBench currently not working because of vllm dependency issues
