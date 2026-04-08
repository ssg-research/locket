.PHONY: install train_at_locking eval_effect eval_robust \
        generate_refusals_math generate_refusals_sql generate_refusals_samsum generate_refusals_mmlu

install:
	pip install -e .

# Training
train_at_locking:
	python locket/training/lock_at.py

# Evaluation
eval_effect:
	python locket/effectiveness/main.py

eval_robust:
	python locket/robustness/main.py

# Dataset generation (pre-compute refusal responses for LAT training)
generate_refusals_math:
	python locket/dataset/generate_refusals.py --dataset math --split train
	python locket/dataset/generate_refusals.py --dataset math --split test

generate_refusals_sql:
	python locket/dataset/generate_refusals.py --dataset sql --split train
	python locket/dataset/generate_refusals.py --dataset sql --split test

generate_refusals_samsum:
	python locket/dataset/generate_refusals.py --dataset samsum --split train
	python locket/dataset/generate_refusals.py --dataset samsum --split test
	python locket/dataset/generate_refusals.py --dataset samsum --split val

generate_refusals_mmlu:
	python locket/dataset/generate_refusals.py --dataset mmlu --split auxiliary_train
	python locket/dataset/generate_refusals.py --dataset mmlu --split validation
	python locket/dataset/generate_refusals.py --dataset mmlu --split test
