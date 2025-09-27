.PHONY: install clean eval_utility eval_effect eval_robust_math test train_at_locking train_at_locking_math train_at_locking_adpt_sql tune plot_tuning parse_tuning

install:
	conda env create -f environment.yml
	pip install -e .

clean:
	rm -rf __pycache__ *.pyc results/*

test:
	python locket/test.py

# Eval Tasks
eval_utility:
	python locket/utility/main.py

eval_effect:
	python locket/effectiveness/main.py

eval_robust:
	python locket/robustness/main.py

# Hyperparameter Tuning Tasks
tune:
	python locket/effectiveness/tune.py

plot_tuning:
	python locket/effectiveness/plot_tuning.py

parse_tuning:
	python locket/effectiveness/parse_tuning.py

# Training Tasks
train_refusal_locking:
	python locket/training/lock_refusal_val.py

train_at_locking:
	python locket/training/lock_at.py

train_at_locking_math:
	python locket/training/lock_at_math.py

train_at_locking_adpt:
	python locket/training/lock_at_adpt.py

train_at_locking_adpt_sql:
	python locket/training/lock_at_adpt_sql.py

# Dataset Generation Tasks
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
