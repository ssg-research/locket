.PHONY: install train_at_locking eval_effect eval_robust

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
