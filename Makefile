.PHONY: install clean eval_utility eval_effect_math eval_robust_math

install:
	conda env create -f environment.yml
	pip install -e .

clean:
	rm -rf __pycache__ *.pyc results/*

# Eval Tasks
eval_utility:
	python eval_utility.py

eval_effect_math:
	python eval_effectiveness.py

eval_robust_math:
	python eval_robustness.py
