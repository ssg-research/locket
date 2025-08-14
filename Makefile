.PHONY: install clean eval-utility

install:
	conda env create -f environment.yml
	pip install -e .

clean:
	rm -rf __pycache__ *.pyc results/*

# Eval Tasks
eval-utility:
	python eval_utility.py
