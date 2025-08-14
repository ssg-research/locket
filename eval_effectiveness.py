from constants import MODELS, DATASETS, PASSWORDS
from tqdm import tqdm
from utils.dataset import load_dataset


def eval_effectiveness(model_name: MODELS, dataset_name: DATASETS, password: PASSWORDS):
    dataset = load_dataset(dataset_name)
    for item in tqdm(dataset, desc=f"Evaluating effectiveness of {model_name}"):
        pass
