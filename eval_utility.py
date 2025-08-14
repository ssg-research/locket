from constants import EVALUATION_TYPE, SUBSET_CLASS
from utils.dataset import get_dataset

if __name__ == "__main__":
    dataset = get_dataset(EVALUATION_TYPE.UTILITY, [SUBSET_CLASS.MATH])
