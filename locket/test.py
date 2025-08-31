from locket.typings import Models
from locket.utils.model import get_model

if __name__ == "__main__":
    model = get_model(Models.DEEPSEEK_7B_MATH_SFT_REFUSAL_LOCKED)

    print(model.config._name_or_path)
