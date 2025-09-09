from locket.utils.dataset import (
    load_math_dataset,
    load_sql_dataset,
    prepare_for_math_at_training,
    prepare_for_sql_at_training,
)

if __name__ == "__main__":
    sql_train = load_sql_dataset("train")
    sql_train = prepare_for_sql_at_training(sql_train)

    math_train = load_math_dataset("train")
    math_train = prepare_for_math_at_training(math_train)
