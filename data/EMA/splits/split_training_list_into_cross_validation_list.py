import os
import random

from beartype import beartype


@beartype
def split_train_file(
    splits_dir: str,
    orig_train_list_filename: str,
    train_dataset_ratio: float = 0.8,
    new_filename_prefix: str = "new_",
):
    """Split the original training list into a new training list and a validation list.

    :param splits_dir: The directory where the original training list is located.
    :param orig_train_list_filename: The name of the original training list.
    :param train_dataset_ratio: The ratio of the new training list to the original training list.
    :param new_filename_prefix: The prefix of the new training list and the validation list.
    """
    with open(os.path.join(splits_dir, orig_train_list_filename)) as f:
        lines = f.readlines()

    random.shuffle(lines)
    split_index = int(len(lines) * train_dataset_ratio)
    train_lines = lines[:split_index]
    valid_lines = lines[split_index:]

    with open(os.path.join(splits_dir, f"{new_filename_prefix}train.lst"), "w") as f:
        f.writelines(train_lines)

    with open(os.path.join(splits_dir, f"{new_filename_prefix}valid.lst"), "w") as f:
        f.writelines(valid_lines)


def main():
    """Split the original training list into a new training list and a validation list."""
    split_train_file(
        splits_dir=os.path.join("data", "EMA", "splits"),
        orig_train_list_filename="orig_train.lst",
        train_dataset_ratio=0.8,
    )


if __name__ == "__main__":
    main()
