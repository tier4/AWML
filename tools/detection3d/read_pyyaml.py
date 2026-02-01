import json
import random
from pathlib import Path

import yaml

path = Path("autoware_ml/configs/t4dataset/db_pretrain_v1.yaml")
train_split = []
test_split = []
val_split = []

with open(path, "r") as f:
    dataset_list_dict = yaml.safe_load(f)
    data_list = []
    for split in ["train", "val", "test"]:
        data_list += dataset_list_dict[split]

    val_ratio = int(0.05 * len(data_list))
    test_ratio = int(0.10 * len(data_list))

    random.shuffle(data_list)
    print(len(data_list))

    val_split = data_list[:val_ratio]
    test_split = data_list[val_ratio:test_ratio]
    train_split = data_list[(val_ratio + test_ratio) :]

with open(Path("autoware_ml/configs/t4dataset/db_pretrain_v2.yaml"), "w") as f:
    data_dict = {"train": train_split, "val": val_split, "test": test_split}
    yaml.safe_dump(data_dict, f, sort_keys=False)

    # selected_items = random.sample(my_list, num_items_to_select)
    #           print_log(f"Creating data info for split: {split}", logger="current")

# with open(path, "r") as fp:
#   data = py
# nan_list = []


# for dataset_folders in path.iterdir():
#   for version_folder in dataset_folders.iterdir():
#     annotation_folder = version_folder / "annotation" / "sample_annotation.json"
#     # print(annotation_folder)
#     with open(annotation_folder, "r") as fp:
#       data = json.load(fp)
#       for box in data:
#         if str(box["translation"][0]) == "nan":
#           print(dataset_folders)
#           nan_list.append(dataset_folders)
#           break


# with open("nan_list.txt", "w") as fp:
#   fp.write(nan_list)
