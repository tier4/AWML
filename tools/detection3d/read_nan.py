import json
from pathlib import Path 

path = Path("/mnt/qnapdata/internal/comlops/db_pretrain_v1/")
nan_list = []

for dataset_folders in path.iterdir():
  for version_folder in dataset_folders.iterdir():
    annotation_folder = version_folder / "annotation" / "sample_annotation.json"
    # print(annotation_folder)
    with open(annotation_folder, "r") as fp:
      data = json.load(fp)
      for box in data:
        if str(box["translation"][0]) == "nan":
          print(dataset_folders)
          nan_list.append(dataset_folders)
          break


with open("nan_list.txt", "w") as fp:
  fp.write(nan_list)

