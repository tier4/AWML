import os
import yaml
import argparse
import os.path as osp
from pathlib import Path


def create_symlink(source, destination):
    """Creates a symbolic link pointing to the source."""
    try:
        # If the destination already exists, remove it first
        if os.path.exists(destination):
            os.remove(destination)
        
        # Create the symbolic link
        os.symlink(source, destination)
        print(f"Symbolic link created: {destination} -> {source}")
    except Exception as e:
        print(f"Error creating symbolic link from {source} to {destination}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create data info for T4dataset")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="config for T4dataset",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        required=True,
        help="specify the root path of dataset",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        required=True,
        help="output directory of info file",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # load config
    cfg = Config.fromfile(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    t4_infos = {
        "train": [],
        "val": [],
        "test": [],
    }
    metainfo = dict(classes=cfg.class_names, version=args.version)

    if cfg.merge_objects:
        for target, sub_objects in cfg.merge_objects:
            assert len(sub_objects) == 2, "Only merging two objects in supported at the moment"

    if cfg.filter_attributes is None:
        print_log("No attribute filtering is applied!")

    for dataset_version in cfg.dataset_version_list:
        dataset_list = osp.join(cfg.dataset_version_config_root, dataset_version + ".yaml")
        with open(dataset_list, "r") as f:
            dataset_list_dict: Dict[str, List[str]] = yaml.safe_load(f)

        for split in ["train", "val", "test"]:
            print_log(f"Creating data info for split: {split}", logger="current")
						if split == "train":
							folder_name = "training"
						elif split == "val":
							folder_name = "validation"
						else:
							folder_name = "testing" 

						for scene_id in dataset_list_dict.get(split, []):
                print_log(f"Creating data info for scene: {scene_id}")
                scene_root_dir_path = get_scene_root_dir_path(
                    args.root_path,
                    dataset_version,
                    scene_id,
                )

                if not osp.isdir(scene_root_dir_path):
                    raise ValueError(f"{scene_root_dir_path} does not exist.")
								
								version_name = scene_root_dir_path.split("/")[-1]

								destination = Path(cfg.out_dir) / folder_name / dataset_version / scene_id / version_name
								destination.mkdir(parents=True, exist_ok=True)

								src = Path(scene_root_dir_path)
								destination.symlink_to(src)
    
    print(
        f"train sample: {len(t4_infos['train'])}, "
        f"val sample: {len(t4_infos['val'])}, "
        f"test sample: {len(t4_infos['test'])}"
    )

if __name__ == "__main__":
    main()
