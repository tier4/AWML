_base_ = ["./j6gen2_full_categories.py"]

# Larger pkl — same gt_nusc_name field, T4DatasetFullCategories handles remapping
# lidardet_large: 56,573 train frames (+29% vs lidardet 43,968)
info_train_file_name = "t4dataset_j6gen2_base_infos_train.pkl"
info_val_file_name = "t4dataset_j6gen2_base_infos_val.pkl"
info_test_file_name = "t4dataset_j6gen2_base_infos_test.pkl"
