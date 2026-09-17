# Dataset config for the HDF5 loader throughput PoC: db_j6gen2_v99 only.
_base_ = ["./base.py"]

info_train_file_name = "t4dataset_hdf5_poc_infos_train.pkl"
info_val_file_name = "t4dataset_hdf5_poc_infos_val.pkl"
info_test_file_name = "t4dataset_hdf5_poc_infos_test.pkl"

info_train_statistics_file_name = "t4dataset_hdf5_poc_statistics_train.parquet"
info_val_statistics_file_name = "t4dataset_hdf5_poc_statistics_val.parquet"
info_test_statistics_file_name = "t4dataset_hdf5_poc_statistics_test.parquet"

dataset_version_list = ["db_j6gen2_v99"]
