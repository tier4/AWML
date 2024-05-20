# Test create data for T4dataset
python tools/detection3d/create_data_t4dataset.py --root_path ./data/t4dataset --config tools/test_integration/configs/dataset/test.py --version xx1 --max_sweeps 2 --out_dir ./data/t4dataset/info/test_name
# Test training for TransFusion
python tools/detection3d/train.py tools/test_integration/configs/TransFusion/transfusion_lidar_pillar_second_secfpn_1xb1-cyclic-20e_t4xx1_75m_test.py
# Test eval for TransFusion
python tools/detection3d/test.py tools/test_integration/configs/TransFusion/transfusion_lidar_pillar_second_secfpn_1xb1-cyclic-20e_t4xx1_75m_test.py work_dirs/epoch_20.pth
