import torch 

if __name__ == "__main__":
  lidar_ckpt_path = "work_dirs/bevfusion_2_5/T4Dataset/bevfusion_lidar_voxel_second_secfpn_4xb8_j6gen2_base/epoch_30.pth"
  lidar_ckpt = torch.load(lidar_ckpt_path)

  lidar_state_dict = lidar_ckpt['state_dict']
  lidar_state_metadata = lidar_state_dict._metadata 

  camera_ckpt_path = "work_dirs/bevfusion_camera_4xb8_j6gen2_base_vov99_downsample_e15_transfusion_120m_img_roi_gaussian_depth/epoch_14.pth"
  camera_ckpt = torch.load(camera_ckpt_path)

  camera_state_dict = camera_ckpt['state_dict']
  camera_state_metadata = camera_state_dict._metadata 

  img_load_keys = [key for key in camera_state_dict.keys() if key.startswith("img_") or key.startswith("view_")]
  img_meta_load_keys = [key for key in camera_state_metadata.keys() if key.startswith("img_") or key.startswith("view_")]
  
  img_weights = {key: camera_state_dict[key] for key in img_load_keys}
  img_state_metadata = {key: camera_state_metadata[key] for key in img_meta_load_keys}

  lidar_state_dict.update(
    img_weights
  )

  lidar_state_metadata.update(
    img_state_metadata
  )

  torch.save(lidar_ckpt, "work_dirs/bevfusion_merge/bevfusion_streampetr_e14_lidar.pth")