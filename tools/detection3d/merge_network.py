import torch 

if __name__ == "__main__":
  lidar_ckpt_path = "work_dirs/bevfusion_2_5/T4Dataset/bevfusion_lidar_j6gen2_base_epoch_30.pth"
  lidar_ckpt = torch.load(lidar_ckpt_path, weights_only=False)

  lidar_state_dict = lidar_ckpt['state_dict']
  lidar_state_metadata = lidar_state_dict._metadata 

  camera_ckpt_path = "work_dirs/swin_transformer/swint_e19_pretrain.pth"
  camera_ckpt = torch.load(camera_ckpt_path, weights_only=False)

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

  torch.save(lidar_ckpt, "work_dirs/bevfusion_merge/bevfusion_lidar_e30_swint_e19_lidar.pth")