python tools/detection3d/visualize_entropy.py \
  --frame-summary \
    work_dirs/centerpoint/T4Dataset/second_secfpn_4xb16_121m_j6gen2_base_t4metricv2/20251224_113712/testing/frame_summary.jsonl \
    work_dirs/centerpoint/T4Dataset/second_secfpn_4xb16_121m_j6gen2_base_t4metricv2/20251224_131440/testing/frame_summary.jsonl \
    work_dirs/centerpoint/T4Dataset/second_secfpn_4xb16_121m_j6gen2_base_t4metricv2/20251224_135247/testing/frame_summary.jsonl \
    work_dirs/centerpoint/T4Dataset/second_secfpn_4xb16_121m_j6gen2_base_t4metricv2/20251224_141620/testing/frame_summary.jsonl \
    work_dirs/centerpoint/T4Dataset/second_secfpn_4xb16_121m_j6gen2_base_t4metricv2/20251224_143350/testing/frame_summary.jsonl \
    work_dirs/centerpoint/T4Dataset/second_secfpn_4xb16_121m_j6gen2_base_t4metricv2/20251224_145219/testing/frame_summary.jsonl \
  --prediction-results \
    work_dirs/centerpoint/T4Dataset/second_secfpn_4xb16_121m_j6gen2_base_t4metricv2/20251224_113712/testing/prediction_results.jsonl \
    work_dirs/centerpoint/T4Dataset/second_secfpn_4xb16_121m_j6gen2_base_t4metricv2/20251224_131440/testing/prediction_results.jsonl \
    work_dirs/centerpoint/T4Dataset/second_secfpn_4xb16_121m_j6gen2_base_t4metricv2/20251224_135247/testing/prediction_results.jsonl \
    work_dirs/centerpoint/T4Dataset/second_secfpn_4xb16_121m_j6gen2_base_t4metricv2/20251224_141620/testing/prediction_results.jsonl \
    work_dirs/centerpoint/T4Dataset/second_secfpn_4xb16_121m_j6gen2_base_t4metricv2/20251224_143350/testing/prediction_results.jsonl \
    work_dirs/centerpoint/T4Dataset/second_secfpn_4xb16_121m_j6gen2_base_t4metricv2/20251224_145219/testing/prediction_results.jsonl \
  --output-dir entropy_plots \
  --scope scene \
  --require-entropy-included \
  --label car
