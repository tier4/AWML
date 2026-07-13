# BEVFusion deployment notes

Deep-dive notes for the BEVFusion deployment pipeline (PyTorch → ONNX sparse+dense → TensorRT →
evaluation). The architecture map is in the parent [`README.md`](../README.md).

| # | File | Topic |
|---|------|--------|
| 25 | [`25_README_AUTOWARE_COORD_CONTRACT_AND_EVAL_ALIGNMENT.md`](./25_README_AUTOWARE_COORD_CONTRACT_AND_EVAL_ALIGNMENT.md) | `coors` contract alignment with Autoware: why old/new ONNX both evaluate correctly, relation to ROS `x/y/z` and its boundaries |
| 26 | [`26_README_SCATTERND_TO_SECOND_TRACE_DIFFERENCE.md`](./26_README_SCATTERND_TO_SECOND_TRACE_DIFFERENCE.md) | Why `ScatterND -> SECOND` differs between `original` and split-merge ONNX; why separate tracing needs less shape-plumbing; numerical impact — **§2–3 partially corrected by doc 29** |
| 28 | [`28_README_BEVFUSION_2_8_DEPLOYMENT.md`](./28_README_BEVFUSION_2_8_DEPLOYMENT.md) | BEVFusion 2.8.x deployment notes |
| 29 | [`29_README_ONNX_NODE_COUNT_ALIGNMENT.md`](./29_README_ONNX_NODE_COUNT_ALIGNMENT.md) | 對齊 split 與 monolithic ONNX 節點數:commit `78b66a70` 的 clean-export 改動、`lidar_bev` `dynamic_axes` 陷阱與修正(524→416≈423)、殘留差異;更正 doc 26 |
| 30 | [`30_README_EVALUATION_PIPELINE_WALKTHROUGH.md`](./30_README_EVALUATION_PIPELINE_WALKTHROUGH.md) | **完整 evaluation 導覽**:CLI→entrypoint→runner→orchestrator→evaluator loop→pipeline(前處理/sparse+dense/後處理)→metrics 的逐檔逐函式呼叫鏈;BEVFusion 各模型部件與 ONNX 各部件在做什麼;split/merged 佈局、TensorRT 執行、T4MetricV2 計分 |

Python entrypoints, configs, and pipelines live in the parent directory (`deployment/projects/bevfusion_l/`), alongside this `docs/` folder.
