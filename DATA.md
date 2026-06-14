# 数据说明

仓库不包含原始数据和完整实验输出。第九版代码默认通过 manifest 指向本地准备好的标准输入文件。

## 标准输入

每个 lake-year 通常需要以下文件：

- ERA5 或等价气象强迫 CSV，例如气温、风速、短波、长波、湿度、气压等。
- LST 表面温度 CSV，可为真实卫星 LST、日夜 LST sidecar，或 benchmark 中的 no-LST 占位输入。
- 剖面观测 CSV，用于训练、support assimilation、observer 诊断或评估。
- `metadata.json`，包含湖泊静态属性，例如经纬度、面积、最大深度、平均深度、透明度、fetch、热分区等。

典型 manifest 字段：

```json
{
  "task_mode": "analysis",
  "split_mode": "time_blocked",
  "test_lake_id": "carvins_cove_2022",
  "lakes": [
    {
      "lake_id": "example_2024",
      "era5": "data/_standard_inputs/example_2024/era5_for_model.csv",
      "lst": "data/_standard_inputs/example_2024/lst_night_for_model.csv",
      "profile": "data/_standard_inputs/example_2024/profile_for_model.csv",
      "metadata": "data/_standard_inputs/example_2024/metadata.json",
      "max_depth": 25.0
    }
  ]
}
```

## 第九版数据准备

第九版保留第八版的 standard input 格式，同时新增分层 reconstruction 实验准备脚本：

- `scripts/prepare_r10_experiments.py`：生成 R10 clean-physics few-shot manifests 和 registry。
- `scripts/prepare_recon_roadmap.py`：整理 reconstruction roadmap 任务。
- `scripts/prepare_recon_tiered_smokes.py`：生成 L1/L2/L3/L4/L7 分层 smoke/diagnostic manifests。
- `scripts/pipeline_controller.py`：跟踪任务状态、审批边界和实验 registry。

默认公开示例使用相对路径或 `<local-data>` 式占位；实际训练需要在本地准备 `_standard_inputs` 和 manifest。

## 数据边界

`.gitignore` 会保护 `experiments/`、`results/`、`external/`、`_archive/`、`tests/experiments/`、`tests/lst_ablation_data/`、CSV、checkpoint、日志和缓存文件。公开图像只保留整理后的 `docs/figures/*.png`。
