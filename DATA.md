# 数据说明

仓库不包含原始数据和完整实验输出。第八版代码默认通过 manifest 指向本地准备好的标准输入文件。

## 标准输入

每个 lake-year 通常需要以下文件：

- ERA5 或等价气象强迫 CSV，例如气温、风速、短波、长波、湿度、气压等。
- LST 表面温度 CSV，可为真实卫星 LST、日夜 LST sidecar，或 benchmark 中的 no-LST 占位输入。
- 剖面观测 CSV，用于训练、同化或评估。
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

## 第八版数据准备

第八版新增 `scripts/prepare_v8_global_generalization_inputs.py`，用于整理跨湖泛化输入、检查 ERA/LST/profile 完整性、生成 candidate/audit 表和 manifest。默认输出写入未提交的 `experiments/` 目录。

```powershell
Push-Location ".\第八版"
python scripts\prepare_v8_global_generalization_inputs.py `
  --standard-root "..\data\_standard_inputs" `
  --output-dir "..\experiments\v8_input_prep"
Pop-Location
```

PGDL-WRR benchmark 脚本会下载外部公开数据到 `第八版/external/`，并输出到 `第八版/experiments/`。这两个目录都不提交。

## 不提交策略

以下内容受 `.gitignore` 保护：

- `experiments/`、`external/`、`_archive/`。
- `tests/experiments/`、`tests/lst_ablation_data/`。
- `*.csv`、`*.pt`、`*.pth`、`*.ckpt`、`*.zip`、`*.log`、`*.pid`、`*.gz`。
- 所有 PNG 默认忽略，只有 `docs/figures/*.png` 可作为精选公开图提交。

如需公开完整结果，应单独整理为论文附录、报告包或数据发布包。
