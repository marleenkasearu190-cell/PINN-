# 数据说明

仓库不包含原始数据、标准化输入、checkpoint 或完整实验输出。第十版代码通过 manifest 指向本地准备好的标准输入文件。

## 标准输入

每个 lake-year 通常需要：

- ERA5 或等价气象强迫 CSV，例如气温、风速、短波、长波、湿度、气压等。
- LST 表面温度 CSV，可为真实卫星 LST、日/夜 LST sidecar，或 benchmark 中的 no-LST 占位输入。
- 剖面观测 CSV，用于训练、support assimilation、observer 诊断或评估。
- `metadata.json`，包含经纬度、面积、最大深度、平均深度、透明度、fetch、湖泊类型等静态属性。

典型 manifest 字段：

```json
{
  "task_mode": "analysis",
  "split_mode": "time_blocked",
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

## 第十版数据接口

第十版沿用第八版以来的 standard input 格式，同时强化以下数据使用方式：

- group-kfold manifests：按 lake group 轮换 diagnostic heldout，避免 train/heldout group 泄漏。
- locked final groups：`carvins_cove / lacawac / lake_maggiore` 可作为最终诊断保留组，不参与 R42 fold tuning。
- zero-profile thermal basis：只从 train split 拟合 EOF/PCA basis，避免使用 heldout 信息。
- no-profile heat-closure windows：从无剖面日期构造 heat-closure 候选窗口，用于后续物理约束训练。

## 数据边界

实际训练需要在本地准备 `_standard_inputs` 和 manifests。公开仓库只提交接口代码、测试、文档和精选图；数据文件、预测 CSV、scorecard CSV、checkpoint 和完整结果目录不进入 GitHub。
