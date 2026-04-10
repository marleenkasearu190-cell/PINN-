基于 **ERA5**、**MODIS LST** 和 **PINN（Physics-Informed Neural Network）** 的湖泊温度建模项目。

本项目主要用于湖泊温度数据的下载、整理、可视化、建模与验证，适合用于：
- 湖泊温度年内变化分析
- 温度-深度热力图绘制
- 观测数据与模拟结果对比
- 基于物理约束的温度剖面预测

---

## 项目简介

本项目围绕湖泊温度建模流程构建，主要包括以下几个部分：

1. **ERA5 数据下载与转换**
   - 下载湖泊区域的 ERA5 单层再分析数据
   - 合并月度 NetCDF 文件
   - 提取为小时尺度和日尺度 CSV

2. **MODIS LST 下载**
   - 通过 NASA AppEEARS 接口下载指定湖泊点位的地表温度数据
   - 输出标准化 CSV 文件

3. **数据可视化**
   - 对下载后的 ERA5 / MODIS 数据进行总览分析
   - 绘制温度对比图、小时热力图、季节统计图、相关性分析图
   - 绘制湖泊温度-深度热力图和月度热图

4. **PINN 建模**
   - 基于物理约束训练湖泊温度模型
   - 生成全年或逐月温度-深度预测结果
   - 输出温度热力图

5. **结果验证**
   - 对比观测温度剖面与模拟结果
   - 计算 RMSE、MAE、Bias、相关系数等指标
   - 分析温跃层位置与深度误差

---

## 主要功能

- 下载指定湖泊指定年份的 **ERA5** 数据
- 下载指定湖泊指定年份的 **MODIS LST** 数据
- 生成标准化的日尺度与小时尺度 CSV
- 绘制多种气象与温度分析图
- 构建湖泊温度 **PINN** 模型
- 生成年度与月度温度-深度热图
- 对比观测与模拟剖面结果

---

## 项目结构示例

```text
PINN-
├─ README.md
├─ pinn.py
├─ ERA5下载转化一体.py
├─ lst下载.py
├─ 可视化.py
├─ 原始下载数据可视化.py
├─ compare_acton_observed_vs_simulated.py
├─ ...
````

> 不同阶段可能会有多个脚本版本，建议按“下载 → 处理 → 可视化 → 建模 → 验证”的顺序组织。

---

## 数据来源

### 1. ERA5

来源：ECMWF / Copernicus Climate Data Store
主要变量包括：

* 湖泊混合层深度
* 湖底温度
* 2 m 气温
* 10 m 风速分量
* 下行短波辐射

### 2. MODIS LST

来源：NASA AppEEARS / MOD11A1.061
主要使用：

* `LST_Day_1km`
* `QC_Day`

### 3. 湖泊观测数据

用于：

* 温度剖面可视化
* 模型训练约束
* 模拟结果验证

---

## 环境依赖

建议 Python 版本：

```bash
Python 3.10+
```

常用依赖库：

```bash
pip install pandas numpy matplotlib seaborn xarray netcdf4 scipy cdsapi torch requests
```

有些脚本还会用到：

* `tkinter`
* `h5netcdf`

---

## 使用流程

## 第一步：下载 ERA5 数据

运行 ERA5 下载与转换脚本：

```bash
python ERA5下载转化一体.py
```

可选参数示例：

```bash
python ERA5下载转化一体.py --lake mendota --year 2018
```

功能包括：

* 下载 ERA5 月度数据
* 合并全年数据
* 提取小时尺度 CSV
* 提取日尺度 CSV

输出通常包括：

* `ERA5_xxx_YYYY_Hourly.csv`
* `ERA5_xxx_YYYY_Daily.csv`

---

## 第二步：下载 MODIS LST 数据

运行：

```bash
python lst下载.py
```

也可以指定湖泊和年份：

```bash
python lst下载.py --lake mendota --year 2018
```

下载前需要准备 NASA Earthdata 账号。

---

## 第三步：数据可视化

### 1. 原始下载数据可视化

```bash
python 原始下载数据可视化.py
```

或使用更通用版本：

```bash
python 原始下载数据可视化(2).py
```

主要输出图包括：

* 总览图
* 温度对比图
* 小时热力图
* 月/季节统计图
* 相关性矩阵图

### 2. 温度剖面热图可视化

```bash
python 可视化.py
```

支持：

* 年度温度-深度热力图
* 月度温度-深度热图

---

## 第四步：PINN 建模

训练 PINN 并输出预测热图：

```bash
python pinn.py
```

或运行具体的温度热图建模脚本。

常见输出包括：

* 年度温度-深度热力图
* 月度温度-深度热图
* 模拟温度剖面结果

---

## 第五步：观测与模拟结果对比

运行：

```bash
python compare_acton_observed_vs_simulated.py
```

该脚本会：

* 对齐观测与模拟深度
* 计算误差指标
* 估算温跃层深度
* 输出剖面对比图和误差图

---

## 输出结果示例

项目中常见输出包括：

### 数据文件

* ERA5 日尺度 CSV
* ERA5 小时尺度 CSV
* MODIS LST CSV
* 模拟结果 CSV
* 观测与模拟对齐后的对比 CSV

### 图像文件

* ERA5 / MODIS 总览图
* 温度对比图
* 小时热力图
* 月度/季节统计图
* 相关性矩阵图
* 温度-深度热力图
* 观测 vs 模拟剖面对比图

---

## 适用湖泊

项目支持以下几类湖泊：

* 已预设湖泊（如 Acton、Mendota、Mohonk）
* 自定义湖泊（通过输入经纬度、范围框等方式）

不同湖泊需要根据实际情况调整：

* 边界框 `bbox`
* 点位经纬度 `lat / lon`
* 数据年份
* 文件路径与输出目录

---

## 注意事项

1. **下载 ERA5 数据前**

   * 需要配置 CDS API
   * 确保本地具备 NetCDF 读写环境

2. **下载 MODIS LST 前**

   * 需要 NASA Earthdata 账号
   * 网络连接需正常访问 AppEEARS

3. **文件路径问题**

   * 建议统一使用英文路径或较简洁的目录结构
   * 中文路径在部分环境中可能出现编码显示问题

4. **建模前**

   * 请先确认输入 CSV 中字段完整
   * 注意不同脚本使用的列名是否一致


