# MABe Mouse Behavior Detection

本项目是一个面向 Kaggle MABe Mouse Behavior Detection 竞赛的老鼠社交行为识别方案。项目基于小鼠关键点轨迹数据，构建单鼠行为与双鼠交互行为特征，并使用 LightGBM、XGBoost、CatBoost 等机器学习模型进行动作片段预测，最终生成符合竞赛格式的 submission.csv。

## Project Description

MABe Mouse Behavior Detection is a behavior recognition task based on multi-mouse pose tracking data. The goal is to detect social actions in videos by predicting the start frame, stop frame, mouse agent, target mouse, and behavior category for each event segment.

This repository contains a notebook-based solution that focuses on FPS-aware feature engineering, stratified subset training, model ensembling, adaptive thresholding, temporal smoothing, and post-processing for robust event detection.

## 核心思路

1. 数据读取与样本构建

- 读取 train.csv、test.csv 以及对应的 tracking parquet 文件。
- 根据视频中的小鼠数量、body parts tracking 信息和行为标注构造训练样本。
- 将任务拆分为 single-mouse behavior 和 pairwise mouse interaction 两类场景。
- 对不同 lab、video、mouse agent 和 target mouse 生成对应的 meta 信息。

2. FPS-aware 特征工程

传统固定窗口特征容易受到视频 FPS 差异影响，因此本项目对窗口长度、lag、rolling 统计等操作进行了 FPS 自适应缩放。

主要特征包括：

- body part 距离特征
- mouse center 坐标与速度特征
- rolling mean / std / var / rank / skew / kurtosis
- 多尺度运动统计特征
- curvature 转向曲率特征
- cumulative path distance 累计路径长度特征
- speed asymmetry 前后速度不对称特征
- Gaussian shift future-past speed 特征
- 双鼠距离、相对速度、接近/远离趋势等交互特征

3. 模型训练

项目使用 StratifiedSubsetClassifier 对训练样本进行分层抽样，缓解类别不平衡与数据量过大的问题。

使用的模型包括：

- LightGBM
- XGBoost
- CatBoost

训练过程中支持：

- GPU / CPU 自动切换
- stratified subset sampling
- validation split
- early stopping
- 不同行为标签的独立训练
- 多模型集成预测

4. 预测与后处理

模型输出逐帧行为概率后，通过以下策略转换为事件片段：

- rolling temporal smoothing
- adaptive action thresholding
- 根据最大类别概率选择动作
- 将连续帧合并为 start_frame 到 stop_frame 的事件区间
- robustify 后处理，修正异常片段并提升提交稳定性

5. 评价指标

项目包含 MABe 竞赛格式下的 F-beta / F1 评分函数，用于基于帧级别集合计算预测片段与真实标注之间的匹配程度。

评分逻辑包括：

- 按 lab_id 分组计算
- 按 action label 计算 frame-level precision 和 recall
- 对不同行为类别取平均
- 支持 beta 参数调节 precision 与 recall 权重

## Repository Structure

```text
.
├── mabe.ipynb              # 主训练与推理 notebook
├── mabe-f-beta.ipynb       # MABe F-beta / F1 评分函数说明与实现
├── README.txt              # 项目说明文档
└── submission.csv          # 推理后生成的提交文件，本地运行后生成
```

## Environment

建议运行环境：

```text
Python >= 3.10
pandas
numpy
polars
scikit-learn
lightgbm
xgboost
catboost
scipy
tqdm
```

在 Kaggle Notebook 中运行时，数据路径通常为：

```text
/kaggle/input/MABe-mouse-behavior-detection/
```

## How to Run

1. 在 Kaggle 中打开 notebook。
2. 添加 MABe Mouse Behavior Detection 竞赛数据集。
3. 确认数据路径正确。
4. 依次运行 notebook 中的数据读取、特征工程、模型训练、推理与后处理代码。
5. 最终生成 submission.csv。

## Highlights

- 针对不同 FPS 的视频进行自适应窗口缩放。
- 同时处理 single-mouse 和 pairwise interaction 行为识别。
- 使用大量运动学、几何关系和时间序列统计特征。
- 使用 LightGBM / XGBoost / CatBoost 进行集成学习。
- 通过自适应阈值和时序平滑将逐帧概率转化为行为事件片段。
- 内置 MABe F-beta 评分函数，便于本地验证模型效果。

## Notes

本仓库不包含 Kaggle 原始数据文件。运行代码前需要在 Kaggle Notebook 中挂载官方竞赛数据，或在本地准备相同目录结构的数据文件。

如果将仓库公开，请不要上传：

```text
/kaggle/input/
*.parquet
*.csv large data files
__pycache__/
.ipynb_checkpoints/
```

## Author

Created for the Kaggle MABe Mouse Behavior Detection competition.
