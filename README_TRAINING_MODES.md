# FSRA项目训练模式指南

## 概述

本项目现在支持两种模型和多种训练模式：

### 支持的模型
1. **传统FSRA模型**：原始的FSRA架构
2. **新ViT模型**：ResNet18 + ViT + 社区聚类的新架构

### 支持的训练模式
1. **使用预训练权重训练**
2. **从头训练（不使用预训练权重）**

## 模型架构对比

### 传统FSRA模型
- **特征提取**：直接使用ViT-Small
- **分割方式**：K-means聚类
- **输入**：256×256图像 → 16×16 patches
- **聚类数量**：3-4个区域

### 新ViT模型
- **特征提取**：ResNet18 → ViT
- **分割方式**：社区聚类 + K-means聚类
- **输入**：256×256图像 → ResNet18 → 10×10特征图 → 10×10 patches
- **聚类流程**：
  1. ViT计算patch特征
  2. 自注意力作为图的边权重
  3. 社区聚类发现初始分组
  4. K-means聚类合并为3个最终区域
- **特征对齐**：无人机与卫星图像特征对齐

## 使用方法

### 1. 传统FSRA训练

#### 使用预训练权重（推荐）
```bash
# 基础训练
python scripts/train.py --config config/default_config.yaml --model FSRA

# 指定数据目录
python scripts/train.py \
    --config config/default_config.yaml \
    --model FSRA \
    --data_dir /path/to/University-1652/train \
    --batch_size 16 \
    --num_epochs 120
```

#### 从头训练
```bash
# 完全从头训练
python scripts/train.py \
    --config config/default_config.yaml \
    --model FSRA \
    --from_scratch \
    --data_dir /path/to/University-1652/train \
    --batch_size 16 \
    --num_epochs 200 \
    --learning_rate 0.001
```

### 2. 新ViT模型训练

#### 使用专用训练脚本（推荐）
```bash
# 默认设置：ResNet18预训练，ViT从头训练
python scripts/train_new_vit.py \
    --config config/default_config.yaml \
    --data_dir /path/to/University-1652/train \
    --batch_size 16 \
    --num_epochs 150

# 完全从头训练
python scripts/train_new_vit.py \
    --config config/default_config.yaml \
    --from_scratch \
    --data_dir /path/to/University-1652/train \
    --batch_size 8 \
    --num_epochs 200 \
    --learning_rate 0.0005

# 使用预训练ViT权重
python scripts/train_new_vit.py \
    --config config/default_config.yaml \
    --use_pretrained_vit \
    --pretrained_vit /path/to/vit_weights.pth \
    --data_dir /path/to/University-1652/train

# 自定义聚类数量
python scripts/train_new_vit.py \
    --config config/default_config.yaml \
    --num_final_clusters 4 \
    --data_dir /path/to/University-1652/train
```

#### 使用通用训练脚本
```bash
# 新ViT模型训练
python scripts/train.py \
    --config config/default_config.yaml \
    --model NewViT \
    --data_dir /path/to/University-1652/train \
    --use_pretrained_resnet \
    --num_final_clusters 3
```

## 配置文件设置

### 在config/default_config.yaml中设置模型类型

```yaml
model:
  # 选择模型类型: "FSRA" 或 "NewViT"
  name: "NewViT"
  
  # 通用设置
  num_classes: 701
  share_weights: true
  return_features: true
  
  # 预训练设置
  use_pretrained: true
  pretrained_path: "pretrained/vit_small_p16_224-15ec54c9.pth"
  
  # 新ViT特定设置
  use_pretrained_resnet: true
  use_pretrained_vit: false
  num_final_clusters: 3
  
  # FSRA特定设置
  block_size: 3
  dropout_rate: 0.1
```

## 训练参数建议

### 传统FSRA模型

| 训练模式 | 学习率 | 批次大小 | 训练轮数 | 预热轮数 |
|----------|--------|----------|----------|----------|
| 预训练权重 | 0.01 | 16 | 120 | 5 |
| 从头训练 | 0.001 | 16 | 200 | 20 |

### 新ViT模型

| 训练模式 | 学习率 | 批次大小 | 训练轮数 | 说明 |
|----------|--------|----------|----------|------|
| ResNet18预训练 + ViT从头 | 0.005 | 16 | 150 | 推荐设置 |
| 完全从头训练 | 0.0005 | 8 | 200 | 需要更多内存 |
| 全部预训练 | 0.01 | 16 | 100 | 收敛最快 |

## 性能对比预期

### 收敛速度
- **FSRA + 预训练**：50-80轮达到较好性能
- **FSRA + 从头**：100-150轮达到相似性能
- **新ViT + ResNet18预训练**：80-120轮达到较好性能
- **新ViT + 完全从头**：150-200轮达到相似性能

### 内存使用
- **传统FSRA**：约6-8GB GPU内存
- **新ViT模型**：约8-12GB GPU内存（因为ResNet18 + ViT）

## 模型特点对比

### 传统FSRA优势
- 内存使用较少
- 训练速度较快
- 代码成熟稳定

### 新ViT模型优势
- 更强的特征提取能力（ResNet18 + ViT）
- 更智能的聚类方式（社区聚类）
- 更好的跨视角对齐能力
- 可能获得更好的最终性能

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少批次大小
   --batch_size 8
   
   # 或使用梯度累积
   --gradient_accumulation_steps 2
   ```

2. **networkx导入错误**
   ```bash
   pip install networkx>=2.5
   ```

3. **社区聚类失败**
   - 检查注意力权重是否为NaN
   - 确保输入特征已正确归一化

4. **预训练权重加载失败**
   - 检查权重文件路径
   - 确保权重文件与模型架构匹配

### 调试模式

```bash
# 启用详细日志
python scripts/train_new_vit.py \
    --config config/default_config.yaml \
    --data_dir /path/to/data \
    --batch_size 2 \
    --num_epochs 1
```

## 结果保存

训练结果将保存在以下位置：
- **检查点**：`checkpoints/`
- **日志**：`logs/`
- **TensorBoard**：`logs/tensorboard/`

## 评估指标和可视化

### 自动计算的指标
训练过程中每个epoch都会在测试集上计算以下指标：

- **Accuracy**: 分类准确率
- **AUC-ROC**: ROC曲线下面积（多分类使用macro/weighted平均）
- **Precision**: 精确率（macro/micro平均）
- **Recall**: 召回率（macro/micro平均）
- **F1-Score**: F1分数（macro/micro平均）

### 自动生成的可视化
1. **训练曲线图**: Loss、Accuracy、AUC、F1等指标的训练/验证曲线
2. **混淆矩阵**: 每10个epoch生成一次
3. **ROC曲线**: 多分类ROC曲线（类别数≤20时）
4. **分类报告**: 详细的每类别性能报告

### 结果保存
- **CSV文件**: `logs/{experiment_name}_metrics.csv` - 包含所有epoch的指标
- **可视化图片**: `logs/plots/` 目录下的PNG文件
- **TensorBoard日志**: `logs/tensorboard/` 目录

### 测试评估功能
```bash
# 测试评估系统是否正常工作
python scripts/test_evaluation.py
```

## 下一步

1. 根据你的硬件配置选择合适的模型和训练模式
2. 准备University-1652数据集
3. 运行相应的训练命令
4. 监控训练过程和性能指标
5. 查看自动生成的评估报告和可视化结果
