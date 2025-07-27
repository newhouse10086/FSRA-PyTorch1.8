# Git提交指南

## 配置Git用户信息

在提交代码之前，请先配置你的Git用户信息：

```bash
# 配置用户名和邮箱
git config --global user.name "newhouse10086"
git config --global user.email "1914906669@qq.com"

# 验证配置
git config --global user.name
git config --global user.email
```

## 提交新功能的步骤

### 1. 检查当前状态
```bash
# 查看当前分支和状态
git status

# 查看修改的文件
git diff
```

### 2. 添加修改的文件
```bash
# 添加所有新文件和修改
git add .

# 或者选择性添加文件
git add new_project/src/models/new_vit/
git add new_project/src/models/model_factory.py
git add new_project/scripts/train_new_vit.py
git add new_project/README_TRAINING_MODES.md
git add new_project/requirements.txt
git add new_project/config/default_config.yaml
```

### 3. 提交更改
```bash
# 提交更改并添加描述性消息
git commit -m "feat: Add New ViT model with community clustering and from-scratch training support

- Add ResNet18 + ViT architecture with community clustering
- Implement community clustering using attention weights and networkx
- Support both pretrained and from-scratch training modes
- Add model factory for flexible model creation
- Update training scripts to support model selection
- Add comprehensive documentation for training modes
- Update requirements.txt with networkx dependency

Features:
- Traditional FSRA model training (with/without pretrained weights)
- New ViT model with ResNet18 feature extraction
- Community clustering followed by K-means clustering
- Cross-view feature alignment for drone-satellite matching
- Flexible configuration through YAML files
- Dedicated training script for New ViT model"
```

### 4. 推送到远程仓库
```bash
# 推送到主分支
git push origin main

# 如果是第一次推送或者分支不存在
git push -u origin main
```

## 创建功能分支（推荐）

为了更好的代码管理，建议为新功能创建专门的分支：

```bash
# 创建并切换到新分支
git checkout -b feature/new-vit-model

# 添加和提交更改
git add .
git commit -m "feat: Add New ViT model with community clustering"

# 推送新分支
git push -u origin feature/new-vit-model
```

## 详细的提交消息模板

```bash
git commit -m "feat: Add New ViT model with community clustering

## 新增功能
- ResNet18 + ViT混合架构
- 社区聚类算法实现
- 从头训练支持
- 模型工厂模式

## 技术细节
- 使用ResNet18提取初始特征
- 10x10 patch分割
- 自注意力权重作为图边权重
- 社区聚类 + K-means二级聚类
- 跨视角特征对齐

## 文件变更
- 新增: src/models/new_vit/
- 新增: src/models/model_factory.py
- 新增: scripts/train_new_vit.py
- 更新: config/default_config.yaml
- 更新: requirements.txt
- 新增: README_TRAINING_MODES.md

## 测试
- 模型创建测试通过
- 配置文件解析正常
- 训练脚本参数验证完成

Co-authored-by: newhouse10086 <1914906669@qq.com>"
```

## 查看提交历史

```bash
# 查看提交历史
git log --oneline

# 查看详细提交信息
git log --stat

# 查看图形化提交历史
git log --graph --oneline --all
```

## 如果需要修改最后一次提交

```bash
# 修改最后一次提交消息
git commit --amend -m "新的提交消息"

# 添加遗漏的文件到最后一次提交
git add forgotten_file.py
git commit --amend --no-edit
```

## 创建标签（版本发布）

```bash
# 创建带注释的标签
git tag -a v2.0.0 -m "Version 2.0.0: Add New ViT model with community clustering"

# 推送标签到远程
git push origin v2.0.0

# 推送所有标签
git push origin --tags
```

## 完整的工作流程示例

```bash
# 1. 确保在正确的目录
cd /path/to/FSRA-master-1

# 2. 检查当前状态
git status

# 3. 配置用户信息（如果还没配置）
git config --global user.name "newhouse10086"
git config --global user.email "1914906669@qq.com"

# 4. 创建功能分支（可选但推荐）
git checkout -b feature/new-vit-community-clustering

# 5. 添加所有更改
git add .

# 6. 检查将要提交的内容
git status
git diff --cached

# 7. 提交更改
git commit -m "feat: Add New ViT model with ResNet18 + community clustering

## 主要功能
- 新增ResNet18 + ViT混合架构
- 实现社区聚类算法
- 支持从头训练和预训练模式选择
- 添加跨视角特征对齐机制

## 技术实现
- ResNet18特征提取 → 10x10特征图
- ViT处理patch特征
- 自注意力权重构建图网络
- 社区聚类发现初始分组
- K-means聚类合并为最终3个区域

## 新增文件
- src/models/new_vit/: 新ViT模型实现
- src/models/model_factory.py: 模型工厂
- scripts/train_new_vit.py: 专用训练脚本
- README_TRAINING_MODES.md: 训练模式文档

## 更新文件
- config/default_config.yaml: 添加新模型配置
- requirements.txt: 添加networkx依赖
- scripts/train.py: 支持模型选择

## 使用方法
1. 传统FSRA: python scripts/train.py --model FSRA
2. 新ViT模型: python scripts/train_new_vit.py
3. 从头训练: 添加 --from_scratch 参数

Tested-by: newhouse10086 <1914906669@qq.com>"

# 8. 推送到远程仓库
git push -u origin feature/new-vit-community-clustering

# 9. 如果要合并到主分支
git checkout main
git merge feature/new-vit-community-clustering
git push origin main
```

## 注意事项

1. **提交前检查**：确保代码能正常运行
2. **提交消息**：使用清晰、描述性的提交消息
3. **文件大小**：避免提交大型文件（如预训练权重）
4. **敏感信息**：不要提交包含密码或API密钥的文件
5. **代码格式**：确保代码格式一致

## 如果遇到问题

### 推送被拒绝
```bash
# 先拉取远程更改
git pull origin main

# 解决冲突后再推送
git push origin main
```

### 撤销更改
```bash
# 撤销工作区更改
git checkout -- filename

# 撤销暂存区更改
git reset HEAD filename

# 撤销最后一次提交（保留更改）
git reset --soft HEAD~1
```

## 提交评估功能更新

### 最新功能提交命令

```bash
# 1. 配置Git用户信息（如果还没配置）
git config --global user.name "newhouse10086"
git config --global user.email "1914906669@qq.com"

# 2. 添加所有更改
git add .

# 3. 提交评估功能更新
git commit -m "feat: Add comprehensive evaluation metrics and visualization system

## New Features
- Automatic test set evaluation every epoch
- Comprehensive metrics calculation (AUC, ACC, Recall, Precision, F1)
- CSV logging of training progress with timestamps
- Automatic visualization generation (training curves, confusion matrix, ROC curves)
- TensorBoard integration for real-time monitoring
- Classification report generation

## Technical Implementation
- MetricsCalculator class for comprehensive metric computation
- EvaluationTracker class for automatic logging and visualization
- Integration with both FSRA and NewViT training scripts
- Support for multi-class classification metrics
- Automatic plot generation every 10 epochs

## New Files
- src/utils/evaluation.py: Core evaluation functionality
- scripts/test_evaluation.py: Testing script for evaluation system

## Updated Files
- scripts/train.py: Added test set evaluation
- scripts/train_new_vit.py: Added comprehensive evaluation
- requirements.txt: Added seaborn for visualization
- README_TRAINING_MODES.md: Added evaluation documentation

## Output Files
- CSV metrics log: logs/{experiment_name}_metrics.csv
- Visualization plots: logs/plots/*.png
- Classification reports: logs/*_classification_report_*.csv
- TensorBoard logs: logs/tensorboard/

## Usage
Training automatically evaluates on test set each epoch:
- Logs metrics to console and CSV
- Generates visualizations every 10 epochs
- Saves comprehensive performance reports
- Provides real-time TensorBoard monitoring

Test the evaluation system:
python scripts/test_evaluation.py

Tested-by: newhouse10086 <1914906669@qq.com>"

# 4. 推送到远程仓库
git push origin main
```

现在你可以按照这个指南提交你的代码更新了！
