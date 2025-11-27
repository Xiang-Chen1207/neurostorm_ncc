# ABIDE Dataset Training Scripts

这个目录包含了两个训练脚本，分别用于ABIDE数据集的不同任务。

## 数据准备

确保你的数据目录结构如下：

```
neurostorm_ncc/data/abide/
├── abide_train.txt       # 训练集文件路径列表
├── abide_val.txt         # 验证集文件路径列表
├── abide_test.txt        # 测试集文件路径列表
└── abide.csv             # 标签文件，包含 SUB_ID, AGE_AT_SCAN, age_group 列
```

### 数据处理特点

- **每个被试只使用第一个文件**：如果一个被试有多个.npz文件，只会读取排序后的第一个文件
- **每个文件只读取前20帧**：从每个.npz文件中提取前20帧fMRI图像
- **被试ID提取**：从文件路径中提取被试ID（例如：CMU_a_0050642_func_preproc → 50642）

## 训练脚本

### 1. 年龄回归任务 (Age Regression)

**脚本**: `train.sh`

**任务类型**: 回归 (Regression)

**标签**: 使用 `abide.csv` 中的 `AGE_AT_SCAN` 列（连续值）

**运行方式**:
```bash
# 使用默认batch size (2)
bash scripts/abide_downstream/train.sh

# 或指定batch size
bash scripts/abide_downstream/train.sh 4
```

**输出**:
- 模型权重保存在: `output/neurostorm/abide_ft_neurostorm_age_regression/`
- 指标: MSE (Mean Squared Error), Pearson相关系数 (R²)
- 预测结果CSV: 包含预测年龄和真实年龄

---

### 2. 年龄组别四分类任务 (Age Group Classification)

**脚本**: `train_age_group_classification.sh`

**任务类型**: 四分类 (4-class Classification)

**标签**: 使用 `abide.csv` 中的 `age_group` 列（0, 1, 2, 3）

**运行方式**:
```bash
# 使用默认batch size (2)
bash scripts/abide_downstream/train_age_group_classification.sh

# 或指定batch size
bash scripts/abide_downstream/train_age_group_classification.sh 4
```

**输出**:
- 模型权重保存在: `output/neurostorm/abide_ft_neurostorm_age_group_classification/`
- 指标: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- 预测结果CSV: 包含预测的年龄组别和真实年龄组别

---

## 重要参数说明

### 共同参数

- `--dataset_name ABIDE`: 数据集名称
- `--image_path /home/user/neurostorm_ncc/data/abide`: 数据路径
- `--sequence_length 20`: 每个样本使用20帧
- `--batch_size 2`: 批大小（根据GPU显存调整）
- `--max_epochs 30`: 训练轮数
- `--learning_rate 5e-5`: 学习率
- `--load_model_path`: 预训练模型路径

### 任务特定参数

**回归任务**:
- `--downstream_task_type "regression"`
- `--task_name "age"`

**分类任务**:
- `--downstream_task_type "classification"`
- `--task_name "age_group"`
- `--num_classes 4`

## GPU设置

默认使用GPU 0和1，可以通过修改脚本中的 `CUDA_VISIBLE_DEVICES` 环境变量来调整：

```bash
export CUDA_VISIBLE_DEVICES=0,1  # 使用GPU 0和1
export CUDA_VISIBLE_DEVICES=0    # 只使用GPU 0
```

## 监控训练

训练日志使用TensorBoard记录，可以通过以下命令查看：

```bash
# 回归任务
tensorboard --logdir output/neurostorm/abide_ft_neurostorm_age_regression

# 分类任务
tensorboard --logdir output/neurostorm/abide_ft_neurostorm_age_group_classification
```

## 注意事项

1. **内存要求**: 根据你的GPU显存调整 `--batch_size` 参数
2. **预训练模型**: 确保预训练模型文件存在于指定路径
3. **数据路径**: 确保 txt 文件中的路径是绝对路径且文件存在
4. **被试ID匹配**: CSV文件中的 `SUB_ID` 需要与文件路径中的被试ID对应

## 故障排查

如果遇到问题，请检查：

1. 数据文件是否存在且路径正确
2. CSV文件格式是否正确（包含必要的列）
3. 被试ID是否能从文件路径中正确提取
4. GPU显存是否足够（尝试减小batch_size）
5. 预训练模型文件是否存在
