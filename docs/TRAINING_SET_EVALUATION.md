# Training Set Evaluation for Overfitting Detection

## 功能说明

模型现在会定期评估训练集性能，让您可以对比训练集、验证集和测试集的指标，判断是否存在过拟合。

## 默认行为

- **评估频率**: 每5个epoch评估一次训练集
- **第一个epoch**: 总是评估（epoch 0）
- **后续epochs**: 每5个epoch评估一次（epoch 5, 10, 15, 20...）

## 在训练日志中查看

### TensorBoard

启动 TensorBoard 后，您会看到以下指标：

```bash
tensorboard --logdir output/neurostorm/
```

**回归任务指标**：
```
train_r_squared        # 训练集 R²
valid_r_squared        # 验证集 R²
test_r_squared         # 测试集 R²

train_mse              # 训练集 MSE (normalized)
valid_mse              # 验证集 MSE (normalized)
test_mse               # 测试集 MSE (normalized)

train_mae              # 训练集 MAE (normalized)
valid_mae              # 验证集 MAE (normalized)
test_mae               # 测试集 MAE (normalized)

train_pearson_coef     # 训练集 Pearson相关系数
valid_pearson_coef     # 验证集 Pearson相关系数
test_pearson_coef      # 测试集 Pearson相关系数
```

### 控制台输出

训练时会显示：

```
[TRAIN EVAL] Evaluating training set at epoch 5...
[DDP] Before gathering - Rank 0 has X train samples from Y unique subjects
[DDP] After gathering - Total Z train samples from W unique subjects

TRAIN Set - Detailed Predictions (Regression)
================================================================================
Total samples: Z | Unique subjects: W
Note: Metrics below are in ORIGINAL SCALE for interpretability
      But training and evaluation use NORMALIZED (standardized) values
================================================================================
Subject                                            | Predicted    | True         | Error
--------------------------------------------------------------------------------
50642                                              | 25.3421      | 26.0000      | 0.6579
...
--------------------------------------------------------------------------------
Metrics (Original Scale): MAE=3.2145, MSE=15.8234
Metrics (Normalized):     MAE=0.2134, MSE=0.0856, R²=0.4523
================================================================================

[INFO] Predictions saved to: .../predictions/predictions_train_epoch5.csv
[INFO] Metrics saved to: .../predictions/metrics_train_epoch5.csv
[INFO] MSE (normalized): 0.0856, R² (normalized): 0.4523

[TRAIN EVAL] Completed training set evaluation
```

## 调整评估频率

### 方法1：命令行参数

```bash
# 每个epoch都评估训练集
bash scripts/abide_downstream/train_8gpu_optimized.sh 8 --eval_train_every 1

# 每10个epoch评估一次（减少计算开销）
bash scripts/abide_downstream/train_8gpu_optimized.sh 8 --eval_train_every 10

# 禁用训练集评估（设置为很大的数字）
bash scripts/abide_downstream/train_8gpu_optimized.sh 8 --eval_train_every 999
```

### 方法2：编辑脚本

在训练脚本中修改 `--eval_train_every` 参数：

```bash
# train_8gpu_optimized.sh
python main.py \
  ... \
  --eval_train_every 1  # 改为1表示每个epoch都评估
```

## 判断过拟合

### 正常训练（无过拟合）

```
Epoch   Train R²   Valid R²   Test R²
--------------------------------------
0       0.05       0.04       0.03
5       0.25       0.22       0.21
10      0.42       0.38       0.37
15      0.56       0.52       0.50
20      0.63       0.58       0.57
```

**特征**：
- ✅ 训练集、验证集、测试集R²都在提升
- ✅ 三者差距不大（< 0.1）
- ✅ 验证集和测试集接近

### 轻度过拟合

```
Epoch   Train R²   Valid R²   Test R²
--------------------------------------
0       0.05       0.04       0.03
5       0.28       0.22       0.21
10      0.50       0.38       0.37
15      0.68       0.45       0.43
20      0.78       0.48       0.46
```

**特征**：
- ⚠️ 训练集R²显著高于验证集/测试集（差距 > 0.2）
- ⚠️ 验证集/测试集R²增长放缓
- ⚠️ 建议：增加正则化、减少训练epochs

### 严重过拟合

```
Epoch   Train R²   Valid R²   Test R²
--------------------------------------
0       0.05       0.04       0.03
5       0.35       0.22       0.21
10      0.65       0.30       0.28
15      0.82       0.28       0.26  ← 验证集开始下降
20      0.91       0.25       0.23  ← 继续恶化
```

**特征**：
- ❌ 训练集R²接近1.0（完美拟合训练数据）
- ❌ 验证集/测试集R²停止增长甚至下降
- ❌ 训练集和验证集差距巨大（> 0.5）
- ❌ 必须采取措施：
  - 增加weight_decay
  - 添加dropout
  - 减少训练epochs
  - 增加数据增强

## 应对过拟合的方法

### 1. 增加正则化

```bash
# 增加 weight_decay
--weight_decay 0.05  # 从0.01增加到0.05
```

### 2. 启用数据增强

```bash
--augment_during_training
```

### 3. Early Stopping

观察验证集R²，当连续5个epoch不再提升时停止训练。

### 4. 减少训练epochs

如果发现在epoch 15后开始过拟合，下次只训练到15 epochs。

### 5. 使用Dropout（需要修改模型）

在regression head中添加dropout层。

## 性能影响

### 计算开销

训练集评估需要额外的前向传播：

```
每5个epoch评估一次（默认）:
- 额外时间: ~2-3分钟 per epoch (取决于训练集大小)
- 总影响: 30 epochs × (1/5) × 3分钟 = ~18分钟额外开销
- 相对影响: ~10-15%总训练时间

每个epoch都评估（--eval_train_every 1）:
- 额外时间: 30 epochs × 3分钟 = ~90分钟
- 相对影响: ~50%总训练时间
```

### 建议

- **开发阶段**: 使用 `--eval_train_every 5`（默认），平衡信息量和开销
- **快速实验**: 使用 `--eval_train_every 10`，减少开销
- **最终训练**: 使用 `--eval_train_every 1`，获得完整的训练曲线
- **生产环境**: 使用 `--eval_train_every 999`，禁用（只关注验证集）

## CSV文件输出

每次评估会生成两个文件：

### 1. 预测文件
```
output/.../predictions/predictions_train_epoch5.csv
```

包含：
- subject: 被试ID
- predicted_value: 预测值（原始尺度）
- true_value: 真实值（原始尺度）
- absolute_error: 绝对误差
- predicted_normalized: 预测值（标准化）
- true_normalized: 真实值（标准化）

### 2. 指标文件
```
output/.../predictions/metrics_train_epoch5.csv
```

包含：
- mode: "train"
- epoch: 5
- mse_normalized: MSE（标准化）
- mae_normalized: MAE（标准化）
- r_squared_normalized: R²（标准化）
- pearson_coef: Pearson相关系数
- mse_original: MSE（原始尺度）
- mae_original: MAE（原始尺度）
- num_samples: 样本数

## 示例：完整训练流程

```bash
# 1. 启动训练（每5个epoch评估训练集）
bash scripts/abide_downstream/train_8gpu_optimized.sh 8

# 2. 在另一个终端启动TensorBoard
tensorboard --logdir output/neurostorm/

# 3. 打开浏览器查看 http://localhost:6006

# 4. 观察指标：
#    - SCALARS tab: 查看 train_r_squared vs valid_r_squared vs test_r_squared
#    - 如果train_r_squared >> valid_r_squared，说明过拟合

# 5. 如果发现过拟合，调整参数重新训练：
bash scripts/abide_downstream/train_8gpu_optimized.sh 8 \
  --weight_decay 0.05 \
  --augment_during_training \
  --max_epochs 20  # 减少epochs
```

## 总结

✅ **优点**：
- 清楚看到模型是否过拟合
- 及时发现问题并调整
- 数据保存到CSV，方便后续分析

⚠️ **注意**：
- 会增加10-50%的训练时间（取决于频率）
- 训练集评估时模型处于eval模式（BatchNorm/Dropout关闭）
- 使用DDP时会自动同步所有GPU的结果

🎯 **建议**：
- 首次训练：使用默认设置（每5个epoch）
- 发现异常：改为每个epoch评估（`--eval_train_every 1`）
- 确认无问题后：可以减少评估频率节省时间
