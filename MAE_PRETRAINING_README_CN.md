# fMRI基础模型的MAE预训练

本指南说明如何使用掩码自编码器（MAE）预训练实现为NeuroSTORM fMRI基础模型训练自定义数据。

## 概述

MAE预训练任务通过以下方式学习表征：
1. 随机掩码输入fMRI体素的部分patch
2. 对未掩码的patch进行编码
3. 在掩码位置重建原始信号
4. 仅在掩码patch上计算重建损失

## 数据格式

### 预期输入形状
- 每个数据文件应包含4D fMRI体素: `(H, W, D, T)`
  - H, W, D: 空间维度（高度、宽度、深度）
  - T: 时间点数量（时间维度）
- 示例: `(96, 96, 96, 200)` - 96x96x96体素，200个时间点

### 支持的文件格式
- `.npz` - NumPy压缩数组
- `.npy` - NumPy数组
- `.pt` - PyTorch张量
- `.nii.gz` / `.nii` - NIfTI格式

## 快速开始

### 1. 准备数据文件列表

创建一个`data.txt`文件，每行包含一个数据文件路径：

```bash
# data.txt示例
/path/to/subject_001.npz
/path/to/subject_002.npy
/path/to/subject_003.pt
/path/to/subject_004.nii.gz
```

**重要提示:**
- 使用绝对路径或相对于`--image_path`目录的相对路径
- 以`#`开头的行被视为注释
- 空行会被忽略

### 2. 运行训练

#### 方法A: 使用Shell脚本（推荐）

```bash
# 使用默认设置的基本用法
bash train_mae_custom.sh

# 指定batch size
bash train_mae_custom.sh 12

# 如需要，先使脚本可执行
chmod +x train_mae_custom.sh
```

#### 方法B: 使用Python脚本

```bash
# 基本用法
python train_mae_custom.py --data_txt_path data.txt --batch_size 8

# 使用自定义参数
python train_mae_custom.py \
  --data_txt_path data.txt \
  --batch_size 12 \
  --mask_ratio 0.5 \
  --max_epochs 50 \
  --learning_rate 1e-4 \
  --sequence_length 40 \
  --img_size 96 96 96 40
```

#### 方法C: 直接调用main.py

```bash
python main.py \
  --accelerator gpu \
  --max_epochs 30 \
  --dataset_name CustomMAE \
  --image_path . \
  --batch_size 8 \
  --pretraining \
  --use_mae \
  --spatial_mask window \
  --time_mask random \
  --mask_ratio 0.5 \
  --model neurostorm \
  --sequence_length 40 \
  --img_size 96 96 96 40 \
  --window_size 4 4 4 4 \
  --depth 2 2 6 2 \
  --embed_dim 36 \
  --learning_rate 5e-5 \
  --project_name custom_mae_pretraining
```

## 配置参数

### 关键MAE参数

- `--mask_ratio`: 要掩码的token比例（默认: 0.5）
  - 范围: 0.0到1.0
  - 更高的值 = 更具挑战性的重建任务
  - 推荐: 0.5到0.8

- `--spatial_mask`: 空间掩码策略
  - `window`: 掩码整个窗口（推荐）
  - `random`: 随机掩码单个patch

- `--time_mask`: 时间掩码策略
  - `random`: 跨时间随机掩码
  - `tube`: 掩码时间管

### 数据参数

- `--sequence_length`: 每个样本使用的时间点数（默认: 40）
  - 必须 ≤ 数据的T维度
  - 对于形状为(96,96,96,200)的数据，最多可使用200

- `--img_size`: 目标维度[H W D T]（默认: [96, 96, 96, 40]）
  - 如果维度不匹配，数据将自动调整大小
  - 最后一个值应匹配`--sequence_length`

### 模型参数

- `--embed_dim`: 嵌入维度（默认: 36）
- `--depth`: 每层的块数（默认: [2, 2, 6, 2]）
- `--window_size`: 注意力窗口大小（默认: [4, 4, 4, 4]）
- `--c_multiplier`: 层间通道倍增器（默认: 2）

### 训练参数

- `--batch_size`: 训练批量大小（默认: 8）
  - 根据GPU内存调整
  - 更大的批量 = 更好的训练但需要更多内存

- `--learning_rate`: 学习率（默认: 5e-5）
- `--max_epochs`: 最大训练轮次（默认: 30）
- `--num_workers`: 数据加载工作进程数（默认: 4）

## 适应不同的数据维度

如果您的数据形状为(96, 96, 96, 200):

```bash
python train_mae_custom.py \
  --sequence_length 50 \
  --img_size 96 96 96 50 \
  --batch_size 8 \
  --mask_ratio 0.5
```

对于更长的序列（例如，使用全部200个时间点）:

```bash
python train_mae_custom.py \
  --sequence_length 200 \
  --img_size 96 96 96 200 \
  --batch_size 4 \
  --mask_ratio 0.5
```

## 输出和检查点

训练输出保存到:
```
output/neurostorm/<project_name>/
├── checkpt-epoch=XX-valid_loss=Y.YY.ckpt  # 最佳检查点
├── last.ckpt                               # 最新检查点
└── tensorboard logs/                       # 训练日志
```

查看训练进度:
```bash
tensorboard --logdir output/neurostorm/<project_name>/
```

## CustomMAE数据集实现细节

`CustomMAE`数据集类:
- 从`data.txt`加载文件路径
- 验证文件存在性和数据形状
- 如果文件包含多个时间点，则创建多个样本
- 统一处理不同的文件格式
- 支持自动调整大小到目标维度

主要特性:
- **灵活加载**: 支持多种文件格式
- **错误处理**: 跳过损坏或无效的文件
- **进度跟踪**: 显示加载进度
- **内存高效**: 在训练期间按需加载数据

## 故障排除

### 问题: "data.txt not found"
**解决方案**: 在训练脚本所在目录创建data.txt，或使用`--data_txt_path`指定完整路径

### 问题: "No data files found in data.txt"
**解决方案**: 确保data.txt包含至少一个有效的文件路径，且不全是注释/空行

### 问题: "Out of memory"
**解决方案**: 减少`--batch_size`或`--sequence_length`

### 问题: 数据文件"File not found"
**解决方案**: 
- 在data.txt中使用绝对路径，或
- 设置`--image_path`为包含数据文件的目录，并在data.txt中使用相对路径

### 问题: "Unexpected shape"
**解决方案**: 验证您的数据文件形状为(H, W, D, T)，其中T >= sequence_length

## 高级用法

### 从检查点恢复训练

```bash
python train_mae_custom.py --auto_resume
```

这将自动查找并从最后一个检查点恢复。

### 多种掩码比例

训练不同掩码比例的模型以找到最佳设置:

```bash
# 轻度掩码
python train_mae_custom.py --mask_ratio 0.3 --project_name mae_mask30

# 中度掩码（推荐）
python train_mae_custom.py --mask_ratio 0.5 --project_name mae_mask50

# 重度掩码
python train_mae_custom.py --mask_ratio 0.8 --project_name mae_mask80
```

### 多GPU训练

```bash
# 使用多个GPU
python train_mae_custom.py --gpus 0,1,2,3 --batch_size 32
```

## 文件说明

- `models/neurostorm.py`: 包含NeuroSTORMMAE模型实现
- `datasets/custom_mae_dataset.py`: CustomMAE数据集类
- `train_mae_custom.sh`: Shell训练脚本
- `train_mae_custom.py`: Python训练脚本
- `data.txt`: 数据文件路径列表（需要填写您的数据路径）
- `MAE_PRETRAINING_README.md`: 英文版文档

## 参考

基于NeuroSTORM论文:
- 论文: "Towards a General-Purpose Foundation Model for fMRI Analysis"
- GitHub: https://github.com/CUHK-AIM-Group/NeuroSTORM

MAE架构受以下启发:
- He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022

## 支持和反馈

如有问题或建议，请在GitHub上提交issue。
