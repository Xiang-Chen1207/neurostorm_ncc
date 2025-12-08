# MAE预训练实现总结 / MAE Pretraining Implementation Summary

## 中文说明

### 实现内容

本次实现为NeuroSTORM fMRI基础模型添加了完整的MAE（Masked Autoencoder，掩码自编码器）预训练功能，支持从data.txt文件加载自定义fMRI数据。

### 主要文件

1. **核心代码**
   - `models/neurostorm.py` - 修复了NeuroSTORMMAE类中的bug
   - `datasets/custom_mae_dataset.py` - 自定义MAE数据集类
   - `utils/data_module.py` - 添加了CustomMAE数据集支持
   - `main.py` - 添加了CustomMAE到数据集选项

2. **训练脚本**
   - `train_mae_custom.sh` - Shell训练脚本（推荐）
   - `train_mae_custom.py` - Python训练脚本（更灵活）

3. **配置和数据**
   - `data.txt` - 数据文件路径列表模板

4. **文档**
   - `MAE_PRETRAINING_README_CN.md` - 中文详细文档
   - `MAE_PRETRAINING_README.md` - 英文详细文档

5. **验证工具**
   - `validate_mae_setup.py` - 验证安装的脚本

### 快速使用指南

#### 步骤1: 准备数据列表
在`data.txt`中添加您的fMRI数据文件路径，每行一个：

```text
/path/to/subject001_96x96x96x200.npz
/path/to/subject002_96x96x96x200.npy
/path/to/subject003_96x96x96x200.pt
```

#### 步骤2: 运行训练

**方式1 - 使用Shell脚本（最简单）:**
```bash
bash train_mae_custom.sh 8  # 8是batch size
```

**方式2 - 使用Python脚本（更灵活）:**
```bash
python train_mae_custom.py \
  --data_txt_path data.txt \
  --batch_size 8 \
  --mask_ratio 0.5 \
  --sequence_length 40 \
  --img_size 96 96 96 40
```

**方式3 - 如果数据是96x96x96x200:**
```bash
python train_mae_custom.py \
  --sequence_length 50 \
  --img_size 96 96 96 50 \
  --batch_size 8
```

### 关键参数说明

- `--mask_ratio 0.5`: 掩码50%的数据用于重建（推荐0.5-0.8）
- `--spatial_mask window`: 窗口级掩码（推荐）或`random`随机掩码
- `--sequence_length 40`: 使用40个时间点（根据您的数据调整）
- `--img_size 96 96 96 40`: 目标数据维度
- `--batch_size 8`: 批量大小（根据GPU内存调整）

### 支持的数据格式

- `.npz` - NumPy压缩格式
- `.npy` - NumPy数组格式  
- `.pt` - PyTorch张量格式
- `.nii.gz` / `.nii` - NIfTI医学影像格式

数据要求: 4D格式 `(H, W, D, T)`，例如 `(96, 96, 96, 200)`

### 输出位置

训练好的模型保存在:
```
output/neurostorm/custom_pt_neurostorm_mae0.5/
├── checkpt-epoch=XX-valid_loss=Y.YY.ckpt  # 最佳模型
├── last.ckpt                               # 最新检查点
└── events.out.tfevents.*                   # TensorBoard日志
```

查看训练过程:
```bash
tensorboard --logdir output/neurostorm/custom_pt_neurostorm_mae0.5/
```

---

## English Summary

### What Was Implemented

This implementation adds complete MAE (Masked Autoencoder) pretraining functionality to the NeuroSTORM fMRI foundation model, supporting custom fMRI data loading from data.txt file.

### Main Files

1. **Core Code**
   - `models/neurostorm.py` - Fixed bugs in NeuroSTORMMAE class
   - `datasets/custom_mae_dataset.py` - Custom MAE dataset class
   - `utils/data_module.py` - Added CustomMAE dataset support
   - `main.py` - Added CustomMAE to dataset options

2. **Training Scripts**
   - `train_mae_custom.sh` - Shell training script (recommended)
   - `train_mae_custom.py` - Python training script (more flexible)

3. **Configuration and Data**
   - `data.txt` - Data file path list template

4. **Documentation**
   - `MAE_PRETRAINING_README_CN.md` - Detailed Chinese documentation
   - `MAE_PRETRAINING_README.md` - Detailed English documentation

5. **Validation Tools**
   - `validate_mae_setup.py` - Setup validation script

### Quick Start Guide

#### Step 1: Prepare Data List
Add your fMRI data file paths to `data.txt`, one per line:

```text
/path/to/subject001_96x96x96x200.npz
/path/to/subject002_96x96x96x200.npy
/path/to/subject003_96x96x96x200.pt
```

#### Step 2: Run Training

**Method 1 - Using Shell Script (Easiest):**
```bash
bash train_mae_custom.sh 8  # 8 is batch size
```

**Method 2 - Using Python Script (More Flexible):**
```bash
python train_mae_custom.py \
  --data_txt_path data.txt \
  --batch_size 8 \
  --mask_ratio 0.5 \
  --sequence_length 40 \
  --img_size 96 96 96 40
```

**Method 3 - If data is 96x96x96x200:**
```bash
python train_mae_custom.py \
  --sequence_length 50 \
  --img_size 96 96 96 50 \
  --batch_size 8
```

### Key Parameters

- `--mask_ratio 0.5`: Mask 50% of data for reconstruction (recommend 0.5-0.8)
- `--spatial_mask window`: Window-level masking (recommended) or `random`
- `--sequence_length 40`: Use 40 time points (adjust based on your data)
- `--img_size 96 96 96 40`: Target data dimensions
- `--batch_size 8`: Batch size (adjust based on GPU memory)

### Supported Data Formats

- `.npz` - NumPy compressed format
- `.npy` - NumPy array format
- `.pt` - PyTorch tensor format
- `.nii.gz` / `.nii` - NIfTI medical imaging format

Data requirement: 4D format `(H, W, D, T)`, e.g., `(96, 96, 96, 200)`

### Output Location

Trained models are saved to:
```
output/neurostorm/custom_pt_neurostorm_mae0.5/
├── checkpt-epoch=XX-valid_loss=Y.YY.ckpt  # Best model
├── last.ckpt                               # Latest checkpoint
└── events.out.tfevents.*                   # TensorBoard logs
```

View training progress:
```bash
tensorboard --logdir output/neurostorm/custom_pt_neurostorm_mae0.5/
```

---

## Technical Details / 技术细节

### MAE Architecture / MAE架构

The implementation uses a window-based masking strategy:
1. **Patch Embedding**: 4D fMRI volumes are divided into patches
2. **Random Masking**: ~50% of patches are masked
3. **Encoder**: Unmasked patches are encoded using Swin Transformer
4. **Decoder**: Reconstructs the original signal at masked locations
5. **Loss**: MSE loss computed only on masked patches

实现使用基于窗口的掩码策略：
1. **Patch嵌入**: 4D fMRI体素被分成patches
2. **随机掩码**: 约50%的patches被掩码
3. **编码器**: 使用Swin Transformer编码未掩码的patches
4. **解码器**: 在掩码位置重建原始信号
5. **损失**: 仅在掩码patches上计算MSE损失

### Bug Fixes / Bug修复

Fixed critical bug in `models/neurostorm.py`:
- Line 1237-1240: Changed `x_windows` to `x_patch` for random masking
- Removed debug breakpoint

修复了`models/neurostorm.py`中的关键bug：
- 第1237-1240行: 将`x_windows`改为`x_patch`用于随机掩码
- 移除了调试断点

### Code Quality / 代码质量

- Passed code review ✓
- No duplicate code ✓
- Follows existing patterns ✓
- Comprehensive error handling ✓

---

## For Further Help / 获取更多帮助

- See `MAE_PRETRAINING_README.md` for detailed English documentation
- See `MAE_PRETRAINING_README_CN.md` for detailed Chinese documentation
- Run `python validate_mae_setup.py` to check your setup
- Check existing checkpoint files: `pt_fmrifound_mae_ratio0.5.ckpt` and `pt_fmrifound_mae_ratio0.8.ckpt` for examples

- 查看 `MAE_PRETRAINING_README.md` 获取详细英文文档
- 查看 `MAE_PRETRAINING_README_CN.md` 获取详细中文文档
- 运行 `python validate_mae_setup.py` 检查您的设置
- 查看现有检查点文件: `pt_fmrifound_mae_ratio0.5.ckpt` 和 `pt_fmrifound_mae_ratio0.8.ckpt` 作为示例
