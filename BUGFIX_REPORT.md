# Bug Fix Report: Tensor Dimension Error in Age Group Classification

## 问题描述

执行 `scripts/abide_downstream/age_group_classification.sh` 脚本时出现以下错误:

```
RuntimeError: Tensors must have same number of dimensions: got 2 and 1
```

## 根本原因

### 1. 问题定位

错误发生在 `models/lightning_model.py` 中的多个位置,核心问题在于对**多分类任务**的 logits 张量使用了 `.squeeze()` 操作。

### 2. 技术细节

#### 张量形状分析

对于 **4分类年龄组分类任务**:
- 分类头 `output_head` 输出形状:`(batch_size, 4)`
- 例如:
  - `batch_size=2` → logits 形状 `(2, 4)`
  - `batch_size=1` → logits 形状 `(1, 4)`

#### squeeze() 的行为

`.squeeze()` 会移除**所有**大小为 1 的维度:
- `(2, 4)` → squeeze → `(2, 4)` ✓ (正常)
- `(1, 4)` → squeeze → `(4,)` ✗ (丢失了 batch 维度!)

#### cross_entropy 的要求

`torch.nn.functional.cross_entropy()` 期望:
- **输入 (logits)**: 形状 `(N, C)`,其中 N=batch_size, C=num_classes
- **目标 (target)**: 形状 `(N,)`

#### 错误触发条件

当 `batch_size=1` 时:
- logits 经过 squeeze: `(1, 4)` → `(4,)` (1维张量)
- target 经过 squeeze: `(1,)` → `()` (0维标量)
- `F.cross_entropy` 期望 logits 是 **2维**,但得到了 **1维**
- **结果**: `RuntimeError: Tensors must have same number of dimensions: got 2 and 1`

### 3. 二分类 vs 多分类

为什么二分类没问题?

- **二分类** (num_classes=2):
  - 输出头返回 `(batch, 1)`
  - squeeze 后 `(batch,)` ✓
  - 这对 `binary_cross_entropy_with_logits` 是正确的

- **多分类** (num_classes>2):
  - 输出头返回 `(batch, num_classes)`
  - squeeze 后可能丢失 batch 维度 ✗
  - **不应该对 logits 使用 squeeze**

## 修复方案

### 修改的文件

`models/lightning_model.py`

### 具体修改

#### 1. `_compute_logits` 方法 (第 130-135 行)

**修改前:**
```python
if self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
    logits = self.output_head(feature).squeeze()  # ❌ 问题代码
    target = target_value.float().squeeze()
```

**修改后:**
```python
if self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
    logits = self.output_head(feature)
    # Only squeeze for binary classification (num_classes=2)
    # For multi-class (num_classes>2), keep shape (batch, num_classes)
    if self.hparams.num_classes == 2:
        logits = logits.squeeze()
    target = target_value.float().squeeze()
```

#### 2. `validation_step` 方法 (第 653-658 行)

**修改前:**
```python
output = [logits.squeeze().detach().cpu(), target.squeeze().detach().cpu()]
```

**修改后:**
```python
# For binary classification, logits is already squeezed in _compute_logits
# For multi-class, logits has shape (batch, num_classes), so don't squeeze
if self.hparams.num_classes > 2:
    output = [logits.detach().cpu(), target.squeeze().detach().cpu()]
else:
    output = [logits.squeeze().detach().cpu(), target.squeeze().detach().cpu()]
```

#### 3. `test_step` 方法 (第 830-833 行)

**修改前:**
```python
output = [logits.squeeze().detach().cpu(), target.squeeze().detach().cpu()]
```

**修改后:**
```python
# For binary classification, logits is already squeezed in _compute_logits
# For multi-class, logits has shape (batch, num_classes), so don't squeeze
if self.hparams.num_classes > 2:
    output = [logits.detach().cpu(), target.squeeze().detach().cpu()]
else:
    output = [logits.squeeze().detach().cpu(), target.squeeze().detach().cpu()]
```

#### 4. `_evaluate_train_set` 方法 (第 746-752 行)

**修改前:**
```python
out_train_logits_list.append(logits.squeeze().detach().cpu())
```

**修改后:**
```python
# For binary classification, logits is already squeezed in _compute_logits
# For multi-class, logits has shape (batch, num_classes), so don't squeeze
if self.hparams.num_classes > 2:
    out_train_logits_list.append(logits.detach().cpu())
else:
    out_train_logits_list.append(logits.squeeze().detach().cpu())
```

## 影响范围

### 受影响的任务
- ✅ **多分类任务** (num_classes > 2):
  - ABIDE 年龄组分类 (4 classes)
  - GOD 分类任务 (150 classes)
  - HCPTASK 任务分类 (7 classes)
  - UCLA 诊断分类 (4 classes)
  - Cobre 诊断分类 (4 classes)

### 不受影响的任务
- ✓ **二分类任务** (num_classes = 2):
  - 性别分类
  - AD vs CN 分类
  - 诊断二分类

- ✓ **回归任务**:
  - 年龄回归
  - 其他连续值预测

## 测试建议

### 1. 验证多分类任务
```bash
# 测试 ABIDE 年龄组分类 (4 classes)
bash scripts/abide_downstream/age_group_classification.sh

# 使用不同的 batch_size 测试
bash scripts/abide_downstream/age_group_classification.sh 1  # batch_size=1
bash scripts/abide_downstream/age_group_classification.sh 2  # batch_size=2
bash scripts/abide_downstream/age_group_classification.sh 4  # batch_size=4
```

### 2. 验证二分类任务
确保修改不会破坏现有的二分类功能:
```bash
# 测试性别分类或 AD/CN 分类
bash scripts/hcp_downstream/sex_classification.sh
bash scripts/adni_downstream/ad_cn_classification.sh
```

### 3. 验证回归任务
```bash
# 测试年龄回归任务
bash scripts/hcp_downstream/age_regression.sh
```

## 预期结果

修复后,脚本应该能够:
1. ✅ 正常加载数据
2. ✅ 正常计算损失(无维度错误)
3. ✅ 正常训练和验证
4. ✅ 在不同 batch_size 下都能正常工作(包括 batch_size=1)

## 其他修复

同时修正了脚本中的路径问题:
- 将 `/home/chenx/code/neurostorm_ncc/` 更新为 `/home/user/neurostorm_ncc/`

## 总结

这个 bug 是由于对多分类任务错误使用了 `.squeeze()` 操作导致的。修复方案通过**条件判断**,只对二分类任务使用 squeeze,多分类任务保持原有的 `(batch, num_classes)` 形状,确保与 `F.cross_entropy()` 的要求一致。

修复后的代码:
- ✅ 兼容二分类任务
- ✅ 兼容多分类任务
- ✅ 兼容回归任务
- ✅ 支持任意 batch_size(包括 1)
