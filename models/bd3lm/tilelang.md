# TileLang Flash Attention 实现笔记

本文档记录了在 BD3LM 项目中使用 TileLang 实现 Flash Attention 时遇到的问题和解决方案。

## 概述

TileLang 是一个用于编写高性能 GPU kernel 的 DSL（领域特定语言）。我们使用它实现了支持 BD3LM block diffusion mask 的 Flash Attention。

## 遇到的关键问题

### 1. GEMM 与 Mask 的操作顺序

**错误做法：**
```python
# 先设置 mask 值
for i, j in T.Parallel(block_M, block_N):
    acc_s[i, j] = T.if_then_else(valid, 0, -T.infinity(acc_s.dtype))

# 再计算 GEMM（期望累加到 acc_s）
T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)
```

**问题：** GEMM 可能会覆盖而不是累加到 `acc_s`，导致 mask 值丢失。

**正确做法：**
```python
# 先清空并计算 GEMM
T.clear(acc_s)
T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)

# 再应用 mask
for i, j in T.Parallel(block_M, block_N):
    acc_s[i, j] = T.if_then_else(valid, acc_s[i, j], mask_value)
```

### 2. 使用 `-infinity` 导致的 NaN 问题

**问题场景：**

在 online softmax 算法中，当某个 query block 无法 attend 到某个 KV block 的任何位置时（所有位置都被 mask），会出现 NaN。

**数学原理：**

Online softmax 需要计算缩放因子：
```
scores_scale = exp2(scores_max_prev * scale - scores_max_new * scale)
```

当所有位置都是 `-inf` 时：
1. `scores_max_prev = -inf`（初始值或之前全被 mask）
2. `reduce_max` 在全 `-inf` 输入时返回 `-inf`
3. `scores_max_new = -inf`
4. `scores_scale = exp2(-inf - (-inf)) = exp2(-inf + inf) = exp2(NaN) = NaN`

**IEEE 754 浮点数规则：** `-inf + inf` 是不定形式，结果为 `NaN`。

**BD3LM 中的具体表现：**

在 BD3LM 的 block diffusion mask 中：
- xt 区域的 block 1（位置 64-128）不能 attend 到 xt 区域的 block 0（位置 0-64）
- 所以当处理第一个 KV block（k=0）时，所有位置都被 mask
- 这导致整个 xt block 1 及之后的 block 输出都是 NaN

**解决方案：**

使用有限的大负数代替 `-infinity`：

```python
# scores_max 初始化
T.fill(scores_max, -1e30)  # 而不是 -T.infinity(accum_dtype)

# mask 值
mask_value = -1e9  # 而不是 -T.infinity(acc_s.dtype)
acc_s[i, j] = T.if_then_else(valid, acc_s[i, j], mask_value)
```

这样：
- `max(-1e30, -1e9) = -1e9`（有限值）
- `exp2(-1e30 * scale - (-1e9) * scale)` 会下溢到 0，但不是 NaN
- 所有后续计算都保持数值稳定

### 3. reduce_max 的 clear 参数

**问题：** `T.reduce_max(acc_s, scores_max, dim=1, clear=False)` 的行为可能不符合预期。

**解决方案：** 使用 `clear=True` 并显式管理 running max：

```python
# 保存旧的 max
T.copy(scores_max, scores_max_prev)

# 计算当前 block 的 max（使用临时变量）
T.reduce_max(acc_s, block_max_temp, dim=1, clear=True)

# 显式取 max
for i in T.Parallel(block_M):
    scores_max[i] = T.max(scores_max_prev[i], block_max_temp[i])
```

## 性能结果

修复后的 TileLang 实现：

| Backend | Forward | Backward | Total | Speedup vs SDPA |
|---------|---------|----------|-------|-----------------|
| SDPA | 0.444 ms | 1.401 ms | 1.846 ms | 1.00x |
| FlexAttention | 0.454 ms | 0.791 ms | 1.245 ms | 1.48x |
| **TileLang** | **0.277 ms** | 0.931 ms | 1.208 ms | **1.53x** |

- 数值正确性：forward 和 backward 都通过 `torch.allclose` 测试
- Forward pass 速度最快（0.277ms vs SDPA 的 0.444ms）

## 最佳实践总结

1. **GEMM 后再 mask：** 先计算 Q@K^T，再应用 mask，不要反过来

2. **避免使用 `-infinity`：**
   - 初始化 `scores_max` 用 `-1e30`
   - Mask 值用 `-1e9`
   - 这些值足够大以模拟 "负无穷" 的效果，但避免了 NaN

3. **显式管理 reduce 操作：** 使用 `clear=True` 并手动累积结果，而不是依赖 `clear=False` 的隐式行为

4. **调试技巧：**
   - 先禁用 mask 验证基本逻辑
   - 检查每个 block 的 NaN 分布，找出问题模式
   - 对比小序列（如 128）和大序列（如 1024）的行为差异

## 相关文件

- `tilelang_attention.py`: TileLang Flash Attention 实现
- `test_attention_backends.py`: 测试脚本，对比 SDPA、FlexAttention 和 TileLang
