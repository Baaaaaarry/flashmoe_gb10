# GB10 性能材料更新稿

## 1. Cold Boot: Model Load

- Total model: **80.76 GB** (`8.16 GB dense + 72.6 GB experts`)
- PCIe 6.0 x4: **32 GiB/s** vs PCIe 5.0 x4: **15.75 GiB/s** -> **2.03x faster**
- IO BW utilization: **56%** (measured on GB10)

### 公式

```text
T = Model_Size / (BW * Utilization)
```

- PCIe 6.0 x4: `80.76 / (32 * 0.56) = 4.51 s`
- PCIe 5.0 x4: `80.76 / (15.75 * 0.56) = 9.16 s`
- Speedup = `9.16 / 4.51 = 2.03x`

### 更新结论

- Cold Boot 主要是 **SSD/NVMe -> UMA/DDR** 的装载过程。
- 当前 `56%` 的 IO 利用率来自 GB10 实测。
- 若升级到 `PCIe 6.0 x4 / 32 GiB/s`，Cold Boot 时间可从 **9.16 s** 降到 **4.51 s**。

---

## 2. Prefill: Layer Pipeline IO vs Compute

### 更新基础计算逻辑

- Load 阶段已经把模型参数装入 UMA/DDR；Prefill 不再按“SSD 专家冷加载”建模。
- `Prefill Memory Utility` 使用 **resident model traffic** 反推的有效带宽利用率。
- `GPU Utility` 当前来自 `nvidia-smi GPU-Util`，应理解为 **GPU Busy Proxy**，不是严格 MAC 利用率。
- 小上下文下，Prefill 更受 resident weight bandwidth / fixed cost 影响；长上下文下，更多受 context compute path 影响。

### 更新表格

| Context Length | Prefill TPS GB10 (BW 273 GB w/ 123T) | Memory Utility | GPU Busy Proxy | Prefill TPS Mem BW 273 GB w/120T | Prefill TPS Mem BW 546 GB w/120T |
|---:|---:|---:|---:|---:|---:|
| 128 | 148.98 | 34.43% | 0.00% | 148.98 | 297.96 |
| 1,024 | 379.51 | 10.96% | 5.00% | 376.56 | 571.18 |
| 2,048 | 396.40 | 5.73% | 7.00% | 391.02 | 502.55 |
| 8,192 | 390.78 | 5.64% | 9.90% | 384.65 | 468.33 |
| 65,536 | 336.37 | 4.86% | 87.08% | 328.59 | 337.30 |
| 131,072 | 290.63 | 4.20% | 92.05% | 283.84 | 290.02 |

### 更新结论

- `128` token 下，Prefill 仍明显受 resident bandwidth / fixed cost 主导，带宽翻倍收益接近 **2x**。
- `1K ~ 8K` 区间进入平台，说明 resident 权重带宽项被摊薄，主要看 pipeline 执行效率。
- `64K ~ 128K` 下，长上下文路径开始主导；单纯把带宽从 `273 -> 546 GB/s` 翻倍，收益已很有限。

---

## 3. Decode: Hot Weights in UMA/DDR

### 更新基础计算逻辑

- Load 阶段已经把全部参数加载进 UMA/DDR。
- Decode 不再按 SSD/PCIe miss 建模；`PCIe 7.87 / 15.75 GB/s` 不再是 decode 的主瓶颈。
- Decode active weights / token:

```text
10.97 GiB/token = 8.20 dense + 1.07 shared + 72.56 * 6 / 256 routed
```

- `Memory Utility` 使用 decode active-weight traffic 反推的有效 UMA 带宽利用率。
- `GPU Utility` 当前仍是 **GPU Busy Proxy**，不是严格 MAC 利用率。

### 更新表格

| Context Length | Decode TPS GB10 (BW 273 GB) | Memory Utility | GPU Busy Proxy | Decode TPS GB10 (BW 273 GB w/120T) | Decode TPS GB10 (BW 546 GB w/120T) |
|---:|---:|---:|---:|---:|---:|
| 128 | 16.12 | 64.78% | 8.62% | 16.07 | 28.70 |
| 1,024 | 15.26 | 61.32% | 8.29% | 15.21 | 27.13 |
| 2,048 | 14.58 | 58.59% | 8.13% | 14.54 | 25.85 |
| 8,192 | 14.36 | 57.70% | 94.38% | 14.14 | 17.39 |
| 65,536 | 12.25 | 49.22% | 96.00% | 12.05 | 14.46 |
| 131,072 | 10.73 | 43.12% | 96.00% | 10.55 | 12.44 |

### 更新结论

- Decode 的主瓶颈仍是 **UMA/DDR 权重供给 + 长上下文 KV/state 访问**。
- 短上下文下，若带宽从 `273 -> 546 GB/s` 翻倍，Decode 可接近 **1.8x** 提升。
- 长上下文下，随着 KV/state 与 context path 加重，带宽翻倍收益被压缩到 **15% ~ 20%**。

---

## 4. Performance Summary

### 标题更新

```text
Performance Summary: PCIe 5.0 x4 (15.75 GB/s) vs PCIe 6.0 x4 (32 GB/s) NVMe
+ ds4 on GB10 (273 GB/s / 123 TOPS)
```

### 更新表格

| Scenario | PCIe 5.0 (15.75 GB/s) | PCIe 6.0 (32 GB/s) | Improvement | Bottleneck |
|---|---:|---:|---:|---|
| Cold Boot | 9.16 s | 4.51 s | 2.03x faster | Storage BW (I/O-bound) |
| Prefill C=128 | 148.98 tok/s | 297.96 tok/s | 2.00x | GPU Mem BW + fixed cost |
| Prefill C=2,048 | 396.40 tok/s | 502.55 tok/s | 1.27x | Hybrid (resident BW + compute) |
| Decode @128 ctx | 16.12 tok/s | 28.70 tok/s | 1.78x | UMA weight bandwidth |
| Decode @8K ctx | 14.36 tok/s | 17.39 tok/s | 1.21x | Mixed (memory + context path) |
| Decode @128K ctx | 10.73 tok/s | 12.44 tok/s | 1.16x | Long-context KV/state + compute path |

### 总结

- Cold Boot 受 SSD/NVMe 带宽主导，升级到 PCIe 6.0 x4 近似 **2x** 提升。
- Prefill 在小上下文下受 resident bandwidth / fixed cost 主导；长上下文下越来越偏向 context compute path。
- Decode 在当前实现下不再受 SSD/PCIe miss 主导，而主要受 UMA/DDR 带宽与长上下文 KV/state 路径影响。
