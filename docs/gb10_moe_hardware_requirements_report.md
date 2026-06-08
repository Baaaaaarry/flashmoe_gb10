# 大参数 MoE 在 GB10 上部署的硬件需求与架构方案说明

## 1. 目标与结论

本文档的目标不是复盘实现细节，而是回答三个更本质的问题：

1. 如果没有 `ds4` 这类专用窄引擎，`DeepSeek-V4-Flash` 这类超大参数 MoE 模型为什么几乎不可能直接部署在 `GB10 120G` 上？
2. `ds4` 是如何通过模型格式、量化、执行图和缓存策略，把“不可部署”变成“可运行、可正确生成、可继续优化”的？
3. `FlashMoE` 的技术路线在这个问题中解决的是哪一部分硬件矛盾，理论上能节省多少内存、降低多少带宽压力，又会引入多少 SSD / I/O 开销？

当前阶段的工程结论可以先明确写成：

- **未量化的大参数 MoE 部署的第一瓶颈是权重容量，其次是权重带宽，不是纯算力。**
- **`ds4` 的核心价值是把模型变成“专用 GGUF + 专用窄引擎 + 激进量化 + 活跃专家执行”。**
- **`FlashMoE` 的核心价值不是重写 dense 主图，而是把 routed experts 从“常驻大块权重”变成“外存 + 热缓存 + 按需加载”，从而继续降低容量与启动预热压力。**

---

## 2. 基础硬件对象：GB10 / DGX Spark

本项目当前验证目标平台是 `GB10`。按 NVIDIA 官方公开资料，`DGX Spark` 的核心硬件特征可概括为：

- `128 GB` 统一系统内存
- 峰值内存带宽约 `273 GB/s`
- FP4 AI 算力级别约 `1 PFLOP`

官方来源：

- [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)

这里要强调两点：

1. `128 GB` 是一个**统一内存预算**，不是“模型显存单独 128G，再加系统内存”。
2. `273 GB/s` 对超大模型来说并不高。对大参数 MoE，**如果权重必须频繁从主存搬运，带宽会比算力更早成为瓶颈**。

---

## 3. 如果没有 ds4，未量化 DeepSeek-V4-Flash 为什么几乎无法部署

### 3.1 当前模型规模

基于当前 `ds4 --inspect` 对 `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` 的输出，模型规模是：

- `logical parameters: 284.33 B`
- `file size: 80.76 GiB`

这两个数字分别表示：

- `284.33 B`：模型的逻辑参数总量
- `80.76 GiB`：量化后的专用 GGUF 文件体积

### 3.2 未量化权重容量

把 `284.33 B` 参数量换算成常见部署格式：

- `BF16`：`284.33B × 2 bytes ≈ 568.66 GB`
- `FP32`：`284.33B × 4 bytes ≈ 1.14 TB`

这意味着：

- **仅权重本身，BF16 就需要约 `568.66 GB`**
- 这还没有算：
  - KV cache
  - runtime workspace
  - graph buffer
  - allocator 碎片
  - 启动预热 cache

与 `GB10 128 GB` 统一内存相比：

- `568.66 / 128 ≈ 4.44x`

所以不使用量化与专用执行图时，**单从容量上就不可能原样部署**。

### 3.3 未量化路径的带宽下界

如果做一个最粗的下界估算：**每生成 1 token，需要扫过一遍有效权重**。

那么：

- BF16 权重体积：`568.66 GB / token`
- 如果目标是 `15 tok/s`
- 对应理论带宽需求下界为：
  - `568.66 × 15 ≈ 8.53 TB/s`

即使这个估算明显偏保守，它也足够说明问题：

- `GB10` 的 `273 GB/s` 带宽与 `TB/s` 级需求差了一个数量级以上

所以未量化部署的真实问题不是：

- “GPU 算力不够”

而是：

- **容量放不下**
- **带宽也拖不动**

### 3.4 算力为什么反而不是第一瓶颈

继续做粗估算：

- `284.33 B` 参数
- 若按每参数 1 次乘加，约记为 `2 FLOPs`
- 则每 token 约为：
  - `284.33B × 2 ≈ 568.66 GFLOPs/token`

若目标 `15 tok/s`：

- `568.66 × 15 ≈ 8.53 TFLOPs`

这个数量级本身并不离谱。与 `GB10` 面向低比特 AI 推理的理论算力相比，**单从 FLOPs 看并没有先撞墙**。

结论是：

- **未量化大参数 MoE 的第一阻塞是权重容量**
- **第二阻塞是权重搬运带宽**
- **算力是第三顺位问题**

---

## 4. ds4 是如何把“不可部署”变成“可部署”的

`ds4` 不是“把原始 HF 模型直接跑起来”，而是做了三层关键处理：

1. **模型格式改造**
2. **激进量化**
3. **专用窄引擎执行**

### 4.1 模型格式：不是原始 checkpoint 直读，而是专用 GGUF

`ds4` 当前部署依赖的是：

- DeepSeek-V4-Flash 专用 GGUF
- 专用 tensor 命名
- 专用布局
- 专用 CUDA / Metal kernel

这意味着它不是通用推理框架，而是：

- **模型固定**
- **执行图固定**
- **张量布局固定**

它的价值在于：

- 避免通用框架的中间态和不必要 materialization
- 让缓存策略、tensor 预热、attention / moe kernel 全部围绕一个模型优化

### 4.2 量化后的体积收益

当前 `inspect` 给出的量化后 tensor 体积分布为：

- `iq2_xxs`: `44.34 GiB`
- `q2_k`: `28.22 GiB`
- `q8_0`: `6.15 GiB`
- `f16`: `2.04 GiB`

总计：

- `80.76 GiB`

与 BF16 全量权重相比：

- `568.66 GB -> 80.76 GiB`

压缩倍率约为：

- `568.66 / 80.76 ≈ 7.04x`

这一步直接把问题从：

- “绝对放不下”

变成：

- “有可能在 `120~128 GB` 统一内存里，通过受控 cache 和运行时策略跑起来”

### 4.3 active experts：并不是每个 token 都计算全部参数

当前模型的 `inspect` 输出还表明：

- `experts: count=256 used=6`

也就是说：

- 每层总 expert 数：`256`
- 每 token 实际活跃 routed experts：`6`

这意味着：

- 总参数量可以非常大
- 但每 token 真正参与 routed MLP 计算的只是其中很小一部分

这为后续 `FlashMoE` 的 routed expert 外存化提供了理论基础。

### 4.4 实际运行层面的 ds4 收益

当前我们已经验证到：

- `ds4 --cpu --inspect` 正常
- `ds4 --cuda --inspect` 正常
- `ds4` 在 GB10 上可以跑出正确语义输出
- 当前 generation 量级大约 `15 tok/s`

这说明 `ds4` 已经完成了：

- 可部署性
- 正确性
- 基础性能闭环

它解决的是“能不能把这个模型放进 GB10 里跑起来”这个问题，而不是已经完成了最终极性能优化。

---

## 5. FlashMoE 方案的硬件意义

### 5.1 FlashMoE 不应该替换整个 ds4 主图

当前最合理的集成方式不是：

- 用 FlashMoE 重写 tokenizer / dense / attention / KV / service

而是：

- **保留 ds4 的主模型图**
- **只替换 routed experts 的存储与数据来源**

因此架构应当是：

- `dense / attention / tokenizer / KV / shared expert`
  - 继续走 ds4 原生主图
- `router`
  - 继续由 ds4 决定当前 token 命中哪些 experts
- `routed experts`
  - 从 ds4 常驻 GGUF resident path 切换到 FlashMoE：
    - manifest
    - external expert packs
    - runtime blob cache
    - hot/cold expert 管理

### 5.2 我们已经验证过的 FlashMoE 技术点

在前期 `gb10_flashmoe` 里，已经验证过这些关键机制：

1. `manifest` 索引
   - 用 `(layer_id, expert_id) -> path/offset/size_bytes` 定位 expert

2. expert pack / blob
   - 多个 expert 可以按层打包进一个外部文件

3. runtime hot cache
   - 第一次命中时读取 expert blob
   - 后续命中从内存缓存复用
   - 超预算时 eviction

4. routed expert 真实数学
   - 不是占位符，而是真实：
     - `gate`
     - `up`
     - `down`
     - `route-weight combine`

5. 路由驱动按需执行
   - 每 token 只对 router 选中的 experts 触发 I/O / unpack / compute

### 5.3 当前 DeepSeek-V4-Flash routed experts 的理论大小

根据当前 `ds4` 固定模型规格：

- hidden size: `4096`
- FF expert dim: `2048`
- experts/layer: `256`
- layers: `43`

每个 routed expert 有 3 个矩阵：

- gate: `4096 × 2048`
- up: `4096 × 2048`
- down: `2048 × 4096`

总参数量：

- `8,388,608 + 8,388,608 + 8,388,608 = 25,165,824 params / expert`

所有 routed experts 总参数：

- `25,165,824 × 256 × 43 = 277,025,390,592`
- 约 `277.03 B params`

这意味着：

- **几乎整个模型的大头都在 routed experts**

如果按 BF16 计：

- `277.03B × 2 bytes ≈ 554.05 GB`

这就是 FlashMoE 技术路线的意义所在：

- **不把 routed experts 常驻，理论上就能绕开数百 GB 级的常驻压力**

### 5.4 当前 quantized routed experts 的总量

按 ds4 当前 expert quant block 规格：

- gate / up：`IQ2_XXS`
- down：`Q2_K`

单 expert 的量化后大小可直接算出来：

#### gate / up

- 输入维度：`4096`
- `QK_K = 256`
- 每行 block 数：`4096 / 256 = 16`
- `IQ2_XXS block = 66 bytes`
- 每行：`16 × 66 = 1056 bytes`
- 每个矩阵：`1056 × 2048 = 2,162,688 bytes`

#### down

- 输入维度：`2048`
- 每行 block 数：`2048 / 256 = 8`
- `Q2_K block = 84 bytes`
- 每行：`8 × 84 = 672 bytes`
- 每个矩阵：`672 × 4096 = 2,752,512 bytes`

#### 单 expert 总量

- `2,162,688 + 2,162,688 + 2,752,512 = 7,077,888 bytes`
- 约 `6.75 MiB / expert`

#### 全部 routed experts 总量

- `7,077,888 × 256 × 43 = 77,950,525,184 bytes`
- 约 `72.60 GiB`

这和当前 GGUF 的 tensor 类型分布是高度一致的：

- `IQ2_XXS + Q2_K ≈ 44.34 + 28.22 = 72.56 GiB`

可以认为：

- **当前模型里约 `72.6 GiB` 的量化权重主要是 routed experts**

这就是 FlashMoE 在现阶段最重要的理论收益：

- **如果 routed experts 不常驻，而改成外存 + 热缓存，理论上就有机会从常驻集里剥离约 `72.6 GiB` 的量化权重**

### 5.5 FlashMoE 对 SSD / I/O 的代价

FlashMoE 不是白送收益，它把一部分 DRAM/统一内存压力转移成了：

- SSD 读取
- expert blob unpack
- runtime cache miss 开销

按当前模型：

- 每个 expert ≈ `6.75 MiB`
- 每层活跃 experts：`6`
- 总层数：`43`

如果极端情况下每层 6 个 experts 全是 cold miss：

- 每 token cold I/O：
  - `6.75 MiB × 6 × 43 ≈ 1.70 GiB/token`

若目标是 `15 tok/s`，则最坏情况下需要：

- `1.70 × 15 ≈ 25.5 GiB/s`

这显然不现实。

因此 FlashMoE 成立的关键前提是：

- **命中率必须足够高**

例如：

- 若 miss ratio = `5%`
- 那么 `15 tok/s` 下的 SSD 带宽压力约：
  - `25.5 × 0.05 ≈ 1.28 GiB/s`

这个量级才开始接近“高性能 NVMe + 顺序读取 + 热点复用”可承受范围。

所以 FlashMoE 的本质 tradeoff 是：

- 用 **热点复用 + 高命中率**
- 来换掉 routed experts 的巨额常驻权重

---

## 6. 面向硬件定义的需求结论

如果目标是定义“大参数 MoE 可跑、高效、极致性能”所需硬件条件，那么可以明确分成三档。

### 6.1 “可跑”级

要求：

- 能正确生成
- 启动可成功
- 不 OOM

核心硬件要求：

- **足够大的统一内存 / 主存**
  - 至少能承载量化后模型 + runtime cache + KV + workspace
- **具备可用的 GPU 加速路径**
- **NVMe 具备可接受的顺序读吞吐**

结论：

- `ds4 + quantized GGUF` 已经把这一档跑通

### 6.2 “高效”级

要求：

- generation 进入双位数 tok/s
- 启动预热与 resident cache 可控

核心硬件要求：

- **更大的有效 resident cache 空间**
- **更高的统一内存带宽**
- **更成熟的 CUDA/Metal 内核与图执行**

结论：

- 这一档目前主要由 `ds4` 的量化 + 窄引擎承担
- FlashMoE 如果把 routed experts 从 resident 集里剥离，会进一步降低常驻压力

### 6.3 “极致性能”级

要求：

- 更接近模型理想 decode 吞吐
- 冷启动和长上下文都要稳

核心硬件要求：

- **更高的总内存容量**
- **更高的统一内存带宽**
- **更高命中率的权重 resident 体系**
- **如果采用 FlashMoE，则需要高命中率 + 高吞吐 NVMe**

换句话说，极致性能不是简单堆算力，而是要求三者一起够强：

- 容量
- 带宽
- cache 命中率 / I/O 体系

---

## 7. 当前工程状态与下一步

### 7.1 当前已经完成的

- `ds4` 上游已能在 GB10 上：
  - 正确加载
  - 正确输出
  - 约 `15 tok/s`

- `flashmoe_integration` 已接入：
  - routed expert 导出
  - `expert_root + manifest`
  - `layer-pack v2`
  - runtime host blob cache
  - routed expert resident preload 剥离
  - prefill 并行流式读
  - prefill GPU staging 复用
  - decode GPU `blob-slot` 常驻缓存
  - decode kernel 直接按 `blob_slots + slot/stride` 消费 external pack

- `inspect` 现在已经能打印：
  - 模型总体体积
  - routed experts 体积
  - dense/non-routed 体积
  - bf16/fp32 等价体积

### 7.2 当前还没有完成的关键点

当前已经不是“控制面接上、CUDA 主路径没接上”的阶段了。  
`FlashMoE` 现在已经真正进入了 CUDA decode / prefill 主路径，但它还没有达到原生 `ds4` 的理想 decode 性能。

当前已经验证成立的关键事实是：

- routed experts 已经从原生 resident 路径中剥离
- 常驻内存显著下降到约 `42 GiB` 量级
- prefill 已经稳定到约 `9 t/s`
- decode 在 direct `blob-slot` 路径下，热启动可到约 `7.4 t/s`

但当前也已经验证出新的主瓶颈：

- **冷启动性能高度受 layer-pack 读取影响**
- **热启动 decode 的主瓶颈已经从 H2D 转移到 `host_pack`**
- `router_readback` 仍然是 decode 的固定成本

也就是说，当前瓶颈已经不再是：

- routed expert 是否还在 GGUF resident path
- 或 H2D `gate/up/down` 整组上传

而是：

- miss expert 的 host 侧组织成本
- 冷态 `SSD -> page cache` 的 direct-read 成本

### 7.3 当前实测口径

在 direct `blob-slot` 路径下，当前应明确区分冷启动与热启动：

- 冷启动：
  - `prefill: 2.36 t/s`
  - `decode: 3.35 t/s`
  - `prefill read ≈ 5616 ms`
  - `decode host_pack ≈ 1744 ms`

- 热启动：
  - `prefill: 8.87 t/s`
  - `decode: 7.42 t/s`
  - `prefill read ≈ 633 ms`
  - `decode host_pack ≈ 435 ms`

这组数据说明：

- 命中率并没有显著变化
- 性能跃迁主要来自：
  - OS page cache 预热
  - GPU slot cache 的热专家复用

所以当前 `FlashMoE` 路线已经形成两档性能口径：

- `cold`
- `warm`

后续任何汇报或 benchmark 都必须明确区分这两者。

### 7.4 下一步真正决定成败的事

如果后续目标是“在保持正确性的前提下，进一步提升 decode，并缩小冷/热差距”，下一步重点必须是：

1. **继续压缩 decode 的 `host_pack`**
   - 当前这是热启动 decode 的最大瓶颈
   - direct `blob-slot` 已经把 `h2d_write` 压低，但没有消除 host 侧 miss expert 组织

2. **继续降低冷态 layer-pack direct-read 成本**
   - 当前冷启动 prefill 仍然主要受 `SSD / page cache` 影响

3. **继续评估 slot 命中收益与容量上限**
   - `DS4_FLASHMOE_DECODE_SLOT_COUNT` 已经支持更大 GPU slot cache
   - 但当前实测表明：单纯扩槽位不是 decode 提升的决定性因素

4. **后续再考虑 predictor / 预取**
   - 只有在现有 direct `blob-slot` 路线稳定后，这类“提前加载 expert”的策略才值得接入
   - 它的主要价值是进一步压 miss path，而不是替代当前 decode 主路径

---

## 8. 最终结论

面向“大参数 MoE 在 GB10 上可跑、高效、极致性能”的硬件定义，可以用一句话总结：

- **未量化大模型的瓶颈首先是容量，其次是带宽，最后才是算力。**
- **`ds4` 通过“量化 + 专用格式 + 窄引擎 + 活跃专家执行”把模型从不可部署变成了可部署。**
- **当前 `FlashMoE` 路线已经证明：routed experts 可以从常驻权重集中剥离出来，并在 `ds4` 主图上把常驻内存压到约 `42 GiB`、把热启动 prefill / decode 分别做到约 `9 t/s` 和 `7.4 t/s`。**
- **但这条路线的冷启动性能仍然明显受 `SSD / page cache / host_pack` 影响。**
- **因此，`FlashMoE` 这条路线是否真正优于原生 resident 方案，最终不只取决于命中率本身，还取决于：能否把 miss path 的 host 组织成本和冷态 I/O 成本继续压低。**
