---
id: flashrt-pi05-thor-nvfp4
title: FlashRT Pi0.5 Thor NVFP4 端到端：结果、关键技术与对我方工作的启示
type: reference
summary: FlashRT 在 NVIDIA Thor SM110 上把 Pi0.5（SigLIP+Encoder FFN+Decoder）全压 NVFP4/INT4，3 视图 e2e p50 49.0→31.7ms（1.54×）；1 视图精度失败诊断为流匹配轨迹分岔；AWQ α=0.8 是全 FFN NVFP4 过门关键；含 Hadamard 旋转 INT4、SFA/SFB 零初始化防 NaN、Thor 测量方法论
tags: [quantization, nvfp4, pi05, thor, flashrt, inference, robotics, awq, hadamard, e2e]
status: draft
created: 2026-09-08
updated: 2026-09-08
source: session:oc_039419815ca8fa8923d371b60250cc38
confidence: high
review_after: 2026-10-08
links: [wiki/quantization/awq-activation-aware-weight-quantization.md]
superseded_by:
refuted_because:
---

## 背景

来源：FlashRT 项目文档《Pi0.5 Thor NVFP4 End-to-End Results》（https://github.com/flashrt-project/FlashRT/blob/main/docs/pi05_thor_decoder_fp4_e2e.md，当前结果 2026-08-05）。与我方 π0.5/Thor NVFP4 工作（正式包 pi05_g1d_nvfp4_ffn_p1_se14_f3）是同一问题的平行实现，可直接对标。

## 头条结果（同会话 A/B，FA4 两边固定，加速只归因 NVFP4）

| 视图 | FP8 p50 | NVFP4+FA4 p50 | 加速 | 精度门 |
|---:|---:|---:|---:|---|
| 1 | 32.92 ms | 23.01 ms | 1.431× | fail（worst cos 0.965） |
| 2 | 38.70 ms | 27.17 ms | 1.424× | pass（raw worst 0.998） |
| 3 | 49.02 ms | 31.74 ms | 1.544× | pass（raw worst 0.9977） |

延迟覆盖完整 `infer()`：图像预处理上传、SigLIP、encoder、18 层 decoder × 10 去噪步、CUDA Graph 回放、同步、动作下载、后处理。环境：Thor CC 11.0 MAXN、Torch 2.10.0 + CUDA 13.0、8 条 LIBERO 观测 p99.9 校准、匹配噪声种子、FP8/FP4 独立进程。

## 技术栈构成

1. Decoder 全投影 NVFP4（qkv/o/gate_up/down，v10 窄 N tile 128×64×256，M=10），权重从 FP16 safetensors 一次性量化，无 FP8 反量化路径。
2. SigLIP FFN **全 27 层** NVFP4 + **AWQ α=0.8**（量化 Up 权重）——无 AWQ 时 16 层以上就把 worst cosine 压破 0.995 门。
3. Encoder FFN 17 层 NVFP4 + AWQ；attention O 投影 NVFP4；**QKV 保持 FP8**（4-bit Q/K 破门，AWQ 能救但 encoder 序列长度下 QKV GEMM 是 compute-bound，量化不赚反亏 ~0.33ms/帧，默认关）。
4. **半宽融合 GeGLU epilogue**（encoder+decoder 默认）：gate/up 交错单 GEMM，visitor 内算 gelu(gate)*up 并直接量化写 FP4+SFA——FFN 隐状态只量化一次而非两次，3v 再省 2ms 且精度更好。
5. 向量化 LayerNorm/RoPE/量化 kernel（寄存器驻留单遍，LayerNorm-to-FP8 2.16×，逐字节一致）。

## 关键发现

**① 1 视图精度失败 = 流匹配轨迹分岔，非实现缺陷**
- 夹爪符号翻转旧描述不成立（夹爪维度 cosine +1.0000）；失败样本是动作轨迹主平移分量整体换向。
- 决定性证据：全 FP4 栈失败样本 A（0.843）；encoder 回 FP8 后 A 恢复 0.999 但**另一个样本 B 失败（0.816）**——同族量化配置翻转不同样本 ⇒ 这些观测坐在 flow-matching 速度场决策边界附近，单视图信息下模型自身不确定，任何小扰动推进另一动作盆地。
- 结论：对单一 FP8 参考的逐样本 cosine 对 1v 过严，任务级评测才是有意义裁判。与我方"此前『过门』实为包间相对回归"的发现互相印证。
- 1v 全过配置：encoder FP8 + decoder 旋转全 INT4（E0M3 权重+激活+Hadamard），28.99ms 全门 PASS。

**② Hadamard 旋转 + 均匀 INT4 网格 > E2M1+MSE**
- 每 16 值块 16×16 正交 Hadamard 旋转（shfl_xor 蝶形/寄存器 FWHT，对 GEMM 数学惰性），旋转高斯化块内分布后均匀网格在所有 cosine 指标击败 E2M1-with-MSE（raw min 0.99904，史上最准 decoder 量化）。
- SM110 tcgen05 element-format 字段 value 0 解码**未公开文档的 E0M3 符号-幅度均匀 INT4**（幅度 0..7），无需二进制 patch。

**③ SFA/SFB padding 未清零 → 依赖分配器历史的 NaN**
- tile 交错布局把 K 取整到 64 原子，量化 kernel 只写真实条目；K 非 64 倍数（SigLIP 4320）时 padding 残留分配器垃圾，0x7F/0xFF 解码 UE4M3 NaN 毒化累加器。是否出 NaN 取决于分配器历史（复用已释放权重暂存内存必现，新页隐藏）。修复：六个 SFA/SFB 分配点全部零初始化。

**④ 融合的正确姿势：融合 GEMM 之间的胶水，不融合 GEMM 本身**
- 全 decoder attention 链（QK^T+softmax+AV+量化）单 SIMT kernel：三种调度全部 5-7× 慢——瘦形状下 GEMM 是 tensor-core-bound（cuBLAS ~1µs SIMT 不可逼近），per-(head,row) 网格把 KV cache 重读放大超 L2。
- 全宽 GeGLU epilogue（K 扩展 2× down 投影）是 wash：DRAM 权重带宽 bound，翻倍流权重 1.8× 代价恰好抵消省掉的 combiner；**半宽变体**（紧凑粒度量化、down 保持原 K）才是 2ms 净赢。

**⑤ Thor 测量方法论**
- 锁 GPC/NVD 仍有两个相差 ~3ms 的持续负载时钟 regime（EMC cap 是上限非锁）——只有同跑内比值有意义，跨批次绝对值不可比；用 A/B/A 三明治定漂移（实测 0.01ms 窗口）。
- 20 次 warmup 不够（41.4/29.1 → 38.6/27.3 ms with 300）。
- Cluster-launch tile 在 isolated 与 pipeline 基准间反转（2×1 cluster 隔离最快，管线 +2.2ms；2×2/2×4 +11-14ms）——tile 选择必须过 pipeline 验证。
- 验证契约：匹配噪声种子、独立进程、artifact SHA-256、schema_version、公共 API 与 preset 表用测试锁死防漂移；`--construct load_model`（公共 API）与 `--construct frontend`（探索旋钮）分离，后者结果不算公共 API 数字。

## 与我方工作对照

| 维度 | FlashRT | 我方正式包 f3 |
|---|---|---|
| 3v e2e p50 | 31.74 ms（FP8 基线 49.0） | 61.83 ms（Apex 真实 dump） |
| 精度 | 2v/3v 全过 | §5.2 FAIL（vs Genie cos 0.670） |
| FFN NVFP4 | SigLIP 27 层 + encoder 17 层，带 AWQ α=0.8 | 全 FFN NVFP4（无 AWQ） |

启示：① AWQ 是他们全 FFN NVFP4 过门的关键（对应我方"过门量化配方"下一步，与 AWQ 论文闭环）；② "不同量化配置翻转不同样本"印证我方"包间相对回归"发现——单一参考逐样本 cosine 本身过严；③ 31.74ms@3v 是同硬件同模型存在性证明，值得逐项对标差距（FA4、GeGLU 半宽融合、向量化 norm、v10 tile、时钟 regime）。

## 适用边界

- 结论基于 Pi0.5 + NVIDIA Thor SM110 + Torch 2.10/CUDA 13.0 + FA4 + CUDA Graph + B=1 推理（CFG/批量/导出显式报错，无隐式 FP8 回退）。
- 加速比只在该硬件与该管线形状（M=10 decoder、瘦 attention）下成立；encoder QKV 量化不赚的结论依赖 encoder 序列长度（compute-bound），带宽 bound 的形状另论。
- 1v 轨迹分岔结论针对 flow-matching 去噪 + 单视图观测；多视图或任务级评测下不构成阻塞。
