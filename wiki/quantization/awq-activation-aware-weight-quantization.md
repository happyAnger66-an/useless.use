---
id: awq-activation-aware-weight-quantization
title: AWQ 激活感知权重量化：方法要点与实验结论
type: reference
summary: AWQ（MLSys 2024 最佳论文）核心方法——1% 显著权重按激活分布识别、逐通道等效缩放保护、免反向传播不过拟合校准集；全规模优于 GPTQ、多模态首次、TinyChat 3.2-3.9× 加速
tags: [quantization, awq, llm, inference, weight-only, int4, tinychat, edge-deployment]
status: draft
created: 2026-09-08
updated: 2026-09-08
source: session:oc_039419815ca8fa8923d371b60250cc38
confidence: high
review_after: 2026-10-08
links: []
superseded_by:
refuted_because:
---

## 背景

论文：*AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration*（arXiv:2306.00978，v6 2026-04-25；MIT 韩松组 + 上海交大 + NVIDIA + 清华；MLSys 2024 **最佳论文奖**）。代码：https://github.com/mit-han-lab/llm-awq

## 核心方法（为什么有效）

1. **权重重要性不均等**：LLM 中存在 0.1%-1% 的显著权重，跳过对它们的量化即可大幅降低量化损失（OPT-6.7B INT3-g128：PPL 23.54 → 11.39）。
2. **显著权重按激活分布识别，不按权重分布**：按权重范数选几乎无效（与随机选择相当）；按激活幅度选，只保留 0.1% 通道为 FP16 就显著有效——幅度大的输入特征更重要，保留对应权重可保住这些特征。
3. **逐通道等效缩放代替混合精度**：混合精度（1% FP16 + 其余 INT）硬件不友好。数学推导：对显著权重 w 乘 s>1、对输入 x 除 s，量化误差变为原来的 (Δ′/Δ)·(1/s) ≈ 1/s（Round 误差期望不变 ~0.25，单元素放大通常不改变组内最大值故 Δ′≈Δ）——显著通道相对误差降低。但 s 过大会放大非显著通道误差（s=4 时 21.2% 通道 Δ′/Δ>1），最优在 s=2 附近（PPL 23.54→11.92）。
4. **缩放搜索空间极简**：s = s_X^α（s_X 为逐通道激活平均幅度），单超参 α 在 [0,1] 网格搜索（20 格）+ 权重裁剪最小化 MSE。目标函数 L(s)=‖Q(W·diag(s))(diag(s)⁻¹X)−WX‖。
5. **免反向传播/免回归**：只从校准集测每通道平均激活幅度 → 不过拟合校准集（GPTQ 的重建过程会过拟合），泛化到指令微调、多模态、编程数学等分布外场景。

## 关键实验结论

- **LLaMA/Llama-2 全规模（7B-70B）一致优于 RTN 和 GPTQ**（含重排序版 GPTQ-R），INT3-g128 和 INT4-g128 均如此；Mistral（GQA）和 Mixtral（MoE）架构同样有效。
- **多模态首次**：OpenFlamingo-9B COCO Captioning，INT4-g128 量化退化（32-shot CIDEr）从 RTN 的 -4.57 降到 **-1.17**（FP16 81.70 → AWQ 80.53）；VILA-7B/13B 在 11 个视觉语言基准上一致无损。
- **数据效率**：比 GPTQ 小 10 倍的校准集（16 vs 192 条序列）即达更好 PPL；换校准分布（PubMed↔Enron）AWQ PPL 只增 0.5-0.6，GPTQ 恶化 2.3-4.9。
- **与 GPTQ 正交**：INT2-g64 极低比特下 AWQ+GPTQ 组合显著优于单独 GPTQ（OPT-6.7B：16.65 → 15.71）。
- **TinyChat 系统**（把 W4A16 理论节省变实测加速）：即时反量化融入矩阵乘 kernel（不写回 DRAM）、SIMD 感知权重打包（ARM NEON 128-bit 寄存器存 32 个 4-bit 权重，3 条 SIMD 指令解包，+1.2×）、算子融合（4090 上单 FP16 kernel ~0.01ms 与启动开销同量级）。实测：4090/Orin 上比 HF FP16 快 3.2-3.9×；8GB 4070 跑 13B @33 tok/s；比 llama.cpp 快至 1.7×；树莓派可跑 7B。
- **加速根因**（端侧 batch=1）：生成阶段 memory-bound（FP16 算术强度≈1），权重访问比激活大几个数量级 → weight-only 4-bit 把算术强度提 4×。
- **工业采用**：HF Transformers、TensorRT-LLM、vLLM、DirectML、Vertex AI、Intel Neural Compressor、SageMaker、AMD、LMDeploy 等；Falcon-180B 单卡 H200。

## 适用边界

- AWQ 结论适用于 weight-only 分组量化（group size 128，INT4/INT3/INT2）场景；W8A8（激活也量化）是另一条线（SmoothQuant 等）。
- 论文数据基于 2023-2026 的模型（LLaMA/OPT/Mistral/Vicuna/OpenFlamingo/VILA）；AWQ 已进主流推理栈（TensorRT-LLM/vLLM 默认支持），实际使用直接调框架即可，无需自实现。
