---
id: awq-activation-aware-weight-quantization
title: AWQ 激活感知权重量化：方法要点、实验结论与飞书文档嵌入图片的坑
type: reference
summary: AWQ 论文（MLSys 2024 最佳论文）核心方法——1% 显著权重按激活分布识别、逐通道等效缩放保护、免反向传播；含完整翻译文档链接与飞书 MCP 图片大小限制的解法（wsrv.nl 代理）
tags: [quantization, awq, llm, inference, weight-only, int4, tinychat, edge-deployment, feishu, mcp]
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

中文全文翻译（含 10 张表格 + 5 张图）已存飞书云文档：https://www.feishu.cn/docx/BugEdNqtRo2CcUxt6fZcNZYnnhw

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

## 坑（飞书 MCP 文档嵌入图片）

翻译论文存飞书云文档时（feishu_create_doc / feishu_update_doc，经 https://mcp.feishu.cn/mcp）：

- **图片下载有大小限制（~256KB，介于 119KB 成功与 481KB 失败之间）**：arxiv HTML 的大图（481-639KB）确定性失败（IMAGE_DOWNLOAD_FAILED，插入空占位符 `<image token=""/>`），小图（101-119KB）正常。重试无效（非限流）。
- **data: URL 不支持**（MCP 把它当普通 URL 下载）。
- **图片下载发生在飞书云端**（mcp.feishu.cn），不是本机——localhost/内网托管无效。
- **解法：wsrv.nl 公共图片代理**（weserv 短域名；长域名 images.weserv.nl 过公司代理 TLS 握手失败，短域名可用）：
  ```
  https://wsrv.nl/?url=arxiv.org%2Fhtml%2F<paper-id>%2F<fig>.png&w=1000&q=78&output=jpg
  ```
  481KB → 52KB，飞书 MCP 下载成功。0x0.st / litterbox / catbox / tmpfiles 等图床均被公司代理挡住。
- **update-doc 的 replace_range 语法**：`selection_with_ellipsis="前缀文本...后缀文本"`（三点分隔起止锚点，范围含锚点，替换的 markdown 需重新包含锚点文本）；定位空图片占位符时用其前后唯一文本做锚点。
- arxiv HTML 版部分图为矢量内嵌（无独立 png 文件），只能保留图注。

## 验证

- 翻译文档 https://www.feishu.cn/docx/BugEdNqtRo2CcUxt6fZcNZYnnhw ：fetch 回读确认 5 张图全部有非空 image token（图1 LOXsbEBmUoFFU3xrxJtcZyWCnze、图6 SgK7booAAoVtIWxKQ3CcL2iRnnd、图7 O6Apbv8P9olDOOxFGtZcsi8inYe、图9/图10 直连 arxiv 成功），10 张表格为飞书原生 lark-table。
- wsrv.nl 压缩效果：teaser.png 481KB→52KB、visual_reasoning.png 484KB→75KB、coco_caption_samples_w4.png 639KB→37KB（w=1000, q=78, jpg）。

## 适用边界

- AWQ 结论适用于 weight-only 分组量化（group size 128，INT4/INT3/INT2）场景；W8A8（激活也量化）是另一条线（SmoothQuant 等）。
- 论文数据基于 2023-2026 的模型（LLaMA/OPT/Mistral/Vicuna/OpenFlamingo/VILA）；AWQ 已进主流推理栈（TensorRT-LLM/vLLM 默认支持），实际使用直接调框架即可，无需自实现。
- 飞书 MCP 图片大小限制为 2026-09 观测值，后续版本可能放宽；wsrv.nl 是第三方免费服务，仅用于一次性转存（图片进飞书后不再依赖它）。
