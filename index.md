# 索引

由 frontmatter 生成，不要手工重排；追加一行没关系，下次会归位。

## 参考 · reference（4 条）

- [Claude 图工程（Graph Engineering）：模式精要与与 DAG 的真实区别](wiki/agents/claude-graph-engineering.md) `draft` — Claude Code dynamic workflows 的图编排模式（节点契约/假边测试/菱形/检查器/loop-until-dry/模型分层/pipeline vs parallel），及批判分析——图论层面就是 DAG，真创新是编排从对话层下沉到代码层（节点烧 token，边免费） · 2026-09-08
- [NVFP4-A16 Blackwell GEMV（TensorRT-Edge-LLM）：SIMT 瘦形状 kernel 设计深挖](wiki/kernels/nvfp4-a16-blackwell-gemv.md) `draft` — TensorRT-Edge-LLM 的 nvfp4A16BlackwellGemv kernel 深挖——M=1 decode 专用 SIMT 路径（M>1 走 tcgen05）；激活 evict_last/权重 no_allocate 缓存二分、硬件 FP4/FP8 转换器、shared memory 中点 pad… · 2026-09-08
- [AWQ 激活感知权重量化：方法要点与实验结论](wiki/quantization/awq-activation-aware-weight-quantization.md) `draft` — AWQ（MLSys 2024 最佳论文）核心方法——1% 显著权重按激活分布识别、逐通道等效缩放保护、免反向传播不过拟合校准集；全规模优于 GPTQ、多模态首次、TinyChat 3.2-3.9× 加速 · 2026-09-08
- [FlashRT Pi0.5 Thor NVFP4 端到端：结果、关键技术与对我方工作的启示](wiki/quantization/flashrt-pi05-thor-nvfp4.md) `draft` — FlashRT 在 NVIDIA Thor SM110 上把 Pi0.5（SigLIP+Encoder FFN+Decoder）全压 NVFP4/INT4，3 视图 e2e p50 49.0→31.7ms（1.54×）；1 视图精度失败诊断为流匹配轨迹分岔；AWQ α=0.8 是全 FFN NVFP4 过门关键；含 … · 2026-09-08
