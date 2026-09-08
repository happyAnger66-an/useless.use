---
id: nvfp4-a16-blackwell-gemv
title: NVFP4-A16 Blackwell GEMV（TensorRT-Edge-LLM）：SIMT 瘦形状 kernel 设计深挖
type: reference
summary: TensorRT-Edge-LLM 的 nvfp4A16BlackwellGemv kernel 深挖——M=1 decode 专用 SIMT 路径（M>1 走 tcgen05）；激活 evict_last/权重 no_allocate 缓存二分、硬件 FP4/FP8 转换器、shared memory 中点 padding 防 bank conflict、按精确形状实测的 split-K 策略表、NVRTC JIT + digest 校验
tags: [gemv, nvfp4, blackwell, thor, sm110, cuda, kernel, simt, tensorrt-edgellm, split-k, cache-hint, jit]
status: draft
created: 2026-09-08
updated: 2026-09-08
source: session:oc_039419815ca8fa8923d371b60250cc38
confidence: high
review_after: 2026-10-08
links: [wiki/quantization/flashrt-pi05-thor-nvfp4.md]
superseded_by:
refuted_because:
---

## 背景

代码位置：`/home/nio/codes/ai_infra/r-infra/TensorRT-Edge-LLM`（NVIDIA 官方边缘推理运行时，v0.10.1）。四个组成部分：

- `kernelSrcs/nvfp4A16BlackwellGemv/nvfp4A16BlackwellGemv.cu`（334 行，kernel 本体，JIT 编译源）
- `cpp/kernels/nvfp4A16BlackwellGemv/nvfp4A16BlackwellGemvJitCompiler.{h,cpp}`（NVRTC 编译 + digest）
- `cpp/kernels/nvfp4A16BlackwellGemv/nvfp4A16BlackwellGemvJitRunner.{h,cpp}`（加载/launch/workspace）
- `cpp/plugins/nvfp4A16GemmPlugin/nvfp4A16BlackwellDispatchPolicy.h`（GEMV vs TCGen05 分发）

问题形状：NVFP4 权重（E2M1 + UE4M3 块 scale + 全局 scale 三级）× FP16/BF16 激活 → FP16/BF16 输出，瘦 M。

## 分发规则（何时用 GEMV）

`getNvfp4A16BlackwellDispatch(backend, m, n, k, dtype)`：
- **默认：M==1 → GEMV；M>1 → TCGen05（tensor core）**。显式 backend 覆盖优先。
- M 模板实例 {1,2,4,8,16}，但 M>1 默认不路由到 GEMV——M=1 时 tensor core 无用武之地（纯带宽 bound），SIMT + 硬件 FP4 转换器才是正确工具。
- 非 TMA 可表示形状或非 FP16/BF16 → unsupported（不静默回退）。
- 布局 ABI=1 / 源 ABI=4：权重布局与 tcgen05 路径同族，**同一份量化权重服务两个后端**。

## Kernel 设计要点

**线程组织**（256 线程/块）：8 warp × 每 warp 16 行 = 128 输出行/块（kN_TILE=128）；K-tile 64 元素，`lane/2` 选行内位置、`lane&1` 选 K 半区（各 32 元素）；grid.x = N/128，grid.y = splitK。

**访存缓存策略（精髓，可迁移到任何带宽 bound kernel）**：
```cuda
ld.global.L1::evict_last.v4.u32    // 激活：被 M×(N/128) 块反复读，钉住缓存
ld.global.L1::no_allocate.v4.u32   // 权重：每块只读一次，流式过境不污染缓存
```

**反量化全走硬件转换器**（标量分支梯慢 2.9×，FlashRT 实测）：
```cuda
__nv_cvt_fp4x2_to_halfraw2(..., __NV_E2M1)   // FP4 权重对
__nv_cvt_fp8x2_to_halfraw2(..., __NV_E4M3)   // 块 scale 对
```

**计算流水**（全程 FP32 累加）：
1. 激活转 float2 暂存 shared memory，**中点 +4 padding**（kSMEM_HALF_STRIDE=36、kSMEM_ROW_STRIDE=68）规避 bank conflict
2. 每 16 元素块一个 E4M3 scale：`acc = fmaf(dot2(w,x), scale, acc)`
3. 跨线程归约：`__shfl_xor_sync`（mask=1）合并 K 半区两线程
4. 全局 scale 最后一步乘（NVFP4 三级 scale：global × block × E2M1，块内 FMA → 块 scale → 全局 scale 的应用顺序）

**Split-K**：1/2/4 档；splitK>1 写 FP32 partials workspace + 独立 reduce kernel。M8/M16 有 `_runtime` 变体（splitK 作 kernel 参数，保留 SM110 上"剥首 tile + 稳态循环"调度）；M16 用 `__maxnreg__(88)` 限寄存器。

## Split-K 策略表（SM110 实测调优）

```cpp
// 3 warmups + 10 warm + 10 cold-L2 样本，最小化 warm+cold 中位数
kSharedUpPolicy   {{2,2,2,2,4}};  // N=3712, K=2688（Gemma up 投影）
kSharedDownPolicy {{4,4,4,4,4}};  // N=2688, K=3712（Gemma down）
kLmHeadPolicy     {{1,1,1,1,1}};  // N=131072——1024 个 N-tile 已够填满 SM
kGenericPolicy    {{1,1,1,1,1}};
```

物理逻辑：N=2688 只有 21 个 CTA（Thor ~64 SM 喂不饱），splitK=4 → 84 CTA；LM head N=131072 天然饱和。注释记录调优证据的 SHA256——可复现性实践。

## JIT 体系

- NVRTC 按 (sm, layout, n, k, dtype, sourceAbi) 为 key 编译，`-DGEMV_N/K/DATA_TYPE`
- cubin + key 做 digest 哈希，以 `gemv_jit_bundle` 字段嵌入 TRT engine——运行时零编译（新形状才触发编译）
- digest 校验防止 engine 内 cubin 与 key 不匹配

## 对 Pi0.5 decoder（M=10）的适用边界

- **M=10 不走此 GEMV**（默认 M>1 → tcgen05；M 模板无 10）——瘦 GEMM 对标 tcgen05 路线（FlashRT v10 tile 128×64×256）。
- 可迁移的设计模式：evict_last/no_allocate 缓存二分、shared memory 中点 padding、小 N 时 split-K 填 SM、按精确形状实测调优（非拍脑袋）、硬件转换器、三级 scale 应用顺序。
- 若做 M=1 单步 decode（如 action expert 逐步生成），形状（M=1, N=1024~8192, K=1024~4096）几乎直接对口。
