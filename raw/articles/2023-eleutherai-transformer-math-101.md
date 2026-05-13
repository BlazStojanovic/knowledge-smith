---
source: article
url: https://blog.eleuther.ai/transformer-math/
retrieved: 2026-04-23
title: "Transformer Math 101"
author: Quentin Anthony, Stella Biderman, Hailey Schoelkopf
publication: EleutherAI Blog
date: 2023-04-18
license: check
---

# Transformer Math 101

Authors: Quentin Anthony, Stella Biderman, Hailey Schoelkopf. Published 2023-04-18.

## Core training equation

C ≈ τT = 6PD

where C = total FLOPs, τ = aggregate throughput, T = training time, P = parameters, D = dataset tokens. C_forward ≈ 2PD, C_backward ≈ 4PD.

## Compute-optimal allocation

Chinchilla: D = 20P. EleutherAI recommends against fewer than 200B tokens. Select largest feasible model maintaining acceptable inference cost.

## Engineering benchmarks

- GPT-NeoX: 150 TFLOP/s per A100 (standard attention)
- With Flash Attention: 180 TFLOP/s per A100
- Target: ~120 TFLOP/s per A100; below 115 suggests config issues

## Memory requirements

### Inference

Model weights: int8 = 1 byte/param, fp16/bf16 = 2 bytes/param, fp32 = 4 bytes/param. Total inference: ~1.2× model memory.

### Training

| Component | Bytes per parameter |
|---|---|
| Model (mixed precision) | 2 (+ fp32 copy in optimizer) |
| AdamW optimizer | 12 (fp32 copy + momentum + variance) |
| 8-bit optimizer | 6 |
| SGD with momentum | 8 |
| Gradients (fp32) | 4 |
| Gradients (fp16) | 2 |

### Activation memory

Without recomputation: sbhL(10 + 24/t + 5as/ht) bytes
Selective recomputation: sbhL(10 + 24/t) bytes
Full recomputation: 2·sbhL bytes

s = sequence length, b = batch size/GPU, h = hidden size, L = layers, a = attention heads, t = tensor parallel degree.

## Distributed training

**ZeRO stages:**
- ZeRO-1: shard optimizer states → Optimizer Memory / N_GPUs
- ZeRO-2: + shard gradients
- ZeRO-3: + shard parameters

**3D parallelism:**
- Data parallel: distribute data across replicas
- Tensor parallel: split parameter matrices within nodes
- Pipeline parallel: split layers across nodes
- DP degree = N_GPUs / (PP × TP)

ZeRO-3 has high communication overhead; ZeRO-1 preferred with pipeline parallelism.
