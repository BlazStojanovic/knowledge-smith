---
source: article
url: https://blog.eleuther.ai/mutransfer/
retrieved: 2026-04-23
title: "The Practitioner's Guide to the Maximal Update Parameterization"
author: Nolan Dey, Quentin Anthony, Joel Hestness
publication: EleutherAI Blog (joint with Cerebras)
date: 2024-09-19
license: check
---

# The Practitioner's Guide to the Maximal Update Parameterization

Authors: Nolan Dey, Quentin Anthony, Joel Hestness. EleutherAI + Cerebras. Published 2024-09-19.

## Core principle

μP controls every operation so outputs do not scale with model width. When width changes, training dynamics remain stable and optimal hyperparameters transfer.

## Four benefits

1. Stable optimal HPs across scale
2. Improved loss at large scale (up to 2× compute savings)
3. Stable training dynamics (fewer NaN / divergence events)
4. Tighter scaling law fits

## Implementation table

| Parameterization | Standard (SP) | μP |
|---|---|---|
| Embedding init variance | σ²_base | σ²_base |
| Embedding LR | η_base | η_base |
| Embedding forward | xW_emb | α_input · xW_emb |
| Hidden init variance | σ²_base | σ²_base / m_d |
| Hidden LR (Adam) | η_base | η_base / m_d |
| Output logit forward | xW_emb^T | α_output · xW_emb^T / m_d |
| Attention logits | Q^TK / √d_head | Q^TK / d_head |

m_d = d / d_base (width multiplier).

## Three operations controlled

1. **Forward**: y = xW → control activation magnitudes via σ_W = σ_base² / m_d
2. **Backward**: ∇_x L = (∇_y L)W^T → same init controls gradient magnitudes
3. **Weight update**: Δy = xΔW → LR scaling η = η_base / m_d ensures updates don't scale with width

## μTransfer protocol

1. **Select proxy model**: hidden size 256, depth matching target
2. **Random HP search**: tune σ_base, η_base, α_input, α_output. Train ~20 tokens/param
3. **Transfer to large scale**: use identical HP values; 1/m_d scaling is automatic
4. **Re-tune when architecture changes**: new attention mechanisms, nonlinearities, position embeddings, vocab size changes

## Coordinate check verification

Train models of varying widths for 10 steps, track mean absolute activation per layer type. Under proper μP: activations remain stable across width changes.

Implementation progression:
1. SP: activations grow with width
2. +μP init (σ²/m_d): controls initial activations
3. +μP LR (η/m_d): controls hidden activations through training
4. +output scaling (1/m_d): controls output scale
5. +attention logit scaling (1/d_head): complete

## Tunable vs fixed

**Tunable**: α_input, α_output, σ_base, η_base
**Fixed (auto-scaled)**: hidden layer LR (1/m_d), hidden init (1/m_d), attention (1/d_head)
**No correction needed**: bias, layer normalization

## Reference implementation

EleutherAI nanoGPT-mup: github.com/EleutherAI/nanoGPT-mup
