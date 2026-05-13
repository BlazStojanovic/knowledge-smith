---
tags: [type/index, status/stable, domain/models]
created: 2026-04-22
updated: 2026-05-04
---

# Model Cards

Use [[templates/model-architecture]] for architecture-focused model notes, [[templates/model-card]] for system/safety cards.

Source: [[raw/articles/2026-raschka-big-llm-architecture-comparison]] for initial architecture survey; [[raw/articles/2026-raschka-llm-architecture-gallery]] for ongoing 2026 updates.

![[raw/images/articles/raschka-llm-architecture/figure-01-overview.png]]
*Architecture landscape overview — from Raschka, "The Big LLM Architecture Comparison"*

## Index

```dataview
TABLE WITHOUT ID
  file.link AS "Card",
  year AS "Year",
  created AS "Created",
  updated AS "Updated",
  read AS "Read",
  join(filter(tags, (t) => startswith(t, "model-type/")), ", ") AS "Type"
FROM "notes/model-cards"
WHERE kind = "model-card"
SORT year DESC, file.name ASC
```

## By Architecture Pattern

### Dense

- [[notes/model-cards/2025-olmo-2]] — Allen AI, 7B/13B/32B, post-norm, QK-norm
- [[notes/model-cards/2025-olmo-3]] — Allen AI, 7B/32B, post-norm, sliding window, YaRN
- [[notes/model-cards/2025-gemma-3]] — Google, 1B–27B, hybrid local/global attention
- [[notes/model-cards/2025-gemma-3n]] — Google, mobile-optimized, PLE + MatFormer
- [[notes/model-cards/2026-gemma-4]] — Google, 30.7B, 5:1 sliding/global, p-RoPE, 256K context
- [[notes/model-cards/2025-mistral-small-3-1]] — Mistral, 24B, inference-optimized
- [[notes/model-cards/2025-smollm3]] — Hugging Face, 3B, NoPE
- [[notes/model-cards/2025-kimi-linear]] — Moonshot, 48B, linear attention hybrid

### MoE — Fine-Grained (100+ Experts)

- [[notes/model-cards/2024-deepseek-v3]] — DeepSeek, 671B/37B, MLA, 256 experts
- [[notes/model-cards/2025-deepseek-v3-2]] — DeepSeek, 671B/37B, MLA + sparse attention
- [[notes/model-cards/2025-qwen3-moe]] — Qwen, 30B-A3B / 235B-A22B, 128 experts, no shared
- [[notes/model-cards/2025-qwen3-next]] — Qwen, 80B-A3B, 1024 experts, linear attention hybrid
- [[notes/model-cards/2026-qwen3-6]] — Qwen, 35B-A3B (MoE) + 27B dense, 3:1 DeltaNet/full hybrid, 256 experts
- [[notes/model-cards/2025-kimi-k2]] — Moonshot, 1T, MLA, >256 experts
- [[notes/model-cards/2026-kimi-k2-6]] — Moonshot, 1T-A32B, native multimodal, MLA
- [[notes/model-cards/2026-glm-5-1]] — Zhipu, 744B-A40B, MLA + DeepSeek Sparse Attention, MTP
- [[notes/model-cards/2026-deepseek-v4]] — DeepSeek, 284B-Flash / 1.6T-Pro, CSA/HCA + mHC, 1M context, hash routing

### MoE — Coarse-Grained (<100 Experts)

- [[notes/model-cards/2025-llama-4]] — Meta, 400B/17B, 128 experts, interleaved dense/MoE
- [[notes/model-cards/2025-gpt-oss]] — OpenAI, 20B–120B, 32 experts, attention sinks
- [[notes/model-cards/2025-grok-2-5]] — xAI, 270B, 8 experts, shared expert
- [[notes/model-cards/2025-glm-4-5]] — Zhipu, 355B/106B, dense-first layers, attention bias
- [[notes/model-cards/2025-minimax-m1]] — MiniMax, 456B/46B, linear attention (superseded)
- [[notes/model-cards/2025-minimax-m2]] — MiniMax, 230B, reverted to GQA

### Dense (Standard)

- [[notes/model-cards/2025-qwen3]] — Qwen, 0.6B–32B, deep architectures

### System / Safety Cards

- [[notes/model-cards/2026-claude-mythos-preview-system-card]]
