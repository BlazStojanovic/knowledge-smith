---
arxiv: '2605.15514'
authors:
- Yufeng Du
- Phillip Harris
- Minyang Tian
- Eliu A Huerta
- Srikanth Ronanki
- Subendhu Rongali
- Aram Galstyan
- Hao Peng
created: '2026-05-18'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.15514
  raw: '[[raw/papers/md/2026-rope-distinguishes-neither-positions-nor-tokens-in-long-contexts-provably]]'
  source: https://arxiv.org/abs/2605.15514
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-rope-distinguishes-neither-positions-nor-tokens-in-long-contexts-provably.md
raw_pdf: raw/papers/pdf/2026-rope-distinguishes-neither-positions-nor-tokens-in-long-contexts-provably.pdf
read: false
slug: rope-distinguishes-neither-positions-nor-tokens-in-long-contexts-provably
tags:
- type/paper
- status/stub
- long-context
- positional-encoding
- transformer
- attention
- theory
title: RoPE Distinguishes Neither Positions Nor Tokens in Long Contexts, Provably
type: note
updated: '2026-05-18'
year: 2026
---

# RoPE Distinguishes Neither Positions Nor Tokens in Long Contexts, Provably

> *Yufeng Du, Phillip Harris, Minyang Tian, Eliu A Huerta, Srikanth Ronanki, et al.* — arXiv 2026

## TL;DR

Theory paper proving RoPE attention breaks down as context length grows, with bounds that depend only on length, not content. Two failures: (1) **locality bias is lost** — RoPE becomes no more likely to favour nearer positions than far ones; (2) **token-relevance consistency is lost** — a key that out-scores an alternative at one position can under-score it at another. In both, the failure probability approaches 0.5 (random guessing). Attention scores can stay unchanged when a key token is moved or even swapped for a different token — a provable inability to distinguish positions or tokens. Tuning the RoPE base trades the two against each other: raising the base (standard long-context practice) helps token discrimination but sacrifices positional discrimination — you cannot keep both. Multi-head / multi-layer architectures empirically do not rescue this. Implication: long-context Transformers may need fundamentally new position/order encodings. (Summary from abstract; note unread.)

## Abstract

We identify intrinsic limitations of Rotary Positional Embeddings (RoPE) in Transformer-based long-context language models. Our theoretical analysis abstracts away from the specific content of the context and depends only on its length. We prove that as context length increases, RoPE-based attention becomes unpredictable and loses two properties that are central to its effectiveness. First, it loses its locality bias: RoPE is no more likely to favor nearer positions than substantially farther ones. Second, it loses consistency in token relevance: a key vector that receives a higher attention score than an alternative at one position may receive a lower score at another. In both cases, the probability of failure approaches 0.5, no better than random guessing. We further prove that the attention score can remain unchanged when a key token is moved to a different position, or even replaced by a different token, indicating a failure to distinguish positions or tokens. Adjusting the RoPE base trades off distinguishing positions against distinguishing tokens but cannot preserve both at the same time. Increasing the RoPE base hyperparameter, a common practice in today's long-context models, helps distinguish different tokens, but inevitably sacrifices the ability to distinguish positions. Our empirical analysis shows that multi-head, multi-layer architectures are insufficient to overcome these limitations. Our findings suggest that fundamentally new mechanisms for encoding position and token order may be needed in future Transformer long-context language models.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.15514>
- PDF: [[raw/papers/pdf/2026-rope-distinguishes-neither-positions-nor-tokens-in-long-contexts-provably.pdf]]
- Raw markdown: [[raw/papers/md/2026-rope-distinguishes-neither-positions-nor-tokens-in-long-contexts-provably]]
