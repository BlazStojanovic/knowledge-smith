---
authors:
- Qwen Team
created: '2026-05-12'
kind: paper
links:
  code: null
  paper: https://qianwen-res.oss-accelerate.aliyuncs.com/qwen-scope/Qwen_Scope.pdf
  raw: '[[raw/papers/md/2026-qwen-scope-sparse-features-development-tools]]'
  source: https://qianwen-res.oss-accelerate.aliyuncs.com/qwen-scope/Qwen_Scope.pdf
owner: blaz
parser: read
raw_md: raw/papers/md/2026-qwen-scope-sparse-features-development-tools.md
raw_pdf: raw/papers/pdf/2026-qwen-scope-sparse-features-development-tools.pdf
read: false
slug: qwen-scope-sparse-features-development-tools
tags:
- type/paper
- status/stub
title: 'Qwen-Scope: Turning Sparse Features into Development Tools for Large Language
  Models'
type: note
updated: '2026-05-12'
venue: tech report
year: 2026
---

# Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models

> *Qwen Team* — Alibaba, 2026-04-30

## TL;DR

(stub — fill in after reading)

## Abstract

Large language models have achieved remarkable capabilities across diverse tasks, yet their internal decision-making processes remain largely opaque, limiting our ability to inspect, control, and systematically improve them. This opacity motivates a growing body of research in mechanistic interpretability, with sparse autoencoders (SAEs) emerging as one of the most promising tools for decomposing model activations into sparse, interpretable feature representations. We introduce Qwen-Scope, an open-source suite of SAEs built on the Qwen model family, comprising 14 groups of SAEs across 7 model variants from the Qwen3 and Qwen3.5 series, covering both dense and mixture-of-expert architectures. Built on top of these SAEs, we show that SAEs can go beyond post-hoc analysis to serve as practical interfaces for model development along four directions: (i) inference-time steering, where SAE feature directions control language, concepts, and preferences without modifying model weights; (ii) evaluation analysis, where activated SAE features provide a representation-level proxy for benchmark redundancy and capability coverage; (iii) data-centric workflows, where SAE features support multilingual toxicity classification and safety-oriented data synthesis; and (iv) post-training optimization, where SAE-derived signals are incorporated into supervised fine-tuning and reinforcement learning objectives to mitigate undesirable behaviors such as code-switching and repetition. Together, these results demonstrate that SAEs can serve not only as post-hoc analysis tools, but also as reusable representation-level interfaces for diagnosing, controlling, evaluating, and improving large language models. By open-sourcing Qwen-Scope, we aim to support mechanistic research and accelerate practical workflows that connect model internals to downstream behavior.

## Notes

(stub)

## Source

- Tech report PDF: <https://qianwen-res.oss-accelerate.aliyuncs.com/qwen-scope/Qwen_Scope.pdf>
- Models: <https://huggingface.co/collections/Qwen/qwen-scope>
- PDF: [[raw/papers/pdf/2026-qwen-scope-sparse-features-development-tools.pdf]]
