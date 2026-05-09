---
arxiv: '2604.16027'
authors:
- Constantinos Karouzos
- Xingwei Tan
- Nikolaos Aletras
created: '2026-05-09'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2026-where-does-output-diversity-collapse-in-post-training.md
raw_pdf: raw/papers/pdf/2026-where-does-output-diversity-collapse-in-post-training.pdf
read: false
slug: where-does-output-diversity-collapse-in-post-training
tags:
- llm
- rlhf
- fine-tuning
title: Where does output diversity collapse in post-training?
type: note
updated: '2026-05-09'
url: https://arxiv.org/abs/2604.16027
venue: null
year: 2026
---

# Where does output diversity collapse in post-training?

> *Constantinos Karouzos, Xingwei Tan, Nikolaos Aletras* — arXiv 2604.16027, 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Post-trained language models produce less varied outputs than their base counterparts. This output diversity collapse undermines inference-time scaling methods that rely on varied samples, and risks homogenizing model outputs on creative and value-laden tasks. Prior work attributes collapse to specific post-training methods, without separating the role of training data composition from the method, or the generation format from the model weights. We trace output diversity through three parallel post-training lineages of Olmo 3, Think (chain-of-thought distillation), Instruct (broad multi-source data), and RL-Zero, across 15 tasks and four text diversity metrics. We find that the location of collapse co-varies with data composition: the Think lineage loses most semantic diversity at supervised fine-tuning, and the effect of DPO is larger in Instruct than in Think. Suppressing chain-of-thought reasoning at inference in Think models drops accuracy on hard tasks, yet leaves answer-level diversity unchanged, showing that the collapse is embedded in the model weights by training data, not imposed by the generation format. Decomposing diversity loss on six verifiable tasks into a quality-control component (removal of incorrect outputs) and a residual component (genuine narrowing among correct outputs) reveals that the split is task-dependent, and Think models retain more correct-answer diversity than Instruct despite collapsing more in aggregate. Our results indicate that diversity collapse is determined during training by data composition and cannot be addressed at inference time alone.

## Notes

(your synthesis)

## Source

- Raw markdown: [[raw/papers/md/2026-where-does-output-diversity-collapse-in-post-training]]
- PDF: [[raw/papers/pdf/2026-where-does-output-diversity-collapse-in-post-training.pdf]]
- arXiv: <https://arxiv.org/abs/2604.16027>
