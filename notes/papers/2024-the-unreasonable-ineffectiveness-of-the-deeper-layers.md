---
arxiv: '2403.17887'
authors:
- Andrey Gromov
- Kushal Tirumala
- Hassan Shapourian
- Paolo Glorioso
- Daniel A. Roberts
created: '2026-05-09'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2024-the-unreasonable-ineffectiveness-of-the-deeper-layers.md
raw_pdf: raw/papers/pdf/2024-the-unreasonable-ineffectiveness-of-the-deeper-layers.pdf
read: false
slug: the-unreasonable-ineffectiveness-of-the-deeper-layers
tags:
- transformer
- interpretability
- fine-tuning
title: The Unreasonable Ineffectiveness of the Deeper Layers
type: note
updated: '2026-05-09'
url: https://arxiv.org/abs/2403.17887
venue: null
year: 2024
---

# The Unreasonable Ineffectiveness of the Deeper Layers

> *Andrey Gromov, Kushal Tirumala, Hassan Shapourian…* — arXiv 2403.17887, 2024

## TL;DR

(stub — fill in after reading)

## Abstract

How is knowledge stored in an LLM's weights? We study this via layer pruning: if removing a certain layer does not affect model performance in common question-answering benchmarks, then the weights in that layer are not necessary for storing the knowledge needed to answer those questions. To find these unnecessary parameters, we identify the optimal block of layers to prune by considering similarity across layers; then, to "heal" the damage, we perform a small amount of finetuning. Surprisingly, with this method we find minimal degradation of performance until after a large fraction (up to half) of the layers are removed for some common open-weight models. From a scientific perspective, the robustness of these LLMs to the deletion of layers implies either that current pretraining methods are not properly leveraging the parameters in the deeper layers of the network or that the shallow layers play a critical role in storing knowledge. For our study, we use parameter-efficient finetuning (PEFT) methods, specifically quantization and Low Rank Adapters (QLoRA), such that each of our experiments can be performed on a single 40GB A100 GPU.

## Notes

(your synthesis)

## Source

- Raw markdown: [[raw/papers/md/2024-the-unreasonable-ineffectiveness-of-the-deeper-layers]]
- PDF: [[raw/papers/pdf/2024-the-unreasonable-ineffectiveness-of-the-deeper-layers.pdf]]
- arXiv: <https://arxiv.org/abs/2403.17887>
