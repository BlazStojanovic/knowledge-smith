---
arxiv: '2410.24210'
authors:
- Yury Gorishniy
- Akim Kotelnikov
- Artem Babenko
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2410.24210.md
raw_pdf: raw/papers/2410.24210.pdf
read: false
slug: tabm-advancing-tabular-deep-learning-with-parameter
tags: []
title: 'TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2410.24210v3
venue: null
year: 2024
---

# TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling

> *Yury Gorishniy, Akim Kotelnikov, Artem Babenko* — arXiv 2410.24210, 2024

## Abstract

Deep learning architectures for supervised learning on tabular data range from simple multilayer perceptrons (MLP) to sophisticated Transformers and retrieval-augmented methods. This study highlights a major, yet so far overlooked opportunity for designing substantially better MLP-based tabular architectures. Namely, our new model TabM relies on efficient ensembling, where one TabM efficiently imitates an ensemble of MLPs and produces multiple predictions per object. Compared to a traditional deep ensemble, in TabM, the underlying implicit MLPs are trained simultaneously, and (by default) share most of their parameters, which results in significantly better performance and efficiency. Using TabM as a new baseline, we perform a large-scale evaluation of tabular DL architectures on public benchmarks in terms of both task performance and efficiency, which renders the landscape of tabular DL in a new light. Generally, we show that MLPs, including TabM, form a line of stronger and more practical models compared to attention- and retrieval-based architectures. In particular, we find that TabM demonstrates the best performance among tabular DL models. Then, we conduct an empirical analysis on the ensemble-like nature of TabM. We observe that the multiple predictions of TabM are weak individually, but powerful collectively. Overall, our work brings an impactful technique to tabular DL and advances the performance-efficiency trade-off with TabM -- a simple and powerful baseline for researchers and practitioners.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2410.24210]]
- PDF: `raw/papers/2410.24210.pdf`
- arXiv: <http://arxiv.org/abs/2410.24210v3>

<!-- ks-crosslink -->
**Writing-tier note:** [[../papers/2025-gorishniy-tabm]]
