---
arxiv: '2410.24210'
authors:
- Yury Gorishniy
- Akim Kotelnikov
- Artem Babenko
created: '2026-05-08'
doi: null
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2410.24210
  raw: '[[raw/papers/md/2024-tabm-advancing-tabular-deep-learning-with-parameter]]'
  source: http://arxiv.org/abs/2410.24210v3
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2024-tabm-advancing-tabular-deep-learning-with-parameter.md
raw_pdf: raw/papers/pdf/2024-tabm-advancing-tabular-deep-learning-with-parameter.pdf
read: false
slug: tabm-advancing-tabular-deep-learning-with-parameter
tags:
- type/paper
- tabular
- mlp
- distillation
- benchmark
- status/stub
title: 'TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling'
type: note
updated: '2026-05-09'
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

- Raw markdown: [[raw/papers/md/2024-tabm-advancing-tabular-deep-learning-with-parameter]]
- PDF: [[raw/papers/pdf/2024-tabm-advancing-tabular-deep-learning-with-parameter.pdf]]
- arXiv: <http://arxiv.org/abs/2410.24210v3>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2025-gorishniy-tabm.md` before that tree was retired.*

## Core claim

A *single* TabM model imitates an ensemble of MLPs while sharing most parameters across the ensemble members, producing multiple predictions per object at near the cost of one. This parameter-efficient design — when used as a new MLP-based baseline — outperforms attention-based and retrieval-based tabular DL architectures on a large-scale public benchmark. The diagnostic conclusion the paper draws is sharp: the *MLP family* (with proper input encoding from [@gorishniy2022embeddings] and proper ensembling from TabM) is a stronger and more practical line than the attention or retrieval families.

## Architecture

A TabM block contains $k$ implicit MLPs that share *most* weight tensors but differ in a small set of per-member parameters (for example, member-specific scale-and-shift affine corrections after each linear). Forward-passing one input produces $k$ predictions in essentially the cost of one MLP forward pass. Training optimises all $k$ predictions jointly under the task loss; at inference the $k$ outputs are averaged.

The mechanism is a generalisation of *BatchEnsemble* (Wen, Tran, Ba 2020) adapted to MLPs and tabular input encodings. The shared-vs-distinct parameter split is small enough that the ensemble cost stays close to a single MLP, but large enough that the $k$ implicit models de-correlate during training and behave as a meaningful ensemble.

The paper also conducts a large-scale benchmark across 30+ public tabular datasets, using TabM as the new baseline and re-evaluating attention-based (FT-Transformer, SAINT, NPT) and retrieval-based (TabR) architectures under matched protocols.

## Key result

- TabM is the **best-performing tabular DL model** on the paper's public-benchmark evaluation (averaged ranking across datasets).
- The implicit MLPs inside TabM are **individually weak** but **collectively strong** — the paper's empirical analysis confirms that the gain comes from ensemble diversity, not from individual member capacity.
- MLP-family models (RealMLP + TabM) are the strongest line; attention-based (FT-Transformer, SAINT, NPT) and retrieval-based (TabR) architectures finish behind.

## Why it matters for §2.4.4 (the methodological correction)

TabM is the 2025 endpoint of the methodological-correction line:

- **Kadra (2021):** plain MLP + cocktail HPO beats specialised architectures.
- **Gorishniy (2022) embeddings:** MLP + proper numerical embeddings catches up to FT-Transformer.
- **Holzmüller (2024) RealMLP:** MLP + meta-defaults catches up to GBDTs without HPO.
- **Gorishniy (2025) TabM:** MLP + parameter-efficient ensembling beats attention and retrieval architectures.

For §2.4.4 the load-bearing claim is the *cumulative* one: by the time you stack good encodings + good defaults + good ensembling onto a plain MLP, the architectural gap to attention/retrieval models closes or inverts, and the GBDT gap on most public benchmarks closes too. The §2.1/§2.2/§2.3 priors are still real, but the methodological lever is dominant in practice.

A second diagnostic note for §2.4.4: the TabM result reframes the tabular DL landscape. The 2017–2022 architecture proliferation looked, from inside the field, like a search for the right backbone. The 2024–2025 verdict is closer to "the right *training discipline* on a plain MLP wins." Architectural innovation isn't worthless, but it's not the binding constraint either.

## Caveats

- The benchmark is public-dataset focused; production-scale workloads with strong temporal structure (TabReD [@rubachev2025tabred]) shift specific rankings.
- "Best DL model" is contingent on the comparator pool. Strong fair-protocol GBDTs remain competitive, and the paper does not claim universal dominance over GBDTs.
- BatchEnsemble-style ensembling has been known since 2020; the contribution is more "we systematically validated this is the right tabular MLP design" than a brand-new mechanism.
