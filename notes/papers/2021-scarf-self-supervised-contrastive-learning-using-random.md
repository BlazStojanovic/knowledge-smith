---
arxiv: '2106.15147'
authors:
- Dara Bahri
- Heinrich Jiang
- Yi Tay
- Donald Metzler
created: '2026-05-08'
doi: null
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2106.15147
  raw: '[[raw/papers/md/2021-scarf-self-supervised-contrastive-learning-using-random]]'
  source: http://arxiv.org/abs/2106.15147v2
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2021-scarf-self-supervised-contrastive-learning-using-random.md
raw_pdf: raw/papers/pdf/2021-scarf-self-supervised-contrastive-learning-using-random.pdf
read: false
slug: scarf-self-supervised-contrastive-learning-using-random
tags:
- type/paper
- tabular
- self-supervised
- contrastive
- pretraining
- status/stub
title: 'SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption'
type: note
updated: '2026-05-09'
venue: null
year: 2021
---

# SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption

> *Dara Bahri, Heinrich Jiang, Yi Tay…* — arXiv 2106.15147, 2021

## Abstract

Self-supervised contrastive representation learning has proved incredibly successful in the vision and natural language domains, enabling state-of-the-art performance with orders of magnitude less labeled data. However, such methods are domain-specific and little has been done to leverage this technique on real-world tabular datasets. We propose SCARF, a simple, widely-applicable technique for contrastive learning, where views are formed by corrupting a random subset of features. When applied to pre-train deep neural networks on the 69 real-world, tabular classification datasets from the OpenML-CC18 benchmark, SCARF not only improves classification accuracy in the fully-supervised setting but does so also in the presence of label noise and in the semi-supervised setting where only a fraction of the available training data is labeled. We show that SCARF complements existing strategies and outperforms alternatives like autoencoders. We conduct comprehensive ablations, detailing the importance of a range of factors.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2021-scarf-self-supervised-contrastive-learning-using-random]]
- PDF: [[raw/papers/pdf/2021-scarf-self-supervised-contrastive-learning-using-random.pdf]]
- arXiv: <http://arxiv.org/abs/2106.15147v2>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2022-bahri-scarf.md` before that tree was retired.*

## Core claim

Contrastive self-supervised representation learning, dominant in vision and NLP, can be adapted to tabular data by constructing positive pairs via *random feature corruption* — replacing a random subset of features with values drawn from each feature's empirical marginal distribution. The simple recipe pretrains deep tabular networks usefully across 69 OpenML-CC18 classification datasets, improving fully-supervised accuracy, semi-supervised accuracy, and robustness to label noise — and beats autoencoder-style SSL alternatives.

## Method

SCARF is a SimCLR-style contrastive framework specialised to tables:

1. **Two views per row.** For an input $x$, draw a random feature subset (uniform over features, sampling probability $p$) and replace those features with values sampled from the *training-set marginal* of each chosen feature. Do this twice with independent draws, producing two augmented views $\tilde{x}_1, \tilde{x}_2$.
2. **Encoder.** A shared MLP encoder $\phi$ produces embeddings $z_1 = \phi(\tilde{x}_1), z_2 = \phi(\tilde{x}_2)$.
3. **Projection head.** A non-linear projector $g$ produces normalised vectors that feed into the contrastive loss.
4. **InfoNCE loss.** Pull $g(\phi(\tilde{x}_1))$ and $g(\phi(\tilde{x}_2))$ together; push them away from $g(\phi(\tilde{x}'))$ for other rows in the batch.

After pretraining, the encoder $\phi$ is fine-tuned on the downstream supervised task.

The marginal-sampling corruption is the same trick VIME [@yoon2020vime] uses for its dual pretext tasks; SCARF's contribution is showing it works equally well as a *contrastive* augmentation, without needing reconstruction or mask-estimation losses.

## Key result

- **69 real-world tabular classification datasets** from OpenML-CC18.
- SCARF pretraining improves classification accuracy in the **fully-supervised** setting.
- **Robust to label noise**: pretraining helps performance more when labels are corrupted.
- **Semi-supervised gains**: the pretrained encoder transfers usefully to low-label regimes.
- **Beats autoencoder baselines**: contrastive views > reconstruction loss for tabular SSL.
- ICLR 2022 Spotlight (top ~5% of accepted papers).

## Why it matters for §2.4.5 (single-table SSL limit)

SCARF is the canonical *contrastive* tabular SSL paper, complementing VIME [@yoon2020vime]'s reconstruction-based and TabNet [@arik2021tabnet]'s mask-estimation framings. By 2022, the SSL-on-tables programme had three branches:

- **Reconstruction / mask estimation** (VIME, TabNet's SSL component, SubTab [@ucar2021subtab]).
- **Contrastive / pull-augmentations-together** (SCARF, STab [@hajiramezanali2022stab] — augmentation-free contrastive).
- **Cross-table transfer** (TransTab [@wang2022transtab], later XTab and the LLM-for-tables line in Chapter 5).

SCARF's importance for §2.4.5: it shows the contrastive SSL recipe ports cleanly to tables when augmentation is done via marginal-sampling, but the practical transfer remains modest. Rubachev et al.'s [@rubachev2022pretraining] subsequent fair-protocol benchmarking finds that SCARF-style and VIME-style gains are real but small and sensitive to tuning — a single-table-SSL ceiling. The Chapter 5 foundation-model story is the response to that ceiling: pretrain on a *different corpus* (synthetic SCMs, real-table aggregation) rather than on a single table.

## Caveats

- 69 datasets is a moderately broad benchmark, but the gains' magnitude is moderate; SCARF doesn't dethrone tuned GBDTs in fully-supervised settings.
- "Beats autoencoders" is at matched depth/width; cocktail-tuned MLPs [@kadra2021welltuned] are a separate stronger baseline that SCARF is not always above.
- Marginal-sampling augmentation breaks joint structure between features; on tables where joint structure matters (highly correlated columns, tight constraints), this corruption may hurt more than help.
