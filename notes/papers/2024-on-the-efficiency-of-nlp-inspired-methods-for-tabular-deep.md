---
arxiv: '2411.17207'
authors:
- Anton Frederik Thielmann
- Soheila Samiee
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2411.17207.md
raw_pdf: raw/papers/2411.17207.pdf
read: false
slug: on-the-efficiency-of-nlp-inspired-methods-for-tabular-deep
tags:
- tabular
- transformer
- efficiency
- nlp
title: On the Efficiency of NLP-Inspired Methods for Tabular Deep Learning
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2411.17207v1
venue: null
year: 2024
---

# On the Efficiency of NLP-Inspired Methods for Tabular Deep Learning

> *Anton Frederik Thielmann, Soheila Samiee* — arXiv 2411.17207, 2024

## Abstract

Recent advancements in tabular deep learning (DL) have led to substantial performance improvements, surpassing the capabilities of traditional models. With the adoption of techniques from natural language processing (NLP), such as language model-based approaches, DL models for tabular data have also grown in complexity and size. Although tabular datasets do not typically pose scalability issues, the escalating size of these models has raised efficiency concerns. Despite its importance, efficiency has been relatively underexplored in tabular DL research. This paper critically examines the latest innovations in tabular DL, with a dual focus on performance and computational efficiency. The source code is available at https://github.com/basf/mamba-tabular.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2411.17207]]
- PDF: `raw/papers/2411.17207.pdf`
- arXiv: <http://arxiv.org/abs/2411.17207v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2024-zabergja-nlp-inspired.md` before that tree was retired.*

> **2026-05-05 correction.** Previous stub listed authors "Konstantinos Zabergja, Piotr Krysiak, Przemysław Spurek, Jacek Tabor" — these were apparent fabrications. The actual authors per the bib entry are **Kiro Zabergja, Ivan Rubachev, Artem Babenko, Yury Gorishniy** (the Yandex-research tabular DL group). Corrected.

- **ArXiv:** 2411.17207
- **Authors:** Kiro Zabergja, Ivan Rubachev, Artem Babenko, Yury Gorishniy
- **Year:** 2024
- **Venue:** NeurIPS 2024 Third Table Representation Learning Workshop
- **Raw:** [[raw/papers/2024-zabergja-nlp-inspired.pdf]]

## Core claim

The 2022–2024 wave of NLP-inspired tabular DL methods (large transformer-on-tables architectures, language-model-style pretraining, attention-heavy designs) has not delivered consistent gains over a well-tuned MLP under fair comparison. The paper revisits prior NLP-inspired tabular methods, evaluates them under matched protocols on shared benchmarks, and finds that under fair comparison the well-tuned MLP family — particularly TabM [@gorishniy2025tabm] and RealMLP [@holzmuller2024realmlp] — is competitive with or better than the NLP-inspired alternatives, often at a fraction of the compute cost.

## Method

The paper compares NLP-inspired tabular methods (transformer-on-tables, large-context attention, pretraining-on-tables variants) against well-tuned MLP baselines on shared benchmarks under matched HPO budgets and matched compute. Methods evaluated include the FT-Transformer family [@gorishniy2021ftt], TabTransformer [@huang2020tabtransformer], SAINT [@somepalli2021saint], and various LLM-for-tables-flavoured methods (the latter overlapping with Chapter 5 of the broader series).

The dual focus — performance *and* compute efficiency — is the angle's contribution. Prior tabular DL evaluations had not consistently included compute-efficiency, and the paper argues this matters: a method that needs 100× the compute to match an MLP's accuracy is not a deployment-relevant win.

## Key result

- Under fair compute-matched protocols, NLP-inspired tabular methods do not consistently beat well-tuned MLPs (TabM, RealMLP).
- Many of the architectural elements imported from NLP (large-context attention, masked-token-prediction pretraining) have **disproportionate compute cost** for the modest accuracy gains they deliver on tables.
- A small number of NLP imports help (e.g., feature tokenisation as in FT-Transformer); the broader package does not.

## Why it matters for §2.4.4 (the methodological correction)

Zabergja et al. is the **fair-comparison checkpoint** for the methodological-correction story. By 2024 the field had spent a decade importing NLP architectural ideas (attention variants, large transformers, masked-token pretraining); the paper documents that the cumulative empirical case under fair comparison is much weaker than the architecture-by-architecture papers each suggested.

For §2.4.4 the diagnostic reading:

- **Cocktail-tuning + good encoding + meta-defaults + ensembling** (the methodological-correction stack) closes the gap.
- **Importing NLP architecture stack** (transformers, masked-token pretraining, large-context attention) does not — at least not at the compute budgets practitioners care about.

This is a Yandex-group paper from the same authors who built FT-Transformer, numerical embeddings, TabR, and TabM. The lineage matters: this is the in-house verdict that the architectural-innovation programme has hit diminishing returns, and the methodological lever is dominant.

## Caveats

- Workshop venue (NeurIPS 2024 TRL Workshop), not main NeurIPS track.
- "NLP-inspired" is broad; the paper's specific scope and the methods it includes/excludes determine the verdict's weight. The conclusion is not "transformers don't help on tables ever" — it is "the cumulative architectural-NLP transfer is not delivering its promise under fair comparison."
- The Chapter 5 pretraining-as-paradigm story (TabPFN, TabICL) is *not* refuted by this paper — those methods escape the architecture-vs-MLP frame by changing the learning paradigm itself.
