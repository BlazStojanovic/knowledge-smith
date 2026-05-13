---
arxiv: '2502.10361'
authors:
- Bettina Messmer
- Guilherme Penedo
- Vinko Sabolčec
- Negar Foroutan
- Martin Jaggi
- Leandro von Werra
- Thomas Wolf (HuggingFace)
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2502.10361
  raw: https://arxiv.org/pdf/2502.10361
  source: https://arxiv.org/abs/2502.10361
owner: blaz
raw_pdf: raw/papers/pdf/2025-fineweb-2-hq-enhancing-multilingual-pretraining.pdf
read: false
slug: fineweb-2-hq-enhancing-multilingual-pretraining
tags:
- type/paper
- status/stub
- source/primary
- confidential/public-source
- domain/pretraining
- domain/data-mix
title: Enhancing Multilingual LLM Pretraining with Model-Based Data Selection
type: note
updated: '2026-05-10'
year: 2025
---

# Enhancing Multilingual LLM Pretraining with Model-Based Data Selection

## Citation

- URL: https://arxiv.org/abs/2502.10361
- PDF: https://arxiv.org/pdf/2502.10361
- Authors: Bettina Messmer, Guilherme Penedo, Vinko Sabolčec, Negar Foroutan, Martin Jaggi, Leandro von Werra, Thomas Wolf (HuggingFace)
- Year / venue: 2025-02 arXiv preprint

## Short Summary

Model-based filtering framework for multilingual pretraining datasets. Trains separate FastText and MLP quality scorers per language using diverse multilingual training sources. Comprehensive ablation on FineWeb-2 web crawl data across diverse language families, scripts, and resource availability. Shows that per-language quality filtering substantially outperforms uniform heuristic thresholds, and that model-based selection identifies a diverse set of structured and knowledge-rich samples.

## Open Threads

- How do per-language FastText classifiers compare to multilingual encoder-based classifiers (JQL approach)?
- Does the quality threshold need language-specific tuning or can a universal percentile cutoff work?
- How does model-based selection interact with register-based annotation — do they capture overlapping or orthogonal signals?
