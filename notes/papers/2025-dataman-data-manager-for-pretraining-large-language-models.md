---
aliases:
- DataMan
- dataman
arxiv: '2502.19363'
authors:
- Ru Peng
- Kexin Yang
- Yawen Zeng
- Junyang Lin
- Dayiheng Liu
- Junbo Zhao (Zhejiang University
- Alibaba / Qwen)
created: 2026-04-23
kind: paper
links:
  code: https://github.com/pengr/DataMan
  paper: https://arxiv.org/abs/2502.19363
  raw: '[[raw/papers/md/2025-dataman-data-manager-for-pretraining-large-language-models]]'
  source: https://arxiv.org/abs/2502.19363
owner: blaz
raw_pdf: raw/papers/pdf/2025-dataman-data-manager-for-pretraining-large-language-models.pdf
read: false
slug: dataman-data-manager-for-pretraining-large-language-models
tags:
- type/paper
- status/draft
- source/primary
- confidential/public-source
- domain/pretraining
- domain/data-mix
title: 'DataMan: Data Manager for Pre-training Large Language Models'
type: note
updated: '2026-05-10'
year: 2025
---

# DataMan: Data Manager for Pre-training Large Language Models

## Citation

- URL: https://arxiv.org/abs/2502.19363
- PDF: https://arxiv.org/pdf/2502.19363
- Authors: Ru Peng, Kexin Yang, Yawen Zeng, Junyang Lin, Dayiheng Liu, Junbo Zhao (Zhejiang University, Alibaba / Qwen)
- Year / venue: 2025-02 arXiv preprint; ICLR 2025
- Raw PDF: [[raw/papers/pdf/2025-dataman-data-manager-for-pretraining-large-language-models.pdf]]

## Core Claim

"Reverse thinking" — prompting LLMs to self-identify which quality criteria benefit their own performance via perplexity anomaly analysis — yields 14 complementary quality criteria and 15 domain types, enabling a small Data Manager model to annotate 447B tokens with multi-dimensional quality ratings that outperform single-score and prior multi-score approaches.

## Key Paper Ideas

- **Reverse thinking**: extract documents from top 2% and bottom 2% PPL from different sources; prompt a Super LLM (GPT-4) to identify reasons for anomalous perplexity. Through iterative refinement, derived 14 quality criteria.
- **14 quality criteria**: Accuracy, Coherence, Creativity, Grammatical Diversity, Knowledge Novelty, Language Consistency, Originality, Professionalism, Semantic Density, Sensitivity, Structural Standardization, Style Consistency, Topic Focus, Overall Score.
- **15 application domains**: Medicine, Finance, Law, etc. — for domain mixing support.
- **Pointwise over pairwise**: argues pointwise rating is more practical at scale than QuRating's pairwise approach. $N$ documents need $N$ ratings vs. $N \times (N-1)$ pairwise comparisons. Bounded pointwise rating error via NDCG-loss relationship.
- **Complementarity of criteria**: quality criteria have low mutual correlation and low correlation with perplexity, confirming they capture orthogonal dimensions.
- **PPL ≠ ICL**: perplexity captures general understanding; ICL performance captures generalization ability. These are not aligned, motivating multi-dimensional assessment.

## Methodology

- Prompt GPT-4 to annotate 35,700 documents with 14 quality ratings (1–5 scale) and domain type. Cost: $13,858.
- Fine-tune a small LLM (DataMan) via text generation + pointwise learning-to-rank loss on the annotation dataset.
- Annotate 447B tokens from SlimPajama corpus.
- Select 30B tokens maximizing source diversity and domain balance; train Sheared-LLaMA-1.3B from scratch.

## Key Results

- Best model (Overall Score $l = 5$) surpasses a model trained with 50% more data using uniform sampling.
- ICL improvement: 0.4%–4.3% over SOTA data selection baseline.
- Instruction-following: all Sample-with-DataMan models surpass existing baseline with win rate 67.1%–78.5%.
- Domain-specific continued pretraining with high-rated domain-specific data further improves domain ICL.
- Quality criteria are complementary: filtering on multiple criteria outperforms filtering on any single criterion.

## Core Concepts

- Existing concepts: [[concepts/multi-property-data-curation]], [[concepts/data-filtering-paradigms]]

## Relevance To Poolside

*Our interpretation.* DataMan's 14-criterion approach directly extends the multi-property curation concept. The "reverse thinking" methodology — using PPL anomalies to discover quality criteria — is an interesting alternative to human-designed criteria. For Poolside, the key question is whether code-specific quality criteria (e.g., reasoning depth, structural correctness) would emerge from a similar reverse-thinking process applied to code corpora.

## Key Follow-Ups / Jumping-Off Points

- Comparison with [[notes/papers/2024-qurating-selecting-high-quality-data-for-training-language-models]] — same experimental setup (1.3B on 30B from SlimPajama), direct comparison possible.
- [[notes/papers/2025-meta-rater-multi-dimensional-data-selection]] — uses DataMan's criteria as input features alongside other signals.
- [[notes/papers/2026-propella-1-multi-property-document-annotation-for-llm-data-curation-at-scale]] — extends to 18 properties with structured categorical outputs rather than scalar ratings.

## Related Notes

- Concepts: [[concepts/multi-property-data-curation]], [[concepts/data-filtering-paradigms]]
- Maps: [[maps/evaluation/point-level]]

## Caveats

- Annotation via GPT-4 creates dependency on a specific teacher model.
- Only validated at 1.3B scale on 30B tokens from SlimPajama.
- Pointwise rating may struggle with fine-grained quality distinctions that pairwise captures (DataMan acknowledges this).
