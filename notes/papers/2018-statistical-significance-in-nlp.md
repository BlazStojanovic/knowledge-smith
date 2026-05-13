---
arxiv: '1809.01448'
authors:
- Dror
- Baumer
- Shlomov
- Reichart (Technion)
created: 2026-04-22
kind: paper
links:
  code: https://github.com/rtmdrr/testSignificanceNLP
  paper: https://arxiv.org/abs/1809.01448
  raw: '[[raw/papers/md/2018-statistical-significance-in-nlp]]'
  source: https://arxiv.org/abs/1809.01448
owner: blaz
raw_pdf: raw/papers/pdf/2018-statistical-significance-in-nlp.pdf
read: false
slug: statistical-significance-in-nlp
tags:
- type/paper
- status/draft
- domain/evals
- domain/general
- source/primary
title: The Hitchhiker's Guide to Testing Statistical Significance in Natural Language
  Processing
type: note
updated: '2026-05-10'
year: 2018
---

# The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing

## Citation

- URL: https://arxiv.org/abs/1809.01448
- Authors: Dror, Baumer, Shlomov, Reichart (Technion)
- Year / venue: 2018 / ACL 2018
- **Raw**: [[raw/papers/pdf/2018-statistical-significance-in-nlp]]

## Core Claim

65% of ACL 2017 experimental papers either omit significance testing or use the wrong test. Proposes a decision-tree protocol for selecting appropriate significance tests based on metric properties, and provides a metric-to-test mapping table.

## Key Paper Ideas

- **Decision tree for test selection**: (1) Is the test statistic distribution known? → parametric. (2) No → is data small? → bootstrap/permutation. (3) No → sampling-free non-parametric (sign test, McNemar's, Wilcoxon).
- **Metric-to-test mapping**: accuracy/recall/UAS/LAS → t-test valid (CLT); precision/F-score → t-test invalid (Yeh 2000), use bootstrap/permutation; BLEU/ROUGE/METEOR → not normally distributed, use bootstrap/permutation; perplexity → Wilcoxon signed-rank.
- **Multiple comparison corrections**: only 3/110 ACL 2017 papers that should have corrected did so. K-Bonferroni estimator for dependent test sets.
- **Paired tests are the NLP default**: comparing two algorithms on the same dataset → paired (matched-pair t-test, paired bootstrap, paired permutation).

## Methodology

Survey of all ACL 2017 long papers (196) and TACL 2017 (37). Classification of significance testing practices. Decision tree and metric-to-test table as normative guidance.

## Key Results

| Finding | Value |
|---|---|
| ACL 2017 papers with significance test | 63/180 (35%) |
| Of those, unnamed or wrong test | 27 |
| Most common test | Student's t-test (17 uses) |
| TACL papers without testing | 15/33 (45%) |
| Papers correcting for multiplicity | 3/110 that should have |

## Core Concepts

- [[concepts/evaluation-variance]] — which test to use is determined by variance structure
- [[maps/model-evaluation/statistical-reliability]] — this paper provides the foundational test-selection protocol

## Relevance To Poolside

Directly applicable to any eval comparison at Poolside. McNemar's for accuracy-based evals (MMLU, ARC, etc.), bootstrap/permutation for BLEU/ROUGE-based evals, and the multiple-comparison concern applies when comparing across the full PRETRAINING_DEFAULT suite.

## Blaz Notes

- 

## Related Notes

- Papers: [[notes/papers/2020-with-little-power]]
- Concepts: [[concepts/evaluation-variance]]
- Maps: [[maps/model-evaluation/statistical-reliability]]

## Caveats

- Does not address statistical power (filled by Card et al. 2020)
- Dependent observations in NLP test sets remain an open problem
- Pre-LLM era — focused on traditional NLP metrics, not pass@k or LLM-as-judge
