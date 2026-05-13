---
arxiv: '2310.16789'
authors:
- Shi*
- Ajith*
- Xia
- Huang
- Liu
- Blevins
- Chen
- Zettlemoyer (University of Washington
- Princeton University)
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2310.16789
  raw: '[[raw/papers/md/2024-min-k-percent]]'
  source: https://arxiv.org/abs/2310.16789
owner: blaz
raw_pdf: raw/papers/pdf/2024-min-k-percent.pdf
read: false
slug: min-k-percent
tags:
- type/paper
- source/primary
- status/draft
- domain/evals
title: Detecting Pretraining Data from Large Language Models
type: note
updated: '2026-05-10'
year: 2024
---

# Detecting Pretraining Data from Large Language Models

Reference-free membership inference for LLM pretraining data: select the k% lowest-probability tokens and threshold their average log-likelihood.

## Citation

- URL: https://arxiv.org/abs/2310.16789
- Authors: Shi*, Ajith*, Xia, Huang, Liu, Blevins, Chen, Zettlemoyer (University of Washington, Princeton University)
- Year / venue: 2024 / ICLR 2024
- **Raw**: [[raw/papers/pdf/2024-min-k-percent]]

## Core Claim

An unseen text is likely to contain a few outlier tokens with unusually low model probability, while a seen (trained-on) text is not. Selecting the k% tokens with minimum log-probability and averaging them gives a simple, reference-free membership inference signal (Min-K% Prob) that outperforms prior methods by 7.4% AUC on the authors' WikiMIA benchmark. The method requires only black-box token-level log-probabilities and no reference model or knowledge of the pretraining corpus.

## Key Paper Ideas

- **Min-K% Prob method**: given a token sequence x = x_1, ..., x_N, compute the conditional log-probability log p(x_i | x_1, ..., x_{i-1}) for each token. Select the k% of tokens with the lowest probability (highest negative log-likelihood) to form the set Min-K%(x). The membership score is the average log-likelihood over this set. Higher score (less negative) implies the text was likely in training data. Threshold for binary classification.
- **Intuition**: trained-on text has been "seen" so even its least-likely tokens have relatively high probability. Unseen text has more genuinely surprising outlier tokens, dragging the Min-K% average down.
- **Reference-free**: prior MIA methods (Carlini et al., 2022; Watson et al., 2022) require a shadow/reference model trained on similar data. This is impractical for LLM pretraining where the data distribution is undisclosed and retraining is prohibitively expensive. Min-K% Prob needs only token-level log-probabilities from the target model.
- **WikiMIA benchmark**: dynamic benchmark using Wikipedia event pages. Member data: events from pre-2017 Wikipedia (guaranteed in pretraining corpora of LLaMA, GPT-NeoX, OPT, Pythia). Non-member data: events from post-2023 Wikipedia (guaranteed absent from pretraining). 394 examples per class. Includes original and paraphrase settings, plus length-bucketed evaluation (32, 64, 128, 256 tokens).
- **Detection difficulty factors**: detection becomes easier with (1) larger model size, (2) longer text, (3) higher learning rate, (4) higher occurrence frequency. For outlier contaminants (e.g., downstream benchmark examples inserted into pretraining data), detection also becomes easier with more pretraining data (outlier memorization effect). For in-distribution contaminants, detection becomes harder with more data, matching theoretical expectations.

## Methodology

- **Hyperparameter**: k = 20 (selected from sweep over {10, 20, 30, 40, 50} on held-out validation using LLaMA-65B). Used without further tuning across all experiments.
- **Baselines**: LOSS Attack / PPL (sentence-level perplexity), Neighbor attack (probability curvature, equivalent to DetectGPT), Zlib (perplexity vs zlib compression entropy), Lowercase (perplexity vs lowercased perplexity), Smaller Ref (perplexity ratio between target and smaller model trained on same data).
- **Metrics**: AUC (area under ROC curve), TPR@5%FPR.
- **Models tested on WikiMIA**: Pythia-2.8B, GPT-NeoX-20B, LLaMA-30B, LLaMA-65B, OPT-66B.
- **Case study 1 — Copyrighted book detection**: detect Books3 excerpts in GPT-3 (text-davinci-003). Validation set: 50 known-memorized books (positive) + 50 post-2023 books (negative), 100 snippets of 512 words each. Test set: 100 Books3 books, 100 snippets each.
- **Case study 2 — Downstream contamination**: continual pretraining of LLaMA-7B on RedPajama data purposefully contaminated with 200 examples each from BoolQ, IMDB, TruthfulQA, CommonsenseQA (0.1% of 27M token corpus). Ablation studies on dataset size, occurrence frequency, learning rate.
- **Case study 3 — Machine unlearning audit**: test whether LLaMA2-7B-WhoIsHarryPotter (Eldan & Russinovich, 2023) truly unlearned Harry Potter content. Compare Min-K% Prob scores between original and unlearned model to identify suspicious chunks.

## Key Results

| Setting | Metric | Value |
|---|---|---|
| WikiMIA (128-token, original+paraphrase, 5 models) | Average AUC | 0.72 |
| WikiMIA improvement over best baseline (PPL) | AUC delta | +7.4% |
| Copyrighted book detection (GPT-3, Books3) | AUC | 0.88 |
| Copyrighted book detection — best baseline (PPL) | AUC | 0.84 |
| Books3 test: books with contamination rate > 50% | Fraction | ~90% |
| Downstream contamination (LLaMA-7B, 4 tasks) | Average AUC | 0.86 |
| Downstream contamination — best baseline (PPL) | Average AUC | 0.84 |
| Downstream contamination — TPR@5%FPR improvement over best baseline | Delta | +12.2% |
| Machine unlearning: suspicious chunks identified (HP books 1-4) | Count | 188 / ~1000 |
| Machine unlearning: completions with GPT-4 similarity >= 4 | Fraction | 5.3% |

**Scaling trends on WikiMIA**:
- AUC increases with model size (LLaMA 7B -> 13B -> 30B -> 65B).
- AUC increases with text length (32 -> 64 -> 128 -> 256 tokens).
- Higher learning rate (1e-4 vs 1e-5) increases AUC substantially across all downstream tasks.
- Higher occurrence frequency correlates positively with AUC.

## Core Concepts

- [[concepts/benchmark-contamination]] — Min-K% Prob is a black-box contamination detection tool applicable when pretraining data is undisclosed
- [[concepts/evaluation-variance]] — detection difficulty depends on text length, model size, and learning rate, creating systematic variance in contamination audit reliability

## Relevance To Poolside

Directly applicable to auditing whether Poolside pretraining data contains benchmark examples. The reference-free property is critical: no shadow model needed, only token-level log-probabilities from the target model. Could be used to (1) audit training data for eval contamination before training, (2) post-hoc audit trained models for contamination on specific benchmarks, (3) verify that decontamination procedures worked. The finding that outlier contaminants (benchmark examples in pretraining data) become easier to detect with more pretraining data is encouraging for large-scale auditing.

## Blaz Notes

- 

## Key Follow-Ups / Jumping-Off Points

- How does Min-K% Prob compare to n-gram overlap methods used by LLaMA/GPT for contamination reporting?
- Does the method work for code benchmarks (HumanEval, MBPP) where structured syntax may affect token-level probability distributions differently than natural language?
- The k=20 hyperparameter was tuned on LLaMA-65B — does it transfer well to different model families and scales?
- Can Min-K% Prob detect paraphrased contamination reliably? The WikiMIA paraphrase setting shows lower AUC than original — how much does this degrade for heavily paraphrased benchmark items?

## Related Notes

- Maps: [[maps/model-evaluation/landscape]]
- Concepts: [[concepts/benchmark-contamination]]
- Papers: (Carlini et al., 2022 — membership inference from first principles; Eldan & Russinovich, 2023 — Who's Harry Potter unlearning)

## Caveats

- WikiMIA AUC of 0.72 average is moderate — far from reliable single-instance detection, more useful for aggregate/statistical contamination assessment
- Paraphrase setting consistently harder than verbatim — method may miss contamination when training data contains edited/rephrased benchmark items
- Hyperparameter k=20 tuned on LLaMA-65B only; transferability to other architectures not systematically validated
- All experiments use Wikipedia-domain text (WikiMIA) or specific domains (books, QA datasets) — generalization to code or other structured text unclear
- The method assumes black-box access to token-level log-probabilities, which some API providers do not expose
- Machine unlearning audit is qualitative (similarity scores, cherry-picked examples) rather than a controlled quantitative evaluation of unlearning completeness
