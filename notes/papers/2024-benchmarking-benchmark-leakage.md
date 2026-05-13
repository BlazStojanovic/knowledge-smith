---
arxiv: '2404.18824'
authors:
- Xu
- Wang
- Fan
- Liu (GAIR / Shanghai Jiao Tong University)
created: 2026-04-22
kind: paper
links:
  code: https://github.com/GAIR-NLP/benbench
  paper: https://arxiv.org/abs/2404.18824
  raw: null
  source: https://arxiv.org/abs/2404.18824
owner: blaz
raw_pdf: raw/papers/pdf/2024-benchmarking-benchmark-leakage.pdf
read: false
slug: benchmarking-benchmark-leakage
tags:
- type/paper
- source/primary
- status/draft
- domain/evals
title: Benchmarking Benchmark Leakage in Large Language Models
type: note
updated: '2026-05-10'
year: 2024
---

# Benchmarking Benchmark Leakage in Large Language Models

Systematic detection pipeline for benchmark data leakage in LLMs, applied to 31 models on GSM8K and MATH, revealing widespread training-set contamination and proposing a "Benchmark Transparency Card."

## Citation

- URL: https://arxiv.org/abs/2404.18824
- Authors: Xu, Wang, Fan, Liu (GAIR / Shanghai Jiao Tong University)
- Year / venue: 2024 / arXiv
- Raw: [[raw/papers/pdf/2024-benchmarking-benchmark-leakage.pdf]]

## Core Contribution

1. **Taxonomy of training behaviors and detection challenges.** Formalizes three training scenarios (no benchmark data, training split only, test split too) and four detection challenges (test set not guaranteed clean, threshold ambiguity, unknown benchmark utilization/augmentation, inaccessible model weights).
2. **Detection pipeline** using two atomic metrics -- Perplexity and N-gram Accuracy -- plus reference benchmark synthesis via paraphrasing.
3. **Large-scale evaluation** of 31 open-source LLMs on GSM8K and MATH.
4. **Benchmark Transparency Card** proposal for model documentation.

## Detection Methodology

### Atomic Metrics

| Metric | Type | Measures | Closed-source? |
|---|---|---|---|
| **Perplexity (PPL)** | Continuous | Average negative log-likelihood on answer portion | No (requires logits) |
| **N-gram Accuracy** | Discrete | Fraction of n-grams (n=5 or 10) correctly predicted via greedy generation from uniformly sampled start points (K=5 per sample) | Yes (generation-only) |

PPL is calculated only on the answer (solution) portion to avoid noise from models that may not compute loss on the question. N-gram Accuracy spans the full concatenated question+answer text.

### Reference Benchmark Synthesis

Since the test set itself may be contaminated, comparing train vs test PPL alone is unreliable. Instead, the pipeline synthesizes three paraphrased versions of each benchmark split using ChatGPT (gpt-3.5-turbo-0125, temperature 0.7, top-p 0.9). These preserve reasoning difficulty but alter surface text.

### Pipeline Steps

1. **Synthesize** three reference benchmark datasets via paraphrasing.
2. **Compute** the atomic metric on both original and (averaged) synthesized benchmarks.
3. **Compute the percentage decrease** delta = (M_ori - M_ref) / M_ori x 100% for each split (train, test).
4. **Compare** delta_train-test = delta_train - delta_test. High positive values indicate the model is disproportionately familiar with the training split (likely contamination). Near-zero values suggest either no contamination or simultaneous contamination of both splits.

### Instance-Level Detection

N-gram Accuracy enables instance-level detection: if all K sampled n-grams for a sample are predicted correctly, the sample is likely memorized. Three matching criteria used: Exact Match, ROUGE-L > 0.75, edit distance similarity > 0.9.

## Key Results

### Meta-Experiment (Controlled Validation)

Mistral-7B-v0.1 trained on 1,000 benchmark samples ("seen") vs 1,000 held-out ("unseen"), under both SFT and pretraining objectives:
- Trained models show elevated delta_seen and delta_seen-unseen; untrained baseline shows near-zero delta_seen-unseen.
- SFT yields higher delta_seen-unseen than pretraining (stronger per-sample memorization, easier to detect).
- Ranking of delta_seen-unseen is consistent across metrics and benchmarks.

### Wild Evaluation (31 LLMs)

Models ranked by delta_train-test on 5-gram accuracy (GSM8K), highest risk first:

| Model | delta_train-test (5-gram, GSM8K) | Notes |
|---|---|---|
| Qwen-7B | 35.54 | |
| Qwen-14B | 35.16 | |
| InternLM2-7B | 30.12 | Continual pretrain on STEM data per tech report |
| InternLM2-20B | 29.11 | Same |
| Aquila2-7B | 23.24 | Documentation confirms GSM8K training |
| Aquila2-34B | 23.12 | Documentation confirms GSM8K training |
| Qwen-1.8B | 22.87 | |
| Baichuan2-13B-Base | 17.85 | |
| ... | | |
| LLaMA-7B | 1.07 | Near-zero: minimal leakage risk |
| Mistral-7B-v0.1 | 0.96 | Near-zero |
| Llama-3-8B | -2.13 | Near-zero |

Models like LLaMA, Mistral-7B, and Llama-3-8B show near-zero delta_train-test, consistent with minimal benchmark contamination. The top-ranked models (Qwen, InternLM2, Aquila2) show strong evidence of training-set utilization.

PPL-based rankings strongly align with N-gram-based rankings despite measuring different aspects of memorization.

### Instance-Level Leakage

- **Qwen-1.8B** accurately predicted all 5-grams in **223 examples** from the GSM8K training set and **67** from the MATH training set, with an additional **25** correct predictions on the MATH test set.
- **Aquila2-34B** (known to have been accidentally exposed to the full GSM8K test set) consistently predicts n-grams as "The answer is" instead of the placeholder "####", suggesting it trained on a reformatted version -- this makes N-gram Accuracy detection harder but PPL still captures the signal.

### ChatGLM2 Reformatting Case

ChatGLM2 predicts "Answer: \\boxed" instead of the golden n-gram "\n### 12" near answers, suggesting training on a reformatted version of GSM8K. PPL captures this where N-gram Accuracy fails.

## Recommendations

1. **Model Documentation**: Release a Benchmark Transparency Card specifying which benchmarks were used in training and whether augmentation was applied.
2. **Benchmark Construction**: Use latest corpus; regularly update with dynamic benchmarks.
3. **Benchmark Public Access**: Avoid uploading paired Q&A online; encrypt test sets or use private leaderboards.
4. **Evaluation**: Supplement static benchmarks with temporal holdouts (new exam questions, dynamic benchmarks).

## Limitations

- Cannot detect contamination when benchmark data has been augmented/reformatted (N-gram Accuracy fails; PPL partially mitigates).
- Cannot detect simultaneous leakage of both train and test sets via delta_train-test alone (instance-level N-gram detection partially mitigates).
- Reference benchmark synthesis via paraphrasing may introduce surface-level biases.
- Pipeline validated only on mathematical reasoning benchmarks (GSM8K, MATH); generalizability to other domains not demonstrated.
- Threshold for delta_train-test significance is not formally established -- affected by model size, training strategy, data distribution.

## Connections

- Map: [[maps/model-evaluation/contamination-methods]] -- this paper provides a systematic pipeline combining perplexity and n-gram approaches
- Concept: [[concepts/benchmark-contamination]] -- the problem this paper addresses
- Concept: [[concepts/temporal-benchmark-robustness]] -- the paper's recommendation for dynamic benchmarks aligns with temporal contamination resistance

## Related Papers

- **Min-K% Prob** ([[notes/papers/2024-min-k-percent]]): Instance-level, gray-box, likelihood-based detection. Complementary approach -- Min-K% uses a different statistical criterion on token probabilities.
- **GSM1k** ([[notes/papers/2024-gsm1k]]): Parallel holdout approach to contamination detection on GSM8K specifically. Performance-gap signal.
- **Time Travel in LLMs** ([[notes/papers/2023-time-travel-in-llms]]): Black-box guided completion probing.
- **LLM-Decontaminator** ([[notes/papers/2023-llm-decontaminator]]): White-box semantic matching that catches rephrased contamination.
- **Proving Test Set Contamination** ([[notes/papers/2023-proving-test-set-contamination]]): Ordering exchangeability test (Oren et al.).
- **LiveCodeBench** ([[notes/papers/2024-livecodebench]]): Temporal-gating approach to contamination-free evaluation.
- **Rethinking Benchmark and Contamination** (Yang et al., 2023): Paraphrasing benchmarks for contamination detection -- similar reference-synthesis approach.
- **CLEAN-EVAL** (Zhu et al., 2023): Paraphrasing-based clean evaluation of contaminated LLMs.
