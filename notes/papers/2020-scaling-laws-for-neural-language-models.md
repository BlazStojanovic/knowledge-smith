---
arxiv: '2001.08361'
authors:
- Jared Kaplan
- Sam McCandlish
- Tom Henighan
- Tom B. Brown
- Benjamin Chess
- Rewon Child
- Scott Gray
- Alec Radford
- Jeffrey Wu
- Dario Amodei
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2001.08361
  raw: '[[raw/papers/md/2020-scaling-laws-for-neural-language-models]]'
  source: https://arxiv.org/abs/2001.08361
owner: blaz
raw_pdf: raw/papers/pdf/2020-scaling-laws-for-neural-language-models.pdf
read: false
slug: scaling-laws-for-neural-language-models
tags:
- type/paper
- status/draft
- source/primary
- domain/pretraining
- domain/llm
- domain/training
title: Scaling Laws for Neural Language Models
type: note
updated: '2026-05-10'
year: 2020
---

# Scaling Laws for Neural Language Models

## Citation

- URL: https://arxiv.org/abs/2001.08361
- Authors: Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei
- Affiliation: OpenAI, Johns Hopkins University
- Year / venue: 2020 / arXiv preprint
- arXiv: 2001.08361
- **Raw**: [[raw/papers/pdf/2020-scaling-laws-for-neural-language-models.pdf]]

## Core Claim

Cross-entropy loss of Transformer language models scales as a power law with model size (non-embedding parameters N), dataset size (tokens D), and training compute (C), with trends spanning more than seven orders of magnitude. Architectural details such as depth vs. width have minimal effect within a wide range. Optimally compute-efficient training allocates most additional compute to larger models rather than more data or longer training, resulting in very large models trained on relatively modest data and stopped well before convergence.

## Key Paper Ideas

- **Three independent power laws.** When not bottlenecked by the other two factors, loss follows L(N) = (N_c/N)^{alpha_N}, L(D) = (D_c/D)^{alpha_D}, and L(C_min) = (C_c^{min}/C_min)^{alpha_C^{min}} with fitted exponents alpha_N ~ 0.076, alpha_D ~ 0.095, alpha_C^{min} ~ 0.050.
- **Architecture insensitivity.** Performance depends very weakly on shape hyperparameters (depth, width, number of attention heads, feed-forward dimension) when total non-embedding parameter count N is held fixed. Aspect ratio (d_model / n_layer) can vary by 40x with only ~3% loss change.
- **Joint N-D scaling and overfitting law.** A single equation L(N, D) = [(N_c/N)^{alpha_N/alpha_D} + (D_c/D)]^{alpha_D} governs simultaneous dependence on model and dataset size, predicting overfitting onset via the ratio N^{0.74}/D.
- **Sample efficiency of large models.** Larger models reach the same loss with fewer optimization steps and fewer data points. The data requirement to avoid overfitting grows sub-linearly: D >= (5 x 10^3) * N^{0.74}.
- **Compute-optimal allocation (Kaplan prescription).** Given a fixed compute budget C, optimally: N proportional to C^{0.73}, B proportional to C^{0.24}, S proportional to C^{0.03}, D proportional to C^{0.27}. Most additional compute should go to model size; data grows slowly; training steps barely increase.
- **Convergence is inefficient.** Compute-efficient training stops at ~10% above the converged loss (alpha_N / alpha_S ~ 10%). Training to full convergence wastes ~65% of compute relative to the efficient frontier.
- **Critical batch size.** B_crit follows a power law in the loss: B_crit = B* / L^{1/alpha_B} with B* ~ 2 x 10^8 tokens, alpha_B ~ 0.21. B_crit is independent of model size, depending only on current loss.
- **Transfer.** Out-of-distribution loss tracks in-distribution loss with a roughly constant offset, independent of training duration or model depth.

## Methodology

- **Dataset**: WebText2 (extended WebText), 2.29 x 10^{10} tokens, 20.3M documents, 96 GB text. BPE tokenization with vocab size 50257. Context length 1024 tokens.
- **Architecture**: Decoder-only Transformer. Also tested LSTMs and Universal Transformers for comparison.
- **Model range**: 768 to 1.5 billion non-embedding parameters. Shapes from (n_layer=2, d_model=128) to (n_layer=207, d_model=768) and (n_layer=6, d_model=4288). Total of approximately 768 models trained (not stated as a single number in the paper; the figure is from the breadth of sweeps over size, shape, dataset, batch size, context length).
- **Optimizer**: Adam, 2.5 x 10^5 steps, batch size 512 sequences of 1024 tokens (= 2^{19} tokens). Largest models (>1B) used Adafactor. Learning rate: 3000-step linear warmup then cosine decay to zero.
- **Compute definition**: C ~ 6NBS FLOPs (non-embedding). Reported in PF-days (1 PF-day = 8.64 x 10^{19} FLOP).
- **Parameter counting**: N excludes vocabulary and positional embeddings; N ~ 12 * n_layer * d_model^2 (with standard d_attn = d_ff/4 = d_model). This exclusion was shown to produce significantly cleaner scaling.
- **Methodology for compute-optimal frontier**: For a given compute budget C, scan over model sizes N to find the one with best loss at step S = C/(6BS). Then adjust to critical batch size via S_min and C_min corrections (Equations 5.4, 5.5).

## Key Results

### Fitted power-law exponents (Table 5 in paper, Appendix A)

| Exponent | Value | Scale constant |
|---|---|---|
| alpha_N (parameters) | 0.076 | N_c = 8.8 x 10^{13} params |
| alpha_D (data) | 0.095 | D_c = 5.4 x 10^{13} tokens |
| alpha_C (compute, naive fixed-batch) | 0.057 | C_c = 1.6 x 10^7 PF-days |
| alpha_C^{min} (compute, optimal batch) | 0.050 | C_c^{min} = 3.1 x 10^8 PF-days |
| alpha_B (critical batch size) | 0.21 | B* = 2.1 x 10^8 tokens |
| alpha_S (training steps) | 0.76 | S_c = 2.1 x 10^3 steps |

### Compute-optimal allocation (Table 6 in paper)

| Quantity | Scaling with C_min | Scale constant |
|---|---|---|
| N_opt (model size) | C^{0.73} | N_e = 1.3 x 10^9 params |
| B (batch size ~ B_crit) | C^{0.24} | B_e = 2.0 x 10^6 tokens |
| S_min (training steps) | C^{0.03} | S_e = 5.4 x 10^3 steps |
| D_opt (dataset, 1 epoch) | C^{0.27} | D_e = 2 x 10^{10} tokens |

### Joint L(N, D) fit (Table 2)

| Parameter | Value |
|---|---|
| alpha_N | 0.076 |
| alpha_D | 0.103 |
| N_c | 6.4 x 10^{13} |
| D_c | 1.8 x 10^{13} |

### Other key findings

- Doubling N yields loss multiplied by 2^{-0.076} = 0.949 (roughly 5% improvement).
- Predicted intersection point where compute-efficient scaling laws break down: C* ~ 10^4 PF-days, N* ~ 10^{12} parameters, D* ~ 10^{12} tokens, L* ~ 1.7 nats/token.
- Compute-efficient training uses 7.7x fewer steps, 2.7x more parameters, and 65% less compute than training to near-convergence at a fixed loss target.
- LSTMs match Transformers on early-context tokens but cannot match on later tokens; Transformers improve throughout the full 1024-token context.

## Caveats

- **Chinchilla correction (Hoffmann et al. 2022).** The compute-optimal allocation N proportional to C^{0.73} was shown to significantly overweight model size relative to data. Chinchilla found N and D should scale roughly equally with compute (both ~ C^{0.5}), implying this paper undertrained on data. The D proportional to C^{0.27} recommendation was the most consequential error.
- **No theoretical explanation.** The authors acknowledge they have no solid theoretical understanding of the scaling laws. The power-law exponents are purely empirical.
- **WebText2 only.** All results on a single training distribution. The exponents may differ for other data distributions or tokenizations (the scale constants N_c, D_c, C_c explicitly depend on tokenization).
- **Regularization not optimized.** Fixed 10% dropout. No exploration of how regularization interacts with the scaling laws.
- **Small data regime poorly characterized.** Fits were poor for the smallest datasets (~2 x 10^7 tokens, where an epoch = 40 steps).
- **Compute estimate ignores context-dependent cost.** C ~ 6NBS omits the attention cost proportional to n_ctx. This could confound results for very large context lengths (n_ctx > 12 * d_model).
- **Batch size extrapolation uncertain.** The B_crit(L) prediction may not hold for loss values far outside the observed range.
- **Learning rate sensitivity.** Optimal learning rate depends on target loss; short runs might benefit from larger LR, which was not explored.

## Core Concepts

- [[concepts/scaling-laws-foundational]] -- this paper provides the original power-law formulations for N, D, C
- [[concepts/compute-optimal-methodology]] -- IsoFLOP methodology for finding optimal model size at fixed compute
- [[maps/scaling-laws/landscape]] -- foundational paper in the scaling laws domain

## Relevance To Poolside

*Our interpretation, explicitly labelled.*

- **Training budget allocation.** The core question of how to split a compute budget between model size, data, and training steps is directly relevant to every Poolside pre-training run. While the specific Kaplan exponents have been superseded by Chinchilla and subsequent work, the methodology of fitting power laws to IsoFLOP profiles remains the standard approach.
- **Data scaling.** The finding that data needs grow sub-linearly with model size (even if the specific exponent was wrong) frames the question of how much synthetic data Poolside needs to generate to support larger models.
- **Sample efficiency.** The claim that larger models are more sample-efficient is relevant to Poolside's strategy of training large models. Each token of high-quality synthetic data has more impact on a larger model.
- **Architecture insensitivity.** The finding that width/depth ratio barely matters (within reason) reduces the search space for architecture decisions.
- **Overfitting law.** The N^{0.74}/D overfitting predictor, while the exponent may need recalibration, provides a framework for monitoring whether a training run has enough data.

## Blaz Notes

- 

## Related Notes

- [[notes/papers/2022-training-compute-optimal-large-language-models]] -- Chinchilla, corrected the compute-optimal allocation to N ~ C^{0.5}, D ~ C^{0.5}
- [[notes/papers/2024-over-training-scaling]] -- Gadre et al., extended scaling laws to the over-training regime
- [[notes/papers/2023-scaling-data-constrained-language-models]] -- Muennighoff et al., scaling under data constraints and epoch repetition
- [[notes/papers/2025-scaling-laws-of-synthetic-data-for-language-models]] -- scaling laws for synthetic data specifically
- [[concepts/scaling-laws-foundational]]
- [[concepts/compute-optimal-methodology]]
- [[maps/scaling-laws/landscape]]
