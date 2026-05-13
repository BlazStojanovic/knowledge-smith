---
arxiv: '2203.15556'
authors:
- Jordan Hoffmann
- Sebastian Borgeaud
- Arthur Mensch
- Elena Buchatskaya
- Trevor Cai
- Eliza Rutherford
- Diego de Las Casas
- Lisa Anne Hendricks
- et al
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2203.15556
  raw: '[[raw/papers/md/2022-training-compute-optimal-large-language-models]]'
  source: https://arxiv.org/abs/2203.15556
owner: blaz
raw_pdf: raw/papers/pdf/2022-training-compute-optimal-large-language-models.pdf
read: false
slug: training-compute-optimal-large-language-models
tags:
- type/paper
- status/draft
- source/primary
- domain/pretraining
- domain/llm
- domain/training
title: Training Compute-Optimal Large Language Models
type: note
updated: '2026-05-10'
year: 2022
---

# Training Compute-Optimal Large Language Models

## Citation

- URL: https://arxiv.org/abs/2203.15556
- PDF: https://arxiv.org/pdf/2203.15556
- Authors: Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, et al.
- Year / venue: 2022-03-29 arXiv preprint
- arXiv: 2203.15556v1
- Raw PDF: [[raw/papers/pdf/2022-training-compute-optimal-large-language-models.pdf]]

## Core Claim

For a fixed compute budget, model size and the number of training tokens should be scaled in approximately equal proportions. This directly contradicts Kaplan et al. (2020), which recommended scaling model size ~5.5x for every 10x compute increase while increasing tokens only ~1.8x. The paper validates this by training Chinchilla (70B parameters, 1.4T tokens), which uses the same compute as Gopher (280B, 300B tokens) but uniformly outperforms it on downstream tasks.

## Key Paper Ideas

- **Three independent estimation approaches** all converge on the same conclusion: equal scaling of parameters and data with compute.
  - Approach 1 (training curve envelopes): fix model sizes (70M--10B), vary training horizons (4 cosine schedules per size), extract the loss-minimizing frontier across FLOPs. Yields `a = 0.50, b = 0.50`.
  - Approach 2 (IsoFLOP profiles): fix 9 FLOP budgets (6e18--3e21), vary model size (up to 16B) at each budget, find the valley in loss vs. parameters. Yields `a = 0.49, b = 0.51`.
  - Approach 3 (parametric loss fit): fit a closed-form loss function to all final losses from Approaches 1 and 2. Yields `a = 0.46, b = 0.54`.

- **Parametric loss form:**
  ```
  L(N, D) = E + A / N^alpha + B / D^beta
  ```
  Three terms: irreducible entropy of natural text, functional approximation gap (finite model), and optimization suboptimality (finite data/steps). Fitted via Huber loss (delta = 1e-3) with L-BFGS from a grid of initializations.

- **Fitted constants** (Approach 3, Equation 10):
  - E = 1.69 (irreducible loss / entropy of text)
  - A = 406.4
  - B = 410.7
  - alpha = 0.34
  - beta = 0.28

- **Compute-optimal allocation** from the parametric form:
  ```
  N_opt(C) = G * (C/6)^a,    D_opt(C) = G^{-1} * (C/6)^b
  where G = (alpha*A / beta*B)^{1/(alpha+beta)},  a = beta/(alpha+beta),  b = alpha/(alpha+beta)
  ```

- **Practical rule:** for compute-optimal training, D ~ 20N (i.e., ~20 tokens per parameter). At Gopher's compute budget (5.76e23 FLOPs), the optimal model is ~67B params trained on ~1.5T tokens, not 280B on 300B.

- **Chinchilla 70B result:** matches Gopher 280B in training FLOPs but outperforms it on nearly every downstream task. MMLU 67.6% vs. 60.0% (Gopher), 43.9% (GPT-3). BIG-bench average 65.1% vs. 54.4%. Also outperforms GPT-3 175B, Jurassic-1 178B, and MT-NLG 530B on most benchmarks.

- **Contradiction of Kaplan et al. (2020):** Kaplan's scaling exponents were a = 0.73, b = 0.27 (strongly favoring model size). The key methodological difference: Kaplan used a fixed learning rate schedule and fixed token budget across all model sizes, which overestimates loss for shorter training runs (the cosine schedule does not decay properly when the cycle is much longer than training). This led Kaplan to underestimate the value of training on more data.

## Methodology

**Scale of experiments.** Over 400 models, 70M to 16B parameters, trained on 5B to 400B+ tokens. Each model configuration trained at multiple training horizons. Trained on MassiveText (same dataset as Gopher). All runs on TPUv3/TPUv4 with JAX and Haiku.

**Learning rate tuning per model (critical).** Cosine schedule with 10x decay, where the cycle length is approximately matched to the number of training tokens. This is the single biggest methodological difference from Kaplan et al. (2020), who used a fixed schedule for all runs. Overestimating training steps by more than 25% in the cosine schedule leads to clear performance degradation (Figure A1). Max LR: 2e-4 for smallest models, 1.25e-4 for largest.

**Training dataset.** MassiveText, composed of: MassiveWeb (45%), Books (30%), C4 (10%), News (10%), GitHub (4%), Wikipedia (1%). For Chinchilla's 1.4T tokens, MassiveWeb is used for 1.24 epochs, Wikipedia for 3.4 epochs; other subsets stay under 1 epoch.

**Chinchilla specifics.** 70B params, 80 layers, 64 heads, key/value size 128, d_model 8192. Trained with AdamW (not Adam as Gopher). Slightly modified SentencePiece tokenizer without NFKC normalization (helps math/chemistry representation; 94.15% token overlap with Gopher). Batch size ramped 1.5M to 3M tokens midway through training. Max LR 1e-4.

**Consistency check.** IsoFLOP analysis repeated on C4 and GitHub code datasets yields similar exponents (C4: a=0.50, b=0.50; GitHub: a=0.53, b=0.47), suggesting the result is dataset-independent in the single-epoch regime.

## Key Results

### Scaling exponents (Table 2)

| Approach | a (N_opt ~ C^a) | b (D_opt ~ C^b) |
|---|---|---|
| 1. Training curve envelopes | 0.50 (0.488, 0.502) | 0.50 (0.501, 0.512) |
| 2. IsoFLOP profiles | 0.49 (0.462, 0.534) | 0.51 (0.483, 0.529) |
| 3. Parametric loss fit | 0.46 (0.454, 0.455) | 0.54 (0.542, 0.543) |
| Kaplan et al. (2020) | 0.73 | 0.27 |

Parenthetical ranges are 10th/90th percentiles from bootstrap (80% of data sampled 100 times).

### Fitted parametric constants (Equation 10)

| Parameter | Value | Interpretation |
|---|---|---|
| E | 1.69 | Irreducible loss (entropy of natural text) |
| A | 406.4 | Model-size coefficient |
| B | 410.7 | Data-size coefficient |
| alpha | 0.34 | Model-size exponent |
| beta | 0.28 | Data-size exponent |

### Chinchilla vs. existing models

| Model | Params | Tokens | MMLU (5-shot) | BIG-bench avg |
|---|---|---|---|---|
| Chinchilla | 70B | 1.4T | 67.6% | 65.1% |
| Gopher | 280B | 300B | 60.0% | 54.4% |
| GPT-3 | 175B | 300B | 43.9% | — |
| MT-NLG 530B | 530B | 270B | — | — |

Chinchilla also outperforms Gopher on all 20 Pile evaluation subsets, on RACE-m/h by >10% absolute, on Natural Questions (35.5% vs. 28.2% 64-shot), on TriviaQA (73.2% vs. 63.6% 5-shot), and on TruthfulQA (43.6% vs. 29.5% 0-shot).

### Compute-optimal frontier projections (Table 3, Approach 1)

| Parameters | FLOPs | Optimal Tokens |
|---|---|---|
| 400M | 1.92e19 | 8.0B |
| 1B | 1.21e20 | 20.2B |
| 10B | 1.23e22 | 205.1B |
| 67B | 5.76e23 | 1.5T |
| 175B | 3.85e24 | 3.7T |
| 280B | 9.90e24 | 5.9T |
| 1T | 1.27e26 | 21.2T |

## Extracted From Repetition Memo

- Source review: [[raw/reviews/2026-scaling-laws-data-repetition-review]].
- Role in the memo: background baseline for compute-optimal training, not a repetition paper itself.
- Repetition relevance: the memo treats Chinchilla as the one-pass scaling law that Muennighoff et al. extend by replacing raw tokens with effective repeated tokens.

## Caveats

- **Validated only up to 70B.** The parametric fit and scaling exponents are extrapolated beyond the training range (max 16B for scaling runs). The Chinchilla 70B run is the only large-scale validation point; no intermediate-scale confirmations exist.
- **No public validation at 200B+.** The prediction that a 280B model needs ~6T tokens and a 1T model needs ~21T tokens is pure extrapolation.
- **Negative curvature in the frontier.** The authors note concavity in log N_opt at high compute budgets (Appendix E), suggesting that optimal models at very large scale may be even smaller than predicted—the power-law form may underestimate this effect.
- **Does not cover post-training.** The analysis is on pre-training loss only. How compute-optimal pre-training interacts with SFT, RLHF, or other post-training stages is not addressed.
- **Parametric form is assumed, not derived.** The three-term decomposition (Equation 2) has theoretical motivation from function approximation and stochastic optimization but the specific power-law form is an empirical assumption.
- **Single-epoch regime only.** All training runs see data less than once. The paper explicitly notes that the multiple-epoch regime is future work—this is exactly the gap Muennighoff et al. (2023) later fill.
- **Dataset-specific confounds.** Chinchilla uses AdamW (vs. Adam for Gopher), a different tokenizer, and slightly different MassiveText sampling weights. The ablation in Appendix G addresses optimizer differences but these remain confounds for the headline comparison.

## Core Concepts

- [[concepts/scaling-laws-foundational]] — this paper provides the corrected compute-optimal allocation
- [[concepts/compute-optimal-methodology]] — three independent estimation methods
- [[concepts/hyperparameter-scaling]] — per-model LR tuning was critical to the result
- [[maps/scaling-laws/landscape]] — foundational domain

## Relevance To Poolside

Our interpretation: use this as the baseline scaling-law reference when discussing whether a Laguna-scale run is data-limited, compute-limited, or operating in a repeated-data regime.

The D ~ 20N rule provides the anchor for determining when a model is undertrained or when data repetition becomes necessary. For Laguna-class models, the Chinchilla frontier predicts token requirements in the trillions, which makes data-constrained scaling (Muennighoff et al. 2023) and data quality/repetition directly relevant to our regime.

The methodological lesson about per-model LR schedule matching is directly actionable: scaling experiments that reuse a single fixed cosine schedule across model sizes will systematically underestimate the value of training smaller models on more data.

## Blaz Notes

- 

## Related Notes

- [[notes/papers/2020-scaling-laws-for-neural-language-models]] — Kaplan, the paper Chinchilla corrected
- [[notes/papers/2024-over-training-scaling]] — Gadre, extended to over-training
- [[notes/papers/2023-scaling-data-constrained-language-models]] — Muennighoff, extended to repeated data
- [[concepts/scaling-laws-foundational]]
- [[concepts/compute-optimal-methodology]]
- [[maps/scaling-laws/landscape]]
- [[hypotheses/seed-repetition-at-laguna-xs-can-hurt-quality]]
