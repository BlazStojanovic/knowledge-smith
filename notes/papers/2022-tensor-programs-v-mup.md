---
arxiv: '2203.03466'
authors:
- Greg Yang
- Edward J. Hu
- Igor Babuschkin
- Szymon Sidor
- Xiaodong Liu
- David Farhi
- Nick Ryder
- Jakub Pachocki
- Weizhu Chen
- Jianfeng Gao
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2203.03466
  raw: '[[raw/papers/md/2022-tensor-programs-v-mup]]'
  source: https://arxiv.org/abs/2203.03466
owner: blaz
raw_pdf: raw/papers/pdf/2022-tensor-programs-v-mup.pdf
read: false
slug: tensor-programs-v-mup
tags:
- type/paper
- status/draft
- source/primary
- domain/pretraining
- domain/training
- domain/llm
title: 'Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter
  Transfer'
type: note
updated: '2026-05-10'
year: 2022
---

# Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer

## Citation

- URL: https://arxiv.org/abs/2203.03466
- Authors: Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, Jianfeng Gao
- Affiliation: Microsoft Research, OpenAI
- Year / venue: 2022 / arXiv (ICML 2022 proceedings)
- arXiv: 2203.03466
- **Raw**: [[raw/papers/pdf/2022-tensor-programs-v-mup.pdf]]

## Core Claim

Under the Maximal Update Parameterization (muP), many optimal hyperparameters remain stable as model width changes. This enables **zero-shot hyperparameter transfer**: tune HPs on a small proxy model parametrized in muP, then copy them directly to the full-sized target model without any further tuning. The paper argues (and provides theoretical justification from Tensor Programs theory) that muP is the *unique* abc-parametrization that admits such transfer.

## Key Paper Ideas

### What muP is

muP is a parametrization — a rule for how initialization variance, learning rate, and parameter multipliers scale with width for each parameter tensor. It ensures that parameter updates have a consistent effect on activations regardless of model width, preventing the blow-up or vanishing of hidden activations during training that occurs under standard parametrization (SP).

In SP, after even 1 step of SGD/Adam, output logits and attention logits blow up with width, while word embeddings update by a width-independent amount. This means no single global LR is appropriate for all layers at all widths. muP fixes this by assigning width-dependent LR multipliers per parameter type.

### muP vs SP: scaling rules

The paper presents three parameter categories based on how many dimensions scale with width:
- **Input weights & all biases** ("vector-like", finite fan_in, infinite fan_out)
- **Output weights** ("vector-like", infinite fan_in, finite fan_out)
- **Hidden weights** ("matrix-like", both dimensions infinite)

**Table 3 formulation (main paper):**

| | Input weights & biases | Output weights | Hidden weights |
|---|---|---|---|
| **Init variance** | 1/fan_in *(same as SP)* | 1/fan_in^2 *(SP: 1/fan_in)* | 1/fan_in *(same as SP)* |
| **SGD LR** | fan_out * eta *(SP: eta)* | 1/fan_in * eta *(SP: eta)* | eta *(same as SP)* |
| **Adam LR** | eta *(same as SP)* | 1/fan_in * eta *(SP: eta)* | 1/fan_in * eta *(SP: eta)* |

**Table 8 formulation (easier to implement, compatible with weight tying):**

| | Input weights & biases | Output weights | Hidden weights |
|---|---|---|---|
| **Init variance** | 1/fan_in *(same as SP)* | 1/fan_in *(same as SP)* | 1/fan_in *(same as SP)* |
| **Multiplier** | 1 | 1/fan_in | 1 |
| **SGD LR** | fan_out * eta | 1 * eta | eta |
| **Adam LR** | eta | eta | 1/fan_in * eta |

The Table 8 formulation unifies all vector-like parameters (input, output, biases) under the same init/LR scaling and adds a 1/fan_in *multiplier* on the output instead. This enables input-output weight tying (common in Transformers).

### Transformer-specific modifications

1. **Attention scaling**: Use `1/d` instead of `1/sqrt(d)` for attention logits, i.e., `q^T k / d` instead of `q^T k / sqrt(d)`. Rationale: during training q and k become correlated, so q^T k scales like d (Law of Large Numbers), not sqrt(d) (CLT applies only at init).

2. **Base width for compatibility**: Define a base width `d_model_0` (e.g., 128). At that width, muP equals SP. Width ratio `d_tilde = d_model / d_model_0` determines the scaling factors.

3. **Per-parameter LR in a Transformer with Adam** (from Appendix B.1):
   - Word/positional embeddings: `eta_emb` (constant in width)
   - LayerNorm weights/biases: `eta_LN` (constant in width)
   - Q, K, V projections: `eta_qkv / d_tilde_model` (scales down with width)
   - Output projection W_o: `eta_o / (d_tilde_v * n_tilde_head)` (scales down with width)
   - MLP W1, W2: `eta_mlp / d_tilde` (scales down with width)
   - Unembedding: `eta_unemb / d_tilde_model` (scales down with width)

### The muTransfer protocol

Algorithm 1 from the paper:

1. Parametrize the target model in muP.
2. Tune a smaller version of the target model (shrink width and/or depth). This is the *proxy model*.
3. Copy the tuned hyperparameters directly to the target model.

The proxy model can be very small: 13M for BERT-large (350M), 40M for GPT-3 6.7B.

### What transfers and what does not

**muTransferable** (Table 1):
- Optimization-related: learning rate, momentum, Adam beta, LR schedule
- Initialization: per-layer init variance
- Parameter multipliers: multiplicative constants after weights/biases

**Not muTransferable**:
- Regularization: dropout, weight decay (these depend on both model and data size)

**muTransferred across** (the scaling dimensions):
- Width (theoretically justified)
- Depth (empirically validated on pre-layernorm only, with caveats)
- Batch size, training time, sequence length (empirically validated with caveats)

### Caveats and limitations

- **Depth transfer is weaker than width transfer**: init std does not transfer well across depth. Post-layernorm depth transfer does not work. Practical workaround: fix init std, tune other HPs when transferring across depth.
- **Minimum scale requirements**: proxy model needs minimum width (~256), depth (~4), batch size (~32), sequence length (~128), training steps (~5000) for transfer to work.
- **Regularization does not transfer**: dropout and weight decay depend on model + data size jointly.
- **Squashing activations** (tanh, sigmoid) reduce transfer quality compared to ReLU, because narrow networks saturate more.
- The paper focuses on training loss. In settings where regularization is the bottleneck (e.g., small-dataset fine-tuning), muTransfer may not suffice.

## Methodology

### Architectures and tasks

- **MLP**: 2-hidden-layer, ReLU, CIFAR-10 (SGD). Width 256-8192.
- **Transformer**: pre-LN and post-LN, Wikitext-2 (Adam). Width 128-8192. Used for HP stability sweeps.
- **IWSLT14 De-En**: post-LN Transformer (40M params). Proxy: 0.25x width (4M). Random search over LR, alpha_output, alpha_attn.
- **WMT14 En-De**: large post-LN Transformer (211M). Proxy: 15M (shrink d_model, d_ffn, n_head).
- **BERT**: Megatron pre-LN BERT-base (110M) and BERT-large (350M). Proxy: BERT-prototype (~13M). Transfer across both width and depth. HP search: 256 random combinations, 10^5 steps each.
- **GPT-3 6.7B**: 32 blocks, width 4096, relative attention. Proxy: width 256 (~40M params, 168x smaller). Trained in FP32 (vs FP16 baseline) due to numerical issues discovered during training.

### Experimental controls

- HP search repeated across multiple independent trials (e.g., 25 for IWSLT).
- Each selected HP combination evaluated with 5 random seeds, mean reported.
- "Naive transfer" baseline = copy HPs from small SP model to large SP model (diverges or fails).
- Compute budgets matched in FLOPs between muTransfer and conventional tuning.

## Key Results

### BERT (Table 6)

| Model | Method | Test loss | MNLI (m/mm) | QQP |
|---|---|---|---|---|
| BERT-base | Megatron default | 1.995 | 84.2 / 84.2 | 90.6 |
| BERT-base | muTransfer (13M proxy) | 1.970 | 84.3 / 84.8 | 90.8 |
| BERT-large | Megatron default | 1.731 | 86.3 / 86.2 | 90.9 |
| BERT-large | muTransfer (13M proxy) | **1.683** | **87.0 / 86.5** | **91.4** |

Total tuning cost = cost of pretraining 1 BERT-large. Naive transfer diverged in both cases.

### GPT-3 6.7B (Table 7, selected)

| Metric | 6.7B + muP | 6.7B re-run | 6.7B (Brown et al.) | 13B (Brown et al.) |
|---|---|---|---|---|
| Validation loss | **1.98** | 2.03 | - | - |
| PTB perplexity | **11.4** | 13.0 | - | - |
| WikiText-103 PPL | **8.56** | 9.13 | - | - |
| LAMBADA zero-shot | **73.5** | 70.8 | 70.3 | 72.5 |
| HellaSwag zero-shot | **72.0** | 66.7 | 67.4 | 70.9 |

The muTransferred 6.7B model outperforms both the 6.7B baseline and the re-run, and is comparable to the 13B model from Brown et al. Tuning cost was 7% of total pretraining cost. Proxy model was 40M (168x smaller).

### IWSLT14 De-En (Table 4)

muTransfer from 0.25x proxy: median 35.33 BLEU vs 35.00 for conventional tuning, given the same FLOP budget (64 samples). Naive transfer diverged. The compute-performance Pareto frontier of muTransfer strictly dominates conventional tuning.

### Wider-is-better property

Under muP with fixed HPs, wider models always achieve better training loss at every point during training (monotonically, modulo init noise). This fails under SP. Verified up to width 32,768 on GPT-3 architecture. This property can be used as a cheap sanity check for muP implementation correctness.

### Coordinate checking

The paper proposes "coordinate checking" as a diagnostic: plot the average coordinate magnitude of every (pre)activation vector across widths over a few training steps. Under muP all quantities stay O(1); under SP, logits and attention logits blow up. This is analogous to gradient checking for autograd and is included in the `mup` PyTorch package.

## Practical implementation tips (from Appendix D)

1. **Zero-init output layer and query projection**: reduces mismatch between proxy and target due to the initial Gaussian process; empirically does not hurt performance.
2. **Use non-squashing activations**: ReLU > tanh/sigmoid for transfer quality.
3. **Enlarge d_k**: making key dimension larger improves transfer precision in Transformers.
4. **AdamW is preferred over Adam with weight decay** under muP, because AdamW automatically scales weight decay correctly.
5. **Gradient clipping**: compatible if clip value is held constant with width.
6. **epsilon in Adam**: if non-negligible, scale like 1/fan_in^2 (before sqrt) or 1/fan_in (after sqrt).

## Theoretical foundation (brief)

The paper is the 5th installment of the Tensor Programs series. The core theoretical argument:
- By the CLT analogy (Section 2): correct parametrization = one where the loss function converges to a nontrivial limit as width -> infinity, so that the optimal HP converges too.
- muP is the unique abc-parametrization where: (a) every layer's activations and updates are O(1) in width, (b) feature learning is preserved (not kernel regime), and (c) the infinite-width limit is nontrivial.
- SP fails because hidden-to-output signals blow up with width after training begins, while input signals don't scale enough. No single global LR compensates for both.

## Caveats

- Width scaling well-validated theoretically and empirically. Depth scaling is empirical only and limited to pre-layernorm; init std does not transfer across depth.
- Requires implementing muP from the start of the project; retrofitting to an existing SP codebase requires changing init, LR, and attention scaling.
- The GPT-3 experiment used FP32 due to numerical divergence in FP16 — precision interaction with muP is not fully understood.
- Regularization HPs (dropout, weight decay magnitude) do not transfer and must still be set per model size.
- Recent extensions address some limitations: U-muP (depth scaling), HyperP (broader HP classes), Cerebras muP guide (practitioner recipes).

## Core Concepts

- [[concepts/hyperparameter-scaling]] — muP is the foundational framework for HP transfer
- [[concepts/compute-optimal-methodology]] — muTransfer as a methodology for efficient scaling
- [[concepts/scaling-laws-foundational]] — HP scaling is entangled with loss scaling
- [[maps/scaling-laws/landscape]] — hyperparameters domain

## Relevance To Poolside

muP directly affects Poolside's training economics:
- **HP tuning at scale**: Poolside trains multi-billion parameter models. muTransfer allows tuning HPs on a small proxy (e.g., 40M-100M) and transferring to the production-scale model, reducing tuning cost from prohibitive to a fraction of one pretraining run.
- **Wider-is-better guarantee**: Under muP, scaling up width is monotonically beneficial — removes the risk that a wider model underperforms a narrower one due to HP mismatch.
- **Exploration-to-scaling**: Researchers can experiment with new ideas (data mixes, architectures, schedules) at small scale and trust that found HPs will transfer up. This directly supports the fast iteration loop needed for synthetic data experiments.
- **Implementation requirement**: muP must be baked into the training framework (Titan). Retrofitting is nontrivial — requires per-parameter LR groups, modified attention scaling, and init changes.

## Blaz Notes

- 

## Related Notes

- [[notes/papers/2022-training-compute-optimal-large-language-models]] — Chinchilla (HP tuning was critical to its result; muP could have improved their sweep efficiency)
- [[notes/papers/2022-scaling-laws-and-interpretability-of-learning-from-repeated-data]] — scaling laws context
- [[concepts/hyperparameter-scaling]]
- [[concepts/compute-optimal-methodology]]
- [[maps/scaling-laws/landscape]]
