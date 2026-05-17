---
arxiv: '2604.23747'
authors:
- Alexis Limozin
- Eduard Durech
- Torsten Hoefler
- Imanol Schlag
- Valentina Pyatkin
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning
url: https://arxiv.org/abs/2604.23747
year: 2026
---

# SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning

Alexis Limozin
Affiliation: EPFL
  
Eduard Durech
  
Torsten Hoefler
  
Imanol Schlag
Equal senior authorship.
  
Valentina Pyatkin11footnotemark: 1
Affiliation: Allen Institute for AI[0.3em]
Corresponding author: alexis@limozin.net
  
[0.5em]
ETH AI Center
  
ETH Zürich

###### Abstract

Recent mixed-policy optimization methods for LLM reasoning that interleave or blend supervised and reinforcement learning signals report improvements over the standard SFT-then-RL pipeline. We show that numerous recently published research papers rely on a faulty baseline caused by two distinct bugs: a CPU-offloaded optimizer bug in DeepSpeed that silently drops intermediate micro-batches during gradient accumulation (affecting multiple downstream frameworks including TRL, OpenRLHF and Llama-Factory), and a loss aggregation bug in OpenRLHF that incorrectly weights per-mini-batch losses. Together they suppress SFT performance, with the optimizer bug accounting for most of the gap and the loss aggregation bug contributing a smaller additional effect. Once corrected, the standard SFT-then-RL pipeline surpasses every published mixed-policy method we evaluate by +3.8 points on math benchmarks with Qwen2.5-Math-7B and by +22.2 points with Llama-3.1-8B. Even a truncated variant with just 50 RL steps outperforms mixed-policy methods on math benchmarks while using fewer FLOPs.

[alek6kun/sft\_then\_rl](https://github.com/alek6kun/sft_then_rl)

!(/html/2604.23747/assets/x1.png)

Figure 1: Results on Qwen2.5-Math-7B. Left: Training reward vs. compute for SFT→\toRL, and the buggy SFT→\toRL. Right: SFT evaluation score as bugs are progressively fixed (error bars: std over 3 runs).

## 1 Introduction

The ability of large language models to perform complex mathematical and scientific reasoning has improved dramatically over the past year. OpenAI’s o1 series [openai2024openaio1card] first demonstrated that reinforcement learning enables models to produce chains of thought [wei2022chain] that solve competition-level problems [openai2024openaio1card]. DeepSeek-R1 [Guo\_2025] and Kimi k1.5 [kimiteam2025kimik15scalingreinforcement] subsequently laid out open training recipes that brought these reasoning capabilities to open-weight models, exhibiting emergent behaviors such as self-verification, backtracking, and multi-step planning, and inspiring a wave of open-weight reasoning models with increasingly strong performance. In particular, the two-stage pipeline of SFT-then-RL popularized by large-scale post-training efforts [lambert2025tulu, Guo\_2025] became the standard approach for training reasoning models today: supervised fine-tuning (SFT) on expert-generated chain-of-thought demonstrations to equip the model with domain knowledge and output formatting through memorization [chu2025sft, mecklenburg2024injectingnewknowledgelarge], followed by reinforcement learning from verifiable rewards (RLVR) to sharpen the policy toward higher-reward reasoning strategies [chu2025sft, yue2025does, zhao2025echo, matsutani2026rl].

Recent work on post-training for LLM reasoning has moved beyond the standard SFT followed by RL pipeline toward *mixed-policy* methods that interleave or blend supervised and reinforcement learning signals during training. The motivation is intuitive: on-policy RL suffers from sparse rewards on hard problems where the model rarely generates a correct solution, and SFT alone does not develop the model’s own reasoning capacity. By combining on-policy rollouts with off-policy expert demonstrations, mixed-policy methods aim to leverage the complementary strengths of both: SFT’s ability to learn from demonstrations beyond the model’s initial capabilities, and RL’s ability to refine and sharpen existing reasoning.

A growing body of methods has emerged along this direction [yan2025learning, ma2026learning, fu2026srft, huang2026blending, lv2026unifiedviewlargelanguage, yuan2025mitigatingforgettingsupervisedreinforcement, guan2025recallextenddynamicsenhancingsmall, liu2025uft, zhang2026onpolicy, liu2025superrlreinforcementlearningsupervision, chen2025stepwiseadaptiveintegrationsupervised, wu2025templaterlstructuredtemplateguidedreinforcement], many claiming state-of-the-art results on mathematical reasoning benchmarks. Numerous methods [yan2025learning, ma2026learning, fu2026srft, yuan2025mitigatingforgettingsupervisedreinforcement, huang2026blending, lv2026unifiedviewlargelanguage, guan2025recallextenddynamicsenhancingsmall] follow the SFT setup of yan2025learning and report improvements over SFT and SFT→\toRL baselines. We show that their SFT baselines are weakened by bugs in shared training frameworks ([Section˜2](#S2 "2 Bugs in Widely-Used SFT Frameworks ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning")), and that a correctly implemented SFT followed by RL pipeline matches or exceeds their reported scores on average. The sequential pipeline also achieves this while using fewer FLOPs ([Figure˜1](#S0.F1 "In SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning"), left), as SFT ensures dense reward signal for RL from the start, whereas mixed-policy methods must learn and refine simultaneously under weak early signal.

Our contributions can be summarized as follows:

* •

  We discover that numerous recently published research papers on mixed-policy training rely on a faulty baseline: we identify and fix two SFT bugs – a CPU-offloaded optimizer bug in DeepSpeed [DeepSpeed] that silently drops intermediate micro-batches during gradient accumulation (affecting multiple downstream training frameworks), and a loss aggregation bug in OpenRLHF [hu-etal-2025-openrlhf] that incorrectly weights per-mini-batch losses. Together they deflate SFT performance by up to 5.7 points with Qwen2.5-Math-7B.
* •

  Once corrected, the standard SFT followed by RL pipeline achieves state-of-the-art results, surpassing the best published mixed-policy method by +3.8 points on in-distribution tasks with Qwen2.5-Math-7B and by +22.2 points with Llama-3.1-8B, while maintaining generalizability on out-of-distribution benchmarks. Even a truncated variant with just 50 instead of 500 RL iterations outperforms all mixed-policy methods on math benchmarks while using fewer FLOPs, calling their claimed improvements into question.
* •

  Our findings restore confidence in the SFT followed by RL paradigm and highlight the importance of cross-framework validation, as silent bugs in widely used SFT pipelines were sufficient to systematically deflate baselines across multiple independent studies.

## 2 Bugs in Widely-Used SFT Frameworks

We identify two bugs in widely used open-source training frameworks that silently degrade SFT quality. Both are triggered by distributed training configurations, making them difficult to detect without cross-framework validation.

### 2.1 CPU-Offloaded Optimizer Bug

yan2025learning use OpenRLHF [hu-etal-2025-openrlhf] with DeepSpeed [DeepSpeed] ZeRO Stage 2 and CPU-offloaded Adam to reduce GPU memory consumption during SFT training. However, a bug in the CPU-offloaded gradient accumulation routine causes only the first micro-batch’s gradients to reach the optimizer. Specifically, the offloading code copies gradients to CPU inside an else branch that executes only when micro\_step\_id == 0 (i.e., the first micro-batch), while intermediate micro-batches accumulate gradients on GPU correctly but never trigger a copy. On the next full optimizer step, the CPU-side optimizer therefore sees only the first micro-batch’s gradients rather than the accumulated sum. The fix is straightforward: moving the copy\_gradients\_to\_cpu() call outside the else branch so it executes after every micro-batch’s accumulation. This bug affects any framework using DeepSpeed ZeRO Stage 1 or 2 with an offloaded optimizer, including TRL [vonwerra2020trl] and Llama-Factory [zheng2024llamafactory]; ma2026learning use Llama-Factory for SFT, though they have not released their training configuration. The mixed-policy methods themselves are unaffected, as they are implemented entirely in verl [Sheng\_2025], which does not use DeepSpeed, the standalone SFT baselines they compare against are impacted.

This bug was introduced in September 2024 in DeepSpeed PR #6550.111<https://github.com/deepspeedai/DeepSpeed/pull/6550> We validate the fix against two independent baselines: the GPU-resident DeepSpeed optimizer in OpenRLHF and the PyTorch FSDP [FSDP] optimizer in verl. We show in [Section˜4](#S4 "4 Main Experiments ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning") that the patched optimizer matches both in average loss and gradient norms. We submitted a pull request to DeepSpeed with the fix, which has been merged upstream; the patch is detailed in [Appendix˜C](#A3 "Appendix C DeepSpeed Bug Fix Patch ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning").

### 2.2 Loss Aggregation Bug

The standard cross-entropy loss for SFT averages over all response tokens in a batch, masking out both padding and prompt tokens. As documented by unsloth2024gradient and huggingface2024gradient, a common bug in gradient accumulation computes a *mean of per-mini-batch means* instead of the true per-token mean: since mini-batches contain different numbers of response tokens, this weights each mini-batch equally regardless of how many active tokens it contains. The same distortion arises across distributed data-parallel ranks, where each rank independently computes its local mean loss before averaging across ranks. OpenRLHF [hu-etal-2025-openrlhf], Llama-Factory [zheng2024llamafactory], and early versions of verl [Sheng\_2025] all exhibit this bug. The fix requires aggregating token-level loss sums and counts across all ranks and mini-batches before dividing. Verl fixed this in November 2025222<https://github.com/verl-project/verl/pull/3994>; OpenRLHF and Llama-Factory have not as of the time of writing. We have submitted a pull request to OpenRLHF with the fix; the pseudocode is detailed in [Appendix˜D](#A4 "Appendix D OpenRLHF Loss Aggregation Bug Fix Pseudocode ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning").

This bug is inherited from pretraining codebases, where data packing ensures equal active token counts across ranks, making the mean-of-means equivalent to the true per-token mean. In SFT, however, prompts and responses vary in length across samples, so mini-batches and ranks almost always have different active token counts, meaning the bug affects almost every training step. Its impact on SFT performance is smaller than the CPU-offloaded optimizer bug, but not negligible: as shown in [Table˜2](#S4.T2 "In Decomposing the baseline gap. ‣ 4 Main Experiments ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning"), fixing the loss aggregation bug on top of the optimizer fix still yields a measurable improvement and stabilizes loss variability (see [Section˜4](#S4 "4 Main Experiments ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning")).

## 3 Experimental Setup

We reproduce the SFT baselines and mixed-policy methods under controlled conditions to isolate the impact of the bugs described in [Section˜2](#S2 "2 Bugs in Widely-Used SFT Frameworks ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning"). All methods share the same base model, dataset, and evaluation protocol, which we detail below before describing the method-specific setups.

#### Training dataset.

The training dataset is OpenR1-Math-46k-8192 [yan2025learning], a length-filtered subset of OpenR1-Math-220k [openr1] whose prompts are drawn from NuminaMath 1.5 [li2024numinamath] and whose off-policy reasoning traces are generated by DeepSeek-R1 [Guo\_2025]. yan2025learning filter out generations longer than 8192 tokens and those verified incorrect by Math-Verify333<https://github.com/huggingface/Math-Verify>. The dataset contains approximately 46k prompts paired with high-quality demonstrations. We use the full dataset for both SFT (prompts paired with demonstrations) and RL (prompts paired with ground-truth answers for reward verification). This dataset is also used by LUFFY [yan2025learning], ReLIFT [ma2026learning], SRFT [fu2026srft], Prefix-RFT [huang2026blending], and HPT [lv2026unifiedviewlargelanguage] as their common training set.

#### Models.

We follow previous works [yan2025learning, ma2026learning, fu2026srft, yuan2025mitigatingforgettingsupervisedreinforcement, huang2026blending, lv2026unifiedviewlargelanguage] and use Qwen2.5-Math-7B [yang2024qwen25mathtechnicalreportmathematical] with an increased max context length from 4096 to 16,384 and an increased RoPE theta from 10,000 to 40,000 as the base model. We also run experiments on Llama-3.1-8B [grattafiori2024llama3herdmodels] to verify generalization across model families.

#### Evaluation.

Following the evaluation protocol used by the mixed-policy methods [yan2025learning, ma2026learning, fu2026srft, huang2026blending, lv2026unifiedviewlargelanguage], we evaluate on six mathematical reasoning benchmarks: AIME24, AIME25, AMC [li2024numinamath], MATH-500 [hendrycks2021measuring], Minerva [lewkowycz2022solving], and OlympiadBench (Olympiad) [he-etal-2024-olympiadbench]. For benchmarks with limited sample sizes (AIME24/25 and AMC), we report avg@32; for the rest, we use pass@1. We also evaluate on the ARC-Challenge (ARC-c) [clark2018thinksolvedquestionanswering], GPQA-Diamond (GPQA) [rein2024gpqa], and MMLU-Pro [wang2024mmlupro] to assess out-of-distribution generalization, as these benchmarks cover science and general knowledge domains outside the mathematical training distribution from OpenR1-Math-46k-8192. For all multiple-choice questions, we randomly shuffle the option order to mitigate information leakage. The evaluation temperature is 0.6 with a maximum response length of 8192 tokens. We use Math-Verify as the verifier during training and for testing. For each method that uses SFT, we run 3 independent training runs with different random seeds; for SFT→\toRL pipelines, both the SFT and the subsequent RL stages use different seeds across runs, so each of the 3 runs are independent end-to-end.

Table 1: Corrected baselines vs. mixed-policy methods on Qwen2.5-Math-7B (mean ±\pm std over 3 seeds for our baselines). Left: in-distribution (ID) benchmarks. Right: out-of-distribution (OOD) benchmarks. Bold indicates best results, underline indicates second-best.

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | AIME 24 | AIME 25 | AMC | MATH-500 | Minerva | Olympiad | Avg ID | ARC-c | GPQA | MMLU-Pro | Avg OOD |
| *Corrected baselines* (verl) | | | | | | | | | | | |
| SFT | 33.6±\pm1.4 | 28.2±\pm0.7 | 65.4±\pm0.9 | 90.0±\pm0.9 | 40.6±\pm0.6 | 55.7±\pm0.6 | 52.2±\pm0.2 | 82.5±\pm0.6 | 24.6±\pm2.9 | 50.2±\pm0.3 | 52.4±\pm1.1 |
| SFT→\toRL | 40.4±\pm0.3 | 29.8±\pm0.7 | 73.2±\pm1.2 | 92.0±\pm0.2 | 43.9±\pm0.6 | 62.8±\pm0.6 | 57.0±\pm0.3 | 84.0±\pm0.7 | 40.6±\pm1.8 | 55.1±\pm0.6 | 59.9±\pm0.9 |
| *Mixed-policy methods (reproduction attempt)* | | | | | | | | | | | |
| LUFFY [yan2025learning] | 25.0 | 16.9 | 64.2 | 85.2 | 35.7 | 50.8 | 46.3 | 81.4 | 41.8 | 50.9 | 58.0 |
| ReLIFT [ma2026learning] | 25.6 | 21.0 | 63.0 | 86.2 | 41.5 | 55.6 | 48.8 | 80.7 | 36.7 | 51.3 | 56.2 |
| *Mixed-policy methods (reported)* | | | | | | | | | | | |
| LUFFY [yan2025learning] | 29.4 | 23.1 | 65.6 | 87.6 | 37.5 | 57.2 | 50.1 | 80.5 | 39.9 | 53.0 | 57.8 |
| ReLIFT [ma2026learning] | 28.3 | 22.9 | 65.1 | 87.9 | – | 57.3 | – | 81.6 | 43.1 | 53.9 | 59.5 |
| SRFT [fu2026srft] | 35.3 | 21.6 | 74.3 | 89.8 | 39.7 | 58.3 | 53.2 | 85.3 | 46.4 | 55.9 | 62.5 |
| Prefix-RFT [huang2026blending] | 31.8 | 26.4 | 68.2 | 88.4 | 40.3 | 55.7 | 51.8 | 84.0 | 39.1 | 52.1 | 58.4 |
| HPT [lv2026unifiedviewlargelanguage] | 33.0 | 21.9 | 69.4 | 89.2 | 46.0 | 56.9 | 52.7 | 81.6 | 42.9 | 52.5 | 59.0 |

#### Baselines.

We compare SFT→\toRL, which first fine-tunes on expert traces then runs GRPO for 500 steps, against five mixed-policy methods. All methods use Qwen2.5-Math-7B as the base model and train on the OpenR1-Math-46k-8192 dataset with 8 rollouts per prompt; every method trains for 500 steps. LUFFY [yan2025learning], which mixes off-policy expert traces into on-policy GRPO; ReLIFT [ma2026learning], which interleaves SFT and RL phases; SRFT [fu2026srft], which jointly optimizes SFT and RL losses within each batch; Prefix-RFT [huang2026blending], which samples prefixes from expert demonstrations for on-policy continuation; and HPT [lv2026unifiedviewlargelanguage], which dynamically switches between SFT and GRPO based on rollout accuracy. Full descriptions and discussion of each method’s baseline issues are in [Section˜5](#S5 "5 Related Work ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning"); our SFT and RL hyperparameters are in [Appendix˜A](#A1 "Appendix A Technical Setup and Hyperparameters ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning"). We exclude MIFO [yuan2025mitigatingforgettingsupervisedreinforcement] and RED [guan2025recallextenddynamicsenhancingsmall], which share the same training data but are not directly comparable: the former reports on different evaluation benchmarks, and the latter does not evaluate on our model families.

#### Reproduction of OpenRLHF vs. verl training.

We train four SFT variants using OpenRLHF, progressively fixing the loss aggregation bug, the CPU-offloaded optimizer bug, or both, and compare against an independently implemented verl baseline to isolate each bug’s contribution on a representative subset of benchmarks ([Table˜2](#S4.T2 "In Decomposing the baseline gap. ‣ 4 Main Experiments ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning")). For this reproduction we use a single node rather than the 4 nodes specified in our technical setup ([Appendix˜A](#A1.SS0.SSS0.Px2 "SFT hyperparameters. ‣ Appendix A Technical Setup and Hyperparameters ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning")), since the number of nodes affects the reported gradient norm.

#### Reproduction of LUFFY and ReLIFT.

We select LUFFY and ReLIFT as the two mixed-policy methods to fully reimplement into a recent version of the verl codebase we use, using their original hyperparameters. Due to the high computational cost of each run, we report single-seed results; the performance gaps substantially exceed the variance observed across our 3-seed baselines. Because verl does not use DeepSpeed, the optimizer bug affects only the standalone SFT baselines these methods compare against, not the methods themselves, deflating the baseline side of every comparison. The same asymmetry applies to the other mixed-policy methods. We do, however, patch the loss aggregation issue ([Section˜2.2](#S2.SS2 "2.2 Loss Aggregation Bug ‣ 2 Bugs in Widely-Used SFT Frameworks ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning")) in LUFFY’s RL component and in ReLIFT’s SFT and RL stage losses.

#### Chat template.

Prior works [yan2025learning, ma2026learning, fu2026srft, yuan2025mitigatingforgettingsupervisedreinforcement, huang2026blending, lv2026unifiedviewlargelanguage, guan2025recallextenddynamicsenhancingsmall] use a simplified template for Llama-3.1-8B, reporting that it cannot follow the full Qwen system prompt [yan2025learning]. With a correctly implemented SFT stage, Llama-3.1-8B follows the full prompt without issue, so we use the same template for both model families ([Appendix˜E](#A5 "Appendix E Chat Template ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning")). This is consistent with the hypothesis that the inability to follow the system prompt was a symptom of undertrained SFT rather than a model limitation.

## 4 Main Experiments

[Table˜1](#S3.T1 "In Evaluation. ‣ 3 Experimental Setup ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning") compares the corrected baselines against published mixed-policy methods. On in-distribution tasks, the corrected SFT alone (52.2) already surpasses LUFFY (46.3), ReLIFT (48.8), and Prefix-RFT (51.8); adding RL pushes the average to 57.0, outperforming the next-best method, SRFT (53.2), by +3.8 points. The perceived gains stem from comparison against deflated SFT baselines, caused by framework bugs for LUFFY and ReLIFT ([Table˜2](#S4.T2 "In Decomposing the baseline gap. ‣ 4 Main Experiments ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning")), and by suboptimal hyperparameters for SRFT ([Table˜3](#S4.T3 "In Hyperparameter sensitivity. ‣ 4 Main Experiments ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning")). Prefix-RFT [huang2026blending] inherits the same deflated baselines from LUFFY [yan2025learning]. On out-of-distribution tasks, SFT→\toRL (59.9) is second only to SRFT (62.5), suggesting the pipeline generalizes well beyond math reasoning benchmarks.

#### Decomposing the baseline gap.

[Table˜2](#S4.T2 "In Decomposing the baseline gap. ‣ 4 Main Experiments ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning") isolates the contribution of each bug in the OpenRLHF SFT pipeline. Fixing only the loss aggregation bug yields a modest improvement from 48.3 to 49.1 average score, but fixing only the CPU-offloaded optimizer bug recovers nearly the entire gap, raising the average from 48.3 to 53.4. Applying both fixes together reaches 54.0: the optimizer bug accounts for the larger share of the gap, with the loss aggregation bug contributing an additional improvement on top. The fully patched OpenRLHF matches the 53.8 average score achieved by the independently implemented verl SFT.

[Figure˜2](#S4.F2 "In Decomposing the baseline gap. ‣ 4 Main Experiments ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning") shows that, beyond final scores, the fixes also stabilize training dynamics: the buggy OpenRLHF SFT exhibits both a shifted mean loss and high variability; fixing the CPU-offloaded optimizer corrects the mean level but leaves the variability; fixing the loss aggregation alone reduces the variability but does not shift the mean; applying both brings the loss curve in line with verl (correct mean, low variability). For gradient norms, only the optimizer bug has an effect: the buggy configuration reports substantially lower gradient norms, while the patched optimizer matches both the GPU-resident OpenRLHF optimizer and the verl baseline. This stability carries over to the subsequent RL phase: RL initialized from a correctly trained SFT checkpoint starts with a higher initial reward and achieves a higher final score.

Table 2: Impact of OpenRLHF bugs on SFT performance on Qwen2.5-Math-7B (mean ±\pm std over 3 seeds for our baselines). Each row isolates the contribution of each bug fix. The CPU-offloaded optimizer bug accounts for nearly all of the performance gap; fixing the loss aggregation bug alone has a smaller effect.

| SFT Configuration | AIME 24 | AIME 25 | AMC | MATH-500 | Olympiad | MMLU-Pro | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OpenRLHF baseline | 25.8±\pm0.7 | 25.2±\pm0.5 | 59.0±\pm1.1 | 85.9±\pm1.0 | 50.4±\pm1.8 | 43.8±\pm1.8 | 48.3±\pm0.8 |
| + fix loss aggregation | 28.7±\pm2.4 | 24.4±\pm0.3 | 60.1±\pm1.2 | 85.5±\pm1.0 | 51.8±\pm0.7 | 44.0±\pm1.7 | 49.1±\pm0.9 |
| + fix optimizer | 35.1±\pm0.9 | 26.9±\pm0.3 | 64.9±\pm1.3 | 88.6±\pm0.2 | 55.3±\pm1.1 | 49.4±\pm0.5 | 53.4±\pm0.4 |
| + fix both | 34.9±\pm0.7 | 28.2±\pm1.4 | 65.5±\pm0.6 | 89.7±\pm1.1 | 55.6±\pm0.7 | 49.8±\pm0.8 | 54.0±\pm0.2 |
| verl | 33.6±\pm1.4 | 28.2±\pm0.7 | 65.4±\pm0.9 | 90.0±\pm0.9 | 55.7±\pm0.6 | 50.2±\pm0.3 | 53.8±\pm0.1 |

!(/html/2604.23747/assets/x2.png)

Figure 2: SFT training stability across configurations. Left: training loss. Right: gradient norm. Each bug has a distinct effect on the loss: the aggregation bug introduces variability while the optimizer bug shifts the mean. Only both fixes together match the verl baseline. For gradient norms, only the optimizer bug has an effect, suppressing norms well below the verl reference, because only the first micro-batch’s gradients reach the optimizer.

#### Hyperparameter sensitivity.

Beyond framework bugs, the choice of SFT hyperparameters also affects baseline strength. Our SFT configuration ([Appendix˜A](#A1.SS0.SSS0.Px2 "SFT hyperparameters. ‣ Appendix A Technical Setup and Hyperparameters ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning")) follows yan2025learning and ma2026learning, both using the same settings. fu2026srft, however, uses a batch size of 128, a learning rate of 5×10−65\times 10^{-6} with a linear schedule and 10% warmup. We reproduced the SFT baseline with fu2026srft’s reported hyperparameters and obtained a weaker baseline, with the average score dropping from 53.8 to 48.3 ([Table˜3](#S4.T3 "In Hyperparameter sensitivity. ‣ 4 Main Experiments ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning")). Switching to the LUFFY/ReLIFT hyperparameters recovers this gap, which in turn reduces the margin between the SFT baseline and the final result of fu2026srft (55.9 average score on this set of evaluations; see [Table˜1](#S3.T1 "In Evaluation. ‣ 3 Experimental Setup ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning")) from 7.6 points down to 2.1 points.

Table 3: Impact of SFT hyperparameters on baseline performance on Qwen2.5-Math-7B (mean ±\pm std over 3 seeds for our baselines). Reproducing the SFT stage of fu2026srft with their hyperparameters (10x lower learning rate, 2x batch size, linear schedule) yields a weaker baseline; switching to the LUFFY/ReLIFT hyperparameters recovers 5.5 points on average.

| SFT Baseline | AIME 24 | AIME 25 | AMC | MATH-500 | Olympiad | MMLU-Pro | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fu2026srft reported | 31.1 | 20.3 | 62.8 | 85.2 | 53.3 | 45.7 | 49.7 |
| fu2026srft reproduced | 26.0±\pm0.1 | 23.5±\pm1.1 | 58.7±\pm0.4 | 85.5±\pm1.1 | 50.0±\pm0.6 | 46.5±\pm0.3 | 48.3±\pm0.2 |
| Tuned (ours) | 33.6±\pm1.4 | 28.2±\pm0.7 | 65.4±\pm0.9 | 90.0±\pm0.9 | 55.7±\pm0.6 | 50.2±\pm0.3 | 53.8±\pm0.1 |

### 4.1 Extension to Llama-3.1-8B

To verify that our findings are not specific to Qwen2.5-Math-7B, we repeat the experiment on Llama-3.1-8B. [Table˜4](#S4.T4 "In 4.1 Extension to Llama-3.1-8B ‣ 4 Main Experiments ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning") reports results on the benchmark subset for which all methods provide scores. The pattern is even more pronounced: the corrected SFT→\toRL baseline achieves 43.7 average score, outperforming every published mixed-policy method by +22.2 points, with the closest competitor being HPT at 21.5. The corrected SFT baseline alone reaches 33.9, already exceeding all mixed-policy results by +12.4 points.

The failure of mixed-policy methods on Llama is particularly instructive. Unlike Qwen, which is pre-trained with emphasis on math data, Llama is a general-purpose model with little mathematical reasoning in its pre-training distribution [shao2026spuriousrewardsrethinkingtraining], making the bootstrapping role of SFT far more critical. The training dynamics in [Figure˜3](#S4.F3 "In 4.2 Training Dynamics ‣ 4 Main Experiments ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning") (bottom row) make this concrete: with SFT→\toRL, the RL phase starts at approximately 60% training reward because the preceding SFT stage has already injected the mathematical knowledge that Llama lacks natively. Mixed-policy methods, by contrast, start near zero reward and improve only slowly: LUFFY’s training reward remains largely flat over 500 steps, indicating that the off-policy demonstrations provide a learning signal but one far too sparse to match the dense reward that SFT→\toRL enjoys from the start.

This points to a fundamental efficiency gap: mixed-policy methods must simultaneously bootstrap mathematical reasoning and refine it through RL, but the RL signal is extremely weak when the model cannot yet produce correct solutions on its own. The off-policy demonstrations provide some supervision, but it is diluted by on-policy rollouts that carry little to no reward signal early in training. SFT-then-RL cleanly separates these two objectives: SFT first bootstraps the model into a regime where it reliably generates correct solutions, and RL then refines this already-capable policy with dense reward signal from the start. While mixed-policy methods may eventually converge to comparable scores given enough steps, they do so far less efficiently because the RL component contributes minimally until the model has acquired sufficient knowledge – precisely the knowledge that a dedicated SFT stage provides upfront.

Table 4: Corrected baselines versus published mixed-policy methods on Llama-3.1-8B. We report the subset of benchmarks for which all methods provide a score. Bold indicates best results, underline indicates second-best. \* denotes results taken from the respective papers, which use a different chat template (see [Section˜3](#S3.SS0.SSS0.Px7 "Chat template. ‣ 3 Experimental Setup ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning")).

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Method | AIME 24 | AMC | MATH-500 | Minerva | Olympiad | Avg |
| SFT (ours) | 7.0±\pm1.1 | 40.4±\pm0.6 | 68.4±\pm2.3 | 17.3±\pm0.9 | 36.2±\pm1.1 | 33.9±\pm0.9 |
| SFT→\toRL (ours) | 16.5±\pm0.8 | 53.9±\pm1.3 | 78.6±\pm1.2 | 20.7±\pm1.5 | 48.5±\pm0.1 | 43.7±\pm0.2 |
| LUFFY [yan2025learning] | 0.8 | 13.1 | 34.6 | 12.5 | 10.8 | 14.4 |
| ReLIFT [ma2026learning] | 0.6 | 14.5 | 35.4 | 14.7 | 12.8 | 15.6 |
| SRFT\* [fu2026srft] | 1.9 | 14.3 | 40.1 | 15.3 | 9.5 | 16.2 |
| Prefix-RFT\* [huang2026blending] | 1.3 | 13.3 | 40.6 | 18.1 | 11.9 | 17.0 |
| HPT\* [lv2026unifiedviewlargelanguage] | 2.1 | 18.6 | 47.8 | 18.8 | 20.4 | 21.5 |

### 4.2 Training Dynamics

[Figure˜3](#S4.F3 "In 4.2 Training Dynamics ‣ 4 Main Experiments ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning") compares the training dynamics of the RL phase of SFT→\toRL against ReLIFT and LUFFY on both models. For Qwen, the RL phase of SFT→\toRL starts above 80% training reward, because the preceding SFT stage has already internalized expert knowledge, and continues to improve. Mixed-policy methods [yan2025learning, ma2026learning] never reach the reward level that SFT→\toRL begins with, suggesting that interleaving off- and on-policy signals within a single stage is less efficient than performing them sequentially, or even than performing SFT alone. On Llama-3.1-8B, the gap is starker: LUFFY and ReLIFT never exceed 30% training reward after 500 steps, while SFT→\toRL starts at approximately 60% and continues climbing.

For response length, the RL phase of SFT→\toRL starts at approximately 5k tokens on Qwen, reflecting the verbose style inherited from SFT on DeepSeek-R1 demonstrations, and gradually converges toward 3.7k tokens as RL prunes unnecessary verbosity while retaining problem-solving capability. Mixed-policy methods instead show gradually increasing lengths, which ma2026learning interpret as developing more thorough reasoning. On Llama, mixed-policy response lengths quickly diverge toward the context limit.

For entropy, yan2025learning and ma2026learning emphasize that their approaches maintain higher entropy compared to pure RL, which they attribute to sustained exploration. In SFT→\toRL, the entropy remains relatively constant for both models, consistent with RL refining an already strong policy rather than exploring from scratch.

!(/html/2604.23747/assets/x3.png)

Figure 3: Training dynamics comparison between the RL part of SFT→\toRL, LUFFY, and ReLIFT. Top row: Qwen2.5-Math-7B. Bottom row: Llama-3.1-8B. Left to right: training reward, mean response length, and policy entropy.

### 4.3 Training Efficiency

Since the post-SFT Qwen model already exhibits strong performance, we investigate whether the RL phase can be shortened without significant degradation.
Concretely, we increase the learning rate from 1×10−61\times 10^{-6} to 5×10−65\times 10^{-6} and train for only 50 RL steps: 10×\times fewer than the default setup described in [Appendix˜A](#A1.SS0.SSS0.Px3 "RL hyperparameters. ‣ Appendix A Technical Setup and Hyperparameters ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning").

[Table˜5](#S4.T5 "In 4.3 Training Efficiency ‣ 4 Main Experiments ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning") summarizes the results.
On in-distribution benchmarks, the shortened schedule achieves an average score of 55.6, only 1.4 points behind the full 500-step run and still +2.4 points ahead of the next-best mixed-policy baseline (SRFT at 53.2, see [Table˜1](#S3.T1 "In Evaluation. ‣ 3 Experimental Setup ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning")).
On out-of-distribution benchmarks, the gap between the full run (59.9) and the short run (59.2) is similarly narrow at 0.7 points, while remaining competitive with mixed-policy baselines (only surpassed by SRFT at 62.5 and ReLIFT at 59.5).
These results suggest that, given a strong SFT initialization, the majority of RL gains can materialize within the early stages of RL training.

Table 5: Faster RL on Qwen2.5-Math-7B. Left: ID benchmarks. Right: OOD benchmarks.

| Method | AIME 24 | AIME 25 | AMC | MATH-500 | Minerva | Olympiad | Avg ID | ARC-c | GPQA | MMLU-Pro | Avg OOD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SFT | 33.6±\pm1.4 | 28.2±\pm0.7 | 65.4±\pm0.9 | 90.0±\pm0.9 | 40.6±\pm0.6 | 55.7±\pm0.6 | 52.2±\pm0.2 | 82.5±\pm0.6 | 24.6±\pm2.9 | 50.2±\pm0.3 | 52.4±\pm1.1 |
| SFT→\toRL (50 steps) | 37.1±\pm2.7 | 29.4±\pm0.6 | 70.3±\pm1.5 | 91.7±\pm0.8 | 44.1±\pm0.0 | 61.3±\pm0.9 | 55.6±\pm0.5 | 84.0±\pm0.5 | 39.6±\pm1.0 | 53.9±\pm0.4 | 59.2±\pm0.2 |
| SFT→\toRL | 40.4±\pm0.3 | 29.8±\pm0.7 | 73.2±\pm1.2 | 92.0±\pm0.2 | 43.9±\pm0.6 | 62.8±\pm0.6 | 57.0±\pm0.3 | 84.0±\pm0.7 | 40.6±\pm1.8 | 55.1±\pm0.6 | 59.9±\pm0.9 |

Table 6: Estimated training FLOPs for each method on Qwen2.5-Math-7B.

| Method | FLOPs (×1019\times 10^{19}) |
| --- | --- |
| SFT→\toRL (50 steps) | 3.63 |
| LUFFY | 6.65 |
| ReLIFT | 8.76 |

#### FLOPs.

[Table˜6](#S4.T6 "In 4.3 Training Efficiency ‣ 4 Main Experiments ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning") summarizes the results (calculation details in [Appendix˜B](#A2 "Appendix B FLOPs Calculation Details ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning")). Our truncated SFT→\toRL pipeline requires 3.63×10193.63\times 10^{19} FLOPs, fewer than LUFFY (6.65×10196.65\times 10^{19}) and ReLIFT (8.76×10198.76\times 10^{19}). SFT is cheap relative to RL as it processes tokens without the overhead of multiple rollouts, and the strong initialization it provides allows truncating RL to just 50 steps. Since SFT→\toRL (50 steps) also outperforms both LUFFY and ReLIFT on ID benchmarks and matches OOD benchmarks ([Table˜1](#S3.T1 "In Evaluation. ‣ 3 Experimental Setup ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning")), the pipeline delivers stronger results at lesser compute.

## 5 Related Work

We distinguish between methods we directly evaluate in our controlled setting and those we discuss qualitatively due to differences in training setups, datasets, or evaluation protocols.

LUFFY [yan2025learning] extends GRPO by mixing on-policy rollouts with off-policy expert traces. It applies regularized importance sampling with a shaped policy f​(π)=π/(π+λ)f(\pi)=\pi/(\pi+\lambda) to control distribution mismatch and avoid imitation collapse. LUFFY uses OpenRLHF [hu-etal-2025-openrlhf] for SFT, which is affected by the bugs described in [Section˜2](#S2 "2 Bugs in Widely-Used SFT Frameworks ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning"). ReLIFT [ma2026learning] alternates between SFT and RL phases. It identifies unsolved problems via pass rate, applies SFT on expert demonstrations to inject missing knowledge, and resumes RL. This iterative process shrinks the set of hard problems over time. ReLIFT uses Llama-Factory [zheng2024llamafactory] for SFT, which is affected by the bugs described in [Section˜2](#S2 "2 Bugs in Widely-Used SFT Frameworks ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning"). SRFT [fu2026srft] jointly optimizes supervised and reinforcement losses within each batch, using an adaptive entropy-based weight to balance the two signals. However, its SFT baseline uses a tenfold lower learning rate than other methods, producing a weaker starting point that inflates the apparent mixed-policy gain (see [Section˜4](#S4.SS0.SSS0.Px2 "Hyperparameter sensitivity. ‣ 4 Main Experiments ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning")). Prefix-RFT [huang2026blending] integrates demonstrations into RFT by sampling a prefix from expert solutions and generating on-policy continuations, forming hybrid trajectories for policy updates. It adopts its SFT baselines from LUFFY, which are affected by the bugs described in [Section˜2](#S2 "2 Bugs in Widely-Used SFT Frameworks ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning"). HPT [lv2026unifiedviewlargelanguage] unifies SFT and RL under a common policy gradient objective and uses a binary gate: SFT when all rollouts fail, GRPO otherwise. Notably, its own ablations show that pure SFT matches or beats off-policy RL for the offline signal. HPT does not detail its SFT setup, stating only that they "endeavored to follow previous works as closely as possible".

#### Other mixed-policy methods that may warrant SFT auditing.

UFT [liu2025uft] and CHORD [zhang2026onpolicy] combine SFT and RL into a single loss, with CHORD using dynamic weighting and token-level uncertainty scaling. SuperRL [liu2025superrlreinforcementlearningsupervision] instead applies RL only when correct rollouts exist and falls back to SFT otherwise, avoiding signal mixing and highlighting risks of off-policy collapse.
SASR [chen2025stepwiseadaptiveintegrationsupervised] adaptively balances SFT and RL using a KL-based switching signal.
RED [guan2025recallextenddynamicsenhancingsmall] dynamically balances SFT and RL by regulating their contributions based on entropy changes and sample accuracy. It introduces an accuracy-aware policy shift and entropy-based weighting to control when the model imitates offline data versus explores with RL, aiming to stabilize training under distribution mismatch and limited exploration. TemplateRL [wu2025templaterlstructuredtemplateguidedreinforcement] increases exploration diversity by augmenting prompts with MCTS-derived templates, achieving large gains while remaining fully on-policy. MIFO [yuan2025mitigatingforgettingsupervisedreinforcement] interleaves RL and SFT by buffering hard questions for targeted SFT, while freezing RL-critical parameters during SFT phases to mitigate catastrophic forgetting.

## 6 Conclusion and Limitations

We have shown that the reported gains of mixed-policy optimization methods for LLM reasoning trace back to deflated baselines rather than methodological innovation. A CPU-offloaded optimizer bug in DeepSpeed silently drops intermediate micro-batches during gradient accumulation, affecting downstream frameworks including TRL, OpenRLHF and Llama-Factory; fixing it alone recovers most of the gap. A second loss aggregation bug in OpenRLHF incorrectly weights per-mini-batch losses and, on top of adding training instability, contributes a further measurable degradation. Together the two fixes close the gap to the independently implemented verl baseline.

Once corrected, the standard SFT followed by RL pipeline, to the best of our knowledge, exceeds all five mixed-policy methods we evaluate on in-distribution tasks and achieves comparable or superior out-of-distribution performance, on both Qwen2.5-Math-7B and Llama-3.1-8B. The corrected pipeline is also more efficient, converging in 50 RL steps using fewer FLOPs than LUFFY or ReLIFT, because SFT bootstraps the model into a regime with dense RL reward signal rather than requiring simultaneous knowledge acquisition and refinement. These findings carry two implications: first, framework-level diversity is essential for robustness, as silent bugs in widely used SFT pipelines were sufficient to deflate baselines across multiple papers; second, empirical ML results should be cross-validated across independently implemented frameworks to guard against systematic baseline deflation. Because these bugs affect general-purpose SFT pipelines rather than mixed-policy code specifically, they could potentially invalidate research findings in other areas that rely on the same affected frameworks.

#### Limitations.

Our study focuses on mathematical reasoning with Qwen2.5-Math-7B and Llama-3.1-8B; we do not train on other domains (e.g., code, general reasoning) or larger scales, where SFT quality may matter differently. We also do not consider other model families, since the mixed-policy papers themselves do not. There is a nuance in our comparison: these methods position themselves as single-stage alternatives to SFT-then-RL, and that framing is their core contribution. Applying mixed-policy training atop a correctly trained SFT checkpoint might yield further gains, but none of these works report such a setup, as it would counter their thesis. Our results therefore invalidate the published comparisons without precluding that a mixed-policy stage could add value on top of a proper SFT baseline. Finally, for ReLIFT and HPT we could not fully verify the SFT setup: ReLIFT uses Llama-Factory but has not released its exact configuration, and HPT does not detail its SFT hyperparameters; our assessment assumes both inherit the same affected pipeline.

## Acknowledgments and Disclosure of Funding

We thank Nathan Ranchin, Lorenzo Paleari, and Yixuan Xu for helpful discussions. This work was conducted during Alexis Limozin’s Master thesis at the ETH AI Center, while he was a Master student at EPFL. This work was supported as part of the Swiss AI Initiative by compute grant infra01 from the Swiss National Supercomputing Centre (CSCS) on Alps.

## Appendix A Technical Setup and Hyperparameters

#### Technical setup.

All experiments were conducted on 4 nodes of 4 NVIDIA GH200 GPUs (16 GPUs total). All training runs use the verl [Sheng\_2025] framework, except for the SFT bug reproduction runs in [Table˜2](#S4.T2 "In Decomposing the baseline gap. ‣ 4 Main Experiments ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning") which use OpenRLHF [hu-etal-2025-openrlhf] paired with DeepSpeed [DeepSpeed]. We use PyTorch Fully Sharded Data Parallel [FSDP] for distributed training, and vLLM [vllm2023] for rollout generation. The bugs described in [Section˜2](#S2 "2 Bugs in Widely-Used SFT Frameworks ‣ SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning") were identified in DeepSpeed v0.18.9 and OpenRLHF v0.9.10.

#### SFT hyperparameters.

Batch size 64, learning rate 5×10−55\times 10^{-5} with cosine scheduling (10% warmup, minimum ratio 0.1), AdamW with β1=0.9\beta\_{1}=0.9, β2=0.999\beta\_{2}=0.999, and weight decay 0.01, trained for 3 epochs.

#### RL hyperparameters.

Rollout batch size 128, 8 rollouts per prompt, mini-batch size 64, constant learning rate 1×10−61\times 10^{-6}, entropy coefficient 0, temperature 1.0, maximum response length 8192 tokens, trained for 500 steps. Following recent works [yu2025dapo, liu2025understanding, yan2025learning], we use an improved version of GRPO with asymmetric clipping [yu2025dapo] (ϵlow=0.2\epsilon\_{\text{low}}=0.2 and ϵhigh=0.28\epsilon\_{\text{high}}=0.28) and token-level loss [yu2025dapo], remove the KL loss term [yu2025dapo, yan2025learning], and remove length normalization and standard error normalization [liu2025understanding].

## Appendix B FLOPs Calculation Details

To enable a fair comparison of overall training costs across methods employing different framework-level optimization strategies, we estimate FLOPs following [zhang2025bread, snell2025scaling, hoffmann2022an, sardana2024beyond].
A forward pass through a model with NN parameters on DD tokens costs approximately 2​N​D2ND FLOPs, and a backward pass costs roughly 4​N​D4ND; supervised finetuning therefore requires 6​N​D6ND FLOPs per sample.
For RL, we additionally account for rollout generation by the inference engine separately from the training forward pass: each on-policy sequence is generated once by the inference engine (2​N​D2ND FLOPs) and then re-processed by the training engine for the policy gradient update (one forward + one backward, 6​N​D6ND FLOPs), for a total of 8​N​D8ND FLOPs per on-policy rollout. Off-policy expert traces, when used, incur only the training cost (6​N​D6ND) since no generation is required.
For all methods we use N=7×109N{=}7\times 10^{9} (Qwen2.5-Math-7B) and an average off-policy demonstration data response length of Ddata=4,200D\_{\text{data}}{=}4{,}200 tokens.

* •

  LUFFY trains for 500 steps at batch size 128 with 7 on-policy rollouts (Drollout=2,200D\_{\text{rollout}}{=}2{,}200 tokens average response length) and 1 off-policy expert trace per sample question. The on-policy rollouts cost 8×7×N×Drollout=8.62×10148\times 7\times N\times D\_{\text{rollout}}=8.62\times 10^{14} (generation + training), and the expert trace costs 6×N×Ddata=1.76×10146\times N\times D\_{\text{data}}=1.76\times 10^{14} (training only), totalling 1.04×10151.04\times 10^{15} FLOPs per sample and 6.65×𝟏𝟎𝟏𝟗\mathbf{6.65\times 10^{19}} FLOPs overall.
* •

  ReLIFT trains for 500 steps at batch size 128 with 8 on-policy rollouts (Drollout=3,000D\_{\text{rollout}}{=}3{,}000 tokens average response length), costing 8×8×N×Drollout=1.34×10158\times 8\times N\times D\_{\text{rollout}}=1.34\times 10^{15} FLOPs per sample. It also interleaves 138 SFT updates at batch size 64 (1.56×10181.56\times 10^{18} FLOPs), bringing its total to 8.76×𝟏𝟎𝟏𝟗\mathbf{8.76\times 10^{19}} FLOPs.
* •

  SFT →\boldsymbol{\to} RL finetunes for 3 epochs over 4646k samples at sequence length DdataD\_{\text{data}} (2.43×10192.43\times 10^{19} FLOPs), then runs 50 RL steps at batch size 128 with 8 on-policy rollouts (average response length Drollout=4,200D\_{\text{rollout}}{=}4{,}200 tokens, contributing 8×8×N×Drollout=1.88×10158\times 8\times N\times D\_{\text{rollout}}=1.88\times 10^{15} FLOPs per sample, 1.20×10191.20\times 10^{19} FLOPs total), for a combined cost of 3.63×𝟏𝟎𝟏𝟗\mathbf{3.63\times 10^{19}} FLOPs.

All estimates count only response tokens; including prompt tokens would scale all values proportionally without affecting relative rankings.

## Appendix C DeepSpeed Bug Fix Patch

Patch

```
@@ -1394,8 +1394,7 @@ class DeepSpeedZeroOptimizer(ZeROOptimizer):

         if self.micro_step_id > 0:
             accumulate_gradients()
-        else:
-            copy_gradients_to_cpu()
+        copy_gradients_to_cpu()

     def set_norm_for_param_grad(self, param):
         param_id = self.get_param_id(param)
```

Figure 4: Patch for deepspeed/runtime/zero/stage\_1\_and\_2.py

This fix has been submitted and merged upstream: [deepspeedai/DeepSpeed#7967](https://github.com/deepspeedai/DeepSpeed/pull/7967).

## Appendix D OpenRLHF Loss Aggregation Bug Fix Pseudocode

Algorithm 1  Original OpenRLHF SFT loss computation (buggy).

1:per-token log-probs ℓi\ell\_{i}, loss mask mim\_{i} on local DP rank kk

2:ℒk←MaskedMean​(−ℓi,mi)\mathcal{L}\_{k}\leftarrow\textsc{MaskedMean}(-\ell\_{i},\;m\_{i}) ⊳\triangleright Local mean over tokens on rank kk

3:backward(ℒk)(\mathcal{L}\_{k}) ⊳\triangleright Distributed optimizer averages gradients across DP ranks

Algorithm 2  Fixed OpenRLHF SFT loss computation.

1:per-token log-probs ℓi\ell\_{i}, loss mask mim\_{i} on local DP rank kk, DP size DD

2:Sk←MaskedSum​(−ℓi,mi)S\_{k}\leftarrow\textsc{MaskedSum}(-\ell\_{i},\;m\_{i}) ⊳\triangleright Local loss sum

3:nk←∑imin\_{k}\leftarrow\sum\_{i}m\_{i} ⊳\triangleright Local token count

4:N←AllReduce​(nk,Sum)N\leftarrow\textsc{AllReduce}(n\_{k},\;\textsc{Sum}) ⊳\triangleright Global token count across all DP ranks

5:ℒk←SkN×D\mathcal{L}\_{k}\leftarrow\dfrac{S\_{k}}{N}\times D ⊳\triangleright Scale by DD to account for gradient averaging across DP ranks

6:backward(ℒk)(\mathcal{L}\_{k})

This fix has been submitted upstream: [OpenRLHF/OpenRLHF#1216](https://github.com/OpenRLHF/OpenRLHF/pull/1216).

## Appendix E Chat Template

Chat template

System: Your task is to follow a systematic, thorough reasoning process before providing the final solution. This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. Structure your response into two sections: Thought and Solution. In the Thought section, present your reasoning using the format: "<think>\n {thoughts} </think>\n". Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. After "</think>\n," in the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section. If applicable, include the answer in \boxed{} for closed-form results like multiple choices or mathematical solutions.
User: This is the problem:
{Question}
Assistant: <think>

Figure 5: Chat template used for both Qwen2.5-Math-7B and Llama-3.1-8B across all experiments.
