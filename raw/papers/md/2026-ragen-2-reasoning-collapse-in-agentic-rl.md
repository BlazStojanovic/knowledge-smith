---
arxiv: '2604.06268'
authors:
- Zihan Wang
- Chi Gui
- Xing Jin
- Qineng Wang
- Licheng Liu
- Kangrui Wang
- Shiqi Chen
- Linjie Li
- Zhengyuan Yang
- Pingyue Zhang
- Yiping Lu
- Jiajun Wu
- Li Fei-Fei
- Lijuan Wang
- Yejin Choi
- Manling Li
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'RAGEN-2: Reasoning Collapse in Agentic RL'
url: https://arxiv.org/abs/2604.06268
year: 2026
---

# RAGEN-2: Reasoning Collapse in Agentic RL

Zihan Wang†∗,1,
Chi Gui†,12,
Xing Jin†,3,
Qineng Wang†,1,
Licheng Liu†,4,Kangrui Wang1,
Shiqi Chen5,
Linjie Li6,
Zhengyuan Yang7,
Pingyue Zhang1,
Yiping Lu1,
Jiajun Wu8,
Li Fei-Fei8,
Lijuan Wang7,
Yejin Choi8,
Manling Li1
  
†Core contributors. ∗Project lead.
  
1Northwestern University 2UIUC 3Independent 4Imperial College London
  
5Oxford University 6University of Washington 7Microsoft 8Stanford University
  
<https://ragen-ai.github.io/v2/>

###### Abstract

RL training of multi-turn LLM agents is inherently unstable, and reasoning quality directly determines task performance. Entropy is widely used to track reasoning stability. However, entropy only measures diversity within the same input, and cannot tell whether reasoning actually responds to different inputs. In RAGEN-2, we find that even with stable entropy, models can rely on fixed templates that look diverse but are input-agnostic. We call this template collapse, a failure mode invisible to entropy and all existing metrics.
To diagnose this failure, we decompose reasoning quality into within-input diversity (Entropy) and cross-input distinguishability (Mutual Information, MI), and introduce a family of mutual information proxies for online diagnosis. Across diverse tasks, mutual information correlates with final performance much more strongly than entropy, making it a more reliable proxy for reasoning quality. We further explain template collapse with a *signal-to-noise ratio* (SNR) mechanism. Low reward variance weakens task gradients, letting regularization terms dominate and erase cross-input reasoning differences. To address this, we propose SNR-Aware Filtering to select high-signal prompts per iteration using reward variance as a lightweight proxy. Across planning, math reasoning, web navigation, and code execution, the method consistently improves both input dependence and task performance.

## 1 Introduction

Training multi-turn LLM agents with reinforcement learning (RL) is inherently challenging qi2025defeatingtraininginferencemismatchfp16; zhang2026precisiontraininginferencemismatchoptimization; yu2025dapo. Researchers therefore monitor reward for outcome stability and entropy for
reasoning process stability schulman2017proximalpolicyoptimizationalgorithms; ouyang2022traininglanguagemodelsfollow; xu2025epoentropyregularizedpolicyoptimization,
treating both as stability indicators of RL training.

However, entropy can be an ambiguous signal to understand reasoning quality. When entropy decreases, it may simply reflect the model becoming more specialized and confident on the task, which is a natural outcome of RL optimization yu2025dapo; xu2025epoentropyregularizedpolicyoptimization. When entropy remains high, reasoning can still drift toward fixed templates that appear diverse within any single input but are effectively the same across inputs (Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ RAGEN-2: Reasoning Collapse in Agentic RL")). We call this template collapse, a failure mode invisible to both metrics.
This risk is especially acute in multi-turn settings: sparse rewards cannot distinguish input-driven reasoning from templated reasoning that merely happens to succeed wang2025practitionersguidemultiturnagentic; wang2025ragenunderstandingselfevolutionllm, and reasoning chains are hard to get directly supervised shao2024deepseekmathpushinglimitsmathematical; cui2025processreinforcementimplicitrewards. As a result, template collapse can persist unnoticed during training, making agents unreliable and silently hurting their reasoning abilities.

To understand and mitigate template collapse, this paper addresses two questions.
(Q1) How to diagnose? (§[2](#S2 "2 Template Collapse in Multi-turn Agent RL ‣ RAGEN-2: Reasoning Collapse in Agentic RL")) Entropy-based metrics wei2025gtrguidedthoughtreinforcement; yao2025diversityawarepolicyoptimizationlarge; yun2025priceformatdiversitycollapse track
within-input variability but miss input dependence across inputs, so they fail to detect template collapse. We propose a mutual information (MI)
proxy coverthomas2006elements that scores each reasoning chain against all batch inputs to measure input dependence, without external models.
(Q2) Why does it happen? (§[3](#S3 "3 The Mechanism of Template Collapse: A Signal-to-Noise Ratio (SNR) View ‣ RAGEN-2: Reasoning Collapse in Agentic RL")) We explain through a signal-to-noise ratio (SNR) lens. Task gradients draw signal from reward differences across within-input trajectories. Sampling noise and input-agnostic regularization (KL divergence and
entropy regularization schulman2017proximalpolicyoptimizationalgorithms; xu2025epoentropyregularizedpolicyoptimization) dilute this signal. Low SNR lets noise dominate, erasing cross-input reasoning differences.

To address template collapse, based on the SNR view, we introduce SNR-Aware Filtering, which uses reward variance as a lightweight SNR proxy to select high-signal prompts each iteration, without additional supervision. Throughout training, the MI proxy monitors input dependence; across experiments, MI correlates with task performance significantly more strongly than entropy, validating it as a diagnostic for template collapse.

Together, they constitute a diagnostic framework for a systematic failure mode in multi-turn agent RL, validated across planning schrader2018gymsokoban, mathematical
reasoning yu2023metamath; katz2025countdown, web navigation, code execution, and tool use, under multiple RL algorithms, model scales, and modalities. SNR-Aware Filtering consistently improves input dependence and task performance, providing direct experimental support for the SNR mechanism.

Our contributions are summarized as follows:

1. 1.

   Identifying template collapse. We find that template collapse occurs when reasoning appears diverse within inputs but becomes input-agnostic across inputs. We propose a mutual information proxy to detect it without external models.
2. 2.

   Explaining template collapse via SNR. We show that low reward variance weakens task gradients while input-agnostic regularization remains constant, erasing input dependence. We provide gradient decomposition evidence across reward-variance buckets.
3. 3.

   SNR-Aware Filtering. We propose filtering prompts by reward variance before each update. We demonstrate that this improves input dependence and performance across tasks, algorithms, scales, and modalities.

!(/html/2604.06268/assets/table_figures/teaser.png)

Figure 1: Left: input-driven reasoning adapts to the current state; templated reasoning produces nearly identical responses across different inputs. Right: four reasoning regimes characterized along two axes: conditional entropy H​(Z∣X)H(Z\mid X) (within-input diversity) and mutual information I​(X;Z)I(X;Z) (input dependence). Details in Section [2](#S2 "2 Template Collapse in Multi-turn Agent RL ‣ RAGEN-2: Reasoning Collapse in Agentic RL").

## 2 Template Collapse in Multi-turn Agent RL

### 2.1 Setup and Preliminaries

We study closed-loop multi-turn agent reinforcement learning wang2025ragenunderstandingselfevolutionllm, where a policy πθ\pi\_{\theta} is trained by repeatedly rolling out trajectories under the current policy and environment and updating on the collected experience. At each time step tt, the agent observes oto\_{t}, generates a response consisting of reasoning tokens ztz\_{t} and an executable action ata\_{t}, and receives reward rtr\_{t}, forming a trajectory τ={(ot,zt,at,rt)}t=1T\tau=\{(o\_{t},z\_{t},a\_{t},r\_{t})\}\_{t=1}^{T}.

We use XX to denote the full context available to the model immediately before generating reasoning at turn tt: this comprises the system prompt, all prior observations o1:to\_{1:t}, actions a1:t−1a\_{1:t-1}, and reasoning tokens z1:t−1z\_{1:t-1}. We use ZZ to denote the reasoning token sequence the model generates for that turn, excluding action tokens and boundary markers (e.g., </think>).

The standard PPO/GRPO objective contains regularization terms (KL divergence, entropy bonus) that act uniformly across all inputs regardless of their content:

L(θ) = E\_x,τ[A(τ, x)] - λ\_KL D\_KL(π\_θ∥ π\_ref) + λ\_H H(π\_θ),

where A​(τ,x)A(\tau,x) is the advantage.

### 2.2 Rethinking Reasoning Collapse from an Information-Theoretic Lens

Why entropy is insufficient to measure reasoning quality? Researchers proxy process stability with entropy and outcome stability with reward, treating both as
evidence of healthy training. Stable entropy, however, does not guarantee stable reasoning. Reasoning diversity (marginal entropy) H​(Z)H(Z) decomposes via the standard
identity coverthomas2006elements:

H(Z) = I(X;Z) + H(Z∣X),

where I​(X;Z)I(X;Z) is input dependence (mutual information between input XX and reasoning ZZ), and H​(Z∣X)H(Z\mid X) is within-input diversity (conditional entropy of
reasoning given input). Entropy metrics proxy H​(Z∣X)H(Z\mid X), but neither captures a decline in I​(X;Z)I(X;Z): the policy can sustain high H​(Z∣X)H(Z\mid X) while I​(X;Z)I(X;Z) drops
to zero, producing diverse but input-agnostic boilerplate. We call this template collapse.

Reasoning regimes with a mutual information view. Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ RAGEN-2: Reasoning Collapse in Agentic RL") illustrates four reasoning states along these two axes:
(i) *Diverse Reasoning* (high H​(Z∣X)H(Z\mid X), high I​(X;Z)I(X;Z)): the desired regime where reasoning is both varied within each input and systematically grounded across different inputs;
(ii) *Template Collapse* (high H​(Z∣X)H(Z\mid X), low I​(X;Z)I(X;Z)): superficially diverse but input-agnostic—the systematic blind spot of existing stability metrics;
(iii) *Compressed Reasoning* (low H​(Z∣X)H(Z\mid X), high I​(X;Z)I(X;Z)): input-faithful but overly deterministic; and
(iv) *Low-Entropy Collapse* (low H​(Z∣X)H(Z\mid X), low I​(X;Z)I(X;Z)): fully degenerate with deterministic and input-agnostic outputs.
Among these, Template Collapse is uniquely problematic because entropy-based metrics can remain high while input dependence collapses. Empirically, I​(X;Z)I(X;Z) correlates significantly more strongly with task performance than entropy does (Figure [8](#S5.F8 "Figure 8 ‣ 5.1 MI Diagnoses Collapse Better Than Entropy Across All Interventions ‣ 5 Analysis ‣ RAGEN-2: Reasoning Collapse in Agentic RL")).

### 2.3 Mutual Information Proxy Family

Table 1: MI proxy family. All variants are derived from in-batch cross-scoring of reasoning traces against prompts, using matched (per-token log-prob under the true prompt) and marginal (per-token log-prob under the uniform prompt mixture) as base quantities. First-turn variants use only the first agent turn; trajectory variants sample across all turns.

|  |  |  |  |
| --- | --- | --- | --- |
| Type | Proxy | Formula | Notes |
| Discrete | Retrieval-Acc | 1P​G​∑i,k𝟏​[arg⁡maxj⁡𝐋i,k,j=i]\frac{1}{PG}\sum\_{i,k}\mathbf{1}[\arg\max\_{j}\mathbf{L}\_{i,k,j}=i] | Chance level 1/P1/P under template collapse |
| bi lilin k | Recall@kk | 1P​G​∑i,k𝟏​[i∈top​-​kj​(𝐋i,k,j)]\frac{1}{PG}\sum\_{i,k}\mathbf{1}[i\in\mathrm{top}\text{-}k\_{j}(\mathbf{L}\_{i,k,j})] | k∈{2,4,8}k\in\{2,4,8\} |
| Continuous (raw) | MI-Est | 1P​G​∑i,k(matchedi,k−marginali,k)\frac{1}{PG}\sum\_{i,k}(\mathrm{matched}\_{i,k}-\mathrm{marginal}\_{i,k}) | Per-token; approaches 0 under collapse |
| MI-Seq-Est | 1P​G​∑i,k(𝐋i,k,i−log⁡1P​∑je𝐋i,k,j)\frac{1}{PG}\sum\_{i,k}\bigl(\mathbf{L}\_{i,k,i}-\log\tfrac{1}{P}\sum\_{j}e^{\mathbf{L}\_{i,k,j}}\bigr) | Per-sequence; no length normalization |
| Continuous (z-score) | MI-ZScore | 1P​G​∑i,kmatchedi,k−marginali,kσbatch+ϵ\frac{1}{PG}\sum\_{i,k}\frac{\mathrm{matched}\_{i,k}-\mathrm{marginal}\_{i,k}}{\sigma\_{\mathrm{batch}}+\epsilon} | Normalized by current-batch marginal std |
| MI-ZScore-EMA | 1P​G​∑i,kmatchedi,k−marginali,kσEMA+ϵ\frac{1}{PG}\sum\_{i,k}\frac{\mathrm{matched}\_{i,k}-\mathrm{marginal}\_{i,k}}{\sigma\_{\mathrm{EMA}}+\epsilon} | σEMA(t)=α​σEMA(t−1)+(1−α)​σbatch(t)\sigma\_{\mathrm{EMA}}^{(t)}=\alpha\,\sigma\_{\mathrm{EMA}}^{(t-1)}+(1{-}\alpha)\,\sigma\_{\mathrm{batch}}^{(t)} |

How do we estimate mutual information?
True mutual information I​(X;Z)I(X;Z) has no closed form for high-dimensional token sequences, so we propose an empirical proxy I^​(X;Z)\widehat{I}(X;Z) based on retrieval. The intuition: mutual information I​(X;Z)I(X;Z) measures how much knowing the reasoning ZZ tells us about which input XX produced it. When I​(X;Z)I(X;Z) is high, different inputs yield distinguishable reasoning patterns—the model adapts its reasoning to the specific problem. When I​(X;Z)I(X;Z) is low, reasoning becomes input-agnostic: observing ZZ gives little clue about which XX it came from. This is the signature of template collapse. If reasoning truly collapses into templates, it should be easy to detect: a reasoning trace ZZ generated from input XiX\_{i} will be equally likely under any other input XjX\_{j}.

Method: In-Batch Cross-Scoring.
Given PP prompts and GG reasoning samples per prompt from training rollouts, we compute teacher-forced log-likelihoods for every (Zi,k,Xj)(Z\_{i,k},X\_{j}) pair, forming the scoring matrix 𝐋i,k,j=log⁡pθ​(Zi,k∣Xj)\mathbf{L}\_{i,k,j}=\log p\_{\theta}(Z\_{i,k}\mid X\_{j}). We extract two length-normalized quantities:

|  |  |  |  |
| --- | --- | --- | --- |
|  | matchedi,k=𝐋i,k,i|Zi,k|,marginali,k=1|Zi,k|​log⁡1P​∑jexp⁡(𝐋i,k,j),\mathrm{matched}\_{i,k}=\frac{\mathbf{L}\_{i,k,i}}{|Z\_{i,k}|},\qquad\mathrm{marginal}\_{i,k}=\frac{1}{|Z\_{i,k}|}\log\frac{1}{P}\sum\_{j}\exp(\mathbf{L}\_{i,k,j}), |  | (1) |

where matchedi,k\mathrm{matched}\_{i,k} is the per-token log-likelihood of reasoning Zi,kZ\_{i,k} under its true source input XiX\_{i}, and marginali,k\mathrm{marginal}\_{i,k} approximates the marginal log-likelihood log⁡pθ​(Zi,k)\log p\_{\theta}(Z\_{i,k}) via a uniform mixture over all prompts in the batch.

Two Primary Proxies.
We use two complementary proxies derived from Eq. [1](#S2.E1 "Equation 1 ‣ 2.3 Mutual Information Proxy Family ‣ 2 Template Collapse in Multi-turn Agent RL ‣ RAGEN-2: Reasoning Collapse in Agentic RL"):

(1) Retrieval-Acc (discrete, interpretable): We define

|  |  |  |
| --- | --- | --- |
|  | Acc=1P​G​∑i=1P∑k=1G𝕀​[i=arg⁡maxj⁡𝐋i,k,j].\mathrm{Acc}=\frac{1}{PG}\sum\_{i=1}^{P}\sum\_{k=1}^{G}\mathbb{I}\Big[i=\arg\max\_{j}\,\mathbf{L}\_{i,k,j}\Big]. |  |

Under collapse, Acc\mathrm{Acc} approaches chance level 1/P1/P (1.56% at P=64P{=}64), providing an absolute reference.

(2) MI-ZScore-EMA (continuous, robust): We estimate input dependence as

|  |  |  |
| --- | --- | --- |
|  | I^​(X;Z)=1P​G​∑i=1P∑k=1G(matchedi,k−marginali,k),\widehat{I}(X;Z)=\frac{1}{PG}\sum\_{i=1}^{P}\sum\_{k=1}^{G}\Big(\mathrm{matched}\_{i,k}-\mathrm{marginal}\_{i,k}\Big), |  |

which increases when reasoning is much more compatible with its source input than with the batch mixture. In template-collapse regimes, matchedi,k≈marginali,k\mathrm{matched}\_{i,k}\approx\mathrm{marginal}\_{i,k} for many samples and thus I^​(X;Z)\widehat{I}(X;Z) approaches 0. We apply z-score normalization and exponential moving average (EMA) to stabilize training monitoring, yielding MI-ZScore-EMA.

Proxy Variants and Validation.
Table [1](#S2.T1 "Table 1 ‣ 2.3 Mutual Information Proxy Family ‣ 2 Template Collapse in Multi-turn Agent RL ‣ RAGEN-2: Reasoning Collapse in Agentic RL") lists additional proxy variants, varying along three dimensions: (1) turn scope (first-turn only vs. trajectory-uniform sampling); (2) aggregation (discrete retrieval vs. continuous MI estimate); (3) length normalization (per-token vs. per-sequence). For comparison, conditional entropy H​(Z∣X)=−1P​G​∑i,kmatchedi,kH(Z\mid X)=-\frac{1}{PG}\sum\_{i,k}\mathrm{matched}\_{i,k} and marginal entropy H​(Z)=−1P​G​∑i,kmarginali,kH(Z)=-\frac{1}{PG}\sum\_{i,k}\mathrm{marginal}\_{i,k} are logged in parallel, satisfying H​(Z)=I^​(X;Z)+H​(Z∣X)H(Z)=\widehat{I}(X;Z)+H(Z\mid X). We set ϵ=10−3\epsilon=10^{-3} and α=0.9\alpha=0.9 for z-score normalization and EMA, respectively.

Empirically, Retrieval-Acc and MI-ZScore-EMA achieve positive Spearman correlation with final task performance (+0.39+0.39 for Trajectory MI-ZScore), substantially above entropy metrics, which show negative correlations (−0.11-0.11 to −0.14-0.14), confirming entropy is misleading in direction (Figure [8](#S5.F8 "Figure 8 ‣ 5.1 MI Diagnoses Collapse Better Than Entropy Across All Interventions ‣ 5 Analysis ‣ RAGEN-2: Reasoning Collapse in Agentic RL")). All proxies reuse (Xi,Zi,k)(X\_{i},Z\_{i,k}) pairs from the training rollout and require no additional model or inference pass; implementation details are in Appendix C.

!(/html/2604.06268/assets/x1.png)

Figure 2: 
Schematic Signal-to-Noise Ratio (SNR) view of RL updates.
Left: total gradient decomposes into task gradient (sharpens with higher within-input reward variance) and regularization gradient. Right: high reward variance yields strong task gradient and better convergence (high SNR); low reward variance makes regularization gradient dominate, producing erratic updates and input-agnostic reasoning (low SNR).

## 3 The Mechanism of Template Collapse: A Signal-to-Noise Ratio (SNR) View

We have defined template collapse (low I​(X;Z)I(X;Z), high H​(Z∣X)H(Z\mid X)) and introduced an MI proxy to diagnose it. This section explains why RL training produces this failure mode and how to mitigate it. Our core finding: when policy gradient updates are dominated by input-agnostic noise rather than task-discriminative signal—low signal-to-noise ratio (SNR)—reasoning drifts toward templates that appear diverse within each input but ignore cross-input differences.

### 3.1 Observing Signal-Noise Imbalance in RL Gradients

We begin with an empirical observation that motivates the mechanistic analysis. Sorting training prompts by their within-input reward variance Var^​(R∣X)\widehat{\mathrm{Var}}(R\mid X) and grouping them into equal-sized buckets, we measure the gradient norms contributed by task objectives versus regularization terms (Figure [3](#S3.F3 "Figure 3 ‣ 3.1 Observing Signal-Noise Imbalance in RL Gradients ‣ 3 The Mechanism of Template Collapse: A Signal-to-Noise Ratio (SNR) View ‣ RAGEN-2: Reasoning Collapse in Agentic RL")). Three patterns emerge consistently across algorithms:

1. 1.

   Task gradient scales with reward variance: ‖gtask‖\|g\_{\text{task}}\| increases monotonically with bucket RV. High-variance prompts yield strong task-discriminative gradients; low-variance prompts produce weak gradients even when non-zero.
2. 2.

   Regularization gradient is flat: ‖greg‖\|g\_{\text{reg}}\| (from KL and entropy terms) remains constant across all buckets, applying uniform contraction to every reasoning chain regardless of its source prompt or reward signal.
3. 3.

   Low-RV prompts produce gradient updates dominated by regularization: In the lowest-variance buckets, task gradients nearly vanish while regularization gradients persist, meaning updates are driven almost entirely by input-agnostic noise.

!(/html/2604.06268/assets/x2.png)

Figure 3: 
Prompts sorted into six equal-sized reward-variance buckets Q1–Q6. We find: (a) Task gradient norm increases monotonically with bucket RV; (b) When RV near 0, substantial task gradients persist despite carrying almost no useful signal; (c) Regularizer gradient norm (KL + entropy) is flat across buckets. This directly supports the SNR mechanism under both algorithms.

This gradient imbalance suggests that low reward variance weakens the task-discriminative component of updates, allowing input-agnostic regularization to dominate. When many prompts fall into this regime, the model learns to produce reasoning that satisfies regularization constraints (diverse, fluent) but ignores input-specific requirements—exactly the signature of template collapse.

### 3.2 Formalizing the SNR Mechanism via Gradient Decomposition

The empirical pattern above can be formalized through a signal-to-noise decomposition of policy gradients. Low within-input reward variance collapses advantages toward zero, weakening the task gradient.
Simultaneously, input-agnostic regularization terms apply uniform contraction to every reasoning
chain regardless of its source prompt. When the task gradient is weak, regularization dominates
every update and pushes reasoning toward input-agnostic patterns, lowering I​(X;Z)I(X;Z). This is the
gradient-level mechanism behind template collapse ([Figure˜2](#S2.F2 "In 2.3 Mutual Information Proxy Family ‣ 2 Template Collapse in Multi-turn Agent RL ‣ RAGEN-2: Reasoning Collapse in Agentic RL"); regularizer-dominance analysis in Appendix [K](#A11 "Appendix K Reward-Agnostic Regularizers and Update Dominance ‣ RAGEN-2: Reasoning Collapse in Agentic RL")).

For input xx with GG sampled trajectories, the advantage estimate is Ag=Rg−R¯​(x)A\_{g}=R\_{g}-\bar{R}(x) and the task gradient is

|  |  |  |
| --- | --- | --- |
|  | gtask​(x)=1G​∑gAg​∇θlog⁡πθ​(τg∣x).g\_{\mathrm{task}}(x)=\frac{1}{G}\sum\_{g}A\_{g}\,\nabla\_{\theta}\log\pi\_{\theta}(\tau\_{g}\mid x). |  |

The Cauchy-Schwarz inequality gives (Appendix [H](#A8 "Appendix H RV Controls Task-Signal Magnitude and SNR ‣ RAGEN-2: Reasoning Collapse in Agentic RL")):

|  |  |  |
| --- | --- | --- |
|  | |gtask​(x)|≤Var^​(R∣X=x)⋅C.|g\_{\mathrm{task}}(x)|\leq\sqrt{\widehat{\mathrm{Var}}(R\mid X=x)}\cdot C. |  |

Low reward variance therefore weakens gtaskg\_{\mathrm{task}} while leaving gregg\_{\mathrm{reg}} unchanged, driving I​(X;Z)→0I(X;Z)\to 0. Critically, H​(Z∣X)H(Z\mid X) need not decline: entropy regularization can sustain within-input diversity while input dependence collapses.

We formalize this through a three-noise decomposition of the total gradient:
gtotal=gsignal+gtask-noise+greg.g\_{\text{total}}=g\_{\text{signal}}+g\_{\text{task-noise}}+g\_{\text{reg}}.

Table 2: Three-noise decomposition of the policy update gradient.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Component | Source | Level | Ctrl. | Mitigation |
| gsignalg\_{\text{signal}} | Meaningful reward differences across same-prompt trajectories | Prompt | No | SNR-Aware Filtering |
| gtask-noiseg\_{\text{task-noise}} | Sampling and environment stochasticity | Prompt | No | Filter high-noise prompts |
| gregg\_{\text{reg}} | Uniform contraction per chain, independent of input (KL, entropy) | Chain | Yes | Tune λKL\lambda\_{\text{KL}}, λent\lambda\_{\text{ent}} |

Signal and task noise both vary across prompts, but only the former carries
task-discriminative information. Regularization noise acts uniformly at the chain level:
every reasoning chain receives the same KL/entropy contraction regardless of its source prompt,
making it inherently input-agnostic and the direct suppressive force on cross-input differences (Table [2](#S3.T2 "Table 2 ‣ 3.2 Formalizing the SNR Mechanism via Gradient Decomposition ‣ 3 The Mechanism of Template Collapse: A Signal-to-Noise Ratio (SNR) View ‣ RAGEN-2: Reasoning Collapse in Agentic RL")).

In practice, gtask=gsignal+gtask-noiseg\_{\text{task}}=g\_{\text{signal}}+g\_{\text{task-noise}} merges the two
prompt-level components. The SNR is
SNR​(x)=‖gsignal​(x)‖/(‖gtask-noise​(x)‖+‖greg‖).\mathrm{SNR}(x)=\|g\_{\text{signal}}(x)\|/(\|g\_{\text{task-noise}}(x)\|+\|g\_{\text{reg}}\|).
Low SNR shifts updates toward input-agnostic directions, lowering I​(X;Z)I(X;Z) even when
H​(Z∣X)H(Z\mid X) remains high (Appendix [K](#A11 "Appendix K Reward-Agnostic Regularizers and Update Dominance ‣ RAGEN-2: Reasoning Collapse in Agentic RL")).

Low reward variance but non-vanishing gradient norm.
When Var^​(R∣X)≈0\widehat{\mathrm{Var}}(R\mid X)\approx 0, advantages collapse to zero and
gtask≈0g\_{\text{task}}\approx 0, yet ‖gtotal‖≈‖greg‖\|g\_{\text{total}}\|\approx\|g\_{\text{reg}}\| because
gregg\_{\text{reg}} is independent of reward variance. Low-RV prompts therefore produce updates
driven entirely by input-agnostic regularization noise, systematically lowering I​(X;Z)I(X;Z).
SNR-Aware Filtering removes these task-useless but regularization-active updates by filtering
out low-RV prompts, the core mechanism by which the method restores input-conditioned reasoning.

!(/html/2604.06268/assets/table_figures/RV-filter.png)

Figure 4: SNR-Aware Filtering workflow. At each training iteration: (1) rollout generation collects trajectories; (2) within-prompt reward variance is computed as SNR proxy; (3) prompts are ranked by RV and top-p fraction retained, policy update performed only on high-signal subset. This filtering loop can prevent updating on noisy rollouts and requires no additional models/rollouts beyond standard RL.

### 3.3 SNR-Aware Filtering: Prioritizing High-Signal Updates

The gradient analysis above identifies the mechanism behind template collapse: low reward variance weakens task signal, allowing regularization noise to dominate and push reasoning toward input-agnostic patterns. This suggests a direct mitigation strategy: prioritize prompts with higher within-input reward variance, where advantage estimates carry stronger task-discriminative information and regularization is less likely to dominate the update.

We propose SNR-Aware Filtering: at each training iteration, estimate Var^​(R∣X)\widehat{\mathrm{Var}}(R\mid X) for each prompt and retain only the top fraction by variance before computing parameter updates (workflow in Figure [4](#S3.F4 "Figure 4 ‣ 3.2 Formalizing the SNR Mechanism via Gradient Decomposition ‣ 3 The Mechanism of Template Collapse: A Signal-to-Noise Ratio (SNR) View ‣ RAGEN-2: Reasoning Collapse in Agentic RL")). This concentrates gradient budget on high-SNR prompts and filters out low-variance updates that would be dominated by input-agnostic regularization.

Reward variance as SNR proxy.
At each iteration, we estimate Var​(R∣X)\mathrm{Var}(R\mid X) at the prompt level by sampling GG trajectories for the same prompt XX and computing the sample variance of episode returns:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Var^​(R∣X)\displaystyle\widehat{\mathrm{Var}}(R\mid X) | =1G−1​∑g=1G(Rg​(X)−R¯​(X))2,\displaystyle=\frac{1}{G-1}\sum\_{g=1}^{G}\big(R\_{g}(X)-\overline{R}(X)\big)^{2}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | R¯​(X)\displaystyle\overline{R}(X) | =1G​∑g=1GRg​(X).\displaystyle=\frac{1}{G}\sum\_{g=1}^{G}R\_{g}(X). |  |

Higher Var^​(R∣X)\widehat{\mathrm{Var}}(R\mid X) indicates trajectories can be meaningfully distinguished by reward, strengthening advantage estimates and increasing the likelihood that gradients align with task-relevant directions (Appendix [H](#A8 "Appendix H RV Controls Task-Signal Magnitude and SNR ‣ RAGEN-2: Reasoning Collapse in Agentic RL")).

Top-pp filtering by reward variance.
We keep the top fraction of prompts by variance score with keep rate ρ∈(0,1]\rho\in(0,1], analogous to nucleus sampling holtzman2020curiouscaseneuraltext but ranking by per-prompt reward variance rather than token probability. Given PP prompts indexed by i=1,…,Pi=1,\dots,P with variance scores Var^​(R∣X=xi)\widehat{\mathrm{Var}}(R\mid X=x\_{i}), we rank by descending variance:
^Var(R∣X=x\_σ(1)) ≥^Var(R∣X=x\_σ(2)) ≥⋯≥^Var(R∣X=x\_σ(P)),
where σ:{1,…,P}→{1,…,P}\sigma:\{1,\dots,P\}\to\{1,\dots,P\} is a permutation. Define the selection threshold as
τ= ρ∑\_i=1^P ^Var(R∣X=x\_i),
and accumulate variance mass from the top until reaching τ\tau:
S = {σ(1),…,σ(k^\*)},  where  k^\* = min{k : ∑\_j=1^k ^Var(R∣X=x\_σ(j)) ≥τ}.
The filtered objective becomes ℒρ​(θ)=1k∗​∑i∈S∑j∈ℬiLθ​(ξj)\mathcal{L}\_{\rho}(\theta)=\frac{1}{k^{\*}}\sum\_{i\in S}\sum\_{j\in\mathcal{B}\_{i}}L\_{\theta}(\xi\_{j}), where ℬi\mathcal{B}\_{i} is the set of samples in group ii. This adaptive selection naturally concentrates updates on high-signal prompts while automatically adjusting the kept count based on the variance distribution. Other filtering strategies (top-kk, min-pp) and implementation details are in Appendix [G](#A7 "Appendix G Formal Definition of the Filtering Operator ‣ RAGEN-2: Reasoning Collapse in Agentic RL").

Table 3: Summary of the features of the environments used.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Task | Stochastic | Multi-turn | State | Reward |
| Sokoban | ✗ | ✓ | Grid | Dense |
| FrozenLake | ✓ | ✓ | Grid | Binary |
| MetaMathQA | ✗ | ✓ | Text | Dense |
| Countdown | ✗ | ✗ | Text | Binary |
| SearchQA | ✗ | ✓ | Text | Dense |
| WebShop | ✗ | ✓ | Text | Dense |
| DeepCoder | ✗ | ✗ | Text | Dense |

## 4 Experiments

We first establish that template collapse occurs reliably across training configurations ([Section˜4.2](#S4.SS2 "4.2 Template Collapse as a Consistent Failure Mode ‣ 4 Experiments ‣ RAGEN-2: Reasoning Collapse in Agentic RL")), then evaluate SNR-Aware Filtering as an intervention across tasks, algorithms, model scales, and modalities ([Section˜4.3](#S4.SS3 "4.3 SNR-Aware Filtering Consistently Improves Performance ‣ 4 Experiments ‣ RAGEN-2: Reasoning Collapse in Agentic RL"); Table [4](#S4.T4 "Table 4 ‣ 4.2 Template Collapse as a Consistent Failure Mode ‣ 4 Experiments ‣ RAGEN-2: Reasoning Collapse in Agentic RL")).

### 4.1 Experimental Testbed

We adopt the RAGEN wang2025ragenunderstandingselfevolutionllm testbed and evaluate LLM agents on four controllable tasks that stress complementary decision-making regimes: irreversible planning (Sokoban), sparse-reward long-horizon navigation under stochastic transitions (FrozenLake), and symbolic math reasoning (MetaMathQA, Countdown). To further evaluate multi-turn reasoning and decision-making capabilities, we also include SearchQA rllm2025, WebShop yao2022webshop, and DeepCoder mattern2025synthetic1; li2023taco; jain2024livecodebench (see Appendix [B.1](#A2.SS1 "B.1 Environments and Tasks ‣ Appendix B Detailed Experimental Settings ‣ RAGEN-2: Reasoning Collapse in Agentic RL") for detailed descriptions).

#### Environments and tasks.

Our testbed spans seven diverse environments with complementary characteristics (Table [3](#S3.T3 "Table 3 ‣ 3.3 SNR-Aware Filtering: Prioritizing High-Signal Updates ‣ 3 The Mechanism of Template Collapse: A Signal-to-Noise Ratio (SNR) View ‣ RAGEN-2: Reasoning Collapse in Agentic RL")). Sokoban is a grid puzzle where the agent pushes boxes onto target cells; actions are effectively irreversible since boxes cannot be pulled (schrader2018gymsokoban). FrozenLake is a navigation task with sparse rewards and stochastic transitions (slippery dynamics) (brockman2016openai\_gym). MetaMathQA is a math QA task derived from MetaMathQA (yu2023metamath) where the agent may revise answers over multiple attempts, and we apply a diminishing reward across retries (halving each retry). Countdown is a single-turn numbers game (katz2025countdown) where the agent constructs an arithmetic expression to hit a target. SearchQA is a multi-turn question-answering task where the agent iteratively searches and synthesizes information to answer complex queries (rllm2025). WebShop is an interactive web navigation task where the agent must search and purchase products matching user specifications (yao2022webshop). DeepCoder is a code synthesis challenge where the agent generates program solutions to meet specified input-output requirements (mattern2025synthetic1; li2023taco; jain2024livecodebench).

#### Training and evaluation setup.

We train Qwen2.5-3B (qwen2024qwen25) with the veRL/HybridFlow stack (sheng2024hybridflow), following RAGEN wang2025ragenunderstandingselfevolutionllm defaults unless otherwise stated. We compare PPO (schulman2017proximalpolicyoptimizationalgorithms), DAPO (yu2025dapo), GRPO (shao2024deepseekmathpushinglimitsmathematical), and Dr. GRPO (liu2025understandingr1zero) for up to 400 rollout–update iterations. Each iteration collects K=P×G=128K=P\times G=128 trajectories per environment, with prompt batch size P=8P=8 and group size G=16G=16 trajectories per prompt. When applying SNR-Aware Filtering with keep rate ρ\rho, we reduce the effective minibatch size accordingly and scale the per-step loss by ρ\rho, so the optimization step size remains comparable.

### 4.2 Template Collapse as a Consistent Failure Mode

Across all training configurations, RL-trained agents reliably develop reasoning that is fluent but input-agnostic: I​(X;Z)I(X;Z) declines while H​(Z∣X)H(Z\mid X) remains high, and this drift is invisible to entropy-based monitoring.

Observing template collapse through MI dynamics.
We track three key metrics during training: task success rate, our MI proxy I^​(X;Z)\widehat{I}(X;Z) (Retrieval-Acc), and conditional entropy H​(Z∣X)H(Z\mid X) (Figure [5](#S4.F5 "Figure 5 ‣ 4.2 Template Collapse as a Consistent Failure Mode ‣ 4 Experiments ‣ RAGEN-2: Reasoning Collapse in Agentic RL")). We present dynamics for all MI proxies in Appendix [D.1](#A4.SS1 "D.1 MI Proxy Metrics During Training ‣ Appendix D Additional Experimental Visualizations ‣ RAGEN-2: Reasoning Collapse in Agentic RL"). The trajectory reveals a critical pattern: mutual information declines significantly before task performance degrades, while conditional entropy remains elevated throughout. This divergence is the hallmark of template collapse. Reasoning appears diverse within each input (high H​(Z∣X)H(Z\mid X)) but becomes increasingly input-agnostic across inputs (low I​(X;Z)I(X;Z)).

The early decline of I^​(X;Z)\widehat{I}(X;Z) demonstrates that our MI proxy serves as an early warning signal, detecting reasoning degradation that entropy-based metrics miss entirely. This finding motivates using MI as a primary diagnostic alongside task performance, rather than relying solely on entropy for process monitoring.

!(/html/2604.06268/assets/x3.png)

Figure 5: Training dynamics under different intervention strategies. (a) Task success rate, (b) MI proxy (retrieval accuracy), and (c) reasoning entropy. Without filtering, MI degrades early while entropy spikes, signaling template collapse. Filtering effectively mitigates the decline in retrieval accuracy, with top-p SNR-Aware filtering best preserving both task performance and reasoning diversity.

!(/html/2604.06268/assets/table_figures/drawer/F17/output/F17_top_p_top_k_no_filter_4envs.png)

Figure 6: Comparison of filtering strategies showing Top-pp consistently outperforming Top-kk and no-filter baselines across four environments.

Behavioral manifestation of template collapse.
Beyond diagnostic metrics, template collapse manifests behaviorally as systematic reasoning compression. We do a more broadly evaluation by reproducing several experiments from existing evaluations spanning spatial agents yin2025mindcube, logic puzzle agents chen2025internalizingworldmodelsselfplay, visual agents wang2025vagen, and math agents liu2026ufo.
Figure [7](#S4.F7 "Figure 7 ‣ 4.2 Template Collapse as a Consistent Failure Mode ‣ 4 Experiments ‣ RAGEN-2: Reasoning Collapse in Agentic RL") shows that reasoning length declines monotonically across all eight environments. As agents converge toward reusable templates, they produce shorter, more formulaic outputs—a behavioral signature of template collapse that complements MI-based diagnostics.

Table 4: SNR-Aware Filtering results (%) across algorithms, model scales, types, and modalities. Each cell reports baseline peak with filter delta in parentheses; Qwen2.5-VL-3B includes text (T) and image (V) inputs. Filtering improves average score across all variants.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Experiment Variants | Sokoban | FrozenLake | MetaMathQA | Countdown | Average |
| Baseline | | | | | |
| PPO (schulman2017proximalpolicyoptimizationalgorithms), |  |  |  |  |  |
| Qwen2.5-3B (qwen2024qwen25) | 12.9 (+16.0) | 67.0 (+10.9) | 92.6 (+0.6) | 97.9 (+0.0) | 67.6 (+6.9) |
| Algorithm | | | | | |
| DAPO (yu2025dapo) | 16.2 (+5.1) | 66.8 (+2.1) | 90.8 (+2.8) | 95.7 (+1.6) | 67.4 (+2.9) |
| GRPO (shao2024deepseekmathpushinglimitsmathematical) | 12.1 (+9.0) | 70.9 (-3.0) | 91.2 (+1.2) | 95.7 (+2.2) | 67.5 (+3.7) |
| Dr. GRPO (liu2025understandingr1zero) | 12.1 (-0.4) | 23.2 (+0.6) | 91.2 (+1.4) | 96.5 (+1.4) | 55.8 (+0.8) |
| Model Scale (PPO) | | | | | |
| Qwen2.5-0.5B (qwen2024qwen25) | 3.3 (+22.9) | 19.5 (+0.0) | 10.0 (-0.2) | 23.0 (-0.7) | 14.0 (+5.5) |
| Qwen2.5-1.5B (qwen2024qwen25) | 17.0 (+6.2) | 36.5 (+1.6) | 80.3 (+7.0) | 56.6 (+1.6) | 47.6 (+4.1) |
| Qwen2.5-7B (qwen2024qwen25) | 42.4 (+4.9) | 85.0 (-0.6) | 84.0 (+11.7) | 97.7 (+0.3) | 77.3 (+4.1) |
| Model Type | | | | | |
| Qwen2.5-3B-Instruct (qwen2024qwen25) | 22.5 (+14.2) | 83.6 (+2.3) | 91.2 (+0.4) | 96.3 (-0.6) | 73.4 (+4.1) |
| Llama3.2-3B (meta2024llama32card) | 24.4 (+18.8) | 84.6 (-0.2) | 86.1 (+3.7) | 99.2 (-1.2) | 73.6 (+5.3) |
| Modality (Input Type) | | | | | |
| Qwen2.5-VL-3B (T) (qwen2025qwen25vl) | 53.0 (+6.0) | 16.0 (+53.5) | - | - | 34.5 (+29.8) |
| Qwen2.5-VL-3B (V) (qwen2025qwen25vl) | 65.0 (+12.0) | 19.5 (+59.5) | - | - | 42.3 (+35.8) |

!(/html/2604.06268/assets/table_figures/drawer/F13/output/F13_reasoning_length_overview.png)

Figure 7: Reasoning length decline across eight environments, showing systematic compression as a behavioral signature of template collapse.

### 4.3 SNR-Aware Filtering Consistently Improves Performance

Comparing filtering strategies across environments.
To evaluate the effectiveness of different filtering approaches, we compare three strategies: Top-p (nucleus-style) filtering, Top-k (fixed-count) filtering, and no filtering baseline. Figure [6](#S4.F6 "Figure 6 ‣ 4.2 Template Collapse as a Consistent Failure Mode ‣ 4 Experiments ‣ RAGEN-2: Reasoning Collapse in Agentic RL") shows task success rates across four representative environments (Sokoban, FrozenLake, MetaMathQA, Countdown).

Top-p filtering consistently achieves higher success rates throughout training compared to both alternatives. The advantage over Top-k filtering is particularly noteworthy: while both methods prioritize high-variance prompts, Top-p’s adaptive selection naturally adjusts to the variance distribution, rejecting entire batches when most prompts carry weak signal. In contrast, Top-k retains a fixed fraction regardless of signal quality, potentially including low-quality updates that dilute the training signal.

The no-filter baseline shows the weakest performance. This confirms that indiscriminate updates on all prompts, including those with near-zero reward variance, systematically degrades learning. These results motivate our choice of Top-p filtering as the primary SNR-Aware mechanism, with more comprehensive cross-environment results reported in Table [4](#S4.T4 "Table 4 ‣ 4.2 Template Collapse as a Consistent Failure Mode ‣ 4 Experiments ‣ RAGEN-2: Reasoning Collapse in Agentic RL").

Table [4](#S4.T4 "Table 4 ‣ 4.2 Template Collapse as a Consistent Failure Mode ‣ 4 Experiments ‣ RAGEN-2: Reasoning Collapse in Agentic RL") summarizes our experimental matrix over four tasks, multiple RL algorithms, model scales/types, and input modalities.
Across this grid, SNR-Aware Filtering yields two consistent effects.
First, it improves peak task success rate in most settings (reported as the +Δ+\Delta next to each peak), demonstrating that prioritizing high-signal updates strengthens learning efficiency.
Second, the gains span multiple experimental axes, including (i) the RL optimizer (PPO / DAPO / GRPO / Dr. GRPO), (ii) the model family and scale (Qwen2.5 from 0.5B to 7B; Llama3.2-3B), and (iii) the input modality (text- and image-conditioned Qwen2.5-VL).
Here, DAPO and Dr. GRPO are recent strong baselines that directly target stable training and mitigate collapse-like failure modes.
In Table [4](#S4.T4 "Table 4 ‣ 4.2 Template Collapse as a Consistent Failure Mode ‣ 4 Experiments ‣ RAGEN-2: Reasoning Collapse in Agentic RL"), DAPO “no-filter” results correspond to the original algorithms without our filtering applied.
DAPO itself also includes a filtering/acceptance step; it can be interpreted as a special case of our framework where the selection is fixed (equivalently, a top-PP filter with P→1.0P\to 1.0), while our SNR-Aware Filtering provides an explicit, tunable SNR knob via the keep rate ρ\rho.
This breadth suggests SNR-Aware Filtering serves as a general-purpose SNR control knob and works alongside standard stabilization terms (e.g., KL\mathrm{KL} and entropy regularization).

Compute overhead of group sampling. SNR-Aware Filtering requires at least G=2G{=}2 trajectories per prompt (group size) to estimate per-prompt RV. Since the total rollout budget is fixed at K=128K{=}128 trajectories, varying the prompt batch size PP and group size GG is a repartitioning of that budget — all configurations incur identical rollout cost. Table [5](#S4.T5 "Table 5 ‣ 4.3 SNR-Aware Filtering Consistently Improves Performance ‣ 4 Experiments ‣ RAGEN-2: Reasoning Collapse in Agentic RL") shows performance and wall-clock step time across configurations on Sokoban (Qwen2.5-3B). RV computation itself adds <0.1%{<}0.1\% of iteration time. With filtering (ρ=0.9\rho{=}0.9), fewer groups enter gradient computation, reducing per-step time by 26–41%. Configurations with group size G≥4G\geq 4 and SNR-Aware Filtering match or outperform the 128×1128{\times}1 baseline, confirming that the gains come at no additional compute cost.

Table 5: Sweep over prompt batch size PP and group size GG (trajectories per prompt) on Sokoban (Qwen2.5-3B).
NF = no filtering; F = SNR-Aware Filtering (ρ=0.9\rho{=}0.9). Total rollout budget is fixed at 128 trajectories across all configurations.

|  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PP (prompts) ×\times GG (traj/prompt) | Task Perf. (%) | | | Step Time (s) | | | VRAM (GB) | | |
| NF | F | Δ\Delta | NF | F | Δ\Delta% | NF | F | Δ\Delta |
| 128×1128\times 1 | 23.6 | – | – | 89.8 | – | – | 201.80 | – | – |
| 64×264\times 2 | 18.8 | 27.3 | +8.6+8.6 | 91.8 | 64.9 | −29%-29\% | 201.39 | 201.83 | +0.44+0.44 |
| 32×432\times 4 | 24.2 | 27.4 | +3.2+3.2 | 89.8 | 52.6 | −41%-41\% | 202.11 | 201.67 | −0.44-0.44 |
| 8×168\times 16 | 15.6 | 23.6 | +8.0+8.0 | 89.2 | 65.9 | −26%-26\% | 201.54 | 201.90 | +0.36+0.36 |

## 5 Analysis

### 5.1 MI Diagnoses Collapse Better Than Entropy Across All Interventions

We demonstrate that MI separates high- and low-performance runs across all three intervention families better, and entropy could conflate them. At the same training budget, stronger SNR-Aware Filtering moves runs toward higher MI and better performance; KL and entropy tuning shift entropy without moving MI. We sweep three families of interventions (entropy regularization strength, KL\mathrm{KL} constraint strength, and SNR-Aware Filtering keep rate) and compare their trajectories in both diagnostic spaces at fixed training steps (Figure [13](#S6.F13 "Figure 13 ‣ 6 Related Work ‣ RAGEN-2: Reasoning Collapse in Agentic RL")). Entropy- and KL\mathrm{KL}-based stabilizers induce larger changes in H​(Z∣X)H(Z\mid X) than in I^​(X;Z)\widehat{I}(X;Z), and rarely move the model into the high-I^​(X;Z)\widehat{I}(X;Z) regime with clearly improved performance. In contrast, SNR-Aware Filtering traces a monotone improvement in both I^​(X;Z)\widehat{I}(X;Z) and task success; pushing entropy too high leads to instability and performance collapse, while KL\mathrm{KL} constraint mainly anchors the policy near its reference distribution without boosting input dependence.

We compute Spearman correlation between task success rate and each candidate diagnostic across runs with varying entropy regularization strength, KL constraint strength, and Top-p filtering kept mass (Figure [8](#S5.F8 "Figure 8 ‣ 5.1 MI Diagnoses Collapse Better Than Entropy Across All Interventions ‣ 5 Analysis ‣ RAGEN-2: Reasoning Collapse in Agentic RL")). MI-family metrics achieve positive correlations, with Trajectory MI-ZScore reaching +0.39+0.39. In contrast, Reasoning Entropy and Conditional Entropy metrics show near-zero or negative correlations (between −0.11-0.11 and −0.14-0.14). This confirms that MI predicts performance twice as reliably as entropy does, and entropy actually points in the wrong direction. These results validate MI as a superior training monitor compared to entropy-based diagnostics for multi-turn agent RL.

!(/html/2604.06268/assets/table_figures/drawer/F08/output/F08_metric_family_vs_performance_trajectory_only.png)

Figure 8: Spearman correlations showing MI-family metrics positively predict performance while entropy metrics are near-zero or negative.

The no-filter baseline shows the weakest performance. This confirms that indiscriminate updates on all prompts, including those with near-zero reward variance, systematically degrades learning. These results motivate our choice of Top-p filtering as the primary SNR-Aware mechanism, with more comprehensive cross-environment results reported in Table [4](#S4.T4 "Table 4 ‣ 4.2 Template Collapse as a Consistent Failure Mode ‣ 4 Experiments ‣ RAGEN-2: Reasoning Collapse in Agentic RL").

Table [4](#S4.T4 "Table 4 ‣ 4.2 Template Collapse as a Consistent Failure Mode ‣ 4 Experiments ‣ RAGEN-2: Reasoning Collapse in Agentic RL") summarizes our experimental matrix over four tasks, multiple RL algorithms, model scales/types, and input modalities.
Across this grid, SNR-Aware Filtering yields two consistent effects.
First, it improves peak task success rate in most settings (reported as the +Δ+\Delta next to each peak), demonstrating that prioritizing high-signal updates strengthens learning efficiency.
Second, the gains span multiple experimental axes, including (i) the RL optimizer (PPO / DAPO / GRPO / Dr. GRPO), (ii) the model family and scale (Qwen2.5 from 0.5B to 7B; Llama3.2-3B), and (iii) the input modality (text- and image-conditioned Qwen2.5-VL).
Here, DAPO and Dr. GRPO are recent strong baselines that directly target stable training and mitigate collapse-like failure modes.
In Table [4](#S4.T4 "Table 4 ‣ 4.2 Template Collapse as a Consistent Failure Mode ‣ 4 Experiments ‣ RAGEN-2: Reasoning Collapse in Agentic RL"), DAPO “no-filter” results correspond to the original algorithms without our filtering applied.
DAPO itself also includes a filtering/acceptance step; it can be interpreted as a special case of our framework where the selection is fixed (equivalently, a top-PP filter with P→1.0P\to 1.0), while our SNR-Aware Filtering provides an explicit, tunable SNR knob via the keep rate ρ\rho.
This breadth suggests SNR-Aware Filtering serves as a general-purpose SNR control knob and works alongside standard stabilization terms (e.g., KL\mathrm{KL} and entropy regularization).

!(/html/2604.06268/assets/table_figures/drawer/F25/output/F25_filter_vs_no_filter_last_8_median_success.png)

Figure 9: In FrozenLake, median success rates for both Top-p filtering (orange) and no filtering (gray) decrease as environment stochasticity increases from 0% to 100%. SNR-Aware Filtering maintains a clear advantage from 0% to 50% stochasticity, but the gap closes at 80%–100%, where high transition noise weakens reward variance as an informative signal proxy.

### 5.2 Does SNR Mechanism Really Interpret Agent RL?

The SNR framing makes a concrete causal claim: template collapse is a gradient-level consequence of low reward variance, not a side effect of aggressive regularization or model capacity. We stress-test this claim with four questions: (1) Does directly controlling RV level causally drive performance and MI? (2) Does injecting environmental noise predictably weaken MI? (3) Do gains come from signal quality rather than prompt-distribution bias? (4) When does the filtering condition hold in practice? A positive answer to all four makes the SNR account difficult to dismiss.

Quartile ablation provides direct causal evidence. To move beyond correlation between RV and performance, we run a controlled intervention. We sort all prompt groups by within-prompt RV, divide them into four quartiles (Q1 = highest, Q4 = lowest), and train four separate runs — each updating on one quartile only, all other settings fixed (Table [6](#S5.T6 "Table 6 ‣ 5.2 Does SNR Mechanism Really Interpret Agent RL? ‣ 5 Analysis ‣ RAGEN-2: Reasoning Collapse in Agentic RL")). Task performance and MI degrade monotonically from Q1 to Q4. Combined with Theorem G.1 (‖gtask‖≤RV\|g\_{\text{task}}\|\leq\sqrt{\text{RV}}), this establishes the full causal chain: reward variance →\to gradient quality →\to input-dependent reasoning.

Table 6: Quartile ablation on Sokoban (Qwen2.5-3B, P=8P{=}8, G=16G{=}16, keeping 25% of prompts per step). Task performance and MI degrade monotonically from Q1 to Q4.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Quartile | RV Range | Task Perf (%) | MI Proxy | Entropy |
| Q1 (highest RV) | [4.4–5.6] | 21.1 | 0.95 | 2.02 |
| Q2 | [1.5–4.2] | 19.5 | 0.93 | 1.53 |
| Q3 | [0.0–0.2] | 10.7 | 0.81 | 1.41 |
| Q4 (lowest RV) | [0.0–0.1] | 11.0 | 0.73 | 1.87 |

Controlled noise injection weakens MI. We run a direct intervention: varying environmental stochasticity and asking whether MI declines *predictably* in response. As environment and policy randomness increases, task return drops, conditional entropy rises, and I^​(X;Z)\widehat{I}(X;Z) decreases monotonically (Figure [9](#S5.F9 "Figure 9 ‣ 5.1 MI Diagnoses Collapse Better Than Entropy Across All Interventions ‣ 5 Analysis ‣ RAGEN-2: Reasoning Collapse in Agentic RL")). This is the expected consequence of the SNR chain. Additional noise inflates within-prompt return variance in a signal-free way, diluting the advantage estimates that task gradients depend on. Importantly, the filter’s advantage also attenuates at very high noise (80–100%), which is itself informative: when the environment is so stochastic that even high-effort prompts yield noisy rewards, RV loses its discriminative power. The mechanism predicts exactly this boundary condition.

Prompt-level filtering outperforms trajectory-level filtering. The gains from SNR-Aware filtering could come from selecting discriminative prompts, or from discarding hard/noisy trajectories. We disentangle these with a trajectory-level baseline: we keep all prompts but retain only the top-8 and bottom-8 trajectories per prompt by reward, preserving the prompt distribution while improving per-prompt SNR (Table [7](#S5.T7 "Table 7 ‣ 5.2 Does SNR Mechanism Really Interpret Agent RL? ‣ 5 Analysis ‣ RAGEN-2: Reasoning Collapse in Agentic RL")). Trajectory-level filtering improves over no filtering. However, prompt-level SNR-Aware Filtering outperforms it by a wider margin. Within a naturally low-RV prompt, forcing within-prompt variance by sub-selecting trajectories amplifies noise. Selecting prompts that naturally produce discriminative signals is more effective.

Table 7: Trajectory-level vs. prompt-level filtering on Sokoban (Qwen2.5-3B). Prompt-level SNR-Aware Filtering provides the largest gains; trajectory-level filtering confirms that the benefit is not due to prompt-distribution bias.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Method | Prompts Used | Traj/Update | Task Perf (%) | MI Proxy |
| No filter | 8/8 | 128 | 12.9 | 0.83 |
| Prompt-level RV (ρ=0.9\rho{=}0.9) | 3.2/8 | 50.6 | 23.6 | 1.80 |
| Trajectory-level | 8/8 | 64 | 16.8 | 0.20 |

When does SNR-Aware Filtering help? Finally, we find SNR-Aware Filtering improves performance better when cross-prompt RV heterogeneity is large enough to separate signal-rich from noise-only prompts. We find the metric Std(RV)/Mean(RV), computable from a single rollout batch, can effectively predict this (Table [8](#S5.T8 "Table 8 ‣ 5.2 Does SNR Mechanism Really Interpret Agent RL? ‣ 5 Analysis ‣ RAGEN-2: Reasoning Collapse in Agentic RL")). When the ratio is high, the per-prompt RV distribution is bimodal and filtering cleanly separates signal from noise. When the ratio is near zero, all prompts carry similar RV and filtering discards data uniformly, like FrozenLake GRPO (Δ=−5.0%\Delta{=}{-5.0\%}, ratio 0.330.33). This ratio is a cheap diagnostic which can be done before training.

Table 8: Per-setting RV statistics and filtering effectiveness. Std/Mean of RV predicts whether SNR-Aware Filtering helps: high ratio means bimodal RV and effective filtering; low ratio means uniform RV and random discarding.

|  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Setting | Filter Δ\Delta | P | G | RV Mean | RV Std | RV Var | RV Min | RV Max | Std/Mean |
| Sokoban, 14B | +4.6%+4.6\% | 8 | 8 | 2.24 | 2.88 | 8.32 | 0.10 | 6.00 | 1.29 |
| Sokoban, 3B | +3.2%+3.2\% | 32 | 4 | 2.49 | 2.89 | 8.35 | 0.05 | 6.52 | 1.16 |
| FrozenLake, 3B (GRPO) | −5.0%-5.0\% | 32 | 8 | 0.54 | 0.18 | 0.03 | 0.22 | 0.76 | 0.33 |

How filtering adapts as training progresses? With the four predictions confirmed, we can now characterize how SNR-Aware Filtering behaves over the full training trajectory. Figure [10](#S5.F10 "Figure 10 ‣ 5.2 Does SNR Mechanism Really Interpret Agent RL? ‣ 5 Analysis ‣ RAGEN-2: Reasoning Collapse in Agentic RL") tracks the effective kept ratio ρeff\rho\_{\text{eff}} and zero-variance prompt count over training. Both move in the expected direction: as the policy improves and converges, more prompts yield near-identical rollout rewards (zero-variance count rises), and the filter responds by becoming more selective (kept ratio falls). This automatic tightening is precisely what a fixed strategy like Top-kk with constant kk cannot replicate. It would continue absorbing gradient budget from uninformative prompts even as signal quality deteriorates.

Reward collapse is visible at the distribution level. Figure [11](#S5.F11 "Figure 11 ‣ 5.2 Does SNR Mechanism Really Interpret Agent RL? ‣ 5 Analysis ‣ RAGEN-2: Reasoning Collapse in Agentic RL") provides a complementary view of the same dynamics, tracking prompt-level reward distributions across early, mid, and late training in Sokoban. The shift is systematic: the hard portion shrinks as the policy improves, the mixed portion expands, and overall prompt-level variance collapses toward the late stages. This distribution-level signature mirrors the gradient-level story. Late training is not simply "easier" for the policy; it is a regime where reward variation has been compressed to the point that gradient updates carry progressively less task-discriminative information.

!(/html/2604.06268/assets/table_figures/drawer/F04/output/F04_dynamic_filtering.png)

Figure 10: Effective kept ratio and zero-variance prompt count, showing adaptive selection pressure as variance collapses during training.

Format validity cannot substitute for content-sensitive diagnostics. One might hope that a coarser signal (whether the model’s output follows the required format) could serve as a collapse indicator without the overhead of MI estimation. Figure [12](#S5.F12 "Figure 12 ‣ 5.2 Does SNR Mechanism Really Interpret Agent RL? ‣ 5 Analysis ‣ RAGEN-2: Reasoning Collapse in Agentic RL") shows this does not hold: format validity is largely decoupled from collapse, with runs maintaining near-perfect validity while exhibiting low MI. Structural correctness and semantic input-dependence are separate dimensions. This reinforces the need for content-sensitive diagnostics, and explains why the MI proxy provides signal that format-based checks miss.

!(/html/2604.06268/assets/table_figures/drawer/F16/F16.png)

Figure 11: Prompt-level reward distribution across training phases, showing RV collapse as prompts shift toward uniform reward structures.

RV is largely orthogonal to entropy and response length, which explains why entropy-based stabilizers cannot prevent template collapse. Reward variance correlates weakly with conditional entropy (Spearman −0.14-0.14) and response length (0.120.12), while correlating strongly with task reward (0.630.63). RV therefore targets a distinct axis of update quality rather than surface statistics, making it a complementary control knob to KL\mathrm{KL} and entropy regularization. Figure [10](#S5.F10 "Figure 10 ‣ 5.2 Does SNR Mechanism Really Interpret Agent RL? ‣ 5 Analysis ‣ RAGEN-2: Reasoning Collapse in Agentic RL") further shows that the effective kept ratio adapts over training: as more prompts drift toward near-zero RV, the filter automatically concentrates gradient updates on the shrinking pool of still-informative prompts.

!(/html/2604.06268/assets/table_figures/drawer/F9/F9.png)

Figure 12: Format validity versus MI and entropy diagnostics, showing that high validity does not guarantee high input dependence.

What is the relationship between SNR-Aware Filtering and KL/entropy tuning stabilization?
When training RL agents, practitioners typically tune KL penalty and entropy regularization coefficients to maintain training stability and prevent mode collapse. However, these interventions primarily control within-input diversity (H​(Z∣X)H(Z\mid X)) and cannot directly address the signal-to-noise imbalance that drives template collapse. Even with carefully tuned regularization, if most prompts have low reward variance, the task gradient remains weak and regularization forces still dominate the update direction.

SNR-Aware Filtering is complementary: it selects high-signal prompts at each iteration, directly boosting the fraction of task-discriminative gradient in each update. This acts as a signal-enhancement mechanism rather than a noise-control mechanism. We provide a detailed empirical comparison of KL tuning, entropy tuning, and SNR-Aware Filtering in [Section˜5.1](#S5.SS1 "5.1 MI Diagnoses Collapse Better Than Entropy Across All Interventions ‣ 5 Analysis ‣ RAGEN-2: Reasoning Collapse in Agentic RL"), showing that the three interventions move training dynamics along different axes (Figure [13](#S6.F13 "Figure 13 ‣ 6 Related Work ‣ RAGEN-2: Reasoning Collapse in Agentic RL")).

## 6 Related Work

!(/html/2604.06268/assets/x4.png)

Figure 13: Training dynamics under three interventions. For each setting, we choose two checkpoints (steps 10/400) and connect them into a trajectory (arrows point to later steps). Color intensity indicates weaker to stronger intervention.

*Reasoning collapse and policy degeneracy in closed-loop LM and Agent RL training.*

LLM-agent RL reports various collapse phenomena (guo\_2025; wei2025gtrguidedthoughtreinforcement): *reasoning collapse* (rationales becoming templated with weaker input correspondence) (wei2025gtrguidedthoughtreinforcement; yao2025diversityawarepolicyoptimizationlarge; yun2025priceformatdiversitycollapse) and *policy-level degeneracy* (behavior concentrating on easy-to-reproduce patterns) (feng2025groupingrouppolicyoptimizationllm; wang2025practitionersguidemultiturnagentic). These echo model collapse in self-training, even when average metrics appear stable (gerstgrasser2024modelcollapseinevitablebreaking; Shumailov2024AIMC).

*Evaluating reasoning diversity, input dependence, and reasoning faithfulness.*

Most diversity metrics do not test whether differences are *systematically driven by inputs* (tevet2021evaluatingevaluationdiversitynatural; yun2025priceformatdiversitycollapse). Common measures include lexical overlap (li2016diversitypromotingobjectivefunctionneural; zhu2018texygenbenchmarkingplatformtext), embedding dispersion (pillutla2021mauvemeasuringgapneural; tevet2021evaluatingevaluationdiversitynatural), and uncertainty analyses (montahaei2019jointlymeasuringdiversityquality; semeniuta2019accurateevaluationganslanguage), primarily capturing within-input variability and missing cross-input shifts (semeniuta2019accurateevaluationganslanguage; tevet2021evaluatingevaluationdiversitynatural). Recent work probes input dependence via behavioral tests (gardner2020evaluatingmodelslocaldecision; ribeiro2020accuracybehavioraltestingnlp; zhu2024promptbenchunifiedlibraryevaluation) and retrieval-style matching (morris2023languagemodelinversion; gao2024dorydeliberativepromptrecovery; zhang2024extractingpromptsinvertingllm; li2025reversepromptengineering). Work on *reasoning faithfulness* asks whether explanations reflect true decision bases (lanham2023measuringfaithfulnesschainofthoughtreasoning; turpin2023languagemodelsdontsay; siegel2024probabilitiesmatterfaithfulmetric; zaman2025chainofthoughtreallyexplainabilitychainofthought). We instead focus on the phenomenon that reasoning may become less input-sensitive after RL.

*Stabilizing multi-turn Agent RL under closed-loop sampling.*

Stability work spans KL control, entropy regularization, clipping, reward shaping, curricula, and replay mixtures (schulman2017trustregionpolicyoptimization; schulman2017proximalpolicyoptimizationalgorithms; schulman2018highdimensionalcontinuouscontrolusing; haarnoja2019softactorcriticalgorithmsapplications; stiennon2022learningsummarizehumanfeedback; ouyang2022traininglanguagemodelsfollow; rafailov2024directpreferenceoptimizationlanguage; sun2024fastbestofndecodingspeculative; feng2025groupingrouppolicyoptimizationllm; wang2025practitionersguidemultiturnagentic; wang2025harnessinguncertaintyentropymodulatedpolicy; xu2025epoentropyregularizedpolicyoptimization; yao2025diversityawarepolicyoptimizationlarge). For multi-step agents, stepwise rewards and self-correction signals are common (cobbe2021trainingverifierssolvemath; nakano2022webgptbrowserassistedquestionansweringhuman; uesato2022solvingmathwordproblems; madaan2023selfrefineiterativerefinementselffeedback; shinn2023reflexionlanguageagentsverbal; wang2023voyageropenendedembodiedagent; yao2023reactsynergizingreasoningacting; dou2025rerestreflectionreinforcedselftraininglanguage; wei2025gtrguidedthoughtreinforcement). However, these methods do not prevent drift toward input-agnostic templates: if rollouts receive similar rewards regardless of reasoning quality, gradients carry little information (moskovitz2023confrontingrewardmodeloveroptimization; o'mahony2024attributing; Shumailov2024AIMC; yun2025priceformatdiversitycollapse). We adopt a SNR view, using reward variance filtering low-signal samples to maintain effective SNR.

## 7 Conclusions and Limitations

We find closed-loop multi-turn agent RL can fail silently: reasoning drifts toward fluent but input-agnostic boilerplate while conditional entropy remains stable. We define this as template collapse.
Built on this, the paper makes three contributions. First, we introduce a mutual information (MI) proxy between inputs and reasoning, which interprets template collapse and tracks task performance better than conditional entropy. To explain why collapse occurs, we propose SNR mechanism in RL and show that low within-input reward variance suppresses task gradients and lets regularization forces dominate, pushing policy outputs toward input-agnostic templates. To address this, we introduce SNR-Aware Filtering to prioritize prompts with reward variance before each parameter update, improving performance on average across tasks, model scales, and modalities and can integrate easily with existing training pipelines.

Limitations. The SNR decomposition assumes task-signal and regularization noise separate cleanly, though they may couple through gradient accumulation in practice. All
experiments are single-agent; how template collapse propagates in multi-agent RL remains open. A capable model could game the filtering criterion by artificially inflating
reward variance, a risk worth monitoring over long training horizons. The method requires reward variance to be a reliable signal proxy, which degrades in sparse or noisy
reward environments. Aggressive filtering may narrow exploration coverage; the kept mass requires per-task tuning.

## 8 Acknowledgements

We thank Yuxiang Lin for help with RAGEN infrastructure and environments, and Kyunghyun Cho for insightful discussions on the manuscript.

## Appendix Contents

## Appendix A Extended Related Work

#### *Reasoning collapse and policy degeneracy in closed-loop LM and agent RL training.*

We study a family of degradation phenomena in closed-loop LLM-agent reinforcement learning that has not yet been uniformly defined, but has been repeatedly reported across settings (guo\_2025; wei2025gtrguidedthoughtreinforcement). After the model is updated on self-sampled trajectories over time, it may gradually exhibit *reasoning collapse* and *policy-level degeneracy* (guo\_2025; wei2025gtrguidedthoughtreinforcement). Here, *reasoning collapse* mainly refers to the rationales, plans, or explanations becoming increasingly templated and less diverse, while their correspondence to the input goal weakens (wei2025gtrguidedthoughtreinforcement; yao2025diversityawarepolicyoptimizationlarge; yun2025priceformatdiversitycollapse). In contrast, *policy-level degeneracy* refers to behavioral choices concentrating on a small set of easy-to-reproduce action patterns that yield stable scores, with less exploration and less error correction (feng2025groupingrouppolicyoptimizationllm; wang2025practitionersguidemultiturnagentic).

This family of phenomena echoes earlier findings in self-training, self-distillation, and iterative fine-tuning on synthetic or model-generated data. When a model repeatedly trains on its own generated distribution, the feedback loop can gradually narrow the effective data distribution, amplify a few high-probability modes, and suppress long-tail behaviors, even when average quality metrics appear stable (gerstgrasser2024modelcollapseinevitablebreaking; Shumailov2024AIMC). In the agent RL setting, closed-loop optimization on on-policy trajectories introduces additional risks, but these risks do not necessarily appear first as an overt failure of the behavioral policy. Instead, a commonly reported pattern is that, even when the agent’s external behavior remains effective or yields stable rewards, language-level reasoning expressions can become concentrated earlier. Plans and explanations may converge to a few reusable narrative skeletons, and their alignment with the specific input goal can weaken (wei2025gtrguidedthoughtreinforcement; xu2025epoentropyregularizedpolicyoptimization). In other words, reasoning-level degeneration can decouple from policy-level degeneracy, and in some settings it may precede it (wang2025practitionersguidemultiturnagentic). In multi-turn interaction, related work also describes several visible signatures of this degradation family, such as within-task convergence across repeated rollouts, cross-task templating where different prompts share the same planning or rhetorical skeleton, and late-stage degeneration where later turns become more mechanical or more conservative (wang2025harnessinguncertaintyentropymodulatedpolicy; xu2025epoentropyregularizedpolicyoptimization).

#### *Evaluating reasoning diversity, input dependence, and reasoning faithfulness.*

Prior work on evaluating *reasoning diversity* often answers how different the outputs are, but less directly answers whether these differences are *systematically driven by the input goal*, which can blur the interpretation of template-like degeneration under closed-loop training (tevet2021evaluatingevaluationdiversitynatural; yun2025priceformatdiversitycollapse).
Concretely, common metrics range from lexical measures such as n-gram statistics and self-BLEU (li2016diversitypromotingobjectivefunctionneural; zhu2018texygenbenchmarkingplatformtext), to embedding-based dispersion and distributional distances (pillutla2021mauvemeasuringgapneural; tevet2021evaluatingevaluationdiversitynatural), as well as token-level uncertainty proxies and multi-sample coverage or consistency analyses (montahaei2019jointlymeasuringdiversityquality; semeniuta2019accurateevaluationganslanguage).
These metrics primarily capture overall randomness or within-input variability, and they are often less sensitive to whether the reasoning distribution changes coherently *across* inputs (semeniuta2019accurateevaluationganslanguage; tevet2021evaluatingevaluationdiversitynatural).
Other evaluation protocols rely on model scoring or human preference judgments to compare overall response quality, but they are not designed to isolate input-conditioned reasoning differences, and they may conflate prompt-coupled variation with prompt-agnostic surface diversity, especially when outputs converge to shared formats (kirk2024understandingeffectsrlhfllm; yun2025priceformatdiversitycollapse).
This leaves a gap for scalable evaluation of whether reasoning is *diagnostic of the input*, which is particularly salient in multi-turn, stochastic environments where a fixed agent policy can produce diverse yet reusable templates (wang2025practitionersguidemultiturnagentic).
Recent work has started to probe input dependence via behavioral tests and local boundary checks (gardner2020evaluatingmodelslocaldecision; ribeiro2020accuracybehavioraltestingnlp), prompt robustness benchmarks (zhu2024promptbenchunifiedlibraryevaluation), and retrieval-style output–input matching or prompt reconstruction signals (morris2023languagemodelinversion; gao2024dorydeliberativepromptrecovery; zhang2024extractingpromptsinvertingllm; li2025reversepromptengineering).
However, a unified and scalable treatment tailored to closed-loop agent RL remains limited, even as algorithmic work continues to address long-horizon stability and collapse (feng2025groupingrouppolicyoptimizationllm; yao2025diversityawarepolicyoptimizationlarge).

A closely related line studies *reasoning faithfulness* (explanation faithfulness), which asks whether a rationale reflects the true basis of a decision rather than a plausible post-hoc story (lanham2023measuringfaithfulnesschainofthoughtreasoning; turpin2023languagemodelsdontsay; siegel2024probabilitiesmatterfaithfulmetric; zaman2025chainofthoughtreallyexplainabilitychainofthought).
Our question is related but not equivalent: faithfulness emphasizes whether reasoning causally supports a particular decision, while we focus on a different degeneration risk in closed-loop optimization, namely whether reasoning gradually becomes *less sensitive to the input* and drifts toward reusable templates, even when local explanations remain self-consistent (kirk2024understandingeffectsrlhfllm).
This motivates our decomposition of reasoning diversity into within-input variability and cross-input dependence, and our scalable proxy for the latter through an information-theoretic lens.

#### *Stabilizing multi-turn Agent RL under closed-loop sampling.*

To improve training stability when aligning LLMs and LLM-based agents, prior work has proposed a broad set of algorithmic and system-level techniques.
These include KL control or trust-region style constraints, entropy regularization, clipping and normalization in policy-gradient updates, reward shaping and credit assignment, curriculum design, replay or offline–online mixtures, as well as rejection sampling and best-of-N selection (schulman2017trustregionpolicyoptimization; schulman2017proximalpolicyoptimizationalgorithms; schulman2018highdimensionalcontinuouscontrolusing; haarnoja2019softactorcriticalgorithmsapplications; stiennon2022learningsummarizehumanfeedback; ouyang2022traininglanguagemodelsfollow; rafailov2024directpreferenceoptimizationlanguage; sun2024fastbestofndecodingspeculative; feng2025groupingrouppolicyoptimizationllm; wang2025practitionersguidemultiturnagentic; wang2025harnessinguncertaintyentropymodulatedpolicy; xu2025epoentropyregularizedpolicyoptimization; yao2025diversityawarepolicyoptimizationlarge).
For multi-step agents, researchers have also explored stepwise rewards and intermediate supervision, imitation-to-RL pipelines, and self-correction or reflection signals to support longer-horizon planning and reduce brittle behaviors (cobbe2021trainingverifierssolvemath; nakano2022webgptbrowserassistedquestionansweringhuman; uesato2022solvingmathwordproblems; madaan2023selfrefineiterativerefinementselffeedback; shinn2023reflexionlanguageagentsverbal; wang2023voyageropenendedembodiedagent; yao2023reactsynergizingreasoningacting; dou2025rerestreflectionreinforcedselftraininglanguage; wei2025gtrguidedthoughtreinforcement).

Despite these advances, many stabilization methods are tuned to prevent optimization collapse or to improve overall reward.
When the effective learning signal in the closed loop becomes weak or noisy, these methods do not necessarily prevent drift toward prompt-agnostic templates.
For example, if most rollouts for the same prompt receive similar rewards regardless of reasoning quality, then the gradient update carries little information about which reasoning path matters (moskovitz2023confrontingrewardmodeloveroptimization; o'mahony2024attributing; Shumailov2024AIMC; yun2025priceformatdiversitycollapse).
This motivates methods that explicitly manage the balance between task-specific signal and task-agnostic pressure.
We adopt a signal-to-noise view of closed-loop updates: we use within-prompt reward variance as a proxy for signal strength, and we filter low-signal samples to maintain an effective SNR, so that exploration and input-conditioned reasoning are less likely to be washed out over long-horizon multi-turn optimization (romoff2018rewardestimationvariancereduction; shao2024deepseekmathpushinglimitsmathematical; tao2025hybridreinforcementrewardsparse; feng2025groupingrouppolicyoptimizationllm; yao2025diversityawarepolicyoptimizationlarge).

## Appendix B Detailed Experimental Settings

### B.1 Environments and Tasks

We construct a diverse seven-environment testbed to evaluate LLM agents across complementary axes of decision-making complexity, including planning under irreversible dynamics (Sokoban), long-horizon control with non-deterministic transitions (FrozenLake), multi-step symbolic reasoning in mathematics (MetaMathQA, Countdown), multi-turn search and information synthesis (SearchQA), goal-directed web navigation (WebShop), and program synthesis from input-output specifications (DeepCoder). All environments are synthetic and fully controllable, enabling clean analysis of RL learning from scratch without relying on real-world priors.

Sokoban. We use the puzzle Sokoban schrader2018gymsokoban to study multi-turn agent interaction with irreversible dynamics. The agent must push boxes to designated target locations within a grid-based warehouse. Unlike standard navigation tasks, Sokoban is characterized by irreversibility: boxes can only be pushed, not pulled, meaning a single misstep can create unsolvable dead-ends where boxes become permanently stuck against walls or corners. This requires the agent to reason ahead and plan multi-step sequences before committing to actions. The reward signal encourages both efficiency and accuracy: +1+1 for each box successfully placed on a target, −1-1 for moving a box off a target, +10+10 upon task completion, and −0.1-0.1 per action as a step penalty. We use procedurally generated puzzles with configurable room dimensions and box counts to ensure diverse training scenarios.

Frozen Lake. This environment of FrozenLake brockman2016openai\_gym combines long-horizon decision-making with deterministic transitions. The agent navigates a grid of frozen tiles to reach a goal while avoiding holes that terminate the episode. We use the 2% random rate variant of Frozen Lake, where each intended action is executed at a 98% probability. Rewards are sparse: only successful goal-reaching trials receive a reward of +1+1, with all other outcomes yielding 0. The combination of sparse rewards and long-horizon planning makes this environment challenging for credit assignment.

MetaMathQA. To evaluate mathematical reasoning capabilities, we include MetaMathQA yu2023metamath, a question-answering task drawn from the MetaMathQA dataset. Each episode presents the agent with a mathematical problem requiring multi-step reasoning—ranging from arithmetic and algebra to word problems and geometry. The agent must produce a final answer, and correctness is determined by exact match with the ground truth. To encourage efficient reasoning, we employ a diminishing reward scheme: correct answers on the first attempt receive full reward (1.01.0), with rewards halving for each subsequent attempt (0.50.5, 0.250.25, …).

Countdown. Inspired by the numbers game from the TV show “Countdown” katz2025countdown, this environment tests compositional arithmetic reasoning. The agent is given a target number and a set of source numbers, and must construct an arithmetic expression using each source number at most once to reach the target exactly. For example, given target 2424 and numbers [1,5,6,7][1,5,6,7], a valid solution is 6×(7−5+1)+66\times(7-5+1)+6. Rewards distinguish between format correctness and solution correctness: full reward (1.01.0) for correct solutions, partial reward (0.10.1) for expressions that use the correct numbers but yield incorrect results, and zero for malformed expressions.

DeepCoder. To evaluate agent capabilities in coding environments, we use DeepCoder, a coding benchmark consisting of competitive programming problems. It was used to train DeepSeek-R1-Distill-Qwen-14B with reinforcement learning. The benchmark draws from three resources: PrimeIntellect mattern2025synthetic1, TACOli2023taco, and LiveCodeBench v5 (LCBv5) jain2024livecodebench. In this environment, agents are required to generate a Python function that solves the given programming problem and passes all hidden and public test cases. During training, rewards are assigned based on the number of test cases successfully passed.

SearchQA. To evaluate multi-turn search and question-answering capabilities, we include SearchQA from the RLLM framework rllm2025, specifically the Search R1 variant. This environment requires the agent to perform iterative web search and reasoning to answer open-domain questions. The agent must formulate search queries, extract relevant information from retrieved documents, and synthesize answers across multiple interaction turns. Rewards are based on answer correctness and search efficiency, encouraging the agent to balance exploration breadth with reasoning depth.

WebShop. We use WebShop yao2022webshop, an interactive e-commerce environment for evaluating goal-directed multi-turn decision-making. The agent is presented with a shopping instruction (e.g., “find a red shirt under $30”) and must navigate a simulated online shopping website by issuing search queries, clicking on products, and selecting appropriate items. The environment features a large action space with realistic product catalogs and requires the agent to perform language understanding, attribute matching, and sequential decision-making. Rewards are assigned based on how well the purchased item matches the specified attributes and constraints.

### B.2 Training and Evaluation Setup

We conduct our main experiments using Qwen2.5-3B and train with four policy-gradient variants—PPO, DAPO, GRPO, and Dr.GRPO—for up to 400 rollout–update iterations on NVIDIA GPUs using the veRL framework, with early stopping enabled as described below. Each iteration collects K=128K=128 trajectories per environment, organized as P=8P=8 prompt groups with G=16G=16 parallel samples per prompt.

Episode horizons. To match task structure, the interactive environments (Sokoban, Frozen Lake) use up to 5 interaction turns with 2 actions per turn (10 total actions per trajectory). The single-step reasoning tasks (Countdown, MetaMathQA) use 1 turn with 1 action.

Optimization. We use an update batch size of 32 and a per-GPU minibatch size of 4. Policy optimization uses GAE with (γ,λ)=(1.0,1.0)(\gamma,\lambda)=(1.0,1.0) and Adam with (β1,β2)=(0.9,0.999)(\beta\_{1},\beta\_{2})=(0.9,0.999). The actor learning rate is 1×10−61\times 10^{-6} and the critic learning rate is 1×10−51\times 10^{-5}. We apply entropy regularization with coefficient β=0.001\beta=0.001. For PPO-based methods, we use asymmetric clipping with ϵlow=0.2\epsilon\_{\text{low}}=0.2 and ϵhigh=0.28\epsilon\_{\text{high}}=0.28. We additionally impose a format penalty of −0.1-0.1 when the agent fails to output a valid structured response (e.g., missing <think> or <answer> tags).

Early stopping. We stop training if either (i) reward-variance collapse is detected—the reward variance drops below 10% of the baseline variance (defined as the mean variance over the first 10 training iterations) for 5 consecutive iterations—or (ii) the validation success rate remains below 1% for 5 consecutive evaluation checkpoints.

Filtering ablation. We compare filtered rollouts with top\_p=0.9\texttt{top\\_p}=0.9 (keeping the top 90% of trajectory groups ranked by reward variance) against an unfiltered setting.

Evaluation. We evaluate on a fixed set of 512 validation prompts per environment and decode with temperature T=0.5T=0.5 using stochastic sampling. We report success rate as the primary metric across all environments.

## Appendix C Filtering Ablation Results

We conduct our filtering experiments using Qwen2.5-3B model on Sokoban environment. We summarize the filtering ablation results in Table [9](#A3.T9 "Table 9 ‣ Appendix C Filtering Ablation Results ‣ RAGEN-2: Reasoning Collapse in Agentic RL"). Each row reports the absolute value of each metric, with the change relative to a section-specific baseline shown in parentheses. Within each block, the first row labeled *baseline* defines the reference point, and all deltas are computed relative to that baseline. We report four metrics: Task Performance, defined as the maximum validation success rate attained during training; MI Proxy, measured as retrieval accuracy at the training step where task performance peaks; Entropy, an estimate of reasoning entropy at the same step; and Collapse, a binary indicator of whether validation success ever falls below 0.010.01 during training.

Sampling Settings.
We first study the interaction between filtering and sampling by varying sampling thresholds while holding the reward-variance (RV) filter fixed. Relative to the top\_p=1.0\texttt{top\\_p}=1.0 baseline, reducing top\_p or min\_p generally improves task performance while reducing entropy, but with heterogeneous effects on MI retention. In contrast, top\_k sampling induces a sharper trade-off: MI proxy is often preserved or improved, while gains in task performance are less consistent. These results indicate that filtering behavior is strongly modulated by the sampling regime, even when the underlying filter metric is unchanged.

Filtering Metrics.
Next, we fix the sampling scheme and vary the filtering criterion. Switching between RV, entropy-based, entropy-variance, and length-based filters leads to substantial differences in both peak task performance and MI proxy. In particular, SNR-Aware Filtering consistently achieves strong task performance while better preserving MI compared to entropy-based alternatives. Entropy- and length-based filters either suppress MI or fail to prevent collapse, suggesting that reward variance provides a more stable and informative signal for selecting useful rollouts.

Keep Strategy.
Finally, we compare *keep-largest* and *keep-smallest* strategies under the same top\_k configuration. As expected, retaining high-variance trajectory groups yields substantially higher task performance and MI proxy, while keeping the smallest-variance groups degrades both and markedly increases entropy. This asymmetry supports the hypothesis that high-variance rollouts contain more informative training signal, whereas low-variance rollouts are largely uninformative or noisy.

Summary.
Overall, the ablation reveals strong interactions between sampling strategy and filtering choice. More aggressive filtering is not universally beneficial, and the choice of filtering metric is critical: reward-variance filtering consistently improves task performance while maintaining information content, whereas entropy-based heuristics are less reliable and more prone to collapse.

Table 9: Ablation results for sampling strategies, filtering metrics, and keep strategies. Values in parentheses denote the change relative to the corresponding baseline in each block. A crossmark in the Stable column indicates training collapse.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| EXPERIMENT SETUP | TASK PERF | MI PROXY | ENTROPY | STABLE |
| Sampling Strategies | | | | |
| Top-p = 1.0 (Baseline) | 0.17 | 0.54 | 2.76 | ✗ |
| Top-p = 0.9 | 0.38 (+0.20) | 0.84 (+0.29) | 1.64 (-1.12) | ✓ |
| Top-p = 0.5 | 0.29 (+0.12) | 0.83 (+0.29) | 1.88 (-0.88) | ✓ |
| Min-p = 0.05 | 0.42 (+0.25) | 0.67 (+0.13) | 1.64 (-1.12) | ✓ |
| Min-p = 0.2 | 0.45 (+0.27) | 0.36 (-0.18) | 3.01 (+0.26) | ✓ |
| Top-k = 0.25 | 0.22 (+0.05) | 0.86 (+0.32) | 1.28 (-1.48) | ✓ |
| Top-k = 0.5 | 0.44 (+0.27) | 0.89 (+0.35) | 1.47 (-1.29) | ✓ |
| Filtering Metrics | | | | |
| No Filter (Baseline) | 0.17 | 0.54 | 2.76 | ✗ |
| Reward Variance | 0.38 (+0.20) | 0.84 (+0.29) | 1.64 (-1.12) | ✓ |
| Reward Sum | 0.24 (+0.07) | 0.80 (+0.26) | 4.18 (+1.42) | ✗ |
| Entropy | 0.20 (+0.02) | 0.41 (-0.14) | 2.20 (-0.56) | ✗ |
| Entropy Variance | 0.23 (+0.06) | 0.70 (+0.16) | 2.94 (+0.18) | ✗ |
| Length | 0.16 (-0.02) | 0.91 (+0.36) | 1.65 (-1.10) | ✗ |
| Keep Strategies | | | | |
| Keep Largest (Baseline) | 0.44 | 0.89 | 1.47 | ✓ |
| Keep Smallest | 0.29 (-0.15) | 0.47 (-0.42) | 5.31 (+3.84) | ✓ |

## Appendix D Additional Experimental Visualizations

This section presents supplementary visualizations that provide deeper insights into the mechanisms and diagnostics discussed in the main paper. These figures complement the core experimental results with detailed breakdowns of gradient dynamics, diagnostic validity, and reward distribution patterns.

### D.1 MI Proxy Metrics During Training

Figure [14](#A4.F14 "Figure 14 ‣ D.1 MI Proxy Metrics During Training ‣ Appendix D Additional Experimental Visualizations ‣ RAGEN-2: Reasoning Collapse in Agentic RL") presents six alternative mutual-information proxy metrics tracked over the course of training, complementing the retrieval accuracy shown in Figure [5](#S4.F5 "Figure 5 ‣ 4.2 Template Collapse as a Consistent Failure Mode ‣ 4 Experiments ‣ RAGEN-2: Reasoning Collapse in Agentic RL") of the main paper. All proxies exhibit a consistent pattern: under the *No Filtering* baseline, MI proxies degrade sharply as training progresses, while the three intervention strategies (entropy regularization, KL regularization, and top-pp filtering) maintain stable information retention throughout training.

!(/html/2604.06268/assets/x5.png)

Figure 14: Six MI proxy metrics over training steps. (a) MI Estimate, (b) MI Z-Score (EMA), (c) Retrieval Above Chance, (d) Retrieval Accuracy @4, (e) Retrieval Accuracy @8, (f) Conditional Entropy H​(Z|X)H(Z|X). Consistent with Figure [5](#S4.F5 "Figure 5 ‣ 4.2 Template Collapse as a Consistent Failure Mode ‣ 4 Experiments ‣ RAGEN-2: Reasoning Collapse in Agentic RL"), all proxies confirm that without filtering, MI degrades early, signaling reasoning collapse. Filtering effectively preserves information retention across all metrics, with top-pp SNR-aware filtering best maintaining reasoning diversity throughout training.

## Appendix E Notation and basic identities

### E.1 Random variables and distributions

###### Definition E.1 (Prompts, trajectories, and rollouts).

Let XX denote an input prompt and ZZ a reasoning trajectory. A rollout sample is

|  |  |  |
| --- | --- | --- |
|  | ξ=(x,z,r),\xi=(x,z,r), |  |

with xx the prompt, zz the realized trajectory, and r∈ℝr\in\mathbb{R} the scalar reward.

We write πθ​(z∣x)\pi\_{\theta}(z\mid x) as the policy and P​(X)P(X) the prompt distribution. Rollouts are generated by

|  |  |  |
| --- | --- | --- |
|  | x∼P(X),z∼πθ(⋅∣x),r=R(z;x),x\sim P(X),\qquad z\sim\pi\_{\theta}(\cdot\mid x),\qquad r=R(z;x), |  |

where R​(z;x)R(z;x) is the reward function.

###### Definition E.2 (Baseline and advantage).

Let b​(x)b(x) be any function of xx only. Define the advantage

|  |  |  |
| --- | --- | --- |
|  | A​(z;x):=R​(z;x)−b​(x).A(z;x):=R(z;x)-b(x). |  |

A standard choice is the conditional-mean baseline
b​(x):=𝔼​[R​(Z;x)∣X=x].b(x):=\mathbb{E}[R(Z;x)\mid X=x].
Then the advantage is zero-mean within each prompt:

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[A​(Z;x)∣X=x]=𝔼​[R​(Z;x)∣X=x]−b​(x)=0.\mathbb{E}[A(Z;x)\mid X=x]=\mathbb{E}[R(Z;x)\mid X=x]-b(x)=0. |  |

###### Definition E.3 (Score function).

Define the score function

|  |  |  |
| --- | --- | --- |
|  | s​(z;x):=∇θlog⁡πθ​(z∣x).s(z;x):=\nabla\_{\theta}\log\pi\_{\theta}(z\mid x). |  |

It satisfies the normalization identity

|  |  |  |
| --- | --- | --- |
|  | 𝔼z∼πθ(⋅∣x)​[s​(z;x)]=∇θ​∫πθ​(z∣x)​𝑑z=0.\mathbb{E}\_{z\sim\pi\_{\theta}(\cdot\mid x)}[\,s(z;x)\,]=\nabla\_{\theta}\int\pi\_{\theta}(z\mid x)\,dz=0. |  |

###### Definition E.4 (Within-prompt reward variance).

We quantify within-prompt variation of observed rewards across rollouts by

|  |  |  |
| --- | --- | --- |
|  | RV(x):=Var(R(Z;x)∣X=x),Z∼πθ(⋅∣x).\mathrm{RV}(x):=\mathrm{Var}(R(Z;x)\mid X=x),\qquad Z\sim\pi\_{\theta}(\cdot\mid x). |  |

Low RV​(x)\mathrm{RV}(x) implies rewards are nearly constant within the prompt, so rollouts are weakly distinguishable by the reward signal. High RV​(x)\mathrm{RV}(x) indicates large within-prompt variation of observed rewards which may arise from trajectory-dependent signal or evaluation noise.

### E.2 Entropy and mutual information

###### Definition E.5 (Conditional entropy).

The within-input variability of reasoning is measured by

|  |  |  |
| --- | --- | --- |
|  | H​(Z∣X):=𝔼x∼P​(X)​[H​(Z∣X=x)]=−𝔼x∼P(X),z∼πθ(⋅∣x)​[log⁡πθ​(z∣x)].H(Z\mid X):=\mathbb{E}\_{x\sim P(X)}\!\big[H(Z\mid X=x)\big]=-\mathbb{E}\_{x\sim P(X),\,z\sim\pi\_{\theta}(\cdot\mid x)}\!\big[\log\pi\_{\theta}(z\mid x)\big]. |  |

The cross-input dependence of reasoning is measured by

|  |  |  |
| --- | --- | --- |
|  | I​(X;Z):=𝔼x∼P(X),z∼πθ(⋅∣x)​[log⁡πθ​(z∣x)pθ​(z)],pθ​(z):=𝔼x∼P​(X)​[πθ​(z∣x)].I(X;Z):=\mathbb{E}\_{x\sim P(X),\,z\sim\pi\_{\theta}(\cdot\mid x)}\!\left[\log\frac{\pi\_{\theta}(z\mid x)}{p\_{\theta}(z)}\right],\qquad p\_{\theta}(z):=\mathbb{E}\_{x\sim P(X)}\big[\pi\_{\theta}(z\mid x)\big]. |  |

Equivalently, I(X;Z)=𝔼x∼P​(X)[KL(πθ(⋅∣x)∥pθ)]I(X;Z)=\mathbb{E}\_{x\sim P(X)}\!\big[\mathrm{KL}(\pi\_{\theta}(\cdot\mid x)\,\|\,p\_{\theta})\big].

#### Decomposition identity (Shannon quantities).

For the true distribution induced by πθ\pi\_{\theta}, the Shannon identity

|  |  |  |  |
| --- | --- | --- | --- |
|  | H​(Z)=H​(Z∣X)+I​(X;Z),H(Z)\;=\;H(Z\mid X)\;+\;I(X;Z), |  | (2) |

serves only as conceptual equation: it specifies the two components we aim to track (within-prompt variability and cross-prompt dependence).
In practice we replace these Shannon quantities by scorer-defined proxies, e.g.,

|  |  |  |
| --- | --- | --- |
|  | 𝒟^q:=NLL^q​(Z∣X)+I^q​(X;Z),\widehat{\mathcal{D}}\_{q}:=\widehat{\mathrm{NLL}}\_{q}(Z\mid X)+\widehat{I}\_{q}(X;Z), |  |

which is in log-likelihood units under qq and does not in general satisfy the Shannon identity unless qq matches the evaluated distribution.

#### Interpretation for reasoning diversity.

In our setting, ZZ is a proxy for a reasoning process (e.g., a chain-of-thought trajectory).
A relative decrease in H​(Z∣X)H(Z\mid X) indicates within-prompt concentration of πθ(⋅∣x)\pi\_{\theta}(\cdot\mid x) (entropy collapse).
A relative decrease in I​(X;Z)I(X;Z) indicates weakened input dependence, i.e., trajectories become less diagnostic of xx.
In our analysis, this can occur when reward-driven updates are weak (e.g., low RV​(x)\mathrm{RV}(x)) and the total update is dominated by *reward-agnostic* components (e.g., KL/entropy regularizers).
We therefore track these two axes separately; in experiments we use scorer-defined proxies for H​(Z∣X)H(Z\mid X) and I​(X;Z)I(X;Z).

## Appendix F Scorer-based Proxies for Reasoning Diversity

### F.1 Setup and notation

We define scorer-based proxies using a fixed collection of prompts and multiple rollouts per prompt. Throughout this appendix, the scorer qq is fixed and used for evaluation.

###### Definition F.1 (Prompt groups).

Using the notation from Definition [E.1](#A5.Thmtheorem1 "Definition E.1 (Prompts, trajectories, and rollouts). ‣ E.1 Random variables and distributions ‣ Appendix E Notation and basic identities ‣ RAGEN-2: Reasoning Collapse in Agentic RL"), sample PP prompts {xi}i=1P∼P​(X)\{x\_{i}\}\_{i=1}^{P}\sim P(X). For each prompt xix\_{i}, sample GG trajectories

|  |  |  |
| --- | --- | --- |
|  | zi,k∼πθ(⋅∣xi),k=1,…,G.z\_{i,k}\sim\pi\_{\theta}(\cdot\mid x\_{i}),\qquad k=1,\dots,G. |  |

We refer to the set {zi,k}k=1G\{z\_{i,k}\}\_{k=1}^{G} as a *prompt group*.

###### Definition F.2 (Teacher-forced scorer and matched-pair score).

Let qq be a fixed language model used to score how compatible a trajectory zz is with a prompt xx. Define the matched-pair score

|  |  |  |
| --- | --- | --- |
|  | ℓi​(z):=log⁡q​(z∣xi).\ell\_{i}(z)\;:=\;\log q(z\mid x\_{i}). |  |

All proxies in this appendix are built from ℓi​(z)\ell\_{i}(z) and therefore are measured in log-likelihood units under qq.

###### Definition F.3 (Mixture score across prompts).

We evaluate each trajectory zz under all prompts {xj}j=1P\{x\_{j}\}\_{j=1}^{P} and define the mixture score

|  |  |  |
| --- | --- | --- |
|  | ℓmix​(z):=log⁡(1P​∑j=1Pexp⁡(ℓj​(z)))=log⁡(1P​∑j=1Pq​(z∣xj)).\ell\_{\mathrm{mix}}(z)\;:=\;\log\!\left(\frac{1}{P}\sum\_{j=1}^{P}\exp(\ell\_{j}(z))\right)=\log\!\left(\frac{1}{P}\sum\_{j=1}^{P}q(z\mid x\_{j})\right). |  |

This is the log-likelihood of zz under the uniform mixture over prompts induced by qq. Equivalently, ℓmix​(z)=log⁡(1P​∑j=1Pq​(z∣xj))\ell\_{\mathrm{mix}}(z)=\log\!\big(\frac{1}{P}\sum\_{j=1}^{P}q(z\mid x\_{j})\big) is the log-probability of zz under the empirical prompt mixture.

The quantities defined above depend on the sampled prompt set {xi}i=1P\{x\_{i}\}\_{i=1}^{P} and on the fixed scorer qq. They are proxies for within-prompt variability and input dependence of trajectories, and should not be interpreted as exact Shannon entropies or mutual information unless qq matches the evaluated conditional distribution.

## Appendix G Formal Definition of the Filtering Operator

###### Definition G.1 (Filtering operator).

Let ℬ\mathcal{B} be a minibatch of samples. A *filtering operator* is specified by:

(i) Grouping key. A grouping function g:ℬ→𝒢g:\mathcal{B}\to\mathcal{G} that assigns each sample ξ∈ℬ\xi\in\mathcal{B} a group label

|  |  |  |
| --- | --- | --- |
|  | u=g​(ξ).u=g(\xi). |  |

For u∈𝒢u\in\mathcal{G}, define the induced group subset

|  |  |  |
| --- | --- | --- |
|  | ℬu:={ξ∈ℬ:g​(ξ)=u}.\mathcal{B}\_{u}:=\{\xi\in\mathcal{B}:g(\xi)=u\}. |  |

(ii) Group statistic. A statistic ϕ:2ℬ→ℝ\phi:\mathcal{2^{\mathcal{B}}}\to\mathbb{R} that depends only on the samples in the group, and we write ϕ​(ℬu)\phi(\mathcal{B}\_{u}) for the value computed from ℬu\mathcal{B}\_{u}.

(iii) Selection rule (mask). Given a threshold τ∈ℝ\tau\in\mathbb{R}, the binary mask is

|  |  |  |
| --- | --- | --- |
|  | m​(u):=𝟏​{ϕ​(ℬu)≥τ}.m(u):=\mathbf{1}\{\phi(\mathcal{B}\_{u})\geq\tau\}. |  |

(iv) Filtered objective. For a per-sample RL loss Lθ​(ξ)L\_{\theta}(\xi), the filtered objective is

|  |  |  |
| --- | --- | --- |
|  | ℒfilt​(θ)=1|ℬ|​∑ξ∈ℬm​(g​(ξ))​Lθ​(ξ).\mathcal{L}\_{\mathrm{filt}}(\theta)\;=\;\frac{1}{|\mathcal{B}|}\sum\_{\xi\in\mathcal{B}}m\!\big(g(\xi)\big)\,L\_{\theta}(\xi). |  |

#### Remark (post-sampling).

Filtering is applied after sampling and only masks gradients; it does not change the rollout distribution.

#### Remark (normalization).

In practice one may normalize by the number of kept samples or kept groups (instead of |ℬ||\mathcal{B}|), which rescales the gradient but does not change which samples contribute nonzero gradients.

### G.1 Filtering Strategy Variants

We compare multiple filtering strategies for selecting high-signal prompt groups. All variants share the same grouping structure (prompts with GG rollouts each) and statistic (reward variance Var^​(R∣X=xi)\widehat{\mathrm{Var}}(R\mid X=x\_{i}) for group ii), but differ in the selection rule.

#### Top-p (nucleus-style) filtering.

The main method used in this paper. Given keep rate ρ∈(0,1]\rho\in(0,1], rank prompts by descending reward variance and select the smallest prefix whose cumulative variance mass reaches ρ​∑iVar^​(R∣X=xi)\rho\sum\_{i}\widehat{\mathrm{Var}}(R\mid X=x\_{i}). Formally, let σ\sigma be the permutation such that Var^​(R∣X=xσ​(1))≥⋯≥Var^​(R∣X=xσ​(N))\widehat{\mathrm{Var}}(R\mid X=x\_{\sigma(1)})\geq\cdots\geq\widehat{\mathrm{Var}}(R\mid X=x\_{\sigma(N)}), and define

|  |  |  |
| --- | --- | --- |
|  | k∗=min⁡{k:∑j=1kVar^​(R∣X=xσ​(j))≥ρ​∑i=1NVar^​(R∣X=xi)}.k^{\*}=\min\left\{k:\sum\_{j=1}^{k}\widehat{\mathrm{Var}}(R\mid X=x\_{\sigma(j)})\geq\rho\sum\_{i=1}^{N}\widehat{\mathrm{Var}}(R\mid X=x\_{i})\right\}. |  |

The kept set is S={σ​(1),…,σ​(k∗)}S=\{\sigma(1),\dots,\sigma(k^{\*})\}. This adaptive selection concentrates updates on high-variance prompts while automatically adjusting the kept count based on the variance distribution. When the batch contains many near-zero-variance prompts, top-p can reject the entire batch if the threshold cannot be reached, providing a natural safeguard against degenerate updates.

#### Top-k (proportional) filtering.

An alternative fixed-proportion baseline. Given ρ∈(0,1]\rho\in(0,1], compute k=⌊ρ​N⌋k=\lfloor\rho N\rfloor and select the top kk prompts by reward variance:

|  |  |  |
| --- | --- | --- |
|  | S={σ​(1),…,σ​(k)}.S=\{\sigma(1),\dots,\sigma(k)\}. |  |

Unlike top-p, top-k always retains exactly kk groups regardless of the variance distribution. This can be less adaptive: when most prompts have near-zero variance, top-k still keeps the highest-variance subset even if all retained prompts carry weak signal.

#### Min-p (threshold) filtering.

Inspired by min-p sampling, this strategy keeps all prompts whose variance exceeds a fraction of the maximum variance. Given threshold parameter p∈(0,1]p\in(0,1], define

|  |  |  |
| --- | --- | --- |
|  | τ=p⋅maxi⁡Var^​(R∣X=xi),\tau=p\cdot\max\_{i}\widehat{\mathrm{Var}}(R\mid X=x\_{i}), |  |

and keep all groups above the threshold:

|  |  |  |
| --- | --- | --- |
|  | S={i:Var^​(R∣X=xi)≥τ}.S=\left\{i:\widehat{\mathrm{Var}}(R\mid X=x\_{i})\geq\tau\right\}. |  |

This directly enforces a minimum quality bar: only prompts within a factor of pp of the best prompt are retained. The kept count varies with the variance distribution, making this method highly adaptive but potentially unstable when the maximum variance fluctuates.

#### Reverse top-p (low-variance) filtering.

A diagnostic baseline that intentionally selects low-variance prompts. Rank prompts by *ascending* reward variance and select the smallest prefix whose cumulative variance mass reaches ρ​∑iVar^​(R∣X=xi)\rho\sum\_{i}\widehat{\mathrm{Var}}(R\mid X=x\_{i}). This inverted strategy is used in ablation studies to confirm that high variance is essential for effective updates: training on low-variance prompts should degrade both MI and task performance, validating the SNR hypothesis.

#### Implementation notes.

All strategies can be configured to exclude zero-variance groups (setting include\_zero=False) before selection, which removes prompts where all rollouts received identical rewards. For top-p, we use a small epsilon ε=0.01\varepsilon=0.01 to ensure numerical stability when checking whether the cumulative threshold is reached. Additional implementation details and hyperparameter sensitivity are in the codebase.

## Appendix H RV Controls Task-Signal Magnitude and SNR

### H.1 Setup

We use the policy/score/baseline/advantage notation from Appendix [E](#A5 "Appendix E Notation and basic identities ‣ RAGEN-2: Reasoning Collapse in Agentic RL").

In particular, for a fixed prompt xx we write z∼πθ(⋅∣x)z\sim\pi\_{\theta}(\cdot\mid x),
s​(z;x)=∇θlog⁡πθ​(z∣x)s(z;x)=\nabla\_{\theta}\log\pi\_{\theta}(z\mid x),
A​(z;x)=R​(z;x)−b​(x)A(z;x)=R(z;x)-b(x) with b​(x)=𝔼​[R∣X=x]b(x)=\mathbb{E}[R\mid X=x],
and RV​(x)=Var​(R∣X=x)=𝔼​[A2∣X=x]\mathrm{RV}(x)=\mathrm{Var}(R\mid X=x)=\mathbb{E}[A^{2}\mid X=x].

### H.2 Assumption

###### Assumption H.1 (Reward decomposition).

The observed reward admits a decomposition

|  |  |  |
| --- | --- | --- |
|  | R​(z;x)=μ​(x,z)+ε,μ​(x,z):=𝔼​[R​(z;x)∣x,z],R(z;x)=\mu(x,z)+\varepsilon,\qquad\mu(x,z):=\mathbb{E}[R(z;x)\mid x,z], |  |

where μ​(x,z)\mu(x,z) is the trajectory-dependent mean reward and ε\varepsilon is a zero-mean noise term satisfying

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[ε∣x,z]=0,Var​(ε∣x,z)=σ2​(x)≥0.\mathbb{E}[\varepsilon\mid x,z]=0,\qquad\mathrm{Var}(\varepsilon\mid x,z)=\sigma^{2}(x)\geq 0. |  |

Moreover, the score s​(z;x)=∇θlog⁡πθ​(z∣x)s(z;x)=\nabla\_{\theta}\log\pi\_{\theta}(z\mid x) is a deterministic (measurable) function of (x,z)(x,z).

### H.3 Task-gradient magnitude is RV-controlled

The next result shows that the task-gradient norm for a given prompt is at most proportional to the square root of its within-prompt reward variance RV​(x)\mathrm{RV}(x). In particular, when RV​(x)\mathrm{RV}(x) is small, the task gradient is provably weak.

###### Theorem H.2 (Task gradient magnitude is RV-controlled).

Assume the baseline is the conditional mean b​(x)=𝔼​[R∣X=x]b(x)=\mathbb{E}[R\mid X=x], and gtask​(x):=𝔼​[A​(z;x)​s​(z;x)∣X=x]g\_{\mathrm{task}}(x):=\mathbb{E}[A(z;x)\,s(z;x)\mid X=x]. Then

|  |  |  |
| --- | --- | --- |
|  | ‖gtask​(x)‖≤RV​(x)​𝔼​[‖s​(z;x)‖2∣X=x].\|g\_{\mathrm{task}}(x)\|\;\leq\;\sqrt{\mathrm{RV}(x)}\,\sqrt{\mathbb{E}[\|s(z;x)\|^{2}\mid X=x]}. |  |

###### Proof.

Fix a prompt xx and take randomness over z∼πθ(⋅∣x)z\sim\pi\_{\theta}(\cdot\mid x).
For brevity write A:=A​(z;x)A:=A(z;x) and s:=s​(z;x)s:=s(z;x).
Then

|  |  |  |
| --- | --- | --- |
|  | gtask​(x)=𝔼​[A​s∣X=x].g\_{\mathrm{task}}(x)=\mathbb{E}[A\,s\mid X=x]. |  |

For any unit vector u∈ℝdu\in\mathbb{R}^{d} with ‖u‖=1\|u\|=1,

|  |  |  |
| --- | --- | --- |
|  | |⟨u,gtask(x)⟩|=|𝔼[A⟨u,s⟩∣X=x]|≤𝔼​[A2∣X=x]𝔼​[⟨u,s⟩2∣X=x],\big|\langle u,g\_{\mathrm{task}}(x)\rangle\big|=\big|\mathbb{E}[A\,\langle u,s\rangle\mid X=x]\big|\leq\sqrt{\mathbb{E}[A^{2}\mid X=x]}\;\sqrt{\mathbb{E}[\langle u,s\rangle^{2}\mid X=x]}, |  |

where the inequality is Cauchy-Schwarz.
Moreover, ⟨u,s⟩2≤‖u‖2​‖s‖2=‖s‖2\langle u,s\rangle^{2}\leq\|u\|^{2}\|s\|^{2}=\|s\|^{2}, hence

|  |  |  |
| --- | --- | --- |
|  | |⟨u,gtask​(x)⟩|≤𝔼​[A2∣X=x]​𝔼​[‖s‖2∣X=x].\big|\langle u,g\_{\mathrm{task}}(x)\rangle\big|\leq\sqrt{\mathbb{E}[A^{2}\mid X=x]}\;\sqrt{\mathbb{E}[\|s\|^{2}\mid X=x]}. |  |

Taking the supremum over all unit vectors uu yields

|  |  |  |
| --- | --- | --- |
|  | ‖gtask​(x)‖≤𝔼​[A2∣X=x]​𝔼​[‖s‖2∣X=x].\|g\_{\mathrm{task}}(x)\|\leq\sqrt{\mathbb{E}[A^{2}\mid X=x]}\;\sqrt{\mathbb{E}[\|s\|^{2}\mid X=x]}. |  |

Finally, with b​(x)=𝔼​[R∣X=x]b(x)=\mathbb{E}[R\mid X=x] we have 𝔼​[A∣X=x]=0\mathbb{E}[A\mid X=x]=0 and thus

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[A2∣X=x]=Var​(R∣X=x)=RV​(x).\mathbb{E}[A^{2}\mid X=x]=\mathrm{Var}(R\mid X=x)=\mathrm{RV}(x). |  |

Substituting completes the proof.
∎

### H.4 SNR is upper bounded by RV and reward noise

The following theorem shows that the signal-to-noise ratio of the GG-sample Monte Carlo gradient estimator is upper-bounded by G⋅RV​(x)/σ​(x)\sqrt{G}\cdot\sqrt{\mathrm{RV}(x)}/\sigma(x). When reward variance is low relative to reward noise, the estimator is dominated by noise.

###### Theorem H.3 (SNR upper bound by RV and noise).

Let g^task​(x)\widehat{g}\_{\mathrm{task}}(x) be the GG-sample Monte Carlo estimator

|  |  |  |
| --- | --- | --- |
|  | g^task​(x):=1G​∑k=1GAk​sk,Ak:=A​(zk;x),sk:=s​(zk;x),\widehat{g}\_{\mathrm{task}}(x):=\frac{1}{G}\sum\_{k=1}^{G}A\_{k}\,s\_{k},\qquad A\_{k}:=A(z\_{k};x),\ s\_{k}:=s(z\_{k};x), |  |

with z1,…,zG∼i.i.d.πθ(⋅∣x)z\_{1},\dots,z\_{G}\stackrel{{\scriptstyle\text{i.i.d.}}}{{\sim}}\pi\_{\theta}(\cdot\mid x).
Define

|  |  |  |
| --- | --- | --- |
|  | SNR​(x):=‖gtask​(x)‖𝔼​[‖g^task​(x)−gtask​(x)‖2∣X=x].\mathrm{SNR}(x):=\frac{\|g\_{\mathrm{task}}(x)\|}{\sqrt{\mathbb{E}\big[\|\widehat{g}\_{\mathrm{task}}(x)-g\_{\mathrm{task}}(x)\|^{2}\mid X=x\big]}}. |  |

Under Assumption [H.1](#A8.Thmtheorem1 "Assumption H.1 (Reward decomposition). ‣ H.2 Assumption ‣ Appendix H RV Controls Task-Signal Magnitude and SNR ‣ RAGEN-2: Reasoning Collapse in Agentic RL") and with baseline b​(x)=𝔼​[R∣X=x]b(x)=\mathbb{E}[R\mid X=x],

|  |  |  |
| --- | --- | --- |
|  | SNR​(x)≤G⋅RV​(x)σ​(x).\mathrm{SNR}(x)\;\leq\;\sqrt{G}\cdot\frac{\sqrt{\mathrm{RV}(x)}}{\sigma(x)}. |  |

If σ​(x)=0\sigma(x)=0, the bound is vacuous.

###### Proof.

Fix a prompt xx. Let z1,…,zG∼i.i.d.πθ(⋅∣x)z\_{1},\dots,z\_{G}\stackrel{{\scriptstyle\mathrm{i.i.d.}}}{{\sim}}\pi\_{\theta}(\cdot\mid x) and write

|  |  |  |
| --- | --- | --- |
|  | g^=1G​∑k=1GAk​sk,g=𝔼​[A​s∣x],\widehat{g}\;=\;\frac{1}{G}\sum\_{k=1}^{G}A\_{k}s\_{k},\qquad g\;=\;\mathbb{E}[As\mid x], |  |

where (Ak,sk)=(A​(zk;x),s​(zk;x))(A\_{k},s\_{k})=(A(z\_{k};x),s(z\_{k};x)) and (A,s)=(A​(z;x),s​(z;x))(A,s)=(A(z;x),s(z;x)) for z∼πθ(⋅∣x)z\sim\pi\_{\theta}(\cdot\mid x).

Let Yk:=Ak​skY\_{k}:=A\_{k}s\_{k}. Then g^=1G​∑k=1GYk\widehat{g}=\frac{1}{G}\sum\_{k=1}^{G}Y\_{k} and g=𝔼​[Y1∣x]g=\mathbb{E}[Y\_{1}\mid x], hence

|  |  |  |
| --- | --- | --- |
|  | g^−g=1G​∑k=1G(Yk−g).\widehat{g}-g=\frac{1}{G}\sum\_{k=1}^{G}(Y\_{k}-g). |  |

Using i.i.d. conditional on xx,

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[‖g^−g‖2∣x]\displaystyle\mathbb{E}\!\left[\|\widehat{g}-g\|^{2}\mid x\right] | =1G2​𝔼​[‖∑k=1G(Yk−g)‖2|x]\displaystyle=\frac{1}{G^{2}}\mathbb{E}\!\left[\left\|\sum\_{k=1}^{G}(Y\_{k}-g)\right\|^{2}\Bigm|x\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1G2​∑k=1G𝔼​[‖Yk−g‖2∣x]+1G2​∑k≠ℓ𝔼​[⟨Yk−g,Yℓ−g⟩∣x]\displaystyle=\frac{1}{G^{2}}\sum\_{k=1}^{G}\mathbb{E}\!\left[\|Y\_{k}-g\|^{2}\mid x\right]+\frac{1}{G^{2}}\sum\_{k\neq\ell}\mathbb{E}\!\left[\langle Y\_{k}-g,\;Y\_{\ell}-g\rangle\mid x\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1G2​∑k=1G𝔼​[‖Yk−g‖2∣x]\displaystyle=\frac{1}{G^{2}}\sum\_{k=1}^{G}\mathbb{E}\!\left[\|Y\_{k}-g\|^{2}\mid x\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1G​𝔼​[‖A​s−g‖2∣x].\displaystyle=\frac{1}{G}\,\mathbb{E}\!\left[\|As-g\|^{2}\mid x\right]. |  |

Under Assumption [H.1](#A8.Thmtheorem1 "Assumption H.1 (Reward decomposition). ‣ H.2 Assumption ‣ Appendix H RV Controls Task-Signal Magnitude and SNR ‣ RAGEN-2: Reasoning Collapse in Agentic RL") and with baseline b​(x)=𝔼​[R∣X=x]b(x)=\mathbb{E}[R\mid X=x], write R=μ+εR=\mu+\varepsilon with μ​(x,z)=𝔼​[R∣x,z]\mu(x,z)=\mathbb{E}[R\mid x,z]. Since b​(x)=𝔼​[R∣x]=𝔼​[μ∣x]b(x)=\mathbb{E}[R\mid x]=\mathbb{E}[\mu\mid x],

|  |  |  |
| --- | --- | --- |
|  | A=R−b(x)=(μ−𝔼[μ∣x])+ε=:Aμ+ε.A\;=\;R-b(x)\;=\;(\mu-\mathbb{E}[\mu\mid x])+\varepsilon\;=:\;A\_{\mu}+\varepsilon. |  |

Using A=Aμ+εA=A\_{\mu}+\varepsilon,

|  |  |  |
| --- | --- | --- |
|  | A​s−g=(Aμ​s−g)+ε​s,As-g=(A\_{\mu}s-g)+\varepsilon s, |  |

so

|  |  |  |
| --- | --- | --- |
|  | ‖A​s−g‖2=‖Aμ​s−g‖2+‖ε​s‖2+2​⟨Aμ​s−g,ε​s⟩.\|As-g\|^{2}=\|A\_{\mu}s-g\|^{2}+\|\varepsilon s\|^{2}+2\langle A\_{\mu}s-g,\;\varepsilon s\rangle. |  |

Moreover,

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[⟨Aμ​s−g,ε​s⟩∣x]=𝔼​[𝔼​[⟨Aμ​s−g,ε​s⟩∣x,z]|x]=𝔼​[⟨Aμ​s−g,s⟩​𝔼​[ε∣x,z]|x]=0,\mathbb{E}\!\left[\langle A\_{\mu}s-g,\;\varepsilon s\rangle\mid x\right]=\mathbb{E}\!\left[\mathbb{E}\!\left[\langle A\_{\mu}s-g,\;\varepsilon s\rangle\mid x,z\right]\Bigm|x\right]=\mathbb{E}\!\left[\langle A\_{\mu}s-g,\;s\rangle\,\mathbb{E}[\varepsilon\mid x,z]\Bigm|x\right]=0, |  |

hence

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[‖A​s−g‖2∣x]≥𝔼​[‖ε​s‖2∣x].\mathbb{E}\!\left[\|As-g\|^{2}\mid x\right]\;\geq\;\mathbb{E}\!\left[\|\varepsilon s\|^{2}\mid x\right]. |  |

Combining with the variance decomposition above,

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[‖g^−g‖2∣x]≥1G​𝔼​[‖ε​s‖2∣x].\mathbb{E}\!\left[\|\widehat{g}-g\|^{2}\mid x\right]\;\geq\;\frac{1}{G}\,\mathbb{E}\!\left[\|\varepsilon s\|^{2}\mid x\right]. |  |

Since ‖ε​s‖2=ε2​‖s‖2\|\varepsilon s\|^{2}=\varepsilon^{2}\|s\|^{2} and ss is measurable given (x,z)(x,z),

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[‖ε​s‖2∣x]\displaystyle\mathbb{E}\!\left[\|\varepsilon s\|^{2}\mid x\right] | =𝔼​[𝔼​[ε2​‖s‖2∣x,z]|x]\displaystyle=\mathbb{E}\!\left[\mathbb{E}\!\left[\varepsilon^{2}\|s\|^{2}\mid x,z\right]\Bigm|x\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =𝔼​[‖s‖2​𝔼​[ε2∣x,z]|x]\displaystyle=\mathbb{E}\!\left[\|s\|^{2}\,\mathbb{E}\!\left[\varepsilon^{2}\mid x,z\right]\Bigm|x\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =𝔼​[‖s‖2​σ2​(x)|x]=σ2​(x)​𝔼​[‖s‖2∣x],\displaystyle=\mathbb{E}\!\left[\|s\|^{2}\,\sigma^{2}(x)\Bigm|x\right]=\sigma^{2}(x)\,\mathbb{E}\!\left[\|s\|^{2}\mid x\right], |  |

where 𝔼​[ε2∣x,z]=Var​(ε∣x,z)=σ2​(x)\mathbb{E}[\varepsilon^{2}\mid x,z]=\mathrm{Var}(\varepsilon\mid x,z)=\sigma^{2}(x) by Assumption [H.1](#A8.Thmtheorem1 "Assumption H.1 (Reward decomposition). ‣ H.2 Assumption ‣ Appendix H RV Controls Task-Signal Magnitude and SNR ‣ RAGEN-2: Reasoning Collapse in Agentic RL").
Therefore

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[‖g^−g‖2∣x]≥1G​σ2​(x)​𝔼​[‖s‖2∣x].\mathbb{E}\!\left[\|\widehat{g}-g\|^{2}\mid x\right]\;\geq\;\frac{1}{G}\,\sigma^{2}(x)\,\mathbb{E}\!\left[\|s\|^{2}\mid x\right]. |  |

By Theorem [H.2](#A8.Thmtheorem2 "Theorem H.2 (Task gradient magnitude is RV-controlled). ‣ H.3 Task-gradient magnitude is RV-controlled ‣ Appendix H RV Controls Task-Signal Magnitude and SNR ‣ RAGEN-2: Reasoning Collapse in Agentic RL"),

|  |  |  |
| --- | --- | --- |
|  | ∥g∥=∥𝔼[As∣x]∥≤RV​(x)𝔼​[‖s‖2∣x].\|g\|=\left\|\mathbb{E}[As\mid x]\right\|\leq\sqrt{\mathrm{RV}(x)}\;\sqrt{\mathbb{E}\!\left[\|s\|^{2}\mid x\right]}. |  |

Thus, with SNR​(x):=‖g‖𝔼​[‖g^−g‖2∣x]\mathrm{SNR}(x):=\frac{\|g\|}{\sqrt{\mathbb{E}[\|\widehat{g}-g\|^{2}\mid x]}},

|  |  |  |
| --- | --- | --- |
|  | SNR​(x)≤RV​(x)​𝔼​[‖s‖2∣x]1G​σ2​(x)​𝔼​[‖s‖2∣x]=G⋅RV​(x)σ​(x).∎\mathrm{SNR}(x)\leq\frac{\sqrt{\mathrm{RV}(x)}\sqrt{\mathbb{E}[\|s\|^{2}\mid x]}}{\sqrt{\frac{1}{G}\sigma^{2}(x)\mathbb{E}[\|s\|^{2}\mid x]}}=\sqrt{G}\cdot\frac{\sqrt{\mathrm{RV}(x)}}{\sigma(x)}.\qed |  |

### H.5 Low-SNR updates induce parameter drift

When updates carry no directional signal (zero mean), the parameter drifts away from initialization at a rate linear in the number of steps. This illustrates why sustained low-SNR updates are harmful even if they do not systematically push in a wrong direction.

###### Theorem H.4 (Illustrative random-walk drift under zero-mean noise).

Consider SGD-style updates

|  |  |  |
| --- | --- | --- |
|  | θt+1=θt+η​ξt,\theta\_{t+1}=\theta\_{t}+\eta\,\xi\_{t}, |  |

where {ξt}t≥0\{\xi\_{t}\}\_{t\geq 0} are independent, 𝔼​[ξt]=0\mathbb{E}[\xi\_{t}]=0, and 𝔼​[‖ξt‖2]=v<∞\mathbb{E}[\|\xi\_{t}\|^{2}]=v<\infty for all tt.
Then for any T≥1T\geq 1,

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[‖θT−θ0‖2]=η2​T​v.\mathbb{E}\big[\|\theta\_{T}-\theta\_{0}\|^{2}\big]=\eta^{2}\,T\,v. |  |

###### Proof.

Unrolling the recursion yields

|  |  |  |
| --- | --- | --- |
|  | θT−θ0=η​∑t=0T−1ξt.\theta\_{T}-\theta\_{0}=\eta\sum\_{t=0}^{T-1}\xi\_{t}. |  |

Therefore,

|  |  |  |
| --- | --- | --- |
|  | ‖θT−θ0‖2=η2​‖∑t=0T−1ξt‖2=η2​(∑t=0T−1‖ξt‖2+2​∑0≤i<j≤T−1⟨ξi,ξj⟩).\|\theta\_{T}-\theta\_{0}\|^{2}=\eta^{2}\left\|\sum\_{t=0}^{T-1}\xi\_{t}\right\|^{2}=\eta^{2}\left(\sum\_{t=0}^{T-1}\|\xi\_{t}\|^{2}+2\sum\_{0\leq i<j\leq T-1}\langle\xi\_{i},\xi\_{j}\rangle\right). |  |

Taking expectation and using independence with 𝔼​[ξt]=0\mathbb{E}[\xi\_{t}]=0,

|  |  |  |
| --- | --- | --- |
|  | 𝔼​⟨ξi,ξj⟩=⟨𝔼​[ξi],𝔼​[ξj]⟩=0,i≠j.\mathbb{E}\langle\xi\_{i},\xi\_{j}\rangle=\left\langle\mathbb{E}[\xi\_{i}],\,\mathbb{E}[\xi\_{j}]\right\rangle=0,\qquad i\neq j. |  |

Hence the cross terms vanish and

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[‖θT−θ0‖2]=η2​∑t=0T−1𝔼​[‖ξt‖2]=η2​T​v,\mathbb{E}\big[\|\theta\_{T}-\theta\_{0}\|^{2}\big]=\eta^{2}\sum\_{t=0}^{T-1}\mathbb{E}[\|\xi\_{t}\|^{2}]=\eta^{2}\,T\,v, |  |

where we used 𝔼​[‖ξt‖2]=v\mathbb{E}[\|\xi\_{t}\|^{2}]=v for all tt.
∎

## Appendix I Template Mixing Reduces Input Dependence

If the policy’s conditional distribution is contaminated by a prompt-independent component q​(z)q(z) with mixing weight α\alpha, the resulting mutual information Iα​(X;Z)I\_{\alpha}(X;Z) contracts by at least a factor of (1−α)(1-\alpha). This formalizes the intuition that even partial drift toward a shared template erodes input dependence.

###### Lemma I.1 (Template mixing contracts mutual information).

Let X∼P​(X)X\sim P(X) and Z∣X=x∼p​(z∣x)Z\mid X=x\sim p(z\mid x) with marginal p​(z)=𝔼x∼P​[p​(z∣x)]p(z)=\mathbb{E}\_{x\sim P}[p(z\mid x)].
Fix any prompt-independent distribution q​(z)q(z). For α∈[0,1]\alpha\in[0,1], define the mixed conditional and marginal

|  |  |  |
| --- | --- | --- |
|  | pα​(z∣x):=(1−α)​p​(z∣x)+α​q​(z),pα​(z):=(1−α)​p​(z)+α​q​(z).p\_{\alpha}(z\mid x):=(1-\alpha)p(z\mid x)+\alpha q(z),\qquad p\_{\alpha}(z):=(1-\alpha)p(z)+\alpha q(z). |  |

Let Iα​(X;Z)I\_{\alpha}(X;Z) denote the mutual information under pα​(x,z)=P​(x)​pα​(z∣x)p\_{\alpha}(x,z)=P(x)p\_{\alpha}(z\mid x). Then

|  |  |  |
| --- | --- | --- |
|  | Iα​(X;Z)≤(1−α)​I​(X;Z).I\_{\alpha}(X;Z)\;\leq\;(1-\alpha)\,I(X;Z). |  |

###### Proof.

For any fixed xx,

|  |  |  |  |
| --- | --- | --- | --- |
|  | KL(pα(⋅∣x)∥pα(⋅))\displaystyle\mathrm{KL}\!\big(p\_{\alpha}(\cdot\mid x)\,\|\,p\_{\alpha}(\cdot)\big) | =𝔼z∼pα(⋅∣x)​[log⁡pα​(z∣x)pα​(z)]\displaystyle=\mathbb{E}\_{z\sim p\_{\alpha}(\cdot\mid x)}\left[\log\frac{p\_{\alpha}(z\mid x)}{p\_{\alpha}(z)}\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =𝔼z∼pα(⋅∣x)​[log⁡pα​(z∣x)−log⁡pα​(z)].\displaystyle=\mathbb{E}\_{z\sim p\_{\alpha}(\cdot\mid x)}\big[\log p\_{\alpha}(z\mid x)-\log p\_{\alpha}(z)\big]. |  |

Taking expectation over x∼P​(x)x\sim P(x) gives

|  |  |  |
| --- | --- | --- |
|  | Iα(X;Z)=𝔼x[KL(pα(⋅∣x)∥pα(⋅))].I\_{\alpha}(X;Z)=\mathbb{E}\_{x}\Big[\mathrm{KL}\!\big(p\_{\alpha}(\cdot\mid x)\,\|\,p\_{\alpha}(\cdot)\big)\Big]. |  |

The same identity holds for I​(X;Z)I(X;Z) with pαp\_{\alpha} replaced by pp.

By joint convexity of KL(⋅∥⋅)\mathrm{KL}(\cdot\|\cdot) (coverthomas2006elements, Theorem 2.7.2), for any distributions a,b,c,da,b,c,d and any α∈[0,1]\alpha\in[0,1],

|  |  |  |
| --- | --- | --- |
|  | KL​((1−α)​a+α​b∥(1−α)​c+α​d)≤(1−α)​KL​(a∥c)+α​KL​(b∥d).\mathrm{KL}\!\big((1-\alpha)a+\alpha b\,\|\,(1-\alpha)c+\alpha d\big)\leq(1-\alpha)\mathrm{KL}(a\|c)+\alpha\,\mathrm{KL}(b\|d). |  |

Let a=p(⋅∣x)a=p(\cdot\mid x), b=qb=q, c=p​(⋅)c=p(\cdot), and d=qd=q. Since

|  |  |  |
| --- | --- | --- |
|  | pα(⋅∣x)=(1−α)p(⋅∣x)+αq,pα(⋅)=(1−α)p(⋅)+αq,p\_{\alpha}(\cdot\mid x)=(1-\alpha)p(\cdot\mid x)+\alpha q,\qquad p\_{\alpha}(\cdot)=(1-\alpha)p(\cdot)+\alpha q, |  |

we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | KL(pα(⋅∣x)∥pα(⋅))\displaystyle\mathrm{KL}\!\big(p\_{\alpha}(\cdot\mid x)\,\|\,p\_{\alpha}(\cdot)\big) | ≤(1−α)KL(p(⋅∣x)∥p(⋅))+αKL(q∥q)\displaystyle\leq(1-\alpha)\mathrm{KL}\!\big(p(\cdot\mid x)\,\|\,p(\cdot)\big)+\alpha\,\mathrm{KL}(q\|q) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(1−α)KL(p(⋅∣x)∥p(⋅)).\displaystyle=(1-\alpha)\mathrm{KL}\!\big(p(\cdot\mid x)\,\|\,p(\cdot)\big). |  |

Averaging over x∼P​(x)x\sim P(x) yields

|  |  |  |
| --- | --- | --- |
|  | 𝔼x[KL(pα(⋅∣x)∥pα(⋅))]≤(1−α)𝔼x[KL(p(⋅∣x)∥p(⋅))].\mathbb{E}\_{x}\Big[\mathrm{KL}\!\big(p\_{\alpha}(\cdot\mid x)\,\|\,p\_{\alpha}(\cdot)\big)\Big]\leq(1-\alpha)\,\mathbb{E}\_{x}\Big[\mathrm{KL}\!\big(p(\cdot\mid x)\,\|\,p(\cdot)\big)\Big]. |  |

Using the identity I(X;Z)=𝔼x[KL(p(⋅∣x)∥p(⋅))]I(X;Z)=\mathbb{E}\_{x}\big[\mathrm{KL}(p(\cdot\mid x)\,\|\,p(\cdot))\big] (and the analogous one for IαI\_{\alpha}), we obtain

|  |  |  |
| --- | --- | --- |
|  | Iα​(X;Z)≤(1−α)​I​(X;Z),I\_{\alpha}(X;Z)\leq(1-\alpha)\,I(X;Z), |  |

which proves the lemma.
∎

###### Remark I.2.

The continuity bound f​(ε)f(\varepsilon) depends on log⁡(|𝒳|​|𝒵|)\log(|\mathcal{X}||\mathcal{Z}|), which can be extremely large for LLM token spaces. Therefore, this result should be understood as a qualitative guarantee that KL-closeness implies MI-closeness in principle, rather than a tight quantitative bound in practice.

## Appendix J Filtering Reduces Gradient-Estimation MSE

### J.1 Setup

Consider PP groups indexed by i∈{1,…,P}i\in\{1,\dots,P\}. Group ii contains GG rollouts, and g^i∈ℝd\widehat{g}\_{i}\in\mathbb{R}^{d} denotes the *group-level* gradient estimator (already averaged over the GG rollouts in the group).
We model

|  |  |  |
| --- | --- | --- |
|  | g^i=gi+εi,𝔼​[εi]=0,𝔼​‖εi‖2=σi2,\widehat{g}\_{i}\;=\;g\_{i}+\varepsilon\_{i},\qquad\mathbb{E}[\varepsilon\_{i}]=0,\qquad\mathbb{E}\|\varepsilon\_{i}\|^{2}=\sigma\_{i}^{2}, |  |

where {εi}i=1P\{\varepsilon\_{i}\}\_{i=1}^{P} are independent across groups.
For a kept set SS of groups, we write n:=|S|n:=|S| for the number of kept groups.

### J.2 Unfiltered vs. filtered estimators

Define the unfiltered batch estimator and its mean:

|  |  |  |
| --- | --- | --- |
|  | g¯^:=1P​∑i=1Pg^i,g¯:=1P​∑i=1Pgi.\widehat{\bar{g}}:=\frac{1}{P}\sum\_{i=1}^{P}\widehat{g}\_{i},\qquad\bar{g}:=\frac{1}{P}\sum\_{i=1}^{P}g\_{i}. |  |

Let S⊆{1,…,P}S\subseteq\{1,\dots,P\} be the set of kept groups with |S|=n|S|=n. Define the filtered estimator and its mean:

|  |  |  |
| --- | --- | --- |
|  | g¯^S:=1n​∑i∈Sg^i,g¯S:=1n​∑i∈Sgi.\widehat{\bar{g}}\_{S}:=\frac{1}{n}\sum\_{i\in S}\widehat{g}\_{i},\qquad\bar{g}\_{S}:=\frac{1}{n}\sum\_{i\in S}g\_{i}. |  |

By retaining only a subset of prompt groups, the filtered estimator’s mean-squared error depends solely on the noise variances of the kept groups. Dropping high-noise (low-RV) groups directly lowers the estimation error.

###### Theorem J.1 (MSE of the filtered estimator).

g¯^S\widehat{\bar{g}}\_{S} is unbiased for g¯S\bar{g}\_{S} and satisfies

|  |  |  |
| --- | --- | --- |
|  | 𝔼​‖g¯^S−g¯S‖2=1n2​∑i∈Sσi2.\mathbb{E}\big\|\widehat{\bar{g}}\_{S}-\bar{g}\_{S}\big\|^{2}\;=\;\frac{1}{n^{2}}\sum\_{i\in S}\sigma\_{i}^{2}. |  |

###### Proof.

By the setup, g^i=gi+εi\widehat{g}\_{i}=g\_{i}+\varepsilon\_{i} with 𝔼​[εi]=0\mathbb{E}[\varepsilon\_{i}]=0, hence

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[g^i]=gi.\mathbb{E}[\widehat{g}\_{i}]=g\_{i}. |  |

Therefore,

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[g¯^S]=1n​∑i∈S𝔼​[g^i]=1n​∑i∈Sgi=g¯S.\mathbb{E}[\widehat{\bar{g}}\_{S}]=\frac{1}{n}\sum\_{i\in S}\mathbb{E}[\widehat{g}\_{i}]=\frac{1}{n}\sum\_{i\in S}g\_{i}=\bar{g}\_{S}. |  |

Moreover,

|  |  |  |
| --- | --- | --- |
|  | g¯^S−g¯S=1n​∑i∈S(g^i−gi)=1n​∑i∈Sεi.\widehat{\bar{g}}\_{S}-\bar{g}\_{S}=\frac{1}{n}\sum\_{i\in S}(\widehat{g}\_{i}-g\_{i})=\frac{1}{n}\sum\_{i\in S}\varepsilon\_{i}. |  |

Therefore,

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​‖g¯^S−g¯S‖2\displaystyle\mathbb{E}\big\|\widehat{\bar{g}}\_{S}-\bar{g}\_{S}\big\|^{2} | =1n2​𝔼​‖∑i∈Sεi‖2\displaystyle=\frac{1}{n^{2}}\,\mathbb{E}\left\|\sum\_{i\in S}\varepsilon\_{i}\right\|^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1n2​(∑i∈S𝔼​‖εi‖2+∑i,j∈Si≠j𝔼​⟨εi,εj⟩).\displaystyle=\frac{1}{n^{2}}\left(\sum\_{i\in S}\mathbb{E}\|\varepsilon\_{i}\|^{2}+\sum\_{\begin{subarray}{c}i,j\in S\\ i\neq j\end{subarray}}\mathbb{E}\langle\varepsilon\_{i},\varepsilon\_{j}\rangle\right). |  |

By independence and 𝔼​[εi]=0\mathbb{E}[\varepsilon\_{i}]=0, for i≠ji\neq j we have

|  |  |  |
| --- | --- | --- |
|  | 𝔼​⟨εi,εj⟩=⟨𝔼​[εi],𝔼​[εj]⟩=0,\mathbb{E}\langle\varepsilon\_{i},\varepsilon\_{j}\rangle=\left\langle\mathbb{E}[\varepsilon\_{i}],\,\mathbb{E}[\varepsilon\_{j}]\right\rangle=0, |  |

so the cross terms vanish. Hence

|  |  |  |
| --- | --- | --- |
|  | 𝔼​‖g¯^S−g¯S‖2=1n2​∑i∈S𝔼​‖εi‖2=1n2​∑i∈Sσi2.\mathbb{E}\big\|\widehat{\bar{g}}\_{S}-\bar{g}\_{S}\big\|^{2}=\frac{1}{n^{2}}\sum\_{i\in S}\mathbb{E}\|\varepsilon\_{i}\|^{2}=\frac{1}{n^{2}}\sum\_{i\in S}\sigma\_{i}^{2}. |  |

∎

#### Remark (bias relative to the original objective).

While g¯^S\widehat{\bar{g}}\_{S} is unbiased for the *filtered* mean gradient g¯S\bar{g}\_{S}, it is generally biased for the *unfiltered* mean gradient g¯\bar{g} unless SS is chosen independently of {gi}\{g\_{i}\} or gig\_{i} is constant across groups.

## Appendix K Reward-Agnostic Regularizers and Update Dominance

### K.1 Setup

Similarly, fix a prompt xx and consider trajectories z∼πθ(⋅∣x)z\sim\pi\_{\theta}(\cdot\mid x) with reward R​(z;x)R(z;x) and baseline b​(x)b(x).
Define the reward-driven (task) gradient

|  |  |  |
| --- | --- | --- |
|  | gtask​(x):=𝔼​[(R​(z;x)−b​(x))​s​(z;x)∣X=x],s​(z;x):=∇θlog⁡πθ​(z∣x).g\_{\mathrm{task}}(x)\;:=\;\mathbb{E}\!\left[(R(z;x)-b(x))\,s(z;x)\mid X=x\right],\qquad s(z;x):=\nabla\_{\theta}\log\pi\_{\theta}(z\mid x). |  |

Let greg​(x)g\_{\mathrm{reg}}(x) denote an update component that is computed without multiplying the reward (or advantage), e.g.,

|  |  |  |
| --- | --- | --- |
|  | greg​(x):=λKL​gKL​(x)+λent​gent​(x),g\_{\mathrm{reg}}(x):=\lambda\_{\mathrm{KL}}\,g\_{\mathrm{KL}}(x)+\lambda\_{\mathrm{ent}}\,g\_{\mathrm{ent}}(x), |  |

where gKL​(x)g\_{\mathrm{KL}}(x) and gent​(x)g\_{\mathrm{ent}}(x) are gradients of prompt-level distributional regularizers.
We write the total expected update as

|  |  |  |
| --- | --- | --- |
|  | gtotal​(x)=gtask​(x)+greg​(x).g\_{\mathrm{total}}(x)=g\_{\mathrm{task}}(x)+g\_{\mathrm{reg}}(x). |  |

To summarize relative influence, define the dominance ratio

|  |  |  |
| --- | --- | --- |
|  | ρ​(x):=‖greg​(x)‖‖gtask​(x)‖+‖greg​(x)‖∈[0,1].\rho(x):=\frac{\|g\_{\mathrm{reg}}(x)\|}{\|g\_{\mathrm{task}}(x)\|+\|g\_{\mathrm{reg}}(x)\|}\in[0,1]. |  |

We refer to greg​(x)g\_{\mathrm{reg}}(x) as *reward-agnostic* since it does not use within-prompt reward differences to weight trajectories.

### K.2 Low-RV prompts amplify regularizer influence

When reward variance is small, the task gradient weakens (by Theorem [H.2](#A8.Thmtheorem2 "Theorem H.2 (Task gradient magnitude is RV-controlled). ‣ H.3 Task-gradient magnitude is RV-controlled ‣ Appendix H RV Controls Task-Signal Magnitude and SNR ‣ RAGEN-2: Reasoning Collapse in Agentic RL")) while regularizer gradients remain largely flat across prompts. Consequently, the regularizer’s share of the total update grows on low-RV prompts, formalizing why these prompts are more prone to input-agnostic drift.

By Theorem [H.2](#A8.Thmtheorem2 "Theorem H.2 (Task gradient magnitude is RV-controlled). ‣ H.3 Task-gradient magnitude is RV-controlled ‣ Appendix H RV Controls Task-Signal Magnitude and SNR ‣ RAGEN-2: Reasoning Collapse in Agentic RL"), for any prompt xx,

|  |  |  |
| --- | --- | --- |
|  | ‖gtask​(x)‖≤RV​(x)​𝔼​[‖s‖2∣X=x].\|g\_{\mathrm{task}}(x)\|\leq\sqrt{\mathrm{RV}(x)}\;\sqrt{\mathbb{E}[\|s\|^{2}\mid X=x]}. |  |

Therefore the dominance ratio

|  |  |  |
| --- | --- | --- |
|  | ρ​(x)=‖greg​(x)‖‖gtask​(x)‖+‖greg​(x)‖\rho(x)=\frac{\|g\_{\mathrm{reg}}(x)\|}{\|g\_{\mathrm{task}}(x)\|+\|g\_{\mathrm{reg}}(x)\|} |  |

admits the lower bound

|  |  |  |
| --- | --- | --- |
|  | ρ​(x)≥‖greg​(x)‖‖greg​(x)‖+RV​(x)​𝔼​[‖s‖2∣X=x].\rho(x)\;\geq\;\frac{\|g\_{\mathrm{reg}}(x)\|}{\|g\_{\mathrm{reg}}(x)\|+\sqrt{\mathrm{RV}(x)}\sqrt{\mathbb{E}[\|s\|^{2}\mid X=x]}}. |  |

In particular, if ‖greg​(x)‖\|g\_{\mathrm{reg}}(x)\| and 𝔼​[‖s‖2∣X=x]\mathbb{E}[\|s\|^{2}\mid X=x] vary slowly across prompts compared to RV​(x)\mathrm{RV}(x), then smaller RV​(x)\mathrm{RV}(x) implies larger ρ​(x)\rho(x), i.e., the total update is more strongly shaped by reward-agnostic regularizers on low-RV\mathrm{RV} prompts.

## Appendix L KL-Closeness to the Base Implies MI-Closeness

If the current policy stays uniformly close to a reference policy in KL divergence, then the mutual information I​(X;Z)I(X;Z) between inputs and reasoning also remains close. This means strong KL constraints preserve—but do not necessarily increase—input dependence.

###### Theorem L.1.

To avoid measure-theoretic issues, assume XX is supported on a finite set 𝒳\mathcal{X} and ZZ takes values in a finite set 𝒵\mathcal{Z}.
Let P​(X)P(X) be the prompt distribution and define

|  |  |  |
| --- | --- | --- |
|  | Pθ​(x,z):=P​(x)​πθ​(z∣x),P0​(x,z):=P​(x)​π0​(z∣x).P\_{\theta}(x,z):=P(x)\pi\_{\theta}(z\mid x),\qquad P\_{0}(x,z):=P(x)\pi\_{0}(z\mid x). |  |

If

|  |  |  |
| --- | --- | --- |
|  | supx∈𝒳KL(πθ(⋅∣x)∥π0(⋅∣x))≤ε,\sup\_{x\in\mathcal{X}}\mathrm{KL}\!\left(\pi\_{\theta}(\cdot\mid x)\,\|\,\pi\_{0}(\cdot\mid x)\right)\leq\varepsilon, |  |

then there exists f​(ε)→0f(\varepsilon)\to 0 as ε→0\varepsilon\to 0 such that

|  |  |  |
| --- | --- | --- |
|  | |Iθ​(X;Z)−I0​(X;Z)|≤f​(ε).\big|I\_{\theta}(X;Z)-I\_{0}(X;Z)\big|\leq f(\varepsilon). |  |

###### Proof.

By the chain rule for KL divergence,

|  |  |  |
| --- | --- | --- |
|  | KL(Pθ(X,Z)∥P0(X,Z))=𝔼x∼P[KL(πθ(⋅∣x)∥π0(⋅∣x))].\mathrm{KL}\!\left(P\_{\theta}(X,Z)\,\|\,P\_{0}(X,Z)\right)=\mathbb{E}\_{x\sim P}\!\left[\mathrm{KL}\!\left(\pi\_{\theta}(\cdot\mid x)\,\|\,\pi\_{0}(\cdot\mid x)\right)\right]. |  |

Under the assumption supx∈𝒳KL(πθ(⋅∣x)∥π0(⋅∣x))≤ε\sup\_{x\in\mathcal{X}}\mathrm{KL}\!\left(\pi\_{\theta}(\cdot\mid x)\,\|\,\pi\_{0}(\cdot\mid x)\right)\leq\varepsilon, we obtain

|  |  |  |
| --- | --- | --- |
|  | KL​(Pθ​(X,Z)∥P0​(X,Z))≤ε.\mathrm{KL}\!\left(P\_{\theta}(X,Z)\,\|\,P\_{0}(X,Z)\right)\leq\varepsilon. |  |

By Pinsker’s inequality,

|  |  |  |
| --- | --- | --- |
|  | ∥Pθ(X,Z)−P0(X,Z)∥TV≤12​KL​(Pθ​(X,Z)∥P0​(X,Z))≤ε2=:δ.\|P\_{\theta}(X,Z)-P\_{0}(X,Z)\|\_{\mathrm{TV}}\leq\sqrt{\tfrac{1}{2}\,\mathrm{KL}\!\left(P\_{\theta}(X,Z)\,\|\,P\_{0}(X,Z)\right)}\leq\sqrt{\tfrac{\varepsilon}{2}}=:\delta. |  |

Since ‖Pθ​(X,Z)−P0​(X,Z)‖TV≤δ\|P\_{\theta}(X,Z)-P\_{0}(X,Z)\|\_{\mathrm{TV}}\leq\delta and (X,Z)(X,Z) takes values in a finite alphabet 𝒳×𝒵\mathcal{X}\times\mathcal{Z}, the Fannes-Audenaert inequality implies

|  |  |  |
| --- | --- | --- |
|  | |Hθ​(X,Z)−H0​(X,Z)|≤δ​log⁡(|𝒳|​|𝒵|−1)+h2​(δ),\big|H\_{\theta}(X,Z)-H\_{0}(X,Z)\big|\leq\delta\log(|\mathcal{X}||\mathcal{Z}|-1)+h\_{2}(\delta), |  |

where Hθ​(⋅)H\_{\theta}(\cdot) denotes entropy under PθP\_{\theta}, and h2​(⋅)h\_{2}(\cdot) is the binary entropy.
Moreover, total variation does not increase under marginalization, so

|  |  |  |
| --- | --- | --- |
|  | ‖Pθ​(Z)−P0​(Z)‖TV≤δ,\|P\_{\theta}(Z)-P\_{0}(Z)\|\_{\mathrm{TV}}\leq\delta, |  |

and applying Fannes-Audenaert on the alphabet 𝒵\mathcal{Z} yields

|  |  |  |
| --- | --- | --- |
|  | |Hθ​(Z)−H0​(Z)|≤δ​log⁡(|𝒵|−1)+h2​(δ)≤δ​log⁡(|𝒳|​|𝒵|−1)+h2​(δ).\big|H\_{\theta}(Z)-H\_{0}(Z)\big|\leq\delta\log(|\mathcal{Z}|-1)+h\_{2}(\delta)\leq\delta\log(|\mathcal{X}||\mathcal{Z}|-1)+h\_{2}(\delta). |  |

Finally, using I​(X;Z)=H​(X)+H​(Z)−H​(X,Z)I(X;Z)=H(X)+H(Z)-H(X,Z) and noting that Pθ​(X)=P0​(X)=P​(X)P\_{\theta}(X)=P\_{0}(X)=P(X) (hence Hθ​(X)=H0​(X)H\_{\theta}(X)=H\_{0}(X)),

|  |  |  |  |
| --- | --- | --- | --- |
|  | |Iθ​(X;Z)−I0​(X;Z)|\displaystyle\big|I\_{\theta}(X;Z)-I\_{0}(X;Z)\big| | =|(Hθ​(Z)−H0​(Z))−(Hθ​(X,Z)−H0​(X,Z))|\displaystyle=\big|(H\_{\theta}(Z)-H\_{0}(Z))-(H\_{\theta}(X,Z)-H\_{0}(X,Z))\big| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤|Hθ​(Z)−H0​(Z)|+|Hθ​(X,Z)−H0​(X,Z)|\displaystyle\leq\big|H\_{\theta}(Z)-H\_{0}(Z)\big|+\big|H\_{\theta}(X,Z)-H\_{0}(X,Z)\big| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤2​(δ​log⁡(|𝒳|​|𝒵|−1)+h2​(δ)).\displaystyle\leq 2\Big(\delta\log(|\mathcal{X}||\mathcal{Z}|-1)+h\_{2}(\delta)\Big). |  |

Thus we may take

|  |  |  |
| --- | --- | --- |
|  | f​(ε):=2​(δ​log⁡(|𝒳|​|𝒵|−1)+h2​(δ)),δ:=ε2,f(\varepsilon):=2\Big(\delta\log(|\mathcal{X}||\mathcal{Z}|-1)+h\_{2}(\delta)\Big),\qquad\delta:=\sqrt{\tfrac{\varepsilon}{2}}, |  |

which satisfies f​(ε)→0f(\varepsilon)\to 0 as ε→0\varepsilon\to 0.
∎

## Appendix M Decomposing Changes in Input Dependence

###### Definition M.1 (Entropy changes).

Let XX be prompts and let Z∼πθ(⋅∣X)Z\sim\pi\_{\theta}(\cdot\mid X) under the current policy, with reference policy π0\pi\_{0}.
Define the conditional-entropy and marginal-entropy changes

|  |  |  |
| --- | --- | --- |
|  | Δin:=Hθ​(Z∣X)−H0​(Z∣X),Δmarg:=Hθ​(Z)−H0​(Z).\Delta\_{\mathrm{in}}:=H\_{\theta}(Z\mid X)-H\_{0}(Z\mid X),\qquad\Delta\_{\mathrm{marg}}:=H\_{\theta}(Z)-H\_{0}(Z). |  |

The change in mutual information decomposes as Δ​I=Δmarg−Δin\Delta I=\Delta\_{\mathrm{marg}}-\Delta\_{\mathrm{in}}. If an intervention (e.g., an entropy bonus) increases within-prompt variability H​(Z∣X)H(Z\mid X) more than it increases the marginal diversity H​(Z)H(Z), input dependence necessarily decreases.

###### Theorem M.2.

With Δin\Delta\_{\mathrm{in}} and Δmarg\Delta\_{\mathrm{marg}} defined above,

|  |  |  |
| --- | --- | --- |
|  | Iθ​(X;Z)−I0​(X;Z)=Δmarg−Δin.I\_{\theta}(X;Z)-I\_{0}(X;Z)=\Delta\_{\mathrm{marg}}-\Delta\_{\mathrm{in}}. |  |

In particular, if Δin≥Δmarg+γ\Delta\_{\mathrm{in}}\geq\Delta\_{\mathrm{marg}}+\gamma for some γ>0\gamma>0, then

|  |  |  |
| --- | --- | --- |
|  | Iθ​(X;Z)≤I0​(X;Z)−γ,I\_{\theta}(X;Z)\leq I\_{0}(X;Z)-\gamma, |  |

and especially Iθ​(X;Z)<I0​(X;Z)I\_{\theta}(X;Z)<I\_{0}(X;Z) whenever Δin>Δmarg\Delta\_{\mathrm{in}}>\Delta\_{\mathrm{marg}}.

###### Proof.

Using I​(X;Z)=H​(Z)−H​(Z∣X)I(X;Z)=H(Z)-H(Z\mid X),

|  |  |  |  |
| --- | --- | --- | --- |
|  | Iθ​(X;Z)−I0​(X;Z)\displaystyle I\_{\theta}(X;Z)-I\_{0}(X;Z) | =(Hθ​(Z)−H0​(Z))−(Hθ​(Z∣X)−H0​(Z∣X))\displaystyle=\big(H\_{\theta}(Z)-H\_{0}(Z)\big)-\big(H\_{\theta}(Z\mid X)-H\_{0}(Z\mid X)\big) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =Δmarg−Δin.\displaystyle=\Delta\_{\mathrm{marg}}-\Delta\_{\mathrm{in}}. |  |

The sufficient-condition statements follow by rearranging the inequality.
∎

An entropy bonus acts directly on the per-prompt dispersion and increases Hθ​(Z∣X)H\_{\theta}(Z\mid X), but it does not explicitly encourage cross-prompt separation that would increase the marginal entropy Hθ​(Z)H\_{\theta}(Z) by a comparable amount. Hence it is plausible that Δin\Delta\_{\mathrm{in}} exceeds Δmarg\Delta\_{\mathrm{marg}}, in which case Theorem [M.2](#A13.Thmtheorem2 "Theorem M.2. ‣ Appendix M Decomposing Changes in Input Dependence ‣ RAGEN-2: Reasoning Collapse in Agentic RL") implies Iθ​(X;Z)I\_{\theta}(X;Z) decreases.

Appendix [K](#A11 "Appendix K Reward-Agnostic Regularizers and Update Dominance ‣ RAGEN-2: Reasoning Collapse in Agentic RL") explains that when RV​(x)\mathrm{RV}(x) is small, the task update can be weak, so reward-agnostic regularizers can have larger relative influence on the total update.

## Appendix N GRPO Normalization Amplifies Noise at Low RV

GRPO-style normalization divides the advantage by RV​(x)\sqrt{\mathrm{RV}(x)}, which induces a RV​(x)−1\mathrm{RV}(x)^{-1} noise amplification in the mean-squared error of the per-prompt gradient estimator.

For a fixed prompt xx, define the normalized advantage

|  |  |  |
| --- | --- | --- |
|  | A~​(z;x):=A​(z;x)RV​(x),A​(z;x):=R​(z;x)−b​(x),b​(x):=𝔼z∼πθ(⋅∣x)​[R​(z;x)].\widetilde{A}(z;x):=\frac{A(z;x)}{\sqrt{\mathrm{RV}(x)}},\qquad A(z;x):=R(z;x)-b(x),\qquad b(x):=\mathbb{E}\_{z\sim\pi\_{\theta}(\cdot\mid x)}[R(z;x)]. |  |

Given KK i.i.d. rollouts z1,…,zK∼πθ(⋅∣x)z\_{1},\dots,z\_{K}\sim\pi\_{\theta}(\cdot\mid x), define

|  |  |  |
| --- | --- | --- |
|  | g^GRPO​(x):=1K​∑k=1KA~k​sk,gGRPO​(x):=𝔼​[A~​s∣X=x],\widehat{g}\_{\mathrm{GRPO}}(x):=\frac{1}{K}\sum\_{k=1}^{K}\widetilde{A}\_{k}\,s\_{k},\qquad g\_{\mathrm{GRPO}}(x):=\mathbb{E}[\widetilde{A}\,s\mid X=x], |  |

where sk=∇θlog⁡πθ​(zk∣x)s\_{k}=\nabla\_{\theta}\log\pi\_{\theta}(z\_{k}\mid x).

Dividing the advantage by RV​(x)\sqrt{\mathrm{RV}(x)} causes the gradient estimator’s variance floor to scale as RV​(x)−1\mathrm{RV}(x)^{-1}, so prompts with small reward variance suffer disproportionately noisy updates under GRPO-style normalization.

###### Proposition N.1 (GRPO variance floor).

Under Assumption [H.1](#A8.Thmtheorem1 "Assumption H.1 (Reward decomposition). ‣ H.2 Assumption ‣ Appendix H RV Controls Task-Signal Magnitude and SNR ‣ RAGEN-2: Reasoning Collapse in Agentic RL"), the GRPO estimator satisfies

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[‖g^GRPO​(x)−gGRPO​(x)‖2∣X=x]≥1K⋅σ2​(x)RV​(x)​𝔼​[‖s‖2∣X=x].\mathbb{E}\!\left[\big\|\widehat{g}\_{\mathrm{GRPO}}(x)-g\_{\mathrm{GRPO}}(x)\big\|^{2}\mid X=x\right]\;\geq\;\frac{1}{K}\cdot\frac{\sigma^{2}(x)}{\mathrm{RV}(x)}\;\mathbb{E}[\|s\|^{2}\mid X=x]. |  |

If σ​(x)=0\sigma(x)=0, the lower bound is zero and thus vacuous.

This bound makes explicit that smaller RV​(x)\mathrm{RV}(x) yields a larger variance floor for the normalized estimator if all other factors are the same.

## Appendix O Core Author Contributions

Zihan Wang contributed across the full project lifecycle, including conceptualization, codebase and environment development, formal analysis, experiments, figures and plots, paper writing, and project correspondence. Chi Gui and Xing Jin contributed to the key ideas, software infrastructure, experiments, plots, and paper writing. Licheng Liu primarily contributed to the formal analysis and theory, experiments, and participated in paper writing. Qineng Wang contributed to the key ideas, software infrastructure, figures and plots, experiments, and paper writing.
