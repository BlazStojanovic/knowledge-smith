---
arxiv: '2604.26256'
authors:
- Tianhao Hu
- Xiangcheng Liu
- Youshao Xiao
- Yang Zheng
- Xuan Huang
- Jinrui Ding
- Yufei Zhang
- Tao Liang
- Hongyu Zang
- Quan Chen
- Yueqing Sun
- Wenjie Shi
- Chao Zhang
- Wei Wang
- Qi Gu
- Yerui Sun
- Yucheng Xie
- Xunliang Cai
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model
  Training'
url: https://arxiv.org/abs/2604.26256
year: 2026
---

[2604.26256] DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training














function detectColorScheme(){
var theme="light";
var current\_theme = localStorage.getItem("ar5iv\_theme");
if(current\_theme){
if(current\_theme == "dark"){
theme = "dark";
} }
else if(!window.matchMedia) { return false; }
else if(window.matchMedia("(prefers-color-scheme: dark)").matches) {
theme = "dark"; }
if (theme=="dark") {
document.documentElement.setAttribute("data-theme", "dark");
} else {
document.documentElement.setAttribute("data-theme", "light"); } }
detectColorScheme();
function toggleColorScheme(){
var current\_theme = localStorage.getItem("ar5iv\_theme");
if (current\_theme) {
if (current\_theme == "light") {
localStorage.setItem("ar5iv\_theme", "dark"); }
else {
localStorage.setItem("ar5iv\_theme", "light"); } }
else {
localStorage.setItem("ar5iv\_theme", "dark"); }
detectColorScheme(); }



# DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training

Tianhao Hu
  
Xiangcheng Liu
  
Youshao Xiao
  
Yang Zheng
  
Xuan HuangJinrui Ding, Yufei Zhang, Tao Liang, Hongyu Zang, Quan Chen, Yueqing Sun
  
Wenjie Shi
  
Chao Zhang
  
Wei Wang
  
Qi Gu
  
Yerui Sun
  
Yucheng Xie
  
Xunliang CaiMeituan, China
  
zhangchao167@meituan.com
Equal contributions. Corresponding authors.

###### Abstract

Reinforcement learning (RL) has become a critical paradigm for LLM post-training, yet the rollout phase—accounting for 50–80% of total step time—is bottlenecked by skewed generation: long-tailed trajectories indispensable for model performance block the entire training pipeline. Asynchronous training offers a natural remedy by overlapping generation with training, but it introduces a fundamental tension between efficiency and algorithmic convergence. We identify three constraints in asynchronous training to preserve model convergence: intra-trajectory policy consistency, data integrity, and bounded staleness. Different asynchronous strategies navigate this tension by making different design tradeoffs: some overlap pipeline stages without eliminating generation bubbles, while others segment long-tailed trajectories across policy iterations, introducing additional algorithmic considerations. The problem is also further exacerbated in Mix-of-Experts architectures. Therefore, we propose DORA (Dynamic ORchestration for Asynchronous Rollout), which addresses these challenges through algorithm-system co-design and offers practitioners a new asynchronous paradigm with the standard RL formulation. DORA introduces *multi-version streaming training*, a novel asynchronous paradigm that maintains multiple policy versions in the rollout instances in parallel and overlaps the training—simultaneously achieving full bubble elimination without compromising algorithmic constraints. A centralized *load-balancing orchestrator* dynamically re-partitions resources across versions and migrates requests to resolve resource fragmentation. A distinctive co-design insight is that intra-trajectory policy consistency yields KV-Cache equivalence across same-version instances, enabling *zero-re-prefill migration*. Experimental results in the open-source benchmarks demonstrate that our DORA system achieves substantial improvements in efficiency—up to 2.12×2.12\times in end-to-end throughput and 8.2×8.2\times in rollout stage only when compared to the synchronous training—without compromising convergence. Furthermore, in large-scale industrial applications with thousands of accelerators, DORA accelerates the rollout stage up to 6.2×\times compared to synchronous training across various production scenarios. The resulting open-source models, LongCat-Flash-Thinking, achieve competitive performance on complex reasoning benchmarks, matching that of leading open-source reasoning models.

## 1 Introduction

Reinforcement Learning (RL) has become a pivotal paradigm for LLM post-training, leveraging test-time scaling (snell2024scaling) to advance complex reasoning and agentic capabilities (Claude-Opus-4.5; openai\_o1; deepseek-math-v2; team2025introducing; team2026longcat). The RL training loop sequentially cycles through rollout (trajectory generation), experience preparation (reward and reference computation), and model training. In practice, rollout accounts for 50%–80% of total step duration, making it the dominant throughput bottleneck (team2025introducing; wu2025llamarl; xiao2023adaptive). The bottleneck is further exacerbated by the skewed generation problem: under synchronous training, the entire batch is blocked by the single longest output, leaving most devices idle. In mathematics and coding tasks, for example, the 99th-percentile output length can exceed the median by over an order of magnitude (Figures [3](#S3.F3 "Figure 3 ‣ 3.1 Asynchronous RL Training ‣ 3 Problem Formulation ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training") and [3](#S3.F3 "Figure 3 ‣ 3.1 Asynchronous RL Training ‣ 3 Problem Formulation ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training")), and generation workloads are memory-bound, so additional compute resources yield negligible speedup. Critically, these long-tailed trajectories are indispensable: recent work on reasoning LLMs (openai\_o1; guo2025deepseek) demonstrates that extended chain-of-thought trajectories—precisely those in the tail of the generation-length distribution—are the primary carriers of emergent reasoning capabilities during RL training.

This motivates asynchronous training, where generation and training overlap to improve training efficiency. However, asynchronous training introduce a fundamental tension between efficiency and algorithmic correctness. Asynchronous training inherently yields stale trajectories, necessitating importance sampling to bridge the distribution gap between the rollout and the latest policy. Crucially, even with these corrections, the requirement for intra-trajectory policy consistency remains absolute. Widely adopted algorithms like PPO (schulman2017proximal) and GRPO (shao2024deepseekmath) rely on this assumption to underpin their convergence guarantees.

To achieve stable convergence in asynchronous training, we identify three key constraints: (i) intra-trajectory policy consistency (C1), ensuring each trajectory generated by a single policy version; (ii) data integrity (C2) preventing the discard of trajectories, especially crucial long-tailed reasoning trajectories; and (iii) bounded staleness (C3) limiting the lag between generating and training policies. We further formalize these constraints in Section [3.1](#S3.SS1 "3.1 Asynchronous RL Training ‣ 3 Problem Formulation ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training").

Existing asynchronous approaches navigate this design space with different tradeoffs (detailed analysis in Section [3.3](#S3.SS3 "3.3 Tradeoffs in Existing Approaches ‣ 3 Problem Formulation ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training")).
One-step off-policy methods (noukhovitch2024asynchronous; luo2025deepcoder; zhong2025streamrl; han2025asyncflow) satisfy all three constraints and overlap pipeline stages, though intra-node and inter-node bubbles within the rollout remain. Replication-based methods (gao2025rollpacker; zhang2025sortedrl) overprovide the samples in response or prompts level and may filtering the long trajectories, which could possible undermine data integrity (C2) by discarding incomplete long-tailed trajectories. This practice introduces significant length-biased sampling and results in the loss of pivotal long CoT trajectories, which may ultimately compromise model convergence. Partial rollout methods (team2025kimi; wu2025llamarl; fu2025areal) effectively eliminate bubbles by segmenting long trajectories across weight updates which relaxes C1, with corresponding algorithmic adaptations such as gradient masking (team2025kimi) or decoupled PPO objectives (fu2025areal). This paradigm makes an explicit tradeoff: it gains direct bubble elimination at the cost of relaxing intra-trajectory consistency (C1) and incurring re-prefill overhead upon each weight update—costs that are amplified in MoE architectures (jiang2024mixtral; guo2025deepseek) and long-context scenarios. This motivates exploring a new paradigm that can achieve full bubble elimination while preserving the standard RL formulation.

We propose DORA (Dynamic ORchestration for Asynchronous Rollout), which addresses the long-tail generation problem through *algorithm-system co-design*—maximizing rollout efficiency while simultaneously satisfying all three algorithmic constraints.
DORA introduces *multi-version streaming training*, a novel asynchronous paradigm that maintains multiple policy versions concurrently on rollout instances, enabling trajectory-level streaming without batch barriers. Each trajectory is generated entirely by a single policy version (C1); long-tailed trajectories are retained inner the group (C2); and a sliding window enforces a deterministic staleness bound (C3).
A centralized Load-Balancing Orchestrator dynamically re-partitions resources across versions and orchestrates request migrations to maximize utilization.
A distinctive insight is that C1—beyond its algorithmic necessity—yields a powerful system-level property: KV-Cache states are mathematically equivalent across instances hosting the same policy version, enabling zero-re-prefill migration—an optimization uniquely enabled by maintaining C1.

Our main contributions are summarized as follows:

∙\bullet Multi-Version Streaming Training.
We formalize asynchronous RL training as a constrained optimization—maximizing generation efficiency subject to three algorithmic constraints C1–C3.
We introduce multi-version streaming training, a novel asynchronous paradigm that maintains multiple policy versions concurrently on rollout instances, enabling trajectory-level streaming that simultaneously achieves full bubble elimination and satisfies all three constraints.

∙\bullet Dynamic Orchestration.
We design a centralized Load-Balancing Orchestrator that dynamically re-partitions DP groups across model versions proportional to real-time workloads, performs staleness-aware data supplementation that prioritizes sample freshness, and orchestrates P2P weight and request migrations—all while respecting C2 (no trajectory abandoned) and C3 (sample freshness prioritized).

∙\bullet Zero Re-prefill KV-Cache Reuse.
We exploit the system-level consequence of C1: KV-Cache mathematical equivalence across same-version instances enables cross-instance transfer that completely avoids re-prefill during migration—especially beneficial for long-context reasoning and MoE architectures.

∙\bullet Extensive Evaluation.
Our experiments on open-source benchmarks illustrate DORA achieves up to 2.12 ×\times end-to-end throughput improvement and 8.2×8.2\times acceleration on the rollout stage over synchronous training while preserving convergence parity. Additionally, our large-scale industrial deployment produces the competitive open-source model LongCat-Flash-Thinking.

## 2 Preliminaries

### 2.1 Synchronous RL Training

In the following section, we take the GRPO (guo2025deepseek), a variant of PPO, as an example to illustrate the synchronous RL training. We denote π\pi as the autoregressive language model parameterized by θ\theta. Given a prompt xx from the training set 𝒟\mathcal{D}, the likelihood of response yy is denoted as πθ​(y|x)=∏t=1Lπθ​(yt|x,y<t)\pi\_{\theta}(y|x)=\prod\_{t=1}^{L}\pi\_{\theta}(y\_{t}|x,y\_{<t}), where LL denote the length of the response. Using the trajectory generated from behavior policy πw\pi\_{w}, synchronous training optimizes the policy model within a trust region with group-level advantages, via the following objective (omitting the KL regularization term):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒥sync​(θ)=\displaystyle\mathcal{J}\_{\text{sync}}(\theta)= | 𝔼x∼𝒟,{yi}i=1G∼πw(⋅|x)​[1G​∑i=1G1Li​∑t=1Li(min⁡(ri,t​(θ)​A^i,t,clipε⁡(ri,t​(θ))​A^i,t))],\displaystyle\mathbb{E}\_{\begin{subarray}{c}x\sim\mathcal{D},\\ \{y\_{i}\}\_{i=1}^{G}\sim\pi\_{w}(\cdot|x)\end{subarray}}\Bigg[\frac{1}{G}\sum\_{i=1}^{G}\frac{1}{L\_{i}}\sum\_{t=1}^{L\_{i}}\Bigg(\min\Big(r\_{i,t}(\theta)\hat{A}\_{i,t},\operatorname{clip}\_{\varepsilon}\big(r\_{i,t}(\theta)\big)\hat{A}\_{i,t}\Big)\Bigg)\Bigg], |  | (1) |

where ri,t​(θ)=πθ​(yi,t|x,yi,<t)πw​(yi,t|x,yi,<t)r\_{i,t}(\theta)=\frac{\pi\_{\theta}(y\_{i,t}|x,y\_{i,<t})}{\pi\_{w}(y\_{i,t}|x,y\_{i,<t})} is the importance ratio, clipε\operatorname{clip}\_{\varepsilon} clips to [1−ε,1+ε][1-\varepsilon,1+\varepsilon], and GG is the number of responses sampled per prompt. The advantage A^i,t=A^i\hat{A}\_{i,t}=\hat{A}\_{i} is shared across all tokens in trajectory yiy\_{i} and computed via group-level advantage:

|  |  |  |  |
| --- | --- | --- | --- |
|  | A^i=ri−mean​({ri}i=1G)std​({ri}i=1G),\hat{A}\_{i}=\frac{r\_{i}-\mathrm{mean}(\{r\_{i}\}\_{i=1}^{G})}{\mathrm{std}(\{r\_{i}\}\_{i=1}^{G})}, |  | (2) |

where rir\_{i} denotes the reward of trajectory yiy\_{i} and yi∼πw(⋅∣x)y\_{i}\sim\pi\_{w}(\cdot\mid x) for a single model version ww.

### 2.2 Model Placement

Distributed RL training systems for LLMs adopt one of three primary model placement strategies: colocated (yao2023deepspeed; sheng2025hybridflow), disaggregated (xiao2023adaptive; hu2024openrlhf; zhong2025streamrl), or elastic colocated (team2025longcat; wang2025seamlessflow; yu2025rlinf). In the colocated architecture (yao2023deepspeed; sheng2025hybridflow), all RL roles share the same device, with runtime role switching achieved through parameter resharding or context switching with offloading techniques. However, this approach tightly couples hardware resources, limiting efficiency for heterogeneous workloads. In contrast, the disaggregated architecture assigns different roles to distinct physical resources, typically deploying actor models for generation and training actor on separate device groups. This enables flexible allocation of accelerator types and quantities to suit the specific demands of each workload, though it can lead to device idleness. The elastic colocated architecture allows rollout instances to scale up or down dynamically, optimizing device utilization when training devices are idle, while retaining the benefits of both colocated and disaggregated architectures.

### 2.3 LLM Training and Inference Engine

Modern RL training systems build on mature training engines (Megatron-LM (megatron-lm), Deepspeed (rajbhandari2020zero; zhang2024rethinking)), FSDP (FSDP23) and inference engines (vLLM (kwon2023efficient), SGLang (zheng2024sglang)) for their respective workloads (xiao2023adaptive; hu2024openrlhf; sheng2025hybridflow). For the generation engine, mainstream LLM inference engines primarily employ paged attention with KV-cache techniques (pope2023efficiently; kwon2023efficient; zheng2024sglang) and high-performance, inference-optimized kernels (e.g., fused kernels) to minimize latency and maximize throughput. It is important to note that KV-cache memory is particularly limited and scarce in long-context scenarios, where devices require additional GPU or CPU memory to maintain KV-cache blocks for the rapidly increasing input and output tokens, ensuring optimal generation efficiency (qin2024mooncake).

## 3 Problem Formulation

### 3.1 Asynchronous RL Training

We formalize the GRPO objective and its algorithmic constraints under asynchronous training, where trajectories in a single training batch may originate from up to KK-step staleness policy versions WK={wj,…,wj−K+1}W\_{K}=\{w\_{j},\ldots,w\_{j-K+1}\}.

Let ℬj\mathcal{B}\_{j} denote the set of trajectory indices generated by version wjw\_{j}, with ∑j=1K|ℬj|=G\sum\_{j=1}^{K}|\mathcal{B}\_{j}|=G.
Each trajectory yi(j)y\_{i}^{(j)} is the complete response generated by πwj\pi\_{w\_{j}} for prompt xx, where the superscript (j)(j) marks its version of behavior model.
The asynchronous objective aggregates across all KK versions:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒥async​(θ)=\displaystyle\mathcal{J}\_{\text{async}}(\theta)= | 𝔼x∼𝒟,{yi(j)}i∈ℬj∼πwj(⋅|x)​[1G​∑j=1K∑i∈ℬj⏞C2: data integrity​1Li(j)​∑t=1Li(j)min⁡(ri,t(j)​(θ)​A^i,t(j),clipε⁡(ri,t(j)​(θ))​A^i,t(j))],\displaystyle\mathbb{E}\_{\begin{subarray}{c}x\sim\mathcal{D},\\ \{y\_{i}^{(j)}\}\_{i\in\mathcal{B}\_{j}}\sim\pi\_{w\_{j}}(\cdot|x)\end{subarray}}\Bigg[\overbrace{\frac{1}{G}\sum\_{j=1}^{K}\sum\_{i\in\mathcal{B}\_{j}}}^{\textbf{\text{C2: data integrity}}}\frac{1}{L\_{i}^{(j)}}\sum\_{t=1}^{L\_{i}^{(j)}}\min\!\Big(r\_{i,t}^{(j)}(\theta)\,\hat{A}\_{i,t}^{(j)},\;\operatorname{clip}\_{\varepsilon}\!\big(r\_{i,t}^{(j)}(\theta)\big)\,\hat{A}\_{i,t}^{(j)}\Big)\Bigg], |  | (3) |

where the importance ratio and constraints are defined:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ri,t(j)​(θ)=πθ​(yi,t(j)∣⋅)πwj​(yi,t(j)∣⋅),∀t:yi,t(j)∼πwj(⋅∣x,yi,<t(j))⏟C1: intra-traj. consistency,v​(θ)−v​(wj)≤K⏟C3: bounded staleness,r\_{i,t}^{(j)}(\theta)=\frac{\pi\_{\theta}(y\_{i,t}^{(j)}\mid\cdot)}{\pi\_{w\_{j}}(y\_{i,t}^{(j)}\mid\cdot)},\quad\underbrace{\forall\,t:\;y\_{i,t}^{(j)}\sim\pi\_{w\_{j}}(\cdot\mid x,\,y\_{i,<t}^{(j)})}\_{\textbf{C1: intra-traj.\ consistency}},\quad\underbrace{v(\theta)-v(w\_{j})\leq K}\_{\textbf{C3: bounded staleness}}, |  | (4) |

where v​(⋅)v(\cdot) denotes the version number of a policy model. The staleness of any trajectory—measured as the version gap between the current training policy πθ\pi\_{\theta} and its behavior policy πwj\pi\_{w\_{j}}—is bounded by the maximum staleness KK.

* •

  Intra-trajectory Consistency (C1): ∀t∈[1,Li],yi,t(j)∼πwj(⋅∣x,yi,<t(j))\forall\,t\in[1,L\_{i}],\;y\_{i,t}^{(j)}\sim\pi\_{w\_{j}}(\cdot\mid x,y\_{i,<t}^{(j)}) for a single version wjw\_{j}. Every token within a trajectory is generated by the same policy version, preserving the well-defined RL formulation.
* •

  Data Integrity (C2): All GG responses {yi}i=1G\{y\_{i}\}\_{i=1}^{G} must not be abandoned. As discussed in Section [1](#S1 "1 Introduction ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training"), long-tailed CoT trajectories are indispensable; systematically losing such samples removes high-value gradient signals and corrupts the group-relative advantage estimation A^i\hat{A}\_{i}, especially for GRPO.
* •

  Bounded Staleness (C3): For every trajectory yi(j)y\_{i}^{(j)} in the training batch, the version gap v​(θ)−v​(wj)≤Kv(\theta)-v(w\_{j})\leq K must be bounded. Classical theorems in asynchronous optimization (lian2015asynchronous; zheng2017asynchronous) demonstrate that unbounded staleness precludes convergence guarantees.

The training phase additionally requires the number of sampled trajectories to match the train batch size (T​B​STBS) specified by algorithm experts.

![Refer to caption](/html/2604.26256/assets/x1.png)


Figure 1: Response length distribution in the RL training in the DAPO-Math-17K dataset.

![Refer to caption](/html/2604.26256/assets/x2.png)


Figure 2: Response length distribution in the production. The stacked bar is due to the overlong truncation.

![Refer to caption](/html/2604.26256/assets/x3.png)


Figure 3: Prefill duration of Longcat-Flash (team2025longcat) when EP is 128 on mid-end accelerators.

### 3.2 Skewed Generation Problem

For the training efficiency, an RL training step consists of rollout (prefill + decode, 50–80% of step duration) and training (experience preparation + model training). The objective is to minimize total step duration:

|  |  |  |  |
| --- | --- | --- | --- |
|  | min⁡Ttrain+TPrefill+TDecode⇒min⁡Ttrain+TPrefill+τ​maxj⁡maxi∈Devicej⁡{Li}⏟intra-node bubble⏞inter-node bubble\min\;T\_{\text{train}}+T\_{\text{Prefill}}+T\_{\text{Decode}}\;\Rightarrow\;\min\;T\_{\text{train}}+T\_{\text{Prefill}}+\overbrace{\tau\max\_{j}\underbrace{\max\_{i\in\text{Device}\_{j}}\{L\_{i}\}}\_{\text{intra-node bubble}}}^{\text{inter-node bubble}} |  | (5) |

* •

  Request Characteristics: Each request has an input token length (known) and an output token length Li∈[1,Lmax]L\_{i}\in[1,L\_{\max}] that is hard to predict in practice and typically follows a long-tailed distribution in long-context scenarios (Figure [3](#S3.F3 "Figure 3 ‣ 3.1 Asynchronous RL Training ‣ 3 Problem Formulation ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training") and Figure [3](#S3.F3 "Figure 3 ‣ 3.1 Asynchronous RL Training ‣ 3 Problem Formulation ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training")). Partial-rollout or migration techniques may additionally introduce long-tailed input lengths due to re-prefill of interrupted trajectories.
* •

  Inference Efficiency: Rollout phase processes R​B​SRBS (Rollout Batch Size) prompts. Without substantially downgrading the rollout efficiency, each request incurs a compute-bound prefill (TprefillT\_{\text{prefill}}) and a memory-bound decode (TdecodeT\_{\text{decode}}) duration. Each device accommodates up to KK concurrent requests to avoid recomputation, constrained by accelerator memory (model footprint and KV-cache, which scales with sequence length). Under fixed batch size, the time per output token (TPOT) τ\tau is approximately constant.

![Refer to caption](/html/2604.26256/assets/x4.png)


Figure 4: Skewed Bubbles in the Synchronous Training

![Refer to caption](/html/2604.26256/assets/x5.png)


Figure 5: The non-EP general matrix multiplication is unbalanced when handling long-tailed inputs, and expert parallelism is 128.

Skewed Bubbles in Synchronous Training.
Because decode duration is determined by the longest output (Equation [5](#S3.E5 "Equation 5 ‣ 3.2 Skewed Generation Problem ‣ 3 Problem Formulation ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training")), auto-regressive generation creates two types of device idleness (Figure [5](#S3.F5 "Figure 5 ‣ 3.2 Skewed Generation Problem ‣ 3 Problem Formulation ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training")): the intra-node bubble, where completed slots on a device sit idle while a long-tailed request continues, and the inter-node bubble, where faster instances wait for the slowest node. Oversampling with a large R​B​SRBS is ineffective in synchronous training, as only T​B​STBS samples are consumed per step and others are discarded due to staleness.

### 3.3 Tradeoffs in Existing Approaches

We now analyze existing asynchronous strategies through the lens of the three constraints defined above. Each strategy navigates the efficiency–convergence tradeoff differently, making distinct design choices among the three constraints:

* •

  Replication-based methods (gao2025rollpacker; zhang2025sortedrl) generate R​B​S>T​B​SRBS>TBS prompts and discard excess trajectories once T​B​STBS is met, reducing effective TDecodeT\_{\text{Decode}} but violating data integrity (C2).
* •

  One-step off-policy methods (zhong2025streamrl; han2025asyncflow; he2025history) overlap rollout with training by using the previous step’s weights, yielding min⁡max⁡{TPrefill+TDecode,TTrain}\min\max\{T\_{\text{Prefill}}+T\_{\text{Decode}},\,T\_{\text{Train}}\}. While satisfying all three constraints (staleness≤1\,\leq 1), this does not address intra-node or inter-node bubbles, which are inherent to the rollout.
* •

  Partial-rollout methods (team2025kimi; wu2025llamarl; fu2025areal; slime\_github) alleviate bubbles by breaking up the long responses into segments and utilizes the latest policy model to generate each segment across different iterations. It trades intra-trajectory consistency for direct bubble reduction. This design choice introduces two additional challenges: (i) Algorithmic: allowing different policy versions within a single trajectory relaxes C1, requiring additional corrections (e.g., gradient masking, decoupled objectives) to maintain convergence. (ii) System: each weight update requires re-prefill of all active trajectories. As shown in Figure [3](#S3.F3 "Figure 3 ‣ 3.1 Asynchronous RL Training ‣ 3 Problem Formulation ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training"), these re-prefill costs grow significantly with increasing context length, especially on mid-end accelerators.

Exacerbation in MoE Architectures
This challenge is further exacerbated in MoE architectures (e.g., Figure [5](#S3.F5 "Figure 5 ‣ 3.2 Skewed Generation Problem ‣ 3 Problem Formulation ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training")), where re-prefill costs and workload imbalances in non-MoE layers within expert parallelism groups create severe bottlenecks for the whole EP group.

These observations motivate a new asynchronous paradigm that maximizes rollout efficiency subject to C1–C3. In the next section, we present DORA, which achieves this through algorithm-system co-design: multi-version streaming training that naturally satisfies all constraints, dynamic cross-version orchestration that respects constraint boundaries, and KV-Cache reuse that turns C1 into a system-level efficiency advantage.

## 4 DORA Design

### 4.1 System Overview

Without loss of generality, we present DORA using a disaggregated architecture, although it readily extends to the colocated architecture. The design rests on four interlocking mechanisms, each addressing a specific facet of the constrained optimization:

*(i) Multi-version streaming training* (Section [4.2](#S4.SS2 "4.2 Multi-version Streaming Training ‣ 4 DORA Design ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training")) maintains multiple policy versions concurrently on rollout instances, enabling trajectory-level streaming that eliminates both intra-node and inter-node bubbles while naturally satisfying C1–C3.

*(ii) Dynamic resource orchestration* (Section [4.3](#S4.SS3 "4.3 Dynamic Resource Orchestration ‣ 4 DORA Design ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training")) resolves the resource fragmentation inherent in multi-version rollout by dynamically re-partitioning DP groups and migrating requests—all scheduling decisions respecting C2 and C3.

*(iii) KV-Cache reuse* (Section [4.4](#S4.SS4 "4.4 KV-Cache Reuse ‣ 4 DORA Design ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training")) exploits a distinctive co-design insight: C1 is not merely an algorithmic necessity but yields mathematical equivalence of KV-Cache states across instances of the same policy version, enabling nearly zero-re-prefill migration—an optimization uniquely enabled by DORA’s adherence to C1.

As shown in Figure [6](#S4.F6 "Figure 6 ‣ 4.1 System Overview ‣ 4 DORA Design ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training"), the workflow of our DORA system proceeds as follows. A RolloutManager dispatches prompts to rollout instances, tagging each with a policy version to enforce C1. Completed trajectories stream into an asynchronous TransferQueue equipped with staleness monitoring for staleness control. The Trainer consumes the number of T​B​STBS samples for experience preparation and model training, then synchronizes the latest weights with rollout instances. A Load-balancing orchestrator monitors per-version workloads and triggers resource re-partitioning and request migration as needed, preserving the intermediate execution state. Notably, these components run on different nodes and are primarily responsible for logic control and coordination via Remote Procedure Call (RPC), while workers running on accelerators execute the actual tasks.

![Refer to caption](/html/2604.26256/assets/x6.png)


Figure 6: The execution timeline of DORA’s multi-version streaming training system.

### 4.2 Multi-version Streaming Training

Trajectory-level streaming
DORA eliminates the synchronous barrier by streaming completed trajectories directly to training without waiting for straggling trajectories. During the rollout phase, we maintain multiple versions of policy weights in the rollout instances, where each Data Parallel (DP) group hosts a single version of the policy weights. At the onset of each step, we overprovide the rollout prompts, where R​B​S>T​B​SRBS>TBS generation requests are dispatched to the rollout instances. Training begins as soon as T​B​STBS samples are collected; unfinished long-tailed trajectories continue under their original policy version and flow into subsequent steps, ensuring the long trajectories are not abandoned or block the training. As illustrated in Figure [6](#S4.F6 "Figure 6 ‣ 4.1 System Overview ‣ 4 DORA Design ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training"), the Rollout and training processes execute non-blockingly; only after a training iteration concludes does the Trainer notify the RolloutManager to synchronize the latest weights.

Multi-version policy management
The key insight is that maintaining multiple policy versions concurrently allows long-tailed trajectories to continue under their original version while the training proceeds with completed trajectories. As illustrated in Figure [6](#S4.F6 "Figure 6 ‣ 4.1 System Overview ‣ 4 DORA Design ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training"), each prompt is tagged with a version wjw\_{j} upon dispatch, ensuring at∼πwj(⋅∣st)a\_{t}\sim\pi\_{w\_{j}}(\cdot\mid s\_{t}) for every token —naturally satisfying C1 without algorithmic modifications. For example, in Figure [6](#S4.F6 "Figure 6 ‣ 4.1 System Overview ‣ 4 DORA Design ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training"), Trajectory 4 (a long-tail request under version w1w\_{1}) spans two training steps while Trajectories 1–3 complete and stream into training during Step 1. The system proceeds to Step 2 with updated weights w2w\_{2} without waiting for Trajectory 4 to complete, which continues under its original version w1w\_{1} in a dedicated DP group. This allows the legacy-version requests to execute in parallel, thereby alleviating the inter-node bubble and maximizing the utilization of all Data Parallel groups in the rollout instances.

Sliding-window staleness control
Active versions are managed through a sliding window W={wj,…,wj−K+1}W=\{w\_{j},\ldots,w\_{j-K+1}\} of size |W|≤K|W|\leq K. The window advancement follows a strict protocol to control the staleness and ensure C3. The window slides forward only when all trajectories from the oldest version wj−K+1w\_{j-K+1} have been collected and forwarded to training. This enforces C3 and provides a deterministic upper bound on policy staleness. The staleness bound KK serves as an explicit control knob for the convergence–throughput tradeoff: a smaller KK yields more on-policy data at the cost of rollout efficiency; a larger KK increases throughput with controlled convergence impact.

Remaining challenges
While multi-version streaming satisfies C1–C3 and partially alleviates both bubble types, it introduces two second-order efficiency challenges that motivate the subsequent mechanisms:
*(i) Resource fragmentation.* We observe the orphan requests that the legacy version’s pending requests decrease monotonically, yet its bound resources remain fixed, leading to notable device underutilization. This motivates the dynamic orchestration described in Section [4.3](#S4.SS3 "4.3 Dynamic Resource Orchestration ‣ 4 DORA Design ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training").
*(ii) Re-prefill overhead.* Migrating requests across DP groups naively re-triggers the full prefill phase—prohibitive for long-context scenarios (64k–128k tokens). This motivates the KV-Cache reuse mechanism in Section [4.4](#S4.SS4 "4.4 KV-Cache Reuse ‣ 4 DORA Design ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training"), which exploits the mathematical equivalence guaranteed by C1.

![Refer to caption](/html/2604.26256/assets/x7.png)


Figure 7: The workflow of the Dynamic Resource Orchestration.

### 4.3 Dynamic Resource Orchestration

Under multi-version streaming training, the pending request count RwR\_{w} of each legacy version ww decreases monotonically as trajectories complete, yet the DP groups assigned to ww remain fixed. This implies that static resource allocation leads to progressive underutilization—a legacy version’s DP groups may each serve only one or two residual requests, while the latest version, which carries the majority of new prompts, is under-provisioned. Therefore, DORA requires proactively rebalancing workloads while simultaneously controlling staleness.

To resolve this resource fragmentation, DORA employs a centralized orchestrator that dynamically re-partitions resources across model versions. The orchestrator maintains real-time metrics—active request counts per version, KV-Cache utilization, and generation progress—and supports three re-balancing triggers: (1) update-driven, mandatory upon the completion of each training step to promote the new policy version; (2) utilization-based, activated when KV-Cache pressure exceeds a threshold, avoiding eviction-induced recomputation; and (3) temporal-based, periodic execution to prevent orphan requests from lingering in legacy versions.

As illustrated in Figure [7](#S4.F7 "Figure 7 ‣ 4.2 Multi-version Streaming Training ‣ 4 DORA Design ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training"), each re-balancing cycle executes three coordinated operations:

* •

  Resource Partitioning Plan The orchestrator assesses the distribution of active and pending requests across all maintained versions WW to produce a migration plan. It computes the target DP group count for each version w∈Ww\in W proportional to its current workload: 𝒟​𝒫w=Round​(𝒟​𝒫t​o​t​a​l×Rw/∑w′Rw′)\mathcal{DP}\_{w}=\text{Round}(\mathcal{DP}\_{total}\times R\_{w}/\sum\_{w^{\prime}}R\_{w^{\prime}}), preventing resources from being stranded on legacy versions with dwindling tasks. This addresses the inter-instance data skewness identified in Section [3.2](#S3.SS2 "3.2 Skewed Generation Problem ‣ 3 Problem Formulation ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training").
* •

  P2P weight update and request migration Once the migration plan is determined, the orchestrator generates a mapping from the current partition {𝒟​𝒫wcur}\{\mathcal{DP}\_{w}^{\text{cur}}\} to the target partition {𝒟​𝒫wtgt}\{\mathcal{DP}\_{w}^{\text{tgt}}\}. For each version whose allocation changes (𝒟​𝒫wtgt≠𝒟​𝒫wcur\mathcal{DP}\_{w}^{\text{tgt}}\neq\mathcal{DP}\_{w}^{\text{cur}}), it leverages P2P weight transfers to rescale the DP groups—decrease the legacy versions and increase the latest one. Active requests on re-assigned nodes are migrated to their new DP groups with execution states fully preserved via KV-Cache reuse (Section [4.4](#S4.SS4 "4.4 KV-Cache Reuse ‣ 4 DORA Design ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training")). Notably, no trajectory is abandoned during this process, maintaining C2.
* •

  Staleness-aware data supplementation To control data staleness and ensure C3 compliance, the orchestrator prioritizes the latest policy version for proactive data injection, dispatching supplemental prompts until the R​B​SRBS is fully met. This strategy maximizes sample freshness by ensuring that the majority of new trajectories are generated using the most recent model weights. Subsequently, to maintain high-watermark utilization across the cluster, the orchestrator performs opportunistic data injection following the request migration phase. Legacy versions are only supplemented with sufficient prompts to fill their residual idle slots. This tiered injection approach effectively saturates all rollout instances while preventing the over-production of stale trajectories, striking an optimal balance between hardware occupancy and algorithmic freshness.

This optimization cycle exemplifies algorithm-system co-design: Proportional Resource Partitioning resolves the resource fragmentation inherent in multi-version rollout; Request Migration preserves trajectory execution state and maintains C2; and Staleness-Aware Data Supplementation ensures sustained high hardware occupancy while respecting C3. The detailed logic is formalized in Algorithm [1](#alg1 "Algorithm 1 ‣ 4.3 Dynamic Resource Orchestration ‣ 4 DORA Design ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training").

1: Input: Active versions Wa​c​t​i​v​eW\_{active}, Request counts {Rw}w∈W\{R\_{w}\}\_{w\in W}, Total DP groups 𝒟​𝒫t​o​t​a​l\mathcal{DP}\_{total}, Request number R​B​SRBS

2: Trigger: Threshold met or Training Step completed

3: // Step 1: Pre-balancing Data Supplementation

4: Rs​u​m←∑w∈Wa​c​t​i​v​eRwR\_{sum}\leftarrow\sum\_{w\in W\_{active}}R\_{w}

5: if Rs​u​m<R​B​SR\_{sum}<RBS then

6:  Inject Rn​e​w←R​B​S−Rs​u​mR\_{new}\leftarrow RBS-R\_{sum} new prompts to wn​e​ww\_{new} // Prioritize latest policy version

7:  Rwn​e​w←Rwn​e​w+Rn​e​wR\_{w\_{new}}\leftarrow R\_{w\_{new}}+R\_{new} // Update request count for re-partitioning

8: end if

9: // Step 2: Resource Re-partitioning (Workload-aware)

10: for each version w∈Wa​c​t​i​v​ew\in W\_{active} do

11:  𝒟​𝒫w←Round​(𝒟​𝒫t​o​t​a​l×Rw∑Rw)\mathcal{DP}\_{w}\leftarrow\text{Round}(\mathcal{DP}\_{total}\times\frac{R\_{w}}{\sum R\_{w}}) // Calculate target DP groups

12: end for

13: // Step 3: Resource and State Migration

14: ℳw​e​i​g​h​t​s←GenerateP2PMaps​(𝒟​𝒫c​u​r,𝒟​𝒫t​g​t)\mathcal{M}\_{weights}\leftarrow\text{GenerateP2PMaps}(\mathcal{DP}\_{cur},\mathcal{DP}\_{tgt})

15: ℳr​e​q​u​e​s​t​s←GenerateMigrateMaps​(ℛc​u​r,ℛt​g​t)\mathcal{M}\_{requests}\leftarrow\text{GenerateMigrateMaps}(\mathcal{R}\_{cur},\mathcal{R}\_{tgt})

16: Execute P2P Weight Update and Request Transfer with KV Cache reuse

17: // Step 4: Opportunistic Legacy Filling

18: for each 𝒟​𝒫w∈{𝒟​𝒫w′}∖{𝒟​𝒫l​a​t​e​s​t}\mathcal{DP}\_{w}\in\{\mathcal{DP}\_{w^{\prime}}\}\setminus\{\mathcal{DP}\_{latest}\} do

19:  Fill requests to saturate the device

20: end for

Algorithm 1 Dynamic Load-Balancing and Orchestration

### 4.4 KV-Cache Reuse

Note that the Request migration across DP groups naively re-triggers the prefill phase. The re-prefill cost scales with the boosting context length and is further amplified in MoE architectures (jiang2024mixtral; guo2025deepseek), where it causes workload imbalance across non-MoE layers as shown in Section [3](#S3 "3 Problem Formulation ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training").

However, constraint C1—beyond its algorithmic necessity—yields a powerful system-level property: since all tokens in a trajectory are generated by the same policy version πw\pi\_{w}, the KV-Cache states are *mathematically equivalent* across any physical instance hosting version ww. This equivalence enables cross-instance KV-Cache transfer that completely avoids re-prefill. Methods that relax C1 forfeit this optimization: each weight update forces a full re-prefill of all ongoing trajectories, up to the output length.

When the Load-Balancing Orchestrator triggers a resource re-allocation, DORA executes a coordinated two-phase state transfer:

* •

  Metadata forwarding Request metadata (request ID, generation state, decoded token count, and version tag) is transmitted via lightweight RPC. This control-plane transfer is negligible in both latency and bandwidth.
* •

  KV-Cache data transfer The voluminous KV Cache data—often comprising gigabytes of memory for long contexts—is transferred using high-performance collective communication primitives.

Locality-aware scheduling
To minimize transfer volume, the orchestrator prioritizes re-assigning requests back to their original ranks when possible—preserving data locality and avoiding physical migration entirely. Only requests that must relocate due to version transitions incur transfer costs.

Hierarchical memory management
To alleviate VRAM pressure of the aggregated requests during long-context training, DORA temporarily offloads KV-Caches to host memory (qin2024mooncake), freeing device memory for active computations while preserving the state for deferred generation. Such a hierarchical memory management safeguards the system efficiency even under heavy and extreme long-tailed workloads.

This mechanism illustrates the central theme of DORA’s co-design: algorithmic constraints are not merely costs paid for correctness, but structural properties that enable system-level optimizations.

## 5 Experiments and Analysis

We evaluate DORA along two axes aligned with its constrained optimization formulation: *efficiency* (the optimization objective—rollout acceleration, end-to-end step time, and throughput) and *algorithmic convergence* (constraints C1–C3—convergence parity and staleness robustness). We further quantify the system overhead introduced by dynamic orchestration to validate DORA’s scalability.

### 5.1 Experimental Setup

Testbed Our experiments are conducted on a cluster consisting of 16 nodes, each equipped with 8 NVIDIA H800 GPUs. Intra-node communication is facilitated by NVLink with a bandwidth of 400 GB/s, while inter-node connectivity is provided by 8×\times400 Gbps network interfaces. Additionally, our production cluster employs non-CUDA accelerators, each providing approximately 60 GB of available device memory.

Models and Metrics Our experiments use Qwen2.5-32B (qwen2025qwen25technicalreport) for dense architectures and LongCat-Flash (team2025longcat) for MoE architectures. We measure end-to-end throughput (tokens/s), calculated as the total tokens (prompts and responses) processed per second, and average step time (min), which represents the wall-clock time per RL iteration. All reported numbers are averaged over five RL iterations after the warm-up phase to reflect steady-state performance.

Datasets We utilize the "DAPO-Math-17k" dataset for training, with the maximum input and output sequence lengths set to 2K and 30K tokens, respectively.

Baselines and Implementation We compare DORA against three representative RL training paradigms, each occupying a distinct position in the constraint–efficiency landscape: (1) Synchronous (All-Colocated), which satisfies all constraints but suffers from batch barriers; (2) One-step off-policy, which satisfies all constraints with staleness K=1K{=}1 but only overlaps pipeline stages without eliminating rollout bubbles; and (3) Partial rollout, which effectively eliminates bubbles but relaxs C1, necessitating algorithmic modifications. All baselines are implemented within the same in-house RL framework to ensure a controlled comparison under identical hardware and software configurations. For the partial rollout paradigm, we implemented it in an All-Colocated architecture similar to the work (team2025kimi). Our RL system uses vLLM (kwon2023efficient) as the inference engine, Megatron-LM (megatron-lm) as the training backend, and extends torch RPC (damania2023pytorch) with streaming primitives. The software environment includes CUDA-12.4, PyTorch-2.6.0, vLLM-0.8.5, and NCCL-2.28.

Training Configurations For the RL algorithm, we follow the setting of DAPO (yu2025dapo), a variant of GRPO. Each rollout consists of a prompt batch size of 512, with 16 responses sampled per prompt, resulting in a global training batch size of 8,192. Each training iteration involves 16 update steps with a micro-batch size of 512. To stress-test efficiency under realistic long-tailed generation, we select an intermediate checkpoint where the mean response length is 2.4K tokens and the maximum reaches 30K.

### 5.2 Training Efficiency

We evaluate DORA (k=3k=3) against all baselines on 64 and 128 GPUs, measuring the rollout time fraction, average wall-clock step time, and end-to-end throughput.

![Refer to caption](/html/2604.26256/assets/x8.png)


Figure 8: Average RL step time across different training paradigms on Dense-32B.

![Refer to caption](/html/2604.26256/assets/x9.png)


Figure 9: End-to-End Throughput across different training paradigms on Dense-32B.

![Refer to caption](/html/2604.26256/assets/x10.png)


Figure 10: DORA vs. synchronous training in production using Longcat-Flash (team2025longcat), where the max response length is 64K.

Rollout Acceleration
The most pronounced improvement lies in the rollout phase, which is the dominant bottleneck in synchronous RL training.
As illustrated in Figure [10](#S5.F10 "Figure 10 ‣ 5.2 Training Efficiency ‣ 5 Experiments and Analysis ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training"), on 64 GPUs, the rollout-only phase (where overlapping fails) accounts for 65% of total step time in the synchronous baseline and 46% in partial rollout. For the disaggregated architecture, the rollout-only fraction represents the portion that is not overlapped by training—accounting for 33% in one-off and merely 12% in DORA. In absolute terms, the rollout duration drops from 14.9 min (synchronous) to just 1.8 min, representing a 8.2×\times rollout speedup. Compared to partial rollout (8.0 min), DORA achieves a 4.4×\times rollout acceleration.
On 128 GPUs, the rollout fraction in synchronous baselines increases to 73% and remains high in partial rollout. DORA maintains a rollout-only fraction of 24% (2.5 min), 111When maximizing overlap, the rollout-only phase can be fully eliminated via elastic colocation or enabling a large staleness threshold., yielding 5.9×\times and 2.9×\times rollout speedups over synchronous and partial rollout, respectively. This compression is attributable to DORA’s multi-version streaming design: long-tailed trajectories no longer block the batch—they continue under legacy versions while new requests saturate the released resources (Section [4.3](#S4.SS3 "4.3 Dynamic Resource Orchestration ‣ 4 DORA Design ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training")). Consequently, the training phase—rather than generation—becomes the dominant contributor to step time.

End-to-End Duration
The rollout acceleration translates directly into substantial end-to-end step time reductions (Figure [10](#S5.F10 "Figure 10 ‣ 5.2 Training Efficiency ‣ 5 Experiments and Analysis ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training")).
On 64 GPUs, DORA achieves an average step time of 14.67 min, a 1.56×\times speedup over the synchronous baseline (22.91 min) and one-off (22.94 min), and a 1.18×\times speedup over partial rollout (17.38 min).
On 128 GPUs, DORA achieves 10.46 min per step, outperforming the synchronous (20.23 min) baseline by 1.93×\times, one-off by 1.34×\times, and partial rollout by 1.33×\times.
Notably, DORA delivers these gains while satisfying all three algorithmic constraints (C1–C3), offering practitioners a paradigm that combines high efficiency with the standard RL formulation.

![Refer to caption](/html/2604.26256/assets/x11.png)


Figure 11: Training reward scores for various training paradigms, and we utilize a staleness of 1 and 3 for the DORA system.

End-to-End Throughput We further report token-level throughput (Figure [10](#S5.F10 "Figure 10 ‣ 5.2 Training Efficiency ‣ 5 Experiments and Analysis ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training")), which captures the effective training bandwidth after accounting for the varying response-length distributions across paradigms and steps. On 64 GPUs, DORA achieves 23,327 tokens/s, outperforming the synchronous baseline (14,143 tokens/s) and one-off (14,141 tokens/s) by 1.65×\times, and partial rollout (19,872 tokens/s) by 1.17×\times. On 128 GPUs, DORA reaches 34,135 tokens/s, representing a 2.12×\times improvement over synchronous (16,070 tokens/s), 1.47×\times over one-off, and 1.11×\times over partial rollout.

Scalability
As hardware scales from 64 to 128 GPUs, DORA’s step time drops from 14.67 min to 10.46 min (1.40×\times speedup) and throughput increases from 23,327 to 34,135 tokens/s (1.46×\times). In contrast, the synchronous baseline exhibits poor scaling (step time 22.91→\to20.23 min, only 1.13×\times), as adding more devices does not resolve the memory-bound long-tailed generation bottleneck. The partial rollout paradigm improves from 17.38 min to 13.95 min (1.25×\times).

Production-Scale MoE Evaluation
We further evaluate DORA on LongCat-Flash (team2025longcat), a MoE model with 560B total parameters, using 4,096 accelerators with a maximum response length of 64K tokens on the production cluster. Due to the prohibitive resource cost of running all baselines at this scale, we compare DORA against the well-tuned synchronous baseline—the production default. As shown in Figure [10](#S5.F10 "Figure 10 ‣ 5.2 Training Efficiency ‣ 5 Experiments and Analysis ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training"), DORA achieves a 3.6×\times rollout speedup in mathematical reasoning and tool-integrated reasoning (TIR) scenarios, and a 6.2×\times acceleration in agentic training on the Tau2-bench (tau2-bench) and Vita (vita-bench) benchmarks (team2025introducing; team2026longcat).

### 5.3 Model Convergence

To validate the system integrity of our DORA system, we evaluate its convergence performance on 72 GPUs by monitoring the mean training reward across 100 training steps. As illustrated in Figure [11](#S5.F11 "Figure 11 ‣ 5.2 Training Efficiency ‣ 5 Experiments and Analysis ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training"), we compare the reward trajectories of DORA (under configurations k=1k=1 and k=3k=3, denoting different staleness bounds) against other baseline methods.

Training Convergence As shown in the figure, both DORA variants achieve similar per-step model convergence trend compared to the synchronous training. However, it is noticed that the per-step duration of the DORA is significantly shorter than all baselines (as demonstrated in Section [5.2](#S5.SS2 "5.2 Training Efficiency ‣ 5 Experiments and Analysis ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training")). It is noted that the partial rollout and one-off training achieves a lower convergence speed when compared with our methods in our setting. This translates to a massive acceleration in the actual wall-clock convergence time.

Staleness Robustness By comparing DORA with k=1k=1 and k=3k=3, we demonstrate the robustness of our staleness control. While k=3k=3 allows a larger
sliding window of active versions to maximize system throughput, it still maintains a stable convergence trend comparable to
that of k=1k=1 and the synchronous baseline. This confirms that DORA’s staleness control successfully delivers the
high-throughput benefits of the multi-version streaming architecture without incurring severe convergence degradation. That
said, we observe that convergence under k=3k=3 is moderately slower than under k=1k=1, reflecting a natural trade-off between throughput and convergence speed.

### 5.4 Overhead Analysis

In this section, we quantify the system overheads introduced by DORA’s dynamic orchestration and KV Cache reuse mechanisms. Figure [12](#S5.F12 "Figure 12 ‣ 5.4 Overhead Analysis ‣ 5 Experiments and Analysis ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training") presents a breakdown of the time consumed by three primary operations as a percentage of the total training time: P2P-based Load Balancing, Request Transfer, and Free Cache. Generally, the overal system overhead does not increase when scaling from 64 GPUs to 128 GPUs.

Orchestration Efficiency The Load-Balancing process—which encompasses requests monitoring, resource re-partitioning, and P2P weight synchronization—is highly efficient. At 64 GPUs, it accounts for only 0.414% of the total execution time. As the cluster scales to 128 GPUs, this overhead increases to 1.519%, reflecting the growing complexity of centralized coordination across a larger pool of rollout instances. Despite this slight increase, the orchestration cost remains marginal compared to the substantial throughput gains (as reported in Section [5.2](#S5.SS2 "5.2 Training Efficiency ‣ 5 Experiments and Analysis ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training")) achieved through dynamic load balancing.

![Refer to caption](/html/2604.26256/assets/x12.png)


Figure 12: System Overhead Analysis.

High-Performance State Migration The overhead associated with Request Transfer—comprising the migration of request metadata and physical KV Cache states—remains well-controlled. It accounts for 3.627% of the execution time at 64 GPUs and decreases to 2.123% at 128 GPUs. The migration costs are well-amortized as the total system throughput increases in larger cluster configurations.

Negligible Memory Management Cost Notably, the computational cost for Free Cache operations (responsible for memory reclamation and VRAM state cleaning) is virtually non-existent. As shown in the zoomed-in inset of Figure [12](#S5.F12 "Figure 12 ‣ 5.4 Overhead Analysis ‣ 5 Experiments and Analysis ‣ DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training"), this operation occupies only 0.019% and 0.027% of the training time for 64 and 128 GPUs, respectively. This confirms that our hierarchical memory management and asynchronous swapping strategies operate entirely off the critical path, imposing no perceptible burden on the system’s end-to-end efficiency.

### 5.5 Industrial Deployment

DORA has served as the default asynchronous training paradigm in our in-house RL training framework since 2025, supporting production workloads across tens of thousands of accelerators for LongCat-series models. In large-scale production jobs, DORA achieves 2–4×\times end-to-end speedup over the well-tuned synchronous baseline with no degradation in model quality across reasoning and agentic scenarios. The system overhead is negligible in the production jobs when using thousands of accelerators. The resultant open-source models achieve competitive performance on complex reasoning and agentic benchmarks, producing the competitive open-source models (team2025introducing; team2026longcat).

## 6 Related Work

To accelerate the RL Training, various distributed training systems have been proposed. Previous RL training systems (yao2023deepspeed; xiao2023adaptive; hu2024openrlhf; sheng2025hybridflow) mainly focused on the model placement in synchronous training. It cycles through rollout, experience preparation, and training within each step, providing clean algorithmic semantics but suffering from batch-barrier idle time, especially in long-context scenarios. K-step off-policy methods (luo2025deepcoder; zhong2025streamrl; noukhovitch2024asynchronous; he2025history; han2025asyncflow)—typically one-step off—overlap generation with training by allowing rollouts to use weights from the previous iteration. While this eliminates the pipeline bubble between stages, intra-node and inter-node bubbles persist when rollout durations are highly skewed. Partial rollout methods (team2025kimi; wu2025llamarl; du2025ulorl; fu2025areal) segment long responses and utilize the latest model for each segment, effectively mitigating long-tailed generation. To compensate for the mid-trajectory weight change, some approaches mask earlier segments during loss computation (team2025kimi) while others apply decoupled PPO objectives (fu2025areal). *Oversampling-based* approaches (gao2025rollpacker; zhang2025sortedrl) overprovide rollout prompts and discard excess trajectories once the batch requirement is met, trading data coverage for reduced tail latency. Concurrently, several works  (seed2025seed1; sheng2025laminar) explore the multi-version streaming training system which are conceptually similar to our paper, while it adopts a two-tier CPU relay architecture for weights management but with a different approach on weights and staleness management.

## 7 Limitations

While DORA demonstrates significant efficiency gains with convergence parity, several limitations merit discussion.

Algorithmic
DORA enforces bounded staleness through a sliding-window mechanism, yet the staleness bound KK must be manually configured by practitioners for each training scenario. The current design relies on PPO’s clipping mechanism as the primary safeguard against off-policy bias from stale trajectories; incorporating adaptive staleness control or explicit delay-compensation techniques (zheng2017asynchronous) could further tighten the convergence–throughput tradeoff.

Experimental
Our evaluation compares DORA against different paradigms implemented within the same in-house framework to ensure a controlled comparison under identical hardware and software configurations.
However, we acknowledge the absence of direct benchmarks against publicly available RL training systems such as veRL (sheng2025hybridflow) and AReaL (fu2025areal). Furthermore, due to resource constraints, we utilize production data exclusively for the MoE architecture, without conducting large-scale experiments on open-source MoE models. We plan to address these limitations and report extended results in future work.

## 8 Conclusion

We present DORA, a scalable asynchronous RL system that addresses the tension between training efficiency and algorithmic correctness in long-context scenarios. By formalizing asynchronous RL as a constrained optimization—maximizing rollout efficiency subject to intra-trajectory consistency (C1), data integrity (C2), and bounded staleness (C3)—DORA introduces multi-version streaming training, offering practitioners a new paradigm that combines high efficiency with the standard RL formulation. A key insight is that maintaining C1 yields KV-Cache equivalence across instances, enabling zero-re-prefill migration—an optimization uniquely enabled by maintaining strict policy consistency. Experiments in the open-source benchmarks demonstrate up to 8.2×\times rollout speedup and 2.12×2.12\times end-to-end acceleration over synchronous training with convergence parity, and industrial deployment over tens of thousands of accelerators produces the competitive open-source model.

[◄](/html/2604.26255)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2604.26256)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2604.26256)
[View original  
on arXiv](https://arxiv.org/abs/2604.26256)[►](/html/2604.26258)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Tue May 5 20:12:19 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

var canMathML = typeof(MathMLElement) == "function";
if (!canMathML) {
var body = document.querySelector("body");
body.firstElementChild.setAttribute('style', 'opacity: 0;');
var loading = document.createElement("div");
loading.setAttribute("id", "mathjax-loading-spinner");
var message = document.createElement("div");
message.setAttribute("id", "mathjax-loading-message");
message.innerText = "Typesetting Equations...";
body.prepend(loading);
body.prepend(message);
var el = document.createElement("script");
el.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js";
document.querySelector("head").appendChild(el);
window.MathJax = {
startup: {
pageReady: () => {
return MathJax.startup.defaultPageReady().then(() => {
body.removeChild(loading);
body.removeChild(message);
body.firstElementChild.removeAttribute('style');
}); } } };
}

// Auxiliary function, building the preview feature when
// an inline citation is clicked
function clicked\_cite(e) {
e.preventDefault();
let cite = this.closest('.ltx\_cite');
let next = cite.nextSibling;
if (next && next.nodeType == Node.ELEMENT\_NODE && next.getAttribute('class') == "ar5iv-bibitem-preview") {
next.remove();
return; }
// Before adding a preview modal,
// cleanup older previews, in case they're still open
document.querySelectorAll('span.ar5iv-bibitem-preview').forEach(function(node) {
node.remove();
})
// Create the preview
preview = document.createElement('span');
preview.setAttribute('class','ar5iv-bibitem-preview');
let target = document.getElementById(this.getAttribute('href').slice(1));
target.childNodes.forEach(function (child) {
preview.append(child.cloneNode(true));
});
let close\_x = document.createElement('button');
close\_x.setAttribute("aria-label","Close modal for bibliography item preview");
close\_x.textContent = "×";
close\_x.setAttribute('class', 'ar5iv-button-close-preview');
close\_x.setAttribute('onclick','this.parentNode.remove()');
preview.append(close\_x);
preview.querySelectorAll('.ltx\_tag\_bibitem').forEach(function(node) {
node.remove();
});
cite.parentNode.insertBefore(preview, cite.nextSibling);
return;
}
// Global Document initialization:
// - assign the preview feature to all inline citation links
document.querySelectorAll(".ltx\_cite .ltx\_ref").forEach(function (link) {
link.addEventListener("click", clicked\_cite);
});
