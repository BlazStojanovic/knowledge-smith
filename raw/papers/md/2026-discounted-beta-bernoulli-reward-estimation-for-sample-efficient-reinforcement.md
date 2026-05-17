---
arxiv: '2603.18444'
authors:
- Haechan Kim
- Soohyun Ryu
- Gyouk Chu
- Doohyuk Jang
- Eunho Yang
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Discounted Beta--Bernoulli Reward Estimation for Sample-Efficient Reinforcement
  Learning with Verifiable Rewards
url: https://arxiv.org/abs/2603.18444
year: 2026
---

# Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards

Haechan Kim
  
Soohyun Ryu
  
Gyouk Chu
  
Doohyuk Jang
  
Eunho Yang

###### Abstract

Reinforcement learning with verifiable rewards (RLVR) has emerged as an effective post-training paradigm for improving the reasoning capabilities of large language models.
However, existing group-based RLVR methods often suffer from severe sample inefficiency.
This inefficiency stems from reliance on point estimation of rewards from a small number of rollouts, leading to high estimation variance, variance collapse, and ineffective utilization of generated responses.
In this work, we reformulate RLVR from a statistical estimation perspective by modeling rewards as samples drawn from a policy-induced distribution and casting advantage computation as the problem of estimating the reward distribution from finite data.
Building on this view, we propose Discounted Beta–Bernoulli (DBB) reward estimation, which leverages historical reward statistics for the non-stationary distribution.
Although biased, the resulting estimator exhibits reduced and stable variance, theoretically avoids estimated variance collapse, and achieves lower mean squared error than standard point estimation.
Extensive experiments across six in-distribution and three out-of-distribution reasoning benchmarks demonstrate that GRPO with DBB consistently outperforms naive GRPO, achieving average Acc@8 improvements of 3.22/2.42 points in-distribution and 12.49/6.92 points out-of-distribution on the 1.7B and 8B models, respectively, without additional computational cost or memory usage.

## 1 Introduction

!(/html/2603.18444/assets/x1.png)

Figure 1: Comparison between point estimation and DBB estimation. By trading a small bias for substantial variance reduction via shrinkage, DBB estimation achieves lower mean squared error. Compared to naive GRPO using point estimation, GRPO with DBB estimation consistently demonstrates superior performance across all benchmarks and both model scales.

RLVR (Lambert et al., [2025](#bib.bib3 "Tulu 3: pushing frontiers in open language model post-training")) has recently emerged as a key post-training paradigm for enhancing the complex reasoning capabilities of large language models (Plaat et al., [2025](#bib.bib14 "Multi-step reasoning with large language models, a survey")) and improving their performance on downstream tasks (Guo et al., [2025](#bib.bib1 "Deepseek-r1: incentivizing reasoning capability in llms via reinforcement learning"); Yang et al., [2025](#bib.bib2 "Qwen3 technical report"); Yu et al., [2025b](#bib.bib13 "RLPR: extrapolating rlvr to general domains without verifiers")).
Since most RLVR algorithms, including Group Relative Policy Optimization (GRPO; [Shao et al.](#bib.bib4 "Deepseekmath: pushing the limits of mathematical reasoning in open language models"), [2024](#bib.bib4 "Deepseekmath: pushing the limits of mathematical reasoning in open language models")), rely on group-based advantage estimation (Yu et al., [2025a](#bib.bib8 "DAPO: an open-source llm reinforcement learning system at scale"); Liu et al., [2025](#bib.bib7 "Understanding r1-zero-like training: a critical perspective"); Zheng et al., [2025a](#bib.bib5 "Group sequence policy optimization"); Zhao et al., [2025](#bib.bib6 "Geometric-mean policy optimization")), they require generating multiple responses per prompt, which can account for a large fraction of the overall training time, often approaching 50% (Le et al., [2025](#bib.bib9 "No prompt left behind: exploiting zero-variance prompts in llm reinforcement learning via entropy-guided advantage shaping")).
Despite this substantial computational cost, many existing group-based RLVR algorithms fail to utilize information from the generated responses efficiently.

This sample inefficiency can be attributed to two fundamental characteristics of RLVR.
The first characteristic is that the variance of rewards can collapse to zero when all generated responses receive identical rewards under group-relative estimation.
This *variance collapse* issue not only wastes the computational cost of response generation but also eliminates meaningful training signals within the batch (Zhang et al., [2025](#bib.bib17 "Improving sampling efficiency in rlvr through adaptive rollout and response reuse")).
To address this issue, approaches such as dynamic sampling (Yu et al., [2025a](#bib.bib8 "DAPO: an open-source llm reinforcement learning system at scale")) and GRPO with Efficient Selective Rollout (GRESO; [Zheng et al.](#bib.bib10 "Act only when it pays: efficient reinforcement learning for llm reasoning via selective rollouts"), [2025b](#bib.bib10 "Act only when it pays: efficient reinforcement learning for llm reasoning via selective rollouts")) have been proposed. However, dynamic sampling requires several times more rollout cost, and GRESO cannot fully resolve the problem since it probabilistically filters prompts that are likely to result in variance collapse.

The second characteristic is that, due to the on-policy nature of RLVR algorithms, information from all generated responses is discarded after a single gradient update.
Recent replay-based methods (Li et al., [2025](#bib.bib11 "RePO: replay-enhanced policy optimization"); Zhan et al., [2025](#bib.bib12 "ExGRPO: learning to reason from experience")) attempt to reuse rollouts generated from the training history.
However, unbiased reuse of off-policy data requires importance sampling, which in turn necessitates storing all token-level probabilities under historical policies and performing additional forward passes on the current policy.
As a result, replay-based approaches introduce substantial GPU memory overhead and additional computational cost, limiting their practical scalability.

These two sources of sample inefficiency primarily arise because most group-based RLVR algorithms rely on point estimators that consider only the rewards from the current rollout group when estimating the underlying reward distribution for advantage estimation.
While such estimators can be reliable when the number of sampled responses per group is sufficiently large (Casella and Berger, [2024](#bib.bib16 "Statistical inference")), practical RLVR settings often operate in low-sample regimes due to computational constraints.
In these settings, point estimators remain unbiased but suffer from high variance, which can lead to variance collapse and unstable training dynamics.

To address this limitation, we adopt a statistical perspective that models rewards as stochastic outcomes drawn from a distribution induced by the policy.
Instead of relying on point estimation, we employ a Bayesian framework (Gelman et al., [1995](#bib.bib15 "Bayesian data analysis")) to model uncertainty explicitly and incorporate temporal reward dynamics.
Within this framework, we propose the Discounted Beta–Bernoulli (DBB) reward estimation, which tracks the evolving reward distribution by discounting historical observations.
While DBB introduces a small bias, it significantly reduces variance and avoids variance collapse, thereby providing more stable and informative training signals ([Figure 1](#S1.F1 "In 1 Introduction ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards")).
Empirically, DBB yields a lower mean-squared error with respect to the true reward distribution than point estimation in low-sample scenarios.

We evaluate GRPO with the DBB reward estimator (GRPO-DBB) on two model scales, Qwen3-1.7B-Base and Qwen3-8B-Base (Yang et al., [2025](#bib.bib2 "Qwen3 technical report")), across six in-distribution mathematical reasoning benchmarks, including MATH500, Minerva, AIME24/25, AMC24, and OlympiadBench.
GRPO-DBB consistently outperforms naive GRPO and other baselines across all benchmarks and model sizes.
Specifically, compared to GRPO, GRPO-DBB achieves average Acc@8 improvements of 3.22 and 2.42 points on Qwen3-1.7B-Base and Qwen3-8B-Base, respectively.
Furthermore, on three out-of-distribution reasoning benchmarks such as MMLU-Pro, GPQA-Diamond, and Big-Bench Hard, GRPO-DBB yields average Acc@8 gains of 12.49 and 6.92 points over GRPO on 1.7B and 8B models, respectively.

## 2 Preliminaries

We briefly introduce RLVR and its commonly used baseline, GRPO, to establish the foundation for our method.

### 2.1 Reinforcement Learning with Verifiable Rewards

Given a prompt q∼𝒟q\sim\mathcal{D} from the training dataset and a policy πθ\pi\_{\theta}, a generated response is o∼πθ(⋅∣q)o\sim\pi\_{\theta}(\cdot\mid q).
In RLVR, the reward function r​(⋅)r(\cdot) is typically defined as a binary signal indicating whether the response contains a correct answer.

|  |  |  |  |
| --- | --- | --- | --- |
|  | r​(o,q)={1,if o contains the answer of q0,otherwise.r(o,q)=\begin{cases}1,&\text{if $o$ contains the answer of $q$}\\ 0,&\text{otherwise}.\end{cases} |  | (1) |

### 2.2 Group Relative Policy Optimization (GRPO)

GRPO (Shao et al., [2024](#bib.bib4 "Deepseekmath: pushing the limits of mathematical reasoning in open language models")) is a foundational baseline for RLVR that eliminates the need for an explicit value model and Generalized Advantage Estimation (Schulman et al., [2017](#bib.bib20 "Proximal policy optimization algorithms")). Instead, it relies on group-relative normalization to estimate advantages.

For a given prompt qq, GRPO samples NN independent responses {oi}i=1N\{o\_{i}\}\_{i=1}^{N} from the old policy πθold\pi\_{\theta\_{\mathrm{old}}} and computes advantages by normalizing the corresponding rewards {r​(oi,q)}i=1N\{r(o\_{i},q)\}\_{i=1}^{N} within the group:

|  |  |  |  |
| --- | --- | --- | --- |
|  | A^i=r​(oi,q)−μσ,\hat{A}\_{i}=\frac{r(o\_{i},q)-\mu}{\sigma}, |  | (2) |

where μ=∑i=1Nr​(oi,q)N\mu=\frac{\sum\_{i=1}^{N}r(o\_{i},q)}{N} and σ2=∑i=1N(r​(oi,q)−μ)2N−1\sigma^{2}=\frac{\sum\_{i=1}^{N}\left(r(o\_{i},q)-\mu\right)^{2}}{N-1}.
The resulting advantage A^i\hat{A}\_{i} is broadcast to all tokens in the response, i.e., A^i,t=A^i\hat{A}\_{i,t}=\hat{A}\_{i} for all token positions tt.

The policy is then updated by maximizing a clipped surrogate objective based on estimated advantages:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 𝒥GRPO(θ)=𝔼q∼𝒟,{oi}i=1N∼πθold(⋅∣q)[1N∑i=1N1|oi|∑t=1|oi|\displaystyle\mathcal{J}\_{\mathrm{GRPO}}(\theta)=\mathbb{E}\_{q\sim\mathcal{D},\,\{o\_{i}\}\_{i=1}^{N}\sim\pi\_{\theta\_{\mathrm{old}}}(\cdot\mid q)}\Bigg[\frac{1}{N}\sum\_{i=1}^{N}\frac{1}{|o\_{i}|}\sum\_{t=1}^{|o\_{i}|} |  | (3) |
|  |  | min(wi,t(θ)A^i,t,clip(wi,t(θ),1−ϵ,1+ϵ)A^i,t)],\displaystyle\min\Big(w\_{i,t}(\theta)\,\hat{A}\_{i,t},\mathrm{clip}\!\big(w\_{i,t}(\theta),1-\epsilon,1+\epsilon\big)\,\hat{A}\_{i,t}\Big)\Bigg], |  |

where ϵ\epsilon is the clipping hyperparameter and
wi,t​(θ)=πθ​(oi,t∣q,oi,<t)πθold​(oi,t∣q,oi,<t)w\_{i,t}(\theta)=\frac{\pi\_{\theta}(o\_{i,t}\mid q,o\_{i,<t})}{\pi\_{\theta\_{\mathrm{old}}}(o\_{i,t}\mid q,o\_{i,<t})} denotes the per-token importance weight. Following DAPO (Yu et al., [2025a](#bib.bib8 "DAPO: an open-source llm reinforcement learning system at scale")), we omit the KL divergence term between the online and reference policies, as it may restrict exploration.

## 3 Method

Many group-based RLVR methods (Shao et al., [2024](#bib.bib4 "Deepseekmath: pushing the limits of mathematical reasoning in open language models"); Liu et al., [2025](#bib.bib7 "Understanding r1-zero-like training: a critical perspective"); Xie et al., [2025](#bib.bib18 "Unlocking exploration in rlvr: uncertainty-aware advantage shaping for deeper reasoning"); Le et al., [2025](#bib.bib9 "No prompt left behind: exploiting zero-variance prompts in llm reinforcement learning via entropy-guided advantage shaping")) primarily focus on the design of advantage estimators, while paying comparatively little attention to the more fundamental problem of reward estimation that underlies advantage computation. In practice, computational and memory constraints limit the number of sampled responses per prompt, making accurate reward estimation inherently difficult.

In contrast, we introduce a new perspective that reformulates RLVR through the lens of statistical estimation ([Section 3.1](#S3.SS1 "3.1 Reward Estimation as Distributional Inference ‣ 3 Method ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards")). Specifically, we view the reward not as a deterministic signal, but as a distribution induced by the policy, and frame advantage computation as a problem of estimating this distribution from finite samples. Building on this reformulation, we propose Discounted Beta–Bernoulli (DBB) reward estimation for RLVR, which estimates the non-stationary reward distribution by leveraging historical rewards ([Section 3.2](#S3.SS2 "3.2 Discounted Beta–Bernoulli Reward Estimation ‣ 3 Method ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards")). Despite introducing bias, DBB reduces estimator variance and achieves lower MSE. Moreover, it fundamentally prevents variance collapse and preserves informative training signals ([Section 3.3](#S3.SS3 "3.3 Mean Squared Error of the DBB estimator ‣ 3 Method ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards")).

### 3.1 Reward Estimation as Distributional Inference

For a given prompt q∼𝒟q\sim\mathcal{D} and training step η\eta with corresponding epoch τ\tau, each response generated by the policy πθη,old\pi\_{\theta\_{\eta,\mathrm{old}}} produces a binary reward indicating whether the response is correct. Under our formulation, this reward is naturally modeled as a Bernoulli random variable

|  |  |  |  |
| --- | --- | --- | --- |
|  | Xτ,i∼Bernoulli​(pτ),X\_{\tau,i}\sim\mathrm{Bernoulli}\!\left(p\_{\tau}\right), |  | (4) |

where
pτ=ℙ​(Xτ,i=1∣q,πθη,old)p\_{\tau}=\mathbb{P}\!\left(X\_{\tau,i}=1\mid q,\pi\_{\theta\_{\eta,\mathrm{old}}}\right)
denotes the probability of obtaining a correct response.

Within this framework, group-based RLVR algorithms such as GRPO and Dr.GRPO can be interpreted as implicitly estimating the reward distribution from a finite set of rollout samples. Given NN observed rewards {Xτ,i}i=1N\{X\_{\tau,i}\}\_{i=1}^{N}, existing approaches rely on point estimation of pτp\_{\tau} via the empirical mean

|  |  |  |  |
| --- | --- | --- | --- |
|  | p^τpt=1N​∑i=1NXτ,i.\hat{p}\_{\tau}^{\mathrm{pt}}=\frac{1}{N}\sum\_{i=1}^{N}X\_{\tau,i}. |  | (5) |

The corresponding reward variance is then estimated using the sample variance,

|  |  |  |  |
| --- | --- | --- | --- |
|  | Var^pt​(Xτ)=∑i=1N(Xτ,i−p^τpt)2N−1=N⋅p^τpt​(1−p^τpt)N−1.\displaystyle\widehat{\mathrm{Var}}^{\mathrm{pt}}(X\_{\tau})=\;\frac{\sum\_{i=1}^{N}\left(X\_{\tau,i}-\hat{p}\_{\tau}^{\mathrm{pt}}\right)^{2}}{N-1}=\frac{N\cdot\hat{p}\_{\tau}^{\mathrm{pt}}\bigl(1-\hat{p}\_{\tau}^{\mathrm{pt}}\bigr)}{N-1}\,. |  | (6) |

Since the estimated variance Var^pt\widehat{\mathrm{Var}}^{\mathrm{pt}} is determined by the point estimator (mean) p^τpt\hat{p}\_{\tau}^{\mathrm{pt}}, it suffices to consider only the point estimator in our analysis.
Concretely, the point estimator p^τpt\hat{p}\_{\tau}^{\mathrm{pt}} has expectation and variance as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[p^τpt∣pτ]=pτ,\mathbb{E}[\hat{p}\_{\tau}^{\mathrm{pt}}\mid p\_{\tau}]=p\_{\tau}, |  | (7) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | Var​(p^τpt∣pτ)=pτ​(1−pτ)N.\mathrm{Var}(\hat{p}\_{\tau}^{\mathrm{pt}}\mid p\_{\tau})=\frac{p\_{\tau}(1-p\_{\tau})}{N}. |  | (8) |

While the point estimator is unbiased, its variance grows as the number of rollouts NN decreases. As a result, individual rollout outcomes can exert disproportionate influence on the estimate, causing the estimator to fluctuate widely. Moreover, the estimated variance can collapse to zero when all sampled rewards are identical. Since group-relative methods compute advantages directly from these reward estimates, such estimation noise naturally propagates to the advantage signal, leading to unstable policy updates.

### 3.2 Discounted Beta–Bernoulli Reward Estimation

To estimate the Bernoulli reward distribution beyond empirical averaging, we adopt a well-established Bayesian perspective that leverages the conjugacy between the Beta prior and the Bernoulli likelihood. In this setting, the reward probability is modeled using a Beta prior and updated via a Beta posterior after observing rollout outcomes, leading to the naive Beta–Bernoulli model defined below.

###### Definition 1 (Beta–Bernoulli Reward Model).

To model uncertainty in the Bernoulli reward distribution, we place a Beta prior over the reward probability:

|  |  |  |  |
| --- | --- | --- | --- |
|  | pτ∼Beta​(ατ,βτ),p\_{\tau}\sim\mathrm{Beta}(\alpha\_{\tau},\beta\_{\tau}), |  | (9) |

where (ατ,βτ)(\alpha\_{\tau},\beta\_{\tau}) denote the posterior parameters for a prompt at epoch τ\tau, with initialization α0=β0=1\alpha\_{0}=\beta\_{0}=1.

Given NN rollouts with SτS\_{\tau} successes, the posterior distribution is given by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ατ=ατ−1+Sτ,βτ=βτ−1+N−Sτ.\alpha\_{\tau}=\alpha\_{\tau-1}+S\_{\tau},\qquad\beta\_{\tau}=\beta\_{\tau-1}+N-S\_{\tau}. |  | (10) |

However, the above model assumes a stationary reward distribution. In RLVR, this assumption does not hold: due to the on-policy nature of training, the policy πθ\pi\_{\theta} evolves over time, inducing a non-stationary reward distribution. As a result, historical observations may become outdated and should not be weighted equally with recent rollouts.

Algorithm 1  Discounted Beta–Bernoulli Reward Estimation

0: LLM policy πθ\pi\_{\theta}, training dataset 𝒟\mathcal{D}

1: Initialize (α0q,β0q)←(1,1)∀q∈𝒟(\alpha^{q}\_{0},\beta^{q}\_{0})\leftarrow(1,1)\quad\forall q\in\mathcal{D}

2: for epoch​τ=1,…,I\text{epoch}\ \tau=1,\dots,I do

3:  for iteration=1,…,M\text{iteration}=1,\dots,M do

4:   Sample a minibatch 𝒟b⊂𝒟\mathcal{D}\_{b}\subset\mathcal{D}

5:   Set old policy πθold←πθ\pi\_{\theta\_{\mathrm{old}}}\leftarrow\pi\_{\theta}

6:   for each prompt q∈𝒟bq\in\mathcal{D}\_{b} do

7:    Sample outputs {oi}i=1N∼i.i.d.πθold(⋅∣q)\{o\_{i}\}\_{i=1}^{N}\overset{\text{i.i.d.}}{\sim}\pi\_{\theta\_{\mathrm{old}}}(\cdot\mid q)

8:    Compute rewards {Xi=r​(oi,q)}i=1N\{X\_{i}=r(o\_{i},q)\}\_{i=1}^{N}

9:    Update posterior parameters:

10:     ατq←λ​ατ−1q+∑i=1NXi,βτq←λ​βτ−1q+N−∑i=1NXi\begin{aligned} \alpha^{q}\_{\tau}&\leftarrow\lambda\alpha^{q}\_{\tau-1}+\sum\_{i=1}^{N}X\_{i},\\
\beta^{q}\_{\tau}&\leftarrow\lambda\beta^{q}\_{\tau-1}+N-\sum\_{i=1}^{N}X\_{i}\end{aligned}

11:    Compute advantage A^i\hat{A}\_{i} with (ατq,βτq)(\alpha^{q}\_{\tau},\beta^{q}\_{\tau}) and XiX\_{i}

12:   end for

13:   for update=1,…,U\text{update}=1,\dots,U do

14:    Update πθ\pi\_{\theta} by maximizing the objective (Eq. [3](#S2.E3 "Equation 3 ‣ 2.2 Group Relative Policy Optimization (GRPO) ‣ 2 Preliminaries ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"))

15:   end for

16:  end for

17: end for

To address this challenge, we introduce the Discounted Beta–Bernoulli (DBB) reward model, which gradually discounts past information to estimate the reward distribution accurately.

###### Definition 2 (Discounted Beta–Bernoulli Reward Model).

Given posterior parameters (ατ−1,βτ−1)(\alpha\_{\tau-1},\beta\_{\tau-1}) and NN rollouts with SτS\_{\tau} successes at epoch τ\tau, the update is defined as

|  |  |  |  |
| --- | --- | --- | --- |
|  | ατ=λ​ατ−1+Sτ,βτ=λ​βτ−1+N−Sτ,\alpha\_{\tau}=\lambda\alpha\_{\tau-1}+S\_{\tau},\qquad\beta\_{\tau}=\lambda\beta\_{\tau-1}+N-S\_{\tau}, |  | (11) |

where the discount factor λ∈(0,1]\lambda\in(0,1] controls the influence of historical observations.

Under the DBB reward model, we estimate the mean and variance of the Bernoulli reward distribution as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | p^τdbb=ατατ+βτ,\hat{p}\_{\tau}^{\mathrm{dbb}}=\frac{\alpha\_{\tau}}{\alpha\_{\tau}+\beta\_{\tau}}, |  | (12) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | Var^dbb​(Xτ∣ατ,βτ)=ατ​βτ(ατ+βτ)2.\widehat{\mathrm{Var}}^{\mathrm{dbb}}(X\_{\tau}\mid\alpha\_{\tau},\beta\_{\tau})=\frac{\alpha\_{\tau}\beta\_{\tau}}{(\alpha\_{\tau}+\beta\_{\tau})^{2}}. |  | (13) |

To understand the statistical behavior of the DBB estimator, we analyze its expectation and variance:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[p^τdbb∣pτ,ατ−1,βτ−1]=w​μτ−1+(1−w)​pτ,\mathbb{E}\!\left[\hat{p}\_{\tau}^{\mathrm{dbb}}\mid p\_{\tau},\alpha\_{\tau-1},\beta\_{\tau-1}\right]=w\,\mu\_{\tau-1}+(1-w)\,p\_{\tau}, |  | (14) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | Var​(p^τdbb∣pτ,ατ−1,βτ−1)=(1−w)2​pτ​(1−pτ)N,\mathrm{Var}\!\left(\hat{p}\_{\tau}^{\mathrm{dbb}}\mid p\_{\tau},\alpha\_{\tau-1},\beta\_{\tau-1}\right)=(1-w)^{2}\,\frac{p\_{\tau}(1-p\_{\tau})}{N}, |  | (15) |

where μτ−1=ατ−1ατ−1+βτ−1\mu\_{\tau-1}=\frac{\alpha\_{\tau-1}}{\alpha\_{\tau-1}+\beta\_{\tau-1}} denotes the historical posterior mean and
w=λ​(ατ−1+βτ−1)λ​(ατ−1+βτ−1)+Nw=\frac{\lambda(\alpha\_{\tau-1}+\beta\_{\tau-1})}{\lambda(\alpha\_{\tau-1}+\beta\_{\tau-1})+N} controls the contribution of past observations.

These show that the DBB estimator introduces bias through shrinkage toward the historical mean, but substantially reduces variance relative to the point estimator ([Equation 8](#S3.E8 "In 3.1 Reward Estimation as Distributional Inference ‣ 3 Method ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards")). Importantly, unlike estimated variance based on small-sample rollouts, the posterior variance in DBB cannot collapse to zero, ensuring stable and informative reward signals throughout training. Algorithm [1](#alg1 "Algorithm 1 ‣ 3.2 Discounted Beta–Bernoulli Reward Estimation ‣ 3 Method ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards") summarizes the overall training procedure of the group-based RLVR algorithm with DBB.

### 3.3 Mean Squared Error of the DBB estimator

To assess the effectiveness of our DBB estimator in predicting the reward distribution, we compute its mean squared error (MSE). To make the estimator’s dependence on the underlying reward probabilities {p1,…,pτ}\{p\_{1},\ldots,p\_{\tau}\}, we re-express its expectation and variance, originally formulated in terms of the posterior parameters (ατ−1,βτ−1)(\alpha\_{\tau-1},\beta\_{\tau-1}).

Based on the update defined in [Equation 11](#S3.E11 "In Definition 2 (Discounted Beta–Bernoulli Reward Model). ‣ 3.2 Discounted Beta–Bernoulli Reward Estimation ‣ 3 Method ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"), posterior parameters can be expressed as

|  |  |  |
| --- | --- | --- |
|  | ατ=λτ​α0+∑k=1τλτ−k​Sk,βτ=λτ​β0+∑k=1τλτ−k​(N−Sk).\alpha\_{\tau}=\lambda^{\tau}\alpha\_{0}+\sum\_{k=1}^{\tau}\lambda^{\tau-k}S\_{k},\;\beta\_{\tau}=\lambda^{\tau}\beta\_{0}+\sum\_{k=1}^{\tau}\lambda^{\tau-k}(N-S\_{k}). |  |

For convenience, define the sum of the posterior parameters as HτH\_{\tau}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Hτ=ατ+βτ=λτ​(α0+β0)+N​∑k=1τλτ−k.H\_{\tau}=\alpha\_{\tau}+\beta\_{\tau}=\lambda^{\tau}(\alpha\_{0}+\beta\_{0})+N\sum\_{k=1}^{\tau}\lambda^{\tau-k}. |  | (16) |

Under our formulation ([Equation 4](#S3.E4 "In 3.1 Reward Estimation as Distributional Inference ‣ 3 Method ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards")), the number of successful rollouts at epoch kk satisfies Sk∼Binomial​(N,pk)S\_{k}\sim\mathrm{Binomial}(N,p\_{k}). Given this, the expectation and variance of the DBB estimator can be written as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[p^τdbb∣p1:τ]=∑k=0τck​pk,\mathbb{E}\!\left[\hat{p}\_{\tau}^{\mathrm{dbb}}\mid p\_{1:\tau}\right]=\sum\_{k=0}^{\tau}c\_{k}p\_{k}, |  | (17) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | Var​(p^τdbb∣p1:τ)=∑k=1τλ2​(τ−k)​N​pk​(1−pk)Hτ2,\mathrm{Var}\!\left(\hat{p}\_{\tau}^{\mathrm{dbb}}\mid p\_{1:\tau}\right)=\frac{\sum\_{k=1}^{\tau}\lambda^{2(\tau-k)}Np\_{k}(1-p\_{k})}{H\_{\tau}^{2}}, |  | (18) |

where p0=α0α0+β0p\_{0}=\frac{\alpha\_{0}}{\alpha\_{0}+\beta\_{0}} is the reward probability of the initial prior, and the weights c0=λτ​(α0+β0)Hτc\_{0}=\frac{\lambda^{\tau}(\alpha\_{0}+\beta\_{0})}{H\_{\tau}} and ck=N​λτ−kHτc\_{k}=\frac{N\lambda^{\tau-k}}{H\_{\tau}} are constants. A detailed derivation is provided in Appendix [B](#A2 "Appendix B Derivation of the Mean, Variance, and MSE of the DBB Estimator ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").

Combining Equations ([17](#S3.E17 "Equation 17 ‣ 3.3 Mean Squared Error of the DBB estimator ‣ 3 Method ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards")) and ([18](#S3.E18 "Equation 18 ‣ 3.3 Mean Squared Error of the DBB estimator ‣ 3 Method ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards")), the MSE of the DBB estimator under non-stationary rewards is

|  |  |  |  |
| --- | --- | --- | --- |
|  | MSE​(p^τdbb∣p1:τ)=(𝔼​[p^τdbb∣p1:τ]−pτ)2+Var​(p^τdbb∣p1:τ).\mathrm{MSE}\!\left(\hat{p}\_{\tau}^{\mathrm{dbb}}\mid p\_{1:\tau}\right)=\left(\mathbb{E}\!\left[\hat{p}\_{\tau}^{\mathrm{dbb}}\mid p\_{1:\tau}\right]-p\_{\tau}\right)^{2}+\mathrm{Var}\!\left(\hat{p}\_{\tau}^{\mathrm{dbb}}\mid p\_{1:\tau}\right). |  | (19) |

In contrast, the MSE of the point estimator depends only on the current reward distribution

|  |  |  |  |
| --- | --- | --- | --- |
|  | MSE​(p^τpt∣pτ)=pτ​(1−pτ)N,\mathrm{MSE}\!\left(\hat{p}\_{\tau}^{\mathrm{pt}}\mid p\_{\tau}\right)=\frac{p\_{\tau}(1-p\_{\tau})}{N}, |  | (20) |

and is therefore more prone to high variance, especially in low-sample settings.

The bias of the DBB estimator becomes smaller when the underlying reward probabilities evolve gradually, as past values {pk}k<τ\{p\_{k}\}\_{k<\tau} remain close to the current pτp\_{\tau}.
In contrast, the variance depends on both the discount factor λ\lambda and the number of rollouts NN.
On the other hand, the point estimator relies solely on pτp\_{\tau} and is therefore more susceptible to high variance, particularly in low-sample settings.
Considering these factors, we empirically evaluate the resulting MSE and present the findings in Section [5.2](#S5.SS2 "5.2 Ablation Study on 𝜆 ‣ 5 Analysis & Discussion ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").

## 4 Experiments

We empirically demonstrate that HBB estimation is effective in the RLVR setting. Section [4.1](#S4.SS1 "4.1 Experimental Settings ‣ 4 Experiments ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards") describes the models, datasets, baselines, and detailed setup used in our experiments, while Section [4.2](#S4.SS2 "4.2 Main Results ‣ 4 Experiments ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards") presents the experimental results.

### 4.1 Experimental Settings

#### Models & Datasets.

We conduct experiments on two model sizes, Qwen3-1.7B-Base and Qwen3-8B-Base (Yang et al., [2025](#bib.bib2 "Qwen3 technical report")), to investigate the effect of model scales.
For both model scales, we use DAPO-Math-17k (Yu et al., [2025a](#bib.bib8 "DAPO: an open-source llm reinforcement learning system at scale")) as the training dataset.
As in-distribution evaluation benchmarks for mathematical reasoning, we consider six widely used datasets: MATH500 (Hendrycks et al., [2021](#bib.bib21 "Measuring mathematical problem solving with the math dataset")), Minerva (Lewkowycz et al., [2022](#bib.bib26 "Solving quantitative reasoning problems with language models")), AIME24/25, AMC24 (Li et al., [2024](#bib.bib27 "Numinamath: the largest public dataset in ai4maths with 860k pairs of competition math problems and solutions")), and OlympiadBench (Olympiad; [He et al.](#bib.bib22 "OlympiadBench: a challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems"), [2024](#bib.bib22 "OlympiadBench: a challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems")).
To further assess out-of-distribution generalization, we evaluate performance on MMLU-Pro (Wang et al., [2024](#bib.bib23 "MMLU-pro: a more robust and challenging multi-task language understanding benchmark")), GPQA-Diamond (GPQA-D; [Rein et al.](#bib.bib24 "GPQA: a graduate-level google-proof q&a benchmark"), [2023](#bib.bib24 "GPQA: a graduate-level google-proof q&a benchmark")), and Big-Bench Hard (BBH; [Suzgun et al.](#bib.bib25 "Challenging big-bench tasks and whether chain-of-thought can solve them"), [2022](#bib.bib25 "Challenging big-bench tasks and whether chain-of-thought can solve them")).

#### Baselines.

The DBB reward estimation aims to improve reward distribution estimation by partially leveraging historical information.
To isolate the effect of DBB, we first compare GRPO with the DBB estimator (GRPO-DBB) and naive GRPO, which relies on point estimation.
In addition, to compare against methods that more fully exploit historical information, we include RePO as a representative method in this category.

#### Training & Evaluation Setups.

All training experiments are conducted using the verl111<https://github.com/verl-project/verl> framework, and Math-Verify222<https://github.com/huggingface/Math-Verify> is employed to extract and verify final answers.
Experiments with Qwen3-1.7B-Base and Qwen3-8B-Base are conducted on 4×\timesH200 and 8×\timesH200 GPUs, respectively, except for RePO.
Due to the requirement for additional GPU memory, RePO is trained on 8×\timesH200 GPUs for both model scales.
The rollout batch size is set to 128, and the gradient update batch size is set to 64.
We sample 8 responses per query from the on-policy model and train the model for a total of four epochs.
For evaluation, we sample 8 responses per query for all benchmarks and report Avg@8 as the evaluation metric.
The training setup details for each baseline and the sampling parameters used for validation and evaluation are also provided in Appendix [C](#A3 "Appendix C Implementation Details ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").

### 4.2 Main Results

Table 1: 
In-distribution (ID) and out-of-distribution (OOD) evaluation results trained on DAPO-Math-17k.
GRPO with the DBB process (GRPO-DBB) consistently outperforms naive GRPO across all benchmarks and model scales.
Δ\Delta denotes the absolute Acc@8 improvement of GRPO-DBB over GRPO.

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | In-Distribution | | | | | | | Out-of-Distribution | | | |
| Method | MATH500 | Minerva | AIME24 | AIME25 | AMC24 | Olympiad | Avg. | MMLU-Pro | GPQA-D | BBH | Avg. |
| Qwen3-8B-Base trained with DAPO-Math-17k | | | | | | | | | | | |
| GRPO | 88.05 | 39.25 | 30.00 | 26.67 | 56.39 | 54.82 | 49.20 | 50.02 | 42.93 | 59.18 | 50.71 |
| RePO | 86.20 | 35.71 | 27.92 | 22.92 | 56.11 | 52.61 | 46.91 | 53.15 | 43.93 | 42.96 | 46.68 |
| GRPO-DBB | 88.92 | 39.48 | 34.17 | 30.83 | 60.00 | 56.34 | 51.62 | 63.12 | 46.46 | 63.32 | 57.63 |
| Δ\Delta w.r.t. GRPO | +0.87 | +0.23 | +4.17 | +4.16 | +3.61 | +1.52 | +2.42 | +13.10 | +3.53 | +4.14 | +6.92 |
| Qwen3-1.7B-Base trained with DAPO-Math-17k | | | | | | | | | | | |
| GRPO | 68.37 | 25.74 | 8.75 | 4.58 | 23.61 | 29.41 | 26.74 | 33.72 | 20.70 | 12.71 | 22.38 |
| RePO | 67.87 | 23.76 | 6.67 | 6.67 | 21.67 | 29.17 | 25.97 | 35.49 | 26.26 | 30.95 | 30.90 |
| GRPO-DBB | 71.95 | 26.31 | 14.17 | 7.08 | 26.94 | 33.29 | 29.96 | 40.83 | 29.80 | 33.97 | 34.87 |
| Δ\Delta w.r.t. GRPO | +3.58 | +0.57 | +5.42 | +2.50 | +3.33 | +3.88 | +3.22 | +7.11 | +9.10 | +21.26 | +12.49 |

Table [1](#S4.T1 "Table 1 ‣ 4.2 Main Results ‣ 4 Experiments ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards") reports Acc@8 results across six mathematical reasoning in-distribution benchmarks and three general reasoning out-of-distribution (OOD) benchmarks for three different RLVR algorithms.
For GRPO-DBB, the discounting factor λ\lambda is set to 0.50.5 for Qwen3-1.7B-Base and 0.750.75 for Qwen3-8B-Base; these values are empirically selected based on performance trends under varying λ\lambda.
Further details are provided in Section [5.2](#S5.SS2 "5.2 Ablation Study on 𝜆 ‣ 5 Analysis & Discussion ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").

For in-distribution evaluation, GRPO-DBB consistently outperforms both GRPO and RePO across all benchmarks and model scales. Compared to GRPO, Acc@8 improves by an average of 3.22 points for Qwen3-1.7B-Base and 2.42 points for Qwen3-8B-Base.
This indicates that, in the RLVR setting, replacing GRPO’s point estimator with the DBB estimator is more effective for in-distribution downstream tasks. When compared to RePO, GRPO-DBB achieves improvements of 3.99 and 4.71 points for Qwen3-1.7B-Base and Qwen3-8B-Base, respectively. Although RePO leverages diverse forms of historical information, including tokens, token probabilities, and rewards, GRPO-DBB relies solely on rewards. Despite this, its strong downstream performance suggests that the DBB estimator utilizes historical information both efficiently and effectively.

For out-of-distribution evaluation, GRPO-DBB consistently outperforms both baselines across all OOD benchmarks and model scales. In particular, it achieves substantial gains over GRPO, with improvements of 12.49 and 6.92 points for Qwen3-1.7B-Base and Qwen3-8B-Base, respectively. These results demonstrate that the DBB estimation provides a strong advantage in improving generalization, which can be attributed to the characteristic of the Bayesian approach to account for uncertainty.

Table 2: Effect of the discount factor λ\lambda of GRPO-DBB on in-distribution Acc@8 performance.
Qwen3-8B-Base achieves the best performance at λ=0.75\lambda=0.75, while Qwen3-1.7B-Base performs best at λ=0.5\lambda=0.5.

| Method | MATH500 | Minerva | AIME24 | AIME25 | AMC24 | Olympiad | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B-Base trained with DAPO 17k | | | | | | | |
| λ=1.0\lambda=1.0 | 89.08 | 37.91 | 33.33 | 26.25 | 59.44 | 55.19 | 50.20 |
| λ=0.75\lambda=0.75 | 88.92 | 39.48 | 34.17 | 30.83 | 60.00 | 56.34 | 51.62 |
| λ=0.5\lambda=0.5 | 89.35 | 38.56 | 33.33 | 27.08 | 59.17 | 56.19 | 50.61 |
| λ=0.25\lambda=0.25 | 88.82 | 39.98 | 30.83 | 25.83 | 60.83 | 56.34 | 50.44 |
| GRPO | 88.05 | 39.25 | 30.00 | 26.67 | 56.39 | 54.82 | 49.20 |
| Qwen3-1.7B-Base trained with DAPO 17k | | | | | | | |
| λ=1.0\lambda=1.0 | 67.30 | 25.55 | 9.58 | 3.33 | 23.33 | 28.65 | 26.29 |
| λ=0.75\lambda=0.75 | 72.28 | 25.60 | 11.25 | 7.50 | 26.39 | 32.68 | 29.28 |
| λ=0.5\lambda=0.5 | 71.95 | 26.31 | 14.17 | 7.08 | 26.94 | 33.29 | 29.96 |
| λ=0.25\lambda=0.25 | 72.22 | 25.97 | 14.17 | 6.25 | 22.50 | 32.88 | 29.00 |
| GRPO | 68.37 | 25.74 | 8.75 | 4.58 | 23.61 | 29.41 | 26.74 |

## 5 Analysis & Discussion

In this section, we provide an in-depth analysis of the training behavior and estimation properties of GRPO with the DBB reward estimator.
We first examine training dynamics to understand how DBB influences optimization stability, exploration behavior, and reward progression over time ([Section 5.1](#S5.SS1 "5.1 Training Dynamics ‣ 5 Analysis & Discussion ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards")).
We then conduct ablation studies on the discount factor λ\lambda to clarify its role in adapting to different learning regimes ([Section 5.2](#S5.SS2 "5.2 Ablation Study on 𝜆 ‣ 5 Analysis & Discussion ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards")).
Next, we empirically analyze the mean squared error (MSE) of DBB relative to point estimation, connecting estimation accuracy to downstream performance ([Section 5.3](#S5.SS3 "5.3 Analysis of MSE ‣ 5 Analysis & Discussion ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards")).
Finally, we demonstrate that DBB can be readily integrated into alternative advantage formulations, highlighting its generality beyond GRPO ([Section 5.4](#S5.SS4 "5.4 Evaluation on Alternative Advantage Formulations ‣ 5 Analysis & Discussion ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards")).

### 5.1 Training Dynamics

!(/html/2603.18444/assets/x2.png)

Figure 2: Training dynamics of naive GRPO and GRPO with the DBB estimator (GRPO-DBB) on Qwen3-1.7B-Base (top) and Qwen3-8B-Base (bottom). GRPO-DBB achieves higher validation Acc@8 and training rewards, while maintaining longer responses with controlled entropy compared to GRPO, indicating more stable exploration during training.

Figure [2](#S5.F2 "Figure 2 ‣ 5.1 Training Dynamics ‣ 5 Analysis & Discussion ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards") illustrates the validation accuracy and training dynamics of GRPO-DBB and GRPO during training.
For GRPO-DBB, the discounting factor λ\lambda is set to 0.50.5 for Qwen3-1.7B-Base and 0.750.75 for Qwen3-8B-Base.
For both model scales, GRPO-DBB achieves higher validation accuracy and training rewards than GRPO throughout most of the training process.

Examining the response length over the course of training, we observe that GRPO-DBB consistently produces longer responses than GRPO throughout training.
In contrast, the entropy under GRPO grows rapidly, indicating increasingly unstable exploration behavior.
Taken together, these observations suggest that GRPO-DBB attains higher rewards by pruning the breadth of response trajectories while allowing exploration to proceed along longer trajectories, corresponding to depth-wise exploration.

### 5.2 Ablation Study on λ\lambda

To examine the effect of the discount factor λ\lambda in the DBB estimation, we conduct training experiments of GRPO-DBB with λ=1.0,0.75,0.5,\lambda=1.0,0.75,0.5, and 0.250.25 for both model scales.

The hyperparameter λ\lambda controls the extent to which the estimation of the Bernoulli parameter pτp\_{\tau} for a given query at epoch τ\tau depends on historical rollout statistics, namely ατ−1\alpha\_{\tau-1} and βτ−1\beta\_{\tau-1}.
When the learning dynamics are fast and the discrepancy between pτp\_{\tau} and pτ−1p\_{\tau-1} is large, a smaller λ\lambda is generally more effective, as it places greater emphasis on recent observations and reduces estimation bias.
Conversely, when the learning dynamics are slower and more stationary, a larger λ\lambda tends to yield better performance by incorporating more historical information and thereby reducing estimation variance.

This behavior can be interpreted in conjunction with the training reward dynamics shown in Figure [2](#S5.F2 "Figure 2 ‣ 5.1 Training Dynamics ‣ 5 Analysis & Discussion ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
For Qwen3-1.7B-Base, the training reward exhibits sustained changes throughout the training process, indicating continuously evolving estimates of pτp\_{\tau}.
In contrast, for Qwen3-8B-Base, the training reward increases rapidly during the very early stage of training (within approximately the first 0.3 epoch) and remains relatively stationary for the remainder of training.
As a result, a smaller discounting factor (λ=0.5\lambda=0.5) is more effective for Qwen3-1.7B-Base, while a larger value (λ=0.75\lambda=0.75) better matches the slower and more gradual learning dynamics of Qwen3-8B-Base.

!(/html/2603.18444/assets/x3.png)

Figure 3: MSE as a function of the discount factor λ\lambda and the number of rollouts NN. The DBB estimator yields lower MSE than the point estimator across a wide range of λ\lambda, and it achieves lower MSE for rollout budgets up to N=16N=16 when λ=0.4\lambda=0.4.

### 5.3 Analysis of MSE

To evaluate how accurately the point estimator p^τpt\hat{p}\_{\tau}^{\mathrm{pt}} and the DBB estimator p^τdb\hat{p}\_{\tau}^{\mathrm{db}} approximate the Bernoulli parameter pτp\_{\tau} during training, we conduct an empirical estimation error analysis.
Since obtaining the exact ground-truth value of pτp\_{\tau} would require infinitely many samples, we instead approximate it using the empirical mean of rewards from a finite number of rollouts.

During GRPO training of Qwen3-1.7B-Base, when a randomly selected 10% subset of training prompts 𝒬τ\mathcal{Q}\_{\tau} appears in a rollout batch, we additionally sample 128 responses for each such prompt using the policy at that training step.
These additional samples are used exclusively for analysis and are not incorporated into training.
We then define the empirical mean of the resulting rewards as the reference value p~τ​(q)\tilde{p}\_{\tau}(q).

|  |  |  |  |
| --- | --- | --- | --- |
|  | p~τ​(q)≜1128​∑i=1128Xτ,i,\tilde{p}\_{\tau}(q)\;\triangleq\;\frac{1}{128}\sum\_{i=1}^{128}X\_{\tau,i}, |  | (21) |

where
oτ,i∼πθτ(⋅∣q)o\_{\tau,i}\sim\pi\_{\theta\_{\tau}}(\cdot\mid q)
and
Xτ,i=r​(oτ,i,q).X\_{\tau,i}=r(o\_{\tau,i},q).
Since training is conducted for a total of four epochs, this procedure yields a sequence of reference values {p~τ​(q)}τ=14\{\tilde{p}\_{\tau}(q)\}\_{\tau=1}^{4} for each query.

Using p~τ​(q)\tilde{p}\_{\tau}(q) as a surrogate for the true Bernoulli parameter, we measure the average MSE of the DBB estimator and the point estimator following Equation ([19](#S3.E19 "Equation 19 ‣ 3.3 Mean Squared Error of the DBB estimator ‣ 3 Method ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards")) and  ([20](#S3.E20 "Equation 20 ‣ 3.3 Mean Squared Error of the DBB estimator ‣ 3 Method ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards")), respectively.

#### Effect of λ\lambda on MSE

Since the value of the DBB estimator depends on the discount factor λ\lambda, we empirically analyze how the MSE varies as a function of λ\lambda.
As shown in Figure [3](#S5.F3 "Figure 3 ‣ 5.2 Ablation Study on 𝜆 ‣ 5 Analysis & Discussion ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards")(a), the MSE attains its minimum at λ=0.4\lambda=0.4, and for most values of λ\lambda—except for λ=0.95\lambda=0.95 and λ=1.0\lambda=1.0—the DBB estimator yields a lower average MSE than the point estimator.

This trend is closely aligned with downstream performance.
As reported in Table [2](#S4.T2 "Table 2 ‣ 4.2 Main Results ‣ 4 Experiments ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"), GRPO-DBB achieves the best performance at λ=0.5\lambda=0.5, while at λ=1.0\lambda=1.0 it performs slightly worse than GRPO with the point estimator.
These observations mirror the behavior of the MSE across different values of λ\lambda.
Taken together, the results indicate that more accurate reward distribution estimation is strongly correlated with improved RLVR performance.

#### Effect of NN on MSE

We further examine how the MSE varies as the number of rollouts NN increases.
Figure [2](#S4.T2 "Table 2 ‣ 4.2 Main Results ‣ 4 Experiments ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards")(b) reports the average MSE as a function of NN when λ=0.4\lambda=0.4.
The results show that the DBB estimator consistently outperforms the point estimator for NN up to 16.
This suggests that, in regimes where computational resource constraints necessitate small rollout budgets, the DBB estimator can serve as an effective alternative to point estimation.

### 5.4 Evaluation on Alternative Advantage Formulations

Table 3: In-distribution evaluation results on Qwen3-1.7B-Base trained with DAPO-MATH-17k.
We apply the proposed discounted Beta–Bernoulli reward distribution estimation to the advantage term of Dr.GRPO and observe consistent performance improvements.
Δ\Delta denotes the absolute Acc@8 improvement over naive Dr.GRPO.

|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Method | MATH500 | Minerva | AIME24 | AIME25 | AMC24 | Olympiad | Avg. |
| Qwen3-1.7B-Base trained with DAPO-Math-17k | | | | | | | |
| Dr.GRPO | 69.00 | 25.14 | 9.17 | 7.92 | 23.89 | 32.78 | 27.98 |
| Dr.GRPO-DBB | 70.60 | 26.15 | 12.08 | 9.17 | 28.06 | 33.59 | 29.94 |
| Δ\Delta w.r.t. Dr.GRPO | +1.60 | +1.01 | +2.92 | +1.25 | +4.17 | +0.81 | +1.96 |

To examine whether the DBB estimation can be applied to other advantage formulations, we conduct experiments using the advantage term of Dr.GRPO, defined as Ai=Xi−𝔼​[p^τ]A\_{i}=X\_{i}-\mathbb{E}[\hat{p}\_{\tau}].
All experiments are conducted on Qwen3-1.7B-Base with the discounting factor set to λ=0.5\lambda=0.5 for the Dr.GRPO with the DBB estimation (Dr.GRPO-DBB).
As shown in Table [3](#S5.T3 "Table 3 ‣ 5.4 Evaluation on Alternative Advantage Formulations ‣ 5 Analysis & Discussion ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"), Dr.GRPO-DBB consistently improves performance over the naive Dr.GRPO across all in-distribution benchmarks, yielding an average Acc@8 gain of 1.96 points.
These results suggest that the DBB estimation is not limited to the advantage term of GRPO, but can be broadly applied to other RLVR algorithms that rely on point estimation, yielding effective and consistent performance gains.

## 6 Related Work

#### Reinforcement Learning with Verifiable Rewards.

Reinforcement learning with verifiable rewards (RLVR) has emerged as an effective post-training paradigm for improving the reasoning capabilities of large language models by leveraging automatically verifiable reward signals (Lambert et al., [2025](#bib.bib3 "Tulu 3: pushing frontiers in open language model post-training"); Plaat et al., [2025](#bib.bib14 "Multi-step reasoning with large language models, a survey")).
In particular, group-based RLVR algorithms, including GRPO and its variants (Shao et al., [2024](#bib.bib4 "Deepseekmath: pushing the limits of mathematical reasoning in open language models"); Yu et al., [2025a](#bib.bib8 "DAPO: an open-source llm reinforcement learning system at scale"); Liu et al., [2025](#bib.bib7 "Understanding r1-zero-like training: a critical perspective"); Zheng et al., [2025a](#bib.bib5 "Group sequence policy optimization"); Zhao et al., [2025](#bib.bib6 "Geometric-mean policy optimization")), have demonstrated strong performance on reasoning benchmarks without relying on explicit value models (Yu et al., [2025b](#bib.bib13 "RLPR: extrapolating rlvr to general domains without verifiers")).

#### Variance Collapse in Group-Based RLVR.

A well-known limitation of group-based RLVR methods is variance collapse, where the estimated reward variance becomes zero when all sampled responses receive identical rewards, eliminating the training signal and leading to ineffective policy updates (Yu et al., [2025a](#bib.bib8 "DAPO: an open-source llm reinforcement learning system at scale"); Zheng et al., [2025b](#bib.bib10 "Act only when it pays: efficient reinforcement learning for llm reasoning via selective rollouts"); Le et al., [2025](#bib.bib9 "No prompt left behind: exploiting zero-variance prompts in llm reinforcement learning via entropy-guided advantage shaping"); Zhang et al., [2025](#bib.bib17 "Improving sampling efficiency in rlvr through adaptive rollout and response reuse")).
Prior approaches such as Dynamic sAmpling Policy Optimization (DAPO; [Yu et al.](#bib.bib8 "DAPO: an open-source llm reinforcement learning system at scale"), [2025a](#bib.bib8 "DAPO: an open-source llm reinforcement learning system at scale")) and GRESO (Zheng et al., [2025b](#bib.bib10 "Act only when it pays: efficient reinforcement learning for llm reasoning via selective rollouts")) mitigate this issue by modifying the rollout or sampling strategy.
However, these methods either incur substantial additional computational cost or fail to fundamentally eliminate variance collapse due to their probabilistic or heuristic nature.

In contrast, our work addresses variance collapse at the reward estimator level by explicitly modeling uncertainty in the reward distribution.
By maintaining a Beta posterior with strictly positive parameters, the DBB process theoretically prevents variance collapse without additional rollouts or changes to the sampling strategy.

#### Replay- and History-Based RLVR Methods.

Replay-based methods aim to improve sample efficiency in RLVR by reusing historical rollouts, as exemplified by RePO (Li et al., [2025](#bib.bib11 "RePO: replay-enhanced policy optimization")) and ExGRPO (Zhan et al., [2025](#bib.bib12 "ExGRPO: learning to reason from experience")).
While potentially effective, these approaches require storing tokens and their probabilities under historical policies and performing additional forward passes, introducing non-trivial memory and computational overhead.

Our approach differs in that it leverages only historical reward statistics rather than full trajectory information, eliminating the need for additional GPU memory or extra forward passes.
Empirical results demonstrate that reward signals alone can effectively capture useful historical information for improving RLVR performance.

## 7 Conclusion

We revisited RLVR from the perspective of reward distribution estimation and identified point estimation under limited rollouts as a key source of sample inefficiency and variance collapse.
To this end, we proposed Discounted Beta–Bernoulli (DBB) reward estimation for RLVR, which leverages historical reward information for accurate estimation.
Although biased, the DBB estimator achieves substantially lower variance and mean squared error, theoretically avoids variance collapse, and preserves informative training signals without additional rollouts or replay.
Experiments across two model scales demonstrate consistent improvements over baselines on all in-distribution and out-of-distribution benchmarks.
Our findings highlight reward distribution estimation as a critical yet underexplored component of effective RLVR and suggest that principled estimator design offers a promising direction for improving large-scale reinforcement learning of language models.
As future work, we plan to design a dynamic discount factor λ\lambda that adapts across different stages of training.

## Impact Statements

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

## References

* G. Casella and R. Berger (2024)
  Statistical inference.
   Chapman and Hall/CRC.
  Cited by: [§1](#S1.p4.1 "1 Introduction ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* A. Gelman, J. B. Carlin, H. S. Stern, and D. B. Rubin (1995)
  Bayesian data analysis.
   Chapman and Hall/CRC.
  Cited by: [§1](#S1.p5.1 "1 Introduction ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. (2025)
  Deepseek-r1: incentivizing reasoning capability in llms via reinforcement learning.
  arXiv preprint arXiv:2501.12948.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* C. He, R. Luo, Y. Bai, S. Hu, Z. L. Thai, J. Shen, J. Hu, X. Han, Y. Huang, Y. Zhang, J. Liu, L. Qi, Z. Liu, and M. Sun (2024)
  OlympiadBench: a challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems.
  External Links: 2402.14008,
  [Link](https://arxiv.org/abs/2402.14008)
  Cited by: [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Models & Datasets. ‣ 4.1 Experimental Settings ‣ 4 Experiments ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt (2021)
  Measuring mathematical problem solving with the math dataset.
  External Links: 2103.03874,
  [Link](https://arxiv.org/abs/2103.03874)
  Cited by: [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Models & Datasets. ‣ 4.1 Experimental Settings ‣ 4 Experiments ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* N. Lambert, J. Morrison, V. Pyatkin, S. Huang, H. Ivison, F. Brahman, L. J. V. Miranda, A. Liu, N. Dziri, S. Lyu, Y. Gu, S. Malik, V. Graf, J. D. Hwang, J. Yang, R. L. Bras, O. Tafjord, C. Wilhelm, L. Soldaini, N. A. Smith, Y. Wang, P. Dasigi, and H. Hajishirzi (2025)
  Tulu 3: pushing frontiers in open language model post-training.
  External Links: 2411.15124,
  [Link](https://arxiv.org/abs/2411.15124)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§6](#S6.SS0.SSS0.Px1.p1.1 "Reinforcement Learning with Verifiable Rewards. ‣ 6 Related Work ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* T. V. Le, M. Jeon, K. Vu, V. Lai, and E. Yang (2025)
  No prompt left behind: exploiting zero-variance prompts in llm reinforcement learning via entropy-guided advantage shaping.
  External Links: 2509.21880,
  [Link](https://arxiv.org/abs/2509.21880)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§3](#S3.p1.1 "3 Method ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§6](#S6.SS0.SSS0.Px2.p1.1 "Variance Collapse in Group-Based RLVR. ‣ 6 Related Work ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* A. Lewkowycz, A. Andreassen, D. Dohan, E. Dyer, H. Michalewski, V. Ramasesh, A. Slone, C. Anil, I. Schlag, T. Gutman-Solo, Y. Wu, B. Neyshabur, G. Gur-Ari, and V. Misra (2022)
  Solving quantitative reasoning problems with language models.
  External Links: 2206.14858,
  [Link](https://arxiv.org/abs/2206.14858)
  Cited by: [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Models & Datasets. ‣ 4.1 Experimental Settings ‣ 4 Experiments ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* J. Li, E. Beeching, L. Tunstall, B. Lipkin, R. Soletskyi, S. Huang, K. Rasul, L. Yu, A. Q. Jiang, Z. Shen, et al. (2024)
  Numinamath: the largest public dataset in ai4maths with 860k pairs of competition math problems and solutions.
  Hugging Face repository 13 (9),  pp. 9.
  Cited by: [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Models & Datasets. ‣ 4.1 Experimental Settings ‣ 4 Experiments ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* S. Li, Z. Zhou, W. Lam, C. Yang, and C. Lu (2025)
  RePO: replay-enhanced policy optimization.
  External Links: 2506.09340,
  [Link](https://arxiv.org/abs/2506.09340)
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§6](#S6.SS0.SSS0.Px3.p1.1 "Replay- and History-Based RLVR Methods. ‣ 6 Related Work ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* Z. Liu, C. Chen, W. Li, P. Qi, T. Pang, C. Du, W. S. Lee, and M. Lin (2025)
  Understanding r1-zero-like training: a critical perspective.
  External Links: 2503.20783,
  [Link](https://arxiv.org/abs/2503.20783)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§3](#S3.p1.1 "3 Method ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§6](#S6.SS0.SSS0.Px1.p1.1 "Reinforcement Learning with Verifiable Rewards. ‣ 6 Related Work ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* A. Plaat, A. Wong, S. Verberne, J. Broekens, N. van Stein, and T. Back (2025)
  Multi-step reasoning with large language models, a survey.
  External Links: 2407.11511,
  [Link](https://arxiv.org/abs/2407.11511)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§6](#S6.SS0.SSS0.Px1.p1.1 "Reinforcement Learning with Verifiable Rewards. ‣ 6 Related Work ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang, J. Dirani, J. Michael, and S. R. Bowman (2023)
  GPQA: a graduate-level google-proof q&a benchmark.
  External Links: 2311.12022,
  [Link](https://arxiv.org/abs/2311.12022)
  Cited by: [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Models & Datasets. ‣ 4.1 Experimental Settings ‣ 4 Experiments ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov (2017)
  Proximal policy optimization algorithms.
  External Links: 1707.06347,
  [Link](https://arxiv.org/abs/1707.06347)
  Cited by: [§2.2](#S2.SS2.p1.1 "2.2 Group Relative Policy Optimization (GRPO) ‣ 2 Preliminaries ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. (2024)
  Deepseekmath: pushing the limits of mathematical reasoning in open language models.
  arXiv preprint arXiv:2402.03300.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§2.2](#S2.SS2.p1.1 "2.2 Group Relative Policy Optimization (GRPO) ‣ 2 Preliminaries ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§3](#S3.p1.1 "3 Method ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§6](#S6.SS0.SSS0.Px1.p1.1 "Reinforcement Learning with Verifiable Rewards. ‣ 6 Related Work ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* M. Suzgun, N. Scales, N. Schärli, S. Gehrmann, Y. Tay, H. W. Chung, A. Chowdhery, Q. V. Le, E. H. Chi, D. Zhou, and J. Wei (2022)
  Challenging big-bench tasks and whether chain-of-thought can solve them.
  External Links: 2210.09261,
  [Link](https://arxiv.org/abs/2210.09261)
  Cited by: [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Models & Datasets. ‣ 4.1 Experimental Settings ‣ 4 Experiments ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* Y. Wang, X. Ma, G. Zhang, Y. Ni, A. Chandra, S. Guo, W. Ren, A. Arulraj, X. He, Z. Jiang, T. Li, M. Ku, K. Wang, A. Zhuang, R. Fan, X. Yue, and W. Chen (2024)
  MMLU-pro: a more robust and challenging multi-task language understanding benchmark.
  External Links: 2406.01574,
  [Link](https://arxiv.org/abs/2406.01574)
  Cited by: [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Models & Datasets. ‣ 4.1 Experimental Settings ‣ 4 Experiments ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* C. Xie, R. Pan, X. Wu, Y. Zhang, J. Fu, T. Gao, and G. Zhou (2025)
  Unlocking exploration in rlvr: uncertainty-aware advantage shaping for deeper reasoning.
  arXiv preprint arXiv:2510.10649.
  Cited by: [§3](#S3.p1.1 "3 Method ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, C. Zheng, D. Liu, F. Zhou, F. Huang, F. Hu, H. Ge, H. Wei, H. Lin, J. Tang, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Zhou, J. Lin, K. Dang, K. Bao, K. Yang, L. Yu, L. Deng, M. Li, M. Xue, M. Li, P. Zhang, P. Wang, Q. Zhu, R. Men, R. Gao, S. Liu, S. Luo, T. Li, T. Tang, W. Yin, X. Ren, X. Wang, X. Zhang, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Zhang, Y. Wan, Y. Liu, Z. Wang, Z. Cui, Z. Zhang, Z. Zhou, and Z. Qiu (2025)
  Qwen3 technical report.
  External Links: 2505.09388,
  [Link](https://arxiv.org/abs/2505.09388)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§1](#S1.p6.1 "1 Introduction ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Models & Datasets. ‣ 4.1 Experimental Settings ‣ 4 Experiments ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, W. Dai, T. Fan, G. Liu, L. Liu, X. Liu, H. Lin, Z. Lin, B. Ma, G. Sheng, Y. Tong, C. Zhang, M. Zhang, W. Zhang, H. Zhu, J. Zhu, J. Chen, J. Chen, C. Wang, H. Yu, Y. Song, X. Wei, H. Zhou, J. Liu, W. Ma, Y. Zhang, L. Yan, M. Qiao, Y. Wu, and M. Wang (2025a)
  DAPO: an open-source llm reinforcement learning system at scale.
  External Links: 2503.14476,
  [Link](https://arxiv.org/abs/2503.14476)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§1](#S1.p2.1 "1 Introduction ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§2.2](#S2.SS2.p5.2 "2.2 Group Relative Policy Optimization (GRPO) ‣ 2 Preliminaries ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Models & Datasets. ‣ 4.1 Experimental Settings ‣ 4 Experiments ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§6](#S6.SS0.SSS0.Px1.p1.1 "Reinforcement Learning with Verifiable Rewards. ‣ 6 Related Work ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§6](#S6.SS0.SSS0.Px2.p1.1 "Variance Collapse in Group-Based RLVR. ‣ 6 Related Work ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* T. Yu, B. Ji, S. Wang, S. Yao, Z. Wang, G. Cui, L. Yuan, N. Ding, Y. Yao, Z. Liu, M. Sun, and T. Chua (2025b)
  RLPR: extrapolating rlvr to general domains without verifiers.
  External Links: 2506.18254,
  [Link](https://arxiv.org/abs/2506.18254)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§6](#S6.SS0.SSS0.Px1.p1.1 "Reinforcement Learning with Verifiable Rewards. ‣ 6 Related Work ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* R. Zhan, Y. Li, Z. Wang, X. Qu, D. Liu, J. Shao, D. F. Wong, and Y. Cheng (2025)
  ExGRPO: learning to reason from experience.
  External Links: 2510.02245,
  [Link](https://arxiv.org/abs/2510.02245)
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§6](#S6.SS0.SSS0.Px3.p1.1 "Replay- and History-Based RLVR Methods. ‣ 6 Related Work ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* Y. Zhang, W. Yao, C. Yu, Y. Liu, Q. Yin, B. Yin, H. Yun, and L. Li (2025)
  Improving sampling efficiency in rlvr through adaptive rollout and response reuse.
  arXiv preprint arXiv:2509.25808.
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§6](#S6.SS0.SSS0.Px2.p1.1 "Variance Collapse in Group-Based RLVR. ‣ 6 Related Work ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* Y. Zhao, Y. Liu, J. Liu, J. Chen, X. Wu, Y. Hao, T. Lv, S. Huang, L. Cui, Q. Ye, F. Wan, and F. Wei (2025)
  Geometric-mean policy optimization.
  External Links: 2507.20673,
  [Link](https://arxiv.org/abs/2507.20673)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§6](#S6.SS0.SSS0.Px1.p1.1 "Reinforcement Learning with Verifiable Rewards. ‣ 6 Related Work ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* C. Zheng, S. Liu, M. Li, X. Chen, B. Yu, C. Gao, K. Dang, Y. Liu, R. Men, A. Yang, J. Zhou, and J. Lin (2025a)
  Group sequence policy optimization.
  External Links: 2507.18071,
  [Link](https://arxiv.org/abs/2507.18071)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§6](#S6.SS0.SSS0.Px1.p1.1 "Reinforcement Learning with Verifiable Rewards. ‣ 6 Related Work ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
* H. Zheng, Y. Zhou, B. R. Bartoldson, B. Kailkhura, F. Lai, J. Zhao, and B. Chen (2025b)
  Act only when it pays: efficient reinforcement learning for llm reasoning via selective rollouts.
  External Links: 2506.02177,
  [Link](https://arxiv.org/abs/2506.02177)
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"),
  [§6](#S6.SS0.SSS0.Px2.p1.1 "Variance Collapse in Group-Based RLVR. ‣ 6 Related Work ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").

## Appendix A Comparison of Statistics Between Point Estimation and Discounted Beta–Bernoulli Estimation

We present Table [4](#A1.T4 "Table 4 ‣ Appendix A Comparison of Statistics Between Point Estimation and Discounted Beta–Bernoulli Estimation ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards") that summarizes the statistics of point estimation and Discounted Beta–Bernoulli estimation, as discussed in Sections [3.1](#S3.SS1 "3.1 Reward Estimation as Distributional Inference ‣ 3 Method ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards") and [3.2](#S3.SS2 "3.2 Discounted Beta–Bernoulli Reward Estimation ‣ 3 Method ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"), in a concise and accessible manner.

Table 4: Comparison between point estimation and DBB estimation for reward distribution modeling. We denote the Bernoulli distribution by Bern​(⋅)\mathrm{Bern}(\cdot).

|  | Point Estimation | DBB Estimation |
| --- | --- | --- |
| Xτ,iX\_{\tau,i} | Xτ,i∼Bern​(pτ)X\_{\tau,i}\sim\mathrm{Bern}(p\_{\tau}) | Xτ,i∼Bern​(pτ)X\_{\tau,i}\sim\mathrm{Bern}(p\_{\tau}) |
| pτp\_{\tau} | fixed but unknown | pτ∼Beta​(ατ,βτ)p\_{\tau}\sim\mathrm{Beta}(\alpha\_{\tau},\beta\_{\tau}) |
| p^τ\hat{p}\_{\tau} | p^τpt=1N​∑i=1NXτ,i\displaystyle\hat{p}\_{\tau}^{\mathrm{pt}}=\frac{1}{N}\sum\_{i=1}^{N}X\_{\tau,i} | p^τdb=ατατ+βτ\displaystyle\hat{p}\_{\tau}^{\mathrm{db}}=\frac{\alpha\_{\tau}}{\alpha\_{\tau}+\beta\_{\tau}} |
| Var^​(Xτ)\widehat{\mathrm{Var}}(X\_{\tau}) | NN−1​p^τpt​(1−p^τpt)\displaystyle\frac{N}{N-1}\,\hat{p}\_{\tau}^{\mathrm{pt}}\!\left(1-\hat{p}\_{\tau}^{\mathrm{pt}}\right) | p^τdb​(1−p^τdb)\displaystyle\hat{p}\_{\tau}^{\mathrm{db}}\!\left(1-\hat{p}\_{\tau}^{\mathrm{db}}\right) |
| 𝔼​[p^τ]\mathbb{E}[\hat{p}\_{\tau}] | pτp\_{\tau} | w​μτ−1+(1−w)​pτw\mu\_{\tau-1}+(1-w)p\_{\tau} |
| Bias​(p^τ)\mathrm{Bias}(\hat{p}\_{\tau}) | 0 | w​(μτ−1−pτ)w(\mu\_{\tau-1}-p\_{\tau}) |
| Var​(p^τ)\mathrm{Var}(\hat{p}\_{\tau}) | pτ​(1−pτ)N\displaystyle\frac{p\_{\tau}(1-p\_{\tau})}{N} | (1−w)2​pτ​(1−pτ)N\displaystyle(1-w)^{2}\frac{p\_{\tau}(1-p\_{\tau})}{N} |

## Appendix B Derivation of the Mean, Variance, and MSE of the DBB Estimator

In this section, we provide a detailed derivation of Equations ([17](#S3.E17 "Equation 17 ‣ 3.3 Mean Squared Error of the DBB estimator ‣ 3 Method ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards")) and ([18](#S3.E18 "Equation 18 ‣ 3.3 Mean Squared Error of the DBB estimator ‣ 3 Method ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards")) in the main text.
To avoid ambiguity caused by implicit conditioning, we re-express the analysis in terms of the conditional distribution given the sequence of true reward probabilities {p1,p2,…,pτ}\{p\_{1},p\_{2},\dots,p\_{\tau}\}.

### B.1 Expansion of Posterior Parameters under Historical Rewards

At each training step kk, rollout rewards satisfy

|  |  |  |  |
| --- | --- | --- | --- |
|  | Sk∣pk∼Binomial​(N,pk),Xk,i∣pk∼Bernoulli​(pk),S\_{k}\mid p\_{k}\sim\mathrm{Binomial}(N,p\_{k}),\qquad X\_{k,i}\mid p\_{k}\sim\mathrm{Bernoulli}(p\_{k}), |  | (22) |

and the DBB updates are given by

|  |  |  |  |
| --- | --- | --- | --- |
|  | αk=λ​αk−1+Sk,βk=λ​βk−1+(N−Sk).\alpha\_{k}=\lambda\alpha\_{k-1}+S\_{k},\qquad\beta\_{k}=\lambda\beta\_{k-1}+(N-S\_{k}). |  | (23) |

Unrolling the recursion yields

|  |  |  |  |
| --- | --- | --- | --- |
|  | ατ=λτ​α0+∑k=1τλτ−k​Sk,\alpha\_{\tau}=\lambda^{\tau}\alpha\_{0}+\sum\_{k=1}^{\tau}\lambda^{\tau-k}S\_{k}, |  | (24) |

and the total mass

|  |  |  |  |
| --- | --- | --- | --- |
|  | Hτ≜ατ+βτ=λτ​(α0+β0)+N​∑k=1τλτ−k.H\_{\tau}\;\triangleq\;\alpha\_{\tau}+\beta\_{\tau}=\lambda^{\tau}(\alpha\_{0}+\beta\_{0})+N\sum\_{k=1}^{\tau}\lambda^{\tau-k}. |  | (25) |

Crucially, when NN and λ\lambda are fixed, HτH\_{\tau} is deterministic and does not depend on the random variables {Sk}k=1τ\{S\_{k}\}\_{k=1}^{\tau}.
All stochasticity in the estimator arises from ατ\alpha\_{\tau} alone.

### B.2 Mean of the DBB Estimator

The DBB estimator is defined as

|  |  |  |  |
| --- | --- | --- | --- |
|  | p^τdbb=ατHτ.\hat{p}\_{\tau}^{\mathrm{dbb}}=\frac{\alpha\_{\tau}}{H\_{\tau}}. |  | (26) |

Since HτH\_{\tau} is deterministic,

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[p^τdbb∣p1:τ]=𝔼​[ατ∣p1:τ]Hτ.\mathbb{E}\!\left[\hat{p}\_{\tau}^{\mathrm{dbb}}\mid p\_{1:\tau}\right]=\frac{\mathbb{E}[\alpha\_{\tau}\mid p\_{1:\tau}]}{H\_{\tau}}. |  | (27) |

Using 𝔼​[Sk∣pk]=N​pk\mathbb{E}[S\_{k}\mid p\_{k}]=Np\_{k}, we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[ατ∣p1:τ]=λτ​α0+N​∑k=1τλτ−k​pk.\mathbb{E}[\alpha\_{\tau}\mid p\_{1:\tau}]=\lambda^{\tau}\alpha\_{0}+N\sum\_{k=1}^{\tau}\lambda^{\tau-k}p\_{k}. |  | (28) |

Therefore,

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[p^τdbb∣p1:τ]=λτ​α0+N​∑k=1τλτ−k​pkλτ​(α0+β0)+N​∑k=1τλτ−k\boxed{\mathbb{E}\!\left[\hat{p}\_{\tau}^{\mathrm{dbb}}\mid p\_{1:\tau}\right]=\frac{\lambda^{\tau}\alpha\_{0}+N\sum\_{k=1}^{\tau}\lambda^{\tau-k}p\_{k}}{\lambda^{\tau}(\alpha\_{0}+\beta\_{0})+N\sum\_{k=1}^{\tau}\lambda^{\tau-k}}} |  | (29) |

For interpretability, define weights

|  |  |  |  |
| --- | --- | --- | --- |
|  | c0≜λτ​(α0+β0)Hτ,ck≜N​λτ−kHτ,k=1,…,τ,p0≜α0α0+β0.c\_{0}\triangleq\frac{\lambda^{\tau}(\alpha\_{0}+\beta\_{0})}{H\_{\tau}},\qquad c\_{k}\triangleq\frac{N\lambda^{\tau-k}}{H\_{\tau}},\quad k=1,\dots,\tau,\qquad p\_{0}\triangleq\frac{\alpha\_{0}}{\alpha\_{0}+\beta\_{0}}. |  | (30) |

Then the estimator mean can be written as

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼[p^τdbb∣p1:τ]=∑k=0τckpk,∑k=0τck=1,\boxed{\mathbb{E}\!\left[\hat{p}\_{\tau}^{\mathrm{dbb}}\mid p\_{1:\tau}\right]=\sum\_{k=0}^{\tau}c\_{k}p\_{k},\qquad\sum\_{k=0}^{\tau}c\_{k}=1,} |  | (31) |

revealing an exponentially weighted average of historical reward probabilities.

### B.3 Variance of the DBB Estimator

Again using the determinism of HτH\_{\tau},

|  |  |  |  |
| --- | --- | --- | --- |
|  | Var​(p^τdbb∣p1:τ)=Var​(ατ∣p1:τ)Hτ2.\mathrm{Var}\!\left(\hat{p}\_{\tau}^{\mathrm{dbb}}\mid p\_{1:\tau}\right)=\frac{\mathrm{Var}(\alpha\_{\tau}\mid p\_{1:\tau})}{H\_{\tau}^{2}}. |  | (32) |

Since {Sk}\{S\_{k}\} are independent conditioned on {pk}\{p\_{k}\} and
Var​(Sk∣pk)=N​pk​(1−pk)\mathrm{Var}(S\_{k}\mid p\_{k})=Np\_{k}(1-p\_{k}),

|  |  |  |  |
| --- | --- | --- | --- |
|  | Var​(ατ∣p1:τ)=∑k=1τλ2​(τ−k)​N​pk​(1−pk).\mathrm{Var}(\alpha\_{\tau}\mid p\_{1:\tau})=\sum\_{k=1}^{\tau}\lambda^{2(\tau-k)}Np\_{k}(1-p\_{k}). |  | (33) |

Hence,

|  |  |  |  |
| --- | --- | --- | --- |
|  | Var​(p^τdbb∣p1:τ)=∑k=1τλ2​(τ−k)​N​pk​(1−pk)(λτ​(α0+β0)+N​∑k=1τλτ−k)2\boxed{\mathrm{Var}\!\left(\hat{p}\_{\tau}^{\mathrm{dbb}}\mid p\_{1:\tau}\right)=\frac{\sum\_{k=1}^{\tau}\lambda^{2(\tau-k)}Np\_{k}(1-p\_{k})}{\left(\lambda^{\tau}(\alpha\_{0}+\beta\_{0})+N\sum\_{k=1}^{\tau}\lambda^{\tau-k}\right)^{2}}} |  | (34) |

### B.4 Bias and Mean Squared Error of the DBB Estimator

By definition,

|  |  |  |  |
| --- | --- | --- | --- |
|  | MSE​(p^τdbb∣p1:τ)=Bias2+Var.\mathrm{MSE}\!\left(\hat{p}\_{\tau}^{\mathrm{dbb}}\mid p\_{1:\tau}\right)=\mathrm{Bias}^{2}+\mathrm{Var}. |  | (35) |

Using Eq. ([31](#A2.E31 "Equation 31 ‣ B.2 Mean of the DBB Estimator ‣ Appendix B Derivation of the Mean, Variance, and MSE of the DBB Estimator ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards")), the conditional bias is

|  |  |  |  |
| --- | --- | --- | --- |
|  | Bias(p^τdbb∣p1:τ)=∑k=0τ−1ck(pk−pτ),\boxed{\mathrm{Bias}\!\left(\hat{p}\_{\tau}^{\mathrm{dbb}}\mid p\_{1:\tau}\right)=\sum\_{k=0}^{\tau-1}c\_{k}(p\_{k}-p\_{\tau}),} |  | (36) |

where the k=τk=\tau term vanishes since pτ−pτ=0p\_{\tau}-p\_{\tau}=0.

Combining Eqs. ([34](#A2.E34 "Equation 34 ‣ B.3 Variance of the DBB Estimator ‣ Appendix B Derivation of the Mean, Variance, and MSE of the DBB Estimator ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards")) and ([36](#A2.E36 "Equation 36 ‣ B.4 Bias and Mean Squared Error of the DBB Estimator ‣ Appendix B Derivation of the Mean, Variance, and MSE of the DBB Estimator ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards")), we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | MSE(p^τdbb∣p1:τ)=(∑k=0τ−1ck(pk−pτ))2+∑k=1τλ2​(τ−k)​N​pk​(1−pk)Hτ2,\boxed{\mathrm{MSE}\!\left(\hat{p}\_{\tau}^{\mathrm{dbb}}\mid p\_{1:\tau}\right)=\left(\sum\_{k=0}^{\tau-1}c\_{k}(p\_{k}-p\_{\tau})\right)^{2}+\frac{\sum\_{k=1}^{\tau}\lambda^{2(\tau-k)}Np\_{k}(1-p\_{k})}{H\_{\tau}^{2}},} |  | (37) |

where Hτ=λτ​(α0+β0)+N​∑k=1τλτ−kH\_{\tau}=\lambda^{\tau}(\alpha\_{0}+\beta\_{0})+N\sum\_{k=1}^{\tau}\lambda^{\tau-k}.

## Appendix C Implementation Details

Table 5: Common Training hyperparameter settings for all experiments.

|  |  |  |
| --- | --- | --- |
| Hyperparameter | Qwen3-1.7B-Base | Qwen3-8B-Base |
| Training Configuration | | |
| Training batch size | 128 | 128 |
| Mini-batch size | 64 | 64 |
| Number of epochs | 4 | 4 |
| Total gradient steps | 1080 | 1080 |
| Samples per prompt | 8 | 8 |
| Max response length | 4096 | 8192 |
| Sampling Configuration | | |
| Training temperature | 1.0 | 1.0 |
| Training top-pp | 1.0 | 1.0 |
| Validation temperature | 0.6 | 0.6 |
| Validation top-pp | 0.95 | 0.95 |
| Optimization | | |
| Optimizer | AdamW | AdamW |
| Learning rate | 1×10−61\times 10^{-6} | 1×10−61\times 10^{-6} |
| LR warmup steps | 0 | 0 |
| LR scheduler | constant | constant |

In this section, we describe the experimental setup details for the RLVR algorithm with the DBB process (RLVR-DBB) and all baseline methods.
All training experiments are conducted using the verl framework, and Math-Verify is employed to extract and normalize final answers from model responses.
Training Qwen3-1.7B-Base is performed using 4×\timesH200 GPUs, while Qwen3-8B-Base is trained using 8×\timesH200 GPUs.
Due to its substantially higher GPU memory requirements, RePO is trained using 8×\timesH200 GPUs for both model scales.

All training experiments are conducted using the DAPO-Math-17k dataset.
For training stability, we filter prompts to a maximum length of 1024 tokens.
In addition, to facilitate analysis, we further restrict the dataset so that its size is an integer multiple of the rollout batch size, which is 128.
As a result, out of the original 17,391 prompts, 17,280 are used for training.

The hyperparameters shared across all training experiments are summarized in Table [5](#A3.T5 "Table 5 ‣ Appendix C Implementation Details ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
Our experimental setup follows configurations that are most commonly adopted in recent RLVR studies.
Following DAPO, we remove the KL regularization term during training.
In addition, under the clip-higher strategy, we use a clipping range of (cliplow,cliphigh)=(0.2,0.28)(\text{clip}\_{\text{low}},\text{clip}\_{\text{high}})=(0.2,0.28) for all methods except RLVR-DBB.

For RLVR-DBB, the importance ratios tend to be larger than those of naive methods.
As a result, using the same clipping range would cause a substantially larger fraction of meaningful update signals to be clipped.
To avoid overly aggressive clipping, we therefore adopt a wider clipping range of (0.98,0.98)(0.98,0.98) specifically for RLVR-DBB.

For RePO, which introduces additional hyperparameters and configuration options, we follow the optimal settings recommended in the original paper.
Specifically, we set the number of replay samples to 8, use a replay cache size of 16, and adopt the reward-oriented replay strategy.

For both evaluation and validation, we use the sampling parameters specified in Table [5](#A3.T5 "Table 5 ‣ Appendix C Implementation Details ‣ Discounted Beta–Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards").
To reduce training overhead, the validation set is restricted to AIME24/25, AMC24, Minerva, and MATH500.
Evaluation on OlympiadBench and the out-of-distribution benchmarks (MMLU-Pro, GPQA-Diamond, and Big-Bench Hard) is performed only on the best checkpoint selected for each experiment.
