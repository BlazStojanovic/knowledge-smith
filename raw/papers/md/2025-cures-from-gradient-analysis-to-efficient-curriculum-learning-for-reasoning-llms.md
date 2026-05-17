---
arxiv: '2510.01037'
authors:
- Yongcheng Zeng
- Zexu Sun
- Bokai Ji
- Erxue Min
- Hengyi Cai
- Shuaiqiang Wang
- Dawei Yin
- Haifeng Zhang
- Xu Chen
- Jun Wang
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning
  LLMs'
url: https://arxiv.org/abs/2510.01037
year: 2025
---

# CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs

Yongcheng Zeng1,2, , Zexu Sun3,∗, Bokai Ji3, Erxue Min3, Hengyi Cai3,
  
Shuaiqiang Wang3, Dawei Yin3, Haifeng Zhang1,2, , Xu Chen5,†, Jun Wang4,†
  
1Institute of Automation, Chinese Academy of Sciences
  
2School of Artificial Intelligence, University of Chinese Academy of Sciences
  
3Baidu Inc. 4University College London
  
5Gaoling School of Artificial Intelligence, Renmin University of China
  
sunzexu0826@gmals.com, zengyongcheng2022@ia.ac.cn 
Equal Contribution.Corresponding authors.

###### Abstract

Curriculum learning plays a crucial role in enhancing the training efficiency of large language models (LLMs) on reasoning tasks. However, existing methods often fail to adequately account for variations in prompt difficulty or rely on simplistic filtering mechanisms to select prompt datasets within a narrow criterion range, resulting in significant computational waste.
In this work, we approach the problem from the perspective of reinforcement learning gradient optimization, offering a systematic and theoretical investigation into how to improve the training efficiency of LLMs. We identify two key factors influencing training efficiency: the selection of training prompts and the allocation of rollout quantities across different prompts. Our theoretical analysis reveals that the sampling distribution of prompts dictates the convergence rate of gradient descent, while the allocation of the rollout quantity influences the consistency and stability of overall gradient updates.
Based on these insights, we propose CurES, an efficient training method that accelerates convergence and employs Bayesian posterior estimation to minimize computational overhead. Experiments demonstrate that our CurES outperforms Group Relative Policy Optimization (GRPO) by +3.30 points and +4.82 points with 1.5B and 7B models, respectively. Additionally, CurES exhibits faster convergence compared to baselines, including GRPO.

  

GitHub: <https://github.com/ZexuSun/CurES>

## 1 Introduction

Although Reinforcement Learning with Verifiable Reward (RLVR) (Guo et al., [2025](#bib.bib7); Lambert et al., [2024](#bib.bib12); Guo et al., [2025](#bib.bib7); Team et al., [2025](#bib.bib26)) has emerged as a powerful paradigm for reasoning tasks of Large Language Models (LLMs), prevailing approaches often rely on uniform sampling strategies that treat all training instances identically (Zeng et al., [2025](#bib.bib33); Xie et al., [2025](#bib.bib28)). This paradigm fails to account for the inherent heterogeneity in prompt difficulty and the varying training utility that different prompts offer. Consequently, computational resources are inefficiently allocated, being wasted either on trivial prompts that yield diminishing returns or on excessively challenging examples where the model shows negligible progress.

Recent works have investigated progressive training curricula that partition the process into several hand-crafted stages of increasing difficulty (Luo et al., [2025](#bib.bib17); Song et al., [2025](#bib.bib25)). However, such partitioning is overly coarse and struggles to align with the evolving capabilities of reasoning models during training. Other approaches apply online data filtering by generating and then pruning samples (Yu et al., [2025](#bib.bib31); Bae et al., [2025](#bib.bib1); Lin et al., [2025](#bib.bib16)). Yet, this paradigm does little to conserve computational resources and instead leads to suboptimal sample efficiency. Additional studies have begun exploring dynamic computation reallocation across prompts with minimal overhead (Yao et al., [2025](#bib.bib30); Zhang et al., [2025](#bib.bib34); Shi et al., [2025](#bib.bib24)). Nevertheless, these techniques address only isolated facets of training acceleration, without fully accounting for the problem’s inherent dynamism.

!(/html/2510.01037/assets/x1.png)

Figure 1: Illustration of our theoretical and practical contributions. The first part presents our theoretical analysis, which establishes the relationship between the gradient efficiency and models’ question-answering accuracy, denoted as pθ​(x)p\_{\theta}(x). Building upon these insights, we develop CurES, a practical method that initially estimates pθ​(x)p\_{\theta}(x) using a small rollout quantity, then reallocates prompt sampling probabilities and rollout quantities based on the estimated accuracy.
We progressively enhance the confidence of these accuracy estimates through posterior estimation. The figure further contrasts CurES with existing approaches, highlighting differences in managing prompt sampling distributions of Speed-RL (Zhang et al., [2025](#bib.bib34)) and rollout quantities of GVM (Yao et al., [2025](#bib.bib30)).

In this work, we first analyze the efficiency of training optimization for reasoning models from the perspective of gradients, elucidating its close relationship with the sampling probability distribution of prompts and the allocation of rollout quantities across these prompts. Our analysis reveals that the prompt sampling distribution directly influences the speed of gradient descent, while the allocation of rollout quantities affects the consistency and stability of overall gradient updates. Leveraging these insights, we propose CurES, a practical training method. CurES first estimates prompt difficulty via models’ question-answering accuracy, then reallocates prompt sampling probabilities and rollout quantities accordingly. During training, the confidence in these accuracy estimates is progressively refined through posterior estimation based on previously sampled data, thereby improving the robustness of the allocation process. Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs") illustrates the overall approach, and our contributions are summarized below:

* •

  We provide a theoretical analysis from the gradient perspective, elucidating the intrinsic relationship between training optimization efficiency and prompt sampling distribution, as well as the allocation of rollout quantities across prompts.
* •

  Guided by the theoretical analysis, we propose a practical training method that integrates Bayesian posterior estimation, achieving enhanced efficiency and stability in reasoning model training with minimal computational overhead.
* •

  Experimental results show that our CurES outperforms GRPO by +3.30 points and +4.82 points with 1.5B and 7B models, respectively. Additionally, CurES exhibits faster convergence compared to baselines, including GRPO.

## 2 Related Works

Gradient Analysis in Optimization.
Gradient analysis plays a pivotal role in understanding and improving optimization processes (Ruder, [2016](#bib.bib21); Bottou et al., [2018](#bib.bib2); Yang et al., [2024](#bib.bib29)).
A common application involves leveraging gradient analysis to reduce variance in iterative optimization and enhance training stability (Medyakov et al., [2025](#bib.bib18); Yuan et al., [2024](#bib.bib32)).
Additionally, it facilitates adaptive learning rate adjustments by dynamically scaling updates based on gradient histories (Dereich et al., [2024](#bib.bib6); Chen et al., [2024](#bib.bib3)).
Gradient analysis also guides data selection and curriculum learning strategies, enabling dynamic adjustment of training data difficulty (Yao et al., [2025](#bib.bib30); Li et al., [2024b](#bib.bib15)).
Furthermore, it aids in detecting anomalous samples to improve data quality by identifying outliers in the gradient space (Chhabra et al., [2024](#bib.bib4)).
As gradients are directly tied to model optimization, they provide the most immediate insights into training dynamics.
Theoretical analysis of gradients enables predictions of convergence rates and bounds, ensuring robust optimization guarantees (Zhao & Xu, [2024](#bib.bib36)).
In this work, we examine the interplay between model optimization and sample selection from a gradient perspective, deriving methods to enhance training efficiency.

Curriculum Learning and Data Selection in RLVR.
Effective data selection is critical for optimizing RLVR training, yet designing curricula that align with the dynamic capabilities of LLMs remains challenging. Progressive training curricula, such as those proposed in (Luo et al., [2025](#bib.bib17); Song et al., [2025](#bib.bib25)), partition training into hand-crafted stages of increasing difficulty. However, these static approaches often fail to adapt to the evolving proficiency of models during training. Online data filtering methods, such as those in (Yu et al., [2025](#bib.bib31); Bae et al., [2025](#bib.bib1); Lin et al., [2025](#bib.bib16)), generate and prune samples to focus on high-impact data but introduce significant computational overhead, leading to suboptimal sample efficiency. Recent efforts have explored dynamic computation reallocation to prioritize prompts with higher training utility (Yao et al., [2025](#bib.bib30); Zhang et al., [2025](#bib.bib34); Shi et al., [2025](#bib.bib24)). However, these methods address only specific aspects of training acceleration and do not fully account for the interplay between prompt sampling distributions and rollout quantity allocation. In contrast, our work proposes CurES, a method that dynamically adjusts sampling probabilities and rollout quantities based on Bayesian posterior estimation of prompt difficulty, achieving improved sample efficiency with minimal computational overhead.

## 3 Preliminaries

RLVR represents a specialized reinforcement learning paradigm tailored for reasoning tasks, where reward signals can be deterministically verified through programmatic means. This approach is particularly well-suited for domains such as mathematical reasoning, code generation, and logical deduction, where correctness criteria are objectively defined (Lambert et al., [2024](#bib.bib12); Guo et al., [2025](#bib.bib7); Team et al., [2025](#bib.bib26)). Formally, given a policy model πθ\pi\_{\theta} and a prompt distribution ρ\rho, the RLVR objective aims to maximize the expected reward while constraining policy updates within a trust region:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ(θ)=−𝔼x∼ρ,y∼πθ(⋅|x)[Aθold(x,y)],subject to𝔼x∼𝒟[DKL(πθold(⋅|x)∥πθ(⋅|x))]≤δ.\displaystyle\mathcal{L}(\theta)=-\mathbb{E}\_{{x}\sim\rho,{y}\sim\pi\_{\theta}(\cdot|{x})}\big[A\_{\theta\_{\mathrm{old}}}(x,y)\big],\quad\text{subject to}\ \mathbb{E}\_{{x}\sim\mathcal{D}}[D\_{\mathrm{KL}}(\pi\_{\theta\_{\mathrm{old}}}(\cdot|x)\|\pi\_{\theta}(\cdot|x))]\leq\delta. |  | (1) |

Here, Aθold​(x,y)=r​(x,y)−𝔼y∼πθold​[r​(x,y)]A\_{\theta\_{\mathrm{old}}}(x,y)=r(x,y)-\mathbb{E}\_{y\sim\pi\_{\theta\_{\mathrm{old}}}}[r(x,y)] denotes the advantage function, θold\theta\_{\mathrm{old}} represents the policy parameters from the previous iteration, and δ\delta defines the trust region boundary that prevents excessive policy divergence.

A key characteristic of RLVR is its reward formulation. Unlike preference-based RLHF that relies on subjective human judgments, RLVR employs a verifiable reward function defined as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | r​(x,y)={1,if ​y​ is the correct answer for ​x0,otherwiser(x,y)=\begin{cases}1,&\text{if }y\text{ is the correct answer for }x\\ 0,&\text{otherwise}\end{cases} |  | (2) |

The straightforward reward function design partially mitigates the issue of reward hacking.

## 4 Methodology

In this section, we introduce CurES, a novel method designed to enhance the training efficiency of Reasoning LLMs. We begin by establishing a theoretical connection between gradient optimization efficiency and two key factors: the sampling distribution of prompts and the allocation of rollout quantities across these prompts. Based on this analysis, CurES first leverages the estimation of the model’s question-answering accuracy to assess prompt difficulty, which is then used to guide an optimal sampling strategy and rollout quantity allocation. By leveraging Bayesian posterior estimation, we progressively refine the confidence in these accuracy estimates using historical sampling data, ensuring robust and adaptive resource allocation with minimal computational overhead.

### 4.1 Prompt Difficulty Caps Optimization Potential

To facilitate subsequent derivations, we define the question difficulty as the model’s accuracy in answering the question. Given a policy model πθ\pi\_{\theta} and a binary reward function r​(x,y)r(x,y) in Eq. ([2](#S3.E2 "In 3 Preliminaries ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs")), the expression for the model’s question-answering accuracy pθp\_{\theta} is given by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | pθ​(x)=𝔼y∼πθ​[r​(x,y)].\displaystyle p\_{\theta}(x)=\mathbb{E}\_{y\sim\pi\_{\theta}}[r(x,y)]. |  | (3) |

To investigate how prompt difficulty influences model gradient updates, we first consider the following optimization problem for a given prompt xx:

|  |  |  |  |
| --- | --- | --- | --- |
|  | minℒ(x;θ)=min−𝔼y∼πθ(⋅|x)[Aθold(x,y)],s.t.DKL(πθold(⋅|x)∥πθ(⋅|x))≤δ.\min\mathcal{L}(x;\theta)=\min-\mathbb{E}\_{{y}\sim\pi\_{\theta}(\cdot|{x})}\big[A\_{\theta\_{\mathrm{old}}}(x,y)\big],\quad\text{s.t.}\ D\_{\mathrm{KL}}(\pi\_{\theta\_{\mathrm{old}}}(\cdot|x)\|\pi\_{\theta}(\cdot|x))\leq\delta. |  | (4) |

Here, we separately analyze the impact of different prompts on the loss function and theoretically examine how varying prompt difficulty levels affect model training efficiency.

We employ the Lagrange multiplier method to solve the above problem. First, we set θ=θold+d\theta=\theta\_{\text{old}}+d and reformulate the problem as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | d∗=argmindℒ(x;θold+d)+λ(DKL(πθold(⋅|x)∥πθold+d(⋅|x))−δ).d^{\*}=\operatorname\*{argmin}\_{d}\ \mathcal{L}(x;\theta\_{\mathrm{old}}+d)+\lambda(D\_{\mathrm{KL}}(\pi\_{\theta\_{\mathrm{old}}}(\cdot|x)\|\pi\_{\theta\_{\mathrm{old}}+d}(\cdot|x))-\delta). |  | (5) |

By performing first-order Taylor expansion on the loss function ℒ​(x;θ)\mathcal{L}(x;\theta) and second-order Taylor expansion on the KL divergence term DKL(πθold(⋅|x)∥πθ(⋅|x))D\_{\mathrm{KL}}(\pi\_{\theta\_{\mathrm{old}}}(\cdot|x)\|\pi\_{\theta}(\cdot|x)), followed by simplification, we derive the following equation:

|  |  |  |  |
| --- | --- | --- | --- |
|  | d∗=argmindℒ​(x;θold)+∇θℒ​(x;θ)𝖳|θ=θold​d+λ2​d𝖳​F​(x;θold)​d−λ​δ,d^{\*}=\operatorname\*{argmin}\_{d}\ \mathcal{L}(x;\theta\_{\mathrm{old}})+\nabla\_{\theta}\mathcal{L}(x;\theta)^{\mathsf{T}}\Big|\_{\theta=\theta\_{\text{old}}}d+\frac{\lambda}{2}d^{\mathsf{T}}F(x;\theta\_{\text{old}})d-\lambda\delta, |  | (6) |

where F​(x;θ)F(x;\theta) is the Fisher Information Matrix, a metric quantifying the information that observed data provides about parameter estimates in probabilistic models. In reinforcement learning, it primarily serves to construct more reasonable parameter update directions, thereby improving the efficiency and stability of policy optimization. Here, F​(x;θ)F(x;\theta) is represented as

|  |  |  |  |
| --- | --- | --- | --- |
|  | F​(x;θ)=𝔼y∼πθ​[∇θlog⁡πθ​(y|x)​∇θlog⁡πθ​(y|x)𝖳].\displaystyle F(x;\theta)=\mathbb{E}\_{y\sim\pi\_{\theta}}\left[\nabla\_{\theta}\log\pi\_{\theta}(y|x)\nabla\_{\theta}\log\pi\_{\theta}(y|x)^{\mathsf{T}}\right]. |  | (7) |

Through mathematical derivation to address the problem of Eq. ([6](#S4.E6 "In 4.1 Prompt Difficulty Caps Optimization Potential ‣ 4 Methodology ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs")), we obtain the following results:

|  |  |  |  |
| --- | --- | --- | --- |
|  | d∗=−1λ​F−1​(x;θold)​∇θℒ​(x;θ)|θ=θold,\displaystyle d^{\*}=-\frac{1}{\lambda}F^{-1}(x;\theta\_{\text{old}})\nabla\_{\theta}\mathcal{L}(x;\theta)\Big|\_{\theta=\theta\_{\text{old}}}, |  | (8) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | λ=∇θℒ​(x;θ)⊤|θ=θold​F−1​(x;θold)​∇θℒ​(x;θ)|θ=θold2​δ.\displaystyle\lambda=\sqrt{\frac{\nabla\_{\theta}\mathcal{L}(x;\theta)^{\top}\big|\_{\theta=\theta\_{\text{old}}}F^{-1}(x;\theta\_{\text{old}})\nabla\_{\theta}\mathcal{L}(x;\theta)\big|\_{\theta=\theta\_{\text{old}}}}{2\delta}}. |  | (9) |

With Eq. ([8](#S4.E8 "In 4.1 Prompt Difficulty Caps Optimization Potential ‣ 4 Methodology ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs")) and Eq. ([9](#S4.E9 "In 4.1 Prompt Difficulty Caps Optimization Potential ‣ 4 Methodology ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs")), we derive the expression for the update of the loss function:

|  |  |  |  |
| --- | --- | --- | --- |
|  | |ℒ​(x;θold+d)−ℒ​(x;θold)|=2​δ​∇θℒ​(x;θ)𝖳|θ=θo​l​d​F−1​(x;θold)​∇θℒ​(x;θ)|θ=θ old.\left|\mathcal{L}(x;\theta\_{\text{old}}+d)-\mathcal{L}(x;\theta\_{\text{old}})\right|=\sqrt{2\delta\nabla\_{\theta}\mathcal{L}(x;\theta)^{\mathsf{T}}\Big|\_{\theta=\theta\_{old}}F^{-1}(x;\theta\_{\text{old}})\nabla\_{\theta}\mathcal{L}(x;\theta)\Big|\_{\theta=\theta\_{\text{ old}}}}. |  | (10) |

According to the definition, the binary reward function r​(x,y)r(x,y) serves as an unbiased estimator of the model’s question-answering accuracy pθ​(x)p\_{\theta}(x), i.e., pθ​(x)=𝔼y∼πθ​[r​(x,y)]p\_{\theta}(x)=\mathbb{E}\_{y\sim\pi\_{\theta}}[r(x,y)]. Through the application of the Cramér-Rao inequality, we derive the following fundamental result:

|  |  |  |  |
| --- | --- | --- | --- |
|  | |ℒ​(x;θold+d)−ℒ​(x;θold)|≤2​δ​pθold​(x)​(1−pθold​(x)).|\mathcal{L}(x;\theta\_{\text{old}}+d)-\mathcal{L}(x;\theta\_{\text{old}})|\leq\sqrt{2\delta p\_{\theta\_{\text{old}}}(x)\left(1-p\_{\theta\_{\text{old}}}(x)\right)}. |  | (11) |

Therefore, for the loss function ℒ​(θ)\mathcal{L}(\theta), its optimization potential exhibits the following relationship with prompt difficulty:

|  |  |  |  |
| --- | --- | --- | --- |
|  | |ℒ​(θold+d)−ℒ​(θold)|≤𝔼x∼ρ​[2​δ​pθold​(x)​(1−pθold​(x))].|\mathcal{L}(\theta\_{\text{old}}+d)-\mathcal{L}(\theta\_{\text{old}})|\leq\mathbb{E}\_{x\sim\rho}\left[\sqrt{2\delta p\_{\theta\_{\text{old}}}(x)\left(1-p\_{\theta\_{\text{old}}}(x)\right)}\right]. |  | (12) |

This demonstrates that the convergence rate of the model’s loss function is intrinsically related to the difficulty of the prompt dataset, which is quantified by the model’s answering accuracy.
To accelerate training, the sampling distribution ρ\rho should assign varied probabilities to prompts based on difficulty while maintaining a balance with exploration. Thus, we seek the optimal sampling distribution ρ\rho under the entropy maximization constraint. Concretely, we address the following problem:

|  |  |  |  |
| --- | --- | --- | --- |
|  | max⁡𝔼x∼ρ​[2​δ​pθold​(x)​(1−pθold​(x))+α​ℋ​(ρ)],s.t.​∑i=1Nρ​(xi)=1.\max\ \mathbb{E}\_{x\sim\rho}\left[\sqrt{2\delta p\_{\theta\_{\text{old}}}(x)\left(1-p\_{\theta\_{\text{old}}}(x)\right)}+\alpha\mathcal{H}(\rho)\right],\quad\text{s.t.}\ \sum\_{i=1}^{N}\rho(x\_{i})=1. |  | (13) |

Solving the aforementioned problem, we obtain the optimal sampling distribution as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ρ∗​(x)=exp⁡(pθold​(x)​(1−pθold​(x))/τ)∑x′exp⁡(pθold​(x′)​(1−pθold​(x′))/τ),\rho^{\*}(x)=\frac{\exp\left(\sqrt{p\_{\theta\_{\text{old}}}(x)\left(1-p\_{\theta\_{\text{old}}}(x)\right)}/\tau\right)}{\sum\_{x^{\prime}}\exp\left(\sqrt{p\_{\theta\_{\text{old}}}(x^{\prime})\left(1-p\_{\theta\_{\text{old}}}(x^{\prime})\right)}/\tau\right)}, |  | (14) |

where τ=α2​δ\tau=\frac{\alpha}{\sqrt{2\delta}} is a hyperparameter. For the theoretical proof please refer to Appendix [A.1](#A1.SS1 "A.1 Prompt Difficulty Caps Optimization Potential ‣ Appendix A Mathematical Derivations ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs").

### 4.2 Closing the Gap with Theoretical Bound

In the previous section, we derived an upper bound on the gradient update for a given prompt. However, due to the high computational cost of the natural gradient method, it is often avoided in practice, and the theoretical result is instead used to guide prompt sampling. During actual gradient updates, we aim to closely approximate the theoretical efficiency limit within a trust region bounded by a KL divergence constraint of δ\delta. Specifically, after sampling a batch of mm prompts, we seek to optimize operations to approach the bound. Within the curriculum learning framework, we consider optimizing the allocation of rollout quantities across prompts under a fixed total rollout budget of NN to minimize the following loss function:

|  |  |  |  |
| --- | --- | --- | --- |
|  | min⁡𝔼​[(ℒ​(θ^)−ℒ​(θold)−(−𝔼x∼ρ​[2​δ​pθold​(x)​(1−pθold​(x))]))2],s.t.​∑i=1mni=N.\min\ \mathbb{E}\left[\left({{\mathcal{L}}}(\hat{\theta})-{{\mathcal{L}}}(\theta\_{\text{old}})-\left(-\mathbb{E}\_{x\sim\rho}\left[\sqrt{2\delta p\_{\theta\_{\text{old}}}(x)\left(1-p\_{\theta\_{\text{old}}}(x)\right)}\right]\right)\right)^{2}\right],\quad\text{s.t.}\ \sum\_{i=1}^{m}n\_{i}=N. |  | (15) |

Here, θ^\hat{\theta} denotes the updated model parameters obtained from θold\theta\_{\text{old}} after applying the practical gradient update, i.e.:

|  |  |  |  |
| --- | --- | --- | --- |
|  | θ^=θold−η​∇θℒ^​(θ)|θ=θold,ℒ^​(θ)=−∑i=1m1ni​∑yj∈𝒟i[πθ​(yj|xi)πθold​(yj|xi)​Aθold].\displaystyle\hat{\theta}=\theta\_{\text{old}}-\eta\nabla\_{\theta}\hat{\mathcal{L}}(\theta)\big|\_{\theta=\theta\_{\text{old}}},\quad\hat{\mathcal{L}}(\theta)=-\sum\limits\_{i=1}^{m}\frac{1}{n\_{i}}\sum\limits\_{y\_{j}\in\mathcal{D}\_{i}}\left[\frac{\pi\_{\theta}(y\_{j}|x\_{i})}{\pi\_{\theta\_{\text{old}}}(y\_{j}|x\_{i})}A\_{\theta\_{\text{old}}}\right]. |  | (16) |

Where η\eta is the learning rate and nin\_{i} denotes the number of sampled rollouts for question xix\_{i}. We assume that η\eta is chosen such that the policy update remains within a KL divergence constraint of δ\delta.

For convenience, we denote g=∇θℒ​(θ)|θ=θoldg=\nabla\_{\theta}\mathcal{L}(\theta)\big|\_{\theta=\theta\_{\text{old}}},g^=∇θℒ^​(θ)|θ=θold\hat{g}=\nabla\_{\theta}\hat{\mathcal{L}}(\theta)\big|\_{\theta=\theta\_{\text{old}}}. By simplifying the loss function, we can show that the optimization problem reduces to the following:

|  |  |  |  |
| --- | --- | --- | --- |
|  | min⁡g⊤​𝕍​(g^)​g,s.t.​∑i=1mni=N.\displaystyle\min g^{\top}\mathbb{V}(\hat{g})g,\quad\text{s.t.}\ \sum\_{i=1}^{m}n\_{i}=N. |  | (17) |

The theoretical gradient direction gg is typically unknown, and we seek to control the uncertainty of the estimator in all possible directions. Therefore, we instead minimize the total variance Tr​(𝕍​(g^))\text{Tr}(\mathbb{V}(\hat{g})), which corresponds to uniformly reducing the variance in all directions. This approach is a widely adopted technique for variance estimation(Bottou et al., [2018](#bib.bib2); Papini et al., [2018](#bib.bib19); Wang et al., [2013](#bib.bib27)). In other words, we consider the following optimization problem:

|  |  |  |  |
| --- | --- | --- | --- |
|  | min⁡Tr​(𝕍​(g^)),s.t.​∑i=1mni=N.\displaystyle\min\ \text{Tr}(\mathbb{V}(\hat{g})),\quad\text{s.t.}\ \sum\_{i=1}^{m}n\_{i}=N. |  | (18) |

By expanding the variance of the aforementioned gradient, we isolate the rollout quantities nin\_{i} to facilitate analysis:

|  |  |  |  |
| --- | --- | --- | --- |
|  | min⁡1m2​∑i=1mTr​(𝕍y∼πθold​(h​(y,xi;θold)))ni,s.t.​∑i=1mni=N.\displaystyle\min\ \frac{1}{m^{2}}\sum\_{i=1}^{m}\frac{\text{Tr}\left(\mathbb{V}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left(h\left(y,x\_{i};\theta\_{\text{old}}\right)\right)\right)}{n\_{i}},\quad\text{s.t.}\ \sum\_{i=1}^{m}n\_{i}=N. |  | (19) |

where h​(x,y;θ)=∇θπθ​(yj|xi)πθold​(yj|xi)​Aθold​(xi,yj)h(x,y;\theta)=\frac{\nabla\_{\theta}\pi\_{\theta}(y\_{j}|x\_{i})}{\pi\_{\theta\_{\text{old}}}(y\_{j}|x\_{i})}A\_{\theta\_{\text{old}}}(x\_{i},y\_{j}). By applying the Lagrange multiplier method to solve the above problem, we obtain the optimal solution as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ni=σi∑jσj​N,σi=Tr​(𝕍y∼πθold​(h​(y,xi;θold))).n\_{i}=\frac{\sigma\_{i}}{\sum\_{j}\sigma\_{j}}N,\quad\sigma\_{i}=\sqrt{\text{Tr}\left(\mathbb{V}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left(h\left(y,x\_{i};\theta\_{\text{old}}\right)\right)\right)}. |  | (20) |

The remaining challenge is computing σi\sigma\_{i}. By expanding the variance and noting that the advantage function can be evaluated based on whether the rollout
yy is correct, i.e., Aθold​(x,y)=𝕀​(y​ is correct for ​x)−pθold​(x)A\_{\theta\_{\text{old}}}(x,y)=\mathbb{I}(y\text{ is correct for }x)-p\_{\theta\_{\text{old}}}(x),
we derive the following symmetric computational form:

|  |  |  |  |
| --- | --- | --- | --- |
|  |  | Tr​(𝕍y∼πθold​(h​(y,xi;θold)))\displaystyle\text{Tr}\left(\mathbb{V}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left(h\left(y,x\_{i};\theta\_{\text{old}}\right)\right)\right) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =pθold(xi)(1−pθold(xi))2𝔼y∼πθold,r=1[∥∇θlogπθ(y|xi)|θ=θold∥2]+(pθold(xi))2(1−pθold(xi))𝔼y∼πθold,r=0[∥∇θlogπθ(y|xi)|θ=θold∥2]−pθold(xi)2(1−pθold(xi))2∥𝔼y∼πθold,r=1[∇θlogπθ(y|xi))]−𝔼y∼πθold,r=0[∇θlogπθ(y|xi)]∥2.\displaystyle\begin{aligned} =&p\_{\theta\_{\text{old}}}(x\_{i})\left(1-p\_{\theta\_{\text{old}}}(x\_{i})\right)^{2}\underset{y\sim\pi\_{\theta\_{\text{old}}},r=1}{\mathbb{E}}\left[\left\|\nabla\_{\theta}\log\pi\_{\theta}(y|x\_{i})|\_{\theta=\theta\_{\text{old}}}\right\|^{2}\right]\\ +&\left(p\_{\theta\_{\text{old}}}(x\_{i})\right)^{2}(1-p\_{\theta\_{\text{old}}}(x\_{i}))\underset{y\sim\pi\_{\theta\_{\text{old}}},r=0}{\mathbb{E}}\left[\left\|\nabla\_{\theta}\log\pi\_{\theta}(y|x\_{i})|\_{\theta=\theta\_{\text{old}}}\right\|^{2}\right]\\ -&p\_{\theta\_{\text{old}}}(x\_{i})^{2}(1-p\_{\theta\_{\text{old}}}(x\_{i}))^{2}\left\|\underset{y\sim\pi\_{\theta\_{\text{old}}},r=1}{\mathbb{E}}[\nabla\_{\theta}\log\pi\_{\theta}(y|x\_{i}))]-\underset{y\sim\pi\_{\theta\_{\text{old}}},r=0}{\mathbb{E}}[\nabla\_{\theta}\log\pi\_{\theta}(y|x\_{i})]\right\|^{2}.\end{aligned} |  | (21) |

The optimized formula decomposes the variance estimation problem into two categories based on answer correctness, integrating it with the prompt difficulty estimation from Section [4.1](#S4.SS1 "4.1 Prompt Difficulty Caps Optimization Potential ‣ 4 Methodology ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs"). By leveraging algebraic operations on prompt difficulty and policy gradients, it reuses difficulty estimates from sampling and transforms variance estimation into a more tractable form. The theoretical proof is provided in Appendix [A.2](#A1.SS2 "A.2 Closing the Gap with Theoretical Bound ‣ Appendix A Mathematical Derivations ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs").

### 4.3 Prompt Difficulty Assessment and Algorithmic Implementation

Estimating prompt difficulty is crucial for both sampling questions and allocating rollout quantities. However, difficulty changes dynamically during policy training, making accurate estimation challenging. A straightforward approach is to add a pre-evaluation step before each sampling, but this increases computational overhead and fails to leverage new samples for posterior estimation to improve confidence. To address this, we propose a Bayesian inference framework that decomposes rollout into a multi-stage mini-batch process. This refines the posterior estimation of the dataset, dynamically adjusting the sampling distribution based on updated difficulty assessments.

Specifically, as the model πθold\pi\_{\theta\_{\text{old}}} rollouts on a prompt xix\_{i} multiple times, the number of correct answers follows a binomial distribution with success probability pθold​(xi)p\_{\theta\_{\text{old}}}(x\_{i}). We can assume that pθold​(xi)p\_{\theta\_{\text{old}}}(x\_{i}) follows a Beta distribution, the conjugate prior of the binomial distribution, which is a widely adopted technique in Bayesian inference (Kruschke, [2010](#bib.bib11); Qu et al., [2025](#bib.bib20)):

|  |  |  |  |
| --- | --- | --- | --- |
|  | pθold​(xi)∼Beta​(α0​(xi),β0​(xi)),\displaystyle p\_{\theta\_{\text{old}}}(x\_{i})\sim\text{Beta}(\alpha\_{0}(x\_{i}),\beta\_{0}(x\_{i})), |  | (22) |

where α0​(xi)\alpha\_{0}(x\_{i}) and β0​(xi)\beta\_{0}(x\_{i}) can be interpreted as the counts of correct and incorrect answers during sampling, which can be initialized using a small batch of sampled data for cold-start estimation.

Since the Beta distribution is conjugate to the binomial likelihood, the posterior distribution remains Beta-distributed after observing new samples. Let αt−1​(xi)\alpha\_{t-1}(x\_{i}) and βt−1​(xi)\beta\_{t-1}(x\_{i}) denote the cumulative counts of correct and incorrect answers for prompt xix\_{i} up to step t−1t-1. If, at step tt, a mini-batch generates nin\_{i} answer with ss correct, the posterior distribution for pθold​(xi)p\_{\theta\_{\text{old}}}(x\_{i}) after tt steps is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | αt​(xi)=αt−1​(xi)+s,βt​(xi)=βt−1​(xi)+ni−s,\displaystyle\alpha\_{t}(x\_{i})=\alpha\_{t-1}(x\_{i})+s,\quad\beta\_{t}(x\_{i})=\beta\_{t-1}(x\_{i})+n\_{i}-s, |  | (23) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | pθold​(xi)∼Beta​(αt​(xi),βt​(xi)).\displaystyle p\_{\theta\_{\text{old}}}(x\_{i})\sim\text{Beta}(\alpha\_{t}(x\_{i}),\beta\_{t}(x\_{i})). |  | (24) |

To reduce randomness, we use the mean of the Beta distribution to estimate prompt difficulty in our experiments.

This estimation approach enables modeling and estimating the difficulty of each prompt with minimal overhead. However, as the model’s performance evolves during training, the estimation process is susceptible to distribution shift, which becomes more pronounced with increasing training steps.
To mitigate this issue, we adopt a straightforward solution inspired by GVM (Yao et al., [2025](#bib.bib30)). Specifically, we divide the dataset into TT non-overlapped subsets and perform iterative training on these subsets.
We train the model for a fixed training steps of MM in every iteration.
The estimations of prompt difficulty and gradient variance are reset when a new iteration begins.
This method effectively alleviates distribution shift without introducing significant computational overhead. Moreover, the iterative process enables the model to adaptively adjust its sampling allocation based on its own evolving capabilities throughout training. For further details, please refer to Algorithm [1](#alg1 "Algorithm 1 ‣ 4.3 Prompt Difficulty Assessment and Algorithmic Implementation ‣ 4 Methodology ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs").

Algorithm 1  From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs (CurES)

1: Input: initial policy model πθ\pi\_{\theta}; reward function r​(x,y)r(x,y); prompt dataset 𝒟={xi}i=1N\mathcal{D}=\{x\_{i}\}\_{i=1}^{N}; number of iterations TT; prompt batch size mm; learning rate η\eta; parameter τ\tau; pre-rollout size N′N^{\prime}; number of steps per iteration MM.

2: for each t∈1,…,Tt\in 1,\dots,T do

3:  for each xi∈𝒟tx\_{i}\in\mathcal{D}\_{t} do

4:   Sample k=N′k=N^{\prime} rollouts {yi,1,…,yi,k}∼πθold(⋅|xi)\{y\_{i,1},\dots,y\_{i,k}\}\sim\pi\_{\theta\_{\text{old}}}(\cdot|x\_{i}).

5:   Compute rewards r​(xi,yi,j)r(x\_{i},y\_{i,j}).

6:   Initialize counts and difficulty estimations p​(xi)p(x\_{i}) according to Eq. ([22](#S4.E22 "In 4.3 Prompt Difficulty Assessment and Algorithmic Implementation ‣ 4 Methodology ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs")).

7:   Initialize sampling probabilities ρ∗\rho^{\*} according to Eq. ([14](#S4.E14 "In 4.1 Prompt Difficulty Caps Optimization Potential ‣ 4 Methodology ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs")).

8:   Compute gradient contribution according to Eq. ([4.2](#S4.Ex1 "4.2 Closing the Gap with Theoretical Bound ‣ 4 Methodology ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs")).

9:  end for

10:  for step = 1,⋯,M1,\cdots,M do

11:   Update the old policy model πθold←πθ\pi\_{\theta\_{\text{old}}}\leftarrow\pi\_{\theta}.

12:   Sample a batch of prompts ℬ={xi}i=1m\mathcal{B}=\{x\_{i}\}\_{i=1}^{m} with replacement according to ρ∗\rho^{\*}.

13:   Obtain rollout quantities nin\_{i} for xi∈ℬx\_{i}\in\mathcal{B} according to Eq. ([20](#S4.E20 "In 4.2 Closing the Gap with Theoretical Bound ‣ 4 Methodology ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs")).

14:   for each xi∈ℬx\_{i}\in\mathcal{B} do

15:    Sample k=nik=n\_{i} rollouts {yi,1,…,yi,k}∼πθold(⋅|xi)\{y\_{i,1},\dots,y\_{i,k}\}\sim\pi\_{\theta\_{\text{old}}}(\cdot|x\_{i}).

16:    Compute rewards r​(xi,yi,j)r(x\_{i},y\_{i,j}).

17:    Update counts and difficulty estimations p​(xi)p(x\_{i}) according to Eq. ([23](#S4.E23 "In 4.3 Prompt Difficulty Assessment and Algorithmic Implementation ‣ 4 Methodology ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs")).

18:   end for

19:   Update sampling probabilities ρ∗\rho^{\*} according to Eq. ([14](#S4.E14 "In 4.1 Prompt Difficulty Caps Optimization Potential ‣ 4 Methodology ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs")).

20:   Update policy πθ\pi\_{\theta} by applying RL training.

21:  end for

22: end for

23: Return πθ\pi\_{\theta}.

## 5 Experiments

In this section, we present comprehensive experimental results and analysis of our CurES with other baselines. Our experiments focus on the following research questions:

* •

  RQ1: Does CurES outperform other related baseline methods across various benchmarks?
* •

  RQ2: How does CurES adapt its sampling strategy to accelerate learning efficiency?
* •

  RQ3: Does CurES enhance sampling efficiency compared to other baseline methods?

Training Details.
We employ VERL (Sheng et al., [2025](#bib.bib23)) as our training framework and initialize our policy using Qwen2.5-Math models (1.5B and 7B parameters).
For the training dataset, we utilize Numina-Math (Li et al., [2024a](#bib.bib14)), partitioning it into 15 subsets following GVM (Yao et al., [2025](#bib.bib30)).
We conduct iterative training across these subsets, resulting in 15 training iterations.
At the beginning of each iteration, we perform 4 rollouts per prompt to establish an initial difficulty distribution and an assignment of rollout quantities under a total sample budget of 8×10248\times 1024.
During training, we sample prompts according to the difficulty distribution with replacement and conduct rollouts according to the assigned rollout quantities.
To make a fair comparison with GVM, we train 10 steps in each iteration.
We employ GRPO (Shao et al., [2024](#bib.bib22)) and REINFORCE++ (RPP) (Hu et al., [2025](#bib.bib10)) as advantage estimators for all methods.
The learning rate is set to a constant 1×10−61\times 10^{-6}.

Evaluation Benchmarks. To evaluate the complex reasoning capabilities, we choose a broad set of challenging reasoning benchmarks, including MATH500 (Hendrycks et al., [2021](#bib.bib9)), AIME 2024 and 2025 (Li et al., [2024a](#bib.bib14)), AMC 2023 (Li et al., [2024a](#bib.bib14)), GSM8K (Cobbe et al., [2021](#bib.bib5)), Gaokao-EN 2023 (Zhang et al., [2023](#bib.bib35)), Mineva (Lewkowycz et al., [2022](#bib.bib13)) and OlympiadBench (He et al., [2024](#bib.bib8)). These benchmarks comprehensively evaluate mathematical reasoning capabilities. Since AIME 2024, 2025 and AMC 2023 are highly challenging competition benchmarks, which are of limited sizes of test samples, we present the results averaged over 16 runs.

Baselines. To demonstrate the reasoning ability of our CurES, we compare it with many strong baseline methods: GRPO (Shao et al., [2024](#bib.bib22)), RPP (Hu et al., [2025](#bib.bib10)), Speed-RL (Zhang et al., [2025](#bib.bib34)) and GVM (Yao et al., [2025](#bib.bib30)).
Specifically, GRPO and RPP are commonly used in training mathematical problem solving models.
Speed-RL is an adaptive online RL curriculum that selectively chooses samples of intermediate difficulty to maximize learning efficiency (i.e., samples whose accuracy is not 0 or 11).
GVM is a prompt-specific dynamic sample allocation strategy designed to minimize stochastic gradient variance under a computational budget constraint.

### 5.1 Overall Performance (RQ1)

!(/html/2510.01037/assets/x2.png)

Figure 2: Comparison of learning curves between CurES and GVM across different backbone models and advantage estimators. CurES consistently outperforms GVM under the same number of training steps, demonstrating more efficient utilization of samples.

Table 1: Quantitative results of different methods across various datasets. The best and second best results are in bold and underlined.

| Method | Pass@1 | | | | | Average@16 | | | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MATH500 | GSM8K | GAO23 | MINERVA | OLYM | AIME24 | AIME25 | AMC23 |
| Qwen2.5-Math-1.5B | 40.20 | 43.90 | 25.19 | 11.40 | 21.04 | 1.67 | 1.67 | 14.84 | 20.00 |
| +GRPO | 73.80 | 86.43 | 48.83 | 27.94 | 35.41 | 8.54 | 6.67 | 45.47 | 41.64 |
| +RPP | 64.80 | 82.94 | 42.08 | 21.32 | 29.19 | 4.17 | 3.33 | 39.06 | 35.86 |
| +Speed-RL-GRPO | 68.80 | 85.67 | 47.14 | 27.40 | 35.56 | 12.08 | 6.88 | 47.19 | 41.34 |
| +Speed-RL-RPP | 65.80 | 85.67 | 48.31 | 27.94 | 36.30 | 12.58 | 10.08 | 47.34 | 41.75 |
| +GVM-GRPO | 74.80 | 84.23 | 48.83 | 27.21 | 35.56 | 10.21 | 11.25 | 50.47 | 42.82 |
| +GVM-RPP | 75.40 | 84.00 | 49.61 | 24.63 | 35.56 | 11.46 | 6.04 | 50.94 | 42.21 |
| \rowcolorblue!5  +CurES-GRPO | 77.20 | 85.97 | 51.43 | 31.62 | 37.33 | 13.33 | 10.42 | 52.19 | 44.94 |
| \rowcolorblue!5  +CurES-RPP | 75.40 | 85.82 | 51.43 | 28.31 | 37.04 | 12.71 | 11.46 | 50.94 | 44.14 |
| Qwen2.5-Math-7B | 60.20 | 72.40 | 44.68 | 22.79 | 30.81 | 7.92 | 1.88 | 27.19 | 33.48 |
| +GRPO | 80.00 | 91.43 | 51.43 | 31.99 | 38.37 | 20.00 | 10.00 | 57.50 | 47.59 |
| +RPP | 81.20 | 91.89 | 55.58 | 39.71 | 40.00 | 18.54 | 11.67 | 62.81 | 50.18 |
| +Speed-RL-GRPO | 82.80 | 88.70 | 55.58 | 29.41 | 42.37 | 20.21 | 11.46 | 60.16 | 48.84 |
| +Speed-RL-RPP | 78.60 | 91.81 | 53.77 | 37.13 | 42.96 | 17.29 | 12.08 | 62.81 | 49.56 |
| +GVM-GRPO | 81.60 | 91.28 | 54.03 | 32.72 | 42.67 | 23.54 | 15.00 | 64.31 | 50.64 |
| +GVM-RPP | 81.60 | 90.07 | 55.32 | 29.04 | 40.30 | 17.50 | 8.33 | 53.44 | 46.95 |
| \rowcolorblue!5  +CurES-GRPO | 84.80 | 92.27 | 56.62 | 37.87 | 43.56 | 24.58 | 15.21 | 64.38 | 52.41 |
| \rowcolorblue!5  +CurES-RPP | 81.80 | 91.89 | 54.55 | 33.09 | 40.59 | 23.33 | 12.92 | 58.75 | 49.62 |

!(/html/2510.01037/assets/x3.png)

Figure 3: The evolution of the estimated accuracy distributions for the Qwen2.5-Math-1.5B (left) and 7B (right) models across 15 iterations. Each violin shows the distribution of accuracy across samples: the width reflects density, the central line marks the median.

We present the learning curve of Qwen2.5-Math-1.5B and 7B models trained with different methods and advantage estimators in Figure [2](#S5.F2 "Figure 2 ‣ 5.1 Overall Performance (RQ1) ‣ 5 Experiments ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs").
Across all configurations, CurES exhibits higher progressive and final accuracy compared to GVM.
This advantage originates from two key differences: (i) CurES adaptively allocates prompt sampling probabilities based on estimated success rates, which our theoretical analysis confirms enhances training efficiency; (ii) while GVM monotonically decreases rollout allocation as accuracy increases, CurES allocates more rollout budget to prompts of moderate difficulty in Figure [4](#S5.F4 "Figure 4 ‣ 5.2 Sampling Behavior (RQ2) ‣ 5 Experiments ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs"), resulting in more consistent training gradients and improved training stability.

To demonstrate the effectiveness of our CurES, we compare it with representative baselines that are trained for the same number of steps.
The main results are demonstrated in Table [1](#S5.T1 "Table 1 ‣ 5.1 Overall Performance (RQ1) ‣ 5 Experiments ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs").
The results clearly demonstrate that CurES consistently outperforms GVM and other baselines with both GRPO and RPP as advantage estimators.
Across both model scales, CurES establishes state-of-the-art results on several datasets and consistently matches or surpasses the strongest baselines across all settings, confirming the superior generalization ability of our CurES.

### 5.2 Sampling Behavior (RQ2)

!(/html/2510.01037/assets/x4.png)

Figure 4: Allocation of rollout quantities with respect to accuracy in CurES at different training iterations. CurES concentrates more rollouts on moderately difficult prompts.

Figure [3](#S5.F3 "Figure 3 ‣ 5.1 Overall Performance (RQ1) ‣ 5 Experiments ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs") illustrates the evolution of the difficulty distribution for both the Qwen2.5-Math-1.5B and 7B models throughout the training process.
At iteration 1, the estimated accuracy is broadly distributed.
Subsequently, as training progresses, this distribution shifts toward higher values and becomes more concentrated, indicating that the models are effectively learning and mastering the presented samples.
This shift also underscores the importance of redistributing prompt sampling probabilities, as the models’ success rates on problems are primarily bimodal, concentrated at high and low values. Such redistribution enhances training efficiency, whereas uniform sampling followed by answer generation and accuracy-based filtering significantly reduces efficiency under this bimodal distribution.

Meanwhile, another view of how the CurES method dynamically adjusts the rollout quantities assignment during training is presented in Figure [4](#S5.F4 "Figure 4 ‣ 5.2 Sampling Behavior (RQ2) ‣ 5 Experiments ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs").
The trend lines of all iterations approximate a “bell-shaped” distribution, with prompts of intermediate accuracy allocated more rollout quantities, as anticipated given their high efficiency. Furthermore, as the training procedure continues, the distribution becomes progressively sharper and narrower, indicating that CurES dynamically increases rollout quantities for moderately difficult prompts. This pattern aligns with the observation in Figure [3](#S5.F3 "Figure 3 ‣ 5.1 Overall Performance (RQ1) ‣ 5 Experiments ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs"), which shows a gradual reduction in moderately difficult prompts as the model improves. By adaptively increasing rollout quantities, CurES compensates for their diminishing presence, ensuring they remain a substantial portion of each training batch.
By coupling difficulty-based prompt sampling with the adaptive sample sizing, CurES sustains an abundance of informative prompts and thereby maximizes performance gains per step.

### 5.3 Efficiency Analysis (RQ3)

!(/html/2510.01037/assets/x5.png)

Figure 5: Performance convergence of CurES on MATH500 with different sampling configurations.

!(/html/2510.01037/assets/x6.png)

Figure 6: Efficiency comparison of CurES against baselines on MATH500. Gray dashed lines indicate the steps required for CurES and the baseline to reach the highest average accuracy of the baseline during the entire training period.

To analyze the effect of different combinations of pre-sampling scale (N′N^{\prime}) and training-phase sample budgets coefficient (nn), which determines a training-phase sample budget of n×mn\times m, on model performance convergence, we conducted experiments as depicted in Figure [5](#S5.F5 "Figure 5 ‣ 5.3 Efficiency Analysis (RQ3) ‣ 5 Experiments ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs").
A larger N′N^{\prime} leads to a more accurate initial accuracy estimation, while a larger nn provides a greater computation budget.
The results show that increasing either N′N^{\prime} or nn does not yield a proportional performance benefit relative to the increased computational cost.
This finding underscores the efficiency of CurES, which effectively directs the model toward high-yield learning samples with minimal computational overhead, highlighting its superior sample efficiency.

We also provide a direct comparison of CurES against GRPO and RPP in Figure [6](#S5.F6 "Figure 6 ‣ 5.3 Efficiency Analysis (RQ3) ‣ 5 Experiments ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs").
The plots show the learning curve of each method over training steps.
CurES-GRPO achieves the same peak performance as the GRPO in just 5.5×\times fewer steps. Similarly, CurES-RPP reaches its peak performance 1.75x faster than the RPP baseline.
The remarkable sample efficiency is a direct consequence of CurES’s ability to consistently provide the model with optimally challenging samples.

## 6 Conclusion

In this paper, we propose CurES, a theoretically grounded curriculum learning algorithm for RLVR.
By linking gradient efficiency to accuracy, our approach adaptively prioritizes training prompts of optimal difficulty and dynamically allocates rollout budgets.
Beyond the theoretical analysis, our algorithmic design leverages a Bayesian framework to track prompt difficulty in a lightweight yet adaptive manner.
Specifically, we model the success rate of each prompt instance with a Beta distribution, which naturally incorporates prior information and posterior updates as new rollouts are observed.
Combined with the derived sampling distribution and variance-based rollout quantity allocation, this Bayesian mechanism ensures that both question selection and sample budgeting adapt dynamically to the evolving policy, thereby maximizing training efficiency in practice.
Experiments on a wide range of mathematical reasoning benchmarks show that CurES consistently outperforms strong baselines in both accuracy and convergence speed, demonstrating superior sample efficiency.

## References

* Bae et al. (2025)

  Sanghwan Bae, Jiwoo Hong, Min Young Lee, Hanbyul Kim, JeongYeon Nam, and Donghyun Kwak.
  Online difficulty filtering for reasoning oriented reinforcement learning.
  *arXiv preprint arXiv:2504.03380*, 2025.
* Bottou et al. (2018)

  Léon Bottou, Frank E Curtis, and Jorge Nocedal.
  Optimization methods for large-scale machine learning.
  *SIAM review*, 60(2):223–311, 2018.
* Chen et al. (2024)

  Shuang Chen, Changlun Zhang, and Haibing Mu.
  An adaptive learning rate deep learning optimizer using long and short-term gradients based on g–l fractional-order derivative.
  *Neural Processing Letters*, 56(2):106, 2024.
* Chhabra et al. (2024)

  Anshuman Chhabra, Bo Li, Jian Chen, Prasant Mohapatra, and Hongfu Liu.
  Outlier gradient analysis: Efficiently identifying detrimental training samples for deep learning models.
  *arXiv preprint arXiv:2405.03869*, 2024.
* Cobbe et al. (2021)

  Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, et al.
  Training verifiers to solve math word problems.
  *CoRR*, abs/2110.14168, 2021.
  URL <https://arxiv.org/abs/2110.14168>.
* Dereich et al. (2024)

  Steffen Dereich, Arnulf Jentzen, and Adrian Riekert.
  Learning rate adaptive stochastic gradient descent optimization methods: numerical simulations for deep learning methods for partial differential equations and convergence analyses.
  *arXiv preprint arXiv:2406.14340*, 2024.
* Guo et al. (2025)

  Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al.
  Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.
  *arXiv preprint arXiv:2501.12948*, 2025.
* He et al. (2024)

  Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al.
  Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems.
  *arXiv preprint arXiv:2402.14008*, 2024.
* Hendrycks et al. (2021)

  Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt.
  Measuring mathematical problem solving with the math dataset.
  *arXiv preprint arXiv:2103.03874*, 2021.
* Hu et al. (2025)

  Jian Hu, Jason Klein Liu, Haotian Xu, and Wei Shen.
  Reinforce++: An efficient rlhf algorithm with robustness to both prompt and reward models, 2025.
  URL <https://arxiv.org/abs/2501.03262>.
* Kruschke (2010)

  John K Kruschke.
  Bayesian data analysis.
  *Wiley Interdisciplinary Reviews: Cognitive Science*, 1(5):658–676, 2010.
* Lambert et al. (2024)

  Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al.
  T\\backslash” ulu 3: Pushing frontiers in open language model post-training.
  *arXiv preprint arXiv:2411.15124*, 2024.
* Lewkowycz et al. (2022)

  Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al.
  Solving quantitative reasoning problems with language models.
  *Advances in Neural Information Processing Systems*, 35:3843–3857, 2022.
* Li et al. (2024a)

  Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, et al.
  Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions.
  *Hugging Face repository*, 13:9, 2024a.
* Li et al. (2024b)

  Xinyu Li, Wenqing Ye, Yueyi Zhang, and Xiaoyan Sun.
  Grace: Gradient-based active learning with curriculum enhancement for multimodal sentiment analysis.
  In *Proceedings of the 32nd ACM International Conference on Multimedia*, pp. 5702–5711, 2024b.
* Lin et al. (2025)

  Zhihang Lin, Mingbao Lin, Yuan Xie, and Rongrong Ji.
  Cppo: Accelerating the training of group relative policy optimization-based reasoning models.
  *arXiv preprint arXiv:2503.22342*, 2025.
* Luo et al. (2025)

  Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Tianjun Zhang, Li Erran Li, et al.
  Deepscaler: Surpassing o1-preview with a 1.5 b model by scaling rl.
  *Notion Blog*, 2025.
* Medyakov et al. (2025)

  Daniil Medyakov, Gleb Molodtsov, Savelii Chezhegov, Alexey Rebrikov, and Aleksandr Beznosikov.
  Variance reduction methods do not need to compute full gradients: Improved efficiency through shuffling.
  *arXiv preprint arXiv:2502.14648*, 2025.
* Papini et al. (2018)

  Matteo Papini, Damiano Binaghi, Giuseppe Canonaco, Matteo Pirotta, and Marcello Restelli.
  Stochastic variance-reduced policy gradient.
  In *International conference on machine learning*, pp. 4026–4035. PMLR, 2018.
* Qu et al. (2025)

  Yun Qu, Qi Wang, Yixiu Mao, Vincent Tao Hu, Björn Ommer, and Xiangyang Ji.
  Can prompt difficulty be online predicted for accelerating rl finetuning of reasoning models?
  *arXiv preprint arXiv:2507.04632*, 2025.
* Ruder (2016)

  Sebastian Ruder.
  An overview of gradient descent optimization algorithms.
  *arXiv preprint arXiv:1609.04747*, 2016.
* Shao et al. (2024)

  Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al.
  Deepseekmath: Pushing the limits of mathematical reasoning in open language models.
  *arXiv preprint arXiv:2402.03300*, 2024.
* Sheng et al. (2025)

  Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu.
  Hybridflow: A flexible and efficient RLHF framework.
  In *Proceedings of the Twentieth European Conference on Computer Systems, EuroSys 2025, Rotterdam, The Netherlands, 30 March 2025 - 3 April 2025*, pp. 1279–1297. ACM, 2025.
* Shi et al. (2025)

  Taiwei Shi, Yiyang Wu, Linxin Song, Tianyi Zhou, and Jieyu Zhao.
  Efficient reinforcement finetuning via adaptive curriculum learning.
  *arXiv preprint arXiv:2504.05520*, 2025.
* Song et al. (2025)

  Mingyang Song, Mao Zheng, Zheng Li, Wenjie Yang, Xuan Luo, Yue Pan, and Feng Zhang.
  Fastcurl: Curriculum reinforcement learning with stage-wise context scaling for efficient training r1-like reasoning models.
  *arXiv preprint arXiv:2503.17287*, 2025.
* Team et al. (2025)

  Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al.
  Kimi k1. 5: Scaling reinforcement learning with llms.
  *arXiv preprint arXiv:2501.12599*, 2025.
* Wang et al. (2013)

  Chong Wang, Xi Chen, Alexander J Smola, and Eric P Xing.
  Variance reduction for stochastic gradient optimization.
  *Advances in neural information processing systems*, 26, 2013.
* Xie et al. (2025)

  Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu, and Chong Luo.
  Logic-rl: Unleashing llm reasoning with rule-based reinforcement learning.
  *arXiv preprint arXiv:2502.14768*, 2025.
* Yang et al. (2024)

  Hongru Yang, Bhavya Kailkhura, Zhangyang Wang, Yingbin Liang, et al.
  Training dynamics of transformers to recognize word co-occurrence via gradient flow analysis.
  *Advances in Neural Information Processing Systems*, 37:46047–46117, 2024.
* Yao et al. (2025)

  Jiarui Yao, Yifan Hao, Hanning Zhang, Hanze Dong, Wei Xiong, Nan Jiang, and Tong Zhang.
  Optimizing chain-of-thought reasoners via gradient variance minimization in rejection sampling and RL.
  *CoRR*, abs/2505.02391, 2025.
  doi: 10.48550/ARXIV.2505.02391.
  URL <https://doi.org/10.48550/arXiv.2505.02391>.
* Yu et al. (2025)

  Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al.
  Dapo: An open-source llm reinforcement learning system at scale.
  *arXiv preprint arXiv:2503.14476*, 2025.
* Yuan et al. (2024)

  Huizhuo Yuan, Yifeng Liu, Shuang Wu, Xun Zhou, and Quanquan Gu.
  Mars: Unleashing the power of variance reduction for training large models.
  *arXiv preprint arXiv:2411.10438*, 2024.
* Zeng et al. (2025)

  Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He.
  Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild.
  *arXiv preprint arXiv:2503.18892*, 2025.
* Zhang et al. (2025)

  Ruiqi Zhang, Daman Arora, Song Mei, and Andrea Zanette.
  SPEED-RL: faster training of reasoning models via online curriculum learning.
  *CoRR*, abs/2506.09016, 2025.
  doi: 10.48550/ARXIV.2506.09016.
  URL <https://doi.org/10.48550/arXiv.2506.09016>.
* Zhang et al. (2023)

  Xiaotian Zhang, Chunyang Li, Yi Zong, Zhengyu Ying, Liang He, and Xipeng Qiu.
  Evaluating the performance of large language models on GAOKAO benchmark.
  *CoRR*, abs/2305.12474, 2023.
  doi: 10.48550/ARXIV.2305.12474.
  URL <https://doi.org/10.48550/arXiv.2305.12474>.
* Zhao & Xu (2024)

  Hongru Zhao and Jinchao Xu.
  Convergence analysis and trajectory comparison of gradient descent for overparameterized deep linear networks.
  *Transactions on Machine Learning Research*, 2024.

## Appendix A Mathematical Derivations

###### Lemma 1 (Cramér-Rao Inequality)

Let {pθ​(x),θ∈Θ}\{p\_{\theta}(x),\theta\in\Theta\} be a Cramér-Rao regular family with parameter space Θ⊂ℝk\Theta\subset\mathbb{R}^{k}, where the Fisher information matrix I​(θ)I(\theta) is non-singular. Let g​(θ)=(g1​(θ),⋯,gs​(θ))⊤g(\theta)=(g\_{1}(\theta),\cdots,g\_{s}(\theta))^{\top} for s≤ks\leq k, and assume the partial derivatives ∂gi​(θ)/∂θj\partial g\_{i}(\theta)/\partial\theta\_{j} exist for all i=1,⋯,si=1,\cdots,s and j=1,⋯,kj=1,\cdots,k. Suppose T​(X)T(X) is an unbiased estimator of g​(θ)g(\theta) with finite second moment. Denote
G​(θ)=∇θg​(θ)G(\theta)=\nabla\_{\theta}g(\theta), then we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝕍θ​(T​(X))≥G​(θ)​I−1​(θ)​G⊤​(θ).\mathbb{V}\_{\theta}(T(X))\geq G(\theta)I^{-1}(\theta)G^{\top}(\theta). |  | (25) |

### A.1 Prompt Difficulty Caps Optimization Potential

Given an individual sample xx, we first consider the optimization problem as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | min⁡ℒ​(x;θ)=min−𝔼y∼πθ(⋅|x)​[Aθold​(x,y)],s.t.DKL(πθold(⋅|x)∥πθ(⋅|x))≤δ\begin{gathered}\min\mathcal{L}(x;\theta)=\min-\mathbb{E}\_{{y}\sim\pi\_{\theta}(\cdot|{x})}\big[A\_{\theta\_{\mathrm{old}}}(x,y)\big],\\ \text{s.t.}\ D\_{\mathrm{KL}}(\pi\_{\theta\_{\mathrm{old}}}(\cdot|x)\|\pi\_{\theta}(\cdot|x))\leq\delta\end{gathered} |  | (26) |

We define θ=θold+d\theta=\theta\_{\mathrm{old}}+d and rewrite this constrained optimization problem via the Lagrange multiplier method:

|  |  |  |  |
| --- | --- | --- | --- |
|  | d∗=argmindℒ(x;θold+d)+λ(DKL(πθold(⋅|x)∥πθold+d(⋅|x))−δ).d^{\*}=\operatorname\*{argmin}\_{d}\mathcal{L}(x;\theta\_{\mathrm{old}}+d)+\lambda(D\_{\mathrm{KL}}(\pi\_{\theta\_{\mathrm{old}}}(\cdot|x)\|\pi\_{\theta\_{\mathrm{old}}+d}(\cdot|x))-\delta). |  | (27) |

Using the Taylor expansion formula, we have:

|  |  |  |  |
| --- | --- | --- | --- |
|  | d∗=\displaystyle d^{\*}= | argmindℒ(x;θold+d)+λ(DKL(πθold(⋅|x)∥πθold+d(⋅|x))−δ)\displaystyle\operatorname\*{argmin}\_{d}\mathcal{L}(x;\theta\_{\text{old}}+d)+\lambda(D\_{\mathrm{KL}}(\pi\_{\theta\_{\mathrm{old}}}(\cdot|x)\|\pi\_{\theta\_{\text{old}}+d}(\cdot|x))-\delta) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =\displaystyle= | argmindℒ​(x;θold)−λ​δ\displaystyle\operatorname\*{argmin}\_{d}\mathcal{L}(x;\theta\_{\mathrm{old}})-\lambda\delta |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +(∇θℒ(x;θ)⊤d+λ∇θDKL(πθold(⋅|x)∥πθ(⋅|x))d+λ2d⊤∇θ2DKL(πθold(⋅|x)∥πθ(⋅|x))d)|θ=θold.\displaystyle\begin{aligned} +\left(\nabla\_{\theta}\mathcal{L}(x;\theta)^{\top}d+\lambda\nabla\_{\theta}D\_{\mathrm{KL}}(\pi\_{\theta\_{\mathrm{old}}}(\cdot|x)\|\pi\_{\theta}(\cdot|x))d+\frac{\lambda}{2}d^{\top}\nabla\_{\theta}^{2}D\_{\mathrm{KL}}(\pi\_{\theta\_{\mathrm{old}}}(\cdot|x)\|\pi\_{\theta}(\cdot|x))d\right)\Big|\_{\theta=\theta\_{\mathrm{old}}}.\end{aligned} |  | (28) |

We first compute the first-order and second-order derivatives of the KL divergence term:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ∇θDKL(πθold(⋅|x)∥πθ(⋅|x))|θ=θold\displaystyle\nabla\_{\theta}D\_{\mathrm{KL}}(\pi\_{\theta\_{\mathrm{old}}}(\cdot|x)\|\pi\_{\theta}(\cdot|x))\big|\_{\theta=\theta\_{\mathrm{old}}} |  | (29) |
|  | =\displaystyle= | ∇θ𝔼y∼πθold​[log⁡πθold​(y|x)]|θ=θold−∇θ𝔼y∼πθold​[log⁡πθ​(y|x)]|θ=θold\displaystyle\nabla\_{\theta}\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left[\log\pi\_{\theta\_{\text{old}}}(y|x)\right]\big|\_{\theta=\theta\_{\text{old}}}-\nabla\_{\theta}\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left[\log\pi\_{\theta}(y|x)\right]\big|\_{\theta=\theta\_{\text{old}}} |  |
|  | =\displaystyle= | −𝔼y∼πθold​[∇θlog⁡πθ​(y|x)]|θ=θold\displaystyle-\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left[\nabla\_{\theta}\log\pi\_{\theta}(y|x)\right]\big|\_{\theta=\theta\_{\text{old}}} |  |
|  | =\displaystyle= | −𝔼y∼πθold​[∇θπθ​(y|x)πθ​(y|x)]|θ=θold\displaystyle-\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left[\frac{\nabla\_{\theta}\pi\_{\theta}(y|x)}{\pi\_{\theta}(y|x)}\right]\Big|\_{\theta=\theta\_{\text{old}}} |  |
|  | =\displaystyle= | ∑y∇θπθ​(y|x)|θ=θold\displaystyle\sum\limits\_{y}\nabla\_{\theta}\pi\_{\theta}(y|x)\big|\_{\theta=\theta\_{\text{old}}} |  |
|  | =\displaystyle= | ∇θ​∑yπθ​(y|x)|θ=θold\displaystyle\nabla\_{\theta}\sum\limits\_{y}\pi\_{\theta}(y|x)\big|\_{\theta=\theta\_{\text{old}}} |  |
|  | =\displaystyle= | 0.\displaystyle 0. |  |

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ∇θ2DKL(πθold(⋅|x)∥πθ(⋅|x))|θ=θold\displaystyle\nabla\_{\theta}^{2}D\_{\mathrm{KL}}(\pi\_{\theta\_{\mathrm{old}}}(\cdot|x)\|\pi\_{\theta}(\cdot|x))\big|\_{\theta=\theta\_{\mathrm{old}}} |  | (30) |
|  | =\displaystyle= | −∇θ2𝔼y∼πθold​[log⁡πθ​(y|x)]|θ=θold\displaystyle-\nabla\_{\theta}^{2}\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left[\log\pi\_{\theta}(y|x)\right]\big|\_{\theta=\theta\_{\text{old}}} |  |
|  | =\displaystyle= | −𝔼y∼πθold​[∇θ2log⁡πθ​(y|x)]|θ=θold\displaystyle-\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left[\nabla\_{\theta}^{2}\log\pi\_{\theta}(y|x)\right]\big|\_{\theta=\theta\_{\text{old}}} |  |
|  | =\displaystyle= | −𝔼y∼πθold​[∇θ(∇θπθ​(y|x)πθ​(y|x))]|θ=θold\displaystyle-\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left[\nabla\_{\theta}\left(\frac{\nabla\_{\theta}\pi\_{\theta}(y|x)}{\pi\_{\theta}(y|x)}\right)\right]\Big|\_{\theta=\theta\_{\text{old}}} |  |
|  | =\displaystyle= | −𝔼y∼πθold​[(∇θ2πθ(y|x))πθ(y|x)−∇θπθ(y|x))∇θ⊤πθ(y|x)πθ2​(y|x)]|θ=θold\displaystyle-\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left[\frac{(\nabla\_{\theta}^{2}\pi\_{\theta}(y|x))\pi\_{\theta}(y|x)-\nabla\_{\theta}\pi\_{\theta}(y|x))\nabla\_{\theta}^{\top}\pi\_{\theta}(y|x)}{\pi\_{\theta}^{2}(y|x)}\right]\Big|\_{\theta=\theta\_{\text{old}}} |  |
|  | =\displaystyle= | −𝔼y∼πθold​[(∇θ2πθ​(y|x))|θ=θoldπθold​(y|x)]+𝔼y∼πθold​[(∇θπθ​(y|x)πθ​(y|x))​(∇θπθ​(y|x)πθ​(y|x))⊤]|θ=θold\displaystyle-\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left[\frac{(\nabla\_{\theta}^{2}\pi\_{\theta}(y|x))|\_{\theta=\theta\_{\text{old}}}}{\pi\_{\theta\_{\text{old}}}(y|x)}\right]+\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left[\left(\frac{\nabla\_{\theta}\pi\_{\theta}(y|x)}{\pi\_{\theta}(y|x)}\right)\left(\frac{\nabla\_{\theta}\pi\_{\theta}(y|x)}{\pi\_{\theta}(y|x)}\right)^{\top}\right]\Big|\_{\theta=\theta\_{\text{old}}} |  |
|  | =\displaystyle= | 𝔼y∼πθold​[∇θlog⁡πθ​(y|x)​∇θlog⁡πθ​(y|x)⊤]|θ=θold\displaystyle\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left[\nabla\_{\theta}\log\pi\_{\theta}(y|x)\nabla\_{\theta}\log\pi\_{\theta}(y|x)^{\top}\right]\Big|\_{\theta=\theta\_{\text{old}}} |  |
|  | =\displaystyle= | F​(x;θold).\displaystyle F(x;\theta\_{\text{old}}). |  |

where F​(x;θ)=𝔼y∼πθ​[∇θlog⁡πθ​(y|x)​∇θlog⁡πθ​(y|x)⊤]F(x;\theta)=\mathbb{E}\_{y\sim\pi\_{\theta}}\left[\nabla\_{\theta}\log\pi\_{\theta}(y|x)\nabla\_{\theta}\log\pi\_{\theta}(y|x)^{\top}\right] is termed the Fisher information matrix.

Therefore,

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | d∗\displaystyle d^{\*} | =argmindℒ​(x;θold)+∇θℒ​(x;θ)⊤|θ=θold​d+λ2​d⊤​F​(x;θold)​d−λ​δ.\displaystyle=\operatorname\*{argmin}\_{d}\mathcal{L}(x;\theta\_{\mathrm{old}})+\nabla\_{\theta}\mathcal{L}(x;\theta)^{\top}\Big|\_{\theta=\theta\_{\text{old}}}d+\frac{\lambda}{2}d^{\top}F(x;\theta\_{\text{old}})d-\lambda\delta. |  | (31) |

To find the minimum, we take the derivative of the right-hand side and set it to zero:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇θℒ​(x;θ)|θ=θold+λ​F​(x;θold)​d=0.\displaystyle\nabla\_{\theta}\mathcal{L}(x;\theta)\Big|\_{\theta=\theta\_{\text{old}}}+\lambda F(x;\theta\_{\text{old}})d=0. |  | (32) |

So we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | d=−1λ​F−1​(x;θold)​∇θℒ​(x;θ)|θ=θold.\displaystyle d=-\frac{1}{\lambda}F^{-1}(x;\theta\_{\text{old}})\nabla\_{\theta}\mathcal{L}(x;\theta)\Big|\_{\theta=\theta\_{\text{old}}}. |  | (33) |

We now derive the critical point of the constraint condition:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 12d⊤F(x;θold)d≈DKL(πθold(⋅|x)∥πθold+d(⋅|x))=δ.\frac{1}{2}d^{\top}F(x;\theta\_{\text{old}})d\approx D\_{\mathrm{KL}}(\pi\_{\theta\_{\mathrm{old}}}(\cdot|x)\|\pi\_{\theta\_{\mathrm{old}}+d}(\cdot|x))=\delta. |  | (34) |

By simplifying, we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | 12​(1λ2​∇θℒ​(x;θ)⊤|θ=θold​F−1​(x;θold)​∇θℒ​(x;θ)|θ=θold)=δ.\frac{1}{2}\left(\frac{1}{\lambda^{2}}\nabla\_{\theta}\mathcal{L}(x;\theta)^{\top}\big|\_{\theta=\theta\_{\text{old}}}F^{-1}(x;\theta\_{\text{old}})\nabla\_{\theta}\mathcal{L}(x;\theta)\big|\_{\theta=\theta\_{\text{old}}}\right)=\delta. |  | (35) |

Therefore, we have obtained the critical value of λ\lambda:

|  |  |  |  |
| --- | --- | --- | --- |
|  | λ=∇θℒ​(x;θ)⊤|θ=θold​F−1​(x;θold)​∇θℒ​(x;θ)|θ=θold2​δ.\lambda=\sqrt{\frac{\nabla\_{\theta}\mathcal{L}(x;\theta)^{\top}\big|\_{\theta=\theta\_{\text{old}}}F^{-1}(x;\theta\_{\text{old}})\nabla\_{\theta}\mathcal{L}(x;\theta)\big|\_{\theta=\theta\_{\text{old}}}}{2\delta}}. |  | (36) |

In this case, the change of the loss function is computed as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ℒ​(x;θold+d)−ℒ​(x;θold)\displaystyle\mathcal{L}(x;\theta\_{\text{old}}+d)-\mathcal{L}(x;\theta\_{\text{old}}) | =∇θℒ​(x;θ)⊤|θ=θold​d\displaystyle=\nabla\_{\theta}\mathcal{L}(x;\theta)^{\top}\big|\_{\theta=\theta\_{\text{old}}}d |  | (37) |
|  |  | =−1λ​∇θℒ​(x;θ)⊤|θ=θold​F−1​(x;θold)​∇θℒ​(x;θ)|θ=θold\displaystyle=-\frac{1}{\lambda}\nabla\_{\theta}\mathcal{L}(x;\theta)^{\top}\big|\_{\theta=\theta\_{\text{old}}}F^{-1}(x;\theta\_{\text{old}})\nabla\_{\theta}\mathcal{L}(x;\theta)\big|\_{\theta=\theta\_{\text{old}}} |  |
|  |  | =−2​δ​∇θℒ​(x;θ)⊤|θ=θold​F−1​(x;θold)​∇θℒ​(x;θ)|θ=θold\displaystyle=-\sqrt{2\delta\nabla\_{\theta}\mathcal{L}(x;\theta)^{\top}\big|\_{\theta=\theta\_{\text{old}}}F^{-1}(x;\theta\_{\text{old}})\nabla\_{\theta}\mathcal{L}(x;\theta)\big|\_{\theta=\theta\_{\text{old}}}} |  |

Furthermore,

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∇θℒ​(x;θ)|θ=θold\displaystyle\nabla\_{\theta}\mathcal{L}(x;\theta)|\_{\theta=\theta\_{\text{old}}} | =∇θ−𝔼y∼πθ(⋅|x)​[Aθold​(x,y)]|θ=θold\displaystyle=\nabla\_{\theta}-\mathbb{E}\_{{y}\sim\pi\_{\theta}(\cdot|{x})}\big[A\_{\theta\_{\mathrm{old}}}(x,y)\big]\big|\_{\theta=\theta\_{\text{old}}} |  | (38) |
|  |  | =−∑y[∇θπθ​(y|x)​(r​(x,y)−𝔼y∼πθold​[r​(x,y)])]|θ=θold\displaystyle=-\sum\limits\_{y}\left[\nabla\_{\theta}\pi\_{\theta}(y|x)\left(r(x,y)-\mathbb{E}\_{y\sim\pi\_{\theta\_{\mathrm{old}}}}[r(x,y)]\right)\right]\big|\_{\theta=\theta\_{\text{old}}} |  |
|  |  | =−∑y[∇θπθ​(y|x)​(r​(x,y))]|θ=θold\displaystyle=-\sum\limits\_{y}\left[\nabla\_{\theta}\pi\_{\theta}(y|x)\left(r(x,y)\right)\right]\big|\_{\theta=\theta\_{\text{old}}} |  |
|  |  | =−∇θ𝔼y∼πθ​[r​(x,y)]|θ=θold\displaystyle=-\nabla\_{\theta}\mathbb{E}\_{y\sim\pi\_{\theta}}[r(x,y)]\big|\_{\theta=\theta\_{\text{old}}} |  |
|  |  | =−∇θpθ​(x)|θ=θold,\displaystyle=-\nabla\_{\theta}p\_{\theta}(x)|\_{\theta=\theta\_{\text{old}}}, |  |

where pθ​(x)p\_{\theta}(x) is the model’s question-answering accuracy.

Consider that r​(x,y)r(x,y) is an unbiased estimator of pθ​(x)p\_{\theta}(x), according to the Cramér-Rao inequality, we obtain:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | |(ℒ​(θold+d)−ℒ​(θold))|\displaystyle\quad\ |\left(\mathcal{L}(\theta\_{\text{old}}+d)-\mathcal{L}(\theta\_{\text{old}})\right)| | =|𝔼x∼ρ​[ℒ​(x;θold+d)−ℒ​(x;θold)]|\displaystyle=|\mathbb{E}\_{x\sim\rho}\left[\mathcal{L}(x;\theta\_{\text{old}}+d)-\mathcal{L}(x;\theta\_{\text{old}})\right]| |  | (39) |
|  |  | ≤𝔼x∼ρ​[|(ℒ​(x;θold+d)−ℒ​(x;θold))|]\displaystyle\leq\mathbb{E}\_{x\sim\rho}\left[|\left(\mathcal{L}(x;\theta\_{\text{old}}+d)-\mathcal{L}(x;\theta\_{\text{old}})\right)|\right] |  |
|  |  | =𝔼x∼ρ​[2​δ​∇θℒ​(x;θ)⊤|θ=θold​F−1​(x;θold)​∇θℒ​(x;θ)|θ=θold]\displaystyle=\mathbb{E}\_{x\sim\rho}\left[\sqrt{2\delta\nabla\_{\theta}\mathcal{L}(x;\theta)^{\top}\big|\_{\theta=\theta\_{\text{old}}}F^{-1}(x;\theta\_{\text{old}})\nabla\_{\theta}\mathcal{L}(x;\theta)\big|\_{\theta=\theta\_{\text{old}}}}\right] |  |
|  |  | ≤𝔼x∼ρ​[2​δ​𝕍θold​(r​(x,y))]\displaystyle\leq\mathbb{E}\_{x\sim\rho}\left[\sqrt{2\delta\mathbb{V}\_{\theta\_{\text{old}}}(r(x,y))}\right] |  |
|  |  | =𝔼x∼ρ​[2​δ​pθold​(x)​(1−pθold​(x))]\displaystyle=\mathbb{E}\_{x\sim\rho}\left[\sqrt{2\delta p\_{\theta\_{\text{old}}}(x)\left(1-p\_{\theta\_{\text{old}}}(x)\right)}\right] |  |

This indicates that the optimization potential of the loss function is inherently related to the difficulty of the prompt itself. To balance the trade-off between exploration and exploitation, we derive the optimal sampling distribution by solving the following objective function under the maximum entropy constraint with the hyperparameter α\alpha:

|  |  |  |  |
| --- | --- | --- | --- |
|  | max⁡𝔼x∼ρ​[2​δ​pθold​(x)​(1−pθold​(x))+α​ℋ​(ρ)],s.t.​∑i=1Nρ​(xi)=1,ρ​(xi)≥0\begin{gathered}\max\ \mathbb{E}\_{x\sim\rho}\left[\sqrt{2\delta p\_{\theta\_{\text{old}}}(x)\left(1-p\_{\theta\_{\text{old}}}(x)\right)}+\alpha\mathcal{H}(\rho)\right],\\ \text{s.t.}\ \sum\_{i=1}^{N}\rho(x\_{i})=1,\quad\rho(x\_{i})\geq 0\end{gathered} |  | (40) |

To find the optimal distribution ρ\rho, we employ the method of Lagrange multipliers. The objective function becomes:

|  |  |  |  |
| --- | --- | --- | --- |
|  | max⁡𝒥=max​∑j=1Nρ​(xj)​2​δ​pθold​(xj)​(1−pθold​(xj))−α​∑j=1Nρ​(xj)​log⁡ρ​(xj)+μ​(1−∑j=1Nρ​(xj)),\max\ \mathcal{J}=\max\ \sum\_{j=1}^{N}\rho(x\_{j})\sqrt{2\delta p\_{\theta\_{\text{old}}}(x\_{j})\left(1-p\_{\theta\_{\text{old}}}(x\_{j})\right)}-\alpha\sum\_{j=1}^{N}\rho(x\_{j})\log\rho(x\_{j})+\mu\left(1-\sum\_{j=1}^{N}\rho(x\_{j})\right), |  | (41) |

where μ\mu is the Lagrange multiplier associated with the normalization constraint.

Taking the partial derivative of 𝒥\mathcal{J} with respect to ρ​(xj)\rho(x\_{j}):

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂𝒥∂ρ​(xj)=2​δ​pθold​(xj)​(1−pθold​(xj))−α​(log⁡ρ​(xj)+1)−μ.\frac{\partial\mathcal{J}}{\partial\rho(x\_{j})}=\sqrt{2\delta p\_{\theta\_{\text{old}}}(x\_{j})\left(1-p\_{\theta\_{\text{old}}}(x\_{j})\right)}-\alpha(\log\rho(x\_{j})+1)-\mu. |  | (42) |

Then set the derivative to zero:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2​δ​pθold​(xj)​(1−pθold​(xj))−α​(log⁡ρ​(xj)+1)−μ=0.\sqrt{2\delta p\_{\theta\_{\text{old}}}(x\_{j})\left(1-p\_{\theta\_{\text{old}}}(x\_{j})\right)}-\alpha(\log\rho(x\_{j})+1)-\mu=0. |  | (43) |

Solving for log⁡ρ​(xj)\log\rho(x\_{j}):

|  |  |  |  |
| --- | --- | --- | --- |
|  | log⁡ρ​(xj)=2​δ​pθold​(xj)​(1−pθold​(xj))α−1−μα.\log\rho(x\_{j})=\frac{\sqrt{2\delta p\_{\theta\_{\text{old}}}(x\_{j})\left(1-p\_{\theta\_{\text{old}}}(x\_{j})\right)}}{\alpha}-1-\frac{\mu}{\alpha}. |  | (44) |

So we have:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ρ​(xj)=exp⁡(2​δ​pθold​(xj)​(1−pθold​(xj))α−1−μα).\rho(x\_{j})=\exp\left(\frac{\sqrt{2\delta p\_{\theta\_{\text{old}}}(x\_{j})\left(1-p\_{\theta\_{\text{old}}}(x\_{j})\right)}}{\alpha}-1-\frac{\mu}{\alpha}\right). |  | (45) |

Let c=exp⁡(−1−λα)c=\exp\left(-1-\frac{\lambda}{\alpha}\right), then Eq. ([45](#A1.E45 "In A.1 Prompt Difficulty Caps Optimization Potential ‣ Appendix A Mathematical Derivations ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs")) becomes

|  |  |  |  |
| --- | --- | --- | --- |
|  | ρ​(xj)=c​exp⁡(2​δ​pθold​(xj)​(1−pθold​(xj))α).\rho(x\_{j})=c\exp\left(\frac{\sqrt{2\delta p\_{\theta\_{\text{old}}}(x\_{j})\left(1-p\_{\theta\_{\text{old}}}(x\_{j})\right)}}{\alpha}\right). |  | (46) |

Using the constraint ∑j=1Nρ​(xj)=1\sum\_{j=1}^{N}\rho(x\_{j})=1, we can obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | c=1∑j=1Nexp⁡(2​δ​pθold​(xj)​(1−pθold​(xj))α).c=\frac{1}{\sum\_{j=1}^{N}\exp\left(\frac{\sqrt{2\delta p\_{\theta\_{\text{old}}}(x\_{j})\left(1-p\_{\theta\_{\text{old}}}(x\_{j})\right)}}{\alpha}\right)}. |  | (47) |

So the optimal distribution is

|  |  |  |  |
| --- | --- | --- | --- |
|  | ρ∗​(x)=exp⁡(2​δ​pθold​(x)​(1−pθold​(x))α)∑x′exp⁡(2​δ​pθold​(x′)​(1−pθold​(x′))α).\rho^{\*}(x)=\frac{\exp\left(\frac{\sqrt{2\delta p\_{\theta\_{\text{old}}}(x)\left(1-p\_{\theta\_{\text{old}}}(x)\right)}}{\alpha}\right)}{\sum\_{x^{\prime}}\exp\left(\frac{\sqrt{2\delta p\_{\theta\_{\text{old}}}(x^{\prime})\left(1-p\_{\theta\_{\text{old}}}(x^{\prime})\right)}}{\alpha}\right)}. |  | (48) |

With the substitution τ=α2​δ\tau=\frac{\alpha}{\sqrt{2\delta}}, the distribution finally becomes:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ρ∗​(x)=exp⁡(pθold​(x)​(1−pθold​(x))/τ)∑x′exp⁡(pθold​(x′)​(1−pθold​(x′))/τ).\rho^{\*}(x)=\frac{\exp\left(\sqrt{p\_{\theta\_{\text{old}}}(x)\left(1-p\_{\theta\_{\text{old}}}(x)\right)}/\tau\right)}{\sum\_{x^{\prime}}\exp\left(\sqrt{p\_{\theta\_{\text{old}}}(x^{\prime})\left(1-p\_{\theta\_{\text{old}}}(x^{\prime})\right)}/\tau\right)}. |  | (49) |

### A.2 Closing the Gap with Theoretical Bound

In the previous proof, we established a lower bound for single-step gradient descent within a δ\delta-local trust region constrained by KL divergence, i.e.,

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ℒ​(θold+d)−ℒ​(θold)\displaystyle\mathcal{L}(\theta\_{\text{old}}+d)-\mathcal{L}(\theta\_{\text{old}}) | =𝔼x∼ρ​[ℒ​(x;θold+d)−ℒ​(x;θold)]\displaystyle=\mathbb{E}\_{x\sim\rho}\left[\mathcal{L}(x;\theta\_{\text{old}}+d)-\mathcal{L}(x;\theta\_{\text{old}})\right] |  | (50) |
|  |  | =𝔼x∼ρ​[−2​δ​∇θℒ​(x;θ)⊤|θ=θold​F−1​(x;θold)​∇θℒ​(x;θ)|θ=θold]\displaystyle=\mathbb{E}\_{x\sim\rho}\left[-\sqrt{2\delta\nabla\_{\theta}\mathcal{L}(x;\theta)^{\top}\big|\_{\theta=\theta\_{\text{old}}}F^{-1}(x;\theta\_{\text{old}})\nabla\_{\theta}\mathcal{L}(x;\theta)\big|\_{\theta=\theta\_{\text{old}}}}\right] |  |
|  |  | ≥−𝔼x∼ρ​[2​δ​𝕍θold​(r​(x,y))]\displaystyle\geq-\mathbb{E}\_{x\sim\rho}\left[\sqrt{2\delta\mathbb{V}\_{\theta\_{\text{old}}}(r(x,y))}\right] |  |
|  |  | =−𝔼x∼ρ​[2​δ​pθold​(x)​(1−pθold​(x))]\displaystyle=-\mathbb{E}\_{x\sim\rho}\left[\sqrt{2\delta p\_{\theta\_{\text{old}}}(x)\left(1-p\_{\theta\_{\text{old}}}(x)\right)}\right] |  |

However, due to the high computational cost of the natural gradient method, it is often avoided in practice, and the theoretical result is instead used to guide prompt sampling. During actual gradient updates, we aim to closely approximate the theoretical efficiency limit within a trust region bounded by a KL divergence constraint of δ\delta. Specifically, after sampling a batch of mm prompts, we seek to optimize operations to approach the bound. Within the curriculum learning framework, we consider optimizing the allocation of rollouts across prompts under a fixed total rollout budget of NN to minimize the following loss function:

|  |  |  |  |
| --- | --- | --- | --- |
|  | min⁡𝔼​[(ℒ​(θ^)−ℒ​(θold)−(−𝔼x∼ρ​[2​δ​pθold​(x)​(1−pθold​(x))]))2],s.t.​∑i=1mni=N.\min\ \mathbb{E}\left[\left({{\mathcal{L}}}(\hat{\theta})-{{\mathcal{L}}}(\theta\_{\text{old}})-\left(-\mathbb{E}\_{x\sim\rho}\left[\sqrt{2\delta p\_{\theta\_{\text{old}}}(x)\left(1-p\_{\theta\_{\text{old}}}(x)\right)}\right]\right)\right)^{2}\right],\quad\text{s.t.}\ \sum\_{i=1}^{m}n\_{i}=N. |  | (51) |

Here, θ^\hat{\theta} denotes the updated model parameters obtained from θold\theta\_{\text{old}} after applying the practical gradient update, i.e.:

|  |  |  |  |
| --- | --- | --- | --- |
|  | θ^=θold−η​∇θℒ^​(θ)|θ=θold,ℒ^​(θ)=−∑i=1m1ni​∑yj∈𝒟i[πθ​(yj|xi)πθold​(yj|xi)​Aθold].\displaystyle\hat{\theta}=\theta\_{\text{old}}-\eta\nabla\_{\theta}\hat{\mathcal{L}}(\theta)\big|\_{\theta=\theta\_{\text{old}}},\quad\hat{\mathcal{L}}(\theta)=-\sum\limits\_{i=1}^{m}\frac{1}{n\_{i}}\sum\limits\_{y\_{j}\in\mathcal{D}\_{i}}\left[\frac{\pi\_{\theta}(y\_{j}|x\_{i})}{\pi\_{\theta\_{\text{old}}}(y\_{j}|x\_{i})}A\_{\theta\_{\text{old}}}\right]. |  | (52) |

Where η\eta is the learning rate and nin\_{i} denotes the number of sampled rollouts for question xix\_{i}. We assume that η\eta is chosen such that the policy update remains within a KL divergence constraint of δ\delta.

In fact, ℒ^​(θ)\hat{\mathcal{L}}(\theta) is an unbiased estimator of ℒ​(θ){\mathcal{L}}(\theta), that is:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 𝔼​[ℒ^​(θ)]\displaystyle\mathbb{E}\left[\hat{\mathcal{L}}(\theta)\right] |  | (53) |
|  | =\displaystyle= | 𝔼​[−1m​∑i=1m1ni​∑yj∈𝒟i[πθ​(yj|xi)πθold​(yj|xi)​Aθold​(xi,yj)]]\displaystyle\mathbb{E}\left[-\frac{1}{m}\sum\limits\_{i=1}^{m}\frac{1}{n\_{i}}\sum\limits\_{y\_{j}\in\mathcal{D}\_{i}}\left[\frac{\pi\_{\theta}(y\_{j}|x\_{i})}{\pi\_{\theta\_{\text{old}}}(y\_{j}|x\_{i})}A\_{\theta\_{\text{old}}}(x\_{i},y\_{j})\right]\right] |  |
|  | =\displaystyle= | −1m​∑i=1m1ni​∑yj∈𝒟i𝔼xi∼ρ,yj∼πθold(⋅|xi)​[πθ​(yj|xi)πθold​(yj|xi)​Aθold​(xi,yj)]\displaystyle-\frac{1}{m}\sum\limits\_{i=1}^{m}\frac{1}{n\_{i}}\sum\limits\_{y\_{j}\in\mathcal{D}\_{i}}\mathbb{E}\_{x\_{i}\sim\rho,y\_{j}\sim\pi\_{\theta\_{\text{old}}}(\cdot|x\_{i})}\left[\frac{\pi\_{\theta}(y\_{j}|x\_{i})}{\pi\_{\theta\_{\text{old}}}(y\_{j}|x\_{i})}A\_{\theta\_{\text{old}}}(x\_{i},y\_{j})\right] |  |
|  | =\displaystyle= | −1m​∑i=1m𝔼xi∼ρ,y∼πθold(⋅|xi)​[πθ​(y|xi)πθold​(y|xi)​Aθold​(xi,y)]\displaystyle-\frac{1}{m}\sum\limits\_{i=1}^{m}\mathbb{E}\_{x\_{i}\sim\rho,y\sim\pi\_{\theta\_{\text{old}}}(\cdot|x\_{i})}\left[\frac{\pi\_{\theta}(y|x\_{i})}{\pi\_{\theta\_{\text{old}}}(y|x\_{i})}A\_{\theta\_{\text{old}}}(x\_{i},y)\right] |  |
|  | =\displaystyle= | 𝔼x∼ρ,y∼πθold(⋅|x)​[πθ​(y|x)πθold​(y|x)​Aθold​(x,y)]\displaystyle\mathbb{E}\_{x\sim\rho,y\sim\pi\_{\theta\_{\text{old}}}(\cdot|x)}\left[\frac{\pi\_{\theta}(y|x)}{\pi\_{\theta\_{\text{old}}}(y|x)}A\_{\theta\_{\text{old}}}(x,y)\right] |  |
|  | =\displaystyle= | 𝔼x∼ρ,y∼πθ(⋅|x)​[Aθold​(x,y)]\displaystyle\mathbb{E}\_{x\sim\rho,y\sim\pi\_{\theta(\cdot|x)}}\left[A\_{\theta\_{\text{old}}}(x,y)\right] |  |
|  | =\displaystyle= | ℒ​(θ)\displaystyle\mathcal{L}(\theta) |  |

By applying the Taylor expansion, we obtain:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ℒ​(θ^)−ℒ​(θold)≈\displaystyle{{\mathcal{L}}}(\hat{\theta})-{{\mathcal{L}}}(\theta\_{\text{old}})\approx | ∇θℒ​(θ)|θ=θold​(θ^−θold)\displaystyle\nabla\_{\theta}\mathcal{L}(\theta)\big|\_{\theta=\theta\_{\text{old}}}(\hat{\theta}-\theta\_{\text{old}}) |  | (54) |
|  | =\displaystyle= | −η​∇θℒ​(θ)⊤|θ=θold​∇θℒ^​(θ)|θ=θold\displaystyle-\eta\nabla\_{\theta}\mathcal{L}(\theta)^{\top}\big|\_{\theta=\theta\_{\text{old}}}\nabla\_{\theta}\hat{\mathcal{L}}(\theta)\big|\_{\theta=\theta\_{\text{old}}} |  |

where

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇θℒ​(θ)=−𝔼x∼ρ,y∼πθ(⋅|x)​[Aθold​(x,y)​∇θlog⁡πθ​(y|x)],\displaystyle\nabla\_{\theta}\mathcal{L}(\theta)=-\mathbb{E}\_{{x}\sim\rho,{y}\sim\pi\_{\theta}(\cdot|{x})}\big[A\_{\theta\_{\mathrm{old}}}(x,y)\nabla\_{\theta}\log\pi\_{\theta}(y|x)\big], |  | (55) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇θℒ^​(θ)=−1m​∑i=1m1ni​∑yj∈𝒟i[∇θπθ​(yj|xi)πθold​(yj|xi)​Aθold​(xi,yj)]\displaystyle\nabla\_{\theta}\hat{\mathcal{L}}(\theta)=-\frac{1}{m}\sum\limits\_{i=1}^{m}\frac{1}{n\_{i}}\sum\limits\_{y\_{j}\in\mathcal{D}\_{i}}\left[\frac{\nabla\_{\theta}\pi\_{\theta}(y\_{j}|x\_{i})}{\pi\_{\theta\_{\text{old}}}(y\_{j}|x\_{i})}A\_{\theta\_{\text{old}}}(x\_{i},y\_{j})\right] |  | (56) |

For convenience, we adopt the following notation:

|  |  |  |  |
| --- | --- | --- | --- |
|  | g=∇θℒ​(θ)|θ=θold,g^=∇θℒ^​(θ)|θ=θold\displaystyle g=\nabla\_{\theta}\mathcal{L}(\theta)\big|\_{\theta=\theta\_{\text{old}}},\ \ \ \hat{g}=\nabla\_{\theta}\hat{\mathcal{L}}(\theta)\big|\_{\theta=\theta\_{\text{old}}} |  | (57) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Δtheo=𝔼x∼ρ​[2​δ⋅pθold​(x)​(1−pθold​(x))]\displaystyle\Delta\_{\mathrm{theo}}=\mathbb{E}\_{x\sim\rho}\left[\sqrt{2\delta\cdot p\_{\theta\_{\mathrm{old}}}(x)\left(1-p\_{\theta\_{\mathrm{old}}}(x)\right)}\right] |  | (58) |

Therefore, the original problem can then be simplified as follows:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 𝔼​[(ℒ​(θ^)−ℒ​(θold)−(−𝔼x∼ρ​[2​δ​pθold​(x)​(1−pθold​(x))]))2]\displaystyle\mathbb{E}\left[\left({{\mathcal{L}}}(\hat{\theta})-{{\mathcal{L}}}(\theta\_{\text{old}})-\left(-\mathbb{E}\_{x\sim\rho}\left[\sqrt{2\delta p\_{\theta\_{\text{old}}}(x)\left(1-p\_{\theta\_{\text{old}}}(x)\right)}\right]\right)\right)^{2}\right] |  | (59) |
|  | =\displaystyle= | 𝔼​[(−η​g⊤​g^+Δtheo)2]\displaystyle\mathbb{E}\left[\left(-\eta g^{\top}\hat{g}+\Delta\_{\mathrm{theo}}\right)^{2}\right] |  |
|  | =\displaystyle= | η2​𝔼​[(g⊤​g^)2]−2​η​Δtheo​𝔼​[g⊤​g^]+Δtheo2\displaystyle\eta^{2}\mathbb{E}[(g^{\top}\hat{g})^{2}]-2\eta\Delta\_{\mathrm{theo}}\mathbb{E}[g^{\top}\hat{g}]+\Delta\_{\mathrm{theo}}^{2} |  |

Because ℒ^​(θ)\hat{\mathcal{L}}(\theta) is an unbiased estimator of ℒ​(θ)\mathcal{L}(\theta), we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[∇θℒ^​(θ)]=∇θ𝔼​[ℒ^​(θ)]=∇θℒ​(θ)\displaystyle\mathbb{E}\left[\nabla\_{\theta}\hat{\mathcal{L}}(\theta)\right]=\nabla\_{\theta}\mathbb{E}\left[\hat{\mathcal{L}}(\theta)\right]=\nabla\_{\theta}\mathcal{L}(\theta) |  | (60) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[g^]=𝔼​[∇θℒ^​(θ)|θ=θold]=∇θ𝔼​[ℒ^​(θ)|θ=θold]=∇θℒ​(θ)|θ=θold=g\displaystyle\mathbb{E}\left[\hat{g}\right]=\mathbb{E}\left[\nabla\_{\theta}\hat{\mathcal{L}}(\theta)\big|\_{\theta=\theta\_{\text{old}}}\right]=\nabla\_{\theta}\mathbb{E}\left[\hat{\mathcal{L}}(\theta)\big|\_{\theta=\theta\_{\text{old}}}\right]=\nabla\_{\theta}\mathcal{L}(\theta)\big|\_{\theta=\theta\_{\text{old}}}=g |  | (61) |

Therefore, we can obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[g⊤​g^]=g⊤​𝔼​[g^]=g⊤​g\displaystyle\mathbb{E}\left[g^{\top}\hat{g}\right]=g^{\top}\mathbb{E}\left[\hat{g}\right]=g^{\top}g |  | (62) |

Now, regarding the first item:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[(g⊤​g^)2]=𝔼​[g⊤​g^​g^⊤​g]=g⊤​𝔼​[g^​g^⊤]​g\displaystyle\mathbb{E}[(g^{\top}\hat{g})^{2}]=\mathbb{E}[g^{\top}\hat{g}\hat{g}^{\top}g]=g^{\top}\mathbb{E}[\hat{g}\hat{g}^{\top}]g |  | (63) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[g^​g^⊤]=𝕍​(g^)+𝔼​[g^]​𝔼​[g^]⊤=𝕍​(g^)+g​g⊤\displaystyle\mathbb{E}[\hat{g}\hat{g}^{\top}]=\mathbb{V}(\hat{g})+\mathbb{E}[\hat{g}]\mathbb{E}[\hat{g}]^{\top}=\mathbb{V}(\hat{g})+{g}{g}^{\top} |  | (64) |

Hence, the original problem is equivalent to the following formulation:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 𝔼​[(ℒ​(θ^)−ℒ​(θold)−(−𝔼x∼ρ​[2​δ​pθold​(x)​(1−pθold​(x))]))2]\displaystyle\mathbb{E}\left[\left({{\mathcal{L}}}(\hat{\theta})-{{\mathcal{L}}}(\theta\_{\text{old}})-\left(-\mathbb{E}\_{x\sim\rho}\left[\sqrt{2\delta p\_{\theta\_{\text{old}}}(x)\left(1-p\_{\theta\_{\text{old}}}(x)\right)}\right]\right)\right)^{2}\right] |  | (65) |
|  | =\displaystyle= | η2​(g⊤​𝕍​(g^)​g+(g⊤​g)2)−2​η​Δtheo​(g⊤​g)+Δtheo2\displaystyle\eta^{2}\left(g^{\top}\mathbb{V}(\hat{g})g+\left(g^{\top}g\right)^{2}\right)-2\eta\Delta\_{\mathrm{theo}}\left(g^{\top}{g}\right)+\Delta\_{\mathrm{theo}}^{2} |  |

Since we aim to minimize the gap from the theoretical update lower bound by reallocating the rollout quantities per question under a total sampling budget of NN, the simplification of the objective function reveals that this problem only affects the first term g⊤​𝕍​(g^)​gg^{\top}\mathbb{V}(\hat{g})g. Thus, the original optimization problem is equivalent to the following:

|  |  |  |  |
| --- | --- | --- | --- |
|  | min⁡g⊤​𝕍​(g^)​g,s.t.​∑i=1mni=N.\displaystyle\min g^{\top}\mathbb{V}(\hat{g})g,\quad\text{s.t.}\ \sum\_{i=1}^{m}n\_{i}=N. |  | (66) |

The theoretical gradient direction gg is typically unknown, and we seek to control the uncertainty of the estimator in all possible directions. Therefore, we instead minimize the total variance Tr​(𝕍​(g^))\text{Tr}(\mathbb{V}(\hat{g})), which corresponds to uniformly reducing the variance in all directions. This approach is a widely adopted technique for variance estimation(Bottou et al., [2018](#bib.bib2); Papini et al., [2018](#bib.bib19); Wang et al., [2013](#bib.bib27)). In other words, we consider the following optimization problem:

|  |  |  |  |
| --- | --- | --- | --- |
|  | min⁡Tr​(𝕍​(g^)),s.t.​∑i=1mni=N\displaystyle\min\ \text{Tr}(\mathbb{V}(\hat{g})),\quad\text{s.t.}\ \sum\_{i=1}^{m}n\_{i}=N |  | (67) |

Since each yjy\_{j} is independently draws from πθold\pi\_{\theta\_{\text{old}}}, we can conclude that:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝕍​(g^)\displaystyle\mathbb{V}(\hat{g}) | =𝕍​(−1m​∑i=1m1ni​∑yj∈𝒟i∇θπθ​(yj|xi)|θ=θoldπθold​(yj|xi)​Aθold​(xi,yj))\displaystyle=\mathbb{V}\left(-\frac{1}{m}\sum\limits\_{i=1}^{m}\frac{1}{n\_{i}}\sum\limits\_{y\_{j}\in\mathcal{D}\_{i}}\frac{\nabla\_{\theta}\pi\_{\theta}(y\_{j}|x\_{i})\big|\_{\theta=\theta\_{\text{old}}}}{\pi\_{\theta\_{\text{old}}}(y\_{j}|x\_{i})}A\_{\theta\_{\text{old}}}(x\_{i},y\_{j})\right) |  | (68) |
|  |  | =1m2​∑i=1m𝕍yj∼πθold​(1ni​∑yj∈𝒟i∇θπθ​(yj|xi)|θ=θoldπθold​(yj|xi)​Aθold​(xi,yj))\displaystyle=\frac{1}{m^{2}}\sum\limits\_{i=1}^{m}\mathbb{V}\_{y\_{j}\sim\pi\_{\theta\_{\text{old}}}}\left(\frac{1}{n\_{i}}\sum\limits\_{y\_{j}\in\mathcal{D}\_{i}}\frac{\nabla\_{\theta}\pi\_{\theta}(y\_{j}|x\_{i})\big|\_{\theta=\theta\_{\text{old}}}}{\pi\_{\theta\_{\text{old}}}(y\_{j}|x\_{i})}A\_{\theta\_{\text{old}}}(x\_{i},y\_{j})\right) |  |

Let h​(x,y;θ)=∇θπθ​(yj|xi)πθold​(yj|xi)​Aθold​(xi,yj)h(x,y;\theta)=\frac{\nabla\_{\theta}\pi\_{\theta}(y\_{j}|x\_{i})}{\pi\_{\theta\_{\text{old}}}(y\_{j}|x\_{i})}A\_{\theta\_{\text{old}}}(x\_{i},y\_{j}), we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝕍​(g^)=\displaystyle\mathbb{V}(\hat{g})= | 1m2​∑i=1m𝕍yj∼πθold​(1ni​∑j=1nih​(yj,xi;θold))\displaystyle\frac{1}{m^{2}}\sum\_{i=1}^{m}\mathbb{V}\_{y\_{j}\sim\pi\_{\theta\_{\text{old}}}}\left(\frac{1}{n\_{i}}\sum\_{j=1}^{n\_{i}}h(y\_{j},x\_{i};\theta\_{\text{old}})\right) |  | (69) |
|  | =\displaystyle= | 1m2​∑i=1m1ni2⋅ni​(𝕍y∼πθold​(h​(y,xi;θold)))\displaystyle\frac{1}{m^{2}}\sum\_{i=1}^{m}\frac{1}{n\_{i}^{2}}\cdot n\_{i}\left(\mathbb{V}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left(h\left(y,x\_{i};\theta\_{\text{old}}\right)\right)\right) |  |
|  | =\displaystyle= | 1m2​∑i=1m𝕍y∼πθold​(h​(y,xi;θold))ni\displaystyle\frac{1}{m^{2}}\sum\_{i=1}^{m}\frac{\mathbb{V}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left(h\left(y,x\_{i};\theta\_{\text{old}}\right)\right)}{n\_{i}} |  |

Therefore, for the total variance Tr​(𝕍​(g^))\text{Tr}(\mathbb{V}(\hat{g})):

|  |  |  |  |
| --- | --- | --- | --- |
|  | Tr​(𝕍​(g^))=1m2​∑i=1mTr​(𝕍y∼πθold​(h​(y,xi;θold)))ni.\displaystyle\text{Tr}(\mathbb{V}(\hat{g}))=\frac{1}{m^{2}}\sum\_{i=1}^{m}\frac{\text{Tr}\left(\mathbb{V}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left(h\left(y,x\_{i};\theta\_{\text{old}}\right)\right)\right)}{n\_{i}}. |  | (70) |

Then we dive into calculating the value of Tr​(𝕍​(h​(y,xi;θold)))\text{Tr}\left(\mathbb{V}(h(y,x\_{i};\theta\_{\text{old}}))\right):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | Tr​(𝕍y∼πθold​(h​(y,xi;θold)))\displaystyle\ \ \ \text{Tr}\left(\mathbb{V}\_{y\sim\pi\_{\theta\_{\text{old}}}}(h(y,x\_{i};\theta\_{\text{old}}))\right) |  | (71) |
|  |  | =𝔼y∼πθold​[Tr​(∇θπθ​(y|xi)​∇θπθ​(y|xi)⊤|θ=θold)(πθold​(y|xi))2​Aθold2]−Tr​(𝔼y∼πθold​[(∇θπθ​(y|xi)|θ=θoldπθold​(y|xi)​Aθold)]​𝔼y∼πθold​[(∇θπθ​(y|xi)|θ=θoldπθold​(y|xi)​Aθold)]⊤)\displaystyle\begin{aligned} =&\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left[\frac{\text{Tr}\left(\nabla\_{\theta}\pi\_{\theta}(y|x\_{i})\nabla\_{\theta}\pi\_{\theta}(y|x\_{i})^{\top}\big|\_{\theta=\theta\_{\text{old}}}\right)}{\left(\pi\_{\theta\_{\text{old}}}(y|x\_{i})\right)^{2}}A\_{\theta\_{\text{old}}}^{2}\right]\\ &-\text{Tr}\left(\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left[\left(\frac{\nabla\_{\theta}\pi\_{\theta}(y|x\_{i})\big|\_{\theta=\theta\_{\text{old}}}}{\pi\_{\theta\_{\text{old}}}(y|x\_{i})}A\_{\theta\_{\text{old}}}\right)\right]\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left[\left(\frac{\nabla\_{\theta}\pi\_{\theta}(y|x\_{i})\big|\_{\theta=\theta\_{\text{old}}}}{\pi\_{\theta\_{\text{old}}}(y|x\_{i})}A\_{\theta\_{\text{old}}}\right)\right]^{\top}\right)\end{aligned} |  |
|  |  | =𝔼y∼πθold​[Tr​(∇θlog⁡πθ​(y|xi)​∇θlog⁡πθ​(y|xi)⊤|θ=θold)​Aθold2]−Tr​(𝔼y∼πθold​[∇θlog⁡πθ​(y|xi)|θ=θold​Aθold]​𝔼y∼πθold​[∇θlog⁡πθ​(y|xi)|θ=θold​Aθold]⊤)\displaystyle\begin{aligned} =&\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left[\text{Tr}\left(\nabla\_{\theta}\log\pi\_{\theta}(y|x\_{i})\nabla\_{\theta}\log\pi\_{\theta}(y|x\_{i})^{\top}|\_{\theta={\theta}\_{\text{old}}}\right)A\_{\theta\_{\text{old}}}^{2}\right]\\ &-\text{Tr}\left(\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left[\nabla\_{\theta}\log\pi\_{\theta}(y|x\_{i})|\_{\theta={\theta}\_{\text{old}}}A\_{\theta\_{\text{old}}}\right]\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left[\nabla\_{\theta}\log\pi\_{\theta}(y|x\_{i})|\_{\theta={\theta}\_{\text{old}}}A\_{\theta\_{\text{old}}}\right]^{\top}\right)\end{aligned} |  |
|  |  | =𝔼y∼πθold​[Tr​(∇θlog⁡πθ​(y|xi)⊤|θ=θold​∇θlog⁡πθ​(y|xi)|θ=θold)​Aθold2]−Tr​(𝔼y∼πθold​[∇θlog⁡πθ​(y|xi)|θ=θold​Aθold]⊤​𝔼y∼πθold​[∇θlog⁡πθ​(y|xi)|θ=θold​Aθold])\displaystyle\begin{aligned} =&\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left[\text{Tr}\left(\nabla\_{\theta}\log\pi\_{\theta}(y|x\_{i})^{\top}|\_{\theta={\theta}\_{\text{old}}}\nabla\_{\theta}\log\pi\_{\theta}(y|x\_{i})|\_{\theta={\theta}\_{\text{old}}}\right)A\_{\theta\_{\text{old}}}^{2}\right]\\ &-\text{Tr}\left(\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left[\nabla\_{\theta}\log\pi\_{\theta}(y|x\_{i})|\_{\theta={\theta}\_{\text{old}}}A\_{\theta\_{\text{old}}}\right]^{\top}\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left[\nabla\_{\theta}\log\pi\_{\theta}(y|x\_{i})|\_{\theta={\theta}\_{\text{old}}}A\_{\theta\_{\text{old}}}\right]\right)\end{aligned} |  |
|  |  | =𝔼y∼πθold[∥∇θlogπθ(y|xi)|θ=θold∥2Aθold2]−∥𝔼y∼πθold[∇θlogπθ(y|xi)Aθold]∥2\displaystyle={\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}}\left[\left\|\nabla\_{\theta}\log\pi\_{\theta}(y|x\_{i})|\_{\theta=\theta\_{\text{old}}}\right\|^{2}A\_{\theta\_{\text{old}}}^{2}\right]-\left\|{{\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}}}\left[\nabla\_{\theta}\log\pi\_{\theta}(y|x\_{i})A\_{\theta\_{\text{old}}}\right]\right\|^{2} |  |

Consider the advantage function defined as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Aθold​(x,y)=r​(x,y)−𝔼y∼πθold​(r​(x,y)).\displaystyle A\_{\theta\_{\text{old}}}(x,y)=r(x,y)-\mathbb{E}\_{y\sim\pi\_{\theta\_{\text{old}}}}(r(x,y)). |  | (72) |

We classify the rollouts into two categories based on whether the final answer is correct or not:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | Tr​(𝕍y∼πθold​(h​(y,xi;θold)))\displaystyle\ \ \ \text{Tr}\left(\mathbb{V}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left(h\left(y,x\_{i};\theta\_{\text{old}}\right)\right)\right) |  | (73) |
|  |  | =P(r=1|xi)𝔼y∼πθold,r=1[∥∇θlogπθ(y|xi)|θ=θold∥2(1−pθold(xi))2]+P(r=0|xi)𝔼y∼πθold,r=0[∥∇θlogπθ(y|xi)|θ=θold∥2(pθold(xi))2]−∥P(r=1|xi)𝔼y∼πθold,r=1[∇θlogπθ(y|xi)(1−pθold(xi)))]+P(r=0|xi)𝔼y∼πθold,r=0[∇θlogπθ(y|xi)(−pθold(xi))]∥2\displaystyle\begin{aligned} =&P(r=1|x\_{i})\underset{y\sim\pi\_{\theta\_{\text{old}}},r=1}{\mathbb{E}}\left[\left\|\nabla\_{\theta}\log\pi\_{\theta}(y|x\_{i})|\_{\theta=\theta\_{\text{old}}}\right\|^{2}\left(1-p\_{\theta\_{\text{old}}}(x\_{i})\right)^{2}\right]\\ &+P(r=0|x\_{i})\underset{y\sim\pi\_{\theta\_{\text{old}}},r=0}{\mathbb{E}}\left[\left\|\nabla\_{\theta}\log\pi\_{\theta}(y|x\_{i})|\_{\theta=\theta\_{\text{old}}}\right\|^{2}\left(p\_{\theta\_{\text{old}}}(x\_{i})\right)^{2}\right]\\ &-\Big\|P(r=1|x\_{i})\underset{y\sim\pi\_{\theta\_{\text{old}}},r=1}{\mathbb{E}}[\nabla\_{\theta}\log\pi\_{\theta}(y|x\_{i})(1-p\_{\theta\_{\text{old}}}(x\_{i})))]\\ &+P(r=0|x\_{i})\underset{y\sim\pi\_{\theta\_{\text{old}}},r=0}{\mathbb{E}}[\nabla\_{\theta}\log\pi\_{\theta}(y|x\_{i})(-p\_{\theta\_{\text{old}}}(x\_{i}))]\Big\|^{2}\end{aligned} |  |
|  |  | =pθold(xi)(1−pθold(xi))2𝔼y∼πθold,r=1[∥∇θlogπθ(y|xi)|θ=θold∥2]+(pθold(xi))2(1−pθold(xi))𝔼y∼πθold,r=0[∥∇θlogπθ(y|xi)|θ=θold∥2]−pθold(xi)2(1−pθold(xi))2∥𝔼y∼πθold,r=1[∇θlogπθ(y|xi))]−𝔼y∼πθold,r=0[∇θlogπθ(y|xi)]∥2\displaystyle\begin{aligned} =&p\_{\theta\_{\text{old}}}(x\_{i})\left(1-p\_{\theta\_{\text{old}}}(x\_{i})\right)^{2}\underset{y\sim\pi\_{\theta\_{\text{old}}},r=1}{\mathbb{E}}\left[\left\|\nabla\_{\theta}\log\pi\_{\theta}(y|x\_{i})|\_{\theta=\theta\_{\text{old}}}\right\|^{2}\right]\\ &+\left(p\_{\theta\_{\text{old}}}(x\_{i})\right)^{2}(1-p\_{\theta\_{\text{old}}}(x\_{i}))\underset{y\sim\pi\_{\theta\_{\text{old}}},r=0}{\mathbb{E}}\left[\left\|\nabla\_{\theta}\log\pi\_{\theta}(y|x\_{i})|\_{\theta=\theta\_{\text{old}}}\right\|^{2}\right]\\ &-p\_{\theta\_{\text{old}}}(x\_{i})^{2}(1-p\_{\theta\_{\text{old}}}(x\_{i}))^{2}\left\|\underset{y\sim\pi\_{\theta\_{\text{old}}},r=1}{\mathbb{E}}[\nabla\_{\theta}\log\pi\_{\theta}(y|x\_{i}))]-\underset{y\sim\pi\_{\theta\_{\text{old}}},r=0}{\mathbb{E}}[\nabla\_{\theta}\log\pi\_{\theta}(y|x\_{i})]\right\|^{2}\end{aligned} |  |
|  |  | =σi2\displaystyle=\sigma\_{i}^{2} |  |

Therefore, we need to solve the following problem:

|  |  |  |  |
| --- | --- | --- | --- |
|  | min⁡1m2​∑i=1mσi2ni,s.t.​∑i=1mni=N.\begin{gathered}\min\ \frac{1}{m^{2}}\sum\_{i=1}^{m}\frac{\sigma\_{i}^{2}}{n\_{i}},\quad\text{s.t.}\ \sum\_{i=1}^{m}n\_{i}=N.\end{gathered} |  | (74) |

We also employ the Lagrange multiplier method to solve this problem:

|  |  |  |  |
| --- | --- | --- | --- |
|  | min⁡𝒥=min⁡1m2​∑i=1mσi2ni+μ​(∑i=1mni−N),\displaystyle\min\ \mathcal{J}=\min\ \frac{1}{m^{2}}\sum\_{i=1}^{m}\frac{\sigma\_{i}^{2}}{n\_{i}}+\mu(\sum\_{i=1}^{m}n\_{i}-N), |  | (75) |

where μ\mu is the Lagrange multiplier.

By differentiating both sides with respect to nin\_{i} and setting the derivative to 0, we obtain:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂𝒥∂ni=−σi2m2​ni2+μ=0⟹ni2=σi2m2​μ.\displaystyle\frac{\partial\mathcal{J}}{\partial n\_{i}}=-\frac{\sigma\_{i}^{2}}{m^{2}n\_{i}^{2}}+\mu=0\implies n\_{i}^{2}=\frac{\sigma\_{i}^{2}}{m^{2}\mu}. |  | (76) |

That is

|  |  |  |  |
| --- | --- | --- | --- |
|  | ni=σim​μ.\displaystyle n\_{i}=\frac{\sigma\_{i}}{m\sqrt{\mu}}. |  | (77) |

According to the constraint:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑i=1mni=∑i=1mσim​μ=N⟹μ=∑i=1mσiN​m.\displaystyle\sum\_{i=1}^{m}n\_{i}=\sum\_{i=1}^{m}\frac{\sigma\_{i}}{m\sqrt{\mu}}=N\implies\sqrt{\mu}=\frac{\sum\_{i=1}^{m}\sigma\_{i}}{Nm}. |  | (78) |

We get

|  |  |  |  |
| --- | --- | --- | --- |
|  | ni=σi∑jσj​N,σi=Tr​(𝕍y∼πθold​(h​(y,xi;θold))).\displaystyle n\_{i}=\frac{\sigma\_{i}}{\sum\_{j}\sigma\_{j}}N,\quad\sigma\_{i}=\sqrt{\text{Tr}\left(\mathbb{V}\_{y\sim\pi\_{\theta\_{\text{old}}}}\left(h\left(y,x\_{i};\theta\_{\text{old}}\right)\right)\right)}. |  | (79) |

Thus, we derive the rollout quantity allocation strategy for different prompts.

## Appendix B Algorithmic Implementation

### B.1 Extended Experimental Results

!(/html/2510.01037/assets/x7.png)

Figure 7: Comparison of Average Gradient Norms. This figure compares the average gradient norms among CurES-GRPO, CurES-RPP, GRPO, and RPP. The CurES variants consistently exhibit higher gradient norms in three out of the four algorithm-and-model-scale combinations, suggesting that the CurES effectively selects more informative prompts, thereby accelerating the training process.

We further analyze the evolution of average gradient norms across different model scales and optimization algorithms (Figure [7](#A2.F7 "Figure 7 ‣ B.1 Extended Experimental Results ‣ Appendix B Algorithmic Implementation ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs")).
Overall, the CurES variants consistently exhibit stronger gradient signals compared to their corresponding baselines.
On Qwen2.5-Math-1.5B, both CurES-RPP and CurES-GRPO maintain substantially higher gradient norms throughout training.
On Qwen2.5-Math-7B, CurES-RPP continues to yield larger gradients, while CurES-GRPO performs comparably to GRPO.
Higher gradient norms indicate that the model receives more informative learning signals, suggesting that CurES effectively prioritizes prompts that accelerate parameter updates.
Notably, for the larger 7B model, the optimizer tends to dampen gradient magnitudes more significantly, which partially reduces the advantage of CurES; nevertheless, the overall trend demonstrates its robustness and consistent benefit across scales.

!(/html/2510.01037/assets/x8.png)

Figure 8: Distribution of rollout quantities with respect to accuracy in CurES base on Qwen2.5-Math-7B at different training iterations.CurES concentrates more rollouts on moderately difficult prompts.

Figure [8](#A2.F8 "Figure 8 ‣ B.1 Extended Experimental Results ‣ Appendix B Algorithmic Implementation ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs") illustrates the distribution of rollout allocations across prompts with different accuracy levels over successive training iterations.
We observe that CurES adaptively concentrates rollouts on moderately difficult prompts, rather than uniformly sampling across the entire spectrum.
This behavior aligns with the intuition that prompts with intermediate difficulty provide the most informative learning signal—being neither trivially solved nor consistently incorrect.
As training progresses, the distribution becomes increasingly peaked around this region, indicating that CurES dynamically refines its sampling strategy to focus computational effort on prompts that are most beneficial for improving policy performance.

!(/html/2510.01037/assets/x9.png)

Figure 9: Performance convergence of Qwen2.5-Math-CurES-7B on MATH500 with different sampling configurations.

We further present the performance convergence of CurES with different sampling configurations on Qwen2.5-Math-7B, as shown in Figure [9](#A2.F9 "Figure 9 ‣ B.1 Extended Experimental Results ‣ Appendix B Algorithmic Implementation ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs").
We observe that CurES achieves stable improvements across all settings, with only minor differences in convergence speed and final accuracy among varying rollout counts and prompt subsets.
Notably, configurations with a moderate number of rollouts (e.g., N′=4,n=16N^{\prime}=4,n=16) strike a favorable balance, reaching higher accuracy with fewer steps compared to more extreme settings such as very large or very small rollout numbers.
This demonstrates that CurES is robust to sampling configurations and can effectively leverage diverse rollout budgets without significant degradation in performance.

!(/html/2510.01037/assets/x10.png)

Figure 10: Efficiency comparison of CurES against baselines on MATH500 with Qwen2.5-Math-7B. Gray dashed lines indicate the steps required for CurES and the baseline to reach the highest average accuracy of the baseline during the entire training period.

We also compare the training efficiency of CurES against the baselines on MATH500 for 7B models.
As shown in Figure [10](#A2.F10 "Figure 10 ‣ B.1 Extended Experimental Results ‣ Appendix B Algorithmic Implementation ‣ CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs"), CurES achieves the peak accuracy of the baseline 4.3× faster with GRPO and 3.7× faster with RPP, demonstrating significant acceleration.
Importantly, CurES not only converges more rapidly but also exceeds the baselines throughout training.
These findings confirm that the adaptive curriculum learning mechanism of CurES significantly improves sample efficiency, enabling faster convergence and better final model quality.

Case Study 1

Problem Janet goes to the mall and spends $3.50 on ice cream, $7.50 each for movie tickets for herself and her younger sister, and $8.50 on a bracelet. If her parents gave her $40, how many dollars does she have left?
Answer: 1313
GVM Output First, let’s calculate the total amount of money Janet spent. She spent $3.50 on ice cream, $7.50 on a movie ticket for herself, $7.50 on a movie ticket for her younger sister, and $8.50 on a bracelet. So the total amount spent is:
[3.50 + 7.50 + 7.50 + 8.50]
Next, we need to find out how much money she has left after spending this amount. Her parents gave her $40, so the remaining amount is:
[40 - (3.50 + 7.50 + 7.50 + 8.50)] Let’s calculate this step by step using Python.
“‘python
# Amount spent on ice cream
ice-cream = 3.50
# Amount spent on movie tickets for herself and her younger sister
movie-tickets = 7.50 + 7.50
# Amount spent on bracelet
bracelet = 8.50
# Total amount spent
total-spent = ice-cream + movie-tickets + bracelet
# Amount given by parents
amount-given = 40.00
# Amount left
amount-left = amount-given - total-spent
print(amount-left)
“‘
“‘output
14.0
“‘
After calculating, we find that Janet has 1414 dollars left.
CurES Output
Let’s break down the problem step by step and then use Python to verify our calculations.
1. First, we need to calculate the total amount Janet spent:

•

$3.50 on ice cream
•

$7.50 each for movie tickets for herself and her younger sister (so $7.50 + $7.50 = $15.00)
•

$8.50 on a bracelet
So, the total amount spent is $3.50 + $15.00 + $8.50 = $27.00.
2. Janet’s parents gave her $40. To find out how much money she has left, we subtract the total amount spent from the $40 her parents gave her: [40−27=1340-27=13]
The calculations confirm that Janet has 1313 dollars left.

Case Study 2

Problem The Sun’s effective temperature, TeT\_{e}, is 5800 Kelvin, and its radius is 7×1010​cm​(7×108​m)7\times 10^{10}\mathrm{~cm}\left(7\times 10^{8}m\right). Compute the luminosity (power output) of the Sun in erg/s. Treat the Sun as a blackbody radiator, and give your answer to one significant figure.
Answer: 4×10334\times 10^{33}
GVM Output The luminosity LL of a blackbody can be calculated using the Stefan-Boltzmann law:

L=4​π​R2​σ​Te4L=4\pi R^{2}\sigma T\_{e}^{4}
where:

•

RR is the radius,
•

TeT\_{e} is the effective temperature,
•

σ\sigma is the Stefan-Boltzmann constant, approximately 5.67×10−8​W⋅m−2⋅K−45.67\times 10^{-8}\text{W}\cdot\text{m}^{-2}\cdot\text{K}^{-4}.
First, convert the radius from cm to meters:

R=7×1010​cm=7×108​mR=7\times 10^{10}\text{cm}=7\times 10^{8}\text{m}
The effective temperature is given as:

Te=5800​KT\_{e}=5800\text{K}
Now plug the values into the formula:

L=4​π​(7×108)2​(5.67×10−8)​(5800)4L=4\pi(7\times 10^{8})^{2}(5.67\times 10^{-8})(5800)^{4}
Calculate R2R^{2}:

R2=(7×108)2=49×1016=4.9×1017​m2R^{2}=(7\times 10^{8})^{2}=49\times 10^{16}=4.9\times 10^{17}\text{m}^{2}
Calculate Te4T\_{e}^{4}:

Te4=(5800)4=1.1316496×1014​K4T\_{e}^{4}=(5800)^{4}=1.1316496\times 10^{14}\text{K}^{4}
Substitute the intermediate results back into the luminosity equation:

L=4​π​(4.9×1017)​(5.67×10−8)​(1.1316496×1014)L=4\pi(4.9\times 10^{17})(5.67\times 10^{-8})(1.1316496\times 10^{14})
Combine the constants and the powers of 10:

L=4​π​(4.9×5.67×1.1316496×1017−8+14)=4​π​(4.9×5.67×1.1316496×1023)L=4\pi(4.9\times 5.67\times 1.1316496\times 10^{17-8+14})=4\pi(4.9\times 5.67\times 1.1316496\times 10^{23})
Calculate the product of the constants:

4.9×5.67×1.1316496≈31.034.9\times 5.67\times 1.1316496\approx 31.03
So,

L≈4​π​(31.03×1023)=4​π​(3.103×1024)L\approx 4\pi(31.03\times 10^{23})=4\pi(3.103\times 10^{24})
Now compute the numerical value:

L≈4×3.14159×3.103×1024≈39.4784×3.103×1024≈122.5×1024L\approx 4\times 3.14159\times 3.103\times 10^{24}\approx 39.4784\times 3.103\times 10^{24}\approx 122.5\times 10^{24}

L≈1.225×1026​WL\approx 1.225\times 10^{26}\text{W}
Rounded to one significant figure:

L≈1×1026​WL\approx 1\times 10^{26}\text{W}
Since 1​W=1​J/s=107​erg/s1\text{W}=1\text{J/s}=10^{7}\text{erg/s}:

L≈1×1026×107​erg/s=1×1033​erg/sL\approx 1\times 10^{26}\times 10^{7}\text{erg/s}=1\times 10^{33}\text{erg/s}
The luminosity in erg/s is:

1×1033{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\framebox{$1\times 10^{33}$}}
CurES Output The luminosity LL of a blackbody is given by the Stefan-Boltzmann law:

L=4​π​R2​σ​Te4L=4\pi R^{2}\sigma T\_{e}^{4}
where:

•

RR is the radius of the Sun,
•

TeT\_{e} is the effective temperature of the Sun,
•

σ\sigma is the Stefan-Boltzmann constant, σ≈5.67×10−8​W⋅m−2⋅K−4\sigma\approx 5.67\times 10^{-8}\text{W}\cdot\text{m}^{-2}\cdot\text{K}^{-4}.
Convert the solar radius from centimeters to meters:

R=7×1010​cm=7×108​mR=7\times 10^{10}\text{cm}=7\times 10^{8}\text{m}
The effective temperature is:

Te=5800​KT\_{e}=5800\text{K}

L=4​π​(7×108)2×(5.67×10−8)×(5800)4L=4\pi(7\times 10^{8})^{2}\times(5.67\times 10^{-8})\times(5800)^{4}
Calculate R2R^{2}:

R2=(7×108)2=49×1016=4.9×1017​m2R^{2}=(7\times 10^{8})^{2}=49\times 10^{16}=4.9\times 10^{17}\text{m}^{2}
Calculate Te4T\_{e}^{4}:

Te4=(5800)4=1.1316496×1014​K4T\_{e}^{4}=(5800)^{4}=1.1316496\times 10^{14}\text{K}^{4}
Substitute the intermediate results:

L=4​π×(4.9×1017)×(5.67×10−8)×(1.1316496×1014)L=4\pi\times(4.9\times 10^{17})\times(5.67\times 10^{-8})\times(1.1316496\times 10^{14})
Combine the constants and the powers of 10:

L=4​π×(4.9×5.67×1.1316496)×1017−8+14=4​π×(31.37336)×1023​WL=4\pi\times(4.9\times 5.67\times 1.1316496)\times 10^{17-8+14}=4\pi\times(31.37336)\times 10^{23}\text{W}

L≈4×3.14159×31.37336×1023≈394.444×1023​W≈3.94444×1025​WL\approx 4\times 3.14159\times 31.37336\times 10^{23}\approx 394.444\times 10^{23}\text{W}\approx 3.94444\times 10^{25}\text{W}
Rounded to one significant figure:

L≈4×1026​WL\approx 4\times 10^{26}\text{W}
Since 1​W=1​J/s=107​erg/s1\text{W}=1\text{J/s}=10^{7}\text{erg/s}:

L=4×1026×107​erg/s=4×1033​erg/sL=4\times 10^{26}\times 10^{7}\text{erg/s}=4\times 10^{33}\text{erg/s}
The luminosity of the Sun is:

4×1033​erg/s{\color[rgb]{0,1,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,1,0}\framebox{$4\times 10^{33}$}}\text{erg/s}
