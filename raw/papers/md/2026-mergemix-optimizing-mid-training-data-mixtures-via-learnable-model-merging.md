---
arxiv: '2601.17858'
authors:
- Jiapeng Wang
- Changxin Tian
- Kunlong Chen
- Ziqi Liu
- Jiaxin Mao
- Wayne Xin Zhao
- Zhiqiang Zhang
- Jun Zhou
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging'
url: https://arxiv.org/abs/2601.17858
year: 2026
---

# MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging

Jiapeng Wang
  
Changxin Tian
  
Kunlong Chen
  
Ziqi Liu
  
Jiaxin Mao
  
Wayne Xin Zhao
  
Zhiqiang Zhang
  
Jun Zhou

###### Abstract

Optimizing data mixtures is essential for unlocking the full potential of large language models (LLMs), yet identifying the optimal composition remains computationally prohibitive due to reliance on heuristic trials or expensive proxy training. To address this, we introduce MergeMix, a novel approach that efficiently determines optimal data mixing ratios by repurposing model merging weights as a high-fidelity, low-cost performance proxy. By training domain-specific experts on minimal tokens and optimizing their merging weights against downstream benchmarks, MergeMix effectively optimizes the performance of data mixtures without incurring the cost of full-scale training. Extensive experiments on models with 8B and 16B parameters validate that MergeMix achieves performance comparable to or surpassing exhaustive manual tuning while drastically reducing search costs. Furthermore, MergeMix exhibits high rank consistency (Spearman ρ>0.9\rho>0.9) and strong cross-scale transferability, offering a scalable, automated solution for data mixture optimization.

Machine Learning, ICML

## 1 Introduction

Data is the fundamental fuel for large language models (LLMs), with its strategic curation playing a critical role throughout the training lifecycle (Zhao et al., [2023](#bib.bib76); Albalak et al., [2024](#bib.bib3); Luo et al., [2025](#bib.bib38)). During pre-training, broad and unlabeled corpora establish foundational linguistic and world knowledge. In mid-training, carefully curated datasets are introduced to enhance specific capabilities, such as reasoning or coding. Finally, post-training stages employ instruction and preference data to align model outputs with human intentions and safety standards. At each stage, the composition of the data mixture directly shapes the model’s capabilities, safety, and overall performance (Yang et al., [2025](#bib.bib68); Hu et al., [2024](#bib.bib20); Basant et al., [2025](#bib.bib4); Team et al., [2025a](#bib.bib57); Olmo et al., [2025](#bib.bib44)).

!(/html/2601.17858/assets/x1.png)
!(/html/2601.17858/assets/x2.png)

Figure 1: Cost-performance efficiency analysis. (Top) Comparison of estimated computational costs (log scale) and downstream benchmark accuracy across different data mixing strategies. Details on cost estimation are provided in Appendix [C](#A3 "Appendix C Baselines and Computational Cost Analysis ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging"). (Bottom) Conceptual illustration of the search dynamics. Conventional methods require increasing computational investment to asymptotically approach the optimal performance zone through iterative trials or fitting. In contrast, MergeMix leverages weight-space merging as an proxy, identifying the optimal mixtures with minimal cost.

Current industrial practices on data mix heavily rely on heuristic trials guided by human intuition, requiring computationally prohibitive full-scale training runs to identify the best-performing candidate. Although recent studies aim to automate this process, they suffer from some major limitations (Xie et al., [2023](#bib.bib66); Shukor et al., [2025](#bib.bib51); Liu et al., [2025](#bib.bib34); Diao et al., [2025](#bib.bib12)). First, they still incur substantial computational costs, often necessitating dozens to hundreds of proxy training runs to fit scaling laws, train mix regressors, or perform iterative tuning (Shukor et al., [2025](#bib.bib51); Ye et al., [2025](#bib.bib70); Liu et al., [2025](#bib.bib34); Diao et al., [2025](#bib.bib12)). Second, these methods are primarily optimized for language modeling loss (e.g., perplexity) (Xie et al., [2023](#bib.bib66); Belenki et al., [2025](#bib.bib6)). However, reductions in perplexity do not reliably translate to improved performance on complex downstream tasks (Lourie et al., [2025](#bib.bib36)). This objective misalignment can hinder practical deployment, especially for the suboptimal mixtures used during mid- and post-training.
Furthermore, these automatic methods have primarily been evaluated at limited scale and under controlled experimental conditions. This constrains our understanding of their generalization and practical utility, highlighting the need for rigorous, large-scale validation.

To address these limitations, we introduce MergeMix, a novel approach that reframes the data mixture optimization problem as a task of *model merging*. We focus particularly on optimizing data mixtures during *mid‑training*—the stage where specific model capabilities are refined using carefully curated datasets. Our core insight is that, at this stage, linearly interpolating the weights of domain‑specific expert models serves as a high‑fidelity proxy for the outcomes of actual data mixing. Rather than running expensive training trials for every candidate mixture ratio, MergeMix requires training only a small set of experts on minimal data. By optimizing the merging weights of these experts against downstream benchmarks, we effectively convert the high cost of model training into the negligible cost of model merging and inference.
We validate MergeMix using industrial‑scale datasets, conducting extensive mid‑training experiments on models with 8B and 16B parameters. Our experiments show that MergeMix identifies data configurations that match or exceed the performance of exhaustive manual tuning, while reducing the search cost by over 100×\times.

To summarize, our contributions are as follows:

* •

  We propose MergeMix, a novel approach that leverages model merging as a computationally efficient and high-fidelity proxy for evaluating data mixtures. This allows us to optimize data configurations directly against downstream tasks at a drastically reduced cost.
* •

  We provide a theoretical analysis on why weight interpolation between domain-expert models effectively approximates the outcome of direct data-mixture tuning. This is grounded by the identical first-order optimization dynamics between weight interpolation and mixed-data training under shared initialization.
* •

  We conduct extensive experiments to validate MergeMix in industrial-scale mid-training scenarios. Our results show that MergeMix achieves performance comparable to or better than extensive manual tuning, while reducing the computational cost of the search process by 100×\times.

## 2 Related Work

### 2.1 Model Merging

Model merging (Izmailov et al., [2018](#bib.bib23); Wortsman et al., [2022](#bib.bib65)) has recently gained attention as a parameter-space alternative to ensembling or retraining for improving performance and reusing existing models efficiently.
It has been shown that averaging weights of multiple models from the same base often yields a single model with higher accuracy and robustness than the best individual model without increasing inference cost (Wortsman et al., [2022](#bib.bib65)), and recent advancements have moved beyond simple averaging to sophisticated weighted combinations aimed at maximizing cross-domain performance (Maiti et al., [2025](#bib.bib41); Khalifa et al., [2024](#bib.bib27)).
Building on this foundation, a growing body of work has explored diverse merging algorithms to further refine how model parameters are combined (Ilharco et al., [2023](#bib.bib22); Yu et al., [2024](#bib.bib72); Yadav et al., [2023](#bib.bib67); Yang et al., [2024](#bib.bib69)).
Ahmadian et al. ([2024](#bib.bib1)) explores the effectiveness of objective-driven model merging and mixed-data training.
More closely related to our work, several recent studies (Na et al., [2024](#bib.bib43); Tao et al., [2025](#bib.bib56)) explore model merging as a means to ablate data inclusion, but do not investigate the impact of data mixing ratios.

### 2.2 Data Mixture for LLM Pre-training

The strategic allocation of training data plays a pivotal role in shaping the performance characteristics of LLMs. Recent progress on pre-training data mixture optimization has shifted from heuristic sampling toward predictive, automated, and scalable strategies. A notable strand uses scaling laws to model and predict performance across mixture proportions and scales (Ye et al., [2025](#bib.bib70); Ge et al., [2024](#bib.bib14); Gu et al., [2024b](#bib.bib16); Shukor et al., [2025](#bib.bib51)). Another theme focuses on proxy models and automated search for mixture optimization, including DoReMi (Xie et al., [2023](#bib.bib66)) that uses small proxy models to reweight domain proportions, RegMix (Liu et al., [2025](#bib.bib34)) that treats mixture search as a regression prediction task over many small runs, ADMIRE-BayesOpt (Chen et al., [2025](#bib.bib10)) which formulates mixture selection as a Bayesian optimization problem for accelerated search across model scales, CLIMB (Diao et al., [2025](#bib.bib12)), a clustering-based iterative bootstrapping framework that embeds data, evaluates candidate mixtures with proxies, and refines weights, and MixMin (Thudi et al., [2025](#bib.bib59)), which formulates the mixture search as a convex minimization problem to efficiently match target data distributions.
Most closely related to our work is Belenki et al. ([2025](#bib.bib6)), which ensembles multiple domain-specialized experts by combining their logits to approximate the loss on mixed-domain data. However, this approach primarily optimizes for validation loss and incurs an inference cost that scales linearly with the number of experts. Moreover, its effectiveness has not been validated in real-world training scenarios or across diverse downstream benchmarks.

## 3 The MergeMix Framework

The proposed MergeMix framework operates on the premise that in the mid-training phase, the parameter space geometry is sufficiently regular (Frankle et al., [2020](#bib.bib13)) that *weight interpolation* can serve as a computationally efficient proxy for *data interpolation*.
The MergeMix pipeline consists of three stages: First, we train expert models to capture the optimization trajectory of each specific domain. Next, we employ a regression model to approximate the relationship between weight mixing ratios and downstream capabilities, allowing us to efficiently explore the performance landscape without expensive training trials. Finally, we identify the data mixture that maximizes the target utility and apply this derived mixture to the large-scale training run.

### 3.1 Training Domain-Specific Experts

Given a pretrained base model Θbase\Theta\_{\text{base}} and KK data domains {𝒟1,…,𝒟K}\{\mathcal{D}\_{1},\dots,\mathcal{D}\_{K}\}, we train KK independent expert models. Our training procedure incorporates the following key configurations:
(1) Shared initialization: All experts are initialized from the same base model Θbase\Theta\_{\text{base}};
(2) Constant learning rate: A fixed learning rate η\eta is applied throughout training without decay. This prevents the vanishing update step size typical of LR decay, ensuring that the expert models continuously traverse the loss landscape along their respective domain-specific gradient directions;
(3) Restricted training horizon: Training is limited to a short horizon (e.g., fewer than 5B tokens). This keeps each expert within a local neighborhood around the initialization, preserving the geometric alignment required for effective weight merging and preventing divergence into separate loss basins (Frankle et al., [2020](#bib.bib13)).

### 3.2 Efficient Mixture Search via Model Merging

We reframe the costly mixture-tuning problem into a model merging process. This enables the highly efficient instantiation of candidate mixture models.
Instead of relying on inaccurate loss-based predictors, we directly evaluate and optimize the merged models on downstream benchmarks.

#### Parameter-Space Merging.

Given a mixing configuration vector 𝜶∈ΔK−1\boldsymbol{\alpha}\in\Delta^{K-1} (where ∑αk=1\sum\alpha\_{k}=1), we instantiate a candidate model Θmerge​(𝜶)\Theta\_{\text{merge}}(\boldsymbol{\alpha}) via linear interpolation (Izmailov et al., [2018](#bib.bib23); Wortsman et al., [2022](#bib.bib65)):

|  |  |  |
| --- | --- | --- |
|  | Θmerge​(𝜶)=Θbase+∑k=1Kαk​(Θk−Θbase).\Theta\_{\text{merge}}(\boldsymbol{\alpha})=\Theta\_{\text{base}}+\sum\_{k=1}^{K}\alpha\_{k}(\Theta\_{k}-\Theta\_{\text{base}}). |  |

Since this operation involves only element-wise addition, it incurs negligible computational cost compared to training.

#### Performance Surface Mapping.

Directly evaluating every possible combination on the simplex is prohibitively expensive given the cost of full-benchmark evaluation. Instead, we adopt a learnable prediction strategy:
We sample NN seed configurations {𝜶(i)}i=1N\{\boldsymbol{\alpha}^{(i)}\}\_{i=1}^{N} using a coarse grid search combined with heuristic priors (N=40N=40 in our experiments). For each configuration, we construct the merged model, evaluate it across MM capability domains (e.g., mathematics, coding, reasoning, knowledge), and record the resulting capability scores 𝐲(i)∈ℝM\mathbf{y}^{(i)}\in\mathbb{R}^{M}. We then train a set of LightGBM regressors (Ke et al., [2017](#bib.bib26)) f^m:𝜶→ym\hat{f}\_{m}:\boldsymbol{\alpha}\to y\_{m} (one per capability) to approximate the performance landscape:

|  |  |  |
| --- | --- | --- |
|  | 𝐲^​(𝜶)=[f^1​(𝜶),f^2​(𝜶),…,f^M​(𝜶)].\hat{\mathbf{y}}(\boldsymbol{\alpha})=[\hat{f}\_{1}(\boldsymbol{\alpha}),\hat{f}\_{2}(\boldsymbol{\alpha}),\dots,\hat{f}\_{M}(\boldsymbol{\alpha})]. |  |

Using this learned prediction model f^\hat{f}, we can perform a fine-grained search over the simplex to identify the candidate 𝜶∗\boldsymbol{\alpha}^{\*} that maximizes our target utility function. The predicted optimum is finally verified via actual model merging and evaluation to ensure reliability.

#### Utility-Driven Mixture Selection.

With the performance surface f^\hat{f} mapped, selecting the optimal mixture becomes a flexible optimization task governed by a user-defined utility function U​(⋅)U(\cdot). This framework allows practitioners to define objectives tailored to specific needs, ranging from a balanced generalist model to specialized targeting of distinct capabilities (e.g., coding or reasoning). Consequently, MergeMix enables the efficient exploration of capability trade-offs without the need for training trials. We formalize this selection process as:

|  |  |  |
| --- | --- | --- |
|  | 𝜶∗=arg⁡max𝜶∈ΔK−1⁡U​(f^1​(𝜶),…,f^M​(𝜶)).\boldsymbol{\alpha}^{\*}=\arg\max\_{\boldsymbol{\alpha}\in\Delta^{K-1}}U(\hat{f}\_{1}(\boldsymbol{\alpha}),\dots,\hat{f}\_{M}(\boldsymbol{\alpha})). |  |

The derived weight ratios 𝜶∗\boldsymbol{\alpha}^{\*} are then directly adopted as the data mixing ratio 𝝀∗\boldsymbol{\lambda}^{\*}. In our primary experiments, we instantiate UU as a generalist objective, calculated as the macro-average of normalized benchmark scores, to ensure robust and balanced improvements across all domains. We also present results focused on optimizing specific capabilities in Figure [5](#S4.F5 "Figure 5 ‣ Comparison Setup. ‣ 4.3 Large-Scale Mid-Training Validation ‣ 4 Experiment ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging").

### 3.3 Theoretical Analysis: Why Weight Mixing Proxies Data Mixing

In this section, we provide a theoretical analysis to justify substituting computationally expensive mixed-data training with efficient weight interpolation. Our core argument is that in the local mid-training regime, these two processes share identical first-order dynamics, with discrepancies confined to second-order under the specific training protocols defined in Section [3.1](#S3.SS1 "3.1 Training Domain-Specific Experts ‣ 3 The MergeMix Framework ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging").

#### Trajectory Decomposition via Taylor Expansion.

Let Θ0\Theta\_{0} be the pretrained initialization. We analyze an idealized SGD-like dynamics under the assumption that the change in loss can be approximated by decomposing it into first-order gradient contributions and second-order curvature interactions (Wang et al., [2025b](#bib.bib62); Pruthi et al., [2020](#bib.bib46)).

First, consider the data mixing trajectory with ratios 𝝀\boldsymbol{\lambda}. The parameter update accumulates gradients from the mixed loss ℒmix=∑λk​ℒk\mathcal{L}\_{\text{mix}}=\sum\lambda\_{k}\mathcal{L}\_{k}. The final parameters Θmix\Theta^{\text{mix}} can be approximated as (see derivation in Appendix [B](#A2 "Appendix B Analysis of the Weight Mixing Proxy ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | Θmix≈Θ0−η​T​∑kλk​gk+(η​T)22​∑k,jλk​λj​Hk​gj.\Theta^{\text{mix}}\approx\Theta\_{0}-\eta T\sum\_{k}\lambda\_{k}g\_{k}+\frac{(\eta T)^{2}}{2}\sum\_{k,j}\lambda\_{k}\lambda\_{j}H\_{k}g\_{j}. |  | (1) |

where gk=∇ℒk​(Θ0)g\_{k}=\nabla\mathcal{L}\_{k}(\Theta\_{0}) and Hk=∇2ℒk​(Θ0)H\_{k}=\nabla^{2}\mathcal{L}\_{k}(\Theta\_{0}). The second term represents the interaction between domains, where the gradient update of domain jj is distorted by the curvature of domain kk (Wang et al., [2025b](#bib.bib62); Yu et al., [2020](#bib.bib73)).

Next, consider the model merging trajectory. We train KK experts independently, where the kk-th expert’s update is Δ​Θk≈−η​T​gk−(η​T)22​Hk​gk\Delta\Theta\_{k}\approx-\eta Tg\_{k}-\frac{(\eta T)^{2}}{2}H\_{k}g\_{k}. By merging these experts with weights 𝜶\boldsymbol{\alpha} set equal to the data ratios 𝝀\boldsymbol{\lambda}, we obtain:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Θmerge≈Θ0−η​T​∑kλk​gk+(η​T)22​∑kλk​Hk​gk.\Theta^{\text{merge}}\approx\Theta\_{0}-\eta T\sum\_{k}\lambda\_{k}g\_{k}+\frac{(\eta T)^{2}}{2}\sum\_{k}\lambda\_{k}H\_{k}g\_{k}. |  | (2) |

Comparing Eq. ([1](#S3.E1 "Equation 1 ‣ Trajectory Decomposition via Taylor Expansion. ‣ 3.3 Theoretical Analysis: Why Weight Mixing Proxies Data Mixing ‣ 3 The MergeMix Framework ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging")) and Eq. ([2](#S3.E2 "Equation 2 ‣ Trajectory Decomposition via Taylor Expansion. ‣ 3.3 Theoretical Analysis: Why Weight Mixing Proxies Data Mixing ‣ 3 The MergeMix Framework ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging")) reveals that the first-order gradient terms match perfectly. The discrepancy is strictly confined to the second-order terms (see detailed discussion of the error term Δ\Delta in Appendix [B](#A2 "Appendix B Analysis of the Weight Mixing Proxy ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging")).

#### Validity of Proxy.

The approximation is motivated by the dominance of first-order training dynamics in the local optimization regime. Crucially, recent theoretical analysis indicates that such higher-order contributions offer limited additional signal for data contribution estimation compared to first-order gradient alignment (Wang et al., [2025b](#bib.bib62); Pruthi et al., [2020](#bib.bib46); Yeh et al., [2022](#bib.bib71)).
In this work, we therefore focus on first-order effects and treat higher-order terms as secondary, using optimization over the merged weights 𝜶\boldsymbol{\alpha} as a principled and efficient proxy for navigating the data mixing search space.

### 3.4 Computational Cost Analysis

We further quantify the efficiency of MergeMix by analyzing its computational complexity. Let CtrainC\_{\text{train}} be the training cost per token and DtrialD\_{\text{trial}} be the budget allocated for each exploratory training run.
Standard manual tuning requires training NN candidate mixtures, incurring a total cost of approximately 𝒞Manual≈N⋅Dtrial⋅Ctrain\mathcal{C}\_{\text{Manual}}\approx N\cdot D\_{\text{trial}}\cdot C\_{\text{train}}.
Even proxy-based methods (e.g., scaling laws) rely on numerous sub-scale runs, where the accumulated token count remains substantial.
In contrast, MergeMix requires only training KK domain experts on a minimal subset Dexpert≪DtrialD\_{\text{expert}}\ll D\_{\text{trial}}.
Since model merging is computationally negligible, the dominant cost becomes the inference-based evaluation of merged candidates, denoted as CevalC\_{\text{eval}}.
The total cost is thus 𝒞MergeMix≈K⋅Dexpert⋅Ctrain+M⋅Ceval\mathcal{C}\_{\text{MergeMix}}\approx K\cdot D\_{\text{expert}}\cdot C\_{\text{train}}+M\cdot C\_{\text{eval}}, where MM denotes the number of merged models used for grid search or proxy model training.
Given that CevalC\_{\text{eval}} is orders of magnitude lower than training costs, and in our setting, where K=4K=4, N=10N=10, and Dexpert≈0.025​DtrialD\_{\text{expert}}\approx 0.025\,D\_{\text{trial}} (5B vs. 200B tokens), MergeMix achieves a cost reduction of 100×100\times compared to exhaustive manual baselines, which consume 2T tokens versus MergeMix’s 20B tokens.

### 3.5 Hierarchical Data Mix

In industrial mid-training, practitioners typically work with hundreds of distinct datasets, not just a few high-level categories, making direct optimization over a high-dimensional simplex intractable. To navigate this, a hierarchical model merging strategy can be adopted. First, semantically similar datasets can be grouped into clusters (e.g., within the code domain, sub-clusters such as code repositories, code-related text, and programming competitions). Then, the MergeMix framework can be recursively applied.

This hierarchical search can be executed in two ways: (1) bottom-up, where optimal mixing ratios are first determined within each sub-cluster group to form consolidated domain experts, which are subsequently merged at the global level; or (2) top-down, where the global mixing ratios across high-level domains are optimized first, followed by independent search of optimal local ratios within each domain. This divide-and-conquer strategy enables precise, fine-grained control over the data distribution without the combinatorial explosion of the search space. We provide empirical experiment of hierarchical MergeMix in Section [4.6](#S4.SS6 "4.6 Hierarchical MergeMix ‣ 4 Experiment ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging").

## 4 Experiment

### 4.1 Experimental Setup

#### Models and Training Settings.

We conduct experiments using two models based on a standard Mixture-of-Experts (MoE) architecture.
They have 8B and 16B total parameters, respectively, with each activating 1.4B parameters. We refer to them as small and large in the following sections, respectively. The detailed model architecture is shown in Appendix [A](#A1 "Appendix A Model Architecture ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging").
We focus on mid-training with high-quality data starting from pretrained checkpoint.
We employ the AdamW optimizer (Loshchilov & Hutter, [2019](#bib.bib35)), with hyperparameters set to β1=0.9\beta\_{1}=0.9, β2=0.95\beta\_{2}=0.95, and a weight decay of 0.1. Based on preliminary scaling law experiments, we set the peak learning rate and batch size to 3.74×10−43.74\times 10^{-4} and 2048 for the large model, and 4.78×10−44.78\times 10^{-4} and 2048 for the small model, respectively.

#### Benchmarks.

To provide a holistic assessment of model capabilities, we consider a diverse suite of downstream tasks for evaluation. Tasks are grouped into several categories, including: (a) general knowledge/reasoning (ARC (Bhakthavatsalam et al., [2021](#bib.bib7)), AGIEval (Zhong et al., [2024](#bib.bib77)), OpenBookQA (Mihaylov et al., [2018](#bib.bib42)), BBH (Suzgun et al., [2023](#bib.bib53)), WorldSense (Hong et al., [2025](#bib.bib19)), PIQA (Bisk et al., [2020](#bib.bib8)), hellaswag (Zellers et al., [2019](#bib.bib74)) and KOR-Bench (Ma et al., [2025](#bib.bib39))); (b) language understanding (race (Lai et al., [2017](#bib.bib29)), SQuAD 2.0 (Rajpurkar et al., [2018](#bib.bib47)), TriviaQA (Joshi et al., [2017](#bib.bib25)), NQ (Kwiatkowski et al., [2019](#bib.bib28)) and winogrande (Sakaguchi et al., [2021](#bib.bib49))); (c) professional knowledge (e.g., MMLU (Hendrycks et al., [2021a](#bib.bib17)), CMMLU (Li et al., [2024a](#bib.bib30)), C-Eval (Huang et al., [2023](#bib.bib21)), MMLU-Pro (Wang et al., [2024](#bib.bib63)), GPQA (Rein et al., [2023](#bib.bib48)) and SuperGPQA (Team et al., [2025b](#bib.bib58))); (d) math (GSM8K (Cobbe et al., [2021](#bib.bib11)), MATH (Hendrycks et al., [2021b](#bib.bib18)), gaokao (Zhang et al., [2023](#bib.bib75)), GSM-Plus (Li et al., [2024b](#bib.bib31)), mgsm-zh (Shi et al., [2023](#bib.bib50)), CMATH (Wei et al., [2023](#bib.bib64)), MathBench (Liu et al., [2024](#bib.bib32)), minerva\_math (Hendrycks et al., [2021b](#bib.bib18)) and college\_math (Tang et al., [2024](#bib.bib54)); (e) code (HumanEval (Chen et al., [2021](#bib.bib9)), LiveCodeBench (Jain et al., [2025](#bib.bib24)), MBPP (Tao et al., [2024](#bib.bib55)), HumanEval\_plus (Liu et al., [2023](#bib.bib33)), MBPP\_plus (Liu et al., [2023](#bib.bib33)), HumanEval\_cn (Peng et al., [2024](#bib.bib45)), HumanEval\_fim (Bavarian et al., [2022](#bib.bib5)) and CruxEval (Gu et al., [2024a](#bib.bib15))).

### 4.2 Dual-Capability Merging Study

The core premise of MergeMix is that the performance landscape in model weight space mirrors the performance landscape in data mixture ratio space.
To validate the MergeMix framework, we first conduct a controlled dual-capacity experiment to rigorously verify the correlation between model merging weights and data mixing ratios.

!(/html/2601.17858/assets/x3.png)

Figure 2: Rank consistency between model merging and data mixture training. The high correlation indicates that weight interpolation accurately predicts the relative ranking of data mixtures. We also present the value of λ\lambda for each configure in percent.

!(/html/2601.17858/assets/x4.png)

Figure 3: Performance trend comparison between model merging and actual mixture tuning on math and code benchmarks.

Specially, we first examine this hypothesis in two domains known to exhibit strong correlation: math and code, where prior works suggest that training on one data type can impact performance on the other (Ma et al., [2024](#bib.bib40); Lu et al., [2025](#bib.bib37)).
First, we train two domain-specialized expert models, Θmath\Theta\_{\text{math}} and Θcode\Theta\_{\text{code}}, on their respective datasets for 25B tokens each. We then construct two sequences of models:
(1) Weight interpolated models:
Θα=α⋅Θmath+(1−α)⋅Θcode,α∈{0.1,0.2,…,0.9}\Theta\_{\alpha}=\alpha\cdot\Theta\_{\text{math}}+(1-\alpha)\cdot\Theta\_{\text{code}},\quad\alpha\in\{0.1,0.2,\dots,0.9\};
(2) Data mixture trained models: Trained from scratch on mixed datasets 𝒟λ←λ⋅𝒟math+(1−λ)⋅𝒟code\mathcal{D}\_{\lambda}\leftarrow\lambda\cdot\mathcal{D}\_{\text{math}}+(1-\lambda)\cdot\mathcal{D}\_{\text{code}}111Without sacrificing clarity, we adopt the same interpolation formulation to represent the combined datasets sampled from 𝒟math\mathcal{D}\_{\text{math}} and 𝒟code\mathcal{D}\_{\text{code}}, with sampling ratio λ\lambda and 1−λ1-\lambda respectively. , with λ∈{0.1,0.2,…,0.9}\lambda\in\{0.1,0.2,\dots,0.9\}, for a total of 25B tokens per model.

!(/html/2601.17858/assets/x5.png)

Figure 4: Training dynamics comparison across five domains.
The light-colored curves (Pre-Anneal) track the performance of models trained with a constant learning rate.
The dark-colored curves (Annealed) represent the performance after applying learning rate annealing (simulated by merging the most recent 20B tokens).
The horizontal dashed lines denote the final performance by merging the top-16 checkpoints. The model trained with MergeMix-derived ratios consistently matches or outperforms the strong manually tuned baseline.

Table 1: Performance comparison of different data mixing methods across domains.

| Method | Math | Code | Knowledge | Language | Reasoning | Average |
| --- | --- | --- | --- | --- | --- | --- |
| Uniform | 57.1 | 44.6 | 50.0 | 76.0 | 56.7 | 54.2 |
| Adapted RegMix | 59.7 | 51.5 | 55.5 | 78.3 | 57.0 | 58.1 |
| Manual | 59.3 | 50.5 | 55.5 | 78.9 | 57.1 | 57.9 |
| MergeMix | 60.4 | 51.6 | 55.6 | 78.9 | 57.3 | 58.4 |

Table 2: Rank consistency (Spearman ρ\rho) between MergeMix predictions and actual training outcomes across domains.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Domain | Overall | Math | Code | Knowledge | Language | Reasoning |
| Spearman ρ\rho | 0.92 | 0.84 | 0.80 | 0.98 | 0.72 | 0.89 |

We evaluate both configuration sequences on standard math and code benchmarks. As shown in Figure [3](#S4.F3 "Figure 3 ‣ 4.2 Dual-Capability Merging Study ‣ 4 Experiment ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging"), the performance curves of the merged and trained models display highly synchronized trends, with two notable distinctions: (1) there is a consistent offset in performance scale between them, yet their relative trends across mixture ratios remain closely aligned; and (2) the performance curves of the merged models are noticeably smoother. We attribute these differences to the inherent advantages of model merging, which has been shown to mitigate training fluctuations and can even enhance final performance (Wang et al., [2025a](#bib.bib61); Tian et al., [2025](#bib.bib60)). We argue that the smoothness of model merging curves better reflects the underlying data–performance relationship and may provide a more stable and reliable proxy for ablation studies than full retraining compared to the fluctuations inherent in real training.

Considering the performance offset involved in two performance curves, we further assess the rank consistency. Figure [2](#S4.F2 "Figure 2 ‣ 4.2 Dual-Capability Merging Study ‣ 4 Experiment ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging") plots the rank of the trained models against the rank of the merged models across different mixtures. We observed a strong correlation.
This implies that if specific merging ratios are optimal in the weight space, the corresponding data mixture is highly likely to be optimal in the training space.
It is noteworthy that our method does not aim to predict absolute scores; instead, it predicts the relative ranking of mixtures to identify configurations that are comparatively better.
This strong rank consistency justifies using the computationally efficient weight space as a surrogate for the expensive data space.

### 4.3 Large-Scale Mid-Training Validation

#### Comparison Setup.

We proceed to validate the MergeMix framework in a large-scale mid-training setting. Following Nemotron’s data partitioning strategy (Basant et al., [2025](#bib.bib4)), we categorize the mid-training corpus into four primary domains for mixture ratio tuning: mathematics, code, supervised fine-tuning (SFT), and web/others data.
We employ the small MoE model, pre-trained on 2T tokens, and allocate a 200B-token mid-training budget.
Following the WSM schedule (Tian et al., [2025](#bib.bib60)), we train with a constant learning rate and use model merging to simulate the performance after model annealing. Specifically, we merge the best 16 checkpoints (ranked by benchmark performance) to further boost model performance and use this merged model to represent the final performance. We compare MergeMix against three baselines:
(1) *Uniform*: Uniform sampling across the four categories proportional to their total token counts;
(2) *Manual*: A manually tuned ratio derived from extensive ablation studies and heuristic tuning by our data team, which has already been deployed in production for our previous internal flagship model;
(3) *Adapted RegMix* (Liu et al., [2025](#bib.bib34)): A regressor-based predictor trained on the full-scale exploration runs collected from the manual baseline to optimize optimal mixtures. More details about baselines are shown in Appendix [C](#A3 "Appendix C Baselines and Computational Cost Analysis ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging").

!(/html/2601.17858/assets/x6.png)

Figure 5: Performance comparison between the MergeMix-optimized mixture, uni-domain baselines (100% single-domain data), and an aggressive specialization mixture.

!(/html/2601.17858/assets/x7.png)

(a) Cross-scale transfer.

!(/html/2601.17858/assets/x8.png)

(b) Dynamic vs. static schedule.

Figure 6: Cross-scale transfer from an 8B proxy to 16B target model (left) and dynamic data mix schedule comparison (right).

#### Performance and Efficiency.

Table [2](#S4.T2 "Table 2 ‣ 4.2 Dual-Capability Merging Study ‣ 4 Experiment ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging") and Figure [4](#S4.F4 "Figure 4 ‣ 4.2 Dual-Capability Merging Study ‣ 4 Experiment ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging") present the performance comparison across key benchmarks between the model trained with the MergeMix-derived mixture and baselines.
Remarkably, MergeMix matches or surpasses the manually tuned baseline across nearly all capability domains, delivering clear gains in math, code, and reasoning, while maintaining near-lossless performance in the remaining domains. Most critically, this result was achieved with a computational cost reduction of over 100×\times compared to the exhaustive ablation process required for the baseline. By substituting expensive full-scale trial runs with low-cost model merging inference, MergeMix enables highly efficient identification of optimal data mixtures.

#### Predictive Fidelity: Ranking Consistency.

To ensure the reliability of our proxy, we additionally sample 12 distinct mixtures and perform full mid-training runs to verify the predictions. Table [2](#S4.T2 "Table 2 ‣ 4.2 Dual-Capability Merging Study ‣ 4 Experiment ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging") reports the Spearman rank correlation (ρ\rho) between the predicted rank (via merged weights) and the actual mid-training outcomes. We observe high correlation (ρ>0.9\rho>0.9) on overall scores, confirming that the weight-space geometry accurately captures the dynamics of multi-domain data mixing. Notably, the prediction is nearly perfect for knowledge-related metrics (ρ=0.98\rho=0.98).

#### Landscape Analysis: Dissecting Capability Synergy.

Finally, MergeMix enables a holistic view of the correlating dynamics between data sources. Figure [8](#A5.F8 "Figure 8 ‣ Appendix E Capacity Landscape ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging") visualizes the performance heatmaps projected onto the weight simplex, revealing distinct topological patterns. For example, code exhibits high domain orthogonality with performance concentrated near its pure-domain composition, while the optimal math region centers around a mixed distribution, suggesting that reasoning is a composite capability requiring synergy between domain knowledge (math), instruction adherence (SFT), and broad linguistic understanding (web). To validate the superiority of this identified mixture, we focus on single domain and compare the MergeMix-derived configuration against uni-domain baselines (e.g., 100% math, 100% code) and an aggressive specialization mixture (high math/SFT, minimal web). As shown in Figure [5](#S4.F5 "Figure 5 ‣ Comparison Setup. ‣ 4.3 Large-Scale Mid-Training Validation ‣ 4 Experiment ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging"), the MergeMix-predicted mixture consistently outperforms these baselines. This empirically verifies that weight-space exploration accurately locates the “sweet spot” of data composition, harnessing the synergy between domains that heuristic or domain-specialization strategies fail to capture.

Table 3: Performance comparison of fine-grained mixing strategies. Notation ⟨A,B⟩\langle A,B\rangle denotes using strategy AA for coarse-level domain mixture and strategy BB for fine-grained sub-cluster mixture. Remarkably, the fully automated MergeMix approach (⟨MergeMix,MergeMix⟩\langle\text{MergeMix},\text{MergeMix}\rangle) outperforms the fully manual baseline.

| Mixing Strategy | Math | Code | Knowledge | Language | Reasoning | Average |
| --- | --- | --- | --- | --- | --- | --- |
| ⟨Manual,Manual⟩\langle\text{Manual},\text{Manual}\rangle | 59.3 | 50.5 | 55.5 | 78.9 | 57.1 | 57.9 |
| ⟨MergeMix,Uniform⟩\langle\text{MergeMix},\text{Uniform}\rangle | 58.7 | 51.1 | 55.2 | 77.9 | 56.5 | 57.6 |
| ⟨MergeMix,Manual⟩\langle\text{MergeMix},\text{Manual}\rangle | 60.4 | 51.6 | 55.6 | 78.9 | 57.3 | 58.4 |
| ⟨MergeMix,MergeMix⟩\langle\text{MergeMix},\text{MergeMix}\rangle | 59.8 | 51.3 | 55.3 | 79.1 | 57.2 | 58.2 |

### 4.4 Cross-Scale Transfer of Data Mixtures

Industrial computational budgets are often tight, especially for intensive mixture-tuning experiments. A key practical advantage would be the ability to identify an optimal data mixture with a lightweight proxy model and then transfer it effectively to a larger and more expensive target model.
To investigate this, we conduct a data mix transfer experiment. Instead of running the MergeMix pipeline on the large model (16B), we utilize the optimal mixture derived exclusively from the small model (8B). We then apply this ratio directly as the mixture for the mid-training of the 16B model.
The performance of the large model trained with the small model derived ratio is reported in Figure [6(a)](#S4.F6.sf1 "Figure 6(a) ‣ Figure 6 ‣ Comparison Setup. ‣ 4.3 Large-Scale Mid-Training Validation ‣ 4 Experiment ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging"). We observe that the mixture remains highly effective at the larger scale, even outperforming the manually tuned baseline. This cross-scale consistency further reduces computational overhead, enabling practitioners to conduct extensive exploration in weight space using economical, small-scale models to identify effective data mixtures, which can then be applied to larger-scale models.

### 4.5 Dynamic vs. Static Mixtures

Given that model capabilities evolve continuously during training, we investigate whether periodically recalibrating the data mixture yields better performance than maintaining a static mix fixed at initialization.
Specially, we conduct a two-stage mixture adjustment to investigate dynamic recalibration. We compare two strategies: (1) a static schedule, where the data mixture optimized at initialization is used throughout the training; and (2) a dynamic schedule, where the mixture is re-optimized at 50% training progress using MergeMix on the intermediate checkpoint, and updated for the second half.
As shown in Figure [6(b)](#S4.F6.sf2 "Figure 6(b) ‣ Figure 6 ‣ Comparison Setup. ‣ 4.3 Large-Scale Mid-Training Validation ‣ 4 Experiment ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging"), we find that the re-optimized mixture at the midpoint remains remarkably similar to the initial configuration, resulting in negligible performance differences between the two schedules. It suggests that the optimal data mixture is a stable, intrinsic property of the mid-training task. This stability implies that the optimization landscape does not shift significantly enough to warrant dynamic scheduling. Consequently, a single, static MergeMix search at initialization seems to be sufficient for mid-training, validating our method as a robust and efficient solution without the need for costly iterative re-calibration.

### 4.6 Hierarchical MergeMix

Table 4: Fine-grained categorization by primary domain.

| Coarse-level Domain | Fine-Grained Sub-clusters |
| --- | --- |
| Mathematics | Web Math, Math QA, Synthetic Study Notes, Formal Math |
| Code | Code Repos, Code NLP, Code Contest |
| SFT | Exam, General SFT, Long Chain-of-Thought |
| Web / Others | English Web, Chinese Web, Books, Wikipedia, Academic Papers |

In the previous experiments, we operate within a simplified setting of four coarse domains, where intra-domain mixtures are manually set. In this section, we advance toward fine-grained mixture optimization to further reduce manual intervention and facilitate practical deployment. A straightforward approach would be to treat fine-grained groups of datasets as direct input to MergeMix. However, this would drastically expand the dimensionality of the search space, making the optimization process both less efficient and less effective.
To address this, we extend MergeMix into a hierarchical divide-and-conquer framework. As discussed in Section [3.5](#S3.SS5 "3.5 Hierarchical Data Mix ‣ 3 The MergeMix Framework ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging"), this hierarchy can be traversed in two directions. In this experiment, we adopt a top-down strategy: we first optimize the coarse-level mixture based on the optimal values derived in Section [4.3](#S4.SS3 "4.3 Large-Scale Mid-Training Validation ‣ 4 Experiment ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging"), and subsequently refine the fine-grained mixture within each domain.

We decompose the four coarse domains into 16 distinct sub-clusters (taxonomy detailed in Table [4](#S4.T4 "Table 4 ‣ 4.6 Hierarchical MergeMix ‣ 4 Experiment ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging")).
The performance comparison is presented in Table [3](#S4.T3 "Table 3 ‣ Landscape Analysis: Dissecting Capability Synergy. ‣ 4.3 Large-Scale Mid-Training Validation ‣ 4 Experiment ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging").
Remarkably, the fully automated strategy (denoted by ⟨*MergeMix*,*MergeMix*⟩\langle\emph{MergeMix},\emph{MergeMix}\rangle) outperforms the fully manual baseline (i.e., ⟨*Manual*,*Manual*⟩\langle\emph{Manual},\emph{Manual}\rangle). Our method successfully identifies high-value fine-grained sub-clusters, such as assigning higher weights to Math QA and Exam datasets without any human prior. With the coarse-level distribution fixed, MergeMix optimizes fine-grained weights to outperform the uniform baseline (i.e., ⟨*MergeMix*,*Uniform*⟩\langle\emph{MergeMix},\emph{Uniform}\rangle) by 0.6%, achieving performance comparable to exhaustive manual fine-grained tuning (i.e., ⟨*MergeMix*,*Manual*⟩\langle\emph{MergeMix},\emph{Manual}\rangle).

## 5 Conclusion

In this work, we introduce MergeMix, a resource-efficient framework for optimizing mid-training data mixtures for LLMs. By theoretically and empirically establishing that linear weight interpolation serves as a high-fidelity proxy for data gradient accumulation, we transform the computationally prohibitive problem of data mixture tuning into a low-cost model merging optimization task.
Our extensive validation shows that MergeMix identifies data configurations that match or exceed the performance of exhaustive manual tuning and current automated methods, while reducing search costs by orders of magnitude. The framework demonstrates strong rank consistency and cross-scale transferability. MergeMix provides a scalable, automated pathway to enhancing model capabilities, shifting the paradigm of data composition from heuristic guesswork toward precise, objective-driven engineering.

## References

* Ahmadian et al. (2024)

  Ahmadian, A., Goldfarb-Tarrant, S., Ermis, B., Fadaee, M., Hooker, S., et al.
  Mix data or merge models? optimizing for diverse multi-task learning.
  *arXiv preprint arXiv:2410.10801*, 2024.
* Ainslie et al. (2023)

  Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., and Sanghai, S.
  GQA: training generalized multi-query transformer models from multi-head checkpoints.
  In Bouamor, H., Pino, J., and Bali, K. (eds.), *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023*, pp. 4895–4901. Association for Computational Linguistics, 2023.
  doi: 10.18653/V1/2023.EMNLP-MAIN.298.
  URL <https://doi.org/10.18653/v1/2023.emnlp-main.298>.
* Albalak et al. (2024)

  Albalak, A., Elazar, Y., Xie, S. M., Longpre, S., Lambert, N., Wang, X., Muennighoff, N., Hou, B., Pan, L., Jeong, H., Raffel, C., Chang, S., Hashimoto, T., and Wang, W. Y.
  A survey on data selection for language models.
  *Trans. Mach. Learn. Res.*, 2024, 2024.
  URL <https://openreview.net/forum?id=XfHWcNTSHp>.
* Basant et al. (2025)

  Basant, A., Khairnar, A., Paithankar, A., Khattar, A., Renduchintala, A., Malte, A., Bercovich, A., Hazare, A., Rico, A., Ficek, A., Kondratenko, A., Shaposhnikov, A., Bukharin, A., Taghibakhshi, A., Barton, A., Mahabaleshwarkar, A. S., Shen, A., Tao, A., Guan, A., Shors, A., Mandarwal, A., Mehta, A., Venkatesan, A., Sharabiani, A., Aithal, A., Poojary, A., Dattagupta, A., Buddharaju, B., Zhu, B., Simkin, B., Kartal, B., Rouhani, B. D., Chen, B., Ginsburg, B., Norick, B., Yu, B., Catanzaro, B., Wang, C., Truong, C., Mungekar, C., Patel, C., Alexiuk, C., Munley, C., Parisien, C., Su, D., Afrimi, D., Korzekwa, D., Rohrer, D., Gitman, D., Mosallanezhad, D., Narayanan, D., Rekesh, D., Yared, D., Pykhtar, D., Ahn, D., Riach, D., Long, E., Ning, E., Chung, E., Galinkin, E., Bakhturina, E., Prasad, G., Shen, G., Qian, H., Elisha, H., Sharma, H., Ross, H., Ngo, H., Sahota, H., Wang, H., Shin, H. C., Huang, H., Cunningham, I., Gitman, I., Moshkov, I., Jung, J., Kautz, J., Scowcroft, J. P., Casper, J., Zhang, J., Zeng,
  J., Zhang, J., Xue, J., Huang, J., Conway, J., Kamalu, J., Cohen, J. M., Jennings, J., Vialard, J. V., Yi, J., Parmar, J., Briski, K., Cheung, K., Luna, K., Ross, K. W., Santhanam, K., Kong, K., Pawelec, K., and Anik, K.
  NVIDIA nemotron nano 2: An accurate and efficient hybrid mamba-transformer reasoning model.
  *CoRR*, abs/2508.14444, 2025.
  doi: 10.48550/ARXIV.2508.14444.
  URL <https://doi.org/10.48550/arXiv.2508.14444>.
* Bavarian et al. (2022)

  Bavarian, M., Jun, H., Tezak, N., Schulman, J., McLeavey, C., Tworek, J., and Chen, M.
  Efficient training of language models to fill in the middle.
  *CoRR*, abs/2207.14255, 2022.
  doi: 10.48550/ARXIV.2207.14255.
  URL <https://doi.org/10.48550/arXiv.2207.14255>.
* Belenki et al. (2025)

  Belenki, L., Agarwal, A., Shi, T., and Toutanova, K.
  Optimizing pre-training data mixtures with mixtures of data expert models.
  In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025*, pp. 32570–32587. Association for Computational Linguistics, 2025.
  URL <https://aclanthology.org/2025.acl-long.1564/>.
* Bhakthavatsalam et al. (2021)

  Bhakthavatsalam, S., Khashabi, D., Khot, T., Mishra, B. D., Richardson, K., Sabharwal, A., Schoenick, C., Tafjord, O., and Clark, P.
  Think you have solved direct-answer question answering? try arc-da, the direct-answer AI2 reasoning challenge.
  *CoRR*, abs/2102.03315, 2021.
  URL <https://arxiv.org/abs/2102.03315>.
* Bisk et al. (2020)

  Bisk, Y., Zellers, R., Bras, R. L., Gao, J., and Choi, Y.
  PIQA: reasoning about physical commonsense in natural language.
  In *The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020*, pp. 7432–7439. AAAI Press, 2020.
  doi: 10.1609/AAAI.V34I05.6239.
  URL <https://doi.org/10.1609/aaai.v34i05.6239>.
* Chen et al. (2021)

  Chen, M., Tworek, J., Jun, H., Yuan, Q., de Oliveira Pinto, H. P., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, M., Winter, C., Tillet, P., Such, F. P., Cummings, D., Plappert, M., Chantzis, F., Barnes, E., Herbert-Voss, A., Guss, W. H., Nichol, A., Paino, A., Tezak, N., Tang, J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W., Hesse, C., Carr, A. N., Leike, J., Achiam, J., Misra, V., Morikawa, E., Radford, A., Knight, M., Brundage, M., Murati, M., Mayer, K., Welinder, P., McGrew, B., Amodei, D., McCandlish, S., Sutskever, I., and Zaremba, W.
  Evaluating large language models trained on code.
  *CoRR*, abs/2107.03374, 2021.
  URL <https://arxiv.org/abs/2107.03374>.
* Chen et al. (2025)

  Chen, S., Ouyang, X., Pearce, M. A. L., Hartvigsen, T., and Schwarz, J. R.
  Admire-bayesopt: Accelerated data mixture re-weighting for language models with bayesian optimization.
  *CoRR*, abs/2508.11551, 2025.
  doi: 10.48550/ARXIV.2508.11551.
  URL <https://doi.org/10.48550/arXiv.2508.11551>.
* Cobbe et al. (2021)

  Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J.
  Training verifiers to solve math word problems.
  *CoRR*, abs/2110.14168, 2021.
  URL <https://arxiv.org/abs/2110.14168>.
* Diao et al. (2025)

  Diao, S., Yang, Y., Fu, Y., Dong, X., Su, D., Kliegl, M., Chen, Z., Belcak, P., Suhara, Y., Yin, H., Patwary, M., Lin, Y., Kautz, J., and Molchanov, P.
  CLIMB: clustering-based iterative data mixture bootstrapping for language model pre-training.
  *CoRR*, abs/2504.13161, 2025.
  doi: 10.48550/ARXIV.2504.13161.
  URL <https://doi.org/10.48550/arXiv.2504.13161>.
* Frankle et al. (2020)

  Frankle, J., Dziugaite, G. K., Roy, D. M., and Carbin, M.
  Linear mode connectivity and the lottery ticket hypothesis.
  In *Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event*, volume 119 of *Proceedings of Machine Learning Research*, pp. 3259–3269. PMLR, 2020.
  URL <http://proceedings.mlr.press/v119/frankle20a.html>.
* Ge et al. (2024)

  Ge, C., Ma, Z., Chen, D., Li, Y., and Ding, B.
  Bimix: A bivariate data mixing law for language model pretraining.
  *arXiv preprint arXiv:2405.14908*, 2024.
* Gu et al. (2024a)

  Gu, A., Rozière, B., Leather, H. J., Solar-Lezama, A., Synnaeve, G., and Wang, S.
  Cruxeval: A benchmark for code reasoning, understanding and execution.
  In *Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024*. OpenReview.net, 2024a.
  URL <https://openreview.net/forum?id=Ffpg52swvg>.
* Gu et al. (2024b)

  Gu, J., Yang, Z., Ding, C., Zhao, R., and Tan, F.
  CMR scaling law: Predicting critical mixture ratios for continual pre-training of language models.
  In Al-Onaizan, Y., Bansal, M., and Chen, Y. (eds.), *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024*, pp. 16143–16162. Association for Computational Linguistics, 2024b.
  doi: 10.18653/V1/2024.EMNLP-MAIN.903.
  URL <https://doi.org/10.18653/v1/2024.emnlp-main.903>.
* Hendrycks et al. (2021a)

  Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J.
  Measuring massive multitask language understanding.
  In *9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021*. OpenReview.net, 2021a.
  URL <https://openreview.net/forum?id=d7KBjmI3GmQ>.
* Hendrycks et al. (2021b)

  Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J.
  Measuring mathematical problem solving with the MATH dataset.
  In Vanschoren, J. and Yeung, S. (eds.), *Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual*, 2021b.
* Hong et al. (2025)

  Hong, J., Yan, S., Cai, J., Jiang, X., Hu, Y., and Xie, W.
  Worldsense: Evaluating real-world omnimodal understanding for multimodal llms.
  *CoRR*, abs/2502.04326, 2025.
  doi: 10.48550/ARXIV.2502.04326.
  URL <https://doi.org/10.48550/arXiv.2502.04326>.
* Hu et al. (2024)

  Hu, S., Tu, Y., Han, X., He, C., Cui, G., Long, X., Zheng, Z., Fang, Y., Huang, Y., Zhao, W., Zhang, X., Thai, Z. L., Zhang, K., Wang, C., Yao, Y., Zhao, C., Zhou, J., Cai, J., Zhai, Z., Ding, N., Jia, C., Zeng, G., Li, D., Liu, Z., and Sun, M.
  Minicpm: Unveiling the potential of small language models with scalable training strategies.
  *CoRR*, abs/2404.06395, 2024.
  doi: 10.48550/ARXIV.2404.06395.
  URL <https://doi.org/10.48550/arXiv.2404.06395>.
* Huang et al. (2023)

  Huang, Y., Bai, Y., Zhu, Z., Zhang, J., Zhang, J., Su, T., Liu, J., Lv, C., Zhang, Y., Lei, J., Fu, Y., Sun, M., and He, J.
  C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models.
  In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S. (eds.), *Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023*, 2023.
* Ilharco et al. (2023)

  Ilharco, G., Ribeiro, M. T., Wortsman, M., Schmidt, L., Hajishirzi, H., and Farhadi, A.
  Editing models with task arithmetic.
  In *The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023*. OpenReview.net, 2023.
  URL <https://openreview.net/forum?id=6t0Kwf8-jrj>.
* Izmailov et al. (2018)

  Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D. P., and Wilson, A. G.
  Averaging weights leads to wider optima and better generalization.
  In Globerson, A. and Silva, R. (eds.), *Proceedings of the Thirty-Fourth Conference on Uncertainty in Artificial Intelligence, UAI 2018, Monterey, California, USA, August 6-10, 2018*, pp. 876–885. AUAI Press, 2018.
  URL <http://auai.org/uai2018/proceedings/papers/313.pdf>.
* Jain et al. (2025)

  Jain, N., Han, K., Gu, A., Li, W., Yan, F., Zhang, T., Wang, S., Solar-Lezama, A., Sen, K., and Stoica, I.
  Livecodebench: Holistic and contamination free evaluation of large language models for code.
  In *The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025*. OpenReview.net, 2025.
  URL <https://openreview.net/forum?id=chfJJYC3iL>.
* Joshi et al. (2017)

  Joshi, M., Choi, E., Weld, D. S., and Zettlemoyer, L.
  Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension.
  In Barzilay, R. and Kan, M. (eds.), *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers*, pp. 1601–1611. Association for Computational Linguistics, 2017.
  doi: 10.18653/V1/P17-1147.
  URL <https://doi.org/10.18653/v1/P17-1147>.
* Ke et al. (2017)

  Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.
  Lightgbm: A highly efficient gradient boosting decision tree.
  In Guyon, I., von Luxburg, U., Bengio, S., Wallach, H. M., Fergus, R., Vishwanathan, S. V. N., and Garnett, R. (eds.), *Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA*, pp. 3146–3154, 2017.
* Khalifa et al. (2024)

  Khalifa, M., Tan, Y. C., Ahmadian, A., Hosking, T., Lee, H., Wang, L., Üstün, A., Sherborne, T., and Gallé, M.
  If you can’t use them, recycle them: Optimizing merging at scale mitigates performance tradeoffs.
  *CoRR*, abs/2412.04144, 2024.
  doi: 10.48550/ARXIV.2412.04144.
  URL <https://doi.org/10.48550/arXiv.2412.04144>.
* Kwiatkowski et al. (2019)

  Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A. P., Alberti, C., Epstein, D., Polosukhin, I., Devlin, J., Lee, K., Toutanova, K., Jones, L., Kelcey, M., Chang, M., Dai, A. M., Uszkoreit, J., Le, Q., and Petrov, S.
  Natural questions: a benchmark for question answering research.
  *Trans. Assoc. Comput. Linguistics*, 7:452–466, 2019.
  doi: 10.1162/TACL“˙A“˙00276.
  URL <https://doi.org/10.1162/tacl_a_00276>.
* Lai et al. (2017)

  Lai, G., Xie, Q., Liu, H., Yang, Y., and Hovy, E. H.
  RACE: large-scale reading comprehension dataset from examinations.
  In Palmer, M., Hwa, R., and Riedel, S. (eds.), *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, EMNLP 2017, Copenhagen, Denmark, September 9-11, 2017*, pp. 785–794. Association for Computational Linguistics, 2017.
  doi: 10.18653/V1/D17-1082.
  URL <https://doi.org/10.18653/v1/d17-1082>.
* Li et al. (2024a)

  Li, H., Zhang, Y., Koto, F., Yang, Y., Zhao, H., Gong, Y., Duan, N., and Baldwin, T.
  CMMLU: measuring massive multitask language understanding in chinese.
  In Ku, L., Martins, A., and Srikumar, V. (eds.), *Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024*, pp. 11260–11285. Association for Computational Linguistics, 2024a.
  doi: 10.18653/V1/2024.FINDINGS-ACL.671.
  URL <https://doi.org/10.18653/v1/2024.findings-acl.671>.
* Li et al. (2024b)

  Li, Q., Cui, L., Zhao, X., Kong, L., and Bi, W.
  Gsm-plus: A comprehensive benchmark for evaluating the robustness of llms as mathematical problem solvers.
  In Ku, L., Martins, A., and Srikumar, V. (eds.), *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024*, pp. 2961–2984. Association for Computational Linguistics, 2024b.
  doi: 10.18653/V1/2024.ACL-LONG.163.
  URL <https://doi.org/10.18653/v1/2024.acl-long.163>.
* Liu et al. (2024)

  Liu, H., Zheng, Z., Qiao, Y., Duan, H., Fei, Z., Zhou, F., Zhang, W., Zhang, S., Lin, D., and Chen, K.
  Mathbench: Evaluating the theory and application proficiency of llms with a hierarchical mathematics benchmark.
  In Ku, L., Martins, A., and Srikumar, V. (eds.), *Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024*, pp. 6884–6915. Association for Computational Linguistics, 2024.
  doi: 10.18653/V1/2024.FINDINGS-ACL.411.
  URL <https://doi.org/10.18653/v1/2024.findings-acl.411>.
* Liu et al. (2023)

  Liu, J., Xia, C. S., Wang, Y., and Zhang, L.
  Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation.
  In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S. (eds.), *Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023*, 2023.
* Liu et al. (2025)

  Liu, Q., Zheng, X., Muennighoff, N., Zeng, G., Dou, L., Pang, T., Jiang, J., and Lin, M.
  Regmix: Data mixture as regression for language model pre-training.
  In *The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025*. OpenReview.net, 2025.
  URL <https://openreview.net/forum?id=5BjQOUXq7i>.
* Loshchilov & Hutter (2019)

  Loshchilov, I. and Hutter, F.
  Decoupled weight decay regularization.
  In *7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019*. OpenReview.net, 2019.
  URL <https://openreview.net/forum?id=Bkg6RiCqY7>.
* Lourie et al. (2025)

  Lourie, N., Hu, M. Y., and Cho, K.
  Scaling laws are unreliable for downstream tasks: A reality check.
  *CoRR*, abs/2507.00885, 2025.
  doi: 10.48550/ARXIV.2507.00885.
  URL <https://doi.org/10.48550/arXiv.2507.00885>.
* Lu et al. (2025)

  Lu, Z., Zhou, A., Wang, K., Ren, H., Shi, W., Pan, J., Zhan, M., and Li, H.
  Mathcoder2: Better math reasoning from continued pretraining on model-translated mathematical code.
  In *The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025*. OpenReview.net, 2025.
  URL <https://openreview.net/forum?id=1Iuw1jcIrf>.
* Luo et al. (2025)

  Luo, J., Wu, B., Luo, X., Xiao, Z., Jin, Y., Tu, R., Yin, N., Wang, Y., Yuan, J., Ju, W., and Zhang, M.
  A survey on efficient large language model training: From data-centric perspectives.
  In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025*, pp. 30904–30920. Association for Computational Linguistics, 2025.
  URL <https://aclanthology.org/2025.acl-long.1493/>.
* Ma et al. (2025)

  Ma, K., Du, X., Wang, Y., Zhang, H., Wen, Z., Qu, X., Yang, J., Liu, J., Liu, M., Yue, X., Huang, W., and Zhang, G.
  Kor-bench: Benchmarking language models on knowledge-orthogonal reasoning tasks.
  In *The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025*. OpenReview.net, 2025.
  URL <https://openreview.net/forum?id=SVRRQ8goQo>.
* Ma et al. (2024)

  Ma, Y., Liu, Y., Yu, Y., Zhang, Y., Jiang, Y., Wang, C., and Li, S.
  At which training stage does code data help llms reasoning?
  In *The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024*. OpenReview.net, 2024.
  URL <https://openreview.net/forum?id=KIPJKST4gw>.
* Maiti et al. (2025)

  Maiti, S., Budhiraja, A., Gauri, B., Chaurasia, G., Protopopov, A., Audran-Reiss, A., Slater, M., Magka, D., Shavrina, T., Raileanu, R., et al.
  Souper-model: How simple arithmetic unlocks state-of-the-art llm performance.
  *arXiv preprint arXiv:2511.13254*, 2025.
* Mihaylov et al. (2018)

  Mihaylov, T., Clark, P., Khot, T., and Sabharwal, A.
  Can a suit of armor conduct electricity? A new dataset for open book question answering.
  In Riloff, E., Chiang, D., Hockenmaier, J., and Tsujii, J. (eds.), *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018*, pp. 2381–2391. Association for Computational Linguistics, 2018.
  doi: 10.18653/V1/D18-1260.
  URL <https://doi.org/10.18653/v1/d18-1260>.
* Na et al. (2024)

  Na, C., Magnusson, I., Jha, A. H., Sherborne, T., Strubell, E., Dodge, J., and Dasigi, P.
  Scalable data ablation approximations for language models through modular training and merging.
  In Al-Onaizan, Y., Bansal, M., and Chen, Y. (eds.), *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024*, pp. 21125–21141. Association for Computational Linguistics, 2024.
  doi: 10.18653/V1/2024.EMNLP-MAIN.1176.
  URL <https://doi.org/10.18653/v1/2024.emnlp-main.1176>.
* Olmo et al. (2025)

  Olmo, T., Ettinger, A., Bertsch, A., Kuehl, B., Graham, D., Heineman, D., Groeneveld, D., Brahman, F., Timbers, F., Ivison, H., et al.
  Olmo 3.
  *arXiv preprint arXiv:2512.13961*, 2025.
* Peng et al. (2024)

  Peng, Q., Chai, Y., and Li, X.
  Humaneval-xl: A multilingual code generation benchmark for cross-lingual natural language generalization.
  In Calzolari, N., Kan, M., Hoste, V., Lenci, A., Sakti, S., and Xue, N. (eds.), *Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation, LREC/COLING 2024, 20-25 May, 2024, Torino, Italy*, pp. 8383–8394. ELRA and ICCL, 2024.
  URL <https://aclanthology.org/2024.lrec-main.735>.
* Pruthi et al. (2020)

  Pruthi, G., Liu, F., Kale, S., and Sundararajan, M.
  Estimating training data influence by tracing gradient descent.
  In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*, 2020.
  URL <https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html>.
* Rajpurkar et al. (2018)

  Rajpurkar, P., Jia, R., and Liang, P.
  Know what you don’t know: Unanswerable questions for squad.
  In Gurevych, I. and Miyao, Y. (eds.), *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, ACL 2018, Melbourne, Australia, July 15-20, 2018, Volume 2: Short Papers*, pp. 784–789. Association for Computational Linguistics, 2018.
  doi: 10.18653/V1/P18-2124.
  URL <https://aclanthology.org/P18-2124/>.
* Rein et al. (2023)

  Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., and Bowman, S. R.
  GPQA: A graduate-level google-proof q&a benchmark.
  *CoRR*, abs/2311.12022, 2023.
  doi: 10.48550/ARXIV.2311.12022.
  URL <https://doi.org/10.48550/arXiv.2311.12022>.
* Sakaguchi et al. (2021)

  Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y.
  Winogrande: an adversarial winograd schema challenge at scale.
  *Commun. ACM*, 64(9):99–106, 2021.
  doi: 10.1145/3474381.
  URL <https://doi.org/10.1145/3474381>.
* Shi et al. (2023)

  Shi, F., Suzgun, M., Freitag, M., Wang, X., Srivats, S., Vosoughi, S., Chung, H. W., Tay, Y., Ruder, S., Zhou, D., Das, D., and Wei, J.
  Language models are multilingual chain-of-thought reasoners.
  In *The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023*. OpenReview.net, 2023.
  URL <https://openreview.net/forum?id=fR3wGCk-IXp>.
* Shukor et al. (2025)

  Shukor, M., Béthune, L., Busbridge, D., Grangier, D., Fini, E., El-Nouby, A., and Ablin, P.
  Scaling laws for optimal data mixtures.
  *CoRR*, abs/2507.09404, 2025.
  doi: 10.48550/ARXIV.2507.09404.
  URL <https://doi.org/10.48550/arXiv.2507.09404>.
* Su et al. (2024)

  Su, J., Ahmed, M. H. M., Lu, Y., Pan, S., Bo, W., and Liu, Y.
  Roformer: Enhanced transformer with rotary position embedding.
  *Neurocomputing*, 568:127063, 2024.
  doi: 10.1016/J.NEUCOM.2023.127063.
  URL <https://doi.org/10.1016/j.neucom.2023.127063>.
* Suzgun et al. (2023)

  Suzgun, M., Scales, N., Schärli, N., Gehrmann, S., Tay, Y., Chung, H. W., Chowdhery, A., Le, Q. V., Chi, E. H., Zhou, D., and Wei, J.
  Challenging big-bench tasks and whether chain-of-thought can solve them.
  In Rogers, A., Boyd-Graber, J. L., and Okazaki, N. (eds.), *Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023*, pp. 13003–13051. Association for Computational Linguistics, 2023.
  doi: 10.18653/V1/2023.FINDINGS-ACL.824.
  URL <https://doi.org/10.18653/v1/2023.findings-acl.824>.
* Tang et al. (2024)

  Tang, Z., Zhang, X., Wang, B., and Wei, F.
  Mathscale: Scaling instruction tuning for mathematical reasoning.
  In *Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024*. OpenReview.net, 2024.
  URL <https://openreview.net/forum?id=Kjww7ZN47M>.
* Tao et al. (2024)

  Tao, N., Ventresque, A., Nallur, V., and Saber, T.
  Enhancing program synthesis with large language models using many-objective grammar-guided genetic programming.
  *Algorithms*, 17(7):287, 2024.
  doi: 10.3390/A17070287.
  URL <https://doi.org/10.3390/a17070287>.
* Tao et al. (2025)

  Tao, Z. S., Vinken, K., Yeh, H., Cooper, A., and Boix, X.
  Merge to mix: Mixing datasets via model merging.
  *CoRR*, abs/2505.16066, 2025.
  doi: 10.48550/ARXIV.2505.16066.
  URL <https://doi.org/10.48550/arXiv.2505.16066>.
* Team et al. (2025a)

  Team, L., Li, A., Liu, B., Hu, B., Li, B., Zeng, B., Ye, B., Tang, C., Tian, C., Huang, C., et al.
  Every activation boosted: Scaling general reasoner to 1 trillion open language foundation.
  *arXiv preprint arXiv:2510.22115*, 2025a.
* Team et al. (2025b)

  Team, M., Du, X., Yao, Y., Ma, K., Wang, B., Zheng, T., Zhu, K., Liu, M., Liang, Y., Jin, X., Wei, Z., Zheng, C., Deng, K., Jia, S., Jiang, S., Liao, Y., Li, R., Li, Q., Li, S., Li, Y., Li, Y., Ma, D., Ni, Y., Que, H., Wang, Q., Wen, Z., Wu, S., Xing, T., Xu, M., Yang, Z., Wang, Z. M., Zhou, J., Bai, Y., Bu, X., Cai, C., Chen, L., Chen, Y., Cheng, C., Cheng, T., Ding, K., Huang, S., Huang, Y., Li, Y., Li, Y., Li, Z., Liang, T., Lin, C., Lin, H., Ma, Y., Pang, T., Peng, Z., Peng, Z., Qi, Q., Qiu, S., Qu, X., Quan, S., Tan, Y., Wang, Z., Wang, C., Wang, H., Wang, Y., Wang, Y., Xu, J., Yang, K., Yuan, R., Yue, Y., Zhan, T., Zhang, C., Zhang, J., Zhang, X., Zhang, X., Zhang, Y., Zhao, Y., Zheng, X., Zhong, C., Gao, Y., Li, Z., Liu, D., Liu, Q., Liu, T., Ni, S., Peng, J., Qin, Y., Su, W., Wang, G., Wang, S., Yang, J., Yang, M., Cao, M., Yue, X., Zhang, Z., Zhou, W., Liu, J., Lin, Q., Huang, W., and Zhang, G.
  Supergpqa: Scaling LLM evaluation across 285 graduate disciplines.
  *CoRR*, abs/2502.14739, 2025b.
  doi: 10.48550/ARXIV.2502.14739.
  URL <https://doi.org/10.48550/arXiv.2502.14739>.
* Thudi et al. (2025)

  Thudi, A., Rovers, E., Ruan, Y., Thrush, T., and Maddison, C. J.
  Mixmin: Finding data mixtures via convex minimization.
  *arXiv preprint arXiv:2502.10510*, 2025.
* Tian et al. (2025)

  Tian, C., Wang, J., Zhao, Q., Chen, K., Liu, J., Liu, Z., Mao, J., Zhao, W. X., Zhang, Z., and Zhou, J.
  WSM: decay-free learning rate schedule via checkpoint merging for LLM pre-training.
  *CoRR*, abs/2507.17634, 2025.
  doi: 10.48550/ARXIV.2507.17634.
  URL <https://doi.org/10.48550/arXiv.2507.17634>.
* Wang et al. (2025a)

  Wang, J., Tian, C., Chen, K., Liu, Z., Mao, J., Zhao, W. X., Zhang, Z., and Zhou, J.
  Map: A unified framework for reliable evaluation of pre-training dynamics.
  *CoRR*, abs/2510.09295, 2025a.
  doi: 10.48550/ARXIV.2510.09295.
  URL <https://doi.org/10.48550/arXiv.2510.09295>.
* Wang et al. (2025b)

  Wang, J. T., Mittal, P., Song, D., and Jia, R.
  Data shapley in one training run.
  In *The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025*. OpenReview.net, 2025b.
  URL <https://openreview.net/forum?id=HD6bWcj87Y>.
* Wang et al. (2024)

  Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., Li, T., Ku, M., Wang, K., Zhuang, A., Fan, R., Yue, X., and Chen, W.
  Mmlu-pro: A more robust and challenging multi-task language understanding benchmark.
  In Globersons, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J. M., and Zhang, C. (eds.), *Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024*, 2024.
* Wei et al. (2023)

  Wei, T., Luan, J., Liu, W., Dong, S., and Wang, B.
  CMATH: can your language model pass chinese elementary school math test?
  *CoRR*, abs/2306.16636, 2023.
  doi: 10.48550/ARXIV.2306.16636.
  URL <https://doi.org/10.48550/arXiv.2306.16636>.
* Wortsman et al. (2022)

  Wortsman, M., Ilharco, G., Gadre, S. Y., Roelofs, R., Lopes, R. G., Morcos, A. S., Namkoong, H., Farhadi, A., Carmon, Y., Kornblith, S., and Schmidt, L.
  Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.
  In Chaudhuri, K., Jegelka, S., Song, L., Szepesvári, C., Niu, G., and Sabato, S. (eds.), *International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA*, volume 162 of *Proceedings of Machine Learning Research*, pp. 23965–23998. PMLR, 2022.
  URL <https://proceedings.mlr.press/v162/wortsman22a.html>.
* Xie et al. (2023)

  Xie, S. M., Pham, H., Dong, X., Du, N., Liu, H., Lu, Y., Liang, P., Le, Q. V., Ma, T., and Yu, A. W.
  Doremi: Optimizing data mixtures speeds up language model pretraining.
  In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S. (eds.), *Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023*, 2023.
* Yadav et al. (2023)

  Yadav, P., Tam, D., Choshen, L., Raffel, C. A., and Bansal, M.
  Ties-merging: Resolving interference when merging models.
  In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S. (eds.), *Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023*, 2023.
* Yang et al. (2025)

  Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang, J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Bao, K., Yang, K., Yu, L., Deng, L., Li, M., Xue, M., Li, M., Zhang, P., Wang, P., Zhu, Q., Men, R., Gao, R., Liu, S., Luo, S., Li, T., Tang, T., Yin, W., Ren, X., Wang, X., Zhang, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Zhang, Y., Wan, Y., Liu, Y., Wang, Z., Cui, Z., Zhang, Z., Zhou, Z., and Qiu, Z.
  Qwen3 technical report.
  *CoRR*, abs/2505.09388, 2025.
  doi: 10.48550/ARXIV.2505.09388.
  URL <https://doi.org/10.48550/arXiv.2505.09388>.
* Yang et al. (2024)

  Yang, E., Wang, Z., Shen, L., Liu, S., Guo, G., Wang, X., and Tao, D.
  Adamerging: Adaptive model merging for multi-task learning.
  In *The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024*. OpenReview.net, 2024.
  URL <https://openreview.net/forum?id=nZP6NgD3QY>.
* Ye et al. (2025)

  Ye, J., Liu, P., Sun, T., Zhan, J., Zhou, Y., and Qiu, X.
  Data mixing laws: Optimizing data mixtures by predicting language modeling performance.
  In *The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025*. OpenReview.net, 2025.
  URL <https://openreview.net/forum?id=jjCB27TMK3>.
* Yeh et al. (2022)

  Yeh, C.-K., Taly, A., Sundararajan, M., Liu, F., and Ravikumar, P.
  First is better than last for language data influence.
  *Advances in Neural Information Processing Systems*, 35:32285–32298, 2022.
* Yu et al. (2024)

  Yu, L., Yu, B., Yu, H., Huang, F., and Li, Y.
  Language models are super mario: Absorbing abilities from homologous models as a free lunch.
  In *Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024*. OpenReview.net, 2024.
  URL <https://openreview.net/forum?id=fq0NaiU8Ex>.
* Yu et al. (2020)

  Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., and Finn, C.
  Gradient surgery for multi-task learning.
  In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*, 2020.
* Zellers et al. (2019)

  Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y.
  Hellaswag: Can a machine really finish your sentence?
  In Korhonen, A., Traum, D. R., and Màrquez, L. (eds.), *Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers*, pp. 4791–4800. Association for Computational Linguistics, 2019.
  doi: 10.18653/V1/P19-1472.
  URL <https://doi.org/10.18653/v1/p19-1472>.
* Zhang et al. (2023)

  Zhang, X., Li, C., Zong, Y., Ying, Z., He, L., and Qiu, X.
  Evaluating the performance of large language models on GAOKAO benchmark.
  *CoRR*, abs/2305.12474, 2023.
  doi: 10.48550/ARXIV.2305.12474.
  URL <https://doi.org/10.48550/arXiv.2305.12474>.
* Zhao et al. (2023)

  Zhao, W. X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y., Min, Y., Zhang, B., Zhang, J., Dong, Z., Du, Y., Yang, C., Chen, Y., Chen, Z., Jiang, J., Ren, R., Li, Y., Tang, X., Liu, Z., Liu, P., Nie, J., and Wen, J.
  A survey of large language models.
  *CoRR*, abs/2303.18223, 2023.
  doi: 10.48550/ARXIV.2303.18223.
  URL <https://doi.org/10.48550/arXiv.2303.18223>.
* Zhong et al. (2024)

  Zhong, W., Cui, R., Guo, Y., Liang, Y., Lu, S., Wang, Y., Saied, A., Chen, W., and Duan, N.
  Agieval: A human-centric benchmark for evaluating foundation models.
  In Duh, K., Gómez-Adorno, H., and Bethard, S. (eds.), *Findings of the Association for Computational Linguistics: NAACL 2024, Mexico City, Mexico, June 16-21, 2024*, pp. 2299–2314. Association for Computational Linguistics, 2024.
  doi: 10.18653/V1/2024.FINDINGS-NAACL.149.
  URL <https://doi.org/10.18653/v1/2024.findings-naacl.149>.

## Appendix A Model Architecture

The core architectures of our experimental 8B MoE model are detailed in Table [5](#A1.T5 "Table 5 ‣ Appendix A Model Architecture ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging").
The model is configured with 20 layers and a hidden dimension size of 2048. Except for the first layer, all FFNs layers are replaced with MoE layers. We adopt the GQA attention mechanism (Ainslie et al., [2023](#bib.bib2)) and integrate Rotary Position Embedding (RoPE) (Su et al., [2024](#bib.bib52)), enabling the model to support sequence lengths up to 8K tokens. For parameter initialization, all learnable parameters are randomly initialized using a standard deviation of 0.006.
Under this configuration, the model consists of a total of 8.6 billion parameters, of which approximately 1.43 billion are activated for each token during inference.

Table 5: Detailed model architecture.

| Parameter | Value |
| --- | --- |
| Number of layers (nl​a​y​e​r​sn\_{layers}) | 20 |
| Model dimension (dm​o​d​e​ld\_{model}) | 2,048 |
| FFN dimension (df​f​nd\_{ffn}) | 5,120 |
| Expert dimension (de​x​p​e​r​td\_{expert}) | 512 |
| Number of attention heads (nh​e​a​d​sn\_{heads}) | 16 |
| Number of KV heads (nk​v​\_​h​e​a​dn\_{kv\\_head}) | 4 |
| Total experts (EE) | 32 |
| Active experts (EaE\_{a}) | 8 |
| Shared experts (EsE\_{s}) | 1 |
| Total parameters (NN) | 8.6B |
| Active parameters (NaN\_{a}) | 1.43B |

## Appendix B Analysis of the Weight Mixing Proxy

In this appendix, we provide the derivation supporting the claims made in Section 3.3. We formally characterize the discrepancy between the mixed-data training trajectory and the model merging approximation, showing that the error is confined to second-order interactions.

### B.1 Derivation of the Interaction Error

Let Θ0\Theta\_{0} be the initialization. We compare the parameters after TT steps (where the local quadratic approximation holds) under two regimes: data mixing and model merging.

#### Data Mixing Trajectory.

For a data mixture with ratios 𝝀∈ΔK−1\boldsymbol{\lambda}\in\Delta^{K-1}, the total loss is ℒmix​(Θ)=∑k=1Kλk​ℒk​(Θ)\mathcal{L}\_{\text{mix}}(\Theta)=\sum\_{k=1}^{K}\lambda\_{k}\mathcal{L}\_{k}(\Theta). The gradient update at step tt is governed by the Hessian of the entire mixture.
Assuming the learning rate η\eta and training horizon TT are sufficiently small, we can approximate the accumulated update by expanding the gradient ∇ℒmix​(Θt)\nabla\mathcal{L}\_{\text{mix}}(\Theta\_{t}) around Θ0\Theta\_{0}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Θmix\displaystyle\Theta^{\text{mix}} | ≈Θ0−η​∑t=0T−1(∇ℒmix​(Θ0)−t​η​∇2ℒmix​(Θ0)​∇ℒmix​(Θ0))\displaystyle\approx\Theta\_{0}-\eta\sum\_{t=0}^{T-1}\left(\nabla\mathcal{L}\_{\text{mix}}(\Theta\_{0})-t\eta\nabla^{2}\mathcal{L}\_{\text{mix}}(\Theta\_{0})\nabla\mathcal{L}\_{\text{mix}}(\Theta\_{0})\right) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =Θ0−η​T​∇ℒmix​(Θ0)+12​(η​T)2​∇2ℒmix​(Θ0)​∇ℒmix​(Θ0).\displaystyle=\Theta\_{0}-\eta T\nabla\mathcal{L}\_{\text{mix}}(\Theta\_{0})+\frac{1}{2}(\eta T)^{2}\nabla^{2}\mathcal{L}\_{\text{mix}}(\Theta\_{0})\nabla\mathcal{L}\_{\text{mix}}(\Theta\_{0}). |  | (3) |

Substituting ∇ℒmix=∑kλk​gk\nabla\mathcal{L}\_{\text{mix}}=\sum\_{k}\lambda\_{k}g\_{k} and ∇2ℒmix=∑kλk​Hk\nabla^{2}\mathcal{L}\_{\text{mix}}=\sum\_{k}\lambda\_{k}H\_{k}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Θmix≈Θ0−η​T​(∑kλk​gk)+12​(η​T)2​(∑kλk​Hk)​(∑jλj​gj).\Theta^{\text{mix}}\approx\Theta\_{0}-\eta T\left(\sum\_{k}\lambda\_{k}g\_{k}\right)+\frac{1}{2}(\eta T)^{2}\left(\sum\_{k}\lambda\_{k}H\_{k}\right)\left(\sum\_{j}\lambda\_{j}g\_{j}\right). |  | (4) |

Expanding the quadratic term in Eq. ([4](#A2.E4 "Equation 4 ‣ Data Mixing Trajectory. ‣ B.1 Derivation of the Interaction Error ‣ Appendix B Analysis of the Weight Mixing Proxy ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging")) reveals two distinct components:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Quadraticmix=∑kλk2​Hk​gk⏟Self-Interaction+∑k≠jλk​λj​Hk​gj⏟Cross-Interaction.\text{Quadratic}\_{\text{mix}}=\underbrace{\sum\_{k}\lambda\_{k}^{2}H\_{k}g\_{k}}\_{\text{Self-Interaction}}+\underbrace{\sum\_{k\neq j}\lambda\_{k}\lambda\_{j}H\_{k}g\_{j}}\_{\text{Cross-Interaction}}. |  | (5) |

#### Model Merging Trajectory.

For the merging approach, we independently train KK experts. The update for expert kk, trained solely on ℒk\mathcal{L}\_{k}, is approximated as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​Θk≈−η​T​gk+12​(η​T)2​Hk​gk.\Delta\Theta\_{k}\approx-\eta Tg\_{k}+\frac{1}{2}(\eta T)^{2}H\_{k}g\_{k}. |  | (6) |

By merging these experts with weights 𝜶\boldsymbol{\alpha} set equal to the data ratios 𝝀\boldsymbol{\lambda}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Θmerge\displaystyle\Theta\_{\text{merge}} | =Θ0+∑kλk​Δ​Θk\displaystyle=\Theta\_{0}+\sum\_{k}\lambda\_{k}\Delta\Theta\_{k} |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =Θ0−η​T​∑kλk​gk+12​(η​T)2​∑kλk​Hk​gk.\displaystyle=\Theta\_{0}-\eta T\sum\_{k}\lambda\_{k}g\_{k}+\frac{1}{2}(\eta T)^{2}\sum\_{k}\lambda\_{k}H\_{k}g\_{k}. |  | (7) |

#### The Discrepancy Δ\Delta.

Subtracting Eq. ([7](#A2.E7 "Equation 7 ‣ Model Merging Trajectory. ‣ B.1 Derivation of the Interaction Error ‣ Appendix B Analysis of the Weight Mixing Proxy ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging")) from Eq. ([4](#A2.E4 "Equation 4 ‣ Data Mixing Trajectory. ‣ B.1 Derivation of the Interaction Error ‣ Appendix B Analysis of the Weight Mixing Proxy ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging")), the first-order linear terms cancel perfectly. The discrepancy is confined to the second-order terms:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝚫\displaystyle\boldsymbol{\Delta} | :=ΘTmix−Θmerge\displaystyle:=\Theta\_{T}^{\text{mix}}-\Theta\_{\text{merge}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =12​(η​T)2​[(∑kλk2​Hk​gk+∑k≠jλk​λj​Hk​gj)−∑kλk​Hk​gk]\displaystyle=\frac{1}{2}(\eta T)^{2}\left[\left(\sum\_{k}\lambda\_{k}^{2}H\_{k}g\_{k}+\sum\_{k\neq j}\lambda\_{k}\lambda\_{j}H\_{k}g\_{j}\right)-\sum\_{k}\lambda\_{k}H\_{k}g\_{k}\right] |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =12​(η​T)2​[∑k≠jλk​λj​Hk​gj⏟Cross-domain Interference+∑k(λk2−λk)​Hk​gk⏟Self-domain Scaling].\displaystyle=\frac{1}{2}(\eta T)^{2}\left[\underbrace{\sum\_{k\neq j}\lambda\_{k}\lambda\_{j}H\_{k}g\_{j}}\_{\text{Cross-domain Interference}}+\underbrace{\sum\_{k}(\lambda\_{k}^{2}-\lambda\_{k})H\_{k}g\_{k}}\_{\text{Self-domain Scaling}}\right]. |  | (8) |

Equation ([8](#A2.E8 "Equation 8 ‣ The Discrepancy Δ. ‣ B.1 Derivation of the Interaction Error ‣ Appendix B Analysis of the Weight Mixing Proxy ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging")) explicitly characterizes the error introduced by the weight mixing proxy. The error consists of two parts with distinct physical interpretations:

1. 1.

   Cross-domain Interference: The term ∑k≠jλk​λj​Hk​gj\sum\_{k\neq j}\lambda\_{k}\lambda\_{j}H\_{k}g\_{j} represents the distortion of the gradient direction of domain jj by the curvature of domain kk. In data mixing, the optimization trajectory of domain jj is dynamically modulated by the Hessian geometry of domain kk. Model merging, by virtue of independent training, decouples these trajectories, implicitly removing the cross-term curvature effects.
2. 2.

   Self-domain Scaling: The term ∑(λk2−λk)​Hk​gk\sum(\lambda\_{k}^{2}-\lambda\_{k})H\_{k}g\_{k} represents a mismatch in the effective step size for each domain, reflecting curvature-induced saturation.

### B.2 Empirical Observation of Task Orthogonality and Curvature

While we rely on the first-order dominance for ranking mixtures, we provide empirical observations to further characterize the optimization landscape.
To quantify the cross domain interference , we define the relative effective curvature γk​j\gamma\_{kj}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | γk​j=‖Hk​gj‖/‖gj‖‖Hk​gk‖/‖gk‖,\gamma\_{kj}=\frac{\|H\_{k}g\_{j}\|/\|g\_{j}\|}{\|H\_{k}g\_{k}\|/\|g\_{k}\|}, |  | (9) |

where γk​j\gamma\_{kj} measures the normalized curvature response of domain kk induced by the gradient direction of domain jj, relative to its own self-curvature.
We compute the pairwise matrix Mk,j=γk​jM\_{k,j}=\gamma\_{kj} for four distinct mid-training domains using Hessian-vector products (HVP).

As visualized in Figure [7(a)](#A2.F7.sf1 "Figure 7(a) ‣ Figure 7 ‣ B.2 Empirical Observation of Task Orthogonality and Curvature ‣ Appendix B Analysis of the Weight Mixing Proxy ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging"), cross-domain interference is smaller than intra-domain dynamics, resulting in a relatively small approximation error. We also visualize the cosine similarity matrix of the task vectors Δ​Θk=Θk−Θ0\Delta\Theta\_{k}=\Theta\_{k}-\Theta\_{0} in Figure [7(b)](#A2.F7.sf2 "Figure 7(b) ‣ Figure 7 ‣ B.2 Empirical Observation of Task Orthogonality and Curvature ‣ Appendix B Analysis of the Weight Mixing Proxy ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging"), which further illustrates the near-orthogonality of domain-specific updates in parameter space.

!(/html/2601.17858/assets/x9.png)

(a) Relative effective curvature matrix.

!(/html/2601.17858/assets/x10.png)

(b) Task vector cosine similarity.

Figure 7: (a) The matrix exhibits a diagonally dominant structure, where the self-induced curvature consistently outweighs cross-domain interference. (b) The generally low similarity scores indicate that the accumulated parameter updates tend to traverse distinct directions in the parameter space.

## Appendix C Baselines and Computational Cost Analysis

In this section, we provide detailed specifications of the baseline methods compared in our experiments and present a quantitative analysis of their computational costs. Table [6](#A3.T6 "Table 6 ‣ Appendix C Baselines and Computational Cost Analysis ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging") summarizes the resource consumption associated with each strategy.

Algorithm 1  Iterative Human-in-the-loop Optimization (Manual)

1: Input: Initial expert prior 𝝀prior\boldsymbol{\lambda}\_{\text{prior}}, Budget NN trials.

2: 𝒮candidates←{𝝀prior}\mathcal{S}\_{\text{candidates}}\leftarrow\{\boldsymbol{\lambda}\_{\text{prior}}\}

3: for i=1i=1 to NN do

4:  Θi←Train​(𝝀current)\Theta\_{i}\leftarrow\text{Train}(\boldsymbol{\lambda}\_{\text{current}})

5:  s​c​o​r​ei←Evaluate​(Θi)score\_{i}\leftarrow\text{Evaluate}(\Theta\_{i})

6:  Update best score and 𝝀∗\boldsymbol{\lambda}^{\*}

7:  {Human experts adjust ratio based on specific metric drops (e.g., if Math drops, increase λmath\lambda\_{\text{math}})}

8:  𝝀next←HeuristicAdjust​(𝝀current,feedback)\boldsymbol{\lambda}\_{\text{next}}\leftarrow\text{HeuristicAdjust}(\boldsymbol{\lambda}\_{\text{current}},\text{feedback})

9:  Add 𝝀next\boldsymbol{\lambda}\_{\text{next}} to 𝒮candidates\mathcal{S}\_{\text{candidates}}

10: end for

11: Return 𝝀∗\boldsymbol{\lambda}^{\*}

Manual Tuning. This baseline represents the standard high-resource industrial practice. To ensure a strong baseline, we simulate a human-in-the-loop tuning process rather than a random search as shown in Algortithm [1](#alg1 "Algorithm 1 ‣ Appendix C Baselines and Computational Cost Analysis ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging").
Starting from a production-verified prior distribution, we conduct an iterative refinement process involving full-scale training runs (200B tokens each on the 8B model).
In each iteration, human experts analyze the benchmark feedback from the previous run and heuristically adjust the mixing ratios (e.g., increasing the math proportion if math scores stagnate). The final reported score corresponds to the best-performing mixture found throughout this resource-intensive search. In our study, we conduct 10 rounds of iterative refinement. The manual approach, while capable of yielding strong results through extensive experimentation, suffers from inherent limitations: it relies heavily on engineers’ intuition and domain expertise to guide iterative decisions, introducing significant cost and uncertainty. Due to these practical constraints, it is objectively difficult to guarantee that the manually tuned mixture reaches the true optimum.

Adapted RegMix.
The original RegMix (Liu et al., [2025](#bib.bib34)) relies on training numerous small-scale proxy models (1M parameters) to fit a regressor. However, such tiny proxies are ineffective for capturing the emergent capabilities (e.g., complex reasoning) required in mid-training.
To enable an effective comparison and reuse our experimental runs, we adapt RegMix to our setting by training the performance predictor using the outcomes of the 10 full-scale manual runs described above.

CLIMB & Scaling Laws.
For CLIMB (Diao et al., [2025](#bib.bib12)) and Data Scaling Laws (Shukor et al., [2025](#bib.bib51); Ye et al., [2025](#bib.bib70)), due to computational resource constraints, we do not fully reproduce their pipelines. We report the experimental configurations and search budgets in their original papers in Table [6](#A3.T6 "Table 6 ‣ Appendix C Baselines and Computational Cost Analysis ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging").

Table 6: Cost comparison between MergeMix and existing data mixing strategies. Inference overhead is negligible.

| Method | Model Size (NN) | Train Tokens (DD) | Runs | Equivalent Cost (N×D×RunsN\times D\times\text{Runs}) | Relative Cost |
| --- | --- | --- | --- | --- | --- |
| Manual | 8B | 200B | 10 | 16,000 | 100×100\times |
| Adapted RegMix | 8B | 200B | 10 | 16,000 | 100×100\times |
| CLIMB | 350M | 40B | 112 | 1,568 | 9.8×9.8\times |
| Scaling Laws | ∼\sim1B(sum) | 30B | 40 | 1,200 | 7.5×7.5\times |
| MergeMix | 8B | 5B | 4 | 160 | 1.0×\times |

## Appendix D Implementation Details of Model Merging

In this work, model merging is employed in three distinct contexts: (1) merging domain-specific experts to search for optimal data ratios; (2) merging checkpoints during training to simulate learning rate annealing (following the WSM schedule (Tian et al., [2025](#bib.bib60))); and (3) merging the top-16 checkpoints to obtain the final model.
For all these operations, we employ standard element-wise arithmetic averaging (Izmailov et al., [2018](#bib.bib23)) across all model parameters, including the attention layers, MLP layers, expert weights, and the router/gate parameters, without requiring additional alignment steps. Our empirical results confirm that this simple averaging strategy is highly effective in our setting.

## Appendix E Capacity Landscape

!(/html/2601.17858/assets/x11.png)

(a) Total Score

!(/html/2601.17858/assets/x12.png)

(b) Math Score

!(/html/2601.17858/assets/x13.png)

(c) Code Score

!(/html/2601.17858/assets/x14.png)

(d) Knowledge Score

!(/html/2601.17858/assets/x15.png)

(e) Language Score

!(/html/2601.17858/assets/x16.png)

(f) Reasoning Score

Figure 8: Visualizing the performance landscape in the weight space. Since the weights of the four domains sum to 1 (∑wi=1\sum w\_{i}=1), we project the 4D simplex into a 3D view. Warmer colors (red) indicate higher performance, revealing distinct topological optima for different capabilities. To enhance clarity, we visualize only the top 15% of sampled points by performance.

## Appendix F Evaluation Details

Figure [9](#A6.F9 "Figure 9 ‣ Appendix F Evaluation Details ‣ MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging") reports the detailed training dynamics for each benchmark in the main experiment.

!(/html/2601.17858/assets/x17.png)

Figure 9: Detailed training dynamics on each individual benchmark.
