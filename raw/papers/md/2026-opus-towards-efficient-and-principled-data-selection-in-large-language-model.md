---
arxiv: '2602.05400'
authors:
- Shaobo Wang
- Xuan Ouyang
- Tianyi Xu
- Yuzheng Hu
- Jialin Liu
- Guo Chen
- Tianyu Zhang
- Junhao Zheng
- Kexin Yang
- Xingzhang Ren
- Dayiheng Liu
- Linfeng Zhang
parser: ar5iv
retrieved: '2026-05-25'
source: paper
title: 'OPUS: Towards Efficient and Principled Data Selection in Large Language Model
  Pre-training in Every Iteration'
url: https://arxiv.org/abs/2602.05400
year: 2026
---

[2602.05400] OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration



# OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration

Shaobo Wang1,2†   Xuan Ouyang1,3∗   Tianyi Xu1,3∗   Yuzheng Hu4   Jialin Liu1   
  
Guo Chen1   Tianyu Zhang5   Junhao Zheng2   Kexin Yang2   Xingzhang Ren2✉{}^{\text{{\char 0\relax}}}
  
Dayiheng Liu2✉{}^{\text{{\char 0\relax}}}   Linfeng Zhang1✉{}^{\text{{\char 0\relax}}}   
  
Equal contribution. †\daggerWork done while Shaobo Wang ([shaobowang1009@sjtu.edu.cn](mailto:shaobowang1009@sjtu.edu.cn)) was an intern at the Qwen Team, Alibaba. ✉{}^{\text{{\char 0\relax}}} Corresponding authors: Xingzhang Ren ([xingzhang.rxz@alibaba-inc.com](mailto:xingzhang.rxz@alibaba-inc.com)), Dayiheng Liu ([liudayiheng.ldyh@alibaba-inc.com](mailto:liudayiheng.ldyh@alibaba-inc.com)), and Linfeng Zhang ([zhanglinfeng@sjtu.edu.cn](mailto:zhanglinfeng@sjtu.edu.cn))

###### Abstract

As high-quality public text approaches exhaustion, a phenomenon known as the Data Wall (Villalobos et al., [2022](#bib.bib29 "Will we run out of data? limits of llm scaling based on human-generated data")), pre-training is shifting from more tokens to better tokens. However, existing methods either rely on heuristic static filters that ignore training dynamics, or use dynamic yet optimizer-agnostic criteria based on raw gradients. We propose OPUS (Optimizer-induced Projected Utility Selection), a dynamic data selection framework that defines utility in the optimizer-induced update space. OPUS scores candidates by projecting their effective updates, shaped by modern optimizers, onto a target direction derived from a stable, in-distribution proxy. To ensure scalability, we employ Ghost technique with CountSketch for computational efficiency, and Boltzmann sampling for data diversity, incurring only 4.7% additional compute overhead. OPUS achieves remarkable results across diverse corpora, quality tiers, optimizers, and model scales. In pre-training of GPT-2 Large/XL on FineWeb and FineWeb-Edu with 30B tokens, OPUS outperforms industrial-level baselines and even full 200B-token training. Moreover, when combined with industrial-level static filters, OPUS further improves pre-training efficiency, even with lower-quality data. Furthermore, in continued pre-training of Qwen3-8B-Base on SciencePedia, OPUS achieves superior performance using only 0.5B tokens compared to full training with 3B tokens, demonstrating significant data efficiency gains in specialized domains.

1 EPIC Lab, SJTU
   
2 Qwen Team, Alibaba Group
   
3 UW–Madison

4 UIUC
   
5 Mila - Quebec AI Institute

Figure 1: OPUS outperforms random selection by an average of 2.2% accuracy across 10 benchmarks and achieves 8×\times reduction in computation on GPT-XL using FineWeb dataset.

## 1 Introduction

Large language model (LLM) pre-training has entered a critical phase, transitioning from an era of unconstrained data scaling to a regime where the efficiency and quality of every training token are paramount. For the past decade, progress in language modeling has been driven by scaling two primary factors: model size and data volume (Radford et al., [2019](#bib.bib30 "Language models are unsupervised multitask learners"); Brown et al., [2020](#bib.bib31 "Language models are few-shot learners"); Achiam et al., [2023](#bib.bib32 "Gpt-4 technical report"); Yang et al., [2024a](#bib.bib38 "Qwen2 technical report"); [b](#bib.bib39 "Qwen2. 5 technical report"); [2025](#bib.bib40 "Qwen3 technical report"); Guo et al., [2025](#bib.bib33 "DeepSeek-r1 incentivizes reasoning in llms through reinforcement learning"); Liu et al., [2024a](#bib.bib34 "Deepseek-v3 technical report"); Anthropic, [2024](#bib.bib35 "The claude 3 model family: opus, sonnet, haiku")). Scaling laws emphasize that performance is tightly coupled with the efficiency of converting compute into effective training signals (Hoffmann et al., [2022](#bib.bib42 "Training compute-optimal large language models")). Yet the data factor is now saturating: projections suggest that readily available high-quality public text may be exhausted by 2026–2028 (Villalobos et al., [2022](#bib.bib29 "Will we run out of data? limits of llm scaling based on human-generated data")). In this data-wall regime, pre-training must shift from a problem of ingestion capacity to one of control: which tokens should shape the model at this specific optimizer step? When every update consumes scarce tokens, data selection is no longer a pure preprocessing choice but an integral component of the optimization process.

Figure 2: Comparison of different data selection methods.

Existing approaches to this problem present distinct limitations. Static curation methods, such as FineWeb-Edu classifiers (Penedo et al., [2024](#bib.bib8 "The fineweb datasets: decanting the web for the finest text data at scale")) and the DCLM quality classifier (Li et al., [2024](#bib.bib7 "Datacomp-lm: in search of the next generation of training sets for language models")), rely on fixed, training-agnostic heuristics that assume a sample’s utility remains constant as the model evolves. In contrast, prior dynamic selection methods (Wang et al., [2024](#bib.bib2 "Greats: online selection of high-quality data for llm training in every iteration"); [2025a](#bib.bib64 "Data shapley in one training run"); [2025b](#bib.bib66 "Capturing the temporal dependence of training data influence")) score candidates in raw gradient space, implicitly assuming Stochastic Gradient Descent (SGD) dynamics. This induces a fundamental misalignment with modern LLM training, which relies on adaptive optimizers such as AdamW (Loshchilov and Hutter, [2019](#bib.bib41 "Decoupled weight decay regularization")) and Muon (Jordan et al., [2024](#bib.bib3 "Muon: an optimizer for hidden layers in neural networks")) that precondition and reshape the effective update direction. As shown in Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"), existing approaches depart from the optimizer’s actual update geometry, causing unsatisfied optimization trajectory.

To bridge this gap, we introduce OPUS (Optimizer-induced Projected Utility Selection), a framework designed to make data selection in pre-training both principled and scalable. OPUS achieves a principled objective by adapting during training to the model’s evolving needs, unlike static filters, and by defining utility in the optimizer-induced update space. The core insight is that a batch is valuable only insofar as it moves parameters in a direction that improves the model’s performance on a high-quality target distribution, referred to as the proxy, under the optimizer’s specific geometry. OPUS scores each candidate by projecting its optimizer-induced effective update onto the descent direction of this proxy set, eliminating the discrepancy between scoring and training that arises when Adam or Muon training is treated as if it were SGD. To ensure scalability, OPUS estimates these utilities via lightweight projections, avoiding the prohibitive cost of materializing full gradients.

OPUS operationalizes this principle through an objective, an estimator, and a selection rule. First, we formalize utility as the expected one-step improvement on a held-out proxy distribution, measured in the optimizer-induced update geometry, so that scoring aligns with the trajectory induced by AdamW or Muon. Second, we make this objective practical at LLM scale by (i) constructing a stable, in-distribution target direction for the proxy signal and (ii) estimating the required inner products efficiently without materializing per-sample gradients. Third, we use Boltzmann sampling to preserve data diversity. Figure [3](#S1.F3 "Figure 3 ‣ 1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration") summarizes the end-to-end workflow. Our contributions are as follows:

* •

  A principled, optimizer-aware utility for dynamic selection: We introduce optimizer-induced utility as a theoretically grounded objective for dynamic data selection. By deriving closed-form approximations for the effective update directions of AdamW (Loshchilov and Hutter, [2019](#bib.bib41 "Decoupled weight decay regularization")) and Muon (Jordan et al., [2024](#bib.bib3 "Muon: an optimizer for hidden layers in neural networks")), OPUS scores data in the actual optimizer-induced geometry, yielding a model- and optimizer-aware alternative to heuristic filters.
* •

  Stable in-distribution proxy construction: We propose Bench-Proxy, a procedure for constructing a proxy pool by retrieving benchmark-aligned samples directly from the pre-training corpus. This yields a reliable, in-distribution proxy direction that stabilizes utility estimation compared to using raw benchmark validation data.
* •

  Scalable utility estimation via ghost and CountSketch: To make scoring efficient at LLM scale, we avoid per-sample gradient materialization by combining the ghost technique (Wang et al., [2024](#bib.bib2 "Greats: online selection of high-quality data for llm training in every iteration")) with CountSketch projections (Cormode and Muthukrishnan, [2005](#bib.bib18 "An improved data stream summary: the count-min sketch and its applications")), reducing inner products to computations in a low-dimensional space.
* •

  Boltzmann sampling to prevent diversity collapse: To avoid biased or redundant selection induced by greedy top-kk under non-stationary streams, OPUS uses Boltzmann soft sampling with an in-step redundancy penalty.
* •

  OPUS achieves strong empirical gains over industrial baselines: Across from-scratch pre-training of GPT-2 Large/XL on FineWeb and FineWeb-Edu (Penedo et al., [2024](#bib.bib8 "The fineweb datasets: decanting the web for the finest text data at scale")) and continued pre-training of Qwen3-8B-Base (Yang et al., [2025](#bib.bib40 "Qwen3 technical report")) on SciencePedia (SciencePedia Team, [2025](#bib.bib43 "SciencePedia dataset")), OPUS outperforms prior industrial static filters and dynamic selectors with better efficiency.

Figure 3: Overview of OPUS pipeline.

## 2 Related Work

Static pre-training data selection.
Most large-scale LLM pre-training pipelines rely on static corpus filtering, where documents are filtered or reweighted once before training. Representative approaches include classifier- or rule-based filtering over web corpora, exemplified by FineWeb and its educational subset FineWeb-Edu (Penedo et al., [2024](#bib.bib8 "The fineweb datasets: decanting the web for the finest text data at scale")), which document large-scale deduplication and quality filtering choices for Common Crawl derived data. Recent work has also studied more targeted quality signals: QuRating (Wettig et al., [2024](#bib.bib5 "QuRating: selecting high-quality data for training language models")) learns scalar quality ratings from pairwise preferences and shows that balancing quality and diversity improves downstream performance, while DSIR (Xie et al., [2023](#bib.bib6 "Data selection for language models via importance resampling")) formalizes dataset matching via importance resampling in a reduced feature space, enabling scalable selection without human curation. Complementary benchmark and pipeline efforts such as DataComp-LM (DCLM) (Li et al., [2024](#bib.bib7 "Datacomp-lm: in search of the next generation of training sets for language models")) provide standardized corpora and evaluation suites to compare filtering strategies, and UltraFineweb (Wang et al., [2025c](#bib.bib4 "Ultra-fineweb: efficient data filtering and verification for high-quality llm training data")) proposes efficient filtering and verification mechanisms (including lightweight classifier-based pipelines) to further improve web-scale data quality. While effective at removing low-quality noise, these static approaches are inherently training-agnostic: they assume sample utility is time-invariant and do not adapt to the model’s evolving needs across optimization.

Dynamic data selection during pre-training.
To move beyond fixed corpora, dynamic selection chooses samples on-the-fly based on an estimated training utility. Early and widely-used heuristics prioritize samples with large loss or high perplexity, and several works formalize this intuition via online batch selection and importance sampling (Loshchilov and Hutter, [2016](#bib.bib12 "Online batch selection for faster training of neural networks"); Katharopoulos and Fleuret, [2019](#bib.bib13 "Not all samples are created equal: deep learning with importance sampling")). A more rigorous approach uses influence functions (IF) to estimate the impact of training points on validation loss (Koh and Liang, [2017](#bib.bib49 "Understanding black-box predictions via influence functions")). While classic IF methods are computationally intensive and require Hessian inversion, recent approximations have made them more feasible for deep learning. In LLM pre-training, GREATS proposes a principled objective by approximating per-sample validation loss reduction via a Taylor expansion, and then selects a subset each step, typically greedily. It can incur substantial scoring overhead due to per-sample gradient and influence approximations (Wang et al., [2024](#bib.bib2 "Greats: online selection of high-quality data for llm training in every iteration")). More recently, MATES (Yu et al., [2024b](#bib.bib10 "Mates: model-aware data selection for efficient pretraining with data influence models")) learns a lightweight influence model to track evolving data preferences during pre-training, and Group-MATES (Yu et al., [2025](#bib.bib44 "Group-level data selection for efficient pretraining")) emphasizes that utility is not additive and that group-level interactions matter, mitigating redundancy induced by greedy top-kk selection. In parallel, perplexity-based pruning remains a competitive, simple signal for data selection and pruning, including settings where a small reference model computes PPL to prune large-scale corpora (Ankner et al., [2025](#bib.bib11 "Perplexed by perplexity: perplexity-based data pruning with small reference models")). OPUS fits this dynamic-selection family, but differs by aligning utility with the optimizer-induced update and by using efficient projected scoring with soft sampling.

Influence-function scores and data-attribution.
A large line of work studies training-data influence and attribution (Hammoudeh and Lowd, [2024](#bib.bib61 "Training data influence analysis and estimation: a survey"); Deng et al., [2025](#bib.bib60 "A Survey of Data Attribution: Methods, Applications, and Evaluation in the Era of Generative AI"))—estimating how individual samples affect model behavior or validation loss. Classical influence functions approximate the effect of upweighting a training point via Hessian-based sensitivity analysis, enabling fine-grained data attribution without retraining (Koh and Liang, [2017](#bib.bib49 "Understanding black-box predictions via influence functions")). To make influence estimation practical in deep, non-convex settings, some works replace exact second-order IF computation with scalable surrogates (Pruthi et al., [2020](#bib.bib50 "Estimating training data influence by tracing gradient descent"); Guo et al., [2021](#bib.bib51 "Fastif: scalable influence functions for efficient model interpretation and debugging"); Yeh et al., [2018](#bib.bib52 "Representer point selection for explaining deep neural networks")). Related directions also develop first-order or early-training proxies for data importance, such as selecting informative subsets early in training (Paul et al., [2021](#bib.bib54 "Deep learning on a data diet: finding important examples early in training")), leveraging forgetting events to identify noisy or hard-to-learn samples (Toneva et al., [2019](#bib.bib55 "An empirical study of example forgetting during deep neural network learning")), and optimizing subset selection via gradient-matching (Killamsetty et al., [2021](#bib.bib56 "Grad-match: gradient matching based data subset selection for efficient deep model training")) or influence functions (Hu et al., [2024](#bib.bib65 "Most influential subset selection: challenges, promises, and beyond")). Another line of research explores Shapley value, a concept from cooperative game theory, to quantify the value of data (Ghorbani and Zou, [2019](#bib.bib53 "Data shapley: equitable valuation of data for machine learning"); Jia et al., [2021](#bib.bib63 "Scalability vs. utility: do we have to sacrifice one for the other in data importance quantification?"); Wang et al., [2025a](#bib.bib64 "Data shapley in one training run")). Recently, influence and data-attribution signals have been adapted from classical IF literature to practical data selection for large language models, including LoRA-aware influence approximations and gradient-datastore based retrieval (Xia et al., [2024](#bib.bib45 "LESS: selecting influential data for targeted instruction tuning")), as well as more structured selection pipelines that optimize selection objectives for instruction tuning (Du et al., [2023](#bib.bib58 "Mods: model-oriented data selection for instruction tuning"); Liu et al., [2024b](#bib.bib59 "What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning")). Moreover, many approaches implicitly operate in raw-gradient geometry and/or employ deterministic top-kk retrieval, which can become brittle under rapidly changing training dynamics and optimizer-induced transformations. These limitations motivate online selection objectives that remain faithful to the effective optimizer update while preserving scalability and diversity.

## 3 Background

### 3.1 LLM Pre-training

We consider an autoregressive language model fθf\_{\theta} parameterized by θ∈ℝd\theta\in\mathbb{R}^{d}. A training sample is a token sequence z=(x1,…,xL)z=(x\_{1},\dots,x\_{L}) with xi∈𝒱x\_{i}\in\mathcal{V}, where 𝒱\mathcal{V} is the vocabulary and LL is the sequence length. The model defines the next-token distribution pθ​(xi∣x<i)p\_{\theta}(x\_{i}\mid x\_{<i}), and the per-sequence loss is the negative log-likelihood: ℒ​(z;θ)=−1L​∑i=1Llog⁡pθ​(xi∣x<i).\mathcal{L}(z;\theta)=-\frac{1}{L}\sum\_{i=1}^{L}\log p\_{\theta}(x\_{i}\mid x\_{<i}). For any distribution (or finite set) 𝒬\mathcal{Q} over sequences, we define the expected loss
ℒ​(𝒬;θ):=𝔼z∼𝒬​[ℒ​(z;θ)]\mathcal{L}(\mathcal{Q};\theta):=\mathbb{E}\_{z\sim\mathcal{Q}}[\mathcal{L}(z;\theta)] (or its empirical average for a finite 𝒬\mathcal{Q}).
Let 𝒟\mathcal{D} denote the full pre-training corpus. We partition it into (i) a training set 𝒟tr\mathcal{D}\_{\text{tr}} used for parameter updates and (ii) a held-out validation set 𝒟val\mathcal{D}\_{\text{val}} used only to guide selection. Importantly, 𝒟val∩𝒟tr=∅\mathcal{D}\_{\text{val}}\cap\mathcal{D}\_{\text{tr}}=\emptyset, so validation samples never appear in training updates.

### 3.2 Data Selection in Pre-training

Data selection in pre-training aims to choose samples that compress knowledge both efficiently and effectively, which can be categorized into two domains.

Static Data Selection.
Static methods operate offline, filtering the entire candidate pool 𝒟tr\mathcal{D}\_{\text{tr}} before training begins. A scoring function S​(z)S(z) assigns a quality score to each sample z∈𝒟trz\in\mathcal{D}\_{\text{tr}}. A subset 𝒟selected⊂𝒟tr\mathcal{D}\_{\text{selected}}\subset\mathcal{D}\_{\text{tr}} is retained by thresholding or top-kk selection: 𝒟selected={z∈𝒟tr∣S​(z)≥threshold}.\mathcal{D}\_{\text{selected}}=\{z\in\mathcal{D}\_{\text{tr}}\mid S(z)\geq\text{threshold}\}. The model is then trained on 𝒟selected\mathcal{D}\_{\text{selected}} using a standard optimizer. While scalable, static selection ignores the model’s evolving state θt\theta\_{t} during training.

Dynamic Data Selection.
Dynamic methods select data during training at each step tt, adapting to the current model parameter θt\theta\_{t} and optimizer state. At step tt, the algorithm receives a candidate buffer ℬt={z1,…,zN}\mathcal{B}\_{t}=\{z\_{1},\dots,z\_{N}\} of NN sequences from the update stream 𝒟tr\mathcal{D}\_{\text{tr}}. It selects a subset ℬ^t⊂ℬt\widehat{\mathcal{B}}\_{t}\subset\mathcal{B}\_{t} of size K=⌊ρ​N⌋K=\lfloor\rho N\rfloor (selection ratio ρ∈(0,1]\rho\in(0,1]) to update the model, *i.e.*, ℬ^t=Select​(ℬt;st​(⋅),K),\widehat{\mathcal{B}}\_{t}=\textsc{Select}\big(\mathcal{B}\_{t};s\_{t}(\cdot),K\big), where st​(z)s\_{t}(z) is a step-dependent score (or sampling distribution) computed from the current model and proxy signal.

### 3.3 Modern Optimizers in Large-Scale Pre-training

Many dynamic selection methods score candidates using the raw gradient ∇ℒ​(z;θt)\nabla\mathcal{L}(z;\theta\_{t}), implicitly assuming SGD-like geometry. Modern LLM training instead uses optimizers that transform gradients using state, such as momentum and adaptive preconditioning, changing the effective update direction.
We write the optimizer-induced effective update at step tt using an optimizer-induced preconditioner (operator) 𝐏t\mathbf{P}\_{t} applied to per-sample gradients:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​θt​(ℬ^t)=−ηt​∑z∈ℬ^t𝐏t​∇ℒ​(z;θt).\Delta\theta\_{t}(\widehat{\mathcal{B}}\_{t})=-\eta\_{t}\sum\_{z\in\widehat{\mathcal{B}}\_{t}}\mathbf{P}\_{t}\nabla\mathcal{L}(z;\theta\_{t}). |  | (1) |

Here, 𝐏t\mathbf{P}\_{t} encapsulates the optimizer state at step tt and induces the geometry that the training trajectory actually follows. When the optimizer’s transformation is not strictly linear, 𝐏t\mathbf{P}\_{t} should be read as a state-dependent operator acting on the gradient. This motivates defining selection scores in the optimizer-induced geometry rather than raw-gradient space. The details of common optimizers (SGD, AdamW, and Muon) are attached in Section [4](#S4 "4 Optimizer-induced Preconditioners ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").

## 4 Optimizer-induced Preconditioners

### 4.1 Stochastic gradient descent

We include SGD as a minimal reference point, since many prior dynamic selection methods implicitly assume an SGD-like update geometry and score candidates directly using raw gradients. In SGD, the optimizer applies a uniform scalar learning rate (and optional weight decay) without stateful preconditioning, so the effective update direction is aligned with the mini-batch gradient. Consequently, at a fixed step tt, SGD induces an (approximately) identity update geometry, 𝐏t≈𝐈\mathbf{P}\_{t}\approx\mathbf{I}, making raw-gradient similarity a natural scoring signal.

SGD 
Stochastic gradient descent updates parameters by moving along the negative mini-batch gradient:



𝐠t=∇θℒ​(ℬt;θt),Δ​θt=−ηt​𝐠t.\mathbf{g}\_{t}=\nabla\_{\theta}\mathcal{L}(\mathcal{B}\_{t};\theta\_{t}),\qquad\Delta\theta\_{t}=-\eta\_{t}\mathbf{g}\_{t}.
With optional weight decay, the one-step update becomes



Δ​θt=−ηt​(𝐠t+λ​θt).\Delta\theta\_{t}=-\eta\_{t}\big(\mathbf{g}\_{t}+\lambda\theta\_{t}\big).
For online scoring at a fixed step tt, SGD induces an identity update geometry
𝐏t≈𝐈\mathbf{P}\_{t}\approx\mathbf{I}, so utility is naturally measured in raw-gradient space.

### 4.2 Muon preconditioner

We derive the Muon-instantiated preconditioner by linearizing Muon’s one-step *lookahead* update
at a fixed training step tt (the regime used for online selection). Consider a linear weight matrix
Wℒ∈ℝo×iW\_{\mathcal{L}}\in\mathbb{R}^{o\times i} updated by Muon. Ignoring bias-corrections for exposition, Muon maintains
an EMA momentum on the (mini-batch) gradient
𝐠t,ℒ​(S):=1|S|​∑z∈S∇Wℒℒ​(z;θt)\mathbf{g}\_{t,\mathcal{L}}(S):=\frac{1}{|S|}\sum\_{z\in S}\nabla\_{W\_{\mathcal{L}}}\mathcal{L}(z;\theta\_{t}):

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐦t+1,ℒ​(S)=μ​𝐦t,ℒ+(1−μ)​𝐠t,ℒ​(S).\mathbf{m}\_{t+1,\mathcal{L}}(S)=\mu\mathbf{m}\_{t,\mathcal{L}}+(1-\mu)\mathbf{g}\_{t,\mathcal{L}}(S). |  | (2) |

In practice, Muon forms a “double-smoothed” direction fed to the orthogonalizer,

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐪t+1,ℒ​(S):=(1−μ)​𝐠t,ℒ​(S)+μ​𝐦t+1,ℒ​(S)=μ2​𝐦t,ℒ​(S)+(1−μ2)​𝐠t,ℒ​(S).\mathbf{q}\_{t+1,\mathcal{L}}(S):=(1-\mu)\mathbf{g}\_{t,\mathcal{L}}(S)+\mu\mathbf{m}\_{t+1,\mathcal{L}}(S)=\mu^{2}\mathbf{m}\_{t,\mathcal{L}}(S)+(1-\mu^{2})\mathbf{g}\_{t,\mathcal{L}}(S). |  | (3) |

and takes the parameter step

|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​Wt,ℒ​(S):=Wt+1,ℒ​(S)−Wt,ℒ=−ηt​𝒪t,ℒ​(𝐪t+1,ℒ​(S)).\Delta W\_{t,\mathcal{L}}(S):=W\_{t+1,\mathcal{L}}(S)-W\_{t,\mathcal{L}}=-\eta\_{t}\mathcal{O}\_{t,\mathcal{L}}\!\big(\mathbf{q}\_{t+1,\mathcal{L}}(S)\big). |  | (4) |

Online-selection view.
For scoring at fixed step tt, we hold Muon’s state fixed (learning rate ηt\eta\_{t}, momentum coefficient μ\mu,
and the history buffer 𝐦t,ℒ\mathbf{m}\_{t,\mathcal{L}}). Moreover, we *freeze* the Newton–Schulz (NS) operator
during selection by constructing it from a reference direction 𝐪¯t,ℒ\bar{\mathbf{q}}\_{t,\mathcal{L}} available at the
start of step tt (e.g., from the current optimizer buffer / proxy batch), and reuse it for all candidates.
Under this approximation, NS induces an approximately linear left-multiplication map

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒪t,ℒ​(Z)≈𝐒t,ℒ​Z,𝐒t,ℒ=a​𝐈+b​𝐀t,ℒ+c​𝐀t,ℒ2,𝐀t,ℒ:=𝐪¯~t,ℒ​𝐪¯~t,ℒ⊤.\mathcal{O}\_{t,\mathcal{L}}(Z)\approx\mathbf{S}\_{t,\mathcal{L}}Z,\;\mathbf{S}\_{t,\mathcal{L}}=a\mathbf{I}+b\mathbf{A}\_{t,\mathcal{L}}+c\mathbf{A}\_{t,\mathcal{L}}^{2},\;\mathbf{A}\_{t,\mathcal{L}}:=\tilde{\bar{\mathbf{q}}}\_{t,\mathcal{L}}\tilde{\bar{\mathbf{q}}}\_{t,\mathcal{L}}^{\top}. |  | (5) |

where 𝐪¯~t,ℒ:=𝐪¯t,ℒ/‖𝐪¯t,ℒ‖F\widetilde{\bar{\mathbf{q}}}\_{t,\mathcal{L}}:=\bar{\mathbf{q}}\_{t,\mathcal{L}}/\|\bar{\mathbf{q}}\_{t,\mathcal{L}}\|\_{F}
(and a,b,ca,b,c are fixed NS polynomial coefficients). Substituting ([3](#S4.E3 "In 4.2 Muon preconditioner ‣ 4 Optimizer-induced Preconditioners ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")) into ([4](#S4.E4 "In 4.2 Muon preconditioner ‣ 4 Optimizer-induced Preconditioners ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"))
and using ([5](#S4.E5 "In 4.2 Muon preconditioner ‣ 4 Optimizer-induced Preconditioners ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")) yields the linearized lookahead update

|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​Wt,ℒ​(S)≈𝐛t,ℒ−κt​𝐒t,ℒ​𝐠t,ℒ​(S),𝐛t,ℒ:=−ηt​μ2​𝐒t,ℒ​𝐦t,ℒ,κt:=ηt​(1−μ2).\Delta W\_{t,\mathcal{L}}(S)\approx\mathbf{b}\_{t,\mathcal{L}}-\kappa\_{t}\mathbf{S}\_{t,\mathcal{L}}\mathbf{g}\_{t,\mathcal{L}}(S),\;\mathbf{b}\_{t,\mathcal{L}}:=-\eta\_{t}\mu^{2}\mathbf{S}\_{t,\mathcal{L}}\mathbf{m}\_{t,\mathcal{L}},\;\kappa\_{t}:=\eta\_{t}(1-\mu^{2}). |  | (6) |

Since OPUS ranks candidates/subsets by *relative* utility at fixed tt, the SS-independent shift can be
dropped for scoring purposes, and the effective data-dependent update is captured by a layerwise preconditioner

|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​Wt,ℒ​(S)≈−𝐏t,ℒMuon​𝐠t,ℒ​(S)+const,𝐏t,ℒMuon:=κt​𝐒t,ℒ.\Delta W\_{t,\mathcal{L}}(S)\approx-\mathbf{P}^{\mathrm{Muon}}\_{t,\mathcal{L}}\mathbf{g}\_{t,\mathcal{L}}(S)+\mathrm{const},\qquad\mathbf{P}^{\mathrm{Muon}}\_{t,\mathcal{L}}:=\kappa\_{t}\mathbf{S}\_{t,\mathcal{L}}. |  | (7) |

Thus, Muon induces a *dense, sample-independent* (at fixed tt under frozen 𝐒t,ℒ\mathbf{S}\_{t,\mathcal{L}})
left-preconditioner that reshapes gradient directions before scoring; OPUS remains optimizer-agnostic by
plugging 𝐏t,ℒMuon\mathbf{P}^{\mathrm{Muon}}\_{t,\mathcal{L}} into the same utility machinery used for AdamW.

Muon 
Muon targets matrix-shaped parameters W∈ℝo×iW\in\mathbb{R}^{o\times i} by maintaining an accumulated
matrix direction and applying a Newton–Schulz orthogonalization (matrix-sign style) transform:



𝐌t\displaystyle\mathbf{M}\_{t}
=μ​𝐌t−1+(1−μ)​𝐠t,\displaystyle=\mu\mathbf{M}\_{t-1}+(1-\mu)\mathbf{g}\_{t},


𝐐t\displaystyle\mathbf{Q}\_{t}
:=NewtonSchulz​(𝐌t),\displaystyle:=\mathrm{NewtonSchulz}(\mathbf{M}\_{t}),


Δ​Wt\displaystyle\Delta W\_{t}
∝−𝐐t.\displaystyle\propto-\mathbf{Q}\_{t}.
For online selection at fixed step tt, we hold the optimizer state and freeze the Newton–Schulz
operator across candidates, yielding an approximately linear map
NewtonSchulz​(Z)≈𝐒t​Z\mathrm{NewtonSchulz}(Z)\approx\mathbf{S}\_{t}Z.
This induces a dense, layerwise preconditioner 𝐏t\mathbf{P}\_{t} that reshapes update geometry
beyond raw-gradient space.

### 4.3 AdamW preconditioner

We derive the AdamW-instantiated preconditioner by linearizing the one-step *lookahead* update
that OPUS uses to score candidate subsets.
Consider the (decoupled) AdamW update applied to a subset SS at iteration tt:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝐦t​(S)\displaystyle\mathbf{m}\_{t}(S) | =β1​𝐦t−1+(1−β1)​𝐠t​(S),𝐯t​(S)=β2​𝐯t−1+(1−β2)​𝐠t​(S)⊙2,\displaystyle=\beta\_{1}\mathbf{m}\_{t-1}+(1-\beta\_{1})\mathbf{g}\_{t}(S),\qquad\mathbf{v}\_{t}(S)=\beta\_{2}\mathbf{v}\_{t-1}+(1-\beta\_{2})\mathbf{g}\_{t}(S)^{\odot 2}, |  | (8) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝐦^t​(S)\displaystyle\widehat{\mathbf{m}}\_{t}(S) | =𝐦t​(S)1−β1t,𝐯^t​(S)=𝐯t​(S)1−β2t,θt+1​(S)=θt−αt​𝐦^t​(S)𝐯^t​(S)+ϵ−αt​λ​θt.\displaystyle=\frac{\mathbf{m}\_{t}(S)}{1-\beta\_{1}^{t}},\qquad\widehat{\mathbf{v}}\_{t}(S)=\frac{\mathbf{v}\_{t}(S)}{1-\beta\_{2}^{t}},\qquad\theta\_{t+1}(S)=\theta\_{t}-\alpha\_{t}\frac{\widehat{\mathbf{m}}\_{t}(S)}{\sqrt{\widehat{\mathbf{v}}\_{t}(S)}+\epsilon}-\alpha\_{t}\lambda\theta\_{t}. |  | (9) |

where 𝐠t​(S):=1|S|​∑z∈S∇θℒ​(z;θt)\mathbf{g}\_{t}(S):=\frac{1}{|S|}\sum\_{z\in S}\nabla\_{\theta}\mathcal{L}(z;\theta\_{t}) and ⊙\odot denotes elementwise operations.

Online-selection view.
At a fixed training step tt, OPUS compares subsets SS via their *relative* utility under a one-step
lookahead while *holding the optimizer state fixed at the start of step tt*.
Concretely, we treat αt,β1,β2,ϵ,λ\alpha\_{t},\beta\_{1},\beta\_{2},\epsilon,\lambda and the history buffers
(𝐦t−1,𝐯t−1)(\mathbf{m}\_{t-1},\mathbf{v}\_{t-1}) as constants with respect to SS.

Affine dependence on the batch gradient.
Under this view, the bias-corrected first moment is affine in 𝐠t​(S)\mathbf{g}\_{t}(S):

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐦^t​(S)=β11−β1t​𝐦t−1+1−β11−β1t​𝐠t​(S).\widehat{\mathbf{m}}\_{t}(S)=\frac{\beta\_{1}}{1-\beta\_{1}^{t}}\mathbf{m}\_{t-1}+\frac{1-\beta\_{1}}{1-\beta\_{1}^{t}}\mathbf{g}\_{t}(S). |  | (10) |

Frozen preconditioner approximation.
To keep scoring tractable, we freeze the RMS geometry during selection by
dropping the SS-dependence in the second moment update. Using
𝐯^t​(S)=𝐯t​(S)/(1−β2t)\widehat{\mathbf{v}}\_{t}(S)=\mathbf{v}\_{t}(S)/(1-\beta\_{2}^{t}) with
𝐯t​(S)=β2​𝐯t−1+(1−β2)​𝐠t​(S)⊙2\mathbf{v}\_{t}(S)=\beta\_{2}\mathbf{v}\_{t-1}+(1-\beta\_{2})\mathbf{g}\_{t}(S)^{\odot 2}, we approximate

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐯^t​(S)+ϵ=β2​𝐯t−1+(1−β2)​𝐠t​(S)⊙21−β2t+ϵ≈𝐯¯t+ϵ,𝐯¯t:=β2​𝐯t−11−β2t.\sqrt{\widehat{\mathbf{v}}\_{t}(S)}+\epsilon=\sqrt{\frac{\beta\_{2}\mathbf{v}\_{t-1}+(1-\beta\_{2})\mathbf{g}\_{t}(S)^{\odot 2}}{1-\beta\_{2}^{t}}}+\epsilon\approx\sqrt{\overline{\mathbf{v}}\_{t}}+\epsilon,\qquad\overline{\mathbf{v}}\_{t}:=\frac{\beta\_{2}\mathbf{v}\_{t-1}}{1-\beta\_{2}^{t}}. |  | (11) |

Substituting ([10](#S4.E10 "In 4.3 AdamW preconditioner ‣ 4 Optimizer-induced Preconditioners ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")) and ([11](#S4.E11 "In 4.3 AdamW preconditioner ‣ 4 Optimizer-induced Preconditioners ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")) into ([9](#S4.E9 "In 4.3 AdamW preconditioner ‣ 4 Optimizer-induced Preconditioners ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")) yields the linearized form.
Let 𝐃t:=Diag⁡(1𝐯^t−1+ϵ)\mathbf{D}\_{t}:=\operatorname{Diag}\!\left(\frac{1}{\sqrt{\widehat{\mathbf{v}}\_{t-1}}+\epsilon}\right),
At:=αt​β11−β1tA\_{t}:=\alpha\_{t}\frac{\beta\_{1}}{1-\beta\_{1}^{t}}, and
Ct:=αt​1−β11−β1tC\_{t}:=\alpha\_{t}\frac{1-\beta\_{1}}{1-\beta\_{1}^{t}}.
Then we have:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​θt​(S):=θt+1​(S)−θt≈−At​𝐃t​𝐦t−1−αt​λ​θt⏟independent of ​S−Ct​𝐃t​𝐠t​(S).\Delta\theta\_{t}(S):=\theta\_{t+1}(S)-\theta\_{t}\approx\underbrace{-A\_{t}\mathbf{D}\_{t}\mathbf{m}\_{t-1}-\alpha\_{t}\lambda\theta\_{t}}\_{\text{independent of }S}-C\_{t}\mathbf{D}\_{t}\mathbf{g}\_{t}(S). |  | (12) |

Since OPUS ranks subsets by *relative* utility at fixed step tt, the SS-independent shift contributes
an additive constant to the (first-order) utility term and does not affect ranking.
Therefore, the effective *data-dependent* update can be written as

|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​θt​(S)≈−𝐏tAdamW​𝐠t​(S)+const,𝐏tAdamW:=Ct​Diag⁡(1𝐯^t−1+ϵ),Ct:=αt​1−β11−β1t.\Delta\theta\_{t}(S)\approx-\mathbf{P}\_{t}^{\mathrm{AdamW}}\mathbf{g}\_{t}(S)+\mathrm{const},\quad\mathbf{P}\_{t}^{\mathrm{AdamW}}:=C\_{t}\operatorname{Diag}\!\Big(\tfrac{1}{\sqrt{\widehat{\mathbf{v}}\_{t-1}}+\epsilon}\Big),\quad C\_{t}:=\alpha\_{t}\tfrac{1-\beta\_{1}}{1-\beta\_{1}^{t}}. |  | (13) |

AdamW 
AdamW maintains exponential moving averages of the gradient and its elementwise square:



𝐦t\displaystyle\mathbf{m}\_{t}
=β1​𝐦t−1+(1−β1)​𝐠t,𝐦^t=𝐦t/(1−β1t),\displaystyle=\beta\_{1}\mathbf{m}\_{t-1}+(1-\beta\_{1})\mathbf{g}\_{t},\qquad\widehat{\mathbf{m}}\_{t}=\mathbf{m}\_{t}/(1-\beta\_{1}^{t}),


𝐯t\displaystyle\mathbf{v}\_{t}
=β2​𝐯t−1+(1−β2)​𝐠t⊙2,𝐯^t=𝐯t/(1−β2t).\displaystyle=\beta\_{2}\mathbf{v}\_{t-1}+(1-\beta\_{2})\mathbf{g}\_{t}^{\odot 2},\qquad\widehat{\mathbf{v}}\_{t}=\mathbf{v}\_{t}/(1-\beta\_{2}^{t}).
With decoupled weight decay, the one-step update is



Δ​θt=−αt​𝐦^t𝐯^t+ϵ−αt​λ​θt.\Delta\theta\_{t}=-\alpha\_{t}\frac{\widehat{\mathbf{m}}\_{t}}{\sqrt{\widehat{\mathbf{v}}\_{t}}+\epsilon}-\alpha\_{t}\lambda\theta\_{t}.
For online scoring at a fixed step tt, we freeze the RMS geometry and obtain an approximate
diagonal preconditioner 𝐏t≈αt​Diag​((𝐯^t−1+ϵ)−1)\mathbf{P}\_{t}\approx\alpha\_{t}\mathrm{Diag}\!\big((\sqrt{\widehat{\mathbf{v}}\_{t-1}}+\epsilon)^{-1}\big)
that rescales coordinates before measuring utility.

## 5 Methodology: OPUS

We now describe OPUS and organize the section around the requirements that dynamic selection must satisfy in large-scale pre-training. Ideally, dynamic selection in large-scale pre-training should satisfy three desiderata:

* •

  *Principled:* scores are derived from an explicit objective that measures improvement on a held-out proxy distribution under the optimizer-induced update geometry.
* •

  *Efficient:* scoring avoids materializing per-sample gradients in high-dimensional space.
* •

  *Scalable:* overhead remains modest as model dimension mm grows, enabling selection at every step.

Guided by these desiderata, we introduce OPUS, a dynamic data selection framework for LLM pre-training. At each step tt, OPUS receives a candidate buffer ℬt={z1,…,zN}⊂𝒟tr\mathcal{B}\_{t}=\{z\_{1},\dots,z\_{N}\}\subset\mathcal{D}\_{\text{tr}} and selects K=⌊ρ​N⌋K=\lfloor\rho N\rfloor sequences to form the update batch. OPUS also draws a proxy mini-batch of size KproxyK\_{\text{proxy}} from a proxy pool 𝒟proxy\mathcal{D}\_{\text{proxy}}, a finite surrogate for the held-out proxy set 𝒟val\mathcal{D}\_{\text{val}}. Let 𝐏t\mathbf{P}\_{t} denote the optimizer-induced preconditioner at step tt. We use sketch dimension mm for scoring in a projected space and temperature τ>0\tau>0 for stochastic sampling. For details, please refer Algorithm [1](#alg1 "In 5 Methodology: OPUS ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration") for the iterative OPUS algorithm.

1: Input: Model fθf\_{\theta}; Training Data stream 𝒟tr\mathcal{D}\_{\text{tr}}; Proxy pool 𝒟proxy\mathcal{D}\_{\text{proxy}}; Optimizer 𝒪\mathcal{O}; Selection ratio ρ\rho; Projection dim mm.

2: Initialize: Implicit sketch operator Π\Pi using CountSketch with hash h:[d]→[m]h:[d]\to[m] and sign s:[d]→{−1,+1}s:[d]\to\{-1,+1\}.

3: for t=0,1,…t=0,1,\dots do

4:  1. Batch Sampling: Read candidate buffer ℬt={z1,…,zN}\mathcal{B}\_{t}=\{z\_{1},\dots,z\_{N}\} from 𝒟tr\mathcal{D}\_{\text{tr}}.

5:  2. Preconditioner Computation: Construct optimizer-induced preconditioner 𝐏t=𝐏​(𝒪t)\mathbf{P}\_{t}=\mathbf{P}(\mathcal{O}\_{t}) from 𝒪\mathcal{O}’s state at step tt.

6:  3. Proxy Feature Generation: Sample KproxyK\_{\text{proxy}} samples {z~k}\{\tilde{z}\_{k}\} from 𝒟proxy\mathcal{D}\_{\text{proxy}}, obtain ghost factors {𝐚r(z~k),𝐛r(z~k)}\{\mathbf{a}^{(\tilde{z}\_{k})}\_{r},\mathbf{b}^{(\tilde{z}\_{k})}\_{r}\}, and compute
per-layer proxy sketches ψproxy(t,r)←Πr​(1Kproxy​∑k=1Kproxy𝐚r(z~k)⊗𝐛r(z~k))\psi\_{\text{proxy}}^{(t,r)}\leftarrow\Pi\_{r}\Big(\frac{1}{K\_{\text{proxy}}}\sum\_{k=1}^{K\_{\text{proxy}}}\mathbf{a}^{(\tilde{z}\_{k})}\_{r}\otimes\mathbf{b}^{(\tilde{z}\_{k})}\_{r}\Big)
for all r∈ℛr\in\mathcal{R}.

7:  4. Candidate Feature Generation:
Compute per-layer sketches ϕ(t,r)​(z)∈ℝm\phi^{(t,r)}(z)\in\mathbb{R}^{m} implicitly from ghost factors {𝐚r(z),𝐛r(z)}r∈ℛ\{\mathbf{a}^{(z)}\_{r},\mathbf{b}^{(z)}\_{r}\}\_{r\in\mathcal{R}}:

|  |  |  |
| --- | --- | --- |
|  | ϕ(t,r)​(z)←Πr​(𝐏t,r​(𝐚r(z)⊗𝐛r(z))),∀r∈ℛ.\phi^{(t,r)}(z)\leftarrow\Pi\_{r}\Big(\mathbf{P}\_{t,r}\big(\mathbf{a}^{(z)}\_{r}\otimes\mathbf{b}^{(z)}\_{r}\big)\Big),\quad\forall r\in\mathcal{R}. |  |

8:  5. Soft Sampling Loop:

9:  Let target batch size K=⌊ρ​N⌋K=\lfloor\rho N\rfloor, Selected set ℬ^t←∅\widehat{\mathcal{B}}\_{t}\leftarrow\emptyset, and per-layer history Φ(t,r)←𝟎\Phi^{(t,r)}\leftarrow\mathbf{0} for all r∈ℛr\in\mathcal{R}.

10:  for j=1j=1 to KK do

11:   For each z∈ℬt∖ℬ^tz\in\mathcal{B}\_{t}\setminus\widehat{\mathcal{B}}\_{t}, compute Uz(t)U\_{z}^{(t)}:

|  |  |  |
| --- | --- | --- |
|  | Uz(t)←ηt​∑r∈ℛ⟨ϕ(t,r)​(z),ψproxy(t,r)⟩−ηt2​∑r∈ℛ⟨ϕ(t,r)​(z),Φ(t,r)⟩\displaystyle U\_{z}^{(t)}\leftarrow\eta\_{t}\sum\_{r\in\mathcal{R}}\langle\phi^{(t,r)}(z),\psi\_{\text{proxy}}^{(t,r)}\rangle-\eta\_{t}^{2}\sum\_{r\in\mathcal{R}}\langle\phi^{(t,r)}(z),\Phi^{(t,r)}\rangle |  |

12:   Sample index z∗z^{\*} via Softmax: pt​(z∗)∝exp⁡(Uz(t)/τ)p\_{t}({z^{\*}})\propto\exp(U\_{z}^{(t)}/\tau).

13:   Add to batch: ℬ^t←ℬ^t∪{z∗}\widehat{\mathcal{B}}\_{t}\leftarrow\widehat{\mathcal{B}}\_{t}\cup\{z^{\*}\}.

14:   Update history (redundancy): Φ(t,r)←Φ(t,r)+ϕ(t,r)​(z∗)\Phi^{(t,r)}\leftarrow\Phi^{(t,r)}+\phi^{(t,r)}(z^{\*}) for all r∈ℛr\in\mathcal{R}.

15:  end for

16:  6. Update: Train θt+1\theta\_{t+1} using batch ℬ^t\widehat{\mathcal{B}}\_{t} with optimizer 𝒪\mathcal{O}.

17: end for

Algorithm 1 OPUS: Optimizer-induced Projected Utility Selection

### 5.1 Optimizer-Induced Utility Objective

To obtain a principled scoring signal for selection, we define the utility of a candidate batch 𝒮\mathcal{S} as the reduction in loss on validation set 𝒟val\mathcal{D}\_{\text{val}} after one optimization step. Following (Wang et al., [2024](#bib.bib2 "Greats: online selection of high-quality data for llm training in every iteration")), we define utility at step tt as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | U(t)​(𝒮):=ℒ​(𝒟val;θt)−ℒ​(𝒟val;θt+1​(𝒮)).U^{(t)}(\mathcal{S}):=\mathcal{L}(\mathcal{D}\_{\text{val}};\theta\_{t})-\mathcal{L}(\mathcal{D}\_{\text{val}};\theta\_{t+1}(\mathcal{S})). |  | (14) |

Marginal gain.
At each training step tt, we are given a candidate buffer ℬt\mathcal{B}\_{t}
and aim to construct an update subset ℬ^t⊆ℬt\widehat{\mathcal{B}}\_{t}\subseteq\mathcal{B}\_{t}.
Let z∈ℬt∖ℬ^tz\in\mathcal{B}\_{t}\setminus\widehat{\mathcal{B}}\_{t} be a remaining candidate.
We define the marginal utility of adding zz as

|  |  |  |  |
| --- | --- | --- | --- |
|  | Uz(t):=U(t)​(ℬ^t∪{z})−U(t)​(ℬ^t).U\_{z}^{(t)}:=U^{(t)}(\widehat{\mathcal{B}}\_{t}\cup\{z\})-U^{(t)}(\widehat{\mathcal{B}}\_{t}). |  | (15) |

Let θ~t​(ℬ^t)\tilde{\theta}\_{t}(\widehat{\mathcal{B}}\_{t}) denote the *virtual parameters* obtained by applying
one descent step on the selected subset ℬ^t\widehat{\mathcal{B}}\_{t}: θ~t​(ℬ^t)=θt+Δ​θt​(ℬ^t).\tilde{\theta}\_{t}(\widehat{\mathcal{B}}\_{t})=\theta\_{t}+\Delta\theta\_{t}(\widehat{\mathcal{B}}\_{t}). Adding zz induces an additional update Δ​θt​({z})\Delta\theta\_{t}(\{z\}), so the marginal gain can be written as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Uz(t)=ℒ​(𝒟val;θ~t​(ℬ^t))−ℒ​(𝒟val;θ~t​(ℬ^t)+Δ​θt​({z})).U\_{z}^{(t)}=\mathcal{L}(\mathcal{D}\_{\text{val}};\tilde{\theta}\_{t}(\widehat{\mathcal{B}}\_{t}))-\mathcal{L}(\mathcal{D}\_{\text{val}};\tilde{\theta}\_{t}(\widehat{\mathcal{B}}\_{t})+\Delta\theta\_{t}(\{z\})). |  | (16) |

Using a first-order Taylor approximation of the validation loss at
θ~t​(ℬ^t)\tilde{\theta}\_{t}(\widehat{\mathcal{B}}\_{t}), we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ​(𝒟val;θ~t​(ℬ^t)+Δ​θt​({z}))≈ℒ​(𝒟val;θ~t​(ℬ^t))+∇θℒ​(𝒟val;θ~t​(ℬ^t))⊺​Δ​θt​({z}).\begin{split}\mathcal{L}\!\left(\mathcal{D}\_{\text{val}};\tilde{\theta}\_{t}(\widehat{\mathcal{B}}\_{t})+\Delta\theta\_{t}(\{z\})\right)\approx\mathcal{L}\!\left(\mathcal{D}\_{\text{val}};\tilde{\theta}\_{t}(\widehat{\mathcal{B}}\_{t})\right)\\ +\nabla\_{\theta}\mathcal{L}\!\left(\mathcal{D}\_{\text{val}};\tilde{\theta}\_{t}(\widehat{\mathcal{B}}\_{t})\right)^{\intercal}\Delta\theta\_{t}(\{z\}).\end{split} |  | (17) |

Substituting Eq. ([17](#S5.E17 "In 5.1 Optimizer-Induced Utility Objective ‣ 5 Methodology: OPUS ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")) into Eq. ([16](#S5.E16 "In 5.1 Optimizer-Induced Utility Objective ‣ 5 Methodology: OPUS ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")) yields

|  |  |  |  |
| --- | --- | --- | --- |
|  | Uz(t)≈−∇θℒ​(𝒟val;θ~t​(ℬ^t))⊺​Δ​θt​({z}).U\_{z}^{(t)}\approx-\nabla\_{\theta}\mathcal{L}\!\left(\mathcal{D}\_{\text{val}};\tilde{\theta}\_{t}(\widehat{\mathcal{B}}\_{t})\right)^{\intercal}\Delta\theta\_{t}(\{z\}). |  | (18) |

Optimizer-induced geometry.
Unlike vanilla SGD, modern LLM training relies on adaptive optimizers that reshape gradients through a
state-dependent preconditioner. We denote the optimizer state operator at step tt as 𝐏t\mathbf{P}\_{t} and define
the *optimizer-induced effective update direction* as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐮z(t):=𝐏t​∇θℒ​(z;θt).{\mathbf{u}}\_{z}^{(t)}\;:=\;\mathbf{P}\_{t}\nabla\_{\theta}\mathcal{L}(z;\theta\_{t}). |  | (19) |

Accordingly, the optimizer update induced by a subset 𝒮\mathcal{S} can be written as
Δ​θt​(𝒮)=−ηt​∑z∈𝒮𝐮z(t)\Delta\theta\_{t}(\mathcal{S})=-\eta\_{t}\sum\_{z\in\mathcal{S}}{\mathbf{u}}\_{z}^{(t)}.
In particular, adding a single candidate zz contributes an additional update
Δ​θt​({z})=−ηt​𝐮z(t)\Delta\theta\_{t}(\{z\})=-\eta\_{t}\,{\mathbf{u}}\_{z}^{(t)}.
Substituting Δ​θt​({z})\Delta\theta\_{t}(\{z\}) into the marginal approximation
in Eq. ([18](#S5.E18 "In 5.1 Optimizer-Induced Utility Objective ‣ 5 Methodology: OPUS ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")) gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | Uz(t)≈ηt​⟨𝐮z(t),∇θℒ​(𝒟val;θ~t​(ℬ^t))⟩.U\_{z}^{(t)}\approx\eta\_{t}\,\Big\langle{\mathbf{u}}\_{z}^{(t)},\,\nabla\_{\theta}\mathcal{L}\!\left(\mathcal{D}\_{\text{val}};\tilde{\theta}\_{t}(\widehat{\mathcal{B}}\_{t})\right)\Big\rangle. |  | (20) |

Approximating the virtual validation gradient.
The marginal gain of adding a candidate zz to the current subset ℬ^t\widehat{\mathcal{B}}\_{t}, denoted as Uz(t)U\_{z}^{(t)}, depends on the validation gradient evaluated at the *virtual parameters* θ~t​(ℬ^t)\tilde{\theta}\_{t}(\widehat{\mathcal{B}}\_{t}). Specifically, the first-order approximation of the utility is given by the inner product between the optimizer-induced update and the gradient at the virtual point:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Uz(t)≈ηt​⟨𝐮z(t),∇θℒ​(𝒟val;θ~t​(ℬ^t))⟩.U\_{z}^{(t)}\approx\eta\_{t}\Big\langle\mathbf{u}\_{z}^{(t)},\nabla\_{\theta}\mathcal{L}(\mathcal{D}\_{\text{val}};\tilde{\theta}\_{t}(\widehat{\mathcal{B}}\_{t}))\Big\rangle. |  | (21) |

Computing this virtual gradient exactly would require an additional backward pass on 𝒟val\mathcal{D}\_{\text{val}} after every selection step, which is prohibitively expensive. To avoid this cost, we linearize the gradient function 𝐠val​(θ):=∇θℒ​(𝒟val;θ)\mathbf{g}\_{\text{val}}(\theta):=\nabla\_{\theta}\mathcal{L}(\mathcal{D}\_{\text{val}};\theta) around the current parameters θt\theta\_{t}. Let Δ​θt​(ℬ^t):=θ~t​(ℬ^t)−θt\Delta\theta\_{t}(\widehat{\mathcal{B}}\_{t}):=\tilde{\theta}\_{t}(\widehat{\mathcal{B}}\_{t})-\theta\_{t} be the accumulated update from the currently selected subset. A first-order Taylor expansion gives:

|  |  |  |
| --- | --- | --- |
|  | ∇θℒ​(𝒟val;θ~t​(ℬ^t))≈𝐠val​(θt)+∇θ𝐠val​(θt)​Δ​θt​(ℬ^t)=𝐠val(t)+𝐇val(t)​Δ​θt​(ℬ^t),\nabla\_{\theta}\mathcal{L}\!\left(\mathcal{D}\_{\text{val}};\tilde{\theta}\_{t}(\widehat{\mathcal{B}}\_{t})\right)\approx\mathbf{g}\_{\text{val}}(\theta\_{t})+\nabla\_{\theta}\mathbf{g}\_{\text{val}}(\theta\_{t})\,\Delta\theta\_{t}(\widehat{\mathcal{B}}\_{t})=\mathbf{g}\_{\text{val}}^{(t)}+\mathbf{H}\_{\text{val}}^{(t)}\,\Delta\theta\_{t}(\widehat{\mathcal{B}}\_{t}), |  |

where 𝐠val(t)\mathbf{g}\_{\text{val}}^{(t)} is the validation gradient at θt\theta\_{t} and 𝐇val(t)\mathbf{H}\_{\text{val}}^{(t)} is the Hessian.
Using the update rule, the accumulated update is Δ​θt​(ℬ^t)=−ηt​∑zj∈ℬ^t𝐮zj(t)\Delta\theta\_{t}(\widehat{\mathcal{B}}\_{t})=-\eta\_{t}\sum\_{z\_{j}\in\widehat{\mathcal{B}}\_{t}}{\mathbf{u}}\_{z\_{j}}^{(t)}. Substituting the gradient approximation (Eq. ([5.1](#S5.Ex11 "5.1 Optimizer-Induced Utility Objective ‣ 5 Methodology: OPUS ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"))) and the explicit update form into Eq. ([21](#S5.E21 "In 5.1 Optimizer-Induced Utility Objective ‣ 5 Methodology: OPUS ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")), we obtain the final tractable scoring function:

|  |  |  |
| --- | --- | --- |
|  | Uz(t)≈ηt​⟨𝐮z(t),𝐠val(t)−ηt​𝐇val(t)​∑zj∈ℬ^t𝐮zj(t)⟩=ηt​⟨𝐮z(t),𝐠val(t)⟩⏟Alignment−ηt2​⟨𝐮z(t),𝐇val(t)​∑zj∈ℬ^t𝐮zj(t)⟩⏟Redundancy Penalty.U\_{z}^{(t)}\approx\eta\_{t}\left\langle{\mathbf{u}}\_{z}^{(t)},\;\mathbf{g}\_{\text{val}}^{(t)}-\eta\_{t}\mathbf{H}\_{\text{val}}^{(t)}\sum\_{z\_{j}\in\widehat{\mathcal{B}}\_{t}}{\mathbf{u}}\_{z\_{j}}^{(t)}\right\rangle=\underbrace{\eta\_{t}\Big\langle{\mathbf{u}}\_{z}^{(t)},\mathbf{g}\_{\text{val}}^{(t)}\Big\rangle}\_{\text{Alignment}}-\underbrace{\eta\_{t}^{2}\Big\langle{\mathbf{u}}\_{z}^{(t)},\,\mathbf{H}\_{\text{val}}^{(t)}\sum\_{z\_{j}\in\widehat{\mathcal{B}}\_{t}}{\mathbf{u}}\_{z\_{j}}^{(t)}\Big\rangle}\_{\text{Redundancy Penalty}}. |  |

Handling the Hessian complexity.
Materializing 𝐇val\mathbf{H}\_{\text{val}} is intractable at LLM scale. Following (Wang et al., [2024](#bib.bib2 "Greats: online selection of high-quality data for llm training in every iteration")), we adopt an isotropic
approximation for this interaction term, 𝐇val≈𝐈\mathbf{H}\_{\text{val}}\approx\mathbf{I}.
Defining the accumulated effective direction
𝐆(t):=∑zj∈ℬ^t𝐮zj(t){\mathbf{G}}^{(t)}:=\sum\_{z\_{j}\in\widehat{\mathcal{B}}\_{t}}{\mathbf{u}}\_{z\_{j}}^{(t)},
we obtain the practical redundancy-adjusted score:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Uz(t)≈ηt​⟨𝐮z(t),𝐠val(t)⟩−ηt2​⟨𝐮z(t),𝐆(t)⟩.U\_{z}^{(t)}\approx\eta\_{t}\Big\langle{\mathbf{u}}\_{z}^{(t)},\mathbf{g}\_{\text{val}}^{(t)}\Big\rangle-\eta\_{t}^{2}\Big\langle{\mathbf{u}}\_{z}^{(t)},{\mathbf{G}}^{(t)}\Big\rangle. |  | (22) |

Stable proxy construction via Bench-Proxy.
The quality of the proxy direction 𝐠val(t)\mathbf{g}\_{\text{val}}^{(t)} is critical for principled selection. While a random hold-out set provides a low-variance signal, it often fails to capture the specific distribution of downstream tasks. Conversely, using raw benchmark samples directly as the proxy introduces severe distribution shift and gradient noise, destabilizing the ranking.
To bridge this gap, we introduce Bench-Proxy, a retrieval-based construction shown in Fig. [3](#S1.F3 "Figure 3 ‣ 1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")(a).
We embed both (i) the target benchmark validation set and (ii) candidate documents from the pre-training corpus using a frozen text encoder, and retrieve the top-MM most similar pre-training documents to form an *in-distribution* proxy pool 𝒟proxy\mathcal{D}\_{\text{proxy}}.
This approach yields a proxy that is aligned with the target tasks yet remains within the pre-training manifold, ensuring valid gradient estimation.
Concretely, at step tt we draw a proxy mini-batch {z~k}k=1Kproxy⊂𝒟proxy\{\tilde{z}\_{k}\}\_{k=1}^{K\_{\text{proxy}}}\subset\mathcal{D}\_{\text{proxy}} and estimate the direction via 𝐠proxy(t)=1K​∑k=1Kproxy∇θℒ​(z~k;θt)\mathbf{g}\_{\text{proxy}}^{(t)}=\frac{1}{K}\sum\_{k=1}^{K\_{\text{proxy}}}\nabla\_{\theta}\mathcal{L}(\tilde{z}\_{k};\theta\_{t}).
Substituting this proxy estimate into Eq. ([22](#S5.E22 "In 5.1 Optimizer-Induced Utility Objective ‣ 5 Methodology: OPUS ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")), we obtain the final scoring rule:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Uz(t)←ηt​⟨𝐮z(t),𝐠proxy(t)⟩−ηt2​⟨𝐮z(t),𝐆(t)⟩.U\_{z}^{(t)}\leftarrow\eta\_{t}\Big\langle{\mathbf{u}}\_{z}^{(t)},\,\mathbf{g}\_{\text{proxy}}^{(t)}\Big\rangle-\eta\_{t}^{2}\Big\langle{\mathbf{u}}\_{z}^{(t)},\,{\mathbf{G}}^{(t)}\Big\rangle. |  | (23) |

This formulation ensures that selected updates not only reduce loss but specifically align with the benchmark-relevant subspace of the optimization landscape. Further details of Bench-Proxy construction are provided in Sec [6.2](#S6.SS2 "6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").

### 5.2 Scalable Utility Estimation

To score candidates at scale, we leverage the ghost technique (Wang et al., [2024](#bib.bib2 "Greats: online selection of high-quality data for llm training in every iteration"); [2025a](#bib.bib64 "Data shapley in one training run"); Hu et al., [2025](#bib.bib62 "A snapshot of influence: a local data attribution framework for online reinforcement learning")) to avoid per-sample forward/backward passes and the materialization of full gradients. We further apply a low-dimensional sketch to efficiently compute the inner products required for the utility score in Eq. ([23](#S5.E23 "In 5.1 Optimizer-Induced Utility Objective ‣ 5 Methodology: OPUS ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")).

Ghost technique.
Following GREATS (Wang et al., [2024](#bib.bib2 "Greats: online selection of high-quality data for llm training in every iteration")), we exploit the *rank-1 outer product structure* of backpropagated gradients in linear layers.
Consider a linear layer rr with weights 𝐖r\mathbf{W}\_{r}.
For a sample zz, let 𝐚r(z)\mathbf{a}^{(z)}\_{r} denote the input activation vector and 𝐛r(z)\mathbf{b}^{(z)}\_{r} the output gradient vector (error signal).
The per-sample gradient with respect to the weights factorizes as the outer product ∇𝐖rℒ​(z;θt)=𝐚r(z)⊗𝐛r(z)\nabla\_{\mathbf{W}\_{r}}\mathcal{L}(z;\theta\_{t})=\mathbf{a}^{(z)}\_{r}\otimes\mathbf{b}^{(z)}\_{r}, where ⊗\otimes denotes the outer product.
Since 𝐚r(z)\mathbf{a}^{(z)}\_{r} and 𝐛r(z)\mathbf{b}^{(z)}\_{r} are available during the standard forward/backward passes, we can compute gradient statistics without ever materializing the high-dimensional matrix ∇𝐖rℒ\nabla\_{\mathbf{W}\_{r}}\mathcal{L}. In OPUS, we apply it over a set of layers ℛ\mathcal{R} (e.g., linear and embedding matrices).
We concatenate the proxy batch and candidate batch within a single forward/backward pass to collect {𝐚r(z),𝐛r(z)}\{\mathbf{a}^{(z)}\_{r},\mathbf{b}^{(z)}\_{r}\} for all samples. These quantities contain all information required to compute the projected scores, and are discarded layer-by-layer to maintain low memory overhead.

CountSketch projection.
Computing the utility Uz(t)U\_{z}^{(t)} in Eq. ([23](#S5.E23 "In 5.1 Optimizer-Induced Utility Objective ‣ 5 Methodology: OPUS ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")) requires applying the optimizer preconditioner 𝐏t\mathbf{P}\_{t}. We project the resulting effective updates into a low-dimensional sketch space using a sparse CountSketch map Π:ℝd→ℝm\Pi:\mathbb{R}^{d}\to\mathbb{R}^{m} (m≪dm\ll d).
For a linear layer rr with dimensions din×doutd\_{\text{in}}\times d\_{\text{out}}, the per-sample preconditioned sketch feature ϕ(t,r)​(z)∈ℝm\boldsymbol{\phi}^{(t,r)}(z)\in\mathbb{R}^{m} is computed implicitly as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ϕ(t,r)​(z)=Πr​(𝐏t,r​(𝐚r(z)⊗𝐛r(z))).\boldsymbol{\phi}^{(t,r)}(z)=\Pi\_{r}\Big(\mathbf{P}\_{t,r}\big(\mathbf{a}^{(z)}\_{r}\otimes\mathbf{b}^{(z)}\_{r}\big)\Big). |  | (24) |

We instantiate Πr\Pi\_{r} using CountSketch (Cormode and Muthukrishnan, [2005](#bib.bib18 "An improved data stream summary: the count-min sketch and its applications")), which enables computing the projection by streaming over the coordinates of the outer-product gradient without explicitly materializing it.
This choice yields concrete computational benefits depending on the structure of 𝐏t,r\mathbf{P}\_{t,r}.
For AdamW, 𝐏t,r\mathbf{P}\_{t,r} is diagonal (Section [4](#S4 "4 Optimizer-induced Preconditioners ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")), preserving the coordinate-wise separable structure of the outer-product gradient.
This allows the CountSketch projection to be interleaved with preconditioning by applying the diagonal weights on the fly, yielding a projection cost of 𝒪​(din+dout)\mathcal{O}(d\_{\text{in}}+d\_{\text{out}}) rather than the 𝒪​(din​dout)\mathcal{O}(d\_{\text{in}}d\_{\text{out}}) cost required for a dense projection.
In contrast, for optimizers with dense preconditioners such as Muon, coordinate mixing destroys this separability, resulting in a projection cost of 𝒪​(din​dout)\mathcal{O}(d\_{\text{in}}d\_{\text{out}}). We approximate the alignment and redundancy terms by summing dot products in the sketch space across layers:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Uz(t)≈ηt​∑r∈ℛ⟨ϕ(t,r)​(z),ψproxy(t,r)⟩−ηt2​∑r∈ℛ⟨ϕ(t,r)​(z),𝚽(t,r)⟩,U\_{z}^{(t)}\approx\eta\_{t}\sum\_{r\in\mathcal{R}}\langle\boldsymbol{\phi}^{(t,r)}(z),\psi\_{\text{proxy}}^{(t,r)}\rangle-\eta\_{t}^{2}\sum\_{r\in\mathcal{R}}\langle\boldsymbol{\phi}^{(t,r)}(z),\boldsymbol{\Phi}^{(t,r)}\rangle, |  | (25) |

where 𝚽(t,r)=∑zj∈ℬ^tϕ(t,r)​(zj)\boldsymbol{\Phi}^{(t,r)}=\sum\_{z\_{j}\in\widehat{\mathcal{B}}\_{t}}\boldsymbol{\phi}^{(t,r)}(z\_{j}) is the running history of selected sketches.
Note that ψproxy(t,r):=Πr​(1Kproxy​∑k=1Kproxy𝐚r(z~k)⊗𝐛r(z~k))\psi\_{\text{proxy}}^{(t,r)}:=\Pi\_{r}\Big(\frac{1}{K\_{\text{proxy}}}\sum\_{k=1}^{K\_{\text{proxy}}}\mathbf{a}^{(\tilde{z}\_{k})}\_{r}\otimes\mathbf{b}^{(\tilde{z}\_{k})}\_{r}\Big) represents the sketched *unpreconditioned* proxy gradient direction.

### 5.3 Boltzmann Sampling

To preserve diversity under dynamic selection, we replace deterministic greedy top-kk with stochastic sampling.
While our utility formulation in Eq. ([25](#S5.E25 "In 5.2 Scalable Utility Estimation ‣ 5 Methodology: OPUS ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")) explicitly penalizes *geometric* redundancy (vector alignment), greedy selection remains brittle to *estimation noise*: it assumes the proxy direction ψproxy(t,r)\psi\_{\text{proxy}}^{(t,r)} is perfect. In practice, the proxy is a stochastic estimate from a small batch, and the data stream is non-stationary. Always picking the current top-kk can lock the model into transient, noisy features of the proxy batch.
We therefore adopt Boltzmann sampling to improve robustness:

|  |  |  |  |
| --- | --- | --- | --- |
|  | pz(t)∝exp⁡(Uz(t)/τ).p\_{z}^{(t)}\propto\exp\big(U\_{z}^{(t)}/\tau\big). |  | (26) |

This ensures that high-utility candidates are favored, while complementary candidates maintain non-zero probability, preventing overfitting to local proxy noise.

Algorithm [1](#alg1 "In 5 Methodology: OPUS ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration") summarizes OPUS, a step-wise dynamic selection method that scores candidates in the *optimizer-induced update space*. At each step tt, OPUS samples a candidate buffer ℬt\mathcal{B}\_{t}, constructs the preconditioner 𝐏t\mathbf{P}\_{t} from the optimizer state, and builds a proxy *target direction* from an in-distribution pool 𝒟proxy\mathcal{D}\_{\text{proxy}} via ghost factors, yielding per-layer proxy sketches ψproxy(t,r)\psi\_{\text{proxy}}^{(t,r)} for r∈ℛr\in\mathcal{R}. For each candidate z∈ℬtz\in\mathcal{B}\_{t}, it forms a sketch feature ϕ(t,r)​(z)\phi^{(t,r)}(z) by applying 𝐏t,r\mathbf{P}\_{t,r} to the ghost outer-product gradient and projecting with CountSketch Πr\Pi\_{r} into ℝm\mathbb{R}^{m} for efficiency. OPUS then selects K=⌊ρ​N⌋K=\lfloor\rho N\rfloor samples using Boltzmann sampling with a marginal-gain objective that balances proxy alignment and redundancy control, and updates the model on the selected subset ℬ^t\widehat{\mathcal{B}}\_{t}.

## 6 Experiments

### 6.1 Experimental Setup

Models and training settings.
We pre-train GPT-2 Large and GPT-2 XL (Radford et al., [2019](#bib.bib30 "Language models are unsupervised multitask learners")) from scratch under a fixed optimization budget of 30B update tokens. GPT-2 Large consists of 36 layers with a hidden size of 1280, totaling approximately 774M parameters, while GPT-2 XL features a deeper architecture of 48 layers and a hidden size of 1600, amounting to 1.5B parameters. Unless stated otherwise, all methods are compute-matched by performing parameter updates on exactly 30B update tokens. For GPT-2 models, we keep most modules in FP32 but cast the token embedding layers to BF16 for efficiency. We also evaluate OPUS in a continued pre-training setting using the Qwen3-8B-Base  (Yang et al., [2025](#bib.bib40 "Qwen3 technical report")). This model architecture comprises 36 layers with a hidden size of 4096 and approximately 8B parameters. In this configuration, the model is adapted on a science-domain stream, keep the training recipe fixed, and vary only the selection policy. We train with mixed precision in bfloat16. For Qwen3-8B-Base models, we cast the entire model to BF16 to maintain dtype consistency. All experiments run with synchronous data-parallel training using NCCL. Let WW be the number of GPUs (world size) and GG be the gradient accumulation steps; then the global batch size per optimizer update is B=W⋅GB=W\cdot G sequences of length LL, i.e., W⋅G⋅LW\cdot G\cdot L update tokens per step. We apply global gradient-norm clipping with threshold 1.01.0.

Sequence lengths, batch sizes for OPUS.
We use model-specific training sequence lengths due to memory constraints. For GPT-2 we set Ltrain=24,576L\_{\text{train}}{=}24{,}576 (GPT-2 Large) and Ltrain=6,144L\_{\text{train}}{=}6{,}144 (GPT-2 XL), with Lval=32,768L\_{\text{val}}{=}32{,}768 (Large) and Lval=8,192L\_{\text{val}}{=}8{,}192 (XL).111For the Qwen3-8B CPT runs, we use Ltrain=4,096L\_{\text{train}}{=}4{,}096 and Lval=4,096L\_{\text{val}}{=}4{,}096 with FlexAttention.
For OPUS, at each optimization step we score candidates using only Lscore=512L\_{\text{score}}{=}512 tokens of each sequence. We form a candidate buffer of N=32N{=}32 sequences for GPT-2 runs. For Qwen3-8B, we use M=16M{=}16 as a buffer-size multiplier; selection is performed *globally* by gathering scores across all GPUs and selecting the top K=⌊ρ​N⌋K{=}\lfloor\rho N\rfloor sequences with ρ=0.5\rho{=}0.5.
We use the validation split as the proxy set for scoring (proxy batch size 88) and refresh it every step. After selection, the model performs a full forward/backward update on the selected sequences of length LtrainL\_{\text{train}}, and the token budget is counted using LtrainL\_{\text{train}}. The additional forward computation used for scoring is treated as overhead (Sec. [6.6](#S6.SS6 "6.6 Efficiency Analysis ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")). Random projection is disabled in these runs unless stated otherwise.

Optimizers and hyperparameters.
We evaluate two optimizer settings under the same learning-rate schedule and training recipe. In Muon setting, we apply Muon (Jordan et al., [2024](#bib.bib3 "Muon: an optimizer for hidden layers in neural networks"))222The optimizer employed in our implementation combines Muon and AdamW. To simplify notation, we use “Muon” as shorthand for this hybrid optimizer in the remainder of this paper.
 updates to matrix-shaped parameters and use AdamW (Loshchilov and Hutter, [2019](#bib.bib41 "Decoupled weight decay regularization")) for parameter types where Muon-style matrix preconditioning is not directly applicable, such as biases and normalization parameters. In AdamW setting, we use AdamW (Loshchilov and Hutter, [2019](#bib.bib41 "Decoupled weight decay regularization")) for all parameters as a unified baseline.

Table 1: Optimizer assignment by parameter. In our Muon+AdamW setting, Muon is applied to matrix-shaped parameters inside Transformer blocks (model.blocks, ndim≥2\texttt{ndim}\geq 2), while AdamW is applied to embeddings, LM head, and all 0/1D parameters. In the AdamW setting, AdamW is applied to all parameters. Patterns with i=0..L-1 repeat per Transformer layer.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Model | Parameter pattern | Repeats | ndim | Optimizer | Notes |
| GPT2-Large | embed.weight | – | 2D | AdamW | Token embedding table |
| GPT2-Large | lm\_head.weight | – | 2D | AdamW | Tied to embed.weight |
| GPT2-Large | blocks.{i}.attn.qkv\_proj.weight | i=0..35i{=}0..35 | 2D | Muon | Attention QKV projection |
| GPT2-Large | blocks.{i}.attn.c\_proj.weight | i=0..35i{=}0..35 | 2D | Muon | Attention output projection |
| GPT2-Large | blocks.{i}.mlp.c\_fc.weight | i=0..35i{=}0..35 | 2D | Muon | MLP expansion projection |
| GPT2-Large | blocks.{i}.mlp.c\_proj.weight | i=0..35i{=}0..35 | 2D | Muon | MLP contraction projection |
| GPT2-XL | embed.weight | – | 2D | AdamW | Token embedding table |
| GPT2-XL | lm\_head.weight | – | 2D | AdamW | Tied to embed.weight |
| GPT2-XL | blocks.{i}.attn.qkv\_proj.weight | i=0..47i{=}0..47 | 2D | Muon | Attention QKV projection |
| GPT2-XL | blocks.{i}.attn.c\_proj.weight | i=0..47i{=}0..47 | 2D | Muon | Attention output projection |
| GPT2-XL | blocks.{i}.mlp.c\_fc.weight | i=0..47i{=}0..47 | 2D | Muon | MLP expansion projection |
| GPT2-XL | blocks.{i}.mlp.c\_proj.weight | i=0..47i{=}0..47 | 2D | Muon | MLP contraction projection |
| Qwen3-8B-Base | embed.weight | – | 2D | AdamW | Token embedding table |
| Qwen3-8B-Base | lm\_head.weight | – | 2D | AdamW | Tied |
| Qwen3-8B-Base | ln\_f.weight | – | 1D | AdamW | Final RMSNorm weight |
| Qwen3-8B-Base | blocks.{i}.input\_layernorm.weight | i=0..35i{=}0..35 | 1D | AdamW | RMSNorm weight |
| Qwen3-8B-Base | blocks.{i}.post\_attention\_layernorm.weight | i=0..35i{=}0..35 | 1D | AdamW | RMSNorm weight |
| Qwen3-8B-Base | blocks.{i}.self\_attn.q\_norm.weight | i=0..35i{=}0..35 | 1D | AdamW | QK-norm weight |
| Qwen3-8B-Base | blocks.{i}.self\_attn.k\_norm.weight | i=0..35i{=}0..35 | 1D | AdamW | QK-norm weight |
| Qwen3-8B-Base | blocks.{i}.self\_attn.q\_proj.weight | i=0..35i{=}0..35 | 2D | Muon | Attention Q projection |
| Qwen3-8B-Base | blocks.{i}.self\_attn.k\_proj.weight | i=0..35i{=}0..35 | 2D | Muon | Attention K projection |
| Qwen3-8B-Base | blocks.{i}.self\_attn.v\_proj.weight | i=0..35i{=}0..35 | 2D | Muon | Attention V projection |
| Qwen3-8B-Base | blocks.{i}.self\_attn.o\_proj.weight | i=0..35i{=}0..35 | 2D | Muon | Attention output projection |
| Qwen3-8B-Base | blocks.{i}.mlp.gate\_proj.weight | i=0..35i{=}0..35 | 2D | Muon | SwiGLU gate projection |
| Qwen3-8B-Base | blocks.{i}.mlp.up\_proj.weight | i=0..35i{=}0..35 | 2D | Muon | SwiGLU up projection |
| Qwen3-8B-Base | blocks.{i}.mlp.down\_proj.weight | i=0..35i{=}0..35 | 2D | Muon | SwiGLU down projection |
| All | (any remaining parameters) | – | any | AdamW |  |

Optimizer assignment.
For clarity and reproducibility, we explicitly specify how parameters are assigned to optimizers in our experimental settings (Table [1](#S6.T1 "Table 1 ‣ 6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")).
In the Muon setting, we apply Muon updates only to *matrix-shaped* parameters inside Transformer blocks, i.e., parameters under model.blocks with ndim≥2\texttt{ndim}\geq 2 (e.g., attention and MLP projection matrices).
All remaining parameters—including token embeddings, the LM head, and all 0/1D parameters such as RMSNorm weights and biases—are optimized with a distributed AdamW optimizer.
This hybrid design follows the recommended usage of Muon, which is intended for 2D matrices and is not directly applicable to 0/1D parameter types.
In the AdamW setting, we instead optimize all parameters with AdamW optimizer.

Muon optimizer configuration.
We use a hybrid optimizer in which Muon updates the matrix parameters inside Transformer blocks (parameters with ndim≥2\texttt{ndim}\geq 2), excluding the token embedding table and the final LM head. All remaining parameters are updated with AdamW. Muon applies SGD with momentum (μ=0.95\mu=0.95) with no weight decay, followed by an orthogonalization post-processing step on each 2D update. Specifically, we run a Newton–Schulz quintic iteration for 55 steps in BF16 to produce an approximate zeroth-power transform, serving as an efficient surrogate to the U​V⊤UV^{\top} factor in SVD-based orthogonalization. To stabilize updates across differently-shaped matrices, we rescale the effective learning rate for each matrix parameter W∈ℝm×nW\in\mathbb{R}^{m\times n} as

|  |  |  |
| --- | --- | --- |
|  | ηeff=η⋅max⁡(1,mn).\eta\_{\text{eff}}=\eta\cdot\sqrt{\max\!\left(1,\frac{m}{n}\right)}. |  |

For the AdamW-updated parameter groups in this hybrid setup, we use β1=0.8\beta\_{1}=0.8, β2=0.95\beta\_{2}=0.95, ϵ=10−8\epsilon=10^{-8}, and weight decay λ=0\lambda=0, synchronizing gradients via memory-efficient reduce-scatter when dimensions are divisible by the world size and otherwise falling back to all-reduce for correctness.

AdamW optimizer configuration.
For settings that use AdamW, we update all model parameters—including token embeddings, all Transformer block parameters, and the final LM head—with a distributed AdamW optimizer using β1=0.8\beta\_{1}=0.8, β2=0.95\beta\_{2}=0.95, ϵ=10−8\epsilon=10^{-8}, and weight decay λ=0\lambda=0. Gradients are synchronized using reduce-scatter when tensor dimensions are divisible by the world size, and otherwise using all-reduce to ensure numerically correct distributed updates.

Learning rate and optimization hyperparameters.
For GPT-2 XL, we use lradam=2×10−3\mathrm{lr}\_{\text{adam}}{=}2{\times}10^{-3} and lrmuon=1×10−2\mathrm{lr}\_{\text{muon}}{=}1{\times}10^{-2}. AdamW uses β1=0.8\beta\_{1}{=}0.8, β2=0.95\beta\_{2}{=}0.95, ϵ=10−8\epsilon{=}10^{-8}, and no weight decay (λ=0\lambda{=}0). Muon uses momentum μ=0.95\mu{=}0.95 with a short warmup from 0.85→0.950.85\rightarrow 0.95 over the first 300300 steps, and no weight decay.
For Qwen3-8B CPT (SciPedia), we use lradam=10−6\mathrm{lr}\_{\text{adam}}{=}10^{-6} and lrmuon=10−5\mathrm{lr}\_{\text{muon}}{=}10^{-5} with AdamW hyperparameters β1=0.9\beta\_{1}{=}0.9, β2=0.95\beta\_{2}{=}0.95, and weight decay λ=0.01\lambda{=}0.01. We apply global gradient-norm clipping with threshold 1.01.0 in all experiments.
The global batch per optimization step is B=W⋅GB{=}W\cdot G sequences of length LL, where WW is the number of GPUs and GG is the number of gradient-accumulation steps (Qwen3-8B uses W=8W{=}8 and G=1G{=}1). We train Qwen3-8B for a token budget of 1.51.5B tokens and evaluate every 0.50.5B tokens. The learning-rate schedule is implemented as a piecewise multiplier over the base LR with a warmup fraction of 0.010.01.

Random projection configuration.
To accelerate OPUS scoring, we apply a CountSketch-based random projection to per-sample gradients, implementing the sketching operator. Concretely, for each trainable linear weight we form the per-sample gradient in outer-product form (aggregated over time when applicable) and then sketch the flattened gradient into an mm-dimensional vector using CountSketch with a deterministic hash/sign pair; this yields an unbiased estimator of inner products, 𝔼​⟨Π​(g1),Π​(g2)⟩=⟨g1,g2⟩\mathbb{E}\langle\Pi(g\_{1}),\Pi(g\_{2})\rangle=\langle g\_{1},g\_{2}\rangle, enabling us to compute gradient dot-products (and similarity matrices) in the projected space. We set the sketch dimension to m=8192m=8192 with seed 4242, which provides substantial compression for GPT-2 XL where the largest matrix-gradient has dimension on the order of 10.2410.24M, corresponding to an effective compression of roughly 1250×1250\times while preserving the ranking signal used by OPUS. The projection is enabled during scoring and uses cached hash/sign tensors per parameter shape for efficiency; when disabled, we fall back to exact full-dimensional dot-products.

Table 2: Benchmark evaluation configuration. For most benchmarks we use multiple-choice perplexity: score each candidate option by negative log-likelihood and choose the best-scoring option; we report accuracy. MMLU is evaluated separately using zero-shot and log-likelihood on the entire answer following FineWeb-Edu.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Benchmark | Domain | #Choices | Eval mode | Metric |
| Core Benchmarks (in-domain) | | | | |
| MMLU | Knowledge | 4 | LL | Accuracy |
| ANLI | Understanding | 3 | PPL | Accuracy |
| HellaSwag | Commonsense and Reasoning | 4 | PPL | Accuracy |
| PIQA | Commonsense and Reasoning | 2 | PPL | Accuracy |
| SIQA | Commonsense and Reasoning | 3 | PPL | Accuracy |
| WinoGrande | Language | 2 | LL | Accuracy |
| ARC-Easy | Science and Reasoning | 4 | PPL | Accuracy |
| ARC-Challenge | Science and Reasoning | 4 | PPL | Accuracy |
| CommonsenseQA | Commonsense and Reasoning | 5 | PPL | Accuracy |
| WSC | Language | 2 | PPL | Accuracy |
| Other Benchmarks (out-of-domain) | | | | |
| BBH | Reasoning (hard) | – | Generation | Exact Match |
| RACE-Middle | Understanding | 4 | PPL | Accuracy |
| RACE-High | Understanding | 4 | PPL | Accuracy |
| AX-b | Language | 2 | PPL | Accuracy |
| AX-g | Language | 2 | PPL | Accuracy |
| StoryCloze | Understanding | 2 | PPL | Accuracy |

Pre-training corpus.
For from-scratch pre-training, all methods draw candidates from the same 3T-token pool constructed from FineWeb (Penedo et al., [2024](#bib.bib8 "The fineweb datasets: decanting the web for the finest text data at scale")). To test robustness on a higher-quality corpus, we also run the same recipe on FineWeb-Edu (Penedo et al., [2024](#bib.bib8 "The fineweb datasets: decanting the web for the finest text data at scale")). FineWeb-Edu provides a document-level quality classifier that assigns each document a discrete score in {3,4,5}. We partition the FineWeb-Edu pool into two buckets: a 120B-token mid-quality bucket consisting of all score-3 documents, and a 80B-token high-quality bucket formed by merging score-4 and score-5 documents. For static filtering baselines, we score the full pool once and materialize a fixed 30B-token subset for training. For dynamic methods, candidates are streamed from the pool and selected during training. For CPT, we construct a 3B-token pool from SciencePedia (SciencePedia Team, [2025](#bib.bib43 "SciencePedia dataset")) for continued pre-training.

Evaluation. We evaluate all GPT-2 pretraining checkpoints on a variety of benchmarks target diverse capabilities. See Table  [2](#S6.T2 "Table 2 ‣ 6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration") for the summary of the configurations.

Specifically, we evaluate on the following benchmarks to test the general capabilities of our pretrained models:

* •

  MMLU (Hendrycks et al., [2021](#bib.bib20 "Measuring massive multitask language understanding")): broad factual and academic knowledge across many subjects.
* •

  ANLI (Nie et al., [2020](#bib.bib21 "Adversarial NLI: a new benchmark for natural language understanding")): adversarial natural language inference, testing robust entailment and contradiction reasoning.
* •

  HellaSwag (Zellers et al., [2019](#bib.bib22 "HellaSwag: can a machine really finish your sentence?")): commonsense reasoning for plausible continuations.
* •

  PIQA (Bisk et al., [2020](#bib.bib24 "PIQA: reasoning about physical commonsense in natural language")): physical commonsense reasoning about everyday actions.
* •

  SIQA (Sap et al., [2019](#bib.bib74 "Socialiqa: commonsense reasoning about social interactions")): social commonsense and intent reasoning.
* •

  WinoGrande (Sakaguchi et al., [2020](#bib.bib23 "WinoGrande: an adversarial winograd schema challenge at scale")): pronoun/coreference resolution with adversarial bias reduction.
* •

  ARC-E / ARC-C (Clark et al., [2018](#bib.bib26 "Think you have solved question answering? try arc, the ai2 reasoning challenge")): grade-school science questions; Easy and Challenge splits measure increasing reasoning difficulty.
* •

  CommonsenseQA (Talmor et al., [2019](#bib.bib27 "Commonsenseqa: a question answering challenge targeting commonsense knowledge")): commonsense knowledge and reasoning over concepts.
* •

  WSC (Levesque et al., [2012](#bib.bib28 "The winograd schema challenge.")): hard coreference requiring commonsense.

For all above benchmarks except for MMLU, we use OpenCompass (Contributors, [2023](#bib.bib68 "OpenCompass: a universal evaluation platform for foundation models")) with a multiple-choice perplexity scoring rule: for each candidate answer option, we compute its average negative log-likelihood conditioned on the prompt, and predict the option with the lowest perplexity; we then report accuracy.
For WinoGrande, we follow the OpenCompass log-likelihood variant that compares the likelihood of the two candidates. All these benchmarks are evaluated zero-shot. MMLU is evaluated separately with Lighteval (Habib et al., [2023](#bib.bib69 "LightEval: a lightweight framework for llm evaluation")) following the implementation in FineWeb-Edu (Penedo et al., [2024](#bib.bib8 "The fineweb datasets: decanting the web for the finest text data at scale")) evaluation protocol. Since the typical MMLU implementation (which uses ”A”, ”B”, etc as answer targets) gives generally random results on non instruction tuned models, instead, we use the full MMLU answer as the target. We also use zero-shot prompting and then select the answer by comparing the log-likelihood of the entire option string.

In addition, we use the following benchmarks that are not in our bench-proxy set for the generalization evaluation:

* •

  BBH (Suzgun et al., [2023](#bib.bib70 "Challenging big-bench tasks and whether chain-of-thought can solve them")): a challenging subset of BIG-Bench tasks emphasizing multi-step reasoning. We select a set of BBH tasks where base models produce non-degenerate outputs: Tracking Shuffled Objects, Reasoning about Colored Objects, Logical Deduction, Disambiguation QA, Penguins in a Table, and Sports Understanding.
* •

  RACE-M / RACE-H (Lai et al., [2017](#bib.bib71 "RACE: large-scale ReAding comprehension dataset from examinations")): exam-style reading comprehension with multiple choice questions; we use the Middle and High school subsets.
* •

  AX-B / AX-G (Wang et al., [2019](#bib.bib72 "Superglue: a stickier benchmark for general-purpose language understanding systems")): diagnostic evaluation sets from SuperGLUE designed to stress-test linguistic phenomena and generalization.
* •

  StoryCloze (Mostafazadeh et al., [2016](#bib.bib73 "A corpus and cloze evaluation for deeper understanding of commonsense stories")): story ending prediction to test narrative coherence and commonsense continuation.

We evaluate these benchmarks using the OpenCompass framework. All these benchmarks are evaluated zero-shot except for BBH, which uses three-shot. For BBH, many subtasks are near-chance at our model scale, so an aggregate score over all subtasks becomes unstable and less informative. We therefore report results on the curated subset above, where the base model achieves non-trivial accuracy and methods exhibit meaningful separation.

CPT evaluation. We evaluate continued pre-training checkpoints of Qwen3-8B-Base on two science focused benchmarks, OlympicArena (Huang et al., [2024](#bib.bib47 "OlympicArena: benchmarking multi-discipline cognitive reasoning for superintelligent ai")) and SciAssess (Cai et al., [2024](#bib.bib46 "SciAssess: benchmarking llm proficiency in scientific literature analysis")).
For OlympicArena, we evaluate on the test split and use zero-shot prompting.
For SciAssess, we evaluate four subdomains in biology, chemistry, material, medicine using a 3-shot prompting setting with chain-of-thought enabled where available. We use stochastic decoding with temperature 0.60.6, top-p=0.95p=0.95, and top-k=20k=20, and max sequence length of 1024. We report the official accuracy metric for both benchmarks.

Baselines.
We compare OPUS against representative data selection methods. (1) Static baselines. We evaluate five representative static filtering methods: QuRating (Wettig et al., [2024](#bib.bib5 "QuRating: selecting high-quality data for training language models")), DSIR (Xie et al., [2023](#bib.bib6 "Data selection for language models via importance resampling")), DCLM-FastText (Li et al., [2024](#bib.bib7 "Datacomp-lm: in search of the next generation of training sets for language models")), FineWeb-Edu Classifier (Penedo et al., [2024](#bib.bib8 "The fineweb datasets: decanting the web for the finest text data at scale")), and UltraFineweb Classifier (Wang et al., [2025c](#bib.bib4 "Ultra-fineweb: efficient data filtering and verification for high-quality llm training data")). (2) Dynamic selection. We include High-PPL (PPL), which selects the highest-loss sequences under the current model following (Ankner et al., [2025](#bib.bib11 "Perplexed by perplexity: perplexity-based data pruning with small reference models")), and GREATS (Wang et al., [2024](#bib.bib2 "Greats: online selection of high-quality data for llm training in every iteration")), which selects samples whose per-sample gradients best align with a SGD-based proxy direction in post-training. We also report results of random selection at 30B and 60B update tokens for baseline comparison.

### 6.2 Bench-proxy construction

We describe how to construct Bench-Proxy, which estimates the validation direction in
Eq. ([22](#S5.E22 "In 5.1 Optimizer-Induced Utility Objective ‣ 5 Methodology: OPUS ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")) via the retrieval pipeline in Fig. [3](#S1.F3 "Figure 3 ‣ 1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")(a).
The goal is to build a small proxy set 𝒟proxy\mathcal{D}\_{\text{proxy}} that matches the target benchmark’s distribution,
while being sampled from the pre-training corpus so gradients can be computed efficiently and consistently during
pre-training.

Similarity scoring.
We first assign each pre-training document a benchmark relevance score based on its semantic similarity to the
benchmark validation set 𝒟val\mathcal{D}\_{\text{val}}.
Concretely, we use a frozen sentence embedding model Arctic-Embed-L v2 (Yu et al., [2024a](#bib.bib67 "Arctic-embed 2.0: multilingual retrieval without compromise")) to encode (i) each benchmark sample and
(ii) each pre-training document into a shared embedding space, and compute cosine similarities between document
embeddings and benchmark embeddings.
To obtain a single scalar score per document, we reduce the similarity vector by taking the maximum similarity over
all benchmark samples, which captures whether a document is strongly aligned with *any* benchmark instance.
This produces a scored version of the pre-training corpus, where each document is annotated with a benchmark alignment score.

Proxy construction.
We then construct the proxy pool 𝒟proxy\mathcal{D}\_{\text{proxy}} by selecting the highest-scoring documents from the scored corpus.
In practice, we sort documents by their benchmark relevance scores in descending order and greedily accumulate them
until reaching a fixed token budget (30M tokens in our experiments), which yields a compact but benchmark-aligned proxy shard.
During training, we repeatedly sample mini-batches from 𝒟proxy\mathcal{D}\_{\text{proxy}} to estimate the proxy gradient direction
used for within-step ranking.
This design keeps scoring stable and low-variance, while steering selection toward data that matches the target benchmark distribution.

Table 3: Evaluation results after training on FineWeb dataset with 30B tokens. Blocks correspond to model size and optimizer. Bold marks the best compute-matched method per benchmark within each block; a longer-training random-sampling reference at 60B update tokens is included for context. Abbreviations: W.G. = Winogrande; C.QA = CommonsenseQA; WSC = Winograd Schema Challenge.

| Method | MMLU | ANLI | HellaSwag | PIQA | SIQA | W.G. | ARC-E | ARC-C | C.QA | WSC | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-2 Large with Muon optimizer on 30B update tokens of FineWeb | | | | | | | | | | | |
| Random | 28.46 | 32.93 | 42.71 | 69.70 | 40.07 | 49.17 | 37.57 | 28.14 | 31.94 | 36.54 | 39.72 |
| PPL | 28.40 | 33.24 | 42.69 | 70.13 | 40.17 | 48.38 | 36.16 | 23.05 | 31.86 | 36.54 | 39.06 |
| GREATS (Wang et al., [2024](#bib.bib2 "Greats: online selection of high-quality data for llm training in every iteration")) | 28.49 | 33.31 | 42.22 | 70.18 | 39.46 | 49.41 | 36.86 | 24.41 | 33.25 | 36.54 | 39.41 |
| QuRating (Wettig et al., [2024](#bib.bib5 "QuRating: selecting high-quality data for training language models")) | 31.53 | 34.12 | 39.47 | 66.38 | 39.82 | 50.59 | 40.92 | 30.51 | 30.22 | 38.46 | 40.20 |
| DSIR (Xie et al., [2023](#bib.bib6 "Data selection for language models via importance resampling")) | 28.50 | 33.39 | 43.04 | 69.70 | 40.53 | 49.64 | 37.39 | 24.41 | 32.27 | 36.54 | 39.54 |
| DCLM-FastText (Li et al., [2024](#bib.bib7 "Datacomp-lm: in search of the next generation of training sets for language models")) | 29.36 | 33.17 | 44.26 | 71.16 | 39.82 | 49.96 | 37.92 | 24.75 | 32.02 | 36.54 | 39.90 |
| FineWeb-Edu (Penedo et al., [2024](#bib.bib8 "The fineweb datasets: decanting the web for the finest text data at scale")) | 28.83 | 32.67 | 43.09 | 70.02 | 40.28 | 47.75 | 39.15 | 24.75 | 33.66 | 38.46 | 39.87 |
| UltraFineweb (Wang et al., [2025c](#bib.bib4 "Ultra-fineweb: efficient data filtering and verification for high-quality llm training data")) | 29.00 | 32.99 | 44.38 | 71.11 | 40.17 | 48.78 | 37.57 | 25.08 | 33.91 | 38.46 | 40.15 |
| OPUS (Ours) | 28.76 | 33.12 | 42.92 | 69.97 | 39.56 | 50.43 | 38.98 | 29.15 | 33.09 | 36.54 | 40.25 |
| Random (60B) | 28.70 | 33.23 | 45.20 | 71.16 | 40.79 | 49.41 | 39.68 | 25.42 | 31.12 | 36.54 | 40.13 |
| GPT-2 XL with Muon optimizer on 30B update tokens of FineWeb | | | | | | | | | | | |
| Random | 28.73 | 33.98 | 48.01 | 70.46 | 39.61 | 47.91 | 38.98 | 25.42 | 33.25 | 36.54 | 40.29 |
| PPL | 29.35 | 33.42 | 47.87 | 71.55 | 40.69 | 45.86 | 38.45 | 24.07 | 30.38 | 36.54 | 39.82 |
| GREATS (Wang et al., [2024](#bib.bib2 "Greats: online selection of high-quality data for llm training in every iteration")) | 29.95 | 33.58 | 42.26 | 70.18 | 39.61 | 47.67 | 36.33 | 23.73 | 30.55 | 38.46 | 39.23 |
| QuRating (Wettig et al., [2024](#bib.bib5 "QuRating: selecting high-quality data for training language models")) | 33.28 | 33.19 | 48.62 | 70.95 | 41.20 | 48.70 | 37.04 | 26.78 | 30.88 | 36.54 | 40.72 |
| DSIR (Xie et al., [2023](#bib.bib6 "Data selection for language models via importance resampling")) | 29.58 | 33.98 | 48.49 | 71.93 | 39.51 | 47.59 | 38.10 | 26.44 | 32.68 | 38.46 | 40.68 |
| DCLM-FastText (Li et al., [2024](#bib.bib7 "Datacomp-lm: in search of the next generation of training sets for language models")) | 30.40 | 34.08 | 44.07 | 71.38 | 41.97 | 48.38 | 38.80 | 29.49 | 30.88 | 36.54 | 40.60 |
| FineWeb-Edu (Penedo et al., [2024](#bib.bib8 "The fineweb datasets: decanting the web for the finest text data at scale")) | 29.66 | 33.12 | 48.45 | 71.71 | 41.25 | 46.17 | 39.19 | 28.14 | 31.29 | 38.46 | 40.74 |
| UltraFineweb (Wang et al., [2025c](#bib.bib4 "Ultra-fineweb: efficient data filtering and verification for high-quality llm training data")) | 29.95 | 33.31 | 43.11 | 70.57 | 40.79 | 47.51 | 36.51 | 26.44 | 31.70 | 36.54 | 39.64 |
| OPUS (Ours) | 29.89 | 33.29 | 48.39 | 71.27 | 41.10 | 47.99 | 39.68 | 26.44 | 31.37 | 48.08 | 41.75 |
| Random (60B) | 30.24 | 33.84 | 51.10 | 72.25 | 40.89 | 48.78 | 41.98 | 23.05 | 32.35 | 38.46 | 41.29 |
| GPT-2 Large with AdamW on 30B update tokens of FineWeb | | | | | | | | | | | |
| Random | 28.19 | 32.91 | 42.65 | 69.37 | 40.79 | 50.12 | 37.21 | 25.08 | 30.06 | 36.54 | 39.29 |
| PPL | 28.69 | 33.44 | 42.23 | 68.77 | 40.43 | 47.36 | 36.68 | 22.37 | 32.84 | 36.54 | 38.94 |
| GREATS (Wang et al., [2024](#bib.bib2 "Greats: online selection of high-quality data for llm training in every iteration")) | 28.77 | 33.46 | 43.00 | 70.46 | 40.63 | 49.96 | 38.45 | 23.39 | 32.02 | 36.54 | 39.67 |
| QuRating (Wettig et al., [2024](#bib.bib5 "QuRating: selecting high-quality data for training language models")) | 31.87 | 33.08 | 43.22 | 70.24 | 40.74 | 49.88 | 37.21 | 24.75 | 33.58 | 36.54 | 40.11 |
| DSIR (Xie et al., [2023](#bib.bib6 "Data selection for language models via importance resampling")) | 28.22 | 33.18 | 43.42 | 69.53 | 40.02 | 48.93 | 37.92 | 25.08 | 31.20 | 38.46 | 39.60 |
| DCLM-FastText (Li et al., [2024](#bib.bib7 "Datacomp-lm: in search of the next generation of training sets for language models")) | 29.11 | 33.05 | 43.60 | 70.67 | 39.41 | 47.51 | 39.33 | 25.08 | 33.42 | 36.54 | 39.77 |
| FineWeb-Edu (Penedo et al., [2024](#bib.bib8 "The fineweb datasets: decanting the web for the finest text data at scale")) | 29.03 | 35.41 | 42.82 | 70.29 | 40.38 | 47.51 | 39.51 | 27.12 | 31.86 | 38.46 | 40.24 |
| UltraFineweb (Wang et al., [2025c](#bib.bib4 "Ultra-fineweb: efficient data filtering and verification for high-quality llm training data")) | 29.05 | 33.51 | 43.51 | 70.67 | 40.38 | 48.62 | 41.62 | 25.76 | 34.15 | 36.54 | 40.38 |
| OPUS (Ours) | 31.09 | 34.04 | 45.52 | 69.97 | 40.69 | 51.62 | 42.50 | 26.44 | 33.99 | 38.46 | 41.43 |
| Random (60B) | 29.08 | 33.08 | 44.40 | 70.89 | 41.15 | 48.70 | 37.74 | 22.03 | 32.43 | 36.54 | 39.60 |
| GPT-2 XL with AdamW optimizer on 30B update tokens of FineWeb | | | | | | | | | | | |
| Random | 28.76 | 33.56 | 46.63 | 70.35 | 42.37 | 49.19 | 39.15 | 24.41 | 32.68 | 36.54 | 40.36 |
| PPL | 29.32 | 33.67 | 45.31 | 70.08 | 41.71 | 49.72 | 39.68 | 24.75 | 31.29 | 38.46 | 40.02 |
| GREATS (Wang et al., [2024](#bib.bib2 "Greats: online selection of high-quality data for llm training in every iteration")) | 28.81 | 33.49 | 40.73 | 69.53 | 42.48 | 49.01 | 34.22 | 24.75 | 31.04 | 38.46 | 39.25 |
| QuRating (Wettig et al., [2024](#bib.bib5 "QuRating: selecting high-quality data for training language models")) | 32.24 | 32.61 | 34.66 | 66.65 | 38.54 | 50.43 | 36.86 | 24.75 | 28.42 | 36.54 | 38.71 |
| DSIR (Xie et al., [2023](#bib.bib6 "Data selection for language models via importance resampling")) | 29.37 | 33.09 | 45.88 | 70.67 | 39.97 | 47.51 | 38.80 | 24.41 | 33.42 | 36.54 | 39.97 |
| DCLM-FastText (Li et al., [2024](#bib.bib7 "Datacomp-lm: in search of the next generation of training sets for language models")) | 29.43 | 34.47 | 42.45 | 69.91 | 41.86 | 47.59 | 36.33 | 24.41 | 31.53 | 36.54 | 39.45 |
| FineWeb-Edu (Penedo et al., [2024](#bib.bib8 "The fineweb datasets: decanting the web for the finest text data at scale")) | 29.71 | 33.51 | 46.62 | 71.93 | 41.91 | 46.88 | 40.04 | 25.08 | 32.10 | 36.54 | 40.43 |
| UltraFineweb (Wang et al., [2025c](#bib.bib4 "Ultra-fineweb: efficient data filtering and verification for high-quality llm training data")) | 29.25 | 33.51 | 41.76 | 69.21 | 41.40 | 49.57 | 37.92 | 24.07 | 32.76 | 36.54 | 39.60 |
| OPUS (Ours) | 29.43 | 33.51 | 46.12 | 70.35 | 41.35 | 50.36 | 39.33 | 29.15 | 33.99 | 36.54 | 41.01 |
| Random (60B) | 29.55 | 33.57 | 48.75 | 72.09 | 41.10 | 48.78 | 40.92 | 27.12 | 34.48 | 36.54 | 41.29 |

### 6.3 Pre-training from Scratch

Performance on web-scale corpora: FineWeb.
We first evaluate OPUS on FineWeb, a standard large-scale web corpus. Table [3](#S6.T3 "Table 3 ‣ 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration") compares OPUS against prior static and dynamic baselines under a fixed budget of 30B update tokens. Across model scales and optimizer settings, OPUS achieves the best compute-matched average and consistently improves over strong baselines. We also include a longer-training random-sampling reference at 60B update tokens to contextualize the magnitude of these efficiency gains; notably, OPUS often matches or exceeds the performance of baselines trained for twice as long.

Robustness on curated corpora: FineWeb-Edu.
We next evaluate performance on FineWeb-Edu. To test the limits of our method, we subject OPUS to a strict evaluation regime: it selects dynamically from the lower-quality subset (FineWeb-Edu score 3), whereas baselines are trained on the superior high-quality partition (scores 4 and 5). As shown in Table [4](#S6.T4 "Table 4 ‣ 6.3 Pre-training from Scratch ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"), despite this disadvantage in raw data quality, OPUS matches or exceeds prior methods trained on the superior data. For GPT-2 XL with Muon, OPUS achieves the best compute-matched average of 44.99, outperforming all baselines trained on the higher-quality data partitions.

Optimizer-induced selection matters: strong gains under AdamW and Muon. Under AdamW, which utilizes diagonal preconditioning, OPUS achieves the best compute-matched performance for both GPT-2 Large and GPT-2 XL (Table [3](#S6.T3 "Table 3 ‣ 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")). Crucially, this advantage extends to Muon, which employs non-linear matrix preconditioning via Newton-Schulz orthogonalization. For instance, on GPT-2 XL with Muon optimizer on FineWeb, OPUS outperforms Random selection by a significant margin (40.29 →\to 41.75). This empirically validates our central hypothesis: aligning data selection with the preconditioned update trajectory yields a more effective training signal than raw gradient-based selection.

Table 4: Evaluation on FineWeb-Edu dataset with 30B tokens. OPUS is evaluated under a strict constraint: selecting dynamically from the mid-quality subset (score 3), while baselines are trained on the higher-quality partitions (scores ≥4\geq 4). Bold marks the best compute-matched method per benchmark within each block; Random (60B) is shown as a non compute-matched reference.

| Method | MMLU | ANLI | HellaSwag | PIQA | SIQA | W.G. | ARC-E | ARC-C | C.QA | WSC | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-2 Large with Muon optimizer on 30B update tokens of FineWeb-Edu | | | | | | | | | | | |
| Random (Score 3) | 30.52 | 33.16 | 43.95 | 68.87 | 40.58 | 49.02 | 48.39 | 25.08 | 35.54 | 36.54 | 41.17 |
| Random (Score 4+5) | 32.92 | 33.38 | 41.95 | 67.46 | 38.84 | 47.75 | 53.97 | 29.15 | 30.79 | 36.54 | 41.28 |
| PPL (Score 4+5) | 33.17 | 33.87 | 42.25 | 67.63 | 40.33 | 48.22 | 50.79 | 28.47 | 29.48 | 38.46 | 41.27 |
| GREATS (Score 4+5) | 32.73 | 34.38 | 45.86 | 70.95 | 39.30 | 50.36 | 44.62 | 24.75 | 32.92 | 38.46 | 41.43 |
| QuRating (Score 4+5) | 31.32 | 34.07 | 41.70 | 66.92 | 39.71 | 47.83 | 50.79 | 32.88 | 31.94 | 36.54 | 41.37 |
| DSIR (Score 4+5) | 32.54 | 33.54 | 41.07 | 67.95 | 39.36 | 47.28 | 48.68 | 33.90 | 29.57 | 38.46 | 41.24 |
| DCLM-FastText (Score 4+5) | 32.64 | 33.67 | 41.66 | 66.38 | 38.74 | 51.30 | 49.38 | 30.85 | 31.04 | 36.54 | 41.22 |
| FineWeb-Edu (Score 4+5) | 32.00 | 33.46 | 39.95 | 64.74 | 39.87 | 50.51 | 52.20 | 29.15 | 30.30 | 36.54 | 40.87 |
| UltraFineweb (Score 4+5) | 32.60 | 33.02 | 40.70 | 66.05 | 38.23 | 49.72 | 48.32 | 30.17 | 29.24 | 36.54 | 40.46 |
| OPUS (Score 3) | 30.39 | 34.31 | 46.36 | 70.51 | 39.41 | 50.20 | 45.33 | 28.47 | 33.74 | 38.46 | 41.72 |
| OPUS (Score 4+5) | 32.17 | 33.38 | 42.52 | 67.30 | 39.51 | 51.07 | 54.14 | 30.85 | 31.04 | 38.46 | 42.04 |
| Random (60B) (Score 4+5) | 33.21 | 34.03 | 43.66 | 67.95 | 40.07 | 50.04 | 52.56 | 31.86 | 31.61 | 36.54 | 42.15 |
| GPT-2 XL with Muon optimizer on 30B update tokens of FineWeb-Edu | | | | | | | | | | | |
| Random (Score 3) | 31.92 | 33.56 | 48.39 | 70.13 | 41.10 | 48.86 | 44.86 | 28.47 | 34.23 | 36.54 | 41.81 |
| Random (Score 4+5) | 34.32 | 33.78 | 46.39 | 68.72 | 39.36 | 47.59 | 50.44 | 32.54 | 29.48 | 36.54 | 41.92 |
| PPL (Score 4+5) | 32.60 | 33.58 | 46.14 | 69.10 | 40.33 | 51.70 | 50.79 | 30.17 | 31.78 | 36.54 | 42.27 |
| GREATS (Score 4+5) | 33.58 | 33.02 | 46.32 | 68.93 | 39.61 | 52.57 | 49.21 | 33.90 | 28.01 | 36.54 | 42.17 |
| QuRating (Score 4+5) | 33.10 | 33.58 | 44.22 | 66.70 | 39.97 | 49.64 | 50.09 | 32.54 | 28.99 | 36.54 | 41.54 |
| DSIR (Score 4+5) | 34.13 | 33.63 | 45.10 | 67.79 | 39.82 | 48.15 | 49.03 | 32.88 | 28.83 | 36.54 | 41.59 |
| DCLM-FastText (Score 4+5) | 33.19 | 33.02 | 44.36 | 68.23 | 41.15 | 48.86 | 51.32 | 35.59 | 30.14 | 36.54 | 42.24 |
| FineWeb-Edu (Score 4+5) | 32.94 | 33.64 | 43.14 | 68.28 | 39.61 | 51.30 | 52.73 | 32.20 | 31.37 | 36.54 | 42.18 |
| UltraFineweb (Score 4+5) | 33.41 | 33.48 | 44.34 | 68.93 | 38.64 | 48.30 | 49.38 | 33.56 | 29.07 | 36.54 | 41.57 |
| OPUS (Score 4+5) | 33.83 | 33.64 | 46.30 | 70.67 | 38.95 | 51.14 | 50.62 | 29.15 | 30.47 | 39.42 | 42.42 |
| OPUS (Score 3) | 32.62 | 33.11 | 50.54 | 72.20 | 41.04 | 51.46 | 47.62 | 30.85 | 35.63 | 54.81 | 44.99 |
| Random (60B) (Score 4+5) | 33.77 | 33.54 | 46.94 | 69.64 | 39.82 | 49.80 | 50.44 | 32.54 | 30.96 | 38.46 | 42.59 |

Generalization beyond proxy-aligned benchmarks.
Since OPUS uses a benchmark-matched proxy direction to guide training-time selection, it is important to verify that gains are not merely driven by overfitting to the specific evaluation suite used to construct the proxy.
We therefore evaluate on a set of out-of-distribution benchmarks covering challenging reasoning and general language comprehension for generalization evaluation. As shown in Table [5](#S6.T5 "Table 5 ‣ 6.3 Pre-training from Scratch ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"), OPUS achieves the best performance, suggesting that it reflects more general training signal quality, rather than narrow specialization to the proxy-aligned benchmark.

Table 5: Evaluation on out-of-distribution benchmarks. We evaluate the same GPT2-XL checkpoints from Table [3](#S6.T3 "Table 3 ‣ 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration") on out-of-distribution benchmarks that are not included in Bench-Proxy.

| Method | BBH | RACE-M | RACE-H | AX-b | AX-g | StoryCloze | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 9.87 | 24.58 | 25.19 | 52.54 | 50.00 | 66.38 | 38.09 |
| PPL | 9.88 | 24.37 | 25.73 | 54.98 | 51.12 | 67.34 | 38.90 |
| GREATS | 10.44 | 26.04 | 26.04 | 57.34 | 50.84 | 65.79 | 39.42 |
| QuRating | 10.65 | 24.79 | 23.33 | 54.35 | 51.97 | 66.70 | 38.63 |
| DSIR | 9.92 | 25.07 | 26.21 | 53.53 | 49.44 | 67.72 | 38.65 |
| DCLM-FastText | 10.65 | 26.53 | 25.59 | 52.08 | 51.97 | 66.86 | 38.95 |
| FineWeb-Edu | 9.73 | 26.81 | 25.90 | 55.25 | 50.00 | 66.76 | 39.08 |
| UltraFineweb | 9.69 | 23.26 | 22.58 | 48.73 | 48.31 | 67.13 | 36.62 |
| OPUS (Ours) | 11.02 | 25.77 | 27.50 | 58.42 | 50.56 | 67.13 | 40.07 |

Validation loss curves on FineWeb-Edu dataset. We report validation-loss trajectories in Figure [4](#S6.F4 "Figure 4 ‣ 6.3 Pre-training from Scratch ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration") for GPT-2 XL and GPT-2 Large trained from scratch on FineWeb-Edu under the same training recipe and a fixed budget of 30B update tokens.
To make the comparison conservative for OPUS, OPUS selects dynamically from the mid-quality pool with score 3, whereas the baselines are trained on the high-quality pool with scores 4+5.
All curves are evaluated on the same held-out FineWeb-Edu validation split.
We also include a longer-training Random reference at 60B update tokens (not compute-matched) to contextualize convergence speed.

Figure 4: Validation-loss curves on GPT-2 XL and GPT-2 Large pre-trained from scratch on FineWeb-Edu dataset. Left: Results on GPT-2 XL. OPUS compared with representative baselines trained on the high-quality pool, with Random 60B shown as a non compute-matched reference. Curves are shown up to 30B update tokens for compute-matched comparison. Right: Results on GPT2-Large.

As shown in Fig. [4](#S6.F4 "Figure 4 ‣ 6.3 Pre-training from Scratch ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"), OPUS consistently improves optimization dynamics for both model scales: across training, it attains lower validation loss than representative baselines despite selecting from the lower-quality candidate pool. For GPT-2 XL, OPUS reaches the validation loss achieved by Random trained for 60B update tokens using only 17B update tokens, demonstrating substantially faster convergence.
For GPT-2 Large, OPUS exhibits the same trend and maintains a clear gap over baselines throughout training.

OPUS enhances knowledge compression measured by domain perplexity across domains.
To ensure that our selection strategy does not overfit to specific patterns at the expense of broad coverage, we evaluate domain-wise perplexity (PPL). Following the evaluation protocol of WebOrganizer (Wettig et al., [2025](#bib.bib48 "Organize the web: constructing domains enhances pre-training data curation")), we first label documents using the WebOrganizer topic classifier to classify documents into 24 topics and merge these semantically similar topics into ten domains.
We then construct a held-out test set by randomly sampling 1,000 documents from each of ten distinct domains (e.g., Health, Law, Science) to ensure a balanced evaluation. Table [6](#S6.T6 "Table 6 ‣ 6.3 Pre-training from Scratch ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration") indicates that OPUS achieves the lowest average perplexity on FineWeb-Edu dataset.

Table 6: Domain-specific perplexity analysis. Perplexity (PPL; lower is better) on ten domains after 30B update tokens. We construct a validation pool of 10 domains from (Wettig et al., [2025](#bib.bib48 "Organize the web: constructing domains enhances pre-training data curation")), containing 1000 held-out samples per domain.

| Method | Health | Business | Politics | Education | History | Lifestyle | Science | Arts & Lit. | Entertainment | Computing | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-2 Large with Muon optimizer on 30B update tokens of FineWeb | | | | | | | | | | | |
| Random (30B) | 3.21 | 3.26 | 3.28 | 3.31 | 3.32 | 3.37 | 3.40 | 3.49 | 3.56 | 3.62 | 3.38 |
| DSIR | 3.21 | 3.26 | 3.28 | 3.31 | 3.32 | 3.38 | 3.40 | 3.49 | 3.57 | 3.63 | 3.39 |
| DCLM-FastText | 3.17 | 3.24 | 3.26 | 3.30 | 3.36 | 3.37 | 3.36 | 3.46 | 3.54 | 3.60 | 3.37 |
| FineWeb-Edu | 3.17 | 3.24 | 3.25 | 3.28 | 3.26 | 3.41 | 3.34 | 3.48 | 3.58 | 3.61 | 3.36 |
| QuRating | 3.40 | 3.60 | 3.79 | 3.57 | 3.68 | 4.05 | 3.61 | 3.92 | 4.27 | 4.11 | 3.80 |
| UltraFineweb | 3.19 | 3.29 | 3.30 | 3.32 | 3.30 | 3.43 | 3.38 | 3.50 | 3.59 | 3.62 | 3.39 |
| PPL | 3.22 | 3.26 | 3.28 | 3.31 | 3.32 | 3.37 | 3.39 | 3.49 | 3.56 | 3.61 | 3.38 |
| GREATS | 3.25 | 3.31 | 3.33 | 3.36 | 3.38 | 3.42 | 3.46 | 3.55 | 3.62 | 3.66 | 3.43 |
| OPUS (Ours) | 3.18 | 3.23 | 3.25 | 3.28 | 3.30 | 3.34 | 3.37 | 3.47 | 3.54 | 3.58 | 3.35 |
| GPT-2 XL with Muon optimizer on 30B update tokens of FineWeb | | | | | | | | | | | |
| Random (30B) | 3.18 | 3.25 | 3.26 | 3.29 | 3.30 | 3.35 | 3.40 | 3.49 | 3.56 | 3.61 | 3.37 |
| DSIR | 3.15 | 3.22 | 3.23 | 3.26 | 3.25 | 3.32 | 3.35 | 3.44 | 3.52 | 3.56 | 3.33 |
| DCLM-FastText | 3.15 | 3.23 | 3.25 | 3.31 | 3.25 | 3.36 | 3.34 | 3.45 | 3.53 | 3.60 | 3.35 |
| FineWeb-Edu | 3.16 | 3.23 | 3.24 | 3.28 | 3.25 | 3.40 | 3.34 | 3.47 | 3.62 | 3.60 | 3.36 |
| QuRating | 3.27 | 3.53 | 3.67 | 3.47 | 3.59 | 3.91 | 3.51 | 3.83 | 4.14 | 3.96 | 3.69 |
| UltraFineweb | 3.10 | 3.20 | 3.19 | 3.24 | 3.21 | 3.33 | 3.29 | 3.41 | 3.50 | 3.53 | 3.30 |
| PPL | 3.11 | 3.17 | 3.18 | 3.21 | 3.22 | 3.27 | 3.30 | 3.40 | 3.46 | 3.50 | 3.28 |
| GREATS | 3.22 | 3.29 | 3.29 | 3.33 | 3.32 | 3.39 | 3.42 | 3.51 | 3.58 | 3.66 | 3.40 |
| OPUS (Ours) | 3.08 | 3.15 | 3.16 | 3.18 | 3.21 | 3.23 | 3.29 | 3.39 | 3.45 | 3.44 | 3.26 |
| GPT-2 Large with Muon optimizer on 30B update tokens of FineWeb-Edu Subset (score ≥3\geq 3) | | | | | | | | | | | |
| Random (30B) | 3.27 | 3.52 | 3.58 | 3.49 | 3.48 | 3.81 | 3.43 | 3.75 | 4.03 | 3.82 | 3.62 |
| DSIR | 3.29 | 3.55 | 3.61 | 3.52 | 3.49 | 3.84 | 3.46 | 3.77 | 4.05 | 3.86 | 3.64 |
| DCLM-FastText | 3.34 | 3.61 | 3.67 | 3.59 | 3.58 | 3.89 | 3.5 | 3.82 | 4.09 | 3.89 | 3.70 |
| FineWeb-Edu | 3.41 | 3.67 | 3.72 | 3.62 | 3.60 | 3.97 | 3.57 | 3.87 | 4.17 | 3.98 | 3.76 |
| QuRating | 3.46 | 3.76 | 3.90 | 3.65 | 3.79 | 4.13 | 3.70 | 4.00 | 4.36 | 4.16 | 3.89 |
| UltraFineweb | 3.42 | 3.72 | 3.87 | 3.66 | 3.77 | 4.05 | 3.58 | 3.96 | 4.26 | 4.00 | 3.83 |
| PPL | 3.25 | 3.49 | 3.54 | 3.46 | 3.44 | 3.78 | 3.41 | 3.71 | 3.99 | 3.80 | 3.59 |
| GREATS | 3.29 | 3.55 | 3.62 | 3.52 | 3.50 | 3.84 | 3.46 | 3.77 | 4.06 | 3.86 | 3.65 |
| OPUS (Ours) | 3.14 | 3.34 | 3.44 | 3.37 | 3.37 | 3.63 | 3.38 | 3.63 | 3.87 | 3.71 | 3.49 |
| GPT-2 XL with Muon optimizer on 30B update tokens of FineWeb-Edu Subset (score ≥3\geq 3) | | | | | | | | | | | |
| Random (30B) | 3.25 | 3.51 | 3.55 | 3.48 | 3.45 | 3.79 | 3.42 | 3.73 | 4.00 | 3.83 | 3.60 |
| DSIR | 3.24 | 3.50 | 3.54 | 3.47 | 3.44 | 3.78 | 3.41 | 3.72 | 4.00 | 3.81 | 3.59 |
| DCLM-FastText | 3.36 | 3.64 | 3.70 | 3.62 | 3.61 | 3.94 | 3.52 | 3.86 | 4.13 | 3.94 | 3.73 |
| FineWeb-Edu | 3.29 | 3.55 | 3.58 | 3.50 | 3.49 | 3.82 | 3.45 | 3.75 | 4.02 | 3.83 | 3.63 |
| QuRating | 3.50 | 3.79 | 3.93 | 3.70 | 3.83 | 4.18 | 3.73 | 4.04 | 4.39 | 4.24 | 3.93 |
| UltraFineweb | 3.43 | 3.74 | 3.90 | 3.68 | 3.80 | 4.07 | 3.59 | 3.99 | 4.28 | 4.02 | 3.85 |
| PPL | 3.22 | 3.47 | 3.50 | 3.44 | 3.40 | 3.74 | 3.39 | 3.69 | 3.96 | 3.77 | 3.56 |
| GREATS | 3.29 | 3.55 | 3.60 | 3.52 | 3.49 | 3.84 | 3.45 | 3.77 | 4.05 | 3.88 | 3.64 |
| OPUS (Ours) | 3.11 | 3.31 | 3.37 | 3.34 | 3.31 | 3.59 | 3.33 | 3.58 | 3.83 | 3.69 | 3.45 |




Figure 5: CPT domain breakdown on SciencePedia.
Domain-level accuracy of Qwen3-8B-Base and CPT baselines across three token budgets 0.5B, 1B, and 1.5B.
Rows correspond to the CPT token budget. Columns show (a) OlympicArena domains with an appended Avg. and (b) SciAssess domains.
For each panel, we compare Qwen3-8B-Base, Full CPT (3B), Random, DCLM, and OPUS. All results use the official benchmark metrics.

### 6.4 Continued Pre-training

We extend our evaluation to continued pre-training (CPT), a critical setting for adapting general-purpose LLMs to specialized verticals. We continue training Qwen3-8B-Base on SciencePedia.
Figure [6](#S6.F6 "Figure 6 ‣ 6.4 Continued Pre-training ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration") reports the average downstream performance on the specialized SciAssess benchmark and the reasoning-heavy OlympicArena versus CPT tokens. Notably, OPUS reaches the best performance using only 0.5B tokens and already outperforms random CPT trained for 3B tokens, implying a 6×\times gain in data efficiency.

Figure 6: Continued pre-training results on SciencePedia.

Detailed domain-wise CPT Results.
Figure [5](#S6.F5 "Figure 5 ‣ 6.3 Pre-training from Scratch ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration") reports domain breakdowns for continued pre-training on SciencePedia across three token budgets 0.5B, 1B, and 1.5B.
Across OlympicArena (Fig. [5](#S6.F5 "Figure 5 ‣ 6.3 Pre-training from Scratch ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")a) OPUS consistently improves over the base Qwen3-8B-Base and the compute-matched Random baseline in most scientific domains like physics, chemistry, biology, and geography, as well as the text-only and multimodal subsets., with gains that are broadly distributed rather than concentrated in a single category.
Importantly, OPUS is competitive with, and sometimes surpasses, DCLM and even the Full CPT reference despite using at most 1.5B update tokens, indicating strong data efficiency.
On SciAssess (Fig. [5](#S6.F5 "Figure 5 ‣ 6.3 Pre-training from Scratch ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")b), OPUS yields substantial gains on the material and medicine subsets and ties the best baseline on chemistry, leading to the highest average overall, again with at most 1.5B update tokens.

### 6.5 Ablation Study

Soft sampling vs. greedy top-kk.
We replace Boltzmann soft sampling with a deterministic greedy variant that always selects the top-KK candidates by utility.
Table [7](#S6.T7 "Table 7 ‣ 6.5 Ablation Study ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration") shows that greedy selection improves over Random, but remains notably behind full OPUS: the greedy variant reaches an Avg. of 40.49, whereas OPUS achieves 41.75. This supports our motivation that purely greedy top-kk selection can over-concentrate on a narrow set of high-score but overlapping candidates, while stochastic sampling better preserves update diversity and stabilizes training.

Table 7: Ablation study on sampling and validation strategy.

|  |  | OPUS Variants | | |
| --- | --- | --- | --- | --- |
| Benchmark | Random | Greedy | Std. proxy | OPUS |
| MMLU | 28.73 | 29.63 | 29.50 | 29.89 |
| ANLI | 33.98 | 33.52 | 33.70 | 33.29 |
| HellaSwag | 48.01 | 48.17 | 48.18 | 48.39 |
| PIQA | 70.46 | 72.25 | 71.60 | 71.27 |
| SIQA | 39.61 | 41.61 | 40.28 | 41.10 |
| Winogrande | 47.91 | 49.88 | 51.85 | 47.99 |
| ARC-E | 38.98 | 37.39 | 38.80 | 39.68 |
| ARC-C | 25.42 | 24.75 | 26.10 | 26.44 |
| C.QA | 33.25 | 31.12 | 32.76 | 31.37 |
| WSC | 36.54 | 36.54 | 37.50 | 48.08 |
| Average | 40.29 | 40.49 | 41.03 | 41.75 |

Benchmark-matched proxy vs. standard proxy.
OPUS estimates the target update direction using a small proxy pool. We compare the default proxy construction with a benchmark-matched proxy that is retrieved to better reflect the downstream evaluation distribution (Sec. [5](#S5 "5 Methodology: OPUS ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")).
As shown in Table [7](#S6.T7 "Table 7 ‣ 6.5 Ablation Study ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"), the benchmark-matched proxy yields a measurable improvement over the default setting, increasing the average from 41.03 to 41.75. This indicates that sharpening the proxy direction can further increase the effectiveness of utility-based selection.
Table [7](#S6.T7 "Table 7 ‣ 6.5 Ablation Study ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration") also shows that the standard proxy already provides strong gains over Random, improving the average from 40.29 to 41.03.

Table 8: FineWeb results after 30B update tokens for GPT-2 Large pre-trained on FineWeb with the Muon optimizer under varying buffer size btb\_{t}, temperature τ\tau and CountSketch projection dimension mm. See sampling and validation strategy ablations at Table [7](#S6.T7 "Table 7 ‣ 6.5 Ablation Study ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | MMLU | ANLI | HellaSwag | PIQA | SIQA | W.G. | ARC-E | ARC-C | C.QA | WSC | Avg. |
| GPT-2 Large with Muon optimizer (τ=0.9\tau=0.9 m=8192m=8192) | | | | | | | | | | | |
| Random | 28.46 | 32.93 | 42.71 | 69.70 | 40.07 | 49.17 | 37.57 | 28.14 | 31.94 | 36.54 | 39.72 |
| GPT-2 Large with Muon optimizer on different buffer size btb\_{t} (τ=0.9\tau=0.9 d=8192d=8192) | | | | | | | | | | | |
| OPUS (Buffer size 16) | 28.37 | 33.30 | 42.60 | 69.53 | 40.02 | 48.78 | 38.45 | 27.46 | 32.51 | 36.54 | 39.76 |
| OPUS (Buffer size 32) | 29.23 | 33.36 | 42.76 | 70.4 | 39.30 | 49.72 | 37.39 | 25.42 | 33.42 | 36.54 | 39.75 |
| OPUS (Buffer size 64) | 28.76 | 33.12 | 42.92 | 69.97 | 39.56 | 50.43 | 38.98 | 29.15 | 33.09 | 36.54 | 40.25 |
| GPT-2 Large with Muon optimizer on different temperature τ\tau (bt=64b\_{t}=64 m=8192m=8192) | | | | | | | | | | | |
| OPUS (temperature 0.8) | 28.54 | 34.19 | 42.92 | 69.59 | 40.23 | 49.33 | 37.92 | 26.78 | 32.76 | 36.54 | 39.88 |
| OPUS (temperature 1.0) | 28.62 | 33.64 | 43.63 | 70.46 | 39.97 | 50.12 | 37.21 | 24.41 | 32.19 | 38.46 | 39.87 |
| OPUS (temperature 0.9) | 28.76 | 33.12 | 42.92 | 69.97 | 39.56 | 50.43 | 38.98 | 29.15 | 33.09 | 36.54 | 40.25 |
| GPT-2 Large with Muon optimizer on different CountSketch projection dimension mm (bt=64b\_{t}=64 τ=0.9\tau=0.9) | | | | | | | | | | | |
| OPUS (projection dimension 4096) | 28.57 | 33.46 | 42.75 | 68.39 | 40.79 | 48.46 | 38.27 | 26.10 | 33.01 | 36.54 | 39.63 |
| OPUS (projection dimension 16384) | 28.31 | 33.47 | 42.64 | 70.02 | 40.33 | 49.57 | 36.68 | 22.71 | 32.19 | 37.50 | 39.34 |
| OPUS (projection dimension 8192) | 28.76 | 33.12 | 42.92 | 69.97 | 39.56 | 50.43 | 38.98 | 29.15 | 33.09 | 36.54 | 40.25 |

Hyperparameter sensitivity analysis.
We conduct further ablation studies on key hyperparameters of OPUS, including (i) the candidate buffer size btb\_{t}, (ii) the Boltzmann sampling temperature τ\tau, and (iii) the CountSketch projection dimension mm (Table [8](#S6.T8 "Table 8 ‣ 6.5 Ablation Study ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")). Overall, OPUS is reasonably stable across the tested settings and improves over random selection in most configurations. Increasing the buffer size tends to help, with bt=64b\_{t}{=}64 yielding the best average performance among the evaluated choices. For stochastic selection, a moderate temperature offers a better exploration–exploitation trade-off: τ=0.9\tau{=}0.9 performs best compared to both a lower temperature (more greedy) and a higher temperature (closer to uniform sampling). For random projection, we observe sensitivity to the sketch dimension: m=8192m{=}8192 provides the strongest results among the tested dimensions. Based on these results, we adopt bt=64b\_{t}{=}64, τ=0.9\tau{=}0.9, and m=8192m{=}8192 as our default configuration.

Figure 7: Efficiency and computational cost analysis. Time (minutes) and total compute (PFLOPs) are evaluated on GPT-2 XL after pre-training on FineWeb (30B tokens) with Muon.

### 6.6 Efficiency Analysis

A key advantage of OPUS is its minimal computational overhead. Static filtering methods incur a substantial one-time cost to score the entire corpus, while dynamic selection adds per-iteration scoring during training. As shown in Figure [7](#S6.F7 "Figure 7 ‣ 6.5 Ablation Study ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"), a naïve direct implementation of online selection would incur over 3.5×\times slowdown compared to random sampling. By incorporating ghost gradients and CountSketch projections, OPUS reduces this overhead to only 4.7% while achieving the best benchmark performance. In contrast, static methods like QuRating require more compute for selection yet fail to outperform OPUS.

### 6.7 Qualitative comparison of selected samples.

We show the selection from a single candidate buffer of size N=32N{=}32 and selected K=16K{=}16 samples. For each method, we show selected candidates and not selected samples, candidate index, and the method’s raw score (see Appendix [A](#A1 "Appendix A Qualitative Results ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration")). Overall, OPUS tends to select a more diverse mixture of documents, covering both instructional content and broader web text, rather than concentrating on a narrow “educational-only” slice. In contrast, several static filtering method exhibit more extreme preferences—either strongly favoring highly low-diversity patterns or focusing on a limited subset of high-loss samples. These examples support our empirical findings: OPUS’s optimizer-aware utility and stochastic sampling encourage selections that remain broadly suitable for general-purpose pre-training, while still being guided towards high quality samples that align with the proxy direction.

## 7 Conclusion and Future work

We introduced OPUS, a dynamic data selection framework for LLM pre-training that aligns training-time selection with the optimizer’s effective update geometry. Across model scales, optimizers, and corpus quality settings, OPUS consistently improves compute-matched pre-training, suggesting that selection can be substantially strengthened by accounting for how the optimizer actually moves parameters. A natural next step is to extend this optimizer-aligned idea to richer training regimes, such as data mixtures.

## Acknowledgements

We thank Jiachen T. Wang at Princeton University and Meng Ding at University at Buffalo for helpful feedback and discussions.

## References

* J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. (2023)
  Gpt-4 technical report.
  arXiv preprint arXiv:2303.08774.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* Z. Ankner, C. Blakeney, K. Sreenivasan, M. Marion, M. L. Leavitt, and M. Paul (2025)
  Perplexed by perplexity: perplexity-based data pruning with small reference models.
  In The Thirteenth International Conference on Learning Representations,
  External Links: [Link](https://openreview.net/forum?id=1GTARJhxtq)
  Cited by: [§2](#S2.p2.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§6.1](#S6.SS1.p16.1 "6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* A. Anthropic (2024)
  The claude 3 model family: opus, sonnet, haiku.
  Claude-3 Model Card 1 (1),  pp. 4.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* Y. Bisk, R. Zellers, R. LeBras, J. Gao, and Y. Choi (2020)
  PIQA: reasoning about physical commonsense in natural language.
  In AAAI,
   pp. 7432–7439.
  External Links: [Link](https://aaai.org/ojs/index.php/AAAI/article/view/6239)
  Cited by: [4th item](#S6.I1.i4.p1.1 "In 6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. (2020)
  Language models are few-shot learners.
  Advances in neural information processing systems 33,  pp. 1877–1901.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* H. Cai, X. Cai, J. Chang, S. Li, L. Yao, C. Wang, Z. Gao, H. Wang, Y. Li, M. Lin, S. Yang, J. Wang, M. Xu, J. Huang, X. Fang, J. Zhuang, Y. Yin, Y. Li, L. Zhang, and G. Ke (2024)
  SciAssess: benchmarking llm proficiency in scientific literature analysis.
  arXiv preprint arXiv:2403.01976.
  External Links: [Document](https://dx.doi.org/10.48550/arXiv.2403.01976),
  [Link](https://arxiv.org/abs/2403.01976)
  Cited by: [§6.1](#S6.SS1.p15.3 "6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord (2018)
  Think you have solved question answering? try arc, the ai2 reasoning challenge.
  arXiv preprint arXiv:1803.05457.
  Cited by: [7th item](#S6.I1.i7.p1.1 "In 6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* O. Contributors (2023)
  OpenCompass: a universal evaluation platform for foundation models.
  Note: <https://github.com/open-compass/opencompass>
  Cited by: [§6.1](#S6.SS1.p12.1 "6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* G. Cormode and S. Muthukrishnan (2005)
  An improved data stream summary: the count-min sketch and its applications.
  Journal of Algorithms 55 (1),  pp. 58–75.
  Cited by: [3rd item](#S1.I1.i3.p1.1 "In 1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§5.2](#S5.SS2.p3.13 "5.2 Scalable Utility Estimation ‣ 5 Methodology: OPUS ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* J. Deng, Y. Hu, P. Hu, T. Li, S. Liu, J. T. Wang, D. Ley, Q. Dai, B. Huang, J. Huang, C. Jiao, H. A. Just, Y. Pan, J. Shen, Y. Tu, W. Wang, X. Wang, S. Zhang, S. Zhang, R. Jia, H. Lakkaraju, H. Peng, W. Tang, C. Xiong, J. Zhao, H. Tong, H. Zhao, and J. W. Ma (2025)
  A Survey of Data Attribution: Methods, Applications, and Evaluation in the Era of Generative AI.
  Note: working paper or preprint
  External Links: [Link](https://hal.science/hal-05230469)
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* Q. Du, C. Zong, and J. Zhang (2023)
  Mods: model-oriented data selection for instruction tuning.
  arXiv preprint arXiv:2311.15653.
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* A. Ghorbani and J. Zou (2019)
  Data shapley: equitable valuation of data for machine learning.
  In International conference on machine learning,
   pp. 2242–2251.
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi, X. Zhang, X. Yu, Y. Wu, Z. F. Wu, Z. Gou, Z. Shao, Z. Li, Z. Gao, A. Liu, B. Xue, B. Wang, B. Wu, B. Feng, C. Lu, C. Zhao, C. Deng, C. Ruan, D. Dai, D. Chen, D. Ji, E. Li, F. Lin, F. Dai, F. Luo, G. Hao, G. Chen, G. Li, H. Zhang, H. Xu, H. Ding, H. Gao, H. Qu, H. Li, J. Guo, J. Li, J. Chen, J. Yuan, J. Tu, J. Qiu, J. Li, J. L. Cai, J. Ni, J. Liang, J. Chen, K. Dong, K. Hu, K. You, K. Gao, K. Guan, K. Huang, K. Yu, L. Wang, L. Zhang, L. Zhao, L. Wang, L. Zhang, L. Xu, L. Xia, M. Zhang, M. Zhang, M. Tang, M. Zhou, M. Li, M. Wang, M. Li, N. Tian, P. Huang, P. Zhang, Q. Wang, Q. Chen, Q. Du, R. Ge, R. Zhang, R. Pan, R. Wang, R. J. Chen, R. L. Jin, R. Chen, S. Lu, S. Zhou, S. Chen, S. Ye, S. Wang, S. Yu, S. Zhou, S. Pan, S. S. Li, S. Zhou, S. Wu, T. Yun, T. Pei, T. Sun, T. Wang, W. Zeng, W. Liu, W. Liang, W. Gao, W. Yu, W. Zhang, W. L. Xiao, W. An, X. Liu, X. Wang, X. Chen, X. Nie, X. Cheng, X. Liu, X. Xie, X. Liu, X. Yang, X. Li, X. Su, X. Lin, X. Q. Li, X. Jin, X. Shen, X. Chen, X. Sun, X. Wang, X. Song, X. Zhou, X. Wang, X. Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. Zhang, Y. Xu, Y. Li, Y. Zhao, Y. Sun, Y. Wang, Y. Yu, Y. Zhang, Y. Shi, Y. Xiong, Y. He, Y. Piao, Y. Wang, Y. Tan, Y. Ma, Y. Liu, Y. Guo, Y. Ou, Y. Wang, Y. Gong, Y. Zou, Y. He, Y. Xiong, Y. Luo, Y. You, Y. Liu, Y. Zhou, Y. X. Zhu, Y. Huang, Y. Li, Y. Zheng, Y. Zhu, Y. Ma, Y. Tang, Y. Zha, Y. Yan, Z. Z. Ren, Z. Ren, Z. Sha, Z. Fu, Z. Xu, Z. Xie, Z. Zhang, Z. Hao, Z. Ma, Z. Yan, Z. Wu, Z. Gu, Z. Zhu, Z. Liu, Z. Li, Z. Xie, Z. Song, Z. Pan, Z. Huang, Z. Xu, Z. Zhang, and Z. Zhang (2025)
  DeepSeek-r1 incentivizes reasoning in llms through reinforcement learning.
  Nature 645 (8081),  pp. 633–638.
  External Links: ISSN 1476-4687,
  [Link](http://dx.doi.org/10.1038/s41586-025-09422-z),
  [Document](https://dx.doi.org/10.1038/s41586-025-09422-z)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* H. Guo, N. Rajani, P. Hase, M. Bansal, and C. Xiong (2021)
  Fastif: scalable influence functions for efficient model interpretation and debugging.
  In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing,
   pp. 10333–10350.
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* N. Habib, C. Fourrier, H. Kydlíček, T. Wolf, and L. Tunstall (2023)
  LightEval: a lightweight framework for llm evaluation.
  External Links: [Link](https://github.com/huggingface/lighteval)
  Cited by: [§6.1](#S6.SS1.p12.1 "6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* Z. Hammoudeh and D. Lowd (2024)
  Training data influence analysis and estimation: a survey.
  Machine Learning 113 (5),  pp. 2351–2403.
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt (2021)
  Measuring massive multitask language understanding.
  In International Conference on Learning Representations,
  External Links: [Link](https://openreview.net/forum?id=d7KBjmI3GmQ)
  Cited by: [1st item](#S6.I1.i1.p1.1 "In 6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas, L. A. Hendricks, J. Welbl, A. Clark, et al. (2022)
  Training compute-optimal large language models.
  External Links: 2203.15556,
  [Link](https://arxiv.org/abs/2203.15556)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* Y. Hu, P. Hu, H. Zhao, and J. Ma (2024)
  Most influential subset selection: challenges, promises, and beyond.
  Advances in Neural Information Processing Systems 37,  pp. 119778–119810.
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* Y. Hu, F. Wu, H. Ye, D. Forsyth, J. Zou, N. Jiang, J. W. Ma, and H. Zhao (2025)
  A snapshot of influence: a local data attribution framework for online reinforcement learning.
  In The Thirty-ninth Annual Conference on Neural Information Processing Systems,
  External Links: [Link](https://openreview.net/forum?id=sYK4yPDuT1)
  Cited by: [§5.2](#S5.SS2.p1.1 "5.2 Scalable Utility Estimation ‣ 5 Methodology: OPUS ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* Z. Huang, Z. Wang, S. Xia, X. Li, H. Zou, R. Xu, R. Fan, L. Ye, E. Chern, Y. Ye, Y. Zhang, Y. Yang, T. Wu, B. Wang, S. Sun, Y. Xiao, Y. Li, F. Zhou, S. Chern, Y. Qin, Y. Ma, J. Su, Y. Liu, Y. Zheng, S. Zhang, D. Lin, Y. Qiao, and P. Liu (2024)
  OlympicArena: benchmarking multi-discipline cognitive reasoning for superintelligent ai.
  arXiv preprint arXiv:2406.12753.
  External Links: [Document](https://dx.doi.org/10.48550/arXiv.2406.12753),
  [Link](https://arxiv.org/abs/2406.12753)
  Cited by: [§6.1](#S6.SS1.p15.3 "6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* R. Jia, F. Wu, X. Sun, J. Xu, D. Dao, B. Kailkhura, C. Zhang, B. Li, and D. Song (2021)
  Scalability vs. utility: do we have to sacrifice one for the other in data importance quantification?.
  In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
   pp. 8239–8247.
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* K. Jordan, Y. Jin, V. Boza, J. You, F. Cesista, L. Newhouse, and J. Bernstein (2024)
  Muon: an optimizer for hidden layers in neural networks.
  External Links: [Link](https://kellerjordan.github.io/posts/muon/)
  Cited by: [1st item](#S1.I1.i1.p1.1 "In 1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§1](#S1.p2.1 "1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§6.1](#S6.SS1.p3.1 "6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* A. Katharopoulos and F. Fleuret (2019)
  Not all samples are created equal: deep learning with importance sampling.
  External Links: 1803.00942,
  [Link](https://arxiv.org/abs/1803.00942)
  Cited by: [§2](#S2.p2.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* K. Killamsetty, S. Durga, G. Ramakrishnan, A. De, and R. Iyer (2021)
  Grad-match: gradient matching based data subset selection for efficient deep model training.
  In International Conference on Machine Learning,
   pp. 5464–5474.
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* P. W. Koh and P. Liang (2017)
  Understanding black-box predictions via influence functions.
  In International conference on machine learning,
   pp. 1885–1894.
  Cited by: [§2](#S2.p2.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§2](#S2.p3.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* G. Lai, Q. Xie, H. Liu, Y. Yang, and E. Hovy (2017)
  RACE: large-scale ReAding comprehension dataset from examinations.
  In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, M. Palmer, R. Hwa, and S. Riedel (Eds.),
  Copenhagen, Denmark,  pp. 785–794.
  External Links: [Link](https://aclanthology.org/D17-1082/),
  [Document](https://dx.doi.org/10.18653/v1/D17-1082)
  Cited by: [2nd item](#S6.I2.i2.p1.1 "In 6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* H. J. Levesque, E. Davis, and L. Morgenstern (2012)
  The winograd schema challenge..
  KR 2012 (13th),  pp. 3.
  Cited by: [9th item](#S6.I1.i9.p1.1 "In 6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* J. Li, A. Fang, G. Smyrnis, M. Ivgi, M. Jordan, S. Y. Gadre, H. Bansal, E. Guha, S. S. Keh, K. Arora, et al. (2024)
  Datacomp-lm: in search of the next generation of training sets for language models.
  Advances in Neural Information Processing Systems 37,  pp. 14200–14282.
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§2](#S2.p1.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§6.1](#S6.SS1.p16.1 "6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.19.17.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.30.28.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.41.39.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.8.6.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, et al. (2024a)
  Deepseek-v3 technical report.
  arXiv preprint arXiv:2412.19437.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* W. Liu, W. Zeng, K. He, Y. Jiang, and J. He (2024b)
  What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning.
  In The Twelfth International Conference on Learning Representations,
  External Links: [Link](https://openreview.net/forum?id=BTKAeLqLMw)
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* I. Loshchilov and F. Hutter (2016)
  Online batch selection for faster training of neural networks.
  External Links: 1511.06343,
  [Link](https://arxiv.org/abs/1511.06343)
  Cited by: [§2](#S2.p2.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* I. Loshchilov and F. Hutter (2019)
  Decoupled weight decay regularization.
  In International Conference on Learning Representations,
  External Links: [Link](https://arxiv.org/abs/1711.05101)
  Cited by: [1st item](#S1.I1.i1.p1.1 "In 1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§1](#S1.p2.1 "1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§6.1](#S6.SS1.p3.1 "6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* N. Mostafazadeh, N. Chambers, X. He, D. Parikh, D. Batra, L. Vanderwende, P. Kohli, and J. Allen (2016)
  A corpus and cloze evaluation for deeper understanding of commonsense stories.
  In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, K. Knight, A. Nenkova, and O. Rambow (Eds.),
  San Diego, California,  pp. 839–849.
  External Links: [Link](https://aclanthology.org/N16-1098/),
  [Document](https://dx.doi.org/10.18653/v1/N16-1098)
  Cited by: [4th item](#S6.I2.i4.p1.1 "In 6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* Y. Nie, A. Williams, E. Dinan, M. Bansal, J. Weston, and D. Kiela (2020)
  Adversarial NLI: a new benchmark for natural language understanding.
  In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault (Eds.),
  Online,  pp. 4885–4901.
  External Links: [Link](https://aclanthology.org/2020.acl-main.441/),
  [Document](https://dx.doi.org/10.18653/v1/2020.acl-main.441)
  Cited by: [2nd item](#S6.I1.i2.p1.1 "In 6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* M. Paul, S. Ganguli, and G. K. Dziugaite (2021)
  Deep learning on a data diet: finding important examples early in training.
  Advances in neural information processing systems 34,  pp. 20596–20607.
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* G. Penedo, H. Kydlíček, A. Lozhkov, M. Mitchell, C. A. Raffel, L. Von Werra, T. Wolf, et al. (2024)
  The fineweb datasets: decanting the web for the finest text data at scale.
  Advances in Neural Information Processing Systems 37,  pp. 30811–30849.
  Cited by: [5th item](#S1.I1.i5.p1.1 "In 1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§1](#S1.p2.1 "1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§2](#S2.p1.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§6.1](#S6.SS1.p12.1 "6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§6.1](#S6.SS1.p16.1 "6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§6.1](#S6.SS1.p9.1 "6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.20.18.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.31.29.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.42.40.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.9.7.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* G. Pruthi, F. Liu, S. Kale, and M. Sundararajan (2020)
  Estimating training data influence by tracing gradient descent.
  Advances in Neural Information Processing Systems 33,  pp. 19920–19930.
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al. (2019)
  Language models are unsupervised multitask learners.
  OpenAI blog 1 (8),  pp. 9.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§6.1](#S6.SS1.p1.6 "6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi (2020)
  WinoGrande: an adversarial winograd schema challenge at scale.
  In AAAI,
   pp. 8732–8740.
  External Links: [Link](https://aaai.org/ojs/index.php/AAAI/article/view/6399)
  Cited by: [6th item](#S6.I1.i6.p1.1 "In 6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* M. Sap, H. Rashkin, D. Chen, R. LeBras, and Y. Choi (2019)
  Socialiqa: commonsense reasoning about social interactions.
  arXiv preprint arXiv:1904.09728.
  Cited by: [5th item](#S6.I1.i5.p1.1 "In 6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* SciencePedia Team (2025)
  SciencePedia dataset.
  Note: <https://sciencepedia.bohrium.com>Accessed: 2026-01-20
  Cited by: [5th item](#S1.I1.i5.p1.1 "In 1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§6.1](#S6.SS1.p9.1 "6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* M. Suzgun, N. Scales, N. Schärli, S. Gehrmann, Y. Tay, H. W. Chung, A. Chowdhery, Q. Le, E. Chi, D. Zhou, et al. (2023)
  Challenging big-bench tasks and whether chain-of-thought can solve them.
  In Findings of the Association for Computational Linguistics: ACL 2023,
   pp. 13003–13051.
  Cited by: [1st item](#S6.I2.i1.p1.1 "In 6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* A. Talmor, J. Herzig, N. Lourie, and J. Berant (2019)
  Commonsenseqa: a question answering challenge targeting commonsense knowledge.
  In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers),
   pp. 4149–4158.
  Cited by: [8th item](#S6.I1.i8.p1.1 "In 6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* M. Toneva, A. Sordoni, R. T. des Combes, A. Trischler, Y. Bengio, and G. J. Gordon (2019)
  An empirical study of example forgetting during deep neural network learning.
  In International Conference on Learning Representations,
  External Links: [Link](https://openreview.net/forum?id=BJlxm30cKm)
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* P. Villalobos, A. Ho, J. Sevilla, T. Besiroglu, L. Heim, and M. Hobbhahn (2022)
  Will we run out of data? limits of llm scaling based on human-generated data.
  arXiv preprint arXiv:2211.04325.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* A. Wang, Y. Pruksachatkun, N. Nangia, A. Singh, J. Michael, F. Hill, O. Levy, and S. Bowman (2019)
  Superglue: a stickier benchmark for general-purpose language understanding systems.
  Advances in neural information processing systems 32.
  Cited by: [3rd item](#S6.I2.i3.p1.1 "In 6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* J. T. Wang, P. Mittal, D. Song, and R. Jia (2025a)
  Data shapley in one training run.
  In The Thirteenth International Conference on Learning Representations,
  External Links: [Link](https://openreview.net/forum?id=HD6bWcj87Y)
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§2](#S2.p3.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§5.2](#S5.SS2.p1.1 "5.2 Scalable Utility Estimation ‣ 5 Methodology: OPUS ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* J. T. Wang, D. Song, J. Zou, P. Mittal, and R. Jia (2025b)
  Capturing the temporal dependence of training data influence.
  In The Thirteenth International Conference on Learning Representations,
  External Links: [Link](https://openreview.net/forum?id=uHLgDEgiS5)
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* J. T. Wang, T. Wu, D. Song, P. Mittal, and R. Jia (2024)
  Greats: online selection of high-quality data for llm training in every iteration.
  Advances in Neural Information Processing Systems 37,  pp. 131197–131223.
  Cited by: [3rd item](#S1.I1.i3.p1.1 "In 1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§1](#S1.p2.1 "1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§2](#S2.p2.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§5.1](#S5.SS1.p1.3 "5.1 Optimizer-Induced Utility Objective ‣ 5 Methodology: OPUS ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§5.1](#S5.SS1.p7.3 "5.1 Optimizer-Induced Utility Objective ‣ 5 Methodology: OPUS ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§5.2](#S5.SS2.p1.1 "5.2 Scalable Utility Estimation ‣ 5 Methodology: OPUS ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§5.2](#S5.SS2.p2.12 "5.2 Scalable Utility Estimation ‣ 5 Methodology: OPUS ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§6.1](#S6.SS1.p16.1 "6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.16.14.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.27.25.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.38.36.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.5.3.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* Y. Wang, Z. Fu, J. Cai, P. Tang, H. Lyu, Y. Fang, Z. Zheng, J. Zhou, G. Zeng, C. Xiao, et al. (2025c)
  Ultra-fineweb: efficient data filtering and verification for high-quality llm training data.
  arXiv preprint arXiv:2505.05427.
  Cited by: [§2](#S2.p1.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§6.1](#S6.SS1.p16.1 "6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.10.8.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.21.19.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.32.30.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.43.41.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* A. Wettig, A. Gupta, S. Malik, and D. Chen (2024)
  QuRating: selecting high-quality data for training language models.
  In International Conference on Machine Learning,
   pp. 52915–52971.
  Cited by: [§2](#S2.p1.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§6.1](#S6.SS1.p16.1 "6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.17.15.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.28.26.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.39.37.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.6.4.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* A. Wettig, K. Lo, S. Min, H. Hajishirzi, D. Chen, and L. Soldaini (2025)
  Organize the web: constructing domains enhances pre-training data curation.
  In Forty-second International Conference on Machine Learning,
  External Links: [Link](https://openreview.net/forum?id=boSqwdvJVC)
  Cited by: [§6.3](#S6.SS3.p7.1 "6.3 Pre-training from Scratch ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 6](#S6.T6 "In 6.3 Pre-training from Scratch ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* M. Xia, S. Malladi, S. Gururangan, S. Arora, and D. Chen (2024)
  LESS: selecting influential data for targeted instruction tuning.
  In Proceedings of the 41st International Conference on Machine Learning,
   pp. 54104–54132.
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* S. M. Xie, S. Santurkar, T. Ma, and P. S. Liang (2023)
  Data selection for language models via importance resampling.
  Advances in Neural Information Processing Systems 36,  pp. 34201–34227.
  Cited by: [§2](#S2.p1.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§6.1](#S6.SS1.p16.1 "6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.18.16.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.29.27.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.40.38.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [Table 3](#S6.T3.3.1.7.5.1 "In 6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. (2025)
  Qwen3 technical report.
  arXiv preprint arXiv:2505.09388.
  Cited by: [5th item](#S1.I1.i5.p1.1 "In 1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§1](#S1.p1.1 "1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"),
  [§6.1](#S6.SS1.p1.6 "6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* A. Yang, B. Yang, B. Hui, B. Zheng, B. Yu, C. Zhou, C. Li, C. Li, D. Liu, F. Huang, et al. (2024a)
  Qwen2 technical report.
  CoRR.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, et al. (2024b)
  Qwen2. 5 technical report.
  arXiv preprint arXiv:2412.15115.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* C. Yeh, J. Kim, I. E. Yen, and P. K. Ravikumar (2018)
  Representer point selection for explaining deep neural networks.
  Advances in neural information processing systems 31.
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* P. Yu, L. Merrick, G. Nuti, and D. Campos (2024a)
  Arctic-embed 2.0: multilingual retrieval without compromise.
  External Links: 2412.04506,
  [Link](https://arxiv.org/abs/2412.04506)
  Cited by: [§6.2](#S6.SS2.p2.1 "6.2 Bench-proxy construction ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* Z. Yu, S. Das, and C. Xiong (2024b)
  Mates: model-aware data selection for efficient pretraining with data influence models.
  Advances in Neural Information Processing Systems 37,  pp. 108735–108759.
  Cited by: [§2](#S2.p2.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* Z. Yu, S. Das, and C. Xiong (2025)
  Group-level data selection for efficient pretraining.
  External Links: 2502.14709,
  [Link](https://arxiv.org/abs/2502.14709)
  Cited by: [§2](#S2.p2.1 "2 Related Work ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").
* R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi (2019)
  HellaSwag: can a machine really finish your sentence?.
  In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, A. Korhonen, D. Traum, and L. Màrquez (Eds.),
  Florence, Italy,  pp. 4791–4800.
  External Links: [Link](https://aclanthology.org/P19-1472/),
  [Document](https://dx.doi.org/10.18653/v1/P19-1472)
  Cited by: [3rd item](#S6.I1.i3.p1.1 "In 6.1 Experimental Setup ‣ 6 Experiments ‣ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration").

## Appendix A Qualitative Results

Random
  



Sample 1 Selected
  
Candidate #0 score=--
  
As it turns out, the exercises synonymous with strong, attractive abs may not be the best way to train your core—and may be doing damage to your back. Read more If you are worried about the excess holiday pounds many of us are still carrying around. There are a few easy, natural things you can do to shed them, and none of them requires an…



Sample 2 Selected
  
Candidate #1 score=--
  
Wedding & Party Venues - Sort By: Edgartown : (508) 627-9510 A 19th century gothic revival home transformed into the island’s premier eco-boutique hotel. Guests either stay in the 17-room Hob Knob hotel or in the privacy of their own Hob Knob House. Guests can expect individualized Hob Knob hospitality and modern luxury amenities in a rel…

Sample 3 Selected
  
Candidate #2 score=--
  
With the advent of new technologies for sneakers such as Vac Tech, Hyperfuse and Flyknit, the mid 90s and early 2000s methods of production and designing are becoming obsolete in this sneaker world. Nike Running is the future for Nike, generating billions of dollars per year, and we see Nike also not afraid to experiment with technology s…

Sample 4 Selected
  
Candidate #3 score=--
  
starring John Travolta and Sam Jackson The first thing to understand about Basic –the basic thing, let’s say– is that although the commercials make it look like a war movie, it is not, for which we can all be grateful. No, Basic is a plot-twisty whodunnit. If The Usual Suspects died, and its body turned to cheese, and then that cheese-b…

Sample 5 Selected
  
Candidate #4 score=--
  
5 Types of Women’s Underwear That Men Love Underwear can say a lot about a woman. It’s something that men are obsessed with, to the point that, a mere glimpse of a thong waistband causes us to go into shock. On the surface we find them sexy, revealing. We’re able to see who a woman actually is—or maybe some guys are just plain horny. Howe…

Sample 6 Selected
  
Candidate #7 score=--
  
Elizabeth Hurley played as Dalila Release: Dec 8, 1996 Mara and her husband Manoa are both upstanding and religious Israelites living under the harsh and unjust rule of the Philistines. Much to their regret, they have not been able to have children. One day, a mysterious stranger appears to Mara and promises her that she will bear a son w…

Sample 7 Selected
  
Candidate #8 score=--
  
The Unsung Heroes of Your HVAC System: Understanding the Importance of Filters When it comes to your HVAC (Heating, Ventilation, and Air Conditioning) system, you might be quick to think about the thermostat, air ducts, or even the unit itself. However, there’s an unsung hero in your HVAC system that plays a pivotal role in maintaining in…

Sample 8 Selected
  
Candidate #13 score=--
  
In Heart of Darkness it is the white invaders for instance, who are, almost without exception, embodiments of blindness, selfishness, and cruelty; and even in the cognitive domain, where such positive phrases as “to enlighten,” for instance, are conventionally opposed to negative ones such as “to be in the dark,” the traditional expectati…

Sample 9 Selected
  
Candidate #17 score=--
  
Political Parties and Elections Political parties are an established part of modern mass democracy, and the conduct of elections in India is largely dependent on the behaviour of political parties. Although many candidates for Indian elections are independent, the winning candidates for Lok Sabha and Vidhan Sabha elections usually stand a…

Sample 10 Selected
  
Candidate #18 score=--
  
This article originally appeared in the December 2015 issue of Resource Recycling. Subscribe today for access to all print content. Since the 1990s, curbside and drop-off recycling has grown substantially – nearly 90 percent of households now have access, according to recent surveys from Moore Recycling Associates, the American Forest and…

Sample 11 Selected
  
Candidate #21 score=--
  
Nestled in the leafy suburbs of western Berlin, the Wannsee Conference House stands as a poignant reminder of a dark chapter in human history. The Wannsee Conference: A Pivotal Moment The Wannsee Conference, held on January 20, 1942, marked a pivotal moment in the implementation of Nazi Germany’s genocidal plans. Organized by SS-Obergrupp…

Sample 12 Selected
  
Candidate #23 score=--
  
The St. James kindergarteners have been working up to Project Week over the past month. We started slowly by taking walks in our neighborhood while Ms. Meghan and I noted what caught the children’s interest. It became apparent that the class was very interested in the L trains that they saw on our walks. It started with a simple question,…

Sample 13 Selected
  
Candidate #27 score=--
  
24/7 writing help on your phone Save to my list Remove from my list In the tumultuous 19th century, both Italy and Germany found themselves fragmented into numerous separate ruling states. The impetus for change came in the form of rising nationalism and liberalism, paving the way for the unification of these disparate entities. However,…

Sample 14 Selected
  
Candidate #29 score=--
  
Earthquakes are the result of sudden movement along faults within the Earth. The movement releases stored-up ‘elastic strain’ energy in the form of seismic waves, which propagate through the Earth and cause the ground surface to shake. Such movement on the faults is generally a response to long-term deformation and the buildup of stress.…

Sample 15 Selected
  
Candidate #30 score=--
  
Over 1.8 million professionals use CFI to learn accounting, financial analysis, modeling and more. Start with a free account to explore 20+ always-free courses and hundreds of finance templates and cheat sheets. What is the Central Limit Theorem (CLT)? The Central Limit Theorem (CLT) is a statistical concept that states that the sample me…

Sample 16 Selected
  
Candidate #31 score=--
  
One of the challenges of working with ancient DNA samples is that damage accumulates over time, breaking the double helix structure into ever-smaller fragments. In the samples we worked with, these fragments were scattered and mixed with contaminants, making genome reconstruction a major technical challenge. But a shocking paper published…

Sample 17 Not selected
  
Candidate #5 score=--
  
Well this is the big one. So big apparently, that I had to take it there and raise the number from 10 to 15. There’s just that many fails in the world of female rap. Some slight missteps, some EPIC. Nevertheless, they are all worth mentioning. You can probably think of a bunch more, but this is what I have gathered picking up from my prev…

Sample 18 Not selected
  
Candidate #6 score=--
  
Skaters need to check their skate helmets every so often and ask yourself, ”Is it time to replace this helmet?” Well, that depends. Did you crash in it? For starters, most people are aware that you must replace a helmet after any crash where your head hit. The foam part of a helmet is made for one-time use, and after crushing once it is n…

Sample 19 Not selected
  
Candidate #9 score=--
  
“Last night three cargoes of Bohea Tea were emptied into the sea. This is the most magnificent movement of all. There is a dignity, a majesty, a sublimity, in this last effort of the Patriots that I greatly admire.” - John Adams, diary entry, December 17, 1773 - John Adams, diary entry, December 17, 1773 A Novel Idea Is something so new a…

Sample 20 Not selected
  
Candidate #10 score=--
  
Deforestation isn’t just happening in well-known global hotspots like Indonesia and Brazil’s rainforest. A new analysis says forests are also shrinking on state and private land in Oregon, where an estimated 522,000 acres of forest cover have disappeared since 2000. That’s an area six times larger than the city of Portland, equal to more…

Sample 21 Not selected
  
Candidate #11 score=--
  
In decades past, classroom design was often an afterthought and followed a standardised layout. Plain boxed shaped classrooms, with identical chairs and tables throughout were commonplace in many schools. Read the latest issue of School News HERE Recently, though, there has been a shift away from this one-size-fits all approach to classro…

Sample 22 Not selected
  
Candidate #12 score=--
  
Can you please give us a little short bio? (education, professional experiences, select publications, academic specialty, awards won) Public school teacher for 5 years BA art (UC Irvine) PhD. (UCLA) educational psychology Professor of Child Development, (25 years) CSUS Senior Research Scientist (Oregon Research Institute with Institute of…

Sample 23 Not selected
  
Candidate #14 score=--
  
Is your major sustainable enough? Whether you’re pursuing a sustainability degree and want to further your knowledge, or are interested in supplementing your major in another area with sustainability education, plenty of independent learning resources are available. A wide range of credit and noncredit courses—including university- and or…

Sample 24 Not selected
  
Candidate #15 score=--
  
Origami is an art form that combines precision, creativity, and patience. While basic origami is obtainable to every one, mastering complex origami designs can be quite a rewarding and impressive achievement. In this article, we’ll show you with the procedure for creating intricate origami while highlighting essential techniques for achie…

Sample 25 Not selected
  
Candidate #16 score=--
  
What is rotavirus and why does my baby need to be immunised? Rotavirus is a very infectious virus that causes the majority of serious cases of gastroenteritis in babies. It causes diarrhoea, vomiting and abdominal pain, usually lasting around a week. Most children will be infected by rotavirus once by the age of five. Gastroenteritis (cau…

Sample 26 Not selected
  
Candidate #19 score=--
  
Dividing Fractions Using Models Worksheet. This worksheet has six division with fractions issues to be solved — three must be solved with fashions and three with algorithms — options are on the second page. Answer key divide the unit fractions by whole numbers using th e fashions given. Use these resources to help reinforce the following…

Sample 27 Not selected
  
Candidate #20 score=--
  
Conduct Disorder (CD) is a complex and serious behavioural and emotional disorder that can occur in children and adolescents. It’s characterised by a repetitive and persistent pattern of behaviour where the basic rights of others or major age-appropriate societal norms or rules are violated. Here’s an outline of Conduct Disorder in line w…

Sample 28 Not selected
  
Candidate #22 score=--
  
How To Choose Decodable Readers for First Grade To decode or not to decode: really, there is no question. To help rising first graders become successful and enthusiastic readers this summer, decodable readers are essential reading resources. Although “decodable text” might sound like yet another form of educational lingo, parents and educ…

Sample 29 Not selected
  
Candidate #24 score=--
  
Next we will talk about solar radiation, that is, the forms of solar radiation that we receive on earth. Solar radiation is generated by a series of nuclear fusion reactions that occur in the Sun and, as a consequence, emit electromagnetic radiation that reaches the earth. This radiation received by the earth’s surface is measured in W /…

Sample 30 Not selected
  
Candidate #25 score=--
  
KS2 Maths is an important core subject in the National Curriculum and this area of the website covers all the major aspects of the curriculum including numbers, calculations, problems and measures. Each subject area is designed to help children develop their knowledge, whether they are learning in a classroom or home schooling environment…

Sample 31 Not selected
  
Candidate #26 score=--
  
Unveiling the Power: Key Provisions of the Civil Rights Act of 1864 What were the Civil Rights Act of 1864’s key provisions? The Civil Rights Act of 1864 was a pivotal moment in American history, establishing crucial legal protections for African Americans in the face of rampant discrimination. Editor Note: The Civil Rights Act of 1864 la…

Sample 32 Not selected
  
Candidate #28 score=--
  
You really have to be alert when studying science. Galaxies were created after matter. The stars in those galaxies were supposed to move slowly because there was more mass in the center of the galaxy. However, after dark matter was added, the stars appeared to move faster; however, this is not the case in our galaxy, suggesting that there…

OPUS
  



Sample 1 Selected
  
Candidate #8 score=0.00589
  
The Unsung Heroes of Your HVAC System: Understanding the Importance of Filters When it comes to your HVAC (Heating, Ventilation, and Air Conditioning) system, you might be quick to think about the thermostat, air ducts, or even the unit itself. However, there’s an unsung hero in your HVAC system that plays a pivotal role in maintaining in…



Sample 2 Selected
  
Candidate #22 score=0.00471
  
How To Choose Decodable Readers for First Grade To decode or not to decode: really, there is no question. To help rising first graders become successful and enthusiastic readers this summer, decodable readers are essential reading resources. Although “decodable text” might sound like yet another form of educational lingo, parents and educ…

Sample 3 Selected
  
Candidate #27 score=0.00466
  
24/7 writing help on your phone Save to my list Remove from my list In the tumultuous 19th century, both Italy and Germany found themselves fragmented into numerous separate ruling states. The impetus for change came in the form of rising nationalism and liberalism, paving the way for the unification of these disparate entities. However,…

Sample 4 Selected
  
Candidate #4 score=0.0046
  
5 Types of Women’s Underwear That Men Love Underwear can say a lot about a woman. It’s something that men are obsessed with, to the point that, a mere glimpse of a thong waistband causes us to go into shock. On the surface we find them sexy, revealing. We’re able to see who a woman actually is—or maybe some guys are just plain horny. Howe…

Sample 5 Selected
  
Candidate #18 score=0.0044
  
This article originally appeared in the December 2015 issue of Resource Recycling. Subscribe today for access to all print content. Since the 1990s, curbside and drop-off recycling has grown substantially – nearly 90 percent of households now have access, according to recent surveys from Moore Recycling Associates, the American Forest and…

Sample 6 Selected
  
Candidate #30 score=0.0042
  
Over 1.8 million professionals use CFI to learn accounting, financial analysis, modeling and more. Start with a free account to explore 20+ always-free courses and hundreds of finance templates and cheat sheets. What is the Central Limit Theorem (CLT)? The Central Limit Theorem (CLT) is a statistical concept that states that the sample me…

Sample 7 Selected
  
Candidate #0 score=0.0042
  
As it turns out, the exercises synonymous with strong, attractive abs may not be the best way to train your core—and may be doing damage to your back. Read more If you are worried about the excess holiday pounds many of us are still carrying around. There are a few easy, natural things you can do to shed them, and none of them requires an…

Sample 8 Selected
  
Candidate #23 score=0.00418
  
The St. James kindergarteners have been working up to Project Week over the past month. We started slowly by taking walks in our neighborhood while Ms. Meghan and I noted what caught the children’s interest. It became apparent that the class was very interested in the L trains that they saw on our walks. It started with a simple question,…

Sample 9 Selected
  
Candidate #31 score=0.00411
  
One of the challenges of working with ancient DNA samples is that damage accumulates over time, breaking the double helix structure into ever-smaller fragments. In the samples we worked with, these fragments were scattered and mixed with contaminants, making genome reconstruction a major technical challenge. But a shocking paper published…

Sample 10 Selected
  
Candidate #11 score=0.00401
  
In decades past, classroom design was often an afterthought and followed a standardised layout. Plain boxed shaped classrooms, with identical chairs and tables throughout were commonplace in many schools. Read the latest issue of School News HERE Recently, though, there has been a shift away from this one-size-fits all approach to classro…

Sample 11 Selected
  
Candidate #25 score=0.00396
  
KS2 Maths is an important core subject in the National Curriculum and this area of the website covers all the major aspects of the curriculum including numbers, calculations, problems and measures. Each subject area is designed to help children develop their knowledge, whether they are learning in a classroom or home schooling environment…

Sample 12 Selected
  
Candidate #7 score=0.0039
  
Elizabeth Hurley played as Dalila Release: Dec 8, 1996 Mara and her husband Manoa are both upstanding and religious Israelites living under the harsh and unjust rule of the Philistines. Much to their regret, they have not been able to have children. One day, a mysterious stranger appears to Mara and promises her that she will bear a son w…

Sample 13 Selected
  
Candidate #5 score=0.00389
  
Well this is the big one. So big apparently, that I had to take it there and raise the number from 10 to 15. There’s just that many fails in the world of female rap. Some slight missteps, some EPIC. Nevertheless, they are all worth mentioning. You can probably think of a bunch more, but this is what I have gathered picking up from my prev…

Sample 14 Selected
  
Candidate #9 score=0.00384
  
“Last night three cargoes of Bohea Tea were emptied into the sea. This is the most magnificent movement of all. There is a dignity, a majesty, a sublimity, in this last effort of the Patriots that I greatly admire.” - John Adams, diary entry, December 17, 1773 - John Adams, diary entry, December 17, 1773 A Novel Idea Is something so new a…

Sample 15 Selected
  
Candidate #19 score=0.00376
  
Dividing Fractions Using Models Worksheet. This worksheet has six division with fractions issues to be solved — three must be solved with fashions and three with algorithms — options are on the second page. Answer key divide the unit fractions by whole numbers using th e fashions given. Use these resources to help reinforce the following…

Sample 16 Selected
  
Candidate #1 score=0.00348
  
Wedding & Party Venues - Sort By: Edgartown : (508) 627-9510 A 19th century gothic revival home transformed into the island’s premier eco-boutique hotel. Guests either stay in the 17-room Hob Knob hotel or in the privacy of their own Hob Knob House. Guests can expect individualized Hob Knob hospitality and modern luxury amenities in a rel…

Sample 17 Not selected
  
Candidate #15 score=0.00524
  
Origami is an art form that combines precision, creativity, and patience. While basic origami is obtainable to every one, mastering complex origami designs can be quite a rewarding and impressive achievement. In this article, we’ll show you with the procedure for creating intricate origami while highlighting essential techniques for achie…

Sample 18 Not selected
  
Candidate #20 score=0.00518
  
Conduct Disorder (CD) is a complex and serious behavioural and emotional disorder that can occur in children and adolescents. It’s characterised by a repetitive and persistent pattern of behaviour where the basic rights of others or major age-appropriate societal norms or rules are violated. Here’s an outline of Conduct Disorder in line w…

Sample 19 Not selected
  
Candidate #14 score=0.00472
  
Is your major sustainable enough? Whether you’re pursuing a sustainability degree and want to further your knowledge, or are interested in supplementing your major in another area with sustainability education, plenty of independent learning resources are available. A wide range of credit and noncredit courses—including university- and or…

Sample 20 Not selected
  
Candidate #28 score=0.0046
  
You really have to be alert when studying science. Galaxies were created after matter. The stars in those galaxies were supposed to move slowly because there was more mass in the center of the galaxy. However, after dark matter was added, the stars appeared to move faster; however, this is not the case in our galaxy, suggesting that there…

Sample 21 Not selected
  
Candidate #21 score=0.00457
  
Nestled in the leafy suburbs of western Berlin, the Wannsee Conference House stands as a poignant reminder of a dark chapter in human history. The Wannsee Conference: A Pivotal Moment The Wannsee Conference, held on January 20, 1942, marked a pivotal moment in the implementation of Nazi Germany’s genocidal plans. Organized by SS-Obergrupp…

Sample 22 Not selected
  
Candidate #16 score=0.00456
  
What is rotavirus and why does my baby need to be immunised? Rotavirus is a very infectious virus that causes the majority of serious cases of gastroenteritis in babies. It causes diarrhoea, vomiting and abdominal pain, usually lasting around a week. Most children will be infected by rotavirus once by the age of five. Gastroenteritis (cau…

Sample 23 Not selected
  
Candidate #29 score=0.00448
  
Earthquakes are the result of sudden movement along faults within the Earth. The movement releases stored-up ‘elastic strain’ energy in the form of seismic waves, which propagate through the Earth and cause the ground surface to shake. Such movement on the faults is generally a response to long-term deformation and the buildup of stress.…

Sample 24 Not selected
  
Candidate #13 score=0.00445
  
In Heart of Darkness it is the white invaders for instance, who are, almost without exception, embodiments of blindness, selfishness, and cruelty; and even in the cognitive domain, where such positive phrases as “to enlighten,” for instance, are conventionally opposed to negative ones such as “to be in the dark,” the traditional expectati…

Sample 25 Not selected
  
Candidate #26 score=0.00443
  
Unveiling the Power: Key Provisions of the Civil Rights Act of 1864 What were the Civil Rights Act of 1864’s key provisions? The Civil Rights Act of 1864 was a pivotal moment in American history, establishing crucial legal protections for African Americans in the face of rampant discrimination. Editor Note: The Civil Rights Act of 1864 la…

Sample 26 Not selected
  
Candidate #24 score=0.00439
  
Next we will talk about solar radiation, that is, the forms of solar radiation that we receive on earth. Solar radiation is generated by a series of nuclear fusion reactions that occur in the Sun and, as a consequence, emit electromagnetic radiation that reaches the earth. This radiation received by the earth’s surface is measured in W /…

Sample 27 Not selected
  
Candidate #17 score=0.00427
  
Political Parties and Elections Political parties are an established part of modern mass democracy, and the conduct of elections in India is largely dependent on the behaviour of political parties. Although many candidates for Indian elections are independent, the winning candidates for Lok Sabha and Vidhan Sabha elections usually stand a…

Sample 28 Not selected
  
Candidate #10 score=0.00427
  
Deforestation isn’t just happening in well-known global hotspots like Indonesia and Brazil’s rainforest. A new analysis says forests are also shrinking on state and private land in Oregon, where an estimated 522,000 acres of forest cover have disappeared since 2000. That’s an area six times larger than the city of Portland, equal to more…

Sample 29 Not selected
  
Candidate #6 score=0.00401
  
Skaters need to check their skate helmets every so often and ask yourself, ”Is it time to replace this helmet?” Well, that depends. Did you crash in it? For starters, most people are aware that you must replace a helmet after any crash where your head hit. The foam part of a helmet is made for one-time use, and after crushing once it is n…

Sample 30 Not selected
  
Candidate #2 score=0.00384
  
With the advent of new technologies for sneakers such as Vac Tech, Hyperfuse and Flyknit, the mid 90s and early 2000s methods of production and designing are becoming obsolete in this sneaker world. Nike Running is the future for Nike, generating billions of dollars per year, and we see Nike also not afraid to experiment with technology s…

Sample 31 Not selected
  
Candidate #12 score=0.00369
  
Can you please give us a little short bio? (education, professional experiences, select publications, academic specialty, awards won) Public school teacher for 5 years BA art (UC Irvine) PhD. (UCLA) educational psychology Professor of Child Development, (25 years) CSUS Senior Research Scientist (Oregon Research Institute with Institute of…

Sample 32 Not selected
  
Candidate #3 score=0.00333
  
starring John Travolta and Sam Jackson The first thing to understand about Basic –the basic thing, let’s say– is that although the commercials make it look like a war movie, it is not, for which we can all be grateful. No, Basic is a plot-twisty whodunnit. If The Usual Suspects died, and its body turned to cheese, and then that cheese-b…

High-PPL
  



Sample 1 Selected
  
Candidate #3 score=4.57
  
starring John Travolta and Sam Jackson The first thing to understand about Basic –the basic thing, let’s say– is that although the commercials make it look like a war movie, it is not, for which we can all be grateful. No, Basic is a plot-twisty whodunnit. If The Usual Suspects died, and its body turned to cheese, and then that cheese-b…



Sample 2 Selected
  
Candidate #19 score=4.26
  
Dividing Fractions Using Models Worksheet. This worksheet has six division with fractions issues to be solved — three must be solved with fashions and three with algorithms — options are on the second page. Answer key divide the unit fractions by whole numbers using th e fashions given. Use these resources to help reinforce the following…

Sample 3 Selected
  
Candidate #12 score=4.26
  
Can you please give us a little short bio? (education, professional experiences, select publications, academic specialty, awards won) Public school teacher for 5 years BA art (UC Irvine) PhD. (UCLA) educational psychology Professor of Child Development, (25 years) CSUS Senior Research Scientist (Oregon Research Institute with Institute of…

Sample 4 Selected
  
Candidate #5 score=4.21
  
Well this is the big one. So big apparently, that I had to take it there and raise the number from 10 to 15. There’s just that many fails in the world of female rap. Some slight missteps, some EPIC. Nevertheless, they are all worth mentioning. You can probably think of a bunch more, but this is what I have gathered picking up from my prev…

Sample 5 Selected
  
Candidate #2 score=4.04
  
With the advent of new technologies for sneakers such as Vac Tech, Hyperfuse and Flyknit, the mid 90s and early 2000s methods of production and designing are becoming obsolete in this sneaker world. Nike Running is the future for Nike, generating billions of dollars per year, and we see Nike also not afraid to experiment with technology s…

Sample 6 Selected
  
Candidate #1 score=3.89
  
Wedding & Party Venues - Sort By: Edgartown : (508) 627-9510 A 19th century gothic revival home transformed into the island’s premier eco-boutique hotel. Guests either stay in the 17-room Hob Knob hotel or in the privacy of their own Hob Knob House. Guests can expect individualized Hob Knob hospitality and modern luxury amenities in a rel…

Sample 7 Selected
  
Candidate #7 score=3.89
  
Elizabeth Hurley played as Dalila Release: Dec 8, 1996 Mara and her husband Manoa are both upstanding and religious Israelites living under the harsh and unjust rule of the Philistines. Much to their regret, they have not been able to have children. One day, a mysterious stranger appears to Mara and promises her that she will bear a son w…

Sample 8 Selected
  
Candidate #0 score=3.82
  
As it turns out, the exercises synonymous with strong, attractive abs may not be the best way to train your core—and may be doing damage to your back. Read more If you are worried about the excess holiday pounds many of us are still carrying around. There are a few easy, natural things you can do to shed them, and none of them requires an…

Sample 9 Selected
  
Candidate #13 score=3.79
  
In Heart of Darkness it is the white invaders for instance, who are, almost without exception, embodiments of blindness, selfishness, and cruelty; and even in the cognitive domain, where such positive phrases as “to enlighten,” for instance, are conventionally opposed to negative ones such as “to be in the dark,” the traditional expectati…

Sample 10 Selected
  
Candidate #6 score=3.76
  
Skaters need to check their skate helmets every so often and ask yourself, ”Is it time to replace this helmet?” Well, that depends. Did you crash in it? For starters, most people are aware that you must replace a helmet after any crash where your head hit. The foam part of a helmet is made for one-time use, and after crushing once it is n…

Sample 11 Selected
  
Candidate #4 score=3.74
  
5 Types of Women’s Underwear That Men Love Underwear can say a lot about a woman. It’s something that men are obsessed with, to the point that, a mere glimpse of a thong waistband causes us to go into shock. On the surface we find them sexy, revealing. We’re able to see who a woman actually is—or maybe some guys are just plain horny. Howe…

Sample 12 Selected
  
Candidate #9 score=3.64
  
“Last night three cargoes of Bohea Tea were emptied into the sea. This is the most magnificent movement of all. There is a dignity, a majesty, a sublimity, in this last effort of the Patriots that I greatly admire.” - John Adams, diary entry, December 17, 1773 - John Adams, diary entry, December 17, 1773 A Novel Idea Is something so new a…

Sample 13 Selected
  
Candidate #23 score=3.43
  
The St. James kindergarteners have been working up to Project Week over the past month. We started slowly by taking walks in our neighborhood while Ms. Meghan and I noted what caught the children’s interest. It became apparent that the class was very interested in the L trains that they saw on our walks. It started with a simple question,…

Sample 14 Selected
  
Candidate #31 score=3.43
  
One of the challenges of working with ancient DNA samples is that damage accumulates over time, breaking the double helix structure into ever-smaller fragments. In the samples we worked with, these fragments were scattered and mixed with contaminants, making genome reconstruction a major technical challenge. But a shocking paper published…

Sample 15 Selected
  
Candidate #11 score=3.40
  
In decades past, classroom design was often an afterthought and followed a standardised layout. Plain boxed shaped classrooms, with identical chairs and tables throughout were commonplace in many schools. Read the latest issue of School News HERE Recently, though, there has been a shift away from this one-size-fits all approach to classro…

Sample 16 Selected
  
Candidate #10 score=3.38
  
Deforestation isn’t just happening in well-known global hotspots like Indonesia and Brazil’s rainforest. A new analysis says forests are also shrinking on state and private land in Oregon, where an estimated 522,000 acres of forest cover have disappeared since 2000. That’s an area six times larger than the city of Portland, equal to more…

Sample 17 Not selected
  
Candidate #28 score=3.22
  
You really have to be alert when studying science. Galaxies were created after matter. The stars in those galaxies were supposed to move slowly because there was more mass in the center of the galaxy. However, after dark matter was added, the stars appeared to move faster; however, this is not the case in our galaxy, suggesting that there…

Sample 18 Not selected
  
Candidate #17 score=3.19
  
Political Parties and Elections Political parties are an established part of modern mass democracy, and the conduct of elections in India is largely dependent on the behaviour of political parties. Although many candidates for Indian elections are independent, the winning candidates for Lok Sabha and Vidhan Sabha elections usually stand a…

Sample 19 Not selected
  
Candidate #25 score=3.05
  
KS2 Maths is an important core subject in the National Curriculum and this area of the website covers all the major aspects of the curriculum including numbers, calculations, problems and measures. Each subject area is designed to help children develop their knowledge, whether they are learning in a classroom or home schooling environment…

Sample 20 Not selected
  
Candidate #27 score=3.05
  
24/7 writing help on your phone Save to my list Remove from my list In the tumultuous 19th century, both Italy and Germany found themselves fragmented into numerous separate ruling states. The impetus for change came in the form of rising nationalism and liberalism, paving the way for the unification of these disparate entities. However,…

Sample 21 Not selected
  
Candidate #18 score=3.03
  
This article originally appeared in the December 2015 issue of Resource Recycling. Subscribe today for access to all print content. Since the 1990s, curbside and drop-off recycling has grown substantially – nearly 90 percent of households now have access, according to recent surveys from Moore Recycling Associates, the American Forest and…

Sample 22 Not selected
  
Candidate #15 score=2.95
  
Origami is an art form that combines precision, creativity, and patience. While basic origami is obtainable to every one, mastering complex origami designs can be quite a rewarding and impressive achievement. In this article, we’ll show you with the procedure for creating intricate origami while highlighting essential techniques for achie…

Sample 23 Not selected
  
Candidate #29 score=2.91
  
Earthquakes are the result of sudden movement along faults within the Earth. The movement releases stored-up ‘elastic strain’ energy in the form of seismic waves, which propagate through the Earth and cause the ground surface to shake. Such movement on the faults is generally a response to long-term deformation and the buildup of stress.…

Sample 24 Not selected
  
Candidate #22 score=2.90
  
How To Choose Decodable Readers for First Grade To decode or not to decode: really, there is no question. To help rising first graders become successful and enthusiastic readers this summer, decodable readers are essential reading resources. Although “decodable text” might sound like yet another form of educational lingo, parents and educ…

Sample 25 Not selected
  
Candidate #24 score=2.83
  
Next we will talk about solar radiation, that is, the forms of solar radiation that we receive on earth. Solar radiation is generated by a series of nuclear fusion reactions that occur in the Sun and, as a consequence, emit electromagnetic radiation that reaches the earth. This radiation received by the earth’s surface is measured in W /…

Sample 26 Not selected
  
Candidate #14 score=2.73
  
Is your major sustainable enough? Whether you’re pursuing a sustainability degree and want to further your knowledge, or are interested in supplementing your major in another area with sustainability education, plenty of independent learning resources are available. A wide range of credit and noncredit courses—including university- and or…

Sample 27 Not selected
  
Candidate #21 score=2.61
  
Nestled in the leafy suburbs of western Berlin, the Wannsee Conference House stands as a poignant reminder of a dark chapter in human history. The Wannsee Conference: A Pivotal Moment The Wannsee Conference, held on January 20, 1942, marked a pivotal moment in the implementation of Nazi Germany’s genocidal plans. Organized by SS-Obergrupp…

Sample 28 Not selected
  
Candidate #20 score=2.60
  
Conduct Disorder (CD) is a complex and serious behavioural and emotional disorder that can occur in children and adolescents. It’s characterised by a repetitive and persistent pattern of behaviour where the basic rights of others or major age-appropriate societal norms or rules are violated. Here’s an outline of Conduct Disorder in line w…

Sample 29 Not selected
  
Candidate #16 score=2.42
  
What is rotavirus and why does my baby need to be immunised? Rotavirus is a very infectious virus that causes the majority of serious cases of gastroenteritis in babies. It causes diarrhoea, vomiting and abdominal pain, usually lasting around a week. Most children will be infected by rotavirus once by the age of five. Gastroenteritis (cau…

Sample 30 Not selected
  
Candidate #26 score=2.31
  
Unveiling the Power: Key Provisions of the Civil Rights Act of 1864 What were the Civil Rights Act of 1864’s key provisions? The Civil Rights Act of 1864 was a pivotal moment in American history, establishing crucial legal protections for African Americans in the face of rampant discrimination. Editor Note: The Civil Rights Act of 1864 la…

Sample 31 Not selected
  
Candidate #30 score=2.28
  
Over 1.8 million professionals use CFI to learn accounting, financial analysis, modeling and more. Start with a free account to explore 20+ always-free courses and hundreds of finance templates and cheat sheets. What is the Central Limit Theorem (CLT)? The Central Limit Theorem (CLT) is a statistical concept that states that the sample me…

Sample 32 Not selected
  
Candidate #8 score=1.58
  
The Unsung Heroes of Your HVAC System: Understanding the Importance of Filters When it comes to your HVAC (Heating, Ventilation, and Air Conditioning) system, you might be quick to think about the thermostat, air ducts, or even the unit itself. However, there’s an unsung hero in your HVAC system that plays a pivotal role in maintaining in…

GREATS
  



Sample 1 Selected
  
Candidate #8 score=17.40
  
The Unsung Heroes of Your HVAC System: Understanding the Importance of Filters When it comes to your HVAC (Heating, Ventilation, and Air Conditioning) system, you might be quick to think about the thermostat, air ducts, or even the unit itself. However, there’s an unsung hero in your HVAC system that plays a pivotal role in maintaining in…



Sample 2 Selected
  
Candidate #15 score=15.47
  
Origami is an art form that combines precision, creativity, and patience. While basic origami is obtainable to every one, mastering complex origami designs can be quite a rewarding and impressive achievement. In this article, we’ll show you with the procedure for creating intricate origami while highlighting essential techniques for achie…

Sample 3 Selected
  
Candidate #20 score=15.18
  
Conduct Disorder (CD) is a complex and serious behavioural and emotional disorder that can occur in children and adolescents. It’s characterised by a repetitive and persistent pattern of behaviour where the basic rights of others or major age-appropriate societal norms or rules are violated. Here’s an outline of Conduct Disorder in line w…

Sample 4 Selected
  
Candidate #14 score=13.88
  
Is your major sustainable enough? Whether you’re pursuing a sustainability degree and want to further your knowledge, or are interested in supplementing your major in another area with sustainability education, plenty of independent learning resources are available. A wide range of credit and noncredit courses—including university- and or…

Sample 5 Selected
  
Candidate #22 score=13.87
  
How To Choose Decodable Readers for First Grade To decode or not to decode: really, there is no question. To help rising first graders become successful and enthusiastic readers this summer, decodable readers are essential reading resources. Although “decodable text” might sound like yet another form of educational lingo, parents and educ…

Sample 6 Selected
  
Candidate #27 score=13.71
  
24/7 writing help on your phone Save to my list Remove from my list In the tumultuous 19th century, both Italy and Germany found themselves fragmented into numerous separate ruling states. The impetus for change came in the form of rising nationalism and liberalism, paving the way for the unification of these disparate entities. However,…

Sample 7 Selected
  
Candidate #4 score=13.64
  
5 Types of Women’s Underwear That Men Love Underwear can say a lot about a woman. It’s something that men are obsessed with, to the point that, a mere glimpse of a thong waistband causes us to go into shock. On the surface we find them sexy, revealing. We’re able to see who a woman actually is—or maybe some guys are just plain horny. Howe…

Sample 8 Selected
  
Candidate #28 score=13.60
  
You really have to be alert when studying science. Galaxies were created after matter. The stars in those galaxies were supposed to move slowly because there was more mass in the center of the galaxy. However, after dark matter was added, the stars appeared to move faster; however, this is not the case in our galaxy, suggesting that there…

Sample 9 Selected
  
Candidate #21 score=13.45
  
Nestled in the leafy suburbs of western Berlin, the Wannsee Conference House stands as a poignant reminder of a dark chapter in human history. The Wannsee Conference: A Pivotal Moment The Wannsee Conference, held on January 20, 1942, marked a pivotal moment in the implementation of Nazi Germany’s genocidal plans. Organized by SS-Obergrupp…

Sample 10 Selected
  
Candidate #16 score=13.43
  
What is rotavirus and why does my baby need to be immunised? Rotavirus is a very infectious virus that causes the majority of serious cases of gastroenteritis in babies. It causes diarrhoea, vomiting and abdominal pain, usually lasting around a week. Most children will be infected by rotavirus once by the age of five. Gastroenteritis (cau…

Sample 11 Selected
  
Candidate #29 score=13.17
  
Earthquakes are the result of sudden movement along faults within the Earth. The movement releases stored-up ‘elastic strain’ energy in the form of seismic waves, which propagate through the Earth and cause the ground surface to shake. Such movement on the faults is generally a response to long-term deformation and the buildup of stress.…

Sample 12 Selected
  
Candidate #13 score=13.14
  
In Heart of Darkness it is the white invaders for instance, who are, almost without exception, embodiments of blindness, selfishness, and cruelty; and even in the cognitive domain, where such positive phrases as “to enlighten,” for instance, are conventionally opposed to negative ones such as “to be in the dark,” the traditional expectati…

Sample 13 Selected
  
Candidate #26 score=13.01
  
Unveiling the Power: Key Provisions of the Civil Rights Act of 1864 What were the Civil Rights Act of 1864’s key provisions? The Civil Rights Act of 1864 was a pivotal moment in American history, establishing crucial legal protections for African Americans in the face of rampant discrimination. Editor Note: The Civil Rights Act of 1864 la…

Sample 14 Selected
  
Candidate #18 score=12.93
  
This article originally appeared in the December 2015 issue of Resource Recycling. Subscribe today for access to all print content. Since the 1990s, curbside and drop-off recycling has grown substantially – nearly 90 percent of households now have access, according to recent surveys from Moore Recycling Associates, the American Forest and…

Sample 15 Selected
  
Candidate #24 score=12.92
  
Next we will talk about solar radiation, that is, the forms of solar radiation that we receive on earth. Solar radiation is generated by a series of nuclear fusion reactions that occur in the Sun and, as a consequence, emit electromagnetic radiation that reaches the earth. This radiation received by the earth’s surface is measured in W /…

Sample 16 Selected
  
Candidate #17 score=12.64
  
Political Parties and Elections Political parties are an established part of modern mass democracy, and the conduct of elections in India is largely dependent on the behaviour of political parties. Although many candidates for Indian elections are independent, the winning candidates for Lok Sabha and Vidhan Sabha elections usually stand a…

Sample 17 Not selected
  
Candidate #10 score=12.53
  
Deforestation isn’t just happening in well-known global hotspots like Indonesia and Brazil’s rainforest. A new analysis says forests are also shrinking on state and private land in Oregon, where an estimated 522,000 acres of forest cover have disappeared since 2000. That’s an area six times larger than the city of Portland, equal to more…

Sample 18 Not selected
  
Candidate #0 score=12.38
  
As it turns out, the exercises synonymous with strong, attractive abs may not be the best way to train your core—and may be doing damage to your back. Read more If you are worried about the excess holiday pounds many of us are still carrying around. There are a few easy, natural things you can do to shed them, and none of them requires an…

Sample 19 Not selected
  
Candidate #23 score=12.36
  
The St. James kindergarteners have been working up to Project Week over the past month. We started slowly by taking walks in our neighborhood while Ms. Meghan and I noted what caught the children’s interest. It became apparent that the class was very interested in the L trains that they saw on our walks. It started with a simple question,…

Sample 20 Not selected
  
Candidate #30 score=12.31
  
Over 1.8 million professionals use CFI to learn accounting, financial analysis, modeling and more. Start with a free account to explore 20+ always-free courses and hundreds of finance templates and cheat sheets. What is the Central Limit Theorem (CLT)? The Central Limit Theorem (CLT) is a statistical concept that states that the sample me…

Sample 21 Not selected
  
Candidate #31 score=12.04
  
One of the challenges of working with ancient DNA samples is that damage accumulates over time, breaking the double helix structure into ever-smaller fragments. In the samples we worked with, these fragments were scattered and mixed with contaminants, making genome reconstruction a major technical challenge. But a shocking paper published…

Sample 22 Not selected
  
Candidate #11 score=11.85
  
In decades past, classroom design was often an afterthought and followed a standardised layout. Plain boxed shaped classrooms, with identical chairs and tables throughout were commonplace in many schools. Read the latest issue of School News HERE Recently, though, there has been a shift away from this one-size-fits all approach to classro…

Sample 23 Not selected
  
Candidate #6 score=11.82
  
Skaters need to check their skate helmets every so often and ask yourself, ”Is it time to replace this helmet?” Well, that depends. Did you crash in it? For starters, most people are aware that you must replace a helmet after any crash where your head hit. The foam part of a helmet is made for one-time use, and after crushing once it is n…

Sample 24 Not selected
  
Candidate #25 score=11.51
  
KS2 Maths is an important core subject in the National Curriculum and this area of the website covers all the major aspects of the curriculum including numbers, calculations, problems and measures. Each subject area is designed to help children develop their knowledge, whether they are learning in a classroom or home schooling environment…

Sample 25 Not selected
  
Candidate #7 score=11.49
  
Elizabeth Hurley played as Dalila Release: Dec 8, 1996 Mara and her husband Manoa are both upstanding and religious Israelites living under the harsh and unjust rule of the Philistines. Much to their regret, they have not been able to have children. One day, a mysterious stranger appears to Mara and promises her that she will bear a son w…

Sample 26 Not selected
  
Candidate #5 score=11.49
  
Well this is the big one. So big apparently, that I had to take it there and raise the number from 10 to 15. There’s just that many fails in the world of female rap. Some slight missteps, some EPIC. Nevertheless, they are all worth mentioning. You can probably think of a bunch more, but this is what I have gathered picking up from my prev…

Sample 27 Not selected
  
Candidate #9 score=11.29
  
“Last night three cargoes of Bohea Tea were emptied into the sea. This is the most magnificent movement of all. There is a dignity, a majesty, a sublimity, in this last effort of the Patriots that I greatly admire.” - John Adams, diary entry, December 17, 1773 - John Adams, diary entry, December 17, 1773 A Novel Idea Is something so new a…

Sample 28 Not selected
  
Candidate #2 score=11.27
  
With the advent of new technologies for sneakers such as Vac Tech, Hyperfuse and Flyknit, the mid 90s and early 2000s methods of production and designing are becoming obsolete in this sneaker world. Nike Running is the future for Nike, generating billions of dollars per year, and we see Nike also not afraid to experiment with technology s…

Sample 29 Not selected
  
Candidate #19 score=11.04
  
Dividing Fractions Using Models Worksheet. This worksheet has six division with fractions issues to be solved — three must be solved with fashions and three with algorithms — options are on the second page. Answer key divide the unit fractions by whole numbers using th e fashions given. Use these resources to help reinforce the following…

Sample 30 Not selected
  
Candidate #12 score=10.94
  
Can you please give us a little short bio? (education, professional experiences, select publications, academic specialty, awards won) Public school teacher for 5 years BA art (UC Irvine) PhD. (UCLA) educational psychology Professor of Child Development, (25 years) CSUS Senior Research Scientist (Oregon Research Institute with Institute of…

Sample 31 Not selected
  
Candidate #1 score=10.23
  
Wedding & Party Venues - Sort By: Edgartown : (508) 627-9510 A 19th century gothic revival home transformed into the island’s premier eco-boutique hotel. Guests either stay in the 17-room Hob Knob hotel or in the privacy of their own Hob Knob House. Guests can expect individualized Hob Knob hospitality and modern luxury amenities in a rel…

Sample 32 Not selected
  
Candidate #3 score=9.77
  
starring John Travolta and Sam Jackson The first thing to understand about Basic –the basic thing, let’s say– is that although the commercials make it look like a war movie, it is not, for which we can all be grateful. No, Basic is a plot-twisty whodunnit. If The Usual Suspects died, and its body turned to cheese, and then that cheese-b…

QuRating
  



Sample 1 Selected
  
Candidate #26 score=11.87
  
Unveiling the Power: Key Provisions of the Civil Rights Act of 1864 What were the Civil Rights Act of 1864’s key provisions? The Civil Rights Act of 1864 was a pivotal moment in American history, establishing crucial legal protections for African Americans in the face of rampant discrimination. Editor Note: The Civil Rights Act of 1864 la…



Sample 2 Selected
  
Candidate #14 score=10.85
  
Is your major sustainable enough? Whether you’re pursuing a sustainability degree and want to further your knowledge, or are interested in supplementing your major in another area with sustainability education, plenty of independent learning resources are available. A wide range of credit and noncredit courses—including university- and or…

Sample 3 Selected
  
Candidate #22 score=10.82
  
How To Choose Decodable Readers for First Grade To decode or not to decode: really, there is no question. To help rising first graders become successful and enthusiastic readers this summer, decodable readers are essential reading resources. Although “decodable text” might sound like yet another form of educational lingo, parents and educ…

Sample 4 Selected
  
Candidate #31 score=10.42
  
One of the challenges of working with ancient DNA samples is that damage accumulates over time, breaking the double helix structure into ever-smaller fragments. In the samples we worked with, these fragments were scattered and mixed with contaminants, making genome reconstruction a major technical challenge. But a shocking paper published…

Sample 5 Selected
  
Candidate #23 score=10.27
  
The St. James kindergarteners have been working up to Project Week over the past month. We started slowly by taking walks in our neighborhood while Ms. Meghan and I noted what caught the children’s interest. It became apparent that the class was very interested in the L trains that they saw on our walks. It started with a simple question,…

Sample 6 Selected
  
Candidate #12 score=10.05
  
Can you please give us a little short bio? (education, professional experiences, select publications, academic specialty, awards won) Public school teacher for 5 years BA art (UC Irvine) PhD. (UCLA) educational psychology Professor of Child Development, (25 years) CSUS Senior Research Scientist (Oregon Research Institute with Institute of…

Sample 7 Selected
  
Candidate #29 score=9.76
  
Earthquakes are the result of sudden movement along faults within the Earth. The movement releases stored-up ‘elastic strain’ energy in the form of seismic waves, which propagate through the Earth and cause the ground surface to shake. Such movement on the faults is generally a response to long-term deformation and the buildup of stress.…

Sample 8 Selected
  
Candidate #20 score=9.62
  
Conduct Disorder (CD) is a complex and serious behavioural and emotional disorder that can occur in children and adolescents. It’s characterised by a repetitive and persistent pattern of behaviour where the basic rights of others or major age-appropriate societal norms or rules are violated. Here’s an outline of Conduct Disorder in line w…

Sample 9 Selected
  
Candidate #25 score=8.96
  
KS2 Maths is an important core subject in the National Curriculum and this area of the website covers all the major aspects of the curriculum including numbers, calculations, problems and measures. Each subject area is designed to help children develop their knowledge, whether they are learning in a classroom or home schooling environment…

Sample 10 Selected
  
Candidate #30 score=8.95
  
Over 1.8 million professionals use CFI to learn accounting, financial analysis, modeling and more. Start with a free account to explore 20+ always-free courses and hundreds of finance templates and cheat sheets. What is the Central Limit Theorem (CLT)? The Central Limit Theorem (CLT) is a statistical concept that states that the sample me…

Sample 11 Selected
  
Candidate #15 score=8.39
  
Origami is an art form that combines precision, creativity, and patience. While basic origami is obtainable to every one, mastering complex origami designs can be quite a rewarding and impressive achievement. In this article, we’ll show you with the procedure for creating intricate origami while highlighting essential techniques for achie…

Sample 12 Selected
  
Candidate #28 score=8.17
  
You really have to be alert when studying science. Galaxies were created after matter. The stars in those galaxies were supposed to move slowly because there was more mass in the center of the galaxy. However, after dark matter was added, the stars appeared to move faster; however, this is not the case in our galaxy, suggesting that there…

Sample 13 Selected
  
Candidate #24 score=8.12
  
Next we will talk about solar radiation, that is, the forms of solar radiation that we receive on earth. Solar radiation is generated by a series of nuclear fusion reactions that occur in the Sun and, as a consequence, emit electromagnetic radiation that reaches the earth. This radiation received by the earth’s surface is measured in W /…

Sample 14 Selected
  
Candidate #16 score=8.04
  
What is rotavirus and why does my baby need to be immunised? Rotavirus is a very infectious virus that causes the majority of serious cases of gastroenteritis in babies. It causes diarrhoea, vomiting and abdominal pain, usually lasting around a week. Most children will be infected by rotavirus once by the age of five. Gastroenteritis (cau…

Sample 15 Selected
  
Candidate #21 score=8.02
  
Nestled in the leafy suburbs of western Berlin, the Wannsee Conference House stands as a poignant reminder of a dark chapter in human history. The Wannsee Conference: A Pivotal Moment The Wannsee Conference, held on January 20, 1942, marked a pivotal moment in the implementation of Nazi Germany’s genocidal plans. Organized by SS-Obergrupp…

Sample 16 Selected
  
Candidate #11 score=7.76
  
In decades past, classroom design was often an afterthought and followed a standardised layout. Plain boxed shaped classrooms, with identical chairs and tables throughout were commonplace in many schools. Read the latest issue of School News HERE Recently, though, there has been a shift away from this one-size-fits all approach to classro…

Sample 17 Not selected
  
Candidate #18 score=7.47
  
This article originally appeared in the December 2015 issue of Resource Recycling. Subscribe today for access to all print content. Since the 1990s, curbside and drop-off recycling has grown substantially – nearly 90 percent of households now have access, according to recent surveys from Moore Recycling Associates, the American Forest and…

Sample 18 Not selected
  
Candidate #8 score=7.46
  
The Unsung Heroes of Your HVAC System: Understanding the Importance of Filters When it comes to your HVAC (Heating, Ventilation, and Air Conditioning) system, you might be quick to think about the thermostat, air ducts, or even the unit itself. However, there’s an unsung hero in your HVAC system that plays a pivotal role in maintaining in…

Sample 19 Not selected
  
Candidate #10 score=7.21
  
Deforestation isn’t just happening in well-known global hotspots like Indonesia and Brazil’s rainforest. A new analysis says forests are also shrinking on state and private land in Oregon, where an estimated 522,000 acres of forest cover have disappeared since 2000. That’s an area six times larger than the city of Portland, equal to more…

Sample 20 Not selected
  
Candidate #27 score=7.05
  
24/7 writing help on your phone Save to my list Remove from my list In the tumultuous 19th century, both Italy and Germany found themselves fragmented into numerous separate ruling states. The impetus for change came in the form of rising nationalism and liberalism, paving the way for the unification of these disparate entities. However,…

Sample 21 Not selected
  
Candidate #19 score=5.30
  
Dividing Fractions Using Models Worksheet. This worksheet has six division with fractions issues to be solved — three must be solved with fashions and three with algorithms — options are on the second page. Answer key divide the unit fractions by whole numbers using th e fashions given. Use these resources to help reinforce the following…

Sample 22 Not selected
  
Candidate #17 score=5.29
  
Political Parties and Elections Political parties are an established part of modern mass democracy, and the conduct of elections in India is largely dependent on the behaviour of political parties. Although many candidates for Indian elections are independent, the winning candidates for Lok Sabha and Vidhan Sabha elections usually stand a…

Sample 23 Not selected
  
Candidate #9 score=4.21
  
“Last night three cargoes of Bohea Tea were emptied into the sea. This is the most magnificent movement of all. There is a dignity, a majesty, a sublimity, in this last effort of the Patriots that I greatly admire.” - John Adams, diary entry, December 17, 1773 - John Adams, diary entry, December 17, 1773 A Novel Idea Is something so new a…

Sample 24 Not selected
  
Candidate #0 score=3.94
  
As it turns out, the exercises synonymous with strong, attractive abs may not be the best way to train your core—and may be doing damage to your back. Read more If you are worried about the excess holiday pounds many of us are still carrying around. There are a few easy, natural things you can do to shed them, and none of them requires an…

Sample 25 Not selected
  
Candidate #6 score=2.82
  
Skaters need to check their skate helmets every so often and ask yourself, ”Is it time to replace this helmet?” Well, that depends. Did you crash in it? For starters, most people are aware that you must replace a helmet after any crash where your head hit. The foam part of a helmet is made for one-time use, and after crushing once it is n…

Sample 26 Not selected
  
Candidate #13 score=2.34
  
In Heart of Darkness it is the white invaders for instance, who are, almost without exception, embodiments of blindness, selfishness, and cruelty; and even in the cognitive domain, where such positive phrases as “to enlighten,” for instance, are conventionally opposed to negative ones such as “to be in the dark,” the traditional expectati…

Sample 27 Not selected
  
Candidate #7 score=1.46
  
Elizabeth Hurley played as Dalila Release: Dec 8, 1996 Mara and her husband Manoa are both upstanding and religious Israelites living under the harsh and unjust rule of the Philistines. Much to their regret, they have not been able to have children. One day, a mysterious stranger appears to Mara and promises her that she will bear a son w…

Sample 28 Not selected
  
Candidate #2 score=0.321
  
With the advent of new technologies for sneakers such as Vac Tech, Hyperfuse and Flyknit, the mid 90s and early 2000s methods of production and designing are becoming obsolete in this sneaker world. Nike Running is the future for Nike, generating billions of dollars per year, and we see Nike also not afraid to experiment with technology s…

Sample 29 Not selected
  
Candidate #1 score=-0.429
  
Wedding & Party Venues - Sort By: Edgartown : (508) 627-9510 A 19th century gothic revival home transformed into the island’s premier eco-boutique hotel. Guests either stay in the 17-room Hob Knob hotel or in the privacy of their own Hob Knob House. Guests can expect individualized Hob Knob hospitality and modern luxury amenities in a rel…

Sample 30 Not selected
  
Candidate #3 score=-2.09
  
starring John Travolta and Sam Jackson The first thing to understand about Basic –the basic thing, let’s say– is that although the commercials make it look like a war movie, it is not, for which we can all be grateful. No, Basic is a plot-twisty whodunnit. If The Usual Suspects died, and its body turned to cheese, and then that cheese-b…

Sample 31 Not selected
  
Candidate #4 score=-2.84
  
5 Types of Women’s Underwear That Men Love Underwear can say a lot about a woman. It’s something that men are obsessed with, to the point that, a mere glimpse of a thong waistband causes us to go into shock. On the surface we find them sexy, revealing. We’re able to see who a woman actually is—or maybe some guys are just plain horny. Howe…

Sample 32 Not selected
  
Candidate #5 score=-4.08
  
Well this is the big one. So big apparently, that I had to take it there and raise the number from 10 to 15. There’s just that many fails in the world of female rap. Some slight missteps, some EPIC. Nevertheless, they are all worth mentioning. You can probably think of a bunch more, but this is what I have gathered picking up from my prev…

FineWeb-Edu
  



Sample 1 Selected
  
Candidate #28 score=4.62
  
You really have to be alert when studying science. Galaxies were created after matter. The stars in those galaxies were supposed to move slowly because there was more mass in the center of the galaxy. However, after dark matter was added, the stars appeared to move faster; however, this is not the case in our galaxy, suggesting that there…



Sample 2 Selected
  
Candidate #25 score=4.61
  
KS2 Maths is an important core subject in the National Curriculum and this area of the website covers all the major aspects of the curriculum including numbers, calculations, problems and measures. Each subject area is designed to help children develop their knowledge, whether they are learning in a classroom or home schooling environment…

Sample 3 Selected
  
Candidate #29 score=4.61
  
Earthquakes are the result of sudden movement along faults within the Earth. The movement releases stored-up ‘elastic strain’ energy in the form of seismic waves, which propagate through the Earth and cause the ground surface to shake. Such movement on the faults is generally a response to long-term deformation and the buildup of stress.…

Sample 4 Selected
  
Candidate #31 score=4.57
  
One of the challenges of working with ancient DNA samples is that damage accumulates over time, breaking the double helix structure into ever-smaller fragments. In the samples we worked with, these fragments were scattered and mixed with contaminants, making genome reconstruction a major technical challenge. But a shocking paper published…

Sample 5 Selected
  
Candidate #24 score=4.54
  
Next we will talk about solar radiation, that is, the forms of solar radiation that we receive on earth. Solar radiation is generated by a series of nuclear fusion reactions that occur in the Sun and, as a consequence, emit electromagnetic radiation that reaches the earth. This radiation received by the earth’s surface is measured in W /…

Sample 6 Selected
  
Candidate #26 score=4.53
  
Unveiling the Power: Key Provisions of the Civil Rights Act of 1864 What were the Civil Rights Act of 1864’s key provisions? The Civil Rights Act of 1864 was a pivotal moment in American history, establishing crucial legal protections for African Americans in the face of rampant discrimination. Editor Note: The Civil Rights Act of 1864 la…

Sample 7 Selected
  
Candidate #27 score=4.53
  
24/7 writing help on your phone Save to my list Remove from my list In the tumultuous 19th century, both Italy and Germany found themselves fragmented into numerous separate ruling states. The impetus for change came in the form of rising nationalism and liberalism, paving the way for the unification of these disparate entities. However,…

Sample 8 Selected
  
Candidate #30 score=4.50
  
Over 1.8 million professionals use CFI to learn accounting, financial analysis, modeling and more. Start with a free account to explore 20+ always-free courses and hundreds of finance templates and cheat sheets. What is the Central Limit Theorem (CLT)? The Central Limit Theorem (CLT) is a statistical concept that states that the sample me…

Sample 9 Selected
  
Candidate #20 score=4.18
  
Conduct Disorder (CD) is a complex and serious behavioural and emotional disorder that can occur in children and adolescents. It’s characterised by a repetitive and persistent pattern of behaviour where the basic rights of others or major age-appropriate societal norms or rules are violated. Here’s an outline of Conduct Disorder in line w…

Sample 10 Selected
  
Candidate #19 score=4.08
  
Dividing Fractions Using Models Worksheet. This worksheet has six division with fractions issues to be solved — three must be solved with fashions and three with algorithms — options are on the second page. Answer key divide the unit fractions by whole numbers using th e fashions given. Use these resources to help reinforce the following…

Sample 11 Selected
  
Candidate #21 score=3.96
  
Nestled in the leafy suburbs of western Berlin, the Wannsee Conference House stands as a poignant reminder of a dark chapter in human history. The Wannsee Conference: A Pivotal Moment The Wannsee Conference, held on January 20, 1942, marked a pivotal moment in the implementation of Nazi Germany’s genocidal plans. Organized by SS-Obergrupp…

Sample 12 Selected
  
Candidate #22 score=3.92
  
How To Choose Decodable Readers for First Grade To decode or not to decode: really, there is no question. To help rising first graders become successful and enthusiastic readers this summer, decodable readers are essential reading resources. Although “decodable text” might sound like yet another form of educational lingo, parents and educ…

Sample 13 Selected
  
Candidate #17 score=3.85
  
Political Parties and Elections Political parties are an established part of modern mass democracy, and the conduct of elections in India is largely dependent on the behaviour of political parties. Although many candidates for Indian elections are independent, the winning candidates for Lok Sabha and Vidhan Sabha elections usually stand a…

Sample 14 Selected
  
Candidate #23 score=3.73
  
The St. James kindergarteners have been working up to Project Week over the past month. We started slowly by taking walks in our neighborhood while Ms. Meghan and I noted what caught the children’s interest. It became apparent that the class was very interested in the L trains that they saw on our walks. It started with a simple question,…

Sample 15 Selected
  
Candidate #16 score=3.63
  
What is rotavirus and why does my baby need to be immunised? Rotavirus is a very infectious virus that causes the majority of serious cases of gastroenteritis in babies. It causes diarrhoea, vomiting and abdominal pain, usually lasting around a week. Most children will be infected by rotavirus once by the age of five. Gastroenteritis (cau…

Sample 16 Selected
  
Candidate #18 score=3.56
  
This article originally appeared in the December 2015 issue of Resource Recycling. Subscribe today for access to all print content. Since the 1990s, curbside and drop-off recycling has grown substantially – nearly 90 percent of households now have access, according to recent surveys from Moore Recycling Associates, the American Forest and…

Sample 17 Not selected
  
Candidate #10 score=3.30
  
Deforestation isn’t just happening in well-known global hotspots like Indonesia and Brazil’s rainforest. A new analysis says forests are also shrinking on state and private land in Oregon, where an estimated 522,000 acres of forest cover have disappeared since 2000. That’s an area six times larger than the city of Portland, equal to more…

Sample 18 Not selected
  
Candidate #9 score=3.30
  
“Last night three cargoes of Bohea Tea were emptied into the sea. This is the most magnificent movement of all. There is a dignity, a majesty, a sublimity, in this last effort of the Patriots that I greatly admire.” - John Adams, diary entry, December 17, 1773 - John Adams, diary entry, December 17, 1773 A Novel Idea Is something so new a…

Sample 19 Not selected
  
Candidate #11 score=2.95
  
In decades past, classroom design was often an afterthought and followed a standardised layout. Plain boxed shaped classrooms, with identical chairs and tables throughout were commonplace in many schools. Read the latest issue of School News HERE Recently, though, there has been a shift away from this one-size-fits all approach to classro…

Sample 20 Not selected
  
Candidate #15 score=2.93
  
Origami is an art form that combines precision, creativity, and patience. While basic origami is obtainable to every one, mastering complex origami designs can be quite a rewarding and impressive achievement. In this article, we’ll show you with the procedure for creating intricate origami while highlighting essential techniques for achie…

Sample 21 Not selected
  
Candidate #13 score=2.86
  
In Heart of Darkness it is the white invaders for instance, who are, almost without exception, embodiments of blindness, selfishness, and cruelty; and even in the cognitive domain, where such positive phrases as “to enlighten,” for instance, are conventionally opposed to negative ones such as “to be in the dark,” the traditional expectati…

Sample 22 Not selected
  
Candidate #8 score=2.83
  
The Unsung Heroes of Your HVAC System: Understanding the Importance of Filters When it comes to your HVAC (Heating, Ventilation, and Air Conditioning) system, you might be quick to think about the thermostat, air ducts, or even the unit itself. However, there’s an unsung hero in your HVAC system that plays a pivotal role in maintaining in…

Sample 23 Not selected
  
Candidate #12 score=2.72
  
Can you please give us a little short bio? (education, professional experiences, select publications, academic specialty, awards won) Public school teacher for 5 years BA art (UC Irvine) PhD. (UCLA) educational psychology Professor of Child Development, (25 years) CSUS Senior Research Scientist (Oregon Research Institute with Institute of…

Sample 24 Not selected
  
Candidate #14 score=2.68
  
Is your major sustainable enough? Whether you’re pursuing a sustainability degree and want to further your knowledge, or are interested in supplementing your major in another area with sustainability education, plenty of independent learning resources are available. A wide range of credit and noncredit courses—including university- and or…

Sample 25 Not selected
  
Candidate #6 score=1.77
  
Skaters need to check their skate helmets every so often and ask yourself, ”Is it time to replace this helmet?” Well, that depends. Did you crash in it? For starters, most people are aware that you must replace a helmet after any crash where your head hit. The foam part of a helmet is made for one-time use, and after crushing once it is n…

Sample 26 Not selected
  
Candidate #0 score=1.76
  
As it turns out, the exercises synonymous with strong, attractive abs may not be the best way to train your core—and may be doing damage to your back. Read more If you are worried about the excess holiday pounds many of us are still carrying around. There are a few easy, natural things you can do to shed them, and none of them requires an…

Sample 27 Not selected
  
Candidate #7 score=1.39
  
Elizabeth Hurley played as Dalila Release: Dec 8, 1996 Mara and her husband Manoa are both upstanding and religious Israelites living under the harsh and unjust rule of the Philistines. Much to their regret, they have not been able to have children. One day, a mysterious stranger appears to Mara and promises her that she will bear a son w…

Sample 28 Not selected
  
Candidate #5 score=0.957
  
Well this is the big one. So big apparently, that I had to take it there and raise the number from 10 to 15. There’s just that many fails in the world of female rap. Some slight missteps, some EPIC. Nevertheless, they are all worth mentioning. You can probably think of a bunch more, but this is what I have gathered picking up from my prev…

Sample 29 Not selected
  
Candidate #3 score=0.919
  
starring John Travolta and Sam Jackson The first thing to understand about Basic –the basic thing, let’s say– is that although the commercials make it look like a war movie, it is not, for which we can all be grateful. No, Basic is a plot-twisty whodunnit. If The Usual Suspects died, and its body turned to cheese, and then that cheese-b…

Sample 30 Not selected
  
Candidate #2 score=0.880
  
With the advent of new technologies for sneakers such as Vac Tech, Hyperfuse and Flyknit, the mid 90s and early 2000s methods of production and designing are becoming obsolete in this sneaker world. Nike Running is the future for Nike, generating billions of dollars per year, and we see Nike also not afraid to experiment with technology s…

Sample 31 Not selected
  
Candidate #1 score=0.798
  
Wedding & Party Venues - Sort By: Edgartown : (508) 627-9510 A 19th century gothic revival home transformed into the island’s premier eco-boutique hotel. Guests either stay in the 17-room Hob Knob hotel or in the privacy of their own Hob Knob House. Guests can expect individualized Hob Knob hospitality and modern luxury amenities in a rel…

Sample 32 Not selected
  
Candidate #4 score=0.163
  
5 Types of Women’s Underwear That Men Love Underwear can say a lot about a woman. It’s something that men are obsessed with, to the point that, a mere glimpse of a thong waistband causes us to go into shock. On the surface we find them sexy, revealing. We’re able to see who a woman actually is—or maybe some guys are just plain horny. Howe…

Ultra-FineWeb
  



Sample 1 Selected
  
Candidate #26 score=1.000
  
Unveiling the Power: Key Provisions of the Civil Rights Act of 1864 What were the Civil Rights Act of 1864’s key provisions? The Civil Rights Act of 1864 was a pivotal moment in American history, establishing crucial legal protections for African Americans in the face of rampant discrimination. Editor Note: The Civil Rights Act of 1864 la…



Sample 2 Selected
  
Candidate #29 score=0.999
  
Earthquakes are the result of sudden movement along faults within the Earth. The movement releases stored-up ‘elastic strain’ energy in the form of seismic waves, which propagate through the Earth and cause the ground surface to shake. Such movement on the faults is generally a response to long-term deformation and the buildup of stress.…

Sample 3 Selected
  
Candidate #20 score=0.998
  
Conduct Disorder (CD) is a complex and serious behavioural and emotional disorder that can occur in children and adolescents. It’s characterised by a repetitive and persistent pattern of behaviour where the basic rights of others or major age-appropriate societal norms or rules are violated. Here’s an outline of Conduct Disorder in line w…

Sample 4 Selected
  
Candidate #22 score=0.997
  
How To Choose Decodable Readers for First Grade To decode or not to decode: really, there is no question. To help rising first graders become successful and enthusiastic readers this summer, decodable readers are essential reading resources. Although “decodable text” might sound like yet another form of educational lingo, parents and educ…

Sample 5 Selected
  
Candidate #31 score=0.994
  
One of the challenges of working with ancient DNA samples is that damage accumulates over time, breaking the double helix structure into ever-smaller fragments. In the samples we worked with, these fragments were scattered and mixed with contaminants, making genome reconstruction a major technical challenge. But a shocking paper published…

Sample 6 Selected
  
Candidate #19 score=0.987
  
Dividing Fractions Using Models Worksheet. This worksheet has six division with fractions issues to be solved — three must be solved with fashions and three with algorithms — options are on the second page. Answer key divide the unit fractions by whole numbers using th e fashions given. Use these resources to help reinforce the following…

Sample 7 Selected
  
Candidate #30 score=0.978
  
Over 1.8 million professionals use CFI to learn accounting, financial analysis, modeling and more. Start with a free account to explore 20+ always-free courses and hundreds of finance templates and cheat sheets. What is the Central Limit Theorem (CLT)? The Central Limit Theorem (CLT) is a statistical concept that states that the sample me…

Sample 8 Selected
  
Candidate #24 score=0.971
  
Next we will talk about solar radiation, that is, the forms of solar radiation that we receive on earth. Solar radiation is generated by a series of nuclear fusion reactions that occur in the Sun and, as a consequence, emit electromagnetic radiation that reaches the earth. This radiation received by the earth’s surface is measured in W /…

Sample 9 Selected
  
Candidate #28 score=0.964
  
You really have to be alert when studying science. Galaxies were created after matter. The stars in those galaxies were supposed to move slowly because there was more mass in the center of the galaxy. However, after dark matter was added, the stars appeared to move faster; however, this is not the case in our galaxy, suggesting that there…

Sample 10 Selected
  
Candidate #25 score=0.958
  
KS2 Maths is an important core subject in the National Curriculum and this area of the website covers all the major aspects of the curriculum including numbers, calculations, problems and measures. Each subject area is designed to help children develop their knowledge, whether they are learning in a classroom or home schooling environment…

Sample 11 Selected
  
Candidate #8 score=0.955
  
The Unsung Heroes of Your HVAC System: Understanding the Importance of Filters When it comes to your HVAC (Heating, Ventilation, and Air Conditioning) system, you might be quick to think about the thermostat, air ducts, or even the unit itself. However, there’s an unsung hero in your HVAC system that plays a pivotal role in maintaining in…

Sample 12 Selected
  
Candidate #27 score=0.928
  
24/7 writing help on your phone Save to my list Remove from my list In the tumultuous 19th century, both Italy and Germany found themselves fragmented into numerous separate ruling states. The impetus for change came in the form of rising nationalism and liberalism, paving the way for the unification of these disparate entities. However,…

Sample 13 Selected
  
Candidate #15 score=0.928
  
Origami is an art form that combines precision, creativity, and patience. While basic origami is obtainable to every one, mastering complex origami designs can be quite a rewarding and impressive achievement. In this article, we’ll show you with the procedure for creating intricate origami while highlighting essential techniques for achie…

Sample 14 Selected
  
Candidate #23 score=0.927
  
The St. James kindergarteners have been working up to Project Week over the past month. We started slowly by taking walks in our neighborhood while Ms. Meghan and I noted what caught the children’s interest. It became apparent that the class was very interested in the L trains that they saw on our walks. It started with a simple question,…

Sample 15 Selected
  
Candidate #21 score=0.745
  
Nestled in the leafy suburbs of western Berlin, the Wannsee Conference House stands as a poignant reminder of a dark chapter in human history. The Wannsee Conference: A Pivotal Moment The Wannsee Conference, held on January 20, 1942, marked a pivotal moment in the implementation of Nazi Germany’s genocidal plans. Organized by SS-Obergrupp…

Sample 16 Selected
  
Candidate #12 score=0.718
  
Can you please give us a little short bio? (education, professional experiences, select publications, academic specialty, awards won) Public school teacher for 5 years BA art (UC Irvine) PhD. (UCLA) educational psychology Professor of Child Development, (25 years) CSUS Senior Research Scientist (Oregon Research Institute with Institute of…

Sample 17 Not selected
  
Candidate #14 score=0.695
  
Is your major sustainable enough? Whether you’re pursuing a sustainability degree and want to further your knowledge, or are interested in supplementing your major in another area with sustainability education, plenty of independent learning resources are available. A wide range of credit and noncredit courses—including university- and or…

Sample 18 Not selected
  
Candidate #18 score=0.648
  
This article originally appeared in the December 2015 issue of Resource Recycling. Subscribe today for access to all print content. Since the 1990s, curbside and drop-off recycling has grown substantially – nearly 90 percent of households now have access, according to recent surveys from Moore Recycling Associates, the American Forest and…

Sample 19 Not selected
  
Candidate #11 score=0.547
  
In decades past, classroom design was often an afterthought and followed a standardised layout. Plain boxed shaped classrooms, with identical chairs and tables throughout were commonplace in many schools. Read the latest issue of School News HERE Recently, though, there has been a shift away from this one-size-fits all approach to classro…

Sample 20 Not selected
  
Candidate #13 score=0.532
  
In Heart of Darkness it is the white invaders for instance, who are, almost without exception, embodiments of blindness, selfishness, and cruelty; and even in the cognitive domain, where such positive phrases as “to enlighten,” for instance, are conventionally opposed to negative ones such as “to be in the dark,” the traditional expectati…

Sample 21 Not selected
  
Candidate #10 score=0.477
  
Deforestation isn’t just happening in well-known global hotspots like Indonesia and Brazil’s rainforest. A new analysis says forests are also shrinking on state and private land in Oregon, where an estimated 522,000 acres of forest cover have disappeared since 2000. That’s an area six times larger than the city of Portland, equal to more…

Sample 22 Not selected
  
Candidate #17 score=0.470
  
Political Parties and Elections Political parties are an established part of modern mass democracy, and the conduct of elections in India is largely dependent on the behaviour of political parties. Although many candidates for Indian elections are independent, the winning candidates for Lok Sabha and Vidhan Sabha elections usually stand a…

Sample 23 Not selected
  
Candidate #9 score=0.224
  
“Last night three cargoes of Bohea Tea were emptied into the sea. This is the most magnificent movement of all. There is a dignity, a majesty, a sublimity, in this last effort of the Patriots that I greatly admire.” - John Adams, diary entry, December 17, 1773 - John Adams, diary entry, December 17, 1773 A Novel Idea Is something so new a…

Sample 24 Not selected
  
Candidate #16 score=0.211
  
What is rotavirus and why does my baby need to be immunised? Rotavirus is a very infectious virus that causes the majority of serious cases of gastroenteritis in babies. It causes diarrhoea, vomiting and abdominal pain, usually lasting around a week. Most children will be infected by rotavirus once by the age of five. Gastroenteritis (cau…

Sample 25 Not selected
  
Candidate #7 score=0.095
  
Elizabeth Hurley played as Dalila Release: Dec 8, 1996 Mara and her husband Manoa are both upstanding and religious Israelites living under the harsh and unjust rule of the Philistines. Much to their regret, they have not been able to have children. One day, a mysterious stranger appears to Mara and promises her that she will bear a son w…

Sample 26 Not selected
  
Candidate #3 score=0.069
  
starring John Travolta and Sam Jackson The first thing to understand about Basic –the basic thing, let’s say– is that although the commercials make it look like a war movie, it is not, for which we can all be grateful. No, Basic is a plot-twisty whodunnit. If The Usual Suspects died, and its body turned to cheese, and then that cheese-b…

Sample 27 Not selected
  
Candidate #2 score=0.058
  
With the advent of new technologies for sneakers such as Vac Tech, Hyperfuse and Flyknit, the mid 90s and early 2000s methods of production and designing are becoming obsolete in this sneaker world. Nike Running is the future for Nike, generating billions of dollars per year, and we see Nike also not afraid to experiment with technology s…

Sample 28 Not selected
  
Candidate #4 score=0.024
  
5 Types of Women’s Underwear That Men Love Underwear can say a lot about a woman. It’s something that men are obsessed with, to the point that, a mere glimpse of a thong waistband causes us to go into shock. On the surface we find them sexy, revealing. We’re able to see who a woman actually is—or maybe some guys are just plain horny. Howe…

Sample 29 Not selected
  
Candidate #5 score=0.019
  
Well this is the big one. So big apparently, that I had to take it there and raise the number from 10 to 15. There’s just that many fails in the world of female rap. Some slight missteps, some EPIC. Nevertheless, they are all worth mentioning. You can probably think of a bunch more, but this is what I have gathered picking up from my prev…

Sample 30 Not selected
  
Candidate #0 score=0.018
  
As it turns out, the exercises synonymous with strong, attractive abs may not be the best way to train your core—and may be doing damage to your back. Read more If you are worried about the excess holiday pounds many of us are still carrying around. There are a few easy, natural things you can do to shed them, and none of them requires an…

Sample 31 Not selected
  
Candidate #6 score=0.016
  
Skaters need to check their skate helmets every so often and ask yourself, ”Is it time to replace this helmet?” Well, that depends. Did you crash in it? For starters, most people are aware that you must replace a helmet after any crash where your head hit. The foam part of a helmet is made for one-time use, and after crushing once it is n…

Sample 32 Not selected
  
Candidate #1 score=0.000579
  
Wedding & Party Venues - Sort By: Edgartown : (508) 627-9510 A 19th century gothic revival home transformed into the island’s premier eco-boutique hotel. Guests either stay in the 17-room Hob Knob hotel or in the privacy of their own Hob Knob House. Guests can expect individualized Hob Knob hospitality and modern luxury amenities in a rel…

DCLM-FastText
  



Sample 1 Selected
  
Candidate #28 score=0.902
  
You really have to be alert when studying science. Galaxies were created after matter. The stars in those galaxies were supposed to move slowly because there was more mass in the center of the galaxy. However, after dark matter was added, the stars appeared to move faster; however, this is not the case in our galaxy, suggesting that there…



Sample 2 Selected
  
Candidate #29 score=0.761
  
Earthquakes are the result of sudden movement along faults within the Earth. The movement releases stored-up ‘elastic strain’ energy in the form of seismic waves, which propagate through the Earth and cause the ground surface to shake. Such movement on the faults is generally a response to long-term deformation and the buildup of stress.…

Sample 3 Selected
  
Candidate #15 score=0.632
  
Origami is an art form that combines precision, creativity, and patience. While basic origami is obtainable to every one, mastering complex origami designs can be quite a rewarding and impressive achievement. In this article, we’ll show you with the procedure for creating intricate origami while highlighting essential techniques for achie…

Sample 4 Selected
  
Candidate #26 score=0.612
  
Unveiling the Power: Key Provisions of the Civil Rights Act of 1864 What were the Civil Rights Act of 1864’s key provisions? The Civil Rights Act of 1864 was a pivotal moment in American history, establishing crucial legal protections for African Americans in the face of rampant discrimination. Editor Note: The Civil Rights Act of 1864 la…

Sample 5 Selected
  
Candidate #31 score=0.564
  
One of the challenges of working with ancient DNA samples is that damage accumulates over time, breaking the double helix structure into ever-smaller fragments. In the samples we worked with, these fragments were scattered and mixed with contaminants, making genome reconstruction a major technical challenge. But a shocking paper published…

Sample 6 Selected
  
Candidate #27 score=0.483
  
24/7 writing help on your phone Save to my list Remove from my list In the tumultuous 19th century, both Italy and Germany found themselves fragmented into numerous separate ruling states. The impetus for change came in the form of rising nationalism and liberalism, paving the way for the unification of these disparate entities. However,…

Sample 7 Selected
  
Candidate #3 score=0.367
  
starring John Travolta and Sam Jackson The first thing to understand about Basic –the basic thing, let’s say– is that although the commercials make it look like a war movie, it is not, for which we can all be grateful. No, Basic is a plot-twisty whodunnit. If The Usual Suspects died, and its body turned to cheese, and then that cheese-b…

Sample 8 Selected
  
Candidate #30 score=0.294
  
Over 1.8 million professionals use CFI to learn accounting, financial analysis, modeling and more. Start with a free account to explore 20+ always-free courses and hundreds of finance templates and cheat sheets. What is the Central Limit Theorem (CLT)? The Central Limit Theorem (CLT) is a statistical concept that states that the sample me…

Sample 9 Selected
  
Candidate #20 score=0.242
  
Conduct Disorder (CD) is a complex and serious behavioural and emotional disorder that can occur in children and adolescents. It’s characterised by a repetitive and persistent pattern of behaviour where the basic rights of others or major age-appropriate societal norms or rules are violated. Here’s an outline of Conduct Disorder in line w…

Sample 10 Selected
  
Candidate #16 score=0.168
  
What is rotavirus and why does my baby need to be immunised? Rotavirus is a very infectious virus that causes the majority of serious cases of gastroenteritis in babies. It causes diarrhoea, vomiting and abdominal pain, usually lasting around a week. Most children will be infected by rotavirus once by the age of five. Gastroenteritis (cau…

Sample 11 Selected
  
Candidate #21 score=0.126
  
Nestled in the leafy suburbs of western Berlin, the Wannsee Conference House stands as a poignant reminder of a dark chapter in human history. The Wannsee Conference: A Pivotal Moment The Wannsee Conference, held on January 20, 1942, marked a pivotal moment in the implementation of Nazi Germany’s genocidal plans. Organized by SS-Obergrupp…

Sample 12 Selected
  
Candidate #24 score=0.108
  
Next we will talk about solar radiation, that is, the forms of solar radiation that we receive on earth. Solar radiation is generated by a series of nuclear fusion reactions that occur in the Sun and, as a consequence, emit electromagnetic radiation that reaches the earth. This radiation received by the earth’s surface is measured in W /…

Sample 13 Selected
  
Candidate #8 score=0.107
  
The Unsung Heroes of Your HVAC System: Understanding the Importance of Filters When it comes to your HVAC (Heating, Ventilation, and Air Conditioning) system, you might be quick to think about the thermostat, air ducts, or even the unit itself. However, there’s an unsung hero in your HVAC system that plays a pivotal role in maintaining in…

Sample 14 Selected
  
Candidate #17 score=0.080
  
Political Parties and Elections Political parties are an established part of modern mass democracy, and the conduct of elections in India is largely dependent on the behaviour of political parties. Although many candidates for Indian elections are independent, the winning candidates for Lok Sabha and Vidhan Sabha elections usually stand a…

Sample 15 Selected
  
Candidate #12 score=0.067
  
Can you please give us a little short bio? (education, professional experiences, select publications, academic specialty, awards won) Public school teacher for 5 years BA art (UC Irvine) PhD. (UCLA) educational psychology Professor of Child Development, (25 years) CSUS Senior Research Scientist (Oregon Research Institute with Institute of…

Sample 16 Selected
  
Candidate #7 score=0.041
  
Elizabeth Hurley played as Dalila Release: Dec 8, 1996 Mara and her husband Manoa are both upstanding and religious Israelites living under the harsh and unjust rule of the Philistines. Much to their regret, they have not been able to have children. One day, a mysterious stranger appears to Mara and promises her that she will bear a son w…

Sample 17 Not selected
  
Candidate #9 score=0.030
  
“Last night three cargoes of Bohea Tea were emptied into the sea. This is the most magnificent movement of all. There is a dignity, a majesty, a sublimity, in this last effort of the Patriots that I greatly admire.” - John Adams, diary entry, December 17, 1773 - John Adams, diary entry, December 17, 1773 A Novel Idea Is something so new a…

Sample 18 Not selected
  
Candidate #5 score=0.027
  
Well this is the big one. So big apparently, that I had to take it there and raise the number from 10 to 15. There’s just that many fails in the world of female rap. Some slight missteps, some EPIC. Nevertheless, they are all worth mentioning. You can probably think of a bunch more, but this is what I have gathered picking up from my prev…

Sample 19 Not selected
  
Candidate #10 score=0.026
  
Deforestation isn’t just happening in well-known global hotspots like Indonesia and Brazil’s rainforest. A new analysis says forests are also shrinking on state and private land in Oregon, where an estimated 522,000 acres of forest cover have disappeared since 2000. That’s an area six times larger than the city of Portland, equal to more…

Sample 20 Not selected
  
Candidate #4 score=0.024
  
5 Types of Women’s Underwear That Men Love Underwear can say a lot about a woman. It’s something that men are obsessed with, to the point that, a mere glimpse of a thong waistband causes us to go into shock. On the surface we find them sexy, revealing. We’re able to see who a woman actually is—or maybe some guys are just plain horny. Howe…

Sample 21 Not selected
  
Candidate #6 score=0.019
  
Skaters need to check their skate helmets every so often and ask yourself, ”Is it time to replace this helmet?” Well, that depends. Did you crash in it? For starters, most people are aware that you must replace a helmet after any crash where your head hit. The foam part of a helmet is made for one-time use, and after crushing once it is n…

Sample 22 Not selected
  
Candidate #23 score=0.012
  
The St. James kindergarteners have been working up to Project Week over the past month. We started slowly by taking walks in our neighborhood while Ms. Meghan and I noted what caught the children’s interest. It became apparent that the class was very interested in the L trains that they saw on our walks. It started with a simple question,…

Sample 23 Not selected
  
Candidate #19 score=0.012
  
Dividing Fractions Using Models Worksheet. This worksheet has six division with fractions issues to be solved — three must be solved with fashions and three with algorithms — options are on the second page. Answer key divide the unit fractions by whole numbers using th e fashions given. Use these resources to help reinforce the following…

Sample 24 Not selected
  
Candidate #13 score=0.00832
  
In Heart of Darkness it is the white invaders for instance, who are, almost without exception, embodiments of blindness, selfishness, and cruelty; and even in the cognitive domain, where such positive phrases as “to enlighten,” for instance, are conventionally opposed to negative ones such as “to be in the dark,” the traditional expectati…

Sample 25 Not selected
  
Candidate #11 score=0.00455
  
In decades past, classroom design was often an afterthought and followed a standardised layout. Plain boxed shaped classrooms, with identical chairs and tables throughout were commonplace in many schools. Read the latest issue of School News HERE Recently, though, there has been a shift away from this one-size-fits all approach to classro…

Sample 26 Not selected
  
Candidate #22 score=0.00335
  
How To Choose Decodable Readers for First Grade To decode or not to decode: really, there is no question. To help rising first graders become successful and enthusiastic readers this summer, decodable readers are essential reading resources. Although “decodable text” might sound like yet another form of educational lingo, parents and educ…

Sample 27 Not selected
  
Candidate #2 score=0.00324
  
With the advent of new technologies for sneakers such as Vac Tech, Hyperfuse and Flyknit, the mid 90s and early 2000s methods of production and designing are becoming obsolete in this sneaker world. Nike Running is the future for Nike, generating billions of dollars per year, and we see Nike also not afraid to experiment with technology s…

Sample 28 Not selected
  
Candidate #18 score=0.00124
  
This article originally appeared in the December 2015 issue of Resource Recycling. Subscribe today for access to all print content. Since the 1990s, curbside and drop-off recycling has grown substantially – nearly 90 percent of households now have access, according to recent surveys from Moore Recycling Associates, the American Forest and…

Sample 29 Not selected
  
Candidate #0 score=0.00109
  
As it turns out, the exercises synonymous with strong, attractive abs may not be the best way to train your core—and may be doing damage to your back. Read more If you are worried about the excess holiday pounds many of us are still carrying around. There are a few easy, natural things you can do to shed them, and none of them requires an…

Sample 30 Not selected
  
Candidate #14 score=0.000783
  
Is your major sustainable enough? Whether you’re pursuing a sustainability degree and want to further your knowledge, or are interested in supplementing your major in another area with sustainability education, plenty of independent learning resources are available. A wide range of credit and noncredit courses—including university- and or…

Sample 31 Not selected
  
Candidate #1 score=0.000569
  
Wedding & Party Venues - Sort By: Edgartown : (508) 627-9510 A 19th century gothic revival home transformed into the island’s premier eco-boutique hotel. Guests either stay in the 17-room Hob Knob hotel or in the privacy of their own Hob Knob House. Guests can expect individualized Hob Knob hospitality and modern luxury amenities in a rel…

Sample 32 Not selected
  
Candidate #25 score=0.00016
  
KS2 Maths is an important core subject in the National Curriculum and this area of the website covers all the major aspects of the curriculum including numbers, calculations, problems and measures. Each subject area is designed to help children develop their knowledge, whether they are learning in a classroom or home schooling environment…

DSIR
  



Sample 1 Selected
  
Candidate #9 score=11.70
  
“Last night three cargoes of Bohea Tea were emptied into the sea. This is the most magnificent movement of all. There is a dignity, a majesty, a sublimity, in this last effort of the Patriots that I greatly admire.” - John Adams, diary entry, December 17, 1773 - John Adams, diary entry, December 17, 1773 A Novel Idea Is something so new a…



Sample 2 Selected
  
Candidate #26 score=8.06
  
Unveiling the Power: Key Provisions of the Civil Rights Act of 1864 What were the Civil Rights Act of 1864’s key provisions? The Civil Rights Act of 1864 was a pivotal moment in American history, establishing crucial legal protections for African Americans in the face of rampant discrimination. Editor Note: The Civil Rights Act of 1864 la…

Sample 3 Selected
  
Candidate #5 score=4.71
  
Well this is the big one. So big apparently, that I had to take it there and raise the number from 10 to 15. There’s just that many fails in the world of female rap. Some slight missteps, some EPIC. Nevertheless, they are all worth mentioning. You can probably think of a bunch more, but this is what I have gathered picking up from my prev…

Sample 4 Selected
  
Candidate #4 score=4.62
  
5 Types of Women’s Underwear That Men Love Underwear can say a lot about a woman. It’s something that men are obsessed with, to the point that, a mere glimpse of a thong waistband causes us to go into shock. On the surface we find them sexy, revealing. We’re able to see who a woman actually is—or maybe some guys are just plain horny. Howe…

Sample 5 Selected
  
Candidate #28 score=3.89
  
You really have to be alert when studying science. Galaxies were created after matter. The stars in those galaxies were supposed to move slowly because there was more mass in the center of the galaxy. However, after dark matter was added, the stars appeared to move faster; however, this is not the case in our galaxy, suggesting that there…

Sample 6 Selected
  
Candidate #0 score=3.42
  
As it turns out, the exercises synonymous with strong, attractive abs may not be the best way to train your core—and may be doing damage to your back. Read more If you are worried about the excess holiday pounds many of us are still carrying around. There are a few easy, natural things you can do to shed them, and none of them requires an…

Sample 7 Selected
  
Candidate #3 score=3.39
  
starring John Travolta and Sam Jackson The first thing to understand about Basic –the basic thing, let’s say– is that although the commercials make it look like a war movie, it is not, for which we can all be grateful. No, Basic is a plot-twisty whodunnit. If The Usual Suspects died, and its body turned to cheese, and then that cheese-b…

Sample 8 Selected
  
Candidate #12 score=3.30
  
Can you please give us a little short bio? (education, professional experiences, select publications, academic specialty, awards won) Public school teacher for 5 years BA art (UC Irvine) PhD. (UCLA) educational psychology Professor of Child Development, (25 years) CSUS Senior Research Scientist (Oregon Research Institute with Institute of…

Sample 9 Selected
  
Candidate #2 score=2.96
  
With the advent of new technologies for sneakers such as Vac Tech, Hyperfuse and Flyknit, the mid 90s and early 2000s methods of production and designing are becoming obsolete in this sneaker world. Nike Running is the future for Nike, generating billions of dollars per year, and we see Nike also not afraid to experiment with technology s…

Sample 10 Selected
  
Candidate #18 score=2.51
  
This article originally appeared in the December 2015 issue of Resource Recycling. Subscribe today for access to all print content. Since the 1990s, curbside and drop-off recycling has grown substantially – nearly 90 percent of households now have access, according to recent surveys from Moore Recycling Associates, the American Forest and…

Sample 11 Selected
  
Candidate #15 score=2.07
  
Origami is an art form that combines precision, creativity, and patience. While basic origami is obtainable to every one, mastering complex origami designs can be quite a rewarding and impressive achievement. In this article, we’ll show you with the procedure for creating intricate origami while highlighting essential techniques for achie…

Sample 12 Selected
  
Candidate #11 score=2.06
  
In decades past, classroom design was often an afterthought and followed a standardised layout. Plain boxed shaped classrooms, with identical chairs and tables throughout were commonplace in many schools. Read the latest issue of School News HERE Recently, though, there has been a shift away from this one-size-fits all approach to classro…

Sample 13 Selected
  
Candidate #31 score=1.04
  
One of the challenges of working with ancient DNA samples is that damage accumulates over time, breaking the double helix structure into ever-smaller fragments. In the samples we worked with, these fragments were scattered and mixed with contaminants, making genome reconstruction a major technical challenge. But a shocking paper published…

Sample 14 Selected
  
Candidate #24 score=0.518
  
Next we will talk about solar radiation, that is, the forms of solar radiation that we receive on earth. Solar radiation is generated by a series of nuclear fusion reactions that occur in the Sun and, as a consequence, emit electromagnetic radiation that reaches the earth. This radiation received by the earth’s surface is measured in W /…

Sample 15 Selected
  
Candidate #22 score=0.487
  
How To Choose Decodable Readers for First Grade To decode or not to decode: really, there is no question. To help rising first graders become successful and enthusiastic readers this summer, decodable readers are essential reading resources. Although “decodable text” might sound like yet another form of educational lingo, parents and educ…

Sample 16 Selected
  
Candidate #10 score=0.378
  
Deforestation isn’t just happening in well-known global hotspots like Indonesia and Brazil’s rainforest. A new analysis says forests are also shrinking on state and private land in Oregon, where an estimated 522,000 acres of forest cover have disappeared since 2000. That’s an area six times larger than the city of Portland, equal to more…

Sample 17 Not selected
  
Candidate #30 score=0.272
  
Over 1.8 million professionals use CFI to learn accounting, financial analysis, modeling and more. Start with a free account to explore 20+ always-free courses and hundreds of finance templates and cheat sheets. What is the Central Limit Theorem (CLT)? The Central Limit Theorem (CLT) is a statistical concept that states that the sample me…

Sample 18 Not selected
  
Candidate #29 score=-0.299
  
Earthquakes are the result of sudden movement along faults within the Earth. The movement releases stored-up ‘elastic strain’ energy in the form of seismic waves, which propagate through the Earth and cause the ground surface to shake. Such movement on the faults is generally a response to long-term deformation and the buildup of stress.…

Sample 19 Not selected
  
Candidate #6 score=-0.307
  
Skaters need to check their skate helmets every so often and ask yourself, ”Is it time to replace this helmet?” Well, that depends. Did you crash in it? For starters, most people are aware that you must replace a helmet after any crash where your head hit. The foam part of a helmet is made for one-time use, and after crushing once it is n…

Sample 20 Not selected
  
Candidate #23 score=-0.429
  
The St. James kindergarteners have been working up to Project Week over the past month. We started slowly by taking walks in our neighborhood while Ms. Meghan and I noted what caught the children’s interest. It became apparent that the class was very interested in the L trains that they saw on our walks. It started with a simple question,…

Sample 21 Not selected
  
Candidate #8 score=-0.675
  
The Unsung Heroes of Your HVAC System: Understanding the Importance of Filters When it comes to your HVAC (Heating, Ventilation, and Air Conditioning) system, you might be quick to think about the thermostat, air ducts, or even the unit itself. However, there’s an unsung hero in your HVAC system that plays a pivotal role in maintaining in…

Sample 22 Not selected
  
Candidate #14 score=-0.743
  
Is your major sustainable enough? Whether you’re pursuing a sustainability degree and want to further your knowledge, or are interested in supplementing your major in another area with sustainability education, plenty of independent learning resources are available. A wide range of credit and noncredit courses—including university- and or…

Sample 23 Not selected
  
Candidate #20 score=-0.804
  
Conduct Disorder (CD) is a complex and serious behavioural and emotional disorder that can occur in children and adolescents. It’s characterised by a repetitive and persistent pattern of behaviour where the basic rights of others or major age-appropriate societal norms or rules are violated. Here’s an outline of Conduct Disorder in line w…

Sample 24 Not selected
  
Candidate #1 score=-0.923
  
Wedding & Party Venues - Sort By: Edgartown : (508) 627-9510 A 19th century gothic revival home transformed into the island’s premier eco-boutique hotel. Guests either stay in the 17-room Hob Knob hotel or in the privacy of their own Hob Knob House. Guests can expect individualized Hob Knob hospitality and modern luxury amenities in a rel…

Sample 25 Not selected
  
Candidate #7 score=-0.971
  
Elizabeth Hurley played as Dalila Release: Dec 8, 1996 Mara and her husband Manoa are both upstanding and religious Israelites living under the harsh and unjust rule of the Philistines. Much to their regret, they have not been able to have children. One day, a mysterious stranger appears to Mara and promises her that she will bear a son w…

Sample 26 Not selected
  
Candidate #13 score=-1.18
  
In Heart of Darkness it is the white invaders for instance, who are, almost without exception, embodiments of blindness, selfishness, and cruelty; and even in the cognitive domain, where such positive phrases as “to enlighten,” for instance, are conventionally opposed to negative ones such as “to be in the dark,” the traditional expectati…

Sample 27 Not selected
  
Candidate #25 score=-1.26
  
KS2 Maths is an important core subject in the National Curriculum and this area of the website covers all the major aspects of the curriculum including numbers, calculations, problems and measures. Each subject area is designed to help children develop their knowledge, whether they are learning in a classroom or home schooling environment…

Sample 28 Not selected
  
Candidate #21 score=-1.61
  
Nestled in the leafy suburbs of western Berlin, the Wannsee Conference House stands as a poignant reminder of a dark chapter in human history. The Wannsee Conference: A Pivotal Moment The Wannsee Conference, held on January 20, 1942, marked a pivotal moment in the implementation of Nazi Germany’s genocidal plans. Organized by SS-Obergrupp…

Sample 29 Not selected
  
Candidate #16 score=-1.98
  
What is rotavirus and why does my baby need to be immunised? Rotavirus is a very infectious virus that causes the majority of serious cases of gastroenteritis in babies. It causes diarrhoea, vomiting and abdominal pain, usually lasting around a week. Most children will be infected by rotavirus once by the age of five. Gastroenteritis (cau…

Sample 30 Not selected
  
Candidate #17 score=-2.02
  
Political Parties and Elections Political parties are an established part of modern mass democracy, and the conduct of elections in India is largely dependent on the behaviour of political parties. Although many candidates for Indian elections are independent, the winning candidates for Lok Sabha and Vidhan Sabha elections usually stand a…

Sample 31 Not selected
  
Candidate #27 score=-4.50
  
24/7 writing help on your phone Save to my list Remove from my list In the tumultuous 19th century, both Italy and Germany found themselves fragmented into numerous separate ruling states. The impetus for change came in the form of rising nationalism and liberalism, paving the way for the unification of these disparate entities. However,…

Sample 32 Not selected
  
Candidate #19 score=-8.76
  
Dividing Fractions Using Models Worksheet. This worksheet has six division with fractions issues to be solved — three must be solved with fashions and three with algorithms — options are on the second page. Answer key divide the unit fractions by whole numbers using th e fashions given. Use these resources to help reinforce the following…
