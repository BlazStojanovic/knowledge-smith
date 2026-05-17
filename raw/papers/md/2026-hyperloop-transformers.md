---
arxiv: '2604.21254'
authors:
- Abbas Zeitoun
- Lucas Torroba-Hennigen
- Yoon Kim
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Hyperloop Transformers
url: https://arxiv.org/abs/2604.21254
year: 2026
---

# Hyperloop Transformers

Abbas Zeitoun    Lucas Torroba-Hennigen    Yoon Kim
Affiliation: 
Massachusetts Institute of Technology
Affiliation: {zeitoun,lucastor,yoonkim}@mit.edu

###### Abstract

LLM architecture research generally aims to maximize model quality subject to fixed compute/latency budgets. However, many applications of interest such as edge and on-device deployment are further constrained by the model’s memory footprint, thus motivating *parameter-efficient* architectures for language modeling. This paper describes a simple architecture that improves the parameter-efficiency of LLMs. Our architecture makes use of looped Transformers as a core primitive, which reuse Transformer layers across depth and are thus more parameter-efficient than ordinary (depth-matched) Transformers. We organize the looped Transformer into three blocks—begin, middle, and end blocks—where each block itself consists of multiple Transformer layers, and only the middle block is applied recurrently across depth. We augment the looped middle block with *hyper-connections* (mHC), which expand the residual stream into matrix-valued residual streams. Hyper-connections are applied only after each loop, and therefore add minimal new parameters and compute cost. Across various model scales, we find that our *Hyper-Connected Looped Transformer (Hyperloop Transformer)* is able to outperform depth-matched Transformer and mHC Transformer baselines despite using approximately 50% fewer parameters. The outperformance persists through post-training weight quantization, thus positioning Hyperloop Transformers as an attractive architecture for memory-efficient language modeling.

## 1 Introduction

Pushing the Pareto frontier of performance and efficiency is a major goal of modern LLM architecture research. In cloud deployment, efficiency is measured primarily by latency, which depend on both computation and data movement through the memory hierarchy. Because memory is relatively abundant in such environments, a model’s memory footprint is often a secondary concern relative to compute and data movement. This makes parameter-*in*efficient architectures such as mixture-of-experts (MoE; shazeer2017outrageously) viable for cloud deployment. In contrast, edge and on-device deployments are often constrained not only by compute, but also by the total amount of available memory, which is often orders of magnitude smaller. For example, modern smartphones typically have 8GB–16GB of RAM.
In such settings, a model’s memory footprint becomes a major bottleneck, since it directly affects whether a model can be stored and executed at all. Even in cloud deployment, fitting a model on fewer accelerators can reduce communication overhead and simplify serving. Looking ahead, frontier models may become large enough that total parameter memory becomes a first-class constraint even in data-center settings. These factors motivate the study of *parameter-efficient architectures* for language modeling, where the goal is to push the performance-memory frontier for a given compute constraint.

*Looped Transformers*111Other terminology used to describe looped Transformers include *universal Transformers* (dehghani2018universal; tan2023sparse), *recursive Transformers* (bae2024relaxed; bae2025mixture) , and *recurrent-depth Transformers* (geiping2025scaling; pappone2025two). are Transformers that share parameters across depth, and thus enable greater parameter-efficiency than ordinary Transformers. When the number of loops is variable, they have also been shown to overcome certain theoretical limitations of fixed-depth Transformers (giannou2023looped; yang2023looped; xu2024expressive), and recent empirical work suggests that they can perform particularly well on some real-world reasoning tasks (geiping2025scaling; zhu2025scaling). However, when matched for depth, looped Transformers still generally underperform unlooped baselines especially from a perplexity standpoint (saunshi2025reasoning).

This paper develops a simple looped architecture that outperforms depth-matched Transformer baselines while using approximately half the parameters. Following prior work (bae2025mixture), we adopt a “middle cycle” strategy where we organize the Transformer into begin, middle, and end blocks, and only loop the middle block. We then incorporate a variant of *hyper-connections* (HC; mHC), which expand the residual stream into multiple streams, into (only) the looped block. Specifically, we apply hyper-connections at the loop level (i.e., only after each loop iteration) instead of at the layer-level, thus incurring minimal additional parameters and compute. We find that our *Hyper-Connected Looped Transformer (Hyperloop Transformer)* improves the performance-parameter frontier, achieving lower perplexities than depth-matched ordinary Transformers with 240M, 1B, and 2B parameters, despite using 50% fewer parameters. These gains persist through post-training quantization of the model’s weights, thus positioning Hyperloop Transformers as an attractive alternative to ordinary Transformers for memory-efficient language modeling.

## 2 Background

### 2.1 Looped Transformers

For a length TT input, a Transformer transforms input representations at layer 𝐗(l)∈ℝT×C\mathbf{X}^{(l)}\in\mathbb{R}^{T\times C} to obtain the output 𝐗(l+1)∈ℝT×C\mathbf{X}^{(l+1)}\in\mathbb{R}^{T\times C} through an attention layer followed by an MLP layer,

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐇(l)=Attention​(𝐗(l);θattn(l))+𝐗(l),\displaystyle\mathbf{H}^{(l)}=\text{Attention}(\mathbf{X}^{(l)};\theta\_{\text{attn}}^{(l)})+\mathbf{X}^{(l)}, | 𝐗(l+1)=MLP​(𝐗(l);θmlp(l))+𝐇(l).\displaystyle\mathbf{X}^{(l+1)}=\text{MLP}(\mathbf{X}^{(l)};\theta\_{\text{mlp}}^{(l)})+\mathbf{H}^{(l)}. |  |

Here θattn(l),θMLP(l)\theta^{(l)}\_{\text{attn}},\theta^{(l)}\_{\text{MLP}} are the layer-specific parameters for multiheaded attention and the feedforward layers respectively.222The LayerNorm parameters are absorbed into the attention/MLP layers. Letting ℱl​(⋅)\mathcal{F}\_{l}(\cdot) be the application of a Transformer layer ll, an LL-layer Transformer then obtains the final output via 𝐗(L)=ℱL​(…​ℱ2​(ℱ1​(𝐗(1)))​…)\mathbf{X}^{(L)}=\mathcal{F}\_{L}(\dots\mathcal{F}\_{2}(\mathcal{F}\_{1}(\mathbf{X}^{(1)}))\dots). Looped Transformers share parameters across depth, e.g., a fully looped model would have 𝐗(L)=ℱ1​(…​ℱ1​(ℱ1​(𝐗(1)))​…)\mathbf{X}^{(L)}=\mathcal{F}\_{1}(\dots\mathcal{F}\_{1}(\mathcal{F}\_{1}(\mathbf{X}^{(1)}))\dots). More recent works have shown that a “middle cycle” strategy, which partitions the Transformer layers into beginning, middle, and end blocks333The begin/end layers are also called prelude/coda or encoder/decoder blocks in the literature. and only loops the middle block, is particularly effective (bae2025mixture; saunshi2025reasoning). We also adopt this middle cycle strategy in our architecture.

### 2.2 Hyper-Connected Transformers

As shown above, each layer of a Transformer adds to the CC-dimensional *residual stream*. Hyper-connected Transformers (HC) expand the residual stream to an n×Cn\times C dimensional matrix through “hyper-connections”. In the more recent *manifold-constrained hyper-connections* (mHC; mHC), the residual stream at time step tt at depth ll (given by 𝐱t(l)∈ℝC\mathbf{x}\_{t}^{(l)}\in\mathbb{R}^{C}) is expanded by an expansion factor nn to yield nn parallel residual streams 𝐲t(l)∈ℝn×C\mathbf{y}\_{t}^{(l)}\in\mathbb{R}^{n\times C}. This expanded residual stream is then read from, written to, and mixed using input-dependent projections 𝐇l,tpre\mathbf{H}\_{l,t}^{\text{pre}}, 𝐇l,tpost\mathbf{H}\_{l,t}^{\text{post}}, and 𝐇l,tres\mathbf{H}\_{l,t}^{\text{res}}. Specifically,
the transformations at depth ll can be computed as follows:

|  |  |  |
| --- | --- | --- |
|  | 𝐳t(l)=RMSNorm⁡(flatten​(𝐲t(l))),\displaystyle{\mathbf{z}}\_{t}^{(l)}=\operatorname{RMSNorm}(\text{flatten}({\mathbf{y}}\_{t}^{(l)})), |  |
|  |  |  |
| --- | --- | --- |
|  | 𝐇l,tpre=σ​(αlpre⋅(𝐖lpre​𝐳t(l))+𝐛lpre),\displaystyle{\mathbf{H}}\_{l,t}^{\text{pre}}=\sigma(\alpha\_{l}^{\text{pre}}\cdot(\mathbf{W}\_{l}^{\text{pre}}{\mathbf{z}}\_{t}^{(l)})+\mathbf{b}\_{l}^{\text{pre}}), |  |
|  |  |  |
| --- | --- | --- |
|  | 𝐇l,tpost=2⋅σ​(αlpost⋅(𝐖lpost​𝐳t(l))+𝐛lpost),\displaystyle{\mathbf{H}}\_{l,t}^{\text{post}}=2\cdot\sigma(\alpha\_{l}^{\text{post}}\cdot(\mathbf{W}\_{l}^{\text{post}}\mathbf{z}\_{t}^{(l)})+\mathbf{b}\_{l}^{\text{post}}), |  |
|  |  |  |
| --- | --- | --- |
|  | 𝐇l,tres=sinkhorn​(αlres⋅reshape​(𝐖lres​𝐳t(l))+𝐛lres).\displaystyle{\mathbf{H}}\_{l,t}^{\text{res}}=\text{sinkhorn}(\alpha\_{l}^{\text{res}}\cdot\text{reshape}(\mathbf{W}\_{l}^{\text{res}}{\mathbf{z}}\_{t}^{(l)})+\mathbf{b}\_{l}^{\text{res}}). |  |

Here 𝐖lpre∈ℝn×n​C\mathbf{W}\_{l}^{\text{pre}}\in\mathbb{R}^{n\times nC}, 𝐖lpost∈ℝn×n​C\mathbf{W}\_{l}^{\text{post}}\in\mathbb{R}^{n\times nC}, 𝐖lres∈ℝn2×n​C\mathbf{W}\_{l}^{\text{res}}\in\mathbb{R}^{n^{2}\times nC} are linear projections, αlpre\alpha\_{l}^{\text{pre}}, αlpost\alpha\_{l}^{\text{post}}, αlres∈ℝ\alpha\_{l}^{\text{res}}\in\mathbb{R} are learned scalars, 𝐛lpre∈ℝn\mathbf{b}\_{l}^{\text{pre}}\in\mathbb{R}^{n}, 𝐛lpost∈ℝn\mathbf{b}\_{l}^{\text{post}}\in\mathbb{R}^{n}, 𝐛lres∈ℝn×n\mathbf{b}\_{l}^{\text{res}}\in\mathbb{R}^{n\times n} are learned biases, and reshape​(⋅)\text{reshape}(\cdot) is an operator that converts an n2n^{2}-dimensional vector to an n×nn\times n matrix. Finally, sinkhorn​(⋅)\text{sinkhorn}(\cdot) applies the Sinkhorn-Knopp algorithm, which exponentiates the input and iteratively performs column- and row-normalization, ensuring that 𝐇l,tres\mathbf{H}\_{l,t}^{\text{res}} is doubly stochastic (i.e., on the Birkhoff polytope) in the limit. mHC find that 20 Sinkhorn-Knopp iterations are sufficient.

Given the input-dependent matrices 𝐇l,tpre∈ℝ1×n,𝐇l,tpost∈ℝn×1\mathbf{H}\_{l,t}^{\text{pre}}\in\mathbb{R}^{1\times n},\mathbf{H}\_{l,t}^{\text{post}}\in\mathbb{R}^{n\times 1}, 𝐇l,tres∈ℝn×n\mathbf{H}\_{l,t}^{\text{res}}\in\mathbb{R}^{n\times n} and a sub-layer ℱl∈{Attentionl,MLPl}\mathcal{F}\_{l}\in\{\text{Attention}\_{l},\text{MLP}\_{l}\} of a Transformer layer, mHC applies attention/MLP layers in a smaller residual stream of dimension CC via,444In practice mHC uses different input-dependent matrices for attention and MLP layers.

|  |  |  |
| --- | --- | --- |
|  | 𝐲t(l+1)=𝐇l,tres​𝐲t(l)+𝐇l,tpost​ℱl​(𝐇l,tpre​𝐲t(l)).\displaystyle\mathbf{y}^{(l+1)}\_{t}=\mathbf{H}\_{l,t}^{\text{res}}\mathbf{y}^{(l)}\_{t}+\mathbf{H}\_{l,t}^{\text{post}}\mathcal{F}\_{l}(\mathbf{H}\_{l,t}^{\text{pre}}\mathbf{y}\_{t}^{(l)}). |  |

Thus, mHC Transformers make it possible to work with a larger matrix-valued residual stream without incurring much additional compute (since the compute-heavy attention/MLP layers still work with CC-dimensional inputs/outputs).

!(/html/2604.21254/assets/x1.png)

Figure 1: (Left) A vanilla middle-cycle looped Transformer architecture with two loops. (Right) A Hyperloop Transformer, which uses parallel residual streams that are written to after each loop using hyper-connections (mHC).

## 3 Hyperloop Transformers

Our architecture, shown in [Figure 1](#S2.F1 "In 2.2 Hyper-Connected Transformers ‣ 2 Background ‣ Hyperloop Transformers"), is extremely simple. We partition the Transformer into begin, middle, and end blocks, and then apply (a modification of) hyper-connections at the loop-level when we loop the middle block.

Concretely, let 𝐗begin∈ℝT×C\mathbf{X}\_{\text{begin}}\in\mathbb{R}^{T\times C} be the residual stream after applying the begin block. We expand this to nn parallel streams by simply copying it nn times, thus giving 𝐘(0)∈ℝT×n×C\mathbf{Y}^{(0)}\in\mathbb{R}^{T\times n\times C}, which will serve as input to the hyper-connected looped block. We then compute the input-dependent matrices 𝐇0,tpre,𝐇0,tpost,𝐇0,tres∈ℝn×n\mathbf{H}\_{0,t}^{\text{pre}},\mathbf{H}\_{0,t}^{\text{post}},\mathbf{H}\_{0,t}^{\text{res}}\in\mathbb{R}^{n\times n} for all {𝐲t(0)}t=1T\{\mathbf{y}\_{t}^{(0)}\}\_{t=1}^{T} as above, but using a simpler parameterization of 𝐇0,tres\mathbf{H}\_{0,t}^{\text{res}} given by,

|  |  |  |
| --- | --- | --- |
|  | 𝐇0,tres=diag​(σ​(α0res⋅(𝐖0res​𝐳t(0))+𝐛0res)),\displaystyle{\mathbf{H}}\_{0,t}^{\text{res}}=\text{diag}(\sigma(\alpha\_{0}^{\text{res}}\cdot(\mathbf{W}\_{0}^{\text{res}}{\mathbf{z}}\_{t}^{(0)})+\mathbf{b}\_{0}^{\text{res}})), |  |

where 𝐖0res\mathbf{W}\_{0}^{\text{res}} is now an n×n​Cn\times nC matrix (instead of n2×n​Cn^{2}\times nC) and 𝐛0res∈ℝn\mathbf{b}\_{0}^{\text{res}}\in\mathbb{R}^{n}.

We use {𝐇0,tpre}t=1T\{\mathbf{H}\_{0,t}^{\text{pre}}\}\_{t=1}^{T} on 𝐘(0)\mathbf{Y}^{(0)} to obtain the CC-dimensional input to the middle block, apply the middle block, and then use {𝐇0,tpost}t=1T\{\mathbf{H}\_{0,t}^{\text{post}}\}\_{t=1}^{T} to project out into the n×Cn\times C residual stream. We add a “loop position embedding” 𝐞l∈ℝC\mathbf{e}\_{l}\in\mathbb{R}^{C} after the middle block, resulting in the recurrence,

|  |  |  |
| --- | --- | --- |
|  | 𝐲t(l+1)=𝐇l,tres​𝐲t(l)+𝐇l,tpost​(ℱ​(𝐇l,tpre​𝐲t(l))+𝐞l).\displaystyle\mathbf{y}^{(l+1)}\_{t}=\mathbf{H}\_{l,t}^{\text{res}}\mathbf{y}^{(l)}\_{t}+\mathbf{H}\_{l,t}^{\text{post}}\left(\mathcal{F}(\mathbf{H}\_{l,t}^{\text{pre}}\mathbf{y}\_{t}^{(l)})+\mathbf{e}\_{l}\right). |  |

This process continues for LL loops to obtain 𝐘(L)\mathbf{Y}^{(L)}. Finally we average 𝐘(L)\mathbf{Y}^{(L)} across the parallel streams to obtain 𝐗end∈ℝT×C\mathbf{X}\_{\text{end}}\in\mathbb{R}^{T\times C}, which is used as input to the end block.

Our approach differs from the original mHC in that (1) we use a simpler parameterization of 𝐇l,tres\mathbf{H}\_{l,t}^{\text{res}} that substitutes the sinkhorn​(⋅)\text{sinkhorn}(\cdot) operator over a dense matrix with a sigmoid over a diagonal matrix (which we found to be sufficient performance-wise while being more efficient), (2) we add a loop position embedding, which, when viewing the architecture as a “depth-wise RNN” with matrix-valued hidden states 𝐘(0)\mathbf{Y}^{(0)}, acts as the input at each time (i.e., loop) step, and (3) we only apply hyper-connections at the loop level, instead of after every attention/MLP layer (so an architecture with 3 loops would have 3 hyper-connections).
Our architecture can also be seen as a more flexible parameterization of looped Transformers, which allows parameters to vary slightly across loop iterations. Concretely, we have loop-specific parameters {𝐖lτ,𝐛lτ,αlτ,𝐞l}\{\mathbf{W}\_{l}^{\tau},\mathbf{b}\_{l}^{\tau},\alpha\_{l}^{\tau},\mathbf{e}\_{l}\} for τ∈{pre,post,res}\tau\in\{\text{pre},\text{post},\text{res}\} that can vary across loop iterations ll. While the number of additional parameters here is still minimal, we posit that this parameterization allows model representations to change in a more flexible manner compared to ordinary looped Transformers which strictly enforce parameter sharing across each loop iteration.

## 4 Empirical Study

### 4.1 Experimental Setup

We train Hyperloop Transformers at various scales along with depth-matched vanilla, looped, and mHC Transformer baselines on the FineWeb-Edu dataset (fineweb-edu). All models make use of SwiGLU MLP layers (shazeer2020glu) and RoPE embeddings (su2024roformer). We use 4 parallel residual streams for both the mHC and Hyperloop Transformers. For looped models, we allocate (roughly) 25%25\% of the available parameters to the begin block, 25%25\% of the parameters to the end block, and the remaining 50%50\% to the middle block, which is looped three times. This results in looped models that contain half as many parameters as their depth-matched baselines. We ablate on these choices in our ablation study.

We train models on 2.5×2.5\times the Chinchilla-optimal token count of the vanilla Transformer corresponding to their size (hoffmann2022training). We use the Llama-2 tokenizer to tokenize our data and AdamW as our optimizer, with a linear warmup and cosine decay learning rate schedule. Our full hyperparameters can be found in Appendix [A](#A1 "Appendix A Hyperparameters ‣ Hyperloop Transformers"). These hyperparameters are generally off-the-shelf hyperparameters that have been found to work well for ordinary Transformers, i.e., we did not do any hyperparameter tuning for our architecture.

### 4.2 Main Results

For perplexity we evaluate our models on a held-out set consisting of 50M tokens from the FineWeb-Edu dataset. These are shown in Table [1](#S4.T1 "Table 1 ‣ 4.2 Main Results ‣ 4 Empirical Study ‣ Hyperloop Transformers"). Our results show that while vanilla Looped Transformers can underperform depth-matched Transformer baselines, the Hyperloop Transformer only needs 150-300K extra parameters (compared to the vanilla Looped Transformer) to outperform both looped and non-looped depth-matched baseline models.

While perplexity provides a more robust measure of performance at this scale, we also evaluate our models on downstream tasks. Specifically, we evaluate our models on ARC (clark2018thinksolvedquestionanswering), COPA (gordon-etal-2012-semeval), HellaSwag (zellers-etal-2019-hellaswag), LAMBADA (paperno-etal-2016-lambada), OpenBookQA (OpenBookQA2018), PIQA (Bisk2020), RACE (lai-etal-2017-race), SciQ (Welbl2017CrowdsourcingMC), and WinoGrande (sakaguchi2019winogrande). Interestingly, we find that the looped Transformer also outperforms the vanilla Transformer on most tasks, despite using 50%50\% fewer parameters and despite underperforming the Transformer model in perplexity terms. This outperformance corroborates similar findings reported in the literature (saunshi2025reasoning). Hyperloop Transformer outperforms all other baselines overall. Results broken down by task can be found in Appendix [B](#A2 "Appendix B Downstream Task Evaluations ‣ Hyperloop Transformers").

|  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | Dim | Unrolled  Depth | Train  Tokens | Params | | PPL  (BF16) | PPL  (INT4) | Task  Acc | Training  Toks/s |
| Transformer | 1024 | 16 | 12.5B | 238238 | M | 14.65 | 14.85 | 41.1% | 786K |
| mHC | 241241 | M | 14.55 | 14.73 | 41.1% | 514K |
| Looped [2L →\rightarrow 4L (×3)(\times 3) →\rightarrow 2L] | 135.5135.5 | M | 14.85 | 15.18 | 41.4% | 786K |
| Hyperloop [2L →\rightarrow 4L (×3)(\times 3) →\rightarrow 2L] | 135.7135.7 | M | 14.40 | 14.68 | 41.6% | 750K |
| Transformer | 2048 | 18 | 50B | 990.5990.5 | M | 10.19 | 10.27 | 48.0% | 367K |
| mHC | 997.5997.5 | M | 10.07 | 10.16 | 48.6% | 237K |
| Looped [3L →\rightarrow 4L (×3)(\times 3) →\rightarrow 3L] | 579.4579.4 | M | 10.02 | 10.24 | 49.2% | 367K |
| Hyperloop [3L →\rightarrow 4L (×3)(\times 3) →\rightarrow 3L] | 579.7579.7 | M | 9.65 | 9.81 | 49.8% | 354K |
| Transformer | 2048 | 38 | 100B | 20182018 | M | 8.60 | 8.71 | 52.8% | 181K |
| mHC | 20332033 | M | 8.57 | 8.62 | 53.7% | 109K |
| Looped [4L →\rightarrow 10L (×3)(\times 3) →\rightarrow 4L] | 990.5990.5 | M | 8.68 | 8.97 | 53.3% | 183K |
| Hyperloop [4L →\rightarrow 10L (×3)(\times 3) →\rightarrow 4L] | 990.8990.8 | M | 8.49 | 8.59 | 54.6% | 180K |

Table 1: Main results of our architecture and baselines pretrained on FineWeb-Edu. For looped models, [2L →\rightarrow 4L (×3)(\times 3) →\rightarrow 2L] means we have 2 begin layers, 4 middle layers looped 3 times, and 2 end layers. Perplexities are computed with both BF16 and INT4, where we use GPTQ to quantize to INT4. Task accuracies are based on BF16. Training throughput measures tokens/second and is based on eight H100s with NVLink.

#### Post-training quantization.

Post-training quantization of a model’s weights is a standard approach for reducing a model’s memory footprint. While looped models are *parameter*-efficient, models that are harder to quantize would be practically *memory*-inefficient. Insofar as models trained with more tokens have been shown to be generally harder to quantize (huang2024empirical; ouyang2024low), it is possible that looped models would also be harder to quantize because the looped layers are trained on “more” inputs. The interaction effect between looping and quantization has not been investigated before to the best of our knowledge. We quantize our models (originally trained in mixed precision with BF16 weights) using GPTQ (frantar-gptq) to INT4, where we modify the GPTQ algorithm so that the Hessian estimation for a looped layer aggregates activations across all inputs to that layer across loops. We use a calibration set of 1024 sequences from FineWeb-Edu, and use a group size of 128 for all model sizes. The perplexities of the resulting INT4 models are presented in Table [1](#S4.T1 "Table 1 ‣ 4.2 Main Results ‣ 4 Empirical Study ‣ Hyperloop Transformers"). Our results show that while looped Transformers can indeed be somewhat more sensitive to lower-precision quantization compared to non-looped models, hyper-connections help alleviate some of the performance degradation resulting from quantization. As a result, Hyperloop Transformers continue to perform well in the weight-only quantization setting.

#### Training efficiency.

Do the extra hyper-connections add training overhead? We measure the training throughput of each of the pretrained models on a single node with 8×H1008\times\text{H100} GPUs with NVLink and present the results in Table [1](#S4.T1 "Table 1 ‣ 4.2 Main Results ‣ 4 Empirical Study ‣ Hyperloop Transformers"). The models used for these measurements were implemented in PyTorch and compiled with torch.compile but without any further optimizations. Our results show that a straightforward PyTorch implementation of our approach only incurs minimal slowdown compared to the Transformer and Looped Transformer baselines. This can be attributed to only applying hyper-connections at the *inter-loop* level, and also using a simpler structure for 𝐇res\mathbf{H}^{\text{res}} resulting in very little memory and compute overhead.
On the other hand, a straightforward implementation of the mHC Transformer results in nontrivial slowdowns. This overhead can be brought down in theory with the proper low-level optimizations—for example, mHC report a 6.7% overhead with their specialized training kernel. However, this kernel is not publicly available to the best of our knowledge, and the fact that our approach adds almost no overhead without requiring any sophisticated systems engineering is an additional benefit.

|  |  |  |  |
| --- | --- | --- | --- |
|  |  | Train Tokens | |
| Model | Params | 12.5B | 100B |
| Transformer | 238.0 M | 14.65 | 12.15 |
| mHC | 241.0 M | 14.55 | 12.16 |
| Looped | 135.5 M | 14.85 | 12.56 |
| Hyperloop | 135.7 M | 14.40 | 12.19 |

Table 2: Perplexity results for our smallest (16 layer) models trained on more tokens.

#### Training for more tokens.

The number of tokens in our training set is 50×\times the number of parameters of the non-looped baseline models, i.e., 2.5×\times that of the compute-optimal 20×\times recipe suggested by hoffmann2022training. However, modern models are typically trained for many more tokens than is Chinchilla-optimal. For example, LLaMA3-8B (grattafiori2024llama) was trained on 15T tokens, while OLMo3-7B (olmo2025olmo) was trained on 6T tokens. Would the benefits of looped models diminish in such overtraining regimes? To investigate this, we train our smallest class of models, corresponding to 240M non-looped parameters, on 100B tokens from FineWeb-Edu. This corresponds to 20×\times the Chinchilla-optimal number of tokens, or approximately 400 training tokens per parameter. The results are presented in Table [2](#S4.T2 "Table 2 ‣ Training efficiency. ‣ 4.2 Main Results ‣ 4 Empirical Study ‣ Hyperloop Transformers"). We find that while looped Transformers underperform non-looped baselines, Hyperloop Transformers remain competitive with these baselines despite being in an overtrained setting.

### 4.3 Ablations

#### Number of loops.

Our main experiments use 3 loops. How does performance vary as a function of the number of loops, if we hold the parameters constant? We vary the number of loops from 2 to 6 for the 136M- and 579M-parameter looped and Hyperloop Transformers, and show the results in [Figure 2](#S4.F2 "In Number of loops. ‣ 4.3 Ablations ‣ 4 Empirical Study ‣ Hyperloop Transformers"). We observe diminishing returns as we increase the number of loops, although Hyperloop Transformer dominates the Looped Transformer in all cases.

The above experiments hold the parameter count constant and vary the depth by changing the number of loops. We next experiment with restructuring the middle loop by making it smaller but looping it more to keep the depth constant. The results are shown in Table [3](#S4.T3 "Table 3 ‣ Number of loops. ‣ 4.3 Ablations ‣ 4 Empirical Study ‣ Hyperloop Transformers"). We find that we can obtain even greater parameter-efficiency at the cost of a small performance hit. For example, our 477M parameter model still outperforms the full 1B Transformer.

We now conduct more ablations focusing on the 136M-parameter/12.5B-token setting.

!(/html/2604.21254/assets/x2.png)

Figure 2: Perplexity numbers as the number of loops is varied for the 135M (left) and 579M (right) parameter looped models. The non-looped Transformer baselines have 238M (left) and 991M (right) parameters. Each loop consists of 4 Transformer layers.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Model | Structure | Unrolled Depth | Params | PPL |
|  | 2L →\rightarrow 4L (×3)(\times 3) →\rightarrow 2L | 16 layers | 136 M | 14.85314.853 |
| Looped Transformer | 2L →\rightarrow 3L (×4)(\times 4) →\rightarrow 2L | 123 M | 15.18415.184 |
|  | 2L →\rightarrow 2L (×6)(\times 6) →\rightarrow 2L | 110 M | 15.76315.763 |
|  | 2L →\rightarrow 4L (×3)(\times 3) →\rightarrow 2L | 16 layers | 136 M | 14.40414.404 |
| Hyperloop Transformer | 2L →\rightarrow 3L (×4)(\times 4) →\rightarrow 2L | 123 M | 14.61814.618 |
|  | 2L →\rightarrow 2L (×6)(\times 6) →\rightarrow 2L | 110 M | 15.05615.056 |
|  | 3L →\rightarrow 4L (×3)(\times 3) →\rightarrow 3L | 18 layers | 579 M | 10.01910.019 |
| Looped Transformer | 3L →\rightarrow 3L (×4)(\times 4) →\rightarrow 3L | 528 M | 10.12410.124 |
|  | 3L →\rightarrow 2L (×6)(\times 6) →\rightarrow 3L | 477 M | 10.35710.357 |
|  | 3L →\rightarrow 4L (×3)(\times 3) →\rightarrow 2L | 18 layers | 579 M | 9.6489.648 |
| Hyperloop Transformer | 3L →\rightarrow 3L (×4)(\times 4) →\rightarrow 3L | 528 M | 9.7179.717 |
|  | 3L →\rightarrow 2L (×6)(\times 6) →\rightarrow 3L | 477 M | 9.8629.862 |

Table 3: Performance of the looped and Hyperloop Transformers as we vary the looping structure while keeping the depth fixed at 16 or 18 layers.

|  |  |
| --- | --- |
| Number of  Streams nn | PPL |
| 2 | 14.42914.429 |
| 4 | 14.40414.404 |
| 6 | 14.37914.379 |
| 8 | 14.38814.388 |
| 10 | 14.34914.349 |

Table 4: Hyperloop Transformer performance as we vary the number of parallel residual streams.

#### Number of parallel streams.

We pick n=4n=4 parallel residual streams as recommended by the original mHC work. Insofar as the number of parallel streams provides a parameter-efficient axis with which to scale up the model, can we get further gains by increasing nn? [Table 4](#S4.T4 "In Number of loops. ‣ 4.3 Ablations ‣ 4 Empirical Study ‣ Hyperloop Transformers") shows the results on a 135M-parameter Hyperloop Transformer with 3 loops and a varying number of parallel residual streams, where we observe diminishing returns on nn. Thus, while having matrix-valued residual streams does improve performance, this axis of scaling rapidly faces diminishing returns.

#### Number of hyper-connections.

Recall that our Hyperloop Transformer only applies hyper-connections after each loop, which results in minimal additional parameter/compute overhead. We ablate on the number of hyper-connections used within the looped block of a 135M-parameter Hyperloop Transformer.

|  |  |  |
| --- | --- | --- |
| Hyper | | PPL |
| connections | |
| 12 | [every layer] | 14.45 |
| 6 | [every 2 layers] | 14.50 |
| 4 | [every 3 layers] | 14.50 |
| 3 | [every loop (ours)] | 14.40 |
| 2 | [every 6 layers] | 14.50 |
| 1 | [every 12 layers] | 14.63 |

Table 5: Performance as we vary the number of HCs.

To do so, we fix the number of loops and only vary the number of hyper-connections, or equivalently, the number of Transformer blocks skipped over by a single hyper-connection. This results in some hyper-connections being applied within loops or across Transformer blocks from different loops in some setups.
The sub-layers within a Transformer block retain their own original skip-connections, even when hyper-connections are applied after every block.

|  |  |
| --- | --- |
| Parameterization | PPL |
| Identity | 14.61 |
| Sinkhorn | 14.59 |
| Diagonal [ours] | 14.40 |

Table 6: Ablations on parameterization of the transition matrix 𝐇res\mathbf{H}^{\text{res}}.

[Table 5](#S4.T5 "In Number of hyper-connections. ‣ 4.3 Ablations ‣ 4 Empirical Study ‣ Hyperloop Transformers") shows that applying hyper-connections after every loop (instead of every layer) is the most performant setup. This is perhaps unintuitive, given that the every-layer setup uses the most compute/parameters. Our results potentially indicate that at least in the looped case, one must be more careful about choosing where to apply hyper-connections.

#### 𝐇res\mathbf{H}^{\text{res}} parameterization.

The Hyperloop Transformer simplifies the mHC formulation by using a simpler diagonal transition matrix 𝐇res\mathbf{H}^{\text{res}} for the parallel residual stream, in contrast to the doubly-stochastic structure used in mHC. This leads to fewer parameters and less compute. Does this potentially hurt the performance of our approach? As shown in Table [6](#S4.T6 "Table 6 ‣ Number of hyper-connections. ‣ 4.3 Ablations ‣ 4 Empirical Study ‣ Hyperloop Transformers"), we find that this is not the case. The simplest identity parameterization only slightly underperforms the Sinkhorn parameterization, while our (data-dependent) diagonal parameterization further improves performance.

|  |  |  |
| --- | --- | --- |
| LoRA Rank | Params | PPL |
| 0 [Looped Transformer] | 135.5M | 14.85 |
| 4 | 136.0M | 14.85 |
| 8 | 137.0M | 14.81 |
| 16 | 139.0M | 14.80 |
| 32 | 143.0M | 14.77 |
| Transformer | 238.0M | 14.65 |
| Hyperloop | 136.0M | 14.40 |

Table 7: Experiments on allowing Transformer layers to change across loop iterations with LoRA.

#### Comparison vs. LoRA-Looped Transformers.

Since the weights for the hyper-connections are different across loops, our Hyperloop Transformer allows for the looped block to be slightly different across loops. Would using LoRA to modify the parameters across loops (as in relaxed recursive Transformers; bae2024relaxed) perform better? The results of these experiments are shown in Table [7](#S4.T7 "Table 7 ‣ 𝐇^\"res\" parameterization. ‣ 4.3 Ablations ‣ 4 Empirical Study ‣ Hyperloop Transformers"). We find that allowing parameters to change across loops with LoRA does help slightly, but the Hyperloop Transformer provides a much more parameter-efficient approach to improving performance.

### 4.4 Analysis

To better understand our model’s inner workings, we conduct a series of qualitative analyses of its internal representations on 50M tokens from the FineWeb-Edu dataset.

|  |  |  |
| --- | --- | --- |
| Params | Looped | Hyperloop |
| 136M | 0.74290.7429 | 0.73820.7382 |
| 579M | 0.91520.9152 | 0.87230.8723 |
| 991M | 0.92260.9226 | 0.87140.8714 |

Table 8: Average cosine similarity between corresponding layers across loop iterations.

#### Representation similarity.

We hypothesize that the outperformance of the Hyperloop architecture is supported in part by hyper-connections allowing for the model representations to be less constrained than in the ordinary looped case. To investigate this, we analyze the cosine similarity of the residual stream as we vary the depth in [Figure 3](#S4.F3 "In Representation similarity. ‣ 4.4 Analysis ‣ 4 Empirical Study ‣ Hyperloop Transformers").
We see that both looped models exhibit similarity within the looped blocks; in particular, we also see that the representations output by the same layer *across* loops exhibit higher-than-expected similarity.
[Table 8](#S4.T8 "In 4.4 Analysis ‣ 4 Empirical Study ‣ Hyperloop Transformers") quantifies the average similarity of layers across loops (e.g., comparing representations of middle layer 1 across loops) for all the looped layers. We find that Hyperloop models’ representations are indeed less similar, supporting our hypothesis.

!(/html/2604.21254/assets/x3.png)

Figure 3: Pairwise cosine similarity between inner residual streams at each (effective) layer, across model scales (rows) and architectures (columns).

#### Logit lens.

We also perform a logit lens-style analysis (logit-lens).
We observe that the “outer” residual streams
(i.e., the parallel streams in the mHC/Hyperloop Transformers; the regular stream in the other Transformers)
are loosely aligned to the vocabulary space.555For the Hyperloop model, we can compute the effective value of the parallel residual streams by performing an early merge operation from the intra-loop residual stream to the parallel stream.
We are thus able to push these representations through the language modeling head to get a distribution over the next token.
From this, we can compute the evolution of the cross-entropy, entropy, and the accuracy of the argmax of the distribution, as shown in [Figure 4](#S4.F4 "In Logit lens. ‣ 4.4 Analysis ‣ 4 Empirical Study ‣ Hyperloop Transformers").
We see that both Hyperloop and the vanilla looped transformer produce representations that are more aligned with the vocabulary distribution, likely as the looping forces the models to operate closer to the vocabulary space.
The fact that both looped models exhibit maximum alignment toward the vocabulary distribution at the end of a loop further corroborates this claim.
Interestingly, our approach produces models with *higher* alignment than the vanilla looped models, suggesting that the hyper-connections offer additional regularization in this direction. This potentially indicates that Hyperloop Transformers could be more amenable to early-exit-style inference strategies and enable compute savings.

!(/html/2604.21254/assets/x4.png)

Figure 4: Logit lens-inspired analysis across model scales. Each column corresponds to a model scale, and each row shows a different metric: average cross-entropy (top), average entropy of vocabulary distribution (middle), and greedy decoding accuracy (bottom), computed by mapping the outer residual stream via the language modeling head. Loop boundaries are indicated at the top of each panel, though they only apply to looped models.

## 5 Discussion

This work shows that combining recent hyper-connections (mHC) with looped Transformers can push the parameter-performance frontier of language models. We study several methods for incorporating hyper-connections, and show that doing so at the loop-level with a simple data-dependent diagonal transition matrix is effective, while incurring minimal additional parameters/compute. We find suggestive evidence that the outperformance of our approach is supported in part by the hyper-connections allowing the model representations of looped layers to deviate more flexibly. While we have primarily focused on parameter-efficiency controlling for compute, our logit lens analysis further suggests that Hyperloop Transformers could enable compute efficiency gains through early-exit style inference strategies.

Our main limitation is scale. While we performed experiments that were reasonable on academic compute, it is unclear as to whether the overall efficiency gain (i.e., Hyperloop Transformers matching Transformers with 50% fewer parameters) would hold at even larger scales, although we did find that Hyperloop Transformers were effective in the overtrained regime for the smaller models. While the present work was mostly motivated by pushing the performance-efficiency frontier, looped Transformers have been suggested as a better architecture for enabling test-time scaling and improved reasoning (saunshi2025reasoning; geiping2025scaling; zhangmodr; kohli2026loop). It would thus be interesting to train much deeper Hyperloop Transformers (with the hyper-connection parameters potentially shared across loops to enable generalization to longer loops than seen in training) to investigate its test-time scaling and reasoning capabilities.

## 6 Related Work

#### Looped Transformers.

Looped Transformers were first proposed by dehghani2018universal and applied to BERT-style models in ALBERT (lan2019albert). Modern variants of looped Transformers were initially studied in synthetic settings where they were found to generalize better on certain kinds of synthetic tasks (csordas2021devil; csordas2022neural; giannou2023looped; yang2023looped; xu2024expressive). However, more recent works have shown the empirical effectiveness of looped models on real language modeling. csordas2024moeut generalize universal Transformers to the mixture-of-experts case.
saunshi2025reasoning find that despite underperforming unlooped baselines from a perplexity standpoint, looped models perform better on certain kinds of reasoning tasks. kohli2026loop study synthetic multi-hop reasoning with looped models and find that they can generalize to more hops than seen in training. bae2024relaxed and mcleish2025teachingpretrainedlanguagemodels convert pretrained unlooped Transformers into a looped architecture. bae2025mixture propose a looped architecture that allocates a dynamic number of loops on a per-token basis. geiping2025scaling train a looped model on a variable number of loops and show that the model’s performance on downstream tasks scales with the number of loops at test-time. jeddi2026loopformer also enable flexible number of loops at inference time through conditioning on depth during training. zhu2025scaling train looped language models through all stages of a modern language modeling pipeline and propose an entropy-regularized objective for early exiting after a dynamic number of loops. yu2026spiralformer show the effectiveness of looping blocks after downscaling the input. prairie2026parcae develop a more stable parameterization of looped Transformers and derive scaling laws as the number of loops is varied, holding parameter count fixed (i.e., as in [Figure 2](#S4.F2 "In Number of loops. ‣ 4.3 Ablations ‣ 4 Empirical Study ‣ Hyperloop Transformers")). schwethelm2026how study scaling laws of looped Transformers by varying the looping structure and number of parameters.
Finally, blayeney2026mechanistic perform a mechanistic analysis of looped Transformers and find that the latent states follow a cyclic trajectory.

#### Residual connections in Transformers.

Our work is also related to approaches that modify the residual stream connection patterns in Transformers. HC and mHC expand the residual stream into a residual matrix, allowing richer connections from earlier to later layers in the model. pagliardini2024denseformerenhancinginformationflow expand the residual stream differently, averaging the hidden states at the output of a transformer block with earlier hidden states using different sparsity patterns. xiao2025muddformerbreakingresidualbottlenecks, heddes2025deepcrossattention, and kimiteam2026attentionresiduals take that a step further, allowing the model to attend to previous hidden states along the depth axis.
We leave the integration of looped Transformers with other residual connection patterns to future work.

## 7 Conclusion

We propose a simple architecture that combines hyper-connections with looped Transformers and improves the parameter-efficiency of language models while adding minimal additional compute at training and deployment.

## Acknowledgments

We thank Junhyun Lee, Munjo Kim, and Oliver Sieberling for helpful discussions. This study was supported by a Samsung Research grant and the AI2050 program at Schmidt Sciences (Grant G-25-67980).

## Appendix A Hyperparameters

We use a model dimension of 10241024 for our 240M/136M-parameter models and a model dimension of 20482048 for our larger models. We set our SwiGLU feed-forward dimension to be 2.75×2.75\times the model dimension across all model sizes. All model sizes use Multi-Head Attention with 16 attention heads and a RoPE base of 10000 for their position embeddings. We use a weight-untied unembedding matrix and include its number of parameters in our reported model sizes.

Our models are trained on batches of 256 sequences of length 2048, corresponding to 524K tokens per batch. Across all training runs, we use a max learning rate of 4×10−44\times 10^{-4}, with cosine decay down to 4×10−54\times 10^{-5}. We use 1000 warmup steps for our 240M/136M models and 2000 warmup steps for the larger models. For AdamW, we use (β1,β2)=(0.9,0.95)(\beta\_{1},\beta\_{2})=(0.9,0.95) and a weight decay of 0.10.1. We use gradient clipping for gradient norms above 1.01.0.

## Appendix B Downstream Task Evaluations

Table [9](#A2.T9 "Table 9 ‣ Appendix B Downstream Task Evaluations ‣ Hyperloop Transformers") shows the downstream task results broken down by tasks.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Model Parameters | Task | Transformer | mHC | Looped | Hyperloop |
| 240M (non-looped)  / 136M (looped) | ARC-Challenge | 19.45% | 21.25% | 19.71% | 20.56% |
| ARC-Easy | 49.24% | 49.79% | 49.45% | 50.63% |
| COPA | 62.00% | 60.00% | 62.00% | 63.00% |
| HellaSwag⋆ | 31.96% | 31.87% | 31.37% | 32.00% |
| LAMBADA (OpenAI) | 24.14% | 24.76% | 25.17% | 25.09% |
| LAMBADA (Standard) | 17.95% | 17.66% | 18.03% | 17.89% |
| OpenBookQA⋆ | 30.60% | 31.40% | 30.80% | 31.20% |
| PIQA⋆ | 61.53% | 61.37% | 60.94% | 63.33% |
| RACE | 29.19% | 26.32% | 29.86% | 28.71% |
| SciQ | 76.60% | 77.00% | 74.50% | 75.20% |
| WinoGrande | 49.88% | 50.12% | 53.04% | 50.20% |
| Average | 41.14% | 41.05% | 41.35% | 41.62% |
| 1B (non-looped)  / 579M (looped) | ARC-Challenge | 25.68% | 27.90% | 27.73% | 28.07% |
| ARC-Easy | 59.30% | 60.14% | 62.46% | 62.08% |
| COPA | 70.00% | 71.00% | 73.00% | 68.00% |
| HellaSwag⋆ | 42.07% | 42.80% | 43.98% | 46.22% |
| LAMBADA (OpenAI) | 36.64% | 36.11% | 37.16% | 39.05% |
| LAMBADA (Standard) | 28.20% | 27.94% | 28.18% | 30.62% |
| OpenBookQA⋆ | 34.40% | 33.40% | 33.40% | 33.80% |
| PIQA⋆ | 66.43% | 67.08% | 67.14% | 68.72% |
| RACE | 30.53% | 31.10% | 31.77% | 31.77% |
| SciQ | 82.80% | 83.90% | 84.70% | 86.30% |
| WinoGrande | 52.17% | 52.96% | 51.62% | 53.04% |
| Average | 48.02% | 48.58% | 49.19% | 49.79% |
| 2B (non-looped)  / 1B (looped) | ARC-Challenge | 31.66% | 31.74% | 32.17% | 33.70% |
| ARC-Easy | 68.18% | 66.96% | 68.60% | 69.02% |
| COPA | 70.00% | 74.00% | 72.00% | 74.00% |
| HellaSwag⋆ | 51.44% | 51.87% | 52.21% | 53.93% |
| LAMBADA (OpenAI) | 42.89% | 43.94% | 43.37% | 45.55% |
| LAMBADA (Standard) | 35.16% | 35.44% | 34.06% | 37.96% |
| OpenBookQA⋆ | 37.80% | 37.40% | 36.20% | 36.20% |
| PIQA⋆ | 71.49% | 70.89% | 71.33% | 71.27% |
| RACE | 32.54% | 34.64% | 32.34% | 33.59% |
| SciQ | 85.50% | 87.60% | 88.50% | 88.20% |
| WinoGrande | 53.83% | 56.35% | 55.41% | 57.06% |
| Average | 52.77% | 53.71% | 53.29% | 54.59% |

Table 9: Accuracies on downstream tasks across three model-size settings. For tasks marked with a ⋆\star, we normalize the log-likelihood of the different multiple-choice continuations by the number of tokens.
