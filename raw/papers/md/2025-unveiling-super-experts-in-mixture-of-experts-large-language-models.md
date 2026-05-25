---
arxiv: '2507.23279'
authors:
- Zunhai Su
- Qingyuan Li
- Hao Zhang
- Weihao Ye
- Qibo Xue
- YuLei Qian
- Yuchen Xie
- Ngai Wong
- Kehong Yuan
parser: ar5iv
retrieved: '2026-05-25'
source: paper
title: Unveiling Super Experts in Mixture-of-Experts Large Language Models
url: https://arxiv.org/abs/2507.23279
year: 2025
---

[2507.23279] Unveiling Super Experts in Mixture-of-Experts Large Language Models



# Unveiling Super Experts in Mixture-of-Experts Large Language Models

Zunhai Su12,
Qingyuan Li2,
Hao Zhang2,
Yulei Qian2,
Yuchen Xie2,
Kehong Yuan1
Work done as an intern at Meituan.
  
Email: zh-su23@mails.tsinghua.edu.cn

###### Abstract

Sparsely activated Mixture-of-Experts (MoE) models have shown promise in enhancing the learning capacity of large language models (LLMs), yet their substantial number of parameters presents significant challenges for the deployment of MoE LLMs.
Leveraging the intrinsic importance differences among experts, recent research has explored expert-level compression techniques to improve the efficiency of MoE LLMs.
However, existing approaches often rely on empirical criteria to identify critical experts, lacking a deeper exploration and understanding of the heterogeneous importance of experts.
In this study, we present the first discovery and investigation of a distinct subset of experts that play a crucial role in the underlying mechanisms during the model’s forward inference.
These experts are prevalent in open-source MoE LLMs, and despite their limited number, pruning them leads to a significant decline in model performance (e.g., pruning three causes Qwen3-30B-A3B to produce repetitive and uninformative outputs).
We refer to these experts as Super Experts (SEs).
Our comprehensive analysis provides progressively deeper insights into SEs.
(i) SEs are characterized by rare but extreme activation outliers in the output of the down\_proj, which give rise to massive activations in the hidden states between decoder layers.
Moreover, the distribution of SEs remains model-specific and is unaffected by post-training processes.
(ii) By pruning SEs, we assess their significance across a variety of tasks, revealing their considerable impact on the model’s overall performance, particularly in mathematical reasoning.
(iii) We further enhance our understanding of the influence of SEs compression.
Our findings confirm that MoE LLMs rely on SEs to induce attention sinks, which are crucial for the distribution of attention scores but are significantly disrupted by SE pruning.
We also develop an automated tool for the rapid and precise profiling of SEs in new MoE LLMs.
The code is available at https://github.com/ZunhaiSu/Super-Experts-Profilling.

Figure 1: Analysis of experts pruning on Qwen3-30B-A3B using the WikiText-2 dataset.
Pruning three Super Experts results in a significant degradation in perplexity (PPL).

## 1 Introduction

Sparsely activated Mixture-of-Experts (MoE) models employ dynamic routing and sparse activation, demonstrating significant potential in enhancing the learning capacity of large language models (LLMs) (Cai et al. [2024](#bib.bib9); Mu and Lin [2025](#bib.bib51)).
This paradigm has led to the development of state-of-the-art MoE LLMs, including DeepSeek (Guo et al. [2025](#bib.bib27); Liu et al. [2024b](#bib.bib44)), Qwen (Team [2024b](#bib.bib59); Yang et al. [2025](#bib.bib67)), and others (Meta [2025](#bib.bib49); Jiang et al. [2024](#bib.bib35)).
Despite their potential, a significant challenge with MoE LLMs stems from their large parameter size and high computational cost (Li et al. [2023](#bib.bib39); Lu et al. [2024](#bib.bib46); Chowdhury et al. [2024](#bib.bib13)), which present considerable obstacles for deployment.
Model compression techniques, such as quantization (Frantar et al. [2022](#bib.bib24); Xiao et al. [2023a](#bib.bib64); Su et al. [2025](#bib.bib55)) and pruning (Frantar and Alistarh [2023](#bib.bib23); Sun et al. [2023](#bib.bib57); Ma, Fang, and Wang [2023](#bib.bib47)), enable the development of more compact and computationally efficient models (Zhu et al. [2024](#bib.bib76); Wang et al. [2024](#bib.bib63)).

Beyond LLM-oriented compression approaches, expert-level compression methods have been developed by leveraging the structural characteristics of MoE models and the uneven importance of experts induced by training strategies (Chowdhury et al. [2024](#bib.bib13); Chi et al. [2022](#bib.bib12); Lu et al. [2024](#bib.bib46)).
Specifically, expert-level compression employs various expert importance metrics to guide the pruning, merging, or skipping of less critical experts (Lu et al. [2024](#bib.bib46); Huang et al. [2025](#bib.bib31); Xie et al. [2024](#bib.bib66)), prioritize more important ones by assigning higher bit budgets during quantization (Duanmu et al. [2025](#bib.bib22); Li et al. [2024](#bib.bib38); Zheng et al. [2025](#bib.bib75)), and allocate more ranks in low-rank decomposition (Yang et al. [2024](#bib.bib68); Li et al. [2023](#bib.bib39)).
For instance, several studies evaluate expert importance based on access frequency or other statistical features derived from the routers in MoE layers (Chen et al. [2022](#bib.bib11); Li et al. [2024](#bib.bib38), [2023](#bib.bib39); Huang et al. [2025](#bib.bib31)).

Analyzing expert importance not only facilitates model compression but also provides deeper insights into the inner workings of MoE LLMs.
However, existing approaches often rely on empirical criteria to identify critical experts, lacking a deeper exploration and understanding of the heterogeneous importance among experts.
In this study, we address a fundamental yet previously overlooked question:
Is there a distinct subset of experts that plays a critical role in the underlying mechanisms during the forward inference of MoE LLMs?
Through comprehensive analysis of various open-source MoE LLMs, we consistently confirm the existence of such experts.
Despite their limited number, pruning these experts leads to a significant collapse in model performance.
As shown in Figure [1](#S0.F1 "Figure 1 ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models"), pruning just three experts from Qwen3-30B-A3B (Yang et al. [2025](#bib.bib67)) leads to a significant degradation in model performance, while randomly pruning other experts results in a considerably smaller impact.
We refer to these experts as Super Experts (SEs), and our subsequent comprehensive analysis offers increasingly deeper insights into SEs.

In Section [3](#S3 "3 Super Experts: Discovery and Localization ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models"), we first characterize SEs and explore their distribution across different models and input data domains.
SEs are identified by rare but extreme activation outliers in the output of the down\_proj, which induce massive activations (Sun et al. [2024](#bib.bib56)) in the hidden states between decoder layers.
Intriguingly, the distribution of SEs remains model-specific, and the SEs in the base model maintain consistency after post-training processes.
For instance, Qwen3-30B-A3B-Base and Qwen3-30B-A3B exhibit identical SEs distribution patterns.
In Section [4](#S4 "4 The Importance of Super Experts ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models"), we assess the importance of SEs by quantifying performance degradation across multiple capabilities following their pruning.
For non-reasoning models, all few-shot evaluations we conducted demonstrate considerable performance degradation, with the most pronounced decline observed in the GSM8K dataset (Cobbe et al. [2021](#bib.bib17)).
Notably, for reasoning models, pruning SEs leads to a complete performance collapse, with Pass@1 dropping to nearly zero on tasks such as AIME and Math-500 (AIME [2024](#bib.bib1), [2025](#bib.bib2); Lightman et al. [2023](#bib.bib41)).
This highlights the critical role of SEs in the model’s overall performance, particularly in mathematical reasoning tasks.

In Section [5](#S5 "5 Understanding the Impact of Super Experts Compression ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models"), we further enhance our understanding of the influence of SE compression.
Our findings confirm that MoE LLMs rely on SEs to induce attention sinks, which are crucial for the distribution of attention scores and must be preserved during sparse attention or token compression (Xiao et al. [2023b](#bib.bib65); Su et al. [2025](#bib.bib55)).
We quantitatively assess the significant disruption of attention sinks caused by SE compression, as measured by the proposed attention sink decay rate.
Our contributions are summarized as follows:

* •

  We present the first discovery and investigation of Super Experts (SEs) in MoE LLMs.
  We identify and localize SEs across various prominent open-source MoE LLMs, thoroughly validating that their distribution remains model-specific and unaffected by post-training processes.
  Additionally, we develop an automated tool for rapid and precise profiling of SEs in new models.
* •

  We conduct extensive experiments to evaluate the importance of SEs, revealing their significant impact on the model’s overall performance, particularly in mathematical reasoning abilities.
* •

  We further enhance the understanding of SEs compression.
  Our findings confirm that MoE LLMs rely on SEs to induce attention sinks.
  Using the proposed attention sink decay rate, we quantitatively assess the significant disruption of attention sinks caused by SE pruning.
* •

  This discovery of SEs provides new insights into the inner workings of MoE LLMs and highlights the critical importance of preserving SEs during model compression. These insights offer valuable guidance for achieving more effective MoE LLM compression strategies.

Figure 2: Decoder Architecture of MoE LLM.

## 2 Preliminaries on MoE LLMs

#### MoE LLMs

LLMs are typically structured as a stack of Transformer decoder blocks (Vaswani et al. [2017](#bib.bib62)), each consisting of a multi-head self-attention (MHSA) layer and a feed-forward network (FFN) layer.
In MoE LLMs, the FFN layers are replaced by MoE layers, where each layer consists of multiple experts, each represented by a FFN.
A concise overview of MoE LLMs is presented in Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models").
Let H0∈ℝn×dH^{0}\in\mathbb{R}^{n\times d} represent the input to the first decoder, where dd is the embedding dimension, and nn is the length of the tokenized input sequence.
Then, the output of the ll-th decoder block, Hl∈ℝn×dH^{l}\in\mathbb{R}^{n\times d}, is given by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Hl=MoE​(LNm​o​e​(Hl′))+Hl′,H^{l}=\text{MoE}\left(\text{LN}\_{moe}\left(H^{l^{\prime}}\right)\right)+H^{l^{\prime}}, |  | (1) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | Hl′=Ol+Hl−1,Ol=MHSA​(LNm​h​s​a​(Hl−1)),H^{l^{\prime}}=O^{l}+H^{l-1},O^{l}=\text{MHSA}\left(\text{LN}\_{mhsa}\left(H^{l-1}\right)\right), |  | (2) |

where 1≤l≤L1\leq l\leq L, with LL denoting the total number of blocks.
LN refers to layer normalization, OlO^{l} representing the output of the MHSA, and Hl′H^{l^{\prime}} denoting residual summations after the MHSA.

Figure 3: Super Experts mechanism in Qwen3-30B-A3B.
The line plots depict the maximum output magnitudes of down\_proj for experts 68/92/82 across layers.
Massive activation is gradually amplified through expert 68 in layer 1, expert 92 in layer 2, and expert 82 in layer 3.
Extreme activation outliers from these Super Experts are propagated into the hidden states between decoders via residual summation, progressively leading to massive activation.




Figure 4: Impact of SEs pruning on massive activations in Qwen3-30B-A3B.
Massive activations are computed using 100 input samples from the C4 (Raffel et al. [2020](#bib.bib52)) dataset, each with a length of 2K.



| Model | Total Experts | SE Count | SE Proportion | Top 1 | Top 2 | Top 3 | Top 4 | Top 5 | Top 10 | Top 0.5% | Top 1 \* 0.1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-30B-3B | 6144 | 3 | 0.05% | 744.0 | 540.0 | 430.0 | 63.5 | 19.1 | 12.1 | 7.3 | 74.4 |
| DeepSeek-R1 | 15677 | 10 | 0.06% | 616.0 | 536.0 | 171.0 | 143.0 | 143.0 | 67.0 | 36.75 | 61.6 |
| DeepSeek-V2-Lite-Chat | 1782 | 2 | 0.22% | 1424.0 | 462.0 | 112.5 | 89.5 | 37.5 | 24.0 | 34.5 | 142.4 |
| Mixtral-8x7B-Instruct-v0.1 | 256 | 1 | 0.39% | 5600.0 | 302.0 | 286.0 | 258.0 | 253.0 | 139.0 | 5600.0 | 560 |

Table 1: The activations identified as Super Experts are highlighted in bold.

#### MHSA Layer

After LN, the input Hl−1H^{l-1} is projected through the weight matrices WQl,WKl,WVl∈ℝd×dW^{l}\_{Q},W^{l}\_{K},W^{l}\_{V}\in\mathbb{R}^{d\times d} to generate the Query, Key, and Value, which are then divided into KK heads, denoted as Ql,k,Kl,k,Vl,kQ^{l,k},K^{l,k},V^{l,k}, for 1≤k≤K1\leq k\leq K.
The MHSA is computed as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Al,k=Softmax​(Ql,k​Kl,kTdk+M),A^{l,k}=\text{Softmax}\left(\frac{Q^{l,k}K^{l,k^{T}}}{\sqrt{d\_{k}}}+M\right), |  | (3) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | Ol=Concatk=1K​(Al,k​Vl,k)​WOl,\quad O^{l}=\text{Concat}\_{k=1}^{K}\left(A^{l,k}V^{l,k}\right)W^{l}\_{O}, |  | (4) |

where MM represents the attention mask, and dk=d/Kd\_{k}=d/K.
For simplicity, the rotation position encoding (RoPE) (Su et al. [2024](#bib.bib54)) is omitted here.

#### MoE Layer

Next, Hl′H^{l^{\prime}} is passed through a new LN and enters the MoE layer.
First, the router network determines which experts to activate and how to scale their outputs through the weight matrix WGW\_{G}. The routing weights G∈ℝn×EG\in\mathbb{R}^{n\times E} are computed as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | G=softmax​(Hl′​WG).G=\text{softmax}(H^{l^{\prime}}W\_{G}). |  | (5) |

Then, sparse activation of the experts is achieved by selecting the top-kk routing weights for each input token. The output of the activated experts is scaled by the routing weights and aggregated to form the output of the MoE layer:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑i∈Top-k​(Gj)Gj​i⋅FFN​(LNm​o​e​(Hjl′)),∀j=1​…​n,\sum\_{i\in\text{Top-$k$}(G\_{j})}G\_{ji}\cdot\text{FFN}\left(\text{LN}\_{moe}(H^{l^{\prime}}\_{j})\right),\quad\forall j=1\dots n, |  | (6) |

where Top-k​(Gj)\text{Top-$k$}(G\_{j}) denotes the indices of the top-kk routing weights for the jj-th input token. The FFN is defined as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | FFN​(X)=(σ​(X​Wg)⊙X​Wu)​Wd,\text{FFN}\left(X\right)=\left(\sigma\left(XW\_{g}\right)\odot XW\_{u}\right)W\_{d}, |  | (7) |

where WgW\_{g}, WuW\_{u}, and WdW\_{d} are the weight matrices for the gating, up-projection, and down-projection, respectively.
σ\sigma denotes the activation function, and ⊙\odot represents the Hadamard product.

(a) Qwen3-30B-A3B.

(b) DeepSeek-V2-Lite-Chat.

(c) Mixtral-8x7B-v0.1-instruct.

(d) Qwen3-30B-A3B-Base.

(e) DeepSeek-V2-Lite.

(f) Mixtral-8x7B-v0.1.

Figure 5: Heatmap visualizations of the maximum output magnitudes from the down\_proj for each expert across layers.
The analysis is conducted using the C4 (Raffel et al. [2020](#bib.bib52)) dataset.
SEs are highlighted with arrows.

## 3 Super Experts: Discovery and Localization

In this section, we first demonstrate the discovery process of SEs using Qwen’s latest MoE LLM, Qwen3-30B-A3B, as an example.
Next, we analyze SEs across different MoE LLMs and data domains to examine their distribution patterns and highlight the widespread presence of SEs.

### 3.1 Exploring Super Experts

#### Massive Activations in MoE LLMs

The discovery of SEs arises from an exploration and analysis of the formation of massive activations (Sun et al. [2024](#bib.bib56)) in MoE LLMs.
Recent research has explored a distinct class of extreme activation outliers in LLMs, which appear in the hidden states between decoder layers and are known as massive activations (Sun et al. [2024](#bib.bib56); Guo et al. [2024](#bib.bib28); Yu et al. [2024](#bib.bib70)).
They are limited in number, yet their values are orders of magnitude larger than those of other activations (e.g., up to 100,000 times larger).
Existing research has yet to extensively explore massive activations in MoE LLMs and the mechanisms behind their formation.
To address this gap, we first validate the existence of massive activations across several open-source MoE LLMs.
Our findings reveal that massive activations persist in all MoE LLMs we investigated, as shown in the first image on the left in Figure [4](#S2.F4 "Figure 4 ‣ MoE LLMs ‣ 2 Preliminaries on MoE LLMs ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models").

#### Super Experts Induce Massive Activations

Given that massive activations are also present in MoE LLMs, a deeper question arises:
How are these massive activations formulated in MoE LLMs?
Are they the result of all activated experts working together, or are they driven by only a few specific experts, or perhaps caused by other parts of the model?
Through observations of several prominent open-source MoE LLMs (e.g., Qwen series, DeepSeek series, Mistral), we surprisingly find that a small subset of experts consistently generates rare yet extreme activation outliers in the output of their down\_proj layers.
These outliers are subsequently passed onto the hidden states via residual summation after the MoE layers, leading to massive activations.
The entire process is illustrated in Figure [3](#S2.F3 "Figure 3 ‣ MoE LLMs ‣ 2 Preliminaries on MoE LLMs ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models") using the Qwen3-30B-A3B example.
This phenomenon typically occurs in a single layer (e.g., Mixtral) or in just a few layers (e.g., Qwen3-30B-A3B) starting from the initial decoder layers, ultimately leading to stable massive activations across nearly all subsequent layers.
To directly validate this mechanism, we also perform ablation experiments by pruning the SEs in Qwen3-30B-A3B.
As illustrated in Figure [4](#S2.F4 "Figure 4 ‣ MoE LLMs ‣ 2 Preliminaries on MoE LLMs ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models"), pruning SEs from a single layer effectively eliminates their contribution to massive activations.
Furthermore, when all SEs are pruned, massive activations are completely eliminated, confirming that they are directly generated by SEs.

### 3.2 Localization of Super Experts

In this section, we begin by discussing how to accurately identify SEs and then examine their distribution patterns across various models and data domains.

#### Super Experts Profiling

Given that SEs are defined by their influence on the formation of massive activations through the extreme activation outliers they generate, we propose the following broad yet effective quantitative definition.
Specifically, we compute the maximum output magnitudes to the down\_proj for all experts across all layers.
Let LL denote the set of layers responsible for the formation of massive activations.
Let al,ea\_{l,e} denote the maximum output magnitude to the down\_proj of expert ee in layer ll, and let 𝒜={al,e}\mathcal{A}=\{a\_{l,e}\} be the set of all such values across the entire model.
An expert ee in layer ll is classified as a SE if:

|  |  |  |  |
| --- | --- | --- | --- |
|  | al,e>P99.5andal,e>110​amaxandl∈La\_{l,e}>P\_{99.5}\quad\text{and}\quad a\_{l,e}>\frac{1}{10}a\_{\text{max}}\quad\text{and}\quad l\in L |  | (8) |

where P99.5=Percentile99.5​(𝒜)P\_{99.5}=\text{Percentile}\_{99.5}(\mathcal{A}) and amax=max⁡𝒜a\_{\text{max}}=\max\mathcal{A}.
This criterion effectively identifies the experts of interest across various MoE LLMs we investigated, as highlighted in bold in Table [1](#S2.T1 "Table 1 ‣ MoE LLMs ‣ 2 Preliminaries on MoE LLMs ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models").
We have developed an automated tool for rapid and precise SE profiling based on this definition.
The code is available at https://github.com/ZunhaiSu/Super-Experts-Profilling.

| Model | Super Experts |
| --- | --- |
| Qwen3-30B-A3B | |  | | --- | | Layer 1 Expert 68, Layer 2 Expert 92, | | Layer 3 Expert 82 | |
| Qwen3-30B-A3B-Base |
| DeepSeek-V2-Lite-Chat | Layer 2 Expert 54, Layer 3 Expert 38 |
| DeepSeek-V2-Lite |
| Mixtral-8x7B-Instruct-v0.1 | Layer 1 Expert 3 |
| Mixtral-8x7B-v0.1 |

Table 2: Distributions of SEs in Several MoE LLMs.

#### Distribution of SEs Across Different Models

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | Setting | PPL | Avg. | ARC-c | ARC-e | BoolQ | GSM8K | |  | | --- | | Hella | | Swag | | MMLU | |  | | --- | | Open | | BookQA | | PIQA | |  | | --- | | Wino | | Grande | |
| |  | | --- | | Qwen3 | | 30B-A3B | | Baseline | 8.70 | 70.22 | 52.65 | 79.5 | 88.72 | 89.61 | 59.63 | 77.82 | 34.2 | 79.33 | 70.56 |
| Prune SEs | 59.86 | 0.5500 | 46.08 | 76.05 | 70.73 | 42.38 | 39.31 | 56.03 | 29.8 | 72.52 | 62.12 |
| Drop Rate (%) | - | 21.68% | 12.48% | 4.34% | 20.28% | 52.71% | 34.08% | 28.00% | 12.87% | 8.58% | 11.96% |
| Random Pruning | 8.71 | 70.36 | 52.73 | 79.46 | 88.59 | 89.84 | 59.5 | 77.84 | 34 | 79.76 | 71.51 |
| Drop Rate (%) | - | -0.20% | -0.15% | 0.05% | 0.15% | -0.26% | 0.22% | -0.03% | 0.58% | -0.54% | -1.35% |
| |  | | --- | | DeepSeek | | V2-Lite | | Baseline | 6.31 | 60.27 | 46.59 | 78.37 | 79.79 | 37.83 | 58.75 | 55.03 | 34.6 | 80.3 | 71.19 |
| Prune SEs | 10.75 | 43.90 | 29.27 | 54.92 | 68.62 | 9.78 | 43.72 | 41.77 | 21 | 68.28 | 57.7 |
| Drop Rate (%) | - | 27.17% | 37.18% | 29.92% | 14.00% | 74.15% | 25.58% | 24.10% | 39.31% | 14.97% | 18.95% |
| Random Pruning | 6.31 | 60.30 | 46.5 | 78.45 | 80.37 | 37.38 | 58.77 | 55.1 | 34.4 | 80.14 | 71.59 |
| Drop Rate (%) | - | -0.05% | 0.19% | -0.10% | -0.73% | 1.19% | -0.03% | -0.13% | 0.58% | 0.20% | -0.56% |
| |  | | --- | | Mixtral | | 8x7B-v0.1 | | Baseline | 3.84 | 67.84 | 56.57 | 84.26 | 85.02 | 57.32 | 64.89 | 67.83 | 35.6 | 82.48 | 76.56 |
| Prune SEs | 6.23 | 49.38 | 36.01 | 64.44 | 75.66 | 24.34 | 50.6 | 42.47 | 20.6 | 73.12 | 57.22 |
| Drop Rate (%) | - | 27.21% | 36.34% | 23.52% | 11.01% | 57.54% | 22.02% | 37.39% | 42.13% | 11.35% | 25.26% |
| Random Pruning | 3.86 | 67.82 | 56.57 | 84.09 | 85.23 | 58.15 | 64.92 | 68.08 | 35 | 82.21 | 76.16 |
| Drop Rate (%) | - | 0.02% | 0.00% | 0.20% | -0.25% | -1.45% | -0.05% | -0.37% | 1.69% | 0.33% | 0.52% |

Table 3: Evaluation of the importance of SEs in non-reasoning models.
The results of random pruning are obtained by averaging the performance over five runs, each randomly pruning the same number of experts.



|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | Setting | PPL | Avg. | GPQA Diamond | Math-500 | AIME 2024 | AIME 2025 | LiveCodeBench |
| Pass@1 | Pass@1 | Pass@1 | Pass@1 | Pass@1 |
| DeepSeek-R1 | Baseline | 3.33 | 75.64 | 71.50 | 97.60 | 79.33 | 66.33 | 63.44 |
| Prune SEs | 5.18 | 1.81 | 5.05 | 4.00 | 0.00 | 0.00 | 0.00 |
| Drop Rate (%) | - | 97.61% | 93.0% | 95.9% | 100% | 100% | 100% |
| Random Pruning | 3.35 | 75.53 | 72.63 | 98.00 | 77.67 | 67.00 | 62.37 |
| Drop Rate (%) | - | 0.32% | -2.4% | -0.4% | 2.1% | -1.0% | 1.7% |

Table 4: Evaluation of the importance of SEs in DeepSeek-R1.

We select three representative MoE LLMs with distinct designs for analysis: Qwen3-30B-A3B, DeepSeek-V2-Lite-Chat, and Mixtral-8x7B-Instruct-v0.1.
We also include the base models of these three LLMs to illustrate the impact of post-training processes.
Although all of these models are MoE LLMs, they exhibit distinct design differences.
For instance, Qwen3 and Mixtral do not employ shared experts, whereas DeepSeek does.
DeepSeek-V2-Lite adopts a hybrid architecture, wherein the first layer utilizes dense MLPs, while the remaining layers are based on MoE blocks.
Through the proposed SE profiling tool, we identify the SEs in these models.
A summary of the SEs is provided in Table [2](#S3.T2 "Table 2 ‣ Super Experts Profiling ‣ 3.2 Localization of Super Experts ‣ 3 Super Experts: Discovery and Localization ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models"), and heatmap visualizations of the maximum output magnitudes from the down\_proj are shown in Figure [5](#S2.F5 "Figure 5 ‣ MoE Layer ‣ 2 Preliminaries on MoE LLMs ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models").
The key conclusions regarding SEs are summarized as follows:

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | Setting | PPL | Avg. | GPQA Diamond | Math-500 | AIME 2024 | AIME 2025 | HumanEval |
| Pass@1 | Pass@1 | Pass@1 | Pass@1 | Pass@1 |
| Qwen3-30B-A3B | Baseline | 8.70 | 69.37 | 61.62 | 88.00 | 80.00 | 73.33 | 43.90 |
| Prune SEs | 59.86 | 4.02 | 18.69 | 1.40 | 0.00 | 0.00 | 0.00 |
| Drop Rate (%) | - | 93.62% | 69.7% | 98.4% | 100% | 100% | 100% |
| Random Pruning | 8.71 | 69.33 | 61.62 | 89.00 | 80.00 | 73.33 | 42.70 |
| Drop Rate (%) | - | 0.32% | 0.00% | -1.10% | 0.00% | 0.00% | 2.7% |

Table 5: Evaluation of the importance of SEs in Qwen3-30B-A3B.



| DeepSeek-R1 | Problem | Answer |
| --- | --- | --- |
| Original Model | |  | | --- | | How many of the first one hundred positive | | integers are divisible by 3,4,3,4, and 5​?5? | | Please reason step by step, and put your | | final answer within \boxed{}. | | |  | | --- | | Okay, so the problem is asking: How many of the first | | one hundred positive integers are divisible by 3, 4, | | and 5? Hmm, let me think. First, I need to understand | | … … | |
| Prune Super Experts | |  | | --- | | Okay, let’s see here. So the question is how many of | | the first one hundred positive integers are divisible | | by 3,4,3,4, and 55 right? Okay, so first, that’s probably | | going to, like, but, that’s, the way, it’s, the way, it’s, | | the way, it’s, the way, it’s, the way, it’s, the way, it’s, | | … … Repeating … … | | the way, it’s, the way, it’s, the way, it’s, the way, it’s | |
| Random Pruning | |  | | --- | | Okay, so the problem is asking: How many of the first | | one hundred positive integers are divisible by 3, 4, | | and 5? Hmm, let me think. First, I need to understand | | … … | |

Table 6: Responses of DeepSeek-R1 in the MATH-500 (Lightman et al. [2023](#bib.bib41)) benchmark.

(i)
SEs are consistently present across the investigated models, accounting for less than 0.5% of all experts.

(ii)
After post-training processes, the distribution of SEs remains unchanged compared to the base model.

(iii)
Some experts in the final layers also exhibit extreme activation outliers.
However, since they do not contribute to the formation of massive activations, they do not hold the same level of significance as SEs.
Additional experimental results are available in Appendix [A](#A1 "Appendix A Further Analysis of Outlier Experts in Final Layers ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models").

#### Distribution of SEs Across Different Data Domains

We also examine their patterns across various input data domains.
Specifically, in addition to the C4 dataset (Raffel et al. [2020](#bib.bib52)), we analyze SE distributions across several other datasets, including WikiText-2 (Merity et al. [2016](#bib.bib48)), C-Eval (Huang et al. [2023](#bib.bib32)), GSM8K (Cobbe et al. [2021](#bib.bib17)), and HumanEval (Chen et al. [2021](#bib.bib10)).
As shown in the Appendix [B](#A2 "Appendix B Distribution of Super Experts Across Various Data Domains ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models"), the distribution of SEs remains highly stable, regardless of variations in the input data domain.

## 4 The Importance of Super Experts

In this section, we evaluate the significance of SEs by quantifying the performance degradation across various capabilities following their pruning.
We use the original model and the average results from random pruning of an equivalent number of experts as baselines.
To more effectively evaluate the importance of SEs, we utilize distinct benchmark types for non-reasoning and reasoning models.
For non-reasoning models, we conduct a series of few-shot evaluations, whereas for reasoning models, we employ more demanding benchmarks, encompassing tasks such as mathematical reasoning and coding.

### 4.1 Impact on Non-Reasoning Models

For non-reasoning models, we select three models: the non-thinking mode of Qwen3-30B-A3B, DeepSeek-V2-Lite and Mixtral-8x7B-v0.1.
We utilize the datasets listed below and conduct evaluations using lm-eval (Gao et al. [2024](#bib.bib25)), including ARC-challenge (ARC-c), ARC-easy (ARC-e) (Clark et al. [2018](#bib.bib16)), BoolQ (Clark et al. [2019a](#bib.bib14)), GSM8K (Cobbe et al. [2021](#bib.bib17)), HellaSwag (Zellers et al. [2019](#bib.bib71)), MMLU (Hendrycks et al. [2021](#bib.bib29)), OpenBookQA (Mihaylov et al. [2018](#bib.bib50)), PIQA (Bisk et al. [2020](#bib.bib6)), and WinoGrande (Keisuke et al. [2019](#bib.bib36)).
As shown in Table [3](#S3.T3 "Table 3 ‣ Distribution of SEs Across Different Models ‣ 3.2 Localization of Super Experts ‣ 3 Super Experts: Discovery and Localization ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models"), pruning only a few super-experts leads to significant degradation across all tasks, with average accuracy dropping by 21.68% to 27.21%.
In particular, for GSM8K, the degradation ranges from 52.71% to 74.15%.
In contrast, random pruning has a negligible impact, underscoring the crucial role of SEs.

### 4.2 Impact on Reasoning Models

For evaluating the importance of SEs in reasoning models, we select DeepSeek-R1 and the thinking mode of Qwen3-30B-A3B.
We select benchmarks more suitable for testing reasoning models and conduct evaluations based on the EvalScope (Team [2024a](#bib.bib58)), a comprehensive model evaluation and performance benchmarking framework meticulously developed by the ModelScope community (Team [2023](#bib.bib60)).
The generation configurations used in the evaluation align with the corresponding technical reports of the models.
These benchmarks are:

* •

  General Tasks: We use GPQA-Diamond under a 5-shot setting.
  GPQA (Rein et al. [2024](#bib.bib53)) is a challenging dataset of multiple-choice questions authored by domain-specific multidisciplinary experts.
* •

  Math & Text Reasoning:  To evaluate mathematical and logical reasoning skills, we use high-level math benchmarks, including MATH-500 (Lightman et al. [2023](#bib.bib41)), AIME’24 (AIME [2024](#bib.bib1)), and AIME’25 (AIME [2025](#bib.bib2)).
* •

  Agent & Coding:  To test the model’s proficiency in coding and agent-based tasks, we use LiveCodeBench (Jain et al. [2024](#bib.bib33)) and HumanEval (Chen et al. [2021](#bib.bib10)).

The experimental results, presented in Tables [4](#S3.T4 "Table 4 ‣ Distribution of SEs Across Different Models ‣ 3.2 Localization of Super Experts ‣ 3 Super Experts: Discovery and Localization ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models") and [5](#S3.T5 "Table 5 ‣ Distribution of SEs Across Different Models ‣ 3.2 Localization of Super Experts ‣ 3 Super Experts: Discovery and Localization ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models"), show that pruning the SEs causes a significant performance degradation, while random pruning has almost no impact.
The Pass@1 scores for most tasks drop to zero, highlighting the critical role of SEs.
During the review of model responses on the Math-500 benchmark, we made a striking observation: after pruning the SEs, the model consistently generated repetitive responses in nearly every test, continuing until it reached the maximum output length, as shown in Table [6](#S3.T6 "Table 6 ‣ Distribution of SEs Across Different Models ‣ 3.2 Localization of Super Experts ‣ 3 Super Experts: Discovery and Localization ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models").
This behavior suggests that the model loses its ability to reason and solve problems entirely after SE pruning.
More results are in Appendix [C](#A3 "Appendix C Additional Results of Reasoning Models After Super Experts Pruning ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models").

(a) Layer 15 Head 10.

(b) Layer 25 Head 10.

(c) Layer 35 Head 10.

(d) Layer 15 Head 10.

(e) Layer 25 Head 10.

(f) Layer 35 Head 10.

Figure 6: Attention scores of Qwen3-30B-A3B.
Figures (a), (b), and (c) depict the attention score maps of the original model, where the first token clearly functions as an attention sink, consistently attracting the majority of attention.
Figures (d), (e), and (f) illustrate the attention scores following SE pruning, where the attention sink completely disappears.




Figure 7: Attention sink decay rate across layers.

## 5 Understanding the Impact of Super Experts Compression

In this section, we focus on deepening our understanding of the underlying impact of SE compression and further quantitatively assessing its effects.

### 5.1 Super Experts Induce Attention Sinks

Previous studies in dense LLMs (Sun et al. [2024](#bib.bib56); Gu et al. [2024](#bib.bib26); An et al. [2025](#bib.bib3)) suggest that massive activations tend to attract disproportionately high attention scores in multi-head self-attention (MHSA) layers, leading to attention sinks for initial tokens.
StreamLLM (Xiao et al. [2023b](#bib.bib65)) conducted an initial investigation revealing the presence of attention sinks in LLMs.
Subsequent studies (Gu et al. [2024](#bib.bib26); Guo et al. [2024](#bib.bib28)) have demonstrated that although attention sinks often emerge at semantically insignificant tokens, the mechanism itself is critical for model performance.
In efficient LLM techniques such as sparse attention and KV cache compression, maintaining attention sinks is essential for preventing undesirable distributional shifts of attention scores (Xiao et al. [2023b](#bib.bib65); Su et al. [2025](#bib.bib55)).
Building on this insight and our previous findings, we hypothesize that pruning SEs not only eliminates massive activations but also disrupts the underlying attention sink mechanism.

### 5.2 Impact of Super Experts Compression

To validate this insight and quantitatively assess the impact of SE pruning on attention sinks, we introduce (i) Attention Sink Decay Rate, denoted as DsinkD\_{\text{sink}}.
It is defined as the average decay rate of attention sinks for each head:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Dsink=1−1H​∑h=1H∑i∈Spit′∑i∈SpitD\_{\text{sink}}=1-\frac{1}{H}\sum\_{h=1}^{H}\frac{\sum\_{i\in S}{p\_{i}^{t}}^{\prime}}{\sum\_{i\in S}p\_{i}^{t}} |  | (9) |

where HH is the total number of heads, pitp\_{i}^{t} represents the attention score between the Query token tt and the Key token ii before SE pruning, pit′{p\_{i}^{t}}^{\prime} denotes the attention score after SE pruning, and SS refers to the set of sink tokens.

As shown in Figure [7](#S4.F7 "Figure 7 ‣ 4.2 Impact on Reasoning Models ‣ 4 The Importance of Super Experts ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models"), after pruning SEs, the attention sink decay rate remains consistently high, at approximately or even exceeding 90%, demonstrating a substantial disruptive effect on attention sinks.
Figure [6](#S4.F6 "Figure 6 ‣ 4.2 Impact on Reasoning Models ‣ 4 The Importance of Super Experts ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models") visualizes the attention scores for several heads before and after pruning SEs, highlighting the complete disappearance of attention sinks following SE pruning.
Notably, attention sinks introduce implicit attention biases (Sun et al. [2024](#bib.bib56); An et al. [2025](#bib.bib3)) that persist across all subsequent tokens and may encode global or other critical information (Darcet et al. [2023](#bib.bib19)).
Consequently, the impact of SE pruning on attention computation remains both continuous and significant.

## 6 Related Work

### 6.1 Expert-Level Compression

Beyond traditional LLM-focused compression methods, recent advancements have increasingly focused on expert-level compression strategies, capitalizing on the distinct structural features of MoE LLMs and the natural variations in expert importance.
These approaches—including expert pruning, skipping, merging, and expert-level mixed-precision quantization—are being actively explored for their potential to enhance MoE model efficiency.
M-SMoE (Li et al. [2023](#bib.bib39)) performs expert merging by using activation frequencies to consolidate less significant experts, while also applying low-rank techniques to the merged experts to achieve further compression.
NAEE (Lu et al. [2024](#bib.bib46)) introduces plug-and-play pruning and skipping methods that leverage reconstruction loss to selectively compress less critical experts.
MC (Huang et al. [2025](#bib.bib31)) harnesses the significance of both experts and tokens to perform mixed-precision quantization and dynamic expert pruning, achieving extreme compression.
MC-Suite (Jaiswal et al. [2025](#bib.bib34)) reviews various empirical criteria for identifying critical experts, considering four dimensions: weight, expert behavior, intermediate activations, and gradient behavior.
While these methods examine expert importance from various perspectives to optimize expert compression, they lack a deeper exploration and understanding of expert significance, overlooking the mechanistic importance of specific experts.
Our work presents the first discovery and investigation of Super Experts, offering an in-depth analysis of their mechanistic role in inference, thereby filling a critical gap in current research.

## 7 Conclusion

In this work, we present the first identification and systematic study of a distinct subset of experts, termed Super Experts.
We analyze their characteristics, distributions, and critical functional roles within MoE LLMs.
Quantitative evaluations demonstrate that compressing SEs significantly disrupts the attention sink mechanism, leading to a degradation in model performance.
Future research will focus on leveraging SEs to develop more refined compression methods for MoE LLMs.

## References

* AIME (2024)

  AIME. 2024.
  AIME problems and solutions.
  https://aime24.aimedicine.info/.
* AIME (2025)

  AIME. 2025.
  AIME problems and solutions.
  https://artofproblemsolving.com/wiki/index.php/AIMEProblemsandSolutions.
* An et al. (2025)

  An, Y.; Zhao, X.; Yu, T.; Tang, M.; and Wang, J. 2025.
  Systematic outliers in large language models.
  *arXiv preprint arXiv:2502.06415*.
* Ashkboos et al. (2024a)

  Ashkboos, S.; Croci, M. L.; Nascimento, M. G. d.; Hoefler, T.; and Hensman, J. 2024a.
  Slicegpt: Compress large language models by deleting rows and columns.
  *arXiv preprint arXiv:2401.15024*.
* Ashkboos et al. (2024b)

  Ashkboos, S.; Mohtashami, A.; Croci, M. L.; Li, B.; Cameron, P.; Jaggi, M.; Alistarh, D.; Hoefler, T.; and Hensman, J. 2024b.
  Quarot: Outlier-free 4-bit inference in rotated llms.
  *Advances in Neural Information Processing Systems*, 37: 100213–100240.
* Bisk et al. (2020)

  Bisk, Y.; Zellers, R.; Bras, R. L.; Gao, J.; and Choi, Y. 2020.
  PIQA: Reasoning about Physical Commonsense in Natural Language.
  In *Thirty-Fourth AAAI Conference on Artificial Intelligence*.
* Bondarenko, Nagel, and Blankevoort (2021)

  Bondarenko, Y.; Nagel, M.; and Blankevoort, T. 2021.
  Understanding and overcoming the challenges of efficient transformer quantization.
  *arXiv preprint arXiv:2109.12948*.
* Bondarenko, Nagel, and Blankevoort (2023)

  Bondarenko, Y.; Nagel, M.; and Blankevoort, T. 2023.
  Quantizable transformers: Removing outliers by helping attention heads do nothing.
  *Advances in Neural Information Processing Systems*, 36: 75067–75096.
* Cai et al. (2024)

  Cai, W.; Jiang, J.; Wang, F.; Tang, J.; Kim, S.; and Huang, J. 2024.
  A survey on mixture of experts.
  *arXiv preprint arXiv:2407.06204*.
* Chen et al. (2021)

  Chen, M.; Tworek, J.; Jun, H.; Yuan, Q.; de Oliveira Pinto, H. P.; Kaplan, J.; Edwards, H.; Burda, Y.; Joseph, N.; Brockman, G.; and Ray, A. 2021.
  Evaluating Large Language Models Trained on Code.
  *arXiv preprint arXiv:2107.03374*.
* Chen et al. (2022)

  Chen, T.; Huang, S.; Xie, Y.; Jiao, B.; Jiang, D.; Zhou, H.; Li, J.; and Wei, F. 2022.
  Task-specific expert pruning for sparse mixture-of-experts.
  *arXiv preprint arXiv:2206.00277*.
* Chi et al. (2022)

  Chi, Z.; Dong, L.; Huang, S.; Dai, D.; Ma, S.; Patra, B.; Singhal, S.; Bajaj, P.; Song, X.; Mao, X.-L.; et al. 2022.
  On the representation collapse of sparse mixture of experts.
  *Advances in Neural Information Processing Systems*, 35: 34600–34613.
* Chowdhury et al. (2024)

  Chowdhury, M. N. R.; Wang, M.; Maghraoui, K. E.; Wang, N.; Chen, P.-Y.; and Carothers, C. 2024.
  A provably effective method for pruning experts in fine-tuned sparse mixture-of-experts.
  *arXiv preprint arXiv:2405.16646*.
* Clark et al. (2019a)

  Clark, C.; Lee, K.; Chang, M.-W.; Kwiatkowski, T.; Collins, M.; and Toutanova, K. 2019a.
  Boolq: Exploring the surprising difficulty of natural yes/no questions.
  *arXiv preprint arXiv:1905.10044*.
* Clark et al. (2019b)

  Clark, K.; Khandelwal, U.; Levy, O.; and Manning, C. D. 2019b.
  What does bert look at? an analysis of bert’s attention.
  *arXiv preprint arXiv:1906.04341*.
* Clark et al. (2018)

  Clark, P.; Cowhey, I.; Etzioni, O.; Khot, T.; Sabharwal, A.; Schoenick, C.; and Tafjord, O. 2018.
  Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge.
  *arXiv:1803.05457v1*.
* Cobbe et al. (2021)

  Cobbe, K.; Kosaraju, V.; Bavarian, M.; Chen, M.; Jun, H.; Kaiser, L.; Plappert, M.; Tworek, J.; Hilton, J.; Nakano, R.; Hesse, C.; and Schulman, J. 2021.
  Training Verifiers to Solve Math Word Problems.
  *arXiv preprint arXiv:2110.14168*.
* Dai et al. (2024)

  Dai, D.; Deng, C.; Zhao, C.; Xu, R.; Gao, H.; Chen, D.; Li, J.; Zeng, W.; Yu, X.; Wu, Y.; et al. 2024.
  Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models.
  *arXiv preprint arXiv:2401.06066*.
* Darcet et al. (2023)

  Darcet, T.; Oquab, M.; Mairal, J.; and Bojanowski, P. 2023.
  Vision transformers need registers.
  *arXiv preprint arXiv:2309.16588*.
* Devlin et al. (2019)

  Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2019.
  Bert: Pre-training of deep bidirectional transformers for language understanding.
  In *Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers)*, 4171–4186.
* Dosovitskiy et al. (2020)

  Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; Dehghani, M.; Minderer, M.; Heigold, G.; Gelly, S.; et al. 2020.
  An image is worth 16x16 words: Transformers for image recognition at scale.
  *arXiv preprint arXiv:2010.11929*.
* Duanmu et al. (2025)

  Duanmu, H.; Li, X.; Yuan, Z.; Zheng, S.; Duan, J.; Zhang, X.; and Lin, D. 2025.
  MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design.
  *arXiv preprint arXiv:2505.05799*.
* Frantar and Alistarh (2023)

  Frantar, E.; and Alistarh, D. 2023.
  Sparsegpt: Massive language models can be accurately pruned in one-shot.
  In *International Conference on Machine Learning*, 10323–10337. PMLR.
* Frantar et al. (2022)

  Frantar, E.; Ashkboos, S.; Hoefler, T.; and Alistarh, D. 2022.
  Gptq: Accurate post-training quantization for generative pre-trained transformers.
  *arXiv preprint arXiv:2210.17323*.
* Gao et al. (2024)

  Gao, L.; Tow, J.; Abbasi, B.; Biderman, S.; Black, S.; DiPofi, A.; Foster, C.; Golding, L.; Hsu, J.; Le Noac’h, A.; Li, H.; McDonell, K.; Muennighoff, N.; Ociepa, C.; Phang, J.; Reynolds, L.; Schoelkopf, H.; Skowron, A.; Sutawika, L.; Tang, E.; Thite, A.; Wang, B.; Wang, K.; and Zou, A. 2024.
  The Language Model Evaluation Harness.
* Gu et al. (2024)

  Gu, X.; Pang, T.; Du, C.; Liu, Q.; Zhang, F.; Du, C.; Wang, Y.; and Lin, M. 2024.
  When attention sink emerges in language models: An empirical view.
  *arXiv preprint arXiv:2410.10781*.
* Guo et al. (2025)

  Guo, D.; Yang, D.; Zhang, H.; Song, J.; Zhang, R.; Xu, R.; Zhu, Q.; Ma, S.; Wang, P.; Bi, X.; et al. 2025.
  Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.
  *arXiv preprint arXiv:2501.12948*.
* Guo et al. (2024)

  Guo, T.; Pai, D.; Bai, Y.; Jiao, J.; Jordan, M. I.; and Mei, S. 2024.
  Active-dormant attention heads: Mechanistically demystifying extreme-token phenomena in llms.
  *arXiv preprint arXiv:2410.13835*.
* Hendrycks et al. (2021)

  Hendrycks, D.; Burns, C.; Basart, S.; Zou, A.; Mazeika, M.; Song, D.; and Steinhardt, J. 2021.
  Measuring Massive Multitask Language Understanding.
  *Proceedings of the International Conference on Learning Representations (ICLR)*.
* Hu et al. (2025)

  Hu, X.; Chen, Z.; Yang, D.; Xu, Z.; Xu, C.; Yuan, Z.; Zhou, S.; and Yu, J. 2025.
  MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance.
  *arXiv preprint arXiv:2505.03804*.
* Huang et al. (2025)

  Huang, W.; Liao, Y.; Liu, J.; He, R.; Tan, H.; Zhang, S.; Li, H.; Liu, S.; and Qi, X. 2025.
  Mixture Compressor for Mixture-of-Experts LLMs Gains More.
  In *The Thirteenth International Conference on Learning Representations*.
* Huang et al. (2023)

  Huang, Y.; Bai, Y.; Zhu, Z.; Zhang, J.; Zhang, J.; Su, T.; Liu, J.; Lv, C.; Zhang, Y.; Lei, J.; Fu, Y.; Sun, M.; and He, J. 2023.
  C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models.
  *arXiv preprint arXiv:2305.08322*.
* Jain et al. (2024)

  Jain, N.; Han, K.; Gu, A.; Li, W.-D.; Yan, F.; Zhang, T.; Wang, S.; Solar-Lezama, A.; Sen, K.; and Stoica, I. 2024.
  Livecodebench: Holistic and contamination free evaluation of large language models for code.
  *arXiv preprint arXiv:2403.07974*.
* Jaiswal et al. (2025)

  Jaiswal, A.; Wang, J.; Li, Y.; Li, P.; Chen, T.; Wang, Z.; Wang, C.; Pang, R.; and Du, X. 2025.
  Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping Strategies and Observations.
  *arXiv preprint arXiv:2504.05586*.
* Jiang et al. (2024)

  Jiang, A. Q.; Sablayrolles, A.; Roux, A.; Mensch, A.; Savary, B.; Bamford, C.; Chaplot, D. S.; Casas, D. d. l.; Hanna, E. B.; Bressand, F.; et al. 2024.
  Mixtral of experts.
  *arXiv preprint arXiv:2401.04088*.
* Keisuke et al. (2019)

  Keisuke, S.; Ronan, L. B.; Chandra, B.; and Yejin, C. 2019.
  WinoGrande: An Adversarial Winograd Schema Challenge at Scale.
  *Communications of the ACM*.
* Kovaleva et al. (2019)

  Kovaleva, O.; Romanov, A.; Rogers, A.; and Rumshisky, A. 2019.
  Revealing the dark secrets of BERT.
  *arXiv preprint arXiv:1908.08593*.
* Li et al. (2024)

  Li, P.; Jin, X.; Cheng, Y.; and Chen, T. 2024.
  Examining post-training quantization for mixture-of-experts: A benchmark.
  *arXiv preprint arXiv:2406.08155*.
* Li et al. (2023)

  Li, P.; Zhang, Z.; Yadav, P.; Sung, Y.-L.; Cheng, Y.; Bansal, M.; and Chen, T. 2023.
  Merge, then compress: Demystify efficient smoe with hints from its routing policy.
  *arXiv preprint arXiv:2310.01334*.
* Liang et al. (2024)

  Liang, Z.; Xu, Y.; Hong, Y.; Shang, P.; Wang, Q.; Fu, Q.; and Liu, K. 2024.
  A Survey of Multimodel Large Language Models.
  In *Proceedings of the 3rd International Conference on Computer, Artificial Intelligence and Control Engineering*, 405–409.
* Lightman et al. (2023)

  Lightman, H.; Kosaraju, V.; Burda, Y.; Edwards, H.; Baker, B.; Lee, T.; Leike, J.; Schulman, J.; Sutskever, I.; and Cobbe, K. 2023.
  Let’s verify step by step.
  In *The Twelfth International Conference on Learning Representations*.
* Lin et al. (2024)

  Lin, J.; Tang, J.; Tang, H.; Yang, S.; Chen, W.-M.; Wang, W.-C.; Xiao, G.; Dang, X.; Gan, C.; and Han, S. 2024.
  Awq: Activation-aware weight quantization for on-device llm compression and acceleration.
  *Proceedings of Machine Learning and Systems*, 6: 87–100.
* Liu et al. (2024a)

  Liu, A.; Feng, B.; Wang, B.; Wang, B.; Liu, B.; Zhao, C.; Dengr, C.; Ruan, C.; Dai, D.; Guo, D.; et al. 2024a.
  Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model.
  *arXiv preprint arXiv:2405.04434*.
* Liu et al. (2024b)

  Liu, A.; Feng, B.; Xue, B.; Wang, B.; Wu, B.; Lu, C.; Zhao, C.; Deng, C.; Zhang, C.; Ruan, C.; et al. 2024b.
  Deepseek-v3 technical report.
  *arXiv preprint arXiv:2412.19437*.
* Liu et al. (2024c)

  Liu, E.; Zhu, J.; Lin, Z.; Ning, X.; Blaschko, M. B.; Yan, S.; Dai, G.; Yang, H.; and Wang, Y. 2024c.
  Efficient expert pruning for sparse mixture-of-experts language models: Enhancing performance and reducing inference costs.
  *arXiv preprint arXiv:2407.00945*.
* Lu et al. (2024)

  Lu, X.; Liu, Q.; Xu, Y.; Zhou, A.; Huang, S.; Zhang, B.; Yan, J.; and Li, H. 2024.
  Not all experts are equal: Efficient expert pruning and skipping for mixture-of-experts large language models.
  *arXiv preprint arXiv:2402.14800*.
* Ma, Fang, and Wang (2023)

  Ma, X.; Fang, G.; and Wang, X. 2023.
  Llm-pruner: On the structural pruning of large language models.
  *Advances in neural information processing systems*, 36: 21702–21720.
* Merity et al. (2016)

  Merity, S.; Xiong, C.; Bradbury, J.; and Socher, R. 2016.
  Pointer Sentinel Mixture Models.
  arXiv:1609.07843.
* Meta (2025)

  Meta. 2025.
  LLaMA 4: Multimodal Intelligence.
* Mihaylov et al. (2018)

  Mihaylov, T.; Clark, P.; Khot, T.; and Sabharwal, A. 2018.
  Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering.
  In *EMNLP*.
* Mu and Lin (2025)

  Mu, S.; and Lin, S. 2025.
  A comprehensive survey of mixture-of-experts: Algorithms, theory, and applications.
  *arXiv preprint arXiv:2503.07137*.
* Raffel et al. (2020)

  Raffel, C.; Shazeer, N.; Roberts, A.; Lee, K.; Narang, S.; Matena, M.; Zhou, Y.; Li, W.; and Liu, P. J. 2020.
  Exploring the limits of transfer learning with a unified text-to-text transformer.
  *Journal of machine learning research*, 21(140): 1–67.
* Rein et al. (2024)

  Rein, D.; Hou, B. L.; Stickland, A. C.; Petty, J.; Pang, R. Y.; Dirani, J.; Michael, J.; and Bowman, S. R. 2024.
  Gpqa: A graduate-level google-proof q&a benchmark.
  In *First Conference on Language Modeling*.
* Su et al. (2024)

  Su, J.; Ahmed, M.; Lu, Y.; Pan, S.; Bo, W.; and Liu, Y. 2024.
  Roformer: Enhanced transformer with rotary position embedding.
  *Neurocomputing*, 568: 127063.
* Su et al. (2025)

  Su, Z.; Chen, Z.; Shen, W.; Wei, H.; Li, L.; Yu, H.; and Yuan, K. 2025.
  RotateKV: Accurate and Robust 2-Bit KV Cache Quantization for LLMs via Outlier-Aware Adaptive Rotations.
  *arXiv preprint arXiv:2501.16383*.
* Sun et al. (2024)

  Sun, M.; Chen, X.; Kolter, J. Z.; and Liu, Z. 2024.
  Massive activations in large language models.
  *arXiv preprint arXiv:2402.17762*.
* Sun et al. (2023)

  Sun, M.; Liu, Z.; Bair, A.; and Kolter, J. Z. 2023.
  A simple and effective pruning approach for large language models.
  *arXiv preprint arXiv:2306.11695*.
* Team (2024a)

  Team, M. 2024a.
  EvalScope: Evaluation Framework for Large Models.
* Team (2024b)

  Team, Q. 2024b.
  Qwen2 technical report.
  *arXiv preprint arXiv:2407.10671*.
* Team (2023)

  Team, T. M. 2023.
  AIME problems and solutions, 2025.
  https://github.com/modelscope/modelscope.
* Touvron et al. (2023)

  Touvron, H.; Martin, L.; Stone, K.; Albert, P.; Almahairi, A.; Babaei, Y.; Bashlykov, N.; Batra, S.; Bhargava, P.; Bhosale, S.; et al. 2023.
  Llama 2: Open foundation and fine-tuned chat models.
  *arXiv preprint arXiv:2307.09288*.
* Vaswani et al. (2017)

  Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017.
  Attention is all you need.
  *Advances in neural information processing systems*, 30.
* Wang et al. (2024)

  Wang, W.; Chen, W.; Luo, Y.; Long, Y.; Lin, Z.; Zhang, L.; Lin, B.; Cai, D.; and He, X. 2024.
  Model compression and efficient inference for large language models: A survey.
  *arXiv preprint arXiv:2402.09748*.
* Xiao et al. (2023a)

  Xiao, G.; Lin, J.; Seznec, M.; Wu, H.; Demouth, J.; and Han, S. 2023a.
  Smoothquant: Accurate and efficient post-training quantization for large language models.
  In *International Conference on Machine Learning*, 38087–38099. PMLR.
* Xiao et al. (2023b)

  Xiao, G.; Tian, Y.; Chen, B.; Han, S.; and Lewis, M. 2023b.
  Efficient streaming language models with attention sinks.
  *arXiv preprint arXiv:2309.17453*.
* Xie et al. (2024)

  Xie, Y.; Zhang, Z.; Zhou, D.; Xie, C.; Song, Z.; Liu, X.; Wang, Y.; Lin, X.; and Xu, A. 2024.
  MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router.
  *arXiv preprint arXiv:2410.12013*.
* Yang et al. (2025)

  Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025.
  Qwen3 technical report.
  *arXiv preprint arXiv:2505.09388*.
* Yang et al. (2024)

  Yang, C.; Sui, Y.; Xiao, J.; Huang, L.; Gong, Y.; Duan, Y.; Jia, W.; Yin, M.; Cheng, Y.; and Yuan, B. 2024.
  MoE-I2: Compressing Mixture of Experts Models through Inter-Expert Pruning and Intra-Expert Low-Rank Decomposition.
  *arXiv preprint arXiv:2411.01016*.
* Yang, Kim, and Kim (2024)

  Yang, J.; Kim, H.; and Kim, Y. 2024.
  Mitigating quantization errors due to activation spikes in glu-based llms.
  *arXiv preprint arXiv:2405.14428*.
* Yu et al. (2024)

  Yu, M.; Wang, D.; Shan, Q.; and Wan, A. 2024.
  The super weight in large language models.
  *arXiv preprint arXiv:2411.07191*.
* Zellers et al. (2019)

  Zellers, R.; Holtzman, A.; Bisk, Y.; Farhadi, A.; and Choi, Y. 2019.
  HellaSwag: Can a Machine Really Finish Your Sentence?
  In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*.
* Zhang et al. (2024a)

  Zhang, J.; Huang, J.; Jin, S.; and Lu, S. 2024a.
  Vision-language models for vision tasks: A survey.
  *IEEE Transactions on Pattern Analysis and Machine Intelligence*.
* Zhang et al. (2024b)

  Zhang, Z.; Liu, X.; Cheng, H.; Xu, C.; and Gao, J. 2024b.
  Diversifying the expert knowledge for task-agnostic pruning in sparse mixture-of-experts.
  *arXiv preprint arXiv:2407.09590*.
* Zhao et al. (2023)

  Zhao, W. X.; Zhou, K.; Li, J.; Tang, T.; Wang, X.; Hou, Y.; Min, Y.; Zhang, B.; Zhang, J.; Dong, Z.; et al. 2023.
  A survey of large language models.
  *arXiv preprint arXiv:2303.18223*, 1(2).
* Zheng et al. (2025)

  Zheng, Z.; Cui, X.; Zheng, S.; Li, M.; Chen, J.; Chen, X.; et al. 2025.
  MoQa: Rethinking MoE Quantization with Multi-stage Data-model Distribution Awareness.
  *arXiv preprint arXiv:2503.21135*.
* Zhu et al. (2024)

  Zhu, X.; Li, J.; Liu, Y.; Ma, C.; and Wang, W. 2024.
  A survey on model compression for large language models.
  *Transactions of the Association for Computational Linguistics*, 12: 1556–1577.

| Model | Prune Experts | PPL | Super Experts |
| --- | --- | --- | --- |
| Qwen3-30B-A3B | Original Model | 8.70 | - |
| Layer 1 Expert 68, Layer 2 Expert 92, Layer 3 Expert 82 | 59.86 | Yes |
| Layer 47 Expert 8, Layer 47 Expert 48, Layer 47 Expert 100 | 8.71 | No |
| DeepSeek-V2-Lite | Original Model | 6.31 | - |
| Layer 2 Expert 54, Layer 3 Expert 38 | 10.75 | Yes |
| Layer 25 Expert 11, Layer 25 Expert 39 | 6.32 | No |

Table 7: Comparison of expert pruning, with PPL evaluated using the WikiText-2 dataset.



| Model | Super Experts | Outlier Experts |
| --- | --- | --- |
| Qwen3-30B-A3B | |  | | --- | | Layer 1 Expert 68, Layer 2 Expert 92 | | Layer 3 Expert 82 | | |  | | --- | | Layer 1 Expert 8, Layer 47 Expert 48 | | Layer 47 Expert 100 | |
| DeepSeek-R1 | |  | | --- | | Layer 8 Expert 24, Layer 8 Shared\_expert | | Layer 12 Expert 190, Layer 13 Expert 64 | | Layer 14 Expert 202, Layer 14 Shared\_expert | | Layer 22 Shared\_expert, Layer 33 Expert 64 | | Layer 33 Shared\_expert, Layer 35 Shared\_expert | | |  | | --- | | Layer 60 Expert 81, Layer 60 Expert 92 | | Layer 60 Expert 231, Layer 60 Shared\_expert | | Layer 60 Expert 121, Layer 60 Expert 0 | | Layer 60 Expert 60, Layer 60 Expert 237 | | Layer 60 Expert 53, Layer 60 Expert 117 | |

Table 8: Super Experts and Outlier Experts in Qwen3-30B-A3B and DeepSeek-R1 models.

## Appendix A Further Analysis of Outlier Experts in Final Layers

Some experts in the final layers also exhibit extreme activation outliers in the output of the down\_proj, apart from the Super Experts (SEs) in the shallower layers.
We refer to these experts as outlier experts.
Based on our extensive additional experiments and findings, we do not consider outlier experts to have the same mechanistic significance as SEs.

(i) We performed PPL evaluations after pruning outlier experts, and as shown in Table [7](#A0.T7 "Table 7 ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models"), they do not significantly affect the model’s performance in the same way as the SEs.

(ii) In both Qwen3-30B-A3B and DeepSeek-R1, pruning outlier experts does not result in repetitive outputs on reasoning benchmarks such as Math-500 (Lightman et al. [2023](#bib.bib41)), whereas pruning SEs does, as shown in Tables [9](#A3.T9 "Table 9 ‣ Appendix C Additional Results of Reasoning Models After Super Experts Pruning ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models") and [10](#A3.T10 "Table 10 ‣ Appendix C Additional Results of Reasoning Models After Super Experts Pruning ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models").
The pruned SEs and outlier experts is shown in Table [8](#A0.T8 "Table 8 ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models").

(iii) We observed on Qwen3-30B-A3B that the distribution of outlier experts varies with the input dataset, while the distribution of SEs remains quite stable, as illustrated in Figure [8](#A3.F8 "Figure 8 ‣ Appendix C Additional Results of Reasoning Models After Super Experts Pruning ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models") and [9](#A3.F9 "Figure 9 ‣ Appendix C Additional Results of Reasoning Models After Super Experts Pruning ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models").

Based on our previous analysis, we infer that since massive activations occur in the shallower layers, these outlier experts are not involved in the formation of massive activations.
Therefore, these experts do not operate under the same mechanism as SEs and do not hold the same level of significance.
Further investigation into outlier experts is worth exploring in future work.

## Appendix B Distribution of Super Experts Across Various Data Domains

In addition to analyzing the distribution of SEs across different models based on the C4 dataset (Raffel et al. [2020](#bib.bib52)), we also examine their distribution patterns across various input data domains.
We assess the impact of diverse language inputs on SEs using the WikiText-2 (Merity et al. [2016](#bib.bib48)) and C-Eval (Huang et al. [2023](#bib.bib32)) datasets.
Furthermore, we investigate the influence of data from the mathematics and coding domains using the GSM8K (Hendrycks et al. [2021](#bib.bib29)) and HumanEval (Chen et al. [2021](#bib.bib10)) datasets.
As shown in Figures [8](#A3.F8 "Figure 8 ‣ Appendix C Additional Results of Reasoning Models After Super Experts Pruning ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models"), [9](#A3.F9 "Figure 9 ‣ Appendix C Additional Results of Reasoning Models After Super Experts Pruning ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models"), [10](#A3.F10 "Figure 10 ‣ Appendix C Additional Results of Reasoning Models After Super Experts Pruning ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models"), [11](#A3.F11 "Figure 11 ‣ Appendix C Additional Results of Reasoning Models After Super Experts Pruning ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models"), [12](#A3.F12 "Figure 12 ‣ Appendix C Additional Results of Reasoning Models After Super Experts Pruning ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models"), and [13](#A3.F13 "Figure 13 ‣ Appendix C Additional Results of Reasoning Models After Super Experts Pruning ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models"), the distribution of SEs remains highly stable, regardless of variations in the input data domain.

## Appendix C Additional Results of Reasoning Models After Super Experts Pruning

After pruning SEs, we consistently observed repetitive output and a loss of reasoning ability in both Qwen3-30B-A3B and DeepSeek-R1.
The pruned SEs are shown in Table [8](#A0.T8 "Table 8 ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models"), and additional examples from the Math-500 (Lightman et al. [2023](#bib.bib41)) benchmark are presented in Tables [9](#A3.T9 "Table 9 ‣ Appendix C Additional Results of Reasoning Models After Super Experts Pruning ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models") and [10](#A3.T10 "Table 10 ‣ Appendix C Additional Results of Reasoning Models After Super Experts Pruning ‣ Unveiling Super Experts in Mixture-of-Experts Large Language Models").

|  |  |  |  |
| --- | --- | --- | --- |
| DeepSeek-R1 | Problem | Repeating | Answer |
| Original Model | |  | | --- | | What is the domain of the function | | f​(x)=2−xlog⁡(2−log⁡(x−2))f(x)=\frac{2-x}{\log(2-\log(x-2))}, | | where log\log is the base 1010 logarithm function? | | Express your answer in interval notation. | | Please reason step by step, | | and put your final answer within \boxed{}. | | no | |  | | --- | | Okay, so I need to find the domain of the function | | f(x) = (2 - x) / log(2 - log(x - 2)). Hmm, let’s see. The | | domain of a function includes all the real numbers x | | … … | |
| Prune Super Experts | yes | |  | | --- | | Alright, let’s try to first. the function f(x) which given | | that we need to the domain of this function that’s the | | which the of a that’s the given this. that’s the the the | | the the the the the the the the the the the the the the. | | that’s the. that’s the. that’s the. that’s the. that’s the. | | … … Repeating … … | | that’s the. that’s the. that’s the. that’s the. that’s the. | |
| |  | | --- | | Prune Outlier Experts | | of Last Layer | | no | |  | | --- | | Okay, let’s try to figure out the domain of this function | | f​(x)=2−xlog⁡(2−log⁡(x−2))f(x)=\frac{2-x}{\log(2-\log(x-2))}. Hmm, | | domains can be tricky because there are multiple layers | | … … | |
| Random Pruning | no | |  | | --- | | Alright, so I need to find the domain of the function | | f(x) = (2 - x) / log(2 - log(x - 2)). Hmm, okay. Let me | | start by recalling that the domain of a | | … … | |

Table 9: Responses of DeepSeek-R1 in the MATH-500 benchmark.



|  |  |  |  |
| --- | --- | --- | --- |
| Qwen3-30B-A3B | Problem | Repeating | Answer |
| Original Model | |  | | --- | | Let p​(x)p(x) be a polynomial of degree 5 such that | | p​(n)=nn2−1p(n)=\frac{n}{n^{2}-1} | | for n=2,n=2, 3, 4, …,\dots, 7. Find p​(8).p(8). | | Please reason step by step, and put your final | | answer within \boxed{}. | | no | |  | | --- | | Okay, so I need to find the value of p(8) where p(x) | | is a degree 5 polynomial that satisfies | | p(n) = n / (n² - 1) for n = 2, 3, 4, …, 7. | | … … | |
| Prune Super Experts | yes | |  | | --- | | Okay, so I need to find p(8) where p is a polynomial. | | The problem says that for each n, p(n) is equal to | | some fraction, and we need to find p(8). The question | | is from a competition, so it’s probably from an | | Olympiad, and the answer is an integer. But I need to | | reason step by step, and put my final answer within | | \boxed{}. Alright, let’s start by understanding the | | problem. It says that for each n, p(n) is equal to some | | fraction. But the problem is to find p(8). The question | | is from a competition, so it’s probably from an | | Olympiad, and the answer is an integer. But I need to | | reason step by step, and put my final answer within | | … … Repeating … … | |
| |  | | --- | | Prune Outlier Experts | | of Last Layer | | no | |  | | --- | | Okay, so I need to find the value of p(8) where p(x) is | | a degree 5 polynomial that satisfies p(n) = n / (n² - 1) | | for n = 2, 3, 4, 5, 6, 7. Hmm, let me think about how | | … … | |
| Random Pruning | no | |  | | --- | | Okay, so I need to find the value of p(8) where p(x) | | is a degree 5 polynomial that satisfies | | p(n) = n / (n² - 1) for n = 2, 3, 4, …, 7. | | … … | |

Table 10: Responses of Qwen3-30B-A3B in the MATH-500 benchmark after SE pruning.



(a) Qwen3-30B-A3B (WikiText-2).

(b) Qwen3-30B-A3B (C-Eval).

(c) Qwen3-30B-A3B (GSM8K).

(d) Qwen3-30B-A3B (HumanEval).

Figure 8: Heatmap visualizations of the maximum output magnitudes from the down\_proj for each expert in Qwen3-30B-A3B across multiple datasets. SEs are highlighted with arrows.



(a) Qwen3-30B-A3B-Base (WikiText-2).

(b) Qwen3-30B-A3B-Base (C-Eval).

(c) Qwen3-30B-A3B-Base (GSM8K).

(d) Qwen3-30B-A3B-Base (HumanEval).

Figure 9: Heatmap visualizations of the maximum output magnitudes from the down\_proj for each expert in Qwen3-30B-A3B-Base across multiple datasets.
SEs are highlighted with arrows.



(a) DeepSeek-V2-Lite-Chat (WikiText-2).

(b) DeepSeek-V2-Lite-Chat (C-Eval).

(c) DeepSeek-V2-Lite-Chat (GSM8K).

(d) DeepSeek-V2-Lite-Chat (HumanEval).

Figure 10: Heatmap visualizations of the maximum output magnitudes from the down\_proj for each expert in DeepSeek-V2-Lite-Chat across multiple datasets.
SEs are highlighted with arrows.



(a) DeepSeek-V2-Lite (WikiText-2).

(b) DeepSeek-V2-Lite (C-Eval).

(c) DeepSeek-V2-Lite (GSM8K).

(d) DeepSeek-V2-Lite (HumanEval).

Figure 11: Heatmap visualizations of the maximum output magnitudes from the down\_proj for each expert in DeepSeek-V2-Lite across multiple datasets.
SEs are highlighted with arrows.



(a) Mixtral-8x7B-Instruct-v0.1 (WikiText-2).

(b) Mixtral-8x7B-Instruct-v0.1 (C-Eval).

(c) Mixtral-8x7B-Instruct-v0.1 (GSM8K).

(d) Mixtral-8x7B-Instruct-v0.1 (HumanEval).

Figure 12: Heatmap visualizations of the maximum output magnitudes from the down\_proj for each expert in Mixtral-8x7B-Instruct-v0.1 across multiple datasets.
SEs are highlighted with arrows.



(a) Mixtral-8x7B-v0.1 (WikiText-2).

(b) Mixtral-8x7B-v0.1 (C-Eval).

(c) Mixtral-8x7B-v0.1 (GSM8K).

(d) Mixtral-8x7B-v0.1 (HumanEval).

Figure 13: Heatmap visualizations of the maximum output magnitudes from the down\_proj for each expert in Mixtral-8x7B-v0.1 across multiple datasets.
SEs are highlighted with arrows.
