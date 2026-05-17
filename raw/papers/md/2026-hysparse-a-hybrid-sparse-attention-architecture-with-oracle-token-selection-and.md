---
arxiv: '2602.03560'
authors:
- Yizhao Gao
- Jianyu Wei
- Qihao Zhang
- Yu Cheng
- Shimao Chen
- Zhengju Tang
- Zihan Jiang
- Yifan Song
- Hailin Zhang
- Liang Zhao
- Bo Yang
- Gang Wang
- Shijie Cao
- Fuli Luo
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'HySparse: A Hybrid Sparse Attention Architecture with Oracle Token Selection
  and KV Cache Sharing'
url: https://arxiv.org/abs/2602.03560
year: 2026
---

[2602.03560] 1 Introduction

else if(!window.matchMedia) { return false; }
else if(window.matchMedia("(prefers-color-scheme: dark)").matches) {
theme = "dark"; }
if (theme=="dark") {
document.documentElement.setAttribute("data-theme", "dark");
} else {
document.documentElement.setAttribute("data-theme", "light"); } }

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

\titlefont

HySparse: A Hybrid Sparse Attention Architecture

with Oracle Token Selection and KV Cache Sharing

Yizhao Gao Jianyu Wei
Qihao Zhang
Yu Cheng
Shimao Chen
  
Zhengju Tang
Zihan Jiang
Yifan Song
Hailin Zhang
Liang Zhao
  
Bo Yang
Gang Wang
Shijie Cao
Fuli Luo⋄
  
LLM-Core Xiaomi

\abscontent

00footnotetext: ⋄Corresponding author.

## 1 Introduction

The demand for long-context capabilities has become a cornerstone of modern Large Language Models (LLMs), driven by emerging paradigms such as test-time scaling [o1, r1] and agentic workflows [k2, anthropic\_building\_effective\_agents\_2024].
Yet, the self-attention mechanism in standard Transformers scales quadratically with sequence length, causing computational latency and cost to grow prohibitively as context length increases.

Sparse attention offers a straightforward and effective solution to mitigate this quadratic bottleneck [sparsetransformer].
Sparse attention computes attention over a selected subset of important tokens rather than all tokens in the sequence.
Existing methods can be broadly categorized into training-free and trainable approaches. Training-free methods rely on fixed patterns or heuristic to select important tokens [h2o, streamingllm, minference, quest, lserve, duo]. Trainable sparse attention learns which tokens to attend during training, either via low-cost self-distillation [seerattn\_v1, dsa] or by being directly integrated into pre-training [nsa, moba]. Nevertheless, sparse attention architectures still suffer from two fundamental limitations:

(1) Proxy-based Sparse Token Selection.
Sparse attention fundamentally depends on a selection mechanism to identify important tokens prior to attention computation.
Existing methods typically rely on lightweight proxies, such as predefined patterns, heuristics, approximate estimates, or additional selection modules [minference, seerattn\_v1, moba, seer-r, nsa, dsa].
However, these proxies are inherently approximate and may fail to capture true token importance, particularly in long and evolving contexts.
As a result, sparse token selection is often bounded by the fidelity of the proxy, potentially limiting the expressive power of sparse attention.
While learnable sparse attention alleviates selection errors by learning token selection during training, it does not fundamentally eliminate the proxy-based bottleneck and introduces additional selection modules that increase training complexity.

(2) Computation Reduction without Memory Relief.
Modern sparse attention methods increasingly adopt dynamic sparsity to preserve model fidelity. Unlike static patterns (e.g., fixed strides or block structures), which can reduce KV cache storage but often incur noticeable performance degradation, dynamic approaches typically retain the full KV cache. This is because complete KV cache eviction is irreversible and destructive, as token importance may shift as generation progresses and context evolves. While dynamic sparse attention can effectively reduce computation, it provides no relief for memory consumption. Maintaining a full-sized KV cache therefore remains a dominant bottleneck for serving throughput and maximum batch size, limiting the practical benefits of sparse attention in long-context settings.

To address these challenges, we introduce Hybrid Sparse Attention (HySparse).
The key idea is to interleave every full attention layer with multiple sparse attention layers, where the sparse layers strategically derive important token selection and KV caches from the preceding full layer.
This design is motivated by two empirical observations in recent literature: token saliency is stable across consecutive layers (§[2.3](#S2.SS3 "2.3 Cross-Layer Salient Token Stability ‣ 2 Background & Motivation")), and cross-layer KV cache sharing reduces memory without hurting performance (§[2.4](#S2.SS4 "2.4 Cross-layer KV Cache Sharing ‣ 2 Background & Motivation")).
In HySparse, full attention can precisely identify important token selection and already produces KV caches, which sparse layers can directly reuse.
By reusing the important token indices from full attention, sparse selection becomes oracle-guided.
This eliminates the need for auxiliary proxy modules and ensures stable end-to-end training.
By reusing the KV caches from full attention, sparse attention adds no per-layer KV overhead, effectively alleviating the memory pressure associated with dynamic sparse attention.
Meanwhile, inspired by hybrid sliding window attention (SWA) architectures [gemma3, gpt-oss, xiao2026mimo], HySparse augments each sparse attention layer with an additional SWA branch that maintains a small, local KV cache to enhance short-range modeling capacity.

We evaluate HySparse on both 7B dense and 80B Mixture-of-Experts (MoE) model settings.
For the 7B dense model, we adopt a full-to-sparse layer ratio of 1:3, while for the 80B MoE model, a more aggressive 1:11 ratio is used. In both cases, the final layer employs full attention to preserve global aggregation.
Across tasks and context lengths, HySparse consistently outperforms both full attention and hybrid SWA baselines, without incurring any additional KV cache cost relative to the hybrid SWA baseline.
Remarkably, in the HySparse 80B MoE model with 49 total layers, only 5 layers use full attention, meaning nearly 10×10\times KV cache reduction, while the models still delivers substantial performance gains.
Compared with Hybrid SWA, HySparse can significantly reduce the number of full attention layers, effectively pushing the hybrid ratio to its limit.
In summary, these results indicate that HySparse provides a simple and effective architectural solution to the core limitations of sparse attention, achieving strong long-context modeling capability with clear efficiency and memory advantages.

## 2 Background & Motivation

### 2.1 Training-free vs. Trainable Sparse Attention

Sparse attention methods can be divided into training-free and trainable approaches.
Training-free methods rely on fixed patterns or heuristics to identify important tokens.
They are applied as a drop-in modification at inference, enabling fast sparsity decisions with minimal computational cost [streamingllm, quest, duo, h2o].
However, applying sparsity only at inference creates a training–inference mismatch, which may lead to error accumulation in long decoding or multi-step reasoning [hu2026lil, liu2025quantizationreasoning, he2025nondeterminism].
Trainable sparse attention methods, in contrast, learn token importance during training through lightweight selection modules.
By integrating sparsity into the training process, they improve alignment between training and inference, selecting more informative tokens with higher recall and overall accuracy [seerattn\_v1, seer-r, dsa, nsa, moba, minicpm4, zhao2025infllm].
However, training these selection modules are non-trivial.
One approach uses auxiliary losses, such as self-distillation, to align the gating or indexer module with the original dense attention [seerattn\_v1, seer-r, dsa].
These methods are simple but suboptimal.
Alternatively, NSA [nsa] performs end-to-end sparse pretraining by injecting the compressed attention (selection module) output into the main attention.
This design allows the selection module to receive learning signals only indirectly through the final attention output, without direct supervision on its token selection decisions.

### 2.2 Hybrid Attention Architecture

To reduce quadratic compute and KV cache costs, hybrid attention has emerged as a promising solution for scaling context length.
For instance, MiniMax-01 [li2025minimax] integrates both linear attention and softmax attention mechanisms in a structured pattern. Similarly, Qwen3-Next [qwen3next2025] and Kimi Linear [kda] incorporate Gated DeltaNet [gdn] or its variants. The Nemotron family [nemotron2, blakeman2025nemotronh] and Jamba [lieber2024jamba] integrate Mamba modules [mamba, mamba2] with standard self-attention modules. Models such as GPT-OSS [gpt-oss], Gemma3 [gemma3], and MiMo-V2-Flash [xiao2026mimo] employ a heterogeneous interleaving of sliding window attention and global full attention layers. The sliding window size can be as small as 128 tokens with negligible KV cache overhead. Yet, the hybrid model with dynamic sparse attention has not been fully explored.

### 2.3 Cross-Layer Salient Token Stability

Several concurrent works have observed that salient tokens (sparse tokens with higher attention scores) tend to remain relatively stable across consecutive layers in standard transformer models [yang2024tidaldecode, hao2025omnikv, yang2025lessismore, zarch2025delta, deshmukh2025kascade].
These methods exploit this property to accelerate inference as a training-free manner. Specifically, they identify important tokens using a full attention layer and reuse the salient token indices in subsequent layers to perform sparse attention computation.
Inspired by these works, we elevate this empirical observation to a hybrid attention architecture for pretraining, in which full attention layers identify important tokens that are subsequently reused by following sparse attention layers.

### 2.4 Cross-layer KV Cache Sharing

Cross-layer KV cache sharing is a memory optimization technique in which Key and Value tensors computed in one layer are reused by subsequent layers, instead of being recomputed and stored independently for every layer.
This design substantially reduces the KV cache memory footprint, while empirical studies show that it incurs little to no degradation in model accuracy.
YOCO [yoco], CLA [cla], the Apple Foundation Model [li2025apple], and Gemma 3n [gemma3] integrate cross-layer KV cache sharing mechanisms directly into their model architectures.
SwiftKV [qiao2025swiftkv] adapts standard pretrained models to support cross-layer KV cache sharing over a subset of layers via distillation.
MiniCache [liu2024minicache] also observes that KV cache exhibits high similarity between adjacent layers in the middle-to-deep regions of LLMs, and proposes cross-layer compression methods to exploit this redundancy.

## 3 Methodology

### 3.1 HySparse Overview

!(/html/2602.03560/assets/x1.png)

Figure 1: HySparse Architecture Diagram. Each full attention layers is interleaved with multiple sparse attention layers. Sparse attention directly reuses the KV cache and important token indices from the preceding full attention layer.

As shown in Figure [1](#S3.F1 "Figure 1 ‣ 3.1 HySparse Overview ‣ 3 Methodology"), HySparse architecture replaces the standard Transformer backbone with repeated hybrid blocks that are composed of one full attention layer followed by NN consecutive sparse attention layers.
At its core, both the sparse important token indices and the KV caches used in these sparse layers are directly derived from the preceding full attention layer within the same block.

The full attention layer computes standard scaled dot product self-attention but additionally outputs block-wise attention importance scores SS, from which we derive TopK block indices. These indices are then reused by the next NN sparse layers. To reduce KV cache memory and bandwidth consumption, HySparse further incorporates cross-layer KV cache sharing. The sparse attention layers reuse the KV cache produced by the preceding full attention layer within a hybrid block for the block sparse attention branch. The sliding window attention (SWA) branch, in contrast, maintains its own lightweight KV cache to enhance short-range modeling capacity. Finally, sigmoid gates are applied to the output of the two branches [qiu2025gated] before summed as the final attention output.

### 3.2 Full Attention Layers

The full attention layers follow the standard softmax self-attention formulation used in Transformers. To identify salient tokens for subsequent sparse attention, the attention scores must be exposed for selection. However, materializing the full attention matrix is prohibitively expensive in terms of both memory and bandwidth. Consequently, modern Transformers rely on FlashAttention [flash1, flash2, flash3], which avoids explicitly storing attention scores by computing softmax in a tiled manner with online normalization.

To mitigate this issue, instead of outputting the full attention score matrix, HySparse only materializes block-level (tile-level) maximum attention scores for TopK selection. With the standard self-attention formulation:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒒t,𝒌t,𝒗t=𝐖q/k/v​𝒙t.\displaystyle\boldsymbol{q}\_{t},\ \boldsymbol{k}\_{t},\ \boldsymbol{v}\_{t}=\mathbf{W}\_{q/k/v}\,\boldsymbol{x}\_{t}. |  | (1) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒐t=∑i=1texp⁡(𝒒t⊤​𝒌id)∑j=1texp⁡(𝒒t⊤​𝒌jd)​𝒗i,\displaystyle\boldsymbol{o}\_{t}=\sum\_{i=1}^{t}\frac{\exp\!\left(\frac{\boldsymbol{q}\_{t}^{\top}\boldsymbol{k}\_{i}}{\sqrt{d}}\right)}{\sum\_{j=1}^{t}\exp\!\left(\frac{\boldsymbol{q}\_{t}^{\top}\boldsymbol{k}\_{j}}{\sqrt{d}}\right)}\,\boldsymbol{v}\_{i}, |  | (2) |

where dd is the head dimension size. Let BB be the block size of our attention score output, and there will be ⌈t/B⌉\lceil t/B\rceil number of blocks. We define the column token index set at block index ii as
ℬi={(i−1)​B+1,…,min⁡(i​B,N)}\mathcal{B}\_{i}=\{(i-1)B+1,\ldots,\min(iB,N)\}. Then the block-level max attention score 𝐒∈Rt×⌈t/B⌉\mathbf{S}\in R^{t\times\lceil t/B\rceil} will be:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐒ti=maxi′∈ℬi⁡(exp⁡(𝒒t⊤​𝒌i′d)∑j=1texp⁡(𝒒t⊤​𝒌jd))\displaystyle\mathbf{S}\_{t}^{i}=\max\_{i^{{}^{\prime}}\in\mathcal{B}\_{i}}\ \left(\frac{\exp\!\left(\frac{\boldsymbol{q}\_{t}^{\top}\boldsymbol{k}\_{i^{{}^{\prime}}}}{\sqrt{d}}\right)}{\sum\_{j=1}^{t}\exp\!\left(\frac{\boldsymbol{q}\_{t}^{\top}\boldsymbol{k}\_{j}}{\sqrt{d}}\right)}\right) |  | (3) |

We find that the 𝐒t\mathbf{S}\_{t} can be easily obtained by slightly modifying the FlashAttention kernel, following the approach similar to [seerattn\_v1, seer-r]. Specifically, FlashAttention already computes the row-wise maximum of the attention logits during its online softmax procedure, and this intermediate result can be reused to derive block-wise attention scores by storing and appropriately rescaling it.

Algorithm 1  FlashAttention with Block Attention Score Output (assuming B=BNB=B\_{N} for simplicity)

1:Queries 𝐐∈ℝt×d\mathbf{Q}\in\mathbb{R}^{t\times d}, Keys 𝐊∈ℝt×d\mathbf{K}\in\mathbb{R}^{t\times d}, Values 𝐕∈ℝt×d\mathbf{V}\in\mathbb{R}^{t\times d}, softmax scale τ\tau

2:Attention output 𝐎∈ℝt×d\mathbf{O}\in\mathbb{R}^{t\times d}, Block attention scores 𝐒∈ℝt×⌈t/B⌉\mathbf{S}\in\mathbb{R}^{t\times\lceil t/B\rceil}

3:Tr←⌈t/BM⌉T\_{r}\leftarrow\lceil t/B\_{M}\rceil,  Tc←⌈t/BN⌉T\_{c}\leftarrow\lceil t/B\_{N}\rceil

4:Initialize: 𝐎i\mathbf{O}\_{i}, 𝐒i\mathbf{S}\_{i}, 𝒎i\boldsymbol{m}\_{i}, ℓi\boldsymbol{\ell}\_{i}

5:for i=0,…,Tr−1i=0,\ldots,T\_{r}-1 do

6:  Load 𝐐i∈ℝBM×d\mathbf{Q}\_{i}\in\mathbb{R}^{B\_{M}\times d} from HBM to SRAM

7:  for j=0,…,Tc−1j=0,\ldots,T\_{c}-1 do

8:   Load 𝐊j,𝐕j∈ℝBN×d\mathbf{K}\_{j},\mathbf{V}\_{j}\in\mathbb{R}^{B\_{N}\times d} from HBM to SRAM

9:   𝐀i​j←𝐐i​𝐊j⊤⋅τ\mathbf{A}\_{ij}\leftarrow\mathbf{Q}\_{i}\mathbf{K}\_{j}^{\top}\cdot\tau

10:   𝒎~i​j←rowmax​(𝐀i​j)\tilde{\boldsymbol{m}}\_{ij}\leftarrow\text{rowmax}(\mathbf{A}\_{ij}), Store 𝒎~i​j\tilde{\boldsymbol{m}}\_{ij} to 𝐒i​j\mathbf{S}\_{ij}

11:   𝒎i←max​(𝒎i,𝒎~i​j)\boldsymbol{m}\_{i}\leftarrow\text{max}(\boldsymbol{m}\_{i},\tilde{\boldsymbol{m}}\_{ij})

12:   Update 𝐎i,ℓi\mathbf{O}\_{i},\boldsymbol{\ell}\_{i} as in FlashAttention

13:  end for

14:  𝐎i←𝐎i/ℓi\mathbf{O}\_{i}\leftarrow\mathbf{O}\_{i}/\boldsymbol{\ell}\_{i}, Write 𝐎i\mathbf{O}\_{i} to HBM

15:  for j=0,…,Tc−1j=0,\ldots,T\_{c}-1 do

16:   𝐒i​j←(𝐒i​j−𝒎i)/ℓi\mathbf{S}\_{ij}\leftarrow(\mathbf{S}\_{ij}-\boldsymbol{m}\_{i})\,/\,\boldsymbol{\ell}\_{i}, Write 𝐒i​j\mathbf{S}\_{ij} to HBM

17:  end for

18:end for

19:return 𝐎,𝐒\mathbf{O},\mathbf{S}

Algorithm [1](#alg1 "Algorithm 1 ‣ 3.2 Full Attention Layers ‣ 3 Methodology") summarizes the modified FlashAttention procedure, and we assume the sparse attention block size BB is the same as BNB\_{N} for simplicity. In addition to the standard attention output, the kernel emits block-level attention scores that can be directly used for salient block selection in later sparse attention layers with negligible overhead.

With the block-wise attention scores SS, we apply a TopK operator to select key-block indices ℐ\mathcal{I} that are reused by the subsequent sparse attention layers. Noted that attending to kk tokens in sparse attention corresponding to selecting k/Bk/B TopK blocks of tokens, where BB is the block size. In HySparse, the default kk and BB is 1024 and 64, respectively.
Under Grouped-Query Attention (GQA) [gqa], we further aggregate SS within each query group (via a group-wise maximum) so that all heads in the same group share identical sparse indices, improving sparse attention kernel efficiency and reducing indexing overhead.

### 3.3 Sparse Attention Layers

Each sparse layer contains two attention branches that operate on the same query but use different KV sources.
Block Sparse Attention branch attends only to key-value blocks indexed by ℐ\mathcal{I}, where both ℐ\mathcal{I} and the KV cache are derived from the preceding full attention layer.
SWA branch attends to a local sliding window of size ww with its own small KV cache, improving locality and expressivity. ww is set to be 128 in HySparse implementation.
The two branch outputs are then fused via a lightweight sigmoid gate. The detailed processes can be model as below. First, we compute the standard SWA branch output using:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 𝒒t′,𝒌t′,𝒗t′=𝐖q′/k′/v′​𝒙t\displaystyle\boldsymbol{q}\_{t}^{\prime},\boldsymbol{k}\_{t}^{\prime},\boldsymbol{v}\_{t}^{\prime}=\mathbf{W}\_{q^{\prime}/k^{\prime}/v^{\prime}}\boldsymbol{x}\_{t} |  | (4) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒐t′=∑i=t−w+1t\displaystyle\boldsymbol{o}\_{t}^{\prime}=\sum\_{i=t-w+1}^{t} | exp⁡(𝒒t′⁣⊤​𝒌i′d)∑j=t−w+1texp⁡(𝒒t′⁣⊤​𝒌j′d)​𝒗i′\displaystyle\frac{\exp\!\left(\frac{\boldsymbol{q}\_{t}^{\prime\top}\boldsymbol{k}^{\prime}\_{i}}{\sqrt{d}}\right)}{\sum\_{j=t-w+1}^{t}\exp\!\left(\frac{\boldsymbol{q}\_{t}^{\prime\top}\boldsymbol{k}^{\prime}\_{j}}{\sqrt{d}}\right)}\,\boldsymbol{v}\_{i}^{\prime} |  | (5) |

Then, when computing the sparse attention branch, we concatenate the selected key and value blocks from the shared 𝐊\mathbf{K}, 𝐕\mathbf{V} from the full attention layers using block indices ℐ\mathcal{I}. The sparse attention and SWA branch uses the same query 𝒒t′\boldsymbol{q}\_{t}^{\prime}, and the output can then be written as:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝐊~,𝐕~\displaystyle\tilde{\mathbf{K}},\tilde{\mathbf{V}} | =concat​({𝐊/𝐕[(j−1)​B+1:j​B]}j∈ℐ)\displaystyle=\mathrm{concat}\Big(\{\mathbf{K}/\mathbf{V}\_{[(j-1)B+1:\,jB]}\}\_{j\in\mathcal{I}}\Big) |  | (6) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒐~t\displaystyle\tilde{\boldsymbol{o}}\_{t} | =∑i=1kexp⁡(𝒒t′⁣⊤​𝒌~id)∑j=1kexp⁡(𝒒t′⁣⊤​𝒌~jd)​𝒗~i\displaystyle=\sum\_{i=1}^{k}\frac{\exp\!\left(\frac{\boldsymbol{q}\_{t}^{\prime\top}\tilde{\boldsymbol{k}}\_{i}}{\sqrt{d}}\right)}{\sum\_{j=1}^{k}\exp\!\left(\frac{\boldsymbol{q}\_{t}^{\prime\top}\tilde{\boldsymbol{k}}\_{j}}{\sqrt{d}}\right)}\,\tilde{\boldsymbol{v}}\_{i} |  | (7) |

Finally, we apply a sigmoid gate on both branch outputs and sum them to obtain the final attention layer output [qiu2025gated].

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | g~t,gt′\displaystyle\tilde{g}\_{t},g\_{t}^{\prime} | =σ​(𝐖g~/g′​𝒙t)\displaystyle=\sigma\!\left(\mathbf{W}\_{\tilde{g}/g^{\prime}}\boldsymbol{x}\_{t}\right) |  | (8) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒐t\displaystyle\boldsymbol{o}\_{t} | =g~t⊙𝒐~t+gt′⊙𝒐t′\displaystyle=\tilde{g}\_{t}\odot\tilde{\boldsymbol{o}}\_{t}\;+\;g\_{t}^{\prime}\odot\boldsymbol{o}\_{t}^{\prime} |  | (9) |

Through our experiments, we find that maintaining an independent KV cache for the SWA branch is essential for preserving model expressivity. One possible explanation is that SWA primarily serves as a local information pathway and requires different representations to capture short-range coherence, whereas the KV shared from the preceding full attention layer is optimized for global retrieval and may lack sufficient local features. The two-branch gated fusion yields a dynamic mixture of global and local information while remaining efficient in both computation and memory usage.

## 4 Experiments

### 4.1 Experiments Setup

|  |  |  |
| --- | --- | --- |
| Configuration | 7B Dense | 80B MoE |
| Layers | 3636 | 4949 |
| Attention Heads (Q / KV) | 32/832/8 | 64/464/4 |
| Head Dimensions | 128128 | 128128 |
| Hidden Size | 40964096 | 20482048 |
| Hybrid Ratio (Full : Sparse) | 1:31:3 | 1:111:11 |
| Sliding Window Size | 128128 | 128128 |
| Sparse Attn Block Size | 6464 | 6464 |
| Sparse Attn TopK Tokens | 10241024 | 10241024 |
| MoE Expert (Activated/Total) | – | 8:5128:512 |

Table 1: Model Architecture Configurations.

#### Model Configuration

In the following experiments, we use a standard 7B dense Transformer with 36 layers and an 80B-A3B MoE model with 49 layers.
We adopt Grouped-Query Attention (GQA), using 32 query heads with 8 KV heads for the 7B dense model, and 64 query heads with 4 KV heads for the 80B MoE model.
We evaluate three architectures:
(1) Full-Attn: all layers use standard full attention.
(2) Hybrid SWA: hybrid sliding window attention models, using a full-to-SWA layer ratio of 1:3 for the 7B model and 1:11 for the 80B MoE model.
(3) HySparse: hybrid models with the same hybrid ratios as Hybrid SWA, but augmenting SWA with the proposed sparse attention (Top-1024 tokens with block size 64).
For all hybrid models, the final layer uses full attention. For sparse attention and sliding window attention, we incorporate per-head learnable sink biases, following the approach in gpt-oss [gpt-oss].
For Full-Attn in the MoE setting, we additionally employ gated attention [qiu2025gated] to stabilize training.
Detailed model configurations are listed in Table [1](#S4.T1 "Table 1 ‣ 4.1 Experiments Setup ‣ 4 Experiments").

#### Training Hyper-parameters

For the 7B models, we first train on 1T tokens with a sequence length of 8,192 using the AdamW optimizer
(β1=0.9\beta\_{1}=0.9, β2=0.95\beta\_{2}=0.95, ϵ=10−10\epsilon=10^{-10}), weight decay 0.1, and gradient clipping with a maximum norm of 1.0.
Training uses BF16 precision and a WSD schedule with a maximum learning rate of 8.3×10−48.3\times 10^{-4}.
To extend the models for long-context evaluation, we further train on 200B tokens with a sequence length of 32,768 and a learning rate of 3.0×10−53.0\times 10^{-5}. The RoPE base frequency is adjusted to 640,000 at this stage.
For the 80B MoE model, training is performed on 500B tokens with a sequence length of 32,768 using the WSD schedule with a maximum learning rate of 1×10−31\times 10^{-3}, and the RoPE base frequency is also set to 640,000.

#### Evaluation Benchmark

We evaluate HySparse based on a series of benchmarks, encompassing various capabilities: (1) General language understanding and reasoning, including BBH [suzgun2022challenging], MMLU [hendrycks2020measuring], MMLU-Redux [gema2024we], MMLU-Pro [wang2024mmlu], DROP [dua2019drop], ARC [clark2018think], HellaSwag [zellers2019hellaswag], WinoGrande [sakaguchi2021winogrande], TriviaQA [joshi2017triviaqa],
(2) Mathematics reasoning, including
GSM8K [cobbe2021training], MATH [hendrycks2021measuring]
(3) Coding, including HumanEval [humaneval], MBPP [mbpp],
(4) Chinese understanding, including C-Eval [huang2023c], CMMLU [li2023cmmlu].
(5) Long context, Ruler [hsieh2024ruler].

### 4.2 Performance of HySparse on General Benchmarks

|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Benchmark | # Shots | 7B Dense (Hybrid 1:3) | | | 80B MoE (Hybrid 1:11) | | |
| Full-Attn | Hybrid SWA | HySparse | Full-Attn | Hybrid SWA | HySparse |
| General | | | | | | | |
| BBH | 3-shot | 52.2 | 54.0 | 53.5 | 56.1 | 48.2 | 56.3 |
| MMLU | 5-shot | 56.9 | 57.5 | 58.8 | 61.8 | 54.9 | 62.2 |
| MMLU-Redux | 5-shot | 59.6 | 60.8 | 61.6 | 65.6 | 57.4 | 66.2 |
| MMLU-Pro | 5-shot | 26.8 | 26.5 | 29.0 | 33.8 | 27.2 | 32.6 |
| DROP | 3-shot | 53.1 | 43.8 | 52.4 | 56.7 | 47.8 | 56.5 |
| ARC-Challenge | 25-shot | 70.2 | 74.9 | 75.0 | 78.4 | 63.9 | 77.6 |
| HellaSwag | 10-shot | 77.5 | 77.8 | 78.1 | 78.2 | 77.1 | 79.0 |
| WinoGrande | 5-shot | 73.7 | 74.9 | 74.3 | 71.2 | 69.0 | 72.1 |
| TriviaQA | 5-shot | 50.1 | 50.0 | 51.1 | 54.7 | 52.2 | 55.5 |
| Mathematics | | | | | | | |
| GSM8K | 8-shot | 33.3 | 35.6 | 37.9 | 53.8 | 45.3 | 54.1 |
| MATH | 4-shot | 9.2 | 9.2 | 10.1 | 28.6 | 25.8 | 30.8 |
| Code | | | | | | | |
| HumanEval | 0-shot | 25.0 | 22.0 | 23.5 | 35.4 | 31.7 | 38.4 |
| MBPP | 3-shot | 51.0 | 52.8 | 51.6 | 55.3 | 51.9 | 59.3 |
| Chinese | | | | | | | |
| C-Eval | 5-shot | 50.6 | 50.6 | 52.2 | 64.6 | 58.8 | 65.0 |
| CMMLU | 5-shot | 52.5 | 52.9 | 54.5 | 66.7 | 58.4 | 67.0 |

Table 2: 
Comparison of HySparse across 7B dense models (trained on 1T tokens) and 80B MoE models (trained on 500B tokens) with Full-Attn and Hybrid SWA. Best results in each row are highlighted in bold.

#### 7B Dense Performance

Table [2](#S4.T2 "Table 2 ‣ 4.2 Performance of HySparse on General Benchmarks ‣ 4 Experiments") compares 7B models under three attention variants.
Overall, HySparse achieves strong performance across general benchmarks, mathematics, and Chinese understanding, while Hybrid SWA offers a competitive and computationally efficient baseline that is particularly strong on BBH and MBPP+.
Specifically, HySparse surpasses the Full-Attn baseline on a broad set of knowledge and reasoning benchmarks, including MMLU (58.8 vs. 56.9), MMLU-Redux (61.6 vs. 59.6), and MMLU-Pro (29.0 vs. 26.8), suggesting that sparse token selection can preserve (and even enhance) global reasoning and factual recall despite reduced attention computation.
HySparse also yields consistent gains on multi-step reasoning tasks such as GSM8K and MATH.
On classic multiple-choice commonsense and reading comprehension benchmarks, HySparse is either best or comparable, with slight improvements on ARC-Challenge, HellaSwag, and TriviaQA.
For Chinese benchmarks, HySparse provides clear gains on both C-Eval (52.2 vs. 50.6) and CMMLU (54.5 vs. 52.5).

#### Scaling to 80B MoE

We further evaluate HySparse in an 80B-A3B MoE setting with a more aggressive full-to-sparse layer ratio of 1:11.
Despite having only five full attention layers, HySparse outperforms both Full-Attn and Hybrid SWA across nearly all benchmarks; only MMLU-Pro, DROP, and ARC-C are slightly lower than Full-Attn.
Notably, the performance gains are often larger than those observed in the 7B dense setting.
In this regime, Hybrid SWA exhibits noticeable accuracy degradation compared to Full-Attn on several benchmarks, including BBH, DROP, the MMLU series, GSM8K, and Chinese understanding.
This suggests that relying solely on local window attention becomes insufficient as the hybrid ratio becomes more aggressive.
By introducing the sparse attention branch, HySparse mitigates this gap by recovering access to globally relevant tokens selected from the full attention, and in many cases even surpasses the Full-Attn baseline, while requiring 10×10\times less KV cache.
These results highlight a key advantage of HySparse: the number of full attention layers can be substantially reduced without sacrificing modeling capability, which is also the design rationale of HySparse.

### 4.3 Long-context Benchmarks

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Size | Ctx | Type | S1 | S2 | S3 | MK1 | MK2 | MK3 | MQ | MV | VT | CWE | FWE | Total |
| 7B | 16k | Full-Attn | 100.0 | 100.0 | 100.0 | 98.6 | 100.0 | 96.4 | 99.2 | 99.4 | 94.4 | 37.1 | 97.4 | 93.0 |
| Hybrid SWA | 100.0 | 100.0 | 99.8 | 97.6 | 99.8 | 96.6 | 97.4 | 94.7 | 99.1 | 23.7 | 98.5 | 91.6 |
| HySparse | 100.0 | 100.0 | 100.0 | 99.4 | 99.6 | 96.6 | 96.7 | 89.8 | 97.2 | 60.8 | 95.5 | 94.1 |
| 32k | Full-Attn | 100.0 | 100.0 | 100.0 | 98.0 | 99.4 | 75.8 | 96.4 | 98.3 | 90.5 | 16.6 | 95.1 | 88.2 |
| Hybrid SWA | 100.0 | 100.0 | 99.6 | 96.2 | 98.8 | 53.4 | 93.4 | 87.2 | 98.5 | 10.4 | 88.5 | 84.2 |
| HySparse | 100.0 | 100.0 | 99.8 | 98.2 | 99.6 | 76.2 | 88.8 | 94.7 | 91.2 | 38.8 | 95.1 | 89.3 |
| 80B | 16k | Full-Attn | 100.0 | 99.8 | 92.6 | 99.6 | 99.2 | 93.0 | 97.3 | 94.9 | 95.4 | 74.5 | 80.4 | 93.6 |
| Hybrid SWA | 95.2 | 94.8 | 70.8 | 93.2 | 86.4 | 69.4 | 83.2 | 57.8 | 66.7 | 13.3 | 69.2 | 72.7 |
| HySparse | 100.0 | 99.8 | 99.0 | 98.2 | 100.0 | 99.6 | 92.2 | 91.3 | 90.3 | 40.2 | 86.4 | 90.6 |
| 32k | Full-Attn | 100.0 | 99.2 | 81.2 | 99.0 | 99.4 | 77.0 | 86.0 | 79.5 | 74.5 | 40.7 | 66.7 | 82.1 |
| Hybrid SWA | 100.0 | 98.2 | 58.8 | 89.6 | 90.2 | 61.0 | 74.1 | 56.6 | 73.1 | 8.4 | 54.3 | 69.5 |
| HySparse | 100.0 | 100.0 | 99.4 | 96.8 | 99.0 | 98.4 | 89.5 | 85.7 | 89.6 | 20.8 | 82.1 | 87.4 |

Table 3: RULER benchmark performance for 7B dense and 80B MoE models.
Note that all 7B models were first trained on 1T tokens with a sequence length of 8K, and then further trained on 200B tokens with a sequence length of 32K.
All 80B MoE models were trained on 500B tokens with a sequence length of 32K.

Table [3](#S4.T3 "Table 3 ‣ 4.3 Long-context Benchmarks ‣ 4 Experiments") shows that HySparse consistently preserves strong long-context accuracy.
For the 7B dense model, HySparse improves the overall score over both baselines at 16k and 32k, reaching totals of 94.1 and 89.3 (vs. 93.0 and 88.2 for Full-Attn and 91.6 and 84.2 for Hybrid SWA). The gains are most apparent on the harder multi-key/value and reasoning-heavy subsets, e.g., HySparse substantially boosts CWE compared to baselines (60.8 at 16k and 38.8 at 32k), indicating better robustness as context grows.
For the 80B MoE model, Hybrid SWA degrades sharply (72.7 at 16k and 69.5 at 32k) under an aggressive hybrid ratio, whereas HySparse remains competitive with Full-Attn at 16k (90.6 vs. 93.6) and notably surpasses it at 32k (87.4 vs. 82.1), driven by large recoveries on difficult components such as MK3 (98.4 vs. 77.0) and stronger VT/MQ/MV stability. In general, HySparse provides on-par long-context capabilities compared with Full-Attn across settings, while significantly reducing computation and KV cache size.

### 4.4 Ablation Study

In this section, we present detailed ablation studies on key architecture design choices, focusing on: (1) whether to include an intra-layer SWA branch within each sparse layer, and (2) how KV cache sharing is applied across the sparse attention and SWA branches. All ablation studies are conducted on the 7B dense models. The results are summarized in Table [4](#S4.T4 "Table 4 ‣ 4.4 Ablation Study ‣ 4 Experiments") and Figure [2](#S4.F2 "Figure 2 ‣ 4.4 Ablation Study ‣ 4 Experiments").

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Group | Method | DROP | GSM8K | MMLU | MMLU-Pro | BBH |
| Baselines | Full-Attn | 52.6 | 32.6 | 56.8 | 26.8 | 52.1 |
| Hybrid SWA | 43.9 | 35.7 | 57.6 | 26.6 | 54.3 |
| Oracle Token Selection  (w/o KV cache sharing) | HySparse (w/o intra-layer-SWA) | 46.4 | 29.7 | 57.1 | 25.0 | 48.2 |
| HySparse (w/ intra-layer-SWA) | 52.2 | 37.7 | 56.1 | 26.5 | 52.4 |
| KV Cache Sharing  (w/ intra-layer SWA) | HySparse (sharing for SA & SWA) | 47.9 | 30.2 | 52.8 | 23.2 | 47.2 |
| HySparse (sharing only for SA) | 51.9 | 36.7 | 58.4 | 29.0 | 53.9 |

Table 4: Ablation of Different Architecture Design Choices on 7B experiments.

!(/html/2602.03560/assets/x2.png)

Figure 2: HySparse Accuracy vs. Training Iterations.

#### Study 1: Intra-layer Hybridization with SWA

This study investigates whether sparse attention layers still benefit from an additional SWA branch when the sparse indices are provided by oracle token selection (with KV cache sharing disabled). A natural hypothesis is that the SWA branch might be redundant: if recent or local-context tokens are important, oracle selection from the preceding full attention layer should already include them among the selected blocks. However, our results indicate that removing the SWA branch leads to a clear accuracy drop. Specifically, we compare HySparse (w/o intra-layer SWA) against HySparse (w/ intra-layer SWA) under the same oracle-selected sparse indices, with both branches sharing the same QKV projection layers.

As shown in Table [4](#S4.T4 "Table 4 ‣ 4.4 Ablation Study ‣ 4 Experiments"), adding the intra-layer SWA branch yields consistent improvements across most benchmarks: DROP increases from 46.4 to 52.2 (+5.8), GSM8K from 29.7 to 37.7 (+8.0), MMLU-Pro from 25.0 to 26.5 (+1.5), and BBH from 48.2 to 52.4 (+4.2). These gains suggest that even with high-quality sparse selection, a dedicated sliding window pathway remains important for modeling short-range coherence and local computation patterns that are not reliably captured by sparse global retrieval alone. Additionally, the SWA branch may help stabilize optimization by providing a consistent local pathway, particularly during early training stages.

#### Study 2: Cross-layer KV Cache Sharing Configuration

This study examines how KV cache sharing should be applied when each sparse layer contains both a sparse attention (SA) branch for global retrieval and a sliding window attention (SWA) branch for local modeling.
A natural design choice is to maximize memory reuse by sharing the same KV cache for both branches, i.e., reusing the KV cache produced by the preceding full attention layer for both SA and SWA.
However, this coupling can be overly restrictive because the two branches serve distinct roles: SA primarily requires globally informative keys and values aligned with block-level retrieval, whereas SWA benefits from a dedicated local representation that emphasizes short-range coherence and local computation patterns.
In our KV cache sharing experiments, we compare HySparse (sharing for both SA & SWA) against HySparse (sharing only for SA).

As shown in Table [4](#S4.T4 "Table 4 ‣ 4.4 Ablation Study ‣ 4 Experiments"), sharing KV caches for both SA and SWA substantially degrades accuracy. In contrast, sharing KV only for the SA branch while maintaining an independent KV cache for SWA recovers and improves performance across all evaluated tasks: DROP increases from 47.9 to 51.9 (+4.0), GSM8K from 30.2 to 36.7 (+6.5), MMLU from 52.8 to 58.4 (+5.6), MMLU-Pro from 23.2 to 29.0 (+5.8), and BBH from 47.2 to 53.9 (+6.7). These results suggest that SA can safely reuse the cross-layer KV cache from full attention to save GPU memory, whereas SWA should maintain its own KV cache to preserve strong local information. Forcing SWA to reuse the KV cache from the preceding full attention layer likely deprives it of the short-range, local features it requires and entangles it with globally optimized representations, thereby weakening the local pathway and reducing overall accuracy.

## 5 Discussion & Future Works

#### Can We Ultimately Avoid Full Attention?

Our findings connect to a broader trend in efficient attention, including hybrid attention and sparse attention methods. A recurring theme is that it remains challenging to completely eliminate O​(n2)O(n^{2})-style full attention components in practice: hybrid models retain explicit full attention layers, while sparse attention methods such as SeerAttention [seerattn\_v1] and DSA [dsa] typically rely on gating or indexing mechanisms that still operate in O​(n2)O(n^{2}), albeit in a compressed form. In this context, what matters most is the *ratio* of expensive global computation to cheaper local or sparse computation, as well as GPU memory usage. Our results suggest that HySparse, with oracle token selection and cross-layer KV sharing, provides a promising approach to reduce this ratio while preserving long-context modeling capabilities.

#### Potential of HySparse for Efficient KV Cache Offloading

HySparse also points to a straightforward systems-level strategy for long-context serving: offload the full attention KV cache to external memory and pre-fetch it before computation, while keeping only the persistent selected/sparse KV on the GPU for subsequent sparse attention layers. Previous work such as OmniKV [hao2025omnikv] has explored similar approaches in a post-training setting. This technique has the potential to significantly reduce the KV cache footprint on GPU, enabling larger batch sizes and improving overall inference efficiency.

## 6 Conclusion

In this work, we introduced Hybrid Sparse Attention (HySparse), a simple yet effective hybrid attention architecture that interleaves each full attention layer with multiple sparse-attention layers.
By strategically deriving important token selections and KV caches from preceding full attention layers, HySparse eliminates the need for proxy-based token selection and enables sparse layers to operate without additional memory overhead.
Importantly, HySparse allows for a substantial reduction in the number of full attention layers in hybrid models without compromising modeling capabilities.
In future work, we plan to scale HySparse to even larger model sizes and train on more tokens to fully exploit its potential for efficient and accurate long-context modeling.

## References
