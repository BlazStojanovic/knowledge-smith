---
arxiv: '2512.12087'
authors:
- Jiayi Yuan
- Cameron Shinn
- Kai Xu
- Jingze Cui
- George Klimiashvili
- Guangxuan Xiao
- Perkz Zheng
- Bo Li
- Yuxin Zhou
- Zhouhai Ye
- Weijie You
- Tian Zheng
- Dominic Brown
- Pengbo Wang
- Markus Hoehnerbach
- Richard Cai
- Julien Demouth
- John D. Owens
- Xia Hu
- Song Han
- Timmy Liu
- Huizi Mao
parser: ar5iv
retrieved: '2026-05-28'
source: paper
title: 'BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding'
url: https://arxiv.org/abs/2512.12087
year: 2025
---

[2512.12087] 1 Introduction



marginparsep has been altered.
  
topmargin has been altered.
  
marginparwidth has been altered.
  
marginparpush has been altered.
  
The page layout violates the ICML style.Please do not change the page layout, or include packages like geometry,
savetrees, or fullpage, which change it for you.
We’re not able to reliably undo arbitrary changes to the style. Please remove
the offending package(s), or layout-changing commands and try again.

 

BLASST: Dynamic BLocked Attention Sparsity
  
via Softmax Thresholding

 

Anonymous Authors1

###### Abstract

The growing demand for long-context inference capabilities in Large Language Models (LLMs) has intensified the computational and memory bottlenecks inherent to the standard attention mechanism. To address this challenge, we introduce BLASST, a drop-in sparse attention method that dynamically prunes the attention matrix without any pre-computation or proxy scores. Our method uses a fixed threshold and existing information from online softmax to identify negligible attention scores, skipping softmax computation, Value block loading, and the subsequent matrix multiplication. This fits seamlessly into existing FlashAttention kernel designs with negligible latency overhead. The approach is applicable to both prefill and decode stages across all attention variants (MHA, GQA, MQA, and MLA), providing a unified solution for accelerating long-context inference. We develop an automated calibration procedure that reveals a simple inverse relationship between optimal threshold and context length, enabling robust deployment across diverse scenarios. Maintaining high accuracy, we demonstrate a 1.62×\times speedup for prefill at 74.7% sparsity and a 1.48×\times speedup for decode at 73.2% sparsity on modern GPUs. Furthermore, we explore sparsity-aware training as a natural extension, showing that models can be trained to be inherently more robust to sparse attention patterns, pushing the accuracy-sparsity frontier even further.

††footnotetext: 1Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country.
Correspondence to: Anonymous Author <anon.email@domain.com>.
  
Preliminary work. Under review by the
Machine Learning and Systems (MLSys) Conference. Do not distribute.

## 1 Introduction

Large Language Models (LLMs) have revolutionized natural language processing, achieving remarkable performance across diverse tasks. However, their practical deployment faces a critical bottleneck: the quadratic computational complexity of the attention mechanism. As applications increasingly demand longer context windows — from processing entire codebases roziere2023code to analyzing lengthy documents zeng2025glm and maintaining extended conversations achiam2023gpt — this bottleneck becomes increasingly severe. Recent models like Deepseek-R1 guo2025deepseek and Qwen3 yang2025qwen3 support context lengths up to 128K tokens, with some models pushing to 1M tokens comanici2025gemini. Yet processing such long sequences remains computationally prohibitive, with attention computation dominating both latency and memory consumption. For a sequence of length nn, the attention mechanism requires O​(n2)O(n^{2}) operations and memory accesses, making real-world deployment of long-context

Figure 1: Overview of BLASST. Blocks along a row of the attention matrix are sequentially processed. We (1) update the running row max (m(j)m^{(j)}) as in FlashAttention, (2) compute the block max (m~(j)\tilde{m}^{(j)}) for each SjS\_{j} block (Q​Kj⊤QK\_{j}^{\top}), and (3) skip subsequent work if the block max is lower than the running max by more than the input threshold, ln⁡(λ)\ln(\lambda). Full details can be found in Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1.2 Algorithm Design ‣ 3.1 Pruning Attention with Running Maximums ‣ 3 Methodology").

models are challenging even with state-of-the-art hardware. While FlashAttention dao2022flashattention and its successors have optimized memory bandwidth utilization through tiling and kernel fusion, they still compute the full attention matrix, leaving the fundamental quadratic complexity unaddressed.

*Sparse attention* methods have emerged as a promising solution by computing only a subset of the full attention matrix. However, existing approaches suffer from fundamental limitations. First, they require non-trivial operations to determine sparsity patterns: methods like MInference jiang2024minference and XAttention xu2025xattention perform expensive pre-computation passes to identify important blocks, often negating their theoretical speedup; while static sparsity patterns xiao2023efficient avoid pre-computation but are inflexible and often suboptimal for diverse attention distributions across different tasks and context lengths. Second, these methods rely on proxy importance scores such as accumulated attention weights or query-key similarities, which can be inaccurate and miss critical token interactions. Third, most existing sparse attention methods focus exclusively on either the prefill or decode phase, missing opportunities for end-to-end optimization.

In this paper, we present BLASST (BLocked Attention Sparsity via Softmax Thresholding), a simple yet effective sparse attention method that dynamically prunes negligible attention blocks without any pre-computation overhead. Our key insight is that during FlashAttention’s block-wise online-softmax, we can identify and skip blocks whose contribution to the final output will be negligible based solely on already-computed information. Specifically, when processing blocks sequentially, we maintain a running maximum of attention scores. As shown in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction"), if a block’s local maximum score is significantly smaller than this running maximum (by a threshold λ\lambda), its post-softmax values will be near zero after normalization. We can therefore skip three expensive operations for such blocks: (1) computing the exponential for softmax, (2) loading the corresponding value block from HBM, and (3) performing the matrix multiplication with values. This simple pruning rule requires only a single comparison per block and seamlessly integrates into existing FlashAttention implementations.

To maximize the practical impact of BLASST, we develop highly optimized CUDA kernels that implement our sparse attention algorithm. Our kernels are designed with two key goals: (1) introduce minimal overhead for the block-skipping decision logic by reusing already-computed statistics, and (2) strategically target the bottleneck resources in each phase—reducing CUDA core and tensor core usage in compute-bound prefill, and reducing memory bandwidth consumption in memory-bound decode. We implement specialized kernels for both the prefill and decode phases, with optimizations tailored to their distinct computational patterns. On modern GPUs (H200, B200), our kernels achieve up to 1.62×\times speedup for prefill at 74.7% sparsity and 1.48×\times speedup for decode at 73.2% sparsity over FlashAttention baselines shah2024flashattention, while maintaining numerical stability and supporting various attention variants including grouped-query attention and sliding window attention.

Beyond the core algorithm and kernel implementation, we develop two key techniques to enhance BLASST’s deployment and performance. First, we propose an automated calibration procedure that determines optimal thresholds for any target sparsity level. Our calibration reveals a robust inverse relationship λ=a/L\lambda=a/L between threshold and context length LL, enabling reliable deployment across diverse scenarios without manual tuning. Second, we explore sparsity-aware training as a natural extension, showing that models can be trained to be inherently more robust to sparse attention patterns. This training approach further pushes the accuracy-sparsity frontier, enabling even higher sparsity levels with minimal loss in accuracy.

Our contributions include:

1. 1.

   A drop-in method with no pre-computation overhead and no proxy scores, achieving minimal accuracy loss.
2. 2.

   Automated hyperparameter selection and sparsity-aware training for robust, flexible, and extensible deployment.
3. 3.

   Optimized FlashAttention-based CUDA kernels for both prefill and decode, with high performance.

## 2 Related Works

Effectively exploiting the sparse attention property requires either reducing compute on unimportant interactions or reducing memory footprint (e.g., KV cache) without expensive selection overheads or retraining. Compared to the following related works, BLASST addresses both dimensions simultaneously, in a training-free manner.

### 2.1 Compute-Optimized Sparsity

Several approaches reduce attention *compute* by selecting important interactions. Static pattern methods like Sparse Transformer child2019generating, LongFormer beltagy2020longformer, and BigBird zaheer2020big reduce complexity through local or block-based attention. Retrieval head-based methods wu2024retrieval; xiao2024duoattention accelerate model decoding by focusing compute on crucial retrieval heads. Dynamic sparsity methods like MInference jiang2024minference use pre-computed importance scores, XAttention xu2025xattention ranks anti-diagonal blocks, and FlexPrefill lai2025flexprefill offers compiler-supported, flexible block patterns; while effective for prefill, their pre-computation and scheduling overheads can limit realized speedups. Training-aided sparsity such as SeerAttention gao2025seerattention induces high sparsity via (pre)training gates, improving efficiency but adding training cost and showing mixed downstream performance.

SpargeAttention zhang2025spargeattn has the closest design to BLASST. We differ in three key aspects: (1) BLASST optimizes both prefill and decode with specialized kernels, while SpargeAttention targets prefill only; (2) we make skip decisions directly using already-computed statistics with zero overhead, while SpargeAttention uses a separate prediction step; (3) our decode kernel skips Value loading from HBM, addressing memory-bound bottlenecks on top of compute savings. Additionally, we provide automated calibration and sparsity-aware training.

### 2.2 Memory-Optimized Sparsity

Token/KV sparsity focuses on reducing *memory* footprint and decode-time cost. H2O zhang2023h2o, TOVA oren2024transformers, and InfLLM xiao2024infllm discard tokens based on query patterns. StreamingLLM xiao2023efficient retains initial and recent tokens for consistent latency and memory usage. Quest tang2024quest prunes tokens conditioned on the current query, Rectified Sparse Attention sun2025rectified adaptively selects tokens to maintain accuracy at high sparsity, RocketKV behnam2025rocketkv compresses the KV cache with selective eviction, and recent KV compression for hyper-scaling lancucki2025inference further extends effective context; TidalDecode yang2024tidaldecode stabilizes decode efficiency with position-persistent patterns. These approaches primarily target memory via KV pruning/compression on decode phase, whereas BLASST directly reduces compute in both prefill and decode while remaining training-free.

### 2.3 New Attention Variants

Beyond the above methods, alternative mechanisms include Sliding Window Attention beltagy2020longformer, Linear or Gated Attention qiu2025gated, and State-Space Models (SSM) gu2023mamba. Native Sparse Attention (NSA) yuan2025native and DeepSeek Sparse Attention (DSA) deepseekai2024deepseekv32 while effective in some regimes, they often require architectural changes or retraining. By contrast, BLASST is a post-training method that accelerates both prefill and decode without proxy scores or complex pre-computation, integrating seamlessly with FlashAttention implementations.

## 3 Methodology

### 3.1 Pruning Attention with Running Maximums

The core insight of BLASST lies in the observation that during the computation of attention scores in FlashAttention, many blocks contribute negligibly to the final output after softmax normalization. Our method identifies and skips these blocks dynamically during the forward pass, without requiring pre-computation or proxy scores.

#### 3.1.1 Key Insight

In the standard attention mechanism, the softmax operation computes:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Attention​(Q,K,V)=softmax​(Q​K⊤dk)​V\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^{\top}}{\sqrt{d\_{k}}}\right)V |  | (1) |

During FlashAttention’s block-wise computation, we maintain a running maximum mi(j)m\_{i}^{(j)} across blocks. If a block’s local maximum m~i(j)\tilde{m}\_{i}^{(j)} is significantly smaller than the current running maximum, i.e., m~i(j)−mi(j)<ln⁡(λ)\tilde{m}\_{i}^{(j)}-m\_{i}^{(j)}<\ln(\lambda) for some threshold λ\lambda, then after exponentiation:

|  |  |  |  |
| --- | --- | --- | --- |
|  | exp⁡(m~i(j)−mi(j))<λ≈0\exp(\tilde{m}\_{i}^{(j)}-m\_{i}^{(j)})<\lambda\approx 0 |  | (2) |

Since the maximum value is bounded by λ\lambda, the block’s contribution to the final attention output will be negligible, allowing us to skip its computation entirely.

Intuitively, this criterion follows a three-step approximation. First, the ideal importance of each score Si​jS\_{ij} is its value relative to the (unknown) global maximum. Second, computing the true maximum on-the-fly is too expensive, so we use the running maximum as a tractable proxy and compare Si​jS\_{ij} against it. Third, to enable an efficient block-level decision inside the kernel, we replace token-level Si​jS\_{ij} with the block-local maximum, yielding the inexpensive condition (block\_max−running\_max)<ln⁡(λ)(\text{block\\_max}-\text{running\\_max})<\ln(\lambda).

#### 3.1.2 Algorithm Design

Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1.2 Algorithm Design ‣ 3.1 Pruning Attention with Running Maximums ‣ 3 Methodology") presents our modified FlashAttention forward pass. The key modification is the introduction of a dynamic pruning condition that saves both computation and memory bandwidth. Where We Save: When m~i(j)−mi(j)<ln⁡(λ)\tilde{m}\_{i}^{(j)}-m\_{i}^{(j)}<\ln(\lambda) (line 7), we skip:

1. 1.

   Compute savings (CUDA cores): The expensive exp⁡(⋅)\exp(\cdot) operations for computing P~i​j\tilde{P}\_{ij} require multiple instructions per element: MUFU.EX2 (exponential), FMUL (multiplication), and FADD (addition). We also skip the rowsum reduction operations (FADD instructions) for normalizing attention weights. For a typical block, this saves thousands of CUDA core instructions.
2. 2.

   Compute savings (Tensor cores) The matrix multiplication P~i​j​Vj\tilde{P}\_{ij}V\_{j}. In prefill phase, where kernels are compute-bound, avoiding these MMA operations provides a substantial speedup.
3. 3.

   Memory bandwidth savings: Loading the Value block VjV\_{j} from HBM to SRAM. This is particularly critical in decode phase, where attention is memory-bound.

Algorithm 1  FlashAttention with BLASST

1:Query blocks {Qi}i=1Tr\{Q\_{i}\}\_{i=1}^{T\_{r}}, Key blocks {Kj}j=1Tc\{K\_{j}\}\_{j=1}^{T\_{c}}, Value blocks {Vj}j=1Tc\{V\_{j}\}\_{j=1}^{T\_{c}}, threshold λ\lambda

2:Output blocks {Oi}i=1Tr\{O\_{i}\}\_{i=1}^{T\_{r}}

3:for i=1i=1 to TrT\_{r} do

4:  Initialize mi(0)=−∞m\_{i}^{(0)}=-\infty, Oi(0)=0O\_{i}^{(0)}=0, li(0)=0l\_{i}^{(0)}=0

5:  for j=1j=1 to TcT\_{c} do

6:   Compute Si​j=Qi​Kj⊤S\_{ij}=Q\_{i}K\_{j}^{\top} ⊳\triangleright Attention scores

7:   m~i(j)=rowmax⁡(Si​j)\tilde{m}\_{i}^{(j)}=\operatorname{rowmax}(S\_{ij}) ⊳\triangleright Local maximum

8:   mi(j)=max⁡(mi(j−1),m~i(j))m\_{i}^{(j)}=\max(m\_{i}^{(j-1)},\tilde{m}\_{i}^{(j)}) ⊳\triangleright Running maximum

9:   if m~i(j)−mi(j)<ln⁡(𝝀)\tilde{m}\_{i}^{(j)}-m\_{i}^{(j)}<\ln(\boldsymbol{\lambda}) then

10:     continue ⊳\triangleright *Skip this block*

11:   end if

12:   P~i​j=exp⁡(Si​j−mi(j))\tilde{P}\_{ij}=\exp(S\_{ij}-m\_{i}^{(j)}) ⊳\triangleright Compute attention weights

13:   li(j)=emi(j−1)−mi(j)​li(j−1)+rowsum⁡(P~i​j)l\_{i}^{(j)}=e^{m\_{i}^{(j-1)}-m\_{i}^{(j)}}l\_{i}^{(j-1)}+\operatorname{rowsum}(\tilde{P}\_{ij})

14:   Oi(j)=emi(j−1)−mi(j)​Oi(j−1)+P~i​j​VjO\_{i}^{(j)}=e^{m\_{i}^{(j-1)}-m\_{i}^{(j)}}O\_{i}^{(j-1)}+\tilde{P}\_{ij}V\_{j}

15:  end for

16:  Oi=Oi(Tc)/li(Tc)O\_{i}=O\_{i}^{(T\_{c})}/l\_{i}^{(T\_{c})} ⊳\triangleright Final normalization

17:end for

18:return {Oi}i=1Tr\{O\_{i}\}\_{i=1}^{T\_{r}}

Our approach directly reduces the total amount of computation by dynamically identifying and skipping negligible attention blocks during the forward pass. This simple yet effective modification requires minimal changes to the existing FlashAttention implementation while providing significant computational savings.

### 3.2 Calibration for Optimal Sparsity

A critical challenge in deploying BLASST is selecting the appropriate threshold λ\lambda that balances sparsity and accuracy. To understand this relationship, we conducted experiments on Llama-3.1-8B across RULER benchmark challenging subsets (NIAH\_MULTI, VT, FWE) with context lengths from 8K to 64K tokens.

Sparsity Determines Accuracy. Figure [2](#S3.F2 "Figure 2 ‣ 3.2 Calibration for Optimal Sparsity ‣ 3 Methodology") (left) shows relative accuracy degradation as a function of sparsity ratio. We normalize each curve by full attention result for fair comparison. Remarkably, all curves exhibit consistent degradation patterns: performance remains stable up to  60-70% sparsity, beyond which accuracy drops sharply. This consistency across diverse tasks and sequence lengths reveals that accuracy degradation is primarily determined by sparsity ratio itself, not dataset type or sequence length.

Threshold Calibration is Essential. To achieve consistent performance, we should maintain a fixed sparsity ratio rather than a fixed threshold. However, Figure [2](#S3.F2 "Figure 2 ‣ 3.2 Calibration for Optimal Sparsity ‣ 3 Methodology") (right) shows that achieving 75% sparsity requires λ≈1​e−4\lambda\approx 1e-4 for 8K contexts but only 1​e−51e-5 for 64K contexts. This necessitates adaptive calibration. Importantly, by targeting fixed sparsity through calibration, users can control and foresee the computational speedup, since performance gains scale predictably with the achieved sparsity level.

Figure 2: (Left) Relative accuracy drop across different datasets and context lengths shows consistent degradation patterns. All curves are normalized to their initial accuracy. (Right) Relationship between threshold and achieved sparsity levels across different sequence lengths, demonstrating the need for threshold calibration to maintain fixed sparsity across varying contexts.

Through empirical analysis, we find that the optimal threshold follows an inversely proportional relationship with context length LL:

|  |  |  |  |
| --- | --- | --- | --- |
|  | λ=aL\lambda=\frac{a}{L} |  | (3) |

where aa is a model-specific constant. This inverse relationship has theoretical grounding: since attention scores are row-normalized to sum to 1, longer sequences have lower average scores per token, requiring proportionally smaller thresholds. Without calibration, fixed thresholds would cause vastly different sparsity levels across sequence lengths.

Algorithm 2  BLASST Calibration

1:Target sparsity SS, calibration dataset 𝒟\mathcal{D}, context lengths {Lk}k=1K\{L\_{k}\}\_{k=1}^{K}, lambda set Λ\Lambda, tolerance δ\delta

2:Calibration parameter aa

3:Initialize data points 𝒫=∅\mathcal{P}=\emptyset

4:for each context length LkL\_{k} do

5:  Sample sequences of length LkL\_{k} from 𝒟\mathcal{D}

6:  Initialize λbest=None\lambda\_{\text{best}}=\text{None}, min\_gap=∞\text{min\\_gap}=\infty

7:  for each λ∈Λ\lambda\in\Lambda do

8:   s=MeasureSparsity​(λ,Lk)s=\text{MeasureSparsity}(\lambda,L\_{k})

9:   gap=|s−S|\text{gap}=|s-S|

10:   if gap<min\_gap\text{gap}<\text{min\\_gap} then

11:     λbest=λ\lambda\_{\text{best}}=\lambda

12:     min\_gap=gap\text{min\\_gap}=\text{gap}

13:   end if

14:  end for

15:  if min\_gap<δ\text{min\\_gap}<\delta then ⊳\triangleright Only keep if sparsity is close enough

16:   Add (1/Lk,λbest)(1/L\_{k},\lambda\_{\text{best}}) to 𝒫\mathcal{P}

17:  end if

18:end for

19:Fit linear regression: λ=a⋅(1/L)\lambda=a\cdot(1/L) using 𝒫\mathcal{P}

20:return Regression coefficient aa



(a) Normal FlashAttention prefill pipeline schedule.

(b) BLASST prefill pipeline schedule with T0 and T1 both skipping loops 1 and 3.

Figure 3: Prefill pipeline schedules for FlashAttention and BLASST at 50% sparsity across 4 loop iterations (L0-L3). Rows are separated based on warp/warpgroup specializations. Darker and lighter hues correspond to ops for different tile rows (T0/T1). The MMA warp’s BMM1 and BMM2 ops are indicated with B1 and B2. The softmax warpgroups are primarily bottlenecked by exponentiation (EX2), but they also perform the skip check, row sum and softmax scaling (not shown). Mainloop iterations are enclosed by solid lines.

To find the optimal value of aa for a given target sparsity SS, we propose the calibration procedure detailed in Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Calibration for Optimal Sparsity ‣ 3 Methodology"). The process involves empirically finding the best-fitting threshold λbest\lambda\_{\text{best}} for several context lengths {Lk}\{L\_{k}\} that achieves the target sparsity SS (within a tolerance δ\delta). We then perform a linear regression on the transformed data points (1/Lk,λbest)(1/L\_{k},\lambda\_{\text{best}}) to find the slope aa, which defines our calibration function λ​(L)=a/L\lambda(L)=a/L.

More importantly, by targeting fixed sparsity levels, our calibration ensures predictable computational speedup across different context lengths. This is a crucial property for production deployment where consistent performance is required.

### 3.3 Sparsity-Aware Training

While BLASST is primarily designed as a post-training inference optimization, we explore sparsity-aware training as a simple extension to further improve the accuracy-sparsity trade-off. The motivation is straightforward: if models learn to concentrate important information in high-scoring attention blocks during training, they should maintain higher accuracy when those blocks are pruned during inference.

Our method is simple: during fine-tuning, we apply BLASST in the forward pass to skip negligible attention blocks based on the threshold criterion. In the backward pass, skipped blocks naturally receive no gradients since they were not computed in the forward pass. This encourages the model to adapt its attention patterns to be more compatible with sparsity, concentrating important information in blocks that pass the threshold test. This approach requires no architectural changes or auxiliary losses — it is simply training with the same sparse attention that will be used at inference time.

## 4 Kernel Design

The BLASST kernels were designed with two primary goals: (1) minimal changes to existing FlashAttention kernel interfaces and implementation structure, and (2) minimal overhead for block skipping decision logic. Our key insight is to reuse statistics already computed during the standard FlashAttention algorithm — specifically, the local maximum and running maximum values that exist in every thread during online softmax.

Skip Decision Implementation. The decision process (line 7 in Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1.2 Algorithm Design ‣ 3.1 Pruning Attention with Running Maximums ‣ 3 Methodology")) requires only a few additional instructions per block: (1) setting a predicate per thread based on the threshold comparison, (2) issuing a VOTE instruction to determine if all threads within a warp agree to skip, and (3) a single ATOMIC instruction to shared memory issued by one thread per warp to coordinate the block-level decision across the softmax warpgroup. We carefully design the kernel such that the decision-making instructions are hidden behind existing operations, adding negligible latency overhead.

Since prefill and decode phases have fundamentally different performance characteristics, we implement specialized optimizations for each.

(a) Normal FlashAttention decode pipeline schedule.

(b) BLASST decode pipeline schedule when skipping loops 1, 2, and 4.

Figure 4: Decode pipeliene schedules for FlashAttention and BLASST skipping loops 1, 2, and 4. The prologue is not shown, and we focus on the steady state of the first 6 loop iterations (L0-L5). We split out the TMA warp’s pipeline stages to show how multiple TMA loads are issued at once. Loads in Figure [4(b)](#S4.F4.sf2 "In Figure 4 ‣ 4 Kernel Design") finish more quickly because there are fewer simultaneous loads. Arrows indicate scoreboard dependencies from the skip check after BMM1. Note that The MMA warp’s BMM1 and BMM2 ops are indicated with B1 and B2.

### 4.1 Prefill Kernel: Compute-Bound Optimization

Prefill kernels are typically compute-bound, bottlenecked by CUDA core (softmax) and tensor core (matrix multiplication) throughput rather than memory bandwidth. Therefore, our prefill kernel is designed to skip both softmax computation and MMA operations (attention-value multiplication) for pruned blocks.

Figure [3](#S3.F3 "Figure 3 ‣ 3.2 Calibration for Optimal Sparsity ‣ 3 Methodology") illustrates our changes to the pipeline schedule for the BLASST prefill kernel, which is optimized for compute-bound scenarios by overlapping different compute tasks. The pipeline schedules operations across Tensor Cores (math warp/matrix multiplication) and CUDA cores (softmax and correction logic). Figure [3(b)](#S3.F3.sf2 "In Figure 3 ‣ 3.2 Calibration for Optimal Sparsity ‣ 3 Methodology") shows that even as all Q​K⊤QK^{\top} (BMM1) operations are computed, the kernel dynamically skips the compute-heavy softmax and attention-value multiplication (BMM2) for blocks identified as negligible (e.g., loop 1 and loop 3 in Figure [3(b)](#S3.F3.sf2 "In Figure 3 ‣ 3.2 Calibration for Optimal Sparsity ‣ 3 Methodology")). By skipping these compute operations, the kernel frees up execution units, allowing subsequent operations to be scheduled earlier. This compresses the entire schedule, reducing the total runtime from 18 time units in Figure [3(a)](#S3.F3.sf1 "In Figure 3 ‣ 3.2 Calibration for Optimal Sparsity ‣ 3 Methodology") to 14 units in Figure [3(b)](#S3.F3.sf2 "In Figure 3 ‣ 3.2 Calibration for Optimal Sparsity ‣ 3 Methodology").

The Value blocks remain loaded from HBM in the prefill kernel because: (1) memory bandwidth is not the bottleneck, (2) the prefetching pipeline benefits from predictable memory access patterns, and (3) the latency of conditional Value loading would exceed the savings. By focusing on eliminating compute operations, we achieve speedups that scale nearly linearly with sparsity in the compute-bound regime. Our current design prioritizes the common case where prefill is compute-bound on modern GPUs; however, Value loading could be skipped in prefill if future workloads or hardware architectures shift to be memory bandwidth-bound.

Table 1: Performance of BLASST at different sparsity levels across all models and benchmarks. We evaluate on Llama-3.1-8B and Qwen3-8B across three deployment scenarios: prefill-only optimization (long-context tasks: RULER, LongBench), decode-only optimization (reasoning tasks: MATH500, AIME 2024, GPQA), and combined prefill+decode optimization. Results show minimal accuracy degradation even at ∼\sim75% sparsity, with occasional improvements over dense baseline.

| Model | Sparsity | Prefill Phase | | Decode Phase | | | Prefill + Decode Phase | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RULER-32K | LongBench | MATH500 | AIME2024 | GPQA | RULER-32K | LongBench |
| Llama-3.1-8B | Dense | 92.33 | 31.40 | 73.40 | 46.66 | 46.71 | 92.33 | 31.40 |
| ∼\sim50% | 91.81 | 31.80 | 73.71 | 46.15 | 46.31 | 91.79 | 32.40 |
| ∼\sim75% | 91.67 | 31.80 | 73.89 | 46.01 | 45.95 | 91.67 | 31.80 |
| Qwen3-8B | Dense | 91.90 | 33.60 | 95.87 | 75.00 | 61.21 | 91.90 | 33.60 |
| ∼\sim50% | 92.08 | 35.10 | 96.23 | 76.50 | 61.56 | 92.07 | 33.30 |
| ∼\sim75% | 92.11 | 34.40 | 96.07 | 75.33 | 61.51 | 91.74 | 33.10 |

### 4.2 Decode Kernel: Memory-Bound Optimization

Decode kernels are typically memory-bound, bottlenecked by the HBM bandwidth required to fetch the KV cache rather than compute, as attention involves only a single Query against all Keys. Our kernel thus focuses on skipping the memory-intensive load of the Value matrix VjV\_{j} for pruned blocks, directly addressing this HBM bottleneck. This optimization reduces memory traffic in proportion to the sparsity level, while overlapping the threshold and Key operations with the remaining Value loads to achieve a substantial speedup, reflecting the different performance characteristics of decode versus prefill.

Figure [4](#S4.F4 "Figure 4 ‣ 4 Kernel Design") illustrates our changes to the pipeline schedule for the BLASST decode kernel. The long durations to load KK and VV tiles show how the kernel behaves in this memory-bound scenario. By skipping the VV tile load and BMM2 for loops 1, 2, and 4, the GPU can complete outstanding loads from other TMA pipeline stages more quickly. As a result, Figure [4(a)](#S4.F4.sf1 "In Figure 4 ‣ 4 Kernel Design") takes 30 time units to complete all V loads, whereas Figure [4(b)](#S4.F4.sf2 "In Figure 4 ‣ 4 Kernel Design") takes 23 units.

For attention mechanisms like Multi-head Latent Attention (MLA) liu2024deepseek that are more compute-bound even in decode, we additionally skip softmax operations for pruned blocks, providing further speedup beyond memory savings alone.

## 5 Experiments

### 5.1 Experimental Setup

Models. We evaluate BLASST on state-of-the-art language models to demonstrate its effectiveness across different architectures. Our evaluation focuses on two 8B parameter models: Llama-3.1-8B-Instruct and Qwen3-8B-Instruct, both supporting context lengths up to 128K tokens. For long-generation reasoning tasks, we use Llama-3.1-8B-Instruct distilled from DeepSeek-R1 guo2025deepseek, which provides enhanced reasoning capabilities while maintaining compatibility with our sparse attention approach.

Baselines. We compare BLASST against dense attention and SOTA sparse attention methods. For prefill optimization, we compare against MInference jiang2024minference, FlexPrefill lai2025flexprefill, and XAttention xu2025xattention. For decode optimization, we evaluate against Quest tang2024quest, RocketKV behnam2025rocketkv. For each baseline, we adopt its best-performing configuration as reported in its respective paper to ensure fair comparison.

Datasets. We evaluate on two categories of benchmarks: (1) Long-context tasks: RULER hsieh2024ruler (synthetic retrieval and reasoning from 4K-128K tokens) and LongBench v2 bai2024longbench (real-world QA, summarization, and code completion). (2) Reasoning tasks: MATH500 (mathematical problem solving), AIME 2024 (advanced mathematics), GPQA (graduate-level science), and LiveCodeBench (code generation). These reasoning benchmarks test whether sparse attention preserves complex multi-step reasoning capabilities. We use the NVIDIA NeMo-Skills framework111https://github.com/NVIDIA-NeMo/Skills for standardized evaluation of reasoning tasks.

Implementation Details. We implement BLASST as optimized CUDA kernels integrated with the flashinfer framework ye2025flashinfer. For calibration (Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Calibration for Optimal Sparsity ‣ 3 Methodology")), we sample approximately 1000 sequences from the RULER dataset across different context lengths (4K, 8K, 16K, 32K, 64K) to determine the optimal threshold relationship λ=a/L\lambda=a/L for target sparsity levels. For sparsity-aware training, we adopt the curriculum training approach from ProLong gao2024train, applying BLASST during the finetuning phase with a fixed sparsity threshold.

For evaluation, we use different sampling strategies depending on the task type. For long-context benchmarks (RULER and LongBench), we use greedy decoding with temperature =0=0 and perform a single run per example to ensure deterministic and reproducible results. For reasoning tasks that benefit from sampling diversity, we use temperature =0.6=0.6 and top-p =0.95=0.95. Specifically, we generate 10 samples per problem for MATH500, GPQA, and LiveCodeBench, and 20 samples per problem for AIME 2024 due to its greater difficulty. For these reasoning tasks, we report the best-of-N performance where the final answer is selected using majority voting or self-consistency.

Table 2: Prefill phase comparison on Llama-3.1-8B-Instruct across RULER and LongBench. BLASST achieves the best performance among all sparse attention methods, closely matching dense attention while requiring no pre-computation or proxy scores.

| Method | RULER | | | | | | LongBench | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4K | 8K | 16K | 32K | 64K | Average | Easy | Hard | Short | Medium | Long | Overall |
| Dense Attention | 96.16 | 95.07 | 94.80 | 92.33 | 87.69 | 93.21 | 29.7 | 32.5 | 38.3 | 28.8 | 25.0 | 31.4 |
| FlexPrefill | 95.99 | 93.67 | 92.73 | 88.14 | 81.14 | 87.72 | 28.8 | 23.8 | 24.4 | 26.5 | 26.2 | 25.7 |
| MInference | 96.54 | 94.06 | 91.37 | 85.79 | 83.03 | 84.15 | 28.6 | 32.8 | 36.7 | 30.2 | 24.1 | 31.2 |
| XAttention | 96.37 | 94.47 | 94.48 | 91.91 | 85.01 | 92.44 | 29.2 | 31.5 | 38.3 | 26.0 | 26.9 | 30.6 |
| BLASST (∼\sim50%) | 96.17 | 94.70 | 94.61 | 91.81 | 87.06 | 92.87 | 30.7 | 32.5 | 38.3 | 29.8 | 25.0 | 31.8 |




Table 3: Decode phase comparison on Qwen3-8B across diverse reasoning and generation tasks. BLASST matches or exceeds dense baseline on all benchmarks, including mathematical reasoning (MATH500, AIME 2024), graduate-level science (GPQA), and code generation (LiveCodeBench), while maintaining long-context performance (RULER, LongBench).

| Method | RULER-32K | LongBench | MATH500 | AIME 2024 | LiveCodeBench | GPQA | Average |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense Attention | 91.90 | 33.60 | 95.87 | 75.00 | 53.83 | 61.21 | 68.57 |
| Quest | 56.23 | 30.30 | 94.18 | 71.50 | 52.17 | 60.12 | 60.75 |
| RocketKV | 87.89 | 30.60 | 95.88 | 73.54 | 53.10 | 60.50 | 66.91 |
| BLASST ∼\sim50% | 91.55 | 33.90 | 96.23 | 76.50 | 54.15 | 61.51 | 68.97 |

### 5.2 Main Results

Overall Performance. Table [1](#S4.T1 "Table 1 ‣ 4.1 Prefill Kernel: Compute-Bound Optimization ‣ 4 Kernel Design") presents the performance of BLASST at ∼\sim50% and ∼\sim75% sparsity levels across Llama-3.1-8B and Qwen3-8B on diverse benchmarks. Remarkably, BLASST not only maintains accuracy with minimal degradation but occasionally *outperforms* the dense baseline. For instance, on Qwen3-8B, we observe improvements on MATH500 (96.23 vs 95.87) and AIME 2024 (76.50 vs 75.00) at 50% sparsity. This counterintuitive result can be attributed to two factors. First, in long-context tasks where information is inherently sparse, pruning low-attention blocks forces the model to concentrate probability mass on the most relevant tokens, effectively acting as implicit denoising. Second, for long-generation reasoning tasks, some intermediate reasoning steps or tokens may be redundant or even detrimental sui2025stop; by skipping blocks with negligible attention scores, we filter out such distractions, allowing the model to focus on essential reasoning chains. These results suggest that BLASST provides not only computational efficiency but also improves output quality in certain scenarios.

Prefill Phase Comparison. Table [2](#S5.T2 "Table 2 ‣ 5.1 Experimental Setup ‣ 5 Experiments") compares BLASST against state-of-the-art prefill-optimized sparse attention methods on Llama-3.1-8B. Across RULER (4K-64K context lengths) and LongBench, BLASST achieves the best overall performance (92.87 RULER average, 31.8 LongBench) among all sparse methods, closely matching dense attention (93.21, 31.4) while requiring no pre-computation. Notably, BLASST significantly outperforms MInference (84.15 RULER) and FlexPrefill (87.72 RULER), demonstrating the effectiveness of our threshold-based pruning over proxy-based importance estimation.

Decode Phase Comparison. Table [3](#S5.T3 "Table 3 ‣ 5.1 Experimental Setup ‣ 5 Experiments") evaluates BLASST on Qwen3-8B across reasoning-intensive tasks. At ∼\sim50% sparsity, BLASST matches or exceeds dense baseline performance on all benchmarks, while maintaining long-context capabilities. We note that all existing methods employ different optimization strategies and target different deployment scenarios, making direct comparison challenging. We include Quest and RocketKV as reference points to contextualize BLASST’s performance. For instance, RocketKV shows 87.89 RULER and 30.60 LongBench accuracy, illustrating the trade-offs involved in aggressive KV cache compression versus BLASST’s approach of preserving critical attention patterns.

Table 4: BLASST’s prefill and decode speedups on increasing levels of sparsity on B200 (Blackwell) GPU. We vary the threshold (λ\lambda) to demonstrate our performance across a range of sparsities.
Prefill configuration: batch size 148, 1 Q head, 1 KV head, 32K sequence length, 128 head dim. Decode configuration: batch size 148, 32 Q heads, 4 KV heads, 32K sequence length, 128 head dim.

| Blackwell Prefill | | Blackwell Decode | |
| --- | --- | --- | --- |
| Sparsity | Speedup | Sparsity | Speedup |
| 0.00% | 0.99×\times | 0.00% | 0.98×\times |
| 23.24% | 1.07×\times | 36.91% | 1.18×\times |
| 32.78% | 1.13×\times | 46.68% | 1.25×\times |
| 42.26% | 1.18×\times | 61.23% | 1.34×\times |
| 50.67% | 1.24×\times | 73.15% | 1.48×\times |
| 63.96% | 1.34×\times | 82.62% | 1.64×\times |
| 73.47% | 1.41×\times | 87.01% | 1.71×\times |
| 80.35% | 1.49×\times | 91.99% | 1.79×\times |




Figure 5: Speedup of BLASST prefill on Hopper GPU (H200)

### 5.3 GPU Kernel Performance

We implement and benchmark highly optimized kernels for both Blackwell (B200) and Hopper (H200) GPU architectures, demonstrating that BLASST achieves substantial real-world speedups. Table [4](#S5.T4 "Table 4 ‣ 5.2 Main Results ‣ 5 Experiments") and Figure [5](#S5.F5 "Figure 5 ‣ 5.2 Main Results ‣ 5 Experiments") show performance scaling across increasing sparsity levels for both prefill and decode phases. All speedups are measured against FlashAttention-3 BF16 baselines.

Key Results. At near-lossless accuracy (∼\sim50% sparsity), we achieve approximately 1.24×\times speedup for prefill and 1.23×\times speedup for decode on Blackwell. At higher sparsity (∼\sim70%), the prefill and decode speedup increases to 1.40×\times. On Hopper, prefill achieves up to 1.62×\times speedup at 74.7% sparsity. These speedups scale predictably with sparsity: higher sparsity yields greater performance gains, allowing users to choose their preferred accuracy-efficiency trade-off.

Importantly, we observe no significant performance degradation at 0% sparsity (0.99-1.03×\times baseline), ensuring BLASST adds minimal overhead when sparsity is low.

### 5.4 Calibration Results

A key motivation for our calibration approach is that fixed thresholds produce inconsistent sparsity across different context lengths, making deployment unreliable. Table [5](#S5.T5 "Table 5 ‣ 5.4 Calibration Results ‣ 5 Experiments") demonstrates the effectiveness of our calibration method across varying sequence lengths. For a target sparsity of 50%, the fixed threshold approach produces highly unstable sparsity ranging from 23% at 4K to 75% at 64K, making it impractical for production deployment. In contrast, our calibrated λ=a/L\lambda=a/L approach maintains sparsity within a tight range with an average error of only 1.2% from the target. Similar improvements are observed at 70% target sparsity. These results confirm that our calibration enables reliable, predictable sparsity control across diverse sequence lengths without manual tuning.

Table 5: Sparsity stability across context lengths: calibrated vs. fixed threshold on Llama-3.1-8B. Our calibration method maintains consistent sparsity levels across different context lengths, while fixed thresholds produce high variance. Values in parentheses indicate deviation of achieved sparsity from the target.

| Method | 4K | 8K | 16K | 32K | 64K |
| --- | --- | --- | --- | --- | --- |
| Target Sparsity: 50% | | | | | |
| Fixed λ=1​e−3\lambda=1e-3 | 23.09 | 37.92 | 52.38 | 65.72 | 74.63 |
|  | (-26.91) | (-12.08) | (+2.38) | (+15.72) | (+24.63) |
| Calibrated λ=a/L\lambda=a/L | 54.20 | 49.70 | 52.20 | 46.96 | 48.75 |
|  | (+4.20) | (-0.30) | (+2.20) | (-3.04) | (-1.25) |
| Target Sparsity: 70% | | | | | |
| Fixed λ=3​e−3\lambda=3e-3 | 42.35 | 57.54 | 69.83 | 79.36 | 84.63 |
|  | (-27.65) | (-12.46) | (-0.17) | (+9.36) | (+14.63) |
| Calibrated λ=a/L\lambda=a/L | 67.99 | 74.65 | 73.64 | 72.54 | 74.63 |
|  | (-2.01) | (+4.65) | (+3.64) | (+2.54) | (+4.63) |




Figure 6: Sparsity-aware training pushes the accuracy-sparsity frontier. Models fine-tuned with BLASST active during training maintain higher accuracy at aggressive sparsity levels compared to post-training sparsity application. By training with sparse attention, models learn to concentrate information in high-scoring blocks, making them more robust to pruning.

### 5.5 Sparsity-Aware Training Results

Figure [6](#S5.F6 "Figure 6 ‣ 5.4 Calibration Results ‣ 5 Experiments") demonstrates that sparsity-aware training improves the accuracy-sparsity trade-off on RULER benchmarks. At low sparsity levels, sparse-trained models even slightly outperform the dense baseline, suggesting the model learns more robust attention patterns. In the target sparsity range of 50%-75%, sparse-trained models achieve substantially better accuracy than applying sparsity post-training, reducing accuracy degradation by up to 1.7×\times. These results confirm that models can be trained to concentrate information in high-scoring attention blocks, making them inherently more compatible with sparse attention patterns and pushing the Pareto frontier of efficient attention.

### 5.6 Ablation Studies

Sparsity Distribution Analysis. Figure [7](#S5.F7 "Figure 7 ‣ 5.6 Ablation Studies ‣ 5 Experiments") illustrates how sparsity varies across layers and attention heads, revealing the natural distribution of attention importance and patterns in the model. We observe substantial heterogeneity: different layers exhibit different sparsity levels, and individual heads within each layer also show significant variance. Crucially, BLASST naturally incorporates this heterogeneity without requiring explicit mechanisms like top-k selection or head pruning—by applying the same threshold across all layers and heads, our method automatically adapts to each layer’s and head’s natural attention distribution, pruning more aggressively where attention is naturally more concentrated and preserving more blocks where attention is more diffuse.

Figure 7: Sparsity distribution across layers and heads for Llama-8B on 8K context. Taken from NIAH benchmark sample with threshold λ=0.03\lambda=0.03. Substantial head-level and layer-level variance motivates adaptive thresholding strategies.

Combination with Other Sparsity Methods. Table [6](#S5.T6 "Table 6 ‣ 5.6 Ablation Studies ‣ 5 Experiments") explores combining BLASST with other attention sparsity techniques. We find that BLASST can be effectively composed with both prefill-optimized methods (XAttention) and KV cache compression methods (RocketKV). When combining XAttention (prefill) with BLASST (decode), accuracy degradation remains minimal, demonstrating that the methods are largely orthogonal. Similarly, combining BLASST (prefill) with RocketKV maintains strong performance. These results show that BLASST provides a flexible building block for end-to-end optimization in existing sparse attention pipelines.

Table 6: Performance of combining BLASST with other sparsity methods on Qwen 8b. BLASST can be effectively composed with both prefill-optimized methods (XAttention) and KV cache compression methods (RocketKV), providing flexible deployment options. Numbers in parentheses show change from dense baseline.

|  |  |  |  |
| --- | --- | --- | --- |
| Prefill Method | Decode Method | RULER-16K | LongBench-16K |
| Dense Attention | Dense Attention | 93.22 | 29.4 |
| XAttention | Dense Attention | 92.99 (-0.23) | 29.1 (-0.3) |
| XAttention | BLASST | 92.89 (-0.33) | 28.8 (-0.6) |
| Dense Attention | RocketKV | 92.72 (-0.50) | 30.0 (+0.6) |
| BLASST | RocketKV | 92.60 (-0.62) | 29.4 (-0.0) |




Table 7: Performance on very long sequences with RepoQA benchmark. We evaluate BLASST on code repository understanding tasks at 16K and 200K context lengths, showing sparsity in prefill (P) and decode (D) phases.

| Context | Attention Mode | Sparsity (P) | Sparsity (D) | Accuracy |
| --- | --- | --- | --- | --- |
| Qwen3-Coder-30B-A3B-Instruct, 16K Context | | | | |
| 16K | Full (Dense) | 0% | 0% | 0.897 |
| 16K | BLASST Prefill | 64.1% | 0% | 0.904 |
| 16K | BLASST Prefill+Decode | 64.1% | 48.4% | 0.882 |
| Qwen3-Coder-30B-A3B-Instruct, 200K Context | | | | |
| 200K | Full (Dense) | 0% | 0% | 0.850 |
| 200K | BLASST Prefill | 57.5% | 0% | 0.841 |
| 200K | BLASST Prefill+Decode | 57.5% | 40.8% | 0.838 |

Very Long Sequence Lengths. We evaluate BLASST on extremely long sequences using the RepoQA benchmark liu2024repoqa. Table [7](#S5.T7 "Table 7 ‣ 5.6 Ablation Studies ‣ 5 Experiments") presents results on Qwen3-Coder-30B at 16K and 200K context lengths. At 200K tokens, BLASST achieves high prefill sparsity (∼\sim58%) with minimal accuracy drop, and applying sparsity to both prefill and decode phases provides additional computational savings with negligible incremental cost. Notably, longer contexts exhibit higher natural sparsity, making our method increasingly effective for extreme-length scenarios where dense attention becomes impractical.

Tile Row Reordering. We investigated whether permuting the tile-row processing order could improve pruning accuracy. The motivation comes from the local window phenomenon observed in StreamingLLM xiao2023efficient, where recent tokens (local window) often receive high attention scores alongside attention sink tokens at the beginning. By processing tiles containing the local window first, the running maximum mim\_{i} can be quickly populated with these high-scoring tokens, establishing a better proxy for the global maximum earlier in the computation. This would enable more accurate skip decisions for subsequent blocks. Importantly, BLASST supports such reordering flexibility at the kernel scheduling level with negligible overhead.

Figure [8](#S5.F8 "Figure 8 ‣ 5.6 Ablation Studies ‣ 5 Experiments") compares standard sequential processing against reordered processing on VT and FWE tasks. The results show dataset-dependent behavior: reordering yields similar performance on VT but provides noticeable improvements on FWE. This suggests that the effectiveness of reordering largely depends on the specific attention patterns of each dataset. Nevertheless, this demonstrates a valuable property of BLASST: the algorithm is robust to different processing orders and can accommodate various optimization strategies. The flexibility to support tile reordering shows the potential for dataset-specific optimizations without requiring fundamental algorithmic changes.

Figure 8: Effect of tile row reordering on the accuracy-sparsity trade-off for Llama 3.1 8B (ctx=8192). We compare *Standard Cummax* (processing tiles sequentially) with *Reordered Cummax* (processing tiles in reverse order). The plots for both VT and FWE benchmarks show that reordering has a negligible impact on model accuracy at a given sparsity level.




Figure 9: Accuracy-sparsity trade-off at high sparsity levels on RULER-16K for Qwen3-8B. BLASST shows more stable degradation compared to XAttention, maintaining better accuracy at aggressive sparsity settings. This shows the effectiveness of using actual softmax statistics versus proxy-based importance scores.

Extreme Sparsity Analysis. Figure [9](#S5.F9 "Figure 9 ‣ 5.6 Ablation Studies ‣ 5 Experiments") shows BLASST’s behavior at higher sparsity levels (70-90%) on RULER benchmarks. Compared to XAttention, BLASST demonstrates more stable accuracy degradation across increasing sparsity levels. While XAttention shows sharper accuracy drops at high sparsity, BLASST’s threshold-based pruning using actual softmax statistics (rather than proxy scores) enables more graceful degradation. This stability makes BLASST more suitable for aggressive sparsity settings where computational efficiency is critical.

## 6 Conclusion

We presented BLASST, a simple yet effective sparse attention method that dynamically prunes attention computations by reusing online softmax statistics, requiring no pre-computation or proxy scores. Achieving over 50% sparsity with minimal accuracy degradation and up to 1.6×\times speedup on modern GPUs, BLASST makes long-context inference significantly more practical. Our automated calibration and sparsity-aware training further enhance its robustness and flexibility, providing a practical foundation for efficient long-context transformers.

Looking forward, we believe the combination of hardware-aware sparse patterns, learned sparsity through training, and adaptive hybrid methods will be key to unlocking the full potential of future agentic AI systems.
