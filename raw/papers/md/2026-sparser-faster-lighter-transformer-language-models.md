---
arxiv: '2603.23198'
authors:
- Edoardo Cetin
- Stefano Peluchetti
- Emilio Castillo
- Akira Naruse
- Mana Murakami
- Llion Jones
parser: ar5iv
retrieved: '2026-05-09'
source: paper
title: Sparser, Faster, Lighter Transformer Language Models
url: https://arxiv.org/abs/2603.23198
year: 2026
---

[2603.23198] Sparser, Faster, Lighter Transformer Language Models














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



\correspondingauthor

Edoardo Cetin (edo@sakana.ai), Emilio Castillo (ecastillo@nvidia.com)

# Sparser, Faster, Lighter Transformer Language Models

Edoardo Cetin\*
Sakana AI

Stefano Peluchetti\*
Sakana AI

Emilio Castillo\*
NVIDIA
  
\*Core contributors

Akira Naruse
NVIDIA
  
\*Core contributors

Mana Murakami
NVIDIA
  
\*Core contributors

Llion Jones
Sakana AI

###### Abstract

Scaling autoregressive large language models (LLMs) has driven unprecedented progress but comes with vast computational costs. In this work, we tackle these costs by leveraging unstructured sparsity within an LLM’s feedforward layers, the components accounting for most of the model parameters and execution FLOPs. To achieve this, we introduce a new sparse packing format and a set of CUDA kernels designed to seamlessly integrate with the optimized execution pipelines of modern GPUs, enabling efficient sparse computation during LLM inference and training. To substantiate our gains, we provide a quantitative study of LLM sparsity, demonstrating that simple L1 regularization can induce over 99% sparsity with negligible impact on downstream performance. When paired with our kernels, we show that these sparsity levels translate into substantial throughput, energy efficiency, and memory usage benefits that increase with model scale. We will release all code and kernels under an open-source license to promote adoption and accelerate research toward establishing sparsity as a practical axis for improving the efficiency and scalability of modern foundation models.

![[Uncaptioned image]](/html/2603.23198/assets/x1.png)  Code: [github.com/SakanaAI/sparser-faster-llms](https://github.com/SakanaAI/sparser-faster-llms)

## 1 Introduction

![Refer to caption](/html/2603.23198/assets/x2.png)


Figure 1: Comparison of ELL with our new TwELL and Hybrid sparse formats designed for LLM inference and training, respectively.

Large Language Models (LLMs) have revolutionized natural language processing, demonstrating unprecedented capabilities in text generation, reasoning, and knowledge retrieval (openai2023gpt4; gemini). The core component driving these advancements has been massive computational investments into scaling the seminal Transformer architecture, with current LLMs reaching hundreds of billions of parameters vaswani2017attention; gpt2; gpt3. However, with increasingly larger models requiring vast computational resources for both inference and training, there is a growing need for fundamental efficiency improvements to ensure the present and future sustainability of the field (schwartz2020green; luccioni2023bloom).

One seminal avenue for improving the efficiency of machine learning models is sparsity (lecun1989optimal; han2015learning; hoefler2021sparsity). For modern overparameterized LLMs, recent investigations have even documented that sparsity arises naturally in their feed-forward layers, with only a small fraction of hidden neurons activated for any given token (zhang-etal-2022-moefication; li2023lazyneuronphenomenonemergence). Thus, with feed-forward computation accounting for over two-thirds of the parameters and over 80% of the total FLOPs in larger models (llms-flops-params), sparsity seemingly offers a natural opportunity for concrete computational savings.

However, a frustrating paradox has blocked progress: despite performing far less theoretical computation, official kernels implementing sparse operations can often run slower than dense operations on modern GPUs. The culprit is a fundamental mismatch between unstructured sparsity and GPU architectures, whose hardware and software stacks have been heavily optimized for dense computation patterns (BLAS-library; nvidia-cublas; nvidia-cublasdx; nvidia-cutlass). In contrast, heterogeneous workloads together with the overheads from materializing and managing sparse indices have been critical challenges preventing generalized computational savings. Due to these challenges, previous attempts to realize efficiency gains have relied on considerable deviations from modern training recipes and have yet to see practical adoption (dejavu\_contectual\_sparsity\_related; q\_sparse\_top\_k\_related).

In this work, we introduce new kernels designed for modern NVIDIA GPUs to bridge this gap and leverage unstructured sparsity to deliver substantial speedups while reducing memory requirements and energy consumption during both LLM inference and training. Our kernels build on Tile-wise ELLPACK (TwELL), a new packing format for sparse data that can be naturally materialized in the epilogue of highly-optimized matrix multiplication kernels, removing a canonical bottleneck of prior packing schemes. Starting from TwELL, our inference kernels fuse multiple matrix-multiplications into a single optimized pipeline that minimizes computation, while our training kernels further reduce the sparse representation to a hybrid format that trivializes storage costs of intermediate activations.

To substantiate our gains, we provide a quantitative study of LLM sparsity across model scales, demonstrating that mild levels of L1 regularization can achieve over 99% sparsity with negligible impact on downstream performance. Through our new kernels, we show these sparsity levels translate into increasing benefits with larger parameter counts in terms of processing throughput, energy savings, and memory requirements – delivering up to 20.5% and 21.9% speedups in forward execution and training for models with billions of parameters. We analyze how these benefits specifically come from the computational unevenness across network layers and natural language data, which can be inherently leveraged in sparse models. By providing a clear demonstration of its practical benefits, we hope this work will help establish sparsity as a new axis for improving the scalability and performance of modern foundation models.

In summary, our main contributions are threefold:

1. 1.

   We introduce and share new CUDA kernels for inference and training, with several key innovations to make sparse LLMs cheaper, faster, and lighter on modern GPUs.
2. 2.

   We provide a quantitative analysis showing that high levels of unstructured sparsity can be achieved using mild L1 regularization with negligible compromises on performance.
3. 3.

   We demonstrate and analyze how our kernels leverage such sparsity with substantial and increasing benefits at larger scales for LLMs with billions of parameters.

## 2 Large Language Models, Feedforward Blocks, and Sparsity

While the original transformer used a simple 2-layer feedforward block, this module has seen considerable evolution since its conception (vaswani2017attention). The most recent architectures have largely converged to a 3-layer gated design that has consistently proven empirical superiority when evaluated at large scale (shazeer2020glu). While in this work we release kernels for both the original and gated blocks, we focus our main text on the newer design and defer to Appendix [C](#A3 "Appendix C Parameter Studies and Ablations ‣ Sparser, Faster, Lighter Transformer Language Models") for further discussions, results, and comparisons with the older variant.

### 2.1 Feed-forward Modules as Sparse Knowledge Stores

A modern gated feed-forward block (shazeer2020glu) is parameterized by three weight matrices Wg∈ℝK×NW\_{g}\in\mathbb{R}^{K\times N}, Wu∈ℝK×NW\_{u}\in\mathbb{R}^{K\times N}, and Wd∈ℝN×KW\_{d}\in\mathbb{R}^{N\times K} representing the gate, up, and down projection matrices, respectively. In our notation, we use MM to denote the feed-forward block’s effective batch size over all batched sequences and positions, KK to denote its input/output dimensions, and NN to denote its hidden expanded dimension. The gate and up projection matrices both process the block’s input batch x∈ℝM×Kx\in\mathbb{R}^{M\times K} and produce the up and gate activations hgh\_{g} and hu∈ℝM×Nh\_{u}\in\mathbb{R}^{M\times N}, where the symmetry between the two is broken with a non-linear activation function σ\sigma.
These projections are then combined with elementwise multiplication into a unified hidden representation h∈ℝM×Nh\in\mathbb{R}^{M\times N}, before being projected back to their original dimensionality using the down projection weights WdW\_{d}, to compute the block’s outputs y∈ℝM×Ky\in\mathbb{R}^{M\times K}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | hu=x​Wu,hg=σ​(x​Wg),h=hu⊙hg,y=h​Wd.h\_{u}=xW\_{u},\quad h\_{g}=\sigma(xW\_{g}),\quad h=h\_{u}\odot h\_{g},\quad y=hW\_{d}. |  | (1) |

Since the hidden dimension NN is typically much larger than KK, feed-forward blocks can often account for most of the model’s parameters and FLOPs. We note that a common conceptualization of these architectural components is that of a *dynamic key-value memory* (geva-etal-2021-transformer; dai-etal-2022-knowledge). In this mental model, the inner products between xx and the columns of WgW\_{g} and WuW\_{u} induce *keys* hh, while the rows of WdW\_{d} are seen as *values* acting as memory slots that can be dynamically retrieved based on the input.

### 2.2 Simple Ingredients for Training Sparse LLMs

We employ a simple recipe to induce varying levels of sparsity in the feed-forward activations, making minimal deviations from established architectures and training objectives. First, we use the ReLU as the activation function of choice following the gate projections. Second, we add a simple L1 loss to the standard cross-entropy with a tunable coefficient L1L\_{1} to promote sparsity across the model’s LL layers:

|  |  |  |  |
| --- | --- | --- | --- |
|  | L1×1L​∑l=1L1M​N​∑m=1M∑n=1N|hl​[m,n]|.L\_{1}\times\frac{1}{L}\sum\_{l=1}^{L}\frac{1}{MN}\sum\_{m=1}^{M}\sum\_{n=1}^{N}\lvert h^{l}[m,n]\rvert. |  | (2) |

We note that many recent LLM architectures have deviated from using ReLUs in favor of smoother activation functions such as SiLU, with minor but consistent benefits (shazeer2020glu; llama2; qwen2). In Appendix [C](#A3 "Appendix C Parameter Studies and Ablations ‣ Sparser, Faster, Lighter Transformer Language Models"), we provide direct empirical comparisons between these choices and also refer to orthogonal studies in the recent literature showing that domain-specific performance differences can be bridged with targeted training techniques (mirzadeh2023relu\_apple\_finetune; lomeli2025stochasticactivations).

## 3 Making Sparse LLMs Fast

We introduce new CUDA kernels for inference and training that leverage unstructured sparsity to efficiently rework the computation in the feed-forward blocks of an LLM. The algorithms underlying our kernels build on TwELL, a new sparse format specifically designed for seamless kernel fusion to realize the inherent throughput and memory benefits of sparsity with minimal overheads.
In this section, we describe the core components and advantages of our new kernels with algorithmic descriptions that summarize their logic at the level of individual cooperative thread arrays (CTAs). We refer to Appendix [A](#A1 "Appendix A Kernels Implementation Details ‣ Sparser, Faster, Lighter Transformer Language Models") for code listings and more detailed design discussions of the thread-level CUDA implementations for H100 GPUs.

Algorithm 1  Algorithmic description of gate projection with our matmul kernel with TwELL storage

  

1: Parameters: Tile sizes Tn,TmT\_{n},T\_{m}, compression ratio C

2: Input: Dense x∈ℝM×Kx\!\in\!\mathbb{R}^{M\times K}, Wg∈ℝK×NW\_{g}\!\in\!\mathbb{R}^{K\times N},

3: Output: Sparse hv∈ℝM×N/Ch\_{v}\!\in\!\mathbb{R}^{M\times N/C},
hI∈ℕM×N/Ch\_{I}\!\in\!\mathbb{N}^{M\times N/C}, hn​z∈ℕM×NTh\_{nz}\!\in\!\mathbb{N}^{M\times N\_{T}}

4: for all tiles starting at (m0,n0)(m\_{0},n\_{0}) in parallel across CTAs do

5:  S←x[m0:m0+Tm,:]Wg[:,n0:n0+Tn]S\leftarrow x[m\_{0}{:}m\_{0}{+}T\_{m},:]\;W\_{g}[:,n\_{0}{:}n\_{0}{+}T\_{n}]

6:  for r←0r\leftarrow 0 …Tm−1T\_{m}{-}1 do

7:  m←m0+rm\leftarrow m\_{0}+r {global row index}

8:  z←0z\leftarrow 0 {running count of non-zeros in tile}

9:  for c←0c\leftarrow 0 …Tn−1T\_{n}{-}1 do

10:    if (S​[r,c]>0)(S[r,c]>0) then

11:    n←n0/C+zn\leftarrow n\_{0}/C+z {global TwELL column index}

12:    hI​[m,n]←n0+ch\_{I}[m,\,n]\leftarrow n\_{0}+c {store non-zero index}

13:    hv​[m,n]←S​[r,c]h\_{v}[m,\,n]\leftarrow S[r,c] {store non-zero value}

14:    z←z+1z\leftarrow z+1 {increment non-zero count}

15:    end if

16:  end for

17:  hn​z​[m,n0/Tn]←zh\_{nz}[m,\,n\_{0}/T\_{n}]\leftarrow z {store final count of non-zeros}

18:  end for

19: end for




Algorithm 2  Algorithmic description of fused up and down projections from gate activations in the TwELL format

  

1: Parameters: Tile size TnT\_{n}, compression ratio C

2: Input: Sparse hv∈ℝM×N/Ch\_{v}\!\in\!\mathbb{R}^{M\times N/C},
hI∈ℕM×N/Ch\_{I}\!\in\!\mathbb{N}^{M\times N/C}, hn​z∈ℕM×NTh\_{nz}\!\in\!\mathbb{N}^{M\times N\_{T}}; dense x∈ℝM×Kx\!\in\!\mathbb{R}^{M\times K}, Wu∈ℝK×NW\_{u}\!\in\!\mathbb{R}^{K\times N}, Wd∈ℝN×KW\_{d}\!\in\!\mathbb{R}^{N\times K}

3: Output: Dense y∈ℝM×Ky\!\in\!\mathbb{R}^{M\times K}

4: for all m∈π(0..M−1)m\in\pi(0..M{-}1) in parallel across CTAs do

5:  xm←x​[m,:]x\_{m}\leftarrow x[m,:]; ym←0y\_{m}\leftarrow 0

6:  for t←0t\leftarrow 0 …NT−1N\_{T}-1 do

7:  z←hn​z​[m,t]z\leftarrow h\_{nz}[m,t]

8:  for c←0c\leftarrow 0 …z−1z-1 do

9:    n←hI​[m,t×Tn/C+c]n\leftarrow h\_{I}[m,\,t\times T\_{n}/C{+}c] {non-zero column index}

10:    wu=Wu​[:,n]w\_{u}=W\_{u}[:,n] {nn-th column of WuW\_{u}}

11:    u←(xm⋅wu)u\leftarrow(x\_{m}\cdot w\_{u}) {sparse hu​[m,n]h\_{u}[m,n] element}

12:    wd←Wd​[n,:]w\_{d}\leftarrow W\_{d}[n,:] {nn-th row of WdW\_{d}}

13:    ym←ym+(hv​[m,t×Tn/C+c]×u)​wdy\_{m}\leftarrow y\_{m}+(h\_{v}[m,\,t\times T\_{n}/C+c]\times u)\;w\_{d}

14:  end for

15:  end for

16:  y​[m,:]←ymy[m,:]\leftarrow y\_{m}

17: end for

### 3.1 Sparse Formats and Kernels

The ELLPACK format (ELL) is considered the state-of-the-art for fast and efficient sparse matmuls (ellpack\_original). This format was leveraged in some of the earliest GPU implementations of sparse algebra (first\_sparse\_gpu\_impl), with more recent work focused on developing packing and sorting variants for better performance (sell\_c\_standard\_on\_gpus\_sem; sell\_c\_standard\_on\_gpus\_impl). As shown in part a. of Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Sparser, Faster, Lighter Transformer Language Models"), an M×NM\times N matrix hh in the ELL format is stored as two padded matrices hvh\_{v} and hIh\_{I} of size M×Nn​zM\times N\_{nz} with the non-zero values of xx and their column indices packed at the beginning of each row. This format prioritizes downstream usability over storage, padding the rows up to the maximum number of nonzero elements Nn​zN\_{nz} for efficient retrieval.

The main logic in most matmul kernels to perform y=h​Wy=hW with ELL, is to launch different parallel accumulations for each row m=0,…,M−1m=0,\dots,M-1 of the sparse matrix hh using a set number of threads. In each accumulation, the kernel iterates for n=0,…,Nn​z−1n=0,...,N\_{nz}-1 times, loading each column index i=hI​[i,j]i=h\_{I}[i,j] and value v=hv​[i,j]v=h\_{v}[i,j] of hh, and multiplying it with the K−K-dimensional row of the dense weight W​[i,:]W[i,:]. The key advantage of this format is that only a fraction of the weight columns and input values need to be processed, skipping the remaining zeros. To further reduce data access and computation, some later extensions like ELLPACK-R (ell\_nnzs\_ellpack\_r) also store the number of non-zeros in each row in a separate vector hn​zh\_{nz}.

### 3.2 TwELL, a Sparse Data Format for Kernel Fusion

An effective predominant design for modern kernel pipelines is to maximize operator fusion and avoid unnecessary global memory accesses in order to best leverage the high compute throughput of modern NVIDIA GPUs. To this end, in a gated feed-forward block where sparsity patterns are determined by the gate activations hgh\_{g}, prior sparse formats such as ELL suffer a major drawback. In essence, representing hgh\_{g} with ELL requires first accessing all elements in every row to count, compare, and align the non-zero values and indices. However, existing matmul kernels for dense inputs rely on parallelizing computation across small 2D tiles Tm×TnT\_{m}\times T\_{n} of the outputs, computed independently in separate CTAs. Thus, obtaining the gate activations directly in the ELL format from the non-sparse inputs cannot be done in the same kernel of hg=ReLU⁡(x​W)h\_{g}=\operatorname{ReLU}(xW) without introducing expensive synchronization among different CTAs. In contrast, launching a separate kernel to do the conversion inherently introduces non-trivial overheads that concretely limit attainable throughput gains of the whole computation.

To address these limitations, we introduce Tile-wise ELLPACK (TwELL). As illustrated in part b. of Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Sparser, Faster, Lighter Transformer Language Models"), rather than focusing on whole rows, TwELL divides the columns of hgh\_{g} in groups of horizontal 1D tiles of size TT. Within each group of columns, TwELL stores the non-zero values present and their indices in a local ELL-based packing format, with the data of each row aligned at the beginning of each horizontal tile. This results in two matrices containing locally aligned values hv∈RM×N/Ch\_{v}\in R^{M\times N/C} and indices hI∈RM×N/Ch\_{I}\in R^{M\times N/C}, where CC is a compression factor set such that T/CT/C is higher than the maximum number of non-zeros in any tile to avoid storage overflow. In our implementation of TwELL, we also store an additional matrix with the number of non-zero elements hn​z∈RM×NTh\_{nz}\in R^{M\times N\_{T}} to facilitate further computations, with as many columns as total tiles NT=⌈N/T⌉N\_{T}=\lceil N/T\rceil. While inherently less expensive to derive, the main advantage of TwELL over ELL is actually ease of materialization following a modern tiled matmul: by setting the horizontal tiling dimensions to match, T=TnT=T\_{n}, the TwELL format can be recovered in the same kernel performing hg=ReLU⁡(x​W)h\_{g}=\operatorname{ReLU}(xW) before storing the outputs to DRAM. Fusing the two operations removes the requirement of performing additional kernel spawns, memory reads, or synchronization steps, leading to a natural integration into existing LLM pipelines.

### 3.3 Kernels for TwELL Construction and Fast, Fused Inference

In Algorithm [1](#alg1 "Algorithm 1 ‣ 3 Making Sparse LLMs Fast ‣ Sparser, Faster, Lighter Transformer Language Models"), we provide pseudocode to summarize the logic of our CUDA matmul kernel storing the sparse outputs in the TwELL format (lines 6-18). Given the output distribution patterns of tensor core operations, we obtain the memory addresses to store the packed non-zero values hvh\_{v} and their indices hIh\_{I} by keeping a local non-zero count that only requires warp-level synchronization. While not an inherent requirement of TwELL, storing the number of non-zeros in each tile, hn​zh\_{nz}, allows us to forego the overhead from initializing hIh\_{I} with any “padding” value and from the additional control logic of checking validity in future usages. While omitted from Algorithm [1](#alg1 "Algorithm 1 ‣ 3 Making Sparse LLMs Fast ‣ Sparser, Faster, Lighter Transformer Language Models"), we leverage fast asynchronous TMA reads and writes by first caching the dense inputs and sparse TwELL outputs to shared memory. We also pipeline computation and global memory accesses with a persistent cooperative design similar to the one in CUTLASS (nvidia-cutlass).

For inference, we introduce a single additional kernel to perform the rest of the computation in the feedforward block, leveraging the gate activations stored in the TwELL format to efficiently fuse the up and down projections together. This kernel, summarized in Algorithm [2](#alg2 "Algorithm 2 ‣ 3 Making Sparse LLMs Fast ‣ Sparser, Faster, Lighter Transformer Language Models"), is launched on a grid made of single warp CTAs each processing a different row mm of the input activations xx. Minimizing the size of each CTA serves the purpose of maximizing concurrency and L2-hits across the grid, as non-zero activations tend to have high correlation within input sequences. The fused matmuls are executed by traversing over the sparsified activations with two nested for loops: the first one statically-unrolled over the number of column tiles (line 6) and the second one dynamically iterating over the corresponding number of non-zeros in each tile (line 8). For each non-zero activation at index nn, the CTA collectively loads the nt​hn^{th} row of WuW\_{u} and column of WdW\_{d} to perform a dot product, followed by a scalar-vector product and accumulating its results (lines 9-13) – corresponding to the following computation:

|  |  |  |  |
| --- | --- | --- | --- |
|  | y​[m,:]=∑t=0NT−1∑c=0hn​z​[m,t]−1hv​[m,t×Tn/C+c]⏟hv​ non-zero value​(x​[m,:]⋅Wu​[:,n])⏟hu​ element​Wd​[n,:]⏟Wd​ row, where ​n=hI​[m,t×Tn/C+c]⏟hI​ non-zero index.y[m,:]\!=\!\sum\_{t=0}^{\scriptscriptstyle N\_{T}\!-\!1}\sum\_{c=0}^{\scriptscriptstyle h\_{nz}[m,t]\!-\!1}\underbrace{h\_{v}[m,\,t\times T\_{n}/C\!+\!c]}\_{\small h\_{v}\text{ non-zero value}}\underbrace{\left(x[m,:]\cdot W\_{u}[:,n]\right)}\_{\small h\_{u}\text{ element}}\underbrace{W\_{d}[n,:]}\_{\small W\_{d}\text{ row}},\text{ where }n=\underbrace{h\_{I}[m,\,t\times T\_{n}/C\!+\!c]}\_{\small h\_{I}\text{ non-zero index}}. |  | (3) |

Implicitly materializing the huh\_{u} values only inside the kernel serves to further reduce DRAM access to maximize throughput. Together, the kernels in our inference pipeline align the core principles of tiling and operator fusion into a single execution flow, harnessing the computational advantages of sparsity while minimizing its canonical overheads.

### 3.4 Hybrid Conversion for Efficient Storage

During training, memory becomes a key bottleneck for throughput as large intermediate activations and optimizer states are needed for backpropagation. Here, sparsity provides a natural opportunity to tackle these bottlenecks by trivializing intermediate storage costs and accelerating gradient computations. However, directly using TwELL with a high compression ratio or other ELL-based formats for this purpose inherently relies on the maximum number of non-zeros Nn​zN\_{nz} to be known ahead of time and strictly small. However, as we will illustrate in Section [4](#S4.2 "4 Experimental Results ‣ Sparser, Faster, Lighter Transformer Language Models"), we find that these conditions are practically never met during LLM training as sparsity patterns exhibit significant non-uniformity across different tokens, with the maximum number of non-zeros often orders of magnitude larger than the average.

We overcome these limitations by first converting the TwELL activations to a hybrid sparse format and introducing a new set of custom kernels designed specifically for memory-efficient training. As illustrated part c. of Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Sparser, Faster, Lighter Transformer Language Models"), our format dynamically partitions and stores the rows of hgh\_{g} either in an aggressively compact ELL matrix hgs∈RMs×Nn​z^h^{s}\_{g}\in R^{M^{s}\times N\_{\hat{nz}}} or a dense backup hgd∈RMd×Nh^{d}\_{g}\in R^{M^{d}\times N}. The partitioning logic simply routes the rows of hgh\_{g} based on their non-zero counts, which are cheaply computed from the locally aligned TwELL tiles. Our hybrid format also maintains a lightweight array of column indices hI∈RMs×Nn​z^h\_{I}\in R^{M^{s}\times N\_{\hat{nz}}} matching the size of the sparsified ELL matrix, and a simple binary vector indicating the storage location of each row hb∈RMh\_{b}\in R^{M}. In practice, we find that we can set Nn​z^N\_{\hat{nz}} over an order-of-magnitude lower than NN with minimal overflow into hgdh^{d}\_{g}, avoiding stringent ELL requirements while still trivializing memory and computation during the rest of the training step.

### 3.5 Kernels for Lightweight Efficient Training

After materializing hgh\_{g} in our hybrid format from the pre-activations x​WgxW\_{g}, we design custom kernels to perform efficient hybrid-to-dense and dense-to-hybrid matmuls. We directly use these kernels to execute the rest of the forward pass, computing hu=x​Wuh\_{u}=xW\_{u} and y=h​Wdy=hW\_{d}. Unlike inference, during training we execute the up and down projections in separate steps, allowing us to efficiently store the sparsified hidden states and minimize recomputation in the backward pass. In Algorithm [3](#alg3 "Algorithm 3 ‣ 4 Experimental Results ‣ Sparser, Faster, Lighter Transformer Language Models"), we outline the logic of the hybrid-to-dense matmul for a generic input hh and weight WW, with the dense-to-hybrid variant also following the same general structure. Our approach combines a typical ELL kernel, with each CTA processing individual rows of the output yy (lines 4-13), and a traditional tiled kernel using Tensor cores for the dense backup rows (lines 14-17). During the sparse portion of the matmul computation, we opt to statically-unroll the accumulation up to the maximum number of non-zeros Nn​z^N\_{\hat{nz}} for each row. Moreover, we also statically pre-allocate the dense backup portions of all the activations based on the sparsity statistics observed during training. We note that these design choices introduce minimal extra computation and storage costs, which are largely offset by avoiding dynamic overheads.

During the backward pass, we retrieve the sparsified activations together with the L1 and output gradients ∇y\nabla y, allowing us to backpropagate without performing any expensive dense computation. This is achieved using two additional kernels that support efficient injection of L1 gradients into a given sparsity pattern and efficient transposition of our hybrid format for future coalesced accesses. We first use the stored sparsity pattern of hh to obtain its gradients via our efficient dense-to-hybrid matmul ∇y​WdT\nabla yW^{T}\_{d}, followed by the L1 injection. With ∇h\nabla h available, we recover the rest of the input and weight gradients with direct applications of our hybrid-to-dense and transposed kernels:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∇hu\displaystyle\nabla h\_{u} | =∇h⊙hg,∇hg=∇h⊙hu,\displaystyle=\nabla h\odot h\_{g},\quad\nabla h\_{g}=\nabla h\odot h\_{u}, |  | (4) |
|  | ∇Wu=x⊤​∇\displaystyle\nabla W\_{u}=x^{\top}\nabla | hu,∇Wg=x⊤​∇hg,∇Wd=h⊤​∇y,\displaystyle h\_{u},\quad\nabla W\_{g}=x^{\top}\nabla h\_{g},\quad\nabla W\_{d}=h^{\top}\nabla y,\quad |  |
|  |  | ∇x=∇hu​Wu⊤+∇hg​Wg⊤.\displaystyle\nabla x=\nabla h\_{u}\,W\_{u}^{\top}+\nabla h\_{g}\,W\_{g}^{\top}. |  |

Crucially, our execution logic reflects a deliberate design choice: rather than aggressively fusing individual operators, the training kernel pipeline is structured around the training step in its entirety. In this setting, the hybrid format minimizes backward computation and memory overheads, allowing us to avoid dense calculations while remaining robust to the highly non-uniform sparsity patterns that make ELL-based approaches traditionally brittle.

## 4 Experimental Results

![Refer to caption](/html/2603.23198/assets/x3.png)
  


Figure 2: Training curves of LLMs across L1 regularization levels.




Algorithm 3  Algorithmic description of matmul from input activations in the hybrid format

  

1: Input: Tm,Tk,TnT\_{m},T\_{k},T\_{n}, h:=(hs,hd,hI,hb)h:=(h^{s},h^{d},h\_{I},h\_{b}),
W∈ℝK×NW\!\in\!\mathbb{R}^{K\times N}

2: Output: y∈ℝM×Ny\!\in\!\mathbb{R}^{M\times N}

3: πs←{m:hb​[m]=0}\pi\_{s}\leftarrow\{m:h\_{b}[m]=0\}; πd←{m:hb​[m]=1}\pi\_{d}\leftarrow\{m:h\_{b}[m]=1\}

4: for all ms∈0..Ms−1m\_{s}\in 0..M^{s}{-}1 in parallel do

5:  m←πs​[ms]m\leftarrow\pi\_{s}[m\_{s}] {global row index}

6:  ym←0y\_{m}\leftarrow 0 {row accumulator}

7:  for j←0j\leftarrow 0 …Nn​z^−1N\_{\hat{nz}}{-}1 do

8:  n←hI​[ms,j]n\leftarrow h\_{I}[m\_{s},j] {non-zero column index}

9:  v←hs​[ms,j]v\leftarrow h^{s}[m\_{s},j] {sparse value}

10:  ym←ym+v⋅W​[n,:]y\_{m}\leftarrow y\_{m}+v\cdot W[n,:] {sparse row update}

11:  end for

12:  y​[m,:]←ymy[m,:]\leftarrow y\_{m}

13: end for

14: for all tiles starting at (m0,n0)(m\_{0},n\_{0}) in parallel do

15:  S←hd[m0:m0+Tm,:]W[:,n0:n0+Tn]S\leftarrow h^{d}[m\_{0}{:}m\_{0}{+}T\_{m},:]\;W[:,n\_{0}{:}n\_{0}{+}T\_{n}]

16:  y[πd[m0:m0+Tm],n0:n0+Tn]←Sy[\pi\_{d}[m\_{0}{:}m\_{0}{+}T\_{m}],\,n\_{0}{:}n\_{0}{+}T\_{n}]\leftarrow S

17: end for

### 4.1 Training and Evaluation Settings

We provide quantitative results evaluating the performance and efficiency of LLMs at different sparsity levels and scales. Our models are based on the “Transformer++” architecture, common to recent LLMs such as Qwen and Llama (llama2; qwen2) with the gated feedforward blocks described in Section [2](#S2 "2 Large Language Models, Feedforward Blocks, and Sparsity ‣ Sparser, Faster, Lighter Transformer Language Models"). We train our models just above the chinchilla-optimal number of tokens for each model size (chinchilla), using the fineweb dataset (fineweb). We default to a context length of 2048, a batch size of 1M tokens, and the AdamW optimizer with a weight decay of 0.1 and a cosine schedule (adamw). Other hyperparameters, such as the hidden size and total number of layers, are based on the model size and chosen based on modern practices (chinchilla). We note that our sparse models use the same training hyperparameters as our non-sparse baselines, as the addition of L1 regularization in the feedforward blocks did not seem to affect other choices.

To measure model performance, we use cross-entropy scores and seven different common downstream tasks assessing logic and reasoning capabilities after pretraining (bench\_1\_arc; bench\_2\_hellaswag; bench\_3\_openbook\_qa; bench\_4\_piqa; bench\_5\_winogrande; bench\_6\_commonsenseqa). In this Section, we focus on aggregated performance metrics for conciseness, and we refer to Appendix [D](#A4 "Appendix D Extended Results ‣ Sparser, Faster, Lighter Transformer Language Models") for the full per-task breakdowns. To measure efficiency, we analyze the throughput gains at different sparsity levels when integrating our training and inference kernels by recording execution times, memory requirements, and energy consumption at each stage. Across our experiments, we keep a fixed sequence length of 2048 and vary the micro batch size based on the available memory. Unless otherwise specified, we train and collect our results on a single node of eight H100 PCIe GPUs, a commonly available infrastructure in current listings of cloud providers and scientific clusters. We refer to Appendix [B](#A2 "Appendix B Hyperparameters and Datasets ‣ Sparser, Faster, Lighter Transformer Language Models") for further details on our training and evaluation settings, together with full hyperparameters specific to each of the considered model sizes.

![Refer to caption](/html/2603.23198/assets/x4.png)
  


Figure 3: Downstream accuracy and sparsity statistics of LLMs across L1 regularization levels.

![Refer to caption](/html/2603.23198/assets/x5.png)
  


Figure 4: Forward pass speedups and energy savings from our sparse LLM inference kernels across L1 regularization levels.

### 4.2 More Efficient LLMs with Unstructured Sparsity

![Refer to caption](/html/2603.23198/assets/x6.png)


Figure 5: Training speedups and peak memory reduction from our sparse LLM training kernels across L1 regularization levels.

Performance across sparsity levels. We start by evaluating the effect of introducing different levels of L1 regularization on the performance and sparsity of a 1.5B parameter model. In particular, we consider eight values for the L1L\_{1} coefficient, ranging from no regularization (L1=0L\_{1}=0) to the point where less than a single neuron on average remains activated after training (L1=1×10−4L\_{1}=1\times 10^{-4}).
In Figure [2](#S4.F2 "Figure 2 ‣ 4 Experimental Results ‣ Sparser, Faster, Lighter Transformer Language Models"), we show the training curves of the different models, while in Figure [3](#S4.F3 "Figure 3 ‣ 4.1 Training and Evaluation Settings ‣ 4 Experimental Results ‣ Sparser, Faster, Lighter Transformer Language Models") we report downstream task performance together with the final number of non-zero activations averaged across the feed-forward blocks. While our 1.5B model has a hidden feedforward dimensionality of 5632, we find that the non-regularized model already attains more than 20% sparsity with only 911 neurons activated. Moreover, consistently with mirzadeh2023relu\_apple\_finetune, we find that introducing small levels of regularization already pushes the average number of non-zeros orders of magnitude lower but with high variations across different tokens and layers. In particular, even at the highest regularization point, we find that a small fraction of tokens still excite several hundred neurons, indicating a reallocation of capacity. Despite this adaptivity, performance-wise, we do start seeing some performance degradation below 0.5% of activated neurons. Nonetheless, our results suggest that smaller levels of regularization do not visibly hinder capacity beyond the weight decay already induced by the AdamW optimizer: up until L1=3×10−5L\_{1}=3\times 10^{-5}, we record essentially no drop in task performance and a negligible increase of final cross-entropy within 2% of the unregularized baseline.

Table 1: Comparison of performance and efficiency statistics of sparse LLMs leveraging our kernels with traditional models.

| Model scale | Sparse | Mean task accuracy | Forward execution [-1pt](input tokens/ms) | Energy per token [-1pt](mJ) | Training step [-1pt](input token/ms) | Peak memory [-1pt](GB) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5B params [-1pt]10B tokens | ✗ | 40.4% | 410 +0(0.0%) | 1.63 +0(0.0%) | 97.3 -(0.0%) | 26.2 +0(0.0%) |
| ✓ | 40.4% | 480 (+17.0%) | 1.43 (-11.8%) | 95.9 (-1.5%) | 21.2 (-19.2%) |
| 1B params [-1pt]20B tokens | ✗ | 44.6% | 185 +0(0.0%) | 3.71 +0(0.0%) | 48.6 +0(0.0%) | 44.5 +0(0.0%) |
| ✓ | 44.7% | 219 (+18.1%) | 3.17 (-14.6%) | 52.1 (+7.1%) | 33.1 (-25.5%) |
| 1.5B params [-1pt]30B tokens | ✗ | 46.4% | 119 +0(0.0%) | 5.73 +0(0.0%) | 31.8 +0(0.0%) | 62.8 +0(0.0%) |
| ✓ | 46.2% | 141 (+18.8%) | 4.87 (-15.0%) | 35.5 (+11.6%) | 45.1 (-28.1%) |
| 2B params [-1pt]40B tokens | ✗ | 49.1% | 87.8 +0(0.0%) | 7.85 +0(0.0%) | 22.4+0(0.0%) | 46.7 +0(0.0%) |
| ✓ | 48.8% | 106 (+20.5%) | 6.51 (-17.0%) | 27.3 (+21.9%) | 57.1 (+22.3%) |

Leveraging sparsity for faster and lighter LLMs. We contrast our performance results by analyzing the efficiency improvements from integrating our kernels at different sparsity levels. In Figure [4](#S4.F4 "Figure 4 ‣ 4.1 Training and Evaluation Settings ‣ 4 Experimental Results ‣ Sparser, Faster, Lighter Transformer Language Models"), we provide the average relative speedups and total energy savings recorded during forward execution through our LLMs. Across all considered sparsity levels above L1=0L\_{1}=0, we find that our inference kernels lead to visible throughput gains ranging up to 30%. These throughput gains are compounded by nearly 3% less GPU power draw above L1=3×10−5L\_{1}=3\times 10^{-5}, resulting in even higher energy savings. In Figure [5](#S4.F5 "Figure 5 ‣ 4.2 More Efficient LLMs with Unstructured Sparsity ‣ 4 Experimental Results ‣ Sparser, Faster, Lighter Transformer Language Models"), we also show the average relative speedups and peak memory reduction with our training kernel.
In line with our inference kernels, the speedups recorded throughout training significantly increase up to 24% with sparser models. Furthermore, the peak GPU memory required for training decreases by more than 24% even for the lowest considered sparsity level, reducing hardware barriers for efficient training at billion-parameter scales (we refer to Appendix [D](#A4 "Appendix D Extended Results ‣ Sparser, Faster, Lighter Transformer Language Models") for results on an RTX6000). Taken together, we believe our results provide compelling evidence that specialized targeted kernels can make sparsity a new viable axis for the design of modern LLMs, leading to significant efficiency improvements across their full lifecycle.

Sparsity across model scales. We analyze how model scale affects the performance and efficiency of sparse LLMs. For this analysis, we set L1=2×10−5L\_{1}=2\times 10^{-5} based on our earlier results on the 1.5B model, which we recommend as a conservative threshold to avoid any significant performance degradation. In Table [1](#S4.T1 "Table 1 ‣ 4.2 More Efficient LLMs with Unstructured Sparsity ‣ 4 Experimental Results ‣ Sparser, Faster, Lighter Transformer Language Models"), we compare the performance and efficiency of sparse and non-sparse LLMs at the chinchilla-optimality boundary – ranging from a 0.5B model trained on 10B tokens to a 2B model trained on 40B tokens. Consistent with our earlier results, we find no performance drops beyond random deviations for all scales when mild L1 regularization is introduced. Furthermore, we find that LLMs become increasingly effective at supporting sparsity at larger scales, resulting in a lower number of average non-zero elements (from 39 to 24, going from the 0.5B to the 2B model). In turn, this makes all the aforementioned throughput and memory benefits of our kernels grow: the 2B sparse model processes tokens 20.5% faster during inference and trains 21.9% more efficiently with a larger micro-batch size.
These findings suggest that sparsity aligns well with recently prevailing scaling trends, highlighting its growing potential relevance for future model development.

### 4.3 The Properties of Sparse Large Language Models

![Refer to caption](/html/2603.23198/assets/x7.png)


Figure 6: Sparsity statistics and speedup contributions across different layers of our sparse LLMs.

We analyze how LLMs effectively allocate sparsity across their layers and batched samples. For our analysis, we collect the activations from 2202^{20} input tokens using our 1.5B model trained with the suggested performance-preserving L1=2×10−5L\_{1}=2\times 10^{-5}. We complement this subsection with additional results in Appendix [D](#A4 "Appendix D Extended Results ‣ Sparser, Faster, Lighter Transformer Language Models"), looking at additional levels of L1 regularization together with how sparsity evolves throughout training and its effects on dead neurons.

Sparsity and model depth. Figure [6](#S4.F6 "Figure 6 ‣ 4.3 The Properties of Sparse Large Language Models ‣ 4 Experimental Results ‣ Sparser, Faster, Lighter Transformer Language Models") examines activations across model depth, relating the non-zero statistics of each layer to its respective contribution to inference speed-ups. While the average non-zeros across all layers is less than 30, the figure highlights clear variations in sparsity both across and within individual layers.
In particular, the first two layers are the least active, followed by a pronounced hump in the average number of non-zeros across the first half of the network.
This sparsity pattern, peaking during early-middle layers, appears consistent with prior work suggesting that a substantial portion of an LLM’s reasoning and knowledge retrieval occurs precisely at these depths (llamas\_eng\_an).
Furthermore, within each layer, the maximum number of non-zeros often exceeds the layer’s mean by more than an order of magnitude and shows no consistent pattern across the architecture. We also observe an intuitive and pronounced inverse correlation between each layer’s average non-zeros and its relative speed-up, with a Pearson coefficient of over -0.996. In contrast, the maximum activation counts have a more limited effect on inference speedups, only noticeably in layer 8. This robustness is due to our kernel design, which hides the latency of highly activated tokens through maximally parallelized execution.

![Refer to caption](/html/2603.23198/assets/x8.png)


Figure 7: Sparsity statistics across LLM input tokens and positions.

Sparsity and input properties. Given the high level of unevenness across activations, we analyze what inputs spur the peaks and troughs in non-zero activation counts. In part a. of Figure [7](#S4.F7 "Figure 7 ‣ 4.3 The Properties of Sparse Large Language Models ‣ 4 Experimental Results ‣ Sparser, Faster, Lighter Transformer Language Models"), we identify common tokens with the six lowest and highest average number of non-zeros, filtering out outliers occurring at a lower frequency than 1/2141/2^{14}. We find that the tokens with the lowest non-zero activity often represent parts of common web links (doi, nlm, gov, nih) or contractions (doesn, couldn) that precede predictable next tokens in crawled web corpora. In contrast, tokens providing important contextual information about a passage have the highest activity, including particular verbs (loud, enduring) or nouns representing specific locations or substances (Vermont, Greeks, formaldehyde, ACH"). In part b. of Figure [7](#S4.F7 "Figure 7 ‣ 4.3 The Properties of Sparse Large Language Models ‣ 4 Experimental Results ‣ Sparser, Faster, Lighter Transformer Language Models"), we then plot how the average non-zeros vary with token position in the input sequence on a log-log scale. Interestingly, we find that the LLM allocates a much greater number of non-zeros to the very first tokens in a sequence, with an exponential decrease thereafter. Intuitively, these results indicate that LLMs appear to effectively focus their computational efforts on tokens with high information content and sequence positions where contextual cues from prior tokens are missing. Here, introducing sparsity not only provides an interpretable lens on model behavior, but also enables our kernels to leverage this inherent information unevenness for significant training and inference speedups.

## 5 Related Work

The emergence of sparsity in LLMs with ReLU activations and its theoretical benefits have been repeatedly documented in earlier work (li2023lazyneuronphenomenonemergence; mirzadeh2023relu\_apple\_finetune).
Since then, more recent methods have been proposed to enhance sparsity by altering modern gated architectures, claiming speedups running sparse feed-forward layers in isolation on older generations of devices. TurboSparse (song2024turbo) studies boosting sparsity via repeated ReLU non-linearities, while ProSparse (song2024prosparse) finetunes pretrained models by manually thresholding activations. Q-Sparse (q\_sparse\_top\_k\_related) further deviates from standard models by using a straight-through estimator and retaining only top-K activations. Other work instead focused on introducing structured sparsity post-training, such as by predicting (dejavu\_contectual\_sparsity\_related) and pruning activations to accelerate computation (cats\_post\_training\_thresh\_related; teal\_post\_training\_thresh\_related). Unlike these efforts, our paper introduces general-purpose kernels to leverage unstructured sparsity, demonstrating empirical efficiency benefits during both LLM inference and training. We refer to Appendix [E](#A5 "Appendix E Further Related Work ‣ Sparser, Faster, Lighter Transformer Language Models") for an extended overview of prior work that more fundamentally reshapes architecture design.

## 6 Discussion and Future Work

In this work, we leverage unstructured sparsity to lessen the computational burdens of modern LLMs. For inference, we design a new sparse format and fused operations to efficiently execute the whole gated feed-forward blocks in only two kernel launches, minimizing global memory accesses and computation.
For training, we introduce a new hybrid algorithm that dynamically schedules computation on both CUDA and Tensor cores, while trivializing storage costs of intermediate activations for backpropagation. We demonstrate that mild L1 regularization induces considerable levels of sparsity with negligible impact on downstream performance – which our kernels translate into significant gains in throughput, energy efficiency, and memory footprint at billion-parameter scales. While our work serves to provide a concrete demonstration of the benefits of sparse LLMs, there are numerous exciting avenues for future extensions. For instance, in Appendix [C](#A3 "Appendix C Parameter Studies and Ablations ‣ Sparser, Faster, Lighter Transformer Language Models"), we provide preliminary results indicating that the performance of highly sparse LLMs can be further improved with strategies targeted at dead-neuron mitigation. Moreover, fine-tuning existing dense models via recent sparsification approaches (mirzadeh2023relu\_apple\_finetune; song2024prosparse) would allow bringing the benefits of our kernels to the vast library of pretrained LLMs available in the wild. By sharing our kernels, we hope our work will help promote sparsity as a new design axis to leverage for efficiency, ultimately reducing the growing energy and hardware costs of large-scale foundation models.

## Author contribution

Edoardo Cetin conceived the TwELL format, led the implementation and design of the CUDA kernels using TwELL, led model training and benchmarking, and made contributions to writing.

Stefano Peluchetti did early work on sparse model training, advised the project, and made contributions to writing.

Emilio Castillo conceived the hybrid format, co-led the implementation of the CUDA kernels and designed the training extensions, made contributions to kernel benchmarking, and made contributions to writing.

Akira Naruse advised the project, was involved in early discussions about method design, and worked on early implementations of the sparse kernels.

Mana Murakami was involved in early discussions about method design.

Llion Jones did initial explorations of sparse model training, advised the project, was involved in early discussions about method design, and made contributions to writing.

## References

## Appendix A Kernels Implementation Details

### A.1 Inference Kernels Selection

[⬇](data:text/plain;base64,dGVtcGxhdGUgPGNvbnN0IGludCBUX20sIGNvbnN0IGludCBUX24sIGNvbnN0IGludCBUX2s+CnN0cnVjdCBUaWxlcwp7CiAgICBhbGlnbmFzKDEyOCkgX19udl9iZmxvYXQxNiBhW1RfbV1bVF9rXTsKICAgIGFsaWduYXMoMTI4KSBfX252X2JmbG9hdDE2IGJbVF9uXVtUX2tdOwp9OwoKdGVtcGxhdGUgPAogICAgY29uc3QgaW50IFRfbSwKICAgIGNvbnN0IGludCBUX24sCiAgICBjb25zdCBpbnQgVF9rLAogICAgY29uc3QgaW50IFFVRVVFX1NJWkUsCiAgICBjb25zdCBpbnQgVF9uX2NvbXByZXNzZWQsCiAgICBpbnQgUEFERElORyA9IDQKPgpzdHJ1Y3QgU21lbVN0b3JhZ2UKewogICAgVGlsZXM8VF9tLCBUX24sIFRfaz4gcXVldWVbUVVFVUVfU0laRV07CiAgICBhbGlnbmFzKDEyOCkgdWludDMyX3QgY19wYWNrZWRbVF9tXVtUX25fY29tcHJlc3NlZCArIFBBRERJTkddOwp9OwoKdGVtcGxhdGUgPAogICAgY29uc3QgaW50IFRfbSwKICAgIGNvbnN0IGludCBUX24sCiAgICBjb25zdCBpbnQgVF9rLAogICAgY29uc3QgaW50IENMVVNURVJfRElNX20sCiAgICBjb25zdCBpbnQgQ0xVU1RFUl9ESU1fbiwKICAgIGNvbnN0IGludCBRVUVVRV9TSVpFLAogICAgY29uc3QgaW50IE5VTV9BQ1RJVkVfU01zLAogICAgY29uc3QgaW50IFRfbl9jb21wcmVzc2VkLAogICAgY29uc3QgYm9vbCBMT09QX09WRVJGTE9XX1NUT1JBR0UKPgpfX2dsb2JhbF9fIF9fbGF1bmNoX2JvdW5kc19fKE5VTV9USFJFQURTX1BFUl9CTE9DSykKICAgICAgICAgICBfX2NsdXN0ZXJfZGltc19fKENMVVNURVJfRElNX20gKiBDTFVTVEVSX0RJTV9uLCAxLCAxKQp2b2lkIG1tX3dnbW1hX250X2tlcm5lbCgKICAgIGNvbnN0IENVdGVuc29yTWFwIF9fZ3JpZF9jb25zdGFudF9fIEFfdG0sCiAgICBjb25zdCBDVXRlbnNvck1hcCBfX2dyaWRfY29uc3RhbnRfXyBCX3RtLAogICAgY29uc3QgQ1V0ZW5zb3JNYXAgX19ncmlkX2NvbnN0YW50X18gQ19wYWNrZWRfdG0sCiAgICBjb25zdCBpbnQqIHNjaGVkdWxlX2dtZW1fcHRyLAogICAgY29uc3QgaW50IHNjaGVkdWxlX3NpemVfcGVyX3NtLAogICAgY29uc3QgaW50IEsKKQp7CiAgICBzdGF0aWNfYXNzZXJ0KAogICAgICAgIChUX20gPT0gNjQgKiAyKSwKICAgICAgICAiT25seSBUX20gPT0gMTI4IHN1cHBvcnRlZCIKICAgICk7CgogICAgY29uc3RleHByIGludCBDTFVTVEVSX1NJWkUgPSBDTFVTVEVSX0RJTV9tICogQ0xVU1RFUl9ESU1fbjsKICAgIGV4dGVybiBfX3NoYXJlZF9fIF9fYWxpZ25fXygxMDI0KSB1bnNpZ25lZCBjaGFyIGR5bmFtaWNfc21lbVtdOwoKICAgIGludCBjbHVzdGVyX2lkeDsKICAgIGFzbSAoIm1vdi51MzIgJQoKICAgIGludCBjbHVzdGVyX2xhbmVfbTsKICAgIGFzbSB2b2xhdGlsZSgibW92LnUzMiAlCgogICAgaW50IGNsdXN0ZXJfbGFuZV9uID0gY2x1c3Rlcl9sYW5lX20gJQogICAgY2x1c3Rlcl9sYW5lX20gLz0gQ0xVU1RFUl9ESU1fbjsKCiAgICBhdXRvJiB0aWxlc19zID0KICAgICAgICAqcmVpbnRlcnByZXRfY2FzdDwKICAgICAgICAgICAgU21lbVN0b3JhZ2U8VF9tLCBUX24sIFRfaywgUVVFVUVfU0laRSwgVF9uX2NvbXByZXNzZWQ+KgogICAgICAgID4oZHluYW1pY19zbWVtKTsKICAgIGludCogc2NoZWR1bGVfcyA9IHJlaW50ZXJwcmV0X2Nhc3Q8aW50Kj4oCiAgICAgICAgZHluYW1pY19zbWVtCiAgICAgICAgKyBzaXplb2YoU21lbVN0b3JhZ2U8VF9tLCBUX24sIFRfaywgUVVFVUVfU0laRSwgVF9uX2NvbXByZXNzZWQ+KQogICAgKTsKCiAgICBzY2hlZHVsZV9nbWVtX3B0ciArPSBjbHVzdGVyX2lkeCAqIHNjaGVkdWxlX3NpemVfcGVyX3NtOwogICAgaWYgKHRocmVhZElkeC54IDwgc2NoZWR1bGVfc2l6ZV9wZXJfc20pIHsKICAgICAgICBzY2hlZHVsZV9zW3RocmVhZElkeC54XSA9IHNjaGVkdWxlX2dtZW1fcHRyW3RocmVhZElkeC54XTsKICAgIH0KCiAgICBfX3N5bmN0aHJlYWRzKCk7CgogICAgX19zaGFyZWRfXyBfX2FsaWduX18oOCkgdWludDY0X3QgcXVldWVfZnVsbFtRVUVVRV9TSVpFXTsKICAgIF9fc2hhcmVkX18gX19hbGlnbl9fKDgpIHVpbnQ2NF90IHF1ZXVlX2VtcHR5W1FVRVVFX1NJWkVdOwoKICAgIGlmICh0aHJlYWRJZHgueCA9PSAwKSB7CiAgICAgICAgI3ByYWdtYSB1bnJvbGwKICAgICAgICBmb3IgKGludCBxdWV1ZV9pZHggPSAwOyBxdWV1ZV9pZHggPCBRVUVVRV9TSVpFOyArK3F1ZXVlX2lkeCkgewogICAgICAgICAgICBwdHhfaW5pdF9zbWVtX2JhcnJpZXIoJnF1ZXVlX2Z1bGxbcXVldWVfaWR4XSwgMSk7CiAgICAgICAgICAgIHB0eF9pbml0X3NtZW1fYmFycmllcigmcXVldWVfZW1wdHlbcXVldWVfaWR4XSwgMiAqIENMVVNURVJfU0laRSk7CiAgICAgICAgfQogICAgfQoKICAgIGFzbSB2b2xhdGlsZSgiYmFycmllci5jbHVzdGVyLmFycml2ZTtcbiIgOiA6KTsKICAgIGFzbSB2b2xhdGlsZSgiYmFycmllci5jbHVzdGVyLndhaXQ7XG4iIDogOik7CgogICAgaWYgKHRocmVhZElkeC54IDwgV0FSUF9HUk9VUF9TSVpFKSB7CiAgICAgICAgYXNtIHZvbGF0aWxlKCJzZXRtYXhucmVnLmRlYy5zeW5jLmFsaWduZWQudTMyICUKCiAgICAgICAgaWYgKHRocmVhZElkeC54ID09IDApIHsKICAgICAgICAgICAgaW50IHF1ZXVlX2lkeCA9IDA7CiAgICAgICAgICAgIGludCBxdWV1ZV9waGFzZSA9IDA7CiAgICAgICAgICAgIHVpbnQxNl90IG1hc2tfbXVsdGljYXN0X20gPSAwOwoKICAgICAgICAgICAgaWYgY29uc3RleHByIChDTFVTVEVSX0RJTV9tID4gMSkgewogICAgICAgICAgICAgICAgZm9yIChpbnQgaSA9IDA7IGkgPCBDTFVTVEVSX0RJTV9tOyArK2kpIHsKICAgICAgICAgICAgICAgICAgICBtYXNrX211bHRpY2FzdF9tIHw9ICgxdSA8PCAoaSAqIENMVVNURVJfRElNX24pKTsKICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgIG1hc2tfbXVsdGljYXN0X20gPDw9IGNsdXN0ZXJfbGFuZV9uOwogICAgICAgICAgICB9CgogICAgICAgICAgICB1aW50MTZfdCBtYXNrX211bHRpY2FzdF9uOwogICAgICAgICAgICBpZiBjb25zdGV4cHIgKENMVVNURVJfRElNX24gPiAxKSB7CiAgICAgICAgICAgICAgICBtYXNrX211bHRpY2FzdF9uID0KICAgICAgICAgICAgICAgICAgICAoKDF1IDw8IENMVVNURVJfRElNX24pIC0gMSkKICAgICAgICAgICAgICAgICAgICA8PCAoY2x1c3Rlcl9sYW5lX20gKiBDTFVTVEVSX0RJTV9uKTsKICAgICAgICAgICAgfQoKICAgICAgICAgICAgZm9yIChpbnQgc2NoZWR1bGVfaXQgPSAwOwogICAgICAgICAgICAgICAgIHNjaGVkdWxlX2l0IDwgc2NoZWR1bGVfc2l6ZV9wZXJfc207CiAgICAgICAgICAgICAgICAgKytzY2hlZHVsZV9pdCkgewogICAgICAgICAgICAgICAgY29uc3QgaW50IHBhY2tlZF90aWxlID0gc2NoZWR1bGVfc1tzY2hlZHVsZV9pdF07CiAgICAgICAgICAgICAgICBpZiAocGFja2VkX3RpbGUgPT0gLTEpIHsKICAgICAgICAgICAgICAgICAgICBicmVhazsKICAgICAgICAgICAgICAgIH0KCiAgICAgICAgICAgICAgICBpbnQgdGlsZV9jb29yZF9tID0gcGFja2VkX3RpbGUgPj4gMTY7CiAgICAgICAgICAgICAgICBpbnQgdGlsZV9jb29yZF9uID0gcGFja2VkX3RpbGUgJiAweEZGRkY7CgogICAgICAgICAgICAgICAgaWYgY29uc3RleHByIChDTFVTVEVSX0RJTV9uID4gMSkgewogICAgICAgICAgICAgICAgICAgIHRpbGVfY29vcmRfbiAqPSBDTFVTVEVSX0RJTV9uOwogICAgICAgICAgICAgICAgICAgIHRpbGVfY29vcmRfbiArPSBjbHVzdGVyX2xhbmVfbjsKICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgIGlmIGNvbnN0ZXhwciAoQ0xVU1RFUl9ESU1fbSA+IDEpIHsKICAgICAgICAgICAgICAgICAgICB0aWxlX2Nvb3JkX20gKj0gQ0xVU1RFUl9ESU1fbTsKICAgICAgICAgICAgICAgICAgICB0aWxlX2Nvb3JkX20gKz0gY2x1c3Rlcl9sYW5lX207CiAgICAgICAgICAgICAgICB9CgogICAgICAgICAgICAgICAgZm9yIChpbnQgdGlsZV9zdGFydF9rID0gMDsKICAgICAgICAgICAgICAgICAgICAgdGlsZV9zdGFydF9rIDwgSzsKICAgICAgICAgICAgICAgICAgICAgdGlsZV9zdGFydF9rICs9IFRfaywgKytxdWV1ZV9pZHgpIHsKICAgICAgICAgICAgICAgICAgICBpZiAocXVldWVfaWR4ID09IFFVRVVFX1NJWkUpIHsKICAgICAgICAgICAgICAgICAgICAgICAgcXVldWVfaWR4ID0gMDsKICAgICAgICAgICAgICAgICAgICAgICAgcXVldWVfcGhhc2UgXj0gMTsKICAgICAgICAgICAgICAgICAgICB9CgogICAgICAgICAgICAgICAgICAgIHB0eF93YWl0X2JhcnJpZXIoJnF1ZXVlX2VtcHR5W3F1ZXVlX2lkeF0sIHF1ZXVlX3BoYXNlKTsKICAgICAgICAgICAgICAgICAgICBwdHhfYXJyaXZlX3R4X3NtZW1fYmFycmllcigKICAgICAgICAgICAgICAgICAgICAgICAgJnF1ZXVlX2Z1bGxbcXVldWVfaWR4XSwKICAgICAgICAgICAgICAgICAgICAgICAgc2l6ZW9mKHRpbGVzX3MucXVldWVbcXVldWVfaWR4XS5hKQogICAgICAgICAgICAgICAgICAgICAgICArIHNpemVvZih0aWxlc19zLnF1ZXVlW3F1ZXVlX2lkeF0uYikKICAgICAgICAgICAgICAgICAgICApOwoKICAgICAgICAgICAgICAgICAgICBpZiBjb25zdGV4cHIgKENMVVNURVJfRElNX24gPiAxKSB7CiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChjbHVzdGVyX2xhbmVfbiA9PSAwKSB7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICBwdHhfbG9hZF90aWxlX3RtYV9tdWx0aWNhc3RfMmQoCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJnRpbGVzX3MucXVldWVbcXVldWVfaWR4XS5hWzBdWzBdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICZBX3RtLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRpbGVfY29vcmRfbSAqIFRfbSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aWxlX3N0YXJ0X2ssCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbWFza19tdWx0aWNhc3RfbiwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAmcXVldWVfZnVsbFtxdWV1ZV9pZHhdCiAgICAgICAgICAgICAgICAgICAgICAgICAgICApOwogICAgICAgICAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAgICAgfSBlbHNlIHsKICAgICAgICAgICAgICAgICAgICAgICAgcHR4X2xvYWRfdGlsZV90bWFfMmQoCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAmdGlsZXNfcy5xdWV1ZVtxdWV1ZV9pZHhdLmFbMF1bMF0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAmQV90bSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRpbGVfY29vcmRfbSAqIFRfbSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRpbGVfc3RhcnRfaywKICAgICAgICAgICAgICAgICAgICAgICAgICAgICZxdWV1ZV9mdWxsW3F1ZXVlX2lkeF0KICAgICAgICAgICAgICAgICAgICAgICAgKTsKICAgICAgICAgICAgICAgICAgICB9CgogICAgICAgICAgICAgICAgICAgIGlmIGNvbnN0ZXhwciAoQ0xVU1RFUl9ESU1fbSA+IDEpIHsKICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGNsdXN0ZXJfbGFuZV9tID09IDApIHsKICAgICAgICAgICAgICAgICAgICAgICAgICAgIHB0eF9sb2FkX3RpbGVfdG1hX211bHRpY2FzdF8yZCgKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAmdGlsZXNfcy5xdWV1ZVtxdWV1ZV9pZHhdLmJbMF1bMF0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJkJfdG0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGlsZV9jb29yZF9uICogVF9uLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRpbGVfc3RhcnRfaywKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBtYXNrX211bHRpY2FzdF9tLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICZxdWV1ZV9mdWxsW3F1ZXVlX2lkeF0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICk7CiAgICAgICAgICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgICAgICB9IGVsc2UgewogICAgICAgICAgICAgICAgICAgICAgICBwdHhfbG9hZF90aWxlX3RtYV8yZCgKICAgICAgICAgICAgICAgICAgICAgICAgICAgICZ0aWxlc19zLnF1ZXVlW3F1ZXVlX2lkeF0uYlswXVswXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICZCX3RtLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgdGlsZV9jb29yZF9uICogVF9uLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgdGlsZV9zdGFydF9rLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgJnF1ZXVlX2Z1bGxbcXVldWVfaWR4XQogICAgICAgICAgICAgICAgICAgICAgICApOwogICAgICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgfQogICAgICAgIH0KICAgIH0gZWxzZSB7CiAgICAgICAgYXNtIHZvbGF0aWxlKCJzZXRtYXhucmVnLmluYy5zeW5jLmFsaWduZWQudTMyICUKICAgICAgICBpbnQgcXVldWVfaWR4ID0gMDsKICAgICAgICBpbnQgcXVldWVfcGhhc2UgPSAwOwogICAgICAgIGNvbnN0IGludCBjb25zdW1lcl93YXJwZ3JvdXBfaWQgPQogICAgICAgICAgICAodGhyZWFkSWR4LnggLSBXQVJQX0dST1VQX1NJWkUpIC8gV0FSUF9HUk9VUF9TSVpFOwogICAgICAgIGNvbnN0IGludCB0aWxlX3N0YXJ0X20gPSBjb25zdW1lcl93YXJwZ3JvdXBfaWQgKiBXR01NQV9tOwogICAgICAgIGNvbnN0IGludCBjb25zdW1lcl90aHJlYWRfaWQgPSB0aHJlYWRJZHgueCAlCiAgICAgICAgY29uc3QgdWludCB0aHJlYWRfbGFuZV9pZHhfbiA9IChjb25zdW1lcl90aHJlYWRfaWQgJQoKICAgICAgICBjb25zdCBpbnQgdGhyZWFkX3N0b3JlX29mZnNldF9tID0gKAogICAgICAgICAgICB0aWxlX3N0YXJ0X20KICAgICAgICAgICAgKyBjb25zdW1lcl90aHJlYWRfaWQgLyAzMiAqIDE2CiAgICAgICAgICAgICsgKGNvbnN1bWVyX3RocmVhZF9pZCAlCiAgICAgICAgKTsKICAgICAgICBjb25zdCBpbnQgdGhyZWFkX3N0b3JlX29mZnNldF9uID0KICAgICAgICAgICAgKChjb25zdW1lcl90aHJlYWRfaWQgJQoKICAgICAgICBpZiAoY29uc3VtZXJfdGhyZWFkX2lkIDwgQ0xVU1RFUl9TSVpFKSB7CiAgICAgICAgICAgIGZvciAoaW50IHF1ZXVlX2lkeCA9IDA7IHF1ZXVlX2lkeCA8IFFVRVVFX1NJWkU7ICsrcXVldWVfaWR4KSB7CiAgICAgICAgICAgICAgICBwdHhfYXJyaXZlX2JhcnJpZXJfYWNyb3NzX2NsdXN0ZXIoCiAgICAgICAgICAgICAgICAgICAgJnF1ZXVlX2VtcHR5W3F1ZXVlX2lkeF0sCiAgICAgICAgICAgICAgICAgICAgY29uc3VtZXJfdGhyZWFkX2lkLAogICAgICAgICAgICAgICAgICAgIDEKICAgICAgICAgICAgICAgICk7CiAgICAgICAgICAgIH0KICAgICAgICB9CgogICAgICAgIGZsb2F0IENfYWNjdW1bVF9uLzE2XVs4XTsKICAgICAgICBmb3IgKGludCBzY2hlZHVsZV9pdCA9IDA7CiAgICAgICAgICAgICBzY2hlZHVsZV9pdCA8IHNjaGVkdWxlX3NpemVfcGVyX3NtOwogICAgICAgICAgICAgKytzY2hlZHVsZV9pdCkgewogICAgICAgICAgICBjb25zdCBpbnQgcGFja2VkX3RpbGUgPSBzY2hlZHVsZV9zW3NjaGVkdWxlX2l0XTsKICAgICAgICAgICAgaWYgKHBhY2tlZF90aWxlID09IC0xKSB7CiAgICAgICAgICAgICAgICBicmVhazsKICAgICAgICAgICAgfQoKICAgICAgICAgICAgaW50IHRpbGVfY29vcmRfbSA9IHBhY2tlZF90aWxlID4+IDE2OwogICAgICAgICAgICBpbnQgdGlsZV9jb29yZF9uID0gcGFja2VkX3RpbGUgJiAweEZGRkY7CgogICAgICAgICAgICBpZiBjb25zdGV4cHIgKENMVVNURVJfRElNX24gPiAxKSB7CiAgICAgICAgICAgICAgICB0aWxlX2Nvb3JkX24gKj0gQ0xVU1RFUl9ESU1fbjsKICAgICAgICAgICAgICAgIHRpbGVfY29vcmRfbiArPSBjbHVzdGVyX2xhbmVfbjsKICAgICAgICAgICAgfQogICAgICAgICAgICBpZiBjb25zdGV4cHIgKENMVVNURVJfRElNX20gPiAxKSB7CiAgICAgICAgICAgICAgICB0aWxlX2Nvb3JkX20gKj0gQ0xVU1RFUl9ESU1fbTsKICAgICAgICAgICAgICAgIHRpbGVfY29vcmRfbSArPSBjbHVzdGVyX2xhbmVfbTsKICAgICAgICAgICAgfQoKICAgICAgICAgICAgaWYgKHF1ZXVlX2lkeCA9PSBRVUVVRV9TSVpFKSB7CiAgICAgICAgICAgICAgICBxdWV1ZV9pZHggPSAwOwogICAgICAgICAgICAgICAgcXVldWVfcGhhc2UgXj0gMTsKICAgICAgICAgICAgfQoKICAgICAgICAgICAgcHR4X3dhaXRfYmFycmllcigmcXVldWVfZnVsbFtxdWV1ZV9pZHhdLCBxdWV1ZV9waGFzZSk7CiAgICAgICAgICAgIGFzbSB2b2xhdGlsZSgid2dtbWEuZmVuY2Uuc3luYy5hbGlnbmVkOyIgOjo6ICJtZW1vcnkiKTsKCiAgICAgICAgICAgIHdnbW1hPFRfbiwgMCwgMSwgMSwgMCwgMD4oCiAgICAgICAgICAgICAgICBDX2FjY3VtLAogICAgICAgICAgICAgICAgJnRpbGVzX3MucXVldWVbcXVldWVfaWR4XS5hW3RpbGVfc3RhcnRfbV1bMF0sCiAgICAgICAgICAgICAgICAmdGlsZXNfcy5xdWV1ZVtxdWV1ZV9pZHhdLmJbMF1bMF0KICAgICAgICAgICAgKTsKCiAgICAgICAgICAgICNwcmFnbWEgdW5yb2xsCiAgICAgICAgICAgIGZvciAoaW50IHdnbW1hX3N0YXJ0X2sgPSBXR01NQV9rOwogICAgICAgICAgICAgICAgIHdnbW1hX3N0YXJ0X2sgPCBUX2s7CiAgICAgICAgICAgICAgICAgd2dtbWFfc3RhcnRfayArPSBXR01NQV9rKSB7CiAgICAgICAgICAgICAgICB3Z21tYTxUX24sIDEsIDEsIDEsIDAsIDA+KAogICAgICAgICAgICAgICAgICAgIENfYWNjdW0sCiAgICAgICAgICAgICAgICAgICAgJnRpbGVzX3MucXVldWVbcXVldWVfaWR4XS5hW3RpbGVfc3RhcnRfbV1bd2dtbWFfc3RhcnRfa10sCiAgICAgICAgICAgICAgICAgICAgJnRpbGVzX3MucXVldWVbcXVldWVfaWR4XS5iWzBdW3dnbW1hX3N0YXJ0X2tdCiAgICAgICAgICAgICAgICApOwogICAgICAgICAgICB9CgogICAgICAgICAgICBhc20gdm9sYXRpbGUoIndnbW1hLmNvbW1pdF9ncm91cC5zeW5jLmFsaWduZWQ7IiA6OjogIm1lbW9yeSIpOwogICAgICAgICAgICBhc20gdm9sYXRpbGUoIndnbW1hLndhaXRfZ3JvdXAuc3luYy5hbGlnbmVkICUKCiAgICAgICAgICAgIGlmIChjb25zdW1lcl90aHJlYWRfaWQgPCBDTFVTVEVSX1NJWkUpIHsKICAgICAgICAgICAgICAgIHB0eF9hcnJpdmVfYmFycmllcl9hY3Jvc3NfY2x1c3RlcigKICAgICAgICAgICAgICAgICAgICAmcXVldWVfZW1wdHlbcXVldWVfaWR4XSwKICAgICAgICAgICAgICAgICAgICBjb25zdW1lcl90aHJlYWRfaWQsCiAgICAgICAgICAgICAgICAgICAgMQogICAgICAgICAgICAgICAgKTsKICAgICAgICAgICAgfQoKICAgICAgICAgICAgcXVldWVfaWR4Kys7CgogICAgICAgICAgICBmb3IgKGludCB0aWxlX2lkeF9rID0gMTsKICAgICAgICAgICAgICAgICB0aWxlX2lkeF9rIDwgSyAvIFRfazsKICAgICAgICAgICAgICAgICArK3RpbGVfaWR4X2ssICsrcXVldWVfaWR4KSB7CiAgICAgICAgICAgICAgICBpZiAocXVldWVfaWR4ID09IFFVRVVFX1NJWkUpIHsKICAgICAgICAgICAgICAgICAgICBxdWV1ZV9pZHggPSAwOwogICAgICAgICAgICAgICAgICAgIHF1ZXVlX3BoYXNlIF49IDE7CiAgICAgICAgICAgICAgICB9CgogICAgICAgICAgICAgICAgcHR4X3dhaXRfYmFycmllcigmcXVldWVfZnVsbFtxdWV1ZV9pZHhdLCBxdWV1ZV9waGFzZSk7CiAgICAgICAgICAgICAgICBhc20gdm9sYXRpbGUoIndnbW1hLmZlbmNlLnN5bmMuYWxpZ25lZDsiIDo6OiAibWVtb3J5Iik7CgogICAgICAgICAgICAgICAgI3ByYWdtYSB1bnJvbGwKICAgICAgICAgICAgICAgIGZvciAoaW50IHdnbW1hX3N0YXJ0X2sgPSAwOwogICAgICAgICAgICAgICAgICAgICB3Z21tYV9zdGFydF9rIDwgVF9rOwogICAgICAgICAgICAgICAgICAgICB3Z21tYV9zdGFydF9rICs9IFdHTU1BX2spIHsKICAgICAgICAgICAgICAgICAgICB3Z21tYTxUX24sIDEsIDEsIDEsIDAsIDA+KAogICAgICAgICAgICAgICAgICAgICAgICBDX2FjY3VtLAogICAgICAgICAgICAgICAgICAgICAgICAmdGlsZXNfcy5xdWV1ZVtxdWV1ZV9pZHhdLmFbdGlsZV9zdGFydF9tXVt3Z21tYV9zdGFydF9rXSwKICAgICAgICAgICAgICAgICAgICAgICAgJnRpbGVzX3MucXVldWVbcXVldWVfaWR4XS5iWzBdW3dnbW1hX3N0YXJ0X2tdCiAgICAgICAgICAgICAgICAgICAgKTsKICAgICAgICAgICAgICAgIH0KCiAgICAgICAgICAgICAgICBhc20gdm9sYXRpbGUoIndnbW1hLmNvbW1pdF9ncm91cC5zeW5jLmFsaWduZWQ7IiA6OjogIm1lbW9yeSIpOwogICAgICAgICAgICAgICAgYXNtIHZvbGF0aWxlKCJ3Z21tYS53YWl0X2dyb3VwLnN5bmMuYWxpZ25lZCAlCgogICAgICAgICAgICAgICAgaWYgKGNvbnN1bWVyX3RocmVhZF9pZCA8IENMVVNURVJfU0laRSkgewogICAgICAgICAgICAgICAgICAgIHB0eF9hcnJpdmVfYmFycmllcl9hY3Jvc3NfY2x1c3RlcigKICAgICAgICAgICAgICAgICAgICAgICAgJnF1ZXVlX2VtcHR5W3F1ZXVlX2lkeF0sCiAgICAgICAgICAgICAgICAgICAgICAgIGNvbnN1bWVyX3RocmVhZF9pZCwKICAgICAgICAgICAgICAgICAgICAgICAgMQogICAgICAgICAgICAgICAgICAgICk7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgIH0KCiAgICAgICAgICAgIHsKICAgICAgICAgICAgICAgIGFzbSB2b2xhdGlsZSgiY3AuYXN5bmMuYnVsay53YWl0X2dyb3VwLnJlYWQgMDtcbiIpOwogICAgICAgICAgICAgICAgaWYgKHRocmVhZF9sYW5lX2lkeF9uIDw9IDEpIHsKICAgICAgICAgICAgICAgICAgICB0aWxlc19zLmNfcGFja2VkWwogICAgICAgICAgICAgICAgICAgICAgICB0aHJlYWRfc3RvcmVfb2Zmc2V0X20gKyA4ICogdGhyZWFkX2xhbmVfaWR4X24KICAgICAgICAgICAgICAgICAgICBdWzBdID0gMDsKICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgIF9fc3luY3dhcnAoKTsKCiAgICAgICAgICAgICAgICAjcHJhZ21hIHVucm9sbAogICAgICAgICAgICAgICAgZm9yIChpbnQgcXVhZHJhbnRfc2xpY2VfbSA9IDA7CiAgICAgICAgICAgICAgICAgICAgIHF1YWRyYW50X3NsaWNlX20gPCA0OwogICAgICAgICAgICAgICAgICAgICBxdWFkcmFudF9zbGljZV9tICs9IDIpIHsKICAgICAgICAgICAgICAgICAgICBpbnQgcXVhZHJhbnRfc3RvcmVfb2Zmc2V0X20gPQogICAgICAgICAgICAgICAgICAgICAgICB0aHJlYWRfc3RvcmVfb2Zmc2V0X20gKyBxdWFkcmFudF9zbGljZV9tICogNDsKCiAgICAgICAgICAgICAgICAgICAgI3ByYWdtYSB1bnJvbGwKICAgICAgICAgICAgICAgICAgICBmb3IgKGludCB3Z21tYV9zbGljZV9uID0gMDsKICAgICAgICAgICAgICAgICAgICAgICAgIHdnbW1hX3NsaWNlX24gPCBUX24gLyAxNjsKICAgICAgICAgICAgICAgICAgICAgICAgICsrd2dtbWFfc2xpY2VfbikgewogICAgICAgICAgICAgICAgICAgICAgICBpbnQgcXVhZHJhbnRfc3RvcmVfb2Zmc2V0X24gPQogICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhyZWFkX3N0b3JlX29mZnNldF9uICsgd2dtbWFfc2xpY2VfbiAqIDE2OwoKICAgICAgICAgICAgICAgICAgICAgICAgI3ByYWdtYSB1bnJvbGwKICAgICAgICAgICAgICAgICAgICAgICAgZm9yIChpbnQgcXVhZHJhbnRfc2xpY2VfbiA9IDA7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcXVhZHJhbnRfc2xpY2VfbiA8IDg7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcXVhZHJhbnRfc2xpY2VfbiArPSA0KSB7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAjcHJhZ21hIHVucm9sbAogICAgICAgICAgICAgICAgICAgICAgICAgICAgZm9yIChpbnQgZWxlbWVudF9uID0gMDsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZWxlbWVudF9uIDwgMjsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgKytlbGVtZW50X24pIHsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAoCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIENfYWNjdW1bd2dtbWFfc2xpY2Vfbl1bCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBxdWFkcmFudF9zbGljZV9tCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICArIHF1YWRyYW50X3NsaWNlX24KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICsgZWxlbWVudF9uCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIF0gPiAwCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgKSB7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHVpbnQgY3VycmVudF9zdG9yZV9pZHggPSBfX252X2F0b21pY19mZXRjaF9hZGQoCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAmdGlsZXNfcy5jX3BhY2tlZFtxdWFkcmFudF9zdG9yZV9vZmZzZXRfbV1bMF0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAxdSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIF9fTlZfQVRPTUlDX1JFTEFYRUQsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBfX05WX1RIUkVBRF9TQ09QRV9CTE9DSwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICApOwoKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY29uc3QgdWludDMyX3QgcGFja2VkX3ZhbHVlID0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRpbGVfY29vcmRfbiAqIFRfbgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgKyBxdWFkcmFudF9zdG9yZV9vZmZzZXRfbgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgKyBxdWFkcmFudF9zbGljZV9uICogMgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgKyBlbGVtZW50X24KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHwgKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHN0YXRpY19jYXN0PHVpbnQzMl90PigKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgX19iZmxvYXQxNl9hc191c2hvcnQoCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBfX2Zsb2F0MmJmbG9hdDE2KAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIENfYWNjdW1bd2dtbWFfc2xpY2Vfbl1bCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHF1YWRyYW50X3NsaWNlX20KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgKyBxdWFkcmFudF9zbGljZV9uCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICsgZWxlbWVudF9uCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgXQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgKSA8PCAxNgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgKTsKCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmIGNvbnN0ZXhwciAoTE9PUF9PVkVSRkxPV19TVE9SQUdFKSB7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aWxlc19zLmNfcGFja2VkW3F1YWRyYW50X3N0b3JlX29mZnNldF9tXVsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAoY3VycmVudF9zdG9yZV9pZHggJiAoVF9uX2NvbXByZXNzZWQgLSAxKSkgKyAxCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBdID0gcGFja2VkX3ZhbHVlOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9IGVsc2UgewogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGlsZXNfcy5jX3BhY2tlZFtxdWFkcmFudF9zdG9yZV9vZmZzZXRfbV1bCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY3VycmVudF9zdG9yZV9pZHggKyAxCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBdID0gcGFja2VkX3ZhbHVlOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgfQoKICAgICAgICAgICAgICAgIGFzbSB2b2xhdGlsZSgiZmVuY2UucHJveHkuYXN5bmMuc2hhcmVkOjpjdGE7XG4iKTsKICAgICAgICAgICAgICAgIGFzbSB2b2xhdGlsZSgiYmFyLnN5bmMgMTAsIDI1NjtcbiIpOwoKICAgICAgICAgICAgICAgIGlmICh0aHJlYWRJZHgueCA9PSAxMjgpIHsKICAgICAgICAgICAgICAgICAgICBwdHhfc3RvcmVfdHJhbnNwb3NlZF90aWxlX3RtYV8zZDx1aW50MzJfdCwgVF9uX2NvbXByZXNzZWQ+KAogICAgICAgICAgICAgICAgICAgICAgICAmQ19wYWNrZWRfdG0sCiAgICAgICAgICAgICAgICAgICAgICAgICZ0aWxlc19zLmNfcGFja2VkWzBdWzBdLAogICAgICAgICAgICAgICAgICAgICAgICB0aWxlX2Nvb3JkX20gKiBUX20sCiAgICAgICAgICAgICAgICAgICAgICAgIHRpbGVfY29vcmRfbiAqIFRfbl9jb21wcmVzc2VkCiAgICAgICAgICAgICAgICAgICAgKTsKICAgICAgICAgICAgICAgICAgICBhc20gdm9sYXRpbGUoImNwLmFzeW5jLmJ1bGsuY29tbWl0X2dyb3VwO1xuIik7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgIH0KICAgICAgICB9CiAgICB9Cn0=)

1template <const int T\_m, const int T\_n, const int T\_k>

2struct Tiles

3{

4 alignas(128) \_\_nv\_bfloat16 a[T\_m][T\_k];

5 alignas(128) \_\_nv\_bfloat16 b[T\_n][T\_k];

6};

7

8template <

9 const int T\_m,

10 const int T\_n,

11 const int T\_k,

12 const int QUEUE\_SIZE,

13 const int T\_n\_compressed,

14 int PADDING = 4

15>

16struct SmemStorage

17{

18 Tiles<T\_m, T\_n, T\_k> queue[QUEUE\_SIZE];

19 alignas(128) uint32\_t c\_packed[T\_m][T\_n\_compressed + PADDING];

20};

21

22template <

23 const int T\_m,

24 const int T\_n,

25 const int T\_k,

26 const int CLUSTER\_DIM\_m,

27 const int CLUSTER\_DIM\_n,

28 const int QUEUE\_SIZE,

29 const int NUM\_ACTIVE\_SMs,

30 const int T\_n\_compressed,

31 const bool LOOP\_OVERFLOW\_STORAGE

32>

33\_\_global\_\_ \_\_launch\_bounds\_\_(NUM\_THREADS\_PER\_BLOCK)

34 \_\_cluster\_dims\_\_(CLUSTER\_DIM\_m \* CLUSTER\_DIM\_n, 1, 1)

35void mm\_wgmma\_nt\_kernel(

36 const CUtensorMap \_\_grid\_constant\_\_ A\_tm,

37 const CUtensorMap \_\_grid\_constant\_\_ B\_tm,

38 const CUtensorMap \_\_grid\_constant\_\_ C\_packed\_tm,

39 const int\* schedule\_gmem\_ptr,

40 const int schedule\_size\_per\_sm,

41 const int K

42)

43{

44 static\_assert(

45 (T\_m == 64 \* 2),

46 "Only T\_m == 128 supported"

47 );

48

49 constexpr int CLUSTER\_SIZE = CLUSTER\_DIM\_m \* CLUSTER\_DIM\_n;

50 extern \_\_shared\_\_ \_\_align\_\_(1024) unsigned char dynamic\_smem[];

51

52 int cluster\_idx;

53 asm ("mov.u32 %

54

55 int cluster\_lane\_m;

56 asm volatile("mov.u32 %

57

58 int cluster\_lane\_n = cluster\_lane\_m %

59 cluster\_lane\_m /= CLUSTER\_DIM\_n;

60

61 auto& tiles\_s =

62 \*reinterpret\_cast<

63 SmemStorage<T\_m, T\_n, T\_k, QUEUE\_SIZE, T\_n\_compressed>\*

64 >(dynamic\_smem);

65 int\* schedule\_s = reinterpret\_cast<int\*>(

66 dynamic\_smem

67 + sizeof(SmemStorage<T\_m, T\_n, T\_k, QUEUE\_SIZE, T\_n\_compressed>)

68 );

69

70 schedule\_gmem\_ptr += cluster\_idx \* schedule\_size\_per\_sm;

71 if (threadIdx.x < schedule\_size\_per\_sm) {

72 schedule\_s[threadIdx.x] = schedule\_gmem\_ptr[threadIdx.x];

73 }

74

75 \_\_syncthreads();

76

77 \_\_shared\_\_ \_\_align\_\_(8) uint64\_t queue\_full[QUEUE\_SIZE];

78 \_\_shared\_\_ \_\_align\_\_(8) uint64\_t queue\_empty[QUEUE\_SIZE];

79

80 if (threadIdx.x == 0) {

81 #pragma unroll

82 for (int queue\_idx = 0; queue\_idx < QUEUE\_SIZE; ++queue\_idx) {

83 ptx\_init\_smem\_barrier(&queue\_full[queue\_idx], 1);

84 ptx\_init\_smem\_barrier(&queue\_empty[queue\_idx], 2 \* CLUSTER\_SIZE);

85 }

86 }

87

88 asm volatile("barrier.cluster.arrive;\n" : :);

89 asm volatile("barrier.cluster.wait;\n" : :);

90

91 if (threadIdx.x < WARP\_GROUP\_SIZE) {

92 asm volatile("setmaxnreg.dec.sync.aligned.u32 %

93

94 if (threadIdx.x == 0) {

95 int queue\_idx = 0;

96 int queue\_phase = 0;

97 uint16\_t mask\_multicast\_m = 0;

98

99 if constexpr (CLUSTER\_DIM\_m > 1) {

100 for (int i = 0; i < CLUSTER\_DIM\_m; ++i) {

101 mask\_multicast\_m |= (1u << (i \* CLUSTER\_DIM\_n));

102 }

103 mask\_multicast\_m <<= cluster\_lane\_n;

104 }

105

106 uint16\_t mask\_multicast\_n;

107 if constexpr (CLUSTER\_DIM\_n > 1) {

108 mask\_multicast\_n =

109 ((1u << CLUSTER\_DIM\_n) - 1)

110 << (cluster\_lane\_m \* CLUSTER\_DIM\_n);

111 }

112

113 for (int schedule\_it = 0;

114 schedule\_it < schedule\_size\_per\_sm;

115 ++schedule\_it) {

116 const int packed\_tile = schedule\_s[schedule\_it];

117 if (packed\_tile == -1) {

118 break;

119 }

120

121 int tile\_coord\_m = packed\_tile >> 16;

122 int tile\_coord\_n = packed\_tile & 0xFFFF;

123

124 if constexpr (CLUSTER\_DIM\_n > 1) {

125 tile\_coord\_n \*= CLUSTER\_DIM\_n;

126 tile\_coord\_n += cluster\_lane\_n;

127 }

128 if constexpr (CLUSTER\_DIM\_m > 1) {

129 tile\_coord\_m \*= CLUSTER\_DIM\_m;

130 tile\_coord\_m += cluster\_lane\_m;

131 }

132

133 for (int tile\_start\_k = 0;

134 tile\_start\_k < K;

135 tile\_start\_k += T\_k, ++queue\_idx) {

136 if (queue\_idx == QUEUE\_SIZE) {

137 queue\_idx = 0;

138 queue\_phase ^= 1;

139 }

140

141 ptx\_wait\_barrier(&queue\_empty[queue\_idx], queue\_phase);

142 ptx\_arrive\_tx\_smem\_barrier(

143 &queue\_full[queue\_idx],

144 sizeof(tiles\_s.queue[queue\_idx].a)

145 + sizeof(tiles\_s.queue[queue\_idx].b)

146 );

147

148 if constexpr (CLUSTER\_DIM\_n > 1) {

149 if (cluster\_lane\_n == 0) {

150 ptx\_load\_tile\_tma\_multicast\_2d(

151 &tiles\_s.queue[queue\_idx].a[0][0],

152 &A\_tm,

153 tile\_coord\_m \* T\_m,

154 tile\_start\_k,

155 mask\_multicast\_n,

156 &queue\_full[queue\_idx]

157 );

158 }

159 } else {

160 ptx\_load\_tile\_tma\_2d(

161 &tiles\_s.queue[queue\_idx].a[0][0],

162 &A\_tm,

163 tile\_coord\_m \* T\_m,

164 tile\_start\_k,

165 &queue\_full[queue\_idx]

166 );

167 }

168

169 if constexpr (CLUSTER\_DIM\_m > 1) {

170 if (cluster\_lane\_m == 0) {

171 ptx\_load\_tile\_tma\_multicast\_2d(

172 &tiles\_s.queue[queue\_idx].b[0][0],

173 &B\_tm,

174 tile\_coord\_n \* T\_n,

175 tile\_start\_k,

176 mask\_multicast\_m,

177 &queue\_full[queue\_idx]

178 );

179 }

180 } else {

181 ptx\_load\_tile\_tma\_2d(

182 &tiles\_s.queue[queue\_idx].b[0][0],

183 &B\_tm,

184 tile\_coord\_n \* T\_n,

185 tile\_start\_k,

186 &queue\_full[queue\_idx]

187 );

188 }

189 }

190 }

191 }

192 } else {

193 asm volatile("setmaxnreg.inc.sync.aligned.u32 %

194 int queue\_idx = 0;

195 int queue\_phase = 0;

196 const int consumer\_warpgroup\_id =

197 (threadIdx.x - WARP\_GROUP\_SIZE) / WARP\_GROUP\_SIZE;

198 const int tile\_start\_m = consumer\_warpgroup\_id \* WGMMA\_m;

199 const int consumer\_thread\_id = threadIdx.x %

200 const uint thread\_lane\_idx\_n = (consumer\_thread\_id %

201

202 const int thread\_store\_offset\_m = (

203 tile\_start\_m

204 + consumer\_thread\_id / 32 \* 16

205 + (consumer\_thread\_id %

206 );

207 const int thread\_store\_offset\_n =

208 ((consumer\_thread\_id %

209

210 if (consumer\_thread\_id < CLUSTER\_SIZE) {

211 for (int queue\_idx = 0; queue\_idx < QUEUE\_SIZE; ++queue\_idx) {

212 ptx\_arrive\_barrier\_across\_cluster(

213 &queue\_empty[queue\_idx],

214 consumer\_thread\_id,

215 1

216 );

217 }

218 }

219

220 float C\_accum[T\_n/16][8];

221 for (int schedule\_it = 0;

222 schedule\_it < schedule\_size\_per\_sm;

223 ++schedule\_it) {

224 const int packed\_tile = schedule\_s[schedule\_it];

225 if (packed\_tile == -1) {

226 break;

227 }

228

229 int tile\_coord\_m = packed\_tile >> 16;

230 int tile\_coord\_n = packed\_tile & 0xFFFF;

231

232 if constexpr (CLUSTER\_DIM\_n > 1) {

233 tile\_coord\_n \*= CLUSTER\_DIM\_n;

234 tile\_coord\_n += cluster\_lane\_n;

235 }

236 if constexpr (CLUSTER\_DIM\_m > 1) {

237 tile\_coord\_m \*= CLUSTER\_DIM\_m;

238 tile\_coord\_m += cluster\_lane\_m;

239 }

240

241 if (queue\_idx == QUEUE\_SIZE) {

242 queue\_idx = 0;

243 queue\_phase ^= 1;

244 }

245

246 ptx\_wait\_barrier(&queue\_full[queue\_idx], queue\_phase);

247 asm volatile("wgmma.fence.sync.aligned;" ::: "memory");

248

249 wgmma<T\_n, 0, 1, 1, 0, 0>(

250 C\_accum,

251 &tiles\_s.queue[queue\_idx].a[tile\_start\_m][0],

252 &tiles\_s.queue[queue\_idx].b[0][0]

253 );

254

255 #pragma unroll

256 for (int wgmma\_start\_k = WGMMA\_k;

257 wgmma\_start\_k < T\_k;

258 wgmma\_start\_k += WGMMA\_k) {

259 wgmma<T\_n, 1, 1, 1, 0, 0>(

260 C\_accum,

261 &tiles\_s.queue[queue\_idx].a[tile\_start\_m][wgmma\_start\_k],

262 &tiles\_s.queue[queue\_idx].b[0][wgmma\_start\_k]

263 );

264 }

265

266 asm volatile("wgmma.commit\_group.sync.aligned;" ::: "memory");

267 asm volatile("wgmma.wait\_group.sync.aligned %

268

269 if (consumer\_thread\_id < CLUSTER\_SIZE) {

270 ptx\_arrive\_barrier\_across\_cluster(

271 &queue\_empty[queue\_idx],

272 consumer\_thread\_id,

273 1

274 );

275 }

276

277 queue\_idx++;

278

279 for (int tile\_idx\_k = 1;

280 tile\_idx\_k < K / T\_k;

281 ++tile\_idx\_k, ++queue\_idx) {

282 if (queue\_idx == QUEUE\_SIZE) {

283 queue\_idx = 0;

284 queue\_phase ^= 1;

285 }

286

287 ptx\_wait\_barrier(&queue\_full[queue\_idx], queue\_phase);

288 asm volatile("wgmma.fence.sync.aligned;" ::: "memory");

289

290 #pragma unroll

291 for (int wgmma\_start\_k = 0;

292 wgmma\_start\_k < T\_k;

293 wgmma\_start\_k += WGMMA\_k) {

294 wgmma<T\_n, 1, 1, 1, 0, 0>(

295 C\_accum,

296 &tiles\_s.queue[queue\_idx].a[tile\_start\_m][wgmma\_start\_k],

297 &tiles\_s.queue[queue\_idx].b[0][wgmma\_start\_k]

298 );

299 }

300

301 asm volatile("wgmma.commit\_group.sync.aligned;" ::: "memory");

302 asm volatile("wgmma.wait\_group.sync.aligned %

303

304 if (consumer\_thread\_id < CLUSTER\_SIZE) {

305 ptx\_arrive\_barrier\_across\_cluster(

306 &queue\_empty[queue\_idx],

307 consumer\_thread\_id,

308 1

309 );

310 }

311 }

312

313 {

314 asm volatile("cp.async.bulk.wait\_group.read 0;\n");

315 if (thread\_lane\_idx\_n <= 1) {

316 tiles\_s.c\_packed[

317 thread\_store\_offset\_m + 8 \* thread\_lane\_idx\_n

318 ][0] = 0;

319 }

320 \_\_syncwarp();

321

322 #pragma unroll

323 for (int quadrant\_slice\_m = 0;

324 quadrant\_slice\_m < 4;

325 quadrant\_slice\_m += 2) {

326 int quadrant\_store\_offset\_m =

327 thread\_store\_offset\_m + quadrant\_slice\_m \* 4;

328

329 #pragma unroll

330 for (int wgmma\_slice\_n = 0;

331 wgmma\_slice\_n < T\_n / 16;

332 ++wgmma\_slice\_n) {

333 int quadrant\_store\_offset\_n =

334 thread\_store\_offset\_n + wgmma\_slice\_n \* 16;

335

336 #pragma unroll

337 for (int quadrant\_slice\_n = 0;

338 quadrant\_slice\_n < 8;

339 quadrant\_slice\_n += 4) {

340 #pragma unroll

341 for (int element\_n = 0;

342 element\_n < 2;

343 ++element\_n) {

344 if (

345 C\_accum[wgmma\_slice\_n][

346 quadrant\_slice\_m

347 + quadrant\_slice\_n

348 + element\_n

349 ] > 0

350 ) {

351 uint current\_store\_idx = \_\_nv\_atomic\_fetch\_add(

352 &tiles\_s.c\_packed[quadrant\_store\_offset\_m][0],

353 1u,

354 \_\_NV\_ATOMIC\_RELAXED,

355 \_\_NV\_THREAD\_SCOPE\_BLOCK

356 );

357

358 const uint32\_t packed\_value =

359 tile\_coord\_n \* T\_n

360 + quadrant\_store\_offset\_n

361 + quadrant\_slice\_n \* 2

362 + element\_n

363 | (

364 static\_cast<uint32\_t>(

365 \_\_bfloat16\_as\_ushort(

366 \_\_float2bfloat16(

367 C\_accum[wgmma\_slice\_n][

368 quadrant\_slice\_m

369 + quadrant\_slice\_n

370 + element\_n

371 ]

372 )

373 )

374 ) << 16

375 );

376

377 if constexpr (LOOP\_OVERFLOW\_STORAGE) {

378 tiles\_s.c\_packed[quadrant\_store\_offset\_m][

379 (current\_store\_idx & (T\_n\_compressed - 1)) + 1

380 ] = packed\_value;

381 } else {

382 tiles\_s.c\_packed[quadrant\_store\_offset\_m][

383 current\_store\_idx + 1

384 ] = packed\_value;

385 }

386 }

387 }

388 }

389 }

390 }

391

392 asm volatile("fence.proxy.async.shared::cta;\n");

393 asm volatile("bar.sync 10, 256;\n");

394

395 if (threadIdx.x == 128) {

396 ptx\_store\_transposed\_tile\_tma\_3d<uint32\_t, T\_n\_compressed>(

397 &C\_packed\_tm,

398 &tiles\_s.c\_packed[0][0],

399 tile\_coord\_m \* T\_m,

400 tile\_coord\_n \* T\_n\_compressed

401 );

402 asm volatile("cp.async.bulk.commit\_group;\n");

403 }

404 }

405 }

406 }

407}

Listing 1: Efficient matrix multiplication with TwELL output storage.

In Figure LABEL:lst:appA:dense\_to\_twell, we provide code listings with the device code for our kernel implementing a custom matmul with our new TwELL storage, which we use to run the gate projection in our model. We omit device functions wrapping longer PTX injections for readability. As explained in Section [3](#S3.2 "3 Making Sparse LLMs Fast ‣ Sparser, Faster, Lighter Transformer Language Models") in the main text, this kernel executes an efficient tiled matrix multiplication, loading the dense input and the dense weight matrix and storing the output values using the Tensor Memory Accelerator (TMA) introduced with Hopper GPUs, while storing the output in the TwELL format during the kernel’s epilogue. The base kernel follows a persistent design with pipelined computation based on persistent cooperative kernels in CUTLASS [nvidia-cutlass] and open-source CUDA reproductions [hilbert-blog]. Unlike CUTLASS, the tile scheduler follows a pre-constructed ordering based on a Hilbert curve to maximize the reuse of the L2 cache [hilbert-paper, hilbert-blog]. In practice, we opt to pack the TwELL values hh, indices hIh\_{I}, and number of non-zeros hn​zh\_{nz} in a single 32-bit matrix in ℝM×N/C\mathbb{R}^{M\times N/C}. This is done by placing the number of non-zeros for each tile row in the first column and fitting the 16-bit TwELL value and index in the remaining entries. This design ensures strong locality and allows storing and loading the number of non-zeros together with the first 31 TwELL indices and values in a single coalesced access. While this loses a storage position, we set TwELL compression factors very conservatively for each sparsity level, making the occurrence of overflow practically impossible. For instance, we set the compression factor to 8 for our recommended sparsity regularization studied in our main results, with models ranging from 39-24 average non-zeros and an expected chance of overflow of the order of 10−3410^{-34}.
The TwELL conversion occurs when mapping the partial outputs of the asynchronous warpgroup level matmul instructions (WGMMA) from registers to shared memory via a fast CTA-scoped atomic operation with relaxed semantics. To avoid bank conflicts when resetting the number of non-zeros, we minimally pad the TwELL output with four extra elements in the last dimension. In this instance, we found our padding approach to work significantly better than swizzling, due to the lower register pressure introduced in the kernel’s epilogue. In an alternative implementation, we also explored a different packing layout, placing the number of non-zeros across the diagonal of the first 32-dimensional subtile to cover all memory banks, an approach we found brought minimal throughput improvements at the cost of extra complexity. We note that for the non-gated variant of our models, we use this same kernel to perform the up projection, as this layer is the one determining the overall sparsity pattern of the tile.

[⬇](data:text/plain;base64,dGVtcGxhdGUgPAogICAgY29uc3QgaW50IFRfbiwKICAgIGNvbnN0IGludCBUX25fY29tcHJlc3NlZCwKICAgIGNvbnN0IGludCBOVU1fVF9uLAogICAgY29uc3QgaW50IE9VVF9ESU0KPgpfX2dsb2JhbF9fIF9fbGF1bmNoX2JvdW5kc19fKFdBUlBfU0laRSkKdm9pZCBtbV90MmRfa2VybmVsKAogICAgY29uc3QgX19udl9iZmxvYXQxNiogSU5fZCwKICAgIGNvbnN0IHVpbnQzMl90KiBHQVRFX09VVF90d2VsbF9wYWNrZWRfZCwKICAgIGNvbnN0IF9fbnZfYmZsb2F0MTYqIFVQX3RyYW5zcG9zZWRfZCwKICAgIGNvbnN0IF9fbnZfYmZsb2F0MTYqIERPV05fZCwKICAgIF9fbnZfYmZsb2F0MTYqIE9VVF9kCikKewogICAgc3RhdGljX2Fzc2VydCgKICAgICAgICAoT1VUX0RJTSAlCiAgICAgICAgIk9VVF9ESU0gbXVzdCBiZSBkaXZpc2libGUgYnkgV0FSUF9TSVpFLiIKICAgICk7CiAgICBzdGF0aWNfYXNzZXJ0KFRfbl9jb21wcmVzc2VkID09IFdBUlBfU0laRSwKICAgICAgICAiV2FycC1zeW5jIFR3RUxMLXRvLWRlbnNlIGFzc3VtZXMgYSAzMi13aWRlIGNvbXByZXNzZWQgdGlsZS4iKTsKCiAgICBjb25zdGV4cHIgaW50IE5VTV9MT0FEX0lURVJTID0gT1VUX0RJTSAvIFNUUklERV84eFdBUlA7CiAgICBmbG9hdCBPVVRfYWNjdW1bTlVNX0xPQURfSVRFUlNdWzhdID0gezAuMGZ9OwogICAgX19udl9iZmxvYXQxNjIgSU5fY2FjaGVkW05VTV9MT0FEX0lURVJTXVs0XTsKCiAgICBJTl9kICs9IGJsb2NrSWR4LnggKiBPVVRfRElNICsgdGhyZWFkSWR4LnggKiA4OwogICAgR0FURV9PVVRfdHdlbGxfcGFja2VkX2QgKz0gKAogICAgICAgIGJsb2NrSWR4LnggKiBUX25fY29tcHJlc3NlZCAqIE5VTV9UX24gKyB0aHJlYWRJZHgueAogICAgKTsKICAgIFVQX3RyYW5zcG9zZWRfZCArPSB0aHJlYWRJZHgueCAqIDg7CiAgICBET1dOX2QgKz0gdGhyZWFkSWR4LnggKiA4OwogICAgT1VUX2QgKz0gYmxvY2tJZHgueCAqIE9VVF9ESU0gKyB0aHJlYWRJZHgueCAqIDg7CgogICAgI3ByYWdtYSB1bnJvbGwKICAgIGZvciAoaW50IGl0ZXJfaWR4ID0gMDsgaXRlcl9pZHggPCBOVU1fTE9BRF9JVEVSUzsgKytpdGVyX2lkeCkgewogICAgICAgICpyZWludGVycHJldF9jYXN0PHVpbnQ0Kj4oJklOX2NhY2hlZFtpdGVyX2lkeF1bMF0pID0KICAgICAgICAgICAgKnJlaW50ZXJwcmV0X2Nhc3Q8Y29uc3QgdWludDQqPigKICAgICAgICAgICAgICAgIElOX2QgKyBpdGVyX2lkeCAqIFNUUklERV84eFdBUlAKICAgICAgICAgICAgKTsKICAgIH0KCiAgICAjcHJhZ21hIHVucm9sbCAxCiAgICBmb3IgKGludCB0aWxlX2lkeCA9IDA7IHRpbGVfaWR4IDwgTlVNX1RfbjsgKyt0aWxlX2lkeCkgewogICAgICAgIGNvbnN0IGludCBsYW5lX3RpbGVfcmVnaXN0ZXIgPQogICAgICAgICAgICBHQVRFX09VVF90d2VsbF9wYWNrZWRfZFt0aWxlX2lkeCAqIFRfbl9jb21wcmVzc2VkXTsKICAgICAgICBjb25zdCBpbnQgbnVtX25vbnplcm9zID0KICAgICAgICAgICAgX19zaGZsX3N5bmMoMHhGRkZGRkZGRnUsIGxhbmVfdGlsZV9yZWdpc3RlciwgMCk7CgogICAgICAgICNwcmFnbWEgdW5yb2xsIDEKICAgICAgICBmb3IgKGludCBpZHggPSAxOyBpZHggPCBudW1fbm9uemVyb3MgKyAxOyArK2lkeCkgewogICAgICAgICAgICBjb25zdCB1aW50MzJfdCBjb21wcmVzc2VkX2lkeF9iZjE2ID0KICAgICAgICAgICAgICAgIF9fc2hmbF9zeW5jKDB4RkZGRkZGRkZ1LCBsYW5lX3RpbGVfcmVnaXN0ZXIsIGlkeCk7CiAgICAgICAgICAgIGNvbnN0IHVpbnQzMl90IG5vbnplcm9faWR4ID0gY29tcHJlc3NlZF9pZHhfYmYxNiAmIDB4RkZGRnU7CgogICAgICAgICAgICBmbG9hdCBVUF9PVVRfYWNjdW0gPSAwLjBmOwoKICAgICAgICAgICAgI3ByYWdtYSB1bnJvbGwKICAgICAgICAgICAgZm9yIChpbnQgaXRlcl9pZHggPSAwOyBpdGVyX2lkeCA8IE5VTV9MT0FEX0lURVJTOyArK2l0ZXJfaWR4KSB7CiAgICAgICAgICAgICAgICBjb25zdCB1aW50NCBwYWNrZWRfYmZsb2F0c194OCA9CiAgICAgICAgICAgICAgICAgICAgKnJlaW50ZXJwcmV0X2Nhc3Q8Y29uc3QgdWludDQqPigKICAgICAgICAgICAgICAgICAgICAgICAgVVBfdHJhbnNwb3NlZF9kCiAgICAgICAgICAgICAgICAgICAgICAgICsgbm9uemVyb19pZHggKiBPVVRfRElNCiAgICAgICAgICAgICAgICAgICAgICAgICsgaXRlcl9pZHggKiBTVFJJREVfOHhXQVJQCiAgICAgICAgICAgICAgICAgICAgKTsKICAgICAgICAgICAgICAgIGNvbnN0IF9fbnZfYmZsb2F0MTYyIHBhY2tlZF9iZmxvYXRzXzEgPQogICAgICAgICAgICAgICAgICAgICpyZWludGVycHJldF9jYXN0PGNvbnN0IF9fbnZfYmZsb2F0MTYyKj4oCiAgICAgICAgICAgICAgICAgICAgICAgICZwYWNrZWRfYmZsb2F0c194OC54CiAgICAgICAgICAgICAgICAgICAgKTsKICAgICAgICAgICAgICAgIF9fbnZfYmZsb2F0MTYyIHNjYWxlZF9iZmxvYXRzXzEgPQogICAgICAgICAgICAgICAgICAgIF9faG11bDIoSU5fY2FjaGVkW2l0ZXJfaWR4XVswXSwgcGFja2VkX2JmbG9hdHNfMSk7CiAgICAgICAgICAgICAgICBmbG9hdDIgc2NhbGVkX2Zsb2F0c18xID0gX19iZmxvYXQxNjIyZmxvYXQyKHNjYWxlZF9iZmxvYXRzXzEpOwogICAgICAgICAgICAgICAgVVBfT1VUX2FjY3VtICs9IHNjYWxlZF9mbG9hdHNfMS54ICsgc2NhbGVkX2Zsb2F0c18xLnk7CgogICAgICAgICAgICAgICAgY29uc3QgX19udl9iZmxvYXQxNjIgcGFja2VkX2JmbG9hdHNfMiA9CiAgICAgICAgICAgICAgICAgICAgKnJlaW50ZXJwcmV0X2Nhc3Q8Y29uc3QgX19udl9iZmxvYXQxNjIqPigKICAgICAgICAgICAgICAgICAgICAgICAgJnBhY2tlZF9iZmxvYXRzX3g4LnkKICAgICAgICAgICAgICAgICAgICApOwogICAgICAgICAgICAgICAgX19udl9iZmxvYXQxNjIgc2NhbGVkX2JmbG9hdHNfMiA9CiAgICAgICAgICAgICAgICAgICAgX19obXVsMihJTl9jYWNoZWRbaXRlcl9pZHhdWzFdLCBwYWNrZWRfYmZsb2F0c18yKTsKICAgICAgICAgICAgICAgIGZsb2F0MiBzY2FsZWRfZmxvYXRzXzIgPSBfX2JmbG9hdDE2MjJmbG9hdDIoc2NhbGVkX2JmbG9hdHNfMik7CiAgICAgICAgICAgICAgICBVUF9PVVRfYWNjdW0gKz0gc2NhbGVkX2Zsb2F0c18yLnggKyBzY2FsZWRfZmxvYXRzXzIueTsKCiAgICAgICAgICAgICAgICBjb25zdCBfX252X2JmbG9hdDE2MiBwYWNrZWRfYmZsb2F0c18zID0KICAgICAgICAgICAgICAgICAgICAqcmVpbnRlcnByZXRfY2FzdDxjb25zdCBfX252X2JmbG9hdDE2Mio+KAogICAgICAgICAgICAgICAgICAgICAgICAmcGFja2VkX2JmbG9hdHNfeDguegogICAgICAgICAgICAgICAgICAgICk7CiAgICAgICAgICAgICAgICBfX252X2JmbG9hdDE2MiBzY2FsZWRfYmZsb2F0c18zID0KICAgICAgICAgICAgICAgICAgICBfX2htdWwyKElOX2NhY2hlZFtpdGVyX2lkeF1bMl0sIHBhY2tlZF9iZmxvYXRzXzMpOwogICAgICAgICAgICAgICAgZmxvYXQyIHNjYWxlZF9mbG9hdHNfMyA9IF9fYmZsb2F0MTYyMmZsb2F0MihzY2FsZWRfYmZsb2F0c18zKTsKICAgICAgICAgICAgICAgIFVQX09VVF9hY2N1bSArPSBzY2FsZWRfZmxvYXRzXzMueCArIHNjYWxlZF9mbG9hdHNfMy55OwoKICAgICAgICAgICAgICAgIGNvbnN0IF9fbnZfYmZsb2F0MTYyIHBhY2tlZF9iZmxvYXRzXzQgPQogICAgICAgICAgICAgICAgICAgICpyZWludGVycHJldF9jYXN0PGNvbnN0IF9fbnZfYmZsb2F0MTYyKj4oCiAgICAgICAgICAgICAgICAgICAgICAgICZwYWNrZWRfYmZsb2F0c194OC53CiAgICAgICAgICAgICAgICAgICAgKTsKICAgICAgICAgICAgICAgIF9fbnZfYmZsb2F0MTYyIHNjYWxlZF9iZmxvYXRzXzQgPQogICAgICAgICAgICAgICAgICAgIF9faG11bDIoSU5fY2FjaGVkW2l0ZXJfaWR4XVszXSwgcGFja2VkX2JmbG9hdHNfNCk7CiAgICAgICAgICAgICAgICBmbG9hdDIgc2NhbGVkX2Zsb2F0c180ID0gX19iZmxvYXQxNjIyZmxvYXQyKHNjYWxlZF9iZmxvYXRzXzQpOwogICAgICAgICAgICAgICAgVVBfT1VUX2FjY3VtICs9IHNjYWxlZF9mbG9hdHNfNC54ICsgc2NhbGVkX2Zsb2F0c180Lnk7CiAgICAgICAgICAgIH0KCiAgICAgICAgICAgICNwcmFnbWEgdW5yb2xsCiAgICAgICAgICAgIGZvciAoaW50IGJ1dHRlcmZseV9zdHJpZGUgPSBXQVJQX1NJWkUgLyAyOwogICAgICAgICAgICAgICAgIGJ1dHRlcmZseV9zdHJpZGUgPiAwOwogICAgICAgICAgICAgICAgIGJ1dHRlcmZseV9zdHJpZGUgLz0gMikgewogICAgICAgICAgICAgICAgVVBfT1VUX2FjY3VtICs9IF9fc2hmbF94b3Jfc3luYygKICAgICAgICAgICAgICAgICAgICAweEZGRkZGRkZGdSwKICAgICAgICAgICAgICAgICAgICBVUF9PVVRfYWNjdW0sCiAgICAgICAgICAgICAgICAgICAgYnV0dGVyZmx5X3N0cmlkZQogICAgICAgICAgICAgICAgKTsKICAgICAgICAgICAgfQoKICAgICAgICAgICAgY29uc3QgX19udl9iZmxvYXQxNjIgbm9uemVyb19mZWF0dXJlID0KICAgICAgICAgICAgICAgIF9fYmZsb2F0MTYyYmZsb2F0MTYyKAogICAgICAgICAgICAgICAgICAgIF9faG11bCgKICAgICAgICAgICAgICAgICAgICAgICAgcmVpbnRlcnByZXRfY2FzdDxjb25zdCBfX252X2JmbG9hdDE2Kj4oCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAmY29tcHJlc3NlZF9pZHhfYmYxNgogICAgICAgICAgICAgICAgICAgICAgICApWzFdLAogICAgICAgICAgICAgICAgICAgICAgICBfX2Zsb2F0MmJmbG9hdDE2X3JuKFVQX09VVF9hY2N1bSkKICAgICAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICApOwoKICAgICAgICAgICAgI3ByYWdtYSB1bnJvbGwKICAgICAgICAgICAgZm9yIChpbnQgaXRlcl9pZHggPSAwOyBpdGVyX2lkeCA8IE5VTV9MT0FEX0lURVJTOyArK2l0ZXJfaWR4KSB7CiAgICAgICAgICAgICAgICBjb25zdCB1aW50NCBwYWNrZWRfYmZsb2F0c194OCA9CiAgICAgICAgICAgICAgICAgICAgKnJlaW50ZXJwcmV0X2Nhc3Q8Y29uc3QgdWludDQqPigKICAgICAgICAgICAgICAgICAgICAgICAgRE9XTl9kCiAgICAgICAgICAgICAgICAgICAgICAgICsgbm9uemVyb19pZHggKiBPVVRfRElNCiAgICAgICAgICAgICAgICAgICAgICAgICsgaXRlcl9pZHggKiBTVFJJREVfOHhXQVJQCiAgICAgICAgICAgICAgICAgICAgKTsKICAgICAgICAgICAgICAgIGNvbnN0IF9fbnZfYmZsb2F0MTYyIHBhY2tlZF9iZmxvYXRzXzEgPQogICAgICAgICAgICAgICAgICAgICpyZWludGVycHJldF9jYXN0PGNvbnN0IF9fbnZfYmZsb2F0MTYyKj4oCiAgICAgICAgICAgICAgICAgICAgICAgICZwYWNrZWRfYmZsb2F0c194OC54CiAgICAgICAgICAgICAgICAgICAgKTsKICAgICAgICAgICAgICAgIF9fbnZfYmZsb2F0MTYyIHNjYWxlZF9iZmxvYXRzXzEgPQogICAgICAgICAgICAgICAgICAgIF9faG11bDIobm9uemVyb19mZWF0dXJlLCBwYWNrZWRfYmZsb2F0c18xKTsKICAgICAgICAgICAgICAgIGZsb2F0MiBzY2FsZWRfZmxvYXRzXzEgPSBfX2JmbG9hdDE2MjJmbG9hdDIoc2NhbGVkX2JmbG9hdHNfMSk7CiAgICAgICAgICAgICAgICBPVVRfYWNjdW1baXRlcl9pZHhdWzBdICs9IHNjYWxlZF9mbG9hdHNfMS54OwogICAgICAgICAgICAgICAgT1VUX2FjY3VtW2l0ZXJfaWR4XVsxXSArPSBzY2FsZWRfZmxvYXRzXzEueTsKCiAgICAgICAgICAgICAgICBjb25zdCBfX252X2JmbG9hdDE2MiBwYWNrZWRfYmZsb2F0c18yID0KICAgICAgICAgICAgICAgICAgICAqcmVpbnRlcnByZXRfY2FzdDxjb25zdCBfX252X2JmbG9hdDE2Mio+KAogICAgICAgICAgICAgICAgICAgICAgICAmcGFja2VkX2JmbG9hdHNfeDgueQogICAgICAgICAgICAgICAgICAgICk7CiAgICAgICAgICAgICAgICBfX252X2JmbG9hdDE2MiBzY2FsZWRfYmZsb2F0c18yID0KICAgICAgICAgICAgICAgICAgICBfX2htdWwyKG5vbnplcm9fZmVhdHVyZSwgcGFja2VkX2JmbG9hdHNfMik7CiAgICAgICAgICAgICAgICBmbG9hdDIgc2NhbGVkX2Zsb2F0c18yID0gX19iZmxvYXQxNjIyZmxvYXQyKHNjYWxlZF9iZmxvYXRzXzIpOwogICAgICAgICAgICAgICAgT1VUX2FjY3VtW2l0ZXJfaWR4XVsyXSArPSBzY2FsZWRfZmxvYXRzXzIueDsKICAgICAgICAgICAgICAgIE9VVF9hY2N1bVtpdGVyX2lkeF1bM10gKz0gc2NhbGVkX2Zsb2F0c18yLnk7CgogICAgICAgICAgICAgICAgY29uc3QgX19udl9iZmxvYXQxNjIgcGFja2VkX2JmbG9hdHNfMyA9CiAgICAgICAgICAgICAgICAgICAgKnJlaW50ZXJwcmV0X2Nhc3Q8Y29uc3QgX19udl9iZmxvYXQxNjIqPigKICAgICAgICAgICAgICAgICAgICAgICAgJnBhY2tlZF9iZmxvYXRzX3g4LnoKICAgICAgICAgICAgICAgICAgICApOwogICAgICAgICAgICAgICAgX19udl9iZmxvYXQxNjIgc2NhbGVkX2JmbG9hdHNfMyA9CiAgICAgICAgICAgICAgICAgICAgX19obXVsMihub256ZXJvX2ZlYXR1cmUsIHBhY2tlZF9iZmxvYXRzXzMpOwogICAgICAgICAgICAgICAgZmxvYXQyIHNjYWxlZF9mbG9hdHNfMyA9IF9fYmZsb2F0MTYyMmZsb2F0MihzY2FsZWRfYmZsb2F0c18zKTsKICAgICAgICAgICAgICAgIE9VVF9hY2N1bVtpdGVyX2lkeF1bNF0gKz0gc2NhbGVkX2Zsb2F0c18zLng7CiAgICAgICAgICAgICAgICBPVVRfYWNjdW1baXRlcl9pZHhdWzVdICs9IHNjYWxlZF9mbG9hdHNfMy55OwoKICAgICAgICAgICAgICAgIGNvbnN0IF9fbnZfYmZsb2F0MTYyIHBhY2tlZF9iZmxvYXRzXzQgPQogICAgICAgICAgICAgICAgICAgICpyZWludGVycHJldF9jYXN0PGNvbnN0IF9fbnZfYmZsb2F0MTYyKj4oCiAgICAgICAgICAgICAgICAgICAgICAgICZwYWNrZWRfYmZsb2F0c194OC53CiAgICAgICAgICAgICAgICAgICAgKTsKICAgICAgICAgICAgICAgIF9fbnZfYmZsb2F0MTYyIHNjYWxlZF9iZmxvYXRzXzQgPQogICAgICAgICAgICAgICAgICAgIF9faG11bDIobm9uemVyb19mZWF0dXJlLCBwYWNrZWRfYmZsb2F0c180KTsKICAgICAgICAgICAgICAgIGZsb2F0MiBzY2FsZWRfZmxvYXRzXzQgPSBfX2JmbG9hdDE2MjJmbG9hdDIoc2NhbGVkX2JmbG9hdHNfNCk7CiAgICAgICAgICAgICAgICBPVVRfYWNjdW1baXRlcl9pZHhdWzZdICs9IHNjYWxlZF9mbG9hdHNfNC54OwogICAgICAgICAgICAgICAgT1VUX2FjY3VtW2l0ZXJfaWR4XVs3XSArPSBzY2FsZWRfZmxvYXRzXzQueTsKICAgICAgICAgICAgfQogICAgICAgIH0KICAgIH0KCiAgICAjcHJhZ21hIHVucm9sbAogICAgZm9yIChpbnQgaXRlcl9pZHggPSAwOyBpdGVyX2lkeCA8IE5VTV9MT0FEX0lURVJTOyArK2l0ZXJfaWR4KSB7CiAgICAgICAgX19udl9iZmxvYXQxNjIgIF9fYWxpZ25fXyg4KSBwYWNrZWRfYmZsb2F0c194OFs0XTsKICAgICAgICBwYWNrZWRfYmZsb2F0c194OFswXSA9IF9fZmxvYXRzMmJmbG9hdDE2Ml9ybigKICAgICAgICAgICAgT1VUX2FjY3VtW2l0ZXJfaWR4XVswXSwgT1VUX2FjY3VtW2l0ZXJfaWR4XVsxXQogICAgICAgICk7CiAgICAgICAgcGFja2VkX2JmbG9hdHNfeDhbMV0gPSBfX2Zsb2F0czJiZmxvYXQxNjJfcm4oCiAgICAgICAgICAgIE9VVF9hY2N1bVtpdGVyX2lkeF1bMl0sIE9VVF9hY2N1bVtpdGVyX2lkeF1bM10KICAgICAgICApOwogICAgICAgIHBhY2tlZF9iZmxvYXRzX3g4WzJdID0gX19mbG9hdHMyYmZsb2F0MTYyX3JuKAogICAgICAgICAgICBPVVRfYWNjdW1baXRlcl9pZHhdWzRdLCBPVVRfYWNjdW1baXRlcl9pZHhdWzVdCiAgICAgICAgKTsKICAgICAgICBwYWNrZWRfYmZsb2F0c194OFszXSA9IF9fZmxvYXRzMmJmbG9hdDE2Ml9ybigKICAgICAgICAgICAgT1VUX2FjY3VtW2l0ZXJfaWR4XVs2XSwgT1VUX2FjY3VtW2l0ZXJfaWR4XVs3XQogICAgICAgICk7CgogICAgICAgICpyZWludGVycHJldF9jYXN0PHVpbnQ0Kj4oT1VUX2QgKyBpdGVyX2lkeCAqIFNUUklERV84eFdBUlApID0KICAgICAgICAgICAgKnJlaW50ZXJwcmV0X2Nhc3Q8dWludDQqPihwYWNrZWRfYmZsb2F0c194OCk7CiAgICB9Cn0K)

1template <

2 const int T\_n,

3 const int T\_n\_compressed,

4 const int NUM\_T\_n,

5 const int OUT\_DIM

6>

7\_\_global\_\_ \_\_launch\_bounds\_\_(WARP\_SIZE)

8void mm\_t2d\_kernel(

9 const \_\_nv\_bfloat16\* IN\_d,

10 const uint32\_t\* GATE\_OUT\_twell\_packed\_d,

11 const \_\_nv\_bfloat16\* UP\_transposed\_d,

12 const \_\_nv\_bfloat16\* DOWN\_d,

13 \_\_nv\_bfloat16\* OUT\_d

14)

15{

16 static\_assert(

17 (OUT\_DIM %

18 "OUT\_DIM must be divisible by WARP\_SIZE."

19 );

20 static\_assert(T\_n\_compressed == WARP\_SIZE,

21 "Warp-sync TwELL-to-dense assumes a 32-wide compressed tile.");

22

23 constexpr int NUM\_LOAD\_ITERS = OUT\_DIM / STRIDE\_8xWARP;

24 float OUT\_accum[NUM\_LOAD\_ITERS][8] = {0.0f};

25 \_\_nv\_bfloat162 IN\_cached[NUM\_LOAD\_ITERS][4];

26

27 IN\_d += blockIdx.x \* OUT\_DIM + threadIdx.x \* 8;

28 GATE\_OUT\_twell\_packed\_d += (

29 blockIdx.x \* T\_n\_compressed \* NUM\_T\_n + threadIdx.x

30 );

31 UP\_transposed\_d += threadIdx.x \* 8;

32 DOWN\_d += threadIdx.x \* 8;

33 OUT\_d += blockIdx.x \* OUT\_DIM + threadIdx.x \* 8;

34

35 #pragma unroll

36 for (int iter\_idx = 0; iter\_idx < NUM\_LOAD\_ITERS; ++iter\_idx) {

37 \*reinterpret\_cast<uint4\*>(&IN\_cached[iter\_idx][0]) =

38 \*reinterpret\_cast<const uint4\*>(

39 IN\_d + iter\_idx \* STRIDE\_8xWARP

40 );

41 }

42

43 #pragma unroll 1

44 for (int tile\_idx = 0; tile\_idx < NUM\_T\_n; ++tile\_idx) {

45 const int lane\_tile\_register =

46 GATE\_OUT\_twell\_packed\_d[tile\_idx \* T\_n\_compressed];

47 const int num\_nonzeros =

48 \_\_shfl\_sync(0xFFFFFFFFu, lane\_tile\_register, 0);

49

50 #pragma unroll 1

51 for (int idx = 1; idx < num\_nonzeros + 1; ++idx) {

52 const uint32\_t compressed\_idx\_bf16 =

53 \_\_shfl\_sync(0xFFFFFFFFu, lane\_tile\_register, idx);

54 const uint32\_t nonzero\_idx = compressed\_idx\_bf16 & 0xFFFFu;

55

56 float UP\_OUT\_accum = 0.0f;

57

58 #pragma unroll

59 for (int iter\_idx = 0; iter\_idx < NUM\_LOAD\_ITERS; ++iter\_idx) {

60 const uint4 packed\_bfloats\_x8 =

61 \*reinterpret\_cast<const uint4\*>(

62 UP\_transposed\_d

63 + nonzero\_idx \* OUT\_DIM

64 + iter\_idx \* STRIDE\_8xWARP

65 );

66 const \_\_nv\_bfloat162 packed\_bfloats\_1 =

67 \*reinterpret\_cast<const \_\_nv\_bfloat162\*>(

68 &packed\_bfloats\_x8.x

69 );

70 \_\_nv\_bfloat162 scaled\_bfloats\_1 =

71 \_\_hmul2(IN\_cached[iter\_idx][0], packed\_bfloats\_1);

72 float2 scaled\_floats\_1 = \_\_bfloat1622float2(scaled\_bfloats\_1);

73 UP\_OUT\_accum += scaled\_floats\_1.x + scaled\_floats\_1.y;

74

75 const \_\_nv\_bfloat162 packed\_bfloats\_2 =

76 \*reinterpret\_cast<const \_\_nv\_bfloat162\*>(

77 &packed\_bfloats\_x8.y

78 );

79 \_\_nv\_bfloat162 scaled\_bfloats\_2 =

80 \_\_hmul2(IN\_cached[iter\_idx][1], packed\_bfloats\_2);

81 float2 scaled\_floats\_2 = \_\_bfloat1622float2(scaled\_bfloats\_2);

82 UP\_OUT\_accum += scaled\_floats\_2.x + scaled\_floats\_2.y;

83

84 const \_\_nv\_bfloat162 packed\_bfloats\_3 =

85 \*reinterpret\_cast<const \_\_nv\_bfloat162\*>(

86 &packed\_bfloats\_x8.z

87 );

88 \_\_nv\_bfloat162 scaled\_bfloats\_3 =

89 \_\_hmul2(IN\_cached[iter\_idx][2], packed\_bfloats\_3);

90 float2 scaled\_floats\_3 = \_\_bfloat1622float2(scaled\_bfloats\_3);

91 UP\_OUT\_accum += scaled\_floats\_3.x + scaled\_floats\_3.y;

92

93 const \_\_nv\_bfloat162 packed\_bfloats\_4 =

94 \*reinterpret\_cast<const \_\_nv\_bfloat162\*>(

95 &packed\_bfloats\_x8.w

96 );

97 \_\_nv\_bfloat162 scaled\_bfloats\_4 =

98 \_\_hmul2(IN\_cached[iter\_idx][3], packed\_bfloats\_4);

99 float2 scaled\_floats\_4 = \_\_bfloat1622float2(scaled\_bfloats\_4);

100 UP\_OUT\_accum += scaled\_floats\_4.x + scaled\_floats\_4.y;

101 }

102

103 #pragma unroll

104 for (int butterfly\_stride = WARP\_SIZE / 2;

105 butterfly\_stride > 0;

106 butterfly\_stride /= 2) {

107 UP\_OUT\_accum += \_\_shfl\_xor\_sync(

108 0xFFFFFFFFu,

109 UP\_OUT\_accum,

110 butterfly\_stride

111 );

112 }

113

114 const \_\_nv\_bfloat162 nonzero\_feature =

115 \_\_bfloat162bfloat162(

116 \_\_hmul(

117 reinterpret\_cast<const \_\_nv\_bfloat16\*>(

118 &compressed\_idx\_bf16

119 )[1],

120 \_\_float2bfloat16\_rn(UP\_OUT\_accum)

121 )

122 );

123

124 #pragma unroll

125 for (int iter\_idx = 0; iter\_idx < NUM\_LOAD\_ITERS; ++iter\_idx) {

126 const uint4 packed\_bfloats\_x8 =

127 \*reinterpret\_cast<const uint4\*>(

128 DOWN\_d

129 + nonzero\_idx \* OUT\_DIM

130 + iter\_idx \* STRIDE\_8xWARP

131 );

132 const \_\_nv\_bfloat162 packed\_bfloats\_1 =

133 \*reinterpret\_cast<const \_\_nv\_bfloat162\*>(

134 &packed\_bfloats\_x8.x

135 );

136 \_\_nv\_bfloat162 scaled\_bfloats\_1 =

137 \_\_hmul2(nonzero\_feature, packed\_bfloats\_1);

138 float2 scaled\_floats\_1 = \_\_bfloat1622float2(scaled\_bfloats\_1);

139 OUT\_accum[iter\_idx][0] += scaled\_floats\_1.x;

140 OUT\_accum[iter\_idx][1] += scaled\_floats\_1.y;

141

142 const \_\_nv\_bfloat162 packed\_bfloats\_2 =

143 \*reinterpret\_cast<const \_\_nv\_bfloat162\*>(

144 &packed\_bfloats\_x8.y

145 );

146 \_\_nv\_bfloat162 scaled\_bfloats\_2 =

147 \_\_hmul2(nonzero\_feature, packed\_bfloats\_2);

148 float2 scaled\_floats\_2 = \_\_bfloat1622float2(scaled\_bfloats\_2);

149 OUT\_accum[iter\_idx][2] += scaled\_floats\_2.x;

150 OUT\_accum[iter\_idx][3] += scaled\_floats\_2.y;

151

152 const \_\_nv\_bfloat162 packed\_bfloats\_3 =

153 \*reinterpret\_cast<const \_\_nv\_bfloat162\*>(

154 &packed\_bfloats\_x8.z

155 );

156 \_\_nv\_bfloat162 scaled\_bfloats\_3 =

157 \_\_hmul2(nonzero\_feature, packed\_bfloats\_3);

158 float2 scaled\_floats\_3 = \_\_bfloat1622float2(scaled\_bfloats\_3);

159 OUT\_accum[iter\_idx][4] += scaled\_floats\_3.x;

160 OUT\_accum[iter\_idx][5] += scaled\_floats\_3.y;

161

162 const \_\_nv\_bfloat162 packed\_bfloats\_4 =

163 \*reinterpret\_cast<const \_\_nv\_bfloat162\*>(

164 &packed\_bfloats\_x8.w

165 );

166 \_\_nv\_bfloat162 scaled\_bfloats\_4 =

167 \_\_hmul2(nonzero\_feature, packed\_bfloats\_4);

168 float2 scaled\_floats\_4 = \_\_bfloat1622float2(scaled\_bfloats\_4);

169 OUT\_accum[iter\_idx][6] += scaled\_floats\_4.x;

170 OUT\_accum[iter\_idx][7] += scaled\_floats\_4.y;

171 }

172 }

173 }

174

175 #pragma unroll

176 for (int iter\_idx = 0; iter\_idx < NUM\_LOAD\_ITERS; ++iter\_idx) {

177 \_\_nv\_bfloat162 \_\_align\_\_(8) packed\_bfloats\_x8[4];

178 packed\_bfloats\_x8[0] = \_\_floats2bfloat162\_rn(

179 OUT\_accum[iter\_idx][0], OUT\_accum[iter\_idx][1]

180 );

181 packed\_bfloats\_x8[1] = \_\_floats2bfloat162\_rn(

182 OUT\_accum[iter\_idx][2], OUT\_accum[iter\_idx][3]

183 );

184 packed\_bfloats\_x8[2] = \_\_floats2bfloat162\_rn(

185 OUT\_accum[iter\_idx][4], OUT\_accum[iter\_idx][5]

186 );

187 packed\_bfloats\_x8[3] = \_\_floats2bfloat162\_rn(

188 OUT\_accum[iter\_idx][6], OUT\_accum[iter\_idx][7]

189 );

190

191 \*reinterpret\_cast<uint4\*>(OUT\_d + iter\_idx \* STRIDE\_8xWARP) =

192 \*reinterpret\_cast<uint4\*>(packed\_bfloats\_x8);

193 }

194}

Listing 2: Fused up and downprojection leveraging gate projections in TwELL format.

In Figure LABEL:lst:appA:twell\_fused\_up\_down, we provide code listings with the device code for our kernel implementing the custom fused up and down projection kernel that leverages the gate projections stored in the TwELL format. As explained in Section [3](#S3.2 "3 Making Sparse LLMs Fast ‣ Sparser, Faster, Lighter Transformer Language Models") in the main text, this kernel is launched on a grid of warp-sized CTAs and fuses the two operations by keeping in memory the input dense feature row and an accumulator. Then, iterating first statically through the TwELL tiles and then dynamically through the number of non-zeros in each tile, it loads the corresponding gate index, which directly maps to a unique column of the up projection and row of the down projection weight matrices. The kernel computes the up-projected feature from a dot product between the input dense feature row and the up projection weight column, multiplies it by the gate value, and finally uses it to scale the down projection weight row before accumulating the output. To ensure coalesced access, we note that the up projection weight matrix is stored in transposed format. This version of the kernel is specialized to handle the case where Tn=256T\_{n}=256 and the compression ratio is 8, leading to a total of 32 elements for each packed TwELL tile. In this specific instance, we load both the number of non-zeros and all the indices and values for the tile in a single fully coalesced access over the CTA’s warp, which later allows loading the full TwELL tile information via minimal warp register shuffle operations without incurring any shared memory overheads. In preliminary experiments, we also found that re-ordering the kernel calls in descending order of non-zeros can further accelerate performance with low batch sizes. However, we note that we did not find this optimization necessary with large batches and omitted it for simplicity.

[⬇](data:text/plain;base64,dGVtcGxhdGUgPAogICAgY29uc3QgaW50IFRfbiwKICAgIGNvbnN0IGludCBUX25fY29tcHJlc3NlZCwKICAgIGNvbnN0IGludCBOVU1fVF9uLAogICAgY29uc3QgaW50IE9VVF9ESU0sCiAgICBjb25zdCBpbnQgU1BMSVRfT1VUX0RJTQo+Cl9fZ2xvYmFsX18gX19sYXVuY2hfYm91bmRzX18oMzIpCnZvaWQgbW1fdDJkX2tlcm5lbCgKICAgIGNvbnN0IHVpbnQzMl90KiBJTl90d2VsbF9wYWNrZWRfZCwKICAgIGNvbnN0IF9fbnZfYmZsb2F0MTYqIERPV05fZCwKICAgIF9fbnZfYmZsb2F0MTYqIE9VVF9kCikKewogICAgc3RhdGljX2Fzc2VydCgKICAgICAgICAoU1BMSVRfT1VUX0RJTSAlCiAgICAgICAgIk9VVF9ESU0gbXVzdCBiZSBkaXZpc2libGUgYnkgV0FSUF9TSVpFLiIKICAgICk7CiAgICBzdGF0aWNfYXNzZXJ0KAogICAgICAgIChPVVRfRElNICUKICAgICAgICAiT1VUX0RJTSBtdXN0IGJlIGRpdmlzaWJsZSBieSBTUExJVF9PVVRfRElNLiIKICAgICk7CiAgICBzdGF0aWNfYXNzZXJ0KFRfbl9jb21wcmVzc2VkID09IFdBUlBfU0laRSwKICAgICAgICAiV2FycC1zeW5jIFR3RUxMLXRvLWRlbnNlIGFzc3VtZXMgYSAzMi13aWRlIGNvbXByZXNzZWQgdGlsZS4iKTsKCiAgICBmbG9hdCBPVVRfYWNjdW1bT1VUX0RJTSAvIFNUUklERV84eFdBUlBdWzhdID0gezAuMGZ9OwogICAgY29uc3RleHByIGludCBOVU1fTE9BRF9JVEVSUyA9IFNQTElUX09VVF9ESU0gLyBTVFJJREVfOHhXQVJQOwoKICAgIElOX3R3ZWxsX3BhY2tlZF9kICs9IGJsb2NrSWR4LnggKiBUX25fY29tcHJlc3NlZCAqIE5VTV9UX24gKyB0aHJlYWRJZHgueDsKICAgIERPV05fZCArPSB0aHJlYWRJZHgueCAqIDggKyBibG9ja0lkeC55ICogU1BMSVRfT1VUX0RJTTsKICAgIE9VVF9kICs9IGJsb2NrSWR4LnggKiBPVVRfRElNICsgdGhyZWFkSWR4LnggKiA4ICsgYmxvY2tJZHgueSAqIFNQTElUX09VVF9ESU07CgogICAgI3ByYWdtYSB1bnJvbGwgMQogICAgZm9yIChpbnQgdGlsZV9pZHggPSAwOyB0aWxlX2lkeCA8IE5VTV9UX247ICsrdGlsZV9pZHgpIHsKICAgICAgICBjb25zdCBpbnQgbGFuZV90aWxlX3JlZ2lzdGVyID0KICAgICAgICAgICAgSU5fdHdlbGxfcGFja2VkX2RbdGlsZV9pZHggKiBUX25fY29tcHJlc3NlZF07CiAgICAgICAgY29uc3QgaW50IG51bV9ub256ZXJvcyA9CiAgICAgICAgICAgIF9fc2hmbF9zeW5jKDB4RkZGRkZGRkZ1LCBsYW5lX3RpbGVfcmVnaXN0ZXIsIDApOwoKICAgICAgICAjcHJhZ21hIHVucm9sbCAxCiAgICAgICAgZm9yIChpbnQgaWR4ID0gMTsgaWR4IDwgbnVtX25vbnplcm9zICsgMTsgKytpZHgpIHsKICAgICAgICAgICAgY29uc3QgdWludDMyX3QgY29tcHJlc3NlZF9pZHhfYmYxNiA9CiAgICAgICAgICAgICAgICBfX3NoZmxfc3luYygweEZGRkZGRkZGdSwgbGFuZV90aWxlX3JlZ2lzdGVyLCBpZHgpOwogICAgICAgICAgICBjb25zdCB1aW50MzJfdCBub256ZXJvX2lkeCA9IGNvbXByZXNzZWRfaWR4X2JmMTYgJiAweEZGRkZ1OwogICAgICAgICAgICBjb25zdCBfX252X2JmbG9hdDE2MiBub256ZXJvX2ZlYXR1cmUgPQogICAgICAgICAgICAgICAgX19iZmxvYXQxNjJiZmxvYXQxNjIoCiAgICAgICAgICAgICAgICAgICAgcmVpbnRlcnByZXRfY2FzdDxjb25zdCBfX252X2JmbG9hdDE2Kj4oCiAgICAgICAgICAgICAgICAgICAgICAgICZjb21wcmVzc2VkX2lkeF9iZjE2CiAgICAgICAgICAgICAgICAgICAgKVsxXQogICAgICAgICAgICAgICAgKTsKCiAgICAgICAgICAgICNwcmFnbWEgdW5yb2xsCiAgICAgICAgICAgIGZvciAoaW50IGl0ZXJfaWR4ID0gMDsgaXRlcl9pZHggPCBOVU1fTE9BRF9JVEVSUzsgKytpdGVyX2lkeCkgewogICAgICAgICAgICAgICAgY29uc3QgdWludDQgcGFja2VkX2JmbG9hdHNfeDggPQogICAgICAgICAgICAgICAgICAgICpyZWludGVycHJldF9jYXN0PGNvbnN0IHVpbnQ0Kj4oCiAgICAgICAgICAgICAgICAgICAgICAgIERPV05fZAogICAgICAgICAgICAgICAgICAgICAgICArIG5vbnplcm9faWR4ICogT1VUX0RJTQogICAgICAgICAgICAgICAgICAgICAgICArIGl0ZXJfaWR4ICogU1RSSURFXzh4V0FSUAogICAgICAgICAgICAgICAgICAgICk7CiAgICAgICAgICAgICAgICBjb25zdCBfX252X2JmbG9hdDE2MiBwYWNrZWRfYmZsb2F0c18xID0KICAgICAgICAgICAgICAgICAgICAqcmVpbnRlcnByZXRfY2FzdDxjb25zdCBfX252X2JmbG9hdDE2Mio+KAogICAgICAgICAgICAgICAgICAgICAgICAmcGFja2VkX2JmbG9hdHNfeDgueAogICAgICAgICAgICAgICAgICAgICk7CiAgICAgICAgICAgICAgICBfX252X2JmbG9hdDE2MiBzY2FsZWRfYmZsb2F0c18xID0KICAgICAgICAgICAgICAgICAgICBfX2htdWwyKG5vbnplcm9fZmVhdHVyZSwgcGFja2VkX2JmbG9hdHNfMSk7CiAgICAgICAgICAgICAgICBmbG9hdDIgc2NhbGVkX2Zsb2F0c18xID0gX19iZmxvYXQxNjIyZmxvYXQyKHNjYWxlZF9iZmxvYXRzXzEpOwogICAgICAgICAgICAgICAgT1VUX2FjY3VtW2l0ZXJfaWR4XVswXSArPSBzY2FsZWRfZmxvYXRzXzEueDsKICAgICAgICAgICAgICAgIE9VVF9hY2N1bVtpdGVyX2lkeF1bMV0gKz0gc2NhbGVkX2Zsb2F0c18xLnk7CgogICAgICAgICAgICAgICAgY29uc3QgX19udl9iZmxvYXQxNjIgcGFja2VkX2JmbG9hdHNfMiA9CiAgICAgICAgICAgICAgICAgICAgKnJlaW50ZXJwcmV0X2Nhc3Q8Y29uc3QgX19udl9iZmxvYXQxNjIqPigKICAgICAgICAgICAgICAgICAgICAgICAgJnBhY2tlZF9iZmxvYXRzX3g4LnkKICAgICAgICAgICAgICAgICAgICApOwogICAgICAgICAgICAgICAgX19udl9iZmxvYXQxNjIgc2NhbGVkX2JmbG9hdHNfMiA9CiAgICAgICAgICAgICAgICAgICAgX19obXVsMihub256ZXJvX2ZlYXR1cmUsIHBhY2tlZF9iZmxvYXRzXzIpOwogICAgICAgICAgICAgICAgZmxvYXQyIHNjYWxlZF9mbG9hdHNfMiA9IF9fYmZsb2F0MTYyMmZsb2F0MihzY2FsZWRfYmZsb2F0c18yKTsKICAgICAgICAgICAgICAgIE9VVF9hY2N1bVtpdGVyX2lkeF1bMl0gKz0gc2NhbGVkX2Zsb2F0c18yLng7CiAgICAgICAgICAgICAgICBPVVRfYWNjdW1baXRlcl9pZHhdWzNdICs9IHNjYWxlZF9mbG9hdHNfMi55OwoKICAgICAgICAgICAgICAgIGNvbnN0IF9fbnZfYmZsb2F0MTYyIHBhY2tlZF9iZmxvYXRzXzMgPQogICAgICAgICAgICAgICAgICAgICpyZWludGVycHJldF9jYXN0PGNvbnN0IF9fbnZfYmZsb2F0MTYyKj4oCiAgICAgICAgICAgICAgICAgICAgICAgICZwYWNrZWRfYmZsb2F0c194OC56CiAgICAgICAgICAgICAgICAgICAgKTsKICAgICAgICAgICAgICAgIF9fbnZfYmZsb2F0MTYyIHNjYWxlZF9iZmxvYXRzXzMgPQogICAgICAgICAgICAgICAgICAgIF9faG11bDIobm9uemVyb19mZWF0dXJlLCBwYWNrZWRfYmZsb2F0c18zKTsKICAgICAgICAgICAgICAgIGZsb2F0MiBzY2FsZWRfZmxvYXRzXzMgPSBfX2JmbG9hdDE2MjJmbG9hdDIoc2NhbGVkX2JmbG9hdHNfMyk7CiAgICAgICAgICAgICAgICBPVVRfYWNjdW1baXRlcl9pZHhdWzRdICs9IHNjYWxlZF9mbG9hdHNfMy54OwogICAgICAgICAgICAgICAgT1VUX2FjY3VtW2l0ZXJfaWR4XVs1XSArPSBzY2FsZWRfZmxvYXRzXzMueTsKCiAgICAgICAgICAgICAgICBjb25zdCBfX252X2JmbG9hdDE2MiBwYWNrZWRfYmZsb2F0c180ID0KICAgICAgICAgICAgICAgICAgICAqcmVpbnRlcnByZXRfY2FzdDxjb25zdCBfX252X2JmbG9hdDE2Mio+KAogICAgICAgICAgICAgICAgICAgICAgICAmcGFja2VkX2JmbG9hdHNfeDgudwogICAgICAgICAgICAgICAgICAgICk7CiAgICAgICAgICAgICAgICBfX252X2JmbG9hdDE2MiBzY2FsZWRfYmZsb2F0c180ID0KICAgICAgICAgICAgICAgICAgICBfX2htdWwyKG5vbnplcm9fZmVhdHVyZSwgcGFja2VkX2JmbG9hdHNfNCk7CiAgICAgICAgICAgICAgICBmbG9hdDIgc2NhbGVkX2Zsb2F0c180ID0gX19iZmxvYXQxNjIyZmxvYXQyKHNjYWxlZF9iZmxvYXRzXzQpOwogICAgICAgICAgICAgICAgT1VUX2FjY3VtW2l0ZXJfaWR4XVs2XSArPSBzY2FsZWRfZmxvYXRzXzQueDsKICAgICAgICAgICAgICAgIE9VVF9hY2N1bVtpdGVyX2lkeF1bN10gKz0gc2NhbGVkX2Zsb2F0c180Lnk7CiAgICAgICAgICAgIH0KICAgICAgICB9CiAgICB9CgogICAgI3ByYWdtYSB1bnJvbGwKICAgIGZvciAoaW50IGl0ZXJfaWR4ID0gMDsgaXRlcl9pZHggPCBOVU1fTE9BRF9JVEVSUzsgKytpdGVyX2lkeCkgewogICAgICAgIF9fbnZfYmZsb2F0MTYyICBfX2FsaWduX18oOCkgcGFja2VkX2JmbG9hdHNfeDhbNF07CiAgICAgICAgcGFja2VkX2JmbG9hdHNfeDhbMF0gPSBfX2Zsb2F0czJiZmxvYXQxNjJfcm4oCiAgICAgICAgICAgIE9VVF9hY2N1bVtpdGVyX2lkeF1bMF0sIE9VVF9hY2N1bVtpdGVyX2lkeF1bMV0KICAgICAgICApOwogICAgICAgIHBhY2tlZF9iZmxvYXRzX3g4WzFdID0gX19mbG9hdHMyYmZsb2F0MTYyX3JuKAogICAgICAgICAgICBPVVRfYWNjdW1baXRlcl9pZHhdWzJdLCBPVVRfYWNjdW1baXRlcl9pZHhdWzNdCiAgICAgICAgKTsKICAgICAgICBwYWNrZWRfYmZsb2F0c194OFsyXSA9IF9fZmxvYXRzMmJmbG9hdDE2Ml9ybigKICAgICAgICAgICAgT1VUX2FjY3VtW2l0ZXJfaWR4XVs0XSwgT1VUX2FjY3VtW2l0ZXJfaWR4XVs1XQogICAgICAgICk7CiAgICAgICAgcGFja2VkX2JmbG9hdHNfeDhbM10gPSBfX2Zsb2F0czJiZmxvYXQxNjJfcm4oCiAgICAgICAgICAgIE9VVF9hY2N1bVtpdGVyX2lkeF1bNl0sIE9VVF9hY2N1bVtpdGVyX2lkeF1bN10KICAgICAgICApOwoKICAgICAgICAqcmVpbnRlcnByZXRfY2FzdDx1aW50NCo+KE9VVF9kICsgaXRlcl9pZHggKiBTVFJJREVfOHhXQVJQKSA9CiAgICAgICAgICAgICpyZWludGVycHJldF9jYXN0PHVpbnQ0Kj4ocGFja2VkX2JmbG9hdHNfeDgpOwogICAgfQp9)

1template <

2 const int T\_n,

3 const int T\_n\_compressed,

4 const int NUM\_T\_n,

5 const int OUT\_DIM,

6 const int SPLIT\_OUT\_DIM

7>

8\_\_global\_\_ \_\_launch\_bounds\_\_(32)

9void mm\_t2d\_kernel(

10 const uint32\_t\* IN\_twell\_packed\_d,

11 const \_\_nv\_bfloat16\* DOWN\_d,

12 \_\_nv\_bfloat16\* OUT\_d

13)

14{

15 static\_assert(

16 (SPLIT\_OUT\_DIM %

17 "OUT\_DIM must be divisible by WARP\_SIZE."

18 );

19 static\_assert(

20 (OUT\_DIM %

21 "OUT\_DIM must be divisible by SPLIT\_OUT\_DIM."

22 );

23 static\_assert(T\_n\_compressed == WARP\_SIZE,

24 "Warp-sync TwELL-to-dense assumes a 32-wide compressed tile.");

25

26 float OUT\_accum[OUT\_DIM / STRIDE\_8xWARP][8] = {0.0f};

27 constexpr int NUM\_LOAD\_ITERS = SPLIT\_OUT\_DIM / STRIDE\_8xWARP;

28

29 IN\_twell\_packed\_d += blockIdx.x \* T\_n\_compressed \* NUM\_T\_n + threadIdx.x;

30 DOWN\_d += threadIdx.x \* 8 + blockIdx.y \* SPLIT\_OUT\_DIM;

31 OUT\_d += blockIdx.x \* OUT\_DIM + threadIdx.x \* 8 + blockIdx.y \* SPLIT\_OUT\_DIM;

32

33 #pragma unroll 1

34 for (int tile\_idx = 0; tile\_idx < NUM\_T\_n; ++tile\_idx) {

35 const int lane\_tile\_register =

36 IN\_twell\_packed\_d[tile\_idx \* T\_n\_compressed];

37 const int num\_nonzeros =

38 \_\_shfl\_sync(0xFFFFFFFFu, lane\_tile\_register, 0);

39

40 #pragma unroll 1

41 for (int idx = 1; idx < num\_nonzeros + 1; ++idx) {

42 const uint32\_t compressed\_idx\_bf16 =

43 \_\_shfl\_sync(0xFFFFFFFFu, lane\_tile\_register, idx);

44 const uint32\_t nonzero\_idx = compressed\_idx\_bf16 & 0xFFFFu;

45 const \_\_nv\_bfloat162 nonzero\_feature =

46 \_\_bfloat162bfloat162(

47 reinterpret\_cast<const \_\_nv\_bfloat16\*>(

48 &compressed\_idx\_bf16

49 )[1]

50 );

51

52 #pragma unroll

53 for (int iter\_idx = 0; iter\_idx < NUM\_LOAD\_ITERS; ++iter\_idx) {

54 const uint4 packed\_bfloats\_x8 =

55 \*reinterpret\_cast<const uint4\*>(

56 DOWN\_d

57 + nonzero\_idx \* OUT\_DIM

58 + iter\_idx \* STRIDE\_8xWARP

59 );

60 const \_\_nv\_bfloat162 packed\_bfloats\_1 =

61 \*reinterpret\_cast<const \_\_nv\_bfloat162\*>(

62 &packed\_bfloats\_x8.x

63 );

64 \_\_nv\_bfloat162 scaled\_bfloats\_1 =

65 \_\_hmul2(nonzero\_feature, packed\_bfloats\_1);

66 float2 scaled\_floats\_1 = \_\_bfloat1622float2(scaled\_bfloats\_1);

67 OUT\_accum[iter\_idx][0] += scaled\_floats\_1.x;

68 OUT\_accum[iter\_idx][1] += scaled\_floats\_1.y;

69

70 const \_\_nv\_bfloat162 packed\_bfloats\_2 =

71 \*reinterpret\_cast<const \_\_nv\_bfloat162\*>(

72 &packed\_bfloats\_x8.y

73 );

74 \_\_nv\_bfloat162 scaled\_bfloats\_2 =

75 \_\_hmul2(nonzero\_feature, packed\_bfloats\_2);

76 float2 scaled\_floats\_2 = \_\_bfloat1622float2(scaled\_bfloats\_2);

77 OUT\_accum[iter\_idx][2] += scaled\_floats\_2.x;

78 OUT\_accum[iter\_idx][3] += scaled\_floats\_2.y;

79

80 const \_\_nv\_bfloat162 packed\_bfloats\_3 =

81 \*reinterpret\_cast<const \_\_nv\_bfloat162\*>(

82 &packed\_bfloats\_x8.z

83 );

84 \_\_nv\_bfloat162 scaled\_bfloats\_3 =

85 \_\_hmul2(nonzero\_feature, packed\_bfloats\_3);

86 float2 scaled\_floats\_3 = \_\_bfloat1622float2(scaled\_bfloats\_3);

87 OUT\_accum[iter\_idx][4] += scaled\_floats\_3.x;

88 OUT\_accum[iter\_idx][5] += scaled\_floats\_3.y;

89

90 const \_\_nv\_bfloat162 packed\_bfloats\_4 =

91 \*reinterpret\_cast<const \_\_nv\_bfloat162\*>(

92 &packed\_bfloats\_x8.w

93 );

94 \_\_nv\_bfloat162 scaled\_bfloats\_4 =

95 \_\_hmul2(nonzero\_feature, packed\_bfloats\_4);

96 float2 scaled\_floats\_4 = \_\_bfloat1622float2(scaled\_bfloats\_4);

97 OUT\_accum[iter\_idx][6] += scaled\_floats\_4.x;

98 OUT\_accum[iter\_idx][7] += scaled\_floats\_4.y;

99 }

100 }

101 }

102

103 #pragma unroll

104 for (int iter\_idx = 0; iter\_idx < NUM\_LOAD\_ITERS; ++iter\_idx) {

105 \_\_nv\_bfloat162 \_\_align\_\_(8) packed\_bfloats\_x8[4];

106 packed\_bfloats\_x8[0] = \_\_floats2bfloat162\_rn(

107 OUT\_accum[iter\_idx][0], OUT\_accum[iter\_idx][1]

108 );

109 packed\_bfloats\_x8[1] = \_\_floats2bfloat162\_rn(

110 OUT\_accum[iter\_idx][2], OUT\_accum[iter\_idx][3]

111 );

112 packed\_bfloats\_x8[2] = \_\_floats2bfloat162\_rn(

113 OUT\_accum[iter\_idx][4], OUT\_accum[iter\_idx][5]

114 );

115 packed\_bfloats\_x8[3] = \_\_floats2bfloat162\_rn(

116 OUT\_accum[iter\_idx][6], OUT\_accum[iter\_idx][7]

117 );

118

119 \*reinterpret\_cast<uint4\*>(OUT\_d + iter\_idx \* STRIDE\_8xWARP) =

120 \*reinterpret\_cast<uint4\*>(packed\_bfloats\_x8);

121 }

122}

Listing 3: Down projection leveraging up projections from the TwELL format for the non-gated model variants.

As mentioned in Section [2](#S2 "2 Large Language Models, Feedforward Blocks, and Sparsity ‣ Sparser, Faster, Lighter Transformer Language Models") of the main text, together with modern gated LLMs, we also provide specific kernels that support older non-gated variants, which we empirically evaluate in Appendix [C](#A3 "Appendix C Parameter Studies and Ablations ‣ Sparser, Faster, Lighter Transformer Language Models"). In Figure LABEL:lst:appA:twell\_down\_nongated, we provide code listings with the device code for our kernel implementing the custom down projection kernel that leverages up projection activations stored in the TwELL format for these experiments. Similarly to the fused kernel explained in Section [3](#S3.2 "3 Making Sparse LLMs Fast ‣ Sparser, Faster, Lighter Transformer Language Models") and examined above, this kernel is launched on a grid of warp-sized CTAs and reads the sparsity pattern, this time from the up projection activations stored in the TwELL format. This time, the kernel maintains in memory the out projection and a float32 accumulator for a small output segment. Then, it first statically iterates through the TwELL tile and then dynamically iterates over the number of non-zeros in each tile. At each iteration, it loads the non-zero index and the corresponding activation down projection column segment, before multiplying the two and accumulating the result. In contrast to our fused kernel, where we have to consider full rows on the input and output to perform dot products between the input features and the up projection weights, introducing the split formulation in this kernel is a deliberate and purposeful choice: by introducing trivial duplication of the non-zero reads we can further increase parallelism, reduce register pressure, increase occupancy, and hide longer latencies from uneven sparsity. In practice, we note that using a split dimension of half the base output dimension, leading to two CTAs per output row, appears optimal on our Hopper GPUs.

### A.2 Training Kernels Selection

[⬇](data:text/plain;base64,X19nbG9iYWxfXyB2b2lkIHR3ZWxsX3RvX2VsbF9rZXJuZWwoCiAgICBjb25zdCBfX252X2JmbG9hdDE2KiBfX3Jlc3RyaWN0X18gQ192YWxzLAogICAgY29uc3QgdWludDhfdCogX19yZXN0cmljdF9fIENfaWR4LAogICAgY29uc3QgdWludDMyX3QqIF9fcmVzdHJpY3RfXyBDX25ueiwKICAgIF9fbnZfYmZsb2F0MTYqIF9fcmVzdHJpY3RfXyBlbGxfdmFsLAogICAgaW50MTZfdCogX19yZXN0cmljdF9fIGVsbF9jb2wsCiAgICBpbnQzMl90KiBfX3Jlc3RyaWN0X18gcm93X25ueiwKICAgIGZsb2F0KiBfX3Jlc3RyaWN0X18gbDBfb3V0LAogICAgZmxvYXQqIF9fcmVzdHJpY3RfXyBsMV9vdXQsCiAgICBpbnQgTSwKICAgIGludCBOX1RJTEVTLAogICAgaW50IEJXLAogICAgaW50IEVMTF9XLAogICAgaW50IFRfbgopCnsKICAgIGNvbnN0IGludCByb3cgPSBibG9ja0lkeC54ICogYmxvY2tEaW0ueSArIHRocmVhZElkeC55OwogICAgaWYgKHJvdyA+PSBNKSB7CiAgICAgICAgcmV0dXJuOwogICAgfQoKICAgIGNvbnN0IGludCB0aWQgPSB0aHJlYWRJZHgueDsKICAgIGludCBjbnQgPSAodGlkIDwgTl9USUxFUykgPyBDX25uelsoc2l6ZV90KXRpZCAqIE0gKyByb3ddIDogMDsKICAgIGNudCA9IG1pbihjbnQsIEJXKTsKCiAgICBpbnQgb2Zmc2V0ID0gY250OwogICAgZm9yIChpbnQgZGVsdGEgPSAxOyBkZWx0YSA8IFdBUlBfU0laRTsgZGVsdGEgPDw9IDEpIHsKICAgICAgICBjb25zdCBpbnQgcmVjdiA9IF9fc2hmbF91cF9zeW5jKDB4RkZGRkZGRkZ1LCBvZmZzZXQsIGRlbHRhKTsKICAgICAgICBpZiAodGlkID49IGRlbHRhKSB7CiAgICAgICAgICAgIG9mZnNldCArPSByZWN2OwogICAgICAgIH0KICAgIH0KCiAgICBjb25zdCBpbnQgc3RhcnQgPSBvZmZzZXQgLSBjbnQ7CiAgICBjb25zdCBpbnQgdG90YWwgPQogICAgICAgIF9fc2hmbF9zeW5jKDB4RkZGRkZGRkZ1LCBvZmZzZXQsIG1pbihOX1RJTEVTIC0gMSwgV0FSUF9TSVpFIC0gMSkpOwoKICAgIGNvbnN0IF9fbnZfYmZsb2F0MTYqIHN2ID0KICAgICAgICBDX3ZhbHMgKyAoc2l6ZV90KXJvdyAqIE5fVElMRVMgKiBCVyArIChzaXplX3QpdGlkICogQlc7CiAgICBjb25zdCB1aW50OF90KiBzaSA9CiAgICAgICAgQ19pZHggKyAoc2l6ZV90KXJvdyAqIE5fVElMRVMgKiBCVyArIChzaXplX3QpdGlkICogQlc7CgogICAgZmxvYXQgbDBfYWNjID0gMC4wZjsKICAgIGZsb2F0IGwxX2FjYyA9IDAuMGY7CiAgICBpZiAobDBfb3V0KSB7CiAgICAgICAgY29uc3QgZmxvYXQgaW52X00gPSAxLjBmIC8gKGZsb2F0KU07CiAgICAgICAgbDBfYWNjID0gKGZsb2F0KWNudCAqIGludl9NOwogICAgICAgIGZvciAoaW50IGkgPSAwOyBpIDwgY250OyArK2kpIHsKICAgICAgICAgICAgbDFfYWNjICs9IF9fYmZsb2F0MTYyZmxvYXQoc3ZbaV0pICogaW52X007CiAgICAgICAgfQogICAgfQoKICAgIGlmIChjbnQgPiAwICYmIHN0YXJ0IDwgRUxMX1cpIHsKICAgICAgICBjb25zdCBpbnQgY29weV9uID0gbWluKGNudCwgRUxMX1cgLSBzdGFydCk7CiAgICAgICAgX19udl9iZmxvYXQxNiogZHYgPSBlbGxfdmFsICsgKHNpemVfdClyb3cgKiBFTExfVyArIHN0YXJ0OwogICAgICAgIGludDE2X3QqIGRjID0gZWxsX2NvbCArIChzaXplX3Qpcm93ICogRUxMX1cgKyBzdGFydDsKICAgICAgICBmb3IgKGludCBpID0gMDsgaSA8IGNvcHlfbjsgKytpKSB7CiAgICAgICAgICAgIGR2W2ldID0gc3ZbaV07CiAgICAgICAgICAgIGRjW2ldID0gKGludDE2X3QpKHNpW2ldKSArIChpbnQxNl90KSh0aWQgKiBUX24pOwogICAgICAgIH0KICAgIH0KCiAgICBpZiAodGlkID09IDApIHsKICAgICAgICByb3dfbm56W3Jvd10gPSB0b3RhbDsKICAgIH0KCiAgICBpZiAobDBfb3V0KSB7CiAgICAgICAgZm9yIChpbnQgcyA9IDE2OyBzID4gMDsgcyA+Pj0gMSkgewogICAgICAgICAgICBsMF9hY2MgKz0gX19zaGZsX2Rvd25fc3luYygweEZGRkZGRkZGdSwgbDBfYWNjLCBzKTsKICAgICAgICAgICAgbDFfYWNjICs9IF9fc2hmbF9kb3duX3N5bmMoMHhGRkZGRkZGRnUsIGwxX2FjYywgcyk7CiAgICAgICAgfQogICAgICAgIGlmICh0aWQgPT0gMCkgewogICAgICAgICAgICBhdG9taWNBZGQobDBfb3V0LCBsMF9hY2MpOwogICAgICAgICAgICBpZiAobDFfb3V0KSB7CiAgICAgICAgICAgICAgICBhdG9taWNBZGQobDFfb3V0LCBsMV9hY2MpOwogICAgICAgICAgICB9CiAgICAgICAgfQogICAgfQp9)

1\_\_global\_\_ void twell\_to\_ell\_kernel(

2 const \_\_nv\_bfloat16\* \_\_restrict\_\_ C\_vals,

3 const uint8\_t\* \_\_restrict\_\_ C\_idx,

4 const uint32\_t\* \_\_restrict\_\_ C\_nnz,

5 \_\_nv\_bfloat16\* \_\_restrict\_\_ ell\_val,

6 int16\_t\* \_\_restrict\_\_ ell\_col,

7 int32\_t\* \_\_restrict\_\_ row\_nnz,

8 float\* \_\_restrict\_\_ l0\_out,

9 float\* \_\_restrict\_\_ l1\_out,

10 int M,

11 int N\_TILES,

12 int BW,

13 int ELL\_W,

14 int T\_n

15)

16{

17 const int row = blockIdx.x \* blockDim.y + threadIdx.y;

18 if (row >= M) {

19 return;

20 }

21

22 const int tid = threadIdx.x;

23 int cnt = (tid < N\_TILES) ? C\_nnz[(size\_t)tid \* M + row] : 0;

24 cnt = min(cnt, BW);

25

26 int offset = cnt;

27 for (int delta = 1; delta < WARP\_SIZE; delta <<= 1) {

28 const int recv = \_\_shfl\_up\_sync(0xFFFFFFFFu, offset, delta);

29 if (tid >= delta) {

30 offset += recv;

31 }

32 }

33

34 const int start = offset - cnt;

35 const int total =

36 \_\_shfl\_sync(0xFFFFFFFFu, offset, min(N\_TILES - 1, WARP\_SIZE - 1));

37

38 const \_\_nv\_bfloat16\* sv =

39 C\_vals + (size\_t)row \* N\_TILES \* BW + (size\_t)tid \* BW;

40 const uint8\_t\* si =

41 C\_idx + (size\_t)row \* N\_TILES \* BW + (size\_t)tid \* BW;

42

43 float l0\_acc = 0.0f;

44 float l1\_acc = 0.0f;

45 if (l0\_out) {

46 const float inv\_M = 1.0f / (float)M;

47 l0\_acc = (float)cnt \* inv\_M;

48 for (int i = 0; i < cnt; ++i) {

49 l1\_acc += \_\_bfloat162float(sv[i]) \* inv\_M;

50 }

51 }

52

53 if (cnt > 0 && start < ELL\_W) {

54 const int copy\_n = min(cnt, ELL\_W - start);

55 \_\_nv\_bfloat16\* dv = ell\_val + (size\_t)row \* ELL\_W + start;

56 int16\_t\* dc = ell\_col + (size\_t)row \* ELL\_W + start;

57 for (int i = 0; i < copy\_n; ++i) {

58 dv[i] = sv[i];

59 dc[i] = (int16\_t)(si[i]) + (int16\_t)(tid \* T\_n);

60 }

61 }

62

63 if (tid == 0) {

64 row\_nnz[row] = total;

65 }

66

67 if (l0\_out) {

68 for (int s = 16; s > 0; s >>= 1) {

69 l0\_acc += \_\_shfl\_down\_sync(0xFFFFFFFFu, l0\_acc, s);

70 l1\_acc += \_\_shfl\_down\_sync(0xFFFFFFFFu, l1\_acc, s);

71 }

72 if (tid == 0) {

73 atomicAdd(l0\_out, l0\_acc);

74 if (l1\_out) {

75 atomicAdd(l1\_out, l1\_acc);

76 }

77 }

78 }

79}

Listing 4: Conversion from TwELL to the hybrid format logic.

In Figure LABEL:lst:appA:blocked\_ell\_to\_ell, we provide code listings with the device code for our training kernel used to convert gate activations stored in the TwELL format into the compact ELL component of our hybrid training representation while accumulating L0L\_{0} and L1L\_{1} statistics. As discussed in Section [3](#S3.2 "3 Making Sparse LLMs Fast ‣ Sparser, Faster, Lighter Transformer Language Models"), the conversion dynamically partitions the rows based on the non-zero counts. We allocate a warp to each row, and let each thread read the number of active entries in a single tile. We then use warp register shuffles to obtain an inclusive prefix scan and determine the starting offset of that tile within the destination ELL row. This design allows for directly compacting the tiled representation into contiguous row-wise ELL storage without requiring any synchronization beyond warp-level or shared memory accesses. The kernel writes the true row occupancy to r​o​w​\_​n​n​zrow\\_nnz even when the row exceeds the configured ELL width E​L​L​\_​WELL\\_W, allowing overflow rows to be detected and promoted to the dense tail of the hybrid format. During training, each warp also reduces simple L0L\_{0} and L1L\_{1} sparsity statistics to compute the sparsity levels and L1 loss before issuing a single atomic update.

[⬇](data:text/plain;base64,X19nbG9iYWxfXyB2b2lkIG1hdG11bF9zYXZlX3NwYXJzZV9saWtlX2VsbCgKICAgIGJmbG9hdDE2KiBBLAogICAgYmZsb2F0MTYqIEJfVCwKICAgIEVMTCogb3V0LAogICAgaW50IE0sCiAgICBpbnQgSywKICAgIGludCBOCikKewogICAgY29uc3QgaW50IHJvdyA9IGJsb2NrSWR4Lng7CiAgICBjb25zdCBpbnQgZWxsX24gPSBvdXQtPnJvd19jb3VudHNbcm93XTsKICAgIGlmIChlbGxfbiA9PSAwIHx8IGVsbF9uID4gRUxMX1dJRFRIKSB7CiAgICAgICAgcmV0dXJuOwogICAgfQoKICAgIGJmbG9hdDE2KiBBX3Jvd19wdHIgPSBBICsgcm93ICogSzsKICAgIGNvbnN0IGludCBsYW5lX2lkID0gdGhyZWFkSWR4LnggJiAzMTsKICAgIGNvbnN0IGludCB3YXJwX2lkID0gdGhyZWFkSWR4LnggPj4gNTsKICAgIGNvbnN0IGludCBudW1fd2FycHMgPSBibG9ja0RpbS54ID4+IDU7CiAgICBjb25zdCBpbnQgbnVtX2NodW5rcyA9IEsgLyA4OwoKICAgIGZvciAoaW50IG91dF9pZHggPSB3YXJwX2lkOyBvdXRfaWR4IDwgZWxsX247IG91dF9pZHggKz0gbnVtX3dhcnBzKSB7CiAgICAgICAgY29uc3QgaW50IGNvbCA9IG91dC0+Y29sc1tyb3cgKiBFTExfV0lEVEggKyBvdXRfaWR4XTsKICAgICAgICBiZmxvYXQxNiogQl9yb3dfcHRyID0gQl9UICsgY29sICogSzsKICAgICAgICBmbG9hdCBhY2MgPSAwLjBmOwoKICAgICAgICBmb3IgKGludCBjaHVua19iYXNlID0gMDsgY2h1bmtfYmFzZSA8IG51bV9jaHVua3M7IGNodW5rX2Jhc2UgKz0gMzIpIHsKICAgICAgICAgICAgY29uc3QgaW50IGNodW5rID0gY2h1bmtfYmFzZSArIGxhbmVfaWQ7CiAgICAgICAgICAgIGlmIChjaHVuayA+PSBudW1fY2h1bmtzKSB7CiAgICAgICAgICAgICAgICBicmVhazsKICAgICAgICAgICAgfQoKICAgICAgICAgICAgaW50NCBhX3JhdyA9ICooaW50NCopKEFfcm93X3B0ciArIGNodW5rICogOCk7CiAgICAgICAgICAgIGludDQgYl9yYXcgPSAqKGludDQqKShCX3Jvd19wdHIgKyBjaHVuayAqIDgpOwogICAgICAgICAgICBiZmxvYXQxNl8yKiBhX3ZlYyA9IChiZmxvYXQxNl8yKikmYV9yYXc7CiAgICAgICAgICAgIGJmbG9hdDE2XzIqIGJfdmVjID0gKGJmbG9hdDE2XzIqKSZiX3JhdzsKCiAgICAgICAgICAgIGZvciAoaW50IHQgPSAwOyB0IDwgNDsgKyt0KSB7CiAgICAgICAgICAgICAgICBmbG9hdDIgYWYgPSBiZmxvYXQxNjIyZmxvYXQyKGFfdmVjW3RdKTsKICAgICAgICAgICAgICAgIGZsb2F0MiBiZiA9IGJmbG9hdDE2MjJmbG9hdDIoYl92ZWNbdF0pOwogICAgICAgICAgICAgICAgYWNjID0gZm1hZihhZi54LCBiZi54LCBhY2MpOwogICAgICAgICAgICAgICAgYWNjID0gZm1hZihhZi55LCBiZi55LCBhY2MpOwogICAgICAgICAgICB9CiAgICAgICAgfQoKICAgICAgICBmb3IgKGludCBvZmZzZXQgPSAxNjsgb2Zmc2V0ID4gMDsgb2Zmc2V0ID4+PSAxKSB7CiAgICAgICAgICAgIGFjYyArPSBfX3NoZmxfeG9yX3N5bmMoMHhGRkZGRkZGRnUsIGFjYywgb2Zmc2V0KTsKICAgICAgICB9CiAgICAgICAgaWYgKGxhbmVfaWQgPT0gMCkgewogICAgICAgICAgICBvdXQtPnZhbHNbcm93ICogRUxMX1dJRFRIICsgb3V0X2lkeF0gPSBmbG9hdDJiZmxvYXQxNihhY2MpOwogICAgICAgIH0KICAgIH0KfQo=)

1\_\_global\_\_ void matmul\_save\_sparse\_like\_ell(

2 bfloat16\* A,

3 bfloat16\* B\_T,

4 ELL\* out,

5 int M,

6 int K,

7 int N

8)

9{

10 const int row = blockIdx.x;

11 const int ell\_n = out->row\_counts[row];

12 if (ell\_n == 0 || ell\_n > ELL\_WIDTH) {

13 return;

14 }

15

16 bfloat16\* A\_row\_ptr = A + row \* K;

17 const int lane\_id = threadIdx.x & 31;

18 const int warp\_id = threadIdx.x >> 5;

19 const int num\_warps = blockDim.x >> 5;

20 const int num\_chunks = K / 8;

21

22 for (int out\_idx = warp\_id; out\_idx < ell\_n; out\_idx += num\_warps) {

23 const int col = out->cols[row \* ELL\_WIDTH + out\_idx];

24 bfloat16\* B\_row\_ptr = B\_T + col \* K;

25 float acc = 0.0f;

26

27 for (int chunk\_base = 0; chunk\_base < num\_chunks; chunk\_base += 32) {

28 const int chunk = chunk\_base + lane\_id;

29 if (chunk >= num\_chunks) {

30 break;

31 }

32

33 int4 a\_raw = \*(int4\*)(A\_row\_ptr + chunk \* 8);

34 int4 b\_raw = \*(int4\*)(B\_row\_ptr + chunk \* 8);

35 bfloat16\_2\* a\_vec = (bfloat16\_2\*)&a\_raw;

36 bfloat16\_2\* b\_vec = (bfloat16\_2\*)&b\_raw;

37

38 for (int t = 0; t < 4; ++t) {

39 float2 af = bfloat1622float2(a\_vec[t]);

40 float2 bf = bfloat1622float2(b\_vec[t]);

41 acc = fmaf(af.x, bf.x, acc);

42 acc = fmaf(af.y, bf.y, acc);

43 }

44 }

45

46 for (int offset = 16; offset > 0; offset >>= 1) {

47 acc += \_\_shfl\_xor\_sync(0xFFFFFFFFu, acc, offset);

48 }

49 if (lane\_id == 0) {

50 out->vals[row \* ELL\_WIDTH + out\_idx] = float2bfloat16(acc);

51 }

52 }

53}

Listing 5: Dense-to-hybrid matmul for populating the sparse ELL component during training using CUDA cores.

In Figure LABEL:lst:appA:save\_sparse, we provide code listings of our custom kernel used to perform the efficient dense-to-hybrid matmuls used during training, focusing on the sparse component. This kernel shows the logic of the dynamic hybrid partitioning, skipping the sparse operation in the non-zeros is recognized to exceed the size of the aggressively compact ELL format. The kernel takes two dense matrices, AA and BB (provided as BTB\_{T}), and a pre-allocated ELL output “out” of shape M×NM\times N, whose column indices encode the sparsity pattern to be evaluated. Rather than computing all M​NMN outputs, each thread block processes a single output row and iterates only over the column indices stored for that row in out. For each selected column, the kernel computes the dot product between A​[r​o​w,:]A[row,:] and BT​[c​o​l,:]B\_{T}[col,:]. To maximize coalescing and enable vectorized memory accesses, BB is stored transposed so that rows of BTB\_{T} are contiguous in memory. To maximize throughput with bfloat16, threads load AA and BTB\_{T} in 128-bit transactions (8 bfloat16 values at a time) and accumulate in float32 using fused multiply-adds. Each warp reduces partial sums using shuffle-based reduction, and the final value is written to the corresponding slot in the ELL value array. This design aligns with ELL’s row-oriented storage: the sparsity pattern is known up front, so the kernel avoids both dense materialization and irregular gathers beyond the indexed rows of BTB\_{T}. In the forward pass, we use this kernel to compute only the entries of the up projection operation x​WuxW\_{u} that will survive the subsequent gating, by copying the sparsity pattern from hgh\_{g} into out and filling its values with the corresponding dot products. In the backward pass, the same kernel is reused for masked gradient matmuls that share a known sparsity pattern. For instance, we use it to compute ∇h=∇y,WdT\nabla h=\nabla y,W\_{d}^{T}. Rows that exceed the ELL capacity are handled by routing the overflow to the dense backup matrix and computing that portion with Tensor Core–optimized kernels, as described in Algorithm [3](#alg3 "Algorithm 3 ‣ 4 Experimental Results ‣ Sparser, Faster, Lighter Transformer Language Models"), and they are multiplied by a binary mask containing the sparsity pattern to be applied.

[⬇](data:text/plain;base64,X19nbG9iYWxfXyB2b2lkIGh5YnJpZF90b19kZW5zZV9tYW10dWwoCiAgICBFTEwqIEEsCiAgICBiZmxvYXQxNiogQiwKICAgIGJmbG9hdDE2KiBDLAogICAgaW50IE0sCiAgICBpbnQgTiwKICAgIGludCBLCikKewogICAgY29uc3QgaW50IHJvdyA9IGJsb2NrSWR4Lng7CiAgICBjb25zdCBpbnQgZWxsX24gPSBBLT5yb3dfY291bnRzW3Jvd107CiAgICBpZiAoZWxsX24gPT0gMCB8fCBlbGxfbiA+IEVMTF9XSURUSCkgewogICAgICAgIHJldHVybjsKICAgIH0KCiAgICBiZmxvYXQxNiogQV9yb3dfdmFscyA9IEEtPnZhbHMgKyByb3cgKiBFTExfV0lEVEg7CiAgICB1aW50MTZfdCogQV9yb3dfaWR4cyA9IEEtPmNvbHMgKyByb3cgKiBFTExfV0lEVEg7CgogICAgZm9yIChpbnQgbl9vdXQgPSB0aHJlYWRJZHgueCAqIDg7IG5fb3V0IDwgTjsgbl9vdXQgKz0gOCAqIGJsb2NrRGltLngpIHsKICAgICAgICBmbG9hdDIgYWNjWzRdOwogICAgICAgIGZvciAoaW50IGkgPSAwOyBpIDwgNDsgKytpKSB7CiAgICAgICAgICAgIGFjY1tpXSA9IG1ha2VfZmxvYXQyKDAuZiwgMC5mKTsKICAgICAgICB9CgogICAgICAgIGZvciAoaW50IGsgPSAwOyBrIDwgRUxMX1dJRFRIOyArK2spIHsKICAgICAgICAgICAgaWYgKGsgPj0gZWxsX24pIHsKICAgICAgICAgICAgICAgIGJyZWFrOwogICAgICAgICAgICB9CgogICAgICAgICAgICBjb25zdCBiZmxvYXQxNiBhX3ZhbCA9IEFfcm93X3ZhbHNba107CiAgICAgICAgICAgIGNvbnN0IHVpbnQxNl90IGNvbF9pZHggPSBBX3Jvd19pZHhzW2tdOwogICAgICAgICAgICBiZmxvYXQxNiogQl9yb3dfcHRyID0gQiArIGNvbF9pZHggKiBOICsgbl9vdXQ7CiAgICAgICAgICAgIGludDQgYl92ZWNfcmF3ID0gKihpbnQ0KikoQl9yb3dfcHRyKTsKICAgICAgICAgICAgYmZsb2F0MTYyKiBiX3ZlYyA9IChiZmxvYXQxNjIqKSgmYl92ZWNfcmF3KTsKICAgICAgICAgICAgY29uc3QgZmxvYXQgYSA9IGJmbG9hdDE2MmZsb2F0KGFfdmFsKTsKCiAgICAgICAgICAgIGZvciAoaW50IHQgPSAwOyB0IDwgNDsgKyt0KSB7CiAgICAgICAgICAgICAgICBmbG9hdDIgYl9mMzIgPSBiZmxvYXQxNjIyZmxvYXQyKGJfdmVjW3RdKTsKICAgICAgICAgICAgICAgIGFjY1t0XS54ID0gZm1hZihhLCBiX2YzMi54LCBhY2NbdF0ueCk7CiAgICAgICAgICAgICAgICBhY2NbdF0ueSA9IGZtYWYoYSwgYl9mMzIueSwgYWNjW3RdLnkpOwogICAgICAgICAgICB9CiAgICAgICAgfQoKICAgICAgICBiZmxvYXQxNjIqIENfcHRyID0gKGJmbG9hdDE2MiopKEMgKyByb3cgKiBOICsgbl9vdXQpOwogICAgICAgIGZvciAoaW50IGkgPSAwOyBpIDwgNDsgKytpKSB7CiAgICAgICAgICAgIENfcHRyW2ldID0gZmxvYXQyMmJmbG9hdDE2MihhY2NbaV0pOwogICAgICAgIH0KICAgIH0KfQ==)

1\_\_global\_\_ void hybrid\_to\_dense\_mamtul(

2 ELL\* A,

3 bfloat16\* B,

4 bfloat16\* C,

5 int M,

6 int N,

7 int K

8)

9{

10 const int row = blockIdx.x;

11 const int ell\_n = A->row\_counts[row];

12 if (ell\_n == 0 || ell\_n > ELL\_WIDTH) {

13 return;

14 }

15

16 bfloat16\* A\_row\_vals = A->vals + row \* ELL\_WIDTH;

17 uint16\_t\* A\_row\_idxs = A->cols + row \* ELL\_WIDTH;

18

19 for (int n\_out = threadIdx.x \* 8; n\_out < N; n\_out += 8 \* blockDim.x) {

20 float2 acc[4];

21 for (int i = 0; i < 4; ++i) {

22 acc[i] = make\_float2(0.f, 0.f);

23 }

24

25 for (int k = 0; k < ELL\_WIDTH; ++k) {

26 if (k >= ell\_n) {

27 break;

28 }

29

30 const bfloat16 a\_val = A\_row\_vals[k];

31 const uint16\_t col\_idx = A\_row\_idxs[k];

32 bfloat16\* B\_row\_ptr = B + col\_idx \* N + n\_out;

33 int4 b\_vec\_raw = \*(int4\*)(B\_row\_ptr);

34 bfloat162\* b\_vec = (bfloat162\*)(&b\_vec\_raw);

35 const float a = bfloat162float(a\_val);

36

37 for (int t = 0; t < 4; ++t) {

38 float2 b\_f32 = bfloat1622float2(b\_vec[t]);

39 acc[t].x = fmaf(a, b\_f32.x, acc[t].x);

40 acc[t].y = fmaf(a, b\_f32.y, acc[t].y);

41 }

42 }

43

44 bfloat162\* C\_ptr = (bfloat162\*)(C + row \* N + n\_out);

45 for (int i = 0; i < 4; ++i) {

46 C\_ptr[i] = float22bfloat162(acc[i]);

47 }

48 }

49}

Listing 6: Hybrid-to-dense sparse matmul using the ELL component during training using CUDA cores.

In Figure LABEL:lst:appA:sparse\_dense, we provide code listings of our custom kernel used to perform the efficient hybrid-to-dense used during training, focusing on the sparse component. Again, this kernel implements the same dynamic hybrid partitioning logic, skipping the sparse operation in the non-zeros is recognized to exceed the size of the aggressively compact ELL format. In particular, the kernel computes a sparse–dense matrix multiplication C=A​BC=AB, where AA is stored in ELL format and BB and CC are dense row-major matrices. The kernel maps one CTA per output row of CC, which aligns naturally with ELL’s row-wise storage and lets the CTA reuse the same sparse row metadata while sweeping across the output columns. Within a CTA, threads partition the output row into contiguous column tiles. For each tile, they iterate over the non-zeros in the corresponding ELL row of AA and accumulate contributions of the form a⋅B​[c​o​li​d​x,:]a\cdot B[col\_{i}dx,:] into C​[r​o​w,:]C[row,:]. To maximize memory throughput for bfloat16, the kernel accesses BB using 128-bit SIMD loads, so that each thread processes 8 output elements at a time. Accumulation is performed in float32, and the results are written back in vectorized form, providing a simple and efficient SpMM for the fixed-width ELL layout. Rows that exceed the ELL capacity are handled by routing the overflow to the dense backup matrix and computing that portion with Tensor Core–optimized kernels, as described in Algorithm [3](#alg3 "Algorithm 3 ‣ 4 Experimental Results ‣ Sparser, Faster, Lighter Transformer Language Models"). This kernel is used in the forward pass to compute the feedforward layer output. In the backward pass, it is also used to compute gradients with respect to the layer parameters as well as the input activations.

[⬇](data:text/plain;base64,X19nbG9iYWxfXyB2b2lkIGh5YnJpZF90cmFuc3Bvc2UoCiAgICBFTEwqIEEsCiAgICBFTEwqIEFfVCwKICAgIGJmbG9hdDE2KiB0YWlsX0EsCiAgICBiZmxvYXQxNl90KiB0YWlsX0FfVCwKICAgIGludCBNLAogICAgaW50IE4KKQp7CiAgICBmb3IgKGludCByb3cgPSBibG9ja0lkeC54OyByb3cgPCBNOyByb3cgKz0gZ3JpZERpbS54KSB7CiAgICAgICAgY29uc3QgaW50IG5uel9yb3cgPSBBLT5yb3dfY291bnRzW3Jvd107CiAgICAgICAgaWYgKG5uel9yb3cgPT0gMCB8fCBubnpfcm93ID4gRUxMX1dJRFRIKSB7CiAgICAgICAgICAgIGNvbnRpbnVlOwogICAgICAgIH0KCiAgICAgICAgZm9yIChpbnQgayA9IHRocmVhZElkeC54OyBrIDwgbm56X3JvdzsgayArPSBibG9ja0RpbS54KSB7CiAgICAgICAgICAgIGNvbnN0IHVpbnQxNl90IGNvbCA9IEEtPmNvbHNbcm93ICogRUxMX1dJRFRIICsga107CiAgICAgICAgICAgIGNvbnN0IGJmbG9hdDE2IHZhbCA9IEEtPnZhbHNbcm93ICogRUxMX1dJRFRIICsga107CiAgICAgICAgICAgIGNvbnN0IGludCBvdXRfcm93ID0gY29sOwogICAgICAgICAgICBjb25zdCBpbnQgb3V0X2NvbCA9IHJvdzsKICAgICAgICAgICAgY29uc3QgaW50IHBvcyA9IGF0b21pY0FkZChBX1QtPnJvd19jb3VudHNbb3V0X3Jvd10sIDEpOwoKICAgICAgICAgICAgaWYgKHBvcyA8IEVMTF9XSURUSCkgewogICAgICAgICAgICAgICAgY29uc3Qgc2l6ZV90IGFkZHIgPSBvdXRfcm93ICogRUxMX1dJRFRIICsgcG9zOwogICAgICAgICAgICAgICAgQV9ULT5jb2xzW2FkZHJdID0gb3V0X2NvbDsKICAgICAgICAgICAgICAgIEFfVC0+dmFsc1thZGRyXSA9IHZhbDsKICAgICAgICAgICAgfSBlbHNlIHsKICAgICAgICAgICAgICAgIGNvbnN0IGludCBkX3JvdyA9CiAgICAgICAgICAgICAgICAgICAgZ2V0X29yX2FsbG9jYXRlX2RlbnNlX3JvdyhvdXRfcm93LCBBX1QtPnRhaWxfbWFwKTsKICAgICAgICAgICAgICAgIHRhaWxfQV9UW2Rfcm93ICogTSArIG91dF9jb2xdID0gdmFsOwogICAgICAgICAgICB9CiAgICAgICAgfQogICAgfQoKICAgIGZvciAoaW50IGRfcm93ID0gYmxvY2tJZHgueDsgZF9yb3cgPCBBLT50YWlsX3Jvd3M7IGRfcm93ICs9IGdyaWREaW0ueCkgewogICAgICAgIGNvbnN0IGludCBzcmNfcm93ID0gQS0+dGFpbF9tYXBfcmV2ZXJzZVtkX3Jvd107CiAgICAgICAgYmZsb2F0MTZfdCogc3JjID0gdGFpbF9BICsgZF9yb3cgKiBOOwoKICAgICAgICBmb3IgKGludCBjb2wwID0gdGhyZWFkSWR4LnggKiA4OyBjb2wwIDwgTjsgY29sMCArPSBibG9ja0RpbS54ICogOCkgewogICAgICAgICAgICBpbnQ0IHJhdyA9ICooaW50NCopKHNyYyArIGNvbDApOwogICAgICAgICAgICBpZiAoKHJhdy54IHwgcmF3LnkgfCByYXcueiB8IHJhdy53KSA9PSAwKSB7CiAgICAgICAgICAgICAgICBjb250aW51ZTsKICAgICAgICAgICAgfQoKICAgICAgICAgICAgZm9yIChpbnQgaSA9IDA7IGkgPCA4OyArK2kpIHsKICAgICAgICAgICAgICAgIGNvbnN0IGJmbG9hdDE2X3QgdmFsID0gdW5wYWNrX2VsZW1lbnQoJnJhdywgaSk7CiAgICAgICAgICAgICAgICBpZiAodmFsID09IDAuMGYpIHsKICAgICAgICAgICAgICAgICAgICBjb250aW51ZTsKICAgICAgICAgICAgICAgIH0KCiAgICAgICAgICAgICAgICBjb25zdCBpbnQgb3V0X3JvdyA9IGNvbDAgKyBpOwogICAgICAgICAgICAgICAgY29uc3QgaW50IG91dF9jb2wgPSBzcmNfcm93OwogICAgICAgICAgICAgICAgY29uc3QgaW50IHBvcyA9IGF0b21pY0FkZChBX1QtPnJvd19jb3VudHNbb3V0X3Jvd10sIDEpOwoKICAgICAgICAgICAgICAgIGlmIChwb3MgPCBFTExfV0lEVEgpIHsKICAgICAgICAgICAgICAgICAgICBjb25zdCBzaXplX3QgYWRkciA9IG91dF9yb3cgKiBFTExfV0lEVEggKyBwb3M7CiAgICAgICAgICAgICAgICAgICAgQV9ULT5jb2xzW2FkZHJdID0gb3V0X2NvbDsKICAgICAgICAgICAgICAgICAgICBBX1QtPnZhbHNbYWRkcl0gPSB2YWw7CiAgICAgICAgICAgICAgICB9IGVsc2UgewogICAgICAgICAgICAgICAgICAgIGNvbnN0IGludCBkZW5zZV9yb3cgPQogICAgICAgICAgICAgICAgICAgICAgICBnZXRfb3JfYWxsb2NhdGVfZGVuc2Vfcm93KG91dF9yb3csIEFfVC0+dGFpbF9tYXApOwogICAgICAgICAgICAgICAgICAgIHRhaWxfQV9UW2RlbnNlX3JvdyAqIE0gKyBvdXRfY29sXSA9IHZhbDsKICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgfQogICAgICAgIH0KICAgIH0KfQ==)

1\_\_global\_\_ void hybrid\_transpose(

2 ELL\* A,

3 ELL\* A\_T,

4 bfloat16\* tail\_A,

5 bfloat16\_t\* tail\_A\_T,

6 int M,

7 int N

8)

9{

10 for (int row = blockIdx.x; row < M; row += gridDim.x) {

11 const int nnz\_row = A->row\_counts[row];

12 if (nnz\_row == 0 || nnz\_row > ELL\_WIDTH) {

13 continue;

14 }

15

16 for (int k = threadIdx.x; k < nnz\_row; k += blockDim.x) {

17 const uint16\_t col = A->cols[row \* ELL\_WIDTH + k];

18 const bfloat16 val = A->vals[row \* ELL\_WIDTH + k];

19 const int out\_row = col;

20 const int out\_col = row;

21 const int pos = atomicAdd(A\_T->row\_counts[out\_row], 1);

22

23 if (pos < ELL\_WIDTH) {

24 const size\_t addr = out\_row \* ELL\_WIDTH + pos;

25 A\_T->cols[addr] = out\_col;

26 A\_T->vals[addr] = val;

27 } else {

28 const int d\_row =

29 get\_or\_allocate\_dense\_row(out\_row, A\_T->tail\_map);

30 tail\_A\_T[d\_row \* M + out\_col] = val;

31 }

32 }

33 }

34

35 for (int d\_row = blockIdx.x; d\_row < A->tail\_rows; d\_row += gridDim.x) {

36 const int src\_row = A->tail\_map\_reverse[d\_row];

37 bfloat16\_t\* src = tail\_A + d\_row \* N;

38

39 for (int col0 = threadIdx.x \* 8; col0 < N; col0 += blockDim.x \* 8) {

40 int4 raw = \*(int4\*)(src + col0);

41 if ((raw.x | raw.y | raw.z | raw.w) == 0) {

42 continue;

43 }

44

45 for (int i = 0; i < 8; ++i) {

46 const bfloat16\_t val = unpack\_element(&raw, i);

47 if (val == 0.0f) {

48 continue;

49 }

50

51 const int out\_row = col0 + i;

52 const int out\_col = src\_row;

53 const int pos = atomicAdd(A\_T->row\_counts[out\_row], 1);

54

55 if (pos < ELL\_WIDTH) {

56 const size\_t addr = out\_row \* ELL\_WIDTH + pos;

57 A\_T->cols[addr] = out\_col;

58 A\_T->vals[addr] = val;

59 } else {

60 const int dense\_row =

61 get\_or\_allocate\_dense\_row(out\_row, A\_T->tail\_map);

62 tail\_A\_T[dense\_row \* M + out\_col] = val;

63 }

64 }

65 }

66 }

67}

Listing 7: Transposition of the hybrid sparse used during training.

In Figure LABEL:lst:appA:transpose\_sparse, we provide code listings of our custom kernel used to perform efficient transposition of a matrix stored in our hybrid training format. The kernel takes as input a matrix AA and produces ATA\_{T} in the same representation: an ELL matrix, plus a dense backup for rows that overflow the maximum number of non-zeros. It operates in two parts. First, it transposes the non-overflow rows stored in the ELL structure by iterating over each row’s non-zeros and inserting them into the corresponding row of ATA\_{T} (since a non-zero at (row,col)(\texttt{row},\texttt{col}) becomes an entry in row col of the transpose). Because many source rows may map to the same destination row, the kernel uses atomic increments to reserve an insertion slot. If the destination row still has capacity, the entry is written into the ELL arrays of ATA\_{T}; otherwise, it is routed to the dense backup, using a per-row mapping that allocates a dense-tail row on demand. Second, it handles the overflow rows that are materialized in the dense tail of AA. These rows are scanned in vectorized chunks (128-bit loads, i.e., 8 bfloat16 values at a time) with a fast zero check to skip all-zero vectors. Only non-zero elements are emitted into ATA\_{T} using the same hybrid partitioning logic. This approach keeps transposition efficient while preserving the hybrid format and avoiding expensive conversions to more general sparse layouts. After this kernel completes, we launch a small helper kernel to copy the entries stored in the ELL arrays for rows that overflowed into the corresponding dense-backup rows. We note that the necessity of this final small step comes from the fact that dense rows are only allocated and populated after the ELL slots for a given output row have been exhausted.

## Appendix B Hyperparameters and Datasets

### B.1 Training Details

Table 2: Default Hyperparameters for Pretraining Sparse and Non-Sparse LLMs.

|  |  |  |
| --- | --- | --- |
| Hyperparameter | Gated LLM | Non-Gated LLM |
| Model architecture | | |
| Hidden size | 2048 | 2048 |
| Hidden MLP size (intermediate) | 5632 | 8192 |
| Gated MLP | true | false |
| Hidden activation | ReLU | ReLU |
| Number of hidden layers | 8/18/28/38 | 8/18/28/38 |
| Number of attention heads | 32 | 32 |
| Number of key–value heads | 32 | 32 |
| Head dimension | 64 | 64 |
| Attention bias | false | false |
| Attention dropout | 0.0 | 0.0 |
| Initializer range | 0.02 | 0.02 |
| RoPE θ\theta | 10,000 | 10,000 |
| Tied word embeddings | true | true |
| Vocabulary size | 49,152 | 49,152 |
| Tokenizer | GPT2 | GPT2 |
| Computation dtype | bfloat16 | bfloat16 |
| MLP bias | false | false |
| Training setup | | |
| Dataset | fineweb | finewebB |
| Maximum sequence length | 2048 | 2048 |
| Tokens per training step | 1,048,576 | 1,048,576 |
| Training steps | 10K/20K/30K/40K | 10K/20K/30K/40K |
| Total training tokens | 10.49B/20.97B/31.46B/41.94B | 10.49B/20.97B/31.46B/41.94B |
| Optimization | | |
| Optimizer | AdamW | AdamW |
| Learning rate | 1.0×10−31.0\times 10^{-3} | 1.0×10−31.0\times 10^{-3} |
| Weight decay | 0.1 | 0.1 |
| Adam parameters (β1,β2,ϵ)(\beta\_{1},\beta\_{2},\epsilon) | (0.9, 0.95, 1×10−81\!\times\!10^{-8}) | (0.9, 0.95, 1×10−81\!\times\!10^{-8}) |
| Learning rate scheduler | Cosine decay | Cosine decay |
| Warmup steps | 600 | 600 |
| Max grad norm | 1.0 | 1.0 |

As explained in Section [4](#S4.2 "4 Experimental Results ‣ Sparser, Faster, Lighter Transformer Language Models") of the main text, our sparse models and dense baselines in the main text implement a “Transformer++” architecture with gated feedforward blocks, as common in recent LLMs such as Qwen and Llama [llama2, qwen2]. Moreover, in Appendix [C](#A3 "Appendix C Parameter Studies and Ablations ‣ Sparser, Faster, Lighter Transformer Language Models"), we also collect results on a non-gated variant of the same architecture, more similar to the traditional architecture, more similar to the original transformer design [vaswani2017attention]. We train all models using the fineweb [fineweb]. In particular, we consider a deduplicated version of the fineweb-edu split, obtained by from an open corpus used to pretrain the SmolLM family of models [smollm]. We note that all our models are trained with the chinchilla-optimal number of tokens [chinchilla]: around 10B tokens for our 0.5B models, 20B tokens for our 1B models, 30B tokens for our 1.5B models, and 40B tokens for our 2B models.

We provide a full list of hyperparameters and training specifications for our training settings and models in Table [2](#A2.T2 "Table 2 ‣ B.1 Training Details ‣ Appendix B Hyperparameters and Datasets ‣ Sparser, Faster, Lighter Transformer Language Models"). For all models, we use context lengths of 2048 tokens, with batches of 512 sequences, resulting in a global batch size of 1M tokens. For our gated variant, we use a dimensionality of 2048 and a hidden dimension of 5632 in the feedforward blocks, roughly eight-thirds of the hidden size. The main difference with the non-gated variant is that we use a much larger intermediate size of 8192, four times the hidden size, leading to the same total number of parameters. We note that both these choices are considered optimal in the current literature with larger model design practices. When varying model sizes, we modify the number of layers to achieve the target parameter counts. In practice, modern small models have also considered shifting even more of the parameters and flops to the feedforward blocks: for instance, the 1B model of the llama 3 family has a feedforward size of 4x the hidden size even while implementing the gated design [llama3]. While the gains from our sparse kernels could be even greater in these settings, we opted for a more conservative design to avoid biasing our results toward smaller models.

To optimize our models, we use the AdamW optimizer [adamw] with a weight decay of 0.1 and a cosine learning rate schedule starting from a peak learning rate of 1.0×10−31.0\times 10^{-3}, after a small warmup of 600 steps. We use the default Adam parameters of (β1,β2,ϵ)=(0.9,0.95,1×10−8)(\beta\_{1},\beta\_{2},\epsilon)=(0.9,0.95,1\times 10^{-8}) and clip gradients at a maximum norm of 1.0. Our vocabulary of tokens comes from a GPT2 tokenizer [gpt2]. We train using standard mixed precision with the bfloat16 format, with our optimizer states stored in full precision.

### B.2 Task Evaluation Details

We evaluate our models using cloze-formulation scores on seven standard downstream multiple-choice benchmarks that probe logical and commonsense reasoning after pretraining. In particular, we consider ARC (Easy and Hard versions) [bench\_1\_arc], a grade-school science question answering benchmark comprising both Easy and Challenge subsets, with the latter designed to defeat simple retrieval- and co-occurrence-based baselines; HellaSwag [bench\_2\_hellaswag], a commonsense sentence completion task that was designed for counterintuitive LLM challenge; OpenBookQA [bench\_3\_openbook\_qa], focused on probing curated sets of science-based and commonsense knowledge; PIQA [bench\_4\_piqa], a benchmark benchmark focused on physical commonsense reasoning; WinoGrande [bench\_5\_winogrande], a Winograd-style large-scale conference benchmark; and CommonsenseQA [bench\_6\_commonsenseqa], evaluating broader commonsense reasoning. We follow standard evaluation protocols and hyperparameters for formatting the input questions.

#### B.2.1 Sparse data structures sizing

We note that the hybrid training format proposed in this paper introduces two core hyperparameters necessary to fulfill its targeted static allocation design: the ELL maximum number of elements per row, and the number of rows in the dense matrix that stores overflowing elements. Both hyperparameters effectively control a trade-off between performance and memory savings, making their value partially dependent on the sparsity level. Moreover, because sparsity can change abruptly during training, we evaluate a set of sizes that can tolerate sudden decreases in sparsity while remaining performant. In practice, we find that setting the maximum number of elements to 128, and the maximum number of backup rows to one-eighth of the token batch size, to be a robust choice for all sparsity levels above 1.5×10−51.5\times 10^{-5}. Moreover, below this point, simply doubling the ELL non-zeros prevents other instabilities. However, we note that with prior knowledge of the sparsity evolution, these structures can often be made smaller within training itself. For example, for L1=1×10−4L\_{1}=1\times 10^{-4}, we observe that after training stabilizes, we can reduce the number of rows in the dense overflow matrix to 512, enabling higher speedups and additional memory savings. Moreover, the requirements on these two limits differ between the forward and backward passes due to the sparse-matrix transposition used in backpropagation. We note that relevant future extensions could characterize these requirements and develop online tuning of these hyperparameters to improve performance and memory savings further. Finally, when the number of elements exceeds the capacity of our data structures, we currently discard the excess values to avoid a hard failure and set a flag that is reported to the CPU at the next GPU synchronization point. This allows the training system to adaptively increase the structure sizes and repeat the latest training optimization step to avoid any deterioration in the learning dynamics.

## Appendix C Parameter Studies and Ablations

### C.1 Performance and Efficiency Across Activation Functions

Table 3: Comparison of performance and efficiency statistics of sparse LLMs leveraging our kernels with traditional gated models using both ReLU and SiLU activations [SiLU\_gelu, SiLU\_swish, shazeer2020glu].

| Model scale | Activation | Sparse | L1 coeff. | Mean task accuracy | Cross-entropy | # non-zeros | Forward execution [-1pt](input tokens/ms) | Energy per token [-1pt](mJ) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.5B params [-1pt]30B tokens | ReLU | ✗ | 0 | 46.4% | 2.255 | 911 | 117.1 +0(0.0%) | 5.77 +0(0.0%) |
| SiLU | ✗ | 0 | 47.1% | 2.240 | 5632 | 116.5 0(-0.5%) | 5.82 0(+0.1%) |
| ReLU | ✓ | 2×10−52\times 10^{-5} | 46.2% | 2.297 | 29 | 138.0 (+17.9%) | 5.07 (-12.1%) |

As noted in Section [2](#S2 "2 Large Language Models, Feedforward Blocks, and Sparsity ‣ Sparser, Faster, Lighter Transformer Language Models"), many recent LLM architectures have deviated from using ReLUs in favor of smoother activation functions such as SiLU [SiLU\_gelu, SiLU\_swish]. To provide a direct comparison between the two activations, we collect additional training runs on 30B tokens with our 1.5B model and collect efficiency and performance results. In Table [3](#A3.T3 "Table 3 ‣ C.1 Performance and Efficiency Across Activation Functions ‣ Appendix C Parameter Studies and Ablations ‣ Sparser, Faster, Lighter Transformer Language Models"), we find that, while final cross-entropy appears equivalent, SiLU activations indeed yield slightly higher task accuracy in our evaluation set. However, we note that SiLU LLMs are already marginally slower than non-sparse ReLU LLMs by 0.5%, and due to their inherent non-sparse nature, they cannot support integration with sparsity and, therefore, the benefits of our kernels. Overall, we find these results are consistent with the ones from mirzadeh2023relu\_apple\_finetune using larger OPT models [opt] – appearing to indicate that the advantages of smooth activation functions are minor and could potentially be offset by efficiency considerations.

### C.2 Non-gated Sparse LLMs

Table 4: Comparison of performance and efficiency statistics of sparse LLMs leveraging our kernels with traditional baselines, considering both gated models [shazeer2020glu], and their original non-gated counterparts used in the original transformer [vaswani2017attention].

| Model scale | Gated | Sparse | L1 coefficient | Mean task accuracy | Forward execution [-1pt](input tokens/ms) | Energy per token [-1pt](mJ) |
| --- | --- | --- | --- | --- | --- | --- |
| 1.5B params [-1pt]30B tokens | ✓ | ✗ | 0 | 46.36% | 117.1 +0(0.0%) | 5.79 +0(0.0%) |
| ✓ | 2×10−52\times 10^{-5} | 46.20% | 138.0 (+17.9%) | 5.07 (-12.5%) |
| ✓ | 3×10−53\times 10^{-5} | 44.83% | 147.0 (+25.5%) | 4.75 (-18.0%) |
| 1.5B params [-1pt]30B tokens | ✗ | ✗ | 0 | 46.57% | 125.8 +0(0.0%) | 5.52 +0(0.0%) |
| ✓ | 2×10−52\times 10^{-5} | 46.46% | 139.9 (+11.2%) | 5.03 (-8.8%) |
| ✓ | 3×10−53\times 10^{-5} | 44.71% | 142.3 (+13.1%) | 4.86 (-12.0%) |

As explained in Section [2](#S2 "2 Large Language Models, Feedforward Blocks, and Sparsity ‣ Sparser, Faster, Lighter Transformer Language Models"), from the simple 2-layer feed-forward block used in the original transformer, there has been a notable shift, with modern models adopting a gated variant due to small but consistent superior empirical results. Nonetheless, in our work, we introduce training and inference kernels for both variants.
In contrast to the gated variant, computing the output activations following [1](#S2.E1 "Equation 1 ‣ 2.1 Feed-forward Modules as Sparse Knowledge Stores ‣ 2 Large Language Models, Feedforward Blocks, and Sparsity ‣ Sparser, Faster, Lighter Transformer Language Models"), for the non-gated variant, the computation simplifies to:

|  |  |  |  |
| --- | --- | --- | --- |
|  | h=ϕ​(x​Wu),y=h​Wd,h=\phi(xW\_{u}),y=hW\_{d}, |  | (5) |

where ϕ\phi is, once again, the non-linear activation function. Thus, when ϕ\phi is a ReLU activation, the sparsity pattern is determined by the up-projection rather than the gate projection. For inference kernels, we note this implies that a difference between the two variants is that the non-gated version performs the up projection rather than the gate projection with our matrix multiplication kernel with TwELL storage introduced in Section [3](#S3.2 "3 Making Sparse LLMs Fast ‣ Sparser, Faster, Lighter Transformer Language Models"). Moreover, as detailed in Appendix [A](#A1 "Appendix A Kernels Implementation Details ‣ Sparser, Faster, Lighter Transformer Language Models"), we designed an additional kernel optimized to perform the down projection alone starting from the TwELL format.

Thus, to provide a direct comparison between the two variants and the relative effects of sparsity and our custom kernels, we collect additional training runs on 30B tokens with our 1.5B model implementing the non-gated parameterization. In particular, we consider two sparsity levels in addition to a non-sparse baseline (L1=0L\_{1}=0): our recommended conservative regularization of L1=2×10−5L\_{1}=2\times 10^{-5} and a more aggressive regularization of L1=3×10−5L\_{1}=3\times 10^{-5}. In Table [4](#A3.T4 "Table 4 ‣ C.2 Non-gated Sparse LLMs ‣ Appendix C Parameter Studies and Ablations ‣ Sparser, Faster, Lighter Transformer Language Models"), we report the collected relative performance and efficiency results for both variants and all three sparsity levels. As shown, we find only minor performance differences between the two variants, which are likely not significant and attributable to random variations. However, we do note that such differences might become visible only when training with token budgets beyond chinchilla optimality [chinchilla]. Efficiency-wise, while both our variants benefit significantly from our sparse kernels, we find such benefits to be larger for the gated variant. The inference speedup of the non-gated variant is 11.2% at L1=2×10−5L\_{1}=2\times 10^{-5} compared to 17.9% for the gated variant at the same sparsity level. At larger sparsity levels, this divide is more pronounced, with the gated variant achieving a 25.5% speedup at L1=3×10−5L\_{1}=3\times 10^{-5} compared to only 13.1% for the non-gated variant. These results are intuitively based on the nature of both models, as the gated variant allows our new inference kernels to leverage the opportunity of efficient fast fusion of both up and down projections. Nonetheless, they also demonstrate that the benefits of our kernels extend beyond a single architectural choice.

### C.3 Strategies for Dead Neuron Mitigation

Table 5: Comparison of performance and efficiency statistics of sparse LLMs leveraging our kernels with traditional baselines trained using our standard recipe, or with dead neuron mitigation strategies such as warming up the coefficient of the L1 loss and applying targeted reinitialization to the gate projection’s weights.

| Model scale | Training [-1pt]modification | Sparse | L1 coefficient | Mean task accuracy | Cross-entropy | # non-zeros | Forward execution [-1pt](input tokens/ms) | Energy per token [-1pt](mJ) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.5B params [-1pt]30B tokens | – | ✗ | 0 | 46.4% | 2.255 | 911 | 117.1 +0(0.0%) | 5.77 +0(0.0%) |
| – | ✓ | 2×10−52\times 10^{-5} | 46.2% | 2.297 | 29 | 138.0 (+17.9%) | 5.07 (-12.1%) |
| Dead neuron reinit. | ✓ | 2×10−52\times 10^{-5} | 46.6% | 2.298 | 29 | 139.4 (+19.1%) | 4.96 (-14.0%) |
| Sparsity warmup | ✓ | 3×10−43\times 10^{-4} | 45.9% | 2.293 | 108 | 119.3 (+1.9%) | 5.76 (-0.1%) |



![Refer to caption](/html/2603.23198/assets/x9.png)

![Refer to caption](/html/2603.23198/assets/x10.png)

Figure 8: Number of non-zeros and fraction of dead neurons of LLMs with different strategies for dead neuron mitigation throughout training.

While we find that using an L1 coefficient of L1=2×10−5L\_{1}=2\times 10^{-5} provides a relevant boost in efficiency without any noticeable downstream performance degradation, we explore preliminary directions to mitigate the potential downsides of sparse training. In particular, as detailed in Appendix [D](#A4 "Appendix D Extended Results ‣ Sparser, Faster, Lighter Transformer Language Models"), when examining the number of active neurons throughout training, we see that over 30% of the neurons become permanently inactive on average across layers when using our recommended L1 coefficient, with this metric considerably rising for higher regularizations. While for our recommended coefficient, this symptom does not seem to evidently reflect on downstream performance, reducing such an effect could potentially allow supporting even higher sparsity before incurring performance degradation.

Based on these considerations, we explore two preliminary extensions to our simple L1-regularized training recipe explained in Section [2](#S2 "2 Large Language Models, Feedforward Blocks, and Sparsity ‣ Sparser, Faster, Lighter Transformer Language Models"). First, we consider simply scheduling the L1 regularization, motivated by our findings that dead neurons appear to arise very early during training. Concretely, we first train our models for 5000 steps without any L1 regularization, followed by a further 5000 steps of linear increase of the L1 coefficient. We make the training setting artificially similar to prior work that focuses on finetuning and continued-pretraining [song2024prosparse, q\_sparse\_top\_k\_related]. Second, we consider implementing a target reinitialization strategy to lower the magnitude and reinject random noise only in the columns of the gate projection that lead to always negative outputs (which then, after ReLU, lead to dead neurons). Given the model’s initialization standard deviation σ=0.02\sigma=0.02, we noised and rescaled to regress the weights to their initial state, essentially interpolating with a coefficient λ\lambda:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Wg​[:,j]←(1−λ)​Wg​[:,j]+λ​𝒩​(0,σ2),W\_{g}[:,j]\leftarrow(1-\lambda)W\_{g}[:,j]+\lambda\mathcal{N}(0,\sigma^{2}), |  | (6) |

We apply this targeted reinitialization after every training step, which we find does not significantly affect training time. In preliminary experiments, We found λ=0.1\lambda=0.1 to be a good choice that avoids affecting training dynamics while injecting sufficient noise to revive dead neurons. We note this strategy is similar to older techniques for reinjecting plasticity into architectures in continual learning and other non-stationary settings [warmstarting\_nn].

In Table [5](#A3.T5 "Table 5 ‣ C.3 Strategies for Dead Neuron Mitigation ‣ Appendix C Parameter Studies and Ablations ‣ Sparser, Faster, Lighter Transformer Language Models"), we report the performance and efficiency results of our two strategies compared to our standard recipe and the non-sparse baseline, while in Figure [8](#A3.F8 "Figure 8 ‣ C.3 Strategies for Dead Neuron Mitigation ‣ Appendix C Parameter Studies and Ablations ‣ Sparser, Faster, Lighter Transformer Language Models") we analyze the number of non-zero activations and dead neurons throughout training. When looking at the dead neuron statistics, we find that both strategies almost entirely mitigate the emergence of dead neurons. However, we immediately see a concerning pattern with the sparsity-warmup strategy, as the number of non-zeros considerably increases throughout training. In particular, the considered coefficient of L1=3×10−4L\_{1}=3\times 10^{-4}, which is ten times larger than our recommended value, leads to over 100 non-zeros on average across layers at the end of training, compared to only 29 non-zeros when using our standard recipe with L1=2×10−5L\_{1}=2\times 10^{-5}. We note that, in early experiments, we found that increasing the L1 coefficient further led to training instabilities and loss spikes. In contrast, using the targeted dead neuron reinitialization, we find similar non-zero statistics to our standard recipe while still effectively mitigating dead neurons. Furthermore, as reported in Table [5](#A3.T5 "Table 5 ‣ C.3 Strategies for Dead Neuron Mitigation ‣ Appendix C Parameter Studies and Ablations ‣ Sparser, Faster, Lighter Transformer Language Models"), we find that this latter strategy provides a small boost in both downstream performance and efficiency, processing tokens 19.1% faster than the non-sparse baseline with our default L1 coefficient of L1=2×10−5L\_{1}=2\times 10^{-5}. We believe these preliminary results suggest that further research in examining optimal sparse training would potentially further increase the relevance and efficiency upsides of sparse LLMs.

## Appendix D Extended Results

### D.1 Sparsity and Dead Neurons During Training

![Refer to caption](/html/2603.23198/assets/x11.png)

![Refer to caption](/html/2603.23198/assets/x12.png)

Figure 9: Number of non-zeros and fraction of dead neurons of LLMs across L1 regularization levels throughout training.

In Figure [9](#A4.F9 "Figure 9 ‣ D.1 Sparsity and Dead Neurons During Training ‣ Appendix D Extended Results ‣ Sparser, Faster, Lighter Transformer Language Models"), we provide detailed results about how activation sparsity and dead neuron occurrence evolve during training for all our different L1 regularization levels. In particular, we record dead neurons at the end of each training step by keeping track, for each hidden feedforward activation of each layer, the last time it was non-zero. If a neuron was never active for a whole training step (just above 1M tokens), we consider it dead for that step.

We make two immediate observations from these results. First, we find that the sparsity levels settle early on to low values after only around 1000 training steps (around 1B tokens). Due to this property, we note that the throughput and memory advantages of our training kernels become relevant almost at the inception of our training runs. Second, we observe that the same trend applies to the number of dead neurons: our recommended L1=2×10−5L\_{1}=2\times 10^{-5} already exceeds 30% inactivity, which further monotonically increases with higher regularization levels. While for our recommended coefficient, this symptom does not seem to evidently reflect on downstream performance, reducing such an effect could potentially allow supporting even higher sparsity before incurring performance degradation. To this end, we note that in Appendix [C](#A3 "Appendix C Parameter Studies and Ablations ‣ Sparser, Faster, Lighter Transformer Language Models") we provide preliminary results indicating that the performance of sparse LLMs could be further improved with strategies targeted at dead-neuron mitigation.

### D.2 Task Performance Details

Table 6: Granular comparison of per-task downstream performance across model scales to complement Table [1](#S4.T1 "Table 1 ‣ 4.2 More Efficient LLMs with Unstructured Sparsity ‣ 4 Experimental Results ‣ Sparser, Faster, Lighter Transformer Language Models").

| Model scale | Sparse | Mean Accuracy | HellaSwag | CQA | PIQA | Winogrande | ARC-easy | ARC-challenge | OpenBookQA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5B params [-1pt]10B tokens | ✗ | 040.4% | 033.7% | 020.9% | 064.5% | 050.9% | 064.1% | 028.1% | 020.8% |
| ✓ | 040.4% | 034.0% | 022.3% | 066.4% | 053.8% | 060.5% | 027.5% | 018.0% |
| 1B params [-1pt]20B tokens | ✗ | 044.6% | 039.2% | 020.0% | 068.7% | 054.4% | 072.6% | 034.0% | 023.6% |
| ✓ | 044.7% | 039.8% | 018.6% | 068.1% | 054.8% | 071.6% | 035.3% | 024.4% |
| 1.5B params [-1pt]30B tokens | ✗ | 046.4% | 041.0% | 020.8% | 070.2% | 055.9% | 072.5% | 036.7% | 027.4% |
| ✓ | 046.2% | 041.1% | 021.0% | 069.1% | 054.4% | 074.3% | 037.5% | 026.0% |
| 2B params [-1pt]40B tokens | ✗ | 049.1% | 045.7% | 021.0% | 072.0% | 057.8% | 077.2% | 041.7% | 028.6% |
| ✓ | 048.8% | 045.0% | 021.3% | 070.9% | 057.5% | 075.6% | 042.2% | 028.8% |

To complement the results in Section [4](#S4.2 "4 Experimental Results ‣ Sparser, Faster, Lighter Transformer Language Models") in the main text, we provide the detailed granular results of downstream task performance across the seven downstream tasks considered, targeting logic and reasoning capabilities after pretraining [bench\_1\_arc, bench\_2\_hellaswag, bench\_3\_openbook\_qa, bench\_4\_piqa, bench\_5\_winogrande, bench\_6\_commonsenseqa]. In particular, we report the per-task accuracies for both sparse models, using our recommended conservative L1 regularization of 2×10−52\times 10^{-5}, and their non-sparse counterparts across all the examined model scales. As shown in Table [6](#A4.T6 "Table 6 ‣ D.2 Task Performance Details ‣ Appendix D Extended Results ‣ Sparser, Faster, Lighter Transformer Language Models") and consistently with our main text analysis, we do not find significant performance differences between sparse and non-sparse models for our regularization level and all considered tasks. We do, indeed, observe an expected performance rise with larger models across the great majority of tasks.

### D.3 Activation Sparsity at High and Low Levels

![Refer to caption](/html/2603.23198/assets/x13.png)


Figure 10: Sparsity statistics and speedup contributions across different layers of non-sparse LLMs.

![Refer to caption](/html/2603.23198/assets/x14.png)


Figure 11: Sparsity statistics and speedup contributions across different layers of an LLM with high regularization L1=104L\_{1}=10^{4}.

![Refer to caption](/html/2603.23198/assets/x15.png)


Figure 12: Training speedups from our sparse LLM training kernels across L1 regularization levels for both H100 and RTX6000 devices.

To complement the analysis results provided in Section [4](#S4.2 "4 Experimental Results ‣ Sparser, Faster, Lighter Transformer Language Models") of the main text, we examine how sparsity regularization affects the distribution of non-zero activations across model depth and relate these metrics to the corresponding speed-up contributions from our kernels during inference. While in our main analysis we reported and analyzed the LLM trained with our recommended conservative L1 regularization of 2×10−52\times 10^{-5}, in Figures [10](#A4.F10 "Figure 10 ‣ D.3 Activation Sparsity at High and Low Levels ‣ Appendix D Extended Results ‣ Sparser, Faster, Lighter Transformer Language Models") we provide analogous results for a non-sparse LLM while in Figure [11](#A4.F11 "Figure 11 ‣ D.3 Activation Sparsity at High and Low Levels ‣ Appendix D Extended Results ‣ Sparser, Faster, Lighter Transformer Language Models") we analyze an LLM trained with the highest regularization regularization level considered (1×10−41\times 10^{-4}). We note that for non-sparse models, due to the high number of non-zeros, the contributions of applying our kernel are actually detrimental – and as such, we report the speed-up contributions as negative percentages. A first observation from the sparsity statistics is that the average number of non-zeros also follows a noticeable trend in the non-sparse model, with the first few layers being the least active, followed by a hump with a peak in activations. However, a key difference comes with the location of the hump: while in our recommended sparse model the peak occurs around layer 6, in the non-sparse LLM the peak still occurs within the first half of the network but is shifted visibly deeper into the architecture around layer 13. Interestingly, in the high-regularization LLM, we actually observe that while the very first layer is again the least active, there are two different peaks – one very early around the second layer and another one in the last layer of the model. Once again, we find that maximum activation counts can easily be well over an order of magnitude higher than the average, with no clear pattern across layers. For the non-sparse model, we again observe a strong inverse correlation between each layer’s average non-zeros and its relative speed-up. In contrast to the high-regularization LLM, this correlation is much less visible, as given the high sparsity encountered, the speedups of our kernels are already at their achievable maximum for almost all layers, essentially making executing the up and down projection negligible in the overall computation time.

### D.4 Improving Efficiency of other Devices

As mentioned in Section [4](#S4.2 "4 Experimental Results ‣ Sparser, Faster, Lighter Transformer Language Models"), given that our kernels consistently reduce memory requirements during training, and as a side benefit, reduce reliance on newer tensor core units, they immediately have higher potential relevance for less capable hardware. Thus, to empirically validate these considerations, we provide additional results comparing the performance speedups of our kernels during training on NVIDIA’s RTX PRO 6000 GPUs against the H100 PCIe GPUs used throughout our main paper and other experiments. Some of the other crucial differences of this GPU come from the memory side, with a considerably reduced memory bandwidth (1.59 TB/s vs. 2.0 TB/s). In contrast, the RTX PRO 6000 can benefit from a larger number of Streaming Multiprocessors than the H100 (188 vs. 114), potentially allowing for greater occupancy for sparse workloads.

As shown in Figure [12](#A4.F12 "Figure 12 ‣ D.3 Activation Sparsity at High and Low Levels ‣ Appendix D Extended Results ‣ Sparser, Faster, Lighter Transformer Language Models"), and in line with our considerations, we find significantly higher speedups on the RTX 6000 GPU across all L1 regularization levels considered. These speedup differences are even more pronounced at higher regularization levels, extending the practical range of L1 coefficients make sparsity provide meaningful efficiency improvements. When dissecting what causes these greater speedups, we first find that thanks to the specific H100 features, such as the higher tensor cores throughput, the runtime of the dense GEMM operations increases from around 400 to 800 microseconds on the RTX 6000. Similarly, kernels that are memory bandwidth bound, including the dense to hybrid matrix multiplication, are also slightly slower by 19% on the RTX 6000 than on the H100. However, once in our hybrid sparse format, due to the larger Streaming Multiprocessors count of the RTX 6000 GPU, the sparse operations run faster than on the H100, with speedups of 1.34×\times and 2.1×\times for sparse-to-dense and transposition operations, respectively. We find these results indicate that leveraging sparsity with targeted kernels could significantly improve the performance of cheaper devices, which do not implement the latest hardware innovations of higher-end units such as the H100, lowering the field’s canonical hardware barriers.

## Appendix E Further Related Work

### E.1 Activation Sparsity in Transformers

Expanding on the findings of zhang-etal-2022-moefication, li2023lazyneuronphenomenonemergence documents that Transformer MLP layers with ReLU activations exhibit inherent activation sparsity across architectures, depths, and data distributions.
Building on this observation, mirzadeh2023relu\_apple\_finetune shows that replacing GELU with ReLU in non-gated feed-forward layers yields negligible performance degradation while enabling up to three times theoretical inference speedup with less computation.
However, they focus on older architectures (OPT models) with non-gated feed-forward blocks and leave efficient kernel implementation to future work.

More recent methods have also been proposed to enhance sparsity after altering modern gated architectures and have claimed speedups when running sparse feedforward layers in isolation on older generations of devices. TurboSparse [song2024turbo] proposes a modification to the feed-forward block itself, introducing dReLU, which applies ReLU to *both* gate and up projections: h=ReLU​(x​Wg)⊙ReLU​(x​Wu)h=\mathrm{ReLU}(xW\_{g})\odot\mathrm{ReLU}(xW\_{u}). ProSparse [song2024prosparse] proposes finetuning pretrained models and artificial thresholding of the activations to increase sparsity. Q-Sparse [q\_sparse\_top\_k\_related], further deviates from standard architectures via maintaining only the top-K activations and applying a straight-through estimator. We also note that additional works proposed introducing structured sparsity post-training, such as by predicting [dejavu\_contectual\_sparsity\_related] and pruning activation to set sparsity levels [cats\_post\_training\_thresh\_related, teal\_post\_training\_thresh\_related]. Unlike these works, our paper introduces general-purpose kernels to leverage unstructured sparsity, demonstrating empirical efficiency benefits during LLM training and inference.

### E.2 Architectural Approaches to Sparsity

Mixture-of-Experts (MoE) architectures [DBLP:journals/corr/ShazeerMMDLHD17, lepikhin2020gshard, fedus2022switch] partition feed-forward layers into separately routed experts, decoupling model capacity from per-token computation.
However, MoE requires predetermining the number of experts and sparsity level before training, limiting adaptability to input complexity.

Product key memory [lample2019largememorylayersproduct] maintains fixed sparsity patterns through O​(log⁡n)O(\log n) key retrieval.
PEER [he2024mixturemillionexperts] extends this approach to over one million single-neuron experts with 99.99% architectural sparsity.
UltraMem [huang2025ultrasparsememorynetwork] improves PKM and scales to 20 million memory slots, showing that it can outperform MoE with the same parameter and computation budgets.
Fast Feedforward Networks [belcak2023fastfeedforwardnetworks] use differentiable binary trees to achieve 99% sparsity.

While these architectural approaches achieve extreme sparsity, they require substantial modifications to standard Transformer training pipelines. Our approach instead works with conventional architectures, requiring only a change of activation function and optional regularization, making it readily applicable to existing models and training infrastructure.

[◄](/html/2603.23197)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2603.23198)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2603.23198)
[View original  
on arXiv](https://arxiv.org/abs/2603.23198)[►](/html/2603.23199)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Mon Apr 6 05:45:15 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
