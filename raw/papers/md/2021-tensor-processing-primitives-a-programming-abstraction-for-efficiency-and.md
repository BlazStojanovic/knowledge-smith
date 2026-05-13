---
arxiv: '2104.05755'
authors:
- Evangelos Georganas
- Dhiraj Kalamkar
- Sasikanth Avancha
- Menachem Adelman
- Deepti Aggarwal
- Cristina Anderson
- Alexander Breuer
- Jeremy Bruestle
- Narendra Chaudhary
- Abhisek Kundu
- Denise Kutnick
- Frank Laub
- Vasimuddin Md
- Sanchit Misra
- Ramanarayan Mohanty
- Hans Pabst
- Brian Retford
- Barukh Ziv
- Alexander Heinecke
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'Tensor Processing Primitives: A Programming Abstraction for Efficiency and
  Portability in Deep Learning & HPC Workloads'
url: https://arxiv.org/abs/2104.05755
year: 2021
---

[2104.05755] Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads














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



\UseRawInputEncoding

# Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads

Evangelos Georganas∗, Dhiraj Kalamkar∗, Sasikanth Avancha∗, Menachem Adelman∗, Deepti Aggarwal∗, Cristina Anderson∗, Alexander Breuer#, Jeremy Bruestle∗, Narendra Chaudhary∗, Abhisek Kundu∗, Denise Kutnick∗, Frank Laub∗, Vasimuddin Md∗, Sanchit Misra∗, Ramanarayan Mohanty∗, Hans Pabst∗, Brian Retford∗, Barukh Ziv∗, Alexander Heinecke∗
  
  
∗Intel Corporation
  
#Friedrich-Schiller–Universit t Jena

###### Abstract.

During the past decade, novel Deep Learning (DL) algorithms, workloads and hardware have been developed to tackle a wide range of problems. Despite the advances in workload and hardware ecosystems, the programming methodology of DL systems is stagnant. DL workloads leverage either highly-optimized, yet platform-specific and inflexible kernels from DL libraries, or in the case of novel operators, reference implementations are built via DL framework primitives with underwhelming performance. This work introduces the Tensor Processing Primitives (TPP), a programming abstraction striving for efficient, portable implementation of DL workloads with high-productivity. TPPs define a compact, yet versatile set of 2D-tensor operators (or a virtual Tensor ISA), which subsequently can be utilized as building-blocks to construct complex operators on high-dimensional tensors. The TPP specification is platform-agnostic, thus code expressed via TPPs is portable, whereas the TPP implementation is highly-optimized and platform-specific. We demonstrate the efficacy and viability of our approach using standalone kernels and end-to-end DL & HPC workloads expressed entirely via TPPs that outperform state-of-the-art implementations on multiple platforms.

## 1. Introduction

Since the advent of Deep Learning (DL) as one of the most promising machine learning paradigms almost 10 years ago, deep neural networks have advanced the fields of computer vision, natural language processing, recommender systems, and gradually pervade an increasing number of scientific domains ([origalexnet,](#bib.bib1) ; [szegedy2015going,](#bib.bib2) ; [simonyan2014very,](#bib.bib3) ; [yu2013feature,](#bib.bib4) ; [wu2016google,](#bib.bib5) ; [cheng2016wide,](#bib.bib6) ; [wolf2020transformers,](#bib.bib7) ; [gawehn2016deep,](#bib.bib8) ; [goh2017deep,](#bib.bib9) ; [raghu2020survey,](#bib.bib10) ).
Due to the diverse nature of the problems under consideration, these DL workloads exhibit a wide range of computational characteristics and demands. Furthermore, due to the immense computational cost of such workloads, industry and academia have developed specialized hardware features on commodity processors, and even specialized accelerators in order to harness these computational needs ([alom2019state,](#bib.bib11) ).

In contrary to the fast-evolving ecosystems of DL workloads and DL-oriented hardware/accelerators, the programming paradigm of DL systems has reached a plateau ([barham2019machine,](#bib.bib12) ). More specifically, the development of novel DL workloads involves two types of components: i) Well-established operators within DL libraries (e.g. 2D convolutions, inner-product, batch-norm layers in oneDNN ([onednn,](#bib.bib13) ) and cuDNN ([chetlur2014cudnn,](#bib.bib14) )), and ii) Unprecedented, custom primitives which typically instantiate new algorithmic concepts/computational motifs. Unfortunately both of these components come with their shortcomings.

On one hand, the operators within DL libraries are heavily optimized and tuned (usually by vendors) in a platform-specific fashion, leading to monolithic, non-portable and inflexible kernels. Additionally, such opaque and high-level operators prohibit modular design choices since the user/frameworks have to adhere to particular interfaces that may not be adapted to fit the operation under consideration. On the other hand, the custom/unprecedented primitives are typically implemented by the user via the available generic/reference primitives of an ML framework which are not optimized and as such yield underwhelming performance. It is up to the user to create optimized implementations for the custom primitives, leading again to code which is non-portable and potentially requires hardware expertise in order to achieve peak performance. Unfortunately, most of the times such expertise is not available to the data/ML scientist who is developing the custom DL primitive. Therefore, the deployment (or even the evaluation) of a new operator typically requires yet another stage in the development cycle where low-level optimization experts are working on the re-write/fine-tuning of the operator. Later on, in case an operator proves to be important for the community, systems researchers and vendors standardize it, and potentially create yet another monolithic kernel within a DL library for further re-use within DL frameworks. This entire development cycle potentially takes a considerable amount of time (up to years in some cases) and inadvertently impedes the efficient exploration of innovative machine learning ideas ([barham2019machine,](#bib.bib12) ). An alternative approach to optimize both types of operators is to leverage contemporary Tensor Compilers (e.g.  ([plaidml,](#bib.bib15) ; [chen2018tvm,](#bib.bib16) ; [vasilache2018tensor,](#bib.bib17) ; [zheng2020ansor,](#bib.bib18) )), however the state-of-the-art tools are only suitable for compiling small code-blocks whereas large-scale operators require prohibitive compilation times, and often the resulting code performs far from the achievable peak ([barham2019machine,](#bib.bib12) ).

We identify that the common source of the problems mentioned in the previous paragraph is the extreme levels of abstraction offered by the DL libraries and the Tensor Compilers. The DL libraries offer coarse-grain, monolithic and inflexible operators whereas the Tensor Compilers usually go to the other extreme, allowing the user to express arbitrary low-level operators without any minimal restrictions that would readily enable efficient lifting and code-generation in their back-ends (e.g. they offer no minimal/compact set of allowed operations on tensors). To exacerbate the challenge of optimal code generation, Tensor Compilers usually undertake the cumbersome tasks of efficient parallelization, loop re-ordering, automatic tiling and layout transformations, which, to date, remain unsolved in the general setup. Also, there is not a well-established way to share state-of-the-art optimizations among the plethora of Tensor Compilers and as a result each one has its own advantages and disadvantages, which translates eventually to sub-optimal performance on real-world scenarios ([li2020deep,](#bib.bib19) ). We note here the recent, promising effort of MLIR ([mlir,](#bib.bib20) ) towards unifying the optimization efforts in the Tensor Compiler IR infrastructure.

In this work we introduce the Tensor Processing Primitives (TPP), a programming abstraction striving for efficient and portable implementation of Tensor operations, with a special focus on DL workloads. TPPs define a set of relatively low-level primitive operators on 2D Tensors, which in turn can be used as basic building blocks to construct more complex operators on high-dimensional tensors. TPPs comprise a minimal and compact, yet expressive set of precision-aware, 2D tensor level operators to which high-level DL operators can be reduced. TPPs’s specification is agnostic to targeted platform, DL framework, and compiler back-end. As such the code which is expressed in terms of TPPs is portable. Since the level of abstraction that TPPs adopt is at the sub-tensor granularity, TPPs can be directly employed by DL workload developers within the frameworks, or could be alternatively used to back up an IR within a Tensor Compiler stack, i.e. TPPs could form the basis of an MLIR dialect.

While the TPP specification is agnostic of the targeted framework/platform/compiler stack, its implementation is platform specific, and is optimized for the target architectures. This subtle detail offers a clear separation of concerns: the user-entity of TPPs, either a developer or a compiler framework, can focus on expressing the desired algorithm and its execution schedule (e.g. parallelization, loop orders) using the TPP tensor abstraction, whereas the efficient, platform-specific code generation pertaining to the TPP operations belongs to the TPP back-end. To this extent, TPPs could be also viewed as a “virtual Tensor ISA” that abstracts the actual physical ISA of the target (e.g. SSE, AVX2, AVX512, AMX for x86, AArch64 and ARMv8 SVE , xPU).

![Refer to caption](/html/2104.05755/assets/x1.png)


Figure 1. Use-cases of TPPs in various software stacks.

Figure [1](#S1.F1 "Figure 1 ‣ 1. Introduction ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") shows various use-cases of TPPs within multiple software stacks. TPPs can be viewed as a layer abstraction of the actual physical target ISA, and the user-entities can rely on the TPP layer for the code generation pertaining to the tensor operations. Also, Figure [1](#S1.F1 "Figure 1 ‣ 1. Introduction ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") illustrates the various user-entities that might leverage TPPs. First, the vendor-optimized DL libraries (e.g. oneDNN or oneDNN Graph) can use TPPs for optimized code generation in their back-end. Second, the user/developer of the DL operators can directly leverage TPPs within a DL framework extension to express the underlying tensor computations (e.g. the user may develop a framework extension for a novel DL operator by employing the TPPs as building blocks). Third, Tensor Compilers can leverage TPPs (e.g. as part of an MLIR dialect) to generate high-quality code for the corresponding tensor operators. As such, the TPP layer abstraction offers a clear separation of concerns where the Tensor Compiler may focus on higher-level optimizations (loop tiling and re-ordering, parallelization etc) whereas the platform-specific code generation of the tensor operations is undertaken by the TPP layer. Such a synergistic Tensor Compiler - TPP paradigm is illustrated in Section [7](#S7 "7. TPP within MLIR and a Tensor Compiler ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads"). Last but not least, TPPs could be leveraged by more general Tensor Libraries (e.g. ATen, Eigen) where tensor computations constitute the primary focus and they can be naturally mapped to TPPs.

In our Proof-Of-Concept (POC) implementation of TPPs we leverage JIT technology to emit performant and platform-specific code during runtime. Furthermore, in our POC we define a mini embedded Domain Specific Language (mini-eDSL) where the TPPs can be combined via matrix equations in order to build high-level operators without sacrificing performance.

We demonstrate the efficiency of our approach on multiple platforms using standalone kernels written entirely with TPPs and compare the performance to vectorized-by-expert code and compiler generated code. Finally, we showcase the expressiveness and viability of our methodology by implementing contemporary end-to-end DL workloads using solely the TPP abstractions and show how we can outperform the state-of-the-art implementations on multiple platforms. The main contributions of this work are:

* •

  A TPP specification/foundation for primitive tensor operations.
* •

  A Proof-Of-Concept implementation of the TPP specification along with a mini-eDSL (called TPP Matrix Equations), enabling efficient fusion of TPPs that lead to portable, high-level tensor operations. We describe in detail various standalone TPP implementations, and also we provide a detailed analysis of our TPP Matrix Equation mini-eDSL framework.
* •

  A demonstration of how contemporary and novel DL algorithmic motifs/workloads can be expressed in their entirety via TPPs.
* •

  An experimental evaluation of the TPP-based DL workloads from all relevant fields (image processing, recommendation systems, natural language processing, graph processing and applications in science) on multiple platforms (different instruction set architectures (ISAs) x86\_64 and aarch64, and micro-architectures for each ISA), including distributed-memory scaling. We show performance that matches/exceeds the state-of-the-art implementations, while maintaining flexibility, portability and obviating the need for low-level platform-specific optimizations.
* •

  We show how TPPs can be leveraged as a virtual Tensor ISA within a Tensor compiler software stack, yielding high-performance DL primitives.
* •

  We illustrate examples of how TPPs are used outside of Deep Learning, in High Performance Computing (HPC) applications in order to accelerate tensor computations.

Section [2](#S2 "2. The TPP Specification ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") details the specification of the TPPs. Then, Section [3](#S3 "3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") illustrates a POC implementation of the TPP specification. Section [4](#S4 "4. TPP Matrix Equations ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") presents an infrastructure that enables efficient TPP fusion. In Section [5](#S5 "5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") we exhibit how contemporary DL motifs/algorithmic paradigms can be expressed via TPPs. Section [6](#S6 "6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") presents an experimental evaluation of TPP-based DL kernels and workloads on multiple platforms. Section [7](#S7 "7. TPP within MLIR and a Tensor Compiler ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") outlines our POC implementation of a TPP backend within a tensor compiler (PlaidML), and also presents some results highlighting the viability of the TPP abstraction as a virtual Tensor ISA within tensor compiler stacks. Section [8](#S8 "8. TPP and HPC Applications ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") presents exemplary usage of TPPs within HPC applications in order to efficiently implement tensor computations.
Sections [9](#S9 "9. Related Work ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") and [10](#S10 "10. Conclusions And Future Work ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") summarize the related work and conclude this paper.

## 2. The TPP Specification

### 2.1. TPP Design Principles

The TPP specification is driven by a few design principles:

1) *Each TPP corresponds to a mathematical operator that takes a number of input(s) and produces an output*. We opt to specify TPPs that correspond to basic, well-defined mathematical tensor operations. In this way we keep the set of TPPs *minimal* albeit *expressive*; basic TPPs can be combined to formulate more complex operators.

2) *The inputs/outputs of the TPPs are abstract 2D tensors that can be fully specified by their shape/size, leading dimensions, and precision*. Additionally, the 2D tensors hold the following complementary *runtime* information: (i) a *primary* field which corresponds to the memory address where the 2D (sub)tensor data resides, (ii) a *secondary* field holding optional data for the tensor (e.g. a mask for the tensor), and (iii) a *tertiary* field holding optional, auxiliary information of the tensor (e.g. scaling factors for a quantized tensor.)

3) *TPPs are specified as “memory-to-memory” operations, or equivalently the input/output tensors are residing in memory locations specified by the user*. This design decision is critical in order to abstract the TPPs from all physical ISAs, and enables true platform-agnostic specification. For example, if the TPPs were accepting vector registers as inputs/outputs, then the number of physical registers, the vector length and dimensionality would be exposed in the API of TPPs, making the specification platform-specific.

4) *TPPs have declarative semantics*. As such, the TPP specification does not preclude potential parallelism (e.g. SIMD, SIMT) in the back-end implementation which is target-specific.

5) *TPPs are composable in a producer-consumer fashion*. Since the output of a TPP is a well-defined tensor O𝑂O, it can be fed as input to a subsequent TPP. In such a scenario, this “intermediate” tensor O𝑂O is not necessarily exposed to the user, unless the user explicitly requires it (e.g. by combining the TPPs in a manual fashion via an explicit temporary O𝑂O buffer/tensor which lives in the user space/application). This flexibility allows the TPP implementation (which is platform-specific) to combine TPPs in the most efficient way for the target architecture (e.g. the O𝑂O tensor can live at the physical register file in the composite TPP in order to avoid redundant memory movement).

6) *The TPP input/output tensors as well as the computation itself are precision aware*. This feature makes mixed precision computations (that are prominent in DL workloads) easy to express from the user point of view, and provides information to the TPP back-end that may enable efficient implementation.

### 2.2. TPP Arguments

As mentioned in the previous subsection, the input to TPPs are 2D tensors. Each 2D tensor can be specified by the number of rows M𝑀M, columns N𝑁N, its leading dimension l​d𝑙𝑑ld and its datatype d​t​y​p​e𝑑𝑡𝑦𝑝𝑒dtype. Additionally, during runtime each tensor gets fully characterized by specifying its location/address as *primary* info, optional companion tensor info as *secondary* (e.g. sparsity bitmask), and optionally *tertiary* info (e.g. in case the tensor shape is dynamically determined at runtime, this info may contain variables specifying M𝑀M/N𝑁N). Each TPP also specifies the shape/precision of the produced/output 2D tensor.

Each TPP also supports input tensors with *broadcast* semantics. More specifically, TPPs accept optional flags dictating that the input 2D tensor should be formed by broadcasting a column/row/scalar N𝑁N/M𝑀M/M×N𝑀𝑁M\times N times respectively. Finally, the TPPs accept optional flags which further specify the TPP operation. For example, in case a TPP is computing a transcendental function, the flags may be specifying various approximation algorithms used for the computation. In the next subsection we present the TPPs in three groups: *unary*, *binary*, and *ternary* TPPs given the number of input tensors they accept.

### 2.3. The TPP collection

| Unary TPP | Description/Comments |
| --- | --- |
| Identity | Copies input to output. Given input/output datatype, it performs datatype conversions |
| Zero | Fills output with zeros |
| Square | Squares input and stores to output |
| Increment / decrement | Increments / Decrements input by 1 and stores to output |
| Square root | Computes the square root of input and stores to output |
| Reciprocal | Computes the reciprocal of input and stores to output |
| Rcp. Sqrt. | Computes the rcp. sqrt. of input and stores to output |
| Exp | Computes the exponential value of the input tensor entries and stores them to output |
| PRNG | Generates an output tensor with pseudo-random entries |
| (De)Quantize | Quantizes / Dequantizes the input |
| Reduce | Reduces the rows/columns of the input and stores to output. The reduction function can be SUM/MUL/MIN/MAX; (optionally) reduces the *squared* input |
| Transform | Transforms input and stores to output. Transformations are: Transpose, VNNI formatting, and VNNI to VNNI-transpose |
| Unpack | Takes each entry xi,jsubscript𝑥  𝑖𝑗x\_{i,j} of the input tensor, splits it in two parts xi,jl​osuperscriptsubscript𝑥  𝑖𝑗𝑙𝑜x\_{i,j}^{lo} and xi,jh​isuperscriptsubscript𝑥  𝑖𝑗ℎ𝑖x\_{i,j}^{hi} with same bit-width, and stores them in two tensors Xl​osuperscript𝑋𝑙𝑜X^{lo}, Xh​isuperscript𝑋ℎ𝑖X^{hi} |
| Replicate columns | Takes an input column/vector, replicates it a variable number of times and forms the output |
| Gather / Scatter | Gathers/Scatters rows/columns from input and forms the tensor |
| 2D Gather / 2D Scatter | Gathers/scatters elements from input using 2D offsets |
| 2D-strided loads / stores | Loads/stores elements from/to a tensor using primary and secondary strides |
| Tanh &Tanh\_inv | Computes the hyperbolic tangent function (or its inv used for back-propagation) on input |
| RELU & RELU\_inv | Apply a Rectified Linear Unit function (or its inv used for back-propagation) on input |
| Sigmoid & Sigmoid\_inv | Computes the logistic sigmoid (or its inv used for back-propagation) on input |
| GELU & GELU\_inv | Apply a Gaussian Error Linear Unit function (or its inv used for back-propagation) on input |
| Dropout & Dropout\_inv | Drops out values from the input tensor with probability p𝑝p. For the inv/back-propagation pass, the same dropped units are zeroed out |

Table 1. Unary TPPs



| Binary TPP | Description/Comments |
| --- | --- |
| Add | Add two inputs |
| Sub | Subtracts two inputs |
| Mul | Multiples (elementwise) two inputs |
| Div | Divides two inputs |
| Max/Min | Finds element-wise max/min of two inputs |
| MatMul | Performs matrix multiplication of two input |
| Pack | Concatenates pairs of entries xi,jl​osuperscriptsubscript𝑥  𝑖𝑗𝑙𝑜x\_{i,j}^{lo} and xi,jh​isuperscriptsubscript𝑥  𝑖𝑗ℎ𝑖x\_{i,j}^{hi} from the inputs Xl​osuperscript𝑋𝑙𝑜X^{lo}, Xh​isuperscript𝑋ℎ𝑖X^{hi} into xi,jsubscript𝑥  𝑖𝑗x\_{i,j} and stores it to the output X𝑋X |
| Compare | Compares element-wise two inputs and stores a bitmask of the comparison |

Table 2. Binary TPPs



| Ternary TPP | Description/Comments |
| --- | --- |
| GEMM | Performs on 2D inputs A𝐴A, B𝐵B, C𝐶C, scalar β𝛽\beta: C=β​C+A×B𝐶𝛽𝐶𝐴𝐵C=\beta C+A\times B |
| Batch-Reduce GEMM | Performs on 2D inputs Aisubscript𝐴𝑖A\_{i}, Bisubscript𝐵𝑖B\_{i} (with i=0,1,𝑖  01i=0,1,…,n−1,n-1), C𝐶C, scalar β𝛽\beta: C=β​C+∑i=0i=n−1Ai×Bi𝐶𝛽𝐶superscriptsubscript𝑖0𝑖𝑛1subscript𝐴𝑖subscript𝐵𝑖C=\beta C+\sum\_{i=0}^{i=n-1}A\_{i}\times B\_{i} |
| (N)MulAdd | Performs on 2D inputs A𝐴A, B𝐵B, C𝐶C: C=C+A⊙B𝐶𝐶direct-product𝐴𝐵C=C+A\odot B (or C=C−A⊙B𝐶𝐶direct-product𝐴𝐵C=C-A\odot B ); ⊙direct-product\odot denotes element-wise multiplication |
| Blend | Blends 2D input tensors A𝐴A, B𝐵B according to bitmask C𝐶C |

Table 3. Ternary TPPs

First, we highlight the ternary *Batch-Reduce GEMM* (BRGEMM) TPP which is the main building block for general tensor contractions in DL kernels ([georganas2020harnessing,](#bib.bib21) ). BRGEMM materializes the operation C=β⋅C+∑i=0n−1Ai×Bi𝐶⋅𝛽𝐶superscriptsubscript𝑖0𝑛1subscript𝐴𝑖subscript𝐵𝑖C=\beta\cdot C+\sum\_{i=0}^{n-1}A\_{i}\times B\_{i}.
In essence, this kernel multiplies the specified blocks AiM×Ksuperscriptsubscript𝐴𝑖𝑀𝐾A\_{i}^{M\times K} and BiK×Nsuperscriptsubscript𝐵𝑖𝐾𝑁B\_{i}^{K\times N} and reduces the partial results to a block CM×Nsuperscript𝐶𝑀𝑁C^{M\times N}. It is noteworthy that tensors A𝐴A and B𝐵B can alias and also the blocks Aisubscript𝐴𝑖A\_{i} and Bisubscript𝐵𝑖B\_{i} can reside in any position in the input (potentially high-dimensional) tensors A𝐴A and B𝐵B. Previous work ([georganas2020harnessing,](#bib.bib21) ) has shown that this single building block is sufficient to express efficiently tensor contractions in the most prevalent DL computational motifs, namely: Convolution Neural Networks (CNN), Fully-Connected networks (FC), Multi-Layer Perceptrons (MLP), Recurrent Neural Networks (RNN)/Long Short-Term Memory (LSTM) Networks. In Section [5](#S5 "5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") we exhibit how BRGEMM can be further used to build efficient Attention Cells that comprise the cornerstone of modern Natural Language Processing (NLP) workloads. BRGEMM can be specialized to one of the following three variants that may enable more efficient implementations on various platforms: (i) *address-based BRGEMM*, where the addresses of the blocks Aisubscript𝐴𝑖A\_{i} and Bisubscript𝐵𝑖B\_{i} are explicitly provided by the user, (ii) *offset-based BRGEMM*, where the addresses of Aisubscript𝐴𝑖A\_{i} and Bisubscript𝐵𝑖B\_{i} can be computed as a​d​d​r​e​s​s​\_​Ai=a​d​d​r​e​s​s​\_​A+o​ff​s​e​tAi𝑎𝑑𝑑𝑟𝑒𝑠𝑠\_subscript𝐴𝑖𝑎𝑑𝑑𝑟𝑒𝑠𝑠\_𝐴𝑜ff𝑠𝑒subscript𝑡subscript𝐴𝑖address\\_A\_{i}=address\\_A+o\textit{ff}set\_{A\_{i}} and a​d​d​r​e​s​s​\_​Bi=a​d​d​r​e​s​s​\_​B+o​ff​s​e​tBi𝑎𝑑𝑑𝑟𝑒𝑠𝑠\_subscript𝐵𝑖𝑎𝑑𝑑𝑟𝑒𝑠𝑠\_𝐵𝑜ff𝑠𝑒subscript𝑡subscript𝐵𝑖address\\_B\_{i}=address\\_B+o\textit{ff}set\_{B\_{i}}, and (iii) *stride-based BRGEMM*, where the addresses of Aisubscript𝐴𝑖A\_{i} and Bisubscript𝐵𝑖B\_{i} are: a​d​d​r​e​s​s​\_​Ai=a​d​d​r​e​s​s​\_​Ai−1+s​t​r​i​d​e​\_​A𝑎𝑑𝑑𝑟𝑒𝑠𝑠\_subscript𝐴𝑖𝑎𝑑𝑑𝑟𝑒𝑠𝑠\_subscript𝐴𝑖1𝑠𝑡𝑟𝑖𝑑𝑒\_𝐴address\\_A\_{i}=address\\_A\_{i-1}+stride\\_A and a​d​d​r​e​s​s​\_​Bi=a​d​d​r​e​s​s​\_​Bi−1+s​t​r​i​d​e​\_​B𝑎𝑑𝑑𝑟𝑒𝑠𝑠\_subscript𝐵𝑖𝑎𝑑𝑑𝑟𝑒𝑠𝑠\_subscript𝐵𝑖1𝑠𝑡𝑟𝑖𝑑𝑒\_𝐵address\\_B\_{i}=address\\_B\_{i-1}+stride\\_B. In subsection [3.2](#S3.SS2 "3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") we present the implementation of the BRGEMM TPP in more depth for various ISAs and platforms.

Table [1](#S2.T1 "Table 1 ‣ 2.3. The TPP collection ‣ 2. The TPP Specification ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") presents the unary TPPs that accept one 2D tensor as input. Since most of these TPPs map directly to the equivalent math function, we further elaborate only on the ones which are more complex. The *Identity* TPP essentially copies the input to the output. Since the input and output are fully specified in terms of their precision, this TPP can be also used to perform datatype conversions between tensors.

The *Quantize & Dequantize* TPPs are used to quantize/dequantize the input tensor whereas the exact algorithm employed is specified by a TPP flag.

The *Transform* TPP uses a flag to determine the exact transformation applied on the input 2D tensor. The *Transpose* transformation is the usual mathematical matrix transpose. The rest two types of transformation, namely *VNNI formatting*, and *VNNI to VNNI-transpose* are DL specific. More specifically, modern hardware (e.g. Intel’s Cooper Lake) requires tensors to be in specific format called *VNNI* in order to employ hardware acceleration for specific operations, e.g. dot-products (see section [3.2.2](#S3.SS2.SSS2 "3.2.2. Mixed Precision BRGEMM and its emulation ‣ 3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") for more details). This format represents a logical 2D tensor [D1]​[D0]delimited-[]subscript𝐷1delimited-[]subscript𝐷0[D\_{1}][D\_{0}] as a 3D tensor [D1/α]​[D0]​[α]delimited-[]subscript𝐷1𝛼delimited-[]subscript𝐷0delimited-[]𝛼[D\_{1}/\alpha][D\_{0}][\alpha] where essentially the dimension D1subscript𝐷1D\_{1} is blocked in chunks of size α𝛼\alpha, which in turn are set as the inner-most tensor dimension. The *VNNI formatting* TPP performs this exact transformation: [D1]​[D0]→[D1/α]​[D0]​[α]→delimited-[]subscript𝐷1delimited-[]subscript𝐷0delimited-[]subscript𝐷1𝛼delimited-[]subscript𝐷0delimited-[]𝛼[D\_{1}][D\_{0}]\rightarrow[D\_{1}/\alpha][D\_{0}][\alpha] and the *VNNI to VNNI-transpose* transposes a tensor which is already laid out in VNNI format, i.e. performs [D1/α1]​[D0]​[α1]→[D0/α0]​[D1]​[α0]→delimited-[]subscript𝐷1subscript𝛼1delimited-[]subscript𝐷0delimited-[]subscript𝛼1delimited-[]subscript𝐷0subscript𝛼0delimited-[]subscript𝐷1delimited-[]subscript𝛼0[D\_{1}/\alpha\_{1}][D\_{0}][\alpha\_{1}]\rightarrow[D\_{0}/\alpha\_{0}][D\_{1}][\alpha\_{0}]. In subsection [3.3.1](#S3.SS3.SSS1 "3.3.1. Transform-Transpose TPP via Shuffle Networks ‣ 3.3. Examples of Non-Trivial Non-GEMM TPPs ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") we outline how the Transform TPPs are implemented via Shuffle Networks.

The last four entries of Table [1](#S2.T1 "Table 1 ‣ 2.3. The TPP collection ‣ 2. The TPP Specification ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") correspond to DL-specific operations. They correspond to activation functions typically encountered in DL workloads. All these activation functions have a counterpart which is required during the back-propagation pass of training DL networks. These DL specific TPPs could be built on top of other TPPs, however since they are prevalent in DL workloads we opt to define them as self-contained TPPs for ease of usage. In subsection [3.3.2](#S3.SS3.SSS2 "3.3.2. Approximations for non-linear TPP Activation Functions ‣ 3.3. Examples of Non-Trivial Non-GEMM TPPs ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") we describe the TPP implementation of non-linear approximations for several activation functions on various ISAs.

Table [2](#S2.T2 "Table 2 ‣ 2.3. The TPP collection ‣ 2. The TPP Specification ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") and Table [3](#S2.T3 "Table 3 ‣ 2.3. The TPP collection ‣ 2. The TPP Specification ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") present the binary/ternary TPPs that accept two/three 2D tensor as inputs respectively.

## 3. TPP Implementation

In this Section we briefly describe our Proof-Of-Concept (POC) implementation of the TPP specification. Our implementation targets multiple CPU architectures from various vendors that support different ISAs, but could be readily extended to support even GPU ISAs. We build upon and extend the open source LIBXSMM ([libxsmm,](#bib.bib22) ) library which leverages JIT techniques. Such JIT techniques have been successfully used for optimal code generation on CPUs by taking advantage of the known (at runtime) tensor shapes/dimensions in HPC and DL applications ([libxsmm,](#bib.bib22) ; [sc18,](#bib.bib23) ; [georganas2020harnessing,](#bib.bib21) ). Nevertheless, the TPP specification is platform-agnostic and does not preclude any TPP back-end implementation. In our POC implementation, the usage of TPPs is governed by two APIs: i) A dispatch API with which the user can request the code generation of a specific TPP, and such a dispatch call JITs a function implementing the requested operation, ii) an API to call the JITed TPP kernel. First, in Subsection [3.1](#S3.SS1 "3.1. Generic TPP Implementation Blueprint ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") we provide a generic blueprint of our TPP implementation. Then, in subsection [3.2](#S3.SS2 "3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") we describe in more detail the BRGEMM TPP implementation which comprises the main tensor contraction tool in the TPP abstractions. Subsection [3.3.1](#S3.SS3.SSS1 "3.3.1. Transform-Transpose TPP via Shuffle Networks ‣ 3.3. Examples of Non-Trivial Non-GEMM TPPs ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") details the implementation of the unary transform TPPs via shuffle networks since their efficient implementation diverts from the generic TPP blueprint. Finally, subsection [3.3.2](#S3.SS3.SSS2 "3.3.2. Approximations for non-linear TPP Activation Functions ‣ 3.3. Examples of Non-Trivial Non-GEMM TPPs ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") outlines the approximation techniques we leverage in our TPP implementation of non-linear activation functions; such approximations are essential in achieving high-performance, while at the same time their accuracy is sufficient for the purposes of training DL workloads.

### 3.1. Generic TPP Implementation Blueprint

1:Inputs: XM×NsuperscriptX𝑀𝑁\textbf{X}^{M\times N}, (YM×NsuperscriptY𝑀𝑁\textbf{Y}^{M\times N}, ZM×NsuperscriptZ𝑀𝑁\textbf{Z}^{M\times N} if binary/ternary)

2:Output: OM×NsuperscriptO𝑀𝑁\textbf{O}^{M\times N}

3:for in=0​…​N−1​with step ​𝐧𝐛subscript𝑖𝑛0…𝑁1with step subscript𝐧𝐛i\_{n}=0\dots N-1\ \textbf{with\ step\ }\mathbf{n\_{b}} do

4:   for im=0​…​M−1​with step ​𝐦𝐛subscript𝑖𝑚0…𝑀1with step subscript𝐦𝐛i\_{m}=0\dots M-1\ \textbf{with\ step\ }\mathbf{m\_{b}} do

5:      *▷▷\triangleright Generic loads, may have broadcast/gather semantics,*

6:      *▷▷\triangleright and may perform datatype conversions*

7:      Xbsubscript𝑋𝑏X\_{b} ←←\leftarrow load\_generic mb×nbsubscript𝑚𝑏subscript𝑛𝑏m\_{b}\times n\_{b} X𝑋X-subblockim,insubscriptsubblock

subscript𝑖𝑚subscript𝑖𝑛\text{subblock}\_{i\_{m},i\_{n}}

8:      if (unary TPP) then

9:         Xbsubscript𝑋𝑏X\_{b} ←←\leftarrow Unary\_op(Xb)subscript𝑋𝑏(X\_{b})

10:      if (binary TPP) then

11:         Ybsubscript𝑌𝑏Y\_{b} ←←\leftarrow load\_generic mb×nbsubscript𝑚𝑏subscript𝑛𝑏m\_{b}\times n\_{b} Y𝑌Y-subblockim,insubscriptsubblock

subscript𝑖𝑚subscript𝑖𝑛\text{subblock}\_{i\_{m},i\_{n}}

12:         Xbsubscript𝑋𝑏X\_{b} ←←\leftarrow Binary\_op(Xb,Yb)subscript𝑋𝑏subscript𝑌𝑏(X\_{b},Y\_{b})

13:      if (ternary TPP) then

14:         Ybsubscript𝑌𝑏Y\_{b} ←←\leftarrow load\_generic mb×nbsubscript𝑚𝑏subscript𝑛𝑏m\_{b}\times n\_{b} Y𝑌Y-subblockim,insubscriptsubblock

subscript𝑖𝑚subscript𝑖𝑛\text{subblock}\_{i\_{m},i\_{n}}

15:         Zbsubscript𝑍𝑏Z\_{b} ←←\leftarrow load\_generic mb×nbsubscript𝑚𝑏subscript𝑛𝑏m\_{b}\times n\_{b} Z𝑍Z-subblockim,insubscriptsubblock

subscript𝑖𝑚subscript𝑖𝑛\text{subblock}\_{i\_{m},i\_{n}}

16:         Xbsubscript𝑋𝑏X\_{b} ←←\leftarrow Ternary\_op(Xb,Yb,Zb)subscript𝑋𝑏subscript𝑌𝑏subscript𝑍𝑏(X\_{b},Y\_{b},Z\_{b})

17:      *▷▷\triangleright Generic store, may have scatter semantics, and may*

18:      *▷▷\triangleright perform datatype conversion*

19:      O𝑂O-subblockim,instore\_generic←

subscriptsubblock

subscript𝑖𝑚subscript𝑖𝑛←store\_generic\text{subblock}\_{i\_{m},i\_{n}}\ \ \overleftarrow{\text{store\\_generic}}\ \  Xbsubscript𝑋𝑏X\_{b}

Algorithm 1  The generic unary/binary/ternary TPP algorithm

Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1. Generic TPP Implementation Blueprint ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") exhibits at a high-level the pseudocode that is used to implement the Unary/Binary/Ternary TPPs in a unified fashion. The inputs of the TPPs are tensors X𝑋X, Y𝑌Y (in case of binary/ternary TPPs) and Z𝑍Z (in case of ternary TPP), and an output tensor O𝑂O. For the purposes of this simplified presentation we assume all tensors are of size M×N𝑀𝑁M\times N, however, depending on the operation these might have different sizes. For example, if the unary OP is a reduction-by-columns and the input is M×N𝑀𝑁M\times N, then the output is an M×1𝑀1M\times 1 vector. First, we show that the M𝑀M/N𝑁N loops are blocked with factors mbsubscript𝑚𝑏m\_{b}/nbsubscript𝑛𝑏n\_{b} such that the working sets of each microkernels fits on the available register file. The latter is architecture specific, e.g. AVX2-enabled ISAs expose 16 256-bit vector registers, AVX512-enabled ISAs expose 32 512-bit vector registers and Aarch64 features 32 128-bit (NEON)/512-bit (SVE) vector registers. The “load\_generic” function used in Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1. Generic TPP Implementation Blueprint ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") denotes the loading of a sub-tensor to a register block; this load may imply row/column/scalar broadcast semantics if the user specified the TPP in that way, or even may have strided-load/gather semantics if the TPP involves a strided-load/gather operation. Also, for simplicity we do not show here the handling of “secondary” fields of the tensors that may be required (e.g. indices array for the gather operation, bitmasks arrays). Additionally, the generic load also handles datatype conversion, for instance provided the input is in bfloat16 (BF16) ([bfloat16\_tf,](#bib.bib24) ) whereas the compute is going to happen in FP32 precision. Once all the required sub-tensors are loaded, then the corresponding Unary/Binary/Ternary operator is applied. This operator may be directly mapped to an available instruction (e.g. a vector add in case of binary addition), or to a sequence of instructions for more complicated operators (e.g. reductions, random number generation via xorshift algorithm ([marsaglia2003xorshift,](#bib.bib25) ), approximation algorithms for transcendental functions ([banerjee2019optimizing,](#bib.bib26) )). Last but not least, the optimal sequence generation depends on the available instructions and this is handled by the TPP back-end/JITer. For example, some ISAs may have masking/predicate support (e.g. AVX512 & SVE) that enable efficient handling of loop remainders, the selected unrolling degree heavily depends on the instructions in use, their latency and the number of available architectural registers. Once the result is computed, the resulting register block is stored back to the corresponding output sub-tensor position. Similarly to the generic load, the “generic” store may induce strided accesses or may be even a scatter operation. Additionally, the generic store also handles potential datatype conversions.

### 3.2. The BRGEMM TPP Implementation

#### 3.2.1. The BRGEMM kernel structure

1:Inputs: AiM×K,BiK×N​for​i=0,…,n​-​1formulae-sequence

superscriptsubscript𝐴𝑖𝑀𝐾superscriptsubscript𝐵𝑖𝐾𝑁for𝑖
0

…𝑛-1A\_{i}^{M\times K},B\_{i}^{K\times N}\ \text{for}\ i=0,...,n\text{-}1, CM×Nsuperscript𝐶𝑀𝑁C^{M\times N}, β∈I​R𝛽IR\beta\in{\rm I\!R}

2:Output:C=β⋅C+∑i=0n−1Ai×Bi𝐶⋅𝛽𝐶superscriptsubscript𝑖0𝑛1subscript𝐴𝑖subscript𝐵𝑖\ C=\beta\cdot C+\sum\_{i=0}^{n-1}A\_{i}\times B\_{i}

3:for in=0​…​N−1​with step ​𝐧𝐛subscript𝑖𝑛0…𝑁1with step subscript𝐧𝐛i\_{n}=0\dots N-1\ \textbf{with\ step\ }\mathbf{n\_{b}} do

4:   for im=0​…​M−1​with step ​𝐦𝐛subscript𝑖𝑚0…𝑀1with step subscript𝐦𝐛i\_{m}=0\dots M-1\ \textbf{with\ step\ }\mathbf{m\_{b}} do

5:      acc\_regs ←←\leftarrow load\_generic mb×nbsubscript𝑚𝑏subscript𝑛𝑏m\_{b}\times n\_{b} C𝐶C-subblockim,insubscriptsubblock

subscript𝑖𝑚subscript𝑖𝑛\text{subblock}\_{i\_{m},i\_{n}}

6:      for i=0​…​n−1​with step ​𝟏𝑖0…𝑛1with step 1i=0\dots n-1\ \textbf{with\ step\ }\mathbf{1} do

7:         for ik=0​…​K−1​with step ​𝐤𝐛subscript𝑖𝑘0…𝐾1with step subscript𝐤𝐛i\_{k}=0\dots K-1\ \textbf{with\ step\ }\mathbf{k\_{b}} do

8:            *▷▷\triangleright Outer product GEMM microkernel*

9:            acc\_regs +=absent\mathrel{+}= Ai​sub-panelim,ik×Bi​sub-panelik,insubscript𝐴𝑖subscriptsub-panel

subscript𝑖𝑚subscript𝑖𝑘subscript𝐵𝑖subscriptsub-panel

subscript𝑖𝑘subscript𝑖𝑛A\_{i}\ \text{sub-panel}\_{i\_{m},i\_{k}}\times B\_{i}\ \text{sub-panel}\_{i\_{k},i\_{n}}

10:      C𝐶C-subblockim,in​store\_generic←subscriptsubblock

subscript𝑖𝑚subscript𝑖𝑛←store\_generic\text{subblock}\_{i\_{m},i\_{n}}\overleftarrow{\text{store\\_generic}}\  acc\_regs

Algorithm 2  The batch-reduce GEMM TPP

![Refer to caption](/html/2104.05755/assets/x2.png)


Figure 2. Outer product GEMM microkernels, *Left*: On a platform with 32 vector registers, *Middle*: On a platform with 16 vector registers, *Right*: On a platform with 8 2D registers (tiles).

We present in more detail the BRGEMM TPP because it comprises the tensor contraction tool in the TPP abstraction, and is ubiquitous in the DL kernels and workloads described in Section [5](#S5 "5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads"). Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2.1. The BRGEMM kernel structure ‣ 3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") exhibits the high-level algorithm implementing: C=β⋅C+∑i=0n−1Ai×Bi𝐶⋅𝛽𝐶superscriptsubscript𝑖0𝑛1subscript𝐴𝑖subscript𝐵𝑖\ C=\beta\cdot C+\sum\_{i=0}^{n-1}A\_{i}\times B\_{i} . Lines 1-2 block the computation of the result C𝐶C in mb×nbsubscript𝑚𝑏subscript𝑛𝑏m\_{b}\times n\_{b} tensor sub-blocks. Once such a subblock is loaded into the accumulation registers (line 3), we loop over all pairs Ai,Bi

subscript𝐴𝑖subscript𝐵𝑖A\_{i},\ B\_{i} (line 4) and we accumulate into the loaded registers the products of the corresponding mb×Ksubscript𝑚𝑏𝐾m\_{b}\times K subblocks of Aisubscript𝐴𝑖A\_{i} with the relevant K×nb𝐾subscript𝑛𝑏K\times n\_{b} subblocks of Bisubscript𝐵𝑖B\_{i} (lines 5-7). In order to calculate a partial product of an mb×kbsubscript𝑚𝑏subscript𝑘𝑏m\_{b}\times k\_{b} sub-panel of Aisubscript𝐴𝑖A\_{i} with a kb×nbsubscript𝑘𝑏subscript𝑛𝑏k\_{b}\times n\_{b} sub-panel of Bisubscript𝐵𝑖B\_{i}, we follow an outer product formulation. The loading of Aisubscript𝐴𝑖A\_{i} and Bisubscript𝐵𝑖B\_{i} sub-panels, and the outer-product formulation is heavily dependent on the target platform. We provide BRGEMM implementations for multiple x86 ISAs: SSE, AVX, AVX2, AVX512, including the recently introduced Intel AMX (Advanced Matrix Extensions) ISA ([intelisa,](#bib.bib27) ). Additionally, we have implemented the BRGEMM TPP for AArch64 and ARMv8 SVE ISAs. Depending on the targeted platform, the “register” can be either a typical vector register with varying width (e.g.128-512bit vector length), or in the case of AMX-enabled target the “register” is a 2D tile-register. Similarly, the outer-product formulation may employ the available Fused-Multiply-Add (FMA) instructions, or even 2D tile-multiplication instructions. In all these cases, the TPP implementation emits the appropriate load/store/prefetch/FMA instructions, and takes into account the available architectural registers/unrolling factors/instruction mix in order to achieve close to peak performance. Last but not least, the BRGEMM supports multiple datatypes (FP64, FP32, BF16, INT8), and whenever possible employs hardware acceleration, e.g. via specialized FMA instructions for INT8/BF16 datatypes. In order to highlight the differences of the outer product GEMM microkernels that are heavily dependent on the target platform, we show in Figure [2](#S3.F2 "Figure 2 ‣ 3.2.1. The BRGEMM kernel structure ‣ 3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") three different implementations.

Figure [2](#S3.F2 "Figure 2 ‣ 3.2.1. The BRGEMM kernel structure ‣ 3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Left shows an exemplary outer product microkernel on a platform with 32 available vector registers, for example an x86 with AVX512 or on ARM AArch64/SVE. In this case vector register v7-v30 constitute the accumulators, vector registers v1-v6 hold a broadcasted subrow of B, and vector register v0 is used to load a partial subcolumn of A. First, we load on v1-v6 a subrow of B via broadcasts, then we load on v0 the first chunk of the A subcolumn and with 6 fused multiply-add (FMA) instructions (v0 with v1-v6) we multiply-and-add the corresponding partial results on the accumulators v7-v12 (first logical row of accumulators). Then, we load on v0 the second chunk of the A subcolumn, and subsequently with yet another 6 FMA instructions (v0 with v1-v6) we multiply-and-add the computed partial results on the accumulators v13-v18 (second logical row of accumulators) etc. The registers v1-v6 are reused 4 times throughout the outer product computation, and v0 is reused 6 times for each loaded A chunk. In other words, the corresponding A subcolumn and B subrow are loaded from memory/cache into the vector registers exactly once and we get to reuse them from the register file. Also, in such a formulation we expose 24 independent accumulation chains which is critical in order to hide the latency of the FMA instruction. Last but not least, the platform (i.e. vector register width) and the datatype of the microkernel determine the exact values of the blocking parameters mbsubscript𝑚𝑏m\_{b}, nbsubscript𝑛𝑏n\_{b}, and kbsubscript𝑘𝑏k\_{b}. For example for single precision datatype FP32 and an x86 AVX512 platform, each vector register can hold 16 FP32 values (the vector registers are 512-bit wide). Therefore, this microkernel operates with blocking values mb=64subscript𝑚𝑏64m\_{b}=64, nb=6subscript𝑛𝑏6n\_{b}=6, and kb=1subscript𝑘𝑏1k\_{b}=1 and it calculates a small matrix multiplication C64×6+=A64×1×B1×6C\_{64\times 6}\mathrel{+}=A\_{64\times 1}\times B\_{1\times 6}.

Figure [2](#S3.F2 "Figure 2 ‣ 3.2.1. The BRGEMM kernel structure ‣ 3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Middle shows an exemplary outer product microkernel on a platform with 16 vector registers, for example an x86 with up to AVX2 ISA. The microkernel is similar with the previous case; since we have only 16 vector registers available, we dedicate 12 of those as C𝐶C accumulators, 3 vector register are utilized for holding a partial B subrow, and 1 vector register is used to load a chunk of an A subcolumn. In this case 12 independent accumulation chains are also sufficient to hide the FMA latency. Analogously to the previous case, for single precision datatype FP32 and an x86 AVX2 platform, each vector register can hold now 8 FP32 values (the vector registers are now 256-bit wide). Thus, this microkernel operates with blocking values mb=32subscript𝑚𝑏32m\_{b}=32, nb=3subscript𝑛𝑏3n\_{b}=3, and kb=1subscript𝑘𝑏1k\_{b}=1 and it calculates a small matrix multiplication C32×3+=A32×1×B1×3C\_{32\times 3}\mathrel{+}=A\_{32\times 1}\times B\_{1\times 3}.

Figure [2](#S3.F2 "Figure 2 ‣ 3.2.1. The BRGEMM kernel structure ‣ 3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Right shows a small GEMM microkernel on a platform with 8 2D registers (tiles), for example what is available in the recently introduced Intel AMX (Advanced Matrix Extensions) ISA. In this case each 2D tile register has size (up to) 1KB, logically holds (up to) 16 rows of a submatrix, and can be loaded with a proper tile-load instruction. In this particular example, tiles 0-3 comprise the C𝐶C accumulators, tiles 4-5 are used to hold a subpanel of A and tiles 6-7 are used to hold a subpanel of B. Once we load the subpanels of A and B onto the respective tiles, we can perform 4 tile multiply-and-add instructions: tile0+=tile4×tile6tile0\mathrel{+}=tile4\times tile6, tile1+=tile4×tile7tile1\mathrel{+}=tile4\times tile7, tile2+=tile5×tile6tile2\mathrel{+}=tile5\times tile6 and tile3+=tile5×tile7tile3\mathrel{+}=tile5\times tile7, and we update the C𝐶C accumulators. In such a microkernel, each A/B tile is reused 2 times. Given each tile may have size up to 1KB and may hold up to 16 rows of a submatrix, by considering BF16 datatype for A/B matrices and FP32 accumulator tiles, such a microkernel operates with blocking values mb=32subscript𝑚𝑏32m\_{b}=32, nb=32subscript𝑛𝑏32n\_{b}=32, kb=32subscript𝑘𝑏32k\_{b}=32, and can compute (up to) a small matrix multiplication C32×32+=A32×32×B32×32C\_{32\times 32}\mathrel{+}=A\_{32\times 32}\times B\_{32\times 32}. Each A/B tile represents a logical 16×32163216\times 32 BF16 A/B submatrix, and each C tile represents a 16×16161616\times 16 FP32 accumulator. The AMX instructions will be available within the upcoming Intel Xeon processors code-named Sapphire Rapids, and the corresponding BF16-input/FP32-output tile multiplication instructions can deliver up to 16×16\times more FLOPs/cycle compared to FP32 AVX512 FMA instructions on current Xeon platforms.

These considerably different GEMM microkernel variants highlight yet another aspect of the TPPs: The TPPs specify *what* needs to be done rather than how it is done/implemented. In this case, the user may just specify/employ a BRGEMM TPP in order to perform a tensor contraction, whereas the TPP backend/implementation is responsible for generating the optimal code for each platform at hand. In this methodology, all the architectural nuances are hidden completely by the user, and the same exact user code written in terms of TPPs may be reused across platforms with different characteristic/ISAs without sacrificing performance or portability.

#### 3.2.2. Mixed Precision BRGEMM and its emulation

While the previous section presents the general structure of mapping matrix multiplication to various physical ISAs, this paragraph is used to demonstrate how the idea of a virtual ISA allows to implement operations efficiently which are not natively supported by a specific physical ISA. The example we are choosing here is our GEMM kernel and its support for bfloat16 and int8 on architectures which don’t support these novel ISA SIMD-extension.

![Refer to caption](/html/2104.05755/assets/x3.png)


Figure 3. Mixed-precision dot-product instructions, *Left*: 16 bit integer and bfloat16 on Intel AVX512, *Middle*: 8bit integer using Intel AVX512, *Right*: 8 bit integer using ARM ASIMD.

Before going into the details of the emulation, we first need to introduce special memory layouts which are used by x86 and aarch64 mixed-precision dot-product instructions as shown in Figure [3](#S3.F3 "Figure 3 ‣ 3.2.2. Mixed Precision BRGEMM and its emulation ‣ 3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads"). As we can see in all cases (x86/aarch64 and bf16/int8), the overall concept is identical: Although doing mixed-precision and mixed-datatype-length computations, these instructions are functioning from a matrix multiplication point-of-view similar to 32 bit instructions. This is achieved by having an implicit 2-wide (BF16/int16) and 4-wide (int8) dot-product of Aisubscript𝐴𝑖A\_{i} and Bisubscript𝐵𝑖B\_{i} values leading to a horizontal summation per single 32 bit Cisubscript𝐶𝑖C\_{i}, e.g. C0=A0⋅B0+A1⋅B1+A2⋅B2+A3⋅B3+C0subscript𝐶0⋅subscript𝐴0subscript𝐵0⋅subscript𝐴1subscript𝐵1⋅subscript𝐴2subscript𝐵2⋅subscript𝐴3subscript𝐵3subscript𝐶0C\_{0}=A\_{0}\cdot B\_{0}+A\_{1}\cdot B\_{1}+A\_{2}\cdot B\_{2}+A\_{3}\cdot B\_{3}+C\_{0} as shown for the int8 variant. If we apply blockings with these instructions as discussed in Figure [2](#S3.F2 "Figure 2 ‣ 3.2.1. The BRGEMM kernel structure ‣ 3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Left and Figure [2](#S3.F2 "Figure 2 ‣ 3.2.1. The BRGEMM kernel structure ‣ 3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Middle, then we realize that matrix B𝐵B is still read via 32-bit broadcast (containing 2 16-bit or 4 8-bit values along the inner-product or common dimension). However, matrix A𝐴A is in need of reformatting. This is due to the fact that the GEMM kernel in Figure [2](#S3.F2 "Figure 2 ‣ 3.2.1. The BRGEMM kernel structure ‣ 3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Left and Figure [2](#S3.F2 "Figure 2 ‣ 3.2.1. The BRGEMM kernel structure ‣ 3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Middle requires full SIMD-width contiguous loads for optimal performance (which is along M𝑀M and not K𝐾K). Therefore, we need to reformat A𝐴A into [Ko]​[M]​[Ki]delimited-[]superscript𝐾𝑜delimited-[]𝑀delimited-[]superscript𝐾𝑖[K^{o}][M][K^{i}] with Ko⋅Ki=K⋅superscript𝐾𝑜superscript𝐾𝑖𝐾K^{o}\cdot K^{i}=K and Ki=2superscript𝐾𝑖2K^{i}=2 for 16-bit and Ko=4superscript𝐾𝑜4K^{o}=4 for 8-bit inputs. We refer to such a format as *VNNI-format* throughout this paper. After such reformatting of A𝐴A, we can perform full SIMD loads on A𝐴A; combined with the 32-bit broadcast loads on B𝐵B we have a 32-bit GEMM kernel which has a shorter K𝐾K dimension, 2×\times for 16-bit datatypes and 4×\times for 8-bit datatypes.

![Refer to caption](/html/2104.05755/assets/x4.png)


Figure 4. Emulation of a bit accurate GEMM kernel using AVX512F instructions matching a GEMM kernel as depicted in Figure [2](#S3.F2 "Figure 2 ‣ 3.2.1. The BRGEMM kernel structure ‣ 3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") using vdpbf16ps AVX512 instructions. The glossary contains detailed descriptions of the used intrinsic functions.

In case these novel instructions are not available, especially for bfloat16 as this is a relatively new format, one might think, that an efficient mapping to a classic FP32 SIMD ISA is not possible. This is correct as long as the machine does not offer int16 support. However, with int16 support and SIMD masking we can implement the aforementioned non-trivial mixed-precision dot-product efficiently and even bit-accurately as shown in Figure [4](#S3.F4 "Figure 4 ‣ 3.2.2. Mixed Precision BRGEMM and its emulation ‣ 3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads"). This is done by processing Kisuperscript𝐾𝑖K^{i} in two rounds in the case of bfloat16 datatype. As shown in Figure [4](#S3.F4 "Figure 4 ‣ 3.2.2. Mixed Precision BRGEMM and its emulation ‣ 3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") we first process the odd (or upper) bfloat16 number. This is done by exploiting the fact that a bfoat16 number perfectly aliases with an FP32 number in its 16 MSBs. Therefore, on AVX512 we can just execute a full SIMD load as a 16-bit-typed load with masking. As a mask we chose 0xaaaaaaaa and as masking-mode we use zero masking. With this trick we automatically turn on-load the upper bfloat16 numbers in A𝐴A into 16 valid FP32 numbers, and for B𝐵B we broadcast and then perform an overriding register move. A little bit more work is needed for the lower/even bfloat16 number: In this case we perform an unmasked load and then we use a 32-bit integer shift by 16 to create valid FP32 numbers. A simple inspection of the instruction sequence in Figure [4](#S3.F4 "Figure 4 ‣ 3.2.2. Mixed Precision BRGEMM and its emulation ‣ 3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") shows that we are mainly executing fused-multiply-add instructions with little overhead compared to a classic FP32 GEMM as illustrated in Figure [2](#S3.F2 "Figure 2 ‣ 3.2.1. The BRGEMM kernel structure ‣ 3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Left and Figure [2](#S3.F2 "Figure 2 ‣ 3.2.1. The BRGEMM kernel structure ‣ 3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Middle. Therefore, we can execute a bfloat16 GEMM with a reformatted matrix A𝐴A with close to FP32-peak and still benefit from the smaller memory footprint (and therefore a small performance gain, as we will show later in section [6](#S6 "6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")). Replacement sequences for int16 and int8 matrix inputs can be carried out in a similar way and their detailed discussion is skipped here.

In addition to the presented emulation of mixed-precision GEMM kernels using SIMD instructions, we have also added support for emulation of Intel AMX instructions bit-accurately on AVX512. This addition enables running numerical accuracy experiments, such as convergence studies, before the release of a chip that supports Intel AMX instructions. A similar path is possible for ARM’s SME instruction set and subject to future work. These emulation capabilities further highlight the aspect of TPP as a virtual tensor ISA.

### 3.3. Examples of Non-Trivial Non-GEMM TPPs

The previous sections covered most of the TPP implementations: straightforward element-wise unary/binary/ternary operations and various forms of mixed precision GEMMs including their emulation on older hardware. However, there are cases in which we are not operating on the data in an element-wise fashion, e.g. transpose, or the Unary\_op, Binary\_op or Ternary\_op is not an elementary operation. The goal of this section is to shed some light on these cases by presenting the transpose TPP in detail, and sketching fast non-linear approximations on SIMD machines that match the accuracy requirements of deep learning applications.

#### 3.3.1. Transform-Transpose TPP via Shuffle Networks

![Refer to caption](/html/2104.05755/assets/x5.png)


Figure 5. Sketch of a shuffle network for a 32-bit transpose of a 16×\times16 matrix using Intel AVX512 instructions. Via 4 stages (each one having 16 independent shuffles that double in width per stage), the 16×\times16 matrix (256 elements) can be transposed with only 64 instructions and fully leverages the 32 architectural registers.

![Refer to caption](/html/2104.05755/assets/x6.png)


Figure 6. Comparison of X86 and ARM code for a simple 4×\times4 single precision transpose using unpack instructions. The glossary contains detailed descriptions of the used intrinsic functions.

When working with matrices, the transpose kernel is ubiquitous. It is needed to access the matrix’s elements in various contractions along the mathematically correct dimension. However, a transpose operation is scalar at first sight. In this subsection we exhibit how transpose can be implemented using shuffle networks in a fully vectorized fashion, e.g. Figure [5](#S3.F5 "Figure 5 ‣ 3.3.1. Transform-Transpose TPP via Shuffle Networks ‣ 3.3. Examples of Non-Trivial Non-GEMM TPPs ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") demonstrates how a 16×\times16 matrix with 256 32-bit elements can be transposed in 64 cycles using AVX512 instructions.

The shuffle-network presented in Figure [5](#S3.F5 "Figure 5 ‣ 3.3.1. Transform-Transpose TPP via Shuffle Networks ‣ 3.3. Examples of Non-Trivial Non-GEMM TPPs ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") is a blueprint for all datatype-lengths and ISAs: in log2⁡SIMD-Lengthsubscript2SIMD-Length\log\_{2}\text{\it{SIMD-Length}} stages we can transpose a matrix held in a set of SIMD registers. In this particular example, we need log2⁡16=4subscript2164\log\_{2}16=4 stages and in each stage we increase the shuffling/interleaving width of logical elements, and also increase the distance at which we access the 32 registers grouped into two sets of 16 registers each. More specifically, we start with registers i0subscript𝑖0i\_{0} to i15subscript𝑖15i\_{15} and interleave elements at the same position in a pair of registers close to each other. This constructs now pairs of 32 bit values in o0subscript𝑜0o\_{0} and o1subscript𝑜1o\_{1} which are already containing the transpose’s result for 2 out of 16 elements and we repeat this for all other 7 input register pairs. The analogous transformation is now repeated in the second stage with 64-bit values and accessing o0subscript𝑜0o\_{0} and o2subscript𝑜2o\_{2} as input pair pattern. This constructs a new set output registers i0subscript𝑖0i\_{0} and i1subscript𝑖1i\_{1} which are holding the transpose’s result at 128-bit granularity. After that, stage 3 is shuffling at 128-bit granularity on register pairs which have a distance of “4” and creates output registers that hold 256-bit of transposed data. Finally, in stage 4, these 256-bit transposed input registers are shuffled once again creating the final set of 16 register holding the transposed 16 ×\times 16 matrix. For non-square matrices we a) just use masked loads or set registers to zero, b) transpose the zeros as well, and then c) don’t store all registers or employ masked stores. This basic kernel is used as a basic building block to create large transpose operators by simply adding outer loops.

This algorithm can be implemented by any SIMD ISA which offers support for picking arbitrary values from a pair of SIMD registers to construct a result register containing values from the two sources, i.e. a general shuffler. However, “structured” shuffle instructions are adequate as shown in Figure [6](#S3.F6 "Figure 6 ‣ 3.3.1. Transform-Transpose TPP via Shuffle Networks ‣ 3.3. Examples of Non-Trivial Non-GEMM TPPs ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads"). Both x86 and aarch64 offer instructions exactly implementing the needs for 32-bit and 64-bit interleaves as needed in the first two stages covered in the previous description. In the case of 128-bit-wide SIMD registers this is enough to carry out the entire transpose of 4 ×\times 4 matrices as shown in Figure [6](#S3.F6 "Figure 6 ‣ 3.3.1. Transform-Transpose TPP via Shuffle Networks ‣ 3.3. Examples of Non-Trivial Non-GEMM TPPs ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads").

Finally, we want to note that broadcast loads, as supported by various ISAs, can be used to implement the first stage of the shuffle network. This has the advantage that one stage of the shuffle network can be executed faster and in parallel to the shuffler. The shuffle operations needed in all of these networks are relatively expensive in hardware, therefore modern CPUs often only provide one execution unit for such operations (such “shuffle-viruses” like transposes are pretty rare in general code). However, broadcasts on the load path are cheap and can run in parallel to the shuffle unit, hence the overall performance of the transpose operation improves. This microkernel variation leads to relatively complex code, and as such we skip its presentation. However our TPP implementation backend employs all these microkernel variations.

#### 3.3.2. Approximations for non-linear TPP Activation Functions

Activation functions are used to represent non-linear behavior of neural networks. Popular known activation functions are sigmoid, tanh and GELU. These activation functions can be approximated to increase the efficiency of deep learning networks without effecting it’s non-linear characteristics.
In this section we will discuss different approximation techniques based on Padé rational polynomials, piecewise minimax polynomials and Taylor expansions, along with their TPP implementation on different ISAs. For simplicity we present the relevant algorithms in terms of x86 and arm intrinsics (see glossary for the semantics of these intrinsics), however the actual TPP implementation relies on JIT code generation.

Rational Padé polynomials

![Refer to caption](/html/2104.05755/assets/x7.png)


Figure 7. Rational Padé 7/8 tanh approximation pseudocode with equivalent intrinsics on x86 and Arm/AArch64. We highlight here how the FMADD instruction on x86 ISAs has an equivalent instruction sequence on AArch64.

The Padé approximation of a function f𝑓f is the ratio of two polynomials with degrees p and q:

|  |  |  |
| --- | --- | --- |
|  | P​a​d​e´[p/q]​f​(x)=∑i=0pai​xi∑i=0qbi​xi𝑃𝑎𝑑subscript´𝑒delimited-[]𝑝𝑞𝑓𝑥superscriptsubscript𝑖0𝑝subscript𝑎𝑖superscript𝑥𝑖superscriptsubscript𝑖0𝑞subscript𝑏𝑖superscript𝑥𝑖Pad\acute{e}\_{[p/q]f}(x)=\frac{\sum\_{i=0}^{p}a\_{i}x^{i}}{\sum\_{i=0}^{q}b\_{i}x^{i}} |  |

The coefficients aisubscript𝑎𝑖a\_{i} and bisubscript𝑏𝑖b\_{i} can be calculated by considering the first p+q𝑝𝑞p+q derivatives of f𝑓f at zero and solving the corresponding system of equations:

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(0)=𝑓0absent\displaystyle f(0)= | P​a​d​e´[p/q]​f​(0)𝑃𝑎𝑑subscript´𝑒delimited-[]𝑝𝑞𝑓0\displaystyle Pad\acute{e}\_{[p/q]f}(0) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | f′​(0)=superscript𝑓′0absent\displaystyle f^{\prime}(0)= | P​a​d​e´[p/q]​f′​(0)𝑃𝑎𝑑subscriptsuperscript´𝑒′delimited-[]𝑝𝑞𝑓0\displaystyle Pad\acute{e}^{\prime}\_{[p/q]f}(0) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⋮⋮\displaystyle\vdots |  | |
|  |  |  |  |
| --- | --- | --- | --- |
|  | f(p+q)​(0)=superscript𝑓𝑝𝑞0absent\displaystyle f^{(p+q)}(0)= | P​a​d​e´[p/q]​f(p+q)​(0)𝑃𝑎𝑑subscriptsuperscript´𝑒𝑝𝑞delimited-[]𝑝𝑞𝑓0\displaystyle Pad\acute{e}^{(p+q)}\_{[p/q]f}(0) |  |

As an example we consider the approximation of the tanh function which has two asymptotes, hence approximating it with a Taylor expansion of lower degree polynomials may not yield good results. The implementation of the P​a​d​e´[7/8]​(x)𝑃𝑎𝑑subscript´𝑒delimited-[]78𝑥Pad\acute{e}\_{[7/8]}(x) tanh approximation is shown in Figure [7](#S3.F7 "Figure 7 ‣ 3.3.2. Approximations for non-linear TPP Activation Functions ‣ 3.3. Examples of Non-Trivial Non-GEMM TPPs ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads"). FMA operations are used to compute the numerators and denominators via Horner’s rule. The reciprocal of the denominator is multiplied by the numerator to get the final result. The accuracy of reciprocal instruction is different among different CPU’s. This difference in accuracy does not affect the non-linear region of the tanh function, keeping the TPP behavior same across different CPU’s. The sigmoid activation function can be approximated via tanh by leveraging the following identity:

|  |  |  |
| --- | --- | --- |
|  | s​i​g​m​o​i​d​(x)=(tanh⁡(x/2)+1)/2𝑠𝑖𝑔𝑚𝑜𝑖𝑑𝑥𝑥212sigmoid(x)=(\tanh(x/2)+1)/2 |  |

Piecewise minimax polynomial approximations

![Refer to caption](/html/2104.05755/assets/x8.png)


Figure 8. Tanh minimax approximation pseudocode with equivalent intrinsics on x86 and Arm/AArch64. We highlight here how the \_mm512\_range\_ps instruction on x86 ISAs has an equivalent instruction sequence on AArch64. Also the permutes on x86 have equivalent Table lookup instructions on AArch64.

In this section we discuss the minimax polynomials approach ([powell1981approximation,](#bib.bib28) ) with the truncated Chebyshev series ([chb,](#bib.bib29) ) for approximations of activation functions. In this approach, the input range of a function f​(x)𝑓𝑥f(x) is divided into intervals and for each interval [a,b]𝑎𝑏[a,b] we find a polynomial p𝑝p of degree max n𝑛n to minimize:

|  |  |  |
| --- | --- | --- |
|  | maxa≤x≤b⁡|f​(x)−p​(x)|subscript𝑎𝑥𝑏𝑓𝑥𝑝𝑥\max\_{a\leq x\leq b}|f(x)-p(x)| |  |

We approximate tanh and GELU activation functions using this approach in our TPP implementation. The input range is divided into 16 intervals and for each interval we investigate a polynomial p𝑝p of 3r​dsuperscript3𝑟𝑑3^{rd} degree (i.e. we find appropriate p𝑝p’s coefficients c0, c1, c2 based on the minimized absolute maximum difference of f𝑓f and p𝑝p). Figure [8](#S3.F8 "Figure 8 ‣ 3.3.2. Approximations for non-linear TPP Activation Functions ‣ 3.3. Examples of Non-Trivial Non-GEMM TPPs ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") shows the x86 and arm implementation of evaluating such minimax polynomials. The register index (idx) is calculated using the exponent and MSB of the respective input values, and represents the 16 intervals where the input values are located. The range intrinsic \_mm512\_range\_ps(A,B) is used to generate the register index (idx) on AVX512 platforms (Figure [8](#S3.F8 "Figure 8 ‣ 3.3.2. Approximations for non-linear TPP Activation Functions ‣ 3.3. Examples of Non-Trivial Non-GEMM TPPs ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Left, line 2). In ARM, the range functionality is emulated with equivalent and, shlq, min and max instructions as shown in Figure [8](#S3.F8 "Figure 8 ‣ 3.3.2. Approximations for non-linear TPP Activation Functions ‣ 3.3. Examples of Non-Trivial Non-GEMM TPPs ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Right, lines 2-4. To evaluate the 3r​dsuperscript3𝑟𝑑3^{rd} degree polynomial we need to locate 3 coefficients (c0,c1,c2) based on the values at the register index (idx), which holds 16 entries. We use 3 look up operations to find the three coefficients, each involving 16 FP32 entries. The 512-bit register length in AVX512 is sufficient to hold 16 coefficients required for each look up, resulting in using 3 registers for 3 look up operations (see Figure [8](#S3.F8 "Figure 8 ‣ 3.3.2. Approximations for non-linear TPP Activation Functions ‣ 3.3. Examples of Non-Trivial Non-GEMM TPPs ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Left, lines 4-6). Each ARM 128-bit wide vector register can only hold 4 FP32 entries, subsequently we are using 12 vector registers to hold the 16 entries for all 3 coefficients of the polynomial. The in-register look-up table is performed using \_mm512\_permutexvar\_ps(A,B) instructions in x86 AVX512 as shown in Figure [10](#S3.F10 "Figure 10 ‣ 3.3.2. Approximations for non-linear TPP Activation Functions ‣ 3.3. Examples of Non-Trivial Non-GEMM TPPs ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads"). In ARM we have byte addressable table look up instructions which are analogous to 32-bit addressable permutes instructions in x86. Hence, we need to convert the 32-bit addressable (0-16) register indexes to byte addressable (0-64 bytes) indexes. In order to do that, we use a constant register A with a table look up instruction to duplicate the register index (idx) to each byte in the 32-bit entry. A constant offset (0,1,2,3) is added to the duplicated byte index to get the byte addressable index for each FP32 entry in 16 FP32 entries (Figure [8](#S3.F8 "Figure 8 ‣ 3.3.2. Approximations for non-linear TPP Activation Functions ‣ 3.3. Examples of Non-Trivial Non-GEMM TPPs ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Right, lines 7-9). The table look up instruction in ARM provides the 64 byte look up capability, which is sufficient enough to search into 4 registers holding the 16 entries of each coefficient; we are using the generated byte indexes as shown in Figure [9](#S3.F9 "Figure 9 ‣ 3.3.2. Approximations for non-linear TPP Activation Functions ‣ 3.3. Examples of Non-Trivial Non-GEMM TPPs ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads"). Finally, 4 FMA operations are used to evaluate the polynomial using Horner’s rule. The FMA instruction in x86 provides the user the flexibility to decide among the sources to destroy and the ones to preserve. ARM requires mov instructions to save intermediate results in order to avoid the data overwriting during FMA operations.

![Refer to caption](/html/2104.05755/assets/x9.png)


Figure 9. Byte addressable Table look up setup in ARM/AArch64. We highlight the conversion of 32bit indexes to byte indexes and the use of byte indexes to get the coefficients in 16 FP32 intervals.

![Refer to caption](/html/2104.05755/assets/x10.png)


Figure 10. 32Bit Addressable Table look up setup on x86 AVX512 platforms.

Approximation with Taylor series

![Refer to caption](/html/2104.05755/assets/x11.png)


Figure 11. Pseudocode for exsuperscript𝑒𝑥e^{x} approximation with Taylor series on AVX512 x86 and ARM.

As an example of approximation with Taylor series we illustrate here the exp() activation function. The exsuperscript𝑒𝑥e^{x} is approximated using the identity ex=2x​log2⁡e=2n+y=2n⋅2ysuperscript𝑒𝑥superscript2𝑥subscript2𝑒superscript2𝑛𝑦⋅superscript2𝑛superscript2𝑦e^{x}=2^{x\log\_{2}e}=2^{n+y}=2^{n}\cdot 2^{y} with n=r​o​u​n​d​(x​log2⁡e)𝑛𝑟𝑜𝑢𝑛𝑑𝑥subscript2𝑒n=round(x\log\_{2}e) and y=x​log2⁡e−n𝑦𝑥subscript2𝑒𝑛y=x\log\_{2}e-n. We need to calculate 2nsuperscript2𝑛2^{n} with n𝑛n being an integer and the term 2ysuperscript2𝑦2^{y} with |y|∈[0,1)𝑦01|y|\in[0,1). A Taylor polynomial of third degree is used to calculate the term 2ysuperscript2𝑦2^{y} with 3 FMA instructions (see Figure [11](#S3.F11 "Figure 11 ‣ 3.3.2. Approximations for non-linear TPP Activation Functions ‣ 3.3. Examples of Non-Trivial Non-GEMM TPPs ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Left, lines 4-6). Once 2ysuperscript2𝑦2^{y} is calculated, we leverage the instruction \_mm512\_scalef\_ps(A,B) which returns a vector register holding ai⋅2f​l​o​o​r​(bi)⋅subscript𝑎𝑖superscript2𝑓𝑙𝑜𝑜𝑟subscript𝑏𝑖a\_{i}\cdot 2^{floor(b\_{i})} for each ai∈Asubscript𝑎𝑖𝐴a\_{i}\in A and bi∈Bsubscript𝑏𝑖𝐵b\_{i}\in B. This scale instruction concludes the exp() approximation on x86 with AVX512. On ARM we calculate 2nsuperscript2𝑛2^{n} and 2ysuperscript2𝑦2^{y} with equivalent replacement instructions as shown in Figure [11](#S3.F11 "Figure 11 ‣ 3.3.2. Approximations for non-linear TPP Activation Functions ‣ 3.3. Examples of Non-Trivial Non-GEMM TPPs ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads").

## 4. TPP Matrix Equations

One of the main design principles of TPPs (as described in Section [2.1](#S2.SS1 "2.1. TPP Design Principles ‣ 2. The TPP Specification ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")) is that they can be composed in a producer-consumer fashion to form complex operations. For example consider the scenario where a user wants to implement the composite operation C=T​a​n​h​(A+B)𝐶𝑇𝑎𝑛ℎ𝐴𝐵C=Tanh(A+B). One way to express this via TPPs would be to allocate an intermediate tensor t​m​p𝑡𝑚𝑝tmp with same shape as A𝐴A and B𝐵B, and perform first t​m​p=A​d​d​(A,B)𝑡𝑚𝑝𝐴𝑑𝑑𝐴𝐵tmp=Add(A,B) via the binary Add TPP. Then the user can compute the final result by leveraging the Tanh Unary TPP: C=T​a​n​h​(t​m​p)𝐶𝑇𝑎𝑛ℎ𝑡𝑚𝑝C=Tanh(tmp). Even though this approach is functionally correct, it requires the explicit management of intermediate tensors/buffers by the user and also may result in low performance since there are redundant loads/stores to the t​m​p𝑡𝑚𝑝tmp tensor.

In order to increase the productivity, efficiency and expressiveness pertaining to composite operators, we implemented an embedded Domain Specific Language (eDSL) in LIBXSMM ([libxsmm,](#bib.bib22) ). Our Proof-Of-Concept implementations allows the user to express the desired composite operator as a Matrix Equation. More specifically, the user can express the composite operator as an equation tree, where the head and internal nodes are the available TPPs, whereas the leaves of the tree are the input 2D tensors of the composite operation. In the next subsections we describe in detail the methodology we employ for JITing matrix equations of TPPs.

### 4.1. Definitions and notations for TPP Matrix Equations

![Refer to caption](/html/2104.05755/assets/x12.png)


Figure 12. Left: TPP Equation tree for O​u​t=T​a​n​h​(T0)+(T1×T2)/(T3−T4)𝑂𝑢𝑡𝑇𝑎𝑛ℎsubscript𝑇0subscript𝑇1subscript𝑇2subscript𝑇3subscript𝑇4Out=Tanh(T\_{0})+(T\_{1}\times T\_{2})/(T\_{3}-T\_{4}). Right: Assigned register scores v𝑣v on the equation TPP nodes after running Algorithm [3](#alg3 "Algorithm 3 ‣ 4.2. Optimized Execution plan for TPP Matrix Equations ‣ 4. TPP Matrix Equations ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads").

A TPP matrix equation is represented as a tree with unary/binary/ternary TPP operations as internal nodes and the equation’s input tensors are the leaves of the tree. The inputs of a TPP tree node are essentially its children in the equation tree. The output of an internal TPP node can be represented as a *temporary* intermediate tensor which in turn can be fed as input to the parent TPP node in the tree. Depending on the TPP node type (unary/binary/ternary), each internal node requires a number of inputs (one/two/three) to be computed/ready before performing the corresponding TPP operation. Let’s consider for example the TPP equation tree in Figure [12](#S4.F12 "Figure 12 ‣ 4.1. Definitions and notations for TPP Matrix Equations ‣ 4. TPP Matrix Equations ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Left that is used to express the following operator:

|  |  |  |  |
| --- | --- | --- | --- |
| (1) |  | O​u​t=T​a​n​h​(T0)+(T1×T2)/(T3−T4)𝑂𝑢𝑡𝑇𝑎𝑛ℎsubscript𝑇0subscript𝑇1subscript𝑇2subscript𝑇3subscript𝑇4Out=Tanh(T\_{0})+(T\_{1}\times T\_{2})/(T\_{3}-T\_{4}) |  |

We will illustrate with this example how our eDSL for TPP Matrix Equations works.

### 4.2. Optimized Execution plan for TPP Matrix Equations

The equation tree in Figure [12](#S4.F12 "Figure 12 ‣ 4.1. Definitions and notations for TPP Matrix Equations ‣ 4. TPP Matrix Equations ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Left can be naively evaluated by assigning to each intermediate node a temporary tensor to hold the corresponding TPP output, and performing e.g. 1) the Tanh operation, 2) the Matrix Multiplication, 3) the Subtract operation, 4) the Div operation , and finally 5) the Add TPP. In such an evaluation schedule we would need 4 intermediate tensors to hold the corresponding intermediate results. In this subsection we illustrate how we can construct optimized execution plans for TPP Matrix Equations that minimize the number of intermediate tensors.

For each TPP node r𝑟r we can assign a *register score* value vrsubscript𝑣𝑟v\_{r} that essentially dictates how many temporary/intermediate tensors are required to calculate the subtree in the equation where node r𝑟r is root. We extend the methodology of  ([flajolet1979number,](#bib.bib30) ) and we generate the register score values using the recursive Algorithm [3](#alg3 "Algorithm 3 ‣ 4.2. Optimized Execution plan for TPP Matrix Equations ‣ 4. TPP Matrix Equations ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads"). This algorithm calculates recursively the register scores of the children for a given node r𝑟r, and in this way we know how many temporary tensors are required for the evaluation for each child. Now, if all of its children have the same register score, the node r𝑟r get an increased register score value, otherwise the node gets as register score the maximum of its children’s register score values. Intuitively this means that we can first evaluate a child c𝑐c and its subtree with whatever intermediate tensor requirements it has, e.g. vcsubscript𝑣𝑐v\_{c} temporary tensors, and eventually we need only one temporary tensor to hold c𝑐c’s output. We can do the same afterwards for all other siblings of c𝑐c, however we can reuse/recycle the rest vc−1subscript𝑣𝑐1v\_{c}-1 temporary tensors that were required by c𝑐c since c𝑐c and its subtree have been already computed.

This algorithm optimizes the number of temporary tensors/storage that are required for the equation evaluation, and it reuses the temporary storage as much as possible. For instance, for the equation in Figure [12](#S4.F12 "Figure 12 ‣ 4.1. Definitions and notations for TPP Matrix Equations ‣ 4. TPP Matrix Equations ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Left, after executing Algorithm [3](#alg3 "Algorithm 3 ‣ 4.2. Optimized Execution plan for TPP Matrix Equations ‣ 4. TPP Matrix Equations ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") on the TPP equation tree, we see that the root’s register score value is 2 (see Figure [12](#S4.F12 "Figure 12 ‣ 4.1. Definitions and notations for TPP Matrix Equations ‣ 4. TPP Matrix Equations ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Right), meaning that only 2 intermediate tensors are required to evaluate the entire TPP tree rather than naively assigning one temporary tensor to each internal TPP node which would result in 4 intermediate tensors.

1:Input: TPP equation tree with root node r𝑟r

2:Output: TPP equation tree with assigned register score values on its nodes

3:if is\_Leaf( r𝑟r ) then

4:   vrsubscript𝑣𝑟v\_{r} ←←\leftarrow 0

5:if r𝑟r is unary TPP then

6:   Assign\_Register\_Score( Left\_Child( r𝑟r ) )

7:   *▷▷\triangleright If child is leaf, then we assign current register score of 1, else we assign the child’s register score*

8:   if  is\_Leaf( Left\_Child( r𝑟r ) ) then

9:      vrsubscript𝑣𝑟v\_{r} ←←\leftarrow 1

10:   else

11:      vrsubscript𝑣𝑟v\_{r} ←←\leftarrow Register\_Score(Left\_Child( r𝑟r ))

12:if r𝑟r is binary TPP then

13:   Assign\_Register\_Score( Left\_Child( r𝑟r ) )

14:   Assign\_Register\_Score( Right\_Child( r𝑟r ) )

15:   *▷▷\triangleright If the register scores of children are equal, then we get the children’s register score increased by one, otherwise we get the max value of the children’s register score*

16:   if  Register\_Score(Left\_Child( r𝑟r )) equals Register\_Score(Right\_Child( r𝑟r )) then

17:      vrsubscript𝑣𝑟v\_{r} ←←\leftarrow Register\_Score(Left\_Child( r𝑟r )) + 1

18:   else

19:      vLsubscript𝑣𝐿v\_{L} ←←\leftarrow Register\_Score(Left\_Child( r𝑟r ))

20:      vRsubscript𝑣𝑅v\_{R} ←←\leftarrow Register\_Score(Right\_Child( r𝑟r ))

21:      vrsubscript𝑣𝑟v\_{r} ←←\leftarrow MAX(vLsubscript𝑣𝐿v\_{L} , vRsubscript𝑣𝑅v\_{R})

22:if r𝑟r is ternary TPP then

23:   Assign\_Register\_Score( Left\_Child( r𝑟r ) )

24:   Assign\_Register\_Score( Middle\_Child( r𝑟r ) )

25:   Assign\_Register\_Score( Right\_Child( r𝑟r ) )

26:   *▷▷\triangleright If all children are leaves, then we assign current register score of 1, otherwise we get the max value of the children’s register score*

27:   if  is\_Leaf( Left\_Child( r𝑟r ) ) AND is\_Leaf( Middle\_Child( r𝑟r ) ) AND is\_Leaf( Right\_Child( r𝑟r ) ) then

28:      vrsubscript𝑣𝑟v\_{r} ←←\leftarrow 1

29:   else

30:      vLsubscript𝑣𝐿v\_{L} ←←\leftarrow Register\_Score(Left\_Child( r𝑟r ))

31:      vMsubscript𝑣𝑀v\_{M} ←←\leftarrow Register\_Score(Middle\_Child( r𝑟r ))

32:      vRsubscript𝑣𝑅v\_{R} ←←\leftarrow Register\_Score(Right\_Child( r𝑟r ))

33:      vrsubscript𝑣𝑟v\_{r} ←←\leftarrow MAX(3, vLsubscript𝑣𝐿v\_{L}, vMsubscript𝑣𝑀v\_{M}, vRsubscript𝑣𝑅v\_{R})

Algorithm 3  Assign\_Register\_Score( r𝑟r )

![Refer to caption](/html/2104.05755/assets/x13.png)


Figure 13. Left: TPP equation tree with assigned register scores v𝑣v on the nodes. Right: TPP equation tree with assigned traversal timestamps t𝑡t and temporary tensor ids t​m​p𝑡𝑚𝑝tmp after executing Algorithm [4](#alg4 "Algorithm 4 ‣ 4.3. Implementation of Optimized Execution plan for TPP Matrix Equations ‣ 4. TPP Matrix Equations ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads").

Now that we have assigned the register scores for each node we can devise an execution plan for the TPP equation tree that minimizes the number of required intermediate tensors. Algorithm [4](#alg4 "Algorithm 4 ‣ 4.3. Implementation of Optimized Execution plan for TPP Matrix Equations ‣ 4. TPP Matrix Equations ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") recursively creates such an optimal execution plan and essentially it calculates: 1) the order/traversal timestamps t𝑡t with which the TPP equation nodes have to be evaluated, and also 2) assigns to each intermediate node r𝑟r a temporary tensor id t​m​pr𝑡𝑚subscript𝑝𝑟tmp\_{r} that holds the intermediate resulting tensor of that TPP node. Figure [13](#S4.F13 "Figure 13 ‣ 4.2. Optimized Execution plan for TPP Matrix Equations ‣ 4. TPP Matrix Equations ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Right shows the optimized execution plan by applying Algorithm [4](#alg4 "Algorithm 4 ‣ 4.3. Implementation of Optimized Execution plan for TPP Matrix Equations ‣ 4. TPP Matrix Equations ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") on our example equation. This algorithm recursively visits/evaluates the children of a node r𝑟r in order of decreasing register score value. This means that the child/subtree with the maximum register score value is evaluated first, one of the temporary tensors is dedicated to hold that child’s intermediate output, whereas the remaining temporary tensors can be reused for the evaluation of the siblings/subtrees, which per definition/order of traversal, require less or equal number of intermediate tensors. Such a strategy guarantees that the temporary tensors are optimally reused/recycled, and as a result we can leverage the minimum required temporary tensors for the evaluation of the entire equation TPP tree. For simplicity in our description, we assumed that all intermediate temporary tensors have the same size, however our implementation considers the actual sizes of the intermediate output tensors and takes the maximum one as representative size for all temporary tensors.

### 4.3. Implementation of Optimized Execution plan for TPP Matrix Equations

By employing Algorithm [4](#alg4 "Algorithm 4 ‣ 4.3. Implementation of Optimized Execution plan for TPP Matrix Equations ‣ 4. TPP Matrix Equations ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") we can devise an optimal execution plan for the TPP Matrix equation, and here we describe the implementation of such a plan. We consider three implementation strategies:

* •

  *Strategy 1*: Using stack-allocated buffers as intermediate temporary tensors
* •

  *Strategy 2*: Using vector-register blocks as intermediate temporary tensors
* •

  *Strategy 3*: Hybrid implementation where some intermediate temporary tensors are stack-allocated buffers and some are vector-register blocks

So far in our description we have used the abstract notation “temporary tensor” without specifying how such a temporary tensor is instantiated in the implementation. The exact instantiation of a temporary/intermediate tensor is the differentiation factor among the 3 implementation strategies for the TPP matrix equations.

Strategy 1 considers each intermediate tensor as a physical buffer, and our TPP equation implementation allocates on the stack some space/buffer for each temporary tensor. Then, by following the timestamp order of the optimal execution plan (e.g. see Figure [13](#S4.F13 "Figure 13 ‣ 4.2. Optimized Execution plan for TPP Matrix Equations ‣ 4. TPP Matrix Equations ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Right), we emit/JIT the corresponding TPP code (e.g. see Algorithms [1](#alg1 "Algorithm 1 ‣ 3.1. Generic TPP Implementation Blueprint ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") and [2](#alg2 "Algorithm 2 ‣ 3.2.1. The BRGEMM kernel structure ‣ 3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")) where the input tensors might be either the equation’s input buffers provided by the user, or one of the stack allocated buffers representing an intermediate result. The fact that we have minimized the number of intermediate temporary buffers/tensors is critical for performance since these stack-allocated buffers may remain in some level of cache. Such a strategy is generic and can be leveraged to implement arbitrary equations. However, Strategy 1 may suffer from store-to-load forwarding inefficiencies on modern processors. Additionally, some of the intermediate tensors may spill from cache (e.g. when the intermediate outputs exceed the corresponding cache capacity) which would make the communication of temporary tensors among TPP nodes via loads/stores from/to stack allocated buffers quite expensive.

Strategy 2 considers each intermediate tensor as an rm×rnsubscript𝑟𝑚subscript𝑟𝑛r\_{m}\times r\_{n} vector-register block. For example, on an AVX512 platform with 32 512-bit wide registers we have available 2 KBytes of register file that may be used for intermediate tensors. Each one of such 512-bit wide vector registers can hold 16 single-precision values and by stacking e.g. 4 of these we can form a logical 16×\times4 intermediate tensor and in total we have available 32/4=8324832/4=8 of such intermediate tensors that could be used by the equation. In Strategy 2 we block the computation of the equation’s output in blocks with size rm×rnsubscript𝑟𝑚subscript𝑟𝑛r\_{m}\times r\_{n}, and we can calculate the corresponding rm×rnsubscript𝑟𝑚subscript𝑟𝑛r\_{m}\times r\_{n} output by following the timestamp order of the optimal execution plan. We emit/JIT the corresponding TPP code for sub-tensors with size rm×rnsubscript𝑟𝑚subscript𝑟𝑛r\_{m}\times r\_{n} where each intermediate output tensor is the assigned temporary vector-register block. Essentially this strategy performs vertical register fusion within the equation TPP nodes and incurs *no* communication via loads/stores from/to stack allocated buffers. However, such a methodology is limited by the number of available vector registers on each platform.

Strategy 3 combines the strengths of Strategies 1 and 2 by considering some intermediate tensors as stack-allocated buffers and some intermediate tensors as vector-register blocks. As such, in Strategy 3 the TPP operations/subtrees which exhibit *both* high register pressure and reuse (e.g.  transposes, GEMM/BRGEMM, transcendental approximations), propagate the intermediate results towards the rest of the TPPs in the tree via stack-allocated temporal tensors. On the other hand, TPP subtrees without large register pressure are implemented using Strategy 2 that employs vertical register fusion and avoids loads/stores from/to stack-allocated buffers.

In addition to the aforementioned 3 strategies, in the TPP equation back-end we identify idioms/motifs of combined TPPs (e.g. a gather TPP followed by a reduce TPP) and we JIT an instruction sequence which is optimal for the composite access pattern. In subsection [5.1.5](#S5.SS1.SSS5 "5.1.5. Sparse Embedding Kernel ‣ 5.1. TPP-based Kernels ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") we show an example of such a combined TPP motif that is optimized by the TPP backend.

Even though we developed a rudimentary method/POC of combining the TPPs via Matrix Equation Trees, we have found that it is sufficient to express all the complex operators we encountered in a wide-range of workloads discussed further in Section [5](#S5 "5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads"). Nevertheless, we envision that when/if TPPs are widely adopted within Tensor Compiler frameworks (e.g.  as an MLIR dialect) then more complicated Graphs (instead of simple trees) and more sophisticated analyses/optimization passes can be leveraged during the composition of TPPs. The key-ingredient that makes the composition of TPPs amenable to optimization opportunities is the TPP specification itself: TPPs comprise a small, well-defined compact set of tensor operators with declarative semantics as shown in Section [2](#S2 "2. The TPP Specification ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads").

We would like also to highlight one use-case of Matrix Equations that can be beneficial for specialized DL accelerators. The BRGEMM TPP described in Section [3.2](#S3.SS2 "3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") corresponds to an output-stationary flow that is suitable for CPUs and GPUs. Given an accelerator that favors e.g. A𝐴A-stationary GEMM formulations, one could express the following Matrix Equation: internal nodes Gisubscript𝐺𝑖G\_{i} would be GEMM ternary TPPs, for each GEMM node Gisubscript𝐺𝑖G\_{i} we would have the same input leaf A𝐴A and a varying input Bisubscript𝐵𝑖B\_{i}, and the output of each node would be a result Cisubscript𝐶𝑖C\_{i}. Essentially this formulation dictates an A𝐴A-stationary flow, and the back-end could optimize accordingly for the specific accelerator.

1:Input: TPP equation tree with root node r𝑟r and assigned register score values on its nodes

2:Output: TPP equation tree with assigned traversal timestamps t𝑡t and temporary tensor ids t​m​p𝑡𝑚𝑝tmp

3:if is\_Leaf( r𝑟r ) then

4:   return

5:if r𝑟r is unary TPP then

6:   Create\_Execution\_Plan( Left\_Child( r𝑟r ) )

7:   trsubscript𝑡𝑟t\_{r} ←←\leftarrow global\_timesteamp++

8:   *▷▷\triangleright If child is leaf, reserve a new tmp, else re-use tmp from child*

9:   if  is\_Leaf( Left\_Child( r𝑟r ) ) then

10:      t​m​pr𝑡𝑚subscript𝑝𝑟tmp\_{r} ←←\leftarrow Reserve\_Tmp()

11:   else

12:      t​m​pr𝑡𝑚subscript𝑝𝑟tmp\_{r} ←←\leftarrow tmp\_Left\_Child( r𝑟r )

13:if r𝑟r is binary TPP then

14:   *▷▷\triangleright Recursively visit children in order of decreasing register score*

15:   Create\_Execution\_Plan( Child\_Max\_Register\_Score( r𝑟r ) )

16:   Create\_Execution\_Plan( Child\_Min\_Register\_Score( r𝑟r ) )

17:   trsubscript𝑡𝑟t\_{r} ←←\leftarrow global\_timesteamp++

18:   *▷▷\triangleright If all children are leaves, reserve a new tmp, else re-use the tmp from a non-leaf child and recycle the tmp of the other non-leaf child*

19:   if  is\_Leaf( Left\_Child( r𝑟r ) AND is\_Leaf( Right\_Child( r𝑟r ) ) ) then

20:      t​m​pr𝑡𝑚subscript𝑝𝑟tmp\_{r} ←←\leftarrow Reserve\_Tmp()

21:   else

22:      if  not\_Leaf( Left\_Child( r𝑟r )  then

23:         t​m​pr𝑡𝑚subscript𝑝𝑟tmp\_{r} ←←\leftarrow tmp\_Left\_Child( r𝑟r )

24:         Recycle\_Tmp( tmp\_Right\_Child( r𝑟r ) )

25:      else

26:         t​m​pr𝑡𝑚subscript𝑝𝑟tmp\_{r} ←←\leftarrow tmp\_Right\_Child( r𝑟r )

27:         Recycle\_Tmp( tmp\_Left\_Child( r𝑟r ) )

28:if r𝑟r is ternary TPP then

29:   *▷▷\triangleright Recursively visit children in order of decreasing register score*

30:   Create\_Execution\_Plan( Child\_Max\_Register\_Score( r𝑟r ) )

31:   Create\_Execution\_Plan( Child\_Mid\_Register\_Score( r𝑟r ) )

32:   Create\_Execution\_Plan( Child\_Min\_Register\_Score( r𝑟r ) )

33:   trsubscript𝑡𝑟t\_{r} ←←\leftarrow global\_timesteamp++

34:   *▷▷\triangleright If all children are leaves, reserve a new tmp, else re-use the tmp from a non-leaf child and recycle the tmps of the other non-leaf children*

35:   if  is\_Leaf( Left\_Child( r𝑟r ) ) AND is\_Leaf( Middle\_Child( r𝑟r ) ) AND is\_Leaf( Right\_Child( r𝑟r ) ) then

36:      t​m​pr𝑡𝑚subscript𝑝𝑟tmp\_{r} ←←\leftarrow Reserve\_Tmp()

37:   else

38:      if  not\_Leaf( Left\_Child( r𝑟r )  then

39:         t​m​pr𝑡𝑚subscript𝑝𝑟tmp\_{r} ←←\leftarrow tmp\_Left\_Child( r𝑟r )

40:         Recycle\_Tmp( tmp\_Middle\_Child( r𝑟r ) ) , Recycle\_Tmp( tmp\_Right\_Child( r𝑟r ) )

41:      else

42:         if  not\_Leaf( Right\_Child( r𝑟r )  then

43:            t​m​pr𝑡𝑚subscript𝑝𝑟tmp\_{r} ←←\leftarrow tmp\_Right\_Child( r𝑟r )

44:            Recycle\_Tmp( tmp\_Middle\_Child( r𝑟r ) ) , Recycle\_Tmp( tmp\_Left\_Child( r𝑟r ) )

45:         else

46:            t​m​pr𝑡𝑚subscript𝑝𝑟tmp\_{r} ←←\leftarrow tmp\_Middle\_Child( r𝑟r )

47:            Recycle\_Tmp( tmp\_Left\_Child( r𝑟r ) ) , Recycle\_Tmp( tmp\_Right\_Child( r𝑟r ) )

Algorithm 4  Create\_Execution\_Plan( r𝑟r )

## 5. TPP-based Kernels & Workloads

This section covers how DL kernels and workloads (image processing, recommendation systems, natural language processing, graph processing and applications in science) can leverage TPPs to achieve high performance. Although this paper’s work is targeting CPUs, we cover the entire training pipeline and not only inference. The main purpose of this is to demonstrate the versatility of TPPs which is valuable in the more complicated backward pass kernels, and to handle training’s implications to the forward pass.

### 5.1. TPP-based Kernels

#### 5.1.1. Softmax Kernel

![Refer to caption](/html/2104.05755/assets/x14.png)


Figure 14. Softmax operator by combining TPPs.

Figure [14](#S5.F14 "Figure 14 ‣ 5.1.1. Softmax Kernel ‣ 5.1. TPP-based Kernels ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") illustrates two Matrix Equation trees that are used to express the softmax operator ([gibbs2014elementary,](#bib.bib31) ):

|  |  |  |  |
| --- | --- | --- | --- |
| (2) |  | Y=softmax​(X)​with​yi​j=e(xi​j−maxxi​j∈X⁡xi​j)∑xi​j∈Xe(xi​j−maxxi​j∈X⁡xi​j)𝑌softmax𝑋withsubscript𝑦𝑖𝑗superscript𝑒subscript𝑥𝑖𝑗subscriptsubscript𝑥𝑖𝑗𝑋subscript𝑥𝑖𝑗subscriptsubscript𝑥𝑖𝑗𝑋superscript𝑒subscript𝑥𝑖𝑗subscriptsubscript𝑥𝑖𝑗𝑋subscript𝑥𝑖𝑗Y=\text{softmax}(X)\ \text{with}\ y\_{ij}=\frac{e^{\left({x\_{ij}-\max\_{x\_{ij}\in X}x\_{ij}}\right)}}{\sum\limits\_{x\_{ij}\in X}e^{\left({x\_{ij}-\max\_{x\_{ij}\in X}x\_{ij}}\right)}} |  |

Equation [2](#S5.E2 "In 5.1.1. Softmax Kernel ‣ 5.1. TPP-based Kernels ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") shows the formula for the softmax operator ([gibbs2014elementary,](#bib.bib31) ), which is often used as the last activation function of a neural network, aiming to normalize its output to a probability distribution. We can represent this operator via two TPP equation trees illustrated in Figure [14](#S5.F14 "Figure 14 ‣ 5.1.1. Softmax Kernel ‣ 5.1. TPP-based Kernels ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads"). The left tree computes the nominator of Equation [2](#S5.E2 "In 5.1.1. Softmax Kernel ‣ 5.1. TPP-based Kernels ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads"): first the maximum value of the input tensor X𝑋X is found (via the max-reduce TPP), then we subtract this max value from each entry of X𝑋X (note the broadcast semantics in the second argument of the subtraction TPP), and a new tensor X′superscript𝑋′X^{\prime} is computed by calculating the element-wise exponent on the earlier subtraction’s outcome. Finally, in the right TPP tree each entry of the tensor X′superscript𝑋′X^{\prime} is normalized by the sum of all values in X′superscript𝑋′X^{\prime} to obtain the softmax output, a tensor Y𝑌Y. This example illustrates the expressiveness of the TPP abstractions, since the components of the mathematical formula map to TPPs in a straightforward way. At the same time, this example highlights the separation of concerns: the user does not need to worry about the efficient implementation of this equation on each different platform, since the TPP back-end is responsible for optimized code generation which is target-specific (contrary to the TPP expression itself which is platform-agnostic).

#### 5.1.2. Normalization Kernels

Batch normalization (batchnorm) is a technique  ([ioffe2015batch,](#bib.bib32) ) that normalizes neuron layer input tensors to improve the overall training process. Batchnorm removes the need for careful parameter initialization and reduces the required training steps ([ioffe2015batch,](#bib.bib32) ) in the neural networks.
The batchnorm computations can be divided in two stages: i) First the mean and variance of the input tensor are computed across the “batch” dimension:
μj=∑i=0n−1xi​jsubscript𝜇𝑗superscriptsubscript𝑖0𝑛1subscript𝑥𝑖𝑗\mu\_{j}=\sum\_{i=0}^{n-1}x\_{ij}, σj2=1n​∑i=0n−1(xi​j−μi)2superscriptsubscript𝜎𝑗21𝑛superscriptsubscript𝑖0𝑛1superscriptsubscript𝑥𝑖𝑗subscript𝜇𝑖2\sigma\_{j}^{2}=\frac{1}{n}\sum\_{i=0}^{n-1}(x\_{ij}-\mu\_{i})^{2} where i𝑖i is the “batch” dimension and j𝑗j is the “feature” dimension, ii) then the tensor entries xi​jsubscript𝑥𝑖𝑗x\_{ij} are normalized based on μ𝜇\mu and σ𝜎\sigma: xi​j′=(xi​j−μj)/(σj2+ϵ)superscriptsubscript𝑥𝑖𝑗′subscript𝑥𝑖𝑗subscript𝜇𝑗superscriptsubscript𝜎𝑗2italic-ϵx\_{ij}^{\prime}=(x\_{ij}-\mu\_{j})/(\sqrt{\sigma\_{j}^{2}+\epsilon}).

Depending upon the workload, different TPPs and TPP equations can be employed to implement the batchnorm. Here, we take an example of batchnorm on a ResNet50 ([he2016deep,](#bib.bib33) ) convolution layer tensor X𝑋X. The input tensor X𝑋X has a four-dimensional shape of {N, C, H, W} with dimensions of batch (N𝑁N), feature (C𝐶C), height (H𝐻H), and width (W𝑊W). We first use sum-reduce TPPs on H𝐻H and W𝑊W dimensions to compute the sum (m​[N,C]𝑚𝑁𝐶m[N,C]) and the sum of squared elements (v​[N,C]𝑣𝑁𝐶v[N,C]) matrices. Subsequently, we use binary add TPPs across the batch dimension of m​[N,C]𝑚𝑁𝐶m[N,C] and v​[N,C]𝑣𝑁𝐶v[N,C] matrices for eventual computation of mean (μ​[C]𝜇delimited-[]𝐶\mu[C]) and variance (σ2​[C]superscript𝜎2delimited-[]𝐶\sigma^{2}[C]) vectors. In the next step, we use a scaling equation to normalize each element of the input tensor. The scaling equation Y=(m′∗X+v′)∗G+B𝑌superscript𝑚′𝑋superscript𝑣′𝐺𝐵Y=(m^{\prime}\*X+v^{\prime})\*G+B converts the input tensor X𝑋X into a normalized tensor Y𝑌Y. Here, G​[C]𝐺delimited-[]𝐶G[C] and B​[C]𝐵delimited-[]𝐶B[C] are scaling vector inputs to batchnorm, and m′​[C]superscript𝑚′delimited-[]𝐶m^{\prime}[C] and v′​[C]superscript𝑣′delimited-[]𝐶v^{\prime}[C] are intermediate vectors that are computed from mean and variance vectors. We implement the scaling equation by a single TPP equation containing two FMADD ternary TPPs. The second equation tree of Figure [15](#S5.F15 "Figure 15 ‣ 5.1.2. Normalization Kernels ‣ 5.1. TPP-based Kernels ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") shows an analogous scaling equation implementation. However, for this particular implementation, we broadcast m′,v′,G,B

superscript𝑚′superscript𝑣′𝐺𝐵m^{\prime},v^{\prime},G,B vectors into H𝐻H, W𝑊W, and N𝑁N dimensions inside the TPP equation tree. An efficient implementation of batchnorm uses blocking on the C𝐶C, H𝐻H, and W𝑊W dimensions along with multi-threading on the N𝑁N and feature block dimension. We do not show the details of this implementation for sake of simplicity.

![Refer to caption](/html/2104.05755/assets/x15.png)


Figure 15. Layernorm via TPPs.

Layer normalization (layernorm) ([ba2016layer,](#bib.bib34) ) is a technique that normalizes the neurons *within* a layer, and was motivated by the limitations of Batch Normalization ([ioffe2015batch,](#bib.bib32) ) in Recurrent Neural Networks.
The layernorm computations can be divided in two stages: i) First the mean and variance of the input tensor are computed across the “feature” dimension:
μi=∑j=0m−1xi​jsubscript𝜇𝑖superscriptsubscript𝑗0𝑚1subscript𝑥𝑖𝑗\mu\_{i}=\sum\_{j=0}^{m-1}x\_{ij}, σi2=1m​∑j=0m−1(xi​j−μi)2superscriptsubscript𝜎𝑖21𝑚superscriptsubscript𝑗0𝑚1superscriptsubscript𝑥𝑖𝑗subscript𝜇𝑖2\sigma\_{i}^{2}=\frac{1}{m}\sum\_{j=0}^{m-1}(x\_{ij}-\mu\_{i})^{2} where i𝑖i is the batch dimension and j𝑗j is the “feature” dimension, ii) then the tensor entries xi​jsubscript𝑥𝑖𝑗x\_{ij} are normalized based on μ𝜇\mu and σ𝜎\sigma: xi​j′=(xi​j−μi)/(σi2+ϵ)superscriptsubscript𝑥𝑖𝑗′subscript𝑥𝑖𝑗subscript𝜇𝑖superscriptsubscript𝜎𝑖2italic-ϵx\_{ij}^{\prime}=(x\_{ij}-\mu\_{i})/(\sqrt{\sigma\_{i}^{2}+\epsilon}). Depending on the workload (e.g. attention cell in BERT), the scaled tensor may be further scaled with two other tensors γ𝛾\gamma and β𝛽\beta. Figure [15](#S5.F15 "Figure 15 ‣ 5.1.2. Normalization Kernels ‣ 5.1. TPP-based Kernels ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") illustrates two TPP equation trees that implement this composite layernorm operator. The left equation is using the sum-reduce TPP to compute the sum and sum of squared elements of the input tensor, namely m𝑚m and v𝑣v. These two scalars are combined (not shown in the equation for simplicity), and are fed as inputs to the right TPP tree, where the FMADD ternary TPP is used to scale the input tensor X𝑋X. Finally, a cascading FMADD ternary TPP computes the final result via the scaling tensors G𝐺G and B𝐵B. We illustrate this layernorm via means of TPPs since all DL norming layers essentially exhibit similar computational motif, and this specific norm is used in the BERT workload described in subsection [5.2.3](#S5.SS2.SSS3 "5.2.3. Natural Language Processing - BERT ‣ 5.2. TPP-based Workloads ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads").

Group normalization (groupnorm) ([wu2018group,](#bib.bib35) ) is a technique that normalizes the neurons within a group of features. Groupnorm was proposed as an alternative to batchnorm ([ioffe2015batch,](#bib.bib32) ) to reduce normalization error for smaller batch sizes. In groupnorm, features are divided into groups, and mean and variance are computed within each group for normalization. Groupnorm is also a generalization of the layer normalization ([ba2016layer,](#bib.bib34) ) and instance normalization ([ulyanov2016instance,](#bib.bib36) ) approach. Layernorm is groupnorm with a single group, and instance norm is groupnorm with group size equal to one. Groupnorm can be implemented with the same set of TPPs and TPP equations that were used in the batchnorm kernel. We again take the example of ResNet50 ([he2016deep,](#bib.bib33) ) convolution layer tensor X𝑋X and apply groupnorm on it with g𝑔g number of groups. We can ignore the batch dimension (N𝑁N) for this discussion as groupnorm works independently upon each batch. Therefore, the input tensor X𝑋X now has a three-dimensional shape of {C, H, W} with dimensions of feature (C𝐶C), height (H𝐻H), and width (W𝑊W). We first use sum-reduce TPPs on H𝐻H and W𝑊W dimensions to compute the sum (m​[C]𝑚delimited-[]𝐶m[C]) and the sum of squared elements (v​[C]𝑣delimited-[]𝐶v[C]) vectors. Subsequently, we add m​[C]𝑚delimited-[]𝐶m[C] and v​[C]𝑣delimited-[]𝐶v[C] values within a feature group for eventual computation of group mean (μ​[g]𝜇delimited-[]𝑔\mu[g]) and group variance (σ2​[g]superscript𝜎2delimited-[]𝑔\sigma^{2}[g]) vectors. Similar to batchnorm, we use a scaling equation to normalize each element of the input tensor. The scaling equation Y=(m′∗X+v′)∗G+B𝑌superscript𝑚′𝑋superscript𝑣′𝐺𝐵Y=(m^{\prime}\*X+v^{\prime})\*G+B converts input tensor X𝑋X into a normalized tensor Y𝑌Y. Here, G​[C]𝐺delimited-[]𝐶G[C] and B​[C]𝐵delimited-[]𝐶B[C] are scaling vector inputs to groupnorm, and m′​[C]superscript𝑚′delimited-[]𝐶m^{\prime}[C] and v′​[C]superscript𝑣′delimited-[]𝐶v^{\prime}[C] are intermediate vectors that are computed from group mean and group variance vectors. The second equation tree of Figure [15](#S5.F15 "Figure 15 ‣ 5.1.2. Normalization Kernels ‣ 5.1. TPP-based Kernels ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") shows an analogous scaling equation implementation. However, for this particular implementation, we broadcast m′,v′,G,B

superscript𝑚′superscript𝑣′𝐺𝐵m^{\prime},v^{\prime},G,B vectors into H𝐻H and W𝑊W dimensions inside the TPP equation tree. We can also apply the same scaling equation to a single group or set of groups with few parameter changes. An efficient implementation of groupnorm uses blocking on the C𝐶C, H𝐻H, and W𝑊W dimensions. We do not show the details of this implementation for sake of simplicity.

#### 5.1.3. BF16 Split-SGD Kernel

![Refer to caption](/html/2104.05755/assets/x16.png)


Figure 16. BF16 Split-SGD operator by combining TPPs.

Unlike the previous kernels which are well-established in DL workloads, and as such potentially optimized in DL libraries, we present here an example of a novel operator, which per definition is not existent in DL libraries. BF16 split-SGD was recently introduced in the context of DLRM training with BF16 datatype ([kalamkar2020optimizing,](#bib.bib37) ). The Split-SGD-BF16 solver aims at efficiently exploiting the aliasing of BF16 and FP32 (i.e. the 16 Most Significant Bits (MSB) on both are identical) in order to save bandwidth during the SGD-solver in training.
The employed trick is that the weights are not stored as FP32 values in a single tensor. Instead, the FP32 tensors are split into their high and low 16bit-wide parts: the 16 MSBs of the FP32 values, and the 16 LSBs of the same values are stored as two separate tensors Xh​isuperscript𝑋ℎ𝑖X^{hi} and Xl​osuperscript𝑋𝑙𝑜X^{lo} respectively. The 16 MSBs represent a valid BF16 number and constitute the model/weight tensors during training. These BF16 weights are used exclusively in the forward and backward passes, whereas the lower 16 bits are only required in optimizer. More specifically, the Xh​isuperscript𝑋ℎ𝑖X^{hi} and Xl​osuperscript𝑋𝑙𝑜X^{lo} tensors are packed together to form an FP32 tensor, resulting in a fully FP32-accurate optimizer. Figure [16](#S5.F16 "Figure 16 ‣ 5.1.3. BF16 Split-SGD Kernel ‣ 5.1. TPP-based Kernels ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") illustrates the BF16 Split-SGD operator written entirely via TPPs. First the Xh​isuperscript𝑋ℎ𝑖X^{hi} and Xl​osuperscript𝑋𝑙𝑜X^{lo} are packed, and the formed FP32 tensor is used in a cascading FMADD TPP that performs the SGD scaling with the corresponding Gradient Weight tensor and learning rate. Finally, the resulting FP32 tensor is unpacked to the Xh​isuperscript𝑋ℎ𝑖X^{hi} and Xl​osuperscript𝑋𝑙𝑜X^{lo} tensors for further use in the training process.

#### 5.1.4. Convolutional Neural Network (CNN) kernel

Convolutional Neural Networks (CNN) consist of layers with multiple neurons connected by weights, and they have been applied with success in image recognition, semantic segmentation, autonomous driving, medical imaging and in an increasing number of scientific applications. Previous work ([sc18,](#bib.bib23) ; [georganas2020harnessing,](#bib.bib21) ) has shown that CNNs, despite their seemingly complicated loop structure due to the involved high-dimensional tensors, can be mapped efficiently onto small 2D GEMMs and BRGEMMs. In this work, we adopt the same strategy to implement CNNs via the BRGEMM TPP. Unlike the previous work which presents only the address-based BRGEMM formulation, here we leverage the CNN kernels with stride-based BRGEMM for 1×\times1 convolutions and offset-based BRGEMM for 3×\times3 convolutions to get even more performant implementations (see Section [2.3](#S2.SS3 "2.3. The TPP collection ‣ 2. The TPP Specification ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") for a brief description of the BRGEMM variants).

#### 5.1.5. Sparse Embedding Kernel

1:Inputs: αT=[0,…,ap1,…,apk,…,0]superscript𝛼𝑇

0…subscript𝑎subscript𝑝1…subscript𝑎subscript𝑝𝑘…0\alpha^{T}=[0,\ldots,a\_{p\_{1}},\ldots,a\_{p\_{k}},\ldots,0] with entries ap=1subscript𝑎𝑝1a\_{p}=1 for p∈{p1,…,pk}𝑝subscript𝑝1…subscript𝑝𝑘p\in\{p\_{1},\ldots,p\_{k}\} and 00 elsewhere, WM×Esuperscript𝑊𝑀𝐸W^{M\times E}

2:Output: oT=aT×Wsuperscript𝑜𝑇superscript𝑎𝑇𝑊o^{T}=a^{T}\times W

3:for j=0​…​E​with step​v​l​e​n⋅U𝑗⋅0…𝐸with step𝑣𝑙𝑒𝑛𝑈j=0\dots E\ \textbf{with step}\ vlen\cdot U do

4:   *▷▷\triangleright Initializing accumulator registers to 0*

5:   for u=0​…​U−1𝑢0…𝑈1u=0\dots U-1 do

6:      v​e​c​\_​o​u​tu←0←𝑣𝑒𝑐\_𝑜𝑢subscript𝑡𝑢0vec\\_out\_{u}\leftarrow 0

7:   *▷▷\triangleright Iterating over non-zero entries/indices in αTsuperscript𝛼𝑇\alpha^{T}*

8:   for i​in​ 1,2,…,k

𝑖in12…𝑘i\ \textbf{in}\ 1,2,\ldots,\ k do

9:      i​d​x=pi𝑖𝑑𝑥subscript𝑝𝑖idx=p\_{i}

10:      n​e​x​t​\_​i​d​x=pi+1𝑛𝑒𝑥𝑡\_𝑖𝑑𝑥subscript𝑝𝑖1next\\_idx=p\_{i+1}

11:      *▷▷\triangleright Unroll innermost kernel U𝑈U times: load indexed vector, prefetch next indexed vector, accumulate loaded vector to accumulator register*

12:      for u=0​…​U−1𝑢0…𝑈1u=0\dots U-1 do

13:         v​e​c​\_​W←←𝑣𝑒𝑐\_𝑊absentvec\\_W\leftarrow load\_vector(W[idx][j+u⋅vlen:j+(u+1)⋅vlen]W[idx][j+u\cdot vlen:j+(u+1)\cdot vlen])

14:         prefetch(W[next\_idx][j+u⋅vlen:j+(u+1)⋅vlen]W[next\\_idx][j+u\cdot vlen:j+(u+1)\cdot vlen])

15:         vec\_outu+=vec\_Wvec\\_out\_{u}\mathrel{+}=vec\\_W

16:   *▷▷\triangleright Store accumulator registers to oTsuperscript𝑜𝑇o^{T}*

17:   for u=0​…​U−1𝑢0…𝑈1u=0\dots U-1 do

18:      oT[j+u⋅vlen:j+(u+1)⋅vlen]←vec\_outuo^{T}[j+u\cdot vlen:j+(u+1)\cdot vlen]\leftarrow vec\\_out\_{u}

Algorithm 5  Sparse Gather-Reduce operation

![Refer to caption](/html/2104.05755/assets/x17.png)


Figure 17. Sparse Embedding Lookups via TPPs

The sparse embedding kernel is comprised of multi-hot encoded lookups into an embedding table WM×Esuperscript𝑊𝑀𝐸W^{M\times E} with M𝑀M being the number of rows and E𝐸E the length of each row, whereas the multi-hot weight-vector is denoted as αT=[0,…,ap1,…,apk,…,0]superscript𝛼𝑇

0…subscript𝑎subscript𝑝1…subscript𝑎subscript𝑝𝑘…0\alpha^{T}=[0,\ldots,a\_{p\_{1}},\ldots,a\_{p\_{k}},\ldots,0] with entries ap=1subscript𝑎𝑝1a\_{p}=1 for p∈{p1,…,pk}𝑝subscript𝑝1…subscript𝑝𝑘p\in\{p\_{1},\ldots,p\_{k}\} and 00 elsewhere (p𝑝p being the index for the corresponding lookup items). Mathematically, the embedding lookup output vector oTsuperscript𝑜𝑇o^{T} can be obtained via oT=aT×Wsuperscript𝑜𝑇superscript𝑎𝑇𝑊o^{T}=a^{T}\times W. This operation (assuming row-major storage for W𝑊W) is equivalent to gathering the rows of W𝑊W based on the non-zero indices apsubscript𝑎𝑝a\_{p}, and then adding them up to get the output vector oTsuperscript𝑜𝑇o^{T}. Figure [17](#S5.F17 "Figure 17 ‣ 5.1.5. Sparse Embedding Kernel ‣ 5.1. TPP-based Kernels ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") illustrates the TPP tree that is used to express the Sparse Embedding lookup kernel.

We note that the TPP backend optimizes this sequence of TPPs, and performs register fusion across the gather and the reduce TPP components. More specifically, given a non-zero index apsubscript𝑎𝑝a\_{p}, the corresponding row of W𝑊W is loaded in vector registers, and is added to a set of running accumulators/vector registers that hold the output oTsuperscript𝑜𝑇o^{T}. Algorithm [5](#alg5 "Algorithm 5 ‣ 5.1.5. Sparse Embedding Kernel ‣ 5.1. TPP-based Kernels ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") illustrates the optimized JITed implementation in our TPP backend. The E𝐸E dimension is vectorized in an SIMD-fashion with vector length v​l​e​n𝑣𝑙𝑒𝑛vlen. Note that in line 13 we expose multiple independent accumulation chains in order to hide the latency of the vector-add SIMD instructions. Since we JIT this sub-procedure, we know the exact value of E𝐸E at runtime. As such, we can pick appropriate unrolling factor U𝑈U as well as the remainder handling can be performed optimally via masking in case E𝐸E is not perfectly divisible by the vector length v​l​e​n𝑣𝑙𝑒𝑛vlen. Last but not least, the JITed aggregation procedure employs prefetching of the subsequent indexed vectors in W𝑊W (line 12) in order to hide the latency of these irregular accesses.

#### 5.1.6. Multi-Layer Perceptron (MLP) kernel

1:Inputs: AMb×Kb×bk×bmsuperscript𝐴subscript𝑀𝑏subscript𝐾𝑏subscript𝑏𝑘subscript𝑏𝑚A^{M\_{b}\times K\_{b}\times b\_{k}\times b\_{m}}, BNb×Kb×bn×bksuperscript𝐵subscript𝑁𝑏subscript𝐾𝑏subscript𝑏𝑛subscript𝑏𝑘B^{N\_{b}\times K\_{b}\times b\_{n}\times b\_{k}}

2:Output: CNb×Mb×bn×bmsuperscript𝐶subscript𝑁𝑏subscript𝑀𝑏subscript𝑏𝑛subscript𝑏𝑚C^{N\_{b}\times M\_{b}\times b\_{n}\times b\_{m}}

3:Based on t​h​r​e​a​d​\_​i​d𝑡ℎ𝑟𝑒𝑎𝑑\_𝑖𝑑thread\\_id calculate Mb​\_​s​t​a​r​tsubscript𝑀𝑏\_𝑠𝑡𝑎𝑟𝑡M\_{b}\\_start, Mb​\_​e​n​dsubscript𝑀𝑏\_𝑒𝑛𝑑M\_{b}\\_end, Nb​\_​s​t​a​r​tsubscript𝑁𝑏\_𝑠𝑡𝑎𝑟𝑡N\_{b}\\_start and Nb​\_​e​n​dsubscript𝑁𝑏\_𝑒𝑛𝑑N\_{b}\\_end to assign output work items

4:for i​bn=Nb​\_​s​t​a​r​t​…​Nb​\_​e​n​d𝑖subscript𝑏𝑛subscript𝑁𝑏\_𝑠𝑡𝑎𝑟𝑡…subscript𝑁𝑏\_𝑒𝑛𝑑ib\_{n}=N\_{b}\\_start\dots N\_{b}\\_end do

5:   for i​bm=Mb​\_​s​t​a​r​t​…​Mb​\_​e​n​d𝑖subscript𝑏𝑚subscript𝑀𝑏\_𝑠𝑡𝑎𝑟𝑡…subscript𝑀𝑏\_𝑒𝑛𝑑ib\_{m}=M\_{b}\\_start\dots M\_{b}\\_end do

6:      O​u​t=&C​[i​bn]​[i​bm]​[0]​[0]𝑂𝑢𝑡𝐶delimited-[]𝑖subscript𝑏𝑛delimited-[]𝑖subscript𝑏𝑚delimited-[]0delimited-[]0Out=\&C[ib\_{n}][ib\_{m}][0][0]

7:      *▷▷\triangleright Stride-based BRGEMM, stride\_A=bk⋅bm⋅subscript𝑏𝑘subscript𝑏𝑚b\_{k}\cdot b\_{m}, stride\_B=bn⋅bk⋅subscript𝑏𝑛subscript𝑏𝑘b\_{n}\cdot b\_{k}*

8:      𝐁𝐑𝐆𝐄𝐌𝐌​(&A​[i​bm]​[0]​[0]​[0],&B​[i​bn]​[0]​[0]​[0],O​u​t,Kb)𝐁𝐑𝐆𝐄𝐌𝐌𝐴delimited-[]𝑖subscript𝑏𝑚delimited-[]0delimited-[]0delimited-[]0𝐵delimited-[]𝑖subscript𝑏𝑛delimited-[]0delimited-[]0delimited-[]0𝑂𝑢𝑡subscript𝐾𝑏\mathbf{BRGEMM}(\&A[ib\_{m}][0][0][0],\&B[ib\_{n}][0][0][0],Out,K\_{b})

9:      C​[i​bn]​[i​bm]​[0]​[0]←𝐔𝐍𝐀𝐑𝐘​(C​[i​bn]​[i​bm]​[0]​[0])←𝐶delimited-[]𝑖subscript𝑏𝑛delimited-[]𝑖subscript𝑏𝑚delimited-[]0delimited-[]0𝐔𝐍𝐀𝐑𝐘𝐶delimited-[]𝑖subscript𝑏𝑛delimited-[]𝑖subscript𝑏𝑚delimited-[]0delimited-[]0C[ib\_{n}][ib\_{m}][0][0]\leftarrow\mathbf{UNARY}(C[ib\_{n}][ib\_{m}][0][0])

Algorithm 6  Fully-Connected Layer with Unary Activation Function

Multilayer perceptrons (MLP) form a class of feed-forward artificial neural networks. An MLP consists of (at least three) *fully connected* layers of neurons. Each neuron in the topology may be using a non-linear activation function. In this section we present the implementation of the *Fully Connected* layers since they constitute the cornerstone of MLP. Even though we illustrate the forward pass of Fully Connected layers, we also implement via TPPs the kernels of the back-propagation training in an analogous fashion. Algorithm [6](#alg6 "Algorithm 6 ‣ 5.1.6. Multi-Layer Perceptron (MLP) kernel ‣ 5.1. TPP-based Kernels ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") shows the fully connected layer implementation which is mapped to TPPs. First we note that the input tensors are conceptually 2D matrices AM×Ksuperscript𝐴𝑀𝐾A^{M\times K} and BK×Nsuperscript𝐵𝐾𝑁B^{K\times N} that need to be multiplied. We follow the approach of previous work ([georganas2020harnessing,](#bib.bib21) ) and we block the dimensions M𝑀M, K𝐾K, and N𝑁N by factors bmsubscript𝑏𝑚b\_{m}, bksubscript𝑏𝑘b\_{k}, and bnsubscript𝑏𝑛b\_{n} respectively. Such a blocked layout is exposing better locality and avoids large, strided sub-tensor accesses which are known to cause TLB misses and cache conflict misses in case the leading dimensions are large powers of 2 ([georganas2020harnessing,](#bib.bib21) ). We leverage the BRGEMM TPP in order to perform the tensor contraction with A𝐴A and B𝐵B across their dimensions Kbsubscript𝐾𝑏K\_{b} and bksubscript𝑏𝑘b\_{k} (which constitute the K𝐾K/inner-product dimension of the original 2D matrices). We employ the stride-based BRGEMM because the sub-blocks “Aisubscript𝐴𝑖A\_{i}” and “Bisubscript𝐵𝑖B\_{i}” that have to be multiplied and reduced are apart by constant strides s​t​r​i​d​e​\_​A=bk⋅bm𝑠𝑡𝑟𝑖𝑑𝑒\_𝐴⋅subscript𝑏𝑘subscript𝑏𝑚stride\\_A=b\_{k}\cdot b\_{m} and s​t​r​i​d​e​\_​B=bn⋅bk𝑠𝑡𝑟𝑖𝑑𝑒\_𝐵⋅subscript𝑏𝑛subscript𝑏𝑘stride\\_B=b\_{n}\cdot b\_{k} respectively. Finally, we apply (optionally) a unary TPP corresponding to the requested activation function (e.g. RELU) onto the just-computed output block of C𝐶C.

### 5.2. TPP-based Workloads

#### 5.2.1. 1D Dilated Convolutions & Computational Biology

Algorithm 7  1D Dilated convolution forward pass using TPPs

Inputs: IC×Wsuperscript𝐼𝐶𝑊I^{C\times W}, WK×C×Ssuperscript𝑊𝐾𝐶𝑆W^{K\times C\times S}, d∈ℝ𝑑ℝd\in\mathbb{R}

Output: OK×Qsuperscript𝑂𝐾𝑄O^{K\times Q}

1:WT←𝐓𝐑𝐀𝐍𝐒𝐏𝐎𝐒𝐄​(W)←superscript𝑊𝑇𝐓𝐑𝐀𝐍𝐒𝐏𝐎𝐒𝐄𝑊W^{T}\leftarrow\mathbf{TRANSPOSE}(W)

2:for p​o​s=0​…​Q−1​with step ​𝐛𝐪𝑝𝑜𝑠0…𝑄1with step subscript𝐛𝐪pos=0\dots Q-1\ \textbf{with\ step\ }\mathbf{b\_{q}} do

3:   *▷▷\triangleright Address-based BRGEMM, prepare arguments Ap​t​r​s,Bp​t​r​s

subscript𝐴𝑝𝑡𝑟𝑠subscript𝐵𝑝𝑡𝑟𝑠A\_{ptrs},\ B\_{ptrs}*

4:   for s=0​…​S−1​with step ​𝟏𝑠0…𝑆1with step 1s=0\dots S-1\ \textbf{with\ step\ }\mathbf{1} do

5:      Ap​t​r​s​[s]=&WT​[s,0,0]subscript𝐴𝑝𝑡𝑟𝑠delimited-[]𝑠superscript𝑊𝑇

𝑠00A\_{ptrs}[s]=\&W^{T}[s,0,0]

6:      Bp​t​r​s​[s]=&I​[0,(p​o​s+s⋅d)]subscript𝐵𝑝𝑡𝑟𝑠delimited-[]𝑠𝐼0𝑝𝑜𝑠⋅𝑠𝑑B\_{ptrs}[s]=\&I[0,(pos+s\cdot d)]

7:   𝐁𝐑𝐆𝐄𝐌𝐌​(Ap​t​r​s,Bp​t​r​s,&O​[0,p​o​s],S)𝐁𝐑𝐆𝐄𝐌𝐌subscript𝐴𝑝𝑡𝑟𝑠subscript𝐵𝑝𝑡𝑟𝑠𝑂0𝑝𝑜𝑠𝑆{\mathbf{BRGEMM}}(A\_{ptrs},B\_{ptrs},\&O[0,pos],S)

In this subsection, we show the implementation of a special type of convolution via TPPs in their entirety, namely one-dimensional (1D) dilated convolution layer of a 1D CNN named ATACworks ([Lal829481,](#bib.bib38) ). ATACworks is used for de-noising and peak calling from ATAC-Seq genomic sequencing data ([Lal829481,](#bib.bib38) ). The 1D dilated convolution layer in ATACworks takes more than 90% of the training time, and it has input tensor width W𝑊W, output tensor width Q𝑄Q, C𝐶C input channels, K𝐾K output channels, filter size of S𝑆S, and dilation d𝑑d. We employ the transpose TPPs, copy TPPs, and BRGEMM TPPs to optimize the forward pass and the backward pass of the PyTorch-based 1D convolution layer. Algorithm [7](#alg7 "Algorithm 7 ‣ 5.2.1. 1D Dilated Convolutions & Computational Biology ‣ 5.2. TPP-based Workloads ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") shows an example of the forward pass procedure with an input tensor I𝐼I, a weight tensor W𝑊W, and an output tensor O𝑂O.

#### 5.2.2. Deep Learning Recommendation Model

Facebook recently proposed a deep learning recommendation model (DLRM) ([naumov2019deep,](#bib.bib39) ). Its purpose is to assist the systematic hardware-software co-design for deep learning systems. DLRM is comprised of the following major components: (a) a sparse embedding (see subsection [5.1.5](#S5.SS1.SSS5 "5.1.5. Sparse Embedding Kernel ‣ 5.1. TPP-based Kernels ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")) involving tables (databases) of varying sizes, (b) a small dense Multi-Layer Perceptron (see subsection [5.1.6](#S5.SS1.SSS6 "5.1.6. Multi-Layer Perceptron (MLP) kernel ‣ 5.1. TPP-based Kernels ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")), and (c) a larger and deeper MLP which is fed by the interaction among (a) and (b). All three parts can be configured (number of features, mini-batch sizes and table sizes) to stress different aspects of the system. We also note that in the case of training with BF16 datatype we leverage the BF16 split-SGD optimizer (see subsection [5.1.3](#S5.SS1.SSS3 "5.1.3. BF16 Split-SGD Kernel ‣ 5.1. TPP-based Kernels ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")). For more details on the workload and CPU-oriented optimizations we refer to prior work ([kalamkar2020optimizing,](#bib.bib37) ).

#### 5.2.3. Natural Language Processing - BERT

The BERT model is a bidirectional transformer pre-trained via a combination of masked language modeling objective, and next-sentence prediction ([devlin2018bert,](#bib.bib40) ). The heart of the BERT model is
comprised by sequence of BERT layers which are built using smaller building blocks. For ease of use and implementation, we followed modular building blocks from Hugging Face transformers library ([wolf-etal-2020-transformers,](#bib.bib41) ) and implemented four fused
layers using TPP building blocks, namely *Bert-Embeddings*, *Bert-SelfAttention*, *Bert-Output*/*Bert-SelfOutput* and *Bert-Intermediate* layers.

The *SelfAttention* layer in turn can be formulated as a bunch of Matrix / batch Matrix-Multiplications mixed with element-wise scale, add, dropout and softmax operators. We formulate these Matrix-Multiplications as tensor contractions on blocked-tensors via the stride-based BRGEMM TPP (similarly to Algorithm [6](#alg6 "Algorithm 6 ‣ 5.1.6. Multi-Layer Perceptron (MLP) kernel ‣ 5.1. TPP-based Kernels ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")). We opt to use blocked tensor layouts for the same reasons briefly described in Section [5.1.6](#S5.SS1.SSS6 "5.1.6. Multi-Layer Perceptron (MLP) kernel ‣ 5.1. TPP-based Kernels ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads"). Furthermore, by working on one small sub-tensor at a time we naturally follow a “dataflow” computation, which has been shown to maximize the out-of-cache-reuse of tensors among cascading operators ([banerjee2019optimizing,](#bib.bib26) ; [zhang2018deepcpu,](#bib.bib42) ). The softmax operator is also formulated entirely by TPPs as described in Section [5.1.1](#S5.SS1.SSS1 "5.1.1. Softmax Kernel ‣ 5.1. TPP-based Kernels ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads"). We note that the sequence of Matrix-Multiplications in the attention layer requires sub-tensors to be transposed (and VNNI transformed in case of BF16 implementation), and for this task we leverage the transpose/transform TPPs. *Bert-Output* and *Bert-SelfOutput* layers perform GEMM over blocked layout, and fuse bias addition, dropout, residual addition and layernorm using TPPs. The *Bert-Embeddings* layer also performs layernorm and dropout after embedding lookups that are also implemented using TPPs. Finally, *Bert-Intermediate* layer performs blocked GEMM followed by bias addition and GELU activation function which we implement using the GELU TPP.

#### 5.2.4. Emerging AI - Graph Neural Networks

![Refer to caption](/html/2104.05755/assets/x18.png)


Figure 18. Binary-Reduce aggregation kernel via TPPs

Graph Neural Networks (GNN) ([hamilton2017inductive,](#bib.bib43) ) form an emerging class of Neural Networks for learning the structure of large, population-scale graphs. Depending on the specific algorithm and task that a GNN is designed for (e.g. node classification, link prediction), feature-vector aggregation precedes or succeeds a shallow neural network. Such a shallow neural network typically materializes one or more linear transformations, followed by a classification or regression mechanism ([avancha2020deep,](#bib.bib44) ), and the relevant TPP-based implementation is essentially the one we present in Algorithm [6](#alg6 "Algorithm 6 ‣ 5.1.6. Multi-Layer Perceptron (MLP) kernel ‣ 5.1. TPP-based Kernels ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads").

We focus here on the TPP-based implementation of the feature-vector aggregation. This aggregation motif can be seen as a sequence of linear algebraic expressions involving node/edge features, along with the relevant operators. Prior work ([avancha2020deep,](#bib.bib44) ) has focused on the following two algebraic sequences: Copy-Reduce and Binary-Reduce. We elaborate here on the latter sequence Binary-Reduce (as the first is even simpler). The feature-vectors (either pertaining to vertices or edges) are represented via dense 2D matrices/tables. At the same time, the adjacency information in the graphs can be eventually found via arrays of indices. Therefore, by providing a set of indices and the appropriate Tables of feature-vectors (assuming column-major storage), one can extract selectively the desired feature-vectors via Gather-columns operations. Then, the extracted feature-vectors are fed into a binary operator, and the outcome of the binary operations are finally reduced (the reduce operation could be sum/max/min etc).

Figure [18](#S5.F18 "Figure 18 ‣ 5.2.4. Emerging AI - Graph Neural Networks ‣ 5.2. TPP-based Workloads ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") illustrates a TPP tree that is used to express the Binary-Reduce aggregation kernel. The TPP back-end optimizes this sequence of TPPs and performs horizontal register fusion across them. More precisely, two feature-vectors namely v0subscript𝑣0v\_{0} and v1subscript𝑣1v\_{1} are extracted at a time from Table 0 and Table 1 respectively by using the relevant indices arrays, and they are combined via the proper binary op to get an intermediate vector visubscript𝑣𝑖v\_{i}. Subsequently, visubscript𝑣𝑖v\_{i} is reduced with a running reduce-vector vosubscript𝑣𝑜v\_{o} that holds the output of this composite operator. Once the running reduction has been completed (i.e. all indexed columns from Table 0 and Table 1 have been accessed, processed and reduced), the output vector vosubscript𝑣𝑜v\_{o} is stored in the corresponding output subtensor.

## 6. Experimental Results of DL kernels & Workloads

We use a variety of platforms that span different ISAs, different vendors and micro-architectures. More specifically, our tested platforms include: i) a 22-core Intel Xeon E5-2699 v4 (BDX) supporting up to AVX2 ISA, ii) a 28-core Intel Xeon 8280 (CLX) supporting up to AVX512 ISA, iii) a recently announced 40-core Intel Xeon 8380 (ICX) supporting also up to AVX512 ISA, iv) a 28-core Intel Xeon 8380H (CPX) supporting up to AVX512 ISA, which also offers BF16 FMA acceleration, v) a 64-core AMD EPYC 7742 (ROME) with AVX2 ISA, vi) an AWS Graviton2 instance with 64 cores at fixed 2.5 GHz and AArch64 ISA, vii) a 48-core Fujitsu A64FX at fixed 1.8 GHz with ARMv8 SVE ISA, and viii) a 4-core client Intel i7-6700 CPU (i7) supporting up to AVX2 ISA. All Intel and AMD chips are operating in Turbo mode. For the cluster experiments we used a 32 node CLX installation with a dual-rail Intel Omnipath 100 pruned 2:1 fat-tree topology.

### 6.1. Performance of standalone DL kernels

![Refer to caption](/html/2104.05755/assets/x19.png)


Figure 19. TPP kernels on CLX

![Refer to caption](/html/2104.05755/assets/x20.png)


Figure 20. TPP kernels on ROME

We start the performance evaluation with standalone TPP kernels presented in subsection [5.1](#S5.SS1 "5.1. TPP-based Kernels ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads"). First, we want to highlight the productivity/efficiency provided by TPPs: the high-level code expressed via TPPs/trees of TPPs can match or outperform code by compilers, and hand-vectorized (thus non-portable code) written by performance experts. Second, we want to show the portability aspect of TPPs, since exactly the same high-level code yields high-performance across different ISAs and micro-architectures.

Figure [19](#S6.F19 "Figure 19 ‣ 6.1. Performance of standalone DL kernels ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Top shows the performance of the Softmax operator of blocked 3D tensors with size S​1×S​2×S​3𝑆1𝑆2𝑆3S1\times S2\times S3, on the CLX platform (i.e. targeting AVX512 ISA). Here we perform S​2𝑆2S2 softmax operations over blocked S​1×S​3𝑆1𝑆3S1\times S3 dimensions. The sizes are chosen such that some of the dimensions do not match perfectly with the vector length. The baseline is the icc generated code with -O3 optimization level and high-zmm usage flags. The second variant is also icc-generated code, but we propagate the tensor sizes/loop bounds via compile-time constants in order to assist the auto-vectorization/optimize remainder handling via masking. The third code variant is the AVX512 hand-vectorized by an expert, where the e​x​p𝑒𝑥𝑝exp function uses fast Taylor approximation. Last, we evaluated the TPP-based softmax implementation. As we can see, by propagating the tensor sizes we achieve (geo-mean) speedup of 1.3×\times over the baseline. The hand-vectorized code is faster by 2.6×\times whereas the TPP-based variant shows similar speedups by being 2.2×\times faster. The main shortcoming of the hand-vectorized code is that it is platform-dependent and as such non-portable. More specifically, we didn’t have to our avail AVX2 hand-optimized code in order to experiment with it on ROME. On the contrary,
Figure [20](#S6.F20 "Figure 20 ‣ 6.1. Performance of standalone DL kernels ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Top shows the softmax performance on AVX2 enabled platform for the compiler-generated code and the TPP based code. The TPP-based softmax exhibits geo-mean speedup of 2.45×\times over the baseline on ROME.

Figure [19](#S6.F19 "Figure 19 ‣ 6.1. Performance of standalone DL kernels ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Middle shows the performance of the layernorm operator on the CLX platform. Since the layernorm code is more straightforward (i.e. no expensive *exp* function is involved), we see that icc with compile-constant bounds outperforms by 1.9×\times the baseline. We inspected the compiler-generated code and identified that the reduction-loops were recognized and were heavily optimized with multiple accumulation chains etc. Similarly, the hand-vectorized code and the TPP based code outperform the baseline by 1.3×\times and 1.5×\times. We also experimented with gcc and the fast-math flag, and it just matched baseline performance. We want to emphasize that propagating the tensor sizes as compile-time constants throughout the operators is not practical for real use-cases within DL frameworks.
Figure [20](#S6.F20 "Figure 20 ‣ 6.1. Performance of standalone DL kernels ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Bottom shows similar performance speedups on ROME, where the TPP-based code is 1.6×\times faster than the auto-vectorized baseline.

Figure [19](#S6.F19 "Figure 19 ‣ 6.1. Performance of standalone DL kernels ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Bottom shows the performance of the BF16 split-SGD operator on CLX. This use-case represents a novel, mixed-precision operator where the compiler (even with compile-time constant tensor sizes) struggles to yield good performance; the TPP-based code has geo-mean speedup of 38×38\times over the compiler generated code.

![Refer to caption](/html/2104.05755/assets/x21.png)


Figure 21. Convolutions via BRGEMM TPP

Figure [21](#S6.F21 "Figure 21 ‣ 6.1. Performance of standalone DL kernels ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") illustrates the TPP-based implementation of various ResNet50 ([he2016deep,](#bib.bib33) ) Convolution layers across all available platforms. The minibatch size used on each platform equals to the number of the corresponding cores. It is noteworthy that the TPP-user code is identical for all targets, hence truly portable; it is merely that the TPP backend optimizes the code generation (BRGEMM in this case) in a platform/ISA-aware fashion. The geomean efficiencies of these convolutions are: 69%percent6969\% for BDX, 72%percent7272\% for CLX, 72%percent7272\% for CPX, 77%percent7777\% for CPX with BF16 datatype, 70%percent7070\% for ICX, 78%percent7878\% for ROME, 81%percent8181\% for Graviton2 and 52%percent5252\% for A64FX.
Previous work ([georganas2020harnessing,](#bib.bib21) ) also showed on an x86 TPP-predecessor that BRGEMM-based convolutions matched or outperformed Intel’s oneDNN library ([onednn,](#bib.bib13) ). Fujitsu recently contributed an A64FX back-end to oneDNN ([onednn-fujitsu,](#bib.bib45) ) and our TPP implementation outperforms this by 22% on the geomean. We observe that our TPP convolutions not only run on all of these different platforms without a single line of code change, but they run at very similar hardware utilization.

### 6.2. Performance of end-to-end DL workloads

#### 6.2.1. 1D Dilated Convolutions and their application to Computational Biology

![Refer to caption](/html/2104.05755/assets/x22.png)


Figure 22. 1D Dilated Convolutions

Here we evaluate the oneDNN ([onednn,](#bib.bib13) ) and TPP-based 1D dilated convolution layer of ATACworks ([Lal829481,](#bib.bib38) ) which takes takes more than 90% of the training time, and it has input tensor width (W𝑊W) of 60400, output tensor width (Q𝑄Q) of 60000, 15 input channels (C𝐶C), 15 filters (K𝐾K), filter size (S𝑆S) of 51, and dilation (d𝑑d) of 8. Figure [22](#S6.F22 "Figure 22 ‣ 6.2.1. 1D Dilated Convolutions and their application to Computational Biology ‣ 6.2. Performance of end-to-end DL workloads ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Top shows the computational efficiency results of the 1D convolution layer. oneDNN is not reaching peak performance for these specialized convolutions, exhibiting 19.9% efficiency for the forward pass and only 4.1% for the backward pass on CLX. Our TPP-based implementation shows 74.3% and 55.7% efficiency for the corresponding training passes. We also highlight the performance portability of our TPP-based approach across all tested platforms. Finally, we show training time per epoch results for ATACworks in Figure [22](#S6.F22 "Figure 22 ‣ 6.2.1. 1D Dilated Convolutions and their application to Computational Biology ‣ 6.2. Performance of end-to-end DL workloads ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Bottom. The TPP-based kernels provide training time speedup of 6.91×\times on CLX when comparing to the oneDNN based implementation. We also show that by leveraging the BF16 FMA acceleration of the CPX platform we can further obtain 1.62×\times speedup compared to the FP32 implementation on the same platform. In total BF16 yields 12.6×\times speedup over the oneDNN baseline.

#### 6.2.2. Deep Learning Recommendation - DLRM

![Refer to caption](/html/2104.05755/assets/x23.png)


Figure 23. DLRM performance on a small config (blue bars) and on the MLPerf config (orange bars)

![Refer to caption](/html/2104.05755/assets/x24.png)


Figure 24. DLRM performance breakdown of small config on multiple platforms

Figure [23](#S6.F23 "Figure 23 ‣ 6.2.2. Deep Learning Recommendation - DLRM ‣ 6.2. Performance of end-to-end DL workloads ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Top shows the FP32 DLRM performance on CLX using two different configurations, namely small DLRM (blue bars) and MLPerf DLRM (orange basrs). We refer to previous work for the detailed specification of these configurations ([kalamkar2020optimizing,](#bib.bib37) ). We evaluated 4 different implementations of DLRM: i) the PyTorch reference implementation, ii) PyTorch reference + custom Embedding extension auto-vectorized by the compiler, iii) DLRM expressed entirely via TPPs, and iv) hand-vectorized Embedding extension + BRGEMM-TPP based MLPs ([kalamkar2020optimizing,](#bib.bib37) ). We conclude that the TPP-based implementation matches the performance of the State-Of-The-Art implementation which is hand-vectorized specifically for AVX512 targets; both of these optimized versions substantially outperform the PyTorch CPU reference implementation by up to 48×\times. Compared to the version with the custom, auto-vectorized variant the TPP-version is up to 4.4% faster.

Figure [23](#S6.F23 "Figure 23 ‣ 6.2.2. Deep Learning Recommendation - DLRM ‣ 6.2. Performance of end-to-end DL workloads ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Bottom shows the DLRM performance of our TPP-based implementation across multiple platforms and compute precisions. We want to highlight two aspects: First, we are able to run the same TPP-code without any change across all platforms, something that is not doable with the hand-vectorized SOTA variant (iv) (since it is not able to run on the AVX2-only BDX and ROME platforms, or on the Graviton2 platform with AArch64 ISA). Second, the TPP-based BF16 shows speedup up to 28% over the variant with auto-vectorized Embedding extension. The culprit here is the mixed precision operations like split-SGD where the compiler struggles to yield efficient code as shown in Section [6.1](#S6.SS1 "6.1. Performance of standalone DL kernels ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads").

Figure [24](#S6.F24 "Figure 24 ‣ 6.2.2. Deep Learning Recommendation - DLRM ‣ 6.2. Performance of end-to-end DL workloads ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") illustrates the performance breakdown of the small config on multiple platforms. The blue portions of the bars correspond to the time spent on the Embedding component, the orange parts reflect the MLP portion, and finally the yellow portions correspond to the remaining components of the DLRM workload. We observe that depending on the platform, the time spent on Embedding varies from 29-37% of the total execution time, the time spent on MLP is in the range of 33-56% of the total time, and the rest components account for 15-23% of the time. We can also observe the correlation of the MLP performance with the compute capabilities of each platform. For example, on CPX which has native BF16 FMA support, the BF16 MLPs are sped up by ∼similar-to\sim2×\times compared to the FP32 MLPs on the same platform. In regard to the time spent on the Embedding kernel which tends to be bandwidth bound, we observe correlation with the corresponding bandwidth capabilities of the machines.

#### 6.2.3. Natural Language Processing - BERT Large

![Refer to caption](/html/2104.05755/assets/x25.png)


Figure 25. BERT Large performance

Figure [25](#S6.F25 "Figure 25 ‣ 6.2.3. Natural Language Processing - BERT Large ‣ 6.2. Performance of end-to-end DL workloads ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Top shows end-to-end performance (in examples/second) on CLX for the BERT large SQuAD fine-tuning task in FP32, using a max sequence length of 384 and minibatch of 24. We observe that the TPP-based implementation (blue bar) matches the performance of the AVX512-hand-vectorized code/orange bar. At the same time, our implementation is 1.69×\times faster than the Reference Hugging Faces CPU reference code ([huggingfaces,](#bib.bib46) ) (green bar).

Figure [25](#S6.F25 "Figure 25 ‣ 6.2.3. Natural Language Processing - BERT Large ‣ 6.2. Performance of end-to-end DL workloads ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Bottom shows the performance of the reference Hugging Faces code (green bars) versus the TPP-based code (blue bars) across multiple platforms (x86 and AArch64/Graviton2) and compute precisions (FP32 for all platforms, and BF16 for the CPX platform). The TPP-based BERT shows speedups ranging from 1.5×\times to 8.5×\times over the Hugging Faces code. This result highlights the performance portability through the TPP abstractions. In regard to various compute precisions, we note that with minimal changes inside the fused operators to handle the VNNI tensor layout (required for BF16 GEMM/BRGEMM), and a couple of lines changes in the application code to enable BF16 training, we were able to realize 2×\times speed up using BF16 training on CPX (compared to FP32 training on CPX) with 28 cores, surpassing 40-core FP32-ICX performance by 37%.

![Refer to caption](/html/2104.05755/assets/x26.png)


Figure 26. BERT Large performance breakdown on multiple platforms.

In order shed light on where the benefits are coming from, we present in Figure [26](#S6.F26 "Figure 26 ‣ 6.2.3. Natural Language Processing - BERT Large ‣ 6.2. Performance of end-to-end DL workloads ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") the performance breakdown of the Hugging Faces reference code and the TPP-based implementation. In particular we focus on 4 components:

1. (1)

   *GEMM* which corresponds to the tensor contractions implemented via either the BRGEMM-TPP in the TPP implementation, or it leverages optimized GEMM routines within BLAS libraries in the Hugging Faces implementation (MKL for x86 platforms and OpenBLAS for AArch64/Graviton2).
2. (2)

   *Dropout* corresponding to the dropout layer in BERT, where the TPP-based implementation employs fast random number generation via xorshift algorithm.
3. (3)

   *GeLU* corresponding to the Gaussian Error Linear Unit activation function in BERT, where the TPP-based implementation leverages fast approximations as discussed in section [3.3.2](#S3.SS3.SSS2 "3.3.2. Approximations for non-linear TPP Activation Functions ‣ 3.3. Examples of Non-Trivial Non-GEMM TPPs ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads").
4. (4)

   *Others* capturing the remaining operators: Transpose, Layer-norm, softmax, bias addition, vnni-reformatting (in case of BF16 training), copy, add, scale, zero-kernel, reduce, optimizer. Note that all these operators map to either unary/binary/ternary TPPs (see section [2](#S2 "2. The TPP Specification ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")) or the can be expressed via Matrix Equation TPPs (see section [5](#S5 "5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")).

![Refer to caption](/html/2104.05755/assets/x27.png)


Figure 27. BERT GEMM/tensor contraction efficiencies via the BRGEMM-TPP on multiple platforms.

First, we note that for the Intel x86 platforms (left part of the breakdown plot) the tensor contractions show speedups over the highly-optimized MKL GEMM implementation in Hugging Faces in the range of 2-6%. On the right side of the breakdown plot we observe that the BRGEMM-TPP benefits are even larger on the non-Intel platforms. More specifically, on AMD Rome (AVX2 x86 platform) the tensor contractions are sped up by 1.9×\times via the BRGEMM-TPP, and on Graviton2 (Arm AArch64 platform) the tensors contractions are 5.7×\times faster via the BRGEMM-TPP compared to the implementation relying on OpenBLAS GEMM calls. To further highlight the performance portability of the tensor contractions via the BRGEMM-TPP across multiple platforms and precisions, Figure [27](#S6.F27 "Figure 27 ‣ 6.2.3. Natural Language Processing - BERT Large ‣ 6.2. Performance of end-to-end DL workloads ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") shows the achieved GEMM performance (Left axis) on each platform for the entire training process (blue bars), whereas the orange line (Right axis) dictates the % of machine peak. The conclusion here is that the BRGEMM-TPP delivers high-efficiency for the corresponding tensor contractions in the range of 66-84% for all tested ISAs and micro-architectures.

The second conclusion we can draw from the performance breakdown in Figure [26](#S6.F26 "Figure 26 ‣ 6.2.3. Natural Language Processing - BERT Large ‣ 6.2. Performance of end-to-end DL workloads ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") is that our fused/dataflow TPP implementation outlined in section [5.2.3](#S5.SS2.SSS3 "5.2.3. Natural Language Processing - BERT ‣ 5.2. TPP-based Workloads ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") makes the dropout and GeLU times shrink substantially, offering speedups in the range of 10-360×\times. The BERT implementation via the dropout/GeLU TPPs in tandem to the BRGEMM TPPs take advantage of temporal locality, and virtually make the corresponding times disappear from the overall execution time. Last but not least, the remaining components are sped-up in the TPP-based implementation by 2.5-14×\times depending on the platform. As a result of these optimizations, the TPP-based BERT implementation spends the majority of the time (75.5-88.8%) in tensor contractions which are executed at high-efficiency as Figure [27](#S6.F27 "Figure 27 ‣ 6.2.3. Natural Language Processing - BERT Large ‣ 6.2. Performance of end-to-end DL workloads ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") shows.

#### 6.2.4. Emerging AI - Graph Neural Networks

![Refer to caption](/html/2104.05755/assets/x28.png)


Figure 28. GNN performance of GaphSAGE Full-batch training for OGB-Products

Figure [28](#S6.F28 "Figure 28 ‣ 6.2.4. Emerging AI - Graph Neural Networks ‣ 6.2. Performance of end-to-end DL workloads ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Top shows end-to-end performance (in seconds/epoch, so lower is better) on CLX for the full-batch training of the GraphSAGE workload on OGB-Products with FP32 and BF16 precision. For the CLX BF16 experiments, since CLX doesn’t have native support for BF16 FMAs, we use bit-wise accurate emulated-BF16 BRGEMM TPPs (see section [3.2.2](#S3.SS2.SSS2 "3.2.2. Mixed Precision BRGEMM and its emulation ‣ 3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")), and we still expect savings due to the bandwidth reduction in the non-GEMM parts, e.g., graph traversal and edge/node aggregation. We observe that the TPP-based implementation outperforms the DGL with Xbyak JIT backend baseline version by 2.65×\times. The TPP-BF16 version yields another 1.66×\times speedup over the TPP-FP32 variant mainly due to reduced bandwidth requirements.

Figure [28](#S6.F28 "Figure 28 ‣ 6.2.4. Emerging AI - Graph Neural Networks ‣ 6.2. Performance of end-to-end DL workloads ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")-Bottom shows the performance of the TPP-based code across multiple platforms (x86 and Arm AArch64) and compute precisions (FP32 and BF16). The relative differences in the performance can be justified by the different compute/bandwidth specs of the benchmarked platforms. We highlight that with minimal changes in the MLP portion to handle VNNI layout required for BF16 BRGEMM, and a couple of lines changes in the application code to enable BF16 training, we were able to realize 1.94×\times speed up using BF16 training on CPX with 28 cores compared to the FP32 training on the same platform.

![Refer to caption](/html/2104.05755/assets/x29.png)


Figure 29. GNN performance breakdown of GaphSAGE Full-batch training for OGB-Products

In order to further analyze the behavior of the various implementations on multiple platforms, we present on Figure [29](#S6.F29 "Figure 29 ‣ 6.2.4. Emerging AI - Graph Neural Networks ‣ 6.2. Performance of end-to-end DL workloads ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") the relevant performance breakdown. The very left bar shows the performance breakdown of the FP32 optimized DGL implementation that leverages JITed kernels through Xbyak on the CLX platform. The blue part corresponds to the Aggregation kernel described in subsection [5.2.4](#S5.SS2.SSS4 "5.2.4. Emerging AI - Graph Neural Networks ‣ 5.2. TPP-based Workloads ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") whereas the orange portion represents the time required by the remaining kernels, namely Multilayer-Peceptrons with Activation functions. In the DGL implementation the activation functions are not fused within the MLP’s tensor contractions. We observe that in this optimized DGL implementation, 82.3% is spent on the Aggregation kernel and only 17.7% is spent on the MLPs. On the second from the left bar (annotated as CLX-FP32) we show the performance of the FP32 TPP-based implementation on the same CLX platform. We conclude that the TPP-based Aggregation kernel exhibits a speedup of 3.29×\times compared to the DGL-Xbyak implementation, and the TPP-based MLP kernels (BRGEMM-TPP tensor contractions with *fused* TPP activation functions) exhibit a speedup of 1.4×\times compared to the respective DGL-Xbyak implementation. The FP32 TPP-based implementation spends 66.4% on the aggregation kernel and 33.6% on the fused MLP kernels.

The last 8 bars on Figure [29](#S6.F29 "Figure 29 ‣ 6.2.4. Emerging AI - Graph Neural Networks ‣ 6.2. Performance of end-to-end DL workloads ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") illustrate the performance breakdown of the TPP-based implementation on various platforms (CLX/BDX/ROME/ICX/GRAVITON2/CPX) and various precisions (FP32 and CPX-BF16). We want to emphasize that all these performance numbers are obtained by employing a the same exact TPP-based code (which is *platform-agnostic*); the only modification is pertaining to the BF16 TPP code where we changed the tensor layouts in the MLP portion in order to deal with the required VNNI format. When comparing the CPX-F32 and the CPX-BF16 performance breakdowns we observe a 2×\times speedup on the Aggregation kernel. This kernel is typically bandwidth bound due to its irregular/indexed accesses, and the BF16 TPP code moves half of the data compared to the FP32 TPP code since all the tensors are halved in size (BF16 vs FP32 datatype). The MLP portion of the TPP-based implementation is sped up by 1.73×\times by using the BF16 BRGEMM-TPP. The CPX platform supports the BF16 FMA instruction which has effectively 2×\times the compute throughput compared to the FP32 FMA on the same platform. The BF16 BRGEMM-TPP internally leverages this BF16 FMA instruction within the GEMM microkernel on CPX (see subsection [3.2](#S3.SS2 "3.2. The BRGEMM TPP Implementation ‣ 3. TPP Implementation ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")) to speed up the tensor contraction. Finally, we highlight here the speedup of the Aggregation kernel when e.g. comparing the CPX and the ICX FP32 TPP-based performance numbers. The ICX platform has STREAM bandwidth of 175 GB/s whereas CPX has 97.7 GB/s, and this trend is reflected also in the performance of the Aggregation kernel (1.54×\times faster on ICX than CPX).

### 6.3. Distributed-memory scaling of DL workloads

![Refer to caption](/html/2104.05755/assets/x30.png)


Figure 30. Distributed-memory scaling of workloads

Even though we focused on the evaluation of the TPP-based workloads on a single node, our approach is seamlessly incorporated into the DL frameworks, hence we can scale to multiple nodes in a cluster to accelerate the training process employing the oneCCL library ([oneccl,](#bib.bib47) ). Figure [30](#S6.F30 "Figure 30 ‣ 6.3. Distributed-memory scaling of DL workloads ‣ 6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") shows the distributed-memory scaling of the TPP-based workloads. DLRM and BERT show almost perfect weak-scaling from 1 to 64 sockets of CLX (32 nodes) with speedups 51.7×\times and 57.9×\times respectively. Regarding the scaling of the GNN workload, the efficiency is directly affected by the quality of the partitions produced by the graph partitioning tools. Using 64 sockets we achieve 10×\times speedup compared to single socket, and further scaling improvements constitute future work. We can conclude that TPPs for single node optimizations combined with small-size cluster level execution can accelerate deep learning training on CPUs by up to two orders of magnitude.

## 7. TPP within MLIR and a Tensor Compiler

![Refer to caption](/html/2104.05755/assets/x31.png)


Figure 31. Example lowering paths within the PlaidML Tensor compiler in order to achieve full network optimization from popular frameworks. The green boxes represent the DL frameworks, the blue boxes correspond to MLIR dialects, the brown box shows the TPP-MLIR dialect within the stack, and the purple box represents the targeted platforms.

In order to illustrate the viability of TPPs as a virtual Tensor ISA within MLIR and Tensor Compilers, we implemented a rudimentary MLIR dialect corresponding to the TPPs. We also implemented lowering passes within the PlaidML ([plaidml,](#bib.bib15) ) Tensor Compiler that transform intermediate MLIR representations to the TPP-MLIR dialect. The TPP-MLIR dialect is subsequently lowered to the corresponding LIBXSMM TPP calls, therefore such a flow is not relying on LLVM for the code generation of the corresponding tensor operations.

The current lowering path through MLIR supports a variety of front-end interfaces with LinAlg or Tile as the lowest level common entry points, i.e. the lowest level of abstraction that inbound programs can be specified in such that they will be subject to the full range of optimizations necessary to achieve full performance. Figure [31](#S7.F31 "Figure 31 ‣ 7. TPP within MLIR and a Tensor Compiler ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") details the lowering paths currently implemented in PlaidML and where key transforms map tensor operations into the TPP dialect. The key transformation is located in the stencil pass of the PXA dialect (Parallel eXtensions for Affine - a staging ground for PlaidML/TPP work that will be proposed upstream to the affine dialect). Operations that cannot be matched to TPP primitives are lowered through standard affine optimization pipelines.

![Refer to caption](/html/2104.05755/assets/x32.png)


Figure 32. FP32 inference with PlaidML on various workloads: ResNet-152, ResNext-50, and I3D-Kinetics-400.

We experimented with the use-case of FP32 inference on a client CPU (Intel i7-6700) on three different workloads: ResNet-152 ([he2016deep,](#bib.bib33) ), ResNext-50 ([xie2017aggregated,](#bib.bib48) ), and I3D-Kinetics-400 ([carreira2017quo,](#bib.bib49) ). Figure [32](#S7.F32 "Figure 32 ‣ 7. TPP within MLIR and a Tensor Compiler ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") shows the results of three implementations: i) The green bars show the performance of the code generated by PlaidML with MLIR for intermediate representations, and LLVM for the code generation, ii) The orange bars show the performance of the code generated by PlaidML with MLIR for intermediate representations, and the TPP-MLIR dialect as virtual Tensor ISA for the code generation of the corresponding tensor contractions, and iii) TensorFlow FP32 inference backed-up by the vendor-optimized oneDNN library. We observe that the Tensor Compiler variant which relies on the TPP-MLIR dialect for the tensor contractions outperforms the variant which relies exclusively on LLVM (for loop-tiling and vectorization) up to  35.6×\times. At the same time, PlaidML assisted by the TPP-MLIR dialect matches/outperforms the performance of TensorFlow which uses internally oneDNN, a highly-tuned vendor library for this CPU target. These preliminary results highlight the viability of the synergistic Tensor Compiler - TPP paradigm as discussed in Section [1](#S1 "1. Introduction ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads").

## 8. TPP and HPC Applications

So far, in this paper the focus was on how the TPP abstraction can be leveraged within the Deep Learning Domain. Tensor computations are ubiquitous, and in particular they constitute the cornerstone of many HPC applications. As such, the TPP abstraction can be readily employed by HPC applications to accelerate tensor computations without sacrificing portability. In the rest of this section we examine how TPPs are used within two HPC applications, namely CP2K and EDGE.

### 8.1. CP2K

The tensor based formulation originated and became common in physics, and it is well adopted in the field of engineering or applied sciences, and in electronic structure (ES) theory in particular. CP2K is an open source ES- and MD-package (molecular dynamics) for atomistic simulations of solid-state, liquid, molecular, and biological systems ([doi:10.1063/5.0007045,](#bib.bib50) ). CP2K is striving for good performance on HPC and massively parallel systems. Even though the use of novel algorithms in CP2K is the norm for scientific reasons, implementations have not widely tapped tensors in an explicit fashion. In contrast, Machine Learning emerged with similar, yet not coherent APIs and frameworks around the notions of tensors, layers and image processing.

While ES calculations can be formulated with tensors of ranks two to four, CP2K (and similar packages) largely remain with matrix based formulation. Various libraries for tensor contractions gained some attraction for scientific applications but the level of generality is key, e.g., as sparse representations are desired. CP2K explored an API for sparse tensor contractions and published a proof of concept implementation built into the DBCSR library ([DBLP:journals/corr/abs-1910-13555,](#bib.bib51) ). Efforts targeting accelerators in CP2K, namely GPUs, are not fully booked hence hardware specifically for Deep Learning (with focus on low and mixed precision arithmetic) is not yet a motivation of tensors as an implementation vehicle (and source of acceleration). Therefore a collection of primitives such as TPP is well-suited for an emerging discussion of a more general API.

CP2K 3.0 introduced LIBXSMM for Small Matrix Multiplications (SMMs). CP2K and DBCSR (previously part of CP2K’s code base) since then additionally introduced element-wise operations (copy and transpose) with ”elements” being small matrices based on LIBXSMM. Reformulating existing code to build on (batched) GEMM TPP and element-wise TPP operations is an established pattern for increased performance in CP2K.

To practically improve performance in CP2K one has to consider:

* •

  Fusing kernels and increasing arithmetic intensity independent of the target being a CPU or an accelerator (performance bound by memory bandwidth).
* •

  Specializing code at runtime based on workload/input of the application, e.g., generating code Just-In-Time (JIT) a.k.a. meta-programming.

These objectives can be delivered by TPPs as a domain-specific language (DSL), enabling the scientist to write more abstract code, e.g., by the means of meta-programming, and by relying on a specification which delivers versatile primitives deferring low-level optimizations to the TPP backend.

For CP2K’s performance evaluation, we refer to BDX, CLX, ICX, and ROME as introduced earlier (section [6](#S6 "6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")). To show the portability of our approach, we augmented our results by using the Oracle Cloud Infrastructure, namely the result for Altra processor (BM.Standard.A1.160 OCI shape). Table [4](#S8.T4 "Table 4 ‣ 8.1. CP2K ‣ 8. TPP and HPC Applications ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") shows the performance benefit of LIBXSMM’s GEMM-TPP in CP2K when compared to Intel’s MKL GEMM routines.

Table 4. CP2K performance (Cases/Day) of three workloads fitting into single systems with two processors. Single-socket performance is reported here for consistency within this paper. Intel MKL or OpenBLAS are always used for general BLAS operations including large GEMMs. Either (BLAS-)GEMM or TPP-GEMM was used for batched multiplication of small matrices (SMMs). Workloads utilizing CP2K’s DBCSR library for distributed block-sparse matrix multiply benefit from (runtime-)specialized GEMM-TPP kernels where the set of matrix shapes is not known at compile-time of the application or depends on the workload in general.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| System | Workloada | BLAS-GEMMb | TPP-GEMMc | TPP-Speedup |
| BDX | H2O-256 | 919191 | 101101101 | 11%percent1111\% |
|  | H2O-512 | 232323 | 272727 | 17%percent1717\% |
| CLX | H2O-256 | 154154154 | 162162162 | 5%percent55\% |
|  | H2O-512 | 393939 | 414141 | 5%percent55\% |
|  | H2O-DFT-LS4 | 454545 | 474747 | 4%percent44\% |
| ICX | H2O-256 | 235235235 | 249249249 | 6%percent66\% |
|  | H2O-512 | 606060 | 656565 | 8%percent88\% |
|  | H2O-DFT-LS4 | 676767 | 707070 | 4%percent44\% |
| ROME | H2O-256 | 225225225 | 244244244 | 8%percent88\% |
|  | H2O-512 | 555555 | 575757 | 4%percent44\% |
|  | H2O-DFT-LS4 | 656565 | 656565 | 0%percent00\% |
| Altra | H2O-256 | 228228228 | 236236236 | 4%percent44\% |
|  | H2O-512 | 606060 | 626262 | 3%percent33\% |
|  | H2O-DFT-LS4 | 606060 | 666666 | 10%percent1010\% |
| aH2O-256 (CP2K bench.), H2O-512 (UEABS Case A) and H2O-DFT-LS NREP=4 (UEABS Case C) from [PRACE UEABS](https://prace-ri.eu/training-support/technical-documentation/benchmark-suites/) 2.1. | | | | |
| --- | --- | --- | --- | --- |
| bIntel MKL (x86-64) or OpenBLAS (otherwise). | | | | |
| cLIBXSMM. | | | | |

### 8.2. EDGE

The Extreme-Scale Discontinuous Galerkin Environment (EDGE) uses the Arbitrary high-order DERivatives (ADER) Discontinuous Galerkin (DG) finite element method to simulate seismic wave propagation ([10.1007/978-3-319-58667-0\_3,](#bib.bib52) ).
The software uses unstructured tetrahedral meshes which are typically adapted to the used seismic velocity models.
Additionally, modelers may introduce mountain topography.
A sophisticated local time stepping scheme allows the solver to operate efficiently in very large and complex settings.
The software is able to fuse multiple ensemble simulations into one execution of the software.
EDGE uses an orthogonal polynomial expansion basis to discretize each of the considered variables in a tetrahedron of the mesh.
In a typical setting, we use three relaxation mechanisms for the viscoelastic part, resulting in a total of 27 seismic variables.
Additionally using a fifth order method gives us 35 basis functions, resulting in a total of 27⋅35=945⋅273594527\cdot 35=945 degrees of freedom per tetrahedral element.
The solver advances the degrees of freedom in time by repeatedly computing a triplet of quadrature-free integrators.
While the actual integrators are part of EDGE, their implementation relies heavily on TPPs.
The GEMM-TPP with small and uncommon matrix sizes is the most crucial operation required by EDGE.
For example, the surface integrator requires the multiplication of a 9×359359\times 35 matrix with a 35×15351535\times 15 matrix.
The solver’s extension with additional, performance-portable TPPs in all parts of the integrators is work-in-progress.
Especially, EDGE’s support for viscoelastic attenuation or local time stepping requires “simpler“ kernels, e.g., the unary TPPs Identity and Zero, or the binary TPPs Mul, Sub and Add.

We evaluate EDGE’s performance-portability through the use of TPPs by studying the performance of a full setup of the Layer Over Halfspace 3 (LOH3) benchmark with 743,066 tetrahedral elements.
The same setting was also used in ([breuer2021nextgen,](#bib.bib53) ) to study the performance of the solver on a single processor of the Frontera supercomputer located at the Texas Advanced Computing Center (position ten in the 06/21 TOP500-list).
Following this study, a sophisticated simulation of the 2014 MwsubscriptM𝑤\text{M}\_{w} 5.1 La Habra earthquake using a mesh with 237,861,634 tetrahedral elements and EDGE’s advanced features yielded a performance of 2.20 FP32-PFLOPS on 1,536 nodes.

Table 5. Sustained 32-bit floating point performance on the studied systems. The performance is given in TFLOPS. Results are presented for Global Time Stepping (GTS) and Local Time Stepping (LTS) when using single and fused forward simulations.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| System | GTS | | LTS | |
|  | single | fuseda | single | fuseda |
| Cascade Lake | 1.08 | 0.78 | 1.02 | 0.74 |
| Ice Lake | 1.29 | 1.01 | 1.23 | 0.96 |
| Rome | 1.20 | 1.08 | 1.12 | 1.01 |
| Milan | 1.39 | 1.16 | 1.29 | 1.07 |
| Altra | 1.27 | 0.73 | 1.51 | 0.76 |
| aEDGE’s fused simulations use sparse matrix kernels. | | | | |

For the EDGE application, we study the software’s raw floating point performance and time-to-solution by extending our LOH3-Frontera-only study ([breuer2021nextgen,](#bib.bib53) ) with diverse processors:

* •

  Cascade Lake (similar to CLX as introduced in section [6](#S6 "6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")): 2.7 GHz 28-core Intel Xeon Platinum 8280 processor of the Frontera system at the Texas Advanced Computing Center. We only used a single 28-core processor of Frontera’s dual-socketed compute nodes in our tests.
* •

  Ice Lake: 2.3 GHz 40-core Intel Xeon Platinum 8380 processor on Intel’s on-premises cluster. We only used a single 40-core processor of the dual-socket compute nodes in our tests.
* •

  Rome (similar to ROME as introduced in section [6](#S6 "6. Experimental Results of DL kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")): 2.25 GHz AMD EPYC 7742 (BM.Standard.E3.128 OCI shape). We only used a single 64-core processor of the bare metal instance in our tests.
* •

  Milan: 2.55 GHz AMD EPYC 7J13 (BM.Standard.E4.128 OCI shape). We only used single 64-core processor of the bare metal instance in our tests.
* •

  Altra: 3.0 GHz Ampere Altra Q80-30 processor (BM.Standard.A1.160 OCI shape). We only used a single 80-Armv8.2-core processor of the bare metal instance in our tests.

Table [5](#S8.T5 "Table 5 ‣ 8.2. EDGE ‣ 8. TPP and HPC Applications ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") shows the sustained floating point performance of the conducted runs.
All numbers are given in FP32-TFLOPS.
Columns two and three present the performance of Global Time Stepping (GTS), whereas columns four and five show that of Local Time Stepping (LTS).
In general, the LTS configurations have a slightly lower peak utilization when compared to their GTS counterparts.
Note, however, that Table [5](#S8.T5 "Table 5 ‣ 8.2. EDGE ‣ 8. TPP and HPC Applications ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") only shows raw floating point performance and does not account for time-to-solution speedups through LTS (theoretically up to 2.67×\times in this case).
The performance of GTS and LTS is further split into running a single forward simulation and fusing multiple simulations.
In fused mode, the solver parallelizes over the right-hand-side by concurrently simulating seismic wave propagation for a collection of seismic sources.
One of the fused mode’s unique advantages is the opportunity for perfect vectorization of all small matrix multiplications, even when considering sparsity ([10.1007/978-3-319-58667-0\_3,](#bib.bib52) ).
In this work we matched the microarchitectures’ SIMD-length by fusing 16 simulations on Cascade Lake and Ice Lake, eight simulations on Rome and Milan, and four simulations on Altra.
Once again, note that Table [5](#S8.T5 "Table 5 ‣ 8.2. EDGE ‣ 8. TPP and HPC Applications ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") does not include the respective sparsity-driven 2.49×2.49\times increase of the floating point operations’ value when running fused simulations.
Comparing the performance of the different systems, we observe very high overall performance with architectural efficiency gains originating from decreasing SIMD-lengths.
This is especially noticeable when running single forward simulations.
In this case, the vectorized dimension of the small dense matrix kernels coincides with the number of basis functions, i.e., M=35𝑀35M=35, which is challenging when optimizing for AVX512 (Cascade Lake and Ice Lake) and AVX2 (Rome and Milan).
The short 128-bit ASIMD vector instruction (Altra) reach a very high peak utilization of 33.2% for GTS and 39.2% in LTS.
For the fused simulations, the differences in relative peak utilization narrow further.

Table 6. Time-to-solution speedups of the studied systems when using different configurations of the solver EDGE. The performance of the Cascade Lake system, running EDGE with Global Time Stepping (GTS) and a single forward simulation, is used as baseline. In contrast to Table [5](#S8.T5 "Table 5 ‣ 8.2. EDGE ‣ 8. TPP and HPC Applications ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads"), the speedups include the higher algorithmic efficiencies of EDGE’s support for Local Time Stepping (LTS) and fused forward simulations.

| System | GTS | | LTS | |
| --- | --- | --- | --- | --- |
|  | single | fused | single | fused |
| Cascade Lake | 1.00 | 1.80 | 2.50 | 4.52 |
| Ice Lake | 1.19 | 2.33 | 3.02 | 5.87 |
| Rome | 1.11 | 2.48 | 2.76 | 6.17 |
| Milan | 1.28 | 2.67 | 3.18 | 6.55 |
| Altra | 1.18 | 1.69 | 3.71 | 4.64 |

Table [6](#S8.T6 "Table 6 ‣ 8.2. EDGE ‣ 8. TPP and HPC Applications ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads") describes the obtained performance numbers in terms of time-to-solution.
Here, we use the runtime of the studied LOH3 setting on Cascade Lake for GTS and a single forward simulation as baseline.
All other settings are given relative to this.
Further, for the fused settings, we consider the per-simulation time.
We observe that EDGE’s overall performance is driven by the high floating point performance through the use of TPPs and the solver’s advanced algorithmic features.
Here, Altra performs best for single forward simulations using LTS, accelerating the baseline by 3.71×\times.
Milan has the best time-to-solution in all other settings and is able to outperform the baseline by 6.55×\times when using LTS and fusing simulations.
This performance lead originates from Milan’s high theoretical peak combined with a high peak utilization (see Table [5](#S8.T5 "Table 5 ‣ 8.2. EDGE ‣ 8. TPP and HPC Applications ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")).

## 9. Related Work

The related work in terms of the development methodology of DL workloads has been referenced in the introduction, so here we mention community efforts that share the same design philosophy with TPPs. Tensor Operator Set Architecture (TOSA) is a recent work, concurrently developed with TPPs, that provides a set of whole-tensor operations commonly employed in DL ([tosa,](#bib.bib54) ). TOSA allows users to express directly operators on up to 4D/5D tensors which are not naturally mapped even on contemporary 2D systolic hardware. We believe that staying at the 2D primitive level is expressive and sufficient, as we can build higher-order ops with loops around 2D operators, e.g. see Algorithm [6](#alg6 "Algorithm 6 ‣ 5.1.6. Multi-Layer Perceptron (MLP) kernel ‣ 5.1. TPP-based Kernels ‣ 5. TPP-based Kernels & Workloads ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads"). Despite the similarities of TPP and TOSA specifications, the TOSA back-end is reference C code and is not showcased in full DL-workloads. CUTLASS ([cutlass,](#bib.bib55) ) and Triton ([tillet2019triton,](#bib.bib56) ) strive for high-performance on GPUs, while also offer flexible composition that can be easily applied to solve new problems related in DL and linear algebra, and share many design principles with TPPs. XLA ([xla,](#bib.bib57) ) is a domain-specific compiler for linear algebra and DL that targets TensorFlow models with potentially no source code changes. JAX ([jax,](#bib.bib58) ) provides automatic differentiation of Python and NumPy functions, and the compilation of the desired operators happens in a user-transparent way with JIT calls, yielding optimized XLA kernels. XLA and JAX share the same philosophy with TPPs: the user is focusing on the DL kernel/workload development using high-level, platform-agnostic, declarative-style programming, whereas the tensor-aware back-end infrastructure undertakes the efficient and portable code generation.

Tensor Compilers (TC) ([plaidml,](#bib.bib15) ; [chen2018tvm,](#bib.bib16) ; [vasilache2018tensor,](#bib.bib17) ; [zheng2020ansor,](#bib.bib18) ) attempt to optimize DL operators in a platform-agnostics way, however their applicability is restricted to relatively small code-blocks whereas full workload integration is cumbersome. Also, TC undertake the tasks of efficient parallelization, loop re-ordering, automatic tiling and layout transformations, nevertheless the obtained performance is typically underwhelming ([barham2019machine,](#bib.bib12) ). We envision that TPPs can be used as a tool by TC in order to attain efficient platform-specific code generation, therefore TC could focus on optimizing the higher level aspects of the tensor programs (e.g. layout transformations). Along these lines, TPPs fit in the MLIR ([mlir,](#bib.bib20) ) ecosystem/stack as a lowering dialect (see Section [7](#S7 "7. TPP within MLIR and a Tensor Compiler ‣ Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads")), and in this way the TPP back-end could be leveraged by multiple TC frameworks.

## 10. Conclusions And Future Work

In this work we presented the Tensor Processing Primitives (TPP), a compact, yet versatile set of 2D-tensor operators, which subsequently can be utilized as building-blocks to construct efficient, portable complex DL operators on high-dimensional tensors. We also show how TPPs can be used within HPC applications in order to accelerate tensor computations. We demonstrate the efficacy of our approach using standalone kernels and end-to-end training DL-workloads (CNNs, dilated convolutions, DLRM, BERT, GNNs) expressed entirely via TPPs that outperform state-of-the-art implementations on multiple platforms. As future work, we plan to create a full-fledged TPP-based MLIR dialect such that Tensor Compilers can leverage the strengths of TPPs . Also, we plan to further enrich the TPP back-end implementation by supporting more ISAs, including GPUs and POWER architectures.

## GLOSSARY

### Intel Pseudo Intrinsics

1. (1)

   *\_mm128*  Represents a vector of width 128 bits.
2. (2)

   *\_mm128\_loadu\_ps(addr)*  Loads 16byte of 32 bit elements.
3. (3)

   *\_mm128\_storeu\_ps(addr)*  Stores 16byte of 32 bit elements.
4. (4)

   *\_mm128\_unpacklo\_ps(A, B)*  Unpacks and interleaves 32 bit elements from the low half of A and B.
5. (5)

   *\_mm128\_unpackhi\_ps(A, B)*  Unpacks and interleaves 32 bit elements from the high half of A and B.
6. (6)

   *\_mm128\_unpacklo\_pd(A. B)*  Unpacks and interleaves 64 bit elements from the low half of A and B.
7. (7)

   *\_mm128\_unpackhi\_pd(A, B)*  Unpacks and interleaves 64 bit elements from the high half of A and B.
8. (8)

   *\_mm512*  Represents a vector of width 512 bits.
9. (9)

   *\_mm512\_permutexvar\_ps(A,B)*  Shuffle single precision floating point elements in 512 wide vector length using indexes specified in B.
10. (10)

    *\_mm512\_roundscale\_ps(A,B)*  Round single precision floating point elements to the rounding mode specified by argument B.
11. (11)

    *\_mm512\_sub\_ps(A,B)*  Subtract single precision floating point elements in A from B.
12. (12)

    *\_mm512\_scalef\_ps(A,B)*  Scales single precision floating point elements in A using values specified in B.
13. (13)

    *\_mm512\_range\_ps(A,B, int imm8)*  Calculates the min, max or absolute max for each single precision- floating point elements in A and B. Lower 2 bits of imm8[1:0] specifies the operation(min/max/absolute max) to be performed.
14. (14)

    *\_mm512\_xor\_ps(A,B)*  Performs XOR operation between each single precision floating point elements in A and B vector.
15. (15)

    *\_mm512\_and\_ps(A,B)*  Performs AND operation between each single precision floating point elements in A and B vector.
16. (16)

    *\_mm512\_rcp14\_ps(A,B)*  Calculates approximate reciprocal of each single precision floating point element in range less then 2-̂14.
17. (17)

    *\_mm512\_cmp\_ps\_mask(A,B,int C)*  Compare the single precision elements in A and B specified by the comparison mode in C.
18. (18)

    *\_mm512\_mask\_blend\_ps(mask A,B,C)*  Copies single precision floating point element from vector A in vector C if the corresponding mask bit is set .
19. (19)

    *\_mm512\_fmadd\_ps(mask A,B,C)*  Fused-Multiply-Add: Multiplies elements from vector A and B and adds them to elements of vector C.
20. (20)

    *\_mm512\_maskz\_loadu\_epi16(mask, addr)*  Loads 64byte of 16bit elements under zero masking from address addr.
21. (21)

    *\_mm512\_set1\_epi32( value )*  sets a 32 bit value into all 16 entries of the vector, e.g. broadcast.
22. (22)

    *\_mm512\_maskz\_mov\_epi16(mask, A)*  Moves 16 bit-type register A under zero-masking to a different register.
23. (23)

    *\_mm512\_slli\_epi32(A, imm)*  Shifts all entries in the vector registers (typed as 32 bit elements) by value imm to the left by shifting 0 in.

### Arm Pseudo Intrinsics

1. (1)

   *vld1q\_f32(addr)* Loads 16byte of 32 bit elements.
2. (2)

   *vst1q\_f32(addr)* Loads 16byte of 32 bit elements.
3. (3)

   *vtrn1q\_f32(A, B)*  Unpacks and interleaves 32 bit elements from the low half of A and B.
4. (4)

   *vtrn2q\_f32(A, B)*  Unpacks and interleaves 32 bit elements from the high half of A and B.
5. (5)

   *vtrn1q\_f64(A. B)*  Unpacks and interleaves 64 bit elements from the low half of A and B.
6. (6)

   *vtrn2q\_f64(A, B)*  Unpacks and interleaves 64 bit elements from the high half of A and B.
7. (7)

   *vmax\_q(A,B)*  Calculates the maximum between each single precision floating point elements in A and B vector.
8. (8)

   *vmin\_q(A,B)*  Calculates the minimum between each single precision floating point elements in A and B vector.
9. (9)

   *vmul\_q(A,B)*  Multiply single precision elements in A and B vector.
10. (10)

    *vsub\_q(A,B)*  Subtract corresponding single precision elements in B from A.
11. (11)

    *vadd\_q(A,B)*  Add single precision elements in B and A.
12. (12)

    *vshlq\_u32(A,B)*  Shift left each single precision elements in A by the value specified in B.
13. (13)

    *vrndmq\_f32(A)*  Round single precision floating point elements in A using minus infinity rounding mode.
14. (14)

    *vcvtmq\_s32\_f32(A)*  Converts single precision floating point elements in A to signed integers using minus infinity rounding mode.
15. (15)

    *float32x4\_t*  Represents 4 single precision floating point elements in vector width of 128.
16. (16)

    *vand\_q(A,B)*  Performs bit-wise AND operation between A and B vector.
17. (17)

    *vfmaq\_f32(A,B,C)*  Multiply single precision elements in A and B.Add the intermediate result to C.
18. (18)

    *vld1q\_f32(A)*  Load a single precision element from scalar to all single precision element in a vector.
19. (19)

    *vtbl1\_u8(A,B)*  Performs a byte look up operation in vector A using byte addressable indexes specified in vector B.
20. (20)

    *vtbl4\_u8(A,B)*  Performs a 64 byte look up operation in vector A, A+1, A+2, A+3 using byte addressable indexes specified in vector B.
21. (21)

    *vbcaxq\_s32(A,B)*  Performs XOR operation between each single precision floating point elements in A and B vector.
22. (22)

    *vcgt\_q(A,B)*  Compare corresponding single precision elements in A and B. If B is greater then A the corresponding bits are set in the destination vector.
23. (23)

    *vrecpe\_f32(A)*  Calculates approximate reciprocal of each single precision floating point element in vector A.
24. (24)

    *vbit\_insert(A,B)*  Copies single precision floating point element from vector A in destination vector if the corresponding bits are set in vector B.

## References

* [1]

  Alex Krizhevsky, I. Sutskever, and G.E. Hinton.
  Image classification with deep convolutional neural networks.
  Advances in neural information processing systems, pages
  1097–1105, 2012.
* [2]

  Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir
  Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich.
  Going deeper with convolutions.
  In Proceedings of the IEEE conference on computer vision and
  pattern recognition, pages 1–9, 2015.
* [3]

  Karen Simonyan and Andrew Zisserman.
  Very deep convolutional networks for large-scale image recognition.
  arXiv preprint arXiv:1409.1556, 2014.
* [4]

  Dong Yu, Michael L Seltzer, Jinyu Li, Jui-Ting Huang, and Frank Seide.
  Feature learning in deep neural networks-studies on speech
  recognition tasks.
  arXiv preprint arXiv:1301.3605, 2013.
* [5]

  Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang
  Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al.
  Google’s neural machine translation system: Bridging the gap between
  human and machine translation.
  arXiv preprint arXiv:1609.08144, 2016.
* [6]

  Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra,
  Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al.
  Wide & deep learning for recommender systems.
  In Proceedings of the 1st Workshop on Deep Learning for
  Recommender Systems, pages 7–10. ACM, 2016.
* [7]

  Thomas Wolf, Julien Chaumond, Lysandre Debut, Victor Sanh, Clement Delangue,
  Anthony Moi, Pierric Cistac, Morgan Funtowicz, Joe Davison, Sam Shleifer,
  et al.
  Transformers: State-of-the-art natural language processing.
  In Proceedings of the 2020 Conference on Empirical Methods in
  Natural Language Processing: System Demonstrations, pages 38–45, 2020.
* [8]

  Erik Gawehn, Jan A Hiss, and Gisbert Schneider.
  Deep learning in drug discovery.
  Molecular informatics, 35(1):3–14, 2016.
* [9]

  Garrett B Goh, Nathan O Hodas, and Abhinav Vishnu.
  Deep learning for computational chemistry.
  Journal of computational chemistry, 38(16):1291–1307, 2017.
* [10]

  Maithra Raghu and Eric Schmidt.
  A survey of deep learning for scientific discovery.
  arXiv preprint arXiv:2003.11755, 2020.
* [11]

  Md Zahangir Alom, Tarek M Taha, Chris Yakopcic, Stefan Westberg, Paheding
  Sidike, Mst Shamima Nasrin, Mahmudul Hasan, Brian C Van Essen, Abdul AS
  Awwal, and Vijayan K Asari.
  A state-of-the-art survey on deep learning theory and architectures.
  Electronics, 8(3):292, 2019.
* [12]

  Paul Barham and Michael Isard.
  Machine learning systems are stuck in a rut.
  In Proceedings of the Workshop on Hot Topics in Operating
  Systems, pages 177–183, 2019.
* [13]

  oneDNN.
  Intel onednn, <https://github.com/oneapi-src/oneDNN>, Accessed on
  3/30/2021.
* [14]

  Sharan Chetlur, Cliff Woolley, Philippe Vandermersch, Jonathan Cohen, John
  Tran, Bryan Catanzaro, and Evan Shelhamer.
  cudnn: Efficient primitives for deep learning.
  arXiv preprint arXiv:1410.0759, 2014.
* [15]

  Tim Zerrell and Jeremy Bruestle.
  Stripe: Tensor compilation via the nested polyhedral model.
  arXiv preprint arXiv:1903.06498, 2019.
* [16]

  Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen
  Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, et al.
  {{\{TVM}}\}: An automated end-to-end optimizing compiler for deep
  learning.
  In 13th {{\{USENIX}}\} Symposium on Operating Systems Design and
  Implementation ({{\{OSDI}}\} 18), pages 578–594, 2018.
* [17]

  Nicolas Vasilache, Oleksandr Zinenko, Theodoros Theodoridis, Priya Goyal,
  Zachary DeVito, William S Moses, Sven Verdoolaege, Andrew Adams, and Albert
  Cohen.
  Tensor comprehensions: Framework-agnostic high-performance machine
  learning abstractions.
  arXiv preprint arXiv:1802.04730, 2018.
* [18]

  Lianmin Zheng, Chengfan Jia, Minmin Sun, Zhao Wu, Cody Hao Yu, Ameer Haj-Ali,
  Yida Wang, Jun Yang, Danyang Zhuo, Koushik Sen, et al.
  Ansor: Generating high-performance tensor programs for deep learning.
  In 14th {{\{USENIX}}\} Symposium on Operating Systems Design and
  Implementation ({{\{OSDI}}\} 20), pages 863–879, 2020.
* [19]

  Mingzhen Li, Yi Liu, Xiaoyan Liu, Qingxiao Sun, Xin You, Hailong Yang, Zhongzhi
  Luan, Lin Gan, Guangwen Yang, and Depei Qian.
  The deep learning compiler: A comprehensive survey.
  IEEE Transactions on Parallel and Distributed Systems,
  32(3):708–727, 2020.
* [20]

  MLIR.
  Multi-level intermediate representation,
  <https://github.com/tensorflow/mlir>, Accessed on 3/30/2021.
* [21]

  Evangelos Georganas, Kunal Banerjee, Dhiraj Kalamkar, Sasikanth Avancha, Anand
  Venkat, Michael Anderson, Greg Henry, Hans Pabst, and Alexander Heinecke.
  Harnessing deep learning via a single building block.
  In 2020 IEEE International Parallel and Distributed Processing
  Symposium (IPDPS), pages 222–233. IEEE, 2020.
* [22]

  Alexander Heinecke, Greg Henry, Maxwell Hutchinson, and Hans Pabst.
  LIBXSMM: Accelerating small matrix multiplications by runtime code
  generation.
  In Proceedings of the International Conference for High
  Performance Computing, Networking, Storage and Analysis, SC ’16, pages
  84:1–84:11, Piscataway, NJ, USA, 2016. IEEE Press.
* [23]

  Evangelos Georganas, Sasikanth Avancha, Kunal Banerjee, Dhiraj Kalamkar, Greg
  Henry, Hans Pabst, and Alexander Heinecke.
  Anatomy of high-performance deep learning convolutions on simd
  architectures.
  In SC18: International Conference for High Performance
  Computing, Networking, Storage and Analysis, pages 830–841. IEEE, 2018.
* [24]

  Bfloat16.
  Using bfloat16 with tensorflow models,
  <https://cloud.google.com/tpu/docs/bfloat16>, Accessed on 4/3/2019.
* [25]

  George Marsaglia et al.
  Xorshift rngs.
  Journal of Statistical Software, 8(14):1–6, 2003.
* [26]

  Kunal Banerjee, Evangelos Georganas, Dhiraj D Kalamkar, Barukh Ziv, Eden Segal,
  Cristina Anderson, and Alexander Heinecke.
  Optimizing deep learning rnn topologies on intel architecture.
  Supercomputing Frontiers and Innovations, 6(3):64–85, 2019.
* [27]

  Intel-ISA.
  Intel architecture instruction set extensions and future features
  programming reference,
  <https://software.intel.com/content/dam/develop/public/us/en/documents/architecture-instruction-set-extensions-programming-reference.pdf>,
  Accessed on 3/30/2021.
* [28]

  Michael James David Powell.
  Approximation theory and methods.
  Cambridge university press, 1981.
* [29]

  Chebyshev-Polynomials.
  Chebyshev polynomials,
  <https://en.wikipedia.org/wiki/Chebyshev_polynomials>, Accessed:
  2021-09-26.
* [30]

  Philippe Flajolet, Jean-Claude Raoult, and Jean Vuillemin.
  The number of registers required for evaluating arithmetic
  expressions.
  Theoretical Computer Science, 9(1):99–125, 1979.
* [31]

  J Willard Gibbs.
  Elementary principles in statistical mechanics.
  Courier Corporation, 2014.
* [32]

  Sergey Ioffe and Christian Szegedy.
  Batch normalization: Accelerating deep network training by reducing
  internal covariate shift.
  In International conference on machine learning, pages
  448–456. PMLR, 2015.
* [33]

  Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
  Deep residual learning for image recognition.
  In Proceedings of the IEEE conference on computer vision and
  pattern recognition, pages 770–778, 2016.
* [34]

  Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton.
  Layer normalization.
  arXiv preprint arXiv:1607.06450, 2016.
* [35]

  Yuxin Wu and Kaiming He.
  Group normalization.
  In Proceedings of the European conference on computer vision
  (ECCV), pages 3–19, 2018.
* [36]

  Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky.
  Instance normalization: The missing ingredient for fast stylization.
  arXiv preprint arXiv:1607.08022, 2016.
* [37]

  Dhiraj Kalamkar, Evangelos Georganas, Sudarshan Srinivasan, Jianping Chen,
  Mikhail Shiryaev, and Alexander Heinecke.
  Optimizing deep learning recommender systems training on cpu cluster
  architectures.
  In SC20: International Conference for High Performance
  Computing, Networking, Storage and Analysis, pages 1–15. IEEE, 2020.
* [38]

  Avantika Lal, Zachary D. Chiang, Nikolai Yakovenko, Fabiana M. Duarte, Johnny
  Israeli, and Jason D. Buenrostro.
  Atacworks: A deep convolutional neural network toolkit for
  epigenomics.
  bioRxiv, 2019.
* [39]

  Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang, Narayanan
  Sundaraman, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
  Alisson G Azzolini, et al.
  Deep learning recommendation model for personalization and
  recommendation systems.
  arXiv preprint arXiv:1906.00091, 2019.
* [40]

  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.
  Bert: Pre-training of deep bidirectional transformers for language
  understanding.
  arXiv preprint arXiv:1810.04805, 2018.
* [41]

  Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue,
  Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe
  Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien
  Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest,
  and Alexander M. Rush.
  Transformers: State-of-the-art natural language processing.
  In Proceedings of the 2020 Conference on Empirical Methods in
  Natural Language Processing: System Demonstrations, pages 38–45, Online,
  October 2020. Association for Computational Linguistics.
* [42]

  Minjia Zhang, Samyam Rajbhandari, Wenhan Wang, and Yuxiong He.
  Deepcpu: Serving rnn-based deep learning models 10x faster.
  In 2018 {{\{USENIX}}\} Annual Technical Conference
  ({{\{USENIX}}\}{{\{ATC}}\} 18), pages 951–965, 2018.
* [43]

  William L Hamilton, Rex Ying, and Jure Leskovec.
  Inductive representation learning on large graphs.
  arXiv preprint arXiv:1706.02216, 2017.
* [44]

  Sasikanth Avancha, Vasimuddin Md, Sanchit Misra, and Ramanarayan Mohanty.
  Deep graph library optimizations for intel (r) x86 architecture.
  arXiv preprint arXiv:2007.06354, 2020.
* [45]

  oneDNN Fugaku.
  A deep dive into a deep learning library for the a64fx fugaku cpu -
  the development story in the developer’s own words,
  <https://blog.fltech.dev/entry/2020/11/19/fugaku-onednn-deep-dive-en>,
  Accessed on 4/9/2021.
* [46]

  Hugging-Faces.
  Hugging faces, <https://github.com/huggingface/transformers>,
  Accessed on 4/9/2021.
* [47]

  oneCCL.
  Intel oneccl, <https://github.com/oneapi-src/oneCCL>, Accessed on
  3/30/2021.
* [48]

  Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He.
  Aggregated residual transformations for deep neural networks.
  In Proceedings of the IEEE conference on computer vision and
  pattern recognition, pages 1492–1500, 2017.
* [49]

  Joao Carreira and Andrew Zisserman.
  Quo vadis, action recognition? a new model and the kinetics dataset.
  In proceedings of the IEEE Conference on Computer Vision and
  Pattern Recognition, pages 6299–6308, 2017.
* [50]

  Thomas D. Kühne, Marcella Iannuzzi, Mauro Del Ben, Vladimir V. Rybkin, Patrick
  Seewald, Frederick Stein, Teodoro Laino, Rustam Z. Khaliullin, Ole Schütt,
  Florian Schiffmann, Dorothea Golze, Jan Wilhelm, Sergey Chulkov,
  Mohammad Hossein Bani-Hashemian, Valéry Weber, Urban Borštnik, Mathieu
  Taillefumier, Alice Shoshana Jakobovits, Alfio Lazzaro, Hans Pabst, Tiziano
  Müller, Robert Schade, Manuel Guidon, Samuel Andermatt, Nico Holmberg,
  Gregory K. Schenter, Anna Hehn, Augustin Bussy, Fabian Belleflamme, Gloria
  Tabacchi, Andreas Glöß, Michael Lass, Iain Bethune, Christopher J. Mundy,
  Christian Plessl, Matt Watkins, Joost VandeVondele, Matthias Krack, and Jürg
  Hutter.
  Cp2k: An electronic structure and molecular dynamics software package
  - quickstep: Efficient and accurate electronic structure calculations.
  The Journal of Chemical Physics, 152(19):194103, 2020.
* [51]

  Ilia Sivkov, Patrick Seewald, Alfio Lazzaro, and Jürg Hutter.
  DBCSR: A blocked sparse tensor algebra library.
  CoRR, abs/1910.13555, 2019.
* [52]

  Alexander Breuer, Alexander Heinecke, and Yifeng Cui.
  Edge: Extreme scale fused seismic simulations with the discontinuous
  galerkin method.
  In Julian M. Kunkel, Rio Yokota, Pavan Balaji, and David Keyes,
  editors, High Performance Computing, pages 41–60, Cham, 2017. Springer
  International Publishing.
* [53]

  Alexander Breuer and Alexander Heinecke.
  Next-generation local time stepping for the ader-dg finite element
  method (submitted to ipdps21).
  2021.
* [54]

  TOSA.
  Tosa, <https://developer.mlplatform.org/w/tosa/>, Accessed on
  3/30/2021.
* [55]

  CUTLASS.
  Nvidia cutlass, <https://github.com/NVIDIA/cutlass>, Accessed on
  3/30/2021.
* [56]

  Philippe Tillet, HT Kung, and David Cox.
  Triton: an intermediate language and compiler for tiled neural
  network computations.
  In Proceedings of the 3rd ACM SIGPLAN International Workshop on
  Machine Learning and Programming Languages, pages 10–19, 2019.
* [57]

  XLA.
  Xla: Optimizing compiler for machine learning,
  <https://www.tensorflow.org/xla>, Accessed on 3/30/2021.
* [58]

  JAX.
  Jax: Autograd and xla, <https://github.com/google/jax>, Accessed
  on 3/30/2021.

Optimization Notice: Software and workloads used in
performance tests may have been optimized for performance only on
Intel microprocessors. Performance tests, such as SYSmark and
MobileMark, are measured using specific computer systems,
components, software, operations and functions. Any change to any
of those factors may cause the results to vary. You should
consult other information and performance tests to assist you in
fully evaluating your contemplated purchases, including the
performance of that product when combined with other products.
For more information go to http://www.intel.com/performance.

Intel, Xeon, and Intel Xeon Phi are trademarks of Intel Corporation in the U.S. and/or other

[◄](/html/2104.05754)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2104.05755)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2104.05755)
[View original  
on arXiv](https://arxiv.org/abs/2104.05755)[►](/html/2104.05756)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Fri Mar 8 00:15:11 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
