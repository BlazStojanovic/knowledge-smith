---
arxiv: '2601.18089'
authors:
- Venmugil Elango
- Nidhi Bhatia
- Roger Waleffe
- Rasoul Shafipour
- Tomer Asida
- Abhinav Khattar
- Nave Assaf
- Maximilian Golub
- Joey Guman
- Tiyasa Mitra
- Ritchie Zhao
- Ritika Borkar
- Ran Zilberstein
- Mostofa Patwary
- Mohammad Shoeybi
- Bita Rouhani
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts'
url: https://arxiv.org/abs/2601.18089
year: 2026
---

[2601.18089] LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts














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



# LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts

Venmugil Elango


 Nidhi Bhatia


 Roger Waleffe


 Rasoul Shafipour


 Tomer Asida


 Abhinav Khattar


 Nave Assaf


 Maximilian Golub


 Joey Guman


 Tiyasa Mitra


 Ritchie Zhao


 Ritika Borkar


 Ran Zilberstein


 Mostofa Patwary


 Mohammad Shoeybi


 Bita Rouhani

###### Abstract

Abstract.
Mixture of Experts (MoEs) have become a central component of many state-of-the-art open-source and proprietary large language models.
Despite their widespread adoption, it remains unclear how close existing MoE architectures are to optimal with respect to inference cost, as measured by accuracy per floating-point operation and per parameter. In this work, we revisit MoE design from a hardware-software co-design perspective, grounded in empirical and theoretical considerations. We characterize key performance bottlenecks across diverse deployment regimes, spanning offline high-throughput execution and online, latency-critical inference. Guided by these insights, we introduce LatentMoE, a new model architecture resulting from systematic design exploration and optimized for maximal accuracy per unit of compute. Empirical design space exploration at scales of up to 95B parameters and over a 1T-token training horizon, together with supporting theoretical analysis, shows that LatentMoE consistently outperforms standard MoE architectures in terms of accuracy per FLOP and per parameter. Given its strong performance, the LatentMoE architecture has been adopted by the flagship Nemotron-3 Super and Ultra models and scaled to substantially larger regimes, including longer token horizons and larger model sizes, as reported in nvidia2025nvidianemotron3efficient.

## 1 Introduction

Transformer-based large language models underpin a wide range of modern AI systems, from conversational agents to code generation and scientific reasoning. As these models continue to scale, practical deployment is increasingly constrained by inference cost, encompassing both computation and memory. As a result, a central objective in modern model design is to maximize achievable accuracy under fixed inference cost constraints.

Mixture-of-Experts (MoE) architectures have emerged as a promising approach towards this goal, enabling models to scale parameter count while keeping the number of Floating-point Operations (FLOPs) per token fixed. Despite their empirical success, the MoE design space remains poorly understood. Existing MoE architectures are largely motivated by high-level sparsity arguments and are optimized primarily for offline, throughput-oriented settings, with limited consideration of online deployments that impose strict latency, memory bandwidth, and communication constraints.

We argue that effective MoE design must be evaluated along two complementary dimensions: accuracy per FLOP and accuracy per parameter. While accuracy per FLOP captures computational efficiency, accuracy per parameter reflects memory footprint, memory bandwidth demands, routing-induced communication, and sharding overheads (factors that are often the dominant bottlenecks in interactive, low-latency inference). Neglecting these factors can lead to architectures that appear efficient in aggregate compute, yet incur substantial inefficiencies in practical deployment.

![Refer to caption](/html/2601.18089/assets/figures/standard_moe.png)


(a) Standard MoE architecture.

![Refer to caption](/html/2601.18089/assets/figures/latent_moe.png)


(b) LatentMoE architecture.

Figure 1: Standard MoE vs. LatentMoE architectures. In LatentMoE, tokens are projected from the model hidden dimension dd into a smaller latent dimension ℓ\ell for expert routing and computation, which reduces routed parameter loads and all‑to‑all traffic by a factor of d/ℓd/\ell. We use this efficiency to increase the total number of experts and the top-k active experts per token by the same factor d/ℓd/\ell, which improves the accuracy of the model while keeping overall inference cost approximately constant.

In this work, we revisit MoE architecture design from a hardware–software co-design perspective. Through a systematic analysis of existing MoE systems across the throughput–latency Pareto frontier, we identify key structural bottlenecks arising from expert parameterization, routing-induced all-to-all communication, and memory access patterns. Combined with detailed accuracy measurements and theoretical analysis, our study identifies structural inefficiencies in prevailing MoE designs that limit achievable accuracy per unit of inference cost.

Guided by these insights, we introduce LatentMoE, a new mixture-of-experts architecture explicitly optimized for both accuracy per FLOP and accuracy per parameter. LatentMoE decouples expert routing and computation from the model hidden dimension by projecting incoming activations into a shared low-dimensional latent space prior to expert processing (Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts")). The latent dimension serves as a direct control knob for computational cost, communication volume, and expert parameter size. At iso-FLOP and iso-parameter count, projecting incoming activations into a lower-dimensional latent space enables a proportional increase in both the number of experts and the routing top-k, while maintaining constant inference cost.

As we show both theoretically and empirically, simultaneously increasing expert count and combinatorial sparsity diversity improves the effective expressivity of the model, leading to higher achievable accuracy. Crucially, these gains arise without increasing memory bandwidth demands or communication overheads, making LatentMoE well suited for both latency-critical and throughput-oriented deployments. We validate the LatentMoE concept through pretraining experiments at scales of up to 95B parameters and over 1T tokens. Across all evaluated regimes, LatentMoE consistently improves upon standard MoE architectures, achieving higher accuracy at fixed inference cost or substantially lower inference cost at fixed accuracy. Given its strong performance, the LatentMoE architecture has been adopted by the flagship Nemotron-3 Super and Ultra models and scaled to substantially larger regimes, including longer token horizons and larger model sizes (nvidia2025nvidianemotron3efficient).

## 2 LatentMoE Core Design Principles

Before delving into the specifics of LatentMoE, we first take a systems-level view of what is required to deploy an MoE model that is both accurate and cost-efficient.

Throughout this section, we use Qwen3-235B-A22B as a running example for our modeling, with
N=128N=128 experts, K=8K=8 active experts per token, a hidden dimension d=4096d=4096, and an intermediate feed-forward dimension m=1536m=1536. For concreteness, we consider deployment on NVIDIA GB200 GPUs interconnected by a high-bandwidth NVLink fabric, which provides approximately 18001800 GB/s of bidirectional bandwidth per GPU (i.e., BWNVL=900\mathrm{BW}\_{\mathrm{NVL}}=900 GB/s per direction). To ensure that expert communication remains within a single NVLink domain, experts are distributed via expert parallelism across EP=64\mathrm{EP}=64 GPUs. Attention layers are executed using data parallelism over the same group of GPUs. Each GB200 GPU delivers a peak FP4 Tensor Core throughput of F=10F=10 PFLOPs and an HBM memory bandwidth of BWHBM=8\mathrm{BW}\_{\mathrm{HBM}}=8 TB/s (nvidia\_blackwell).

### 2.1 Memory Bandwidth Bottleneck

In highly interactive (*i.e*., low-latency) settings that typically use small batch sizes, MoE computation is primarily bottlenecked by memory bandwidth. Figure [2](#S2.F2 "Figure 2 ‣ 2.1 Memory Bandwidth Bottleneck ‣ 2 LatentMoE Core Design Principles ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts") provides a high-level roof-line analysis of performance versus arithmetic intensity.

![Refer to caption](/html/2601.18089/assets/x1.png)


Figure 2: Roofline Analysis for serving Qwen3-235B-A22B.
Operating points correspond to different per-expert token counts texpt\_{\mathrm{exp}} (i.e., effective expert batch sizes after MoE routing), mapped to arithmetic intensity I=2⋅texp⋅d⋅md⋅m+texp⋅(d+m)I=\frac{2\cdot t\_{\mathrm{exp}}\cdot d\cdot m}{d\cdot m+t\_{\mathrm{exp}}\cdot(d+m)}. At latency-critical batch sizes (low II), MoE expert computation is constrained by HBM bandwidth rather than compute, and the operating points lie in the bandwidth-bound regime.

For a GB200 system, a computation becomes compute-bound only if its arithmetic intensity (i.e., FLOPs per byte) exceeds:

|  |  |  |
| --- | --- | --- |
|  | FBWHBM=10×10158×1012=1250​ FLOPs/byte.\frac{F}{\mathrm{BW}\_{\mathrm{HBM}}}=\frac{10\times 10^{15}}{8\times 10^{12}}=1250\text{ FLOPs/byte}. |  |

Let ttotalt\_{\mathrm{total}} be the total number of tokens across the EP\mathrm{EP} GPUs prior to MoE routing. Assuming a uniform distribution of tokens across experts, the number of tokens assigned to a single expert is: texp:=ttotal⋅KNt\_{\mathrm{exp}}:=\frac{t\_{\mathrm{total}}\cdot K}{N}. In the Qwen3-235B-A22B example, with N=128N=128 and EP=64\mathrm{EP}=64, each GPU hosts N/EP=2N/\mathrm{EP}=2 experts; thus each GPU processes approximately 2⋅texp2\cdot t\_{\mathrm{exp}} expert tokens per MoE layer.

The FP4 compute cost for a single expert is Cexp=2⋅texp⋅d⋅mC\_{\mathrm{exp}}=2\cdot t\_{\mathrm{exp}}\cdot d\cdot m, and the corresponding memory traffic in FP4 precision—accounting for weights, inputs, and intermediate activations—is given by Mexp=d⋅m+texp⋅(d+m)M\_{\mathrm{exp}}=d\cdot m+t\_{\mathrm{exp}}\cdot(d+m). Since each GPU processes two experts in our example, the arithmetic intensity II is given by the ratio of the total compute to the total memory traffic:

|  |  |  |
| --- | --- | --- |
|  | I=2⋅Cexp2⋅Mexp=2⋅texp⋅d⋅md⋅m+texp⋅(d+m).I=\frac{2\cdot C\_{\mathrm{exp}}}{2\cdot M\_{\mathrm{exp}}}=\frac{2\cdot t\_{\mathrm{exp}}\cdot d\cdot m}{d\cdot m+t\_{\mathrm{exp}}\cdot(d+m)}. |  |

To operate in the compute-bound regime, we require I≥1250I\geq 1250. Substituting the Qwen3-235B-A22B parameters yields the condition:

|  |  |  |
| --- | --- | --- |
|  | 2⋅texp⋅d⋅md⋅m+texp⋅(d+m)≥1250⟹texp≥1418.\frac{2\cdot t\_{\mathrm{exp}}\cdot d\cdot m}{d\cdot m+t\_{\mathrm{exp}}\cdot(d+m)}\geq 1250\quad\implies\quad t\_{\mathrm{exp}}\geq 1418. |  |

In typical latency-critical deployments, effective batch sizes are small, resulting in texpt\_{\mathrm{exp}} being on the order of a few hundred tokens—well below the threshold of 1418. Consequently, MoE experts operate in the memory-bound region of the roofline curve (Figure [2](#S2.F2 "Figure 2 ‣ 2.1 Memory Bandwidth Bottleneck ‣ 2 LatentMoE Core Design Principles ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts")), where performance is limited by weight loading rather than compute capacity.

Design Principle I

In low-latency serving scenarios, MoE inference is typically dominated by the memory-bandwidth cost of loading model weights. Consequently, maximizing accuracy per parameter is critical for applications with high interactivity requirements.

### 2.2 Communication Bottleneck

In throughput-oriented settings, once experts become compute-bound, communication emerges as a significant contributor to end-to-end execution time in distributed settings. Expert parallelism mandates all-to-all token routing across devices, imposing an overhead that can control end-to-end execution time.
Recall that in our example configuration, each GPU hosts two experts. Since texpt\_{\mathrm{exp}} denotes the tokens processed per expert, each GPU processes a total of 2​texp2t\_{\mathrm{exp}} tokens across its local experts per MoE layer.

Assuming a uniform distribution of tokens, the all-to-all communication volume per GPU per MoE layer is given by:

|  |  |  |
| --- | --- | --- |
|  | Mcomm=2.5⋅(NEP⋅texp⋅d)=5⋅texp⋅d.M\_{\mathrm{comm}}=2.5\cdot\left(\frac{N}{\mathrm{EP}}\cdot t\_{\mathrm{exp}}\cdot d\right)=5\cdot t\_{\mathrm{exp}}\cdot d. |  |

Here, the factor 2.5 accounts for the mixed-precision traffic (0.5 bytes for FP4 dispatch, 2 bytes for BF16 aggregation). On the compute side, the total FLOP count for the two local experts is:

|  |  |  |
| --- | --- | --- |
|  | Ccomp=2⋅(NEP⋅texp⋅d⋅m)=4⋅texp⋅d⋅m.C\_{\mathrm{comp}}=2\cdot\left(\frac{N}{\mathrm{EP}}\cdot t\_{\mathrm{exp}}\cdot d\cdot m\right)=4\cdot t\_{\mathrm{exp}}\cdot d\cdot m. |  |

The corresponding compute time is: tcomp=CcompF=4⋅texp⋅d⋅mFt\_{\mathrm{comp}}=\frac{C\_{\mathrm{comp}}}{F}=\frac{4\cdot t\_{\mathrm{exp}}\cdot d\cdot m}{F}. Similarly, the all-to-all communication time is:
tcomm=McommBWNVL=5⋅texp⋅dBWNVLt\_{\mathrm{comm}}=\frac{M\_{\mathrm{comm}}}{\mathrm{BW}\_{\mathrm{NVL}}}=\frac{5\cdot t\_{\mathrm{exp}}\cdot d}{\mathrm{BW}\_{\mathrm{NVL}}},
where BWNVL=900​GB/s\mathrm{BW}\_{\mathrm{NVL}}=900\penalty 10000\ \text{GB/s} is the effective unidirectional NVLink bandwidth. Consequently, the ratio of communication time to compute time is:

|  |  |  |
| --- | --- | --- |
|  | tcommtcomp=5⋅texp⋅d/BWNVL4⋅texp⋅d⋅m/F=5⋅F4⋅m⋅BWNVL.\frac{t\_{\mathrm{comm}}}{t\_{\mathrm{comp}}}=\frac{5\cdot t\_{\mathrm{exp}}\cdot d/\mathrm{BW}\_{\mathrm{NVL}}}{4\cdot t\_{\mathrm{exp}}\cdot d\cdot m/F}=\frac{5\cdot F}{4\cdot m\cdot\,\mathrm{BW}\_{\mathrm{NVL}}}. |  |

Substituting the parameters for the GB200 NVL72 and Qwen3-235B-A22B yields a ratio of approximately 9. This indicates that in the throughput-oriented regime, MoE layers are heavily dominated by all-to-all communication overhead.

Design Principle II

Improving performance in throughput-oriented MoE deployments requires minimizing the data volume of all-to-all operations. This volume is proportional to:



Mcomm∝NEP⋅texp⋅d=ttotal⋅K⋅dEP.M\_{\mathrm{comm}}\propto\frac{N}{\mathrm{EP}}\cdot t\_{\mathrm{exp}}\cdot d=\frac{t\_{\mathrm{total}}\cdot K\cdot d}{\mathrm{EP}}.
Consequently, communication overhead can be mitigated by reducing the routed hidden dimension dd or the number of active experts KK. Note that modifying the intermediate dimension mm does not affect the token size and thus yields no direct improvement.

### 2.3 Model Quality

Beyond optimizing inference speed, preserving model quality is paramount. To this end, we draw on theoretical insights into neural network expressivity and combinatorial sparsity.
Classical results on Barron functions (barron1993) state that a one-hidden-layer network with uu nonlinear units achieves a mean-squared error of 𝒪​(1/u)\mathcal{O}(1/u), independent of the input dimension dd. In an MoE layer, this effective nonlinear budget per token is proportional to the total width of the selected experts:

|  |  |  |
| --- | --- | --- |
|  | Ueff∝K⋅m.U\_{\mathrm{eff}}\propto K\cdot m. |  |

This implies that reducing the active experts KK or the intermediate dimension mm directly penalizes the effective capacity (UeffU\_{\mathrm{eff}}), risking model quality degradation.

Design Principle III

Maintaining model quality requires preserving the effective nonlinear budget, K⋅mK\cdot m. Consequently, to alleviate memory and communication bottlenecks without sacrificing model quality, we should keep both the number of active experts and the intermediate dimension unchanged.

Every inference task is characterized by an intrinsic feature rank, reffr\_{\mathrm{eff}}, corresponding to the minimum number of degrees of freedom required to preserve task-relevant information. Reducing the hidden dimension dd below this threshold necessarily discards such information, leading to accuracy degradation. Thus, reffr\_{\mathrm{eff}} serves as a task-dependent lower bound on dd.

Design Principle IV

There exists a task-specific feature rank reffr\_{\mathrm{eff}} that imposes a lower limit on the reduction of dd. Reducing dd below this limit precipitates a collapse in model quality.

Additionally, the MoE architecture benefits from combinatorial sparsity, offering (NK)\binom{N}{K} possible expert combinations per token (deepseekmoe). Increasing the total number of experts NN expands this specialization space. Furthermore, scaling both NN and KK by a factor α\alpha exponentially increases the diversity of expert mixtures:

|  |  |  |
| --- | --- | --- |
|  | (α​Nα​K)≥((NK))α.\binom{\alpha N}{\alpha K}\geq\left(\binom{N}{K}\right)^{\alpha}. |  |

Design Principle V

Scaling both the number of experts NN and top-k per token KK enhances model quality by exponentially expanding the space of expert combinations.

##### Putting it all together.

Design Principles I and II indicate that improving inference speed requires reducing both memory bandwidth and communication costs. Memory bandwidth cost scales with
dd and mm, while communication cost scales with KK and dd. However, Principle III cautions against reducing either KK or mm, as doing so would likely degrade model quality. This leaves dd as the most promising dimension to reduce, enabling performance improvements in both throughput- and latency-oriented regimes without significant loss in accuracy. Principle IV further establishes a lower bound, (reffr\_{\mathrm{eff}}), on dd to prevent quality collapse. Moreover, Principle V suggests that increasing NN and KK improves model quality. Since memory bandwidth and communication costs scale linearly with KK, we can simultaneously increase KK by a factor α\alpha and reduce dd by the same factor α\alpha (provided d/α≥reffd/\alpha\geq r\_{\mathrm{eff}}). We hypothesize, and empirically validate in subsequent sections, that this transformation (also depicted in Figure [1(b)](#S1.F1.sf2 "In Figure 1 ‣ 1 Introduction ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts")) preserves memory bandwidth and communication costs while improving network expressivity and combinatorial sparsity, yielding higher accuracy per FLOP and per parameter.

## 3 LatentMoE Architecture

Guided by the design principles outlined in Section [2](#S2 "2 LatentMoE Core Design Principles ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts"), we introduce LatentMoE, a new MoE architecture designed for efficient scaling. LatentMoE first projects each input token x∈ℝdx\in\mathbb{R}^{d} into a lower-dimensional latent space ℝℓ\mathbb{R}^{\ell} using a learnable down-projection matrix W↓∈ℝℓ×dW\_{\downarrow}\in\mathbb{R}^{\ell\times d}. The resulting compressed representation is then routed to the selected experts. Each expert Ei​(⋅;ℓ)E\_{i}(\cdot;\ell) operates entirely within the latent space and is parameterized by weights WFC1(i),Wgate(i)∈ℝm×ℓW\_{\text{FC1}}^{(i)},W\_{\text{gate}}^{(i)}\in\mathbb{R}^{m\times\ell} and WFC2(i)∈ℝℓ×mW\_{\text{FC2}}^{(i)}\in\mathbb{R}^{\ell\times m}. After expert computation, the outputs are aggregated and projected back to the original input dimension using a learnable up-projection matrix W↑∈ℝd×ℓW\_{\uparrow}\in\mathbb{R}^{d\times\ell}.

Since we compress only the input dimension dd to ℓ\ell while keeping the intermediate dimension mm constant, the effective nonlinear budget UeffU\_{\mathrm{eff}} remains unchanged. While Design Principle III suggests this should theoretically preserve accuracy, in practice, larger models are often easier to train and more robust to hyperparameter variations (lottery\_ticket\_hypothesis; novak2018sensitivity; sensitivity\_analysis). To avoid extensive hyperparameter tuning for the compressed model, we leverage Design Principle V by scaling the total number of experts NN by a factor α=d/ℓ\alpha=d/\ell, thereby expanding the combinatorial specialization space.
Crucially, since neither the memory bandwidth cost (in latency-oriented scenarios) nor the communication cost (in throughput-oriented scenarios) depends on NN, this scaling adheres to Design Principles I and II, incurring no additional inference overhead. Hereafter, we refer to this architecture modification as ℓ−MoEeff\operatorname{\ell-MoE}\_{\mathrm{eff}}, formally defined as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℓ−MoEeff⁡(x):=W↑⋅(∑i∈𝒯K,N′pi′​Ei​(W↓⋅x;ℓ))+∑j=N′+1N′+SEj​(x;d).\operatorname{\ell-MoE}\_{\mathrm{eff}}(x):=W\_{\uparrow}\cdot\left(\sum\_{i\in\mathcal{T}\_{K,N^{\prime}}}p^{\prime}\_{i}E\_{i}(W\_{\downarrow}\cdot x;\ell)\right)+\sum\_{j=N^{\prime}+1}^{N^{\prime}+S}E\_{j}(x;d). |  | (1) |

Here, N′=α⋅NN^{\prime}=\alpha\cdot N denotes the expanded set of routed experts. The routed experts Ei​(⋅;ℓ)E\_{i}(\cdot;\ell) function within the latent space, while the shared experts Ej​(⋅;d)E\_{j}(\cdot;d) operate in the original input space. The routing weights p′=Softmax⁡(Wr′⋅x)p^{\prime}=\operatorname{Softmax}(W^{\prime}\_{r}\cdot x) are computed from the original token x∈ℝdx\in\mathbb{R}^{d} using a learnable weight matrix Wr′∈ℝN′×dW^{\prime}\_{r}\in\mathbb{R}^{N^{\prime}\times d}, and 𝒯K,N′\mathcal{T}\_{K,N^{\prime}} denotes the indices of the top-KK experts (out of N′N^{\prime} total) selected based on their routing scores. For simplicity, all operations outside the routed experts—including the MoE routing mechanism and shared experts—continue to operate in the original hidden dimension dd, as they do not significantly contribute to the identified memory and communication bottlenecks.

Following the down-projection W↓W\_{\downarrow}, token dispatch and aggregation occur in the latent space ℝℓ\mathbb{R}^{\ell}. This reduces the communication volume by a factor of α\alpha relative to a standard MoE. Similarly, because the expert weights lie in the latent space (ℝm×ℓ\mathbb{R}^{m\times\ell} and ℝℓ×m\mathbb{R}^{\ell\times m}), the memory bandwidth cost for weight loading is also reduced by a factor of α\alpha.

Design Principle V further suggests that scaling both NN and KK by a factor α\alpha exponentially increases expert diversity, thereby enhancing model quality. Following this principle, the default LatentMoE configuration (a.k.a., ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}}) is defined as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℓ−MoEacc⁡(x):=W↑⋅(∑i∈𝒯K′,N′pi′​Ei​(W↓⋅x;ℓ))+∑j=N′+1N′+SEj​(x;d),\operatorname{\ell-MoE}\_{\mathrm{acc}}(x):=W\_{\uparrow}\cdot\left(\sum\_{i\in\mathcal{T}\_{K^{\prime},N^{\prime}}}p^{\prime}\_{i}E\_{i}(W\_{\downarrow}\cdot x;\ell)\right)+\sum\_{j=N^{\prime}+1}^{N^{\prime}+S}E\_{j}(x;d), |  | (2) |

where K′=α⋅KK^{\prime}=\alpha\cdot K. This formulation differs from ℓ−MoEeff\operatorname{\ell-MoE}\_{\mathrm{eff}} solely in the number of active experts, utilizing the top-k selection function 𝒯K′,N′\mathcal{T}\_{K^{\prime},N^{\prime}}.

Since KK is increased by a factor of α=d/ℓ\alpha=d/\ell, this variant keeps communication cost and memory bandwidth requirements constant relative to a standard MoE. The increased expert diversity and non-linearity budget per token, however, lead to superior model accuracy at iso-inference cost, thereby pushing the Pareto frontier of models to a new level. Table [1](#S3.T1 "Table 1 ‣ 3 LatentMoE Architecture ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts") summarizes the costs and benefits of the two configurations, ℓ−MoEeff\operatorname{\ell-MoE}\_{\mathrm{eff}} and ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}}. For completeness, we evaluate both setups in Section [4](#S4 "4 Evaluation ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts").

Table 1: Comparison of asymptotic communication and memory bandwidth costs per GPU. Costs are normalized by hardware constants. TexpT\_{\text{exp}} denotes the average number of tokens per expert, and EP\mathrm{EP} is the expert-parallel level. Arrows indicate improvement (↑\uparrow), maintenance (→\rightarrow), or baseline (–).

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Architecture | Communication  Cost | Weight Loading  Memory Cost  per Expert | Model  Accuracy | Inference  Efficiency |
| Standard MoE | (N/EP)⋅texp⋅d(N/\mathrm{EP})\cdot\,t\_{\mathrm{exp}}\cdot d | d⋅md\cdot m | – | – |
| ℓ−MoEeff\operatorname{\ell-MoE}\_{\mathrm{eff}} | (N/EP)⋅texp⋅ℓ(N/\mathrm{EP})\cdot\,t\_{\mathrm{exp}}\cdot\ell | ℓ⋅m\ell\cdot m | →\rightarrow | ↑\uparrow |
| ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}} (recommended) | (N/EP)⋅texp⋅d(N/\mathrm{EP})\cdot\,t\_{\mathrm{exp}}\cdot d | d⋅md\cdot m | ↑\uparrow | →\rightarrow |

## 4 Evaluation

In this section, we conduct a thorough design space exploration to verify the effectiveness of the proposed LatentMoE architecture. We start by pretraining Transformer MoE models at two different scales: (1) 16B total parameters with 2B active, which we use for conducting ablation studies, and (2) 95B total parameters with 8B active, which we use as a scaling test of the 16B results. To demonstrate the generalizability of LatentMoE architectures, we further extend our study by training hybrid Mamba-Attention MoE models at scale.

We use the architecture and hyperparameters from DeepSeek-v2-lite (deepseek\_v2) for our 2B active model ablations. For the 8B active Transformer model, we use a cosine learning rate schedule with a max learning rate of 1.2×10−31.2\times 10^{-3} decayed to a minimum of 3×10−63\times 10^{-6}. The 8B active Hybrid model is trained with a WSD schedule with a max learning rate of 8×10−48\times 10^{-4} decayed to 8×10−68\times 10^{-6} in the last 15% of training. Both the 8B active Transformer and hybrid models are trained with a sequence length of 8192, a batch size of 768 (∼6\sim 6 million tokens), and a learning rate warmup of 8.48.4 billion tokens. The 8B active models use a load balancing loss coefficient of 10−410^{-4} along with DeepSeek’s aux-loss-free load balancing strategy (auxiliarylossfreeloadbalancingstrategy) to ensure balanced token load throughout training. Table [2](#S4.T2 "Table 2 ‣ 4 Evaluation ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts") summarizes the model architecture under study in this paper.

Table 2: Architectural specifications of the baseline models used for design space exploration. For the Hybrid model’s Mamba layers, we use Mamba-2 blocks with 128 heads, 64 head dimension, 128 state dimension, and 8 groups.

|  |  |  |  |
| --- | --- | --- | --- |
| Configuration | 16BT-2BA | 95BT-8BA | Hybrid-73BT-8BA |
| Layers (LL) | 27 | 32 | 52 (24 Mamba/MoE, 4 Attn.) |
| Hidden Dimension (dd) | 2048 | 4096 | 4096 |
| Total Routed Experts (NN) | 64 | 128 | 128 |
| Active Experts (KK) | 6 | 6 | 6 |
| Shared Experts (SS) | 2 | 2 | 2 |
| Intermediate FFN Dimension (mm) | 1408 | 2688 | 2688 |
| Activation Function | SwiGLU | Squared-ReLU | Squared-ReLU |
| Attention Heads | 16 | 32 | 32 |
| Query Groups (GQA) | 16 | 8 | 8 |
| Total Parameters | 16B | 95B | 73B |
| Active Parameters | 2B | 8B | 8B |

### 4.1 LatentMoE Ablations

Impact of compression ratio.  Design Principle IV (Section [2](#S2 "2 LatentMoE Core Design Principles ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts")) hypothesizes that there exists an intrinsic rank reffr\_{\mathrm{eff}} such that compressing the latent dimension to ℓ≥reff\ell\geq r\_{\mathrm{eff}} results in negligible information loss. To empirically validate this and estimate reffr\_{\mathrm{eff}}, we pretrain and sweep different compression ratios on top of the ℓ−MoEeff\operatorname{\ell-MoE}\_{\mathrm{eff}} configuration, holding all other hyperparameters constant.
Results in Figure [3](#S4.F3 "Figure 3 ‣ 4.1 LatentMoE Ablations ‣ 4 Evaluation ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts") indicate that model quality is preserved for compression ratios α≤4\alpha\leq 4. Consequently, we adopt α=4\alpha=4 for all subsequent experiments. We empirically verified that this setting remains effective at larger scales as well (i.e., 95B total and 8B active).

![Refer to caption](/html/2601.18089/assets/figures/2B_vary_alpha_lmoe_eff.png)


Figure 3: Effect of compression ratio on model quality. Validation loss for the 16BT-2BA model using the ℓ−MoEeff\operatorname{\ell-MoE}\_{\mathrm{eff}} configuration across varying compression ratios α=d/ℓ\alpha=d/\ell. The total number of experts is scaled by α\alpha, while the base model configuration follows Table [2](#S4.T2 "Table 2 ‣ 4 Evaluation ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts").

![Refer to caption](/html/2601.18089/assets/figures/2B_reduced_params.png)


Figure 4: Impact of expert scaling on model quality. Comparison of validation loss for the 16BT-2BA model using the ℓ−MoEeff\operatorname{\ell-MoE}\_{\mathrm{eff}} configuration when the hidden dimension is compressed by 4×4\times. The green curve utilizes the proposed expert scaling (N′=α​NN^{\prime}=\alpha N), while the red curve does not. Scaling the expert count effectively mitigates the accuracy loss caused by compression, eliminating the need for extensive hyperparameter retuning.

Impact of number of experts. 
In Section [3](#S3 "3 LatentMoE Architecture ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts"), we noted that parameter reduction via compression can impede training stability. To quantify this, we pretrain the ℓ−MoEeff\operatorname{\ell-MoE}\_{\mathrm{eff}} LatentMoE variant of the 16B total and 2B active parameter model with the hidden dimension dd compressed by a factor of 4, both with and without a compensatory increase in the total number of experts, using the baseline hyperparameters. As shown in Figure [4](#S4.F4 "Figure 4 ‣ 4.1 LatentMoE Ablations ‣ 4 Evaluation ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts"), reducing dd without scaling the expert count leads to significant quality degradation, validating the expert scaling strategy employed by LatentMoE.

Comparison between the two variants of LatentMoE. 
In Section [3](#S3 "3 LatentMoE Architecture ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts"), we introduced two LatentMoE variants: ℓ−MoEeff\operatorname{\ell-MoE}\_{\mathrm{eff}}, designed to improve inference efficiency while maintaining baseline accuracy, and ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}}, designed to enhance accuracy at a comparable inference cost. Figure [5](#S4.F5 "Figure 5 ‣ 4.1 LatentMoE Ablations ‣ 4 Evaluation ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts") compares the validation loss of these configurations against the baseline for the 16B total and 2B active parameter model using a latent dimension of ℓ=512\ell=512 (α=4\alpha=4). Consistent with our expectations, ℓ−MoEeff\operatorname{\ell-MoE}\_{\mathrm{eff}} matches the baseline accuracy, whereas ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}} achieves a noticeably lower validation loss. We recommend ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}} for Pareto-optimal accuracy versus inference cost.

![Refer to caption](/html/2601.18089/assets/figures/2b_latent512_baseline_leff_lacc.png)


Figure 5: Comparison between LatentMoE variants. Training trajectories for the baseline 16BT-2BA
model versus the ℓ−MoEeff\operatorname{\ell-MoE}\_{\mathrm{eff}} and ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}} (ℓ=512\ell=512). ℓ−MoEeff\operatorname{\ell-MoE}\_{\mathrm{eff}} matches baseline convergence, while ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}} outperforms the baseline.

### 4.2 LatentMoE Scaling Studies

Leveraging the insights from the 16B model ablations, we train a 95B parameter Transformer using a LatentMoE configuration with a 4×4\times compression ratio. Figure [6](#S4.F6 "Figure 6 ‣ 4.2 LatentMoE Scaling Studies ‣ 4 Evaluation ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts") presents the validation loss trajectories for ℓ−MoEeff\operatorname{\ell-MoE}\_{\mathrm{eff}} and ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}} relative to the baseline. Consistent with the 16BT-2BA results, ℓ−MoEeff\operatorname{\ell-MoE}\_{\mathrm{eff}} matches the baseline, while ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}} demonstrates superior results. Table [3](#S4.T3 "Table 3 ‣ 4.2 LatentMoE Scaling Studies ‣ 4 Evaluation ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts") shows the downstream task accuracy at the 300B token horizon.
We report Code as the average over HumanEval, HumanEval+, MBPP, and MBPP+, Math as the average of GSM8K CoT and MATH-500, and Commonsense understanding as the average of RACE, ARC-Challenge, HellaSwag, and Winogrande.
For simplicity, we used the exact same hyperparameters optimized for the baseline Transformer for LatentMoE. Further hyperparameter tuning might lead to even better accuracy.

![Refer to caption](/html/2601.18089/assets/figures/8B_val_loss.png)


Figure 6: 95B Model Training Convergence. Validation loss curves for the 95BT-8BA baseline, ℓ−MoEeff\operatorname{\ell-MoE}\_{\mathrm{eff}}, and ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}} configurations (ℓ=1024,α=4\ell=1024,\alpha=4). ℓ−MoEeff\operatorname{\ell-MoE}\_{\mathrm{eff}} matches baseline convergence, while ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}} outperforms the baseline.




Table 3: Accuracy comparisons of the 95BT-8BA model with and without LatentMoE. Compared to the baseline, with equivalent parameters, ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}} provides higher accuracy across all downstream tasks, while ℓ−MoEeff\operatorname{\ell-MoE}\_{\mathrm{eff}} provides comparable to or better accuracy with much fewer FLOPs.

|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model | Active Params | Total Params | MMLU Pro | MMLU | Code | Math | Commonsense |
| Baseline | 8.47B | 94.4B | 29.26 | 58.95 | 40.33 | 64.39 | 74.32 |
| ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}} | 8.44B | 94.8B | 34.91 | 62.23 | 41.50 | 64.88 | 75.18 |
| ℓ−MoEeff\operatorname{\ell-MoE}\_{\mathrm{eff}} | 5.62B | 94.8B | 34.75 | 61.06 | 40.68 | 63.61 | 73.72 |

To further validate the effectiveness of the LatentMoE architecture, we also pretrain a series of hybrid Mamba-Attention MoE models. Specifically, we first train a baseline 8B active (73B total) parameter model. As described in Table [2](#S4.T2 "Table 2 ‣ 4 Evaluation ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts"), each MoE layer in the hybrid architecture contains 128 experts, 6 activated experts, 2 shared experts, and uses an intermediate FFN dimension of 2688. We use Squared-ReLU activation and a model dimension of 4096. We then train the ℓ−MoEeff\operatorname{\ell-MoE}\_{\mathrm{eff}} and ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}} LatentMoE variants of the baseline model, using a 4×4\times compression ratio.

Results after training the above models on 1T tokens are shown in Table [4](#S4.T4 "Table 4 ‣ 4.2 LatentMoE Scaling Studies ‣ 4 Evaluation ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts"). All models are trained with identical hyperparameters. As shown, the LatentMoE ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}} variant achieves significantly higher accuracy than the baseline across all tasks, while the ℓ−MoEeff\operatorname{\ell-MoE}\_{\mathrm{eff}} variant achieves accuracy comparable to or better than the standard granular MoE baseline.

Overall, the LatentMoE architecture provides a clear advantage in terms of accuracy per FLOP and per parameter compared to granular MoEs, paving the way for higher accuracy at fixed inference cost or lower inference cost at fixed accuracy.

Table 4: Accuracy comparisons of hybrid Mamba-Attention MoEs with and without LatentMoE. Compared to the baseline, with equivalent parameters, ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}} provides higher accuracy across all downstream tasks, while ℓ−MoEeff\operatorname{\ell-MoE}\_{\mathrm{eff}} provides comparable or better accuracy with much fewer FLOPs.

|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model | Active Params | Total Params | MMLU Pro | MMLU | Code | Math | Commonsense |
| Baseline | 8.09B | 72.6B | 48.30 | 70.10 | 51.95 | 78.32 | 81.73 |
| ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}} | 8.02B | 72.8B | 52.87 | 72.11 | 55.14 | 80.19 | 82.10 |
| ℓ−MoEeff\operatorname{\ell-MoE}\_{\mathrm{eff}} | 5.91B | 72.8B | 51.29 | 71.34 | 53.13 | 77.01 | 80.78 |

### 4.3 Inference Performance

As discussed in Section [3](#S3 "3 LatentMoE Architecture ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts"), ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}} is expected to achieve similar inference speed to the standard MoE baseline while attaining higher accuracy.
Our evaluations in Section [4.2](#S4.SS2 "4.2 LatentMoE Scaling Studies ‣ 4 Evaluation ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts") confirmed that ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}} indeed achieves higher accuracy compared to standard MoE. Here, we evaluate ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}} from the perspective of inference efficiency.
Table [5](#S4.T5 "Table 5 ‣ 4.3 Inference Performance ‣ 4 Evaluation ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts") presents the measured performance of ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}} compared to standard MoE for the Hybrid-73BT-8BA model (see Table [2](#S4.T2 "Table 2 ‣ 4 Evaluation ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts")) on two Hopper H100 GPUs using vLLM with FP8 per-tensor quantization.
We focus our measurements on the hybrid Mamba-Attention baseline, as it represents the most efficient inference architecture.

|  |  |  |
| --- | --- | --- |
| Concurrency | LatentMoE (Tokens/s/GPU) | Standard MoE (Tokens/s/GPU) |
| 1 | 181.6 | 206.6 |
| 4 | 528.5 | 509.8 |
| 16 | 1130.8 | 1204.6 |
| 64 | 1569.6 | 1549.3 |
| 128 | 1625.8 | 1725.9 |

Table 5: Comparison of LatentMoE and Standard MoE performance metrics.

The measurements show that at higher concurrencies, per-GPU throughput drops by only up to 6%. It is important to note that further software optimizations could be performed to mitigate even these small throughput differences between LatentMoE and standard MoE. One proposed optimization is to utilize separate CUDA streams for routed and shared experts, which could reduce end-to-end latency when performing inference with smaller batches or when model dimensions do not saturate the GPU’s compute. A second optimization targets the MoE kernels from the CUTLASS library. Since LatentMoEs decrease the size of the GEMMs for routed experts, it is important to ensure that inner dimensions remain large enough to fully utilize the GPU and avoid SM-bound workloads. When inner dimensions are small, specialized smaller-matrix GEMM kernels should be used.

#### 4.3.1 Projected Serving Impact at Trillion-Parameter Scale

Inference efficiency can be characterized as a three-dimensional trade-off surface, with accuracy along one axis, throughput per GPU along a second axis, and latency (user interactivity) along the third axis. Thus far, we have discussed the accuracy of LatentMoE and presented measured performance at the 95B-parameter scale. In the following section, we examine a two-dimensional slice of this trade-off at the trillion-parameter scale by projecting throughput per GPU and latency Pareto frontiers for accuracy-matched models.

![Refer to caption](/html/2601.18089/assets/figures/normalized_plot.png)


Figure 7: Projected throughput-latency Pareto frontiers at trillion-parameter scale.
Left: Decode-heavy traffic modeled with chunked piggybacking serving. Right: Prefill-heavy traffic modeled with disaggregated serving.
Kimi-K2-1T-LatentMoE attains accuracy-efficient operating points. Matching its accuracy under standard MoE scaling requires an additional ∼\sim350B parameters in our construction and yields a 1.24×1.24\times–3.46×3.46\times projected slowdown across the frontier.

Performance evaluation methodology. We use a high-fidelity proprietary performance simulator to project end-to-end serving performance for a trillion-parameter class model and its LatentMoE variant. We simulate over 200200K operating points to estimate the throughput per GPU and latency Pareto frontiers shown in Figure [7](#S4.F7 "Figure 7 ‣ 4.3.1 Projected Serving Impact at Trillion-Parameter Scale ‣ 4.3 Inference Performance ‣ 4 Evaluation ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts"). We consider two traffic patterns. The first is a decode-heavy setting modeled with chunked piggybacking serving. The second is a prefill-heavy setting modeled with disaggregated serving, where prefill and decode are separated to reflect long-context traffic. The serving strategy (for example, whether to use disaggregation) is selected following the guidance in mitra2025beyond.

Effective Parameter Multiplier (EPM). We use the effective parameter multiplier to construct an iso-accuracy baseline for inference comparisons. We benchmark the native Kimi-K2-1T against our proposed variant, Kimi-K2-1T-LatentMoE. Following the “Effective Parameter Count” framework established by frantar2025compressionscaling, we posit that a treated model with physical parameters Nt​r​e​a​tN\_{treat} behaves like a standard dense baseline with effective parameters Ne​f​fN\_{eff}. We assume baseline performance follows a scaling law f​(N)f(N). For a treated model achieving score St​r​e​a​tS\_{treat}, the effective capacity is obtained by inverting the baseline scaling law:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Ne​f​f=f−1​(St​r​e​a​t)N\_{eff}=f^{-1}(S\_{treat}) |  | (3) |

The EPM is defined as the ratio of effective capacity to physical parameters:

|  |  |  |  |
| --- | --- | --- | --- |
|  | λ=Ne​f​fNt​r​e​a​t\lambda=\frac{N\_{eff}}{N\_{treat}} |  | (4) |

We use λ\lambda to construct an iso-accuracy baseline with parameter count Niso=λ⋅Nt​r​e​a​tN\_{\text{iso}}=\lambda\cdot N\_{treat}. For our evaluations, we derive f​(N)f(N) by fitting a log-linear function to the MMLU accuracy scores of the Qwen-3-Dense model family (0.6B, 1.7B, 4B, 8B, 14B, and 32B):

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(N)=a⋅log⁡N+bf(N)=a\cdot\log N+b |  | (5) |

where aa and bb are fitted parameters.

Iso-accuracy baseline construction. We estimate an Effective Parameter Multiplier of λ≈1.35×\lambda\approx 1.35\times for Kimi-K2-1T-LatentMoE. For a 1T-parameter base model, this implies an iso-accuracy scale of Niso≈1.35N\_{\text{iso}}\approx 1.35T, corresponding to an increase of (1.35−1.0)(1.35-1.0)T ≈0.35\approx 0.35T ≈350\approx 350B parameters. Guided by this target, we construct a physical iso-accuracy baseline, denoted Kimi-K2-1.35T, by scaling the native architecture depth from 61 to 80 layers. This construction matches the projected effective capacity implied by LatentMoE and enables a direct inference-efficiency comparison against a standard model of comparable predictive power.

Accuracy matching with standard MoE scaling is more expensive. ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}} achieves an accuracy gain at fixed parameter and FLOP budget. When we enforce an accuracy-matched comparison using the standard MoE architecture, the required scaling incurs a marked serving penalty. Across the projected Pareto frontier (Figure [7](#S4.F7 "Figure 7 ‣ 4.3.1 Projected Serving Impact at Trillion-Parameter Scale ‣ 4.3 Inference Performance ‣ 4 Evaluation ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts")), Kimi-K2-1.35T is approximately 1.24×1.24\times–3.46×3.46\times slower than Kimi-K2-1T-LatentMoE, indicating that ℓ−MoEacc\operatorname{\ell-MoE}\_{\mathrm{acc}} provides a more favorable accuracy-latency trade-off than increasing model size through the standard MoE architecture to reach the same accuracy target.

Latent projection overhead is modest. Relative to native Kimi-K2-1T, Kimi-K2-1T-LatentMoE introduces additional computation due to latent projection operators. In our projections, native Kimi-K2-1T remains close, within up to ∼\sim9% of Kimi-K2-1T-LatentMoE, indicating that projection overhead is small compared to the cost of achieving the same accuracy gain via standard scaling to Kimi-K2-1.35T.

## 5 Related Work

Mixture-of-Experts (MoE) models have become a cornerstone of state-of-the-art large language model services. In this work, we challenge the original MoE design paradigm for the first time and introduce an alternative architecture that achieves higher accuracy under both iso-parameter and iso-FLOP constraints.

In parallel, the community has developed a rich set of model compression techniques to reduce inference cost, including quantization (rouhani2023microscalingdataformatsdeep; shared\_microexponents) and sparsity (xie2024moeprunerpruningmixtureofexpertslarge). At the expert level, pruning (not\_all\_experts\_are\_equal; reap\_the\_experts; chen2022taskspecificexpertpruningsparse) and merging (li2024mergecompressdemystifyefficient) methods have also been proposed. These approaches are orthogonal to the LatentMoE design and can be composed with it to yield further efficiency gains.

The closest related work to this paper is probably MoLAE (molae). MoLAE is a post-training compression method built on low-rank approximation of expert weights in a latent space. Although the two methods appear similar at a surface level, LatentMoE makes fundamentally different design trade-offs by coupling expert compression with increased network expressivity and combinatorial sparsity. By contrast, to compensate for accuracy loss caused by latent-space projection, MoLAE introduces grouped latent projections and restricts compression to only part of the experts (FC2). These design choices, in turn, forgo communication savings during token dispatch and limit memory bandwidth reduction, ultimately constraining the achievable efficiency gains. As discussed in Section [2](#S2 "2 LatentMoE Core Design Principles ‣ LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts"), efficient MoE serving is not FLOP-bound; reducing FLOPs alone is not enough to improve the Pareto frontier of accuracy vs. throughput vs. latency.

Concurrent work explores improving model quality under fixed compute by modifying residual connectivity rather than the expert path. Manifold-Constrained Hyper-Connections (mHC) (xie2026mhcmanifoldconstrainedhyperconnections) improves quality under iso-compute by widening the residual stream and increasing residual-path connectivity.
Achieving this requires a materially different residual topology (multi-stream residual state) and a learned connection-generation mechanism (RMSNorm →\rightarrow linear →tanh\rightarrow\tanh gating for the connection maps, plus constrained residual mixing for stability). We believe LatentMoE and mHC are complementary and can be stacked on top of one another. Further exploration is left to future work.

## 6 Conclusion

We presented LatentMoE, a revised Mixture-of-Experts architecture designed to maximize accuracy per FLOP and per parameter by explicitly accounting for the dominant memory bandwidth and communication bottlenecks in modern inference systems. By projecting tokens into a lower-dimensional latent space, LatentMoE reduces routing all-to-all communication as well as the memory bandwidth and computation required per expert. These savings are then reinvested into scaling expert count and routing diversity without increasing inference cost. Across extensive experiments up to 95B parameters, hybrid architectures, and projected trillion-parameter serving scenarios, LatentMoE consistently outperforms standard MoEs on the accuracy–efficiency Pareto frontier.

## References

[◄](/html/2601.18088)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2601.18089)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2601.18089)
[View original  
on arXiv](https://arxiv.org/abs/2601.18089)[►](/html/2601.18090)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Thu Feb 5 14:08:02 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
