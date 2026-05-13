---
arxiv: '2511.18890'
authors:
- Yonggan Fu
- Xin Dong
- Shizhe Diao
- Matthijs Van keirsbilck
- Hanrong Ye
- Wonmin Byeon
- Yashaswi Karnati
- Lucas Liebenwein
- Hannah Zhang
- Nikolaus Binder
- Maksim Khadkevich
- Alexander Keller
- Jan Kautz
- Yingyan Celine Lin
- Pavlo Molchanov
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models'
url: https://arxiv.org/abs/2511.18890
year: 2025
---

[2511.18890] Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models














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

X

# Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models

Yonggan Fu


 Xin Dong


 Shizhe Diao


 Matthijs Van keirsbilck


 Hanrong Ye


 Wonmin Byeon


 Yashaswi Karnati


 Lucas Liebenwein


 Hannah Zhang


 Nikolaus Binder


 Maksim Khadkevich


 Alexander Keller


 Jan Kautz


 Yingyan (Celine) Lin†


 Pavlo Molchanov

###### Abstract

Efficient deployment of small language models (SLMs) is essential for numerous real-world applications with stringent latency constraints. While previous work on SLM design has primarily focused on reducing the number of parameters to achieve parameter-optimal SLMs, parameter efficiency does not necessarily translate into proportional real-device speed-ups.
This work aims to identify the key determinants of SLMs’ real-device latency and offer generalizable principles and methodologies for SLM design and training when real-device latency is the primary consideration.
Specifically, we identify two central architectural factors: depth–width ratios and operator choices. The former is crucial for small-batch-size latency, while the latter affects both latency and large-batch-size throughput. In light of this, we first study latency-optimal depth–width ratios, with the key finding that although deep–thin models generally achieve better accuracy under the same parameter budget, they may not lie on the accuracy–latency trade-off frontier. Next, we explore emerging efficient attention alternatives to evaluate their potential as candidate building operators. Using the identified promising operators, we construct an evolutionary search framework to automatically discover latency-optimal combinations of these operators within hybrid SLMs, thereby advancing the accuracy–latency frontier.
In addition to architectural improvements, we further enhance SLM training using a weight normalization technique that enables more effective weight updates and improves final convergence. This technique can serve as a generalizable component for future SLMs.
Combining these methods, we introduce a new family of hybrid SLMs, called Nemotron-Flash, which significantly advances the accuracy–efficiency frontier of state-of-the-art SLMs, e.g., achieving over +5.5% average accuracy, 1.3×\times/1.9×\times lower latency, and 18.7×\times/45.6×\times higher throughput compared to Qwen3-1.7B/0.6B, respectively.

Models on Hugging Face:
[Nemotron-Flash-1B](https://huggingface.co/nvidia/Nemotron-Flash-1B) |
[3B](https://huggingface.co/nvidia/Nemotron-Flash-3B) |
[3B-Instruct](https://huggingface.co/nvidia/Nemotron-Flash-3B-Instruct)

## 1 Introduction

Recent advances in large language models (LLMs) have driven remarkable breakthroughs across a wide range of applications and industries. Despite their impressive capabilities, the substantial computational demands and high latency of LLMs present significant barriers to practical deployment, particularly in latency-sensitive scenarios or on resource-constrained hardware. This challenge has intensified the demand for small language models (SLMs).

Most existing SLM designs prioritize parameter reduction to achieve efficiency; however, parameter-efficient models do not necessarily yield proportional latency reductions, especially on hardware AI accelerators like GPUs and TPUs. Additionally, current SLM development processes rely heavily on empirical trial-and-error rather than systematic and principled methodologies. For instance, deep-thin architectures —such as those employed in MobileLLM [liu2024mobilellm] and SmolLM [allal2025smollm2smolgoesbig] —often result in suboptimal accuracy-latency trade-offs. Furthermore, with the rapid advent of efficient attention operators [dao2022flashattention, katharopoulos2020transformers], the potential synergies of combining these operators in hybrid models have not been thoroughly explored [lieber2024jamba, waleffe2024empirical, dong2024hymba, de2024griffin], leading to manual, heuristic-driven architecture decisions that are increasingly costly and less scalable.

To address these gaps, this paper introduces generalizable principles and automated methodologies for latency-optimal SLM design and training. We perform a comprehensive study of both architectural choices and training strategies to understand their impact on efficiency and accuracy. Based on these insights, we propose a series of techniques for constructing latency-optimal SLMs and analyze their general applicability and effectiveness. These advancements are integrated into a new family of SLMs, which we refer to as Nemotron-Flash.

![Refer to caption](/html/2511.18890/assets/x1.png)


Figure 1: Visualizing (a) the accuracy–latency trade-off and (b) the accuracy-throughput trade-off of our Nemotron-Flash and SOTA SLMs, where the average accuracy is computed across 16 tasks spanning commonsense reasoning, math, coding, and recall tasks. Latency is measured on an NVIDIA H100 GPU for decoding 8k tokens with a batch size of 1 using CUDA Graph. Decoding throughput is measured with a 32k-token input length using the maximum batch size that does not cause out-of-memory (OOM) errors for each model. The marker size represents the model depth.

Specifically, our architectural exploration focuses on two key factors: depth–width ratios and operator selection, where the former is crucial for small-batch-size latency and the latter affects both latency and large-batch-size throughput.
Through extensive training and profiling, we show that (1) deep-thin models, while parameter-efficient, yield suboptimal latency–accuracy trade-offs, and (2) the optimal depth–width ratio scales with the target latency constraints. Guided by this general principle, we also extend existing scaling laws [hu2024minicpm] to relate model loss to both depth and width. This allows the sweet-spot depth–width ratio to be determined by profiling a range of configurations and selecting the one that meets the latency constraint while minimizing loss, as predicted by the scaling law.

Further, we comprehensively evaluate emerging attention operators for their accuracy-latency trade-offs and potential as SLM building blocks. With these findings, we introduce an evolutionary search framework that efficiently identifies optimal combinations of hybrid attention operators. Our search strategy exploits the early stabilization of performance rankings among LM architectures, using short training runs as reliable proxies for final performance, thereby enabling fast and reliable search.

In addition to architectural modifications, based on observations of structural patterns in the weight matrices of trained LMs, we further enhance SLM training by constraining weight norms to increase the effective learning rate. This consistently improves final convergence and downstream accuracy across model families. We also employ learnable meta tokens [dong2024hymba] for cache initialization.

By combining these architectural and training innovations, we develop the Nemotron-Flash model family, which significantly advances the accuracy–latency trade-offs for SLMs. As shown in Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), Nemotron-Flash markedly pushes forward the accuracy–efficiency frontier compared to state-of-the-art (SOTA) SLMs. For example, with all models accelerated using TensorRT-LLM’s AutoDeploy kernels [nvidia2025autodeploy] and CUDA Graph, Nemotron-Flash-3B achieves +2.0%/+5.5% higher average accuracy, 1.7×\times/1.3×\times lower latency, and 6.4×\times/18.7×\times higher throughput compared to Qwen2.5-3B/Qwen3-1.7B, respectively. Similarly, Nemotron-Flash-1B achieves +5.5% higher average accuracy, 1.9×\times lower latency, and 45.6×\times higher throughput than Qwen3-0.6B.

![Refer to caption](/html/2511.18890/assets/x2.png)


Figure 2: The accuracy–parameter/latency trade-offs when varying depth and width. While deeper models generally achieve a better accuracy–parameter trade-off, they may not perform as well in the accuracy–latency trade-off and there exists an
optimal depth-width ratio for a latency budget.

## 2 Related Work

Small language models.
The large model size and computational demands of LLMs [zhang2022opt, jiang2023mistral, touvron2023llama, llama3, team2024gemma1, team2024gemma2, team2025gemma3, achiam2023gpt] hinder their efficient deployment on resource-constrained platforms. This has motivated the development of SLMs, such as MobileLLM [liu2024mobilellm], MiniCPM [hu2024minicpm], PanGu-π\pi Pro [tang2024rethinking], and TinyLlama [zhang2024tinyllama]. These works aim to deliver parameter-efficient SLMs within a given parameter budget. However, parameter efficiency alone often does not translate into proportional latency reductions on real devices. For example, previous SLMs [liu2024mobilellm, tang2024rethinking] often adopt deep-thin model structures, which may result in suboptimal latency-accuracy trade-offs.
Our work targets real-device efficiency and aims to provide insights and methodologies for developing latency-optimal SLMs.

Efficient attention alternatives.
To address the quadratic computation and linearly increasing memory of attention modules, efficient attention alternatives with sub-quadratic complexity in sequence length have been proposed [peng2023rwkv, sun2023retentive, gu2023mamba, dao2024transformers, yang2023gated, katharopoulos2020transformers, yang2024parallelizing, yang2024gated].
Notable examples include RWKV [peng2023rwkv], RetNet [sun2023retentive], Mamba [gu2023mamba], Mamba2 [dao2024transformers], GLA [yang2023gated], DeltaNet [yang2024parallelizing], Gated DeltaNet [yang2024gated], and JetBlock [gu2025jet], each featuring different memory update rules. However, despite their potential, linear attention mechanisms have been found to exhibit limited recall capabilities [waleffe2024empirical] and to underperform standard attention mechanisms on in-context learning tasks [park2024can].

Hybrid language models.
To combine the efficiency of linear attention with the recall capabilities of quadratic attention, hybrid models that incorporate both types of operators have emerged.
Specifically, [park2024can, waleffe2024empirical, lieber2024jamba, glorioso2024zamba, ren2024samba] sequentially stack Mamba and attention layers in hybrid models and show enhanced commonsense reasoning and long-context capabilities. [dong2024hymba] proposes a hybrid-head structure that stacks attention and Mamba in parallel. Other recent works have also explored hybrid models that mix linear RNNs or convolutions with attention [de2024griffin, pilault2024block, saon2023diagonal, yang2024parallelizing]. However, existing hybrid models still rely on manual operator combinations, requiring tedious trial-and-error processes. Our work aims to automate operator combination in hybrid models, enabling more scalable hybrid model development.

## 3 Towards Latency-Optimal SLM Design and Training

To deliver latency-optimal SLMs, both architecture design and training are critical to achieving the best accuracy-latency trade-off. This work explores both aspects. For SLM design, real-device latency is primarily determined by two key factors: the model’s depth and width, and the choice of operators, which are studied in Sec.[3.1](#S3.SS1 "3.1 SLM Design: Depth-Width Ratios ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models") and Sec.[3.2](#S3.SS2 "3.2 SLM Design: Hybrid Operators ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), respectively. For SLM training, we introduce two optimization techniques in Sec.[3.3](#S3.SS3 "3.3 SLM Training: Weight Normalization ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models") and Sec.[3.4](#S3.SS4 "3.4 SLM Training: Meta Tokens ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models").
Finally, we integrate these components to construct a new model family, Nemotron-Flash, in Sec.[4](#S4 "4 Nemotron-Flash: The New Hybrid SLM Family ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), and benchmark against SOTA SLMs in Sec.[4.1](#S4.SS1 "4.1 Benchmark with SOTA Base SLMs ‣ 4 Nemotron-Flash: The New Hybrid SLM Family ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models").

### 3.1 SLM Design: Depth-Width Ratios

Previous SLMs [liu2024mobilellm, tang2024rethinking] find that deep-thin models generally achieve better task accuracy than wide-shallow ones under the same parameter budget. However, when targeting real-device latency, this may not hold, as partially noted by [tang2024rethinking]. Our key question is: Do deeper or wider models better optimize the accuracy-latency trade-off? To answer this, we provide a systematic exploration to understand the impact of depth and width on the accuracy-latency trade-off.

Exploration settings.
We train a series of Llama models with five depth settings: 6, 12, 18, 24, and 30 blocks, where each block contains one attention and one feed-forward network (FFN), on 100B tokens from the Smollm-corpus [benallal2024smollmcorpus]. For each depth setting, we also vary the model width (i.e. hidden size) to create models with different sizes and latencies. We visualize the resulting accuracy-parameter and accuracy-latency trade-offs in Fig. [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models") (a) and (b), respectively. Accuracy is averaged over eight commonsense reasoning (CR) tasks, and latency is measured as the decoding time for a 1k token generation under a batch size of 1 on an NVIDIA A100 GPU.

Observations and analysis. We observe that: ❶ Deeper models generally achieve a better accuracy-parameter trade-off over a wide depth range, although the benefit gradually saturates; ❷ For the accuracy-latency trade-off, the advantage of deep-thin models may not hold, and there exists an optimal depth setting for a given latency budget. For example, when the latency budget is 3 seconds, a depth of 12 achieves the best accuracy among the evaluated settings; ❸ The optimal depth-width ratio generally increases with the latency budget.
These observations highlight the necessity of deliberate depth/width selection based on deployment constraints, rather than defaulting to deep-thin models.

Augmented scaling laws for determining sweet-spot depth-width ratios. Although the general trend in the above analysis holds, detailed curves may shift across devices and generation lengths, complicating the selection of model depth and width. As such, in addition to the above insights, we also explore principled methods to identify the sweet-spot depth–width ratio within a model family.

In light of this, we augment existing scaling laws [hoffmann2022training, hu2024minicpm] by parameterizing model loss with model depth and width.
Specifically, existing LM scaling laws [hoffmann2022training, hu2024minicpm] parameterize language modeling loss ℒ​(P,N)=ℒ0+C​1⋅P−α+C​2⋅N−γ\mathcal{L}(P,N)=\mathcal{L}\_{0}+C1\cdot{P}^{-\alpha}+C2\cdot{N}^{-\gamma}, with PP and NN as model size and data size, respectively, and C​1C1, C​2C2, α\alpha, γ\gamma as fitting parameters. We decouple the model size PP into two factors, model depth DD and width WW, and reformulate the scaling law:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ℒ​(D,W,N)\displaystyle\mathcal{L}(D,W,N) | =ℒ0+a​D−α+b​W−β+c​N−γ\displaystyle=\mathcal{L}\_{0}+aD^{-\alpha}+bW^{-\beta}+cN^{-\gamma} |  | (1) |

where the fitting parameters aa, bb, and cc control the contributions from each dimension, and the exponents α\alpha, β\beta, and γ\gamma govern the diminishing returns from increasing each respective dimension.
Since the impact of data size is additive and decoupled from depth and width, we can study the effect of depth and width on LM loss under a fixed data size, i.e., by neglecting the data size term.

As such, in practice, given a target latency budget and deployment setting, the sweet-spot depth-width ratio can be obtained by profiling a range of depth-width configurations and selecting the one that meets the latency constraint while achieving the lowest loss.

![Refer to caption](/html/2511.18890/assets/x3.png)


Figure 3: Fitting the scaling law and validating on larger depth/width settings.

Fitting and extrapolation.
To validate the effectiveness of this augmented scaling law, we fit it using the above Llama models with varying depth DD and width WW.
Specifically, we use perplexity (PPL) as the loss metric, fit the scaling law on a subset of depth/width settings, and validate it on models with larger width/depth settings to assess extrapolation.
As shown in Fig. [3](#S3.F3 "Figure 3 ‣ 3.1 SLM Design: Depth-Width Ratios ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), we find that the model extrapolates reasonably well to unseen depth/width settings, staying within 5.3% of the ground-truth PPL, demonstrating that the fitted function generalizes beyond the observed training configurations.

Lesson learned: SLM depth-width ratios.
Deep-thin models may not be latency-optimal, and the optimal depth-width ratio generally increases with the target latency budget. Fitting a scaling law with depth and width provides a more principled approach to identifying the sweet-spot ratios.

![Refer to caption](/html/2511.18890/assets/x4.png)


Figure 4:  (a) The validation loss evolution, and (b) the PPL-latency trade-off of different operators.

### 3.2 SLM Design: Hybrid Operators

Beyond model depth and width, the operator used in each layer is another critical dimension. We first train existing LM architectures in a fully controlled setting to identify the most promising operators on the accuracy-latency frontier. We then develop an evolutionary search pipeline to automatically and efficiently discover hybrid combinations of these operators for constructing hybrid SLMs.

Exploration settings.
We train a series of 500M LMs built using emerging efficient attention alternatives, including Mamba [gu2023mamba], Mamba2 [dao2024transformers], GLA [yang2023gated], DeltaNet [yang2024parallelizing], Gated DeltaNet [yang2024gated], RWKV7 [peng2025rwkv], and sliding window attention (SWA) with a window size of 512. We use the official implementation for Mamba/Mamba2, FlashAttention [dao2023flashattention2] for SWA, and FlashLinearAttention [yang2024fla] for all other linear attention variants.
All models follow the designs specified in their original papers (e.g., one FFN after each attention operator, except for Mamba/Mamba2), use the same number of blocks, and are trained on 100B tokens from the Smollm-corpus.
We present the validation loss in Fig. [4](#S3.F4 "Figure 4 ‣ 3.1 SLM Design: Depth-Width Ratios ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models") (a) and the Wikitext PPL-latency trade-off in Fig. [4](#S3.F4 "Figure 4 ‣ 3.1 SLM Design: Depth-Width Ratios ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models") (b), measured by decoding 8k tokens with a batch size of 1 on an NVIDIA A100 GPU with CUDA Graph enabled.

In addition, inspired by recent hybrid LMs [waleffe2024empirical, glorioso2024zamba, lieber2024jamba, ren2024samba, dong2024hymba], which combine attention and Mamba/Mamba2 within the same model, we also integrate the promising operators identified in Fig. [4](#S3.F4 "Figure 4 ‣ 3.1 SLM Design: Depth-Width Ratios ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models") with Mamba2 or SWA to construct hybrid models in a layer-wise interleaved manner, aiming to gain insights into which operator combinations are well-matched and complementary, as shown in Tab. [1](#S3.T1 "Table 1 ‣ 3.2 SLM Design: Hybrid Operators ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"). Note that, for fair comparison, we control the number of blocks in the hybrid models to be the same as in the pure models, based on the insights from Sec. [3.1](#S3.SS1 "3.1 SLM Design: Depth-Width Ratios ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"). More details of the ablation settings are provided in Appendix [A](#A1 "Appendix A Detailed Experimental Settings ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models").

Table 1: Exploration of operator combinations. All models have 500M parameters and the same depth.

|  |  |  |  |
| --- | --- | --- | --- |
| Operator1 | Operator2 | ↓\downarrow Wiki PPL | ↑\uparrow CR Acc (%) |
| Mamba2 | - | 24.36 | 47.72 |
| Attention | - | 24.44 | 48.02 |
| Mamba2 | 23.52 | 48.07 |
| DeltaNet | - | 23.87 | 47.83 |
| Attn | 24.32 | 47.48 |
| Mamba2 | 23.37 | 48.03 |
| Gated DeltaNet | - | 23.80 | 47.96 |
| Attn | 24.31 | 47.80 |
| Mamba2 | 23.37 | 48.03 |
| GLA | - | 25.16 | 46.82 |
| Attention | 24.60 | 47.63 |
| Mamba2 | 24.04 | 47.99 |

Observations and analysis.
We observe that: ❶ In terms of language modeling, DeltaNet and Gated DeltaNet generally emerge as promising candidates, lying on the PPL-latency Pareto frontier; ❷ When integrated into hybrid models with attention or Mamba2, pairing DeltaNet or Gated DeltaNet with Mamba2 typically results in lower PPL and higher accuracy, consistently outperforming the corresponding pure models. In contrast, improvements from pairing with attention are less stable. This indicates both the advantages of hybrid models and the importance of selecting complementary operator combinations; ❸ When used in hybrid models, the performance gap between individual operators may narrow, likely due to the complementary and diverse memory mechanisms introduced by hybrid layers. For example, although Gated DeltaNet outperforms DeltaNet in language modeling, their task performance becomes comparable when integrated with Mamba2, making DeltaNet the preferable operator in hybrid models due to its greater efficiency.

Table 2: Visualizing architectures searched under different efficiency metrics. m2, a, d, and f denote Mamba2, attention, DeltaNet, and FFN, respectively. CR accuracy is averaged over eight tasks.

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| |  | | --- | | Search | | Metric | | Params (M) | |  | | --- | | ↓\downarrow Decoding | | Latency (8k) | | |  | | --- | | ↓\downarrow Wiki | | PPL | | |  | | --- | | ↑\uparrow CR | | Acc (%) | | Searched Operator Combinations |
| Decoding Latency | 837 | 17.71 | 20.70 | 51.04 | [d, f, m2, f, a, f, m2, f, d, f, m2, f, a, f, m2, f, d, f, m2, f, d, f, m2, f] |
| Parameters | 497 | 16.94 | 23.06 | 49.23 | [m2, f, a, f, a, f, d, m2, f, a, f, a, f, d, m2, f, a, f, a, f, d, m2, f, a, f, d, f, f] |

Evolutionary search for operator combinations.
The emergence of various efficient attention mechanisms and their complex synergy in hybrid models motivate an automated framework to identify their efficient and complementary combination in hybrid SLMs. To achieve this, we built an evolutionary search engine to efficiently navigate a complex combinatorial design space.

Short-training PPL as a search proxy. A crucial observation underpinning our method is that the relative performance rankings of different LM architectures stabilize early during training, which can also be observed from the validation loss curves in Fig. [4](#S3.F4 "Figure 4 ‣ 3.1 SLM Design: Depth-Width Ratios ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models") (a). Using this insight, we demonstrate that short-training PPL provides a reliable proxy metric to predict final task performance, substantially reducing training cost for assessing each candidate. We compute the Spearman correlation [sedgwick2014spearman], a measure of rank correlation that is crucial for architecture ranking in our case, between short-training PPL and full-training PPL across multiple LM architectures. We find an 88.8% Spearman correlation, which is sufficient to identify strong architectures within our search space.

Our search space.
Based on the identified promising operator candidates and their synergy in hybrid models, we adopt DeltaNet, Attention, and Mamba2 as candidate operators.
We search over a maximum of three types of building blocks, each assigned to the early, middle, or late stages of the LM architecture. This three-stage strategy balances operator heterogeneity with architectural regularity. The search explores the ratios of each operator and the number of FFNs per block type, and the repetition count of each block type. More details are provided in Appendix [C.1](#A3.SS1 "C.1 Detailed Search Space ‣ Appendix C Evolutionary Search: More Details and Experiments ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models").

![Refer to caption](/html/2511.18890/assets/x5.png)


Figure 5: Visualizing the search trajectories.

Evolutionary search algorithm.
We adopt the aging evolution search [real2019regularized] with the following steps:
① Initialization: Seed and short-train an initial population of architectures, either from known designs or randomly sampled ones;
② Selection: In each evolutionary cycle, we use tournament selection [miller1995genetic] to identify parent architectures that are high-performing based on short-training PPL and meet a predefined target latency budget;
③ Mutation: Selected parents undergo targeted mutations in one of the design factors—operator ratios, FFN ratios, or block type count;
④ Evaluation and replacement: Mutated offspring architectures are trained and evaluated using short-training PPL, with latency accurately estimated from a precomputed look-up table (LUT). The oldest architectures are replaced with new candidates, effectively balancing exploration and exploitation.
A more detailed description of the algorithm is provided in Appendix [C.2](#A3.SS2 "C.2 Our Evolutionary Search Algorithm ‣ Appendix C Evolutionary Search: More Details and Experiments ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models").

Search with decoding latency.
To evaluate the efficacy of our search framework, we conduct a search using decoding latency as the efficiency metric (measured by generating 8k tokens with a batch size of 1 on an NVIDIA A100 GPU). For demonstration purposes, we use a window size of 512 for attention, as this length is sufficient for general CR tasks and for the search proxy. We visualize our search process in Fig. [5](#S3.F5 "Figure 5 ‣ 3.2 SLM Design: Hybrid Operators ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), where 10 new architectures are sampled and evaluated in each cycle. Our search process progressively improves toward better models with lower PPL under the target latency.

The searched architecture is visualized in Tab. [2](#S3.T2 "Table 2 ‣ 3.2 SLM Design: Hybrid Operators ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"). Interestingly, we find that the latency-friendly architecture discovered by the search adopts DeltaNet-FFN-Mamba2-FFN and Attention-FFN-Mamba2-FFN as basic building blocks, stacking them in an interleaved manner. This finding echoes both our earlier observations—that DeltaNet and Mamba2 are strong candidates—and prior work that interleaves attention and state-space models [waleffe2024empirical, glorioso2024zamba, lieber2024jamba, ren2024samba, dong2024hymba].

Table 3: Benchmarking the searched architecture against baselines scaled to the same latency.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Model | Params (M) | ↓\downarrow Wiki PPL | ↑\uparrow CR Acc (%) | ↓\downarrow Latency |
| SWA | 616 | 23.33 | 48.72 | 18.01 |
| GLA | 862 | 22.67 | 48.43 | 18.19 |
| DeltaNet | 852 | 20.90 | 50.38 | 18.18 |
| Gated DeltaNet | 672 | 21.98 | 49.99 | 17.91 |
| Mamba2 | 601 | 23.14 | 48.61 | 17.82 |
| Mamba2 + FFN | 889 | 21.43 | 50.04 | 17.73 |
| Searched (Ours) | 837 | 20.70 | 51.04 | 17.71 |

![Refer to caption](/html/2511.18890/assets/x6.png)


Figure 6:  The weight distributions (absolute values) of the layers in the last attention block of 1B Llama models trained without and with our weight normalization technique.

![Refer to caption](/html/2511.18890/assets/x7.png)


Figure 7:  Validation loss trajectories of four model families with and without weight normalization.

We benchmark against baseline architectures that have the same model depth and a scaled hidden size to match the decoding latency of our searched architecture. All models are trained on 100B tokens from the Smollm-corpus, and we evaluate them using Wikitext PPL and CR accuracy, averaged over eight tasks. As shown in Tab. [3](#S3.T3 "Table 3 ‣ 3.2 SLM Design: Hybrid Operators ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), the searched hybrid architectures outperform their pure-model counterparts in both PPL and accuracy. This improvement is attributed to: (1) efficient operator combinations that allow for larger parameter counts under the same decoding latency, and (2) the complementary roles played by the hybrid operators.

Search with number of parameters.
We also conduct a new round of search using the number of parameters (500M) as the efficiency metric. We find that: ❶ The searched architecture achieves over 1.21% higher CR accuracy and a reduction of more than 0.74 in PPL compared to all 500M baselines (detailed results are provided in Appendix [C.3](#A3.SS3 "C.3 Search with Parameter Efficiency as the Objective ‣ Appendix C Evolutionary Search: More Details and Experiments ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models")); and ❷ as shown in Tab. [2](#S3.T2 "Table 2 ‣ 3.2 SLM Design: Hybrid Operators ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), when comparing the parameter-efficient architecture with the decoding-efficient one, the former generally includes more attention modules, which are parameter-efficient but decoding-unfriendly, and fewer Mamba2/DeltaNet modules, and has greater model depth, which is a parameter-efficient design choice according to Sec. [3.1](#S3.SS1 "3.1 SLM Design: Depth-Width Ratios ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"). This set of experiments validates our search scheme’s efficacy in identifying operator combinations that align with the target efficiency metric.

Lesson learned: SLM operator combinations.
Hybrid models show great promise, but the synergy among different operators is complex, necessitating the identification and combination of complementary operators. The relatively stable ranking of architectures in the early training phases serves as a useful signal for iterating over different designs, and this process can be strategically accelerated using appropriate search algorithms.

### 3.3 SLM Training: Weight Normalization

The potential of SLMs can be better realized when they are properly trained. We observe that model weights trained under a standard scheme exhibit non-smoothness, with large weight magnitudes along certain dimensions, as shown in the first row of Fig. [6](#S3.F6 "Figure 6 ‣ 3.2 SLM Design: Hybrid Operators ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"). As suggested by [karras2024analyzing, loshchilov2024ngpt], larger weight magnitudes can lead to smaller relative weight updates under comparable gradient magnitudes, potentially resulting in ineffective learning during later training stages when the learning rate is low.

Motivated by this and following [loshchilov2024ngpt], we constrain weight magnitudes by projecting model weights onto a unit norm sphere after each training iteration. This normalization step eliminates the radial component and emphasizes angular updates, leading to larger relative weight changes under comparable gradient magnitudes. Specifically, based on the patterns shown in Fig. [6](#S3.F6 "Figure 6 ‣ 3.2 SLM Design: Hybrid Operators ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), where weight matrices applied to hidden features (noted as Case-1) and those whose outputs are added back to hidden features (noted as Case-2) exhibit horizontal and vertical patterns, respectively, we perform weight normalization along the corresponding dimensions.
Formally, for each weight matrix 𝐖∈ℝCo​u​t×Ci​n\mathbf{W}\in\mathbb{R}^{C\_{out}\times C\_{in}}, after each training step, we project it onto a unit norm sphere:
𝐖i,:←𝐖i,:‖𝐖i,:‖2​for ​i=1,…,Co​u​t\mathbf{W}\_{i,:}\leftarrow\frac{\mathbf{W}\_{i,:}}{||\mathbf{W}\_{i,:}||\_{2}}\text{for }i=1,\dots,C\_{out} for Case-1, and
𝐖:,j←𝐖:,j||𝐖∗:,j||2​for ​j=1,…,Ci​n\mathbf{W}\_{:,j}\leftarrow\frac{\mathbf{W}\_{:,j}}{||\mathbf{W}\*{:,j}||\_{2}}\text{for }j=1,\dots,C\_{in} for Case-2. As shown in the second row of Fig. [6](#S3.F6 "Figure 6 ‣ 3.2 SLM Design: Hybrid Operators ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), our weight normalization results in smoother weight distributions.

![Refer to caption](/html/2511.18890/assets/x8.png)


Figure 8: Visualizing the gradient norm and L2 norm of DeltaNet model weights during training.

Evaluation across models.
We apply weight normalization to different models and visualize the validation loss curves in Fig. [7](#S3.F7 "Figure 7 ‣ 3.2 SLM Design: Hybrid Operators ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), along with the average element-wise gradient norm and L2 norm of the weight matrices in Fig. [8](#S3.F8 "Figure 8 ‣ 3.3 SLM Training: Weight Normalization ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"). We can observe that ❶ although in the early training stage, the baseline w/o weight normalization has steeper convergence due to unconstrained weights updates (with radical weight updates), the convergence speed will gradually diminish. With weight normalization, the convergence speed is more constant and the convergence will surpass the baseline in later training stages, leading to consistent better final convergence across model families; ❷ Weight normalization leads to much reduced L2 norm of model weights while slightly increasing the gradient norm compared to the baseline, ensuring larger relative weight changes, which is particularly helpful in late training stages. We observe consistent patterns in other models, provided in Appendix [D.2](#A4.SS2 "D.2 Gradient Norm Evolution ‣ Appendix D Weight Normalization: More Analysis and Experiments ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models").

Table 4: Evaluating weight normalization across 1B models trained on 100B tokens from [benallal2024smollmcorpus].

|  |  |  |  |
| --- | --- | --- | --- |
| Model | Setting | ↓\downarrow Wikitext PPL | ↑\uparrow CR Acc (%) |
| Llama 1B | w/o wnorm | 18.67 | 53.81 |
| w/ wnorm | 18.03 | 54.85 |
| DeltaNet 1B | w/o wnorm | 18.86 | 53.46 |
| w/ wnorm | 18.19 | 54.39 |
| Mamba2 1B | w/o wnorm | 18.44 | 53.30 |
| w/ wnorm | 17.88 | 54.71 |

We also report the corresponding improvements in language modeling and CR accuracy in Tab. [4](#S3.T4 "Table 4 ‣ 3.3 SLM Training: Weight Normalization ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"). We observe that weight normalization consistently improves CR accuracy by +1.20% and reduces PPL by 0.66 on average across model families, indicating its general effectiveness as an add-on component.

Connections with nGPT.
Our weight normalization technique can be viewed as a simplified and efficient variant of nGPT [loshchilov2024ngpt], which enforces all computations in LMs to operate on a unit sphere by introducing multiple activation normalization layers in addition to weight normalization.
We find that: ❶ Weight normalization, along with the resulting more effective weight updates, is key to improved convergence. When applied alone, weight normalization achieves final task performance comparable to the full nGPT solution, as demonstrated in Appendix [D.1](#A4.SS1 "D.1 Weight Normalization vs. nGPT ‣ Appendix D Weight Normalization: More Analysis and Experiments ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models");
❷ More importantly, the additional activation normalization layers in nGPT introduce significant training overhead—i.e., increasing SLM training time by more than 20%—which reduces the number of training tokens given a fixed training duration.
Thus, our contribution lies in identifying the primary contributing component and delivering a more efficient alternative.

Table 5: Evaluating meta tokens on two linear attention models with 48 operators and on our searched model with 24 operators in Tab. [2](#S3.T2 "Table 2 ‣ 3.2 SLM Design: Hybrid Operators ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models").

|  |  |  |  |
| --- | --- | --- | --- |
| Model | Meta Token | ↓\downarrow Wikitext PPL | ↑\uparrow CR Acc (%) |
| Mamba2-48L-1B | w/o | 19.04 | 51.71 |
| w/ | 18.98 | 52.33 |
| DeltaNet-48L-1B | w/o | 19.60 | 52.12 |
| w/ | 19.47 | 52.46 |
| Searched-24L-830M | w/o | 20.61 | 50.74 |
| w/ | 20.49 | 51.13 |

### 3.4 SLM Training: Meta Tokens

Previous work [dong2024hymba] has shown that prepending a set of learnable tokens to regular tokens can alleviate attention sinks [xiao2023efficient], which are caused by the force-to-attend phenomenon on semantically unimportant tokens. We find that these meta tokens can also benefit non-softmax linear attention mechanisms, as they serve as learned cache initialization when reformulating linear attention in a recurrent format during decoding. As shown in Tab. [5](#S3.T5 "Table 5 ‣ 3.3 SLM Training: Weight Normalization ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), prepending 256 meta tokens can consistently improve both language modeling and reasoning accuracy (+0.45% on average) with negligible overhead.

Table 6: We benchmark our Nemotron-Flash-1B/3B against SOTA SLMs across 16 tasks, including MMLU, commonsense reasoning (CR), math, coding, and recall tasks. Latency is measured on an NVIDIA H100 GPU for decoding 8k tokens with a batch size of 1 using CUDA Graph. Throughput (Thr.) is measured with a 32k-token input length using the maximum batch size w/o OOM. Nemotron-Flash-3B-TP is a throughput-optimized variant configured with the 1FA+2SWA setting in Sec. [4.3](#S4.SS3 "4.3 Ablation on Attention Configuration ‣ 4 Nemotron-Flash: The New Hybrid SLM Family ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models").

|  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | Param. | Depth | BS=1  Latency (s) | Max BS  Thr. (tok/s) | MMLU  (%) | CR  (%) | Math  (%) | Coding  (%) | Recall  (%) | Avg.  (%) |
| AMD-OLMo-1B | 1B | 16 | 24.66 | 142 | 27.11 | 51.21 | 12.66 | 7.35 | 60.32 | 35.63 |
| Llama-3.2-1B | 1.2B | 16 | 21.44 | 932 | 31.06 | 50.82 | 17.24 | 24.15 | 65.41 | 41.45 |
| Qwen2.5-0.5B | 0.5B | 24 | 22.81 | 2382 | 47.61 | 47.52 | 32.66 | 32.06 | 65.41 | 45.16 |
| Qwen3-0.6B | 0.6B | 28 | 27.55 | 160 | 52.44 | 48.91 | 36.88 | 24.32 | 62.92 | 44.11 |
| Nemotron-Flash-1B | 0.96B | 12 | 14.45 | 7289 | 44.63 | 54.46 | 34.86 | 37.91 | 67.11 | 49.63 |
| H2O-Danube3-4B | 4B | 24 | 45.92 | 178 | 53.76 | 60.79 | 34.73 | 11.65 | 58.45 | 44.37 |
| SmolLM2-1.7B | 1.7B | 24 | 31.11 | 151 | 50.21 | 58.61 | 32.88 | 21.11 | 62.42 | 46.21 |
| Llama-3.2-3B | 3B | 28 | 45.47 | 173 | 56.30 | 58.11 | 30.07 | 34.80 | 70.01 | 50.89 |
| Qwen2.5-1.5B | 1.5B | 28 | 34.50 | 687 | 60.68 | 56.29 | 48.14 | 42.95 | 69.32 | 54.65 |
| Qwen3-1.7B | 1.7B | 28 | 36.20 | 157 | 62.46 | 57.21 | 53.71 | 43.76 | 66.42 | 55.47 |
| Qwen2.5-3B | 3B | 36 | 49.40 | 459 | 65.56 | 58.88 | 53.83 | 49.46 | 73.03 | 58.96 |
| Nemotron-Flash-3B | 2.7B | 18 | 28.71 | 2939 | 61.19 | 61.02 | 57.62 | 53.33 | 73.25 | 60.98 |
| Nemotron-Flash-3B-TP | 2.7B | 18 | 27.95 | 4657 | 61.19 | 60.94 | 57.88 | 52.40 | 73.77 | 60.84 |




Table 7: Benchmarking Nemotron-Flash-3B-Instruct with SOTA Instruct SLMs across different tasks.

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Instruct Model | Param. | Lat. (s) | Thr. (tok/sec) | MMLU | GPQA | GSM8K | IFEval | Avg |
| SmolLM2-1.7B | 1.7B | 31.11 | 151 | 49.11 | 29.24 | 47.68 | 55.06 | 45.27 |
| Qwen2.5-1.5B | 1.5B | 34.50 | 687 | 59.73 | 30.13 | 56.03 | 46.78 | 48.17 |
| Qwen3-1.7B | 1.7B | 36.20 | 157 | 60.18 | 28.34 | 64.88 | 31.29 | 46.17 |
| Nemotron-Flash-3B | 2.7B | 28.71 | 2939 | 60.34 | 29.54 | 69.45 | 52.03 | 52.84 |

## 4 Nemotron-Flash: The New Hybrid SLM Family

Combining all the above architectural improvements and training techniques, we develop and train a new hybrid SLM family called Nemotron-Flash with two different model sizes.

Model configuration. We adopt the decoding-friendly model structure searched in Sec. [3.2](#S3.SS2 "3.2 SLM Design: Hybrid Operators ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), which interleaves DeltaNet-FFN-Mamba2-FFN (Block-1) and Attention-FFN-Mamba2-FFN (Block-2) as basic building blocks. We build two models Nemotron-Flash-1B/3B with 0.96B/2.7B parameters, respectively, with depth and width configured based on the scaling laws in Sec. [3.1](#S3.SS1 "3.1 SLM Design: Depth-Width Ratios ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"). Specifically, Nemotron-Flash-1B has the same configuration as in Tab. [2](#S3.T2 "Table 2 ‣ 3.2 SLM Design: Hybrid Operators ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), where the larger parameter count comes from a new tokenizer mentioned below. It has a hidden size of 2048 and contains 12 blocks, each with one token-mixing module and one FFN, or 24 operators if counting DeltaNet, Mamba2, Attention, and FFN as separate operators. Nemotron-Flash-3B has a hidden size of 3072 and contains 36 operators, with two additional Block-1 and one additional Block-2.

Tokenizer. Different from previous parameter-efficient SLMs [liu2024mobilellm] that adopt a tokenizer with a small vocabulary size to save parameters, we adopt a tokenizer with a larger vocabulary size [blakeman2025nemotron]. We find that the latency overhead of the enlarged embedding layer/LM head is small, while the more coarse-grained token representations reduce the token count when encoding the same sentence, resulting in more significant latency reduction (see Appendix [E](#A5 "Appendix E The Choice of Tokenizers ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models") for details).

Training settings. Both models are trained using the Adam optimizer (without weight decay, due to the use of weight normalization) and a cosine learning rate schedule with an initial learning rate of 1e-3. We first train the models on Zyda2 [tokpanov2024zyda], then switch to higher-quality datasets, including commonsense reasoning datasets (Climb-Mix [diao2025climb] and Smollm-corpus [benallal2024smollmcorpus]), a proprietary high-quality dataset with high proportions of math and code, and MegaMath [zhou2025megamath]. Both models are trained for 4.5T tokens using 256 NVIDIA H100 GPUs, with a batch size of 2M tokens and a context length of 4096, except for the final 25B tokens, where we extend the context length to 29000.

Deployment settings for latency measurement. To fairly benchmark against baseline models dominated by full attention layers, we adopt TensorRT-LLM’s AutoDeploy kernels [nvidia2025autodeploy] with efficient KV cache management for full attention and use CUDA Graph for further acceleration. For other operators, we use the official implementation of Mamba2 [dao2024transformers] and FlashLinearAttention [yang2024fla] for linear attention layers such as DeltaNet, and we always wrap the entire model in a CUDA Graph.

Baselines and tasks. We benchmark against SOTA SLMs, including Qwen3 [qwen3], Qwen2.5 [qwen2.5], Llama3.2 [grattafiori2024llama], SmolLM2 [allal2025smollm2smolgoesbig], h2o-Danube [singer2024h2o], and AMD-OLMo [AMD-OLMo] series. Accuracy is evaluated using lm-evaluation-harness [eval-harness] across 16 tasks including MMLU, commonsense reasoning (PIQA, ARCC, ARCE, Hellaswag, Winogrande, OBQA), math (GSM8k, MathQA), coding (HumanEval, HumanEval-Plus, MBPP, MBPP-Plus), and recall (FDA, SWDE, Squad). We use 5-shot evaluation for GSM8K and MMLU, 3-shot for MBPP and MBPP-Plus, and 0-shot for all remaining tasks. We report the average accuracy on each domain in Tab. [6](#S3.T6 "Table 6 ‣ 3.4 SLM Training: Meta Tokens ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models") and provide task-wise accuracy in Appendix [B](#A2 "Appendix B More Benchmark Results of Nemotron-Flash ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models").

### 4.1 Benchmark with SOTA Base SLMs

As shown in Tab. [6](#S3.T6 "Table 6 ‣ 3.4 SLM Training: Meta Tokens ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"),
our Nemotron-Flash family achieves the lowest decoding latency and the best accuracy among models of comparable size. For example, Nemotron-Flash-1B delivers 5.5% higher accuracy than Qwen3-0.6B with 1.9×\times latency reduction and 46×\times higher throughput. Similarly, Nemotron-Flash-3B achieves +2.0%/+5.5% higher average accuracy, 1.7×\times/1.3×\times latency reduction, and 6.4×\times/18.7×\times higher throughput than Qwen2.5-3B/Qwen3-1.7B, respectively. In addition, with further optimization of the attention configuration, Nemotron-Flash-3B-TP achieves 10.1×\times and 29.7×\times higher throughput compared to Qwen2.5-3B and Qwen3-1.7B, respectively.

It is worth noting that (1) in addition to achieving the most competitive latency and throughput, Nemotron-Flash-3B attains the highest accuracy in commonsense reasoning, math, coding, and recall tasks among models larger than 1.5B parameters; and (2) although Nemotron-Flash-1B and Nemotron-Flash-3B contain only 2 and 3 full-attention layers, respectively, both achieve the most competitive recall accuracy, suggesting that maintaining full KV cache across all layers is unnecessary, which is consistent with observations from existing hybrid LMs [blakeman2025nemotron, nano2025efficient].

![Refer to caption](/html/2511.18890/assets/x9.png)


Figure 9: Visualizing the NIAH scores of different attention configurations on the Ruler benchmark [hsieh2024ruler].




Table 8: Comparing Nemotron-Flash-3B variants with different attention configurations. Decoding throughput is measured with a 32k-token input length using the maximum batch size without OOM.

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Setting | Lat. (s) | Thr. (tok/sec) | MMLU | CR | Math | Coding | Recall | Avg |
| 3FA | 28.71 | 2939 | 61.19 | 61.02 | 57.62 | 53.33 | 73.25 | 60.98 |
| 2FA+1SWA | 28.06 | 3737 | 61.02 | 60.93 | 58.56 | 51.82 | 73.37 | 60.69 |
| 1FA+2SWA | 27.95 | 4657 | 61.19 | 60.94 | 57.88 | 52.40 | 73.77 | 60.84 |

### 4.2 Benchmark with SOTA Instruct SLMs

We instruction-tune the Nemotron-Flash-3B model using a two-stage supervised fine-tuning (SFT) strategy on two proprietary datasets.
The learning rates for the first and second stages are set to 8e-6 and 5e-6, respectively.
Each stage is trained for one epoch using a cosine learning rate scheduler and a global batch size of 384.
To accelerate training, we adopt the efficient packing strategy from prior work [tunstall2023zephyr, diao2024lmflow, dong2024rlhf], with a block size of 29,000 tokens.

We benchmark Nemotron-Flash-3B-Instruct against Qwen2.5-1.5B and Qwen3-1.7B across MMLU (5-shot), GPQA (0-shot), GSM8K (5-shot), and IFEval.
As shown in Table [7](#S3.T7 "Table 7 ‣ 3.4 SLM Training: Meta Tokens ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), Nemotron-Flash-3B-Instruct demonstrates strong reasoning and instruction-following capabilities, achieving the best average accuracy and efficiency, e.g., over +4.7% average accuracy and 4.3×\times/18.7×\times higher throughput compared to Qwen2.5-1.5B and Qwen3-1.7B, respectively.
Despite having over 1.6×\times more parameters, which contribute to enhanced intelligence, Nemotron-Flash maintains superior real-device efficiency, owing to its architectural improvements.

### 4.3 Ablation on Attention Configuration

Full attention (FA) operators are essential for long-context retrieval; however, they also become the primary bottleneck for large-batch-size long-context throughput. To better understand this trade-off, we conduct an ablation study on the attention configurations of Nemotron-Flash-3B. Starting from the pretrained Nemotron-Flash-3B base model with three FA layers, we perform continuous pretraining with a 29k context length for 25B tokens under three configurations: (1) three FA layers, (2) two FA layers plus one SWA layer with an 8k window size, and (3) one FA layer and two SWA layers.

We report the accuracy on both general benchmarks and three needle-in-a-haystack (NIAH) tasks from Ruler [hsieh2024ruler] in Tab. [8](#S4.T8 "Table 8 ‣ 4.1 Benchmark with SOTA Base SLMs ‣ 4 Nemotron-Flash: The New Hybrid SLM Family ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models") and Fig. [9](#S4.F9 "Figure 9 ‣ 4.1 Benchmark with SOTA Base SLMs ‣ 4 Nemotron-Flash: The New Hybrid SLM Family ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), respectively. The results show that (1) replacing more FA layers with SWA substantially improves throughput, e.g., the 1FA+2SWA configuration achieves 1.6×\times higher throughput than the 3FA setting; (2) general benchmark accuracy, including recall performance, remains largely unaffected when more SWA layers with 8k window size are used; and (3) NIAH performance, however, drops significantly at longer context lengths when the number of FA layers is reduced to one, highlighting the importance of FA operators for long-context capability. Therefore, we recommend maintaining at least two full attention layers even in SLMs.

## 5 Conclusion

Rather than merely offering a smaller LLM, this work re-imagines small models from the perspective of real-world latency and throughput, systematically exploring the key architectural and training factors essential for developing latency-optimal SLMs. By analyzing optimal depth–width ratios, strategically combining efficient attention operators through an evolutionary search framework, and enhancing training with weight normalization and meta tokens, we establish a comprehensive framework that significantly improves both real-device latency and accuracy, and deliver the Nemotron-Flash model family that advances the SOTA accuracy–latency frontier. Beyond the framework, we hope the actionable insights and guidelines provided here will inform future research and development of low-latency, high-throughput SLMs tailored to diverse, latency-sensitive real-world applications.

Table 9: We benchmark our Nemotron-Flash-1B/3B against SOTA SLMs across 16 tasks, including MMLU, commonsense reasoning (CR), math, coding, and recall tasks. Latency is measured on an NVIDIA H100 GPU for decoding 8k tokens with a batch size of 1 using CUDA Graph. Throughput is measured with a 32k-token input length using the maximum batch size w/o OOM.

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | MMLU | Commonsense Reasoning | | | | | | | Math | | | Coding | | | | | Recall | | | | Overall Avg |
| PIQA | ARCC | ARCE | Hellaswag | Winogrande | OBQA | Avg | GSM8k | MathQA | Avg | HumanEval | HumanEval-Plus | MBPP | MBPP-Plus | Avg | FDA | SWDE | Squad | Avg |
| AMD-OLMo-1B | 27.11 | 74.92 | 31.66 | 65.95 | 47.29 | 61.64 | 25.80 | 51.21 | 1.36 | 23.95 | 12.66 | 5.49 | 6.10 | 5.40 | 12.43 | 7.35 | 70.15 | 77.05 | 33.75 | 60.32 | 35.63 |
| Llama-3.2-1B | 31.06 | 74.37 | 31.06 | 65.45 | 47.71 | 59.91 | 26.40 | 50.82 | 5.31 | 29.18 | 17.24 | 17.07 | 14.02 | 26.60 | 38.89 | 24.15 | 74.14 | 84.25 | 37.84 | 65.41 | 41.45 |
| Qwen2.5-0.5B | 47.61 | 70.29 | 29.44 | 64.44 | 40.69 | 55.49 | 24.80 | 47.52 | 36.47 | 28.84 | 32.66 | 28.05 | 25.61 | 29.60 | 44.97 | 32.06 | 68.69 | 79.39 | 48.16 | 65.41 | 45.16 |
| Qwen3-0.6B | 52.44 | 69.69 | 33.36 | 65.65 | 41.06 | 58.88 | 24.80 | 48.91 | 42.84 | 30.92 | 36.88 | 19.51 | 17.07 | 22.60 | 38.10 | 24.32 | 64.88 | 80.29 | 43.60 | 62.92 | 44.11 |
| Nemotron-Flash-1B | 44.63 | 75.41 | 41.47 | 74.83 | 45.80 | 59.67 | 29.60 | 54.46 | 32.52 | 37.19 | 34.86 | 32.93 | 29.27 | 35.20 | 54.23 | 37.91 | 77.59 | 77.23 | 46.51 | 67.11 | 49.63 |
| H2O-Danube3-4B | 53.76 | 79.60 | 46.67 | 77.19 | 58.78 | 68.51 | 34.00 | 60.79 | 41.62 | 27.84 | 34.73 | 1.83 | 1.83 | 17.00 | 25.93 | 11.65 | 33.48 | 89.20 | 52.68 | 58.45 | 44.37 |
| SmolLM2-1.7B | 50.21 | 77.20 | 44.45 | 77.74 | 53.38 | 66.06 | 32.80 | 58.61 | 31.31 | 34.44 | 32.88 | 0.61 | 0.61 | 35.60 | 47.62 | 21.11 | 59.07 | 82.18 | 46.01 | 62.42 | 46.21 |
| Llama-3.2-3B | 56.30 | 76.44 | 42.49 | 74.37 | 55.31 | 69.06 | 31.00 | 58.11 | 25.40 | 34.74 | 30.07 | 26.22 | 23.78 | 36.80 | 52.38 | 34.80 | 78.49 | 89.38 | 42.16 | 70.01 | 50.89 |
| Qwen2.5-1.5B | 60.68 | 75.79 | 41.38 | 75.72 | 50.20 | 63.06 | 31.60 | 56.29 | 61.87 | 34.41 | 48.14 | 36.59 | 31.10 | 44.60 | 59.52 | 42.95 | 71.14 | 86.14 | 50.67 | 69.32 | 54.65 |
| Qwen3-1.7B | 62.46 | 75.89 | 41.64 | 73.23 | 49.39 | 63.93 | 39.20 | 57.21 | 69.60 | 37.82 | 53.71 | 39.63 | 36.59 | 43.00 | 55.82 | 43.76 | 65.43 | 85.33 | 48.49 | 66.42 | 55.47 |
| Qwen2.5-3B | 65.56 | 78.40 | 44.80 | 77.31 | 54.87 | 68.51 | 29.40 | 58.88 | 70.43 | 37.22 | 53.83 | 37.20 | 31.71 | 56.20 | 72.75 | 49.46 | 72.78 | 90.37 | 55.93 | 73.03 | 58.96 |
| Nemotron-Flash-3B | 61.19 | 79.65 | 50.17 | 80.72 | 53.79 | 67.01 | 34.80 | 61.02 | 66.34 | 48.91 | 57.62 | 48.17 | 43.29 | 52.00 | 69.84 | 53.33 | 81.85 | 85.42 | 52.48 | 73.25 | 60.98 |

## Appendix A Detailed Experimental Settings

Depth-width scaling in Sec. [3.1](#S3.SS1 "3.1 SLM Design: Depth-Width Ratios ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"). We train a series of Llama models with five depth settings: 6, 12, 18, 24, and 30 blocks, where each block contains one attention and one FFN. For each depth setting, we further vary the model width, i.e., hidden size, to create different models, ensuring that the resulting models across different depths have relatively comparable parameter ranges. We train each model on 100B tokens from the Smollm-corpus [benallal2024smollmcorpus] using the AdamW optimizer and a cosine learning rate schedule with an initial learning rate of 5e-4.

Architecture explorations in Sec. [3.2](#S3.SS2 "3.2 SLM Design: Hybrid Operators ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"). For vanilla Mamba and Mamba2 models, we do not insert FFNs, following their original papers. For all other pure models we explored, we add one FFN after each token mixing operator. For hybrid models, we adopt the building block of Operator1-Operator2-FFN. For example, when building a hybrid model with DeltaNet and Mamba2, we use the building block DeltaNet-Mamba2-FFN.
All models have 500M parameters and consist of 24 operators, where each token mixing operator and each FFN is counted as one operator. We train each model on 100B tokens from the Smollm-corpus [benallal2024smollmcorpus] using the AdamW optimizer and a cosine learning rate schedule with an initial learning rate of 5e-4.
The same training settings are used for training the searched architectures, as well as for the meta token experiments in Sec. [3.4](#S3.SS4 "3.4 SLM Training: Meta Tokens ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models").

Weight normalization experiments in Sec. [3.3](#S3.SS3 "3.3 SLM Training: Weight Normalization ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"). For all models with and without weight normalization, we train them on 100B tokens from the Smollm-corpus [benallal2024smollmcorpus] using the AdamW optimizer and a cosine learning rate schedule with an initial learning rate of 1e-3. This learning rate is the tuned learning rate that achieves the best convergence for models without weight normalization. We then apply weight normalization on top of this setting to demonstrate that our method can further boost task performance beyond the best baseline.

Tasks for computing average commonsense reasoning (CR) accuracy. Unless otherwise specified, in the above exploration experiments, we adopt eight tasks to compute the average CR accuracy: Lambda, PIQA, ARC-Easy, ARC-Challenge, Hellaswag, Winogrande, TruthfulQA, and SIQA, all from the lm-evaluation-harness [eval-harness].

Input: LUT for latency, population size PP, sample size SS, number of cycles CC, target latency budget Lt​a​r​g​e​tL\_{target}

Output: Best-performing architecture

1

2Initialize population 𝒫←\mathcal{P}\leftarrow Seeded architectures;

3
Short-train each architecture in 𝒫\mathcal{P} and compute their proxy PPL and latency;

4

5for *c​y​c​l​e←1cycle\leftarrow 1 to CC* do

6   
for *each architecture in current cycle in parallel* do

7      
Sample subset 𝒮⊆𝒫\mathcal{S}\subseteq\mathcal{P} randomly, with |𝒮|=S|\mathcal{S}|=S;

8      
Select the parent p←arg⁡minx∈𝒮⁡PPL​(x)p\leftarrow\arg\min\_{x\in\mathcal{S}}\text{PPL}(x) s.t. latency(x)≤Lt​a​r​g​e​t(x)\leq L\_{target};

9      
Create offspring cc by mutating one factor of pp (operator ratios, FFN ratios, or block type counts), adjusting to maintain depth constraint;

10      
Evaluate cc by short-training to obtain proxy PPL and querying LUT for latency;

11      
Remove oldest architecture from 𝒫\mathcal{P};

12      
Add offspring cc to population 𝒫\mathcal{P};

13

14

15return architecture in 𝒫\mathcal{P} with lowest proxy PPL satisfying latency constraint Lt​a​r​g​e​tL\_{target};

Algorithm 1 Aging Evolutionary Search for Hybrid Operator Combinations




Table 10: Benchmarking our searched architecture, optimized for parameter efficiency, against baseline architectures. All models have 500M parameters. CR accuracy is averaged over eight tasks.

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Metric | Attention | GLA | DeltaNet | Gated DeltaNet | RWKV | Mamba | Mamba2 | Searched (Ours) |
| Wiki PPL | 24.44 | 25.16 | 23.87 | 23.80 | 25.47 | 24.55 | 24.36 | 23.06 |
| CR Acc (%) | 48.02 | 46.82 | 47.83 | 47.96 | 46.46 | 47.32 | 47.72 | 49.23 |




Table 11: Architectural details of our Nemotron-Flash models. M2, A, D, and F denote Mamba2, attention, DeltaNet, and FFN, respectively. For attention layers, we use group-query attention with a group size of 4, with the specific number of heads detailed in the table.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Model | Hidden Size | FFN Dim | Attn/KV Heads | #Operators | Operators |
| Nemotron-Flash-1B | 2048 | 6144 | 16/4 | 24 | [D, F, M2, F, A, F, M2, F, D, F, M2, F, A, F, M2, F, D, F, M2, F, D, F, M2, F] |
| Nemotron-Flash-3B | 3072 | 9216 | 24/6 | 36 | [D, F, M2, F, A, F, M2, F, D, F, M2, F, A, F, M2, F, D, F, M2, F, A, F, M2, F, D, F, M2, F, D, F, M2, F, D, F, M2, F] |

## Appendix B More Benchmark Results of Nemotron-Flash

As a complement to the domain-averaged accuracies reported in Sec. [4.1](#S4.SS1 "4.1 Benchmark with SOTA Base SLMs ‣ 4 Nemotron-Flash: The New Hybrid SLM Family ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models") and Tab. [6](#S3.T6 "Table 6 ‣ 3.4 SLM Training: Meta Tokens ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), we present the detailed per-task accuracies within each domain in Tab. [9](#A0.T9 "Table 9 ‣ 5 Conclusion ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models").

## Appendix C Evolutionary Search: More Details and Experiments

### C.1 Detailed Search Space

Search template. We divide the entire architecture into three stages, with each stage repeating one type of building block. This three-stage strategy balances operator heterogeneity with architectural regularity. We search for both the number of blocks in each stage and the structure of each block, including the involved operators, their ratios, and the ratio of FFNs.

Operators and their ratios. Based on the exploration of pure and hybrid models in Sec. [3.2](#S3.SS2 "3.2 SLM Design: Hybrid Operators ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), we include attention, Mamba2, and DeltaNet in our search space, considering both operator efficiency and their complementary roles. To ensure regularity and avoid overly complex blocks, we allow at most two types of operators (excluding FFNs) in each stage. We support different ratios between two hybrid operators: 0:1 (i.e., only one operator), 1:1, 1:2, and 1:3.

FFN ratios. We allow for 0, 1, or 2 FFNs after each token mixing operator in each building block.

Number of building blocks. We allow flexible numbers of building blocks in each stage, as long as the total number of operators does not exceed a predefined limit (i.e., 30). If the total exceeds this limit, we reduce the number of building blocks in the last stage to meet the depth limit.

After determining all the above design factors for a new architecture, we select a hidden size from [1024, 1280, 1536, 1792, 2048, 2304, 2560], choosing the largest size that satisfies the target latency. This strategy allows us to use the short-training PPL solely to rank the searched architectures.

### C.2 Our Evolutionary Search Algorithm

As a complement to the search algorithm described in Sec. [3.2](#S3.SS2 "3.2 SLM Design: Hybrid Operators ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), we present our adopted aging evolutionary search in Alg. [1](#algorithm1 "In Appendix A Detailed Experimental Settings ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), which iteratively refines a population of hybrid language model architectures by selectively sampling and mutating architectures over multiple cycles.

Search process. After initializing a population of architectures from a set of seeded and randomly initialized architectures, a set of candidate architectures (we use 10) is sampled, mutated, and evaluated concurrently in each cycle. Mutations are performed by altering one of the searchable factors: (1) changing one operator in a building block type, (2) varying the hybrid operator ratio in a building block type, (3) adjusting the FFN ratio in a building block type, or (4) modifying the number of building blocks across stages. The sampled architectures are evaluated using short-training perplexity as an efficient proxy for final performance, and latency is quickly estimated using a pre-measured lookup table (LUT). This parallel, proxy-driven strategy effectively balances the exploration of novel designs and the exploitation of known strong performers, facilitating rapid convergence to high-quality architectures.

Short training settings.
We train each model on 10B tokens from the Smollm-corpus [benallal2024smollmcorpus] using the AdamW optimizer and a cosine learning rate schedule with an initial learning rate of 5e-4. Training and evaluating a sampled architecture takes approximately 2 hours using 32 NVIDIA A100 GPUs.

### C.3 Search with Parameter Efficiency as the Objective

As a complement to the searched architecture optimized for parameter efficiency, presented in Sec. [3.2](#S3.SS2 "3.2 SLM Design: Hybrid Operators ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models") and Tab. [3](#S3.T3 "Table 3 ‣ 3.2 SLM Design: Hybrid Operators ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), we provide a benchmark against other baseline architectures, all with 500M parameters, in terms of Wikitext PPL and CR accuracy averaged over eight tasks. All models are trained on 100B tokens from the Smollm-corpus [benallal2024smollmcorpus] using the AdamW optimizer and a cosine learning rate schedule with an initial learning rate of 5e-4.
As shown in Tab. [10](#A1.T10 "Table 10 ‣ Appendix A Detailed Experimental Settings ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), the searched architecture achieves over 1.21% higher CR accuracy and a reduction of more than 0.74 in PPL compared to all 500M baselines.

### C.4 Configurations of Our Final Nemotron-Flash Models

We provide the detailed configurations of our final Nemotron-Flash models in Tab. [11](#A1.T11 "Table 11 ‣ Appendix A Detailed Experimental Settings ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models").

Table 12: Comparing the achieved task accuracy and training speed using vanilla models, weight normalization, nGPT without weight normalization, and full nGPT.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Model | Setting | Wiki PPL | CR Acc (%) | Training Time (sec.) / Iter |
| LLaMA 1B | vanilla | 18.67 | 53.81 | 0.43 |
| wnorm | 18.03 | 54.85 | 0.46 |
| nGPT w/o wnorm | 19.38 | 51.83 | 0.56 |
| nGPT | 17.45 | 54.73 | 0.59 |
| Hybrid 1B | vanilla | 17.96 | 54.36 | 0.78 |
| wnorm | 17.50 | 55.74 | 0.81 |
| nGPT w/o wnorm | 18.96 | 52.44 | 1.01 |
| nGPT | 17.25 | 55.33 | 1.04 |

![Refer to caption](/html/2511.18890/assets/x10.png)


Figure 10:  The gradient norm evolution of different models during training w/ and w/o weight normalization.

## Appendix D Weight Normalization: More Analysis and Experiments

### D.1 Weight Normalization vs. nGPT

As discussed in Sec. [3.3](#S3.SS3 "3.3 SLM Training: Weight Normalization ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), our weight normalization technique can be viewed as a simplified and efficient variant of nGPT [loshchilov2024ngpt], which enforces all computations in LMs to operate on a unit sphere by introducing multiple activation normalization layers in addition to weight normalization.
To analyze the performance breakdown achieved by nGPT, we evaluate our weight normalization, nGPT without weight normalization (i.e., activation normalization only), and full nGPT on two models: a Llama 1B model and a 1B hybrid model with each building block structured as Attention-Mamba-FFN.

As shown in Tab. [12](#A3.T12 "Table 12 ‣ C.4 Configurations of Our Final Nemotron-Flash Models ‣ Appendix C Evolutionary Search: More Details and Experiments ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), we observe that:
(1) Using only our weight normalization achieves strong task performance, with higher CR accuracy than nGPT and slightly worse language modeling PPL;
(2) Weight normalization is essential to nGPT, as removing it leads to a notable drop in accuracy;
(3) Additional activation normalization layers in nGPT introduce over 30% training overhead, reducing the number of training tokens in a fixed training budget.
Thus, our weight normalization offers a more efficient alternative to nGPT while maintaining comparable accuracy.

### D.2 Gradient Norm Evolution

As a complement to Sec. [3.3](#S3.SS3 "3.3 SLM Training: Weight Normalization ‣ 3 Towards Latency-Optimal SLM Design and Training ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), we further visualize the gradient norm evolution during training with and without weight normalization for four models in Fig. [10](#A3.F10 "Figure 10 ‣ C.4 Configurations of Our Final Nemotron-Flash Models ‣ Appendix C Evolutionary Search: More Details and Experiments ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"). We observe that weight normalization results in a slightly increased gradient norm, leading to larger relative weight updates, considering the smaller weight norm. This effect is particularly beneficial in the later stages of training and leads to consistent final accuracy improvements across model families.

## Appendix E The Choice of Tokenizers

Previous parameter-efficient SLMs [liu2024mobilellm] adopt a tokenizer with a small vocabulary size, such as Llama2 [touvron2023llama], to reduce the number of parameters. In contrast, we use a tokenizer with a larger vocabulary size, i.e., Mistral-NeMo-Minitron [blakeman2025nemotron]. The rationale is that a larger vocabulary often results in more concise token representations and fewer tokens when encoding the same sentence. For example, the phrase by the way” can be tokenized into a single by-the-way” token.

Table 13: Tokens per sample on AG News / Wikitext datasets using LLaMA2 [touvron2023llama] and Mistral-NeMo-Minitron [blakeman2025nemotron] tokenizers.

|  |  |  |
| --- | --- | --- |
| Dataset | LLaMA2 | Mistral-NeMo-Minitron |
| AG News | 63.64 | 55.08 (-13.5%) |
| Wikitext | 116.86 | 106.00 (-9.3%) |

As shown in Tab. [13](#A5.T13 "Table 13 ‣ Appendix E The Choice of Tokenizers ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), when averaging the number of tokens per sample over two datasets, the Mistral-NeMo-Minitron tokenizer achieves a 9.3% and 13.5% reduction in token count. At the same time, the decoding latency for generating 8k tokens using models with embedding dimensions corresponding to the two tokenizers shows only a 5.8% latency gap (38.93 seconds vs. 41.31 seconds). This suggests that overall efficiency could be higher when using tokenizers with larger vocabularies. Furthermore, given that larger vocabulary sizes often improve task performance, we adopt the Mistral-NeMo-Minitron tokenizer [blakeman2025nemotron] over smaller alternatives.

Table 14: Decoding latency (sec.) at various decoding lengths for a 550M Llama model across different deployment flows. The batch size is set to 1, as our target is edge applications.

|  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Deployment Flow | 6 | 64 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
| PyTorch | 0.16 | 0.96 | 3.64 | 7.07 | 14.28 | 28.89 | 56.92 | 113.93 | 222.78 | 457.72 |
| vLLM | 0.02 | 0.14 | 0.55 | 1.09 | 2.16 | 4.31 | 8.66 | 17.82 | 67.23 | 169.74 |
| TensorRT-LLM | 0.17 | 0.30 | 0.64 | 1.00 | 1.81 | 3.42 | 6.72 | 13.65 | 28.41 | 60.41 |
| Ours | 0.01 | 0.14 | 0.54 | 1.08 | 2.17 | 4.36 | 8.72 | 17.54 | 35.29 | 71.65 |

## Appendix F Deployment Flow

Existing deployment frameworks such as TensorRT-LLM [nvidia\_tensorrt\_llm] and vLLM [kwon2023efficient] support Attention and Mamba2, while linear attention variants like DeltaNet are only supported by FlashLinearAttention [yang2024fla]. Hybrid LMs involve diverse operators and require a deployment flow that can efficiently accelerate each operator type. To this end, we integrate TensorRT-LLM’s AutoDeploy kernels [nvidia2025autodeploy] with efficient KV cache management for full Attention, the official implementation of Mamba2 [dao2024transformers], and DeltaNet via FlashLinearAttention [yang2024fla] to deploy our Nemotron-Flash. The entire model is wrapped with CUDA Graph to mitigate kernel launch overhead, which is critical for edge applications with a batch size of 1.

To demonstrate the effectiveness of our deployment flow, we benchmark the decoding speed of a Llama model (12 blocks, hidden size of 1536, 550M parameters) using PyTorch, vLLM, TensorRT-LLM, and our deployment flow. As shown in Tab. [14](#A5.T14 "Table 14 ‣ Appendix E The Choice of Tokenizers ‣ Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models"), our deployment flow outperforms vLLM in decoding speed at longer sequence lengths and achieves a latency gap within 15% of TensorRT-LLM for full-attention models. This gap could be further reduced with improved full-attention kernel optimization on top of [nvidia2025autodeploy].
Additionally, our deployment supports linear attention variants, which are not supported by vLLM or TensorRT-LLM.
This set of comparisons validates the effectiveness of our deployment flow.

## Appendix G Limitations and Future Work

In our architecture search for SLMs, we primarily use language modeling as the search proxy, which may not fully capture other capabilities such as long-context understanding.
Additionally, our search focuses on macro-architecture, leaving room to explore more fine-grained factors. We will extend our search framework to optimize for these aspects in future work.

[◄](/html/2511.18889)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2511.18890)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2511.18890)
[View original  
on arXiv](https://arxiv.org/abs/2511.18890)[►](/html/2511.18891)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Fri Dec 5 16:14:35 2025 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
