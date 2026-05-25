---
arxiv: '2310.01334'
authors:
- Pingzhi Li
- Zhenyu Zhang
- Prateek Yadav
- Yi-Lin Sung
- Yu Cheng
- Mohit Bansal
- Tianlong Chen
parser: ar5iv
retrieved: '2026-05-25'
source: paper
title: 'Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing
  Policy'
url: https://arxiv.org/abs/2310.01334
year: 2023
---

[2310.01334] Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy



# Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy

Pingzhi Li1  Zhenyu Zhang2  Prateek Yadav3  Yi-Lin Sung3
  
Yu Cheng4  Mohit Bansal3  Tianlong Chen3,5,6
  
1University of Science and Technology of China  2University of Texas at Austin  5MIT
  
3The University of North Carolina at Chapel Hill  4Rice University  6Harvard University

###### Abstract

Sparsely activated Mixture-of-Experts (SMoE) has shown promise to scale up the learning capacity of neural networks, however, they have issues like: (a𝑎a) High Memory Usage, due to duplication of the network layers into multiple copies as experts; and (b𝑏b) Redundancy in Experts, as common learning-based routing policies suffer from representational collapse. Therefore, vanilla SMoE models are memory inefficient and non-scalable, especially for resource-constrained downstream scenarios. In this paper, we ask: Can we craft a compact SMoE model by consolidating expert information? What is the best recipe to merge multiple experts into fewer but more knowledgeable experts? Our pilot investigation reveals that conventional model merging methods fail to be effective in such expert merging for SMoE. The potential reasons are: (111) redundant information overshadows critical experts; (222) appropriate neuron permutation for each expert is missing to bring all of them in alignment. To address these challenges, we propose a novel merging algorithm for SMoE, i.e., M-SMoE, which leverages routing statistics to guide expert merging. Specifically, it starts with neuron permutation alignment for experts; then, dominant experts and their “group members” are formed based on routing policies; lastly, every expert group is merged into a single expert by utilizing each expert’s activation frequency as their weight for merging, thus diminishing the impact of insignificant experts. Moreover, we draw an interesting observation that our proposed merging promotes a low dimensionality in the merged expert’s weight space, naturally paving the way for additional compression. Hence, our final method, MC-SMoE (i.e., Merge, then Compress SMoE), further decomposes the merged experts into low-rank and structural sparse alternatives. Extensive experiments across 888 benchmarks validate the effectiveness of our proposals. For instance, our MC-SMoE achieves up to 80%percent8080\% memory and a 20%percent2020\% FLOPs reduction, with virtually no loss in performance. Our code is provided at <https://github.com/UNITES-Lab/MC-SMoE>.

## 1 Introduction

Figure 1: Accuracy (%percent\%) on the COPA with the switch-base-32 SMoE. MC-SMoE reaches up to an 80%percent8080\% memory saving with only a negligible compromise in performance.

Transformers (Vaswani et al., [2023](#bib.bib58)) have become the de facto network architecture in various natural language processing (NLP) scenarios (Devlin et al., [2019](#bib.bib12); Yang et al., [2019](#bib.bib66); Liu et al., [2019](#bib.bib38); Raffel et al., [2020](#bib.bib49); Fedus et al., [2022](#bib.bib18); Wei et al., [2022](#bib.bib61)), and even for computer vision applications (Dosovitskiy et al., [2021](#bib.bib15); Touvron et al., [2021](#bib.bib57); Mao et al., [2022](#bib.bib41); Zheng et al., [2021](#bib.bib67); Liu et al., [2021](#bib.bib39)). Nowadays, the parameter counts of such models are commonly measured in billions rather than millions. It is mainly because certain empirical scaling laws (Kaplan et al., [2020](#bib.bib33)) reveal a power-law relationship between the final model quality and the amount of {data, model capacity, and computing time}. Unfortunately, it poses infeasible requirements for computational resources, e.g., training a GPT-based model (Brown et al., [2020](#bib.bib3)) typically leads to thousands of GPU days. Sparse Mixture-of-Experts (SMoE) (Shazeer et al., [2017](#bib.bib54)) was then proposed to trim down the computing cost while enabling efficient scaling of network capacity. For predictions of a given input, it leverages input-dependent conditional computation to sparsely activate (i.e., routing) the relevant model pieces (i.e., experts). Hence, the network parameter counts/capacity can be amplified with minimal extra training cost. For instance, Fedus et al. ([2022](#bib.bib18)) scales the T5-Base (Raffel et al., [2020](#bib.bib49)) dense model to a 35×35\times larger Switch-Base SMoE model, with roughly the same training FLOPS.

However, several crucial limitations persist in SMoE for expanding the capacity of large language models. Firstly, SMoE trades space for FLOPs111FLOPs means the floating point operations per second. Note that the vanilla design of SMoE does not necessarily bring running time benefits. Instead, to mitigate the extra latency costs from routing and diverse experts, it usually requires specialized parallelism (Rajbhandari et al., [2022](#bib.bib50); Fedus et al., [2022](#bib.bib18); He et al., [2021](#bib.bib26); [2022](#bib.bib27)) and hardware designs (Fan et al., [2022](#bib.bib17))., which introduces substantial memory overheads and constrains its practical usage in real-world resource-restricted platforms, especially for downstream deployment and inference. Secondly, SMoE has a poor utilization of its capacity. The prevalent learning-based routing policy in SMoE suffers from representation collapse issues, since it encourages token embeddings to be clustered around expert centroids (Chi et al., [2022](#bib.bib8)) and results in redundant experts (Mittal et al., [2022](#bib.bib44); Chen et al., [2022](#bib.bib6)). A recent investigation (Chen et al., [2023](#bib.bib5)) also points out a similar observation that the “effective capacity” in conventional SMoEs is low. To address these drawbacks and fully unleash the power of SMoE, one possible solution is consolidating information from insignificant experts, aiming to establish a more compact SMoE without hurting performance. Nevertheless, naively combining existing model merging mechanisms leads to substandard results in the SMoE scenarios, as demonstrated in our pilot studies in Section [4.2](#S4.SS2 "4.2 Competitive Performance and Superior Efficiency of MC-SMoE ‣ 4 Experiments ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy"). The potential reasons could be: ① Critical experts are prone to be overshadowed by redundant information during merging, ② Experts are usually initialized and trained along with diverse optimization trajectories, thus an expert permutation can play an essential role in bringing them into alignment (Ainsworth et al., [2022](#bib.bib1)). These primary challenges drive us to ask:

(Q) How to effectively consolidate the redundant experts of SMoE into a selected few ones without sacrificing vital knowledge?

In this paper, we systematically investigate the above research question (Q), and target a compact and high-quality SMoE on downstream fine-tuning/inference scenarios. We discover that the routing policies from SMoE contain the “clues” for effective expert merging. To be specific, (111) the activation frequency of experts indicates its utilization and can be regarded as a great proxy for its importance. It enables an automatic way to determine how many and which experts should be kept in each SMoE layer; (222) The routing decision measures how similar are the experts to each other, in terms of the relevance to given input samples. It helps in associating redundant experts with different dominant experts. Based on these insights, we proposed a novel M-SMoE method for SMoE merging. Furthermore, we find that the merged experts from M-SMoE lie in a low dimensional parameter space, which seems to suggest that an appropriate merging reduces the potential noisy weight signals (Han et al., [2016](#bib.bib25)). We utilize this additional benefit of expert merging to design our MC-SMoE (Merge, then Compress SMoE) method that organically integrates low-rank decomposition techniques for further expert compression. Our main contributions are as follows:

* •

  We propose a novel framework MC-SMoE, i.e., Merge, then Compress SMoE, for SMoE efficiency at the downstream scenarios, including fine-tuning and zero-shot evaluation.
* •

  We design an innovative merging approach (M-SMoE) based on the guidance from routing policies. Specifically, it begins with a customized permutation alignment for experts, then identifies the dominant experts globally along with their “group members” within SMoE layers, and concludes with a weighted averaging according to their activated frequency.
* •

  We observe that resultant experts from M-SMoE inherently exhibit a lower weight dimensionality. This interesting phenomenon paves the way for additional compression, enabling our MC-SMoE method to further boost memory and parameter efficiency.
* •

  Extensive experiments across eight benchmarks validate the effectiveness of our MC-SMoE. An example is presented in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy"). Notably, M-SMoE yields up to a 𝟔𝟎%percent60\bm{60\%} reduction in memory overhead with even slightly improved performance. MC-SMoE achieves up to 𝟖𝟎%percent80\bm{80\%} memory and 𝟐𝟎%percent20\bm{20\%} FLOPs reduction, with only marginal performance drops.

## 2 Related Works

Figure 2: The overview of our proposed MC-SMoE pipeline. (a) In the conventional SMoE, each token embedding is directed to a small number of relevant experts. (b) The routing policy inspires expert merging. Across all SMoE layers, M-SMoE identifies the most frequently activated experts as dominant ones, groups the other non-dominant experts, and then merges them within each group in a frequency-weighted fashion. (c) After merging, the weight space of resulted experts tends to exhibit lower dimensionality, paving the way for additional compression. It clarifies the design of our MC-SMoE.

Sparse Mixture-of-Experts (SMoE). The benefits of scaling model size are widely acknowledged, which usually offers increased learning capacity and enhanced generalization (Brown et al., [2020](#bib.bib3); Kaplan et al., [2020](#bib.bib33); Chung et al., [2022](#bib.bib11); Chowdhery et al., [2022](#bib.bib10)). SMoE is an efficient approach to train larger models with negligible additional overhead, which has been broadly studied in Shazeer et al. ([2017](#bib.bib54)); Lepikhin et al. ([2021](#bib.bib36)); Fedus et al. ([2022](#bib.bib18)). SMoE models activate different pieces of the model for different input tokens as opposed to utilizing the full network parameters. For instance, GShard (Lepikhin et al., [2021](#bib.bib36)), an SMoE model scales up a Transformer-based model from 222B to 600600600B parameters with training cost being lower than a 100100100B dense model. Recently, Fedus et al. ([2022](#bib.bib18)) created a T5 (Raffel et al., [2020](#bib.bib49)) based SMoE model with trillion parameters.

Efficiency Concerns in SMoE and Existing Solutions. SMoE models require huge memory to host experts, moreover, many experts have low utilization during inference. To address this, Chen et al. ([2022](#bib.bib6)); Kim et al. ([2021](#bib.bib35)) prune experts based on their utilization to save memory, however, this leads to lower performance. In contrast, Gao et al. ([2022](#bib.bib21)) uses a tensor decomposition method to share the central tensor’s parameters across experts and keep different auxiliary tensors for each expert. Moreover, some works employ knowledge distillation (KD)  (Rajbhandari et al., [2022](#bib.bib50); Artetxe et al., [2022](#bib.bib2); Fedus et al., [2022](#bib.bib18)) to create either a smaller dense model or SMoE model with fewer layers. However, they also overlook the existing redundancy within SMoE layers.

Model Merging in Language Models.
The abundance of open-source models necessitates harnessing these existing models to create superior ones. Network ensembling (Zhu et al., [2019](#bib.bib68); Ortega et al., [2022](#bib.bib47)) emerges as an intuitive solution, however, its computational burden during inference increases proportionally with the inclusion of more models. Recent literature has increasingly emphasized the concept of model merging (Yadav et al., [2023](#bib.bib63); Cai et al., [2023](#bib.bib4); Ilharco et al., [2022b](#bib.bib29); Matena & Raffel, [2022](#bib.bib42); Jin et al., [2022](#bib.bib30); Don-Yehiya et al., [2022](#bib.bib14); Rame et al., [2023](#bib.bib52)). Yet, most of these studies assume that the merged models originate from the same initialization (Yadav et al., [2023](#bib.bib63); Ilharco et al., [2022a](#bib.bib28); Wortsman et al., [2022](#bib.bib62)), narrowing the pool of potential source models suitable for merging. However, this assumption might not be applicable to SMoE models. Typically, different experts within SMoE start with distinct random parameter initializations, and each expert is optimized with only a subset of the training data, as determined by the routing networks. These characteristics make the task of merging experts in SMoE more challenging.

To tackle these challenges, numerous investigations resort to mode connectivity (Draxler et al., [2018](#bib.bib16); Frankle et al., [2020](#bib.bib19); Freeman & Bruna, [2016](#bib.bib20); Garipov et al., [2018](#bib.bib22)) as a metric to measure the intricacy of merging between two experts. The underlying premise is that models within the same loss basin are mergeable. Additionally, some works employ permutation invariance (Ainsworth et al., [2022](#bib.bib1); Jordan et al., [2022](#bib.bib32); Peña et al., [2023](#bib.bib48)) to transfer models in different error basins into the same one without affecting their functionality. Jolicoeur-Martineau et al. ([2023](#bib.bib31)) applies regularization terms during training to enhance the mergeability of models, and Gueta et al. ([2023](#bib.bib24)) systematically analyzes how training tasks, datasets, and recipes influence the difficulty of merging. A concurrent work, SMEAR (Muqeeth et al., [2023](#bib.bib45)) dynamically merges various experts into a single one during the training process to avoid discrete routing. Note that this approach doesn’t offer any memory reduction and necessitates retaining the whole SMoE during inference.

## 3 Methodology

In this section, we present the details of our proposed MC-SMoE method. Section [3.1](#S3.SS1 "3.1 Routing Policy Guides Experts Merging ‣ 3 Methodology ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy") introduces the expert merging technique M-SMoE and how it is guided by the routing policy. In Section [3.2](#S3.SS2 "3.2 Merging Encourages Expert Decomposition ‣ 3 Methodology ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy"), we illustrate the extra benefit of merged experts and how it leads to further compression. The whole procedure of MC-SMoE is provided at the end in Algorithm [1](#alg1 "Algorithm 1 ‣ Post-Merging Compression of MC-SMoE. ‣ 3.2 Merging Encourages Expert Decomposition ‣ 3 Methodology ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy").

Figure 3: Distribution of expert activation frequencies in the switch-base-32 model, encompassing 121212 SMoE layers with 323232 experts per layer. The top of the heatmap is the first MoE layer while the bottom is the last. The left two tasks, COPA and SQuAD, are characterized by answer-generation prompts. The right two tasks, WikiQA and SST2, are typified by answer-selection prompts. SMoE models fine-tuned on answer-selection tasks demonstrate a more skewed distribution in their transformer decoder layers, wherein a significant portion of experts remain inactivated all the time.

### 3.1 Routing Policy Guides Experts Merging

#### Experts Permutation Alignment.

Our M-SMoE method begins with the alignment of expert weight permutations since merging without it could potentially lead to the inferior fusion of mismatched neurons. In our case, the target experts operate in the same input-output space, which makes the merging more feasible. The experts are 222-layer feed-forward networks, where 𝚆insubscript𝚆in\mathtt{W}\_{\text{in}} and 𝚆outsubscript𝚆out\mathtt{W}\_{\text{out}} denote two weight matrices of input and output layers, respectively. 𝒙𝒙\bm{x} is the input vector and act​(⋅)act⋅\texttt{act}(\cdot) represents the activation function. Then, a feed-forward network is defined as a mapping ℱ:𝒙→𝚆out​(act​(𝚆in​𝒙)):ℱ→𝒙subscript𝚆outactsubscript𝚆in𝒙\mathcal{F}:\bm{x}\to\mathtt{W}\_{\text{out}}(\texttt{act}(\mathtt{W}\_{\text{in}}\bm{x})). Ainsworth et al. ([2022](#bib.bib1)) tells us that for any arbitrary permutation matrix 𝙿𝙿\mathtt{P}, the following equation 𝚆out​(act​(𝚆in​𝒙))=𝚆out​𝙿T​(act​(𝙿𝚆in​𝒙))subscript𝚆outactsubscript𝚆in𝒙subscript𝚆outsuperscript𝙿Tactsubscript𝙿𝚆in𝒙\mathtt{W}\_{\text{out}}(\texttt{act}(\mathtt{W}\_{\text{in}}\bm{x}))=\mathtt{W}\_{\text{out}}\mathtt{P}^{\text{T}}(\texttt{act}(\mathtt{P}\mathtt{W}\_{\text{in}}\bm{x})) always holds. In other words, 𝙿𝙿\mathtt{P} preserves the function ℱℱ\mathcal{F}.

We follow the weight matching optimization in Ainsworth et al. ([2022](#bib.bib1)) to align experts without altering their functionalities. For example, given two experts 𝙴isubscript𝙴𝑖\mathtt{E}\_{i} and 𝙴jsubscript𝙴𝑗\mathtt{E}\_{j} with weight matrices 𝚆isubscript𝚆𝑖\mathtt{W}\_{i} and 𝚆jsubscript𝚆𝑗\mathtt{W}\_{j}, it try to locate the optimal 𝙿isubscript𝙿𝑖\mathtt{P}\_{i} and 𝙿jsubscript𝙿𝑗\mathtt{P}\_{j} by minimizing the ℓ2subscriptℓ2\ell\_{2} distance between their corresponding permutated weights 𝚆i′subscriptsuperscript𝚆′𝑖\mathtt{W}^{\prime}\_{i} and 𝚆j′subscriptsuperscript𝚆′𝑗\mathtt{W}^{\prime}\_{j}. This process provides a beneficial first step for merging.

#### Routing Policies Reflect the Expert Similarity.

One of the main challenges in SMoE expert merging comes from the expert specialization (Mittal et al., [2022](#bib.bib44)) cultivated during the joint training of experts and routers. Although representation collapse happens (Chi et al., [2022](#bib.bib8)) and massive redundancies exist among experts, Figure [3](#S3.F3 "Figure 3 ‣ 3 Methodology ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy") demonstrates that the utilization of several (more than one) experts is significantly larger compared to the rest. Therefore, it is challenging to merge all experts within an SMoE layer into a single dense expert. Instead, we divide them into multiple groups based on their similarity, and keep all dominant (most used) experts to preserve the performance. To meet the goal, our M-SMoE method exploits the implicit guidance from SMoE’s routing policy: (111) Similar rows (output channel) in a router weight matrix tend to feed similar input tokens to their corresponding experts, pushing these experts to be trained in a similar fashion; (222) Intuitively, experts that are similar tend to exhibit similar router logits across the majority of input tokens. Based on this, we can either use the rows in a router weight matrix or the router logits vector derived from a batch of input tokens, to measure expert similarity. Detailed comparisons are provided in Section [4.3](#S4.SS3 "4.3 Ablation Study and Extra Investigation ‣ 4 Experiments ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy") and we describe the superior one here, i.e., router logits, and leave the other to Appendix [A2](#A2 "Appendix A2 More Technique Details ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy"). Specifically, the similarity Sim​(⋅,⋅)Sim⋅⋅\texttt{Sim}(\cdot,\cdot) between experts 𝙴isubscript𝙴𝑖\mathtt{E}\_{i} and 𝙴jsubscript𝙴𝑗\mathtt{E}\_{j} in an SMoE layer is computed by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝙷=𝚆r​(𝚇T),Sim​(𝙴i,𝙴j)=cosine​(𝙷i,∗,𝙷j,∗),formulae-sequence𝙷subscript𝚆𝑟superscript𝚇TSimsubscript𝙴𝑖subscript𝙴𝑗cosinesubscript𝙷  𝑖subscript𝙷  𝑗\displaystyle\mathtt{H}=\mathtt{W}\_{r}(\mathtt{X}^{\mathrm{T}}),\ \texttt{Sim}(\mathtt{E}\_{i},\mathtt{E}\_{j})=\texttt{cosine}(\mathtt{H}\_{i,\*},\mathtt{H}\_{j,\*}), |  | (1) |

where 𝚇𝚇\mathtt{X} is an input embedding, 𝚆rsubscript𝚆𝑟\mathtt{W}\_{r} is the router weight, 𝙷i,∗subscript𝙷

𝑖\mathtt{H}\_{i,\*} and 𝙷j,∗subscript𝙷

𝑗\mathtt{H}\_{j,\*} are row vectors in logits 𝙷𝙷\mathtt{H}.

#### Dominant Experts, Expert Grouping, and Frequency-Based Merging.

Based on the expert utilization as depicted in Figure [3](#S3.F3 "Figure 3 ‣ 3 Methodology ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy"), we first treat the most commonly active experts as dominant experts. Such expert utilization is calculated by inputting and routing a randomly picked subset of training data. Then, as demonstrated in Figure [2](#S2.F2 "Figure 2 ‣ 2 Related Works ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy") (b𝑏b), each non-dominant expert gravitates toward and joins the group led by its most similar dominant expert, using the similarity function defined by Equation [1](#S3.E1 "In Routing Policies Reflect the Expert Similarity. ‣ 3.1 Routing Policy Guides Experts Merging ‣ 3 Methodology ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy"). After grouping, each group consists of a few non-dominant and one dominant expert. Lastly, for a group of k𝑘k experts {𝙴1,⋯,𝙴k}subscript𝙴1⋯subscript𝙴𝑘\{\mathtt{E}\_{1},\cdots,\mathtt{E}\_{k}\}, a frequency-based merging is performed as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝙴merged=∑i=1kαi​𝙴i∑i=1kαi,subscript𝙴mergedsuperscriptsubscript𝑖1𝑘subscript𝛼𝑖subscript𝙴𝑖superscriptsubscript𝑖1𝑘subscript𝛼𝑖\displaystyle\mathtt{E}\_{\text{merged}}=\frac{\sum\_{i=1}^{k}\alpha\_{i}\mathtt{E}\_{i}}{\sum\_{i=1}^{k}\alpha\_{i}}, |  | (2) |

where αisubscript𝛼𝑖\alpha\_{i} is the usage frequency of expert 𝙴isubscript𝙴𝑖\mathtt{E}\_{i}. The superiority of emphasizing the dominant experts is detailed and validated in our ablation study (Section [4.3](#S4.SS3 "4.3 Ablation Study and Extra Investigation ‣ 4 Experiments ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy")).

#### Adaptive Layer-Wise Merging Ratio.

As shown in Figure [3](#S3.F3 "Figure 3 ‣ 3 Methodology ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy"), the activated frequency of each expert varies across different SMoE layers, suggesting a diverse number of dominant experts and corresponding groups. To consider this phenomenon, we normalize the frequencies within each SMoE layer and select the dominant experts in a global manner across all layers222To ensure computational stability, we adjust the frequency of the most active expert in each SMoE layer to 1.01.01.0. In this way, at least one expert will be labeled as dominant. However, our experiments show that there are always at least two dominant experts in each SMoE layer.. Take an extreme case as an example, if the expert routing is uniform in one SMoE layer, then all experts will be treated as dominant ones, echoing our intuitions.

### 3.2 Merging Encourages Expert Decomposition

Figure 4: Experts are more compressible after merging. We calculate the average stable-rank change ratio (after−beforebeforeafterbeforebefore\frac{\text{after}-\text{before}}{\text{before}}) of all experts within each layer of the switch-base-32 SMoE model, reflecting the difference before and after merging. These mostly negative values throughout the SMoE layers emphasize a lower dimensionality achieved through the merging process.

#### Merging Encourages Low-Rank Weights.

We observe that M-SMoE promotes a lower dimensionality in the weight space of merged experts, naturally facilitating additional compression. We adopt the metric from Wang et al. ([2023](#bib.bib60)) to measure the rank of weight spaces. This metric has proved to be practical as it primarily remains unswayed by minuscule singular values, providing a rank estimation for the weight matrix 𝚆𝚆\mathtt{W} from a network layer. It is defined below:

|  |  |  |  |
| --- | --- | --- | --- |
|  | stable-rank​(𝝈)=Σi​𝝈i2max⁡𝝈i2,stable-rank𝝈subscriptΣ𝑖superscriptsubscript𝝈𝑖2superscriptsubscript𝝈𝑖2\displaystyle\texttt{stable-rank}(\bm{\sigma})=\frac{\Sigma\_{i}\bm{\sigma}\_{i}^{2}}{\max~{}\bm{\sigma}\_{i}^{2}}, |  | (3) |

where 𝝈𝝈\bm{\sigma} denotes the singular value vector of 𝚆𝚆\mathtt{W}. Figure [4](#S3.F4 "Figure 4 ‣ 3.2 Merging Encourages Expert Decomposition ‣ 3 Methodology ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy") showcases several stable-rank change ratio instances of SMoEs fine-tuned on various tasks. We measured the stable-rank’s change after merging by calculating the ratio of its difference to its initial value. We see that the averaged stable-rank change ratio of all experts is consistently non-positive, i.e. stable-rank decreases, over most of the SMoE layers, after merging. It inspires us to conduct post-merging compression, as illustrated in Figure [2](#S2.F2 "Figure 2 ‣ 2 Related Works ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy") (c𝑐c).

#### Post-Merging Compression of MC-SMoE.

To enjoy the extra benefits from merging, we tailor the previous SoTA decomposition methods (Chen et al., [2021](#bib.bib7); Li et al., [2023](#bib.bib37)) for SMoE, and propose an upgraded algorithm MC-SMoE for further memory and parameter efficiency. To be specific, the weight matrix 𝚆𝚆\mathtt{W} of a merged expert is decomposed into 𝚄𝚅+𝚂𝚄𝚅𝚂\mathtt{U}\mathtt{V}+\mathtt{S}. Here, the product of 𝚄∈ℝd1×r𝚄superscriptℝsubscript𝑑1𝑟\mathtt{U}\in\mathbb{R}^{d\_{1}\times r} and 𝚅∈ℝr×d2𝚅superscriptℝ𝑟subscript𝑑2\mathtt{V}\in\mathbb{R}^{r\times d\_{2}} represents a low-rank approximation, where r𝑟r is a much smaller rank compared to the full dimensionality of 𝚆𝚆\mathtt{W}. 𝚂𝚂\mathtt{S} contains the incoherent part of weights in 𝚆𝚆\mathtt{W}, and will be further pruned in a structural manner. An importance score of a weight si,jsubscript𝑠

𝑖𝑗s\_{i,j} is computed as ℐ​(si,j)=|si,j⋅∇si,jℒ|ℐsubscript𝑠

𝑖𝑗⋅subscript𝑠

𝑖𝑗subscript∇subscript𝑠

𝑖𝑗ℒ\mathcal{I}(s\_{i,j})=|s\_{i,j}\cdot\nabla\_{s\_{i,j}}\mathcal{L}|, where ℒℒ\mathcal{L} indicates the training objective of SMoEs. To trim down 𝚂𝚂\mathtt{S}, the weight columns with the lowest cumulative scores ∑iℐ​(si,j)subscript𝑖ℐsubscript𝑠

𝑖𝑗\sum\_{i}\mathcal{I}(s\_{i,j}) will be removed, which is determined across all 𝚂𝚂\mathtt{S} weights and naturally leads to a layer-wise adaptive compression ratio. As a summary, Algorithm [1](#alg1 "Algorithm 1 ‣ Post-Merging Compression of MC-SMoE. ‣ 3.2 Merging Encourages Expert Decomposition ‣ 3 Methodology ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy") presents the full procedures of our proposed MC-SMoE framework.

Algorithm 1  The Overall Procedures of MC-SMoE.

1:Initialize: A model ℳℳ\mathcal{M} with l𝑙l SMoE layers, training dataset 𝒯𝒯\mathcal{T} with b𝑏b tokens, the total number of original experts n𝑛n, and the number of the remaining experts k𝑘k.

2:Let 𝙷∈ℝl×b×n𝙷superscriptℝ𝑙𝑏𝑛\mathtt{H}\in\mathbb{R}^{l\times b\times n} and 𝙰∈ℝl×n𝙰superscriptℝ𝑙𝑛\mathtt{A}\in\mathbb{R}^{l\times n} denote the router logits and activated frequencies, respectively

3:Let 𝒟𝒟\mathcal{D} represents the set of dominant experts

4:𝙷,𝙰←forward​(ℳ,𝒯)←

𝙷𝙰
forwardℳ𝒯\mathtt{H},\mathtt{A}\leftarrow\texttt{forward}(\mathcal{M},\mathcal{T}); 𝒟←top​(k,row-normalize​(𝙰))←𝒟top𝑘row-normalize𝙰\mathcal{D}\leftarrow\texttt{top}\left(k,\texttt{row-normalize}(\mathtt{A})\right)

5:for layer t=1,…,l𝑡

1…𝑙t=1,\ldots,l do

6:     for expert i=2,…,nl𝑖

2…𝑛𝑙i=2,\ldots,\frac{n}{l} do

7:         𝙴it←weight-matching​(𝙴it,𝙴1t)←superscriptsubscript𝙴𝑖𝑡weight-matchingsuperscriptsubscript𝙴𝑖𝑡superscriptsubscript𝙴1𝑡\mathtt{E}\_{i}^{t}\leftarrow\texttt{weight-matching}(\mathtt{E}\_{i}^{t},\mathtt{E}\_{1}^{t})
▷▷\triangleright Expert Permutation Alignment

8:     end for

9:     𝒬​(i)≔argmaxj∈𝒟t​cosine​(𝙷t,∗,i,𝙷t,∗,j)≔𝒬𝑖subscriptargmax𝑗superscript𝒟𝑡cosinesubscript𝙷

𝑡𝑖subscript𝙷

𝑡𝑗\mathcal{Q}(i)\coloneqq\texttt{argmax}\_{j\in\mathcal{D}^{t}}\texttt{cosine}\left(\mathtt{H}\_{t,\*,i},\mathtt{H}\_{t,\*,j}\right)
▷▷\triangleright Group Label Assignment

10:     for d∈𝒟t𝑑superscript𝒟𝑡d\in\mathcal{D}^{t} do

11:         𝒢←{i∣𝒬(i)==d}\mathcal{G}\leftarrow\{i\mid\mathcal{Q}(i)==d\}; 𝙴dt←∑i∈𝒢𝙰t,i​𝙴it∑i∈𝒢𝙰t,i←superscriptsubscript𝙴𝑑𝑡subscript𝑖𝒢subscript𝙰

𝑡𝑖superscriptsubscript𝙴𝑖𝑡subscript𝑖𝒢subscript𝙰

𝑡𝑖\mathtt{E}\_{d}^{t}\leftarrow\frac{\sum\_{i\in\mathcal{G}}\mathtt{A}\_{t,i}\mathtt{E}\_{i}^{t}}{\sum\_{i\in\mathcal{G}}\mathtt{A}\_{t,i}}
▷▷\triangleright Merging based on Activated Frequencies

12:         𝙴dt→𝚄dt​𝚅dt+𝚂dt→superscriptsubscript𝙴𝑑𝑡superscriptsubscript𝚄𝑑𝑡superscriptsubscript𝚅𝑑𝑡superscriptsubscript𝚂𝑑𝑡\mathtt{E}\_{d}^{t}\to\mathtt{U}\_{d}^{t}\mathtt{V}\_{d}^{t}+\mathtt{S}\_{d}^{t}
▷▷\triangleright Then compress

13:     end for

14:     for i∉𝒟𝑖𝒟i\notin\mathcal{D} do

15:         Dropping 𝙴itsuperscriptsubscript𝙴𝑖𝑡\mathtt{E}\_{i}^{t} from ℳℳ\mathcal{M}

16:     end for

17:end for

18:Return: A compact SMoE produced from MC-SMoE.

## 4 Experiments

### 4.1 Implementation Details

#### Datasets and Network Backbones.

Table 1: Two SMoE models and their corresponding dense model checkpoints. act-size: number of activated parameters for each token, size: total number of parameters, l: the number of transformer layers, h: hidden dimension, e: the number of number of experts, arch: the type of transformer architecture.

| Model Identifier | act-size | size | l | h | e | arch |
| --- | --- | --- | --- | --- | --- | --- |
| t5-base | 220220220M | 220220220M | 121212 | 768768768 | 111 | enc-dec |
| switch-base-32 | 220220220M | 2.02.02.0B | 121212 | 768768768 | 323232 | enc-dec |
| fairseq-dense-125m | 125125125M | 125125125M | 121212 | 768768768 | 111 | dec |
| fairseq-moe-15b | 125125125M | 151515B | 121212 | 768768768 | 512512512 | dec |

Our experiments adopt the two open-source large language model families with their SMoE variants: (a𝑎a) the Switch Transformers (Fedus et al., [2022](#bib.bib18)) and (b𝑏b) Meta’s GPT-based SMoE models (Artetxe et al., [2022](#bib.bib2)). A summary of the specific model configurations is provided in Table [1](#S4.T1 "Table 1 ‣ Datasets and Network Backbones. ‣ 4.1 Implementation Details ‣ 4 Experiments ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy"). We use eight popular NLP tasks for supervised fine-tuning and evaluation: SST-2 (Socher et al., [2013](#bib.bib55)) for sentiment classification, MRPC (Dolan & Brockett, [2005](#bib.bib13)) for paraphrase identification, MultiRC (Khashabi et al., [2018](#bib.bib34)) for multiple-choice QA, COPA (Gordon et al., [2012](#bib.bib23)) for sentence completion, WinoGrande (Sakaguchi et al., [2019](#bib.bib53)) for conference resolution, SQuAD v1.1 (Rajpurkar et al., [2016](#bib.bib51)) for extractive QA, WikiQA (Yang et al., [2015](#bib.bib64)) and HotpotQA (Yang et al., [2018](#bib.bib65)) for closed-book QA. For zero-shot evaluation, we pick three representative benchmarks: MRPC in GLUE (Wang et al., [2019](#bib.bib59)), WinoGrande for reasoning, and OpenBookQA (Mihaylov et al., [2018](#bib.bib43)) for QA.

#### Comparison Baselines.

We compare our proposals to six baselines including two pruning and four merging methods. Firstly, we consider the “task-specific” expert pruning method from Chen et al. ([2022](#bib.bib6)), which gradually drops non-active experts during fine-tuning. Additionally, we evaluate the one-shot pruning of non-dominant experts as a sanity check. Secondly, given the absence of prior work on expert merging, we directly adapt Averaging (Choshen et al., [2022](#bib.bib9)), ZipIt (Stoica et al., [2023](#bib.bib56)), REPAIR (Jordan et al., [2022](#bib.bib32)) and Git Re-basin (Ainsworth et al., [2022](#bib.bib1)) merging methods to our SMoE scenarios as strong baselines for comparison.

#### Training and Evaluation Details.

For the encoder-decoder models, including the switch-base-32 SMoE model and the t5-base dense model, we report supervised fine-tuning results. For each task, we first undertake a comprehensive hyper-parameter search. This encompasses batch sizes from {888, 161616, 323232, 646464}, learning rates from {3×10−43superscript1043\times 10^{-4}, 1×10−41superscript1041\times 10^{-4}, 3×10−53superscript1053\times 10^{-5}, 1×10−51superscript1051\times 10^{-5}}, and epoch counts spanning {333, 555, 101010, 202020}, to pinpoint the optimal fine-tuned models. Further fine-tuning hyper-parameters are fixed, as shown in Appendix Table [A11](#A2.T11 "Table A11 ‣ Supervised Fine-Tuning Hyper-Parameters ‣ Appendix A2 More Technique Details ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy"). After merging and compression, we proceed to fine-tune the condensed model to restore its performance. Further, we apply knowledge distillation (KD) to compel the M-SMoE and MC-SMoE models to imitate the outputs generated by the full SMoE model on the training dataset. The hyper-parameters in the added KD loss are fixed for all tasks, please refer to Appendix [A2](#A2.SS0.SSS0.Px4 "Knowledge Distillation ‣ Appendix A2 More Technique Details ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy") for more details. As for the decoder-only models, including the fairseq-moe-15b SMoE model and the fairseq-dense-125m dense model, we report zero-shot results, i.e. without undergoing any further training. For the compression phase in MC-SMoE, we set the sparse ratio to 0.10.10.1 and the low-rank factor to 323232, following Li et al. ([2023](#bib.bib37)). The model size and the number of tera floating point operations (TFLOPs) are reported to measure the efficiency. The TFLOPs is evaluated by a batch of the first 646464 samples in the SQuAD dataset, with the input sequence length of 329329329 and the target sequence length of 131313. All experiments are conducted with PyTorch and DeepSpeed on NVIDIA A100100100 and A600060006000.

Table 2: Performance evaluations on the switch-base-32 model with 323232 experts in each SMoE layer, as well as its comparative dense model t5-base. We found the first SMoE layer has a profound impact on the model’s performance, and merging it results in more significant performance degradation compared to other layers. Thus for all merging/compression mechanisms, the first SMoE layer is skipped following Ma et al. ([2023](#bib.bib40)), and it maintains an average of 888 experts in other SMoE layers. We report exact-match/F1-score for SQuAD and HotpotQA, F1-score for MultiRC, and accuracy for other tasks. For each task, we highlight the best performance over all baselines in blue, and mark the performance no worse than full SMoE in bold.

| Methods | Model Size | TFLOPs | SST-2 | MRPC | MultiRC | COPA | WinoGrande | SQuAD | WikiQA | HotpotQA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 220220220M | 4.654.654.65 | 94.6194.6194.61 | 88.9788.9788.97 | 74.2574.2574.25 | 58.0058.0058.00 | 58.7258.7258.72 | 63.6563.6563.65/83.7683.7683.76 | 96.1296.1296.12 | 66.1366.1366.13/83.4583.4583.45 |
| Full SMoE | 2.02.02.0B | 4.654.654.65 | 95.7595.7595.75 | 90.2090.2090.20 | 76.1976.1976.19 | 68.0068.0068.00 | 61.8061.8061.80 | 65.3965.3965.39/85.885.885.81 | 96.4596.4596.45 | 67.5567.5567.55/84.6084.6084.60 |
| Pruning | 733733733M | 4.654.654.65 | 94.5094.5094.50 | 88.9788.9788.97 | 75.1375.1375.13 | 63.0063.0063.00 | 61.6461.6461.64 | 64.8064.8064.80/85.1385.1385.13 | 96.2796.2796.27 | 67.3967.3967.39/84.5684.5684.56 |
| Task-Specific | 733733733M | 4.654.654.65 | 91.2891.2891.28 | 82.0482.0482.04 | 53.6353.6353.63 | 52.0052.0052.00 | 58.5658.5658.56 | 54.4054.4054.40/78.0078.0078.00 | 95.2495.2495.24 | 64.7064.7064.70/82.7682.7682.76 |
| Averaging | 733733733M | 4.654.654.65 | 92.6692.6692.66 | 88.7388.7388.73 | 74.0474.0474.04 | 62.0062.0062.00 | 59.5959.5959.59 | 64.4964.4964.49/84.7584.7584.75 | 96.1996.1996.19 | 67.3667.3667.36/84.6184.6184.61 |
| ZipIt | 733733733M | 4.654.654.65 | 93.1293.1293.12 | 91.1891.18\mathbf{91.18} | 75.2675.2675.26 | 65.0065.0065.00 | 60.3860.3860.38 | 65.0165.0165.01/85.0685.0685.06 | 96.0596.0596.05 | 67.5967.59\mathbf{67.59}/84.7084.70\mathbf{84.70} |
| REPAIR | 733733733M | 4.654.654.65 | 92.8992.8992.89 | 90.4490.44\mathbf{90.44} | 74.4474.4474.44 | 65.0065.0065.00 | 61.4861.4861.48 | 64.6764.6764.67/84.8484.8484.84 | 96.2796.2796.27 | 67.6767.67\mathbf{67.67}/84.7784.77\mathbf{84.77} |
| Git Re-basin | 733733733M | 4.654.654.65 | 93.3593.3593.35 | 88.2488.2488.24 | 74.2574.2574.25 | 65.0065.0065.00 | 59.2559.2559.25 | 64.6164.6164.61/84.9284.9284.92 | 96.2396.2396.23 | 67.2967.2967.29/84.4684.4684.46 |
| M-SMoE | 733733733M | 4.654.654.65 | 94.5094.5094.50 | 90.6990.69\mathbf{90.69} | 75.5775.5775.57 | 68.0068.00\mathbf{68.00} | 61.8061.80\mathbf{61.80} | 65.6665.66\mathbf{65.66}/85.4985.4985.49 | 96.3496.3496.34 | 67.9167.91\mathbf{67.91}/84.8384.83\mathbf{84.83} |
| MC-SMoE | 381381381M | 3.833.833.83 | 93.3593.3593.35 | 89.2289.2289.22 | 73.9873.9873.98 | 67.0067.0067.00 | 59.5259.5259.52 | 65.4165.41\mathbf{65.41}/85.3085.3085.30 | 96.0896.0896.08 | 67.6467.64\mathbf{67.64}/84.7784.77\mathbf{84.77} |

### 4.2 Competitive Performance and Superior Efficiency of MC-SMoE

Table [2](#S4.T2 "Table 2 ‣ Training and Evaluation Details. ‣ 4.1 Implementation Details ‣ 4 Experiments ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy") presents the performance comparisons among M-SMoE, MC-SMoE, and eight baselines in a supervised fine-tuning manner on {SST2, MRPC, MultiRC, COPA, WinoGrande, SQuaD, WikiQA, HotpotQA} datasets. Note that all the compared methods activate the same number of parameters. From Table [2](#S4.T2 "Table 2 ‣ Training and Evaluation Details. ‣ 4.1 Implementation Details ‣ 4 Experiments ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy"), the following observations can be drawn: ❶ M-SMoE achieves 60%percent6060\% memory reduction while retaining performance on {MRPC, COPA, WinoGrande, SQuAD, HotpotQA}, and even obtains {0.490.490.49, 0.250.250.25, 0.410.410.41} (%percent\%) extra performance improvement on {MRPC, SQuAD, HotpotQA} over the full SMoE model, respectively. Although M-SMoE shows a marginal drop in performance for the memory efficiency on {SST2, MultiRC, WikiQA} benchmarks, however, it still outperforms all other pruning and merging baselines. These impressive results validate the superiority of our M-SMoE in consolidating the redundant experts. ❷ MC-SMoE is performed on top of the expert merging from M-SMoE. The resulting model achieves up to 80%percent8080\% in memory and 20%percent2020\% in FLOPs saving, while the performance degradation remains less than 1%percent11\% on {MRPC, COPA, SQuAD, WikiQA, HotpotQA}. ❸ In addition, the zero-shot learning comparisons between ours and baselines with the fairseq-moe-15b SMoE and fairseq-dense-125m dense models are included in Appendix [A1.1](#A1.SS1 "A1.1 Zero-Shot Evaluation Results ‣ Appendix A1 More Experimental Results ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy").

### 4.3 Ablation Study and Extra Investigation

Table 3: Comparison between Uniform and Adaptive (ours) merging ratio with the switch-base-32 model on four datasets.

| Merging Ratio | Uniform | Adaptive |
| --- | --- | --- |
| MultiRC | 74.4874.4874.48 | 75.5775.57\mathbf{75.57} |
| COPA | 63.0063.0063.00 | 68.0068.00\mathbf{68.00} |
| MRPC | 90.4490.4490.44 | 90.6990.69\mathbf{90.69} |
| SQuAD | 64.3664.3664.36/84.5684.5684.56 | 65.6665.66\mathbf{65.66}/85.4985.49\mathbf{85.49} |

#### Ablation on Different Merging Ratio Designs.

To testify whether our adaptive merging ratio is effective or not, we conduct an ablation study on different merging ratios, i.e., uniform (constant ratio per layer) v.s.formulae-sequence𝑣𝑠v.s. adaptive (ours). Experimental results are produced with the switch-base-32 backbone on four datasets, as shown in Table [3](#S4.T3 "Table 3 ‣ 4.3 Ablation Study and Extra Investigation ‣ 4 Experiments ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy"). Our adaptive ratio presents a consistent advantage in terms of merging performance, compared to the uniform ratio. It is within expectation since the pilot study in Figure [3](#S3.F3 "Figure 3 ‣ 3 Methodology ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy") reveals that the number of frequently utilized experts is different across different transformer blocks.

Table 4: Comparison between router-logits (ours) and seven other similarity functions for grouping experts.

| Representations | MultiRC | COPA | MRPC | SQuAD |
| --- | --- | --- | --- | --- |
| Random | 74.6974.6974.69 | 62.0062.0062.00 | 89.9589.9589.95 | 64.9764.9764.97/84.9684.9684.96 |
| Expert-weight | 75.2975.2975.29 | 63.063.063.00 | 89.4689.4689.46 | 64.9864.9864.98/85.1885.1885.18 |
| Expert-weight-feature | 74.9674.9674.96 | 62.0062.0062.00 | 89.9589.9589.95 | 64.9864.9864.98/85.1985.1985.19 |
| Expert-gradient | 75.5075.5075.50 | 59.0059.0059.00 | 89.2289.2289.22 | 64.9364.9364.93/85.0185.0185.01 |
| Expert-feature | 74.7474.7474.74 | 60.0060.0060.00 | 89.9589.9589.95 | 65.0365.0365.03/85.2185.2185.21 |
| Expert-feature.abs | 75.2075.2075.20 | 65.0065.0065.00 | 89.2289.2289.22 | 64.9064.9064.90/85.1585.1585.15 |
| Router-weight | 75.0175.0175.01 | 59.0059.0059.00 | 88.7388.7388.73 | 64.9964.9964.99/85.0285.0285.02 |
| Router-logits (Ours) | 75.5775.57\mathbf{75.57} | 68.0068.00\mathbf{68.00} | 90.6990.69\mathbf{90.69} | 65.6665.66\mathbf{65.66}/85.4985.49\mathbf{85.49} |

#### Ablation on Different Grouping Methods.

A pivotal component of our M-SMoE framework is to compute the similarity among experts by router output logits, i.e. router-logits, which directly determines their grouping statuses. Here, we carry out an ablation study for comparing our router-logits with seven other similarity functions: (i𝑖i) random, which generates a random vector for each expert; (i​i𝑖𝑖ii) expert-weight, using the flattened weight of each expert’s feed-forward network; (i​i​i𝑖𝑖𝑖iii) expert-weight-feature, leveraging the product of the expert’s weight and the L2 norm of its associated features; (i​v𝑖𝑣iv) expert-gradient, utilizing the flattened gradients of each expert’s feed-forward network; (v𝑣v) expert-feature, adopting the average input hidden states of each expert; (v​i𝑣𝑖vi) expert-feature.abs, using the average of absolute values of each expert’s input hidden states; (v​i​i𝑣𝑖𝑖vii) router-weight, adopting the corresponding row vector from the router weight matrix; and our (v​i​i​i𝑣𝑖𝑖𝑖viii) router-logits, which uses the router output logits vector corresponding to the expert after feeding a batch to the SMoE model. Experimental results with the switch-base-32 model across four datasets are presented in Table [4](#S4.T4 "Table 4 ‣ Ablation on Different Merging Ratio Designs. ‣ 4.3 Ablation Study and Extra Investigation ‣ 4 Experiments ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy"). We observe that our router-logits consistently outperforms all other similarity variants. The strength of router-logits lies in its ability to directly reflect the routing decision distribution of input samples. During the training, experts with a similar routing decision are optimized with a similar subset of data, leading to potential redundancy.

Table 5: Comparison between fine-tuning M-SMoE w.o. and w. (ours) KD with the switch-base-32 model.

| Methods | w.o. kD | w. kD |
| --- | --- | --- |
| MultiRC | 74.7774.7774.77 | 75.5775.57\mathbf{75.57} |
| COPA | 64.0064.0064.00 | 68.0068.00\mathbf{68.00} |
| MRPC | 89.2289.2289.22 | 90.6990.69\mathbf{90.69} |
| SQuAD | 63.2563.2563.25/84.0384.0384.03 | 65.6665.66\mathbf{65.66}/85.4985.49\mathbf{85.49} |

#### Contribution from Knowledge Distillation.

Knowledge distillation (KD) has been proven to be effective in inheriting information from large models. Therefore, we by default use KD for all merged and compressed SMoEs, including our M-SMoE, MC-SMoE, and all baselines. To show its contribution, we perform an ablation study comparing M-SMoE w. and w.o. the inclusion of KD loss during fine-tuning. Experimental results presented in Table [5](#S4.T5 "Table 5 ‣ Ablation on Different Grouping Methods. ‣ 4.3 Ablation Study and Extra Investigation ‣ 4 Experiments ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy"), with the switch-base-32 SMoE model across four datasets, underscore the advantages derived from the application of KD.

Table 6: Comparison between M-SMoE w.o. and w. permutation alignment (PA) with the switch-base-32 model.

| Methods | M-SMoE w.o. PA | M-SMoE w. PA |
| --- | --- | --- |
| MultiRC | 74.8474.8474.84 | 75.5775.57\mathbf{75.57} |
| COPA | 66.0066.0066.00 | 68.0068.00\mathbf{68.00} |
| MRPC | 89.9589.9589.95 | 90.6990.69\mathbf{90.69} |
| SQuAD | 64.7364.7364.73/84.7384.7384.73 | 65.6665.66\mathbf{65.66}/85.4985.49\mathbf{85.49} |

#### Contribution from Expert Permutation Alignment.

Consider an expert with two feed-forward layers with an intermediate dimension of d𝑑d, there are d!𝑑d! kinds of permutation possibilities to match and merge two experts. Next, we present an ablation study to compare M-SMoE w. and w.o. alignment to assess the effectiveness of expert permutation alignment. In Table [6](#S4.T6 "Table 6 ‣ Contribution from Knowledge Distillation. ‣ 4.3 Ablation Study and Extra Investigation ‣ 4 Experiments ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy"), we present results with the switch-base-32 SMoE model on four datasets. It demonstrates a clear performance improvement when applying the expert permutation alignment before merging. Therefore, without proper permutation alignment, expert merging could result in an inferior fusion of mismatched neurons.

Table 7: Comparison among M-SMoE that only merges, C-SMoE that only compresses, and MC-SMoE that merges and then compresses. Experiments are conducted with the switch-base-32 model. We highlight the better performance between C-SMoE and MC-SMoE in bold for each task.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Methods | SMoE | M-SMoE | C-SMoE | MC-SMoE |
| Model Size | 2.02.02.0B | 733733733M | 570570570M | 381381381M |
| TFLOPs | 4.654.654.65 | 4.654.654.65 | 3.833.833.83 | 3.833.833.83 |
| COPA | 68.0068.0068.00 | 68.0068.0068.00 | 64.0064.0064.00 | 67.0067.00\mathbf{67.00} |
| MRPC | 90.2090.2090.20 | 90.6990.6990.69 | 88.9788.9788.97 | 89.2289.22\mathbf{89.22} |
| SQuAD | 65.3965.3965.39/85.8185.8185.81 | 65.6665.6665.66/85.4985.4985.49 | 64.7864.7864.78/84.9384.9384.93 | 65.4165.41\mathbf{65.41}/85.3085.30\mathbf{85.30} |

#### Impact of Merging vs. Decomposition.

To quantify the extra benefit of the low dimensionality arising from M-SMoE, we look at the effects of merging experts and compressing SMoEs separately. We consider the evaluation of three tasks using the switch-base-32 SMoE model and compare M-SMoE that only merges experts, C-SMoE that only compresses, and with MC-SMoE that does both merging and compression. From Table [7](#S4.T7 "Table 7 ‣ Contribution from Expert Permutation Alignment. ‣ 4.3 Ablation Study and Extra Investigation ‣ 4 Experiments ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy"), we observe: ❶ M-SMoE reduces the model size while maintaining or boosting performance. In contrast, C-SMoE (i.e., compression only) leads to a significant performance drop. It suggests that merging is a superior option to pursue memory efficiency and maintain model quality. ❷ The success of M-SMoE paves the way for further compression. This is supported by MC-SMoE outperforming C-SMoE with even fewer parameter counts.

Table 8: Comparison among different averaging strategies of Uniform, Fisher-weighted and Frequency-weighted (ours), evaluated with the switch-base-32 SMoE models.

| Methods | Uniform | Fisher-weighted | Frequency-weighted |
| --- | --- | --- | --- |
| MultiRC | 75.1175.1175.11 | 73.7773.7773.77 | 75.5775.57\mathbf{75.57} |
| COPA | 64.0064.0064.00 | 65.0065.0065.00 | 68.0068.00\mathbf{68.00} |
| MRPC | 89.9589.9589.95 | 89.4689.4689.46 | 90.6990.69\mathbf{90.69} |
| SQuAD | 64.5564.5564.55/84.8584.8584.85 | 63.9963.9963.99/84.4484.4484.44 | 65.6665.66\mathbf{65.66}/85.4985.49\mathbf{85.49} |

#### Ablation on Different Merging Strategies.

To examine the effectiveness of our proposed frequency-aware expert merging, an ablation study on different merging strategies is needed. Specifically, we investigate uniform (Wortsman et al., [2022](#bib.bib62)), fisher-weighted (Matena & Raffel, [2022](#bib.bib42)), and frequency-weighted (ours) merging methods with the switch-base-32 model across four datasets. As detailed in Table [8](#S4.T8 "Table 8 ‣ Impact of Merging vs. Decomposition. ‣ 4.3 Ablation Study and Extra Investigation ‣ 4 Experiments ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy"), we see that our frequency-weighted merging consistently reaches the best performance. A possible reason is that merging based on activation frequencies suppresses the impact of less significant experts. In contrast, the uniform approach tends to give inappropriate prominence to redundant information, overshadowing critical experts during the merging process. As for the fisher-weighted merging strategy, which relies on gradient magnitude for expert re-weighting, does not quite hit the mark, since in our case, the experts have already been well pre-trained before merging.

Figure 5: Ratio of remaining parameters after further compressing the dominant experts from MC-SMoE.

#### Visualization of Compact SMoEs from MC-SMoE.

We visualize the distribution of dominant experts in the switch-base-32 SMoE model produced by M-SMoE, and their compressed versions from MC-SMoE in Figure [5](#S4.F5 "Figure 5 ‣ Ablation on Different Merging Strategies. ‣ 4.3 Ablation Study and Extra Investigation ‣ 4 Experiments ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy"). Each grid box denotes a dominant expert, and the darker color indicates more remaining parameters in that expert. Later SMoE layers, at the bottom of the heatmap, seem to be more mergeable and compressible.

## 5 Conclusions

Sparse Mixture-of-Experts (SMoE) is a promising framework to scale up the model capacity, which enjoys roughly unchanged training and inference FLOPs at the cost of significantly increased memory overheads. The memory requirements and expert redundancy highly limit its practical usage. In this work, we propose an innovative SMoE merging approach, i.e., M-SMoE, based on the hints from routing policies, to consolidate expert information into fewer but more knowledgeable ones. Moreover, such merged experts are demonstrated to be more compressible. our proposed, MC-SMoE methods pursue superior memory and parameter efficiency with competitive performance. We conduct comprehensive experiments to support the effectiveness of our proposals. Future works mainly lie in the extension of multi-modality scenarios and co-designs with hardware platforms.

## 6 Reproducibility Statement

To encourage reproducibility, we have made our source code available333Our code is provided at <https://github.com/UNITES-Lab/MC-SMoE>., including the data pre-processing, SMoE merging/compression/pruning, and evaluation scripts. The hyperparameter details are provided in Appendix [A2](#A2 "Appendix A2 More Technique Details ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy") and the detailed pseudo-code about SMoE expert merging is provided in Appendix [A3](#A3 "Appendix A3 More Implementation Details ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy"). We also provide clear and concise Algorithm [1](#alg1 "Algorithm 1 ‣ Post-Merging Compression of MC-SMoE. ‣ 3.2 Merging Encourages Expert Decomposition ‣ 3 Methodology ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy") for our MC-SMoE pipeline.

## References

* Ainsworth et al. (2022)

  Samuel K Ainsworth, Jonathan Hayase, and Siddhartha Srinivasa.
  Git re-basin: Merging models modulo permutation symmetries.
  *arXiv preprint arXiv:2209.04836*, 2022.
* Artetxe et al. (2022)

  Mikel Artetxe, Shruti Bhosale, Naman Goyal, Todor Mihaylov, Myle Ott, Sam Shleifer, Xi Victoria Lin, Jingfei Du, Srinivasan Iyer, Ramakanth Pasunuru, Giri Anantharaman, Xian Li, Shuohui Chen, Halil Akin, Mandeep Baines, Louis Martin, Xing Zhou, Punit Singh Koura, Brian O’Horo, Jeff Wang, Luke Zettlemoyer, Mona Diab, Zornitsa Kozareva, and Ves Stoyanov.
  Efficient large scale language modeling with mixtures of experts.
  *arXiv preprint arXiv:2112.10684*, 2022.
* Brown et al. (2020)

  Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei.
  Language models are few-shot learners.
  *arXiv preprint arXiv:2005.14165*, 2020.
* Cai et al. (2023)

  Ruisi Cai, Zhenyu Zhang, and Zhangyang Wang.
  Robust weight signatures: Gaining robustness as easy as patching weights?
  *arXiv preprint arXiv:2302.12480*, 2023.
* Chen et al. (2023)

  Tianlong Chen, Zhenyu Zhang, Ajay Jaiswal, Shiwei Liu, and Zhangyang Wang.
  Sparse moe as the new dropout: Scaling dense and self-slimmable transformers.
  *arXiv preprint arXiv:2303.01610*, 2023.
* Chen et al. (2022)

  Tianyu Chen, Shaohan Huang, Yuan Xie, Binxing Jiao, Daxin Jiang, Haoyi Zhou, Jianxin Li, and Furu Wei.
  Task-specific expert pruning for sparse mixture-of-experts.
  *arXiv preprint arXiv:2206.00277*, 2022.
* Chen et al. (2021)

  Xuxi Chen, Tianlong Chen, Yu Cheng, Weizhu Chen, Zhangyang Wang, and Ahmed Hassan Awadallah.
  Dsee: Dually sparsity-embedded efficient tuning of pre-trained language models.
  *arXiv preprint arXiv:2111.00160*, 2021.
* Chi et al. (2022)

  Zewen Chi, Li Dong, Shaohan Huang, Damai Dai, Shuming Ma, Barun Patra, Saksham Singhal, Payal Bajaj, Xia Song, and Furu Wei.
  On the representation collapse of sparse mixture of experts.
  *arXiv preprint arXiv:2204.09179*, 2022.
* Choshen et al. (2022)

  Leshem Choshen, Elad Venezian, Noam Slonim, and Yoav Katz.
  Fusing finetuned models for better pretraining.
  *arXiv preprint arXiv:2204.03044*, 2022.
* Chowdhery et al. (2022)

  Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel.
  Palm: Scaling language modeling with pathways.
  *arXiv preprint arXiv:2204.02311*, 2022.
* Chung et al. (2022)

  Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei.
  Scaling instruction-finetuned language models.
  *arXiv preprint arXiv:2210.11416*, 2022.
* Devlin et al. (2019)

  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.
  Bert: Pre-training of deep bidirectional transformers for language understanding.
  *ArXiv*, abs/1810.04805, 2019.
* Dolan & Brockett (2005)

  William B. Dolan and Chris Brockett.
  Automatically constructing a corpus of sentential paraphrases.
  In *Proceedings of the Third International Workshop on Paraphrasing (IWP2005)*, 2005.
  URL <https://aclanthology.org/I05-5002>.
* Don-Yehiya et al. (2022)

  Shachar Don-Yehiya, Elad Venezian, Colin Raffel, Noam Slonim, Yoav Katz, and Leshem Choshen.
  Cold fusion: Collaborative descent for distributed multitask finetuning.
  *arXiv preprint arXiv:2212.01378*, 2022.
* Dosovitskiy et al. (2021)

  Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby.
  An image is worth 16x16 words: Transformers for image recognition at scale.
  *ArXiv*, abs/2010.11929, 2021.
* Draxler et al. (2018)

  Felix Draxler, Kambis Veschgini, Manfred Salmhofer, and Fred Hamprecht.
  Essentially no barriers in neural network energy landscape.
  In *International conference on machine learning*, pp.  1309–1318. PMLR, 2018.
* Fan et al. (2022)

  Zhiwen Fan, Rishov Sarkar, Ziyu Jiang, Tianlong Chen, Kai Zou, Yu Cheng, Cong Hao, Zhangyang Wang, et al.
  M3vit: Mixture-of-experts vision transformer for efficient multi-task learning with model-accelerator co-design.
  *Advances in Neural Information Processing Systems*, 35:28441–28457, 2022.
* Fedus et al. (2022)

  William Fedus, Barret Zoph, and Noam Shazeer.
  Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity.
  *Journal of Machine Learning Research*, 23(120):1–39, 2022.
  URL <http://jmlr.org/papers/v23/21-0998.html>.
* Frankle et al. (2020)

  Jonathan Frankle, Gintare Karolina Dziugaite, Daniel Roy, and Michael Carbin.
  Linear mode connectivity and the lottery ticket hypothesis.
  In *International Conference on Machine Learning*, pp.  3259–3269. PMLR, 2020.
* Freeman & Bruna (2016)

  C Daniel Freeman and Joan Bruna.
  Topology and geometry of half-rectified network optimization.
  *arXiv preprint arXiv:1611.01540*, 2016.
* Gao et al. (2022)

  Ze-Feng Gao, Peiyu Liu, Wayne Xin Zhao, Zhong-Yi Lu, and Ji-Rong Wen.
  Parameter-efficient mixture-of-experts architecture for pre-trained language models.
  *arXiv preprint arXiv:2203.01104*, 2022.
* Garipov et al. (2018)

  Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry P Vetrov, and Andrew G Wilson.
  Loss surfaces, mode connectivity, and fast ensembling of dnns.
  *Advances in neural information processing systems*, 31, 2018.
* Gordon et al. (2012)

  Andrew Gordon, Zornitsa Kozareva, and Melissa Roemmele.
  SemEval-2012 task 7: Choice of plausible alternatives: An evaluation of commonsense causal reasoning.
  In *\*SEM 2012: The First Joint Conference on Lexical and Computational Semantics – Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation (SemEval 2012)*, pp.  394–398, Montréal, Canada, 2012. Association for Computational Linguistics.
  URL <https://aclanthology.org/S12-1052>.
* Gueta et al. (2023)

  Almog Gueta, Elad Venezian, Colin Raffel, Noam Slonim, Yoav Katz, and Leshem Choshen.
  Knowledge is a region in weight space for fine-tuned language models.
  *arXiv preprint arXiv:2302.04863*, 2023.
* Han et al. (2016)

  Song Han, Huizi Mao, and William J. Dally.
  Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding.
  *arXiv preprint arXiv:1510.00149*, 2016.
* He et al. (2021)

  Jiaao He, Jiezhong Qiu, Aohan Zeng, Zhilin Yang, Jidong Zhai, and Jie Tang.
  Fastmoe: A fast mixture-of-expert training system.
  *arXiv preprint arXiv:2103.13262*, 2021.
* He et al. (2022)

  Jiaao He, Jidong Zhai, Tiago Antunes, Haojie Wang, Fuwen Luo, Shangfeng Shi, and Qin Li.
  Fastermoe: Modeling and optimizing training of large-scale dynamic pre-trained models.
  In *Proceedings of the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming*, PPoPP ’22, pp.  120–134, New York, NY, USA, 2022. Association for Computing Machinery.
  ISBN 9781450392044.
  doi: 10.1145/3503221.3508418.
  URL <https://doi.org/10.1145/3503221.3508418>.
* Ilharco et al. (2022a)

  Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi.
  Editing models with task arithmetic.
  *arXiv preprint arXiv:2212.04089*, 2022a.
* Ilharco et al. (2022b)

  Gabriel Ilharco, Mitchell Wortsman, Samir Yitzhak Gadre, Shuran Song, Hannaneh Hajishirzi, Simon Kornblith, Ali Farhadi, and Ludwig Schmidt.
  Patching open-vocabulary models by interpolating weights.
  *Advances in Neural Information Processing Systems*, 35:29262–29277, 2022b.
* Jin et al. (2022)

  Xisen Jin, Xiang Ren, Daniel Preotiuc-Pietro, and Pengxiang Cheng.
  Dataless knowledge fusion by merging weights of language models.
  *arXiv preprint arXiv:2212.09849*, 2022.
* Jolicoeur-Martineau et al. (2023)

  Alexia Jolicoeur-Martineau, Emy Gervais, Kilian Fatras, Yan Zhang, and Simon Lacoste-Julien.
  Population parameter averaging (papa).
  *arXiv preprint arXiv:2304.03094*, 2023.
* Jordan et al. (2022)

  Keller Jordan, Hanie Sedghi, Olga Saukh, Rahim Entezari, and Behnam Neyshabur.
  Repair: Renormalizing permuted activations for interpolation repair.
  *arXiv preprint arXiv:2211.08403*, 2022.
* Kaplan et al. (2020)

  Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei.
  Scaling laws for neural language models.
  *arXiv preprint arXiv:2001.08361*, 2020.
* Khashabi et al. (2018)

  Daniel Khashabi, Snigdha Chaturvedi, Michael Roth, Shyam Upadhyay, and Dan Roth.
  Looking beyond the surface: A challenge set for reading comprehension over multiple sentences.
  In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)*, pp.  252–262, New Orleans, Louisiana, 2018. Association for Computational Linguistics.
  doi: 10.18653/v1/N18-1023.
  URL <https://aclanthology.org/N18-1023>.
* Kim et al. (2021)

  Young Jin Kim, Ammar Ahmad Awan, Alexandre Muzio, Andres Felipe Cruz Salinas, Liyang Lu, Amr Hendy, Samyam Rajbhandari, Yuxiong He, and Hany Hassan Awadalla.
  Scalable and efficient moe training for multitask multilingual models.
  *arXiv preprint arXiv:2109.10465*, 2021.
* Lepikhin et al. (2021)

  Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen.
  {GS}hard: Scaling giant models with conditional computation and automatic sharding.
  In *International Conference on Learning Representations*, 2021.
  URL <https://openreview.net/forum?id=qrwe7XHTmYb>.
* Li et al. (2023)

  Yixiao Li, Yifan Yu, Qingru Zhang, Chen Liang, Pengcheng He, Weizhu Chen, and Tuo Zhao.
  Losparse: Structured compression of large language models based on low-rank and sparse approximation.
  *arXiv preprint arXiv:2306.11222*, 2023.
* Liu et al. (2019)

  Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov.
  Roberta: A robustly optimized bert pretraining approach.
  *arXiv preprint arXiv:1907.11692*, 2019.
* Liu et al. (2021)

  Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo.
  Swin transformer: Hierarchical vision transformer using shifted windows.
  In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp.  10012–10022, October 2021.
* Ma et al. (2023)

  Xinyin Ma, Gongfan Fang, and Xinchao Wang.
  Llm-pruner: On the structural pruning of large language models.
  *arXiv preprint arXiv:2305.11627*, 2023.
* Mao et al. (2022)

  Zhiyuan Mao, Ajay Jaiswal, Zhangyang Wang, and Stanley H. Chan.
  Single frame atmospheric turbulence mitigation: A benchmark study and a new physics-inspired transformer model.
  *ArXiv*, abs/2207.10040, 2022.
* Matena & Raffel (2022)

  Michael S Matena and Colin A Raffel.
  Merging models with fisher-weighted averaging.
  *Advances in Neural Information Processing Systems*, 35:17703–17716, 2022.
* Mihaylov et al. (2018)

  Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal.
  Can a suit of armor conduct electricity? a new dataset for open book question answering.
  In *EMNLP*, 2018.
* Mittal et al. (2022)

  Sarthak Mittal, Yoshua Bengio, and Guillaume Lajoie.
  Is a modular architecture enough?
  *Advances in Neural Information Processing Systems*, 35:28747–28760, 2022.
* Muqeeth et al. (2023)

  Mohammed Muqeeth, Haokun Liu, and Colin Raffel.
  Soft merging of experts with adaptive routing.
  *arXiv preprint arXiv:2306.03745*, 2023.
* Nie et al. (2022)

  Xiaonan Nie, Pinxue Zhao, Xupeng Miao, Tong Zhao, and Bin Cui.
  Hetumoe: An efficient trillion-scale mixture-of-expert distributed training system.
  *arXiv preprint arXiv:2203.14685*, 2022.
* Ortega et al. (2022)

  Luis A Ortega, Rafael Cabañas, and Andres Masegosa.
  Diversity and generalization in neural network ensembles.
  In *International Conference on Artificial Intelligence and Statistics*, pp.  11720–11743. PMLR, 2022.
* Peña et al. (2023)

  Fidel A Guerrero Peña, Heitor Rapela Medeiros, Thomas Dubail, Masih Aminbeidokhti, Eric Granger, and Marco Pedersoli.
  Re-basin via implicit sinkhorn differentiation.
  In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp.  20237–20246, 2023.
* Raffel et al. (2020)

  Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu.
  Exploring the limits of transfer learning with a unified text-to-text transformer.
  *arXiv preprint arXiv:1910.10683*, 2020.
* Rajbhandari et al. (2022)

  Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, and Yuxiong He.
  Deepspeed-moe: Advancing mixture-of-experts inference and training to power next-generation ai scale.
  *arXiv preprint arXiv:2201.05596*, 2022.
* Rajpurkar et al. (2016)

  Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang.
  SQuAD: 100,000+ questions for machine comprehension of text.
  In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*, pp.  2383–2392, Austin, Texas, 2016. Association for Computational Linguistics.
  doi: 10.18653/v1/D16-1264.
  URL <https://aclanthology.org/D16-1264>.
* Rame et al. (2023)

  Alexandre Rame, Kartik Ahuja, Jianyu Zhang, Matthieu Cord, Léon Bottou, and David Lopez-Paz.
  Model ratatouille: Recycling diverse models for out-of-distribution generalization.
  *arXiv preprint arXiv:2212.10445*, 2023.
* Sakaguchi et al. (2019)

  Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi.
  Winogrande: An adversarial winograd schema challenge at scale.
  *arXiv preprint arXiv:1907.10641*, 2019.
* Shazeer et al. (2017)

  Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean.
  Outrageously large neural networks: The sparsely-gated mixture-of-experts layer.
  *arXiv preprint arXiv:1701.06538*, 2017.
* Socher et al. (2013)

  Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts.
  Recursive deep models for semantic compositionality over a sentiment treebank.
  In *Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing*, pp.  1631–1642, Seattle, Washington, USA, 2013. Association for Computational Linguistics.
  URL <https://aclanthology.org/D13-1170>.
* Stoica et al. (2023)

  George Stoica, Daniel Bolya, Jakob Bjorner, Taylor Hearn, and Judy Hoffman.
  Zipit! merging models from different tasks without training.
  *arXiv preprint arXiv:2305.03053*, 2023.
* Touvron et al. (2021)

  Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herv’e J’egou.
  Training data-efficient image transformers & distillation through attention.
  In *ICML*, 2021.
* Vaswani et al. (2023)

  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.
  Attention is all you need.
  *arXiv preprint arXiv:1706.03762*, 2023.
* Wang et al. (2019)

  Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman.
  Glue: A multi-task benchmark and analysis platform for natural language understanding.
  *arXiv preprint arXiv:1804.07461*, 2019.
* Wang et al. (2023)

  Hongyi Wang, Saurabh Agarwal, Pongsakorn U-chupala, Yoshiki Tanaka, Eric P. Xing, and Dimitris Papailiopoulos.
  Cuttlefish: Low-rank model training without all the tuning.
  *arXiv preprint arXiv:2305.02538*, 2023.
* Wei et al. (2022)

  Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou.
  Chain of thought prompting elicits reasoning in large language models.
  *arXiv preprint arXiv:2201.11903*, 2022.
* Wortsman et al. (2022)

  Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, et al.
  Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.
  In *International Conference on Machine Learning*, pp.  23965–23998. PMLR, 2022.
* Yadav et al. (2023)

  Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal.
  Resolving interference when merging models.
  In *NeurIPS*, New Orleans, USA, 2023. Proceedings of Machine Learning Research.
* Yang et al. (2015)

  Yi Yang, Wen-tau Yih, and Christopher Meek.
  WikiQA: A challenge dataset for open-domain question answering.
  In *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing*, pp.  2013–2018, Lisbon, Portugal, 2015. Association for Computational Linguistics.
  doi: 10.18653/v1/D15-1237.
  URL <https://aclanthology.org/D15-1237>.
* Yang et al. (2018)

  Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning.
  HotpotQA: A dataset for diverse, explainable multi-hop question answering.
  In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pp.  2369–2380, Brussels, Belgium, October-November 2018. Association for Computational Linguistics.
  doi: 10.18653/v1/D18-1259.
  URL <https://aclanthology.org/D18-1259>.
* Yang et al. (2019)

  Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le.
  Xlnet: Generalized autoregressive pretraining for language understanding.
  *Advances in neural information processing systems*, 32, 2019.
* Zheng et al. (2021)

  Minghang Zheng, Peng Gao, Renrui Zhang, Xiaogang Wang, Hongsheng Li, and Hao Dong.
  End-to-end object detection with adaptive clustering transformer.
  *ArXiv*, abs/2011.09315, 2021.
* Zhu et al. (2019)

  Shilin Zhu, Xin Dong, and Hao Su.
  Binary ensemble neural network: More bits per network or more networks per bit?
  In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, June 2019.

## Appendix

## Appendix A1 More Experimental Results

### A1.1 Zero-Shot Evaluation Results

We compare our proposed M-SMoE, MC-SMoE with one-shot pruning of non-dominant experts and the “task-specific” expert pruning method, in a zero-shot learning manner. Our M-SMoE consistently outperforms the baseline methods, as shown in Table [A9](#A1.T9 "Table A9 ‣ A1.1 Zero-Shot Evaluation Results ‣ Appendix A1 More Experimental Results ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy"). The performance might be further improved if only we can fine-tune the routers, given that our M-SMoE highly leverages routing information during the merging phase.

Table A9: Performance evaluation on the fairseq-moe-15b model with 512512512 experts in each SMoE layer, as well as its comparative dense model fairseq-dense-125m. Different from the fine-tuned switch-base-32 model, we apply pruning/merging methods on every SMoE layer here and maintain an average of 161616 experts. We highlight the best performance over all baselines in bold.

| Methods | Model Size | TFLOPs | MRPC | OpenBookQA | WinoGrande |
| --- | --- | --- | --- | --- | --- |
| Dense | 125125125M | 5.085.085.08 | 37.7537.7537.75 | 34.0034.0034.00 | 49.4949.4949.49 |
| Full SMoE | 151515B | 5.085.085.08 | 60.5460.5460.54 | 36.8036.8036.80 | 51.7851.7851.78 |
| Pruning | 552552552M | 5.085.085.08 | 52.2052.2052.20 | 30.6030.6030.60 | 48.4648.4648.46 |
| Task-Specific | 552552552M | 5.085.085.08 | 40.1940.1940.19 | 23.6023.6023.60 | 48.3848.3848.38 |
| M-SMoE | 552552552M | 5.085.085.08 | 52.6952.69\mathbf{52.69} | 34.4034.4034.40 | 50.4350.43\mathbf{50.43} |
| MC-SMoE | 166166166M | 4.454.454.45 | 47.5547.5547.55 | 34.6034.60\mathbf{34.60} | 49.0949.0949.09 |

### A1.2 Efficiency Discussions and Limitations

#### Latency Limitations

Despite the {{\{dense, SMoE, M-SMoE, MC-SMoE}}\} models sharing the same theoretical TFLOPs, they do not necessarily produce the same latency. This is because the vanilla design of SMoE in the real world suffers from significant extra latency costs introduced by routing (Nie et al., [2022](#bib.bib46)). Our proposed M-SMoE and MC-SMoE achieve impressive memory and TFLOPs efficiency for SMoE. However, they do not improve latency. Ideally, the merging process is supposed to reduce the number of classes managed by the router classifier due to the reduction in the number of experts in each layer. However, in practical implementation, we face a challenge: explicitly creating a new router for the merged experts is non-trivial. To address this issue, we adopt the following strategy as shown in Appendix [A3](#A3 "Appendix A3 More Implementation Details ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy"): within each group, we retain a representative expert and let other routers point towards this representative. Yet, all such routing decisions into this group will now be directed towards a single new merged expert. This implies that, although the count of experts reduces, the number of classes managed by the router remains constant, i.e. the routing latency costs remain constant. Thus, if we manage to prune the router output channels without affecting its functionality, we can realize a notable improvement in latency efficiency.

To examine the potential efficiency from router pruning upon M-SMoE, we conduct experiments with the switch-base-32 backbone on batch size {{\{32, 256, 512}}\} and compare inference latency of these four models: ① dense, ② SMoE, ③ M-SMoE, ④ M-SMoE w. pruning router. Notably, results in Table [A10](#A1.T10 "Table A10 ‣ Latency Limitations ‣ A1.2 Efficiency Discussions and Limitations ‣ Appendix A1 More Experimental Results ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy") across three batch size settings demonstrate a latency ordering of ②≈\approx③>>④>>①. This indicates the latency limitation and encourages future work for router pruning.

Table A10: Latency analysis of the switch-base-32 model on SQuAD task inference with BF16.

| Models | bsz=323232 | | bsz=256256256 | | bsz=512512512 | |
| --- | --- | --- | --- | --- | --- | --- |
| TFLOPs | Latency (s) | TFLOPs | Latency (s) | TFLOPs | Latency (s) |
| Dense | 2.332.332.33 | 0.080.080.08 | 25.5125.5125.51 | 0.850.850.85 | 59.3259.3259.32 | 2.332.332.33 |
| Full SMoE-32 | 2.332.332.33 | 0.180.180.18 | 25.5125.5125.51 | 1.021.021.02 | 59.3259.3259.32 | 2.502.502.50 |
| M-SMoE-8 | 2.332.332.33 | 0.17¯¯0.17\underline{0.17} | 25.5125.5125.51 | 0.99¯¯0.99\underline{0.99} | 59.3259.3259.32 | 2.48¯¯2.48\underline{2.48} |
| M-SMoE-8 w. pruning router | 2.332.332.33 | 0.13¯¯0.13\underline{0.13} | 25.5125.5125.51 | 0.93¯¯0.93\underline{0.93} | 59.3259.3259.32 | 2.38¯¯2.38\underline{2.38} |

## Appendix A2 More Technique Details

#### Supervised Fine-Tuning Hyper-Parameters

Besides {{\{batch size, learning rate, epoch counts}}\} which vary for each task, we keep other hyper-parameters of supervised fine-tuning fixed for all tasks. These are shown in Table [A11](#A2.T11 "Table A11 ‣ Supervised Fine-Tuning Hyper-Parameters ‣ Appendix A2 More Technique Details ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy").

Table A11: Fine-tuning hyper-parameters of the switch-base-32 model.

|  |  |
| --- | --- |
| Hyper-Parameters | Values |
| Optimizer | AdamW |
| Adam ϵitalic-ϵ\epsilon | 1​e−61𝑒61e\mathrm{-}6 |
| Adam β𝛽\beta | (0.90.90.9, 0.980.980.98) |
| Warm-up steps | 161616 |
| Weight decay | 0.010.010.01 |
| LR scheduler | Linear decay |
| KD α𝛼\alpha | 0.20.20.2 |
| KD T𝑇T | 2.02.02.0 |

#### Details in Zero-Shot Learning

We evaluate our approaches and baselines with the fairseq-moe-15b model in the zero-shot learning setting. Specifically, We use the language model to separately
score each label choice, and pick the one with the highest score as the prediction. Although we utilize the training sets, they are only incorporated when essential in merging/compression, such as when calculating the expert usage frequency. In short, no optimization occurs at any stage of the process, i.e. no fine-tuning at all.

#### Compression Hyper-Parameters

For M-SMoE, we randomly pick 256256256 samples from training data to calculate both expert usage frequency and router-logits similarity for all tasks. For the compression phase in MC-SMoE, following Li et al. ([2023](#bib.bib37)), we adopt the cubic pruning ratio scheduler to control the 𝚂𝚂\mathtt{S} pruning process:

|  |  |  |
| --- | --- | --- |
|  | 𝒫t={10≤t<𝒯i,𝒫𝒯+(1−𝒫𝒯)​(1−t−𝒯i−𝒯f𝒯−𝒯i−𝒯f)3𝒯i≤t<𝒯−𝒯f,𝒫𝒯o.w.,subscript𝒫𝑡cases10𝑡subscript𝒯𝑖subscript𝒫𝒯1subscript𝒫𝒯superscript1𝑡subscript𝒯𝑖subscript𝒯𝑓𝒯subscript𝒯𝑖subscript𝒯𝑓3subscript𝒯𝑖𝑡𝒯subscript𝒯𝑓subscript𝒫𝒯o.w.\displaystyle\mathcal{P}\_{t}=\begin{cases}1&0\leq t<\mathcal{T}\_{i},\\ \mathcal{P}\_{\mathcal{T}}+\left(1-\mathcal{P}\_{\mathcal{T}}\right)\left(1-\frac{t-\mathcal{T}\_{i}-\mathcal{T}\_{f}}{\mathcal{T}-\mathcal{T}\_{i}-\mathcal{T}\_{f}}\right)^{3}&\mathcal{T}\_{i}\leq t<\mathcal{T}-\mathcal{T}\_{f},\\ \mathcal{P}\_{\mathcal{T}}&\text{o.w.}\end{cases}, |  |

where 𝒯𝒯\mathcal{T} is the total steps. 𝒯isubscript𝒯𝑖\mathcal{T}\_{i} is the number of initial warm-up steps. 𝒯fsubscript𝒯𝑓\mathcal{T}\_{f} is the number of final cold-down steps. We set 𝒯𝒯\mathcal{T} to 100001000010000, 𝒯isubscript𝒯𝑖\mathcal{T}\_{i} to 400400400 and 𝒯jsubscript𝒯𝑗\mathcal{T}\_{j} to 160016001600 for all tasks.

#### Knowledge Distillation

In this paragraph we illustrate the detail of knowledge distillation (KD) applied in the supervised fine-tuning setting on all merged and compressed SMoE models for performance recovery, including our M-SMoE, MC-SMoE and all baselines. The goal is to force them, i.e. the students, to imitate the outputs from the full SMoE model, i.e. the teacher. Specifically, the training objective can be formulated as:

|  |  |  |
| --- | --- | --- |
|  | minΘ⁡𝔼(𝒙,𝒚)∼𝒟​[ℒ​(𝒙;Θ)+α​ℒKD​(𝒙;Θ)],subscriptΘsubscript𝔼similar-to𝒙𝒚𝒟delimited-[]ℒ  𝒙Θ𝛼subscriptℒKD  𝒙Θ\displaystyle\min\_{\Theta}\mathbb{E}\_{(\bm{x},\bm{y})\sim\mathcal{D}}\left[\mathcal{L}(\bm{x};\Theta)+\alpha\mathcal{L}\_{\text{KD}}(\bm{x};\Theta)\right], |  |

where the value of α𝛼\alpha is fixed at 0.20.20.2 for all tasks. ℒℒ\mathcal{L} is the cross-entropy loss between predictions and the given hard labels, ℒKDsubscriptℒKD\mathcal{L}\_{\text{KD}} is the KL divergence loss between the predictions and the full SMoE model’s soft labels:

|  |  |  |
| --- | --- | --- |
|  | ℒKD=KL[𝒫(𝒚|𝒙;Θ(f​u​l​l))∥𝒫(𝒚|𝒙;Θ)].\displaystyle\mathcal{L}\_{\text{KD}}=\texttt{KL}\left[\mathcal{P}\left(\bm{y}\,|\,\bm{x}\,;\,\Theta^{(full)}\right)\,\,\|\,\,\mathcal{P}\left(\bm{y}\,|\,\bm{x}\,;\,\Theta\right)\right]. |  |

Moreover, we employ a temperature T𝑇T in the KL divergence to control the smoothness of the output distribution for both student and teacher models, defined as:

|  |  |  |
| --- | --- | --- |
|  | pi=exp⁡(zi/T),subscript𝑝𝑖subscript𝑧𝑖𝑇\displaystyle p\_{i}=\exp(z\_{i}/T), |  |

where zisubscript𝑧𝑖z\_{i} is the logit score for class j𝑗j, and the T𝑇T is fixed at 222 for all tasks.

#### The Router-Weight Similarity Function

We provide a detailed description of the router-weight similarity function in this paragraph, which is inferior to our adopted router-logits in Section [3.1](#S3.SS1 "3.1 Routing Policy Guides Experts Merging ‣ 3 Methodology ‣ Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy"). Specifically, the similarity Sim​(⋅,⋅)Sim⋅⋅\texttt{Sim}(\cdot,\cdot) between experts 𝙴isubscript𝙴𝑖\mathtt{E}\_{i} and 𝙴jsubscript𝙴𝑗\mathtt{E}\_{j} in an SMoE layer is computed by:

|  |  |  |
| --- | --- | --- |
|  | Sim​(𝙴i,𝙴j)=cosine​(𝚆ri,∗,𝚆rj,∗),Simsubscript𝙴𝑖subscript𝙴𝑗cosinesuperscriptsubscript𝚆𝑟  𝑖superscriptsubscript𝚆𝑟  𝑗\displaystyle\texttt{Sim}(\mathtt{E}\_{i},\mathtt{E}\_{j})=\texttt{cosine}(\mathtt{W}\_{r}^{i,\*},\mathtt{W}\_{r}^{j,\*}), |  |

where 𝚆rsubscript𝚆𝑟\mathtt{W}\_{r} is the router weight, and 𝚆ri,∗superscriptsubscript𝚆𝑟

𝑖\mathtt{W}\_{r}^{i,\*} and 𝚆rj,∗superscriptsubscript𝚆𝑟

𝑗\mathtt{W}\_{r}^{j,\*} are row vectors in it.

## Appendix A3 More Implementation Details

We show some pseudocode to demonstrate the implementation of our proposed M-SMoE in a PyTorch-like style.

#### Details of Merging Experts in an SMoE Feed-Forward Layer

In our experiments, the final step of merging involves replacing one expert in a group with the derived weight. Instead of pruning the other experts, we redirect the remaining ones in that group to the newly substituted expert. This implementation ensures that the routing functionality remains consistent. Below is the PyTorch-style pseudo code:

[⬇](data:text/plain;base64,ZGVmIG1lcmdlX2Zmbl9leHBlcnRzKAogICAgICAgIGZmbjogU3dpdGNoVHJhbnNmb3JtZXJzU3BhcnNlTUxQLAogICAgICAgIGdyb3VwX2xhYmVsczogdG9yY2guTG9uZ1RlbnNvciwKICAgICAgICB1c2FnZV9mcmVxdWVuY2llczogdG9yY2guRmxvYXRUZW5zb3IsCikgLT4gU3dpdGNoVHJhbnNmb3JtZXJzU3BhcnNlTUxQOgogICAgIyBFYWNoIGV4cGVydCBoYXMgYSBncm91cCBsYWJlbCBhbmQgYSB1c2FnZSBmcmVxdWVuY3kKICAgIGFzc2VydCBsZW4oZ3JvdXBfbGFiZWxzKSA9PSBsZW4odXNhZ2VfZnJlcXVlbmNpZXMpID09IGxlbihmZm4uZXhwZXJ0cykKCiAgICBmb3IgbGFiZWwgaW4gZ3JvdXBfbGFiZWxzLnVuaXF1ZSgpOgogICAgICAgIGV4cGVydF9pbmRpY2VzID0gdG9yY2gud2hlcmUoZ3JvdXBfbGFiZWxzID09IGxhYmVsKVswXQogICAgICAgIHdpdGggdG9yY2gubm9fZ3JhZCgpOgogICAgICAgICAgICAjIFN0ZXAgMS4gQ2FsY3VsYXRlIHVzYWdlLWZyZXF1ZW5jeS13ZWlnaHRlZCBhdmVyYWdpbmcKICAgICAgICAgICAgZmMxX3dlaWdodCA9IHRvcmNoLnN1bSh0b3JjaC5zdGFjaygKICAgICAgICAgICAgICAgIFtmZm4uZXhwZXJ0c1tmImV4cGVydF97ZXhwZXJ0X2lkeH0iXS5mYzEud2VpZ2h0ICogdXNhZ2VfZnJlcXVlbmNpZXNbZXhwZXJ0X2lkeF0gZm9yIGV4cGVydF9pZHggaW4KICAgICAgICAgICAgICAgICBleHBlcnRfaW5kaWNlc10sIGRpbT0wCiAgICAgICAgICAgICksIGRpbT0wKSAvIHRvcmNoLnN1bSh1c2FnZV9mcmVxdWVuY2llc1tleHBlcnRfaW5kaWNlc10sIGRpbT0wKQogICAgICAgICAgICBmYzJfd2VpZ2h0ID0gdG9yY2guc3VtKHRvcmNoLnN0YWNrKAogICAgICAgICAgICAgICAgW2Zmbi5leHBlcnRzW2YiZXhwZXJ0X3tleHBlcnRfaWR4fSJdLmZjMi53ZWlnaHQgKiB1c2FnZV9mcmVxdWVuY2llc1tleHBlcnRfaWR4XSBmb3IgZXhwZXJ0X2lkeCBpbgogICAgICAgICAgICAgICAgIGV4cGVydF9pbmRpY2VzXSwgZGltPTAKICAgICAgICAgICAgKSwgZGltPTApIC8gdG9yY2guc3VtKHVzYWdlX2ZyZXF1ZW5jaWVzW2V4cGVydF9pbmRpY2VzXSwgZGltPTApCgogICAgICAgICAgICAjIFN0ZXAgMi4gQ29weSB3ZWlnaHQgdG8gdGhlIGZpcnN0IGV4cGVydCBpbiB0aGUgZ3JvdXAKICAgICAgICAgICAgZmlyc3RfZXhwZXJ0ID0gZmZuLmV4cGVydHNbZiJleHBlcnRfe2V4cGVydF9pbmRpY2VzWzBdfSJdCiAgICAgICAgICAgIGZpcnN0X2V4cGVydC5mYzEud2VpZ2h0LmNvcHlfKGZjMV93ZWlnaHQpCiAgICAgICAgICAgIGZpcnN0X2V4cGVydC5mYzIud2VpZ2h0LmNvcHlfKGZjMl93ZWlnaHQpCgogICAgICAgICAgICAjIFN0ZXAgMy4gUmVkaXJlY3Qgb3RoZXIgbWVyZ2VkIGV4cGVydHMgdG8gdGhlIGZpcnN0IG9uZQogICAgICAgICAgICBmb3IgZXhwZXJ0X2lkeCBpbiBleHBlcnRfaW5kaWNlc1sxOl06CiAgICAgICAgICAgICAgICBmZm4uZXhwZXJ0c1tmImV4cGVydF97ZXhwZXJ0X2lkeH0iXSA9IGZmbi5leHBlcnRzW2YiZXhwZXJ0X3tleHBlcnRfaW5kaWNlc1swXX0iXQogICAgcmV0dXJuIGZmbgo=)

def merge\_ffn\_experts(

ffn: SwitchTransformersSparseMLP,

group\_labels: torch.LongTensor,

usage\_frequencies: torch.FloatTensor,

) -> SwitchTransformersSparseMLP:

# Each expert has a group label and a usage frequency

assert len(group\_labels) == len(usage\_frequencies) == len(ffn.experts)

for label in group\_labels.unique():

expert\_indices = torch.where(group\_labels == label)[0]

with torch.no\_grad():

# Step 1. Calculate usage-frequency-weighted averaging

fc1\_weight = torch.sum(torch.stack(

[ffn.experts[f"expert\_{expert\_idx}"].fc1.weight \* usage\_frequencies[expert\_idx] for expert\_idx in

expert\_indices], dim=0

), dim=0) / torch.sum(usage\_frequencies[expert\_indices], dim=0)

fc2\_weight = torch.sum(torch.stack(

[ffn.experts[f"expert\_{expert\_idx}"].fc2.weight \* usage\_frequencies[expert\_idx] for expert\_idx in

expert\_indices], dim=0

), dim=0) / torch.sum(usage\_frequencies[expert\_indices], dim=0)

# Step 2. Copy weight to the first expert in the group

first\_expert = ffn.experts[f"expert\_{expert\_indices[0]}"]

first\_expert.fc1.weight.copy\_(fc1\_weight)

first\_expert.fc2.weight.copy\_(fc2\_weight)

# Step 3. Redirect other merged experts to the first one

for expert\_idx in expert\_indices[1:]:

ffn.experts[f"expert\_{expert\_idx}"] = ffn.experts[f"expert\_{expert\_indices[0]}"]

return ffn
