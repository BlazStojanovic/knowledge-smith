---
arxiv: '2505.12082'
authors:
- Yunshui Li
- Yiyuan Ma
- Shen Yan
- Chaoyi Zhang
- Jing Liu
- Jianqiao Lu
- Ziwen Xu
- Mengzhao Chen
- Minrui Wang
- Shiyi Zhan
- Jin Ma
- Xunhao Lai
- Deyi Liu
- Yao Luo
- Xingyan Bin
- Hongbin Ren
- Mingji Han
- Wenhao Hao
- Bairen Yi
- LingJun Liu
- Bole Ma
- Xiaoying Jia
- Xun Zhou
- Siyuan Qiao
- Liang Xiang
- Yonghui Wu
parser: ar5iv
retrieved: '2026-05-25'
source: paper
title: Model Merging in Pre-training of Large Language Models
url: https://arxiv.org/abs/2505.12082
year: 2025
---

[2505.12082] Model Merging in Pre-training of Large Language Models



]ByteDance Seed
\contributionFull author list in Contributions

# Model Merging in Pre-training of Large Language Models

(May 18, 2025)

###### Abstract

Model merging has emerged as a promising technique for enhancing large language models, though its application in large-scale pre-training remains relatively unexplored.
In this paper, we present a comprehensive investigation of model merging techniques during the pre-training process.
Through extensive experiments with both dense and Mixture-of-Experts (MoE) architectures ranging from millions to over 100 billion parameters,
we demonstrate that merging checkpoints trained with constant learning rates not only achieves significant performance improvements but also enables accurate prediction of annealing behavior.
These improvements lead to both more efficient model development and significantly lower training costs.
Our detailed ablation studies on merging strategies and hyperparameters provide new insights into the underlying mechanisms while uncovering novel applications.
Through comprehensive experimental analysis, we offer the open-source community practical pre-training guidelines for effective model merging.

\correspondence

Yunshui Li at

## 1 Introduction

Modern large language models (LLMs) [[36](#bib.bib36), [1](#bib.bib1), [12](#bib.bib12), [40](#bib.bib40), [48](#bib.bib48)] have demonstrated remarkable capabilities with widespread applications across diverse tasks. Despite their exceptional performance in fundamental tasks, LLMs still face several critical challenges, including the extensive pre-training costs, discounted effectiveness of domain-specific post-training, imprecisely-predictable performance scaling, as well as the instability of large-scale training. Model merging [[49](#bib.bib49)], as a relatively young topic, presents a promising approach to alleviate these practical challenges.

Recently, the benefits of model merging have been primarily studied in the post-training stage, where several models fine-tuned on different downstream tasks are combined into a single but more versatile model [[18](#bib.bib18), [57](#bib.bib57), [51](#bib.bib51)]. For example, using the DARE [[51](#bib.bib51)] method to merge WizardLM [[45](#bib.bib45)] with WizardMath [[29](#bib.bib29)] shows a significant performance enhancement on GSM8K [[7](#bib.bib7)], raising its score from 2.2 to 66.3. In contrast, research on model merging during the pre-training phase remains scarce. Such pre-training merging typically involves combining checkpoints from a single training trajectory, as explored in LAWA [[23](#bib.bib23)] which utilizes model merging to accelerate the LLM training. However, as the model and data scales dramatically, independent researchers struggle to evaluate model merging’s impact on large-scale models, mainly due to limited access to intermediate checkpoints from extensive pre-training. Although DeepSeek [[26](#bib.bib26)] and LLaMA-3 [[11](#bib.bib11)] have both indicated their employment of model merging techniques for model development, detailed information regarding these techniques has not been publicly disclosed.

In this work, we mainly focus on model merging during the pre-training stage,
introducing Pre-trained Model Average (PMA),
a novel strategy for model-level weight merging during pre-training.
To comprehensively evaluate PMA,
we trained a diverse set of LLMs of varying sizes and architectures from scratch,
including Dense models [[11](#bib.bib11)] with parameters spanning from 411M to 70B, as well as
Mixture-of-Experts (MoE) architectures [[38](#bib.bib38)] with activated/total parameters ranging from 0.7B/7B to 20B/200B.
We first investigate the performance impact of PMA
and establish systematic evaluations across different phases of the warmup-stable-decay (WSD) learning schedule, which lately becomes a popular choice of l​rlr scheduler for LLM pre-training since [[15](#bib.bib15)].
Experimental results demonstrate that model merging during the stable training phase yields consistent performance gains at different training steps.
More remarkably, applying PMA at early-stage of the c​o​s​i​n​ecosine-decay phase usually achieve comparable or even superior performance to their final-stage annealing counterparts.
These findings suggest that during the extensively lengthy pre-training stage with constant l​rlr, PMA can serve as a fast, reliable yet low-cost simulator for the annealed performance,
enabling both faster validation cycles and significant computational savings.

Building upon our PMA framework, we first evaluate its performance with various prevalent merging strategies, including Simple Moving Average (SMA) [[20](#bib.bib20)], Weighted Moving Average (WMA) [[32](#bib.bib32)] and Exponential Moving Average (EMA) [[17](#bib.bib17)].
Notably, our experiments demonstrate that the performance differences among these methods gradually become negligible.
We further investigate how these important factors of PMA, namely, the interval between each merging checkpoint, the number of models involved in merging, and the size of the model, would affect merging performance.
Our analysis reveals two important findings:
First, the optimal merging interval exhibits a clear scaling relationship with model size.
Second, incorporating more checkpoints in the merging process consistently improves performance once training is completed.

Furthermore, we also investigated whether PMA could produce more effective initialization weights for the consecutive continued training (CT) or supervised fine-tuning (SFT) [[42](#bib.bib42)] stages to enhance the downstream model performance.
We practically observed that entering CT and SFT stages with PMA applied could yield smoother GradNorm curves, which thus helps stabilize the training dynamics yet without harming the performance, compared to initializing these stages with the latest available checkpoint as usual.
This finding inspire a novel application of model merging for training stabilization, which we dubbed as PMA-init.
We demonstrate that in scenarios when the LLM training experiences severe irrecoverable loss spikes with broken training dynamics,
applying PMA-init over NN preceding checkpoints to resume training, enables reliable recovery from unstable training trajectories.

In summary, our paper makes the following key contributions:

* •

  We present the Pre-trained Model Averaging (PMA) strategy, a novel framework for model merging during LLM pre-training.
  Through extensive experiments across model scales (from millions to over 100B parameters),
  we demonstrate that merging checkpoints from the stable training phase produces consistent and significant performance improvements.
* •

  We delved into novel applications of model merging for weight initialization (PMA-init), to help stabilize training process without harming the downstream performance, especially when it suffers from irrecoverable loss spikes with broken training dynamics.
  Through extensive experiments, we demonstrate the effectiveness of PMA-init on both CT and SFT stages.
* •

  We also comprehensively ablated various model merging techniques with their associated hyper-parameters. Our findings offer the research community practical pre-training guidelines with effective model merging.
  Nevertheless, the low cost and rapid deployment of PMA also make it a reliable and economic monitor for the pre-training process, to flexibly simulate the ultimate model performance after annealing.

Figure 1: Comparison of downstream task performance for MoE models of varying sizes under stable training, before and after model merging.

## 2 Related Work

Model merging is an emerging field undergoing rapid development, with diverse applications across various domains. Typically, model merging is implemented during the post-training phase [[18](#bib.bib18), [57](#bib.bib57), [51](#bib.bib51)], where multiple models fine-tuned on different downstream tasks are combined by merging their weights. This process effectively integrates the distinct capabilities of each individual model, resulting in a unified model that exhibits robust and comprehensive performance.

Recently, several methods have advanced this field significantly. For instance, Task Arithmetic [[18](#bib.bib18)], Ties-Merging [[46](#bib.bib46)], and AdaMerging [[50](#bib.bib50)] integrate Vision Transformer (ViT) models [[9](#bib.bib9)] trained on distinct visual classification tasks, producing a single model capable of multi-task object classification. PAPA [[21](#bib.bib21)] integrates the broad applicability of ensembling with the computational efficiency of weight averaging. MetaGPT [[57](#bib.bib57)] frames model merging as a multi-task learning problem, aiming to minimize the average loss between the merged model and individual task-specific models. Fisher Merging [[30](#bib.bib30)] employs a weighted fusion of model parameters, with weights determined by the Fisher information matrix. RegMean [[19](#bib.bib19)] elegantly addresses the merging process by formulating it as a linear regression problem solvable through closed-form solutions. Evolutionary-model-merge [[2](#bib.bib2)] efficiently optimizes merging coefficients using evolutionary algorithms. Additionally, DARE [[51](#bib.bib51)] merges multiple task-specific language models into a versatile unified model by randomly dropping and subsequently rescaling the delta parameters.

However, research on model merging during the pre-training phase remains relatively limited. Such studies typically refer to incorporating checkpoints within a single training trajectory during large language model (LLM) pre-training.
For example, LAWA [[23](#bib.bib23), [13](#bib.bib13), [25](#bib.bib25)] demonstrated that merging checkpoints at intermediate stages can significantly accelerate training. Sanyal et al. [[35](#bib.bib35)] further indicated that checkpoint averaging combined with a high learning rate in pre-training trajectories contributes to faster convergence. Additionally, Checkpoint Merging [[27](#bib.bib27)] provided a comprehensive evaluation of the effectiveness of merging checkpoints at different stages during the pre-training of the Baichuan2 [[47](#bib.bib47)] LLM. Furthermore, technical reports of large-scale models such as Deepseek V3 [[26](#bib.bib26)] and LLaMA3.1 [[11](#bib.bib11)] also mention the use of model merging techniques during pre-training, although detailed methodologies have not been publicly disclosed.
This paper primarily explores techniques for model merging within the pre-training paradigm. To the best of our knowledge, this is the first study to provide detailed technical insights into scaling model merging methods to significantly larger model sizes. We also discuss practical approaches for effective model merging and analyze its potential capabilities as well as its limitations.

## 3 Preliminaries

In this section, we describe the fundamental experimental framework, introduce the notations and concepts used in model merging, and present multiple variants of model merging techniques.

Experimental setup. In terms of model architecture, we independently trained a series of MoE and dense models. We employ a Warmup-Stable-Decay (WSD) learning rate scheduler [[15](#bib.bib15)], which begins with a short warmup period, followed by an extended period of stable training at a constant learning rate, and concludes with annealing to a relatively small learning rate. The learning rates are determined according to scaling law guidelines [[4](#bib.bib4), [24](#bib.bib24)], employing optimal values for training on an internal pretraining corpus comprising trillions of tokens.
Although specific model architectures and datasets have not yet been publicly released, we posit that our findings are not strongly tied to these particular choices, as subsequent experiments primarily focus on MoE structures. Related conclusions for dense models are provided in the Appendix [7](#S7 "7 The Effect of Model Merging in Dense Models ‣ Model Merging in Pre-training of Large Language Models").
For evaluation, we primarily report results on open-source benchmarks in both few-shot and zero-shot settings, including: ARC-Challenge [[6](#bib.bib6)], BBH [[39](#bib.bib39)], DROP [[10](#bib.bib10)], WinoGrande [[34](#bib.bib34)], HellaSwag [[54](#bib.bib54)], MMLU [[14](#bib.bib14)], C-Eval [[16](#bib.bib16)], TriviaQA [[22](#bib.bib22)], Ape210K [[55](#bib.bib55)], GSM8K [[7](#bib.bib7)], MATH [[55](#bib.bib55)], MBPP [[3](#bib.bib3)], HumanEval [[5](#bib.bib5)], AGIEval [[56](#bib.bib56)], GPQA [[33](#bib.bib33)], and MMLU-Pro [[41](#bib.bib41)]. The weighted average score across these benchmarks serves as the model’s comprehensive performance metric. Unless otherwise specified, we report this score as the model’s performance metric to ensure evaluation reliability.

Notions and concepts.
Our main focus is on model merging during pre-training, where the merged entities are sequential checkpoints along the training trajectory. Suppose we aim to merge NN models, with each model’s parameters denoted as MiM\_{i} (where ii ranges from 11 to NN). Each model has an associated weighting coefficient wiw\_{i}, and the merged model Ma​v​gM\_{avg} is computed as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Mavg=∑i=1Nwi​Mi.M\_{\text{avg}}=\sum\_{i=1}^{N}w\_{i}M\_{i}. |  | (1) |

We assume that the data consumption of these models form an arithmetic sequence with a common difference VV, formulated as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | V=Ti+1−Ti,V=T\_{i+1}-T\_{i}, |  | (2) |

where TiT\_{i} represents the cumulative number of tokens consumed by the ii-th model.

Model merging variants.
Model merging techniques vary primarily in how they assign weights (wiw\_{i}) to individual models. This paper examines three popular approaches for weight assignment, namely the Simple Moving Average (SMA), Exponential Moving Average (EMA), and Weighted Moving Average (WMA).

The first approach, Simple Moving Average (SMA), treats all models equally. For instance, when combining 10 models, each model is assigned a weight of wi=0.1w\_{i}=0.1. The SMA is formulated as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Mavg=1N​∑i=1NMi.M\_{\text{avg}}=\frac{1}{N}\sum\_{i=1}^{N}M\_{i}. |  | (3) |

The second approach, Exponential Moving Average (EMA), emphasizes later models by assigning weights that decay exponentially, making EMA more sensitive to recent changes. The EMA is expressed recursively as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Mavg(i)=α⋅Mi+(1−α)⋅Mavg(i−1),i∈[2,N],M\_{\text{avg}}^{(i)}=\alpha\cdot M\_{i}+(1-\alpha)\cdot M\_{\text{avg}}^{(i-1)},\ i\in[2,N], |  | (4) |

Here, α\alpha, the smoothing factor (typically between 0 and 1), controls the balance between the current model MiM\_{i} and the previous EMA result Mavg(i−1)M\_{\text{avg}}^{(i-1)}.

The third approach, Weighted Moving Average (WMA), also prioritizes recent models but uses a distinct weighting scheme. In WMA, each model is assigned a specific weight, often increasing linearly for later models (e.g., wi=iw\_{i}=i). The weighted sum is then normalized to compute the average, formulated as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Mavg=∑i=1Nwiwsum​Mi,wsum=∑i=1Nwi.M\_{\text{avg}}={\sum\_{i=1}^{N}\frac{w\_{i}}{w\_{\text{sum}}}M\_{i}},\quad w\_{\text{sum}}=\sum\_{i=1}^{N}w\_{i}. |  | (5) |

These methods offer flexible ways to combine models based on their recency and relevance. Choosing the right approach depends on the specific application and desired emphasis on newer data.

## 4 Experiments

In this section, we delve into the experimental core of our study, systematically addressing six critical questions surrounding model merging in the context of pre-training: 1) How does model merging affect performance? 2) How do different merging methods affect final performance? 3) How to determine the optimal interval and number of weights to merge for various model sizes? 4) Do merged pre-trained models contribute to better downstream training? 5) Does model merging improve the stability of training? 6) What processes unfold during model merging? Through these experiments, we aim to provide comprehensive insights into model merging, offering practical guidance for its application and shedding light on its theoretical underpinnings.

### 4.1 How does model merging affect model performance?

Current learning rate schedule methods mainly involve constant learning rates or cosine annealing. In our model pre-training, we employed the Warmup-Stable-Decay (WSD) strategy [[15](#bib.bib15)], which combines a constant learning rate phase with a subsequent cosine decay phase [[28](#bib.bib28)]. To explore the effects of model merging under different learning rate schedules, we conducted experiments during both constant learning rate phase and cosine dacay phase.

In the constant learning rate phase, we merged fully trained models of various sizes. As shown in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Model Merging in Pre-training of Large Language Models"), the merged models exhibited significant performance improvements across multiple downstream tasks. For example, on the Humaneval benchmark, Seed-MoE-1.3B/13B improved from 31.1 to 36.6 points, and Seed-MoE-10B/100B increased from 54.3 to 61.6 points. While larger models showed less pronounced gains on certain benchmarks, such as BBH, this was likely due to the near-saturation of these metrics. Overall, the improvements were robust and consistent across model sizes.

Next, we performed model merging in the cosine annealing phase by collecting weights from the annealing stages of Seed-MoE-1.3B/13B, Seed-MoE-10B/100B, and Seed-MoE-15B/150B. As depicted in Figure [2](#S4.F2 "Figure 2 ‣ 4.1 How does model merging affect model performance? ‣ 4 Experiments ‣ Model Merging in Pre-training of Large Language Models"), as the learning rate gradually decreased, the models converged steadily, with performance continuing to improve. Interestingly, at the early annealing stage, the results of  PMA were comparable to those at the end of the annealing process. In some cases, particularly for larger models, the merged models even surpassed those naturally annealed.

Figure 2: Comparison of overall performance for MoE models of varying sizes under annealing training, before and after model merging. The learning rate follows a cosine schedule during the annealing process. The x-axis shows the count of training tokens.

These findings raised a question: could we simplify the training process by using only the Warmup-Stable phases alongside PMA, skipping the decay phase, and avoiding learning rate adjustments? To investigate, we forked two training runs from the stable phase of Seed-MoE-1.3B/13B at 1.4T tokens. One continued with a constant learning rate, while another underwent annealing, each training for an additional 250B tokens. We then merged the models trained with the constant learning rate. As shown in Figure [3](#S4.F3 "Figure 3 ‣ 4.1 How does model merging affect model performance? ‣ 4 Experiments ‣ Model Merging in Pre-training of Large Language Models"), early in training, the merged models significantly outperformed both the constant learning rate and annealed models. Even later, their performance was comparable to the annealed models.

This suggests that pre-training with a constant learning rate, combined with model merging, can effectively match the performance of an annealed model at any point in the training process without the need for learning rate annealing. This approach accelerates model validation and significantly reduces computational resource demands.

Figure 3: Comparison of downstream task performance between model merging results under stable training and the real annealed model. The x-axis shows the count of training tokens.

### 4.2 How do different merging methods affect final performance?

In this section, we systematically evaluate how different merging strategies affect the performance of merged models.
Specifically, we focus on three distinct approaches: EMA, WMA, and SMA.
The EMA method employs exponentially decaying weights wi=α​(1−α)N−iw\_{i}=\alpha(1-\alpha)^{N-i},
giving higher importance to more recent checkpoints.
WMA assigns linearly increasing weights wi=iw\_{i}=i,
also prioritizing more recent checkpoints.
In contrast, SMA applies uniform weighting, treating all checkpoints equally regardless of their position in the training sequence.

Figure 4: Impact of different model merging methods on final model performance.

We conducted experiments on Seed-MoE-1.3/13B and showed the results in Figure [4](#S4.F4 "Figure 4 ‣ 4.2 How do different merging methods affect final performance? ‣ 4 Experiments ‣ Model Merging in Pre-training of Large Language Models"). At 204B training tokens, all merging methods enhanced model performance compared to the pre-merged model, but WMA delivered the best results. This suggests that in the early phases of training, when model weights undergo significant changes, assigning higher weights to checkpoints with more training tokens produces superior models. This is further supported by the fact that EMAα=0.2\text{EMA}\_{\alpha=0.2} outperforms EMAα=0.1\text{EMA}\_{\alpha=0.1}. However, as training advances to later stages and model weights stabilize, the performance differences between merging methods diminish. For its simplicity and stability, we primarily use SMA for model merging in subsequent experiments.

### 4.3 How to determine the optimal interval and number of weights to merge for various model sizes?

Beyond the merging technique itself, two other factors may also affect the effectiveness of model merging: the interval VV between selected models and the number of models NN. We performed ablation studies on the Seed-MoE-1.3/13B model to investigate these effects, starting with the impact of the interval. As illustrated in the upper part of Figure [5](#S4.F5 "Figure 5 ‣ 4.3 How to determine the optimal interval and number of weights to merge for various model sizes? ‣ 4 Experiments ‣ Model Merging in Pre-training of Large Language Models"), we fixed N=10N=10 and tested intervals of V=4V=4B, 8B, 16B, and 32B. Notably, at 204B with V=32V=32B, we reduced NN to 6 due to insufficient models. In the early stage of training, at 204B tokens, merged results with V=16V=16B and V=32V=32B underperformed the baseline. This is likely because large intervals incorporated unstable weights from the initial training phase, leading to significant weight disparities and suboptimal outcomes. As training progressed and weights stabilized, the performance gap across different VV settings gradually narrowed.
In practice, the optimal interval scales with model size, following these observed patterns: an interval of around 8B tokens for 1.3B/13B models, 4B tokens for 0.7B/7B models, and approximately 80B tokens for 10B/100B models. This aligns with the tendency of larger models to use larger batch sizes [[31](#bib.bib31)].

Next, we set V=8V=8B and explored how the number of merged models NN affects performance, testing N=3N=3, 6, 10, and 15. As shown in the lower part of Figure [5](#S4.F5 "Figure 5 ‣ 4.3 How to determine the optimal interval and number of weights to merge for various model sizes? ‣ 4 Experiments ‣ Model Merging in Pre-training of Large Language Models"), early in training, incorporating more models introduced unstable weights, which reduced the performance of merged models. However, once training was complete, merging a larger number of models led to significant performance improvements. Notably, the overall performance for N=3N=3 was nearly 1 point lower than for N=15N=15. To strike a balance between computational cost and performance gains, we opted for N=10N=10 in further experiments.

Figure 5: Impact of different model merging hyper-parameters on final model performance.

### 4.4 Do merged pre-trained models contribute to better downstream training?

Figure 6: Comparisons of loss curves (left) and performance metrics (right) during CT stage with varying l​rlr schedules, where a cosine scheduler is adopted to decay learning rate from l​rp​e​a​klr\_{peak} to l​re​n​dlr\_{end} (denoted as l​rp​e​a​k→l​re​n​dlr\_{peak}\rightarrow lr\_{end}). PMA and baseline, stand for whether our PMA-init technique is employed or not, respectively.

A complete LLM training process typically involves multiple stages, which are pretraining, continual training (CT), supervised fine-tuning (SFT) and reinforcement learning (RL) in sequence.
In light of the capacity of PMA to improve pretraining performance, we conjecture that merged pretrained models may similarly prove beneficial for downstream stages.
To verify this hypothesis, we initialized downstream training with PMA, which we dubbed as PMA-init, and investigated its impacts over the baselines (which are initialized from their original checkpoints) for both CT and SFT stages.

CT stage.
We first conducted an ablation study to assess the sensitivity of the PMA-init of the CT stage with varying learning rate schedules.
Specifically, we experimented with Seed-MoE-0.7B/7B models merged after stable training on approximately 1 trillion tokens.
As illustrated in Figure [6](#S4.F6 "Figure 6 ‣ 4.4 Do merged pre-trained models contribute to better downstream training? ‣ 4 Experiments ‣ Model Merging in Pre-training of Large Language Models") (left), the initialization weights obtained via PMA consistently achieved marginally lower loss at the initial training phase, against the baseline with the same training configuration.
As training progresses, the loss values for models with different initialization weights converge to comparable levels.
It’s worth noting that in the loss curve, the purple line significantly overlaps with the blue line, and the brown line significantly overlaps with the pink line.
Another observation is made in the Figure [6](#S4.F6 "Figure 6 ‣ 4.4 Do merged pre-trained models contribute to better downstream training? ‣ 4 Experiments ‣ Model Merging in Pre-training of Large Language Models") (right), where evaluation on the MMLU benchmark reveals that the PMA-init models outperform the baseline early in training. While these models tend to retain a slight performance edge in later stages, their results on other tasks may be slightly suboptimal, leading to overall performance parity with the baseline. Experiments across varied learning rate schedules corroborate these findings, indicating that models converge to similar performance levels by the end of training, and no extensive learning rate tuning is required for PMA-init.

SFT stage.
We next analyzed the impact of PMA-init on the SFT stage, where the detailed results can be found in the Appendix [8](#S8 "8 Model Merging at the CT Stage for Supervised Fine-Tuning ‣ Model Merging in Pre-training of Large Language Models").
Although initialization with merged weights occasionally yields performance improvements, such gains are not consistently observed.
Nonetheless, this approach does not adversely affect downstream training outcomes and may be a viable strategy for researchers seeking to enhance model performance.

### 4.5 Does model merging improve the training stability?

In large-scale LLM training, infrastructure issues are almost inevitable and often lead to training instability phenomena such as loss spikes or diverging.
Specifically, a loss spike occurs when, at a specific point during the multi-stage training, the model’s predictions deteriorate significantly compared to previous iterations.
This phenomenon is often observed alongside gradient norm (GradNorm) explosion during backpropagation, which causes large weight updates and eventually lead to a irrecoverable spike in its loss function [[8](#bib.bib8)].
In the experiments detailed in Section [4.4](#S4.SS4 "4.4 Do merged pre-trained models contribute to better downstream training? ‣ 4 Experiments ‣ Model Merging in Pre-training of Large Language Models"), as illustrated in Figure [7](#S4.F7 "Figure 7 ‣ 4.5 Does model merging improve the training stability? ‣ 4 Experiments ‣ Model Merging in Pre-training of Large Language Models") (left), we observed that a model initialized with PMA-init for SFT stage demonstrated a notably more stable GradNorm metric compared to the baseline. This stability is also evident in the reduced frequency of loss spikes relative to the baseline.
Since applying PMA-init for downstream training does not impact the model’s final performance and remains robust across different learning rates, we established a series of experiments to explore whether model merging could enhance training stability.

Given the extremely high expenses associated, it is unfeasible to conduct a direct analysis of training instability in LLM pre-training.
Experiments [[44](#bib.bib44)] show that small models using a relatively large learning rate will exhibit unstable training characteristics similar to those of large models.
We thus reproduce the instability phenomena on small models to study the influence of our PMA-init on training stability.
In one such experiment, we trained a 330M/3.3B MoE model from scratch using an exceptionally high learning rate of 6e-3.
As shown in Figure [7](#S4.F7 "Figure 7 ‣ 4.5 Does model merging improve the training stability? ‣ 4 Experiments ‣ Model Merging in Pre-training of Large Language Models") (right), the model overshot the optimal weights, resulting in unstable training and abrupt loss spikes as expected, and was irreversible to its original trajectory.
To address this, we adopted PMA-init with three checkpoints saved before the training collapse happened, to resume the pre-training process. As depicted by the red line in Figure [7](#S4.F7 "Figure 7 ‣ 4.5 Does model merging improve the training stability? ‣ 4 Experiments ‣ Model Merging in Pre-training of Large Language Models") (right), the resumed training process stabilized, successfully navigating past the point of the loss spike and continuing along its original training trajectory.

These results highlight that PMA-init can reliably enhance the multi-stage training stability.
When a loss spike occurs, one can merge the model checkpoints from before the spike and resume training from that point.
This approach provides an alternative solution to avoid retraining the model from scratch, thereby substantially reducing the waste of computational resources.

Figure 7: Left: GradNorm comparisons for SFT training initialized with PMA-init. Right: Comparison of pre-training loss curves between resuming with PMA-init and the original training.

### 4.6 Investigating the Mechanisms of Model Merging

To gain deeper insight into the underlying mechanisms that enable model merging to be effective, we provide both qualitative and quantitative analyses, employing mathematical derivations and visualizations of weight distributions.

We begin with a second-order Taylor expansion of the loss function L​(θ)L(\theta) around an optimal parameter set θ∗\theta^{\*}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | L​(θ)≈L​(θ∗)+(θ−θ∗)T​∇L​(θ∗)+12​(θ−θ∗)T​H​(θ−θ∗),L(\theta)\approx L(\theta^{\*})+(\theta-\theta^{\*})^{T}\nabla L(\theta^{\*})+\frac{1}{2}(\theta-\theta^{\*})^{T}H(\theta-\theta^{\*}), |  | (6) |

where HH is the Hessian matrix of the loss function evaluated at θ∗\theta^{\*} (the matrix of second partial derivatives), which captures curvature information. Since θ∗\theta^{\*} is an optimal point, the gradient ∇L​(θ∗)\nabla L(\theta^{\*}) is zero. Thus, the expansion simplifies to:

|  |  |  |  |
| --- | --- | --- | --- |
|  | L​(θ)≈L​(θ∗)+12​(θ−θ∗)T​H​(θ−θ∗).L(\theta)\approx L(\theta^{\*})+\frac{1}{2}(\theta-\theta^{\*})^{T}H(\theta-\theta^{\*}). |  | (7) |

Consider kk sets of model parameters θ1,θ2,…,θk\theta\_{1},\theta\_{2},\ldots,\theta\_{k}. Let the deviation vector of each model ii from the optimal parameters be δi=θi−θ∗\delta\_{i}=\theta\_{i}-\theta^{\*}. The loss for each model ii can then be approximated as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | L​(θi)≈L​(θ∗)+12​δiT​H​δi.L(\theta\_{i})\approx L(\theta^{\*})+\frac{1}{2}\delta\_{i}^{T}H\delta\_{i}. |  | (8) |

The average loss of these kk individual models is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1k​∑i=1kL​(θi)≈L​(θ∗)+12​k​∑i=1kδiT​H​δi.\frac{1}{k}\sum\_{i=1}^{k}L(\theta\_{i})\approx L(\theta^{\*})+\frac{1}{2k}\sum\_{i=1}^{k}\delta\_{i}^{T}H\delta\_{i}. |  | (9) |

The parameters of the merged model are θavg=1k​∑i=1kθi\theta\_{\text{avg}}=\frac{1}{k}\sum\_{i=1}^{k}\theta\_{i}. The deviation of this merged model from the optimal parameters is θavg−θ∗=1k​∑i=1kδi\theta\_{\text{avg}}-\theta^{\*}=\frac{1}{k}\sum\_{i=1}^{k}\delta\_{i}.
The loss for the merged model is approximated by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | L​(θavg)≈L​(θ∗)+12​(1k​∑i=1kδi)T​H​(1k​∑i=1kδi)L(\theta\_{\text{avg}})\approx L(\theta^{\*})+\frac{1}{2}\left(\frac{1}{k}\sum\_{i=1}^{k}\delta\_{i}\right)^{T}H\left(\frac{1}{k}\sum\_{i=1}^{k}\delta\_{i}\right) |  | (10) |

Expanding the quadratic term:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 12​(1k​∑i=1kδi)T​H​(1k​∑i=1kδi)=12​k2​∑i=1k∑j=1kδiT​H​δj.\frac{1}{2}\left(\frac{1}{k}\sum\_{i=1}^{k}\delta\_{i}\right)^{T}H\left(\frac{1}{k}\sum\_{i=1}^{k}\delta\_{i}\right)=\frac{1}{2k^{2}}\sum\_{i=1}^{k}\sum\_{j=1}^{k}\delta\_{i}^{T}H\delta\_{j}. |  | (11) |

This can be rewritten by separating diagonal and off-diagonal terms:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 12​k2​(∑i=1kδiT​H​δi+∑i=1k∑j≠iδiT​H​δj).\frac{1}{2k^{2}}\left(\sum\_{i=1}^{k}\delta\_{i}^{T}H\delta\_{i}+\sum\_{i=1}^{k}\sum\_{j\neq i}\delta\_{i}^{T}H\delta\_{j}\right). |  | (12) |

For the merged model to have a lower loss than the average loss of the individual models, i.e., L​(θavg)<1k​∑i=1kL​(θi)L(\theta\_{\text{avg}})<\frac{1}{k}\sum\_{i=1}^{k}L(\theta\_{i}), the following condition must hold:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 12​k2​(∑i=1kδiT​H​δi+∑i=1k∑j≠iδiT​H​δj)<12​k​∑i=1kδiT​H​δi.\frac{1}{2k^{2}}\left(\sum\_{i=1}^{k}\delta\_{i}^{T}H\delta\_{i}+\sum\_{i=1}^{k}\sum\_{j\neq i}\delta\_{i}^{T}H\delta\_{j}\right)<\frac{1}{2k}\sum\_{i=1}^{k}\delta\_{i}^{T}H\delta\_{i}. |  | (13) |

Multiplying by 2​k22k^{2} and rearranging terms, we get:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑i=1kδiT​H​δi+∑i=1k∑j≠iδiT​H​δj<k​∑i=1kδiT​H​δi.\sum\_{i=1}^{k}\delta\_{i}^{T}H\delta\_{i}+\sum\_{i=1}^{k}\sum\_{j\neq i}\delta\_{i}^{T}H\delta\_{j}<k\sum\_{i=1}^{k}\delta\_{i}^{T}H\delta\_{i}. |  | (14) |

Which simplifies to:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑i=1k∑j≠iδiT​H​δj<(k−1)​∑i=1kδiT​H​δi\sum\_{i=1}^{k}\sum\_{j\neq i}\delta\_{i}^{T}H\delta\_{j}<(k-1)\sum\_{i=1}^{k}\delta\_{i}^{T}H\delta\_{i} |  | (15) |

Assuming HH is a positive definite matrix (which is generally true around a local minimum), then each term δiT​H​δi>0\delta\_{i}^{T}H\delta\_{i}>0. The inequality is more easily satisfied if the off-diagonal terms δiT​H​δj\delta\_{i}^{T}H\delta\_{j} (for i≠ji\neq j) are predominantly negative. This "negative correlation" in the context of the Hessian means that the deviation vectors point in somewhat opposing directions relative to the curvature of the loss landscape.

Figure 8: Visualization of MMLU score contour lines, comparing the weights of an original model with those of a merged model. Black dots represent the parameter locations of various individual model checkpoints.

This mathematical analysis can be intuitively interpreted as follows:
1. The effectiveness of model weight merging stems from the fact that different model checkpoints, representing different points in the training trajectory, have explored different local regions or directions within the parameter space.
2. When these explorations exhibit a degree of "complementarity" concerning the geometric structure of the loss function (captured by the Hessian and the cross-terms δiT​H​δj\delta\_{i}^{T}H\delta\_{j}), their average can position the merged model closer to an optimal point than the individual models might be on average.
3. This helps explain why merging models, particularly those from a stable yet ongoing training phase, often improves performance. The averaging process can smooth out idiosyncrasies of individual checkpoints.
This analysis suggests that weight merging is not merely a simple averaging of parameters but rather a process that can leverage the geometric structure of the loss landscape and the diversity among the models being merged.

Additionally, we selected several checkpoints from the pre-training of Seed-MoE-1.3B/13B and visualized the average distribution of two selected parameters from a specific layer. Using these points, we generated contour lines for MMLU scores, as illustrated in Figure [8](#S4.F8 "Figure 8 ‣ 4.6 Investigating the Mechanisms of Model Merging ‣ 4 Experiments ‣ Model Merging in Pre-training of Large Language Models"). The weight positions of various individual models are marked as black dots. These dots are distributed along the MMLU score contours, revealing a discernible "complementary" pattern. The averaged weight position (representative of the merged model) is often situated closer to a region of higher MMLU scores (a better optimum) than many individual model checkpoints. This visualization also provides an intuitive explanation for why model merging yields diminished improvements when models are annealed to a very low learning rate: at such a stage, the models to be merged are already tightly converged within a specific local optimum. Merging them essentially averages points within this already narrow basin, making it unlikely to escape to a significantly better or different optimal region.

## 5 Conclusion

This research pioneers a deeper exploration of model merging within the challenging pre-training stage of large-scale models. By training a spectrum of MoE and Dense models and performing rigorous ablations, we established that merging checkpoints from stable training phases not only yields significant performance gains and predicts annealing but also streamlines development and reduces costs. Our work provides concrete guidance on merging strategies, optimal parameters, and downstream applications, alongside insights into the underlying mechanisms. These contributions equip the open-source community with the knowledge and tools for more efficient model development through pre-training merging.

## Contributions

Project Lead

Yunshui Li1

Algorithm

Yunshui Li1, Yiyuan Ma1, Shen Yan1,2, Chaoyi Zhang1, Jing Liu1, Jianqiao Lu1,3, Minrui Wang1, Mengzhao Chen1,3, Xunhao Lai1,2, Jin Ma1, Shiyi Zhan1, Deyi Liu1, Yao Luo1, Xingyan Bin1

Infrastructure

Ziwen Xu1, Mingji Han1, Wenhao Hao1, Bairen Yi1, Lingjun Liu1, Bole Ma1, Hongbin Ren1, Xiaoying Jia1

Supervision

Yiyuan Ma1, Xun Zhou1, Siyuan Qiao1, Liang Xiang1, Yonghui Wu1

Affiliation

1 ByteDance Seed

2 Peking University

3 The University of Hong Kong

## 6 Acknowledgments

We thank Chengyin Xu, Yantao Du, Xinran Zhao, Renming Pang, Shuang Wu, Bohong Wu, Yutao Zeng, Chen Zheng, Yuan Yang as well as other colleagues at ByteDance for their support for this project.

## References

* Achiam et al. [2023]

  Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al.
  Gpt-4 technical report.
  *arXiv preprint arXiv:2303.08774*, 2023.
* Akiba et al. [2025]

  Takuya Akiba, Makoto Shing, Yujin Tang, Qi Sun, and David Ha.
  Evolutionary optimization of model merging recipes.
  *Nature Machine Intelligence*, pages 1–10, 2025.
* Austin et al. [2021]

  Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al.
  Program synthesis with large language models.
  *arXiv preprint arXiv:2108.07732*, 2021.
* Bi et al. [2024]

  Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al.
  Deepseek llm: Scaling open-source language models with longtermism.
  *arXiv preprint arXiv:2401.02954*, 2024.
* Chen et al. [2021]

  Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al.
  Evaluating large language models trained on code.
  *arXiv preprint arXiv:2107.03374*, 2021.
* Clark et al. [2018]

  Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord.
  Think you have solved question answering? try arc, the ai2 reasoning challenge.
  *arXiv preprint arXiv:1803.05457*, 2018.
* Cobbe et al. [2021]

  Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al.
  Training verifiers to solve math word problems.
  *arXiv preprint arXiv:2110.14168*, 2021.
* Cohen et al. [2021]

  Jeremy Cohen, Simran Kaur, Yuanzhi Li, J Zico Kolter, and Ameet Talwalkar.
  Gradient descent on neural networks typically occurs at the edge of stability.
  In *International Conference on Learning Representations*, 2021.
* Dosovitskiy et al. [2021]

  Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby.
  An image is worth 16x16 words: Transformers for image recognition at scale.
  In *International Conference on Learning Representations*, 2021.
* Dua et al. [2019]

  Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner.
  DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs.
  In Jill Burstein, Christy Doran, and Thamar Solorio, editors, *Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics*, pages 2368–2378, 2019.
* Grattafiori et al. [2024]

  Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al.
  The llama 3 herd of models.
  *arXiv e-prints*, pages arXiv–2407, 2024.
* Guo et al. [2025]

  Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al.
  Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.
  *arXiv preprint arXiv:2501.12948*, 2025.
* Hägele et al. [2024]

  Alex Hägele, Elie Bakouch, Atli Kosson, Leandro Von Werra, Martin Jaggi, et al.
  Scaling laws and compute-optimal training beyond fixed training durations.
  *Advances in Neural Information Processing Systems*, 37:76232–76264, 2024.
* Hendrycks et al. [2021]

  Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
  Measuring massive multitask language understanding.
  In *International Conference on Learning Representations*, 2021.
* Hu et al. [2024]

  Shengding Hu, Yuge Tu, Xu Han, Ganqu Cui, Chaoqun He, Weilin Zhao, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Xinrong Zhang, Zhen Leng Thai, Chongyi Wang, Yuan Yao, Chenyang Zhao, Jie Zhou, Jie Cai, Zhongwu Zhai, Ning Ding, Chao Jia, Guoyang Zeng, dahai li, Zhiyuan Liu, and Maosong Sun.
  MiniCPM: Unveiling the potential of small language models with scalable training strategies.
  In *First Conference on Language Modeling*, 2024.
* Huang et al. [2023]

  Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, jiayi lei, Yao Fu, Maosong Sun, and Junxian He.
  C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models.
  In *Conference on Neural Information Processing Systems Datasets and Benchmarks Track*, 2023.
* Hunter [1986]

  J Stuart Hunter.
  The exponentially weighted moving average.
  *Journal of quality technology*, 18(4):203–210, 1986.
* Ilharco et al. [2023]

  Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi.
  Editing models with task arithmetic.
  In *International Conference on Learning Representations*, 2023.
* Jin et al. [2023]

  Xisen Jin, Xiang Ren, Daniel Preotiuc-Pietro, and Pengxiang Cheng.
  Dataless knowledge fusion by merging weights of language models.
  In *The International Conference on Learning Representations*, 2023.
* Johnston et al. [1999]

  FR Johnston, John E Boyland, Maureen Meadows, and E Shale.
  Some properties of a simple moving average when applied to forecasting a time series.
  *Journal of the Operational Research Society*, 50(12):1267–1271, 1999.
* Jolicoeur-Martineau et al. [2023]

  Alexia Jolicoeur-Martineau, Emy Gervais, Kilian Fatras, Yan Zhang, and Simon Lacoste-Julien.
  Population parameter averaging (papa).
  *arXiv preprint arXiv:2304.03094*, 2023.
* Joshi et al. [2017]

  Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer.
  TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension.
  In Regina Barzilay and Min-Yen Kan, editors, *Proceedings of the Annual Meeting of the Association for Computational Linguistics*, pages 1601–1611, 2017.
* Kaddour [2022]

  Jean Kaddour.
  Stop wasting my time! saving days of imagenet and BERT training with latest weight averaging.
  In *Advances in Neural Information Processing Systems Workshop*, 2022.
* Kaplan et al. [2020]

  Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei.
  Scaling laws for neural language models.
  *arXiv preprint arXiv:2001.08361*, 2020.
* Li et al. [2022]

  Tao Li, Zhehao Huang, Qinghua Tao, Yingwen Wu, and Xiaolin Huang.
  Trainable weight averaging: Efficient training by optimizing historical solutions.
  In *The Eleventh International Conference on Learning Representations*, 2022.
* Liu et al. [2024a]

  Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al.
  Deepseek-v3 technical report.
  *arXiv preprint arXiv:2412.19437*, 2024a.
* Liu et al. [2024b]

  Deyuan Liu, Zecheng Wang, Bingning Wang, Weipeng Chen, Chunshan Li, Zhiying Tu, Dianhui Chu, Bo Li, and Dianbo Sui.
  Checkpoint merging via bayesian optimization in llm pretraining.
  *arXiv preprint arXiv:2403.19390*, 2024b.
* Loshchilov and Hutter [2017]

  Ilya Loshchilov and Frank Hutter.
  SGDR: Stochastic gradient descent with warm restarts.
  In *International Conference on Learning Representations*, 2017.
* Luo et al. [2025]

  Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jian-Guang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, Yansong Tang, and Dongmei Zhang.
  Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct.
  In *International Conference on Learning Representations*, 2025.
* Matena and Raffel [2022]

  Michael S Matena and Colin A Raffel.
  Merging models with fisher-weighted averaging.
  *Advances in Neural Information Processing Systems*, 35:17703–17716, 2022.
* McCandlish et al. [2018]

  Sam McCandlish, Jared Kaplan, Dario Amodei, and OpenAI Dota Team.
  An empirical model of large-batch training.
  *arXiv preprint arXiv:1812.06162*, 2018.
* Perry [2010]

  Marcus B Perry.
  The weighted moving average technique.
  *Wiley Encyclopedia of Operations Research and Management Science*, 2010.
* Rein et al. [2024]

  David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman.
  Gpqa: A graduate-level google-proof q&a benchmark.
  In *First Conference on Language Modeling*, 2024.
* Sakaguchi et al. [2021]

  Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi.
  Winogrande: An adversarial winograd schema challenge at scale.
  *Communications of the ACM*, 64(9):99–106, 2021.
* Sanyal et al. [2024]

  Sunny Sanyal, Atula Tejaswi Neerkaje, Jean Kaddour, Abhishek Kumar, and sujay sanghavi.
  Early weight averaging meets high learning rates for LLM pre-training.
  In *First Conference on Language Modeling*, 2024.
* Seed et al. [2025]

  ByteDance Seed, Yufeng Yuan, Yu Yue, Mingxuan Wang, Xiaochen Zuo, Jiaze Chen, Lin Yan, Wenyuan Xu, Chi Zhang, Xin Liu, et al.
  Seed-thinking-v1. 5: Advancing superb reasoning models with reinforcement learning.
  *arXiv preprint arXiv:2504.13914*, 2025.
* Shao et al. [2024]

  Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al.
  Deepseekmath: Pushing the limits of mathematical reasoning in open language models.
  *arXiv preprint arXiv:2402.03300*, 2024.
* Shazeer et al. [2017]

  Noam Shazeer, \*Azalia Mirhoseini, \*Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean.
  Outrageously large neural networks: The sparsely-gated mixture-of-experts layer.
  In *International Conference on Learning Representations*, 2017.
* Suzgun et al. [2023]

  Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, and Jason Wei.
  Challenging BIG-bench tasks and whether chain-of-thought can solve them.
  In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, *Findings of the Association for Computational Linguistics*, pages 13003–13051, 2023.
* Team et al. [2024]

  Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al.
  Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context.
  *arXiv preprint arXiv:2403.05530*, 2024.
* Wang et al. [2024]

  Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al.
  Mmlu-pro: A more robust and challenging multi-task language understanding benchmark.
  In *The Conference on Neural Information Processing Systems Datasets and Benchmarks Track*, 2024.
* Wei et al. [2022]

  Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le.
  Finetuned language models are zero-shot learners.
  In *International Conference on Learning Representations*, 2022.
* White et al. [2025]

  Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Benjamin Feuer, Siddhartha Jain, Ravid Shwartz-Ziv, Neel Jain, Khalid Saifullah, Sreemanti Dey, Shubh-Agrawal, Sandeep Singh Sandha, Siddartha Venkat Naidu, Chinmay Hegde, Yann LeCun, Tom Goldstein, Willie Neiswanger, and Micah Goldblum.
  Livebench: A challenging, contamination-limited LLM benchmark.
  In *International Conference on Learning Representations*, 2025.
* Wortsman et al. [2024]

  Mitchell Wortsman, Peter J Liu, Lechao Xiao, Katie E Everett, Alexander A Alemi, Ben Adlam, John D Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, Jeffrey Pennington, Jascha Sohl-Dickstein, Kelvin Xu, Jaehoon Lee, Justin Gilmer, and Simon Kornblith.
  Small-scale proxies for large-scale transformer training instabilities.
  In *International Conference on Learning Representations*, 2024.
* Xu et al. [2024]

  Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Qingwei Lin, and Daxin Jiang.
  WizardLM: Empowering large pre-trained language models to follow complex instructions.
  In *The International Conference on Learning Representations*, 2024.
* Yadav et al. [2023]

  Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal.
  TIES-merging: Resolving interference when merging models.
  In *Conference on Neural Information Processing Systems*, 2023.
* Yang et al. [2023]

  Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, et al.
  Baichuan 2: Open large-scale language models.
  *arXiv preprint arXiv:2309.10305*, 2023.
* Yang et al. [2024a]

  An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al.
  Qwen2. 5 technical report.
  *arXiv preprint arXiv:2412.15115*, 2024a.
* Yang et al. [2024b]

  Enneng Yang, Li Shen, Guibing Guo, Xingwei Wang, Xiaochun Cao, Jie Zhang, and Dacheng Tao.
  Model merging in llms, mllms, and beyond: Methods, theories, applications and opportunities.
  *arXiv preprint arXiv:2408.07666*, 2024b.
* Yang et al. [2024c]

  Enneng Yang, Zhenyi Wang, Li Shen, Shiwei Liu, Guibing Guo, Xingwei Wang, and Dacheng Tao.
  Adamerging: Adaptive model merging for multi-task learning.
  In *International Conference on Learning Representations*, 2024c.
* Yu et al. [2024]

  Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li.
  Language models are super mario: Absorbing abilities from homologous models as a free lunch.
  In *International Conference on Machine Learning*, 2024.
* Yu et al. [2025]

  Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al.
  Dapo: An open-source llm reinforcement learning system at scale.
  *arXiv preprint arXiv:2503.14476*, 2025.
* Yuan et al. [2025]

  Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, Xiangpeng Wei, et al.
  Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks.
  *arXiv preprint arXiv:2504.05118*, 2025.
* Zellers et al. [2019]

  Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi.
  Hellaswag: Can a machine really finish your sentence?
  *arXiv preprint arXiv:1905.07830*, 2019.
* Zhao et al. [2020]

  Wei Zhao, Mingyue Shang, Yang Liu, Liang Wang, and Jingming Liu.
  Ape210k: A large-scale and template-rich dataset of math word problems.
  *arXiv preprint arXiv:2009.11506*, 2020.
* Zhong et al. [2024]

  Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan.
  AGIEval: A human-centric benchmark for evaluating foundation models.
  In Kevin Duh, Helena Gomez, and Steven Bethard, editors, *Findings of the Association for Computational Linguistics: NAACL*, pages 2299–2314, 2024.
* Zhou et al. [2024]

  Yuyan Zhou, Liang Song, Bingning Wang, and Weipeng Chen.
  MetaGPT: Merging large language models using model exclusive task arithmetic.
  In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, *Proceedings of the Conference on Empirical Methods in Natural Language Processing*, pages 1711–1724, 2024.

\beginappendix

## 7 The Effect of Model Merging in Dense Models

We also conducted model merging experiments on Dense architecture models, ranging from small Seed-Dense-411M models to large Seed-Dense-70B models. Since the 411M and 2B models were not sufficiently trained, we used a configuration of N=6 for merging, with weight intervals (V) of 2B and 5B tokens, respectively. For the 8B and 70B models, which were trained more thoroughly, we used N=10, with V values of 15B and 40B for merging. As shown in Figure [9](#S7.F9 "Figure 9 ‣ 7 The Effect of Model Merging in Dense Models ‣ Model Merging in Pre-training of Large Language Models"), models of different sizes achieved significant improvements on downstream tasks after model merging. Notably, the performance gains of larger models were not smaller than those of smaller models. Specifically, Seed-Dense-70B improved from 50.6 to 57.9 on humaneval and from 85.9 to 91.3 on GSM8K. This further validates the robustness and generalization ability of PMA, demonstrating that it can work across different model architectures and sizes.

Figure 9: Comparison of downstream task performance for dense models of varying sizes under stable training, before
and after model merging.

## 8 Model Merging at the CT Stage for Supervised Fine-Tuning

We conducted an ablation study to assess the sensitivity of the PMA-init during the SFT stage to varying learning rate schedules.
This study included experiments on merged Seed-MoE-15B/150B models following stable training on approximately 16T tokens, as well as after further training on 1T tokens with cosine annealing.
We conducted SFT training for 220M tokens using both the original weights and PMA-init weights. For the original weights, we used a cosine learning rate schedule with an initial learning rate of 2e-5 and an end learning rate of 2e-6. For the PMA-init weights, we used cosine schedules with initial learning rates of 1e-5, 2e-5, and 4e-5, all with an end learning rate of 2e-6. We evaluated the trained models using Open-Benchmark, which includes MMLU [[14](#bib.bib14)], LiveBench [[43](#bib.bib43)], AMC-2023, GPQA [[33](#bib.bib33)] and LiveCodeBench [[43](#bib.bib43)], as well as our in-house evaluation set comprising OOD, Reasoning, and Instruction Following assessments.

Table 1: Comparisons of performance metrics during SFT stage with varying l​rlr schedules, where a cosine scheduler is adopted to decay learning rate from l​rp​e​a​klr\_{peak} to l​re​n​dlr\_{end} (denoted as l​rp​e​a​k→l​re​n​dlr\_{peak}\rightarrow lr\_{end}). PMA and baseline, stand for whether our PMA-init technique is employed or not, respectively. IF refers to Instruction Following.

| Model | Open-Benchmark | | | | | In-house Evaluation | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MMLU | LiveBench | AMC-2023 | GPQA | LiveCodeBench | OOD | Reasoning | IF |
| Baseline2​e−5→2​e−6\textbf{Baseline}\_{2e^{-5}\rightarrow 2e^{-6}} | 86.8 | 50.5 | 61.0 | 55.2 | 39.7 | 32.6 | 32.1 | 36.3 |
| PMA2​e−5→2​e−6\textbf{PMA}\_{2e^{-5}\rightarrow 2e^{-6}} | 87.1 | 52.0 | 64.0 | 54.0 | 39.4 | 34.7 | 34.0 | 38.8 |
| PMA1​e−5→2​e−6\textbf{PMA}\_{1e^{-5}\rightarrow 2e^{-6}} | 87.2 | 53.2 | 65.5 | 54.4 | 39.7 | 33.8 | 33.2 | 37.3 |
| PMA4​e−5→2​e−6\textbf{PMA}\_{4e^{-5}\rightarrow 2e^{-6}} | 87.0 | 51.3 | 61.4 | 54.0 | 39.2 | 31.8 | 32.6 | 37.2 |

As shown in Table [1](#S8.T1 "Table 1 ‣ 8 Model Merging at the CT Stage for Supervised Fine-Tuning ‣ Model Merging in Pre-training of Large Language Models"), with the same learning rate, PMA-init significantly outperformed the baseline on both Open-Benchmark and our in-house evaluations.
Notably, on the in-house evaluation set, we observed improvements of over two points in OOD and Instruction Following, and a 1.9-point increase in Reasoning.
In the other two experiments with different learning rates, we also saw some degree of improvement compared to the baseline, especially with PMA1​e−5→2​e−6{}\_{1e^{-5}\rightarrow 2e^{-6}}, which showed gains of 2.7 points on Livebench and 4.5 points on AMC-2023.

However, we were unable to replicate such significant gains in subsequent experiments with other model sizes, although it did not negatively impact the final downstream model performance. Therefore, as a low-cost approach, PMA-init is worth trying to obtain a more powerful downstream model.

## 9 Limitations

In our study, we thoroughly investigated the potential of model merging in the pre-training phase, offering significant advantages for teams working on large-scale model pre-training to pursue more daring explorations. This is due to the fact that model merging can replicate the benefits of simulated annealing, greatly shortening the exploration period during pre-training. While our experiments were extensive, certain aspects still remain open for deeper research.

In our experiments, we defaulted to using the optimal learning rate derived from the scaling law for model training, without extensively exploring the impact of learning rate on model merging. In our practice, we believe that training with a higher learning rate could lead to a better model through model merging, which aligns with the findings in [[35](#bib.bib35)]. However, due to the high computational cost, we did not further quantify the impact of learning rate on model merging in a more detailed manner.

Additionally, this paper primarily focuses on the application of model merging in pre-training. In reality, due to innovations in RL algorithms [[52](#bib.bib52), [53](#bib.bib53), [37](#bib.bib37)], RL training has become more stable and often involves longer training cycles, during which a series of adjacent weights can be obtained. This paper does not investigate model merging in the context of post-training scenarios, and we leave this aspect for future research.
