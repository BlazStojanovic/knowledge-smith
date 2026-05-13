---
arxiv: '2603.16127'
authors:
- Kazuki Yano
- Shun Kiyono
- Sosuke Kobayashi
- Sho Takase
- Jun Suzuki
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning
url: https://arxiv.org/abs/2603.16127
year: 2026
---

[2603.16127] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning














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



# Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning

Kazuki Yano†, Shun Kiyono‡, Sosuke Kobayashi†, Sho Takase†, Jun Suzuki†
  
†Tohoku University
  
‡SB Intuitions
  
yano.kazuki@dc.tohoku.ac.jp is-failab-research@grp.tohoku.ac.jp

###### Abstract

We investigate the role of learning rate scheduling in the large-scale pre-training of large language models, focusing on its influence on downstream performance after supervised fine-tuning (SFT).
Decay-based learning rate schedulers are widely used to minimize pre-training loss.
However, despite their widespread use, how these schedulers affect performance after SFT remains underexplored.
In this paper, we examine Warmup-Stable-Only (WSO), which maintains a constant learning rate after warmup without any decay.
Through experiments with 1B and 8B parameter models, we show that WSO consistently outperforms decay-based schedulers in terms of performance after SFT, even though decay-based schedulers may exhibit better performance after pre-training.
The result also holds across different regimes with mid-training and over-training.
Loss landscape analysis further reveals that decay-based schedulers lead models into sharper minima, whereas WSO preserves flatter minima that support adaptability.
These findings indicate that applying LR decay to improve pre-training metrics may compromise downstream adaptability.
Our work also provides practical guidance for training and model release strategies, highlighting that pre-training models with WSO enhances their adaptability for downstream tasks.

## 1 Introduction

Learning rate (LR) scheduling is arguably one of the most critical yet operationally challenging aspects of large language model (LLM) pre-training.
Although Cosine decay has been conventionally employed in numerous models (Brown et al., [2020](#bib.bib5); Le et al., [2022](#bib.bib27); Touvron et al., [2023a](#bib.bib51)), it has proven inflexible in recent training paradigms such as continual pre-training, as it requires heuristic tuning of the LR from the decayed value (Hägele et al., [2024](#bib.bib14); Ibrahim et al., [2024](#bib.bib19)).
To address this inflexibility, recent studies have introduced Warmup-Stable-Decay (WSD), which keeps the LR constant through most of pre-training and decays it only briefly at the end (Hu et al., [2024](#bib.bib17); Liu et al., [2024a](#bib.bib31); Wen et al., [2025](#bib.bib55)).

These previous studies, regardless of the details of the design choices, decayed the LRs to optimize the performance of pre-trained models.
However, the more critical factor for real applications is the performance after post-training, such as supervised fine-tuning (SFT).
Drawing on the findings of Sun & Dredze ([2025](#bib.bib47)) and Springer et al. ([2025](#bib.bib46)), which show that a strong pre-training model does not necessarily imply superior performance after SFT, it is questionable to schedule LRs to the decayed value based on pre-training performance.

In this study, we provide a comprehensive empirical investigation of LR schedulers during pre-training in terms of performance after SFT.
In particular, we examine an underestimated scheduling, Warmup-Stable-Only (WSO), which removes the decay phase from WSD and maintains constant LR to the end.
We show that WSO consistently achieves superior performance after SFT compared to decay-based schedulers, through experiments on 1B and 8B models (Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning")).
Furthermore, we demonstrate that WSO is also effective under modern training paradigms, including mid-training (OLMo et al., [2024](#bib.bib39); Meta, [2024c](#bib.bib37)) and over-training (Sardana et al., [2024](#bib.bib45); Gadre et al., [2025](#bib.bib12)).

To understand why WSO yields superior SFT performance, we draw on insights from the transfer learning literature (Ju et al., [2022](#bib.bib22); Liu et al., [2023](#bib.bib32)), which suggest that models in flatter regions of the loss landscape tend to exhibit better adaptability.
Through an analysis of sharpness values, we show that models trained with WSO reside in flatter regions than those trained with other decay-based LR schedulers, and are therefore more adaptable to post-training tasks.

Our contributions are as follows:
(1) We provide the systematic demonstration that WSO consistently outperforms decay-based schedulers on downstream tasks after SFT, with comprehensive evidence across 1B and 8B models and diverse evaluation benchmarks.
(2) We show that WSO similarly benefits mid-training and over-training scenarios, achieving superior SFT performance compared to conventional decay-based schedulers.
(3) We reveal through loss landscape analysis that WSO preserves flatter minima than decay-based schedulers, explaining why models trained with WSO achieve better performance after SFT.

![Refer to caption](/html/2603.16127/assets/x1.png)


Figure 1: 
Learning rate schedulers used in pre-training and their impact on performance after supervised fine-tuning (SFT).
Warmup-Stable-Only (WSO), which removes the decay phase, achieves the highest performance after SFT.

## 2 Preliminaries

Recent LLMs are typically built with a staged training scheme.
The most common and fundamental training pipeline consists of two stages, namely pre-training and post-training.
In this section, we describe these training stages and review the LR schedulers commonly employed during pre-training.

##### Pre-training.

Pre-training forms the foundation of LLM development, where models learn general language understanding from massive text corpora by minimizing the next-token prediction loss.
Recently, pre-training has sometimes consisted of multiple stages: standard pre-training and mid-training (OLMo et al., [2024](#bib.bib39)).
We describe mid-training in detail later (Section [2.2](#S2.SS2 "2.2 Further considerations ‣ 2 Preliminaries ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning")), and conduct experiments with both the standard pre-training and the multi-stage setup.

##### Post-training.

Post-training adapts pre-trained models to target tasks, enabling them to follow human instructions and avoid generating harmful outputs.
Post-training includes techniques such as supervised fine-tuning (SFT), preference tuning (e.g., DPO (Rafailov et al., [2023](#bib.bib42))), and RL-based alignment (Ouyang et al., [2022](#bib.bib40)).
While post-training could be a multi-stage process with many design choices still under active exploration, SFT is relatively standardized and serves a core stage.
In this paper, we focus on SFT as the canonical post-training stage and evaluate the performance after SFT111The computational cost of pre-training is typically much larger than that of other stages, so identifying a better pre-training configuration has a substantial impact on the efficiency of LLM construction.
In this study, we focus on evaluating LR schedulers during large-scale pre-training and characterize the potential of non-decay schedulers based on the performance after SFT.
An exploration of complex combinations of LR scheduling spanning multiple post-training stages is left to future work.
.

### 2.1 Task Definition

Practically, LLM developers evaluate models at multiple stages, selecting the best-performing one as the starting point for the subsequent stage.
We define 𝚃𝚊𝚜𝚔s​(M)\mathtt{Task}\_{\rm s}(M) as a function that, for a given LLM MM, returns the performance on a set of pre-defined tasks used to assess the target stage ss, where s∈{pre,post}s\in\{\rm pre,post\} denotes the training stage, with pre\rm pre indicating pre-training and post\rm post indicating post-training, respectively.
We write M2​[M1]M\_{2}[M\_{1}] to denote the model M2M\_{2} trained with some configuration and initialization with M1M\_{1}, where MrandM\_{\rm rand} indicates a model whose weights are randomly initialized.
Moreover, we introduce ℳpre\mathcal{M}\_{\rm pre} and ℳpost\mathcal{M}\_{\rm post} to represent the sets of models obtained through pre-training and post-training, respectively, with various hyperparameter configurations.
A typical training pipeline for building LLMs can therefore be expressed as follows:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | M^pre\displaystyle\widehat{M}\_{\rm pre} | =arg​maxMpre∈ℳpre⁡{𝚃𝚊𝚜𝚔pre​(Mpre​[Mrand])},\displaystyle=\operatorname\*{arg\,max}\_{M\_{\rm{pre}}\in\mathcal{M}\_{\rm pre}}\left\{\mathtt{Task}\_{\rm pre}(M\_{\rm{pre}}[M\_{\rm rand}])\right\}, |  | (1) |
|  | M^post\displaystyle\widehat{M}\_{\rm post} | =arg​maxMpost∈ℳpost⁡{𝚃𝚊𝚜𝚔post​(Mpost​[M^pre​[Mrand]])}.\displaystyle=\operatorname\*{arg\,max}\_{M\_{\rm{post}}\in\mathcal{M}\_{\rm{post}}}\left\{\mathtt{Task}\_{\rm post}(M\_{\rm{post}}[\widehat{M}\_{\rm pre}[{M}\_{\rm rand}]])\right\}. |  |

This formulation may lead to a suboptimal solution in terms of the performance of the final model, namely, M^post\widehat{M}\_{\rm post},
since selecting the best-performing models at intermediate stages does not guarantee achieving the best performance in the end.
Therefore, conceptually, we would like to consider the following search problem to obtain a better final model for this training pipeline:

|  |  |  |  |
| --- | --- | --- | --- |
|  | M^post=arg​max(Mpre,Mpost)∈(ℳpre,ℳpost)⁡{𝚃𝚊𝚜𝚔post​(Mpost​[Mpre​[Mrand]])}.\displaystyle\widehat{M}\_{\rm post}=\operatorname\*{arg\,max}\_{(M\_{\rm{pre}},M\_{\rm{post}})\in(\mathcal{M}\_{\rm{pre}},\mathcal{M}\_{\rm{post}})}\ \left\{\mathtt{Task}\_{\rm post}(M\_{\rm{post}}[M\_{\rm{pre}}[M\_{\rm rand}]])\right\}. |  | (2) |

The primary objective of this paper is to empirically examine the search problem by evaluating several LR schedulers during the large-scale training stages that precede post-training.

### 2.2 Further considerations

##### Mid-training.

Mid-training has emerged as a critical intermediate stage in modern language model development, occupying a computational middle ground between large-scale pre-training and task-specific post-training (Meta, [2024c](#bib.bib37); OLMo et al., [2024](#bib.bib39)).
This stage serves multiple strategic objectives, including domain expansion and long-context extension.
For example, OLMo 2 (OLMo et al., [2024](#bib.bib39)) demonstrates performance gains through mid-training on curated high-quality data, establishing this stage as an essential component of the modern training pipeline.
After introducing mid-training, we can rewrite equation [2](#S2.E2 "In 2.1 Task Definition ‣ 2 Preliminaries ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | M^post=arg​max(Mpre,Mmid,Mpost)∈(ℳpre,ℳmid,ℳpost)⁡{𝚃𝚊𝚜𝚔post​(Mpost​[Mmid​[Mpre​[Mrand]]])}.\displaystyle\widehat{M}\_{\rm post}=\operatorname\*{arg\,max}\_{(M\_{\rm{pre}},M\_{\rm{mid}},M\_{\rm{post}})\in(\mathcal{M}\_{\rm{pre}},\mathcal{M}\_{\rm{mid}},\mathcal{M}\_{\rm{post}})}\ \left\{\mathtt{Task}\_{\rm post}(M\_{\rm{post}}[{M}\_{\rm{mid}}[M\_{\rm{pre}}[M\_{\rm rand}]]])\right\}. |  | (3) |

##### Over-training.

Modern LLMs are often trained on trillions of tokens, far beyond the Chinchilla compute-optimal regime of roughly 20 tokens per parameter (Hoffmann et al., [2022](#bib.bib16)).
This practice trades substantially more training compute for improved inference efficiency at deployment.
Recent production systems use hundreds to thousands of tokens per parameter (Sardana et al., [2024](#bib.bib45)).
While full-scale experiments are costly, Section [5](#S5 "5 Experiment 3: Three-stage Setting in the Over-training ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") presents results under such a configuration, showing the generality of our main findings.

### 2.3 Current LR Scheduling Practices

In current LLM training practice, pre-training uses decay-based LR schedulers with Cosine, Linear, or WSD that reduce LR to 0–10% of maximum (Touvron et al., [2023a](#bib.bib51); Hu et al., [2024](#bib.bib17); Bergsma et al., [2025](#bib.bib3)).
Additionally, in mid-training, it is common practice to further decay the LR from the final value reached at the end of the preceding pre-training phase (Meta, [2024c](#bib.bib37); OLMo et al., [2024](#bib.bib39)).
These schedulers are chosen to minimize loss at each respective stage, effectively optimizing 𝚃𝚊𝚜𝚔pre​(Mpre)\mathtt{Task}\_{\rm pre}(M\_{\rm{pre}}) independently.
However, the primary objective should be to maximize 𝚃𝚊𝚜𝚔post​(Mpost)\mathtt{Task}\_{\rm post}(M\_{\rm post}), the performance after the complete pipeline. Thus, optimizing for 𝚃𝚊𝚜𝚔pre​(Mpre)\mathtt{Task}\_{\rm pre}(M\_{\rm{pre}}) may be suboptimal.
For instance, recent findings from Springer et al. ([2025](#bib.bib46)) and Sun & Dredze ([2025](#bib.bib47)) reveal that the better performance after pre-training does not guarantee performance after SFT.
These raise a fundamental question: Is LR decay, which is chosen based on pre-training performance, still the best choice when the model will undergo supervised fine-tuning?
Our work investigates this question by systematically varying LR schedulers in ℳpre\mathcal{M}\_{\rm pre} and ℳmid\mathcal{M}\_{\rm mid} to understand their impact on the final objective, i.e., 𝚃𝚊𝚜𝚔post​(Mpost)\mathtt{Task}\_{\rm post}(M\_{\rm post}).

### 2.4 Formalization of Learning Rate Schedulers

We denote the LR at training step tt as ηScheduler​(t,αpre)\eta^{\text{{Scheduler}}}({t},\alpha\_{\text{pre}}), where Scheduler specifies the LR scheduler and αpre\alpha\_{\text{pre}} controls the minimum LR factor in pre-training.
For example, the WSD scheduler is defined as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ηWSD​(t,αpre)={ηmax⋅tTwarmupt≤TwarmupηmaxTwarmup<t≤Tstableηmax⋅((1−αpre)⋅Tpre−tTpre−Tstable+αpre)Tstable<t≤Tpre\eta^{\text{{WSD}}}({t},\alpha\_{\text{pre}})=\begin{cases}\eta\_{\max}\cdot\frac{t}{T\_{\text{warmup}}}&t\leq T\_{\text{warmup}}\\ \eta\_{\max}&T\_{\text{warmup}}<t\leq T\_{\text{stable}}\\ \eta\_{\max}\cdot\left((1-\alpha\_{\text{pre}})\cdot\frac{T\_{\text{pre}}-t}{T\_{\text{pre}}-T\_{\text{stable}}}+\alpha\_{\text{pre}}\right)&T\_{\text{stable}}<t\leq T\_{\text{pre}}\end{cases} |  | (4) |

where ηmax\eta\_{\max} is the maximum LR, TpreT\_{\text{pre}} denotes the total number of pre-training steps, TwarmupT\_{\text{warmup}} is the number of warmup steps, and TstableT\_{\text{stable}} is the step at which the decay phase begins.

To investigate the effectiveness of the LR scheduler without decay, we consider a simple variant of WSD, which we call Warmup-Stable-Only (WSO).
In this variant, the decay phase is omitted, which corresponds to setting αpre=1.0\alpha\_{\text{pre}}=1.0.

|  |  |  |  |
| --- | --- | --- | --- |
|  | ηWSO​(t,αpre)={ηmax⋅tTwarmupt≤TwarmupηmaxTwarmup<t≤Tpre\eta^{\text{{WSO}}}({t},\alpha\_{\text{pre}})=\begin{cases}\eta\_{\max}\cdot\frac{t}{T\_{\text{warmup}}}&t\leq T\_{\text{warmup}}\\ \eta\_{\max}&T\_{\text{warmup}}<t\leq T\_{\text{pre}}\\ \end{cases} |  | (5) |

In our experiments, we investigate four LR schedulers: Scheduler∈{WSO,WSD,Cosine,Linear}\texttt{Scheduler}\in\{\text{WSO},\text{WSD},\text{Cosine},\text{Linear}\}.
The detailed formulations for Cosine ηCosine​(t,αpre)\eta^{\text{{Cosine}}}({t},\alpha\_{\text{pre}}) and Linear ηLinear​(t,αpre)\eta^{\text{{Linear}}}({t},\alpha\_{\text{pre}}) are provided in Appendix [B](#A2 "Appendix B Learning Rate Scheduler Formulations ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning").

![Refer to caption](/html/2603.16127/assets/x2.png)


Figure 2: Mid-training LR schedulers with different αpre\alpha\_{\text{pre}} and αmid\alpha\_{\text{mid}} values.

##### LR Scheduling in Mid-training.

We parameterize mid-training schedulers with αmid\alpha\_{\text{mid}} (Figure [2](#S2.F2 "Figure 2 ‣ 2.4 Formalization of Learning Rate Schedulers ‣ 2 Preliminaries ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning")), where αmid=0.0\alpha\_{\text{mid}}=0.0 applies Linear decay to zero while αmid=1.0\alpha\_{\text{mid}}=1.0 maintains the LR constant throughout mid-training.
When combined with αpre=1.0\alpha\_{\text{pre}}=1.0, the configuration of αpre=1.0\alpha\_{\text{pre}}=1.0 and αmid=1.0\alpha\_{\text{mid}}=1.0 extends WSO across both pre-training and mid-training stages.
The detailed formulation for mid-training LR schedulers is provided in Appendix [B](#A2 "Appendix B Learning Rate Scheduler Formulations ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning").

## 3 Experiment 1: Two-stage (Pre- and Post-training) Setting

We investigate whether decaying LRs during pre-training truly benefit downstream SFT performance.

### 3.1 Experimental Setup

##### Model Architectures.

We conduct experiments on two model scales following the Llama 3 architecture family: 1B and 8B parameter models (same architecture as Llama-3.2-1B (Meta, [2024b](#bib.bib36)) and Llama-3.1-8B (Meta, [2024a](#bib.bib35)), respectively).
Full details are provided in Appendix [A](#A1 "Appendix A Model Architecture ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning").

##### Pre-training Configuration.

Models are pre-trained on FineWeb-Edu (Penedo et al., [2024](#bib.bib41)) with a maximum LR ηmax=3×10−4\eta\_{\max}=3\times 10^{-4}.
We investigate three LR schedulers as formalized in Section [2.4](#S2.SS4 "2.4 Formalization of Learning Rate Schedulers ‣ 2 Preliminaries ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning"), experimenting with WSO (Equation [5](#S2.E5 "In 2.4 Formalization of Learning Rate Schedulers ‣ 2 Preliminaries ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning")), WSD (Equation [4](#S2.E4 "In 2.4 Formalization of Learning Rate Schedulers ‣ 2 Preliminaries ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning")), Cosine, and Linear schedulers (detailed in Appendix [B](#A2 "Appendix B Learning Rate Scheduler Formulations ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning")).
For each scheduler, we vary the minimum LR factor αpre∈{0.0,0.1,1.0}\alpha\_{\text{pre}}\in\{0.0,0.1,1.0\}, following our notation ηScheduler​(t,αpre)\eta^{\text{{Scheduler}}}({t},\alpha\_{\text{pre}}).
Setting αpre=0.0\alpha\_{\text{pre}}=0.0 corresponds to decay to zero.
Recent work by Bergsma et al. ([2025](#bib.bib3)) shows that this achieves better pre-training performance.
Setting αpre=0.1\alpha\_{\text{pre}}=0.1 corresponds to decay to 10% of maximum, a choice commonly used in practice by Chinchilla (Hoffmann et al., [2022](#bib.bib16)), Llama 3 (Meta, [2024c](#bib.bib37)) and OLMo 2 (OLMo et al., [2024](#bib.bib39)).
Finally, setting αpre=1.0\alpha\_{\text{pre}}=1.0 corresponds to WSO.
Further hyperparameter details are provided in Appendix [C](#A3 "Appendix C Pre-training Hyperparameters ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning").

##### SFT Configuration.

We perform SFT using the Tulu-3 SFT mixture222<https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture/tree/main>
.
We conduct a comprehensive LR sweep ranging from 5×10−75\times 10^{-7} to 1×10−31\times 10^{-3} to identify the best hyperparameters for each pre-trained model333Full details about SFT are provided in Appendix [D](#A4 "Appendix D SFT Configuration ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning").
.

##### Evaluation.

We evaluate models at two stages: after pre-training and after SFT.
For pre-trained models, we assess zero-shot performance on standard benchmarks, including question answering (ARC-Easy, ARC-Challenge (Clark et al., [2018](#bib.bib7)), OpenBookQA (Mihaylov et al., [2018](#bib.bib38)), BoolQ (Clark et al., [2019](#bib.bib6))) and commonsense reasoning (HellaSwag (Zellers et al., [2019](#bib.bib56)), PIQA (Bisk et al., [2020](#bib.bib4)), WinoGrande (Sakaguchi et al., [2021](#bib.bib43))), along with validation loss.

For fine-tuned models, we follow the setup of OLMo (Groeneveld et al., [2024](#bib.bib13)) and evaluate along three key dimensions: instruction-following capability (AlpacaEval (Li et al., [2023](#bib.bib29))), multi-task language understanding (MMLU (Hendrycks et al., [2021](#bib.bib15))), and truthfulness (TruthfulQA (Lin et al., [2022](#bib.bib30))).

To highlight how LR decay affects both pre-training and SFT differently, we present results as relative performance metrics normalized against the best decay-based scheduler for each stage.
For pre-training, we report both validation loss and the average accuracy across all zero-shot benchmarks (PT Task Avg).
For fine-tuning, we report the average across AlpacaEval, TruthfulQA, and MMLU (SFT Task Avg)444Detailed evaluation settings are provided in Appendix [E](#A5 "Appendix E Evaluation Details ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning").
.

### 3.2 Results

Table 1: Relative performance across pre-training (PT) and supervised fine-tuning (SFT). For each model size and each metric, values are differences (Δ\Delta) from the best-performing decay-based scheduler for that metric. Note that WSO could perform poorly after PT but best after SFT. Bold indicates the best performance.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Model | Scheduler | αpre\alpha\_{\text{pre}} | PT Valid Loss ↓\downarrow Δ\Delta | PT Task Avg Δ\Delta | SFT Task Avg Δ\Delta |
| 1B | Warmup-Stable-Only (WSO) | 1.0 | +0.071 | -1.7 | +0.3 |
| WSD | 0.1 | +0.004 | -1.5 | +0.0 |
| 0.0 | +0.000 | -1.2 | -1.0 |
| Linear | 0.1 | +0.021 | -2.0 | -0.7 |
| 0.0 | +0.016 | +0.0 | -0.9 |
| Cosine | 0.1 | +0.019 | -0.1 | -0.7 |
| 0.0 | +0.016 | -2.5 | -0.7 |
| 8B | Warmup-Stable-Only (WSO) | 1.0 | +0.127 | -0.8 | +1.1 |
| WSD | 0.1 | +0.019 | -0.2 | -0.8 |
| 0.0 | +0.014 | +0.0 | -0.3 |
| Linear | 0.1 | +0.013 | -1.9 | -0.6 |
| 0.0 | +0.000 | -1.8 | +0.0 |
| Cosine | 0.1 | +0.009 | -2.2 | -0.3 |
| 0.0 | +0.008 | -2.3 | -0.1 |

Table [1](#S3.T1 "Table 1 ‣ 3.2 Results ‣ 3 Experiment 1: Two-stage (Pre- and Post-training) Setting ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") shows an inversion in model performance across training stages555Detailed per-task evaluation results for all models are provided in Appendix [F](#A6 "Appendix F Full Evaluation Results ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning")..
For pre-training performance, decay-based schedulers achieve the best performance with αpre=0\alpha\_{\text{pre}}=0.
Specifically, Linear and WSD with αpre=0\alpha\_{\text{pre}}=0 achieve the best PT Task Avg scores for the 1B and 8B models, respectively.
This result is consistent with existing findings (Bergsma et al., [2025](#bib.bib3)).
In contrast, after SFT, WSO achieves the best performance for both model sizes, even though it underperforms decay-based schedulers in pre-training metrics.
These results demonstrate that while decay-based schedulers may yield superior performance in terms of pre-training metrics, WSO is more effective in the overall training pipeline, including SFT.

Table 2: Relative performance across mid-training (MT) and SFT stages.
Values are differences from the best decay-based schedule.
WSO throughout both stages yields the best SFT performance.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Model | (Pre-training) Scheduler | αpre\alpha\_{\text{pre}} | αmid\alpha\_{\text{mid}} | MT Valid Loss ↓\downarrow Δ\Delta | MT Task Avg Δ\Delta | SFT Task Avg Δ\Delta |
| 1B | Warmup-Stable-Only (WSO) | 1.0 | 1.0 | +0.062 | -0.1 | +0.8 |
| WSD | 1.0 | 0.0 | +0.000 | +0.0 | +0.0 |
| 0.1 | 1.0 | +0.038 | -1.5 | -0.5 |
| 0.1 | 0.0 | +0.047 | -1.7 | -1.3 |
| Linear | 0.1 | 1.0 | +0.053 | -2.1 | -2.5 |
| 0.1 | 0.0 | +0.058 | -3.3 | -3.8 |
| Cosine | 0.1 | 1.0 | +0.053 | -2.4 | -2.9 |
| 0.1 | 0.0 | +0.059 | -3.1 | -3.7 |
| 8B | Warmup-Stable-Only (WSO) | 1.0 | 1.0 | +0.102 | -2.1 | +1.1 |
| WSD | 1.0 | 0.0 | +0.000 | +0.0 | -1.4 |
| 0.1 | 1.0 | +0.057 | -5.0 | +0.0 |
| 0.1 | 0.0 | +0.081 | -5.6 | -1.1 |
| Linear | 0.1 | 1.0 | +0.067 | -8.3 | -2.2 |
| 0.1 | 0.0 | +0.082 | -9.0 | -3.7 |
| Cosine | 0.1 | 1.0 | +0.068 | -8.0 | -3.5 |
| 0.1 | 0.0 | +0.084 | -10.1 | -4.1 |

## 4 Experiment 2: Three-stage (Pre-, Mid-, and Post-training) Setting

Recent LLM developments (OLMo et al., [2024](#bib.bib39); Meta, [2024c](#bib.bib37)) add a mid-training stage between pre-training and post-training, which makes LR scheduling across stages more complex due to the various combinations of pre-training and mid-training LR schedulers.
We investigate whether using WSO in both pre-training and mid-training stages yields better performance after SFT than decay-based schedulers.

### 4.1 Experimental Setup

To investigate the effect of LR scheduling during mid-training, we conduct experiments following a three-stage training pipeline: pre-training, mid-training, and post-training.
We systematically vary the LR schedulers in both pre-training and mid-training stages to understand their individual and combined effects on downstream performance.
To ensure comparability with recent mid-training work, our setup largely follows OLMo 2 (OLMo et al., [2024](#bib.bib39)), a representative study of mid-training.

##### Pre-training Stage.

We pre-train 1B and 8B models using the same architecture and configuration as described in Section [3](#S3 "3 Experiment 1: Two-stage (Pre- and Post-training) Setting ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning").
We adopt pre-training dataset olmo-mix-1124 (OLMo et al., [2024](#bib.bib39)) used in OLMo 2.
Following standard practice in modern LLM development (Meta, [2024c](#bib.bib37); OLMo et al., [2024](#bib.bib39)), we employ four LR schedulers with different minimum LR factors, including WSD, Cosine, and Linear schedulers with αpre=0.1\alpha\_{\text{pre}}=0.1, and additionally WSO.

##### Mid-training Stage and Learning Rate Schedules.

Following OLMo 2  (OLMo et al., [2024](#bib.bib39)), we conduct mid-training on the dolmino-mix-1124 dataset.
We investigate the two mid-training strategies shown in Figure [2](#S2.F2 "Figure 2 ‣ 2.4 Formalization of Learning Rate Schedulers ‣ 2 Preliminaries ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning"), with αmid=0.0\alpha\_{\text{mid}}=0.0 applying further Linear decay following common practice (Meta, [2024c](#bib.bib37); OLMo et al., [2024](#bib.bib39)), and αmid=1.0\alpha\_{\text{mid}}=1.0 maintaining a constant LR throughout mid-training666Further training configurations of mid-training are provided in Appendix [G](#A7 "Appendix G Mid-training Configuration Details ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning")..

##### SFT and Evaluation.

For SFT, we follow the configuration described in Section [3](#S3 "3 Experiment 1: Two-stage (Pre- and Post-training) Setting ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning").
For mid-trained models (before SFT), we evaluate on standard benchmarks to assess the impact of mid-training LR schedulers, following the evaluation suite used in OLMo 2 (OLMo et al., [2024](#bib.bib39)).
We select benchmarks that comprehensively assess model capabilities, including reasoning tasks (ARC-Challenge (Clark et al., [2018](#bib.bib7)), HellaSwag (Zellers et al., [2019](#bib.bib56)), WinoGrande (Sakaguchi et al., [2021](#bib.bib43))), reading comprehension (DROP (Dua et al., [2019](#bib.bib10))), and mathematical reasoning (GSM8K (Cobbe et al., [2021](#bib.bib8))).
Following SFT, we assess models using an expanded evaluation suite including AlpacaEval (Li et al., [2023](#bib.bib29)) for instruction following, TruthfulQA (Lin et al., [2022](#bib.bib30)) for factual accuracy, GSM8K (Cobbe et al., [2021](#bib.bib8)) for mathematical reasoning, DROP (Dua et al., [2019](#bib.bib10)) for reading comprehension, AGI Eval (Zhong et al., [2024](#bib.bib57)) for general intelligence capabilities, BigBench-Hard (Suzgun et al., [2023](#bib.bib48)) for challenging reasoning tasks, and MMLU for multitask understanding777The detailed evaluation settings for these benchmarks are described in Appendix [E](#A5 "Appendix E Evaluation Details ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning")..
Similar to Section [3](#S3 "3 Experiment 1: Two-stage (Pre- and Post-training) Setting ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning"), we present results as relative improvements compared to the best decay-based scheduler.

### 4.2 Results

Table 3: Relative performance after over-training (2T tokens). Values are differences (Δ\Delta) from the best-performing decay-based scheduler for each metric. Similar to Section [3](#S3 "3 Experiment 1: Two-stage (Pre- and Post-training) Setting ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning"), WSO achieves the best SFT performance.

| Model | Scheduler | αpre\alpha\_{\text{pre}} | PT Valid Loss ↓\downarrow Δ\Delta | PT Task Avg Δ\Delta | SFT Task Avg Δ\Delta |
| --- | --- | --- | --- | --- | --- |
| 1B | Warmup-Stable-Only (WSO) | 1.0 | +0.048 | -1.5 | +0.7 |
| WSD | 0.1 | +0.004 | +0.0 | +0.0 |
| 0.0 | +0.000 | +0.0 | -0.3 |
| Linear | 0.1 | +0.021 | -0.9 | -0.5 |
| 0.0 | +0.017 | -0.4 | -0.6 |
| Cosine | 0.1 | +0.017 | +0.0 | -0.4 |
| 0.0 | +0.017 | -1.3 | -0.3 |




Table 4: Relative performance after over-training with mid-training (2T + 500B tokens). Values are differences (Δ\Delta) from the best-performing decay-based scheduler for each metric. Similar to Section [4](#S4 "4 Experiment 2: Three-stage (Pre-, Mid-, and Post-training) Setting ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning"), WSO yields the best SFT performance.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Model | (Pre-training) Scheduler | αpre\alpha\_{\text{pre}} | αmid\alpha\_{\text{mid}} | MT Valid Loss ↓\downarrow Δ\Delta | MT Task Avg Δ\Delta | SFT Task Avg Δ\Delta |
| 1B | Warmup-Stable-Only (WSO) | 1.0 | 1.0 | +0.055 | -0.3 | +1.4 |
| WSD | 1.0 | 0.0 | +0.000 | -1.6 | -0.5 |
| 0.1 | 1.0 | +0.033 | +0.0 | -1.0 |
| 0.1 | 0.0 | +0.038 | -1.7 | -1.2 |
| Linear | 0.1 | 1.0 | +0.068 | -2.2 | +0.0 |
| 0.1 | 0.0 | +0.051 | -2.8 | -0.6 |
| Cosine | 0.1 | 1.0 | +0.046 | -1.8 | -0.7 |
| 0.1 | 0.0 | +0.054 | -2.3 | -1.2 |

Table [2](#S3.T2 "Table 2 ‣ 3.2 Results ‣ 3 Experiment 1: Two-stage (Pre- and Post-training) Setting ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") shows an inversion similar to our pre-training findings888Detailed per-task evaluation results for all models are provided in Appendix [F](#A6 "Appendix F Full Evaluation Results ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning")..
For mid-training performance, the decay-based scheduler with αpre=1.0\alpha\_{\text{pre}}=1.0 and αmid=0.0\alpha\_{\text{mid}}=0.0 achieve the best performance.
However, SFT performance again shows the opposite trend.
WSO achieves the best downstream task performance after SFT, even though it underperforms the best decay-based schedulers in mid-training metrics.
Additionally, we find that introducing decay at any stage reduces SFT performance.
Notably, for models pre-trained with decay (αpre=0.1\alpha\_{\text{pre}}=0.1), avoiding decay during mid-training (αmid=1.0\alpha\_{\text{mid}}=1.0) improves both mid-training metrics and SFT performance compared to applying decay.

These results extend our findings to multi-stage training pipelines, where decay at any stage consistently harms SFT performance.
WSO, which maintains constant learning rates throughout both pre-training and mid-training, shows the best performance across the overall training pipeline, including mid-training and SFT.

## 5 Experiment 3: Three-stage Setting in the Over-training

To further probe generality, we evaluate a third regime with a substantially larger training budget.
This over-training setting serves as a test of whether the benefits of WSO persist when training on trillions of tokens.

### 5.1 Experimental Setup

##### Pre- and Mid-training.

We pre-train 1B models on 2T tokens, which is approximately 100×\times the Chinchilla-optimal amount of data for this model size, to evaluate whether WSO maintains its advantages at this data scale.
We use the same datasets as in Sections [3](#S3 "3 Experiment 1: Two-stage (Pre- and Post-training) Setting ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") and [4](#S4 "4 Experiment 2: Three-stage (Pre-, Mid-, and Post-training) Setting ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") for pre-training and mid-training, respectively.
We investigate the same set of LR schedulers as in Section [3](#S3 "3 Experiment 1: Two-stage (Pre- and Post-training) Setting ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning").
We additionally conduct mid-training experiments using 500B tokens, following the same experimental setup as in Section [4](#S4 "4 Experiment 2: Three-stage (Pre-, Mid-, and Post-training) Setting ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning").

##### Evaluation.

We evaluate all LR schedulers using the same methodology as in Sections [3](#S3 "3 Experiment 1: Two-stage (Pre- and Post-training) Setting ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") and [4](#S4 "4 Experiment 2: Three-stage (Pre-, Mid-, and Post-training) Setting ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning"), measuring performance both after pre-training (or mid-training) and after SFT.
Detailed configurations are provided in Appendices [C](#A3 "Appendix C Pre-training Hyperparameters ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") and [D](#A4 "Appendix D SFT Configuration ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning").

### 5.2 Results

Tables [3](#S4.T3 "Table 3 ‣ 4.2 Results ‣ 4 Experiment 2: Three-stage (Pre-, Mid-, and Post-training) Setting ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") and [4](#S4.T4 "Table 4 ‣ 4.2 Results ‣ 4 Experiment 2: Three-stage (Pre-, Mid-, and Post-training) Setting ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") confirm that the inversion observed in Sections [3](#S3 "3 Experiment 1: Two-stage (Pre- and Post-training) Setting ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") and [4](#S4 "4 Experiment 2: Three-stage (Pre-, Mid-, and Post-training) Setting ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") persists even in the over-training regime using 2T tokens.
Across all investigated schedulers, WSO (αpre=1.0\alpha\_{\text{pre}}=1.0) consistently yields worse intermediate metrics but superior SFT performance compared to decay-based schedulers.
Similar to our earlier findings, decay-based schedulers achieve better pre-training and mid-training metrics, yet WSO outperforms them after SFT.
This pattern holds both for single-stage over-training (Table [3](#S4.T3 "Table 3 ‣ 4.2 Results ‣ 4 Experiment 2: Three-stage (Pre-, Mid-, and Post-training) Setting ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning")) and when combined with mid-training (Table [4](#S4.T4 "Table 4 ‣ 4.2 Results ‣ 4 Experiment 2: Three-stage (Pre-, Mid-, and Post-training) Setting ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning")), demonstrating that the benefits of WSO are robust across different data scales and training configurations.

## 6 Understanding Adaptability Through Loss Landscape Geometry

![Refer to caption](/html/2603.16127/assets/x3.png)


Figure 3: Sharpness​(θt)\text{Sharpness}(\theta\_{t}) during pre-training of the 1B model. Vertical line at step TstableT\_{\text{stable}} indicating where WSD decays LR. Decay-based schedulers (αpre=0\alpha\_{\text{pre}}=0 or αpre=0.1\alpha\_{\text{pre}}=0.1) lead to sharper minima, while WSO (αpre=1.0\alpha\_{\text{pre}}=1.0) maintains flatter landscapes.

To understand why models trained with WSO achieve superior SFT performance, we analyze the loss landscape geometry throughout the pre-training phase.
As suggested in the transfer learning literature (Ju et al., [2022](#bib.bib22); Liu et al., [2023](#bib.bib32)), we focus on sharpness as a key geometric property that characterizes the curvature of the loss landscape around converged parameters.

The relation between lower sharpness and better SFT performance stems from how models respond to parameter updates during fine-tuning.
When the parameters of the model lie in a flatter region of the loss landscape, which corresponds to lower sharpness, the model demonstrates superior adaptability to downstream tasks (Foret et al., [2021](#bib.bib11); Li et al., [2025](#bib.bib28)).
The intuition is that the performance of the model remains stable during the parameter updates of SFT.
A model in a flat landscape experiences less fluctuation in its loss value when its parameters are updated, which translates to more stable performance.
This characteristic is believed to confer higher adaptability, as the model can incorporate new data without compromising its pre-trained capabilities (Andriushchenko et al., [2023](#bib.bib1)).

There are several ways to quantify sharpness, such as the largest eigenvalue of the Hessian (capturing the most curved direction) or the trace of the Hessian (capturing the average curvature) (Dinh et al., [2017](#bib.bib9); Kaur et al., [2023](#bib.bib25)).
Following established practice in optimization and generalization studies (Ju et al., [2022](#bib.bib22); Liu et al., [2023](#bib.bib32)), we adopt the trace as our sharpness measure, since it provides a scalar summary of curvature across all parameter dimensions.

###### Definition 6.1 (Sharpness).

Let ℒ​(θt;𝒟)\mathcal{L}(\theta\_{t};\mathcal{D}) denote the loss function evaluated on dataset 𝒟\mathcal{D} with model parameters θt∈ℝd\theta\_{t}\in\mathbb{R}^{d}. At training step tt, the sharpness of the loss landscape at parameters θt\theta\_{t} is defined as the trace of the Hessian matrix:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Sharpness​(θt)=Tr⁡(𝐇ℒ​(θt))=∑i=1d∂2ℒ​(θt;𝒟)∂θi2\text{Sharpness}(\theta\_{t})=\operatorname{Tr}(\mathbf{H}\_{\mathcal{L}}(\theta\_{t}))=\sum\_{i=1}^{d}\frac{\partial^{2}\mathcal{L}(\theta\_{t};\mathcal{D})}{\partial\theta\_{i}^{2}} |  | (6) |

where 𝐇ℒ​(θt)∈ℝd×d\mathbf{H}\_{\mathcal{L}}(\theta\_{t})\in\mathbb{R}^{d\times d} is the Hessian matrix of the loss with respect to the parameters at θt\theta\_{t}.

Since computing the full Hessian trace is computationally prohibitive for billion-parameter models, we employ Hutchinson’s unbiased estimator (Hutchinson, [1989](#bib.bib18); Liu et al., [2024b](#bib.bib33)).
This method requires only Hessian-vector products, which can be efficiently computed through automatic differentiation.
Details of our sampling procedure and computational details are provided in Appendix [H](#A8 "Appendix H Sharpness Computation Details ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning").

We measure sharpness throughout pre-training on validation sets from both the pre-training dataset and the SFT dataset.
Figure [3](#S6.F3 "Figure 3 ‣ 6 Understanding Adaptability Through Loss Landscape Geometry ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") shows the sharpness for the 1B model from Section [3](#S3 "3 Experiment 1: Two-stage (Pre- and Post-training) Setting ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning").
We illustrate a vertical line at step TstableT\_{\text{stable}} to indicate the point at which WSD decays LR.
The figure reveals distinct patterns across schedulers.
Specifically, Cosine and Linear schedulers exhibit steadily increasing sharpness as the LR decays, while WSD shows a rise during its decay phase.
In contrast, WSO maintains lower sharpness.
Across both datasets, models with decaying LRs converge to regions with about 2–3×\times higher sharpness compared to WSO models.
Flatter regions obtained by WSO allow more flexible parameter adaptation during SFT, enabling better downstream performance.

##### Correlation between sharpness and downstream adaptability.

![Refer to caption](/html/2603.16127/assets/x4.png)


Figure 4: Pre-training sharpness negatively correlates with downstream SFT performance.

To provide empirical evidence linking the loss landscape to downstream adaptability, we analyze the correlation between the sharpness of pre-trained models and their subsequent SFT performance.
Figure [4](#S6.F4 "Figure 4 ‣ Correlation between sharpness and downstream adaptability. ‣ 6 Understanding Adaptability Through Loss Landscape Geometry ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") presents the SFT performance plotted against the sharpness measured on the pre-training validation set for the 1B model across all investigated learning rate schedulers.
The analysis reveals a negative correlation (Pearson r=−0.709r=-0.709) between the sharpness of the minima and the model’s performance after SFT.
The WSO scheduler (αpre=1.0\alpha\_{\text{pre}}=1.0) resides in the low-sharpness, high-performance region, while decay-based schedulers converge to sharper minima with lower SFT scores.
While the sample size is limited, this pattern is consistent with our hypothesis that preserving flatter minima during pre-training enhances the model’s adaptability.

## 7 Related Work

##### Learning Rate Scheduling in LLM Training.

LR decay has been considered effective for LLM pre-training, with Cosine decay remaining the de facto standard (Kaplan et al., [2020](#bib.bib24); Hoffmann et al., [2022](#bib.bib16); Touvron et al., [2023b](#bib.bib52)).
Recent large-scale studies advocate for even more aggressive decay, showing that Linear decay to zero achieves lower pre-training loss in compute-optimal settings (Bergsma et al., [2025](#bib.bib3)).
Warmup-Stable-Decay (WSD) delays decay until the final phase of training (Hu et al., [2024](#bib.bib17)), while theoretical analysis suggests that decay may confine models to narrow loss valleys (Wen et al., [2025](#bib.bib55)).
Some methods attempt to avoid the decay phase through checkpoint averaging (Sanyal et al., [2024](#bib.bib44)) or model merging (Tian et al., [2025](#bib.bib49)).
Jin et al. ([2023](#bib.bib21)) investigated learning rate tuning strategies within individual training phases, but did not examine how pre-training LR choices propagate through to post-training performance.
While these studies have advanced our understanding of LR scheduling within a single training phase, they share a common limitation: evaluation is restricted to the phase in which the schedule is applied, without considering the downstream consequences for subsequent training stages such as SFT.

##### Relationship to Continual Pre-training and Fine-tuning Studies.

Recent work on continual pre-training (CPT) has explored LR scheduling for domain adaptation.
Wang et al. ([2025](#bib.bib53)) showed that models with higher loss potential achieve lower CPT validation loss and advocated for releasing high loss potential versions to facilitate downstream tasks.
Wang et al. ([2024](#bib.bib54)) proposed a path-switching paradigm for LR scheduling in model version updates, though their experimental setup still applies LR decay before performing SFT.
Tissue et al. ([2024](#bib.bib50)) introduced a scaling law describing loss dynamics in relation to learning rate annealing; however, they explicitly note that post-training scenarios involving distribution shift are out of scope.
These CPT studies focus on settings where the objective function remains language modeling, leaving the impact on SFT unexplored.
Meanwhile, a growing body of work has examined the gap between pre-training quality and downstream performance more broadly.
Sun & Dredze ([2025](#bib.bib47)) showed that stronger pre-training performance does not necessarily translate to superior fine-tuning outcomes, and Springer et al. ([2025](#bib.bib46)) demonstrated that over-trained models become harder to fine-tune.
These findings collectively motivate our investigation, as LR schedulers chosen to optimize potentially unreliable pre-training metrics may not be optimal for the overall
training pipeline.
Our work identifies LR decay as a specific factor that improves pre-training metrics at the cost of downstream adaptability.

##### Adaptability and Loss Landscape Geometry.

Early work showed that parameters in flatter loss regions generalize better than those in sharp minima (Keskar et al., [2017](#bib.bib26)), motivating sharpness-aware minimization (Foret et al., [2021](#bib.bib11)) and stochastic weight averaging (Izmailov et al., [2018](#bib.bib20)).
Recent theoretical advances explain WSD through a river valley loss landscape perspective (Wen et al., [2025](#bib.bib55)), where the stable phase explores along the valley floor while the decay phase converges toward the center.
Concurrent work confirmed that sharpness increase during decay is universal across architectures (Belloni et al., [2025](#bib.bib2)).
Flat-minima optimizers work well under distribution shift (Kaddour et al., [2022](#bib.bib23)), a property that extends to the pre-training/fine-tuning paradigm, where the fine-tuning data distribution differs substantially from pre-training (Ju et al., [2022](#bib.bib22); Liu et al., [2023](#bib.bib32)).
While prior work focused on understanding sharpness dynamics during pre-training (Belloni et al., [2025](#bib.bib2); Wen et al., [2025](#bib.bib55)), we demonstrate how these dynamics concretely impact SFT performance, showing that WSO preserves flatness and enhances downstream adaptability.

## 8 Conclusion

In this study, we investigated the effectiveness of LR schedulers, which have been widely reported as effective for pre-training, in practical scenarios with a focus on post-training performance.
In particular, we examine Warmup-Stable-Only (WSO), which removes the decay phase from WSD. Experimental results show that WSO consistently outperforms decay-based schedulers in downstream tasks after SFT across
standard pre-training, mid-training, and over-training regimes.
Loss landscape analysis further reveals that WSO preserves flatter minima, explaining its superior adaptability.
WSO is simple to apply and yields improved post-training performance, making it a promising alternative for constructing more portable models. We also recommend releasing LLMs trained with WSO so that practitioners can benefit from their adaptability.

## Ethics Statement

This work investigates learning rate scheduling for LLM training to improve downstream adaptability.
While our methods may provide new findings on LR scheduling on pre-training, we acknowledge the broader implications of advancing LLM capabilities.
We encourage responsible deployment with appropriate safety measures during post-training.
We exclusively used publicly available datasets for
pre-training, supervised fine-tuning, and evaluation. Moreover,
we developed the language models entirely from
scratch, avoiding the use of any publicly available
models to ensure reproducibility.

## Reproducibility Statement

To ensure reproducibility of our results, we provide comprehensive experimental details throughout the paper and appendices.
Model architectures for both 1B and 8B parameter models are specified in Appendix [A](#A1 "Appendix A Model Architecture ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning"), including all layer configurations and attention mechanisms.
All pre-training hyperparameters, including optimizer settings, batch sizes, and training steps, are detailed in Appendix [C](#A3 "Appendix C Pre-training Hyperparameters ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning").
The supervised fine-tuning configuration, including the learning rate sweep range and evaluation protocols, is described in Appendix [D](#A4 "Appendix D SFT Configuration ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning").
Our sharpness computation methodology using Hutchinson’s estimator is fully specified in Appendix [H](#A8 "Appendix H Sharpness Computation Details ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning").
We use publicly available datasets (FineWeb-Edu, olmo-mix-1124, dolmino-mix-1124, and Tulu-3 SFT mixture) and standard evaluation benchmarks, with detailed evaluation settings provided in Appendix [E](#A5 "Appendix E Evaluation Details ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning").
Full numerical results for all experiments are reported in Appendix [F](#A6 "Appendix F Full Evaluation Results ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") to facilitate comparison and validation.

#### Acknowledgments

This work was partly supported
by JST Moonshot R&D Grant Number JPMJMS2011-35 (fundamental research).

## References

* Andriushchenko et al. (2023)

  Maksym Andriushchenko, Francesco Croce, Maximilian Müller, Matthias Hein, and Nicolas Flammarion.
  A modern look at the relationship between sharpness and generalization.
  In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pp. 840–902. PMLR, 23–29 Jul 2023.
  URL <https://proceedings.mlr.press/v202/andriushchenko23a.html>.
* Belloni et al. (2025)

  Annalisa Belloni, Lorenzo Noci, and Antonio Orvieto.
  Universal dynamics of warmup stable decay: understanding WSD beyond transformers.
  In *ICML 2025 Workshop on Methods and Opportunities at Small Scale*, 2025.
  URL <https://openreview.net/forum?id=2HNQqMBvC2>.
* Bergsma et al. (2025)

  Shane Bergsma, Nolan Simran Dey, Gurpreet Gosal, Gavia Gray, Daria Soboleva, and Joel Hestness.
  Straight to zero: Why linearly decaying the learning rate to zero works best for LLMs.
  In *The Thirteenth International Conference on Learning Representations*, 2025.
  URL <https://openreview.net/forum?id=hrOlBgHsMI>.
* Bisk et al. (2020)

  Yonatan Bisk, Rowan Zellers, Ronan bras, Jianfeng Gao, and Choi Yejin.
  Piqa: Reasoning about physical commonsense in natural language.
  *Proceedings of the AAAI Conference on Artificial Intelligence*, 34:7432–7439, 04 2020.
  doi: 10.1609/aaai.v34i05.6239.
* Brown et al. (2020)

  Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei.
  Language models are few-shot learners.
  In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), *Advances in Neural Information Processing Systems*, volume 33, pp. 1877–1901. Curran Associates, Inc., 2020.
  URL <https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf>.
* Clark et al. (2019)

  Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova.
  BoolQ: Exploring the surprising difficulty of natural yes/no questions.
  In Jill Burstein, Christy Doran, and Thamar Solorio (eds.), *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pp. 2924–2936, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.
  doi: 10.18653/v1/N19-1300.
  URL <https://aclanthology.org/N19-1300/>.
* Clark et al. (2018)

  Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord.
  Think you have solved question answering? try arc, the ai2 reasoning challenge, 2018.
* Cobbe et al. (2021)

  Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al.
  Training verifiers to solve math word problems.
  *arXiv preprint arXiv:2110.14168*, 2021.
* Dinh et al. (2017)

  Laurent Dinh, Razvan Pascanu, Samy Bengio, and Yoshua Bengio.
  Sharp minima can generalize for deep nets.
  In *International Conference on Machine Learning*, pp. 1019–1028. PMLR, 2017.
* Dua et al. (2019)

  Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner.
  DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs.
  In Jill Burstein, Christy Doran, and Thamar Solorio (eds.), *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pp. 2368–2378, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.
  doi: 10.18653/v1/N19-1246.
  URL <https://aclanthology.org/N19-1246/>.
* Foret et al. (2021)

  Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur.
  Sharpness-aware minimization for efficiently improving generalization.
  In *International Conference on Learning Representations*, 2021.
  URL <https://openreview.net/forum?id=6Tm1mposlrM>.
* Gadre et al. (2025)

  Samir Yitzhak Gadre, Georgios Smyrnis, Vaishaal Shankar, Suchin Gururangan, Mitchell Wortsman, Rulin Shao, Jean Mercat, Alex Fang, Jeffrey Li, Sedrick Keh, Rui Xin, Marianna Nezhurina, Igor Vasiljevic, Luca Soldaini, Jenia Jitsev, Alex Dimakis, Gabriel Ilharco, Pang Wei Koh, Shuran Song, Thomas Kollar, Yair Carmon, Achal Dave, Reinhard Heckel, Niklas Muennighoff, and Ludwig Schmidt.
  Language models scale reliably with over-training and on downstream tasks.
  In *The Thirteenth International Conference on Learning Representations*, 2025.
  URL <https://openreview.net/forum?id=iZeQBqJamf>.
* Groeneveld et al. (2024)

  Dirk Groeneveld, Iz Beltagy, Evan Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David Atkinson, Russell Authur, Khyathi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar, Yuling Gu, Jack Hessel, Tushar Khot, William Merrill, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew Peters, Valentina Pyatkin, Abhilasha Ravichander, Dustin Schwenk, Saurabh Shah, William Smith, Emma Strubell, Nishant Subramani, Mitchell Wortsman, Pradeep Dasigi, Nathan Lambert, Kyle Richardson, Luke Zettlemoyer, Jesse Dodge, Kyle Lo, Luca Soldaini, Noah Smith, and Hannaneh Hajishirzi.
  OLMo: Accelerating the science of language models.
  In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 15789–15809, Bangkok, Thailand, August 2024. Association for Computational Linguistics.
  doi: 10.18653/v1/2024.acl-long.841.
  URL <https://aclanthology.org/2024.acl-long.841/>.
* Hägele et al. (2024)

  Alex Hägele, Elie Bakouch, Atli Kosson, Leandro Von Werra, Martin Jaggi, et al.
  Scaling laws and compute-optimal training beyond fixed training durations.
  *Advances in Neural Information Processing Systems*, 37:76232–76264, 2024.
* Hendrycks et al. (2021)

  Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
  Measuring massive multitask language understanding.
  In *International Conference on Learning Representations*, 2021.
  URL <https://openreview.net/forum?id=d7KBjmI3GmQ>.
* Hoffmann et al. (2022)

  Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katherine Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Oriol Vinyals, Jack William Rae, and Laurent Sifre.
  An empirical analysis of compute-optimal large language model training.
  In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), *Advances in Neural Information Processing Systems*, 2022.
  URL <https://openreview.net/forum?id=iBBcRUlOAPR>.
* Hu et al. (2024)

  Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al.
  Minicpm: Unveiling the potential of small language models with scalable training strategies.
  *arXiv preprint arXiv:2404.06395*, 2024.
* Hutchinson (1989)

  M.F. Hutchinson.
  A stochastic estimator of the trace of the influence matrix for laplacian smoothing splines.
  *Communications in Statistics - Simulation and Computation*, 18(3):1059–1076, 1989.
  doi: 10.1080/03610918908812806.
  URL <https://doi.org/10.1080/03610918908812806>.
* Ibrahim et al. (2024)

  Adam Ibrahim, Benjamin Thérien, Kshitij Gupta, Mats Leon Richter, Quentin Gregory Anthony, Eugene Belilovsky, Timothée Lesort, and Irina Rish.
  Simple and scalable strategies to continually pre-train large language models.
  *Transactions on Machine Learning Research*, 2024.
  ISSN 2835-8856.
  URL <https://openreview.net/forum?id=DimPeeCxKO>.
* Izmailov et al. (2018)

  Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson.
  Averaging weights leads to wider optima and better generalization.
  In *Conference on Uncertainty in Artificial Intelligence*, pp. 876–885, 2018.
* Jin et al. (2023)

  Hongpeng Jin, Wenqi Wei, Xuyu Wang, Wenbin Zhang, and Yanzhao Wu.
  Rethinking learning rate tuning in the era of large language models.
  In *2023 IEEE 5th International Conference on Cognitive Machine Intelligence (CogMI)*, pp. 112–121. IEEE, 2023.
* Ju et al. (2022)

  Haotian Ju, Dongyue Li, and Hongyang R Zhang.
  Robust fine-tuning of deep neural networks with hessian-based generalization guarantees.
  In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), *Proceedings of the 39th International Conference on Machine Learning*, volume 162 of *Proceedings of Machine Learning Research*, pp. 10431–10461. PMLR, 17–23 Jul 2022.
  URL <https://proceedings.mlr.press/v162/ju22a.html>.
* Kaddour et al. (2022)

  Jean Kaddour, Linara Adilova Key, Bernhard Schölkopf, and Andrew Gordon Wilson.
  When do flat minima optimizers work?
  In *Advances in Neural Information Processing Systems*, volume 35, pp. 16577–16595, 2022.
* Kaplan et al. (2020)

  Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei.
  Scaling laws for neural language models.
  *arXiv preprint arXiv:2001.08361*, 2020.
* Kaur et al. (2023)

  Simran Kaur, Jeremy Cohen, and Zachary Chase Lipton.
  On the maximum hessian eigenvalue and generalization.
  In Javier Antorán, Arno Blaas, Fan Feng, Sahra Ghalebikesabi, Ian Mason, Melanie F. Pradier, David Rohde, Francisco J. R. Ruiz, and Aaron Schein (eds.), *Proceedings on ”I Can’t Believe It’s Not Better! - Understanding Deep Learning Through Empirical Falsification” at NeurIPS 2022 Workshops*, volume 187 of *Proceedings of Machine Learning Research*, pp. 51–65. PMLR, 03 Dec 2023.
  URL <https://proceedings.mlr.press/v187/kaur23a.html>.
* Keskar et al. (2017)

  Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang.
  On large-batch training for deep learning: Generalization gap and sharp minima.
  In *International Conference on Learning Representations*, 2017.
  URL <https://openreview.net/forum?id=H1oyRlYgg>.
* Le et al. (2022)

  Teven Le, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, et al.
  Bloom: A 176b-parameter open-access multilingual language model.
  *arXiv preprint arXiv:2211.05100*, 2022.
* Li et al. (2025)

  Tao Li, Zhengbao He, Yujun Li, Yasheng Wang, Lifeng Shang, and Xiaolin Huang.
  Flat-loRA: Low-rank adaptation over a flat loss landscape.
  In *Forty-second International Conference on Machine Learning*, 2025.
  URL <https://openreview.net/forum?id=3Qj3xSwN2I>.
* Li et al. (2023)

  Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto.
  Alpacaeval: An automatic evaluator of instruction-following models.
  <https://github.com/tatsu-lab/alpaca_eval>, 5 2023.
* Lin et al. (2022)

  Stephanie Lin, Jacob Hilton, and Owain Evans.
  TruthfulQA: Measuring how models mimic human falsehoods.
  In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 3214–3252, Dublin, Ireland, May 2022. Association for Computational Linguistics.
  doi: 10.18653/v1/2022.acl-long.229.
  URL <https://aclanthology.org/2022.acl-long.229/>.
* Liu et al. (2024a)

  Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al.
  Deepseek-v3 technical report.
  *arXiv preprint arXiv:2412.19437*, 2024a.
* Liu et al. (2023)

  Hong Liu, Sang Michael Xie, Zhiyuan Li, and Tengyu Ma.
  Same pre-training loss, better downstream: Implicit bias matters for language models.
  In *International Conference on Machine Learning*, pp. 22188–22214. PMLR, 2023.
* Liu et al. (2024b)

  Hong Liu, Zhiyuan Li, David Leo Wright Hall, Percy Liang, and Tengyu Ma.
  Sophia: A scalable stochastic second-order optimizer for language model pre-training.
  In *The Twelfth International Conference on Learning Representations*, 2024b.
  URL <https://openreview.net/forum?id=3xHDeA8Noi>.
* Loshchilov & Hutter (2019)

  Ilya Loshchilov and Frank Hutter.
  Decoupled weight decay regularization.
  In *International Conference on Learning Representations*, 2019.
  URL <https://openreview.net/forum?id=Bkg6RiCqY7>.
* Meta (2024a)

  Meta.
  Meta-llama-3.1-8b model card.
  <https://huggingface.co/meta-llama/Meta-Llama-3.1-8B>, 2024a.
  Accessed: 2025-09-21.
* Meta (2024b)

  Meta.
  Llama-3.2-1b model card.
  <https://huggingface.co/meta-llama/Llama-3.2-1B>, 2024b.
  Accessed: 2025-09-21.
* Meta (2024c)

  AIat Meta.
  The llama 3 herd of models, 2024c.
  URL <https://arxiv.org/abs/2407.21783>.
* Mihaylov et al. (2018)

  Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal.
  Can a suit of armor conduct electricity? a new dataset for open book question answering.
  In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (eds.), *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pp. 2381–2391, Brussels, Belgium, October-November 2018. Association for Computational Linguistics.
  doi: 10.18653/v1/D18-1260.
  URL <https://aclanthology.org/D18-1260>.
* OLMo et al. (2024)

  Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, et al.
  2 olmo 2 furious.
  *arXiv preprint arXiv:2501.00656*, 2024.
* Ouyang et al. (2022)

  Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al.
  Training language models to follow instructions with human feedback.
  *Advances in neural information processing systems*, 35:27730–27744, 2022.
* Penedo et al. (2024)

  Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf.
  The FineWeb datasets: Decanting the Web for the finest text data at scale.
  In *The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track*, 2024.
  URL <https://openreview.net/forum?id=n6SCkn2QaG>.
* Rafailov et al. (2023)

  Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn.
  Direct preference optimization: Your language model is secretly a reward model.
  In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023.
  URL <https://openreview.net/forum?id=HPuSIXJaa9>.
* Sakaguchi et al. (2021)

  Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi.
  Winogrande: An adversarial winograd schema challenge at scale.
  *Communications of the ACM*, 64(9):99–106, 2021.
* Sanyal et al. (2024)

  Sunny Sanyal, Atula Tejaswi Neerkaje, Jean Kaddour, Abhishek Kumar, and Sujay Sanghavi.
  Early weight averaging meets high learning rates for LLM pre-training.
  In *First Conference on Language Modeling*, 2024.
  URL <https://openreview.net/forum?id=IA8CWtNkUr>.
* Sardana et al. (2024)

  Nikhil Sardana, Jacob Portes, Sasha Doubov, and Jonathan Frankle.
  Beyond chinchilla-optimal: Accounting for inference in language model scaling laws.
  In *Forty-first International Conference on Machine Learning*, 2024.
  URL <https://openreview.net/forum?id=0bmXrtTDUu>.
* Springer et al. (2025)

  Jacob Mitchell Springer, Sachin Goyal, Kaiyue Wen, Tanishq Kumar, Xiang Yue, Sadhika Malladi, Graham Neubig, and Aditi Raghunathan.
  Overtrained language models are harder to fine-tune.
  *Proceedings of the International Conference on Machine Learning*, 2025.
* Sun & Dredze (2025)

  Kaiser Sun and Mark Dredze.
  Amuro & char: Analyzing the relationship between pre-training and fine-tuning of large language models.
  In Vaibhav Adlakha, Alexandra Chronopoulou, Xiang Lorraine Li, Bodhisattwa Prasad Majumder, Freda Shi, and Giorgos Vernikos (eds.), *Proceedings of the 10th Workshop on Representation Learning for NLP (RepL4NLP-2025)*, pp. 131–151, Albuquerque, NM, May 2025. Association for Computational Linguistics.
  ISBN 979-8-89176-245-9.
  doi: 10.18653/v1/2025.repl4nlp-1.11.
  URL <https://aclanthology.org/2025.repl4nlp-1.11/>.
* Suzgun et al. (2023)

  Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, and Jason Wei.
  Challenging BIG-bench tasks and whether chain-of-thought can solve them.
  In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), *Findings of the Association for Computational Linguistics: ACL 2023*, pp. 13003–13051, Toronto, Canada, July 2023. Association for Computational Linguistics.
  doi: 10.18653/v1/2023.findings-acl.824.
  URL <https://aclanthology.org/2023.findings-acl.824/>.
* Tian et al. (2025)

  Changxin Tian, Jiapeng Wang, Qian Zhao, Kunlong Chen, Jia Liu, Ziqi Liu, Jiaxin Mao, Wayne Xin Zhao, Zhiqiang Zhang, and Jun Zhou.
  Wsm: Decay-free learning rate schedule via checkpoint merging for llm pre-training.
  *arXiv preprint arXiv:2507.17634*, 2025.
* Tissue et al. (2024)

  Howe Tissue, Venus Wang, and Lu Wang.
  Scaling law with learning rate annealing.
  *arXiv preprint arXiv:2408.11029*, 2024.
* Touvron et al. (2023a)

  Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al.
  Llama: Open and efficient foundation language models.
  *arXiv preprint arXiv:2302.13971*, 2023a.
* Touvron et al. (2023b)

  Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al.
  Llama 2: Open foundation and fine-tuned chat models.
  *arXiv preprint arXiv:2307.09288*, 2023b.
* Wang et al. (2025)

  Xingjin Wang, Howe Tissue, Lu Wang, Linjing Li, and Daniel Dajun Zeng.
  Learning dynamics in continual pre-training for large language models.
  In *Forty-second International Conference on Machine Learning*, 2025.
  URL <https://openreview.net/forum?id=Vk1rNMl0J1>.
* Wang et al. (2024)

  Zhihao Wang, Shiyu Liu, Jianheng Huang, Wang Zheng, YiXuan Liao, Xiaoxin Chen, Junfeng Yao, and Jinsong Su.
  A learning rate path switching training paradigm for version updates of large language models.
  In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pp. 13581–13594, Miami, Florida, USA, November 2024. Association for Computational Linguistics.
  doi: 10.18653/v1/2024.emnlp-main.752.
  URL <https://aclanthology.org/2024.emnlp-main.752/>.
* Wen et al. (2025)

  Kaiyue Wen, Zhiyuan Li, Jason S. Wang, David Leo Wright Hall, Percy Liang, and Tengyu Ma.
  Understanding warmup-stable-decay learning rates: A river valley loss landscape view.
  In *The Thirteenth International Conference on Learning Representations*, 2025.
  URL <https://openreview.net/forum?id=m51BgoqvbP>.
* Zellers et al. (2019)

  Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi.
  HellaSwag: Can a machine really finish your sentence?
  In Anna Korhonen, David Traum, and Lluís Màrquez (eds.), *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pp. 4791–4800, Florence, Italy, July 2019. Association for Computational Linguistics.
  doi: 10.18653/v1/P19-1472.
  URL <https://aclanthology.org/P19-1472>.
* Zhong et al. (2024)

  Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan.
  AGIEval: A human-centric benchmark for evaluating foundation models.
  In Kevin Duh, Helena Gomez, and Steven Bethard (eds.), *Findings of the Association for Computational Linguistics: NAACL 2024*, pp. 2299–2314, Mexico City, Mexico, June 2024. Association for Computational Linguistics.
  doi: 10.18653/v1/2024.findings-naacl.149.
  URL <https://aclanthology.org/2024.findings-naacl.149/>.

## Appendix A Model Architecture

We provide detailed specifications for the models used in our experiments.
Both the 1B and 8B models follow the Llama 3 architecture (Meta, [2024c](#bib.bib37)), employing RMSNorm, SwiGLU activation, and Rotary Position Embeddings.
We use the Llama 3 tokenizer with a vocabulary size of 128,256 tokens for all models.

Table 5: Model configurations for the 1B and 8B models.

|  |  |  |
| --- | --- | --- |
| Configuration | 1B | 8B |
| Hidden dimension | 2048 | 4096 |
| FFN dimension | 8192 | 14336 |
| Number of layers | 16 | 32 |
| Number of heads | 32 | 32 |
| Number of KV heads | 8 | 8 |
| Head dimension | 64 | 128 |
| Vocabulary size | 128256 | 128256 |
| RoPE θ\theta | 10000 | 10000 |
| RMS norm ϵ\epsilon | 10−510^{-5} | 10−510^{-5} |
| Activation function | SwiGLU | SwiGLU |

## Appendix B Learning Rate Scheduler Formulations

We provide the complete formulations for the WSD, Cosine, and Linear LR schedulers used in our experiments.

WSD Schedule: After warmup, the LR remains constant until TstableT\_{\text{stable}}, then decays linearly to αpre⋅ηmax\alpha\_{\text{pre}}\cdot\eta\_{\max} at step TT:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ηWSD​(t,αpre)={ηmax⋅tTwarmupt≤TwarmupηmaxTwarmup<t≤Tstableηmax⋅((1−αpre)⋅Tpre−tTpre−Tstable+αpre)Tstable<t≤Tpre\eta^{\text{{WSD}}}({t},\alpha\_{\text{pre}})=\begin{cases}\eta\_{\max}\cdot\frac{t}{T\_{\text{warmup}}}&t\leq T\_{\text{warmup}}\\ \eta\_{\max}&T\_{\text{warmup}}<t\leq T\_{\text{stable}}\\ \eta\_{\max}\cdot\left((1-\alpha\_{\text{pre}})\cdot\frac{T\_{\text{pre}}-t}{T\_{\text{pre}}-T\_{\text{stable}}}+\alpha\_{\text{pre}}\right)&T\_{\text{stable}}<t\leq T\_{\text{pre}}\end{cases} |  | (7) |

WSO Schedule: Obtained by setting αpre=1\alpha\_{\text{pre}}=1 in WSD. After warmup, the LR stays constant:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ηWSO​(t,αpre)={ηmax⋅tTwarmupt≤TwarmupηmaxTwarmup<t≤Tpre\eta^{\text{{WSO}}}({t},\alpha\_{\text{pre}})=\begin{cases}\eta\_{\max}\cdot\frac{t}{T\_{\text{warmup}}}&t\leq T\_{\text{warmup}}\\ \eta\_{\max}&T\_{\text{warmup}}<t\leq T\_{\text{pre}}\\ \end{cases} |  | (8) |

Cosine Schedule: After warmup, the LR follows a Cosine decay to αpre⋅ηmax\alpha\_{\text{pre}}\cdot\eta\_{\max}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ηCosine​(t,αpre)={ηmax⋅tTwarmupt≤Twarmupηmax⋅(αpre+1−αpre2​(1+cos⁡(t−TwarmupTpre−Twarmup⋅π)))t>Twarmup\eta^{\text{{Cosine}}}({t},\alpha\_{\text{pre}})=\begin{cases}\eta\_{\max}\cdot\frac{t}{T\_{\text{warmup}}}&t\leq T\_{\text{warmup}}\\ \eta\_{\max}\cdot\left(\alpha\_{\text{pre}}+\frac{1-\alpha\_{\text{pre}}}{2}\left(1+\cos\left(\frac{t-T\_{\text{warmup}}}{T\_{\text{pre}}-T\_{\text{warmup}}}\cdot\pi\right)\right)\right)&t>T\_{\text{warmup}}\end{cases} |  | (9) |

Linear Schedule: After warmup, the LR decays linearly to αpre⋅ηmax\alpha\_{\text{pre}}\cdot\eta\_{\max}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ηLinear​(t,αpre)={ηmax⋅tTwarmupt≤Twarmupηmax⋅((1−αpre)⋅Tpre−tTpre−Twarmup+αpre)t>Twarmup\eta^{\text{{Linear}}}({t},\alpha\_{\text{pre}})=\begin{cases}\eta\_{\max}\cdot\frac{t}{T\_{\text{warmup}}}&t\leq T\_{\text{warmup}}\\ \eta\_{\max}\cdot\left((1-\alpha\_{\text{pre}})\cdot\frac{T\_{\text{pre}}-t}{T\_{\text{pre}}-T\_{\text{warmup}}}+\alpha\_{\text{pre}}\right)&t>T\_{\text{warmup}}\end{cases} |  | (10) |

All the schedulers use the same warmup phase as described in Section [2.4](#S2.SS4 "2.4 Formalization of Learning Rate Schedulers ‣ 2 Preliminaries ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning"), and their decay is controlled by the minimum LR factor αpre∈[0.0,1.0]\alpha\_{\text{pre}}\in[0.0,1.0].

##### Mid-training LR Scheduling.

In the mid-training stage, we extend the pre-training learning rate schedulers.
The mid-training learning rate at time step tt is defined as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ηScheduler​(t,αpre,αmid)=ηScheduler​(Tpre,αpre)⋅((1−αmid)⋅Tpre+Tmid−tTmid+αmid)\eta^{\text{Scheduler}}(t,\alpha\_{\text{pre}},\alpha\_{\text{mid}})=\eta^{\text{Scheduler}}(T\_{\text{pre}},\alpha\_{\text{pre}})\cdot\left((1-\alpha\_{\text{mid}})\cdot\frac{T\_{\text{pre}}+T\_{\text{mid}}-t}{T\_{\text{mid}}}+\alpha\_{\text{mid}}\right) |  | (11) |

for t∈[Tpre+1,Tpre+Tmid]t\in[T\_{\text{pre}}+1,T\_{\text{pre}}+T\_{\text{mid}}], where TpreT\_{\text{pre}} is the total number of pre-training steps and TmidT\_{\text{mid}} is the total number of mid-training steps.

## Appendix C Pre-training Hyperparameters

Table 6: Pre-training hyperparameters for 1B and 8B models. The WSD stable ratio ρ=0.75\rho=0.75 means the LR remains stable for 75% of training after warmup, with decay occurring in the final 25% when αpre<1\alpha\_{\text{pre}}<1.

|  |  |  |
| --- | --- | --- |
| Hyperparameter | 1B | 8B |
| Training Configuration | | |
| Total training steps | 80,000 | 80,000 |
| Total tokens | 350B | 500B |
| Batch size (tokens) | 4,194,304 | 12,582,912 |
| Sequence length | 2,048 | 2,048 |
| Optimizer (AdamW) | | |
| Max LR (ηmax\eta\_{\max}) | 3×10−43\times 10^{-4} | 3×10−43\times 10^{-4} |
| Weight decay | 0.1 | 0.1 |
| Adam β1\beta\_{1} | 0.9 | 0.9 |
| Adam β2\beta\_{2} | 0.95 | 0.95 |
| Adam ϵ\epsilon | 1×10−81\times 10^{-8} | 1×10−81\times 10^{-8} |
| Gradient clipping | 1.0 | 1.0 |
| LR Schedule | | |
| Warmup steps (TwarmupT\_{\text{warmup}}) | 1,000 | 1,000 |
| WSD stable ratio (ρ\rho) | 0.75 | 0.75 |
| Min LR factor (αpre\alpha\_{\text{pre}}) | {0.0, 0.1, 1.0} | {0.0, 0.1, 1.0} |
| Other | | |
| Precision | bfloat16 | bfloat16 |




Table 7: Over-training configuration for the 1B model trained on 2T tokens. All other hyperparameters are identical to those in Table [6](#A3.T6 "Table 6 ‣ Appendix C Pre-training Hyperparameters ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning").

| Hyperparameter | Value |
| --- | --- |
| Training Configuration | |
| Total training steps | 120,000 |
| Total tokens | 2T |
| Batch size (tokens) | 16,777,216 |

We provide detailed hyperparameters used for pre-training our models in Table [6](#A3.T6 "Table 6 ‣ Appendix C Pre-training Hyperparameters ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning").
All experiments use the AdamW optimizer (Loshchilov & Hutter, [2019](#bib.bib34)) with mixed precision.
For over-training experiments, we modify the training duration as shown in Table [7](#A3.T7 "Table 7 ‣ Appendix C Pre-training Hyperparameters ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning"), where the 1B model is trained for 120,000 steps to process 2T tokens and set different batch sizes while maintaining the other hyperparameters in Table [6](#A3.T6 "Table 6 ‣ Appendix C Pre-training Hyperparameters ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning").

## Appendix D SFT Configuration

We performed supervised fine-tuning for all models using the Tulu-3 SFT mixture dataset. Since the official dataset does not provide a predefined train-validation split, we create our own using a 9:1 ratio for training and validation, respectively.
We perform full parameter training for all models.
Table [8](#A4.T8 "Table 8 ‣ Appendix D SFT Configuration ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") presents the hyperparameters used in our experiments.

Table 8: SFT hyperparameters used in our experiments. We perform a
sweep over the specified LRs and select the best value based on AlpacaEval
performance.

| Hyperparameter | Value |
| --- | --- |
| LR | 5.0×10−75.0\times 10^{-7}, 1.0×10−61.0\times 10^{-6}, 5.0×10−65.0\times 10^{-6}, 1.0×10−51.0\times 10^{-5}, 5.0×10−55.0\times 10^{-5}, 1.0×10−41.0\times 10^{-4}, 5.0×10−45.0\times 10^{-4}, 1.0×10−31.0\times 10^{-3} |
| Global Batch size | 128 |
| LR scheduler | Cosine with warmup |
| Minimum LR | 0 |
| Optimizer | AdamW |
| Weight decay | 0.0 |
| Gradient clipping | 1.0 |
| Warmup steps | 100 |
| Epochs | 1 |
| Training precision | bfloat16 |

## Appendix E Evaluation Details

For pre-trained models, all benchmarks are evaluated in a zero-shot setting.

For mid-trained models (before SFT), we evaluate on standard benchmarks following the evaluation suite used in OLMo 2 (OLMo et al., [2024](#bib.bib39)).
We assess reasoning capabilities using ARC-Challenge (Clark et al., [2018](#bib.bib7)), HellaSwag (Zellers et al., [2019](#bib.bib56)), and WinoGrande (Sakaguchi et al., [2021](#bib.bib43)).
Reading comprehension is evaluated with DROP (Dua et al., [2019](#bib.bib10)) using 5-shot prompting, while mathematical reasoning is assessed using GSM8K (Cobbe et al., [2021](#bib.bib8)) with 8-shot chain-of-thought (CoT) prompting.

For SFT models, we use the following evaluation configurations.
For AlpacaEval, following Springer et al. ([2025](#bib.bib46)), rather than comparing against GPT-4o, where the win rates would be uniformly low, we use a reference model of the same architecture to better distinguish performance differences between LR schedules. Specifically, we use the WSO model with αpre=1.0\alpha\_{\text{pre}}=1.0, fine-tuned with the lowest LR from our sweep (5×10−75\times 10^{-7}) as our reference, ensuring stable and meaningful comparisons within each model scale.
Evaluations are performed by Llama-3-70B-Instruct.
For MMLU (5-shot), evaluation covers 57 subjects spanning STEM, humanities, social sciences, and other domains.
For TruthfulQA, we use the standard evaluation protocol.
After mid-training and SFT, we additionally evaluate on GSM8K (1-shot), DROP (5-shot), AGI Eval (Zhong et al., [2024](#bib.bib57)) (3-shot), and BigBench-Hard (Suzgun et al., [2023](#bib.bib48)) (3-shot with CoT).

## Appendix F Full Evaluation Results

This section provides complete per-task evaluation results for all pre-trained and fine-tuned models across different LR schedules. While the main text presents aggregated metrics and relative performance comparisons, here we report the absolute performance values for each individual benchmark.

### F.1 Pre-training Evaluation Results

Table 9: 
Pre-training evaluation results. Models with more decay (αpre=0\alpha\_{\text{pre}}=0) generally achieve lower validation loss, but not always better zero-shot task performance.

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | Scheduler | αpre\alpha\_{\text{pre}} | Valid Loss ↓\downarrow | ARC-e | ARC-c | BoolQ | Hella | OBQA | PIQA | Wino | Avg. |
| 1B | Warmup-Stable-Only (WSO) | 1.0 | 2.431 | 70.8 | 42.2 | 62.0 | 56.3 | 45.4 | 70.8 | 58.5 | 58.0 |
| WSD | 0.1 | 2.364 | 72.0 | 40.0 | 62.1 | 57.4 | 46.4 | 72.5 | 57.1 | 58.2 |
| 0.0 | 2.360 | 72.2 | 39.7 | 63.7 | 57.6 | 45.6 | 72.2 | 58.6 | 58.5 |
| Linear | 0.1 | 2.380 | 70.3 | 42.6 | 63.2 | 55.6 | 45.2 | 71.6 | 55.7 | 57.7 |
| 0.0 | 2.376 | 74.4 | 43.4 | 65.7 | 58.4 | 47.4 | 70.9 | 57.5 | 59.7 |
| Cosine | 0.1 | 2.379 | 71.1 | 43.6 | 66.5 | 59.9 | 47.8 | 71.7 | 56.3 | 59.6 |
| 0.0 | 2.376 | 74.6 | 41.9 | 50.7 | 58.5 | 48.4 | 71.0 | 55.4 | 57.2 |
| 8B | Warmup-Stable-Only (WSO) | 1.0 | 2.119 | 79.4 | 52.6 | 69.1 | 69.1 | 52.8 | 76.3 | 64.5 | 66.3 |
| WSD | 0.1 | 2.011 | 80.4 | 52.8 | 69.1 | 72.6 | 53.2 | 75.9 | 64.0 | 66.9 |
| 0.0 | 2.005 | 81.0 | 53.0 | 67.2 | 72.9 | 54.2 | 76.3 | 65.0 | 67.1 |
| Linear | 0.1 | 2.004 | 79.4 | 53.7 | 64.1 | 71.2 | 50.4 | 75.0 | 62.4 | 65.2 |
| 0.0 | 1.992 | 76.6 | 48.2 | 71.1 | 71.5 | 53.6 | 74.9 | 61.3 | 65.3 |
| Cosine | 0.1 | 2.001 | 76.3 | 47.6 | 71.3 | 71.5 | 52.4 | 74.3 | 60.9 | 64.9 |
| 0.0 | 2.000 | 74.2 | 46.8 | 71.7 | 71.4 | 52.6 | 76.3 | 60.8 | 64.8 |

Table [9](#A6.T9 "Table 9 ‣ F.1 Pre-training Evaluation Results ‣ Appendix F Full Evaluation Results ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") presents comprehensive zero-shot evaluation results for all pre-trained models across different LR schedules.

#### F.1.1 Pre-training Evaluation Results in Over-training

Table 10: 
Pre-training evaluation results for over-trained 1B models (2T tokens).

| Model | Scheduler | αpre\alpha\_{\text{pre}} | Valid Loss ↓\downarrow | ARC-e | ARC-c | BoolQ | Hella | OBQA | PIQA | Wino | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1B | Warmup-Stable-Only (WSO) | 1.0 | 2.625 | 74.4 | 43.3 | 59.7 | 63.5 | 48.6 | 73.2 | 62.0 | 60.7 |
| WSD | 0.1 | 2.582 | 75.4 | 45.8 | 60.0 | 66.6 | 50.0 | 74.7 | 62.7 | 62.2 |
| 0.0 | 2.578 | 75.3 | 46.8 | 59.2 | 66.2 | 50.4 | 74.4 | 63.0 | 62.2 |
| Linear | 0.1 | 2.599 | 73.4 | 45.6 | 64.7 | 65.4 | 48.0 | 73.2 | 58.8 | 61.3 |
| 0.0 | 2.595 | 73.9 | 44.4 | 66.6 | 65.2 | 49.2 | 73.6 | 59.8 | 61.8 |
| Cosine | 0.1 | 2.595 | 72.9 | 44.1 | 65.7 | 64.9 | 52.0 | 74.0 | 61.5 | 62.2 |
| 0.0 | 2.595 | 73.7 | 44.4 | 64.4 | 64.0 | 45.8 | 72.7 | 61.1 | 60.9 |

Table [10](#A6.T10 "Table 10 ‣ F.1.1 Pre-training Evaluation Results in Over-training ‣ F.1 Pre-training Evaluation Results ‣ Appendix F Full Evaluation Results ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") shows that, also in the over-training regime with 2T tokens, the Cosine scheduler with decay achieves slightly better zero-shot task performance and lower validation loss compared to WSO.

### F.2 SFT Evaluation Results

Table 11: SFT learning rates selected for each pre-trained model based on AlpacaEval performance.

|  |  |  |  |
| --- | --- | --- | --- |
| Model | Scheduler | αpre\alpha\_{\text{pre}} | Selected SFT LR |
| 1B | Warmup-Stable-Only (WSO) | 1.0 | 3×10−43\times 10^{-4} |
| WSD | 0.1 | 1×10−41\times 10^{-4} |
| 0.0 | 1×10−41\times 10^{-4} |
| Linear | 0.1 | 1×10−41\times 10^{-4} |
| 0.0 | 1×10−41\times 10^{-4} |
| Cosine | 0.1 | 1×10−41\times 10^{-4} |
| 0.0 | 1×10−41\times 10^{-4} |
| 8B | Warmup-Stable-Only (WSO) | 1.0 | 3×10−43\times 10^{-4} |
| WSD | 0.1 | 3×10−43\times 10^{-4} |
| 0.0 | 1×10−41\times 10^{-4} |
| Linear | 0.1 | 1×10−41\times 10^{-4} |
| 0.0 | 1×10−41\times 10^{-4} |
| Cosine | 0.1 | 1×10−41\times 10^{-4} |
| 0.0 | 3×10−53\times 10^{-5} |




Table 12: 
SFT evaluation results. Models pre-trained with WSO achieve the best downstream performance.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Model | Scheduler | αpre\alpha\_{\text{pre}} | AlpacaEval | TruthfulQA | MMLU | Avg. |
| 1B | Warmup-Stable-Only (WSO) | 1.0 | 84.0 | 43.4 | 35.9 | 54.4 |
| WSD | 0.1 | 83.9 | 41.9 | 36.6 | 54.1 |
| 0.0 | 82.3 | 40.2 | 36.7 | 53.1 |
| Linear | 0.1 | 82.0 | 42.0 | 36.3 | 53.4 |
| 0.0 | 82.4 | 41.7 | 35.6 | 53.2 |
| Cosine | 0.1 | 83.6 | 41.0 | 35.5 | 53.4 |
| 0.0 | 83.6 | 41.0 | 35.6 | 53.4 |
| 8B | Warmup-Stable-Only (WSO) | 1.0 | 79.7 | 42.5 | 42.7 | 55.0 |
| WSD | 0.1 | 77.1 | 40.8 | 41.4 | 53.1 |
| 0.0 | 77.3 | 39.9 | 43.7 | 53.6 |
| Linear | 0.1 | 76.4 | 41.4 | 42.1 | 53.3 |
| 0.0 | 78.4 | 40.6 | 42.8 | 53.9 |
| Cosine | 0.1 | 78.6 | 39.9 | 42.3 | 53.6 |
| 0.0 | 77.8 | 40.3 | 43.3 | 53.8 |

We select the best learning rate for each pre-trained model based on AlpacaEval performance, as the primary objective of SFT is to enhance instruction-following capabilities.
Selecting hyperparameters based on such as validation loss does not necessarily yield better downstream task performance, which is consistent with our main finding that lower pre-training loss does not guarantee better post-SFT performance.
We apply an identical learning rate sweep to all pre-trained models, ensuring that no scheduler receives a selective advantage.
Table [11](#A6.T11 "Table 11 ‣ F.2 SFT Evaluation Results ‣ Appendix F Full Evaluation Results ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") shows the selected learning rates for each model.

Table [12](#A6.T12 "Table 12 ‣ F.2 SFT Evaluation Results ‣ Appendix F Full Evaluation Results ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") shows performance after SFT across different pre-training schedules. Models pre-trained with WSO or moderate decay (αpre=0.1\alpha\_{\text{pre}}=0.1) often achieve comparable or better downstream performance than those with aggressive decay (αpre=0.0\alpha\_{\text{pre}}=0.0), despite having worse pre-training metrics.

#### F.2.1 SFT Evaluation Results in Over-training

Table 13: SFT learning rates selected for each over-trained 1B model based on AlpacaEval performance.

| Model | Scheduler | αpre\alpha\_{\text{pre}} | Selected SFT LR |
| --- | --- | --- | --- |
| 1B | Warmup-Stable-Only (WSO) | 1.0 | 1×10−41\times 10^{-4} |
| WSD | 0.1 | 3×10−53\times 10^{-5} |
| 0.0 | 3×10−53\times 10^{-5} |
| Linear | 0.1 | 3×10−53\times 10^{-5} |
| 0.0 | 3×10−53\times 10^{-5} |
| Cosine | 0.1 | 1×10−51\times 10^{-5} |
| 0.0 | 1×10−41\times 10^{-4} |




Table 14: 
SFT evaluation results for over-trained 1B models (pre-trained on 2T tokens).

| Model | Scheduler | αpre\alpha\_{\text{pre}} | AlpacaEval | TruthfulQA | MMLU | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| 1B | Warmup-Stable-Only (WSO) | 1.0 | 78.1 | 38.7 | 34.5 | 50.4 |
| WSD | 0.1 | 77.2 | 38.3 | 33.6 | 49.7 |
| 0.0 | 76.0 | 38.4 | 33.7 | 49.4 |
| Linear | 0.1 | 75.6 | 37.8 | 34.2 | 49.2 |
| 0.0 | 75.5 | 37.9 | 33.9 | 49.1 |
| Cosine | 0.1 | 76.0 | 37.9 | 33.9 | 49.3 |
| 0.0 | 76.4 | 37.9 | 33.9 | 49.4 |

Table [13](#A6.T13 "Table 13 ‣ F.2.1 SFT Evaluation Results in Over-training ‣ F.2 SFT Evaluation Results ‣ Appendix F Full Evaluation Results ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") shows the learning rates selected for each over-trained model.
Table [14](#A6.T14 "Table 14 ‣ F.2.1 SFT Evaluation Results in Over-training ‣ F.2 SFT Evaluation Results ‣ Appendix F Full Evaluation Results ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") demonstrates that even after over-training with 2T tokens, WSO achieves superior SFT performance compared to decay-based schedulers.

### F.3 Mid-training Evaluation Results

Table 15: 
Mid-training evaluation results in Section [4](#S4 "4 Experiment 2: Three-stage (Pre-, Mid-, and Post-training) Setting ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning")

|  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | Pre-training Scheduler | αpre\alpha\_{\text{pre}} | αmid\alpha\_{\text{mid}} | Valid Loss ↓\downarrow | ARC-C | HellaSwag | WinoGrande | DROP | GSM8K | Avg. |
| 1B | Warmup-Stable-Only (WSO) | 1.0 | 1.0 | 2.335 | 47.0 | 60.5 | 58.6 | 23.9 | 20.4 | 42.1 |
| WSD | 1.0 | 0.0 | 2.273 | 45.0 | 61.1 | 60.4 | 23.3 | 21.1 | 42.2 |
| 0.1 | 0.0 | 2.320 | 45.0 | 62.0 | 60.7 | 23.8 | 11.0 | 40.5 |
| 0.1 | 1.0 | 2.310 | 45.1 | 60.7 | 59.8 | 24.5 | 13.0 | 40.6 |
| Cosine | 0.1 | 0.0 | 2.332 | 43.8 | 61.4 | 59.5 | 20.2 | 10.7 | 39.1 |
| 0.1 | 1.0 | 2.326 | 44.3 | 60.7 | 59.7 | 21.4 | 12.8 | 39.8 |
| Linear | 0.1 | 0.0 | 2.330 | 43.0 | 60.3 | 60.5 | 19.6 | 11.0 | 38.9 |
| 0.1 | 1.0 | 2.325 | 43.2 | 60.3 | 60.1 | 23.6 | 13.3 | 40.1 |
| 8B | Warmup-Stable-Only (WSO) | 1.0 | 1.0 | 2.009 | 64.9 | 75.4 | 69.4 | 49.7 | 52.8 | 62.4 |
| WSD | 1.0 | 0.0 | 1.907 | 69.7 | 77.9 | 70.6 | 50.6 | 53.9 | 64.5 |
| 0.1 | 0.0 | 1.988 | 61.4 | 80.0 | 71.1 | 42.6 | 39.7 | 59.0 |
| 0.1 | 1.0 | 1.964 | 62.4 | 79.4 | 71.0 | 42.4 | 42.4 | 59.5 |
| Cosine | 0.1 | 0.0 | 1.991 | 54.3 | 77.0 | 69.7 | 35.4 | 36.0 | 54.5 |
| 0.1 | 1.0 | 1.975 | 57.1 | 77.5 | 69.1 | 38.6 | 40.3 | 56.5 |
| Linear | 0.1 | 0.0 | 1.989 | 55.5 | 77.3 | 71.0 | 36.2 | 37.7 | 55.5 |
| 0.1 | 1.0 | 1.974 | 56.7 | 77.5 | 69.9 | 36.6 | 40.3 | 56.2 |

Table [15](#A6.T15 "Table 15 ‣ F.3 Mid-training Evaluation Results ‣ Appendix F Full Evaluation Results ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") presents evaluation results after the mid-training stage.

#### F.3.1 Mid-training Evaluation Results in Over-training

Table 16: 
Mid-training evaluation results for over-trained 1B models (pre-trained on 2T tokens, mid-trained on 500B tokens).

|  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | Pre-training Scheduler | αpre\alpha\_{\text{pre}} | αmid\alpha\_{\text{mid}} | Valid Loss ↓\downarrow | ARC-C | HellaSwag | WinoGrande | DROP | GSM8K | Avg. |
| 1B | Warmup-Stable-Only (WSO) | 1.0 | 1.0 | 2.254 | 46.7 | 61.3 | 60.4 | 27.0 | 23.1 | 43.7 |
| WSD | 1.0 | 0.0 | 2.199 | 47.1 | 65.2 | 62.2 | 23.7 | 13.7 | 42.4 |
| 0.1 | 0.0 | 2.237 | 47.1 | 65.2 | 62.2 | 23.4 | 13.7 | 42.3 |
| 0.1 | 1.0 | 2.231 | 47.4 | 65.7 | 62.6 | 25.3 | 19.1 | 44.0 |
| Cosine | 0.1 | 0.0 | 2.253 | 46.0 | 65.1 | 62.3 | 23.8 | 11.4 | 41.7 |
| 0.1 | 1.0 | 2.245 | 43.5 | 64.7 | 62.1 | 25.9 | 14.8 | 42.2 |
| Linear | 0.1 | 0.0 | 2.250 | 47.2 | 63.4 | 59.4 | 20.9 | 15.4 | 41.3 |
| 0.1 | 1.0 | 2.267 | 45.6 | 63.3 | 60.1 | 21.4 | 18.7 | 41.8 |

Table [16](#A6.T16 "Table 16 ‣ F.3.1 Mid-training Evaluation Results in Over-training ‣ F.3 Mid-training Evaluation Results ‣ Appendix F Full Evaluation Results ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") shows that after over-training and mid-training, WSO achieves superior overall performance despite having nearly identical validation loss.

### F.4 SFT Evaluation Results After Mid-training

Table 17: SFT learning rates selected for each model configuration based on AlpacaEval performance.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Model | Scheduler | αpre\alpha\_{\text{pre}} | αmid\alpha\_{\text{mid}} | Selected SFT LR |
| 1B | Warmup-Stable-Only (WSO) | 1.0 | 1.0 | 3×10−43\times 10^{-4} |
| WSD | 1.0 | 0.0 | 3×10−53\times 10^{-5} |
| 0.1 | 1.0 | 1×10−41\times 10^{-4} |
| 0.1 | 0.0 | 3×10−53\times 10^{-5} |
| Linear | 0.1 | 1.0 | 3×10−53\times 10^{-5} |
| 0.1 | 0.0 | 3×10−53\times 10^{-5} |
| Cosine | 0.1 | 1.0 | 3×10−53\times 10^{-5} |
| 0.1 | 0.0 | 1×10−41\times 10^{-4} |
| 8B | Warmup-Stable-Only (WSO) | 1.0 | 1.0 | 1×10−61\times 10^{-6} |
| WSD | 1.0 | 0.0 | 1×10−61\times 10^{-6} |
| 0.1 | 1.0 | 1×10−41\times 10^{-4} |
| 0.1 | 0.0 | 3×10−53\times 10^{-5} |
| Linear | 0.1 | 1.0 | 1×10−51\times 10^{-5} |
| 0.1 | 0.0 | 1×10−51\times 10^{-5} |
| Cosine | 0.1 | 1.0 | 1×10−51\times 10^{-5} |
| 0.1 | 0.0 | 1×10−51\times 10^{-5} |




Table 18: 
SFT evaluation results after mid-training. WSO throughout pre- and mid-training generally achieves better SFT performance.

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | Pre-training Scheduler | αpre\alpha\_{\text{pre}} | αmid\alpha\_{\text{mid}} | AlpacaEval | TruthfulQA | GSM8K | DROP | AGI Eval | BBH | MMLU | Avg. |
| 1B | Warmup-Stable-Only (WSO) | 1.0 | 1.0 | 79.4 | 41.8 | 29.0 | 22.7 | 21.8 | 23.1 | 35.7 | 36.2 |
| WSD | 1.0 | 0.0 | 79.4 | 39.9 | 27.2 | 22.0 | 21.5 | 22.7 | 35.4 | 35.4 |
| 0.1 | 0.0 | 76.8 | 41.0 | 18.9 | 22.0 | 22.4 | 23.8 | 34.2 | 34.2 |
| 0.1 | 1.0 | 78.7 | 40.0 | 21.2 | 23.7 | 23.1 | 23.8 | 34.4 | 35.0 |
| Cosine | 0.1 | 0.0 | 72.9 | 38.1 | 19.9 | 17.6 | 22.1 | 17.9 | 33.9 | 31.8 |
| 0.1 | 1.0 | 74.3 | 37.9 | 22.2 | 17.1 | 22.6 | 19.6 | 34.0 | 32.5 |
| Linear | 0.1 | 0.0 | 73.2 | 39.1 | 14.0 | 16.2 | 22.1 | 22.3 | 34.3 | 31.6 |
| 0.1 | 1.0 | 76.3 | 40.8 | 17.7 | 16.3 | 22.8 | 21.4 | 35.1 | 32.9 |
| 8B | Warmup-Stable-Only (WSO) | 1.0 | 1.0 | 64.1 | 43.4 | 54.7 | 36.4 | 40.2 | 31.2 | 42.9 | 44.7 |
| WSD | 1.0 | 0.0 | 68.6 | 44.8 | 34.5 | 32.6 | 40.0 | 30.9 | 44.3 | 42.2 |
| 0.1 | 0.0 | 66.8 | 44.1 | 40.9 | 28.3 | 36.4 | 31.5 | 49.6 | 42.5 |
| 0.1 | 1.0 | 69.7 | 43.9 | 47.3 | 29.9 | 36.3 | 29.0 | 49.5 | 43.6 |
| Cosine | 0.1 | 0.0 | 64.7 | 41.1 | 41.0 | 26.9 | 32.3 | 27.9 | 43.0 | 39.6 |
| 0.1 | 1.0 | 63.9 | 41.9 | 40.8 | 28.8 | 34.6 | 28.5 | 42.8 | 40.2 |
| Linear | 0.1 | 0.0 | 63.9 | 42.5 | 36.8 | 28.3 | 33.6 | 29.3 | 44.9 | 39.9 |
| 0.1 | 1.0 | 63.8 | 41.3 | 43.5 | 30.5 | 33.0 | 31.0 | 46.8 | 41.4 |

Table [17](#A6.T17 "Table 17 ‣ F.4 SFT Evaluation Results After Mid-training ‣ Appendix F Full Evaluation Results ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") shows the optimal learning rates selected for each pre-trained model based on AlpacaEval performance.

Table [18](#A6.T18 "Table 18 ‣ F.4 SFT Evaluation Results After Mid-training ‣ Appendix F Full Evaluation Results ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") shows SFT performance after mid-training.
WSO during mid-training (αmid=1.0\alpha\_{\text{mid}}=1.0) generally achieves better SFT performance compared to those with decay (αmid=0.0\alpha\_{\text{mid}}=0.0).

### F.5 SFT Evaluation Results After Over-training with Mid-training

Table 19: SFT learning rates selected for each over-trained and mid-trained 1B model based on AlpacaEval performance.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Model | Scheduler | αpre\alpha\_{\text{pre}} | αmid\alpha\_{\text{mid}} | Selected SFT LR |
| 1B | Warmup-Stable-Only (WSO) | 1.0 | 1.0 | 1×10−51\times 10^{-5} |
| WSD | 1.0 | 0.0 | 1×10−51\times 10^{-5} |
| 0.1 | 1.0 | 3×10−53\times 10^{-5} |
| 0.1 | 0.0 | 1×10−51\times 10^{-5} |
| Linear | 0.1 | 1.0 | 1×10−51\times 10^{-5} |
| 0.1 | 0.0 | 1×10−51\times 10^{-5} |
| Cosine | 0.1 | 1.0 | 1×10−51\times 10^{-5} |
| 0.1 | 0.0 | 1×10−51\times 10^{-5} |




Table 20: 
SFT evaluation results for over-trained 1B models after mid-training (pre-trained on 2T tokens, mid-trained on 500B tokens, then supervised fine-tuned).

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | Pre-training Scheduler | αpre\alpha\_{\text{pre}} | αmid\alpha\_{\text{mid}} | AlpacaEval | TruthfulQA | GSM8K | DROP | AGI Eval | BBH | MMLU | Avg. |
| 1B | Warmup-Stable-Only (WSO) | 1.0 | 1.0 | 66.2 | 38.1 | 30.3 | 19.4 | 24.1 | 24.8 | 36.6 | 34.2 |
| WSD | 1.0 | 0.0 | 64.0 | 40.4 | 20.4 | 18.4 | 21.3 | 26.1 | 35.5 | 32.3 |
| 0.1 | 0.0 | 64.8 | 39.8 | 15.6 | 19.5 | 21.4 | 23.7 | 36.0 | 31.5 |
| 0.1 | 1.0 | 62.1 | 39.7 | 21.9 | 16.8 | 21.1 | 25.0 | 35.9 | 31.8 |
| Cosine | 0.1 | 0.0 | 62.5 | 41.1 | 18.7 | 20.5 | 23.2 | 18.8 | 35.9 | 31.5 |
| 0.1 | 1.0 | 64.6 | 42.0 | 21.0 | 18.7 | 23.0 | 20.0 | 35.4 | 32.1 |
| Linear | 0.1 | 0.0 | 64.7 | 39.0 | 20.2 | 19.6 | 22.6 | 24.2 | 34.8 | 32.2 |
| 0.1 | 1.0 | 66.8 | 39.4 | 22.3 | 19.5 | 22.6 | 23.8 | 35.0 | 32.2 |

Table [19](#A6.T19 "Table 19 ‣ F.5 SFT Evaluation Results After Over-training with Mid-training ‣ Appendix F Full Evaluation Results ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") shows the selected learning rates for each over-trained model, and Table [20](#A6.T20 "Table 20 ‣ F.5 SFT Evaluation Results After Over-training with Mid-training ‣ Appendix F Full Evaluation Results ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") shows that WSO achieves superior SFT performance compared to decay-based schedulers.

## Appendix G Mid-training Configuration Details

Table 21: Mid-training configuration for 1B and 8B models.

| Hyperparameter | 1B | 8B |
| --- | --- | --- |
| Training Configuration | | |
| Total training steps | 36,000 | 36,000 |
| Total tokens | 150B | 225B |
| Batch size (tokens) | 4,194,304 | 12,582,912 |
| Sequence length | 2,048 | 2,048 |




Table 22: Mid-training configurations in over-training settings for the 1B model trained on 500BT tokens. All other hyperparameters are identical to those in Table [6](#A3.T6 "Table 6 ‣ Appendix C Pre-training Hyperparameters ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning").

| Hyperparameter | Value |
| --- | --- |
| Training Configuration | |
| Total training steps | 30,000 |
| Total tokens | 500BT |
| Batch size (tokens) | 16,777,216 |

We provide the detailed configuration used for mid-training experiments in Table [21](#A7.T21 "Table 21 ‣ Appendix G Mid-training Configuration Details ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning").
Other hyperparameters are the same as the configurations of pre-training in Table [6](#A3.T6 "Table 6 ‣ Appendix C Pre-training Hyperparameters ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning")
Mid-training is conducted on the dolmino-mix-1124 dataset, which consists of diverse high-quality data sources.

Additionally, we provide the detailed hyperparameters used for mid-training in over-training settings in Section [5](#S5 "5 Experiment 3: Three-stage Setting in the Over-training ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") in Table [22](#A7.T22 "Table 22 ‣ Appendix G Mid-training Configuration Details ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning")

## Appendix H Sharpness Computation Details

We compute the sharpness (Hessian trace) using Hutchinson’s stochastic trace estimator (Hutchinson, [1989](#bib.bib18)), which provides an unbiased estimate through random vector sampling. For a Hessian matrix 𝐇\mathbf{H}, the trace is estimated as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Tr​(𝐇)≈1N​∑i=1N𝐳iT​𝐇𝐳i\mathrm{Tr}(\mathbf{H})\approx\frac{1}{N}\sum\_{i=1}^{N}\mathbf{z}\_{i}^{T}\mathbf{H}\mathbf{z}\_{i} |  | (12) |

where 𝐳i\mathbf{z}\_{i} are random vectors sampled from a Rademacher distribution (i.e., each element is ±1\pm 1 with equal probability).

##### Implementation Details.

We compute Hessian-vector products using automatic differentiation, which allows efficient computation without explicitly constructing the full Hessian matrix.

Table 23: Configuration for sharpness (Hessian trace) computation using Hutchinson’s estimator.

|  |  |
| --- | --- |
| Hyperparameter | Value |
| Sequence length | 1,024 |
| Batch size | 1 |
| Number of views | 2 |
| Hutchinson samples | 50 |
| Maximum batches | 4,096 |
| Maximum texts | 16,192 |

Table [23](#A8.T23 "Table 23 ‣ Implementation Details. ‣ Appendix H Sharpness Computation Details ‣ Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning") shows computation configurations for Hutchinson’s estimator.
We measure sharpness at regular intervals throughout pre-training (every 4,000 steps) on held-out validation sets from both the pre-training dataset and the SFT dataset to understand how the loss landscape geometry evolves during training.

[◄](/html/2603.16126)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2603.16127)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2603.16127)
[View original  
on arXiv](https://arxiv.org/abs/2603.16127)[►](/html/2603.16128)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Mon Apr 6 03:18:44 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
