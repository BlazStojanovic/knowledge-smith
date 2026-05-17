---
arxiv: '2504.12491'
authors:
- Hansi Zeng
- Kai Hui
- Honglei Zhuang
- Zhen Qin
- Zhenrui Yue
- Hamed Zamani
- Dana Alon
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?
url: https://arxiv.org/abs/2504.12491
year: 2025
---

# Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?

Hansi Zeng

Kai Hui
Google DeepMind

Honglei Zhuang
Google DeepMind

Zhen Qin
Google DeepMind

Zhenrui Yue
University of Illinois Urbana-Champaign

Hamed Zamani
University of Massachusetts Amherst

Dana Alon
Google DeepMind

###### Abstract

While metrics available during pre-training, such as perplexity, correlates well with model performance at scaling-laws studies, their predictive capacities at a fixed model size remains unclear, hindering effective model selection and development.
To address this gap, we formulate the task of selecting pre-training checkpoints to maximize downstream fine-tuning performance as a pairwise classification problem: predicting which of two LLMs, differing in their pre-training, will perform better after supervised fine-tuning (SFT). We construct a dataset using 50 1B parameter LLM variants with systematically varied pre-training configurations, e.g., objectives or data, and evaluate them on diverse downstream tasks after SFT. We first conduct a study and demonstrate that the conventional perplexity is a misleading indicator. As such, we introduce novel unsupervised and supervised proxy metrics derived from pre-training that successfully reduce the relative performance prediction error rate by over 50%. Despite the inherent complexity of this task, we demonstrate the practical utility of our proposed proxies in specific scenarios, paving the way for more efficient design of pre-training schemes optimized for various downstream tasks.

## 1 Introduction

Large Language Models (LLMs) (Google et al., [2024](#bib.bib13); OpenAI, [2023](#bib.bib31); Chowdhery et al., [2023](#bib.bib7); Grattafiori et al., [2024](#bib.bib14)) are central to contemporary NLP, powering systems like Chatbots and specialized assistants. They are typically employed via few-shot prompting or task-specific fine-tuning.
Despite the accessibility afforded by prompting, fine-tuning on downstream tasks is often indispensable for optimal model performance, particularly within specific application domains or when utilizing private data (Singhal et al., [2025](#bib.bib40); Lee et al., [2024b](#bib.bib25); Lai et al., [2023](#bib.bib23)).

While LLMs demonstrably improve on supervised fine-tuning (SFT) tasks with increasing scale (Zhang et al., [2024](#bib.bib52); Isik et al., [2025](#bib.bib18)), the substantial costs associated with larger models strongly motivate performance optimization at a fixed size. These efforts often concentrate on refining pre-training elements, such as data compositions (Shen et al., [2024](#bib.bib39); Penedo et al., [2024](#bib.bib35)) or training objectives (Raffel et al., [2020](#bib.bib37); Tay et al., [2023a](#bib.bib42), [b](#bib.bib43)). This context underscores a critical need: the ability to reliably forecast the post-SFT performance of same-size LLM variants using only indicators available during pre-training. Although metrics like perplexity correlate well with scaling-driven performance gains (lower perplexity generally corresponds to better few-shot (Grattafiori et al., [2024](#bib.bib14)) and fine-tuning (Isik et al., [2025](#bib.bib18)) results as model size expands), their predictive efficacy for fine-tuning outcomes within a constant model size remains uncertain. Practically, dependable predictors are essential to avoid the prohibitive expense of fine-tuning numerous checkpoints. This requirement is especially pronounced for monitoring and guiding decisions throughout the lengthy pre-training cycles (often months) of very large models (Liu et al., [2024a](#bib.bib26); Grattafiori et al., [2024](#bib.bib14)), and also when subsequent fine-tuning involves substantial datasets,
including potentially stopping unpromising runs early.

!(/html/2504.12491/assets/figures/proxy_sft_error.png)

Figure 1: 
Mean pairwise error rates across three SFT tasks (separate plots). Each plot compares perplexity, the best individual proxy (Section [3](#S3 "3 Predictive Power on SFT Tasks ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?")), and the learning-to-compare proxy (shown on the x-axis). The y-axis represents the error rate, defined as the proportion of mis-classified LLM pairs regarding post-SFT performance.

To investigate the predictability of fine-tuning outcome within feasible computational limits, our study employs a controlled methodology using smaller models. We train multiple variants of a 1B-parameter language model, each incorporating systematic variations in its pre-training configuration. We then evaluate the accuracy of potential predictors by comparing their values at the final pre-training checkpoints against the models’ eventual performance after supervised fine-tuning (SFT). While simplified, we posit that this approach provides representative insights into the core question–whether fine-tuning outcome can be reliably predicted during and after pre-training.
Specifically, we generated 50 distinct 1B-parameter LLM variants by systematically altering pre-training objectives (Raffel et al., [2020](#bib.bib37); Tay et al., [2023a](#bib.bib42), [b](#bib.bib43)), data composition strategies (Shen et al., [2024](#bib.bib39)), and data processing techniques such as filtering and domain tagging (Penedo et al., [2024](#bib.bib35)).
These pre-trained models were subsequently fine-tuned across a diverse suite of tasks, including commonsense reasoning, retrieval-augmented generation and closed-book question answering. Specifically, we select five datasets (Clark et al., [2019](#bib.bib8); Zellers et al., [2019](#bib.bib51); Bisk et al., [2019](#bib.bib4); Mihaylov et al., [2018](#bib.bib28); Sakaguchi et al., [2021](#bib.bib38)) for commonsense reasoning, four (Kwiatkowski et al., [2019](#bib.bib22); Joshi et al., [2017](#bib.bib19); Yang et al., [2018](#bib.bib49); Ho et al., [2020](#bib.bib16)) for retrieval-augmented generation, and two (Kwiatkowski et al., [2019](#bib.bib22); Joshi et al., [2017](#bib.bib19)) for closed-book question answering.
To align with the practical model development scenarios where the primary goal is to identify top performers from a set of candidate models, we formulate the prediction challenge as a pairwise classification task: given two pre-trained models differing only in pre-training, the goal is to predict which model will achieve superior performance after SFT.

We begin by evaluating conventional perplexity, computed using a causal language modeling objective (Brown et al., [2020](#bib.bib5)), as a predictor of SFT performance. Surprisingly, this standard metric correlates poorly with the downstream results of the LLMs after fine-tuning, resulting in prediction error rates exceeding 60% across all three evaluated tasks–worse than the 50% error rate of random guessing (Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?")).
Motivated by prior work (Raffel et al., [2020](#bib.bib37); Tay et al., [2023a](#bib.bib42); Von Oswald et al., [2023](#bib.bib45)), we then introduce alternative pre-training available proxies, including span corruption-based perplexity and k-shot learning performance (Min et al., [2022](#bib.bib29)). These proxies yield substantially improved prediction accuracy; the best-performing proxy for each task reduces the error rate by nearly half compared to conventional perplexity (Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?")). For example, in the commonsense reasoning task, the error rate drops from 69.4%69.4\% to 31.3%31.3\%.
Furthermore, we propose a learning-to-compare (LTC) framework that integrates multiple proxies via supervised classification. By learning interactions across these heterogeneous signals, the LTC approach achieves more robust performance estimation and further decreases the predictive error.
The contributions of this paper are three-folds.

* •

  We present the first formal study focused on predicting post-SFT performance across LLMs of identical size using pretraining signals—departing from prior scaling-based analyses.
* •

  Our work demonstrates the insufficiency of perplexity for this prediction task and introduces novel unsupervised and supervised proxies achieving over a 50% reduction in error rates.
* •

  Our work underscores the challenges of predicting supervised fine-tuning performance and confirms the practical value of the proposed proxies in specific scenarios; to foster further research, we provide the SFT performance data and individual pre-training proxy measurements in Appendix Table [6](#A7.T6 "Table 6 ‣ Appendix G Supervised Finetuned, Perplexity and Kshot Results of LLMs ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?").

## 2 Problem Definition and Setup

This section defines the problem and details the setup, including the generation of diverse LLM variants, the target SFT tasks, and the pre-training signals used as prediction proxies.

### 2.1 LLM Variants and Target SFT Tasks

#### LLM model variations.

To approximate pre-training studies while maintaining reasonable computational resources, we continuously trained a 1B parameter LLM with 100B tokens, systematically ablating pre-training objectives, data mixture re-weighting, and data filtering and tagging. This continuous pre-training approach allowed us to generate a wider range of model variants while managing computational resources.
Pre-training objectives: We explored seven pre-training objectives: causal language modeling (CLM) (Brown et al., [2020](#bib.bib5)), span corruption (SC) (Raffel et al., [2020](#bib.bib37)), prefix language modeling (PLM) (Raffel et al., [2020](#bib.bib37)), SC+CLM, UL2 (Tay et al., [2023a](#bib.bib42)), UL2R (Tay et al., [2023b](#bib.bib43)), and UL2R+CLM (Garcia et al., [2023](#bib.bib12)). CLM and PLM generate tokens left-to-right, with CLM using the full context and PLM conditioning on a prefix. SC reconstructs masked spans, parameterized by noise density and mean span length, set to (0.15,3)(0.15,3) following (Raffel et al., [2020](#bib.bib37)). SC+CLM jointly trains SC and CLM. UL2 mixes six SC variants with PLM, while UL2R uses two SC settings—(0.15,3)(0.15,3) and (0.5,32)(0.5,32)—with PLM. UL2R+CLM extends UL2R by adding a CLM objective.
Mixture re-weighting: We train on the 627B-token Slimpajama corpus (Soboleva et al., [2023](#bib.bib41)), which includes seven diverse domains. We reweigh different domains following  (Shen et al., [2024](#bib.bib39)), producing six 100B-token subsets by adjusting domain distributions (detailed in Table [3](#A1.T3 "Table 3 ‣ Appendix A Pretraining and LLMs ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?") in Appendix);
Data filtering and tagging: Source domain metadata was integrated by pre-pending each instance with its respective domain label (e.g., [Common Crawl]). Length-based sub-corpora were generated by selecting instances within the [25%, 75%] and [75%, 100%] token length quantiles.
We in total produced 50 distinct LLM variants, the specifications of which are provided in Table 
[4](#A1.T4 "Table 4 ‣ Appendix A Pretraining and LLMs ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?") in Appendix.

#### Target SFT tasks.

We employed commonsense reasoning (CMS), retrieval-augmented generation (RAG), and closed-book question answering (CBQA) as the target supervised fine-tuning (SFT) tasks. These tasks were chosen to assess critical LLM capabilities such as reasoning, context utilization, and memorization, which are complex and challenging. Furthermore, they are well-established within the NLP community and offer ample training data. To obtain task-level SFT scores, we averaged dataset-specific scores within each task. Specifically, CMS included BoolQ (Clark et al., [2019](#bib.bib8)), PIQA (Bisk et al., [2019](#bib.bib4)), HellaSwag (Zellers et al., [2019](#bib.bib51)), Winogrande (Sakaguchi et al., [2021](#bib.bib38)), and OpenBookQA (Mihaylov et al., [2018](#bib.bib28)); RAG utilized NQ (Kwiatkowski et al., [2019](#bib.bib22)), TriviaQA (Joshi et al., [2017](#bib.bib19)), HotpotQA (Yang et al., [2018](#bib.bib49)), and 2Wiki (Ho et al., [2020](#bib.bib16)); and CBQA used NQ (Kwiatkowski et al., [2019](#bib.bib22)) and TriviaQA (Joshi et al., [2017](#bib.bib19)).

### 2.2 Prediction Proxies

This study investigates two distinct prediction proxies: Perplexity (PPL) and k-shot learning (Kshot). Perplexity is a prevalent prediction proxy for monitoring LLM pre-training, whereas the intuitive rationale for k-shot learning lies in its potential correlation with fine-tuned performance on the identical task (Tay et al., [2023a](#bib.bib42); Ahn et al., [2023](#bib.bib2); Von Oswald et al., [2023](#bib.bib45)).

Perplexity (PPL) is calculated through two distinct methods. PPL-CLM represents the conventional causal language modeling perplexity.
Driven by UL2’s (Tay et al., [2023a](#bib.bib42)) demonstration of span corruption’s efficacy in supervised fine-tuning, we present the PPL-SC proxy. This metric is derived from the span corruption methodology, as in T5 (Raffel et al., [2020](#bib.bib37)), and computes perplexity over randomly sampled text spans. Both perplexities are computed on the PILE development set (Gao et al., [2020](#bib.bib11)), with span corruption parameters (0.15,3)(0.15,3) (Raffel et al., [2020](#bib.bib37)).
For the purposes of clarity in presentation, we utilize the inverse of the actual perplexity values, namely, 1Perplexity\frac{1}{\text{Perplexity}}. This transformation aligns with Kshot such that higher proxy values correspond to improved SFT performance. Unless explicitly stated otherwise, PPL-CLM and PPL-SC in this paper refer to these inverted values.
K-shot performance is calculated by averaging the results from evaluating test sets of target datasets for each SFT task. The actual prompts are detailed in Appendix [F](#A6 "Appendix F Prompts ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?").
Akin to Chowdhery et al. ([2023](#bib.bib7)), we use 1 shot for CMS and 5 shots for RAG and CBQA.
This yields five efficient proxy scores for each model: PPL-CLM, PPL-SC, Kshot-CMS, Kshot-RAG, and Kshot-CBQA.

### 2.3 Pairwise Accuracy as a Measure of Predictive Power

We evaluated each pre-trained LLM variant by fine-tuning it on individual target dataset training sets and assessing performance on the corresponding evaluation sets. Task-level scores (SFT-CMS, SFT-RAG, SFT-CBQA) were computed by averaging these dataset results.
Since practical model selection often involves choosing the best from a small candidate pool, our primary analysis focused on evaluating the discriminating power of prediction proxies (like perplexity). To achieve this, we formulated the evaluation as a pairwise prediction task. We generated all 1225 unique pairs from the 50 LLM variants and measured how accurately each proxy could predict which model in a pair would achieve better aggregated task-level SFT performance. This pairwise prediction accuracy is our main metric for proxy effectiveness.

## 3 Predictive Power on SFT Tasks

|  |  |  |  |
| --- | --- | --- | --- |
|  | SFT-CMS | SFT-RAG | SFT-CBQA |
| Conventional Perplexity | |  |  |
| PPL-CLM | .332 | .380 | .354 |
| Individual Prediction Proxies | |  |  |
| PPL-SC | .703 | .622 | .609 |
| Kshot-CMS | .573 | .569 | .525 |
| Kshot-RAG | .696 | .766 | .704 |
| Kshot-CBQA | .437 | .447 | .467 |
| Aggregated Prediction Proxies | | | |
| Combine Five Proxies | .622 | .598 | .564 |
| Analytical Exploration of Headroom Potential | | | |
| PPL-SC + Kshot-RAG | .744 | .696 | .642 |
| PPL-SC + Kshot-RAG - PPL-CLM | .763 | .692 | .635 |

Table 1: Accuracy of Individual vs. Aggregated Proxy Predictors.

#### Accuracy of individual prediction proxies to SFT performance.

Table [1](#S3.T1 "Table 1 ‣ 3 Predictive Power on SFT Tasks ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?") details the pairwise SFT prediction accuracy of various proxy metrics across 50 LLM variants. Conventional perplexity (PPL-CLM) exhibited low accuracy (e.g., 0.3 on SFT-CMS), contrasting sharply with its known correlation strength in scaling studies. The span corruption perplexity (PPL-SC) performed better (>> 0.5 accuracy), consistent with prior findings on span corruption benefits (UL2) (Tay et al., [2023a](#bib.bib42)). Few-shot (k-shot) proxies achieved higher accuracy still, with Kshot-RAG reaching ≈\approx 0.7 on SFT-CMS and SFT-RAG. Despite these improvements, no single proxy proved universally reliable across all tested SFT tasks.

#### Aggregating diverse prediction proxies.

We explore improving prediction by combining normalized proxy scores (details in Table [1](#S3.T1 "Table 1 ‣ 3 Predictive Power on SFT Tasks ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?")). While averaging all five proxies underperforme Kshot-RAG alone, combining PPL-SC and Kshot-RAG matched Kshot-RAG’s performance and surpass PPL-SC.
Despite these improvements, even the best individual or combined proxies yield pairwise error rates around 30%, suggesting inherent task difficulty limits performance. Nevertheless, these simple arithmetic combinations (e.g., PPL-SC + Kshot-RAG - PPL-CLM) demonstrate the potential to outperform individual proxies through effective aggregation.

#### A predictive power case study using varied pre-training objectives.

To understand proxy limitations, we analyzed how well PPL-CLM, PPL-SC, and Kshot-RAG predict relative SFT performance between models differing only in their pre-training objective. We grouped models by objective (CLM, SC, UL2, etc.) and evaluated pairwise prediction accuracy for comparisons between these groups (details in Figure [2](#S3.F2 "Figure 2 ‣ A predictive power case study using varied pre-training objectives. ‣ 3 Predictive Power on SFT Tasks ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?"); Appendix [B](#A2 "Appendix B Proxy Predictive Accuracy ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?") covers data variations).
Confirming earlier results, PPL-SC and Kshot-RAG consistently outperformed PPL-CLM. However, their accuracy depended significantly on two factors: (1) The specific pre-training difference: Proxies better captured large performance gaps caused by different objectives (e.g., SC vs. CLM, often ≥0.6\geq 0.6 accuracy) than smaller variations. (2) The target SFT task: A specific comparison (e.g., SC vs. SC+CLM) could yield low accuracy on one task (SFT-CMS, 0.2) but high accuracy on others (SFT-RAG/SFT-CBQA, ≥0.6\geq 0.6).

!(/html/2504.12491/assets/figures/pointwise_obj_obj_acc.png)

Figure 2: 
Pairwise prediction accuracy for PPL-CLM, PPL-SC, and Kshot-RAG comparing LLMs differing only in pre-training objective, across three SFT tasks (rows) and the three proxies (columns). Each cell indicates average accuracy of pairs where the proxy prediction agreed with the SFT result.

## 4 Learning to Compare

|  |  |  |  |
| --- | --- | --- | --- |
|  | SFT-CMS | SFT-RAG | SFT-CBQA |
| Conventional Perplexity | |  |  |
| PPL-CLM | .306±\pm.081 | .366±\pm.060 | .331±\pm.054 |
| Individual and Aggregated Proxies | |  |  |
| Kshot-RAG | .687±\pm.073 | .724±\pm.047 | .683±\pm.077 |
| Combine Five Proxies | .612±\pm.055 | .585±\pm.051 | .540±\pm.104 |
| Learning To Compare (% Relative to Kshot-RAG) | |  |  |
| Trained on the target task | |  |  |
| Learning-to-compare | .753±\pm.054 (+9.6%) | .727±\pm.039 (+0.4%) | .753±\pm.060 (+10.2%) |
| Trained on the source task | | |  |
| SFT-CMS (Src) | .753±\pm.054 (+9.6%) | .712±\pm.054 (-1.7%) | .707±\pm.057 (+3.3%) |
| SFT-RAG (Src) | .734±\pm.047 (+6.8%) | .727±\pm.039 (+0.4%) | .717±\pm.071 (+5.0%) |
| SFT-CBQA (Src) | .734±\pm.052 (+6.8%) | .718±\pm.050 (-0.1%) | .753±\pm.060 (+10.2%) |

Table 2: 
Pairwise prediction accuracy (mean ±\pm std dev, 20 runs): Unsupervised baselines vs. supervised classifiers on SFT-CMS, SFT-RAG, SFT-CBQA.

Recognizing the complementary strengths of individual proxies amidst their challenges (Section [3](#S3 "3 Predictive Power on SFT Tasks ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?"), Table [1](#S3.T1 "Table 1 ‣ 3 Predictive Power on SFT Tasks ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?"), Figure [2](#S3.F2 "Figure 2 ‣ A predictive power case study using varied pre-training objectives. ‣ 3 Predictive Power on SFT Tasks ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?")), we now explore supervised classifiers to combine these signals for potentially enhanced SFT performance prediction.

### 4.1 Formulation

Given two LLMs mim\_{i} and mjm\_{j}, our goal is to predict which model achieves better downstream SFT performance. We denote the values of the five proxies for each model mim\_{i} as
{Pmik}k∈𝒟\{P\_{m\_{i}}^{k}\}\_{k\in\mathcal{D}}, where 𝒟={PPL-CLM,PPL-SC,Kshot-CMS,Kshot-RAG,Kshot-CBQA}\mathcal{D}=\{\text{PPL-CLM},\text{PPL-SC},\text{Kshot-CMS},\text{Kshot-RAG},\text{Kshot-CBQA}\}.
The learning-to-compare model leverages these proxies by training a binary classifier ff to predict the fine-tuned performance comparison between model pair (mi,mj)(m\_{i},m\_{j}). For each proxy kk, we construct the feature vector:
hk​(pmi,pmj)=[pmik−pmjk,pmik⋅pmjk,pmik,pmjk]∈ℝ4.h\_{k}(p\_{m\_{i}},p\_{m\_{j}})=\left[p^{k}\_{m\_{i}}-p^{k}\_{m\_{j}},\;p^{k}\_{m\_{i}}\cdot p^{k}\_{m\_{j}},\;p^{k}\_{m\_{i}},\;p^{k}\_{m\_{j}}\right]\in\mathbb{R}^{4}.
We concatenate features from all five proxies to form the input and lead to 20 features,
namely,
H​(pmi,pmj)∈ℝ20.H(p\_{m\_{i}},p\_{m\_{j}})\in\mathbb{R}^{20}.
We define the ground-truth label yi​jy\_{ij} as a binary value, where yi​j=1y\_{ij}=1 if LLM mim\_{i} performs better after SFT than mjm\_{j}, and yi​j=0y\_{ij}=0 otherwise.
The classifier is trained by minimizing the binary cross-entropy loss (formulation is provided in Appendix Section [C](#A3 "Appendix C Classifier Implementation Detail ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?")).

### 4.2 Experiment Setup

We implemented the supervised classifier using LightGBM (details on other models in Appendix Section [C](#A3 "Appendix C Classifier Implementation Detail ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?")), training separate models per SFT task (CMS, RAG, CBQA). To ensure robustness, we performed 20 runs, each using a random 60%/40% split of the 50 LLM variants to generate training/testing pairs (splits varied per run). We report mean accuracy and standard deviation over the 20 runs in Table [2](#S4.T2 "Table 2 ‣ 4 Learning to Compare ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?") (middle section), compared against unsupervised baselines including PPL-CLM and Kshot-RAG.

### 4.3 Results

#### Learning-to-compare enhances predictive power beyond the best-performing proxies.

Despite the challenges of constructing prediction proxies, supervised learning significantly enhances predictive performance compared to individual or aggregated proxies. As shown in Table [2](#S4.T2 "Table 2 ‣ 4 Learning to Compare ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?"), LightGBM outperforms the best individual proxy, Kshot-RAG, by a substantial margin on the SFT-CMS and SFT-CBQA tasks, improving predictive power by 10% while maintaining comparable performance on SFT-RAG. This confirms that combining diverse proxies can further boost predictive accuracy.

#### Learning-to-compare generalizes well across different target tasks.

We further assessed LightGBM’s generalization by training on one SFT task (source) and evaluating on others (target), using all five proxies as input. The aim was to determine if a classifier learned for one task could predict performance on different ones. Results (Table [2](#S4.T2 "Table 2 ‣ 4 Learning to Compare ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?"), bottom section) reveal effective generalization: models trained on a source task maintained high predictive accuracy on target tasks, typically performing within 2-3% of classifiers trained directly on the target task. This demonstrates the robustness of the learning-to-compare approach across different SFT domains without significant performance loss.

#### Proxy importance.

We quantify each proxy’s contribution to the LightGBM classifiers by computing their normalized gain-based importance scores, as illustrated in Figure [3](#S4.F3 "Figure 3 ‣ Proxy importance. ‣ 4.3 Results ‣ 4 Learning to Compare ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?") (detailed in Appendix Section [E](#A5 "Appendix E Proxy Normalized Importance Score for LightGBM ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?")). Kshot-RAG consistently emerged as the most influential proxy across the three SFT tasks, showing particular dominance in SFT-RAG and SFT-CBQA. PPL-SC and PPL-CLM represented the next tier of importance; for instance, PPL-SC was second most important for SFT-CMS, while PPL-CLM ranked second for SFT-CBQA. Intriguingly, PPL-CLM contributed more significantly to the LightGBM model’s predictions than Kshot-CMS and Kshot-CBQA, despite possessing lower standalone accuracy (Table [1](#S3.T1 "Table 1 ‣ 3 Predictive Power on SFT Tasks ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?")). Our hypothesis is that the supervised classifier effectively utilizes the strong negative correlation observed between PPL-CLM and SFT task performance.

!(/html/2504.12491/assets/figures/feat-imp-lightGBM.png)

Figure 3: 
Relative influence of proxy metrics in the LTC framework (LightGBM).

## 5 Can Post SFT LLM Performance be Reliably Predicted?

While the learning-to-compare method doubles prediction accuracy over perplexity (Table [2](#S4.T2 "Table 2 ‣ 4 Learning to Compare ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?")), its persistent  25% pairwise error rate limits general applicability. This section analyzes its practical utility.
Analysis shows pairwise prediction accuracy depends heavily on the magnitude of the actual SFT performance difference, proving less reliable for subtle distinctions.
But we demonstrate reliable recall of top models within small candidate sets, suggesting value for initial model filtering.

### 5.1 Impact of Performance Gaps on Prediction Reliability

Predicting the relative performance between two language models is expected to be more reliable when their actual performance levels are significantly different. Conversely, distinguishing between models with similar performances poses a greater challenge. This section investigates how the magnitude of the performance gap between model pairs influences the reliability of our prediction classifiers.

To explore the relationship between performance disparity and classifier accuracy, we first calculated the absolute difference in supervised fine-tuning (SFT) performance for each model pair on the target task. We hypothesized that classification accuracy would correlate positively with the size of this performance gap. For quantitative analysis, we categorized the model pairs into five quantiles based on their true post-SFT performance difference: [0–20%], [20–40%], [40–60%], [60–80%], and [80–100%]. Subsequently, we evaluated and compared the classification accuracy for three predictors—PPL-CLM, Kshot-RAG, and Learning-to-compare—within each quantile. These results are visualized in Figure [4](#S5.F4 "Figure 4 ‣ 5.1 Impact of Performance Gaps on Prediction Reliability ‣ 5 Can Post SFT LLM Performance be Reliably Predicted? ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?").

The findings show that prediction reliability for both Kshot-RAG and the Learning-to-compare predictors indeed improves as the performance gap between models widens. For pairs with minimal performance differences ([0–20%] quantile), where models perform almost identically after fine-tuning, prediction accuracy is low, near chance levels (approximately 0.5). As the absolute performance difference increases, accuracy steadily rises, reaching approximately 0.9 for the most distinct pairs ([80–100%] quantile). This confirms that these classifiers yield more reliable predictions when comparing models that are easier to distinguish. Interestingly, PPL-CLM demonstrates the opposite behavior: its accuracy diminishes as the performance gap increases, further highlighting that conventional perplexity is not a dependable indicator for this prediction scenario. Among the methods tested, the learning-to-compare classifier consistently outperformed both PPL-CLM and Kshot-RAG across the quantiles, showing particular strength on the SFT-CMS and SFT-CBQA tasks.

!(/html/2504.12491/assets/figures/bucket_accs.png)

Figure 4: 
Accuracy comparison of PPL-CLM, Kshot-RAG, and Learning-to-Compare (LTC) on SFT tasks (CMS, RAG, CBQA), grouped into five quantiles by absolute SFT performance difference.

### 5.2 Recall the Best Model from a Small Candidate Set

One key practical use for LLM performance predictors is to identify the most promising models within a group of candidates, which can lead to significant cost savings by reducing the number of models that undergo supervised fine-tuning. To assess our classifier’s effectiveness in this critical application—specifically, its ability to recall the best pre-trained LLMs—we performed a ranking experiment where pairwise comparisons between models were predicted and then aggregated into an overall ranking using Borda Count scoring (detailed in Appendix [D](#A4 "Appendix D Ranking using Borda Count ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?")) Dwork et al. ([2001](#bib.bib10)). Models achieving more pairwise ’wins’ received higher scores, indicating better predicted performance. The evaluation results, presented as top-1 and top-5 recall in Figure [5](#S5.F5 "Figure 5 ‣ 5.2 Recall the Best Model from a Small Candidate Set ‣ 5 Can Post SFT LLM Performance be Reliably Predicted? ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?"), show that our “learning-to-compare” method consistently identified the top-performing LLMs. Impressively, it achieved perfect top-1 recall for the SFT-CMS, SFT-RAG, and SFT-CBQA tasks by focusing on the top 7, 7, and 8 predicted models respectively, demonstrating its effectiveness even when narrowing down a relatively small candidate pool (as few as 8 models). Additionally, the unsupervised Kshot-RAG method showed strong performance, corroborating observations from Section [3](#S3 "3 Predictive Power on SFT Tasks ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?").

!(/html/2504.12491/assets/figures/hits_GT.png)

Figure 5: 
Top-1 (top row) and Top-5 (bottom row) recall comparison at various cutoffs: supervised Learning-to-compare (LTC) vs. unsupervised baselines on SFT-CMS, SFT-RAG, and SFT-CBQA tasks.

## 6 Related Work

LLM pre-training fundamentally shapes capabilities like reasoning (Wei et al., [2022](#bib.bib47); Kojima et al., [2022](#bib.bib21); Zellers et al., [2019](#bib.bib51)), knowledge (Chang et al., [2024](#bib.bib6)), and tool use (Yao et al., [2023](#bib.bib50); Mo et al., [2023](#bib.bib30)). Critical pre-training design choices include the training objective—such as dominant CLM (Brown et al., [2020](#bib.bib5); OpenAI, [2023](#bib.bib31)) for generation, SC (Raffel et al., [2020](#bib.bib37)) which aids fine-tuning (Tay et al., [2023a](#bib.bib42)), or combined UL2-style approaches (Tay et al., [2023a](#bib.bib42), [b](#bib.bib43); Garcia et al., [2023](#bib.bib12)) potentially using PrefixLM (Du et al., [2022](#bib.bib9); Chowdhery et al., [2023](#bib.bib7))—and pre-trained corpus composition, which involves quality curation (Rae et al., [2021](#bib.bib36); Touvron et al., [2023](#bib.bib44)), filtering (Penedo et al., [2023](#bib.bib34); Xia et al., [2024](#bib.bib48)), and source mixing (Weber et al., [2024](#bib.bib46); Shen et al., [2024](#bib.bib39)) to ensure broad coverage and robustness. Given the variety of design options, lightweight methods to predict final performance are highly desirable for efficient model development. This work investigates predictors for supervised fine-tuning outcomes, utilizing systematic variations across several pre-training design factors in our study.

The ability to predict the performance of large language models (LLMs) after fine-tuning has gained significant importance, largely driven by the substantial computational investment required for pre-training. Previous research (Kaplan et al., [2020](#bib.bib20); Hoffmann et al., [2022](#bib.bib17); Henighan et al., [2020](#bib.bib15)) established scaling laws showing that increasing pre-training FLOPs typically reduces perplexity on held-out data, correlating with enhancements in capabilities like chain-of-thought reasoning (Wei et al., [2022](#bib.bib47); Kojima et al., [2022](#bib.bib21)), preference alignment (Ouyang et al., [2022](#bib.bib32); Bai et al., [2022](#bib.bib3)), and multilingual understanding (Chowdhery et al., [2023](#bib.bib7)), suggesting larger models generally yield better downstream performance. Analogous scaling phenomena, where lower perplexity often corresponds to improved outcomes, have also been noted when fine-tuning LLMs for specific applications (Zhang et al., [2024](#bib.bib52); Isik et al., [2025](#bib.bib18)); for instance, Isik et al. ([2025](#bib.bib18)) reported such a correlation for machine translation performance. Nevertheless, the dependability of perplexity as a universal predictor has recently come under scrutiny in certain contexts, particularly for tasks involving long-context generation (Liu et al., [2024b](#bib.bib27)) or many-shot in-context learning (Agarwal et al., [2024](#bib.bib1)), implying it may not be a robust indicator across all downstream tasks.

## 7 Conclusion and Future Directions

This study focused on the challenge of predicting LLM performance after supervised fine-tuning (SFT) using only pre-training indicators, establishing that conventional perplexity is unreliable for this purpose. We approached this as a pairwise classification task, using 1B parameter LLM variants with diverse pre-training configurations. We introduced novel unsupervised (Kshot-RAG, PPL-SC) and supervised (“learning-to-compare”) proxy metrics, which successfully reduced relative performance prediction error by over 50% compared to perplexity. These proxies proved effective for predicting outcomes, particularly between models with large performance gaps, and for identifying top-performing candidates, thereby enabling more efficient LLM development pathways.

Future research could focus on validating the generalizability of these methods across larger model scales, a wider range of tasks, and different fine-tuning techniques.
Furthermore, investigating whether signals from intermediate checkpoints during long pre-training cycles can predict final fine-tuning outcomes represents an important research topic not covered here, especially for very large language models.

## Acknowledgements

I would like to thank my mentor Kai Hui for his invaluable guidance and support throughout this project. I also benefited greatly from the insightful feedback and discussions consistently provided by Honglei Zhuang, Zhen Qin, and Vinh Q. Tran. Additionally, special thanks to Aviel Atias, Vinh Q. Tran, Hamed Zamani, Dana Alon, and Donald Metzler for their thoughtful reviews and suggestions on earlier drafts of this paper.

\nobibliography

\*

## References

* Agarwal et al. (2024)

  R. Agarwal, A. Singh, L. M. Zhang, B. Bohnet, L. Rosias, S. C. Chan, B. Zhang, A. Faust, and H. Larochelle.
  Many-shot in-context learning.
  In *ICML 2024 Workshop on In-Context Learning*, 2024.
  URL <https://openreview.net/forum?id=goi7DFHlqS>.
* Ahn et al. (2023)

  K. Ahn, X. Cheng, H. Daneshmand, and S. Sra.
  Transformers learn to implement preconditioned gradient descent for in-context learning.
  In *Proceedings of the 37th International Conference on Neural Information Processing Systems*, NIPS ’23, Red Hook, NY, USA, 2023. Curran Associates Inc.
* Bai et al. (2022)

  Y. Bai, A. Jones, K. Ndousse, A. Askell, A. Chen, N. Dassarma, D. Drain, S. Fort, D. Ganguli, T. Henighan, N. Joseph, S. Kadavath, J. Kernion, T. Conerly, S. El-Showk, N. Elhage, Z. Hatfield-Dodds, D. Hernandez, T. Hume, S. Johnston, S. Kravec, L. Lovitt, N. Nanda, C. Olsson, D. Amodei, T. B. Brown, J. Clark, S. McCandlish, C. Olah, B. Mann, and J. Kaplan.
  Training a helpful and harmless assistant with reinforcement learning from human feedback.
  *ArXiv*, abs/2204.05862, 2022.
  URL <https://api.semanticscholar.org/CorpusID:248118878>.
* Bisk et al. (2019)

  Y. Bisk, R. Zellers, R. L. Bras, J. Gao, and Y. Choi.
  Piqa: Reasoning about physical commonsense in natural language.
  In *AAAI Conference on Artificial Intelligence*, 2019.
  URL <https://api.semanticscholar.org/CorpusID:208290939>.
* Brown et al. (2020)

  T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei.
  Language models are few-shot learners.
  In *Proceedings of the 34th International Conference on Neural Information Processing Systems*, NIPS ’20, Red Hook, NY, USA, 2020. Curran Associates Inc.
  ISBN 9781713829546.
* Chang et al. (2024)

  H. Chang, J. Park, S. Ye, S. Yang, Y. Seo, D.-S. Chang, and M. Seo.
  How do large language models acquire factual knowledge during pretraining?
  *arXiv preprint arXiv:2406.11813*, 2024.
* Chowdhery et al. (2023)

  A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann, P. Schuh, K. Shi, S. Tsvyashchenko, J. Maynez, A. Rao, P. Barnes, Y. Tay, N. Shazeer, V. Prabhakaran, E. Reif, N. Du, B. Hutchinson, R. Pope, J. Bradbury, J. Austin, M. Isard, G. Gur-Ari, P. Yin, T. Duke, A. Levskaya, S. Ghemawat, S. Dev, H. Michalewski, X. Garcia, V. Misra, K. Robinson, L. Fedus, D. Zhou, D. Ippolito, D. Luan, H. Lim, B. Zoph, A. Spiridonov, R. Sepassi, D. Dohan, S. Agrawal, M. Omernick, A. M. Dai, T. S. Pillai, M. Pellat, A. Lewkowycz, E. Moreira, R. Child, O. Polozov, K. Lee, Z. Zhou, X. Wang, B. Saeta, M. Diaz, O. Firat, M. Catasta, J. Wei, K. Meier-Hellstern, D. Eck, J. Dean, S. Petrov, and N. Fiedel.
  Palm: scaling language modeling with pathways.
  *J. Mach. Learn. Res.*, 24(1), Jan. 2023.
  ISSN 1532-4435.
* Clark et al. (2019)

  C. Clark, K. Lee, M.-W. Chang, T. Kwiatkowski, M. Collins, and K. Toutanova.
  BoolQ: Exploring the surprising difficulty of natural yes/no questions.
  In J. Burstein, C. Doran, and T. Solorio, editors, *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 2924–2936, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.
  [10.18653/v1/N19-1300](https:/doi.org/10.18653/v1/N19-1300).
  URL <https://aclanthology.org/N19-1300/>.
* Du et al. (2022)

  Z. Du, Y. Qian, X. Liu, M. Ding, J. Qiu, Z. Yang, and J. Tang.
  GLM: General language model pretraining with autoregressive blank infilling.
  In S. Muresan, P. Nakov, and A. Villavicencio, editors, *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 320–335, Dublin, Ireland, May 2022. Association for Computational Linguistics.
  [10.18653/v1/2022.acl-long.26](https:/doi.org/10.18653/v1/2022.acl-long.26).
  URL <https://aclanthology.org/2022.acl-long.26/>.
* Dwork et al. (2001)

  C. Dwork, R. Kumar, M. Naor, and D. Sivakumar.
  Rank aggregation methods for the web.
  In *Proceedings of the 10th International Conference on World Wide Web*, WWW ’01, page 613–622, New York, NY, USA, 2001. Association for Computing Machinery.
  ISBN 1581133480.
  [10.1145/371920.372165](https:/doi.org/10.1145/371920.372165).
  URL <https://doi.org/10.1145/371920.372165>.
* Gao et al. (2020)

  L. Gao, S. Biderman, S. Black, L. Golding, T. Hoppe, C. Foster, J. Phang, H. He, A. Thite, N. Nabeshima, S. Presser, and C. Leahy.
  The pile: An 800gb dataset of diverse text for language modeling.
  *ArXiv*, abs/2101.00027, 2020.
  URL <https://api.semanticscholar.org/CorpusID:230435736>.
* Garcia et al. (2023)

  X. Garcia, Y. Bansal, C. Cherry, G. Foster, M. Krikun, M. Johnson, and O. Firat.
  The unreasonable effectiveness of few-shot learning for machine translation.
  In *Proceedings of the 40th International Conference on Machine Learning*, ICML’23. JMLR.org, 2023.
* Google et al. (2024)

  G. T. Google, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, and A. M. D. et al.
  Gemini: A family of highly capable multimodal models, 2024.
  URL <https://arxiv.org/abs/2312.11805>.
* Grattafiori et al. (2024)

  A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, et al.
  The llama 3 herd of models.
  *arXiv preprint arXiv:2407.21783*, 2024.
* Henighan et al. (2020)

  T. Henighan, J. Kaplan, M. Katz, M. Chen, C. Hesse, J. Jackson, H. Jun, T. B. Brown, P. Dhariwal, S. Gray, C. Hallacy, B. Mann, A. Radford, A. Ramesh, N. Ryder, D. M. Ziegler, J. Schulman, D. Amodei, and S. McCandlish.
  Scaling laws for autoregressive generative modeling.
  *ArXiv*, abs/2010.14701, 2020.
  URL <https://api.semanticscholar.org/CorpusID:225094178>.
* Ho et al. (2020)

  X. Ho, A.-K. Duong Nguyen, S. Sugawara, and A. Aizawa.
  Constructing a multi-hop QA dataset for comprehensive evaluation of reasoning steps.
  In D. Scott, N. Bel, and C. Zong, editors, *Proceedings of the 28th International Conference on Computational Linguistics*, pages 6609–6625, Barcelona, Spain (Online), Dec. 2020. International Committee on Computational Linguistics.
  [10.18653/v1/2020.coling-main.580](https:/doi.org/10.18653/v1/2020.coling-main.580).
  URL <https://aclanthology.org/2020.coling-main.580/>.
* Hoffmann et al. (2022)

  J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas, L. A. Hendricks, J. Welbl, A. Clark, T. Hennigan, E. Noland, K. Millican, G. van den Driessche, B. Damoc, A. Guy, S. Osindero, K. Simonyan, E. Elsen, O. Vinyals, J. W. Rae, and L. Sifre.
  Training compute-optimal large language models.
  In *Proceedings of the 36th International Conference on Neural Information Processing Systems*, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc.
  ISBN 9781713871088.
* Isik et al. (2025)

  B. Isik, N. Ponomareva, H. Hazimeh, D. Paparas, S. Vassilvitskii, and S. Koyejo.
  Scaling laws for downstream task performance in machine translation.
  In *The Thirteenth International Conference on Learning Representations*, 2025.
  URL <https://openreview.net/forum?id=vPOMTkmSiu>.
* Joshi et al. (2017)

  M. Joshi, E. Choi, D. Weld, and L. Zettlemoyer.
  TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension.
  In R. Barzilay and M.-Y. Kan, editors, *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1601–1611, Vancouver, Canada, July 2017. Association for Computational Linguistics.
  [10.18653/v1/P17-1147](https:/doi.org/10.18653/v1/P17-1147).
  URL <https://aclanthology.org/P17-1147/>.
* Kaplan et al. (2020)

  J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei.
  Scaling laws for neural language models.
  *ArXiv*, abs/2001.08361, 2020.
  URL <https://api.semanticscholar.org/CorpusID:210861095>.
* Kojima et al. (2022)

  T. Kojima, S. S. Gu, M. Reid, Y. Matsuo, and Y. Iwasawa.
  Large language models are zero-shot reasoners.
  In *Proceedings of the 36th International Conference on Neural Information Processing Systems*, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc.
  ISBN 9781713871088.
* Kwiatkowski et al. (2019)

  T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh, C. Alberti, D. Epstein, I. Polosukhin, J. Devlin, K. Lee, K. Toutanova, L. Jones, M. Kelcey, M.-W. Chang, A. M. Dai, J. Uszkoreit, Q. Le, and S. Petrov.
  Natural questions: A benchmark for question answering research.
  *Transactions of the Association for Computational Linguistics*, 7:452–466, 2019.
  [10.1162/tacl\_a\_00276](https:/doi.org/10.1162/tacl_a_00276).
  URL <https://aclanthology.org/Q19-1026/>.
* Lai et al. (2023)

  J. Lai, W. Gan, J. Wu, Z. Qi, and P. S. Yu.
  Large language models in law: A survey.
  *ArXiv*, abs/2312.03718, 2023.
  URL <https://api.semanticscholar.org/CorpusID:266054920>.
* Lee et al. (2024a)

  J. Lee, Z. Dai, X. Ren, B. Chen, D. Cer, J. R. Cole, K. Hui, M. Boratko, R. Kapadia, W. Ding, Y. Luan, S. M. K. Duddu, G. H. Abrego, W. Shi, N. Gupta, A. Kusupati, P. Jain, S. R. Jonnalagadda, M.-W. Chang, and I. Naim.
  Gecko: Versatile text embeddings distilled from large language models.
  *ArXiv*, abs/2403.20327, 2024a.
  URL <https://api.semanticscholar.org/CorpusID:268793455>.
* Lee et al. (2024b)

  J. Lee, N. Stevens, S. C. Han, and M. Song.
  A survey of large language models in finance (finllms).
  *arXiv preprint arXiv:2402.02315*, 2024b.
* Liu et al. (2024a)

  A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, et al.
  Deepseek-v3 technical report.
  *arXiv preprint arXiv:2412.19437*, 2024a.
* Liu et al. (2024b)

  N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang.
  Lost in the middle: How language models use long contexts.
  *Transactions of the Association for Computational Linguistics*, 12:157–173, 2024b.
  [10.1162/tacl\_a\_00638](https:/doi.org/10.1162/tacl_a_00638).
  URL <https://aclanthology.org/2024.tacl-1.9/>.
* Mihaylov et al. (2018)

  T. Mihaylov, P. Clark, T. Khot, and A. Sabharwal.
  Can a suit of armor conduct electricity? a new dataset for open book question answering.
  In *Conference on Empirical Methods in Natural Language Processing*, 2018.
  URL <https://api.semanticscholar.org/CorpusID:52183757>.
* Min et al. (2022)

  S. Min, X. Lyu, A. Holtzman, M. Artetxe, M. Lewis, H. Hajishirzi, and L. Zettlemoyer.
  Rethinking the role of demonstrations: What makes in-context learning work?
  In Y. Goldberg, Z. Kozareva, and Y. Zhang, editors, *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 11048–11064, Abu Dhabi, United Arab Emirates, Dec. 2022. Association for Computational Linguistics.
  [10.18653/v1/2022.emnlp-main.759](https:/doi.org/10.18653/v1/2022.emnlp-main.759).
  URL <https://aclanthology.org/2022.emnlp-main.759/>.
* Mo et al. (2023)

  F. Mo, K. Mao, Y. Zhu, Y. Wu, K. Huang, and J.-Y. Nie.
  ConvGQR: Generative query reformulation for conversational search.
  In A. Rogers, J. Boyd-Graber, and N. Okazaki, editors, *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 4998–5012, Toronto, Canada, July 2023. Association for Computational Linguistics.
  [10.18653/v1/2023.acl-long.274](https:/doi.org/10.18653/v1/2023.acl-long.274).
  URL <https://aclanthology.org/2023.acl-long.274/>.
* OpenAI (2023)

  OpenAI.
  Gpt-4 technical report, 2023.
  URL <https://arxiv.org/abs/2303.08774>.
* Ouyang et al. (2022)

  L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe.
  Training language models to follow instructions with human feedback.
  In *Proceedings of the 36th International Conference on Neural Information Processing Systems*, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc.
  ISBN 9781713871088.
* Pedregosa et al. (2011)

  F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay.
  Scikit-learn: Machine learning in python.
  *J. Mach. Learn. Res.*, 12(null):2825–2830, Nov. 2011.
  ISSN 1532-4435.
* Penedo et al. (2023)

  G. Penedo, Q. Malartic, D. Hesslow, R. Cojocaru, H. Alobeidli, A. Cappelli, B. Pannier, E. Almazrouei, and J. Launay.
  The refinedweb dataset for falcon llm: outperforming curated corpora with web data only.
  In *Proceedings of the 37th International Conference on Neural Information Processing Systems*, NIPS ’23, Red Hook, NY, USA, 2023. Curran Associates Inc.
* Penedo et al. (2024)

  G. Penedo, H. Kydlíček, A. Lozhkov, M. Mitchell, C. A. Raffel, L. Von Werra, T. Wolf, et al.
  The fineweb datasets: Decanting the web for the finest text data at scale.
  *Advances in Neural Information Processing Systems*, 37:30811–30849, 2024.
* Rae et al. (2021)

  J. W. Rae, S. Borgeaud, T. Cai, K. Millican, J. Hoffmann, F. Song, J. Aslanides, S. Henderson, R. Ring, S. Young, E. Rutherford, T. Hennigan, J. Menick, A. Cassirer, R. Powell, G. van den Driessche, L. A. Hendricks, M. Rauh, P.-S. Huang, A. Glaese, J. Welbl, S. Dathathri, S. Huang, J. Uesato, J. F. J. Mellor, I. Higgins, A. Creswell, N. McAleese, A. Wu, E. Elsen, S. M. Jayakumar, E. Buchatskaya, D. Budden, E. Sutherland, K. Simonyan, M. Paganini, L. Sifre, L. Martens, X. L. Li, A. Kuncoro, A. Nematzadeh, E. Gribovskaya, D. Donato, A. Lazaridou, A. Mensch, J.-B. Lespiau, M. Tsimpoukelli, N. K. Grigorev, D. Fritz, T. Sottiaux, M. Pajarskas, T. Pohlen, Z. Gong, D. Toyama, C. de Masson d’Autume, Y. Li, T. Terzi, V. Mikulik, I. Babuschkin, A. Clark, D. de Las Casas, A. Guy, C. Jones, J. Bradbury, M. G. Johnson, B. A. Hechtman, L. Weidinger, I. Gabriel, W. S. Isaac, E. Lockhart, S. Osindero, L. Rimell, C. Dyer, O. Vinyals, K. W. Ayoub, J. Stanway, L. L. Bennett, D. Hassabis, K. Kavukcuoglu, and G. Irving.
  Scaling language models: Methods, analysis & insights from training gopher.
  *ArXiv*, abs/2112.11446, 2021.
  URL <https://api.semanticscholar.org/CorpusID:245353475>.
* Raffel et al. (2020)

  C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu.
  Exploring the limits of transfer learning with a unified text-to-text transformer.
  *J. Mach. Learn. Res.*, 21(1), Jan. 2020.
  ISSN 1532-4435.
* Sakaguchi et al. (2021)

  K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi.
  Winogrande: an adversarial winograd schema challenge at scale.
  *Commun. ACM*, 64(9):99–106, Aug. 2021.
  ISSN 0001-0782.
  [10.1145/3474381](https:/doi.org/10.1145/3474381).
  URL <https://doi.org/10.1145/3474381>.
* Shen et al. (2024)

  Z. Shen, T. Tao, L. Ma, W. Neiswanger, Z. Liu, H. Wang, B. Tan, J. Hestness, N. Vassilieva, D. Soboleva, and E. Xing.
  Slimpajama-dc: Understanding data combinations for llm training, 2024.
  URL <https://arxiv.org/abs/2309.10818>.
* Singhal et al. (2025)

  K. Singhal, T. Tu, J. Gottweis, R. Sayres, E. Wulczyn, M. Amin, L. Hou, K. Clark, S. R. Pfohl, H. Cole-Lewis, D. Neal, Q. M. Rashid, M. Schaekermann, A. Wang, D. Dash, J. H. Chen, N. H. Shah, S. Lachgar, P. A. Mansfield, S. Prakash, B. Green, E. Dominowska, B. A. y Arcas, N. Tomaev, Y. Liu, R. Wong, C. Semturs, S. S. Mahdavi, J. Barral, D. R. Webster, G. S. Corrado, Y. Matias, S. Azizi, A. Karthikesalingam, and V. Natarajan.
  Toward expert-level medical question answering with large language models.
  *Nature Medicine*, 31:943 – 950, 2025.
  URL <https://api.semanticscholar.org/CorpusID:275427710>.
* Soboleva et al. (2023)

  D. Soboleva, F. Al-Khateeb, R. Myers, J. R. Steeves, J. Hestness, and N. Dey.
  SlimPajama: A 627B token cleaned and deduplicated version of RedPajama.
  <https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama>, 2023.
  URL <https://huggingface.co/datasets/cerebras/SlimPajama-627B>.
* Tay et al. (2023a)

  Y. Tay, M. Dehghani, V. Q. Tran, X. Garcia, J. Wei, X. Wang, H. W. Chung, S. Shakeri, D. Bahri, T. Schuster, H. S. Zheng, D. Zhou, N. Houlsby, and D. Metzler.
  Ul2: Unifying language learning paradigms, 2023a.
  URL <https://arxiv.org/abs/2205.05131>.
* Tay et al. (2023b)

  Y. Tay, J. Wei, H. Chung, V. Tran, D. So, S. Shakeri, X. Garcia, S. Zheng, J. Rao, A. Chowdhery, D. Zhou, D. Metzler, S. Petrov, N. Houlsby, Q. Le, and M. Dehghani.
  Transcending scaling laws with 0.1% extra compute.
  In H. Bouamor, J. Pino, and K. Bali, editors, *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 1471–1486, Singapore, Dec. 2023b. Association for Computational Linguistics.
  [10.18653/v1/2023.emnlp-main.91](https:/doi.org/10.18653/v1/2023.emnlp-main.91).
  URL <https://aclanthology.org/2023.emnlp-main.91/>.
* Touvron et al. (2023)

  H. Touvron, L. Martin, K. R. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. M. Bikel, L. Blecher, C. C. Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. S. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. M. Kloumann, A. V. Korenev, P. S. Koura, M.-A. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. H. M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom.
  Llama 2: Open foundation and fine-tuned chat models.
  *ArXiv*, abs/2307.09288, 2023.
  URL <https://api.semanticscholar.org/CorpusID:259950998>.
* Von Oswald et al. (2023)

  J. Von Oswald, E. Niklasson, E. Randazzo, J. a. Sacramento, A. Mordvintsev, A. Zhmoginov, and M. Vladymyrov.
  Transformers learn in-context by gradient descent.
  In *Proceedings of the 40th International Conference on Machine Learning*, ICML’23. JMLR.org, 2023.
* Weber et al. (2024)

  M. Weber, D. Y. Fu, Q. G. Anthony, Y. Oren, S. Adams, A. Alexandrov, X. Lyu, H. Nguyen, X. Yao, V. Adams, B. Athiwaratkun, R. Chalamala, K. Chen, M. Ryabinin, T. Dao, P. Liang, C. Re, I. Rish, and C. Zhang.
  Redpajama: an open dataset for training large language models.
  In *The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track*, 2024.
  URL <https://openreview.net/forum?id=lnuXaRpwvw>.
* Wei et al. (2022)

  J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. H. Chi, Q. V. Le, and D. Zhou.
  Chain-of-thought prompting elicits reasoning in large language models.
  In *Proceedings of the 36th International Conference on Neural Information Processing Systems*, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc.
  ISBN 9781713871088.
* Xia et al. (2024)

  M. Xia, S. Malladi, S. Gururangan, S. Arora, and D. Chen.
  Less: selecting influential data for targeted instruction tuning.
  In *Proceedings of the 41st International Conference on Machine Learning*, ICML’24. JMLR.org, 2024.
* Yang et al. (2018)

  Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. Cohen, R. Salakhutdinov, and C. D. Manning.
  HotpotQA: A dataset for diverse, explainable multi-hop question answering.
  In E. Riloff, D. Chiang, J. Hockenmaier, and J. Tsujii, editors, *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 2369–2380, Brussels, Belgium, Oct.-Nov. 2018. Association for Computational Linguistics.
  [10.18653/v1/D18-1259](https:/doi.org/10.18653/v1/D18-1259).
  URL <https://aclanthology.org/D18-1259/>.
* Yao et al. (2023)

  S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao.
  React: Synergizing reasoning and acting in language models.
  In *The Eleventh International Conference on Learning Representations*, 2023.
  URL <https://openreview.net/forum?id=WE_vluYUL-X>.
* Zellers et al. (2019)

  R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi.
  HellaSwag: Can a machine really finish your sentence?
  In A. Korhonen, D. Traum, and L. Màrquez, editors, *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 4791–4800, Florence, Italy, July 2019. Association for Computational Linguistics.
  [10.18653/v1/P19-1472](https:/doi.org/10.18653/v1/P19-1472).
  URL <https://aclanthology.org/P19-1472/>.
* Zhang et al. (2024)

  B. Zhang, Z. Liu, C. Cherry, and O. Firat.
  When scaling meets LLM finetuning: The effect of data, model and finetuning method.
  In *The Twelfth International Conference on Learning Representations*, 2024.
  URL <https://openreview.net/forum?id=5HCnKDeTws>.

## Appendix A Pretraining and LLMs

We use SlimPajama Soboleva et al. ([2023](#bib.bib41)) as our pretraining corpus, which consists of data from seven domains. Following Shen et al. ([2024](#bib.bib39)), we apply domain re-weighting to create six dataset variants. The detailed domain proportions for each variant are provided in Table [3](#A1.T3 "Table 3 ‣ Appendix A Pretraining and LLMs ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?").

We pretrain 50 LLMs, each with 1 billion parameters, on 100 billion tokens. Model variants are generated by varying pretraining objectives, dataset composition strategies, and learning rates. The detailed pretraining configuration for each model is provided in Table [4](#A1.T4 "Table 4 ‣ Appendix A Pretraining and LLMs ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?").

|  | Sub Dataset | DC-0 | DC-1 | DC-2 | DC-3 | DC-4 | DC-5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SlimPajama | Commoncrawl | 52.2% | 100.0% | 90.9% | 75.8% | 75.8% | 75.8% |
|  | C4 | 26.7% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
|  | GitHub | 5.2% | 0.0% | 9.1% | 24.2% | 0.0% | 9.1% |
|  | Books | 4.2% | 0.0% | 0.0% | 0.0% | 0.0% | 7.9% |
|  | ArXiv | 4.6% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
|  | Wikipedia | 3.8% | 0.0% | 0.0% | 0.0% | 24.2% | 7.3% |
|  | StackExchange | 3.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

Table 3: six configurations of sub dataset combinations in Slimpajama

| Model ID | Pretrained Objective | Domain Re-weight | LR | Domain Tagging | Length Filtering |
| --- | --- | --- | --- | --- | --- |
| 1 | CLM | DC-0 | 1e-4 | ✗ | ✗ |
| 2 | CLM | DC-0 | 2.5e-4 | ✗ | ✗ |
| 3 | CLM | DC-0 | 5e-4 | ✗ | ✗ |
| 4 | CLM | DC-0 | 7.5e-4 | ✗ | ✗ |
| 5 | CLM | DC-0 | 1e-3 | ✗ | ✗ |
| 6 | SC | DC-0 | 1e-4 | ✗ | ✗ |
| 7 | SC | DC-0 | 2.5e-4 | ✗ | ✗ |
| 8 | SC | DC-0 | 5e-4 | ✗ | ✗ |
| 9 | SC | DC-0 | 7.5e-4 | ✗ | ✗ |
| 10 | SC | DC-0 | 1e-3 | ✗ | ✗ |
| 11 | PLM | DC-0 | 1e-4 | ✗ | ✗ |
| 12 | PLM | DC-0 | 2.5e-4 | ✗ | ✗ |
| 13 | PLM | DC-0 | 5e-4 | ✗ | ✗ |
| 14 | PLM | DC-0 | 7.5e-4 | ✗ | ✗ |
| 15 | PLM | DC-0 | 1e-3 | ✗ | ✗ |
| 16 | SC+CLM | DC-0 | 1e-4 | ✗ | ✗ |
| 17 | SC+CLM | DC-0 | 2.5e-4 | ✗ | ✗ |
| 18 | SC+CLM | DC-0 | 5e-4 | ✗ | ✗ |
| 19 | SC+CLM | DC-0 | 7.5e-4 | ✗ | ✗ |
| 20 | SC+CLM | DC-0 | 1e-3 | ✗ | ✗ |
| 21 | UL2 | DC-0 | 1e-4 | ✗ | ✗ |
| 22 | UL2 | DC-0 | 2.5e-4 | ✗ | ✗ |
| 23 | UL2 | DC-0 | 5e-4 | ✗ | ✗ |
| 24 | UL2 | DC-0 | 7.5e-4 | ✗ | ✗ |
| 25 | UL2 | DC-0 | 1e-3 | ✗ | ✗ |
| 26 | UL2R | DC-0 | 1e-4 | ✗ | ✗ |
| 27 | UL2R | DC-0 | 2.5e-4 | ✗ | ✗ |
| 28 | UL2R | DC-0 | 5e-4 | ✗ | ✗ |
| 29 | UL2R | DC-0 | 7.5e-4 | ✗ | ✗ |
| 30 | UL2R | DC-0 | 1e-3 | ✗ | ✗ |
| 31 | UL2R+CLM | DC-0 | 1e-4 | ✗ | ✗ |
| 32 | UL2R+CLM | DC-0 | 2.5e-4 | ✗ | ✗ |
| 33 | UL2R+CLM | DC-0 | 5e-4 | ✗ | ✗ |
| 34 | UL2R+CLM | DC-0 | 7.5e-4 | ✗ | ✗ |
| 35 | UL2R+CLM | DC-0 | 1e-3 | ✗ | ✗ |
| 36 | CLM | DC-1 | 2.5e-4 | ✗ | ✗ |
| 37 | CLM | DC-2 | 2.5e-4 | ✗ | ✗ |
| 38 | CLM | DC-3 | 2.5e-4 | ✗ | ✗ |
| 39 | CLM | DC-4 | 2.5e-4 | ✗ | ✗ |
| 40 | CLM | DC-5 | 2.5e-4 | ✗ | ✗ |
| 41 | PLM | DC-1 | 2.5e-4 | ✗ | ✗ |
| 42 | PLM | DC-2 | 2.5e-4 | ✗ | ✗ |
| 43 | PLM | DC-3 | 2.5e-4 | ✗ | ✗ |
| 44 | PLM | DC-4 | 2.5e-4 | ✗ | ✗ |
| 45 | PLM | DC-5 | 2.5e-4 | ✗ | ✗ |
| 46 | CLM | DC-0 | 2.5e-4 | ✗ | [25%   75%] |
| 47 | CLM | DC-0 | 2.5e-4 | ✗ | [75%   100%] |
| 48 | CLM | DC-0 | 2.5e-4 | ✓ | ✗ |
| 49 | CLM | DC-0 | 2.5e-4 | ✓ | [25%   75%] |
| 50 | CLM | DC-0 | 2.5e-4 | ✓ | [75%   100%] |

Table 4: Pre-trained configurations of LLMs

## Appendix B Proxy Predictive Accuracy

Similar to Section [3](#S3 "3 Predictive Power on SFT Tasks ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?"), we group the pre-trained LLMs into six categories either based on their domain re-weighting or tagging & length filtering configurations. In both cases, paired models share the same pretraining configurations except for the group-specific factor (domain re-weighting or tagging & length filtering). We compute the predictive accuracy of each proxy on three SFT tasks and report the results in the Figure [6](#A2.F6 "Figure 6 ‣ Appendix B Proxy Predictive Accuracy ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?") and Figure [7](#A2.F7 "Figure 7 ‣ Appendix B Proxy Predictive Accuracy ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?").

!(/html/2504.12491/assets/figures/domain_reweight_acc.png)

Figure 6: Predictive accuracy of PPL-CLM, PPL-SC, and Kshot-RAG in distinguishing the better-performing model between two LLMs with different pre-trained dataset domain re-weighting (other pre-trained configurations fixed). DC-0 to DC-5 referes to different dataset variants, detailed in Table [3](#A1.T3 "Table 3 ‣ Appendix A Pretraining and LLMs ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?").

!(/html/2504.12491/assets/figures/tag_length_filter_acc.png)

Figure 7: Predictive accuracy of PPL-CLM, PPL-SC, and Kshot-RAG in distinguishing the better-performing model between two LLMs with different length & filtering methods (other pre-trained configuration fixed). The naming follows the format of [Tagging]-[Length Filtering]. “Tag” and “NoTag” indicate whether domain tags are added. “All” keeps all examples, “Mid” keeps samples with lengths in the 25–75% quantile range, and “Max” keeps the longest 25% of examples.

## Appendix C Classifier Implementation Detail

|  |  |  |  |
| --- | --- | --- | --- |
|  | SFT-CMS | SFT-RAG | SFT-CBQA |
| Conventional Perplexity | |  |  |
| PPL-CLM | .306±\pm.081 | .366±\pm.060 | .331±\pm.054 |
| Individual and Combined Proxies | |  |  |
| Kshot-RAG | .687±\pm.073 | .724±\pm.047 | .683±\pm.077 |
| Combine Five Proxies | .612±\pm.055 | .585±\pm.051 | .540±\pm.104 |
| Learning To Compare | |  |  |
| Train and Evaluate on the same task | |  |  |
| Logistic Regression | .738±\pm.044 | .688±\pm.054 | .624±\pm.087 |
| Neural Networks | .778±\pm.056 | .691±\pm.055 | .673±\pm.071 |
| LightGBM | .753±\pm.054 | .727±\pm.039 | .753±\pm.060 |
| Train on SRC task | | |  |
| Logistic Regresion |  |  |  |
| SFT-CMS (Src) | .738±\pm.044 | .669±\pm.059 | .636±\pm.060 |
| SFT-RAG (Src) | .724±\pm.074 | .688±\pm.054 | .641±\pm.079 |
| SFT-CBQA (SRC) | .708±\pm.069 | .680±\pm.049 | .624±\pm.087 |
| Neural Networks |  |  |  |
| SFT-CMS (Src) | .778±\pm.056 | .706±\pm.060 | 0.683±\pm.062 |
| SFT-RAG (Src) | .742±\pm.073 | .691±\pm.055 | 0.667±\pm.075 |
| SFT-CBQA (Src) | .748±\pm.067 | .695±\pm.059 | .673±\pm.071 |
| LightGBM |  |  |  |
| SFT-CMS (Src) | .753±\pm.054 | .712±\pm.054 | .707±\pm.057 |
| SFT-RAG (Src) | .734±\pm.047 | .727±\pm.039 | .717±\pm.071 |
| SFT-CBQA (Src) | .734±\pm.052 | .718±\pm.050 | .753±\pm.060 |

Table 5: Performance comparison of unsupervised baselines and supervised classifiers (Logistic Regression, Neural Networks, LightGBM) for predicting SFT-CMS, SFT-RAG, and SFT-CBQA. Results are reported as mean accuracy ±\pm standard deviation over 20 runs.

Loss function: Assuming the LLMs in training set as ℳt​r​a​i​n\mathcal{M}\_{train},
we train the classifier using the binary cross-entropy loss.

|  |  |  |
| --- | --- | --- |
|  | ℒ=1C​∑mi,mj∈ℳt​r​a​i​n​and​i≠j−yi​j​log⁡f​(H​(pmi,pmj))−(1−yi​j)​log⁡(1−f​(H​(pmi,pmj)))\mathcal{L}=\frac{1}{C}\displaystyle\sum\_{m\_{i},m\_{j}\in\mathcal{M}\_{train}\ \text{and}\ i\neq j}-y\_{ij}\log f\left(H(p\_{m\_{i}},p\_{m\_{j}})\right)-(1-y\_{ij})\log\left(1-f\left(H(p\_{m\_{i}},p\_{m\_{j}})\right)\right) |  |

Where CC is the total number of pairs in ℳt​r​a​i​n\mathcal{M}\_{train} equals to |ℳt​r​a​i​n|​(|ℳt​r​a​i​n|−1)2\frac{|\mathcal{M}\_{train}|(|\mathcal{M}\_{train}|-1)}{2}.

We also instantiate the learning-to-compare framework using Logistic Regression and Neural Networks as backbone models. Their performance, compared with unsupervised baselines, is reported in Table [5](#A3.T5 "Table 5 ‣ Appendix C Classifier Implementation Detail ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?").

The implementation details are as follows:
For logistic regression, we use scikit-learn’s (Pedregosa et al., [2011](#bib.bib33)) LogisticRegression with the default lbfgs solver for binary classification. The model applies L2L\_{2} regularization with strength C=1.0C=1.0, fits an intercept, and runs up to 100 iterations. Class weighting is not applied.
For the neural network, we use scikit-learn’s MLPClassifier with two hidden layers of size 32 each and ReLU activation. The model is optimized using the Adam solver and trained for a maximum of 100 iterations. All other hyperparameters are set to their default values.
For LightGBM, we use the LGBMClassifie from the official lightgbm library 111 <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html>. The objective is set to binary with binary\_logloss as the evaluation metric. All other hyperparameters follow the default settings: num\_leaves=31, learning\_rate=0.1, n\_estimators=100, feature\_fraction=1.0, bagging\_fraction=1.0, and no regularization (lambda\_l1=0.0, lambda\_l2=0.0).

## Appendix D Ranking using Borda Count

We adopt a Borda Count-style scoring method Dwork et al. ([2001](#bib.bib10)) to transform the pairwise prediction between models to a global ranking. For each model mim\_{i}, we compute its total score by counting the number of pairwise wins over all other models.

|  |  |  |
| --- | --- | --- |
|  | Score​(mi)=∑j≠i𝟙​(f​(mi,mj)>0.5),\text{Score}(m\_{i})=\sum\_{j\neq i}\mathbb{1}\left(f(m\_{i},m\_{j})>0.5\right), |  |

where f​(mi,mj)f(m\_{i},m\_{j}) denotes the classifier’s predicted probability that mim\_{i} outperforms mjm\_{j}.
𝟙​(⋅)\mathbb{1}(\cdot) is the indicator function.
Finally, models are ranked based on their total scores, with higher scores indicating better predicted fine-tuned performance.

## Appendix E Proxy Normalized Importance Score for LightGBM

We use LightGBM’s gain-based feature importance, which quantifies how much each feature contributes to reducing the model’s loss. Specifically, for each feature ff, the importance is defined as the total reduction in the loss function (binary log-loss in our case) due to splits on that feature across all trees in the ensemble.

Let 𝒯\mathcal{T} denote the set of all decision trees in the trained LightGBM model. For each tree t∈𝒯t\in\mathcal{T} and each split node s∈ts\in t, let fsf\_{s} be the feature used at split ss, and let Δ​ℒ​(s)\Delta\mathcal{L}(s) denote the reduction in the loss function caused by that split.
Then, the gain-based importance for feature ff is computed as:

|  |  |  |
| --- | --- | --- |
|  | Gain​(f)=∑t∈𝒯∑s∈tfs=fΔ​ℒ​(s)\text{Gain}(f)=\sum\_{t\in\mathcal{T}}\sum\_{\begin{subarray}{c}s\in t\\ f\_{s}=f\end{subarray}}\Delta\mathcal{L}(s) |  |

In our setting, we construct a 20-dimensional feature vector H​(pmi,pmj)∈ℝ20H(p\_{m\_{i}},p\_{m\_{j}})\in\mathbb{R}^{20} for each model pair (mi,mj)(m\_{i},m\_{j}) using five proxies, with each proxy contributing four dimensions as defined in:

|  |  |  |
| --- | --- | --- |
|  | hk​(pmi,pmj)=[pmik−pmjk,pmik⋅pmjk,pmik,pmjk]h\_{k}(p\_{m\_{i}},p\_{m\_{j}})=\left[p^{k}\_{m\_{i}}-p^{k}\_{m\_{j}},\;p^{k}\_{m\_{i}}\cdot p^{k}\_{m\_{j}},\;p^{k}\_{m\_{i}},\;p^{k}\_{m\_{j}}\right] |  |

To compute proxy-level importance, we group every four dimensions corresponding to each proxy and sum their individual gain scores:

|  |  |  |
| --- | --- | --- |
|  | Gain​(k)=∑f∈ℱkGain​(f)\text{Gain}(k)=\sum\_{f\in\mathcal{F}\_{k}}\text{Gain}(f) |  |

where ℱk\mathcal{F}\_{k} denotes the set of four features derived from proxy kk.

This aggregation allows us to assess the overall contribution of each proxy to the classifier’s predictions. To facilitate comparison across proxies, we normalize the aggregated importance scores. Specifically, let I​(p)I(p) denote the total importance score for proxy pp (i.e., the sum of importance scores for its four associated features). The normalized importance for proxy pp is computed as:

|  |  |  |
| --- | --- | --- |
|  | I~​(p)=I​(p)∑p′∈𝒫I​(p′)\widetilde{I}(p)=\frac{I(p)}{\sum\_{p^{\prime}\in\mathcal{P}}I(p^{\prime})} |  |

where 𝒫\mathcal{P} is the set of all proxies. This yields a distribution over proxies, where higher values indicate greater influence on the classifier’s decision.

## Appendix F Prompts

The exampled prompts used for Kshot-CMS, Kshot-RAG, and Kshot-CBQA tasks are shown in Figure [10](#A6.F10 "Figure 10 ‣ Appendix F Prompts ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?"), Figure [10](#A6.F10 "Figure 10 ‣ Appendix F Prompts ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?") and Figure [10](#A6.F10 "Figure 10 ‣ Appendix F Prompts ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?") respectively.

You are an expert in commonsense reasoning tasks.
// five in-context examples in total.
Question: do iran and afghanistan speak the same language
Answer: True
…
Question: does canada’s worst driver lose their license
Answer: No
Question: does canada’s worst driver lose their license
Answer:

Figure 8: Prompt used for Kshot-CMS

You are an expert in question answering. I am going to give you five example triples of context, question and answer, in which the context may or may not be relevant to the question. The examples will be written.
// five in-context examples in total.
Context: <Retrieved documents>
Question: who sang the original blinded by the light
Answer: Bruce Springsteen
…
Context: <Retrieved documents>
Question: who played vincent in nanny mcphee and the big bang
Answer: Oscar Steer
Context: <Retrieved documents>
Question: how many episodes are there in dragon ball z
Answer:

Figure 9: Prompt used for Kshot-RAG. For each question, we retrieve the top-1 document as context using the Gecko-1B retriever Lee et al. ([2024a](#bib.bib24)).

You are an expert in question answering. I am going to give you five example of question-answer pairs as the in-context examples first. Your task is to generate a answer given a question.
// five in-context examples in total.
Question: the first life forms to appear on earth were
Answer: putative fossilized microorganisms
…
Question: who made the beavis and butthead theme song
Answer: Mike Judge
Question: what network is showing the monday night football game
Answer:

Figure 10: Prompt used for Kshot-CBQA.

## Appendix G Supervised Finetuned, Perplexity and Kshot Results of LLMs

The all supervised fine-tuned, perplexity and Kshot-learning results are detailed in Table [6](#A7.T6 "Table 6 ‣ Appendix G Supervised Finetuned, Perplexity and Kshot Results of LLMs ‣ Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?").

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Performance after Supervised Fine-tuning | | | Individual Proxies from Pre-Training | | | | |
| Model ID | SFT-CMS | SFT-RAG | SFT-CBQA | PPL-CLM | PPL-SC | Kshot-CMS | Kshot-RAG | Kshot-CBQA |
| 1 | 69.800 | 47.275 | 35.600 | 0.395 | 0.089 | 61.560 | 34.990 | 20.390 |
| 2 | 70.980 | 47.600 | 36.350 | 0.394 | 0.094 | 61.660 | 33.130 | 20.130 |
| 3 | 70.520 | 47.850 | 36.000 | 0.391 | 0.087 | 60.680 | 21.230 | 19.950 |
| 4 | 70.900 | 48.425 | 0.150 | 0.389 | 0.092 | 61.100 | 34.011 | 0.121 |
| 5 | 70.900 | 48.375 | 38.550 | 0.388 | 0.079 | 55.000 | 39.072 | 19.315 |
| 6 | 73.560 | 48.200 | 36.950 | 0.377 | 0.141 | 59.780 | 35.980 | 18.280 |
| 7 | 70.260 | 47.900 | 37.350 | 0.385 | 0.131 | 60.300 | 36.500 | 17.410 |
| 8 | 74.560 | 48.600 | 38.250 | 0.360 | 0.143 | 58.420 | 35.300 | 17.810 |
| 9 | 75.200 | 48.600 | 38.300 | 0.331 | 0.141 | 56.920 | 42.692 | 19.221 |
| 10 | 75.360 | 48.725 | 37.750 | 0.306 | 0.140 | 56.460 | 42.494 | 18.945 |
| 11 | 70.000 | 47.750 | 36.250 | 0.394 | 0.096 | 61.960 | 37.710 | 21.090 |
| 12 | 70.420 | 47.675 | 36.000 | 0.387 | 0.097 | 61.480 | 37.300 | 19.440 |
| 13 | 72.160 | 48.125 | 37.800 | 0.387 | 0.102 | 61.980 | 37.900 | 20.260 |
| 14 | 73.240 | 48.475 | 38.250 | 0.386 | 0.104 | 62.240 | 42.300 | 19.177 |
| 15 | 73.560 | 48.925 | 38.750 | 0.382 | 0.094 | 62.240 | 43.003 | 19.422 |
| 16 | 70.440 | 47.725 | 35.600 | 0.395 | 0.129 | 61.560 | 36.800 | 20.350 |
| 17 | 71.620 | 48.000 | 37.500 | 0.392 | 0.132 | 61.480 | 36.810 | 20.200 |
| 18 | 72.980 | 48.650 | 37.900 | 0.388 | 0.143 | 61.480 | 36.490 | 19.860 |
| 19 | 72.940 | 48.650 | 38.450 | 0.385 | 0.143 | 61.180 | 42.789 | 19.297 |
| 20 | 73.420 | 48.825 | 38.900 | 0.382 | 0.143 | 61.620 | 43.306 | 19.522 |
| 21 | 73.140 | 47.150 | 34.900 | 0.394 | 0.170 | 61.940 | 37.100 | 20.780 |
| 22 | 70.540 | 46.775 | 36.900 | 0.376 | 0.153 | 59.500 | 34.810 | 15.950 |
| 23 | 74.200 | 48.350 | 38.050 | 0.383 | 0.178 | 61.420 | 37.760 | 20.610 |
| 24 | 75.140 | 48.825 | 38.400 | 0.378 | 0.172 | 61.200 | 42.933 | 19.286 |
| 25 | 75.340 | 49.025 | 39.100 | 0.375 | 0.173 | 61.700 | 42.931 | 19.637 |
| 26 | 68.720 | 47.150 | 35.500 | 0.386 | 0.129 | 61.100 | 36.380 | 18.290 |
| 27 | 69.760 | 46.600 | 35.750 | 0.378 | 0.130 | 60.180 | 35.740 | 17.170 |
| 28 | 73.000 | 48.425 | 37.900 | 0.386 | 0.131 | 61.660 | 37.950 | 21.610 |
| 29 | 73.840 | 48.625 | 38.800 | 0.382 | 0.134 | 61.600 | 42.658 | 19.467 |
| 30 | 74.340 | 48.675 | 39.050 | 0.379 | 0.133 | 61.820 | 42.700 | 19.592 |
| 31 | 70.400 | 47.425 | 35.900 | 0.395 | 0.130 | 61.780 | 37.470 | 20.970 |
| 32 | 71.540 | 48.100 | 37.300 | 0.393 | 0.125 | 62.180 | 37.690 | 21.700 |
| 33 | 72.900 | 47.875 | 35.850 | 0.390 | 0.127 | 62.080 | 37.710 | 21.080 |
| 34 | 72.820 | 48.650 | 38.800 | 0.388 | 0.130 | 62.120 | 42.775 | 19.465 |
| 35 | 73.640 | 48.600 | 38.450 | 0.385 | 0.129 | 61.560 | 42.711 | 19.290 |
| 36 | 71.620 | 47.625 | 37.700 | 0.364 | 0.102 | 61.680 | 31.760 | 20.280 |
| 37 | 71.700 | 47.900 | 37.250 | 0.373 | 0.102 | 61.640 | 33.080 | 19.940 |
| 38 | 70.200 | 47.650 | 37.700 | 0.374 | 0.096 | 51.580 | 11.330 | 1.230 |
| 39 | 71.080 | 47.825 | 37.550 | 0.387 | 0.110 | 60.800 | 33.860 | 20.290 |
| 40 | 71.480 | 48.000 | 37.850 | 0.389 | 0.107 | 60.720 | 33.170 | 19.250 |
| 41 | 72.400 | 48.000 | 37.800 | 0.360 | 0.101 | 61.880 | 37.180 | 19.720 |
| 42 | 72.300 | 48.125 | 37.300 | 0.368 | 0.103 | 62.200 | 37.610 | 19.390 |
| 43 | 72.360 | 48.100 | 37.350 | 0.368 | 0.104 | 62.180 | 37.370 | 20.040 |
| 44 | 72.800 | 48.350 | 37.550 | 0.382 | 0.111 | 62.300 | 37.660 | 20.320 |
| 45 | 72.480 | 47.825 | 38.000 | 0.383 | 0.111 | 61.560 | 37.870 | 20.860 |
| 46 | 72.220 | 47.900 | 37.650 | 0.380 | 0.104 | 61.860 | 26.500 | 20.160 |
| 47 | 72.040 | 47.575 | 37.300 | 0.387 | 0.106 | 61.120 | 32.380 | 20.200 |
| 48 | 71.800 | 47.325 | 37.350 | 0.386 | 0.107 | 61.160 | 33.210 | 18.540 |
| 49 | 72.220 | 47.900 | 37.650 | 0.380 | 0.104 | 61.860 | 26.500 | 20.160 |
| 50 | 72.040 | 47.575 | 37.300 | 0.387 | 0.106 | 61.120 | 32.380 | 20.200 |

Table 6: SFT, perplexity and kshot performance for all pretrained LLMs.
