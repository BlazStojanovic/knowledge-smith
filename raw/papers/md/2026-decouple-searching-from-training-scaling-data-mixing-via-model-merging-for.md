---
arxiv: '2602.00747'
authors:
- Shengrui Li
- Fei Zhao
- Kaiyan Zhao
- Jieying Ye
- Haifeng Liu
- Fangcheng Shi
- Zheyong Xie
- Yao Hu
- Shaosheng Cao
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'Decouple Searching from Training: Scaling Data Mixing via Model Merging for
  Large Language Model Pre-training'
url: https://arxiv.org/abs/2602.00747
year: 2026
---

[2602.00747] Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training















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



# Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training

Shengrui Li
  
Fei Zhao
  
Kaiyan Zhao
  
Jieying Ye
  
Haifeng Liu
  
Fangcheng Shi
  
Zheyong Xie
  
Yao Hu
  
Shaosheng Cao

###### Abstract

Determining an effective data mixture is a key factor in Large Language Model (LLM) pre-training, where models must balance general competence with proficiency on hard tasks such as math and code. However, identifying an optimal mixture remains an open challenge, as existing approaches either rely on unreliable tiny-scale proxy experiments or require prohibitively expensive large-scale exploration.
To address this, we propose Decouple Searching from Training Mix (DeMix), a novel framework that leverages model merging to predict optimal data ratios. Instead of training proxy models for every sampled mixture, DeMix trains component models on candidate datasets at scale and derives data mixture proxies via weighted model merging. This paradigm decouples search from training costs, enabling evaluation of unlimited sampled mixtures without extra training burden and thus facilitating better mixture discovery through more search trials.
Extensive experiments demonstrate that DeMix breaks the trade-off between sufficiency, accuracy and efficiency, obtaining the optimal mixture with higher benchmark performance at lower search cost.
Additionally, we release the DeMix Corpora, a comprehensive 22T-token dataset comprising high-quality pre-training data with validated mixtures to facilitate open research. Our
code and DeMix Corpora is available at <https://github.com/Lucius-lsr/DeMix>.

Machine Learning, ICML

## 1 Introduction

Large Language Models (LLMs) have achieved remarkable success across a wide range of domains (Shao et al., [2024](#bib.bib10 "Deepseekmath: pushing the limits of mathematical reasoning in open language models"); Guo et al., [2025](#bib.bib9 "Deepseek-r1: incentivizing reasoning capability in llms via reinforcement learning"); Kimi Team et al., [2025](#bib.bib11 "Kimi k2: open agentic intelligence"); Bai et al., [2025](#bib.bib12 "Qwen2. 5-vl technical report")), largely driven by the massive pre-training (Gemini Team et al., [2023](#bib.bib13 "Gemini: a family of highly capable multimodal models"); Achiam et al., [2023](#bib.bib14 "Gpt-4 technical report")).
Beyond scale alone, the composition of the pre-training corpus plays a critical role in shaping model capabilities (Feng et al., [2024](#bib.bib26 "Maximize your data’s potential: enhancing llm accuracy with two-phase pretraining"); Basant et al., [2025](#bib.bib27 "Nvidia nemotron nano 2: an accurate and efficient hybrid mamba-transformer reasoning model"); Blakeman et al., [2025](#bib.bib28 "Nemotron 3 nano: open, efficient mixture-of-experts hybrid mamba-transformer model for agentic reasoning")).

![Refer to caption](/html/2602.00747/assets/x1.png)

Figure 1: 
Methods such as RegMix and CLIMB require extensive proxies. Scaling each proxy leads to unaffordable overall budget. While our DeMix only require a few component models to merge unlimited training-free proxies.

However, optimizing the data mixture for pre-training remains challenging, as it requires balancing general-purpose language abilities with strong performance on complex tasks such as mathematical reasoning and code generation (Cobbe et al., [2021a](#bib.bib31 "Training verifiers to solve math word problems"); Chen et al., [2021](#bib.bib32 "Evaluating large language models trained on code"); Wei et al., [2022](#bib.bib33 "Emergent abilities of large language models")).

A commonly adopted strategy for data mixture selection is to conduct limited large-scale proxy experiments, where mid-sized models (e.g., 8B) are trained on sampled data mixtures with substantial token budgets (e.g., 100B) (Li et al., [2025](#bib.bib18 "Minimax-01: scaling foundation models with lightning attention"); Nie et al., [2025](#bib.bib34 "Large language diffusion models"); Blakeman et al., [2025](#bib.bib28 "Nemotron 3 nano: open, efficient mixture-of-experts hybrid mamba-transformer model for agentic reasoning")). While such proxies can provide relatively accurate signals, they remain computationally expensive and are insufficient for systematically identifying optimal data mixtures.
In contrast, recent lines of work like RegMix (Liu et al., [2024](#bib.bib35 "Regmix: data mixture as regression for language model pre-training")) and CLIMB (Diao et al., [2025](#bib.bib36 "Climb: clustering-based iterative data mixture bootstrapping for language model pre-training")) aim to fully automate the process of searching the optimal data mixture ratios.
These approaches rely on extensive tiny-scale proxy experiments with smaller models and reduced data budgets to train regression-based predictors that map data mixture to loss or downstream performance.
However, the validity of these lightweight proxies has been increasingly questioned (Allal et al., [2025b](#bib.bib37 "The smol training playbook: the secrets to building world-class llms")).
Due to the substantial capability gap between proxy and target models, these automated strategies often fail to generalize to complex tasks such as math and code.

Moreover, despite the abundance of domain-specific pre-training data, there is a notable lack of benchmarked corpora with validated data mixture ratios that can be directly reused for large-scale pre-training.

To address these challenges, we propose Decouple Searching from Training Mix (DeMix), together with the pre-training dataset DeMix Corpora. Instead of training a large number of proxy models under different data mixture ratios, DeMix decouples mixture search from proxy training by leveraging model merging: component models are merged to synthesize an effectively unlimited number of proxy models at virtually no additional cost.
Specifically, we first train a set of component models at scale, each corresponding to a candidate dataset.
We then construct proxy models via weighted model merging over these trained component models, where the merging weights represent target data mixture ratios.
To evaluate the fidelity of model-merged proxy models, we compare them with reference models that are trained directly on sampled data mixtures with massive tokens. We find that these model-merged proxies exhibit substantially higher ranking consistency with these reference models than proxies obtained through traditional small-scale training.

As shown in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"), DeMix achieves substantially higher proxy accuracy than training-based proxy models under the same limited budget, and reach comparable accuracy with approximately 6×\times less computation budget (200 v.s. 1200). Thus, DeMix simultaneously achieves sufficiency (unlimited proxy models), accuracy (faithful proxy performances) and efficiency (fixed token budgets).

Based on these merged proxy models, we predict the optimal data mixture via regression-based methods and apply it to sample 50B tokens for pre-training a 1.7B model. Through extensive experiments across multiple benchmarks covering general language understanding, mathematical reasoning, and code generation, we observe that DeMix significantly outperforms other state-of-the-art data mixture methods such as Regmix and CLIMB while requiring less computational budget. To summarize, our contributions are as follows:

* •

  We propose DeMix, a framework that efficiently decouples data mixture search from model training by constructing proxy models through weighted merging of component models.
* •

  We demonstrate that model-merging proxies faithfully preserve the performance ordering of reference models trained on real data mixtures, providing a reliable signal for mixture selection.
* •

  We release DeMix Corpora, a 22T high-quality, large-scale dataset with validated mixtures that can be directly used for LLM pre-training.

![Refer to caption](/html/2602.00747/assets/x2.png)


Figure 2: Pipeline for DeMix. After (1) cleaning and categorizing massive data, (2) component models are trained on individual candidate datasets. Instead of large-scale training for every ratio, (3) weighted model merging serves as a computationally efficient proxy to estimate performance for various mixture ratios. Finally, (4) a predictor is trained on the benchmarked proxy models to regress the relationship between mixing ratios and performance, utilizing iterative resampling to converge on the optimal mixture.

## 2 Method

The overall pipeline of DeMix is depicted in Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training") and is structured into four sequential phases. In this section, we first briefly describe how candidate datasets are obtained through rigorous filtering. Second, we detail the preparation of component models. Third, we then present our approach for constructing proxy models and discuss the feasibility of using merged models as proxies in our setting. Finally, we introduce an iterative mixture strategy designed to further enhance performance.

### 2.1 Dataset Preprocessing

We first collect large-scale data from a variety of sources, including general-domain corpora, mathematical datasets, and code collections. We then apply rigorous data cleaning, which consists of deduplication, perplexity filtering, FastText filtering and so on. Finally, we perform data-level evaluation and categorize the cleaned corpus into multiple candidate datasets, each representing a distinct data source or domain for subsequent mixture optimization. More details on data preprocessing can be found in Appendix [A](#A1 "Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").

### 2.2 Component Model Preparation

Given the NN candidate datasets, we can construct NN component models, each trained individually on one of the NN candidate datasets. To ensure the robust performance of these component models, we implement a two-step training protocol. First, all component models are initialized from a shared base model trained from scratch on a general-purpose dataset DbaseD\_{\text{base}}, which provides foundational language capabilities (denoted as the Base Model in Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training")). Second, each component model is further trained on its corresponding domain-specific candidate dataset (e.g., mathematics or code), mixed with general data at a fixed ratio β\beta. This procedure encourages each component model to specialize in its target domain while retaining general language competence, enabling them to effectively serve as building blocks for subsequent model merging and data mixture optimization.

### 2.3 Model Merging as Proxy

We first briefly explain why model merging can be used to construct proxy models instead of training new proxy models for each sampled data mixture.

Building upon the former sections, let Θbase∈ℝd\Theta\_{\text{base}}\in\mathbb{R}^{d} denote the parameter vector of a pre-trained base model. Consider a collection of NN candidate datasets {D1,D2,…,DN}\{D\_{1},D\_{2},\dots,D\_{N}\}, we define a training operator 𝒯:𝒟×ℝd→ℝd\mathcal{T}:\mathcal{D}\times\mathbb{R}^{d}\to\mathbb{R}^{d}, where 𝒯​(D,Θbase)\mathcal{T}(D,\Theta\_{\text{base}}) results in the model parameters after training on dataset distribution DD initialized from Θbase\Theta\_{\text{base}}.
For any dataset DiD\_{i}, the parameter update vector (or weight delta) can be defined as (Yang et al., [2024](#bib.bib56 "Model merging in llms, mllms, and beyond: methods, theories, applications, and opportunities")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​(Di)≜𝒯​(Di,Θbase)−Θbase.\Delta(D\_{i})\triangleq\mathcal{T}(D\_{i},\Theta\_{\text{base}})-\Theta\_{\text{base}}. |  | (1) |

Consequently, the individually trained component model corresponds to parameters Θi=Θbase+Δ​(Di)\Theta\_{i}=\Theta\_{\text{base}}+\Delta(D\_{i}).

The objective of data mixing is to identify an optimal mixture distribution (Xie et al., [2023](#bib.bib47 "Doremi: optimizing data mixtures speeds up language model pretraining")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒟mix=∑i=1Nαi​Di,with ​αi≥0​ and ​∑i=1Nαi=1,\mathcal{D}\_{\text{mix}}=\sum\_{i=1}^{N}\alpha\_{i}D\_{i},\quad\text{with }\alpha\_{i}\geq 0\text{ and }\sum\_{i=1}^{N}\alpha\_{i}=1, |  | (2) |

so that the resulting model parameters:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Θmix=𝒯​(𝒟mix,Θbase).\Theta\_{\text{mix}}=\mathcal{T}(\mathcal{D}\_{\text{mix}},\Theta\_{\text{base}}). |  | (3) |

achieve the best performance across the target benchmarks.

To formalize the connection between data mixing and model merging, we first state the relevant constraints.

In practice, the magnitude of parameter updates Δ​(D)\Delta(D) remains relatively small compared to the initialization scale (Gueta et al., [2023](#bib.bib87 "Knowledge is a region in weight space for fine-tuned language models"); Wu et al., [2025](#bib.bib60 "Shadow-ft: tuning instruct via base")). Formally, for any dataset DD in our context, we define (Wu et al., [2025](#bib.bib60 "Shadow-ft: tuning instruct via base")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | δ=∑|𝒯​(D,Θbase)−Θbase|∑|𝒯​(D,Θbase)|+∑|Θbase|≪1.\delta=\frac{\sum|\mathcal{T}(D,\Theta\_{\text{base}})-\Theta\_{\text{base}}|}{\sum|\mathcal{T}(D,\Theta\_{\text{base}})|+\sum|\Theta\_{\text{base}}|}\ll 1. |  | (4) |

In our experiments, δ\delta is approximately 10%, which satisfies this small-update assumption. Empirical studies have shown that as long as δ≪1\delta\ll 1, the arithmetic sum of weight deltas from models trained on separate datasets closely approximates the weight delta obtained by training on their union (Qin et al., [2022](#bib.bib88 "Exploring mode connectivity for pre-trained language models"); Wu et al., [2025](#bib.bib60 "Shadow-ft: tuning instruct via base"); Lin et al., [2025b](#bib.bib61 "Efficient model development through fine-tuning transfer")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​(Di∪Dj)≈Δ​(Di)+Δ​(Dj),\Delta(D\_{i}\cup D\_{j})\approx\Delta(D\_{i})+\Delta(D\_{j}), |  | (5) |

which indicates that the model parameters trained on a weighted mixture of datasets 𝒟mix=∑αi​Di\mathcal{D}\_{\text{mix}}=\sum\alpha\_{i}D\_{i} can be approximated by the weighted average of parameters ∑i=1Nαi​Θi\sum\_{i=1}^{N}\alpha\_{i}\Theta\_{i} trained on separate datasets.

Based on this proposition, merging the component models {Θi}\{\Theta\_{i}\} at a specific ratio {αi}\{\alpha\_{i}\} can obtain a proxy model for any real Θmix\Theta\_{\text{mix}} trained on {αi​Di}\{\alpha\_{i}D\_{i}\}.
The merged model MmixjM\_{\text{{mix}}}^{j} is calculated as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Mmixj=∑i=1Nαij​Θi,M\_{\text{mix}}^{j}=\sum\_{i=1}^{N}\alpha\_{i}^{j}\Theta\_{i}, |  | (6) |

where Θi\Theta\_{i} represents the component model of the corresponding candidate dataset DiD\_{i}.

### 2.4 Mixture Weight Optimization

With an accurate proxy computed as Mm​i​xj=∑i=1Nαij​ΘiM\_{mix}^{j}=\sum\_{i=1}^{N}\alpha\_{i}^{j}\Theta\_{i}, we can search for the optimal mixture through iterative optimization of the mixture weights. Note that inference on any merged proxy model Mm​i​xjM\_{mix}^{j} can be performed directly, incurring no additional training cost.

We then define the average ranking of the merged model across a suite of benchmarks covering general language understanding, mathematical reasoning, and code generation as the gold-standard evaluation signal. Ranking-based evaluation is chosen for its robustness to scale mismatches and its direct relevance to mixture selection.

Next, we perform the following steps for iterative prediction of the optimal mixture ratio:

* •

  Step 1: Randomly sampling a large set of mixture weights ratios {αij}\{\alpha\_{i}^{j}\} uniformly from the simplex.
* •

  Step 2: For each sampled weight ratio {αij}\{\alpha\_{i}^{j}\}, we construct a proxy model by weighted merging of the component models using Equation [6](#S2.E6 "Equation 6 ‣ 2.3 Model Merging as Proxy ‣ 2 Method ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training") and evaluate it on the benchmark suite to obtain its average ranking score rjr^{j}.
* •

  Step 3: Using the collected pairs (αij,rj)(\alpha\_{i}^{j},r^{j}), we train a predictor ff that maps mixture weights to ranking scores. In practice, we follow Liu et al. ([2024](#bib.bib35 "Regmix: data mixture as regression for language model pre-training")) and adopt LightGBM (Ke et al., [2017](#bib.bib71 "LightGBM: a highly efficient gradient boosting decision tree")) as the regression model.
* •

  Step 4: The trained predictor ff is then used to score a large number of newly sampled mixture ratios. We select the top ratios according to the predicted rjr^{j} and iteratively execute Step 2-4 three times, thereby refining the predictions for the high‑ranking ratios.

After the final iteration, we sample a large number of mixture ratios using the trained predictor and select the top-ranked candidates. The final optimal mixture ratio is computed as the average of these candidates, which defines our final mixed dataset for pre-training.

## 3 Experimental Settings

##### Models

In our experiments, we utilize the Qwen3-1.7B architecture (Yang et al., [2025](#bib.bib15 "Qwen3 technical report")) as the foundational backbone. To maintain fundamental capabilities, before training on the candidate datasets, each component model is trained from scratch on 50 billion tokens of general data.

##### Implementation Details

For the candidate dataset, we consistently set the data mixing ratio β\beta to 0.5 throughout our experiments.
In training the component models, we employ a global batch size of 512 and a sequence length of 8192, with an initial learning rate of 3e-4 optimized via a cosine learning rate schedule that retains a minimum of 20% of the initial learning rate.
For the model merging procedure, we adopt a straightforward yet effective weighted linear merging strategy.
In the iterative prediction process, given a proxy budget of 112, we sample 64, 32, and 16 mixtures in each respective iteration, and subsequently average the top 128 mixtures to derive the final optimal mixture configuration.
For the LightGBM model, we set the learning rate to 0.02 and the number of iterations to 300.
For model evaluation, we utilize the OpenCompass benchmark (OpenCompass Contributors, [2023](#bib.bib80 "Opencompass: a universal evaluation platform for foundation models")).

### 3.1 Benchmark

For evaluation, we adopt ARC-E (Clark et al., [2018](#bib.bib72 "Think you have solved question answering? try arc, the ai2 reasoning challenge")), HellaSwag (Zellers et al., [2019](#bib.bib74 "Hellaswag: can a machine really finish your sentence?")), WinoGrande (Sakaguchi et al., [2021](#bib.bib75 "Winogrande: an adversarial winograd schema challenge at scale")), PIQA (Bisk et al., [2020](#bib.bib73 "Piqa: reasoning about physical commonsense in natural language")), and SIQA (Sap et al., [2019](#bib.bib76 "Socialiqa: commonsense reasoning about social interactions")) as general benchmarks; HumanEval (Chen et al., [2021](#bib.bib32 "Evaluating large language models trained on code")) and MBPP (Austin et al., [2021](#bib.bib77 "Program synthesis with large language models")) as code benchmarks; and GSM8K (Cobbe et al., [2021b](#bib.bib78 "Training verifiers to solve math word problems")) and MATH (Hendrycks et al., [2021](#bib.bib79 "Measuring mathematical problem solving with the math dataset")) as math benchmarks. As shown in Table [1](#S3.T1 "Table 1 ‣ 3.1 Benchmark ‣ 3 Experimental Settings ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"), the cost of benchmarking is negligible compared to the training cost. To facilitate comparison, we convert the GPU hour consumption and find that the cost of one benchmarking run is equivalent to training 0.013B tokens.

Table 1: GPU (H800) hour cost for training and benchmarking. The cost of one benchmarking run is equivalent to training 0.013B tokens.

|  |  |  |
| --- | --- | --- |
| Budget / Proxy | Training | Benchmarking |
| 2B | 46 GPUh | 0.3 GPUh |
| 0.013B | 0.3 GPUh | 0.3 GPUh |

### 3.2 Baselines

We adopt the state-of-the-art methods including RegMix (Liu et al., [2024](#bib.bib35 "Regmix: data mixture as regression for language model pre-training")) and CLIMB (Diao et al., [2025](#bib.bib36 "Climb: clustering-based iterative data mixture bootstrapping for language model pre-training")) as our baselines. We modify most configurations of RegMix and CLIMB—including the model, data, and training budget—to align with our setup. We use 112 proxies by default following the original setting in CLIMB (64+32+16), and we also experiment with other proxy counts. The detailed settings are available in Appendix [B](#A2 "Appendix B Baseline Details. ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"). We exclude earlier inferior methods, including DoReMi (Xie et al., [2023](#bib.bib47 "Doremi: optimizing data mixtures speeds up language model pretraining")) and Rho Loss (Mindermann et al., [2022](#bib.bib81 "Prioritized training on points that are learnable, worth learning, and not yet learnt")), as they depend on evaluation loss instead of proxies.

Table 2: Comparison of training costs and proxy accuracy between DeMix and training-based proxies. †\dagger is the benchmarking cost derived from equal GPU hour. The best scores for DeMix and trained proxy are bolded.

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | Train Cost (B) ↓\downarrow | | | | Spearman’s ρ↑\rho\uparrow | | | | Top 25% Spearman’s ρ↑\rho\uparrow | | | | Capability Recovery ↑\uparrow | | | |
| Total Budget | Pre-Cost | No. Proxies | Budget / Proxy | General | Code | Math | Macro Avg. | General | Code | Math | Macro Avg. | General | Code | Math | Macro Avg. |
| Trained Proxy  (RegMix/CLIMB) | \cellcolororange!15224 | \cellcolororange!15- | \cellcolororange!15112 | \cellcolororange!152 | \cellcolororange!150.12 | \cellcolororange!150.60 | \cellcolororange!150.85 | \cellcolororange!150.53 | \cellcolororange!150.17 | \cellcolororange!150.63 | \cellcolororange!15-0.20 | \cellcolororange!150.20 | \cellcolororange!150.98 | \cellcolororange!150.76 | \cellcolororange!150.57 | \cellcolororange!150.77 |
| 448 | - | 112 | 4 | -0.24 | 0.81 | 0.96 | 0.51 | -0.27 | 0.67 | 0.73 | 0.38 | 0.98 | 0.80 | 0.63 | 0.80 |
| 896 | - | 112 | 8 | 0.41 | 0.77 | 0.97 | 0.71 | 0.40 | 0.43 | 0.57 | 0.47 | 0.99 | 0.83 | 0.69 | 0.83 |
| \cellcolorred!151344 | \cellcolorred!15- | \cellcolorred!15112 | \cellcolorred!1512 | \cellcolorred!150.54 | \cellcolorred!150.94 | \cellcolorred!150.97 | \cellcolorred!150.82 | \cellcolorred!150.03 | \cellcolorred!150.70 | \cellcolorred!150.97 | \cellcolorred!150.57 | \cellcolorred!150.99 | \cellcolorred!150.89 | \cellcolorred!150.74 | \cellcolorred!150.87 |
| DeMix  (Ours) | 15 | 2×\times7 | 112 | 0.01†\dagger | 0.25 | 0.51 | 0.88 | 0.55 | -0.17 | 0.01 | 0.97 | 0.27 | 0.98 | 0.77 | 0.54 | 0.76 |
| 71 | 10×\times7 | 112 | 0.01 | 0.13 | 0.74 | 0.95 | 0.60 | -0.10 | 0.63 | 0.71 | 0.41 | 0.99 | 0.79 | 0.64 | 0.80 |
| \cellcolorgreen!15211 | \cellcolorgreen!1530×\times7 | \cellcolorgreen!15112 | \cellcolorgreen!150.01 | \cellcolorgreen!150.64 | \cellcolorgreen!150.81 | \cellcolorgreen!150.98 | \cellcolorgreen!150.81 | \cellcolorgreen!150.00 | \cellcolorgreen!150.80 | \cellcolorgreen!150.97 | \cellcolorgreen!150.59 | \cellcolorgreen!151.00 | \cellcolorgreen!150.81 | \cellcolorgreen!150.68 | \cellcolorgreen!150.83 |
| 351 | 50×\times7 | 112 | 0.01 | 0.66 | 0.76 | 0.97 | 0.80 | 0.20 | 0.37 | 0.93 | 0.50 | 1.01 | 0.82 | 0.73 | 0.85 |

### 3.3 Evaluation Metrics

We conduct two kinds of experiments to evaluate DeMix: Proxy Consistency and Mixture Quality. Proxy Consistency consists of two metrics: proxy accuracy for validating the ranking consistency of the proxy against reference models, and capability recovery indicates how effectively do proxy models maintain absolute performance. While Mixture Quality assesses the downstream performance of the model trained on final data mixture through benchmark score and rank.

#### 3.3.1 Proxy Consistency

##### Proxy Accuracy

Proxy Accuracy is defined as the measure of ranking consistency between the proxy models and the ground-truth reference models. To evaluate this, we randomly sample 96 mixture ratios and train 96 corresponding reference models on a large-scale corpus of 50B tokens, serving as the performance standard. In parallel, proxy models are constructed via efficient model merging. We calculate the Spearman’s rank correlation coefficient ρ\rho (Spearman, [1961](#bib.bib82 "The proof and measurement of association between two things.")) between the benchmark scores of the proxies and references to quantify their alignment. Additionally, to verify the precision in identifying high-performing mixtures, we report the Spearman’s ρ\rho specifically for the top 25% reference models. A higher ρ\rho indicates that the proxy models can accurately predict the relative performance of data mixtures, validating the effectiveness of our training-free approach.

##### Capability Recovery

To quantitatively assess the extent of performance retention in the merged models, we introduce the Capability Recovery Rate. This metric is defined as the quotient of the average benchmark score of the proxy model to that of the corresponding reference model. By directly comparing absolute performance levels, the Capability Recovery Rate serves as a critical indicator of how effectively the proxy model inherits and upholds the fundamental competencies of the reference model without incurring additional training costs.

#### 3.3.2 Mixture Quality

##### Benchmark Score and Rank

We evaluate the final mixture ratio by training a model on 50B tokens with this mixture and evaluating it on general, math, and code benchmarks, aiming for versatility. Since the benchmark scores vary substantially across different domains, we report relative rankings with respect to the 96 reference models, and use the macro-averaged rank across general language understanding, mathematical reasoning, and code generation benchmarks as the final rank metric.

## 4 Experimental Results

Table 3: Comparison of the optimal mixture performance obtained from different data mixing methods. †\dagger is the benchmarking cost derived from equal GPU hour. The best average rank score across all methods is marked in bold, and the best score for the optimal setting within each method is underlined.

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | Train Cost (B) ↓\downarrow | | | | Benchmarks (%) ↑\uparrow | | | | | | | | | | | | Rank ↓\downarrow | | | |
|  |  |  |  | General | | | | | | Code | | | Math | | |  | | | |
| Total Budget | Pre-Cost | No. Proxies | Budget / Proxy | ARC-E | HellaSwag | PIQA | SIQA | WinoGrande | Avg. | MBPP | HumanEval | Avg. | GSM8K | MATH | Avg. | General | Code | Math | Macro Avg. |
| Uniform | - | - | - | - | 71.34 | 54.33 | 73.18 | 40.52 | 55.67 | 59.01 | 18.50 | 18.19 | 18.34 | 12.50 | 6.74 | 9.62 | 9 | 44 | 57 | 36.67 |
| Heuristic | - | - | - | - | 69.57 | 53.72 | 71.75 | 40.32 | 55.27 | 58.13 | 24.27 | 24.90 | 24.58 | 18.20 | 10.16 | 14.18 | 94 | 5 | 28 | 42.33 |
| RegMix | 224 | - | 112 | 2 | 70.81 | 54.34 | 73.45 | 40.41 | 54.58 | 58.72 | 17.70 | 20.58 | 19.14 | 19.13 | 10.43 | 14.78 | 52 | 41 | 21 | 38.00 |
| 224 | - | 28 | 8 | 70.58 | 54.62 | 73.21 | 40.36 | 55.77 | 58.91 | 16.92 | 23.78 | 20.35 | 14.89 | 7.91 | 11.40 | 22 | 29 | 54 | 35.00 |
| 448 | - | 56 | 8 | 71.43 | 54.55 | 73.54 | 40.76 | 55.63 | 59.18 | 18.33 | 21.85 | 20.09 | 15.30 | 7.96 | 11.63 | 2 | 32 | 50 | 28.00 |
| CLIMB | 224 | - | 112 | 2 | 70.75 | 54.44 | 73.41 | 40.37 | 54.93 | 58.78 | 17.47 | 20.32 | 18.90 | 19.21 | 10.65 | 14.93 | 41 | 42 | 21 | 34.67 |
| 224 | - | 28 | 8 | 71.63 | 54.79 | 73.51 | 40.56 | 55.50 | 59.20 | 15.83 | 24.19 | 20.01 | 13.67 | 7.36 | 10.52 | 2 | 32 | 56 | 30.00 |
| 448 | - | 56 | 8 | 70.31 | 54.19 | 73.22 | 40.40 | 55.57 | 58.74 | 19.53 | 22.66 | 21.10 | 20.53 | 11.61 | 16.07 | 47 | 25 | 11 | 27.67 |
| DeMix  (Ours) | 211 | 30×\times7 | 56 | 0.01†\dagger | 70.37 | 53.43 | 73.21 | 40.52 | 55.29 | 58.56 | 21.63 | 21.95 | 21.79 | 24.43 | 14.10 | 19.26 | 66 | 21 | 1 | 29.33 |
| 211 | 30×\times7 | 112 | 0.01 | 70.81 | 53.71 | 73.26 | 40.34 | 55.96 | 58.81 | 19.73 | 22.56 | 21.15 | 19.75 | 10.13 | 14.94 | 31 | 25 | 21 | 25.67 |
| \cellcolorgreen!15212 | \cellcolorgreen!1530×\times7 | \cellcolorgreen!15224 | \cellcolorgreen!150.01 | \cellcolorgreen!1571.08 | \cellcolorgreen!1553.78 | \cellcolorgreen!1572.94 | \cellcolorgreen!1540.63 | \cellcolorgreen!1555.43 | \cellcolorgreen!1558.77 | \cellcolorgreen!1520.70 | \cellcolorgreen!1524.29 | \cellcolorgreen!1522.49 | \cellcolorgreen!1520.98 | \cellcolorgreen!1510.55 | \cellcolorgreen!1515.76 | \cellcolorgreen!1546 | \cellcolorgreen!1512 | \cellcolorgreen!1514 | \cellcolorgreen!1524.00 |
| 214 | 30×\times7 | 448 | 0.01 | 70.61 | 54.29 | 73.12 | 40.22 | 55.89 | 58.83 | 22.03 | 23.17 | 22.60 | 15.73 | 8.35 | 12.04 | 31 | 10 | 42 | 27.67 |

### 4.1 Proxy Consistency

Table [2](#S3.T2 "Table 2 ‣ 3.2 Baselines ‣ 3 Experimental Settings ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training") compares the cost-effectiveness and proxy accuracy of DeMix against conventional training-based approaches. We fix the number of proxies to 112, aligning with the configuration used in CLIMB. Accordingly, we arrive at the following conclusion:

DeMix provides accurate proxies at significantly lower cost.
The results demonstrate that DeMix produces sufficiently strong and accurate proxies, significantly outperforming training-based baselines under the same computational budget. Specifically, when merging component models trained on 30B tokens, DeMix achieves a macro avg. ρ\rho of 0.81 and top‑25% ρ\rho of 0.59, while consuming a total token budget of only 212B.
In contrast, the training-based approach reaches only 0.53 and 0.20 under a comparable budget. To attain a similar performance level, it requires a prohibitive 1344B tokens, corresponding to a 6.4× increase in cost relative to our method.
Additionally, DeMix maintains a high capability recovery rate (up to 0.85), confirming that weighted model merging serves as a highly reliable and efficient proxy for real data mixtures. Based on these results, we select component models trained with 30B tokens for subsequent experiments due to their efficiency and effectiveness.

### 4.2 Mixture Quality

We present the performance of data mixtures produced by different methods in Table [3](#S4.T3 "Table 3 ‣ 4 Experimental Results ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"), with specific mixture details provided in Appendix [C](#A3 "Appendix C Detailed Mixtures in Experiments. ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"). For the baseline methods RegMix and CLIMB, we evaluate their performance using both 2B and 8B proxy models across a varying number of proxies. We also report results for Uniform and Heuristic strategies to provide essential comparisons against other tuned baselines. The results demonstrate the following:

##### DeMix achieves superior mixture quality with lower training cost.

The total training budget of DeMix remains roughly unchanged as the number of proxies increases. Among all baseline methods, DeMix with 224 merged proxies (the green row) achieves the top performance rank of 24.00. Under a comparable training budget, neither RegMix nor CLIMB delivers comparable results. Specifically, neither 112 2B-trained proxies nor 28 8B-trained proxies surpass DeMix. Moreover, 2B-trained proxies yield inferior performance relative to 8B-trained proxies under the same budget, suggesting small proxy performs poorly for this type of task. When the training budget is scaled up to 448B with 56 8B-trained proxies, both RegMix and CLIMB exhibit performance improvements. Yet DeMix still outperforms these methods with a substantially lower training budget.

##### Scaling the proxy count enhances the mixture quality within a certain range.

For DeMix, the performance shows a continuous upward trend as the proxy count is scaled from 56 to 224, with the rank improving from 29.33 to 24.00. A similar improvement is also observed for the 8B-trained proxy versions of RegMix and CLIMB when the proxy count increases from 28 to 56. Further performance gains can be reasonably anticipated, yet such gains would be accompanied by substantial computational overhead. This phenomenon illustrates that the proxy count plays a critical role in determining mixture quality.
Meanwhile, the rank of DeMix is observed to decline once the proxy count reaches 448, indicating that an excessively large number of proxies may introduce the risk of overfitting noise.

Nevertheless, it is of great importance to explore a broader range of proxies. For instance, expanding the pool of candidate datasets will increase the dimensionality of the mixture ratios, which in turn increases the number of proxies required to conduct an effective search. Therefore, DeMix retains a distinct advantage in reducing the computational burden associated with the search procedure.

### 4.3 Ablation Study

To identify the properties essential for effective DeMix, we conduct ablation studies by varying the merging strategies and the proportion of general data in the candidate dataset. For each configuration, we report the mean macro-average ρ\rho and capability recovery rate across 96 different mixture ratios. These experiments involve component models trained on 30B, 40B, and 50B tokens.

Table 4: Comparison of different merging methods. HP-Free denotes hyperparameter-free. Linear merging serves as a simple and effective method.

|  |  |  |  |
| --- | --- | --- | --- |
| Method | HP-Free | ρ\rho | Capability Recovery |
| Linear | ✓ | 0.787 | 0.845 |
| Multi-SLERP | ✓ | 0.785 | 0.813 |
| Breadcrumbs | ✓ | 0.735 | 0.831 |
| DARE | ✗ | 0.757 | 0.835 |
| DELLA | ✗ | 0.784 | 0.778 |
| TIES | ✗ | 0.783 | 0.786 |

#### 4.3.1 Merging Methods

We compare different merging methods by evaluating the accuracy of their corresponding proxies in Table [4](#S4.T4 "Table 4 ‣ 4.3 Ablation Study ‣ 4 Experimental Results ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"). Results are shown for Linear (Wortsman et al., [2022a](#bib.bib49 "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time")), Multi-SLERP (Goddard et al., [2024](#bib.bib48 "Arcee’s mergekit: a toolkit for merging large language models")), DARE (Yu et al., [2024](#bib.bib83 "Language models are super mario: absorbing abilities from homologous models as a free lunch")), Breadcrumbs (Davari and Belilovsky, [2024](#bib.bib52 "Model breadcrumbs: scaling multi-task model merging with sparse masks")), DELLA (Deep et al., [2024](#bib.bib84 "Della-merging: reducing interference in model merging through magnitude-based sampling")), and TIES (Yadav et al., [2023](#bib.bib51 "Ties-merging: resolving interference when merging models")). Among others, Linear is an intuitive, simple, and hyperparameter-free (HP-free) method that achieves the best capability recovery rate and macro-average ρ\rho.

Table 5: Comparison of performances with different proportion of mixed general data in candidate dataset.

|  |  |  |
| --- | --- | --- |
| General Data Proportion | ρ\rho | Capability Recovery |
| 50% | 0.787 | 0.845 |
| 25% | 0.667 | 0.796 |
| 0% | 0.652 | 0.795 |

#### 4.3.2 Proportion of Candidate Datasets

Prior to training the component model, we incorporate general data into the candidate data at a specific ratio. Table [5](#S4.T5 "Table 5 ‣ 4.3.1 Merging Methods ‣ 4.3 Ablation Study ‣ 4 Experimental Results ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training") presents the impact of different mixing ratios on the final proxy accuracy. When the ratio is reduced to 25%, the proxy accuracy drops significantly to a ρ\rho value of 0.667, with a capacity recovery rate of 0.796. The performance further declines when the ratio decreases to 0%. These results illustrate that the absence or insufficient proportion of general data results in a sharp reduction in both ρ\rho and the capacity recovery rate. This validates the necessity of employing this data regularization technique.

## 5 DeMix Corpora

Table 6: Comparison of public high-quality pre-training datasets.

|  |  |  |  |
| --- | --- | --- | --- |
| Dataset | Multilingual | Math&  Code | Validated  Mixture |
| DCLM-baseline | ✗ | ✗ | ✗ |
| FineWeb-Edu | ✗ | ✗ | ✗ |
| ClimbMix | ✗ | ✗ | ✗ |
| DOLMA-v1.7 | ✗ | ✓ | ✗ |
| SmolLM-Corpus | ✓ | ✓ | ✗ |
| Nemotron-Pretrain | ✓ | ✓ | ✗ |
| DeMix Corpora | ✓ | ✓ | ✓ |

Public pre-training corpora can be categorized into several types. Web-derived general English corpora include FineWeb-Edu (Penedo et al., [2024](#bib.bib20 "The fineweb datasets: decanting the web for the finest text data at scale")), DCLM-baseline (Li et al., [2024](#bib.bib21 "Datacomp-lm: in search of the next generation of training sets for language models")), and ClimbMix (Diao et al., [2025](#bib.bib36 "Climb: clustering-based iterative data mixture bootstrapping for language model pre-training")); composite corpora include SmolLM-Corpus (Ben Allal et al., [2024](#bib.bib86 "SmolLM-corpus")), DOLMA (Soldaini et al., [2024](#bib.bib43 "Dolma: an open corpus of three trillion tokens for language model pretraining research")), and Nemotron-Pretrain (Basant et al., [2025](#bib.bib27 "Nvidia nemotron nano 2: an accurate and efficient hybrid mamba-transformer reasoning model")). In addition, multilingual (Messmer et al., [2025](#bib.bib39 "Enhancing multilingual llm pretraining with model-based data selection"); De Gibert et al., [2024](#bib.bib40 "A new massive multilingual dataset for high-performance language technologies")), code (Li et al., [2023](#bib.bib41 "Starcoder: may the source be with you!")), and mathematics (Allal et al., [2025a](#bib.bib30 "SmolLM2: when smol goes big–data-centric training of a small language model")) corpora are not directly applicable for pre-training. As shown in Table [6](#S5.T6 "Table 6 ‣ 5 DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"), there remains a scarcity of high-quality corpora with validated mixture. In contrast, our proposed DeMix Corpora (15T original tokens and 22T mixture tokens) serves as a comprehensive, high-quality, large-scale, and carefully mixed resource that can be directly employed for pre-training.

![Refer to caption](/html/2602.00747/assets/x3.png)

Figure 3: 
General performance of high-quality general datasets.



![Refer to caption](/html/2602.00747/assets/x4.png)

Figure 4: 
The data mixture constructed using our DeMix framework. The three hierarchical levels from the inside out are domain, data category, and data origin.




Table 7: Multi-Domain performance of mixed datasets after mid-training 50B tokens.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Dataset | General | Code | Math | Avg. Rank↓\downarrow |
| SmolLM-Corpus | 59.13 | 21.01 | 9.14 | 31.33 |
| Nemotron-Pretrain | 57.67 | 28.35 | 16.12 | 36.00 |
| DeMix Corpora | 58.77 | 22.49 | 15.76 | 24.00 |

To ensure high data quality, we curate the corpora from heterogeneous open-source sources, followed by a comprehensive data cleaning pipeline (see Appendix [A](#A1 "Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training")).
As shown in Figure [3](#S5.F3 "Figure 3 ‣ 5 DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"), when training a Qwen3-1.7B model from scratch and evaluating on general benchmarks, the high-quality (HQ) general data subset of the DeMix Corpora outperforms all other general datasets.

In addition, DeMix Corpora not only features large and high-quality general corpus subset, but also adopts a validated optimal data mixture that best balances multi-domain capabilities of pre-training.
The final mixtures are illustrated in Figure [4](#S5.F4 "Figure 4 ‣ 5 DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
We compare DeMix Corpora with existing mixed datasets. The results presented in Table [7](#S5.T7 "Table 7 ‣ 5 DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training") demonstrate that DeMix Corpora achieves a superior balance between general and domain-specific capabilities, yielding the best average rank of 24.00. In contrast, Nemotron-Pretrain exhibits poor general capability due to an insufficient volume of general-domain training data. SmolLM-Corpus features high-quality general data with a large proportion, but its math capability is severely deficient.

## 6 Related Works

### 6.1 Data Mixture

Data mixture plays a critical role in successful LLM pre-training (Chen et al., [2023](#bib.bib44 "Skill-it! a data-driven skills framework for understanding and training language models"); Shen et al., [2023](#bib.bib45 "Slimpajama-dc: understanding data combinations for llm training"); Ye et al., [2024](#bib.bib46 "Data mixing laws: optimizing data mixtures by predicting language modeling performance"); Xie et al., [2023](#bib.bib47 "Doremi: optimizing data mixtures speeds up language model pretraining")). Many LLM teams conduct large-scale proxy experiments using mid-sized models with a sufficient token budget (Li et al., [2025](#bib.bib18 "Minimax-01: scaling foundation models with lightning attention"); Nie et al., [2025](#bib.bib34 "Large language diffusion models"); Blakeman et al., [2025](#bib.bib28 "Nemotron 3 nano: open, efficient mixture-of-experts hybrid mamba-transformer model for agentic reasoning")). While such approaches can yield accurate estimates given massive computational resources, their cost makes them impractical for many research institutes. Recently, researchers have begun to propose automated methods for data mixture optimization. RegMix (Liu et al., [2024](#bib.bib35 "Regmix: data mixture as regression for language model pre-training")) samples a large number of data ratios for tiny-scale proxy experiments and generates optimal predictions based on the results of a regression predictor. CLIMB (Diao et al., [2025](#bib.bib36 "Climb: clustering-based iterative data mixture bootstrapping for language model pre-training")) iteratively samples and calibrates sample points with higher predicted scores, thereby improving the prediction accuracy for high-performing samples. However, such automated methods are only validated to optimize simple general capabilities. When jointly optimizing more challenging tasks such as math and code (Kimi Team et al., [2025](#bib.bib11 "Kimi k2: open agentic intelligence"); Feng et al., [2024](#bib.bib26 "Maximize your data’s potential: enhancing llm accuracy with two-phase pretraining"); Basant et al., [2025](#bib.bib27 "Nvidia nemotron nano 2: an accurate and efficient hybrid mamba-transformer reasoning model")), tiny-scale proxy experiments tend to become inaccurate due to insufficient training (Allal et al., [2025b](#bib.bib37 "The smol training playbook: the secrets to building world-class llms"); Li et al., [2025](#bib.bib18 "Minimax-01: scaling foundation models with lightning attention")).

### 6.2 Model Merging

Model merging has emerged as a promising training-free approach that combines multiple LLMs with identical structures into a new LLM using a series of arithmetic operations. (Goddard et al., [2024](#bib.bib48 "Arcee’s mergekit: a toolkit for merging large language models"); Wortsman et al., [2022a](#bib.bib49 "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time"); Ilharco et al., [2022](#bib.bib50 "Editing models with task arithmetic"); Yadav et al., [2023](#bib.bib51 "Ties-merging: resolving interference when merging models"); Davari and Belilovsky, [2024](#bib.bib52 "Model breadcrumbs: scaling multi-task model merging with sparse masks")). This technique can be used to regularize models (Luo et al., [2025](#bib.bib53 "How learning rate decay wastes your best data in curriculum-based llm pretraining"); Wortsman et al., [2022b](#bib.bib54 "Robust fine-tuning of zero-shot models")), improve generalization (Izmailov et al., [2018](#bib.bib55 "Averaging weights leads to wider optima and better generalization"); Yang et al., [2024](#bib.bib56 "Model merging in llms, mllms, and beyond: methods, theories, applications, and opportunities")), mitigate catastrophic forgetting (Xiao et al., [2024](#bib.bib57 "Lm-cocktail: resilient tuning of language models via model merging")), or even replace fine-tuning entirely (Ahmadian et al., [2024](#bib.bib58 "Mix data or merge models? optimizing for diverse multi-task learning")).
Recent works have been exploring model merging in data selection. For instance, Merge-to-Mix (Tao et al., [2025](#bib.bib59 "Merge to mix: mixing datasets via model merging")) employs unweighted model averaging to enumerate all binary subset choices to discard datasets during fine-tuning. In contrast, optimizing pre-training data mixtures is substantially more challenging: mixing weights are continuous-valued and the feasible space is unbounded.
Recently, several studies have shown that merging weight deltas from models sharing the same base but trained on different datasets is a highly effective alternative to directly merging their training processes, provided that parameter updates remain relatively small (Wu et al., [2025](#bib.bib60 "Shadow-ft: tuning instruct via base"); Lin et al., [2025b](#bib.bib61 "Efficient model development through fine-tuning transfer")). Our work significantly extend this line of work by leveraging weighted model merging as a proxy for real data mixture during pre-training. By ensuring that a single proxy is adequately accurate, we eliminate the overhead of training, thereby enabling extensive sampling to search for the optimal ratio.

## 7 Conclusion

In this work, we introduce DeMix, a pre-training data mixture optimization framework that decouples mixture search from costly proxy training by leveraging weighted model merging. DeMix simultaneously achieves sufficiency
(unlimited proxy models), accuracy (faithful proxy performances)
and efficiency (fixed token budgets). Across comprehensive evaluations, DeMix yields the best data mixture that balances the diverse capability demands of LLM pre-training across general language understanding, mathematical reasoning, and code generation. To further support reproducible research and practical pre-training, we release DeMix Corpora, a 22T-token high-quality dataset accompanied by validated mixture ratios, providing a resource for large-scale LLM pre-training development.

## References

* J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. (2023)
  Gpt-4 technical report.
  arXiv preprint arXiv:2303.08774.
  Cited by: [§A.1](#A1.SS1.SSS0.Px6.p1.1 "Instance-Level Data Labeling ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§1](#S1.p1.1 "1 Introduction ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* A. Ahmadian, S. Goldfarb-Tarrant, B. Ermis, M. Fadaee, S. Hooker, et al. (2024)
  Mix data or merge models? optimizing for diverse multi-task learning.
  arXiv preprint arXiv:2410.10801.
  Cited by: [§6.2](#S6.SS2.p1.1 "6.2 Model Merging ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* L. B. Allal, A. Lozhkov, E. Bakouch, G. M. Blázquez, G. Penedo, L. Tunstall, A. Marafioti, H. Kydlíček, A. P. Lajarín, V. Srivastav, et al. (2025a)
  SmolLM2: when smol goes big–data-centric training of a small language model.
  arXiv preprint arXiv:2502.02737.
  Cited by: [§A.1](#A1.SS1.SSS0.Px8.p1.1 "Candidate Data Preparation ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§A.2](#A1.SS2.p1.1 "A.2 Data Composition Across Stages ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§5](#S5.p1.1 "5 DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* L. B. Allal, L. Tunstall, N. Tazi, E. Bakouch, E. Beeching, C. M. Patiño, C. Fourrier, T. Frere, A. Lozhkov, C. Raffel, L. von Werra, and T. Wolf (2025b)
  The smol training playbook: the secrets to building world-class llms.
  Cited by: [§A.1](#A1.SS1.SSS0.Px8.p1.1 "Candidate Data Preparation ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§1](#S1.p3.1 "1 Introduction ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§6.1](#S6.SS1.p1.1 "6.1 Data Mixture ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry, Q. Le, et al. (2021)
  Program synthesis with large language models.
  arXiv preprint arXiv:2108.07732.
  Cited by: [§3.1](#S3.SS1.p1.1 "3.1 Benchmark ‣ 3 Experimental Settings ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, et al. (2025)
  Qwen2. 5-vl technical report.
  arXiv preprint arXiv:2502.13923.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* P. Bansal and S. Sanghavi (2025)
  Context-free synthetic data mitigates forgetting.
  arXiv preprint arXiv:2505.13811.
  Cited by: [§A.1](#A1.SS1.SSS0.Px8.p1.1 "Candidate Data Preparation ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* A. Basant, A. Khairnar, A. Paithankar, A. Khattar, A. Renduchintala, A. Malte, A. Bercovich, A. Hazare, A. Rico, A. Ficek, et al. (2025)
  Nvidia nemotron nano 2: an accurate and efficient hybrid mamba-transformer reasoning model.
  arXiv preprint arXiv:2508.14444.
  Cited by: [§A.1](#A1.SS1.SSS0.Px1.p1.1 "Data Collection ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§A.1](#A1.SS1.SSS0.Px8.p1.1 "Candidate Data Preparation ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§A.2](#A1.SS2.p1.1 "A.2 Data Composition Across Stages ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§1](#S1.p1.1 "1 Introduction ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§5](#S5.p1.1 "5 DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§6.1](#S6.SS1.p1.1 "6.1 Data Mixture ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* L. Ben Allal, A. Lozhkov, G. Penedo, T. Wolf, and L. von Werra (2024)
  SmolLM-corpus
  External Links: [Link](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)
  Cited by: [§5](#S5.p1.1 "5 DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* Y. Bisk, R. Zellers, J. Gao, Y. Choi, et al. (2020)
  Piqa: reasoning about physical commonsense in natural language.
  In Proceedings of the AAAI conference on artificial intelligence,
  Vol. 34,  pp. 7432–7439.
  Cited by: [§3.1](#S3.SS1.p1.1 "3.1 Benchmark ‣ 3 Experimental Settings ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* A. Blakeman, A. Grattafiori, A. Basant, A. Gupta, A. Khattar, A. Renduchintala, A. Vavre, A. Shukla, A. Bercovich, A. Ficek, et al. (2025)
  Nemotron 3 nano: open, efficient mixture-of-experts hybrid mamba-transformer model for agentic reasoning.
  arXiv preprint arXiv:2512.20848.
  Cited by: [§A.1](#A1.SS1.SSS0.Px8.p1.1 "Candidate Data Preparation ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§A.2](#A1.SS2.p1.1 "A.2 Data Composition Across Stages ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§1](#S1.p1.1 "1 Introduction ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§1](#S1.p3.1 "1 Introduction ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§6.1](#S6.SS1.p1.1 "6.1 Data Mixture ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba (2021)
  Evaluating large language models trained on code.
  External Links: 2107.03374
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§3.1](#S3.SS1.p1.1 "3.1 Benchmark ‣ 3 Experimental Settings ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* M. Chen, N. Roberts, K. Bhatia, J. Wang, C. Zhang, F. Sala, and C. Ré (2023)
  Skill-it! a data-driven skills framework for understanding and training language models.
  Advances in Neural Information Processing Systems 36,  pp. 36000–36040.
  Cited by: [§6.1](#S6.SS1.p1.1 "6.1 Data Mixture ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord (2018)
  Think you have solved question answering? try arc, the ai2 reasoning challenge.
  arXiv preprint arXiv:1803.05457.
  Cited by: [§3.1](#S3.SS1.p1.1 "3.1 Benchmark ‣ 3 Experimental Settings ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman (2021a)
  Training verifiers to solve math word problems.
  arXiv preprint arXiv:2110.14168.
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al. (2021b)
  Training verifiers to solve math word problems.
  arXiv preprint arXiv:2110.14168.
  Cited by: [§3.1](#S3.SS1.p1.1 "3.1 Benchmark ‣ 3 Experimental Settings ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen, et al. (2025)
  Gemini 2.5: pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities.
  arXiv preprint arXiv:2507.06261.
  Cited by: [§A.1](#A1.SS1.SSS0.Px6.p1.1 "Instance-Level Data Labeling ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* M. Davari and E. Belilovsky (2024)
  Model breadcrumbs: scaling multi-task model merging with sparse masks.
  In European Conference on Computer Vision,
   pp. 270–287.
  Cited by: [§4.3.1](#S4.SS3.SSS1.p1.1 "4.3.1 Merging Methods ‣ 4.3 Ablation Study ‣ 4 Experimental Results ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§6.2](#S6.SS2.p1.1 "6.2 Model Merging ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* O. De Gibert, G. Nail, N. Arefyev, M. Bañón, J. Van Der Linde, S. Ji, J. Zaragoza-Bernabeu, M. Aulamo, G. Ramírez-Sánchez, A. Kutuzov, et al. (2024)
  A new massive multilingual dataset for high-performance language technologies.
  arXiv preprint arXiv:2403.14009.
  Cited by: [§5](#S5.p1.1 "5 DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* P. T. Deep, R. Bhardwaj, and S. Poria (2024)
  Della-merging: reducing interference in model merging through magnitude-based sampling.
  arXiv preprint arXiv:2406.11617.
  Cited by: [§4.3.1](#S4.SS3.SSS1.p1.1 "4.3.1 Merging Methods ‣ 4.3 Ablation Study ‣ 4 Experimental Results ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* S. Diao, Y. Yang, Y. Fu, X. Dong, D. Su, M. Kliegl, Z. Chen, P. Belcak, Y. Suhara, H. Yin, et al. (2025)
  Climb: clustering-based iterative data mixture bootstrapping for language model pre-training.
  arXiv preprint arXiv:2504.13161.
  Cited by: [§A.1](#A1.SS1.SSS0.Px8.p1.1 "Candidate Data Preparation ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§1](#S1.p3.1 "1 Introduction ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§3.2](#S3.SS2.p1.1 "3.2 Baselines ‣ 3 Experimental Settings ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§5](#S5.p1.1 "5 DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§6.1](#S6.SS1.p1.1 "6.1 Data Mixture ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* A. Fan, Y. Jernite, E. Perez, D. Grangier, J. Weston, and M. Auli (2019)
  ELI5: long form question answering.
  arXiv preprint arXiv:1907.09190.
  Cited by: [§A.1](#A1.SS1.SSS0.Px4.p1.1 "FastText Filtering ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* S. Feng, S. Prabhumoye, K. Kong, D. Su, M. Patwary, M. Shoeybi, and B. Catanzaro (2024)
  Maximize your data’s potential: enhancing llm accuracy with two-phase pretraining.
  arXiv preprint arXiv:2412.15285.
  Cited by: [§A.1](#A1.SS1.SSS0.Px8.p1.1 "Candidate Data Preparation ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§A.2](#A1.SS2.p1.1 "A.2 Data Composition Across Stages ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§1](#S1.p1.1 "1 Introduction ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§6.1](#S6.SS1.p1.1 "6.1 Data Mixture ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* K. Fujii, Y. Tajima, S. Mizuki, H. Shimada, T. Shiotani, K. Saito, M. Ohi, M. Kawamura, T. Nakamura, T. Okamoto, et al. (2025)
  Rewriting pre-training data boosts llm performance in math and code.
  arXiv preprint arXiv:2505.02881.
  Cited by: [§A.1](#A1.SS1.SSS0.Px1.p1.1 "Data Collection ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* Gemini Team, R. Anil, S. Borgeaud, J. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al. (2023)
  Gemini: a family of highly capable multimodal models.
  arXiv preprint arXiv:2312.11805.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* C. Goddard, S. Siriwardhana, M. Ehghaghi, L. Meyers, V. Karpukhin, B. Benedict, M. McQuade, and J. Solawetz (2024)
  Arcee’s mergekit: a toolkit for merging large language models.
  arXiv preprint arXiv:2403.13257.
  Cited by: [§4.3.1](#S4.SS3.SSS1.p1.1 "4.3.1 Merging Methods ‣ 4.3 Ablation Study ‣ 4 Experimental Results ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§6.2](#S6.SS2.p1.1 "6.2 Model Merging ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* A. Gueta, E. Venezian, C. Raffel, N. Slonim, Y. Katz, and L. Choshen (2023)
  Knowledge is a region in weight space for fine-tuned language models.
  In Findings of the Association for Computational Linguistics: EMNLP 2023, H. Bouamor, J. Pino, and K. Bali (Eds.),
  Singapore,  pp. 1350–1370.
  External Links: [Link](https://aclanthology.org/2023.findings-emnlp.95/),
  [Document](https://dx.doi.org/10.18653/v1/2023.findings-emnlp.95)
  Cited by: [§2.3](#S2.SS3.p5.2 "2.3 Model Merging as Proxy ‣ 2 Method ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. (2025)
  Deepseek-r1: incentivizing reasoning capability in llms via reinforcement learning.
  arXiv preprint arXiv:2501.12948.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt (2021)
  Measuring mathematical problem solving with the math dataset.
  arXiv preprint arXiv:2103.03874.
  Cited by: [§3.1](#S3.SS1.p1.1 "3.1 Benchmark ‣ 3 Experimental Settings ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* S. Huang, T. Cheng, J. K. Liu, W. Xu, J. Hao, L. Song, Y. Xu, J. Yang, J. Liu, C. Zhang, et al. (2025)
  Opencoder: the open cookbook for top-tier code large language models.
  In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
   pp. 33167–33193.
  Cited by: [§A.1](#A1.SS1.SSS0.Px1.p1.1 "Data Collection ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* G. Ilharco, M. T. Ribeiro, M. Wortsman, S. Gururangan, L. Schmidt, H. Hajishirzi, and A. Farhadi (2022)
  Editing models with task arithmetic.
  arXiv preprint arXiv:2212.04089.
  Cited by: [§6.2](#S6.SS2.p1.1 "6.2 Model Merging ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* P. Izmailov, D. Podoprikhin, T. Garipov, D. Vetrov, and A. G. Wilson (2018)
  Averaging weights leads to wider optima and better generalization.
  arXiv preprint arXiv:1803.05407.
  Cited by: [§6.2](#S6.SS2.p1.1 "6.2 Model Merging ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T. Liu (2017)
  LightGBM: a highly efficient gradient boosting decision tree.
  In Proceedings of the 31st International Conference on Neural Information Processing Systems,
  NIPS’17, Red Hook, NY, USA,  pp. 3149–3157.
  External Links: ISBN 9781510860964
  Cited by: [3rd item](#S2.I1.i3.p1.2 "In 2.4 Mixture Weight Optimization ‣ 2 Method ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* Kimi Team, Y. Bai, Y. Bao, G. Chen, J. Chen, N. Chen, R. Chen, Y. Chen, Y. Chen, Y. Chen, et al. (2025)
  Kimi k2: open agentic intelligence.
  arXiv preprint arXiv:2507.20534.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§6.1](#S6.SS1.p1.1 "6.1 Data Mixture ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* K. Lee, D. Ippolito, A. Nystrom, C. Zhang, D. Eck, C. Callison-Burch, and N. Carlini (2022)
  Deduplicating training data makes language models better.
  In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
   pp. 8424–8445.
  Cited by: [§A.1](#A1.SS1.SSS0.Px2.p1.1 "Deduplication ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* A. Li, B. Gong, B. Yang, B. Shan, C. Liu, C. Zhu, C. Zhang, C. Guo, D. Chen, D. Li, et al. (2025)
  Minimax-01: scaling foundation models with lightning attention.
  arXiv preprint arXiv:2501.08313.
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§6.1](#S6.SS1.p1.1 "6.1 Data Mixture ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* J. Li, A. Fang, G. Smyrnis, M. Ivgi, M. Jordan, S. Y. Gadre, H. Bansal, E. Guha, S. S. Keh, K. Arora, et al. (2024)
  Datacomp-lm: in search of the next generation of training sets for language models.
  Advances in Neural Information Processing Systems 37,  pp. 14200–14282.
  Cited by: [§A.1](#A1.SS1.SSS0.Px1.p1.1 "Data Collection ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§5](#S5.p1.1 "5 DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* R. Li, L. B. Allal, Y. Zi, N. Muennighoff, D. Kocetkov, C. Mou, M. Marone, C. Akiki, J. Li, J. Chim, et al. (2023)
  Starcoder: may the source be with you!.
  arXiv preprint arXiv:2305.06161.
  Cited by: [§5](#S5.p1.1 "5 DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* J. Lin, T. Wang, and K. Qian (2025a)
  Rec-r1: bridging generative large language models and user-centric recommendation systems via reinforcement learning.
  arXiv preprint arXiv:2503.24289.
  Cited by: [§A.1](#A1.SS1.SSS0.Px8.p1.1 "Candidate Data Preparation ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* P. Lin, R. Balasubramanian, F. Liu, N. Kandpal, and T. Vu (2025b)
  Efficient model development through fine-tuning transfer.
  arXiv preprint arXiv:2503.20110.
  Cited by: [§2.3](#S2.SS3.p6.2 "2.3 Model Merging as Proxy ‣ 2 Method ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§6.2](#S6.SS2.p1.1 "6.2 Model Merging ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* Q. Liu, X. Zheng, N. Muennighoff, G. Zeng, L. Dou, T. Pang, J. Jiang, and M. Lin (2024)
  Regmix: data mixture as regression for language model pre-training.
  arXiv preprint arXiv:2407.01492.
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [3rd item](#S2.I1.i3.p1.2 "In 2.4 Mixture Weight Optimization ‣ 2 Method ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§3.2](#S3.SS2.p1.1 "3.2 Baselines ‣ 3 Experimental Settings ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§6.1](#S6.SS1.p1.1 "6.1 Data Mixture ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* Y. Liu, R. Jin, L. Shi, Z. Yao, and D. Xiong (2025)
  Finemath: a fine-grained mathematical evaluation benchmark for chinese large language models.
  ACM Transactions on Asian and Low-Resource Language Information Processing 24 (12),  pp. 1–15.
  Cited by: [§A.1](#A1.SS1.SSS0.Px1.p1.1 "Data Collection ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* K. Luo, Z. Sun, H. Wen, X. Shi, J. Cui, C. Dang, K. Lyu, and W. Chen (2025)
  How learning rate decay wastes your best data in curriculum-based llm pretraining.
  arXiv preprint arXiv:2511.18903.
  Cited by: [§6.2](#S6.SS2.p1.1 "6.2 Model Merging ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* B. Messmer, V. Sabolčec, and M. Jaggi (2025)
  Enhancing multilingual llm pretraining with model-based data selection.
  arXiv.
  External Links: [Link](https://arxiv.org/abs/2502.10361)
  Cited by: [§5](#S5.p1.1 "5 DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* S. Mindermann, J. M. Brauner, M. T. Razzak, M. Sharma, A. Kirsch, W. Xu, B. Höltgen, A. N. Gomez, A. Morisot, S. Farquhar, et al. (2022)
  Prioritized training on points that are learnable, worth learning, and not yet learnt.
  In International Conference on Machine Learning,
   pp. 15630–15649.
  Cited by: [§3.2](#S3.SS2.p1.1 "3.2 Baselines ‣ 3 Experimental Settings ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* S. Nie, F. Zhu, Z. You, X. Zhang, J. Ou, J. Hu, J. Zhou, Y. Lin, J. Wen, and C. Li (2025)
  Large language diffusion models.
  arXiv preprint arXiv:2502.09992.
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§6.1](#S6.SS1.p1.1 "6.1 Data Mixture ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* OpenCompass Contributors (2023)
  Opencompass: a universal evaluation platform for foundation models.
  Cited by: [§3](#S3.SS0.SSS0.Px2.p1.1 "Implementation Details ‣ 3 Experimental Settings ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* G. Penedo, H. Kydlíček, A. Lozhkov, M. Mitchell, C. A. Raffel, L. Von Werra, T. Wolf, et al. (2024)
  The fineweb datasets: decanting the web for the finest text data at scale.
  Advances in Neural Information Processing Systems 37,  pp. 30811–30849.
  Cited by: [§A.1](#A1.SS1.SSS0.Px1.p1.1 "Data Collection ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§A.1](#A1.SS1.SSS0.Px2.p1.1 "Deduplication ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§A.1](#A1.SS1.SSS0.Px5.p1.1 "Quality Filtering for Chinese General-Domain Corpora ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§5](#S5.p1.1 "5 DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* G. Penedo, H. Kydlíček, V. Sabolčec, B. Messmer, N. Foroutan, A. H. Kargaran, C. Raffel, M. Jaggi, L. Von Werra, and T. Wolf (2025)
  FineWeb2: one pipeline to scale them all–adapting pre-training data processing to every language.
  arXiv preprint arXiv:2506.20920.
  Cited by: [§A.1](#A1.SS1.SSS0.Px2.p1.1 "Deduplication ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* G. Penedo, Q. Malartic, D. Hesslow, R. Cojocaru, A. Cappelli, H. Alobeidli, B. Pannier, E. Almazrouei, and J. Launay (2023)
  The refinedweb dataset for falcon llm: outperforming curated corpora with web data, and web data only.
  arXiv preprint arXiv:2306.01116.
  Cited by: [§A.1](#A1.SS1.SSS0.Px4.p1.1 "FastText Filtering ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* Y. Qin, C. Qian, J. Yi, W. Chen, Y. Lin, X. Han, Z. Liu, M. Sun, and J. Zhou (2022)
  Exploring mode connectivity for pre-trained language models.
  In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, Y. Goldberg, Z. Kozareva, and Y. Zhang (Eds.),
  Abu Dhabi, United Arab Emirates,  pp. 6726–6746.
  External Links: [Link](https://aclanthology.org/2022.emnlp-main.451/),
  [Document](https://dx.doi.org/10.18653/v1/2022.emnlp-main.451)
  Cited by: [§2.3](#S2.SS3.p6.2 "2.3 Model Merging as Proxy ‣ 2 Method ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi (2021)
  Winogrande: an adversarial winograd schema challenge at scale.
  Communications of the ACM 64 (9),  pp. 99–106.
  Cited by: [§3.1](#S3.SS1.p1.1 "3.1 Benchmark ‣ 3 Experimental Settings ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* M. Sap, H. Rashkin, D. Chen, R. LeBras, and Y. Choi (2019)
  Socialiqa: commonsense reasoning about social interactions.
  arXiv preprint arXiv:1904.09728.
  Cited by: [§3.1](#S3.SS1.p1.1 "3.1 Benchmark ‣ 3 Experimental Settings ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. (2024)
  Deepseekmath: pushing the limits of mathematical reasoning in open language models.
  arXiv preprint arXiv:2402.03300.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* Z. Shen, T. Tao, L. Ma, W. Neiswanger, Z. Liu, H. Wang, B. Tan, J. Hestness, N. Vassilieva, D. Soboleva, et al. (2023)
  Slimpajama-dc: understanding data combinations for llm training.
  arXiv preprint arXiv:2309.10818.
  Cited by: [§6.1](#S6.SS1.p1.1 "6.1 Data Mixture ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* L. Soldaini, R. Kinney, A. Bhagia, D. Schwenk, D. Atkinson, R. Authur, B. Bogin, K. Chandu, J. Dumas, Y. Elazar, V. Hofmann, A. H. Jha, S. Kumar, L. Lucy, X. Lyu, N. Lambert, I. Magnusson, J. Morrison, N. Muennighoff, A. Naik, C. Nam, M. E. Peters, A. Ravichander, K. Richardson, Z. Shen, E. Strubell, N. Subramani, O. Tafjord, P. Walsh, L. Zettlemoyer, N. A. Smith, H. Hajishirzi, I. Beltagy, D. Groeneveld, J. Dodge, and K. Lo (2024)
  Dolma: an open corpus of three trillion tokens for language model pretraining research.
  arXiv preprint.
  Cited by: [§A.1](#A1.SS1.SSS0.Px1.p1.1 "Data Collection ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§5](#S5.p1.1 "5 DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* C. Spearman (1961)
  The proof and measurement of association between two things..
  Cited by: [§3.3.1](#S3.SS3.SSS1.Px1.p1.3 "Proxy Accuracy ‣ 3.3.1 Proxy Consistency ‣ 3.3 Evaluation Metrics ‣ 3 Experimental Settings ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* Z. S. Tao, K. Vinken, H. Yeh, A. Cooper, and X. Boix (2025)
  Merge to mix: mixing datasets via model merging.
  arXiv preprint arXiv:2505.16066.
  Cited by: [§6.2](#S6.SS2.p1.1 "6.2 Model Merging ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* Teknium (2023)
  OpenHermes 2.5: an open dataset of synthetic data for generalist llm assistants.
   HuggingFace.
  External Links: [Link](https://huggingface.co/datasets/teknium/OpenHermes-2.5)
  Cited by: [§A.1](#A1.SS1.SSS0.Px4.p1.1 "FastText Filtering ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* X. Wang, Y. Chen, and W. Zhu (2021)
  A survey on curriculum learning.
  IEEE transactions on pattern analysis and machine intelligence 44 (9),  pp. 4555–4576.
  Cited by: [§A.2](#A1.SS2.p1.1 "A.2 Data Composition Across Stages ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* Y. Wang, Z. Fu, J. Cai, P. Tang, H. Lyu, Y. Fang, Z. Zheng, J. Zhou, G. Zeng, C. Xiao, et al. (2025)
  Ultra-fineweb: efficient data filtering and verification for high-quality llm training data.
  arXiv preprint arXiv:2505.05427.
  Cited by: [§A.1](#A1.SS1.SSS0.Px1.p1.1 "Data Collection ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* J. Wei, Y. Tay, R. Bommasani, C. Raffel, B. Zoph, S. Borgeaud, D. Yogatama, M. Bosma, D. Zhou, D. Metzler, et al. (2022)
  Emergent abilities of large language models.
  arXiv preprint arXiv:2206.07682.
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* M. Wortsman, G. Ilharco, S. Y. Gadre, R. Roelofs, R. Gontijo-Lopes, A. S. Morcos, H. Namkoong, A. Farhadi, Y. Carmon, S. Kornblith, et al. (2022a)
  Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.
  In International conference on machine learning,
   pp. 23965–23998.
  Cited by: [§4.3.1](#S4.SS3.SSS1.p1.1 "4.3.1 Merging Methods ‣ 4.3 Ablation Study ‣ 4 Experimental Results ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§6.2](#S6.SS2.p1.1 "6.2 Model Merging ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* M. Wortsman, G. Ilharco, J. W. Kim, M. Li, S. Kornblith, R. Roelofs, R. G. Lopes, H. Hajishirzi, A. Farhadi, H. Namkoong, et al. (2022b)
  Robust fine-tuning of zero-shot models.
  In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,
   pp. 7959–7971.
  Cited by: [§6.2](#S6.SS2.p1.1 "6.2 Model Merging ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* T. Wu, R. Yang, J. Li, P. Hu, N. Wong, and Y. Yang (2025)
  Shadow-ft: tuning instruct via base.
  arXiv preprint arXiv:2505.12716.
  Cited by: [§2.3](#S2.SS3.p5.2 "2.3 Model Merging as Proxy ‣ 2 Method ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§2.3](#S2.SS3.p6.2 "2.3 Model Merging as Proxy ‣ 2 Method ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§6.2](#S6.SS2.p1.1 "6.2 Model Merging ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* S. Xiao, Z. Liu, P. Zhang, and X. Xing (2024)
  Lm-cocktail: resilient tuning of language models via model merging.
  In Findings of the Association for Computational Linguistics: ACL 2024,
   pp. 2474–2488.
  Cited by: [§6.2](#S6.SS2.p1.1 "6.2 Model Merging ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* S. M. Xie, H. Pham, X. Dong, N. Du, H. Liu, Y. Lu, P. S. Liang, Q. V. Le, T. Ma, and A. W. Yu (2023)
  Doremi: optimizing data mixtures speeds up language model pretraining.
  Advances in Neural Information Processing Systems 36,  pp. 69798–69818.
  Cited by: [§2.3](#S2.SS3.p3.1 "2.3 Model Merging as Proxy ‣ 2 Method ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§3.2](#S3.SS2.p1.1 "3.2 Baselines ‣ 3 Experimental Settings ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§6.1](#S6.SS1.p1.1 "6.1 Data Mixture ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* P. Yadav, D. Tam, L. Choshen, C. A. Raffel, and M. Bansal (2023)
  Ties-merging: resolving interference when merging models.
  Advances in Neural Information Processing Systems 36,  pp. 7093–7115.
  Cited by: [§4.3.1](#S4.SS3.SSS1.p1.1 "4.3.1 Merging Methods ‣ 4.3 Ablation Study ‣ 4 Experimental Results ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§6.2](#S6.SS2.p1.1 "6.2 Model Merging ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. (2025)
  Qwen3 technical report.
  arXiv preprint arXiv:2505.09388.
  Cited by: [§A.1](#A1.SS1.SSS0.Px6.p1.1 "Instance-Level Data Labeling ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§3](#S3.SS0.SSS0.Px1.p1.1 "Models ‣ 3 Experimental Settings ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* E. Yang, L. Shen, G. Guo, X. Wang, X. Cao, J. Zhang, and D. Tao (2024)
  Model merging in llms, mllms, and beyond: methods, theories, applications, and opportunities.
  ACM Computing Surveys.
  Cited by: [§2.3](#S2.SS3.p2.8 "2.3 Model Merging as Proxy ‣ 2 Method ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"),
  [§6.2](#S6.SS2.p1.1 "6.2 Model Merging ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* J. Ye, P. Liu, T. Sun, J. Zhan, Y. Zhou, and X. Qiu (2024)
  Data mixing laws: optimizing data mixtures by predicting language modeling performance.
  arXiv preprint arXiv:2403.16952.
  Cited by: [§6.1](#S6.SS1.p1.1 "6.1 Data Mixture ‣ 6 Related Works ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* L. Yu, B. Yu, H. Yu, F. Huang, and Y. Li (2024)
  Language models are super mario: absorbing abilities from homologous models as a free lunch.
  In Forty-first International Conference on Machine Learning,
  Cited by: [§4.3.1](#S4.SS3.SSS1.p1.1 "4.3.1 Merging Methods ‣ 4.3 Ablation Study ‣ 4 Experimental Results ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi (2019)
  Hellaswag: can a machine really finish your sentence?.
  arXiv preprint arXiv:1905.07830.
  Cited by: [§3.1](#S3.SS1.p1.1 "3.1 Benchmark ‣ 3 Experimental Settings ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").
* F. Zhou, Z. Wang, N. Ranjan, Z. Cheng, L. Tang, G. He, Z. Liu, and E. P. Xing (2025)
  Megamath: pushing the limits of open math corpora.
  arXiv preprint arXiv:2504.02807.
  Cited by: [§A.1](#A1.SS1.SSS0.Px1.p1.1 "Data Collection ‣ A.1 Data Curation Pipeline ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").

## Appendix A Details of DeMix Corpora

### A.1 Data Curation Pipeline

This subsection presents a unified pipeline for building the DeMix Corpora from heterogeneous sources including general-domain, multilingual, mathematical, and code data. Global exact and fuzzy deduplication is scaled to enhance the corpus’s overall quality and eliminate redundant content; dataset-specific perplexity filtering via a lightweight scoring model further refines data validity and relevance. A FastText-based quality classifier, trained on well-constructed positive-negative data mixtures and validated through controlled pre-training, is leveraged to identify and retain semantically meaningful high-quality data samples. In addition, for Chinese general-domain corpora, we introduce a dedicated quality classifier to filter low-quality samples and upsample high-quality signals, further improving the quality distribution of the Chinese data. Finally, hierarchical instance-level labeling—bootstrapped from high-confidence large-model annotations and distilled into a lightweight labeler—is introduced to enable subsequent data analysis and targeted data selection by category, with the corpus quality additionally verified through small-model training experiments and benchmark comparisons.

##### Data Collection

The general-domain data we collected mainly covers well-known open-source datasets such as FineWeb-Edu (Penedo et al., [2024](#bib.bib20 "The fineweb datasets: decanting the web for the finest text data at scale")), DCLM-Baseline (Li et al., [2024](#bib.bib21 "Datacomp-lm: in search of the next generation of training sets for language models")), DOLMA-v1.7(Soldaini et al., [2024](#bib.bib43 "Dolma: an open corpus of three trillion tokens for language model pretraining research")), Ultra-FineWeb (Wang et al., [2025](#bib.bib65 "Ultra-fineweb: efficient data filtering and verification for high-quality llm training data")), and the NVIDIA Nemotron corpus (Basant et al., [2025](#bib.bib27 "Nvidia nemotron nano 2: an accurate and efficient hybrid mamba-transformer reasoning model")). For mathematical data, we include sources such as FineMath (Liu et al., [2025](#bib.bib62 "Finemath: a fine-grained mathematical evaluation benchmark for chinese large language models")), MegaMath (Zhou et al., [2025](#bib.bib63 "Megamath: pushing the limits of open math corpora")), and SwallowMath (Fujii et al., [2025](#bib.bib22 "Rewriting pre-training data boosts llm performance in math and code")). For code data, we include sources such as OpenCoder (Huang et al., [2025](#bib.bib64 "Opencoder: the open cookbook for top-tier code large language models")) and SwallowCode (Fujii et al., [2025](#bib.bib22 "Rewriting pre-training data boosts llm performance in math and code")). In addition, we incorporated multilingual data and reasoning data. A complete list of our data sources is provided in Table [8](#A1.T8 "Table 8 ‣ A.2 Data Composition Across Stages ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").

##### Deduplication

We perform global exact deduplication and fuzzy deduplication on most datasets (Lee et al., [2022](#bib.bib66 "Deduplicating training data makes language models better")), while retaining the original data only for confirmed high-quality small-scale datasets. Although some works (Penedo et al., [2024](#bib.bib20 "The fineweb datasets: decanting the web for the finest text data at scale"), [2025](#bib.bib67 "FineWeb2: one pipeline to scale them all–adapting pre-training data processing to every language")) argue that global deduplication may remove a substantial number of high-quality samples, we still adopt it for large-scale web data, primarily because the greatly increased data volume mitigates the impact of this issue.
For fuzzy deduplication, we extract 24-grams from each document, compute 260 MinHash functions per document, and partition them into 20 bands of 13 hashes each, targeting documents with at least 90% similarity. Any pair of documents that share an identical 13-MinHash signature in any band is considered duplicates.

##### Perplexity Filtering

We perform perplexity-based data filtering using the Qwen3-0.6B base model as the scoring model. For each dataset, we manually inspect high-perplexity samples and determine dataset-specific thresholds. Based on these thresholds, samples with extremely low perplexity are removed, resulting in an overall data reduction of approximately 2%.

##### FastText Filtering

We construct a binary data quality classifier based on FastText. Both the positive and negative sets contain over one million samples. The positive set is composed of randomly sampled data, low-perplexity data, and high-quality subsets from ELI5-category (Fan et al., [2019](#bib.bib68 "ELI5: long form question answering")) and OpenHermes-2.5 (Teknium, [2023](#bib.bib70 "OpenHermes 2.5: an open dataset of synthetic data for generalist llm assistants")), while the negative set is partially sourced from Falcon-RefinedWeb (Penedo et al., [2023](#bib.bib69 "The refinedweb dataset for falcon llm: outperforming curated corpora with web data, and web data only")) and high-perplexity data.
Since high-perplexity data is dominated by short samples, we apply weighted sampling to long samples to increase the likelihood of selecting low-quality long-form data.
To determine appropriate source proportions for the positive and negative sets, we construct multiple candidate sample mixtures and train Qwen3-0.6B from scratch on each set separately.
Experimental results show that the model trained on the positive set achieves a lower average evaluation loss than the model trained on the negative set, demonstrating a clear quality distinction between the two sets.
Based on these validated samples, we subsequently train a FastText classifier. The resulting classifier removes approximately 3% of the English corpus and effectively identifies low-quality web and code data.

##### Quality Filtering for Chinese General-Domain Corpora

We propose a quality assessment and sampling framework for large-scale Chinese general-domain web corpora, inspired by prior work such as FineWeb-Edu(Penedo et al., [2024](#bib.bib20 "The fineweb datasets: decanting the web for the finest text data at scale")). In this framework, education-related and information-dense signals are treated as proxies for high-quality text, with the goal of preferentially retaining and emphasizing content with higher informational value and structural coherence during corpus construction.
Specifically, we categorize text samples into five quality levels: undetermined (e.g., non-Chinese or nonsensical content), extremely low quality, low quality, high quality, and extremely high quality. The labeling criteria jointly consider linguistic completeness, information density, structural coherence, and the degree to which the text exhibits knowledge-bearing or explanatory characteristics, without restricting the data to explicitly educational contexts.
To obtain reliable quality annotations, we first manually labeled 5,000 Chinese samples and fine-tuned a 32B-parameter language model to learn the above quality distinctions. We then used this model to automatically annotate approximately 1 million Chinese web documents. Based on these pseudo-labeled samples, we trained a lightweight text quality classifier, which uses gte-multilingual-base as the text embedding backbone followed by a classification head.
During corpus construction, the trained classifier is applied to remove samples classified as extremely low quality or undetermined, while samples labeled as extremely high quality are upsampled, thereby improving the overall quality distribution and structural characteristics of Chinese general-domain training data.

##### Instance-Level Data Labeling

To estimate the domain distribution of our pre-training corpus, we build a three-level hierarchical taxonomy. We define level-1 labels following the standard graduate disciplinary classification, and derive level-2 and level-3 labels under each parent via Gemini 2.5 Pro (Comanici et al., [2025](#bib.bib85 "Gemini 2.5: pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities")) and GPT-4o (Achiam et al., [2023](#bib.bib14 "Gpt-4 technical report")). We then annotate the corpus using Qwen3-235B-A22B (Yang et al., [2025](#bib.bib15 "Qwen3 technical report")) and retain 3.63M high-confidence labeled instances to train a 4B model, which is used to label the full pre-training dataset.

##### Data Evaluation

To validate the quality of individual datasets, we train a relatively small model (Qwen-3-1.7B) on 50B tokens. For general datasets, we train from scratch. For math/code datasets, we mix in 80%/60% general data and train from a pre-trained mid checkpoint. The benchmarks we use are widely adopted, ensuring that they can exceed random performance after a small amount of training and maintain stable ranking across subsequent training stages.

##### Candidate Data Preparation

DeMix is leveraged in the late stages of pre-training: for general data, we select only the highest-quality tier; for math and code, we discard the lowest-quality tiers, stratify the remaining corpora by both category and quality, and merge similar sources to reduce the dataset count. This constitutes a common engineering trade-off (Diao et al., [2025](#bib.bib36 "Climb: clustering-based iterative data mixture bootstrapping for language model pre-training")) and a standard practice for cross-domain data mixing (Basant et al., [2025](#bib.bib27 "Nvidia nemotron nano 2: an accurate and efficient hybrid mamba-transformer reasoning model"); Feng et al., [2024](#bib.bib26 "Maximize your data’s potential: enhancing llm accuracy with two-phase pretraining")). By reducing the dimensionality of mixture ratios, we substantially alleviate the search burden of identifying a high-performance mixture. While this may lower the attainable performance upper bound, it is a necessary compromise given the exorbitant cost of large-scale pre-training.
Ultimately, we obtain seven dataset categories that require fine-grained mixing.
For each candidate dataset, we mix it with 50% general data as a form of regularization, since an excessive proportion of domain-specific data can significantly degrade the model’s general capabilities during LLM pre-training (Lin et al., [2025a](#bib.bib89 "Rec-r1: bridging generative large language models and user-centric recommendation systems via reinforcement learning"); Bansal and Sanghavi, [2025](#bib.bib90 "Context-free synthetic data mitigates forgetting"); Allal et al., [2025b](#bib.bib37 "The smol training playbook: the secrets to building world-class llms")).
Although we can only search within the subspace where non-general data accounts for less than 50%, this constraint—that non-general data should not dominate the pre-training corpus—is consistent with the consensus (Basant et al., [2025](#bib.bib27 "Nvidia nemotron nano 2: an accurate and efficient hybrid mamba-transformer reasoning model"); Blakeman et al., [2025](#bib.bib28 "Nemotron 3 nano: open, efficient mixture-of-experts hybrid mamba-transformer model for agentic reasoning"); Allal et al., [2025a](#bib.bib30 "SmolLM2: when smol goes big–data-centric training of a small language model")).

##### Component Model Training

While DeMix is adopted for the late stages, for early pre-training, the model exhibits limited capacity and thus cannot tackle more challenging tasks; we therefore evaluate data mixtures using general benchmarks, which is a single-objective optimization goal that can be cultivated during early training, making the construction of data mixtures relatively simple and straightforward. In practice, we find that direct upsampling/downsampling based on quality scores yield consistent and rational data mixtures.
We train a 1.7B model from scratch using the stage-1 mixture for a sufficient number of tokens, which serves as the base model for component models. We then train seven component models {Mi}\{M\_{i}\} on each candidate dataset {Di}\{D\_{i}\} on 50B tokens, ensuring that they possess sufficient emergent capabilities to solve challenging math and code problems.

### A.2 Data Composition Across Stages

![Refer to caption](/html/2602.00747/assets/x5.png)

Figure 5: 
Data mixtures for the three stages of pre-training in the DeMix Corpora, with approximately 14T, 6T, and 2T tokens, respectively. The three hierarchical levels from the inside out are domain, data category, and data origin.




Table 8: Composition of DeMix Corpora and the mixture in each stage, measured in tokens (B).

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Category | Quality | Original Source | Amount | Stage 1 | Stage 2 | Stage 3 |
| General | High | FineWeb-Edu | 1434 | 740 | 338 | 98 |
| Nemo-CC-High | 384 | 1191 | 543 | 158 |
| Nemo-CC-Synth | 1100 | 3630 | 1656 | 480 |
| Nemo-CC-QA | 547 | 2190 | 999 | 290 |
| Medium-High | DCLM | 509 | 509 | 0 | 0 |
| DOLMA (Social) | 72 | 72 | 0 | 0 |
| Nemo-CC-Med-High | 392 | 392 | 0 | 0 |
| OpenCoder (Web) | 38 | 38 | 0 | 0 |
| Ultra-FineWeb (en) | 80 | 80 | 0 | 0 |
| DOLMA (Others) | 54 | 54 | 0 | 0 |
| Nemo-SFT (General) | 67 | 67 | 0 | 0 |
| Medium | DOLMA (Web) | 1289 | 774 | 0 | 0 |
| Web Crawl (en) | 1436 | 330 | 0 | 0 |
| Nemo-CC-Medium | 1608 | 965 | 0 | 0 |
| DOLMA (Scholar) | 78 | 78 | 0 | 0 |
| Multilingual | High | Ultra-FineWeb (zh) | 102 | 102 | 102 | 30 |
| Nemo-CC-QA-Mul | 562 | 562 | 562 | 163 |
| Medium | Web Crawl (zh) | 3304 | 991 | 0 | 0 |
| Math | High | Nemo-Math Mind (4+) | 69 | 69 | 196 | 148 |
| Nemo-Math (4+) | 50 | 50 | 142 | 107 |
| Nemo-SFT (Math) | 169 | 169 | 481 | 362 |
| Medium-High | Reason (Math) | 37 | 37 | 37 | 0 |
|  | SwallowMath | 31 | 31 | 31 | 0 |
| Medium | MegaMath | 93 | 93 | 87 | 58 |
| FineMath (4+) | 8 | 8 | 8 | 5 |
| Low | Nemo-Math (3) | 77 | 77 | 0 | 0 |
| FineMath (3) | 20 | 20 | 0 | 0 |
|  | Others | 49 | 49 | 0 | 0 |
| Code | High | Nemo-SFT (Code) | 46 | 46 | 150 | 39 |
| OpenCoder | 6 | 6 | 20 | 5 |
| Medium-High | Nemo-Code-Synth | 141 | 141 | 209 | 86 |
| MegaMath (Code) | 5 | 5 | 7 | 3 |
| Reason (Code) | 33 | 33 | 49 | 20 |
| Medium | Source Code | 579 | 579 | 210 | 87 |
|  | SwallowCode | 47 | 47 | 17 | 7 |
|  | Low | StarCoder | 207 | 207 | 0 | 0 |
| Total | | | 14723 | 14432 | 5844 | 2146 |

The data mixture used in LLM pre-training is inherently dynamic rather than static (Feng et al., [2024](#bib.bib26 "Maximize your data’s potential: enhancing llm accuracy with two-phase pretraining"); Basant et al., [2025](#bib.bib27 "Nvidia nemotron nano 2: an accurate and efficient hybrid mamba-transformer reasoning model"); Blakeman et al., [2025](#bib.bib28 "Nemotron 3 nano: open, efficient mixture-of-experts hybrid mamba-transformer model for agentic reasoning")).
During the initial phases, the priority is typically given to data diversity, whereas the later stages shift focus toward high data quality (Wang et al., [2021](#bib.bib29 "A survey on curriculum learning")).
In practice, early training stages typically emphasize data diversity, while late stages increasingly prioritize high-quality data to refine advanced capabilities (Wang et al., [2021](#bib.bib29 "A survey on curriculum learning")).
Consequently, pre-training is often organized into multiple stages, with the proportion of high-quality math and code data significantly increased in the late stages (Basant et al., [2025](#bib.bib27 "Nvidia nemotron nano 2: an accurate and efficient hybrid mamba-transformer reasoning model"); Allal et al., [2025a](#bib.bib30 "SmolLM2: when smol goes big–data-centric training of a small language model")).
As a result, modern pre-training pipelines are often organized into multiple stages, with the proportion of high-quality math and code data substantially increased in the late stages (Basant et al., [2025](#bib.bib27 "Nvidia nemotron nano 2: an accurate and efficient hybrid mamba-transformer reasoning model"); Allal et al., [2025a](#bib.bib30 "SmolLM2: when smol goes big–data-centric training of a small language model")).

In practice, we partition the pre-training data into three stages, with the proportion of high-quality data such as mathematical and coding data gradually increasing across successive stages. In the first stage, as the model primarily acquires broad general knowledge during the initial phase of training, data quality exerts a less significant impact than in subsequent stages. Consequently, we only adjust the data mixture using basic general benchmarks, performing upsampling and downsampling based on quality scores. In contrast, during Stages 2 and 3, data quality directly influences the final training performance; thus, we employ the DeMix framework to optimize the data mixture ratio. Furthermore, to prevent excessive repetition of individual samples, we reduce the threshold for the allowable number of repetitions in Stage 2, yielding a more balanced data distribution compared to Stage 3.

We presents the detailed composition of DeMix Corpora in Figure [5](#A1.F5 "Figure 5 ‣ A.2 Data Composition Across Stages ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training") and Table [8](#A1.T8 "Table 8 ‣ A.2 Data Composition Across Stages ‣ Appendix A Details of DeMix Corpora ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training"), including the amount of tokens after our curation pipeline and the token allocation in 3 training stages after data mixing by DeMix.

## Appendix B Baseline Details.

For RegMix and CLIMB, the only differences from DeMix lie in how proxy models are obtained and how the predictor is trained.
For RegMix, we sample a required number of mixtures and train the model from the same initialization as the DeMix component models, using a fixed number of tokens. We then evaluate the trained models on to obtain benchmark scores, and fit a LightGBM predictor using the same hyperparameters as in DeMix. Finally, we apply the trained predictor to the same large set of sampled candidates and select the top-ranked mixtures as the final optimal mixture.
For CLIMB, the only difference is that it adopts iterative sampling. Consistent with our DeMix procedure, it performs three iterations: for 28 points, we sample 16+8+416+8+4; for 56 points, 32+16+832+16+8; for 112 points, 64+32+1664+32+16.

## Appendix C Detailed Mixtures in Experiments.

Table 9: Detailed mixtures from different experiments.

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | Train Cost (B) ↓\downarrow | | | | General | Math-1 | Math-2 | Math-3 | Code-1 | Code-2 | Code-3 |
| Uniform | - | - | - | - | 0.500 | 0.111 | 0.027 | 0.039 | 0.020 | 0.055 | 0.248 |
| Heuristic-2 | - | - | - | - | 0.400 | 0.194 | 0.024 | 0.017 | 0.052 | 0.098 | 0.216 |
| Heuristic-3 | - | - | - | - | 0.200 | 0.273 | 0.033 | 0.000 | 0.098 | 0.136 | 0.259 |
| RegMix | 224 | - | 112 | 2 | 0.426 | 0.336 | 0.020 | 0.012 | 0.030 | 0.074 | 0.092 |
| 224 | - | 28 | 8 | 0.710 | 0.203 | 0.003 | 0.001 | 0.079 | 0.002 | 0.002 |
| 448 | - | 56 | 8 | 0.417 | 0.422 | 0.027 | 0.011 | 0.079 | 0.008 | 0.035 |
| CLIMB | 224 | - | 112 | 2 | 0.481 | 0.366 | 0.018 | 0.008 | 0.033 | 0.034 | 0.060 |
| 224 | - | 28 | 8 | 0.714 | 0.202 | 0.002 | 0.001 | 0.078 | 0.002 | 0.002 |
| 448 | - | 56 | 8 | 0.417 | 0.422 | 0.027 | 0.011 | 0.079 | 0.008 | 0.035 |
| DeMix  (Ours) | 211 | 30×\times7 | 56 | 0.01†\dagger | 0.017 | 0.187 | 0.406 | 0.017 | 0.135 | 0.217 | 0.022 |
| 211 | 30×\times7 | 112 | 0.01 | 0.271 | 0.414 | 0.009 | 0.057 | 0.046 | 0.131 | 0.073 |
| 212 | 30×\times7 | 224 | 0.01 | 0.218 | 0.403 | 0.002 | 0.063 | 0.044 | 0.176 | 0.094 |
| 214 | 30×\times7 | 448 | 0.01 | 0.454 | 0.220 | 0.006 | 0.015 | 0.077 | 0.194 | 0.034 |
|  | 219 | 30×\times7 | 896 | 0.01 | 0.181 | 0.430 | 0.004 | 0.135 | 0.029 | 0.149 | 0.071 |

* •

  †\dagger is the benchmarking cost derived from equal GPU‑hour as in Table [1](#S3.T1 "Table 1 ‣ 3.1 Benchmark ‣ 3 Experimental Settings ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").

We extend Table [3](#S4.T3 "Table 3 ‣ 4 Experimental Results ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training") by listing the detailed mixture ratios in Table [9](#A3.T9 "Table 9 ‣ Appendix C Detailed Mixtures in Experiments. ‣ Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training").

[◄](/html/2602.00746)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2602.00747)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2602.00747)
[View original  
on arXiv](https://arxiv.org/abs/2602.00747)[►](/html/2602.00748)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Thu Mar 5 21:05:54 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
