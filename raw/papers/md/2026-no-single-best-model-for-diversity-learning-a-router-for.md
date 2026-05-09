---
arxiv: '2604.02319'
authors:
- Yuhan Liu
- Fangyuan Xu
- Vishakh Padmakumar
- Daphne Ippolito
- Eunsol Choi
parser: ar5iv
retrieved: '2026-05-09'
source: paper
title: 'No Single Best Model for Diversity: Learning a Router for Sample Diversity'
url: https://arxiv.org/abs/2604.02319
year: 2026
---

[2604.02319] No Single Best Model for Diversity: Learning a Router for Sample Diversity














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



# No Single Best Model for Diversity: Learning a Router for Sample Diversity

Yuhan Liu  Fangyuan Xu  Vishakh Padmakumar
  
Daphne Ippolito   Eunsol Choi
Affiliation:  New York University   Stanford University   Carnegie Mellon Universityyl13579@nyu.edu

###### Abstract

When posed with prompts that permit a large number of valid answers, comprehensively generating them is the first step towards satisfying a wide range of users. In this paper, we study methods to elicit a comprehensive set of valid responses.
To evaluate this, we introduce diversity coverage, a metric that measures the total quality scores assigned to each unique answer in the predicted answer set relative to the best possible answer set with the same number of answers.
Using this metric, we evaluate 18 LLMs, finding no single model dominates at generating diverse responses to a wide range of open-ended prompts. Yet, per each prompt, there exists a model that outperforms all other models significantly at generating a diverse answer set.
Motivated by this finding, we introduce a router that predicts the best model for each query. On *NB-WildChat*, our trained router outperforms the single best model baseline (26.3%26.3\% vs 23.8%23.8\%). We further show generalization to an out-of-domain dataset (*NB-Curated*) as well as different answer-generation prompting strategies. Our work lays foundation for studying generating comprehensive answers when we have access to a suite of models.

## 1 Introduction

Various tasks require language models (LMs) to generate diverse high-quality responses.
These range from creative writing (padmakumar2023does; chung2025modifying), dialogues (lintomlin2025usersim), code (wu2026x), math (wu2025mode), scientific discovery (novikov2025alphaevolve; gottweis2025towards), survey response simulation (meister2024benchmarking) to synthetic data generation (honovich2022unnatural). Evaluation on these tasks thus should move beyond the quality of a single output to the diversity and quality of a set of outputs. However, existing metrics fall short in quantifying the coverage of open-ended answer space. In this work, we introduce *diversity coverage*, a metric that measures the total quality scores assigned to all unique answers in the predicted answer set, relative to the best possible answer set with the same number of answers.

Prior works focused on methods for improving the diversity of generations from a single LLM, such as changing the inference hyperparameters (holtzman2019curious; kambhatla2022surfacing; santurkar2023whose; nguyen2024turning) or prompt (lu2024benchmarking; zhang2025verbalized).
In this paper, we ask the question of whether *diversity coverage* can be further improved by taking advantage of bountiful LMs, each with differing behaviours towards the same prompt.
We hypothesize that heterogeneous LLMs can be effectively ensembled in order to leverage their complementary strengths.
Through a pilot study, we identify that no single LLM dominates in generating a diverse and high-quality output set, and different LLMs excel at answering different open-ended questions ([Section 3](#S3 "3 A pilot study on ensembling models to maximize diversity coverage ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity")). If we can pick the optimal model for each example, 33.0%33.0\% diversity coverage can be achieved on *NB-WildChat*, revealing a large gap compared to using the overall best model in the ensemble (23.8%23.8\%).

However, determining the optimal model for a question can be challenging, as optimizing diversity involves analyzing the joint effect of all sampled answers. Although many previous works have proposed routing to select the best LLM (Jiang2023LLMBlenderEL; Zhang2025RouterR1TL; lu-etal-2024-routing), candidates are ranked based on a single response to the ground truth. This problem becomes exponentially more challenging when comparing answer sets as it is expensive to sample and compare multiple outputs for each example. The difficulty is further increased for open-ended questions as there are no gold answer sets available. Lastly, one might expect that simple heuristics based on model metadata (e.g., family or size) could help predict diversity. In fact, we show in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity") that models of different families and sizes generate the most diverse outputs for a disjoint set of queries.

Motivated by this, we propose a simple approach to predict the best model to respond to any given query.
We train a router that scores the diversity and quality of each candidate model in the routing pool and routes to the best LLM to generate the answers. We frame it as a classification task and create a training dataset of queries and model scores. Our router outperforms top overall baselines on the *NB-WildChat* and also generalizes to *NB-Curated*. Using the same trained router, we explore further ensembling outputs from two models per query, which brings further performance gains. We will release the code and dataset publicly. Our key contributions are:

1. 1.

   We propose a new metric, diversity coverage, to jointly measure the diversity and quality of an answer set to open-ended questions with diverse output space.
2. 2.

   Motivated by the finding that no single model excels at generating diverse outputs to all queries, we propose to train a router which predicts the best model for generation given an input query. We evaluate our method comprehensively on multiple datasets, demonstrating competitive performance on both in-domain (*NB-WildChat*) and out-of-domain (*NB-Curated*) settings.
3. 3.

   We further investigate the effect of training data size, prompting strategies as well as inference efficiency of using a diversity router. More broadly, our findings highlight the potential of multi-LLM systems, where models with complementary strengths are combined to produce more diverse and high-quality solutions.

![Refer to caption](/html/2604.02319/assets/x1.png)


Figure 1: 
Left: LLMs exhibit different diversity coverage. Right: There is no universal best model on *NB-WildChat*. A model is only considered to be the best model if its diversity scores are 5%5\% higher than the second most best candidate. Queries without a model satisfying this margin are labeled as “No dominant single models”. On *Simple Questions*, all models perform similarly, resulting in 100%100\% of “No dominant single models”. On *NB-WildChat*, there is no model that consistently dominates all queries.

## 2 Task Formulation

Many queries admit multiple valid responses rather than a single correct answer. We summarize datasets containing such queries in Table [1](#S2.T1 "Table 1 ‣ 2 Task Formulation ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity"). Depending on the number of possible answers, we categorize the datasets into two types:

* •

  Fixed answer set. Each query has a finite ground truth answer set A∗A^{\*}. The predicted answer is correct if it belongs to A∗A^{\*}.
* •

  Open-ended answer set. Queries admit infinitely many valid answers, and listing them comprehensively is infeasible. The validity of the predicted answer can be evaluated based on quality and assigned a scalar value.

|  |  |  |  |
| --- | --- | --- | --- |
| Dataset | # | Example | |
| Question | Possible Valid Answers |
| *Simple Questions*  (zhang2024forcing) | 2323 | Output a random  country in North America. | United States/ Canada/ Mexico / … |
| *NB-Curated* 111The original released dataset has 100 questions. We filter out 4 questions that ask for multiple answers (violating our hypothesis) and 4 questions that do not have multiple correct answers.  (zhang2025noveltybench) | 9292 | Tell me a funny joke. | Who is Adam and  why he is optimizing my code? /  What did a late tomato say to other  tomatoes? I will ketchup, … |
| *NB-WildChat*  (zhang2025noveltybench) | 11k | Give me a very simple  way to remember the  formula for tangent. | Tangent = Opposite /Adjacent /  Draw a unit circle … |
| *Infinity-Chat*  (Jiang2025ArtificialHT) | 2626k | Write a story about  America. | In the heartland of America there was  a small town …/  My decision to go to the United States… |

Table 1: Dataset statistics. *Simple Questions* has a fixed answer set and all other datasets have open-ended answer sets.

### 2.1 Task definition

Given a query qq and generation budget BB (i.e., the number of answers produced by model), the task is to derive an answer set A={a1,…,aB}A=\{a\_{1},\ldots,a\_{B}\} that covers as many distinct and high-quality answers as possible. Our task definition assumes two functions:

* •

  uniq(q,Aq,A): Given a query qq and an answer set AA, it outputs a subset of AdA\_{d}, consisting of distinct answers only (i.e., no two answers in AdA\_{d} are equivalent to each other). We follow prior work (zhang2025noveltybench) to derive AdA\_{d}. We iterate over answers in AA and greedily add a new answer to AdA\_{d} if the current answer aa is not equivalent to any answer already in AdA\_{d}. The equivalence of two answers is determined by exact string match for queries with fixed answer set and an equivalence classifier for open-ended question.
  The process is described in Appendix [C](#A3 "Appendix C Diversity coverage calculation details on open-ended questions ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity").
* •

  quality(q,aiq,a\_{i}): Given an individual answer aia\_{i} for query qq, it outputs a scalar value representing the quality of the answer aia\_{i}. This can be done through either comparing against ground truth answer sets (factual queries) or using a reward model (open-ended queries).

### 2.2 New Evaluation Metric: Diversity Coverage

Given a predicted set of answers A={a1,…,aB}A=\{a\_{1},\ldots,a\_{B}\} to query qq,
we introduce a new metric, *diversity coverage (div-cov)* as follows:

|  |  |  |
| --- | --- | --- |
|  | div−cov⁡(q,A)≔1max−uniq−sum⁡(q,B)​∑a∈uniq​(q,A)quality⁡(q,a)\operatorname{div-cov}(q,A)\coloneqq\frac{1}{\operatorname{max-uniq-sum}(q,B)}\sum\_{a\in\text{uniq}({q,A})}\operatorname{quality}(q,a) |  |

We define max-uniq-sum(q,Bq,B) as the maximum score that one can reach by generating an answer set of size BB where each answer in the set is distinct and achieves maximum quality.

For questions with a fixed answer set A∗A^{\*}, assuming B≥|A∗|B\geq|A^{\*}|, this measures the proportion of unique ground-truth answers covered by the answer set, equivalent to the coverage rate metric proposed by zhang2024forcing.

Prior works measure diversity by the number of unique, valid outputs for questions with fixed answer set (zhang2024forcing); or via pairwise embedding-based similarity
(zhang2025verbalized; Jiang2025ArtificialHT). Such metric either does not work for questions with open-ended answer space, or do not account for the quality of the answers (zhang2025verbalized; Jiang2025ArtificialHT).
zhang2025noveltybench proposes a unified metric for quality and diversity considering an ordered answer list, which penalizes answers generated later to account for user patience. This paper focuses on evaluating the quality and diversity of a set of answers, regardless of generation order, making diversity coverage better suited to our purpose.

## 3 A pilot study on ensembling models to maximize diversity coverage

In this section, we first study whether a strong LLM can dominate other models in diversity coverage for a range of questions (Section [3](#S3 "3 A pilot study on ensembling models to maximize diversity coverage ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity")). We found no single model dominates, motivating us to explore the upper bound of gains if use a pool of LLMs instead of a single LLM (Section [3.2](#S3.SS2 "3.2 Oracle experiment: how much does picking the best model(s) per query improve? ‣ 3 A pilot study on ensembling models to maximize diversity coverage ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity")). We compare several ensembling strategies under oracle model selection setting. We find that picking the best LLM per question is the most promising, which leads us to develop model router in later sections.

### 3.1 No single model is best at diversity coverage for all questions

#### Model sets

We study 1818 models from four open-source model families with different parameter counts: Llama (Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B, Llama-3.3-70B), Qwen (Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Qwen3-14B, Qwen2.5-72B), OLMo (OLMo-2-0425-1B, OLMo-2-1124-7B, OLMo-2-1124-13B, OLMo-2-0325-32B), Gemma(gemma-3-1b, gemma-3-4b, gemma-3-12b, gemma-3-27b).

#### Settings

For each model and query, we sample NN answers with a prompt which instructs the model to enumerate as many valid answers as possible (see Appendix [B](#A2 "Appendix B Prompts ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity") for the full template). This prompt encourages models to explore the space of possible responses rather than produce a single canonical answer.222Prior work zhang2025verbalized has found that this method elicits more diverse answer set compared to sampling multiple single answer. We further compare different prompt templates (e.g. generate one or two answers in a single generation) in Appendix [H.1](#A8.SS1 "H.1 Different Prompt Templates ‣ Appendix H Discussion ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity"), finding that generating all answers in a single prompt elicits the most diverse answer set.
We keep the decoding method fixed throughout the paper (described in Appendix [D](#A4 "Appendix D Decoding settings ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity")).
For each query, we define the “dominant” model by the following two criteria: (1) the model achieves the highest diversity coverage, and (2) the score is at least 5%5\% higher than that of the answer sets generated by any other models.

#### Results

On *Simple Questions*, we find no dominant model for any query. For *NB-WildChat* queries, this remains true for 30% of queries, and more than 5 models are dominant at least on 5% of queries, suggesting optimizing model choices per question can be fruitful. Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity") compares per-model frequency of achieving the best diversity coverage on *NB-WildChat*.

### 3.2 Oracle experiment: how much does picking the best model(s) per query improve?

Motivated by the finding that different models can generate diverse outputs for different queries, can we ensemble outputs from models to achieve diverse outputs? We assume an oracle setting, where we have access to the diversity coverage scores of all LLMs on all queries.
We compare three strategies to ensemble models with a fixed generation budget BB333In our experiments, we fix BB to be 5050 answers per question if not otherwise stated.:

|  |  |  |  |
| --- | --- | --- | --- |
|  | SQ | Curated | WildChat |
| Top overall model | 96.9%96.9\% | 47.0%47.0\% | 23.8%23.8\% |
| Top two overall models | 97.1%97.1\% | 45.6%45.6\% | 25.6%25.6\% |
| Random model / query | 92.7%92.7\% | 37.5%37.5\% | 18.1%18.1\% |
| Top model / query | 97.9% | 59.6% | 33.0% |

Table 2: Diversity coverage scores for ensembling multiple LLMs on *Simple Questions* (SQ), *NB-Curated* (Curated) and *NB-WildChat* (WildChat).

* •

  Top overall model. We select the single model with the best average diversity coverage per dataset, representing the best possible performance without ensembling. The selected top models are respectively: Llama-3.1-8B, Qwen3-14B and OLMo-2-0425-1B.
* •

  Top two overall models. We select two models with the highest average diversity coverage score per the dataset, then ensemble their outputs, generating B/2B/2 answers per model. The selected model pairs are respectively: (Llama-3.1-8B, Llama-3.3-70B), (Qwen3-14B, Llama-3.1-8B), (OLMo-2-0425-1B, OLMo-2-1124-7B).
* •

  Top model per query. For each query, we select the model with highest diversity coverage.
  This represents the oracle performance of always choosing the best LLM per given query. We also report the performance of randomly choosing a model per query (Random model per query) as a baseline method.

Table [2](#S3.T2 "Table 2 ‣ 3.2 Oracle experiment: how much does picking the best model(s) per query improve? ‣ 3 A pilot study on ensembling models to maximize diversity coverage ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity") shows that query-level model selection (*Top model per query*) is consistently the best strategy among all three datasets. The gap increases as questions become more open-ended (on *NB-Curated* and *NB-WildChat*). For *Simple Questions*, using one best single LLM (*Top overall model*) can recover 96.9%96.9\% of the ground truth targets. Open-ended questions, however, are more challenging, and choosing the best model per query yields non-trivial gains. This is evidenced by results on *NB-Curated*, where selecting the top model per query (59.6%59.6\%) yields a 27%27\% relative improvement over the second-best baseline (47.0%47.0\%), and on *NB-WildChat* where the improvement is 29%29\%.

## 4 Learning to ensemble multiple models for diverse outputs

Oracle routing significantly improves diversity coverage, but it is costly as we need to sample and evaluate outputs from all candidate LLMs. This motivates us to train a router to predict the most promising model without sampling the entire answer sets from all models.

### 4.1 Router

#### Problem setting

Given a query qq and a suite of models M={m1,m2,⋯​mn}M=\{m\_{1},m\_{2},\cdots m\_{n}\}, a router ranks them, by div−cov⁡(q,Ai)\operatorname{div-cov}(q,A\_{i}) where AiA\_{i} is the generated answer set from mim\_{i} for query qq for some budget BB. The oracle model index for qq is defined as
i∗=arg⁡maxi⁡div−cov⁡(q,A(i))i^{\*}=\arg\max\_{i}\operatorname{div-cov}(q,A^{(i)}). Such index ij∗i\_{j}^{\*} for each query qjq\_{j} consist of the router training data 𝒟={(qj,ij∗)}\mathcal{D}=\{(q\_{j},i\_{j}^{\*})\}.

#### Classification Objectives

We compare two classification formulation for the router:

* •

  |ℳ||\mathcal{M}|-way classification:
  the router is a single classifier
  rθ:𝒬→{1,…,|ℳ|}r\_{\theta}:\mathcal{Q}\rightarrow\{1,\ldots,|\mathcal{M}|\} which predicts the oracle best model index ij∗i\_{j}^{\*} for each query qjq\_{j}.
  Let rθ​(q)ir\_{\theta}(q)\_{i} denote the predicted probability of selecting model mim\_{i},
  we train the router with cross-entropy loss:
  ℒmulti=𝔼(qj,ij∗)∼𝒟​[−log⁡rθ​(qj)ij∗].\mathcal{L}\_{\text{multi}}=\mathbb{E}\_{(q\_{j},i\_{j}^{\*})\sim\mathcal{D}}\left[-\log r\_{\theta}(q\_{j})\_{i\_{j}^{\*}}\right].
* •

  Binary classification: For each LLM mim\_{i}, we derive a binary training dataset
  𝒟(i)={(qj,yj(i))}\mathcal{D}^{(i)}=\{(q\_{j},y\_{j}^{(i)})\} from 𝒟\mathcal{D}, where
  yj(i)=𝟙​[i=ij∗]y\_{j}^{(i)}=\mathbbm{1}[i=i\_{j}^{\*}] indicates whether mim\_{i} is the oracle best model for query qjq\_{j}.
  We then train a binary classifier
  rθ(i):𝒬→[0,1]r\_{\theta}^{(i)}:\mathcal{Q}\rightarrow[0,1]
  to predict this label using binary cross-entropy loss.
  At inference time, the router evaluates all binary classifiers {rθ(i)​(q)}i=1|ℳ|\{r\_{\theta}^{(i)}(q)\}\_{i=1}^{|\mathcal{M}|} and selects the model with the highest predicted score:
  arg⁡maxi=1|ℳ|⁡rθ(i)​(q)\arg\max\_{i=1}^{|\mathcal{M}|}r\_{\theta}^{(i)}(q).

#### Query encoding

We experiment with two input featurizations:
(1) infly/inf-retriever-v1 (infly-ai\_2025), a retriever fine-tuned from Qwen-2-7B for information retrieval tasks. We refer to it as model-agnostic encodings (agn).
(2) Model hidden states: we encode the query using each model mim\_{i} and extract the representation from the final layer’s last hidden state. We hypothesize that this representation encodes rich information on how the model decodes its outputs. We refer to it as model-specific encodings (spec).

### 4.2 Experiment settings

#### Training and evaluation data

We split the 1,0001,000 *NB-WildChat*  prompts from (zhang2025noveltybench) into train, validation and test sets containing 70%, 10% and 20% of the data respectively. We conduct out-of-domain evaluation on *NB-Curated* questions.

#### Evaluation metrics

Diversity coverage jointly measures the diversity and quality of the generated answer set.
To disentangle the effect, we additionally report metrics that measure each aspect.
Quality (Qual) measures the average quality score across all sampled answers:
1|A|​∑a∈Aquality⁡(q,a).\frac{1}{|A|}\sum\_{a\in A}\operatorname{quality}(q,a).
Uniqueness (Unq) measures the number of semantically non-equivalent answers:
|Ad||A\_{d}|.
Unique Quality (Unq Qual) measures the average quality score over unique answers only:
1|Ad|​∑a∈uniq​(q,A)quality⁡(q,a).\frac{1}{|A\_{d}|}\sum\_{a\in{\text{uniq}(q,A)}}\operatorname{quality}(q,a). Together, these metrics reveal whether improvements in cumulative diversity arise from generating more distinct answers, improving answer quality, or both.

#### Baselines

We consider several non-routing baselines. For a fair comparison with our trained routers, we restrict these methods to access only training-set labels and evaluate them on the test set.
We implement baselines from Section [3.2](#S3.SS2 "3.2 Oracle experiment: how much does picking the best model(s) per query improve? ‣ 3 A pilot study on ensembling models to maximize diversity coverage ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity"): *Top overall*, *Top two overall*, *Random model per query*.
We also include a *Frequency* baseline, where models are sampled proportional to their frequency of reaching highest diversity coverage.
We additionally compute *Top model per query* and *Top two models per query* as oracle performance on diversity coverage, using ground-truth labels on the test set. Specifically, *Top model per query* is implemented by selecting the best model per query.
*Top two models per query* are the best pair over all model combinations. If two models are selected, we take half from each model.

#### Router Models

We implement three types of router models, and describe their implementation details in Appendix [E](#A5 "Appendix E Router implementation details ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity").

* •

  KNN (fix1985discriminatory) This is a simple, non-parametric classifier, where the predictions are obtained from K nearest neighbours from the training data, K∈1,5K\in{1,5}.
* •

  BERT (devlin2019bert) Following other routing literature(Ong2024RouteLLMLT; Zhang2025RouterR1TL), we fine-tune BERT with a classification head which makes a selection over the |ℳ||\mathcal{M}| models following.444We did not experiment with implementing |ℳ||\mathcal{M}| BERT models for binary classification given that fine-tuning BERT is computationally more expensive than fine-tuning the 2-layer MLP router.
* •

  MLP We report results for training |ℳ||\mathcal{M}| binary MLP classifiers and training one MLP classifier for |ℳ||\mathcal{M}|-way classification. We report results for (1) using inf-retriever to encode the query for all classifiers: Binary MLP (agn) and M-way MLP (agn), (2) using the candidate model mim\_{i}’s last layer hidden states to encode the query for the respective classifier: Binary MLP (spec) and M-way MLP (spec).

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | NB-WildChat | | | | NB-Curated (OOD) | | | |
| #Unq | Qual | Unq Qual | Cov. | #Unq | Qual | Unq Qual | Cov. |
| Top overall | 42.6 | 3.0 | 2.9 | 23.8% | 35.4 | 6.0 | 5.7 | 38.6% |
| Frequency | 33.1 | 3.8 | 3.6 | 21.0% | 28.2 | 7.2 | 7.1 | 39.6% |
| Random model per query | 27.8 | 3.7 | 3.6 | 18.1% | 27.8 | 7.2 | 7.0 | 37.5% |
| Top model per query (oracle) | 38.8 | 4.5 | 4.4 | 33.0% | 30.3 | 7.6 | 7.4 | 59.6% |
| KNN (N=1) | 34.3 | 3.7 | 3.6 | 23.1% | 28.2 | 7.3 | 7.1 | 39.7% |
| KNN (N=5) | 34.9 | 3.8 | 3.7 | 24.1% | 29.8 | 7.3 | 7.1 | 40.2% |
| M-way BERT | 40.3 | 3.3 | 3.2 | 24.4% | 35.0 | 6.3 | 6.2 | 40.3% |
| M-way MLP (agn) | 35.1 | 3.9 | 3.8 | 25.3% | 30.1 | 7.6 | 7.5 | 40.3% |
| M-way MLP (spec) | 39.3 | 3.5 | 3.4 | 25.9% | 34.6 | 6.3 | 6.1 | 40.2% |
| Binary MLP (agn) | 38.4 | 3.5 | 3.4 | 25.7%∗∗ | 32.8 | 7.1 | 7.0 | 40.7%∗∗ |
| Binary MLP (spec) | 38.1 | 3.6 | 3.5 | 26.3%∗∗ | 30.8 | 7.0 | 6.8 | 39.3%ns |

Table 3: A per-query router selects over 18 models to maximize diversity coverage (Cov.).
We train our best MLP router for 5 runs with random seeds to compute statistical significance for our best system (bolded) against *Top overall*, ∗∗ indicating significantly better and ns indicating not significant.

## 5 Results

### 5.1 Performance Evaluation

We report performances in Table [3](#S4.T3 "Table 3 ‣ Router Models ‣ 4.2 Experiment settings ‣ 4 Learning to ensemble multiple models for diverse outputs ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity").555We also report accuracy (i.e., how frequently it predicted ground truth best model) in Table [6](#A7.T6 "Table 6 ‣ Appendix G Router Performance ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity") in Appendix.
*Top overall* is the best-performing non-routing baseline for in-domain evaluation on *NB-WildChat*. This indicates that the LLM chosen from training labels maintains strong diversity coverage on the test set.
*Frequency* baseline generalizes better to out-of-domain *NB-Curated* questions.
KNN routers yield only marginal improvement. MLP-based routers outperform other baselines. Specifically,
binary routers with model-specific query encodings bring the greatest gains (26.3%26.3\%), surpassing the *Top overall* baseline (23.8%23.8\%). On MLP classifiers, model-specific query encodings (spec) provide more useful information than model-agnostic encoding (agn), but show worse generalization.

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | NB-WildChat | | | | NB-Curated (OOD) | | | |
| #Unq | Qual | Unq Qual | Cov. | #Unq | Qual | Unq Qual | Cov. |
| Top 2 overall | 39.1 | 3.4 | 3.2 | 23.8% | 31.6 | 6.7 | 6.3 | 38.3% |
| Top 2 per query | 40.7 | 4.5 | 4.5 | 35.8% | 41.3 | 7.7 | 7.6 | 62.6% |
| Router | 38.4 | 3.8 | 3.6 | 26.7%∗∗ | 32.3 | 7.1 | 6.8 | 42.2%∗∗ |

Table 4: Performance of ensembling two models per query. We report the performance of the best single model router (Binary MLP (spec)) in [Table 3](#S4.T3 "Table 3 ‣ Router Models ‣ 4.2 Experiment settings ‣ 4 Learning to ensemble multiple models for diverse outputs ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity") by ensembling the top 2 models ranked by the prediction scores. ∗∗ indicating significantly better compared to *Top 2 overall*.

![Refer to caption](/html/2604.02319/assets/x2.png)


Figure 2: Scaling training data improves router performance on *Infinity-Chat*.

#### Router trained to select single model can be used to ensemble outputs from two models which provides further gains.

We observe consistent gains when using our trained router (Binary MLP(spec)) to select two models, as presented in Table [4](#S5.T4 "Table 4 ‣ 5.1 Performance Evaluation ‣ 5 Results ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity") both in-domain (26.41%26.41\% vs. 23.8%23.8\%) and out-of-domain. Moving from *Top overall* to *Top 2 overall*, the best diversity coverage of the non-routing baseline does not improve. But the oracle (*Top per query*) stably increases from one to two models. We show that our best routers are significantly better than *Top overall* and *Top two overall* baseline cross 5 checkpoints trained under different random seeds. We further discuss how the number of model selected affect the answer diversity in Section [H.2](#A8.SS2 "H.2 Discussions: Other configurations/hyperparameters that we can vary ‣ Appendix H Discussion ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity").

#### Scaling training data size consistently produces a better router.

Would training on a larger data set improve performance? On *Infinity-Chat*, we show in Figure [2](#S5.F2 "Figure 2 ‣ 5.1 Performance Evaluation ‣ 5 Results ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity") that router performance increases steadily with training data sizes varying from 500, 1k to 2k. We further find that training also scales on *NB-WildChat* and can incur generalization across the two datasets in Appendix [F](#A6 "Appendix F Scaling router training data ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity").

### 5.2 Efficiency Evaluation

In Figure [3](#S5.F3 "Figure 3 ‣ 5.2 Efficiency Evaluation ‣ 5 Results ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity"), we show the inference time efficiency of generating an answer set of various methods on *NB-WildChat*. We use 2H200 GPUs for answer sampling666We assume no parallelization in sampling generations. If two models are selected, the process is performed sequentially (i.e. model by model). and 1 H200 GPU for diversity coverage calculation. We compare the latency of three methods: Top (*Top overall*), Router, which is the Binary MLP classifier (spec), and Oracle (*Top model per query*).

![Refer to caption](/html/2604.02319/assets/x3.png)


Figure 3: Efficiency analysis comparing the time (seconds per query) across routing (Router), *Top overall* (Top) and *Top model per query* (Oracle). We include routing to one model per query and routing to 2 models per query.
Samplen denotes sample nn answers. Oracle incurs the highest cost, as the routing requires exhaustively comparing all candidate models or model pairs.

Inferencing with our router is about 2-3×2\text{-}3\times slower than the Top baseline. The routing itself is not very costly, yet sampling becomes more expensive as the router often directs to a bigger LLM than the top baseline model (OLMo-2-0425-1B).

Oracle setting, while showing the strongest performance, is also much more expensive, introducing up to 19×19\times computation overhead compared to our router. This is because its routing involves brute-force computing diversity coverage for all models to find the best candidate per query.
In contrast, our router introduces only a fixed overhead that does not scale with the number of selected models.

## 6 Discussions: Different Prompt Templates

|  |  |  |  |
| --- | --- | --- | --- |
| Method | Prompt Template | | |
| Gen 1 | Gen 2 | Gen All |
| Top overall | 18.5% | 19.7% | 23.8% |
| Random | 9.9% | 13.2% | 18.1% |
| Frequency | 15.6% | 17.1% | 21.0% |
| Oracle (G-1) | \cellcolorLightGrey25.6% | \cellcolorLightGrey!4022.3% | \cellcolorLightGrey!4020.4% |
| Oracle (G-2) | \cellcolorLightGrey!4019.5% | \cellcolorLightGrey28.3% | \cellcolorLightGrey!4021.0% |
| Oracle (G-All) | \cellcolorLightGrey!4014.7% | \cellcolorLightGrey!4018.8% | \cellcolorLightGrey33.0% |
| Router (G-1) | \cellcolorLightGrey19.1% | \cellcolorLightGrey!4020.4% | \cellcolorLightGrey!4014.4% |
| Router (G-2) | \cellcolorLightGrey!4019.7% | \cellcolorLightGrey21.6% | \cellcolorLightGrey!4019.0% |
| Router (G-All) | \cellcolorLightGrey!4014.8% | \cellcolorLightGrey!4018.1% | \cellcolorLightGrey26.2% |

Figure 4: Div-Cov (%) results on *NB-WildChat* with various prompting strategies. Training the router under each prompting strategy (in domain and out-of-domain evaluation).

Large amount of work in diversity has focused on improving the prompt, while throughout this paper we used a fixed prompt template to sample answers and compute diversity coverage. In this last section, we explore two alternative prompt templates, with the exact prompts provided in Appendix [B](#A2 "Appendix B Prompts ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity"):

* •

  Generate one (G-1): to produce one random answer for the given question.
* •

  Generate two (G-2): to provide two different answers for given question.
* •

  Generate all (G-All): to list all possible answers sequentially. This is our default.

Table [7](#A8.T7 "Table 7 ‣ H.1 Different Prompt Templates ‣ Appendix H Discussion ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity") summarizes the results. We use the same baselines as in Table [3](#S4.T3 "Table 3 ‣ Router Models ‣ 4.2 Experiment settings ‣ 4 Learning to ensemble multiple models for diverse outputs ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity") in the first block. Comparing across three prompts, we find our default prompt (G-All) overall achieves the highest performance,as shown by the diversity scores (Cov.) in the first block.

In the second and third block, we report the oracle (*Top model per query*) and router results. We use our best router (Binary MLP(spec)) for the experiment.
Router (X) is a router trained under prompt type X. Oracle (X) denotes that we always use ground truth labels derived by sampling with prompt X as predictions. Training a router improves diversity for all prompts, as all routers beat their *Top overall* baselines. However, we see little generalization in both oracle and trained router across prompts. For instance, when generating with G-1 prompt, Oracle model chosen for the G-All prompt performs worse than baselines under G-1 prompt. Moreover, larger gains are observed when routing under better prompts.
You can find more detailed comparison in Appendix [H.1](#A8.SS1 "H.1 Different Prompt Templates ‣ Appendix H Discussion ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity").

#### Degrading Answer Quality While Listing Multiple Answers

Should we always use G-All prompt?
Figure [5](#S6.F5 "Figure 5 ‣ Degrading Answer Quality While Listing Multiple Answers ‣ 6 Discussions: Different Prompt Templates ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity") plots average answer quality under two prompt strategies (G-1 and G-All). For generate-all prompt, we plot the quality of answers at different location within the same generation. For generate-one prompt producing one answer per generation, the quality is plotted as one dashed line.
We find two trends: (1) G-1 prompt consistently generates answers with higher average answer quality than G-All prompt and (2) In G-All prompt, as the generation continues, the answer quality decreases and the variance of quality scores increases. Therefore, when individual answer quality is more important, G-1 prompt, while harder to elicit diverse answers, can be more appropriate.

![Refer to caption](/html/2604.02319/assets/x4.png)


Figure 5: Generate-one prompt has higher answer quality. Under the generate-all prompts, as more answers are listed, the quality decreases with large variations if using generate-all prompt.

## 7 Related Work

#### Improving output diversity

Concerns about the output diversity of LLMs (padmakumar2023does; anderson2024homogenization; west2025base) promoted two categories of solutions: methods that modify model weights (lanchantin2025diverse; chung2025modifying; sorensen2025spectrum; puri2026reachingmoderldistributional) and inference methods (welleck2024decoding; levy-etal-2023-diverse; meister2024benchmarking; xiao2025role; kambhatla2022surfacing; santurkar2023whose; hayati-etal-2024-far; wang2025multilingual).
A suite of work proposes advanced prompting strategies, such as denial prompting (lu2024benchmarking), probabilistic prompting (wong2024simplestrat), and verbalized sampling (zhang2025verbalized). All of these methods focus on improving the diversity of a single model, whereas we study a multi-LLM setting.

#### Routers for LLMs

Researchers find that looping in multiple models is often better than sticking to one (Jiang2023LLMBlenderEL; feng-etal-2024-modular; feng2025llmdroolsmultillmcollaboration; feng2025heterogeneousswarmsjointlyoptimizing; feng2026mocoonestopshopmodel; feng2026singlemultievolutionloopselfimproving).
Building on this insight, many works train a router that selects among multiple LLMs to achieve better task performance (Jiang2023LLMBlenderEL; Zhang2025RouterR1TL; lu-etal-2024-routing) or efficiency (Chen2024RouterDCQR; Ding2024HybridLC; Ong2024RouteLLMLT; Zhang2025RouterR1TL). Simple methods (Ding2024HybridLC; Ong2024RouteLLMLT) demonstrate the effectiveness of routing by switching between a stronger and a weaker model, which balances cost and quality. Other works (Jiang2023LLMBlenderEL; lu-etal-2024-routing) train routers with many top performing LLMs to leverage their complementary expertise.
All existing methods are proposed to enhance the end performance measured within a single generation per question.
However, none of the above discusses how routing can benefit the diversity and quality of a set of derived answers.
To the best of our knowledge, we are the first to propose a router to promote diversity coverage by harnessing the complementary efforts from heterogeneous models.

## 8 Conclusion

In this paper, we study mixing outputs from multiple LLMs as a strategy to improve response diversity. We first formalize diversity as the coverage of high-quality responses and propose unified evaluation metrics that apply to both finite and open-ended answer spaces. To optimize these metrics, we introduce a router that dynamically selects the most suitable LLM(s) for each query, showing improved performance. Further scaling the training data consistently improves the router.
We make few simplifying assumptions: (1) when two models are selected, their outputs are mixed in equal proportions; and (2) only one or two models are used per query. Future research is encouraged to relax these limitations and explore efficiency-aware routing.

## Acknowledgments

This work was supported in part through the NYU IT High Performance Computing resources, services, and staff expertise. The work is partially funded by NSF CAREER award 2443271.

## Appendix A The distribution of most diverse models

We attach in Figure [6](#A1.F6 "Figure 6 ‣ Appendix A The distribution of most diverse models ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity") the freauency of each model being the best model if threshold set to 10%10\%.

![Refer to caption](/html/2604.02319/assets/x5.png)


Figure 6: Frequency(%) of each LLM being the most diverse model.A model is only considered to be the best model if its diversity coverage is 10%10\% higher than the second best candidate. Queries with no meaningful gap are labeled as “No dominant single models”. On *NB-WildChat*, there is no model that consistently dominates all queries. On *Simple Questions*, all models have similar diversity coverage.

## Appendix B Prompts

Prompt for generate one, simple questions

Output a randomly selected day of the week.

```
Output only the day between two curly braces,
like this: {day}.
Don’t output code.
```



Figure 7: Prompt, generate one, simple questions



Prompt for generate two, simple questions

Output two different randomly selected days of the week.

```
Output only the days between curly braces separated by a comma,
like this: {answer_1,answer_2}.
Don’t output code.
```



Figure 8: Prompt, generate two, simple questions



Prompt for generate all, simple questions

Output all days of the week.

```
Output only the days between two curly braces,
like this: {day_1, day_2, ...}.
Don’t output code.
```



Figure 9: Prompt, generate all, simple questions

Prompt for generate one, open-ended questions

I am working on a memoir of a computer science PhD student who worked on machine translation in the 1990s. Suggest a single title and nothing else.

```
Please use the following format:
{
    {
        "answer-id": 1,
        "content": "Your answer here"
    },
}
```



Figure 10: Prompt, generate one, open-ended questions



Prompt for generate two, open-ended questions

I am working on a memoir of a computer science PhD student who worked on machine translation in the 1990s. Suggest a single title and nothing else. Give me two different suggestions.

```
Please use the following format:
{
    {
        "answer-id": 1,
        "content": "Your answer here"
    },
    {   "answer-id": 2,
        "content": "Your answer here"
    }
}
```



Figure 11: Prompt, generate two, open-ended questions



Prompt for generate all, open-ended questions

I am working on a memoir of a computer science PhD student who worked on machine translation in the 1990s. Suggest a single title and nothing else. List all the possible answers you can think of.

```
Please use the following format:
{
    {    "answer-id": 1,
         "content": "Your answer here"
    },
    {    "answer-id": 2,
         "content": "Your answer here"
    },
    ...
}
```



Figure 12: Prompt, generate all, open-ended questions

Prompt\_verbalized\_all, simple questions

Output all days of the week.

```
For each output, also provide a numeric probability of sampling that output.
Please sample at random from the full distribution.
Output only the days and probabilities between two curly braces, like this:
{(day_1,probability_1), (day_2,probability_2) ...}. Don’t output code."
```



Figure 13: Prompt\_verbalized\_all, simple questions



Prompt\_verbalized\_all, open-ended questions

I am working on a memoir of a computer science PhD student who worked on machine translation in the 1990s.
List all the possible answers you can think of. For each answer, also provide a numeric probability of sampling that answer.

```
Please use the following format:
{
    {
        \"answer-id\": 1,
        \"content\": \"Your answer here\",
        \"probability\": \"The probability of this answer\"
    },
    {
        \"answer-id\": 2,
        \"content\": \"Your answer here\",
        \"probability\": \"The probability of this answer\"
    },
    ...
}
```



Figure 14: Prompt\_verbalized\_all, open-ended questions

System prompt for system\_vanilla, simple questions and open-ended questions

You are a helpful assistant. For each query, please generate all possible responses, each within a separate <response> tag. Responses should each include a <text>.


Figure 15: System\_vanilla, simple questions and open-ended questions



System prompt for system\_verbalized\_all, simple questions and open-ended questions

You are a helpful assistant. For each query, please generate all possible responses, each within a separate <response> tag. Responses should each include a <text> and a numeric <probability>. Please sample at random from the full distribution.


Figure 16: System\_verbalized\_all, simple questions and open-ended questions

## Appendix C Diversity coverage calculation details on open-ended questions

Here we discuss the details of how we evaluate the quality and diversity of answers to open-ended questions. We follow the exact procedure to partition the answer set and calculate. the quality scores in zhang2025noveltybench. To determine semantic equivalence, we apply their equivalence classifier (used in line 5 in algorithm below)
to all pairs of generations and retain a subset with no mutually equivalent pairs (see Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix C Diversity coverage calculation details on open-ended questions ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity") below). This classifier is finetuned with 11001100 pairs of human annotated generations conditioned on prompts sampled from *NB-Curated* and *NB-WildChat*.
We then score the quality of each answer a′∈Ad{a^{\prime}}\in{A\_{d}}, following their process : the score is first derived by a reward model and later mapped to {1,…,10}\{1,\ldots,10\}777We use Skywork-Reward-Gemma-2-27B-v0.2 model (liu2024skywork) as the reward model and the equivalence classifier released by  zhang2025noveltybench at <https://huggingface.co/yimingzhang/deberta-v3-large-generation-similarity).>. Their mapping is calibrated by aligning the distribution of reward model scores (from 2,400 MT-Bench generations) with GPT-4–judged quality scores, using thresholds to map reward values to the 1–10 scale.

Algorithm 1  Extract semantically nonequivalent generations

1:Sampled generations A={a1,…,aB}A=\{a\_{1},\dots,a\_{B}\}

2:Equivalence classifier Eq​(⋅,⋅)\textsc{Eq}(\cdot,\cdot)

3:Set of semantically nonequivalent generations AdA\_{d}

4:Ad←∅A\_{d}\leftarrow\emptyset

5:for i=1i=1 to BB do

6:  i​s​\_​d​u​p​l​i​c​a​t​e←Falseis\\_duplicate\leftarrow\textbf{False}

7:  for each a′∈Ada^{\prime}\in A\_{d} do

8:   s←Eq​(ai,a′)s\leftarrow\textsc{Eq}(a\_{i},a^{\prime}) ⊳\triangleright Similarity score

9:   if s>τs>\tau then

10:     i​s​\_​d​u​p​l​i​c​a​t​e←Trueis\\_duplicate\leftarrow\textbf{True}

11:     break

12:   end if

13:  end for

14:  if not i​s​\_​d​u​p​l​i​c​a​t​eis\\_duplicate then

15:   Ad←Ad∪{ai}A\_{d}\leftarrow A\_{d}\cup\{a\_{i}\}

16:  end if

17:end for

18:return AdA\_{d}

## Appendix D Decoding settings

We set target number (NN) of answers to 5050 if not otherwise stated. The temperature and top−p-p are fixed to be 1.01.0 and 1.01.0 respectively. The max tokens is set to be 40964096. We use 2 H200 GPUs for all models. The batch size is 6464. We repeat the sampling process until NN answers are collected. The inference time varies by model sizes and familities. We disable the thinking mode for Qwen models.

## Appendix E Router implementation details

We use Adam optimizer, a learning rate of 1​e−31e-3. For BERT classifier, we use the AdamW optimizer with a learning rate of 2​e−52e-5. During training, we perform a grid search over options of {soft, one-hot} labels, weight decay and hidden dimensions. Routers are selected based on the best scores on the validation set. We experiment with soft labels and one-hot labels to provide the training signals. The soft labels are drawn by normalizing the diverse coverage scores against the most diverse model for this query.
One-hot labels are derived by 𝟙​[mi=mj∗]\mathbbm{1}[m\_{i}=m\_{j}^{\*}]. We find that soft labels work best with M-way MLP classifier while one-hot labels are best for Binary MLP classifier.

## Appendix F Scaling router training data

Specifically, we experiment with training the router on 500500 and 11K samples from *NB-WildChat* , and 500500, 11K, and 22K samples from *Infinity-Chat* (Jiang2025ArtificialHT). The results are shown in Table [5](#A6.T5 "Table 5 ‣ Appendix F Scaling router training data ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity"). Increasing *NB-WildChat*  training data from 500500 to 11K improves diversity coverage on the *NB-WildChat* test set, though it does not transfer to *Infinity-Chat*. In contrast, scaling *Infinity-Chat*  data from 500500 to 22K steadily improves performance on both the *Infinity-Chat*  test set and the *NB-WildChat*  test set, indicating stronger generalization. Finally, jointly training on a combination of *NB-WildChat*  and *Infinity-Chat*  further improves performance, slightly surpassing the best router (26.4%26.4\% vs. 26.3%26.3\%) trained on 11K *NB-WildChat*  data in Table [3](#S4.T3 "Table 3 ‣ Router Models ‣ 4.2 Experiment settings ‣ 4 Learning to ensemble multiple models for diverse outputs ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity").

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  |  | Evaluation Data | |
| Method |  |  | *NB-WildChat* | *Infinity-Chat* |
| Random |  |  | 18.13% | 18.24% |
| Top Overall |  |  | 23.83% | 23.13% |
| Oracle |  |  | 33.04% | 30.50% |
|  | Training Data | Size |  |  |
| Router | *NB-WildChat* | 500 | 25.28% ±0.28%\scriptstyle\pm 0.28\% | 22.58% ±0.30%\scriptstyle\pm 0.30\% |
| Router | *NB-WildChat* | 1K | 26.27% ±0.13%\scriptstyle\pm 0.13\% | 22.58% ±0.39%\scriptstyle\pm 0.39\% |
| Router | *Infinity-Chat* | 500 | 23.98% ±0.67%\scriptstyle\pm 0.67\% | 22.54% ±0.60%\scriptstyle\pm 0.60\% |
| Router | *Infinity-Chat* | 1K | 24.95% ±0.28%\scriptstyle\pm 0.28\% | 23.54% ±0.12%\scriptstyle\pm 0.12\% |
| Router | *Infinity-Chat* | 2K | 25.13% ±0.36%\scriptstyle\pm 0.36\% | 23.78% ±0.16%\scriptstyle\pm 0.16\% |
| Router | *NB-WildChat*  and *Infinity-Chat* | 1K and 1K | 26.05% ±0.32%\scriptstyle\pm 0.32\% | 23.36% ±0.23%\scriptstyle\pm 0.23\% |
| Router | *NB-WildChat*  and *Infinity-Chat* | 1K and 2K | 26.40% ±0.21%\scriptstyle\pm 0.21\% | 23.55% ±0.10%\scriptstyle\pm 0.10\% |

Table 5: Router performance (diversity coverage) steadily improves with more training data. We report the average and variance of 5 training runs with different random seeds.

## Appendix G Router Performance

|  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | NB-WildChat | | | | | NB-Curated (OOD) | | | | |
| Acc | #U | Q | UQ | Cov. | Acc | #U | Q | UQ | Cov. |
| Top Overall | 19.5% | 42.6 | 3.0 | 2.9 | 23.8% | 3.4% | 35.4 | 6.0 | 5.7 | 38.6% |
| Random M / Q | 5.9% | 27.8 | 3.7 | 3.6 | 18.1% | 5.6% | 27.8 | 7.2 | 7.0 | 37.5% |
| Frequency | 12.0% | 33.1 | 3.8 | 3.6 | 21.0% | 9.0% | 28.2 | 7.2 | 7.1 | 39.6% |
| Top M / Q (oracle) | 100% | 38.8 | 4.5 | 4.4 | 33.0% | 100% | 30.3 | 7.6 | 7.4 | 59.6% |
| 11NN | 16.5% | 34.3 | 3.7 | 3.6 | 23.1% | 5.6% | 28.2 | 7.3 | 7.1 | 39.7% |
| 55NN | 17.5% | 34.9 | 3.8 | 3.7 | 24.1% | 12.4% | 29.8 | 7.3 | 7.1 | 40.2% |
| M-way BERT | 22.0% | 40.3 | 3.3 | 3.2 | 24.4% | 11.2% | 35.0 | 6.3 | 6.2 | 40.3% |
| M-way MLP(agn) | 24.0% | 35.1 | 3.9 | 3.8 | 25.3% | 12.4% | 30.1 | 7.6 | 7.5 | 40.3% |
| M-way MLP(spec) | 27.0% | 39.3 | 3.5 | 3.4 | 25.9% | 5.6% | 34.6 | 6.3 | 6.1 | 40.2% |
| Binary MLP (agn) | 23.9% | 38.4 | 3.5 | 3.4 | 25.7%∗∗ | 10.8% | 32.8 | 7.1 | 7.0 | 40.7%∗∗ |
| Binary MLP (spec) | 23.9% | 38.1 | 3.6 | 3.5 | 26.3%∗∗ | 13.3% | 30.8 | 7.0 | 6.8 | 39.3%ns |

Table 6: A per-query router selecting over 18 models to maximize diversity coverage (Cov.).#U denotes number of unique outputs, Q denotes average quality, and UQ denotes quality of unique outputs. Accuracy(Acc) measures how frequently the router predicts the oracle model(ground truth target). Random M / Q denotes random model per query, and Top M / Q denotes top model per query.

## Appendix H Discussion

### H.1 Different Prompt Templates

Prompting methods affect generation diversity (zhang2025verbalized)
. We show that model ensembling is effective for answers generated by sequential prompting: model are asked to generate as many distinct answers in one generation, where the latter answers are dependent of previous answers. Does it also work for other prompting methods? We extend the study in section [3](#S3 "3 A pilot study on ensembling models to maximize diversity coverage ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity") to compare three different prompt types888Please refer to Appendix [B](#A2 "Appendix B Prompts ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity") for the exact prompts.:

* •

  Generate one: The model is prompted to produce one random answer for the given question.
* •

  Generate two: The model is prompted to provide two possible and different answers for given question.
* •

  Generate all (our default setting): The model is prompted to list out all possible answers sequentially.

|  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | Gen 1 | | | Gen 2 | | | Gen All | | |
| #Unq | Quality | Cov. | #Unq | Quality | Cov. | #Unq | Quality | Cov. |
| Random | 13.7 | 4.9 | 9.9% | 18.3 | 4.5 | 13.2% | 27.8 | 3.7 | 18.1% |
| Frequency | 25.4 | 3.9 | 15.6% | 25.2 | 4.1 | 17.1% | 33.1 | 3.8 | 21.0% |
| Top Overall | 31.8 | 3.2 | 18.5% | 34.8 | 3.1 | 19.7% | 42.6 | 3.0 | 23.8% |
| Oracle (G-1) | \cellcolorLightGrey 32.8 | \cellcolorLightGrey4.1 | \cellcolorLightGrey25.6% | \cellcolorLightGrey!4032.1 | \cellcolorLightGrey!403.7 | \cellcolorLightGrey!4022.3% | \cellcolorLightGrey!4031.0 | \cellcolorLightGrey!403.3 | \cellcolorLightGrey!4020.4% |
| Oracle (G-2) | \cellcolorLightGrey!4024.1 | \cellcolorLightGrey!404.3 | \cellcolorLightGrey!4019.5% | \cellcolorLightGrey32.3 | \cellcolorLightGrey4.7 | \cellcolorLightGrey28.3% | \cellcolorLightGrey!4031.6 | \cellcolorLightGrey!403.3 | \cellcolorLightGrey!4021.0% |
| Oracle (G-All) | \cellcolorLightGrey!4016.4 | \cellcolorLightGrey!405.0 | \cellcolorLightGrey!4014.7% | \cellcolorLightGrey!4023.1 | \cellcolorLightGrey!404.6 | \cellcolorLightGrey!4018.8% | \cellcolorLightGrey38.8 | \cellcolorLightGrey4.5 | \cellcolorLightGrey33.0% |
| Router (G-1) | \cellcolorLightGrey33.1 | \cellcolorLightGrey3.2 | \cellcolorLightGrey19.1% | \cellcolorLightGrey!4033.3 | \cellcolorLightGrey!403.3 | \cellcolorLightGrey!4020.4% | \cellcolorLightGrey!4025.8 | \cellcolorLightGrey!402.8 | \cellcolorLightGrey!4014.4% |
| Router (G-2) | \cellcolorLightGrey!4026.5 \cellcolorLightGrey!40 | \cellcolorLightGrey!404.2 | \cellcolorLightGrey!4019.7% | \cellcolorLightGrey30.5 | \cellcolorLightGrey3.9 | \cellcolorLightGrey21.6% | \cellcolorLightGrey!4029.2 | \cellcolorLightGrey!403.2 | \cellcolorLightGrey!4019.0% |
| Router (G-All) | \cellcolorLightGrey!4018.3 | \cellcolorLightGrey!404.7 | \cellcolorLightGrey!4014.8% | \cellcolorLightGrey!4024.4 | \cellcolorLightGrey!404.3 | \cellcolorLightGrey!4018.1% | \cellcolorLightGrey37.5 | \cellcolorLightGrey3.7 | \cellcolorLightGrey26.2% |

Table 7: Training the router under different prompting strategies (in domain and out-of-domain evaluation) on *NB-WildChat*. Router (X) is a router trained under prompt type X. Oracle (X) denotes that we always use ground truth labels of prompt X as predictions.
Training a router improves diversity for all prompts, as all routers beat their Top Overall baselines. However, different prompt templates seem to elicit different levels of diversity in LLMs, as the oracle predictions don’t generalize across prompts.

#### Routing improves diversity for all prompts, yet a router trained on one prompt does not generalize to others.

We ablate the prompting strategies, retrain routers, and evaluate them on all types of prompts. The performance is presented in Table [7](#A8.T7 "Table 7 ‣ H.1 Different Prompt Templates ‣ Appendix H Discussion ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity"). We find that generate all prompt incurs most diversity coverage, as shown by the diversity scores (Cov.) of the random/ oracle baselines. Training a router in-domain consistently improves diversity coverage, yet neither oracle labels nor routers generalize across prompts. Finally, larger gains are observed when routing under better prompts.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Gen 1 | | Gen 2 | | Gen All | |
|  | Cov. | Len | Cov. | Len | Cov. | Len |
| Random | 16.7%16.7\% | 49.349.3 | 22.4%22.4\% | 37.037.0 | 37.5%37.5\% | 17.117.1 |
| Top model | 33.5%33.5\% | 47.747.7 | 35.4%35.4\% | 38.838.8 | 47.0%47.0\% | 22.622.6 |
| Top 2 models | 32.7%32.7\% | 31.631.6 | 36.1%36.1\% | 34.234.2 | 45.6%45.6\% | 24.024.0 |
| Top model per query | 43.6%43.6\% | 37.837.8 | 46.6%46.6\% | 30.330.3 | 59.6%59.6\% | 21.921.9 |

Table 8: Oracle divresity coverage (Cov.) and answer lengths (Len) for different prompt temlates on *NB-Curated* Questions.

![Refer to caption](/html/2604.02319/assets/x6.png)


Figure 17: Average answer quality for responses generated from different prompts on *NB-Curated* questions.

#### Tradeoff of the generate all prompt.

Despite being the best method, there is a trade-off between diversity and quality for generate all. Under the routing setting, we observe in Table [8](#A8.T8 "Table 8 ‣ Routing improves diversity for all prompts, yet a router trained on one prompt does not generalize to others. ‣ H.1 Different Prompt Templates ‣ Appendix H Discussion ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity") that the length of the answers decreases from generate one, generate two to generate all, up to 66%66\% (from 49.349.3 to 17.117.1). Besides, as shown in Table [7](#A8.T7 "Table 7 ‣ H.1 Different Prompt Templates ‣ Appendix H Discussion ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity"), though the number of unique answers sampled increases, the average answer quality deteriorates from generate one, generate two to generate all.
This claim is further supported
by comparing average answer quality among different prompting methods in Figure  [17](#A8.F17 "Figure 17 ‣ Routing improves diversity for all prompts, yet a router trained on one prompt does not generalize to others. ‣ H.1 Different Prompt Templates ‣ Appendix H Discussion ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity"). It shows that generate all has the lowest answer quality while generate one has the highest. These findings hold for models across different sizes and families. Interestingly, a closer look into the answer generation process suggests that answers generated at later positions have worse quality under the sequential generate all prompt in Figure [5](#S6.F5 "Figure 5 ‣ Degrading Answer Quality While Listing Multiple Answers ‣ 6 Discussions: Different Prompt Templates ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity").

### H.2 Discussions: Other configurations/hyperparameters that we can vary

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Ratio | #Unq | Qual | Unq Qual | Cov. |
| Oracle model pair | | | | |
| 0:50 | 42.50 | 4.16 | 3.95 | 35.40% |
| 5:45 | 43.90 | 4.11 | 3.94 | 36.82% |
| 10:40 | 45.00 | 4.13 | 3.99 | 37.56% |
| 15:35 | 45.50 | 4.09 | 3.88 | 37.10% |
| 20:30 | 46.30 | 3.93 | 3.80 | 36.46% |
| 25:25 | 45.30 | 4.00 | 3.87 | 36.40% |
| Top 2 model pair | | | | |
| 0:50 | 36.70 | 3.74 | 3.53 | 24.36% |
| 5:45 | 38.10 | 3.73 | 3.46 | 25.58% |
| 10:40 | 39.20 | 3.58 | 3.34 | 25.60% |
| 15:35 | 40.50 | 3.40 | 3.20 | 25.34% |
| 20:30 | 42.10 | 3.33 | 3.15 | 26.28% |
| 25:25 | 43.60 | 3.26 | 3.12 | 26.98% |
| 30:20 | 44.30 | 3.12 | 3.04 | 26.54% |
| 35:15 | 45.60 | 3.06 | 3.02 | 27.48% |
| 40:10 | 46.20 | 2.99 | 2.98 | 27.62% |
| 45:5 | 46.70 | 2.91 | 2.90 | 27.10% |
| 50:0 | 47.20 | 2.85 | 2.88 | 27.10% |
| avg | 42.75 | 3.27 | 3.15 | 26.36% |
| Random model pair | | | | |
| 0:50 | 35.93 | 3.24 | 3.23 | 20.86% |
| 5:45 | 36.56 | 3.26 | 3.24 | 21.51% |
| 10:40 | 37.19 | 3.23 | 3.21 | 21.72% |
| 15:35 | 37.49 | 3.22 | 3.20 | 21.79% |
| 20:30 | 37.73 | 3.23 | 3.20 | 21.86% |
| 25:25 | 37.72 | 3.22 | 3.19 | 21.89% |

Table 9: Exploring different ratios while varying models per question. Performance is reported on 10 questions sampled from *NB-WildChat*. Top 2 models are olmo-2-0425-1b and olmo-2-0325-32b.



|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Strategy | #Unq | Qual | Unq Qual | Cov. |
| Oracle ratio | 44.80 | 3.66 | 3.55 | 32.58% |
| Overall best (40:10) | 46.20 | 2.99 | 2.98 | 27.62% |
| Half/half | 43.60 | 3.26 | 3.12 | 26.98% |

Table 10: Always use top 2 models ( olmo-2-0425-1b and olmo-2-0325-32b), while varying ratios per question.

#### More flexible proportions of sampling per model

In the previous setting of routing to two models, we fix the sampled answers to be split equally (i.e., if there are two models selected to generate 50 answers, each would contribute to 25 answers). Will a more flexible proportion lead to more diversity?
Under the same setting of sampling 50 answers from two models, we experiment with a set of possible ratios 0.0:1.0, 0.1:0.9, 0.2:0.8, 0.3:0.7, 0.4:0.6, 0.5:0.5(original) to assign the budget between two models. We conduct two experiments: (1) pick a ratio for all the questions, vary model choices (2) fix two models to ensemble (top 2 by individual performance), varying ratios for each question. We present the results in table [9](#A8.T9 "Table 9 ‣ H.2 Discussions: Other configurations/hyperparameters that we can vary ‣ Appendix H Discussion ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity") and table [10](#A8.T10 "Table 10 ‣ H.2 Discussions: Other configurations/hyperparameters that we can vary ‣ Appendix H Discussion ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity") respectively. We find that for oracle/random/top 2 model pairs, different global ratios don’t have much difference in output diversity. If we fix 2 models to ensemble and optimize ratios for each question, the score can be improved over rigid half/half mixing (32.58%32.58\% vs 26.98%26.98\%).

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| NN | #Unq | Qual | Unq Qual | Cov. |
| 1 | 42.50 | 4.16 | 3.95 | 35.40% |
| 2 | 45.30 | 4.00 | 3.87 | 36.08% |
| 3 | 44.00 | 4.04 | 3.86 | 35.72% |
| 4 | 44.20 | 4.13 | 3.79 | 35.96% |
| 5 | 43.80 | 4.13 | 3.79 | 35.74% |
| 6 | 44.30 | 3.93 | 3.58 | 35.04% |
| 7 | 43.40 | 4.08 | 3.76 | 35.12% |
| 8 | 43.30 | 3.94 | 3.70 | 34.02% |
| 9 | 42.90 | 4.02 | 3.73 | 34.00% |
| 10 | 43.10 | 3.92 | 3.65 | 33.36% |
| 11 | 42.40 | 3.87 | 3.66 | 31.82% |
| 12 | 42.40 | 3.80 | 3.56 | 30.92% |
| 13 | 40.40 | 3.83 | 3.65 | 29.56% |
| 14 | 40.10 | 3.80 | 3.63 | 28.88% |
| 15 | 38.80 | 3.70 | 3.58 | 27.38% |
| 16 | 37.90 | 3.66 | 3.53 | 26.10% |
| 17 | 37.30 | 3.73 | 3.54 | 25.66% |
| 18 | 36.60 | 3.67 | 3.56 | 24.32% |

Table 11: Oracle models: fix NN models to select for all questions and vary model choices per question.



|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| NN | #Unq | Qual | Unq Qual | Cov. |
| 1 | 47.20 | 2.85 | 2.88 | 27.10% |
| 2 | 43.60 | 3.26 | 3.12 | 26.98% |
| 3 | 41.80 | 3.21 | 3.06 | 24.90% |
| 4 | 41.90 | 3.31 | 3.07 | 25.40% |
| 5 | 40.60 | 3.47 | 3.27 | 25.42% |
| 6 | 40.70 | 3.53 | 3.34 | 26.60% |
| 7 | 40.90 | 3.59 | 3.42 | 27.00% |
| 8 | 40.00 | 3.65 | 3.54 | 27.26% |
| 9 | 38.40 | 3.85 | 3.60 | 26.48% |
| 10 | 38.90 | 3.84 | 3.64 | 27.20% |
| 11 | 37.70 | 3.81 | 3.52 | 25.70% |
| 12 | 38.50 | 3.62 | 3.30 | 25.00% |
| 13 | 35.20 | 3.70 | 3.48 | 23.22% |
| 14 | 37.20 | 3.57 | 3.35 | 23.96% |
| 15 | 36.40 | 3.60 | 3.44 | 23.98% |
| 16 | 35.80 | 3.67 | 3.48 | 23.82% |
| 17 | 36.40 | 3.38 | 3.28 | 21.98% |
| 18 | 34.90 | 3.57 | 3.52 | 22.50% |

Table 12: Top NN models: fix NN models to select for all questions and vary model choices per question.



|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| NN | #Unq | Qual | Unq Qual | Cov. |
| 1 | 35.93 | 3.24 | 3.23 | 20.86% |
| 2 | 37.72 | 3.22 | 3.19 | 21.85% |
| 3 | 38.63 | 3.23 | 3.17 | 22.38% |
| 4 | 39.03 | 3.21 | 3.13 | 22.60% |
| 5 | 39.11 | 3.22 | 3.13 | 22.89% |
| 6 | 38.89 | 3.29 | 3.18 | 23.36% |
| 7 | 38.53 | 3.34 | 3.22 | 23.58% |
| 8 | 38.33 | 3.37 | 3.26 | 23.70% |
| 9 | 38.35 | 3.44 | 3.32 | 24.08% |
| 10 | 38.05 | 3.47 | 3.34 | 24.26% |
| 11 | 37.82 | 3.46 | 3.32 | 23.66% |
| 12 | 37.45 | 3.49 | 3.36 | 23.80% |
| 13 | 37.44 | 3.53 | 3.39 | 23.70% |
| 14 | 36.93 | 3.54 | 3.41 | 23.49% |
| 15 | 36.46 | 3.58 | 3.44 | 23.49% |
| 16 | 35.94 | 3.59 | 3.45 | 23.32% |
| 17 | 36.66 | 3.63 | 3.51 | 23.92% |
| 18 | 36.60 | 3.67 | 3.56 | 24.32% |

Table 13: Random models: fix NN models to select for all questions and vary model choices per question.



|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Strategy | #Unq | Qual | Unq Qual | Cov. |
| Oracle NN | 44.10 | 3.93 | 3.73 | 34.24% |
| Best overall (N=8N=8) | 40.00 | 3.65 | 3.54 | 27.26% |
| Random NN | 39.23 | 3.53 | 3.35 | 25.25% |

Table 14: Fix model order to ensemle (ranking of individual performance) and vary NN per question.

#### Varying the number of models to ensemble from

In previous experiments, we fix the number of activated models to be 11 (routing the best model per query) or 22 (routing to two best models per query). Will sampling answers from more models, while keeping the total number of answers unchanged, improve diversity? We answer this question by two experiments: (1) fix number of models NN for all, vary selected models per questions (2) fix the order of model to be selected (ranked by individual performance), vary the number NN per question. We present the results in Table [11](#A8.T11 "Table 11 ‣ More flexible proportions of sampling per model ‣ H.2 Discussions: Other configurations/hyperparameters that we can vary ‣ Appendix H Discussion ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity"), Table [12](#A8.T12 "Table 12 ‣ More flexible proportions of sampling per model ‣ H.2 Discussions: Other configurations/hyperparameters that we can vary ‣ Appendix H Discussion ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity"), Table [13](#A8.T13 "Table 13 ‣ More flexible proportions of sampling per model ‣ H.2 Discussions: Other configurations/hyperparameters that we can vary ‣ Appendix H Discussion ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity"), and Table [14](#A8.T14 "Table 14 ‣ More flexible proportions of sampling per model ‣ H.2 Discussions: Other configurations/hyperparameters that we can vary ‣ Appendix H Discussion ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity"). We find that routing to a custom model per question remains the most promising approach (under oracle settings). Routing to two models can offer further gains. But ensembling >> 2 models does not improve output diversity.

#### Scaling the number of candidates and generations

In this paper, we study selecting models from a pool of 18 candidates. However, in a real-world setting, there are hundreds of models users can choose from. Therefore, future work can explore employing a larger pool of LLMs that better harness their complementary strengths of uncovering more diverse answers. Besides, the number of answers to open-ended questions is infinitely large, and future work is encouraged to explore sample sizes beyond 5050.

## Appendix I Extended related work

#### Measuring output diversity

Traditional metrics to measure lexical diversity and text style are based on token and POS n-grams statistics (roemmele2017evaluating; see-etal-2019-massively; tevet-berant-2021-evaluating; meister2023locally) and embedding similarity between candidatespadmakumar2023does. Later works go beyond the distinctness of outputs and also measure the validity of each response.
zhang2024forcing propose to evaluate the diversity of LLMs by calculating the coverage of gold targets and the KL-divergence from the desired distribution. However, providing ground-truth distributions for open-ended questions is non-trivial.
Closely related to our work, zhang2025noveltybench introduce the notion of *user-perceived utility*, which jointly models uniqueness and quality while accounting for user patience.
In this framework, uniqueness is computed by partitioning sampled answers into non-equivalent groups, and answer quality is estimated using reward model scores.
However, this metric penalizes later-generated responses, whereas our goal is to assess how well a set of answers covers the answer space regardless of generation order.

Similarly, sorensen2025spectrum evaluates how well a model covers an open-ended output space using validity and diversity metrics.
However, their evaluation relies on expensive human annotations and thus is only experimented with a single model with four generations per prompt.
In this work, we build on the framework of zhang2025noveltybench and propose diversity coverage, a metric that evaluates how well a set of generated answers covers the valid answer space across many generations and multiple LLMs without requiring additional human supervision.

## Appendix J Verbalized Sampling

Similar to our baselines, Verbalized Sampling is a recent prompting technique that increases LLM output diversity. We decided not to include it in the main experiments since it performs similarly (if not worse than) our generated all baseline. We include the evidence below in Table [15](#A10.T15 "Table 15 ‣ Appendix J Verbalized Sampling ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity") and Table [16](#A10.T16 "Table 16 ‣ Appendix J Verbalized Sampling ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity"):

|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Prompt | Model | Cov. % | | | | | |
| 1 | 10 | 20 | 50 | 100 | 1000 |
| prompt\_vanilla | Llama 8B | 6.096.09 | 47.9647.96 | 66.7066.70 | 88.5688.56 | 92.4492.44 | 96.5196.51 |
| prompt\_verbalized\_all | 6.096.09 | 47.9647.96 | 66.6966.69 | 89.8389.83 | 93.8193.81 | 97.7697.76 |
| system\_vanilla | 6.096.09 | 43.4243.42 | 55.3155.31 | 72.0072.00 | 83.0283.02 | 95.2395.23 |
| system\_verbalized\_all | 6.096.09 | 44.1144.11 | 60.2660.26 | 79.1679.16 | 89.3089.30 | 94.9894.98 |
| prompt\_vanilla | GPT-4o | 6.096.09 | 48.2048.20 | 67.1867.18 | 89.8089.80 | 92.8092.80 | 94.1094.10 |
| prompt\_verbalized\_all | 6.096.09 | 46.6046.60 | 62.2762.27 | 84.2684.26 | 87.1587.15 | 92.2192.21 |
| system\_vanilla | 6.096.09 | 43.8343.83 | 54.7754.77 | 68.1268.12 | 76.3376.33 | 91.2191.21 |
| system\_verbalized\_all | 6.096.09 | 44.4644.46 | 56.0756.07 | 68.5968.59 | 77.6277.62 | 92.0092.00 |

Table 15: Compared results with verbalized sampling on *Simple Questions* generating up to 1,10,20,…​10001,10,20,...1000 answers. Prompt\_vanilla is the existing generate-all prompt. System\_verbalized\_all is the original prompt proposed in verbalized sampling.



|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prompt | Model | Cov. % | | | | | | |
| 1 | 5 | 10 | 20 | 50 | 100 | 200 |
| prompt\_vanilla | Llama 8B | 0.380.38 | 1.671.67 | 3.043.04 | 5.225.22 | 10.0010.00 | 16.1716.17 | 26.1326.13 |
| prompt\_verbalized\_all | 0.370.37 | 1.631.63 | 2.982.98 | 4.994.99 | 9.249.24 | 14.4614.46 | 23.7623.76 |
| system\_vanilla | 0.340.34 | 1.341.34 | 2.422.42 | 4.194.19 | 8.918.91 | 14.8614.86 | 24.5524.55 |
| system\_verbalized\_all | 0.370.37 | 1.571.57 | 2.822.82 | 4.544.54 | 9.059.05 | 15.3115.31 | 25.5325.53 |
| prompt\_vanilla | GPT-4o | 0.420.42 | 1.841.84 | 3.513.51 | 6.306.30 | 11.0111.01 | 16.4716.47 | 24.7824.78 |
| prompt\_verbalized\_all | 0.400.40 | 1.861.86 | 3.293.29 | 5.005.00 | 8.658.65 | 13.0913.09 | 19.6119.61 |
| system\_vanilla | 0.410.41 | 1.571.57 | 2.692.69 | 4.084.08 | 7.307.30 | 11.0511.05 | 16.9416.94 |
| system\_verbalized\_all | 0.400.40 | 1.711.71 | 2.682.68 | 4.164.16 | 7.307.30 | 11.1211.12 | 17.5017.50 |

Table 16: Compared results with verbalized sampling on *NB-Curated* generating up to 1,5,10,…​2001,5,10,...200 responses.Prompt\_vanilla is the existing generate-all prompt. System\_verbalized\_all is the original prompt proposed in verbalized sampling.

## Appendix K Generating diverse outputs out of a single model

![Refer to caption](/html/2604.02319/assets/x7.png)

![Refer to caption](/html/2604.02319/assets/x8.png)

Figure 18: Comparing different prompts for generating diverse outputs. The number of samples (x-axis) vs. the diversity coverage scores (y-axis).
We compare three prompts on *Simple Questions* (top) and *NB-Curated* (bottom) datasets.

![Refer to caption](/html/2604.02319/assets/x9.png)


Figure 19: The number of unique answers on *NB-Curated*. The X-axis is the log of the number of answers generated. Y-axis measures the diversity coverage of all the unique answers divided by the max possible score(generate 200 different good answers).

### K.1 Experiment settings

#### Decoding settings

For each prompting strategy and desired number of answers NN, we repeatedly sample generations from the model until we collect NN answers. For *Simple Questions*, we use N={1,10,20,50,100,1000}N=\{1,10,20,50,100,1000\}. For *NB-Curated*, we use N={1,5,10,20,50,100,200}N=\{1,5,10,20,50,100,200\}. We set the temperature to 1.01.0, top\_p to 1.01.0. The max\_len is set to 2048 by default. For the generate all setting in *NB-Curated*, we extend the max\_len to 4096 because generations can not be finished within 2048 tokens.

### K.2 Results

#### Compare different prompting strategies

Figure [18](#A11.F18 "Figure 18 ‣ Appendix K Generating diverse outputs out of a single model ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity") shows how different prompting strategies affect the diversity of combined answers.
For all models on both datasets, sequential generation enables a lot more answer diversity than parallel methods. With the best prompt, models on *Simple Questions* saturate to more than 90%90\% of coverage rate. This suggests that for easy diversity questions, nearly all models have good knowledge of the full answer space. On *NB-Curated*, answer diversity keeps growing as more generations are inferred. This reveals large diversity potential in uncovering more unique and high-quality responses to open-ended queries.

![Refer to caption](/html/2604.02319/assets/x10.png)


Figure 20: 
Smaller models generate more unique answers. As number of inferred answers increases,
Llama models have better output uniqueness than Qwen, but also have more within-family variance.

#### How does model size affect diversity?

According to Figure [18](#A11.F18 "Figure 18 ‣ Appendix K Generating diverse outputs out of a single model ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity"), most models’ performances are pretty similar on *Simple Questions*, except for the smallest model Qwen 0.6B). On *NB-Curated*, medium-sized models (Llama 8B and Qwen 14B) consistently have higher overall diversity than extremely large or small ones. We hypothesize that these models balance answer distinctness and quality best, therefore achieving the highest diversity performance. Figure [20](#A11.F20 "Figure 20 ‣ Compare different prompting strategies ‣ K.2 Results ‣ Appendix K Generating diverse outputs out of a single model ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity") shows answer uniqueness is inversely proportional to the model sizes. And Figure [17](#A8.F17 "Figure 17 ‣ Routing improves diversity for all prompts, yet a router trained on one prompt does not generalize to others. ‣ H.1 Different Prompt Templates ‣ Appendix H Discussion ‣ No Single Best Model for Diversity: Learning a Router for Sample Diversity") shows answer quality is proportional to model size.
Finally, we also noticed that model rankings are largely unchanged regardless of the number of collected answers.

[◄](/html/2604.02318)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2604.02319)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2604.02319)
[View original  
on arXiv](https://arxiv.org/abs/2604.02319)[►](/html/2604.02320)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Wed May 6 01:55:48 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
