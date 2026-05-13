---
arxiv: '2503.19551'
authors:
- Zeyu Qin
- Qingxiu Dong
- Xingxing Zhang
- Li Dong
- Xiaolong Huang
- Ziyi Yang
- Mahmoud Khademi
- Dongdong Zhang
- Hany Hassan Awadalla
- Yi R. Fung
- Weizhu Chen
- Minhao Cheng
- Furu Wei
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Scaling Laws of Synthetic Data for Language Models
url: https://arxiv.org/abs/2503.19551
year: 2025
---

[2503.19551] Scaling Laws of Synthetic Data for Language Models














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



# Scaling Laws of Synthetic Data for Language Models

Zeyu Qin,  Qingxiu Dong,  Xingxing Zhang,  Li Dong,  Xiaolong Huang,  Ziyi Yang
  
Mahmoud Khademi,  Dongdong Zhang,  Hany Hassan Awadalla,  Yi R. Fung
  
  Weizhu Chen,  Minhao Cheng,  Furu Wei
  
  
<https://aka.ms/GeneralAI>
X. Zhang (xingxing.zhang@microsoft.com), L. Dong, X. Huang, Z. Yang, M. Khademi, D. Zhang, H. Awadalla, W. Chen and F. Wei are with Microsoft. Z. Qin, and Y. Fung are with Hong Kong University of Science and Technology. Q. Dong is with Peking University and M. Chen is with Pennsylvania State University.

###### Abstract

Large language models (LLMs) achieve strong performance across diverse tasks, largely driven by high-quality web data used in pre-training. However, recent studies indicate this data source is rapidly depleting. Synthetic data emerges as a promising alternative, but it remains unclear whether synthetic datasets exhibit predictable scalability comparable to raw pre-training data. In this work, we systematically investigate the scaling laws of synthetic data by introducing SynthLLM, a scalable framework that transforms pre-training corpora into diverse, high-quality synthetic datasets. Our approach achieves this by automatically extracting and recombining high-level concepts across multiple documents using a graph algorithm.
Key findings from our extensive mathematical experiments on SynthLLM include:
(1) SynthLLM generates synthetic data that reliably adheres to the *rectified scaling law* across various model sizes; (2) Performance improvements plateau near 300B tokens; and (3) Larger models approach optimal performance with fewer training tokens. For instance, an 8B model peaks at 1T tokens, while a 3B model requires 4T.
Moreover, comparisons with existing synthetic data generation and augmentation methods demonstrate that SynthLLM achieves superior performance and scalability. Our findings highlight synthetic data as a scalable and reliable alternative to organic pre-training corpora, offering a viable path toward continued improvement in model performance.

![Refer to caption](/html/2503.19551/assets/x1.png)


Figure 1: Scaling laws on different model sizes. The x axis denotes the number of training tokens. The y axis represents the models’ error rates on MATH. The green points represent the data sizes used to fit the scaling laws, while the red points are used to test the prediction performance of the fitted curves.

## 1 Introduction

Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of real-world applications, including chat assistant [[39](#bib.bib39), [4](#bib.bib4)] and code generation [[9](#bib.bib9), [3](#bib.bib3)]. Their success is fundamentally driven by the availability of vast amounts of high-quality data, which fuels model development at scale. As an essential resource, pre-training data provides the fundamental knowledge and generalization capability that underpin LLMs’ broad applicability [[41](#bib.bib41), [7](#bib.bib7)]. The importance of pre-training data is further reinforced by scaling laws [[20](#bib.bib20), [18](#bib.bib18), [38](#bib.bib38)], which suggest that larger models trained on more data tend to exhibit predictable and consistent performance improvements.

However, recent studies [[48](#bib.bib48), [37](#bib.bib37), [44](#bib.bib44)] have raised concerns that our supply of high-quality pre-training data is rapidly depleting. With the web-scraped textual corpora that power LLMs approaching saturation, the conventional scaling paradigm may face diminishing returns, potentially slowing down future progress. Rather than continuing to rely solely on next-token prediction over raw pre-training data, it is crucial to explore more principled methods to better harness the potential of existing corpora. One promising direction is the use of synthetic data [[13](#bib.bib13), [27](#bib.bib27), [30](#bib.bib30), [35](#bib.bib35), [32](#bib.bib32), [1](#bib.bib1)], which offers a way to amplify and extend the utility of existing human-generated data, potentially sustaining performance improvements even as natural data resources dwindle.
This prompts a natural question: Are there scaling laws for synthetic data?
While scaling laws for pre-training data are well-documented [[20](#bib.bib20), [18](#bib.bib18)], whether similar principles apply to synthetic data remains unclear. Can scaling synthetic datasets sustain performance gains, or do fundamental limitations arise? Understanding its scaling behavior is key to assessing its long-term viability as a solution to data scarcity.

To investigate the scaling laws of synthetic data, we aim to design a scalable approach capable of generating synthetic data at the scale.
Conventional approaches to synthetic dataset curation depend heavily on limited human-annotated seed examples from target domains [[50](#bib.bib50), [43](#bib.bib43), [53](#bib.bib53), [36](#bib.bib36), [46](#bib.bib46), [12](#bib.bib12), [23](#bib.bib23)]. This dependency fundamentally constrains both the diversity and scalability of the resulting datasets. In contrast, pre-training corpus—being both vast and highly diverse—still remains an underutilized resource for scalable synthetic data generation. We develop SynthLLM, a scalable web-scale synthetic data generation method designed to systematically transform pre-training data into high-quality synthetic datasets.

SynthLLM operates through three distinct stages. Initially, our method autonomously identifies and filters high-quality web documents within a target domain (e.g., Mathematics). Subsequently, leveraging these high-quality reference documents, we generate large-scale, diverse questions (or prompts) by employing open-source LLMs through three complementary methods, each strategically designed to progressively enhance question diversity. In final stage, we produce corresponding answers (or responses) to these generated questions, again utilizing open-source LLMs. It is worth noting that in the second stage, previous approaches have typically employed either direct question extraction approaches [[55](#bib.bib55), [56](#bib.bib56), [54](#bib.bib54)] or implemented document back-translation [[26](#bib.bib26)] to generate questions. However, these methods demonstrate inherent limitations in scalability, as question generation is constrained either by the finite size of reference documents containing high-quality questions or by the necessity of training dedicated back-translation models. Our method goes beyond *direct extraction* by automatically extracting and randomly combining high-level concepts from multiple documents with a graph algorithm, while establishing grounding across reference documents. Detailed ablation studies highlight the superior performance and scalability gained by our methods.

We apply SynthLLM to the mathematical reasoning domain to investigate scaling laws in synthetic data.
We observe the following:
(1) Synthetic data consistently follow the rectified scaling law [[29](#bib.bib29)] across various model sizes, as illustrated in Figure [1](#S0.F1 "Figure 1 ‣ Scaling Laws of Synthetic Data for Language Models").
(2) Performance gains start to diminish once the amount of synthetic data exceeds approximately 300B tokens.
(3) Larger models reach optimal performance more quickly compared to smaller ones. For instance, the 8B model requires only 1T tokens to achieve its best performance, whereas the 3B model needs 4T tokens.

These results underscore the critical importance of scaling synthetic data. Even with limited pre-training data, systematically expanding the synthetic dataset yields sustained and predictable performance gains, reinforcing its role as a scalable solution to reduce reliance on human-generated data while enhancing model capabilities.

## 2 Related Work

#### Synthetic Data Generation.

Synthetic data has emerged as a promising approach to augment real data, addressing challenges related to data scarcity and the high cost of data collection and annotation. It has been widely applied in pre-training, instruction following, and reasoning tasks, demonstrating notable improvements in model performance. In this work, we focus on synthetic data generation for the post-training phase. Existing synthetic data generation methods typically rely on a small set of high-quality human-annotated seed samples, leveraging LLMs to generate diverse augmentations through various techniques: 1) sampling seed instructions as few-shot examples to elicit LLMs to generate new instructions [[50](#bib.bib50), [19](#bib.bib19), [47](#bib.bib47), [23](#bib.bib23)]; 2) leveraging LLMs to rephrase seed samples [[53](#bib.bib53), [35](#bib.bib35), [12](#bib.bib12), [21](#bib.bib21), [34](#bib.bib34)]. However, the scarcity of high-quality seed examples limits the scalability and diversity of generated data. To address this, Xu et al. [[51](#bib.bib51)] and Li et al. [[24](#bib.bib24)] propose generating synthetic data from scratch, leveraging either the inherent uncertainty of LLMs or a predefined knowledge taxonomy to guide the generation process. In contrast, pre-training data—being
both vast and highly diverse—still remains an underutilized resource for scalable post-training data
generation. Recent methods have explored extracting high-quality samples directly from web documents (similar to our Level-1 method) [[55](#bib.bib55), [56](#bib.bib56), [54](#bib.bib54)] or employing document backtranslation to generate questions [[26](#bib.bib26)]. Our SynthLLM aligns with this line of research and offers a more efficient approach to leveraging web documents for scaling diverse sample generation compared to aforementioned methods.

#### Scaling Law of LLMs.

Scaling laws provide a predictive framework for estimating model performance based on key factors such as model size and pre-training data size, and have been extensively studied [[17](#bib.bib17), [42](#bib.bib42), [20](#bib.bib20), [18](#bib.bib18), [5](#bib.bib5)]. These laws offer valuable insights into how performance scales with computational resources, enabling more informed decisions on the optimal allocation of compute for pre-training LLMs [[20](#bib.bib20), [18](#bib.bib18)]. Recently, more fine-grained scaling laws have been proposed, including data-constrained scaling [[37](#bib.bib37)], hyperparameter scaling [[6](#bib.bib6)], and model distillation [[8](#bib.bib8)]. For LLM fine-tuning, Hernandez et al. [[16](#bib.bib16)] investigate scaling laws in the transition from unsupervised pre-training to fine-tuning, highlighting the critical role of pre-training in altering the scaling behavior of the model. Lin et al. [[29](#bib.bib29)] propose a rectified scaling law tailored specifically for fine-tuning LLMs on downstream tasks. Our work extends the rectified scaling law [[29](#bib.bib29)] and is the first to investigate and empirically validate that scaling laws hold for fine-tuning language models with synthetic data.

## 3 Scaling Law of Synthetic Data

Kaplan et al. [[20](#bib.bib20)] demonstrates that the performance of LLMs exhibits a power-law relationship with respect to both model size and dataset size, primarily focusing on the pre-training phase utilizing organic data.
In this section, we investigate the scaling properties of synthetic data. Specifically, we aim to determine whether a scaling law exists for synthetic data that enables accurate prediction of the performance of LLMs trained on such data.
Details of the approach employed for synthetic data generation (i.e., SynthLLM) will be presented in Section [4](#S4 "4 SynthLLM: Web-Scale Synthetic Data Generation ‣ Scaling Laws of Synthetic Data for Language Models"), which is a method for curating synthetic data from carefully filtered documents drawn from the pre-training corpus (comprised exclusively of organic raw data).

### 3.1 Scaling Laws for Organic Data

The scaling law for pre-training on organic data has been extensively studied in the literature [[20](#bib.bib20), [18](#bib.bib18)]. In particular, the predictive loss is a parametric function of the model parameter size NN and data size DD (i.e., number of tokens) following the power-law form [[18](#bib.bib18)]:

|  |  |  |  |
| --- | --- | --- | --- |
|  | L​(N,D)=ANα+BDβ+E^L(N,D)=\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}+\hat{E} |  | (1) |

This function above models the relationship between scaling loss L​(N,D)L(N,D) (usually measured on a validation set), the number of model parameters NN, and the number of training tokens DD, based on the classical risk decomposition, where the final loss can be decomposed into multiple terms, each corresponding to a different source of error. L​(N,D)L(N,D) is the final scaling loss. NN represents parameter count of the LLM, and DD denotes the number of training tokens. EE, the irreducible term, reflects the fundamental loss inherent in an ideal generative process operating over the natural text distribution. α\alpha and β\beta are decay exponents that determine how the loss decreases as NN and DD increase.
AA and BB regulate the rate of loss convergence and are usually influenced by the model architecture and the characteristics of the data distribution. The coefficients α,β,A,B,\alpha,\beta,A,B, and EE are learnable parameters that are determined by fitting a scaling curve.

Hernandez et al. [[16](#bib.bib16)] investigate scaling laws for transfer (i.e., the transition from unsupervised pre-training to fine-tuning). Their findings indicate that fine-tuning on pre-trained models exhibits more data efficient scaling behavior compared to training models from scratch on downstream tasks, emphasizing the crucial influence of pre-training on scaling dynamics. Extending this line of inquiry, Lin et al. [[29](#bib.bib29)] analyze scaling laws during fine-tuning of LLMs and propose Rectified Scaling Law:

|  |  |  |  |
| --- | --- | --- | --- |
|  | L​(D)=BDl+Dβ+EL(D)=\frac{B}{D\_{l}+D^{\beta}}+E |  | (2) |

In the following, we illustrate how the rectified scaling law is obtained from the *vanilla* scaling law in Eq. ([1](#S3.E1 "In 3.1 Scaling Laws for Organic Data ‣ 3 Scaling Law of Synthetic Data ‣ Scaling Laws of Synthetic Data for Language Models")).

Compared to the original formulation in Eq. ([1](#S3.E1 "In 3.1 Scaling Laws for Organic Data ‣ 3 Scaling Law of Synthetic Data ‣ Scaling Laws of Synthetic Data for Language Models")), we first remove the term associated with model parameters, as the LLM remains fixed during fine-tuning. Consequently, we derive the following marginalized form of Eq. ([1](#S3.E1 "In 3.1 Scaling Laws for Organic Data ‣ 3 Scaling Law of Synthetic Data ‣ Scaling Laws of Synthetic Data for Language Models")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | L​(D)=BDβ+EL(D)=\frac{B}{D^{\beta}}+E |  | (3) |

Here, L​(D)L(D) represents the validation metric used to evaluate the LLM’s performance on the downstream task. The irreducible term EE empirically represents the optimal achievable model performance as the data size approaches infinity. This term depends not only on inherent loss of the data distribution but also on the capacity limitations of the model itself.

Compared to the formulation in Eq. ([3](#S3.E3 "In 3.1 Scaling Laws for Organic Data ‣ 3 Scaling Law of Synthetic Data ‣ Scaling Laws of Synthetic Data for Language Models")), the Rectified Scaling Law introduces an additional term, DlD\_{l}, which represents the pre-learned data size. This term quantifies the latent knowledge relevant to the downstream task that the model has already acquired during pre-training. Fine-tuning on a pretrained LLM often yields better performance than training from scratch on a downstream task. The inclusion of DlD\_{l} reflects this benefit. Consequently, we obtain the final formulation in Eq. ([2](#S3.E2 "In 3.1 Scaling Laws for Organic Data ‣ 3 Scaling Law of Synthetic Data ‣ Scaling Laws of Synthetic Data for Language Models")).
The term BDl\frac{B}{D\_{l}} essentially defines the initial performance of the pre-trained LLM.
Lin et al. [[29](#bib.bib29)] show that this modification offers a more precise representation of the impact of pre-training on the model performance in downstream tasks during fine-tuning.

### 3.2 Scaling Laws for Synthetic Data

Prior research [[20](#bib.bib20), [18](#bib.bib18), [16](#bib.bib16), [29](#bib.bib29)] has primarily focused on investigating the scaling properties of organic, real-world data and their impact on LLM performance. To our knowledge, this work is the first to systematically investigate and empirically validate that analogous scaling laws hold for synthetic data.
We employ SynthLLM (Section [4](#S4 "4 SynthLLM: Web-Scale Synthetic Data Generation ‣ Scaling Laws of Synthetic Data for Language Models")) to generate data from a carefully filtered subset of the pre-training corpus, ensuring both scale and diversity (see Section [5.3](#S5.SS3 "5.3 Ablation Studies about Our SynthLLM Framework ‣ 5 Experiments ‣ Scaling Laws of Synthetic Data for Language Models") for details). Using the full synthetic dataset from the three methods in SynthLLM, we conduct continued training on LLMs of varying sizes (Llama-3.2-1B, Llama-3.2-3B, and Llama-3.1-8B) with progressively larger subsets (100K, 200K, 400K, 800K, 1.6M, 3.2M, and 7.4M examples).
We selected the mathematical reasoning domain for our experiments due to its well-established evaluation protocols and metrics. Specifically, we report the error rate on the MATH dataset [[15](#bib.bib15)], which covers a wide range of mathematical subjects and incorporates varied difficulty levels (ranging from Level-1 to level 5), making it more comprehensive than alternative datasets.
Note that we report error rate rather than accuracy to maintain consistency with previous scaling law literature [[20](#bib.bib20), [18](#bib.bib18)]. It is worth noting that both lower loss and lower error rate indicate superior performance in evaluations. See more details of our synthetic datasets and experiments in Section [5.1](#S5.SS1 "5.1 Experimental Settings ‣ 5 Experiments ‣ Scaling Laws of Synthetic Data for Language Models").

#### Overall Trend.

As shown in Figure [1](#S0.F1 "Figure 1 ‣ Scaling Laws of Synthetic Data for Language Models"), the synthetic data generated by SynthLLM adheres to the rectified scaling law (Equation [2](#S3.E2 "In 3.1 Scaling Laws for Organic Data ‣ 3 Scaling Law of Synthetic Data ‣ Scaling Laws of Synthetic Data for Language Models")).
The xx-axis represents the synthetic dataset size, measured in number of tokens, while the yy-axis depicts the error rate on the MATH dataset. Both axes employ logarithmic scaling, consistent with previous work [[20](#bib.bib20)].
The green points denote the data sizes utilized to fit the scaling laws, whereas the red points serve to evaluate the predictive accuracy of the fitted curves. Our results demonstrate that synthetic data generated by SynthLLM consistently adheres to the rectified scaling law across various model sizes.

In addition to the rectified scaling law (Equation [2](#S3.E2 "In 3.1 Scaling Laws for Organic Data ‣ 3 Scaling Law of Synthetic Data ‣ Scaling Laws of Synthetic Data for Language Models")), we also attempted to fit our results with the power scaling law (Equation [3](#S3.E3 "In 3.1 Scaling Laws for Organic Data ‣ 3 Scaling Law of Synthetic Data ‣ Scaling Laws of Synthetic Data for Language Models")). Our analysis reveals that the power law fails to capture the scaling trend effectively (see Section [5.4](#S5.SS4 "5.4 Ablation Study on Scaling Law Form ‣ 5 Experiments ‣ Scaling Laws of Synthetic Data for Language Models")). We hypothesize that this phenomenon may be attributed to the base models used in our experiments having already acquired certain mathematical problem-solving capabilities during their pre-training phase.

#### Model Size Matters.

The decay exponent β\beta in Eq. ([2](#S3.E2 "In 3.1 Scaling Laws for Organic Data ‣ 3 Scaling Law of Synthetic Data ‣ Scaling Laws of Synthetic Data for Language Models")) quantifies the rate at which the error rate decreases as dataset size increases. A larger value of β\beta corresponds to a more rapid reduction in error rate, given the same amount of synthetic tokens.
As in Figure [1](#S0.F1 "Figure 1 ‣ Scaling Laws of Synthetic Data for Language Models"), in comparison to its large counterparts (3B and 8B models), the 1B model exhibits a smaller β\beta value (0.340.34), suggesting that increasing the volume of synthetic data yields more modest performance improvements for this smaller model.
The ratio BDl\frac{B}{D\_{l}} demonstrates a strong correlation with the initial capacity of models, wherein a smaller value (smaller error rate) indicates greater initial model capability.
The 7B model exhibits the smallest ratio value (56.356.3),

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Model | 10B tokens | 50B tokens | 250B tokens | 300B tokens | 1T tokens | 4T tokens |
| 3B | 64.664.6 | 74.774.7 | 80.580.5 | 80.980.9 | 83.183.1 | 84.484.4 |
| 8B | 73.273.2 | 78.678.6 | 81.481.4 | 81.681.6 | 82.682.6 | 83.283.2 |

Table 1: The predicted MATH accuracies (%) on 3B and 8B models when continuing to scale synthetic data based on our scaling law.

which may be attributed to its superior model capacity and more comprehensive pre-training data exposure.
Furthermore, the 3B and 7B models demonstrate identical decay exponents (β\beta), indicating that performance disparities stem predominantly from differences in initial knowledge acquisition and inherent model capacity rather than from variations in the efficacy of synthetic data scaling.

![Refer to caption](/html/2503.19551/assets/graph/level123_all_scaling.png)


Figure 2: Scaling laws on different model sizes. The x-axis represents the number of training tokens. The y-axis shows the models’ error rates on MATH benchmark. Solid lines indicate fitted scaling curves for different model sizes. The dotted lines represent predicted final performances when synthetic data is further scaled, as extrapolated from these curves. More details are shown in Figure [1](#S0.F1 "Figure 1 ‣ Scaling Laws of Synthetic Data for Language Models").

#### Predictable Performance.

For 3B and 8B models, we evaluated the predictive capability of our fitted scaling curves. As illustrated by the red squares in Figure [1](#S0.F1 "Figure 1 ‣ Scaling Laws of Synthetic Data for Language Models"), these curves accurately forecast error rates on the MATH benchmark for both models (40.0%40.0\% and 28.7%28.7\%, respectively), when synthetic dataset is expanded to 7.4M samples (4.5B tokens). These error rates correspond to accuracies of 60.0%60.0\% and 71.3%71.3\%. This highlights the critical role of scaling synthetic data, demonstrating that even as pre-training data nears saturation, systematically expanding the synthetic dataset continues to yield sustained and predictable performance gains.
Leveraging our established scaling curves, we can extrapolate the future performance of models with continued expansion of synthetic data beyond current volumes. Table [1](#S3.T1 "Table 1 ‣ Model Size Matters. ‣ 3.2 Scaling Laws for Synthetic Data ‣ 3 Scaling Law of Synthetic Data ‣ Scaling Laws of Synthetic Data for Language Models") presents these projections for the 3B and 8B models.
The results indicate that performance improvements begin to diminish after approximately 300B synthetic tokens. We estimate that the 3B model would require approximately 4T synthetic data tokens to approach its performance ceiling, while the 8B model would need approximately 1T synthetic tokens (also see Figure [2](#S3.F2 "Figure 2 ‣ Model Size Matters. ‣ 3.2 Scaling Laws for Synthetic Data ‣ 3 Scaling Law of Synthetic Data ‣ Scaling Laws of Synthetic Data for Language Models")).

## 4 SynthLLM: Web-Scale Synthetic Data Generation

In this section, we present details of SynthLLM for synthetic data generation. For specific target domain (e.g., mathematical reasoning), we begin by filtering high-quality and domain-specific reference documents from Fineweb-Edu [[40](#bib.bib40)], an open repository of web data (Section [4.1](#S4.SS1 "4.1 Reference Documents Filtering ‣ 4 SynthLLM: Web-Scale Synthetic Data Generation ‣ Scaling Laws of Synthetic Data for Language Models")).
Next, using these filtered reference documents, we construct large-scale, diverse questions (or prompts) by prompting open-source LLMs through three distinct methods, each designed to progressively increase question diversity (Section [4.2](#S4.SS2 "4.2 Document-Grounded Question Generation ‣ 4 SynthLLM: Web-Scale Synthetic Data Generation ‣ Scaling Laws of Synthetic Data for Language Models")).
Finally, we produce corresponding answers (or responses) for the generated questions, again employing open-source LLMs (Section [4.3](#S4.SS3 "4.3 Answer Generation ‣ 4 SynthLLM: Web-Scale Synthetic Data Generation ‣ Scaling Laws of Synthetic Data for Language Models")).

### 4.1 Reference Documents Filtering

![Refer to caption](/html/2503.19551/assets/x2.png)


Figure 3: Overview of reference documents filtering. The whole process contains two classifiers: 1) cold-start classifier trained on generated positive samples (documents) and randomly sampled negative examples; 2) fine-grained classifier trained on previously filtered documents and additional randomly sampled documents with extra ratings. We train fine-grained classifier for 2 iterations.

#### Cold-start Domain Classifier.

For a given target domain, we train a binary classifier to filter web documents from Fineweb-Edu. While negative examples (documents) can be randomly sampled, obtaining sufficient positive examples (documents) for each domain is more challenging. To address this, we use synthetic documents as positive examples. Specifically, we generate synthetic domain-specific documents based on syllabi from GLAN [[24](#bib.bib24)], which cover 126 distinct disciplines. Each syllabus is produced by GPT-4 "(almost) from scratch", with further details available in [[24](#bib.bib24)]. As illustrated in Figure [3](#S4.F3 "Figure 3 ‣ 4.1 Reference Documents Filtering ‣ 4 SynthLLM: Web-Scale Synthetic Data Generation ‣ Scaling Laws of Synthetic Data for Language Models"), we first use these syllabi to prompt a large language model (LLM) to generate synthetic documents representative of the target domain (e.g., blog posts, exam papers, and textbooks). Once we obtain the synthetic positive examples and randomly sampled negative examples from Fineweb-Edu, we train a cold-start domain classifier. Applying this classifier yields an initial set of domain-relevant documents, 𝒟0\mathcal{D}^{0}, from Fineweb-Edu.

#### Fine-grained Domain Classifier.

The initial reference document set, 𝒟0\mathcal{D}^{0}, is typically small. To expand it, we iteratively classify Fineweb-Edu documents based on previously identified references, 𝒟t−1\mathcal{D}^{t-1}. At each iteration tt, we randomly sample an additional subset 𝒟Rt−1\mathcal{D}^{t-1}\_{R} from Fineweb-Edu and instruct GPT-4o to evaluate all documents in 𝒟t−1∪𝒟Rt−1\mathcal{D}^{t-1}\cup\mathcal{D}^{t-1}\_{R} on a 1–10 scale, considering relevance to the target domain, clarity, and language quality. These ratings are then used to train a fine-grained classifier, which we apply to Fineweb-Edu to construct the next iteration’s reference set, 𝒟t\mathcal{D}^{t}. We retain documents with scores above a threshold of 6.5, as it yields satisfactory performance in initial experiments. Empirically, two iterations suffice to generate a high-quality and sufficiently large reference corpus.

Both the cold-start and fine-grained classifiers are implemented using random forests, trained on Fineweb-Edu documents represented as vector embeddings generated by StableLM-2-1.6B.

### 4.2 Document-Grounded Question Generation

After applying the filtering process described in Section [4.1](#S4.SS1 "4.1 Reference Documents Filtering ‣ 4 SynthLLM: Web-Scale Synthetic Data Generation ‣ Scaling Laws of Synthetic Data for Language Models"), we obtain the filtered document set 𝒟\mathcal{D}. We then generate questions through three distinct methods, each relying on either single-document or multi-document contexts, with the goal of progressively increasing question diversity. Let 𝐌Q\mathbf{M}^{Q} denote the LLM employed for question generation.

#### Level-1 Generator.

In the first method, we generate questions from a single reference document d∈𝒟d\in\mathcal{D} (where 𝒟\mathcal{D} is filtered document set). Because dd may already contain questions relevant to target domain, we first prompt the LLM 𝐌Q\mathbf{M}^{Q} to determine whether dd includes any existing questions. If such questions exist, we directly extract them. We then instruct 𝐌Q\mathbf{M}^{Q} to create additional questions inspired by the content of dd. Finally, 𝐌Q\mathbf{M}^{Q} annotates each question with a special tag, either <<is\_original>> or <<newly\_created>>, to indicate whether the question is an original question from dd. The detailed prompt for 𝐌Q\mathbf{M}^{Q} is in Figure [8](#A1.F8 "Figure 8 ‣ Appendix A Used Prompts ‣ Scaling Laws of Synthetic Data for Language Models") of Appendix. Similar directly extraction-based synthesis method has been adopted in Yue et al. [[55](#bib.bib55)] Zhou et al. [[56](#bib.bib56)], and Yuan et al. [[54](#bib.bib54)].

However, the Level-1 approach described above may be limited in scalability, as the diversity and content of newly generated questions is constrained by the finite set of high-quality reference documents 𝒟\mathcal{D} containing questions.

#### Level-2 Generator.

To address the aforementioned limitation and more effectively leverage the reference document dd, we draw inspiration from pedagogical principles. In conventional educational settings [[11](#bib.bib11), [2](#bib.bib2)], instructors typically begin by introducing fundamental knowledge and key concepts. They then progressively guide students toward a deeper understanding and application of these concepts through practice or exercises, thereby fostering problem-solving and critical thinking skills. Our Level-2 method mirrors this approach and implement it in two stages, as illustrated in Figure [4](#S4.F4 "Figure 4 ‣ Level-2 Generator. ‣ 4.2 Document-Grounded Question Generation ‣ 4 SynthLLM: Web-Scale Synthetic Data Generation ‣ Scaling Laws of Synthetic Data for Language Models")

![Refer to caption](/html/2503.19551/assets/x3.png)


Figure 4: Illustrations of our Document-Grounded Question Generation methods.

First, we prompt the model to act as an instructor, extracting key topics and concepts from the document dd for teaching. Then, we randomly sample these extracted topics and key concepts and combine them with dd to guide further question generation. Here, the model again acts as an instructor, creating exercises that target the sampled topics and concepts.
Specifically, we prompt 𝐌Q\mathbf{M}^{Q} with a prompt 𝐩k\mathbf{p}\_{k} to extract related topics (e.g., "Curvature", "Central Limit Theorem"), and key concepts (e.g., "Arc length of a curve", "Binomial Approximation to Normal Distribution"). Formally, we compute 𝐊=𝐌Q​(𝐩k,d)\mathbf{K}=\mathbf{M}^{Q}(\mathbf{p}\_{k},d), where 𝐊\mathbf{K} represents the extracted meta-information described above. Note that the detailed prompt 𝐩k\mathbf{p}\_{k} is provided in Figure [11](#A1.F11 "Figure 11 ‣ Appendix A Used Prompts ‣ Scaling Laws of Synthetic Data for Language Models") of Appendix. Figure [12](#A2.F12 "Figure 12 ‣ Appendix B Examples about Extracted Topics and Key Concepts from Reference Document ‣ Scaling Laws of Synthetic Data for Language Models") and [13](#A2.F13 "Figure 13 ‣ Appendix B Examples about Extracted Topics and Key Concepts from Reference Document ‣ Scaling Laws of Synthetic Data for Language Models") illustrate examples of the extracted 𝐊\mathbf{K}.

In the second stage, we focus on generating questions.
To enhance the diversity of generated questions, we inject randomness into the generation process. Specifically, we first instruct 𝐌Q\mathbf{M}^{Q} to randomly select and combine concepts from 𝐊\mathbf{K}, producing a subset 𝐤s\mathbf{k}\_{s}. The final question is generated by 𝐌Q\mathbf{M}^{Q}, drawing on the reference document dd, and the selected subset 𝐤s\mathbf{k}\_{s}.
We aim for the generated questions to remain grounded in the reference document dd and strictly adhere to the selected subset 𝐤s\mathbf{k}\_{s}. We denote the whole question generation process above as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐱=𝐌Q​((𝐩q,ℛ​(𝐊),d)),𝐊=𝐌Q​((𝐩k,d)),\displaystyle\mathbf{x}=\mathbf{M}^{Q}((\mathbf{p}\_{q},\mathcal{R}(\mathbf{K}),d)),\ \ \ \mathbf{K}=\mathbf{M}^{Q}((\mathbf{p}\_{k},d)), |  | (4) |

where 𝐤s=ℛ​(𝐊)\mathbf{k}\_{s}=\mathcal{R}(\mathbf{K}), ℛ\mathcal{R} is random selection operator and 𝐱\mathbf{x} is the generated questions. For simplicity, we integrate subset selection of 𝐤s\mathbf{k}\_{s} and the generation of the final questions into a single prompt 𝐩q\mathbf{p}\_{q}. The detailed prompt 𝐩q\mathbf{p}\_{q} is shown in Figure [9](#A1.F9 "Figure 9 ‣ Appendix A Used Prompts ‣ Scaling Laws of Synthetic Data for Language Models") of Appendix.
As indicated in Equation ([4](#S4.E4 "In Level-2 Generator. ‣ 4.2 Document-Grounded Question Generation ‣ 4 SynthLLM: Web-Scale Synthetic Data Generation ‣ Scaling Laws of Synthetic Data for Language Models")), the final questions 𝐱\mathbf{x} rely on both the (random) concept combinations generated by ℛ​(𝐊)\mathcal{R}(\mathbf{K}) and the document dd. This design intuitively offers greater diversity than the Level-1 method, which depends solely on dd. This brings significant advantages. High-quality question synthesis is no longer constrained by the limited existing questions within reference documents. Instead, by decomposing and recombining key concepts, our Level-2 approach more efficiently exploits limited reference documents, thereby enabling more scalable synthetic question generation. In Section [5.3](#S5.SS3 "5.3 Ablation Studies about Our SynthLLM Framework ‣ 5 Experiments ‣ Scaling Laws of Synthetic Data for Language Models"), we validate the benefits of Level-2 through a series of detailed ablation studies.

The overall set of topics and key concepts 𝐊\mathbf{K} resembles the syllabus from GLAN [[24](#bib.bib24)]; however, a key distinction is that each 𝐊\mathbf{K} in our approach is explicitly grounded in a reference document dd.

#### Level-3 Generator.

Although the Level-2 generator can generate diverse questions by combining different topics and key concepts within a single document, each document inherently covers only a limited set of material.
To further promote diversity of generation questions and make the approach more scalable, the Level-3 generator extends the Level-2 generator by incorporating concepts from multiple documents and grounding across multiple documents. The process of Level-3 generator is illustrated in Figure [4](#S4.F4 "Figure 4 ‣ Level-2 Generator. ‣ 4.2 Document-Grounded Question Generation ‣ 4 SynthLLM: Web-Scale Synthetic Data Generation ‣ Scaling Laws of Synthetic Data for Language Models").

#### *Global Concept Graph Construction*.

In the Level-2 Generator, we have extracted topics and key concepts (i.e., 𝐊\mathbf{K}) from a *single* document. In this section, we extend this modeling to a global perspective by considering relationships among all high-level concepts. Specifically, we construct a graph GG, whose nodes comprise all distinct topics 𝕋\mathbb{T} and key concepts (KCs) ℂ\mathbb{C}. The intuition behind this construction is that topics or key concepts frequently co-occurring within the same document likely indicate meaningful associations. Thus, we utilize the co-occurrence statistics between topics and KCs to define the edge weights in the graph. Consequently, we consider three types of edges: topic-topic edges, topic-KC edges, and KC-KC edges, which results in three distinct sub-graphs: the topic graph, the topic-KC graph, and the KC graph. An edge between two nodes, a topic or KC 𝐮\mathbf{u} and another topic or KC 𝐯\mathbf{v}, is established whenever they appear together within the same document. The edge weight w​(𝐮,𝐯)w(\mathbf{u},\mathbf{v}) is formally defined as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | w​(𝐮,𝐯)=log⁡(freq​(𝐮,𝐯)+ϵ)w(\mathbf{u},\mathbf{v})=\log\big{(}\text{freq}(\mathbf{u},\mathbf{v})+\epsilon\big{)} |  | (5) |

where freq​(𝐮,𝐯)\text{freq}(\mathbf{u},\mathbf{v}) denotes the raw co-occurrence count of nodes 𝐮\mathbf{u} and 𝐯\mathbf{v}, and ϵ\epsilon is a small positive constant introduced to ensure numerical stability.

#### *Concept Combination Sampling*.

Once the concept graph GG has been constructed, we proceed to sample concept combinations. These sampled combinations are then utilized to retrieve relevant grounding documents, ultimately facilitating the generation of new questions.

The sampling process begins with uniform random sampling from the nodes in the topic sub-graph 𝕋\mathbb{T}. In implementation, we iterate through all topics in 𝕋\mathbb{T} over multiple epochs to achieve this. Once an initial topic is randomly selected, we perform a random walk of one to two steps within the topic sub-graph to identify additional related topics. The transition probability of moving from node 𝐮\mathbf{u} to node 𝐯\mathbf{v} during the random walk is computed using a softmax function:

|  |  |  |  |
| --- | --- | --- | --- |
|  | p𝐮,𝐯=exp⁡(w​(𝐮,𝐯))∑𝐯′∈𝒩​(𝐮)exp⁡(w​(𝐮,𝐯′))p\_{\mathbf{u,v}}=\frac{\exp\left(w(\mathbf{u},\mathbf{v})\right)}{\sum\_{\mathbf{v}^{\prime}\in\mathcal{N}(\mathbf{u})}\exp\left(w(\mathbf{u},\mathbf{v}^{\prime})\right)} |  | (6) |

where 𝒩​(𝐮)\mathcal{N}(\mathbf{u}) denotes the set of nodes adjacent to 𝐮\mathbf{u} in the topic sub-graph.

After obtaining the sampled topics, we next sample their corresponding key concepts. We first select an initial key concept by performing a one-step random walk on the topic-KC sub-graph, with transition probabilities computed using Equation ([6](#S4.E6 "In Concept Combination Sampling. ‣ 4.2 Document-Grounded Question Generation ‣ 4 SynthLLM: Web-Scale Synthetic Data Generation ‣ Scaling Laws of Synthetic Data for Language Models")) but applied specifically within this sub-graph. Subsequently, additional related key concepts are sampled through three to four steps of random walks on the KC sub-graph. This process results in a set of sampled topics and key concept 𝐊g\mathbf{K}^{g}.

Since the sampled or extracted high-level concepts essentially serve as a summary, they may omit certain details originally present in the reference documents.
Thus, we search for reference documents closely aligned with our sampled concept set 𝐊g\mathbf{K}^{g}.
Specifically, we calculate the Jaccard similarity between the concept set 𝐊g\mathbf{K}^{g} and the concept sets of each document within the filtered seed document set 𝒟\mathcal{D}. Subsequently, we select the two documents with the highest similarity scores as the final reference documents. Analogous to the procedure in the Level-2 Generator, we instruct 𝐌Q\mathbf{M}^{Q} to randomly select and combine concepts from 𝐊g\mathbf{K}^{g}, and then utilize the selected concepts as well as the two chosen documents as references for question generation. The detailed prompt is shown in Figure [10](#A1.F10 "Figure 10 ‣ Appendix A Used Prompts ‣ Scaling Laws of Synthetic Data for Language Models") of Appendix. In the Level-3 Generator, the combination of concepts go beyond a *single* document. In addition to grounding on multiple documents, by employing random sampling on a concept graph constructed from the entire filtered document set, the diversity of the generated questions is significantly increased, thereby broadening the scope and richness of the resulting question set.

### 4.3 Answer Generation

Using methods described in Section [4.2](#S4.SS2 "4.2 Document-Grounded Question Generation ‣ 4 SynthLLM: Web-Scale Synthetic Data Generation ‣ Scaling Laws of Synthetic Data for Language Models"), we obtain a large set questions. Now we are ready to generation answers. We employ open-source LLMs such as Qwen2.5-Math-72B-Instruct as our answer generator 𝐌A\mathbf{M}^{A}. We have not yet incorporated an additional answer verification process, since preliminary experiments indicate that our synthetic data has already yielded satisfactory results. We do believe that dedicating effort to constructing and validating answers (e.g., majority voting and multi-agent debating) can further enhance the quality of our synthetic data at additional computational cost. We plan to leave the answer verification process in further work.

## 5 Experiments

### 5.1 Experimental Settings

#### Models.

For 𝐌Q\mathbf{M}^{Q}, we utilize Mistral-Large-Instruct-2407 [[45](#bib.bib45)], as we believe that larger models inherently possess the essential internal knowledge required for effective concept extraction and question synthesis. For the answer generator, 𝐌A\mathbf{M}^{A}, we employ Qwen2.5-Math-72B [[52](#bib.bib52)]. For trained models, we adopt the Llama-3.2-1B, Llama-3.2-3B, and Llama-3.1-8B base models [[31](#bib.bib31)].

#### Training Settings.

We employ Supervised Fine-Tuning (SFT) to train the model. We adopt AdamW optimizer [[33](#bib.bib33)] and set learning rate to 1×10−51\times 10^{-5} with a batch size of 512512. We utilize the linear learning rate scheduler and train the model for 3 epochs.

#### Our Datasets.

In this work, we main focus on the math domain. After filtering high-quality documents from Web data, we collected approximately 520​K520\text{K} math documents. For Level-1 and Level-2, we generated 5 questions per reference document. We then deduplicate and decontaminate our synthetic questions using evaluated benchmarks, resulting in roughly 2.3​M2.3\text{M} and 2.6​M2.6\text{M} samples, respectively. For Level-3, the graph constructed for each subset contains around 32 00032\,000 topics and 200 000200\,000 knowledge concepts. We perform random walks on the topic sub-graph for 5 epochs, enumerating each topic in each epoch. For each sampled 𝐊g\mathbf{K}^{g}, we generate 33 questions, which results in approximately 2.6​M2.6\text{M} samples. Due to the substantial training cost associated with large-scale datasets, we limit the number of sampled questions per document in Level-2 and constrain the number of random walk epochs in Level-3 to a relatively small scale (e.g., 5).

#### Statistics of Synthetic Questions and Answers.

The median lengths of the questions generated by Level-1, Level-2, and Level-3 are 3232, 6666, and 8080, respectively, while the corresponding median lengths of the answers are 358358, 497497, and 545545.

#### Evaluation.

To evaluate the math reasoning ability of our model trained on synthetic data, we utilize the evaluation code from Qwen-Math [[52](#bib.bib52)] to conduct assessments on several mainstream benchmarks: MATH [[15](#bib.bib15)], GSM8K [[10](#bib.bib10)], OlympiadBench-Math (Olympiad) [[14](#bib.bib14)], CollegeMath (College) [[43](#bib.bib43)], Minerva Math (Minerva) [[22](#bib.bib22)], and Gaokao 2023 En (Gaokao) [[28](#bib.bib28)].

#### Other Baselines.

We compare our synthetic data with other high-quality synthetic datasets. MAmmoTH2 [[55](#bib.bib55)], NaturalReasoning [[54](#bib.bib54)], and JiuZhang3.0 [[56](#bib.bib56)] employ a direct extraction-based synthesis approach on pre-training documents, similar to our Level-1 method. OpenMathInstruct-2 [[47](#bib.bib47)] is a large-scale synthetic math dataset consisting of 14M samples, with GSM8K and MATH serving as the seed datasets. Additionally, we compare our dataset with NuminaMath [[25](#bib.bib25)], a high-quality dataset that includes both synthetically generated data and extracted question-answer pairs from exam papers and mathematics discussion forums.

Furthermore, we evaluate our Level-2 and Level-3 approaches against existing methods designed to enhance question diversity through augmentation. Specifically, we include two popular augmentation approaches: the rephrase augmentation approach [[53](#bib.bib53), [35](#bib.bib35)] and the persona augmentation approach [[12](#bib.bib12), [21](#bib.bib21)], both of which generate new questions based on the questions extracted using Level-1. For rephrase augmentation, we follow the same procedure as MetaMath [[53](#bib.bib53)], providing two-shot examples in the prompt to rephrase questions. For persona augmentation, we utilize the 200K personas released by the authors as prompts for question augmentation.

### 5.2 Main Results

In this section, we present main evaluation results of SynthLLM. We test our models on various mathematical reasoning benchmarks, which encompass a range of difficulty levels. The results are shown in Table [2](#S5.T2 "Table 2 ‣ SynthLLM achieves superior performance compared to other synthetic datasets. ‣ 5.2 Main Results ‣ 5 Experiments ‣ Scaling Laws of Synthetic Data for Language Models"). The values in parentheses represent the sample size. For compared baseline training datasets, we directly evaluate model checkpoints they have released. We adopt the entire dataset
synthesized with different methods from our framework to train models and denote it as SynthLLM-3B or -8B. The full dataset has a sample size of 7.4M, and we also create a subset with 3.2M samples.

#### SynthLLM achieves superior performance compared to other synthetic datasets.

SynthLLM consistently outperforms other synthetic datasets across all benchmarks, demonstrating the effectiveness of our framework. OpenMathInstruct-2, a high-quality math reasoning dataset generated using the Llama3.1-405B model based on MATH and GSM8K, comprises 14M samples. Despite its significantly larger size, the model trained on smaller dataset (SynthLLM-8B (3.2M)) achieves comparable performance on its in-distribution test sets (MATH and GSM8K) while substantially surpassing OpenMathInstruct-2 on Out-of-Distribution (OOD) test sets. This highlights the superior generalization capabilities of SynthLLM. Compared to the corresponding Llama Instruct models across different sizes, our models consistently achieve superior performance. Furthermore, our models (SynthLLM-8B) match or even surpass much larger models such as NuminaMath-CoT-72B and Llama-3.1-70B-Instruct on benchmarks, further reinforcing the effectiveness of our approach.

|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model | GSM8K | MATH | Minerva | Olympiad | College | Gaokao | Average |
| Llama-3.2-1B-Instruct | 47.247.2 | 28.028.0 | 5.95.9 | 5.55.5 | 18.818.8 | 25.225.2 | 21.821.8 |
| SynthLLM-1B (3.2M) | 45.745.7 | 32.932.9 | 6.66.6 | 7.67.6 | 27.327.3 | 28.828.8 | 24.824.8 |
| SynthLLM-1B (7.4M) | 50.4 | 37.4 | 8.1 | 8.3 | 31.7 | 30.6 | 27.4 |
| Llama-3.2-3B-Instruct | 78.078.0 | 47.547.5 | 17.317.3 | 14.814.8 | 31.831.8 | 37.737.7 | 37.937.9 |
| SynthLLM-3B (3.2M) | 78.178.1 | 54.354.3 | 16.916.9 | 17.517.5 | 38.338.3 | 46.546.5 | 41.941.9 |
| SynthLLM-3B (7.4M) | 80.7 | 60.0 | 18.8 | 21.9 | 42.3 | 50.9 | 45.8 |
| Llama-3.1-8B-Instruct | 84.284.2 | 48.948.9 | 25.725.7 | 13.213.2 | 32.132.1 | 43.443.4 | 41.241.2 |
| Llama-3.1-70B-Instruct | 94.5 | 66.166.1 | 34.2 | 29.629.6 | 41.441.4 | 56.656.6 | 54.254.2 |
| NuminaMath-CoT-7B | 75.475.4 | 55.255.2 | 19.119.1 | 19.919.9 | 36.936.9 | 47.547.5 | 42.342.3 |
| NuminaMath-CoT-72B | 90.890.8 | 66.766.7 | 25.025.0 | 32.632.6 | 39.739.7 | 54.054.0 | 51.551.5 |
| MAmmoTH2-Plus-8B (10M) | 78.478.4 | 41.941.9 | 10.710.7 | 11.311.3 | 16.116.1 | 31.931.9 | 31.731.7 |
| JiuZhang3.0-8B (6M) | 88.788.7 | 51.251.2 | 21.721.7 | 18.818.8 | 37.537.5 | 43.443.4 | 43.643.6 |
| NaturalReasoning-8B (2.8M)\* | - | 55.655.6 | - | - | - | - | - |
| OpenMathInstruct-2-8B (14M) | 91.191.1 | 67.567.5 | 22.522.5 | 27.727.7 | 39.239.2 | 53.553.5 | 50.350.3 |
| SynthLLM-8B (3.2M) | 88.488.4 | 66.166.1 | 25.425.4 | 30.230.2 | 44.344.3 | 56.956.9 | 51.951.9 |
| SynthLLM-8B (7.4M) | 92.192.1 | 71.3 | 26.526.5 | 33.0 | 45.3 | 61.0 | 54.9 |

Table 2: Performance of models of different sizes on various math benchmarks. All metrics are reported as percentages (%). We evaluate three model sizes, with the best result highlighted in bolded. The number in () represents training sample number. NaturalReasoning-8B: Results are directly referenced from the original paper, as the full dataset and model have not been released.

More importantly, for each reference document, we currently generate only five questions per document for Level-1 and Level-2, and three questions for Level-3, which is far from reaching the full potential of our method’s document utilization (as shown in Figure [6](#S5.F6 "Figure 6 ‣ Comparison of Level-2 and Level-3 with directly extraction-based synthesis method. ‣ 5.3 Ablation Studies about Our SynthLLM Framework ‣ 5 Experiments ‣ Scaling Laws of Synthetic Data for Language Models")). We can further enhance performance on benchmarks simply by increasing the number of synthesized questions per document.

### 5.3 Ablation Studies about Our SynthLLM Framework

In this section, we present detailed ablation studies about our Level-2 and Level-3 methods, demonstrating their advantages in scalability and diversity.

|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model | GSM8K | MATH | Minerva | Olympiad | College | Gaokao | Average |
| Llama-3.2-1B-Instruct | 47.2 | 28.028.0 | 5.95.9 | 5.55.5 | 18.818.8 | 25.225.2 | 21.821.8 |
| Level-1-1B (2.3M) | 38.938.9 | 28.528.5 | 6.66.6 | 5.35.3 | 26.5\mathbf{26.5} | 27.0 | 22.222.2 |
| Level-2-1B (2.6M) | 40.140.1 | 27.227.2 | 4.84.8 | 5.55.5 | 23.323.3 | 24.924.9 | 21.021.0 |
| Level-3-1B (2.6M) | 42.642.6 | 30.1 | 9.2 | 7.1 | 25.025.0 | 26.226.2 | 23.4 |
| Llama-3.2-3B-Instruct | 78.0 | 47.547.5 | 17.317.3 | 14.814.8 | 31.831.8 | 37.737.7 | 37.937.9 |
| Level-1-3B (2.3M) | 71.871.8 | 46.846.8 | 16.216.2 | 13.613.6 | 36.336.3 | 42.342.3 | 37.837.8 |
| Level-2-3B (2.6M) | 74.174.1 | 49.9 | 17.217.2 | 15.315.3 | 37.4 | 43.4 | 39.639.6 |
| Level-3-3B (2.6M) | 77.077.0 | 49.149.1 | 18.0 | 17.2 | 37.4 | 42.942.9 | 40.3 |

Table 3: Performance of 1B and 3B models on various math benchmarks. The number in ()() represents the sample size. The best result is in bolded. All metrics are measured in percentage (%).

#### Comparison of Level-2 and Level-3 with directly extraction-based synthesis method.

We compare the directly extraction-based synthesis method (Level-1) with our Level-2 and Level-3 methods across models of varying sizes. The results are shown in Table [3](#S5.T3 "Table 3 ‣ 5.3 Ablation Studies about Our SynthLLM Framework ‣ 5 Experiments ‣ Scaling Laws of Synthetic Data for Language Models"). We observe that, on the 3B model, Level-2 and Level-3 significantly outperform Level-1 and instruct version across benchmarks. On the 1B model, Level-3 surpasses Level-1 by a large margin on most benchmarks, with the exception of College and Gaokao.
Although each document is sampled with 5 questions, the sample size for Level-1 (2.3M) is significantly lower than that of Level-2 (2.6M).

![Refer to caption](/html/2503.19551/assets/x4.png)


Figure 5: Histogram of question similarity within the same document. The x-axis represents the cosine similarity between question pairs from the same document, while the y-axis denotes the frequency of each bin.

This is because Level-1 is based on direct extraction, which leads to a substantial amount of duplicate questions being generated. After removing duplicates, the sample size decreases considerably.
This also highlights the scalability limitations of direct extraction-based synthesis, as it is inherently constrained by the availability of high-quality reference documents containing questions.

Diversity of Generated Questions.
Compared to direct extraction-based synthesis, SynthLLM generates more diverse questions by decomposing and recombining knowledge concepts. To assess this, we compare Level-1 and Level-2 by measuring the similarity among generated questions within the same document. Specifically, we randomly select 2,000 documents and generate 50 questions per document for each method. We adopt all-MiniLM-L6-v2 [[49](#bib.bib49)] to obtain question embeddings and compute cosine similarity between each pair of questions within the same document. The histogram of question similarity is shown in Figure [5](#S5.F5 "Figure 5 ‣ Comparison of Level-2 and Level-3 with directly extraction-based synthesis method. ‣ 5.3 Ablation Studies about Our SynthLLM Framework ‣ 5 Experiments ‣ Scaling Laws of Synthetic Data for Language Models"). We could clearly observe that questions generated by Level-2 exhibit lower similarity, indicating greater diversity among questions derived from the same document.

![Refer to caption](/html/2503.19551/assets/x5.png)


Figure 6: (a): The performances of Level-2, Level-3, and other augmentation methods on MATH benchmark. (b): the average performances across various benchmarks. The x axis denotes the sample number. The y axis represents the accuracy.

#### Comparison with other data augmentation methods on limited reference documents.

We compare our Level-2 and Level-3 methods with mainstream augmentation techniques, including rephrase [[53](#bib.bib53)] and persona augmentation [[12](#bib.bib12)]. We restrict the number of reference documents to a very small set and evaluate different methods for scaling high-quality question generation from a constrained document pool.
The effectiveness of each method will be measured by its ability to consistently improve model performance on downstream tasks as the dataset expands. To achieve this goal, We randomly select 2,000 documents and generate 25, 50, 75, …, 150 questions per document—a significantly large number—using both Level-2 and Level-3 methods.
This results in datasets of 50K, 100K, 150K, …, 300K questions for each method. For rephrase and persona augmentation methods, we first use Level-1 to extract existing questions from documents, yielding approximately 6,500 questions as seed dataset. We then expand this dataset to 300K questions using these two augmentation techniques. We train these datasets on Llama-3.1-8B model, with results presented in Figure [6](#S5.F6 "Figure 6 ‣ Comparison of Level-2 and Level-3 with directly extraction-based synthesis method. ‣ 5.3 Ablation Studies about Our SynthLLM Framework ‣ 5 Experiments ‣ Scaling Laws of Synthetic Data for Language Models"). Figure [6](#S5.F6 "Figure 6 ‣ Comparison of Level-2 and Level-3 with directly extraction-based synthesis method. ‣ 5.3 Ablation Studies about Our SynthLLM Framework ‣ 5 Experiments ‣ Scaling Laws of Synthetic Data for Language Models") (a) illustrates MATH accuracy of different methods, while Figure [6](#S5.F6 "Figure 6 ‣ Comparison of Level-2 and Level-3 with directly extraction-based synthesis method. ‣ 5.3 Ablation Studies about Our SynthLLM Framework ‣ 5 Experiments ‣ Scaling Laws of Synthetic Data for Language Models") (b) shows average accuracy across benchmarks.
Level-2 and Level-3 achieve significantly better performance on math benchmarks compared to augmentation methods. More importantly, as the number of generated questions per document increases to 150, our methods continue to show consistent improvements. In contrast, the performance of the rephrase and persona augmentation methods has already saturated as the dataset size expands. This demonstrates that compared with existing augmentation methods, our knowledge-guided methods more efficiently exploit limited reference documents by decomposing
and recombining knowledge concepts, thereby enabling more scalable high-quality synthetic question generation. In previous experiments, we generated only 5 questions for Level-2 and 3 questions for Level-3. The results presented here indicate that our method has not yet reached its full potential in document utilization.

![Refer to caption](/html/2503.19551/assets/graph/level123_3b_compare.png)


Figure 7: The results of fitting scaling curves using different formulations on the Llama-3.2-3B model. The x axis denotes the number of training tokens. The y axis represents the models’ error rates on MATH benchmark. The green points represent the data sizes used to fit the scaling laws, while the red points are used to test the prediction performance of the fitted curves.

### 5.4 Ablation Study on Scaling Law Form

We compare the rectified form Eq. ([2](#S3.E2 "In 3.1 Scaling Laws for Organic Data ‣ 3 Scaling Law of Synthetic Data ‣ Scaling Laws of Synthetic Data for Language Models")) and marginal form Eq. ([3](#S3.E3 "In 3.1 Scaling Laws for Organic Data ‣ 3 Scaling Law of Synthetic Data ‣ Scaling Laws of Synthetic Data for Language Models")) in terms of their effectiveness in fitting the scaling curve. The results of fitting scaling curves using these two formulations on the Llama-3.2-3B model are presented in Figure [7](#S5.F7 "Figure 7 ‣ Comparison with other data augmentation methods on limited reference documents. ‣ 5.3 Ablation Studies about Our SynthLLM Framework ‣ 5 Experiments ‣ Scaling Laws of Synthetic Data for Language Models"). The black dotted line represents the scaling curve fitted using the marginal form (Eq.([3](#S3.E3 "In 3.1 Scaling Laws for Organic Data ‣ 3 Scaling Law of Synthetic Data ‣ Scaling Laws of Synthetic Data for Language Models"))), while the green solid line corresponds to the fit using the rectified form (Eq.([2](#S3.E2 "In 3.1 Scaling Laws for Organic Data ‣ 3 Scaling Law of Synthetic Data ‣ Scaling Laws of Synthetic Data for Language Models"))). The green points indicate the data sizes used for fitting the scaling laws, whereas the red points are used to evaluate the predictive performance of the fitted curves. From the results, we observe that the marginal form fails to accurately fit the scaling curve and struggles to predict performance when scaling up the synthetic dataset. In contrast, the rectified form effectively captures the scaling law while also demonstrating strong predictive accuracy when scaling up the synthetic dataset. This underscores the importance of introducing the pre-learned size DlD\_{l} for scaling law of fine-tuning LLMs.

## 6 Conclusions and Future Work

In this work, we investigate the scaling laws of synthetic data and propose SynthLLM, a scalable web-scale data synthesis framework that systematically transforms pre-training data into high-quality synthetic datasets. Our findings show that synthetic data generated through SynthLLM consistently adheres to a rectified scaling law across various model sizes. By leveraging the fitted scaling law, we accurately predict model performance when the synthetic dataset size is doubled. These results suggest that systematically expanding synthetic datasets can continue to drive sustained and predictable performance improvements. Furthermore, our comparisons with other synthetic data generation and augmentation methods demonstrate that SynthLLM achieves superior performance and better scalability, providing a more effective approach for further enhancing model performance.

Our framework can be readily extended to other downstream domains, including code, physics and chemistry, and healthcare, expanding its applicability across diverse fields. Furthermore, SynthLLM is not limited to fine-tuning; we aim to explore its effectiveness in continued pre-training and even in the pre-training phase for synthetic data generation. Additionally, we will continue to refine and optimize our framework by developing more efficient strategies for leveraging pre-training data, with the goal of enhancing scaling efficiency of synthetic data.

## References

* Abdin et al. [2024]

  Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauffmann, et al.
  Phi-4 technical report.
  *arXiv preprint arXiv:2412.08905*, 2024.
* Anderson and Krathwohl [2001]

  Lorin W Anderson and David R Krathwohl.
  *A taxonomy for learning, teaching, and assessing: A revision of Bloom’s taxonomy of educational objectives: complete edition*.
  Addison Wesley Longman, Inc., 2001.
* Anthropic [2024a]

  Anthropic.
  Claude 3.5 sonnet, 2024a.
  URL <https://www.anthropic.com/news/claude-3-5-sonnet>.
* Anthropic [2024b]

  Anthropic.
  Claude: Developing a computer use model, 2024b.
  URL <https://www.anthropic.com/news/developing-computer-use>.
* Bahri et al. [2024]

  Yasaman Bahri, Ethan Dyer, Jared Kaplan, Jaehoon Lee, and Utkarsh Sharma.
  Explaining neural scaling laws.
  *Proceedings of the National Academy of Sciences*, 121(27):e2311878121, 2024.
* Bi et al. [2024]

  Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al.
  Deepseek llm: Scaling open-source language models with longtermism.
  *arXiv preprint arXiv:2401.02954*, 2024.
* Brown et al. [2020]

  Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al.
  Language models are few-shot learners.
  *Advances in neural information processing systems*, 33:1877–1901, 2020.
* Busbridge et al. [2025]

  Dan Busbridge, Amitis Shidani, Floris Weers, Jason Ramapuram, Etai Littwin, and Russ Webb.
  Distillation scaling laws.
  *arXiv preprint arXiv:2502.08606*, 2025.
* Chen et al. [2021]

  Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé de Oliveira Pinto, Jared Kaplan, Harrison Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba.
  Evaluating large language models trained on code.
  *CoRR*, abs/2107.03374, 2021.
* Cobbe et al. [2021]

  Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman.
  Training verifiers to solve math word problems.
  *CoRR*, abs/2110.14168, 2021.
* Council et al. [2005]

  National Research Council, Suzanne Donovan, John Bransford, et al.
  *How students learn*.
  National Academies Press Washington, DC, 2005.
* Ge et al. [2024]

  Tao Ge, Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, and Dong Yu.
  Scaling synthetic data creation with 1,000,000,000 personas.
  *arXiv preprint arXiv:2406.20094*, 2024.
* Gunasekar et al. [2023]

  Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, et al.
  Textbooks are all you need.
  *arXiv preprint arXiv:2306.11644*, 2023.
* He et al. [2024]

  Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun.
  Olympiadbench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems.
  In *ACL (1)*, pages 3828–3850. Association for Computational Linguistics, 2024.
* Hendrycks et al. [2021]

  Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt.
  Measuring mathematical problem solving with the MATH dataset.
  In *NeurIPS Datasets and Benchmarks*, 2021.
* Hernandez et al. [2021]

  Danny Hernandez, Jared Kaplan, Tom Henighan, and Sam McCandlish.
  Scaling laws for transfer.
  *arXiv preprint arXiv:2102.01293*, 2021.
* Hestness et al. [2017]

  Joel Hestness, Sharan Narang, Newsha Ardalani, Gregory Diamos, Heewoo Jun, Hassan Kianinejad, Md Mostofa Ali Patwary, Yang Yang, and Yanqi Zhou.
  Deep learning scaling is predictable, empirically.
  *arXiv preprint arXiv:1712.00409*, 2017.
* Hoffmann et al. [2022]

  Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al.
  Training compute-optimal large language models.
  *arXiv preprint arXiv:2203.15556*, 2022.
* Honovich et al. [2022]

  Or Honovich, Thomas Scialom, Omer Levy, and Timo Schick.
  Unnatural instructions: Tuning language models with (almost) no human labor.
  *arXiv preprint arXiv:2212.09689*, 2022.
* Kaplan et al. [2020]

  Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei.
  Scaling laws for neural language models.
  *arXiv preprint arXiv:2001.08361*, 2020.
* Lambert et al. [2024]

  Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al.
  T\\backslash" ulu 3: Pushing frontiers in open language model post-training.
  *arXiv preprint arXiv:2411.15124*, 2024.
* Lewkowycz et al. [2022]

  Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay V. Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra.
  Solving quantitative reasoning problems with language models.
  In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, *Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022*, 2022.
* Li et al. [2024a]

  Chen Li, Weiqi Wang, Jingcheng Hu, Yixuan Wei, Nanning Zheng, Han Hu, Zheng Zhang, and Houwen Peng.
  Common 7b language models already possess strong math capabilities.
  *arXiv preprint arXiv:2403.04706*, 2024a.
* Li et al. [2024b]

  Haoran Li, Qingxiu Dong, Zhengyang Tang, Chaojun Wang, Xingxing Zhang, Haoyang Huang, Shaohan Huang, Xiaolong Huang, Zeqiang Huang, Dongdong Zhang, et al.
  Synthetic data (almost) from scratch: Generalized instruction tuning for language models.
  *arXiv preprint arXiv:2402.13064*, 2024b.
* LI et al. [2024]

  Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu.
  Numinamath, 2024.
* Li et al. [2024]

  Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Omer Levy, Luke Zettlemoyer, Jason Weston, and Mike Lewis.
  Self-alignment with instruction backtranslation.
  In *ICLR*, 2024.
* Li et al. [2023]

  Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee.
  Textbooks are all you need ii: phi-1.5 technical report.
  *arXiv preprint arXiv:2309.05463*, 2023.
* Liao et al. [2024]

  Minpeng Liao, Chengxi Li, Wei Luo, Jing Wu, and Kai Fan.
  MARIO: math reasoning with code interpreter output - A reproducible pipeline.
  In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, *Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024*, pages 905–924. Association for Computational Linguistics, 2024.
* Lin et al. [2024]

  Haowei Lin, Baizhou Huang, Haotian Ye, Qinyu Chen, Zihao Wang, Sujian Li, Jianzhu Ma, Xiaojun Wan, James Zou, and Yitao Liang.
  Selecting large language model to fine-tune via rectified scaling law.
  *arXiv preprint arXiv:2402.02314*, 2024.
* Liu et al. [2024]

  Ruibo Liu, Jerry Wei, Fangyu Liu, Chenglei Si, Yanzhe Zhang, Jinmeng Rao, Steven Zheng, Daiyi Peng, Diyi Yang, Denny Zhou, et al.
  Best practices and lessons learned on synthetic data.
  *arXiv preprint arXiv:2404.07503*, 2024.
* Llama Team [2024]

  AI @ Meta Llama Team.
  The llama 3 herd of models, 2024.
* Long et al. [2024]

  Lin Long, Rui Wang, Ruixuan Xiao, Junbo Zhao, Xiao Ding, Gang Chen, and Haobo Wang.
  On llms-driven synthetic data generation, curation, and evaluation: A survey.
  *arXiv preprint arXiv:2406.15126*, 2024.
* Loshchilov and Hutter [2019]

  Ilya Loshchilov and Frank Hutter.
  Decoupled weight decay regularization.
  In *International Conference on Learning Representations*, 2019.
* Luo et al. [2023]

  Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang.
  Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct.
  *arXiv preprint arXiv:2308.09583*, 2023.
* Maini et al. [2024]

  Pratyush Maini, Skyler Seto, He Bai, David Grangier, Yizhe Zhang, and Navdeep Jaitly.
  Rephrasing the web: A recipe for compute and data-efficient language modeling.
  *arXiv preprint arXiv:2401.16380*, 2024.
* Mitra et al. [2023]

  Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agarwal, Xuxi Chen, Anastasia Razdaibiedina, Erik Jones, Kriti Aggarwal, et al.
  Orca 2: Teaching small language models how to reason.
  *arXiv preprint arXiv:2311.11045*, 2023.
* Muennighoff et al. [2023a]

  Niklas Muennighoff, Alexander Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, and Colin A Raffel.
  Scaling data-constrained language models.
  *Advances in Neural Information Processing Systems*, 36:50358–50376, 2023a.
* Muennighoff et al. [2023b]

  Niklas Muennighoff, Alexander M Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, and Colin Raffel.
  Scaling data-constrained language models.
  *arXiv preprint arXiv:2305.16264*, 2023b.
* OpenAI [2022]

  OpenAI.
  Introducing ChatGPT, 2022.
  URL <https://openai.com/index/chatgpt/>.
* Penedo et al. [2024]

  Guilherme Penedo, Hynek Kydlíček, Anton Lozhkov, Margaret Mitchell, Colin A Raffel, Leandro Von Werra, Thomas Wolf, et al.
  The fineweb datasets: Decanting the web for the finest text data at scale.
  *Advances in Neural Information Processing Systems*, 37:30811–30849, 2024.
* Radford et al. [2019]

  Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al.
  Language models are unsupervised multitask learners.
  *OpenAI blog*, 1(8):9, 2019.
* Rosenfeld et al. [2019]

  Jonathan S Rosenfeld, Amir Rosenfeld, Yonatan Belinkov, and Nir Shavit.
  A constructive prediction of the generalization error across scales.
  *arXiv preprint arXiv:1909.12673*, 2019.
* [43]

  Zhengyang Tang, Xingxing Zhang, Benyou Wang, and Furu Wei.
  Mathscale: Scaling instruction tuning for mathematical reasoning.
  In *Forty-first International Conference on Machine Learning*.
* Team et al. [2025]

  Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al.
  Kimi k1. 5: Scaling reinforcement learning with llms.
  *arXiv preprint arXiv:2501.12599*, 2025.
* team [2024]

  Mistral AI team.
  Large enough, 2024.
  URL <https://mistral.ai/news/mistral-large-2407>.
* Toshniwal et al. [2024]

  Shubham Toshniwal, Wei Du, Ivan Moshkov, Branislav Kisacanin, Alexan Ayrapetyan, and Igor Gitman.
  Openmathinstruct-2: Accelerating ai for math with massive open-source instruction data.
  *arXiv preprint arXiv:2410.01560*, 2024.
* Toshniwal et al. [2025]

  Shubham Toshniwal, Wei Du, Ivan Moshkov, Branislav Kisacanin, Alexan Ayrapetyan, and Igor Gitman.
  Openmathinstruct-2: Accelerating AI for math with massive open-source instruction data.
  In *The Thirteenth International Conference on Learning Representations*, 2025.
* Villalobos et al. [2024]

  Pablo Villalobos, Anson Ho, Jaime Sevilla, Tamay Besiroglu, Lennart Heim, and Marius Hobbhahn.
  Position: Will we run out of data? limits of llm scaling based on human-generated data.
  In *Forty-first International Conference on Machine Learning*, 2024.
* Wang et al. [2020]

  Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou.
  Minilm: Deep self-attention distillation for task-agnostic compression of pre-trained transformers.
  *Advances in neural information processing systems*, 33:5776–5788, 2020.
* Wang et al. [2022]

  Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi.
  Self-instruct: Aligning language models with self-generated instructions.
  *arXiv preprint arXiv:2212.10560*, 2022.
* Xu et al. [2024]

  Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, and Bill Yuchen Lin.
  Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing.
  *arXiv preprint arXiv:2406.08464*, 2024.
* Yang et al. [2024]

  An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al.
  Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement.
  *arXiv preprint arXiv:2409.12122*, 2024.
* Yu et al. [2024]

  Longhui Yu, Weisen Jiang, Han Shi, Jincheng YU, Zhengying Liu, Yu Zhang, James Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu.
  Metamath: Bootstrap your own mathematical questions for large language models.
  In *The Twelfth International Conference on Learning Representations*, 2024.
* Yuan et al. [2025]

  Weizhe Yuan, Jane Yu, Song Jiang, Karthik Padthe, Yang Li, Dong Wang, Ilia Kulikov, Kyunghyun Cho, Yuandong Tian, Jason E Weston, et al.
  Naturalreasoning: Reasoning in the wild with 2.8 m challenging questions.
  *arXiv preprint arXiv:2502.13124*, 2025.
* Yue et al. [2024]

  Xiang Yue, Tianyu Zheng, Ge Zhang, and Wenhu Chen.
  MAmmoTH2: Scaling instructions from the web.
  In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024.
* Zhou et al. [2024]

  Kun Zhou, Beichen Zhang, jiapeng wang, Zhipeng Chen, Xin Zhao, Jing Sha, Zhichao Sheng, Shijin Wang, and Ji-Rong Wen.
  Jiuzhang3.0: Efficiently improving mathematical reasoning by training small data synthesis models.
  In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024.

## Appendix A Used Prompts

In this section, we present prompts used in our SynthLLM framework.

Here is an article crawl from the web, which our classifier has identified as having significant educational value for students learning math.
As a senior \*\*math\*\* instructor, your task is to create \*\*challenging computation-based math questions\*\*. These questions should be suitable for various contexts, such as homework assignments, exams, interview preparations, classroom activities, competitions, and tutoring sessions while enhancing students’ reasoning and critical-thinking skills. Ensure that questions are \*\*non-redundant\*\*, precise, and engaging.
### Guidelines for Creating Computation-based Questions:
1. \*\*Assess Suitability\*\*: If this article does not contain math-related content that can be used to generate engaging and solvable questions, please directly output "NOT SUITABLE for creating questions."
2. \*\*Generate Questions\*\*: If the article is suitable for creating math questions, generate \*\*1 to 5\*\* questions based on the richness and depth of the article content. For articles covering multiple topics, aim to generate more questions to ensure coverage:
   - Each question must be solvable independently and should not rely on answers from previous questions.
   - If a question closely resembles the original text, append "<original\_question>" at the end of the question.
   - If a question is newly created, append "<newly\_created>" at the end of the question.
3. \*\*Content Alignment\*\*: Your math questions must exclusively draw from the content of the article, ensuring they are directly aligned with the concepts presented.
4. \*\*Use Original Questions\*\*: Use original questions from the article whenever possible. However, feel free to rephrase them for clarity and improved understanding.
5. \*\*Create New Questions\*\*: Attempt to formulate \*\*newly diverse and challenging questions\*\* that explore different aspects of the content presented in the article.
6. \*\*Self-Contained Questions\*\*: Ensure that each question is self-contained, meaning students do not need to read the article to answer them.
7. \*\*Logical Consistency\*\*: Verify that the questions are \*\*logically sound and directly aligned with the mathematical principles\*\* in the article. You MUST minimize the use of \*\*sub-questions\*\*, unless they are essential to the problem’s complexity.
8. \*\*Clarity and Precision\*\*:
   - Use precise and unambiguous language.
   - Write all mathematical expressions or formulas in LaTeX for clarity.
   - Clearly state all assumptions or conditions.
   - The answer should either be exact, or if not possible, then the question should clearly say the
   answer is only expected to be approximately correct.
#### Article
{{ text }}
#### Output Format
For each question, provide the following information:
- \*\*Question Content\*\*: The actual math question.
- \*\*School Level\*\*: Specify the school level (e.g., <elementary>, <middle\_school>,
<high\_school>, <college>, <grad\_school>, <competition>).
- \*\*Originality Tag\*\*: Append "<original\_question>" for original questions from the article or"<newly\_created>" for newly created questions.
Example Output:
<Q1>
Question: Content
Orig\_tag:<original\_question>
Level:<high\_school>
</Q1>
<Q2>
Question: Content
Orig\_tag:<newly\_created>
Level:<college>
</Q2>


Figure 8: Prompt for Level-1. The {{ text }} is the placeholder for inserting the reference document.



As a senior \*\*math\*\* instructor, your task is to create \*\*diverse and challenging computation-based math questions\*\*. These questions should demonstrate the application of the provided topics and key concepts while enhancing students’ reasoning and critical-thinking skills. Ensure that questions are \*\*non-redundant\*\*, precise, and engaging.
### Guidelines for Creating Diverse and Challenging Computation-based Questions:
1. \*\*Concept Selection\*\*:
   - Randomly select \*\*up to 2-3 distinct key concepts\*\* from the provided list for each question.
   - Ensure \*\*broad coverage\*\* of the provided concepts across the generated questions, avoiding over-reliance
   on a limited subset of concepts.
   - Avoid repeating the same \*\*concept combinations\*\* or \*\*computational approach\*\* across questions.
2. \*\*Diversity and Challenge\*\*:
   - Ensure that each question explores \*\*different combinations of key concepts\*\* and is \*\*sufficiently
   challenging\*\* (e.g., requiring multi-step computations, integrating real-world scenarios, involving abstract or
   advanced reasoning.).
3. \*\*Clarity and Precision\*\*:
   - Verify that the questions are \*\*logically sound\*\*.
   - Use precise and unambiguous language.
   - Write all mathematical expressions or formulas in LaTeX for clarity.
   - Clearly state all assumptions or conditions.
4. \*\*Reference Material\*\*:
   - Use the provided \*\*reference article\*\* as a source of inspiration for generating \*\*unique, diverse, and
   challenging questions\*\*.
   - The reference material is intended to:
   - Supplement the concept list by introducing \*\*novel perspectives\*\*, \*\*contexts\*\*, or \*\*applications\*\*.
   - Help create questions that are \*\*more complex, realistic, or uncommon\*\* in traditional teaching scenarios.
   - Serve as a resource to craft \*\*real-world scenarios\*\* or \*\*abstract extensions\*\* beyond the given concepts.
5. \*\*Output Diversity\*\*:
   - Create between \*\*1 to 5 questions\*\*.
   - Ensure each question is unique in \*\*structure\*\*, \*\*approach\*\*, and \*\*concept usage\*\*.
   - Minimize the use of \*\*sub-questions\*\*, unless they are essential to the problem’s complexity.
   - The answer should either be exact, or if not possible, then the question should clearly say the answer is only
   expected to be approximately correct.
### Inputs:
- \*\*Article\*\*:
   {{ text }}
- \*\*Concept List\*\*:
   {{ concept }}
#### Output Format:
<Q1>
Selected Concepts: [Only insert 2-3 concepts here]
Question: [Only insert question here]
</Q1>
<Q2>
Selected Concepts: [Only insert 2-3 concepts here]
Question: [Only insert question here]
</Q2>


Figure 9: Prompt for Level-2. The {{ text }} and {{ concept }} are the placeholders for inserting the reference document and meta-information (e.g., topics, key concepts) extracted from document.

As a senior \*\*math\*\* instructor, your task is to create \*\*diverse and challenging computation-based math questions\*\* based on provided topics and knowledge points. These questions should demonstrate the application of the provided topics and key concepts while enhancing students’ reasoning and critical-thinking skills. Ensure that questions are \*\*non-redundant\*\*, precise, and engaging.
You will be provided with a list of key mathematical concepts spanning various topics and two relevant reference materials.
### Guidelines for Creating Diverse and Challenging Computation-based Questions:
1. \*\*Concept Selection\*\*:
   - Adhere to the Provided Topics: Ensure that each question aligns closely with the given topics.
   - Incorporate Multiple Concepts about Different Topics: Each question should encompass \*\*2 or 3 key
   concepts about different math topics\*\*.
   - Ensure \*\*broad coverage\*\* of the provided concepts across the generated questions, avoiding
   \*\*over-reliance\*\* on simple or common applications of concepts.
   - Avoid repeating the same \*\*concept combinations\*\* or \*\*computational approach\*\* across questions.
2. \*\*Diversity and Challenge\*\*:
   - Encourage \*\*Cross-Topic Thinking\*\*: By integrating concepts about different math topics, questions
   will promote holistic understanding and application of mathematical principles.
   - \*\*Leverage the Two Reference Materials\*\*: The combination of both reference materials provides
   a \*\*broader and more diverse context\*\*, allowing for the creation of questions that explore a wider range
   of scenarios and applications. Use this to generate questions that challenge students in both familiar and
   novel contexts.
   - Ensure questions explore \*\*different perspectives\*\* and \*\*applications\*\* of the key concepts. Ensure each
   question is \*\*sufficiently challenging\*\* (e.g., requiring multi-step computations, integrating real-world
   scenarios, involving abstract or advanced reasoning.).
3. \*\*Clarity and Precision\*\*:
   - Use precise and unambiguous language.
   - Write all mathematical expressions or formulas in LaTeX for clarity.
   - Clearly state all assumptions or conditions.
4. \*\*Reference Material\*\*:
   - Use the provided \*\*reference articles about different topics\*\* as sources of inspiration for
   generating \*\*unique, diverse, and challenging questions\*\*.
   - The combination of these two materials allows you to create questions with \*\*more varied perspectives,
   contexts, and applications\*\*, which can help test students’ abilities to apply concepts in different situations.
   - The reference material is intended to:
   - Supplement the concept list by introducing \*\*novel perspectives\*\*, \*\*contexts\*\*, or \*\*applications\*\*.
   - Help create questions that are \*\*more complex, much harder, and uncommon\*\* in traditional
   teaching scenarios.
   - Serve as a resource to craft \*\*real-world scenarios\*\* or \*\*abstract extensions\*\* beyond the
   given concepts.
5. \*\*Output Diversity\*\*:
   - Create between \*\*1 to 3 questions\*\*.
   - Ensure each question is unique in \*\*structure\*\*, \*\*approach\*\*, and \*\*concept usage\*\*.
   - Minimize the use of \*\*sub-questions\*\*, unless they are essential to the problem’s complexity.
   - The answer should either be exact, or if not possible, then the question should clearly say the answer is
   only expected to be approximately correct.
### Inputs:
- \*\*Article\*\*:
   {{ text }}
- \*\*Concept List\*\*:
   {{ concept }}
#### Output Format:
<Q1>
Selected Concepts: [Only insert 2-3 concepts here]
Question: [Only insert question here]
</Q1>
<Q2>
Selected Concepts: [Only insert 2-3 concepts here]
Question: [Only insert question here]
</Q2>


Figure 10: Prompt for Level-3. The {{ text }} and {{ concept }} are the placeholders for inserting reference documents and meta-information sampled from knowledge graph.

Here is an article crawl from the web, which our classifier has identified as having significant educational value for students learning math.
Your task is to analyze this article and extract educational materials, specifically focusing on topics and key concepts that can enhance students’ understanding of mathematics and improve their problem-solving skills.
Pay special attention to uncommon but important mathematical concepts that are crucial for a deeper understanding.
## Tasks
1. \*\*Determine Educational Level:\*\*
   - Identify the appropriate educational level for the article based on its content. Choose from the
   following options:
   - Primary School
   - Middle School
   - High School
   - College
   - Graduate School
   - Competition
   - Other
2. \*\*Identify Subject Area:\*\*
   - Specify the primary subject area of mathematics to which the article belongs (e.g., Calculus,
   Geometry, Algebra, etc.).
3. \*\*Extract Topics and Key Concepts:\*\*
   - \*\*Topics:\*\*
   - List \*\*1 to 5\*\* main topics covered in the article.
   - Use terms commonly recognized in academia or industry.
   - \*\*Key Concepts:\*\*
   - For each identified topic, list \*\*5 to 20\*\* related key concepts.
   - Ensure these concepts are clearly articulated using standard academic or industry terms.
## Guidelines:
- \*\*Terminology:\*\* Use precise and widely recognized academic or industry terminology for subjects, topics, and key concepts to maintain consistency and clarity.
- \*\*Educational Level Selection:\*\* If appropriate, restrict the educational level to one of the following: "Primary School", "Middle School", "High School", "College", "Graduate School", or "Competition" to ensure accurate categorization.
## Text
{{ text }}
## Output Format
<level>Educational Level</level>
<subject>Subject Area</subject>
<topic>
Topics:
1. topic 1
2. topic 2
</topic>
<key\_concept>
Key Concepts:
1. topic 1:
   1.1. key concept
   1.2. key concept
   …
2. topic 2:
   2.1. key concept
   …
…
</key\_concept>
## Output


Figure 11: Prompt for extracting topics and key concepts from the document. The {{ text }} is the placeholders for inserting reference document.

## Appendix B Examples about Extracted Topics and Key Concepts from Reference Document

In this section, we demonstrate some examples about extracted topics and key concepts from reference document by using previous prompt shown in Figure [11](#A1.F11 "Figure 11 ‣ Appendix A Used Prompts ‣ Scaling Laws of Synthetic Data for Language Models"). We adopt a relative large Language model, Mistral-Large-Instruct-2407. The examples are shown in Figure [12](#A2.F12 "Figure 12 ‣ Appendix B Examples about Extracted Topics and Key Concepts from Reference Document ‣ Scaling Laws of Synthetic Data for Language Models") and [13](#A2.F13 "Figure 13 ‣ Appendix B Examples about Extracted Topics and Key Concepts from Reference Document ‣ Scaling Laws of Synthetic Data for Language Models"). We can observe that Mistral-Large-Instruct-2407 is capable of extracting well-structured topics and their corresponding key concepts from the document.

<level>High School</level>
<subject>Trigonometry</subject>
<topic>
Topics:
1. Trigonometric Functions and Identities
2. Geometry on a Sphere
3. Applications of Trigonometry
4. Complex Numbers and Trigonometry
5. Derivations and Proofs in Trigonometry
</topic>
<key\_concept>
Key Concepts:
1. Trigonometric Functions and Identities:
   1.1. Sine, Cosine, and Tangent Functions
   1.2. Trigonometric Identities (e.g., Pythagorean, Co-function, Sum and Difference)
   1.3. Law of Sines and Law of Cosines
   1.4. Inverse Trigonometric Functions
   1.5. Trigonometric Equations and Their Solutions
2. Geometry on a Sphere:
   2.1. Latitude and Longitude
   2.2. Circumference of a Circle on a Sphere
   2.3. Equation of a Circle on a Sphere
   2.4. Spherical Trigonometry
   2.5. Haversine Formula for Distance Calculation
3. Applications of Trigonometry:
   3.1. Circular Motion and Angular Velocity
   3.2. Calculating Angles in Right Triangles
   3.3. Real-world Problems Involving Trigonometry (e.g., Crossing a Canyon)
   3.4. Computing Sine and Cosine without a Calculator
   3.5. Practical Uses of Trigonometric Functions in Engineering and Physics
4. Complex Numbers and Trigonometry:
   4.1. Euler’s Formula and De Moivre’s Theorem
   4.2. Complex Numbers and Trigonometric Functions
   4.3. Roots of Unity and Their Relationship to Trigonometry
   4.4. Inverse Hyperbolic Functions (e.g., Arcsinh)
   4.5. Complex Numbers in Trigonometric Identities
5. Derivations and Proofs in Trigonometry:
   5.1. Derivation of Trigonometric Identities
   5.2. Proof of the Law of Sines and Law of Cosines
   5.3. Derivation of Sum and Difference Formulas
   5.4. Proof of the Haversine Formula
   5.5. Derivation of Heron’s Formula
</key\_concept>


Figure 12: An example about High School’s Trigonometry.

<level>College</level>
<subject>Vector Calculus</subject>
<topic>
Topics:
1. Surface Integrals of Vector Fields
2. Oriented Surfaces
3. Unit Normal Vectors
4. Flux of a Vector Field
5. Parametric Surfaces
</topic>
<key\_concept>
Key Concepts:
1. Surface Integrals of Vector Fields:
   1.1. Definition of surface integral of a vector field
   1.2. Flux of a vector field across a surface
   1.3. Application of surface integrals in fluid dynamics
   1.4. Evaluation of surface integrals using parametric surfaces
   1.5. Surface integrals over closed surfaces
2. Oriented Surfaces:
   2.1. Definition of an oriented surface
   2.2. Positive and negative orientations
   2.3. Unit normal vectors and their role in orientation
   2.4. Orientation conventions for closed surfaces
   2.5. Impact of orientation on surface integrals
3. Unit Normal Vectors:
   3.1. Definition and calculation of unit normal vectors
   3.2. Gradient vector and its role in finding normal vectors
   3.3. Normal vectors for surfaces given by z = f(x, y)
   3.4. Normal vectors for parametric surfaces
   3.5. Adjusting normal vectors to match desired orientation
4. Flux of a Vector Field:
   4.1. Definition of flux
   4.2. Physical interpretation of flux in fluid dynamics
   4.3. Calculation of flux using surface integrals
   4.4. Flux across closed surfaces
   4.5. Application of flux in Gauss’s Law
5. Parametric Surfaces:
   5.1. Definition and representation of parametric surfaces
   5.2. Calculation of normal vectors for parametric surfaces
   5.3. Evaluation of surface integrals using parametric surfaces
   5.4. Parameterization of common surfaces (e.g., spheres, cylinders)
   5.5. Conversion between parametric and non-parametric forms
</key\_concept>


Figure 13: An example about the College’s Vector Calculus.

[◄](/html/2503.19550)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2503.19551)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2503.19551)
[View original  
on arXiv](https://arxiv.org/abs/2503.19551)[►](/html/2503.19552)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Sat Apr 5 16:42:31 2025 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
