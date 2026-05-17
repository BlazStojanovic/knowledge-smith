---
arxiv: '2601.00364'
authors:
- Jiandong Shao
- Raphael Tang
- Crystina Zhang
- Karin Sevegnani
- Pontus Stenetorp
- Jianfei Yang
- Yao Lu
parser: ar5iv
retrieved: '2026-05-15'
source: paper
title: The Role of Mixed-Language Documents for Multilingual Large Language Model
  Pretraining
url: https://arxiv.org/abs/2601.00364
year: 2026
---

[2601.00364] The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining



# The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining

Jiandong Shao,2 Raphael Tang,1 Crystina Zhang,3 Karin Sevegnani,4
  
Pontus Stenetorp,1,5 Jianfei Yang,2 Yao Lu1
  
1University College London  2Nanyang Technological University
  
3University of Waterloo   4NVIDIA   5National Institute of Informatics

###### Abstract

Multilingual large language models achieve impressive cross-lingual performance despite largely monolingual pretraining. While bilingual data in pretraining corpora is widely believed to enable these abilities, details of its contributions remain unclear. We investigate this question by pretraining models from scratch under controlled conditions, comparing the standard web corpus with a monolingual-only version that removes all multilingual documents.
Despite constituting only 2% of the corpus, removing bilingual data causes translation performance to drop 56% in BLEU, while behaviour on cross-lingual QA and general reasoning tasks remains stable, with training curves largely overlapping the baseline.
To understand this asymmetry, we categorize bilingual data into parallel (14%), code-switching (72%), and miscellaneous documents (14%) based on the semantic relevance of content in different languages. We then conduct granular ablations by reintroducing parallel or code-switching data into the monolingual-only corpus. Our experiments reveal that parallel data almost fully restores translation performance (91% of the unfiltered baseline), whereas code-switching contributes minimally. Other cross-lingual tasks remain largely unaffected by either type. These findings reveal that translation critically depends on systematic token-level alignments from parallel data, whereas cross-lingual understanding and reasoning appear to be achievable even without bilingual data.

The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining

Jiandong Shao,2 Raphael Tang,1 Crystina Zhang,3 Karin Sevegnani,4

Pontus Stenetorp,1,5 Jianfei Yang,2 Yao Lu1

1University College London  2Nanyang Technological University

3University of Waterloo   4NVIDIA   5National Institute of Informatics

## 1 Introduction

Figure 1: 
Performance on WMT14 for different pretraining setups. FineWeb: multilingual web data from FineWeb and FineWeb-2; MonoWeb: multilingual web data with bilingual documents removed.

Large language models (LLMs), when pretrained on web-collected data from different language sources, exhibit remarkable emergent capabilities in cross-lingual understanding despite not being pretrained using multilingual-specific objectives (Devlin et al., [2019](#bib.bib1 "BERT: pre-training of deep bidirectional transformers for language understanding"); Achiam et al., [2023](#bib.bib32 "GPT-4 technical report"); Yang et al., [2024](#bib.bib33 "Qwen2.5 technical report")).
Existing research attributes this behaviour not only to a sufficient amount of data from different languages, but also to specific documents where multiple languages co-occur in the same context (Chaudhary et al., [2020](#bib.bib9 "DICT-mlm: improved multilingual pre-training using bilingual dictionaries"); Chi et al., [2020](#bib.bib11 "InfoXLM: an information-theoretic framework for cross-lingual language model pre-training"); Wang et al., [2025](#bib.bib30 "Investigating and scaling up code-switching for multilingual language model pre-training")).
Motivated by this observation, multilingual pretraining strategies have often incorporated multilingual data, under the hypothesis that mixed-language exposure uniformly benefits cross-lingual tasks (Yoo et al., [2024](#bib.bib31 "Code-switching curriculum learning for multilingual transfer in LLMs"); Wang et al., [2025](#bib.bib30 "Investigating and scaling up code-switching for multilingual language model pre-training")).

|  |  |
| --- | --- |
| Category | Example |
| Parallel | Magnifique et lumineux loft Toronto de 1 chambre avec plafonds de 10 pi et grande terrasse extérieure comprenant un barbecue…. |
|  | Beautiful, bright one bedroom Toronto loft with 10ft ceilings and large outdoor terrace including barbeque…… |
|  | [Paragraph-aligned translation with systematic cross-lingual correspondence] |
| Code-switching | The people, filled with joy, chant the anthem “A qua ben fé! A qua ben fé! La tarascou a rou un bré!” |
|  | [Natural language mixing within shared discourse context] |
| Miscellaneous | …and in some cases whether to let the fires burn to create regeneration in the forest. Vous devez avoir la dernière version de Flash Player installée. |
|  | [French text about Flash Player, semantically unrelated to the English content] |

Table 1: Examples of MonoWeb filtered data. We classify documents into three categories: documents with clear parallel structure, documents that exhibit code-switch behaviour, and miscellaneous documents.

However, the high cost of pretraining and large-scale pretraining data classfication has constrained the scope of existing explorations of the role of multilingual data. Studies that rely on continual pretraining typically build on models that may have already been exposed to related data during pretraining, which makes the role of multilingual data more difficult to disentangle.
Among the few works that investigate multilingual data at the pretraining stage, existing studies Briakou et al. ([2023](#bib.bib26 "Searching for needles in a haystack: on the role of incidental bilingualism in palm’s translation capability")); Qorib et al. ([2025](#bib.bib34 "Just go parallel: improving the multilingual capabilities of large language models")); Wang et al. ([2025](#bib.bib30 "Investigating and scaling up code-switching for multilingual language model pre-training")) do not provide a systematic analysis of its role, but instead focus on specific settings or mechanisms.
To this end, we aim to conduct a thorough analysis of the pretraining corpus and design a controlled pretraining setup to explicitly reveal the role of multilingual data.

We construct a monolingual web corpus by filtering out all documents containing more than one language from standard web-collected data. This procedure removes fewer than 2% of documents, making fine-grained analysis of multilingual data feasible at scale.
We then pretrain multilingual LLMs from scratch under two setups: MonoWeb, using the filtered corpus, and FineWeb, using the original web data.
Despite accounting for only 2% of pretraining data, multilingual documents are critical for machine translation. Removing them causes BLEU scores to drop from 22.3 to 9.8 (a 56% relative decrease), effectively collapsing translation performance. In contrast, other cross-lingual tasks are substantially less affected: cross-lingual QA drops by 10% on average, while understanding and reasoning tasks vary by at most 4%. This asymmetry highlights the nuanced role of pretraining data across different multilingual tasks.

To better understand this phenomenon, we analyze the composition of the removed multilingual documents. We find that most consist of bilingual content, which can be grouped into three categories ([Table 1](#S1.T1 "Table 1 ‣ 1 Introduction ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining")): (i) *parallel documents* (14%), providing aligned translations with explicit cross-lingual correspondence, such as multilingual Airbnb webpages; (ii) *code-switching documents* (72%), where languages naturally alternate within shared discourse, as commonly observed in user-generated content on platforms like Pinterest; and (iii) *miscellaneous documents* (14%), where multiple languages co-occur without meaningful semantic alignment.
We then isolate the contributions of different bilingual data types through controlled pretraining from scratch. Our results show that parallel data, despite comprising only 14% of bilingual documents, is the dominant factor for translation performance: reintroducing only parallel data yields a 106% improvement over MonoWeb, largely recovering performance relative to FineWeb (BLEU 20.220.2 vs. 22.322.3). In contrast, reintroducing code-switching data provides only marginal gains (BLEU 12.412.4 vs. 9.89.8), with little effect on other cross-lingual tasks.
Finally, we analyze the underlying causes of this asymmetry. We find that removing bilingual data primarily disrupts lexical-level cross-lingual alignment, leading to severe translation failures, while sentence-level alignment remains largely preserved, explaining the robustness of non-translation tasks.

To summarise, our contributions are threefold:

1. 1.

   We introduce a monolingual dataset, MonoWeb, together with a detailed analysis of multilingual content, pretrain models from scratch to study multilingual behavior without mixed-language exposure, and open-source both the dataset and models.
2. 2.

   Through pretraining from scratch, we demonstrate a task-dependent sensitivity to bilingual data: machine translation critically depends on a tiny fraction (less than 2%) of bilingual documents, whereas other cross-lingual understanding and reasoning tasks remain largely unaffected. We further show that different types of bilingual data contribute unequally, with parallel data playing a disproportionately critical role.
3. 3.

   We provide an in-depth failure mode and representation-level analysis, revealing that the degradation in translation performance is driven by the loss of lexical-level alignment, while sentence-level alignment remains largely preserved.

## 2 Related Work

Multilingual data is widely assumed to drive cross-lingual capabilities in multilingual models. Parallel corpora (sentence-aligned translations) are well-known to be essential for machine translation (Brown et al., [1993](#bib.bib40 "The mathematics of statistical machine translation: parameter estimation")), enabling multilingual MT systems to bridge high- and low-resource languages (Johnson et al., [2016](#bib.bib41 "Google’s multilingual neural machine translation system: enabling zero-shot translation"); Fan et al., [2020](#bib.bib42 "Beyond english-centric multilingual machine translation"); team et al., [2022](#bib.bib43 "No language left behind: scaling human-centered machine translation")). Beyond parallel data, naturally occurring code-switching, where languages alternate within the same discourse, has also attracted attention as a potential mechanism for cross-lingual alignment. Prior work shows that using code-switching for data augmentation can improve zero-shot transfer during finetuning (Qin et al., [2020](#bib.bib44 "CoSDA-ML: multi-lingual code-switching data augmentation for zero-shot cross-lingual NLP")), and that curriculum learning with code-switching enhances transfer to low-resource languages (Yoo et al., [2024](#bib.bib31 "Code-switching curriculum learning for multilingual transfer in LLMs")). These results have motivated practitioners to incorporate multilingual content under the assumption that mixed-language exposure benefits cross-lingual tasks (Qorib et al., [2025](#bib.bib34 "Just go parallel: improving the multilingual capabilities of large language models")).

However, most existing studies focus on finetuning or continued training, which remains limited because models may have already been exposed to similar data during the pretraining stage. Among the few works that investigate multilingual data in the pretraining stage, most focus on specific aspects rather than systematically studying its role: one primarily characterizes incidental bilingualism in existing corpora (Briakou et al., [2023](#bib.bib26 "Searching for needles in a haystack: on the role of incidental bilingualism in palm’s translation capability")), another uses generated data to study curriculum learning effects (Qorib et al., [2025](#bib.bib34 "Just go parallel: improving the multilingual capabilities of large language models")), and a third explores synthetic code-switching for cross-lingual transfer (Wang et al., [2025](#bib.bib30 "Investigating and scaling up code-switching for multilingual language model pre-training")). This leaves a gap in the understanding of the role of multilingual data during pretraining, particularly regarding the differential contributions of parallel versus code-switching data. We aim to fill this gap by systematically ablating different bilingual data types and studying their impact on multilingual LLMs.

## 3 MonoWeb Pretraining Data

To study the heterogeneous role of bilingual data, we first construct a multilingual corpus by sampling 60B tokens per language from FineWeb-Edu (Lozhkov et al., [2024](#bib.bib24 "FineWeb-edu: the finest collection of educational content")) (English) and FineWeb2 (Penedo et al., [2025](#bib.bib25 "FineWeb2: one pipeline to scale them all – adapting pre-training data processing to every language")) (German, Spanish, French), totaling 240B tokens. We then perform a systematic characterization of bilingual documents in this corpus, focusing on English-paired bilingual content (en-de, en-es, en-fr) as English serves as the current dominant lingua franca for cross-lingual scenarios.

### 3.1 Bilingual Data Identification

We identify bilingual documents through a two-stage pipeline, combining rule-based filtering with LLM-based classification to ensure both scalability and accuracy.

#### Stage 1: Candidate Detection via Entropy-based Filtering.

We first detect candidate bilingual documents using language-level entropy as a proxy for language mixing. For each document, we perform sentence segmentation using NLTK (Bird et al., [2009](#bib.bib28 "Natural language processing with python: analyzing text with the natural language toolkit")) and apply fastText language identification (Joulin et al., [2016](#bib.bib27 "Bag of tricks for efficient text classification")) to compute language confidence scores for each sentence. Taking English-French as an example, for each sentence sis\_{i} with length lil\_{i}, fastText outputs confidence scores for English (pienp\_{i}^{\text{en}}) and French (pifrp\_{i}^{\text{fr}}). We then compute a document-level language distribution by aggregating sentence-level scores weighted by sentence length:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Pdoclang=∑ili⋅pilang∑ili,P\_{\text{doc}}^{\text{lang}}=\frac{\sum\_{i}l\_{i}\cdot p\_{i}^{\text{lang}}}{\sum\_{i}l\_{i}}, |  | (1) |

where lang∈{en,fr}\text{lang}\in\{\text{en},\text{fr}\}. After normalization, we obtain a probability distribution over the two languages for the entire document. We compute the entropy of this distribution:

|  |  |  |  |
| --- | --- | --- | --- |
|  | H=−∑langPdoclang​log⁡Pdoclang.H=-\sum\_{\text{lang}}P\_{\text{doc}}^{\text{lang}}\log P\_{\text{doc}}^{\text{lang}}. |  | (2) |

Documents with entropy above a threshold τ=0.1\tau=0.1 (indicating substantial mixing of both languages) are marked as bilingual candidates.
We empirically selected this threshold by examining the distribution of entropy values and verifying that it effectively captures documents with substantial language mixing.
This stage serves as a coarse filtering that optimizes for the recall of potential bilingual data,
while maintaining computational efficiency.
As a result, 5% of the corpus is retained and can be precessed more computationally expensive methods during the subsequent verification stage.

#### Stage 2: LLM-based Classification.

To distinguish different types of bilingual relationships, we employ Llama-3.3-70B-Instruct (Dubey et al., [2024](#bib.bib23 "The Llama 3 herd of models")) for a two-step classification process, whose reliability has been validated through human evaluation. First, the model verifies whether each candidate is genuinely bilingual, which aims to filter out the false negatives
included by entropy-based filtering.
We consider the resulting set of documents after this step as the final verified bilingual documents, which consists of approximately 2% of the entire corpus.
Second, based on the semantic relationship of contents in different languages, the verified bilingual documents are classified into one of the three categories:

* •

  Parallel documents: Paragraph-aligned translations where languages express identical semantic content with systematic correspondences (e.g., dictionaries, translated website; the example in [Table 1](#S1.T1 "Table 1 ‣ 1 Introduction ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining") is from Airbnb).
* •

  Code-switching documents: Documents where both languages appear with semantic relationships but without systematic alignment. This includes naturally occurring mixed-language discourse (e.g., multilingual forum discussions), articles with embedded foreign quotations or terminology, and documents where languages serve complementary communicative functions. Crucially, unlike parallel data, the two languages do not provide translations of each other but rather contribute distinct yet related semantic content.
* •

  Miscellaneous documents: Documents where multiple languages co-occur without meaningful cross-lingual semantic relationships. This category primarily consists of web artifacts such as multilingual boilerplate, advertisements in different languages, or navigation elements appended to otherwise monolingual content.

This two-stage approach balances computational efficiency with classification accuracy: entropy-based filtering reduces the search space from the full corpus to  5% candidates, while LLM classification provides semantic nuance that rule-based methods lack. Table [1](#S1.T1 "Table 1 ‣ 1 Introduction ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining") shows representative examples for each category. The resulting taxonomy enables granular ablations to isolate the effects of different bilingual data types during pretraining.

|  |  |  |  |
| --- | --- | --- | --- |
| Data Type | en-de | en-es | en-fr |
| Bilingual data in Corpus | | | |
| Total Bilingual | 2.80% | 1.62% | 2.40% |
| Bilingual Data Composition | | | |
| Parallel | 10% | 17% | 15% |
| Code-switching | 75% | 69% | 73% |
| Miscellaneous | 15% | 14% | 12% |

Table 2: Bilingual data statistics for each language pair. The top section reports the proportion of bilingual data in the full corpus, showing that such data is generally sparse; the bottom section shows the distribution of the bilingual data types.



|  |  |  |
| --- | --- | --- |
| Type | Representative Sources | % |
| Parallel | Academic (thesis.fr) | 35 |
| Dictionaries (reverso.net) | 15 |
| Travel (airbnb.com) | 15 |
|  | Canadian (umontreal.ca) | 6 |
|  | Professional (docs.microsoft) | 8 |
| Code-switching | Social (pinterest.com) | 25 |
| Forums (forumactif.com) | 10 |
| E-commerce (amazon.fr) | 8 |

Table 3: Approximate domain distribution of bilingual data based on URL analysis of the top 50 sources from the en-fr corpora. Parallel data originates mainly from academic sources and dictionaries with systematic alignments, while code-switching appears in user-generated content with organic language mixing.

### 3.2 Bilingual Data Analysis

[Table 2](#S3.T2 "Table 2 ‣ Stage 2: LLM-based Classification. ‣ 3.1 Bilingual Data Identification ‣ 3 MonoWeb Pretraining Data ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining") presents the statistics of bilingual data in our corpus, where the pattern is similar for all three language pairs: the bilingual data constitutes 2% of the entire 240B-token pretraining corpus, and it is dominated by code-switching data (>> 70%), and a similar amount of parallel and miscellaneous documents (10-20%).
We further analyze the website URLs of the parallel and code-switching documents to understand the main sources of each categories of bilingual data, reported in [Table 3](#S3.T3 "Table 3 ‣ Stage 2: LLM-based Classification. ‣ 3.1 Bilingual Data Identification ‣ 3 MonoWeb Pretraining Data ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").

Parallel data, while comprising less than 20% of bilingual data, originates from high-quality curated sources.
As [Table 3](#S3.T3 "Table 3 ‣ Stage 2: LLM-based Classification. ‣ 3.1 Bilingual Data Identification ‣ 3 MonoWeb Pretraining Data ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining") reveals, academic repositories dominate, particularly doctoral theses with multilingual abstracts. Bilingual dictionaries and language learning platforms (reverso.net) provide sentence-aligned translations, while technical documentation (docs.microsoft.com) contributes systematic correspondences. These sources feature explicit token-level alignments where each segment has an equivalent in another language.

Code-switching dominates at 72% of bilingual data, which originates primarily from social content aggregation sites (pinterest.com), E-commerce with mixed-language reviews (amazon.fr), and forums.

The remaining miscellaneous 14% consists of noise—multilingual boilerplate and web artifacts where languages accidentally co-occur without meaningful relationships.

Overall, the URL analysis reveals a fundamental distinction on the source of bilingual data under different categories: parallel data provides professionally curated alignments from dictionaries and academic repositories, while code-switching reflects spontaneous language mixing in user-generated content.

## 4 Experimental Setup

### 4.1 Pretraining Configurations

We conduct experiments on three language pairs: English-French (en-fr), English-German (en-de), and English-Spanish (en-es). For each language pair, we construct a bilingual corpus by combining 60B English tokens with 60B tokens from the target language (French, German, or Spanish), sampled from FineWeb-Edu and FineWeb2.

For each language pair, we pretrain models using four data configurations:

* •

  FineWeb: Full 120B-token corpus including all bilingual data.
* •

  MonoWeb: All bilingual documents removed, retaining only monolingual content.
* •

  MonoWeb+Parallel: MonoWeb augmented with only parallel documents.
* •

  MonoWeb+CodeSwitch: MonoWeb augmented with only code-switching documents.

We exclude miscellaneous data as it lacks cross-lingual semantic relationships. This yields 12 models in total (3 language pairs × 4 configurations), all trained from scratch to ensure a fair comparison.

### 4.2 Model Architecture and Training

We train decoder-only transformer models with 1.35B parameters using the Llama-2 tokenizer (Touvron et al., [2023](#bib.bib35 "Llama 2: open foundation and fine-tuned chat models")) (32K vocabulary). The architecture consists of 24 layers with a 2048 hidden dimensions, 16 attention heads, and a 2048 context length. All models are trained for 34K steps (1̃43B tokens) with a batch size of 2,048 using the AdamW optimizer (Loshchilov and Hutter, [2017](#bib.bib36 "Decoupled weight decay regularization")) and a 6e-4 learning rate, including 2,000 warmup steps followed by constant decay. We set weight decay to 0.1, apply gradient clipping at 1.0, and use Adam betas of 0.9 and 0.95. Training is performed with Megatron-LM (Shoeybi et al., [2019](#bib.bib49 "Megatron-LM: training multi-billion parameter language models using model parallelism")) and takes about 6,144 A100 GPU hours per model.

### 4.3 Downstream Evaluation Suite

All tasks are evaluated using the lm-evaluation-harness (Gao et al., [2024](#bib.bib50 "The language model evaluation harness")), along with five-shot prompting and default configurations.

#### Machine Translation

We evaluate translation quality on standard benchmarks for all three language pairs: wmt16 en-de (Bojar et al., [2016](#bib.bib13 "Findings of the 2016 conference on machine translation")), wmt14 en-fr (Bojar et al., [2014](#bib.bib12 "Findings of the 2014 workshop on statistical machine translation")), and flores-101 en-es (Goyal et al., [2021](#bib.bib14 "The flores-101 evaluation benchmark for low-resource and multilingual machine translation")), testing both translation directions and report BLEU scores (Papineni et al., [2002](#bib.bib39 "Bleu: a method for automatic evaluation of machine translation")) separately for each direction.

#### Cross-lingual Question Answering

We evaluate cross-lingual question answering using two complementary benchmarks. (1) For XQuAD (Artetxe et al., [2019](#bib.bib16 "On the cross-lingual transferability of monolingual representations")), we adapt the dataset by placing the context in language L1 and both the question and answer in language L2, allowing us to assess the model’s ability to generate answers across languages. (2) For MLQA (Lewis et al., [2019](#bib.bib15 "MLQA: evaluating cross-lingual extractive question answering")), we follow the original setup, where the context and answer are in language L1 and the question is in language L2, which primarily evaluates the model’s ability to retrieve information across languages. We report Exact Match scores for all language pairs.

#### Cross-lingual Understanding and Reasoning

We evaluate models on a suite of benchmarks covering both cross-lingual understanding and reasoning abilities. For cross-lingual natural language understanding, we use XNLI (Conneau et al., [2018](#bib.bib19 "XNLI: evaluating cross-lingual sentence representations")) and PAWS-X (Yang et al., [2019](#bib.bib20 "PAWS-X: a cross-lingual adversarial dataset for paraphrase identification")) to assess whether bilingual data improves the transfer of inference and paraphrase recognition skills. For reasoning tasks, we include HellaSwag (Zellers et al., [2019](#bib.bib18 "HellaSwag: can a machine really finish your sentence?"); Dac Lai et al., [2023](#bib.bib17 "Okapi: instruction-tuned large language models in multiple languages with reinforcement learning from human feedback")) for commonsense reasoning, ARC (Dac Lai et al., [2023](#bib.bib17 "Okapi: instruction-tuned large language models in multiple languages with reinforcement learning from human feedback"); Clark et al., [2018](#bib.bib21 "Think you have solved question answering? Try ARC, the AI2 reasoning challenge")) for knowledge-intensive reasoning, TruthfulQA (Lin et al., [2022](#bib.bib22 "TruthfulQA: measuring how models mimic human falsehoods"); Dac Lai et al., [2023](#bib.bib17 "Okapi: instruction-tuned large language models in multiple languages with reinforcement learning from human feedback")) for factual consistency, and additionally XStoryCloze (Lin et al., [2021](#bib.bib37 "Few-shot learning with multilingual language models")) (en, es) and XWinograd (Tikhonov and Ryabinin, [2021](#bib.bib38 "It’s all in the heads: using attention heads as a baseline for cross-lingual transfer in commonsense reasoning")) (en, fr) for narrative comprehension and coreference resolution. We report accuracy for all tasks.

|  |  |  |
| --- | --- | --- |
| Source (en) | FineWeb (de) | MonoWeb (de) |
| The students should receive a grant immediately. | Die Schüler sollten sofort einen Zuschuss erhalten. | Die Studierenden sollten eine Unterstützung erhalten. |
| This was a conscious decision - diversity is an important topic here. | Dies war eine bewusste Entscheidung - Vielfalt ist ein wichtiges Thema hier. | Das war ein bewusster Entschluss. |
| He’s a hero to his kids and his wife. | Er ist ein Held für seine Kinder und seine Frau. | Er ist ein Held für seine Familie und seine Frau. |

Table 4: Fine-grained information loss in MonoWeb translations. Core propositions are preserved, but precise details are systematically lost: temporal specifications (example 1: "immediately"), explanatory contexts (example 2: diversity rationale), and lexical precision (example 3: "kids" → "Familie" [family]). Bold text shows precise translations; underlined text indicates lost information.



|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Direction | FWB | MWB | MWB+P | MWB+CS |
| en→de | 16.2 | 5.0 | 17.0 | 4.6 |
| de→en | 24.6 | 14.5 | 21.3 | 14.9 |
| en→es | 17.7 | 6.6 | 17.3 | 11.4 |
| es→en | 21.4 | 8.3 | 20.1 | 16.0 |
| en→fr | 25.4 | 12.1 | 22.7 | 17.4 |
| fr→en | 28.6 | 15.3 | 28.8 | 18.7 |
| Average | 22.3 | 9.8 | 20.2 | 12.4 |

Table 5: BLEU scores for each translation direction. Removing bilingual data (MWB) causes a substantial drop, while adding parallel data (MWB+P) largely restores performance. FWB = FineWeb, CS = Code-Switch.

## 5 Results

Table [5](#S4.T5 "Table 5 ‣ Cross-lingual Understanding and Reasoning ‣ 4.3 Downstream Evaluation Suite ‣ 4 Experimental Setup ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining"), [7](#S5.T7 "Table 7 ‣ 5.2 Understanding Translation Collapse: A Two-Fold Failure ‣ 5 Results ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining"), and [9](#S5.T9 "Table 9 ‣ 5.4 Explaining the Asymmetry: Why MT Collapses but Reasoning Persists ‣ 5 Results ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining") present results across all tasks and configurations. A clear task-specific asymmetry emerges: removing bilingual data causes significant degradation for machine translation (56% BLEU drop), moderate decline in cross-lingual QA (<10%), and has almost no effect on understanding and reasoning tasks. This indicates different levels of reliance on bilingual exposure across tasks, suggesting that different cross-lingual abilities may rely on qualitatively different learning signals.

### 5.1 Machine Translation: Critical Dependence on Parallel Data

Table [5](#S4.T5 "Table 5 ‣ Cross-lingual Understanding and Reasoning ‣ 4.3 Downstream Evaluation Suite ‣ 4 Experimental Setup ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining") summarizes translation results across all configurations and language pairs. Removing all bilingual data leads to substantial performance degradation, with average BLEU dropping from 22.3 to 9.8 (56% relative decline). Reintroducing only parallel documents which comprise 10–17% of bilingual content, largely recovers performance (20.2 BLEU, 91% of the original performance). In contrast, adding back code-switched text—72% of bilingual data—yields a minimal improvement (12.4 BLEU, only 56% of original performance).

This pattern is consistent across all six translation directions (Table [5](#S4.T5 "Table 5 ‣ Cross-lingual Understanding and Reasoning ‣ 4.3 Downstream Evaluation Suite ‣ 4 Experimental Setup ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining")). Individual language pairs show a 41–69% relative degradation when bilingual data is removed, and 90–107% recovery when parallel data alone is reintroduced.

These results highlight that translation quality depends critically on explicit cross-lingual alignment rather than incidental code-switching.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | FWB | MWB+P | MWB | MWB+CS |
| German % | 86.6 | 89.7 | 43.6 | 45.2 |
| DE BLEU | 17.4 | 17.8 | 7.70 | 6.21 |

Table 6: Language generation rate and translation quality on En→De. MWB and MWB+CS fail on both: low German generation (45% vs. 85%) and poor quality when generating German (18.5 vs. 25.1 BLEU).

### 5.2 Understanding Translation Collapse: A Two-Fold Failure

To understand the mechanisms behind translation performance degradation, we analyze 1,000 sampled En→De translation outputs, using Llama-3.3-70B-Instruct as a language identifier to classify each as German, English, or mixed-language. Table [6](#S5.T6 "Table 6 ‣ 5.1 Machine Translation: Critical Dependence on Parallel Data ‣ 5 Results ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining") shows a clear disparity: FineWeb and MonoWeb+Parallel generate German in more than 85% of cases, while MonoWeb and MonoWeb+CodeSwitch produce German in only around 45%. The remaining 55% are predominantly English passthroughs, which naturally yield zero BLEU contribution. These results indicate that models trained without parallel data often fail at the most basic requirement of translation—producing text in the target language.

However, language generation failure alone cannot account for the full extent of BLEU degradation. As shown in Table [6](#S5.T6 "Table 6 ‣ 5.1 Machine Translation: Critical Dependence on Parallel Data ‣ 5 Results ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining"), when the evaluation is restricted to outputs that are correctly generated in German, MonoWeb still achieves only 7.70 BLEU which is less than half of FineWeb’s 17.4. Even under comparable output-language conditions, a 56% quality gap remains. MonoWeb+CodeSwitch performs even worse at 6.21 BLEU. This reveals two compounding failure modes: (1) 56% failure to generate target language (43.6% vs. 86.6% German generation), and (2) severely degraded translation quality for the remaining outputs (7.70 vs. 17.4 BLEU). The overall performance decrease compounds both problems.

To further examine the nature of the degraded translation fidelity, we manually analyzed 100 correctly generated German outputs. The analysis reveals a consistent pattern of semantic under-specification: MonoWeb captures only coarse-grained semantics, preserving the basic propositional structure (who does what) but loses fine-grained information about how, when, why, and to what degree.
As illustrated in Table [4](#S4.T4 "Table 4 ‣ Cross-lingual Understanding and Reasoning ‣ 4.3 Downstream Evaluation Suite ‣ 4 Experimental Setup ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining"), FineWeb accurately preserves temporal and explanatory details ("immediately" → "sofort"; "diversity is an important topic here" → "Vielfalt ist hier ein wichtiges Thema"), whereas MonoWeb tends to produce paraphrases that erase such distinctions. Lexical precision also deteriorates, e.g., translating "kids" as "Familie" [family] instead of "Kinder" [kids].

These observations suggest that without parallel supervision, models internalize only approximate cross-lingual alignment, resulting in content-preserving yet information-thinned translations.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Task | FWB | MWB | MWB+P | MWB+CS |
| German | | | | |
| XQuAD | 28.9 | 25.2 | 31.2 | 29.0 |
| MLQA | 20.6 | 22.4 | 21.4 | 19.1 |
| Spanish | | | | |
| XQuAD | 31.8 | 29.7 | 32.1 | 29.9 |
| MLQA | 22.7 | 23.9 | 22.8 | 20.3 |
| XQuAD Avg | 30.4 | 27.5 | 31.7 | 29.5 |
| MLQA Avg | 21.7 | 23.2 | 22.1 | 19.7 |

Table 7: Cross-lingual QA performance averaged over both directions per language pair. XQuAD shows moderate sensitivity to bilingual data, while MLQA remains stable.

### 5.3 Other Cross-lingual Tasks: Minimal Dependence on Bilingual Data

#### Cross-lingual Question Answering

Table [7](#S5.T7 "Table 7 ‣ 5.2 Understanding Translation Collapse: A Two-Fold Failure ‣ 5 Results ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining") presents results for XQuAD and MLQA. The two tasks show different sensitivity to bilingual data removal.
For XQuAD, MonoWeb underperforms FineWeb throughout training (Figure [2](#S5.F2 "Figure 2 ‣ Cross-lingual Question Answering ‣ 5.3 Other Cross-lingual Tasks: Minimal Dependence on Bilingual Data ‣ 5 Results ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining")), achieving 27.5 EM compared to FineWeb’s 30.4 EM (9.5% drop). On MLQA, the training curves (Figure [2](#S5.F2 "Figure 2 ‣ Cross-lingual Question Answering ‣ 5.3 Other Cross-lingual Tasks: Minimal Dependence on Bilingual Data ‣ 5 Results ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining")) show overlapping trajectories across configurations, with the final scores ranging from 21.7 to 23.2 EM. Unlike XQuAD, MLQA exhibits no consistent separation between configurations during training.
This difference may reflect distinct task structures: XQuAD requires generating L2 answers from L1 contexts, while MLQA primarily involves retrieving answers within L1 after understanding L2 questions.

(a)

(b)

(c)

(d)

Figure 2: Training performance across cross-lingual tasks. (a) XQuAD shows consistent separation between configurations, with MonoWeb underperforming throughout training. (b) MLQA exhibits overlapping trajectories across all configurations. (c) HellaSwag performance across language pairs under identical FineWeb setup shows variation, indicating cross-lingual transfer varies by pair. (d) HellaSwag within en-fr: stable performance across bilingual configurations.

#### Understanding and Reasoning Tasks

Table [9](#S5.T9 "Table 9 ‣ 5.4 Explaining the Asymmetry: Why MT Collapses but Reasoning Persists ‣ 5 Results ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining") presents results across five understanding and reasoning benchmarks. All tasks show stability across all bilingual configurations, with performance consistently being within 1-2% of the baseline.

To better understand this stability, we take HellaSwag as a representative case. Figure [2](#S5.F2 "Figure 2 ‣ Cross-lingual Question Answering ‣ 5.3 Other Cross-lingual Tasks: Minimal Dependence on Bilingual Data ‣ 5 Results ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining"), and  [2](#S5.F2 "Figure 2 ‣ Cross-lingual Question Answering ‣ 5.3 Other Cross-lingual Tasks: Minimal Dependence on Bilingual Data ‣ 5 Results ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining") demonstrates two complementary findings: First, Figure [2](#S5.F2 "Figure 2 ‣ Cross-lingual Question Answering ‣ 5.3 Other Cross-lingual Tasks: Minimal Dependence on Bilingual Data ‣ 5 Results ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining") compares different language pairs under the FineWeb setting where all three pairs use identical English data for 1:1 balanced training, and shows discernible variation in HellaSwag\_En performance, indicating that cross-lingual transfer effects exist and vary across language pairs.
Second, Figure [2](#S5.F2 "Figure 2 ‣ Cross-lingual Question Answering ‣ 5.3 Other Cross-lingual Tasks: Minimal Dependence on Bilingual Data ‣ 5 Results ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining") compares different bilingual configurations for the EN–FR pair. Performance remains largely unchanged whether bilingual data is present (FineWeb), absent (MonoWeb), or partially restored (MWB+P, MWB+CS), demonstrating that cross-lingual transfer persists even without bilingual data. Similar patterns also emerge across other benchmarks.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Sentence-level P@1 | | | Lexical-level P@1 | | |
| Layer | FWB | MWB | Δ\Delta | FWB | MWB | Δ\Delta |
| 0 | 1.7 | 1.8 | +0.2 | 5.8 | 8.3 | +2.5 |
| 6 | 60.6 | 55.9 | -4.7 | 40.7 | 19.7 | -21.0 |
| 12 | 93.7 | 92.5 | -1.3 | 68.7 | 55.1 | -13.6 |
| 23 | 81.2 | 79.4 | -1.8 | 25.5 | 18.4 | -7.1 |

Table 8: Layer-wise Alignment Analysis. Lexical-level alignment shows a sharp drop at middle layers in MonoWeb, while sentence-level alignment remains largely stable.

### 5.4 Explaining the Asymmetry: Why MT Collapses but Reasoning Persists

Removing bilingual data causes a severe collapse in machine translation, while cross-lingual reasoning and understanding tasks remain largely unaffected. To explain this phenomenon, we analyze how bilingual data removal impacts cross-lingual alignment at different linguistic granularities.
Specifically, we measure alignment across layers using Precision@1 (P@1) computed with cosine similarity for sentence representations (3,000 WMT parallel sentences) and word representations (2,000 MUSE Conneau et al. ([2017](#bib.bib2 "Word translation without parallel data")) pairs).
As shown in [Table 8](#S5.T8 "Table 8 ‣ Understanding and Reasoning Tasks ‣ 5.3 Other Cross-lingual Tasks: Minimal Dependence on Bilingual Data ‣ 5 Results ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining"), we observe a stark divergence: while MonoWeb preserves robust sentence-level alignment (<2%<2\% drop from FineWeb), it suffers a sharp 13–21% degradation in lexical-level alignment.
This suggests that monolingual pretraining is sufficient to align sentence-level semantics, supporting cross-lingual understanding and reasoning, but fails to establish the fine-grained lexical correspondences required for accurate translation.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Task | FWB | MWB | MWB+P | MWB+CS |
| English (Avg) | | | | |
| XNLI | 46.3 | 45.6 | 45.9 | 46.8 |
| HellaSwag | 39.1 | 39.4 | 39.6 | 39.6 |
| ARC\_C | 32.3 | 33.6 | 34.7 | 33.8 |
| ARC\_E | 68.5 | 68.3 | 68.3 | 67.9 |
| PAWS | 54.5 | 54.9 | 54.0 | 55.5 |
| TruthfulQA | 22.0 | 21.8 | 22.4 | 20.9 |
| Xwinograd | 75.7 | 75.4 | 74.0 | 73.6 |
| Xstorycloze | 64.6 | 65.2 | 64.1 | 65.6 |
| German | | | | |
| XNLI | 44.5 | 43.4 | 43.8 | 41.4 |
| HellaSwag | 34.8 | 35.0 | 35.5 | 35.2 |
| ARC | 22.9 | 24.1 | 24.9 | 25.2 |
| PAWS | 51.9 | 52.0 | 51.6 | 51.8 |
| TruthfulQA | 23.4 | 21.4 | 21.3 | 24.1 |
| Spanish | | | | |
| XNLI | 43.5 | 42.3 | 43.9 | 45.4 |
| HellaSwag | 38.6 | 38.6 | 38.5 | 39.2 |
| ARC | 28.6 | 29.7 | 27.9 | 28.5 |
| PAWS | 50.1 | 53.1 | 51.3 | 51.8 |
| TruthfulQA | 25.6 | 26.7 | 25.2 | 26.7 |
| Xstorycloze | 62.3 | 61.6 | 61.6 | 61.4 |
| French | | | | |
| XNLI | 44.6 | 44.0 | 44.0 | 44.5 |
| HellaSwag | 38.0 | 38.5 | 38.4 | 37.9 |
| ARC | 29.1 | 26.4 | 26.5 | 26.9 |
| PAWS | 52.6 | 47.9 | 52.2 | 53.8 |
| TruthfulQA | 24.4 | 25.8 | 22.9 | 25.4 |
| Xwinograd | 61.5 | 66.3 | 61.5 | 60.2 |

Table 9: 
Multilingual understanding and reasoning performance across all language pairs.
For English, the reported numbers are averaged over several language pairs.

## 6 Conclusion

This study explored the role of bilingual data in multilingual LLM pretraining and uncovered a clear task asymmetry. Translation is highly sensitive to a small fraction of bilingual content (2%), whereas other cross-lingual tasks remain largely unaffected. Further analysis show that parallel data, not code-switching text, drives translation performance. This indicates that explicit cross-lingual alignment is essential for translation, while monolingual exposure largely suffices for broader cross-lingual understanding. These findings imply that multilingual pretraining may benefit more from high-quality parallel data than from large quantities of code-switched text. More broadly, our results highlight that the impact of bilingual data during multilingual pretraining can vary substantially across tasks, suggesting that its role is nuanced even within the pretraining stage.

## 7 Limitations

Our study has several limitations. First, due to computational constraints, we pretrained only 1.35B-parameter models and did not pretrain larger models such as 7B, which may exhibit different sensitivity to bilingual data. Second, our experiments focus on major languages within the Latin script family, leaving open questions about the impact of bilingual data on typologically distant or low-resource languages. Third, our analysis categorizes bilingual data into parallel, code-switching, and miscellaneous types, but finer-grained distinctions, such as domain, register, or sentence-level alignment quality, may further influence cross-lingual learning.

## References

* O. J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, R. Avila, I. Babuschkin, S. Balaji, V. Balcom, P. Baltescu, H. Bao, M. Bavarian, J. Belgum, I. Bello, J. Berdine, G. Bernadett-Shapiro, C. Berner, L. Bogdonoff, O. Boiko, M. Boyd, A. Brakman, G. Brockman, T. Brooks, M. Brundage, K. Button, T. Cai, R. Campbell, A. Cann, B. Carey, C. Carlson, R. Carmichael, B. Chan, C. Chang, F. Chantzis, D. Chen, S. Chen, R. Chen, J. Chen, M. Chen, B. Chess, C. Cho, C. Chu, H. W. Chung, D. Cummings, J. Currier, Y. Dai, C. Decareaux, T. Degry, N. Deutsch, D. Deville, A. Dhar, D. Dohan, S. Dowling, S. Dunning, A. Ecoffet, A. Eleti, T. Eloundou, D. Farhi, L. Fedus, N. Felix, S. P. Fishman, J. Forte, I. Fulford, L. Gao, E. Georges, C. Gibson, V. Goel, T. Gogineni, G. Goh, R. Gontijo-Lopes, J. Gordon, M. Grafstein, S. Gray, R. Greene, J. Gross, S. S. Gu, Y. Guo, C. Hallacy, J. Han, J. Harris, Y. He, M. Heaton, J. Heidecke, C. Hesse, A. Hickey, W. Hickey, P. Hoeschele, B. Houghton, K. Hsu, S. Hu, X. Hu, J. Huizinga, S. Jain, S. Jain, J. Jang, A. Jiang, R. Jiang, H. Jin, D. Jin, S. Jomoto, B. Jonn, H. Jun, T. Kaftan, L. Kaiser, A. Kamali, I. Kanitscheider, N. S. Keskar, T. Khan, L. Kilpatrick, J. W. Kim, C. Kim, Y. Kim, H. Kirchner, J. R. Kiros, M. Knight, D. Kokotajlo, L. Kondraciuk, A. Kondrich, A. Konstantinidis, K. Kosic, G. Krueger, V. Kuo, M. Lampe, I. Lan, T. Lee, J. Leike, J. Leung, D. Levy, C. Li, R. Lim, M. Lin, S. Lin, M. Litwin, T. Lopez, R. Lowe, P. Lue, A. Makanju, K. Malfacini, S. Manning, T. Markov, Y. Markovski, B. Martin, K. Mayer, A. Mayne, B. McGrew, S. M. McKinney, C. McLeavey, P. McMillan, J. McNeil, D. Medina, A. Mehta, J. Menick, L. Metz, A. Mishchenko, P. Mishkin, V. Monaco, E. Morikawa, D. P. Mossing, T. Mu, M. Murati, O. Murk, D. M’ely, A. Nair, R. Nakano, R. Nayak, A. Neelakantan, R. Ngo, H. Noh, O. Long, C. O’Keefe, J. W. Pachocki, A. Paino, J. Palermo, A. Pantuliano, G. Parascandolo, J. Parish, E. Parparita, A. Passos, M. Pavlov, A. Peng, A. Perelman, F. de Avila Belbute Peres, M. Petrov, H. P. de Oliveira Pinto, M. Pokorny, M. Pokrass, V. H. Pong, T. Powell, A. Power, B. Power, E. Proehl, R. Puri, A. Radford, J. W. Rae, A. Ramesh, C. Raymond, F. Real, K. Rimbach, C. Ross, B. Rotsted, H. Roussez, N. Ryder, M. D. Saltarelli, T. Sanders, S. Santurkar, G. Sastry, H. Schmidt, D. Schnurr, J. Schulman, D. Selsam, K. Sheppard, T. Sherbakov, J. Shieh, S. Shoker, P. Shyam, S. Sidor, E. Sigler, M. Simens, J. Sitkin, K. Slama, I. Sohl, B. Sokolowsky, Y. Song, N. Staudacher, F. P. Such, N. Summers, I. Sutskever, J. Tang, N. A. Tezak, M. Thompson, P. Tillet, A. Tootoonchian, E. Tseng, P. Tuggle, N. Turley, J. Tworek, J. F. C. Uribe, A. Vallone, A. Vijayvergiya, C. Voss, C. L. Wainwright, J. J. Wang, A. Wang, B. Wang, J. Ward, J. Wei, C. Weinmann, A. Welihinda, P. Welinder, J. Weng, L. Weng, M. Wiethoff, D. Willner, C. Winter, S. Wolrich, H. Wong, L. Workman, S. Wu, J. Wu, M. Wu, K. Xiao, T. Xu, S. Yoo, K. Yu, Q. Yuan, W. Zaremba, R. Zellers, C. Zhang, M. Zhang, S. Zhao, T. Zheng, J. Zhuang, W. Zhuk, and B. Zoph (2023)
  GPT-4 technical report.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:257532815)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* M. Artetxe, S. Ruder, and D. Yogatama (2019)
  On the cross-lingual transferability of monolingual representations.
  CoRR abs/1910.11856.
  External Links: 1910.11856
  Cited by: [§4.3](#S4.SS3.SSS0.Px2.p1.1 "Cross-lingual Question Answering ‣ 4.3 Downstream Evaluation Suite ‣ 4 Experimental Setup ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* S. Bird, E. Klein, and E. Loper (2009)
  Natural language processing with python: analyzing text with the natural language toolkit.
   O’Reilly Media, Inc..
  Cited by: [§3.1](#S3.SS1.SSS0.Px1.p1.4 "Stage 1: Candidate Detection via Entropy-based Filtering. ‣ 3.1 Bilingual Data Identification ‣ 3 MonoWeb Pretraining Data ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* O. Bojar, C. Buck, C. Federmann, B. Haddow, P. Koehn, J. Leveling, C. Monz, P. Pecina, M. Post, H. Saint-Amand, R. Soricut, L. Specia, and A. Tamchyna (2014)
  Findings of the 2014 workshop on statistical machine translation.
  In Proceedings of the Ninth Workshop on Statistical Machine Translation,
  Baltimore, Maryland, USA,  pp. 12–58.
  External Links: [Link](http://www.aclweb.org/anthology/W/W14/W14-3302)
  Cited by: [§4.3](#S4.SS3.SSS0.Px1.p1.1 "Machine Translation ‣ 4.3 Downstream Evaluation Suite ‣ 4 Experimental Setup ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* O. Bojar, R. Chatterjee, C. Federmann, Y. Graham, B. Haddow, M. Huck, A. Jimeno Yepes, P. Koehn, V. Logacheva, C. Monz, M. Negri, A. Neveol, M. Neves, M. Popel, M. Post, R. Rubino, C. Scarton, L. Specia, M. Turchi, K. Verspoor, and M. Zampieri (2016)
  Findings of the 2016 conference on machine translation.
  In Proceedings of the First Conference on Machine Translation,
  Berlin, Germany,  pp. 131–198.
  External Links: [Link](http://www.aclweb.org/anthology/W/W16/W16-2301)
  Cited by: [§4.3](#S4.SS3.SSS0.Px1.p1.1 "Machine Translation ‣ 4.3 Downstream Evaluation Suite ‣ 4 Experimental Setup ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* E. Briakou, C. Cherry, and G. F. Foster (2023)
  Searching for needles in a haystack: on the role of incidental bilingualism in palm’s translation capability.
  In Annual Meeting of the Association for Computational Linguistics,
  External Links: [Link](https://api.semanticscholar.org/CorpusID:258740723)
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining"),
  [§2](#S2.p2.1 "2 Related Work ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* P. F. Brown, S. D. Pietra, V. J. D. Pietra, and R. L. Mercer (1993)
  The mathematics of statistical machine translation: parameter estimation.
  Comput. Linguistics 19,  pp. 263–311.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:13259913)
  Cited by: [§2](#S2.p1.1 "2 Related Work ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* A. Chaudhary, K. Raman, K. Srinivasan, and J. Chen (2020)
  DICT-mlm: improved multilingual pre-training using bilingual dictionaries.
  ArXiv abs/2010.12566.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:225062397)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* Z. Chi, L. Dong, F. Wei, N. Yang, S. Singhal, W. Wang, X. Song, X. Mao, H. Huang, and M. Zhou (2020)
  InfoXLM: an information-theoretic framework for cross-lingual language model pre-training.
  In North American Chapter of the Association for Computational Linguistics,
  External Links: [Link](https://api.semanticscholar.org/CorpusID:220525491)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord (2018)
  Think you have solved question answering? Try ARC, the AI2 reasoning challenge.
  ArXiv abs/1803.05457.
  Cited by: [§4.3](#S4.SS3.SSS0.Px3.p1.1 "Cross-lingual Understanding and Reasoning ‣ 4.3 Downstream Evaluation Suite ‣ 4 Experimental Setup ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* A. Conneau, G. Lample, M. Ranzato, L. Denoyer, and H. Jégou (2017)
  Word translation without parallel data.
  arXiv preprint arXiv:1710.04087.
  Cited by: [§5.4](#S5.SS4.p1.1 "5.4 Explaining the Asymmetry: Why MT Collapses but Reasoning Persists ‣ 5 Results ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* A. Conneau, G. Lample, R. Rinott, A. Williams, S. R. Bowman, H. Schwenk, and V. Stoyanov (2018)
  XNLI: evaluating cross-lingual sentence representations.
  In Conference on Empirical Methods in Natural Language Processing,
  External Links: [Link](https://api.semanticscholar.org/CorpusID:52271711)
  Cited by: [§4.3](#S4.SS3.SSS0.Px3.p1.1 "Cross-lingual Understanding and Reasoning ‣ 4.3 Downstream Evaluation Suite ‣ 4 Experimental Setup ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* V. Dac Lai, C. Van Nguyen, N. T. Ngo, T. Nguyen, F. Dernoncourt, R. A. Rossi, and T. H. Nguyen (2023)
  Okapi: instruction-tuned large language models in multiple languages with reinforcement learning from human feedback.
  arXiv e-prints,  pp. arXiv–2307.
  Cited by: [§4.3](#S4.SS3.SSS0.Px3.p1.1 "Cross-lingual Understanding and Reasoning ‣ 4.3 Downstream Evaluation Suite ‣ 4 Experimental Setup ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* J. Devlin, M. Chang, K. Lee, and K. Toutanova (2019)
  BERT: pre-training of deep bidirectional transformers for language understanding.
  In North American Chapter of the Association for Computational Linguistics,
  External Links: [Link](https://api.semanticscholar.org/CorpusID:52967399)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Yang, A. Fan, A. Goyal, A. S. Hartshorn, A. Yang, A. Mitra, A. Sravankumar, A. Korenev, A. Hinsvark, A. Rao, A. Zhang, A. Rodriguez, A. Gregerson, A. Spataru, B. Rozière, B. Biron, B. Tang, B. Chern, C. Caucheteux, C. Nayak, C. Bi, C. Marra, C. McConnell, C. Keller, C. Touret, C. Wu, C. Wong, C. C. Ferrer, C. Nikolaidis, D. Allonsius, D. Song, D. Pintz, D. Livshits, D. Esiobu, D. Choudhary, D. Mahajan, D. Garcia-Olano, D. Perino, D. Hupkes, E. Lakomkin, E. A. AlBadawy, E. Lobanova, E. Dinan, E. M. Smith, F. Radenovic, F. Zhang, G. Synnaeve, G. Lee, G. L. Anderson, G. Nail, G. Mialon, G. Pang, G. Cucurell, H. Nguyen, H. Korevaar, H. Xu, H. Touvron, I. Zarov, I. A. Ibarra, I. M. Kloumann, I. Misra, I. Evtimov, J. Copet, J. Lee, J. Geffert, J. Vranes, J. Park, J. Mahadeokar, J. Shah, J. van der Linde, J. Billock, J. Hong, J. Lee, J. Fu, J. Chi, J. Huang, J. Liu, J. Wang, J. Yu, J. Bitton, J. Spisak, J. Park, J. Rocca, J. Johnstun, J. Saxe, J. Jia, K. V. Alwala, K. Upasani, K. Plawiak, K. Li, K. neth Heafield, K. R. Stone, K. El-Arini, K. Iyer, K. Malik, K. Chiu, K. Bhalla, L. Rantala-Yeary, L. van der Maaten, L. Chen, L. Tan, L. Jenkins, L. Martin, L. Madaan, L. Malo, L. Blecher, L. Landzaat, L. de Oliveira, M. Muzzi, M. Pasupuleti, M. Singh, M. Paluri, M. Kardas, M. Oldham, M. Rita, M. Pavlova, M. H. M. Kambadur, M. Lewis, M. Si, M. K. Singh, M. Hassan, N. Goyal, N. Torabi, N. Bashlykov, N. Bogoychev, N. S. Chatterji, O. Duchenne, O. cCelebi, P. Alrassy, P. Zhang, P. Li, P. Vasić, P. Weng, P. Bhargava, P. Dubal, P. Krishnan, P. S. Koura, P. Xu, Q. He, Q. Dong, R. Srinivasan, R. Ganapathy, R. Calderer, R. S. Cabral, R. Stojnic, R. Raileanu, R. Girdhar, R. Patel, R. Sauvestre, R. Polidoro, R. Sumbaly, R. Taylor, R. Silva, R. Hou, R. Wang, S. Hosseini, S. Chennabasappa, S. Singh, S. Bell, S. S. Kim, S. Edunov, S. Nie, S. Narang, S. C. Raparthy, S. Shen, S. Wan, S. Bhosale, S. Zhang, S. Vandenhende, S. Batra, S. Whitman, S. Sootla, S. Collot, S. Gururangan, S. Borodinsky, T. Herman, T. Fowler, T. Sheasha, T. Georgiou, T. Scialom, T. Speckbacher, T. Mihaylov, T. Xiao, U. Karn, V. Goswami, V. Gupta, V. Ramanathan, V. Kerkez, V. Gonguet, V. Do, V. Vogeti, V. Petrovic, W. Chu, W. Xiong, W. Fu, W. Meers, X. Martinet, X. Wang, X. E. Tan, X. Xie, X. Jia, X. Wang, Y. Goldschlag, Y. Gaur, Y. Babaei, Y. Wen, Y. Song, Y. Zhang, Y. Li, Y. Mao, Z. D. Coudert, Z. Yan, Z. Chen, Z. Papakipos, A. K. Singh, A. Grattafiori, A. Jain, A. Kelsey, A. Shajnfeld, A. Gangidi, A. Victoria, A. Goldstand, A. Menon, A. Sharma, A. Boesenberg, A. Vaughan, A. Baevski, A. Feinstein, A. Kallet, A. Sangani, A. Yunus, A. Lupu, A. Alvarado, A. Caples, A. Gu, A. Ho, A. Poulton, A. Ryan, A. Ramchandani, A. Franco, A. Saraf, A. Chowdhury, A. Gabriel, A. Bharambe, A. Eisenman, A. Yazdan, B. James, B. Maurer, B. Leonhardi, P. (. Huang, B. Loyd, B. de Paola, B. Paranjape, B. Liu, B. Wu, B. Ni, B. Hancock, B. Wasti, B. Spence, B. Stojkovic, B. Gamido, B. Montalvo, C. Parker, C. Burton, C. Mejia, C. Wang, C. Kim, C. Zhou, C. Hu, C. Chu, C. Cai, C. Tindal, C. Feichtenhofer, D. Civin, D. Beaty, D. Kreymer, S. Li, D. Wyatt, D. Adkins, D. Xu, D. Testuggine, D. David, D. Parikh, D. Liskovich, D. Foss, D. Wang, D. Le, D. Holland, E. Dowling, E. Jamil, E. Montgomery, E. Presani, E. Hahn, E. Wood, E. Brinkman, E. Arcaute, E. Dunbar, E. Smothers, F. Sun, F. Kreuk, F. Tian, F. Ozgenel, F. Caggioni, F. Guzm’an, F. J. Kanayet, F. Seide, G. M. Florez, G. Schwarz, G. Badeer, G. Swee, G. Halpern, G. Thattai, G. Herman, G. G. Sizov, G. Zhang, G. Lakshminarayanan, H. Shojanazeri, H. Zou, H. Wang, H. Zha, H. Habeeb, H. Rudolph, H. Suk, H. Aspegren, H. Goldman, I. Molybog, I. Tufanov, I. Veliche, I. Gat, J. Weissman, J. Geboski, J. Kohli, J. Asher, J. Gaya, J. Marcus, J. Tang, J. Chan, J. Zhen, J. Reizenstein, J. Teboul, J. Zhong, J. Jin, J. Yang, J. Cummings, J. Carvill, J. Shepard, J. McPhie, J. Torres, J. Ginsburg, J. Wang, K. Wu, U. KamHou, K. Saxena, K. Prasad, K. Khandelwal, K. Zand, K. Matosich, K. Veeraraghavan, K. Michelena, K. Li, K. Huang, K. Chawla, K. Lakhotia, K. Huang, L. Chen, L. Garg, A. Lavender, L. Silva, L. Bell, L. Zhang, L. Guo, L. Yu, L. Moshkovich, L. Wehrstedt, M. Khabsa, M. Avalani, M. Bhatt, M. Tsimpoukelli, M. Mankus, M. Hasson, M. Lennie, M. Reso, M. Groshev, M. Naumov, M. Lathi, M. Keneally, M. L. Seltzer, M. Valko, M. Restrepo, M. Patel, M. Vyatskov, M. Samvelyan, M. Clark, M. Macey, M. Wang, M. J. Hermoso, M. Metanat, M. Rastegari, M. Bansal, N. Santhanam, N. Parks, N. White, N. Bawa, N. Singhal, N. Egebo, N. Usunier, N. P. Laptev, N. Dong, N. Zhang, N. Cheng, O. Chernoguz, O. Hart, O. Salpekar, O. Kalinli, P. Kent, P. Parekh, P. Saab, P. Balaji, P. Rittner, P. Bontrager, P. Roux, P. Dollár, P. Zvyagina, P. Ratanchandani, P. Yuvraj, Q. Liang, R. Alao, R. Rodriguez, R. Ayub, R. Murthy, R. Nayani, R. Mitra, R. Li, R. Hogan, R. Battey, R. Wang, R. Maheswari, R. Howes, R. Rinott, S. J. Bondu, S. Datta, S. Chugh, S. Hunt, S. Dhillon, S. Sidorov, S. Pan, S. Verma, S. Yamamoto, S. Ramaswamy, S. Lindsay, S. Feng, S. Lin, S. C. Zha, S. Shankar, S. Zhang, S. Wang, S. Agarwal, S. Sajuyigbe, S. Chintala, S. Max, S. Chen, S. Kehoe, S. Satterfield, S. Govindaprasad, S. K. Gupta, S. Cho, S. Virk, S. Subramanian, S. Choudhury, S. Goldman, T. Remez, T. Glaser, T. Best, T. Kohler, T. Robinson, T. Li, T. Zhang, T. Matthews, T. Chou, T. Shaked, V. Vontimitta, V. Ajayi, V. Montanez, V. Mohan, V. S. Kumar, V. Mangla, V. Ionescu, V. A. Poenaru, V. T. Mihailescu, V. Ivanov, W. Li, W. Wang, W. Jiang, W. Bouaziz, W. Constable, X. Tang, X. Wang, X. Wu, X. Wang, X. Xia, X. Wu, X. Gao, Y. Chen, Y. Hu, Y. Jia, Y. Qi, Y. Li, Y. Zhang, Y. Zhang, Y. Adi, Y. Nam, Y. Wang, Y. Hao, Y. Qian, Y. He, Z. Rait, Z. DeVito, Z. Rosnbrick, Z. Wen, Z. Yang, and Z. Zhao (2024)
  The Llama 3 herd of models.
  ArXiv abs/2407.21783.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:271571434)
  Cited by: [§3.1](#S3.SS1.SSS0.Px2.p1.1 "Stage 2: LLM-based Classification. ‣ 3.1 Bilingual Data Identification ‣ 3 MonoWeb Pretraining Data ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* A. Fan, S. Bhosale, H. Schwenk, Z. Ma, A. El-Kishky, S. Goyal, M. Baines, O. Çelebi, G. Wenzek, V. Chaudhary, N. Goyal, T. Birch, V. Liptchinsky, S. Edunov, E. Grave, M. Auli, and A. Joulin (2020)
  Beyond english-centric multilingual machine translation.
  J. Mach. Learn. Res. 22,  pp. 107:1–107:48.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:224814118)
  Cited by: [§2](#S2.p1.1 "2 Related Work ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* L. Gao, J. Tow, B. Abbasi, S. Biderman, S. Black, A. DiPofi, C. Foster, L. Golding, J. Hsu, A. Le Noac’h, H. Li, K. McDonell, N. Muennighoff, C. Ociepa, J. Phang, L. Reynolds, H. Schoelkopf, A. Skowron, L. Sutawika, E. Tang, A. Thite, B. Wang, K. Wang, and A. Zou (2024)
  The language model evaluation harness.
   Zenodo.
  External Links: [Document](https://dx.doi.org/10.5281/zenodo.12608602),
  [Link](https://zenodo.org/records/12608602)
  Cited by: [§4.3](#S4.SS3.p1.1 "4.3 Downstream Evaluation Suite ‣ 4 Experimental Setup ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* N. Goyal, C. Gao, V. Chaudhary, P. Chen, G. Wenzek, D. Ju, S. Krishnan, M. Ranzato, F. Guzmán, and A. Fan (2021)
  The flores-101 evaluation benchmark for low-resource and multilingual machine translation.
  Transactions of the Association for Computational Linguistics 10,  pp. 522–538.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:235358129)
  Cited by: [§4.3](#S4.SS3.SSS0.Px1.p1.1 "Machine Translation ‣ 4.3 Downstream Evaluation Suite ‣ 4 Experimental Setup ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* M. Johnson, M. Schuster, Q. V. Le, M. Krikun, Y. Wu, Z. Chen, N. Thorat, F. B. Viégas, M. Wattenberg, G. S. Corrado, M. Hughes, and J. Dean (2016)
  Google’s multilingual neural machine translation system: enabling zero-shot translation.
  Transactions of the Association for Computational Linguistics 5,  pp. 339–351.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:260464809)
  Cited by: [§2](#S2.p1.1 "2 Related Work ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* A. Joulin, E. Grave, P. Bojanowski, and T. Mikolov (2016)
  Bag of tricks for efficient text classification.
  arXiv preprint arXiv:1607.01759.
  Cited by: [§3.1](#S3.SS1.SSS0.Px1.p1.4 "Stage 1: Candidate Detection via Entropy-based Filtering. ‣ 3.1 Bilingual Data Identification ‣ 3 MonoWeb Pretraining Data ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* P. Lewis, B. Oğuz, R. Rinott, S. Riedel, and H. Schwenk (2019)
  MLQA: evaluating cross-lingual extractive question answering.
  arXiv preprint arXiv:1910.07475.
  Cited by: [§4.3](#S4.SS3.SSS0.Px2.p1.1 "Cross-lingual Question Answering ‣ 4.3 Downstream Evaluation Suite ‣ 4 Experimental Setup ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* S. Lin, J. Hilton, and O. Evans (2022)
  TruthfulQA: measuring how models mimic human falsehoods.
  In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
  Dublin, Ireland,  pp. 3214–3252.
  External Links: [Link](https://aclanthology.org/2022.acl-long.229),
  [Document](https://dx.doi.org/10.18653/v1/2022.acl-long.229)
  Cited by: [§4.3](#S4.SS3.SSS0.Px3.p1.1 "Cross-lingual Understanding and Reasoning ‣ 4.3 Downstream Evaluation Suite ‣ 4 Experimental Setup ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* X. V. Lin, T. Mihaylov, M. Artetxe, T. Wang, S. Chen, D. Simig, M. Ott, N. Goyal, S. Bhosale, J. Du, R. Pasunuru, S. Shleifer, P. S. Koura, V. Chaudhary, B. O’Horo, J. Wang, L. Zettlemoyer, Z. Kozareva, M. T. Diab, V. Stoyanov, and X. Li (2021)
  Few-shot learning with multilingual language models.
  CoRR abs/2112.10668.
  External Links: [Link](https://arxiv.org/abs/2112.10668),
  2112.10668
  Cited by: [§4.3](#S4.SS3.SSS0.Px3.p1.1 "Cross-lingual Understanding and Reasoning ‣ 4.3 Downstream Evaluation Suite ‣ 4 Experimental Setup ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* I. Loshchilov and F. Hutter (2017)
  Decoupled weight decay regularization.
  In International Conference on Learning Representations,
  External Links: [Link](https://api.semanticscholar.org/CorpusID:53592270)
  Cited by: [§4.2](#S4.SS2.p1.1 "4.2 Model Architecture and Training ‣ 4 Experimental Setup ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* A. Lozhkov, L. Ben Allal, L. von Werra, and T. Wolf (2024)
  FineWeb-edu: the finest collection of educational content.
   Hugging Face.
  External Links: [Link](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
  [Document](https://dx.doi.org/10.57967/hf/2497)
  Cited by: [§3](#S3.p1.1 "3 MonoWeb Pretraining Data ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* K. Papineni, S. Roukos, T. Ward, and W. Zhu (2002)
  Bleu: a method for automatic evaluation of machine translation.
  In Annual Meeting of the Association for Computational Linguistics,
  External Links: [Link](https://api.semanticscholar.org/CorpusID:11080756)
  Cited by: [§4.3](#S4.SS3.SSS0.Px1.p1.1 "Machine Translation ‣ 4.3 Downstream Evaluation Suite ‣ 4 Experimental Setup ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* G. Penedo, H. Kydlíček, V. Sabolčec, B. Messmer, N. Foroutan, A. H. Kargaran, C. Raffel, M. Jaggi, L. V. Werra, and T. Wolf (2025)
  FineWeb2: one pipeline to scale them all – adapting pre-training data processing to every language.
  External Links: 2506.20920,
  [Link](https://arxiv.org/abs/2506.20920)
  Cited by: [§3](#S3.p1.1 "3 MonoWeb Pretraining Data ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* L. Qin, M. Ni, Y. Zhang, and W. Che (2020)
  CoSDA-ML: multi-lingual code-switching data augmentation for zero-shot cross-lingual NLP.
  ArXiv abs/2006.06402.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:219573540)
  Cited by: [§2](#S2.p1.1 "2 Related Work ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* M. R. Qorib, J. Li, and H. T. Ng (2025)
  Just go parallel: improving the multilingual capabilities of large language models.
  In Annual Meeting of the Association for Computational Linguistics,
  External Links: [Link](https://api.semanticscholar.org/CorpusID:279402494)
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining"),
  [§2](#S2.p1.1 "2 Related Work ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining"),
  [§2](#S2.p2.1 "2 Related Work ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and B. Catanzaro (2019)
  Megatron-LM: training multi-billion parameter language models using model parallelism.
  arXiv preprint arXiv:1909.08053.
  Cited by: [§4.2](#S4.SS2.p1.1 "4.2 Model Architecture and Training ‣ 4 Experimental Setup ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* N. team, M. R. Costa-jussà, J. Cross, O. cCelebi, M. Elbayad, K. Heafield, K. Heffernan, E. Kalbassi, J. Lam, D. Licht, J. Maillard, A. Sun, S. Wang, G. Wenzek, A. Youngblood, B. Akula, L. Barrault, G. M. Gonzalez, P. Hansanti, J. Hoffman, S. Jarrett, K. R. Sadagopan, D. Rowe, S. L. Spruit, C. Tran, P. Y. Andrews, N. F. Ayan, S. Bhosale, S. Edunov, A. Fan, C. Gao, V. Goswami, F. Guzm’an, P. Koehn, A. Mourachko, C. Ropers, S. Saleem, H. Schwenk, and J. Wang (2022)
  No language left behind: scaling human-centered machine translation.
  ArXiv abs/2207.04672.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:250425961)
  Cited by: [§2](#S2.p1.1 "2 Related Work ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* A. Tikhonov and M. Ryabinin (2021)
  It’s all in the heads: using attention heads as a baseline for cross-lingual transfer in commonsense reasoning.
  External Links: 2106.12066
  Cited by: [§4.3](#S4.SS3.SSS0.Px3.p1.1 "Cross-lingual Understanding and Reasoning ‣ 4.3 Downstream Evaluation Suite ‣ 4 Experimental Setup ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* H. Touvron, L. Martin, K. R. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. M. Bikel, L. Blecher, C. C. Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. S. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. M. Kloumann, A. Korenev, P. S. Koura, M. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. H. M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom (2023)
  Llama 2: open foundation and fine-tuned chat models.
  ArXiv abs/2307.09288.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:259950998)
  Cited by: [§4.2](#S4.SS2.p1.1 "4.2 Model Architecture and Training ‣ 4 Experimental Setup ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* Z. Wang, J. Li, H. Zhou, R. Weng, J. Wang, X. Huang, X. Han, J. Feng, C. Deng, and S. Huang (2025)
  Investigating and scaling up code-switching for multilingual language model pre-training.
  In Annual Meeting of the Association for Computational Linguistics,
  External Links: [Link](https://api.semanticscholar.org/CorpusID:277502377)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining"),
  [§1](#S1.p2.1 "1 Introduction ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining"),
  [§2](#S2.p2.1 "2 Related Work ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* Q. A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, G. Dong, H. Wei, H. Lin, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K. Yang, L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li, T. Xia, X. Ren, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Wan, Y. Liu, Z. Cui, Z. Zhang, Z. Qiu, S. Quan, and Z. Wang (2024)
  Qwen2.5 technical report.
  ArXiv abs/2412.15115.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:274859421)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* Y. Yang, Y. Zhang, C. Tar, and J. Baldridge (2019)
  PAWS-X: a cross-lingual adversarial dataset for paraphrase identification.
  In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP),
  Hong Kong, China,  pp. 3687–3692.
  External Links: [Link](https://aclanthology.org/D19-1382),
  [Document](https://dx.doi.org/10.18653/v1/D19-1382)
  Cited by: [§4.3](#S4.SS3.SSS0.Px3.p1.1 "Cross-lingual Understanding and Reasoning ‣ 4.3 Downstream Evaluation Suite ‣ 4 Experimental Setup ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* H. Yoo, C. Park, S. Yun, A. Oh, and H. Lee (2024)
  Code-switching curriculum learning for multilingual transfer in LLMs.
  In Annual Meeting of the Association for Computational Linguistics,
  External Links: [Link](https://api.semanticscholar.org/CorpusID:273822050)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining"),
  [§2](#S2.p1.1 "2 Related Work ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
* R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi (2019)
  HellaSwag: can a machine really finish your sentence?.
  In Annual Meeting of the Association for Computational Linguistics,
  External Links: [Link](https://api.semanticscholar.org/CorpusID:159041722)
  Cited by: [§4.3](#S4.SS3.SSS0.Px3.p1.1 "Cross-lingual Understanding and Reasoning ‣ 4.3 Downstream Evaluation Suite ‣ 4 Experimental Setup ‣ The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining").
