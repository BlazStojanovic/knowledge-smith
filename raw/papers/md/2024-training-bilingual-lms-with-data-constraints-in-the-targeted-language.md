---
arxiv: '2411.12986'
authors:
- Skyler Seto
- Maartje ter Hoeve
- Richard He Bai
- Natalie Schluter
- David Grangier
parser: ar5iv
retrieved: '2026-05-15'
source: paper
title: Training Bilingual LMs with Data Constraints in the Targeted Language
url: https://arxiv.org/abs/2411.12986
year: 2024
---

[2411.12986] Training Bilingual LMs with Data Constraints in the Targeted Language



# Training Bilingual LMs with Data Constraints in the Targeted Language

Skyler Seto, Maartje ter Hoeve11footnotemark: 1, He Bai, Natalie Schluter, David Grangier
  
Apple
  
{sseto,m\_terhoeve,hbai22,natschluter,d\_grangier}@apple.com
Equal contribution

###### Abstract

Large language models are trained on massive scrapes of the web, as required by current scaling laws. Most progress is made for English, given its abundance of high-quality pretraining data. For most other languages, however, such high quality pretraining data is unavailable. In this work, we study how to boost pretrained model performance in a data constrained target language by enlisting data from an auxiliary language for which high quality data is available.
We study this by
quantifying the performance gap between training with data in a data-rich auxiliary language compared with training in the target language, exploring the benefits of translation systems, studying the limitations of model scaling for data constrained languages, and proposing new methods for upsampling data
from the auxiliary language.
Our results show
that stronger auxiliary datasets result in performance gains
without modification to the model or training objective for close languages, and, in particular, that
performance gains due to the development of more information-rich English pretraining datasets can extend
to targeted language settings with limited data.

\pdfcolInitStack

tcb@breakable

Training Bilingual LMs with Data Constraints in the Targeted Language

  

Skyler Seto††thanks: Equal contribution, Maartje ter Hoeve11footnotemark: 1, He Bai, Natalie Schluter, David Grangier

Apple

{sseto,m\_terhoeve,hbai22,natschluter,d\_grangier}@apple.com

## 1 Introduction

Large language models (LLMs) have demonstrated exceptional performance on many tasks, including mathematical reasoning, coding capabilities, and knowledge-based question answering (Brown et al., [2020](#bib.bib5); Bubeck et al., [2023](#bib.bib6); OpenAI, [2023](#bib.bib34)) through pretraining on large corpora of web text. The vast majority of LLMs have been trained and evaluated on English, motivated by the abundance of high quality data. For other languages such large quantities of high quality and information-rich data are unavailable. Consequently, breakthroughs of similar scale are lacking for other languages, due to the lack of good data for pretraining.

Most non-English progress comes from relatively small bilingual models (e.g., Le et al., [2019](#bib.bib23); De Vries et al., [2019](#bib.bib10); Martin et al., [2019](#bib.bib31); Scheible et al., [2020](#bib.bib45); Wei et al., [2023a](#bib.bib50); Faysse et al., [2024](#bib.bib13)), or larger massively multilingual models (e.g., Le Scao et al., [2023](#bib.bib24); Intrator et al., [2024](#bib.bib18); Üstün et al., [2024](#bib.bib47)). Other LLMs such as Llama-2, GPT-3, and PaLM-2 that perform well across a variety of languages are trained primarily on English data with less than 20% of their data coming from other languages Xu et al. ([2024](#bib.bib56)). However, little progress has been made in understanding when an auxiliary language (such as English) can help learning a target language. This is particularly relevant to understand in the case of
target languages for which limited data is available.

(a) Data Pipeline

(b) Auxiliary Data Pretraining

(c) Data Transforms

Figure 1: (a) Data Pipeline: English data pipeline used for building large pretraining corpora in (Penedo et al., [2024](#bib.bib35)). (b) Auxiliary Data Pretraining: Combining high quality domain-specific pretraining data with a small amount of data from the target language for pretraining with limited target data. (c) Data Transforms: Many considerations when building datasets in languages with limited data.

This work studies the challenge of building datasets for pretraining language models with limited target language data. While much of the progress on building datasets for language model pretraining has focused on collecting and filtering more data, typically English (Figure [1(a)](#S1.F1.sf1 "In Figure 1 ‣ 1 Introduction ‣ Training Bilingual LMs with Data Constraints in the Targeted Language")), we investigate whether these advances transfer implicitly to auxiliary-target language learning (Figure [1(b)](#S1.F1.sf2 "In Figure 1 ‣ 1 Introduction ‣ Training Bilingual LMs with Data Constraints in the Targeted Language")), where English becomes an auxiliary language. *In particular, this work investigates whether better English datasets also lead to better models in other languages.* We focus on the impact of dataset size, filtering for data quality and style, and data selection for specialized information relevant to downstream evaluations (Figure [1(c)](#S1.F1.sf3 "In Figure 1 ‣ 1 Introduction ‣ Training Bilingual LMs with Data Constraints in the Targeted Language")), matching recent advancements in state-of-the-art open-source English datasets such as FineWeb-EDU Penedo et al. ([2024](#bib.bib35)), and DataComp Li et al. ([2024](#bib.bib26)).

Our key findings are:

1. 1.

   Auxiliary English data that is generated by some of the existing model-based data filtering pipelines for English can be helpful to complement limited data in a target language (Section [3.2](#S3.SS2 "3.2 Better English Datasets ‣ 3 Using English Data Selection Pipelines to Complement Limited Target Data ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"));
2. 2.

   When training with higher quality auxiliary data, more gains can be attributed to relevant information in the auxiliary data (Section [4.1](#S4.SS1 "4.1 High Quality Filtering ‣ 4 The Effect of Individual Data Transformations ‣ Training Bilingual LMs with Data Constraints in the Targeted Language")-[4.2](#S4.SS2 "4.2 Clustered Dataset Importance Sampling ‣ 4 The Effect of Individual Data Transformations ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"));
3. 3.

   Findings are not the same across multiple languages. We hypothesize that for languages that are “far” from English, gains from better English datasets do not help (Section [5](#S5 "5 Do Findings Hold Across Multiple Languages? ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"));
4. 4.

   There are limits to the size of models that are practical to pretrain with limited target data. This is because data size should scale linearly with model size Kaplan et al. ([2020](#bib.bib21)) and performance in the target language saturates without increasing target data regardless of increasing auxiliary data (Section [6.2](#S6.SS2 "6.2 Model Size Data Scaling ‣ 6 Scaling Limitations for Low Resource Languages ‣ Training Bilingual LMs with Data Constraints in the Targeted Language")).

## 2 Related Work

(a) Medium

(b) XL

Figure 2: Zero-shot accuracy of medium and XL models trained with higher quality English auxiliary data. Results are averaged over six eval datasets. We compare training with different auxiliary datasets (colors) on both English (solid) and German (striped) evaluations. Better English datasets show large increases in English, and smaller increases in German.

#### Multilingual Language Models

While much of the LLM research has focused on English, large-scale transformer-based multilingual language models have been trained on large multilingual corpora including mBERT Pires ([2019](#bib.bib38)), XLM Conneau and Lample ([2019](#bib.bib9)), mT5 Xue ([2020](#bib.bib57)), PolyLM Wei et al. ([2023b](#bib.bib51)), and Bloom Le Scao et al. ([2023](#bib.bib24)). These works focus on training models that are language balanced and can reason in over 100 languages. Other state-of-the-art language models such as Llama 2 Touvron et al. ([2023](#bib.bib46)), Falcon Almazrouei et al. ([2023](#bib.bib1)), and Palm 2 Anil et al. ([2023](#bib.bib2)) have multilingual capabilities but over 90% the training data is English, and these models are shown to perform poorly across a variety of languages, such as south east Asian languages Nguyen et al. ([2023](#bib.bib33)).

Other works focus on training smaller bilingual language models in French Faysse et al. ([2024](#bib.bib13)); Le et al. ([2019](#bib.bib23)); Martin et al. ([2019](#bib.bib31)), German Scheible et al. ([2020](#bib.bib45)), Dutch De Vries et al. ([2019](#bib.bib10)), or Chinese Wei et al. ([2023a](#bib.bib50)), but require a substantial amount of bilingual data: English and the respective language for pretraining. Other works focus on understanding languages LLMs reason in Wendler et al. ([2024](#bib.bib53)), and languages LLMs cannot learn Borenstein et al. ([2024](#bib.bib4)); Kallini et al. ([2024](#bib.bib20)). Still, little work has examined how information seen during pretraining in one language can help down-stream task performance in another language.

#### Cross-Lingual Transfer

Philippy et al. ([2023](#bib.bib37)) present a comprehensive survey on cross-lingual transfer in multilingual language models. The survey explains that cross-lingual transfer is a well studied topic for classic NLP tasks, such as part-of-speech tagging, named entity recognition, dependency parsing, natural language inference, machine translation, etc., although findings are not always consistent. Cross-lingual transfer is less well studied for the language modeling objective, and for modern down-stream evaluation tasks, such as ARC Clark et al. ([2018](#bib.bib7)), HellaSwag Zellers et al. ([2019](#bib.bib59)), PIQA Bisk et al. ([2020](#bib.bib3)), SCIQ Welbl et al. ([2017](#bib.bib52)), WinoGrande Sakaguchi et al. ([2021](#bib.bib44)), etc. [Philippy et al.](#bib.bib37) end with a number of recommendations, one of which is to study cross-lingual transfer in more detail for generative models, given their exceptional performance in recent years. Although our work does not directly study cross-lingual transfer (we do not first train on the auxiliary language, and then on the target language), our findings help understand how an auxiliary language can help a target language for which limited data is available.

#### Data Selection

Selecting high quality data for pretraining LLMs remains an active area of research. Early research on data selection was based on heuristics. For example, the original GPT-2 model was pretrained on outbound links filtered from Reddit based on heuristic indicators of whether users found the links interesting, educational, or funny Radford et al. ([2019](#bib.bib39)). Prior approaches also upsampled documents from high quality sources like Wikipedia Gururangan et al. ([2022](#bib.bib17)), or used combinations of heuristics such as absence of stop words, document length, word length, etc. Rae et al. ([2021](#bib.bib40)). Other works select data based on quality filters and time of collection Longpre et al. ([2023](#bib.bib29)), model-based quality filtering Sachdeva et al. ([2024](#bib.bib43)); Li et al. ([2024](#bib.bib26)), or textbook quality knowledge Gunasekar et al. ([2023](#bib.bib16)); Li et al. ([2023b](#bib.bib28)); Kong et al. ([2024](#bib.bib22)). An alternative approach to data selection is re-weighting data samples to select the best data mixtures for training Fan et al. ([2023](#bib.bib12)); Xie et al. ([2024](#bib.bib54)), or importance sampling based on a downstream task Grangier et al. ([2024b](#bib.bib15), [a](#bib.bib14)); Xie et al. ([2023](#bib.bib55)). Still a majority of these filtering techniques are applied to English-only datasets, and multilingual datasets such as mC4 have limited data filtering Xue ([2020](#bib.bib57)).

## 3 Using English Data Selection Pipelines to Complement Limited Target Data

Existing data selection pipelines have been shown to be effective in monolingual (English) pretraining. We investigate whether these pipelines are useful in the bilingual setup with limited target data. We always take English as the auxiliary language. Initially, we perform all experiments with German as the target language. In Section [5](#S5 "5 Do Findings Hold Across Multiple Languages? ‣ Training Bilingual LMs with Data Constraints in the Targeted Language") we discuss how our findings extend across multiple languages.

(a) Medium

(b) XL

Figure 3: Zero-shot accuracy of medium and XL models trained with higher quality English auxiliary data. Results are averaged over six evaluation datasets. For each setting evaluation is done in English and German.

### 3.1 General Implementation Details

#### Model.

We train decoder-only transformer models Vaswani et al. ([2017](#bib.bib48)) at different scales: medium and XL Brown et al. ([2020](#bib.bib5)). Models use the PolyLM tokenizer Wei et al. ([2023b](#bib.bib51)), with a total vocabulary size of 256K tokens using BPE.
Models are trained for 30K (medium) or 100K steps (XL) with batch size 1024.
Additional hyperparameters and model details are in Appendix [A](#A1 "Appendix A Hyperparameters and Training Details ‣ Training Bilingual LMs with Data Constraints in the Targeted Language").

#### Data.

We consider access to approximately 250M tokens111This is chosen to simulate the amount of data that would exist in the tail of mC4. We use German for our experiments to facilitate comparisons with having additional data, and to study translation and other methods. from the target language.
We upsample the 250M tokens of target language data to total 5% of the training time,
and the remaining 95% are different English auxiliary datasets or mC4 German for comparisons to monolingual models.
All data is pretokenized with packing to the full context length, and shuffled during training.

#### Evaluation.

We consider the average over six general understanding QA tasks: ARC-Easy, ARC-Challenge, Hellaswag, PIQA, SciQ, and Winogrande. These tasks are knowledge-based tasks that small models with limited data still perform well and many require knowledge. Non-English evaluations are conducted via translation of the original dataset. We use a mix of proprietary large language models to ensure good translations.

### 3.2 Better English Datasets

#### Methodology.

We compare the performance of models trained on combinations of German (target) and English (auxiliary) with varying existing English datasets based on the common crawl: mC4 Xue ([2020](#bib.bib57)), RedPajamav2 (RPJv2) Computer ([2023](#bib.bib8)), RefinedWeb (RFW) Penedo et al. ([2023](#bib.bib36)), FineWeb (FW), and FineWeb-EDU (FWE) Penedo et al. ([2024](#bib.bib35)). These datasets have been constructed following a pipeline of different filtering steps outlined in Figure [1(a)](#S1.F1.sf1 "In Figure 1 ‣ 1 Introduction ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"). The resulting datasets have higher quality filtering and cover different snapshots of the common crawl. They are all aimed at general language model pretraining. Further details for the datasets are available in Appendix [B](#A2 "Appendix B Dataset Details ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"). For comparison with monolingual models trained only on German (target), we consider both German mC4, and a low quality version of German mC4 referred to as “no ARC German mC4”.222We downsample data based on the ARC dataset using the clustered importance sampling procedure in Section [4.2](#S4.SS2 "4.2 Clustered Dataset Importance Sampling ‣ 4 The Effect of Individual Data Transformations ‣ Training Bilingual LMs with Data Constraints in the Targeted Language") to approximate having low quality data which is not in-domain but in the target language. This dataset serves as a baseline for a low quality dataset with less relevant data to the downstream tasks. We clarify this procedure in greater detail in the respective section as the details beyond low quality and out of domain are not critical.

#### Findings.

Across all tasks in Figure [2](#S2.F2 "Figure 2 ‣ 2 Related Work ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"), we observe that for English evaluations, better quality English datasets attain substantially higher performance on the English tasks. For the medium models, this corresponds to up to 5% increase between the worst and best performing English datasets. For the XL models, the performance improvement in English is up to 9%. For the same benchmarks translated into German, the performance increase is around 1% for the medium model and 2% for the XL model. Of all compared English datasets, FineWeb-EDU achieves the best average down-stream performance, and similar performance to the no ARC German dataset. There are two primary factors we hypothesize contribute to FineWeb-EDU achieving better performance on downstream tasks: high-quality data filtering, and information filtering. Next, we investigate these factors in more detail.

(a) Medium

(b) XL

Figure 4: Zero-shot accuracy of medium and XL models trained upsampling different downstream datasets. Results are averaged over six eval datasets.

## 4 The Effect of Individual Data Transformations

Section [3.2](#S3.SS2 "3.2 Better English Datasets ‣ 3 Using English Data Selection Pipelines to Complement Limited Target Data ‣ Training Bilingual LMs with Data Constraints in the Targeted Language") investigated the effect of using existing data filtering pipelines in the auxiliary language. Here we study individual data transformations in more detail.
For all experiments, we use the same experimental setup as in Section [A](#A1 "Appendix A Hyperparameters and Training Details ‣ Training Bilingual LMs with Data Constraints in the Targeted Language").
Unless otherwise stated, we refer to the mC4 datasets as “Base.”

### 4.1 High Quality Filtering

#### Motivation.

While filtering strategies achieve strong results on English downstream evaluations, training a filtering model can require more data than available, and typical high quality datasets may not be available in the target language for filtering. Further, it is unclear to what extent better filtering strategies would improve performance in a target language that might not benefit from seeing purely higher quality data in the auxiliary language.

#### Methodology.

To test the impact of filtering, we use the OH+ELI5 fast text classifier to filter data, and filter a large portion of the mC4 to the top 10% high quality documents following Li et al. ([2024](#bib.bib26)). Filtering with this classifier leads the model to train on high quality English, and question answer style data which specializes the model towards downstream evaluation. We compare the performance of models trained with English filtering and without, holding the German data the same for evaluation.

#### Findings.

We find in Figure [3](#S3.F3 "Figure 3 ‣ 3 Using English Data Selection Pipelines to Complement Limited Target Data ‣ Training Bilingual LMs with Data Constraints in the Targeted Language") that for the medium models performance improvements in English evaluations are 1.5%, but for translated German (target) evaluations, there is no improvement. For the XL models, English evaluations improve by 3%, but translated German evaluations are under 1% and within 1 standard error.

### 4.2 Clustered Dataset Importance Sampling

#### Motivation.

Prior work shows that LLMs reason in English and that information may be stored in a language agnostic space Wendler et al. ([2024](#bib.bib53)). However, they do not control how the information is seen during training, and while much of the information may be seen only in English (the predominant language), it may also be seen in other languages as the pretraining datasets are not made publicly available. Further, we note that FineWeb-EDU, in addition to high quality filtering, also filters for educational quality content Penedo et al. ([2024](#bib.bib35)).

#### Methodology.

To explicitly test whether information is shared between auxiliary and target languages, we upsample topics in English and evaluate on the target language comparing with having uniform data.
Specifically, given access to some small dataset representative of target knowledge, we group the data into target clusters and estimate importance sampling weights over the clusters following Grangier et al. ([2024b](#bib.bib15)). To train the clustering model, we take a small subset of the training set, produce embeddings from a smaller sentenceBERT model, then cluster the data according to the embeddings. For our importance sampling experiments, we upweight a subset of roughly 300B tokens from the English dataset. Given a small target set (typically on the order of 1000-10000 samples), we assign each sample to a cluster and upweight the original training set based on the cluster assignment proportions. In practice, we do not optimize for the cluster parameters jointly with the model weights, and instead precompute them based on the pretraining and target task data.

For clustering hyperparameters, we use a lightweight SentenceTransformers multilingual model333The particular model is called paraphrase-multilingual-MiniLM-L12-v2 model and is obtained from <https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2>. for extracting features Reimers and Gurevych ([2019](#bib.bib42)), and a balanced K𝐾K-means algorithm to cluster the embeddings into 64 clusters. To learn importance sampling weights, we use corresponding training sets, and available corpora that would be used for retrieval-based methods on the same tasks.

All upsampling is done in English to facilitate having specialized information in the auxiliary language, but not in the target language to measure the impact of information in a different language. For all experiments, we use the mC4 English dataset. Comparisons are made between having access to more German data of uniform quality, upsampling based on the ARC training set (general science knowledge), and upsampling based on the HellaSwag training set (general knowledge and QA style).

#### Findings.

Results in Figure [4](#S3.F4 "Figure 4 ‣ Findings. ‣ 3.2 Better English Datasets ‣ 3 Using English Data Selection Pipelines to Complement Limited Target Data ‣ Training Bilingual LMs with Data Constraints in the Targeted Language") show that for the medium model experiments, data selection provides 1.5% improvement over base mC4 auxiliary data on English language evaluations, but no improvement in the target language (German). For the XL models, we see 4% improvement in English evaluations, and 2% improvement in target language (German). These results highlight that larger models can take advantage of information in the auxiliary language, and the performance improvements are higher for the information upsampling than for model-based filtering without removing the data, allowing for training on new data at greater token counts.

(a) Medium

(b) XL

Figure 5: Zero-shot accuracy of medium and XL models trained upsampling based on synthetic datasets. Results are averaged over six eval datasets.

*Our findings indicate that while filtering for high quality data as in the OH filter has limited improvement for the target language, but further improvements come from data selection over important topics.*

### 4.3 Upsampling with Synthetic Examples

#### Motivation.

While performance improvements are achieved by data selection based on target downstream evaluations, having such data available at pretraining can be restrictive. For this reason, it can be desirable to be able to generate the necessary data for upsampling. While prior work has examined the use of LLM-generated data for pretraining Maini et al. ([2024](#bib.bib30)) and finetuning Li et al. ([2023a](#bib.bib27)); Yuan et al. ([2024](#bib.bib58)) language models, to our knowledge there is no prior work that investigates data selection based on synthetic examples.

#### Methodology

We generate a small set of synthetic examples following the approach in Maini et al. ([2024](#bib.bib30)). The synthetic data is created by prompting an off-the-shelf instruction finetuned language model to generate sets of questions relating to the topic of interest using the prompts in Section [D](#A4 "Appendix D Synthetic Prompts and Examples ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"). Generating synthetic data using an off-the-shelf language model can be both computationally expensive and challenging, however for the purpose of computing sampling weights, a small number of questions is sufficient.

For our experiments, we generate data using a frozen Mistral-7B instruction tuned model444<https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3> Jiang et al. ([2023](#bib.bib19)). In total around 2000 science question answer pairs (referred to as SciQ) are created with minimal filtering outside of confirming that both a question and answer are specified, and the model does not generate any additional details or explanations.

We additionally generate general instruction data. This data is aimed at broad QA style and general fact information which is helpful for downstream tasks. This data is generated in two stages by first having the frozen Mistral-7B model generate a set of questions, then answering the questions.

Using the sets of questions, we identify the cluster sample weights and upsample data accordingly as in Section [4.2](#S4.SS2 "4.2 Clustered Dataset Importance Sampling ‣ 4 The Effect of Individual Data Transformations ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"). The approach otherwise follows the same setup as in Section [4.2](#S4.SS2 "4.2 Clustered Dataset Importance Sampling ‣ 4 The Effect of Individual Data Transformations ‣ Training Bilingual LMs with Data Constraints in the Targeted Language") except for the synthetic QA pairs.

#### Findings

Results shown in Figure [5](#S4.F5 "Figure 5 ‣ Findings. ‣ 4.2 Clustered Dataset Importance Sampling ‣ 4 The Effect of Individual Data Transformations ‣ Training Bilingual LMs with Data Constraints in the Targeted Language") comparing synthetic data upsampling with upsampling from the downstream task demonstrates that synthetic data can be sufficient for incorporating information into the auxiliary English dataset. For the medium models on both English and German evaluations, models trained by upsampling synthetic data and real ARC attain the same performance, and for the XL models, upsampling synthetic data is within 1% for the English downstream tasks, and within 0.5% for the translated German evaluations.

### 4.4 Translation Systems

(a) Medium

(b) XL

Figure 6: Zero-shot accuracy of medium and XL models trained upsampling based on translating mC4. Results are averaged over six eval datasets.

#### Motivation.

Training directly on auxiliary language data can lead to improvements. However, an alternative strategy is to translate the auxiliary data into the target language, assuming a machine translation system is available between the two languages. This approach offers the benefit of training the model exclusively in one language, and, if the translation system is of high quality, it allows for training on high-quality data in the target language at the expense of translating the corpus. It is, therefore, crucial to also assess the level the translation system must possess to effectively translate data for pretraining.

#### Methodology.

For our experiments, we use light-weight translation systems of roughly 100-200M parameters. We consider three models with BLEU scores 16.0, 26.5, and 31.6 on the WMT-17 EN-DE benchmark task. We denote these models as v1, v2, and v3 corresponding to increasing BLEU score. All models are trained on translated versions of the mC4 english corpus. No other English data is included in the dataset, but we keep the 250M tokens of real German data as in prior experiments.

#### Findings.

We summarize our results in Figure [6](#S4.F6 "Figure 6 ‣ 4.4 Translation Systems ‣ 4 The Effect of Individual Data Transformations ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"). For all translation models, we find little difference from the quality of the translation. Both stronger translation models achieve similar performance with the worst model performing comparably to training on real German data. For the medium models, we see that the performance on the translated German evaluations yields similar performance to training on real German data. For the XL models, the translated data improves by around 1% over new German data for the v2 and v3 translation models. We hypothesize a few reasons this may be possible, but leave investigation of each of these to future work: (1) English CC data is higher quality than German CC data as there may be more data from more diverse sources. (2) Translated German data has a different distribution from real German data and this better matches the translated test evaluations. (3) Translated data from a small translation system might simplify language, making it easier for models to learn, following Eldan and Li ([2023](#bib.bib11)). (4) Portions of the dataset could not be translated by the systems and are removed. These portions might be noisy, and some unintended filtering may lead to slightly higher performance.

## 5 Do Findings Hold Across Multiple Languages?

Sections [3](#S3 "3 Using English Data Selection Pipelines to Complement Limited Target Data ‣ Training Bilingual LMs with Data Constraints in the Targeted Language")-[4](#S4 "4 The Effect of Individual Data Transformations ‣ Training Bilingual LMs with Data Constraints in the Targeted Language") focused on German as the target language. We now investigate to what extent our findings for German hold across different target languages. In Section [5.1](#S5.SS1 "5.1 Experiments Across Multiple Languages ‣ 5 Do Findings Hold Across Multiple Languages? ‣ Training Bilingual LMs with Data Constraints in the Targeted Language") we study additional languages individually, and in Section [5.2](#S5.SS2 "5.2 Multilingual Experiments ‣ 5 Do Findings Hold Across Multiple Languages? ‣ Training Bilingual LMs with Data Constraints in the Targeted Language") we investigate a multilingual setting combining a subset of four languages.

### 5.1 Experiments Across Multiple Languages

#### Motivation.

We add seven languages, across four language families: French, Italian, Portuguese, Spanish (Indo-European, same as German), Chinese (Sino-Tibetan), Japanese (Japonic), and Korean (Koreanic) Lewis et al. ([2015](#bib.bib25)).

#### Methodology.

For these experiments, we train models using the same experimental setup as in Section [3](#S3 "3 Using English Data Selection Pipelines to Complement Limited Target Data ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"). For all additional target language models, we train with approximately 250M tokens from the mC4 corpus in the respective language. For training monolingual models in the target language, we note that the Chinese and Korean mC4 corpora contain fewer than 100B tokens and thus the data is repeated for multiple epochs in the base language experiments.

#### Findings.

We report results for the XL models trained for 100B tokens in Figure [10](#A3.F10 "Figure 10 ‣ Appendix C Evaluation Metrics ‣ Training Bilingual LMs with Data Constraints in the Targeted Language")-[13](#A6.F13 "Figure 13 ‣ F.1 Average Zero Shot Accuracy Plots ‣ Appendix F Results for Multiple Languages ‣ Training Bilingual LMs with Data Constraints in the Targeted Language") in the appendix. Results are similar for the medium models.
Our results indicate that
our findings on English-German training
do not extend across all families of languages. In particular, we see improvements from FineWeb-EDU only on the Indo-European languages.

We investigate the performance differences further by evaluating the perplexity of models trained on combinations of target language data from mC4 and FineWeb-EDU. We selected four languages (German, French, Chinese, and Japanese) to examine the perplexity across two languages with improved performance from auxiliary data and two languages without. We compare data from the FineWeb-EDU dataset seen during training with data from the mC4 English portion which has not been seen during training. We translate 10,000 documents into each of the target languages and evaluate the perplexity of documents for both translated and original auxiliary data. We additionally report a metric we call *exceedance*, which is the fraction of documents in FineWeb-EDU with lower perplexity than the average perplexity of documents in mC4 English. A higher value shows that the data the model is trained on appears more in-distribution, and similar values across original and translated documents would reflect transfer between the languages. We observe that the *exceedance* is similar for French and German between translated and original documents (Table [3](#A6.T3 "Table 3 ‣ F.2 Perplexity Evaluations for Translated Training Data ‣ Appendix F Results for Multiple Languages ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"), Appendix [F.2](#A6.SS2 "F.2 Perplexity Evaluations for Translated Training Data ‣ Appendix F Results for Multiple Languages ‣ Training Bilingual LMs with Data Constraints in the Targeted Language")). However for Japanese and Chinese555For Chinese translations, there may be some artifacts that induce artificially high perplexity in the original Chinese mC4 data and further decrease *exceedance*. In particular, we find several symbols and characters that are not removed in text extraction. Nonetheless we believe the patterns should be consistent from Japanese. translations, the *exceedance* is lower, and translated FineWeb-EDU data has the same or higher perplexity compared with mC4 English data.

### 5.2 Multilingual Experiments

#### Motivation.

Our main findings investigate how auxiliary data benefits evaluations in a target language. However, many models trained on other languages (beyond English) are trained with many languages simultaneously. We investigate whether better auxiliary language datasets also improve multilingual model training.

#### Methodology.

We conduct experiments combining the German, French, Chinese, and Japanese language data towards multilingual training. For these experiments, we train a 1B model with the auxiliary dataset being FineWeb-EDU, and a mix of data from the four languages totaling 5% or 20% of the training. We again use the four language subset as we keep the data ratios the same per language, and did not want to increase the amount of data in target languages beyond the typical multilingual ratios in large open-source models Xu et al. ([2024](#bib.bib56)).

#### Findings.

We summarize the results in Table [1](#S5.T1 "Table 1 ‣ Findings. ‣ 5.2 Multilingual Experiments ‣ 5 Do Findings Hold Across Multiple Languages? ‣ Training Bilingual LMs with Data Constraints in the Targeted Language").
Our findings indicate that training with 20% of the data being a combination of target languages yields similar performance to training with 5%, resulting in a 1% reduction in performance on average when training with 20% multilingual data. When compared with training a bilingual model, we observe performance decreases for German and French, and increases for Chinese and Japanese.

|  | EN | DE | FR | JA | ZH |
| --- | --- | --- | --- | --- | --- |
| 5% | 61.07 | 46.02 | 46.64 | 44.00 | 45.92 |
| 20% | 59.25 | 46.44 | 46.61 | 43.23 | 44.36 |
| Bi |  | 47.16 | 47.52 | 42.73 | 44.38 |

Table 1: Evaluation of XL models in multilingual setting on “General Understanding Tasks” focusing on general reasoning, language understanding, and science knowledge in translated languages. Rows are the average accuracy for the respective language, with 5% or 20% of the training coming from a mix of the four languages. ‘Bi’ refers to the bilingual models.

## 6 Scaling Limitations for Low Resource Languages

We consider the limitations of data scaling for training bilingual language models for target languages with limited data. In Section [6.1](#S6.SS1 "6.1 Auxiliary Data Scaling ‣ 6 Scaling Limitations for Low Resource Languages ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"), we investigate the extent to which training on higher quality English data improves the amount of data needed in the target language to reach similar performance. In Section [6.2](#S6.SS2 "6.2 Model Size Data Scaling ‣ 6 Scaling Limitations for Low Resource Languages ‣ Training Bilingual LMs with Data Constraints in the Targeted Language") we investigate how feasible it is to train larger bilingual models with limited target data.

### 6.1 Auxiliary Data Scaling

Figure 7: Average accuracy over zero-shot benchmark tasks in translated German with increasing number of tokens in both target and auxiliary languages. Models are XL size and trained for 100B tokens.



(a) 1B Model Perplexity

(b) 3B Model Perplexity

Figure 8: Train and validation perplexity for 1.3B and 2.7B parameters with varying amount of data.

#### Motivation.

We measure the amount of data needed in the target language to match training on the high quality auxiliary data. Our goal is to quantify the advantage of training models with additional target data beyond the 250M tokens used in prior experiments. For this experiment we train models at different dataset sizes ranging from 0.1B to 100B tokens.

#### Methodology.

We investigate data scaling over both the mC4 dataset and FineWeb-EDU to measure the increase in German data needed to match higher quality auxiliary datasets.

#### Findings.

Our results are summarized in Figure [7](#S6.F7 "Figure 7 ‣ 6.1 Auxiliary Data Scaling ‣ 6 Scaling Limitations for Low Resource Languages ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"). Models trained on the base mC4 English data achieve similar performance to models trained with only 5B tokens of the base mC4 German data. In contrast, training on a high quality data such as FineWeb-EDU increases the amount of data needed to around between 10-15B tokens or equivalently 2x the amount of data in German. Note that the curves all plateau quickly at around 10B tokens. This corresponds to around 10 repetitions of the data, matching the results in Muennighoff et al. ([2024](#bib.bib32)).

There are two important data scaling considerations from Figure [7](#S6.F7 "Figure 7 ‣ 6.1 Auxiliary Data Scaling ‣ 6 Scaling Limitations for Low Resource Languages ‣ Training Bilingual LMs with Data Constraints in the Targeted Language") that justify training with auxiliary language data. First, for languages that are data constrained, it may be infeasible to collect twice as much data. Second, models trained on FineWeb-EDU attain similar performance at a rate of 5x the number of tokens (roughly 50B tokens on FineWebEDU matches the performance of 10B tokens of mC4 German data). An important avenue of research is to investigate data scaling at larger quantities of tokens. In particular, the FineWeb-EDU corpus totals 5.4T tokens and would require access to 1T tokens of German data, which is 3x the amount of data in mC4. As a result, while the data scaling shows large improvements from little German data, the large amount of readily available English data can make training on auxiliary data practical.

(a) 250M Tokens

(b) 1B Tokens

Figure 9: Zero-shot accuracy for 1B and 3B parameter models trained with either 250M tokens of data in the target language or 1B tokens. Both models are trained on the target language data for 5% of the training steps.

### 6.2 Model Size Data Scaling

In this section, we investigate to what extent results scale for training larger language models when data remains constrained in the target language.

#### Motivation.

In prior experiments, we fixed the amount of data from the target language to approximately 250M tokens and train for 100K steps with 5% of the training being in the target language. However, increasing model size necessitates increasing the number of tokens seen during training according to Chinchilla scaling laws Kaplan et al. ([2020](#bib.bib21)). This is challenging for low resource languages, as the number of repetitions increases with increased amount of training, which can lead to overfitting and saturate model performance.

#### Methodology.

We first examine the amount of overfitting larger models have on the same amount of data and ratio of training time. We train a roughly 3B (non-embedding parameter) model consisting of 32 layers, 32 attention heads, and a hidden dimension size of 2560. The model has a maximum sequence length of 2048 and is trained for 150K steps to match the same scaling ratio as for the 300M and 1B models.

#### Findings.

The perplexity of training and validation data is shown in Figure [8](#S6.F8 "Figure 8 ‣ 6.1 Auxiliary Data Scaling ‣ 6 Scaling Limitations for Low Resource Languages ‣ Training Bilingual LMs with Data Constraints in the Targeted Language") for the 1B (left) and 3B models (right). For 1B models, we see little to no overfitting at 5% ratio of training steps and the model achieves slightly lower perplexity than a model trained with a lower ratio of 1.5% target language data. In contrast, for the 3B model, there is clear overfitting from as early as 25% of the training with 5% ratio of training steps. This indicates that the number of repetitions is too high for the 3B model and performance may degrade.

Reducing the number of repetitions by reducing the fraction of training steps to 1.5% reduces overfitting as seen in Figure [8(b)](#S6.F8.sf2 "In Figure 8 ‣ 6.1 Auxiliary Data Scaling ‣ 6 Scaling Limitations for Low Resource Languages ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"), however perplexity remains similar between the 1B and 3B models when limited to 250M tokens. Increasing the number of available tokens results in decreasing perplexity on validation data and no overfitting for the 3B model, which is unnecessary for the 1B models. An overview of comparison with different data ratios for zero-shot accuracy is also provided in Appendix [E](#A5 "Appendix E Data Ratios ‣ Training Bilingual LMs with Data Constraints in the Targeted Language").

Based on Figure [8](#S6.F8 "Figure 8 ‣ 6.1 Auxiliary Data Scaling ‣ 6 Scaling Limitations for Low Resource Languages ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"), we evaluate both 1B and 3B models with data ratios that mitigate overfitting: 5% for 1B and 1.25% for 3B with 250M tokens in the target language. Results are shown in Figure [9](#S6.F9 "Figure 9 ‣ Findings. ‣ 6.1 Auxiliary Data Scaling ‣ 6 Scaling Limitations for Low Resource Languages ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"). First, Figure [9(a)](#S6.F9.sf1 "In Figure 9 ‣ Findings. ‣ 6.1 Auxiliary Data Scaling ‣ 6 Scaling Limitations for Low Resource Languages ‣ Training Bilingual LMs with Data Constraints in the Targeted Language") shows that the 3B model performs similarly to the 1B model with the same 250M amount of target language data matching the similar perplexity values between the two models. However increasing to 1B tokens results in similar improvements in both English and German when increasing model size from 1B to 3B as shown in Figure [9(b)](#S6.F9.sf2 "In Figure 9 ‣ Findings. ‣ 6.1 Auxiliary Data Scaling ‣ 6 Scaling Limitations for Low Resource Languages ‣ Training Bilingual LMs with Data Constraints in the Targeted Language").

Summarizing, there is a limit to the size of models that can be trained with limited target data. Model performance for 250M tokens saturates at 1B parameter models (more than the available data in mC4 for ∼20similar-toabsent20\sim 20 languages). Increasing to a 3B parameter model necessitates training on 1B tokens of data (more than the available data in mC4 for ∼40similar-toabsent40\sim 40 languages).

## 7 Conclusion

This work studies how an auxiliary language for which an abundance of training data is available can boost pretraining for a target language for which only limited data is available. We find that adding auxiliary high quality data obtained by data filtering can improve performance in a target language. Moreover, our results indicate that most of these gains can be attributed to having relevant information in the auxiliary language data, which may not be present in the target language. However, we find that results are inconsistent across different target languages. We hypothesize that for languages that are further from English, better English datasets are not as helpful as information is not shared between them. Finally, we find limitations to scaling models for languages that are data constrained. This work takes a step towards pretraining language models in languages with limited data, and can inspire more research into bilingual or multilingual learning under dataset constraints.

## 8 Limitations

In this section we list some limitations of our work.

#### Languages included.

Our primary focus is on English-German language training, as these two Germanic family languages share linguistic similarities Lewis et al. ([2015](#bib.bib25)). German is one of the most well-represented languages in the mC4 dataset, facilitating model comparisons with varying amounts of German and English data. Furthermore, the availability of extensive public resources for German, including translation systems and translated evaluation data, further supports our emphasis on this language pair. We experimented with seven additional datasets including French, Spanish, Italian, Portuguese, Korean, Japanese, and Chinese. However, we note that there are many other languages within mC4 and more broadly which can benefit from having auxiliary English data for pretraining. Due to limited evaluation benchmarks and availability of target language data for comparison, we leave investigation for truly low-resource languages to future work.

#### Evaluation data.

Another limitation in evaluating language models for languages other than English is that many datasets have been translated from English. These datasets may contain cultural biases or information that is not available on the web in other languages. As a result, certain aspects of the evaluation may lead to improved performance when using English auxiliary or translated data. Additionally, translated data often exhibits a distribution different from that of real data in the target languages. Therefore, an important direction for future work is the development of evaluation datasets that are not based on translation, which is essential for more accurate evaluation of multilingual language models.

#### Model size.

Finally, this work studies three model sizes up to 3B models. We note that there are many standard benchmarks that can be evaluated at 1B-3B scale, however many more benchmarks and patterns can appear at larger model sizes. It is important future work to evaluate whether the results extend to larger scales including evaluating potential “emergent behaviors” as well as risks at larger scales Wei et al. ([2022](#bib.bib49)).

## Acknowledgements

We are grateful to Masha Fedzechkina Donaldson, Kunal Talwar, and Maureen de Seyssel for their helpful discussions, comments, and thoughtful feedback in reviewing this work.

## References

* Almazrouei et al. (2023)

  Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru, Mérouane Debbah, Étienne Goffinet, Daniel Hesslow, Julien Launay, Quentin Malartic, et al. 2023.
  The falcon series of open language models.
  *arXiv preprint arXiv:2311.16867*.
* Anil et al. (2023)

  Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. 2023.
  Palm 2 technical report.
  *arXiv preprint arXiv:2305.10403*.
* Bisk et al. (2020)

  Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. 2020.
  Piqa: Reasoning about physical commonsense in natural language.
  In *Proceedings of the AAAI conference on artificial intelligence*, volume 34, pages 7432–7439.
* Borenstein et al. (2024)

  Nadav Borenstein, Anej Svete, Robin Chan, Josef Valvoda, Franz Nowak, Isabelle Augenstein, Eleanor Chodroff, and Ryan Cotterell. 2024.
  What languages are easy to language-model? a perspective from learning probabilistic regular languages.
  *arXiv preprint arXiv:2406.04289*.
* Brown et al. (2020)

  Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020.
  Language models are few-shot learners.
  *Advances in neural information processing systems*, 33:1877–1901.
* Bubeck et al. (2023)

  Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. 2023.
  Sparks of artificial general intelligence: Early experiments with gpt-4.
  *arXiv preprint arXiv:2303.12712*.
* Clark et al. (2018)

  Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018.
  Think you have solved question answering? try arc, the ai2 reasoning challenge.
  *arXiv preprint arXiv:1803.05457*.
* Computer (2023)

  Together Computer. 2023.
  [Redpajama: an open dataset for training large language models](https://github.com/togethercomputer/RedPajama-Data).
* Conneau and Lample (2019)

  Alexis Conneau and Guillaume Lample. 2019.
  Cross-lingual language model pretraining.
  *Advances in neural information processing systems*, 32.
* De Vries et al. (2019)

  Wietse De Vries, Andreas van Cranenburgh, Arianna Bisazza, Tommaso Caselli, Gertjan van Noord, and Malvina Nissim. 2019.
  Bertje: A dutch bert model.
  *arXiv preprint arXiv:1912.09582*.
* Eldan and Li (2023)

  Ronen Eldan and Yuanzhi Li. 2023.
  Tinystories: How small can language models be and still speak coherent english?
  *arXiv preprint arXiv:2305.07759*.
* Fan et al. (2023)

  Simin Fan, Matteo Pagliardini, and Martin Jaggi. 2023.
  Doge: Domain reweighting with generalization estimation.
  *arXiv preprint arXiv:2310.15393*.
* Faysse et al. (2024)

  Manuel Faysse, Patrick Fernandes, Nuno Guerreiro, António Loison, Duarte Alves, Caio Corro, Nicolas Boizard, João Alves, Ricardo Rei, Pedro Martins, et al. 2024.
  Croissantllm: A truly bilingual french-english language model.
  *arXiv preprint arXiv:2402.00786*.
* Grangier et al. (2024a)

  David Grangier, Simin Fan, Skyler Seto, and Pierre Ablin. 2024a.
  Task-adaptive pretrained language models via clustered-importance sampling.
  *arXiv preprint arXiv:2410.03735*.
* Grangier et al. (2024b)

  David Grangier, Angelos Katharopoulos, Pierre Ablin, and Awni Hannun. 2024b.
  Specialized language models with cheap inference from limited domain data.
  *arXiv preprint arXiv:2402.01093*.
* Gunasekar et al. (2023)

  Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, et al. 2023.
  Textbooks are all you need.
  *arXiv preprint arXiv:2306.11644*.
* Gururangan et al. (2022)

  Suchin Gururangan, Dallas Card, Sarah K Dreier, Emily K Gade, Leroy Z Wang, Zeyu Wang, Luke Zettlemoyer, and Noah A Smith. 2022.
  Whose language counts as high quality? measuring language ideologies in text data selection.
  *arXiv preprint arXiv:2201.10474*.
* Intrator et al. (2024)

  Yotam Intrator, Matan Halfon, Roman Goldenberg, Reut Tsarfaty, Matan Eyal, Ehud Rivlin, Yossi Matias, and Natalia Aizenberg. 2024.
  [Breaking the language barrier: Can direct inference outperform pre-translation in multilingual LLM applications?](https://doi.org/10.18653/v1/2024.naacl-short.75)
  In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers)*, pages 829–844, Mexico City, Mexico. Association for Computational Linguistics.
* Jiang et al. (2023)

  Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023.
  Mistral 7b.
  *arXiv preprint arXiv:2310.06825*.
* Kallini et al. (2024)

  Julie Kallini, Isabel Papadimitriou, Richard Futrell, Kyle Mahowald, and Christopher Potts. 2024.
  Mission: Impossible language models.
  *arXiv preprint arXiv:2401.06416*.
* Kaplan et al. (2020)

  Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020.
  Scaling laws for neural language models.
  *arXiv preprint arXiv:2001.08361*.
* Kong et al. (2024)

  Xiang Kong, Tom Gunter, and Ruoming Pang. 2024.
  Large language model-guided document selection.
  *arXiv preprint arXiv:2406.04638*.
* Le et al. (2019)

  Hang Le, Loïc Vial, Jibril Frej, Vincent Segonne, Maximin Coavoux, Benjamin Lecouteux, Alexandre Allauzen, Benoit Crabbé, Laurent Besacier, and Didier Schwab. 2019.
  Flaubert: Unsupervised language model pre-training for french.
  *arXiv preprint arXiv:1912.05372*.
* Le Scao et al. (2023)

  Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. 2023.
  Bloom: A 176b-parameter open-access multilingual language model.
* Lewis et al. (2015)

  M. Paul Lewis, Gary F. Simons, and Charles D. Fennig. 2015.
  Ethnologue: Languages of the world, eighteenth edition.
  SIL International, Dallas, Texas.
* Li et al. (2024)

  Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, et al. 2024.
  Datacomp-lm: In search of the next generation of training sets for language models.
  *arXiv preprint arXiv:2406.11794*.
* Li et al. (2023a)

  Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Omer Levy, Luke Zettlemoyer, Jason Weston, and Mike Lewis. 2023a.
  Self-alignment with instruction backtranslation.
  *arXiv preprint arXiv:2308.06259*.
* Li et al. (2023b)

  Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. 2023b.
  Textbooks are all you need ii: phi-1.5 technical report.
  *arXiv preprint arXiv:2309.05463*.
* Longpre et al. (2023)

  Shayne Longpre, Gregory Yauney, Emily Reif, Katherine Lee, Adam Roberts, Barret Zoph, Denny Zhou, Jason Wei, Kevin Robinson, David Mimno, et al. 2023.
  A pretrainer’s guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity.
  *arXiv preprint arXiv:2305.13169*.
* Maini et al. (2024)

  Pratyush Maini, Skyler Seto, He Bai, David Grangier, Yizhe Zhang, and Navdeep Jaitly. 2024.
  Rephrasing the web: A recipe for compute and data-efficient language modeling.
  *arXiv preprint arXiv:2401.16380*.
* Martin et al. (2019)

  Louis Martin, Benjamin Muller, Pedro Javier Ortiz Suárez, Yoann Dupont, Laurent Romary, Éric Villemonte de La Clergerie, Djamé Seddah, and Benoît Sagot. 2019.
  Camembert: a tasty french language model.
  *arXiv preprint arXiv:1911.03894*.
* Muennighoff et al. (2024)

  Niklas Muennighoff, Alexander Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, and Colin A Raffel. 2024.
  Scaling data-constrained language models.
  *Advances in Neural Information Processing Systems*, 36.
* Nguyen et al. (2023)

  Xuan-Phi Nguyen, Wenxuan Zhang, Xin Li, Mahani Aljunied, Qingyu Tan, Liying Cheng, Guanzheng Chen, Yue Deng, Sen Yang, Chaoqun Liu, et al. 2023.
  Seallms–large language models for southeast asia.
  *arXiv preprint arXiv:2312.00738*.
* OpenAI (2023)

  OpenAI. 2023.
  Gpt-4 technical report.
  *ArXiv*, abs/2303.08774.
* Penedo et al. (2024)

  Guilherme Penedo, Hynek Kydlíček, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, Thomas Wolf, et al. 2024.
  The fineweb datasets: Decanting the web for the finest text data at scale.
  *arXiv preprint arXiv:2406.17557*.
* Penedo et al. (2023)

  Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. 2023.
  The refinedweb dataset for falcon llm: outperforming curated corpora with web data, and web data only.
  *arXiv preprint arXiv:2306.01116*.
* Philippy et al. (2023)

  Fred Philippy, Siwen Guo, and Shohreh Haddadan. 2023.
  [Towards a common understanding of contributing factors for cross-lingual transfer in multilingual language models: A review](https://doi.org/10.18653/v1/2023.acl-long.323).
  In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 5877–5891, Toronto, Canada. Association for Computational Linguistics.
* Pires (2019)

  T Pires. 2019.
  How multilingual is multilingual bert.
  *arXiv preprint arXiv:1906.01502*.
* Radford et al. (2019)

  Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019.
  Language models are unsupervised multitask learners.
  *OpenAI blog*, 1(8):9.
* Rae et al. (2021)

  Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. 2021.
  Scaling language models: Methods, analysis & insights from training gopher.
  *arXiv preprint arXiv:2112.11446*.
* Raffel et al. (2020)

  Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020.
  Exploring the limits of transfer learning with a unified text-to-text transformer.
  *The Journal of Machine Learning Research*, 21(1):5485–5551.
* Reimers and Gurevych (2019)

  Nils Reimers and Iryna Gurevych. 2019.
  [Sentence-bert: Sentence embeddings using siamese bert-networks](http://arxiv.org/abs/1908.10084).
  In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics.
* Sachdeva et al. (2024)

  Noveen Sachdeva, Benjamin Coleman, Wang-Cheng Kang, Jianmo Ni, Lichan Hong, Ed H Chi, James Caverlee, Julian McAuley, and Derek Zhiyuan Cheng. 2024.
  How to train data-efficient llms.
  *arXiv preprint arXiv:2402.09668*.
* Sakaguchi et al. (2021)

  Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021.
  Winogrande: An adversarial winograd schema challenge at scale.
  *Communications of the ACM*, 64(9):99–106.
* Scheible et al. (2020)

  Raphael Scheible, Fabian Thomczyk, Patric Tippmann, Victor Jaravine, and Martin Boeker. 2020.
  Gottbert: a pure german language model.
  *arXiv preprint arXiv:2012.02110*.
* Touvron et al. (2023)

  Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023.
  Llama 2: Open foundation and fine-tuned chat models.
  *arXiv preprint arXiv:2307.09288*.
* Üstün et al. (2024)

  Ahmet Üstün, Viraat Aryabumi, Zheng Yong, Wei-Yin Ko, Daniel D’souza, Gbemileke Onilude, Neel Bhandari, Shivalika Singh, Hui-Lee Ooi, Amr Kayid, Freddie Vargus, Phil Blunsom, Shayne Longpre, Niklas Muennighoff, Marzieh Fadaee, Julia Kreutzer, and Sara Hooker. 2024.
  [Aya model: An instruction finetuned open-access multilingual language model](https://doi.org/10.18653/v1/2024.acl-long.845).
  In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 15894–15939, Bangkok, Thailand. Association for Computational Linguistics.
* Vaswani et al. (2017)

  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017.
  Attention is all you need.
  *Advances in neural information processing systems*, 30.
* Wei et al. (2022)

  Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. 2022.
  Emergent abilities of large language models.
  *arXiv preprint arXiv:2206.07682*.
* Wei et al. (2023a)

  Tianwen Wei, Liang Zhao, Lichang Zhang, Bo Zhu, Lijie Wang, Haihua Yang, Biye Li, Cheng Cheng, Weiwei Lü, Rui Hu, et al. 2023a.
  Skywork: A more open bilingual foundation model.
  *arXiv preprint arXiv:2310.19341*.
* Wei et al. (2023b)

  Xiangpeng Wei, Haoran Wei, Huan Lin, Tianhao Li, Pei Zhang, Xingzhang Ren, Mei Li, Yu Wan, Zhiwei Cao, Binbin Xie, et al. 2023b.
  Polylm: An open source polyglot large language model.
  *arXiv preprint arXiv:2307.06018*.
* Welbl et al. (2017)

  Johannes Welbl, Nelson F Liu, and Matt Gardner. 2017.
  Crowdsourcing multiple choice science questions.
  *arXiv preprint arXiv:1707.06209*.
* Wendler et al. (2024)

  Chris Wendler, Veniamin Veselovsky, Giovanni Monea, and Robert West. 2024.
  Do llamas work in english? on the latent language of multilingual transformers.
  *arXiv preprint arXiv:2402.10588*.
* Xie et al. (2024)

  Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy S Liang, Quoc V Le, Tengyu Ma, and Adams Wei Yu. 2024.
  Doremi: Optimizing data mixtures speeds up language model pretraining.
  *Advances in Neural Information Processing Systems*, 36.
* Xie et al. (2023)

  Sang Michael Xie, Shibani Santurkar, Tengyu Ma, and Percy S Liang. 2023.
  Data selection for language models via importance resampling.
  *Advances in Neural Information Processing Systems*, 36:34201–34227.
* Xu et al. (2024)

  Yuemei Xu, Ling Hu, Jiayi Zhao, Zihan Qiu, Yuqi Ye, and Hanwen Gu. 2024.
  A survey on multilingual large language models: Corpora, alignment, and bias.
  *arXiv preprint arXiv:2404.00929*.
* Xue (2020)

  L Xue. 2020.
  mt5: A massively multilingual pre-trained text-to-text transformer.
  *arXiv preprint arXiv:2010.11934*.
* Yuan et al. (2024)

  Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. 2024.
  Self-rewarding language models.
  *arXiv preprint arXiv:2401.10020*.
* Zellers et al. (2019)

  Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019.
  Hellaswag: Can a machine really finish your sentence?
  *arXiv preprint arXiv:1905.07830*.

## Appendix A Hyperparameters and Training Details

The medium-scale (300M non-embedding parameter) model consists of 24 layers, 16 attention heads, and a hidden dimension size of 1024. The XL-scale (1.3B non-embedding parameter) model consists of 24 layers, 16 attention heads, and a hidden dimension size of 2048. Both models have a maximum sequence length of 1024.

The baseline models are trained using NVIDIA’s Megatron-LM666<https://github.com/NVIDIA/Megatron-LM> repository for pretraining language models. The medium size models are trained for a total of 30K steps, and 100K steps for the XL models at a batch size of 1024. All models are trained using a maximum learning rate of 0.00030.00030.0003 for the medium model and 0.00020.00020.0002 for the XL model, and a minimum learning rate of 0.000010.000010.00001 with a cosine learning rate scheduler and warmup for 1%percent11\% of the total steps. For regularization, we use a weight decay of 0.010.010.01, along with a gradient clipping norm of 1.01.01.0. Models are trained with the Adam optimizer using β1=0.9subscript𝛽10.9\beta\_{1}=0.9 and β2=0.999subscript𝛽20.999\beta\_{2}=0.999.

The total training time for XL models on roughly 100B tokens is around 1000 GPUh on Nvidia H100 GPUs. For medium size models, the total training time is around 200 hours for roughly 30B tokens.

## Appendix B Dataset Details

### B.1 Train Sets

* •

  mC4: The primary pretraining corpus in our experiments is multilingual Colossal Clean Crawled Corpus (mC4), a curated text dataset comprising over 6.3T tokens. This corpus is derived from CommonCrawl and used for pretraining numerous language models Brown et al. ([2020](#bib.bib5)); Raffel et al. ([2020](#bib.bib41)); Touvron et al. ([2023](#bib.bib46)). The dataset is chosen as all languages have similar data extraction pipelines including line length filter, cld3 language detection, and deduplication Xue ([2020](#bib.bib57)). The English portion contains 2.7T tokens, German contains 350B tokens, French contains 320B tokens, Spanish contains 430B tokens, Portuguese contains 146B tokens, Italian contains 160B tokens, Korean contains 26B tokens, Japanese contains 160B tokens, and Chinese contains 40B tokens.
* •

  RedPajamav2: A pretraining corpus with light filtering (primarily only deduplication) comprising 30T tokens and 20T tokens of English text. We focus on the English portion of the dataset only and train using a random shuffled subset of both the head and middle portions Computer ([2023](#bib.bib8)).
* •

  RefinedWeb: The dataset is also derived from the CommonCrawl, however has a more stringent filtering process including trafilatura text extraction, document and line level rules, and fuzzy duplication over the original C4 processing Penedo et al. ([2023](#bib.bib36)).
* •

  FineWeb: This dataset is derived from the CommonCrawl with the aim of replicating RefinedWeb at larger scales. The dataset has some additional filtering including Gopher filtering Rae et al. ([2021](#bib.bib40)), additional C4 filters, and custom filters for text quality  Penedo et al. ([2024](#bib.bib35)).
* •

  FineWeb-EDU: A subset of the FineWeb dataset which is filtered according to a classifier trained on annotations for educational quality from Llama-3 70B model Penedo et al. ([2024](#bib.bib35)).

### B.2 Zero Shot Evaluations

* •

  SciQ: A dataset of science exam questions, specifically designed to evaluate the ability of NLP models in understanding and reasoning within the scientific domain (Welbl et al., [2017](#bib.bib52)).
* •

  ARC Challenge (ARC-C): This dataset is part of the AI2 Reasoning Challenge (ARC) (Clark et al., [2018](#bib.bib7)), containing science exam questions from grades 3 to 9. The ARC Challenge set includes more difficult questions that necessitate higher-order reasoning.
* •

  ARC Easy (ARC-E): The Easy set of the AI2 Reasoning Challenge (Clark et al., [2018](#bib.bib7)) features questions from the same source as ARC-C but are considered less challenging and do not require as advanced reasoning skills.
* •

  Winogrande (Wino.): This dataset challenges models on common sense reasoning in a language context, focusing on pronoun disambiguation tasks (Sakaguchi et al., [2021](#bib.bib44)).
* •

  PIQA: Physical Interaction Question Answering tests the understanding of everyday physical processes, an aspect of practical common sense (Bisk et al., [2020](#bib.bib3)).
* •

  HellaSwag: This dataset evaluates a model’s ability to complete scenarios in a contextually and logically coherent manner, requiring both language understanding and common sense reasoning (Zellers et al., [2019](#bib.bib59)).

For each of the eval datasets, we include the number of samples for each translated evaluation in Table [2](#A2.T2 "Table 2 ‣ B.2 Zero Shot Evaluations ‣ Appendix B Dataset Details ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"). For our evaluations, we use the lm-eval-harness repository777<https://github.com/EleutherAI/lm-evaluation-harness> for zero-shot accuracy on QA tasks.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Dataset | EN | DE | FR | ZH | JA |
| ARC-C | 1,172 | 1,137 | 1,147 | 1,146 | 1,147 |
| ARC-E | 2,376 | 2,260 | 2,271 | 2,271 | 2,271 |
| HS | 10,042 | 9,368 | 9,338 | 9,266 | 10,033 |
| PIQA | 1,838 | 1,838 | 1838 | 1,838 | 1,838 |
| SCIQ | 1,000 | 950 | 953 | 1,000 | 926 |
| WG | 1,267 | 1,184 | 1,215 | 1,059 | 1,096 |

Table 2: Evaluation set sizes for each language.

### B.3 Number of Data Files for Filtering Experiments

The mC4 English portion of the dataset is split into roughly 11,264 files totaling 2.7T tokens of data Xue ([2020](#bib.bib57)). For each of our experiments, data is filtered differently, and as such varying numbers of files are needed for training. At a baseline, we consider the first 1500 files totaling roughly 350B tokens of data. This number was selected to match the total amount of German data which is recorded as 347B tokens using the mT5 tokenizer Xue ([2020](#bib.bib57)). For the OH classifier, we use the first 10,000 files and filter down to 10% of the dataset. For German, Japanese, Spanish, Portuguese, Italian, and French models, we use the first two files of data totaling roughly 250-300M tokens of data. For Chinese and Korean models, we use the first 7 files totaling roughly 250M tokens.

### B.4 License and Attribution

All datasets used in this paper are supported by public licenses including ODC and Apache. The pre-trained models including Mistral and OH FastText classifiers are also supported by public licenses including Apache and MIT licenses. We use the Megatron codebase under the Nvidia license for pre-training and the lm-eval-harness (MIT) for evaluations. All models and datasets are collected from Huggingface via the datasets library where possible.
We use a proprietary translation system for fast translation at scale and are thus unable to provide details of the license at this time.

## Appendix C Evaluation Metrics

The metric utilized for evaluation is the macro token level perplexity. Given a batch of encoded texts, the perplexity at the token level was computed as follows:

Given the accumulated loss over the entire dataset, denoted as L𝐿L, and the total number of tokens, represented by T𝑇T, the macro token-level perplexity, denoted as 𝒫𝒫\mathcal{P}, is calculated as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒫=exp⁡(min⁡(20,LT))𝒫20𝐿𝑇\mathcal{P}=\exp\left(\min\left(20,\frac{L}{T}\right)\right) |  | (1) |

Where:

* •

  exp\exp is the exponential function.
* •

  L𝐿L is the cumulative loss over all shifted logits and labels in the dataset.
* •

  T𝑇T is the total number of tokens in the dataset.

The value of 20 acts as an upper limit to stabilize the metric in cases of high loss values.

For zero-shot MCQ accuracy evaluations, we compute the perplexity of each sentence completion, and choose the lowest perplexity choice. We use the lm-evaluation-harness and where possible evaluate with the length-normalized accuracy. Unless otherwise stated, all evaluations are zero-shot.

(a) German

(b) French

Figure 10: Zero-shot accuracy of XL models trained with various English auxiliary data for German and French. Results are averaged over six eval datasets.

## Appendix D Synthetic Prompts and Examples

For building the synthetic corpus used in our data selection experiments, we consider three prompts for generating science questions (similar to many downstream tasks), fact-based QA data, and instruction-based writing (such as emails, books, lists, etc.). For generating science questions, we generate both the question and answer. For the fact and instruction data, we first generate the questions using the prompt, and subsequently generate the answer without any additional prompting.

Science Question Prompt

Give me a set of ten question and answer pairs on topics relating to Physics, Chemistry and Biology that a high school student would be able to answer. The response should be in the form Question: <question> \n Answer: <answer> \n \n with an answer that is less than ten words. The response should not contain any other details or explanations about the question or answer.

Facts Question Prompt

People from different social and educational backgrounds, beliefs, ethnicity and gender are asking an AI assistant for information. They are looking for detailed explanations about encyclopedic facts on Wikipedia and in textbooks, about philosophy, nature, science, entertainment, literature, geography, socialogy, law, history, etc. Write an interesting and difficult question that would be sent to the AI assistant:

Instruction Writing Prompt

People from different social and educational backgrounds, beliefs, ethnicity and gender are asking an AI assistant to help them write a piece of text that they need for their work or their personal life. They can ask the AI Assistant to write a document (email, letter, official document...). Each request comes with a long, precise and detailed description of what needs to be in the text, and why they need this document. The request may also include information about the writing style, the tone, the target audience or the layout of the text. The description of the task is formal, detailed and clear. Each request is composed of a few paragraphs written in English, and starts with the tag <request>. Here is some of the most interesting and original requests sent to the AI assistant:

(a) Spanish

(b) Portuguese

Figure 11: Zero-shot accuracy of XL models trained with various English auxiliary data for Spanish and Portuguese. Results are averaged over six eval datasets.

## Appendix E Data Ratios

We experiment with different data ratios beyond 5% used during training. For our ablation, we study the XL model (1.3B) and a 2.7B non-embedding parameters model. Results are reported in Figure [14](#A6.F14 "Figure 14 ‣ F.2 Perplexity Evaluations for Translated Training Data ‣ Appendix F Results for Multiple Languages ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"). We see that performance is around the same for XL models, but 2.7B models perform worse with larger data ratios.

(a) Italian

(b) Korean

Figure 12: Zero-shot accuracy of XL models trained with various English auxiliary data for Italian and Korean. Results are averaged over six eval datasets.

## Appendix F Results for Multiple Languages

### F.1 Average Zero Shot Accuracy Plots

We present experimental results comparing the best performing approaches for French and German languages in Figure [10](#A3.F10 "Figure 10 ‣ Appendix C Evaluation Metrics ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"), Spanish and Portuguese in Figure [11](#A4.F11 "Figure 11 ‣ Appendix D Synthetic Prompts and Examples ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"), Italian and Korean in Figure [12](#A5.F12 "Figure 12 ‣ Appendix E Data Ratios ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"), and for Japanese and Chinese languages in Figure [13](#A6.F13 "Figure 13 ‣ F.1 Average Zero Shot Accuracy Plots ‣ Appendix F Results for Multiple Languages ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"). For each language, we take the best performing dataset and model found in German from Sections [3.2](#S3.SS2 "3.2 Better English Datasets ‣ 3 Using English Data Selection Pipelines to Complement Limited Target Data ‣ Training Bilingual LMs with Data Constraints in the Targeted Language")-[5.1](#S5.SS1 "5.1 Experiments Across Multiple Languages ‣ 5 Do Findings Hold Across Multiple Languages? ‣ Training Bilingual LMs with Data Constraints in the Targeted Language").

(a) Japanese

(b) Chinese

Figure 13: Zero-shot accuracy of XL models trained with various English auxiliary data for Japanese and Chinese. Results are averaged over six eval datasets.

### F.2 Perplexity Evaluations for Translated Training Data

In Sections [5.1](#S5.SS1 "5.1 Experiments Across Multiple Languages ‣ 5 Do Findings Hold Across Multiple Languages? ‣ Training Bilingual LMs with Data Constraints in the Targeted Language") and [F.1](#A6.SS1 "F.1 Average Zero Shot Accuracy Plots ‣ Appendix F Results for Multiple Languages ‣ Training Bilingual LMs with Data Constraints in the Targeted Language") we found that performance trends were not the same across languages. In particular, French, German, Portuguese, and Spanish (belonging to the same language family) have similar patterns, however, performance for Chinese, Japanese, and Korean exhibit different patterns. To further test whether the models retain knowledge from one language in another, we translate a small portion of the training set from FineWeb-EDU and mC4 English totaling 10,000 documents. We then translate the data using the v3 translation system from Section [4.4](#S4.SS4 "4.4 Translation Systems ‣ 4 The Effect of Individual Data Transformations ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"). We measure both the macro perplexity of all documents as well as the fraction of times where the translated and original data from FineWeb-EDU (training set) have lower loss than the average loss of documents from mC4 English (not part of the training set but from a similar distribution). We refer to this quantity as translated and original *exceedance*. Having lower loss means the data is more familiar to the model, and having an equal *exceedance* across the original and translated data means the model can reason equally in either language. Our results are summarized in Table [3](#A6.T3 "Table 3 ‣ F.2 Perplexity Evaluations for Translated Training Data ‣ Appendix F Results for Multiple Languages ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"). We find that perplexity is nearly identical for original data, but much higher for translated data in all languages. For *exceedance*, in English, we see that the scores are all around 80%. However, we see that for Japanese and Chinese these values are much lower, indicating that seeing the data in English for these languages does not lower the perplexity in the target language and that the model is not making use of information in the other language. For Chinese evaluations, we note that the perplexity is much higher than for other languages indicating that the translation system potentially causes higher perplexity and lower *exceedance*. However, we still note that for Japanese, the *exceedance* is lower and expect with better translation quality, the Chinese evaluations will be similar to Japanese.

(a) 1.3B Model

(b) 2.7B Model

Figure 14: Zero-shot QA performance for 1B and 3B models at varying data ratios during training. For all models the total amount of available data in the target language is 250M tokens.



|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Language | mC4-Train | mc4-Val | mC4-EN | mC4-EN Translated | FWE | FWE Translated | Original EX | Translated EX |
| German | 8.41 | 16.41 | 14.25 | 25.58 | 10.61 | 21.31 | 78.70 | 75.00 |
| French | 6.24 | 12.75 | 14.37 | 20.31 | 10.66 | 14.37 | 79.24 | 88.64 |
| Japanese | 6.52 | 11.21 | 14.56 | 25.56 | 10.64 | 23.89 | 80.41 | 56.97 |
| Chinese | 4.35 | 21.38 | 15.37 | 165.90 | 10.69 | 210.43 | 84.07 | 27.71 |

Table 3: Perplexity evaluations for mC4 English and FineWeb-EDU comparing original data and translated versions for 1B models trained with 250M tokens from the target language and FineWeb-EDU as the auxiliary dataset.

### F.3 Individual Eval Dataset Results

Results for individual evaluation datasets are shown for all languages in Tables [4](#A6.T4 "Table 4 ‣ F.3 Individual Eval Dataset Results ‣ Appendix F Results for Multiple Languages ‣ Training Bilingual LMs with Data Constraints in the Targeted Language")-[11](#A6.T11 "Table 11 ‣ F.3 Individual Eval Dataset Results ‣ Appendix F Results for Multiple Languages ‣ Training Bilingual LMs with Data Constraints in the Targeted Language").

| Model Name | ARC-C | ARC-E | HS | PIQA | SCIQ | WGrande | AVG |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base DE | 22.78±1.23plus-or-minus22.781.2322.78\pm 1.23 | 37.92±1.00plus-or-minus37.921.0037.92\pm 1.00 | 33.06±0.47plus-or-minus33.060.4733.06\pm 0.47 | 62.35±1.13plus-or-minus62.351.1362.35\pm 1.13 | 64.70±1.51plus-or-minus64.701.5164.70\pm 1.51 | 51.14±1.40plus-or-minus51.141.4051.14\pm 1.40 | 45.3345.3345.33 |
| Base EN | 25.94±1.28plus-or-minus25.941.2825.94\pm 1.28 | 48.15±1.03plus-or-minus48.151.0348.15\pm 1.03 | 48.50±0.50plus-or-minus48.500.5048.50\pm 0.50 | 71.55±1.05plus-or-minus71.551.0571.55\pm 1.05 | 73.10±1.40plus-or-minus73.101.4073.10\pm 1.40 | 52.01±1.40plus-or-minus52.011.4052.01\pm 1.40 | 53.2153.2153.21 |
| Base EN + OH/ELI5 Filter | 29.52±1.33plus-or-minus29.521.3329.52\pm 1.33 | 57.03±1.02plus-or-minus57.031.0257.03\pm 1.02 | 49.96±0.50plus-or-minus49.960.5049.96\pm 0.50 | 71.87±1.05plus-or-minus71.871.0571.87\pm 1.05 | 77.90±1.31plus-or-minus77.901.3177.90\pm 1.31 | 53.83±1.40plus-or-minus53.831.4053.83\pm 1.40 | 56.6956.6956.69 |
| ARC EN | 31.83±1.36plus-or-minus31.831.3631.83\pm 1.36 | 57.24±1.02plus-or-minus57.241.0257.24\pm 1.02 | 48.80±0.50plus-or-minus48.800.5048.80\pm 0.50 | 73.07±1.04plus-or-minus73.071.0473.07\pm 1.04 | 77.00±1.33plus-or-minus77.001.3377.00\pm 1.33 | 52.72±1.40plus-or-minus52.721.4052.72\pm 1.40 | 56.7856.7856.78 |
| SciQ+Inst | 29.10±1.33plus-or-minus29.101.3329.10\pm 1.33 | 55.22±1.02plus-or-minus55.221.0255.22\pm 1.02 | 48.34±0.50plus-or-minus48.340.5048.34\pm 0.50 | 71.71±1.05plus-or-minus71.711.0571.71\pm 1.05 | 78.50±1.30plus-or-minus78.501.3078.50\pm 1.30 | 53.04±1.40plus-or-minus53.041.4053.04\pm 1.40 | 55.9855.9855.98 |
| FineWeb EDU | 38.14±1.42plus-or-minus38.141.4238.14\pm 1.42 | 66.37±0.97plus-or-minus66.370.9766.37\pm 0.97 | 54.88±0.50plus-or-minus54.880.5054.88\pm 0.50 | 72.25±1.04plus-or-minus72.251.0472.25\pm 1.04 | 84.60±1.14plus-or-minus84.601.1484.60\pm 1.14 | 56.04±1.39plus-or-minus56.041.3956.04\pm 1.39 | 62.0562.0562.05 |
| Base DE | 27.44±1.32plus-or-minus27.441.3227.44\pm 1.32 | 38.81±1.03plus-or-minus38.811.0338.81\pm 1.03 | 39.53±0.51plus-or-minus39.530.5139.53\pm 0.51 | 63.17±1.13plus-or-minus63.171.1363.17\pm 1.13 | 67.68±1.52plus-or-minus67.681.5267.68\pm 1.52 | 52.62±1.45plus-or-minus52.621.4552.62\pm 1.45 | 48.2148.2148.21 |
| Base EN | 25.07±1.29plus-or-minus25.071.2925.07\pm 1.29 | 37.70±1.02plus-or-minus37.701.0237.70\pm 1.02 | 36.18±0.50plus-or-minus36.180.5036.18\pm 0.50 | 59.36±1.15plus-or-minus59.361.1559.36\pm 1.15 | 63.05±1.57plus-or-minus63.051.5763.05\pm 1.57 | 50.51±1.45plus-or-minus50.511.4550.51\pm 1.45 | 45.3145.3145.31 |
| Base EN + OH/ELI5 Filter | 25.77±1.30plus-or-minus25.771.3025.77\pm 1.30 | 40.00±1.03plus-or-minus40.001.0340.00\pm 1.03 | 35.94±0.50plus-or-minus35.940.5035.94\pm 0.50 | 59.25±1.15plus-or-minus59.251.1559.25\pm 1.15 | 65.05±1.55plus-or-minus65.051.5565.05\pm 1.55 | 51.18±1.45plus-or-minus51.181.4551.18\pm 1.45 | 46.2046.2046.20 |
| ARC EN | 26.91±1.32plus-or-minus26.911.3226.91\pm 1.32 | 39.34±1.03plus-or-minus39.341.0339.34\pm 1.03 | 35.77±0.50plus-or-minus35.770.5035.77\pm 0.50 | 58.65±1.15plus-or-minus58.651.1558.65\pm 1.15 | 67.58±1.52plus-or-minus67.581.5267.58\pm 1.52 | 52.11±1.45plus-or-minus52.111.4552.11\pm 1.45 | 46.7346.7346.73 |
| SciQ+Inst | 25.24±1.29plus-or-minus25.241.2925.24\pm 1.29 | 39.38±1.03plus-or-minus39.381.0339.38\pm 1.03 | 35.34±0.49plus-or-minus35.340.4935.34\pm 0.49 | 59.74±1.14plus-or-minus59.741.1459.74\pm 1.14 | 64.21±1.56plus-or-minus64.211.5664.21\pm 1.56 | 52.70±1.45plus-or-minus52.701.4552.70\pm 1.45 | 46.1046.1046.10 |
| FineWeb EDU | 26.91±1.32plus-or-minus26.911.3226.91\pm 1.32 | 42.39±1.04plus-or-minus42.391.0442.39\pm 1.04 | 37.40±0.50plus-or-minus37.400.5037.40\pm 0.50 | 60.45±1.14plus-or-minus60.451.1460.45\pm 1.14 | 65.37±1.54plus-or-minus65.371.5465.37\pm 1.54 | 50.42±1.45plus-or-minus50.421.4550.42\pm 1.45 | 47.1647.1647.16 |

Table 4: Evaluation of 111B parameter XL model on “General Understanding Tasks” focusing on general reasoning, language understanding, and science knowledge in English followed by translated German. Results show the length normalized accuracy for individual datasets and the average over all datasets for all datasets.



| Model Name | ARC-C | ARC-E | HS | PIQA | SCIQ | WGrande | AVG |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base FR | 24.66±1.26plus-or-minus24.661.2624.66\pm 1.26 | 36.78±0.99plus-or-minus36.780.9936.78\pm 0.99 | 32.89±0.47plus-or-minus32.890.4732.89\pm 0.47 | 60.66±1.14plus-or-minus60.661.1460.66\pm 1.14 | 63.90±1.52plus-or-minus63.901.5263.90\pm 1.52 | 50.51±1.41plus-or-minus50.511.4150.51\pm 1.41 | 44.9044.9044.90 |
| Base EN | 26.28±1.29plus-or-minus26.281.2926.28\pm 1.29 | 49.92±1.03plus-or-minus49.921.0349.92\pm 1.03 | 48.62±0.50plus-or-minus48.620.5048.62\pm 0.50 | 71.98±1.05plus-or-minus71.981.0571.98\pm 1.05 | 75.20±1.37plus-or-minus75.201.3775.20\pm 1.37 | 53.91±1.40plus-or-minus53.911.4053.91\pm 1.40 | 54.3254.3254.32 |
| Base EN + OH/ELI5 Filter | 29.52±1.33plus-or-minus29.521.3329.52\pm 1.33 | 56.44±1.02plus-or-minus56.441.0256.44\pm 1.02 | 49.92±0.50plus-or-minus49.920.5049.92\pm 0.50 | 72.74±1.04plus-or-minus72.741.0472.74\pm 1.04 | 79.70±1.27plus-or-minus79.701.2779.70\pm 1.27 | 54.06±1.40plus-or-minus54.061.4054.06\pm 1.40 | 57.0657.0657.06 |
| ARC EN | 29.69±1.34plus-or-minus29.691.3429.69\pm 1.34 | 57.28±1.02plus-or-minus57.281.0257.28\pm 1.02 | 48.96±0.50plus-or-minus48.960.5048.96\pm 0.50 | 72.91±1.04plus-or-minus72.911.0472.91\pm 1.04 | 78.90±1.29plus-or-minus78.901.2978.90\pm 1.29 | 53.12±1.40plus-or-minus53.121.4053.12\pm 1.40 | 56.8156.8156.81 |
| SciQ+Inst | 28.41±1.32plus-or-minus28.411.3228.41\pm 1.32 | 53.16±1.02plus-or-minus53.161.0253.16\pm 1.02 | 47.44±0.50plus-or-minus47.440.5047.44\pm 0.50 | 70.51±1.06plus-or-minus70.511.0670.51\pm 1.06 | 78.90±1.29plus-or-minus78.901.2978.90\pm 1.29 | 54.14±1.40plus-or-minus54.141.4054.14\pm 1.40 | 55.4355.4355.43 |
| FineWeb EDU | 36.69±1.41plus-or-minus36.691.4136.69\pm 1.41 | 65.40±0.98plus-or-minus65.400.9865.40\pm 0.98 | 54.81±0.50plus-or-minus54.810.5054.81\pm 0.50 | 72.74±1.04plus-or-minus72.741.0472.74\pm 1.04 | 82.50±1.20plus-or-minus82.501.2082.50\pm 1.20 | 55.41±1.40plus-or-minus55.411.4055.41\pm 1.40 | 61.2661.2661.26 |
| Base FR | 25.98±1.30plus-or-minus25.981.3025.98\pm 1.30 | 38.53±1.02plus-or-minus38.531.0238.53\pm 1.02 | 41.71±0.51plus-or-minus41.710.5141.71\pm 0.51 | 64.25±1.12plus-or-minus64.251.1264.25\pm 1.12 | 62.12±1.57plus-or-minus62.121.5762.12\pm 1.57 | 53.25±1.43plus-or-minus53.251.4353.25\pm 1.43 | 47.6447.6447.64 |
| Base EN | 23.54±1.25plus-or-minus23.541.2523.54\pm 1.25 | 37.03±1.01plus-or-minus37.031.0137.03\pm 1.01 | 37.66±0.50plus-or-minus37.660.5037.66\pm 0.50 | 58.98±1.15plus-or-minus58.981.1558.98\pm 1.15 | 60.44±1.58plus-or-minus60.441.5860.44\pm 1.58 | 49.05±1.43plus-or-minus49.051.4349.05\pm 1.43 | 44.4544.4544.45 |
| Base EN + OH/ELI5 Filter | 26.50±1.30plus-or-minus26.501.3026.50\pm 1.30 | 38.93±1.02plus-or-minus38.931.0238.93\pm 1.02 | 38.85±0.50plus-or-minus38.850.5038.85\pm 0.50 | 60.50±1.14plus-or-minus60.501.1460.50\pm 1.14 | 66.11±1.53plus-or-minus66.111.5366.11\pm 1.53 | 50.37±1.43plus-or-minus50.371.4350.37\pm 1.43 | 46.8846.8846.88 |
| ARC EN | 25.54±1.29plus-or-minus25.541.2925.54\pm 1.29 | 39.32±1.03plus-or-minus39.321.0339.32\pm 1.03 | 38.17±0.50plus-or-minus38.170.5038.17\pm 0.50 | 59.41±1.15plus-or-minus59.411.1559.41\pm 1.15 | 64.22±1.55plus-or-minus64.221.5564.22\pm 1.55 | 50.29±1.44plus-or-minus50.291.4450.29\pm 1.44 | 46.1646.1646.16 |
| SciQ+Inst | 25.72±1.29plus-or-minus25.721.2925.72\pm 1.29 | 38.71±1.02plus-or-minus38.711.0238.71\pm 1.02 | 37.61±0.50plus-or-minus37.610.5037.61\pm 0.50 | 59.03±1.15plus-or-minus59.031.1559.03\pm 1.15 | 63.06±1.56plus-or-minus63.061.5663.06\pm 1.56 | 49.55±1.43plus-or-minus49.551.4349.55\pm 1.43 | 45.6145.6145.61 |
| FineWeb EDU | 27.38±1.32plus-or-minus27.381.3227.38\pm 1.32 | 40.51±1.03plus-or-minus40.511.0340.51\pm 1.03 | 40.67±0.51plus-or-minus40.670.5140.67\pm 0.51 | 60.94±1.14plus-or-minus60.941.1460.94\pm 1.14 | 64.85±1.55plus-or-minus64.851.5564.85\pm 1.55 | 50.78±1.43plus-or-minus50.781.4350.78\pm 1.43 | 47.5247.5247.52 |

Table 5: Evaluation of 111B parameter XL model on “General Understanding Tasks” focusing on general reasoning, language understanding, and science knowledge in English followed by French. Results show the length normalized accuracy for individual datasets and the average over all datasets for all datasets.



| Model Name | ARC-C | ARC-E | HS | PIQA | SCIQ | WGrande | AVG |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base ES | 23.29±1.24plus-or-minus23.291.2423.29\pm 1.24 | 38.55±1.00plus-or-minus38.551.0038.55\pm 1.00 | 33.86±0.47plus-or-minus33.860.4733.86\pm 0.47 | 60.83±1.14plus-or-minus60.831.1460.83\pm 1.14 | 67.90±1.48plus-or-minus67.901.4867.90\pm 1.48 | 51.07±1.40plus-or-minus51.071.4051.07\pm 1.40 | 45.9245.9245.92 |
| Base EN | 26.19±1.28plus-or-minus26.191.2826.19\pm 1.28 | 48.95±1.03plus-or-minus48.951.0348.95\pm 1.03 | 48.16±0.50plus-or-minus48.160.5048.16\pm 0.50 | 71.11±1.06plus-or-minus71.111.0671.11\pm 1.06 | 74.30±1.38plus-or-minus74.301.3874.30\pm 1.38 | 52.01±1.40plus-or-minus52.011.4052.01\pm 1.40 | 53.4553.4553.45 |
| Base EN + OH/ELI5 Filter | 29.86±1.34plus-or-minus29.861.3429.86\pm 1.34 | 58.08±1.01plus-or-minus58.081.0158.08\pm 1.01 | 50.55±0.50plus-or-minus50.550.5050.55\pm 0.50 | 72.36±1.04plus-or-minus72.361.0472.36\pm 1.04 | 79.40±1.28plus-or-minus79.401.2879.40\pm 1.28 | 54.62±1.40plus-or-minus54.621.4054.62\pm 1.40 | 57.4857.4857.48 |
| ARC EN | 30.89±1.35plus-or-minus30.891.3530.89\pm 1.35 | 58.71±1.01plus-or-minus58.711.0158.71\pm 1.01 | 49.41±0.50plus-or-minus49.410.5049.41\pm 0.50 | 73.45±1.03plus-or-minus73.451.0373.45\pm 1.03 | 78.50±1.30plus-or-minus78.501.3078.50\pm 1.30 | 54.06±1.40plus-or-minus54.061.4054.06\pm 1.40 | 57.5057.5057.50 |
| SciQ+Inst | 29.44±1.33plus-or-minus29.441.3329.44\pm 1.33 | 57.49±1.01plus-or-minus57.491.0157.49\pm 1.01 | 48.89±0.50plus-or-minus48.890.5048.89\pm 0.50 | 70.57±1.06plus-or-minus70.571.0670.57\pm 1.06 | 79.70±1.27plus-or-minus79.701.2779.70\pm 1.27 | 54.22±1.40plus-or-minus54.221.4054.22\pm 1.40 | 56.7256.7256.72 |
| FWE | 36.60±1.41plus-or-minus36.601.4136.60\pm 1.41 | 64.86±0.98plus-or-minus64.860.9864.86\pm 0.98 | 54.95±0.50plus-or-minus54.950.5054.95\pm 0.50 | 71.87±1.05plus-or-minus71.871.0571.87\pm 1.05 | 81.50±1.23plus-or-minus81.501.2381.50\pm 1.23 | 57.85±1.39plus-or-minus57.851.3957.85\pm 1.39 | 61.2761.2761.27 |
| Base ES | 27.99±1.33plus-or-minus27.991.3327.99\pm 1.33 | 44.12±1.04plus-or-minus44.121.0444.12\pm 1.04 | 43.64±0.51plus-or-minus43.640.5143.64\pm 0.51 | 65.94±1.11plus-or-minus65.941.1165.94\pm 1.11 | 69.72±1.49plus-or-minus69.721.4969.72\pm 1.49 | 52.70±1.42plus-or-minus52.701.4252.70\pm 1.42 | 50.6950.6950.69 |
| Base EN | 25.28±1.28plus-or-minus25.281.2825.28\pm 1.28 | 38.71±1.02plus-or-minus38.711.0238.71\pm 1.02 | 39.32±0.50plus-or-minus39.320.5039.32\pm 0.50 | 60.94±1.14plus-or-minus60.941.1460.94\pm 1.14 | 65.83±1.54plus-or-minus65.831.5465.83\pm 1.54 | 47.13±1.42plus-or-minus47.131.4247.13\pm 1.42 | 46.2046.2046.20 |
| Base EN + OH/ELI5 Filter | 24.41±1.27plus-or-minus24.411.2724.41\pm 1.27 | 35.31±1.00plus-or-minus35.311.0035.31\pm 1.00 | 33.01±0.49plus-or-minus33.010.4933.01\pm 0.49 | 56.64±1.16plus-or-minus56.641.1656.64\pm 1.16 | 64.35±1.55plus-or-minus64.351.5564.35\pm 1.55 | 50.36±1.42plus-or-minus50.361.4250.36\pm 1.42 | 44.0144.0144.01 |
| ARC EN | 25.89±1.29plus-or-minus25.891.2925.89\pm 1.29 | 42.01±1.04plus-or-minus42.011.0442.01\pm 1.04 | 39.49±0.50plus-or-minus39.490.5039.49\pm 0.50 | 62.02±1.13plus-or-minus62.021.1362.02\pm 1.13 | 66.25±1.53plus-or-minus66.251.5366.25\pm 1.53 | 52.54±1.42plus-or-minus52.541.4252.54\pm 1.42 | 48.0348.0348.03 |
| SciQ+Inst | 25.89±1.29plus-or-minus25.891.2925.89\pm 1.29 | 41.08±1.03plus-or-minus41.081.0341.08\pm 1.03 | 39.35±0.50plus-or-minus39.350.5039.35\pm 0.50 | 61.37±1.14plus-or-minus61.371.1461.37\pm 1.14 | 67.93±1.51plus-or-minus67.931.5167.93\pm 1.51 | 51.82±1.42plus-or-minus51.821.4251.82\pm 1.42 | 47.9147.9147.91 |
| FWE | 27.99±1.33plus-or-minus27.991.3327.99\pm 1.33 | 43.24±1.04plus-or-minus43.241.0443.24\pm 1.04 | 42.05±0.51plus-or-minus42.050.5142.05\pm 0.51 | 62.19±1.13plus-or-minus62.191.1362.19\pm 1.13 | 70.45±1.48plus-or-minus70.451.4870.45\pm 1.48 | 51.17±1.42plus-or-minus51.171.4251.17\pm 1.42 | 49.5149.5149.51 |

Table 6: Evaluation of 111B parameter XL model on “General Understanding Tasks” focusing on general reasoning, language understanding, and science knowledge in English followed by Spanish. Results show the length normalized accuracy for individual datasets and the average over all datasets for all datasets.



| Model Name | ARC-C | ARC-E | HS | PIQA | SCIQ | WGrande | AVG |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base PT | 23.98±1.25plus-or-minus23.981.2523.98\pm 1.25 | 39.14±1.00plus-or-minus39.141.0039.14\pm 1.00 | 33.26±0.47plus-or-minus33.260.4733.26\pm 0.47 | 60.99±1.14plus-or-minus60.991.1460.99\pm 1.14 | 66.30±1.50plus-or-minus66.301.5066.30\pm 1.50 | 52.01±1.40plus-or-minus52.011.4052.01\pm 1.40 | 45.9545.9545.95 |
| Base EN | 25.34±1.27plus-or-minus25.341.2725.34\pm 1.27 | 48.02±1.03plus-or-minus48.021.0348.02\pm 1.03 | 47.19±0.50plus-or-minus47.190.5047.19\pm 0.50 | 71.60±1.05plus-or-minus71.601.0571.60\pm 1.05 | 74.00±1.39plus-or-minus74.001.3974.00\pm 1.39 | 50.04±1.41plus-or-minus50.041.4150.04\pm 1.41 | 52.7052.7052.70 |
| Base EN + OH/ELI5 Filter | 29.44±1.33plus-or-minus29.441.3329.44\pm 1.33 | 53.79±1.02plus-or-minus53.791.0253.79\pm 1.02 | 49.53±0.50plus-or-minus49.530.5049.53\pm 0.50 | 71.49±1.05plus-or-minus71.491.0571.49\pm 1.05 | 80.30±1.26plus-or-minus80.301.2680.30\pm 1.26 | 55.33±1.40plus-or-minus55.331.4055.33\pm 1.40 | 56.6556.6556.65 |
| ARC EN | 30.72±1.35plus-or-minus30.721.3530.72\pm 1.35 | 57.83±1.01plus-or-minus57.831.0157.83\pm 1.01 | 48.79±0.50plus-or-minus48.790.5048.79\pm 0.50 | 73.01±1.04plus-or-minus73.011.0473.01\pm 1.04 | 78.30±1.30plus-or-minus78.301.3078.30\pm 1.30 | 52.64±1.40plus-or-minus52.641.4052.64\pm 1.40 | 56.8856.8856.88 |
| SciQ+Inst | 29.35±1.33plus-or-minus29.351.3329.35\pm 1.33 | 57.45±1.01plus-or-minus57.451.0157.45\pm 1.01 | 48.26±0.50plus-or-minus48.260.5048.26\pm 0.50 | 71.55±1.05plus-or-minus71.551.0571.55\pm 1.05 | 78.80±1.29plus-or-minus78.801.2978.80\pm 1.29 | 53.99±1.40plus-or-minus53.991.4053.99\pm 1.40 | 56.5656.5656.56 |
| FWE | 35.07±1.39plus-or-minus35.071.3935.07\pm 1.39 | 64.39±0.98plus-or-minus64.390.9864.39\pm 0.98 | 54.81±0.50plus-or-minus54.810.5054.81\pm 0.50 | 72.69±1.04plus-or-minus72.691.0472.69\pm 1.04 | 82.10±1.21plus-or-minus82.101.2182.10\pm 1.21 | 57.62±1.39plus-or-minus57.621.3957.62\pm 1.39 | 61.1161.1161.11 |
| Base PT | 30.25±1.36plus-or-minus30.251.3630.25\pm 1.36 | 43.55±1.04plus-or-minus43.551.0443.55\pm 1.04 | 42.36±0.51plus-or-minus42.360.5142.36\pm 0.51 | 64.80±1.11plus-or-minus64.801.1164.80\pm 1.11 | 69.88±1.49plus-or-minus69.881.4969.88\pm 1.49 | 52.11±1.42plus-or-minus52.111.4252.11\pm 1.42 | 50.4950.4950.49 |
| Base EN | 24.06±1.26plus-or-minus24.061.2624.06\pm 1.26 | 38.04±1.02plus-or-minus38.041.0238.04\pm 1.02 | 36.04±0.50plus-or-minus36.040.5036.04\pm 0.50 | 59.68±1.14plus-or-minus59.681.1459.68\pm 1.14 | 59.18±1.59plus-or-minus59.181.5959.18\pm 1.59 | 51.06±1.42plus-or-minus51.061.4251.06\pm 1.42 | 44.6844.6844.68 |
| Base EN + OH/ELI5 Filter | 25.37±1.29plus-or-minus25.371.2925.37\pm 1.29 | 39.59±1.03plus-or-minus39.591.0339.59\pm 1.03 | 37.15±0.50plus-or-minus37.150.5037.15\pm 0.50 | 58.71±1.15plus-or-minus58.711.1558.71\pm 1.15 | 66.11±1.53plus-or-minus66.111.5366.11\pm 1.53 | 48.70±1.42plus-or-minus48.701.4248.70\pm 1.42 | 45.9445.9445.94 |
| ARC EN | 27.90±1.32plus-or-minus27.901.3227.90\pm 1.32 | 38.79±1.02plus-or-minus38.791.0238.79\pm 1.02 | 36.87±0.50plus-or-minus36.870.5036.87\pm 0.50 | 60.07±1.14plus-or-minus60.071.1460.07\pm 1.14 | 64.32±1.55plus-or-minus64.321.5564.32\pm 1.55 | 49.35±1.42plus-or-minus49.351.4249.35\pm 1.42 | 46.2246.2246.22 |
| SciQ+Inst | 27.03±1.31plus-or-minus27.031.3127.03\pm 1.31 | 40.38±1.03plus-or-minus40.381.0340.38\pm 1.03 | 37.72±0.50plus-or-minus37.720.5037.72\pm 0.50 | 60.72±1.14plus-or-minus60.721.1460.72\pm 1.14 | 66.00±1.54plus-or-minus66.001.5466.00\pm 1.54 | 50.73±1.42plus-or-minus50.731.4250.73\pm 1.42 | 47.1047.1047.10 |
| FWE | 28.86±1.34plus-or-minus28.861.3428.86\pm 1.34 | 43.11±1.04plus-or-minus43.111.0443.11\pm 1.04 | 39.81±0.51plus-or-minus39.810.5139.81\pm 0.51 | 60.34±1.14plus-or-minus60.341.1460.34\pm 1.14 | 66.84±1.53plus-or-minus66.841.5366.84\pm 1.53 | 50.57±1.42plus-or-minus50.571.4250.57\pm 1.42 | 48.2548.2548.25 |

Table 7: Evaluation of 111B parameter XL model on “General Understanding Tasks” focusing on general reasoning, language understanding, and science knowledge in English followed by Portuguese. Results show the length normalized accuracy for individual datasets and the average over all datasets for all datasets.



| Model Name | ARC-C | ARC-E | HS | PIQA | SCIQ | WGrande | AVG |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base IT | 21.59±1.20plus-or-minus21.591.2021.59\pm 1.20 | 35.77±0.98plus-or-minus35.770.9835.77\pm 0.98 | 32.54±0.47plus-or-minus32.540.4732.54\pm 0.47 | 59.09±1.15plus-or-minus59.091.1559.09\pm 1.15 | 63.70±1.52plus-or-minus63.701.5263.70\pm 1.52 | 51.07±1.40plus-or-minus51.071.4051.07\pm 1.40 | 43.9643.9643.96 |
| Base EN | 26.11±1.28plus-or-minus26.111.2826.11\pm 1.28 | 48.53±1.03plus-or-minus48.531.0348.53\pm 1.03 | 47.16±0.50plus-or-minus47.160.5047.16\pm 0.50 | 71.71±1.05plus-or-minus71.711.0571.71\pm 1.05 | 73.50±1.40plus-or-minus73.501.4073.50\pm 1.40 | 53.35±1.40plus-or-minus53.351.4053.35\pm 1.40 | 53.3953.3953.39 |
| Base EN + OH/ELI5 Filter | 31.14±1.35plus-or-minus31.141.3531.14\pm 1.35 | 57.37±1.01plus-or-minus57.371.0157.37\pm 1.01 | 50.89±0.50plus-or-minus50.890.5050.89\pm 0.50 | 72.42±1.04plus-or-minus72.421.0472.42\pm 1.04 | 79.50±1.28plus-or-minus79.501.2879.50\pm 1.28 | 53.12±1.40plus-or-minus53.121.4053.12\pm 1.40 | 57.4057.4057.40 |
| ARC EN | 30.46±1.34plus-or-minus30.461.3430.46\pm 1.34 | 56.06±1.02plus-or-minus56.061.0256.06\pm 1.02 | 48.78±0.50plus-or-minus48.780.5048.78\pm 0.50 | 73.01±1.04plus-or-minus73.011.0473.01\pm 1.04 | 75.90±1.35plus-or-minus75.901.3575.90\pm 1.35 | 52.57±1.40plus-or-minus52.571.4052.57\pm 1.40 | 56.1356.1356.13 |
| SciQ+Inst | 30.38±1.34plus-or-minus30.381.3430.38\pm 1.34 | 56.78±1.02plus-or-minus56.781.0256.78\pm 1.02 | 48.49±0.50plus-or-minus48.490.5048.49\pm 0.50 | 71.27±1.06plus-or-minus71.271.0671.27\pm 1.06 | 80.20±1.26plus-or-minus80.201.2680.20\pm 1.26 | 54.85±1.40plus-or-minus54.851.4054.85\pm 1.40 | 56.9956.9956.99 |
| FWE | 37.03±1.41plus-or-minus37.031.4137.03\pm 1.41 | 65.61±0.97plus-or-minus65.610.9765.61\pm 0.97 | 54.91±0.50plus-or-minus54.910.5054.91\pm 0.50 | 72.74±1.04plus-or-minus72.741.0472.74\pm 1.04 | 84.20±1.15plus-or-minus84.201.1584.20\pm 1.15 | 54.85±1.40plus-or-minus54.851.4054.85\pm 1.40 | 61.5661.5661.56 |
| Base IT | 26.33±1.30plus-or-minus26.331.3026.33\pm 1.30 | 40.38±1.03plus-or-minus40.381.0340.38\pm 1.03 | 39.89±0.51plus-or-minus39.890.5139.89\pm 0.51 | 64.74±1.11plus-or-minus64.741.1164.74\pm 1.11 | 61.76±1.58plus-or-minus61.761.5861.76\pm 1.58 | 51.42±1.42plus-or-minus51.421.4251.42\pm 1.42 | 47.4247.4247.42 |
| Base EN | 25.81±1.29plus-or-minus25.811.2925.81\pm 1.29 | 36.02±1.01plus-or-minus36.021.0136.02\pm 1.01 | 35.29±0.50plus-or-minus35.290.5035.29\pm 0.50 | 58.65±1.15plus-or-minus58.651.1558.65\pm 1.15 | 58.82±1.60plus-or-minus58.821.6058.82\pm 1.60 | 52.23±1.42plus-or-minus52.231.4252.23\pm 1.42 | 44.4744.4744.47 |
| Base EN + OH/ELI5 Filter | 24.93±1.28plus-or-minus24.931.2824.93\pm 1.28 | 32.01±0.98plus-or-minus32.010.9832.01\pm 0.98 | 31.76±0.49plus-or-minus31.760.4931.76\pm 0.49 | 54.95±1.16plus-or-minus54.951.1654.95\pm 1.16 | 61.97±1.57plus-or-minus61.971.5761.97\pm 1.57 | 53.20±1.42plus-or-minus53.201.4253.20\pm 1.42 | 43.1443.1443.14 |
| ARC EN | 25.72±1.29plus-or-minus25.721.2925.72\pm 1.29 | 39.15±1.02plus-or-minus39.151.0239.15\pm 1.02 | 36.18±0.50plus-or-minus36.180.5036.18\pm 0.50 | 60.55±1.14plus-or-minus60.551.1460.55\pm 1.14 | 63.24±1.56plus-or-minus63.241.5663.24\pm 1.56 | 51.34±1.42plus-or-minus51.341.4251.34\pm 1.42 | 46.0346.0346.03 |
| SciQ+Inst | 26.07±1.30plus-or-minus26.071.3026.07\pm 1.30 | 40.03±1.03plus-or-minus40.031.0340.03\pm 1.03 | 36.67±0.50plus-or-minus36.670.5036.67\pm 0.50 | 58.76±1.15plus-or-minus58.761.1558.76\pm 1.15 | 62.29±1.57plus-or-minus62.291.5762.29\pm 1.57 | 50.93±1.42plus-or-minus50.931.4250.93\pm 1.42 | 45.7945.7945.79 |
| FWE | 29.90±1.35plus-or-minus29.901.3529.90\pm 1.35 | 41.04±1.03plus-or-minus41.041.0341.04\pm 1.03 | 38.24±0.51plus-or-minus38.240.5138.24\pm 0.51 | 62.68±1.13plus-or-minus62.681.1362.68\pm 1.13 | 61.03±1.58plus-or-minus61.031.5861.03\pm 1.58 | 51.18±1.42plus-or-minus51.181.4251.18\pm 1.42 | 47.3447.3447.34 |

Table 8: Evaluation of 111B parameter XL model on “General Understanding Tasks” focusing on general reasoning, language understanding, and science knowledge in English followed by Italian. Results show the length normalized accuracy for individual datasets and the average over all datasets for all datasets.



| Model Name | ARC-C | ARC-E | HS | PIQA | SCIQ | WGrande | AVG |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base KO | 22.10±1.21plus-or-minus22.101.2122.10\pm 1.21 | 37.46±0.99plus-or-minus37.460.9937.46\pm 0.99 | 28.99±0.45plus-or-minus28.990.4528.99\pm 0.45 | 59.19±1.15plus-or-minus59.191.1559.19\pm 1.15 | 60.70±1.55plus-or-minus60.701.5560.70\pm 1.55 | 51.62±1.40plus-or-minus51.621.4051.62\pm 1.40 | 43.3443.3443.34 |
| Base EN | 26.02±1.28plus-or-minus26.021.2826.02\pm 1.28 | 47.47±1.02plus-or-minus47.471.0247.47\pm 1.02 | 47.71±0.50plus-or-minus47.710.5047.71\pm 0.50 | 71.11±1.06plus-or-minus71.111.0671.11\pm 1.06 | 75.00±1.37plus-or-minus75.001.3775.00\pm 1.37 | 52.72±1.40plus-or-minus52.721.4052.72\pm 1.40 | 53.3453.3453.34 |
| Base EN + OH/ELI5 Filter | 30.03±1.34plus-or-minus30.031.3430.03\pm 1.34 | 56.69±1.02plus-or-minus56.691.0256.69\pm 1.02 | 50.38±0.50plus-or-minus50.380.5050.38\pm 0.50 | 71.76±1.05plus-or-minus71.761.0571.76\pm 1.05 | 78.90±1.29plus-or-minus78.901.2978.90\pm 1.29 | 53.20±1.40plus-or-minus53.201.4053.20\pm 1.40 | 56.8356.8356.83 |
| ARC EN | 30.03±1.34plus-or-minus30.031.3430.03\pm 1.34 | 57.37±1.01plus-or-minus57.371.0157.37\pm 1.01 | 49.41±0.50plus-or-minus49.410.5049.41\pm 0.50 | 73.78±1.03plus-or-minus73.781.0373.78\pm 1.03 | 78.20±1.31plus-or-minus78.201.3178.20\pm 1.31 | 53.12±1.40plus-or-minus53.121.4053.12\pm 1.40 | 56.9856.9856.98 |
| SciQ+Inst | 30.97±1.35plus-or-minus30.971.3530.97\pm 1.35 | 58.54±1.01plus-or-minus58.541.0158.54\pm 1.01 | 48.16±0.50plus-or-minus48.160.5048.16\pm 0.50 | 71.49±1.05plus-or-minus71.491.0571.49\pm 1.05 | 80.30±1.26plus-or-minus80.301.2680.30\pm 1.26 | 52.33±1.40plus-or-minus52.331.4052.33\pm 1.40 | 56.9756.9756.97 |
| FWE | 36.01±1.40plus-or-minus36.011.4036.01\pm 1.40 | 63.59±0.99plus-or-minus63.590.9963.59\pm 0.99 | 54.41±0.50plus-or-minus54.410.5054.41\pm 0.50 | 72.96±1.04plus-or-minus72.961.0472.96\pm 1.04 | 80.10±1.26plus-or-minus80.101.2680.10\pm 1.26 | 55.56±1.40plus-or-minus55.561.4055.56\pm 1.40 | 60.4460.4460.44 |
| Base KO | 28.07±1.33plus-or-minus28.071.3328.07\pm 1.33 | 42.18±1.04plus-or-minus42.181.0442.18\pm 1.04 | 35.48±0.48plus-or-minus35.480.4835.48\pm 0.48 | 60.66±1.14plus-or-minus60.661.1460.66\pm 1.14 | 71.64±1.47plus-or-minus71.641.4771.64\pm 1.47 | 49.28±1.46plus-or-minus49.281.4649.28\pm 1.46 | 47.8947.8947.89 |
| Base EN | 22.76±1.24plus-or-minus22.761.2422.76\pm 1.24 | 33.82±0.99plus-or-minus33.820.9933.82\pm 0.99 | 28.88±0.45plus-or-minus28.880.4528.88\pm 0.45 | 55.22±1.16plus-or-minus55.221.1655.22\pm 1.16 | 56.08±1.62plus-or-minus56.081.6256.08\pm 1.62 | 51.23±1.46plus-or-minus51.231.4651.23\pm 1.46 | 41.3341.3341.33 |
| Base EN + OH/ELI5 Filter | 24.85±1.28plus-or-minus24.851.2824.85\pm 1.28 | 34.13±1.00plus-or-minus34.131.0034.13\pm 1.00 | 29.35±0.45plus-or-minus29.350.4529.35\pm 0.45 | 55.55±1.16plus-or-minus55.551.1655.55\pm 1.16 | 63.92±1.56plus-or-minus63.921.5663.92\pm 1.56 | 52.34±1.46plus-or-minus52.341.4652.34\pm 1.46 | 43.3543.3543.35 |
| ARC EN | 25.28±1.28plus-or-minus25.281.2825.28\pm 1.28 | 33.73±0.99plus-or-minus33.730.9933.73\pm 0.99 | 29.33±0.45plus-or-minus29.330.4529.33\pm 0.45 | 53.86±1.16plus-or-minus53.861.1653.86\pm 1.16 | 64.44±1.56plus-or-minus64.441.5664.44\pm 1.56 | 51.15±1.46plus-or-minus51.151.4651.15\pm 1.46 | 42.9742.9742.97 |
| SciQ+Inst | 23.19±1.25plus-or-minus23.191.2523.19\pm 1.25 | 34.26±1.00plus-or-minus34.261.0034.26\pm 1.00 | 30.01±0.46plus-or-minus30.010.4630.01\pm 0.46 | 55.60±1.16plus-or-minus55.601.1655.60\pm 1.16 | 64.02±1.56plus-or-minus64.021.5664.02\pm 1.56 | 50.04±1.46plus-or-minus50.041.4650.04\pm 1.46 | 42.8642.8642.86 |
| FWE | 22.93±1.24plus-or-minus22.931.2422.93\pm 1.24 | 32.32±0.98plus-or-minus32.320.9832.32\pm 0.98 | 29.09±0.45plus-or-minus29.090.4529.09\pm 0.45 | 53.26±1.16plus-or-minus53.261.1653.26\pm 1.16 | 64.76±1.55plus-or-minus64.761.5564.76\pm 1.55 | 48.43±1.46plus-or-minus48.431.4648.43\pm 1.46 | 41.8041.8041.80 |

Table 9: Evaluation of 111B parameter XL model on “General Understanding Tasks” focusing on general reasoning, language understanding, and science knowledge in English followed by Korean. Results show the length normalized accuracy for individual datasets and the average over all datasets for all datasets.



| Model Name | ARC-C | ARC-E | HS | PIQA | SCIQ | WGrande | AVG |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base JA | 23.46±1.24plus-or-minus23.461.2423.46\pm 1.24 | 37.42±0.99plus-or-minus37.420.9937.42\pm 0.99 | 29.07±0.45plus-or-minus29.070.4529.07\pm 0.45 | 59.14±1.15plus-or-minus59.141.1559.14\pm 1.15 | 68.10±1.47plus-or-minus68.101.4768.10\pm 1.47 | 50.28±1.41plus-or-minus50.281.4150.28\pm 1.41 | 44.5844.5844.58 |
| Base EN | 26.19±1.28plus-or-minus26.191.2826.19\pm 1.28 | 49.24±1.03plus-or-minus49.241.0349.24\pm 1.03 | 48.69±0.50plus-or-minus48.690.5048.69\pm 0.50 | 72.63±1.04plus-or-minus72.631.0472.63\pm 1.04 | 74.40±1.38plus-or-minus74.401.3874.40\pm 1.38 | 53.28±1.40plus-or-minus53.281.4053.28\pm 1.40 | 54.0754.0754.07 |
| Base EN + OH/ELI5 Filter | 29.78±1.34plus-or-minus29.781.3429.78\pm 1.34 | 55.30±1.02plus-or-minus55.301.0255.30\pm 1.02 | 50.26±0.50plus-or-minus50.260.5050.26\pm 0.50 | 72.14±1.05plus-or-minus72.141.0572.14\pm 1.05 | 78.90±1.29plus-or-minus78.901.2978.90\pm 1.29 | 52.72±1.40plus-or-minus52.721.4052.72\pm 1.40 | 56.5256.5256.52 |
| ARC EN | 29.35±1.33plus-or-minus29.351.3329.35\pm 1.33 | 56.94±1.02plus-or-minus56.941.0256.94\pm 1.02 | 49.05±0.50plus-or-minus49.050.5049.05\pm 0.50 | 73.78±1.03plus-or-minus73.781.0373.78\pm 1.03 | 79.80±1.27plus-or-minus79.801.2779.80\pm 1.27 | 54.22±1.40plus-or-minus54.221.4054.22\pm 1.40 | 57.1957.1957.19 |
| SciQ+Inst | 30.38±1.34plus-or-minus30.381.3430.38\pm 1.34 | 56.14±1.02plus-or-minus56.141.0256.14\pm 1.02 | 48.32±0.50plus-or-minus48.320.5048.32\pm 0.50 | 72.42±1.04plus-or-minus72.421.0472.42\pm 1.04 | 78.60±1.30plus-or-minus78.601.3078.60\pm 1.30 | 52.64±1.40plus-or-minus52.641.4052.64\pm 1.40 | 56.4256.4256.42 |
| FineWeb EDU | 34.73±1.39plus-or-minus34.731.3934.73\pm 1.39 | 63.59±0.99plus-or-minus63.590.9963.59\pm 0.99 | 54.80±0.50plus-or-minus54.800.5054.80\pm 0.50 | 72.91±1.04plus-or-minus72.911.0472.91\pm 1.04 | 82.20±1.21plus-or-minus82.201.2182.20\pm 1.21 | 56.43±1.39plus-or-minus56.431.3956.43\pm 1.39 | 60.7860.7860.78 |
| Base JA | 25.28±1.28plus-or-minus25.281.2825.28\pm 1.28 | 40.69±1.03plus-or-minus40.691.0340.69\pm 1.03 | 35.11±0.48plus-or-minus35.110.4835.11\pm 0.48 | 58.98±1.15plus-or-minus58.981.1558.98\pm 1.15 | 69.98±1.51plus-or-minus69.981.5169.98\pm 1.51 | 51.19±1.51plus-or-minus51.191.5151.19\pm 1.51 | 46.8746.8746.87 |
| Base EN | 25.28±1.28plus-or-minus25.281.2825.28\pm 1.28 | 36.15±1.01plus-or-minus36.151.0136.15\pm 1.01 | 31.71±0.46plus-or-minus31.710.4631.71\pm 0.46 | 56.26±1.16plus-or-minus56.261.1656.26\pm 1.16 | 66.41±1.55plus-or-minus66.411.5566.41\pm 1.55 | 50.82±1.51plus-or-minus50.821.5150.82\pm 1.51 | 44.4444.4444.44 |
| Base EN + OH/ELI5 Filter | 26.50±1.30plus-or-minus26.501.3026.50\pm 1.30 | 36.99±1.01plus-or-minus36.991.0136.99\pm 1.01 | 31.11±0.46plus-or-minus31.110.4631.11\pm 0.46 | 56.96±1.16plus-or-minus56.961.1656.96\pm 1.16 | 67.60±1.54plus-or-minus67.601.5467.60\pm 1.54 | 50.18±1.51plus-or-minus50.181.5150.18\pm 1.51 | 44.8944.8944.89 |
| ARC EN | 27.55±1.32plus-or-minus27.551.3227.55\pm 1.32 | 37.43±1.02plus-or-minus37.431.0237.43\pm 1.02 | 31.56±0.46plus-or-minus31.560.4631.56\pm 0.46 | 57.29±1.15plus-or-minus57.291.1557.29\pm 1.15 | 65.55±1.56plus-or-minus65.551.5665.55\pm 1.56 | 48.72±1.51plus-or-minus48.721.5148.72\pm 1.51 | 44.6844.6844.68 |
| SciQ+Inst | 27.38±1.32plus-or-minus27.381.3227.38\pm 1.32 | 36.42±1.01plus-or-minus36.421.0136.42\pm 1.01 | 31.44±0.46plus-or-minus31.440.4631.44\pm 0.46 | 56.58±1.16plus-or-minus56.581.1656.58\pm 1.16 | 67.28±1.54plus-or-minus67.281.5467.28\pm 1.54 | 50.36±1.51plus-or-minus50.361.5150.36\pm 1.51 | 44.9144.9144.91 |
| FineWeb EDU | 25.28±1.28plus-or-minus25.281.2825.28\pm 1.28 | 33.73±0.99plus-or-minus33.730.9933.73\pm 0.99 | 30.91±0.46plus-or-minus30.910.4630.91\pm 0.46 | 56.09±1.16plus-or-minus56.091.1656.09\pm 1.16 | 63.28±1.58plus-or-minus63.281.5863.28\pm 1.58 | 47.08±1.51plus-or-minus47.081.5147.08\pm 1.51 | 42.7342.7342.73 |

Table 10: Evaluation of 111B parameter XL model on “General Understanding Tasks” focusing on general reasoning, language understanding, and science knowledge in English followed by translated Japanese. Results show the length normalized accuracy for individual datasets and the average over all datasets for all datasets.



| Model Name | ARC-C | ARC-E | HS | PIQA | SCIQ | WGrande | AVG |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base ZH | 21.16±1.19plus-or-minus21.161.1921.16\pm 1.19 | 33.16±0.97plus-or-minus33.160.9733.16\pm 0.97 | 27.63±0.45plus-or-minus27.630.4527.63\pm 0.45 | 55.98±1.16plus-or-minus55.981.1655.98\pm 1.16 | 56.60±1.57plus-or-minus56.601.5756.60\pm 1.57 | 49.09±1.41plus-or-minus49.091.4149.09\pm 1.41 | 40.6140.6140.61 |
| Base EN | 25.85±1.28plus-or-minus25.851.2825.85\pm 1.28 | 47.90±1.03plus-or-minus47.901.0347.90\pm 1.03 | 48.54±0.50plus-or-minus48.540.5048.54\pm 0.50 | 71.93±1.05plus-or-minus71.931.0571.93\pm 1.05 | 73.90±1.39plus-or-minus73.901.3973.90\pm 1.39 | 52.64±1.40plus-or-minus52.641.4052.64\pm 1.40 | 53.4653.4653.46 |
| Base EN + OH/ELI5 Filter | 28.84±1.32plus-or-minus28.841.3228.84\pm 1.32 | 55.85±1.02plus-or-minus55.851.0255.85\pm 1.02 | 49.63±0.50plus-or-minus49.630.5049.63\pm 0.50 | 71.22±1.06plus-or-minus71.221.0671.22\pm 1.06 | 78.70±1.30plus-or-minus78.701.3078.70\pm 1.30 | 53.12±1.40plus-or-minus53.121.4053.12\pm 1.40 | 56.2356.2356.23 |
| ARC EN | 30.72±1.35plus-or-minus30.721.3530.72\pm 1.35 | 57.49±1.01plus-or-minus57.491.0157.49\pm 1.01 | 48.38±0.50plus-or-minus48.380.5048.38\pm 0.50 | 73.29±1.03plus-or-minus73.291.0373.29\pm 1.03 | 80.10±1.26plus-or-minus80.101.2680.10\pm 1.26 | 53.75±1.40plus-or-minus53.751.4053.75\pm 1.40 | 57.2957.2957.29 |
| SciQ+Inst | 30.38±1.34plus-or-minus30.381.3430.38\pm 1.34 | 56.31±1.02plus-or-minus56.311.0256.31\pm 1.02 | 47.66±0.50plus-or-minus47.660.5047.66\pm 0.50 | 71.16±1.06plus-or-minus71.161.0671.16\pm 1.06 | 78.10±1.31plus-or-minus78.101.3178.10\pm 1.31 | 52.49±1.40plus-or-minus52.491.4052.49\pm 1.40 | 56.0256.0256.02 |
| FWE | 36.18±1.40plus-or-minus36.181.4036.18\pm 1.40 | 67.13±0.96plus-or-minus67.130.9667.13\pm 0.96 | 54.07±0.50plus-or-minus54.070.5054.07\pm 0.50 | 73.99±1.02plus-or-minus73.991.0273.99\pm 1.02 | 80.90±1.24plus-or-minus80.901.2480.90\pm 1.24 | 55.96±1.40plus-or-minus55.961.4055.96\pm 1.40 | 61.3761.3761.37 |
| Base ZH | 25.65±1.29plus-or-minus25.651.2925.65\pm 1.29 | 38.88±1.02plus-or-minus38.881.0238.88\pm 1.02 | 33.07±0.49plus-or-minus33.070.4933.07\pm 0.49 | 56.37±1.16plus-or-minus56.371.1656.37\pm 1.16 | 69.90±1.45plus-or-minus69.901.4569.90\pm 1.45 | 49.86±1.54plus-or-minus49.861.5449.86\pm 1.54 | 45.6245.6245.62 |
| Base EN | 25.22±1.28plus-or-minus25.221.2825.22\pm 1.28 | 36.77±1.01plus-or-minus36.771.0136.77\pm 1.01 | 31.62±0.48plus-or-minus31.620.4831.62\pm 0.48 | 56.09±1.16plus-or-minus56.091.1656.09\pm 1.16 | 68.00±1.48plus-or-minus68.001.4868.00\pm 1.48 | 48.35±1.54plus-or-minus48.351.5448.35\pm 1.54 | 44.3444.3444.34 |
| Base EN + OH/ELI5 Filter | 23.56±1.25plus-or-minus23.561.2523.56\pm 1.25 | 38.22±1.02plus-or-minus38.221.0238.22\pm 1.02 | 32.45±0.49plus-or-minus32.450.4932.45\pm 0.49 | 54.68±1.16plus-or-minus54.681.1654.68\pm 1.16 | 68.50±1.47plus-or-minus68.501.4768.50\pm 1.47 | 52.79±1.53plus-or-minus52.791.5352.79\pm 1.53 | 45.0345.0345.03 |
| ARC EN | 23.21±1.25plus-or-minus23.211.2523.21\pm 1.25 | 37.52±1.02plus-or-minus37.521.0237.52\pm 1.02 | 32.06±0.48plus-or-minus32.060.4832.06\pm 0.48 | 55.44±1.16plus-or-minus55.441.1655.44\pm 1.16 | 70.90±1.44plus-or-minus70.901.4470.90\pm 1.44 | 48.54±1.54plus-or-minus48.541.5448.54\pm 1.54 | 44.6144.6144.61 |
| SciQ+Inst | 22.16±1.23plus-or-minus22.161.2322.16\pm 1.23 | 38.00±1.02plus-or-minus38.001.0238.00\pm 1.02 | 32.24±0.49plus-or-minus32.240.4932.24\pm 0.49 | 54.03±1.16plus-or-minus54.031.1654.03\pm 1.16 | 68.60±1.47plus-or-minus68.601.4768.60\pm 1.47 | 53.07±1.53plus-or-minus53.071.5353.07\pm 1.53 | 44.6844.6844.68 |
| FWE | 25.04±1.28plus-or-minus25.041.2825.04\pm 1.28 | 36.50±1.01plus-or-minus36.501.0136.50\pm 1.01 | 31.89±0.48plus-or-minus31.890.4831.89\pm 0.48 | 54.62±1.16plus-or-minus54.621.1654.62\pm 1.16 | 66.30±1.50plus-or-minus66.301.5066.30\pm 1.50 | 51.94±1.54plus-or-minus51.941.5451.94\pm 1.54 | 44.3844.3844.38 |

Table 11: Evaluation of 111B parameter XL model on “General Understanding Tasks” focusing on general reasoning, language understanding, and science knowledge in English followed by translated Chinese. Results show the length normalized accuracy for individual datasets and the average over all datasets for all datasets.

## Appendix G Results for Individual datasets for German Target Language Models

Results for the 300M models on individual eval datasets are also provides in Tables [12](#A7.T12 "Table 12 ‣ Appendix G Results for Individual datasets for German Target Language Models ‣ Training Bilingual LMs with Data Constraints in the Targeted Language")-[13](#A7.T13 "Table 13 ‣ Appendix G Results for Individual datasets for German Target Language Models ‣ Training Bilingual LMs with Data Constraints in the Targeted Language"). Results for 1B models on English evaluation tasks are shown in Table [14](#A7.T14 "Table 14 ‣ Appendix G Results for Individual datasets for German Target Language Models ‣ Training Bilingual LMs with Data Constraints in the Targeted Language") and for translated German benchmarks in Table [15](#A7.T15 "Table 15 ‣ Appendix G Results for Individual datasets for German Target Language Models ‣ Training Bilingual LMs with Data Constraints in the Targeted Language").

|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No ARC Base DE | 21.08±1.19plus-or-minus21.081.1921.08\pm 1.19 | 31.52±0.95plus-or-minus31.520.9531.52\pm 0.95 | 28.09±0.45plus-or-minus28.090.4528.09\pm 0.45 | 56.47±1.16plus-or-minus56.471.1656.47\pm 1.16 | 54.80±1.57plus-or-minus54.801.5754.80\pm 1.57 | 51.30±1.40plus-or-minus51.301.4051.30\pm 1.40 | 40.5440.5440.54 |
| Base DE | 20.73±1.18plus-or-minus20.731.1820.73\pm 1.18 | 33.33±0.97plus-or-minus33.330.9733.33\pm 0.97 | 28.15±0.45plus-or-minus28.150.4528.15\pm 0.45 | 57.40±1.15plus-or-minus57.401.1557.40\pm 1.15 | 56.30±1.57plus-or-minus56.301.5756.30\pm 1.57 | 50.59±1.41plus-or-minus50.591.4150.59\pm 1.41 | 41.0941.0941.09 |
| Base EN | 23.81±1.24plus-or-minus23.811.2423.81\pm 1.24 | 41.92±1.01plus-or-minus41.921.0141.92\pm 1.01 | 35.73±0.48plus-or-minus35.730.4835.73\pm 0.48 | 67.03±1.10plus-or-minus67.031.1067.03\pm 1.10 | 66.10±1.50plus-or-minus66.101.5066.10\pm 1.50 | 50.75±1.41plus-or-minus50.751.4150.75\pm 1.41 | 47.5647.5647.56 |
| Base EN + OH/ELI5 Filter | 25.43±1.27plus-or-minus25.431.2725.43\pm 1.27 | 46.17±1.02plus-or-minus46.171.0246.17\pm 1.02 | 35.82±0.48plus-or-minus35.820.4835.82\pm 0.48 | 67.19±1.10plus-or-minus67.191.1067.19\pm 1.10 | 67.50±1.48plus-or-minus67.501.4867.50\pm 1.48 | 52.49±1.40plus-or-minus52.491.4052.49\pm 1.40 | 49.1049.1049.10 |
| ARC EN | 26.28±1.29plus-or-minus26.281.2926.28\pm 1.29 | 46.80±1.02plus-or-minus46.801.0246.80\pm 1.02 | 36.44±0.48plus-or-minus36.440.4836.44\pm 0.48 | 67.46±1.09plus-or-minus67.461.0967.46\pm 1.09 | 70.30±1.45plus-or-minus70.301.4570.30\pm 1.45 | 51.38±1.40plus-or-minus51.381.4051.38\pm 1.40 | 49.7849.7849.78 |
| HS EN | 23.72±1.24plus-or-minus23.721.2423.72\pm 1.24 | 42.42±1.01plus-or-minus42.421.0142.42\pm 1.01 | 39.64±0.49plus-or-minus39.640.4939.64\pm 0.49 | 70.13±1.07plus-or-minus70.131.0770.13\pm 1.07 | 67.20±1.49plus-or-minus67.201.4967.20\pm 1.49 | 50.43±1.41plus-or-minus50.431.4150.43\pm 1.41 | 48.9348.9348.93 |
| HS+ARC EN | 24.23±1.25plus-or-minus24.231.2524.23\pm 1.25 | 44.40±1.02plus-or-minus44.401.0244.40\pm 1.02 | 38.51±0.49plus-or-minus38.510.4938.51\pm 0.49 | 68.50±1.08plus-or-minus68.501.0868.50\pm 1.08 | 68.70±1.47plus-or-minus68.701.4768.70\pm 1.47 | 50.43±1.41plus-or-minus50.431.4150.43\pm 1.41 | 49.1349.1349.13 |
| SciQ | 26.62±1.29plus-or-minus26.621.2926.62\pm 1.29 | 49.33±1.03plus-or-minus49.331.0349.33\pm 1.03 | 31.94±0.47plus-or-minus31.940.4731.94\pm 0.47 | 63.44±1.12plus-or-minus63.441.1263.44\pm 1.12 | 72.40±1.41plus-or-minus72.401.4172.40\pm 1.41 | 50.83±1.41plus-or-minus50.831.4150.83\pm 1.41 | 49.0949.0949.09 |
| Inst | 23.55±1.24plus-or-minus23.551.2423.55\pm 1.24 | 43.27±1.02plus-or-minus43.271.0243.27\pm 1.02 | 36.40±0.48plus-or-minus36.400.4836.40\pm 0.48 | 67.25±1.09plus-or-minus67.251.0967.25\pm 1.09 | 68.90±1.46plus-or-minus68.901.4668.90\pm 1.46 | 51.14±1.40plus-or-minus51.141.4051.14\pm 1.40 | 48.4248.4248.42 |
| SciQ+Inst | 26.19±1.28plus-or-minus26.191.2826.19\pm 1.28 | 46.93±1.02plus-or-minus46.931.0246.93\pm 1.02 | 36.02±0.48plus-or-minus36.020.4836.02\pm 0.48 | 66.21±1.10plus-or-minus66.211.1066.21\pm 1.10 | 73.10±1.40plus-or-minus73.101.4073.10\pm 1.40 | 50.67±1.41plus-or-minus50.671.4150.67\pm 1.41 | 49.8549.8549.85 |
| v1 Base EN | 21.16±1.19plus-or-minus21.161.1921.16\pm 1.19 | 36.83±0.99plus-or-minus36.830.9936.83\pm 0.99 | 29.19±0.45plus-or-minus29.190.4529.19\pm 0.45 | 60.07±1.14plus-or-minus60.071.1460.07\pm 1.14 | 63.50±1.52plus-or-minus63.501.5263.50\pm 1.52 | 52.33±1.40plus-or-minus52.331.4052.33\pm 1.40 | 43.8443.8443.84 |
| v2 Base EN | 21.16±1.19plus-or-minus21.161.1921.16\pm 1.19 | 34.89±0.98plus-or-minus34.890.9834.89\pm 0.98 | 29.60±0.46plus-or-minus29.600.4629.60\pm 0.46 | 57.45±1.15plus-or-minus57.451.1557.45\pm 1.15 | 59.50±1.55plus-or-minus59.501.5559.50\pm 1.55 | 50.36±1.41plus-or-minus50.361.4150.36\pm 1.41 | 42.1642.1642.16 |
| v3 Base EN | 19.54±1.16plus-or-minus19.541.1619.54\pm 1.16 | 35.02±0.98plus-or-minus35.020.9835.02\pm 0.98 | 29.46±0.45plus-or-minus29.460.4529.46\pm 0.45 | 59.47±1.15plus-or-minus59.471.1559.47\pm 1.15 | 61.40±1.54plus-or-minus61.401.5461.40\pm 1.54 | 50.51±1.41plus-or-minus50.511.4150.51\pm 1.41 | 42.5742.5742.57 |
| RPJv2 | 25.09±1.27plus-or-minus25.091.2725.09\pm 1.27 | 43.27±1.02plus-or-minus43.271.0243.27\pm 1.02 | 37.23±0.48plus-or-minus37.230.4837.23\pm 0.48 | 65.02±1.11plus-or-minus65.021.1165.02\pm 1.11 | 66.30±1.50plus-or-minus66.301.5066.30\pm 1.50 | 49.80±1.41plus-or-minus49.801.4149.80\pm 1.41 | 47.7847.7847.78 |
| RefinedWeb | 24.40±1.26plus-or-minus24.401.2624.40\pm 1.26 | 43.98±1.02plus-or-minus43.981.0243.98\pm 1.02 | 39.75±0.49plus-or-minus39.750.4939.75\pm 0.49 | 68.66±1.08plus-or-minus68.661.0868.66\pm 1.08 | 69.80±1.45plus-or-minus69.801.4569.80\pm 1.45 | 52.49±1.40plus-or-minus52.491.4052.49\pm 1.40 | 49.8549.8549.85 |
| FineWeb | 25.00±1.27plus-or-minus25.001.2725.00\pm 1.27 | 44.23±1.02plus-or-minus44.231.0244.23\pm 1.02 | 40.89±0.49plus-or-minus40.890.4940.89\pm 0.49 | 69.53±1.07plus-or-minus69.531.0769.53\pm 1.07 | 68.40±1.47plus-or-minus68.401.4768.40\pm 1.47 | 51.78±1.40plus-or-minus51.781.4051.78\pm 1.40 | 49.9749.9749.97 |
| FineWebEDU | 28.67±1.32plus-or-minus28.671.3228.67\pm 1.32 | 56.06±1.02plus-or-minus56.061.0256.06\pm 1.02 | 40.85±0.49plus-or-minus40.850.4940.85\pm 0.49 | 66.65±1.10plus-or-minus66.651.1066.65\pm 1.10 | 72.60±1.41plus-or-minus72.601.4172.60\pm 1.41 | 52.09±1.40plus-or-minus52.091.4052.09\pm 1.40 | 52.8252.8252.82 |

Table 12: Evaluation of 300300300M parameter medium model on “General Understanding Tasks” focusing on general reasoning, language understanding, and science knowledge in English. Results show the length normalized accuracy for individual datasets and the average over all datasets for all datasets.



|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model Name | ARC-C-DE | ARC-E-DE | HS-DE | PIQA-DE | SCIQ-DE | WGrande-DE | AVG-DE |
| No ARC Base DE | 23.48±1.26plus-or-minus23.481.2623.48\pm 1.26 | 32.30±0.98plus-or-minus32.300.9832.30\pm 0.98 | 30.45±0.48plus-or-minus30.450.4830.45\pm 0.48 | 56.09±1.16plus-or-minus56.091.1656.09\pm 1.16 | 60.32±1.59plus-or-minus60.321.5960.32\pm 1.59 | 52.11±1.45plus-or-minus52.111.4552.11\pm 1.45 | 42.4642.4642.46 |
| Base DE | 24.98±1.28plus-or-minus24.981.2824.98\pm 1.28 | 34.87±1.00plus-or-minus34.871.0034.87\pm 1.00 | 32.28±0.48plus-or-minus32.280.4832.28\pm 0.48 | 59.79±1.14plus-or-minus59.791.1459.79\pm 1.14 | 61.26±1.58plus-or-minus61.261.5861.26\pm 1.58 | 51.52±1.45plus-or-minus51.521.4551.52\pm 1.45 | 44.1244.1244.12 |
| Base EN | 23.83±1.26plus-or-minus23.831.2623.83\pm 1.26 | 30.66±0.97plus-or-minus30.660.9730.66\pm 0.97 | 29.69±0.47plus-or-minus29.690.4729.69\pm 0.47 | 56.37±1.16plus-or-minus56.371.1656.37\pm 1.16 | 60.84±1.58plus-or-minus60.841.5860.84\pm 1.58 | 51.01±1.45plus-or-minus51.011.4551.01\pm 1.45 | 42.0742.0742.07 |
| Base EN + OH/ELI5 Filter | 24.27±1.27plus-or-minus24.271.2724.27\pm 1.27 | 32.74±0.99plus-or-minus32.740.9932.74\pm 0.99 | 29.68±0.47plus-or-minus29.680.4729.68\pm 0.47 | 55.88±1.16plus-or-minus55.881.1655.88\pm 1.16 | 60.42±1.59plus-or-minus60.421.5960.42\pm 1.59 | 51.69±1.45plus-or-minus51.691.4551.69\pm 1.45 | 42.4542.4542.45 |
| ARC EN | 23.83±1.26plus-or-minus23.831.2623.83\pm 1.26 | 33.05±0.99plus-or-minus33.050.9933.05\pm 0.99 | 29.59±0.47plus-or-minus29.590.4729.59\pm 0.47 | 56.37±1.16plus-or-minus56.371.1656.37\pm 1.16 | 60.11±1.59plus-or-minus60.111.5960.11\pm 1.59 | 51.60±1.45plus-or-minus51.601.4551.60\pm 1.45 | 42.4342.4342.43 |
| HS EN | 23.75±1.26plus-or-minus23.751.2623.75\pm 1.26 | 31.81±0.98plus-or-minus31.810.9831.81\pm 0.98 | 30.32±0.47plus-or-minus30.320.4730.32\pm 0.47 | 57.40±1.15plus-or-minus57.401.1557.40\pm 1.15 | 59.68±1.59plus-or-minus59.681.5959.68\pm 1.59 | 51.52±1.45plus-or-minus51.521.4551.52\pm 1.45 | 42.4142.4142.41 |
| HS+ARC EN | 24.54±1.28plus-or-minus24.541.2824.54\pm 1.28 | 33.01±0.99plus-or-minus33.010.9933.01\pm 0.99 | 30.10±0.47plus-or-minus30.100.4730.10\pm 0.47 | 55.01±1.16plus-or-minus55.011.1655.01\pm 1.16 | 60.11±1.59plus-or-minus60.111.5960.11\pm 1.59 | 50.84±1.45plus-or-minus50.841.4550.84\pm 1.45 | 42.2742.2742.27 |
| SciQ | 23.66±1.26plus-or-minus23.661.2623.66\pm 1.26 | 32.21±0.98plus-or-minus32.210.9832.21\pm 0.98 | 28.84±0.47plus-or-minus28.840.4728.84\pm 0.47 | 56.26±1.16plus-or-minus56.261.1656.26\pm 1.16 | 61.58±1.58plus-or-minus61.581.5861.58\pm 1.58 | 52.96±1.45plus-or-minus52.961.4552.96\pm 1.45 | 42.5842.5842.58 |
| Inst | 25.42±1.29plus-or-minus25.421.2925.42\pm 1.29 | 32.79±0.99plus-or-minus32.790.9932.79\pm 0.99 | 29.59±0.47plus-or-minus29.590.4729.59\pm 0.47 | 55.93±1.16plus-or-minus55.931.1655.93\pm 1.16 | 61.05±1.58plus-or-minus61.051.5861.05\pm 1.58 | 50.68±1.45plus-or-minus50.681.4550.68\pm 1.45 | 42.5842.5842.58 |
| SciQ+Inst | 23.22±1.25plus-or-minus23.221.2523.22\pm 1.25 | 33.67±0.99plus-or-minus33.670.9933.67\pm 0.99 | 29.42±0.47plus-or-minus29.420.4729.42\pm 0.47 | 55.93±1.16plus-or-minus55.931.1655.93\pm 1.16 | 61.68±1.58plus-or-minus61.681.5861.68\pm 1.58 | 51.27±1.45plus-or-minus51.271.4551.27\pm 1.45 | 42.5342.5342.53 |
| v1 Base EN | 23.92±1.27plus-or-minus23.921.2723.92\pm 1.27 | 35.27±1.01plus-or-minus35.271.0135.27\pm 1.01 | 32.79±0.49plus-or-minus32.790.4932.79\pm 0.49 | 59.74±1.14plus-or-minus59.741.1459.74\pm 1.14 | 62.42±1.57plus-or-minus62.421.5762.42\pm 1.57 | 50.68±1.45plus-or-minus50.681.4550.68\pm 1.45 | 44.1444.1444.14 |
| v2 Base EN | 24.27±1.27plus-or-minus24.271.2724.27\pm 1.27 | 36.19±1.01plus-or-minus36.191.0136.19\pm 1.01 | 32.72±0.48plus-or-minus32.720.4832.72\pm 0.48 | 59.41±1.15plus-or-minus59.411.1559.41\pm 1.15 | 63.89±1.56plus-or-minus63.891.5663.89\pm 1.56 | 52.53±1.45plus-or-minus52.531.4552.53\pm 1.45 | 44.8444.8444.84 |
| v3 Base EN | 23.83±1.26plus-or-minus23.831.2623.83\pm 1.26 | 36.19±1.01plus-or-minus36.191.0136.19\pm 1.01 | 34.06±0.49plus-or-minus34.060.4934.06\pm 0.49 | 61.04±1.14plus-or-minus61.041.1461.04\pm 1.14 | 63.26±1.56plus-or-minus63.261.5663.26\pm 1.56 | 51.77±1.45plus-or-minus51.771.4551.77\pm 1.45 | 45.0345.0345.03 |
| RPJv2 | 24.10±1.27plus-or-minus24.101.2724.10\pm 1.27 | 33.10±0.99plus-or-minus33.100.9933.10\pm 0.99 | 30.06±0.47plus-or-minus30.060.4730.06\pm 0.47 | 55.44±1.16plus-or-minus55.441.1655.44\pm 1.16 | 59.05±1.60plus-or-minus59.051.6059.05\pm 1.60 | 50.08±1.45plus-or-minus50.081.4550.08\pm 1.45 | 41.9741.9741.97 |
| RefinedWeb | 25.33±1.29plus-or-minus25.331.2925.33\pm 1.29 | 32.12±0.98plus-or-minus32.120.9832.12\pm 0.98 | 30.56±0.48plus-or-minus30.560.4830.56\pm 0.48 | 57.34±1.15plus-or-minus57.341.1557.34\pm 1.15 | 62.32±1.57plus-or-minus62.321.5762.32\pm 1.57 | 51.77±1.45plus-or-minus51.771.4551.77\pm 1.45 | 43.2443.2443.24 |
| FineWeb | 24.45±1.28plus-or-minus24.451.2824.45\pm 1.28 | 33.67±0.99plus-or-minus33.670.9933.67\pm 0.99 | 31.20±0.48plus-or-minus31.200.4831.20\pm 0.48 | 56.31±1.16plus-or-minus56.311.1656.31\pm 1.16 | 62.32±1.57plus-or-minus62.321.5762.32\pm 1.57 | 51.01±1.45plus-or-minus51.011.4551.01\pm 1.45 | 43.1643.1643.16 |
| FineWebEDU | 25.59±1.29plus-or-minus25.591.2925.59\pm 1.29 | 35.00±1.00plus-or-minus35.001.0035.00\pm 1.00 | 31.19±0.48plus-or-minus31.190.4831.19\pm 0.48 | 56.86±1.16plus-or-minus56.861.1656.86\pm 1.16 | 62.32±1.57plus-or-minus62.321.5762.32\pm 1.57 | 51.52±1.45plus-or-minus51.521.4551.52\pm 1.45 | 43.7543.7543.75 |

Table 13: Evaluation of 300300300M parameter XL model on “General Understanding Tasks” focusing on general reasoning, language understanding, and science knowledge in translated German. Results show the length normalized accuracy for individual datasets and the average over all datasets for all datasets.



|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model Name | ARC-C | ARC-E | HS | PIQA | SCIQ | WGrande | AVG |
| No ARC Base DE | 21.33±1.20plus-or-minus21.331.2021.33\pm 1.20 | 35.40±0.98plus-or-minus35.400.9835.40\pm 0.98 | 30.88±0.46plus-or-minus30.880.4630.88\pm 0.46 | 59.14±1.15plus-or-minus59.141.1559.14\pm 1.15 | 60.50±1.55plus-or-minus60.501.5560.50\pm 1.55 | 50.67±1.41plus-or-minus50.671.4150.67\pm 1.41 | 42.9942.9942.99 |
| Base DE | 22.78±1.23plus-or-minus22.781.2322.78\pm 1.23 | 37.92±1.00plus-or-minus37.921.0037.92\pm 1.00 | 33.06±0.47plus-or-minus33.060.4733.06\pm 0.47 | 62.35±1.13plus-or-minus62.351.1362.35\pm 1.13 | 64.70±1.51plus-or-minus64.701.5164.70\pm 1.51 | 51.14±1.40plus-or-minus51.141.4051.14\pm 1.40 | 45.3345.3345.33 |
| Base EN | 25.94±1.28plus-or-minus25.941.2825.94\pm 1.28 | 48.15±1.03plus-or-minus48.151.0348.15\pm 1.03 | 48.50±0.50plus-or-minus48.500.5048.50\pm 0.50 | 71.55±1.05plus-or-minus71.551.0571.55\pm 1.05 | 73.10±1.40plus-or-minus73.101.4073.10\pm 1.40 | 52.01±1.40plus-or-minus52.011.4052.01\pm 1.40 | 53.2153.2153.21 |
| Base EN + OH/ELI5 Filter | 29.52±1.33plus-or-minus29.521.3329.52\pm 1.33 | 57.03±1.02plus-or-minus57.031.0257.03\pm 1.02 | 49.96±0.50plus-or-minus49.960.5049.96\pm 0.50 | 71.87±1.05plus-or-minus71.871.0571.87\pm 1.05 | 77.90±1.31plus-or-minus77.901.3177.90\pm 1.31 | 53.83±1.40plus-or-minus53.831.4053.83\pm 1.40 | 56.6956.6956.69 |
| ARC EN | 31.83±1.36plus-or-minus31.831.3631.83\pm 1.36 | 57.24±1.02plus-or-minus57.241.0257.24\pm 1.02 | 48.80±0.50plus-or-minus48.800.5048.80\pm 0.50 | 73.07±1.04plus-or-minus73.071.0473.07\pm 1.04 | 77.00±1.33plus-or-minus77.001.3377.00\pm 1.33 | 52.72±1.40plus-or-minus52.721.4052.72\pm 1.40 | 56.7856.7856.78 |
| HS EN | 28.84±1.32plus-or-minus28.841.3228.84\pm 1.32 | 49.83±1.03plus-or-minus49.831.0349.83\pm 1.03 | 54.83±0.50plus-or-minus54.830.5054.83\pm 0.50 | 75.14±1.01plus-or-minus75.141.0175.14\pm 1.01 | 75.10±1.37plus-or-minus75.101.3775.10\pm 1.37 | 54.85±1.40plus-or-minus54.851.4054.85\pm 1.40 | 56.4356.4356.43 |
| HS+ARC EN | 28.84±1.32plus-or-minus28.841.3228.84\pm 1.32 | 56.10±1.02plus-or-minus56.101.0256.10\pm 1.02 | 53.04±0.50plus-or-minus53.040.5053.04\pm 0.50 | 73.67±1.03plus-or-minus73.671.0373.67\pm 1.03 | 77.90±1.31plus-or-minus77.901.3177.90\pm 1.31 | 54.38±1.40plus-or-minus54.381.4054.38\pm 1.40 | 57.3257.3257.32 |
| SciQ | 30.63±1.35plus-or-minus30.631.3530.63\pm 1.35 | 57.28±1.02plus-or-minus57.281.0257.28\pm 1.02 | 39.79±0.49plus-or-minus39.790.4939.79\pm 0.49 | 68.28±1.09plus-or-minus68.281.0968.28\pm 1.09 | 80.40±1.26plus-or-minus80.401.2680.40\pm 1.26 | 53.91±1.40plus-or-minus53.911.4053.91\pm 1.40 | 55.0555.0555.05 |
| Inst | 27.22±1.30plus-or-minus27.221.3027.22\pm 1.30 | 53.66±1.02plus-or-minus53.661.0253.66\pm 1.02 | 50.30±0.50plus-or-minus50.300.5050.30\pm 0.50 | 71.49±1.05plus-or-minus71.491.0571.49\pm 1.05 | 76.20±1.35plus-or-minus76.201.3576.20\pm 1.35 | 53.51±1.40plus-or-minus53.511.4053.51\pm 1.40 | 55.4055.4055.40 |
| SciQ+Inst | 29.10±1.33plus-or-minus29.101.3329.10\pm 1.33 | 55.22±1.02plus-or-minus55.221.0255.22\pm 1.02 | 48.34±0.50plus-or-minus48.340.5048.34\pm 0.50 | 71.71±1.05plus-or-minus71.711.0571.71\pm 1.05 | 78.50±1.30plus-or-minus78.501.3078.50\pm 1.30 | 53.04±1.40plus-or-minus53.041.4053.04\pm 1.40 | 55.9855.9855.98 |
| v1 Base EN | 20.73±1.18plus-or-minus20.731.1820.73\pm 1.18 | 38.97±1.00plus-or-minus38.971.0038.97\pm 1.00 | 36.09±0.48plus-or-minus36.090.4836.09\pm 0.48 | 63.33±1.12plus-or-minus63.331.1263.33\pm 1.12 | 65.40±1.51plus-or-minus65.401.5165.40\pm 1.51 | 52.01±1.40plus-or-minus52.011.4052.01\pm 1.40 | 46.0946.0946.09 |
| v2 Base EN | 22.70±1.22plus-or-minus22.701.2222.70\pm 1.22 | 38.97±1.00plus-or-minus38.971.0038.97\pm 1.00 | 36.55±0.48plus-or-minus36.550.4836.55\pm 0.48 | 64.80±1.11plus-or-minus64.801.1164.80\pm 1.11 | 69.30±1.46plus-or-minus69.301.4669.30\pm 1.46 | 51.22±1.40plus-or-minus51.221.4051.22\pm 1.40 | 47.2647.2647.26 |
| v3 Base EN | 20.65±1.18plus-or-minus20.651.1820.65\pm 1.18 | 40.36±1.01plus-or-minus40.361.0140.36\pm 1.01 | 36.57±0.48plus-or-minus36.570.4836.57\pm 0.48 | 63.66±1.12plus-or-minus63.661.1263.66\pm 1.12 | 70.50±1.44plus-or-minus70.501.4470.50\pm 1.44 | 51.54±1.40plus-or-minus51.541.4051.54\pm 1.40 | 47.2147.2147.21 |
| RPJv2 | 26.96±1.30plus-or-minus26.961.3026.96\pm 1.30 | 50.42±1.03plus-or-minus50.421.0350.42\pm 1.03 | 51.10±0.50plus-or-minus51.100.5051.10\pm 0.50 | 70.57±1.06plus-or-minus70.571.0670.57\pm 1.06 | 77.60±1.32plus-or-minus77.601.3277.60\pm 1.32 | 55.64±1.40plus-or-minus55.641.4055.64\pm 1.40 | 55.3855.3855.38 |
| RefinedWeb | 27.90±1.31plus-or-minus27.901.3127.90\pm 1.31 | 54.59±1.02plus-or-minus54.591.0254.59\pm 1.02 | 54.91±0.50plus-or-minus54.910.5054.91\pm 0.50 | 73.23±1.03plus-or-minus73.231.0373.23\pm 1.03 | 77.60±1.32plus-or-minus77.601.3277.60\pm 1.32 | 56.35±1.39plus-or-minus56.351.3956.35\pm 1.39 | 57.4357.4357.43 |
| FineWeb | 27.82±1.31plus-or-minus27.821.3127.82\pm 1.31 | 52.15±1.03plus-or-minus52.151.0352.15\pm 1.03 | 56.24±0.50plus-or-minus56.240.5056.24\pm 0.50 | 73.94±1.02plus-or-minus73.941.0273.94\pm 1.02 | 74.40±1.38plus-or-minus74.401.3874.40\pm 1.38 | 55.72±1.40plus-or-minus55.721.4055.72\pm 1.40 | 56.7156.7156.71 |
| FineWebEDU | 38.14±1.42plus-or-minus38.141.4238.14\pm 1.42 | 66.37±0.97plus-or-minus66.370.9766.37\pm 0.97 | 54.88±0.50plus-or-minus54.880.5054.88\pm 0.50 | 72.25±1.04plus-or-minus72.251.0472.25\pm 1.04 | 84.60±1.14plus-or-minus84.601.1484.60\pm 1.14 | 56.04±1.39plus-or-minus56.041.3956.04\pm 1.39 | 62.0562.0562.05 |

Table 14: Evaluation of 111B parameter XL model on “General Understanding Tasks” focusing on general reasoning, language understanding, and science knowledge in English. Results show the length normalized accuracy for individual datasets and the average over all datasets for all datasets.



|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model Name | ARC-C-DE | ARC-E-DE | HS-DE | PIQA-DE | SCIQ-DE | WGrande-DE | AVG-DE |
| No ARC Base DE | 25.07±1.29plus-or-minus25.071.2925.07\pm 1.29 | 36.95±1.02plus-or-minus36.951.0236.95\pm 1.02 | 35.64±0.49plus-or-minus35.640.4935.64\pm 0.49 | 59.74±1.14plus-or-minus59.741.1459.74\pm 1.14 | 64.53±1.55plus-or-minus64.531.5564.53\pm 1.55 | 52.62±1.45plus-or-minus52.621.4552.62\pm 1.45 | 45.7645.7645.76 |
| Base DE | 27.44±1.32plus-or-minus27.441.3227.44\pm 1.32 | 38.81±1.03plus-or-minus38.811.0338.81\pm 1.03 | 39.53±0.51plus-or-minus39.530.5139.53\pm 0.51 | 63.17±1.13plus-or-minus63.171.1363.17\pm 1.13 | 67.68±1.52plus-or-minus67.681.5267.68\pm 1.52 | 52.62±1.45plus-or-minus52.621.4552.62\pm 1.45 | 48.2148.2148.21 |
| Base EN | 25.07±1.29plus-or-minus25.071.2925.07\pm 1.29 | 37.70±1.02plus-or-minus37.701.0237.70\pm 1.02 | 36.18±0.50plus-or-minus36.180.5036.18\pm 0.50 | 59.36±1.15plus-or-minus59.361.1559.36\pm 1.15 | 63.05±1.57plus-or-minus63.051.5763.05\pm 1.57 | 50.51±1.45plus-or-minus50.511.4550.51\pm 1.45 | 45.3145.3145.31 |
| Base EN + OH/ELI5 Filter | 25.77±1.30plus-or-minus25.771.3025.77\pm 1.30 | 40.00±1.03plus-or-minus40.001.0340.00\pm 1.03 | 35.94±0.50plus-or-minus35.940.5035.94\pm 0.50 | 59.25±1.15plus-or-minus59.251.1559.25\pm 1.15 | 65.05±1.55plus-or-minus65.051.5565.05\pm 1.55 | 51.18±1.45plus-or-minus51.181.4551.18\pm 1.45 | 46.2046.2046.20 |
| ARC EN | 26.91±1.32plus-or-minus26.911.3226.91\pm 1.32 | 39.34±1.03plus-or-minus39.341.0339.34\pm 1.03 | 35.77±0.50plus-or-minus35.770.5035.77\pm 0.50 | 58.65±1.15plus-or-minus58.651.1558.65\pm 1.15 | 67.58±1.52plus-or-minus67.581.5267.58\pm 1.52 | 52.11±1.45plus-or-minus52.111.4552.11\pm 1.45 | 46.7346.7346.73 |
| HS EN | 24.89±1.28plus-or-minus24.891.2824.89\pm 1.28 | 36.90±1.02plus-or-minus36.901.0236.90\pm 1.02 | 37.11±0.50plus-or-minus37.110.5037.11\pm 0.50 | 60.77±1.14plus-or-minus60.771.1460.77\pm 1.14 | 63.47±1.56plus-or-minus63.471.5663.47\pm 1.56 | 52.87±1.45plus-or-minus52.871.4552.87\pm 1.45 | 46.0046.0046.00 |
| HS+ARC EN | 26.74±1.31plus-or-minus26.741.3126.74\pm 1.31 | 38.50±1.02plus-or-minus38.501.0238.50\pm 1.02 | 36.71±0.50plus-or-minus36.710.5036.71\pm 0.50 | 59.96±1.14plus-or-minus59.961.1459.96\pm 1.14 | 63.89±1.56plus-or-minus63.891.5663.89\pm 1.56 | 51.35±1.45plus-or-minus51.351.4551.35\pm 1.45 | 46.1946.1946.19 |
| SciQ | 27.35±1.32plus-or-minus27.351.3227.35\pm 1.32 | 39.38±1.03plus-or-minus39.381.0339.38\pm 1.03 | 33.23±0.49plus-or-minus33.230.4933.23\pm 0.49 | 58.76±1.15plus-or-minus58.761.1558.76\pm 1.15 | 64.21±1.56plus-or-minus64.211.5664.21\pm 1.56 | 51.18±1.45plus-or-minus51.181.4551.18\pm 1.45 | 45.6945.6945.69 |
| Inst | 25.59±1.29plus-or-minus25.591.2925.59\pm 1.29 | 38.14±1.02plus-or-minus38.141.0238.14\pm 1.02 | 36.22±0.50plus-or-minus36.220.5036.22\pm 0.50 | 60.01±1.14plus-or-minus60.011.1460.01\pm 1.14 | 64.74±1.55plus-or-minus64.741.5564.74\pm 1.55 | 51.01±1.45plus-or-minus51.011.4551.01\pm 1.45 | 45.9545.9545.95 |
| SciQ+Inst | 25.24±1.29plus-or-minus25.241.2925.24\pm 1.29 | 39.38±1.03plus-or-minus39.381.0339.38\pm 1.03 | 35.34±0.49plus-or-minus35.340.4935.34\pm 0.49 | 59.74±1.14plus-or-minus59.741.1459.74\pm 1.14 | 64.21±1.56plus-or-minus64.211.5664.21\pm 1.56 | 52.70±1.45plus-or-minus52.701.4552.70\pm 1.45 | 46.1046.1046.10 |
| v1 Base EN | 26.12±1.30plus-or-minus26.121.3026.12\pm 1.30 | 40.93±1.03plus-or-minus40.931.0340.93\pm 1.03 | 40.39±0.51plus-or-minus40.390.5140.39\pm 0.51 | 61.48±1.14plus-or-minus61.481.1461.48\pm 1.14 | 67.58±1.52plus-or-minus67.581.5267.58\pm 1.52 | 51.94±1.45plus-or-minus51.941.4551.94\pm 1.45 | 48.0748.0748.07 |
| v2 Base EN | 25.51±1.29plus-or-minus25.511.2925.51\pm 1.29 | 42.30±1.04plus-or-minus42.301.0442.30\pm 1.04 | 40.75±0.51plus-or-minus40.750.5140.75\pm 0.51 | 62.19±1.13plus-or-minus62.191.1362.19\pm 1.13 | 71.37±1.47plus-or-minus71.371.4771.37\pm 1.47 | 52.45±1.45plus-or-minus52.451.4552.45\pm 1.45 | 49.0949.0949.09 |
| v3 Base EN | 26.21±1.30plus-or-minus26.211.3026.21\pm 1.30 | 40.84±1.03plus-or-minus40.841.0340.84\pm 1.03 | 43.08±0.51plus-or-minus43.080.5143.08\pm 0.51 | 64.09±1.12plus-or-minus64.091.1264.09\pm 1.12 | 66.74±1.53plus-or-minus66.741.5366.74\pm 1.53 | 51.52±1.45plus-or-minus51.521.4551.52\pm 1.45 | 48.7548.7548.75 |
| RPJv2 | 24.71±1.28plus-or-minus24.711.2824.71\pm 1.28 | 37.17±1.02plus-or-minus37.171.0237.17\pm 1.02 | 36.53±0.50plus-or-minus36.530.5036.53\pm 0.50 | 58.65±1.15plus-or-minus58.651.1558.65\pm 1.15 | 65.26±1.55plus-or-minus65.261.5565.26\pm 1.55 | 52.70±1.45plus-or-minus52.701.4552.70\pm 1.45 | 45.8445.8445.84 |
| RefinedWeb | 25.42±1.29plus-or-minus25.421.2925.42\pm 1.29 | 38.81±1.03plus-or-minus38.811.0338.81\pm 1.03 | 38.25±0.50plus-or-minus38.250.5038.25\pm 0.50 | 58.38±1.15plus-or-minus58.381.1558.38\pm 1.15 | 64.21±1.56plus-or-minus64.211.5664.21\pm 1.56 | 51.01±1.45plus-or-minus51.011.4551.01\pm 1.45 | 46.0146.0146.01 |
| FineWeb | 26.12±1.30plus-or-minus26.121.3026.12\pm 1.30 | 36.90±1.02plus-or-minus36.901.0236.90\pm 1.02 | 38.09±0.50plus-or-minus38.090.5038.09\pm 0.50 | 59.85±1.14plus-or-minus59.851.1459.85\pm 1.14 | 64.11±1.56plus-or-minus64.111.5664.11\pm 1.56 | 53.89±1.45plus-or-minus53.891.4553.89\pm 1.45 | 46.4946.4946.49 |
| FineWebEDU | 26.91±1.32plus-or-minus26.911.3226.91\pm 1.32 | 42.39±1.04plus-or-minus42.391.0442.39\pm 1.04 | 37.40±0.50plus-or-minus37.400.5037.40\pm 0.50 | 60.45±1.14plus-or-minus60.451.1460.45\pm 1.14 | 65.37±1.54plus-or-minus65.371.5465.37\pm 1.54 | 50.42±1.45plus-or-minus50.421.4550.42\pm 1.45 | 47.1647.1647.16 |

Table 15: Evaluation of 111B parameter XL model on “General Understanding Tasks” focusing on general reasoning, language understanding, and science knowledge in translated German. Results show the length normalized accuracy for individual datasets and the average over all datasets for all datasets.
