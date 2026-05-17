---
arxiv: '2602.15210'
authors:
- DatologyAI
- ':'
- Aldo Gael Carranza
- Kaleigh Mentzer
- Ricardo Pio Monti
- Alex Fang
- Alvin Deng
- Amro Abbas
- Anshuman Suri
- Brett Larsen
- Cody Blakeney
- Darren Teh
- David Schwab
- Diego Kiner
- Fan Pan
- Haakon Mongstad
- Haoli Yin
- Jack Urbanek
- Jason Lee
- Jason Telanoff
- Josh Wills
- Luke Merrick
- Maximilian Böther
- Parth Doshi
- Paul Burstein
- Pratyush Maini
- Rishabh Adiga
- Siddharth Joshi
- Spandan Das
- Tony Jiang
- Vineeth Dorna
- Zhengping Wang
- Bogdan Gaza
- Ari Morcos
- Matthew Leavitt
parser: ar5iv
retrieved: '2026-05-15'
source: paper
title: 'ÜberWeb: Insights from Multilingual Curation for a 20-Trillion-Token Dataset'
url: https://arxiv.org/abs/2602.15210
year: 2026
---

[2602.15210] Insights from Multilingual Curation for a 20-Trillion-Token Dataset



# Insights from Multilingual Curation for a 20-Trillion-Token Dataset

###### Abstract

Multilinguality is a core capability for modern foundation models, yet training high-quality multilingual models remains challenging due to uneven data availability across languages. A further challenge is the performance interference that can arise from joint multilingual training, commonly referred to as the “curse of multilinguality”. We study multilingual data curation across thirteen languages spanning multiple scripts, language families, and resource levels, showing that many reported regressions are not inherent to multilingual scaling but instead stem from correctable deficiencies in data quality and composition rather than fundamental capacity limits. In controlled bilingual experiments, improving data quality for any single language benefits others: curating English improves non-English performance on MMLU, ARC-Challenge, and Belebele in 12 of 13 languages (3.9% average relative gain), while curating non-English yields reciprocal improvements in English (1.2% average gain). Bespoke per-language curation produces substantially larger within-language improvements, with up to 16.9% relative gains over uncurated baselines. Extending these findings to large-scale general-purpose training mixtures, we show that curated multilingual allocations comprising under 8% of total tokens remain remarkably effective. We operationalize this approach within a broader large-scale effort that produced a 20T-token pretraining corpus derived entirely from public sources. Models with 3B and 8B parameters trained on a 1T-token random subset achieve competitive multilingual accuracy with 4–10× fewer training FLOPs than strong public baselines, establishing a new Pareto frontier in multilingual performance versus compute (Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset")). Moreover, these benefits extend to frontier model scale: the 20T-token corpus served as part of the pretraining dataset for Trinity Large (400B/A13B), which exhibits strong multilingual performance relative to its training FLOPs. Together, these results show that targeted, per-language data curation mitigates multilingual interference and enables compute-efficient multilingual scaling.

DatologyAI Team111See Contributions and Acknowledgments (§ [6](#S6 "6 Contributions and Acknowledgements ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset")) for full author list.

> The future is already here – it’s just not evenly distributed.
>
> — William Gibson

## 1 Introduction

Large language models (LLMs) have fundamentally reshaped the landscape of artificial intelligence, yet their benefits remain unevenly distributed across languages. Although modern models have demonstrated remarkable capabilities in English, these capabilities often degrade substantially when applied to non-English settings
(Ahuja et al., [2024](#bib.bib49 "Megaverse: benchmarking large language models across languages, modalities, models and tasks"); Khanna and Li, [2025](#bib.bib36 "Invisible languages of the llm universe")). Bridging this gap is not merely an architectural challenge,
but fundamentally a data-centric one: training on large volumes of high-quality data is essential for achieving frontier-level model capabilities.
English benefits from multiple large-scale, carefully curated public corpora
(Penedo et al., [2024](#bib.bib70 "The fineweb datasets: decanting the web for the finest text data at scale"); Li et al., [2024](#bib.bib37 "Datacomp-lm: in search of the next generation of training sets for language models"); Su et al., [2025](#bib.bib43 "Nemotron-cc: transforming common crawl into a refined long-horizon pretraining dataset"); Olmo et al., [2025](#bib.bib44 "Olmo 3")),
whereas multilingual corpora are far more fragmented.
Many non-English languages occupy a long tail characterized by limited, noisy, or inconsistently curated data, constraining multilingual model performance regardless of architectural capacity.

Figure 1: A new compute-performance Pareto frontier for English and multilingual capabilities. We report
error rate (log-scale; 1-accuracy, lower is better)
as a function of training FLOPs (log-scale) across English (MMLU+ARC) and three multilingual
benchmarks (Multilingual MMLU, Multilingual ARC, and Belebele).
All evaluations use a multiple-choice format;
multilingual scores are averaged over 13 languages.
The shaded gray region summarizes the performance–compute envelope of representative open-weight baselines (e.g., Qwen3-4B/8B, Granite-4.0-3B).
DatologyAI models occupy the bottom-left region relative to these baselines, indicating substantially lower multilingual error at reduced compute. We restrict our English-language evaluations to MMLU and ARC-Challenge for parity with the multilingual evaluations, and reserve comprehensive English and quantitative benchmarking for forthcoming companion releases.

Beyond uneven data availability across languages, multilingual modeling faces an additional distinct challenge: the so-called “curse of multilinguality” (Conneau et al., [2020](#bib.bib23 "Unsupervised cross-lingual representation learning at scale"); Chang et al., [2024](#bib.bib35 "When is multilinguality a curse? language modeling for 250 high- and low-resource languages")). This term refers to the empirical observation that training a single model across an increasing number of languages often leads to degraded per-language performance, even under comparable training budgets.
Historically, the consensus view attributed this phenomenon to a capacity bottleneck, framing multilingual modeling as a zero-sum game in which distinct languages compete for finite parameters
or model capacity
(Xue et al., [2021](#bib.bib2 "MT5: a massively multilingual pre-trained text-to-text transformer"); Conneau et al., [2020](#bib.bib23 "Unsupervised cross-lingual representation learning at scale"); Chang et al., [2024](#bib.bib35 "When is multilinguality a curse? language modeling for 250 high- and low-resource languages")). Under this paradigm, the primary solutions have been to scale model size
(Blevins et al., [2024](#bib.bib34 "Breaking the curse of multilinguality with cross-lingual expert language models"); Pfeiffer et al., [2022](#bib.bib14 "Lifting the curse of multilinguality by pre-training modular transformers"))
or increase the number of training tokens (Longpre et al., [2025](#bib.bib30 "ATLAS: adaptive transfer scaling laws for multilingual pretraining, finetuning, and decoding the curse of multilinguality")), both of which substantially increase the computational cost of multilingual training. Such strategies, however, assume access to abundant, high-quality multilingual text, bringing them into direct tension with the uneven data availability across languages discussed above.

Recent evidence suggests this capacity-centric view is incomplete. Emerging research indicates that the “curse” may stem less from parameter scarcity
and more from the interference caused by suboptimal data quality. Both Seto et al. ([2025](#bib.bib42 "Assessing the role of data quality in training bilingual language models")) and Foroutan et al. ([2025](#bib.bib21 "Revisiting multilingual data mixtures in language model pretraining")) demonstrate that the trade-off between English and multilingual performance is not inevitable; they find that replacing significant portions of English data with high-quality multilingual text need not degrade English capabilities. Similarly, advances in model-based data selection (Messmer et al., [2025](#bib.bib51 "Enhancing multilingual LLM pretraining with model-based data selection")) and systematic filtering pipelines (Penedo et al., [2025](#bib.bib12 "FineWeb2: one pipeline to scale them all — adapting pre-training data processing to every language")) reveal that when data quality is rigorously controlled,
models can accommodate significantly more linguistic diversity without quality degradation.
Taken together, these findings suggest that the apparent capacity constraints
in multilingual scaling are often
induced by low-quality data.
This motivates a shift in emphasis:
optimal scaling of multilingual capabilities requires intentional,
multilingual-targeted data curation.

In this work, we study multilingual foundation model training through the lens of data curation, arguing that careful curation can simultaneously address the two central challenges of multilingual modeling: limited high-quality data for many languages and performance interference arising from joint multilingual training. By improving data quality, curation enhances cross-lingual transfer, reducing the amount of language-specific data required to achieve strong performance. The complement to this phenomenon is that targeted multilingual curation mitigates interference effects, alleviating the curse of multilinguality without relying solely on increasing compute.
We validate these claims through controlled 60B-token bilingual studies, large-scale 1T-token pretraining, and frontier-scale pretraining at multi-tens-of-trillions of tokens.

We summarize the key contributions of this work as follows:

1. 1.

   Cross-lingual transfer improves with data quality. We demonstrate that refining data quality drives significant cross-lingual performance gains
   through controlled bilingual experiments with 3B-parameter models trained on 60B tokens.
   Crucially, we find this relationship is bidirectional:
   enhancing the quality of English data improves non-English performance in 12 out of 13 examined languages, yielding an average relative
   improvement of 3.91% across
   multilingual evaluations, while improving the quality of non-English data benefits English capabilities in 12 out of 13 languages, with an average relative improvement of 1.21% on English evaluations.
2. 2.

   Optimal performance requires bespoke multilingual curation.
   While English data curation does improve multilingual capabilities, we
   find that the best performance is obtained when tailored curation
   pipelines are built for each language.
   Our findings highlight that
   English-centric curation strategies cannot be applied blindly to other languages. Instead, it is imperative to construct tailored pipelines designed for each language’s specific needs.
   For our 3B-parameter models trained on 60B tokens, while English curation alone drove the aforementioned 3.91% relative improvement, applying bespoke language-specific curation yielded a significantly higher 16.87% relative
   improvement over the uncurated baseline
3. 3.

   Data quality persists through translation.
   Building on findings of recent large-scale translation efforts (Wang et al., [2025](#bib.bib52 "Multilingual language model pretraining using machine-translated data"); Penedo et al., [2026](#bib.bib22 "FineTranslations")),
   we explore various strategies to translate English data into non-English languages.
   Large-scale translation provides a mechanism for expanding training data across languages, but we find that the choice of source data critically determines its effectiveness.
   We observe that prioritizing high-quality English documents for translation can significantly boost performance over
   translations of arbitrary English documents.
   In experiments with 3B-parameter models, we find that augmenting the uncurated baseline with translations of random English data yields marginal gains, whereas translating high-quality, score-filtered English data leads to an average relative
   improvement of 5.09% over an uncurated baseline.
   Moreover, we find that translation is most effective when embedded within a holistic, per-language curation framework, which yields the strongest overall performance.
4. 4.

   Curation makes multilingual scaling remarkably compute-efficient.
   Under a 1T-token training budget drawn from a curated general-purpose pretraining corpus, we find that allocating approximately 8% of tokens to high-quality multilingual data (∼\sim80B tokens across 13 languages) is sufficient to achieve very strong multilingual performance, in many cases comparable to or exceeding competitive open-weight models.
   Our 3B and 8B models trained for 1T tokens achieve 4–10×\times greater training FLOPs efficiency than strong public baselines. For example, a DatologyAI 3B model trained for 1T tokens (1.8×10221.8\times 10^{22} FLOPs) outperforms LFM-2.5-1.2B (Liquid AI, [2026](#bib.bib142 "Introducing LFM2.5: The next generation of on-device AI")), a 1.2B model trained for 28T tokens (1.9×10231.9\times 10^{23} FLOPs). Similarly, a DatologyAI 8B model trained for 1T tokens (4.8×10224.8\times 10^{22} FLOPs) outperforms SmolLM3-3B (Bakouch et al., [2025](#bib.bib6 "SmolLM3: smol, multilingual, long-context reasoner")) and Granite-4.0-3B (IBM Granite Team, [2024](#bib.bib5 "Granite 3.0 language models")), both trained with an order of magnitude more compute. Importantly, these efficiency gains persist at frontier scale: our multilingual curation framework forms part of the 17T-token pretraining corpus for Trinity Large Base (400B-parameter MoE, 13B active; Singh et al. ([2026](#bib.bib39 "Arcee trinity large technical report"))), which exhibits exceptionally strong multilingual performance for its training FLOPs budget.

Our results demonstrate the critical role of multilingual data curation for multilingual model capability. From controlled 60B-token studies to 1T-token training mixtures to 17T-token frontier-scale pretraining, we show that language-aware improvements in data quality systematically enhance cross-lingual transfer, mitigate multilingual interference, and substantially improve within-language performance. These effects collectively shift the performance–compute Pareto frontier (see Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset")). Together, these findings position high-quality, per-language curation as a practical and scalable mechanism for compute-efficient multilingual foundation model training, advancing progress toward more language-inclusive foundation models and a more evenly-distributed future.

## 2 Related Work

##### Multilingual Data Curation.

The field has moved from scale-focused web ingestion (Xue et al., [2021](#bib.bib2 "MT5: a massively multilingual pre-trained text-to-text transformer"); Ortiz Suárez et al., [2020](#bib.bib4 "A monolingual approach to contextualized word embeddings for mid-resource languages")) to systematic, reproducible data curation that prioritizes quality, auditing, and strong filtering to reduce noise and contamination.
Recent efforts such as FineWeb (Penedo et al., [2024](#bib.bib70 "The fineweb datasets: decanting the web for the finest text data at scale")) and FineWeb2 (Penedo et al., [2025](#bib.bib12 "FineWeb2: one pipeline to scale them all — adapting pre-training data processing to every language")) have formalized the curation process, releasing reproducible pipelines that scale high-quality filtering across thousands of languages.
In addition to these advances in pre-training, the Aya initiative (Singh et al., [2024](#bib.bib17 "Aya dataset: an open-access collection for multilingual instruction tuning")) presents a multilingual
post-training dataset focused on instruction-following across 65 languages.

Beyond the construction of large-scale multilingual corpora, there have also been recent efforts to
improve the quality of these corpora via multilingual curation;
both Messmer et al. ([2025](#bib.bib51 "Enhancing multilingual LLM pretraining with model-based data selection")) and Chen et al. ([2025](#bib.bib146 "MuRating: a high quality data selecting approach to multilingual large language model pretraining"))
propose general purpose model-based filtering solutions to improve multilingual data quality.
There are also examples of highly specialized, language-specific curation efforts such as
Burns et al. ([2025](#bib.bib147 "Aleph-alpha-germanweb: improving german-language llm pre-training with model-based data curation and synthetic data generation")) for German
and Khan et al. ([2024](#bib.bib16 "IndicLLMSuite: a blueprint for creating pre-training and fine-tuning datasets for indian languages")) for Indic languages.
Our focus in this work is closely aligned with these multilingual curation efforts, and helps to further
emphasize the performance improvements which can be unlocked via data curation.

##### Data Mixing and Interference.

While the existence of cross-lingual transfer is well established (Pires et al., [2019](#bib.bib149 "How multilingual is multilingual bert?")), a central challenge remains how to drive positive transfer across
languages while minimizing negative interference
(Conneau et al., [2020](#bib.bib23 "Unsupervised cross-lingual representation learning at scale"); Wang et al., [2020](#bib.bib20 "Negative interference in multilingual models: findings and a meta-learning treatment")).
While temperature-based sampling is a standard heuristic (Conneau and Lample, [2019](#bib.bib46 "Cross-lingual language model pretraining")), it often leads to overfitting in low-resource regimes. Strategies like UniMax (Chung et al., [2023](#bib.bib1 "UniMax: fairer and more effective language sampling for large-scale multilingual pretraining")) address this by capping repetition to ensure more representative coverage.
A further area of research is
the use of dynamic curricula
to drive
multilingual performance;
Choi et al. ([2023](#bib.bib47 "Order matters in the presence of dataset imbalance for multilingual learning")) advocate for a two-stage training paradigm which
first pre-trains on high resource languages and subsequently
fine-tunes on lower resource languages. Conversely,
Foroutan et al. ([2025](#bib.bib21 "Revisiting multilingual data mixtures in language model pretraining"))
arrive at the conclusion
that staging the introduction of languages does not
yield tangible improvements.
While this work contains some examination of curricula and multilingual mixture proportions, the
central theme is demonstrating that careful
curation can significantly improve cross-lingual transfer dynamics, thus reducing interference.

##### Multilingual Scaling Laws.

While scaling behaviors for English-centric models are well-characterized (Hestness et al., [2017](#bib.bib15 "Deep learning scaling is predictable, empirically"); Kaplan et al., [2020](#bib.bib31 "Scaling laws for neural language models"); Hoffmann et al., [2022](#bib.bib32 "Training compute-optimal large language models")), extending these laws to the multilingual setting introduces significant complexity due to cross-lingual transfer dynamics.
Early attempts primarily focused on machine translation (Fernandes et al., [2023](#bib.bib33 "Scaling laws for multilingual neural machine translation")), but recent work has targeted general-purpose decoder-only architectures. He et al. ([2025](#bib.bib25 "Scaling laws for multilingual language models")) propose a “family-based” scaling law, demonstrating that the test loss for a language family is primarily determined by its own sampling ratio, largely independent of other families in the mixture. This simplifies the analysis of inter-language competition but does not fully account for the “curse of multilinguality” phenomena observed when scaling to many languages.
Addressing this, the ATLAS project recently conducted the largest study to date, covering over 400 languages and exploring cross-lingual transfer across 38 languages (Longpre et al., [2025](#bib.bib30 "ATLAS: adaptive transfer scaling laws for multilingual pretraining, finetuning, and decoding the curse of multilinguality")).
Their work derives an adaptive transfer scaling law that explicitly models the trade-off between adding languages and maintaining performance per parameter. This provides a first-principles guide for optimal capacity allocation in massively multilingual settings.
While these laws focus on parameter-based trade-offs,
our results instead demonstrate that careful,
language-specific curation allows
us to significantly improve on current scaling laws by shifting the
bottleneck from model capacity to data quality.
In this way, we are able to demonstrate that the
“curse of multilinguality”
is the result of correctable
deficiencies in the training data.

## 3 Experimental Setup and Methodology

Pretraining Data. In this work we curate exclusively on top of open source corpora. For English, we leverage the DCLM corpus (Li et al., [2024](#bib.bib37 "Datacomp-lm: in search of the next generation of training sets for language models")), FineWeb (Penedo et al., [2024](#bib.bib70 "The fineweb datasets: decanting the web for the finest text data at scale")), and the non-synthetic components of Nemotron CC v1 (Su et al., [2025](#bib.bib43 "Nemotron-cc: transforming common crawl into a refined long-horizon pretraining dataset")). For non-English data, we rely on the FineWeb2 corpus (Penedo et al., [2025](#bib.bib12 "FineWeb2: one pipeline to scale them all — adapting pre-training data processing to every language")). While FineWeb2 supports over 1,000 languages, in this work we focus on a set of 13 diverse non-English languages spanning multiple writing systems and language families (see Table [1](#S3.T1 "Table 1 ‣ 3 Experimental Setup and Methodology ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset")).

| Language | Family | Script | FineWeb2 Documents (M) | Llama 3.2 1B Tokens (B) |
| --- | --- | --- | --- | --- |
| Russian | Slavic | Cyrillic | 699.1 | 1004.6 |
| Chinese | Sino-Tibetan | Hanzi | 636.1 | 743.4 |
| German | Germanic | Latin | 496.0 | 407.0 |
| Spanish | Romance | Latin | 441.3 | 352.3 |
| Japanese | Japonic | Kanji + Kana | 400.1 | 404.4 |
| French | Romance | Latin | 360.1 | 306.4 |
| Portuguese | Romance | Latin | 199.7 | 160.1 |
| Indonesian | Austronesian | Latin | 100.2 | 101.8 |
| Arabic | Semitic | Arabic | 62.0 | 63.5 |
| Vietnamese | Austroasiatic | Latin | 61.1 | 47.8 |
| Korean | Koreanic | Hangul | 60.9 | 59.5 |
| Hindi | Indo-Aryan | Devanagari | 22.1 | 25.1 |
| Bengali | Indo-Aryan | Bengali | 15.2 | 38.7 |

Table 1: Non-English languages included in this study.

The languages above also span a wide range of resource levels in publicly available web text: Spanish is high-resource (with hundreds of billions of available tokens), whereas Hindi, Bengali, and Arabic are comparatively low-resource, making them particularly sensitive to data scarcity and quality.

Data curation. Building on our work at DatologyAI, we develop language-specific data curation pipelines for each of the languages above. For English, we build on our state-of-the-art web curation pipeline (DatologyAI, [2024](#bib.bib129 "DatologyAI technical deep-dive: curating our way to a billion-state-of-the-art text dataset")), which integrates complementary strategies including model-based filtering, embedding-based selection, and targeted synthetic data generation (DatologyAI, [2025](#bib.bib137 "BeyondWeb: lessons from scaling synthetic data for trillion-scale pretraining")).
For each non-English language, we tailor our curation pipeline to language’s linguistic and distributional characteristics rather than directly applying the English recipe. Concretely, this includes selecting, validating, and/or training language-appropriate models for 1) filtering, 2) embedding for geometry-based curation, and 3) synthetic rephrasing. We also adapt filtering and mixing strategies to account for script- and language-specific artifacts and varying token scarcity across languages.
To quantify the impact of these interventions, we compare to uncurated baselines, defined throughout this work as samples drawn at random from
the DCLM for uncurated English and FineWeb2 for uncurated non-English corpora222We note that DCLM and FineWeb2 were heavily curated as part of their development, and use the term “uncurated” as meaning “not subject to DatologyAI curation”..

Model. We present results on both 3B and 8B parameter models using a Llama-based architecture (Touvron et al., [2023](#bib.bib45 "Llama: open and efficient foundation language models")).
Throughout this work we use the Llama-3.2 tokenizer.
All models are trained and evaluated with a context window of 4096 tokens.
We note that because the focus of this work is on the effects of data curation, we did not attempt to optimize model quality via any means other than data curation. All models of a given size used identical training configurations in every way except the dataset.

Evaluation. We evaluate our models on three complementary multilingual benchmarks:

* •

  Multilingual MMLU (Singh et al., [2025](#bib.bib29 "Global mmlu: understanding and addressing cultural and linguistic biases in multilingual evaluation")): measures broad knowledge and academic-style reasoning across diverse subject areas, including STEM, humanities, and social sciences.
* •

  Multilingual ARC Challenge (Lai et al., [2023](#bib.bib28 "Okapi: instruction-tuned large language models in multiple languages with reinforcement learning from human feedback")): measures multi-step reasoning on grade-school science questions; it covers a narrower domain than MMLU but places greater emphasis on compositional reasoning.
* •

  Belebele (Bandarkar et al., [2024](#bib.bib24 "The belebele benchmark: a parallel reading comprehension dataset in 122 language variants")): measures multilingual reading comprehension and semantic reasoning over aligned passages, with minimal dependence on memorized factual knowledge.

We complement these multilingual evaluations with English MMLU and ARC evaluations.333We restrict our English-language evaluations to MMLU and ARC-Challenge for parity with the multilingual evaluations, and reserve comprehensive English and quantitative benchmarking for forthcoming companion releases.
Throughout this work, we
rely on the lighteval (Habib et al., [2023](#bib.bib26 "LightEval: a lightweight framework for llm evaluation")) framework for all our evaluations. We report zero-shot performance. For large-scale experiments (e.g., Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset")), we adopt the multiple-choice formulation (MCF), following common practice (Gu et al., [2025](#bib.bib140 "Olmes: a standard for language model evaluations"); Li et al., [2024](#bib.bib37 "Datacomp-lm: in search of the next generation of training sets for language models")). For smaller runs (e.g., our 3B, 60B-token setting), we instead use the cloze formulation. This choice is motivated by statistical efficiency: cloze-style scoring yields a denser learning signal and typically reduces variance relative to discrete option selection, making it better suited for low-resource or early-training regimes where multiple-choice accuracy can be dominated by near-random guessing (Gu et al., [2025](#bib.bib140 "Olmes: a standard for language model evaluations"); Li et al., [2024](#bib.bib37 "Datacomp-lm: in search of the next generation of training sets for language models")).
Finally, we note that all models in this manuscript are base (pretrained) models and were evaluated without any post-training or fine-tuning.
A full list of evaluation datasets by language is provided in Appendix [A.1](#A1.SS1 "A.1 Evaluation datasets per language ‣ Appendix A Appendix ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").

## 4 Main Findings

### 4.1 The impact of curation on multilingual transfer dynamics and language-specific performance

*Cross-lingual transfer*
refers to the observation that improving representations in one language can benefit performance in other languages. As models scale and language coverage increases,
such transfer becomes increasingly important.
In this section, we investigate the impact of data quality on cross-lingual transfer and identify opportunities to improve downstream performance via data
curation
interventions on both English and non-English data.

#### 4.1.1 Improving English data quality improves cross-lingual performance

Most multilingual language models are trained on predominantly English corpora, making English data quality a central determinant of multilingual performance. Yet the extent to which English data quality governs cross-lingual transfer remains insufficiently characterized. In a series of controlled experiments, we show that English-to–non-English transfer is strongly mediated by the quality of the English training data.
In particular,
improving the quality of the English portion of training data mixtures
yields consistent performance gains across
almost all non-English languages considered.
We train a suite of 3B-parameter models for 60B tokens under a range of dataset compositions, focusing on bilingual settings consisting of English paired with a single “target” language. This design yields 13 language pairs (e.g., English–Spanish, English–German, etc), each trained with a fixed 50/50 mixture ratio. For every pair, we compare three curation regimes:

1. i.

   Uncurated English DCLM and uncurated FineWeb2 non-English data (i.e., random samples from DCLM and FineWeb2).
2. ii.

   DatologyAI-curated English and uncurated FineWeb2 non-English data.
3. iii.

   DatologyAI-curated English and DatologyAI-curated non-English data.

Cross-lingual impact of English curation.
Figure [2](#S4.F2 "Figure 2 ‣ 4.1.1 Improving English data quality improves cross-lingual performance ‣ 4.1 The impact of curation on multilingual transfer dynamics and language-specific performance ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset") summarizes performance across 13 languages, reporting average scores over multilingual MMLU, ARC Challenge, and Belebele. English-only curation (light blue bars) yields consistent gains over the uncurated baseline (dark purple bars) in every language except Bengali, indicating that improving English data quality alone can measurably strengthen multilingual capabilities in otherwise uncurated languages.
Averaged across languages, English curation yields a 3.91% relative improvement in non-English performance compared to an uncurated baseline.

Figure 2: Impact of Curation Strategy on Multilingual Performance (bilingual models).
Performance comparison for 3B parameter models trained on 60BT tokens (50:50 English:non-English ratio). Results are averaged across multilingual MMLU, ARC, and Belebele.
Across 13 languages, results show that improved English curation (light blue bars) consistently improves performance over the uncurated baseline (dark purple bars; improvement in 12 of 13 languages), while combining curated English with curated multilingual data (dark blue bars) yields the highest average scores across all languages.

#### 4.1.2 Cross-lingual curation gains correlate with language similarity

While English curation improves performance in the uncurated language for 12 out of 13 examined languages, the magnitude of these benefits is not uniform.
Languages such as Spanish, French, and German, which are linguistically more similar to English, exhibit more pronounced uplifts than
languages such as Hindi and Arabic (8.56% compared to 3.94% relative gains, respectively).
This finding is in line with Longpre et al. ([2025](#bib.bib30 "ATLAS: adaptive transfer scaling laws for multilingual pretraining, finetuning, and decoding the curse of multilinguality")), who report that bilingual transfer (as measured by cross-entropy loss) is predicted by language similarity across varying sampling ratios.

We ask a similar, though distinct question here: what is the relationship between language similarity and the impact of English curation on non-English *model capabilities*.
We consider two heuristic approaches to quantify linguistic distances: similarity
in embedding space and perplexity. Crucially, to ensure these metrics capture linguistic divergence rather than topical shifts in the underlying text, we compute both measures on parallel samples from the FLoRes dataset (Goyal et al., [2022](#bib.bib53 "The flores-101 evaluation benchmark for low-resource and multilingual machine translation")).
For embedding distance, we report the average cosine distance between English and the target language across three distinct models: LaBSE (Feng et al., [2022](#bib.bib8 "Language-agnostic BERT sentence embedding")), e5-small (Wang et al., [2022](#bib.bib10 "Text embeddings by weakly-supervised contrastive pre-training")), and sentence-transformers (Reimers and Gurevych, [2019](#bib.bib9 "Sentence-BERT: sentence embeddings using Siamese BERT-networks")).
Our perplexity proxy is defined as the average log perplexity per word as measured on the target language samples under a model trained exclusively on curated English data. We explicitly do not normalize by word length, allowing this metric to serve as a raw measure of how well English-centric patterns generalize to the target distribution.

Figure [3](#S4.F3 "Figure 3 ‣ 4.1.2 Cross-lingual curation gains correlate with language similarity ‣ 4.1 The impact of curation on multilingual transfer dynamics and language-specific performance ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset") illustrates
a significant negative correlation between
proxy distance metrics and the relative improvement gained through English curation alone. Specifically, embedding
distance yields a Pearson correlation of −0.62-0.62 (p=0.024p=0.024),
while perplexity shows an even stronger correlation of −0.70-0.70 (p=0.018p=0.018).
These results
provide evidence that language similarity, quantified using
two distinct approaches,
is significantly correlated with the
cross-lingual gains from English-only curation.

Figure 3: Correlation between language similarity to English and cross-lingual transfer benefit. We evaluate linguistic distance using two proxies: (a) average log embedding distance across LaBSE, e5-small, and sentence-transformers, and (b) log perplexity of the target language under an English-only model. Both metrics show a significant negative correlation (Pearson r=−0.62r=-0.62 and r=−0.70r=-0.70 respectively) with the performance uplift gained from English data curation. These results demonstrate that linguistically similar languages, such as Spanish and French, receive the most pronounced benefits from high-quality English data, while more distant languages like Bengali and Arabic show significantly lower transfer gains.

#### 4.1.3 Optimal multilingual performance requires bespoke multilingual curation

While curating English consistently improves cross-lingual performance, it is not sufficient to reach optimal performance in any given target language. Figure [2](#S4.F2 "Figure 2 ‣ 4.1.1 Improving English data quality improves cross-lingual performance ‣ 4.1 The impact of curation on multilingual transfer dynamics and language-specific performance ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset") shows a persistent gap in performance when non-English data is curated (dark purple) versus when it is not (purple and light blue).
Moreover, we note that the magnitude of improvements from curating for each language individually far exceeds the benefits associated with only curating English data.
This finding, while not necessarily surprising, further emphasizes the need to pay careful attention to data quality and curation for each language individually.
This observation reinforces the argument by Messmer et al. ([2025](#bib.bib51 "Enhancing multilingual LLM pretraining with model-based data selection")) that generic, English-centric heuristics will not generalize
across diverse alphabets and scripts.
These results are also consistent
with Foroutan et al. ([2025](#bib.bib21 "Revisiting multilingual data mixtures in language model pretraining")), who posited that
it is the presence of noisy, uncurated data which harms multilingual models rather than an issue of model capacity. That is, the issue
“resembles a curse of data quality” rather than a curse of multilinguality.

#### 4.1.4 Improved non-English data curation also benefits English capabilities

Prior sections highlighted
that while English curation improves non-English performance,
optimal non-English performance comes from careful curation of
both English and non-English data.
In this section we study the effect of multilingual data curation on English performance.
Despite many recent findings, we observe that the benefits of data curation
are bidirectional:
our results demonstrate that improving the quality of the non-English data component also yields consistent gains on English benchmarks. Figure [4](#S4.F4 "Figure 4 ‣ 4.1.4 Improved non-English data curation also benefits English capabilities ‣ 4.1 The impact of curation on multilingual transfer dynamics and language-specific performance ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset") compares the performance on English tasks (MMLU and ARC Average) between models trained with uncurated versus curated non-English data, while keeping the English data constant (we use curated English data throughout).
The results demonstrate an average relative improvement of 1.2%. Concretely, in
12 out of the 13 language considered,
the bilingual model trained with fully curated data (i.e., both English and non-English data curated) outperformed the
version with uncurated non-English data.

Figure 4: Non-English Curation Benefits English Performance.
Performance comparison for 3B parameter models trained on 60BT tokens (50:50 English:non-English ratio).
Results are average of English MMLU and ARC. We contrast performance when
when the accompanying multilingual data is uncurated (dark purple) versus curated (dark blue). We observe positive transfer in 12 out of 13 languages, with an overall relative improvement of 1.21%.

At a high-level, these findings suggest that high-quality data acts as a globally beneficial signal in model training, providing a means to mitigate the “curse of multilinguality” by systematically improving the quality of data for each individual language.

### 4.2 The efficacy of translation as augmentation is determined by source quality

Machine translation has increasingly surfaced as a viable strategy for enhancing multilingual model performance,
serving as a scalable source of synthetic data (Seto et al., [2025](#bib.bib42 "Assessing the role of data quality in training bilingual language models"); Wang et al., [2025](#bib.bib52 "Multilingual language model pretraining using machine-translated data")).
Prior efforts, such as FineTranslations (Penedo et al., [2026](#bib.bib22 "FineTranslations")), have successfully utilized large-scale translation pipelines to map multilingual content into English with an explicit focus of improving
translation capabilities. In this work, we instead investigate the
effectiveness of translation as a general tool to drive overall
multilingual capabilities.
We demonstrate that this strategy can indeed drive performance gains; however, its effectiveness is heavily contingent on source data quality. Our results indicate that translating only high-quality documents, selected via score filters, leads to markedly better improvements.

To understand how to best leverage translation
as a tool within multilingual data curation,
we conducted controlled experiments on three languages: Hindi, Bengali, and Arabic.
We trained 3B parameter models for 60B tokens, keeping the English component fixed and curated, while varying the non-English strategy.
We compared three approaches: (1) an uncurated baseline, (2) augmenting the baseline with translations of randomly selected English data (i.e., blind translations), and (3) augmenting with translations of high-quality, scored English data. When scoring English data, we used a fasttext classifier similar to Penedo et al. ([2024](#bib.bib70 "The fineweb datasets: decanting the web for the finest text data at scale")).

Figure 5: Evaluation of benefits associated with Random vs Scored Translation for Low-Resource Languages. Performance curves for Hindi, Bengali and Arabic showing that while augmenting training data with translated English text (red and cyan lines) improves over the uncurated baseline (dark gray), it still falls short of the performance achieved by bespoke DatologyAI curation (dark blue).

The results, illustrated in Figure [5](#S4.F5 "Figure 5 ‣ 4.2 The efficacy of translation as augmentation is determined by source quality ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"), reveal a clear hierarchy of performance. Augmenting the uncurated baseline with
arbitarily translated English data
only yields marginal performance improvements over a purely
uncurated baseline. However, the magnitude of improvements grows when translating high quality, score-filtered data. This
result mirrors findings in DatologyAI ([2025](#bib.bib137 "BeyondWeb: lessons from scaling synthetic data for trillion-scale pretraining")), which showed that the quality of input documents for synthetic rephrasing is crucial to obtaining strong performance.

However, a significant performance gap remains. Our bespoke curation strategy (dark blue line) substantially outperforms both the uncurated baseline and the translation-augmented models.
These findings imply that while translation
can be a valuable component of multilingual data curation, as reported by Wang et al. ([2025](#bib.bib52 "Multilingual language model pretraining using machine-translated data")), its efficacy is ultimately determined by source data quality, and the best performance is obtained with a holistic curation approach across all target languages.

### 4.3 Integrating multilingual curation into a general pretraining mix

Sections [4.1](#S4.SS1 "4.1 The impact of curation on multilingual transfer dynamics and language-specific performance ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset") and [4.2](#S4.SS2 "4.2 The efficacy of translation as augmentation is determined by source quality ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset") presented smaller-scale, controlled experiments
intended to dissociate the impact of different multilingual
curation choices.
However, an open question is how such
curation strategies can scale
to larger token budgets and models, and how multilingual curation
interacts with general purpose
curation. To that end,
we curated a 20T token dataset intended for foundation model training and frontier capabilities across English, multilingual, code, STEM, and reasoning skills.
The curation included generating over 8T tokens of synthetic English and non-English web data, and code and STEM data.
A random 1T subset of this
dataset was used to train both 3B and 8B parameter models following a Llama architecture.

Multi-Phase Data Curriculum.
To balance multiple diverse data streams, we implemented a multi-phase training curriculum that progressively increases the density of multilingual tokens.
The mixture of tokens across three phases followed that used in the
Trinity Large model (Singh et al., [2026](#bib.bib39 "Arcee trinity large technical report"), Section 3.1).
The training process was divided into three distinct phases:

* •

  Phase 1: 650B tokens with 5% multilingual data
* •

  Phase 2: 250B tokens with 10% multilingual data
* •

  Phase 3: 100B tokens with 20% multilingual data

Across the full training duration, this schedule resulted
in an overall allocation of 7.75% tokens to our multilingual curation pipeline, supporting
13 languages and thus resulting in an average of 6B tokens per langauge.
Despite this seemingly modest multilingual budget, our bespoke curation strategy allows
models to obtain competitive performance across
diverse languages spanning Latin, Cyrillic, Arabic, Indic, and CJK scripts.

##### Establishing a New Pareto Frontier.

[Figure 1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset") positions DatologyAI models
against several
open-weights models across
English (MMLU + ARC Average) and three multilingual benchmarks (Multilingual MMLU, ARC, and Belebele).
The y-axis denotes the error rate (1−Average Accuracy1-\text{Average Accuracy}) in log scale, where lower values indicate superior capabilities. The shaded gray region encapsulates the performance-compute trade-off established by leading open-weights baselines,
including
Qwen3 (Yang et al., [2025](#bib.bib130 "Qwen3 technical report")), Granite (IBM Granite Team, [2024](#bib.bib5 "Granite 3.0 language models")), SmolLM3 (Bakouch et al., [2025](#bib.bib6 "SmolLM3: smol, multilingual, long-context reasoner")), LFM-2.5 (Liquid AI, [2026](#bib.bib142 "Introducing LFM2.5: The next generation of on-device AI")),
and Tiny Aya (Cohere Labs, [2026](#bib.bib38 "Tiny aya technical report")). Our results demonstrate a marked shift in efficiency: the
DatologyAI models
consistently improve upon the established Pareto frontier. By
achieving error rates comparable to significantly larger and/or more compute-intensive baselines, we effectively redefine the Pareto frontier for multilingual foundational models.

##### Data curation unlocks token efficient data mixtures.

We emphasize that the DatologyAI models in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset") use the smallest multilingual data mixture among all models that report mixture composition. Specifically, DatologyAI allocates only 7.75%
of training tokens to multilingual data while supporting a substantially
broader set of languages. This proportion is markedly lower than those reported by comparable baselines, including LFM, which uses 20% multilingual tokens (Liquid AI, [2026](#bib.bib142 "Introducing LFM2.5: The next generation of on-device AI")), and SmolLM3, which employs a 12% multilingual mixture (Bakouch et al., [2025](#bib.bib6 "SmolLM3: smol, multilingual, long-context reasoner")).

In Appendix [A.3](#A1.SS3 "A.3 Per language evaluation performance ‣ Appendix A Appendix ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"), we report per-language performance breakdowns for each evaluation. We present
results across three groups of languages:
Latin-script languages (Figure [6](#A1.F6 "Figure 6 ‣ A.3 Per language evaluation performance ‣ Appendix A Appendix ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset")), Indic and Arabic (Figure [7](#A1.F7 "Figure 7 ‣ A.3 Per language evaluation performance ‣ Appendix A Appendix ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset")), and Chinese, Japanese, Korean, and Russian (Figure [8](#A1.F8 "Figure 8 ‣ A.3 Per language evaluation performance ‣ Appendix A Appendix ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset")). Across all 13 languages, DatologyAI’s multilingual curation consistently improves the performance–compute Pareto frontier, yielding higher accuracy at a given FLOP budget (or comparable accuracy with less compute).
We also present results comparing to various language-specialized base models, i.e.
models
that have a focus on achieving strong performance on particular languages.
Examples include the
Sarvam-1 model (Sarvam AI, [2024](#bib.bib97 "Sarvam 1: the first indian language llm")), focused on Indic languages, in Figure [7](#A1.F7 "Figure 7 ‣ A.3 Per language evaluation performance ‣ Appendix A Appendix ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"); Trillion Labs Tri-7B (Trillion Labs, [2025](#bib.bib141 "Tri-7b-base")), focused on Korean, Japanese, and Chinese, in Figure [8](#A1.F8 "Figure 8 ‣ A.3 Per language evaluation performance ‣ Appendix A Appendix ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"); and SEA-LION-v3-9B
(Ng et al., [2025](#bib.bib98 "SEA-lion: southeast asian languages in one network")), focused on Southeast Asian languages, also in Figure [8](#A1.F8 "Figure 8 ‣ A.3 Per language evaluation performance ‣ Appendix A Appendix ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
The specialized models also significantly improve upon the Pareto frontier, but their performance is comparable to that of the DatologyAI models on the particular languages they focus on; for example, Figures [7](#A1.F7 "Figure 7 ‣ A.3 Per language evaluation performance ‣ Appendix A Appendix ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset")
and Figures [8](#A1.F8 "Figure 8 ‣ A.3 Per language evaluation performance ‣ Appendix A Appendix ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset")
show that DatologyAI models can meet or exceed performance
of specialized models such as Sarvam-1 and Tri-7B, which are trained using similar or larger FLOPs budgets.

The rightmost columns in Figures [6](#A1.F6 "Figure 6 ‣ A.3 Per language evaluation performance ‣ Appendix A Appendix ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset")–[8](#A1.F8 "Figure 8 ‣ A.3 Per language evaluation performance ‣ Appendix A Appendix ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset") illustrate the relationship between language-specific performance and aggregate multilingual proficiency. DatologyAI models consistently align with the line of unity, reflecting a data curation strategy
that prioritizes broad multilingual parity over individual language optimization. In contrast, specialized models like Sarvam-1 and Tri-7B exhibit a clear departure from this trend, appearing above the line of unity for their target languages. However,
their aggregate multilingual performance (shown along x-axis) reveals a
substantial degradation in overall capabilities. This highlights that
these models have traded general multilingualism for localized expertise. Notably, models curated with DatologyAI achieve competitive results without necessitating such performance tradeoffs. Finally, Figure [9](#A1.F9 "Figure 9 ‣ A.4 Multilingual data efficiency gains ‣ Appendix A Appendix ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset") visualizes the performance on
various individual languages as a function of the number of training tokens
in that language
for DatologyAI models and the subset of the models we evaluated where we could obtain reasonable estimates for the per-language training tokens
(we describe our methodology in Appendix [A.4](#A1.SS4 "A.4 Multilingual data efficiency gains ‣ Appendix A Appendix ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset")). This figure clearly visualizes the orders-of-magnitude improvements in
per-language data efficiency obtained with DatologyAI curation.

Taken together, the results in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset") and
Figures [6](#A1.F6 "Figure 6 ‣ A.3 Per language evaluation performance ‣ Appendix A Appendix ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset")–[9](#A1.F9 "Figure 9 ‣ A.4 Multilingual data efficiency gains ‣ Appendix A Appendix ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset")
demonstrate that DatologyAI multilingual data curation is both highly effective
and scales to the frontier model training regime.
The latter point is reinforced by results from Trinity Large, which was pretrained on 17T tokens drawn from the broader DatologyAI-curated corpus and exhibits exceptionally strong multilingual performance.

## 5 Conclusion

Multilinguality is an essential capability for modern foundation models, yet achieving high-quality multilingual performance with broad coverage remains challenging due to uneven data availability across languages and the so-called “curse of multilinguality”.
In this work, we revisit multilingual pretraining from a data-centric perspective and show that many of observed constraints and regressions are not inherent to multilinguality, but instead reflect deficiencies in data quality and curation.

Through controlled bilingual experiments, we demonstrate that cross-lingual transfer is strongly mediated by data quality: improving English curation alone yields consistent gains across nearly all non-English languages examined (3.91% average relative improvement across multilingual MMLU, ARC-Challenge, and Belebele), while improving non-English curation reciprocally benefits English performance (1.21% average relative improvement).
These findings challenge
a purely zero-sum framing of multilingual modeling: higher-quality training data can strengthen multilingual capability without requiring commensurate sacrifices elsewhere in the mixture.

English curation alone, however, is insufficient for optimal performance. Bespoke, per-language pipelines tailored to linguistic and distributional properties deliver substantially larger gains, reaching 16.87% relative improvement in controlled settings. We further show that translation is most effective when it preserves source quality: translating score-filtered English documents yields materially larger gains than translating arbitrary text, and integrating high-quality document translation as part of a holistic multilingual curation strategy yields far superior results overall.

We productionized these principles through a 20T-token general-purpose pretraining corpus, whose multilingual component was constructed using the curation strategies explored here. Under a controlled 1T-token training budget, 3B and 8B models achieve comparable or stronger multilingual performance than competitive open-weight baselines at 4–10× lower training compute, redefining the multilingual performance–compute Pareto frontier. These efficiency gains persist at frontier scale: Trinity Large Base (400B/A13B), trained on 17T tokens of this corpus, exhibits exceptionally strong multilingual performance relative to its FLOPs budget, validating that the curation principles described here remain effective in the multi–tens-of-trillions regime. We emphasize that for both the 1T-token training budget experiments as well as for Trinity Large, the multilingual performance is obtained using a comparatively minor multilingual token budget of 7.75% of total training tokens.

Several avenues for future work follow. Our results motivate more systematic, compute-aware mixture design, including per-language sampling strategies and phased curricula that balance improvements in one language against interference in others while ensuring adequate support for low-resource languages. Scaling this agenda will likely require more robust multilingual evaluation frameworks (Liang et al., [2022](#bib.bib99 "Holistic evaluation of language models")). Finally, extending these data-centric principles to multimodal and vision–language model (VLM) training remains an important direction, where evaluation quality and coverage are also central bottlenecks (Joshi et al., [2026](#bib.bib40 "DatBench: discriminative, faithful, and efficient vlm evaluations")).

In conclusion, viewed through the data-centric lens advanced in this work, multilinguality need not be a curse of scale, but instead an opportunity to leverage language-aware curation to achieve inclusive, capable foundation models.

## 6 Contributions and Acknowledgements

Core and technical contributors listed alphabetically.

|  |  |
| --- | --- |
| Core Contributors | Aldo Gael Carranza, Kaleigh Mentzer, and Ricardo Pio Monti |
| Technical Contributors | Alex Fang, Alvin Deng, Amro Abbas, Anshuman Suri, Brett Larsen, Cody Blakeney, Darren Teh, David Schwab, Diego Kiner, Fan Pan, Haakon Mongstad, Haoli Yin, Jack Urbanek, Jason Lee, Jason Telanoff, Josh Wills, Luke Merrick, Maximilian Böther, Parth Doshi, Paul Burstein, Pratyush Maini, Rishabh Adiga, Spandan Das, Siddharth Joshi, Tony Jiang, Vineeth Dorna, and Zhengping Wang |
| Leadership | Bogdan Gaza, Ari Morcos, and Matthew Leavitt |
| Acknowledgements | Liz Gatapia, Jacqueline Liu, Tiffanie Pham, Sylvia Hoang, Kylie Clement, Elise Clark |

## References

* S. Ahuja, D. Aggarwal, V. Gumma, I. Watts, A. Sathe, M. Ochieng, R. Hada, P. Jain, M. Ahmed, K. Bali, et al. (2024)
  Megaverse: benchmarking large language models across languages, modalities, models and tasks.
  In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers),
   pp. 2598–2637.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* E. Bakouch, L. Ben Allal, A. Lozhkov, N. Tazi, L. Tunstall, C. M. Patiño, E. Beeching, A. Roucher, A. J. Reedi, Q. Gallouédec, K. Rasul, N. Habib, C. Fourrier, H. Kydlicek, G. Penedo, H. Larcher, M. Morlon, V. Srivastav, J. Lochner, X. Nguyen, C. Raffel, L. von Werra, and T. Wolf (2025)
  SmolLM3: smol, multilingual, long-context reasoner.
  Note: <https://huggingface.co/blog/smollm3>
  Cited by: [item 4](#S1.I1.i4.p1.5 "In 1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§4.3](#S4.SS3.SSS0.Px1.p1.1 "Establishing a New Pareto Frontier. ‣ 4.3 Integrating multilingual curation into a general pretraining mix ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§4.3](#S4.SS3.SSS0.Px2.p1.1 "Data curation unlocks token efficient data mixtures. ‣ 4.3 Integrating multilingual curation into a general pretraining mix ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* L. Bandarkar, D. Liang, B. Muller, M. Artetxe, S. N. Shukla, D. Husa, N. Goyal, A. Krishnan, L. Zettlemoyer, and M. Khabsa (2024)
  The belebele benchmark: a parallel reading comprehension dataset in 122 language variants.
  In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
   pp. 749–775.
  Cited by: [§A.1](#A1.SS1.p1.1 "A.1 Evaluation datasets per language ‣ Appendix A Appendix ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [3rd item](#S3.I1.i3.p1.1.1 "In 3 Experimental Setup and Methodology ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* T. Blevins, T. Limisiewicz, S. Gururangan, M. Li, H. Gonen, N. A. Smith, and L. Zettlemoyer (2024)
  Breaking the curse of multilinguality with cross-lingual expert language models.
  In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing,
   pp. 10822–10837.
  External Links: [Link](https://aclanthology.org/2024.emnlp-main.604/)
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* T. F. Burns, L. Parcalabescu, S. Wäldchen, M. Barlow, G. Ziegltrum, V. Stampa, B. Harren, and B. Deiseroth (2025)
  Aleph-alpha-germanweb: improving german-language llm pre-training with model-based data curation and synthetic data generation.
  External Links: 2505.00022,
  [Link](https://arxiv.org/abs/2505.00022)
  Cited by: [§2](#S2.SS0.SSS0.Px1.p2.1 "Multilingual Data Curation. ‣ 2 Related Work ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* T. A. Chang, C. Arnett, Z. Tu, and B. K. Bergen (2024)
  When is multilinguality a curse? language modeling for 250 high- and low-resource languages.
  In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing,
   pp. 4074–4096.
  External Links: [Link](https://aclanthology.org/2024.emnlp-main.236/)
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* Z. Chen, P. Guo, W. Han, Y. Zhang, H. Lin, F. Liu, Y. Zhao, B. Zhang, T. Wang, Y. Zheng, T. Cohn, and M. Fang (2025)
  MuRating: a high quality data selecting approach to multilingual large language model pretraining.
  In The Thirty-ninth Annual Conference on Neural Information Processing Systems,
  Cited by: [§2](#S2.SS0.SSS0.Px1.p2.1 "Multilingual Data Curation. ‣ 2 Related Work ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* D. Choi, D. Xin, H. Dadkhahi, J. Gilmer, A. Garg, O. Firat, C. Yeh, A. M. Dai, and B. Ghorbani (2023)
  Order matters in the presence of dataset imbalance for multilingual learning.
  Advances in Neural Information Processing Systems 36,  pp. 66902–66922.
  Cited by: [§2](#S2.SS0.SSS0.Px2.p1.1 "Data Mixing and Interference. ‣ 2 Related Work ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* H. W. Chung, X. Garcia, A. Roberts, Y. Tay, O. Firat, S. Narang, and N. Constant (2023)
  UniMax: fairer and more effective language sampling for large-scale multilingual pretraining.
  In The Eleventh International Conference on Learning Representations,
  Cited by: [§2](#S2.SS0.SSS0.Px2.p1.1 "Data Mixing and Interference. ‣ 2 Related Work ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* Cohere Labs (2026)
  Tiny aya technical report.
  Technical report
   Cohere Labs / Cohere AI.
  Note: Technical report, available at <https://github.com/Cohere-Labs/tiny-aya-tech-report/blob/main/tiny_aya_tech_report.pdf>
  External Links: [Link](https://github.com/Cohere-Labs/tiny-aya-tech-report/blob/main/tiny_aya_tech_report.pdf)
  Cited by: [§4.3](#S4.SS3.SSS0.Px1.p1.1 "Establishing a New Pareto Frontier. ‣ 4.3 Integrating multilingual curation into a general pretraining mix ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* A. Conneau, K. Khandelwal, N. Goyal, V. Chaudhary, G. Wenzek, F. Guzmán, E. Grave, M. Ott, L. Zettlemoyer, and V. Stoyanov (2020)
  Unsupervised cross-lingual representation learning at scale.
  In Proceedings of the 58th annual meeting of the association for computational linguistics,
   pp. 8440–8451.
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§2](#S2.SS0.SSS0.Px2.p1.1 "Data Mixing and Interference. ‣ 2 Related Work ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* A. Conneau and G. Lample (2019)
  Cross-lingual language model pretraining.
  In Advances in Neural Information Processing Systems (NeurIPS),
   pp. 7059–7069.
  External Links: [Link](https://proceedings.neurips.cc/paper/2019/hash/c04c19c2c2474dbf5f7ac4372c5b9af1-Abstract.html)
  Cited by: [§2](#S2.SS0.SSS0.Px2.p1.1 "Data Mixing and Interference. ‣ 2 Related Work ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* DatologyAI (2024)
  DatologyAI technical deep-dive: curating our way to a billion-state-of-the-art text dataset.
  Technical report
   DatologyAI.
  Note: Contributors: Aldo Carranza, Alvin Deng, Pratyush Maini, Muhammed Razzak, Jack Urbanek, Amro Abbas, Paul Burstein, Ning Cao, Priya Goyal, Joshua McGrath, Fan Pan, Josh Wills, Haoli Yin, Vineeth Kada, Vishwa Shah, Vishruth Veerendranath, Bogdan Gaza, Ari Morcos, Matthew Leavitt.
  External Links: [Link](https://www.datologyai.com/blog/technical-deep-dive-curating-our-way-to-a-state-of-the-art-text-dataset)
  Cited by: [§3](#S3.p3.1 "3 Experimental Setup and Methodology ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* DatologyAI (2025)
  BeyondWeb: lessons from scaling synthetic data for trillion-scale pretraining.
  Note: Author list: Pratyush Maini, Vineeth Dorna, Parth Doshi, Aldo Carranza, Fan Pan, Jack Urbanek, Paul Burstein, Alex Fang, Alvin Deng, Amro Abbas, Brett Larsen, Cody Blakeney, Charvi Bannur, Christina Baek, Darren Teh, David Schwab, Haakon Mongstad, Haoli Yin, Josh Wills, Kaleigh Mentzer, Luke Merrick, Ricardo Monti, Rishabh Adiga, Siddharth Joshi, Spandan Das, Zhengping Wang, Bogdan Gaza, Ari Morcos, Matthew Leavitt.
  External Links: 2508.10975,
  [Link](https://arxiv.org/abs/2508.10975)
  Cited by: [§3](#S3.p3.1 "3 Experimental Setup and Methodology ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§4.2](#S4.SS2.p3.1 "4.2 The efficacy of translation as augmentation is determined by source quality ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* F. Feng, Y. Yang, D. Cer, N. Arivazhagan, and W. Wang (2022)
  Language-agnostic BERT sentence embedding.
  In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
   pp. 870–883.
  Cited by: [§4.1.2](#S4.SS1.SSS2.p2.1 "4.1.2 Cross-lingual curation gains correlate with language similarity ‣ 4.1 The impact of curation on multilingual transfer dynamics and language-specific performance ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* P. Fernandes, B. Ghorbani, X. Garcia, M. Freitag, and O. Firat (2023)
  Scaling laws for multilingual neural machine translation.
  In International Conference on Machine Learning,
   pp. 10053–10071.
  Cited by: [§2](#S2.SS0.SSS0.Px3.p1.1 "Multilingual Scaling Laws. ‣ 2 Related Work ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* N. Foroutan, P. Teiletche, A. K. Tarun, and A. Bosselut (2025)
  Revisiting multilingual data mixtures in language model pretraining.
  arXiv preprint arXiv:2510.25947.
  External Links: [Link](https://arxiv.org/abs/2510.25947)
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§2](#S2.SS0.SSS0.Px2.p1.1 "Data Mixing and Interference. ‣ 2 Related Work ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§4.1.3](#S4.SS1.SSS3.p1.1 "4.1.3 Optimal multilingual performance requires bespoke multilingual curation ‣ 4.1 The impact of curation on multilingual transfer dynamics and language-specific performance ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* N. Goyal, C. Gao, V. Chaudhary, P. Chen, G. Wenzek, D. Ju, S. Krishnan, M. Ranzato, F. Guzmán, and A. Fan (2022)
  The flores-101 evaluation benchmark for low-resource and multilingual machine translation.
  Transactions of the Association for Computational Linguistics 10,  pp. 522–538.
  Cited by: [§4.1.2](#S4.SS1.SSS2.p2.1 "4.1.2 Cross-lingual curation gains correlate with language similarity ‣ 4.1 The impact of curation on multilingual transfer dynamics and language-specific performance ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* Y. Gu, O. Tafjord, B. Kuehl, D. Haddad, J. Dodge, and H. Hajishirzi (2025)
  Olmes: a standard for language model evaluations.
  In Findings of the Association for Computational Linguistics: NAACL 2025,
   pp. 5005–5033.
  Cited by: [§3](#S3.p5.2 "3 Experimental Setup and Methodology ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* N. Habib, C. Fourrier, H. Kydlíček, T. Wolf, and L. Tunstall (2023)
  LightEval: a lightweight framework for llm evaluation.
  External Links: [Link](https://github.com/huggingface/lighteval)
  Cited by: [§3](#S3.p5.2 "3 Experimental Setup and Methodology ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* Y. He, A. Benhaim, B. Patra, P. Vaddamanu, S. Ahuja, P. Chopra, V. Chaudhary, H. Zhao, and X. Song (2025)
  Scaling laws for multilingual language models.
  In Findings of the Association for Computational Linguistics: ACL 2025,
   pp. 4257–4273.
  Cited by: [§2](#S2.SS0.SSS0.Px3.p1.1 "Multilingual Scaling Laws. ‣ 2 Related Work ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* J. Hestness, S. Narang, N. Ardalani, G. Diamos, H. Jun, H. Kianinejad, M. M. A. Patwary, Y. Yang, and Y. Zhou (2017)
  Deep learning scaling is predictable, empirically.
  arXiv preprint arXiv:1712.00409.
  External Links: [Link](https://arxiv.org/abs/1712.00409)
  Cited by: [§2](#S2.SS0.SSS0.Px3.p1.1 "Multilingual Scaling Laws. ‣ 2 Related Work ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas, L. A. Hendricks, J. Welbl, A. Clark, et al. (2022)
  Training compute-optimal large language models.
  In Proceedings of the 36th International Conference on Neural Information Processing Systems,
   pp. 30016–30030.
  Cited by: [§2](#S2.SS0.SSS0.Px3.p1.1 "Multilingual Scaling Laws. ‣ 2 Related Work ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* IBM Granite Team (2024)
  Granite 3.0 language models.
  Note: Technical reportAccessed 2026-01-12
  External Links: [Link](https://github.com/ibm-granite/granite-3.0-language-models)
  Cited by: [item 4](#S1.I1.i4.p1.5 "In 1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§4.3](#S4.SS3.SSS0.Px1.p1.1 "Establishing a New Pareto Frontier. ‣ 4.3 Integrating multilingual curation into a general pretraining mix ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* S. Joshi, H. Yin, R. Adiga, R. Monti, A. Carranza, A. Fang, A. Deng, A. Abbas, B. Larsen, C. Blakeney, et al. (2026)
  DatBench: discriminative, faithful, and efficient vlm evaluations.
  arXiv preprint arXiv:2601.02316.
  Cited by: [§5](#S5.p5.1 "5 Conclusion ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei (2020)
  Scaling laws for neural language models.
  arXiv preprint arXiv:2001.08361.
  External Links: [Link](https://arxiv.org/abs/2001.08361)
  Cited by: [§2](#S2.SS0.SSS0.Px3.p1.1 "Multilingual Scaling Laws. ‣ 2 Related Work ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* M. Khan, P. Mehta, A. Sankar, U. Kumaravelan, S. Doddapaneni, S. Jain, A. Kunchukuttan, P. Kumar, R. Dabre, M. M. Khapra, et al. (2024)
  IndicLLMSuite: a blueprint for creating pre-training and fine-tuning datasets for indian languages.
  In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
   pp. 15831–15879.
  Cited by: [§2](#S2.SS0.SSS0.Px1.p2.1 "Multilingual Data Curation. ‣ 2 Related Work ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* S. Khanna and X. Li (2025)
  Invisible languages of the llm universe.
  arXiv preprint arXiv:2510.11557.
  External Links: [Link](https://arxiv.org/abs/2510.11557)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* V. Lai, C. Nguyen, N. Ngo, T. Nguyen, F. Dernoncourt, R. Rossi, and T. Nguyen (2023)
  Okapi: instruction-tuned large language models in multiple languages with reinforcement learning from human feedback.
  In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations,
   pp. 318–327.
  Cited by: [§A.1](#A1.SS1.p1.1 "A.1 Evaluation datasets per language ‣ Appendix A Appendix ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [2nd item](#S3.I1.i2.p1.1.1 "In 3 Experimental Setup and Methodology ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* J. Li, A. Fang, G. Smyrnis, M. Ivgi, M. Jordan, S. Y. Gadre, H. Bansal, E. Guha, S. S. Keh, K. Arora, et al. (2024)
  Datacomp-lm: in search of the next generation of training sets for language models.
  Advances in Neural Information Processing Systems 37,  pp. 14200–14282.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§3](#S3.p1.1 "3 Experimental Setup and Methodology ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§3](#S3.p5.2 "3 Experimental Setup and Methodology ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* P. Liang, R. Bommasani, T. Lee, D. Tsipras, D. Soylu, M. Yasunaga, Y. Zhang, D. Narayanan, Y. Wu, A. Kumar, et al. (2022)
  Holistic evaluation of language models.
  External Links: 2211.09110
  Cited by: [§5](#S5.p5.1 "5 Conclusion ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* Liquid AI (2026)
  Introducing LFM2.5: The next generation of on-device AI.
  Note: Liquid AI BlogAccessed: February 8, 2026
  External Links: [Link](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai)
  Cited by: [item 4](#S1.I1.i4.p1.5 "In 1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§4.3](#S4.SS3.SSS0.Px1.p1.1 "Establishing a New Pareto Frontier. ‣ 4.3 Integrating multilingual curation into a general pretraining mix ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§4.3](#S4.SS3.SSS0.Px2.p1.1 "Data curation unlocks token efficient data mixtures. ‣ 4.3 Integrating multilingual curation into a general pretraining mix ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* S. Longpre, S. Kudugunta, N. Muennighoff, I. Hsu, I. Caswell, A. Pentland, S. Arik, C. Lee, and S. Ebrahimi (2025)
  ATLAS: adaptive transfer scaling laws for multilingual pretraining, finetuning, and decoding the curse of multilinguality.
  arXiv preprint arXiv:2510.22037.
  External Links: [Link](https://arxiv.org/abs/2510.22037)
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§2](#S2.SS0.SSS0.Px3.p1.1 "Multilingual Scaling Laws. ‣ 2 Related Work ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§4.1.2](#S4.SS1.SSS2.p1.1 "4.1.2 Cross-lingual curation gains correlate with language similarity ‣ 4.1 The impact of curation on multilingual transfer dynamics and language-specific performance ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* B. Messmer, V. Sabolčec, and M. Jaggi (2025)
  Enhancing multilingual LLM pretraining with model-based data selection.
  In Proceedings of the 10th edition of the Swiss Text Analytics Conference, J. Gerber, M. Cieliebak, D. Tuggener, and M. Hürlimann (Eds.),
  Winterthur, Switzerland,  pp. 31–56.
  External Links: [Link](https://aclanthology.org/2025.swisstext-1.4/)
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§2](#S2.SS0.SSS0.Px1.p2.1 "Multilingual Data Curation. ‣ 2 Related Work ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§4.1.3](#S4.SS1.SSS3.p1.1 "4.1.3 Optimal multilingual performance requires bespoke multilingual curation ‣ 4.1 The impact of curation on multilingual transfer dynamics and language-specific performance ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* R. Ng, T. N. Nguyen, Y. Huang, N. C. Tai, W. Y. Leong, W. Q. Leong, X. Yong, J. G. Ngui, Y. Susanto, N. Cheng, H. Rengarajan, and P. Limkonchotiwat (2025)
  SEA-lion: southeast asian languages in one network.
  External Links: 2504.05747,
  [Link](https://arxiv.org/abs/2504.05747)
  Cited by: [§4.3](#S4.SS3.SSS0.Px2.p2.1 "Data curation unlocks token efficient data mixtures. ‣ 4.3 Integrating multilingual curation into a general pretraining mix ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* T. Olmo, A. Ettinger, A. Bertsch, B. Kuehl, D. Graham, D. Heineman, D. Groeneveld, F. Brahman, F. Timbers, H. Ivison, et al. (2025)
  Olmo 3.
  arXiv preprint arXiv:2512.13961.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* P. J. Ortiz Suárez, L. Romary, and B. Sagot (2020)
  A monolingual approach to contextualized word embeddings for mid-resource languages.
  In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics,
   pp. 1703–1714.
  External Links: [Link](https://aclanthology.org/2020.acl-main.156/)
  Cited by: [§2](#S2.SS0.SSS0.Px1.p1.1 "Multilingual Data Curation. ‣ 2 Related Work ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* G. Penedo, H. Kydlíček, A. H. Kargaran, and L. von Werra (2026)
  FineTranslations.
   Hugging Face.
  Note: <https://huggingface.co/datasets/HuggingFaceFW/finetranslations>
  Cited by: [item 3](#S1.I1.i3.p1.1 "In 1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§4.2](#S4.SS2.p1.1 "4.2 The efficacy of translation as augmentation is determined by source quality ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* G. Penedo, H. Kydlíček, A. Lozhkov, M. Mitchell, C. A. Raffel, L. Von Werra, T. Wolf, et al. (2024)
  The fineweb datasets: decanting the web for the finest text data at scale.
  Advances in Neural Information Processing Systems 37,  pp. 30811–30849.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Multilingual Data Curation. ‣ 2 Related Work ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§3](#S3.p1.1 "3 Experimental Setup and Methodology ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§4.2](#S4.SS2.p2.1 "4.2 The efficacy of translation as augmentation is determined by source quality ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* G. Penedo, H. Kydlíček, V. Sabolčec, B. Messmer, N. Foroutan, A. H. Kargaran, C. Raffel, M. Jaggi, L. V. Werra, and T. Wolf (2025)
  FineWeb2: one pipeline to scale them all — adapting pre-training data processing to every language.
  In Second Conference on Language Modeling,
  External Links: [Link](https://openreview.net/forum?id=jnRBe6zatP)
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Multilingual Data Curation. ‣ 2 Related Work ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§3](#S3.p1.1 "3 Experimental Setup and Methodology ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* J. Pfeiffer, N. Goyal, X. V. Lin, X. Li, J. Cross, S. Riedel, and M. Artetxe (2022)
  Lifting the curse of multilinguality by pre-training modular transformers.
  In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies,
   pp. 3479–3495.
  External Links: [Link](https://aclanthology.org/2022.naacl-main.255/)
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* T. Pires, E. Schlinger, and D. Garrette (2019)
  How multilingual is multilingual bert?.
  arXiv preprint arXiv:1906.01502.
  Cited by: [§2](#S2.SS0.SSS0.Px2.p1.1 "Data Mixing and Interference. ‣ 2 Related Work ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* N. Reimers and I. Gurevych (2019)
  Sentence-BERT: sentence embeddings using Siamese BERT-networks.
  In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP),
   pp. 3982–3992.
  Cited by: [§4.1.2](#S4.SS1.SSS2.p2.1 "4.1.2 Cross-lingual curation gains correlate with language similarity ‣ 4.1 The impact of curation on multilingual transfer dynamics and language-specific performance ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* Sarvam AI (2024)
  Sarvam 1: the first indian language llm.
  Note: Blog postAccessed: 2026-02-09
  External Links: [Link](https://www.sarvam.ai/blogs/sarvam-1)
  Cited by: [§4.3](#S4.SS3.SSS0.Px2.p2.1 "Data curation unlocks token efficient data mixtures. ‣ 4.3 Integrating multilingual curation into a general pretraining mix ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* S. Seto, M. Ter Hoeve, M. de Seyssel, and D. Grangier (2025)
  Assessing the role of data quality in training bilingual language models.
  In Findings of the Association for Computational Linguistics: EMNLP 2025,
   pp. 22694–22720.
  External Links: [Link](https://aclanthology.org/2025.findings-emnlp.1236/)
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§4.2](#S4.SS2.p1.1 "4.2 The efficacy of translation as augmentation is determined by source quality ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* S. Singh, A. Romanou, C. Fourrier, D. I. Adelani, J. G. Ngui, D. Vila-Suero, P. Limkonchotiwat, K. Marchisio, W. Q. Leong, Y. Susanto, et al. (2025)
  Global mmlu: understanding and addressing cultural and linguistic biases in multilingual evaluation.
  In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
   pp. 18761–18799.
  Cited by: [§A.1](#A1.SS1.p1.1 "A.1 Evaluation datasets per language ‣ Appendix A Appendix ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [1st item](#S3.I1.i1.p1.1.1 "In 3 Experimental Setup and Methodology ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* S. Singh, F. Vargus, D. Dsouza, B. F. Karlsson, A. Mahendiran, W. Ko, H. Shandilya, J. Patel, D. Mataciunas, L. O’Mahony, et al. (2024)
  Aya dataset: an open-access collection for multilingual instruction tuning.
  In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
   pp. 11521–11567.
  External Links: [Link](https://aclanthology.org/2024.acl-long.620)
  Cited by: [§2](#S2.SS0.SSS0.Px1.p1.1 "Multilingual Data Curation. ‣ 2 Related Work ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* V. Singh, L. Krauss, S. Jaghouar, M. Sirovatka, C. Goddard, F. Obeid, J. M. Ong, J. Straube, Fern, A. Harley, et al. (2026)
  Arcee trinity large technical report.
  Technical Report
   Arcee AI and Prime Intellect.
  Note: Accessed 2026-02-02
  External Links: [Link](https://github.com/arcee-ai/trinity-large-tech-report/blob/main/Arcee%20Trinity%20Large.pdf)
  Cited by: [item 4](#S1.I1.i4.p1.5 "In 1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§4.3](#S4.SS3.p2.1 "4.3 Integrating multilingual curation into a general pretraining mix ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* D. Su, K. Kong, Y. Lin, J. Jennings, B. Norick, M. Kliegl, M. Patwary, M. Shoeybi, and B. Catanzaro (2025)
  Nemotron-cc: transforming common crawl into a refined long-horizon pretraining dataset.
  In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
   pp. 2459–2475.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§3](#S3.p1.1 "3 Experimental Setup and Methodology ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* Thunder Research Group (2025)
  Korean benchmarks.
  Note: <https://github.com/mcrl/korean_benchmarks>GitHub repository
  Cited by: [§A.1](#A1.SS1.p1.1 "A.1 Evaluation datasets per language ‣ Appendix A Appendix ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* H. Touvron, T. Lavril, G. Izacard, X. Martinet, M. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. (2023)
  Llama: open and efficient foundation language models.
  arXiv preprint arXiv:2302.13971.
  Cited by: [§3](#S3.p4.1 "3 Experimental Setup and Methodology ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* Trillion Labs (2025)
  Tri-7b-base.
  Note: Hugging Face model cardAccessed: 2026-02-09. License: Apache-2.0. A 7.76B-parameter pretrained causal LM (Korean/English/Japanese), 4096 context.
  External Links: [Link](https://huggingface.co/trillionlabs/Tri-7B-Base)
  Cited by: [§4.3](#S4.SS3.SSS0.Px2.p2.1 "Data curation unlocks token efficient data mixtures. ‣ 4.3 Integrating multilingual curation into a general pretraining mix ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* J. Wang, Y. Lu, M. Weber, M. Ryabinin, D. Adelani, Y. Chen, R. Tang, and P. Stenetorp (2025)
  Multilingual language model pretraining using machine-translated data.
  External Links: [Link](https://arxiv.org/abs/2502.13252)
  Cited by: [item 3](#S1.I1.i3.p1.1 "In 1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§4.2](#S4.SS2.p1.1 "4.2 The efficacy of translation as augmentation is determined by source quality ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§4.2](#S4.SS2.p4.1 "4.2 The efficacy of translation as augmentation is determined by source quality ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* L. Wang, N. Yang, X. Huang, B. Jiao, L. Yang, D. Jiang, R. Majumder, and F. Wei (2022)
  Text embeddings by weakly-supervised contrastive pre-training.
  arXiv preprint arXiv:2212.03533.
  Cited by: [§4.1.2](#S4.SS1.SSS2.p2.1 "4.1.2 Cross-lingual curation gains correlate with language similarity ‣ 4.1 The impact of curation on multilingual transfer dynamics and language-specific performance ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* Z. Wang, Z. C. Lipton, and Y. Tsvetkov (2020)
  Negative interference in multilingual models: findings and a meta-learning treatment.
  In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP),
   pp. 4373–4388.
  External Links: [Link](https://aclanthology.org/2020.emnlp-main.356/)
  Cited by: [§2](#S2.SS0.SSS0.Px2.p1.1 "Data Mixing and Interference. ‣ 2 Related Work ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* L. Xue, N. Constant, A. Roberts, M. Kale, R. Al-Rfou, A. Siddhant, A. Barua, and C. Raffel (2021)
  MT5: a massively multilingual pre-trained text-to-text transformer.
  In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies,
   pp. 483–498.
  External Links: [Link](https://aclanthology.org/2021.naacl-main.41/)
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"),
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Multilingual Data Curation. ‣ 2 Related Work ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").
* A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. (2025)
  Qwen3 technical report.
  arXiv preprint arXiv:2505.09388.
  Cited by: [§4.3](#S4.SS3.SSS0.Px1.p1.1 "Establishing a New Pareto Frontier. ‣ 4.3 Integrating multilingual curation into a general pretraining mix ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset").

## Appendix A Appendix

### A.1 Evaluation datasets per language

In the table below, Global MMLU is the dataset provided by Singh et al. ([2025](#bib.bib29 "Global mmlu: understanding and addressing cultural and linguistic biases in multilingual evaluation")) while
Indic MMLU refers to the translation into Indic languages444available here: [https://huggingface.co/datasets/sarvamai/mmlu-indic](sarvam).
For ARC evaluations, we rely on
evaluation datasets released as part of the Okapi framework (Lai et al., [2023](#bib.bib28 "Okapi: instruction-tuned large language models in multiple languages with reinforcement learning from human feedback")). This contains evaluations for the majority of the languages we consider, with the exception of
Korean, Portuguese, Hindi and Bengali.
For Korean, we use the Ko-ARC evaluation (Thunder Research Group, [2025](#bib.bib27 "Korean benchmarks")), for Portuguese
we use the translated version provided by LumiOpen555available here: [https://huggingface.co/datasets/LumiOpen/arc\_challenge\_mt](lumi). Finally, for Indic ARC evaluations we use Indic ARC666available here:
[https://huggingface.co/datasets/sarvamai/arc-challenge-indic](sarvam).
In the case of Belebele, the original evaluation dataset supports all our languages (Bandarkar et al., [2024](#bib.bib24 "The belebele benchmark: a parallel reading comprehension dataset in 122 language variants"))777availabel here: [https://huggingface.co/datasets/facebook/belebele](belebele).

|  |  |  |  |
| --- | --- | --- | --- |
| Language | MMLU | ARC | Belebele |
| Spanish | Global MMLU | Okapi | Belebele |
| Portuguese | Global MMLU | LumiOpen | Belebele |
| French | Global MMLU | Okapi | Belebele |
| German | Global MMLU | Okapi | Belebele |
| Italian | Global MMLU | Okapi | Belebele |
| Vietnamese | Global MMLU | Okapi | Belebele |
| Indonesian | Global MMLU | Okapi | Belebele |
| Russian | Global MMLU | Okapi | Belebele |
| Arabic | Global MMLU | Okapi | Belebele |
| Hindi | Indic MMLU | Indic ARC | Belebele |
| Bengali | Indic MMLU | Indic ARC | Belebele |
| Chinese | Global MMLU | Okapi | Belebele |
| Japanese | Global MMLU | Not present | Belebele |
| Korean | Global MMLU | Ko-ARC | Belebele |

Table 2: Table describing the choice of evaluation datasets.

### A.2 Details of FLOP budget computations for open-source models

We summarize the training compute (in FLOPs) for each open-source baseline
reported in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"). Throughout this work, we estimate
training FLOPs using the standard approximation

|  |  |  |
| --- | --- | --- |
|  | Total FLOPs≈6×N×D,\text{Total FLOPs}\approx 6\times N\times D, |  |

where NN is the number of (trainable) parameters and DD is the number of
training tokens. In the table, B=109\mathrm{B}=10^{9} and T=1012\mathrm{T}=10^{12}.
For MoE models, we use the *active* parameter count per token as NN.

| Model | Total parameters (NN) | Total tokens (DD) | FLOPs (≈6​N​D\approx 6ND) |
| --- | --- | --- | --- |
| DatologyAI 3B | 3B | 1T | 1.8×10221.8\times 10^{22} |
| DatologyAI 8B | 8B | 1T | 4.8×10224.8\times 10^{22} |
| Llama-3.2-1B | 1B | 9T | 5.4×10225.4\times 10^{22} |
| Llama-3.2-3B | 3B | 9T | 1.6×10231.6\times 10^{23} |
| Llama-3.2-8B | 8B | 15T | 7.2×10237.2\times 10^{23} |
| SmolLM3-3B | 3B | 11T | 1.9×10231.9\times 10^{23} |
| Granite-4.0-microbase | 3B | 15T | 2.7×10232.7\times 10^{23} |
| Qwen3-1.7B | 1.7B | 36T | 3.7×10233.7\times 10^{23} |
| Qwen3-4B | 4B | 36T | 8.6×10238.6\times 10^{23} |
| Qwen3-8B | 8B | 36T | 1.7×10241.7\times 10^{24} |
| LFM-2-2.6B | 2.6B | 10T | 1.5×10231.5\times 10^{23} |
| LFM-2.5-1.2B | 1.17B | 28T | 1.9×10231.9\times 10^{23} |
| Trillion Labs Tri-7B | 7B | 2T | 9.3×10229.3\times 10^{22} |
| Sarvam-1 | 2B | 4T | 4.8×10224.8\times 10^{22} |
| Trinity Large | 13B active (400B total) | 17T | 1.3×10241.3\times 10^{24} |

Table 3: Estimated training compute (FLOPs) for the open-source baselines in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"), computed as ≈6​N​D\approx 6ND. For MoE models, NN denotes the number of *active* parameters per token.

### A.3 Per language evaluation performance

Figure 6: Per-language performance vs. training compute for latin-script languages. Rows correspond to Spanish, Portuguese, French, German, Vietnamese, and Indonesian. Columns 1–3 report performance on MMLU, ARC, and Belebele as a function of training FLOPs (x-axis). The rightmost column compares each model’s language-specific average score (y-axis) to its all-language average across multilingual evaluations (x-axis); the dashed line indicates parity (y = x).

Figure 7: Per-language performance vs. training compute for Indic and Arabic-script languages. Rows correspond to Hindi, Bengali, and Arabic. Columns 1–3 report performance on MMLU, ARC, and Belebele as a function of training FLOPs (x-axis). The rightmost column compares each model’s language-specific average score (y-axis) to its all-language average across multilingual evaluations (x-axis); the dashed line indicates parity (y = x).




Figure 8: Per-language performance vs. training compute for CJK languages and Russian. Rows correspond to Chinese, Japanese, Korean, and Russian. Columns 1–3 report performance on MMLU, ARC, and Belebele as a function of training FLOPs (x-axis). The rightmost column compares each model’s language-specific average score (y-axis) to its all-language average across multilingual evaluations (x-axis); the dashed line indicates parity (y = x). We note that there is no ARC Challenge evaluation available for Japanese.

### A.4 Multilingual data efficiency gains

In this section, we quantify multilingual performance as a function of the training token count dedicated to each language. Conducting this analysis is challenging for several reasons. First, several of the evaluated models use their own tokenizers,
which makes token counts imperfectly comparable across models.
Second, precise per-language token counts are often
unavailable for open-source models, and so we aim to estimate them as best we can.
We nevertheless include this analysis because DatologyAI curation yields improvements in multilingual data efficiency that are large, often by orders of magnitude,
so the qualitative conclusion is robust even under reasonable uncertainty in these estimates.

In this section we only include models for which we could obtain a reasonably reliable
estimate of per-language tokens using public information. These models are:

* •

  SmolLM3: This model used 12% multilingual data over 11T, supporting a range of languages including Spanish, German, French and Portuguese. We compute the amount of tokens per language directly from the configurations which were publicly shared888Available at [https://huggingface.co/datasets/HuggingFaceTB/smollm3-configs](configs).
* •

  Llama3.2: This model used 8% multilingual data over 9T, supporting seven languages. This is approximately 100B tokens per language.
* •

  Sarvam-1: This was was trained on a 2T Indic language corpus, which contained 20% Hindi tokens and 10% Bengali tokens. This corresponds to 200B and 100B tokens for each language respectively.
* •

  Trillion Labs 7B: this model was trained on a 2T dataset, 10% of which was multilingual with a primary focus on Korean. As such, we estimated this model was trained with approximately 200B Korean tokens.
* •

  DatologyAI: as referenced in section [4.3](#S4.SS3 "4.3 Integrating multilingual curation into a general pretraining mix ‣ 4 Main Findings ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset"), DatologyAI models were trained for 1T tokens with a 7.75% multilingual component. This corresponds to a total of 75B multilingual tokens across thirteen languages, approximately 6B tokens per language.

Figure [9](#A1.F9 "Figure 9 ‣ A.4 Multilingual data efficiency gains ‣ Appendix A Appendix ‣ Insights from Multilingual Curation for a 20-Trillion-Token Dataset") visualizes the performance across various languages as a function of estimated tokens in that language. We observe that DatologyAI curation has orders of magnitude gains in token efficiency.

Figure 9: Per-Language Performance vs. Multilingual Training Tokens.
We visualize the number of language-specific training tokens (x-axis, billions) and the
average downstream performance across a range of models. We only include models where the
number of tokens per langauge could be reasonably estimated based on public
information. DatologyAI models were trained with only 6B tokens per language (7.75% mutlilingual overall).
The plots demonstrate significant data efficiency improvements from DatologyAI curation compared to
open-source baselines such as Llama-3.2, SmolLM2, and language-specific models like Sarvam-2B. We add an asterix beside the name of all non-DatologyAI models, to highlight that we estimated the number of tokens per language to the best of our ability based on publicly available information.

#### A.4.1 Numerical results

Below we report numerical evaluation results por language.

| Model | MMLU | ARC | Belebele |
| --- | --- | --- | --- |
| Datology 3B | 0.46 | 0.63 | 0.62 |
| Datology 8B | 0.55 | 0.73 | 0.76 |
| Llama-3.2 1B | 0.30 | 0.31 | 0.38 |
| Llama-3.2 3B | 0.47 | 0.56 | 0.56 |
| Llama-3.1 8B | 0.55 | 0.67 | 0.74 |
| Qwen3 4B | 0.67 | 0.86 | 0.85 |
| Qwen3 8B | 0.71 | 0.90 | 0.86 |
| SmolLM3 3B | 0.53 | 0.66 | 0.62 |
| Granite-4.0 Micro | 0.54 | 0.70 | 0.74 |
| LFM2.5 1.2B | 0.50 | 0.66 | 0.54 |
| Trinity Large (MoE) | 0.77 | 0.92 | 0.91 |

Table 4: Evaluations for language Spanish



| Model | MMLU | ARC | Belebele |
| --- | --- | --- | --- |
| Datology 3B | 0.44 | 0.58 | 0.64 |
| Datology 8B | 0.55 | 0.71 | 0.77 |
| Llama-3.2 1B | 0.31 | 0.30 | 0.35 |
| Llama-3.2 3B | 0.46 | 0.53 | 0.55 |
| Llama-3.1 8B | 0.54 | 0.66 | 0.79 |
| Qwen3 4B | 0.65 | 0.84 | 0.84 |
| Qwen3 8B | 0.68 | 0.87 | 0.86 |
| SmolLM3 3B | 0.49 | 0.63 | 0.62 |
| Granite-4.0 Micro | 0.54 | 0.69 | 0.79 |
| LFM2.5 1.2B | 0.46 | 0.61 | 0.45 |
| Trinity Large (MoE) | 0.75 | 0.92 | 0.91 |

Table 5: Evaluations for language Portuguese



| Model | MMLU | ARC | Belebele |
| --- | --- | --- | --- |
| Datology 3B | 0.43 | 0.58 | 0.63 |
| Datology 8B | 0.55 | 0.74 | 0.77 |
| Llama-3.2 1B | 0.29 | 0.31 | 0.32 |
| Llama-3.2 3B | 0.46 | 0.54 | 0.58 |
| Llama-3.1 8B | 0.54 | 0.68 | 0.78 |
| Qwen3 4B | 0.66 | 0.84 | 0.87 |
| Qwen3 8B | 0.69 | 0.87 | 0.89 |
| SmolLM3 3B | 0.52 | 0.68 | 0.61 |
| Granite-4.0 Micro | 0.53 | 0.70 | 0.74 |
| LFM2.5 1.2B | 0.50 | 0.65 | 0.48 |
| Trinity Large (MoE) | 0.71 | 0.90 | 0.92 |

Table 6: Evaluations for language French



| Model | MMLU | ARC | Belebele |
| --- | --- | --- | --- |
| Datology 3B | 0.44 | 0.59 | 0.62 |
| Datology 8B | 0.54 | 0.71 | 0.76 |
| Llama-3.2 1B | 0.26 | 0.26 | 0.28 |
| Llama-3.2 3B | 0.44 | 0.50 | 0.56 |
| Llama-3.1 8B | 0.52 | 0.63 | 0.69 |
| Qwen3 4B | 0.65 | 0.85 | 0.86 |
| Qwen3 8B | 0.69 | 0.87 | 0.88 |
| SmolLM3 3B | 0.50 | 0.64 | 0.61 |
| Granite-4.0 Micro | 0.52 | 0.69 | 0.79 |
| LFM2.5 1.2B | 0.48 | 0.65 | 0.52 |
| Trinity Large (MoE) | 0.73 | 0.91 | 0.92 |

Table 7: Evaluations for language German



| Model | MMLU | ARC | Belebele |
| --- | --- | --- | --- |
| Datology 3B | 0.43 | 0.59 | 0.56 |
| Datology 8B | 0.52 | 0.67 | 0.73 |
| Llama-3.2 1B | 0.27 | 0.26 | 0.34 |
| Llama-3.2 3B | 0.38 | 0.46 | 0.48 |
| Llama-3.1 8B | 0.48 | 0.57 | 0.67 |
| Qwen3 4B | 0.59 | 0.79 | 0.81 |
| Qwen3 8B | 0.65 | 0.83 | 0.85 |
| SmolLM3 3B | 0.42 | 0.51 | 0.50 |
| Granite-4.0 Micro | 0.43 | 0.52 | 0.65 |
| LFM2.5 1.2B | 0.26 | 0.27 | 0.28 |
| Trinity Large (MoE) | 0.71 | 0.88 | 0.91 |

Table 8: Evaluations for language Vietnamese



| Model | MMLU | ARC | Belebele |
| --- | --- | --- | --- |
| Datology 3B | 0.42 | 0.59 | 0.56 |
| Datology 8B | 0.52 | 0.68 | 0.72 |
| Llama-3.2 1B | 0.31 | 0.29 | 0.36 |
| Llama-3.2 3B | 0.42 | 0.50 | 0.51 |
| Llama-3.1 8B | 0.49 | 0.60 | 0.69 |
| Qwen3 4B | 0.64 | 0.83 | 0.83 |
| Qwen3 8B | 0.68 | 0.87 | 0.86 |
| SmolLM3 3B | 0.42 | 0.48 | 0.53 |
| Granite-4.0 Micro | 0.49 | 0.57 | 0.68 |
| LFM2.5 1.2B | 0.29 | 0.30 | 0.26 |
| Trinity Large (MoE) | 0.76 | 0.91 | 0.91 |

Table 9: Evaluations for language Indonesian



| Model | MMLU | ARC | Belebele |
| --- | --- | --- | --- |
| Datology 3B | 0.32 | 0.41 | 0.45 |
| Datology 8B | 0.39 | 0.54 | 0.54 |
| Llama-3.2 1B | 0.26 | 0.27 | 0.29 |
| Llama-3.2 3B | 0.34 | 0.42 | 0.41 |
| Llama-3.1 8B | 0.38 | 0.49 | 0.53 |
| Qwen3 4B | 0.48 | 0.71 | 0.71 |
| Qwen3 8B | 0.52 | 0.79 | 0.74 |
| SmolLM3 3B | 0.37 | 0.47 | 0.45 |
| Granite-4.0 Micro | 0.36 | 0.46 | 0.54 |
| LFM2.5 1.2B | 0.24 | 0.25 | 0.24 |
| Trinity Large (MoE) | 0.67 | 0.88 | 0.84 |
| Sarvam1 2B | 0.42 | 0.58 | 0.59 |

Table 10: Evaluations for language Hindi



| Model | MMLU | ARC | Belebele |
| --- | --- | --- | --- |
| Datology 3B | 0.32 | 0.42 | 0.43 |
| Datology 8B | 0.36 | 0.48 | 0.51 |
| Llama-3.2 1B | 0.25 | 0.24 | 0.25 |
| Llama-3.2 3B | 0.31 | 0.33 | 0.39 |
| Llama-3.1 8B | 0.36 | 0.42 | 0.53 |
| Qwen3 4B | 0.45 | 0.67 | 0.71 |
| Qwen3 8B | 0.50 | 0.76 | 0.77 |
| SmolLM3 3B | 0.30 | 0.28 | 0.30 |
| Granite-4.0 Micro | 0.36 | 0.42 | 0.51 |
| Trinity Large (MoE) | 0.62 | 0.79 | 0.80 |
| Sarvam1 2B | 0.41 | 0.56 | 0.55 |

Table 11: Evaluations for language Bengali



| Model | MMLU | ARC | Belebele |
| --- | --- | --- | --- |
| Datology 3B | 0.42 | 0.59 | 0.58 |
| Datology 8B | 0.49 | 0.67 | 0.72 |
| Llama-3.2 1B | 0.27 | 0.27 | 0.29 |
| Llama-3.2 3B | 0.38 | 0.46 | 0.50 |
| Llama-3.1 8B | 0.43 | 0.55 | 0.65 |
| Qwen3 4B | 0.56 | 0.78 | 0.82 |
| Qwen3 8B | 0.63 | 0.83 | 0.85 |
| SmolLM3 3B | 0.47 | 0.59 | 0.61 |
| Granite-4.0 Micro | 0.44 | 0.60 | 0.75 |
| LFM2.5 1.2B | 0.45 | 0.62 | 0.47 |
| Trinity Large (MoE) | 0.72 | 0.89 | 0.91 |

Table 12: Evaluations for language Arabic



| Model | MMLU | ARC | Belebele |
| --- | --- | --- | --- |
| Datology 3B | 0.45 | 0.63 | 0.64 |
| Datology 8B | 0.52 | 0.72 | 0.74 |
| Llama-3.2 1B | 0.33 | 0.33 | 0.35 |
| Llama-3.2 3B | 0.46 | 0.58 | 0.66 |
| Llama-3.1 8B | 0.53 | 0.67 | 0.78 |
| Qwen3 4B | 0.66 | 0.84 | 0.87 |
| Qwen3 8B | 0.70 | 0.88 | 0.88 |
| SmolLM3 3B | 0.49 | 0.65 | 0.63 |
| Granite-4.0 Micro | 0.51 | 0.67 | 0.76 |
| LFM2.5 1.2B | 0.49 | 0.65 | 0.46 |
| Trinity Large (MoE) | 0.74 | 0.90 | 0.92 |

Table 13: Evaluations for language Chinese



| Model | MMLU | Belebele |
| --- | --- | --- |
| Datology 3B | 0.44 | 0.55 |
| Datology 8B | 0.52 | 0.66 |
| Llama-3.2 1B | 0.26 | 0.25 |
| Llama-3.2 3B | 0.40 | 0.44 |
| Llama-3.1 8B | 0.48 | 0.65 |
| Qwen3 4B | 0.59 | 0.79 |
| Qwen3 8B | 0.63 | 0.81 |
| SmolLM3 3B | 0.46 | 0.51 |
| Granite-4.0 Micro | 0.48 | 0.71 |
| LFM2.5 1.2B | 0.49 | 0.44 |
| Trinity Large (MoE) | 0.73 | 0.87 |

Table 14: Evaluations for language Japanese



| Model | MMLU | ARC | Belebele |
| --- | --- | --- | --- |
| Datology 3B | 0.43 | 0.62 | 0.61 |
| Datology 8B | 0.49 | 0.69 | 0.73 |
| Llama-3.2 1B | 0.26 | 0.25 | 0.26 |
| Llama-3.2 3B | 0.39 | 0.47 | 0.49 |
| Llama-3.1 8B | 0.48 | 0.59 | 0.68 |
| Qwen3 4B | 0.59 | 0.81 | 0.82 |
| Qwen3 8B | 0.64 | 0.87 | 0.84 |
| SmolLM3 3B | 0.45 | 0.57 | 0.57 |
| Granite-4.0 Micro | 0.46 | 0.61 | 0.73 |
| LFM2.5 1.2B | 0.44 | 0.61 | 0.51 |
| Trinity Large (MoE) | 0.72 | 0.92 | 0.90 |
| TrillionLabs 7B | 0.50 | 0.68 | 0.67 |

Table 15: Evaluations for language Korean



| Model | MMLU | ARC | Belebele |
| --- | --- | --- | --- |
| Datology 3B | 0.41 | 0.58 | 0.61 |
| Datology 8B | 0.52 | 0.70 | 0.75 |
| Llama-3.2 1B | 0.31 | 0.31 | 0.34 |
| Llama-3.2 3B | 0.43 | 0.55 | 0.56 |
| Llama-3.1 8B | 0.53 | 0.67 | 0.77 |
| Qwen3 4B | 0.64 | 0.81 | 0.85 |
| Qwen3 8B | 0.69 | 0.86 | 0.87 |
| SmolLM3 3B | 0.48 | 0.61 | 0.61 |
| Granite-4.0 Micro | 0.50 | 0.62 | 0.75 |
| LFM2.5 1.2B | 0.31 | 0.33 | 0.38 |
| Trinity Large (MoE) | 0.76 | 0.91 | 0.92 |

Table 16: Evaluations for language Russian
