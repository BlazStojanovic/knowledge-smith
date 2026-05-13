---
arxiv: '2601.02316'
authors:
- DatologyAI
- ':'
- Siddharth Joshi
- Haoli Yin
- Rishabh Adiga
- Ricardo Monti
- Aldo Carranza
- Alex Fang
- Alvin Deng
- Amro Abbas
- Brett Larsen
- Cody Blakeney
- Darren Teh
- David Schwab
- Fan Pan
- Haakon Mongstad
- Jack Urbanek
- Jason Lee
- Jason Telanoff
- Josh Wills
- Kaleigh Mentzer
- Luke Merrick
- Parth Doshi
- Paul Burstein
- Pratyush Maini
- Scott Loftin
- Spandan Das
- Tony Jiang
- Vineeth Dorna
- Zhengping Wang
- Bogdan Gaza
- Ari Morcos
- Matthew Leavitt
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'DatBench: Discriminative, Faithful, and Efficient VLM Evaluations'
url: https://arxiv.org/abs/2601.02316
year: 2026
---

[2601.02316] DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations














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



# [Uncaptioned image] DatBench Discriminative, Faithful, and Efficient VLM Evaluations

###### Abstract

Empirical evaluation serves as the primary compass guiding research progress in foundation models. Despite a large body of work focused on training frontier vision-language models (VLMs), approaches to their evaluation remain nascent. To guide their maturation, we propose three desiderata that evaluations should satisfy: (1) faithfulness to the modality and application, (2) discriminability between models of varying quality, and (3) efficiency in compute. Through this lens, we identify critical failure modes that violate faithfulness and discriminability, misrepresenting model capabilities: (i) multiple-choice formats reward guessing, do not represent downstream use-cases, and saturate early as models improve; (ii) ‘blindly-solvable’ questions which can be answered without images, constitute up to 70% of some evaluations; and (iii) mislabeled or ambiguous samples compromise up to 42% of examples in certain datasets. Regarding efficiency, the computational burden of evaluating frontier models has become prohibitive: by some accounts, nearly 20% of development compute is devoted to evaluation alone.
Rather than discarding existing benchmarks, we curate them via transformation and filtering to maximize their fidelity and discriminability.
We find that transformations such as converting MCQs to generative tasks reveal sharp capability drops of up to 35%. In addition, filtering blindly-solvable and mislabeled samples enhances the discriminative power of these evaluations, while simultaneously reducing their computational cost. We release DatBench-Full, a cleaned evaluation suite of 33 datasets spanning nine VLM capabilities, and DatBench, a discriminative subset that achieves 13× average speedup (up to 50×) while closely matching the discriminative power of the original datasets. Our work provides a path towards evaluation practices that are both rigorous and sustainable as VLMs continue to scale.

[![[Uncaptioned image]](/html/2601.02316/assets/figures/hf_logo.png) DatBench: https://huggingface.co/datasets/DatologyAI/DatBench](https://huggingface.co/datasets/DatologyAI/DatBench)
  
[![[Uncaptioned image]](/html/2601.02316/assets/figures/hf_logo.png) DatBench-Full: https://huggingface.co/datasets/DatologyAI/DatBench-Full](https://huggingface.co/datasets/DatologyAI/DatBench-Full)
  
[![[Uncaptioned image]](/html/2601.02316/assets/figures/github_logo.png) Code: https://github.com/datologyai/DatBench](https://github.com/datologyai/DatBench)

DatologyAI Team111See Contributions and Acknowledgments (§ [7](#S7 "7 Contributions and Acknowledgements ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")) for full author list.

## 1 Introduction

![Refer to caption](/html/2601.02316/assets/figures/discrimination_multi_capability_uniform_light.png)

![Refer to caption](/html/2601.02316/assets/figures/speedup.png)

Figure 1: 
DatBench reduces evaluation cost while increasing discriminative signal.
Panel (a) shows discriminative power as a function of retained data (for select capabilities), demonstrating that targeted selection reaches full-benchmark discriminative power using as little as 40% of the samples.
Panel (b) reports average H100 hours and relative speedup across nine capabilities.

Empirical evaluation is the primary mechanism through which progress in foundation models is recognized, compared, and acted upon. As machine learning has shifted from narrow, task-specific systems to general-purpose vision-language models (VLMs) with broad and compositional capabilities (Wei et al., [2022](#bib.bib20 "Emergent abilities of large language models")), benchmarks now play an outsized role: they define what counts as progress and directly shape how substantial computational and human resources are allocated. Evaluations are no longer a passive reporting tool but an active driver of research direction.

However, modern evaluation pipelines are increasingly misaligned with the behaviors they aim to measure. As model inputs span multiple modalities and outputs become increasingly generative and stochastic, benchmarks must better disentangle genuine capabilities from superficial heuristics and inherent variance (Lu et al., [2022](#bib.bib57 "Fantastically ordered prompts and where to find them: overcoming few-shot prompt order sensitivity")). While the evaluation of language-only models has received sustained methodological attention (Srivastava et al., [2023](#bib.bib54 "Beyond the imitation game: quantifying and extrapolating the capabilities of language models"); OpenCompass, [2023](#bib.bib55 "OpenCompass: a universal evaluation platform for foundation models")), VLM evaluation remains comparatively under-examined.

Recent evidence suggests that this gap has become a serious liability. Existing VLM benchmarks suffer from pervasive data quality failures, including mislabeled or ambiguous examples, questions solvable without visual input, and heavy reliance on multiple-choice formats that are not representative of downstream use-cases and are vulnerable to spurious correlations (Masry et al., [2025](#bib.bib9 "ChartQAPro: a more diverse and challenging benchmark for chart question answering"); Liu et al., [2024](#bib.bib21 "Mmbench: is your multi-modal model an all-around player?"); xAI, [2024](#bib.bib48 "RealWorldQA: a benchmark for real-world visual understanding"); Zhang et al., [2024b](#bib.bib14 "MME-realworld: could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans?"); Yang et al., [2024](#bib.bib12 "CC-ocr: a comprehensive and challenging ocr benchmark for evaluating large multimodal models in literacy")). These artifacts inflate reported accuracy, introduce a substantial noise floor, and reduce the signal-to-noise ratio of the evaluations. In such a regime, small improvements, often on the order of a few percent, are more plausibly explained by overfitting to benchmark idiosyncrasies than by real capability gains, rendering the research community vulnerable to hill-climbing on noise (Recht et al., [2019](#bib.bib58 "Do imagenet classifiers generalize to imagenet?"); Wei et al., [2023](#bib.bib59 "Inverse scaling can become u-shaped")).

At the same time, evaluation has become a major computational bottleneck. Running comprehensive VLM evaluation suites now consumes a nontrivial fraction of total development compute (Strubell et al., [2019](#bib.bib60 "Energy and policy considerations for deep learning in nlp")). For example, during the development of OLMo3, nearly 20% of the total compute budget for the post-training phase was reportedly dedicated to evaluation alone (Lambert, [2025](#bib.bib69 "Good researchers obsess over evals: the story of OLMo 3 (post-training), told through evals")). This burden is amplified for VLMs by the dense visual token sequences required to represent high-resolution images and the extended reasoning traces at inference time, which can collectively exceed tens of thousands of tokens per example (Bai et al., [2025](#bib.bib38 "Qwen3-vl technical report")). Detailed analyses indicate that much of this cost is spent evaluating samples that are either trivial, noisy, or weakly discriminative (Schick and others, [2025](#bib.bib3 "Fluid language model benchmarking"); Polo et al., [2024](#bib.bib24 "TinyBenchmarks: evaluating llms with fewer examples")).

In this work, we argue that the design of effective evaluation should be treated as a data curation problem. Rather than repeatedly constructing new benchmarks from scratch, we propose to systematically transform and filter evaluation data to maximize faithfulness, discriminative power, and efficiency. This perspective mirrors recent successes in training data curation, in which careful data transformation and selection has produced large gains in model quality and compute efficiency (Fang et al., [2023](#bib.bib19 "Data filtering networks"); Joshi and Mirzasoleiman, [2023](#bib.bib17 "Data-efficient contrastive self-supervised learning: most beneficial examples for supervised learning contribute the least"); Abbas et al., [2023](#bib.bib52 "SemDeDup: data-efficient learning at web-scale through semantic deduplication"); Joshi et al., [2024](#bib.bib18 "Data-efficient contrastive language-image pretraining: prioritizing data quality over quantity"), [2025a](#bib.bib62 "Dataset distillation via knowledge distillation: towards efficient self-supervised pre-training of deep networks"), [2025b](#bib.bib51 "MM-gen: enhancing task performance through targeted multimodal data curation"); DatologyAI et al., [2024a](#bib.bib73 "DatologyAI Technical Deep-Dive: Image-Text Data Curation at the Billion-Sample Scale"), [b](#bib.bib72 "DatologyAI Technical Deep-Dive: Curating Our Way Curation to a Billion-State-of-the-Art Text Dataset"), [2025](#bib.bib79 "BeyondWeb: lessons from scaling synthetic data for trillion-scale pretraining")). We show that the same principles apply, with similar impact, to evaluation.

Guided by this view, we define three desiderata for modern VLM evaluation datasets, (i) faithfulness: examples should genuinely require visual input and reflect intended downstream use cases; (ii) discriminability: examples should reliably separate stronger models from weaker ones; and (iii) efficiency: evaluation should maximize signal per unit of compute.
These criteria expose four systematic failure modes in existing benchmarks and motivate targeted interventions (Section [3](#S3 "3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")).

First, multiple-choice formats are both unfaithful and weakly discriminative in generative settings. Converting MCQs to open-ended generation reveals large hidden capability gaps. On AI2D, for example, average accuracy drops from 77.56% to 40.53%, with the strongest MCQ model losing nearly 35 points. When generative conversion is infeasible, circular evaluation (Liu et al., [2024](#bib.bib21 "Mmbench: is your multi-modal model an all-around player?")) collapses chance baselines and exposes similar inflation effects.

Second, many VLM benchmarks can be solved without vision. By evaluating models with the image removed, we find that over 70% of samples in VQA-v2 (Goyal et al., [2017](#bib.bib46 "Making the V in VQA matter: elevating the role of image understanding in Visual Question Answering")) can be answered correctly using language priors alone. Such examples fundamentally fail to measure multimodal reasoning.

Third, low-resolution inputs and inaccurate or ambiguous annotations introduce substantial noise. Using a multi-stage filtering pipeline, we discard up to 42.07% of samples in benchmarks such as MME-RealWorld (Autonomous Driving) (Zhang et al., [2024b](#bib.bib14 "MME-realworld: could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans?")). In these instances, evaluation is confounded by factual labeling errors and indeterminable ground truths—where poor image quality renders the target objects unrecognizable even to a human observer—effectively precluding reliable performance assessment.

Fourth, existing evaluation suites are inefficient. By explicitly selecting items with high discriminative power across a diverse set of 1B–10B scale models, we achieve speedups of up to 50×\times (13×\times on average) while closely matching the discriminative power of full benchmarks using a small fraction of the data (Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")).

Applying these interventions, we introduce DatBench (Section [4](#S4 "4 Introducing DatBench and DatBench-Full ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")), a curated suite of VLM evaluations designed to be faithful, discriminative, and compute-efficient. To construct it, we partition the large pool of existing datasets into nine fundamental VLM capabilities and release two resulting artifacts:

* •

  DatBench, a high-efficiency subset for rapid iteration that provides a 13×\times speedup on average across all capabilities while increasing signal per sample.
* •

  DatBench-Full, the full collection of high-quality samples remaining after excluding blind-solvable or objectively low-quality data.

Beyond efficiency, our work provides empirical insights across 27 state-of-the-art VLMs, revealing structural limitations that are invisible under conventional evaluation (Section [5](#S5 "5 Diagnosing VLM Pathologies with DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")). We show that inference-time scaling can actively degrade perceptual performance through an overthinking penalty, that current VLMs exhibit a sharp tension between high-level reasoning and low-level perception, and that language priors systematically mask true multimodal capability across popular benchmarks. Together, these resources and findings improve evaluation quality while dramatically reducing its cost, offering a path toward evaluation practices that keep pace with the rapid advancement of vision-language models.

## 2 Related Work

Faithful Evaluation.
Recent research has identified significant issues with the validity of VLM benchmarks, prompting various mitigation strategies.
To address inflated performance caused by high-risk baselines in multiple-choice evaluations, several studies propose reformulating tasks into generative answer-matching settings (Chandak et al., [2025](#bib.bib1 "Answer matching outperforms multiple choice for language model evaluation")) or employing circular evaluation techniques (Liu et al., [2024](#bib.bib21 "Mmbench: is your multi-modal model an all-around player?")). More broadly, prior work shows that ambiguous and hard-to-solve comparative prompts can systematically induce spurious preferences in models, meaning that the evaluation prompts themselves can become a hidden source of bias when they implicitly force a choice without sufficient grounding or context (Adiga et al., [2025](#bib.bib56 "Attention speaks volumes: localizing and mitigating bias in language models")). This further motivates interventions like circular evaluations and other option-robust MCQ protocols. Other efforts focus on statistical refinement of evaluation metrics. For instance, Schick and others ([2025](#bib.bib3 "Fluid language model benchmarking")) apply Item Response Theory (IRT) motivated weighting to account for item difficulty and discrimination beyond simple average accuracy.

Beyond these issues, multiple-choice formats are also misaligned with real-world VLM usage, where models are typically deployed in open-ended, generative settings rather than selecting from a small, predefined set of options. As a result, strong MCQ performance may overstate practical capability by rewarding option elimination or prompt-specific biases, as MCQ-based evaluations systematically misrepresent model abilities by constraining outputs and failing to probe the generative behaviors that dominate real-world LLM and VLM deployment (Li et al., [2024b](#bib.bib6 "Can multiple-choice questions really be useful in detecting the abilities of llms?")).

Additional analyses suggest that many VLMs can perform well on certain benchmarks without meaningfully leveraging visual input, calling into question whether such evaluations truly measure visual understanding or multimodal reasoning (Lee et al., [2025](#bib.bib28 "VLind-bench: measuring language priors in large vision-language models"); Li et al., [2024a](#bib.bib29 "A survey on benchmarks of multimodal large language models"); Lin et al., [2024](#bib.bib31 "Revisiting the role of language priors in vision-language models"); Wang et al., [2024a](#bib.bib27 "Is a picture worth a thousand words? delving into spatial reasoning for vision language models"); Zhang et al., [2025](#bib.bib30 "Debiasing multimodal large language models via penalization of language priors")). In contrast to approaches that seek to recover signal through post hoc statistical modeling, our method improves evaluation reliability at the source by enhancing data quality via systematic transformation and filtering of benchmark examples, building on both prior work and newly introduced techniques.

Efficient & Discriminative Evaluation.
Efforts to improve the efficiency of model evaluation largely draw from (1) psychometric modeling, and (2) exploiting semantic structure in evaluation data. IRT–based methods (Schick and others, [2025](#bib.bib3 "Fluid language model benchmarking"); Polo et al., [2024](#bib.bib24 "TinyBenchmarks: evaluating llms with fewer examples")) model latent capability variables in order to estimate item difficulty and discrimination. In practice, however, these approaches typically require large, dense response matrices (many models evaluated on many items) to fit parameters stably. Without this scale, estimates can become highly sensitive to hyperparameter choices.

An alternative line of work leverages semantic structure. For example, Vivek et al. ([2024](#bib.bib23 "Anchor points: benchmarking models with much fewer examples")) employ embedding-based clustering to select representative subsets, while Scales++ (Bean et al., [2025](#bib.bib22 "Scales++: compute efficient evaluation subset selection with cognitive scales embeddings")) relies on qualitative, rubric-driven segmentation of tasks. These approaches face notable limitations. Clustering outcomes are tightly coupled to the choice of embedding model, a significant concern given the lack of unified multimodal embeddings, while rubric-based methods are inherently labor-intensive and subjective.

More broadly, approaches that optimize solely for preserving model rankings suffer from an inherent limitation. As we show in Section [3.4](#S3.SS4 "3.4 High Discrimination with Limited Compute ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"), rank correlation saturates quickly and can often be achieved even by random subsets whose individual samples do not reliably discriminate between weak and strong models. Consequently, prioritizing rank stability risks overfitting to a fixed set of evaluated models without guaranteeing the quality of the underlying examples. Prior work (Ghosh et al., [2025](#bib.bib71 "ONEBench to test them all: sample-level benchmarking over open-ended capabilities")) has also proposed aggregating heterogeneous evaluations via Plackett-Luce models, emphasizing ordinal rankings for their robustness to metric calibration issues. While this addresses the challenge of combining diverse measurements, it operates downstream of data quality, aggregating rankings over noisy or blind-solvable samples still propagates those artifacts into the final ordering.

In contrast to these approaches, we shift the focus from preserving global rankings to the targeted curation of individual samples. First, we systematically transform and filter evaluation data to resolve quality issues such as low resolution and labeling errors. Second, we employ a discriminative subset selection strategy that, unlike rank-preservation methods, identifies high-signal samples without requiring the large-scale model response matrices necessary for stable IRT parameter fitting.

## 3 The Making of DatBench

Table 1: Evaluation Suite. We select a suite of 33 diverse datasets that balance standard academic baselines with modern “in-the-wild” challenges. Horizontal rules separate distinct datasets within each capability pillar.

| Capability | Dataset | Selection Rationale & Coverage |
| --- | --- | --- |
| Chart | ChartQA (Masry et al., [2022](#bib.bib7 "ChartQA: a benchmark for question answering about charts with visual and logical reasoning")) | Standard benchmark for basic chart understanding and data extraction. |
| ChartQA Pro (Masry et al., [2025](#bib.bib9 "ChartQAPro: a more diverse and challenging benchmark for chart question answering")) | Challenging counterpart requiring expert-authored reasoning on complex charts. |
| CharXiv (Descriptive & Reasoning) (Wang et al., [2024c](#bib.bib8 "CharXiv: charting gaps in realistic chart understanding in multimodal llms")) | Scientific charts requiring domain-specific knowledge and terminology. |
| InfoVQA (Mathew et al., [2021a](#bib.bib10 "InfographicVQA")) | Mixed-media infographics (combining dense captions with visual diagrams). |
| Document | CC-OCR (Document Parsing & KIE) (Yang et al., [2024](#bib.bib12 "CC-ocr: a comprehensive and challenging ocr benchmark for evaluating large multimodal models in literacy")) | Key Information Extraction (KIE) from structured forms and receipts. |
| OCR-VQA (Mishra et al., [2019](#bib.bib11 "OCR-vqa: visual question answering by reading text in images")) | OCR centric Q&A on book covers. |
| OCRBench-V2 (Fu et al., [2025](#bib.bib75 "OCRBench v2: an improved benchmark for evaluating large multimodal models on visual text localization and reasoning")) | Comprehensive bilingual OCR benchmark; 31 scenarios covering text recognition, localization, extraction, and reasoning. |
|  | DocVQA (Mathew et al., [2021b](#bib.bib13 "DocVQA: a dataset for vqa on document images")) | Standard benchmark for Q&A on spatial document layouts. |
| Scene OCR | TextVQA (Singh et al., [2019](#bib.bib15 "Towards vqa models that can read")) | Industry standard for recognition of text embedded in natural street scenes. |
| MME-RW (OCR in the wild) (Zhang et al., [2024b](#bib.bib14 "MME-realworld: could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans?")) | “In-the-wild” challenges including mobile screens and digital signage. |
| CC-OCR (Multi-Scene OCR) (Yang et al., [2024](#bib.bib12 "CC-ocr: a comprehensive and challenging ocr benchmark for evaluating large multimodal models in literacy")) | Perspectively distorted and artistically stylized text. |
| Math / Logic | MathVista (Lu et al., [2024](#bib.bib39 "MathVista: evaluating mathematical reasoning of foundation models in visual contexts")) | Broad coverage of algebraic reasoning and geometry problems. |
| Mathverse (with & without reasoning) (Zhang et al., [2024a](#bib.bib76 "MathVerse: does your multi-modal llm truly see the diagrams in visual math problems?")) | Visual math benchmark with diagram-based problems across 6 information variants; tests true visual reasoning vs. text-only deduction. |
|  | MathVision (Wang et al., [2024b](#bib.bib77 "Measuring multimodal mathematical reasoning with math-vision dataset")) | Real math competition problems with diagrams; 16 disciplines from algebra to topology. |
|  | LogicVista (Xiao et al., [2024](#bib.bib40 "LogicVista: multimodal llm logical reasoning benchmark in visual contexts")) | Interleaved text-visual clues strictly separating logic from language priors. |
| Spatial | RealWorldQA (xAI, [2024](#bib.bib48 "RealWorldQA: a benchmark for real-world visual understanding")) | Physical grounding in everyday photos (depth estimation, spatial relations). |
| MME-RW (Video Monitoring & Autonomous Driving) (Zhang et al., [2024b](#bib.bib14 "MME-realworld: could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans?")) | Safety-critical spatial awareness (autonomous driving, remote sensing). |
| Grounding | RefCOCO (Kazemzadeh et al., [2014](#bib.bib41 "ReferItGame: referring to objects in photographs of natural scenes")) | General referring expressions (allows both appearance and location words). |
| RefCOCO+ (Kazemzadeh et al., [2014](#bib.bib41 "ReferItGame: referring to objects in photographs of natural scenes")) | Strict appearance-based grounding (disallows spatial words like “left”). |
| RefCOCO-g (Mao et al., [2016](#bib.bib47 "Generation and comprehension of unambiguous object descriptions")) | Long, complex syntactic descriptions (testing recursive understanding). |
|  | RefCOCO-M (moondream, [2025](#bib.bib74 "RefCOCO-M: Refined Referring Expression Segmentation")) | Cleaned and improved version of the RefCOCO (UNC) validation split for referring expression segmentation |
|  | Pixmo-Point (Deitke et al., [2024](#bib.bib42 "Molmo and pixmo: open weights and open data for state-of-the-art vision-language models")) | Precision test: requires coordinate-level localization vs. bounding boxes. |
| Counting | CountBench (Paiss et al., [2023](#bib.bib43 "Teaching clip to count to ten")) | Adversarial distractors to prevent density-map estimation or guessing. |
|  | TallyQA (Acharya et al., [2019](#bib.bib78 "TallyQA: answering complex counting questions")) | Open-ended counting VQA; distinguishes simple (detection-only) vs. complex (reasoning-required) questions. |
| Diagrams | AI2D (Kembhavi et al., [2016](#bib.bib44 "A diagram is worth a dozen images")) | Standard baseline for science and engineering schematic parsing. |
| MME-RW (Diagram/Table) (Zhang et al., [2024b](#bib.bib14 "MME-realworld: could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans?")) | High-resolution, complex tables found in professional reports. |
| General | MMMU-Pro (Yue et al., [2025](#bib.bib45 "MMMU-pro: a more robust multi-discipline multimodal understanding benchmark")) | Hardest exam-style questions for reasoning depth across disciplines. |
| MMBench (Liu et al., [2024](#bib.bib21 "Mmbench: is your multi-modal model an all-around player?")) | Evaluates conversation fidelity and instruction following. |
| VQA-v2 (Goyal et al., [2017](#bib.bib46 "Making the V in VQA matter: elevating the role of image understanding in Visual Question Answering")) | Legacy baseline for open-ended visual questioning. |

### 3.1 MCQ Evaluations: High Noise, Low Fidelity

In this section, we present the methodology for DatBench, a framework designed to transform noisy, large-scale VLM evaluation suites into high-quality, discriminative benchmarks. Our approach systematically addresses four critical failures in current evaluation regimes: (1) signal dilution in Multiple Choice Questions (MCQs), (2) examples solvable without visual context, (3) incorrect, ambiguous, or low-resolution samples, and (4) prohibitively high computational costs. Collectively, the first three interventions enhance the faithfulness and discrimination of the evaluation data, while the fourth ensures the resulting benchmark is both efficient and discriminative.

#### Datasets & Capabilities.

We define our goal as establishing a faithful, discriminative, and efficient evaluation for nine distinct VLM capabilities (c.f. Figure [2](#S3.F2 "Figure 2 ‣ Models. ‣ 3.1 MCQ Evaluations: High Noise, Low Fidelity ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")):
(1) Chart Understanding: extracting quantitative data and performing trend analysis on bar charts, pie charts, line graphs, and infographics;
(2) Document Understanding: parsing structured layouts and extracting key information from digital or scanned documents, with a focus on text-heavy visual processing;
(3) Scene OCR: recognizing and interpreting textual information found in natural environments, such as storefront names, street signs, and product labels;
(4) Math & Logic: solving multimodal mathematical problems, including geometry, physics mechanics diagrams, and complex logical puzzles;
(5) Spatial Reasoning: assessing the relative positions of objects and demonstrating a directional and physical understanding of 3D space;
(6) Grounding: identifying and localizing specific regions or objects referred to in text through bounding boxes or segmentation-style tasks;
(7) Counting: accurately enumerating specific objects across varied environments and overlapping visual contexts;
(8) Diagrams & Tables: interpreting grade-school diagrams and structured tables to extract data points and infer underlying relationships; and
(9) General: performing high-level Visual Question Answering (VQA) based on holistic image descriptions and real-world scene comprehension.
To achieve this, we source a diverse pool of evaluation sets for each capability and apply our methodology to address problems (1)–(4), transforming them into refined, high-quality benchmarks. Table [1](#S3.T1 "Table 1 ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations") details the specific dataset composition and selection rationale used to ensure broad coverage of image distributions across each capability.

#### Models.

We leverage a diverse suite of 27 state-of-the-art models to evaluate and refine our benchmarks. The model families and their corresponding parameter sizes used in this study include:
(1) Qwen3-VL (2B, 4B, and 8B Instruct variants, as well as 2B, 4B, and 8B Thinking models);
(2) Qwen2.5-VL (3B and 7B Instruct variants);
(3) Qwen2.5-Omni (3B and 7B multimodal versions);
(4) InternVL3.5 (2B, 4B, and 8B Instruct variants);
(5) InternVL3 (2B and 9B Instruct variants);
(6) InternVL2.5 (2B, 4B, and 8B variants);
(7) InternVL2 (2B, 4B, and 8B variants); and
(8) Thinking & Specialist Models, comprising GLM-4.1V-9B (Base and Thinking), R-4B, SmolVLM2-2.2B, Phi-3.5-vision, and Gemma-3-4B-it.
Using these models as a broad empirical base allows us to ensure our data-centric improvements generalize beyond any single model family.

For all experiments detailed in this study, model generation was standardized with a maximum output length of 4,096 tokens and suggested sampling configs per the corresponding model card or code repository.

Chart
Understanding

Doc Understanding

Scene OCR
OCR-in-the-Wild

Math & Logic
Geometry, Puzzles

Spatial Reasoning

Grounding

Counting

Diagrams & Tables

General VQA

Figure 2: Capability Partition. We evaluate the model across 9 distinct axes of multimodal performance, ranging from low-level perception (OCR, Grounding) to high-level reasoning (Math, Charts).

#### Problem: Chance Baselines and The Evaluation-Deployment Gap

Standard MCQ formats systematically overestimate model capability through two primary mechanisms: random guessing and task misalignment.
First, multiple-choice questions introduce a non-trivial chance baseline (1/N1/N for NN options), allowing models to achieve inflated scores that add significant noise to performance metrics.
This inflation is compounded when evaluating across multiple stochastic samples or models; the probability of an item appearing ”solved” by at least one of MM uniform random guesses grows rapidly as 1−(1−1/N)M1-(1-1/N)^{M}. Second, there is a fundamental mismatch between evaluation and deployment: while most VLMs are used in generative contexts, MCQs merely test the ability to pick a candidate from a pre-defined list. This ”closed-set” evaluation fails to capture the generative reasoning required for real-world tasks and allows models to rely on superficial shortcuts or linguistic priors within the options themselves (Chandak et al., [2025](#bib.bib1 "Answer matching outperforms multiple choice for language model evaluation")). As shown in Figure [3](#S3.F3 "Figure 3 ‣ Problem: Chance Baselines and The Evaluation-Deployment Gap ‣ 3.1 MCQ Evaluations: High Noise, Low Fidelity ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")a, this creates a ”perceived capability” bubble in which models appear proficient in MCQ formats while failing to produce the same answers in a fully generative regime.

![Refer to caption](/html/2601.02316/assets/figures/ai2d_mcq_vs_gen_by_size_light.png)


((a)) Generative transformation reveals the non-linear capability gap masked by MCQ guessing.

![Refer to caption](/html/2601.02316/assets/figures/circular_eval_vs_vanilla_mcq_by_dataset_light.png)


((b)) Circular evaluation yields a more discriminative signal by filtering for consistent reasoning.

Figure 3: Mitigating performance inflation in multiple-choice formats.

#### Solution: MCQ-to-Generative Transformation and Circular MCQ Evaluation

To bridge this gap, we adopt a two-pronged strategy to ensure measured performance reflects genuine visual reasoning. Wherever viable, we transform MCQs into open-ended generative tasks by removing candidate options and requiring the model to formulate a direct response. To score these free-form outputs without the brittleness of exact-string matching, we employ an LLM-as-judge (specifically Qwen3-30B (Yang et al., [2025](#bib.bib63 "Qwen3 technical report")), a cost-effective and capable judge) to perform semantic answer matching as in Chandak et al. ([2025](#bib.bib1 "Answer matching outperforms multiple choice for language model evaluation")).

We illustrate the impact of this transformation in Figure [3](#S3.F3 "Figure 3 ‣ Problem: Chance Baselines and The Evaluation-Deployment Gap ‣ 3.1 MCQ Evaluations: High Noise, Low Fidelity ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")a, which compares standard MCQ accuracy against our generative transformation across 27 models on the AI2D dataset. We observe a distinct non-linear relationship: while high-performing models (80%+ MCQ accuracy) show tighter convergence between generative and discriminative performance, lower-tier models exhibit a sharp drop-off in the generative setting. This confirms that for weaker models, traditional MCQ benchmarks often mask a fundamental lack of generative skill through random guessing and closed-set shortcuts.

For tasks where options are structurally necessary, specifically inherently discriminative questions like ”Which of the following…” where generative conversion would alter the question’s core intent, we implement Circular Evaluation (Liu et al., [2024](#bib.bib21 "Mmbench: is your multi-modal model an all-around player?")). By rotating option permutations across NN passes and crediting a point only if the model identifies the correct answer across all rotations, we effectively collapse the chance baseline. As shown in Figure [3](#S3.F3 "Figure 3 ‣ Problem: Chance Baselines and The Evaluation-Deployment Gap ‣ 3.1 MCQ Evaluations: High Noise, Low Fidelity ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")b across 27 models, circular evaluation yields a steeper-than-unity slope relative to vanilla MCQ. This slope captures the persistence of the ”false floor” inherent in standard formats: while vanilla MCQs grant models a significant head start (often 20-30% accuracy) through random guessing and position bias, circular evaluation reveals that genuine reasoning capability remains near zero for these same models. The steepness of the curve illustrates that vanilla MCQ continues to significantly inflate perceived performance while true accuracy remains low (<50%<50\%); it is only as models achieve high-level robustness that the two metrics begin to align. By stripping away this artificial inflation, we ensure the benchmark can accurately signal the transition from zero to genuine competence, a critical signal that is otherwise obscured by the noisy MCQ baseline.

Correcting this inflation is crucial: a benchmark is most valuable when it can accurately track the emergence of a new capability. By allowing MCQ formats to provide a ”false floor” of performance, we lose the ability to signal when a model truly transitions from zero to non-zero capability. Ultimately, these stricter criteria ensure that DatBench provides a more faithful representation of genuine model competence by stripping away the artificial inflation inherent in traditional formats.

### 3.2 The Mirage of Visual Understanding

#### Problem: Language Priors are often all you need

A significant challenge in VLM evaluation is “blind solvability”, a phenomenon in which models correctly answer questions without visual input by exploiting the language priors encoded in their LM backbones. This phenomenon fundamentally decouples benchmark performance from actual multimodal reasoning, inadvertently rewarding models with stronger language priors rather than superior visual understanding. This creates a “mirage” of progress, due to which improvements in the vision encoder or cross-modal connector are masked by the overwhelming influence of the text-based backbone. Consequently, models with more capable LMs are often deemed to be stronger VLMs simply because they are better at guessing answers from context.

![Refer to caption](/html/2601.02316/assets/figures/ai2d_no_image_distribution_generative_scoring_function.png)


((a)) AI2D

![Refer to caption](/html/2601.02316/assets/figures/charxiv_descriptive_no_image_distribution_generative_scoring_function.png)


((b)) CharXiv-Descriptive

Figure 4: Language Priors are often all you need: Histograms show the fraction of dataset examples (yy-axis) correctly answered by a specific number of models (xx-axis) using text alone. Red dashed lines denote rejection thresholds (τ\tau): examples solved by more models than τ\tau are discarded. We apply a strict τ=1\tau=1 for generative tasks, while using higher thresholds for MCQ and constrained-answer tasks to adjust for random guessing baselines (c.f. Appendix [F](#A6 "Appendix F Can be solved blind threshold ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"))



| Dataset | Question (text-only part) | Model Output | Ground Truth |
| --- | --- | --- | --- |
| AI2D | How many stages are in the development of a mosquito? | 4 | 4 |
| MathVision | Please solve the problem step by step. Mary has written all the numbers from 1 to 30. How many times has she written digit 2? | To determine how many times the digit 2 appears in the numbers from 1 to 30, we break it down by place value: Units place: The digit 2 appears in 2, 12, and 22 (3 times). Tens place: The digit 2 appears in 20–29 (10 times). Adding both contributions gives 3+10=133+10=13. | 13 |
| VQA-v2 | What color is the toilet? | Most toilets are white, as white is the most common color for bathroom fixtures. Based on this common prior, the toilet is white. | white |

Table 2: Qualitative examples of *blind-solvable* questions. Models correctly answer these questions without access to images, relying solely on language priors, world knowledge, and symbolic reasoning rather than visual understanding.

#### Solution: Filtering Blind-Solvable Questions

To ensure DatBench measures genuine vision-language integration, we systematically identify and remove samples that models can solve “blind.” We conduct a comprehensive evaluation where all 27 models in our suite are queried using only the text prompts from each dataset, without the corresponding image inputs. For each dataset, we visualize this in a histogram (Figure [4](#S3.F4 "Figure 4 ‣ Problem: Language Priors are often all you need ‣ 3.2 The Mirage of Visual Understanding ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")) where the xx-axis represents the number of models answering correctly and the yy-axis represents the fraction of the dataset solved at that model-frequency.

As shown in Table [2](#S3.T2 "Table 2 ‣ Problem: Language Priors are often all you need ‣ 3.2 The Mirage of Visual Understanding ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"), blind-solvable questions typically fall into three categories: (1) World Knowledge, where the answer is physically or culturally standard (e.g., a mosquito’s four-stage life cycle); (2) Visual Stereo-typicality, where models exploit the skewed distribution of attributes in natural images to predict answers without visual confirmation (e.g., toilets usually being white); and (3) Purely Symbolic Reasoning, where the question contains all necessary information for a LLM to solve via logic or arithmetic (e.g., counting digits in a range).

We employ a systematic thresholding strategy (τ\tau) to define rejection criteria based on task format. For datasets with open-ended, generative answers where the probability of a model guessing the exact string is negligible, we set a strict threshold (τ=1\tau=1); any sample solved by even a single model without visual input is discarded (e.g., CharXiv-Descriptive). Conversely, for tasks with a constrained solution space—such as Multiple Choice Questions (MCQ) or specialized counting tasks—we set higher thresholds to account for the increased baseline of random guessing and language priors. This includes datasets like CountBench, where answers are concentrated at low integers, or specific questions in AI2D that feature a limited set of candidate solutions evident from the prompt (see Row 1 of Table [2](#S3.T2 "Table 2 ‣ Problem: Language Priors are often all you need ‣ 3.2 The Mirage of Visual Understanding ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")).

As illustrated in Figure [4](#S3.F4 "Figure 4 ‣ Problem: Language Priors are often all you need ‣ 3.2 The Mirage of Visual Understanding ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")a for AI2D, the distribution shows a significant “tail” of questions solvable by nearly all models without an image. Even in more recent evaluations like CharXiv Descriptive (Figure [4](#S3.F4 "Figure 4 ‣ Problem: Language Priors are often all you need ‣ 3.2 The Mirage of Visual Understanding ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")b), a large fraction of samples are solvable through language priors alone despite the descriptive nature of the task. In the General capability, this issue is most acute: over 70% of examples can be answered without the image. By removing these samples, DatBench ensures the evaluation focuses on high-quality data where visual reasoning is mandatory for success.

### 3.3 Incorrect Ground Truth and Ambiguity

![Refer to caption](/html/2601.02316/assets/figures/ground_truth_quality_vertical_3cat_light.png)


Figure 5: VLM-as-Judge Quality Filtering. Percentage of samples discarded per capability due to ambiguous questions, incorrect ground truth, and samples that are too low resolution (log scale). Spatial capability exhibits the highest discard rate (42.07%), while well-curated capabilities like Grounding and General require minimal filtering (<1%<1\%). Note that a single sample may be labeled under multiple discard categories and is counted in each applicable category.

![Refer to caption](/html/2601.02316/assets/figures/fig5.png)


Figure 6: Top half: Judge identifying incorrect ground-truth samples. Bottom half: Quality control via VLM-as-judge filtering. Examples shown were flagged by unanimous model failure and confirmed as low-quality by the judge.

#### Problem: The Cost of Evaluative Noise

Despite significant curation efforts, many existing VLM benchmarks contain non-trivial proportions of examples with incorrect ground-truth labels, ambiguous questions, or insufficient image resolution to support the required reasoning. Such noise fundamentally compromises benchmark validity; when a dataset punishes a model for providing a correct answer that contradicts a flawed label, it obscures genuine capability gains and encourages “hill-climbing on noise”. Since we source DatBench from a massive aggregate pool of candidate datasets, we have a surplus of examples that allows us to prioritize rigorous data quality over raw quantity.

#### Solution: Two-Stage Quality Filtering with VLM-as-Judge

To identify and purge these artifacts, we employ a two-stage filtering pipeline. In the first stage, we flag examples that all evaluated models (1–10B parameters) answer incorrectly. Unanimous failure across a diverse suite of state-of-the-art models typically indicates either a data quality issue or a genuinely difficult frontier case, both of which warrant closer inspection. In the second stage, a strong VLM judge (GPT-5.2) verifies each flagged sample with access to the ground-truth answer as privileged information.

Our choice of a frontier model as a judge is motivated by prior work suggesting that models are significantly stronger verifiers than generators (Liao et al., [2025b](#bib.bib33 "Can large vision-language models correct semantic grounding errors by themselves?"); Saad-Falcon et al., [2025](#bib.bib34 "Shrinking the generation-verification gap with weak verifiers"); Venktesh et al., [2025](#bib.bib35 "Trust but verify! a survey on verification design for test-time scaling"); Guan et al., [2024](#bib.bib32 "Language models hallucinate, but may excel at fact verification")); we therefore expect the judge to accurately identify errors even in cases that are too challenging for contemporary models to solve autonomously. Given that we operate in a regime of abundant evaluation data across our 9 capabilities, we intentionally err on the side of caution. We adopt a stringent filtering policy, discarding any item flagged as (1) ambiguous, (2) incorrectly labeled, or (3) unsolvable due to insufficient resolution, ensuring that the resulting DatBench subset represents only the highest quality of evaluation data. The impact of this pipeline is most visible in the Spatial capability, which exhibits a 42.07% discard rate, primarily due to insufficient resolution in “in-the-wild” images. Similarly, complex expert-authored sets like ChartQA Pro (17.2% removed) and MMMU-Pro (24.3% removed) show significantly higher noise rates than standard benchmarks (c.f. Appendix [D](#A4 "Appendix D VLM-as-Judge Filtering Results ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations") for per dataset / capability counts of filtered examples). While these high attrition rates reflect significant noise in frontier evaluations, we recognize that a judge might occasionally misinterpret specialized, valid reasoning as a data defect. To maintain evaluative headroom, we retain only the subset of these examples that the judge explicitly validates as correct and unambiguous. Our aggregate data surplus allows us to prioritize this high-fidelity subset, accepting the risk that a conservative filtering policy may sacrifice some valid frontier samples to ensure the remaining benchmark remains strictly noise-free.

### 3.4 High Discrimination with Limited Compute

#### Problem: The Computational Burden of Comprehensive Evaluation

As VLMs grow in sophistication and expand their set of capabilities, comprehensive evaluation imposes a prohibitive computational burden. This is exacerbated by the emergence of “thinking” models; for instance, Bai et al. ([2025](#bib.bib38 "Qwen3-vl technical report")) utilize inference-time compute scaling, often generating chains-of-thought exceeding 32K tokens. Consequently, evaluating a single capability like OCR (often containing >100>100K examples) can require generating over 3 billion tokens, an untenable cost for iterative research.

Selecting a representative subset of examples is a natural approach to reducing evaluation costs. The intuitive heuristic for such a selection is to preserve the model ranking induced by the full dataset, typically quantified using rank correlation measures such as Spearman’s ρ\rho or Kendall’s τ\tau (Spearman, [1904](#bib.bib65 "The proof and measurement of association between two things"); Voorhees, [2001](#bib.bib66 "Evaluation by highly relevant documents"); Buckley and Voorhees, [2004](#bib.bib67 "Retrieval evaluation with incomplete information")). While rank preservation is a necessary condition for a representative subset, it is theoretically insufficient: rank correlation is agnostic to *which* specific samples are retained. In practice, even random subsets can preserve global model rankings by retaining items that separate coarse capability tiers (e.g., small versus large models), while failing to retain the high-discrimination examples needed to distinguish models along the Pareto frontier. More broadly, methods that optimize solely for rank preservation face a fundamental limitation, rank correlation saturates rapidly and is often achieved by subsets whose individual samples are weakly or inconsistently informative about underlying capabilities (Sakai, [2007](#bib.bib68 "On the reliability of information retrieval metrics"); Voorhees, [2001](#bib.bib66 "Evaluation by highly relevant documents")). In such regimes, apparent ranking stability may be driven by spurious correlations or superficial artifacts rather than genuine reasoning ability.

Instead, we turn to Item Response Theory (IRT) for inspiration, originally formalized by Lord ([1952](#bib.bib36 "A theory of test scores")). IRT posits that items differ not just in difficulty, but in item discrimination, a parameter that determines how sharply an item distinguishes between subjects of varying ability levels (Baker, [2001](#bib.bib37 "The basics of item response theory")). However, directly applying standard IRT methodologies (Schick and others, [2025](#bib.bib3 "Fluid language model benchmarking")) to VLM evaluation is often infeasible due to the limited number of diverse observations available per sample in the current research landscape (Bean et al., [2025](#bib.bib22 "Scales++: compute efficient evaluation subset selection with cognitive scales embeddings"); Liao et al., [2025a](#bib.bib64 "Toward a unified framework for data-efficient evaluation of large language models")). Effectively fitting IRT models typically requires stable evaluations from hundreds of diverse state-of-the-art models; without this scale, IRT models become highly sensitive to hyperparameters and are notoriously difficult to fit stably.

Consequently, simply prioritizing rank stability risks overfitting to the evaluated model suite, without guaranteeing the quality or generalizability of the underlying examples. In effect, this produces a “coarse” measuring stick: it yields a subset that is discriminative enough to recover a specific ranking but lacks the resolution to generalize to unseen models or distinguish those with similar capabilities. Therefore, the core optimization problem is not merely to maintain ranking stability, but to maximize total discrimination. By ensuring every sampled example possesses high discriminative power, we can implicitly guarantee robust ranking while maximizing the information content per inference token.

![Refer to caption](/html/2601.02316/assets/figures/discrimination_curve_document_light.png)


((a)) Discrimination vs. Budget

![Refer to caption](/html/2601.02316/assets/figures/correlation_curve_document_full_light.png)


((b)) Rank Correlation vs. Budget

Figure 7: Efficiency and Discrimination Analysis. (a) DatBench (blue) maintains significantly higher discriminative power at low sample budgets compared to random sampling (gray). The peak at  75% budget followed by a dip indicates the removal of anomalous items (negative rp​br\_{pb}) that degrade evaluation quality. (b) Rank correlation saturates rapidly for both methods due to the distinct ability gaps in the model suite, highlighting why correlation alone is an insufficient metric for subset selection. Data shown for Document Understanding; trends are consistent across all capabilities (c.f. Appendix [E](#A5 "Appendix E Item-Discrimination Subset Selection ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")).

#### Solution: Item-Discrimination Based Subset Selection

To avoid the instability of IRT models that are sensitive to hyperparameters and sample size, we operationalize item-discrimination using the point-biserial correlation (rp​br\_{pb}): a robust, hyperparameter-free measure of the association between a binary item response and continuous model capability. Intuitively, rp​br\_{pb} measures the extent to which success on a specific question acts as a proxy for global performance. An item with high rp​br\_{pb} is one that strong models consistently answer correctly and weak models consistently miss; conversely, a low or negative rp​br\_{pb} indicates a noisy item that fails to track with underlying capability. We define total discriminative power as the sum of discrimination of each example (item).

We select subsets by prioritizing examples with the highest rp​br\_{pb} to maximize information density (c.f. Appendix [E](#A5 "Appendix E Item-Discrimination Subset Selection ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")). As demonstrated in Figure [7](#S3.F7 "Figure 7 ‣ Problem: The Computational Burden of Comprehensive Evaluation ‣ 3.4 High Discrimination with Limited Compute ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")a, DatBench achieves approximately 90% of the total discriminative power using only 40% of the full dataset, whereas random sampling scales linearly and provides less than half that signal at the same budget. Notably, our selection curve peaks above 1.0 before the full dataset is included; this occurs because we intentionally deprioritize ”anomalous items” at the end of our selection process. These are questions with negative rp​br\_{pb} where weaker models outperform stronger ones—likely due to spurious text-based correlations, prompt sensitivity, or test-set leakage—which effectively introduce noise into the evaluation.

While Figure [7](#S3.F7 "Figure 7 ‣ Problem: The Computational Burden of Comprehensive Evaluation ‣ 3.4 High Discrimination with Limited Compute ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")b shows that both random and discriminative subsets rapidly achieve high rank correlation, this similarity is deceptive. Because our model suite contains distinct performance tiers (e.g., 1B vs. 8B), the global ranking is easily recovered even by uninformative samples. Rank correlation is thus a ”low-bar” metric that saturates too quickly to reflect subset quality. By maximizing discrimination, DatBench provides a higher-fidelity instrument that remains sensitive to marginal capability gains and ensures that evaluation remains stable across unseen model architectures.

## 4 Introducing DatBench and DatBench-Full

By applying our four-stage pipeline: MCQ transformation (Section [3.1](#S3.SS1 "3.1 MCQ Evaluations: High Noise, Low Fidelity ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")), blind-solvability filtering (Section [3.2](#S3.SS2 "3.2 The Mirage of Visual Understanding ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")), quality filtering (Section [3.3](#S3.SS3 "3.3 Incorrect Ground Truth and Ambiguity ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")), and discriminative selection (Section [3.4](#S3.SS4 "3.4 High Discrimination with Limited Compute ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")), we transform noisy, redundant dataset aggregations into precise evaluation artifacts. These artifacts cover nine distinct capabilities: Chart Understanding, Document Understanding, Scene OCR, Grounding, Counting, Spatial Reasoning, Math & Logic, Diagrams & Tables, and General VQA. We release two versions of the benchmark to cater to varying computational budgets.

For the final DatBench subset, we execute steps 1 through 4. However, the discrimination-based selection in Step 4 naturally discards ”frontier” examples—items that all evaluated models fail—as they offer near-zero discrimination by construction. To prevent benchmark saturation and ensure evaluative headroom for future models, we manually allocate up to 20% of the DatBench subsets for these valid frontier cases, specifically those verified by our VLM-as-judge as correct and unambiguous. This strategic inclusion ensures that DatBench maintains a high difficulty ceiling while remaining a robust instrument for measuring progress at the frontier of vision-language modeling.

* •

  DatBench: Our primary, high-efficiency subset tailored for rapid iterative development. Constructed via item-wise point-biserial correlation (rp​br\_{pb}), this set maintains high ranking fidelity while minimizing inference costs. We explicitly retain a partition of verified, high-quality “frontier” examples—currently unsolvable by 1B–10B models—to ensure the benchmark remains an effective measuring stick as model capabilities scale.
* •

  DatBench-Full: The complete aggregation of all high-quality samples remaining after our systematic filtering pipeline (Steps 1–3). While these sets include all examples validated as objectively high-quality, their scale varies significantly across capabilities based on the severity of the filtering required. For capabilities such as Counting and Spatial Reasoning, where high noise and blind-solvability rates resulted in massive attrition, DatBench-Full is comparable in size to the DatBench subset. However, for most capabilities, DatBench-Full evaluation sets are an order of magnitude larger, reaching up to 50×50\times the size of their efficient counterparts. These are intended for extensive, fine-grained error analysis and established as a comprehensive resource for deep-dive capability assessment.

#### Usage Guide.

We recommend DatBench in high-iteration contexts such as training loops and ablation studies, in which compute costs for evaluation can rapidly balloon but discriminative signal should be maximized. DatBench-Full should be reserved for final model reporting where computational constraints are relaxed and maximum coverage is desired. Collectively, these artifacts transition multimodal evaluation from a regime of noisy data to one of precise measurement.

![Refer to caption](/html/2601.02316/assets/figures/bigtable.png)


Figure 8: DatBench vs. Original Performance. We plot accuracy on DatBench (yy-axis) against original baselines (xx-axis) for 27 models, demonstrating the impact of our refinements on evaluative faithfulness and discrimination. Points below the diagonal indicate a more rigorous evaluation following the removal of non-discriminative and blind-solvable items, while a larger slope and higher dispersion—most notable in Document Understanding and Math—reveal a higher-resolution instrument that exposes latent differences between models previously masked by noise. Conversely, upward shifts in categories like Counting reflect increased faithfulness achieved by purging incorrectly labeled artifacts that penalized correct reasoning. Despite these interventions, the consistency of model clusters across all nine plots confirms that our methodology establishes a discriminative and efficient evaluation that accurately captures model rankings while remaining sensitive to marginal performance gains.

Having established these artifacts, we provide a comprehensive statistical analysis of how DatBench transforms raw benchmark data into faithful and discriminative instruments for the efficient estimation of VLM capabilities.

#### DatBench discards samples that are too easy / too hard.

The most immediate impact of our filtering is the removal of samples that act as statistical noise. In the General capability, model performance is significantly shifted downward from the y=xy=x diagonal (Figure [8](#S4.F8 "Figure 8 ‣ Usage Guide. ‣ 4 Introducing DatBench and DatBench-Full ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")), a direct result of Stage 2 filtering (c.f. Appendix [F](#A6 "Appendix F Can be solved blind threshold ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")) which discarded 72.07% of samples solvable via language priors alone. Conversely, the Spatial Reasoning capability underwent rigorous quality filtering in Stage 3 (c.f. Appendix [D](#A4 "Appendix D VLM-as-Judge Filtering Results ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")), with 42.07% of samples removed due to ambiguity or insufficient resolution. This systematic removal of evaluative noise shifts model assessments to a more faithful performance tier, ensuring that benchmark outcomes accurately reflect genuine multimodal reasoning.

#### DatBench is more discriminative.

Our item-selection methodology amplifies performance differences between models, increasing measurement resolution. On the original General benchmarks, models compress into a narrow 65–80% accuracy band; on DatBench, they spread across 10–65%, a nearly 4× expansion in effective score range. This “stretching” reflects our point-biserial selection criterion (Section 3.4): by retaining only items where strong models reliably succeed and weak models reliably fail, small capability differences that were previously masked now manifest as measurable gaps. The steep slopes observed in General and Document Understanding (Figure [8](#S4.F8 "Figure 8 ‣ Usage Guide. ‣ 4 Introducing DatBench and DatBench-Full ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")) confirm this effect; equivalent spacing on the original benchmarks translates to greater separation on DatBench.

#### DatBench preserves discrimination power with far fewer samples.

Despite aggressive filtering, ranking stability is maintained. For capabilities such as Chart Understanding and Grounding, DatBench points fall almost perfectly on the y=xy=x line (Figure [8](#S4.F8 "Figure 8 ‣ Usage Guide. ‣ 4 Introducing DatBench and DatBench-Full ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")), confirming that the subset preserves the discriminability and model rankings from the full dataset. As shown in our Stage 4 efficiency analysis (c.f. Appendix [E](#A5 "Appendix E Item-Discrimination Subset Selection ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")), DatBench maintains high total discriminative power even at severely restricted budgets, whereas random sampling suffers from linear signal degradation.

#### Limitations and Future Directions.

While our methodology offers a substantial leap forward, several avenues remain for future exploration:

* •

  Scaling to Larger Regimes: Our current analysis focuses on models in the 1B–10B parameter range and inference traces within standard context windows. While the methodology is scale-invariant, the specific subsets of highly discriminative questions will likely shift for larger models and extended inference budgets (e.g., exceeding the 4096 tokens used in our work). Future work can apply this pipeline to larger model families and longer reasoning traces to identify the discriminative frontier for state-of-the-art systems.
* •

  Diversity Guarantees: Our current subset selection prioritizes the highest individual discrimination scores, which implicitly relies on the inherent variety of the source data rather than an explicit diversity constraint. Consequently, this objective does not formally account for redundancy between samples; in pathological cases (e.g., duplicate but highly discriminative examples), the selection could theoretically yield a degenerate or repetitive subset. While we mitigate this through rigorous initial curation, future iterations could incorporate explicit diversity-aware objectives to ensure broader coverage of the capability space.
* •

  Expanding Capabilities: We aim to extend our capability map beyond static images to include long-form video understanding, UI/GUI grounding, and robotics perception.
* •

  DatBench-Live: Finally, discrimination is a moving target; questions that distinguish today’s models will eventually become trivial. We envision a dynamic, “living” benchmark where subsets are recomputed periodically as new models shift the capability distribution and new datasets emerge.

## 5 Diagnosing VLM Pathologies with DatBench

In this section, we leverage the high-signal artifacts produced by the DatBench pipeline to diagnose the behavioral pathologies of modern VLMs. By analyzing performance across 27 state-of-the-art models spanning the 1B–10B parameter range, we uncover fundamental trade-offs between semantic reasoning and perceptual grounding, risks and rewards of inference-time scaling, and the impact of language priors on evaluation metrics.

#### Takeaway 1: Capability Correlations Reveal a ”Reasoning vs. Perception” Trade-off.

To identify hidden relationships between tasks, we calculated Pearson correlations (rr) between all capability scores across our model suite (Figure [9(a)](#S5.F9.sf1 "In Figure 9 ‣ Takeaway 1: Capability Correlations Reveal a ”Reasoning vs. Perception” Trade-off. ‣ 5 Diagnosing VLM Pathologies with DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")). We identify a tight Reasoning Cluster in which Chart Understanding, Math, and General VQA exhibit exceptionally high pairwise correlations, such as r=0.90r=0.90 between Chart and General tasks. This analysis confirms that General VQA benchmarks, such as MMBench and MMMU-Pro, primarily test abstract reasoning capabilities that are also fundamental to Math, evidenced by a strong correlation of r=0.76r=0.76 between these two domains. Furthermore, a distinct Spatial-Semantic Trade-off exists: Grounding correlates negatively with text-heavy tasks like Document Understanding (r=−0.29r=-0.29) and OCR (r=−0.19r=-0.19). These negative relationships, alongside the inverse correlation between Math and Spatial Reasoning (r=−0.19r=-0.19), suggest a latent conflict in current training paradigms between high-level semantic processing and low-level perceptual fidelity. Hierarchical clustering (Figure [9(b)](#S5.F9.sf2 "In Figure 9 ‣ Takeaway 1: Capability Correlations Reveal a ”Reasoning vs. Perception” Trade-off. ‣ 5 Diagnosing VLM Pathologies with DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")) corroborates this dichotomy, revealing two distinct clusters: reasoning (Chart, Math, General) and perception (OCR, Spatial, Diagram).

![Refer to caption](/html/2601.02316/assets/figures/capability_heatmap.png)


((a)) Strong links exist between reasoning tasks, while spatial tasks often conflict with text-heavy ones.

![Refer to caption](/html/2601.02316/assets/figures/dendrogram.png)


((b)) Hierarchical clustering (average linkage, distance =1−r=1-r) reveals two main clusters: reasoning (Chart, Math, General) and perception (OCR, Spatial, Diagram).

Figure 9: Correlation analysis of model capabilities across 26 vision-language models. Pairwise Pearson correlations are computed between mean accuracy scores for each capability. (a) Capability correlations. (b) Capability clustering.

![Refer to caption](/html/2601.02316/assets/figures/radar.png)


Figure 10: Capability Profiles. Models show clear trade-offs between general reasoning and specialized visual tasks.

#### Takeaway 2: Capability Profiles Reveal Specialist-Generalist Trade-offs.

This trade-off manifests in distinct model archetypes (Figure [10](#S5.F10 "Figure 10 ‣ Takeaway 1: Capability Correlations Reveal a ”Reasoning vs. Perception” Trade-off. ‣ 5 Diagnosing VLM Pathologies with DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")). GLM-4.1V-9B acts as a perception specialist, leading in diagram understanding (66.4%) and spatial reasoning (36.8%) but struggling with math (17.4%). Balanced generalists are rare: Qwen3-VL-4B is a notable exception, maintaining strong document understanding (71.0%), OCR (77.9%), and reasoning (59.9%). Most tellingly, R-4B reaches the highest math score (43.4%) at the cost of the lowest spatial performance (11.4%), suggesting that (current) reasoning-focused training can degrade visual grounding.

![Refer to caption](/html/2601.02316/assets/figures/thinking_vs_no_thinking_delta_light.png)


((a)) Inference Scaling Impact

![Refer to caption](/html/2601.02316/assets/figures/thinking_vs_nonthinking_tokens_light.png)


((b)) Computational Cost of Errors

Figure 11: The Overthinking Penalty. (a) Scaling compute helps reasoning but hurts perception. (b) Incorrect ”thinking” responses use ≈14×\approx 14\times more tokens than standard models.

#### Takeaway 3: The ”Overthinking” Penalty: Inference-Time Scaling Degrades Perception at High Cost.

Comparing Thinking models to standard counterparts reveals that extra test-time compute is a double-edged sword (Figure [11(a)](#S5.F11.sf1 "In Figure 11 ‣ Takeaway 2: Capability Profiles Reveal Specialist-Generalist Trade-offs. ‣ 5 Diagnosing VLM Pathologies with DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")). To quantify this, we define the Thinking relative advantage as the percentage gain in accuracy of the thinking model over its instruct counterpart, normalized by the instruct baseline: (A​c​ct​h​i​n​k​i​n​g−A​c​ci​n​s​t​r​u​c​t)/A​c​ci​n​s​t​r​u​c​t×100(Acc\_{thinking}-Acc\_{instruct})/Acc\_{instruct}\times 100. Scaling helps Math (≈+36.8%\approx+36.8\%) and Charts (≈+10.8%\approx+10.8\%) but causes massive regressions in OCR (≈−53.5%\approx-53.5\%) and Document Understanding (≈−47.8%\approx-47.8\%). This regression is also extremely computationally wasteful: while correct thinking answers use ≈425\approx 425 tokens, incorrect attempts balloon to ≈1196.9\approx 1196.9 tokens, a ≈14×\approx 14\times increase over non-thinking models (Figure [11(b)](#S5.F11.sf2 "In Figure 11 ‣ Takeaway 2: Capability Profiles Reveal Specialist-Generalist Trade-offs. ‣ 5 Diagnosing VLM Pathologies with DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")). We observed that this is due to models entering unproductive thinking loops on perceptual tasks they cannot solve. Prior work has observed a similar overthinking penalty for language models (Hochlehnert et al., [2025](#bib.bib83 "A sober look at progress in language model reasoning: pitfalls and paths to reproducibility"); Su et al., [2025](#bib.bib80 "Between underthinking and overthinking: an empirical study of reasoning length and correctness in llms"); Wang et al., [2025](#bib.bib81 "Thoughts are all over the place: on the underthinking of o1-like llms"); Wu et al., [2025](#bib.bib82 "When more is less: understanding chain-of-thought length in llms")).

![Refer to caption](/html/2601.02316/assets/figures/vision_delta_by_capability.png)


Figure 12: Vision vs. Text Priors. Tasks like Counting require seeing the image, whereas Math can often be solved by language prior alone.

#### Takeaway 4: Language Priors Mask True Multimodal Performance across Capabilities.

To isolate the actual visual requirement of each task, we analyze the vision delta (VΔV\_{\Delta}), defined as the performance gap between standard multimodal evaluation and a blind text-only baseline (Figure [12](#S5.F12 "Figure 12 ‣ Takeaway 3: The ”Overthinking” Penalty: Inference-Time Scaling Degrades Perception at High Cost. ‣ 5 Diagnosing VLM Pathologies with DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")). Our results show that reliance on language priors varies drastically by capability, often distorting perceived progress in multimodal reasoning. Capabilities such as Counting (VΔ=60.2%V\_{\Delta}=60.2\%) and Grounding (VΔ=42.3%V\_{\Delta}=42.3\%) exhibit high vision dependency, making them the most faithful indicators of true perceptual accuracy. Conversely, Math (VΔ=13.0%V\_{\Delta}=13.0\%) and Spatial Reasoning (VΔ=14.9%V\_{\Delta}=14.9\%) show significant language prior distortion, relying heavily on textual patterns that allow models to guess correctly without the image. These findings confirm that without the rigorous filtering introduced in DatBench, i.e. discarding samples that can be solved with the language prior alone, high scores in capabilities like Math may inadvertently reward stronger language models rather than superior vision-language integration.

## 6 Conclusion

In this work, we addressed the dual challenges of data quality and computational cost in the evaluation of Vision-Language Models (VLMs). We introduced a framework of three desiderata that evaluations should satisfy: (1) faithfulness to the modality and application, (2) discriminability between models of varying quality, and (3) efficiency in compute. We then applied this lens to expose four critical pathologies in existing benchmarks: multiple-choice formats are both unfaithful and weakly discriminative; many VLM benchmarks can be solved without vision; incorrect and ambiguous ground truth introduces substantial noise; and existing evaluation suites are inefficient. We used these insights to distill these benchmarks into high-signal evaluation suites.

Our primary contribution, DatBench, serves as a precise, psychometrically grounded instrument for measuring multimodal capability. Motivated by Item Response Theory (IRT) and operationalizing discrimination via point‑biserial correlation (rpb), we demonstrated that maximizing total test discrimination yields subsets that are not only computationally lightweight but also significantly more robust and generalizable than those derived via random sampling or simple rank correlation. Our accompanying analysis of “thinking” models and language priors further validates that DatBench is capable of surfacing nuanced behavioral insights that are often obscured in aggregate metrics. We release two versions of the benchmark, the efficiency-focused DatBench for rapid iterative development (yielding 13×\times average speedup), and the comprehensive DatBench-Full for final reporting, to standardize comparison and accelerate progress at the pareto frontier. Our work provides a path towards evaluation practices that are both rigorous and sustainable as VLMs continue to scale.

## 7 Contributions and Acknowledgements

|  |  |
| --- | --- |
| Core Contributors | Siddharth Joshi, Haoli Yin, Rishabh Adiga, and Ricardo Monti. |
|  | *for fusing modalities, wrangling the datasets, and ensuring the evaluation pipeline didn’t hallucinate.* |
| Technical Contributors | Aldo Carranza, Alex Fang, Alvin Deng, Amro Abbas, Brett Larsen, Cody Blakeney, Darren Teh, David Schwab, Fan Pan, Haakon Mongstad, Jack Urbanek, Jason Lee, Jason Telanoff, Josh Wills, Kaleigh Mentzer, Luke Merrick, Parth Doshi, Paul Burstein, Pratyush Maini, Scott Loftin, Spandan Das, Tony Jiang, Vineeth Dorna, and Zhengping Wang. |
|  | *the ensemble of experts who maximized our few-shot performance and ablated every hyperparameter.* |
| Leadership | Bogdan Gaza, Ari Morcos, and Matthew Leavitt. |
|  | *the ground truth oracles who steered the project and prevented collective mode collapse.* |
| Acknowledgements | Liz Gatapia (for incredible logo design), Jacqueline Liu, Tiffanie Pham, Sylvia Hoang, Jayla Lindsey, Kylie Clement, Elise Clark |
|  | *the human-in-the-loop feedback that provided essential regularization and support.* |

## References

* A. Abbas, K. Tirumala, D. Simig, S. Ganguli, and A. S. Morcos (2023)
  SemDeDup: data-efficient learning at web-scale through semantic deduplication.
  External Links: 2303.09540,
  [Link](https://arxiv.org/abs/2303.09540)
  Cited by: [§1](#S1.p5.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* M. Acharya, K. Kafle, and C. Kanan (2019)
  TallyQA: answering complex counting questions.
  In AAAI,
  Cited by: [Table 1](#S3.T1.3.1.25.24.2 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* R. Adiga, B. Nushi, and V. Chandrasekaran (2025)
  Attention speaks volumes: localizing and mitigating bias in language models.
  In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar (Eds.),
  Vienna, Austria,  pp. 26403–26423.
  External Links: [Link](https://aclanthology.org/2025.acl-long.1281/),
  [Document](https://dx.doi.org/10.18653/v1/2025.acl-long.1281),
  ISBN 979-8-89176-251-0
  Cited by: [§2](#S2.p1.1 "2 Related Work ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, W. Ge, Z. Guo, Q. Huang, J. Huang, F. Huang, B. Hui, S. Jiang, Z. Li, M. Li, M. Li, K. Li, Z. Lin, J. Lin, X. Liu, J. Liu, C. Liu, Y. Liu, D. Liu, S. Liu, D. Lu, R. Luo, C. Lv, R. Men, L. Meng, X. Ren, X. Ren, S. Song, Y. Sun, J. Tang, J. Tu, J. Wan, P. Wang, P. Wang, Q. Wang, Y. Wang, T. Xie, Y. Xu, H. Xu, J. Xu, Z. Yang, M. Yang, J. Yang, A. Yang, B. Yu, F. Zhang, H. Zhang, X. Zhang, B. Zheng, H. Zhong, J. Zhou, F. Zhou, J. Zhou, Y. Zhu, and K. Zhu (2025)
  Qwen3-vl technical report.
  External Links: 2511.21631,
  [Link](https://arxiv.org/abs/2511.21631)
  Cited by: [§1](#S1.p4.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"),
  [§3.4](#S3.SS4.SSS0.Px1.p1.1 "Problem: The Computational Burden of Comprehensive Evaluation ‣ 3.4 High Discrimination with Limited Compute ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* F. B. Baker (2001)
  The basics of item response theory.
   ERIC.
  Cited by: [§3.4](#S3.SS4.SSS0.Px1.p3.1 "Problem: The Computational Burden of Comprehensive Evaluation ‣ 3.4 High Discrimination with Limited Compute ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* A. M. Bean, N. Seedat, S. Chen, and J. R. Schwarz (2025)
  Scales++: compute efficient evaluation subset selection with cognitive scales embeddings.
  arXiv preprint arXiv:2510.26384.
  Cited by: [§2](#S2.p5.1 "2 Related Work ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"),
  [§3.4](#S3.SS4.SSS0.Px1.p3.1 "Problem: The Computational Burden of Comprehensive Evaluation ‣ 3.4 High Discrimination with Limited Compute ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* C. Buckley and E. M. Voorhees (2004)
  Retrieval evaluation with incomplete information.
  In SIGIR,
  Cited by: [§3.4](#S3.SS4.SSS0.Px1.p2.2 "Problem: The Computational Burden of Comprehensive Evaluation ‣ 3.4 High Discrimination with Limited Compute ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* N. Chandak, S. Goel, A. Prabhu, M. Hardt, and J. Geiping (2025)
  Answer matching outperforms multiple choice for language model evaluation.
  arXiv preprint arXiv:2507.02856.
  Cited by: [§2](#S2.p1.1 "2 Related Work ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"),
  [§3.1](#S3.SS1.SSS0.Px3.p1.4 "Problem: Chance Baselines and The Evaluation-Deployment Gap ‣ 3.1 MCQ Evaluations: High Noise, Low Fidelity ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"),
  [§3.1](#S3.SS1.SSS0.Px4.p1.1 "Solution: MCQ-to-Generative Transformation and Circular MCQ Evaluation ‣ 3.1 MCQ Evaluations: High Noise, Low Fidelity ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* DatologyAI, :, P. Maini, V. Dorna, P. Doshi, A. Carranza, F. Pan, J. Urbanek, P. Burstein, A. Fang, A. Deng, A. Abbas, B. Larsen, C. Blakeney, C. Bannur, C. Baek, D. Teh, D. Schwab, H. Mongstad, H. Yin, J. Wills, K. Mentzer, L. Merrick, R. Monti, R. Adiga, S. Joshi, S. Das, Z. Wang, B. Gaza, A. Morcos, and M. Leavitt (2025)
  BeyondWeb: lessons from scaling synthetic data for trillion-scale pretraining.
  External Links: 2508.10975,
  [Link](https://arxiv.org/abs/2508.10975)
  Cited by: [§1](#S1.p5.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* DatologyAI, A. Abbas, J. Wills, H. Yin, P. Burstein, N. Cao, A. Carranza, A. Deng, P. Goyal, P. Maini, J. McGrath, F. Pan, J. Urbanek, V. Kada, M. Razzak, V. Shah, V. Veerendranath, B. Gaza, A. Morcos, and M. Leavitt (2024a)
  DatologyAI Technical Deep-Dive: Image-Text Data Curation at the Billion-Sample Scale.
  Technical report
   DatologyAI.
  External Links: [Link](https://www.datologyai.com/blog/productionized-multimodal-data-curation-at-the-billion-sample-scale)
  Cited by: [§1](#S1.p5.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* DatologyAI, A. Carranza, A. Deng, P. Maini, M. Razzak, J. Urbanek, A. Abbas, P. Burstein, N. Cao, P. Goyal, J. McGrath, F. Pan, J. Wills, H. Yin, V. Kada, V. Shah, V. Veerendranath, B. Gaza, A. Morcos, and M. Leavitt (2024b)
  DatologyAI Technical Deep-Dive: Curating Our Way Curation to a Billion-State-of-the-Art Text Dataset.
  Technical report
   DatologyAI.
  External Links: [Link](https://www.datologyai.com/blog/technical-deep-dive-curating-our-way-to-a-state-of-the-art-text-dataset)
  Cited by: [§1](#S1.p5.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* M. Deitke, C. Clark, S. Lee, R. Tripathi, Y. Yang, J. S. Park, M. Salehi, N. Muennighoff, K. Lo, L. Soldaini, J. Lu, T. Anderson, E. Bransom, K. Ehsani, H. Ngo, Y. Chen, A. Patel, M. Yatskar, C. Callison-Burch, A. Head, R. Hendrix, F. Bastani, E. VanderBilt, N. Lambert, Y. Chou, A. Chheda, J. Sparks, S. Skjonsberg, M. Schmitz, A. Sarnat, B. Bischoff, P. Walsh, C. Newell, P. Wolters, T. Gupta, K. Zeng, J. Borchardt, D. Groeneveld, C. Nam, S. Lebrecht, C. Wittlif, C. Schoenick, O. Michel, R. Krishna, L. Weihs, N. A. Smith, H. Hajishirzi, R. Girshick, A. Farhadi, and A. Kembhavi (2024)
  Molmo and pixmo: open weights and open data for state-of-the-art vision-language models.
  External Links: 2409.17146,
  [Link](https://arxiv.org/abs/2409.17146)
  Cited by: [Table 1](#S3.T1.3.1.23.22.2 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* A. Fang, A. M. Jose, A. Jain, L. Schmidt, A. Toshev, and V. Shankar (2023)
  Data filtering networks.
  External Links: 2309.17425,
  [Link](https://arxiv.org/abs/2309.17425)
  Cited by: [§1](#S1.p5.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* L. Fu, Z. Kuang, J. Song, M. Huang, B. Yang, Y. Li, L. Zhu, Q. Luo, X. Wang, H. Lu, Z. Li, G. Tang, B. Shan, C. Lin, Q. Liu, B. Wu, H. Feng, H. Liu, C. Huang, J. Tang, W. Chen, L. Jin, Y. Liu, and X. Bai (2025)
  OCRBench v2: an improved benchmark for evaluating large multimodal models on visual text localization and reasoning.
  External Links: 2501.00321,
  [Link](https://arxiv.org/abs/2501.00321)
  Cited by: [Table 1](#S3.T1.3.1.8.7.1 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* A. Ghosh, S. Dziadzio, A. Prabhu, V. Udandarao, S. Albanie, and M. Bethge (2025)
  ONEBench to test them all: sample-level benchmarking over open-ended capabilities.
  External Links: 2412.06745,
  [Link](https://arxiv.org/abs/2412.06745)
  Cited by: [§2](#S2.p6.1 "2 Related Work ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh (2017)
  Making the V in VQA matter: elevating the role of image understanding in Visual Question Answering.
  In Conference on Computer Vision and Pattern Recognition (CVPR),
  Cited by: [§1](#S1.p8.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"),
  [Table 1](#S3.T1.3.1.30.29.1 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* J. Guan, J. Dodge, D. Wadden, M. Huang, and H. Peng (2024)
  Language models hallucinate, but may excel at fact verification.
  In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), K. Duh, H. Gomez, and S. Bethard (Eds.),
  Mexico City, Mexico,  pp. 1090–1111.
  External Links: [Link](https://aclanthology.org/2024.naacl-long.62/),
  [Document](https://dx.doi.org/10.18653/v1/2024.naacl-long.62)
  Cited by: [§3.3](#S3.SS3.SSS0.Px2.p2.1 "Solution: Two-Stage Quality Filtering with VLM-as-Judge ‣ 3.3 Incorrect Ground Truth and Ambiguity ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* A. Hochlehnert, H. Bhatnagar, V. Udandarao, S. Albanie, A. Prabhu, and M. Bethge (2025)
  A sober look at progress in language model reasoning: pitfalls and paths to reproducibility.
  External Links: 2504.07086,
  [Link](https://arxiv.org/abs/2504.07086)
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.8 "Takeaway 3: The ”Overthinking” Penalty: Inference-Time Scaling Degrades Perception at High Cost. ‣ 5 Diagnosing VLM Pathologies with DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* S. Joshi, A. Jain, A. Payani, and B. Mirzasoleiman (2024)
  Data-efficient contrastive language-image pretraining: prioritizing data quality over quantity.
  In Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, S. Dasgupta, S. Mandt, and Y. Li (Eds.),
  Proceedings of Machine Learning Research, Vol. 238,  pp. 1000–1008.
  External Links: [Link](https://proceedings.mlr.press/v238/joshi24a.html)
  Cited by: [§1](#S1.p5.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* S. Joshi and B. Mirzasoleiman (2023)
  Data-efficient contrastive self-supervised learning: most beneficial examples for supervised learning contribute the least.
  In Proceedings of the 40th International Conference on Machine Learning, A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett (Eds.),
  Proceedings of Machine Learning Research, Vol. 202,  pp. 15356–15370.
  External Links: [Link](https://proceedings.mlr.press/v202/joshi23b.html)
  Cited by: [§1](#S1.p5.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* S. Joshi, J. Ni, and B. Mirzasoleiman (2025a)
  Dataset distillation via knowledge distillation: towards efficient self-supervised pre-training of deep networks.
  External Links: 2410.02116,
  [Link](https://arxiv.org/abs/2410.02116)
  Cited by: [§1](#S1.p5.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* S. Joshi, B. Nushi, V. Balachandran, V. Chandrasekaran, V. Vineet, N. Joshi, and B. Mirzasoleiman (2025b)
  MM-gen: enhancing task performance through targeted multimodal data curation.
  External Links: 2501.04155,
  [Link](https://arxiv.org/abs/2501.04155)
  Cited by: [§1](#S1.p5.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* S. Kazemzadeh, V. Ordonez, M. Matten, and T. Berg (2014)
  ReferItGame: referring to objects in photographs of natural scenes.
  In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), A. Moschitti, B. Pang, and W. Daelemans (Eds.),
  Doha, Qatar,  pp. 787–798.
  External Links: [Link](https://aclanthology.org/D14-1086/),
  [Document](https://dx.doi.org/10.3115/v1/D14-1086)
  Cited by: [Table 1](#S3.T1.3.1.19.18.2 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"),
  [Table 1](#S3.T1.3.1.20.19.1 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* A. Kembhavi, M. Salvato, E. Kolve, M. Seo, H. Hajishirzi, and A. Farhadi (2016)
  A diagram is worth a dozen images.
  External Links: 1603.07396,
  [Link](https://arxiv.org/abs/1603.07396)
  Cited by: [Table 1](#S3.T1.3.1.26.25.2 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* N. Lambert (2025)
  Good researchers obsess over evals: the story of OLMo 3 (post-training), told through evals.
   Allen Institute for AI.
  Note: Presentation at Evaluating the Evolving LLM Lifecycle Workshop, NeurIPS 2025Slide 20
  External Links: [Link](https://docs.google.com/presentation/d/10DT7MM8vbKFrxPiwAfcyvea0V4d6XpFBhe6_Pmneu2A/)
  Cited by: [§1](#S1.p4.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* K. Lee, M. Kim, S. Yoon, M. Kim, D. Lee, H. Koh, and K. Jung (2025)
  VLind-bench: measuring language priors in large vision-language models.
  In Findings of the Association for Computational Linguistics: NAACL 2025, L. Chiruzzo, A. Ritter, and L. Wang (Eds.),
  Albuquerque, New Mexico,  pp. 4129–4144.
  External Links: [Link](https://aclanthology.org/2025.findings-naacl.231/),
  [Document](https://dx.doi.org/10.18653/v1/2025.findings-naacl.231),
  ISBN 979-8-89176-195-7
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* J. Li, W. Lu, H. Fei, M. Luo, M. Dai, M. Xia, Y. Jin, Z. Gan, D. Qi, C. Fu, Y. Tai, W. Yang, Y. Wang, and C. Wang (2024a)
  A survey on benchmarks of multimodal large language models.
  External Links: 2408.08632,
  [Link](https://arxiv.org/abs/2408.08632)
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* X. Li, Y. Li, R. Zhang, J. Zhou, and M. Sun (2024b)
  Can multiple-choice questions really be useful in detecting the abilities of llms?.
  In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING),
  External Links: [Link](https://aclanthology.org/2024.lrec-main.251/)
  Cited by: [§2](#S2.p2.1 "2 Related Work ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* L. Liao, Q. Zhang, R. Wu, and G. Fang (2025a)
  Toward a unified framework for data-efficient evaluation of large language models.
  External Links: 2510.04051,
  [Link](https://arxiv.org/abs/2510.04051)
  Cited by: [§3.4](#S3.SS4.SSS0.Px1.p3.1 "Problem: The Computational Burden of Comprehensive Evaluation ‣ 3.4 High Discrimination with Limited Compute ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* Y. Liao, R. Mahmood, S. Fidler, and D. Acuna (2025b)
  Can large vision-language models correct semantic grounding errors by themselves?.
  In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
   pp. 14667–14678.
  Cited by: [§3.3](#S3.SS3.SSS0.Px2.p2.1 "Solution: Two-Stage Quality Filtering with VLM-as-Judge ‣ 3.3 Incorrect Ground Truth and Ambiguity ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* Z. Lin, X. Chen, D. Pathak, P. Zhang, and D. Ramanan (2024)
  Revisiting the role of language priors in vision-language models.
  External Links: 2306.01879,
  [Link](https://arxiv.org/abs/2306.01879)
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu, et al. (2024)
  Mmbench: is your multi-modal model an all-around player?.
  In European conference on computer vision,
   pp. 216–233.
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"),
  [§1](#S1.p7.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"),
  [§2](#S2.p1.1 "2 Related Work ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"),
  [§3.1](#S3.SS1.SSS0.Px4.p3.2 "Solution: MCQ-to-Generative Transformation and Circular MCQ Evaluation ‣ 3.1 MCQ Evaluations: High Noise, Low Fidelity ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"),
  [Table 1](#S3.T1.3.1.29.28.1 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* F. M. Lord (1952)
  A theory of test scores.
  Psychometrika measures 7 (1).
  Cited by: [§3.4](#S3.SS4.SSS0.Px1.p3.1 "Problem: The Computational Burden of Comprehensive Evaluation ‣ 3.4 High Discrimination with Limited Compute ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* P. Lu, H. Bansal, T. Xia, J. Liu, C. Li, H. Hajishirzi, H. Cheng, K. Chang, M. Galley, and J. Gao (2024)
  MathVista: evaluating mathematical reasoning of foundation models in visual contexts.
  In International Conference on Learning Representations (ICLR),
  Cited by: [Table 1](#S3.T1.3.1.13.12.2 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* Y. Lu, M. Bartolo, A. Moore, S. Riedel, and P. Stenetorp (2022)
  Fantastically ordered prompts and where to find them: overcoming few-shot prompt order sensitivity.
  External Links: 2104.08786,
  [Link](https://arxiv.org/abs/2104.08786)
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* J. Mao, J. Huang, A. Toshev, O. Camburu, A. L. Yuille, and K. Murphy (2016)
  Generation and comprehension of unambiguous object descriptions.
  In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
  Cited by: [Table 1](#S3.T1.3.1.21.20.1 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* A. Masry, M. S. Islam, M. Ahmed, A. Bajaj, F. Kabir, A. Kartha, M. T. R. Laskar, M. Rahman, S. Rahman, M. Shahmohammadi, M. Thakkar, M. R. Parvez, E. Hoque, and S. Joty (2025)
  ChartQAPro: a more diverse and challenging benchmark for chart question answering.
  External Links: 2504.05506,
  [Link](https://arxiv.org/abs/2504.05506)
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"),
  [Table 1](#S3.T1.3.1.3.2.1 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* A. Masry, D. X. Long, J. Q. Tan, S. Joty, and E. Hoque (2022)
  ChartQA: a benchmark for question answering about charts with visual and logical reasoning.
  External Links: 2203.10244,
  [Link](https://arxiv.org/abs/2203.10244)
  Cited by: [Table 1](#S3.T1.3.1.2.1.2 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* M. Mathew, V. Bagal, R. P. Tito, D. Karatzas, E. Valveny, and C. V. Jawahar (2021a)
  InfographicVQA.
  External Links: 2104.12756,
  [Link](https://arxiv.org/abs/2104.12756)
  Cited by: [Table 1](#S3.T1.3.1.5.4.1 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* M. Mathew, D. Karatzas, and C. V. Jawahar (2021b)
  DocVQA: a dataset for vqa on document images.
  External Links: 2007.00398,
  [Link](https://arxiv.org/abs/2007.00398)
  Cited by: [Table 1](#S3.T1.3.1.9.8.2 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* A. Mishra, S. Shekhar, A. K. Singh, and A. Chakraborty (2019)
  OCR-vqa: visual question answering by reading text in images.
  In ICDAR,
  Cited by: [Table 1](#S3.T1.3.1.7.6.1 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* moondream (2025)
  RefCOCO-M: Refined Referring Expression Segmentation.
  Note: Hugging Face DatasetsDataset repo; uploaded Nov 17, 2025; accessed 2025-12-24.
  External Links: [Link](https://huggingface.co/datasets/moondream/refcoco-m)
  Cited by: [Table 1](#S3.T1.3.1.22.21.2 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* C. OpenCompass (2023)
  OpenCompass: a universal evaluation platform for foundation models.
  Note: <https://github.com/open-compass/opencompass>
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* R. Paiss, A. Ephrat, O. Tov, S. Zada, I. Mosseri, M. Irani, and T. Dekel (2023)
  Teaching clip to count to ten.
  External Links: 2302.12066,
  [Link](https://arxiv.org/abs/2302.12066)
  Cited by: [Table 1](#S3.T1.3.1.24.23.2 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* F. M. Polo, L. Weber, L. Choshen, Y. Sun, G. Xu, and M. Yurochkin (2024)
  TinyBenchmarks: evaluating llms with fewer examples.
  arXiv preprint arXiv:2402.14992.
  Cited by: [§1](#S1.p4.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"),
  [§2](#S2.p4.1 "2 Related Work ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* B. Recht, R. Roelofs, L. Schmidt, and V. Shankar (2019)
  Do imagenet classifiers generalize to imagenet?.
  External Links: 1902.10811,
  [Link](https://arxiv.org/abs/1902.10811)
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* J. Saad-Falcon, E. K. Buchanan, M. F. Chen, T. Huang, B. McLaughlin, T. Bhathal, S. Zhu, B. Athiwaratkun, F. Sala, S. Linderman, et al. (2025)
  Shrinking the generation-verification gap with weak verifiers.
  arXiv preprint arXiv:2506.18203.
  Cited by: [§3.3](#S3.SS3.SSS0.Px2.p2.1 "Solution: Two-Stage Quality Filtering with VLM-as-Judge ‣ 3.3 Incorrect Ground Truth and Ambiguity ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* T. Sakai (2007)
  On the reliability of information retrieval metrics.
  In SIGIR,
  Cited by: [§3.4](#S3.SS4.SSS0.Px1.p2.2 "Problem: The Computational Burden of Comprehensive Evaluation ‣ 3.4 High Discrimination with Limited Compute ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* T. Schick et al. (2025)
  Fluid language model benchmarking.
  arXiv preprint arXiv:2509.11106.
  Cited by: [§1](#S1.p4.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"),
  [§2](#S2.p1.1 "2 Related Work ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"),
  [§2](#S2.p4.1 "2 Related Work ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"),
  [§3.4](#S3.SS4.SSS0.Px1.p3.1 "Problem: The Computational Burden of Comprehensive Evaluation ‣ 3.4 High Discrimination with Limited Compute ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* A. Singh, V. Natarjan, M. Shah, Y. Jiang, X. Chen, D. Parikh, and M. Rohrbach (2019)
  Towards vqa models that can read.
  In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,
   pp. 8317–8326.
  Cited by: [Table 1](#S3.T1.3.1.10.9.2 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* C. Spearman (1904)
  The proof and measurement of association between two things.
  American Journal of Psychology.
  Cited by: [§3.4](#S3.SS4.SSS0.Px1.p2.2 "Problem: The Computational Burden of Comprehensive Evaluation ‣ 3.4 High Discrimination with Limited Compute ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* A. Srivastava, A. Rastogi, A. Rao, A. A. M. Shoeb, A. Abid, A. Fisch, A. R. Brown, A. Santoro, A. Gupta, A. Garriga-Alonso, A. Kluska, A. Lewkowycz, A. Agarwal, A. Power, A. Ray, A. Warstadt, A. W. Kocurek, A. Safaya, A. Tazarv, A. Xiang, A. Parrish, A. Nie, A. Hussain, A. Askell, A. Dsouza, A. Slone, A. Rahane, A. S. Iyer, A. Andreassen, A. Madotto, A. Santilli, A. Stuhlmüller, A. Dai, A. La, A. Lampinen, A. Zou, A. Jiang, A. Chen, A. Vuong, A. Gupta, A. Gottardi, A. Norelli, A. Venkatesh, A. Gholamidavoodi, A. Tabassum, A. Menezes, A. Kirubarajan, A. Mullokandov, A. Sabharwal, A. Herrick, A. Efrat, A. Erdem, A. Karakaş, B. R. Roberts, B. S. Loe, B. Zoph, B. Bojanowski, B. Özyurt, B. Hedayatnia, B. Neyshabur, B. Inden, B. Stein, B. Ekmekci, B. Y. Lin, B. Howald, B. Orinion, C. Diao, C. Dour, C. Stinson, C. Argueta, C. F. Ramírez, C. Singh, C. Rathkopf, C. Meng, C. Baral, C. Wu, C. Callison-Burch, C. Waites, C. Voigt, C. D. Manning, C. Potts, C. Ramirez, C. E. Rivera, C. Siro, C. Raffel, C. Ashcraft, C. Garbacea, D. Sileo, D. Garrette, D. Hendrycks, D. Kilman, D. Roth, D. Freeman, D. Khashabi, D. Levy, D. M. González, D. Perszyk, D. Hernandez, D. Chen, D. Ippolito, D. Gilboa, D. Dohan, D. Drakard, D. Jurgens, D. Datta, D. Ganguli, D. Emelin, D. Kleyko, D. Yuret, D. Chen, D. Tam, D. Hupkes, D. Misra, D. Buzan, D. C. Mollo, D. Yang, D. Lee, D. Schrader, E. Shutova, E. D. Cubuk, E. Segal, E. Hagerman, E. Barnes, E. Donoway, E. Pavlick, E. Rodola, E. Lam, E. Chu, E. Tang, E. Erdem, E. Chang, E. A. Chi, E. Dyer, E. Jerzak, E. Kim, E. E. Manyasi, E. Zheltonozhskii, F. Xia, F. Siar, F. Martínez-Plumed, F. Happé, F. Chollet, F. Rong, G. Mishra, G. I. Winata, G. de Melo, G. Kruszewski, G. Parascandolo, G. Mariani, G. Wang, G. Jaimovitch-López, G. Betz, G. Gur-Ari, H. Galijasevic, H. Kim, H. Rashkin, H. Hajishirzi, H. Mehta, H. Bogar, H. Shevlin, H. Schütze, H. Yakura, H. Zhang, H. M. Wong, I. Ng, I. Noble, J. Jumelet, J. Geissinger, J. Kernion, J. Hilton, J. Lee, J. F. Fisac, J. B. Simon, J. Koppel, J. Zheng, J. Zou, J. Kocoń, J. Thompson, J. Wingfield, J. Kaplan, J. Radom, J. Sohl-Dickstein, J. Phang, J. Wei, J. Yosinski, J. Novikova, J. Bosscher, J. Marsh, J. Kim, J. Taal, J. Engel, J. Alabi, J. Xu, J. Song, J. Tang, J. Waweru, J. Burden, J. Miller, J. U. Balis, J. Batchelder, J. Berant, J. Frohberg, J. Rozen, J. Hernandez-Orallo, J. Boudeman, J. Guerr, J. Jones, J. B. Tenenbaum, J. S. Rule, J. Chua, K. Kanclerz, K. Livescu, K. Krauth, K. Gopalakrishnan, K. Ignatyeva, K. Markert, K. D. Dhole, K. Gimpel, K. Omondi, K. Mathewson, K. Chiafullo, K. Shkaruta, K. Shridhar, K. McDonell, K. Richardson, L. Reynolds, L. Gao, L. Zhang, L. Dugan, L. Qin, L. Contreras-Ochando, L. Morency, L. Moschella, L. Lam, L. Noble, L. Schmidt, L. He, L. O. Colón, L. Metz, L. K. Şenel, M. Bosma, M. Sap, M. ter Hoeve, M. Farooqi, M. Faruqui, M. Mazeika, M. Baturan, M. Marelli, M. Maru, M. J. R. Quintana, M. Tolkiehn, M. Giulianelli, M. Lewis, M. Potthast, M. L. Leavitt, M. Hagen, M. Schubert, M. O. Baitemirova, M. Arnaud, M. McElrath, M. A. Yee, M. Cohen, M. Gu, M. Ivanitskiy, M. Starritt, M. Strube, M. Swedrowski, M. Bevilacqua, M. Yasunaga, M. Kale, M. Cain, M. Xu, M. Suzgun, M. Walker, M. Tiwari, M. Bansal, M. Aminnaseri, M. Geva, M. Gheini, M. V. T, N. Peng, N. A. Chi, N. Lee, N. G. Krakover, N. Cameron, N. Roberts, N. Doiron, N. Martinez, N. Nangia, N. Deckers, N. Muennighoff, N. S. Keskar, N. S. Iyer, N. Constant, N. Fiedel, N. Wen, O. Zhang, O. Agha, O. Elbaghdadi, O. Levy, O. Evans, P. A. M. Casares, P. Doshi, P. Fung, P. P. Liang, P. Vicol, P. Alipoormolabashi, P. Liao, P. Liang, P. Chang, P. Eckersley, P. M. Htut, P. Hwang, P. Miłkowski, P. Patil, P. Pezeshkpour, P. Oli, Q. Mei, Q. Lyu, Q. Chen, R. Banjade, R. E. Rudolph, R. Gabriel, R. Habacker, R. Risco, R. Millière, R. Garg, R. Barnes, R. A. Saurous, R. Arakawa, R. Raymaekers, R. Frank, R. Sikand, R. Novak, R. Sitelew, R. LeBras, R. Liu, R. Jacobs, R. Zhang, R. Salakhutdinov, R. Chi, R. Lee, R. Stovall, R. Teehan, R. Yang, S. Singh, S. M. Mohammad, S. Anand, S. Dillavou, S. Shleifer, S. Wiseman, S. Gruetter, S. R. Bowman, S. S. Schoenholz, S. Han, S. Kwatra, S. A. Rous, S. Ghazarian, S. Ghosh, S. Casey, S. Bischoff, S. Gehrmann, S. Schuster, S. Sadeghi, S. Hamdan, S. Zhou, S. Srivastava, S. Shi, S. Singh, S. Asaadi, S. S. Gu, S. Pachchigar, S. Toshniwal, S. Upadhyay, Shyamolima, Debnath, S. Shakeri, S. Thormeyer, S. Melzi, S. Reddy, S. P. Makini, S. Lee, S. Torene, S. Hatwar, S. Dehaene, S. Divic, S. Ermon, S. Biderman, S. Lin, S. Prasad, S. T. Piantadosi, S. M. Shieber, S. Misherghi, S. Kiritchenko, S. Mishra, T. Linzen, T. Schuster, T. Li, T. Yu, T. Ali, T. Hashimoto, T. Wu, T. Desbordes, T. Rothschild, T. Phan, T. Wang, T. Nkinyili, T. Schick, T. Kornev, T. Tunduny, T. Gerstenberg, T. Chang, T. Neeraj, T. Khot, T. Shultz, U. Shaham, V. Misra, V. Demberg, V. Nyamai, V. Raunak, V. Ramasesh, V. U. Prabhu, V. Padmakumar, V. Srikumar, W. Fedus, W. Saunders, W. Zhang, W. Vossen, X. Ren, X. Tong, X. Zhao, X. Wu, X. Shen, Y. Yaghoobzadeh, Y. Lakretz, Y. Song, Y. Bahri, Y. Choi, Y. Yang, Y. Hao, Y. Chen, Y. Belinkov, Y. Hou, Y. Hou, Y. Bai, Z. Seid, Z. Zhao, Z. Wang, Z. J. Wang, Z. Wang, and Z. Wu (2023)
  Beyond the imitation game: quantifying and extrapolating the capabilities of language models.
  External Links: 2206.04615,
  [Link](https://arxiv.org/abs/2206.04615)
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* E. Strubell, A. Ganesh, and A. McCallum (2019)
  Energy and policy considerations for deep learning in nlp.
  External Links: 1906.02243,
  [Link](https://arxiv.org/abs/1906.02243)
  Cited by: [§1](#S1.p4.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* J. Su, J. Healey, P. Nakov, and C. Cardie (2025)
  Between underthinking and overthinking: an empirical study of reasoning length and correctness in llms.
  External Links: 2505.00127,
  [Link](https://arxiv.org/abs/2505.00127)
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.8 "Takeaway 3: The ”Overthinking” Penalty: Inference-Time Scaling Degrades Perception at High Cost. ‣ 5 Diagnosing VLM Pathologies with DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* V. Venktesh, M. Rathee, and A. Anand (2025)
  Trust but verify! a survey on verification design for test-time scaling.
  External Links: 2508.16665,
  [Link](https://arxiv.org/abs/2508.16665)
  Cited by: [§3.3](#S3.SS3.SSS0.Px2.p2.1 "Solution: Two-Stage Quality Filtering with VLM-as-Judge ‣ 3.3 Incorrect Ground Truth and Ambiguity ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* R. Vivek, K. Ethayarajh, D. Yang, and D. Kiela (2024)
  Anchor points: benchmarking models with much fewer examples.
  In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers),
   pp. 1576–1601.
  Cited by: [§2](#S2.p5.1 "2 Related Work ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* E. M. Voorhees (2001)
  Evaluation by highly relevant documents.
  In SIGIR,
  Cited by: [§3.4](#S3.SS4.SSS0.Px1.p2.2 "Problem: The Computational Burden of Comprehensive Evaluation ‣ 3.4 High Discrimination with Limited Compute ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* J. Wang, Y. Ming, Z. Shi, V. Vineet, X. Wang, S. Li, and N. Joshi (2024a)
  Is a picture worth a thousand words? delving into spatial reasoning for vision language models.
  Advances in Neural Information Processing Systems 37,  pp. 75392–75421.
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* K. Wang, J. Pan, W. Shi, Z. Lu, H. Ren, A. Zhou, M. Zhan, and H. Li (2024b)
  Measuring multimodal mathematical reasoning with math-vision dataset.
  In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track,
  External Links: [Link](https://openreview.net/forum?id=QWTCcxMpPA)
  Cited by: [Table 1](#S3.T1.3.1.15.14.2 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* Y. Wang, Q. Liu, J. Xu, T. Liang, X. Chen, Z. He, L. Song, D. Yu, J. Li, Z. Zhang, R. Wang, Z. Tu, H. Mi, and D. Yu (2025)
  Thoughts are all over the place: on the underthinking of o1-like llms.
  External Links: 2501.18585,
  [Link](https://arxiv.org/abs/2501.18585)
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.8 "Takeaway 3: The ”Overthinking” Penalty: Inference-Time Scaling Degrades Perception at High Cost. ‣ 5 Diagnosing VLM Pathologies with DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* Z. Wang, M. Xia, L. He, H. Chen, Y. Liu, R. Zhu, K. Liang, X. Wu, H. Liu, S. Malladi, A. Chevalier, S. Arora, and D. Chen (2024c)
  CharXiv: charting gaps in realistic chart understanding in multimodal llms.
  External Links: 2406.18521,
  [Link](https://arxiv.org/abs/2406.18521)
  Cited by: [Table 1](#S3.T1.3.1.4.3.1 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* J. Wei, N. Kim, Y. Tay, and Q. V. Le (2023)
  Inverse scaling can become u-shaped.
  External Links: 2211.02011,
  [Link](https://arxiv.org/abs/2211.02011)
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* J. Wei, Y. Tay, R. Bommasani, C. Raffel, B. Zoph, S. Borgeaud, D. Yogatama, M. Bosma, D. Zhou, D. Metzler, E. H. Chi, T. Hashimoto, O. Vinyals, P. Liang, J. Dean, and W. Fedus (2022)
  Emergent abilities of large language models.
  External Links: 2206.07682,
  [Link](https://arxiv.org/abs/2206.07682)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* Y. Wu, Y. Wang, Z. Ye, T. Du, S. Jegelka, and Y. Wang (2025)
  When more is less: understanding chain-of-thought length in llms.
  External Links: 2502.07266,
  [Link](https://arxiv.org/abs/2502.07266)
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.8 "Takeaway 3: The ”Overthinking” Penalty: Inference-Time Scaling Degrades Perception at High Cost. ‣ 5 Diagnosing VLM Pathologies with DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* xAI (2024)
  RealWorldQA: a benchmark for real-world visual understanding.
  Note: <https://huggingface.co/datasets/xai-org/RealworldQA>Released with Grok-1.5 Vision Preview
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"),
  [Table 1](#S3.T1.3.1.17.16.2 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* Y. Xiao, E. Sun, T. Liu, and W. Wang (2024)
  LogicVista: multimodal llm logical reasoning benchmark in visual contexts.
  External Links: 2407.04973,
  [Link](https://arxiv.org/abs/2407.04973)
  Cited by: [Table 1](#S3.T1.3.1.16.15.2 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, C. Zheng, D. Liu, F. Zhou, F. Huang, F. Hu, H. Ge, H. Wei, H. Lin, J. Tang, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Zhou, J. Lin, K. Dang, K. Bao, K. Yang, L. Yu, L. Deng, M. Li, M. Xue, M. Li, P. Zhang, P. Wang, Q. Zhu, R. Men, R. Gao, S. Liu, S. Luo, T. Li, T. Tang, W. Yin, X. Ren, X. Wang, X. Zhang, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Zhang, Y. Wan, Y. Liu, Z. Wang, Z. Cui, Z. Zhang, Z. Zhou, and Z. Qiu (2025)
  Qwen3 technical report.
  External Links: 2505.09388,
  [Link](https://arxiv.org/abs/2505.09388)
  Cited by: [§3.1](#S3.SS1.SSS0.Px4.p1.1 "Solution: MCQ-to-Generative Transformation and Circular MCQ Evaluation ‣ 3.1 MCQ Evaluations: High Noise, Low Fidelity ‣ 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* Z. Yang, J. Tang, Z. Li, P. Wang, J. Wan, H. Zhong, X. Liu, M. Yang, P. Wang, S. Bai, L. Jin, and J. Lin (2024)
  CC-ocr: a comprehensive and challenging ocr benchmark for evaluating large multimodal models in literacy.
  External Links: 2412.02210,
  [Link](https://arxiv.org/abs/2412.02210)
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"),
  [Table 1](#S3.T1.3.1.12.11.1 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"),
  [Table 1](#S3.T1.3.1.6.5.2 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* X. Yue, T. Zheng, Y. Ni, Y. Wang, K. Zhang, S. Tong, Y. Sun, B. Yu, G. Zhang, H. Sun, Y. Su, W. Chen, and G. Neubig (2025)
  MMMU-pro: a more robust multi-discipline multimodal understanding benchmark.
  External Links: 2409.02813,
  [Link](https://arxiv.org/abs/2409.02813)
  Cited by: [Table 1](#S3.T1.3.1.28.27.2 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* R. Zhang, D. Jiang, Y. Zhang, H. Lin, Z. Guo, P. Qiu, A. Zhou, P. Lu, K. Chang, P. Gao, and H. Li (2024a)
  MathVerse: does your multi-modal llm truly see the diagrams in visual math problems?.
  External Links: 2403.14624,
  [Link](https://arxiv.org/abs/2403.14624)
  Cited by: [Table 1](#S3.T1.3.1.14.13.1 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* Y. Zhang, H. Zhang, H. Tian, C. Fu, S. Zhang, J. Wu, F. Li, K. Wang, Q. Wen, Z. Zhang, et al. (2024b)
  MME-realworld: could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans?.
  arXiv preprint arXiv:2408.13257.
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"),
  [§1](#S1.p9.1 "1 Introduction ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"),
  [Table 1](#S3.T1.3.1.11.10.1 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"),
  [Table 1](#S3.T1.3.1.18.17.1 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations"),
  [Table 1](#S3.T1.3.1.27.26.1 "In 3 The Making of DatBench ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").
* Y. Zhang, Y. Shi, W. Yu, Q. Wen, X. Wang, W. Yang, Z. Zhang, L. Wang, and R. Jin (2025)
  Debiasing multimodal large language models via penalization of language priors.
  External Links: 2403.05262,
  [Link](https://arxiv.org/abs/2403.05262)
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").

## Appendix A Main Results

Table 3:  Comprehensive Evaluation. Comparison across DatBench (DB), DatBench-Full (Full), and the Original datasets (Orig). Values are percentages.

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Chart | | | Doc | | | OCR | | | Grd | | | Cnt | | | Spa | | | Mth | | | Dia | | | Gen | | |
| Model | DB | Full | Orig | DB | Full | Orig | DB | Full | Orig | DB | Full | Orig | DB | Full | Orig | DB | Full | Orig | DB | Full | Orig | DB | Full | Orig | DB | Full | Orig |
| Qwen2.5-VL | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| Qwen2.5-VL-3B-Instruct | 54.2 | 56.2 | 58.4 | 71.1 | 78.5 | 79.3 | 75.1 | 70.2 | 73.5 | 0.3 | 0.2 | 0.4 | 90.2 | 71.8 | 77.0 | 27.5 | 27.5 | 21.5 | 13.8 | 14.9 | 30.4 | 54.4 | 51.1 | 49.1 | 53.2 | 59.8 | 78.6 |
| Qwen2.5-VL-7B-Instruct | 64.1 | 63.8 | 63.3 | 62.8 | 66.3 | 65.6 | 82.6 | 75.4 | 78.3 | 0.2 | 0.3 | 0.4 | 92.4 | 76.5 | 80.4 | 20.8 | 20.8 | 16.7 | 27.9 | 26.1 | 41.3 | 61.9 | 56.2 | 54.7 | 56.8 | 62.1 | 70.3 |
| Qwen3-VL | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| Qwen3-VL-2B-Instruct | 53.3 | 54.8 | 55.2 | 62.7 | 62.6 | 65.0 | 72.3 | 67.9 | 70.8 | 76.2 | 75.6 | 77.4 | 90.3 | 70.8 | 76.2 | 20.3 | 20.3 | 16.3 | 19.5 | 17.6 | 28.3 | 24.1 | 23.1 | 25.3 | 42.2 | 51.5 | 72.8 |
| Qwen3-VL-4B-Instruct | 63.8 | 63.5 | 66.7 | 71.0 | 71.6 | 74.1 | 77.9 | 72.3 | 76.1 | 86.2 | 85.7 | 86.4 | 92.2 | 74.6 | 79.1 | 21.8 | 21.8 | 15.8 | 35.2 | 30.7 | 43.6 | 55.4 | 51.6 | 51.9 | 59.9 | 59.4 | 78.8 |
| Qwen3-VL-8B-Instruct | 70.5 | 68.4 | 71.8 | 57.2 | 63.5 | 68.9 | 54.4 | 54.7 | 59.1 | 84.9 | 83.7 | 84.4 | 92.6 | 75.4 | 80.0 | 19.7 | 19.7 | 13.2 | 35.9 | 31.2 | 44.7 | 32.7 | 31.3 | 33.5 | 63.7 | 59.8 | 78.9 |
| Qwen2.5-Omni | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| Qwen2.5-Omni-3B | 40.1 | 44.9 | 50.3 | 69.1 | 71.3 | 69.8 | 60.5 | 58.2 | 61.8 | 0.4 | 0.4 | 0.5 | 83.0 | 64.0 | 70.8 | 12.0 | 12.0 | 11.4 | 8.5 | 10.8 | 26.5 | 34.6 | 31.0 | 32.7 | 30.1 | 45.8 | 67.5 |
| Qwen2.5-Omni-7B | 46.3 | 50.1 | 54.7 | 70.1 | 74.7 | 70.7 | 76.8 | 70.6 | 73.7 | 0.5 | 0.4 | 0.5 | 89.2 | 69.8 | 75.5 | 21.2 | 21.2 | 17.0 | 16.3 | 17.5 | 32.3 | 52.8 | 48.1 | 48.7 | 41.7 | 49.7 | 68.8 |
| InternVL2 | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| InternVL2-2B | 25.0 | 32.5 | 38.3 | 12.4 | 39.2 | 48.4 | 29.1 | 35.8 | 44.1 | 33.2 | 32.9 | 37.1 | 75.0 | 64.8 | 62.9 | 11.5 | 11.5 | 12.0 | 7.0 | 9.8 | 18.8 | 16.0 | 17.2 | 20.0 | 12.7 | 41.7 | 63.5 |
| InternVL2-4B | 42.8 | 47.2 | 48.5 | 18.0 | 39.1 | 49.8 | 35.8 | 40.3 | 49.0 | 63.4 | 62.6 | 66.1 | 75.2 | 65.3 | 68.6 | 13.9 | 13.9 | 11.9 | 9.4 | 11.5 | 23.7 | 23.2 | 23.0 | 26.8 | 21.3 | 45.3 | 67.2 |
| InternVL2-8B | 46.4 | 50.4 | 51.3 | 18.3 | 41.6 | 51.8 | 46.0 | 47.7 | 56.5 | 73.9 | 73.0 | 75.1 | 87.0 | 68.6 | 71.3 | 19.4 | 19.4 | 14.6 | 11.9 | 14.4 | 27.5 | 26.3 | 26.3 | 30.4 | 34.9 | 50.9 | 71.4 |
| InternVL2.5 | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| InternVL2\_5-2B | 34.9 | 40.7 | 46.5 | 15.9 | 42.3 | 50.5 | 38.3 | 43.0 | 50.1 | 40.8 | 39.9 | 41.1 | 89.3 | 71.4 | 75.1 | 15.1 | 15.1 | 15.5 | 9.9 | 11.8 | 24.0 | 19.0 | 20.5 | 21.9 | 27.9 | 47.9 | 69.1 |
| InternVL2\_5-4B | 50.0 | 52.8 | 56.6 | 27.2 | 51.0 | 59.5 | 52.4 | 52.1 | 59.4 | 62.7 | 60.7 | 62.9 | 90.3 | 72.6 | 77.9 | 25.4 | 25.4 | 20.5 | 12.0 | 13.9 | 31.6 | 39.8 | 37.9 | 39.6 | 37.2 | 49.7 | 73.1 |
| InternVL2\_5-8B | 48.2 | 51.5 | 56.2 | 41.4 | 57.9 | 62.4 | 49.0 | 51.5 | 59.2 | 70.9 | 69.2 | 71.3 | 90.3 | 72.7 | 73.8 | 21.8 | 21.8 | 16.9 | 12.7 | 14.6 | 28.6 | 23.8 | 23.1 | 27.8 | 44.3 | 55.4 | 74.7 |
| InternVL3 | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| InternVL3-2B-Instruct | 45.2 | 48.9 | 49.0 | 25.7 | 50.2 | 56.5 | 54.5 | 55.3 | 61.1 | 34.4 | 33.4 | 35.5 | 89.5 | 71.9 | 77.1 | 22.0 | 22.0 | 17.3 | 12.1 | 13.7 | 27.9 | 24.0 | 24.0 | 26.3 | 42.5 | 54.3 | 75.0 |
| InternVL3-9B-Instruct | 54.8 | 56.1 | 61.3 | 32.3 | 56.4 | 55.5 | 66.0 | 63.8 | 69.8 | 70.9 | 69.3 | 72.6 | 91.4 | 74.5 | 78.5 | 32.9 | 32.9 | 22.6 | 19.0 | 19.5 | 34.4 | 40.3 | 39.6 | 39.1 | 47.4 | 54.6 | 75.3 |
| InternVL3.5 | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| InternVL3\_5-2B-Instruct | 48.7 | 51.4 | 52.2 | 24.5 | 46.7 | 54.4 | 46.2 | 48.6 | 55.6 | 41.0 | 40.3 | 41.9 | 85.5 | 67.6 | 73.7 | 13.7 | 13.7 | 13.6 | 16.6 | 17.0 | 32.0 | 19.4 | 19.2 | 23.3 | 27.0 | 46.2 | 68.7 |
| InternVL3\_5-4B-Instruct | 58.8 | 59.5 | 59.8 | 29.5 | 51.3 | 57.2 | 51.3 | 51.3 | 59.1 | 66.6 | 65.5 | 67.0 | 90.6 | 71.3 | 76.8 | 21.4 | 21.4 | 16.7 | 19.7 | 19.7 | 36.3 | 39.2 | 36.6 | 39.0 | 39.3 | 49.8 | 71.7 |
| InternVL3\_5-8B-Instruct | 60.4 | 61.1 | 64.9 | 37.0 | 56.2 | 60.4 | 53.3 | 54.2 | 62.2 | 60.8 | 59.1 | 60.8 | 91.0 | 72.5 | 77.8 | 24.1 | 24.1 | 18.7 | 19.3 | 19.9 | 37.6 | 44.5 | 41.1 | 43.3 | 49.3 | 54.6 | 74.8 |
| Other Models | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| GLM-4.1V-9B-Base | 66.3 | 65.5 | 67.2 | 29.5 | 48.6 | 60.2 | 73.6 | 69.0 | 72.3 | 85.2 | 83.3 | 83.7 | 92.4 | 75.6 | 80.0 | 36.8 | 36.8 | 25.9 | 17.4 | 16.5 | 31.1 | 66.4 | 60.0 | 59.4 | 54.1 | 56.6 | 76.3 |
| SmolVLM2-2.2B-Instruct | 31.5 | 37.1 | 34.8 | – | – | – | 24.2 | 30.3 | 39.4 | 0.0 | 0.0 | 0.1 | 71.8 | 65.2 | 70.6 | 6.6 | 6.6 | 8.2 | 8.8 | 11.6 | 19.7 | 9.2 | 7.5 | 11.0 | 11.6 | 41.8 | 65.9 |
| Phi-3.5-vision-instruct | 34.6 | 40.3 | 42.6 | 60.7 | 68.3 | 69.2 | 27.3 | 35.0 | 46.2 | 2.2 | 1.9 | 2.3 | 77.2 | 65.3 | 69.9 | 15.9 | 15.9 | 16.4 | 8.2 | 10.8 | 20.7 | 13.9 | 12.7 | 17.6 | 38.0 | 50.2 | 73.0 |
| Gemma-3-4B-it | 24.9 | 31.7 | 34.3 | 9.2 | 37.1 | 46.2 | 37.7 | 43.3 | 51.1 | 3.4 | 3.4 | 5.9 | 5.8 | 51.5 | 50.2 | 8.1 | 8.1 | 7.6 | 15.0 | 15.2 | 27.8 | 12.1 | 11.4 | 16.8 | 17.1 | 41.6 | 53.1 |
| Thinking Models | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| GLM-4.1V-9B-Thinking | 76.0 | 73.8 | 72.9 | 26.5 | 44.3 | 56.7 | 59.0 | 55.2 | 61.6 | 87.9 | 86.3 | 87.2 | 92.3 | 76.7 | 80.5 | 19.4 | 19.4 | 15.1 | 32.4 | 28.5 | 39.5 | 65.2 | 60.2 | 59.5 | 54.1 | 54.0 | 75.4 |
| R-4B | 66.2 | 65.4 | 66.6 | 54.6 | 60.9 | 65.6 | 43.0 | 47.1 | 56.7 | 83.5 | 81.7 | 83.1 | 92.7 | 74.7 | 78.5 | 11.4 | 11.4 | 7.8 | 43.4 | 37.8 | 50.7 | 33.7 | 30.4 | 33.7 | 50.7 | 52.2 | 73.0 |
| Qwen3-VL-2B-Thinking | 62.3 | 61.9 | 61.5 | 23.3 | 45.5 | 53.7 | 21.5 | 22.2 | 31.3 | 84.6 | 83.2 | 84.3 | 91.2 | 72.8 | 76.7 | 9.0 | 9.0 | 5.8 | 25.7 | 22.6 | 31.9 | 15.9 | 13.3 | 18.0 | 51.3 | 55.2 | 75.3 |
| Qwen3-VL-4B-Thinking | 68.9 | 66.9 | 68.6 | 24.7 | 46.9 | 56.5 | 22.1 | 22.7 | 31.4 | 87.3 | 85.8 | 86.4 | 92.3 | 74.4 | 79.1 | 9.1 | 9.1 | 6.1 | 38.2 | 33.0 | 42.0 | 24.4 | 21.0 | 25.4 | 59.8 | 57.4 | 77.3 |
| Qwen3-VL-8B-Thinking | 73.2 | 70.7 | 72.3 | 27.0 | 48.6 | 58.1 | 26.0 | 25.6 | 34.5 | 88.8 | 87.1 | 87.6 | 92.5 | 75.6 | 79.6 | 10.2 | 10.2 | 6.7 | 43.3 | 37.4 | 47.5 | 28.6 | 25.7 | 29.4 | 62.4 | 58.2 | 77.6 |

## Appendix B Benchmark Computational Cost

| Model | Chart | Count | Doc | Gen | Ground | Math | Scene | Spat | Table |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Samples | 12,249 | 39,080 | 118,581 | 246,475 | 36,961 | 12,368 | 38,950 | 40,131 | 32,753 |
| SmolVLM2-2.2B | 0.35 | 1.11 | — | 9.02 | 0.74 | 0.71 | 1.05 | 0.60 | 0.38 |
| InternVL2-2B | 0.30 | 0.77 | 2.36 | 5.97 | 0.59 | 0.71 | 0.88 | 0.56 | 0.35 |
| InternVL2-4B | 0.34 | 0.78 | 3.13 | 6.28 | 0.79 | 1.28 | 0.89 | 0.50 | 0.31 |
| InternVL2-8B | 0.38 | 0.95 | 3.51 | 8.03 | 0.84 | 1.58 | 1.10 | 0.58 | 0.37 |
| InternVL2.5-2B | 0.28 | 0.70 | 2.13 | 6.38 | 0.64 | 0.70 | 0.82 | 0.55 | 0.34 |
| InternVL2.5-4B | 0.29 | 0.70 | 2.45 | 5.90 | 0.71 | 0.92 | 0.84 | 0.51 | 0.34 |
| InternVL2.5-8B | 0.39 | 0.95 | 3.71 | 8.07 | 0.84 | 1.28 | 1.04 | 0.60 | 0.38 |
| InternVL3-2B | 0.34 | 0.63 | 2.05 | 5.55 | 0.77 | 0.69 | 0.80 | 0.54 | 0.34 |
| InternVL3-9B | 0.46 | 1.00 | 4.02 | 8.88 | 1.00 | 1.61 | 1.13 | 0.61 | 0.41 |
| InternVL3.5-2B | 0.45 | 0.68 | 2.47 | 6.66 | 0.67 | 1.40 | 0.91 | 0.66 | 0.36 |
| InternVL3.5-4B | 0.60 | 0.77 | 3.12 | 8.01 | 0.84 | 1.31 | 0.99 | 0.56 | 0.38 |
| InternVL3.5-8B | 0.80 | 0.88 | 3.89 | 12.86 | 0.98 | 2.62 | 1.03 | 0.68 | 0.40 |
| Qwen2.5-Omni-3B | 0.29 | 0.56 | 2.35 | 4.68 | 0.64 | 0.51 | 1.18 | 0.74 | 0.46 |
| Qwen2.5-Omni-7B | 0.35 | 0.55 | 3.06 | 6.15 | 0.77 | 1.11 | 1.26 | 0.80 | 0.46 |
| Qwen2.5-VL-3B | 0.29 | 0.71 | 2.52 | 4.65 | 0.63 | 1.19 | 1.19 | 0.70 | 0.43 |
| Qwen2.5-VL-7B | 0.56 | 0.76 | 3.42 | 6.04 | 0.70 | 2.42 | 1.23 | 0.81 | 0.48 |
| Qwen3-VL-2B | 0.98 | 0.72 | 2.88 | 8.65 | 0.58 | 3.62 | 1.29 | 0.71 | 2.89 |
| Qwen3-VL-2B-T | 5.63 | 4.32 | 18.94 | 34.78 | 4.25 | 10.37 | 5.55 | 9.86 | 13.88 |
| Qwen3-VL-4B | 0.89 | 0.74 | 3.55 | 9.32 | 0.68 | 4.89 | 1.62 | 3.65 | 1.83 |
| Qwen3-VL-4B-T | 7.71 | 8.75 | 40.79 | 57.55 | 6.81 | 14.35 | 8.49 | 16.86 | 13.62 |
| Qwen3-VL-8B | 1.39 | 0.76 | 4.82 | 12.74 | 1.00 | 6.89 | 3.28 | 4.49 | 3.81 |
| Qwen3-VL-8B-T | 7.91 | 6.41 | 27.30 | 57.59 | 7.90 | 17.53 | 8.98 | 19.43 | 14.29 |
| R-4B | 2.04 | 4.70 | 8.07 | 32.70 | 4.16 | 5.24 | 2.18 | 4.67 | 3.47 |
| gemma-3-4b-it | 0.34 | 0.49 | 2.42 | 3.94 | 0.59 | 2.75 | 0.64 | 0.36 | 0.27 |
| Phi-3.5-vision-instruct | 0.17 | 0.46 | 2.19 | 3.60 | 0.61 | 0.50 | 0.79 | 0.31 | 0.22 |
| GLM-4.1V-9B | 1.22 | 0.40 | 4.08 | 11.41 | 0.58 | 4.79 | 1.34 | 0.98 | 0.57 |
| GLM-4.1V-9B-T | 4.78 | 9.13 | 18.36 | 48.30 | 4.10 | 9.61 | 7.51 | 12.27 | 8.22 |

Table 4: H100 hours per model and capability. Sample counts are shown in the second row (italic). Values represent total H100 hours required to process all samples for each capability. Missing entries (—) indicate the capability was not evaluated for that model.

## Appendix C Considerations converting from MCQ to Generative

### C.1 Qualitative Example

We provide an example of converting an eval sample from AI2D from MCQ to generative in Fig. [13](#A3.F13 "Figure 13 ‣ C.1 Qualitative Example ‣ Appendix C Considerations converting from MCQ to Generative ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")

![Refer to caption](/html/2601.02316/assets/figures/ai2dmcqgen.png)


Figure 13: Example of converting a sample from AI2D from its native MCQ format to the generative setting

### C.2 Evals that we had to keep as MCQ

* •

  LogicVista - questions are mensa style puzzles where options are the next image in the sequence so cannot be generated de-novo
* •

  MME-Realworld and MMBench - We keep these in their original MCQ format because many questions are underspecified as free-form prompts: the answer choices provide crucial context about what kind of response is expected. Converting these items to generative QA would not give models sufficient signal about the task definition (see Figure [14](#A3.F14 "Figure 14 ‣ C.2 Evals that we had to keep as MCQ ‣ Appendix C Considerations converting from MCQ to Generative ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations")). Further, many counting questions involve scenes with a very large number of objects, where an exact count is both ambiguous and brittle to minor visual uncertainty. In these cases, the benchmark is primarily probing whether the model can distinguish coarse scales (e.g., few vs many, tens vs hundreds) rather than recover a precise integer, so enforcing an exact-match generative answer would add noise and misrepresent the intended capability being measured.

![Refer to caption](/html/2601.02316/assets/figures/MME_realworld_example.png)


((a)) Samples from MME-Realworld

![Refer to caption](/html/2601.02316/assets/figures/MMBench_example.png)


((b)) Samples from MMBench

Figure 14: Dependence on options for solving various tasks in MME-Realworld and MMbench

## Appendix D VLM-as-Judge Filtering Results

Table [5](#A4.T5 "Table 5 ‣ Appendix D VLM-as-Judge Filtering Results ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations") reports filtering statistics for examples that all evaluated models answered incorrectly. Examples flagged by the VLM judge as ambiguous, incorrectly labeled, or requiring higher resolution are removed from the benchmark. Retained examples represent “frontier” cases where current models uniformly fail on valid, high-quality data, indicating the benchmark has not yet saturated.

Table 5: Two-Stage Quality Filtering Statistics

| Dataset | Total Samples | Flagged | Removed by Judge | | | |
| --- | --- | --- | --- | --- | --- | --- |
| Ambiguous | Incorrect | Low Resolution | Total (%) |
| AI2D | 3,088 | 342 | 111 | 72 | 8 | 131 (4.2%) |
| CC-OCR (kie) | 2,008 | 1 | 0 | 0 | 0 | 0 (0.0%) |
| CC-OCR (Multi Scene OCR) | 2,750 | 4 | 4 | 2 | 4 | 4 (0.1%) |
| ChartQA | 2,500 | 57 | 19 | 25 | 1 | 36 (1.4%) |
| ChartQAPro | 1,948 | 693 | 234 | 190 | 56 | 335 (17.2%) |
| CharXiv (DQ) | 4,000 | 37 | 2 | 15 | 3 | 17 (0.4%) |
| CharXiv (RQ) | 1,000 | 69 | 10 | 17 | 4 | 23 (2.3%) |
| CountBench | 491 | 4 | 3 | 4 | 2 | 4 (0.8%) |
| DocVQA | 5,349 | 6 | 1 | 1 | 1 | 2 (0.0%) |
| InfoVQA | 2,801 | 37 | 7 | 4 | 6 | 12 (0.4%) |
| LogicVista | 448 | 8 | 2 | 1 | 0 | 2 (0.4%) |
| MathVerse (reasoning) | 3,940 | 608 | 141 | 185 | 50 | 234 (5.9%) |
| MathVerse (wo) | 3,940 | 580 | 130 | 169 | 36 | 221 (5.6%) |
| MathVision | 3,040 | 512 | 133 | 153 | 65 | 220 (7.2%) |
| MathVista | 1,000 | 59 | 41 | 38 | 4 | 48 (4.8%) |
| MME-RealWorld (Autonomous Driving) | 5,004 | 2560 | 2076 | 1723 | 759 | 2198 (43.9%) |
| MME-RealWorld (Diagram / Table) | 5,933 | 577 | 77 | 161 | 157 | 307 (5.2%) |
| MME-RealWorld (Video Monitoring) | 2,694 | 1555 | 1197 | 898 | 1015 | 1342 (49.8%) |
| MME-RealWorld (OCR-in-the-Wild) | 6,240 | 478 | 141 | 229 | 101 | 311 (5.0%) |
| MMBench | 4,329 | 152 | 82 | 75 | 0 | 107 (2.5%) |
| MMMU-Pro | 1,730 | 855 | 331 | 256 | 74 | 420 (24.3%) |
| OCR-VQA | 100,424 | 6068 | 3506 | 5245 | 547 | 5531 (5.5%) |
| OCRBench\_v​2v2 | 10,000 | 1360 | 340 | 302 | 146 | 533 (5.3%) |
| Pixmo-Pointing | 394 | 80 | 29 | 24 | 0 | 36 (9.1%) |
| RealWorldQA | 764 | 38 | 12 | 18 | 1 | 20 (2.6%) |
| Ref-COCO-M | 5,598 | 35 | 19 | 8 | 0 | 24 (0.4%) |
| RefCOCO+ (testA) | 5,726 | 65 | 34 | 21 | 0 | 45 (0.8%) |
| RefCOCO+ (testB) | 4,889 | 120 | 60 | 33 | 1 | 76 (1.6%) |
| RefCOCO (testA) | 5,657 | 45 | 24 | 16 | 0 | 32 (0.6%) |
| RefCOCO (testB) | 5,095 | 77 | 38 | 27 | 0 | 52 (1.0%) |
| RefCOCO-G | 9,602 | 137 | 38 | 42 | 1 | 63 (0.7%) |
| TallyQA | 38,589 | 798 | 230 | 416 | 103 | 487 (1.3%) |
| TextVQA | 5,000 | 94 | 40 | 31 | 20 | 56 (1.1%) |
| VQA-V2 | 214,354 | 2297 | 1433 | 447 | 373 | 1585 (0.7%) |

VLM-as-Judge Verification Prompt

```
You are an expert Quality Assurance verifier for a Vision-Language Benchmark.

Task: You will be shown an Image, a Question, and the dataset’s Ground Truth Answer.
Your job is to verify if the Ground Truth Answer is strictly correct and unambiguous
based on the image.

Context: None of the current state-of-the-art models were able to answer this
question correctly. We need to know if this is because the task is hard (valid
Frontier Example) or because the Ground Truth is flawed (Invalid).

Criteria for marking as ’Invalid’ (ground_truth_wrong=true):
1. The image is missing, corrupted, or unreadable.
2. The question refers to details not present in the image.
3. The Ground Truth Answer is factually incorrect based on the image.
4. The question is ambiguous and has multiple valid answers, but the Ground Truth
   only accepts one specific phrasing.

Output Format: You must return ONLY a JSON object with three boolean fields and
one concise rationale field:
{
  "needs_high_resolution": true|false,
  "ground_truth_wrong": true|false,
  "question_is_ambiguous": true|false,
  "reason": "<one short sentence explaining your decision>"
}

Conservative Strategy: When in doubt about whether the ground truth is correct,
prefer marking it as invalid (ground_truth_wrong=true). We prefer False Positives
(discarding a valid hard example) over False Negatives (keeping a bad example).

Is the Ground Truth valid?
```

Table 6: Quality Filtering Statistics Aggregated by Capability

| Capability | Total Samples | Samples Removed | Discarded (%) |
| --- | --- | --- | --- |
| Spatial | 8,462 | 3,560 | 42.07% |
| Math / Logic | 12,368 | 725 | 5.86% |
| Document | 107,781 | 5,533 | 5.13% |
| Table | 9,021 | 438 | 4.86% |
| Chart | 12,249 | 423 | 3.45% |
| Scene OCR | 13,990 | 371 | 2.65% |
| Counting | 39,080 | 491 | 1.26% |
| General | 220,413 | 2,112 | 0.96% |
| Grounding | 36,961 | 328 | 0.89% |
| Total | 460,325 | 13,981 | 3.04% |

## Appendix E Item-Discrimination Subset Selection

![Refer to caption](/html/2601.02316/assets/figures/discrimination_by_capability_k5000_light.png)


Figure 15: Discriminative power of DatBench compared with Random subsets (kk=5000)



![Refer to caption](/html/2601.02316/assets/figures/discriminative_power_plots/discrimination_curve_spatial_light.png)


((a)) Spatial

![Refer to caption](/html/2601.02316/assets/figures/discriminative_power_plots/discrimination_curve_math_light.png)


((b)) Math

![Refer to caption](/html/2601.02316/assets/figures/discriminative_power_plots/discrimination_curve_document_light.png)


((c)) Document

![Refer to caption](/html/2601.02316/assets/figures/discriminative_power_plots/discrimination_curve_table_light.png)


((d)) Table

![Refer to caption](/html/2601.02316/assets/figures/discriminative_power_plots/discrimination_curve_chart_light.png)


((e)) Chart

![Refer to caption](/html/2601.02316/assets/figures/discriminative_power_plots/discrimination_curve_scene_light.png)


((f)) Scene

![Refer to caption](/html/2601.02316/assets/figures/discriminative_power_plots/discrimination_curve_counting_light.png)


((g)) Counting

![Refer to caption](/html/2601.02316/assets/figures/discriminative_power_plots/discrimination_curve_general_light.png)


((h)) General

![Refer to caption](/html/2601.02316/assets/figures/discriminative_power_plots/discrimination_curve_grounding_light.png)


((i)) Grounding

Figure 16: Discriminative power as a function of retained data across all capabilities



![Refer to caption](/html/2601.02316/assets/figures/correlation_plots/correlation_curve_spatial_full_light.png)


((a)) Spatial

![Refer to caption](/html/2601.02316/assets/figures/correlation_plots/correlation_curve_math_full_light.png)


((b)) Math

![Refer to caption](/html/2601.02316/assets/figures/correlation_plots/correlation_curve_document_full_light.png)


((c)) Document

![Refer to caption](/html/2601.02316/assets/figures/correlation_plots/correlation_curve_table_full_light.png)


((d)) Table

![Refer to caption](/html/2601.02316/assets/figures/correlation_plots/correlation_curve_chart_full_light.png)


((e)) Chart

![Refer to caption](/html/2601.02316/assets/figures/correlation_plots/correlation_curve_scene_full_light.png)


((f)) Scene

![Refer to caption](/html/2601.02316/assets/figures/correlation_plots/correlation_curve_counting_full_light.png)


((g)) Counting

![Refer to caption](/html/2601.02316/assets/figures/correlation_plots/correlation_curve_general_full_light.png)


((h)) General

![Refer to caption](/html/2601.02316/assets/figures/correlation_plots/correlation_curve_grounding_full_light.png)


((i)) Grounding

Figure 17: Rank correlation as a function of retained data across all capabilities

## Appendix F Can be solved blind threshold

| Capability | Dataset | Blind Threshold |
| --- | --- | --- |
| Chart | ChartQA; ChartQA Pro; CharXiv; InfoVQA | 1 |
| Counting | CountBench | 4 |
|  | TallyQA | 6 |
| Diagrams / Tables | MME-RW (Diagrams / Tables) | 1 |
|  | AI2D | 5 |
| Document | OCR-VQA; CC-OCR (Document Parsing and KIE); DocVQA; OCRBench\_v2 | 1 |
| Scene OCR | CC-OCR (Multi-Scene OCR) | 5 |
|  | MME-RW (OCR in the wild) | 1 |
|  | TextVQA | 5 |
| Spatial | MME-RW (Autonomous Driving, Remote Sensing) | 1 |
|  | RealWorldQA | 8 |
| Math / Logic | LogicVista; MathVerse | 6 |
|  | MathVision; MathVista (generative) | 1 |
|  | MathVision; MathVista (MCQ) | 6 |
| Grounding | RefCOCO; RefCOCO+; RefCOCO-G; RefCOCO-M; Pixmo-Point | 1 |
| General | VQA-v2 | 1 |
|  | MMBench; MMMU-Pro | 6 |

Table 7: Can-be-solved-blind thresholds for each evaluation dataset. Thresholds indicate the number of models that can correctly answer a question without visual input, above which the question is considered potentially solvable blind.



| Capability | Total Samples | Samples Removed | Fraction Removed (%) |
| --- | --- | --- | --- |
| General | 220,413 | 158,841 | 72.07% |
| Math | 12,367 | 5,908 | 47.77% |
| Chart | 12,248 | 5,873 | 47.95% |
| Scene OCR | 13,990 | 6,054 | 43.27% |
| Counting | 39,079 | 16,263 | 41.62% |
| Document | 118,580 | 47,898 | 40.39% |
| Grounding | 36,960 | 10,221 | 27.65% |
| Spatial | 8,462 | 1,606 | 18.98% |
| Table | 9,021 | 1,482 | 16.43% |
| Total | 471,120 | 254,146 | 53.95% |

Table 8: Blind solvable samples filtering statistics aggregated by capability.

For each evaluation subset, we define a *can-be-solved-blind* threshold, corresponding to the number of models that correctly answer a question without access to the image. Thresholds are chosen based on observed inflection points in blind accuracy curves and known sources of bias such as multiple-choice guessing, answer distribution skew, or lenient scoring functions. Thresholds for all datasets are summarized in Table [7](#A6.T7 "Table 7 ‣ Appendix F Can be solved blind threshold ‣ DatBenchDiscriminative, Faithful, and Efficient VLM Evaluations").

As representative examples, inherently visual tasks such as chart understanding (e.g., ChartQA and related variants) exhibit near-zero blind solvability, with a clear inflection at a single model, motivating a minimal threshold of one. In contrast, multiple-choice evaluations such as RealWorldQA or MMMU-Pro admit non-trivial blind success due to chance-level guessing; in these cases, thresholds are set above random baselines (e.g., exceeding ⌊0.25×N⌋\lfloor 0.25\times N\rfloor models for four-option questions). Similarly, counting benchmarks such as CountBench and TallyQA show systematic biases toward small-number answers, leading to higher blind accuracy despite missing visual input; thresholds are therefore selected at empirical inflection points rather than at one. Finally, for datasets with lenient or continuous scoring metrics (e.g., multi-scene OCR), higher thresholds mitigate false positives arising from partial string matches or overly permissive correctness criteria.

![Refer to caption](/html/2601.02316/assets/figures/histograms/chartqa_no_image_distribution_generative_scoring_function.png)


((a)) ChartQA

![Refer to caption](/html/2601.02316/assets/figures/histograms/tallyqa_no_image_distribution_generative_scoring_function.png)


((b)) TallyQA

![Refer to caption](/html/2601.02316/assets/figures/histograms/ai2d_no_image_distribution_generative_scoring_function.png)


((c)) AI2D

![Refer to caption](/html/2601.02316/assets/figures/histograms/ocr-vqa_no_image_distribution_generative_scoring_function.png)


((d)) OCR-VQA

![Refer to caption](/html/2601.02316/assets/figures/histograms/text-vqa_no_image_distribution_generative_scoring_function.png)


((e)) TextVQA

![Refer to caption](/html/2601.02316/assets/figures/histograms/realworldqa_no_image_distribution_generative_scoring_function.png)


((f)) RealworldQA

Figure 18: Blind solvable thresholds and histograms across datasets (part 1 of 2).



![Refer to caption](/html/2601.02316/assets/figures/histograms/mathvision_no_image_distribution_generative_scoring_function.png)


((a)) Mathvision

![Refer to caption](/html/2601.02316/assets/figures/histograms/refcoco_m_val_no_image_distribution_generative_scoring_function.png)


((b)) RefCOCO-M

![Refer to caption](/html/2601.02316/assets/figures/histograms/vqa-v2_no_image_distribution_generative_scoring_function.png)


((c)) VQAv2

Figure 19: 
Blind solvable thresholds and histograms across datasets (part 2 of 2).

[◄](/html/2601.02315)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2601.02316)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2601.02316)
[View original  
on arXiv](https://arxiv.org/abs/2601.02316)[►](/html/2601.02317)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Thu Feb 5 17:26:35 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
