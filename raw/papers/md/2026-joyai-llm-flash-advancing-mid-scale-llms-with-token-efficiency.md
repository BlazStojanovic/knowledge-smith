---
arxiv: '2604.03044'
authors:
- Aichen Cai
- Anmeng Zhang
- Anyu Li
- Bo Zhang
- Bohua Cai
- Chang Li
- Changjian Jiang
- Changkai Lu
- Chao Xue
- Chaocai Liang
- Cheng Zhang
- Dongkai Liu
- Fei Wang
- Guoqiang Huang
- Haijian Ke
- Han Lin
- Hao Wang
- Ji Miao
- Jiacheng Zhang
- Jialong Shi
- Jifeng Zhu
- Jingjing Qian
- Junhui Luo
- Junwu Xiong
- Lam So
- Liang Huang
- Ming Ke
- Mingyang Li
- Panfeng Shi
- Peng Hao
- Qi Wang
- Qian Lai
- Qiaoqiao Yuan
- Qingyu Yin
- Qiong Cao
- Qixiang Wang
- Rongcheng Bian
- Rongduo Han
- Shaoqiang Zheng
- Shi Hu
- Shi Suo
- Shijie Ren
- Shijin Zhang
- Shiying Fan
- Shuai Xie
- Tianyi Zhang
- Wei Liu
- Wentao Tan
- Xianghan Meng
- Xiaodong He
- Xing Pan
- Xiran Wang
- Xuyang Peng
- Ya Zhang
- Yang Liu
- Yangyang Duan
- Yanxu Chen
- Yicheng Gong
- Yidan Huang
- Yifei Liu
- Yinhao Bai
- Yongqiang Liu
- Yuesong Zhang
- Yuqi Zhang
- Zerui Xie
- Zhenfang Wang
- Zhennan Shen
- Zheyuan Liu
- Zhuwei Zeng
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency'
url: https://arxiv.org/abs/2604.03044
year: 2026
---

# JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency

JD.com

###### Abstract

We introduce JoyAI-LLM Flash, an efficient Mixture-of-Experts
(MoE) language model designed to redefine the trade-off between
strong performance and token efficiency in the sub-50B
parameter regime. JoyAI-LLM Flash is pretrained on a massive
corpus of 20 trillion tokens and further optimized through a
rigorous post-training pipeline, including supervised
fine-tuning (SFT), Direct Preference Optimization (DPO), and
large-scale reinforcement learning (RL) across diverse
environments. To improve token efficiency, JoyAI-LLM Flash
strategically balances *thinking* and *non-thinking*
cognitive modes and introduces FiberPO, a novel RL algorithm
inspired by fibration theory that decomposes trust-region
maintenance into global and local components, providing unified
multi-scale stability control for LLM policy optimization. To
enhance architectural sparsity, the model comprises 48B total
parameters while activating only 2.7B parameters per forward
pass, achieving a substantially higher sparsity ratio than
contemporary industry leading models of comparable scale. To
further improve inference throughput, we adopt a joint
training–inference co-design that incorporates dense
Multi-Token Prediction (MTP) and Quantization-Aware Training
(QAT). We release the checkpoints for both JoyAI-LLM-48B-A3B
Base and its post-trained variants on Hugging Face to support
the open-source community.

## 1 Introduction

The development of highly capable Large Language Models (LLMs) is
increasingly constrained by two intertwined challenges: poor
token efficiency and high computational cost
[du2026ockbenchmeasuringefficiencyllm]. During inference,
many models consume an excessive number of tokens to produce
accurate outputs. Although scaling test-time computation
[snell2024scalingllmtesttimecompute] has historically
yielded substantial performance gains, there is a growing need to
fundamentally rethink intelligence from the perspective of
efficiency.

We introduce JoyAI-LLM Flash, an instruct language
model [liu2024deepseekv2] with chat, short chain-of-thought
(sCoT), and agentic capabilities. Built on a sparse
Mixture-of-Experts (MoE) architecture, JoyAI-LLM Flash
substantially advances both throughput and performance at
inference time by activating only a small fraction of its
parameters in each forward pass. Specifically, JoyAI-LLM Flash
adopts a pure attention-based architecture with a learned MLP
router that activates 8 out of 256 experts, along with 1 shared
expert. The model comprises 48B total parameters, of which only
2.7B are activated per forward pass (or 3.2B including
embeddings). As shown in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency"), JoyAI-LLM Flash
achieves competitive or superior token-efficiency performance
compared with other state-of-the-art models of similar scale. The
figure reports the average accuracy and token consumption across
eighteen benchmarks used in post-training evaluation, where
models in the upper-right region are more token-efficient.
Furthermore, under the 8K-input/16K-output setting, JoyAI-LLM
Flash achieves 1.45×\times and 1.07×\times speedups over the
pure attention-based models GLM-4.7-Flash
[5team2025glm45agenticreasoningcoding] and Qwen3-30B-A3B
[qwen3], respectively. In terms of multi-token prediction
(MTP) efficiency, defined as the inference speedup of the MTP
model over its non-MTP counterpart, JoyAI-LLM Flash achieves a
1.87×\times speedup, surpassing the hybrid-attention models
Qwen3.5-35B-A3B [qwen3.5] (1.61×\times) and Step-3.5-Flash
[huang2026step35flashopen] (1.39×\times). We open-source
both the base and chat model weights in multiple quantization
formats.

The base model of JoyAI-LLM Flash was pretrained on an extensive
text-only corpus of over 20 trillion tokens, employing a
Warmup-Constant-Cosine-Decay learning rate schedule. To maximize
token utilization and incrementally build model capabilities, we
divide the pretraining curriculum into four stages:

* •

  Foundational Phase: Exposing the model to diverse tokens to build general linguistic capabilities.
* •

  Code-Math-Enhancement Phase: Processing tokens with a significantly upweighted proportion of code and math data.
* •

  Mid-Training Phase: Focusing on ultra-high-quality tokens to refine reasoning and alignment.
* •

  Long-Context Phase: Utilizing nature long context tokens specifically engineered to extend the context window to 128K.

Empirically, JoyAI-LLM Flash Base achieves the competitive
efficacy and efficiency for its size class across general
knowledge, math, code, and comprehensive understanding
evaluations.

Following pretraining, we implemented a rigorous post-training
pipeline designed not only to align the model with human intent
and enhance its autonomy, but also to fundamentally optimize
token efficiency. The pipeline starts with heavily supervised
fine-tuning (SFT) on a diverse set of high-quality traces, which
strategically balance ”thinking” and ”non-thinking” cognitive
modes, followed by Direct Preference Optimization (DPO) to refine
responses and ensure robust human preference alignment. To
further advance its reasoning and agentic problem-solving skills,
we perform large-scale Reinforcement Learning (RL) across diverse
environments. Inspired by the algebraic concept of
*fibration*, we introduce a novel RL algorithm, FiberPO,
that decomposes trust-region maintenance into global and local
components, providing unified multi-scale stability control for
LLM policy optimization. Together with these post-training
advances, JoyAI-LLM Flash emerges as a powerful foundation model,
establishing a new baseline for strong performance, token
efficiency, and computational efficiency within the sub-50B
parameter regime.

!(/html/2604.03044/assets/x1.png)

Figure 1: Model performance *vs.* token consumption across different middle-scale LLMs. The accuracy and token consumption averaged across eighteen benchmarks used in post-training evaluation (Table [3](#S3.T3 "Table 3 ‣ 3.4 Instruct Model Evaluation ‣ 3 Post-Training ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency")) are illustrated, where the upper-right region indicates more token-efficient models. Bubble size represents the model parameter count.

We also quantized JoyAI-LLM Flash from bfloat16 to FP8, INT8, FP4 and GGUF. Along with this report, we are releasing the model as follows:

##### Checkpoints

* •

  [JoyAI-LLM Flash Base](https://huggingface.co/jdopensource/JoyAI-LLM-Flash-Base) : the pre-trained base model
* •

  [JoyAI-LLM Flash BF16](https://huggingface.co/jdopensource/JoyAI-LLM-Flash) : the post-trained model
* •

  [JoyAI-LLM Flash FP8](https://huggingface.co/jdopensource/JoyAI-LLM-Flash-FP8) : the post-trained model quantized with the FP8 format delivering an excellent trade-off between performance and efficiency
* •

  [JoyAI-LLM Flash INT8](https://huggingface.co/jdopensource/JoyAI-LLM-Flash-INT8) : the post-trained model quantized with the INT8 format achieving an optimal trade-off between performance and efficiency and compatible with some AI accelerators
* •

  [JoyAI-LLM Flash INT4](https://huggingface.co/jdopensource/JoyAI-LLM-Flash-INT4) : the post-trained model quantized with the INT4 format serving as an ultra-compact variant tailored for environments with extremely restricted VRAM, such as consumer-level processors
* •

  [JoyAI-LLM Flash GGUF](https://huggingface.co/jdopensource/JoyAI-LLM-Flash-GGUF) : the post-trained model quantized with the GGUF ultra-low-bit format delivering broad compatibility across personal computers

The rest of this report is arranged as follows. Section
[2](#S2 "2 Pretraining ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency") describes the pretraining process of
JoyAI-LLM Flash. Section [3](#S3 "3 Post-Training ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency") describes the
post-training process, and Section [4](#S4 "4 Inference ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency") shows the
inference technology used in JoyAI-LLM Flash. Section
[5](#S5 "5 Conclusion and Future Work ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency") concludes the report and presents future
work.

## 2 Pretraining

In this section, we introduce the key features of JoyAI-LLM Flash Base, including its architecture, hyperparameters, and pretraining data. We also demonstrate that JoyAI-LLM Flash Base achieves competitive results to
the state-of-the-art models.

Table 1: Detailed architectural configurations of JoyAI-LLM Flash.

|  |  |
| --- | --- |
| Hyperparameter | JoyAI-LLM Flash 48B-A3B |
| General Settings | |
| Total Layers (NlayersN\_{\text{layers}}) | 40 |
| Dense Layers | 1 |
| Hidden Dimension (dmodeld\_{\text{model}}) | 2048 |
| Vocabulary Size (|V||V|) | 129K |
| Max Context Length | 128K |
| Activation Function | SwiGLU |
| Attention Mechanism | |
| Attention Type | MLA |
| Attention Heads (nhn\_{h}) | 32 |
| QK Non-RoPE Dimension (dnoped\_{\text{nope}}) | 64 |
| QK RoPE Dimension (droped\_{\text{rope}}) | 128 |
| Value Dimension (dvd\_{v}) | 128 |
| Mixture-of-Experts | |
| Total Routed Experts (NrN\_{r}) | 256 |
| Activated Experts (KK) | 8 |
| Shared Experts (NsN\_{s}) | 1 |
| Expert Intermediate Size (ded\_{e}) | 768 |

### 2.1 Model Architecture

As summarized in Table [1](#S2.T1 "Table 1 ‣ 2 Pretraining ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency"), JoyAI-LLM Flash is a Mixture-of-Experts (MoE) model with 48.9B total parameters, of which 3.28B are activated per token. Its micro-architecture draws inspiration from DeepSeek-V3 [liu2024deepseek] and Kimi-K2 [team2025kimi], utilizing Multi-head Latent Attention (MLA) [liu2024deepseekv2] with hidden dimensions of 2048 and 768, respectively. The model incorporates standard components such as RMSNorm [zhang2019root] for layer normalization, RoPE [su2024roformer] for positional encoding, and SwiGLU [dauphin2017language] activation within the feed-forward blocks.

In terms of macro-architecture, JoyAI-LLM Flash consists of 40 Transformer layers. The first layer utilizes a standard dense feed-forward network, while the remaining 39 layers are sparse MoE layers. The MoE module employs a fine-grained architecture with 256 total experts. For each input token, the model activates a total of nine experts: eight routed experts are dynamically selected via a Top-8 gating mechanism, and a single dedicated shared expert is always activated to capture common knowledge. To ensure numerical stability, the gating scores are computed using a sigmoid function and executed in FP32 precision. Additionally, we implement an auxiliary-loss-free load-balancing strategy [wang2024auxiliary] to maintain optimal utilization across the expert pool.

Muon Optimizer. To maximize sample efficiency and accelerate convergence, we employ the Muon optimizer [team2025kimi, jordan6muon]. Unlike standard Adam, which relies on element-wise scaling, Muon optimizes parameters by leveraging matrix orthogonalization, effectively performing a form of Newton-style update on the spectral norm of the gradients. Previous studies [team2025kimi, jordan6muon] have demonstrated that this approach significantly accelerates model convergence compared to Adam-based optimizers. Beyond these advantages, our empirical results reveal that Muon substantially enhances training robustness. During our experiments, training sessions utilizing Adam were frequently plagued by several loss spikes, which required manual intervention or careful adjustment of learning rates. In contrast, training with Muon remained exceptionally stable, with no significant numerical anomalies observed.

Multi-Token Prediction. We append a lightweight, single-layer dense Multi-Token Prediction (MTP) head to jointly optimize training and inference [liu2024deepseek, xiao2026mimo]. During pre-training, this module enriches the learning signal, enabling the model to capture multi-step dependencies for improved data efficiency. During inference, it natively enables speculative decoding. By predicting multiple future tokens in parallel, this mechanism overcomes the latency bottleneck of standard autoregressive decoding and accelerates generation without requiring an external draft model.

### 2.2 Infrastructure

The JoyAI-LLM Flash training system is built upon a highly optimized extension of the Megatron-Core [shoeybi2019megatron] framework. Along with foundational parallelization strategies—including Data Parallelism (DP), Tensor Parallelism (TP) [shoeybi2019megatron], Sequence Parallelism (SP) [shoeybi2019megatron], and the 1F1B Pipeline Parallelism (PP) [huang2019gpipe, harlap1806pipedream, harlap1806pipedream, lamy2023breadth, liu2023hanayo, narayanan2021efficient, qi2023zero, combine-1f1b] schedule, we have introduced several architectural enhancements to maximize throughput and computational efficiency. Our specific configuration employs 2-way Pipeline Parallelism, 8-way Expert Parallelism (EP) [lepikhin2020gshard] spanning two nodes, and ZeRO-1 Data Parallelism [rajbhandari2020zero]. To further accelerate core operations, we integrate FlashAttention-3 [NEURIPS2024\_7ede97c3] for high-performance attention kernels and utilize the DeepEP [liu2024deepseek] library to minimize latency during token dispatch and combination within the MoE layers. Additionally, by leveraging distributed asynchronous checkpointing, we reduce loading times from 15 minutes to 30 seconds, ensuring the model can recover and resume training in less than a minute. We also implement a packing training strategy utilizing block-diagonal masks to isolate unrelated samples and preserve strict causal boundaries. Compared to the conventional full lower-triangular masking approach, this method accelerates the attention forward and backward passes by 50% and 20%, respectively.

### 2.3 Pretraining Data

In this section, we detail the composition and processing pipeline of our pretraining corpus. Our model was trained on a total of 20.7 trillion high-quality tokens. The dataset is curated from four main sources: diverse web crawls, reasoning-intensive code repositories, high-fidelity PDF documents, and large-scale synthetic data. The composition is designed to balance broad general knowledge with deep reasoning capabilities and domain-specific expertise.

#### 2.3.1 Web Data Pipeline

We processed Common Crawl data up to October 2025 using a high-efficiency pipeline:

Text Extraction. To achieve high-quality content extraction, we process WARC files using the Trafilatura library, which more effectively removes boilerplate and menu text while filtering out HTML artifacts to extract the core semantic text.

Rule-based Cleaning. Our data refinement process utilizes the Datatrove framework [penedo2024datatrove], integrated with several customized modules for high-precision filtering:

* •

  Standard Filtering: We employ URL filtering to block known malicious domains and a fastText classifier to retain only high-confidence English and Chinese documents. Content quality is further ensured through quality and heuristic repetition filters.
* •

  Privacy Preservation: The PII (Personally Identifiable Information) detection logic was significantly expanded to cover a more comprehensive set of global identity markers, providing robust anonymization.
* •

  Optimized Decontamination: To address the issue of excessive data removal during decontamination, we introduced an n-gram whitelisting mechanism. This refinement reduces the probability of ”false deletions”, ensuring that only true evaluation overlaps are removed.
* •

  Semantic Safety Classifier: A dedicated BERT-based sensitive content classifier was added to our pipeline. This model performs deep semantic analysis to detect policy-prohibited data, ensuring the final dataset aligns with safety and ethical guidelines.

Deduplication. To mitigate redundancy, we developed a distributed deduplication pipeline based on MinHash-LSH[code-lsh, code-minhash] on a Ray-based distributed cluster. Our approach involves decomposing documents into 7-gram shingles, followed by the generation of compact 128-permutation MinHash signatures. We employ a Jaccard similarity threshold of 0.9 to identify near-duplicate candidates. These candidates are subsequently clustered using a parallelized Union-Find algorithm, ensuring that only a single canonical representative from each equivalence class is retained in the final corpus.

Model-based Filtering. To address the limitations of static rules in capturing nuanced quality, we fine-tuned Qwen [qwen3] model series to create two specialized filtering models:

* •

  Line-level Noise Filter: In filtered web crawled data, we observed persistent noise such as embedded advertisements, navigation bars, and templated boilerplate. We trained a lightweight classifier to evaluate each line, removing non-narrative content while preserving the semantic integrity of the document.
* •

  Multi-dimensional Scoring and Classification: To ensure the highest data quality, we evaluates every document across several key metrics: factual accuracy, linguistic coherence, information density, grammatical correctness, thematic depth, web noise, safety and multi-topic identification.
  We only retained documents categorized as ”high-quality” based on the integrated scores. This strict filtering significantly improved the model’s learning efficiency.

#### 2.3.2 Code Data Pipeline

Rule-Based Cleaning Pipeline. Our raw corpus primarily comes from The Stack v2[code-stackv2] and large-scale GitHub code extraction. We follow a multi-stage cleaning workflow: rule-based filtering to remove obvious noise, model-based scoring for second-stage selection and stratification, and deduplication to reduce redundancy. Generic quality signals capture repetition (high-order n-gram duplication and duplicate-line ratios), length and scale (character/line counts, file size, extreme line lengths), character composition (ratios of alphabetic/digit/whitespace characters, hex-like fragments, hyperlinks/HTML tags), and suspicious patterns such as autogenerated or encoded content. Language-specific signals further characterize structural and semantic parseability (e.g., AST availability), function-to-line ratios, test-file patterns, preprocessing-directive density, and excessive trivial statements (e.g., print, assert, pass). Guided by downstream scoring, we relax overly aggressive heuristics by moving them from hard filtering to the scoring stage, improving the precision–coverage trade-off.

Model-Based Quality Scoring. To reduce the cost of per-sample LLM evaluation, we train a lightweight regression scorer to approximate a large-capacity judge model. We use Qwen2.5-3B-Instruct[qwen2025qwen25technicalreport] as the backbone and target sequences shorter than 32k tokens. Labels are produced by Qwen2.5-Coder-32B-Instruct[code-qwen25coder], which scores each sample 10 times; we take the minimum score to mitigate stochasticity. We define medium-quality data as scores in (2.5, 6) and high-quality data as scores greater than 6.

MinHash-LSH Deduplication. To remove exact and near-duplicate code, we reuse the MinHash-LSH based Ray deduplication scheme introduced in our data cleaning pipeline. In practice, most code duplicates we eliminate are exact file-level copies rather than merely similar variants.

Long-Context Code Construction. For long-context code data, we construct 64k and 128k token sequences by concatenating longer QA pairs and repository-level code that have already passed the preceding filtering and deduplication stages. For repositories, we build lightweight language-specific dependency graphs over modules and files, and within each connected component we apply a topological ordering routine to obtain file sequences that respect dependency directions from lower-level to higher-level modules—that is, an ordering in which every dependency appears before any module that depends on it; for repository-level data we explicitly split the mixture so that roughly half of the sequences follow this topological order and the other half use a random file ordering, exposing the model both to realistic project layouts and to more diverse context permutations.

Code Rewriting and Composition. For synthetic code data, we adopt a single-pass rewriting strategy inspired by SwallowCode[code-swallowcode] and OLMo3[code-olmo3] (SGCR+SCOR). We start from the deduplicated and filtered GitHub corpus and a curated top-20-language subset as seeds, and ask an LLM to rewrite functions, files, and small multi-file snippets into more instruction- and documentation-like forms while preserving semantic intent; rewritten outputs are again deduplicated against both the seed pool and the original source files, routed through the same rule filtering and model scoring pipeline, and we retain samples with scores ¿ 7, most of which empirically fall into the high-quality range (typically scoring above 8).

For QA-style code data, we primarily rely on Nemotron-Competitive-Programming-v1[nvidia2025nemotron3nanoopen] as seed problems and solutions, and use the DeepSeek V3.2[deepseekai2025deepseekv32] model to produce paired “thinking” and “no-thinking” variants; these rewritten QA examples are then treated as code–QA compositions and subjected to the same deduplication and quality filtering stack as the rest of the corpus.

#### 2.3.3 PDF Parsing and Knowledge Extraction

To mitigate the scarcity of high-quality, specialized content in open-web corpora, we curated a massive dataset comprising tens of millions of PDF documents. This collection prioritizes domain-specific knowledge often underrepresented in general web text, including STEM, Medicine, Social Sciences, Education, Humanities, and Law. The processing pipeline follows a rigorous workflow similar to our web text pipeline, with specialized enhancements for the PDF format:

* •

  Text Extraction: We leverage MinerU [niu2025mineru25decoupledvisionlanguagemodel] and DeepSeek-OCR [wei2025deepseek] to perform high-fidelity document parsing. This ensures the precise recovery of complex mathematical formulas, tables, and hierarchical structures.
* •

  Filtering and Deduplication: We implement a series of heuristic rules to rectify post-extraction artifacts, such as repetition or extraction noise. Furthermore, documents are unsuitable for linguistic modeling are systematically filtered.
* •

  Semantic Chunking: Documents are partitioned into segments of approximately 4,096 tokens by utilizing double-newline delimiters as natural boundaries, we ensure that chunks respect semantic integrity, thereby avoiding arbitrary truncation during the training phase.
* •

  Quality Scoring: Finally, the extracted text undergoes a scoring and filtering process consistent with our web-scale pipeline to ensure high data quality.

Unlike standard web crawls, these documents provide the structured, professional knowledge essential for the model to acquire advanced academic and technical expertise.

#### 2.3.4 Large-Scale Data Synthesis

Synthetic data plays a critical role in our data pipeline, evolving from strengthening factual knowledge in early training to eliciting advanced, multi-step reasoning and agentic behavior in later stages.

Factual-knowledge reformulation. We synthesize factual pre-training data via two complementary transformations. First, we apply the MAGA reformulation method [hao2025reformulationpretrainingdataaugmentation] to rewrite high-quality web passages into diversified, instruction-following styles while preserving the original semantics, thereby expanding stylistic coverage and reducing template bias. Second, we perform Nemotron-CC–style QA rewriting [Su2024NemotronCCTC], converting curated Common Crawl text into question–answer pairs and short instructional exemplars that elicit explicit information retrieval and grounded responses.

Long-form reasoning QA synthesis. We construct reasoning-intensive supervision through two streams. First, we use DeepSeek V3.2 [deepseekai2025deepseekv32] to generate full solutions for real-world STEM problems, and retain only responses that are verified either by major voting or by matching available ground truth. Second, inspired by Nvidia’s RQA method [nvidia2025nemotron3nanoopen], we derive graduate-level “Thinking QA” from STEM papers by turning technical content into questions that require multi-step derivations, and we synthesize multiple independent reasoning paths per question to encourage robust, self-consistent reasoning rather than pattern imitation. Overall, during the mid-training stage, we increase the proportion of synthetic data to above 60% of total tokens to explicitly prioritize advanced reasoning.

Agentic trajectory synthesis. To further augment general agentic capabilities, we complement the reasoning-centric mixture with large-scale tool-use trajectories generated through a staged execution pipeline, as illustrated in Figure [2](#S2.F2 "Figure 2 ‣ 2.3.4 Large-Scale Data Synthesis ‣ 2.3 Pretraining Data ‣ 2 Pretraining ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency"). We first sample diverse atomic tasks across domains, then compose them into more challenging multi-intent tasks while removing repeated goals and duplicate combinations. These tasks are compiled into executable scripts and instantiated via multi-turn simulations, where GLM-4.6 [5team2025glm45agenticreasoningcoding] serves as the primary agentic actor to simulate complex user–agent interactions under varied patterns. We subsequently employ an LLM-based evaluator to filter trajectories against a comprehensive set of rubrics, covering aspects such as role consistency, task completeness, and planning conciseness.

!(/html/2604.03044/assets/x2.png)

Figure 2: Agentic trajectory synthesis pipeline

### 2.4 Training Strategy

We train the model using a rampup strategy, combined with a Warmup-Stable-Decay (WSD) learning rate schedule [hu2024minicpm]. A sequence length of 4,096 tokens is used throughout training, except during the context extension stage.

Stage 1 (Foundational pretraining). During the warmup phase, we train on 100B tokens while linearly increasing the batch size from approximately 13M to 38M tokens and ramping the learning rate to a peak of 4.2×10−44.2\times 10^{-4}. In the Stable phase, we maintain a batch size of 38M tokens and a learning rate of 4.2×10−44.2\times 10^{-4}.

Stage 2 (Code-Math-Enhancement pretraining). This phase is also regarded as the decay phase, we train with the learning rate following a cosine schedule from 4.2×10−44.2\times 10^{-4} to 1.4×10−41.4\times 10^{-4}.

Stage 3 (Mid-training). We continue training high-quality data to further refine the model. The learning rate decays from 1.4×10−41.4\times 10^{-4} to 4.2×10−54.2\times 10^{-5}. In this stage, we enable a single-layer dense Multi-Token Prediction (MTP) [liu2024deepseek, xiao2026mimo] with a loss scaling factor of 0.1.

Stage 4 (Context Extension). Training proceeds in two steps, retaining the same Multi-Token Prediction configuration as Stage 2. First, we train with a 64K context window and a batch size of approximately 34M tokens, decaying the learning rate from 4.2×10−54.2\times 10^{-5} to 3.2×10−53.2\times 10^{-5}. We then extend the context window to 128K and train with the learning rate further decaying to 2.0×10−52.0\times 10^{-5}.

Scaling laws. During our model training process, we utilized the scaling law algorithm to guide and inform our approach. Building on previous work [scalinglaw\_google, scalinglaw\_google2, scalinglaw\_openai], we experimented with scaling both the model size and data volume. This strategy was crucial for anticipating the model’s training needs due to the extensive resource requirements.
The scaling law provided predictive insights throughout the training, particularly in terms of resource allocation, as well as adjustments to training hyperparameters and data compositions. As shown in the Figure [3](#S2.F3 "Figure 3 ‣ 2.4 Training Strategy ‣ 2 Pretraining ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency"), we compared the scaling law with model training loss and domain-specific benchmarks. The learning curve for training loss aligned perfectly with the scaling law’s step predictions. Although downstream tasks displayed more variability, their performance was still commendable, generally fluctuating around the scaling law trend.
A noteworthy discovery was the use of model merging to simulate a decay in the learning rate. This approach led to more stable improvements in the performance of sub-domain tasks, aligning with the scaling law trajectory. This finding underscores the potential of model merging to optimize learning dynamics within the framework of scaling laws.

!(/html/2604.03044/assets/x3.png)

(a) LOSS

!(/html/2604.03044/assets/x4.png)

(b) Benchmark

Figure 3: Scaling laws for JOYAI-LLM Flash. The plot illustrates the relationship between training compute and model performance. Data points represent empirical observations, while solid lines indicate the power-law fits.

### 2.5 Base Model Evaluation

To evaluate the comprehensive performance of JoyAI-LLM Flash, we select Qwen3-30B-A3B-Base [qwen3] and the latest Qwen3.5-35B-A3B-Base [qwen3.5] as our competitive baselines. Our evaluation framework is structured around four core domains: General Knowledge, Math, Coding, and Long-Context Processing. This multidimensional evaluation thoroughly validates the model’s foundational capabilities with nine benchmarks.

* •

  General Knowledge: MMLU [hendrycks2020measuring] (5-shot, Cot), MMLU-Pro [wang2024mmlu] (5-shot, Cot), CMMLU [li2024cmmlu] (5-shot, Cot).
* •

  Math: GSM8K [cobbe2021training] (4-shot, Cot), MATH [hendrycks2021measuring] (4-shot), MATH-500 [hendrycks2021measuring] (4-shot).
* •

  Coding: HumanEval [chen2021evaluating] (5-shot), LiveCodeBench [jain2024livecodebench] (v6, 2023.05-2025.04).
* •

  Long-Context: RULER [hsieh2024ruler].

To ensure fair and reproducible comparisons, we adopt a standardized evaluation pipeline. Most benchmarks are executed with OpenCompass under greedy decoding for deterministic reporting. LiveCodeBench and RULER are evaluated using their official repositories to remain consistent with their native leaderboards. For LiveCodeBench, we use the default settings (Temperature=0.20.2, Top-pp=0.950.95, Top-kk=2020, Repetition Penalty=1.051.05); for RULER, we follow the official execution protocol.

Table 2: Comparison of Base Model between Qwen3-30B-A3B, Qwen3.5-35B-A3B and JoyAI-LLM Flash. Best results are marked in bold.

|  |  |  |  |
| --- | --- | --- | --- |
| Task | Qwen3-30B-A3B-Base | Qwen3.5-35B-A3B-Base | JoyAI-LLM Flash-Base |
| General Knowledge | | | |
| MMLU | 82.1 | 88.4 | 84.7 |
| MMLU-Pro | 61.7 | 60.7 | 73.1 |
| CMMLU | 83.6 | 86.1 | 83.1 |
| Math | | | |
| GSM8K | 90.3 | 90.5 | 88.7 |
| MATH | 59.6 | 56.0 | 78.1 |
| MATH-500 | 58.0 | 54.8 | 77.0 |
| Coding | | | |
| HumanEval | 87.8 | 79.8 | 85.3 |
| LiveCodeBench | 37.3 | 42.6 | 39.9 |
| Long-Context | | | |
| RULER (128K) | 61.8 | 88.3 | 77.0 |

Based on the results in Table [2](#S2.T2 "Table 2 ‣ 2.5 Base Model Evaluation ‣ 2 Pretraining ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency"), JoyAI-LLM Flash shows a competitive profile relative to the Qwen models. On broad general-knowledge benchmarks, it is slightly behind the strongest baseline, suggesting its factual coverage is comparable but not consistently better. On reasoning-intensive and math evaluations, it performs more strongly, indicating better robustness on harder multi-step problem solving under the reported setup. For coding, results are broadly on par with the baselines, with small differences depending on the benchmark.

## 3 Post-Training

In contrast to contemporary mid-scale models, JoyAI-LLM Flash dedicates a significantly larger proportion of its computational budget to the post-training phase. We structure this rigorous alignment pipeline into three sequential stages: Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Reinforcement Learning (RL). During the SFT stage, we deliberately interleave ”thinking” and ”non-thinking” cognitive data mixtures. Empirical observations indicate that this hybrid training approach substantially enhances the performance of the instruct model (non-thinking model).
Following SFT, we introduce a dedicated DPO phase to refine response quality and mitigate hallucinations. The inclusion of DPO before RL is strategically motivated by its rapid convergence, providing a highly efficient mechanism for penalizing negative or undesirable responses early in the alignment process.
Finally, building upon the DPO-aligned foundation, we apply a novel, large-scale RL algorithm designed to maximize token efficiency and further elevate the model’s agentic problem-solving capabilities.

### 3.1 Supervised Fine Tuning (SFT)

We establish that the SFT phase is fundamental to realizing the comprehensive capabilities of JoyAI-LLM Flash. Rather than merely aligning output formats or following instructions, this stage is instrumental in expanding the model’s knowledge and amplifying its core cognitive capacities. To this end, we implement a heavily weighted SFT protocol comprising a diverse training mixture across three distinct categories: general SFT, environment and agent learning, and tool-integrated reasoning (TIR).

#### 3.1.1 General SFT

Our general Supervised Fine-Tuning (SFT) corpus encompasses a comprehensive spectrum of domains, including mathematics, coding, tool utilization, instruction following, safety, science, Lean theorem proving, creative writing, role-playing, language and multilingual understanding. To construct this dataset, we aggregate substantial volumes of real-world and synthetic prompts, paired with high-quality responses derived from both human annotators and state-of-the-art open-source models, such as JoyAI-LLM Pro, DeepSeek-V3.2 [deepseekai2025deepseekv32], Qwen3-235B-A22B [qwen3], and GPT-OSS 120B [openai2025gptoss120bgptoss20bmodel]. To maintain stringent quality standards, we employ Qwen3-30B-A3B [qwen3] as a specialized filter to systematically remove low-quality queries. Notably, we eschew curriculum learning during this phase in favor of a unified training approach.

Recognizing the pivotal role that data mixture plays across both the mid-training and SFT stages, we apply a human-in-the-loop scheme [omniforce] to dynamically optimize domain proportions. Specifically, we heavily weight coding and agent-centric data to constitute approximately 30% of the mixture, followed closely by general chat and STEM domains.

To maximize computational efficiency during training, we pack sequences to a context length of 128K using a best-fit packing algorithm. This technique reduces the padding ratio to a negligible 0.01%, ensuring that computational resources are exclusively allocated to effective tokens. To preserve strict causal boundaries and prevent cross-contamination between unrelated samples, we apply block-diagonal attention masks within the packed sequences. Overall, this sequence packing strategy yields a 1.5x improvement in training throughput compared to standard padding methodologies.

#### 3.1.2 Environment and Agent Learning

!(/html/2604.03044/assets/x5.png)

Figure 4: Verifiable Environment Pipeline

To systematically optimize Software Engineering (SWE) tasks, as illustrated in Figure 4, we modeled the workflow as a conversion funnel, analogous to the traffic funnels used in internet marketing. From the initial mining of tasks on GitHub to the final rollout of agent trajectories, each successive stage involves a natural ”conversion loss,” where the pool of viable tasks diminishes. By monitoring each stage of this pipeline through the lens of a funnel model, we can precisely analyze bottlenecks and guide optimization efforts to improve the overall conversion rate of usable tasks. The pipeline is structured into three primary phases:

* •

  Candidate Task Mining The initial stage focuses on excavating raw candidate tasks from GitHub. This involves filtering repositories and issues based on specific heuristics to ensure the tasks are substantive and representative of real-world engineering challenges.
* •

  Verifiable Environment Construction Following mining, we transition to the environment setup phase. Here, we transform raw tasks into reproducible environments. The objective is to ensure that each task has a functional ”ground truth” (e.g., passing/failing tests) that can be used for automated verification and subsequent Reinforcement Learning (RL).
* •

  Trajectory Generation and Cold Start The final stage involves rolling out interaction trajectories using an agent framework. These successful trajectories (pass the test cases) serve as high-quality data for Supervised Fine-Tuning (SFT) to ”cold start” the model’s performance.

Candidate Task Mining. The candidate task pool is primarily derived from real-world GitHub issues, Pull Requests (PRs), and commit metadata. To ensure the data is suitable for model training, we apply a series of heuristic filters—such as the clarity of PR descriptions and the number of files modified—to select tasks with an appropriate level of complexity. Notably, our tasks are not restricted to Python. Instead, our pipeline encompasses 12 mainstream programming languages. To guarantee the quality and maturity of the source repositories, we enforce strict selection criteria: each repository must have over 10 stars and a history of at least two successfully merged Pull Requests. The tasks are categorized into several distinct engineering domains to ensure diverse functional coverage:

* •

  Bug Fix Identifying, diagnosing, and resolving defects within the source code.
* •

  Feature Enhancement Improving or expanding existing functionality to add value to the project.
* •

  Refactoring Modifying source code to optimize internal structure without changing external behavior.
* •

  Test Case Generation Automatically generating unit tests and utilizing test suites to verify code integrity.

Verifiable Environment Construction. To support the training requirements of SWE tasks, we have developed a large-scale infrastructure for execution sandboxes. In practice, we observed that simply constructing a Docker image capable of hosting the repository is insufficient; the execution of test suites often fails due to unresolved package dependencies and other environmental inconsistencies. Consequently, a repository-level image alone cannot adequately support the training workflow. To address this, we have decoupled the environment construction process into two distinct phases. One is the Docker Image Provisioning stage, which focuses on the baseline containerization of the repository. The other is the Test Case Validation stage, where we execute the test suites to verify whether the constructed environment is ”verifiable” — ensuring that the dependencies are correctly configured and the environment is functional for downstream training and evaluation.

* •

  Phase 1: Build & Initialize. Manually constructing executable sandbox environments is both time-consuming and labor-intensive. To address this, we employ an Agent-based approach, leveraging autonomous agents to build Docker images directly from existing GitHub repositories. The construction process utilizes an agentic automation workflow: the agent iteratively attempts to build the image, while human intervention is introduced only for execution failures. Through this process, we extract and refine successful patterns into a reusable Skill Library. By continuously repeating this agentic loop, the system’s efficiency and success rate improve over time. Upon the completion of an image build, a mandatory ”smoke test” is executed. This serves as a preliminary validation step to determine whether the environment possesses the foundational operational capabilities required for subsequent tasks.
* •

  Phase 2: Test Execution & Verification. We employ an Agent-based approach to locate relevant test cases and generate executable test commands. By testing the project code in its states both before and after the Pull Request (PR), we identify Pass-to-Pass (P2P) and Pass-to-Fail (P2F) transitions, which serve as the primary success/failure signals. To provide more precise reward signals for a given patch, we developed a multilingual test log parser. This tool is utilized not only to extract test results but also to evaluate the test coverage of the aforementioned schemes. Based on these metrics, we systematically prune Docker images with insufficient coverage to ensure the quality.

Building Trajectories. To maximize the utility of the verifiable environments, we utilized the SWE-Smith [yang2025swesmith] framework to synthesize a batch of tasks. To ensure sufficient task complexity, we filtered out overly simplistic cases by monitoring the number of modified lines and edited files. Moving forward, the pipeline bifurcates into two distinct branches:

* •

  Trajectory Rollout. We utilize multiple agent frameworks (e.g., OpenHands [openhands], SWE-agent [yang2024sweagent], mini-swe-agent [yang2024sweagent]) to generate (rollout) interaction trajectories that successfully pass the predefined test cases.
* •

  Atomic Capability Task Generation. We derive specialized tasks focused on the atomic capabilities of an agent, such as precise code editing and automated test case generation.

#### 3.1.3 Tool-Integrated Reasoning

Tool-Integrated Reasoning (TIR) enhances Large Language Models (LLMs) by incorporating external tool use into their reasoning process. Unlike traditional models that rely exclusively on pre-trained parametric knowledge, TIR enables models to decide when to invoke tools and to generate specific instructions—such as Python code or search queries. The execution results are then fed back into the model to inform subsequent reasoning and response generation. This iterative approach improves computational accuracy and information freshness, effectively addressing the inherent limitations of LLMs like calculation errors and knowledge cutoffs.

In this section, we construct and analyze four specialized TIR datasets tailored to distinct functional domains. We develop an automated, scalable data synthesis pipeline to generate high-quality reasoning trajectories. This pipeline focuses on capturing the specific reasoning patterns needed for Code Interpretation, Agentic Search, and hybrid scenarios that require the coordinated use of both tools.
Although our proposed JoyAI-LLM Flash is an instruct model, we find adding reasoning/thinking data in the SFT stage can improve the non-thinking capacity of the instruct model.

##### Code-Centric Trajectories.

We extract complex mathematical problems and, where available, their corresponding ground truths from datasets such as OpenR1-Math-220k [openr1\_math\_220k] and Nemotron-Math-v2 [du2025nemotron]. Using DeepSeek-V3.2 [deepseekai2025deepseekv32], we distill these into TIR trajectories.

We have developed an interactive environment between the model and the Python interpreter, allowing the LLM to dynamically utilize the Python interpreter for iterative and symbolic computations. Our Python setup includes essential libraries that enable robust mathematical operations and solving capabilities. For example, the setup employs the math library for basic arithmetic and sympy for symbolic mathematics, with commands demonstrating tasks like initializing symbols and solving equations using from sympy import symbols, Eq, solve.

During data distillation, the model is restricted to a maximum of 20 tool invocations per problem session. Occasionally, the model may produce erroneous code, which the interpreter catches and returns as tool responses, aiding the model in debugging and retrying. After distillation, incomplete trajectories, as well as those containing plotting code such as matplotlib, are rigorously filtered out.

This process culminates in a comprehensive dataset of multi-round TIR records, integrating Python tools and providing a substantial foundation to assess and enhance LLM tool-integrated reasoning capabilities.

##### Search-Centric Trajectories.

To cultivate robust information-seeking, multi-hop reasoning, and agentic tool-use capabilities, we construct a search-centric trajectory corpus by distilling execution traces from DeepSeek-V3.2 [deepseekai2025deepseekv32] across three data sources: Complex QA & Agentic Benchmarks ( 6,601 queries aggregated from seven established benchmarks including 2WikiMultihopQA [2wiki], MuSiQue [musique], Bamboogle [bamboolge], SimpleQA [simpleqa], FRAMES [frames], ScholarSearch [scholarsearch], and GAIA [gaia], covering multi-hop reasoning, factuality, complex RAG, and real-world agentic tasks), TaskCraft [taskcraft] (17K search-relevant instances selected from a large pool of tool-intensive agentic tasks spanning single-step to expert-level multi-step executions), and Nemotron-Science-v1 [NemotronPostTrainingDatasetV1] (20K sampled instances from a multiple-choice scientific reasoning corpus). For sources with verifiable ground-truth labels, we retain only trajectories with correct final answers. The resulting corpus comprises trajectories with an average of 8.64 search invocations each.

Rather than exposing the model to raw search results, we introduce a summary agent as the sole interface to search results. Upon each search invocation, the model issues a structured call with explicit search keywords and an intent statement; the summary agent returns a concise, query-focused synthesis of the retrieved web pages. This design prevents overly long contexts, reducing computational overhead while preserving model performance during both training and inference.

##### Hybrid Tool-Integrated Trajectories.

Beyond task-specific datasets, we further curate a sophisticated category of trajectories that necessitate the synergistic coordination of both code interpreters and search engines. In these complex scenarios, the model must exhibit high-level planning: typically utilizing Agentic Search to retrieve specialized domain knowledge or external constants, followed by Code Interpretation to perform rigorous algorithmic verification or numerical modeling based on the retrieved data.

##### Terminal-Centric Trajectories.

To enhance the model’s proficiency in terminal-based operations, we synthesize a diverse set of task scenarios within a standardized, constrained Docker environment. Despite the restricted scope of the environment, we achieve high task diversity by utilizing capability decomposition and evolutionary sampling to expand from initial seed data. We employ DeepSeek-V3.2 [deepseekai2025deepseekv32] to generate reasoning-action trajectories for each task. To ensure data quality, we implement an automated validation pipeline in which an LLM-based judge evaluates each trajectory across five key dimensions: completion, correctness, efficiency, safety, and overall quality. Trajectories with low scores are strictly excluded. This filtering mechanism ensures that only functionally viable and safe data remain for training, providing a stable yet diverse signal for subsequent SFT.

By constructing these four TIR datasets, we aim to comprehensively evaluate and improve models’ abilities to integrate diverse external tools into their reasoning process, thereby pushing the frontier of automated, tool-augmented artificial intelligence.

### 3.2 Direct Preference Optimization (DPO)

During the Direct Preference Optimization (DPO) phase, we train the model for 1,000 steps utilizing a learning rate of 1e-6 with cosine-decay to 1e-7 and a global batch size of 256. To construct the DPO dataset, we curate STEM, general conversational, and safety queries distinct from those of the SFT stage. We form preference pairs by contrasting high-quality responses with negative examples derived from rejection sampling during SFT, specifically targeting prevalent failure cases such as hallucinations and instruction-following deviations. Ultimately, the DPO stage is crucial to the model’s final performance; its rapid convergence provides a highly efficient mechanism to systematically penalize and eliminate undesirable outputs prior to RL.

### 3.3 Reinforcement Learning (RL)

Large language models are no longer single, monolithic policies:
they are increasingly deployed and trained as heterogeneous
systems—agentic pipelines spanning domains and tools,
mixture-of-experts (MoE) architectures with conditional routing,
and distributed/asynchronous training stacks where optimization
noise and data nonstationarity are structural rather than
incidental. In this regime, alignment via
RLHF [ouyang2022training] must simultaneously handle
multi-scale instability: token-level stochasticity,
trajectory-level drift, and system-level heterogeneity
(domains/experts/agents) interacting in the same update. Existing
PPO-style “proximal”
objectives [schulman2017proximal, shao2024deepseekmath, yu2025dapo]
provide only coarse local controls (mostly per-token clipping)
and limited diagnostics when failures arise from global structure
(e.g., a drifting subset of trajectories, an expert partition, or
a domain slice). This motivates importing more expressive
mathematical structure, beyond new loss heuristics, to build
controllers that can allocate stability budgets across the
relevant global contexts. In our JoyAI-LLM, we develop Fiber Bundle
Gating (FBG), a geometric framework grounded in fiber bundle
theory, and derive FiberPO from it, a concrete policy
optimization objective that decomposes trust-region maintenance
into compositional global and local components, providing
multi-scale stability control with first-order fidelity to the
true RL objective near on-policy and a restorative gradient
structure less explored in existing methods.

FiberPO rests on a principled theoretical foundation developed
in [fiberpo2026]. Classical TRPO [schulman2015trust]
trust regions collapse at the undiscounted horizon γ=1\gamma{=}1
required by LLM RL (the TRPO Vanishing Theorem
in [fiberpo2026]). This does not preclude trust-region-style
stabilization, but it shows that the classical radius cannot be used
as-is, necessitating a decoupling of how trust regions are maintained
(ratio clipping) from the specific radius prescribed by TRPO’s
monotonic improvement guarantee. An intermediate result,
Aggregational Policy Censoring Objective (APC-Obj), achieves this
decoupling by proving that the clipping-based surrogate can exactly
reproduce trust-region updates (the APC-Obj-TRPO Equivalence Theorem
in [fiberpo2026]), so that the clipping mechanism remains
well-defined at any positive radius δ>0\delta>0. APC-Obj also
provides a unified Ratio Gating Formalism from which
PPO [schulman2017proximal],
GRPO [shao2024deepseekmath], and GSPO [zheng2025group]
are each derived as identified relaxations. This taxonomy reveals a
structural gap: token-wise methods (PPO, GRPO) do not bound
trajectory-level drift, while sequence-wise methods (GSPO) suppress
within-trajectory variation. To compose the two scales, we introduce
Fiber Bundle Gating (FBG), a geometric framework that organizes
tokens as a fiber bundle over trajectory-level contexts and
decomposes ratio gating into compositional base-level and fiber-level
operations, with provable first-order agreement with the true RL
objective near on-policy (the FBG First-Order Agreement Theorem
in [fiberpo2026]).

FiberPO is the concrete instantiation of FBG derived from the
APC-Obj objective through a sequence of controlled
transformations [fiberpo2026]. The FiberPO objective
factorizes each token’s gated importance ratio into a
trajectory-level base weight and a token-level gated residual.
The base weight maintains a trust-region budget on
trajectory-level drift through a piecewise-linear aggregate gate
gaggg^{\rm agg}, while the gated residual clips each token’s
deviation from its trajectory mean via logclip\operatorname{logclip}.
This two-scale decomposition provides independent control at both
levels, a structural property rarely explored in all prior
methods. Because fibrations compose algebraically, the same
gating mechanism extends to arbitrary hierarchical depth:
[fiberpo2026] derives a Fibration Gating Hierarchy (FGH) and
instantiates FiberPO-Domain, a four-level variant with
independent trust-region budgets at the domain, prompt group,
trajectory, and token levels. In this report we present the
two-level trajectory-token case for conciseness. The full
theoretical development is given in [fiberpo2026].

#### 3.3.1 The FiberPO Objective

Let ri:=πθ​(ai|si)/πθold​(ai|si)r\_{i}:=\pi\_{\theta}(a\_{i}|s\_{i})/\pi\_{\theta\_{\rm old}}(a\_{i}|s\_{i}) denote the importance ratio for token ii, A^i\hat{A}\_{i} the estimated advantage, and TτT\_{\tau} the length of trajectory τ\tau.
The augmented token space 𝒳¯:={(st​(τ),at​(τ),τ,t)}\bar{\mathcal{X}}:=\{(s\_{t}(\tau),a\_{t}(\tau),\tau,t)\} indexes each token by its trajectory membership and timestep.

###### Definition 3.1 (FiberPO).

The FiberPO objective is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | J^FiberPO​(θ|θold)=∑(s,a,τ,t)∈𝒳¯1|Tjθold|​1Tτ⋅𝒢​(r∙)s,a,τ,t⋅A^s,a,{\hat{J}^{{\text{FiberPO}}}}(\theta|\theta\_{\rm old})=\sum\_{(s,a,\tau,t)\in\bar{\mathcal{X}}}{\frac{1}{{|\mathrm{Tj}^{\theta\_{\rm old}}|}}{\frac{1}{{T\_{\tau}}}}\cdot\mathcal{G}(r\_{\bullet})\_{s,a,\tau,t}\;\cdot\hat{A}\_{s,a}}, |  | (1) |

where Tjθold\mathrm{Tj}^{\theta\_{\rm old}} is the set of sampled trajectories and the gating map 𝒢\mathcal{G} decomposes multiplicatively for each token i≡(s,a,τ,t)i\equiv(s,a,\tau,t):

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒢​(r∙)i=exp∘gagg​(log⁡sτ+,C+,Tτ)exp∘gagg​(log⁡sτ−,C−,Tτ)⏟wτbase:base weight⋅logclip​((sτ(li))−li​ri,ϵ)logclip​((sτ(−li))−li,ϵ)⏟r~ifiber:gated residual.\mathcal{G}(r\_{\bullet})\_{i}\;=\;\underbrace{\frac{\exp\circ\;g^{\rm agg}(\log s\_{\tau}^{+},\;C^{+},\;T\_{\tau})}{\exp\circ\;g^{\rm agg}(\log s\_{\tau}^{-},\;C^{-},\;T\_{\tau})}}\_{w^{\rm base}\_{\tau}\;:\;\text{base weight}}\;\cdot\;\underbrace{\frac{{\rm logclip}\!\left((s\_{\tau}^{(l\_{i})})^{-l\_{i}}\,r\_{i},\;\epsilon\right)}{{\rm logclip}\!\left((s\_{\tau}^{(-l\_{i})})^{-l\_{i}},\;\epsilon\right)}}\_{\tilde{r}\_{i}^{\rm fiber}\;:\;\text{gated residual}}. |  | (2) |

Each constituent is defined in detail below. At a high level, the
decomposition reflects the fiber bundle structure of sampled RLHF
data [fiberpo2026]: each token’s log-ratio is split into a
trajectory-level component (how much the trajectory as a whole
has drifted) and a token-level residual (how much that token
deviates from its trajectory’s drift). The base weight wτbasew^{\rm base}\_{\tau} corresponds to the base gate in the Fiber Bundle
Gating (FBG). It depends only on trajectory-level aggregate
log-ratios sτ+,sτ−s\_{\tau}^{+},s\_{\tau}^{-}, which separately track positive
and negative drift within each trajectory based on the sign label
li:=sign⁡(log⁡ri)∈{+1,−1}l\_{i}:=\operatorname{sign}(\log r\_{i})\in\{+1,-1\} (with
sτ(li)s\_{\tau}^{(l\_{i})} selecting the same-sign channel and
sτ(−li)s\_{\tau}^{(-l\_{i})} the opposite), and is shared by all tokens in
trajectory τ\tau, controlling how much gradient signal the
trajectory as a whole is permitted to contribute through the
piecewise-linear gate gaggg^{\rm agg}. The gated residual r~ifiber\tilde{r}\_{i}^{\rm fiber} corresponds to the fiber gate. It captures each
token’s deviation from the trajectory aggregate, gated by
logclip\operatorname{logclip} to prevent individual token spikes.
Together, the two components provide compositional multi-scale
control: the base weight maintains a trust-region budget at the
trajectory level, while the gated residual regulates per-token
outliers within each trajectory.

##### Intuitive example.

To illustrate the practical
significance of this decomposition, consider two trajectories
answering “Name a famous landmark”:

*“I love Paris and the Eiffel Tower”*  vs. *“I love Rome and the Colosseum.”*

Globally (trajectory-level), the policy may strongly prefer the
Paris response, perhaps it scores higher overall, so the
aggregate ratio sτ+s\_{\tau}^{+} for that trajectory is large. Without
decoupling, this global preference bias leaks into every token’s
gradient: the token *Colosseum* in the Rome trajectory
receives a weaker learning signal not because the token-level
association “Rome →\to Colosseum” is poor, but simply because
its trajectory is globally less preferred. The residual
decomposition in Eq. [2](#S3.E2 "In Definition 3.1 (FiberPO). ‣ 3.3.1 The FiberPO Objective ‣ 3.3 Reinforcement Learning (RL) ‣ 3 Post-Training ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency") prevents this
contamination. By subtracting the trajectory aggregate from each
token’s log-ratio, the fiber gate r~ifiber\tilde{r}\_{i}^{\rm fiber}
isolates the pure local association, how much “Colosseum”
co-varies with “Rome” relative to what the trajectory drift
alone would predict, and gates it independently via
logclip\operatorname{logclip}. Within each trajectory, token-level
learning thus operates at a uniform, unbiased scale:
P​(Colosseum∣Rome)P(\text{Colosseum}\mid\text{Rome}) and P​(Eiffel
Tower∣Paris)P(\text{Eiffel
Tower}\mid\text{Paris}) are each refined on their own
statistical merits, free from the global preference
P​(Paris trajectory)≫P​(Rome trajectory)P(\text{Paris trajectory})\gg P(\text{Rome trajectory}). The
base weight wτbasew^{\rm base}\_{\tau} then re-couples the
trajectory-level preference when the two scales are composed, so
that global significance is preserved without polluting local
precision. This is the orthogonal, non-interfering decomposition
guaranteed by the reflecting condition πE⁣∗∘K=id𝐁\pi\_{E\*}\circ K=\mathrm{id}\_{\mathbf{B}} [fiberpo2026].

The gating map 𝒢\mathcal{G} satisfies three structural properties
established in [fiberpo2026]: (i) trajectory independence, the
Jacobian of 𝒢\mathcal{G} is block-diagonal over trajectories, fully
decoupling each trajectory’s gradient, (ii) first-order agreement, at
the on-policy point (r∙=𝟏r\_{\bullet}=\mathbf{1}) the Jacobian reduces to
identity, recovering the true RL objective to first order near
on-policy, and (iii) scale separation, the local self-gating term has
O​(1)O(1) magnitude while trajectory-mediated coupling is weighted by
1/Tτ1/T\_{\tau}, so that local gradients dominate near on-policy and
trajectory-level corrections engage only as aggregate drift grows.

We now introduce each constituent of Eq. [2](#S3.E2 "In Definition 3.1 (FiberPO). ‣ 3.3.1 The FiberPO Objective ‣ 3.3 Reinforcement Learning (RL) ‣ 3 Post-Training ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency").
For the base weight wτbasew^{\rm base}\_{\tau} term, the positive and
negative aggregate ratios decompose the trajectory-level drift by
sign:

|  |  |  |  |
| --- | --- | --- | --- |
|  | log⁡sτ+:=1Tτ​∑t=0Tτ−1max⁡(log⁡rst​(τ),at​(τ), 0),log⁡sτ−:=1Tτ​∑t=0Tτ−1max⁡(−log⁡rst​(τ),at​(τ), 0).\log s^{+}\_{\tau}:=\frac{1}{T\_{\tau}}\sum\_{t=0}^{T\_{\tau}-1}\max(\log r\_{s\_{t}(\tau),a\_{t}(\tau)},\,0),\qquad\log s^{-}\_{\tau}:=\frac{1}{T\_{\tau}}\sum\_{t=0}^{T\_{\tau}-1}\max(-\log r\_{s\_{t}(\tau),a\_{t}(\tau)},\,0). |  | (3) |

!(/html/2604.03044/assets/figures/fiberpo/fiberpo_fiber_weight.png)

Figure 5: (a) Aggregate gate gaggg^{\rm agg} (Eq. [4](#S3.E4 "In Intuitive example. ‣ 3.3.1 The FiberPO Objective ‣ 3.3 Reinforcement Learning (RL) ‣ 3 Post-Training ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency")) with three regimes: pass-through (|x|≤C|x|\leq C, slope 11), rollback (C<|x|<C∗:=(1+Tτ−1)​CC<|x|<C^{\*}:=(1+T\_{\tau}^{-1})C, slope −Tτ-T\_{\tau}), and zeroed (|x|≥C∗|x|\geq C^{\*}, output 0). As TτT\_{\tau} increases, the rollback zone narrows (width C/TτC/T\_{\tau}) and gaggg^{\rm agg} approaches a hard clip at ±C\pm C.
(b) Base weight log⁡wτbase\log w\_{\tau}^{\rm base} (Eq. [2](#S3.E2 "In Definition 3.1 (FiberPO). ‣ 3.3.1 The FiberPO Objective ‣ 3.3 Reinforcement Learning (RL) ‣ 3 Post-Training ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency")) in (log⁡s+,log⁡s−)(\log s^{+},\log s^{-})-space with asymmetric thresholds. Dashed lines mark the budget boundaries C±C^{\pm} (onset of rollback), and dotted lines mark the full-gating thresholds C∗±C^{\*\pm} (onset of zeroing). The five global regimes follow a non-monotonic pattern: |log⁡w||\log w| rises through the rollback onset (G-II,r), peaks when one channel is fully gated (G-II), declines under mutual rollback (G-III,r), and collapses to zero when both channels are fully gated (G-III, wτbase=1w^{\rm base}\_{\tau}=1).

The aggregate gating function gaggg^{\rm agg} is a piecewise-linear gate on each sign channel:

|  |  |  |  |
| --- | --- | --- | --- |
|  | gagg​(x,C,Tτ):={xif ​|x|≤Csign⁡(x)​(Tτ+1)​C−Tτ​xif ​C<|x|<(1+Tτ−1)​C0otherwiseg^{\rm agg}(x,C,T\_{\tau}):=\left\{\begin{array}[]{ll}x&\text{if }|x|\leq C\\[5.0pt] \operatorname{sign}(x)(T\_{\tau}+1)C-T\_{\tau}x&\text{if }C<|x|<(1+T\_{\tau}^{-1})C\\[5.0pt] 0&\text{otherwise}\end{array}\right. |  | (4) |

where C∈{C+,C−}C\in\{C^{+},C^{-}\} is the per-channel trust-region budget
satisfying C++C−=δC^{+}+C^{-}=\delta, with C−<C+C^{-}<C^{+} recommended to
compensate the intrinsic KL bias log⁡sτ−≥log⁡sτ+\log s^{-}\_{\tau}\geq\log s^{+}\_{\tau}. The three regimes are: pass-through (|x|≤C|x|\leq C,
gate outputs xx unchanged), rollback (C<|x|<(1+Tτ−1)​CC<|x|<(1+T\_{\tau}^{-1})C, slope reverses to −Tτ-T\_{\tau} producing a
restorative gradient), and zeroed (|x|≥(1+Tτ−1)​C|x|\geq(1+T\_{\tau}^{-1})C,
output 0, fully blocking gradient signal). Since wτbasew^{\rm base}\_{\tau} is a ratio of two independently gated channels
(Eq. [2](#S3.E2 "In Definition 3.1 (FiberPO). ‣ 3.3.1 The FiberPO Objective ‣ 3.3 Reinforcement Learning (RL) ‣ 3 Post-Training ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency")), the combined behavior produces
five global regimes (G-I through G-III) depending on which zone
each sign channel occupies. These range from nominal pass-through
(G-I: both channels transparent, base weight equals the
unmodified importance-sampling ratio) through one-channel
rollback (G-II,r: restorative gradient actively opposes the
drifting channel), one-channel fully gated (G-II: the drifting
channel is zeroed, delivering maximum one-sided correction),
mutual rollback (G-III,r), and extinction (G-III: both channels
fully gated, wτbase=1w^{\rm base}\_{\tau}=1, trajectory-level gradient
vanishes). The restorative rollback property is absent in PPO,
GRPO, and GSPO. Figure [5](#S3.F5 "Figure 5 ‣ Intuitive example. ‣ 3.3.1 The FiberPO Objective ‣ 3.3 Reinforcement Learning (RL) ‣ 3 Post-Training ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency") visualizes
gaggg^{\rm agg} and the five global regimes; see [fiberpo2026]
for detailed regime definitions.

For the gated residual r~ifiber\tilde{r}\_{i}^{\rm fiber} term in
Eq. [2](#S3.E2 "In Definition 3.1 (FiberPO). ‣ 3.3.1 The FiberPO Objective ‣ 3.3 Reinforcement Learning (RL) ‣ 3 Post-Training ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency"), the sign label lil\_{i} partitions the
tokens within each trajectory into two channels, reflecting the
FBG fiber bundle structure [fiberpo2026] where the base
space B=Tjθold×{−1,+1}B=\mathrm{Tj}^{\theta\_{\rm old}}\times\{-1,+1\}
indexes each trajectory by a sign channel: the fiber over (τ,+1)(\tau,+1) collects all tokens whose likelihood increased, and the
fiber over (τ,−1)(\tau,-1) collects those whose likelihood
decreased. Splitting by sign rather than averaging all log-ratios
is essential because the total trajectory drift log⁡r¯τ=log⁡sτ+−log⁡sτ−\overline{\log r}\_{\tau}=\log s^{+}\_{\tau}-\log s^{-}\_{\tau} can be small even when
both log⁡sτ+\log s^{+}\_{\tau} and log⁡sτ−\log s^{-}\_{\tau} are individually large.
In this case, the trajectory contains many tokens that have
shifted substantially in both directions, and the
trajectory-level total variation distance (≈(1/Tτ)​∑t|log⁡rst,at|\approx(1/T\_{\tau})\sum\_{t}|\log r\_{s\_{t},a\_{t}}|) is large and may require
control. Averaging all log-ratios into a single mean would mask
this need for regulation. By tracking each sign channel
independently, gaggg^{\rm agg} detects high total variation in the
importance weights even when the signed average nearly cancels,
and can apply rollback on the offending channel without
suppressing the well-behaved one.

The log-clipping function is logclip⁡(x,ϵ):=exp⁡(clip⁡(log⁡x,±ϵ))\operatorname{logclip}(x,\epsilon):=\exp(\operatorname{clip}(\log x,\pm\epsilon)). In ratio
space, this clamps the argument to [e−ϵ,e+ϵ][e^{-\epsilon},e^{+\epsilon}]. The fiber residual ui:=li​log⁡ri−log⁡sτ(li)u\_{i}:=l\_{i}\log r\_{i}-\log s\_{\tau}^{(l\_{i})} measures each token’s deviation from its
same-sign trajectory mean. Define also the opposite-sign
aggregate vi:=−log⁡sτ(−li)v\_{i}:=-\log s\_{\tau}^{(-l\_{i})}. In terms of uiu\_{i} and
viv\_{i}, the gated residual Eq. [2](#S3.E2 "In Definition 3.1 (FiberPO). ‣ 3.3.1 The FiberPO Objective ‣ 3.3 Reinforcement Learning (RL) ‣ 3 Post-Training ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency") can be
written equivalently as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | r~ifiber=exp⁡(clip⁡(li​ui,±ϵ)−clip⁡(li​vi,±ϵ)).\tilde{r}\_{i}^{\rm fiber}\;=\;\exp\!\bigl(\operatorname{clip}(l\_{i}\,u\_{i},\;\pm\epsilon)-\operatorname{clip}(l\_{i}\,v\_{i},\;\pm\epsilon)\bigr). |  | (5) |

The numerator’s logclip\operatorname{logclip} acts on eli​uie^{l\_{i}u\_{i}},
which involves only the same-sign channel aggregate
sτ(li)s\_{\tau}^{(l\_{i})}, preventing opposite-channel contamination. The
denominator’s logclip\operatorname{logclip} incorporates the
opposite-sign aggregate sτ(−li)s\_{\tau}^{(-l\_{i})} to complete the
subtraction by the trajectory-mean log-ratio log⁡r¯τ\overline{\log r}\_{\tau}. The ϵ\epsilon-clip on the fiber residual uiu\_{i}
induces three local regimes: L-I (unclipped, all tokens retain
full gradient), L-II (selective clipping of outlier tokens), and
L-III (all saturated, gradient governed entirely by the base
weight). The recommended hyper-parameter relationship ϵ≪δ\epsilon\ll\delta ensures that local regulation engages before global
regulation: the fiber gate clips outlier tokens (L-I to L-II)
well before trajectory-level aggregates reach the gaggg^{\rm agg}
budget threshold (G-I to G-II). See [fiberpo2026] for the
joint local–global regime visualization on the probability simplex.

When neither logclip\operatorname{logclip} saturates (i.e., |ui|≤ϵ|u\_{i}|\leq\epsilon and |vi|≤ϵ|v\_{i}|\leq\epsilon), the clips are inactive
and we obtain log⁡r~ifiber=log⁡ri−log⁡r¯τ\log\tilde{r}\_{i}^{\rm fiber}=\log r\_{i}-\overline{\log r}\_{\tau}, the trajectory-mean-centered log-ratio,
recovering the true linear surrogate after multiplication by the
base weight in G-I.

This fiber residual formulation yields a concrete
*token-efficiency* advantage (with empirical evidence in
Section [3.3.2](#S3.SS3.SSS2 "3.3.2 Single-Domain Evaluation ‣ 3.3 Reinforcement Learning (RL) ‣ 3 Post-Training ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency")). Because the logclip acts
on ui=li​log⁡ri−log⁡sτ(li)u\_{i}=l\_{i}\log r\_{i}-\log s\_{\tau}^{(l\_{i})} rather than on log⁡ri\log r\_{i} directly, a token is clipped only when it deviates from the
same-sign trajectory mean by more than ϵ\epsilon, regardless of
the magnitude of the trajectory aggregate log⁡sτ(li)\log s\_{\tau}^{(l\_{i})}
itself. Tokens that shift in concert with the trajectory-level
drift always pass through the FiberPO gating map 𝒢\mathcal{G}
unattenuated, retaining their full gradient signal and
contributing *finer, discriminative per-token update
directions* even when the signed trajectory-level drift is
large (|log⁡sτ±|>ϵ|\log s\_{\tau}^{\pm}|>\epsilon). By contrast, methods such
as PPO and GRPO that clip log⁡ri\log r\_{i} directly tie the clip
threshold to the absolute log-ratio. Once the trajectory-level
drift exceeds the clip bound, the majority of tokens saturate
simultaneously, destroying token-level discrimination and
collapsing the gradient to a coarse trajectory-level signal.

#### 3.3.2 Single-Domain Evaluation

We evaluate FiberPO in a single-domain math RLVR setting and
perform a pure algorithmic comparison against GRPO and GSPO, with
no additional stabilizers (no curriculum learning, no overlong
reward shaping or filters, etc.). We train on DAPO-Math-17k and
evaluate on AIME 2024 following the default evaluation protocol
in DAPO [yu2025dapo]. We initialize our reinforcement
learning phase using the checkpoint produced by the
aforementioned rigorous SFT and DPO stages. This presents a
deliberately challenging scenario for RL optimization, as the
extensive prior alignment significantly diminishes the policy’s
entropy. Figure [6](#S3.F6 "Figure 6 ‣ 3.3.2 Single-Domain Evaluation ‣ 3.3 Reinforcement Learning (RL) ‣ 3 Post-Training ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency") shows that
FiberPO’s training and validation curves both rise monotonically
in the latter half of training. All methods are trained in
verl [sheng2024hybridflow] with matched infrastructure. We
use a learning rate of 10−610^{-6} for the RL stage.

!(/html/2604.03044/assets/figures/fiberpo/math_comparison.png)

Figure 6: Single-domain RLVR on
DAPO-Math-17k [yu2025dapo]: (a) training reward and
(b) validation accuracy (AIME 2024 mean@1) vs. training
step. GRPO collapses after step 60. GSPO stagnates. FiberPO
improves steadily on both metrics.

!(/html/2604.03044/assets/x6.png)

Figure 7: Training diagnostics for the single-domain DAPO math
run. Top row (comparative, all three methods): (a) policy
entropy, (b) mean importance ratio on log scale, (c) mean
response length. Bottom row (FiberPO-specific): (d)
gradient norm on log scale, (e) fiber residual and
token-level clip fraction (both in the safe zone throughout
training).

Figure [7](#S3.F7 "Figure 7 ‣ 3.3.2 Single-Domain Evaluation ‣ 3.3 Reinforcement Learning (RL) ‣ 3 Post-Training ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency") provides training diagnostics
that can be interpreted through FiberPO’s theoretical framework.
GRPO’s entropy collapses to 0.038 nats (a 91% reduction from
initialization) and its mean importance ratio exceeds 10310^{3}.
These observations are consistent with the structural gap
identified in Section [3.3.1](#S3.SS3.SSS1 "3.3.1 The FiberPO Objective ‣ 3.3 Reinforcement Learning (RL) ‣ 3 Post-Training ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency"): since GRPO
gates each token’s ratio rir\_{i} independently without bounding
trajectory-level aggregate drift, once the trajectory-level drift
exceeds the per-token clip bound, the majority of tokens in a
trajectory can saturate simultaneously, destroying token-level
discrimination. This plausibly triggers a feedback loop in which
imprecise updates accelerate further drift. A contributing factor
is that GRPO clips the absolute log-ratio log⁡ri\log r\_{i} rather than
a fiber residual: the clip threshold is shared between
trajectory-level drift and token-level variation, so trajectory
drift consumes the clip budget and forces tokens into saturation
even when their *within-trajectory* deviations are small.
Additionally, GRPO lacks a restorative gradient: when a token
exceeds the clip, its gradient is zeroed rather than reversed,
providing no mechanism to oppose drift. GSPO exhibits the
complementary failure mode. Its entropy diverges to 1.99 nats with
response lengths remaining between 7,380 and 8,870 tokens:
by collapsing each trajectory to a single aggregate ratio, GSPO
suppresses within-trajectory variation and prevents the optimizer
from distinguishing token-level quality differences within the
same trajectory. The elevated entropy suggests diffuse probability
mass rather than concentration on efficient solution paths.

FiberPO addresses both failure modes through its two-scale
decomposition. It preserves entropy at 0.43 nats throughout
training while improving validation accuracy from 0.668 to 0.786.
Its mean importance ratio mostly remains at 1.13, near the
on-policy value of 1, consistent with the first-order agreement
property (first-order agreement, property 2
in [fiberpo2026]): the FiberPO Jacobian reduces to the
identity at on-policy, so each update step provides an accurate
gradient direction when the policy has not drifted far. The scale
separation property (property 3 in [fiberpo2026]) further
suggests that near on-policy, the per-token local gradient
dominates with O​(1)O(1) magnitude while trajectory-level
corrections contribute at most O​(1/Tτ)O(1/T\_{\tau}) per token, engaging
primarily only as drift grows. Unlike GRPO, the
logclip\operatorname{logclip} acts on the fiber residual ui=li​log⁡ri−log⁡sτ(li)u\_{i}=l\_{i}\log r\_{i}-\log s\_{\tau}^{(l\_{i})}
(Eq. [5](#S3.E5 "In Intuitive example. ‣ 3.3.1 The FiberPO Objective ‣ 3.3 Reinforcement Learning (RL) ‣ 3 Post-Training ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency")), separating the clip budget from
trajectory-level drift so that tokens with small
within-trajectory deviations retain their full gradient signal.
The fiber residual and token-level clip fraction
(Figure [7](#S3.F7 "Figure 7 ‣ 3.3.2 Single-Domain Evaluation ‣ 3.3 Reinforcement Learning (RL) ‣ 3 Post-Training ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency")e) remain in the safe zone
throughout training, confirming that most tokens satisfy |ui|<ϵ|u\_{i}|<\epsilon and stay in the unsaturated L-I regime. Unlike GSPO,
the per-token gated residual r~ifiber\tilde{r}\_{i}^{\rm fiber} preserves
within-trajectory discrimination, allowing the optimizer to
directly receive and update according to individual token
contributions. The gradient norm supports this interpretation:
FiberPO’s norm increases only 1.5×1.5\times over 100 steps (0.033
to 0.049), compared to 12×12\times for GRPO and 7.5×7.5\times for
GSPO.

The *token-efficiency property*
(Section [3.3.1](#S3.SS3.SSS1 "3.3.1 The FiberPO Objective ‣ 3.3 Reinforcement Learning (RL) ‣ 3 Post-Training ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency")) offers a plausible
explanation for the joint behavior of response length, entropy,
and validation accuracy. FiberPO decreases mean response length
from 7,902 to 4,543 tokens while simultaneously increasing
validation accuracy and preserving entropy. Because the
logclip\operatorname{logclip} acts on the fiber residual ui=li​log⁡ri−log⁡sτ(li)u\_{i}=l\_{i}\log r\_{i}-\log s\_{\tau}^{(l\_{i})} rather than on log⁡ri\log r\_{i}
directly, tokens that shift in concert with trajectory-level
drift pass through unattenuated, retaining their full gradient
signal. The optimizer therefore receives discriminative per-token
directions even under moderate trajectory-level drift, which may
favor concise, correct reasoning paths over verbose ones. GRPO
also shortens responses (7,904 to 2,216 tokens at step 100),
but the accompanying entropy collapse and accuracy degradation
suggest degenerate compression rather than efficiency: once
trajectory-level drift exceeds the clip bound and token-level
discrimination is lost, the model can no longer distinguish valid
from invalid tokens. GSPO maintains high response lengths with
high variance, consistent with its suppressed per-token signal
preventing concentration of probability on efficient solution
paths.

#### 3.3.3 Multi-Domain Extension

Single-domain RL training commonly degrades capabilities outside
the trained domain: a model fine-tuned exclusively on mathematics
may lose instruction-following or coding ability. Multi-domain
training addresses this by optimizing across diverse environments
simultaneously, preserving existing capabilities while gaining on
the trained domains. However, mixing heterogeneous reward
distributions intensifies the stability challenge, since
trajectory-level drift statistics vary across domains. FiberPO’s
compositional two-scale gating is well suited to this setting: the
trajectory-level gate maintains per-trajectory trust regions
regardless of which domain the trajectory belongs to, while the
token-level gate preserves fine-grained credit assignment within
each domain’s distinct reward structure.

We compose our training data from several domains, such as coding agent, math, knowledge, instruction following, language, and so on. All domains supply verifiable reward signals, making the full blend suitable for RLVR without learned reward models.

Following the Gaussian curriculum strategy introduced
in [nvidia2025nemotron3nanoopen], we progressively shift
training from easier to harder prompts. Prior to training, every
prompt is profiled with K=10K{=}10 rollout samples from the DPO
checkpoint to obtain an empirical pass rate pi∈[0,1]p\_{i}\in[0,1].
Prompts with pi=1p\_{i}=1 (already solved) are filtered out. At
training step tt the target difficulty is parameterized by a
Gaussian mean

|  |  |  |  |
| --- | --- | --- | --- |
|  | μt=μ0+(μT−μ0)​tT,\mu\_{t}=\mu\_{0}+(\mu\_{T}-\mu\_{0})\,\frac{t}{T}, |  | (6) |

which decays linearly from μ0=0.8\mu\_{0}=0.8 (easy) to μT=0.2\mu\_{T}=0.2
(hard) over TT total steps
Each prompt receives a sampling weight

|  |  |  |  |
| --- | --- | --- | --- |
|  | wi(t)=exp⁡(−12​(pi−μtσ)2),σ=0.15,w\_{i}^{(t)}=\exp\!\Bigl(-\tfrac{1}{2}\bigl(\tfrac{p\_{i}-\mu\_{t}}{\sigma}\bigr)^{2}\Bigr),\qquad\sigma=0.15, |  | (7) |

concentrating the batch around the current target difficulty.

To preserve domain balance across all heterogeneous environments,
we extend the flat Gaussian sampling with a two-level
domain-balanced scheme. At each draw, a domain group gg is first
selected with probability αg\alpha\_{g}, then a prompt is drawn
within gg with probability proportional to wi(t)w\_{i}^{(t)}.
Unspecified group weights default to the square-root-proportional
heuristic αg∝Ng\alpha\_{g}\propto\sqrt{N\_{g}}, where NgN\_{g} is the
number of valid prompts in group gg, with manual overrides for
selected domains.

We reuse the training protocol and FiberPO hyperparameters from our single-domain baseline without introducing any multi-domain-specific tuning. Despite this lack of targeted adjustment, FiberPO achieves stable gains across all domains without exhibiting catastrophic degradation in any single area, confirming that the per-trajectory trust regions employed by FiberPO inherently generalize across heterogeneous reward distributions

### 3.4 Instruct Model Evaluation

To provide a comprehensive assessment of JoyAI-LLM Flash, we evaluate the model on a diverse set of widely used LLM benchmarks covering multiple capabilities, including

* •

  General Knowledge: MMLU [hendrycks2020measuring], MMLU-Pro [wang2024mmlu], HellaSwag [zellers2019hellaswag], CMMLU [li2024cmmlu], C-Eval [huang2023c], GPQA-Diamond [rein2024gpqa], SuperGPQA [pteam2025supergpqa].
* •

  Math Reasoning: MATH-500 [hendrycks2021measuring], AIME′25.
* •

  Coding Ability: HumanEval [chen2021evaluating], LiveCodeBench v6 [jain2024livecodebench], SWE-bench Verified [jimenez2024swebench].
* •

  Instruction Following: AlignBench [liu-etal-2024-alignbench], IFEval [zhou2023ifeval].
* •

  Long-context Ability: RULER [hsieh2024ruler].
* •

  General Tasks:
  LiveBench2024-11-25{}\_{\text{2024-11-25}} [livebench].
* •

  Agent & OpenClaw:
  τ2\tau^{2}-Bench [barres2025tau2], PinchBench.

The experimental results are shown in Table [3](#S3.T3 "Table 3 ‣ 3.4 Instruct Model Evaluation ‣ 3 Post-Training ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency"), where Qwen3-Next-80B-A3B, Qwen3.5-35B-A3B, and Qwen3-30B-A3B are the baseline instruct models. We also include GLM-4.7-Flash-Thinking for comparison, as its reasoning capabilities provide a more direct alignment with our JoyAI-LLM Flash than its standard instruct counterpart.

As can be observed, our JoyAI-LLM Flash achieves remarkable token efficiency. Specifically, on the LiveCodeBench, JoyAI-LLM Flash surpasses GLM-4.7-Flash-Thinking by 1.6% accuracy with a 85% reduction in token usage. Beyond efficiency, JoyAI-LLM Flash also demonstrates strong competitive performance across diverse tasks, including mathematics and long-context understanding.
The token efficiency of JoyAI-LLM Flash is illustrated in Figure [8](#S3.F8 "Figure 8 ‣ 3.4 Instruct Model Evaluation ‣ 3 Post-Training ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency"). Specifically, the accuracy (left bar) and token usage (right bar) are compared across different models, highlighting our efficiency gains.
Notably, although JoyAI-LLM Flash consumes more tokens on PinchBench, it achieves the best accuracy compared to other models.

!(/html/2604.03044/assets/x7.png)

(a)

!(/html/2604.03044/assets/x8.png)

(b)

!(/html/2604.03044/assets/x9.png)

(c)

!(/html/2604.03044/assets/x10.png)

(d)

!(/html/2604.03044/assets/x11.png)

(e)

!(/html/2604.03044/assets/x12.png)

(f)

Figure 8: Comparison of model performance (left bars) and token consumption (right bars) across six benchmarks. Qwen3.5-35B-A3B and JoyAI-LLM Flash are instruct models, “GLM-4.7-Flash-T” refers to GLM-4.7-Flash-Thinking, which is included due to its comparable performance.

Table 3: Comparison with baseline models. Qwen3-Next-80B-A3B, Qwen3.5-35B-A3B, and Qwen3-30B-A3B are instruct models; “GLM-4.7-Flash-T” refers to GLM-4.7-Flash-Thinking, which is additionally included due to its more comparable performance. Results marked with asterisk∗ are directly cited from original papers and differ substantially from our reproduced results.

|  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Task | Qwen3-Next-80B-A3B | | Qwen3.5-35B-A3B | | Qwen3-30B-A3B | | GLM-4.7-Flash-T | | JoyAI-LLM Flash | |
| Acc | #Token | Acc | #Token | Acc | #Token | Acc | #Token | Acc | #Token |
| General Knowledge | | | | | | | | | | |
| MMLU | 86.4 | 400 | 91.2 | 900 | 88.0 | 200 | 88.2 | 2000 | 89.1 | 200 |
| MMLU-Pro | 76.7 | 1500 | 81.6 | 4000 | 78.7 | 1600 | 75.8 | 6200 | 81.6 | 900 |
| HellaSwag | 88.1 | ¡100 | 89.1 | ¡100 | 86.2 | ¡100 | 71.5 | 3000 | 91.7 | ¡100 |
| CMMLU | 89.1 | 500 | 89.5 | 600 | 87.1 | 500 | 80.2 | 4900 | 86.7 | 400 |
| C-Eval | 89.2 | 700 | 91.5 | 800 | 87.8 | 800 | 77.0 | 12200 | 88.7 | 600 |
| GPQA-Diamond | 73.0 | 4000 | 74.2 | 4300 | 66.2 | 4100 | 76.7 | 19900 | 74.5 | 2800 |
| SuperGPQA | 60.0 | 1900 | 62.0 | 3700 | 53.0 | 2000 | 41.0 | 8400 | 55.7 | 1300 |
| Math | | | | | | | | | | |
| MATH-500 | 97.4 | 1700 | 95.8 | 1500 | 97.6 | 1400 | 99.2 | 3700 | 98.2 | 1300 |
| AIME′25 | 70.4 | 6900 | 56.7 | 7000 | 63.8 | 6100 | 92.1 | 26000 | 72.9 | 5400 |
| Coding | | | | | | | | | | |
| HumanEval | 95.1 | 600 | 93.9 | 300 | 92.1 | 400 | 93.9 | 7200 | 94.5 | 900 |
| LiveCodeBench v6 | 58.8 | 8500 | 58.0 | 11800 | 48.1 | 15900 | 64.0∗ | 53600 | 65.6 | 7300 |
| SWE-bench Verified | 31.2 | 25400 | 57.4 | 23200 | 29.0 | 16400 | 59.4∗ | 51400 | 62.6 | 24400 |
| Instruction Following | | | | | | | | | | |
| AlignBench | 8.3 | 800 | 8.1 | 1000 | 7.9 | 1200 | 6.9 | 3600 | 8.0 | 700 |
| IFEval | 84.7 | 600 | 81.8 | 1000 | 80.8 | 500 | 85.4 | 1900 | 82.8 | 400 |
| Long-Context | | | | | | | | | | |
| RULER | 94.2 | 100 | 96.0 | ¡100 | 93.7 | ¡100 | 74.7 | 8300 | 95.7 | ¡100 |
| General Tasks | | | | | | | | | | |
| LiveBench2024-11-25{}\_{\text{2024-11-25}} | 75.9 | 2300 | 71.3 | 4600 | 68.5 | 2700 | 60.9 | 5600 | 72.9 | 1800 |
| Agent & OpenClaw | | | | | | | | | | |
| τ2\tau^{2}-Bench | 38.4 | 2200 | 76.6 | 2800 | 31.6 | 1900 | 69.9 | 3400 | 74.1 | 3000 |
| PinchBench | 83.7 | 107700 | 81.7 | 85900 | 67.0 | 210400 | 77.1 | 145000 | 82.4 | 109100 |

## 4 Inference

The architecture of JoyAI-LLM Flash deliberately adopts a compact parameter scale of 48 billion together with a highly sparse Mixture-of-Experts (MoE) structure.
Meanwhile, we employ a co-design of training and inference optimizations, including Quantization-Aware Training (QAT) and dense Multi-Token Prediction (MTP).
Furthermore, we evaluate the inference throughput performance across various context lengths under the prefill–decode disaggregation setting.

### 4.1 Quantization

JoyAI-LLM Flash adopts both Quantization-Aware Training (QAT) and Post-Training Quantization (PTQ) to achieve the optimal trade-off between model accuracy and throughput.
All quantized models are open-sourced on [HuggingFace](https://huggingface.co/collections/jdopensource/JoyAI-LLM%20Flash).

During the QAT phase, JoyAI-LLM Flash simulates INT4 quantization during training by inserting fake-quantization operators (Quantize →\rightarrow DeQuantize →\rightarrow Quantize) and keeping weights in high precision for stable optimization.
To cope with non-differentiable operations such as rounding and clamping, we adopt Straight-Through Estimators (STE) [yin2019understanding] in backpropagation, allowing gradients to flow to the master weights.
This aligns with the practice of mainstream LLMs such as Kimi-K2-Thinking [kimi\_k2\_thinking] and GLM-5 [glm5\_arxiv].
A beneficial side effect is the stabilization of the reinforcement learning stage: INT4 quantization fosters more robust model rollouts and reduces noise diversity by narrowing the numerical space.
JoyAI-LLM Flash thus maintains high accuracy even when applying the simple Round-To-Nearest quantization scheme at lower bit widths.

During the PTQ phase,
we compare JoyAI-LLM Flash with Qwen3-30B-A3B BF16 baseline under BF16, FP8, and W4AFP8 [modelopt\_quant] quantization in Figure [9](#S4.F9 "Figure 9 ‣ 4.1 Quantization ‣ 4 Inference ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency").
All the experiments were conducted using vLLM [vllm\_repo] and TRT-LLM [tensorrtllm\_repo]. We reported the best out of the two for each model.
Model accuracy is evaluated by the mean accuracy across three distinct domain datasets: MATH-500[hendrycks2021measuring], GPQA [rein2024gpqa], and MBPP [austin2021program].
The results indicate that JoyAI-LLM Flash architecture consistently outperforms the Qwen baseline in terms of accuracy and throughput.
Qwen3-30B-A3B FP8 model yields a throughput gain of 10%, while suffering noticeable accuracy degradation.
In contrast, JoyAI-LLM Flash FP8 model improves throughput by 17% with nearly no accuracy degradation. Moreover, the W4AFP8 model maximizes the throughput gain of nearly 28% over the baseline with a slight accuracy drop of 1.2%.
These findings demonstrate that JoyAI-LLM Flash achieves an optimal trade-off between accuracy and efficiency, even with 1.63× larger model weights than the Qwen baseline.

Furthermore, to accommodate model usage on edge devices, we also released the effective low-bit GGUF [ggufformat] variants of JoyAI-LLM Flash.
Inspired by the practices of NVIDIA NVFP4 quantization and GGUF’s K-Quants, we propose a “DoubleQuant” strategy tailored for less sensitive weights:
partition the weight matrix into super blocks,
apply the aforementioned quantization method within each block,
perform a global-like quantization across blocks and store the double-quantized scales at lower precision (e.g., 6-bit or 8-bit).
Experimental results show that this DoubleQuant strategy achieves comparable accuracy to the BF16 baseline, validating the effectiveness of the proposed approach.

!(/html/2604.03044/assets/figures/TecReport_Accuracy-Throughput.png)

Figure 9: Comparison of Accuracy and Throughput for Quantized Models: JoyAI-LLM Flash vs. Qwen3-30B-A3B. The accuracy is measured by the mean accuracy across three distinct domain datasets: MATH-500, GPQA, and MBPP.

### 4.2 Multi-Token Prediction

JoyAI-LLM Flash features a lightweight dense MTP architecture, achieving state-of-the-art speedup despite a medium acceptance length.
Table [4](#S4.T4 "Table 4 ‣ 4.2 Multi-Token Prediction ‣ 4 Inference ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency") evaluates the MTP performance of JoyAI-LLM Flash on SpechBench [xia-etal-2024-unlocking] under the MTP 3 layers and concurrency 64 configuration. The performance is evaluated by the acceptance length, ratio and speedup over the non-MTP counterpart.
We compare JoyAI-LLM Flash with a suite of MTP-optimized LLMs, including
Qwen3.5-35B-A3B [qwen3.5],
Step-3.5-Flash [huang2026step35flashopen],
MiMo-V2-Flash [xiao2026mimo],
GLM-5 [glm5team2026glm5vibecodingagentic],
GLM-4.7-Flash [5team2025glm45agenticreasoningcoding],
DeepSeek-V3.2 [deepseekai2025deepseekv32],
and DeepSeek-V3 [liu2024deepseek].
JoyAI-LLM Flash achieves the highest speedup of 1.87×, representing a 3% improvement over the closest competitor, GLM-5 (1.82×), and a 72% improvement over the slowest model, GLM-4.7-Flash (1.09×).

Table 4: SpecBench MTP-3 Speculative Decoding Performance. Best results are marked in
bold.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Model | GPU | Speedup | Acceptance | | TPS | |
| length | ratio | user | server |
| JoyAI-LLM Flash | 1 | 1.87× | 2.20 | 40.35 | 66 | 4241 |
| GLM-5 [glm5team2026glm5vibecodingagentic] | 8 | 1.82× | 3.03 | 75.84 | 31 | 1969 |
| DeepSeek-V3.2 [deepseekai2025deepseekv32] | 8 | 1.79× | 2.55 | 63.72 | 31 | 1958 |
| Qwen3.5-35B-A3B [qwen3.5] | 1 | 1.61× | 3.18 | 72.56 | 85 | 5428 |
| MiMo-V2-Flash [xiao2026mimo] | 4 | 1.61× | 2.69 | 67.22 | 47 | 3033 |
| Step-3.5-Flash [huang2026step35flashopen] | 4 | 1.39× | 2.21 | 55.23 | 48 | 3048 |
| DeepSeek-V3 [liu2024deepseek] | 8 | 1.21× | 2.71 | 56.86 | 25 | 1608 |
| GLM-4.7-Flash [5team2025glm45agenticreasoningcoding] | 1 | 1.09× | 2.11 | 36.84 | 26 | 1695 |

To better align with user habits, we integrate MTP with quantization and conduct joint optimization.
The overall performance of MTP is evaluated by the next-n configuration.
We evaluate inference throughput across three quantization formats—BF16, FP8, and W4AFP8—on a randomly sampled dataset with ISL=1K and OSL=2K.
All quantization settings were deployed with TensorRT-LLM [tensorrtllm\_repo], integrated with CUDA Graph, Torch Compile, and Overlap Scheduling techniques.
As shown in Figure [10](#S4.F10 "Figure 10 ‣ 4.2 Multi-Token Prediction ‣ 4 Inference ‣ JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency"), both BF16 and FP8 model achieved optimal throughput under the MTP setting of next-n=3.
Compared to the baseline configuration (BF16 with next-n=0), these settings delivered a throughput acceleration of 1.57× and 1.81×, respectively.
The overall best performance was observed with W4AFP8 quantization at next-n=2, yielding a 1.96× speedup over the same baseline.
The experiments indicate that excessively high MTP layers (e.g., next-n¿3) could introduce diminishing or even negative returns, which is attributed to the additional computational overhead incurred in predicting multiple future tokens.

!(/html/2604.03044/assets/figures/mtp_performance_v2_large1.3.png)

Figure 10: Joint Optimization of MTP and Quantization (ISL/OSL = 1K/2K, Concurrency = 64)

### 4.3 Serving and Scheduling

Modern LLM inference workloads exhibit highly variable sequence lengths, bursty arrival patterns, and drastically distinct compute–memory profiles between prefill and decode stages.
And the serving efficiency is usually workload-dependent: short-context requests are governed by the trade-off of Time To First Token (TTFT) and Time Per Output Token (TPOT), whereas long-context requests are prefill-dominated and benefit more from cross-request prefix reuse.
We therefore evaluate JoyAI-LLM Flash under these two distinct regimes.

Short-context workloads such as interactive chat typically prioritize user interaction experience and use TTFT and TPOT as the key performance metrics.
We build a workload grid with input lengths ranging from 128 to 2048 tokens and output lengths of 128 and 512 tokens. Request rates were varied from 1 to 8 requests per second (RPS).
JoyAI-LLM Flash maintains excellent responsiveness across all tested scales.
To achieve comparable or better performance gains in real-world environments, we provide the following deployment insights based on our practical experience:

* •

  Intra-node PD Improves overall Performance: For smaller models like JoyAI-LLM Flash, we recommend colocating prefill and decode instances on the same node to minimize communication overhead.
* •

  Dynamic PD Adapts well to Real-time Workloads Variations: Performance simulation tools such as AIConfigurator [aiconfigurator\_repo] enable runtime dynamic scaling of PD instances according to SLA targets.

Long-context workloads such as Retrieval-Augmented Generation (RAG) and multi-turn agent tasks typically require large KV cache capacities.
However, limited memory frequently triggers KV cache eviction, which induces redundant recomputation and restricts cross-request KV reuse.
This constraint poses severe challenges for latency-sensitive metrics such as TTFT and degrades user experience.
To simulate this workload, we randomly generated data with no prefix reuse, featuring input lengths of 20,000 tokens and output lengths of 100 tokens. Request rates were varied from 0.25 to 3.0 requests per second (RPS).
Experiments were conducted with a Prefill-Decode (PD) disaggregated deployment. And Mooncake [qin2024mooncake] was employed as a centralized KV cache store to manage cache across requests.
Also, we provide the following deployment insights under this scenario:

* •

  PD Disaggregation Delivers Better Flexibility:
  Compared with aggregated deployment, PD disaggregation supports independent scaling of prefill and decode, and centralized KV caching lets us tune their instance ratio to better match workload demands.
* •

  Choose KV Cache Management Wisely:
  Two mainstream centralized KV cache management schemes exist: remote KV pooling and peer-to-peer (P2P) CPU sharing.
  Selection depends on hardware infrastructure, with the goal of minimizing data transfer overhead.
  Across both approaches, we strongly discourage TCP as the data-plane transport due to its high latency and overhead.
* •

  Choose the Appropriate Cache Write Strategy:
  The write strategy for KV cache should account for available bandwidth.
  When bandwidth is sufficient, we recommend writing to the distributed cache layer immediately upon each cache hit to maximize hit rates.
  While in scenarios with constrained I/O bandwidth, we recommend reducing write frequency and using an eviction policy to preserve hot data.
* •

  Trade-off Between Recomputation and Transfer:
  Smaller models exhibit higher sensitivity to data transfer overhead, even when KV caches are exchanged via P2P RDMA between instances.
  Transfer latency can outweigh recomputation cost, making centralized KV cache management a net-negative optimization.
  Careful evaluation is required to identify the crossover point for a given model and hardware setup.

## 5 Conclusion and Future Work

We present JoyAI-LLM Flash, a state-of-the-art medium-sized instruct language model with 3 billion activated parameters and 48 billion total parameters. JoyAI-LLM Flash was pretrained on 20 trillion text tokens using Muon optimizer, followed by large-scale supervised fine-tuning (SFT), direct preference optimization (DPO), and reinforcement learning (RL) across diverse environments. JoyAI-LLM Flash achieves strong performance across frontier knowledge, reasoning, coding tasks, and agentic capabilities. Moving forward, we aim to extend the model’s paradigm by integrating continual learning and persistent memory, enabling the LLM to dynamically adapt and retain knowledge over time.

## 6 Contribution

##### Project Leaders

Chao Xue, Xiaodong He††Corresponding Author

##### Core Contributors

Bo Zhang, Bohua Cai, Chang Li, Chao Xue, Dongkai Liu, Guoqiang Huang, Jialong Shi, Liang Huang, Ming Ke, Panfeng Shi, Qi Wang, Qiaoqiao Yuan, Qiong Cao, Qixiang Wang, Rongcheng Bian, Shi Suo, Shijie Ren, Shijin Zhang, Shiying Fan, Shuai Xie, Tianyi Zhang, Wei Liu, Wentao Tan, Xiaodong He, Xuyang Peng, Ya Zhang, Yifei Liu, Yinhao Bai, Yuqi Zhang, Yuesong Zhang, Zhenfang Wang

##### Contributors

Aichen Cai, Anmeng Zhang, Anson Li, Changjian Jiang, Changkai Lu, Chaocai Liang, Cheng Zhang, Fei Wang, Haijian Ke, Han Lin, Hao Wang, Ji Miao, Jiacheng Zhang, Jifeng Zhu, Jingjing Qian, Junhui Luo, Junwu Xiong, Lam So, Mingyang Li, Peng Hao, Qian Lai, Qingyu Yin, Rongduo Han, Shaoqiang Zheng, Shi Hu, Xianghan Meng, Xing Pan, Xiran Wang, Yanxu Chen, Yang Liu, Yangyang Duan, Yicheng Gong, Yidan Huang, Yongqiang Liu, Zerui Xie, Zhennan Shen, Zheyuan Liu, Zhuwei Zeng

##### Acknowledgment

We thank Jiepeng Zhou, Kaiqing Lei, Shaoxiong Zhan, Tshihao Tsu, Yao Yao, Yaren Zhang, Yihui Wang, Zhengda Zhou, Zhenting Huang, and Zhihao Gong for their efforts.
