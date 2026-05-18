---
arxiv: '2404.06654'
authors:
- Cheng-Ping Hsieh
- Simeng Sun
- Samuel Kriman
- Shantanu Acharya
- Dima Rekesh
- Fei Jia
- Yang Zhang
- Boris Ginsburg
parser: ar5iv
retrieved: '2026-05-18'
source: paper
title: 'RULER: What''s the Real Context Size of Your Long-Context Language Models?'
url: https://arxiv.org/abs/2404.06654
year: 2024
---

[2404.06654] \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?



# \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?

Cheng-Ping Hsieh∗, Simeng Sun∗, Samuel Kriman, Shantanu Acharya
  
Dima Rekesh, Fei Jia, Yang Zhang, Boris Ginsburg
  
NVIDIA
  
{chsieh,simengs}@nvidia.com

###### Abstract

The needle-in-a-haystack (NIAH) test, which examines the ability to retrieve a piece of information (the “needle”) from long distractor texts (the “haystack”), has been widely adopted to evaluate long-context language models (LMs). However, this simple retrieval-based test is indicative of only a superficial form of long-context understanding. To provide a more comprehensive evaluation of long-context LMs, we create a new synthetic benchmark Ruler with flexible configurations for customized sequence length and task complexity. Ruler expands upon the vanilla NIAH test to encompass variations with diverse types and quantities of needles. Moreover, Ruler introduces new task categories *multi-hop tracing* and *aggregation* to test behaviors beyond searching from context. We evaluate ten long-context LMs with 13 representative tasks in Ruler. Despite achieving nearly perfect accuracy in the vanilla NIAH test, all models exhibit large performance drops as the context length increases. While these models all claim context sizes of 32K tokens or greater, only four models (GPT-4, Command-R, Yi-34B, and Mixtral) can maintain satisfactory performance at the length of 32K. Our analysis of Yi-34B, which supports context length of 200K, reveals large room for improvement  as we increase input length and task complexity. We open source Ruler to spur comprehensive evaluation of long-context LMs. ††\* Authors contributed equally.

## 1 Introduction

Recent advancements in AI system engineering (Dao et al., [2022](#bib.bib14); Jacobs et al., [2023](#bib.bib28); Fu et al., [2024](#bib.bib21)) and language model designs (Chen et al., [2023](#bib.bib9); Xiong et al., [2023](#bib.bib75)) have enabled efficient scaling up of context length for language models (Liu et al., [2024a](#bib.bib39); Young et al., [2024](#bib.bib78)). Previous works (AI21, [2024](#bib.bib1); X.AI, [2024](#bib.bib72); Reid et al., [2024](#bib.bib53); Anthropic, [2024](#bib.bib4)) commonly adopt synthetic tasks, such as passkey retrieval (Mohtashami & Jaggi, [2023](#bib.bib44)) and needle-in-a-haystack (Kamradt, [2023](#bib.bib32)) to evaluate long-context LMs. However, these evaluations are used inconsistently across works and reveal merely the retrieval capability, failing to gauge other forms of long-context understanding.

In this work, we propose Ruler, a new benchmark to evaluate long-context modeling capabilities for language models.
Ruler contains four task categories to test behaviors (Ribeiro et al., [2020](#bib.bib54)) beyond simple retrieval from context:

1. 1.

   Retrieval: we extend the needle-in-a-haystack (Kamradt, [2023](#bib.bib32), NIAH) test to evaluate retrieval capability with diverse types and quantities of needles.
2. 2.

   Multi-hop Tracing: we propose *variable tracking*, a minimal proxy task for coreference chain resolution to check the behavior of tracing entities with multi-hop connections.
3. 3.

   Aggregation: we propose *common*/*frequent words extraction*, proxy tasks for summarization to test the ability to aggregate relevant information that spans long-range context.
4. 4.

   Question Answering: we add distracting information to the input of existing short-context QA datasets to evaluate question answering capability at various context sizes.

Compared to existing realistic benchmarks (Table [1](#S1.T1 "Table 1 ‣ 1 Introduction ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?")), Ruler consists solely of synthetic tasks, which offer the flexibility to control sequence length and task complexity.
The synthetic input in Ruler reduces reliance on parametric knowledge, which interferes with the utilization of long-context input in realistic tasks (Shaham et al., [2023](#bib.bib55); Bai et al., [2023](#bib.bib6)).

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Benchmark & Task | Avg Len | Type | |  | | --- | | Diverse | | Tasks | | |  | | --- | | Min. Parametric | | Knowledge | | |  | | --- | | Controllable | | Context | |
| ZeroSCROLLS | ∼similar-to\sim10k | realistic | ✓ | ✗ | ✗ |
| L-Eval | ∼similar-to\sim8k | realistic | ✓ | ✗ | ✗ |
| BAMBOO | ∼similar-to\sim16k | realistic | ✓ | ✓ | ✗ |
| LongBench | ∼similar-to\sim8k | hybrid | ✓ | ✗ | ✗ |
| LooGLE | ∼similar-to\sim20k | hybrid | ✓ | ✓ | ✗ |
| InfiniteBench | ∼similar-to\sim200k | hybrid | ✓ | ✓ | ✗ |
| Needle-in-a-haystack (NIAH) | any | synthetic | ✗ | ✓ | ✓ |
| Passkey / Line / KV Retrieval | any | synthetic | ✗ | ✓ | ✓ |
| Ruler (Ours) | any | synthetic | ✓ | ✓ | ✓ |

Table 1: Comparison between existing long-context benchmarks and Ruler. “Realistic” type refers to human-annotated while “synthetic” type refers to auto-generated. Ruler includes diverse task domains beyond retrieval, reduces reliance on parametric knowledge with synthetic input, and offers flexibility to control the contexts for different sequence lengths and task complexities. In Ruler, contexts can be adjusted by changing the volume or placement of relevant and distracted information.

Using Ruler, we benchmark GPT-4  (OpenAI: Josh Achiam et al., [2023](#bib.bib47)) and nine open-source models with context length ranging from 4k to 128k. Despite achieving nearly perfect performance on the vanilla NIAH test, all models exhibit large degradation on more complex tasks in Ruler as sequence length increases. While all models claim context size of 32k tokens or greater, our results indicate that only four of them can effectively handle sequence length of 32K by exceeding a qualitative threshold. Moreover, almost all models fall below the threshold before reaching the claimed context lengths.
To obtain fine-grained model comparisons, we aggregate performance from 4k to 128k with two weighted average scores where the weights simulate the length distribution of real-world use cases. The top models - GPT-4, Command-R (Cohere, [2024](#bib.bib12)), Yi-34B (Young et al., [2024](#bib.bib78)), and Mixtral (Jiang et al., [2024](#bib.bib30)), consistently outperform other models regardless of the chosen weighting scheme.

We further analyze Yi-34B, which claims context length of 200K and achieves the 2nd place on Ruler among open-source models. Our results demonstrate large degradation in Yi’s performance as we increase input length and task complexity. At large context sizes, Yi-34B often returns incomplete answers and fails to precisely locate the relevant information. Furthermore, we observe two behaviors emerging with the scaling of context size across multiple models: the increased reliance on parametric knowledge and the increased tendency to copy from context for non-retrieval tasks. Our additional ablations demonstrate that training on longer sequences does not always lead to better performance on Ruler, and that larger model sizes positively correlate with better long-context capabilities. Finally, we show that non-Transformer architectures, such as RWKV and Mamba, still lag behind Transformer by large margins on Ruler.

Our contributions are as follows:

* •

  We propose a new benchmark Ruler for evaluating long-context language models via synthetic tasks with flexible configurations.
* •

  We introduce new task categories, specifically multi-hop tracing and aggregation, to test behaviors other than retrieval from long context.
* •

  We evaluate ten long-context LMs using Ruler and perform analysis across models and task complexities.

We open source Ruler to spur future research in long-context language models.111<https://github.com/hsiehjackson/RULER>

## 2 Related Work

#### Long-context Language Models.

Numerous long-context language models have been introduced lately owing to the progress in engineering, architectural, and algorithmic designs. Flash attention (Dao et al., [2022](#bib.bib14); Dao, [2023](#bib.bib13)) and Ring attention (Liu et al., [2023](#bib.bib38)) significantly reduce the memory footprint required for processing long context. Various sparse attention mechanisms (Child et al., [2019](#bib.bib11); Jaszczur et al., [2021](#bib.bib29)) such as shifted sparse attention (Chen et al., [2024](#bib.bib10)), dilated attention (Ding et al., [2023](#bib.bib15)), and attention sinks (Han et al., [2023](#bib.bib25); Xiao et al., [2024b](#bib.bib74)) were employed to enable efficient context scaling. Novel position embedding methods were proposed to improve length extrapolation in Transformers (Vaswani et al., [2017](#bib.bib68)), including ALiBi (Press et al., [2022](#bib.bib51)), xPOS (Sun et al., [2023b](#bib.bib59)), and RoPE (Su et al., [2023](#bib.bib56)) variants (Chen et al., [2023](#bib.bib9); Xiong et al., [2023](#bib.bib75); Peng et al., [2024](#bib.bib49); Liu et al., [2024b](#bib.bib40); Ding et al., [2024](#bib.bib16); Zhu et al., [2024](#bib.bib81)). Another line of research focuses on reducing context size. This can be achieved by caching previous context using recurrence mechanism (Zhang et al., [2024a](#bib.bib79); Bulatov et al., [2023](#bib.bib7); Martins et al., [2022](#bib.bib42); Wu et al., [2022](#bib.bib71)), or preserving only the salient information within long context via retrieval (Xu et al., [2024](#bib.bib76); Mohtashami & Jaggi, [2023](#bib.bib44); Wang et al., [2024](#bib.bib69); Tworkowski et al., [2024](#bib.bib66); Xiao et al., [2024a](#bib.bib73)) or compression (Jiang et al., [2023](#bib.bib31)). Finally, novel architectures (Gu et al., [2022](#bib.bib24); Fu et al., [2023a](#bib.bib19); Poli et al., [2023](#bib.bib50); Fu et al., [2023b](#bib.bib20); Sun et al., [2023a](#bib.bib58)) such as Mamba (Gu & Dao, [2023](#bib.bib23)) and RWKV (Peng et al., [2023](#bib.bib48)) have also been proposed to efficiently handle long-context input.

#### Long-context Benchmarks and Tasks.

Our work is closely related to other works on benchmarking long-context language models. ZeroSCROLLS (Shaham et al., [2023](#bib.bib55)) covers ten realistic natural language tasks, such as long-document QA and (query-based) summarization. L-Eval (An et al., [2024](#bib.bib2)) also uses realistic data, which was filtered manually to ensure quality. LongBench (Bai et al., [2023](#bib.bib6)) contains tasks in a bilingual setting. InfiniteBench (Zhang et al., [2024b](#bib.bib80)) includes tasks with length greater than 100K tokens. LTM (Castillo et al., [2024](#bib.bib8)) targets the evaluation of long-term conversations. To isolate the effect of parametric knowledge, previous works (Dong et al., [2023](#bib.bib17); Li et al., [2023b](#bib.bib37)) also propose to use documents posted online later than a certain cutoff date, or leverage extremely low-resource materials (Tanzer et al., [2024](#bib.bib60)).
Compared to realistic benchmarks, synthetic tasks are more flexible to control the setup (e.g., sequence length and task complexity) and less affected by parametric knowledge. Recent works have mostly focused on retrieval-based synthetic tasks(Kamradt, [2023](#bib.bib32); Mohtashami & Jaggi, [2023](#bib.bib44); Li et al., [2023a](#bib.bib36); Liu et al., [2024c](#bib.bib41)), with a few on other types of long-context usage, including various types of reasoning (Tay et al., [2021](#bib.bib61)) and long-range discourse modeling (Sun et al., [2022](#bib.bib57)).

## 3 The Ruler Benchmark

Ruler comprises tasks across four categories: *retrieval*, *multi-hop tracing*, *aggregation*, and *question answering* with
all tasks configurable for varying length and complexity (see Table [2](#S3.T2 "Table 2 ‣ 3 The Ruler Benchmark ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?")).

| Task | Configuration | Example |
| --- | --- | --- |
| |  | | --- | | Single | | NIAH | | (S-NIAH) | | |  | | --- | | type\_key = word | | type\_value = number | | type\_haystack = essay | | size\_haystack ∝proportional-to\propto context length | | (essays) ……    One of the special magic numbers for long-context is: 12345. ……    What is the special magic number for long-context mentioned in the provided text?    Answer: 12345 |
| |  | | --- | | Multi-keys | | NIAH | | (MK-NIAH) | | |  | | --- | | num\_keys = 2 | | type\_key = word | | type\_value = number | | type\_haystack = essay | | size\_haystack ∝proportional-to\propto context length | | (essays) ……    One of the special magic numbers for long-context is: 12345.    One of the special magic numbers for large-model is: 54321.    ……    What is the special magic number for long-context mentioned in the provided text?    Answer: 12345 |
| |  | | --- | | Multi-values | | NIAH | | (MV-NIAH) | | |  | | --- | | num\_values = 2 | | type\_key = word | | type\_value = number | | type\_haystack = essay | | size\_haystack ∝proportional-to\propto context length | | (essays) ……    One of the special magic numbers for long-context is: 12345.    One of the special magic numbers for long-context is: 54321.    ……    What are all the special magic numbers for long-context mentioned in the provided text?    Answer: 12345 54321 |
| |  | | --- | | Multi-queries | | NIAH | | (MQ-NIAH) | | |  | | --- | | num\_queries = 2 | | type\_key = word | | type\_value = number | | type\_haystack = essay | | size\_haystack ∝proportional-to\propto context length | | (essays) ……    One of the special magic numbers for long-context is: 12345.    One of the special magic numbers for large-model is: 54321.    ……    What are all the special magic numbers for long-context and large-model mentioned in the provided text?    Answer: 12345 54321 |
| |  | | --- | | Variable | | Tracking | | (VT) | | |  | | --- | | num\_chains = 2 | | num\_hops = 2 | | size\_noises ∝proportional-to\propto context length | | (noises) ……    VAR X1 = 12345 …… VAR Y1 = 54321 ……    VAR X2 = X1 …… VAR Y2 = Y1 ……    VAR X3 = X2 …… VAR Y3 = Y2 ……    Find all variables that are assigned the value 12345.    Answer: X1 X2 X3 |
| |  | | --- | | Common Words | | Extraction | | (CWE) | | |  | | --- | | freq\_cw = 2, freq\_ucw = 1 | | num\_cw = 10 | | num\_ucw ∝proportional-to\propto context length | | aaa bbb ccc aaa ddd eee ccc fff ggg hhh iii iii ……    What are the 10 most common words in the above list?    Answer: aaa ccc iii …… |
| |  | | --- | | Frequent Words | | Extraction | | (FWE) | | |  | | --- | | α𝛼\alpha = 2 | | num\_word ∝proportional-to\propto context length | | aaa bbb ccc aaa ddd eee ccc fff ggg aaa hhh aaa ccc iii iii ……    What are the 3 most frequently appeared words in the above coded text?    Answer: aaa ccc iii |
| |  | | --- | | Question | | Answering | | (QA) | | |  | | --- | | dataset = SQuAD | | num\_document ∝proportional-to\propto context length | | Document 1: …… aaa ……    Document 2: …… bbb ……    Document 3: …… ccc ……    Question: question    Answer: bbb |

Table 2: Task examples with flexible configurations in Ruler.
We use different colors to highlight queries, keys, values, and distractors in our examples.

### 3.1 Retrieval: Needle-in-a-haystack (NIAH)

Recent works (Reid et al., [2024](#bib.bib53); Anthropic, [2023](#bib.bib3)) commonly employ the needle-in-a-haystack (Kamradt, [2023](#bib.bib32), NIAH) test to evaluate long-context modeling capability. The NIAH test is reminiscent of the extensively studied (Hopfield, [1982](#bib.bib26); Graves et al., [2014](#bib.bib22); Olsson et al., [2022](#bib.bib46); Arora et al., [2024](#bib.bib5)) *associative recall* tasks, in which relevant information needs to be retrieved from context given a sufficient query. In Ruler, we include multiple retrieval-based tasks, extending the vanilla NIAH test to evaluate models based on three criteria. Concretely, the retrieval capability should be (1) agnostic to the type of the “needle” and the “haystack”, (2) strong enough to disregard hard distractors, and (3) of high recall when multiple items need to be retrieved. Based on these criteria, we develop four NIAH tasks. The “needle” in each of these tasks is a *key-value* pair inserted into the “haystack” (long distractor texts). The *query* is located at the end of the sequence and serves as a cue for matching the *keys* in the context and subsequently retrieving the associated *values*.

* •

  Single NIAH (S-NIAH):  This is the vanilla NIAH test where a single “needle”222Similar to Liu et al. ([2024a](#bib.bib39)), we use “*the special magic number for XXX is: YYY*” as the needle due to its extendability instead of the sentence about San Francisco proposed by Kamradt ([2023](#bib.bib32)). needs to be retrieved from the “haystack”.
  The *query*/*key*/*value* can take the form of words, numbers (7 digits), or UUIDs (32 digits). The “haystack” can be repeated noise sentences333 Following Mohtashami & Jaggi ([2023](#bib.bib44)), we use “*The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.*” as noise sentences. or Paul Graham essays (Kamradt, [2023](#bib.bib32)).
* •

  Multi-keys NIAH (MK-NIAH):  Multiple “needles” are inserted into the “haystack”, and only one of them needs to be retrieved. The additional “needles” are hard distractors. The most challenging setting is a version where the “haystack” is filled with distractor needles.
* •

  Multi-values NIAH (MV-NIAH):  Multiple “needles” sharing the same *key* are inserted into the “haystack”. All *values* associated with the same *key* need to be retrieved.
* •

  Multi-queries NIAH (MQ-NIAH):  Multiple “needles” are inserted into the “haystack”. All “needles” with distinct keys need to be retrieved. This is the same *multi-query associative recall* task setup used by Arora et al. ([2024](#bib.bib5)). Together with MV-NIAH, these two tasks evaluate the retrieval capability without missing any critical information.

### 3.2 Multi-hop Tracing: Variable Tracking (VT)

Effective discourse comprehension (van Dijk & Kintsch, [1983](#bib.bib67)) is contingent upon successful recognition of newly mentioned entities and establishing the chain of references co-referring to the same entity (Karttunen, [1969](#bib.bib33)) throughout the long context. We develop a new task *variable tracking* to emulate a minimal coreference chain resolution (Ng, [2010](#bib.bib45)) task. This task checks the behavior of tracking relevant co-occurrence patterns and drawing skipped connections within long input. Specifically, a variable X​1𝑋1X1 is initialized with a value V𝑉V, followed by a linear *chain* of variable name binding statements (e.g., X​2=X​1,X​3=X​2,…formulae-sequence𝑋2𝑋1𝑋3

𝑋2…X2=X1,X3=X2,...), which are inserted at various positions of the input. The objective is to return *all* variable names pointing to the same value V𝑉V. The task complexity can be increased by adding more hops (i.e., the times of name binding) or more chains, similar to adding hard distractors in MK-NIAH.

### 3.3 Aggregation: Common Words (CWE) and Frequent Words Extraction (FWE)

In Ruler, we introduce a new category as a proxy for summarization tasks where relevant information constitutes much larger portion of the context, and the target output depends on accurate aggregation of the relevant input.
Concretely, we construct an input sequence by sampling words from a pre-defined (synthetic) word list. In the common word extraction task (CWE), words are sampled from discrete uniform distributions, with the number of common words fixed while the number of uncommon words increases with the sequence length. In the frequent words extraction task (FWE), words are sampled from Zeta distribution.444We draw inspiration from Zipf’s Law (Kingsley Zipf, [1932](#bib.bib34)). Let N𝑁N be the total number of words, which is determined by the context size, the frequency of the k𝑘k-th ranked word (the k𝑘k-th most frequently appeared word) is k−α​Nζ​(α)superscript𝑘𝛼𝑁𝜁𝛼\frac{k^{-\alpha}N}{\zeta(\alpha)}, where ζ​(α)𝜁𝛼\zeta(\alpha) is the Zeta function. We set the top-ranked word to noise. Figure [1](#S3.F1 "Figure 1 ‣ 3.3 Aggregation: Common Words (CWE) and Frequent Words Extraction (FWE) ‣ 3 The Ruler Benchmark ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?") shows an illustration of word frequency in the constructed input. A model needs to return the top-K𝐾K frequent words in the context. In CWE, K𝐾K equals to the number of common words. In FWE, we set K𝐾K to 3, as increasing K𝐾K leads to poor performance even at small context sizes for most models. The task complexity can be adjusted by varying the number of common words or the parameter of Zeta distribution.

Figure 1: In aggregation tasks, we sample words from a vocabulary following the two distributions above. The common words extraction (CWE) samples from uniform distributions. In the frequent words extraction (FWE), the frequency of each word is determined by its rank in the vocabulary and the parameter α𝛼\alpha of Zeta distribution.

### 3.4 Question Answering (QA)

The majority of existing QA datasets (Rajpurkar et al., [2018](#bib.bib52); Yang et al., [2018](#bib.bib77); Trivedi et al., [2022](#bib.bib65)) are designed to answer questions based on short passages. These datasets can be extended to simulate long-context input by adding distracting information. In this task category, we insert the golden paragraphs (i.e., the paragraphs that contain answers) into paragraphs randomly sampled from the same dataset. This category is a real-world adaptation (Ivgi et al., [2023](#bib.bib27)) of NIAH, where the question serves as the query, the golden paragraphs are the “needles”, and the distracting paragraphs form the “haystack”.

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Models | |  | | --- | | Claimed | | Length | | |  | | --- | | Effective | | Length | | 4k | 8k | 16k | 32k | 64k | 128k | |  | | --- | | Avg. | | |  | | --- | | wAvg. | | (inc) | | |  | | --- | | wAvg. | | (dec) | |
| Llama2-7B (chat) | 4k | - | 85.6 | | | |  |  |  |  |  |
| GPT-4 | 128k | 64k | 96.6 | 96.3 | 95.2 | 93.2 | 87.0 | 81.2 | 91.6 | 89.0(1st) | 94.1(1st) |
| Command-R (35B) | 128k | 32k | 93.8 | 93.3 | 92.4 | 89.5 | 84.9 | 76.0 | 88.3 | 85.5(2nd) | 91.1(2nd) |
| Yi (34B) | 200k | 32k | 93.3 | 92.2 | 91.3 | 87.5 | 83.2 | 77.3 | 87.5 | 84.8(3th) | 90.1(3th) |
| Mixtral (8x7B) | 32k | 32k | 94.9 | 92.1 | 92.5 | 85.9 | 72.4 | 44.5 | 80.4 | 72.8(4th) | 87.9(4th) |
| Mistral (7B) | 32k | 16k | 93.6 | 91.2 | 87.2 | 75.4 | 49.0 | 13.8 | 68.4 | 55.6(7th) | 81.2(5th) |
| ChatGLM (6B) | 128k | 4k | 87.8 | 83.4 | 78.6 | 69.9 | 56.0 | 42.0 | 69.6 | 62.0(6th) | 77.2(6th) |
| LWM (7B) | 1M | <4k | 82.3 | 78.4 | 73.7 | 69.1 | 68.1 | 65.0 | 72.8 | 69.9(5th) | 75.7(7th) |
| Together (7B) | 32k | 4k | 88.2 | 81.1 | 69.4 | 63.0 | 0.0 | 0.0 | 50.3 | 33.8(8th) | 66.7(8th) |
| LongChat (7B) | 32k | <4k | 84.7 | 79.9 | 70.8 | 59.3 | 0.0 | 0.0 | 49.1 | 33.1(9th) | 65.2(9th) |
| LongAlpaca (13B) | 32k | <4k | 60.6 | 57.0 | 56.6 | 43.6 | 0.0 | 0.0 | 36.3 | 24.7(10th) | 47.9(10th) |

Table 3: Long Context Performance (%) of selected models evaluated at length from 4k to 128k. Each score is computed by averaging accuracy of 13 tasks in Ruler. The performance exceeding the Llama2-7B performance at 4K (85.6%) is underlined. The effective context length is the maximum length passing this threshold. Weighted average score (wAvg.) aggregates performance across all context sizes, with the weights linearly increasing (inc) or decreasing (dec) to simulate length distribution of real-world usage. We put the rank of each model in the subscript. More details about the selected models are in Appendix [A](#A1 "Appendix A Models ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?").

## 4 Experiments & Results

#### Models & Inference setup

We select 10 long-context LLMs, including 9 open-source models and one closed-source model (GPT-4), covering diverse model sizes (6B to 8x7B with MoE architecture) and claimed context lengths (32k to 1M). Complete information about these models is included in Appendix [A](#A1 "Appendix A Models ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?"). We evaluate all models using vLLM (Kwon et al., [2023](#bib.bib35)), an LLM serving system with efficient KV cache memory management. For all models, we run the inference in BFloat16 on 8 NVIDIA A100 GPUs with greedy decoding.

#### Task configurations

We test all models on 13 tasks ranging diverse complexities from the four categories of Ruler.
The test configurations have been selected (shown in Appendix [B](#A2 "Appendix B Task Configurations ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?")) based on a task correlational study described in Appendix [C](#A3 "Appendix C Task Correlation Analysis ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?").
For each task, we evaluate each model with 500 examples generated for each length from the series (4k, 8k, 16k, 32k, 64k, 128k), while complying with each model’s necessary chat template.555See Appendix [D](#A4 "Appendix D Prompt Templates ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?") for model and tasks templates details.
To prevent the model from refusing to answer a query or generating explanations, we append the task input with an answer prefix and check the presence of the target output with recall-based accuracy.

#### Effective Context Size

We notice large performance degradation in all models as we increase input length in Ruler. To determine the maximum context size a model can *effectively* handle, we grade each model with a fixed threshold, passing which indicates satisfactory performance at the length of evaluation. We use the performance of Llama2-7b model at the 4K context length as the threshold.
We report in Table [3](#S3.T3 "Table 3 ‣ 3.4 Question Answering (QA) ‣ 3 The Ruler Benchmark ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?") the maximum length exceeding the threshold as the “effective length” along with the “claimed length”.

#### Model Ranking Criteria

While the threshold-based grading reveals the discrepancy between claimed and effective length, it lacks details for fine-grained model comparisons. As such, we use a weighted average score to aggregate model performance across various context sizes. We rank models under two weighting schemes: wAvg. (inc) and wAvg. (dec) where the weight linearly increases and decreases with sequence length respectively. Ideally, the weight for each length should be determined by the length distribution of model usage, here we choose the two schemes to simulate the scenarios where longer sequences (inc) or shorter sequences (dec) dominate the distribution.

#### Main Results

We include the results of ten long-context LMs in comparison with the Llama2-7B baseline in Table [3](#S3.T3 "Table 3 ‣ 3.4 Question Answering (QA) ‣ 3 The Ruler Benchmark ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?").666Performance of base models and breakdown by task categories can be found in Appendix [F](#A6 "Appendix F Additional Results ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?").
The performance at a certain length is the average of all 13 tasks in Ruler. While these models all claim effective context of 32K tokens or greater, none of them maintains performance above the Llama2-7B baseline at their claimed length, except for Mixtral, which achieves moderate performance on length doubling the claimed 32K context size. Despite achieving nearly perfect performance on the
passkey retrieval and the vanilla NIAH task (shown in Appendix [E](#A5 "Appendix E Passkey Retrieval and Vanilla NIAH Results ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?")), all models exhibit large degradation in RULER as sequence length increases. The best performant model on Ruler is GPT-4, which has the highest performance at length of 4k and demonstrates the least but non-marginal degradation (15.4) when extending the context to 128K. The top three ranked open-source models, Command-R, Yi-34B and Mixtral, all use a large base frequency in RoPE and are larger in parameter size than other models.
Despite having been trained with context size of 1M, the LWM performs worse than Llama2-7B even at 4K. However, it shows smaller degradation with the increase of context size, therefore achieves higher rank than Mistral-7B when longer sequences receive larger weight (wAvg. inc). This result suggests a trade-off in evaluation between absolute performance on short sequences and the relative degradation with the scaling of context size.

## 5 Task Error Analysis

We evaluate Yi-34B-200K, the 2nd best open-source model on Ruler, with increased input lengths (up to 256K) on more complex tasks to understand the effect of task configurations and failure modes on Ruler.

#### Non-robustness to “needle” types.

Figure [2](#S5.F2 "Figure 2 ‣ Return incomplete information. ‣ 5 Task Error Analysis ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?") (left) shows that while Yi achieves almost perfect performance when using needle of word-number pair in the standard passkey retrieval and vanilla NIAH, performance degrades when the needle takes other forms. We observe the largest degradation in the task of retrieving UUIDs, for which Yi sometimes fail to return the complete 32 digits given long (>>128K) input context.

#### Failure to ignore distractors.

Figure [2](#S5.F2 "Figure 2 ‣ Return incomplete information. ‣ 5 Task Error Analysis ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?") (middle-left) shows that increasing the number of distracting needles steadily lowers performance, with Yi dropping by ∼similar-to\sim40 points at 256K in the extreme version, where the context is full of irrelevant needles (#K=FULL). Error analysis reveals that Yi fails to effectively ignore the hard distractors given long input context, thus incorrectly retrieves values associated with the distractor keys. In the extreme version, Yi often returns values from the vicinity of the target, suggesting coarse match of the range but the lack of precision to locate the key when the target is in-distribution of the noises.

#### Return incomplete information.

Consistent with previous works (Liu et al., [2024a](#bib.bib39); Reid et al., [2024](#bib.bib53)), we notice significant degradation in performance when the model needs to retrieve multiple items from a long input. For instance, increasing the number of queries from 1 to 8 drops the performance by ∼similar-to\sim15 points (Figure [2](#S5.F2 "Figure 2 ‣ Return incomplete information. ‣ 5 Task Error Analysis ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?") right). When the model needs to retrieve multiple values associated with the same key (Figure [2](#S5.F2 "Figure 2 ‣ Return incomplete information. ‣ 5 Task Error Analysis ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?") middle-right), Yi often outputs duplicated answers without returning the complete set of values, implying uneven associations between the key and each of its values.

Figure 2: Performance of Yi-34B in the needle-in-a-haystack (NIAH) tasks. By default, we use word-number as the key-value pair and Paul Graham essays as the haystack. Yi is not robust to the change of needle types and degrades with the increasing amount of distractors. (W: words; N: numbers; U: UUIDs; Full: entire haystack).



Figure 3: Performance of Yi-34B in variable tracking (VT), frequent words extraction (FWE), and QA tasks across different task complexities. Yi shows large degradation and distinct trends with scaled context size in these non-retrieval tasks, demonstrating the need to evaluate behavior beyond retrieval from context.

#### Tendency to copy from context.

We notice that Yi has a strong tendency to copy from context verbatim when scaling the input length. This tendency is most notable in *variable tracking* (VT) and *common words extraction* (CWE) where we include one in-context demonstration at the beginning of the sequence. Over 80% of Yi’s output in the CWE task at 128K is simply a string copied from the one-shot example, whereas the copying is nonexistent for short sequences. 777We also experimented with removing the one-shot example. The model will simply copy the string of the beginning of the input, likely due to the attention sinks (Xiao et al., [2024b](#bib.bib74)). This copying behavior is also present in the LWM model and LongAlpaca, however it is less prevalent in other models, such as Mixtral. This finding further reinforces the need to test behaviors other than retrieval given long input context.

#### Unreliable tracking within context.

For the *variable tracking* task, both adding more chains and more hops contribute to large degradation in Yi’s performance. Yi consistently degrades in the more-hops setting as we increase context size (Figure [3](#S5.F3 "Figure 3 ‣ Return incomplete information. ‣ 5 Task Error Analysis ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?") left), whereas the degradation in the more-chains setting is most significant for lengths greater than 128K (Figure [3](#S5.F3 "Figure 3 ‣ Return incomplete information. ‣ 5 Task Error Analysis ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?") middle-left). Besides the aforementioned copying issue, Yi makes errors due to incorrectly returning empty strings or variables from other chains, implying a lack of ability to reliably trace the same entity within long context. These errors are also frequently observed in models that do not exhibit the copying behavior.

#### Failure to accurately aggregate.

We observe two common failure modes in aggregation tasks: incorrect use of parametric knowledge and inaccurate aggregation. Models that do not exhibit the copying issue in the CWE task, sometimes ignore the contextual information and instead use parametric knowledge to answer the query, especially at large context sizes. For instance, Mistral (7b-instruct-v0.2) returns high frequency words, such as “the”, “an”, “a”, as output without counting the words in context. For the FWE task which demonstrates less the copying issue, Yi fails to correctly output the top frequent words as we decrease the α𝛼\alpha in Zeta distribution (Figure [3](#S5.F3 "Figure 3 ‣ Return incomplete information. ‣ 5 Task Error Analysis ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?") middle-right). Decreasing α𝛼\alpha leads to smaller difference in frequency among words, increasing the difficulty to distinguish the top-frequent words.

#### Frequent hallucination in long-context QA.

For the QA tasks, Yi’s performance approaches its no-context baseline as we extend the context with distracting paragraphs (Figure [3](#S5.F3 "Figure 3 ‣ Return incomplete information. ‣ 5 Task Error Analysis ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?") right). The degradation stems primarily from hallucination and reduced reliance on contextual information. We notice that, at large context sizes, model predictions sometimes are irrelevant to the question and can coincide with the answers of its no-context baseline. The overall worse performance in QA tasks confirms that the fuzzy matching between a query and a relevant paragraph in long context is a more challenging setting than the simplistic NIAH tests, where keys can be exactly located in context.

## 6 Model Analysis

Figure 4: (Left & middle left): Comparison of LargeWorldModel (LWM) series trained up to various context sizes with fixed parameter size of 7B. (Middle right): Comparison of Yi suite models with different parameter sizes with controlled training context length of 200K. (Right): Performance of non-Transformer architectures lags behind the Transformer baseline Llama2-7B by large margin. Length extrapolation is presented with dashed lines.

#### Effect of training context length.

Do models trained with larger context sizes perform better on Ruler? We evaluate the suite of LargeWorldModels (Liu et al., [2024a](#bib.bib39), LWM) of equal parameter size and trained up to various context lengths. Figure [4](#S6.F4 "Figure 4 ‣ 6 Model Analysis ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?") (left & middle-left) shows that larger context sizes overall lead to better performance, but the ranking can be inconsistent for long sequences. For instance, the model trained with 1M context size (LWM-1M) is worse than the one with 512K at length of 256K, likely due to insufficient training for adjusting to the new base frequency in RoPE. Moreover, we observe abrupt performance drops when models need to extrapolate to unseen lengths (e.g., LMW-128K given input of 256K), and almost linear degradation with input length on log scale within the max training context size.

#### Effect of model size

The top models in our main results are much larger than other models. To ablate the effect of model size, we evaluate Yi-34B-200k, Yi-9B-200k, and Yi-6B-200k, all trained up to the same context length using the same data blend. Figure [4](#S6.F4 "Figure 4 ‣ 6 Model Analysis ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?") (middle-right) shows that the 34B model is significantly better than the 6B model on Ruler for both performance at length of 4K and the relative degradation, suggesting the benefit of scaling model sizes for better long-context modeling.

#### Effect of architecture

We evaluate the effective context length for two models with non-Transformer architectures: RWKV-v5 (Peng et al., [2023](#bib.bib48)) and Mamba-2.8B-slimpj (Gu & Dao, [2023](#bib.bib23)). We find that both models demonstrate significant degradation when extending context size to 8K, and both underperform the Transformer baseline Llama2-7B by large margins up till the length of 4K, beyond which Llama2 shows poor length extrapolation performance (Figure [4](#S6.F4 "Figure 4 ‣ 6 Model Analysis ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?") right).

## 7 Conclusion

We present Ruler, a synthetic benchmark for evaluating long-context language models.
Ruler contains diverse task categories, *retrieval*, *multi-hop tracing*, *aggregation* and *question answering*, providing a flexible and comprehensive evaluation of LLM’s long-context capabilities.
We benchmark ten long-context LMs using Ruler with context sizes ranging from 4K to 128K. Despite achieving perfect results in the widely used needle-in-a-haystack test, all models fail to maintain their performance in other tasks of Ruler as we increase input length. We observe common failure modes at large context sizes, including the failure to ignore distractors and ineffective utilization of long context (e.g., simply copy from context or use parametric knowledge instead). We show that Ruler is challenging for even the top-ranked open-source models as we increase task complexity. Our analysis further reveals the large potential for improvement on Ruler and the benefit of scaling model sizes in achieving better long context capabilities.

## References

* AI21 (2024)

  AI21.
  Introducing jamba: Ai21’s groundbreaking ssm-transformer model, 2024.
  URL <https://www.ai21.com/blog/announcing-jamba>.
* An et al. (2024)

  Chenxin An, Shansan Gong, Ming Zhong, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu.
  L-eval: Instituting standardized evaluation for long context language models.
  In *ICLR*, 2024.
* Anthropic (2023)

  Anthropic.
  Long context prompting for Claude 2.1.
  *Blog*, 2023.
  URL <https://www.anthropic.com/index/claude-2-1-prompting>.
* Anthropic (2024)

  Anthropic.
  Introducing the next generation of claude, 2024.
  URL <https://www.anthropic.com/news/claude-3-family>.
* Arora et al. (2024)

  Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, and Christopher Ré.
  Zoology: Measuring and improving recall in efficient language models.
  In *ICLR*, 2024.
* Bai et al. (2023)

  Yushi Bai et al.
  LongBench: A bilingual, multitask benchmark for long context understanding.
  *arXiv:2308.14508*, 2023.
* Bulatov et al. (2023)

  Aydar Bulatov, Yuri Kuratov, and Mikhail S Burtsev.
  Scaling Transformer to 1M tokens and beyond with RMT.
  *arXiv:2304.11062*, 2023.
* Castillo et al. (2024)

  David Castillo, Joseph Davidson, Finlay Gray, José Solorzano, and Marek Rosa.
  Introducing GoodAI LTM benchmark.
  *Blog*, 2024.
  URL <https://www.goodai.com/introducing-goodai-ltm-benchmark/>.
* Chen et al. (2023)

  Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian.
  Extending context window of large language models via positional interpolation.
  In *ICLR*, 2023.
* Chen et al. (2024)

  Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia.
  LongLoRA: Efficient fine-tuning of long-context large language models.
  In *ICLR*, 2024.
* Child et al. (2019)

  Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever.
  Generating long sequences with sparse Transformers.
  *arXiv:1904.10509*, 2019.
* Cohere (2024)

  Cohere.
  Command r, 2024.
  URL <https://docs.cohere.com/docs/command-r#model-details>.
* Dao (2023)

  Tri Dao.
  FlashAttention-2: Faster attention with better parallelism and work partitioning.
  *arxiv:2307.08691*, 2023.
* Dao et al. (2022)

  Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré.
  FlashAttention: Fast and memory-efficient exact attention with IO-awareness.
  In *NeurIPS*, 2022.
* Ding et al. (2023)

  Jiayu Ding et al.
  LongNet: Scaling Transformers to 1,000,000,000 tokens.
  *arXiv:2307.02486*, 2023.
* Ding et al. (2024)

  Yiran Ding et al.
  LongRoPE: Extending LLM context window beyond 2 million tokens.
  *arXiv:2402.13753*, 2024.
* Dong et al. (2023)

  Zican Dong, Tianyi Tang, Junyi Li, Wayne Xin Zhao, and Ji-Rong Wen.
  Bamboo: A comprehensive benchmark for evaluating long text modeling capacities of large language models.
  *arXiv:2309.13345*, 2023.
* Du et al. (2022)

  Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang.
  GLM: General language model pretraining with autoregressive blank infilling.
  In *Proc of the 60th Annual Meeting of the ACL (Volume 1: Long Papers)*, pp.  320–335, 2022.
* Fu et al. (2023a)

  Daniel Y. Fu, Tri Dao, Khaled K. Saab, Armin W. Thomas, Atri Rudra, and Christopher Ré.
  Hungry Hungry Hippos: Towards language modeling with state space models.
  In *ICLR*, 2023a.
* Fu et al. (2023b)

  Daniel Y. Fu et al.
  Simple hardware-efficient long convolutions for sequence modeling.
  *ICML*, 2023b.
* Fu et al. (2024)

  Yao Fu et al.
  Data engineering for scaling language models to 128k context.
  *arXiv:2402.10171*, 2024.
* Graves et al. (2014)

  Alex Graves, Greg Wayne, and Ivo Danihelka.
  Neural Turing machines.
  *arXiv:1410.5401*, 2014.
* Gu & Dao (2023)

  Albert Gu and Tri Dao.
  Mamba: Linear-time sequence modeling with selective state spaces.
  *arXiv:2312.00752*, 2023.
* Gu et al. (2022)

  Albert Gu, Karan Goel, and Christopher Re.
  Efficiently modeling long sequences with structured state spaces.
  In *ICLR*, 2022.
* Han et al. (2023)

  Chi Han, Qifan Wang, Wenhan Xiong, Yu Chen, Heng Ji, and Sinong Wang.
  Lm-infinite: Simple on-the-fly length generalization for large language models.
  *arXiv:2308.16137*, 2023.
* Hopfield (1982)

  John J. Hopfield.
  Neural networks and physical systems with emergent collective computational abilities.
  *Proc of the National Academy of Sciences of the United States of America*, 79 8:2554–8, 1982.
* Ivgi et al. (2023)

  Maor Ivgi, Uri Shaham, and Jonathan Berant.
  Efficient long-text understanding with short-text models.
  *Transactions of the ACL*, 11:284–299, 2023.
* Jacobs et al. (2023)

  Sam Ade Jacobs et al.
  DeepSpeed Ulysses: System optimizations for enabling training of extreme long sequence Transformer models.
  *arXiv:2309.14509*, 2023.
* Jaszczur et al. (2021)

  Sebastian Jaszczur et al.
  Sparse is enough in scaling transformers.
  In *NeurIPS*, 2021.
* Jiang et al. (2024)

  Albert Q Jiang et al.
  Mixtral of experts.
  *arXiv:2401.04088*, 2024.
* Jiang et al. (2023)

  Huiqiang Jiang et al.
  LongLlmLingua: Accelerating and enhancing LLMs in long context scenarios via prompt compression.
  *arXiv:2310.06839*, 2023.
* Kamradt (2023)

  Gregory Kamradt.
  Needle In A Haystack - pressure testing LLMs.
  *Github*, 2023.
  URL <https://github.com/gkamradt/LLMTest_NeedleInAHaystack/tree/main>.
* Karttunen (1969)

  Lauri Karttunen.
  Discourse referents.
  In *COLING*, 1969.
* Kingsley Zipf (1932)

  George Kingsley Zipf.
  *Selected studies of the principle of relative frequency in language*.
  Harvard university press, 1932.
* Kwon et al. (2023)

  Woosuk Kwon et al.
  Efficient memory management for large language model serving with paged attention.
  In *Proc. of the ACM SIGOPS 29th Symposium on Operating Systems Principles*, 2023.
* Li et al. (2023a)

  Dacheng Li, Rulin Shao, et al.
  How long can open-source LLMs truly promise on context length?, 2023a.
  URL <https://lmsys.org/blog/2023-06-29-longchat>.
* Li et al. (2023b)

  Jiaqi Li, Mengmeng Wang, Zilong Zheng, and Muhan Zhang.
  Loogle: Can long-context language models understand long contexts?
  *arXiv:2311.04939*, 2023b.
* Liu et al. (2023)

  Hao Liu, Matei Zaharia, and Pieter Abbeel.
  Ring attention with blockwise Transformers for near-infinite context.
  In *ICLR*, 2023.
* Liu et al. (2024a)

  Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel.
  World model on million-length video and language with Ring Attention.
  *arxiv:2402.08268*, 2024a.
* Liu et al. (2024b)

  Jiaheng Liu et al.
  E2-LLM: Efficient and extreme length extension of large language models.
  *arXiv:2401.06951*, 2024b.
* Liu et al. (2024c)

  Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang.
  Lost in the middle: How language models use long contexts.
  *Transactions of the ACL*, 12:157–173, 2024c.
* Martins et al. (2022)

  Pedro Henrique Martins, Zita Marinho, and Andre Martins.
  ∞\infty-former: Infinite memory Transformer.
  In *Proc. of the 60th Annual Meeting of the ACL (Volume 1: Long Papers)*, 2022.
* Mistral.AI (2023)

  Mistral.AI.
  La plateforme, 2023.
  URL <https://mistral.ai/news/la-plateforme/>.
* Mohtashami & Jaggi (2023)

  Amirkeivan Mohtashami and Martin Jaggi.
  Landmark attention: Random-access infinite context length for Transformers.
  In *Workshop on Efficient Systems for Foundation Models @ ICML*, 2023.
* Ng (2010)

  Vincent Ng.
  Supervised noun phrase coreference research: The first fifteen years.
  In *Proc. of the 48th Annual Meeting of the ACL*, 2010.
* Olsson et al. (2022)

  Catherine Olsson et al.
  In-context learning and induction heads.
  *Transformer Circuits Thread*, 2022.
  https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html.
* OpenAI: Josh Achiam et al. (2023)

  OpenAI: Josh Achiam et al.
  GPT-4 technical report.
  *arXiv:2303.08774*, 2023.
* Peng et al. (2023)

  Bo Peng et al.
  RWKV: Reinventing RNNs for the transformer era.
  In *EMNLP*, 2023.
* Peng et al. (2024)

  Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole.
  YaRN: Efficient context window extension of large language models.
  In *ICLR*, 2024.
* Poli et al. (2023)

  Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, and Christopher Ré.
  Hyena hierarchy: Towards larger convolutional language models.
  In *ICML*, 2023.
* Press et al. (2022)

  Ofir Press, Noah Smith, and Mike Lewis.
  Train short, test long: Attention with linear biases enables input length extrapolation.
  In *ICLR*, 2022.
* Rajpurkar et al. (2018)

  Pranav Rajpurkar, Robin Jia, and Percy Liang.
  Know what you don’t know: Unanswerable questions for SQuAD.
  In *Proc. of the 56th Annual Meeting of the ACL (Volume 2: Short Papers)*, 2018.
* Reid et al. (2024)

  Machel Reid et al.
  Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context.
  *arXiv:2403.05530*, 2024.
* Ribeiro et al. (2020)

  Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh.
  Beyond accuracy: Behavioral testing of NLP models with CheckList.
  In *Proc. of the 58th Annual Meeting of the ACL*, 2020.
* Shaham et al. (2023)

  Uri Shaham, Maor Ivgi, Avia Efrat, Jonathan Berant, and Omer Levy.
  ZeroSCROLLS: A zero-shot benchmark for long text understanding.
  In *EMNLP*, 2023.
* Su et al. (2023)

  Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu.
  RoFormer: Enhanced Transformer with rotary position embedding.
  *arXiv:2104.09864*, 2023.
* Sun et al. (2022)

  Simeng Sun, Katherine Thai, and Mohit Iyyer.
  ChapterBreak: A challenge dataset for long-range language models.
  In *Proc. of the 2022 Conference of the North American Chapter of the ACL: Human Language Technologies*, 2022.
* Sun et al. (2023a)

  Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei.
  Retentive network: A successor to Transformer for large language models.
  *arXiv:2307.08621*, 2023a.
* Sun et al. (2023b)

  Yutao Sun, Li Dong, Barun Patra, Shuming Ma, Shaohan Huang, Alon Benhaim, Vishrav Chaudhary, Xia Song, and Furu Wei.
  A length-extrapolatable Transformer.
  In *Proc. of the 61st Annual Meeting of the ACL (Volume 1: Long Papers)*, 2023b.
* Tanzer et al. (2024)

  Garrett Tanzer, Mirac Suzgun, Eline Visser, Dan Jurafsky, and Luke Melas-Kyriazi.
  A benchmark for learning to translate a new language from one grammar book.
  In *ICLR*, 2024.
* Tay et al. (2021)

  Yi Tay et al.
  Long Range Arena: A benchmark for efficient Transformers.
  In *ICLR*, 2021.
* Together.AI (2023a)

  Together.AI.
  Preparing for the era of 32k context: Early learnings and explorations, 2023a.
  URL <https://www.together.ai/blog/llama-2-7b-32k>.
* Together.AI (2023b)

  Together.AI.
  Llama-2-7b-32k-instruct — and fine-tuning for llama-2 models with together api, 2023b.
  URL <https://www.together.ai/blog/llama-2-7b-32k-instruct>.
* Touvron et al. (2023)

  Hugo Touvron et al.
  Llama 2: Open foundation and fine-tuned chat models.
  *arXiv:2307.09288*, 2023.
* Trivedi et al. (2022)

  Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
  Musique: Multihop questions via single-hop question composition.
  *Transactions of the ACL*, 10:539–554, 2022.
* Tworkowski et al. (2024)

  Szymon Tworkowski et al.
  Focused Transformer: Contrastive training for context scaling.
  *NeurIPS*, 36, 2024.
* van Dijk & Kintsch (1983)

  Teun A. van Dijk and Walter Kintsch.
  Strategies of discourse comprehension.
  In *Academic Press*, 1983.
* Vaswani et al. (2017)

  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin.
  Attention is all you need.
  In *NeurIPS*, 2017.
* Wang et al. (2024)

  Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, and Furu Wei.
  Augmenting language models with long-term memory.
  *NeurIPS*, 36, 2024.
* Wolf et al. (2019)

  Thomas Wolf et al.
  Huggingface’s Transformers: State-of-the-art natural language processing.
  *arXiv:1910.03771*, 2019.
* Wu et al. (2022)

  Qingyang Wu, Zhenzhong Lan, Kun Qian, Jing Gu, Alborz Geramifard, and Zhou Yu.
  Memformer: A memory-augmented Transformer for sequence modeling.
  In *Findings of the ACL: AACL-IJCNLP*, 2022.
* X.AI (2024)

  X.AI.
  Announcing grok-1.5, 2024.
  URL <https://x.ai/blog/grok-1.5>.
* Xiao et al. (2024a)

  Chaojun Xiao et al.
  InfLLM: Unveiling the intrinsic capacity of LLMs for understanding extremely long sequences with training-free memory.
  *arXiv:2402.04617*, 2024a.
* Xiao et al. (2024b)

  Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis.
  Efficient streaming language models with attention sinks.
  In *ICLR*, 2024b.
* Xiong et al. (2023)

  Wenhan Xiong et al.
  Effective long-context scaling of foundation models.
  *arXiv:2309.16039*, 2023.
* Xu et al. (2024)

  Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee, Chen Zhu, Zihan Liu, Sandeep Subramanian, Evelina Bakhturina, Mohammad Shoeybi, and Bryan Catanzaro.
  Retrieval meets long context large language models.
  In *ICLR*, 2024.
* Yang et al. (2018)

  Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning.
  HotpotQA: A dataset for diverse, explainable multi-hop question answering.
  In *EMNLP*, 2018.
* Young et al. (2024)

  Alex Young et al.
  Yi: Open foundation models by 01.AI.
  *arXiv:2403.04652*, 2024.
* Zhang et al. (2024a)

  Peitian Zhang, Zheng Liu, Shitao Xiao, Ninglu Shao, Qiwei Ye, and Zhicheng Dou.
  Soaring from 4k to 400k: Extending LLM’s context with activation beacon.
  *arXiv:2401.03462*, 2024a.
* Zhang et al. (2024b)

  Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Khai Hao, Xu Han, Zhen Leng Thai, Shuo Wang, Zhiyuan Liu, and Maosong Sun.
  ∞\inftybench: Extending long context evaluation beyond 100k tokens.
  *arXiv:2402.13718*, 2024b.
* Zhu et al. (2024)

  Dawei Zhu, Nan Yang, Liang Wang, Yifan Song, Wenhao Wu, Furu Wei, and Sujian Li.
  PoSE: Efficient context window extension of LLMs via positional skip-wise training.
  In *ICLR*, 2024.

## Appendix A Models

We select in total 30 models for evaluation and analysis. Our results in the main text only include aligned models (one closed-source model GPT-4 and 9 open-source models). Besides the aligned models, we also evaluate 7 open-source base models using Ruler. We use the performance of Llama2-7b (base) and Llama2-7b (chat) at context length of 4k as the threshold for determining effective context size.
In our analysis section, we evaluate in total 11 models, including model series Yi and LWM, as well as models of novel architectures, including Mamba and RWKV.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Model | Aligned | Size | Context Length | Huggingface (Wolf et al., [2019](#bib.bib70)) / API |
| GPT-4 (OpenAI: Josh Achiam et al., [2023](#bib.bib47)) | ✓ | - | 128K | gpt-4-1106-preview |
| Command-R (Cohere, [2024](#bib.bib12)) | ✓ | 35B | 128K | CohereForAI/c4ai-command-r-v01 |
| Yi (Young et al., [2024](#bib.bib78)) | ✓ | 34B | 200K | 01-ai/Yi-34B-200K |
| Mixtral (Jiang et al., [2024](#bib.bib30)) | ✓ | 8x7B | 32K | mistralai/Mixtral-8x7B-Instruct-v0.1 |
| Mistral (Mistral.AI, [2023](#bib.bib43)) | ✓ | 7B | 32K | mistralai/Mistral-7B-Instruct-v0.2 |
| ChatGLM (Du et al., [2022](#bib.bib18)) | ✓ | 6B | 128M | THUDM/chatglm3-6b-128k |
| LWM (Liu et al., [2024a](#bib.bib39)) | ✓ | 7B | 1M | LargeWorldModel/LWM-Text-Chat-1M |
| Together (Together.AI, [2023b](#bib.bib63)) | ✓ | 7B | 32K | togethercomputer/Llama-2-7B-32K-Instruct |
| LongChat (Li et al., [2023a](#bib.bib36)) | ✓ | 7B | 32K | lmsys/longchat-7b-v1.5-32k |
| LongAlpaca (Chen et al., [2024](#bib.bib10)) | ✓ | 13B | 32K | Yukang/LongAlpaca-13B |
| Mixtral-base (Jiang et al., [2024](#bib.bib30)) | ✗ | 8x7B | 32K | mistralai/Mixtral-8x7B-v0.1 |
| Mistral-base (Mistral.AI, [2023](#bib.bib43)) | ✗ | 7B | 32K | alpindale/Mistral-7B-v0.2-hf |
| LWM-base (Liu et al., [2024a](#bib.bib39)) | ✗ | 7B | 1M | LargeWorldModel/LWM-Text-1M |
| LongLoRA-base (Chen et al., [2024](#bib.bib10)) | ✗ | 7B | 100K | Yukang/Llama-2-7b-longlora-100k-ft |
| Yarn-base(Peng et al., [2024](#bib.bib49)) | ✗ | 7B | 128K | NousResearch/Yarn-Llama-2-7b-128k |
| Together-base (Together.AI, [2023a](#bib.bib62)) | ✗ | 7B | 32K | togethercomputer/Llama-2-7B-32K |
| Jamba-base (AI21, [2024](#bib.bib1)) | ✗ | 52B | 256K | ai21labs/Jamba-v0.1 |
| Llama2 (chat) (Touvron et al., [2023](#bib.bib64)) | ✓ | 7B | 4K | meta-llama/Llama-2-7b-chat-hf |
| Llama2 (base) (Touvron et al., [2023](#bib.bib64)) | ✗ | 7B | 4K | meta-llama/Llama-2-7b-hf |
| Yi series (Young et al., [2024](#bib.bib78)) | ✓ | 6B/9B | 200K | 01-ai/Yi-6B-200K, 01-ai/Yi-9B-200K |
| LWM series (Liu et al., [2024a](#bib.bib39)) | ✓ | 7B | 128/256/512K | LargeWorldModel/LWM-Text-Chat-128/256/512K |
| LWM-base series (Liu et al., [2024a](#bib.bib39)) | ✗ | 7B | 32/128/256/512K | LargeWorldModel/LWM-Text-32/128/256/512K |
| Mamba (Gu & Dao, [2023](#bib.bib23)) | ✗ | 2.8B | 2K | state-spaces/mamba-2.8b-slimpj |
| RWKV (Peng et al., [2023](#bib.bib48)) | ✗ | 7B | 4K | RWKV/v5-Eagle-7B-HF |

Table 4: Information of evaluated and analyzed models in Ruler.

## Appendix B Task Configurations

Ruler is designed to be configurable to allow for diverse sequence lengths and task complexities. For each task, there arises combinatorially large number of configurations one can adopt. In the main text, we evaluate the models with 13 representative tasks spanning the four categories of Ruler. Our task selection process is described in the next appendix section.

* •

  Retrieval: In S-NIAH, we include the passkey retrieval (Mohtashami & Jaggi, [2023](#bib.bib44)) and the vanilla NIAH (Kamradt, [2023](#bib.bib32)), both use word-number as key-value and differ only by the background haystack. Additionally, we change the value type to UUID, for the purpose of testing model robustness at retrieving long strings from context. For MK-NIAH, we add three distractor needles into the haystack. We also include existing setups from previous works: line retrieval (Li et al., [2023a](#bib.bib36)) and key-value retrieval (Liu et al., [2024c](#bib.bib41)) with the haystack filled entirely with distractor needles.
  For MV-NIAH and MQ-NIAH, we test 4 values and queries respectively.
* •

  Multi-hop tracing: For VT, we insert 1 chain with 4 name-binding hops, totally 5 variable names need to be returned.
* •

  Aggregation: For CWE, in total 10 common words need to be returned, each appears 30 times whereas the uncommon words appear 3 times each.
  For FWE, we set α𝛼\alpha to 2.0 in Zeta distribution for sampling synthetic words.
* •

  QA: For QA, we augment SQuAD (Rajpurkar et al., [2018](#bib.bib52)) and HotpotQA (Yang et al., [2018](#bib.bib77)) to simulate long-context scenario. They are representative of single-hop and multi-hop question answering tasks respectively.

|  |  |  |  |
| --- | --- | --- | --- |
| Task | Configurations | | |
| Subtask-1 | Subtask-2 | Subtask-3 |
| |  | | --- | | Single | | NIAH | | type\_key = word  type\_value = number  type\_haystack = repeat  ∼similar-to\simpasskey retrieval | type\_key = word  type\_value = number  type\_haystack = essay  ∼similar-to\simvanilla NIAH | type\_key = word  type\_value = uuid  type\_haystack = essay |
| MK-NIAH | num\_keys = 4  type\_key = word  type\_value = number  type\_haystack = essay | num\_keys = full haystack  type\_key = word  type\_value = number  ∼similar-to\simline retrieval | num\_keys = full haystack  type\_key = uuid  type\_value = uuid  ∼similar-to\simKV retrieval |
| MV-NIAH | num\_values = 4, type\_key = word, type\_value = number, type\_haystack = essay | | |
| MQ-NIAH | num\_queries = 4, type\_key = word, type\_value = number, type\_haystack = essay | | |
| VT | num\_chains = 1, num\_hops = 4 | | |
| CWE | freq\_cw = 30, freq\_ucw = 3, num\_cw = 10 | | |
| FWE | α𝛼\alpha = 2.0 | | |
| QA | dataset = SQuAD | dataset = HotpotQA | |

Table 5: Our total 13 task configurations in Ruler.

## Appendix C Task Correlation Analysis

Ruler is designed under the assumption that tasks across different categories are able to reveal distinct model behaviors. We conduct a preliminary correlational study to confirm the validity of task categories and guide the selection of representative tasks. We evaluate eight models (not including GPT-4 and Command-R) at various context sizes across 18 task configurations. Each task can then be represented with a vector of model performance at various context sizes. The 18 task vectors are then clustered via agglomorative clustering algorithm, using correlation coefficient as the distance metric.
As shown in Figure [5](#A3.F5 "Figure 5 ‣ Appendix C Task Correlation Analysis ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?"), while certain tasks exhibit moderate correlations with others, tasks in each of the four categories (NIAH, VT, AG, QA) form cohesive clusters of their own without redundancy. We further eliminate a few tasks that correlate highly with other tasks within the same cluster, and finalize 13 tasks for later large scale evaluation.

Figure 5: Correlation heatmap among 18 tasks with diverse task configurations. We remove redundant tasks (in red) and only preserve 13 representative tasks in Ruler. (W: words; N: numbers; U: UUIDs; Full: entire haystack)

## Appendix D Prompt Templates

We decompose the input prompt template into the model template in Table [6](#A4.T6 "Table 6 ‣ Appendix D Prompt Templates ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?") and the task template in Table [7](#A4.T7 "Table 7 ‣ Appendix D Prompt Templates ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?") [8](#A4.T8 "Table 8 ‣ Appendix D Prompt Templates ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?") [9](#A4.T9 "Table 9 ‣ Appendix D Prompt Templates ‣ \faRulerRuler: What’s the Real Context Size of Your Long-Context Language Models?"). The model template is the model chat format while the task template combines instruction, context, and query. To prevent models from refusing to answer our questions, we append the input with an answer prefix to elicit model responses. For VT and CWE, we use one task sample as in-context demonstration.

| Model | Template |
| --- | --- |
| GPT-4 | {task\_template} Do not provide any explanation. Please directly give me the answer. {task\_answer\_prefix} |
| Yi/Base | {task\_template} {task\_answer\_prefix} |
| Command-R | ⟨⟨\langleBOS\_TOKEN⟩⟩\rangle  ⟨|\langle|START\_OF\_TURN\_TOKEN|⟩|\rangle  ⟨|\langle|USER\_TOKEN|⟩|\rangle{task\_template}  ⟨|\langle|END\_OF\_TURN\_TOKEN|⟩|\rangle  ⟨|\langle|START\_OF\_TURN\_TOKEN|⟩|\rangle  ⟨|\langle|CHATBOT\_TOKEN|⟩|\rangle{task\_answer\_prefix} |
| LWM | You are a helpful assistant. USER: {task\_template} ASSISTANT: {task\_answer\_prefix} |
| LongChat | A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user’s questions. USER: {task\_template} ASSISTANT: {task\_answer\_prefix} |
| ChatGLM | [ gMASK] sop⟨|\langle|user|⟩|\rangle  {task\_template}⟨|\langle|assistant|⟩|\rangle  {task\_answer\_prefix} |
| Others | [ INST]  {task\_template} [ /INST]  {task\_answer\_prefix} |

Table 6: Model chat templates. We append a task answer prefix in model response to prevent models from refusing to answer our questions. The addition of answer prefix does not break the models’ chat template.

|  |  |  |  |
| --- | --- | --- | --- |
| |  | | --- | | S-NIAH | | Subtask-1 | | Task Template:    Some special magic numbers are hidden within the following text. Make sure to memorize it. I will quiz you about the numbers afterwards.    The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.    …… One of the special magic numbers for {word} is: {number}. ……    What is the special magic number for {word} mentioned in the provided text?    Task Answer Prefix:    The special magic number for {word} mentioned in the provided text is |
| |  | | --- | | S-NIAH | | Subtask-2 | | Task Template:    Some special magic numbers are hidden within the following text. Make sure to memorize it. I will quiz you about the numbers afterwards.    Paul Graham Essays.    …… One of the special magic numbers for {word} is: {number}. ……    What is the special magic number for {word} mentioned in the provided text?    Task Answer Prefix:    The special magic number for {word} mentioned in the provided text is |
| |  | | --- | | S-NIAH | | Subtask-3 | | Task Template:    Some special magic words are hidden within the following text. Make sure to memorize it. I will quiz you about the words afterwards.    Paul Graham Essays.    …… One of the special magic words for {word} is: {word}. ……    What is the special magic word for {word} mentioned in the provided text?    Task Answer Prefix:    The special magic word for {word} mentioned in the provided text is |
| |  | | --- | | MK-NIAH | | Subtask-1 | | Task Template:    Some special magic numbers are hidden within the following text. Make sure to memorize it. I will quiz you about the numbers afterwards.    Paul Graham Essays.    …… One of the special magic numbers for {word-1} is: {number-1}. ……    …… One of the special magic numbers for {word-2} is: {number-2}. ……    …… One of the special magic numbers for {word-3} is: {number-3}. ……    …… One of the special magic numbers for {word-4} is: {number-4}. ……    What is the special magic number for {word-4} mentioned in the provided text?    Task Answer Prefix:    The special magic number for {word-4} mentioned in the provided text is |
| |  | | --- | | MK-NIAH | | Subtask-2 | | Task Template:    Some special magic numbers are hidden within the following text. Make sure to memorize it. I will quiz you about the numbers afterwards.    One of the special magic numbers for {word-1} is: {number-1}.    One of the special magic numbers for {word-2} is: {number-2}.    …… One of the special magic numbers for {word-x} is: {number-x}. ……    One of the special magic numbers for {word-n-1} is: {number-n-1}.    One of the special magic numbers for {word-n} is: {number-n}.    What is the special magic number for {word-x} mentioned in the provided text?    Task Answer Prefix:    The special magic number for {word-x} mentioned in the provided text is |
| |  | | --- | | MK-NIAH | | Subtask-3 | | Task Template:    Some special magic uuids are hidden within the following text. Make sure to memorize it. I will quiz you about the uuids afterwards.    One of the special magic uuids for {uuid-1} is: {uuid-1}.    One of the special magic uuids for {uuid-2} is: {uuid-2}.    …… One of the special magic uuids for {uuid-x} is: {uuid-x}. ……    One of the special magic uuids for {uuid-n-1} is: {uuid-n-1}.    One of the special magic uuids for {uuid-n} is: {uuid-n}.    What is the special magic number for {uuid-x} mentioned in the provided text?    Task Answer Prefix:    The special magic number for {uuid-x} mentioned in the provided text is |

Table 7: S-NIAH and MK-NIAH templates.



|  |  |  |
| --- | --- | --- |
| |  | | --- | | MV-NIAH | | Task Template:    Some special magic numbers are hidden within the following text. Make sure to memorize it. I will quiz you about the numbers afterwards.    Paul Graham Essays.    …… One of the special magic numbers for {word} is: {number-1}. ……    …… One of the special magic numbers for {word} is: {number-2}. ……    …… One of the special magic numbers for {word} is: {number-3}. ……    …… One of the special magic numbers for {word} is: {number-4}. ……    What are all the special magic numbers for {word} mentioned in the provided text?    Task Answer Prefix:    The special magic numbers for {word} mentioned in the provided text are |
| |  | | --- | | MQ-NIAH | | Task Template:    Some special magic numbers are hidden within the following text. Make sure to memorize it. I will quiz you about the numbers afterwards.    Paul Graham Essays.    …… One of the special magic numbers for {word-1} is: {number-1}. ……    …… One of the special magic numbers for {word-2} is: {number-2}. ……    …… One of the special magic numbers for {word-3} is: {number-3}. ……    …… One of the special magic numbers for {word-4} is: {number-4}. ……    What are all the special magic numbers for {word-1}, {word-2}, {word-3}, and {word-4} mentioned in the provided text?    Task Answer Prefix:    The special magic numbers for {word-1}, {word-2}, {word-3}, and {word-4} mentioned in the provided text are |
| |  | | --- | | VT | | Task Template:    {one task example}    Memorize and track the chain(s) of variable assignment hidden in the following text.    The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. …… VAR {X1} = {number} ……    …… VAR {X2} = {X1} ……    …… VAR {X3} = {X2} ……    …… VAR {X4} = {X3} ……    …… VAR {X5} = {X4} ……    Question: Find all variables that are assigned the value {number} in the text above.    Task Answer Prefix:    Answer: According to the chain(s) of variable assignment in the text above, 5 variables are assigned the value {number}, they are: |
| |  | | --- | | CWE | | Task Template:    {one task example}    Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.    1. word-a 2. word-b 3. word-c 4. word-a 5. word-d 6. word-a 7. word-e 8. word-f ……    Question: What are the 10 most common words in the above list?    Task Answer Prefix:    Answer: The top 10 words that appear most often in the list are: |
| |  | | --- | | FWE | | Task Template:    Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words. … … word-a … word-b … … … word-c … word-a … word-d word-e … word-a … … word-f … … … … word-g … word-h … word-a … word-i ……    Question: Do not provide any explanation. Please ignore the dots ’….’. What are the three most frequently appeared words in the above coded text?    Task Answer Prefix:    Answer: According to the coded text above, the three most frequently appeared words are: |

Table 8: MV-NIAH, MQ-NIAH, VT, CWE, and FWE templates.



|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| |  | | --- | | Single | | Hop | | QA | | Task Template:    Answer the question based on the given documents. Only give me the answer and do not output any other words.    The following are given documents.    Document 1:    {document-1}    ……    Document x:    {document-x}    ……    Document n:    {document-n}    Answer the question based on the given documents. Only give me the answer and do not output any other words.    Question: question    Task Answer Prefix:    Answer: |
| |  | | --- | | Multi | | Hop | | QA | | Task Template:    Answer the question based on the given documents. Only give me the answer and do not output any other words.    The following are given documents.    Document 1:    {document-1}    ……    Document x:    {document-x}    ……    Document y:    {document-y}    ……    Document n:    {document-n}    Answer the question based on the given documents. Only give me the answer and do not output any other words.    Question: question    Task Answer Prefix:    Answer: |

Table 9: QA templates.

## Appendix E Passkey Retrieval and Vanilla NIAH Results

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Models | |  | | --- | | Claimed | | Length | | 4k | 8k | 16k | 32k | 64k | 128k | |  | | --- | | Avg. | |
| GPT-4 | 128k | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Command-R (35B) | 128k | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Yi (34B) | 200k | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Mixtral (8x7B) | 32k | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 97.0 | 99.5 |
| Mistral (7B) | 32k | 100.0 | 100.0 | 100.0 | 100.0 | 99.6 | 69.6 | 94.9 |
| ChatGLM (6B) | 128k | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| LWM (7B) | 1M | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Together (7B) | 32k | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 66.7 |
| LongChat (7B) | 32k | 100.0 | 100.0 | 100.0 | 99.4 | 0.0 | 0.0 | 66.6 |
| LongAlpaca (13B) | 32k | 88.2 | 88.6 | 86.4 | 82.4 | 0.0 | 0.0 | 57.6 |
| Mixtral-base (8x7B) | 32k | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 46.8 | 91.1 |
| Mistral-base (7B) | 32k | 100.0 | 100.0 | 100.0 | 100.0 | 99.6 | 70.8 | 95.1 |
| Jamba-base (52B) | 256k | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| LWM-base (7B) | 1M | 99.8 | 100.0 | 99.6 | 99.6 | 98.2 | 96.0 | 98.9 |
| LongLoRA-base (7B) | 100k | 99.6 | 99.4 | 99.0 | 99.4 | 99.4 | 0.0 | 82.8 |
| Yarn-base (7B) | 128k | 100.0 | 100.0 | 99.0 | 100.0 | 99.2 | 39.6 | 89.6 |
| Together-base (7B) | 32k | 100.0 | 100.0 | 99.8 | 100.0 | 0.0 | 0.0 | 66.6 |

Table 10: Performance of selected aligned and base models across length 4k to 128k in passkey retrieval of Ruler. Almost all models have perfect score at their claimed length.



|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Models | |  | | --- | | Claimed | | Length | | 4k | 8k | 16k | 32k | 64k | 128k | |  | | --- | | Avg. | |
| GPT-4 | 128k | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Command-R (35B) | 128k | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 98.0 | 99.7 |
| Yi (34B) | 200k | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Mixtral (8x7B) | 32k | 100.0 | 100.0 | 100.0 | 100.0 | 93.2 | 43.8 | 89.5 |
| Mistral (7B) | 32k | 100.0 | 100.0 | 100.0 | 97.0 | 70.0 | 7.4 | 79.1 |
| ChatGLM (6B) | 128k | 100.0 | 100.0 | 99.0 | 99.6 | 90.8 | 87.0 | 96.1 |
| LWM (7B) | 1M | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Together (7B) | 32k | 100.0 | 100.0 | 100.0 | 99.8 | 0.0 | 0.0 | 66.7 |
| LongChat (7B) | 32k | 100.0 | 100.0 | 97.6 | 98.4 | 0.0 | 0.0 | 66.0 |
| LongAlpaca (13B) | 32k | 90.2 | 90.2 | 88.4 | 83.4 | 0.0 | 0.0 | 58.7 |
| Mixtral-base (8x7B) | 32k | 100.0 | 100.0 | 100.0 | 100.0 | 85.2 | 34.8 | 86.7 |
| Mistral-base (7B) | 32k | 100.0 | 100.0 | 100.0 | 100.0 | 94.8 | 0.4 | 82.5 |
| Jamba-base (52B) | 256k | 100.0 | 100.0 | 98.8 | 99.8 | 99.8 | 86.4 | 97.5 |
| LWM-base (7B) | 1M | 100.0 | 99.4 | 97.8 | 98.6 | 98.2 | 98.6 | 98.8 |
| LongLoRA-base (7B) | 100k | 99.8 | 100.0 | 100.0 | 99.8 | 100.0 | 0.0 | 83.3 |
| Yarn-base (7B) | 128k | 97.4 | 97.8 | 91.4 | 85.4 | 86.6 | 20.0 | 79.8 |
| Together-base (7B) | 32k | 100.0 | 100.0 | 100.0 | 99.8 | 0.0 | 0.0 | 66.6 |

Table 11: Performance of selected aligned and base models across length 4k to 128k in vanilla NIAH of Ruler. Almost all models have perfect score at their claimed length.

## Appendix F Additional Results

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Models | |  | | --- | | Claimed | | Length | | |  | | --- | | Effective | | Length | | 4k | 8k | 16k | 32k | 64k | 128k | |  | | --- | | Avg. | | |  | | --- | | wAvg. | | (inc) | | |  | | --- | | wAvg. | | (dec) | |
| Llama2-7B (base) | 4k | - | 79.4 | | | |  |  |  |  |  |
| Mixtral-base (8x7B) | 32k | 32k | 91.8 | 91.0 | 89.5 | 85.8 | 66.9 | 29.0 | 75.7 | 66.4(1st) | 85.0(1st) |
| Mistral-base (7B) | 32k | 16k | 91.6 | 89.8 | 86.3 | 77.2 | 52.3 | 8.0 | 67.5 | 54.7(4th) | 80.4(2nd) |
| Jamba-base (52B) | 256k | 4k | 81.2 | 75.4 | 68.8 | 65.3 | 61.0 | 51.4 | 67.2 | 62.5(3rd) | 71.8(4th) |
| LWM-base (7B) | 1M | <4k | 77.5 | 74.0 | 69.6 | 64.6 | 61.3 | 59.0 | 67.7 | 64.4(2nd) | 70.9(5th) |
| LongLoRA-base (7B) | 100k | 8k | 81.9 | 80.4 | 75.6 | 65.1 | 60.8 | 0.0 | 60.6 | 49.2(5th) | 72.0(3rd) |
| Yarn-base (7B) | 128k | <4k | 77.3 | 67.5 | 59.0 | 47.3 | 38.6 | 13.9 | 50.6 | 40.7(6th) | 60.5(7th) |
| Together-base (7B) | 32k | 4k | 84.6 | 78.7 | 68.3 | 57.9 | 0.0 | 0.0 | 48.2 | 32.3(7th) | 64.2(6th) |

Table 12: Performance of selected base models across length 4k to 128k by averaging 13 task scores in Ruler.



|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Models | |  | | --- | | Claimed | | Length | | |  | | --- | | Effective | | Length | | 4k | 8k | 16k | 32k | 64k | 128k | |  | | --- | | Avg. | | |  | | --- | | wAvg. | | (inc) | | |  | | --- | | wAvg. | | (dec) | |
| Llama2-7B (chat) | 4k | - | 96.9 | | | |  |  |  |  |  |
| GPT-4 | 128k | 32k | 99.9 | 99.9 | 98.7 | 98.3 | 90.9 | 84.8 | 95.4 | 92.9(3rd) | 97.9(2nd) |
| Command-R (35B) | 128k | 64k | 99.5 | 99.3 | 99.3 | 98.9 | 97.4 | 89.6 | 97.3 | 96.0(1st) | 98.6(1st) |
| Yi (34B) | 200k | 16k | 98.1 | 96.9 | 97.4 | 95.1 | 93.0 | 90.2 | 95.1 | 93.8(2nd) | 96.4(3rd) |
| Mixtral (8x7B) | 32k | 16k | 99.4 | 98.3 | 98.8 | 94.3 | 73.8 | 42.6 | 84.5 | 75.9(5th) | 93.1(4th) |
| Mistral (7B) | 32k | 4k | 98.1 | 96.2 | 94.3 | 85.5 | 51.1 | 10.7 | 72.6 | 58.8(7th) | 86.5(7th) |
| ChatGLM (6B) | 128k | 4k | 97.5 | 95.9 | 91.9 | 83.6 | 67.6 | 50.9 | 81.2 | 73.5(6th) | 89.0(5th) |
| LWM (7B) | 1M | <4k | 92.5 | 92.1 | 87.6 | 83.7 | 84.1 | 83.4 | 87.2 | 85.5(4th) | 89.0(6th) |
| Together (7B) | 32k | <4k | 96.2 | 89.9 | 82.3 | 80.2 | 0.0 | 0.0 | 58.1 | 40.2(8th) | 76.0(8th) |
| LongChat (7B) | 32k | <4k | 93.3 | 92.2 | 81.1 | 67.3 | 0.0 | 0.0 | 55.7 | 37.6(9th) | 73.7(9th) |
| LongAlpaca (13B) | 32k | <4k | 74.9 | 72.2 | 70.8 | 53.2 | 0.0 | 0.0 | 45.2 | 30.7(10th) | 59.7(10th) |
| Llama2-7B (base) | 4k | - | 90.9 | | | |  |  |  |  |  |
| Mixtral-base (8x7B) | 32k | 32k | 99.9 | 99.7 | 98.4 | 94.8 | 72.1 | 29.1 | 82.3 | 71.8(2nd) | 92.8(1st) |
| Mistral-base (7B) | 32k | 16k | 99.3 | 97.5 | 95.7 | 89.8 | 56.8 | 10.2 | 74.9 | 61.2(4th) | 88.6(2nd) |
| Jamba-base (52B) | 256k | <4k | 86.4 | 80.5 | 73.7 | 72.3 | 68.1 | 56.9 | 73.0 | 68.5(3th) | 77.4(5th) |
| LWM-base (7B) | 1M | <4k | 88.5 | 87.7 | 84.5 | 79.6 | 76.1 | 74.2 | 81.8 | 79.1(1st) | 84.4(4th) |
| LongLoRA-base (7B) | 100k | 16k | 95.3 | 95.6 | 92.7 | 81.5 | 76.2 | 0.0 | 73.5 | 60.6(5th) | 86.5(3rd) |
| Yarn-base (7B) | 128k | <4k | 89.9 | 86.1 | 78.4 | 59.0 | 49.5 | 17.5 | 63.4 | 51.7(6th) | 75.1(7th) |
| Together-base (7B) | 32k | 8k | 95.4 | 91.5 | 86.1 | 75.1 | 0.0 | 0.0 | 58.0 | 39.9(7th) | 76.2(6th) |

Table 13: Performance of selected aligned and base models across length 4k to 128k by averaging 8 task scores in Retrieval (NIAH) of RULER.



|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Models | |  | | --- | | Claimed | | Length | | |  | | --- | | Effective | | Length | | 4k | 8k | 16k | 32k | 64k | 128k | |  | | --- | | Avg. | | |  | | --- | | wAvg. | | (inc) | | |  | | --- | | wAvg. | | (dec) | |
| Llama2-7B (chat) | 4k | - | 89.7 | | | |  |  |  |  |  |
| GPT-4 | 128k | 128k | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 99.6 | 99.9 | 99.9(1st) | 100.0(1st) |
| Command-R (35B) | 128k | 128k | 99.9 | 100.0 | 99.8 | 99.5 | 99.3 | 89.8 | 98.1 | 96.8(2nd) | 99.3(2nd) |
| Yi (34B) | 200k | 64k | 99.8 | 99.2 | 98.8 | 94.5 | 92.5 | 76.8 | 93.6 | 90.3(3rd) | 96.9(3rd) |
| Mixtral (8x7B) | 32k | 32k | 99.9 | 99.7 | 98.4 | 93.6 | 86.9 | 64.2 | 90.5 | 85.2(4th) | 95.7(4th) |
| Mistral (7B) | 32k | 16k | 98.9 | 96.0 | 92.2 | 85.0 | 74.5 | 0.0 | 74.4 | 60.9(5th) | 87.9(5th) |
| ChatGLM (6B) | 128k | <4k | 84.0 | 81.2 | 78.0 | 66.0 | 38.4 | 13.0 | 60.1 | 48.3(6th) | 71.9(7th) |
| LWM (7B) | 1M | <4k | 84.4 | 80.1 | 67.2 | 52.2 | 45.9 | 15.2 | 57.5 | 46.5(7th) | 68.6(8th) |
| Together (7B) | 32k | <4k | 89.2 | 88.8 | 48.3 | 16.6 | 0.0 | 0.0 | 40.5 | 22.8(9th) | 58.2(9th) |
| LongChat (7B) | 32k | 8k | 97.6 | 93.5 | 83.4 | 62.4 | 0.0 | 0.0 | 56.2 | 37.4(8th) | 75.0(6th) |
| LongAlpaca (13B) | 32k | <4k | 8.5 | 2.1 | 18.2 | 17.0 | 0.0 | 0.0 | 7.6 | 6.5(10th) | 8.8(10th) |
| Llama2-7B (base) | 4k | - | 58.8 | | | |  |  |  |  |  |
| Mixtral-base (8x7B) | 32k | 64k | 100.0 | 99.9 | 100.0 | 98.4 | 87.3 | 43.3 | 88.1 | 80.5(2nd) | 95.8(1st) |
| Mistral-base (7B) | 32k | 64k | 99.0 | 98.4 | 96.5 | 89.1 | 86.1 | 0.0 | 78.2 | 65.4(4th) | 91.0(2nd) |
| Jamba-base (52B) | 256k | 128k | 87.5 | 87.6 | 86.2 | 88.1 | 86.0 | 77.8 | 85.5 | 84.3(1st) | 86.7(3rd) |
| LWM-base (7B) | 1M | 128k | 80.2 | 82.7 | 79.3 | 76.4 | 70.7 | 66.1 | 75.9 | 73.3(3th) | 78.5(4th) |
| LongLoRA-base (7B) | 100k | 64k | 92.5 | 87.4 | 73.1 | 56.0 | 69.2 | 0.0 | 63.0 | 50.3(5th) | 75.8(5th) |
| Yarn-base (7B) | 128k | 4k | 84.6 | 43.6 | 24.8 | 43.0 | 20.9 | 0.0 | 36.1 | 24.9(7th) | 47.4(7th) |
| Together-base (7B) | 32k | 16k | 95.0 | 90.6 | 69.6 | 43.2 | 0.0 | 0.0 | 49.7 | 31.3(6th) | 68.1(6th) |

Table 14: Performance of selected aligned and base models across length 4k to 128k in Multi-hop tracing (VT) of RULER.

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Models | |  | | --- | | Claimed | | Length | | |  | | --- | | Effective | | Length | | 4k | 8k | 16k | 32k | 64k | 128k | |  | | --- | | Avg. | | |  | | --- | | wAvg. | | (inc) | | |  | | --- | | wAvg. | | (dec) | |
| Llama2-7B-chat | 4k | - | 84.8 | | | |  |  |  |  |  |
| GPT-4 | 128k | 64k | 99.0 | 98.3 | 98.0 | 95.0 | 90.1 | 79.7 | 93.4 | 90.4(1st) | 96.3(1st) |
| Command-R (35B) | 128k | 16k | 93.7 | 93.8 | 89.3 | 73.7 | 54.7 | 42.0 | 74.5 | 65.2(4th) | 83.8(2nd) |
| Yi (34B) | 200k | 16k | 91.4 | 90.9 | 86.2 | 75.3 | 58.5 | 43.4 | 74.3 | 66.0(3rd) | 82.6(3rd) |
| Mixtral (8x7B) | 32k | 16k | 91.7 | 82.7 | 85.7 | 69.9 | 80.9 | 52.7 | 77.3 | 72.1(2nd) | 82.4(4th) |
| Mistral (7B) | 32k | 8k | 94.3 | 90.4 | 77.4 | 48.5 | 42.4 | 33.7 | 64.4 | 53.1(5th) | 75.8(5th) |
| ChatGLM (6B) | 128k | <4k | 79.6 | 64.3 | 53.5 | 43.2 | 39.6 | 35.5 | 52.6 | 45.4(6th) | 59.9(6th) |
| LWM (7B) | 1M | <4k | 61.3 | 43.6 | 38.3 | 32.8 | 29.1 | 29.1 | 39.0 | 34.0(7th) | 44.0(9th) |
| Together (7B) | 32k | <4k | 82.3 | 64.5 | 43.3 | 34.8 | 0.0 | 0.0 | 37.5 | 22.9(9th) | 52.1(7th) |
| LongChat (7B) | 32k | <4k | 74.3 | 50.7 | 46.7 | 51.1 | 0.0 | 0.0 | 37.1 | 24.8(8th) | 49.5(8th) |
| LongAlpaca (13B) | 32k | <4k | 33.0 | 27.0 | 26.0 | 23.2 | 0.0 | 0.0 | 18.2 | 12.3(10th) | 24.1(10th) |
| Llama2-7B (base) | 4k | - | 73.1 | | | |  |  |  |  |  |
| Mixtral-base (8x7B) | 32k | 32k | 96.5 | 94.8 | 93.1 | 87.8 | 68.6 | 24.3 | 77.5 | 66.9(1st) | 88.1(1st) |
| Mistral-base (7B) | 32k | 16k | 94.8 | 93.1 | 81.6 | 53.3 | 36.7 | 9.2 | 61.4 | 46.5(2nd) | 76.3(2nd) |
| Jamba-base (52B) | 256k | 4k | 75.9 | 63.5 | 51.7 | 38.5 | 33.3 | 28.0 | 48.5 | 40.3(3rd) | 56.6(3rd) |
| LWM-base (7B) | 1M | <4k | 67.1 | 48.4 | 36.0 | 26.3 | 21.5 | 18.7 | 36.3 | 28.4(5th) | 44.2(5th) |
| LongLoRA-base (7B) | 100k | <4k | 70.3 | 64.4 | 50.7 | 39.9 | 29.4 | 0.0 | 42.4 | 31.3(4th) | 53.6(4th) |
| Yarn-base (7B) | 128k | <4k | 70.6 | 49.2 | 28.9 | 20.5 | 17.0 | 2.1 | 31.4 | 20.7(6th) | 42.0(6th) |
| Together-base (7B) | 32k | <4k | 69.1 | 53.0 | 19.9 | 20.6 | 0.0 | 0.0 | 27.1 | 15.1(7th) | 39.1(7th) |

Table 15: Performance of selected aligned and base models across length 4k to 128k by averaging 2 task scores in Aggregation (CWE/FWE) of RULER.



|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Models | |  | | --- | | Claimed | | Length | | |  | | --- | | Effective | | Length | | 4k | 8k | 16k | 32k | 64k | 128k | |  | | --- | | Avg. | | |  | | --- | | wAvg. | | (inc) | | |  | | --- | | wAvg. | | (dec) | |
| Llama2-7B (chat) | 4k | - | 49.7 | | | |  |  |  |  |  |
| GPT-4 | 128k | 128k | 79.0 | 78.0 | 76.0 | 68.0 | 61.6 | 59.0 | 70.3 | 66.5(1st) | 74.0(1st) |
| Command-R (34B) | 128k | 64k | 67.9 | 65.4 | 63.9 | 62.6 | 58.1 | 48.4 | 61.1 | 58.2(3rd) | 63.9(4th) |
| Yi (34B) | 200k | 128k | 72.7 | 71.5 | 68.4 | 66.2 | 64.1 | 59.9 | 67.1 | 65.0(2nd) | 69.2(2nd) |
| Mixtral (8x7B) | 32k | 64k | 77.6 | 73.0 | 71.2 | 64.5 | 51.2 | 34.1 | 61.9 | 55.0(5th) | 68.8(3rd) |
| Mistral (7B) | 32k | 32k | 72.4 | 70.0 | 65.7 | 57.6 | 34.4 | 13.3 | 52.2 | 42.5(6th) | 62.0(5th) |
| ChatGLM (6B) | 128k | 16k | 59.1 | 53.5 | 50.9 | 43.5 | 34.6 | 27.3 | 44.8 | 39.5(7th) | 50.1(7th) |
| LWM (7B) | 1M | 128k | 61.2 | 57.8 | 56.7 | 55.4 | 54.7 | 52.6 | 56.4 | 55.1(4th) | 57.7(6th) |
| Together (7B) | 32k | 16k | 61.1 | 58.3 | 54.2 | 45.6 | 0.0 | 0.0 | 36.5 | 24.9(8th) | 48.2(8th) |
| LongChat (7B) | 32k | 8k | 54.5 | 53.6 | 47.6 | 34.0 | 0.0 | 0.0 | 31.6 | 21.0(10th) | 42.3(10th) |
| LongAlpaca (13B) | 32k | 16k | 57.2 | 53.5 | 49.7 | 39.0 | 0.0 | 0.0 | 33.2 | 22.3(9th) | 44.1(9th) |
| Llama2-7B (base) | 4k | - | 48.6 | | | |  |  |  |  |  |
| Mixtral-base (8x7B) | 32k | 4k | 50.8 | 47.7 | 45.3 | 41.3 | 34.4 | 26.4 | 41.0 | 37.0(3rd) | 44.9(3rd) |
| Mistral-base (7B) | 32k | 8k | 53.5 | 51.0 | 48.4 | 44.7 | 32.8 | 2.2 | 38.8 | 31.3(4th) | 46.3(2nd) |
| Jamba-base (52B) | 256k | 32k | 62.7 | 60.6 | 57.9 | 52.6 | 47.5 | 39.6 | 53.5 | 49.7(1st) | 57.3(1st) |
| LWM-base (7B) | 1M | <4k | 42.7 | 40.2 | 38.7 | 37.1 | 37.3 | 34.6 | 38.4 | 37.2(2nd) | 39.6(4th) |
| LongLoRA-base (7B) | 100k | <4k | 34.5 | 32.1 | 33.6 | 29.4 | 26.1 | 0.0 | 26.0 | 21.3(6th) | 30.6(6th) |
| Yarn-base (7B) | 128k | <4k | 29.7 | 23.5 | 28.6 | 29.7 | 25.5 | 18.1 | 25.9 | 24.6(5th) | 27.1(7th) |
| Together-base (7B) | 32k | 4k | 52.0 | 47.5 | 44.6 | 33.6 | 0.0 | 0.0 | 29.6 | 19.8(7th) | 39.5(5th) |

Table 16: Performance of selected aligned and base models across length 4k to 128k by averaging 2 task scores in Question Answering of RULER.
