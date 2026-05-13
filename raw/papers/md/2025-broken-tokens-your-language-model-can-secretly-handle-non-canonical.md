---
arxiv: '2506.19004'
authors:
- Brian Siyuan Zheng
- Alisa Liu
- Orevaoghene Ahia
- Jonathan Hayase
- Yejin Choi
- Noah A. Smith
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Broken Tokens? Your Language Model can Secretly Handle Non-Canonical Tokenizations
url: https://arxiv.org/abs/2506.19004
year: 2025
---

[2506.19004] Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations














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



# Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations

Brian Siyuan Zheng♡ Alisa Liu♡ Orevaoghene Ahia♡ 
  
Jonathan Hayase♡ Yejin Choi♠ Noah A. Smith♡♣ 
  
♡University of Washington ♣Allen Institute for AI  ♠Stanford University
  
zhengbr@cs.washington.edu

###### Abstract

Modern tokenizers employ deterministic algorithms to map text into a single “canonical” token sequence, yet the same string can be encoded as many non-canonical tokenizations using the tokenizer vocabulary.
In this work, we investigate the robustness of LMs to text encoded with non-canonical tokenizations entirely unseen during training.
Surprisingly, when evaluated across 20 benchmarks, we find that instruction-tuned models retain up to 93.4% of their original performance when given a randomly sampled tokenization, and 90.8% with character-level tokenization.
We see that overall stronger models tend to be more robust, and robustness diminishes as the tokenization departs farther from the canonical form.
Motivated by these results, we then identify settings where non-canonical tokenization schemes can improve performance, finding that character‑level segmentation improves string manipulation and code understanding tasks by up to +14%, and right‑aligned digit grouping enhances large‑number arithmetic by +33%.
Finally, we investigate the source of this robustness, finding that it arises in the instruction-tuning phase.
We show that while both base and post-trained models grasp the semantics of non-canonical tokenizations (perceiving them as containing misspellings), base models try to mimic the imagined mistakes and degenerate into nonsensical output, while post-trained models are committed to fluent responses.
Overall, our findings suggest that models are less tied to their tokenizer than previously believed, and demonstrate the promise of intervening on tokenization at inference time to boost performance.111Code is available at <https://github.com/Brianzhengca/Tokenizer-Robustness>.

## 1 Introduction

Tokenizers segment text into a sequence of discrete tokens in the language model’s (LM) vocabulary.
Most of today’s LMs use deterministic subword tokenization, which produces a single canonical token sequence for a given piece of text, and further, for each whitespace-delimited word.
One commonly discussed limitation of this approach is that, by mapping byte strings to symbolic token IDs, the orthographic makeup of tokens is obscured to the LM [[46](#bib.bib46), [17](#bib.bib17)].
This can be especially harmful for LM understanding of numbers [[42](#bib.bib42), [57](#bib.bib57), [54](#bib.bib54)] and morphologically rich languages [[2](#bib.bib2), [24](#bib.bib24)], and has motivated efforts to model text directly at the byte level [[12](#bib.bib12), [61](#bib.bib61), [60](#bib.bib60), [56](#bib.bib56), [63](#bib.bib63), [41](#bib.bib41), [44](#bib.bib44), [1](#bib.bib1), [35](#bib.bib35)].

To shed more light on this perceived limitation, in this work we study whether LMs can adapt *at inference time*, without any additional training, to a different tokenization scheme than the one they were trained with.
While the tokenizer deterministically outputs a canonical tokenization of any text into tokens (usually by applying an ordered list of merge rules), non-canonical tokenizations of the same text using the same vocabulary are generally possible (see example in [Figure 1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations")).
Here, we evaluate how LMs trained with deterministic tokenizers behave when given non-canonical tokenizations of text.
Surprisingly, we find that instruction-tuned LMs across many model families are extremely robust to non-canonical tokenizations ([§2](#S2 "2 Language Models are Robust to Non-Canonical Tokenizations ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations")).
For example, when evaluated across 20 benchmarks, Qwen-2.5-7B-Instruct retains 93.4% of its original performance when presented with a random non-canonical tokenization, and 90.8% when presented with character-level tokens (see [Figure 1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations")).
Thus, far from not understanding the makeup of their tokens, LMs are able to compose token sequences in entirely new ways at inference time [[30](#bib.bib30)].

![Refer to caption](/html/2506.19004/assets/Images/figure1.png)

![Refer to caption](/html/2506.19004/assets/Images/figure1.jpg)

Figure 1: Left: An example of how Llama-3.1-8B-Instruct responds when given canonically tokenized input, versus a random tokenization and character-level tokenization.
The responses are surprisingly similar, demonstrating their ability to handle non-canonical tokenizations.
Moreover, LMs generally respond with correctly tokenized output regardless of the tokenization scheme used for the context.
Right: Performance of Qwen-2.5-7B-Instruct across various benchmarks and tokenization schemes. Much of the original performance is preserved when given non-canonical tokenizations.

This leads to an intriguing possibility: if LMs can process non-canonical tokenizations, can we use different tokenization schemes at inference time to improve performance?
For instance, prior work has found that better segmentation of large numbers can improve accuracy on arithmetic [[54](#bib.bib54), [50](#bib.bib50)].
Indeed, we identify several settings where non-canonical tokenization schemes improve performance for Llama-3.1-Instruct ([§3](#S3 "3 Can non-canonical tokenizations improve model performance? ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations")): character-level tokenization brings up to +14% improvement on string manipulation and code understanding tasks, perhaps by granting LMs more direct access to orthographic cues.
Meanwhile, right-aligned digit groups, which provide a consistent grouping of digits by powers of a thousand, improves arithmetic on large numbers by +33%.
These performance gains are achieved without any model finetuning, pointing to the promise of tokenization as a means of inference-time control.

Finally, we investigate the origins of model robustness to non-canonical tokenizations ([§4](#S4 "4 Investigating the Source of Robustness ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations")).
Across multiple model families, we find that pretrained-only LMs consistently fail to produce fluent continuations given non-canonically tokenized context.
By studying models at different stages of post-training, we identify that robustness arises during the supervised instruction-tuning (SFT) phase ([§4.1](#S4.SS1 "4.1 When does robustness appear in model training? ‣ 4 Investigating the Source of Robustness ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations")).
We then ablate differences between pretraining and SFT procedures and find that the separation of the instruction and response as distinct turns of conversation is key ([§4.2](#S4.SS2 "4.2 Why do instruction-tuned models become robust? ‣ 4 Investigating the Source of Robustness ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations")).
From here, we provide evidence for a plausible explanation: while both base and post-trained models grasp the semantics of non-canonical tokenizations, they also perceive them as containing misspellings ([§4.2](#S4.SS2 "4.2 Why do instruction-tuned models become robust? ‣ 4 Investigating the Source of Robustness ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations")).
Base models attempt to mimic the imagined mistakes and degenerate into nonsense, whereas post-trained models are not bound by the style of the instruction and thus able to produce fluent responses.

Overall, despite being trained with deterministic tokenization, instruction-tuned LMs readily accommodate new tokenizations at inference time, suggesting that LMs are less constrained by their tokenizer than previously believed [[40](#bib.bib40)]. Moreover, in settings where different representations of text are beneficial, we can intervene on tokenization at inference time for performance gains.
We hope our work sheds new light on the discussion of strengths and limitations of tokenization, and points to the possibility of dynamically finding the optimal representation of text after pretraining.

Table 1: 
Evaluated across many benchmarks, models are surprisingly robust to non-canonical tokenizations of the context.
We show the absolute drop in performance when given a randomly sampled non-canonical tokenization (Rand Δ\Delta) and character-level tokenization (Char Δ\Delta), relative to the canonical (Canon) tokenization. We also summarize the model’s ability to retain performance across benchmarks and tokenization strategies (bottom).

|  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Qwen-2.5-7B-Instruct | | | Llama-3.1-8B-Instruct | | | Olmo-2-7B-Instruct | | |
| Benchmark | Canon | Rand Δ\Delta | Char Δ\Delta | Canon | Rand Δ\Delta | Char Δ\Delta | Canon | Rand Δ\Delta | Char Δ\Delta |
| Multiple choice (MC) | | | | | | | | | |
| ARC-C | 86.4 | –1.80 | –2.60 | 76.2 | –14.10 | –22.40 | 77.0 | –37.40 | –44.60 |
| ARC-E | 94.4 | –2.20 | –3.60 | 91.3 | –12.60 | –21.50 | 85.4 | –37.00 | –49.00 |
| COPA | 97.0 | –1.80 | –7.40 | 97.2 | –9.60 | –14.80 | 93.8 | –21.40 | –33.60 |
| Winogrande | 46.0 | –4.60 | –3.00 | 59.6 | +2.00 | –5.00 | 58.6 | –7.80 | –7.00 |
| Winograd | 72.4 | –16.80 | –26.60 | 74.4 | –9.40 | –13.00 | 72.4 | –8.60 | –19.00 |
| CSQA | 85.6 | –8.60 | –9.20 | 77.6 | –11.40 | –20.00 | 75.4 | –31.60 | –40.00 |
| OpenbookQA | 87.2 | –7.00 | –7.80 | 82.0 | –13.80 | –20.20 | 76.2 | –30.80 | –40.80 |
| PIQA | 87.0 | –6.00 | –14.00 | 84.0 | –12.60 | –18.40 | 78.2 | –17.40 | –25.00 |
| MMLU | 71.7 | –3.70 | –7.30 | 68.2 | –11.60 | –24.00 | 59.5 | –16.30 | –29.10 |
| BoolQ | 84.8 | –5.00 | –4.80 | 86.2 | –19.20 | –17.20 | 71.0 | –4.00 | –9.20 |
| HellaSwag | 79.6 | –6.20 | –7.40 | 68.6 | –14.20 | –23.40 | 68.0 | –26.80 | –39.80 |
| Short answer (SA) | | | | | | | | | |
| WikidataQA | 81.2 | –7.60 | –9.00 | 78.6 | –12.40 | –18.00 | 73.2 | –28.80 | –32.20 |
| TOFU | 77.8 | +0.00 | –1.70 | 82.1 | +1.70 | +0.80 | 82.9 | –12.80 | –23.90 |
| TriviaQA | 70.0 | –1.00 | –3.80 | 76.6 | –9.80 | –13.60 | 70.0 | –22.20 | –34.80 |
| JeopardyQA | 49.4 | –5.60 | –8.00 | 43.6 | –2.20 | –10.20 | 42.6 | –21.60 | –24.20 |
| AlpacaEval | 50.0 | –3.70 | –1.80 | 50.0 | –5.70 | –7.50 | 50.0 | –2.10 | –11.30 |
| MATH | 53.9 | –2.40 | –2.70 | 32.0 | –4.20 | –9.70 | 22.7 | –5.20 | –9.20 |
| CUTE | 70.0 | –4.90 | –8.80 | 68.0 | –11.10 | –15.30 | 55.3 | –10.20 | –5.70 |
| GSM8K | 87.3 | –6.10 | –8.50 | 82.0 | –11.70 | –16.00 | 73.9 | –23.10 | –35.80 |
| DROP | 88.2 | –0.80 | +0.00 | 88.8 | –0.60 | –5.00 | 77.0 | –5.60 | –7.60 |
| Overall | | | | | | | | | |
| Avg MC Retention (%) | | 92.4±5.9792.4\_{\pm 5.97} | 89.2±9.3389.2\_{\pm 9.33} |  | 85.6±6.8685.6\_{\pm 6.86} | 76.8±7.9976.8\_{\pm 7.99} |  | 71.2±14.771.2\_{\pm 14.7} | 59.2±17.859.2\_{\pm 17.8} |
| Avg SA Retention (%) | | 94.6±3.9794.6\_{\pm 3.97} | 92.7±5.3592.7\_{\pm 5.35} |  | 90.3±6.7890.3\_{\pm 6.78} | 82.7±9.6582.7\_{\pm 9.65} |  | 75.4±15.175.4\_{\pm 15.1} | 65.4±17.465.4\_{\pm 17.4} |
| Avg Overall Retention (%) | | 93.4±5.1593.4\_{\pm 5.15} | 90.8±7.8190.8\_{\pm 7.81} |  | 87.7±7.0587.7\_{\pm 7.05} | 79.4±9.0579.4\_{\pm 9.05} |  | 73.1±14.773.1\_{\pm 14.7} | 62.0±17.562.0\_{\pm 17.5} |

## 2 Language Models are Robust to Non-Canonical Tokenizations

In our main experiments, we evaluate the robustness of LMs to non-canonical tokenizations by comparing their performance on downstream tasks when given different tokenizations of the input.

### 2.1 Background

Most LMs today, and all the models we study, use the Byte-Pair Encoding (BPE) [[52](#bib.bib52)] algorithm for tokenization.
The BPE tokenizer is learned by splitting a corpus of text into bytes, which form the initial vocabulary, then iteratively merging the most frequent pair of tokens into a new token that is added to the vocabulary.
To encode a new text, it is split into bytes, and the learned merges are applied in the same order.
As a result, a BPE tokenizer always produces the same token sequence for the same text.
Further, because BPE tokens do not cross whitespace boundaries, the same whitespace-delimited word is always represented with the same token or token sequence.

A natural observation is that given a tokenizer vocabulary, there exist many token sequences that decode to the same text.
For instance,  ␣ cat could be tokenized as [ ␣ cat], [ ␣ , cat], [ ␣ , c, at], [ ␣ , c, a, t], etc.
In general, the number of non-canonical tokenizations grows exponentially with the length of the text.
Many previous works have argued that the probability of a string should be calculated as the sum of probabilities of all possible tokenizations [[7](#bib.bib7), [9](#bib.bib9), [19](#bib.bib19)].
However, less attention has been paid to how non-canonical tokenizations affect LMs in generative settings.

### 2.2 Setup

We consider two non-canonical tokenization schemes.
(1) Random tokenization produces a tokenization (uniformly at random) from the set of tokenizations more granular than the canonical one.
This can be achieved by recursively splitting individual tokens into a valid pair of tokens, similarly to [[53](#bib.bib53)]; the pseudocode and a proof of correctness is provided in [Appendix A](#A1 "Appendix A Random Non-Canonical Tokenization ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations").
(2) Character-level tokenization decomposes the string into character tokens, i.e., using no subword token from the vocabulary.
For text containing only English letters and punctuation (where each character is exactly one byte), this produces the most granular possible tokenization.

![Refer to caption](/html/2506.19004/assets/Images/granularity.jpg)


Figure 2: Model performance generally declines as the tokenization becomes more granular. We achieve variation in tokenization length using different values of pp in BPE-dropout, and group tokenizations into buckets based on how many times longer it is than the canonical tokenization.

We consider three models, Llama-3.1-8B-Instruct [[38](#bib.bib38)], OLMO2-7B-Instruct [[43](#bib.bib43)], and Qwen-2.5-7B-Instruct [[47](#bib.bib47)], which we evaluate on 20 benchmarks shown in [Table 1](#S1.T1 "Table 1 ‣ 1 Introduction ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations").
Please see [§B.1](#A2.SS1 "B.1 General benchmarks ‣ Appendix B Evaluation Details ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations") for further description of the datasets and evaluation setup.

### 2.3 Results

Shown in [Table 1](#S1.T1 "Table 1 ‣ 1 Introduction ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations"), while random tokenization consistently leads to worse performance compared to the canonical tokenization, the effect is small.
On average across benchmarks, Qwen-2.5 retains 93.4% of its performance when given random tokenization, followed by Llama-3.1 at 87.7% and Olmo-2 at 73.1%.
The performance drops further with character-level tokenization, with retention of 90.8%, 79.4%, and 62.0% for the three models, respectively.
This ranking of models in terms of retention is consistent with their ranking in absolute accuracy (under canonical tokenization), suggesting that stronger models are generally more robust to non-canonical tokenization strategies.

We also observe that all models retain performance better on short answer (SA) benchmarks (where the model generates an output in free-form) compared to multiple choice (MC) benchmarks (where the model is instructed to directly output the correct answer choice).
In addition, LMs consistently *produce* correct token sequences even when conditioning on non-canonical tokenizations. We hypothesize that, in the SA setting, models benefit from eventually conditioning on recent correctly-tokenized context.

### 2.4 Analysis: How does granularity of the tokenization affect robustness?

We next study whether tokenization fine-grainedness correlates in general with model robustness.
We measure the fine-grainedness of a given non-canonical tokenization by how many times longer it is (in tokens) than the canonical tokenization, which we call the “length ratio.”
Finer-grained tokenizations have higher ratios, while coarser ones have ratios closer to 1.
We produce tokenizations with diverse length ratios by applying BPE dropout [[46](#bib.bib46)] with p∈[0.1,0.2,…,0.9]p\in[0.1,0.2,...,0.9], which controls the probability with which each merge is dropped.
(High pp leads to finer-grained segmentations, and p=0.0p=0.0 corresponds to conventional BPE.)

[Figure 2](#S2.F2 "Figure 2 ‣ 2.2 Setup ‣ 2 Language Models are Robust to Non-Canonical Tokenizations ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations") shows the relationship between the length ratio and the average performance retention relative to canonical tokenization, with finer-grained tokenization generally leading to worse performance.
When performance retention is averaged across tasks, the negative correlation is statistically significant under Kendall’s τ\tau with p=0.003p=0.003.

## 3 Can non-canonical tokenizations improve model performance?

Table 2: Examples from tasks we construct where non-canonical tokenizations lead to improved performance for Llama-3.1-7B-Instruct ([§3](#S3 "3 Can non-canonical tokenizations improve model performance? ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations")).

|  |
| --- |
| Counting characters: Count the number of the letter 'r'  in the word strawberry. |
| Acronyms: Come up with a sequence of words where the first letters would form this acronym: isman |
| Codeline Description: What does the following code do: |
| {code block} |
| A. Counts paths from a point to reach Origin |
| B. Program to check if a matrix is symmetric |
| C. Longest subsequence from an array of pairs having first element increasing and second element decreasing . |
| D. Count the number of strings in an array whose distinct characters are less than equal to M |
| Arithmetic: 8492079913 + 4877278482 = |

If LMs can process non-canonical tokenizations, this points to the exciting possibility that tokenization schemes can be modified completely at inference-time.
This would be useful if, in certain settings, there exists a better representation of text than what the tokenizer produces.
In this section, we develop a suite of tasks that intuitively require understanding of the orthography of the text, and show that Llama-3.1-8B-Instruct performs better under non-canonical tokenization schemes.

### 3.1 Tasks

Please see [Table 2](#S3.T2 "Table 2 ‣ 3 Can non-canonical tokenizations improve model performance? ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations") for an example question in each task and [§B.2](#A2.SS2 "B.2 Constructed Benchmarks ‣ Appendix B Evaluation Details ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations") for further details on dataset construction.
For all tasks except Arithmetic, we use character-level tokenization.

##### Counting Characters

This task asks the model to count the number of occurrences of the most common letter in 5-10 character tokens in Llama-3.1’s vocabulary, and contains 1001 samples.

##### Acronyms

This task asks models to generate a list of words whose first letters form a given acronym.
We construct 3594 5-letter acronyms by sampling each letter uniformly at random from the alphabet.

##### Code Description

For a more real-world application, we construct a task where the model is given a code snippet and asked to identify the function of the code in natural language from four MC options.
The setup is inspired by the Codeline Description task from BIG-Bench [[5](#bib.bib5)], but to increase the difficulty we use more complex code snippets and corresponding natural language descriptions from XLCoST [[65](#bib.bib65)].
To collect incorrect answers, we sample three other code descriptions from the dataset. This task contains 4800 samples across 6 programming languages.

##### Arithmetic

Prior work has suggested that arithmetic is difficult for LMs in part due to poor segmentation of digits [[42](#bib.bib42), [57](#bib.bib57)].
We curate a simple arithmetic dataset by constructing addition and subtraction tasks for 10 digit numbers.
Here, we use a different segmentation strategy.
The Llama-3.1 tokenizer segments numbers into groups of three left-to-right (e.g., 1000000 is encoded as [“100”, “000”, “0”]), due to the pretokenization regular expression looking for matches greedily from the left.
Inspired by [[54](#bib.bib54)], we instead segment digits into groups of three right-to-left (e.g., [“1”, “000”, “000”]).
This task contains 1000 addition and subtraction questions in total.

Table 3: On several tasks, Llama-3.1-8B-Instruct achieves better performance when using a non-canonical tokenization scheme. For the first four tasks, the input is tokenized at the character level; for Arithmetic, we segment digits into groups of three digits from right to left.

|  |  |  |  |
| --- | --- | --- | --- |
| Task | Canonical | Alternative | Δ\Delta |
| Counting Characters | 66.5 | 73.5 | +6.99 |
| Acronyms | 49.7 | 57.4 | +7.74 |
| Code Description | 68.6 | 82.9 | +14.3 |
| Arithmetic | 36.5 | 70.2 | +33.7 |

### 3.2 Results

Shown in [Table 3](#S3.T3 "Table 3 ‣ Arithmetic ‣ 3.1 Tasks ‣ 3 Can non-canonical tokenizations improve model performance? ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations"), in all the tasks we construct, the non-canonical tokenization strategy leads to substantially better performance compared to the canonical tokenization.
In particular, we observe a +14.3% improvement on code description and +33.7% on arithmetic.
Our results show that the tokenization scheme used in training is not necessarily the optimal one at inference-time, and replacing them with intuitively meaningful tokenizations can bring substantial performance gains.
We leave automatically identifying the optimal tokenization as a promising direction for future work.

## 4 Investigating the Source of Robustness

Thus far, our experiments have used post-trained “instruct” models.
In this section, we find that pretrained-only models are actually unable to produce fluent continuations of unusually tokenized context ([§4.1](#S4.SS1 "4.1 When does robustness appear in model training? ‣ 4 Investigating the Source of Robustness ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations")), perform ablations to identify the conditions enabling robustness ([§4.2](#S4.SS2 "4.2 Why do instruction-tuned models become robust? ‣ 4 Investigating the Source of Robustness ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations")), and finally provide support for an explanation of why generative robustness arises during post-training ([§4.3](#S4.SS3 "4.3 Disentangling understanding from generation ‣ 4 Investigating the Source of Robustness ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations")).

### 4.1 When does robustness appear in model training?

We first quantify the robustness of models at different stages of the model development pipeline by using the Olmo2 and Tulu3 [[33](#bib.bib33)] model families which include the base, SFT, DPO, and final instruct models.
For simplicity, we focus on AlpacaEval and use character-level tokenization.
For base models, we construct the prompt by placing the instruction in a question-answer template (Question: {instruction}\nAnswer:).
We define three simple measures of generation quality.

##### Spelling

We measure the proportion of (whitespace-delimited) words in the generation that can be found in a collection of the top 10,000 most common English words.222<https://github.com/first20hours/google-10000-english>

##### Grammaticality

We use LanguageTool’s grammar checker333<https://github.com/languagetool-org/languagetool> to count the number of grammatical mistakes, which we normalize by the number of words in the generation and subtract from 1 to produce a grammaticality score where higher is better.

##### Win rate

To measure overall generation quality, we use alpaca\_eval\_gpt4 as an LM judge in the AlpacaEval framework and report the win rate of the generation given alternative against canonical tokenizations of the context.
Unlike the previous two metrics, this measures not only the quality of the generation but also its relevance to the context.

![Refer to caption](/html/2506.19004/assets/Images/combined_posttraining_robustness.jpg)

Figure 3: Pretrained-only models completely fail to generate coherent output conditioned on non-canonical tokenizations of context; robustness is gained in the SFT stage. We evaluate the spelling, grammaticality, and AlpacaEval win rate of model generations.
Note that since Tulu3 uses Llama-3 as the base model, its base scores are computed using Llama-3’s base model scores.

Shown in [Figure 3](#S4.F3 "Figure 3 ‣ Win rate ‣ 4.1 When does robustness appear in model training? ‣ 4 Investigating the Source of Robustness ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations"), the base models of Olmo2 and Llama-3.1 are both unable to produce sensible output conditioned on character-level tokenizations of context, scoring at best 0.317 on spelling and 0.260 on grammaticality.
Qualitatively, generations are extremely difficult to parse and often involve odd character substitutions and repetitions (e.g., Yoou, haviin).
Despite this, they sometimes reflect an understanding of the prompt.
Consider, for example,

|  |
| --- |
| Question: I like to host guests at my home from time to time [...] Can you give me a recipe for Canjeero?    Answer: I aam glade tio hear tio hear tio hear tio hear that yoou enjoy    haviin gauests at yoour hoome an tio keeep tio keeep tio keeep |

In contrast, the post-trained models are more robust across all three metrics, with much of the improvement coming from the SFT stage alone.

### 4.2 Why do instruction-tuned models become robust?

We first replicate the finding from [§4.1](#S4.SS1 "4.1 When does robustness appear in model training? ‣ 4 Investigating the Source of Robustness ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations") that SFT yields robustness to non-canonical tokenizations by finetuning the Llama-3.2-1B base model on the Tulu 3 SFT Personas Instruction Following dataset.
Then, we perform the following interventions on the SFT training data and procedure to shed light on the possible source.

##### Gradient over full sequence

SFT on instruction-response pairs conventionally uses a loss mask over the instruction tokens, so that only the response tokens contribute to the loss.
We remove this loss mask and instead compute gradients over the entire instruction and response.

##### Question/answer template

We replace the chat template with a simple question-answer template, Question: {instruction} Answer: {response}, both for training and evaluation.

##### Removing the chat template

We remove the chat template by concatenating the instruction and response without any special formatting.
In evaluation, we again provide the instruction alone.

##### Removing the instruction

After SFT training, the LM’s goal is no longer to continue a given text prefix, but rather to generate a response to the given instruction.
To ablate the nature of the data itself, we take only the responses from the SFT data, and randomly split each into a new “prompt” and “response,” which we format with the SFT template.444We match the instruction length distribution by counting the number of tokens nn in the original instruction, and formatting the first nn tokens of the response as the new “instruction.”
At test time, we similarly provide an incomplete response within “instruction” tags.
Since the purpose of the passage is generally inferrable from the first few words of the gold response (“Sure, here’s a recipe for Kubdari…”), we are able to evaluate generated responses under the same AlpacaEval framework.

Our results are shown in [Figure 4](#S4.F4 "Figure 4 ‣ 4.3 Disentangling understanding from generation ‣ 4 Investigating the Source of Robustness ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations").
We replicate the finding that SFT (No ablation) leads the model to be able to handle non-canonical tokenizations.
This persists when computing gradients over the entire instruction and response (Full gradient) so that the training procedure matches regular pretraining.
Replacing the original chat template with a simple question-answer template (QA template) also maintains model robustness.
However, the usage of a template is crucial — when directly concatenating the instruction and response (Removing chat template), the model fails to produce coherent generations, with the spelling score dropping from 0.786 in the no ablation setting to 0.0698.
Inserting the chat template into pretraining-style data (Removing the instruction) also does not yield robustness, with a spelling and grammaticality scores remaining low at 0.181 and 0.158, respectively.
Overall, these findings suggest that in order for the LM to generate fluent continuations given non-canonical tokenizations, the context and expected continuation need to represent separate turns of dialogue, and additionally, be demarcated with a special template.

### 4.3 Disentangling understanding from generation

One plausible explanation for our findings thus far is that both base and instruction-tuned models grasp the semantics of non-canonical tokenizations, yet falsely perceive them as containing misspellings.
While base LMs attempt to faithfully continue these mistakes and degenerate into nonsensical output, instruction-tuned models are trained to provide fluent responses regardless of the instruction, leading to the results observed in [§4.1](#S4.SS1 "4.1 When does robustness appear in model training? ‣ 4 Investigating the Source of Robustness ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations").
To test this hypothesis, we construct two simple tests:

1. 1.

   Word Repeat: To determine if a model perceives the meaning of a word with non-canonical tokenization, we prompt the model to repeat a given word (while correcting any typos).
2. 2.

   Identifying Misspellings: To determine if a model perceives a misspelling, we ask it to identify the word with a misspelling among two options: a (correctly tokenized) misspelling of a word and an non-canonical tokenization of that word (correctly spelled).

![Refer to caption](/html/2506.19004/assets/Images/ablation.jpg)


Figure 4: Ablations on the SFT training data and procedure indicate that the separation of the context and expected continuation — as different turns of dialogue demarcated with a special token — is key to robustness to non-canonical tokenizations.

Results are shown in [Table 4](#S4.T4 "Table 4 ‣ 4.3 Disentangling understanding from generation ‣ 4 Investigating the Source of Robustness ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations").
Consistent with our hypothesis, we find that both the base and instruct models from the Llama-3.1-8B family score highly (>90%>90\%) on Word Repeat.
This means that the base model, despite its poor performance in [§4.1](#S4.SS1 "4.1 When does robustness appear in model training? ‣ 4 Investigating the Source of Robustness ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations"), actually recognizes the correct form of non-canonical tokenizations as well as its post-trained counterpart.
In addition, both models perform at random when asked to distinguish non-canonical tokenization from true misspellings.
In other words, the instruct model produces fluent responses ([§2](#S2 "2 Language Models are Robust to Non-Canonical Tokenizations ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations")) while interpreting the instruction as heavily misspelled!
While instruct models evidently overcome this, the base model likely attempts to mimic the (perceived) idiosyncratic surface form, thus producing nonsensical (yet sometimes relevant) outputs.

Table 4: Both base and instruct models from the Llama-3.1 family recognize words represented with non-canonical tokenizations (performing well on Word Repeat), but incorrectly perceive that there are misspellings (performing at random on Identifying Misspellings).

|  |  |  |
| --- | --- | --- |
|  | Word Repeat | Identifying Misspellings |
| Llama-3.1-8B | 90.8 | 48.2 |
| Llama-3.1-8B-Instruct | 92.0 | 55.8 |

## 5 Related Work

##### Robustness to tokenization

The extent to which LMs are limited by their tokenization is a topic of much debate, with the story evolving as LMs become larger and more capable.
It is commonly argued that tokenization obscures orthographic information about tokens from the LM, leading to unexpected failures [[17](#bib.bib17), [8](#bib.bib8), [59](#bib.bib59)].
As a result, there have been many efforts towards linguistically-informed tokenization that make derivational, compound, and morphological boundaries within words explicit [[32](#bib.bib32), [24](#bib.bib24), [25](#bib.bib25), [62](#bib.bib62), [4](#bib.bib4)].
Similarly, BPE-dropout [[46](#bib.bib46)] and related methods [[53](#bib.bib53)] introduce variation in how a given string is tokenized to make models more robust to rare, misspelled, and unseen words.

However, other evidence suggests that LMs naturally overcome these limitations.
For instance, token embeddings have been found to robustly encode character-level information, especially in larger models [[31](#bib.bib31), [26](#bib.bib26)].
This may be because word variants that do not share tokens in common (consider e.g., [ ␣ dictionary] and [ ␣ diction, aries], as tokenized by GPT-2) incentivize the model to learn spelling as a general solution to understanding their relations [[31](#bib.bib31)].
Other works argue that LMs maintain an implicit vocabulary, and can compose arbitrary token sequences (including non-canonical ones) into useful higher-level representations [[18](#bib.bib18), [30](#bib.bib30)].
Even in domains like biomedical text where terms are highly agglutinative, using tokenizers that segment on meaningful components does not lead to improved models [[27](#bib.bib27)].
Recent works have even found that coarser superword tokenization [[36](#bib.bib36), [51](#bib.bib51)], which capture common word sequences in a single token, preserve character-level understanding while providing benefits in compression and downstream performance.

Our work informs this conversation by showing that LMs can effectively leverage character-level knowledge of their tokens and glean potential benefits of improved representation at inference time.

##### Non-canonical tokenizations

It has long been recognized that there are many possible ways to segment a string into tokens with a fixed vocabulary [[10](#bib.bib10)], which in principle should be considered in the calculation of a string’s likelihood [[7](#bib.bib7), [9](#bib.bib9), [19](#bib.bib19)].
In contemporaneous work, Geh et al. [[20](#bib.bib20)] also show that LMs retain semantic understanding of non-canonical tokenizations.
However, they only study this over a small set of 15 examples, as their main focus is to show that non-canonical tokenizations can be constructed adversarially to trigger unsafe completions.
In contrast, we provide a more systematic study of LM robustness using benchmark evaluations and additionally study its source.

Somewhat relatedly, other works have provided algorithms for sampling at the character- or byte-level from tokenizer-based LMs [[45](#bib.bib45), [58](#bib.bib58), [3](#bib.bib3), [21](#bib.bib21)].
Together, these directions suggest that despite being trained with one deterministic tokenization scheme, LMs can both condition on and produce token sequences over a different (sub)vocabulary.

## 6 Conclusion

Despite being trained with deterministic tokenization algorithms, we show that instruction-tuned language models are surprisingly robust to token sequences not seen in training.
In certain domains, such as arithmetic or code, more intuitively meaningful tokenizations can even be swapped-in at inference time for improved performance.
We analyze the source of this robustness, and find that while the base and instruct models both perceive the semantics of non-canonical tokenizations, only instruct models are capable of providing fluent continuations..
Our work demonstrates a way in which LMs are not necessarily tied to tokenizer they were trained with, and highlights the potential of finding more optimal representations of text after pretraining.

#### Acknowledgments

We would like to thank Joseph An and Ricky Koppolu, as well as the broader UW NLP community, for helpful conversations about this work.
AL and JH are supported by the NSF Graduate Research Fellowship.

## References

* Ahia et al. [2024]

  O. Ahia, S. Kumar, H. Gonen, V. Hofmann, T. Limisiewicz, Y. Tsvetkov, and N. A. Smith.
  MAGNET: Improving the multilingual fairness of language models with adaptive gradient-based tokenization.
  In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024.
  URL <https://openreview.net/forum?id=1e3MOwHSIX>.
* Arnett and Bergen [2025]

  C. Arnett and B. Bergen.
  Why do language models perform worse for morphologically complex languages?
  In *Proceedings of the 31st International Conference on Computational Linguistics*, pages 6607–6623, 2025.
* Athiwaratkun et al. [2024]

  B. Athiwaratkun, S. Wang, M. Shang, Y. Tian, Z. Wang, S. K. Gonugondla, S. K. Gouda, R. Kwiatkowski, R. Nallapati, P. Bhatia, and B. Xiang.
  Token alignment via character matching for subword completion.
  In L.-W. Ku, A. Martins, and V. Srikumar, editors, *Findings of the Association for Computational Linguistics: ACL 2024*, pages 15725–15738, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics.
  doi: 10.18653/v1/2024.findings-acl.929.
  URL <https://aclanthology.org/2024.findings-acl.929>.
* Bauwens and Delobelle [2024]

  T. Bauwens and P. Delobelle.
  BPE-knockout: Pruning pre-existing BPE tokenisers with backwards-compatible morphological semi-supervision.
  In K. Duh, H. Gomez, and S. Bethard, editors, *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pages 5810–5832, Mexico City, Mexico, June 2024. Association for Computational Linguistics.
  doi: 10.18653/v1/2024.naacl-long.324.
  URL <https://aclanthology.org/2024.naacl-long.324>.
* bench authors [2023]

  B. bench authors.
  Beyond the imitation game: Quantifying and extrapolating the capabilities of language models.
  *Transactions on Machine Learning Research*, 2023.
  ISSN 2835-8856.
  URL <https://openreview.net/forum?id=uyTL5Bvosj>.
* Bisk et al. [2020]

  Y. Bisk, R. Zellers, R. L. Bras, J. Gao, and Y. Choi.
  Piqa: Reasoning about physical commonsense in natural language.
  In *Thirty-Fourth AAAI Conference on Artificial Intelligence*, 2020.
* Cao and Rimell [2021]

  K. Cao and L. Rimell.
  You should evaluate your language model on marginal likelihood over tokenisations.
  In M.-F. Moens, X. Huang, L. Specia, and S. W.-t. Yih, editors, *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 2104–2114, Online and Punta Cana, Dominican Republic, Nov. 2021. Association for Computational Linguistics.
  doi: 10.18653/v1/2021.emnlp-main.161.
  URL <https://aclanthology.org/2021.emnlp-main.161>.
* Chai et al. [2024]

  Y. Chai, Y. Fang, Q. Peng, and X. Li.
  Tokenization falling short: On subword robustness in large language models.
  In Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, editors, *Findings of the Association for Computational Linguistics: EMNLP 2024*, pages 1582–1599, Miami, Florida, USA, Nov. 2024. Association for Computational Linguistics.
  doi: 10.18653/v1/2024.findings-emnlp.86.
  URL <https://aclanthology.org/2024.findings-emnlp.86>.
* Chirkova et al. [2023]

  N. Chirkova, G. Kruszewski, J. Rozen, and M. Dymetman.
  Should you marginalize over possible tokenizations?
  In A. Rogers, J. Boyd-Graber, and N. Okazaki, editors, *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, pages 1–12, Toronto, Canada, July 2023. Association for Computational Linguistics.
  doi: 10.18653/v1/2023.acl-short.1.
  URL <https://aclanthology.org/2023.acl-short.1>.
* Church [2020]

  K. W. Church.
  Emerging trends: Subwords, seriously?
  *Natural Language Engineering*, 26(3):375–382, 2020.
  doi: 10.1017/S1351324920000145.
* Clark et al. [2019]

  C. Clark, K. Lee, M.-W. Chang, T. Kwiatkowski, M. Collins, and K. Toutanova.
  BoolQ: Exploring the surprising difficulty of natural yes/no questions.
  In J. Burstein, C. Doran, and T. Solorio, editors, *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 2924–2936, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.
  doi: 10.18653/v1/N19-1300.
  URL <https://aclanthology.org/N19-1300>.
* Clark et al. [2022]

  J. H. Clark, D. Garrette, I. Turc, and J. Wieting.
  Canine: Pre-training an efficient tokenization-free encoder for language representation.
  *Transactions of the Association for Computational Linguistics*, 10:73–91, 2022.
  doi: 10.1162/tacl˙a˙00448.
  URL <https://aclanthology.org/2022.tacl-1.5>.
* Clark et al. [2018]

  P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord.
  Think you have solved question answering? try arc, the ai2 reasoning challenge, 2018.
  URL <https://arxiv.org/abs/1803.05457>.
* Cobbe et al. [2021]

  K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman.
  Training verifiers to solve math word problems, 2021.
  URL <https://arxiv.org/abs/2110.14168>.
* Dua et al. [2019]

  D. Dua, Y. Wang, P. Dasigi, G. Stanovsky, S. Singh, and M. Gardner.
  DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs.
  In J. Burstein, C. Doran, and T. Solorio, editors, *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 2368–2378, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.
  doi: 10.18653/v1/N19-1246.
  URL <https://aclanthology.org/N19-1246>.
* Dubois et al. [2023]

  Y. Dubois, X. Li, R. Taori, T. Zhang, I. Gulrajani, J. Ba, C. Guestrin, P. Liang, and T. B. Hashimoto.
  Alpacafarm: A simulation framework for methods that learn from human feedback, 2023.
* Edman et al. [2024]

  L. Edman, H. Schmid, and A. Fraser.
  CUTE: Measuring LLMs’ understanding of their tokens.
  In Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, editors, *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pages 3017–3026, Miami, Florida, USA, Nov. 2024. Association for Computational Linguistics.
  doi: 10.18653/v1/2024.emnlp-main.177.
  URL <https://aclanthology.org/2024.emnlp-main.177>.
* Feucht et al. [2024]

  S. Feucht, D. Atkinson, B. C. Wallace, and D. Bau.
  Token erasure as a footprint of implicit vocabulary items in LLMs.
  In Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, editors, *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pages 9727–9739, Miami, Florida, USA, Nov. 2024. Association for Computational Linguistics.
  doi: 10.18653/v1/2024.emnlp-main.543.
  URL <https://aclanthology.org/2024.emnlp-main.543>.
* Geh et al. [2024]

  R. Geh, H. Zhang, K. Ahmed, B. Wang, and G. Van Den Broeck.
  Where is the signal in tokenization space?
  In Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, editors, *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pages 3966–3979, Miami, Florida, USA, Nov. 2024. Association for Computational Linguistics.
  doi: 10.18653/v1/2024.emnlp-main.230.
  URL <https://aclanthology.org/2024.emnlp-main.230>.
* Geh et al. [2025]

  R. L. Geh, Z. Shao, and G. V. den Broeck.
  Adversarial tokenization, 2025.
  URL <https://arxiv.org/abs/2503.02174>.
* Hayase et al. [2025]

  J. Hayase, A. Liu, N. A. Smith, and S. Oh.
  Sampling from your language model one byte at a time, 2025.
  URL <https://arxiv.org/abs/2506.14123>.
* Hendrycks et al. [2021a]

  D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt.
  Measuring massive multitask language understanding.
  In *International Conference on Learning Representations*, 2021a.
  URL <https://openreview.net/forum?id=d7KBjmI3GmQ>.
* Hendrycks et al. [2021b]

  D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt.
  Measuring mathematical problem solving with the MATH dataset.
  In *Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)*, 2021b.
  URL <https://openreview.net/forum?id=7Bywt2mQsCe>.
* Hofmann et al. [2021]

  V. Hofmann, J. Pierrehumbert, and H. Schütze.
  Superbizarre is not superb: Derivational morphology improves BERT’s interpretation of complex words.
  In C. Zong, F. Xia, W. Li, and R. Navigli, editors, *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 3594–3608, Online, Aug. 2021. Association for Computational Linguistics.
  doi: 10.18653/v1/2021.acl-long.279.
  URL <https://aclanthology.org/2021.acl-long.279>.
* Hofmann et al. [2022]

  V. Hofmann, H. Schuetze, and J. Pierrehumbert.
  An embarrassingly simple method to mitigate undesirable properties of pretrained language model tokenizers.
  In S. Muresan, P. Nakov, and A. Villavicencio, editors, *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, pages 385–393, Dublin, Ireland, May 2022. Association for Computational Linguistics.
  doi: 10.18653/v1/2022.acl-short.43.
  URL <https://aclanthology.org/2022.acl-short.43>.
* Itzhak and Levy [2022]

  I. Itzhak and O. Levy.
  Models in a spelling bee: Language models implicitly learn the character composition of tokens.
  In M. Carpuat, M.-C. de Marneffe, and I. V. Meza Ruiz, editors, *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 5061–5068, Seattle, United States, July 2022. Association for Computational Linguistics.
  doi: 10.18653/v1/2022.naacl-main.373.
  URL <https://aclanthology.org/2022.naacl-main.373>.
* Jimenez Gutierrez et al. [2023]

  B. Jimenez Gutierrez, H. Sun, and Y. Su.
  Biomedical language models are robust to sub-optimal tokenization.
  In D. Demner-fushman, S. Ananiadou, and K. Cohen, editors, *The 22nd Workshop on Biomedical Natural Language Processing and BioNLP Shared Tasks*, pages 350–362, Toronto, Canada, July 2023. Association for Computational Linguistics.
  doi: 10.18653/v1/2023.bionlp-1.32.
  URL <https://aclanthology.org/2023.bionlp-1.32>.
* Joshi et al. [2017]

  M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer.
  Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension, 2017.
  URL <https://arxiv.org/abs/1705.03551>.
* Kaggle [2019]

  Kaggle.
  200,000+ jeopardy! questions, 2019.
  URL <https://www.kaggle.com/datasets/tunguz/200000-jeopardy-questions>.
* Kaplan et al. [2025]

  G. Kaplan, M. Oren, Y. Reif, and R. Schwartz.
  From tokens to words: On the inner lexicon of LLMs.
  In *The Thirteenth International Conference on Learning Representations*, 2025.
  URL <https://openreview.net/forum?id=328vch6tRs>.
* Kaushal and Mahowald [2022]

  A. Kaushal and K. Mahowald.
  What do tokens know about their characters and how do they know it?
  In M. Carpuat, M.-C. de Marneffe, and I. V. Meza Ruiz, editors, *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 2487–2507, Seattle, United States, July 2022. Association for Computational Linguistics.
  doi: 10.18653/v1/2022.naacl-main.179.
  URL <https://aclanthology.org/2022.naacl-main.179>.
* Klein and Tsarfaty [2020]

  S. Klein and R. Tsarfaty.
  Getting the ##life out of living: How adequate are word-pieces for modelling complex morphology?
  In G. Nicolai, K. Gorman, and R. Cotterell, editors, *Proceedings of the 17th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology*, pages 204–209, Online, July 2020. Association for Computational Linguistics.
  doi: 10.18653/v1/2020.sigmorphon-1.24.
  URL <https://aclanthology.org/2020.sigmorphon-1.24>.
* Lambert et al. [2025]

  N. Lambert, J. Morrison, V. Pyatkin, S. Huang, H. Ivison, F. Brahman, L. J. V. Miranda, A. Liu, N. Dziri, S. Lyu, Y. Gu, S. Malik, V. Graf, J. D. Hwang, J. Yang, R. L. Bras, O. Tafjord, C. Wilhelm, L. Soldaini, N. A. Smith, Y. Wang, P. Dasigi, and H. Hajishirzi.
  Tulu 3: Pushing frontiers in open language model post-training, 2025.
  URL <https://arxiv.org/abs/2411.15124>.
* Levesque et al. [2012]

  H. J. Levesque, E. Davis, and L. Morgenstern.
  The winograd schema challenge.
  In *Proceedings of the Thirteenth International Conference on Principles of Knowledge Representation and Reasoning*, page 552–561. AAAI Press, 2012.
* Limisiewicz et al. [2024]

  T. Limisiewicz, T. Blevins, H. Gonen, O. Ahia, and L. Zettlemoyer.
  MYTE: Morphology-driven byte encoding for better and fairer multilingual language modeling.
  In L.-W. Ku, A. Martins, and V. Srikumar, editors, *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 15059–15076, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics.
  doi: 10.18653/v1/2024.acl-long.804.
  URL <https://aclanthology.org/2024.acl-long.804>.
* Liu et al. [2025]

  A. Liu, J. Hayase, V. Hofmann, S. Oh, N. A. Smith, and Y. Choi.
  SuperBPE: Space travel for language models.
  *arXiv preprint arXiv:2503.13423*, 2025.
  URL <https://arxiv.org/abs/2503.13423>.
* Maini et al. [2024]

  P. Maini, Z. Feng, A. Schwarzschild, Z. C. Lipton, and J. Z. Kolter.
  Tofu: A task of fictitious unlearning for llms, 2024.
  URL <https://arxiv.org/abs/2401.06121>.
* Meta [2024]

  Meta.
  The llama 3 herd of models, 2024.
  URL <https://arxiv.org/abs/2407.21783>.
* Mihaylov et al. [2018]

  T. Mihaylov, P. Clark, T. Khot, and A. Sabharwal.
  Can a suit of armor conduct electricity? a new dataset for open book question answering, 2018.
  URL <https://arxiv.org/abs/1809.02789>.
* Minixhofer et al. [2024]

  B. Minixhofer, E. Ponti, and I. Vulić.
  Zero-shot tokenizer transfer.
  In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024.
  URL <https://openreview.net/forum?id=RwBObRsIzC>.
* Nawrot et al. [2023]

  P. Nawrot, J. Chorowski, A. Lancucki, and E. M. Ponti.
  Efficient transformers with dynamic token pooling.
  In A. Rogers, J. Boyd-Graber, and N. Okazaki, editors, *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 6403–6417, Toronto, Canada, July 2023. Association for Computational Linguistics.
  doi: 10.18653/v1/2023.acl-long.353.
  URL <https://aclanthology.org/2023.acl-long.353>.
* Nogueira et al. [2021]

  R. Nogueira, Z. Jiang, and J. Lin.
  Investigating the limitations of transformers with simple arithmetic tasks, 2021.
  URL <https://arxiv.org/abs/2102.13019>.
* OLMo et al. [2024]

  T. OLMo, P. Walsh, L. Soldaini, D. Groeneveld, K. Lo, S. Arora, A. Bhagia, Y. Gu, S. Huang, M. Jordan, N. Lambert, D. Schwenk, O. Tafjord, T. Anderson, D. Atkinson, F. Brahman, C. Clark, P. Dasigi, N. Dziri, M. Guerquin, H. Ivison, P. W. Koh, J. Liu, S. Malik, W. Merrill, L. J. V. Miranda, J. Morrison, T. Murray, C. Nam, V. Pyatkin, A. Rangapur, M. Schmitz, S. Skjonsberg, D. Wadden, C. Wilhelm, M. Wilson, L. Zettlemoyer, A. Farhadi, N. A. Smith, and H. Hajishirzi.
  2 olmo 2 furious, 2024.
  URL <https://arxiv.org/abs/2501.00656>.
* Pagnoni et al. [2024]

  A. Pagnoni, R. Pasunuru, P. Rodriguez, J. Nguyen, B. Muller, M. Li, C. Zhou, L. Yu, J. Weston, L. Zettlemoyer, G. Ghosh, M. Lewis, A. Holtzman, and S. Iyer.
  Byte latent transformer: Patches scale better than tokens, 2024.
  URL <https://arxiv.org/abs/2412.09871>.
* Phan et al. [2024]

  B. Phan, M. Havasi, M. Muckley, and K. Ullrich.
  Understanding and mitigating tokenization bias in language models, 2024.
  URL <https://arxiv.org/abs/2406.16829>.
* Provilkov et al. [2020]

  I. Provilkov, D. Emelianenko, and E. Voita.
  BPE-dropout: Simple and effective subword regularization.
  In D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault, editors, *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 1882–1892, Online, July 2020. Association for Computational Linguistics.
  doi: 10.18653/v1/2020.acl-main.170.
  URL <https://aclanthology.org/2020.acl-main.170>.
* Qwen [2025]

  Qwen.
  Qwen2.5 technical report, 2025.
  URL <https://arxiv.org/abs/2412.15115>.
* Roemmele et al. [2011]

  M. Roemmele, C. A. Bejan, and A. S. Gordon.
  Choice of Plausible Alternatives: An Evaluation of Commonsense Causal Reasoning.
  In *AAAI Spring Symposium on Logical Formalizations of Commonsense Reasoning*, Stanford University, Mar. 2011.
  URL <http://ict.usc.edu/pubs/Choice%20of%20Plausible%20Alternatives-%20An%20Evaluation%20of%20Commonsense%20Causal%20Reasoning.pdf>.
* Sakaguchi et al. [2021]

  K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi.
  Winogrande: an adversarial winograd schema challenge at scale.
  *Commun. ACM*, 64(9):99–106, Aug. 2021.
  ISSN 0001-0782.
  URL <https://doi.org/10.1145/3474381>.
* Sathe et al. [2025]

  A. Sathe, D. Aggarwal, and S. Sitaram.
  Improving consistency in LLM inference using probabilistic tokenization.
  In L. Chiruzzo, A. Ritter, and L. Wang, editors, *Findings of the Association for Computational Linguistics: NAACL 2025*, pages 4766–4778, Albuquerque, New Mexico, Apr. 2025. Association for Computational Linguistics.
  ISBN 979-8-89176-195-7.
  URL <https://aclanthology.org/2025.findings-naacl.268/>.
* Schmidt et al. [2025]

  C. W. Schmidt, V. Reddy, C. Tanner, and Y. Pinter.
  Boundless byte pair encoding: Breaking the pre-tokenization barrier, 2025.
  URL <https://arxiv.org/abs/2504.00178>.
* Sennrich et al. [2016]

  R. Sennrich, B. Haddow, and A. Birch.
  Neural machine translation of rare words with subword units.
  In K. Erk and N. A. Smith, editors, *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1715–1725, Berlin, Germany, Aug. 2016. Association for Computational Linguistics.
  doi: 10.18653/v1/P16-1162.
  URL <https://aclanthology.org/P16-1162>.
* Sims et al. [2025]

  A. Sims, C. Lu, K. Kaleb, J. N. Foerster, and Y. W. Teh.
  StochasTok: Improving fine-grained subword understanding in LLMs.
  In *ICLR 2025 Workshop on Building Trust in Language Models and Applications*, 2025.
  URL <https://openreview.net/forum?id=PZnDZdkGsE>.
* Singh and Strouse [2024]

  A. K. Singh and D. Strouse.
  Tokenization counts: the impact of tokenization on arithmetic in frontier llms, 2024.
  URL <https://arxiv.org/abs/2402.14903>.
* Talmor et al. [2019]

  A. Talmor, J. Herzig, N. Lourie, and J. Berant.
  CommonsenseQA: A question answering challenge targeting commonsense knowledge.
  In J. Burstein, C. Doran, and T. Solorio, editors, *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 4149–4158, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.
  doi: 10.18653/v1/N19-1421.
  URL <https://aclanthology.org/N19-1421>.
* Tay et al. [2022]

  Y. Tay, V. Q. Tran, S. Ruder, J. Gupta, H. W. Chung, D. Bahri, Z. Qin, S. Baumgartner, C. Yu, and D. Metzler.
  Charformer: Fast character transformers via gradient-based subword tokenization.
  In *International Conference on Learning Representations*, 2022.
  URL <https://openreview.net/forum?id=JtBRnrlOEFN>.
* Thawani et al. [2021]

  A. Thawani, J. Pujara, F. Ilievski, and P. Szekely.
  Representing numbers in NLP: a survey and a vision.
  In K. Toutanova, A. Rumshisky, L. Zettlemoyer, D. Hakkani-Tur, I. Beltagy, S. Bethard, R. Cotterell, T. Chakraborty, and Y. Zhou, editors, *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 644–656, Online, June 2021. Association for Computational Linguistics.
  doi: 10.18653/v1/2021.naacl-main.53.
  URL <https://aclanthology.org/2021.naacl-main.53>.
* Vieira et al. [2024]

  T. Vieira, B. LeBrun, M. Giulianelli, J. L. Gastaldi, B. DuSell, J. Terilla, T. J. O’Donnell, and R. Cotterell.
  From language models over tokens to language models over characters, 2024.
  URL <https://arxiv.org/abs/2412.03719>.
* Wang et al. [2024a]

  D. Wang, Y. Li, J. Jiang, Z. Ding, G. Jiang, J. Liang, and D. Yang.
  Tokenization matters! degrading large language models through challenging their tokenization, 2024a.
  URL <https://arxiv.org/abs/2405.17067>.
* Wang et al. [2024b]

  J. Wang, T. Gangavarapu, J. N. Yan, and A. M. Rush.
  Mambabyte: Token-free selective state space model.
  In *First Conference on Language Modeling*, 2024b.
  URL <https://openreview.net/forum?id=X1xNsuKssb>.
* Xue et al. [2022]

  L. Xue, A. Barua, N. Constant, R. Al-Rfou, S. Narang, M. Kale, A. Roberts, and C. Raffel.
  ByT5: Towards a token-free future with pre-trained byte-to-byte models.
  *Transactions of the Association for Computational Linguistics*, 10:291–306, 2022.
  doi: 10.1162/tacl˙a˙00461.
  URL <https://aclanthology.org/2022.tacl-1.17>.
* Yehezkel and Pinter [2023]

  S. Yehezkel and Y. Pinter.
  Incorporating context into subword vocabularies.
  In A. Vlachos and I. Augenstein, editors, *Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics*, pages 623–635, Dubrovnik, Croatia, May 2023. Association for Computational Linguistics.
  doi: 10.18653/v1/2023.eacl-main.45.
  URL <https://aclanthology.org/2023.eacl-main.45>.
* Yu et al. [2023]

  L. Yu, D. Simig, C. Flaherty, A. Aghajanyan, L. Zettlemoyer, and M. Lewis.
  MEGABYTE: Predicting million-byte sequences with multiscale transformers.
  In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023.
  URL <https://openreview.net/forum?id=JTmO2V9Xpz>.
* Zellers et al. [2019]

  R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi.
  HellaSwag: Can a machine really finish your sentence?
  In A. Korhonen, D. Traum, and L. Màrquez, editors, *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 4791–4800, Florence, Italy, July 2019. Association for Computational Linguistics.
  doi: 10.18653/v1/P19-1472.
  URL <https://aclanthology.org/P19-1472>.
* Zhu et al. [2022]

  M. Zhu, A. Jain, K. Suresh, R. Ravindran, S. Tipirneni, and C. K. Reddy.
  Xlcost: A benchmark dataset for cross-lingual code intelligence, 2022.
  URL <https://arxiv.org/abs/2206.08474>.

## Appendix A Random Non-Canonical Tokenization

In this section, we provide our algorithm for producing a random non-canonical tokenization, and a proof that each non-canonical tokenization that is more fine-grained than the canonical one has equal probability of being output.

### A.1 Algorithm

We will split each token in a canonical tokenization into smaller tokens (that each exists in the tokenizer’s vocabulary).
We formulate our problem as: Given a valid token tt, and a set of vocabulary 𝒱\mathcal{V}, construct a sequence of tokens s​e​qseq using tokens that exist in 𝒱\mathcal{V} and form tt when concatenated together. We produce s​e​qseq using a recursive algorithm. Since there can be many possible s​e​qseq for each tt, we need to randomly choose one and guarantee that each possible s​e​qseq is chosen with equal probability. We achieve this by considering recursion as producing a tree. Each path down the tree corresponds to one possible way to segment tt. Each node of the tree represents a segmentation state where we have chosen some number of sub-tokens. At each node, we weigh the choice of which child node to visit by the number of leaves in the sub-tree that is rooted at each child node. This guarantees that each path down the tree is chosen with equal probability since the number of paths down a tree is equal to the number of leaf nodes in that tree. The pseudocode for the algorithm is in [1](#alg1 "Algorithm 1 ‣ A.2 Proof ‣ Appendix A Random Non-Canonical Tokenization ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations").

### A.2 Proof

Goal:  To prove that the random segmentation algorithm chooses one valid segmentation from all possible valid segmentations with uniform probability.

Notation:  Let W​(i)W(i) denote the number of valid segmentation completions (i.e., the number of leaves in the recursive tree) for the substring starting at index ii. In particular, W​(|t​o​k​e​n|)=1W(|token|)=1. Note that W​(i)W(i) is calculated by the memoized recursive function c​o​u​n​t​S​e​g​m​e​n​t​s​(i)countSegments(i), which calculates the number of leaves of the subtree rooted at ii.

Base Case:  Consider the node corresponding to i=|t​o​k​e​n|i=|token| (the end of the token). Here, there is exactly one valid segmentation (the empty segmentation), so the algorithm returns it with probability 1. That is, every segmentation (in this case, the only one) is chosen with probability

|  |  |  |
| --- | --- | --- |
|  | 1W​(|t​o​k​e​n|)=11=1.\frac{1}{W(|token|)}=\frac{1}{1}=1. |  |

Thus, the base case holds.

Inductive Hypothesis:  Assume that for any node corresponding to an index jj with j>ij>i (i.e., deeper in the recursion tree), every complete segmentation (leaf) in the subtree rooted at jj is chosen with probability

|  |  |  |
| --- | --- | --- |
|  | 1W​(j).\frac{1}{W(j)}. |  |

Inductive Step:  Now consider a node corresponding to index ii (with i<|t​o​k​e​n|i<|token|). Suppose that from ii there are kk valid branches corresponding to choosing substrings that end at indices j1,j2,…,jkj\_{1},j\_{2},\dots,j\_{k}, where for each jj we have i<j≤|t​o​k​e​n|i<j\leq|token| and the substring token[i:j]token[i:j] is in the vocabulary. By definition,

|  |  |  |
| --- | --- | --- |
|  | W​(i)=∑r=1kW​(jr).W(i)=\sum\_{r=1}^{k}W(j\_{r}). |  |

The algorithm selects the branch from ii to a specific child jj with probability

|  |  |  |
| --- | --- | --- |
|  | P​(i→j)=W​(j)W​(i).P(i\rightarrow j)=\frac{W(j)}{W(i)}. |  |

Once branch i→ji\rightarrow j is chosen, by the inductive hypothesis every complete segmentation (leaf) in the subtree rooted at jj is chosen with probability

|  |  |  |
| --- | --- | --- |
|  | 1W​(j).\frac{1}{W(j)}. |  |

Thus, the probability P​(S)P(S) of obtaining a particular complete segmentation 𝒱\mathcal{V} that starts at ii by first taking the branch i→ji\rightarrow j and then following a specific path in the subtree rooted at jj is

|  |  |  |
| --- | --- | --- |
|  | P​(S)=W​(j)W​(i)⋅1W​(j)=1W​(i).P(S)=\frac{W(j)}{W(i)}\cdot\frac{1}{W(j)}=\frac{1}{W(i)}. |  |

Since the factor W​(j)W(j) cancels, the probability P​(S)P(S) is independent of the particular child jj chosen.

Conclusion:  By the inductive step, every complete segmentation (leaf) in the subtree rooted at any index ii is chosen with probability 1W​(i)\frac{1}{W(i)}. In particular, when i=0i=0 (the start of the token), every valid segmentation of the entire token is selected with uniform probability 1W​(0)\frac{1}{W(0)}. This completes the proof.

Algorithm 1  Random Token Segmentation

1:function countSegments(s​t​a​r​tstart) ⊳\triangleright Cached (using memoization)

2:  if s​t​a​r​t=|t​o​k​e​n|start=|token| then

3:   return 11 ⊳\triangleright Reached end; valid segmentation

4:  end if

5:  t​o​t​a​l←0total\leftarrow 0

6:  for e​n​d←s​t​a​r​t+1end\leftarrow start+1 to |t​o​k​e​n||token| do

7:   substring←token[start:end]substring\leftarrow token[start:end]

8:   if s​u​b​s​t​r​i​n​g∈v​o​c​a​b​u​l​a​r​ysubstring\in vocabulary then

9:     t​o​t​a​l←t​o​t​a​l+countSegments​(e​n​d)total\leftarrow total+\textsc{countSegments}(end)

10:   end if

11:  end for

12:  return t​o​t​a​ltotal

13:end function

14:function buildSegments(s​t​a​r​tstart)

15:  if s​t​a​r​t=|t​o​k​e​n|start=|token| then

16:   return ∅\varnothing ⊳\triangleright Empty segmentation

17:  end if

18:  v​a​l​i​d​S​e​g​m​e​n​t​s←[]validSegments\leftarrow[]

19:  w​e​i​g​h​t​s←[]weights\leftarrow[]

20:  for e​n​d←s​t​a​r​t+1end\leftarrow start+1 to |t​o​k​e​n||token| do

21:   substring←token[start:end]substring\leftarrow token[start:end]

22:   if s​u​b​s​t​r​i​n​g∈v​o​c​a​b​u​l​a​r​ysubstring\in vocabulary then

23:     s​e​g​C​o​u​n​t←countSegments​(e​n​d)segCount\leftarrow\textsc{countSegments}(end)

24:     if s​e​g​C​o​u​n​t>0segCount>0 then

25:      Append s​u​b​s​t​r​i​n​gsubstring to v​a​l​i​d​S​e​g​m​e​n​t​svalidSegments

26:      Append s​e​g​C​o​u​n​tsegCount to w​e​i​g​h​t​sweights

27:     end if

28:   end if

29:  end for

30:  if v​a​l​i​d​S​e​g​m​e​n​t​svalidSegments is empty then

31:   return ∅\varnothing

32:  end if

33:  c​h​o​s​e​n​S​e​g​m​e​n​t←chosenSegment\leftarrow weightedRandomChoice(v​a​l​i​d​S​e​g​m​e​n​t​s,w​e​i​g​h​t​s)(validSegments,weights)

34:  return [chosenSegment]∥buildSegments(start+|chosenSegment|)[chosenSegment]\;\|\;\textsc{buildSegments}(start+|chosenSegment|)
⊳\triangleright Concatenate chosen segment with segmentation of remaining token

35:end function

36:procedure SegmentToken(t​o​k​e​n,v​o​c​a​b​u​l​a​r​ytoken,vocabulary)

37:  if countSegments​(0)=0\textsc{countSegments}(0)=0 then

38:   return ∅\varnothing ⊳\triangleright No valid segmentation exists

39:  else

40:   return buildSegments(0)

41:  end if

42:end procedure

## Appendix B Evaluation Details

### B.1 General benchmarks

For short-answer benchmarks, the system prompt is:

```
You are a helpful assistant.
```

For multiple-choice benchmarks, the system prompt is:

```
You are a helpful assistant. For the following multiple choice questions,
return the answer only, without any additional reasoning or explanation.
```

##### MATH

MATH is a dataset composed of fairly difficult, competition level math problems [[23](#bib.bib23)]. The test set is composed of short answer problem that describe some scenario and asks the model to output a mathematically correct answer.

##### GSM8K

GSM8K is a dataset consisting of relatively simple math questions that would appear in grade school math exams [[14](#bib.bib14)]. For GSM8K, the evaluations were done in the same manner as MATH.

##### MMLU

MMLU is a benchmarks comprising of multiple choice questions from a wide variety of subjects. [[22](#bib.bib22)] We sampled 500 questions from MMLU for our evaluation. We instructed the model to only output one answer to each question without any explanation.

##### Alpaca Eval

Alpaca Eval is an evaluation benchmark where generations from language models against given prompts are compared and judged by an annotator model. [[16](#bib.bib16)] The metric used was raw winrate of the perturbed model as judged by a language model. The annotator we used was alpaca\_eval\_gpt4, which has been shown to have the highest Spearman and Pearson correlation coefficient with human annotators.

##### ARC Challenge and ARC Easy

Contains multiple choice questions with four options each, taken from grade school science exams [[13](#bib.bib13)]. ARC Easy is tests basic science knowledge while ARC Challenge requires some procedural reasoning.

##### BoolQ

Contains true or false questions along with a context passage that provides the answer to the question. [[11](#bib.bib11)]

##### CommonsenseQA

Contains multiple choice questions with five options each that requires common sense knowledge to answer. [[55](#bib.bib55)]

##### COPA

Contains multiple choice questions with two options each that tests knowledge of cause and effect. [[48](#bib.bib48)]

##### CUTE

Contains questions that require the model to manipulate sentence-level, word-level, and character-level structure for strings. [[17](#bib.bib17)]

##### DROP

contains questions that potentially require reasoning multiple pieces of information present in a given passage. [[15](#bib.bib15)]

##### HellaSwag

contains multiples choice questions with four options each that asks for the most natural continuation to some given context. [[64](#bib.bib64)]

##### JeopardyQA

contains short answer questions from the “Jeopardy!” game show. [[29](#bib.bib29)]

##### OpenbookQA

contains multiple choice questions with four options each that require some multi-step and common sense reasoning. [[39](#bib.bib39)]

##### PIQA

contains multiple choice questions that require reasoning about the physical world. [[6](#bib.bib6)]

##### TriviaQA

contains short answer questions that requires knowledge of the world. [[28](#bib.bib28)]

##### Winograd

contains multiple choice questions with two options that asks to determine what a pronoun might refer to. Answering these questions require knowledge of commen sense and surrounding context. [[34](#bib.bib34)]

##### Winogrande

contains questions in the same format of Winograd but there are more questions and the questions are harder. [[49](#bib.bib49)]

##### TOFU

contains general short answer questions that tests the model’s ability to process world knowledge. This is the retain set of the task of fictitious unlearning dataset.
[[37](#bib.bib37)]

##### WikidataQA

require models to complete factual statements. [[5](#bib.bib5)]

### B.2 Constructed Benchmarks

Table 5: System prompt for tasks in [§3](#S3 "3 Can non-canonical tokenizations improve model performance? ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations"). See [Table 2](#S3.T2 "Table 2 ‣ 3 Can non-canonical tokenizations improve model performance? ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations") for example instructions.

|  |
| --- |
| Counting characters: You are a helpful assistant. The following prompt will ask you to return a sequence of words. Only return the sequence, separated by spaces. Do not provide any additional text or explanation. |
| Code Description: You are a programming assistant trained to analyze and interpret code snippets. When provided with a code snippet and a set of answer choices (A, B, C, or D), your task is to evaluate the code, determine its behavior, and select the answer that best describes this behavior. Your response must be a single letter: A, B, C, or D. Do not provide explanations or additional text unless explicitly requested. |
| Arithmetic: You are a computational assistant trained to evaluate arithmetic operations. When provided with an arithmetic expression, calculate the result and round it to the nearest integer. Respond only with the rounded result, without any additional text or explanation. |

In this section, we provide more detail on how datasets we use in [§3](#S3 "3 Can non-canonical tokenizations improve model performance? ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations") are constructed.

##### Count Characters Task

The prompt asks the model to count the number of occurrences of a given character in a 10-character word; we always use the most frequently occurring character.
Evaluation was done, similar to GSM and MATH, by finding the last number in the generated response.
Generations without any numbers are considered incorrect.

##### Generate Acronym Task

The model is asked to generate a sequence of words whose first letters form a randomly sampled five character string.
For evaluation, we take the first character of each whitespace-delimited word and check if it matches the desired acronym.

##### Codeline Description Task

The model is asked to comprehend a piece of code and choose the best description from four options.

##### Arithmetic Task

The model is asked to perform addition or subtraction with 10 digit numbers. We use regex to extract numbers from the generation, which are then compared to the ground truth answer.

### B.3 Metrics of generation quality

Here we provide additional details on the metrics defined in [§4.1](#S4.SS1 "4.1 When does robustness appear in model training? ‣ 4 Investigating the Source of Robustness ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations").

##### Spelling

We use the top 10000 most frequently appearing English words in Google’s trillion word corpus.
We only consider words with more than one character.
This is because sometimes base models will repeatedly generate the same letter, and since all English letters are in the word list, the generation would receive a high score.

##### Grammaticality

One drawback with this evaluation method is that oftentimes the model would repeat the same letter over and over again, or start counting numbers.
In both of these cases, there are no detected grammar mistakes, however they are still obviously gibberish. Therefore, we only calculate grammaticality scores for generations that receive a score ≥0.5\geq 0.5 on spelling; otherwise, we give it a grammaticality score of 0.

##### Win rate

Similar to evaluation in [§2](#S2 "2 Language Models are Robust to Non-Canonical Tokenizations ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations"), we also used alpaca\_eval\_gpt4 as the evaluator and report raw win rate.
In [4.1](#S4.SS1 "4.1 When does robustness appear in model training? ‣ 4 Investigating the Source of Robustness ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations"), the win rate is calculated against generations conditioned on input with canonical tokenization. In [4.2](#S4.SS2 "4.2 Why do instruction-tuned models become robust? ‣ 4 Investigating the Source of Robustness ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations"), the win rate is against generations from the No Ablation setting when also given character-level tokenization.
By construction, the win rate of the No Ablation setting itself is 50%.

Table 6: Data format of ablations in [§4.2](#S4.SS2 "4.2 Why do instruction-tuned models become robust? ‣ 4 Investigating the Source of Robustness ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations").

|  |
| --- |
| No ablation: <|user|>Provide a detailed analysis of Candace Parker’s defensive techniques in her recent games, excluding the words "aggressive" and "blocking", in the format of a sports commentary script. <|assistant|>[Sports Commentary Script] |
| [Opening Scene... |
| QA Template: Question: Provide a detailed analysis of Candace Parker’s defensive techniques in her recent games, excluding the words "aggressive" and "blocking", in the format of a sports commentary script. Answer: [Sports Commentary Script] |
| [Opening Scene... |
| Removing the chat template: Provide a detailed analysis of Candace Parker’s defensive techniques in her recent games, excluding the words "aggressive" and "blocking", in the format of a sports commentary script. [Sports Commentary Script] |
| [Opening Scene... |
| Removing the instruction: <|user|>[Sports Commentary Script] |
| [Opening Scene: A packed basketball arena, with fans eagerly awaiting the analysis of Candace Parker’s recent performances on the court.] |
| Commentator 1: Welcome back, basketball fans! <|assistant|>Tonight, we’re diving into the defensive prowess of Candace Parker... |

### B.4 Ablation Settings

For ablations on the data format, see examples of formatted data in [Table 6](#A2.T6 "Table 6 ‣ Win rate ‣ B.3 Metrics of generation quality ‣ Appendix B Evaluation Details ‣ Broken Tokens? Your Language Model can Secretly Handle No n - Ca noni cal Tokenizations").
Our finetuning code was forked from [allenai/open-instruct](https://github.com/allenai/open-instruct).

### B.5 Disentangling understanding from generation

For these tasks, we use 500 words randomly sampled from Google’s 10000 English word list555<https://github.com/first20hours/google-10000-english/blob/master/google-10000-english.txt>.

##### Word Repeat

An example prompt is shown below.

```
Repeat each word directly, while correcting any typos.

Question: guarantees
Answer: guarantees

Question: revelation (character-level tokenization)
Answer:
```

##### Identifying Misspellings

We obtain the misspelled word by randomly adding, removing, or substituting a single character from the word.
An example prompt is shown below.

```
Question: Which of the two words contains a misspelling? Respond directly
with the answer option.

Question:

A. guarantees
B. garantees

Answer: B

{9 more in context examples}

Question:

A. farmer (character-level tokenization)
B. farme (canonical tokenization)
```

[◄](/html/2506.19003)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2506.19004)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2506.19004)
[View original  
on arXiv](https://arxiv.org/abs/2506.19004)[►](/html/2506.19005)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Sat Jul 5 19:16:41 2025 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
