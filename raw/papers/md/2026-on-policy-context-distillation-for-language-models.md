---
arxiv: '2602.12275'
authors:
- Tianzhu Ye
- Li Dong
- Xun Wu
- Shaohan Huang
- Furu Wei
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: On-Policy Context Distillation for Language Models
url: https://arxiv.org/abs/2602.12275
year: 2026
---

# On-Policy Context Distillation for Language Models

Tianzhu Ye        Li Dong11footnotemark: 1
  
Xun Wu      Shaohan Huang      Furu Wei
  
 Microsoft Research
  
 <https://aka.ms/GeneralAI>
 Equal contribution.

###### Abstract

Context distillation enables language models to internalize in-context knowledge into their parameters. In our work, we propose On-Policy Context Distillation (OPCD), a framework that bridges on-policy distillation with context distillation by training a student model on its own generated trajectories while minimizing reverse Kullback-Leibler divergence against a context-conditioned teacher. We demonstrate the effectiveness of OPCD on two important applications: experiential knowledge distillation, where models extract and consolidate transferable knowledge from their historical solution traces, and system prompt distillation, where models internalize beneficial behaviors encoded in optimized prompts. Across mathematical reasoning, text-based games, and domain-specific tasks, OPCD consistently outperforms baseline methods, achieving higher task accuracy while better preserving out-of-distribution capabilities. We further show that OPCD enables effective cross-size distillation, where smaller student models can internalize experiential knowledge from larger teachers.

## 1 Introduction

Large language models (LLMs) exhibit remarkable in-context learning capabilities, allowing them to adapt their behavior based on the information provided in the prompt without parameter updates [[4](#bib.bib48 "Language models are few-shot learners"), [7](#bib.bib2 "A survey on in-context learning")].
By prepending instructions, few-shot demonstrations, or retrieved documents to the input, users can steer model behavior without updating parameters.
However, in-context knowledge is transient. In other words, valuable insights generated or retrieved during a session are lost once the context is reset, requiring the model to “re-learn” from the prompt every time.

A natural question arises: Can we internalize transient in-context knowledge into the model’s permanent parameters?
Context distillation [[2](#bib.bib307 "A general language assistant as a laboratory for alignment"), [19](#bib.bib306 "Learning by distilling context")] addresses this by training a student model to mimic the behavior of a context-conditioned teacher, effectively compressing the context into the student’s weights. Once trained, the student can reproduce the teacher’s context-aware behavior without requiring the context at inference time, effectively “internalizing” the context.

Despite its appeal, existing context distillation methods face a fundamental limitation: they rely on off-policy training with forward Kullback-Leibler (KL) divergence minimization on a fixed dataset. However, this off-policy approach suffers from distinct drawbacks.
First, it induces exposure bias, where the student is trained on teacher-generated or ground-truth data but must generate its own autoregressive sequences at inference time.
Second, minimizing forward KL encourages mode-covering behavior, causing the student to assign probability mass to all teacher-generated tokens, often resulting in “hallucinations” or overly broad distributions when the student lacks the capacity to fully model the teacher’s complex, context-aware distribution [[9](#bib.bib312 "MiniLLM: on-policy distillation of large language models")].

In this work, we propose On-Policy Context Distillation (OPCD), a method that bridges on-policy distillation [[9](#bib.bib312 "MiniLLM: on-policy distillation of large language models"), [12](#bib.bib314 "On-policy distillation"), [1](#bib.bib27 "On-policy distillation of language models: learning from self-generated mistakes")] with context distillation to internalize in-context knowledge more effectively. The key is that the student model learns from its own generation trajectories rather than those of the teacher.
Specifically, OPCD samples responses from the student model (without context), then computes the reverse KL divergence between the student’s token distributions and those of a context-conditioned teacher at each position along the student’s trajectory. This on-policy approach ensures that the student learns to correct its own mistakes and align its generation distribution with the teacher’s context-aware behavior.

We demonstrate the effectiveness of OPCD on two important applications. First, we introduce experiential knowledge distillation, where a model extracts transferable knowledge from its historical solution traces and internalizes this accumulated experience into its parameters. We show that models can progressively improve by accumulating experiential knowledge from solved problems, and that OPCD successfully consolidates this knowledge without requiring the extended context at inference time. Second, we apply OPCD to system prompt distillation, enabling models to internalize beneficial behaviors encoded in externally optimized prompts for specialized tasks such as medical question answering and safety classification.

Our experiments span mathematical reasoning, text-based games, and domain-specific tasks with optimized system prompts. Across all settings, OPCD consistently outperforms baseline methods, achieving higher task accuracy while better preserving out-of-distribution capabilities and relieving catastrophic forgetting. We further demonstrate that OPCD enables effective teacher-student distillation, where smaller student models can internalize experiential knowledge from larger teachers. In contrast, directly injecting teacher-generated knowledge into smaller model contexts degrades performance.

## 2 Related Work

##### Context Distillation

Context distillation compresses in-context knowledge into model parameters, eliminating the inference overhead of context processing [[2](#bib.bib307 "A general language assistant as a laboratory for alignment"), [19](#bib.bib306 "Learning by distilling context")]. While prior methods rely on off-policy forward KL minimization, they suffer from exposure bias due to the mismatch between teacher-guided training and autoregressive inference. In contrast, our method employs on-policy sampling, allowing the student to learn from its own trajectories and bridging the gap between training and deployment distributions.

##### On-Policy Distillation

On-policy distillation methods [[9](#bib.bib312 "MiniLLM: on-policy distillation of large language models"), [12](#bib.bib314 "On-policy distillation"), [1](#bib.bib27 "On-policy distillation of language models: learning from self-generated mistakes")] mitigate exposure bias by training students on their own generated trajectories. By minimizing the reverse KL divergence [[9](#bib.bib312 "MiniLLM: on-policy distillation of large language models")], these approaches promote mode-seeking behavior, compelling the student to focus on the teacher’s high-likelihood regions and avoiding the mode-averaging issues of standard forward KL.
[[23](#bib.bib305 "Black-box on-policy distillation of large language models")] has extended this to black-box settings.
Our work adapts the on-policy distillation paradigm specifically for the problem of context internalization, allowing a model to efficiently consolidate transient in-context knowledge into its permanent weights.

##### Self-Distillation

Recent research has increasingly explored self-distillation mechanisms in which a model improves by learning from its own output or a conditioned version of itself.
[[25](#bib.bib315 "STar: bootstrapping reasoning with reasoning")] demonstrates that a model can bootstrap its reasoning capabilities by iteratively training self-generated solutions that lead to correct answers.
Closer to our approach, concurrent works [[26](#bib.bib308 "Self-distilled reasoner: on-policy self-distillation for large language models"), [11](#bib.bib311 "Reinforcement learning via self-distillation"), [17](#bib.bib310 "Self-distillation enables continual learning"), [15](#bib.bib309 "Privileged information distillation for language models")] utilize on-policy self-distillation conditioning on privileged information (such as ground-truth solutions, environmental feedback, or demonstrations) to supervise the model sharing the same weights.
In comparison, the teacher model in our framework can be a different model or the same model, and it can be updated simultaneously or kept frozen. This allows us to adapt to various training scenarios and objectives, whereas self-distillation methods typically focus on a single model learning from itself without the flexibility of incorporating external knowledge or different training dynamics.

## 3 Method

!(/html/2602.12275/assets/x1.png)

Figure 1: Overview of on-policy context distillation (OPCD). Given a context and an input prompt, the student model generates a response without the context. It is then trained to minimize the reverse KL divergence to the teacher model that conditions on the context. The student internalizes the contextual information with on-policy learning.

We present On-Policy Context Distillation (OPCD), a method that internalizes in-context knowledge into model parameters by bridging on-policy distillation [[9](#bib.bib312 "MiniLLM: on-policy distillation of large language models"), [12](#bib.bib314 "On-policy distillation"), [1](#bib.bib27 "On-policy distillation of language models: learning from self-generated mistakes")] with context distillation [[2](#bib.bib307 "A general language assistant as a laboratory for alignment"), [19](#bib.bib306 "Learning by distilling context")]. Our approach enables models to consolidate contextual information (such as experience knowledge or instructions) directly into their weights.
The fundamental goal is to compress a specific prompt or context cc into the parameters θ\theta of a student model πθ\pi\_{\theta}, such that the student can replicate the behavior of a context-aware teacher πteacher\pi\_{\mathrm{teacher}} without requiring the context at inference time.

Formally, given an input xx, we minimize the divergence between the student distribution πθ(⋅∣x)\pi\_{\theta}(\cdot\mid x) and the teacher distribution πteacher(⋅∣c,x)\pi\_{\mathrm{teacher}}(\cdot\mid c,x), where the teacher has access to the guiding context cc prepended to the input.
OPCD optimizes the reverse Kullback-Leibler (KL) divergence [[9](#bib.bib312 "MiniLLM: on-policy distillation of large language models")] between the student and teacher distributions using on-policy sampling.

We decompose sequence-level divergence into the sum of token-level divergences. The loss function is defined as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ(θ)=𝔼(x,c)∼𝒟,y∼πθ(⋅∣x)[1|y|∑t=1|y|DKL(πθ(⋅∣x,y<t)∥πteacher(⋅∣c,x,y<t))],\mathcal{L}(\theta)=\mathbb{E}\_{(x,c)\sim\mathcal{D},y\sim\pi\_{\theta}(\cdot\mid x)}\left[\frac{1}{|y|}\sum\_{t=1}^{|y|}{D\_{\mathrm{KL}}\left(\pi\_{\theta}(\cdot\mid x,y\_{<t})\|\pi\_{\mathrm{teacher}}(\cdot\mid c,x,y\_{<t})\right)}\right], |  | (1) |

where cc is the in-context knowledge that we aim to internalize, 𝒟\mathcal{D} represents training data, and yy is sampled from the student model.

The token-level reverse KL divergence is computed via:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | DKL(πθ(⋅∣x,y<t)∥πteacher(⋅∣c,x,y<t))\displaystyle D\_{\mathrm{KL}}\left(\pi\_{\theta}(\cdot\mid x,y\_{<t})\|\pi\_{\mathrm{teacher}}(\cdot\mid c,x,y\_{<t})\right) |  | (2) |
|  | =\displaystyle= | 𝔼yt′∼πθ(⋅∣x,y<t)​[log⁡πθ​(yt′∣x,y<t)πteacher​(yt′∣c,x,y<t)]\displaystyle\ \mathbb{E}\_{y\_{t}^{\prime}\sim\pi\_{\theta}(\cdot\mid x,y\_{<t})}\left[\log\frac{\pi\_{\theta}(y\_{t}^{\prime}\mid x,y\_{<t})}{\pi\_{\mathrm{teacher}}(y\_{t}^{\prime}\mid c,x,y\_{<t})}\right] |  |
|  | =\displaystyle= | ∑yt′∈𝒱πθ​(yt′∣x,y<t)​(log⁡πθ​(yt′∣x,y<t)−log⁡πteacher​(yt′∣c,x,y<t))\displaystyle\sum\_{y\_{t}^{\prime}\in\mathcal{V}}\pi\_{\theta}(y\_{t}^{\prime}\mid x,y\_{<t}){\left(\log\pi\_{\theta}(y\_{t}^{\prime}\mid x,y\_{<t})-\log\pi\_{\mathrm{teacher}}(y\_{t}^{\prime}\mid c,x,y\_{<t})\right)} |  |

where 𝒱\mathcal{V} is the vocabulary.
In our implementation, we approximate the analytic KL divergence by restricting the summation to the top-kk tokens predicted by the student model, i.e., 𝒱top−k\mathcal{V}\_{\operatorname{top-k}} is the set of kk tokens with the highest probability under πθ(⋅∣x,y<t)\pi\_{\theta}(\cdot\mid x,y\_{<t}).

By minimizing the reverse KL divergence via on-policy sampling, OPCD encourages mode-seeking behavior: the student focuses on generating tokens that are high-probability under the teacher’s distribution, ignoring the long tail of less relevant possibilities.
Intuitively, if the student generates a token that the teacher (conditioned on context cc) considers highly probable compared to the student’s current belief, encouraging the student to increase the probability of that token.
Conversely, if the student assigns a high probability to a token that the teacher considers unlikely, the behavior is suppressed.
The student πθ\pi\_{\theta} progressively aligns its generation trajectory with the context-aware teacher πteacher\pi\_{\text{teacher}}, effectively internalizing the context cc in its parameters.

Algorithm [1](#alg1 "Algorithm 1 ‣ 3 Method ‣ On-Policy Context Distillation for Language Models") presents the pseudocode for OPCD training.
The training process follows an on-policy rollout mechanism. In each training step, we sample input xx from the training data and let the student model πθ\pi\_{\theta} generate complete response trajectories yy. Importantly, these trajectories are generated without context cc. Once the trajectory is formed, we evaluate it using the teacher model πteacher\pi\_{\mathrm{teacher}}, which processes the concatenated sequence [c;x;y][c;x;y] to compute the target probabilities.

Algorithm 1  OPCD: On-Policy Context Distillation

Training data 𝒟={(x,c)}\mathcal{D}=\{(x,c)\}, where xx is input, and cc is in-context knowledge that we are internalizing; Student LLM πθ\pi\_{\theta}; Teacher LLM πteacher\pi\_{\mathrm{teacher}}

Trained student model πθ\pi\_{\theta}

for each batch (x,c)∼𝒟(x,c)\sim\mathcal{D} do

// On-policy rollout (student model without context cc)

Sample response y∼πθ(⋅∣x)y\sim\pi\_{\theta}(\cdot\mid x)

// Compute token-level reverse KL according to Equation ([2](#S3.E2 "Equation 2 ‣ 3 Method ‣ On-Policy Context Distillation for Language Models"))

DKL(t)←∑yt′∈𝒱πθ​(yt′∣x,y<t)​(log⁡πθ​(yt′∣x,y<t)−log⁡πteacher​(yt′∣c,x,y<t))D\_{\mathrm{KL}}^{(t)}\leftarrow\sum\_{y\_{t}^{\prime}\in\mathcal{V}}\pi\_{\theta}(y\_{t}^{\prime}\mid x,y\_{<t})\left(\log\pi\_{\theta}(y\_{t}^{\prime}\mid x,y\_{<t})-\log\pi\_{\mathrm{teacher}}(y\_{t}^{\prime}\mid c,x,y\_{<t})\right)

ℒ​(θ)←1|y|​∑t=1|y|DKL(t)\mathcal{L}(\theta)\leftarrow\frac{1}{|y|}\sum\_{t=1}^{|y|}{D\_{\mathrm{KL}}^{(t)}}

// Update student model according to Equation ([1](#S3.E1 "Equation 1 ‣ 3 Method ‣ On-Policy Context Distillation for Language Models"))

Update θ\theta by minimizing ℒ​(θ)\mathcal{L}(\theta)

end for

return πθ\pi\_{\theta}

### 3.1 Teacher Model Configurations

Our framework allows for flexibility in the choice of the teacher model. We consider the following two configurations.

##### Teacher-Student Distillation (πteacher≠πθ\pi\_{\mathrm{teacher}}\neq\pi\_{\theta})

First, the teacher model can be a larger or more capable model than the student. In this scenario, the student benefits from both the in-context knowledge and the superior capabilities of the larger teacher model.
Second, the teacher and student models are initialized from the same weights but are not updated simultaneously. The teacher receives additional contextual information cc. The parameters of the teacher model can remain frozen or undergo periodic updates, making training more stable. Teacher-student distillation is also our default configuration.

##### Self-Distillation (πteacher=πθ\pi\_{\mathrm{teacher}}=\pi\_{\theta})

The teacher and the student share the same underlying model weights and are updated simultaneously.
The divergence arises solely from the input: the teacher sees [c;x][c;x] while the student sees only xx.
This allows a model to “teach itself” [[11](#bib.bib311 "Reinforcement learning via self-distillation"), [26](#bib.bib308 "Self-distilled reasoner: on-policy self-distillation for large language models"), [15](#bib.bib309 "Privileged information distillation for language models"), [17](#bib.bib310 "Self-distillation enables continual learning")] to internalize a prompt.

## 4 Experiments

### 4.1 Evaluation Tasks

#### 4.1.1 Experiential Knowledge Distillation

We introduce an experiential knowledge distillation task in which a language model extracts transferable experiential knowledge from test-time solution traces as context cc for future problems, eventually internalizing this knowledge via on-policy context distillation 111Different from Reinforcement Learning with Verifiable Rewards (RLVR), experiential knowledge distillation at test time does not rely on ground-truth labels. In the math setting, no labels are needed, and in the game setting, the model interacts with the environment.. The process consists of three primary stages:

1. 1.

   Experiential Knowledge Extraction. The model is given problems and produces solution traces to them. Conditioning on each problem and its self-generated solution (notably without ground-truth labels), the model is prompted to generate experiential knowledge learned from it.
2. 2.

   Experiential Knowledge Accumulation. Experiential knowledge from different problems is combined together to form an experiential knowledge context cc for future problems. Prepending experiential knowledge context on new problems can improve the model’s performance.
3. 3.

   Experiential Knowledge Consolidation. We apply on-policy context distillation to transition experiential knowledge from the context space into the student model’s weights. This allows the student model to internalize the experience from the teacher without the overhead of extended context.

In our experiments, we use itemized experiential knowledge formatted as “-- EXPERIENCE ITEM:” and we directly concatenated experiential knowledge from different problems in the experiential knowledge accumulation step.
Refer to Appendix [A.1](#A1.SS1 "A.1 Prompt Templates ‣ Appendix A Experiential Knowledge Distillation Details ‣ On-Policy Context Distillation for Language Models") for prompt templates for the three stages.

##### Datasets

For experiential knowledge distillation task, we train our models on three datasets: English math problems from DAPO-Math-17K [[24](#bib.bib10 "Dapo: an open-source llm reinforcement learning system at scale")] and two text-based game environments, Frozen Lake and Sokoban, implemented in TextArena [[10](#bib.bib12 "TextArena")].
DAPO-Math-17K contains approximately 14K verifiable English math problems, each with a numerical answer.
Frozen Lake is a grid-based navigation task where the model must reach a goal while avoiding holes.
Sokoban is a spatial reasoning puzzle where the model must push a box to a designated target without falling into holes or becoming trapped against walls.
TextArena provides textual descriptions of the current game state at each step. The language model interacts with the game environments in a multi-turn setting. Detailed descriptions of datasets are provided in Appendix [A.2](#A1.SS2 "A.2 Dataset Details ‣ Appendix A Experiential Knowledge Distillation Details ‣ On-Policy Context Distillation for Language Models").

#### 4.1.2 System Prompt Distillation

System prompts are widely used to steer LLM behavior toward desired objectives, such as enhancing domain expertise or enforcing safety constraints. However, prepending system prompts at inference time increases computational overhead and latency, particularly for lengthy prompts. We distill system prompts as context cc into the student model, enabling it to internalize beneficial behaviors encoded in externally optimized prompts without requiring explicit prompting during deployment.

##### Datasets

We use system prompts optimized for medical and safety tasks from MetaSPO [[6](#bib.bib6 "System prompt optimization with meta-learning")]. For medical system prompt, we adopt MedMCQA [[14](#bib.bib5 "Medmcqa: a large-scale multi-subject multi-choice dataset for medical domain question answering")] dataset and hold out 500 samples for testing.
For safety system prompt, we combine Tweet Eval [[3](#bib.bib4 "TweetEval: unified benchmark and comparative evaluation for tweet classification")], Hatecheck [[16](#bib.bib3 "HateCheck: functional tests for hate speech detection models")], and Ethos [[13](#bib.bib1 "ETHOS: a multi-label hate speech detection dataset")] datasets, and similarly reserve 500 samples for testing. Detailed system prompts are provided in Appendix [B.1](#A2.SS1 "B.1 System Prompts ‣ Appendix B System Prompt Distillation Details ‣ On-Policy Context Distillation for Language Models").

### 4.2 Setup

##### Models

For experiential knowledge distillation task, we use thinking mode of Qwen3-8B [[21](#bib.bib26 "Qwen3 technical report")] as teacher to generate traces and extract experiential knowledge on a validation split from DAPO for math problems. We train Qwen3-8B, Qwen3-4B, and Qwen3-1.7B with thinking mode as students using OPCD.
For Frozen Lake, we use the thinking mode of Qwen3-1.7B as the teacher and the student.
For Sokoban, we use the non-thinking model Qwen3-4B-Instruct-2507 as the teacher and the student.
For system prompt distillation task, we use Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct [[22](#bib.bib34 "Qwen2.5 Technical Report")], as well as Llama-3.1-8B-Instruct and Llama-3.2-3B-Instruct [[8](#bib.bib33 "The llama 3 herd of models")].

##### Training

For experiential knowledge distillation task, we sample problems from the validation split to construct a pool of 300 experiential knowledge contexts (30 accumulation steps for 10 times). The maximum experiential knowledge length is set to 16384 tokens for math and 8192 tokens for text games.
For the test-time experiential knowledge distillation setting, we randomly select experiential knowledge from this pool of 300 for further OPCD training. This setting emulates a test-time experiential knowledge distillation scenario in which no ground-truth labels are available and the quality of experiential knowledge is not pre-evaluated. For the filtered experiential knowledge distillation setting, we score each candidate experiential knowledge by prepending it to new problems and evaluating performance on 1000 math validation examples or 128 text-game validation examples. The highest-scoring experiential knowledge is then selected for subsequent OPCD training.

We then distill the student model on training split of math and text-game datasets using the selected experiential knowledge context for 50 steps with a batch size of 128. For math, we set the maximum response length to 16384 tokens. For text games, the model interacts with the game environment for up to 5 rounds, each with a maximum response length of 1024 tokens.
For system prompt distillation task, we distill the student model on the training splits of the medical and safety datasets, conditioning on the corresponding system prompts. Training runs for 50 steps with batch size of 128. The maximum generated response length is set to 512 tokens. More training details can be found in Appendix [A.3](#A1.SS3 "A.3 Training Details ‣ Appendix A Experiential Knowledge Distillation Details ‣ On-Policy Context Distillation for Language Models") and Appendix [B.2](#A2.SS2 "B.2 Training Details ‣ Appendix B System Prompt Distillation Details ‣ On-Policy Context Distillation for Language Models").

##### Evaluation

For experiential knowledge distillation, we report accuracy on the test split of the math dataset (1000 samples) and text-game datasets (128 samples) as the metric for in-distribution performance. For out-of-distribution evaluation, we report prompt-level strict accuracy on IF-Eval [[27](#bib.bib9 "Instruction-following evaluation for large language models")]. For system prompt distillation, we report test accuracy on a 500-sample test split.
We compare against the context-distillation baseline [[2](#bib.bib307 "A general language assistant as a laboratory for alignment"), [19](#bib.bib306 "Learning by distilling context")], which trains on off-policy data generated by the teacher and uses forward KL minimization.

### 4.3 Results

##### Experiential Knowledge Consolidation

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Model | Task | Method | Accuracy | |  | | --- | | IF-Eval | | (Out-of-Distribution) | |
| Qwen3-8B | Math | Base Model | 75.0 | 81.3 |
| In-Context | 77.6 ±\pm 1.1 | — |
| Context Distill. | 78.5 ±\pm 0.5 | 81.2 ±\pm 0.2 |
| OPCD | 79.7 ±\pm 0.5 | 81.7 ±\pm 0.4 |
| Qwen3-1.7B | Frozen Lake | Base Model | 6.3 | 67.3 |
| In-Context | 20.2 ±\pm 2.2 | — |
| Context Distill. | 22.9 ±\pm 4.0 | 65.1 ±\pm 0.5 |
| OPCD | 26.5 ±\pm 6.4 | 67.1 ±\pm 0.5 |

Table 1: Results of test-time experiential knowledge consolidation. OPCD consistently outperforms off-policy context distillation on test accuracy and OOD task performance.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Model | Task | Method | Accuracy | |  | | --- | | IF-Eval | | (Out-of-Distribution) | |
| Qwen3-8B | Math | Base Model | 75.0 | 81.3 |
| In-Context | 79.0 | — |
| Context Distill. | 79.5 | 80.4 |
| OPCD | 80.9 | 80.8 |
| Qwen3-1.7B | Frozen Lake | Base Model | 6.3 | 67.3 |
| In-Context | 31.4 | — |
| Context Distill. | 35.2 | 65.4 |
| OPCD | 38.3 | 66.7 |
| Qwen3-4B-Ins | Sokoban | Base Model | 9.4 | 82.8 |
| In-Context | 48.4 | — |
| Context Distill. | 51.6 | 82.3 |
| OPCD | 53.9 | 82.4 |

Table 2: Results of filtered experiential knowledge consolidation. OPCD consistently outperforms off-policy context distillation on test accuracy and OOD task performance on math and text-games.

We present experiential knowledge consolidation results in [Tables˜1](#S4.T1 "In Experiential Knowledge Consolidation ‣ 4.3 Results ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models") and [2](#S4.T2 "Table 2 ‣ Experiential Knowledge Consolidation ‣ 4.3 Results ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models"). In all experiments, the teacher and student use the same model size, and we use teacher-student distillation where the teacher is frozen. For the test-time experiential knowledge setting, we sample three random experiential knowledge contexts after ten steps of accumulation from the knowledge pool. We compare OPCD against: the base model without experiential knowledge, the base model with experiential knowledge provided in context (denoted as In-Context), and context distillation [[2](#bib.bib307 "A general language assistant as a laboratory for alignment"), [19](#bib.bib306 "Learning by distilling context")] which is off-policy.

As shown in [Tables˜1](#S4.T1 "In Experiential Knowledge Consolidation ‣ 4.3 Results ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models") and [2](#S4.T2 "Table 2 ‣ Experiential Knowledge Consolidation ‣ 4.3 Results ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models"), on both math and text-game tasks, OPCD outperforms the context distillation baseline, achieving higher test accuracy.
We also observe that OPCD can surpass the original model with experiential knowledge in the context. During consolidation, the student model is exposed to consolidation training data that the original model did not access (the experiential knowledge was extracted with validation data), thereby providing an additional learning signal.

##### System Prompt Distillation

|  |  |  |
| --- | --- | --- |
| Model | Method | Accuracy |
| Llama-3.1-8B-Ins | Base Model | 68.4 |
| In-Context | 72.2 |
| Context Distill. | 75.2 |
| OPCD | 76.7 |
| Llama-3.2-3B-Ins | Base Model | 59.4 |
| In-Context | 66.4 |
| Context Distill. | 71.0 |
| OPCD | 76.3 |
| Qwen2.5-7B-Ins | Base Model | 46.4 |
| In-Context | 52.6 |
| Context Distill. | 58.5 |
| OPCD | 62.3 |

Table 3: System prompt distillation on Medical.

|  |  |  |
| --- | --- | --- |
| Model | Method | Accuracy |
| Llama-3.1-8B-Ins | Base Model | 70.7 |
| In-Context | 75.3 |
| Context Distill. | 77.2 |
| OPCD | 79.6 |
| Llama-3.2-3B-Ins | Base Model | 30.7 |
| In-Context | 69.5 |
| Context Distill. | 83.3 |
| OPCD | 83.1 |
| Qwen2.5-7B-Ins | Base Model | 69.1 |
| In-Context | 72.7 |
| Context Distill. | 77.0 |
| OPCD | 78.1 |

Table 4: System prompt distillation on Safety.

We present medical system prompt distillation results in [Section˜4.3](#S4.SS3.SSS0.Px2 "System Prompt Distillation ‣ 4.3 Results ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models") and safety system prompt distillation in [Section˜4.3](#S4.SS3.SSS0.Px2 "System Prompt Distillation ‣ 4.3 Results ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models"). In all experiments, the teacher and student use the same model size, and we use teacher-student distillation where the teacher is frozen. OPCD outperforms the off-policy context distillation baseline in test accuracy across most configurations on the medical and safety system prompt distillation. We also observe on-policy training provides more stable improvements in training process compared to off-policy context distillation.

### 4.4 Effect of Model Size

!(/html/2602.12275/assets/x2.png)

Figure 2: OPCD consistently improves the evaluation results of smaller Qwen3 models using experiential knowledge distilled from a frozen Qwen3-8B teacher. In contrast, directly injecting this knowledge into smaller-model contexts degrades performance.

We scale student model sizes from Qwen3-1.7B to Qwen3-4B and Qwen3-8B using OPCD. Experiential knowledge is generated by Qwen3-8B. We also use it as a frozen teacher. As shown in [Figure˜2](#S4.F2 "In 4.4 Effect of Model Size ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models"), we report both OPCD results and the original Qwen3 baselines, and we also evaluate a direct injection of teacher-generated experiential knowledge into the contexts of Qwen3-1.7B and Qwen3-4B. We observe that OPCD consistently improves test accuracy across student model scales.

We find directly injecting experiential knowledge into the context of a smaller model can even degrade its performance (“In-Context” curve in [Figure˜2](#S4.F2 "In 4.4 Effect of Model Size ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models")). This suggests that on-policy alignment between experiential knowledge and the model that consumes it is also crucial.
While such knowledge is effective for the teacher model that collects it, it may not transfer reliably when placed directly into a different model’s context. Instead, integrating experiential knowledge within the teacher’s context and then applying OPCD to train the student can improve its performance.
In practice, the teacher model can be deployed in real environments and across diverse users to accumulate experiential knowledge at test time, which can then be periodically consolidated into the student.

### 4.5 On-Policy Context Distillation Mitigates Forgetting

!(/html/2602.12275/assets/x3.png)

Figure 3: Comparison of OPCD and off-policy context distillation on in-distribution (safety) and out-of-distribution (medical) tasks when distilling from safety system prompt. Left: accuracy on the safety test dataset. Right: accuracy on the medical test dataset. OPCD achieves superior in-distribution performance while mitigating forgetting on OOD tasks.

Compared to off-policy context distillation, OPCD samples from the student distribution, thereby mitigating forgetting on out-of-distribution (OOD) tasks. In [Tables˜1](#S4.T1 "In Experiential Knowledge Consolidation ‣ 4.3 Results ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models") and [2](#S4.T2 "Table 2 ‣ Experiential Knowledge Consolidation ‣ 4.3 Results ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models"), we evaluate models distilled with experiential knowledge on the OOD IF-Eval benchmark. OPCD achieves approximately 2% higher IF-Eval scores than the context distillation baseline on Frozen Lake.

In [Figure˜3](#S4.F3 "In 4.5 On-Policy Context Distillation Mitigates Forgetting ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models"), we distill the student Qwen2.5-3B-Instruct from the frozen teacher Qwen2.5-7B-Instruct using the safety system prompt. The left subfigure shows accuracy on the safety test dataset as an in-distribution performance measure, while the right subfigure reports accuracy on the medical test dataset for OOD evaluation. As shown, on-policy context distillation achieves higher in-distribution performance than off-policy context distillation. OPCD also maintains OOD performance compared to the initial student, surpassing the off-policy baseline by approximately 4 points. This finding is consistent with prior work demonstrating that on-policy training mitigates forgetting on OOD tasks [[18](#bib.bib8 "Rl’s razor: why online reinforcement learning forgets less"), [5](#bib.bib7 "Retaining by doing: the role of on-policy data in mitigating forgetting")].

### 4.6 Teacher-Student Distillation vs. Self-Distillation

|  |  |  |
| --- | --- | --- |
| Task | Configuration | Accuracy |
| Sokoban | Self | 18.8 |
| Teacher-Student | 53.9 |
| Medical | Self | 50.0 |
| Teacher-Student | 56.8 |

Table 5: Teacher-student-OPCD is more stable than self-OPCD and outperforms it.

We find teacher-student distillation is more stable than self-distillation and outperforms it. We compare two configurations of OPCD: (i) teacher-student distillation, our default configuration, which employs a frozen teacher model, and (ii) self-distillation, where the continuously updated model serves as both teacher and student. As shown in Table [5](#S4.T5 "Table 5 ‣ 4.6 Teacher-Student Distillation vs. Self-Distillation ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models"), we train experiential knowledge distillation with Qwen3-4B-Instruct-2507 on Sokoban and medical system prompt distillation with Qwen2.5-3B-Instruct. The teacher-student configuration substantially outperforms self-distillation on both tasks. Furthermore, we observe that the teacher-student configuration exhibits more stable training dynamics, whereas self-distillation can diverge after some training steps. We attribute this instability to the high variance introduced by using a continuously evolving model as the teacher during RL training, which destabilizes the learning signal222EMA of student parameters as teacher can alleviate the instability of self-distillation [[17](#bib.bib310 "Self-distillation enables continual learning"), [11](#bib.bib311 "Reinforcement learning via self-distillation")].. This finding also aligns with [Section˜4.4](#S4.SS4 "4.4 Effect of Model Size ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models"), reinforcing that on-policy alignment between experiential knowledge and the model that consumes it is crucial.

### 4.7 Importance of Learning from Experiential Knowledge

|  |  |  |
| --- | --- | --- |
| Model | Experience Type | Accuracy |
| Qwen3-8B | w/o Experience | 75.1 |
| Qwen3-8B | Raw Trace | 70.5 |
| Qwen3-8B | Knowledge | 77.4 |
|  | + OPCD | 79.7 |

Table 6: Using raw response traces from previous problems as experiential context degrades performance on the math validation dataset.

We show the necessity of extracting experiential knowledge in [Table˜6](#S4.T6 "In 4.7 Importance of Learning from Experiential Knowledge ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models"). We report averaged accuracy over experiential knowledge accumulation steps on math validation dataset after ten steps.
Simply prepending raw traces (previous problems and model outputs) as context during experience accumulation stage degrades accuracy, as seen in the “Raw Trace” row.
In contrast, using model to extract experiential knowledge from previous traces and prepending it leads to higher validation accuracy than the original model as in “Knowledge” row.

## 5 Conclusion

In this work, we introduced On-Policy Context Distillation (OPCD), a framework that enables language models to internalize in-context knowledge into their parameters through on-policy distillation. By minimizing the reverse KL divergence between a context-aware teacher and a context-free student, OPCD effectively consolidates transient contextual information, such as experiential knowledge and system prompts, into the model’s weights. Our experiments demonstrate that OPCD outperforms baseline methods across various tasks, including math problem solving and text-based games, while also enhancing out-of-distribution generalization. Furthermore, we showed that OPCD can scale effectively with model size and consistently improves performance when distilling optimized system prompts. Our work opens avenues for future research on continual accumulation of experiential knowledge, adaptive context selection strategies, and scaling OPCD to broader domains requiring persistent knowledge internalization.

## Acknowledgements

We are grateful to Qingxiu Dong for setting up the text-based games and to Yu Li, Yuxian Gu for discussions.

## References

* [1]
  R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. R. Garea, M. Geist, and O. Bachem (2024)
  On-policy distillation of language models: learning from self-generated mistakes.
  In The twelfth international conference on learning representations,
  Cited by: [§1](#S1.p4.1 "1 Introduction ‣ On-Policy Context Distillation for Language Models"),
  [§2](#S2.SS0.SSS0.Px2.p1.1 "On-Policy Distillation ‣ 2 Related Work ‣ On-Policy Context Distillation for Language Models"),
  [§3](#S3.p1.4 "3 Method ‣ On-Policy Context Distillation for Language Models").
* [2]
  A. Askell, Y. Bai, A. Chen, D. Drain, D. Ganguli, T. J. Henighan, A. Jones, N. Joseph, B. Mann, N. Dassarma, N. Elhage, Z. Hatfield-Dodds, D. Hernandez, J. Kernion, K. Ndousse, C. Olsson, D. Amodei, T. B. Brown, J. Clark, S. McCandlish, C. Olah, and J. Kaplan (2021)
  A general language assistant as a laboratory for alignment.
  ArXiv abs/2112.00861.
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ On-Policy Context Distillation for Language Models"),
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Context Distillation ‣ 2 Related Work ‣ On-Policy Context Distillation for Language Models"),
  [§3](#S3.p1.4 "3 Method ‣ On-Policy Context Distillation for Language Models"),
  [§4.2](#S4.SS2.SSS0.Px3.p1.1 "Evaluation ‣ 4.2 Setup ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models"),
  [§4.3](#S4.SS3.SSS0.Px1.p1.1 "Experiential Knowledge Consolidation ‣ 4.3 Results ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models").
* [3]
  F. Barbieri, J. Camacho-Collados, L. E. Anke, and L. Neves (2020)
  TweetEval: unified benchmark and comparative evaluation for tweet classification.
  In Findings of the association for computational linguistics: EMNLP 2020,
   pp. 1644–1650.
  Cited by: [§B.1](#A2.SS1.p1.1 "B.1 System Prompts ‣ Appendix B System Prompt Distillation Details ‣ On-Policy Context Distillation for Language Models"),
  [§4.1.2](#S4.SS1.SSS2.Px1.p1.1 "Datasets ‣ 4.1.2 System Prompt Distillation ‣ 4.1 Evaluation Tasks ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models").
* [4]
  T. Brown, B. Mann, N. Ryder, M. Subbiah, et al. (2020)
  Language models are few-shot learners.
  In Proceedings of NeurIPS,
  External Links: [Link](https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ On-Policy Context Distillation for Language Models").
* [5]
  H. Chen, N. Razin, K. Narasimhan, and D. Chen (2025)
  Retaining by doing: the role of on-policy data in mitigating forgetting.
  arXiv preprint arXiv:2510.18874.
  Cited by: [§4.5](#S4.SS5.p2.1 "4.5 On-Policy Context Distillation Mitigates Forgetting ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models").
* [6]
  Y. Choi, J. Baek, and S. J. Hwang (2025)
  System prompt optimization with meta-learning.
  arXiv preprint arXiv:2505.09666.
  Cited by: [§B.1](#A2.SS1.p1.1 "B.1 System Prompts ‣ Appendix B System Prompt Distillation Details ‣ On-Policy Context Distillation for Language Models"),
  [§4.1.2](#S4.SS1.SSS2.Px1.p1.1 "Datasets ‣ 4.1.2 System Prompt Distillation ‣ 4.1 Evaluation Tasks ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models").
* [7]
  Q. Dong, L. Li, D. Dai, C. Zheng, J. Ma, R. Li, H. Xia, J. Xu, Z. Wu, B. Chang, X. Sun, L. Li, and Z. Sui (2024-11)
  A survey on in-context learning.
  In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Y. Al-Onaizan, M. Bansal, and Y. Chen (Eds.),
  Miami, Florida, USA,  pp. 1107–1128.
  External Links: [Link](https://aclanthology.org/2024.emnlp-main.64/),
  [Document](https://dx.doi.org/10.18653/v1/2024.emnlp-main.64)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ On-Policy Context Distillation for Language Models").
* [8]
  A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, et al. (2024)
  The llama 3 herd of models.
  arXiv preprint arXiv:2407.21783.
  Cited by: [§4.2](#S4.SS2.SSS0.Px1.p1.1 "Models ‣ 4.2 Setup ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models").
* [9]
  Y. Gu, L. Dong, F. Wei, and M. Huang (2024)
  MiniLLM: on-policy distillation of large language models.
  In The Twelfth International Conference on Learning Representations,
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ On-Policy Context Distillation for Language Models"),
  [§1](#S1.p4.1 "1 Introduction ‣ On-Policy Context Distillation for Language Models"),
  [§2](#S2.SS0.SSS0.Px2.p1.1 "On-Policy Distillation ‣ 2 Related Work ‣ On-Policy Context Distillation for Language Models"),
  [§3](#S3.p1.4 "3 Method ‣ On-Policy Context Distillation for Language Models"),
  [§3](#S3.p2.4 "3 Method ‣ On-Policy Context Distillation for Language Models").
* [10]
  L. Guertler, B. Cheng, S. Yu, B. Liu, L. Choshen, and C. Tan (2025)
  TextArena.
  arXiv preprint arXiv:2504.11442.
  Cited by: [§A.2](#A1.SS2.p1.2 "A.2 Dataset Details ‣ Appendix A Experiential Knowledge Distillation Details ‣ On-Policy Context Distillation for Language Models"),
  [§4.1.1](#S4.SS1.SSS1.Px1.p1.1 "Datasets ‣ 4.1.1 Experiential Knowledge Distillation ‣ 4.1 Evaluation Tasks ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models").
* [11]
  J. Hübotter, F. Lübeck, L. Behric, A. Baumann, M. Bagatella, D. Marta, I. Hakimi, I. Shenfeld, T. K. Buening, C. Guestrin, and A. Krause (2026)
  Reinforcement learning via self-distillation.
  External Links: 2601.20802,
  [Link](https://arxiv.org/abs/2601.20802)
  Cited by: [§2](#S2.SS0.SSS0.Px3.p1.1 "Self-Distillation ‣ 2 Related Work ‣ On-Policy Context Distillation for Language Models"),
  [§3.1](#S3.SS1.SSS0.Px2.p1.2 "Self-Distillation (𝜋_teacher=𝜋_𝜃) ‣ 3.1 Teacher Model Configurations ‣ 3 Method ‣ On-Policy Context Distillation for Language Models"),
  [footnote 2](#footnote2 "In 4.6 Teacher-Student Distillation vs. Self-Distillation ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models").
* [12]
  K. Lu and T. M. Lab (2025)
  On-policy distillation.
  Thinking Machines Lab: Connectionism.
  Note: https://thinkingmachines.ai/blog/on-policy-distillation
  External Links: [Document](https://dx.doi.org/10.64434/tml.20251026)
  Cited by: [§1](#S1.p4.1 "1 Introduction ‣ On-Policy Context Distillation for Language Models"),
  [§2](#S2.SS0.SSS0.Px2.p1.1 "On-Policy Distillation ‣ 2 Related Work ‣ On-Policy Context Distillation for Language Models"),
  [§3](#S3.p1.4 "3 Method ‣ On-Policy Context Distillation for Language Models").
* [13]
  I. Mollas, Z. Chrysopoulou, S. Karlos, and G. Tsoumakas (2022)
  ETHOS: a multi-label hate speech detection dataset.
  Complex & Intelligent Systems 8 (6),  pp. 4663–4678.
  Cited by: [§B.1](#A2.SS1.p1.1 "B.1 System Prompts ‣ Appendix B System Prompt Distillation Details ‣ On-Policy Context Distillation for Language Models"),
  [§4.1.2](#S4.SS1.SSS2.Px1.p1.1 "Datasets ‣ 4.1.2 System Prompt Distillation ‣ 4.1 Evaluation Tasks ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models").
* [14]
  A. Pal, L. K. Umapathi, and M. Sankarasubbu (2022)
  Medmcqa: a large-scale multi-subject multi-choice dataset for medical domain question answering.
  In Conference on health, inference, and learning,
   pp. 248–260.
  Cited by: [§B.1](#A2.SS1.p1.1 "B.1 System Prompts ‣ Appendix B System Prompt Distillation Details ‣ On-Policy Context Distillation for Language Models"),
  [§4.1.2](#S4.SS1.SSS2.Px1.p1.1 "Datasets ‣ 4.1.2 System Prompt Distillation ‣ 4.1 Evaluation Tasks ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models").
* [15]
  E. Penaloza, D. Vattikonda, N. Gontier, A. Lacoste, L. Charlin, and M. Caccia (2026)
  Privileged information distillation for language models.
  External Links: 2602.04942,
  [Link](https://arxiv.org/abs/2602.04942)
  Cited by: [§2](#S2.SS0.SSS0.Px3.p1.1 "Self-Distillation ‣ 2 Related Work ‣ On-Policy Context Distillation for Language Models"),
  [§3.1](#S3.SS1.SSS0.Px2.p1.2 "Self-Distillation (𝜋_teacher=𝜋_𝜃) ‣ 3.1 Teacher Model Configurations ‣ 3 Method ‣ On-Policy Context Distillation for Language Models").
* [16]
  P. Röttger, B. Vidgen, D. Nguyen, Z. Talat, H. Margetts, and J. Pierrehumbert (2021)
  HateCheck: functional tests for hate speech detection models.
  In Proceedings of the 59th annual meeting of the association for computational linguistics and the 11th international joint conference on natural language processing (volume 1: long papers),
   pp. 41–58.
  Cited by: [§B.1](#A2.SS1.p1.1 "B.1 System Prompts ‣ Appendix B System Prompt Distillation Details ‣ On-Policy Context Distillation for Language Models"),
  [§4.1.2](#S4.SS1.SSS2.Px1.p1.1 "Datasets ‣ 4.1.2 System Prompt Distillation ‣ 4.1 Evaluation Tasks ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models").
* [17]
  I. Shenfeld, M. Damani, J. Hübotter, and P. Agrawal (2026)
  Self-distillation enables continual learning.
  External Links: 2601.19897,
  [Link](https://arxiv.org/abs/2601.19897)
  Cited by: [§2](#S2.SS0.SSS0.Px3.p1.1 "Self-Distillation ‣ 2 Related Work ‣ On-Policy Context Distillation for Language Models"),
  [§3.1](#S3.SS1.SSS0.Px2.p1.2 "Self-Distillation (𝜋_teacher=𝜋_𝜃) ‣ 3.1 Teacher Model Configurations ‣ 3 Method ‣ On-Policy Context Distillation for Language Models"),
  [footnote 2](#footnote2 "In 4.6 Teacher-Student Distillation vs. Self-Distillation ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models").
* [18]
  I. Shenfeld, J. Pari, and P. Agrawal (2025)
  Rl’s razor: why online reinforcement learning forgets less.
  arXiv preprint arXiv:2509.04259.
  Cited by: [§4.5](#S4.SS5.p2.1 "4.5 On-Policy Context Distillation Mitigates Forgetting ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models").
* [19]
  C. Snell, D. Klein, and R. Zhong (2022)
  Learning by distilling context.
  External Links: 2209.15189,
  [Link](https://arxiv.org/abs/2209.15189)
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ On-Policy Context Distillation for Language Models"),
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Context Distillation ‣ 2 Related Work ‣ On-Policy Context Distillation for Language Models"),
  [§3](#S3.p1.4 "3 Method ‣ On-Policy Context Distillation for Language Models"),
  [§4.2](#S4.SS2.SSS0.Px3.p1.1 "Evaluation ‣ 4.2 Setup ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models"),
  [§4.3](#S4.SS3.SSS0.Px1.p1.1 "Experiential Knowledge Consolidation ‣ 4.3 Results ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models").
* [20]
  S. Wang, Y. Wu, and Z. Xu (2025)
  Cogito, ergo ludo: an agent that learns to play by reasoning and planning.
  arXiv preprint arXiv:2509.25052.
  Cited by: [§A.2](#A1.SS2.p1.2 "A.2 Dataset Details ‣ Appendix A Experiential Knowledge Distillation Details ‣ On-Policy Context Distillation for Language Models").
* [21]
  A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. (2025)
  Qwen3 technical report.
  arXiv preprint arXiv:2505.09388.
  Cited by: [§4.2](#S4.SS2.SSS0.Px1.p1.1 "Models ‣ 4.2 Setup ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models").
* [22]
  A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, L. Chengyuan, D. Liu, F. Huang, H. Wei, et al. (2025)
  Qwen2.5 Technical Report.
  arXiv preprint arXiv:2412.15115.
  Cited by: [§4.2](#S4.SS2.SSS0.Px1.p1.1 "Models ‣ 4.2 Setup ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models").
* [23]
  T. Ye, L. Dong, Z. Chi, X. Wu, S. Huang, and F. Wei (2026)
  Black-box on-policy distillation of large language models.
  External Links: 2511.10643,
  [Link](https://arxiv.org/abs/2511.10643)
  Cited by: [§2](#S2.SS0.SSS0.Px2.p1.1 "On-Policy Distillation ‣ 2 Related Work ‣ On-Policy Context Distillation for Language Models").
* [24]
  Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, W. Dai, T. Fan, G. Liu, L. Liu, et al. (2025)
  Dapo: an open-source llm reinforcement learning system at scale.
  arXiv preprint arXiv:2503.14476.
  Cited by: [§A.2](#A1.SS2.p1.2 "A.2 Dataset Details ‣ Appendix A Experiential Knowledge Distillation Details ‣ On-Policy Context Distillation for Language Models"),
  [§4.1.1](#S4.SS1.SSS1.Px1.p1.1 "Datasets ‣ 4.1.1 Experiential Knowledge Distillation ‣ 4.1 Evaluation Tasks ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models").
* [25]
  E. Zelikman, Y. Wu, J. Mu, and N. Goodman (2022)
  STar: bootstrapping reasoning with reasoning.
  In Advances in Neural Information Processing Systems, A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho (Eds.),
  External Links: [Link](https://openreview.net/forum?id=_3ELRdg2sgI)
  Cited by: [§2](#S2.SS0.SSS0.Px3.p1.1 "Self-Distillation ‣ 2 Related Work ‣ On-Policy Context Distillation for Language Models").
* [26]
  S. Zhao, Z. Xie, M. Liu, J. Huang, G. Pang, F. Chen, and A. Grover (2026)
  Self-distilled reasoner: on-policy self-distillation for large language models.
  External Links: 2601.18734,
  [Link](https://arxiv.org/abs/2601.18734)
  Cited by: [§2](#S2.SS0.SSS0.Px3.p1.1 "Self-Distillation ‣ 2 Related Work ‣ On-Policy Context Distillation for Language Models"),
  [§3.1](#S3.SS1.SSS0.Px2.p1.2 "Self-Distillation (𝜋_teacher=𝜋_𝜃) ‣ 3.1 Teacher Model Configurations ‣ 3 Method ‣ On-Policy Context Distillation for Language Models").
* [27]
  J. Zhou, T. Lu, S. Mishra, S. Brahma, S. Basu, Y. Luan, D. Zhou, and L. Hou (2023)
  Instruction-following evaluation for large language models.
  arXiv preprint arXiv:2311.07911.
  Cited by: [§4.2](#S4.SS2.SSS0.Px3.p1.1 "Evaluation ‣ 4.2 Setup ‣ 4 Experiments ‣ On-Policy Context Distillation for Language Models").

## Appendix A Experiential Knowledge Distillation Details

### A.1 Prompt Templates

For experiential knowledge extraction on math dataset, we use the prompt template in [Figure˜4](#A1.F4 "In A.1 Prompt Templates ‣ Appendix A Experiential Knowledge Distillation Details ‣ On-Policy Context Distillation for Language Models").

You are an AI language model that continuously refines its internal experience.
Here is the latest interaction (including the user’s question and your answer):
{latest\_experience}
Your task:
Based on the latest interaction and the previous experience, generate an additional experience for future learning.
Rules:
- The experience you generate MUST be formatted strictly as a markdown list where each item starts with "- EXPERIENCE ITEM:", one per line:
- EXPERIENCE ITEM: …
- EXPERIENCE ITEM: …
- EXPERIENCE ITEM: …
- The experience you generate will be directly appended to the previous experience.
- The change should introduce a general, high-level, widely applicable insight, not a detail from the specific interaction. The updated experience must remain concise, structured, and meaningful.
- If the new insight conflicts with any previous experience item, you are can describe the conflict and provide a resolution in the new item.
After careful reasoning step by step, output the final result in exactly this format:
Additional Experience:
# Experience
- EXPERIENCE ITEM: …
- EXPERIENCE ITEM: …
- EXPERIENCE ITEM: …

Figure 4: The prompt wrapper for experiential knowledge extraction on math dataset.

We extract lines that start with “-- EXPERIENCE ITEM:” as valid experiential knowledge.

For experiential knowledge extraction on text-based games, we use the prompt template in [Figure˜5](#A1.F5 "In A.1 Prompt Templates ‣ Appendix A Experiential Knowledge Distillation Details ‣ On-Policy Context Distillation for Language Models").

You are an AI language model that continuously refines its internal experience.
Here is the interaction history (the game environment (input) and your response and action (output)):
{latest\_experience}
Your task:
Based on the multi-round interaction history, generate experience for future learning. You should conduct a deep, comparative analysis to infer the game rules and the fundamental principles behind winning and losing. Using the interaction history and environment feedback, hypothesize the game rules and effective winning strategies, and organize these insights into 1-2 concise, high-level, and widely applicable experience items that help the player succeed in the game.
Rules:
- The experience you generate MUST be formatted strictly as a markdown item which starts with "- EXPERIENCE ITEM:":
- EXPERIENCE ITEM: …
- EXPERIENCE ITEM: …
- The experience you generate will be directly appended to the previous experience. Do not repeat the previous experience. Make sure the newly generated experience is different from the previous experience.
- Your generated experience should be possible rules, instructions or winning strategies for the game. The experience should be generally useful rather than only applicable for the current map (board).
After careful reasoning step by step, output the final result in exactly this format:
Additional Experience (Rules or Strategies):
# Experience
- EXPERIENCE ITEM: …

Figure 5: The prompt wrapper for experiential knowledge extraction on text games.

We extract lines that start with “-- EXPERIENCE ITEM:” as valid experiential knowledge.

For new problems we embed experiential knowledge with the prompt template in [Figure˜6](#A1.F6 "In A.1 Prompt Templates ‣ Appendix A Experiential Knowledge Distillation Details ‣ On-Policy Context Distillation for Language Models").

Given previous learned experience:
# Experience
{experience}
Solve the new problem and explain what part of experience you use and how you use it in the reasoning process:
{prompt}

Figure 6: The prompt wrapper for experiential knowledge extraction on text games.

### A.2 Dataset Details

We train our models on three datasets: English math problems from DAPO-Math-17K [[24](#bib.bib10 "Dapo: an open-source llm reinforcement learning system at scale")] and two text-based game environments, Frozen Lake and Sokoban, implemented in TextArena [[10](#bib.bib12 "TextArena")].
DAPO-Math-17K contains approximately 14K verifiable English math problems, each with a numerical answer.
Frozen Lake is a grid-based navigation task where the model must reach a goal while avoiding holes. We place two holes on a 3 ×\times 3 grid.
Sokoban is a spatial reasoning puzzle where the model must push a box to a designated target without falling into holes or becoming trapped against walls. We place one box on a 6 ×\times 6 grid. We remove a subset of explicit rules for the model to infer them through exploration [[20](#bib.bib11 "Cogito, ergo ludo: an agent that learns to play by reasoning and planning")].
TextArena provides textual descriptions of the current game state at each step. The language model interacts with the game environments in a multi-turn setting.

### A.3 Training Details

We begin by sampling 30 problems from validation data split and prompting the teacher model to produce response traces one by one. The teacher then extracts experiential knowledge from each trace (without ground-truth labels), which we iteratively concatenate to form 30 experiential knowledge contexts. Repeating this procedure 10 times with different random seeds yields 300 distinct experiential knowledge contexts. The maximum experiential knowledge length is set to 16384 tokens for math and 8192 tokens for text games.
For the test-time experiential knowledge distillatione setting, we randomly select experiential knowledge from this pool of 300 for further OPCD training. This setting emulates a test-time experiential knowledge distillation scenario in which no ground-truth labels are available and the quality of experiential knowledge is not pre-evaluated. For the filtered experiential knowledge distillation setting, we score each candidate experiential knowledge by prepending it to new problems and evaluating performance on 1000 math validation examples or 128 text-game validation examples. The highest-scoring experiential knowledge is then selected for subsequent OPCD training.

We then distill the student model on training split of math and text-game datasets using the selected experiential knowledge context for 50 steps. We compute the reverse KL divergence using the top 256 vocabulary tokens with the highest student model probabilities. We use a batch size of 128 and search learning rate in [1e-6, 5e-6]. For math, we set the maximum response length to 16384 tokens. For text games, the model interacts with the game environment for up to 5 rounds, each with a maximum response length of 1024 tokens. We save checkpoints every 2 steps and choose the checkpoint with highest test accuracy.

### A.4 Experiential Knowledge Accumulation

!(/html/2602.12275/assets/x4.png)

Figure 7: Validation accuracy improves with the accumulation of experiential knowledge from different problems. Left: experiential knowledge accumulation on the DAPO math dataset. Right: experiential knowledge accumulation on the Frozen Lake text game.

We sample 30 validation problems for the teacher model to solve and extract experiential knowledge, repeating this procedure 10 times. In [Figure˜7](#A1.F7 "In A.4 Experiential Knowledge Accumulation ‣ Appendix A Experiential Knowledge Distillation Details ‣ On-Policy Context Distillation for Language Models"), we demonstrate validation accuracy improves with accumulation of experiential knowledge from different problems.

### A.5 Experiential Knowledge Examples

We provide some experiential knowledge examples for math in [Figure˜8](#A1.F8 "In A.5 Experiential Knowledge Examples ‣ Appendix A Experiential Knowledge Distillation Details ‣ On-Policy Context Distillation for Language Models").

- EXPERIENCE ITEM: Recognizing that combining interdependent sequences can reveal simpler underlying patterns, such as Fibonacci-like recurrences, simplifies complex problems.
- EXPERIENCE ITEM: Modular arithmetic often reveals periodicity, which can drastically reduce computational effort by allowing predictions based on cycle lengths.
- EXPERIENCE ITEM: The sum of a number’s digits is congruent to the number modulo 9, which is fundamental for determining digital roots and simplifying large computations.
- EXPERIENCE ITEM: When solving problems involving circular arrangements with symmetry constraints, it’s often beneficial to fix positions to eliminate rotational symmetry and then account for reflectional symmetry by dividing by 2.
- EXPERIENCE ITEM: The shoelace formula is a versatile tool for finding the area of any polygon given its vertices, reinforcing the value of systematic, coordinate-based approaches.

Figure 8: Some experiential knowledge examples for math problems.

We provide some experiential knowledge examples for Frozen Lake in [Figure˜9](#A1.F9 "In A.5 Experiential Knowledge Examples ‣ Appendix A Experiential Knowledge Distillation Details ‣ On-Policy Context Distillation for Language Models").

- EXPERIENCE ITEM: The shortest path to the goal involves moving systematically toward the target, prioritizing direct routes and minimizing unnecessary backtracking. Strategic use of available actions (e.g., down or right) to reach the goal in the fewest steps is key to success.
- EXPERIENCE ITEM: The game rules dictate that the player can move in four directions (up, down, left, right) but must avoid obstacles represented by ’H’ (walls). Movement is restricted to empty cells, and the goal is to reach the ’G’ cell.
- EXPERIENCE ITEM: The optimal strategy involves moving right to expand the grid and then down to reach the third row, ensuring a path to the goal while avoiding obstacles. This requires careful planning to minimize steps and avoid blocked cells.

Figure 9: Some experiential knowledge examples for Frozen Lake game.

## Appendix B System Prompt Distillation Details

### B.1 System Prompts

For system prompt distillation task, we use system prompts optimized for medical and safety tasks from MetaSPO [[6](#bib.bib6 "System prompt optimization with meta-learning")]. For medical system prompt, we adopt MedMCQA [[14](#bib.bib5 "Medmcqa: a large-scale multi-subject multi-choice dataset for medical domain question answering")] dataset and hold out 500 samples for testing. Each query requires the model to answer a multiple-choice medical question and provide a supporting explanation. Distillation is performed on both the teacher model’s selected choice and its explanation.
For safety system prompt, we combine Tweet Eval [[3](#bib.bib4 "TweetEval: unified benchmark and comparative evaluation for tweet classification")], Hatecheck [[16](#bib.bib3 "HateCheck: functional tests for hate speech detection models")], and Ethos [[13](#bib.bib1 "ETHOS: a multi-label hate speech detection dataset")] datasets, and similarly reserve 500 samples for testing. These queries require the model to determine whether a statement violates safety standards and justify its judgement. Distillation is applied to both the teacher model’s classification and its explanation.

The medical system prompt we use are present in [Figure˜10](#A2.F10 "In B.1 System Prompts ‣ Appendix B System Prompt Distillation Details ‣ On-Policy Context Distillation for Language Models").

You are a knowledgeable and analytical assistant specializing in medical topics. Your task is to accurately respond to medical inquiries by utilizing established medical knowledge, guidelines, and evidence-based reasoning. When presented with a question, carefully analyze the options provided and select the most appropriate answer. Ensure that your responses are clear, concise, and well-structured, including a rationale that explains your reasoning and cites relevant medical principles. Prioritize accuracy and logical coherence in all your responses.

Figure 10: The medical system prompt.

The safety system prompt we use are present in [Figure˜11](#A2.F11 "In B.1 System Prompts ‣ Appendix B System Prompt Distillation Details ‣ On-Policy Context Distillation for Language Models").

You are an analytical assistant. Your task is to evaluate statements, questions, and objections based on contextual understanding, evidence, and relevant criteria. When providing responses, please adhere to the following guidelines:
1. Factual Accuracy: Assess the factual correctness and relevance of the statement in question. Provide context where necessary.
2. Opinion vs. Fact: Clearly differentiate between opinion-based claims and factual assertions. Explain why a statement is considered an opinion or a fact.
3. Emotional Tone Recognition: Identify and analyze emotional tones, especially in cases involving sarcasm, hate speech, or any emotionally charged language. Discuss the implications of tone in your assessment.
4. Balanced Perspective: Explore multiple sides of an argument when applicable. Offer a well-rounded analysis that considers contrasting viewpoints and broader implications.
5. Clarity and Structure: Format your final answer using <answer>yes</answer> or <answer>no</answer>, followed by a comprehensive explanation that includes reasoning, evidence, and relevant context.
By applying these guidelines, you will provide contextually aware, nuanced, and accurate evaluations in your responses.

Figure 11: The safety system prompt.

The user instruction used for MedMCQA is presented in [Figure˜12](#A2.F12 "In B.1 System Prompts ‣ Appendix B System Prompt Distillation Details ‣ On-Policy Context Distillation for Language Models").

Given the following question and candidate answers, choose the best answer. <Question>{question}
Choices:
A. {opa}
B. {opb}
C. {opc}
D. {opd}
</Question>
At the end present your answer in <answer> and </answer> using an uppercase letter (A, B, C, or D)

Figure 12: The user instruction used for MedMCQA.

The user instruction used for safety datasets is presented in [Figure˜13](#A2.F13 "In B.1 System Prompts ‣ Appendix B System Prompt Distillation Details ‣ On-Policy Context Distillation for Language Models").

# For Tweet Eval
Determine whether the following text is offensive. <Question>{text}</Question>
At the end present your answer in <answer>yes</answer> or <answer>no</answer>.
# For Hatecheck
Is the following text hate speech? <Question>{text}</Question>
At the end present your answer in <answer>yes</answer> or <answer>no</answer>.
# For Ethos
Is the following text hate speech? <Question>{text}</Question>
At the end present your answer in <answer>yes</answer> or <answer>no</answer>.

Figure 13: The user instruction used for safety datasets.

### B.2 Training Details

We distill the student model on the training splits of the medical and safety datasets, conditioning on the corresponding system prompts. Training runs for 50 steps using reverse KL divergence, computed over the top 256 vocabulary tokens ranked by the student model’s predicted probabilities. We use a batch size of 128 and sweep the learning rate over [1e-6, 5e-6]. The maximum generated response length is set to 512 tokens. Checkpoints are saved every 2 steps, and we report the test accuracy averaged over the three best-performing checkpoints.
