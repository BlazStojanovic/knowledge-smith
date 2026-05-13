---
arxiv: '2603.29957'
authors:
- Xue Jiang
- Tianyu Zhang
- Ge Li
- Mengyang Liu
- Taozhi Chen
- Zhenhua Xu
- Binhua Li
- Wenpin Jiao
- Zhi Jin
- Yongbin Li
- Yihong Dong
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Think Anywhere in Code Generation
url: https://arxiv.org/abs/2603.29957
year: 2026
---

[2603.29957] Think Anywhere in Code Generation














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



# Think Anywhere in Code Generation

Xue Jiang1,2,🖂, Tianyu Zhang1, Ge Li1,🖂, Mengyang Liu1, Taozhi Chen1, Zhenhua Xu1,
  
Binhua Li2, Wenpin Jiao1, Zhi Jin1, Yongbin Li2, Yihong Dong1,2,🖂
  
1 School of Computer Science, Peking University
  
2 Tongyi Lab, Alibaba Group
  
{jiangxue, dongyh}@stu.pku.edu.cn  lige@pku.edu.cn

###### Abstract

Recent advances in reasoning Large Language Models (LLMs) have primarily relied on upfront thinking, where reasoning occurs before final answer. However, this approach suffers from critical limitations in code generation, where upfront thinking is often insufficient as problems’ full complexity only reveals itself during code implementation. Moreover, it cannot adaptively allocate reasoning effort throughout the code generation process where difficulty varies significantly. In this paper, we propose Think-Anywhere, a novel reasoning mechanism that enables LLMs to invoke thinking on-demand at any token position during code generation. We achieve Think-Anywhere by first teaching LLMs to imitate the reasoning patterns through cold-start training, then leveraging outcome-based RL rewards to drive the model’s autonomous exploration of when and where to invoke reasoning.
Extensive experiments on four mainstream code generation benchmarks (i.e., LeetCode, LiveCodeBench, HumanEval, and MBPP) show that Think-Anywhere achieves state-of-the-art performance over both existing reasoning methods and recent post-training approaches, while demonstrating consistent generalization across diverse LLMs.
Our analysis further reveals that Think-Anywhere enables the model to adaptively invoke reasoning at high-entropy positions, providing enhanced interpretability.

11footnotetext: Work done during Xue Jiang and Yihong Dong’s internship at Tongyi Lab.22footnotetext: Our source code and data are available at <https://github.com/jiangxxxue/Think-Anywhere>.

## 1 Introduction

Recent advances in Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation tasks (Rozière et al., [2023](#bib.bib105 "Code llama: open foundation models for code"), Lozhkov et al., [2024](#bib.bib16 "StarCoder 2 and the stack v2: the next generation"), Guo et al., [2024](#bib.bib41 "DeepSeek-coder: when the large language model meets programming–the rise of code intelligence"), Dong et al., [2024a](#bib.bib109 "Self-collaboration code generation via chatgpt"); [2025a](#bib.bib111 "A survey on code generation with llm-based agents")). A pivotal breakthrough in this domain has been the integration of reasoning mechanisms, particularly exemplified by Chain-of-Thought (CoT) prompting (Wei et al., [2022](#bib.bib51 "Chain-of-thought prompting elicits reasoning in large language models"), Jiang et al., [2024](#bib.bib3 "Self-planning code generation with large language models")). Recent reasoning-optimized LLMs, such as industry-leading OpenAI’s o1 (Jaech et al., [2024](#bib.bib5 "OpenAI o1 system card")), DeepSeek-R1 (Guo et al., [2025a](#bib.bib26 "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning")), and Kimi K2 (Bai et al., [2025](#bib.bib15 "Kimi K2: open agentic intelligence")), have achieved unprecedented performance by scaling up reasoning through reinforcement learning (RL). These models are trained to first complete global planning and logical deliberation within an internal thinking block, and then proceed to generate the final output. This upfront thinking approach has become the dominant technical pathway for enhancing complex reasoning capabilities in code generation (Jaech et al., [2024](#bib.bib5 "OpenAI o1 system card"), Jiang et al., [2026](#bib.bib114 "KOCO-BENCH: can large language models leverage domain knowledge in software development?"), Guo et al., [2025a](#bib.bib26 "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning")).

While the upfront thinking approach has proven effective, it exhibits two limitations in code generation. First, upfront thinking is often insufficient, as the full complexity of problems typically only reveals itself during implementation. For instance, LLMs usually perform only plan-level thinking in the upfront reasoning phase, while new problems emerge during the code implementation stage, leading to bugs due to the lack of adequate reasoning, as shown in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Think Anywhere in Code Generation"). Second, upfront thinking cannot precisely allocate reasoning effort to the positions where it is needed. Different positions in code generation vary in difficulty, with simple boilerplate code requiring minimal computation while complex algorithmic decisions or edge case handling demanding deep reasoning.
By contrast, human coding cognition shows that developers not only think before coding but also pause to think at any point during implementation, which proves a more reasonable thinking approach.
Motivated by these observations, we desire a mechanism that enables models to invoke reasoning at any token position during code generation based on immediate context and local complexity, which we term Think-Anywhere. The Think-Anywhere mechanism is demonstrated in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Think Anywhere in Code Generation").

Realizing the Think-Anywhere mechanism presents significant challenges. Since LLMs do not spontaneously invoke reasoning during code generation, they must be explicitly taught this capability. We achieve this through cold-start training by constructing supervised learning samples that demonstrate reasoning invocation patterns of Think-Anywhere. While cold-start training can teach models to invoke reasoning blocks within code, it cannot effectively teach models where to reasoning is necessary. The decision of which token positions to invoke thinking requires the model to identify its own moments of high complexity or logical risk, which demands adaptive judgment that goes beyond pattern matching in supervised data. To address this challenge, we employ Reinforcement Learning with Verifiable Rewards (RLVR) to enable LLMs to autonomously learn where to trigger reasoning during code generation, allowing the model to discover optimal thinking positions through reward-driven exploration.

![Refer to caption](/html/2603.29957/assets/x1.png)


Figure 1: Illustration of Think-Anywhere. Reasoning can be invoked at any token position during code generation. The ellipsis (“…”) within <think> or <thinkanywhere> represents truncated thinking content for brevity.

In this work, we propose Think-Anywhere, a novel reasoning mechanism of LLMs for code generation that enables models to invoke thinking at any token position based on LLM’s demands. Think-Anywhere is realized through a two-stage training pipeline. First, through cold-start training with carefully constructed code generation samples that demonstrate Think-Anywhere, we teach models the fundamental capability of pausing to think at arbitrary token positions during code generation. Second, we employ RLVR to further reinforce this capability, allowing models to autonomously explore and discover the optimal positions and strategies for invoking reasoning that suit the specific challenges they encounter. Think-Anywhere enables models to think on-demand at critical moments during code generation, precisely allocating computational resources to tokens that necessitate deep thinking. Moreover, by observing where and how models think during code generation, Think-Anywhere provides greater transparency into the decision-making process, enhancing the interpretability.

Extensive experiments demonstrate that Think-Anywhere achieves state-of-the-art performance over existing LLM reasoning-enhanced methods and recently proposed post-training methods on four mainstream code generation benchmarks, including LeetCode, LiveCodeBench, HumanEval, and MBPP. Think-Anywhere also exhibits strong generalization across different LLM families and model sizes. Ablation studies reveal that combining cold-start initialization with RLVR yields optimal results, and token-level thinking outperforms alternative variants such as line-level thinking. Further analysis highlights that LLMs tend to invoke thinking at positions with higher entropy, demonstrating that Think-Anywhere can reason at appropriate positions on demand.

## 2 Related Work

#### Reasoning and Planning Mechanisms in LLMs.

Enhancing the reasoning and planning capabilities of LLMs has emerged as a central research focus in recent years.
A seminal advancement in this direction is Chain-of-Thought (CoT) prompting (Wei et al., [2022](#bib.bib51 "Chain-of-thought prompting elicits reasoning in large language models")), which elicits complex reasoning by guiding LLMs to generate intermediate reasoning steps before arriving at a final answer.
Subsequent studies build on CoT with richer prompting strategies and search mechanisms (Kojima et al., [2022](#bib.bib8 "Large language models are zero-shot reasoners"), Wang et al., [2023](#bib.bib9 "Self-consistency improves chain of thought reasoning in language models"), Zhou et al., [2023](#bib.bib10 "Least-to-most prompting enables complex reasoning in large language models"), Yao et al., [2023](#bib.bib11 "Tree of thoughts: deliberate problem solving with large language models")).
In the domain of code generation, Self-Planning (Jiang et al., [2024](#bib.bib3 "Self-planning code generation with large language models")) conducts problem decomposition and planning prior to code generation to reduce task complexity.
While these methods treat reasoning as an upfront thinking phase, recent work explores interleaved strategies that tightly couple thinking with task execution.
For instance, Interleaved Thinking (Xie et al., [2025](#bib.bib6 "Interleaved reasoning for large language models via reinforcement learning"), Liang et al., [2025](#bib.bib14 "Plantain: plan-answer interleaved reasoning")) guides LLMs to alternate between thinking and answering, enabling incremental refinement based on intermediate results. TwiG (Guo et al., [2025b](#bib.bib4 "Thinking-while-generating: interleaving textual reasoning throughout visual generation")) interleaves textual reasoning throughout visual generation trajectories, allowing reasoning to guide upcoming synthesis and reflect on previously generated content.

Recent advances in reasoning LLMs, such as DeepSeek-R1 (Guo et al., [2025a](#bib.bib26 "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning")) and Kimi-K2 (Bai et al., [2025](#bib.bib15 "Kimi K2: open agentic intelligence")), have achieved remarkable success by employing upfront thinking. While recent work on Interleaved Thinking allows reasoning to occur during implementation, it requires thinking at each sub-step and lacks the flexibility for on-demand invocation. This limitation introduces unnecessary computational overhead, while failing to allocate deeper reasoning effort to the most challenging portions of a task.

#### Post-Training of LLMs for Code Generation.

Post-training has become important for improving the code generation capabilities of LLMs beyond pretraining, as it can better exploit task-specific data and verifiable execution signals.
One major approach is distillation from stronger reasoning LLMs. For example, OlympicCoder (Hugging Face, [2025](#bib.bib95 "Open r1: a fully open reproduction of deepseek-r1")) fine-tunes models on competitive programming tasks using reasoning trajectories distilled from DeepSeek-R1. Similarly, OCR-Qwen-7B (Ahmad et al., [2025](#bib.bib96 "OpenCodeReasoning: advancing data distillation for competitive coding")) is distilled from DeepSeek-R1, leveraging a large-scale dataset of over 730K reasoning-annotated samples for open-source reproduction.
Another major approach is RL from executable feedback, which has been widely adopted to strengthen code generation and reasoning capabilities.
Skywork-OR1 (He et al., [2025](#bib.bib97 "Skywork open reasoner 1 technical report")) employs large-scale RLVR training following DeepSeek-R1’s pipeline for code generation.
CodePRM (Li et al., [2025b](#bib.bib98 "CodePRM: execution feedback-enhanced process reward model for code generation")) introduces a process reward model that provides step-level rewards for intermediate steps during generation.
CodeBoost (Wang et al., [2025](#bib.bib39 "CodeBoost: boosting code LLMs by squeezing knowledge from code snippets with rl")) enhances code generation through RL training on code reasoning tasks.
CodeRL+ (Jiang et al., [2025](#bib.bib7 "CodeRL+: improving code generation via reinforcement with execution semantics alignment")) further enriches the learning signal by aligning code generation with execution semantics beyond binary pass/fail feedback.

Existing post-training methods, regardless of whether they are based on distillation or RL, predominantly adopt the upfront thinking practice. This introduces the limitations discussed in Section [1](#S1 "1 Introduction ‣ Think Anywhere in Code Generation"), necessitating a shift in the thinking approach for code generation.

## 3 Methodology

### 3.1 Defining Think-Anywhere

We begin by formally defining the Think-Anywhere mechanism and contrasting it with the conventional upfront thinking method.
Let xx denote the requirement and cc denote the generated code. We define two special token pairs: ⟨think⟩\langle\texttt{think}\rangle and ⟨/think⟩\langle\texttt{/think}\rangle for the upfront thinking block, and ⟨thinkanywhere⟩\langle\texttt{thinkanywhere}\rangle and ⟨/thinkanywhere⟩\langle\texttt{/thinkanywhere}\rangle for the Think-Anywhere thinking block.

#### Upfront Thinking.

In the upfront thinking method adopted by existing reasoning-enhanced LLMs (Jaech et al., [2024](#bib.bib5 "OpenAI o1 system card"), Guo et al., [2025a](#bib.bib26 "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning")), the generation process can be decomposed into two sequential phases. Given input xx, the model first generates a complete reasoning trace ss enclosed within ⟨think⟩\langle\texttt{think}\rangle and ⟨/think⟩\langle\texttt{/think}\rangle tokens, and then generates the code cc conditioned on both xx and ss:

|  |  |  |  |
| --- | --- | --- | --- |
|  | P​(c,s∣x)=P​(s∣x)⏟upfront reasoning⋅P​(c∣x,s)⏟code generation.P(c,s\mid x)=\underbrace{P(s\mid x)}\_{\text{upfront reasoning}}\cdot\underbrace{P(c\mid x,s)}\_{\text{code generation}}. |  | (1) |

This formulation enforces a strict separation between reasoning and code generation, making LLM difficult to invoke additional reasoning in code generation process.

#### Think-Anywhere.

Think-Anywhere enables LLM to precisely reason at any position where deliberation is needed during code generation. Considering the non-uniform distribution of logical complexity in code generation, Think-Anywhere allows the model to dynamically scale its reasoning length at challenging bottlenecks, achieving a truly on-demand allocation of computational resources.
Formally, the model generates a mixed sequence 𝐲\mathbf{y}. This sequence naturally decomposes into code segments and thinking blocks:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐲=(s,c(1),h(1),c(2),h(2),…,c(M),h(M),c(M+1)),\mathbf{y}=(s,c^{(1)},h^{(1)},c^{(2)},h^{(2)},\ldots,c^{(M)},h^{(M)},c^{(M+1)}), |  | (2) |

where ss denotes the initial thinking block enclosed within ⟨think⟩\langle\texttt{think}\rangle and ⟨/think⟩\langle\texttt{/think}\rangle, each c(i)c^{(i)} represents a code segment, and each h(i)h^{(i)} represents a thinking block enclosed within ⟨thinkanywhere⟩\langle\texttt{thinkanywhere}\rangle and ⟨/thinkanywhere⟩\langle\texttt{/thinkanywhere}\rangle tokens that is placed between code segments. The number of thinking blocks M≥0M\geq 0 and their positions are dynamically determined by the model during generation.

The generation process of Think-Anywhere can be formulated as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | P​(𝐲∣x)=P​(s∣x)⋅∏i=1M[P​(c(i)∣x,𝐲<c(i))⋅P​(h(i)∣x,𝐲<h(i))]⋅P​(c(M+1)∣x,𝐲<c(M+1)),\begin{split}P(\mathbf{y}\mid x)=P(s\mid x)\cdot\prod\_{i=1}^{M}\Big[P(c^{(i)}\mid x,\mathbf{y}\_{<c^{(i)}})\cdot P(h^{(i)}\mid x,\mathbf{y}\_{<h^{(i)}})\Big]\cdot P(c^{(M+1)}\mid x,\mathbf{y}\_{<c^{(M+1)}}),\end{split} |  | (3) |

where 𝐲<c(i)\mathbf{y}\_{<c^{(i)}} and 𝐲<h(i)\mathbf{y}\_{<h^{(i)}} denote all preceding tokens before code segment c(i)c^{(i)} and thinking block h(i)h^{(i)}, respectively. Notably, upfront thinking can be viewed as a special case of Think-Anywhere where thinking occurs exclusively at the beginning.

The final executable code cc is obtained by removing all thinking blocks from 𝐲\mathbf{y}, including the initial ⟨think⟩\langle\texttt{think}\rangle block and all ⟨thinkanywhere⟩\langle\texttt{thinkanywhere}\rangle blocks:

|  |  |  |  |
| --- | --- | --- | --- |
|  | c=c(1)⊕c(2)⊕⋯⊕c(M+1),c=c^{(1)}\oplus c^{(2)}\oplus\cdots\oplus c^{(M+1)}, |  | (4) |

where ⊕\oplus denotes sequence concatenation.

|  |
| --- |
| You are a coding assistant that generates both code and inline self-guidance signals. First output <think>... </think> with brief reasoning, then output the final code. |
| MUST FOLLOW Rules for <thinkanywhere>...</thinkanywhere> tags: |
| 1. You MUST use <thinkanywhere>...</thinkanywhere> tags for self-guidance or intermediate reasoning. |
| 2. <thinkanywhere>...</thinkanywhere> MUST be embedded within an existing program statement token sequence. |
| 3. The code must remain valid and executable after removing all <thinkanywhere>...</thinkanywhere> segments. |
| User: Prompt. Assistant: |

Table 1: Template for Think-Anywhere. Prompt will be replaced with the specific coding requirement.

#### Training Template.

To train Think-Anywhere, we design a template that guides LLMs to follow the Think-Anywhere generation format, as shown in Table [1](#S3.T1 "Table 1 ‣ Think-Anywhere. ‣ 3.1 Defining Think-Anywhere ‣ 3 Methodology ‣ Think Anywhere in Code Generation"). The template instructs the model to first produce initial reasoning within ⟨think⟩\langle\texttt{think}\rangle tags, then generate code with ⟨thinkanywhere⟩\langle\texttt{thinkanywhere}\rangle blocks invoked at positions requiring deliberation. We constrain only the structural format while avoiding content-specific biases, allowing the model to discover optimal thinking patterns through subsequent reinforcement learning.

### 3.2 Cold Start for Think-Anywhere

LLMs do not invoke thinking blocks during code generation, and even explicit instructions in prompts often fail to enforce this behavior reliably. Therefore, they must be explicitly taught this capability through training. The goal of cold start is to equip the model with the fundamental ability to reason at arbitrary positions within code.

#### Automatic Data Construction.

We leverage strong reasoning LLMs with our training template to automatically construct training data that demonstrates the Think-Anywhere pattern. Specifically, we prompt the reasoning LLMs to solve coding problems while explicitly invoking thinking blocks enclosed within ⟨thinkanywhere⟩\langle\texttt{thinkanywhere}\rangle and ⟨/thinkanywhere⟩\langle\texttt{/thinkanywhere}\rangle tokens at positions where deliberation is needed during code generation.

To ensure data quality, we filter out samples with incorrect formatting, such as malformed thinking block boundaries or improper nesting of special tokens. Following prior work (Li et al., [2025a](#bib.bib19 "CodeI/o: condensing reasoning patterns via code input-output prediction")) that demonstrates both correct and incorrect solutions contribute to model learning, we retain samples regardless of code correctness. This process of data construction yields approximately 5,000 training samples.

We perform supervised fine-tuning using LoRA (Hu et al., [2022](#bib.bib18 "LoRA: low-rank adaptation of large language models")) on the constructed training samples as cold start. Following (Schulman and Lab, [2025](#bib.bib2 "LoRA without regret")), we adopt LoRA over full-parameter SFT as it achieves comparable performance with greater robustness and lower computational overhead. This training enables the model to learn the pattern of invoking ⟨thinkanywhere⟩\langle\texttt{thinkanywhere}\rangle blocks within code, acquiring the basic capability that serves as the foundation for subsequent reinforcement learning.

#### Dedicated Reasoning Trigger Token.

In default implementation, ⟨thinkanywhere⟩\langle\texttt{thinkanywhere}\rangle is tokenized into multiple ordinary tokens, each carrying its own lexical meaning. Requiring the model to use these tokens simultaneously as lexical units and as a trigger signal for invoking reasoning introduces semantic ambiguity. Moreover, generating a multi-token delimiter increases the prediction path length for a single control decision, making the trigger less reliable. We therefore introduce a special token variant (Think-Anywhere\*) that represents the thinking delimiter as a single dedicated vocabulary entry, providing an unambiguous and efficient signal for invoking inline reasoning.

However, directly adding randomly initialized special tokens is ineffective, as the limited post-training data is insufficient for the model to learn meaningful representations from scratch. To address this, we propose a semantic-aware initialization strategy that composes the embedding from two complementary sources: the semantic content of the trigger and the structural role of a delimiter. Specifically, we initialize the embeddings of the new special tokens as:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝐞⟨ta⟩\displaystyle\mathbf{e}\_{\langle\texttt{ta}\rangle} | =0.5⋅mean​(𝐞think,𝐞any,𝐞where)+0.5⋅𝐞⟨im\_start⟩,\displaystyle=0.5\cdot\text{mean}(\mathbf{e}\_{\texttt{think}},\mathbf{e}\_{\texttt{any}},\mathbf{e}\_{\texttt{where}})+0.5\cdot\mathbf{e}\_{\langle\texttt{im\\_start}\rangle}, |  | (5) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝐞⟨/ta⟩\displaystyle\mathbf{e}\_{\langle\texttt{/ta}\rangle} | =0.5⋅mean​(𝐞think,𝐞any,𝐞where)+0.5⋅𝐞⟨im\_end⟩,\displaystyle=0.5\cdot\text{mean}(\mathbf{e}\_{\texttt{think}},\mathbf{e}\_{\texttt{any}},\mathbf{e}\_{\texttt{where}})+0.5\cdot\mathbf{e}\_{\langle\texttt{im\\_end}\rangle}, |  | (6) |

where 𝐞⟨ta⟩\mathbf{e}\_{\langle\texttt{ta}\rangle} and 𝐞⟨/ta⟩\mathbf{e}\_{\langle\texttt{/ta}\rangle} denote the embeddings of the opening and closing special tokens, respectively. The first term encodes the semantic intent of “think anywhere,” while the second term inherits the structural behavior of existing delimiter tokens (⟨im\_start⟩\langle\texttt{im\\_start}\rangle and ⟨im\_end⟩\langle\texttt{im\\_end}\rangle), which the model has already learned to treat as mode-switching boundaries during pretraining.

To effectively train the dedicated trigger tokens, we adopt a two-stage cold-start procedure:

1. 1.

   Stage 1: Embedding alignment. We freeze the model parameters and train only the input embeddings and LM head weights. This stage allows the tokens to develop appropriate representations without disrupting the model’s existing capabilities.
2. 2.

   Stage 2: Joint fine-tuning. We continue training the special token embeddings and LM head jointly with LoRA adapters applied to the model, enabling the model to learn how to generate and respond to the dedicated trigger tokens in context.

The subsequent RLVR stage proceeds identically to the default version.

### 3.3 RLVR for Think-Anywhere

We then employ RLVR to enable the LLMs to autonomously discover optimal thinking positions and strategies through reward-driven exploration.

#### Reinforcement Learning Algorithm.

We adopt Group Relative Policy Optimization (GRPO) (Shao et al., [2024](#bib.bib59 "DeepSeekMath: pushing the limits of mathematical reasoning in open language models")) as our reinforcement learning algorithm. Unlike Proximal Policy Optimization (PPO) (Schulman et al., [2017](#bib.bib62 "Proximal policy optimization algorithms")) which requires a separate value model to estimate baselines, GRPO computes baselines from group-level statistics, eliminating the need for an additional value model and significantly reducing computational overhead.

Specifically, for each input xx, GRPO samples a group of GG candidate outputs {y1,y2,…,yG}\{y\_{1},y\_{2},\ldots,y\_{G}\} from the current policy πθ\pi\_{\theta}. The reward for each output yiy\_{i} is computed as R​(yi)R(y\_{i}), and the group-normalized advantage is calculated as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | A^i=R​(yi)−mean​({R​(yj)}j=1G)std​({R​(yj)}j=1G).\hat{A}\_{i}=\frac{R(y\_{i})-\text{mean}(\{R(y\_{j})\}\_{j=1}^{G})}{\text{std}(\{R(y\_{j})\}\_{j=1}^{G})}. |  | (7) |

The policy is then optimized by maximizing the clipped surrogate objective with a KL divergence penalty:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒGRPO​(θ)=𝔼​[min⁡(ρi​A^i,clip​(ρi,1−ϵ,1+ϵ)​A^i)−β⋅DKL​(πθ∥πref)],\displaystyle\mathcal{L}\_{\text{GRPO}}(\theta)=\mathbb{E}\Biggl[\min\left(\rho\_{i}\hat{A}\_{i},\text{clip}(\rho\_{i},1-\epsilon,1+\epsilon)\hat{A}\_{i}\right)-\beta\cdot D\_{\text{KL}}(\pi\_{\theta}\|\pi\_{\text{ref}})\Biggr], |  | (8) |

where ρi=πθ​(yi∣x)πold​(yi∣x)\rho\_{i}=\frac{\pi\_{\theta}(y\_{i}\mid x)}{\pi\_{\text{old}}(y\_{i}\mid x)} denotes the probability ratio, ϵ\epsilon is the clipping threshold, and β\beta controls the strength of the KL penalty against the reference policy πref\pi\_{\text{ref}}.

#### Reward Modeling.

We design a hierarchical reward function for Think-Anywhere. The reward consists of two components: a reasoning structure reward RstructR\_{\text{struct}} and a code correctness reward RcorrectR\_{\text{correct}}, combined in a gated manner:

|  |  |  |  |
| --- | --- | --- | --- |
|  | R(y)=α⋅Rstruct(y)+⋅Rcorrect(y),\displaystyle R(y)=\alpha\cdot R\_{\text{struct}}(y)+\cdot R\_{\text{correct}}(y), |  | (9) |

where α=0.1\alpha=0.1 controls the weight between the two components.

The reasoning structure reward Rstruct∈{0,1}R\_{\text{struct}}\in\{0,1\} verifies that the model adheres to the Think-Anywhere reasoning definition. Specifically, it checks whether the output contains an initial thinking block within ⟨think⟩\langle\texttt{think}\rangle and ⟨/think⟩\langle\texttt{/think}\rangle tags, followed by code that incorporates ⟨thinkanywhere⟩\langle\texttt{thinkanywhere}\rangle blocks:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Rstruct​(y)=𝟙​[HasInitialThinking​(y)∧HasThinkAnywhere​(y)],\displaystyle R\_{\text{struct}}(y)=\mathbbm{1}\left[\texttt{HasInitialThinking}(y)\land\texttt{HasThinkAnywhere}(y)\right], |  | (10) |

where HasInitialThinking​(⋅)\texttt{HasInitialThinking}(\cdot) verifies the presence of the initial ⟨think⟩\langle\texttt{think}\rangle block, and HasThinkAnywhere​(⋅)\texttt{HasThinkAnywhere}(\cdot) ensures that at least one ⟨thinkanywhere⟩\langle\texttt{thinkanywhere}\rangle block is embedded within the generated code. This reward encourages the model to actively engage in on-demand reasoning throughout the generation process.

The code correctness reward Rcorrect∈{0,1}R\_{\text{correct}}\in\{0,1\} evaluates the functional correctness of the generated code by executing it against the provided test cases:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Rcorrect​(y)=𝟙​[PassAllTests​(c)].R\_{\text{correct}}(y)=\mathbbm{1}\left[\texttt{PassAllTests}(c)\right]. |  | (11) |

Table 2: Performance of Think-Anywhere compared to post-training methods and reasoning-enhanced methods. Best results are in bold. Think-Anywhere\* denotes the special token variant with semantic-aware initialization (see Section 3.2).

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Method | LeetCode | LiveCodeBench | HumanEval | MBPP | Average |
| Base Model | 50.6 | 34.3 | 88.4 | 70.7 | 61.0 |
| Post-Training Methods | | | | | |
| OlympicCoder | 45.3 | 30.9 | 75.6 | 67.2 | 54.8 |
| OCR-Qwen-7B | 53.3 | 33.0 | 86.8 | 58.9 | 58.0 |
| CodePRM | 52.8 | 34.8 | 88.4 | 73.9 | 62.5 |
| CodeBoost | 53.3 | 34.6 | 87.2 | 65.7 | 60.2 |
| CodeRL+ | 63.3 | 36.9 | 90.9 | 76.2 | 66.8 |
| Reasoning-Enhanced Methods | | | | | |
| CoT | 53.9 | 30.9 | 86.6 | 77.7 | 62.3 |
| Self-planning | 49.2 | 31.1 | 86.9 | 77.9 | 61.3 |
| Interleaved Thinking | 50.6 | 30.7 | 86.4 | 79.2 | 61.7 |
| GRPO | 67.3 | 36.0 | 88.6 | 81.7 | 68.4 |
| Think-Anywhere (Prompting) | 41.1 | 34.4 | 84.8 | 67.4 | 56.9 |
| Think-Anywhere\* (SFT) | 46.7 | 32.5 | 79.9 | 78.2 | 59.3 |
| Think-Anywhere\* (Ours) | 68.9 | 36.7 | 90.2 | 84.5 | 70.0 |
| Think-Anywhere (SFT) | 47.9 | 32.3 | 82.9 | 79.4 | 60.6 |
| Think-Anywhere (Ours) | 69.4 | 37.2 | 91.5 | 82.9 | 70.3 |

## 4 Experiments

### 4.1 Experiment Setup

#### Training Details.

Follow previous work (He et al., [2025](#bib.bib97 "Skywork open reasoner 1 technical report")), our training corpus comprises 14K programming problem from the Skywork dataset. By default, we employ Qwen2.5-Coder-7B-Instruct (Hui et al., [2024](#bib.bib47 "Qwen2. 5-coder technical report")) as the base model for our experiments. The RL algorithm is implemented using the VeRL framework (Sheng et al., [2024](#bib.bib48 "HybridFlow: a flexible and efficient rlhf framework")). Training parameters are set as follows: batch size of 128, mini-batch size of 64, learning rate of 1e-06, and 2 training epochs. Each problem generates 8 rollout samples up to 4096 tokens. The experiments run on 8 NVIDIA A100 GPUs (40G). We employ Google’s Gemini 2.5 Flash (Google DeepMind, [2025](#bib.bib1 "Gemini 3 flash model card")) to synthesize cold-start training data.

#### Evaluation Details.

Following established practices in prior work (Li et al., [2025b](#bib.bib98 "CodePRM: execution feedback-enhanced process reward model for code generation"), Tang et al., [2025](#bib.bib99 "CodeReasoner: enhancing the code reasoning ability with reinforcement learning"), Wang et al., [2025](#bib.bib39 "CodeBoost: boosting code LLMs by squeezing knowledge from code snippets with rl"), Dong et al., [2025b](#bib.bib115 "RL-PLUS: countering capability boundary collapse of llms in reinforcement learning with hybrid-policy optimization"); [2024b](#bib.bib110 "Generalization or memorization: data contamination and trustworthy evaluation for large language models")), our evaluation encompasses four widely-used code generation benchmarks: HumanEval (Chen et al., [2021](#bib.bib71 "Evaluating large language models trained on code")), MBPP (Austin et al., [2021](#bib.bib17 "Program synthesis with large language models")), LeetCode (Xia et al., [2025](#bib.bib107 "Leetcodedataset: a temporal dataset for robust evaluation and efficient training of code llms")), and LiveCodeBench (Jain et al., [2024](#bib.bib42 "Livecodebench: holistic and contamination free evaluation of large language models for code")). We adopt pass@1 as our primary evaluation metric. To ensure reproducibility and consistency across all experiments, we employ greedy sampling with the temperature fixed at 0.

#### Baselines.

Beyond the base model and standard GRPO method (Shao et al., [2024](#bib.bib59 "DeepSeekMath: pushing the limits of mathematical reasoning in open language models")), we compare Think-Anywhere with two categories of methods, all using the same base model.
The first category includes the reasoning-enhanced approaches that incorporate thinking mechanisms, including CoT (Wei et al., [2022](#bib.bib51 "Chain-of-thought prompting elicits reasoning in large language models")), Self-Planning (Jiang et al., [2024](#bib.bib3 "Self-planning code generation with large language models")), and Interleaved Thinking (Xie et al., [2025](#bib.bib6 "Interleaved reasoning for large language models via reinforcement learning"))111As Interleaved Thinking does not provide source code, we adapt it to the code generation setting by prompting the model to alternate between reasoning and code implementation, following the method described in the original work..
The second category includes the recently proposed post-training models and methods developed for code generation, including OlympicCoder (Hugging Face, [2025](#bib.bib95 "Open r1: a fully open reproduction of deepseek-r1")), OCR-Qwen-7B (Ahmad et al., [2025](#bib.bib96 "OpenCodeReasoning: advancing data distillation for competitive coding")), CodePRM (Li et al., [2025b](#bib.bib98 "CodePRM: execution feedback-enhanced process reward model for code generation")), CodeBoost (Wang et al., [2025](#bib.bib39 "CodeBoost: boosting code LLMs by squeezing knowledge from code snippets with rl")), and CodeRL+ (Jiang et al., [2025](#bib.bib7 "CodeRL+: improving code generation via reinforcement with execution semantics alignment")).

### 4.2 Experiment Results

#### Performance of Think-Anywhere.

Table [2](#S3.T2 "Table 2 ‣ Reward Modeling. ‣ 3.3 RLVR for Think-Anywhere ‣ 3 Methodology ‣ Think Anywhere in Code Generation") presents the main results of Think-Anywhere compared to baselines on four benchmarks. Overall, Think-Anywhere achieves the best performance across all benchmarks, with an average score of 70.3%, representing a 9.3% absolute improvement over the base model.
Compared to Post-Training Methods, Think-Anywhere surpasses the best-performing baseline CodeRL+, demonstrating the effectiveness of our approach over other RL-based approaches.
Compared to Reasoning-Enhanced Methods, Think-Anywhere substantially outperforms CoT, Self-planning, Interleaved Thinking, and GRPO across all metrics. Notably, Some methods exhibit inconsistent improvements across different benchmarks. In contrast, Think-Anywhere achieves consistent improvements on both simple and challenging tasks, suggesting that our dynamic thinking strategy is more effective than fixed reasoning patterns.
Furthermore, the comparison among Think-Anywhere variants reveals the importance of RL-based training. Think-Anywhere (Prompting) and Think-Anywhere (SFT) underperform the base model on several benchmarks, whereas Think-Anywhere with RL training achieves substantial gains, highlighting that reinforcement learning is crucial for learning effective thinking patterns.

We also report the results of the special token variant (Think-Anywhere\*). With semantic-aware initialization and the two-stage cold-start procedure, Think-Anywhere\* achieves an average score of 70.0%, which is comparable to the default text-based version (70.3%).
We observe that the text-based version tends to invoke thinking blocks at stereotyped positions (*e.g*., after “=” tokens), while the special token variant exhibits more diverse and contextually appropriate placement. However, the limited post-training data constrains the special token variant from fully learning the semantics of the new tokens. We believe that natively integrating Think-Anywhere special tokens during large-scale pretraining would further unlock their potential. Since the text-based version slightly outperforms the special token variant under our post-training setting, we adopt the text-based version for all subsequent experiments.

#### Cross-Domain Generalization.

To investigate whether Think-Anywhere generalizes beyond code generation, we directly evaluate our code-domain-trained model on mathematical reasoning benchmarks, including AIME 2024 (Mathematical Association of America, [2024](#bib.bib116 "American invitational mathematics examination (aime) 2024")), AIME 2025 (Mathematical Association of America, [2025](#bib.bib117 "American invitational mathematics examination (aime) 2025")), and HMMT 2025 (Balunović et al., [2025](#bib.bib118 "MathArena: evaluating llms on uncontaminated math competitions")). Table [3](#S4.T3 "Table 3 ‣ Cross-Domain Generalization. ‣ 4.2 Experiment Results ‣ 4 Experiments ‣ Think Anywhere in Code Generation") reports the results under pass@1, pass@5, and pass@10 settings. Notably, although Think-Anywhere is trained exclusively on code generation tasks, it achieves consistent and substantial improvements over both the base model and GRPO across three mathematical reasoning benchmarks. For instance, on AIME 2024, Think-Anywhere improves pass@1 from 5.3% (Base Model) and 6.0% (GRPO) to 17.3%, representing a remarkable gain. Similar trends are observed on AIME 2025 and HMMT 2025, where Think-Anywhere achieves 17.7% and 14.4% pass@1 respectively. These results suggest that the think-on-demand reasoning capability acquired through Think-Anywhere is not domain-specific but transfers across tasks, demonstrating strong cross-domain generalization.

Table 3: Cross-domain generalization of Think-Anywhere to mathematical reasoning benchmarks. The model is trained solely on code generation tasks.

| Method | AIME 2024 | | | AIME 2025 | | | HMMT 2025 | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pass@1 | pass@5 | pass@10 | pass@1 | pass@5 | pass@10 | pass@1 | pass@5 | pass@10 |
| Base Model | 5.3 | 14.6 | 20.0 | 4.0 | 13.4 | 16.7 | 0.0 | 0.0 | 0.0 |
| GRPO | 6.0 | 16.8 | 23.3 | 4.7 | 17.2 | 26.7 | 0.3 | 1.7 | 3.3 |
| Think-Anywhere | 17.3 | 32.9 | 40.2 | 17.7 | 28.0 | 33.2 | 14.4 | 18.5 | 19.6 |




Table 4: Generalizability of Think-Anywhere across different model families and scales.

|  |  |  |
| --- | --- | --- |
| Model | Average | Δ\Delta vs. Base |
| Qwen2.5-Coder-7B-Instruct | | |
| Base Model | 61.0 | – |
| + GRPO | 68.4 | +7.4 |
| + Think-Anywhere | 70.3 | +9.3 |
| Qwen2.5-Coder-1.5B-Instruct | | |
| Base Model | 40.6 | – |
| + GRPO | 51.9 | +11.3 |
| + Think-Anywhere | 54.5 | +13.9 |
| LLaMA-3.1-8B-Instruct | | |
| Base Model | 38.4 | – |
| + GRPO | 42.0 | +3.6 |
| + Think-Anywhere | 43.8 | +5.4 |

#### Application on Various LLMs.

To validate the generalizability of Think-Anywhere, we evaluated its performance on three diverse LLMs spanning different model families and parameter scales: LLaMA-3.1-8B-Instruct (Meta AI, [2024](#bib.bib100 "Introducing llama 3.1: our most capable models to date")), Qwen2.5-Coder-7B-Instruct (Hui et al., [2024](#bib.bib47 "Qwen2. 5-coder technical report")), and Qwen2.5-Coder-1.5B-Instruct (Hui et al., [2024](#bib.bib47 "Qwen2. 5-coder technical report")). Table [4](#S4.T4 "Table 4 ‣ Cross-Domain Generalization. ‣ 4.2 Experiment Results ‣ 4 Experiments ‣ Think Anywhere in Code Generation") reports the average performance across four benchmarks. The results demonstrate that Think-Anywhere consistently outperforms both the base model and GRPO across all LLMs, with substantial margins over GRPO. Notably, Think-Anywhere achieves up to +13.9% improvement over the base model. Furthermore, Think-Anywhere exhibits strong scalability across different model sizes. On the smaller Qwen2.5-Coder-1.5B-Instruct, Think-Anywhere achieves a substantial improvement over the base model, indicating that our method is particularly effective for smaller models with limited capacity.

### 4.3 Ablation Study

To understand the contribution of each component in Think-Anywhere, we conduct comprehensive ablation studies comparing multiple variants on LeetCode benchmarks. We first ablate different training strategies: 1) Think-Anywhere: Our complete method incorporates both cold-start training and RLVR in a two-stage pipeline. 2) Only Cold Start: Model trained solely with supervised learning on annotated samples of Think-Anywhere, without RL phase. 3) Only RLVR: Model trained directly with RLVR of Think-Anywhere from scratch, bypassing the cold-start phase. 4) Line-level Thinking: A variant where RLVR encourages line-level thinking (similar to comment-style reasoning) rather than arbitrary token positions. 5) No Upfront Thinking: A variant of our approach that removes the initial thinking block and relies solely on Think-Anywhere within the code. To isolate the impact of the Think-Anywhere mechanism itself, we evaluate an inference variant: 6) Padding Thinking: During Think-Anywhere generation, the content within <thinkanywhere> blocks is replaced with padding tokens before continuing generation.

Table 5: The results of ablation study.

|  |  |  |
| --- | --- | --- |
| Method | Pass@1 | Δ\Delta |
| Think-Anywhere | 69.4 | – |
| Only Cold Start | 47.9 | -21.5 |
| Only RLVR | 63.4 | -6.0 |
| Line-level Thinking | 67.2 | -2.2 |
| No Upfront Thinking | 66.6 | -2.8 |
| Padding Thinking | 67.6 | -1.8 |

The results are presented in Table [5](#S4.T5 "Table 5 ‣ 4.3 Ablation Study ‣ 4 Experiments ‣ Think Anywhere in Code Generation"). We have the following observations. First, both cold-start and RLVR are essential. Removing either training stage leads to substantial performance degradation. Only Cold Start performs poorly, indicating that supervised learning alone is insufficient for the model to learn effective thinking. Only RLVR performs better but still lags behind the full method, suggesting that cold-start initialization helps stabilize RL training. Second, Line-level Thinking underperforms our token-level approach, suggesting that restricting thinking to line boundaries limits the model’s ability to invoke reasoning at optimal positions, validating our design choice of allowing thinking at arbitrary token positions.
Third, No Upfront Thinking incurs only a moderate drop (-2.8%), indicating that the primary performance gains of Think-Anywhere stem from the Think-Anywhere mechanism within the code rather than the upfront thinking phase. Finally, Padding Thinking also shows moderate performance degradation, demonstrating that the reasoning content within <thinkanywhere> blocks is indeed valuable. However, the performance does not fully deteriorate to the base model level, suggesting that identifying appropriate thinking positions is also important. Through the subsequent padding tokens, the model still performs some implicit reasoning during the forward pass (Goyal et al., [2024](#bib.bib113 "Think before you speak: training language models with pause tokens"), Pfau et al., [2024](#bib.bib112 "Let’s think dot by dot: hidden computation in transformer language models")).

### 4.4 Further Analysis

![Refer to caption](/html/2603.29957/assets/x2.png)


(a) Token entropy analysis.

![Refer to caption](/html/2603.29957/assets/x3.png)


(b) Syntactic context analysis.

Figure 2: Results of Thinking Position Analysis.

#### Thinking Position Analysis.

Understanding where Think-Anywhere chooses to invoke reasoning during code generation provides crucial insights into the model’s perception of code complexity and validates whether it truly allocates computational resources efficiently. We analyze generated solutions on the LeetCode benchmark through two perspectives: 1) Token entropy analysis: We compute the average token entropy over the n tokens following each <thinkanywhere> block and compare it against a baseline where no thinking blocks are generated, thereby quantifying the impact of <thinkanywhere> on entropy. We empirically set n=10 for entropy analysis, as this window size typically captures a statement unit, thereby mitigating individual token noise. 2) Syntactic context analysis: We employ an AST parser to identify the syntactic category of the statement enclosing each thinking position (e.g., If, While, FunctionDef, BinOp), characterizing where the model chooses to think within the code structure.

Figure [2(a)](#S4.F2.sf1 "In Figure 2 ‣ 4.4 Further Analysis ‣ 4 Experiments ‣ Think Anywhere in Code Generation") shows the distribution of entropy differences between thinking-disabled/enabled runs at positions where the model originally invoked <thinkanywhere>. We observe that the differences are predominantly positive, indicating higher entropy when thinking is disabled. This suggests that the model tends to invoke <thinkanywhere> at positions where it anticipates high uncertainty, effectively identifying challenging points in code generation.
Figure [2(b)](#S4.F2.sf2 "In Figure 2 ‣ 4.4 Further Analysis ‣ 4 Experiments ‣ Think Anywhere in Code Generation") presents the top five syntactic categories where <thinkanywhere> is invoked. The model most frequently invokes thinking at assignment statements, likely because assignments often involve complex computations or variable updates that benefit from intermediate reasoning. Return statements rank second, which we attribute to the model’s tendency to deliberate on final outputs to ensure correctness before concluding a function.

#### Computational Efficiency Comparison.

![Refer to caption](/html/2603.29957/assets/x4.png)


Figure 3: Token cost of different methods.

We evaluate the inference efficiency of Think-Anywhere by measuring the average number of tokens generated per solution. We compare Think-Anywhere against two reasoning baselines: GRPO (upfront thinking) and CoT prompting. As shown in Figure [3](#S4.F3 "Figure 3 ‣ Computational Efficiency Comparison. ‣ 4.4 Further Analysis ‣ 4 Experiments ‣ Think Anywhere in Code Generation"), Think-Anywhere consistently generates fewer tokens than both baselines across benchmarks. The reduction in total token cost is attributed to the fact that Think-Anywhere shortens the upfront thinking phase while introducing additional <thinkanywhere> tokens on demand. Since GRPO and CoT can only reason before code generation, they are forced to deliberate exhaustively at the upfront thinking stage, anticipating all potential implementation challenges upfront, which results in lengthy reasoning traces. Think-Anywhere, by contrast, invokes deliberation where it is needed. The upfront thinking phase therefore only needs to handle high-level planning, and its length is substantially reduced. The token savings from the shortened upfront thinking far outweigh the cost of the additional <thinkanywhere> blocks, resulting in a net reduction in total token usage. A detailed breakdown of the upfront thinking length and <thinkanywhere> block length is provided in Appendix [A](#A1 "Appendix A Token Cost Breakdown ‣ Think Anywhere in Code Generation").

![Refer to caption](/html/2603.29957/assets/x5.png)


Figure 4: Pass@k comparison between GRPO and Think-Anywhere across four benchmarks.

#### Pass@k Analysis.

Pass@k reflects the upper bound of a model’s capability by evaluating whether at least one correct solution exists among kk sampled candidates. We report pass@k results for both GRPO and Think-Anywhere across all benchmarks to investigate whether Think-Anywhere expands the model’s capability boundary. As shown in Figure [4](#S4.F4 "Figure 4 ‣ Computational Efficiency Comparison. ‣ 4.4 Further Analysis ‣ 4 Experiments ‣ Think Anywhere in Code Generation"), Think-Anywhere consistently outperforms GRPO across all values of kk on benchmarks. More importantly, the performance gap between Think-Anywhere and GRPO widens significantly as kk increases, particularly on LeetCode and MBPP. This widening gap demonstrates that Think-Anywhere substantially raises the model’s capability ceiling.

## 5 Conclusion

In this work, we introduce Think-Anywhere, a novel reasoning mechanism that enables LLMs to invoke thinking at any token position during code generation. Unlike conventional upfront thinking approaches that enforce a strict separation between reasoning and code implementation, Think-Anywhere allows models to deliberate precisely where complexity arises. Extensive experiments across multiple mainstream benchmarks demonstrate that Think-Anywhere achieves SOTA performance, with strong generalization across different LLMs. Beyond performance gains, our analysis reveals that LLMs naturally learn to invoke thinking at high-entropy positions, suggesting that Think-Anywhere enables adaptive computation where reasoning effort is dynamically allocated based on local complexity.
We believe Think-Anywhere opens promising directions for future research, including extending Think-Anywhere to other domains beyond code generation, and investigating how models can learn what *not* to think, further optimizing the trade-off between reasoning depth and computational efficiency.

## References

* W. U. Ahmad, S. Narenthiran, S. Majumdar, A. Ficek, S. Jain, J. Huang, V. Noroozi, and B. Ginsburg (2025)
  OpenCodeReasoning: advancing data distillation for competitive coding.
  arXiv preprint arXiv:2504.01943.
  Cited by: [§2](#S2.SS0.SSS0.Px2.p1.1 "Post-Training of LLMs for Code Generation. ‣ 2 Related Work ‣ Think Anywhere in Code Generation"),
  [§4.1](#S4.SS1.SSS0.Px3.p1.1 "Baselines. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* J. Austin, A. Odena, M. I. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. J. Cai, M. Terry, Q. V. Le, and C. Sutton (2021)
  Program synthesis with large language models.
  CoRR abs/2108.07732.
  Cited by: [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Evaluation Details. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* Y. Bai, Y. Bao, G. Chen, J. Chen, N. Chen, R. Chen, Y. Chen, Y. Chen, Y. Chen, Z. Chen, J. Cui, H. Ding, M. Dong, A. Du, C. Du, D. Du, Y. Du, Y. Fan, Y. Feng, K. Fu, B. Gao, H. Gao, P. Gao, T. Gao, X. Gu, L. Guan, H. Guo, J. Guo, H. Hu, X. Hao, T. He, W. He, W. He, C. Hong, Y. Hu, Z. Hu, W. Huang, Z. Huang, Z. Huang, T. Jiang, Z. Jiang, X. Jin, Y. Kang, G. Lai, C. Li, F. Li, H. Li, M. Li, W. Li, Y. Li, Y. Li, Z. Li, Z. Li, H. Lin, X. Lin, Z. Lin, C. Liu, C. Liu, H. Liu, J. Liu, J. Liu, L. Liu, S. Liu, T. Y. Liu, T. Liu, W. Liu, Y. Liu, Y. Liu, Y. Liu, Y. Liu, Z. Liu, E. Lu, L. Lu, S. Ma, X. Ma, Y. Ma, S. Mao, J. Mei, X. Men, Y. Miao, S. Pan, Y. Peng, R. Qin, B. Qu, Z. Shang, L. Shi, S. Shi, F. Song, J. Su, Z. Su, X. Sun, F. Sung, H. Tang, J. Tao, Q. Teng, C. Wang, D. Wang, F. Wang, and H. Wang (2025)
  Kimi K2: open agentic intelligence.
  CoRR abs/2507.20534.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Think Anywhere in Code Generation"),
  [§2](#S2.SS0.SSS0.Px1.p2.1 "Reasoning and Planning Mechanisms in LLMs. ‣ 2 Related Work ‣ Think Anywhere in Code Generation").
* M. Balunović, J. Dekoninck, I. Petrov, N. Jovanović, and M. Vechev (2025)
  MathArena: evaluating llms on uncontaminated math competitions.
   SRI Lab, ETH Zurich.
  External Links: [Link](https://matharena.ai/)
  Cited by: [§4.2](#S4.SS2.SSS0.Px2.p1.1 "Cross-Domain Generalization. ‣ 4.2 Experiment Results ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba (2021)
  Evaluating large language models trained on code.
  arXiv preprint arXiv:2107.03374.
  Cited by: [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Evaluation Details. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* Y. Dong, X. Jiang, Z. Jin, and G. Li (2024a)
  Self-collaboration code generation via chatgpt.
  ACM Trans. Softw. Eng. Methodol. 33 (7),  pp. 189:1–189:38.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Think Anywhere in Code Generation").
* Y. Dong, X. Jiang, H. Liu, Z. Jin, B. Gu, M. Yang, and G. Li (2024b)
  Generalization or memorization: data contamination and trustworthy evaluation for large language models.
  In ACL (Findings),
  Findings of ACL,  pp. 12039–12050.
  Cited by: [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Evaluation Details. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* Y. Dong, X. Jiang, J. Qian, T. Wang, K. Zhang, Z. Jin, and G. Li (2025a)
  A survey on code generation with llm-based agents.
  CoRR abs/2508.00083.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Think Anywhere in Code Generation").
* Y. Dong, X. Jiang, Y. Tao, H. Liu, K. Zhang, L. Mou, R. Cao, Y. Ma, J. Chen, B. Li, Z. Jin, F. Huang, Y. Li, and G. Li (2025b)
  RL-PLUS: countering capability boundary collapse of llms in reinforcement learning with hybrid-policy optimization.
  CoRR abs/2508.00222.
  Cited by: [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Evaluation Details. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* Google DeepMind (2025)
  Gemini 3 flash model card.
  Note: <https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Flash-Model-Card.pdf>Model card published December 2025
  Cited by: [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Training Details. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* S. Goyal, Z. Ji, A. S. Rawat, A. K. Menon, S. Kumar, and V. Nagarajan (2024)
  Think before you speak: training language models with pause tokens.
  In The Twelfth International Conference on Learning Representations,
  External Links: [Link](https://openreview.net/forum?id=ph04CRkPdC)
  Cited by: [§4.3](#S4.SS3.p2.1 "4.3 Ablation Study ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, X. Zhang, X. Yu, Y. Wu, Z. F. Wu, Z. Gou, Z. Shao, Z. Li, Z. Gao, A. Liu, B. Xue, B. Wang, B. Wu, B. Feng, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, D. Dai, D. Chen, D. Ji, E. Li, F. Lin, F. Dai, F. Luo, G. Hao, G. Chen, G. Li, H. Zhang, H. Bao, H. Xu, H. Wang, H. Ding, H. Xin, H. Gao, H. Qu, H. Li, J. Guo, J. Li, J. Wang, J. Chen, J. Yuan, J. Qiu, J. Li, J. L. Cai, J. Ni, J. Liang, J. Chen, K. Dong, K. Hu, K. Gao, K. Guan, K. Huang, K. Yu, L. Wang, L. Zhang, L. Zhao, L. Wang, L. Zhang, L. Xu, L. Xia, M. Zhang, M. Zhang, M. Tang, M. Li, M. Wang, M. Li, N. Tian, P. Huang, P. Zhang, Q. Wang, Q. Chen, Q. Du, R. Ge, R. Zhang, R. Pan, R. Wang, R. J. Chen, R. L. Jin, R. Chen, S. Lu, S. Zhou, S. Chen, S. Ye, S. Wang, S. Yu, S. Zhou, S. Pan, and S. S. Li (2025a)
  DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning.
  Nature 645 (8081),  pp. 633.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Think Anywhere in Code Generation"),
  [§2](#S2.SS0.SSS0.Px1.p2.1 "Reasoning and Planning Mechanisms in LLMs. ‣ 2 Related Work ‣ Think Anywhere in Code Generation"),
  [§3.1](#S3.SS1.SSS0.Px1.p1.7 "Upfront Thinking. ‣ 3.1 Defining Think-Anywhere ‣ 3 Methodology ‣ Think Anywhere in Code Generation").
* D. Guo, Q. Zhu, D. Yang, Z. Xie, K. Dong, W. Zhang, G. Chen, X. Bi, Y. Wu, Y. K. Li, F. Luo, Y. Xiong, and W. Liang (2024)
  DeepSeek-coder: when the large language model meets programming–the rise of code intelligence.
  arXiv preprint arXiv:2401.14196.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Think Anywhere in Code Generation").
* Z. Guo, R. Zhang, H. Li, M. Zhang, X. Chen, S. Wang, Y. Feng, P. Pei, and P. Heng (2025b)
  Thinking-while-generating: interleaving textual reasoning throughout visual generation.
  External Links: 2511.16671,
  [Link](https://arxiv.org/abs/2511.16671)
  Cited by: [§2](#S2.SS0.SSS0.Px1.p1.1 "Reasoning and Planning Mechanisms in LLMs. ‣ 2 Related Work ‣ Think Anywhere in Code Generation").
* J. He, J. Liu, C. Y. Liu, R. Yan, C. Wang, P. Cheng, X. Zhang, F. Zhang, J. Xu, W. Shen, S. Li, L. Zeng, T. Wei, C. Cheng, B. An, Y. Liu, and Y. Zhou (2025)
  Skywork open reasoner 1 technical report.
  arXiv preprint arXiv:2505.22312.
  Cited by: [§2](#S2.SS0.SSS0.Px2.p1.1 "Post-Training of LLMs for Code Generation. ‣ 2 Related Work ‣ Think Anywhere in Code Generation"),
  [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Training Details. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen (2022)
  LoRA: low-rank adaptation of large language models.
  In ICLR,
  Cited by: [§3.2](#S3.SS2.SSS0.Px1.p3.1 "Automatic Data Construction. ‣ 3.2 Cold Start for Think-Anywhere ‣ 3 Methodology ‣ Think Anywhere in Code Generation").
* Hugging Face (2025)
  Open r1: a fully open reproduction of deepseek-r1.
  External Links: [Link](https://github.com/huggingface/open-r1)
  Cited by: [§2](#S2.SS0.SSS0.Px2.p1.1 "Post-Training of LLMs for Code Generation. ‣ 2 Related Work ‣ Think Anywhere in Code Generation"),
  [§4.1](#S4.SS1.SSS0.Px3.p1.1 "Baselines. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* B. Hui, J. Yang, Z. Cui, J. Yang, D. Liu, L. Zhang, T. Liu, J. Zhang, B. Yu, K. Dang, A. Yang, R. Men, F. Huang, X. Ren, X. Ren, J. Zhou, and J. Lin (2024)
  Qwen2. 5-coder technical report.
  arXiv preprint arXiv:2409.12186.
  Cited by: [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Training Details. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Think Anywhere in Code Generation"),
  [§4.2](#S4.SS2.SSS0.Px3.p1.1 "Application on Various LLMs. ‣ 4.2 Experiment Results ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney, A. Iftimie, A. Karpenko, A. T. Passos, A. Neitz, A. Prokofiev, A. Wei, A. Tam, A. Bennett, A. Kumar, A. Saraiva, A. Vallone, A. Duberstein, A. Kondrich, A. Mishchenko, A. Applebaum, A. Jiang, A. Nair, B. Zoph, B. Ghorbani, B. Rossen, B. Sokolowsky, B. Barak, B. McGrew, B. Minaiev, B. Hao, B. Baker, B. Houghton, B. McKinzie, B. Eastman, C. Lugaresi, C. Bassin, C. Hudson, C. M. Li, C. de Bourcy, C. Voss, C. Shen, C. Zhang, C. Koch, C. Orsinger, C. Hesse, C. Fischer, C. Chan, D. Roberts, D. Kappler, D. Levy, D. Selsam, D. Dohan, D. Farhi, D. Mely, D. Robinson, D. Tsipras, D. Li, D. Oprica, E. Freeman, E. Zhang, E. Wong, E. Proehl, E. Cheung, E. Mitchell, E. Wallace, E. Ritter, E. Mays, F. Wang, F. P. Such, F. Raso, F. Leoni, F. Tsimpourlas, F. Song, F. von Lohmann, F. Sulit, G. Salmon, G. Parascandolo, G. Chabot, G. Zhao, G. Brockman, G. Leclerc, H. Salman, H. Bao, H. Sheng, H. Andrin, H. Bagherinezhad, H. Ren, H. Lightman, H. W. Chung, I. Kivlichan, I. O’Connell, I. Osband, I. C. Gilaberte, and I. Akkaya (2024)
  OpenAI o1 system card.
  CoRR abs/2412.16720.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Think Anywhere in Code Generation"),
  [§3.1](#S3.SS1.SSS0.Px1.p1.7 "Upfront Thinking. ‣ 3.1 Defining Think-Anywhere ‣ 3 Methodology ‣ Think Anywhere in Code Generation").
* N. Jain, K. Han, A. Gu, W. Li, F. Yan, T. Zhang, S. Wang, A. Solar-Lezama, K. Sen, and I. Stoica (2024)
  Livecodebench: holistic and contamination free evaluation of large language models for code.
  arXiv preprint arXiv:2403.07974.
  Cited by: [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Evaluation Details. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* X. Jiang, Y. Dong, M. Liu, H. Deng, T. Wang, Y. Tao, R. Cao, B. Li, Z. Jin, W. Jiao, F. Huang, Y. Li, and G. Li (2025)
  CodeRL+: improving code generation via reinforcement with execution semantics alignment.
  CoRR abs/2510.18471.
  Cited by: [§2](#S2.SS0.SSS0.Px2.p1.1 "Post-Training of LLMs for Code Generation. ‣ 2 Related Work ‣ Think Anywhere in Code Generation"),
  [§4.1](#S4.SS1.SSS0.Px3.p1.1 "Baselines. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* X. Jiang, Y. Dong, L. Wang, Z. Fang, Q. Shang, G. Li, Z. Jin, and W. Jiao (2024)
  Self-planning code generation with large language models.
  ACM Trans. Softw. Eng. Methodol. 33 (7),  pp. 182:1–182:30.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Think Anywhere in Code Generation"),
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Reasoning and Planning Mechanisms in LLMs. ‣ 2 Related Work ‣ Think Anywhere in Code Generation"),
  [§4.1](#S4.SS1.SSS0.Px3.p1.1 "Baselines. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* X. Jiang, J. Qian, X. Shi, C. Li, H. Zhu, Z. Wang, J. Zhang, Z. Zhao, K. Zhang, J. Li, W. Jiao, Z. Jin, G. Li, and Y. Dong (2026)
  KOCO-BENCH: can large language models leverage domain knowledge in software development?.
  CoRR abs/2601.13240.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Think Anywhere in Code Generation").
* T. Kojima, S. S. Gu, M. Reid, Y. Matsuo, and Y. Iwasawa (2022)
  Large language models are zero-shot reasoners.
  In NeurIPS,
  Cited by: [§2](#S2.SS0.SSS0.Px1.p1.1 "Reasoning and Planning Mechanisms in LLMs. ‣ 2 Related Work ‣ Think Anywhere in Code Generation").
* J. Li, D. Guo, D. Yang, R. Xu, Y. Wu, and J. He (2025a)
  CodeI/o: condensing reasoning patterns via code input-output prediction.
  CoRR abs/2502.07316.
  Cited by: [§3.2](#S3.SS2.SSS0.Px1.p2.1 "Automatic Data Construction. ‣ 3.2 Cold Start for Think-Anywhere ‣ 3 Methodology ‣ Think Anywhere in Code Generation").
* Q. Li, X. Dai, X. Li, W. Zhang, Y. Wang, R. Tang, and Y. Yu (2025b)
  CodePRM: execution feedback-enhanced process reward model for code generation.
  In Findings of the Association for Computational Linguistics: ACL 2025,
   pp. 8169–8182.
  Cited by: [§2](#S2.SS0.SSS0.Px2.p1.1 "Post-Training of LLMs for Code Generation. ‣ 2 Related Work ‣ Think Anywhere in Code Generation"),
  [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Evaluation Details. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Think Anywhere in Code Generation"),
  [§4.1](#S4.SS1.SSS0.Px3.p1.1 "Baselines. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* A. Liang, J. Berant, A. Fisch, A. Goyal, K. Krishna, and J. Eisenstein (2025)
  Plantain: plan-answer interleaved reasoning.
  External Links: 2512.03176,
  [Link](https://arxiv.org/abs/2512.03176)
  Cited by: [§2](#S2.SS0.SSS0.Px1.p1.1 "Reasoning and Planning Mechanisms in LLMs. ‣ 2 Related Work ‣ Think Anywhere in Code Generation").
* A. Lozhkov, R. Li, L. B. Allal, F. Cassano, J. Lamy-Poirier, N. Tazi, A. Tang, D. Pykhtar, J. Liu, Y. Wei, T. Liu, M. Tian, D. Kocetkov, A. Zucker, Y. Belkada, Z. Wang, Q. Liu, D. Abulkhanov, I. Paul, Z. Li, W. Li, M. Risdal, J. Li, J. Zhu, T. Y. Zhuo, E. Zheltonozhskii, N. O. O. Dade, W. Yu, L. Krauß, N. Jain, Y. Su, X. He, M. Dey, E. Abati, Y. Chai, N. Muennighoff, X. Tang, M. Oblokulov, C. Akiki, M. Marone, C. Mou, M. Mishra, A. Gu, B. Hui, T. Dao, A. Zebaze, O. Dehaene, N. Patry, C. Xu, J. McAuley, H. Hu, T. Scholak, S. Paquet, J. Robinson, C. J. Anderson, N. Chapados, M. Patwary, N. Tajbakhsh, Y. Jernite, C. M. Ferrandis, L. Zhang, S. Hughes, T. Wolf, A. Guha, L. von Werra, and H. de Vries (2024)
  StarCoder 2 and the stack v2: the next generation.
  External Links: 2402.19173,
  [Link](https://arxiv.org/abs/2402.19173)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Think Anywhere in Code Generation").
* Mathematical Association of America (2024)
  American invitational mathematics examination (aime) 2024.
  Note: <https://maa.org/maa-invitational-competitions/>
  Cited by: [§4.2](#S4.SS2.SSS0.Px2.p1.1 "Cross-Domain Generalization. ‣ 4.2 Experiment Results ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* Mathematical Association of America (2025)
  American invitational mathematics examination (aime) 2025.
  Note: <https://maa.org/maa-invitational-competitions/>
  Cited by: [§4.2](#S4.SS2.SSS0.Px2.p1.1 "Cross-Domain Generalization. ‣ 4.2 Experiment Results ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* Meta AI (2024)
  Introducing llama 3.1: our most capable models to date.
  Note: <https://ai.meta.com/blog/meta-llama-3-1/>Accessed: 2025-10-06
  Cited by: [§4.2](#S4.SS2.SSS0.Px3.p1.1 "Application on Various LLMs. ‣ 4.2 Experiment Results ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* J. Pfau, W. Merrill, and S. R. Bowman (2024)
  Let’s think dot by dot: hidden computation in transformer language models.
  In First Conference on Language Modeling,
  External Links: [Link](https://openreview.net/forum?id=NikbrdtYvG)
  Cited by: [§4.3](#S4.SS3.p2.1 "4.3 Ablation Study ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* B. Rozière, J. Gehring, F. Gloeckle, S. Sootla, I. Gat, X. E. Tan, Y. Adi, J. Liu, R. Sauvestre, T. Remez, J. Rapin, A. Kozhevnikov, I. Evtimov, J. Bitton, M. Bhatt, C. C. Ferrer, A. Grattafiori, W. Xiong, A. Défossez, J. Copet, F. Azhar, H. Touvron, L. Martin, N. Usunier, T. Scialom, and G. Synnaeve (2023)
  Code llama: open foundation models for code.
  arXiv preprint arXiv:2308.12950.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Think Anywhere in Code Generation").
* J. Schulman and T. M. Lab (2025)
  LoRA without regret.
  Thinking Machines Lab: Connectionism.
  Note: https://thinkingmachines.ai/blog/lora/
  External Links: [Document](https://dx.doi.org/10.64434/tml.20250929)
  Cited by: [§3.2](#S3.SS2.SSS0.Px1.p3.1 "Automatic Data Construction. ‣ 3.2 Cold Start for Think-Anywhere ‣ 3 Methodology ‣ Think Anywhere in Code Generation").
* J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov (2017)
  Proximal policy optimization algorithms.
  arXiv preprint arXiv:1707.06347.
  Cited by: [§3.3](#S3.SS3.SSS0.Px1.p1.1 "Reinforcement Learning Algorithm. ‣ 3.3 RLVR for Think-Anywhere ‣ 3 Methodology ‣ Think Anywhere in Code Generation").
* Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, M. Zhang, Y. K. Li, Y. Wu, and D. Guo (2024)
  DeepSeekMath: pushing the limits of mathematical reasoning in open language models.
  CoRR abs/2402.03300.
  Cited by: [§3.3](#S3.SS3.SSS0.Px1.p1.1 "Reinforcement Learning Algorithm. ‣ 3.3 RLVR for Think-Anywhere ‣ 3 Methodology ‣ Think Anywhere in Code Generation"),
  [§4.1](#S4.SS1.SSS0.Px3.p1.1 "Baselines. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* G. Sheng, C. Zhang, Z. Ye, X. Wu, W. Zhang, R. Zhang, Y. Peng, H. Lin, and C. Wu (2024)
  HybridFlow: a flexible and efficient rlhf framework.
  arXiv preprint arXiv: 2409.19256.
  Cited by: [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Training Details. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* L. Tang, H. Ye, Z. Liu, X. Ren, and L. Bao (2025)
  CodeReasoner: enhancing the code reasoning ability with reinforcement learning.
  arXiv preprint arXiv:2507.17548.
  Cited by: [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Evaluation Details. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* S. Wang, Q. Guo, K. Zhao, Y. Zhang, X. Li, X. Li, S. Li, R. She, S. Yu, and W. P. Tay (2025)
  CodeBoost: boosting code LLMs by squeezing knowledge from code snippets with rl.
  arXiv preprint arXiv:2508.05242.
  Cited by: [§2](#S2.SS0.SSS0.Px2.p1.1 "Post-Training of LLMs for Code Generation. ‣ 2 Related Work ‣ Think Anywhere in Code Generation"),
  [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Evaluation Details. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Think Anywhere in Code Generation"),
  [§4.1](#S4.SS1.SSS0.Px3.p1.1 "Baselines. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* X. Wang, J. Wei, D. Schuurmans, Q. V. Le, E. H. Chi, S. Narang, A. Chowdhery, and D. Zhou (2023)
  Self-consistency improves chain of thought reasoning in language models.
  In ICLR,
  Cited by: [§2](#S2.SS0.SSS0.Px1.p1.1 "Reasoning and Planning Mechanisms in LLMs. ‣ 2 Related Work ‣ Think Anywhere in Code Generation").
* J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. H. Chi, Q. V. Le, and D. Zhou (2022)
  Chain-of-thought prompting elicits reasoning in large language models.
  Advances in neural information processing systems 35,  pp. 24824–24837.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Think Anywhere in Code Generation"),
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Reasoning and Planning Mechanisms in LLMs. ‣ 2 Related Work ‣ Think Anywhere in Code Generation"),
  [§4.1](#S4.SS1.SSS0.Px3.p1.1 "Baselines. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* Y. Xia, W. Shen, Y. Wang, J. K. Liu, H. Sun, S. Wu, J. Hu, and X. Xu (2025)
  Leetcodedataset: a temporal dataset for robust evaluation and efficient training of code llms.
  arXiv preprint arXiv:2504.14655.
  Cited by: [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Evaluation Details. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* R. Xie, D. Qiu, D. Gopinath, D. Lin, Y. Sun, C. Wang, S. Potdar, and B. Dhingra (2025)
  Interleaved reasoning for large language models via reinforcement learning.
  CoRR abs/2505.19640.
  Cited by: [§2](#S2.SS0.SSS0.Px1.p1.1 "Reasoning and Planning Mechanisms in LLMs. ‣ 2 Related Work ‣ Think Anywhere in Code Generation"),
  [§4.1](#S4.SS1.SSS0.Px3.p1.1 "Baselines. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Think Anywhere in Code Generation").
* S. Yao, D. Yu, J. Zhao, I. Shafran, T. Griffiths, Y. Cao, and K. Narasimhan (2023)
  Tree of thoughts: deliberate problem solving with large language models.
  In NeurIPS,
  Cited by: [§2](#S2.SS0.SSS0.Px1.p1.1 "Reasoning and Planning Mechanisms in LLMs. ‣ 2 Related Work ‣ Think Anywhere in Code Generation").
* D. Zhou, N. Schärli, L. Hou, J. Wei, N. Scales, X. Wang, D. Schuurmans, C. Cui, O. Bousquet, Q. V. Le, and E. H. Chi (2023)
  Least-to-most prompting enables complex reasoning in large language models.
  In ICLR,
  Cited by: [§2](#S2.SS0.SSS0.Px1.p1.1 "Reasoning and Planning Mechanisms in LLMs. ‣ 2 Related Work ‣ Think Anywhere in Code Generation").

## Appendix A Token Cost Breakdown

Table [6](#A1.T6 "Table 6 ‣ Appendix A Token Cost Breakdown ‣ Think Anywhere in Code Generation") provides a detailed breakdown of reasoning token costs. For GRPO and CoT, the token cost consists entirely of upfront thinking tokens. For Think-Anywhere, we separately report the upfront thinking length and the <thinkanywhere> block length. The upfront thinking phase of Think-Anywhere is substantially shorter than that of GRPO and CoT across benchmarks, and the additional <thinkanywhere> tokens are modest in comparison, resulting in a net reduction in total reasoning token usage.

Table 6: Breakdown of reasoning token costs. For Think-Anywhere, the two numbers denote upfront thinking length + <thinkanywhere> block length.

| Method | HumanEval | MBPP | LeetCode |
| --- | --- | --- | --- |
| GRPO | 309.4 | 325.2 | 440.7 |
| CoT | 348.8 | 372.0 | 577.0 |
| Think-Anywhere | 215.6 + 22.5 | 183.2 + 23.2 | 283.0 + 22.9 |

## Appendix B Thinking Block Statistics Across Training Stages

To clarify the respective contributions of cold-start SFT and RLVR, we analyze the average frequency (Avg.Freq) and average length (Avg.Len) of <thinkanywhere> blocks across different training stages, as shown in Table [7](#A2.T7 "Table 7 ‣ Appendix B Thinking Block Statistics Across Training Stages ‣ Think Anywhere in Code Generation").

The base model never invokes thinking blocks during code generation. Even with explicit prompting (Think-Anywhere Prompting), the model produces very few <thinkanywhere> blocks (frequency near zero) with abnormally long lengths, indicating that prompting alone cannot reliably elicit Think-Anywhere behavior and that this capability is unlikely to originate from pre-training. After cold-start SFT, the model generates <thinkanywhere> blocks at a normal frequency and length, demonstrating that SFT successfully teaches the model to imitate the Think-Anywhere reasoning pattern and establishes a solid foundation for subsequent RL training. After RL training, the frequency and length of <thinkanywhere> blocks decrease slightly compared to the SFT stage, while pass@1 improves substantially (as shown in Table [5](#S4.T5 "Table 5 ‣ 4.3 Ablation Study ‣ 4 Experiments ‣ Think Anywhere in Code Generation")). This indicates that RL does not simply increase the number of thinking tokens; rather, it refines the model’s ability to invoke reasoning on demand, enabling more concise and targeted deliberation at positions where it is truly needed. This is fully consistent with the design goal of Think-Anywhere.

Table 7: Average frequency and length of <thinkanywhere> blocks across training stages.

|  |  |  |  |
| --- | --- | --- | --- |
| Dataset | Model | Avg.Freq | Avg.Len |
| HumanEval | Base Model | 0 | 0 |
| Think-Anywhere (Prompting) | 0.24 | 113.5 |
| Think-Anywhere (SFT) | 6.69 | 31.9 |
| Think-Anywhere (Ours) | 6.15 | 22.5 |
| MBPP | Base Model | 0 | 0 |
| Think-Anywhere (Prompting) | 0.53 | 66.4 |
| Think-Anywhere (SFT) | 5.76 | 33.4 |
| Think-Anywhere (Ours) | 5.24 | 23.2 |
| LeetCode | Base Model | 0 | 0 |
| Think-Anywhere (Prompting) | 0.31 | 219.7 |
| Think-Anywhere (SFT) | 11.28 | 34.5 |
| Think-Anywhere (Ours) | 11.26 | 22.9 |

[◄](/html/2603.29955)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2603.29957)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2603.29957)
[View original  
on arXiv](https://arxiv.org/abs/2603.29957)[►](/html/2603.29958)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Sun Apr 5 21:59:13 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
