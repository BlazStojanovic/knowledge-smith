---
arxiv: '2603.26535'
authors:
- Zelin Tan
- Zhouliang Yu
- Bohan Lin
- Zijie Geng
- Hejia Geng
- Yudong Zhang
- Mulei Zhang
- Yang Chen
- Shuyue Hu
- Zhenfei Yin
- Chen Zhang
- Lei Bai
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization'
url: https://arxiv.org/abs/2603.26535
year: 2026
---

[2603.26535] PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization














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



# PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization

Zelin Tan1,2  Zhouliang Yu3  Bohan Lin1  Zijie Geng1  Hejia Geng4  Yudong Zhang1
  
Mulei Zhang5  Yang Chen2  Shuyue Hu2  Zhenfei Yin4†  Chen Zhang2†  Lei Bai2
  
1University of Science and Technology of China  2Shanghai Artificial Intelligence Laboratory
  
3The Chinese University of Hong Kong  4University of Oxford  5Wuhan University
  
†Corresponding authors
  
 [Code](https://github.com/tanzelin430/PAPO)
  [Dataset](https://huggingface.co/datasets/Artemis0430/NuminaMath-20k-Stratified)

###### Abstract

We propose Process-Aware Policy Optimization (PAPO), a method that integrates process-level evaluation into Group Relative Policy Optimization (GRPO) through decoupled advantage normalization, to address two limitations of existing reward designs. Outcome reward models (ORM) evaluate only final-answer correctness, treating all correct responses identically regardless of reasoning quality, and gradually lose the advantage signal as groups become uniformly correct. Process reward models (PRM) offer richer supervision, but directly using PRM scores causes reward hacking, where models exploit verbosity to inflate scores while accuracy collapses. PAPO resolves both by composing the advantage from an outcome component AoutA\_{\text{out}}, derived from ORM and normalized over all responses, and a process component AprocA\_{\text{proc}}, derived from a rubric-based PRM and normalized exclusively among correct responses. This decoupled design ensures that AoutA\_{\text{out}} anchors training on correctness while AprocA\_{\text{proc}} differentiates reasoning quality without distorting the outcome signal. Experiments across multiple model scales and six benchmarks demonstrate that PAPO consistently outperforms ORM, reaching 51.3% vs. 46.3% on OlympiadBench while continuing to improve as ORM plateaus and declines.

## 1 Introduction

As large language models (LLMs) (Yang et al., [2025b](#bib.bib1 "Qwen2.5 technical report"); Kamath et al., [2025](#bib.bib3 "Gemma 3 technical report"); Touvron et al., [2023](#bib.bib4 "LLaMA: open and efficient foundation language models")) continue to advance, reasoning has become a key capability for accomplishing complex tasks (Guo et al., [2025](#bib.bib6 "DeepSeek-r1 incentivizes reasoning in llms through reinforcement learning")). Reinforcement learning (RL) plays an essential role in improving the reasoning capabilities of LLMs, especially in mathematics (Shao et al., [2024](#bib.bib5 "DeepSeekMath: pushing the limits of mathematical reasoning in open language models"), [2025](#bib.bib7 "DeepSeekMath-v2: towards self-verifiable mathematical reasoning")). Among RL methods, Group Relative Policy Optimization (GRPO) (Shao et al., [2024](#bib.bib5 "DeepSeekMath: pushing the limits of mathematical reasoning in open language models")) with outcome reward models (ORM) has emerged as a core approach in rule-based verifiable reinforcement learning for its simplicity. It eliminates the learned critic network used in PPO (Schulman et al., [2017](#bib.bib8 "Proximal policy optimization algorithms")), instead computing advantages through group-level reward normalization. However, GRPO with ORM evaluates only final-answer correctness, providing no signal on reasoning process quality, which leads to two critical issues.

First, all correct responses are assigned identical advantage irrespective of reasoning quality, resulting in the same credit assignment for guessed answers and answers derived via rigorous step-by-step reasoning. Second, as the model improves during training, an increasing proportion of response groups become uniformly correct, yielding zero advantage and zero gradient. This progressive loss of informative signal causes performance to plateau and eventually decline, as shown in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization")a.

Process reward models (PRM) (Zheng et al., [2023](#bib.bib14 "Judging llm-as-a-judge with mt-bench and chatbot arena"); Lightman et al., [2023](#bib.bib12 "Let’s verify step by step"); Wang et al., [2024](#bib.bib13 "Math-shepherd: verify and reinforce llms step-by-step without human annotations")) offer a natural remedy by evaluating reasoning quality. In particular, rubric-based evaluation (Yuan et al., [2025](#bib.bib16 "Curing miracle steps in llm mathematical reasoning with rubric rewards"); Shao et al., [2025](#bib.bib7 "DeepSeekMath-v2: towards self-verifiable mathematical reasoning"); Sheng et al., [2026](#bib.bib10 "Reinforcing chain-of-thought reasoning with self-evolving rubrics"); Yang et al., [2026](#bib.bib11 "Proof-rm: a scalable and generalizable reward model for math proof")) through the LLM-as-Judge paradigm has attracted growing interest as a scalable form of process supervision that requires no step-level annotations. However, we find that *directly integrating rubric-based PRM into GRPO as rewards leads to reward hacking (Skalse et al., [2022](#bib.bib19 "Defining and characterizing reward hacking"))*, where models learn to inflate PRM scores by generating increasingly verbose responses, ultimately causing accuracy to collapse. Even a naive multiplicative combination of ORM and PRM fails to surpass ORM, as the process signal is suppressed within a single normalization pass, as shown in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization")b.

![Refer to caption](/html/2603.26535/assets/x1.png)


(a) OlympiadBench evaluation results.

![Refer to caption](/html/2603.26535/assets/x2.png)


(b) Accuracy on competition math.

Figure 1: (a) ORM (blue) plateaus and declines after step 750 due to signal exhaustion; PRM (red, dashed) collapses via reward hacking; ORM×\timesPRM (purple, dash-dotted) tracks ORM closely without exceeding it; PAPO (green) continues improving throughout training, reaching 51.3%. (b) Comparison across AIME 2024/2025, OlympiadBench, and their average. Naive multiplicative combination (ORM×\timesPRM) barely improves over ORM, while PAPO’s decoupled normalization yields substantial gains on all benchmarks.

To solve this problem, we propose Process-Aware Policy Optimization (PAPO), which integrates rubric-based process evaluation into GRPO through decoupled advantage normalization. PAPO constructs the advantage from two independently normalized components:

* •

  An outcome advantage AoutA\_{\text{out}}, computed from binary ORM rewards via standard GRPO normalization, anchoring the training signal on answer correctness.
* •

  A process advantage AprocA\_{\text{proc}}, computed from rubric-based PRM scores normalized *exclusively among correct responses*, differentiating reasoning quality within the correct group.

This decoupled formulation mitigates reward hacking by separating PRM-based reasoning rewards from the final-answer signal. As a result, AprocA\_{\text{proc}} can still provide non-zero gradients even when all responses are correct. Importantly, normalization is performed only over correct responses, ensuring that PRM scores do not distort the outcome signal and that incorrect responses are not rewarded solely for strong reasoning traces.

As shown in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization")a, PAPO continues improving throughout training, reaching 51.3% on OlympiadBench (He et al., [2024](#bib.bib22 "OlympiadBench: a challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems")) while ORM peaks at 46.3% and declines, and Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization")b shows that the improvement generalizes to other math competition benchmarks like AIME (Balunović et al., [2025](#bib.bib20 "MathArena: evaluating llms on uncontaminated math competitions")).

To summarize, our contributions are as follows:

1. 1.

   We identify a dilemma in GRPO reward design: binary ORM lacks process supervision and suffers from signal exhaustion, yet naively using PRM scores causes reward hacking and training collapse.
2. 2.

   We propose PAPO, which composes the advantage from independently normalized outcome and process components with correct-subset normalization, resolving this dilemma.
3. 3.

   We show that PAPO consistently outperforms ORM across model scales from 3B to 14B and six benchmarks, including larger models where ORM is already strong.

## 2 Background

### 2.1 Group Relative Policy Optimization

GRPO (Shao et al., [2024](#bib.bib5 "DeepSeekMath: pushing the limits of mathematical reasoning in open language models")) is a variant of policy gradient methods for LLM fine-tuning that eliminates the critic network used in PPO (Schulman et al., [2017](#bib.bib8 "Proximal policy optimization algorithms")). For each prompt xx, the model samples a group of GG responses {o1,…,oG}\{o\_{1},\ldots,o\_{G}\} from the old policy πθold\pi\_{\theta\_{\text{old}}}, and the objective is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒGRPO=1G​∑i=1G1|oi|​∑t=1|oi|{min⁡[ρ​(θ)​A^i,t,clip⁡(ρ​(θ),1−ε, 1+ε)​A^i,t]−β​DKL},\begin{aligned} \mathcal{L}\_{\text{GRPO}}=\frac{1}{G}\sum\_{i=1}^{G}\frac{1}{|o\_{i}|}\sum\_{t=1}^{|o\_{i}|}\Bigl\{\min\Bigl[\rho(\theta)\,\hat{A}\_{i,t},\;\operatorname{clip}\!\Bigl(\rho(\theta),1\!-\!\varepsilon,\,1\!+\!\varepsilon\Bigr)\hat{A}\_{i,t}\Bigr]-\beta\,\mathrm{D}\_{\mathrm{KL}}\Bigr\},\end{aligned} |  | (1) |

where ρ​(θ)=πθ​(oi,t|x,oi,<t)/πθold​(oi,t|x,oi,<t)\rho(\theta)=\pi\_{\theta}(o\_{i,t}|x,o\_{i,<t})/\pi\_{\theta\_{\text{old}}}(o\_{i,t}|x,o\_{i,<t}) is the importance sampling ratio. Each response oio\_{i} receives a scalar reward rir\_{i}, and the advantage is computed via group-level normalization:

|  |  |  |  |
| --- | --- | --- | --- |
|  | A^i,t=ri−mean​(𝐫)std​(𝐫).\hat{A}\_{i,t}=\frac{r\_{i}-\mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r})}. |  | (2) |

By replacing the learned value function with sample-based normalization, GRPO avoids training a separate critic model, substantially reducing memory and compute requirements.

### 2.2 Reward Models for Mathematical Reasoning

#### Outcome Reward Models (ORM).

An ORM provides a binary signal based solely on the final answer: rout=𝟏​[a^=a∗]r^{\text{out}}=\mathbf{1}[\hat{a}=a^{\*}], where a^\hat{a} is the predicted answer and a∗a^{\*} is the ground truth. For mathematical reasoning, ORMs are typically implemented as rule-based answer checkers that compare extracted answers against reference solutions (Hendrycks et al., [2021](#bib.bib21 "Measuring mathematical problem solving with the math dataset")). While reliable and deterministic, ORMs provide no information about the quality of the reasoning process.

#### Process Reward Models (PRM).

PRMs evaluate the quality of intermediate reasoning steps (Zheng et al., [2023](#bib.bib14 "Judging llm-as-a-judge with mt-bench and chatbot arena"); Lightman et al., [2023](#bib.bib12 "Let’s verify step by step"); Wang et al., [2024](#bib.bib13 "Math-shepherd: verify and reinforce llms step-by-step without human annotations")). We focus on *rubric-based PRMs* implemented via the LLM-as-Judge paradigm (Yuan et al., [2025](#bib.bib16 "Curing miracle steps in llm mathematical reasoning with rubric rewards"); Sheng et al., [2026](#bib.bib10 "Reinforcing chain-of-thought reasoning with self-evolving rubrics"); Yang et al., [2026](#bib.bib11 "Proof-rm: a scalable and generalizable reward model for math proof")), where a capable LLM evaluates the solution against a scoring rubric. In our setting, a rubric-based PRM assigns a score rproc∈{0,0.5,1.0}r^{\text{proc}}\in\{0,0.5,1.0\} reflecting whether the reasoning is fully correct, scored as 1.0, largely correct with minor issues, scored as 0.5, or fatally flawed, scored as 0.0. The complete rubric prompt is provided in Appendix [A.3](#A1.SS3 "A.3 Rubric Prompt ‣ Appendix A Implementation Details ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").

Rubric-based PRMs offer several advantages: they require no step-level annotations, provide interpretable feedback, and can leverage the growing capabilities of frontier LLMs. However, as we demonstrate in §[3](#S3 "3 Empirical analysis of PRM and ORM ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"), naive integration of PRM scores into GRPO leads to training instability.

## 3 Empirical analysis of PRM and ORM

In this section, we identify two complementary failure modes of reward signal design in GRPO for mathematical reasoning: *signal exhaustion* with outcome-only rewards, discussed in §[3.1](#S3.SS1 "3.1 Signal Exhaustion with Binary ORM ‣ 3 Empirical analysis of PRM and ORM ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"), and *reward hacking* with process-only rewards, discussed in §[3.2](#S3.SS2 "3.2 Reward Hacking with Direct PRM Integration ‣ 3 Empirical analysis of PRM and ORM ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").

### 3.1 Signal Exhaustion with Binary ORM

When GRPO uses binary ORM rewards (ri∈{0,1}r\_{i}\in\{0,1\}), the group normalization in Eq. [2](#S2.E2 "In 2.1 Group Relative Policy Optimization ‣ 2 Background ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization") assigns identical advantage to all correct responses and identical (negative) advantage to all incorrect responses. This has two consequences.

#### Lack of quality differentiation.

A response that arrives at the correct answer through rigorous reasoning receives the same advantage as one that reaches it through guesswork or flawed reasoning with a lucky cancellation of errors. The model has no incentive to improve reasoning quality beyond what is needed to produce correct final answers.

#### Vanishing advantage from uniform groups.

When all GG responses in a group are correct, or all incorrect,
the rewards are identical, so std​(𝐫)=0\mathrm{std}(\mathbf{r})=0
in Eq. [2](#S2.E2 "In 2.1 Group Relative Policy Optimization ‣ 2 Background ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization") and the advantage is set to zero for
the entire group. As the model’s capability improves during training, the
proportion of such uniform groups grows, causing an increasing
fraction of training samples to receive zero advantage. We refer
to this progressive loss of informative signal as *signal
exhaustion*. Figure [4](#S5.F4 "Figure 4 ‣ 5.3 Advantage Signal Analysis ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization")a confirms this
empirically: on Qwen2.5-7B, the fraction of zero-advantage
samples rises from approximately 40% to 69% over training,
coinciding with the accuracy plateau and further decline in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").

### 3.2 Reward Hacking with Direct PRM Integration

A natural solution to ORM’s limitations is to replace binary rewards with rubric-based PRM scores that capture both answer correctness and reasoning quality in a single continuous signal. However, directly using PRM scores as the GRPO reward (ri=riprocr\_{i}=r\_{i}^{\text{proc}}) conflates these two objectives, leading to catastrophic reward hacking, as we analyze in detail using Figure [2](#S3.F2 "Figure 2 ‣ 3.2 Reward Hacking with Direct PRM Integration ‣ 3 Empirical analysis of PRM and ORM ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").

![Refer to caption](/html/2603.26535/assets/x3.png)


Figure 2: Reward signal analysis. (a) Training reward: PRM reward climbs to 1.0 (perfect score gaming); ORM×\timesPRM stays moderate. (b) Response length: PRM generates increasingly verbose responses; ORM×\timesPRM shows moderate length increase (up to ∼\sim1700 tokens). (c) OlympiadBench accuracy: PRM collapses after step 600; ORM×\timesPRM tracks ORM but fails to exceed it, showing that naive signal combination does not resolve signal exhaustion.

#### Three phases of collapse.

Training with PRM-only rewards reveals a characteristic three-phase trajectory, shown as the red curve in Figure [2](#S3.F2 "Figure 2 ‣ 3.2 Reward Hacking with Direct PRM Integration ‣ 3 Empirical analysis of PRM and ORM ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").

1. 1.

   Normal learning, steps 0–300. Accuracy rises to 44.0% on OlympiadBench and training reward increases naturally.
2. 2.

   Length exploitation, steps 300–600. The model discovers that verbose responses receive higher PRM scores. Response length increases sharply while accuracy stagnates and begins to decline.
3. 3.

   Collapse, steps 600–700. Accuracy drops from 29.6% to 2.4% within 100 steps while training reward saturates at 1.0.

#### Mechanism.

The collapse follows a positive feedback loop: longer, more verbose responses receive higher PRM scores from the LLM judge, which assigns higher rubric ratings to responses that appear thorough and detailed. GRPO’s group normalization then assigns high advantage to these inflated scores, reinforcing the length-exploitation strategy.

We also provide a case study for reward hacking happened during training in Appendix [B.6](#A2.SS6 "B.6 Case Study: PRM Reward Hacking ‣ Appendix B Additional Results ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization") that confirms the post-collapse responses drift to memorized high-scoring filler content unrelated to the stated question.

#### Naive combination.

The multiplicative reward r=rout×rprocr=r^{\text{out}}\times r^{\text{proc}} gates incorrect responses via rout=0r^{\text{out}}=0, avoiding PRM’s collapse as shown by the purple curve in Figure [2](#S3.F2 "Figure 2 ‣ 3.2 Reward Hacking with Direct PRM Integration ‣ 3 Empirical analysis of PRM and ORM ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"). However, ORM×\timesPRM tracks ORM closely throughout training, peaking at 46.7% vs. ORM’s 46.3% on OlympiadBench. Because the combined reward still passes through a single GRPO normalization, all-correct groups with similar PRM scores continue to yield near-zero advantage, and the process signal fails to provide meaningful differentiation beyond ORM.

#### Implications.

ORM is stable but lacks process-level signal; PRM is information-rich but unstable. Multiplicative gating avoids PRM’s collapse yet fails to break through ORM’s performance exhaustion ceiling. This motivates us to combine the two at the *advantage* level rather than the *reward* level, enabling the model to simultaneously optimize for both answer correctness and reasoning quality, as we describe in §[4](#S4 "4 Method: PAPO ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").

## 4 Method: PAPO

We propose Process-Aware Policy Optimization (PAPO), which integrates rubric-based process evaluation into GRPO through *decoupled advantage normalization*. The key idea is to construct the advantage from independently normalized outcome and process components, preventing the reward hacking that arises from direct PRM integration while preserving fine-grained quality signals. Figure [3](#S4.F3 "Figure 3 ‣ 4 Method: PAPO ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization") provides an overview of the framework.

![Refer to caption](/html/2603.26535/assets/x4.png)


Figure 3: Overview of PAPO. Given a prompt, the policy generates GG responses. Each response is evaluated by two reward signals: an outcome reward (ORM, binary correctness) and a process reward (PRM, rubric-based quality, only for correct responses). The advantage is computed through decoupled normalization: AoutA\_{\text{out}} is normalized over all responses via standard GRPO, while AprocA\_{\text{proc}} is normalized exclusively among correct responses (*correct-subset normalization*). The combined advantage Atotal=Aout+AprocA\_{\text{total}}=A\_{\text{out}}+A\_{\text{proc}} provides both correctness direction and quality differentiation.

### 4.1 Dual Reward Signals

PAPO uses the ORM and rubric-based PRM described in §[2.2](#S2.SS2 "2.2 Reward Models for Mathematical Reasoning ‣ 2 Background ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization") as two complementary reward signals: an outcome reward riout∈{0,1}r\_{i}^{\text{out}}\in\{0,1\} for answer correctness, and a process reward riproc∈{0,0.5,1.0}r\_{i}^{\text{proc}}\in\{0,0.5,1.0\} for reasoning quality. The PRM is only invoked for correct responses; incorrect responses are assigned riproc=0r\_{i}^{\text{proc}}=0.

### 4.2 Decoupled Advantage Normalization

Given a group of GG responses with rewards {(riout,riproc)}i=1G\{(r\_{i}^{\text{out}},r\_{i}^{\text{proc}})\}\_{i=1}^{G}, PAPO computes the advantage in three steps, as illustrated in Figure [3](#S4.F3 "Figure 3 ‣ 4 Method: PAPO ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"):

#### Step 1: Outcome advantage AoutA\_{\text{out}}.

The outcome reward is normalized using standard GRPO group normalization:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Aout,i=riout−μoutmax⁡(σout,ϵ)A\_{\text{out},i}=\frac{r\_{i}^{\text{out}}-\mu^{\text{out}}}{\max(\sigma^{\text{out}},\epsilon)} |  | (3) |

where μout\mu^{\text{out}} and σout\sigma^{\text{out}} are the mean and standard deviation of {riout}i=1G\{r\_{i}^{\text{out}}\}\_{i=1}^{G}, and ϵ\epsilon is a small constant for numerical stability. This component determines the “correct vs. incorrect” gradient direction, identical to standard GRPO.

#### Step 2: Process advantage AprocA\_{\text{proc}} (correct-subset normalization).

The process reward is normalized *only among correct responses*:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Aproc,i={riproc−μCprocmax⁡(σCproc,ϵ)if ​riout=1,|C|≥20otherwiseA\_{\text{proc},i}\!=\!\begin{cases}\displaystyle\frac{r\_{i}^{\text{proc}}-\mu\_{C}^{\text{proc}}}{\max(\sigma\_{C}^{\text{proc}},\epsilon)}&\!\text{if }r\_{i}^{\text{out}}\!=\!1,\;|C|\!\geq\!2\\[6.0pt] 0&\!\text{otherwise}\end{cases} |  | (4) |

where C={j:rjout=1}C=\{j:r\_{j}^{\text{out}}=1\} is the set of correct responses, and μCproc,σCproc\mu\_{C}^{\text{proc}},\sigma\_{C}^{\text{proc}} are computed over {rjproc}j∈C\{r\_{j}^{\text{proc}}\}\_{j\in C}.

#### Step 3: Combined advantage.

The total advantage is the sum of the two components:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Atotal,i=Aout,i+Aproc,iA\_{\text{total},i}=A\_{\text{out},i}+A\_{\text{proc},i} |  | (5) |

Since both AoutA\_{\text{out}} and AprocA\_{\text{proc}} are independently normalized to zero mean and unit variance, they contribute on equal footing without requiring additional weighting.

### 4.3 Design Rationale

The decoupled normalization addresses both failure modes identified in §[3](#S3 "3 Empirical analysis of PRM and ORM ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"):

#### Quality differentiation without reward hacking.

Among correct responses, those with rigorous reasoning receive positive AprocA\_{\text{proc}} and are reinforced, while those with sloppy or lucky reasoning receive negative AprocA\_{\text{proc}} and are penalized. This quality signal is entirely absent in standard GRPO with binary ORM. Meanwhile, restricting normalization to the correct subset prevents incorrect responses from exploiting high PRM scores to gain positive AprocA\_{\text{proc}}, keeping the process signal decoupled from the outcome signal. We ablate this choice against full-group normalization in §[6](#S6 "6 Ablation Studies ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").

#### Resolving signal exhaustion.

When all GG responses are correct, Aout=0A\_{\text{out}}=0 for the entire group and standard GRPO produces no gradient. AprocA\_{\text{proc}} remains active in this case, differentiating responses by reasoning quality and providing non-zero gradients that sustain learning. As shown in Figure [4](#S5.F4 "Figure 4 ‣ 5.3 Advantage Signal Analysis ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization")(a), this reduces the zero-advantage ratio from 69% under ORM to 44% under PAPO, maintaining a denser training signal throughout optimization. When fewer than two responses are correct, i.e., |C|<2|C|<2, AprocA\_{\text{proc}} defaults to zero, gracefully reducing to standard GRPO.

## 5 Experiments

### 5.1 Experimental Setup

#### Base models.

We conduct experiments on Qwen2.5-3B/7B/14B-Base (Yang et al., [2025b](#bib.bib1 "Qwen2.5 technical report")) and Qwen3-4B-Base (Yang et al., [2025a](#bib.bib2 "Qwen3 technical report")). All models are trained from pretrained base checkpoints to isolate the effect of our method from confounding factors introduced by prior post-training.

#### Training.

We train using GRPO and DAPO with the verl (Sheng et al., [2024](#bib.bib23 "HybridFlow: a flexible and efficient rlhf framework")) and ROLL (Wang et al., [2025](#bib.bib24 "Reinforcement learning optimization for large-scale learning: an efficient and user-friendly scaling library")) frameworks for 8 epochs. For Qwen3-4B, we additionally evaluate PAPO on top of DAPO (Yu et al., [2025a](#bib.bib29 "DAPO: an open-source llm reinforcement learning system at scale")), which extends GRPO with dynamic sampling and decoupled clipping, to test compatibility with alternative optimization algorithms. The process reward model is GPT-OSS-20B with a three-tier scoring rubric of 0, 0.5, and 1.0 that evaluates reasoning quality independently of answer correctness, as described in §[2.2](#S2.SS2 "2.2 Reward Models for Mathematical Reasoning ‣ 2 Background ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"). Full hyperparameters are listed in Appendix [A](#A1 "Appendix A Implementation Details ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").

#### Data.

For training, we sample 20k mathematics problems from NuminaMath-1.5-RL-Verifiable (LI et al., [2024](#bib.bib18 "NuminaMath"); nlile, [2025](#bib.bib17 "NuminaMath-1.5-rl-verifiable")), a 131k-problem subset of the 896k-problem NuminaMath-1.5 dataset filtered to retain only problems with automatically verifiable answers. We apply stratified sampling across five difficulty tiers with 4k problems each, detailed in Appendix [A.2](#A1.SS2 "A.2 Training Data ‣ Appendix A Implementation Details ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"). For evaluation, we use six benchmarks: OlympiadBench (He et al., [2024](#bib.bib22 "OlympiadBench: a challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems")) with 674 problems as the primary metric, MATH-500 (Lightman et al., [2023](#bib.bib12 "Let’s verify step by step")), AIME 2024/2025(Balunović et al., [2025](#bib.bib20 "MathArena: evaluating llms on uncontaminated math competitions")), GPQA-Diamond (Rein et al., [2023](#bib.bib26 "GPQA: a graduate-level google-proof q&a benchmark")), and HumanEval (Chen et al., [2021](#bib.bib25 "Evaluating large language models trained on code")). All results report avg@4.

### 5.2 Main Results

|  |  | Competition Mathematics | | | Standard | STEM | Code | Average | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | Method | Olympiad | AIME’24 | AIME’25 | MATH-500 | GPQA | HumanEval | Math | All |
| Qwen2.5-3B | Base | 22.4 | 2.5 | 0.8 | 54.4 | 27.6 | 35.3 | 20.0 | 23.8 |
| ORM(GRPO) | 35.3 | 8.3 | 4.2 | 69.0 | 38.8 | 39.3 | 29.2 | 32.5 |
| PAPO | 38.6 | 10.0 | 3.3 | 73.2 | 36.3 | 52.5 | 31.3 | 35.7 |
| Δ\Delta | +3.3 | +1.7 | -0.9 | +4.2 | -2.5 | +13.2 | +2.1 | +3.2 |
| Qwen2.5-7B | Base | 32.8 | 5.0 | 7.5 | 67.3 | 30.1 | 54.6 | 28.2 | 32.9 |
| ORM(GRPO) | 46.3 | 10.8 | 10.8 | 80.2 | 40.7 | 61.6 | 37.0 | 41.7 |
| PAPO | 51.3 | 15.8 | 13.1 | 82.3 | 42.4 | 63.9 | 40.6 | 44.8 |
| Δ\Delta | +5.0 | +5.0 | +2.3 | +2.1 | +1.7 | +2.3 | +3.6 | +3.1 |
| Qwen2.5-14B | Base | 35.5 | 10.8 | 8.3 | 68.5 | 34.4 | 56.9 | 30.8 | 35.7 |
| ORM(GRPO) | 54.3 | 19.0 | 13.8 | 82.3 | 47.0 | 66.0 | 42.4 | 47.1 |
| PAPO | 59.8 | 24.3 | 17.8 | 87.4 | 55.0 | 70.0 | 47.3 | 52.4 |
| Δ\Delta | +5.5 | +5.3 | +4.0 | +5.1 | +8.0 | +4.0 | +4.9 | +5.3 |
| Qwen3-4B-Base | Base | 36.8 | 11.2 | 5.0 | 70.9 | 34.9 | 49.1 | 31.0 | 34.6 |
| ORM(DAPO) | 55.0 | 30.8 | 26.0 | 82.0 | 39.0 | 53.8 | 48.5 | 47.8 |
| PAPO | 61.1 | 34.5 | 30.7 | 84.6 | 43.0 | 63.9 | 52.7 | 53.0 |
| Δ\Delta | +6.1 | +3.7 | +4.7 | +2.6 | +4.0 | +10.1 | +4.2 | +5.2 |

Table 1: Accuracy (avg@4, %) across six benchmarks and four model configurations. PAPO consistently improves over both GRPO and DAPO baselines: Qwen2.5-3B (+3.2), 7B (+3.1), 14B (+5.3), and Qwen3-4B on DAPO (+5.2). For Qwen3-4B-Base, PAPO is applied on top of DAPO. Bold indicates the better method.

Table [1](#S5.T1 "Table 1 ‣ 5.2 Main Results ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization") presents results across six benchmarks and four model configurations. PAPO consistently outperforms ORM, and crucially, the advantage *widens over the course of training* rather than converging. On Qwen2.5-7B, both methods improve at comparable rates during early training, but ORM peaks at 46.3% on OlympiadBench at step 750 and declines to 43.0% by step 1090, while PAPO continues improving to 51.3%, as shown in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization")a. The resulting 8.3 point gap is still widening at training’s end.

In terms of final accuracy, PAPO improves OlympiadBench by 5.0 points on Qwen2.5-7B and by 5.5 points on Qwen2.5-14B, with the largest single-benchmark gain of 8.0 points on GPQA-Diamond at 14B. The mathematics average improvement grows from 2.1 points at 3B to 3.6 at 7B and 4.9 at 14B, suggesting that stronger models benefit more from PAPO. PRM-only training on Qwen2.5-7B collapses to near-zero accuracy due to reward hacking, as analyzed in §[3.2](#S3.SS2 "3.2 Reward Hacking with Direct PRM Integration ‣ 3 Empirical analysis of PRM and ORM ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").

Beyond mathematics, PAPO also improves HumanEval across all scales, suggesting that process-level quality differentiation can transfer to code generation.

We also evaluate PAPO on top of DAPO (Yu et al., [2025a](#bib.bib29 "DAPO: an open-source llm reinforcement learning system at scale")) using Qwen3-4B-Base. As shown in the bottom section of Table [1](#S5.T1 "Table 1 ‣ 5.2 Main Results ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"), PAPO improves over DAPO by 6.1 points on OlympiadBench (61.1% vs. 55.0%) with consistent gains across all six benchmarks, confirming that decoupled advantage normalization composes naturally with DAPO’s optimization improvements. We also provide the additional training curves in Appendix [B.1](#A2.SS1 "B.1 Cross-Scale Training Curves ‣ Appendix B Additional Results ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").

### 5.3 Advantage Signal Analysis

![Refer to caption](/html/2603.26535/assets/x5.png)


Figure 4: Signal quality comparison on Qwen2.5-7B. (a) Zero-advantage ratio: ORM’s sparsity grows to 69% while PAPO maintains 44%. (b) Advantage standard deviation, reflecting gradient signal strength. (c) Positive-advantage ratio, reflecting reinforcement density.

The widening gap observed in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization")a can be traced to signal exhaustion in ORM’s advantage computation. We analyze this on Qwen2.5-7B using the metrics in Figure [4](#S5.F4 "Figure 4 ‣ 5.3 Advantage Signal Analysis ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").

#### Signal exhaustion in ORM.

With growing model capability over training, more response groups become uniformly correct and yield zero advantage under ORM. Figure [4](#S5.F4 "Figure 4 ‣ 5.3 Advantage Signal Analysis ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization")a quantifies this trend. The zero-advantage ratio of ORM, defined as the fraction of samples producing zero gradient, rises from approximately 40% to 69% over the course of training. By the late training stage, more than two-thirds of samples provide no contribution to learning. The advantage standard deviation in panel b and the positive-advantage ratio in panel c convey a consistent pattern. The gradient signal from ORM gradually becomes sparser and weaker.

#### Signal preservation in PAPO.

PAPO holds the zero-advantage ratio at ∼44%{\sim}44\%, a 25-point reduction that translates to roughly 80% more informative samples per batch. This is because AprocA\_{\text{proc}} remains active in all-correct groups where Aout=0A\_{\text{out}}=0, differentiating responses by reasoning quality rather than leaving them with zero gradient. The fraction of groups where AprocA\_{\text{proc}} activates grows from ∼30%{\sim}30\% to 70% as the model improves, providing increasingly dense quality signal precisely as ORM exhausts.

#### Quality penalization.

Beyond filling zero-signal gaps, AprocA\_{\text{proc}} actively discourages sloppy reasoning even among correct responses. The minimum advantage within the correct group reaches ≈−1.49\approx-1.49 under PAPO compared to identically 0.0 under ORM, meaning correct responses with poor derivations receive negative advantage and are suppressed. This quality pressure is entirely absent in ORM-based GRPO. More detailed analyses of the process advantage effect and advantage composition are provided in Appendix [B.3](#A2.SS3 "B.3 Process Advantage Effect ‣ Appendix B Additional Results ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization") and [B.4](#A2.SS4 "B.4 Advantage Composition ‣ Appendix B Additional Results ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").

| Method | AIME | AIME | Olympiad | MATH |
| --- | --- | --- | --- | --- |
|  | 2024 | 2025 | Bench | -500 |
| ORM | 10.8 | 10.8 | 46.3 | 80.2 |
| Mult (rout×rprocr^{\text{out}}\!\times\!r^{\text{proc}}) | 12.1 | 10.2 | 46.7 | 80.2 |
| Fullnorm | 12.5 | 10.4 | 49.6 | 81.2 |
| PAPO | 15.8 | 13.1 | 51.3 | 82.3 |

Table 2: Ablation results (%, avg@4, Qwen2.5-7B). All methods report accuracy.

## 6 Ablation Studies

We validate the key design choices of PAPO through two ablation experiments on Qwen2.5-7B. Table [2](#S5.T2 "Table 2 ‣ Quality penalization. ‣ 5.3 Advantage Signal Analysis ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization") summarizes the results; training curves are provided in Appendix [B](#A2 "Appendix B Additional Results ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").

#### Correct-subset vs. full normalization.

The Fullnorm variant normalizes AprocA\_{\text{proc}} over all GG responses including incorrect ones. It performs slightly below PAPO across all benchmarks. The gap arises because including incorrect responses (rproc=0r^{\text{proc}}=0) in the normalization causes AprocA\_{\text{proc}} to partially recapitulate the correct-vs-incorrect distinction already captured by AoutA\_{\text{out}}, diluting the fine-grained quality signal.

#### Multiplicative reward baseline.

The Mult variant combines rewards via ri=riout×riprocr\_{i}=r\_{i}^{\text{out}}\times r\_{i}^{\text{proc}} with standard GRPO normalization. Mult performs comparably to ORM and falls well below PAPO. In a single normalization pass, the outcome difference dominates in mixed-correctness groups, effectively suppressing the process quality information.

Both ablations confirm that PAPO’s gains require both *decoupled normalization*, validated by Mult’s inferior performance, and *correct-subset normalization*, validated by Fullnorm’s inferior performance, working in concert.

## 7 Qualitative Analysis

We perform qualitative analysis on reasoning traces from the ORM baseline and PAPO to understand how process reward shapes model behavior.

### 7.1 Process Supervision Corrects Reasoning Errors

Without supervision on intermediate reasoning steps, ORM-trained models can develop sloppy reasoning habits that lead to surprisingly elementary errors. We illustrate this with OlympiadBench #507, which asks to compute the number of ordered pairs (x,y)(x,y) of positive integers satisfying x2−8​x+y2+4​y=5x^{2}-8x+y^{2}+4y=5.

Both models correctly complete the square to obtain (x−4)2+(y+2)2=25(x-4)^{2}+(y+2)^{2}=25. The ORM model then *incorrectly excludes* the valid solution (x,y)=(1,2)(x,y)=(1,2) with the fabricated justification that “x=1x=1 is not a positive integer in the context of the circle’s radius.” This is a nonsensical claim, since x=1x=1 is plainly a positive integer, and this elementary error directly causes a wrong final answer.

The PAPO model, by contrast, carefully verifies each candidate against the positivity constraint and correctly identifies all four solutions. Process supervision penalizes flawed intermediate steps regardless of the final answer, discouraging the kind of careless reasoning that produces such errors in the first place.

### 7.2 Process Reward Cultivates Self-Verification

Beyond correcting errors, we find that process reward training cultivates a *self-verification habit*, where the model spontaneously substitutes the derived answer back into the original problem as a final check. This behavior emerges naturally because responses that include verification receive higher process scores, reinforcing the habit over time.

For instance, on OlympiadBench #129, which asks to find xx such that log2⁡(log2⁡(2​x−2))=2\log\_{2}(\log\_{2}(2x-2))=2, both models correctly derive x=9x=9 through successive exponentiation. The ORM model stops here. The PAPO model adds an explicit check, confirming that log2⁡(log2⁡(16))=log2⁡(4)=2\log\_{2}(\log\_{2}(16))=\log\_{2}(4)=2.

This is not an isolated case. Across OlympiadBench problems where both models answer correctly, PAPO responses contain explicit verification steps in 39.7% of cases, compared to 22.7% for ORM, a 1.7×\times higher rate. Process reward incentivizes the model to develop a disciplined reasoning style that makes its outputs more robust, even on problems it can already solve.

## 8 Related Work

Recent advancements in the post-training of large language models have been significantly driven by reinforcement learning with verifiable rewards (RLVR). This section systematically reviews several key areas that are highly relevant to and interconnected with the present study.

### 8.1 Policy Optimization in RLVR.

Following DeepSeek-R1’s validation of critic-free reinforcement learning (Guo et al., [2025](#bib.bib6 "DeepSeek-r1 incentivizes reasoning in llms through reinforcement learning")), Group Relative Policy Optimization (GRPO)(Shao et al., [2024](#bib.bib5 "DeepSeekMath: pushing the limits of mathematical reasoning in open language models")) has emerged as a compute-efficient alternative to PPO (Schulman et al., [2017](#bib.bib8 "Proximal policy optimization algorithms")). However, its instability under long contexts and sparse rewards has spurred various enhancements. To mitigate policy staleness and granularity mismatch, CISPO clips importance sampling weights (MiniMax, [2025](#bib.bib27 "MiniMax-m1: scaling test-time compute efficiently with lightning attention")) and GSPO employs sequence-level ratio calculations (Zheng et al., [2025](#bib.bib28 "Group sequence policy optimization")). For long-horizon and agentic tasks, DAPO introduces dynamic sampling and decoupled clipping (Yu et al., [2025a](#bib.bib29 "DAPO: an open-source llm reinforcement learning system at scale")) and SAPO utilizes a temperature-controlled smooth gating function to form a continuous trust region (Gao et al., [2025](#bib.bib30 "Soft adaptive policy optimization")). Orthogonally, our work addresses GRPO’s fundamental limitation of reward signal exhaustion by decoupling advantage normalization at the objective level. Since PAPO modifies the advantage computation rather than the optimization procedure, it can be composed with these enhancements, as we demonstrate with DAPO in §[5.2](#S5.SS2 "5.2 Main Results ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").

### 8.2 Rubric-Based Training.

Traditional outcome reward models frequently suffer from reward hacking, exploiting miracle steps to achieve correct final answers via flawed logic (Yuan et al., [2025](#bib.bib16 "Curing miracle steps in llm mathematical reasoning with rubric rewards")). To enforce rigorous logical constraints without prohibitive step-level human annotation, rubric-based process supervision has gained traction. Generative rubric reward models explicitly penalize disconnected derivations, drastically curtailing lucky guesses (Liang et al., [2025](#bib.bib36 "Generative reward modeling via synthetic criteria preference learning")). In pursuit of self-verification, DeepSeekMath-V2 constructed a specialized LLM verifier to reversely drive the generator to patch reasoning loopholes (Shao et al., [2025](#bib.bib7 "DeepSeekMath-v2: towards self-verifiable mathematical reasoning")). Furthermore, in complex open-domain tasks, frameworks like Agent-RRM propose multi-faceted evaluation systems enforcing explicit reasoning traces and actionable critiques (Fan et al., [2026](#bib.bib31 "Exploring reasoning reward model for agents")). These studies establish that effective process supervision necessitates multi-dimensional criteria and explicit negative penalties. Building upon these insights, our work leverages a rubric-based process evaluation but introduces a correct-subset normalization mechanism to prevent verbosity-driven reward hacking.

### 8.3 Generative Reward Models.

To overcome the expressivity limitations of traditional discriminative models, reward modeling is shifting toward a generative paradigm (Mahan et al., [2024](#bib.bib32 "Generative reward models")). By autoregressively generating deep comparative analyses prior to preference scoring, generative reward models significantly enhance interpretability and accuracy. This paradigm excels in abstract domains like formal proofs, as demonstrated by Proof-RM (Yang et al., [2026](#bib.bib11 "Proof-rm: a scalable and generalizable reward model for math proof")), and enables verifier-free reinforcement learning extrapolation in broad domains, as seen in RLPR (Yu et al., [2025b](#bib.bib33 "RLPR: extrapolating rlvr to general domains without verifiers")). However, generative evaluators remain susceptible to intrinsic biases such as imperfect sensitivity and specificity. To ensure statistically rigorous training, recent frameworks incorporate bias-correction estimators and adaptive calibration to construct statistically sound confidence intervals (Lee et al., [2026](#bib.bib34 "How to correctly report llm-as-a-judge evaluations")). Aligning with this generative shift, our methodology utilizes a generative judge to assess reasoning quality, yet uniquely confines its influence within the advantage space of correct solutions to immunize the policy against structural biases.

## 9 Conclusion

In this paper, we identify the reasoning limitations of ORM-based GRPO on mathematical reasoning tasks. While a simple binary ORM reward is effective, its lack of supervision over reasoning quality causes all reasoning tokens to share the same credit. On the other hand, a naive integration of PRM fails to avoid reward hacking.

To address this challenge, we propose PAPO, which incorporates a rubric-based PRM into GRPO through decoupled advantage normalization. By composing the advantage from independently normalized outcome and process components, with the process advantage normalized only over correct responses, PAPO prevents reward hacking while providing supervision over reasoning quality.

Experiments across four model configurations (3B–14B) and six benchmarks show that PAPO consistently outperforms both GRPO and DAPO baselines, with gains growing as model scale increases. In future work, we aim to explore adaptive weighting between outcome and process signals and extension to broader domains such as scientific reasoning.

## Limitations

Our experiments are conducted exclusively on Qwen-family models and have not been verified on other architectures such as Llama and Gemma. Additionally, we use only GPT-OSS-20B as the rubric-based PRM; the effect of different judge models remains unexplored.

## Acknowledgements

This work was supported by Shanghai Artificial Intelligence Laboratory.

## References

* M. Balunović, J. Dekoninck, I. Petrov, N. Jovanović, and M. Vechev (2025)
  MathArena: evaluating llms on uncontaminated math competitions.
   SRI Lab, ETH Zurich.
  External Links: [Link](https://matharena.ai/)
  Cited by: [§1](#S1.p5.1 "1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§5.1](#S5.SS1.SSS0.Px3.p1.1 "Data. ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, et al. (2021)
  Evaluating large language models trained on code.
  External Links: 2107.03374
  Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1 "Data. ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* K. Fan, K. Feng, M. Zhang, T. Peng, Z. Li, Y. Jiang, S. Chen, P. Pei, X. Cai, and X. Yue (2026)
  Exploring reasoning reward model for agents.
  arXiv preprint arXiv:2601.22154.
  Cited by: [§8.2](#S8.SS2.p1.1 "8.2 Rubric-Based Training. ‣ 8 Related Work ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* C. Gao, C. Zheng, X. Chen, K. Dang, S. Liu, B. Yu, A. Yang, S. Bai, J. Zhou, and J. Lin (2025)
  Soft adaptive policy optimization.
  External Links: 2511.20347,
  [Link](https://arxiv.org/abs/2511.20347)
  Cited by: [§8.1](#S8.SS1.p1.1 "8.1 Policy Optimization in RLVR. ‣ 8 Related Work ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, et al. (2025)
  DeepSeek-r1 incentivizes reasoning in llms through reinforcement learning.
  Nature 645,  pp. 633 – 638.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:275789950)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§8.1](#S8.SS1.p1.1 "8.1 Policy Optimization in RLVR. ‣ 8 Related Work ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* C. He, R. Luo, Y. Bai, S. Hu, Z. L. Thai, J. Shen, J. Hu, X. Han, Y. Huang, Y. Zhang, J. Liu, L. Qi, Z. Liu, and M. Sun (2024)
  OlympiadBench: a challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems.
  In Annual Meeting of the Association for Computational Linguistics,
  External Links: [Link](https://api.semanticscholar.org/CorpusID:267770504)
  Cited by: [§1](#S1.p5.1 "1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§5.1](#S5.SS1.SSS0.Px3.p1.1 "Data. ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. X. Song, and J. Steinhardt (2021)
  Measuring mathematical problem solving with the math dataset.
  ArXiv abs/2103.03874.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:232134851)
  Cited by: [§2.2](#S2.SS2.SSS0.Px1.p1.3 "Outcome Reward Models (ORM). ‣ 2.2 Reward Models for Mathematical Reasoning ‣ 2 Background ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* A. Kamath, J. Ferret, S. Pathak, N. Vieillard, R. Merhej, et al. (2025)
  Gemma 3 technical report.
  External Links: 2503.19786,
  [Link](https://arxiv.org/abs/2503.19786)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* C. Lee, T. Zeng, J. Jeong, J. Sohn, and K. Lee (2026)
  How to correctly report llm-as-a-judge evaluations.
  External Links: 2511.21140,
  [Link](https://arxiv.org/abs/2511.21140)
  Cited by: [§8.3](#S8.SS3.p1.1 "8.3 Generative Reward Models. ‣ 8 Related Work ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* J. LI, E. Beeching, L. Tunstall, B. Lipkin, R. Soletskyi, S. C. Huang, K. Rasul, L. Yu, A. Jiang, Z. Shen, Z. Qin, B. Dong, L. Zhou, Y. Fleureau, G. Lample, and S. Polu (2024)
  NuminaMath.
   Numina.
  Note: <https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf>
  Cited by: [§A.2](#A1.SS2.p1.1 "A.2 Training Data ‣ Appendix A Implementation Details ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§5.1](#S5.SS1.SSS0.Px3.p1.1 "Data. ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* X. Liang, H. Zhang, J. Li, K. Chen, Q. Zhu, and M. Zhang (2025)
  Generative reward modeling via synthetic criteria preference learning.
  In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar (Eds.),
  Vienna, Austria,  pp. 26755–26769.
  External Links: [Link](https://aclanthology.org/2025.acl-long.1297/),
  [Document](https://dx.doi.org/10.18653/v1/2025.acl-long.1297),
  ISBN 979-8-89176-251-0
  Cited by: [§8.2](#S8.SS2.p1.1 "8.2 Rubric-Based Training. ‣ 8 Related Work ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe (2023)
  Let’s verify step by step.
  External Links: 2305.20050,
  [Link](https://arxiv.org/abs/2305.20050)
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§2.2](#S2.SS2.SSS0.Px2.p1.1 "Process Reward Models (PRM). ‣ 2.2 Reward Models for Mathematical Reasoning ‣ 2 Background ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§5.1](#S5.SS1.SSS0.Px3.p1.1 "Data. ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* D. Mahan, D. V. Phung, R. Rafailov, C. Blagden, N. Lile, L. Castricato, J. Fränken, C. Finn, and A. Albalak (2024)
  Generative reward models.
  External Links: 2410.12832,
  [Link](https://arxiv.org/abs/2410.12832)
  Cited by: [§8.3](#S8.SS3.p1.1 "8.3 Generative Reward Models. ‣ 8 Related Work ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* MiniMax (2025)
  MiniMax-m1: scaling test-time compute efficiently with lightning attention.
  External Links: 2506.13585,
  [Link](https://arxiv.org/abs/2506.13585)
  Cited by: [§8.1](#S8.SS1.p1.1 "8.1 Policy Optimization in RLVR. ‣ 8 Related Work ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* nlile (2025)
  NuminaMath-1.5-rl-verifiable.
   Hugging Face.
  Note: <https://huggingface.co/datasets/nlile/NuminaMath-1.5-RL-Verifiable>
  Cited by: [§A.2](#A1.SS2.p1.1 "A.2 Training Data ‣ Appendix A Implementation Details ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§5.1](#S5.SS1.SSS0.Px3.p1.1 "Data. ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang, J. Dirani, J. Michael, and S. R. Bowman (2023)
  GPQA: a graduate-level google-proof q&a benchmark.
  ArXiv abs/2311.12022.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:265295009)
  Cited by: [§5.1](#S5.SS1.SSS0.Px3.p1.1 "Data. ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov (2017)
  Proximal policy optimization algorithms.
  ArXiv abs/1707.06347.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:28695052)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§2.1](#S2.SS1.p1.4 "2.1 Group Relative Policy Optimization ‣ 2 Background ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§8.1](#S8.SS1.p1.1 "8.1 Policy Optimization in RLVR. ‣ 8 Related Work ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* Z. Shao, Y. Luo, C. Lu, Z. Z. Ren, J. Hu, T. Ye, Z. Gou, S. Ma, and X. Zhang (2025)
  DeepSeekMath-v2: towards self-verifiable mathematical reasoning.
  External Links: 2511.22570,
  [Link](https://arxiv.org/abs/2511.22570)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§1](#S1.p3.1 "1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§8.2](#S8.SS2.p1.1 "8.2 Rubric-Based Training. ‣ 8 Related Work ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, M. Zhang, Y. Li, Y. Wu, and D. Guo (2024)
  DeepSeekMath: pushing the limits of mathematical reasoning in open language models.
  arXiv preprint arXiv:2402.03300.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§2.1](#S2.SS1.p1.4 "2.1 Group Relative Policy Optimization ‣ 2 Background ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§8.1](#S8.SS1.p1.1 "8.1 Policy Optimization in RLVR. ‣ 8 Related Work ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* G. Sheng, C. Cao, Z. Li, X. Gu, et al. (2024)
  HybridFlow: a flexible and efficient rlhf framework.
  arXiv preprint arXiv:2409.19256.
  Cited by: [§A.1](#A1.SS1.p1.1 "A.1 Training Configuration ‣ Appendix A Implementation Details ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§5.1](#S5.SS1.SSS0.Px2.p1.1 "Training. ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* L. Sheng, W. Ma, R. Hong, X. Wang, A. Zhang, and T. Chua (2026)
  Reinforcing chain-of-thought reasoning with self-evolving rubrics.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:285470628)
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§2.2](#S2.SS2.SSS0.Px2.p1.1 "Process Reward Models (PRM). ‣ 2.2 Reward Models for Mathematical Reasoning ‣ 2 Background ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* J. M. V. Skalse, N. H. R. Howe, D. Krasheninnikov, and D. Krueger (2022)
  Defining and characterizing reward hacking.
  ArXiv abs/2209.13085.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:252545256)
  Cited by: [§1](#S1.p3.1.1 "1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* H. Touvron, T. Lavril, G. Izacard, X. Martinet, M. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample (2023)
  LLaMA: open and efficient foundation language models.
  External Links: 2302.13971,
  [Link](https://arxiv.org/abs/2302.13971)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* P. Wang, L. Li, Z. Shao, R. X. Xu, D. Dai, Y. Li, D. Chen, Y. Wu, and Z. Sui (2024)
  Math-shepherd: verify and reinforce llms step-by-step without human annotations.
  External Links: 2312.08935,
  [Link](https://arxiv.org/abs/2312.08935)
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§2.2](#S2.SS2.SSS0.Px2.p1.1 "Process Reward Models (PRM). ‣ 2.2 Reward Models for Mathematical Reasoning ‣ 2 Background ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* W. Wang, S. Xiong, G. Chen, W. Gao, S. Guo, Y. He, J. Huang, J. Liu, Z. Li, X. Li, et al. (2025)
  Reinforcement learning optimization for large-scale learning: an efficient and user-friendly scaling library.
  arXiv preprint arXiv:2506.06122.
  Cited by: [§A.1](#A1.SS1.p1.1 "A.1 Training Configuration ‣ Appendix A Implementation Details ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§5.1](#S5.SS1.SSS0.Px2.p1.1 "Training. ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, et al. (2025a)
  Qwen3 technical report.
  External Links: 2505.09388,
  [Link](https://arxiv.org/abs/2505.09388)
  Cited by: [§5.1](#S5.SS1.SSS0.Px1.p1.1 "Base models. ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, et al. (2025b)
  Qwen2.5 technical report.
  External Links: 2412.15115,
  [Link](https://arxiv.org/abs/2412.15115)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§5.1](#S5.SS1.SSS0.Px1.p1.1 "Base models. ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* H. Yang, Z. Wang, S. Kang, S. Yang, W. Yu, X. Niu, Y. Sun, Y. Hu, Z. Lin, and M. Zhang (2026)
  Proof-rm: a scalable and generalizable reward model for math proof.
  External Links: 2602.02377,
  [Link](https://arxiv.org/abs/2602.02377)
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§2.2](#S2.SS2.SSS0.Px2.p1.1 "Process Reward Models (PRM). ‣ 2.2 Reward Models for Mathematical Reasoning ‣ 2 Background ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§8.3](#S8.SS3.p1.1 "8.3 Generative Reward Models. ‣ 8 Related Work ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, et al. (2025a)
  DAPO: an open-source llm reinforcement learning system at scale.
  External Links: 2503.14476,
  [Link](https://arxiv.org/abs/2503.14476)
  Cited by: [§A.1](#A1.SS1.p2.1 "A.1 Training Configuration ‣ Appendix A Implementation Details ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§5.1](#S5.SS1.SSS0.Px2.p1.1 "Training. ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§5.2](#S5.SS2.p4.1 "5.2 Main Results ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§8.1](#S8.SS1.p1.1 "8.1 Policy Optimization in RLVR. ‣ 8 Related Work ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* T. Yu, B. Ji, S. Wang, S. Yao, Z. Wang, G. Cui, L. Yuan, N. Ding, Y. Yao, Z. Liu, M. Sun, and T. Chua (2025b)
  RLPR: extrapolating rlvr to general domains without verifiers.
  External Links: 2506.18254,
  [Link](https://arxiv.org/abs/2506.18254)
  Cited by: [§8.3](#S8.SS3.p1.1 "8.3 Generative Reward Models. ‣ 8 Related Work ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* Y. Yuan, Q. Mang, J. Chen, H. Wan, X. Liu, J. Xu, J. Huang, W. Wang, W. Jiao, and P. He (2025)
  Curing miracle steps in llm mathematical reasoning with rubric rewards.
  ArXiv abs/2510.07774.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:281951384)
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§2.2](#S2.SS2.SSS0.Px2.p1.1 "Process Reward Models (PRM). ‣ 2.2 Reward Models for Mathematical Reasoning ‣ 2 Background ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§8.2](#S8.SS2.p1.1 "8.2 Rubric-Based Training. ‣ 8 Related Work ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* C. Zheng, S. Liu, M. Li, X. Chen, B. Yu, C. Gao, K. Dang, Y. Liu, R. Men, A. Yang, J. Zhou, and J. Lin (2025)
  Group sequence policy optimization.
  arXiv preprint arXiv:2507.18071.
  Cited by: [§8.1](#S8.SS1.p1.1 "8.1 Policy Optimization in RLVR. ‣ 8 Related Work ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").
* L. Zheng, W. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica (2023)
  Judging llm-as-a-judge with mt-bench and chatbot arena.
  External Links: 2306.05685,
  [Link](https://arxiv.org/abs/2306.05685)
  Cited by: [§1](#S1.p3.1 "1 Introduction ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"),
  [§2.2](#S2.SS2.SSS0.Px2.p1.1 "Process Reward Models (PRM). ‣ 2.2 Reward Models for Mathematical Reasoning ‣ 2 Background ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").

## Appendix A Implementation Details

### A.1 Training Configuration

We use the verl (Sheng et al., [2024](#bib.bib23 "HybridFlow: a flexible and efficient rlhf framework")) and ROLL (Wang et al., [2025](#bib.bib24 "Reinforcement learning optimization for large-scale learning: an efficient and user-friendly scaling library")) frameworks with Megatron backend for distributed training on 8 GPUs. Key hyperparameters are listed in Table [3](#A1.T3 "Table 3 ‣ A.1 Training Configuration ‣ Appendix A Implementation Details ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").

|  |  |
| --- | --- |
| Hyperparameter | Value |
| Prompts per batch | 128 |
| Responses per prompt (GG) | 8 |
| Max response length | 8192 tokens |
| Sampling temperature | 1.0 |
| PPO clip range (ϵ\epsilon) | 0.2 |
| KL penalty coefficient (β\beta) | 0.0 |
| Learning rate | 1e-6 |
| Evaluation frequency | every 10 steps |
| Evaluation samples per prompt | 8 |

Table 3: Training hyperparameters.

For Qwen3-4B-Base experiments with DAPO (Yu et al., [2025a](#bib.bib29 "DAPO: an open-source llm reinforcement learning system at scale")), we additionally apply the DAPO-specific hyperparameters listed in Table [4](#A1.T4 "Table 4 ‣ A.1 Training Configuration ‣ Appendix A Implementation Details ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").

| Hyperparameter | Value |
| --- | --- |
| Max response length | 8192 tokens |
| Clip ratio low (ϵlow\epsilon\_{\text{low}}) | 0.2 |
| Clip ratio high (ϵhigh\epsilon\_{\text{high}}) | 0.28 |
| Overlong buffer length | 4096 tokens |
| Overlong penalty factor | 1.0 |
| Evaluation frequency | every 20 steps |

Table 4: Additional DAPO-specific hyperparameters for the Qwen3-4B-Base experiments. All other hyperparameters follow Table [3](#A1.T3 "Table 3 ‣ A.1 Training Configuration ‣ Appendix A Implementation Details ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").

### A.2 Training Data

We construct a 20K stratified training set from NuminaMath-1.5-RL-Verifiable (LI et al., [2024](#bib.bib18 "NuminaMath"); nlile, [2025](#bib.bib17 "NuminaMath-1.5-rl-verifiable")). We first filter out unsolvable problems (zero pass rate under a reference model), then apply stratified sampling across five difficulty tiers defined by pass rate. Table [5](#A1.T5 "Table 5 ‣ A.2 Training Data ‣ Appendix A Implementation Details ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization") summarizes the difficulty distribution.

| Difficulty | Pass Rate Range | Count |
| --- | --- | --- |
| Trivial | (0.875, 1.0](0.875,\,1.0] | 4,000 |
| Easy | (0.625, 0.875](0.625,\,0.875] | 4,000 |
| Medium | (0.375, 0.625](0.375,\,0.625] | 4,000 |
| Hard | (0.125, 0.375](0.125,\,0.375] | 4,000 |
| Very Hard | (0, 0.125](0,\,0.125] | 4,000 |
| Total |  | 20,000 |

Table 5: Difficulty distribution of the stratified training set. Pass rates are measured using a reference model with 8 attempts at temperature 1.0. Equal sampling across tiers ensures balanced difficulty coverage.

### A.3 Rubric Prompt

The rubric-based PRM uses the following prompt template, adapted from the DeepSeek Math V2 grading standard with chain-of-thought analysis. The PRM is implemented using GPT-OSS-20B served via vLLM with tensor parallelism across 2 GPUs, with a maximum context length of 8192 tokens and temperature 0.0 for deterministic scoring.

Rubric Prompt

## Instruction
Your task is to evaluate the quality of a student’s solution to a mathematical problem.
## Scoring Rubric

•

Score 1: If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1.
•

Score 0.5: If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5.
•

Score 0: If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0.
Special Rule on References: Additionally, referencing anything from any paper does not save the need to prove the reference. It’s okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1.
## Problem
{problem\_statement}
## Reference Solution
{solution}
## Student Solution
{student\_answer}
## Evaluation
Analyze the solution step by step, then provide your score.
Analysis: …
Score: …\boxed{\ldots}

## Appendix B Additional Results

### B.1 Cross-Scale Training Curves

![Refer to caption](/html/2603.26535/assets/x6.png)


Figure 5: Training curves (avg@4) across two model scales and three benchmarks. Top row: Qwen2.5-3B; bottom row: Qwen2.5-7B. PAPO consistently outperforms ORM throughout training on both scales, with gains widening in later stages as ORM’s signal exhaustion worsens. The pattern is consistent across OlympiadBench (competition math), MATH-500 (standard math), and HumanEval (code generation). Qwen2.5-14B results are reported in Table [1](#S5.T1 "Table 1 ‣ 5.2 Main Results ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").

Figure [5](#A2.F5 "Figure 5 ‣ B.1 Cross-Scale Training Curves ‣ Appendix B Additional Results ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization") shows training curves across two model scales (3B and 7B) and three benchmarks. On both Qwen2.5-3B and 7B, PAPO maintains a consistent lead over ORM throughout training, with the gap widening in later stages. Combined with the 14B results in Table [1](#S5.T1 "Table 1 ‣ 5.2 Main Results ‣ 5 Experiments ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"), this confirms that the improvements reflect sustained training dynamics rather than checkpoint-specific artifacts.

### B.2 Ablation Training Curves

![Refer to caption](/html/2603.26535/assets/x7.png)


Figure 6: Ablation: correct-subset normalization (PAPO) vs. full normalization (Fullnorm) on OlympiadBench and MATH-500. Both methods improve over ORM, but PAPO’s correct-subset design maintains a consistent advantage over Fullnorm throughout training.

![Refer to caption](/html/2603.26535/assets/x8.png)


Figure 7: Ablation: decoupled normalization (PAPO) vs. multiplicative reward (Mult) on OlympiadBench and MATH-500. Mult improves over ORM but falls substantially short of PAPO, confirming that decoupled normalization is superior to naive reward combination.

Figures [6](#A2.F6 "Figure 6 ‣ B.2 Ablation Training Curves ‣ Appendix B Additional Results ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization") and [7](#A2.F7 "Figure 7 ‣ B.2 Ablation Training Curves ‣ Appendix B Additional Results ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization") show the training dynamics of the ablation variants. Both confirm that PAPO’s design choices yield consistent advantages throughout training, not just at specific checkpoints.

### B.3 Process Advantage Effect

![Refer to caption](/html/2603.26535/assets/x9.png)


Figure 8: (a) Minimum advantage among correct responses. ORM: identically 0.0 (all correct responses treated equally). PAPO: ≈−1.49\approx-1.49 (the process advantage penalizes correct responses with poor reasoning). (b) Standard deviation of advantages among correct responses. ORM: zero (no differentiation). PAPO: non-zero (quality-based ranking within correct group).

Figure [8](#A2.F8 "Figure 8 ‣ B.3 Process Advantage Effect ‣ Appendix B Additional Results ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization") reveals how PAPO’s process advantage operates within the correct response group. In standard ORM-based GRPO, the minimum advantage among correct responses is identically 0.0—every correct response receives identical positive advantage. In PAPO, correct\_min ≈−1.49\approx-1.49, meaning correct responses with poor reasoning receive *negative* total advantage. The process advantage actively discourages solutions that arrive at the right answer through flawed reasoning. Panel (b) confirms that the process advantage creates a meaningful quality ranking among correct responses (non-zero std), enabling the model to learn which reasoning strategies produce better solutions.

### B.4 Advantage Composition

![Refer to caption](/html/2603.26535/assets/x10.png)


Figure 9: PAPO advantage composition. (a) Signal strength (std) of AoutA\_{\text{out}} and AprocA\_{\text{proc}}: the outcome advantage provides the dominant gradient signal while the process advantage contributes a complementary quality signal. (b) Process signal active ratio: the fraction of groups where AprocA\_{\text{proc}} is non-zero (≥2\geq 2 correct responses), growing from ∼\sim30% to 70% as the model improves. (c) Mean advantage by correctness: correct responses receive positive advantage while wrong responses receive negative, confirming the outcome signal functions correctly despite the added process component.

Figure [9](#A2.F9 "Figure 9 ‣ B.4 Advantage Composition ‣ Appendix B Additional Results ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization") visualizes the internal structure of PAPO’s composed advantage. Panel (a) shows that the outcome advantage provides the dominant gradient signal, while the process advantage contributes a substantial secondary signal. Panel (b) tracks the process signal active ratio—the fraction of groups where AprocA\_{\text{proc}} activates—which grows from ∼30%{\sim}30\% to 70% as the model improves, indicating that the process signal’s contribution increases precisely as ORM’s signal exhaustion worsens. Panel (c) confirms that the combined advantage correctly separates correct from incorrect responses: the addition of AprocA\_{\text{proc}} does not disrupt the fundamental separation provided by AoutA\_{\text{out}}.

### B.5 Training Reward Dynamics

The training reward (average ORM score across the batch) follows similar trajectories for ORM and PAPO, both plateauing around 55% to 62%. This indicates that PAPO does not achieve its gains by inflating outcome reward; the improvement comes from better reasoning quality among correct responses, as captured by the process advantage.

In contrast, PRM-only training shows the training reward climbing to 1.0 by step 1000, confirming complete reward gaming. The model learns to produce responses that maximize the PRM score regardless of mathematical correctness.

### B.6 Case Study: PRM Reward Hacking

To empirically illustrate the reward hacking mechanism described in §[3.2](#S3.SS2 "3.2 Reward Hacking with Direct PRM Integration ‣ 3 Empirical analysis of PRM and ORM ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"), we generate responses from three checkpoints—the base model (Qwen2.5-7B, step 0), the PRM-only model at step 1000 (post-collapse, 2.4% accuracy on OlympiadBench), and the ORM baseline at step 1000—on five math problems of varying difficulty, sampling 3 responses per model with temperature 0.7 and a generation cap of 8192 tokens. We report the three most informative problems in Table [6](#A2.T6 "Table 6 ‣ B.6 Case Study: PRM Reward Hacking ‣ Appendix B Additional Results ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"); on the remaining two easier problems (combinatorics, geometry), PRM performs comparably to baselines.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Problem | Model | Tokens | Correct | Observation |
| Competition number theory | Base | 866 | 3/3 | Focused proofs |
| ORM | 1583 | 3/3 | Longer but correct |
| PRM | 3072 | 0/3 | Drift to filler∗ |
| Algebra (proof) | Base | 1059 | 0/3 | Attempts proof |
| ORM | 1492 | 0/3 | Attempts proof |
| PRM | 3227 | 0/3 | 2 hit cap; 1 drifts |
| Number theory | Base | 484 | 2/3 | Concise, correct |
| ORM | 519 | 3/3 | Concise, correct |
| PRM | 1522 | 3/3 | 1 inflated to 3291 |
| ∗ All 3 PRM samples drift to the *identical* filler | | | | |
| problem (vector perpendicularity, answer t=2t\!=\!2). | | | | |

Table 6: Case study: response statistics on three math problems (Qwen2.5-7B). PRM reward hacking is severe on harder problems (top two), with 2–3×\times longer responses and characteristic topic drift. On easier problems, PRM remains comparable to baselines (omitted for brevity).

#### Results.

Table [6](#A2.T6 "Table 6 ‣ B.6 Case Study: PRM Reward Hacking ‣ Appendix B Additional Results ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization") summarizes the findings. On the easier number theory problem, PRM produces correct but inflated responses. On harder problems (competition number theory, proof-based algebra), the model degenerates dramatically: responses are 2–3×\times longer than ORM, and none arrive at the correct answer.

#### Qualitative analysis: topic drift to a fixed filler template.

The most striking finding is in the competition number theory problem (“Find all positive integers nn such that n2+1∣n!+1n^{2}+1\mid n!+1”; answer: n=1n=1). All three PRM step-1000 responses begin correctly—setting up the divisibility condition, checking small cases—but after ∼\sim1500–2000 tokens of genuine work, each response *drifts to the identical unrelated problem*: computing the dot product of vectors m→=(t,1)\overrightarrow{m}=(t,1) and n→=(1,−2)\overrightarrow{n}=(1,-2), solving t−2=0t-2=0, and concluding 2\boxed{2}. This vector problem is entirely unrelated to the stated question.

The convergence of all three independent samples to the same filler content reveals a learned exploitation strategy: when the model cannot solve a hard problem, it transitions to a *memorized high-scoring template*—a short, well-structured solution to a simple problem that the LLM judge would rate as “all steps executed properly” (rubric score 1.0). The same pattern appears in the algebra proof and the inflated number theory sample, where responses drift to digit-sum sequences or other unrelated derivations.

In contrast, ORM produces focused solutions (1494–1646 tokens) that arrive at the correct answer n=1n=1 on all three samples. The ORM model may produce longer proofs than the base model (1583 vs. 866 avg tokens), but the additional length consists of genuine mathematical reasoning (additional case analysis), not filler content.

#### Implications.

The case study reveals that PRM reward hacking is not random degeneration but a *structured exploitation strategy*: (1) attempt the stated problem; (2) when stuck, seamlessly transition to memorized high-scoring content; (3) produce a confident final answer to the wrong problem. The responses remain well-formatted and superficially mathematical throughout—precisely the qualities an LLM-as-Judge PRM rates favorably. This confirms the positive feedback loop of §[3.2](#S3.SS2 "3.2 Reward Hacking with Direct PRM Integration ‣ 3 Empirical analysis of PRM and ORM ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization") and motivates the decoupled design of PAPO, where the binary ORM anchors correctness assessment independently of the PRM signal.

## Appendix C Case Study: Full Response Comparison

We present the complete model responses for the two qualitative examples discussed in §[7](#S7 "7 Qualitative Analysis ‣ PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization").

### C.1 OlympiadBench #507 (Reasoning Error)

Problem. Compute the number of ordered pairs (x,y)(x,y) of positive integers satisfying x2−8​x+y2+4​y=5x^{2}-8x+y^{2}+4y=5. Ground truth: 4

ORM Response (incorrect answer: 3)

To find the number of ordered pairs (x,y)(x,y) of positive integers satisfying the equation x2−8​x+y2+4​y=5x^{2}-8x+y^{2}+4y=5, we will complete the square for both xx and yy.
First, complete the square for xx:
x2−8​x=(x−4)2−16.x^{2}-8x=(x-4)^{2}-16.
Next, complete the square for yy:
y2+4​y=(y+2)2−4.y^{2}+4y=(y+2)^{2}-4.
Substitute these into the original equation:
(x−4)2−16+(y+2)2−4=5,(x-4)^{2}-16+(y+2)^{2}-4=5,
(x−4)2+(y+2)2=25.(x-4)^{2}+(y+2)^{2}=25.
This is the equation of a circle with center (4,−2)(4,-2) and radius 5. We need to find the number of positive integer solutions (x,y)(x,y) that lie on this circle.
The possible values for (x−4)2(x-4)^{2} and (y+2)2(y+2)^{2} must be perfect squares that sum to 25. The pairs (a,b)(a,b) such that a2+b2=25a^{2}+b^{2}=25 are (0,5)(0,5), (3,4)(3,4), (4,3)(4,3), and (5,0)(5,0).

1.

(x−4)2=0,(y+2)2=25(x-4)^{2}=0,\;(y+2)^{2}=25: x=4x=4, y=3y=3 or −7-7. Only (4,3)(4,3) valid.
2.

(x−4)2=9,(y+2)2=16(x-4)^{2}=9,\;(y+2)^{2}=16: x=7x=7 or 11, y=2y=2 or −6-6. Positive integer solutions are (7,2)(7,2) and (1,2)(1,2), but “(1,2)(1,2) is not valid since x=1x=1 is not a positive integer in the context of the circle’s radius.”
3.

(x−4)2=16,(y+2)2=9(x-4)^{2}=16,\;(y+2)^{2}=9: x=8x=8 or 0, y=1y=1 or −5-5. Only (8,1)(8,1) valid.
4.

(x−4)2=25,(y+2)2=0(x-4)^{2}=25,\;(y+2)^{2}=0: x=9x=9 or −1-1, y=−2y=-2. No valid solutions.
Therefore, the positive integer solutions are (4,3)(4,3), (7,2)(7,2), and (8,1)(8,1). The number of ordered pairs is 3\boxed{3}.

PAPO Response (correct answer: 4)

To solve the problem of finding the number of ordered pairs (x,y)(x,y) of positive integers satisfying the equation x2−8​x+y2+4​y=5x^{2}-8x+y^{2}+4y=5, we will follow these steps:
1. Complete the square for xx and yy:
x2−8​x=(x−4)2−16x^{2}-8x=(x-4)^{2}-16,  y2+4​y=(y+2)2−4y^{2}+4y=(y+2)^{2}-4.
2. Substitute:
(x−4)2+(y+2)2−20=5⇒(x−4)2+(y+2)2=25(x-4)^{2}+(y+2)^{2}-20=5\;\Rightarrow\;(x-4)^{2}+(y+2)^{2}=25.
3. Identify all integer pairs (a,b)(a,b) with a2+b2=25a^{2}+b^{2}=25:
(±5,0)(\pm 5,0), (0,±5)(0,\pm 5), (±3,±4)(\pm 3,\pm 4), (±4,±3)(\pm 4,\pm 3) — 12 pairs total.
4. Substitute a=x−4a=x{-}4, b=y+2b=y{+}2, check x,y>0x,y>0:



(5,0)(5,0): x=9,y=−2x{=}9,y{=}{-}2
inv.
(−5,0)(-5,0): x=−1,y=−2x{=}{-}1,y{=}{-}2
inv.

(0,5)(0,5): x=4,y=3x{=}4,y{=}3
valid
(0,−5)(0,-5): x=4,y=−7x{=}4,y{=}{-}7
inv.

(3,4)(3,4): x=7,y=2x{=}7,y{=}2
valid
(3,−4)(3,-4): x=7,y=−6x{=}7,y{=}{-}6
inv.

(−3,4)(-3,4): x=1,y=2x{=}1,y{=}2
valid
(−3,−4)(-3,-4): x=1,y=−6x{=}1,y{=}{-}6
inv.

(4,3)(4,3): x=8,y=1x{=}8,y{=}1
valid
(4,−3)(4,-3): x=8,y=−5x{=}8,y{=}{-}5
inv.

(−4,3)(-4,3): x=0,y=1x{=}0,y{=}1
inv.
(−4,−3)(-4,-3): x=0,y=−5x{=}0,y{=}{-}5
inv.
5. Valid pairs: (4,3),(7,2),(1,2),(8,1)(4,3),(7,2),(1,2),(8,1). There are 4 valid pairs. 4\boxed{4}

### C.2 OlympiadBench #129 (Self-Verification)

Problem. What is the value of xx such that log2⁡(log2⁡(2​x−2))=2\log\_{2}\left(\log\_{2}(2x-2)\right)=2? Ground truth: 9

ORM Response (correct but no verification)

To solve the equation log2⁡(log2⁡(2​x−2))=2\log\_{2}(\log\_{2}(2x-2))=2, we need to follow a step-by-step approach.
First, let’s rewrite the equation in a more manageable form. Since log2⁡(log2⁡(2​x−2))=2\log\_{2}(\log\_{2}(2x-2))=2, we can exponentiate both sides with base 2 to eliminate the outer logarithm:



log2⁡(2​x−2)=22\log\_{2}(2x-2)=2^{2}
This simplifies to:



log2⁡(2​x−2)=4\log\_{2}(2x-2)=4
Next, we exponentiate both sides again with base 2 to eliminate the logarithm:



2​x−2=242x-2=2^{4}
This simplifies to:



2​x−2=162x-2=16
Now, we solve for xx by isolating it on one side of the equation. First, add 2 to both sides:



2​x=182x=18
Then, divide both sides by 2:



x=9x=9
Thus, the value of xx is 9\boxed{9}.

PAPO Response (correct with verification)

To solve the equation log2⁡(log2⁡(2​x−2))=2\log\_{2}\left(\log\_{2}(2x-2)\right)=2, we will follow these steps:
1. Solve the outer logarithm equation:
log2⁡(log2⁡(2​x−2))=2\log\_{2}\left(\log\_{2}(2x-2)\right)=2 implies that log2⁡(2​x−2)=22=4\log\_{2}(2x-2)=2^{2}=4.
2. Solve the inner logarithm equation:
log2⁡(2​x−2)=4\log\_{2}(2x-2)=4 implies that 2​x−2=24=162x-2=2^{4}=16.
3. Solve for xx:
2​x−2=162x-2=16, so 2​x=182x=18, giving x=9x=9.
4. Verify the solution:
Substitute x=9x=9 back into the original equation to ensure it satisfies the equation:



log2⁡(log2⁡(2⋅9−2))=log2⁡(log2⁡(16))\log\_{2}\left(\log\_{2}(2\cdot 9-2)\right)=\log\_{2}\left(\log\_{2}(16)\right)
Since log2⁡(16)=4\log\_{2}(16)=4 and log2⁡(4)=2\log\_{2}(4)=2, the solution is verified.
Therefore, the value of xx is 9\boxed{9}.

## Appendix D Ethics Statement

This work focuses on improving reinforcement learning algorithms for mathematical reasoning. Our research exclusively utilizes publicly available resources, including open-source models (Qwen2.5) and established datasets (NuminaMath-1.5), thereby mitigating concerns related to data privacy or human subjects. The application domain of mathematical problem-solving does not inherently present risks of direct societal harm. The primary ethical consideration is the environmental impact of the computational resources required for large-scale model training, a challenge common to the field.

## Appendix E The Use of Large Language Models

We used Large Language Model (LLM) to refine our initial draft. This process included checking for obvious grammatical and syntactical errors, as well as making the language more formal and academic. We reviewed the content generated by the LLM to ensure that no prohibited generated content appeared in the article.

[◄](/html/2603.26534)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2603.26535)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2603.26535)
[View original  
on arXiv](https://arxiv.org/abs/2603.26535)[►](/html/2603.26536)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Sun Apr 5 23:48:26 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
