---
arxiv: '2601.18734'
authors:
- Siyan Zhao
- Zhihui Xie
- Mengchen Liu
- Jing Huang
- Guan Pang
- Feiyu Chen
- Aditya Grover
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models'
url: https://arxiv.org/abs/2601.18734
year: 2026
---

[2601.18734] Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models















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



# Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models

Siyan Zhao†
  
Zhihui Xie
  
Mengchen Liu
  
Jing Huang
  
Guan Pang
  
Feiyu Chen∗,‡
  
Aditya Grover∗

###### Abstract

Knowledge distillation improves large language model (LLM) reasoning by compressing the knowledge of a teacher LLM to train smaller LLMs. On-policy distillation advances this approach by having the student sample its own trajectories while a teacher LLM provides dense token-level supervision, addressing the distribution mismatch between training and inference in off-policy distillation methods. However, on-policy distillation typically requires a separate, often larger, teacher LLM and does not explicitly leverage ground-truth solutions available in reasoning datasets. Inspired by the intuition that a sufficiently capable LLM can rationalize external privileged reasoning traces and teach its weaker self (i.e., the version without access to privileged information), we introduce *On-Policy Self-Distillation* (OPSD), a framework where a single model acts as both teacher and student by conditioning on different contexts. The teacher policy conditions on privileged information (e.g., verified reasoning traces) while the student policy sees only the question; training minimizes the per-token divergence between these distributions over the student’s own rollouts. We demonstrate the efficacy of our method on multiple mathematical reasoning benchmarks, achieving 4-8× token efficiency compared to reinforcement learning methods such as GRPO and superior performance over off-policy distillation methods.

Machine Learning, ICML

## 1 Introduction

![Refer to caption](/html/2601.18734/assets/x1.png)


Figure 1: Overview of On-Policy Self-Distillation (OPSD): Given a reasoning dataset 𝒮={(xi,yi⋆)}i=1N\mathcal{S}=\{(x\_{i},y\_{i}^{\star})\}\_{i=1}^{N}, we instantiate two policies from the same LLM: a *student policy* pS(⋅∣x)p\_{S}(\cdot\mid x) and a *teacher policy* pT(⋅∣x,y⋆)p\_{T}(\cdot\mid x,y^{\star}). The student generates an on-policy response y^∼pS(⋅∣x)\hat{y}\sim p\_{S}(\cdot\mid x). Both policies then evaluate this trajectory to produce next-token distributions pS(⋅∣x,y^<n)p\_{S}(\cdot\mid x,\hat{y}\_{<n}) and pT(⋅∣x,y⋆,y^<n)p\_{T}(\cdot\mid x,y^{\star},\hat{y}\_{<n}) at each step nn. The learning objective minimizes the per-token divergence D​(pT∥pS)D(p\_{T}\|p\_{S}) along the student’s rollout. Crucially, gradients backpropagate only through the student’s logits, allowing the model to self-distil.

Recent advances in large language models (LLMs) have demonstrated impressive capabilities in reasoning and instruction following. Achieving these capabilities during post-training typically relies on reinforcement learning methods such as Reinforcement Learning with Verifiable Rewards (RLVR) (e.g., GRPO (shao2024deepseekmath; guo2025deepseek; team2025kimi; rastogi2025magistral; yu2025dapo)), supervised fine-tuning (SFT) on high-quality reasoning datasets (guha2025openthoughtsdatarecipesreasoning; team2025kimi; xiao2026mimov2flashtechnicalreport), or knowledge distillation, where recent work has shown that distillation from advanced teacher models can outperform RL in both performance and training efficiency (qwen3; xiao2026mimov2flashtechnicalreport; lu2025onpolicydistillation).

Despite their respective successes, each approach has inherent limitations. RLVR suffers from inefficiencies including: (1) sampling a group of responses per prompt is computationally expensive and can introduce high variance in estimating the true value function; moreover, when all samples are either correct or incorrect, the gradient signal vanishes (yu2025dapo; zhao2025inpainting); and (2) the reward signal is sparse and uniformly applied across all tokens in the generated output, neglecting fine-grained token-level feedback. Supervised fine-tuning suffers from exposure bias and weaker generalization (agarwal2024policy; chu2025sft). Traditional knowledge distillation provides dense token-level supervision from a teacher model but relies on off-policy data (hinton2015distillingknowledgeneuralnetwork). Recent advances in on-policy distillation—where a student model samples its own trajectories while a teacher policy provides dense token-level supervision—have demonstrated superior sample efficiency by combining the distributional realism of on-policy training with dense feedback (agarwal2024policy; lu2025onpolicydistillation).

While on-policy distillation has shown strong performance, it relies on a distinct teacher model to supervise the student. Given that modern LLMs already exhibit strong reasoning capabilities, we ask this research question: *can a model effectively serve as its own teacher through self-distillation?* Our approach is inspired by human learning: after solving a problem incorrectly, a student can examine the correct solution, rationalize its steps, and identify where their reasoning failed. Prior work has shown that for LLMs, evaluation is often easier than generation (sun2024easy; naor1996evaluation). We hypothesize that *rationalization*—explaining a given correct answer—is similarly easier than generation. Motivated by this, we instantiate both the teacher and student policies from a single LLM. The teacher policy is provided with privileged information y⋆y^{\star}, such as the ground-truth answer or a reference chain-of-thought, while the student policy conditions only on the problem xx. Concretely, the teacher policy pT(⋅∣x,y⋆)p\_{T}(\cdot\mid x,y^{\star}) conditions on both the problem and the privileged answer, whereas the student policy pS(⋅∣x)p\_{S}(\cdot\mid x) observes only the problem. We preserve the on-policy training paradigm by sampling trajectories y^\hat{y} exclusively from the student policy, which then receives dense, token-level supervision from the privileged teacher policy.

We therefore propose On-Policy Self-Distillation (OPSD), a framework in which a single model plays both teacher and student roles. The student samples its own trajectories y^∼pS(⋅∣x)\hat{y}\sim p\_{S}(\cdot\mid x); we then compute the per-token divergence between the student and teacher distributions and minimize it over the student’s own rollouts. This formulation (i) uses on-policy supervision (the student’s own trajectories), (ii) provides dense per-token feedback, (iii) exploits ground-truth solutions y⋆y^{\star}, and (iv) requires no separate teacher model. The learning process is captured by the loss

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒOPSD\displaystyle\mathcal{L}\_{\mathrm{OPSD}} | (θ)=𝔼(x,y⋆)∼𝒮​𝔼y^∼pS(⋅∣x)​∑n=1|y^|\displaystyle(\theta)=\mathbb{E}\_{(x,y^{\star})\sim\mathcal{S}}\;\mathbb{E}\_{\hat{y}\sim p\_{S}(\cdot\mid x)}\sum\_{n=1}^{|\hat{y}|} |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | D(pT(⋅∣x,y⋆,y^<n)∥pS(⋅∣x,y^<n)).\displaystyle\quad D\!\Bigl(p\_{T}\!\left(\cdot\mid x,y^{\star},\hat{y}\_{<n}\right)\;\Big\|\;p\_{S}\!\left(\cdot\mid x,\hat{y}\_{<n}\right)\Bigr). |  | (1) |

In summary, our contributions are as follows:

* •

  We introduce On-Policy Self-Distillation, a novel framework that enables a single model to act as both teacher and student, leveraging ground-truth answers to provide dense token-level supervision on student rollouts.
* •

  We evaluate OPSD on four competition-level mathematical reasoning tasks, demonstrating that it outperforms both RLVR (e.g., GRPO) and supervised fine-tuning baselines.
* •

  We show that OPSD achieves better performance with nearly 8×8\times improved token efficiency and lower computational cost than GRPO.
* •

  We analyze the impact of model scale, finding that moderate model capacity is necessary for successful self-distillation. We further compare different divergence objectives and analyze the effect of student generation length.

|  | SFT/Off-Policy | GRPO | On-Policy | On-Policy |
| --- | --- | --- | --- | --- |
|  | Distillation |  | Distillation | Self-Distillation (Ours) |
| On-Policy Data | ✗ | ✓ | ✓ | ✓ |
| Dense Learning Signal | ✓ | ✗ | ✓ | ✓ |
| Low Sampling Cost | ✓ | ✗ | ✓ | ✓ |
| No External Teacher | ✓ | ✓ | ✗ | ✓ |

Table 1: Comparison of training methods for reasoning tasks. On-Policy Self-Distillation (OPSD) combines the advantages of on-policy training with dense feedback without requiring an external teacher model.

## 2 Background

### 2.1 Knowledge Distillation for Autoregressive Large Language Models

Knowledge distillation transfers knowledge from a larger teacher model to a smaller student model by training the student to mimic the teacher’s behavior (hinton2015distillingknowledgeneuralnetwork; kim2016sequence; sanh2019distilbert). The core insight is that the teacher’s soft probability distribution over classes contains richer information than hard labels alone, as it reveals the teacher’s learned similarities between classes. For auto-regressive language models, given a dataset 𝒮={(x,y⋆)}\mathcal{S}=\{(x,y^{\star})\} where xx denotes an input and y⋆y^{\star} is the corresponding reference output, both teacher pTp\_{T} and student pSp\_{S} define token-level distributions over vocabulary 𝒱\mathcal{V}. Traditional supervised distillation minimizes a divergence DD between teacher and student distributions averaged over a fixed dataset:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒSupervised Distillation​(θ)=𝔼(x,y)∼𝒮​[D​(pT∥pS)​(y|x)],\mathcal{L}\_{\text{Supervised Distillation}}(\theta)=\mathbb{E}\_{(x,y)\sim\mathcal{S}}[D(p\_{T}\|p\_{S})(y|x)], |  | (2) |

where D(pT∥pS)(y|x)=1|y|∑n=1|y|D(pT(⋅|y<n,x)∥pS(⋅|y<n,x))D(p\_{T}\|p\_{S})(y|x)=\frac{1}{|y|}\sum\_{n=1}^{|y|}D(p\_{T}(\cdot|y\_{<n},x)\|p\_{S}(\cdot|y\_{<n},x)) measures per-token discrepancy. However, this off-policy approach suffers from distribution mismatch: the student encounters different partial sequences y<ny\_{<n} during auto-regressive generation at inference than those seen during training on the fixed dataset, leading to compounding errors. On-policy distillation (agarwal2024policy; lu2025onpolicydistillation; xuspeculative) addresses this by training the student on its own generated sequences y^∼pS(⋅|x)\hat{y}\sim p\_{S}(\cdot|x), obtaining dense token-level feedback from the teacher on these on-policy samples:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒOn-Policy Distillation​(θ)=𝔼x∼𝒮​[𝔼y^∼pS(⋅|x)​[D​(pT∥pS)​(y^|x)]].\mathcal{L}\_{\text{On-Policy Distillation}}(\theta)=\mathbb{E}\_{x\sim\mathcal{S}}[\mathbb{E}\_{\hat{y}\sim p\_{S}(\cdot|x)}[D(p\_{T}\|p\_{S})(\hat{y}|x)]]. |  | (3) |

This approach connects distillation to imitation learning (ross2011reduction), where the student iteratively improves by learning from the teacher’s guidance on its own outputs, combining the on-policy relevance of reinforcement learning with the dense reward signal of supervised learning, thereby mitigating exposure bias while maintaining computational efficiency.

### 2.2 Reinforcement Learning with Verifiable Rewards

Reinforcement learning with verifiable rewards (RLVR) has emerged as a popular approach for post-training large language models, particularly on tasks with easily verifiable outcomes such as mathematics and coding, using algorithms like Proximal Policy Optimization (PPO) (schulman2017proximal) and Group Relative Policy Optimization (GRPO) (shao2024deepseekmath).

GRPO trains by sampling a group of GG responses {o1,o2,…,oG}\{o\_{1},o\_{2},\ldots,o\_{G}\} from the current policy πθ\pi\_{\theta} for each problem xx. Each response oio\_{i} receives a binary reward ri∈{0,1}r\_{i}\in\{0,1\} indicating correctness. The method then assigns advantages to all tokens k=1,…,|oi|k=1,\ldots,|o\_{i}| within response oio\_{i} using a group-normalized reward:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Ai=ri−mean​({rj}j=1G)std​({rj}j=1G).A\_{i}=\frac{r\_{i}-\text{mean}(\{r\_{j}\}\_{j=1}^{G})}{\text{std}(\{r\_{j}\}\_{j=1}^{G})}. |  | (4) |

This formulation can be understood through the value function lens: mean​({rj}j=1G)\text{mean}(\{r\_{j}\}\_{j=1}^{G}) serves as a GG-sample Monte Carlo estimate of the value function V​(x)V(x), while the sparse binary reward rir\_{i} represents the (undiscounted) state-action value Q​(x,oi)Q(x,o\_{i}). Critically, all tokens within a response share the same advantage, as the reward signal is provided only at the sequence level. The GRPO objective incorporates a clipped surrogate loss to moderate policy updates, along with a reverse KL penalty to prevent excessive deviation from a reference policy:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒGRPO(θ)=𝔼x∼𝒮o1,…,oG∼πθ(⋅|x)[1G∑i=1G1|oi|∑n=1|oi|min⁡(ρin​Ai,clip​(ρin,1−ε,1+ε)​Ai)−βDKL[πθ(⋅|x)∥πref(⋅|x)]]\begin{split}\mathcal{L}\_{\text{GRPO}}(\theta)=\mathbb{E}\_{\begin{subarray}{c}x\sim\mathcal{S}\\ o\_{1},\ldots,o\_{G}\sim\pi\_{\theta}(\cdot|x)\end{subarray}}\Bigg[\frac{1}{G}\sum\_{i=1}^{G}\frac{1}{|o\_{i}|}\sum\_{n=1}^{|o\_{i}|}\\ \min\left(\rho\_{i}^{n}A\_{i},\text{clip}\left(\rho\_{i}^{n},1-\varepsilon,1+\varepsilon\right)A\_{i}\right)\\ -\beta D\_{\text{KL}}[\pi\_{\theta}(\cdot|x)\|\pi\_{\text{ref}}(\cdot|x)]\Bigg]\end{split} |  | (5) |

where ρin=πθ​(oin|x,oi<n)πθold​(oin|x,oi<n)\rho\_{i}^{n}=\frac{\pi\_{\theta}(o\_{i}^{n}|x,o\_{i}^{<n})}{\pi\_{\theta\_{\text{old}}}(o\_{i}^{n}|x,o\_{i}^{<n})} is the importance ratio, πθold\pi\_{\theta\_{\text{old}}} is the policy before the update, and ε\varepsilon controls the clipping range.

While RLVR methods have demonstrated strong empirical performance, they face two key limitations: (1) the reward signal is sparse, providing only sequence-level feedback rather than token-level guidance on where errors occur, and (2) when all sampled responses receive identical rewards (all correct or all incorrect), the advantages become zero, preventing any policy update despite the computational cost of sampling.

## 3 Methods

Student Prompt

Problem: Find the derivative of f​(x)=3​x2+2​x−5f(x)=3x^{2}+2x-5 at x=2x=2
  
Answer:

Teacher Prompt

Problem: Find the derivative of f​(x)=3​x2+2​x−5f(x)=3x^{2}+2x-5 at x=2x=2
  
Here is a reference solution:
  
First find f′​(x)=6​x+2f^{\prime}(x)=6x+2, then evaluate at x=2x=2: f′​(2)=6​(2)+2=14f^{\prime}(2)=6(2)+2=14
  
After understanding the reference solution, please try to solve this problem using your own approach below:
  
Answer:

Figure 2: Prompt example for student and teacher policies. Both policies share the same parameters θ\theta but differ in conditioning context. The teacher receives the ground-truth solution y⋆y^{\star} as privileged information before generation. To ensure a natural transition before evaluating the student’s rollout, the teacher is prompted to rationalize and generate its own solution.

### 3.1 Learning from Verifiable Reasoning Dataset

We consider a dataset of problem-solution pairs
𝒮={(xi,yi⋆)}i=1N,\mathcal{S}=\{(x\_{i},y\_{i}^{\star})\}\_{i=1}^{N},
where each xix\_{i} denotes a problem and yi⋆y\_{i}^{\star} is the corresponding reference solution, which may include chain-of-thought reasoning. For brevity, we omit the sample index ii and use (x,y⋆)(x,y^{\star}) to denote a generic sample from the dataset. We can exploit learning signals from this dataset from different ways: Standard supervised fine-tuning (SFT) on 𝒮\mathcal{S} can be viewed as off-policy distillation/imitation learning using expert trajectories, but it suffers from distribution mismatch between training and inference. Reinforcement learning from verifiable rewards (RLVR), such as GRPO, addresses this by optimizing on-policy samples and assigning binary rewards by comparing generated answers against y⋆y^{\star}. However, RLVR is computationally expensive and the reward signal is sparse, providing same feedback across all tokens regardless of where errors occur. Alternatively, one can train a process reward model (PRM) to provide dense, token-level feedback during RL. However, acquiring labels for PRM training is prohibitively expensive and difficult to scale (lightman2023let; zhang2025lessons). On-policy distillation works (agarwal2024policy; xuspeculative; lu2025onpolicydistillation) address distribution shift by training on the student’s own samples, but require a separate, often larger, teacher model to provide supervision. We instead seek a training signal that is *dense*, *on-policy*, and *does not require external teachers or reward models*. This motivates our On-Policy Self-Distillation approach. We summarize the differences of these methods in Table [1](#S1.T1 "Table 1 ‣ 1 Introduction ‣ Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models").

Algorithm 1  On-Policy Self-Distillation (OPSD)

Reasoning dataset 𝒮={(xi,yi⋆)}i=1N\mathcal{S}=\{(x\_{i},y\_{i}^{\star})\}\_{i=1}^{N}; language model pθp\_{\theta}; divergence DD(e.g., JSDβ\mathrm{JSD}\_{\beta})
Define student policy pS(⋅∣x):=pθ(⋅∣x)p\_{S}(\cdot\mid x):=p\_{\theta}(\cdot\mid x)Define teacher policy pT(⋅∣x,y⋆):=pθ(⋅∣x,y⋆)p\_{T}(\cdot\mid x,y^{\star}):=p\_{\theta}(\cdot\mid x,y^{\star})same parameters; different conditioning
not converged
Sample a minibatch ℬ⊂𝒮\mathcal{B}\subset\mathcal{S}(x,y⋆)∈ℬ(x,y^{\star})\in\mathcal{B}Sample on-policy response y^∼pS(⋅∣x)\hat{y}\sim p\_{S}(\cdot\mid x)Compute the token-wise divergence along the student rollout:

|  |  |  |
| --- | --- | --- |
|  | ℓ(x,y⋆)←D(pT∥pS)(y^∣x)=1|y^|∑n=1|y^|D(pT(⋅∣y^<n,x,y⋆)∥pS(⋅∣y^<n,x))\ell(x,y^{\star})\leftarrow D\big(p\_{T}\,\|\,p\_{S}\big)(\hat{y}\mid x)=\frac{1}{|\hat{y}|}\sum\_{n=1}^{|\hat{y}|}D\!\left(p\_{T}(\cdot\mid\hat{y}\_{<n},x,y^{\star})\,\big\|\,p\_{S}(\cdot\mid\hat{y}\_{<n},x)\right) |  |

Batch loss ℒOPSD​(θ)←1|ℬ|​∑(x,y⋆)∈ℬℓ​(x,y⋆)\mathcal{L}\_{\mathrm{OPSD}}(\theta)\leftarrow\frac{1}{|\mathcal{B}|}\sum\_{(x,y^{\star})\in\mathcal{B}}\ell(x,y^{\star})Update θ←θ−η​∇θℒOPSD​(θ)\theta\leftarrow\theta-\eta\,\nabla\_{\theta}\mathcal{L}\_{\mathrm{OPSD}}(\theta)Returntrained parameters θ\thetafor inference-time policy pS(⋅∣x)p\_{S}(\cdot\mid x)

\Require

\State

\State

\Comment

\While

\State

\ForAll

\State

\State

\EndFor

\State

\State

\EndWhile

\State

### 3.2 On-Policy Self-Distillation

##### Motivation: Learning by understanding solutions.

We propose a different perspective inspired by how students learn: when struggling with a problem, rather than extended trial-and-error, a student can examine the solution, understand the reasoning, and internalize the approach. Similarly, if a model has access to the correct answer or reasoning y⋆y^{\star} and is sufficiently capable, it can rationalize the reasoning steps and teach itself—analogous to a student reviewing a solution and retracing why it works. This intuition motivates our framework: we exploit the ground-truth solution y⋆y^{\star} directly as privileged information during training, enabling the model to serve as its own teacher without requiring external reward models or larger teacher models.

##### Teacher and student policies.

We instantiate two conditional distributions from the same language model pθp\_{\theta} by varying the
conditioning context. The *teacher policy* conditions on privileged information—both the
problem xx and the reference solution y⋆y^{\star}:

|  |  |  |
| --- | --- | --- |
|  | pT(⋅∣x,y⋆)≜pθ(⋅∣x,y⋆).p\_{T}(\cdot\mid x,y^{\star})\;\triangleq\;p\_{\theta}(\cdot\mid x,y^{\star}). |  |

The *student policy* observes only the problem statement, matching the inference-time condition:

|  |  |  |
| --- | --- | --- |
|  | pS(⋅∣x)≜pθ(⋅∣x).p\_{S}(\cdot\mid x)\;\triangleq\;p\_{\theta}(\cdot\mid x). |  |

Critically, both policies share the same parameters θ\theta but differ only in their conditioning
context. The teacher has access to information unavailable at test time, allowing it to provide
informed guidance. To encourage the teacher to naturally evaluate the student’s generation, we add a prompt asking the teacher to generate a new solution after rationalization, as shown in [Figure 2](#S3.F2 "In 3 Methods ‣ Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models").

##### On-policy sampling from the student.

Given a problem xx, the student generates an on-policy response

|  |  |  |
| --- | --- | --- |
|  | y^=(y^1,…,y^|y^|)∼pS(⋅∣x).\hat{y}=(\hat{y}\_{1},\ldots,\hat{y}\_{|\hat{y}|})\sim p\_{S}(\cdot\mid x). |  |

Both policies then evaluate this student-generated trajectory. At each position nn, they induce
*next-token* distributions over yn∈𝒱y\_{n}\in\mathcal{V} conditioned on the same student prefix:

|  |  |  |
| --- | --- | --- |
|  | pS​(yn∣x,y^<n),pT​(yn∣x,y⋆,y^<n),p\_{S}\!\left(y\_{n}\mid x,\hat{y}\_{<n}\right),\qquad p\_{T}\!\left(y\_{n}\mid x,y^{\star},\hat{y}\_{<n}\right), |  |

where y^<n≜(y^1,…,y^n−1)\hat{y}\_{<n}\triangleq(\hat{y}\_{1},\ldots,\hat{y}\_{n-1}).

##### Training objective: Full-vocabulary divergence.

We instantiate a *full-vocabulary divergence objective* that matches the teacher and student
next-token distributions at each position. Given a student-generated sequence y^\hat{y}, define
the trajectory-averaged, token-wise divergence

|  |  |  |  |
| --- | --- | --- | --- |
|  | D​(pT∥pS)​(y^∣x)≜1|y^|∑n=1|y^|D(pT(⋅∣x,y⋆,y^<n)∥pS(⋅∣x,y^<n)),\begin{split}D\bigl(p\_{T}\,\|\,p\_{S}\bigr)(\hat{y}\mid x)&\triangleq\frac{1}{|\hat{y}|}\sum\_{n=1}^{|\hat{y}|}D\biggl(p\_{T}\!\left(\cdot\mid x,y^{\star},\hat{y}\_{<n}\right)\\ &\qquad\big\|\;p\_{S}\!\left(\cdot\mid x,\hat{y}\_{<n}\right)\biggr),\end{split} |  | (6) |

where pS(⋅∣x,y^<n)p\_{S}(\cdot\mid x,\hat{y}\_{<n}) and pT(⋅∣x,y⋆,y^<n)p\_{T}(\cdot\mid x,y^{\star},\hat{y}\_{<n}) denote distributions over the next token yn∈𝒱y\_{n}\in\mathcal{V}. Here, DD can be any distribution divergence measure such as the *generalized Jensen-Shannon divergence* JSDβ\operatorname{JSD}\_{\beta}, defined for a weight β∈[0,1]\beta\in[0,1] as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | JSDβ⁡(pT∥pS)=β​DK​L​(pT∥m)+(1−β)​DK​L​(pS∥m)\operatorname{JSD}\_{\beta}(p\_{T}\|p\_{S})=\beta D\_{KL}(p\_{T}\|m)+(1-\beta)D\_{KL}(p\_{S}\|m) |  | (7) |

where m=β​pT+(1−β)​pSm=\beta p\_{T}+(1-\beta)p\_{S} is the interpolated mixture distribution. This full-vocabulary formulation provides dense, token-level feedback: the teacher, informed by y⋆y^{\star}, exposes the student to the entire distribution over plausible next tokens and guides it toward reasoning paths that lead to the correct answer.

We minimize the expected divergence between teacher and student over on-policy student samples:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ​(θ)=𝔼(x,y⋆)∼𝒮​[𝔼y^∼pS(⋅∣x)​[D​(pT∥pS)​(y^∣x)]].\mathcal{L}(\theta)=\mathbb{E}\_{(x,y^{\star})\sim\mathcal{S}}\left[\mathbb{E}\_{\hat{y}\sim p\_{S}(\cdot\mid x)}\left[D\bigl(p\_{T}\,\|\,p\_{S}\bigr)(\hat{y}\mid x)\right]\right]. |  | (8) |

Gradients are backpropagated only through the student policy pSp\_{S}, while the teacher pTp\_{T} acts as
a fixed full-distribution target conditioned on privileged information (x,y⋆)(x,y^{\star}).

##### Alternative objective: Sampled-token distillation through policy gradient.

Alternatively, following recent on-policy distillation methods (lu2025onpolicydistillation),
we form a sampled-token shaping signal (equivalently, a reverse-KL signal on sampled actions) and
optimize with policy gradient. For each position nn in a sampled sequence y^\hat{y}, define the
advantage term

|  |  |  |
| --- | --- | --- |
|  | An​(x,y^)=log⁡pT​(y^n∣x,y⋆,y^<n)−log⁡pS​(y^n∣x,y^<n),A\_{n}(x,\hat{y})=\log p\_{T}\!\left(\hat{y}\_{n}\mid x,y^{\star},\hat{y}\_{<n}\right)-\log p\_{S}\!\left(\hat{y}\_{n}\mid x,\hat{y}\_{<n}\right), |  |

and optimize the policy-gradient-style objective

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ​(θ)=−𝔼(x,y⋆)∼𝒮[𝔼y^∼pS(⋅∣x)[1|y^|∑n=1|y^|An(x,y^)×logpS(y^n∣x,y^<n)]].\begin{split}\mathcal{L}(\theta)&=-\mathbb{E}\_{(x,y^{\star})\sim\mathcal{S}}\biggl[\mathbb{E}\_{\hat{y}\sim p\_{S}(\cdot\mid x)}\biggl[\frac{1}{|\hat{y}|}\sum\_{n=1}^{|\hat{y}|}A\_{n}(x,\hat{y})\\ &\qquad\times\log p\_{S}\!\left(\hat{y}\_{n}\mid x,\hat{y}\_{<n}\right)\biggr]\biggr].\end{split} |  | (9) |

In practice, An​(x,y^)A\_{n}(x,\hat{y}) is treated as a constant with respect to θ\theta (i.e., gradients do
not flow through the advantage), so that gradients take the usual policy-gradient form
An​∇θlog⁡pSA\_{n}\nabla\_{\theta}\log p\_{S}.
Compared to the full-vocabulary divergence objective, this on-policy shaping objective operates only
on sampled tokens, using the teacher’s log-probabilities to provide dense, trajectory-level shaping
signals without explicitly matching the full distribution at each step.

Table 2: Performance comparison across mathematical reasoning benchmarks for Qwen3 models from 1.7B to 8B. We report average@16 using suggested sampling parameters from the Qwen3 blog with temperature of 1.2 and generation length of 38k, with detailed parameter in [Table 5](#S8.T5 "In 8.1 Experimental Details ‣ 8 Appendix ‣ Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models").

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Method | AIME24 | AIME25 | HMMT25 | AMO-Bench | Average |
| Qwen3-8B | | | | | |
| Base (Instruct) | 75.2 | 68.3 | 43.1 | 13.4 | 50.0 |
| + SFT | 76.3 | 66.2 | 44.7 | 12.9 | 50.0 |
| + GRPO | 76.7 | 68.7 | 45.0 | 14.8 | 51.3 |
| + OPSD | 77.5 | 69.8 | 47.1 | 14.3 | 52.2 |
| Qwen3-4B | | | | | |
| Base (Instruct) | 74.6 | 65.8 | 40.3 | 12.4 | 48.3 |
| + SFT | 75.2 | 66.3 | 44.4 | 12.5 | 49.6 |
| + GRPO | 75.6 | 67.1 | 42.7 | 12.8 | 49.6 |
| + OPSD | 76.0 | 66.9 | 45.8 | 13.5 | 50.6 |
| Qwen3-1.7B | | | | | |
| Base (Instruct) | 50.2 | 35.2 | 25.4 | 4.3 | 28.8 |
| + SFT | 48.3 | 36.3 | 23.3 | 3.9 | 28.0 |
| + GRPO | 52.1 | 38.3 | 26.7 | 4.5 | 30.5 |
| + OPSD | 51.4 | 39.5 | 25.8 | 5.0 | 30.4 |

## 4 Experiments

In this section, we conduct comprehensive experiments to answer the following research questions:

* (1)

  How does OPSD compare to SFT and GRPO in terms of mathematical reasoning performance and what’s the improved sample efficiency? (§[4.2](#S4.SS2 "4.2 Main Results ‣ 4 Experiments ‣ Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models"))
* (2)

  How does OPSD scale across different model sizes, does self-distillation require more powerful model ability? (§[4.3.1](#S4.SS3.SSS1 "4.3.1 Effect of Model Scale ‣ 4.3 Discussions ‣ 4 Experiments ‣ Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models"))
* (3)

  What is the effect of generation length on training performance and sample efficiency? (§[4.3.2](#S4.SS3.SSS2 "4.3.2 Effect of Generation Length ‣ 4.3 Discussions ‣ 4 Experiments ‣ Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models"))
* (4)

  Does computing divergence over the full vocabulary logits provide benefits compared to computing it only over sampled tokens and optimizing through policy gradient? (§[4.3.3](#S4.SS3.SSS3 "4.3.3 Learning Objective Comparison: Full Vocabulary Logits Distillation vs. Sampled-Token Distillation ‣ 4.3 Discussions ‣ 4 Experiments ‣ Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models"))

### 4.1 Experimental Setup

Models and datasets. We experiment with the Qwen3 (qwen3technicalreport) model family at three scales: Qwen3-1.7B, Qwen3-4B, and Qwen3-8B, using the instruct-tuned versions. For training data, we use the mathematical reasoning subset of OpenThoughts (guha2025openthoughtsdatarecipesreasoning), sampling up to 30K problem-solution pairs with chain-of-thought reasoning. We evaluate on competition-level mathematics benchmarks including AIME 2024, AIME 2025, HMMT 2025 and Amo-Bench (an2025amo).

Baselines. We compare against two methods trained on the same dataset: (1) SFT, standard supervised fine-tuning on expert trajectories, which can be seen as off-policy distillation from a more powerful LLM that generated the reasoning traces; (2) GRPO (shao2024deepseekmath), group relative policy optimization with binary outcome rewards verified against ground-truth answers.

Implementation details. For GRPO, we sample 8 responses per problem. For OPSD, we sample 1 response per problem. We use Adam optimizer with a learning rate of 1e-5, warmup ratio of 0.1, and cosine learning rate decay. For the divergence measure in Eq. [6](#S3.E6 "Equation 6 ‣ Training objective: Full-vocabulary divergence. ‣ 3.2 On-Policy Self-Distillation ‣ 3 Methods ‣ Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models"), we use JSDβ=0.5\operatorname{JSD}\_{\beta=0.5}. Importantly, we fix the teacher policy to be the initial policy, rather than the currently updating learning policy, as we find this helps stabilize training and implicitly acts as regularization to prevent excessive deviation from the initial policy. All experiments are conducted on 8×A100 GPUs with LoRA (hu2022lora). More experimental details are in Appendix [8.1](#S8.SS1 "8.1 Experimental Details ‣ 8 Appendix ‣ Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models").

### 4.2 Main Results

![Refer to caption](/html/2601.18734/assets/x2.png)


Figure 3: Token Efficiency of OPSD. We compare OPSD and GRPO on Qwen3-4B under the same effective training batch size, reporting average@16 performance as a function of gradient update steps and total generated tokens. Both methods are trained with the same effective batch size in terms of sampled generations per update, but differ in generation length: each generation is capped at 2048 tokens for OPSD and 16384 tokens for GRPO. OPSD achieves comparable or better performance with substantially fewer generated tokens, resulting in lower sampling cost and reduced training time. In this experiment, OPSD can be 4-8×\times more token-efficient than GRPO.

Table [2](#S3.T2 "Table 2 ‣ Alternative objective: Sampled-token distillation through policy gradient. ‣ 3.2 On-Policy Self-Distillation ‣ 3 Methods ‣ Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models") reports results on competition-level mathematical reasoning benchmarks.
OPSD consistently outperforms SFT and improves over the base model across scales; it matches or exceeds GRPO at 4B/8B, and is comparable at 1.7B. Notably, OPSD accomplishes these gains using only a single rollout per problem, whereas GRPO requires 8 rollouts, demonstrating improved sample efficiency.

##### Superior Token Efficiency from Dense Teacher Feedback.

In addition to improved accuracy, OPSD is significantly more token-efficient than GRPO.
Figure [3](#S4.F3 "Figure 3 ‣ 4.2 Main Results ‣ 4 Experiments ‣ Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models") compares the two methods under the same effective training batch size on Qwen3-4B.
While GRPO relies on 8 rollouts with long generation budgets of 16k, OPSD achieves higher performance using substantially fewer generated tokens of 2k and needs only 1 rollout per prompt.
This efficiency stems from dense token-level supervision from the teacher distribution, reducing sampling cost and training time without sacrificing performance. We hypothesize that the early tokens are more important for distillation than the later tokens, as the earlier tokens can represent more important branching points.

![Refer to caption](/html/2601.18734/assets/x3.png)


Figure 4: Pass@K performance averaged across four mathematical reasoning benchmarks for Qwen3-4B. We study the effect of the generation length of on-policy sampled student responses in OPSD, comparing 1024, 2048, and 4096 tokens. Longer generations provide more teacher signals. Increasing the generation length from 1k to 2k and 4k consistently improves pass@K, with both 2k and 4k substantially outperforming the 1k setting.

### 4.3 Discussions

#### 4.3.1 Effect of Model Scale

Our method relies on the teacher policy’s ability to rationalize reference solutions when conditioned on privileged information. Under a fixed dataset, this capability depends on sufficient model capacity and is expected to scale with model size. We therefore hypothesize that OPSD becomes increasingly effective as models grow more capable of leveraging privileged context. To evaluate this, we apply OPSD to the Qwen3 family at three scales: 1.7B, 4B, and 8B parameters. As shown in [Table 2](#S3.T2 "In Alternative objective: Sampled-token distillation through policy gradient. ‣ 3.2 On-Policy Self-Distillation ‣ 3 Methods ‣ Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models"), OPSD provides limited gains over GRPO at the 1.7B scale although OPSD still improves over base and SFT at 1.7B., while yielding progressively larger improvements at the 4B and 8B scales, consistent with our hypothesis.

#### 4.3.2 Effect of Generation Length

Since our objective operates at the token level (Eq. [6](#S3.E6 "Equation 6 ‣ Training objective: Full-vocabulary divergence. ‣ 3.2 On-Policy Self-Distillation ‣ 3 Methods ‣ Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models")), the number of generated tokens per sample directly determines the amount of supervision signal available to the student. Longer sequences expose the student to more teacher feedback, but they also increase computational cost and may introduce noisy or uninformative continuations.

To study this trade-off, we conduct an ablation on Qwen3-4B by varying the generation length of on-policy sampled student responses among 1024, 2048, and 4096 tokens and use full-vocabulary logit distillation. As shown in Figure [4](#S4.F4 "Figure 4 ‣ Superior Token Efficiency from Dense Teacher Feedback. ‣ 4.2 Main Results ‣ 4 Experiments ‣ Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models"), increasing the generation length leads to clear improvements in pass@K performance. In particular, both the 2048-token and 4096-token settings significantly outperform the 1024-token baseline, indicating that longer generations provide more effective reasoning supervision.

Table 3: 
Ablation on divergence computation strategies for OPSD on Qwen3-4B with 2048 generation length for distillation.
We report pass@8 accuracy on AIME25 and HMMT25.
Full-distribution objectives (logit distillation) outperform sampled-token objectives.

| Method Variant | AIME25 | HMMT25 |
| --- | --- | --- |
| OPSD w/ Full-vocabulary logit distillation (agarwal2024policy) | 84.1 | 60.0 |
| OPSD w/ Sampled-token distillation (lu2025onpolicydistillation) | 82.1 | 57.3 |

#### 4.3.3 Learning Objective Comparison: Full Vocabulary Logits Distillation vs. Sampled-Token Distillation

Our objective in Eq. [6](#S3.E6 "Equation 6 ‣ Training objective: Full-vocabulary divergence. ‣ 3.2 On-Policy Self-Distillation ‣ 3 Methods ‣ Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models") is defined as a per-token discrepancy between the teacher and student *distributions*. In practice, OPSD can instantiate this objective in two ways. (1) Full-vocabulary logit distillation (as in GKD (agarwal2024policy)): for each token position, we compute D​(pT∥pS)D(p\_{T}\,\|\,p\_{S}) over the entire vocabulary via a full softmax, yielding a proper token-level ff-divergence between the two policies. (2) Sampled-token advantage policy-gradient objective (as in the on-policy distillation method of lu2025onpolicydistillation): we evaluate teacher and student log-probabilities only at the token actually sampled by the student, y^n\hat{y}\_{n}, and use the reverse-KL term as a scalar advantage inside a policy-gradient-style loss. Thus, the first variant directly matches full token distributions, whereas the second optimizes an on-policy RL objective shaped by the teacher’s log-probabilities rather than a full-distribution divergence. We compare these variants on Qwen3-4B using a 2048-token generation budget during distillation.
Table [3](#S4.T3 "Table 3 ‣ 4.3.2 Effect of Generation Length ‣ 4.3 Discussions ‣ 4 Experiments ‣ Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models") summarizes the results.
The full-vocabulary divergence objective provides a consistent gain over the sampled-token objective, improving AIME25 from 82.1% to 84.1% and HMMT25 from 57.3% to 60.0%.
This suggests that exposing the student to the full teacher distribution offers richer supervision than relying solely on per-token on-policy shaping.
However, the full-vocabulary computation incurs higher peak memory usage due to storing vocabulary-sized logits at every position, indicating a trade-off between performance and efficiency.

## 5 Related Work

On-Policy Distillation methods train a student model directly on trajectories sampled from its own policy, while a teacher model provides per-token guidance through KL-based regularization or related objectives (agarwal2024policy; xuspeculative; gu2024minillm; lu2025onpolicydistillation; xiao2026mimov2flashtechnicalreport; qwen3).
These approaches mitigate distribution shift by optimizing directly on the student’s visitation distribution, but they typically rely on a distinct and often larger teacher model.
In this work, we explore whether an LLM can teach itself by conditioning on more privileged answer information and leveraging its own reasoning capability to guide a weaker version of itself toward improved reasoning.
On-policy training paradigms are also widely used in robotics and deep reinforcement learning, such as DAgger (ross2011reduction), where a human teacher provides corrective supervision on the states visited by the student policy.

Improving LLM Reasoning through SFT and RL.
SFT and RL are two primary methods for improving LLM reasoning ability.
SFT on high-quality reasoning traces has demonstrated strong performance (yu2023metamath; numina\_math\_datasets; pasteropenwebmath; openthoughts), and that smaller, carefully curated datasets can outperform larger but noisier collections (ye2025limoreasoning; muennighoff2025s1; zhou2023lima).
However, prior work shows that SFT-based reasoning often relies on memorization rather than robust generalization (chu2025sft).
In contrast, RL-based approaches optimize directly for outcome-based objectives can exhibit stronger transfer to novel problems (huan2025does).
More recent algorithms such as GRPO (guo2025deepseek; shao2024deepseekmath) enable scalable RL by estimating advantages from group-level rewards without requiring an explicit critic as in PPO (schulman2017proximal).
Building on this line of work, a growing body of research highlights the effectiveness of reinforcement learning with verifiable rewards (RLVR) for reasoning tasks (yu2025dapo; liu2025understanding; yue2025vapo; Polaris2025; zheng2025group).

##### LLM Self-Training.

Our work is related to a growing body of research demonstrating that LLMs can improve by generating and exploiting their own supervision signals (allentowards; xu2024survey; chen2024self). Self-Instruct (wang2023self) and Self-Align (sun2023principle) demonstrate that large language models can bootstrap instruction-following and alignment with minimal human supervision by leveraging small sets of human-written seeds—either instructions or principles—to generate synthetic training data. Context distillation (snell2022learning) shows that models can internalize the benefits of privileged context tokens (e.g., instructions or scratchpads) by training a student to reproduce the same outputs without access to such context at inference time through SFT. Recent work on in-context editing (qicontext) demonstrates that models can learn new knowledge by optimizing toward self-induced contextual distributions rather than one-hot targets for knowledge editing. In the reasoning domain, ReST (gulcehre2023reinforced) and STaR (zelikman2022star) improve performance through iterative loops of rationale generation, filtering based on rewards or ground-truth answers, and fine-tuning on successful samples. LLM can also be used as a judge to generate RL rewards (yuan2024self) for itself. While aligned with this self-training paradigm, OPSD introduces a distinct approach: we perform on-policy, token-level self-distillation where the model learns from its own outputs conditioned on privileged access to ground-truth solutions. This transforms reasoning improvement into learning a conditional distribution induced by both the dataset’s ground-truth answers and the model’s own understanding of how to reach them.

## 6 Conclusion

We introduced On-Policy Self-Distillation (OPSD), a simple yet effective framework for post-training large language models on reasoning tasks. The intuition behind OPSD is that a sufficiently capable reasoning LLM can teach itself when it has access to privileged information about the answer to a reasoning problem, utilizing its own rationalization ability to grade its weaker self without access to the ground truth. We experimentally demonstrated that OPSD achieves better performance than off-policy distillation/SFT, and performs on par with or better than GRPO, while exhibiting significantly better sample efficiency than GRPO. Our ablation studies reveal that sufficiently large language models are required for successful self-distillation, and that generating more tokens during the online sampling phase and full-vocabulary logit distillation leads to improved learning.

## 7 Limitations and Future Directions

Due to computational constraints, our experiments are limited to models up to 8B parameters. While we observe that larger models benefit more from OPSD—consistent with our hypothesis that self-rationalization requires sufficient model capacity—it remains an open question whether this trend continues at scales beyond 8B parameters, such as 70B or larger frontier models.
Several promising directions warrant further investigation. First, our current framework does not explicitly leverage correctness verification of generated answers; incorporating such signals could provide additional learning objectives beyond distribution matching.
Finally, problem difficulty plays a crucial role in self-distillation: if reasoning problems exceed the model’s comprehension threshold, the teacher policy cannot provide meaningful supervision even with access to ground-truth solutions. This suggests that curriculum learning strategies—gradually increasing problem difficulty as the model improves—could enhance training effectiveness. Exploring adaptive curricula that maintain problems at the frontier of model capabilities represents an important direction for scaling OPSD to more challenging reasoning tasks.

## References

## 8 Appendix

### 8.1 Experimental Details

We provide the training and evaluation configurations for our SFT, GRPO and OPSD experiments in Tables [5](#S8.T5 "Table 5 ‣ 8.1 Experimental Details ‣ 8 Appendix ‣ Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models"), [6](#S8.T6 "Table 6 ‣ 8.1 Experimental Details ‣ 8 Appendix ‣ Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models") and [5](#S8.T5 "Table 5 ‣ 8.1 Experimental Details ‣ 8 Appendix ‣ Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models"). Both GRPO and OPSD methods use the same base hyperparameters where applicable to ensure fair comparison.

Table 4: Training Configuration for SFT.

|  |  |
| --- | --- |
| Parameter | SFT |
| Learning Rate | 2×10−52\times 10^{-5} |
| Batch Size (per device) | 2 |
| Gradient Accumulation Steps | 4 |
| Effective Batch Size | 64 |
| LoRA Rank (rr) | 64 |
| LoRA Alpha (α\alpha) | 128 |
| LoRA Target Modules | q\_proj, k\_proj, v\_proj, o\_proj, |
|  | gate\_proj, up\_proj, down\_proj |
| Max Sequence Length | 16000 |
| Number of Training Epochs | 4 |
| Training Dataset Size | 30k |

Table 5: Evaluation Parameters.

| Parameter | Value |
| --- | --- |
| Max New Tokens | 38912 |
| Thinking Mode | Enabled |
| Top-p | 0.95 |
| Top-k | -1 |
| Min-p | 0.0 |
| Presence Penalty | 0.0 |
| Samples per Prompt | 16 |




Table 6: Training Configuration for GRPO and OPSD

|  |  |  |
| --- | --- | --- |
| Parameter | GRPO | OPSD |
| Learning Rate | 2×10−52\times 10^{-5} | 2×10−52\times 10^{-5} |
| Batch Size (per device) | 1 | 1 |
| Gradient Accumulation Steps | 4 | 4 |
| Effective Batch Size | 32 | 32 |
| LoRA Rank (rr) | 64 | 64 |
| LoRA Alpha (α\alpha) | 128 | 128 |
| LoRA Target Modules | q\_proj, k\_proj, v\_proj, o\_proj, | |
|  | gate\_proj, up\_proj, down\_proj | |
| Max Completion Length | 16000 | 2048 |
| Number of Generations per Prompt | 8 | 1 |
| Temperature | 1.2 | 1.2 |
| KL Coefficient (β\beta) | 0.0 | – |

All experiments were conducted using 8 A100 GPUs with gradient checkpointing and Flash Attention 2 for memory efficiency. We use the AdamW (loshchilov2017decoupled) optimizer and bfloat16 precision for all training runs. For OPSD, unless otherwise stated, we used full-vocabulary logit distillation.

[◄](/html/2601.18733)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2601.18734)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2601.18734)
[View original  
on arXiv](https://arxiv.org/abs/2601.18734)[►](/html/2601.18735)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Thu Feb 5 20:28:20 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
