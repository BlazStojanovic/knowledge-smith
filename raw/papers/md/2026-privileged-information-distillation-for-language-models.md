---
arxiv: '2602.04942'
authors:
- Emiliano Penaloza
- Dheeraj Vattikonda
- Nicolas Gontier
- Alexandre Lacoste
- Laurent Charlin
- Massimo Caccia
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Privileged Information Distillation for Language Models
url: https://arxiv.org/abs/2602.04942
year: 2026
---

# Privileged Information Distillation for Language Models

Emiliano Penaloza
  
Dheeraj Vattikonda
  
Nicolas Gontier
  

  
Alexandre Lacoste
  
Laurent Charlin
  
Massimo Caccia
[
[
[
[
[
[emilianopp550@gmail.com](mailto:emilianopp550@gmail.com)

###### Abstract

Training-time privileged information (PI) can enable language models to succeed on tasks they would otherwise fail, making it a powerful tool for Reinforcement Learning (RL) in hard, long-horizon settings. However, transferring capabilities learned with PI to policies that must act without it at inference time remains a fundamental challenge.
We study this problem in the context of distilling frontier models for multi-turn agentic environments, which typically hide their internal reasoning and expose only action trajectories. This breaks standard distillation pipelines, since successful behavior is observable, but the reasoning process is not.
For this, we introduce π\pi-Distill, a joint teacher-student objective that trains a PI-conditioned teacher and an unconditioned student simultaneously using the same model. Additionally, we also introduce On-Policy Self-Distillation (OPSD), an alternative approach that trains using RL with a reverse KL-penalty between the student and the PI-conditioned teacher.
We show that both of these algorithms effectively distill frontier agents using action-only PI. Specifically, we find that π\pi-Distill and, in some cases, OPSD, outperform industry standard practices (Supervised finetuning followed by RL) that assume access to full Chain-of-Thought supervision across multiple agentic benchmarks, models, and forms of PI.
We complement our results with extensive analysis that characterizes the factors enabling effective learning with PI, focusing primarily on π\pi-Distill and characterizing when OPSD is competitive.

\correspondence

!(/html/2602.04942/assets/x1.png)

Figure 1: Overview of the π\pi-Distill framework.
(1) Successful trajectories (not shown) are collected from a frontier agent that exposes only actions while hiding its Chain-of-Thought.
(2) These trajectories are transformed into training-time privileged information (PI) and used to sample using a PI-conditioned teacher policy πθT​(𝐨∣s,𝐈){\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})}.
(3) The PI-conditioned teacher and an unconditioned student πθS​(𝐨|s){\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}(\mathbf{o}|s)} share parameters and are trained jointly, enabling transfer of privileged knowledge to a test-time policy that acts without PI.

!(/html/2602.04942/assets/x2.png)

Figure 2: Results for Qwen3-8B on TravelPlanner and τ\tau-Bench retail.
The dashed line separating SFT w/ CoT + RL denotes that this method is not a required baseline, as all other methods do not rely on frontier-model CoT traces.
We find that both π\pi-Distill and OPSD substantially outperform all baselines in this setting.

## 1 Introduction

Language Models (LMs) have the unique ability to converse, which provides a superior user interface and more straightforward interactions than other machine learning systems.
Crafting good prompts, or equivalently, conditioning on the right context, remains essential to obtain the best possible performance from an LM. This raises the question of whether LMs can learn from informative prompts to generalize to less informative prompts. In other words, how can an LM learn to transfer training-time privileged information (PI) (vapnik2009\_learning\_privileged) to test tasks (that do not contain PI)?

Training-time PI can be particularly useful for Reinforcement Learning (RL) with LMs, where learning is contingent on the model’s ability to first succeed on a task (yue2025doesreinforcementlearningreally). That is, the model can leverage train-time PI to succeed at tasks it would otherwise fail, effectively enabling it to bootstrap its learning from these successful experiences. As noted above, the underlying challenge is transfer: training a model with PI and obtaining a test-time policy that retains those enhanced capabilities without PI.

In this work, we show that leveraging train-time PI is highly effective under certain conditions. For instance, conditioning a policy on PI can drastically shift the sampling distribution away from the unconditioned one, making transfer significantly harder. Additionally, we find that non-frontier models often struggle to accurately leverage PI and must explicitly learn to use it.
To enable effective training in this setting, we introduce two complementary distillation objectives: Privileged Information Distillation (π\pi-Distill), our main method, and On-Policy Self-Distillation (OPSD), an on-policy alternative.
π\pi-Distill adapts the typical teacher-student setup by using a single shared-parameter model in which the teacher has access to PI.
Importantly, π\pi-Distill trains the teacher and student jointly, enabling the teacher to learn how to use PI while actively mitigating distribution shift during transfer. OPSD similarly allows the teacher and student to share parameters, regularizing on-policy RL with a reverse KL penalty between the student and a PI-conditioned teacher.

We ground our work in the task of distilling frontier models for complex multi-turn agentic settings. Typically, the industry standard for these tasks involves Supervised Fine-Tuning (SFT) on frontier model outputs followed by Reinforcement Learning (RL). Unfortunately, some model providers restrict important information, most notably the model’s full Chain-of-Thought (CoT) reasoning traces (openai2024openaio1card), providing only a summary alongside the action they intend to take. This opacity undermines standard distillation methods, as we can observe what successful agents do but not how they reason about it.

We find that π\pi-Distill is highly effective at mitigating the lack of CoT, outperforming industry standards that assume full CoT access, with this holding for OPSD in some cases. We demonstrate this on two agentic tool-use environments, Travel Planner (xie2024travelplannerbenchmarkrealworldplanning) and τ\tau-Bench retail (yao2024taubenchbenchmarktoolagentuserinteraction), while showing proficient Out-of-Domain generalization on the 7 tool-use environments provided by GEM (liu2025gemgymagenticllms) and τ\tau-Bench airline. Our findings are consistent across three models from two distinct families.

Finally, we transform frontier model trajectories into three varying types of PI, each with different information density, providing varying amounts of utility and inducing different degrees of distributional shift between student and teacher. We use this variation to analyze the critical factors for training with PI, finding that for π\pi-Distill, maximizing PI utility while simultaneously mitigating the student-teacher distributional gap and preventing collapse is essential for effective learning. While information content is more important for OPSD.

Contributions

1.

Algorithms for Privileged Information: We introduce π\pi-Distill and On-Policy Self-Distillation (OPSD), two methods for training policies using train-time privileged information.
2.

Distilling without CoT: We show that our methods effectively distill frontier models using *actions alone*, bypassing the need for frontier model CoT traces. Notably, π\pi-Distill consistently outperforms standard SFT+RL baselines that rely on full Chain-of-Thought data.
3.

Generalization: We demonstrate significant gains over baselines on τ\tau-Bench and Travel Planner, with strong generalization to eight additional out-of-distribution (OOD) on GEM tool-use tasks.

## 2 Background

#### Agentic Interaction as an MDP.

We formalize long-horizon, multi-turn agentic environments as a Markov Decision Process (MDP). In this setting, a policy πθ(⋅∣𝐬)\pi\_{\theta}(\cdot\mid\mathbf{s}) interacts with an environment over extended sequences of actions.
To simplify notation, we let 𝐬\mathbf{s} represent the *evolving interaction context*, which aggregates all information available to the model at a given point: the initial user prompt, the model’s past outputs, and all environment responses.
As the agent acts, the context updates via a transition function 𝐬t+1∼P(⋅∣𝐬t,𝐨t)\mathbf{s}\_{t+1}\sim P(\cdot\mid\mathbf{s}\_{t},\mathbf{o}\_{t}), where the environment appends its response to the current sequence to form the next state.
As 𝐬\mathbf{s} encapsulates the interaction history, we use the following to simplify notation:

|  |  |  |
| --- | --- | --- |
|  | πθ​(𝐨∣𝐬)=∏i=0Tπθ​(𝐳i,𝐚i∣𝐬<i)\pi\_{\theta}(\mathbf{o}\mid\mathbf{s})=\prod\_{i=0}^{T}\pi\_{\theta}(\mathbf{z}\_{i},\mathbf{a}\_{i}\mid\mathbf{s}\_{<i}) |  |

where 𝐨=(𝐳,𝐚)\mathbf{o}=(\mathbf{z},\mathbf{a}) consists of reasoning tokens 𝐳\mathbf{z} and action tokens 𝐚\mathbf{a}.

#### Reinforcement learning.

RL seeks to maximize the expected return obtained through interaction with an environment (sutton1999policy):

|  |  |  |
| --- | --- | --- |
|  | J​(πθ)=𝔼𝐨∼πθ(⋅∣𝐬)𝐬∼P​[R​(𝐨,𝐬)].J(\pi\_{\theta})=\mathbb{E}\_{\begin{subarray}{c}\mathbf{o}\sim\pi\_{\theta}(\cdot\mid\mathbf{s})\\ \mathbf{s}\sim P\end{subarray}}\Big[R(\mathbf{o},\mathbf{s})\Big]. |  |

Here, R​(𝐨,𝐬)∈[−1,1]R(\mathbf{o},\mathbf{s})\in[-1,1] denotes the reward assigned by the environment to the generated trajectory.

#### Policy Optimization.

We optimize J​(πθ)J(\pi\_{\theta}) using Group Relative Policy Optimization (GRPO) (shao2024deepseekmathpushinglimitsmathematical; deepseekr1), with the adjustments recommended by yu2025dapo and liu2025understandingr1zeroliketrainingcritical. For each state 𝐬i\mathbf{s}\_{i}, we sample a group of GG trajectories {𝐨g}g=1G\{\mathbf{o}\_{g}\}\_{g=1}^{G} according to the current sampling policy μ\mu and the transition function PP. For each token kk in trajectory gg, we define the token-level importance ratio:

|  |  |  |
| --- | --- | --- |
|  | ρg,k​(θ)=πθ​(og,k∣𝐬i,𝐨g,<k)μ​(og,k∣𝐬i,𝐨g,<k).\rho\_{g,k}(\theta)=\frac{\pi\_{\theta}(o\_{g,k}\mid\mathbf{s}\_{i},\mathbf{o}\_{g,<k})}{\mu(o\_{g,k}\mid\mathbf{s}\_{i},\mathbf{o}\_{g,<k})}. |  |

We define a group-relative advantage As,gA\_{s,g} by comparing the return of trajectory gg to the average return of the sampled group, and use it to scale clipped importance-weighted policy updates.
The GRPO objective then is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | JGRPO​(θ)=𝔼𝐨∼πθ(⋅∣𝐬)𝐬∼P​[1∑gKg​∑g,kmin⁡(ρg,k​As,g,clip⁡(ρg,k,1−ϵ,1+ϵ)​As,g)]J\_{\text{GRPO}}(\theta)=\mathbb{E}\_{\begin{subarray}{c}\mathbf{o}\sim\pi\_{\theta}(\cdot\mid\mathbf{s})\\ \mathbf{s}\sim P\end{subarray}}\Bigg[\frac{1}{\sum\_{g}K\_{g}}\sum\_{g,k}\min\Big(\rho\_{g,k}A\_{s,g},\operatorname{clip}(\rho\_{g,k},1{-}\epsilon,1{+}\epsilon)A\_{s,g}\Big)\Bigg] |  | (1) |

where KgK\_{g} is the length of trajectory gg in tokens and ϵ\epsilon is the clipping parameter. While the original GRPO objective typically employs a KL-penalty with respect to the base model, we drop this term as recent work shows it can hinder performance (shah2026comedyestimatorsklregularization).

## 3 Methods

In this section, we introduce and motivate two algorithms for leveraging PI during training. Both methods use a teacher–student framework inspired by traditional distillation. However, the teacher and student share parameters, and only the teacher is conditioned on PI.

Privileged Information Distillation (π\pi-Distill) learns from teacher-generated traces by jointly improving both the teacher and the student. In contrast, on-policy self-distillation (OPSD) 111We adopt the naming ”self-distillation” following concurrent work (shenfeld2026selfdistillationenablescontinuallearning; zhao2026selfdistilledreasoneronpolicyselfdistillation; hubotter2026reinforcementlearningselfdistillation) that propose the same objective.  samples trajectories from the student policy and uses a reverse KL divergence between the student and the PI-conditioned teacher as a training penalty. Both algorithms are derived using a variational perspective, which we outline in detail in App. [A](#A1 "Appendix A Connections To Variational EM ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models") and App. [B](#A2 "Appendix B On-Policy Self-Distillation 𝐽_\"OPSD\" ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models").

### 3.1 Privileged Information Distillation (π\pi-Distill)

#### Motivation

A straightforward way to use PI for distillation is to condition a policy on PI to generate successful trajectories, which are then used for fine-tuning. However, this approach has two key limitations. First, base models do not automatically know how to exploit PI (see [section˜7](#S7 "7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models")), needing to learn how to use it, before providing benefit. Second, even once the policy can exploit PI, we still need to transfer this behavior to a policy that must act without PI at test time.

A naive solution is to first train a PI-conditioned policy and then distill its behavior into an unconditioned one. In practice, this sequential pipeline introduces several issues. It is unclear which checkpoint of the conditioned policy should be distilled, learning from its trajectories is off-policy and can be unstable, and training the two policies separately is computationally inefficient. Our early experiments confirm that this setup leads to suboptimal performance (see [figure˜11](#A1.F11 "In A.2 Failures of Variational EM. ‣ Appendix A Connections To Variational EM ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models")).

To address these challenges, we propose Privileged Information Distillation (π\pi-Distill), which trains both policies jointly within a single parameter-shared model. This allows the model to learn to exploit PI while simultaneously learning to act without it.

#### Algorithm.

Our approach uses a single model with shared parameters θ\theta that acts as both a teacher πθT​(𝐨∣s,𝐈){\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})} (conditioned on PI 𝐈\mathbf{I}) and a student πθS​(𝐨|s){\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}(\mathbf{o}|s)} (operating without PI). We train both simultaneously using two objectives.
The teacher objective trains the conditioned policy to maximize reward while maintaining proximity to the student policy:

|  |  |  |  |
| --- | --- | --- | --- |
|  | JTeacher(θ)=𝔼𝐨∼πθT​(𝐨∣s,𝐈)𝐬∼P[R(𝐨,𝐬)]−βDKL(πθT(𝐨∣s,𝐈)∥sg(πθS(𝐨|s))).\small J\_{\text{Teacher}}(\theta)=\mathbb{E}\_{\begin{subarray}{c}\mathbf{o}\sim{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})}\\ \mathbf{s}\sim P\end{subarray}}\Big[R(\mathbf{o},\mathbf{s})\Big]-\beta D\_{\mathrm{KL}}\!\left({\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})}\,\|\,\operatorname{sg}\!\left({\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}(\mathbf{o}|s)}\right)\right). |  | (2) |

This objective samples trajectories from the teacher policy and updates it to increase reward while subject to a reverse KL penalty DKL​(πθT∥πθS)D\_{\mathrm{KL}}\!\left({\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}}\,\|\,{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}}\right) controlled by β\beta, where sg(.)\operatorname{sg}\!\left(.\right) indicates the stop gradient operator. This objective serves two purposes: (i) it encourages the teacher to fit high-reward modes familiar to the student, making learning from its traces easier, and (ii) shared parameters promote transfer of the teacher’s knowledge to the student, even without directly training the student.

The student objective trains the unconditioned policy to learn from the teacher’s trajectories:

|  |  |  |  |
| --- | --- | --- | --- |
|  | JStudent(θ)=𝔼𝐨∼πθT​(𝐨∣s,𝐈)𝐬∼P[πθS​(𝐨|s)sg⁡(πθT​(𝐨∣s,𝐈))R(𝐨,𝐬)]−βDKL(sg(πθT(𝐨∣s,𝐈))∥πθS(𝐨|s)).J\_{\text{Student}}(\theta)=\mathbb{E}\_{\begin{subarray}{c}\mathbf{o}\sim{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})}\\ \mathbf{s}\sim P\end{subarray}}\Big[\dfrac{{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}(\mathbf{o}|s)}}{\operatorname{sg}\!\left({\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})}\right)}R(\mathbf{o},\mathbf{s})\Big]-\beta D\_{\mathrm{KL}}\!\left(\operatorname{sg}\!\left({\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})}\right)\,\|\,{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}(\mathbf{o}|s)}\right). |  | (3) |

This objective samples trajectories from the teacher (which has access to PI) but updates the student policy (which does not). This teaches the student to replicate the teacher’s high-reward behavior without needing PI.

Combining the student and teacher terms gives us our final objective:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Jπ-Distill​(θ)=α​JTeacher​(θ)+(1−α)​JStudent​(θ),J\_{\text{$\pi$-Distill}}(\theta)=\alpha J\_{\text{Teacher}}(\theta)+(1-\alpha)J\_{\text{Student}}(\theta), |  | (4) |

where α∈[0,1]\alpha\in[0,1] controls the balance between student and teacher focused learning.

When α=1\alpha=1, optimization focuses entirely on the teacher, although the student may still improve through shared parameters. When α=0\alpha=0, training is focused on student learning from the teacher’s current behavior. Where, we observe that under certain conditions, parameter sharing can still lead to improvements in the teacher without explicit teacher updates. When α=0.5\alpha=0.5, both are optimized jointly. Shared parameters allow representations learned for using PI to transfer to the student, while student updates keep those representations effective without PI. The full algorithm is given in [algorithm˜1](#alg1 "In Appendix C Detailed Algorithms ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models").

#### Connection to Variational EM.

This approach can be viewed as a form of Variational Expectation-Maximization (EM), where one uses an approximate posterior πT{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}} to approximate a target distribution π∗\pi^{\*}. Here the E-step first improves πT{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}} and the M-step distills this into the student policy πS{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}}. Traditionally this can be trained sequentially or in alternating loops with separate models (see zhou2025variationalreasoninglanguagemodels). We discuss this connection more in depth in App. [10](#A1.F10 "Figure 10 ‣ A.1 Variational EM ‣ Appendix A Connections To Variational EM ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models") and App. [A.2](#A1.SS2 "A.2 Failures of Variational EM. ‣ Appendix A Connections To Variational EM ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models"), characterizing the target distribution π∗\pi^{\*}, derive π\pi-Distill from this perspective and compare against sequential setups (similar to zhou2025variationalreasoninglanguagemodels).

### 3.2 On-Policy Self Distillation

#### Motivation.

π\pi-Distill can be viewed as off-policy learning, where the student is trained on trajectories generated by the PI-conditioned teacher. A complementary line of work studies on-policy distillation, in which the student acts as the sampling policy and knowledge is transferred by minimizing the reverse KL between the student and teacher (agarwal2024onpolicydistillationlanguagemodels; yang2026learningteachergeneralizedonpolicy). Prior work typically utilizes a larger model as the teacher.
We introduce this objective in our PI setting by instantiating it with the same shared-parameter model, where the teacher is additionally conditioned on PI.
We refer to this objective as On-Policy Self-Distillation (OPSD).

#### Algorithm.

The above intuition yields the following objective:

|  |  |  |  |
| --- | --- | --- | --- |
|  | JOPSD(θ)=𝔼𝐨∼πθS​(𝐨|s)𝐬∼P[R(𝐨,𝐬)]−βDKL(πθS(𝐨|s)∥sg(πθT(𝐨∣s,𝐈))).J\_{\text{OPSD}}(\theta)=\mathbb{E}\_{\begin{subarray}{c}\mathbf{o}\sim{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}(\mathbf{o}|s)}\\ \mathbf{s}\sim P\end{subarray}}\Big[R(\mathbf{o},\mathbf{s})\Big]-\beta\,D\_{\text{KL}}\!\left({\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}(\mathbf{o}|s)}\,\parallel\,\operatorname{sg}\!\left({\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})}\right)\right). |  | (5) |

Note that the updates are on-policy as the expectation is taken over πS{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}}. Where the reverse KL acts as a dense per-token reward measuring how closely the student matches the teacher. The full algorithm is given in [algorithm˜2](#alg2 "In Appendix C Detailed Algorithms ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models").
We analyze OPSD in greater depth in App [B](#A2 "Appendix B On-Policy Self-Distillation 𝐽_\"OPSD\" ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models") and characterize the specific target distribution that the algorithm implicitly fits. Concurrent work also propose this objective, demonstrating its effectiveness in settings such as having access to ground truth answers (zhao2026selfdistilledreasoneronpolicyselfdistillation), conditioning on reflective self-feedback (hubotter2026reinforcementlearningselfdistillation), and continual learning (shenfeld2026selfdistillationenablescontinuallearning).
In this work, we introduce and evaluate OPSD for PI transfer without ground truth and characterize it across a variety of PI types, identifying settings in which it fails.

## 4 Experimental Setting

While learning with PI is applicable to many settings, we ground our work in the task of distilling frontier models within multi-turn tool-calling environments. We focus on this domain for two primary reasons. First, non-frontier models often lack the capabilities required for such settings, whereas frontier models demonstrate proficient performance (singh2025openaigpt5card; geminiteam2025geminifamilyhighlycapable). Consequently, weaker models struggle to independently sample successful trajectories, making the PI derived from a frontier model’s actions highly valuable. Second, frontier models typically occlude their CoT reasoning (openai2024openaio1card). This renders standard distillation methods infeasible, a gap our proposed algorithms are designed to fill.

### 4.1 Benchmarks

To evaluate our approach, we first employ τ\tau-Bench (yao2024taubenchbenchmarktoolagentuserinteraction), which simulates customer service interactions where agents book flights (airline domain) or assist shoppers (retail domain) by calling tools and gathering user information. To reduce computational costs, we substitute the GPT-4o user simulator with Qwen-14B. The most substantial change is that we remove the transfer\_to\_human\_agents tool, as it consistently led to reward hacking. The resulting dataset consists of 500 training tasks restricted to the retail domain; we evaluate on 115 held-out retail tasks and 50 airline tasks, utilizing the latter to test Out-Of-Domain (OOD) generalization. Next, we utilize Travel Planner (xie2024travelplannerbenchmarkrealworldplanning), a benchmark focusing on tool use for planning. While the original repository222https://github.com/OSU-NLP-Group/TravelPlanner employs a rubric-based evaluation that prioritizes "easy" constraints before checking "hard" ones, we found this setup causes policies to collapse onto undesired behaviors (see App. [G](#A7 "Appendix G Reward Hacking in Travel Planner ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models")). To address this, we decouple the rewards so that easy constraints are tied directly to their corresponding hard constraints, for example, verifying dietary restrictions immediately after booking a restaurant. We train on the 45 training tasks and report results on the 180 publicly available held-out tasks. Finally, to probe whether training on these environments enhances tool usage to scopes far beyond the training domains, we evaluate on the GEM QA multi-turn tool-usage environment suite (liu2025gemgymagenticllms). This suite equips agents with a search tool consisting of seven environments (2Wiki (xanh2020\_2wikimultihop), PopQA (mallen2023llm\_memorization), TriviaQA (joshi2017triviaqalargescaledistantly), HotpotQA (hotpotqa), Bamboogle (bamboogle), NaturalQuestions (naturalquestions), and Musique (trivedi2022musiquemultihopquestionssinglehop)).

### 4.2 Sources of Privileged Information

To show the benefits of leveraging train-time PI, we aim to distill a frontier model using only its raw output trajectories. For this, we mine trajectories from DeepSeek-chat-v3.1 as it is open-source and allows access to its reasoning tokens. Having access to these tokens allows us to benchmark against standard settings that assume access to CoT. This is an important detail, as baselines that assume access to this information should act as a soft-upper bound.

!(/html/2602.04942/assets/x3.png)

Figure 3: Deriving PI from frontier model traces. The left panel illustrates sampling trajectories from a frontier model, where full CoT reasoning is occluded (openai2024openaio1card). We transform these raw traces into three types of PI with varying information density: (1) Tool Calls & Arguments, retaining the exact actions invoked by the frontier model; (2) Tool Calls Only, where arguments are stripped; and (3) Self-Generated Hints, where the student model summarizes the frontier trajectory into a concise hint.

#### Types of Privileged Information.

We transform the raw trajectories into three distinct forms of PI to analyze how varying information density impacts performance.

1. 1.

   Tool calls & arguments. The complete sequence of actions executed by the frontier model, including function names and input arguments, e.g. GetUserDetails(Name:"Kevin Lau").
2. 2.

   Tool calls only. We provide only the function names like GetUserDetails and require the model to infer the correct arguments from the context.
3. 3.

   Self-generated hints. We prompt the trained model to summarize a successful trajectory produced by the frontier model. This process can help filter inefficient actions and suggest ways to improve upon the expert’s behavior (see App. [H.3](#A8.SS3 "H.3 Self-generated hints ‣ Appendix H Prompts ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models") for an example).

Leveraging these three formats enables us to identify what properties allow successful transfer between teacher and student. We analyze this in [section˜7](#S7 "7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models"), finding that the optimal configuration depends on the value of α\alpha and the specific attributes of the PI (e.g. KL between teacher and student).

Using this setup, we collect 15,885 successful traces for τ\tau-Bench retail and 1,986 for Travel Planner, evenly sampled across tasks. From these traces, for each training task, we select the successful trajectory with the least number of steps to build the PI. In total, we obtain PI for all 4545 tasks in Travel Planner and 300/500300/500 tasks in τ\tau-Bench. We note we exclude the self-generated hints for R1-Deepseek-Distill-LLama-8B as it consistently returns the raw input trace or tool calls as the hint.

### 4.3 Models and Baselines

#### Models.

We employ Qwen3-4B and Qwen3-8B (yang2025qwen3technicalreport) being strong reasoning models. We also evaluate R1-Distill-Llama-8B to cover a distinct model family. We find this model fails to generate correct trajectories even when conditioned on PI in both benchmarks, making direct RL training unfeasible. Thus, exclusively for R1-Distill-Llama-8B, we warm-start it using SFT w/ CoT from expert traces. This setup allows us to determine if PI remains beneficial even when the model has already seen it during training.

#### Baselines.

We instantiate π\pi-Distill with α∈{0,0.5,1}\alpha\in\{0,0.5,1\} and OPSD. We compare against (i) standard RL, (ii) SFT on expert trajectories with and without CoT, and (iii) SFT followed by RL. We utilize the GRPO objective outlined in [equation˜1](#S2.E1 "In Policy Optimization. ‣ 2 Background ‣ Privileged Information Distillation for Language Models") for all RL-based algorithms. Following vattikonda2025trainllmwebagent, we sweep over multiple SFT checkpoints for SFT+RL baselines and report results using the checkpoint that yields the strongest final performance. We use the full set of collected successful traces for all SFT baselines, as this maximized performance. For π\pi-Distill and OPSD in τ\tau-Bench, we utilize PI whenever available otherwise, we sample the traces with the student and perform regular RL for that goal.

#### Implementation Details.

We run all experiments using 2 H100 GPUs with a context limit of 2525k tokens (with one exception, see App. [E](#A5 "Appendix E Implementation setup - Further details ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models")). We found initially that traces often exceeded this limit, thus adopting a length penalty reward that penalizes trajectories exceeding 1515k tokens (full details in App. [E.2](#A5.SS2 "E.2 Length Penalty ‣ Appendix E Implementation setup - Further details ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models")). Additionally, we observed during training that a few tokens referencing the PI consistently exhibited very high KL (e.g., the token “hint”). We thus incorporate a penalty on the frequency of these tokens. Our final experiments incorporate this penalty, though we found in practice it makes little difference for final performance (see App. [D](#A4 "Appendix D Additional Ablations ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models") for full details). All PI is added to the task description in the system prompt, using the prompts in App. [H.4](#A8.SS4 "H.4 Privileged Information Prompts ‣ Appendix H Prompts ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models"). We thoroughly sweep over relevant HPs for all baselines, for a full list see App. [E.3](#A5.SS3 "E.3 Hyper-Parameters ‣ Appendix E Implementation setup - Further details ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models"). Using the best performing set, we run three seeds of this set and report their average. In experiments, we score models using the best running average over three subsequent checkpoints, capturing both performance and stability.

Table 1: Evaluation results on Travel Planner, τ\tau-Bench (Retail), and τ\tau-Bench (Airline). Shaded rows denote our methods. Bold values indicate the best performance within each model category, while underlined values indicate the second-best. Results show mean ±\pm standard deviation across three random seeds. We find that both π\pi-Distill and OPSD effectively leverage PI, consistently outperforming all baselines that lack access to frontier reasoning traces. Furthermore, both methods can surpass SFT w/ CoT + RL on TravelPlanner, with π\pi-Distill also achieving superior performance on τ\tau-Bench.

|  |  |  |  |
| --- | --- | --- | --- |
|  | Travel Planner | τ\tau-Bench Retail | τ\tau-Bench Airline (OOD) |
| \rowcolorMethodYellow!15     DeepSeek V3.1 Chat-671B | | | |
| Base | 45.0%  ±\pm 3.78 | 51.3%  ±\pm 0.212 | 40.0%  ±\pm 0.161 |
| \rowcolorMethodYellow!15     R1-Distill-Llama-8B | | | |
| Base | 0.00%  ±\pm 0.00 | 0.00%  ±\pm 0.00 | 0.00%  ±\pm 0.00 |
| SFT w/ CoT | 6.35%  ±\pm 1.27 | 15.2%  ±\pm 0.44 | 8.00%  ±\pm 5.29 |
| SFT w/o CoT | 2.40%  ±\pm 0.86 | 0.680%  ±\pm 0.17 | 0.670%  ±\pm 1.15 |
| SFT w/ CoT + RL | 12.4%  ±\pm 1.56 | 16.3%  ±\pm 1.49 | 7.33%  ±\pm 4.16 |
| \rowcolorMethodColor!5 SFT w/ CoT + On-Policy Self Distillation | 13.1%  ±\pm 1.09 | 14.5%  ±\pm 0.00 | 2.00%  ±\pm 0.00 |
| \rowcolorMethodColor!5 SFT w/ CoT + π\pi-Distill πS{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}} (α=0\alpha=0) | 7.86%  ±\pm 1.75 | 18.6%  ±\pm 0.50 | 10.0%  ±\pm 2.00 |
| \rowcolorMethodColor!5 SFT w/ CoT + π\pi-Distill πS+πT{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}}+{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}} (α=0.5\alpha=0.5) | 14.0%  ±\pm 1.63 | 18.3%  ±\pm 0.77 | 9.33%  ±\pm 3.06 |
| \rowcolorMethodColor!5 SFT w/ CoT + π\pi-Distill πT{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}} (α=1\alpha=1) | 14.1%  ±\pm 3.27 | 17.7%  ±\pm 0.77 | 7.33%  ±\pm 4.16 |
| \rowcolorMethodYellow!15     QWEN3-4B | | | |
| Base | 17.6%  ±\pm 2.16 | 5.03%  ±\pm 1.88 | 2.21%  ±\pm 1.99 |
| SFT w/ CoT | 21.1%  ±\pm 1.94 | 12.4%  ±\pm 0.60 | 2.67%  ±\pm 1.15 |
| SFT w/o CoT | 20.8%  ±\pm 1.82 | 15.2%  ±\pm 1.70 | 4.00%  ±\pm 5.29 |
| RL | 25.1%  ±\pm 3.75 | 15.2%  ±\pm 0.17 | 5.33%  ±\pm 3.06 |
| SFT w/o CoT + RL | 23.3%  ±\pm 1.33 | 17.6%  ±\pm 2.69 | 5.33%  ±\pm 2.31 |
| \rowcolorMethodColor!5 On-Policy Self Distillation | 29.8%  ±\pm 1.14 | 23.1%  ±\pm 0.04 | 10.6%  ±\pm 6.57 |
| \rowcolorMethodColor!5 π\pi-Distill πS{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}} (α=0\alpha=0) | 28.5%  ±\pm 4.35 | 25.3%  ±\pm 0.60 | 8.00%  ±\pm 5.29 |
| \rowcolorMethodColor!5 π\pi-Distill πS+πT{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}}+{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}} (α=0.5\alpha=0.5) | 33.8%  ±\pm 6.85 | 22.6%  ±\pm 0.93 | 6.00%  ±\pm 2.00 |
| \rowcolorMethodColor!5 π\pi-Distill πT{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}} (α=1\alpha=1) | 28.2%  ±\pm 6.27 | 22.5%  ±\pm 0.93 | 12.0%  ±\pm 5.29 |
| SFT w/ CoT + RL | 26.4%  ±\pm 1.16 | 23.3%  ±\pm 3.02 | 6.67%  ±\pm 5.77 |
| \rowcolorMethodYellow!15     QWEN3-8B | | | |
| Base | 23.6%  ±\pm 2.23 | 3.35%  ±\pm 1.47 | 6.40%  ±\pm 3.02 |
| SFT w/ CoT | 26.0%  ±\pm 2.27 | 16.5%  ±\pm 4.66 | 5.33%  ±\pm 1.15 |
| SFT w/o CoT | 29.8%  ±\pm 1.71 | 12.8%  ±\pm 0.77 | 6.00%  ±\pm 4.00 |
| RL | 27.5%  ±\pm 0.95 | 23.9%  ±\pm 0.44 | 6.67%  ±\pm 3.06 |
| SFT w/o CoT + RL | 31.3%  ±\pm 2.79 | 23.5%  ±\pm 4.27 | 6.00%  ±\pm 2.00 |
| \rowcolorMethodColor!5 On-Policy Self Distillation | 37.5%  ±\pm 1.53 | 27.3%  ±\pm 0.33 | 14.0%  ±\pm 5.66 |
| \rowcolorMethodColor!5 π\pi-Distill πS{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}} (α=0\alpha=0) | 40.7%  ±\pm 1.14 | 31.1%  ±\pm 0.73 | 12.0%  ±\pm 6.00 |
| \rowcolorMethodColor!5 π\pi-Distill πS+πT{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}}+{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}} (α=0.5\alpha=0.5) | 41.1%  ±\pm 7.24 | 30.6%  ±\pm 0.67 | 7.33%  ±\pm 1.15 |
| \rowcolorMethodColor!5 π\pi-Distill πT{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}} (α=1\alpha=1) | 44.1%  ±\pm 2.16 | 29.7%  ±\pm 0.33 | 9.33%  ±\pm 3.06 |
| SFT w/ CoT + RL | 32.3%  ±\pm 1.10 | 29.1%  ±\pm 2.14 | 8.00%  ±\pm 3.46 |

## 5 Main Results

In this section, we demonstrate that leveraging PI provides a potent learning signal on held-out tasks, proving effective even when full CoT supervision is available. [Section˜4.3](#S4.SS3.SSS0.Px3 "Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models") details performance metrics for Travel Planner and τ\tau-Bench. Our primary finding is that π\pi-Distill variants achieve superior performance across all but a single setting. With Qwen3-4B we find that π\pi-Distill using α=0\alpha=0 can outperform SFT + RL w/ CoT, with other variants approximating its performance. Notably, in Qwen3-8B, π\pi-Distill consistently outperforms the industry standard (SFT w/ CoT + RL) regardless of α\alpha. In the best-case scenarios, π\pi-Distill achieves substantial improvements: 11.8% on Travel Planner, and 2.08% and 6.00% on the retail and airline subsets of τ\tau-Bench, respectively. These results are substantial as they confirm that π\pi-Distill effectively distills frontier models even when CoT traces are hidden, enabling non-frontier models to become proficient at complex multi-step agentic tasks. Moreover, π\pi-Distill achieves this with significantly greater efficiency. Unlike SFT w/ CoT + RL, which requires sweeping over multiple SFT checkpoints to achieve peak performance (vattikonda2025trainllmwebagent), π\pi-Distill requires only a single training phase, greatly simplifying the training process.

Regarding α\alpha, we observe that there is no definitive best value, rather the best-performing α\alpha varies by setting. We analyze this nuance further in § [7](#S7 "7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models"). When analyzing other methods that do not assume access to full CoT, we find RL and SFT w/o CoT + RL perform similarly, while RL requires significantly less compute, but find both fail to yield substantial gains, significantly lagging behind SFT w/ CoT + RL. Additionally, we find that OPSD, can substantially outperform these baselines in τ−\tau-Bench, while performing similarly in Travel Planner. We also observe that when OPSD succeeds, it provides substantial gains in OOD settings (τ\tau-Bench Airline), scaling with model capacity, being the second best in Qwen3-4B and best in Qwen3-8B.
Finally, our results on R1-Distill-Llama-8B show that π\pi-Distill remains useful even if the model has previously been SFTd on traces containing the PI.

Takeaways:

1.

π\pi-Distill effectively substitutes for CoT: Our method consistently outperforms the standard SFT w/ CoT + RL baseline, demonstrating that raw actions can serve as a potent learning signal to distill frontier models even when reasoning traces are observable.
2.

Superior Efficiency: Unlike SFT that requires expensive sweeps over multiple SFT checkpoints, π\pi-Distill requires only a single training phase and yields superior performance gains even on models that have already undergone SFT w/ CoT.
3.

OPSD is a potent alternative when CoT is not available:  We find OPSD can substantially outperform base RL and SFT w/o CoT + RL being a potent alternative when CoTs are not available.

|  |
| --- |
| Refer to caption |

Figure 4: Evaluation on Out-of-Domain Environments. We report Pass@1 and Pass@10 on the GEM search-tool benchmark suite (7 datasets) for Qwen 3 models and R1-Distill-Llama-8B, using the best checkpoint selected on τ\tau-Bench Retail. Bars show mean ±\pm standard errors over three seeds per dataset, comparing π\pi-Distill variants (πT{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}} (α=1\alpha=1), πS{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}} (α=0\alpha=0), πT+πS{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}}+{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}} (α=0.5\alpha=0.5)) and OPSD against SFT w/ CoT + RL, πBase\pi\_{\text{Base}}, and standard RL. The dashed line separating SFT w/ CoT + RL denotes that this method is not considered a required baseline, as all PI methods avoid relying on frontier-model CoT traces. We consistently find that both algorithms exhibit substantially less forgetting than standard RL. Moreover, we find π\pi-Distill and OPSD generalize significantly better than SFT w/ CoT + RL when using Qwen 3-8B.

## 6 Out of Domain Experiments (OOD)

In this section, we demonstrate that π\pi-Distill generalizes effectively to OOD tasks, consistently outperforming standard RL and the base model, with this holding true for OPSD on Qwen3-8B. We report Pass@1 and Pass@10 metrics on the GEM (liu2025gemgymagenticllms) search-tool benchmark suite which consists of 7 datasets. To simulate a realistic deployment scenario, we select the single best-performing checkpoint in τ\tau-Bench retail for each model. We then evaluate these checkpoints across the suite using three random seeds and report the aggregated mean and standard errors in [figure˜4](#S5.F4 "In 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models").

We find that for both Qwen3 models, π\pi-Distill variants consistently outperform the base model and standard RL holding true for OPSD under more capable models. When compared to SFT w/ CoT + RL, we observe differences based on model size. On Qwen3-4B, SFT w/ CoT + RL is consistently the top performer with OPSD showing significant degradation. For Qwen3-8B, however, we find that α=0\alpha=0 and α=0.5\alpha=0.5 variants of π\pi-Distill as well as OPSD can significantly outperform SFT w/ CoT + RL. For π\pi-Distill, this aligns with our findings in [section˜5](#S5 "5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models"). Where for Qwen3-4B not all π\pi-Distill variants show substantial improvements over SFT w/ CoT + RL, whereas for Qwen3-8B, the opposite trend is observed. We observe a similar trend for OPSD, where it shows significant degradation on Qwen3-4B, while on Qwen3-8B it shows significant improvements. We attribute this to the possibility that on smaller models, OPSD may overfit the teacher supervision for the task, while the more potent reasoners can provide more generalizable feedback. Overall, results on both π\pi-Distill and OPSD imply that explicit supervision from frontier model’s CoT is more important for smaller models, where stronger reasoners benefit from being more on-policy via self-generated CoT. Additionally, we observe deterioration in R1-Distill-Llama-8B under SFT w/ CoT + RL, where performance drops below the base model. While π\pi-Distill and OPSD do not improve over the base model in this specific case, they avoid significant deterioration. Finally, raw RL consistently exhibits degradation relative to the base model across all evaluations. Overall, these results indicate that π\pi-Distill is highly effective for distillation in OOD settings when CoT traces are occluded, preventing performance from regressing below the base model. Moreover, both π\pi-Distill and OPSD show significant improvements as model size scales.

Takeaways: OOD Generalization

1.

Scaling Benefits for Stronger Reasoners: While SFT w/ CoT + RL is effective for Qwen3-4B, π\pi-Distill and OPSD significantly outperform SFT w/ CoT + RL on Qwen3-8B. This suggests that as model capacity grows, transfer can be enhanced by staying more on-policy.
2.

Prevention of RL Degradation: Across all benchmarks, standard RL consistently performs worse than the base model. In contrast, OPSD avoids degradation
in stronger reasoners, with π\pi-Distill avoiding it in all cases.

!(/html/2602.04942/assets/x5.png)

Figure 5: Impact of PI Types and Algorithms on Performance.
We compare held-out performance on τ\tau-Bench (top row) and Travel Planner (bottom row) across three base models and three PI types (colors). The scatter plots map final scores against the initial teacher-student divergence (DKL​(πbaseT∥πbaseS)D\_{\mathrm{KL}}(\pi^{T}\_{\text{base}}\,\|\,\pi^{S}\_{\text{base}})), while the bar-charts display the PI utility (Δ\Delta) on training tasks. Key observations:
(1) Higher initial KL divergence generally correlates with decreased final performance.
(2) Joint training (α=0.5\alpha=0.5, △\triangle) is the most stable configuration, performing best in 6/16 scenarios and worst in a single one.
(3) Student-only training (α=0\alpha=0, □\square) requires low KL and positive utility (note the failure in Planner QWEN3 8B where Δ<0\Delta<0). Conversely, Teacher-only training (α=1\alpha=1, ○\bigcirc) degrades as KL increases or fails due to policy collapse when KL is negligible.

## 7 What Matters When Using Train-Time PI

While prior sections focused on the best-performing configurations, here we analyze how varying the type of PI affects final performance. Our goal is to isolate the factors that determine success when training with PI. We identify two primary drivers: (i) the divergence between the conditioned and unconditioned base policies ( DKL​(πbaseT∥πbaseS)D\_{\mathrm{KL}}({\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\text{base}}}\,\|\,{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\text{base}}}) for π\pi-Distill and DKL​(πbaseS∥πbaseT)D\_{\mathrm{KL}}({\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\text{base}}}\,\|\,{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\text{base}}}) for OPSD), and (ii) the usefulness of the privileged signal, captured by the utility Δ=score​(πbaseT)−score​(πbaseS)\Delta=\text{score}({\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\text{base}}})-\text{score}({\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\text{base}}}) on training tasks.
Additionally, in [figure˜6](#S7.F6 "In 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models") we report the maximum attainable utility on Δmax=maxt⁡score​(πtPI)−maxt⁡score​(πtRL)\Delta\_{\max}=\max\_{t}\text{score}(\pi^{\text{PI}}\_{t})-\max\_{t}\text{score}(\pi^{\text{RL}}\_{t}), defined as the difference between the best scores on training tasks achieved with PI and without PI (pure RL), which measures how effectively each algorithm converts access to PI into performance gains.

!(/html/2602.04942/assets/x6.png)

Figure 6: Maximum Improvement (Δmax\Delta\_{\text{max}}) across PI Types and Algorithms. We compare the peak performance gain over baselines for different PI variants (x-axis) and training configurations (α\alpha) on TravelPlanner (bottom) and τ\tau-Bench (top). We find that PI types which initially underperform (e.g., self-generated hints on Planner, Qwen3-8B) can yield substantial gains when the teacher is trained to utilize them (α>0\alpha>0), confirming that learning to leverage PI is an important factor in transferring from teacher πT{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}} to student πS{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}}.

### 7.1 What matters for π\pi-Distill

!(/html/2602.04942/assets/x7.png)

Figure 7: Training KL between πT{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}} and πS{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}} during training the teacher πS​(α=1){\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}}(\alpha=1) on τ\tau-Bench. We observe an early KL collapse making πS≈πT{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}}\approx{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}}. We attribute the underperformance of πT{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}} on low KL settings to this collapse.

[Figure˜5](#S6.F5 "In 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models") displays held-out performance on both τ\tau-Bench Retail and Travel Planner across the types of PI defined in [Section˜4.2](#S4.SS2.SSS0.Px1 "Types of Privileged Information. ‣ 4.2 Sources of Privileged Information ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models"). Each subplot maps final performance against the initial divergence DKL​(πbaseT∥πbaseS)D\_{\mathrm{KL}}({\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\text{base}}}\,\|\,{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\text{base}}}), with accompanying bar plots showing the utility Δ\Delta for each type of PI. The most prominent pattern is that as the initial KL increases, final performance tends to decrease, though the full picture requires a more nuanced, α\alpha-dependent analysis.

#### Teacher only training α=𝟏\mathbf{\alpha=1}.

In this setting, we find performance generally declines or maintains as KL divergence increases. The primary exception occurs when using only tool calls in τ\tau-Bench, where this variant underperforms. We trace this failure mode to an early collapse in KL divergence. [Figure˜7](#S7.F7 "In 7.1 What matters for 𝜋-Distill ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models") displays the KL as training progresses. We observe that even when β=0\beta=0 the KL drops to near zero, indicating teacher and student have collapsed onto each other (πT≈πS{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}}\approx{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}}). As a consequence, we find the teacher πT{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}} learns to ignore the P,I causing it to underperform even the RL baseline. We attempted to mitigate this by using πbase\pi\_{\text{base}} instead of πS{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}} when calculating the KL, but found this further degraded results (see App. [D.2](#A4.SS2 "D.2 Reference Model Ablation ‣ Appendix D Additional Ablations ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models")).

In addition, our results confirm that training the teacher is an important part of leveraging PI. We can see this from the observation that even when the initial utility is negative (Δ<0\Delta<0), training the teacher allows the policy to learn to leverage the PI, consistently showing positive Δmax\Delta\_{\max} values in such cases. We find that effectively learning to use the PI is a significant contributing factor to transfer when only training the teacher.

#### Joint training α=0.5\mathbf{\alpha=0.5}.

We identify α=0.5\alpha=0.5 as the most robust configuration. It achieves the best performance in 7 out of 16 scenarios and effectively avoids the failure modes of the other variants, ranking as the worst performer only once. By balancing both teacher and student objectives, α=0.5\alpha=0.5 is able to mitigate the failure cases of independent training. We believe more granular tuning of α\alpha can likely lead to optimal performance in most settings, leaving this as future work.

#### Student only training α=𝟎\mathbf{\alpha=0}.

Here, low KL divergence is a strong predictor of success. In τ\tau-Bench, for example, setting α=0\alpha=0 with only tool calls consistently yields the best results, as the minimal distribution shift makes learning from the conditioned traces significantly easier. As KL divergence rises, performance generally drops, though high-utility PI can occasionally reverse this trend.

Further, we find that PI utility Δ\Delta can play a large role in the success of this variant. For instance, on Travel Planner with Qwen3-8B, α=0\alpha=0 underperforms significantly because the PI provides negative utility (Δ<0\Delta<0), actively degrading performance. Furthermore, as seen in [figure˜6](#S7.F6 "In 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models"), we find that Δmax≈0\Delta\_{\max}\approx 0 in these failure cases, confirming that the algorithm cannot extract value from the PI when the signal itself offers no initial advantage over the base model. On the other hand, we find that when the teacher-student KL is low, student-only training effectively transfers knowledge back onto the teacher. This allows the teacher to learn how to leverage PI even though it is not being directly trained. We observe this behavior in Qwen3-8B on Travel Planner when using only tool calls as PI. While the initial Δ<0\Delta<0, the low KL allows transfer to occur from the student to the teacher and enables the teacher to sample improved traces.

Takeaways: Analysis on Privileged Information

•

Student only training. Useful-low KL PI is easy to learn from using πS​(α=0){\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}}(\alpha=0). Higher KL or low utility PI leads to suboptimal performance.
•

Teacher only training. We find teacher-training, πT​(α=1){\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}}(\alpha=1), provides performance gains under two factors. (1) Ensuring that PI is useful or can be learned (see [figure˜6](#S7.F6 "In 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models")). (2) Avoiding teacher and student collapse πS≉πT{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}}\not\approx{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}}.
•

Joint training. We find joint training, πS+πT​(α=0.5){\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}}+{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}}(\alpha=0.5), to be the most stable mitigating failure cases of independent training. Never significantly working the worst. We recommend this approach when lacking multiple types of PI or sweeping values of α\alpha is unfeasible.

### 7.2 What Matters for OPSD

!(/html/2602.04942/assets/x8.png)

Figure 8: Performance and Stability Analysis of OPSD.
We compare held-out performance on τ\tau-Bench (top row) and Travel Planner (bottom row) across three base models and three PI types (colors). The scatter plots map final scores against the student-teacher KL divergence (DKL​(πS∥πT)D\_{\text{KL}}({\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}}\|{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}})), while the bar-charts display the PI utility (Δ\Delta). Key observations:
(1) Unlike π\pi-Distill, higher KL are not always detrimental, rather, information richness of PI is most important, finding (Tool Calls & Arguments) often performs best (e.g., all results on Travel Planner and Qwen3-4B on τ−\tau-Bench ).
(2) Excessive KL can override positive utility (note Qwen3-8B on τ\tau-Bench, where Δ>0\Delta>0 but the high KL degrades performance).
(3) R1-Distill-Llama-8B consistently struggles, which we attribute to either extreme KL divergence (τ\tau-Bench) or negative PI utility (TravelPlanner).

[Figure˜8](#S7.F8 "In 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models") similarly to [figure˜5](#S6.F5 "In 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models") displays performance on τ\tau-Bench retail and Travel Planner across PI types outlined in [section˜4.2](#S4.SS2.SSS0.Px1 "Types of Privileged Information. ‣ 4.2 Sources of Privileged Information ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models").

#### Findings.

Our main finding is that, contrary to π\pi-Distill, high-KL PI types do not necessarily indicate worse performance. Instead, the information content of the PI is the strongest predictor of successful transfer. For example, Tool Calls & Arguments, being the richest in information, consistently performs best for both Qwen3 models on Travel Planner, and also achieves the best performance for Qwen3-4B on τ\tau-Bench.

Interestingly, although we find that high-information PI is generally best suited for transfer, we observe exceptions. In particular, for Qwen3-8B on τ\tau-Bench, Tool Calls & Arguments performs the worst and exhibits the highest KL. In this case, we observe that Δmax\Delta\_{\max} is negative, indicating that the reverse-KL penalty can inhibit training. Additionally, regarding the failure on R1-Distill-Llama-8B, while this could be due to the SFT phase prior to OPSD, we find that on τ\tau-Bench the KL for both PI types is exceptionally high. While on Travel Planner, the PI utility, Δ\Delta is low or negative, possibly explaining the diminished results. Overall, we find that the strongest predictor of performance for OPSD is the information content of the PI. While using the richest form of PI can lead to the best results, careful consideration is required to ensure the KL does not become excessive and hinder training.

## 8 Ablation on β\beta

In this section, we analyze β\beta, the term controlling the regularization between the student and teacher, finding it important for achieving the best performance in 17/21 ablated configurations. We conduct extensive ablations on β\beta keeping all other parameters fixed, sweeping over β={0,0.1,0.25,0.5}\beta=\{0,0.1,0.25,0.5\}. We analyze *all* values of α\alpha and PI types for both π\pi-Distill and OPSD on τ\tau-Bench using both Qwen3 models, as well as the best-performing PI type for Travel Planner. All results are reported over three random seeds.

[Figure˜9](#S8.F9 "In 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models") shows a subset of the learning curves on held-out data for the best-performing PI type and α\alpha value in τ\tau-Bench Retail and Travel Planner, while [figure˜16](#A4.F16 "In D.3 Full 𝛽 Ablations ‣ Appendix D Additional Ablations ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models") provides the full set for τ\tau-Bench. For π\pi-Distill, we consistently find that β>0\beta>0 aids in obtaining the best performance, particularly when the teacher is being trained (α>0\alpha>0). While β\beta can be sensitive, with no single value being the best in all settings, setting β\beta higher than 0 generally allows for better or matching performance, with only four cases showing inferior results. For OPSD, we find that the value of β\beta is less important with our results on [section˜7](#S7 "7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models") showing that information granularity and student-teacher KL are more important factors.

!(/html/2602.04942/assets/x9.png)

Figure 9:  Evaluation performance throughout training for π\pi-Distill variants across varying KL penalties (β\beta). Runs that deteriorate significantly early are truncated for visual clarity. We observe that for settings involving teacher updates (α>0\alpha>0), a non-zero penalty (β>0\beta>0) is crucial for stabilizing training and achieving peak performance. Here error bars indicate standard errors. Discrepancies between plot and table values are addressed in App. [F](#A6 "Appendix F Plot and Table Discrepancies ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models")

## 9 Related Work

#### Latent Reasoning and Variational Perspectives.

Recent work increasingly frames LM reasoning as a latent-variable inference problem (hu2024amortizingintractableinferencelarge; sordoni2023jointpromptoptimizationstacked; luo2025languagemodelslearnverbal; li2025startselftaughtreasonertools).
Within this framework, the most popular approach is STaR (li2025startselftaughtreasonertools), which uses a hint when the model is unable to correctly answer the question and then uses SFT to fit onto the generated reasoning trace. This is similar to student-only training in π\pi-Distill (α=0\alpha=0), where, rather than a KL-regularized off-policy RL objective, SFT is used, and the teacher is not trained. Most similar to π\pi-Distill is the work by (zhou2025variationalreasoninglanguagemodels), which proposes variational reasoners. This approach can be seen as a modified version of STaR, where both teacher and student are being trained iteratively. The main difference between variational reasoning and our work is that they assume access to oracle answers and perform an iterative version of variational EM using separate parameters for the teacher and student. In comparison, we simplify the training objective by allowing the teacher and student to share parameters and do not assume access to ground truth answers. We compare against a similar setup with minor modifications due to not having access to ground truth solutions in App. [A.2](#A1.SS2 "A.2 Failures of Variational EM. ‣ Appendix A Connections To Variational EM ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models").

#### Self-Bootstrapping, Privileged Signals, and Guided Exploration.

Complementary work focuses on using privileged or auxiliary signals to enable learning in hard regimes. chen2025nudgingboundariesllmreasoning injects self-generated high-level hints into online RL to overcome zero-reward exploration barriers, while qu2026popelearningreasonhard uses privileged oracle solutions as structured on-policy exploration signals for hard reasoning tasks. Both these algorithms can be viewed as training the teacher to use PI with implicit transfer via parameter sharing. Additionally, other lines of work explore distilling certain skills/behaviors in the model via contextualized sampling (didolkar2025metacognitivereuseturningrecurring; didolkar2024metacognitivecapabilitiesllmsexploration; yang-etal-2025-distilling; qu2025rladtrainingllmsdiscover). We believe our proposed methods could be used to distill desired skills/behaviors into model weights in a more effective manner compared to traditional SFT.

#### Context Distillation.

Both our proposed methods can be seen as a form of context distillation. This growing line of work studies how to train models to internalize context-dependent reasoning so that the benefits of rich contexts (e.g., instructions, demonstrations, auxiliary computations) are distilled into the model’s weights (caccia2025trainingplugnplayknowledgemodules; snell2022learningdistillingcontext; huang2022incontextlearningdistillationtransferring). Other works analyze this problem by performing SFT answers they give when having access to context snell2022learningdistillingcontext. Moreover, other works also try to encapsulate such knowledge into adapters that can be plugged and played at test time (caccia2025trainingplugnplayknowledgemodules).

## 10 Limitations & Future Work

While we show that both π\pi-Distill and OPSD are effective algorithms when CoT is not available, all our PI is transformed from frontier model traces. Exploring how to efficiently obtain useful PI when neither frontier model actions nor ground-truth answers are available is an interesting direction. Concurrent work by hubotter2026reinforcementlearningselfdistillation explores this direction by mining PI through self-reflection. Additionally, our experiments are limited to models with ≤8\leq 8B parameters. Further scaling of both π\pi-Distill and OPSD can shine light on additional important factors to enable these algorithms. Moreover, our analysis in [section˜7](#S7 "7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models") is limited to observational studies, where we do not systematically control for all variables, but rather observe properties of different configurations. A study that systematically controls for these factors could help in mitigating failure cases and obtaining the most useful PI.

## 11 Conclusion

We introduce π\pi-Distill and On-Policy Self-Distillation, two algorithms that leverage PI at training time to produce an improved policy even when that information is lacking at test time. By grounding our work in the distillation of frontier models where CoT reasoning is inaccessible, we demonstrate that the absence of these proprietary traces is not a limiting factor. In fact, our approach remains effective even when compared to industry standards that assume access to full CoT reasoning. Through evaluations across two training domains and eight OOD datasets, we find that π\pi-Distill and OPSD are highly effective in all explored settings. Finally, we characterize the factors driving the distillation of training-time PI, showing that its success can often be predicted using only base-model statistics.

## 12 Acknowledgments

We thank and credit Siddarth Venkatraman for the idea to view learning from PI from a variational perspective, which led to the final version of π\pi-Distill. We thank Vedant Shah for helpful discussion and proof reading initial drafts of the paper. Additionally, we thank Michael Noukhovitch for valuable feedback that substantially aided the final draft of this work. We also thank Alexandre Piché for their valuable discussions.

EP acknowledges the support of the NSERC PGS-D grant, the
Bourse en intelligence artificielle provided by Université
de Montréal and the MITACS Accelerate grant. DV acknowledges the support of the MITACS Accelerate grant. LC recognizes the support of NSERC, the Canada CIFAR
AI Chair Program, the Canada First Research Excellence
Fund and IVADO.

## 13 Author Contributions

Emiliano Penaloza proposed the idea of leveraging PI, led the project, implemented all algorithms and the RL environments, conducted most experiments, and was the primary writer of the paper.

Dheeraj Vattikonda was a core part of the project, with the final version heavily improved by their contributions. Specifically, they provided substantial support on all experiments over the course of the project, implemented and conducted experiments on OOD environments, and provided heavy contributions to paper writing and figures.

Nicolas Gontier helped in advising the project and providing detailed feedback that substantially improved experimental rigor.

Alexandre Lacoste advised the project, providing valuable insights to the goal and purpose of leveraging PI, helped improve experimental rigor and analysis, and assisted in paper writing.

Laurent Charlin advised the project, helping influence the framing and perspective of the project, and provided help with writing, with a substantial portion of the introduction written by them.

Massimo Caccia was a core member of the project. They helped shape the research direction, proposed the on-policy self-distillation component, guided the design of extensive ablation studies, and provided detailed technical guidance across all stages. They also contributed substantially to writing and figures. (They are also responsible for the green outline in the abstract when it should have been purple.)

## References

## Appendix A Connections To Variational EM

In this Appendix, we derive Jπ−DistillJ\_{\pi-\text{Distill}} using a variational perspective. Specifically, we show how Jπ−DistillJ\_{\pi-\text{Distill}} can be seen as a joint EM algorithm where E and M are done simultaneously.

Traditional variational-EM aims to fit a parameterized distribution πS{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}} onto a target posterior π∗\pi^{\*}. An assumption this framework makes is that using πS{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}} to approximate π∗\pi^{\*} may be suboptimal, rather, leveraging an approximate posterior πθT​(𝐨∣s,𝐈){\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})} conditioned on additional information can make the approximation easier. This is true when πT{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}} can sample from high-reward modes that πS{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}} cannot, therefore making optimization via RL impossible for πS{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}}. We visualize the procedure in [Figure˜10](#A1.F10 "In A.1 Variational EM ‣ Appendix A Connections To Variational EM ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models")

### A.1 Variational EM

!(/html/2602.04942/assets/x10.png)

Figure 10: Illustration of the Variational EM procedure for policy optimization. The initial student policy, πθS\pi\_{\theta}^{S}, lacks support over trajectories with positive rewards (R​(𝐨,𝐬)>0R(\mathbf{o},\mathbf{s})>0), preventing direct improvement. To address this: (1) the teacher policy πθT\pi\_{\theta}^{T} is optimized via JTeacher​(θ)J\_{\text{Teacher}}(\theta) to sample successful traces and approximate the optimal policy π∗\pi^{\*}; (2) the student policy is then updated via JSFT​(θ)J\_{\text{SFT}}(\theta) to distill the knowledge from the teacher. While this two-step procedure is principled, it is computationally inefficient due to the requirement of maintaining dual parameter sets and distinct training phases. In contrast, π\pi-Distill simplifies this pipeline into a single-phase process, providing superior performance with reduced complexity.

#### E-step JTeacherJ\_{\text{Teacher}}.

We first define the target posterior we want to fit, π∗\pi^{\*}, as a reward-tilted posterior relative to the reference policy πref\pi\_{\text{ref}}. For a given state 𝐬\mathbf{s}, we define:

|  |  |  |  |
| --- | --- | --- | --- |
|  | π∗​(𝐨∣𝐬)=πref​(𝐨∣𝐬)​exp⁡(R​(𝐨,𝐬))Z\pi^{\*}(\mathbf{o}\mid\mathbf{s})=\frac{\pi\_{\text{ref}}(\mathbf{o}\mid\mathbf{s})\exp(R(\mathbf{o},\mathbf{s}))}{Z} |  | (6) |

where Z=∑𝐨′πref​(𝐨′∣𝐬0)​exp⁡(R​(𝐨′,𝐬))Z=\sum\_{\mathbf{o^{\prime}}}\pi\_{\text{ref}}(\mathbf{o^{\prime}}\mid\mathbf{s}\_{0})\exp(R(\mathbf{o^{\prime}},\mathbf{s})) is the partition function and 𝐨=(𝐳,𝐚)\mathbf{o}=(\mathbf{z},\mathbf{a}) consists of internal reasoning tokens 𝐳\mathbf{z} and action tokens 𝐚\mathbf{a}.

The partition function makes this distribution intractable, but it can be approximated using a variational posterior πθT​(𝐨∣s,𝐈){\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})} conditioned on privileged information 𝐈\mathbf{I}. We optimize parameters θ\theta by minimizing the reverse KL between the variational distribution and the target policy:

|  |  |  |  |
| --- | --- | --- | --- |
|  | JTeacher​(θ)\displaystyle J\_{\text{Teacher}}(\theta) | =−DKL(πθT(𝐨∣s,𝐈)∥π∗(⋅∣𝐬))\displaystyle=-D\_{\text{KL}}\!\left({\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})}\parallel\pi^{\*}(\cdot\mid\mathbf{s})\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =−𝔼𝐨∼πθT​(𝐨∣s,𝐈)𝐬∼P​[log⁡πθT​(𝐨∣s,𝐈)π∗​(𝐨∣𝐬)]\displaystyle=-\mathbb{E}\_{\begin{subarray}{c}\mathbf{o}\sim{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})}\\ \mathbf{s}\sim P\end{subarray}}\left[\log\frac{{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})}}{\pi^{\*}(\mathbf{o}\mid\mathbf{s})}\right] |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =𝔼𝐨∼πθT​(𝐨∣s,𝐈)𝐬∼P​[log⁡π∗​(𝐨∣𝐬)−log⁡πθT​(𝐨∣s,𝐈)]\displaystyle=\mathbb{E}\_{\begin{subarray}{c}\mathbf{o}\sim{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})}\\ \mathbf{s}\sim P\end{subarray}}\left[\log\pi^{\*}(\mathbf{o}\mid\mathbf{s})-\log{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})}\right] |  | (7) |

Substituting the definition of π∗\pi^{\*} into the equation:

|  |  |  |  |
| --- | --- | --- | --- |
|  | JTeacher​(θ)\displaystyle J\_{\text{Teacher}}(\theta) | =𝔼𝐨∼πθT​(𝐨∣s,𝐈)𝐬∼P​[log⁡(πref​(𝐨∣𝐬)​exp⁡(R​(𝐨,𝐬))Z)−log⁡πθT​(𝐨∣s,𝐈)]\displaystyle=\mathbb{E}\_{\begin{subarray}{c}\mathbf{o}\sim{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})}\\ \mathbf{s}\sim P\end{subarray}}\left[\log\left(\frac{\pi\_{\text{ref}}(\mathbf{o}\mid\mathbf{s})\exp(R(\mathbf{o},\mathbf{s}))}{Z}\right)-\log{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})}\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =𝔼𝐨∼πθT​(𝐨∣s,𝐈)𝐬∼P​[R​(𝐨,𝐬)+log⁡πref​(𝐨∣𝐬)−log⁡πθT​(𝐨∣s,𝐈)−log⁡Z]\displaystyle=\mathbb{E}\_{\begin{subarray}{c}\mathbf{o}\sim{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})}\\ \mathbf{s}\sim P\end{subarray}}\left[R(\mathbf{o},\mathbf{s})+\log\pi\_{\text{ref}}(\mathbf{o}\mid\mathbf{s})-\log{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})}-\log Z\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =𝔼𝐨∼πθT​(𝐨∣s,𝐈)𝐬∼P​[R​(𝐨,𝐬)−(log⁡πθT​(𝐨∣s,𝐈)−log⁡πref​(𝐨∣𝐬))]−log⁡Z\displaystyle=\mathbb{E}\_{\begin{subarray}{c}\mathbf{o}\sim{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})}\\ \mathbf{s}\sim P\end{subarray}}\left[R(\mathbf{o},\mathbf{s})-\left(\log{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})}-\log\pi\_{\text{ref}}(\mathbf{o}\mid\mathbf{s})\right)\right]-\log Z |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ∝𝔼𝐨∼πθT​(𝐨∣s,𝐈)𝐬∼P[R(𝐨,𝐬)]−DKL(πθT(𝐨∣s,𝐈)∥πref(⋅∣𝐬))\displaystyle\propto\mathbb{E}\_{\begin{subarray}{c}\mathbf{o}\sim{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})}\\ \mathbf{s}\sim P\end{subarray}}[R(\mathbf{o},\mathbf{s})]-D\_{\text{KL}}\!\left({\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})}\parallel\pi\_{\text{ref}}(\cdot\mid\mathbf{s})\right) |  | (8) |

Since log⁡Z\log Z depends only on the context 𝐬\mathbf{s}, it is constant with respect to θ\theta and can be omitted.

#### M-step JSFTJ\_{\text{SFT}}.

If π∗\pi^{\*} were a tractable distribution, we could directly fit the target policy by minimizing the forward KL divergence, DKL​(π∗∥πθS)D\_{\text{KL}}(\pi^{\*}\parallel{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}}), corresponding to standard SFT. Since π∗\pi^{\*} is intractable, we instead substitute it with our learned approximation πθT{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}}, yielding the following objective:

|  |  |  |  |
| --- | --- | --- | --- |
|  | JSFT​(θ)=𝔼𝐨∼πθT​(𝐨∣s,𝐈)𝐬∼P​[log⁡πθS​(𝐨|s)].\displaystyle J\_{\text{SFT}}(\theta)=\mathbb{E}\_{\begin{subarray}{c}\mathbf{o}\sim{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})}\\ \mathbf{s}\sim P\end{subarray}}\left[\log{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}(\mathbf{o}|s)}\right]. |  | (9) |

Traditionally, one would either first fit πθT{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}} to convergence and use it to fit πθS{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}}, or alternate between training the teacher and student. Regardless, this process requires two separate models and can be difficult to optimize. Our implementation is a modified version of zhou2025variationalreasoninglanguagemodels, where the modifications are specific to our multi-turn setting, where no ground truth is provided.

### A.2 Failures of Variational EM.

!(/html/2602.04942/assets/x11.png)

Figure 11: Performance of Sequential EM variants on τ\tau-Bench Retail (Qwen3-8B). We compare these methods against SFT baselines. While standard EM RFT (light blue) fails to match SFT w/ CoT (yellow), replacing RFT with off-policy RL (EM Off-Pol, purple) successfully allows for EM to outperform SFT w CoT. The strongest performance comes from JTeacher​(θ)J\_{\text{Teacher}}(\theta) (dark blue), demonstrating that parameter sharing between the teacher and student yields the most effective transfer. Shaded regions indicate standard error across 3 seeds.

Our initial efforts focused on making variational EM work, similar in spirit to zhou2025variationalreasoninglanguagemodels. We first document initial ideas that were unsuccessful, then we discuss the best version of EM we could find, being a sequential implementation. Finally, we discuss how these experiments ultimately motivated the design decisions for π\pi-Distill.

#### Initial Experiments

First, we find that setting the reference policy πref\pi\_{\text{ref}} as a copy of the current student πθS{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}} is crucial for learning. In contrast, using the fixed base model πbase\pi\_{\text{base}} as a prior significantly hinders performance (see App. [D.2](#A4.SS2 "D.2 Reference Model Ablation ‣ Appendix D Additional Ablations ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models")).

Building on this insight, we initially explored an offline variational EM method similar to zhou2025variationalreasoninglanguagemodels. Here, the goal is learning a CoT sampler as the variational posterior conditioned on expert traces. We do this with the following objective:

|  |  |  |  |
| --- | --- | --- | --- |
|  | JOffline-EM​(θ)\displaystyle J\_{\text{Offline-EM}}(\theta) | =𝔼𝐬∼𝒟​[log​∑𝐳πθS​(𝐚,𝐳∣𝐬)​exp⁡(R​(𝐨,𝐬))]\displaystyle=\mathbb{E}\_{\mathbf{s}\sim\mathcal{D}}\left[\log\sum\_{\mathbf{z}}{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}}(\mathbf{a},\mathbf{z}\mid\mathbf{s})\exp(R(\mathbf{o},\mathbf{s}))\right] |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ∝𝔼𝐚,𝐬∼𝒟,𝐳∼πθT[R(𝐨,𝐬)+logπθS(𝐚∣𝐳,𝐬)]−DKL(πθT(𝐳∣𝐬)∥πθS(𝐳∣𝐬))\displaystyle\propto\mathbb{E}\_{\begin{subarray}{c}\mathbf{a,s}\sim\mathcal{D},\mathbf{z}\sim{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}}\end{subarray}}\left[R(\mathbf{o},\mathbf{s})+\log{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}}(\mathbf{a}\mid\mathbf{z},\mathbf{s})\right]-D\_{\text{KL}}\!\left({\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}}(\mathbf{z}\mid\mathbf{s})\parallel{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}}(\mathbf{z}\mid\mathbf{s})\right) |  | (10) |

Notice here how the states are fixed and given in the offline dataset 𝒟\mathcal{D}. Furthermore, this setup necessitates substantial hyperparameter tuning. Beyond the standard penalty β\beta, the algorithm requires balancing the ratio of teacher-to-student updates, managing separate optimizer states, and calibrating the dataset size sampled from the teacher. Ultimately, we find this approach hard to implement and fails in our multi-turn setting. We attribute the success of zhou2025variationalreasoninglanguagemodels to their access to oracle answers; in our context, we find that fitting a variational posterior to suboptimal frontier model trajectories can even degrade performance.

Our following efforts focused on training the teacher online using [Section˜A.1](#A1.Ex4 "E-step 𝐽_\"Teacher\". ‣ A.1 Variational EM ‣ Appendix A Connections To Variational EM ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models"). We try this with alternating loops between student and teacher training, but find this to be ineffective. As we found it hard to avoid the student and teacher collapsing onto each other using this setup. Ultimately, we found a sequential version of EM worked best, though it still requires specific adjustments to achieve optimal performance.

#### Sequential EM.

We turn to Sequential EM after ruling out offline and alternating approaches. However, we find standard implementations remain unreliable for complex agentic tasks. Here, we analyze the limitations of standard EM and how they ultimately motivated π\pi-Distill.

For this analysis, we use τ\tau-Bench Retail with Qwen3-8B using self-generated hints to train the teacher as this configuration performs best (see § [7](#S7 "7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models")). We compare sequential EM, where we first train the teacher and then perform SFT on successful trajectories (Rejection Fine-tuning or RFT), against SFT with and without CoT. We note that here it would be sufficient to outperform SFT without CoT as we assume no access to the internal CoT of the frontier model, thus in practice, sequential EM could replace the traditional SFT phase before RL.

We report results in [Figure˜11](#A1.F11 "In A.2 Failures of Variational EM. ‣ Appendix A Connections To Variational EM ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models"). First, we observe that standard Sequential EM outperforms naive SFT but lags behind training with CoT. We attribute the performance gap to the lack of negative feedback on failed trajectories. It only mimics the teacher. To address this, we replaced the RFT step with clipped off-policy RL. We find that this change enables sequential EM to outperform SFT with CoT.

Most surprisingly, in these experiments, we find that simply optimizing JTeacher​(θ)J\_{\text{Teacher}}(\theta) substantially improves the student πθS{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}} when both teacher and student share parameters. This drastically outperforms all other baselines.

To summarize, our two main findings are that parameter sharing enables substantial transfer from teacher to student and the importance of leveraging negative traces via off-policy RL when training the student. We use these two findings to instantiate π\pi-Distill while also simplifying the process from two training steps to a single one.

## Appendix B On-Policy Self-Distillation JOPSDJ\_{\text{OPSD}}

Contrary to the prior objectives, OPSD, rather, than trying to approximate π∗\pi^{\*} can be framed as the reverse-Kl between π\pi and the target conditioned policy πT⁣∗\pi^{T\*}, being defined as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | π∗​(𝐨∣𝐬,𝐈)=πref(𝐨∣𝐬,𝐈)exp(R(𝐨,𝐬))βZh\pi^{\*}(\mathbf{o}\mid\mathbf{s},\mathbf{I})=\frac{\pi\_{\text{ref}}(\mathbf{o}\mid\mathbf{s},\mathbf{I})\exp(\frac{R(\mathbf{o},\mathbf{s}))}{\beta}}{Z^{h}} |  | (11) |

where Zh=∑𝐨′πrefT​(𝐨′∣𝐬,𝐈)​exp⁡(R​(𝐨′,𝐬))Z^{h}=\sum\_{\mathbf{o^{\prime}}}\pi\_{\text{ref}}^{T}(\mathbf{o}^{\prime}\mid\mathbf{s},\mathbf{I})\exp(R(\mathbf{o}^{\prime},\mathbf{s})) is the partition function.

One can then approximate this distribution directly via reverse KL, giving the following objective:

To do this, we minimize the reverse KL between the student policy πθS{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}} and the optimal privileged target distribution π∗\pi^{\*}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | DKL​(πθS∥π∗)\displaystyle D\_{\text{KL}}({\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}}\parallel\pi^{\*}) | =𝔼𝐨∼πθS​(𝐨|s)𝐬∼P​[log⁡πθS​(𝐨∣𝐬)π∗​(𝐨∣𝐬,𝐈)]\displaystyle=\mathbb{E}\_{\begin{subarray}{c}\mathbf{o}\sim{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}(\mathbf{o}|s)}\\ \mathbf{s}\sim P\end{subarray}}\left[\log\frac{{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}}(\mathbf{o}\mid\mathbf{s})}{\pi^{\*}(\mathbf{o}\mid\mathbf{s},\mathbf{I})}\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =𝔼𝐨∼πθS​(𝐨|s)𝐬∼P​[log⁡πθS​(𝐨∣𝐬)−log⁡(πref​(𝐨∣𝐬,𝐈)​exp⁡(R​(𝐨,𝐬)/β)Zh)]\displaystyle=\mathbb{E}\_{\begin{subarray}{c}\mathbf{o}\sim{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}(\mathbf{o}|s)}\\ \mathbf{s}\sim P\end{subarray}}\left[\log{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}}(\mathbf{o}\mid\mathbf{s})-\log\left(\frac{\pi\_{\text{ref}}(\mathbf{o}\mid\mathbf{s},\mathbf{I})\exp(R(\mathbf{o},\mathbf{s})/\beta)}{Z^{h}}\right)\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =𝔼𝐨∼πθS​(𝐨|s)𝐬∼P​[log⁡πθS​(𝐨∣𝐬)−log⁡πref​(𝐨∣𝐬,𝐈)−R​(𝐨,𝐬)β+log⁡Zh]\displaystyle=\mathbb{E}\_{\begin{subarray}{c}\mathbf{o}\sim{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}(\mathbf{o}|s)}\\ \mathbf{s}\sim P\end{subarray}}\left[\log{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}}(\mathbf{o}\mid\mathbf{s})-\log\pi\_{\text{ref}}(\mathbf{o}\mid\mathbf{s},\mathbf{I})-\frac{R(\mathbf{o},\mathbf{s})}{\beta}+\log Z^{h}\right] |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =𝔼𝐨∼πθS​(𝐨|s)𝐬∼P​[−R​(𝐨,𝐬)β+log⁡πθS​(𝐨∣𝐬)πref​(𝐨∣𝐬,𝐈)]+log⁡Zh\displaystyle=\mathbb{E}\_{\begin{subarray}{c}\mathbf{o}\sim{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}(\mathbf{o}|s)}\\ \mathbf{s}\sim P\end{subarray}}\left[-\frac{R(\mathbf{o},\mathbf{s})}{\beta}+\log\frac{{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}}(\mathbf{o}\mid\mathbf{s})}{\pi\_{\text{ref}}(\mathbf{o}\mid\mathbf{s},\mathbf{I})}\right]+\log Z^{h} |  | (12) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =−1β𝔼𝐨∼πθS​(𝐨|s)𝐬∼P[R(𝐨,𝐬)]+DKL(πθS(⋅∣𝐬)∥πref(⋅∣𝐬,𝐈))+logZh.\displaystyle=-\frac{1}{\beta}\mathbb{E}\_{\begin{subarray}{c}\mathbf{o}\sim{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}(\mathbf{o}|s)}\\ \mathbf{s}\sim P\end{subarray}}\left[R(\mathbf{o},\mathbf{s})\right]+D\_{\text{KL}}\!\left({\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}}(\cdot\mid\mathbf{s})\parallel\pi\_{\text{ref}}(\cdot\mid\mathbf{s},\mathbf{I})\right)+\log Z^{h}. |  | (13) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ∝−𝔼𝐨∼πθS​(𝐨|s)𝐬∼P[R(𝐨,𝐬)]+βDKL(πθS(⋅∣𝐬)∥πref(⋅∣𝐬,𝐈))\displaystyle\propto-\mathbb{E}\_{\begin{subarray}{c}\mathbf{o}\sim{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}(\mathbf{o}|s)}\\ \mathbf{s}\sim P\end{subarray}}\left[R(\mathbf{o},\mathbf{s})\right]+\beta D\_{\text{KL}}\!\left({\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}\_{\theta}}(\cdot\mid\mathbf{s})\parallel\pi\_{\text{ref}}(\cdot\mid\mathbf{s},\mathbf{I})\right) |  | (14) |

To align with π\pi-Distill and the according to the results in App. [D.2](#A4.SS2 "D.2 Reference Model Ablation ‣ Appendix D Additional Ablations ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models"), we set πref=πθT​(𝐨∣s,𝐈)\pi\_{\text{ref}}={\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}\_{\theta}(\mathbf{o}\mid s,\mathbf{I})}

#### Concurrent On-Policy Distillation Methods.

Concurrent work proposes closely related on-policy self-distillation frameworks. Recent methods train student models on their own trajectories while using privileged or conditioned variants of the same model as teachers, enabling on-policy transfer without off-policy distribution shift (zhao2026selfdistilledreasoneronpolicyselfdistillation; shenfeld2026selfdistillationenablescontinuallearning). These approaches are conceptually aligned with our OCPD formulation, but target supervised reasoning or continual learning settings, whereas our focus is agentic decision-making with training-time privileged information.

## Appendix C Detailed Algorithms

Here we outline the detailed algorithms for both π\pi-Distill and OPSD.

Algorithm 1  π\pi-Distill: Privileged Information Distillation for Language Models

1:  Input: Dataset 𝒟={(s,𝐈)}\mathcal{D}=\{(s,\mathbf{I})\} where 𝐈\mathbf{I} is privileged information, Initial Policy πθ\pi\_{\theta}, Reference πref\pi\_{\text{ref}}, α\alpha, β\beta, ϵ\epsilon, Learning rate η\eta

2:  Initialize: ϕ←θ\phi\leftarrow\theta {Parameters are shared between Teacher and Student}

3:  while not converged do

4:   Sample batch B={(si,𝐈i)}i=1N∼𝒟B=\{(s\_{i},\mathbf{I}\_{i})\}\_{i=1}^{N}\sim\mathcal{D}

5:   // Step 1: Teacher Rollout (with Privileged Information)

6:   for each (s,𝐈)∈B(s,\mathbf{I})\in B do

7:    Sample KK trajectories {𝐨1,…,𝐨K}∼πT(⋅∣s,𝐈)\{\mathbf{o}\_{1},\dots,\mathbf{o}\_{K}\}\sim\pi^{T}(\cdot\mid s,\mathbf{I})

8:    Compute rewards R(𝐨k,s)=Renv(𝐨k,s)−βDKL[πT(⋅∣s,𝐈)∥πref(⋅∣s,𝐈)]R(\mathbf{o}\_{k},s)=R\_{\text{env}}(\mathbf{o}\_{k},s)-\beta D\_{\text{KL}}[\pi^{T}(\cdot\mid s,\mathbf{I})\|\pi\_{\text{ref}}(\cdot\mid s,\mathbf{I})]

9:   end for

10:   // Step 2: Compute Group-Centered Advantages

11:   for each k∈{1,…,K}k\in\{1,\dots,K\} do

12:    R¯=1K​∑j=1KR​(𝐨j,s)\bar{R}=\frac{1}{K}\sum\_{j=1}^{K}R(\mathbf{o}\_{j},s)

13:    Ak=R​(𝐨k,s)−R¯A\_{k}=R(\mathbf{o}\_{k},s)-\bar{R}

14:   end for

15:   // Step 3: Compute Objectives

16:   Teacher Objective (GRPO):

17:   JTeacher=1K​∑k=1Kmin⁡(ρkteacher​Ak,clip​(ρkteacher,1−ϵ,1+ϵ)​Ak)J\_{\text{Teacher}}=\frac{1}{K}\sum\_{k=1}^{K}\min\left(\rho\_{k}^{\text{teacher}}A\_{k},\text{clip}(\rho\_{k}^{\text{teacher}},1-\epsilon,1+\epsilon)A\_{k}\right)

18:   Student Objective (Off-Policy GRPO):

19:   Compute IS weights: ρk=πS​(𝐨k∣𝐬)πT​(𝐨k∣𝐬,𝐈)\rho\_{k}=\frac{\pi^{S}(\mathbf{o}\_{k}\mid\mathbf{s})}{\pi^{T}(\mathbf{o}\_{k}\mid\mathbf{s},\mathbf{I})} {Note: Student input is ss only}

20:   JStudent=1K​∑k=1Kmin⁡(ρk​Ak,clip​(ρk,1−ϵ,1+ϵ)​Ak)J\_{\text{Student}}=\frac{1}{K}\sum\_{k=1}^{K}\min\left(\rho\_{k}A\_{k},\text{clip}(\rho\_{k},1-\epsilon,1+\epsilon)A\_{k}\right)

21:   // Step 4: Joint Update

22:   Jπ​-Distill​(θ)=α​JTeacher+(1−α)​JStudentJ\_{\pi\text{-Distill}}(\theta)=\alpha J\_{\text{Teacher}}+(1-\alpha)J\_{\text{Student}}

23:   θ←θ+η​∇θJπ​-Distill​(θ)\theta\leftarrow\theta+\eta\nabla\_{\theta}J\_{\pi\text{-Distill}}(\theta)

24:  end while

Algorithm 2  On-Policy Self-Distillation

1:  Input: Dataset 𝒟={(s,𝐈)}\mathcal{D}=\{(s,\mathbf{I})\} where 𝐈\mathbf{I} is privileged information, Initial Policy πθ\pi\_{\theta}, β\beta, ϵ\epsilon, Learning rate η\eta

2:  Initialize: θ\theta {Parameters for the Student Policy}

3:  while not converged do

4:   Sample batch B={(si,𝐈i)}i=1N∼𝒟B=\{(s\_{i},\mathbf{I}\_{i})\}\_{i=1}^{N}\sim\mathcal{D}

5:   // Step 1: Student Rollout (On-Policy sampling)

6:   for each (s,𝐈)∈B(s,\mathbf{I})\in B do

7:    Sample KK trajectories {𝐨1,…,𝐨K}∼πS(⋅∣s)\{\mathbf{o}\_{1},\dots,\mathbf{o}\_{K}\}\sim\pi^{S}(\cdot\mid s)

8:    {Reward Computation with Reverse KL}

9:    Compute rewards R(𝐨k,s)=Renv(𝐨k,s)−βDKL[πS(⋅∣s)∥πT(⋅∣s,𝐈)]R(\mathbf{o}\_{k},s)=R\_{\text{env}}(\mathbf{o}\_{k},s)-\beta D\_{\text{KL}}\left[\pi^{S}(\cdot\mid s)\|\pi^{T}(\cdot\mid s,\mathbf{I})\right]

10:   end for

11:   // Step 2: Compute Group-Centered Advantages

12:   for each sample i∈{1,…,N}i\in\{1,\dots,N\} do

13:    R¯i=1K​∑j=1KR​(𝐨i,j,si)\bar{R}\_{i}=\frac{1}{K}\sum\_{j=1}^{K}R(\mathbf{o}\_{i,j},s\_{i})

14:    for each k∈{1,…,K}k\in\{1,\dots,K\} do

15:     Ai,k=R​(𝐨i,k,si)−R¯iA\_{i,k}=R(\mathbf{o}\_{i,k},s\_{i})-\bar{R}\_{i}

16:    end for

17:   end for

18:   // Step 3: Objective (Off-Policy GRPO)

19:   Compute IS weights: ρi,k=πθS​(𝐨i,k∣si)πoldS​(𝐨i,k∣si)\rho\_{i,k}=\frac{\pi^{S}\_{\theta}(\mathbf{o}\_{i,k}\mid s\_{i})}{\pi^{S}\_{\text{old}}(\mathbf{o}\_{i,k}\mid s\_{i})}

20:   J​(θ)=1K​∑k=1Kmin⁡(ρi,k​Ai,k,clip​(ρi,k,1−ϵ,1+ϵ)​Ai,k)J(\theta)=\frac{1}{K}\sum\_{k=1}^{K}\min\left(\rho\_{i,k}A\_{i,k},\text{clip}(\rho\_{i,k},1-\epsilon,1+\epsilon)A\_{i,k}\right)

21:   // Step 4: Policy Update

22:   θ←θ+η​∇θJ​(θ)\theta\leftarrow\theta+\eta\nabla\_{\theta}J(\theta)

23:  end while

## Appendix D Additional Ablations

### D.1 Privilege Information leakage Ablation

We measure privileged-information leakage with a simple keyword detector applied to each assistant message, using the keyword list [“privileged information”, “privileged info”, “priv info”, “secret information”, “secret info”, “correct tool calls”, “secret”, “privileged”, “hint”, “hints”]. For each occurrence, we assign a penalty of −0.1-0.1 and accumulate it over the trajectory, and we mark a trajectory as leaking if any of these appear. Fig [12](#A4.F12 "Figure 12 ‣ D.1 Privilege Information leakage Ablation ‣ Appendix D Additional Ablations ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models") reports results at β=0.25\beta=0.25 and shows that enabling this leakage penalty does not change task performance in any noticeable way, since the learning curves with and without the penalty overlap closely. Fig [13](#A4.F13 "Figure 13 ‣ D.1 Privilege Information leakage Ablation ‣ Appendix D Additional Ablations ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models") shows the corresponding leakage rate during training, and it rises as training progresses across all modes. The pattern with tool-calls showing the highest leakage, tool calls and arguments increasing more moderately, and self-generated hints starting lower but still increasing over time.

!(/html/2602.04942/assets/x12.png)

Figure 12: Performance with and without leakage penalty for πT​(α=1)\pi^{\text{T}}(\alpha=1). We find that although the penalty does reduce the leakage of the privileged information (see [Figure˜13](#A4.F13 "In D.1 Privilege Information leakage Ablation ‣ Appendix D Additional Ablations ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models")) it does not affect performance.

!()

Figure 13: Proportion of traces leaking privileged information as training progresses, we see that regardless PI type, the leakage increases with more gradient steps, finding using a leakage penalty reduces this proportion, but not substantially.

#### Test-time leakage.

As an additional analysis, we analyze whether leakage increases as models are trained with privileged data when evaluating πS{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}}. [Figure˜14](#A4.F14 "In Test-time leakage. ‣ D.1 Privilege Information leakage Ablation ‣ Appendix D Additional Ablations ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models") outlines this for Qwen3-8B for 300 gradient steps. We empirically validate that leakage does not significantly affect or increase when evaluating using πS{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}} as the presence of leaked words does not meaningfully increase as training continues.

!(/html/2602.04942/assets/x14.png)

Figure 14: Proportion of traces leaking privileged information when evaluating πS{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}} as training progresses, we see that there is no increase in leakage when evaluated as training goes on.

### D.2 Reference Model Ablation

We ablate the choice of reference policy used in the KL term by comparing a fixed base reference, πbase\pi\_{\text{base}}, against using the student itself as the reference, πθ\pi\_{\theta} (with stop-gradient on the reference branch). Fig.[15](#A4.F15 "Figure 15 ‣ D.2 Reference Model Ablation ‣ Appendix D Additional Ablations ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models") shows a clear difference in training stability. When πbase\pi\_{\text{base}} is used as the reference, the performance degrades over training and can collapse most clearly in the πh\pi^{h} setting (α=1\alpha=1), where the policy is pushed to move far from the base while still being penalized for that same deviation. In contrast, using πθ\pi\_{\theta} as the reference (orange) yields stable learning across α∈{0,0.5,1}\alpha\in\{0,0.5,1\}, since the KL regularizer stays aligned with the current student distribution. Practically, this choice is also cheaper because using πθ\pi\_{\theta} as the reference avoids maintaining a separate frozen reference model on the GPU, reducing memory and compute overhead.

!(/html/2602.04942/assets/x15.png)

Figure 15: Ablation between using reverse-Kl between the teacher and a prior. We ablate over two possible priors, πbase\pi\_{\text{base}} and πθS\pi^{\text{S}}\_{\theta}. We find that using πθS\pi^{\text{S}}\_{\theta} to be highly important in obtaining best performance.

### D.3 Full β\beta Ablations

[Figure˜16](#A4.F16 "In D.3 Full 𝛽 Ablations ‣ Appendix D Additional Ablations ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models") outlines the full suite of β\beta ablations for π\pi-Distill variants and self-distillation for Qwen3-4/8B. We sweep over β∈{0.1,0.25,0.5}\beta\in\{0.1,0.25,0.5\}.

!(/html/2602.04942/assets/x16.png)

(a) Qwen3-8B

!(/html/2602.04942/assets/x17.png)

(b)Qwen3-4B

Figure 16: Full sweep for β\beta over variants of π\pi-Distill. All experiments are given a 600 gradient step budget, where here we cut off experiment types early if they crash or do not continue learning. Consistent performance gains are seen for β>0\beta>0 across both 8B and 4B scales. We find that β>0\beta>0 is important in 14/18 cases. Specifically, we find it most important when training πT{\color[rgb]{0.390625,0.2578125,0.828125}\definecolor[named]{pgfstrokecolor}{rgb}{0.390625,0.2578125,0.828125}\pi^{\mathrm{T}}} and less important with student-only training πS{\color[rgb]{0.90234375,0.59765625,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.90234375,0.59765625,0}\pi^{\text{S}}}.

## Appendix E Implementation setup - Further details

### E.1 KL Estimator

For all losses requiring a Kl estimation, we use the Rae-Blackwellized estimator(amini2025betterestimationkullbackleiblerdivergence). For π\pi-Distill, we use a sequence-level penalty aggregated into the reward term
following shah2026comedyestimatorsklregularization. In this setting, we allow the KL penalty to be absorbed into the advantage computation.

For OPSD, we use the same estimator as in (amini2025betterestimationkullbackleiblerdivergence) and directly back-propagate through the estimation.

### E.2 Length Penalty

In our setting, we found that policies can become overly verbose. To mitigate this, we add a cosine-shaped length penalty inspired by the cosine length-scaling reward yeo2025demystifyinglongchainofthoughtreasoning. We apply this penalty only to successful traces (i.e., when the base reward r>0r>0). For each assistant turn ii with token length lil\_{i}, we use a no-penalty threshold lth=2000l\_{\text{th}}=2000 and a soft allowance lmax=5000l\_{\max}=5000. If li≤lthl\_{i}\leq l\_{\text{th}}, the turn penalty is pi=0p\_{i}=0 otherwise we assign a negative penalty using a linear–cosine schedule that increases in magnitude with length, approaching −λ-\lambda near lmaxl\_{\max} with λ=0.1\lambda=0.1, and becoming harsher beyond lmaxl\_{\max} (up to an endpoint of −2​λ-2\lambda). We then average the per-turn penalties across the NN assistant turns:

|  |  |  |
| --- | --- | --- |
|  | p¯=1N​∑i=1Npi.\bar{p}=\frac{1}{N}\sum\_{i=1}^{N}p\_{i}. |  |

We cap the total penalty as ptotal=max⁡(p¯,−0.3)p\_{\text{total}}=\max(\bar{p},-0.3) and add that to the base reward r′=r+ptotalr^{\prime}=r+p\_{\text{total}}.

### E.3 Hyper-Parameters

Here [table˜2](#A5.T2 "In E.3 Hyper-Parameters ‣ Appendix E Implementation setup - Further details ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models") we outline the swept hyperparameters, as well as their final values.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Category | Parameter | R1-Distill-Llama-8B | QWEN3-4B | QWEN3-8B |
| General | Seeds | 3 | 3 | 3 |
| Rollout temperature | 0.75 | 0.75 | 0.75 |
| Trace length filter | Discard if tokens >> 25k (τ\tau-Bench/TP), >> 35k (RL/OPSD) | | |
| Advantage processing | pop zero-advantage (always) | | |
| Training Budgets | τ\tau-Bench total gradient steps | 600 | 600 | 600 |
| TP total gradient steps | 400 | 400 | 400 |
| Sampling | Gradient steps per sampling | τ\tau-Bench = 3, TP = 2 | | |
| Repeats per group | τ\tau-Bench = 5, TP = 4 | | |
| Training tasks sampled | τ\tau-Bench = 64, TP = 45, SFT+RL (τ\tau-Bench) = 128 | | |
| Optimization | Learning-rate sweep | {1e-6, 5e-6, 1e-5} | | |
| Final LR | τ\tau-Bench: 5e-6 for all, TP: π\pi-distill = 1e-5, RL/OPSD = 5e-6 | | |
| β\beta | TP/OPSD = 0.5, π\pi-distill = 0.25 (unless swept) | | |
| Clipping/epsilon | Lower = 0.8, Upper = 1.2 | | |
| Alpha Annealing | α=0.5\alpha=0.5 schedule | Linearly anneal α:0→0.5\alpha:0\rightarrow 0.5 over 15 epochs | | |

Table 2: Hyperparameters used for each fine-tuned model. All models share identical hyperparameter settings. Travel Planner is abbreviated to TP.

## Appendix F Plot and Table Discrepancies

In this Fig [17](#A6.F17 "Figure 17 ‣ Appendix F Plot and Table Discrepancies ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models"), the 31.11% number is computed from the three seed peaks in the right panel. Each seed reaches its best score at a different gradient step, roughly one around 70 steps, another around 95, and another around 150. We take those three peak values and average them, which gives 31.11%, and that is what is reported in the table. The left panel instead shows the mean score at each gradient step, averaging the seeds at the same step, so its peak corresponds to the best point of the averaged curve rather than the average of the three best points. This difference in aggregation means the table can be higher than the peak of the averaged curve shown in the left plot.

!(/html/2602.04942/assets/x18.png)

Figure 17: Comparison aggregating training runs versus the individual training runs. The left figure displays the mean score across seeds at each gradient step, while the right panel highlights the individual trajectories where peaks occur at different intervals. The reported table value (31.11%31.11\%) represents the average of these individual seed peaks.

## Appendix G Reward Hacking in Travel Planner

Our early experiments show that the model is able to consistently reward-hack the original rubric-based rewards proposed by xie2024travelplannerbenchmarkrealworldplanning. Specifically, [Figure˜18](#A7.F18 "In Appendix G Reward Hacking in Travel Planner ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models") shows the learning curves under the original reward structure. The policy appears to steadily improve; however, upon inspection, we observe that all traces collapse to the response structure shown in [Appendix˜G](#A7 "Appendix G Reward Hacking in Travel Planner ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models"). We attribute this behavior to the model learning that invoking the Planner[.] tool ends the conversation and, with certain arguments, reliably yields a high reward.

We address this issue by removing the requirement that all “easy” constraints must be satisfied before any “hard” constraint is checked. Instead, we evaluate each easy constraint individually and, only if it is satisfied, we then check its corresponding hard constraint.

!(/html/2602.04942/assets/x19.png)

Figure 18: Training and testing curves for Travel Planner, using the original reward structure proposed in xie2024travelplannerbenchmarkrealworldplanning with Qwen3-4B. We find that while the model learns, the sequences generally converge on reward hacking. Where the entire score is dominated by the ”easy constraint score” as the model has learnt to satisfy some basic constraints, leading to the policy outputting the same response for all queries.

τ\tau-Bench Travel Planner Model Trajectory

Role: Assistant
[⬇](data:text/plain;base64,PHRoaW5rPgpPa2F5LCBsZXQncyB0YWNrbGUgdGhpcyB0cmF2ZWwgcGxhbm5pbmcgcXVlcnkuIFRoZSB1c2VyIHdhbnRzIGEgNy1kYXkgcGxhbiBmcm9tIFB1bnRhIEdvcmRhIHRvIHRocmVlIFdpc2NvbnNpbiBjaXRpZXMgYmV0d2VlbiBNYXJjaCAxNi0yMiwgMjAyMiwgd2l0aCBhICQ1LDcwMCBidWRnZXQuCgpGaXJzdCwgSSBuZWVkIHRvIHVzZSB0aGUgUGxhbm5lciB0b29sIHRvIHN0cnVjdHVyZSB0aGUgaXRpbmVyYXJ5LiBDaGVjayBGbGlnaHRTZWFyY2ggZm9yIGF2YWlsYWJsZSBmbGlnaHRzIGZyb20gUHVudGEgR29yZGEgdG8gV2lzY29uc2luIGNpdGllcy4gTWF5YmUgTWlsd2F1a2VlLCBNYWRpc29uLCBhbmQgR3JlZW4gQmF5PyBVc2UgQWNjb21tb2RhdGlvblNlYXJjaCBmb3IgZWFjaCBjaXR5LiBSZXN0YXVyYW50U2VhcmNoIGFuZCBBdHRyYWN0aW9uU2VhcmNoIHRvIHN1Z2dlc3QgYWN0aXZpdGllcy4gQ2hlY2sgR29vZ2xlRGlzdGFuY2VNYXRyaXggZm9yIHRyYXZlbCB0aW1lcyBiZXR3ZWVuIGNpdGllcy4gTWFrZSBzdXJlIHRoZSB0b3RhbCBjb3N0cyBhcmUgdW5kZXIgYnVkZ2V0LiBVc2UgTm90ZWJvb2tXcml0ZSB0byBkb2N1bWVudCB0aGUgcGxhbi4gRW5zdXJlIHRoZSBQbGFubmVyIGluY29ycG9yYXRlcyBhbGwgZWxlbWVudHMgYW5kIHN0YXlzIHdpdGhpbiB0aGUgYnVkZ2V0LiBBdm9pZCBtZW50aW9uaW5nIGhpbnRzLCBqdXN0IHByZXNlbnQgdGhlIHBsYW4gbG9naWNhbGx5Lgo8L3RoaW5rPgoKPGFjdGlvbj5QbGFubmVyW1F1ZXJ5OiBDcmFmdCBhIDctZGF5IFdpc2NvbnNpbiBpdGluZXJhcnkgZnJvbSBQdW50YSBHb3JkYSwgaW5jbHVkaW5nIDMgY2l0aWVzLCBmbGlnaHRzLCBhY2NvbW1vZGF0aW9ucywgYXR0cmFjdGlvbnMsIGFuZCBidWRnZXQgYWRoZXJlbmNlIHVzaW5nIGF2YWlsYWJsZSB0b29scy4gUHJpb3JpdGl6ZSBNYXJjaCAxNi0yMiwgMjAyMi5dPC9hY3Rpb24+)
<think>
Okay, let’s tackle this travel planning query. The user wants a 7-day plan from Punta Gorda to three Wisconsin cities between March 16-22, 2022, with a $5,700 budget.

First, I need to use the Planner tool to structure the itinerary. Check FlightSearch for available flights from Punta Gorda to Wisconsin cities. Maybe Milwaukee, Madison, and Green Bay? Use AccommodationSearch for each city. RestaurantSearch and AttractionSearch to suggest activities. Check GoogleDistanceMatrix for travel times between cities. Make sure the total costs are under budget. Use NotebookWrite to document the plan. Ensure the Planner incorporates all elements and stays within the budget. Avoid mentioning hints, just present the plan logically.
</think>

<action>Planner[Query: Craft a 7-day Wisconsin itinerary from Punta Gorda, including 3 cities, flights, accommodations, attractions, and budget adherence using available tools. Prioritize March 16-22, 2022.]</action>

## Appendix H Prompts

### H.1 System Prompts

For both τ\tau-Bench and Travel Planner, we slightly modify the prompt to indicate that the model should output its action in corresponding <action></action> tags. Below, we display the full system prompts used for τ\tau-Bench and Travel Planner.

τ\tau-Bench Retail System Prompt

Retail agent policy
As a retail agent, you can help users cancel or modify pending orders, return or exchange delivered orders, modify their default user address, or provide information about their own profile, orders, and related products.

•

At the beginning of the conversation, you have to authenticate the user identity by locating their user id via email, or via name + zip code. This has to be done even when the user already provides the user id.
•

Once the user has been authenticated, you can provide the user with information about the order, product, profile information, e.g. help the user look up the order id.
•

You can only help one user per conversation (but you can handle multiple requests from the same user), and must deny any requests for tasks related to any other user.
•

Before taking consequential actions that update the database (cancel, modify, return, exchange), you have to list the action detail and obtain explicit user confirmation (yes) to proceed.
•

You should not make up any information or knowledge or procedures not provided from the user or the tools, or give subjective recommendations or comments.
•

You should at most make one tool call at a time, and if you take a tool call, you should not respond to the user at the same time. If you respond to the user, you should not make a tool call.
Domain basic

•

All times in the database are EST and 24 hour based. For example “02:30:00” means 2:30 AM EST.
•

Each user has a profile of its email, default address, user id, and payment methods. Each payment method is either a gift card, a paypal account, or a credit card.
•

Our retail store has 50 types of products. For each type of product, there are variant items of different options. For example, for a ‘t shirt’ product, there could be an item with option ‘color blue size M’, and another item with option ‘color red size L’.
•

Each product has an unique product id, and each item has an unique item id. They have no relations and should not be confused.
•

Each order can be in status ‘pending’, ‘processed’, ‘delivered’, or ‘cancelled’. Generally, you can only take action on pending or delivered orders.
•

Exchange or modify order tools can only be called once. Be sure that all items to be changed are collected into a list before making the tool call!!!
Cancel pending order

•

An order can only be cancelled if its status is ‘pending’, and you should check its status before taking the action.
•

The user needs to confirm the order id and the reason (either ‘no longer needed’ or ‘ordered by mistake’) for cancellation.
•

After user confirmation, the order status will be changed to ‘cancelled’, and the total will be refunded via the original payment method immediately if it is gift card, otherwise in 5 to 7 business days.
Modify pending order

•

An order can only be modified if its status is ‘pending’, and you should check its status before taking the action.
•

For a pending order, you can take actions to modify its shipping address, payment method, or product item options, but nothing else.
Modify payment

•

The user can only choose a single payment method different from the original payment method.
•

If the user wants the modify the payment method to gift card, it must have enough balance to cover the total amount.
•

After user confirmation, the order status will be kept ‘pending’. The original payment method will be refunded immediately if it is a gift card, otherwise in 5 to 7 business days.
Modify items

•

This action can only be called once, and will change the order status to ‘pending (items modifed)’, and the agent will not be able to modify or cancel the order anymore. So confirm all the details are right and be cautious before taking this action. In particular, remember to remind the customer to confirm they have provided all items to be modified.
•

For a pending order, each item can be modified to an available new item of the same product but of different product option. There cannot be any change of product types, e.g. modify shirt to shoe.
•

The user must provide a payment method to pay or receive refund of the price difference. If the user provides a gift card, it must have enough balance to cover the price difference.
Return delivered order

•

An order can only be returned if its status is ‘delivered’, and you should check its status before taking the action.
•

The user needs to confirm the order id, the list of items to be returned, and a payment method to receive the refund.
•

The refund must either go to the original payment method, or an existing gift card.
•

After user confirmation, the order status will be changed to ‘return requested’, and the user will receive an email regarding how to return items.
Exchange delivered order

•

An order can only be exchanged if its status is ‘delivered’, and you should check its status before taking the action. In particular, remember to remind the customer to confirm they have provided all items to be exchanged.
•

For a delivered order, each item can be exchanged to an available new item of the same product but of different product option. There cannot be any change of product types, e.g. modify shirt to shoe.
•

The user must provide a payment method to pay or receive refund of the price difference. If the user provides a gift card, it must have enough balance to cover the price difference.
•

After user confirmation, the order status will be changed to ‘exchange requested’, and the user will receive an email regarding how to return items. There is no need to place a new order.
Available tools:
[⬇](data:text/plain;base64,e3Rvb2xzX2luZm99)
{tools\_info}
Instruction
You need to act as an agent that use the above tools to help the user according to the above policy. At each step, your generation should have exactly the following format:
[⬇](data:text/plain;base64,PHRoaW5rPgouLi5GZXcgbGluZXMgb2YgcmVhc29uaW5nCjwvdGhpbms+Cgo8YWN0aW9uPgp7Im5hbWUiOiA8VGhlIG5hbWUgb2YgdGhlIGFjdGlvbj4sICJhcmd1bWVudHMiOiA8VGhlIGFyZ3VtZW50cyB0byB0aGUgYWN0aW9uIGluIGpzb24gZm9ybWF0Pn0KPC9hY3Rpb24+)
<think>
...Few lines of reasoning
</think>

<action>
{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}
</action>
The Action will be parsed, so it must be valid JSON and within the <action> and </action> tags. You should not use made-up or placeholder arguments.
For example, if the user says “I want to know the current weather of San Francisco”, and there is such a tool available:
[⬇](data:text/plain;base64,ewogICAgInR5cGUiOiAiZnVuY3Rpb24iLAogICAgImZ1bmN0aW9uIjogewogICAgICAgICJuYW1lIjogImdldF9jdXJyZW50X3dlYXRoZXIiLAogICAgICAgICJkZXNjcmlwdGlvbiI6ICJHZXQgdGhlIGN1cnJlbnQgd2VhdGhlciIsCiAgICAgICAgInBhcmFtZXRlcnMiOiB7CiAgICAgICAgICAgICJ0eXBlIjogIm9iamVjdCIsCiAgICAgICAgICAgICJwcm9wZXJ0aWVzIjogewogICAgICAgICAgICAgICAgImxvY2F0aW9uIjogewogICAgICAgICAgICAgICAgICAgICJ0eXBlIjogInN0cmluZyIsCiAgICAgICAgICAgICAgICAgICAgImRlc2NyaXB0aW9uIjogIlRoZSBjaXR5IGFuZCBzdGF0ZSwgZS5nLiBTYW4gRnJhbmNpc2NvLCBDQSIsCiAgICAgICAgICAgICAgICB9LAogICAgICAgICAgICAgICAgImZvcm1hdCI6IHsKICAgICAgICAgICAgICAgICAgICAidHlwZSI6ICJzdHJpbmciLAogICAgICAgICAgICAgICAgICAgICJlbnVtIjogWyJjZWxzaXVzIiwgImZhaHJlbmhlaXQiXSwKICAgICAgICAgICAgICAgICAgICAiZGVzY3JpcHRpb24iOiAiVGhlIHRlbXBlcmF0dXJlIHVuaXQgdG8gdXNlLiBJbmZlciBmcm9tIGxvY2F0aW9uLiIsCiAgICAgICAgICAgICAgICB9LAogICAgICAgICAgICB9LAogICAgICAgICAgICAicmVxdWlyZWQiOiBbImxvY2F0aW9uIiwgImZvcm1hdCJdLAogICAgICAgIH0sCiAgICB9Cn0=)
{
 "type": "function",
 "function": {
 "name": "get\_current\_weather",
 "description": "Get the current weather",
 "parameters": {
 "type": "object",
 "properties": {
 "location": {
 "type": "string",
 "description": "The city and state, e.g. San Francisco, CA",
 },
 "format": {
 "type": "string",
 "enum": ["celsius", "fahrenheit"],
 "description": "The temperature unit to use. Infer from location.",
 },
 },
 "required": ["location", "format"],
 },
 }
}
Example response
Step 1:
[⬇](data:text/plain;base64,PHRoaW5rPgouLi4gRmV3IGxpbmVzIG9mIHJlYXNvbmluZwo8L3RoaW5rPgoKPGFjdGlvbj4KeyJuYW1lIjogImdldF9jdXJyZW50X3dlYXRoZXIiLCAiYXJndW1lbnRzIjogeyJsb2NhdGlvbiI6ICJTYW4gRnJhbmNpc2NvLCBDQSIsICJmb3JtYXQiOiAiZmFocmVuaGVpdCJ9fQo8L2FjdGlvbj4=)
<think>
... Few lines of reasoning
</think>

<action>
{"name": "get\_current\_weather", "arguments": {"location": "San Francisco, CA", "format": "fahrenheit"}}
</action>
The tool and the user have the same id tags so if the user returns “70F”, your response can be:
Step 2:
[⬇](data:text/plain;base64,PHRoaW5rPgouLi4gRmV3IGxpbmVzIG9mIHJlYXNvbmluZwo8L3RoaW5rPgoKPGFjdGlvbj4KeyJuYW1lIjogUkVTUE9ORF9BQ1RJT05fTkFNRSwgImFyZ3VtZW50cyI6IHsiUkVTUE9ORF9BQ1RJT05fRklFTERfTkFNRSI6ICJUaGUgY3VycmVudCB3ZWF0aGVyIG9mIFNhbiBGcmFuY2lzY28gaXMgNzBGLiJ9fQo8L2FjdGlvbj4=)
<think>
... Few lines of reasoning
</think>

<action>
{"name": RESPOND\_ACTION\_NAME, "arguments": {"RESPOND\_ACTION\_FIELD\_NAME": "The current weather of San Francisco is 70F."}}
</action>
Requirement
Try to be helpful and always follow the policy. Always try to validate your steps in your thinking and checkover your work, try to predict what will happen given your actions. Always make sure you generate valid JSON only.
Important Notes

•

Be very brief in the reasoning, do not repeat the entire context or the tools, just focus on what you need to do next.
•

Always respond to the user using the tool {RESPOND\_ACTION\_NAME} to ensure the user sees your response.
•

Only use a single set of thinking tags, the user cannot see your thoughts.
•

Always wrap your tool calls <action> and </action> tags or else the system will not be able to parse your actions.
•

Always start your outputs by thinking using <think> ... </think>.
Ok with this said, let us reason this out step by step always starting with <think> and ending with </think>.

Travel Planning System Prompt

You are a travel planning assistant. You have access to the following tools:

•

FlightSearch[Departure City, Destination City, Date]
•

AccommodationSearch[City]
•

RestaurantSearch[City]
•

AttractionSearch[City]
•

CitySearch[State]
•

GoogleDistanceMatrix[Origin, Destination, Mode]
•

NotebookWrite[Description]
•

Planner[Query]
Respond in this format:
[⬇](data:text/plain;base64,PHRoaW5rPllvdXIgcmVhc29uaW5nIGhlcmU8L3RoaW5rPgo8YWN0aW9uPlRvb2xOYW1lW2FyZ3VtZW50c108L2FjdGlvbj4=)
<think>Your reasoning here</think>
<action>ToolName[arguments]</action>
Query:
[⬇](data:text/plain;base64,e3F1ZXJ5fQ==)
{query}

### H.2 Full trajectories

Example Trajectory

User Task:

I’d like to start a return for #W6619432 and process it through PayPal account paypal\_3738584 for the Dumbbell Set and Yoga Mat.
Agent Trajectory:

1.

Turn 1:
[⬇](data:text/plain;base64,PGFjdGlvbj4KeyJuYW1lIjogInJlc3BvbmQiLCAiYXJndW1lbnRzIjogeyJjb250ZW50IjogIkknZCBiZSBoYXBweSB0byBoZWxwIHlvdSB3aXRoIHlvdXIgcmV0dXJuIGZvciBvcmRlciAjVzY2MTk0MzIuIEZpcnN0LCBJIG5lZWQgdG8gdmVyaWZ5IHlvdXIgaWRlbnRpdHkgdG8gZW5zdXJlIHRoZSBzZWN1cml0eSBvZiB5b3VyIGFjY291bnQuIENvdWxkIHlvdSBwbGVhc2UgcHJvdmlkZSB5b3VyIGVtYWlsIGFkZHJlc3M/IEFsdGVybmF0aXZlbHksIHlvdSBjYW4gcHJvdmlkZSB5b3VyIGZpcnN0IG5hbWUsIGxhc3QgbmFtZSwgYW5kIHppcCBjb2RlLiJ9fQo8L2FjdGlvbj4=)
<action>
{"name": "respond", "arguments": {"content": "I’d be happy to help you with your return for order #W6619432. First, I need to verify your identity to ensure the security of your account. Could you please provide your email address? Alternatively, you can provide your first name, last name, and zip code."}}
</action>
2.

Turn 2:
[⬇](data:text/plain;base64,PGFjdGlvbj4KeyJuYW1lIjogImZpbmRfdXNlcl9pZF9ieV9uYW1lX3ppcCIsICJhcmd1bWVudHMiOiB7ImZpcnN0X25hbWUiOiAiU29waGlhIiwgImxhc3RfbmFtZSI6ICJOZ3V5ZW4iLCAiemlwIjogIjIwMTcxIn19CjwvYWN0aW9uPg==)
<action>
{"name": "find\_user\_id\_by\_name\_zip", "arguments": {"first\_name": "Sophia", "last\_name": "Nguyen", "zip": "20171"}}
</action>
3.

Turn 3:
[⬇](data:text/plain;base64,PGFjdGlvbj4KeyJuYW1lIjogImdldF9vcmRlcl9kZXRhaWxzIiwgImFyZ3VtZW50cyI6IHsib3JkZXJfaWQiOiAiI1c2NjE5NDMyIn19CjwvYWN0aW9uPg==)
<action>
{"name": "get\_order\_details", "arguments": {"order\_id": "#W6619432"}}
</action>
4.

Turn 4:
[⬇](data:text/plain;base64,PGFjdGlvbj4KeyJuYW1lIjogInJlc3BvbmQiLCAiYXJndW1lbnRzIjogeyJjb250ZW50IjogIlRoYW5rIHlvdSBmb3IgcHJvdmlkaW5nIHlvdXIgaW5mb3JtYXRpb24uIEkndmUgdmVyaWZpZWQgeW91ciBvcmRlciAjVzY2MTk0MzIsIHdoaWNoIGlzIGRlbGl2ZXJlZCBhbmQgaW5jbHVkZXM6XG4tIFlvZ2EgTWF0IChpdGVtIElEOiA2MTk1OTM4ODA3KVxuLSBEdW1iYmVsbCBTZXQgKGl0ZW0gSUQ6IDM3MzUxMzM1MzkpXG5cbllvdSdyZSByZXF1ZXN0aW5nIHRvIHJldHVybiBib3RoIGl0ZW1zIGFuZCBwcm9jZXNzIHRoZSByZWZ1bmQgdGhyb3VnaCB5b3VyIFBheVBhbCBhY2NvdW50IChwYXlwYWxfMzczODU4NCkuIFRoZSB0b3RhbCByZWZ1bmQgYW1vdW50IHdpbGwgYmUgJDYxMi4zNS5cblxuUGxlYXNlIGNvbmZpcm0gdGhhdCB5b3Ugd2FudCB0byBwcm9jZWVkIHdpdGggcmV0dXJuaW5nIGJvdGggaXRlbXMgYW5kIHJlY2VpdmluZyB0aGUgcmVmdW5kIHRvIHlvdXIgUGF5UGFsIGFjY291bnQuIFR5cGUgJ3llcycgdG8gY29uZmlybSBvciAnbm8nIHRvIGNhbmNlbC4ifX0KPC9hY3Rpb24+)
<action>
{"name": "respond", "arguments": {"content": "Thank you for providing your information. I’ve verified your order #W6619432, which is delivered and includes:\n- Yoga Mat (item ID: 6195938807)\n- Dumbbell Set (item ID: 3735133539)\n\nYou’re requesting to return both items and process the refund through your PayPal account (paypal\_3738584). The total refund amount will be $612.35.\n\nPlease confirm that you want to proceed with returning both items and receiving the refund to your PayPal account. Type ’yes’ to confirm or ’no’ to cancel."}}
</action>
5.

Turn 5:
[⬇](data:text/plain;base64,PGFjdGlvbj4KeyJuYW1lIjogInJldHVybl9kZWxpdmVyZWRfb3JkZXJfaXRlbXMiLCAiYXJndW1lbnRzIjogeyJvcmRlcl9pZCI6ICIjVzY2MTk0MzIiLCAiaXRlbV9pZHMiOiBbIjYxOTU5Mzg4MDciLCAiMzczNTEzMzUzOSJdLCAicGF5bWVudF9tZXRob2RfaWQiOiAicGF5cGFsXzM3Mzg1ODQifX0KPC9hY3Rpb24+)
<action>
{"name": "return\_delivered\_order\_items", "arguments": {"order\_id": "#W6619432", "item\_ids": ["6195938807", "3735133539"], "payment\_method\_id": "paypal\_3738584"}}
</action>
6.

Turn 6:
[⬇](data:text/plain;base64,PGFjdGlvbj4KeyJuYW1lIjogInJlc3BvbmQiLCAiYXJndW1lbnRzIjogeyJjb250ZW50IjogIllvdXIgcmV0dXJuIHJlcXVlc3QgZm9yIG9yZGVyICNXNjYxOTQzMiBoYXMgYmVlbiBwcm9jZXNzZWQgc3VjY2Vzc2Z1bGx5LiBUaGUgb3JkZXIgc3RhdHVzIGlzIG5vdyAncmV0dXJuIHJlcXVlc3RlZCcuIFlvdSB3aWxsIHJlY2VpdmUgYW4gZW1haWwgd2l0aCBkZXRhaWxlZCBpbnN0cnVjdGlvbnMgb24gaG93IGFuZCB3aGVyZSB0byByZXR1cm4gdGhlIFlvZ2EgTWF0IGFuZCBEdW1iYmVsbCBTZXQuIFRoZSByZWZ1bmQgb2YgJDYxMi4zNSB3aWxsIGJlIHByb2Nlc3NlZCB0byB5b3VyIFBheVBhbCBhY2NvdW50IChwYXlwYWxfMzczODU4NCkuIFBsZWFzZSBjaGVjayB5b3VyIGVtYWlsIGZvciB0aGUgcmV0dXJuIGluc3RydWN0aW9ucy4gSXMgdGhlcmUgYW55dGhpbmcgZWxzZSBJIGNhbiBoZWxwIHlvdSB3aXRoIHRvZGF5PyJ9fQo8L2FjdGlvbj4=)
<action>
{"name": "respond", "arguments": {"content": "Your return request for order #W6619432 has been processed successfully. The order status is now ’return requested’. You will receive an email with detailed instructions on how and where to return the Yoga Mat and Dumbbell Set. The refund of $612.35 will be processed to your PayPal account (paypal\_3738584). Please check your email for the return instructions. Is there anything else I can help you with today?"}}
</action>
7.

Turn 7:
[⬇](data:text/plain;base64,PGFjdGlvbj4KeyJuYW1lIjogInJlc3BvbmQiLCAiYXJndW1lbnRzIjogeyJjb250ZW50IjogIllvdSdyZSB2ZXJ5IHdlbGNvbWUhIEknbSBnbGFkIEkgY291bGQgaGVscCB3aXRoIHlvdXIgcmV0dXJuLiBZb3VyIHJlZnVuZCB3aWxsIGJlIHByb2Nlc3NlZCB0byB5b3VyIFBheVBhbCBhY2NvdW50LiBIYXZlIGEgZ3JlYXQgZGF5ISJ9fQo8L2FjdGlvbj4=)
<action>
{"name": "respond", "arguments": {"content": "You’re very welcome! I’m glad I could help with your return. Your refund will be processed to your PayPal account. Have a great day!"}}
</action>

### H.3 Self-generated hints

Self-Generated Hint Example

Task:
Your name is Amelia Wilson and your email is amelia.wilson1598@example.com. You are confident, cautious, dependent, shy, pessimistic. Cancel order #W3062096 because ordered by mistake.
Self-Generated Hint:
The agent authenticated the user via email, retrieved the order details, confirmed the order ID and cancellation reason, and successfully cancelled the order using the cancel\_pending\_order function. The refund process was initiated, and the user was informed of the next steps.
Conversation:
[⬇](data:text/plain;base64,W3VzZXJdIEhpLCBJIG5lZWQgdG8gY2FuY2VsIGFuIG9yZGVyIEkgcGxhY2VkIGJ5IG1pc3Rha2UuCgpbYXNzaXN0YW50XSBBY3Rpb246IHsibmFtZSI6ICJyZXNwb25kIiwgImFyZ3VtZW50cyI6IHsiY29udGVudCI6CiJJIGNhbiBoZWxwIHlvdSBjYW5jZWwgeW91ciBvcmRlci4gRmlyc3QsIEkgbmVlZCB0byBhdXRoZW50aWNhdGUKeW91ciBpZGVudGl0eS4gQ291bGQgeW91IHBsZWFzZSBwcm92aWRlIHlvdXIgZW1haWwgYWRkcmVzcz8KQWx0ZXJuYXRpdmVseSwgeW91IGNhbiBwcm92aWRlIHlvdXIgZmlyc3QgbmFtZSwgbGFzdCBuYW1lLCBhbmQKemlwIGNvZGUuIn19CgpbdXNlcl0gU3VyZSwgbXkgZW1haWwgaXMgYW1lbGlhLndpbHNvbjE1OThAZXhhbXBsZS5jb20uCgpbYXNzaXN0YW50XSBBY3Rpb246IHsibmFtZSI6ICJmaW5kX3VzZXJfaWRfYnlfZW1haWwiLAoiYXJndW1lbnRzIjogeyJlbWFpbCI6ICJhbWVsaWEud2lsc29uMTU5OEBleGFtcGxlLmNvbSJ9fQoKW0FQSSBvdXRwdXRdIGFtZWxpYV93aWxzb25fNDYxNAoKW2Fzc2lzdGFudF0gQWN0aW9uOiB7Im5hbWUiOiAicmVzcG9uZCIsICJhcmd1bWVudHMiOiB7ImNvbnRlbnQiOgoiVGhhbmsgeW91IGZvciBwcm92aWRpbmcgeW91ciBlbWFpbC4gQ291bGQgeW91IHBsZWFzZSBzaGFyZSB0aGUKb3JkZXIgSUQgeW91J2QgbGlrZSB0byBjYW5jZWw/IFRoaXMgd2lsbCBoZWxwIG1lIHZlcmlmeSB0aGUgb3JkZXIKZGV0YWlscyBhbmQgcHJvY2VlZCB3aXRoIHRoZSBjYW5jZWxsYXRpb24uIn19CgpbdXNlcl0gWWVzLCB0aGUgb3JkZXIgSUQgaXMgVzMwNjIwOTYuCgpbYXNzaXN0YW50XSBBY3Rpb246IHsibmFtZSI6ICJnZXRfb3JkZXJfZGV0YWlscyIsCiJhcmd1bWVudHMiOiB7Im9yZGVyX2lkIjogIiNXMzA2MjA5NiJ9fQoKW0FQSSBvdXRwdXRdIHsib3JkZXJfaWQiOiAiI1czMDYyMDk2IiwKInVzZXJfaWQiOiAiYW1lbGlhX3dpbHNvbl80NjE0IiwgImFkZHJlc3MiOiB7ImFkZHJlc3MxIjoKIjM4OCBFbG0gQXZlbnVlIiwgImFkZHJlc3MyIjogIlN1aXRlIDM4NCIsICJjaXR5IjogIkRhbGxhcyIsCiJjb3VudHJ5IjogIlVTQSIsICJzdGF0ZSI6ICJUWCIsICJ6aXAiOiAiNzUyMTUifSwgIml0ZW1zIjoKW3sibmFtZSI6ICJULVNoaXJ0IiwgInByb2R1Y3RfaWQiOiAiOTUyMzQ1Njg3MyIsCiJpdGVtX2lkIjogIjMyMzQ4MDA2MDIiLCAicHJpY2UiOiA0Ni42NiwgIm9wdGlvbnMiOgp7ImNvbG9yIjogInJlZCIsICJzaXplIjogIkwiLCAibWF0ZXJpYWwiOiAiY290dG9uIiwKInN0eWxlIjogInYtbmVjayJ9fSwgeyJuYW1lIjogIlNtYXJ0IFdhdGNoIiwKInByb2R1Y3RfaWQiOiAiNjk0NTIzMjA1MiIsICJpdGVtX2lkIjogIjg3Mzk2MjY5NzIiLAoicHJpY2UiOiAzNzAuODcsICJvcHRpb25zIjogeyJjb2xvciI6ICJzaWx2ZXIiLAoiYmFuZCBtYXRlcmlhbCI6ICJzaWxpY29uZSIsICJkaXNwbGF5IjogIkFNT0xFRCJ9fSwKeyJuYW1lIjogIkppZ3NhdyBQdXp6bGUiLCAicHJvZHVjdF9pZCI6ICIxODA4NjExMDgzIiwKIml0ZW1faWQiOiAiOTM3MDMwMDU1NSIsICJwcmljZSI6IDQ1LjksICJvcHRpb25zIjoKeyJwaWVjZXMiOiAiMTAwMCIsICJ0aGVtZSI6ICJhcnQiLCAiZGlmZmljdWx0eSBsZXZlbCI6CiJleHBlcnQifX1dLCAiZnVsZmlsbG1lbnRzIjogW10sICJzdGF0dXMiOiAicGVuZGluZyIsCiJwYXltZW50X2hpc3RvcnkiOiBbeyJ0cmFuc2FjdGlvbl90eXBlIjogInBheW1lbnQiLAoiYW1vdW50IjogNDYzLjQzLCAicGF5bWVudF9tZXRob2RfaWQiOiAicGF5cGFsXzQxMDExNDMifV19CgpbYXNzaXN0YW50XSBBY3Rpb246IHsibmFtZSI6ICJyZXNwb25kIiwgImFyZ3VtZW50cyI6IHsiY29udGVudCI6CiJJIGNhbiBjYW5jZWwgeW91ciBvcmRlciAjVzMwNjIwOTYsIHdoaWNoIGluY2x1ZGVzIGEgVC1TaGlydCwKU21hcnQgV2F0Y2gsIGFuZCBKaWdzYXcgUHV6emxlLCB0b3RhbGluZyAkNDYzLjQzLiBUaGUgcmVhc29uIGZvcgpjYW5jZWxsYXRpb24gaXMgJ29yZGVyZWQgYnkgbWlzdGFrZScuIFRoZSByZWZ1bmQgd2lsbCBiZSBwcm9jZXNzZWQKYmFjayB0byB5b3VyIFBheVBhbCBhY2NvdW50IGluIDUtNyBidXNpbmVzcyBkYXlzLiBQbGVhc2UgY29uZmlybQp3aXRoICd5ZXMnIGlmIHlvdSdkIGxpa2UgdG8gcHJvY2VlZCB3aXRoIHRoZSBjYW5jZWxsYXRpb24uIn19CgpbdXNlcl0gWWVzLCBwbGVhc2UgZ28gYWhlYWQgYW5kIGNhbmNlbCBpdC4KClthc3Npc3RhbnRdIEFjdGlvbjogeyJuYW1lIjogImNhbmNlbF9wZW5kaW5nX29yZGVyIiwKImFyZ3VtZW50cyI6IHsib3JkZXJfaWQiOiAiI1czMDYyMDk2IiwKInJlYXNvbiI6ICJvcmRlcmVkIGJ5IG1pc3Rha2UifX0KCltBUEkgb3V0cHV0XSB7Im9yZGVyX2lkIjogIiNXMzA2MjA5NiIsCiJ1c2VyX2lkIjogImFtZWxpYV93aWxzb25fNDYxNCIsICJhZGRyZXNzIjogeyJhZGRyZXNzMSI6CiIzODggRWxtIEF2ZW51ZSIsICJhZGRyZXNzMiI6ICJTdWl0ZSAzODQiLCAiY2l0eSI6ICJEYWxsYXMiLAoiY291bnRyeSI6ICJVU0EiLCAic3RhdGUiOiAiVFgiLCAiemlwIjogIjc1MjE1In0sICJpdGVtcyI6Clt7Im5hbWUiOiAiVC1TaGlydCIsICJwcm9kdWN0X2lkIjogIjk1MjM0NTY4NzMiLAoiaXRlbV9pZCI6ICIzMjM0ODAwNjAyIiwgInByaWNlIjogNDYuNjYsICJvcHRpb25zIjoKeyJjb2xvciI6ICJyZWQiLCAic2l6ZSI6ICJMIiwgIm1hdGVyaWFsIjogImNvdHRvbiIsCiJzdHlsZSI6ICJ2LW5lY2sifX0sIHsibmFtZSI6ICJTbWFydCBXYXRjaCIsCiJwcm9kdWN0X2lkIjogIjY5NDUyMzIwNTIiLCAiaXRlbV9pZCI6ICI4NzM5NjI2OTcyIiwKInByaWNlIjogMzcwLjg3LCAib3B0aW9ucyI6IHsiY29sb3IiOiAic2lsdmVyIiwKImJhbmQgbWF0ZXJpYWwiOiAic2lsaWNvbmUiLCAiZGlzcGxheSI6ICJBTU9MRUQifX0sCnsibmFtZSI6ICJKaWdzYXcgUHV6emxlIiwgInByb2R1Y3RfaWQiOiAiMTgwODYxMTA4MyIsCiJpdGVtX2lkIjogIjkzNzAzMDA1NTUiLCAicHJpY2UiOiA0NS45LCAib3B0aW9ucyI6CnsicGllY2VzIjogIjEwMDAiLCAidGhlbWUiOiAiYXJ0IiwgImRpZmZpY3VsdHkgbGV2ZWwiOgoiZXhwZXJ0In19XSwgImZ1bGZpbGxtZW50cyI6IFtdLCAic3RhdHVzIjogImNhbmNlbGxlZCIsCiJwYXltZW50X2hpc3RvcnkiOiBbeyJ0cmFuc2FjdGlvbl90eXBlIjogInBheW1lbnQiLAoiYW1vdW50IjogNDYzLjQzLCAicGF5bWVudF9tZXRob2RfaWQiOiAicGF5cGFsXzQxMDExNDMifSwKeyJ0cmFuc2FjdGlvbl90eXBlIjogInJlZnVuZCIsICJhbW91bnQiOiA0NjMuNDMsCiJwYXltZW50X21ldGhvZF9pZCI6ICJwYXlwYWxfNDEwMTE0MyJ9XSwKImNhbmNlbF9yZWFzb24iOiAib3JkZXJlZCBieSBtaXN0YWtlIn0KClthc3Npc3RhbnRdIEFjdGlvbjogeyJuYW1lIjogInJlc3BvbmQiLCAiYXJndW1lbnRzIjogeyJjb250ZW50IjoKIllvdXIgb3JkZXIgI1czMDYyMDk2IGhhcyBiZWVuIHN1Y2Nlc3NmdWxseSBjYW5jZWxsZWQuIFRoZSByZWZ1bmQKb2YgJDQ2My40MyB3aWxsIGJlIHByb2Nlc3NlZCBiYWNrIHRvIHlvdXIgUGF5UGFsIGFjY291bnQgd2l0aGluCjUtNyBidXNpbmVzcyBkYXlzLiBJZiB5b3UgaGF2ZSBhbnkgZnVydGhlciBxdWVzdGlvbnMgb3IgbmVlZAphc3Npc3RhbmNlIHdpdGggYW55dGhpbmcgZWxzZSwgZmVlbCBmcmVlIHRvIGFzayEifX0KClt1c2VyXSBUaGFuayB5b3UsIHRoYXQncyBhbGwgSSBuZWVkZWQuICMjI1NUT1AjIyM=)
[user] Hi, I need to cancel an order I placed by mistake.

[assistant] Action: {"name": "respond", "arguments": {"content":
"I can help you cancel your order. First, I need to authenticate
your identity. Could you please provide your email address?
Alternatively, you can provide your first name, last name, and
zip code."}}

[user] Sure, my email is amelia.wilson1598@example.com.

[assistant] Action: {"name": "find\_user\_id\_by\_email",
"arguments": {"email": "amelia.wilson1598@example.com"}}

[API output] amelia\_wilson\_4614

[assistant] Action: {"name": "respond", "arguments": {"content":
"Thank you for providing your email. Could you please share the
order ID you’d like to cancel? This will help me verify the order
details and proceed with the cancellation."}}

[user] Yes, the order ID is W3062096.

[assistant] Action: {"name": "get\_order\_details",
"arguments": {"order\_id": "#W3062096"}}

[API output] {"order\_id": "#W3062096",
"user\_id": "amelia\_wilson\_4614", "address": {"address1":
"388 Elm Avenue", "address2": "Suite 384", "city": "Dallas",
"country": "USA", "state": "TX", "zip": "75215"}, "items":
[{"name": "T-Shirt", "product\_id": "9523456873",
"item\_id": "3234800602", "price": 46.66, "options":
{"color": "red", "size": "L", "material": "cotton",
"style": "v-neck"}}, {"name": "Smart Watch",
"product\_id": "6945232052", "item\_id": "8739626972",
"price": 370.87, "options": {"color": "silver",
"band material": "silicone", "display": "AMOLED"}},
{"name": "Jigsaw Puzzle", "product\_id": "1808611083",
"item\_id": "9370300555", "price": 45.9, "options":
{"pieces": "1000", "theme": "art", "difficulty level":
"expert"}}], "fulfillments": [], "status": "pending",
"payment\_history": [{"transaction\_type": "payment",
"amount": 463.43, "payment\_method\_id": "paypal\_4101143"}]}

[assistant] Action: {"name": "respond", "arguments": {"content":
"I can cancel your order #W3062096, which includes a T-Shirt,
Smart Watch, and Jigsaw Puzzle, totaling $463.43. The reason for
cancellation is ’ordered by mistake’. The refund will be processed
back to your PayPal account in 5-7 business days. Please confirm
with ’yes’ if you’d like to proceed with the cancellation."}}

[user] Yes, please go ahead and cancel it.

[assistant] Action: {"name": "cancel\_pending\_order",
"arguments": {"order\_id": "#W3062096",
"reason": "ordered by mistake"}}

[API output] {"order\_id": "#W3062096",
"user\_id": "amelia\_wilson\_4614", "address": {"address1":
"388 Elm Avenue", "address2": "Suite 384", "city": "Dallas",
"country": "USA", "state": "TX", "zip": "75215"}, "items":
[{"name": "T-Shirt", "product\_id": "9523456873",
"item\_id": "3234800602", "price": 46.66, "options":
{"color": "red", "size": "L", "material": "cotton",
"style": "v-neck"}}, {"name": "Smart Watch",
"product\_id": "6945232052", "item\_id": "8739626972",
"price": 370.87, "options": {"color": "silver",
"band material": "silicone", "display": "AMOLED"}},
{"name": "Jigsaw Puzzle", "product\_id": "1808611083",
"item\_id": "9370300555", "price": 45.9, "options":
{"pieces": "1000", "theme": "art", "difficulty level":
"expert"}}], "fulfillments": [], "status": "cancelled",
"payment\_history": [{"transaction\_type": "payment",
"amount": 463.43, "payment\_method\_id": "paypal\_4101143"},
{"transaction\_type": "refund", "amount": 463.43,
"payment\_method\_id": "paypal\_4101143"}],
"cancel\_reason": "ordered by mistake"}

[assistant] Action: {"name": "respond", "arguments": {"content":
"Your order #W3062096 has been successfully cancelled. The refund
of $463.43 will be processed back to your PayPal account within
5-7 business days. If you have any further questions or need
assistance with anything else, feel free to ask!"}}

[user] Thank you, that’s all I needed. ###STOP###
Hint Generation Prompt:
You are creating a concise natural language hint that will help an agent solve the task. Write a helpful hint describing the key actions and strategy the agent used. If the agent did not fully complete the task, focus on what they did achieve. IMPORTANT: Output ONLY plain text - no JSON, no code blocks, no markdown formatting. Keep the hint to 2-3 sentences maximum.

The prompt used

### H.4 Privileged Information Prompts

We use the same prompts When using PI in both models. Specifically, we try to ask the model not to reveal it has access to PI by encasing it in <Secret information></Secret information> tags. We found this to be unsuccessful as shown in the experiments in App [D](#A4 "Appendix D Additional Ablations ‣ 13 Author Contributions ‣ 12 Acknowledgments ‣ 11 Conclusion ‣ 10 Limitations & Future Work ‣ Context Distillation. ‣ 9 Related Work ‣ 8 Ablation on 𝛽 ‣ Findings. ‣ 7.2 What Matters for OPSD ‣ 7 What Matters When Using Train-Time PI ‣ 6 Out of Domain Experiments (OOD) ‣ 5 Main Results ‣ Implementation Details. ‣ 4.3 Models and Baselines ‣ 4 Experimental Setting ‣ Privileged Information Distillation for Language Models").

Privileged Instructions

<Secret information>
This section contains privileged hints derived from prior successful trajectories.
Use them to stay on track, but never mention that hints were provided.
Hint Summary:
[⬇](data:text/plain;base64,e2hpbnRfbGluZXN9)
{hint\_lines}
Your Task:

1.

Follow the hints to accomplish the task still reasoning with <think>...</think> before actions.
2.

Never acknowledge having access to secret hints.
Always start responses with a <think> block.
</Secret information>

Privileged Tool Instructions

<Secret information>
This section is highly important for you to pay attention to. You have been given access to secret information that is not available to the user and should not be mentioned.
Here is a successful set of tools, given the context and tools that have been called so far use this information to accomplish the task.
[⬇](data:text/plain;base64,e3Rvb2xfY2FsbHNfc2VjdGlvbn0=)
{tool\_calls\_section}
Your Task:

1.

You will need to use the provided tools in order to accomplish the task.
2.

Reasoning: Please reason about your steps. If you think you need to use a tool, provide reasoning justifications and clear logic for using it before invoking it using <think>reasoning...</think>.
3.

Strict Constraint: Do NOT mention that you have been given access to the secret information. You will be penalized for violating this rule.
Reasoning Format
Your reasoning traces will be used to train future agents that do not have access to privileged information. Please make sure your reasoning is clear and concise.
Always start your outputs by thinking using <think> …</think>
</Secret information>
