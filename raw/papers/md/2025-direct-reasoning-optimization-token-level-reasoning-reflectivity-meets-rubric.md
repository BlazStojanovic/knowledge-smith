---
arxiv: '2506.13351'
authors:
- Yifei Xu
- Tusher Chakraborty
- Srinagesh Sharma
- Leonardo Nunes
- Swati Sharma
- Kate Drakos Demopulos
- Emre Kıcıman
- Songwu Lu
- Ranveer Chandra
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'Direct Reasoning Optimization: Token-Level Reasoning Reflectivity Meets Rubric
  Gates for Unverifiable Tasks'
url: https://arxiv.org/abs/2506.13351
year: 2025
---

# Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks

Yifei Xu  12,
Tusher Chakraborty∗1,
Srinagesh Sharma1,
Leonardo Nunes1,
  
Emre Kıcıman1,
Songwu Lu2,
Ranveer Chandra1
  
1Microsoft
2University of California, Los Angeles
Equal contribution.

###### Abstract

Recent advances in Large Language Models (LLMs) have showcased impressive reasoning abilities in structured tasks like mathematics and programming, largely driven by Reinforcement Learning with Verifiable Rewards (RLVR), which uses outcome-based signals that are scalable, effective, and robust against reward hacking. However, applying similar techniques to open-ended long-form reasoning tasks remains challenging due to the absence of generic, verifiable reward signals. To address this, we propose Direct Reasoning Optimization (DRO), a reinforcement learning framework for fine-tuning LLMs on open-ended, particularly long-form, reasoning tasks, guided by a new reward signal: the Reasoning Reflection Reward (R3). At its core, R3 selectively identifies and emphasizes key tokens in the reference outcome that reflect the influence of the model’s preceding chain-of-thought reasoning, thereby capturing the consistency between reasoning and reference outcome at a fine-grained level. Crucially, R3 is computed internally using the same model being optimized, enabling a fully self-contained training setup. Additionally, we introduce a dynamic data filtering strategy based on R3 for open-ended reasoning tasks, reducing cost while improving downstream performance. We evaluate DRO on two diverse datasets – ParaRev, a long-form paragraph revision task, and FinQA, a math-oriented QA benchmark – and show that it consistently outperforms strong baselines while remaining broadly applicable across both open-ended and structured domains.

## 1 Introduction

The emergence of reasoning capabilities in Large Language Models (LLMs) has marked a major leap forward, particularly in tasks involving mathematics and programming (Guo et al., [2025](#bib.bib13); Jaech et al., [2024](#bib.bib18); Zeng et al., [2024](#bib.bib54); Yang et al., [2025](#bib.bib49); Kavukcuoglu, Koray, [2025](#bib.bib23)). To enable such reasoning, LLMs are trained using Reinforcement Learning with Verifiable Rewards (RLVR) technique guided by verifiable rewards computed from the model’s own final outcomes (Schulman et al., [2017](#bib.bib37); Guo et al., [2025](#bib.bib13); Liu et al., [2025a](#bib.bib30); Yu et al., [2025](#bib.bib51)). These rewards are derived from objective signals such as matching reference answers in math problems, passing unit tests in coding challenges, or selecting the correct option in multiple-choice questions (MCQ). Compared to traditional approaches like reward model training, verifiable rewards have proven effective in mitigating reward hacking and are relatively straightforward to implement (Guo et al., [2025](#bib.bib13); Shao et al., [2024](#bib.bib38); Yue et al., [2025](#bib.bib52); Liu et al., [2025a](#bib.bib30)).

Building on the success of reasoning capabilities in math, programming, and MCQ tasks with RLVR, there is growing interest in extending these techniques to open-ended tasks that require logical analysis, for example, revising a document in response to comments, composing analytical summaries or reports, or reviewing financial documents. The primary challenge lies in designing a generic, verifiable reward signal akin to those used in math and coding tasks (Zuo et al., [2025](#bib.bib60); Zhao et al., [2025b](#bib.bib57); Su et al., [2025](#bib.bib43); Chen et al., [2025c](#bib.bib6)). Given the limitations and potential inefficacy of training a separate reward model (Guo et al., [2025](#bib.bib13); Shao et al., [2024](#bib.bib38); Zuo et al., [2025](#bib.bib60); Zhao et al., [2025b](#bib.bib57)), the LLM-as-a-judge (Zheng et al., [2023](#bib.bib58); Lee et al., [2023](#bib.bib26)) may seem to be an alternative. However, relying on an external LLM to evaluate the outcomes of an actor LLM in RLVR introduces sensitivities to factors such as prompt design and optimization, model selection, the generator–discriminator gap, and reward hacking (Chen et al., [2025b](#bib.bib5); [b](#bib.bib5); Huang et al., [2024](#bib.bib16); Zuo et al., [2025](#bib.bib60); Xu et al., [2025b](#bib.bib48); Sharma et al., [2024](#bib.bib39)). Evaluating training model’s chain-of-thought (CoT) reasoning in semantic space adds an even greater challenge given how it hides reasoning in the latent space (Chen et al., [2025d](#bib.bib7)). Meanwhile, traditional similarity-based metrics such as ROUGE scores or cosine similarity often fail to capture key logical aspects of open-ended outcomes and remain vulnerable to reward hacking (Christiano et al., [2017](#bib.bib9); Stiennon et al., [2020](#bib.bib42); Su et al., [2025](#bib.bib43)).

To address these challenges, we first introduce a new token-level dense reward called the Reasoning Reflection Reward (R3). Owing to the autoregressive nature of LLMs, the CoT reasoning serves as a latent prefix that conditions the model’s generation of the final outcome. Consequently, the LLM’s token-level certainty of the reference outcome – measured under this reasoning prefix – effectively captures how likely the generated reasoning is to produce the correct outcome. However, in long-form generation, only a limited subset of tokens in the reference intrinsically reflect variations in reasoning paths, while many others are less informative and may dilute the reward signal. To overcome this, R3 selectively identifies and emphasizes the key tokens in the reference that are most sensitive to variations in reasoning, shaping the reward signal to focus on these reasoning-reflective tokens (Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks")). This approach enables the model to directly optimize its reasoning paths toward achieving the reference outcomes in open-ended tasks, promoting outcome-driven reasoning in a manner analogous to RLVR.

!(/html/2506.13351/assets/x1.png)

Figure 1: Illustrative example of Reasoning Reflection Reward (R3). For the paper revision task, the model is prompted to revise a paragraph based on reviewer comments (upper left). R3 computes per-token self-certainty (log-probabilities) in the reference revision (upper right) for each sampled reasoning trace, and highlights reasoning-reflective tokens using σ​(certainty)\sigma(\text{certainty}).
In this example, Reasoning A correctly identifies that Section 4 (overview) has been moved earlier and adjusts the paragraph structure accordingly, with a minor omission of section numbers. Reasoning B gives up. While a vanilla aggregate of certainty prefers B over A due to A’s lower certainty on the token “2”, R3 successfully aligns with the desired ranking by up-weighting high-σ​(certainty)\sigma(\text{certainty}) tokens “gives”, “existing” and “.” that better reflect reasoning effectiveness.

!(/html/2506.13351/assets/x2.png)

Figure 2: Overview of Direct Reasoning Optimization (DRO), a framework that rewards and refines reasoning by directly leveraging feedback from the training model. DRO operates within the GRPO framework, where a group of CoT reasoning traces sampled from the actor policy (πθ\pi\_{\theta}) are scored primarily using the R3 score along with length penalty on final outcome. The reward is computed via an internal policy (πrwd\pi\_{\text{rwd}}), derived from the same base reference policy (πref\pi\_{\text{ref}}) being optimized. DRO employs R3-based dynamic training data filtering for open-ended reasoning tasks to improve data efficiency and downstream task performance.

We then propose Direct Reasoning Optimization (DRO), an RL-based fine-tuning framework that leverages R3 as its core reward signal. To compute R3, DRO directly uses a dynamic reward policy derived from the same reference policy (LLM) being optimized – thereby eliminating the need for any external reward model or signal. Our method builds upon the widely adopted RLVR framework, Group Relative Policy Optimization (GRPO) (Guo et al., [2025](#bib.bib13); Shao et al., [2024](#bib.bib38)), extending its outcome-driven effectiveness to open-ended reasoning tasks. DRO further integrates a ubiquitous data filtering technique for open-ended reasoning tasks, motivated by the growing recognition of data selection’s importance in recent work (Muennighoff et al., [2025](#bib.bib36); Jiang et al., [2025](#bib.bib19); Ye et al., [2025](#bib.bib50); Yang et al., [2025](#bib.bib49)). Our approach leverages R3 to dynamically filter training samples during RL training, without requiring any task-specific filtering heuristics or external frameworks. This filtering strategy improves downstream performance while simultaneously reducing training cost and time.

Finally, we evaluate DRO on two distinct datasets—ParaRev(Jourdan et al., [2025](#bib.bib21)) and FinQA(Chen et al., [2021](#bib.bib8))—using two Qwen reasoning models distilled from DeepSeek-R1. To the best of our knowledge, this is the first work to evaluate reasoning optimization on an open-ended task like paragraph revision (ParaRev), which involves relatively long-form textual outputs beyond the traditional math and programming domains. On ParaRev, DRO outperforms all baseline methods in terms of downstream task performance while achieving around 45%45\% reduction in training cost. We further validate DRO on FinQA, a task with classic math-style answers, demonstrating that it achieves comparable performance to standard binary verifiable reward approaches—highlighting its versatility across both structured and open-ended tasks.

## 2 Related Work

### 2.1 LLM Reasoning

Chain-of-Thought (CoT) reasoning has emerged as a critical driver of advanced reasoning in LLMs, improving accuracy across mathematical, commonsense, and logical tasks while increasing transparency in the decision-making process. Initial prompting-based methods demonstrated that LLMs could be guided to reason step-by-step without additional training, resulting in significant performance gains (Kojima et al., [2022](#bib.bib24); Huang & Chang, [2022](#bib.bib17); Zhang et al., [2022](#bib.bib55); Zelikman et al., [2022](#bib.bib53); Wei et al., [2022](#bib.bib46)). Building on this foundation, recent approaches have incorporated CoT reasoning into the training loop—either through supervised fine-tuning on annotated reasoning traces (Zelikman et al., [2022](#bib.bib53)) or via reinforcement learning with process- or outcome-based rewards (Shao et al., [2024](#bib.bib38); Lambert et al., [2024](#bib.bib25))—to further strengthen reasoning capabilities. By decomposing problems into intermediate steps, LLMs not only improve in accuracy but also become more interpretable and trustworthy, both of which are essential for real-world deployment (Lightman et al., [2023](#bib.bib27)).

### 2.2 Reinforcement Learning with Verifiable Rewards

Reinforcement Learning from Verifiable Rewards (RLVR) has emerged as a powerful framework for improving LLM performance in domains where success can be unambiguously defined and automatically evaluated (Lambert et al., [2024](#bib.bib25); Liu et al., [2025a](#bib.bib30); Su et al., [2025](#bib.bib43)). In areas such as coding and mathematics, RLVR has enabled substantial advancements—models now solve complex problems and generate correct code with unprecedented accuracy and consistency (Shao et al., [2024](#bib.bib38); Yu et al., [2025](#bib.bib51); Muennighoff et al., [2025](#bib.bib36); Ye et al., [2025](#bib.bib50); Hu et al., [2025](#bib.bib15); [Luo et al.,](#bib.bib33) ; Liu & Zhang, [2025](#bib.bib29)). This success stems from the integration of reinforcement learning with deterministic outcome verification, eliminating the need for learned reward models and facilitating large-scale training on diverse problem sets. However, extending RLVR to open-ended reasoning tasks remains a significant challenge. These tasks often involve diverse reasoning paths and multiple valid outcomes, making it difficult to define rule-based or verifiable rewards. As a result, designing reward signals that reliably reflect reasoning quality in such settings is still an open problem.

### 2.3 Reinforcement Learning without External Verifier

Over the past year, considerable efforts have been made to extend the success of RLVR to open-ended reasoning tasks. One line of work focuses on training general-purpose reward models to supervise reasoning optimization (Chen et al., [2025c](#bib.bib6); Liu et al., [2025b](#bib.bib31); Su et al., [2025](#bib.bib43)), which introduces the overhead of developing and maintaining an additional reward model during RL training. A complementary line of research explores the use of internal model feedback, such as self-certainty—as a reward signal, thereby eliminating the need for external verifiers (Zhao et al., [2025b](#bib.bib57); [a](#bib.bib56); Xu et al., [2025a](#bib.bib47); Zuo et al., [2025](#bib.bib60); Zhou et al., [2025](#bib.bib59); Chen et al., [2024](#bib.bib3); Tang et al., [2025](#bib.bib44)). Among these, several concurrent studies (Zhao et al., [2025a](#bib.bib56); Xu et al., [2025a](#bib.bib47); Zhao et al., [2025b](#bib.bib57); Zuo et al., [2025](#bib.bib60)) rely exclusively on intrinsic feedback to optimize reasoning traces without reference answers, while other concurrent studies (Tang et al., [2025](#bib.bib44); Zhou et al., [2025](#bib.bib59)) incorporate reference outcomes to estimate the quality of generated reasoning. However, none of these approaches examine the token-level sensitivity of reasoning-quality rewards in the context of open-ended, long-form generation, as we introduce in Section [4.1](#S4.SS1 "4.1 Reasoning Reflection Reward (R3) ‣ 4 DRO for Open-ended Tasks ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks"). Additionally, prior work does not address data filtering for reasoning training using task-independent, model-internal rewards – an approach we propose in Section [4.2.2](#S4.SS2.SSS2 "4.2.2 Data Filtering and Efficiency ‣ 4.2 Direct Reasoning Optimization with R3 ‣ 4 DRO for Open-ended Tasks ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks") to improve data efficiency. Finally, to the best of our knowledge, we are the first to evaluate RL-based reasoning optimization on a long-form open-ended task such as paragraph revision in ParaRev (Section [5.1](#S5.SS1 "5.1 Experimental Setup ‣ 5 Experiments ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks")).

## 3 Background: Reasoning Optimization with RL

Recent advances in LLM reasoning have largely been driven by reinforcement learning (RL)-based optimization techniques. To ground this process theoretically, we begin by framing RL-based reasoning optimization within the Markov Decision Process (MDP) framework. For LLMs, the MDP can be naturally defined at the token level, as the model generates one token at each time step tt. In this setup, the state sts\_{t} at time tt consists of the input prompt or question q followed by the sequence of output tokens generated so far (o<t\textbf{o}\_{<t}), i.e., st=q;o<ts\_{t}=\textbf{q};\textbf{o}\_{<t}. The LLM, acting as the policy πθ\pi\_{\theta}, takes an stochastic action by picking the next token (oto\_{t}) from its vocabulary based on the current state sts\_{t}. The state then transitions to st+1=st;[ot]s\_{t+1}=s\_{t};[o\_{t}]. With RL-based reasoning optimization, the goal is to learn an optimal policy π∗\pi^{\ast} that generates a sequence of tokens conditioned on the question q in such a way that it leads to a desired final outcome, such as the correct answer to a math question.

In order to optimize the policy πθ\pi\_{\theta}, Shao et al. ([2024](#bib.bib38)) proposed Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO) Schulman et al. ([2017](#bib.bib37)). The surrogate objective in GRPO, which is maximized to learn the optimal policy, is defined as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒥GRPO​(θ)=\displaystyle\mathcal{J}\_{\text{GRPO}}(\theta)= | 𝔼q∼P(Q),{oi}i=1G∼πθold(⋅|q)\displaystyle\mathbb{E}\_{\textbf{q}\sim P(Q),\{\textbf{o}\_{i}\}\_{i=1}^{G}\sim\pi\_{\theta\_{\text{old}}}(\cdot|\textbf{q})} |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | [1G∑i=1G1|oi|∑t=1|oi|{min[πθ​(oi|q,oi<t)πθold​(oi|q,oi<t)A^i,t,clip(πθ​(oi|q,oi<t)πθold​(oi|q,oi<t),1−ϵ,1+ϵ)A^i,t]}\displaystyle\Biggl{[}\frac{1}{G}\sum\_{i=1}^{G}\frac{1}{\lvert\textbf{o}\_{i}\rvert}\sum\_{t=1}^{\lvert\textbf{o}\_{i}\rvert}\left\{\min\left[\frac{\pi\_{\theta}(o\_{i}|\textbf{q},\textbf{o}\_{i<t})}{\pi\_{\theta\_{\text{old}}}(o\_{i}|\textbf{q},\textbf{o}\_{i<t})}\hat{A}\_{i,t},\ \text{clip}\left(\frac{\pi\_{\theta}(o\_{i}|\textbf{q},\textbf{o}\_{i<t})}{\pi\_{\theta\_{\text{old}}}(o\_{i}|\textbf{q},\textbf{o}\_{i<t})},1-\epsilon,1+\epsilon\right)\hat{A}\_{i,t}\right]\right\} |  | (1) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −β𝔻KL(πθ∥πref)]\displaystyle-\beta\,\mathbb{D}\_{\text{KL}}\left(\pi\_{\theta}\|\pi\_{\text{ref}}\right)\Biggr{]} |  |

where ϵ\epsilon is the clipping parameter to maintain stability and πθold\pi\_{\theta\_{\text{old}}} denotes the policy before the most recent update. The key distinction in GRPO lies in the computation of it​hi^{th} token’s advantage estimate A^i,t\hat{A}\_{i,t}, which introduces a structured comparison across a group of generations from the same questions. Specifically, for a given prompt or question, suppose we sample a group of GG outputs {oi}i=1G\left\{\textbf{o}\_{i}\right\}\_{i=1}^{G} from the actor model with corresponding rewards {ri}i=1G\left\{r\_{i}\right\}\_{i=1}^{G}. Then, for each token in the it​hi^{th} output, the advantage is estimated as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | A^i,t=ri−m​e​a​n​({ri}i=1G)s​t​d​({ri}i=1G)\hat{A}\_{i,t}=\frac{r\_{i}-mean(\left\{r\_{i}\right\}\_{i=1}^{G})}{std(\left\{r\_{i}\right\}\_{i=1}^{G})} |  | (2) |

In the context of RLVR, rir\_{i} is typically a verifiable reward computed on the final outcome – such as 1 if the final answer is correct and 0 otherwise. Note that each sampled output (oi\textbf{o}\_{i}) consists of CoT reasoning followed my final answer. This group-normalized formulation encourages the policy to assign higher probabilities to trajectories that outperform their peers, steering generation toward more promising reasoning paths. As a result, the model learns to sample tokens that are more likely to lead to correct or high-reward outcomes.

Finally, GRPO includes a KL divergence regularization term, 𝔻KL\mathbb{D}\_{\text{KL}}, to constrain the updated policy from deviating too much from the reference policy. This regularization is critical in preventing overfitting or reward exploitation – especially when a proxy reward model is used instead of reference. At the same time, a certain degree of exploration is necessary for the policy to meaningfully evolve beyond the reference policy. The hyperparameter β\beta controls this trade-off between exploration and exploitation. In the context of RLVR, where the reward is derived from matching reference answers (rather than a learned model), this risk is mitigated, and therefore recent state-of-the-art approaches often set β=0\beta=0 (Liu et al., [2025a](#bib.bib30); Guo et al., [2025](#bib.bib13)).

## 4 DRO for Open-ended Tasks

The success of the RLVR technique, as outlined in the previous section, stems from its simple yet robust reward mechanism based on verifiable reference outcomes. This outcome-driven reward structure makes RL training more resilient to reward hacking (Silver et al., [2017](#bib.bib40)). RLVR has proven particularly effective in domains such as mathematics, programming, and logical reasoning—where the correctness of a model’s output can be objectively verified and reliably translated into rewards (Shao et al., [2024](#bib.bib38); Su et al., [2025](#bib.bib43)). However, extending RLVR to open-ended, especially long-form generation, tasks – such as text drafting and revision, composing analytical summaries or reports, or form completion—poses a significant challenge. In these scenarios, verifying logical correctness and translating it into a clean reward signal is inherently difficult, even when reference outcomes are available (Zhao et al., [2025b](#bib.bib57); [a](#bib.bib56); Xu et al., [2025a](#bib.bib47); Zhou et al., [2025](#bib.bib59); Lu, [2025](#bib.bib32)). Considering potential solutions, we observe that:

Traditional similarity-based metrics fail to capture the essential features of open-ended reasoning outcomes. An intuitive approach involves measuring the similarity between the model-generated output and the reference text using surface-level metrics such as ROUGE, which rely on n-gram overlap. However, such metrics are ill-suited for evaluating logical coherence or reasoning consistency, as they emphasize lexical similarity rather than logical or structural alignment. Two responses that are logically equivalent but lexically distinct may receive a low ROUGE score, while a response that merely copies phrases from the ground truth – without preserving the underlying logic – may score highly. Embedding-based metrics such as cosine similarity offer a more flexible representation space, but they still struggle to reliably distinguish reasoning-valid outputs from superficially similar yet logically flawed ones.

External dense reward models are infeasible for open-ended reasoning tasks. Leveraging a dedicated reward model to provide dense feedback typically requires preference-style datasets comprising paired examples of preferred outputs – a resource that is organically often unavailable for many open-ended tasks (Ethayarajh et al., [2024](#bib.bib11)). Training such a reward model introduces additional computational and annotation costs, further limiting its practicality. More critically, reward models are susceptible to reward hacking, where models exploit weaknesses in the learned reward signal rather than genuinely improving reasoning quality (Silver et al., [2017](#bib.bib40); Shao et al., [2024](#bib.bib38)).

LLM-as-a-judge is not a turnkey or reliable solution for reasoning reward signal. Recently, LLMs have been increasingly adopted as automated evaluators in place of human judges (Gu et al., [2024](#bib.bib12)). However, multiple studies have raised concerns about their reliability, highlighting issues such as sensitivity to prompt phrasing and evaluation rubrics, self-enhancement bias, prone to reward hacking, and the generator–discriminator gap (Sharma et al., [2024](#bib.bib39); Gu et al., [2024](#bib.bib12); Chen et al., [2025c](#bib.bib6); Liu et al., [2025b](#bib.bib31); Chen et al., [2025b](#bib.bib5)). Moreover, extracting a dense, task-specific reward signal from LLM-as-a-judge remains particularly challenging (Liu et al., [2025b](#bib.bib31); Chen et al., [2025c](#bib.bib6)). This challenge is further compounded when aiming for a scalable and turnkey fine-tuning framework across diverse tasks and datasets (Microsoft, [2024](#bib.bib35); Atreya, [2024](#bib.bib1); Xu et al., [2025b](#bib.bib48)), as the LLM-as-a-judge must be carefully tailored, validated, and maintained for each new use case (Liu et al., [2025b](#bib.bib31)).

### 4.1 Reasoning Reflection Reward (R3)

Before addressing the challenges discussed above, it is important to understand how a reasoning-capable LLM generates outputs in response to a question or prompt. The output of such a model typically consists of two components: a CoT reasoning segment, followed by the final answer. Due to the autoregressive nature of LLMs, the CoT reasoning acts as a latent prefix that conditions the generation of the final answer (Chen et al., [2025d](#bib.bib7); [2024](#bib.bib3)). In this formulation, the CoT reasoning can be viewed as an implicit intermediate state that guides the model’s final outcome generation. Specifically, the final answer is generated according to the conditional probability distribution π(⋅∣q,c^)\pi(\cdot\mid\textbf{q},\hat{\textbf{c}}), where q denotes the input question or prompt, and c^\hat{\textbf{c}} is the CoT reasoning generated in response to q. Intuitively, the quality of the reasoning trace directly influences the likelihood of producing a correct answer – strong reasoning increases this likelihood, while flawed reasoning reduces it.

Building upon this property, we introduce a new reward signal – Reasoning Reflection Reward (R3) – designed specifically for open-ended, particularly long-form, generation tasks. R3 is a token-level dense reward signal that measures the consistency between the CoT reasoning generated by the actor model and the reference outcome by placing special emphasis on the key tokens in the reference that reflect the preceding CoT reasoning. We quantify this consistency by leveraging the model’s own self-certainty (Gupta et al., [2024](#bib.bib14); Kauf et al., [2024](#bib.bib22)) – specifically, the probabilistic likelihood assigned by the LLM to the reference outcome y conditioned on the prompt q and its generated CoT reasoning c^\hat{\textbf{c}}, i.e., π​(y∣q,c^)\pi(\textbf{y}\mid\textbf{q},\hat{\textbf{c}}). Intuitively, if the model’s reasoning c^\hat{\textbf{c}} is correct, the model should assign a high likelihood to the reference outcome y. This likelihood thus serves as a natural reward signal to assess the quality of the generated CoT reasoning. Moreover, since it is grounded in golden answers rather than learned reward models, it offers greater reliability and alignment with the target objective – making it a robust choice for RL training, as recommended in state-of-the-art (Shao et al., [2024](#bib.bib38); Silver et al., [2017](#bib.bib40)). However, an important oversight in this formulation – also overlooked in recent state-of-the-art work (Chen et al., [2024](#bib.bib3); Zhou et al., [2025](#bib.bib59); Tang et al., [2025](#bib.bib44)) – is the uniform treatment of all tokens in the reference outcome y. In practice, this assumption can significantly undermine the effectiveness of the reward signal, and in some cases, even introduce reverse effect – particularly in long-form generation tasks. Next, we present two key empirical observations that reveal why only a selective subset of reference tokens meaningfully contributes to reasoning consistency.

#### 4.1.1 Key Reasoning Token Blindness in Log-Probability Aggregation

A model’s self-certainty over a reference outcome token yjy\_{j}, conditioned on a sampled CoT reasoning trace c^i\hat{\textbf{c}}\_{i}, can be formulated as the conditional probability π​(yj∣q,c^i,y<j)\pi(y\_{j}\mid\textbf{q},\hat{\textbf{c}}\_{i},\textbf{y}\_{<j}). In practice, we compute this by sequentially appending the reference tokens after the sampled reasoning c^i\hat{\textbf{c}}\_{i} and measuring the likelihood of each next reference token given the preceding context. Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks") illustrates such a conditional log-probability distribution over the reference outcome tokens for an example prompt from the ParaRev dataset. By way of background, in the ParaRev task, the goal is to revise a given paragraph in response to a set of reviewers’ comments, where not all comments are necessarily relevant to the paragraph (see Section [5.1](#S5.SS1 "5.1 Experimental Setup ‣ 5 Experiments ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks") for details). To measure the aforementioned consistency between a sampled CoT trace and the reference outcome, we simply begin by computing the aggregate probability of the reference tokens under the model’s predictive distribution, i.e., ∑j=1|y|log⁡(π​(yj∣q,ci^,y<j))\sum\_{j=1}^{\lvert\textbf{y}\rvert}\log\left(\pi(y\_{j}\mid\textbf{q},\hat{\textbf{c}\_{i}},\textbf{y}\_{<j})\right). We then use this aggregate value as reward (rir\_{i}) in Eq.[2](#S3.E2 "In 3 Background: Reasoning Optimization with RL ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks") to compute the corresponding advantage value for the sampled reasoning trace c^i\hat{\textbf{c}}\_{i} (a part of sampled output oi\textbf{o}\_{i}). Our objective is to assign higher advantage scores to higher-quality CoT traces, enabling a cleaner signal in the optimization objective (Eq.[3](#S3.Ex1 "3 Background: Reasoning Optimization with RL ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks")).

To evaluate whether this plain aggregate token-level probability reward effectively distinguishes better CoT traces within a group, we conduct a case study using a representative example from the ParaRev dataset. Specifically, we sample 16 outputs in response to a given prompt, where each output consists of a CoT reasoning trace followed by an answer (i.e., a revised paragraph). We then manually rank these outputs based on the quality of their final answers and CoT traces – assessing how well they address the relevant reviewer comments from the prompt and align with the reference revision. Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks") presents two representative CoT reasoning samples from this set, arranged in descending order of quality. The differences in quality are visibly substantial. For each CoT sample, we show the corresponding advantage values computed using the aggregate conditional log-probabilities over the reference tokens. Interestingly, the derived advantage values show only weak correlation with the actual sample quality and in the figure, even rank lower-quality CoT trace above higher-quality one.

To understand this unexpected behavior, we closely examine the log-probability distributions over the reference outcome shown in Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks"). Most tokens in the reference sequence receive similar log-probability values, regardless of the preceding CoT reasoning. Only a small number of tokens – three in this case – exhibit clear variation in likelihood depending on the prior CoT trace. These reasoning-reflective tokens are the ones that truly encode the effect of the preceding reasoning on the model’s certainty over the outcome. However, since these reflective tokens tend to have lower log-probability values than the bulk of the reference tokens, their influence gets diluted when we compute a sequence-wide aggregate log-probability. As a result, their contribution to the reward for the CoT trace, and thus to the corresponding advantage value is effectively masked. This issue becomes more pronounced when the number of reasoning-reflective tokens is small relative to the total length of the reference outcome. This phenomenon, where critical token-level signals are suppressed by sequence-wide aggregation, has also been observed in other contexts such as model cascading and hallucination detection (Gupta et al., [2024](#bib.bib14); Chen et al., [2025a](#bib.bib4)).

#### 4.1.2 When Reference Tokens Compensate for Poor Reasoning

When computing the reasoning-conditioned probability of the jthj^{\text{th}} reference token using π​(yj∣q,c^i,y<j)\pi(y\_{j}\mid\textbf{q},\hat{\textbf{c}}\_{i},\textbf{y}\_{<j}), we are inherently conditioning not only on the CoT reasoning trace c^i\hat{\textbf{c}}\_{i} but also on all preceding reference tokens. While this formulation is standard in autoregressive models, it introduces a subtle confound in estimating the model’s certainty: preceding reference tokens can influence the likelihood of subsequent ones, potentially inflating the overall reward. For instance, consider a scenario where the question involves identifying a goal scorer, and the reference answer is “Lionel Messi”. If the model’s CoT fails to identify the correct answer, the probability of “Lionel” conditioned on the flawed reasoning may be low. However, once “Lionel” is appended to the sequence, the probability of “Messi” is likely to be high due to strong lexical and semantic associations. In effect, the reference tokens themselves can progressively compensate for reasoning errors, leading to an overestimation of the quality of the CoT trace. This effect becomes more pronounced as the reference sequence grows longer, particularly when later tokens are highly correlated with earlier ones. Similar issues have been documented in studies of hallucination propagation and teacher-forced training within autoregressive generation (Varshney et al., [2023](#bib.bib45); Bachmann & Nagarajan, [2024](#bib.bib2)).

#### 4.1.3 R3 Addresses Token-level Sensitivity

R3 addresses the two aforementioned challenges in leveraging an LLM’s self-certainty over the reference outcome – conditioned on a sampled CoT reasoning trace – as a reward for reasoning quality. First, it mitigates the issue of reasoning-reflective token blindness in aggregate log-probability computation by explicitly identifying and emphasizing such tokens. A natural but impractical approach would be to identify reasoning-reflective tokens via semantic analysis of the prompt and reference outcome, for instance, using an LLM-as-a-judge framework. However, such methods do not scale across prompts and datasets and inherit the reliability concerns associated with LLM-based judges, as discussed earlier. Moreover, many reasoning-reflective tokens may not present themselves as semantically salient in isolation as illustrated in Fig.[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks"). To address this, we adopt a comparative approach: we identify reasoning-reflective tokens as those whose likelihoods exhibit high variation when conditioned on different CoT traces. That is, in reasoning-conditioned log-probability estimation, the tokens in the reference outcome that show substantial variability across a set of sampled CoT traces are likely to reflect the influence of upstream reasoning. This comparative nature is also emphasized in GRPO paper with a connection to preference-based reward modeling (Shao et al., [2024](#bib.bib38)). For example, in Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks"), we highlight three tokens from the reference outcome that exhibit high standard deviation in their log-probabilities across 16 distinct CoT traces. These tokens are not only statistically reflective of reasoning variation but also intuitively important upon qualitative inspection. In R3, we emphasize these reasoning-reflective tokens by weighting each reference token’s log-probability contribution according to its standard deviation. Specifically, the CoT-conditioned likelihood of the reference outcome is computed as: ∑j=1|y|wΔ​(σj)​log⁡(π​(yj∣q,ci^,y<j))\sum\_{j=1}^{\lvert\textbf{y}\rvert}w\_{\Delta}(\sigma\_{j})\log\left(\pi(y\_{j}\mid\textbf{q},\hat{\textbf{c}\_{i}},\textbf{y}\_{<j})\right), where wΔ​(σj)w\_{\Delta}(\sigma\_{j}) assigns greater weight to tokens with higher standard deviation σj\sigma\_{j}, thereby amplifying the influence of reasoning-reflective tokens in the reward estimation.

Next, we turn our attention to the second challenge: the tendency of reference tokens to compensate for poor CoT reasoning. A natural idea is to propagate the self-certainty (i.e., token-level likelihood) of all preceding reference tokens when computing the certainty of a given token. However, this approach is computationally prohibitive for long sequences and risks propagating misleading certainty from unrelated tokens, potentially leading to underestimation of CoT quality. An alternative is to apply a position-based discounting scheme – down-weighting the contribution of later tokens in the reference outcome under the assumption that they benefit more from cumulative context. Yet this strategy introduces a different failure mode: reasoning-reflective tokens that appear later in the sequence may be unfairly penalized, while non-informative early tokens are disproportionately emphasized.

To address these issues, we adopt a more targeted solution that centers around the reasoning-reflective tokens. Our insight is that for poor CoT traces, a reasoning-reflective token is likely to receive low model confidence (i.e., probability). When the reference sequence “corrects” this token – appending it during likelihood computation for next tokens – it begins to influence subsequent tokens, effectively initiating a chain of error compensation. We leverage this observation by introducing controlled self-certainty propagation, which begins at reasoning-reflective tokens and decays over a localized window of subsequent tokens. Formally, for each reasoning-reflective token at position kk, we define a propagation factor: Pkp​r​o​p​(j)=pkR​R​T+(1−pkR​R​T)​(1−e−γ​d)P^{prop}\_{k}(j)=p^{RRT}\_{k}+(1-p^{RRT}\_{k})(1-e^{-\gamma d}) where pkR​R​Tp^{RRT}\_{k} is the self-certainty (probability) of kt​hk^{th} reflection token, dd is the distance from the reflection token to the current token jj, and γ\gamma is a hyperparameter controlling the propagation decay from kt​hk^{th} token. The final reward formulation incorporates both variance-based token weighting and propagation-aware correction: ∑j=1|y|wΔ​(σj)​log⁡(π​(yj∣q,ci^,y<j)​Πk<j​Pkp​r​o​p​(j))\sum\_{j=1}^{\lvert\textbf{y}\rvert}w\_{\Delta}(\sigma\_{j})\log\left(\pi(y\_{j}\mid\textbf{q},\hat{\textbf{c}\_{i}},\textbf{y}\_{<j})\Pi\_{k<j}{P^{prop}\_{k}}(j)\right).

While the targeted decay-based propagation approach is effective when the number of reasoning-reflective tokens is small, it becomes computationally expensive as their proportion increases within the reference outcome. To address this, we propose a more efficient alternative for estimating the self-influence of reference tokens. Specifically, we compute the log-probabilities of reference tokens conditioned on a masked CoT trace, which serves as a baseline estimate of token-level influence originating from the reference itself. For instance, in the earlier football example, the token “Messi” is still likely to receive a high probability due to the presence of the preceding token “Lionel”, even when no reasoning is provided. By subtracting these masked-CoT log-probabilities from those computed with the model-generated CoT, we isolate the self-induced certainty boost by reference tokens. Then, the reward formulation becomes: ∑j=1|y|wΔ​(σj)​[log⁡(π​(yj∣q,ci^,y<j))−log⁡(π​(yj∣q,cmasked,y<j))]\sum\_{j=1}^{\lvert\textbf{y}\rvert}w\_{\Delta}(\sigma\_{j})\left[\log\left(\pi(y\_{j}\mid\textbf{q},\hat{\textbf{c}\_{i}},\textbf{y}\_{<j})\right)-\log\left(\pi(y\_{j}\mid\textbf{q},{\textbf{c}\_{\text{masked}}},\textbf{y}\_{<j})\right)\right].

### 4.2 Direct Reasoning Optimization with R3

We now introduce Direct Reasoning Optimization (DRO), a RL-based fine-tuning framework that that employs R3 as its primary reward signal for guiding reasoning quality and dynamic data filtering for open-ended reasoning tasks.

#### 4.2.1 Design of DRO

Fig.[2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks") presents an overview of DRO, where the model learns to optimize its own reasoning through direct intrenal reward feedback. DRO builds upon the GRPO framework(Shao et al., [2024](#bib.bib38)), which aligns with the group-relative and comparative nature of our core reward, R3. Given a prompt q, the actor policy πθ\pi\_{\theta} generates a group of outputs, each comprising a CoT trace c^i\hat{\textbf{c}}\_{i} followed by a final outcome y^i\hat{\textbf{y}}\_{i}. We replace y^i\hat{\textbf{y}}\_{i} with the ground-truth reference outcome y to compute the R3i score for each c^i\hat{\textbf{c}}\_{i}. To evaluate R3, we use an internal policy πrwd\pi\_{\text{rwd}}, instantiated in three variants: (1) statically using the reference policy πref\pi\_{\text{ref}}, (2) dynamically syncing with πθ\pi\_{\theta}, and (3) using a lagged version of πθ\pi\_{\theta}. Since R3 only scores the reasoning trace and not the generated final outcome, we observed that models tend to produce verbose completions, e.g., appending explanations at the end of revised paragraph in the ParaRev task. To mitigate this, we apply a length penalty solely on the final outcome: rlenβ​(y^,y):=1−β⋅||y|−|y^|||y|r\_{\text{len}}^{\beta}(\hat{\textbf{y}},\textbf{y}):=1-\beta\cdot\frac{|\,|\textbf{y}|-|\hat{\textbf{y}}|\,|}{|\textbf{y}|}, where β\beta controls the strength of the penalty. The final reward is a weighted combination of R3i and the length penalty, which is used to compute the advantage (Eq.[2](#S3.E2 "In 3 Background: Reasoning Optimization with RL ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks")). This advantage is then used in the GRPO objective (Eq.[3](#S3.Ex1 "3 Background: Reasoning Optimization with RL ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks")) to update the model parameters.

#### 4.2.2 Data Filtering and Efficiency

Recent work (Meta, [2025](#bib.bib34); Muennighoff et al., [2025](#bib.bib36); Jiang et al., [2025](#bib.bib19); Ye et al., [2025](#bib.bib50); Yang et al., [2025](#bib.bib49); Costello et al., [2025](#bib.bib10)) highlights the critical role of data filtering in reinforcement learning, demonstrating its impact on both data efficiency and downstream task performance. These approaches typically rely on either LLM-as-a-judge frameworks or verifiable reward signals. However, in open-ended reasoning tasks where no reliable verifiers exist, such strategies are not applicable. Moreover, using LLM-as-a-judge would require designing task and dataset-specific prompts, compounding the complexity and inheriting the limitations discussed earlier. To address this, DRO introduces a generic, dynamic data filtering mechanism tailored for open-ended reasoning tasks leveraging R3, enhancing data efficiency during RL-based training without the need for manual prompt engineering or external verification.

DRO performs data filtering at regular intervals throughout training, beginning with an initial filtering round before the start of training. Each filtering round is guided by the current policy model (πθ\pi\_{\theta}) and is conducted in two stages:

* •

  Filtering Out Beyond-knowledge or Extremely difficult Questions:  The latent presence of prerequisite knowledge within the model largely determines the effectiveness of RL-based reasoning training (Ye et al., [2025](#bib.bib50); Snell et al., [2024](#bib.bib41); Yue et al., [2025](#bib.bib52)). Accordingly, we begin by filtering out questions that are either excessively difficult or likely beyond the model’s current knowledge. For each question/prompt, we sample NN CoT reasoning traces (typically 16 or 32, depending on the setup) using the current policy πθ\pi\_{\theta}. As in R3, we evaluate self-certainty over the reference outcome tokens conditioned on each CoT trace. Instead of log-probabilities, we use token rank (yj(r)y\_{j}^{(r)}) as a proxy for prediction difficulty. For each CoT trace c​i\textbf{c}i, we sort the tokens by rank and compute the average of the bottom ρ%\rho\% (i.e., highest-ranked, least-confident) tokens: a​v​g​(m​a​x​ρi​yj(r))avg(max{\rho}^{i}{y\_{j}^{(r)}}). This average reflects how difficult the reference tokens are to predict given the sampled reasoning. If, across all NN generations for a given question, none achieve a sufficiently low a​v​g​(m​a​xρi​yj(r))avg(max\_{\rho}^{i}{y\_{j}^{(r)}}) ,i.e., within a predefined top-kk threshold, we consider the question either too difficult or outside the model’s current knowledge scope and exclude it from this training round.
* •

  Filtering Out Questions with Low Reasoning Variation:  In the second stage, we filter out questions that exhibit low variation in the reasoning space, which typically corresponds to overly simple questions (assuming the previous stage has already removed most overly difficult ones). We leverage the R3 scores computed in the prior step using the current policy πθ\pi\_{\theta}. Specifically, for each prompt, we compute the maximum per-token standard deviation across NN sampled CoT traces: max⁡(σj)\max(\sigma\_{j}). This value captures the highest degree of reasoning-induced variability in reference token predictions. We then rank all prompts in descending order of max⁡(σj)\max(\sigma\_{j}) and remove a proportion of the lowest-ranked samples. The cutoff is determined based on the available training data size and the model’s capacity.

In each round of filtering, we carry forward 10% of data from the previous training set.

## 5 Experiments

### 5.1 Experimental Setup

Datasets.
We use the following datasets in our experiments:
(1) ParaRev (Jourdan et al., [2025](#bib.bib21)): This dataset contains over 48K original-revised paragraph pairs from scientific papers on OpenReview, along with corresponding reviews. Since many papers undergo multiple revisions, we focus on the initial revision, as it typically reflects the most substantial changes in response to reviewer feedback. As ParaRev does not include the full paper context for each paragraph, which is crucial for reasoning, we extend the dataset by locating the paragraphs in the raw papers and extracting their preceding and following context from CASIMIR (Jourdan et al., [2024](#bib.bib20)). This results in an adapted dataset of 4.8K samples, and we follow a 95%/5% train-test split.
(2) FinQA (Chen et al., [2021](#bib.bib8)): A dataset focused on numerical reasoning over financial data, comprising over 8K samples with expert-written context, questions, reasoning programs, and answers. For our RL training, we use only the context, questions, and answers, adhering to the original train-test split.

Training.
We conduct DRO training on the DeepSeek-R1-Distill-Qwen-7B and 14B models. A learning rate of 1.0×e−61.0\times e^{-6} is used with a warmup ratio of 0.2 and a “constant with warmup” learning rate scheduler. During each training step, the actor model generates 16 responses per question using a temperature of 1.0, top-p sampling with p=0.95p=0.95, a repetition penalty of 1.0, and a maximum completion length of 10,000 tokens for FinQA and 8,000 tokens for ParaRev. We process 256 questions per step for FinQA and 128 for ParaRev. For GRPO optimization, we adopt the loss function from Liu et al. ([2025a](#bib.bib30)), using scaled rewards, masking for truncated completions, and an upper clipping coefficient of ϵhigh=0.2\epsilon\_{\text{high}}=0.2. While prior studies typically set the entropy regularization weight β=0\beta=0, we empirically found β=0.001\beta=0.001 to improve training stability and convergence. Training is conducted across three nodes, each with 8×8\times NVIDIA A100 GPUs. We utilize HuggingFace TRL for reinforcement learning, DeepSpeed for distributed training, and vLLM for rollout generation and R3 computation.

Metrics.
For the FinQA task, where answers are verifiable, we use numerical correctness with a 2% tolerance. For the ParaRev task, we adopt pairwise win rate as the primary evaluation metric. To compute win rates, we adapt the AlpacaEval prompt to the revision setting by providing the paper context, reviewer comments, original paragraph, and reference revision for each sample. Our validation indicates that this prompt yields a 94.6% win rate for expert revisions over GPT-4o revisions, demonstrating strong alignment with human preferences. The full prompt template is provided in Appendix [A](#A1 "Appendix A Adapted Prompt Template for ParaRev Win-Rate Evaluation ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks"). To mitigate potential self-enhancement bias (Zheng et al., [2023](#bib.bib58)), we use both GPT-4o and Claude 3.7 Sonnet as judges.

Baselines.
We mainly compare DRO with the following baselines in our evaluation:
(1) Base Models: The off-the-shelf DeepSeek-R1-Distill-Qwen-7B (for FinQA) and 14B (for ParaRev) models without RL on the specific tasks.
(2) ROUGE (ParaRev): For ParaRev, although the outcomes are not directly verifiable, we use ROUGE-1 F1 score (Lin, [2004](#bib.bib28)) as the reward in GRPO to represent RL with a standard automatic metric as a proxy verifier.
(3) Correctness (FinQA): For FinQA, where outputs are math-like and easily verifiable, we use binary correctness (within a 2% tolerance) as the reward in GRPO to serve as an upper bound where ideal outcome verification is feasible.
(4) Aggregate: To assess the efficacy of R3, we include a set of baselines that use the aggregate certainty across all tokens as the reward. As these baselines share the same training workflow as DRO, we denote them as DRO-Aggr. Specifically for ParaRev, we introduce DRO-Aggr-S and DRO-Aggr-R to represent strict and relaxed length control, respectively, each using different β\beta in the length reward to study its impact.
(5) GPT-4o: A strong baseline using a significantly larger model.

### 5.2 Results

#### 5.2.1 ParaRev

|  |  |  |  |
| --- | --- | --- | --- |
| Model | Win Rate vs. GPT-4o | | Length |
| GPT Judge | Claude Judge |
| DeepSeek-R1-Distill-Qwen-14B (ROUGE) | 31.1 | 42.8 | 570 |
| DeepSeek-R1-Distill-Qwen-14B (DRO-Aggr-S) | 31.1 | 44.0 | 587 |
| DeepSeek-R1-Distill-Qwen-14B (Base) | 43.8 | 48.6 | 1095 |
| DeepSeek-R1-Distill-Qwen-14B (DRO-Aggr-R) | 47.9 | 51.0 | 1038 |
| GPT-4o | (50) | (50) | 889 |
| DeepSeek-R1-Distill-Qwen-14B (DRO-R3) | 51.8 | 58.8 | 743 |
| Original Paragraph (No Revision) | 13.2 | 23.0 | 545 |
| Reference Revision | 94.6 | 100.0 | 613 |

Table 1: Win rates against GPT-4o on ParaRev

DRO with R3 improves reasoning quality and alignment.
As shown in Table [1](#S5.T1 "Table 1 ‣ 5.2.1 ParaRev ‣ 5.2 Results ‣ 5 Experiments ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks"), DRO-R3 achieves higher win rates against GPT-4o than all other variants, outperforming the base model by 8.0% (GPT judge) and 10.2% (Claude judge), and even surpassing GPT-4o itself despite being a much smaller model. It also generates outputs with lengths closer to the reference revisions, indicating more faithful and efficient edits. Given the known length bias in LLM-based evaluators (Zheng et al., [2023](#bib.bib58)), this improvement further reflects better alignment with human preference.

R3 outperforms ROUGE-based rewards.
Compared to the ROUGE-rewarded baseline, R3 yields a win rate improvement of 16.0% (GPT judge) and 20.7% (Claude judge). We observe that the ROUGE-trained model frequently leaves the paragraph unchanged, likely due to the reward favoring textual overlap, resulting in shorter outputs similar in length to the original paragraph. This behavior harms revision quality.

R3 also outperforms aggregate-certainty rewards.
Compared to aggregated certainty rewards, R3 leads to consistently higher win rates regardless of length control settings. Against the same base model, DRO-R3 achieves up to a 4.25×4.25\times improvement over DRO-Aggr-R, highlighting the importance of reasoning-reflective token weighting. Furthermore, strict length control (DRO-Aggr-S) degrades performance, suggesting that rigid enforcement of output length may suppress effective reasoning and degrade revision quality.

!(/html/2506.13351/assets/x3.png)

(a) Aggregate vs. R3

!(/html/2506.13351/assets/x4.png)

(b) Rouge

!(/html/2506.13351/assets/x5.png)

(c) Filtering

Figure 3: ParaRev training insights.

Training insights.
(1) R3 stimulates longer reasoning generation: As shown in Figure [3(a)](#S5.F3.sf1 "In Figure 3 ‣ 5.2.1 ParaRev ‣ 5.2 Results ‣ 5 Experiments ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks"), R3 encourages the model to produce longer CoTs, with generation length growing steadily from 1k to over 2.5k tokens during training. In contrast, aggregate-certainty rewards lead to early collapse below 100 tokens, as the model learns to omit reasoning due to the misleading reward signal.
(2) Implicit improvement in textual similarity: Figure [3(b)](#S5.F3.sf2 "In Figure 3 ‣ 5.2.1 ParaRev ‣ 5.2 Results ‣ 5 Experiments ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks") shows that, despite ROUGE not being part of the reward, DRO with R3 substantially improves ROUGE-L F1 from 0.40.4 to 0.70.7 in the early stage of training, suggesting that optimization toward reasoning-reflective tokens also results in better surface-level alignment.
(3) Filtering accelerates and stabilizes training: As shown in Figure [3(c)](#S5.F3.sf3 "In Figure 3 ‣ 5.2.1 ParaRev ‣ 5.2 Results ‣ 5 Experiments ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks"), on-the-fly data filtering in DRO reduces training time by 45%45\% while achieving comparable final reward scores and smoother convergence, demonstrating its efficiency and robustness.

### 5.3 FinQA

| Model | Pass@k | | | | |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 4 | 8 | 16 |
| DeepSeek-R1-Distill-Qwen-7B (Base) | 61.7 | 70.7 | 75.9 | 79.3 | 81.5 |
| DeepSeek-R1-Distill-Qwen-7B (DRO-Aggr) | 63.0 | 72.5 | 77.4 | 80.4 | 82.6 |
| DeepSeek-R1-Distill-Qwen-7B (DRO-R3) | 67.1 | 74.2 | 78.3 | 81.0 | 82.5 |
| DeepSeek-R1-Distill-Qwen-7B (Correctness) | 68.0 | 73.8 | 77.9 | 80.4 | 82.1 |
| GPT-4o | 69.5 | 73.9 | 76.2 | 78.1 | 79.3 |

Table 2: Pass@k on FinQA

R3 achieves improvement comparable to rewards from an ideal verifier.
On FinQA, a math-like task with reliably verifiable outcomes, DRO-R3 achieves gains comparable to those obtained using correctness-based rewards. Specifically, as shown in Table [2](#S5.T2 "Table 2 ‣ 5.3 FinQA ‣ 5 Experiments ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks"), it falls only 0.9%0.9\% short on Pass@1 but outperforms the correctness baseline on Pass@k for k≥2k\geq 2. This result highlights that R3 can match the benefits of correctness-based rewards without access to a reliable verifier, demonstrating its potential for tasks where ideal outcome verification is difficult to obtain or not well-defined.

R3 outperforms aggregate-certainty rewards even in short-outcome tasks.
Although FinQA involves relatively short outputs where most tokens appear to contribute directly to the final answer, R3 still outperforms the aggregate-certainty reward. Compared to the base model, DRO-R3 achieves a 4.15×4.15\times higher improvement than DRO-Aggr. This indicates that reasoning-reflective tokens are not exclusive to long-form generation. For example, in math-like tasks, tokens such as the decimal point “.” may reflect reasoning quality more than trailing digits.

!(/html/2506.13351/assets/x6.png)

(a) R3

!(/html/2506.13351/assets/x7.png)

(b) Standard deviation of R3

!(/html/2506.13351/assets/x8.png)

(c) Generation length

Figure 4: FinQA training insights.

Training insights.
(1) Steady reward improvement and stabilization: As shown in Figures [4(a)](#S5.F4.sf1 "In Figure 4 ‣ 5.3 FinQA ‣ 5 Experiments ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks") and [4(b)](#S5.F4.sf2 "In Figure 4 ‣ 5.3 FinQA ‣ 5 Experiments ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks"), DRO consistently improves the R3 reward while reducing its standard deviation across sampled reasoning traces, indicating both stronger and more stable reward attribution over time.
(2) Emergence of longer reasoning: Generation length steadily increases from 1k to over 3k tokens (Figure [4(c)](#S5.F4.sf3 "In Figure 4 ‣ 5.3 FinQA ‣ 5 Experiments ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks")). Interestingly, while the R3 improvement slows around step 6 (Figure [4(a)](#S5.F4.sf1 "In Figure 4 ‣ 5.3 FinQA ‣ 5 Experiments ‣ Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks")), the reasoning length continues to grow almost linearly. This divergence suggests that as the reward signal begins to saturate, the model continues to elaborate its reasoning, potentially exploring richer explanations or extended self-reflection beyond what R3 explicitly rewards. This behavior remains effective, as the R3 continues to improve gradually thereafter.

## 6 Conclusion

We introduced Direct Reasoning Optimization (DRO), a reinforcement learning framework designed to optimize CoT reasoning in open-ended, particularly long-form, reasoning tasks using a new reward signal, the Reasoning Reflection Reward (R3). By directly leveraging the model’s own self-certainty over reference outcome tokens conditioned on a CoT reasoning, R3 captures preceding reasoning effectiveness without relying on external evaluators or handcrafted reward models. DRO builds on the GRPO framework and incorporates dynamic data filtering driven by R3, enhancing both training efficiency and downstream performance. Our experiments on ParaRev and FinQA demonstrate that DRO not only generalizes beyond math-style tasks to open-ended reasoning problems but also significantly reduces training cost while maintaining or surpassing task-specific baselines. This work highlights the promise of self-supervised reward design in enabling scalable, outcome-driven reasoning optimization for LLMs.

## References

* Atreya (2024)

  Mohan Atreya.
  Fine-Tuning AI Models with Tuning-as-a-Service Platforms, 2024.
* Bachmann & Nagarajan (2024)

  Gregor Bachmann and Vaishnavh Nagarajan.
  The pitfalls of next-token prediction.
  *arXiv preprint arXiv:2403.06963*, 2024.
* Chen et al. (2024)

  Haolin Chen, Yihao Feng, Zuxin Liu, Weiran Yao, Akshara Prabhakar, Shelby
  Heinecke, Ricky Ho, Phil Mui, Silvio Savarese, Caiming Xiong, et al.
  Language models are hidden reasoners: Unlocking latent reasoning
  capabilities via self-rewarding.
  *arXiv preprint arXiv:2411.04282*, 2024.
* Chen et al. (2025a)

  Kedi Chen, Qin Chen, Jie Zhou, Xinqi Tao, Bowen Ding, Jingwen Xie, Mingchen
  Xie, Peilong Li, and Zheng Feng.
  Enhancing uncertainty modeling with semantic graph for hallucination
  detection.
  In *Proceedings of the AAAI Conference on Artificial
  Intelligence*, volume 39, pp.  23586–23594, 2025a.
* Chen et al. (2025b)

  Nuo Chen, Zhiyuan Hu, Qingyun Zou, Jiaying Wu, Qian Wang, Bryan Hooi, and
  Bingsheng He.
  Judgelrm: Large reasoning models as a judge.
  *arXiv preprint arXiv:2504.00050*, 2025b.
* Chen et al. (2025c)

  Xiusi Chen, Gaotang Li, Ziqi Wang, Bowen Jin, Cheng Qian, Yu Wang, Hongru Wang,
  Yu Zhang, Denghui Zhang, Tong Zhang, et al.
  Rm-r1: Reward modeling as reasoning.
  *arXiv preprint arXiv:2505.02387*, 2025c.
* Chen et al. (2025d)

  Yanda Chen, Joe Benton, Ansh Radhakrishnan, Jonathan Uesato, Carson Denison,
  John Schulman, Arushi Somani, Peter Hase, Misha Wagner, Fabien Roger, et al.
  Reasoning models don’t always say what they think.
  *arXiv preprint arXiv:2505.05410*, 2025d.
* Chen et al. (2021)

  Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan
  Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, et al.
  Finqa: A dataset of numerical reasoning over financial data.
  *arXiv preprint arXiv:2109.00122*, 2021.
* Christiano et al. (2017)

  Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario
  Amodei.
  Deep reinforcement learning from human preferences.
  *Advances in neural information processing systems*, 30, 2017.
* Costello et al. (2025)

  Caia Costello, Simon Guo, Anna Goldie, and Azalia Mirhoseini.
  Think, prune, train, improve: Scaling reasoning without scaling
  models.
  *arXiv preprint arXiv:2504.18116*, 2025.
* Ethayarajh et al. (2024)

  Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela.
  Kto: Model alignment as prospect theoretic optimization.
  *arXiv preprint arXiv:2402.01306*, 2024.
* Gu et al. (2024)

  Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei
  Li, Yinghan Shen, Shengjie Ma, Honghao Liu, et al.
  A survey on llm-as-a-judge.
  *arXiv preprint arXiv:2411.15594*, 2024.
* Guo et al. (2025)

  Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu,
  Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al.
  Deepseek-r1: Incentivizing reasoning capability in llms via
  reinforcement learning.
  *arXiv preprint arXiv:2501.12948*, 2025.
* Gupta et al. (2024)

  Neha Gupta, Harikrishna Narasimhan, Wittawat Jitkrittum, Ankit Singh Rawat,
  Aditya Krishna Menon, and Sanjiv Kumar.
  Language model cascades: Token-level uncertainty and beyond.
  *arXiv preprint arXiv:2404.10136*, 2024.
* Hu et al. (2025)

  Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung
  Shum.
  Open-reasoner-zero: An open source approach to scaling up
  reinforcement learning on the base model.
  *arXiv preprint arXiv:2503.24290*, 2025.
* Huang et al. (2024)

  Chenghua Huang, Zhizhen Fan, Lu Wang, Fangkai Yang, Pu Zhao, Zeqi Lin, Qingwei
  Lin, Dongmei Zhang, Saravan Rajmohan, and Qi Zhang.
  Self-evolved reward learning for llms.
  *arXiv preprint arXiv:2411.00418*, 2024.
* Huang & Chang (2022)

  Jie Huang and Kevin Chen-Chuan Chang.
  Towards reasoning in large language models: A survey.
  *arXiv preprint arXiv:2212.10403*, 2022.
* Jaech et al. (2024)

  Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden
  Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al.
  Openai o1 system card.
  *arXiv preprint arXiv:2412.16720*, 2024.
* Jiang et al. (2025)

  Pengcheng Jiang, Xueqiang Xu, Jiacheng Lin, Jinfeng Xiao, Zifeng Wang, Jimeng
  Sun, and Jiawei Han.
  s3: You don’t need that much data to train a search agent via rl.
  *arXiv preprint arXiv:2505.14146*, 2025.
* Jourdan et al. (2024)

  Léane Jourdan, Florian Boudin, Nicolas Hernandez, and Richard Dufour.
  Casimir: A corpus of scientific articles enhanced with multiple
  author-integrated revisions.
  *arXiv preprint arXiv:2403.00241*, 2024.
* Jourdan et al. (2025)

  Léane Jourdan, Nicolas Hernandez, Richard Dufour, Florian Boudin, and Akiko
  Aizawa.
  Pararev: Building a dataset for scientific paragraph revision
  annotated with revision instruction.
  *arXiv preprint arXiv:2501.05222*, 2025.
* Kauf et al. (2024)

  Carina Kauf, Emmanuele Chersoni, Alessandro Lenci, Evelina Fedorenko, and
  Anna A Ivanova.
  Log probabilities are a reliable estimate of semantic plausibility in
  base and instruction-tuned language models.
  *arXiv preprint arXiv:2403.14859*, 2024.
* Kavukcuoglu, Koray (2025)

  Kavukcuoglu, Koray.
  Gemini 2.5: Our most intelligent AI model.
  <https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/>,
  2025.
  Accessed: 2025-06-02.
* Kojima et al. (2022)

  Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke
  Iwasawa.
  Large language models are zero-shot reasoners.
  *Advances in neural information processing systems*,
  35:22199–22213, 2022.
* Lambert et al. (2024)

  Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish
  Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane
  Lyu, et al.
  T\\backslash” ulu 3: Pushing frontiers in open language model
  post-training.
  *arXiv preprint arXiv:2411.15124*, 2024.
* Lee et al. (2023)

  Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Ren Lu, Thomas Mesnard,
  Johan Ferret, Colton Bishop, Ethan Hall, Victor Carbune, and Abhinav Rastogi.
  Rlaif: Scaling reinforcement learning from human feedback with ai
  feedback.
  2023.
* Lightman et al. (2023)

  Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker,
  Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe.
  Let’s verify step by step.
  In *The Twelfth International Conference on Learning
  Representations*, 2023.
* Lin (2004)

  Chin-Yew Lin.
  Rouge: A package for automatic evaluation of summaries.
  In *Text summarization branches out*, pp.  74–81, 2004.
* Liu & Zhang (2025)

  Jiawei Liu and Lingming Zhang.
  Code-r1: Reproducing r1 for code with reliable rewards.
  *arXiv preprint arXiv:2503.18470*, 3, 2025.
* Liu et al. (2025a)

  Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun
  Lee, and Min Lin.
  Understanding r1-zero-like training: A critical perspective.
  *arXiv preprint arXiv:2503.20783*, 2025a.
* Liu et al. (2025b)

  Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu,
  and Yu Wu.
  Inference-time scaling for generalist reward modeling.
  *arXiv preprint arXiv:2504.02495*, 2025b.
* Lu (2025)

  Xun Lu.
  Writing-zero: Bridge the gap between non-verifiable problems and
  verifiable rewards.
  *arXiv preprint arXiv:2506.00103*, 2025.
* (33)

  Michael Luo, Sijun Tan, Roy Huang, Ameen Patel, Alpay Ariyak, Qingyang Wu,
  Xiaoxiang Shi, Rachel Xin, Colin Cai, Maurice Weber, et al.
  Deepcoder: A fully open-source 14b coder at o3-mini level, 2025.
  *Notion Blog*, 3(4):6.
* Meta (2025)

  Meta.
  The Llama 4 herd: The beginning of a new era of natively multimodal
  AI innovation.
  <https://ai.meta.com/blog/llama-4-multimodal-intelligence/>,
  2025.
  Accessed: 2025-06-02.
* Microsoft (2024)

  Microsoft.
  Microsoft 365 Copilot Tuning overview (preview), 2024.
  URL
  <https://learn.microsoft.com/en-us/copilot/microsoft-365/copilot-tuning-overview>.
* Muennighoff et al. (2025)

  Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei,
  Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and
  Tatsunori Hashimoto.
  s1: Simple test-time scaling.
  *arXiv preprint arXiv:2501.19393*, 2025.
* Schulman et al. (2017)

  John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
  Proximal policy optimization algorithms.
  *arXiv preprint arXiv:1707.06347*, 2017.
* Shao et al. (2024)

  Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei
  Zhang, Mingchuan Zhang, YK Li, Y Wu, et al.
  Deepseekmath: Pushing the limits of mathematical reasoning in open
  language models.
  *arXiv preprint arXiv:2402.03300*, 2024.
* Sharma et al. (2024)

  Archit Sharma, Sedrick Scott Keh, Eric Mitchell, Chelsea Finn, Kushal Arora,
  and Thomas Kollar.
  A critical evaluation of ai feedback for aligning large language
  models.
  *Advances in Neural Information Processing Systems*,
  37:29166–29190, 2024.
* Silver et al. (2017)

  David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja
  Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton,
  et al.
  Mastering the game of go without human knowledge.
  *nature*, 550(7676):354–359, 2017.
* Snell et al. (2024)

  Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar.
  Scaling llm test-time compute optimally can be more effective than
  scaling model parameters.
  *arXiv preprint arXiv:2408.03314*, 2024.
* Stiennon et al. (2020)

  Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea
  Voss, Alec Radford, Dario Amodei, and Paul F Christiano.
  Learning to summarize with human feedback.
  *Advances in neural information processing systems*,
  33:3008–3021, 2020.
* Su et al. (2025)

  Yi Su, Dian Yu, Linfeng Song, Juntao Li, Haitao Mi, Zhaopeng Tu, Min Zhang, and
  Dong Yu.
  Crossing the reward bridge: Expanding rl with verifiable rewards
  across diverse domains.
  *arXiv preprint arXiv:2503.23829*, 2025.
* Tang et al. (2025)

  Yunhao Tang, Sid Wang, and Rémi Munos.
  Learning to chain-of-thought with jensen’s evidence lower bound.
  *arXiv preprint arXiv:2503.19618*, 2025.
* Varshney et al. (2023)

  Neeraj Varshney, Wenlin Yao, Hongming Zhang, Jianshu Chen, and Dong Yu.
  A stitch in time saves nine: Detecting and mitigating hallucinations
  of llms by validating low-confidence generation.
  *arXiv preprint arXiv:2307.03987*, 2023.
* Wei et al. (2022)

  Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V
  Le, Denny Zhou, et al.
  Chain-of-thought prompting elicits reasoning in large language
  models.
  *Advances in neural information processing systems*,
  35:24824–24837, 2022.
* Xu et al. (2025a)

  Fangzhi Xu, Hang Yan, Chang Ma, Haiteng Zhao, Qiushi Sun, Kanzhi Cheng, Junxian
  He, Jun Liu, and Zhiyong Wu.
  Genius: A generalizable and purely unsupervised self-training
  framework for advanced reasoning.
  *arXiv preprint arXiv:2504.08672*, 2025a.
* Xu et al. (2025b)

  Yifei Xu, Tusher Chakraborty, Emre Kıcıman, Bibek Aryal, Eduardo
  Rodrigues, Srinagesh Sharma, Roberto Estevao, Maria Angels de Luis Balaguer,
  Jessica Wolk, Rafael Padilha, et al.
  Rlthf: Targeted human feedback for llm alignment.
  *arXiv preprint arXiv:2502.13417*, 2025b.
* Yang et al. (2025)

  An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen
  Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al.
  Qwen3 technical report.
  *arXiv preprint arXiv:2505.09388*, 2025.
* Ye et al. (2025)

  Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu.
  Limo: Less is more for reasoning.
  *arXiv preprint arXiv:2502.03387*, 2025.
* Yu et al. (2025)

  Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian
  Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al.
  Dapo: An open-source llm reinforcement learning system at scale.
  *arXiv preprint arXiv:2503.14476*, 2025.
* Yue et al. (2025)

  Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao
  Huang.
  Does reinforcement learning really incentivize reasoning capacity in
  llms beyond the base model?
  *arXiv preprint arXiv:2504.13837*, 2025.
* Zelikman et al. (2022)

  Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman.
  Star: Bootstrapping reasoning with reasoning.
  *Advances in Neural Information Processing Systems*,
  35:15476–15488, 2022.
* Zeng et al. (2024)

  Zhiyuan Zeng, Qinyuan Cheng, Zhangyue Yin, Bo Wang, Shimin Li, Yunhua Zhou,
  Qipeng Guo, Xuanjing Huang, and Xipeng Qiu.
  Scaling of search and learning: A roadmap to reproduce o1 from
  reinforcement learning perspective.
  *arXiv preprint arXiv:2412.14135*, 2024.
* Zhang et al. (2022)

  Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola.
  Automatic chain of thought prompting in large language models.
  *arXiv preprint arXiv:2210.03493*, 2022.
* Zhao et al. (2025a)

  Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Matthieu Lin, Shenzhi
  Wang, Qingyun Wu, Zilong Zheng, and Gao Huang.
  Absolute zero: Reinforced self-play reasoning with zero data.
  *arXiv preprint arXiv:2505.03335*, 2025a.
* Zhao et al. (2025b)

  Xuandong Zhao, Zhewei Kang, Aosong Feng, Sergey Levine, and Dawn Song.
  Learning to reason without external rewards.
  *arXiv preprint arXiv:2505.19590*, 2025b.
* Zheng et al. (2023)

  Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao
  Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al.
  Judging llm-as-a-judge with mt-bench and chatbot arena.
  *Advances in Neural Information Processing Systems*,
  36:46595–46623, 2023.
* Zhou et al. (2025)

  Xiangxin Zhou, Zichen Liu, Anya Sims, Haonan Wang, Tianyu Pang, Chongxuan Li,
  Liang Wang, Min Lin, and Chao Du.
  Reinforcing general reasoning without verifiers.
  *arXiv preprint arXiv:2505.21493*, 2025.
* Zuo et al. (2025)

  Yuxin Zuo, Kaiyan Zhang, Shang Qu, Li Sheng, Xuekai Zhu, Biqing Qi, Youbang
  Sun, Ganqu Cui, Ning Ding, and Bowen Zhou.
  Ttrl: Test-time reinforcement learning.
  *arXiv preprint arXiv:2504.16084*, 2025.

## Appendix A Adapted Prompt Template for ParaRev Win-Rate Evaluation

<|im\_start|>user
I want you to create a leaderboard of different large-language models based on the quality of their revisions to a given paragraph of a scientific paper. To do so, I will give you the paper context, reviews, paragraph to revise, golden revision written by human experts, and revisions output by the models. Please rank the models based on which revision would align better with the golden revision written by human experts. Note that alignment should be evaluated based on how effectively the concerns are addressed, rather than on textual similarity. All inputs and outputs should be python dictionaries.
  
Here are the paper context, reviews, paragraph to revise, and golden revision:
  
## Paper Context
  
{paper\_context}
  
## Reviews
  
{reviews}
  
## Paragraph to Revise
  
{paragraph\_to\_revise}
  
## Golden Revision
  
{golden\_revision}
  
Here are the outputs of the models:
[
{
"model": "model\_1",
"revision": """{output\_1}"""
},
{
"model": "model\_2",
"revision": """{output\_2}"""
}
]
  
Now please rank the models by the quality of their revisions, so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:
[
{’model’: <model-name>, ’rank’: <model-rank>},
{’model’: <model-name>, ’rank’: <model-rank>}
]
  
Your response must be a valid Python dictionary and should contain nothing else because we will directly execute it in Python. DO NOT include formatting characters like ‘‘‘python‘‘‘. Please provide the ranking that the majority of humans would give.
<|im\_end|>
