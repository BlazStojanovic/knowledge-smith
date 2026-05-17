---
arxiv: '2603.12273'
authors:
- Thomas Kleine Buening
- Jonas Hübotter
- Barna Pásztor
- Idan Shenfeld
- Giorgia Ramponi
- Andreas Krause
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Aligning Language Models from User Interactions
url: https://arxiv.org/abs/2603.12273
year: 2026
---

# Aligning Language Models from User Interactions

Thomas Kleine Buening11   Jonas Hübotter11   Barna Pásztor11  
  
Idan Shenfeld22   Giorgia Ramponi33   Andreas Krause11
  
11ETH Zurich   22MIT   33University of Zurich

###### Abstract

Multi-turn user interactions are among the most abundant data produced by language models, yet we lack effective methods to learn from them. While typically discarded, these interactions often contain useful information: follow-up user messages may indicate that a response was incorrect, failed to follow an instruction, or did not align with the user’s preferences. Importantly, language models are already able to make use of this information in context. After observing a user’s follow-up, the same model is often able to revise its behavior. We leverage this ability to propose a principled and scalable method for learning directly from user interactions through self-distillation. By conditioning the model on the user’s follow-up message and comparing the resulting token distribution with the original policy, we obtain a target for updating the policy that captures how the model’s behavior changes in hindsight. We then distill this hindsight distribution back into the current policy. Remarkably, we show that training on real-world user conversations from WildChat improves language models across standard alignment and instruction-following benchmarks, without regressing other capabilities. The same mechanism enables personalization, allowing models to continually adapt to individual users through interaction without explicit feedback. Our results demonstrate that raw user interactions that arise naturally during deployment enable alignment, personalization, and continual adaptation.

## Introduction

In modern language models, inference has overtaken training as the dominant consumer of compute, with models serving massive volumes of user queries every day. Yet the information revealed through these interactions is typically discarded and does not contribute to improving the model itself, representing a significant missed opportunity.
At scale, users engage in extended conversations, refining prompts, requesting revisions, and responding directly to model outputs. These interactions are rich with implicit learning signals: follow-up messages may indicate that a response was incorrect, failed to follow an instruction, or did not align with the user’s preferences (Don-Yehiya et al., [2024](#bib.bib80 "Naturally occurring feedback is common, extractable and useful")). For example, a user may report an error after executing generated code, point out that a required format was not followed, or ask for a response to be rewritten in a different style or tone. Such signals arise organically during normal use and reflect how model outputs are received and acted upon during deployment. Finding ways to leverage this data source can open the door to continual learning from deployment at an unprecedented scale.

Despite their scale and richness, we still lack effective methods to learn directly from user interactions. Unlike standard
datasets (Ouyang et al., [2022](#bib.bib51 "Training language models to follow instructions with human feedback"); Chung et al., [2024](#bib.bib54 "Scaling instruction-finetuned language models")), user interactions do not come with explicit labels, expert demonstrations, preference comparisons, or rewards. Instead, feedback is implicit and expressed through natural language responses whose meaning depends on the surrounding interaction context. As a result, it is unclear how to train directly on real-world conversations in a principled manner.

At the same time, we observe that language models already demonstrate the ability to leverage this interactive information in context, a capability known as in-context learning (Brown et al., [2020](#bib.bib52 "Language models are few-shot learners"); Wei et al., [2022](#bib.bib53 "Chain-of-thought prompting elicits reasoning in large language models")). In multi-turn conversations, models often revise their behavior effectively after observing a user’s follow-up.
When a user reports an error in generated code, the model can frequently infer which part of its previous response was incorrect and propose a fix. When a user points out that a required format was not followed, the model is able to correct the structure in a revised answer. When a user expresses dissatisfaction with tone or style, the model can adapt its response to better match the user’s preferences. In these cases, conditioning on the user’s follow-up message leads to responses that are more aligned with the task and the user’s intent.

These observations suggest a simple but powerful perspective: having seen the user’s follow-up message, the model’s behavior is often better aligned than before. The user interaction reveals information that the model can already interpret and act upon, but only after the fact. *In hindsight*.
Crucially, this improvement arises without additional supervision and reflects how the model’s behavior changes when it has access to the user’s follow-up.
This suggests that a model’s in-context learning ability can be used as a lever for learning directly from user interactions in a principled way.

Based on this idea, we introduce a simple and scalable method for learning directly from user interactions by comparing a model’s original behavior to what it would have done in hindsight. Concretely, after observing a user’s follow-up message, we reprompt the same model with this additional context and obtain a hindsight token distribution that reflects how the model would respond if it had access to the information revealed by the user. By comparing the original policy to this hindsight policy at every token of the original generation, we obtain a comparative learning signal that identifies how the model’s behavior should change. We then distill this signal back into the original policy using only the observed interaction. In other words, we distill the model into itself.

Building on recent work on self-distillation (Hübotter et al., [2026](#bib.bib29 "Reinforcement learning via self-distillation")),
we refer to this approach as Self-Distillation Policy Optimization (SDPO) from User Interactions.
We show that this approach (illustrated in [Figure˜1](#S1.F1 "In Introduction ‣ Aligning Language Models from User Interactions")) is simple and scalable, and enables language models to improve from raw, real-world user conversations without explicit supervision, reward models, or preference labels. Remarkably, when applied to real-world user interactions from WildChat (Zhao et al., [2024](#bib.bib15 "Wildchat: 1m chatgpt interaction logs in the wild")), SDPO from User Interactions improves alignment and instruction-following performance across standard benchmarks without degrading other capabilities. We also demonstrate that the same self-distillation mechanism naturally supports personalization and continual adaptation, allowing models to adapt to individual users purely through continued interaction.

!(/html/2603.12273/assets/figures/figure_1_2.png)

Figure 1: Direct Learning from User Interactions via Self-Distillation. From multi-turn user conversations, we obtain several interactions (x,y,o)(x,y,o) that consist of the conversation history xx, the model’s response yy, and the subsequent user message oo. By conditioning on the user’s follow-up, we form the *hindsight policy* and compare it to the original policy, producing token-level advantages that reinforce or penalize parts of the model’s original response. In this example, the user’s follow-up requests a more direct answer, leading to penalizing filler tokens and reinforcing the answer.

## Problem Formulation

The interaction between a language model and a user consists of a sequence of alternating assistant messages yty\_{t} and user messages oto\_{t}. At the tt-th turn, the language model observes the conversation history xt=(o0,y1,o1,…,ot−1)x\_{t}=(o\_{0},y\_{1},o\_{1},\dots,o\_{t-1}),
and generates a response yt∼πθ(⋅∣xt)y\_{t}\sim\pi\_{\theta}(\cdot\mid x\_{t}).111In practice, the assistant may condition on a representation of xtx\_{t}, such as a sliding context window, a learned embedding, or a summary. For simplicity, we treat xtx\_{t} as the full interaction history here. In turn, the user responds with oto\_{t}, assuming the conversation does not terminate.

In case of a fresh conversation initiated by the user, the first interaction reduces to the initial prompt o0o\_{0}, the assistant’s answer y1y\_{1}, and the user’s follow-up o1o\_{1}.
We define the triple (xt,yt,ot)(x\_{t},y\_{t},o\_{t}) as a single interaction.
Consequently, a user conversation (o0,y1,…,yt,ot)(o\_{0},y\_{1},\dots,y\_{t},o\_{t}) yields tt interactions (x1,y1,o1),…,(xt,yt,ot)(x\_{1},y\_{1},o\_{1}),\dots,(x\_{t},y\_{t},o\_{t}), which overlap in their histories, since xt+1x\_{t+1} contains all of xtx\_{t} plus the most recent interaction. For convenience, we use (x,y,o)(x,y,o) to denote a generic single user interaction.

Despite the ubiquity of conversational data of this form, it remains unclear how to leverage user interactions directly. One could attempt to introduce auxiliary mechanisms, such as semantic categorization of conversations (Shi et al., [2024](#bib.bib14 "Wildfeedback: aligning llms with in-situ user interactions and feedback"); Gunjal et al., [2025](#bib.bib76 "Rubrics as rewards: reinforcement learning beyond verifiable domains")), explicit preference annotation (Stephan et al., [2024](#bib.bib66 "Rlvf: learning from verbal feedback without overgeneralization"); Lee et al., [2024](#bib.bib67 "Reinforcement learning from reflective feedback (rlrf): aligning and improving llms via fine-grained self-reflection")), or other post-hoc extracted rewards (Wang et al., [2026](#bib.bib75 "Text2Grad: reinforcement learning from natural language feedback"); Urcelay et al., [2026](#bib.bib74 "From words to rewards: leveraging natural language for reinforcement learning")), but even then it is not obvious how to construct such signals reliably from raw interaction data alone. Any such approach would require additional modeling assumptions and intermediate objectives that are external to the interaction itself. As a result, they do not provide a simple or principled way of training models from user interactions as they naturally occur.

Accordingly, we define the problem of learning directly from multi-turn conversations without other externalities as Direct Learning from User Interactions.
This motivates the central question of this work:

> Can we train language models directly from multi-turn user interactions in a simple, principled, and scalable manner?

More specifically, can we both improve general alignment capabilities and enable continual personalization to individual users, relying only on user interactions without explicit supervision?

## Directly Learning from User Interactions via Self-Distillation

!(/html/2603.12273/assets/illustrations/water_logratios.png)

Figure 2: Example of the token-level advantages ([1](#S3.E1 "Equation 1 ‣ Policy Gradient. ‣ Directly Learning from User Interactions via Self-Distillation ‣ Aligning Language Models from User Interactions")) where the user complains with oo = “I said YES or NO only” after the assistant failed to follow the instruction.

Naturally occurring user interactions often contain implicit signals about the adequacy of an assistant’s response. Follow-up user messages may indicate that a response was incorrect, failed to follow an instruction, or did not align with the user’s preferences, even when no explicit feedback is provided. Such signals arise organically as part of the interaction and reflect how the assistant’s output was received or interpreted by the user.

##### Leveraging In-Context Capabilities.

Modern language models are often able to make effective use of such information in context: when conditioned on a follow-up message such as an error report, a clarification, or a requested revision, the model can frequently produce outputs that correct previous mistakes, better satisfy constraints, or more closely match the user’s preferences. In other words, *in hindsight*, the model’s distribution is better aligned to the task.

We leverage this capability by considering the *hindsight* distribution πθ(⋅∣x,o)\pi\_{\theta}(\cdot\mid x,o), which conditions not only on the interaction history xx but also on the observed user continuation oo through reprompting the original policy with xx and oo (cf. the prompting template in [Table˜1](#S3.T1 "In Policy Gradient. ‣ Directly Learning from User Interactions via Self-Distillation ‣ Aligning Language Models from User Interactions")). This distribution reflects how the model would respond if it were given access to the additional information revealed by the user’s message. Empirically and intuitively, πθ(⋅∣x,o)\pi\_{\theta}(\cdot\mid x,o) is often better aligned with the task at hand than the original policy πθ(⋅∣x)\pi\_{\theta}(\cdot\mid x).

This perspective admits a fine-grained, token-level interpretation. Letting yiy\_{i} be the ii-th token of the completion yy generated from πθ(⋅∣x)\pi\_{\theta}(\cdot\mid x), we can compare the token probabilities πθ​(yi∣x,y<i)\pi\_{\theta}(y\_{i}\mid x,y\_{<i}) and πθ​(yi∣x,o,y<i)\pi\_{\theta}(y\_{i}\mid x,o,y\_{<i}). When the hindsight model π(⋅∣x,o)\pi(\cdot\mid x,o) assigns lower probability to a particular token yiy\_{i}, this indicates that the user’s response provides evidence that this token (or the trajectory it induces) contributed to an undesirable outcome. Conversely, tokens whose likelihood increases under πθ(⋅∣x,o)\pi\_{\theta}(\cdot\mid x,o) are reinforced by the user’s response. The resulting log-ratio (i.e., log-difference) log⁡πθ​(yi∣x,o,y<i)−log⁡πθ​(yi∣x,y<i)\log\pi\_{\theta}(y\_{i}\mid x,o,y\_{<i})-\log\pi\_{\theta}(y\_{i}\mid x,y\_{<i})
can thus act as a comparative learning signal from user interactions, and will serve as the fundamental learning signal throughout the paper.
This now admits two equivalent views.

##### Policy Gradient.

One useful way to interpret the log-ratio is as a *token-level advantage*

|  |  |  |  |
| --- | --- | --- | --- |
|  | Ai​(x,y,o):=log⁡πθ​(yi∣x,o,y<i)πθ​(yi∣x,y<i),A\_{i}(x,y,o):=\log\frac{\pi\_{\theta}(y\_{i}\mid x,o,y\_{<i})}{\pi\_{\theta}(y\_{i}\mid x,y\_{<i})}, |  | (1) |

which measures how the likelihood of a token changes after conditioning on the user’s response.
Using this advantage in a standard policy gradient update reinforces tokens whose probability
increases under the hindsight distribution and penalizes those whose probability decreases.
We will refer to the advantage interchangeably as the token-level advantage or the SDPO advantage. [Figure˜2](#S3.F2 "In Directly Learning from User Interactions via Self-Distillation ‣ Aligning Language Models from User Interactions") provides an illustrative example.
In this view, learning corresponds to increasing the log-ratio in expectation, treating it as a fixed advantage per update step that is not differentiated w.r.t. θ\theta.

Algorithm 1  SDPO: Self-Distillation Policy Optimization from User Interactions

1: input: language model πθ\pi\_{\theta}

2: repeat

3:  observe context xtx\_{t} including the most recent user message ot−1o\_{t-1}

4:  sample answer yt∼πθ(⋅∣xt)y\_{t}\sim\pi\_{\theta}(\cdot\mid x\_{t}) with log-probabilities log⁡πθ​(yt,i∣xt,yt,<i)\log\pi\_{\theta}(y\_{t,i}\mid x\_{t},y\_{t,<i})

5:  observe user message oto\_{t} in response to yty\_{t} assuming the conversation does not terminate

6:  compute token log-probabilities of hindsight policy log⁡πθ​(yt,i∣xt,ot,yt,<i)\log\pi\_{\theta}(y\_{t,i}\mid x\_{t},o\_{t},y\_{t,<i})

7:  update current model πθ\pi\_{\theta} with gradient ∇θℒSDPO​(θ)\nabla\_{\!\theta}\,\mathcal{L}\_{\text{SDPO}}(\theta)

8: until converged

|  |  |
| --- | --- |
| User: | <conversation history including most recent user prompt> xx   <hindsight context> The following is a future user message.  Use this to guide your answer to the user prompt: oo |
| Assistant: | <assistant completion> yy |

Table 1: Chat template for the hindsight policy πθ​(y∣x,o)\pi\_{\theta}(y\mid x,o). We recover the usual template for the base policy πθ​(y∣x)\pi\_{\theta}(y\mid x) when removing “<hindsight context> […]” from the user prompt.

##### Self-Distillation.

Equivalently, and perhaps more conveniently from an optimization perspective, we can update πθ(⋅∣x)\pi\_{\theta}(\cdot\mid x) to more closely match the hindsight policy πθ(⋅∣x,o)\pi\_{\theta}(\cdot\mid x,o) by minimizing the reverse KL divergence. Here, the hindsight policy acts as a teacher and is treated as a fixed target during each update, for which we define the detached hindsight model π¯θ(⋅∣x,o):=stopgrad(πθ(⋅∣x,o))\overline{\pi}\_{\theta}(\cdot\mid x,o):=\textnormal{stopgrad}(\pi\_{\theta}(\cdot\mid x,o)). We first sample y∼πθ(⋅∣x)y\sim\pi\_{\theta}(\cdot\mid x) and then minimize a standard distillation loss,

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒSDPO(θ):=∑iKL(πθ(⋅∣x,y<i)||π¯θ(⋅∣x,o,y<i)),\mathcal{L}\_{\mathrm{SDPO}}(\theta):=\sum\_{i}\mathrm{KL}\big(\pi\_{\theta}(\cdot\mid x,y\_{<i})\,||\,\overline{\pi}\_{\theta}(\cdot\mid x,o,y\_{<i})\big), |  | (2) |

As shown in Hübotter et al. ([2026](#bib.bib29 "Reinforcement learning via self-distillation")), the gradient of ℒSDPO​(θ)\mathcal{L}\_{\mathrm{SDPO}}(\theta) is

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇θℒSDPO​(θ)=−𝔼y∼πθ(⋅∣x)​[∑i𝔼yi∼πθ(⋅∣x,y<i)​[∇θlog⁡πθ​(yi∣x,y<i)​Ai​(x,y,o)]].\nabla\_{\!\theta}\,\mathcal{L}\_{\mathrm{SDPO}}(\theta)=-\mathbb{E}\_{y\sim\pi\_{\theta}(\cdot\mid x)}\!\left[\sum\_{i}\mathbb{E}\_{{y}\_{i}\sim\pi\_{\theta}(\cdot\mid x,y\_{<i})}\Big[\nabla\_{\!\theta}\log\pi\_{\theta}(y\_{i}\mid x,y\_{<i})\,A\_{i}(x,y,o)\Big]\right]. |  | (3) |

Interestingly, [Appendix˜B](#A2 "Appendix B Gradient Derivation ‣ Aligning Language Models from User Interactions") in [Appendix˜B](#A2 "Appendix B Gradient Derivation ‣ Aligning Language Models from User Interactions") demonstrates that the policy gradient with token-level advantages is an unbiased one-sample approximation of the self-distillation gradient.
Hence, the policy gradient and self-distillation perspectives yield equivalent gradient updates in expectation, differing only in whether the log-ratio is interpreted as an advantage or as a distillation loss.
In our experiments, we adopt this policy gradient perspective for its simplicity.

Following recent work on self-distillation (Hübotter et al., [2026](#bib.bib29 "Reinforcement learning via self-distillation")), we refer to the corresponding algorithm as Self-Distillation Policy Optimization (SDPO) from User Interactions. [Algorithm˜1](#alg1 "In Policy Gradient. ‣ Directly Learning from User Interactions via Self-Distillation ‣ Aligning Language Models from User Interactions") outlines SDPO for learning from online user interactions, where an update is performed after observing the user’s next message.
In practice, interaction data is often available as logged conversations, possibly with completions generated from a different model. To this end, it is also natural to consider an offline and off-policy variant of SDPO, which we provide and discuss in [Section˜4.1](#S4.SS1 "General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions").

##### Self-Distillation as an Alignment Objective.

A common conceptual framing of alignment is that a language model should maximize a user’s latent reward function r​(x,y)r(x,y), which is unobserved and difficult to specify or estimate in practice. Since SDPO learns directly from naturally occurring user interactions, it is not immediately obvious how it relates to this traditional reward-maximization view of alignment.
Under a stylized model of user behavior and language model conditioning, we find that SDPO admits a simple and intuitive interpretation as implicitly optimizing the latent reward of the interacting user.

###### Proposition 3.1 (Informal, [Appendix˜A](#A1 "Appendix A A Latent Reward Perspective on SDPO ‣ Aligning Language Models from User Interactions")).

Under idealized assumptions on user responses and model conditioning, the sequence-level self-distillation advantage satisfies

|  |  |  |
| --- | --- | --- |
|  | log⁡πθ​(y∣x,o)πθ​(y∣x)=r​(x,y)−log⁡Z​(x,y),\log\frac{\pi\_{\theta}(y\mid x,o)}{\pi\_{\theta}(y\mid x)}=r(x,y)-\log Z(x,y), |  |

where Z​(x,y)Z(x,y) is a normalization term. In other words, under idealized assumptions, SDPO can be interpreted as implicitly maximizing the interacting user’s latent reward function.

!(/html/2603.12273/assets/images/qwen3_8b_all_benchmarks.png)

Figure 3: Training on real-world user conversations, SDPO improves general alignment and instruction-following performance across benchmarks, without degrading other capabilities. Results for Qwen3-8B before and after training on 14,000 real-world user conversations.

## Experimental Results

We evaluate SDPO with respect to two central questions:

1. 1.

   General Alignment: Can learning directly from raw, real-world user conversations improve the general alignment and instruction-following capabilities of language models?
2. 2.

   Personalization and Continual Adaptation: Can we continually align and personalize language models from online user interactions, without any explicit feedback or preference labels?

[Section˜4.1](#S4.SS1 "General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions") addresses the first question. We train SDPO on offline and off-policy user conversations from WildChat (Zhao et al., [2024](#bib.bib15 "Wildchat: 1m chatgpt interaction logs in the wild")) and WildFeedback (Shi et al., [2024](#bib.bib14 "Wildfeedback: aligning llms with in-situ user interactions and feedback")). These datasets consist of real-world user interactions and contain no explicit supervision signals. We evaluate the resulting models on standard alignment, instruction-following, math and coding, and knowledge tasks.
Remarkably, we show that training on real-world user conversations with SDPO improves alignment and instruction-following, without regressing other capabilities.

[Section˜4.2](#S4.SS2 "Continual Personalization and Adaptation from User Interactions ‣ Experimental Results ‣ Aligning Language Models from User Interactions") addresses the second question. We demonstrate that SDPO enables continual personalization through interaction by simulating users with distinct preferences and evaluating the model’s ability to adapt to these preferences over time from user interactions alone.

Finally, [Section˜4.3](#S4.SS3 "Interpretability and Robustness of SDPO Advantages ‣ Experimental Results ‣ Aligning Language Models from User Interactions") qualitatively analyzes and visualizes the self-distillation advantages from [Equation˜1](#S3.E1 "In Policy Gradient. ‣ Directly Learning from User Interactions via Self-Distillation ‣ Aligning Language Models from User Interactions") at illustrative user interactions. In particular, we show the extraordinary interpretability of the learning signal and its robustness to irrelevant next user messages.

### General Alignment from Real-World User Conversations

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | Alpaca Eval 2.0 (LC Winrate) | IFEval (Prompt-Level) | ArenaHard-v2 (Hard Prompt) | ArenaHard-v2 (Creative Writing) | MMLU-Pro (Chain-of-Thought) |
| Qwen3-4B | 37.9 | 81.9 | 9.0 | 8.0 | 58.1 |
| SDPO | ↑\boldsymbol{\uparrow} 46.1 | ↑\boldsymbol{\uparrow} 83.2 | ↓\boldsymbol{\downarrow} 7.8 | 7.9 | 58.0 |
| Qwen3-8B | 49.3 | 83.9 | 14.0 | 13.7 | 62.5 |
| SDPO | ↑\boldsymbol{\uparrow} 51.9 | ↑\boldsymbol{\uparrow} 85.0 | ↑\boldsymbol{\uparrow} 15.5 | ↑\boldsymbol{\uparrow} 16.2 | ↑\boldsymbol{\uparrow} 63.3 |
| Olmo3-7B-SFT | 34.3 | 80.2 | 2.4 | 1.4 | 23.7 |
| SDPO | ↑\boldsymbol{\uparrow} 35.2 | ↑\boldsymbol{\uparrow} 80.6 | 2.4 | 1.4 | ↑\boldsymbol{\uparrow} 24.0 |
| Olmo3-7B-DPO | 50.4 | 80.2 | 1.7 | 8.2 | 28.4 |
| SDPO | ↑\boldsymbol{\uparrow} 51.8 | ↑\boldsymbol{\uparrow} 80.4 | ↑\boldsymbol{\uparrow} 2.0 | ↑\boldsymbol{\uparrow} 10.0 | ↑\boldsymbol{\uparrow} 28.7 |

Table 2: Across model families and model sizes, SDPO improves alignment and instruction-following without degrading other capabilities. A mild exception is Qwen3-4B, where SDPO significantly increases performance on AlpacaEval 2.0 (+8.2%) and IFEval (+1.3%) but decreases performance on the math and coding tasks of ArenaHard-v2 (-1.2%). We only show arrows when performance changed by more than 0.1 percentage points.

We train SDPO on user conversations from WildChat (Zhao et al., [2024](#bib.bib15 "Wildchat: 1m chatgpt interaction logs in the wild")). In a first step, we consider WildFeedback (Shi et al., [2024](#bib.bib14 "Wildfeedback: aligning llms with in-situ user interactions and feedback")), a curated subset of WildChat containing approximately 20,000 individual conversations. Around 6,000 of these consist only of a single prompt-response pair and therefore do not contain a user follow-up.
We train on the remaining 14,000 conversations. As described in [Section˜2](#S2 "Problem Formulation ‣ Aligning Language Models from User Interactions"), for each user follow-up, we recover an interaction tuple (x,y,o)(x,y,o), where xx is the conversation history including the most recent prompt, yy the assistant response, and oo the next user message if it exists. We here truncate xx to the last 5 user or assistant messages when conversations include many turns. From the 14,000 conversations, we thereby obtain around 50,000 interaction tuples (x,y,o)(x,y,o), corresponding to an average of 4-5 user prompts per conversation.

We evaluate SDPO across two model families and four models overall. Specifically, we use Qwen3-4B and Qwen3-8B (Qwen Team, [2025](#bib.bib18 "Qwen3 technical report")), as well as Olmo3-7B-Instruct-SFT and Olmo3-7B-Instruct-DPO, which are the SFT and DPO checkpoints from the Olmo3 model family (Olmo et al., [2025](#bib.bib21 "Olmo 3")). All models are trained using the same 14,000 user interactions and evaluated using identical benchmark protocols.
We evaluate each model before and after SDPO training on AlpacaEval 2.0 (Dubois et al., [2024](#bib.bib22 "Length-controlled alpacaeval: a simple way to debias automatic evaluators")), IFEval (Zhou et al., [2023](#bib.bib25 "Instruction-following evaluation for large language models")), ArenaHard-v2 (Li et al., [2025](#bib.bib23 "From crowdsourced data to high-quality benchmarks: arena-hard and benchbuilder pipeline"); [2024](#bib.bib24 "From live data to high-quality benchmarks: the arena-hard pipeline")), and MMLU-Pro (Wang et al., [2024a](#bib.bib26 "Mmlu-pro: a more robust and challenging multi-task language understanding benchmark")) to cover alignment, instruction-following, math and coding, creative writing, and knowledge tasks.
We provide additional experimental details in [Appendix˜C](#A3 "Appendix C Experimental Details ‣ Aligning Language Models from User Interactions").

##### Off-Policy SDPO from Logged User Interactions.

As the assistant completions in WildChat were generated by external models (GPT-3.5 Turbo and GPT-4), the interactions are off-policy. In principle, unbiased off-policy policy gradient updates would require access to the behavioral policy or its token-level probabilities, which are not available for these datasets.222One could attempt to approximate the behavioral policy by supervised fine-tuning on the logged completions, but in preliminary experiments this did not lead to meaningful performance differences.
Instead, we optimize a surrogate SDPO objective defined directly over the logged interaction tuples (x,y,o)∼𝒟(x,y,o)\sim\mathcal{D}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ^SDPO(θ)=𝔼(x,y,o)∼𝒟[∑iKL(πθ(⋅∣x,y<i)||π¯θ(⋅∣x,o,y<i))].\displaystyle\widehat{\mathcal{L}}\_{\mathrm{SDPO}}(\theta)=\mathbb{E}\_{(x,y,o)\sim\mathcal{D}}\left[\sum\_{i}\mathrm{KL}\big(\pi\_{\theta}(\cdot\mid x,y\_{<i})\,||\,\overline{\pi}\_{\theta}(\cdot\mid x,o,y\_{<i})\big)\right]. |  | (4) |

While this objective is biased with respect to the on-policy SDPO loss, it can be interpreted as an off-policy approximation of the SDPO objective. In practice, we again use the one-sample approximation of its gradient, which is an unbiased estimator of [Equation˜4](#S4.E4 "In Off-Policy SDPO from Logged User Interactions. ‣ General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions"), analogously to [Appendix˜B](#A2 "Appendix B Gradient Derivation ‣ Aligning Language Models from User Interactions").

##### Main Results.

[Figure˜3](#S3.F3 "In Self-Distillation as an Alignment Objective. ‣ Directly Learning from User Interactions via Self-Distillation ‣ Aligning Language Models from User Interactions") reports the performance of Qwen3-8B before and after training with SDPO across all benchmarks. Training on raw, real-world user conversations consistently improves performance on all evaluated tasks, including AlpacaEval 2.0, IFEval, ArenaHard-v2, and MMLU-Pro. Importantly, we observe no degradation on any benchmark, despite the fact that the training data consists of noisy user interactions without explicit feedback or labels.333For a first-hand impression of the diversity and sometimes chaotic nature of real-world user conversations, we refer the interested reader to the WildChat and WildFeedback datasets on HuggingFace.
Notably, these improvements extend beyond alignment and instruction-following benchmarks. SDPO also improves performance on math and coding tasks in ArenaHard-v2 (Hard Prompt), creative writing queries in ArenaHard-v2 (Creative Writing), and knowledge tasks in MMLU-Pro (Chain-of-Thought), indicating that learning from user interactions does not come at the expense of other capabilities and can, when interactions contain informative corrections or refinements, even strengthen them. Evaluations on additional pre-training benchmarks, provided in [Appendix˜D](#A4 "Appendix D Additional Experimental Results ‣ Aligning Language Models from User Interactions"), demonstrate that SDPO also maintains consistent performance across these tasks

[Table˜2](#S4.T2 "In General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions") summarizes results across all models and benchmarks. For the Olmo3-7B models, including both the SFT and DPO checkpoints, SDPO yields consistent but often modest improvements. In contrast, Qwen3-4B exhibits a clear trade-off: while SDPO substantially improves performance on AlpacaEval 2.0 (+8.2%) and IFEval (+1.3%), it also leads to a mild decrease (-1.2%) on the math and coding tasks in ArenaHard-v2 (Hard Prompt).
Overall, these results suggest that SDPO is most effective when the base model can reliably interpret and exploit the hindsight signal provided by user follow-ups. For smaller or less instruction-tuned models, this signal appears weaker or less stable, leading to smaller gains and, in some cases, task-specific trade-offs.

##### How important is the quality of user conversations?

WildFeedback is a curated subset of WildChat that retains roughly 3% of the original conversations by filtering for interactions that contain implicit feedback signals, such as expressions of dissatisfaction, requests for correction, or revision prompts (Shi et al., [2024](#bib.bib14 "Wildfeedback: aligning llms with in-situ user interactions and feedback")).
In practice, due to the abundant nature of user conversations filtering down to a smaller subset is not a concern. Nevertheless, as a secondary robustness check, we evaluate whether SDPO continues to behave sensibly when trained on fully uncurated user interactions. Concretely, we train SDPO on a randomly sampled subset of WildChat that matches WildFeedback in scale, consisting again of approximately 14,000 conversations and 50,000 interaction tuples (x,y,o)(x,y,o). All other training and evaluation settings are kept identical.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | AlpacaEval 2.0 (LC Winrate) | IFEval (Prompt-Level) | ArenaHard-v2 (Hard Prompt) | ArenaHard-v2 (Creative Writing) | MMLU-Pro (Chain-of-Thought) |
| Qwen3-8B | 49.3 | 83.9 | 14.0 | 13.7 | 62.5 |
| SDPO (WildFeedback) | 51.9 | 85.0 | 15.5 | 16.2 | 63.3 |
| SDPO (WildChat) | ↑\boldsymbol{\uparrow} 50.7 | ↑\boldsymbol{\uparrow} 84.5 | ↓\boldsymbol{\downarrow} 13.4 | ↑\boldsymbol{\uparrow} 14.0 | 62.4 |

Table 3: Even on fully uncurated user interactions, SDPO still yields improvements in alignment and instruction-following and only mild degradation in math and coding. Results for training on a randomly sampled subset of 14,000 conversations (about 50,000 interactions (x,y,o)(x,y,o)) from WildChat. For comparison, we here include the results for training on WildFeedback from [Table˜2](#S4.T2 "In General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions").

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | AlpacaEval 2.0 (LC Winrate) | IFEval (Prompt-Level) | ArenaHard-v2 (Hard Prompt) | ArenaHard-v2 (Creative Writing) | MMLU-Pro (Chain-of-Thought) |
| Qwen3-4B | 37.9 | 81.9 | 9.0 | 8.0 | 58.1 |
| SFT on Dataset | ↓\boldsymbol{\downarrow} 18.9 | ↓\boldsymbol{\downarrow} 73.2 | ↓\boldsymbol{\downarrow} 3.1 | ↓\boldsymbol{\downarrow} 2.6 | ↓\boldsymbol{\downarrow} 51.2 |

Table 4: SDPO is fundamentally different from SFT. As a sanity check, we fine-tune Qwen3-4B on the assistant completions in WildFeedback using standard supervised fine-tuning.

[Table˜3](#S4.T3 "In How important is the quality of user conversations? ‣ General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions") reports the resulting performance for Qwen3-8B. Despite training on fully unfiltered user interactions, SDPO does not exhibit widespread performance degradation. On the contrary, we still observe improvements on AlpacaEval 2.0 and IFEval. Performance on math and coding tasks in ArenaHard-v2 is modestly reduced relative to the base model (-0.6%).
Overall, we find that while filtering for better and feedback-rich conversations strengthens the learning signal, SDPO is surprisingly robust with respect to data quality. Even when trained on fully uncurated conversations, SDPO can extract useful alignment signals from user interactions without collapsing performance.

##### SDPO vs. SFT.

Conceptually, SDPO is fundamentally different from supervised fine-tuning (SFT). While SFT uniformly increases the likelihood of tokens in the training completions, SDPO can explicitly decrease token probabilities whenever the log-ratio ([1](#S3.E1 "Equation 1 ‣ Policy Gradient. ‣ Directly Learning from User Interactions via Self-Distillation ‣ Aligning Language Models from User Interactions")) is negative, for example, when the user follow-up provides evidence of an error or failure to follow instructions. Still, we include a sanity check to confirm that the gains observed with SDPO are not the result of implicitly supervised fine-tuning on the assistant completions in the dataset.
To this end, we fine-tune Qwen3-4B using standard SFT on the context-completion pairs (x,y)(x,y) from WildFeedback, where xx contains previous user-assistant turns to ensure that later prompts remain well contextualized.

As shown in [Table˜4](#S4.T4 "In How important is the quality of user conversations? ‣ General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions"), supervised fine-tuning on the assistant completions leads to a substantial degradation across all benchmarks. This is perhaps unsurprising as Qwen3-4B is already a strongly instruction-tuned model, while the completions in WildFeedback sometimes originate from older models such as GPT-3.5 Turbo, which perform worse on many of the evaluated benchmarks. Moreover, prior analysis of conversations in WildFeedback shows that users express some form of dissatisfaction with the model’s responses in more than half of the conversations (Shi et al., [2024](#bib.bib14 "Wildfeedback: aligning llms with in-situ user interactions and feedback")). Consequently, fine-tuning on these completions can be detrimental.

### Continual Personalization and Adaptation from User Interactions

!(/html/2603.12273/assets/images/personalization/pure_winrate.png)

Figure 4: SDPO adapts online to changing user preferences. The user’s preference about how the model should respond is flipped to its opposite after the first 250 interactions. SDPO with Qwen3-4B is able to quickly reverse the learned behavior.

!(/html/2603.12273/assets/images/personalization/pure_winrate/concise_casual_beginner.png)

Figure 5: SDPO rapidly personalizes to individual users from interaction alone. Win rate of SDPO against its base model (Qwen3-4B) for a user that prefers concise, casual, and beginner-friendly model responses.

Because SDPO learns directly from user interactions, it naturally enables direct personalization from those conversations. Rather than relying on explicit preference labels, rewards, or user profiles, the model can adapt its behavior based solely on how a user responds to previous outputs. In this section, we study whether such interaction-driven updates allow language models to continually personalize to individual users and adapt to changing or evolving preferences.

We evaluate this capability in two complementary experimental settings.
In the first, we study stylistic personalization in a controlled summarization task using prompts from the TL;DR dataset (Stiennon et al., [2020](#bib.bib60 "Learning to summarize from human feedback")). We define user-specific writing-style preferences, such as favoring concise, casual, and beginner-friendly responses, and train Qwen3-4B with SDPO. We use Qwen3-8B to generate simulated user responses as well as the preference-based evaluations. We provide additional experimental details in [Appendix˜C](#A3 "Appendix C Experimental Details ‣ Aligning Language Models from User Interactions").

In the second setting, we consider more complex and complementary user preferences on a broad set of real-world prompts from HelpSteer2 (Wang et al., [2024b](#bib.bib59 "Helpsteer 2: open-source dataset for training top-performing reward models")). Here, preferences emphasize different aspects of responses that are not mutually exclusive. We train Qwen3-8B with SDPO, using Claude Haiku 4.5 to simulate user follow-ups and act as a judge.

!(/html/2603.12273/assets/images/personalization/winrate_with_sd.png)

Figure 6: SDPO enables continual personalization without catastrophic forgetting.
We train a single Qwen3-8B model online with SDPO for 1500 user interactions, during which three complementary user preferences are introduced sequentially (500 interactions each). Each curve reports the win rate of the current model with respect to the model checkpoint at the time the corresponding preference was introduced, thereby isolating the relative improvement along that specific preference dimension. Earlier preferences remain strong as new ones are learned, indicating that SDPO can accumulate complementary preferences over time without forgetting previously learned behavior. Shaded regions indicate standard error over 256 evaluation prompts.

##### Main Results.

[Figure˜5](#S4.F5 "In Continual Personalization and Adaptation from User Interactions ‣ Experimental Results ‣ Aligning Language Models from User Interactions") shows the win rate of SDPO against its base model, Qwen3-4B, as a function of the number of user interactions (x,y,o)(x,y,o). Starting from parity, SDPO rapidly adapts to the user’s preferences within a small number of interactions, achieving over 85% win rate after only 50 interactions and exceeding 95% after 200 interactions. Notably, this adaptation is driven by a very limited amount of interaction data and a correspondingly small number of policy updates.

For reference, we also report the performance of an in-context oracle that is explicitly provided with the full user profile description in its prompt. Continual online adaptation with SDPO matches and can even exceed the performance of this oracle, suggesting that interaction-based learning can extract preference signals that are difficult to encode purely through prompting. Additional results for a range of other user profiles are provided in [Appendix˜D](#A4 "Appendix D Additional Experimental Results ‣ Aligning Language Models from User Interactions").

[Figure˜4](#S4.F4 "In Continual Personalization and Adaptation from User Interactions ‣ Experimental Results ‣ Aligning Language Models from User Interactions") evaluates SDPO under changing user preferences. After an initial phase of 250 user interactions, the user’s preference is abruptly flipped to its opposite (e.g., from concise and casual to detailed and professional). SDPO quickly adjusts the policy to this change, reversing the previously learned behavior and converging to the new preference, demonstrating that outdated preferences can be unlearned when they no longer align with user interactions.

Finally, [Figure˜6](#S4.F6 "In Continual Personalization and Adaptation from User Interactions ‣ Experimental Results ‣ Aligning Language Models from User Interactions") considers continual personalization with multiple, complementary user preferences. We observe that SDPO is able to incorporate new preferences while retaining previously inferred ones, illustrating that continual personalization through SDPO does not require forgetting earlier behavior when preferences are compatible.

### Interpretability and Robustness of SDPO Advantages

While [Section˜4.1](#S4.SS1 "General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions") already demonstrated robustness to noisy and uncurated user interactions at scale, we complement these quantitative results with a qualitative analysis of the learning signal. Specifically, we visualize the SDPO advantages Ai​(x,y,o)A\_{i}(x,y,o) using heatmaps for illustrative user interactions. [Figure˜7](#S4.F7 "In Interpretability and Robustness of SDPO Advantages ‣ Experimental Results ‣ Aligning Language Models from User Interactions") and [Figure˜8](#S4.F8 "In Interpretability and Robustness of SDPO Advantages ‣ Experimental Results ‣ Aligning Language Models from User Interactions") show advantages computed with Qwen3-8B for 24 interactions where the next user message is relevant to the model’s previous completion and where it is unrelated, respectively. Positive advantages (shown in blue) correspond to tokens reinforced by SDPO, while negative advantages (shown in red) correspond to tokens that are penalized.

When user follow-ups provide relevant feedback, such as requests for revision, corrections, or explicit preference statements, we observe strong positive and negative advantages ([Figure˜7](#S4.F7 "In Interpretability and Robustness of SDPO Advantages ‣ Experimental Results ‣ Aligning Language Models from User Interactions")). For example, a follow-up request to rewrite an email in a more formal tone results in large negative advantages on informal tokens, such as *Quick*, *Hey*, *Just*, indicating that these tokens have lower probability under the hindsight policy.

In contrast, when user follow-ups are unrelated to the model’s previous output, the resulting SDPO advantages are close to zero ([Figure˜8](#S4.F8 "In Interpretability and Robustness of SDPO Advantages ‣ Experimental Results ‣ Aligning Language Models from User Interactions")). In these cases, the hindsight policy assigns probabilities similar to those of the original policy, leading to little or no learning signal. We also occasionally observe weakly positive advantages, particularly on tokens for which the model was previously uncertain, which suggests that the hindsight policy frequently treats topic shifts as neutral or mildly positive evidence about the preceding response.

Overall, these visualizations highlight two key properties of our self-distillation approach. First, the token-level advantages are highly interpretable and align with intuitive notions of user feedback when such feedback is present. Second, SDPO is robust to irrelevant or uninformative user follow-ups, naturally suppressing learning updates when the interaction does not convey actionable information about the preceding model output.

!(/html/2603.12273/assets/images/log_ratio_visuals/heatmap_relevant.png)

!(/html/2603.12273/assets/images/log_ratio_visuals/case1_conversation_followup.png)

Figure 7: When user follow-ups are relevant to the model’s completion, we observe strong positive and negative SDPO advantages. We visualize the advantages with Qwen3-8B for user follow-ups that carry relevant information about the model’s answer, such as requests for revisions, positive reactions, or other relevant feedback.
Below: Example (second line in the heatmap), where the user requests a more formal rewrite of the assistant’s draft (*“Rewrite in a formal, professional tone”*). Informal expressions have large negative advantages. Accordingly, SDPO adapts the policy to respond more formally when the user needs help with work emails in the future.

!(/html/2603.12273/assets/images/log_ratio_visuals/heatmap_irrelevant.png)

!(/html/2603.12273/assets/images/log_ratio_visuals/case1_conversation_unrelated.png)

Figure 8: When user follow-ups are unrelated to the model’s response, SDPO advantages are close to zero. We visualize the advantages with Qwen3-8B for user follow-ups that are unrelated to the model’s generation.
Below: Following the request to write an email, the user responds with *“What is 27×\times4?”*, which is unrelated to the original request. The advantages are close to zero everywhere, which means that SDPO does not meaningfully update the policy from these interactions.

## Related Work

##### Preference-Based Alignment.

Much of recent progress in aligning language models comes from supervised instruction tuning and
preference-based post-training, where explicit human or AI feedback is collected as rankings or rewards and optimized via RLHF or direct preference optimization (Ouyang et al., [2022](#bib.bib51 "Training language models to follow instructions with human feedback"); Bai et al., [2022](#bib.bib38 "Constitutional ai: harmlessness from ai feedback"); Rafailov et al., [2023](#bib.bib79 "Direct preference optimization: your language model is secretly a reward model")).
These approaches are effective, but they rely on curated datasets that provide explicit feedback for each generation.
In contrast, we leverage implicit feedback within real-world user conversations.
Though such conversations are abundant, few open datasets of such user conversations exist (Don-Yehiya et al., [2025](#bib.bib62 "The future of open human feedback")), since the community has lacked an effective method for learning from them.

##### Learning from Natural Language Feedback and through Retrospection.

Substantial research has focused on translating verbal feedback into reward functions for RL, for example, by mapping feedback to discrete token-level rewards using an external frozen model (Wang et al., [2026](#bib.bib75 "Text2Grad: reinforcement learning from natural language feedback")) or by using strong external LLMs to explicitly construct state-wise reward functions (Goyal et al., [2019](#bib.bib72 "Using natural language for reward shaping in reinforcement learning"); Xie et al., [2024](#bib.bib73 "Text2reward: reward shaping with language models for reinforcement learning"); Urcelay et al., [2026](#bib.bib74 "From words to rewards: leveraging natural language for reinforcement learning")).
A recent simplified instantiation of this approach has been to manually design so-called rubrics according to which an LLM judge scores generations (Gunjal et al., [2025](#bib.bib76 "Rubrics as rewards: reinforcement learning beyond verifiable domains"); Shao et al., [2025](#bib.bib82 "Dr tulu: reinforcement learning with evolving rubrics for deep research"); Kimi Team et al., [2025](#bib.bib77 "Kimi k2: open agentic intelligence")).

Alternatively, feedback can be utilized without explicit reward modeling.
Recent research explored in-context improvement without updating model weights (Madaan et al., [2023](#bib.bib68 "Self-refine: iterative refinement with self-feedback"); Shinn et al., [2023](#bib.bib69 "Reflexion: language agents with verbal reinforcement learning"); Yao et al., [2024](#bib.bib70 "Retroformer: retrospective large language agents with policy gradient optimization"); Yuksekgonul et al., [2025](#bib.bib71 "Optimizing generative ai by backpropagating language model feedback")).
Other works manually curate preference datasets by pairing responses before and after feedback to train with direct preference optimization (Stephan et al., [2024](#bib.bib66 "Rlvf: learning from verbal feedback without overgeneralization"); Lee et al., [2024](#bib.bib67 "Reinforcement learning from reflective feedback (rlrf): aligning and improving llms via fine-grained self-reflection")).
Chen et al. ([2024](#bib.bib17 "Learning from natural language feedback")) perform SFT on refined generations that incorporate feedback.
Our approach differs from these works in performing direct credit assignment over the initial model’s rollouts without additional generation.
In concurrent work, Auzina et al. ([2026](#bib.bib61 "Intrinsic credit assignment for long horizon interaction")) use a related idea to self-distillation for learning how to elicit information from multi-turn conversations by assigning turn-level implicit rewards.

##### Self-Distillation.

Distillation is a general technique for transferring knowledge from a strong teacher model to a student model by mimicking the teacher’s output distribution or intermediate representations (Hinton et al., [2015](#bib.bib31 "Distilling the knowledge in a neural network"); Agarwal et al., [2024](#bib.bib32 "On-policy distillation of language models: learning from self-generated mistakes"); Lu and Thinking Machines Lab, [2025](#bib.bib33 "On-policy distillation")).
Based on this idea, Snell et al. ([2022](#bib.bib40 "Learning by distilling context")) proposed context distillation which distills the model’s behavior given a fixed context into the model’s weights.
This context distillation has been effective at compressing behavior (Bai et al., [2022](#bib.bib38 "Constitutional ai: harmlessness from ai feedback"); Choi et al., [2022](#bib.bib39 "Prompt injection: parameterization of fixed inputs"); Yang et al., [2024](#bib.bib36 "Self-distillation bridges distribution gap in language model fine-tuning"); [2025](#bib.bib37 "Distilling rule-based knowledge into large language models")) and factual information (Eyuboglu et al., [2026](#bib.bib41 "Cartridges: lightweight and general-purpose long context representations via self-study"); Kujanpää et al., [2025](#bib.bib42 "Efficient knowledge injection in LLMs via self-distillation"); Cao et al., [2025](#bib.bib43 "InfiniteICL: breaking the limit of context window size via long short-term memory transformation")) into model weights.
Beyond compressing a fixed context into model weights, several recent works generate from the self-teacher conditioned on extra context (e.g., “hints”) and train on them with SFT, DPO, or GRPO objectives (Scheurer et al., [2023](#bib.bib44 "Training language models with language feedback at scale"); Dou et al., [2024](#bib.bib45 "Re-rest: reflection-reinforced self-training for language agents"); Zhou et al., [2025](#bib.bib46 "ExPO: unlocking hard reasoning with self-explanation-guided reinforcement learning"); Mitra and Ulukus, [2025](#bib.bib47 "Semantic soft bootstrapping: long context reasoning in llms without reinforcement learning"); Qu et al., [2026](#bib.bib48 "POPE: learning to reason on hard problems via privileged on-policy exploration"); Song et al., [2026](#bib.bib30 "Expanding the capabilities of reinforcement learning via text feedback"); Shi et al., [2026](#bib.bib81 "Experiential reinforcement learning")).
These approaches perform *off-policy* self-distillation where the student is trained on generations from the teacher, whereas SDPO performs *on-policy* self-distillation (Hübotter et al., [2026](#bib.bib29 "Reinforcement learning via self-distillation"); Shenfeld et al., [2026](#bib.bib28 "Self-distillation enables continual learning"); Zhao et al., [2026](#bib.bib27 "Self-distilled reasoner: on-policy self-distillation for large language models"); Penaloza et al., [2026](#bib.bib50 "Privileged information distillation for language models"); Chen et al., [2025](#bib.bib49 "Retrospective in-context learning for temporal credit assignment with large language models")), where the student is trained to avoid mistakes in its own generations.

## Discussion

We introduced a simple and scalable self-distillation approach for learning directly from naturally occurring user interactions. We leverage the language model’s in-context learning capabilities by treating the user’s next message as hindsight information, yielding an interpretable token-level learning signal without requiring other auxiliary mechanisms. Empirically, we showed that SDPO improves general alignment and instruction-following performance when trained on raw, real-world user conversations, supports continual personalization from interaction alone, and remains robust to noisy, uncurated, or irrelevant user follow-ups.

More broadly, our results highlight user interactions as a distinct and underutilized data modality for improving deployed language models. Unlike traditional training data, user interactions arise naturally during deployment and reflect how model outputs are actually used, evaluated, and acted upon in real-world settings. The scale and diversity of such data far exceed that of manually curated datasets, suggesting substantial potential for learning systems that close the loop between deployment and training. Our findings indicate that even simple, local learning signals extracted from user follow-ups can be sufficient to drive meaningful adaptation.

##### Safety and Ethical Considerations.

Learning directly from user interactions introduces important safety and ethical considerations. User follow-ups may implicitly encourage behaviors that conflict with existing safety or alignment constraints, for example by rewarding evasive, misleading, or policy-violating responses through repeated interaction. In particular, continual personalization without additional guardrails raises risks that adaptive updates could be exploited by users attempting to steer the model toward unsafe or manipulative behavior over time. While SDPO derives a local, token-level learning signal and naturally suppresses updates from irrelevant interactions, it does not by itself distinguish between benign and adversarial learning signals. Nevertheless, the hindsight prompt may offer the ability to endow the model with principles based on which to act and interpret user feedback. More broadly, the collection and use of user interaction data for learning must be accompanied by appropriate transparency, consent, and governance mechanisms.

## Acknowledgements

We thank Frederike Lübeck, Alexander Hoyle, and Manish Prajapat for many helpful discussions.

This project was primarily supported by the ETH AI Center through an ETH AI Center Postdoctoral Fellowship to TKB and an ETH AI Center Doctoral Fellowship to BP. JH was supported by the Swiss National Science Foundation under NCCR Automation, grant agreement 51NF40 180545. This project also received support through the Swiss AI compute grant a166.

## References

* R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. R. Garea, M. Geist, and O. Bachem (2024)
  On-policy distillation of language models: learning from self-generated mistakes.
  In ICLR,
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* I. A. Auzina, J. Strüber, S. Hernández-Gutiérrez, S. Goel, A. Prabhu, and M. Bethge (2026)
  Intrinsic credit assignment for long horizon interaction.
  Cited by: [§5](#S5.SS0.SSS0.Px2.p2.1 "Learning from Natural Language Feedback and through Retrospection. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* Y. Bai, S. Kadavath, S. Kundu, A. Askell, J. Kernion, A. Jones, A. Chen, A. Goldie, A. Mirhoseini, C. McKinnon, et al. (2022)
  Constitutional ai: harmlessness from ai feedback.
  arXiv preprint arXiv:2212.08073.
  Cited by: [§5](#S5.SS0.SSS0.Px1.p1.1 "Preference-Based Alignment. ‣ Related Work ‣ Aligning Language Models from User Interactions"),
  [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. (2020)
  Language models are few-shot learners.
  In NeurIPS,
  Cited by: [§1](#S1.p3.1 "Introduction ‣ Aligning Language Models from User Interactions").
* B. Cao, D. Cai, and W. Lam (2025)
  InfiniteICL: breaking the limit of context window size via long short-term memory transformation.
  In ACL,
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* A. Chen, J. Scheurer, J. A. Campos, T. Korbak, J. S. Chan, S. R. Bowman, K. Cho, and E. Perez (2024)
  Learning from natural language feedback.
  TMLR.
  Cited by: [§5](#S5.SS0.SSS0.Px2.p2.1 "Learning from Natural Language Feedback and through Retrospection. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* W. Chen, J. Chen, F. Tajwar, H. Zhu, X. Duan, R. Salakhutdinov, and J. Schneider (2025)
  Retrospective in-context learning for temporal credit assignment with large language models.
  In NeurIPS,
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* E. Choi, Y. Jo, J. Jang, and M. Seo (2022)
  Prompt injection: parameterization of fixed inputs.
  arXiv preprint arXiv:2206.11349.
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y. Tay, W. Fedus, Y. Li, X. Wang, M. Dehghani, S. Brahma, et al. (2024)
  Scaling instruction-finetuned language models.
  JMLR 25 (70),  pp. 1–53.
  Cited by: [§1](#S1.p2.1 "Introduction ‣ Aligning Language Models from User Interactions").
* S. Don-Yehiya, B. Burtenshaw, R. Fernandez Astudillo, C. Osborne, M. Jaiswal, T. Kuo, W. Zhao, I. Shenfeld, A. Peng, M. Yurochkin, et al. (2025)
  The future of open human feedback.
  Nature Machine Intelligence 7 (6),  pp. 825–835.
  Cited by: [§5](#S5.SS0.SSS0.Px1.p1.1 "Preference-Based Alignment. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* S. Don-Yehiya, L. Choshen, and O. Abend (2024)
  Naturally occurring feedback is common, extractable and useful.
  arXiv preprint arXiv:2407.10944.
  Cited by: [§1](#S1.p1.1 "Introduction ‣ Aligning Language Models from User Interactions").
* Z. Dou, C. Yang, X. Wu, K. Chang, and N. Peng (2024)
  Re-rest: reflection-reinforced self-training for language agents.
  In EMNLP,
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* Y. Dubois, B. Galambosi, P. Liang, and T. B. Hashimoto (2024)
  Length-controlled alpacaeval: a simple way to debias automatic evaluators.
  In COLM,
  Cited by: [§4.1](#S4.SS1.p2.1 "General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions").
* S. Eyuboglu, R. Ehrlich, S. Arora, N. Guha, D. Zinsley, E. Liu, W. Tennien, A. Rudra, J. Zou, A. Mirhoseini, et al. (2026)
  Cartridges: lightweight and general-purpose long context representations via self-study.
  In ICLR,
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* P. Goyal, S. Niekum, and R. J. Mooney (2019)
  Using natural language for reward shaping in reinforcement learning.
  In IJCAI,
  Cited by: [§5](#S5.SS0.SSS0.Px2.p1.1 "Learning from Natural Language Feedback and through Retrospection. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* A. Gunjal, A. Wang, E. Lau, V. Nath, Y. He, B. Liu, and S. Hendryx (2025)
  Rubrics as rewards: reinforcement learning beyond verifiable domains.
  arXiv preprint arXiv:2507.17746.
  Cited by: [§2](#S2.p3.1 "Problem Formulation ‣ Aligning Language Models from User Interactions"),
  [§5](#S5.SS0.SSS0.Px2.p1.1 "Learning from Natural Language Feedback and through Retrospection. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* G. Hinton, O. Vinyals, and J. Dean (2015)
  Distilling the knowledge in a neural network.
  arXiv preprint arXiv:1503.02531.
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* J. Hübotter, F. Lübeck, L. Behric, A. Baumann, M. Bagatella, D. Marta, I. Hakimi, I. Shenfeld, T. K. Buening, C. Guestrin, et al. (2026)
  Reinforcement learning via self-distillation.
  arXiv preprint arXiv:2601.20802.
  Cited by: [§1](#S1.p6.1 "Introduction ‣ Aligning Language Models from User Interactions"),
  [§3](#S3.SS0.SSS0.Px3.p2.1 "Self-Distillation. ‣ Directly Learning from User Interactions via Self-Distillation ‣ Aligning Language Models from User Interactions"),
  [§3](#S3.SS0.SSS0.Px3.p3.1 "Self-Distillation. ‣ Directly Learning from User Interactions via Self-Distillation ‣ Aligning Language Models from User Interactions"),
  [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* Kimi Team, Y. Bai, Y. Bao, G. Chen, J. Chen, N. Chen, R. Chen, Y. Chen, Y. Chen, Y. Chen, et al. (2025)
  Kimi k2: open agentic intelligence.
  arXiv preprint arXiv:2507.20534.
  Cited by: [§5](#S5.SS0.SSS0.Px2.p1.1 "Learning from Natural Language Feedback and through Retrospection. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* K. Kujanpää, P. Marttinen, H. Valpola, and A. Ilin (2025)
  Efficient knowledge injection in LLMs via self-distillation.
  TMLR.
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* K. Lee, D. Hwang, S. Park, Y. Jang, and M. Lee (2024)
  Reinforcement learning from reflective feedback (rlrf): aligning and improving llms via fine-grained self-reflection.
  arXiv preprint arXiv:2403.14238.
  Cited by: [§2](#S2.p3.1 "Problem Formulation ‣ Aligning Language Models from User Interactions"),
  [§5](#S5.SS0.SSS0.Px2.p2.1 "Learning from Natural Language Feedback and through Retrospection. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* T. Li, W. Chiang, E. Frick, L. Dunlap, T. Wu, B. Zhu, J. E. Gonzalez, and I. Stoica (2025)
  From crowdsourced data to high-quality benchmarks: arena-hard and benchbuilder pipeline.
  In ICML,
  Cited by: [§4.1](#S4.SS1.p2.1 "General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions").
* T. Li, W. Chiang, E. Frick, L. Dunlap, B. Zhu, J. E. Gonzalez, and I. Stoica (2024)
  From live data to high-quality benchmarks: the arena-hard pipeline.
  External Links: [Link](https://lmsys.org/blog/2024-04-19-arena-hard)
  Cited by: [§4.1](#S4.SS1.p2.1 "General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions").
* S. Lin, J. Hilton, and O. Evans (2022)
  Truthfulqa: measuring how models mimic human falsehoods.
  In ACL,
  Cited by: [Table 6](#A4.T6 "In Additional Results from Section˜4.1 ‣ Appendix D Additional Experimental Results ‣ Aligning Language Models from User Interactions").
* K. Lu and Thinking Machines Lab (2025)
  On-policy distillation.
  Thinking Machines Lab: Connectionism.
  External Links: [Link](https://thinkingmachines.ai/blog/on-policy-distillation)
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* R. Luo, Z. Liu, X. Liu, C. Du, M. Lin, W. Chen, W. Lu, and T. Pang (2025)
  Language models can learn from verbal feedback without scalar rewards.
  arXiv preprint arXiv:2509.22638.
  Cited by: [Appendix A](#A1.p3.3 "Appendix A A Latent Reward Perspective on SDPO ‣ Aligning Language Models from User Interactions").
* A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe, U. Alon, N. Dziri, S. Prabhumoye, Y. Yang, et al. (2023)
  Self-refine: iterative refinement with self-feedback.
  In NeurIPS,
  Cited by: [§5](#S5.SS0.SSS0.Px2.p2.1 "Learning from Natural Language Feedback and through Retrospection. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* P. Mitra and S. Ulukus (2025)
  Semantic soft bootstrapping: long context reasoning in llms without reinforcement learning.
  arXiv preprint arXiv:2512.05105.
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* T. Olmo, A. Ettinger, A. Bertsch, B. Kuehl, D. Graham, D. Heineman, D. Groeneveld, F. Brahman, F. Timbers, H. Ivison, et al. (2025)
  Olmo 3.
  arXiv preprint arXiv:2512.13961.
  Cited by: [§4.1](#S4.SS1.p2.1 "General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions").
* L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. (2022)
  Training language models to follow instructions with human feedback.
  In NeurIPS,
  Cited by: [§1](#S1.p2.1 "Introduction ‣ Aligning Language Models from User Interactions"),
  [§5](#S5.SS0.SSS0.Px1.p1.1 "Preference-Based Alignment. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* E. Penaloza, D. Vattikonda, N. Gontier, A. Lacoste, L. Charlin, and M. Caccia (2026)
  Privileged information distillation for language models.
  arXiv preprint arXiv:2602.04942.
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* Y. Qu, A. Setlur, V. Smith, R. Salakhutdinov, and A. Kumar (2026)
  POPE: learning to reason on hard problems via privileged on-policy exploration.
  arXiv preprint arXiv:2601.18779.
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* Qwen Team (2025)
  Qwen3 technical report.
  arXiv preprint arXiv:2505.09388.
  Cited by: [§4.1](#S4.SS1.p2.1 "General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions").
* R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn (2023)
  Direct preference optimization: your language model is secretly a reward model.
  In NeurIPS,
  Cited by: [§5](#S5.SS0.SSS0.Px1.p1.1 "Preference-Based Alignment. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* J. Scheurer, J. A. Campos, T. Korbak, J. S. Chan, A. Chen, K. Cho, and E. Perez (2023)
  Training language models with language feedback at scale.
  arXiv preprint arXiv:2303.16755.
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* R. Shao, A. Asai, S. Z. Shen, H. Ivison, V. Kishore, J. Zhuo, X. Zhao, M. Park, S. G. Finlayson, D. Sontag, et al. (2025)
  Dr tulu: reinforcement learning with evolving rubrics for deep research.
  arXiv preprint arXiv:2511.19399.
  Cited by: [§5](#S5.SS0.SSS0.Px2.p1.1 "Learning from Natural Language Feedback and through Retrospection. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* I. Shenfeld, M. Damani, J. Hübotter, and P. Agrawal (2026)
  Self-distillation enables continual learning.
  arXiv preprint arXiv:2601.19897.
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* T. Shi, S. Chen, B. Jiang, L. Song, L. Yang, and J. Zhao (2026)
  Experiential reinforcement learning.
  arXiv preprint arXiv:2602.13949.
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* T. Shi, Z. Wang, L. Yang, Y. Lin, Z. He, M. Wan, P. Zhou, S. Jauhar, S. Chen, S. Xia, et al. (2024)
  Wildfeedback: aligning llms with in-situ user interactions and feedback.
  arXiv preprint arXiv:2408.15549.
  Cited by: [§2](#S2.p3.1 "Problem Formulation ‣ Aligning Language Models from User Interactions"),
  [§4.1](#S4.SS1.SSS0.Px3.p1.1 "How important is the quality of user conversations? ‣ General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions"),
  [§4.1](#S4.SS1.SSS0.Px4.p2.1 "SDPO vs. SFT. ‣ General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions"),
  [§4.1](#S4.SS1.p1.6 "General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions"),
  [§4](#S4.p3.1 "Experimental Results ‣ Aligning Language Models from User Interactions").
* N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao (2023)
  Reflexion: language agents with verbal reinforcement learning.
  In NeurIPS,
  Cited by: [§5](#S5.SS0.SSS0.Px2.p2.1 "Learning from Natural Language Feedback and through Retrospection. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* C. Snell, D. Klein, and R. Zhong (2022)
  Learning by distilling context.
  arXiv preprint arXiv:2209.15189.
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* Y. Song, L. Chen, F. Tajwar, R. Munos, D. Pathak, J. A. Bagnell, A. Singh, and A. Zanette (2026)
  Expanding the capabilities of reinforcement learning via text feedback.
  arXiv preprint arXiv:2602.02482.
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* M. Stephan, A. Khazatsky, E. Mitchell, A. S. Chen, S. Hsu, A. Sharma, and C. Finn (2024)
  Rlvf: learning from verbal feedback without overgeneralization.
  In ICML,
  Cited by: [§2](#S2.p3.1 "Problem Formulation ‣ Aligning Language Models from User Interactions"),
  [§5](#S5.SS0.SSS0.Px2.p2.1 "Learning from Natural Language Feedback and through Retrospection. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* N. Stiennon, L. Ouyang, J. Wu, D. M. Ziegler, R. Lowe, C. Voss, A. Radford, D. Amodei, and P. Christiano (2020)
  Learning to summarize from human feedback.
  In NeurIPS,
  Cited by: [§4.2](#S4.SS2.p2.1 "Continual Personalization and Adaptation from User Interactions ‣ Experimental Results ‣ Aligning Language Models from User Interactions").
* A. Talmor, J. Herzig, N. Lourie, and J. Berant (2019)
  Commonsenseqa: a question answering challenge targeting commonsense knowledge.
  In NAACL,
  Cited by: [Table 6](#A4.T6 "In Additional Results from Section˜4.1 ‣ Appendix D Additional Experimental Results ‣ Aligning Language Models from User Interactions").
* B. M. Urcelay, A. Krause, and G. Ramponi (2026)
  From words to rewards: leveraging natural language for reinforcement learning.
  In TMLR,
  Cited by: [§2](#S2.p3.1 "Problem Formulation ‣ Aligning Language Models from User Interactions"),
  [§5](#S5.SS0.SSS0.Px2.p1.1 "Learning from Natural Language Feedback and through Retrospection. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* H. Wang, L. Wang, C. Zhang, T. Mao, S. Qin, Q. Lin, S. Rajmohan, and D. Zhang (2026)
  Text2Grad: reinforcement learning from natural language feedback.
  In ICLR,
  Cited by: [§2](#S2.p3.1 "Problem Formulation ‣ Aligning Language Models from User Interactions"),
  [§5](#S5.SS0.SSS0.Px2.p1.1 "Learning from Natural Language Feedback and through Retrospection. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* Y. Wang, X. Ma, G. Zhang, Y. Ni, A. Chandra, S. Guo, W. Ren, A. Arulraj, X. He, Z. Jiang, et al. (2024a)
  Mmlu-pro: a more robust and challenging multi-task language understanding benchmark.
  In NeurIPS,
  Cited by: [§4.1](#S4.SS1.p2.1 "General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions").
* Z. Wang, Y. Dong, O. Delalleau, J. Zeng, G. Shen, D. Egert, J. Zhang, M. N. Sreedhar, and O. Kuchaiev (2024b)
  Helpsteer 2: open-source dataset for training top-performing reward models.
  In NeurIPS,
  Cited by: [§4.2](#S4.SS2.p3.1 "Continual Personalization and Adaptation from User Interactions ‣ Experimental Results ‣ Aligning Language Models from User Interactions").
* J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. (2022)
  Chain-of-thought prompting elicits reasoning in large language models.
  In NeurIPS,
  Cited by: [§1](#S1.p3.1 "Introduction ‣ Aligning Language Models from User Interactions").
* T. Xie, S. Zhao, C. H. Wu, Y. Liu, Q. Luo, V. Zhong, Y. Yang, and T. Yu (2024)
  Text2reward: reward shaping with language models for reinforcement learning.
  In ICLR,
  Cited by: [§5](#S5.SS0.SSS0.Px2.p1.1 "Learning from Natural Language Feedback and through Retrospection. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* Y. A. Yadkori, I. Kuzborskij, A. György, and C. Szepesvári (2024)
  To believe or not to believe your llm: iterative prompting for estimating epistemic uncertainty.
  In NeurIPS,
  Cited by: [Appendix A](#A1.p3.3 "Appendix A A Latent Reward Perspective on SDPO ‣ Aligning Language Models from User Interactions").
* W. Yang, Y. Lin, J. Zhou, and J. Wen (2025)
  Distilling rule-based knowledge into large language models.
  In COLING,
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* Z. Yang, T. Pang, H. Feng, H. Wang, W. Chen, M. Zhu, and Q. Liu (2024)
  Self-distillation bridges distribution gap in language model fine-tuning.
  In ACL,
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* W. Yao, S. Heinecke, J. C. Niebles, Z. Liu, Y. Feng, L. Xue, R. Murthy, Z. Chen, J. Zhang, D. Arpit, et al. (2024)
  Retroformer: retrospective large language agents with policy gradient optimization.
  In ICLR,
  Cited by: [§5](#S5.SS0.SSS0.Px2.p2.1 "Learning from Natural Language Feedback and through Retrospection. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* M. Yuksekgonul, F. Bianchi, J. Boen, S. Liu, P. Lu, Z. Huang, C. Guestrin, and J. Zou (2025)
  Optimizing generative ai by backpropagating language model feedback.
  Nature 639,  pp. 609–616.
  Cited by: [§5](#S5.SS0.SSS0.Px2.p2.1 "Learning from Natural Language Feedback and through Retrospection. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi (2019)
  HellaSwag: can a machine really finish your sentence?.
  In ACL,
  Cited by: [Table 6](#A4.T6 "In Additional Results from Section˜4.1 ‣ Appendix D Additional Experimental Results ‣ Aligning Language Models from User Interactions").
* S. Zhao, Z. Xie, M. Liu, J. Huang, G. Pang, F. Chen, and A. Grover (2026)
  Self-distilled reasoner: on-policy self-distillation for large language models.
  arXiv preprint arXiv:2601.18734.
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").
* W. Zhao, X. Ren, J. Hessel, C. Cardie, Y. Choi, and Y. Deng (2024)
  Wildchat: 1m chatgpt interaction logs in the wild.
  In ICLR,
  Cited by: [§1](#S1.p6.1 "Introduction ‣ Aligning Language Models from User Interactions"),
  [§4.1](#S4.SS1.p1.6 "General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions"),
  [§4](#S4.p3.1 "Experimental Results ‣ Aligning Language Models from User Interactions").
* J. Zhou, T. Lu, S. Mishra, S. Brahma, S. Basu, Y. Luan, D. Zhou, and L. Hou (2023)
  Instruction-following evaluation for large language models.
  arXiv preprint arXiv:2311.07911.
  Cited by: [§4.1](#S4.SS1.p2.1 "General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions").
* R. Zhou, S. Li, A. Zhang, and L. Leqi (2025)
  ExPO: unlocking hard reasoning with self-explanation-guided reinforcement learning.
  In NeurIPS,
  Cited by: [§5](#S5.SS0.SSS0.Px3.p1.1 "Self-Distillation. ‣ Related Work ‣ Aligning Language Models from User Interactions").

## Appendix A A Latent Reward Perspective on SDPO

Typically, the goal of alignment is expressed as maximizing a user’s latent reward r​(x,y)r(x,y). In practice, this reward is unknown, and even under strong assumptions and access to explicit feedback such as pairwise preferences, identifying and optimizing it requires substantial annotation effort.
To provide intuition for the dynamics of SDPO from this traditional alignment perspective, we here consider a highly stylized model of user and language model behavior. While the assumptions underlying this model are clearly idealized, the resulting analysis offers an interesting interpretation of the log-ratio objective in [Equation˜1](#S3.E1 "In Policy Gradient. ‣ Directly Learning from User Interactions via Self-Distillation ‣ Aligning Language Models from User Interactions").

Let the user’s unknown reward function be defined not only over assistant completions r​(x,y)r(x,y) given the conversation history xx but also over user continuations r​(x,y,o)r(x,y,o) given (x,y)(x,y). We then assume the user’s response follows a Boltzmann-rational continuation model

|  |  |  |  |
| --- | --- | --- | --- |
|  | p​(o∣x,y)∝p​(o∣x)​exp⁡(r​(x,y,o)),\displaystyle p(o\mid x,y)\propto p(o\mid x)\exp(r(x,y,o)), |  | (5) |

where p​(o∣x)p(o\mid x) is a prior over oo given xx. This means that the user chooses their next message approximately according to the reward it induces over future continuations of the interaction. Next, we make a simplifying assumption about the behavior of the language model.
We assume that the hindsight distribution πθ​(y∣x,o)\pi\_{\theta}(y\mid x,o) can be interpreted as behaving
*as if* it were a Bayesian posterior, in the sense that it satisfies

|  |  |  |  |
| --- | --- | --- | --- |
|  | πθ​(y∣x,o)∝πθ​(y∣x)​p​(o∣x,y).\displaystyle\pi\_{\theta}(y\mid x,o)\propto\pi\_{\theta}(y\mid x)\,p(o\mid x,y). |  | (6) |

While clearly idealized, this provides a convenient abstraction for reasoning about how conditioning on the user continuation oo reshapes the model’s distribution over responses. Intuitively, the user’s follow-up can be viewed as an observation that favors responses yy that are more compatible with the preferences, constraints, or corrections revealed through the interaction, thereby reweighting
the prior policy πθ​(y∣x)\pi\_{\theta}(y\mid x).
In practice, the attention mechanism in a transformer does not implement Bayesian conditioning in a literal sense. Still similar posterior-style interpretations can be commonly found in the in-context learning literature (e.g., Yadkori et al. ([2024](#bib.bib20 "To believe or not to believe your llm: iterative prompting for estimating epistemic uncertainty")); Luo et al. ([2025](#bib.bib19 "Language models can learn from verbal feedback without scalar rewards"))).

Entertaining this thought and stylized model, we arrive at an interesting observation.
We consider the *sequence-level* self-distillation advantage given by

|  |  |  |  |
| --- | --- | --- | --- |
|  | A​(x,y,o):=log⁡πθ​(y∣x,o)πθ​(y∣x).A(x,y,o):=\log\frac{\pi\_{\theta}(y\mid x,o)}{\pi\_{\theta}(y\mid x)}. |  | (7) |

Using Bayes rule, i.e., [Equation˜6](#A1.E6 "In Appendix A A Latent Reward Perspective on SDPO ‣ Aligning Language Models from User Interactions"), and the fact that r​(x,y)=𝔼o∼p(⋅∣x,y)​[r​(x,y,o)]r(x,y)=\mathbb{E}\_{o\sim p(\cdot\mid x,y)}[r(x,y,o)], we can write the advantage as

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼o∼p(⋅∣x,y)​[log⁡πθ​(y∣x,o)πθ​(y∣x)]\displaystyle\mathbb{E}\_{o\sim p(\cdot\mid x,y)}\left[\log\frac{\pi\_{\theta}(y\mid x,o)}{\pi\_{\theta}(y\mid x)}\right] | =𝔼o∼p(⋅∣x,y)​[log⁡p​(o∣x,y)p​(o∣x)]\displaystyle=\mathbb{E}\_{o\sim p(\cdot\mid x,y)}\left[\log\frac{p(o\mid x,y)}{p(o\mid x)}\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =r​(x,y)−log⁡Z​(x,y),\displaystyle=r(x,y)-\log Z(x,y), |  |

where Z​(x,y)=𝔼o∼p(⋅∣x)​[exp⁡(r​(x,y,o))]Z(x,y)=\mathbb{E}\_{o\sim p(\cdot\mid x)}[\exp(r(x,y,o))] is the partition function from the Boltzmann-rational user model in [Equation˜5](#A1.E5 "In Appendix A A Latent Reward Perspective on SDPO ‣ Aligning Language Models from User Interactions").

This means that maximizing the sequence-level advantage can be viewed as maximizing the user’s latent reward up to an additive normalization term. While this equivalence relies on strong assumptions, it provides an interpretation of SDPO as implicitly optimizing for user-aligned behavior using interaction data alone, without requiring explicit reward supervision.

## Appendix B Gradient Derivation

###### Lemma B.1.

The one-sample approximation,

|  |  |  |  |
| --- | --- | --- | --- |
|  | −𝔼y∼πθ(⋅∣x)​[∑i∇θlog⁡πθ​(yi∣x,y<i)​Ai​(x,y,o)],-\mathbb{E}\_{y\sim\pi\_{\theta}(\cdot\mid x)}\!\left[\sum\_{i}\nabla\_{\!\theta}\log\pi\_{\theta}(y\_{i}\mid x,y\_{<i})\,A\_{i}(x,y,o)\right], |  | (8) |

is an unbiased estimator of the SDPO gradient of [Equation˜3](#S3.E3 "In Self-Distillation. ‣ Directly Learning from User Interactions via Self-Distillation ‣ Aligning Language Models from User Interactions").

###### Proof of [Appendix˜B](#A2 "Appendix B Gradient Derivation ‣ Aligning Language Models from User Interactions").

Fix context xx and let y=(y1,…,yT)y=(y\_{1},\dots,y\_{T}) be sampled autoregressively from πθ\pi\_{\theta}:

|  |  |  |
| --- | --- | --- |
|  | πθ​(y∣x)=∏i=1Tπθ​(yi∣x,y<i).\pi\_{\theta}(y\mid x)=\prod\_{i=1}^{T}\pi\_{\theta}(y\_{i}\mid x,y\_{<i}). |  |

For each position ii, we define

|  |  |  |  |
| --- | --- | --- | --- |
|  | ϕi​(y<i,yi)\displaystyle\phi\_{i}(y\_{<i},{y}\_{i}) | :=∇θlog⁡πθ​(yi∣x,y<i)​Ai​(x,y,o)andψi​(y<i):=𝔼yi∼πθ(⋅∣x,y<i)​[ϕi​(y<i,yi)].\displaystyle:=\nabla\_{\!\theta}\log\pi\_{\theta}({y}\_{i}\mid x,y\_{<i})\,A\_{i}(x,y,o)\quad\text{and}\quad\psi\_{i}(y\_{<i}):=\mathbb{E}\_{{y}\_{i}\sim\pi\_{\theta}(\cdot\mid x,y\_{<i})}[\phi\_{i}(y\_{<i},{y}\_{i})]. |  |

We consider two estimators,

|  |  |  |
| --- | --- | --- |
|  | g^1​(y):=∑i=1Tψi​(y<i),g^2​(y):=∑i=1Tϕi​(y<i,yi).\widehat{g}\_{1}(y):=\sum\_{i=1}^{T}\psi\_{i}(y\_{<i}),\qquad\widehat{g}\_{2}(y):=\sum\_{i=1}^{T}\phi\_{i}(y\_{<i},y\_{i}). |  |

By definition, 𝔼​[g^1​(y)]=∇θℒSDPO​(θ)\mathbb{E}[\widehat{g}\_{1}(y)]=\nabla\_{\!\theta}\,\mathcal{L}\_{\mathrm{SDPO}}(\theta) is the analytic gradient from [Equation˜3](#S3.E3 "In Self-Distillation. ‣ Directly Learning from User Interactions via Self-Distillation ‣ Aligning Language Models from User Interactions"). 𝔼​[g^2​(Y)]\mathbb{E}[\widehat{g}\_{2}(Y)] is the gradient estimator from [Equation˜8](#A2.E8 "In Lemma B.1. ‣ Appendix B Gradient Derivation ‣ Aligning Language Models from User Interactions") in [Appendix˜B](#A2 "Appendix B Gradient Derivation ‣ Aligning Language Models from User Interactions").

In the following, we prove 𝔼​[g^1​(y)]=𝔼​[g^2​(y)]\mathbb{E}[\widehat{g}\_{1}(y)]=\mathbb{E}[\widehat{g}\_{2}(y)] assuming 𝔼​[‖g^1​(y)‖]<∞\mathbb{E}[\|\widehat{g}\_{1}(y)\|]<\infty (so that all expectations exist).
Fix ii. By construction, yi∣y<i∼πθ(⋅∣x,y<i)y\_{i}\mid y\_{<i}\sim\pi\_{\theta}(\cdot\mid x,y\_{<i}) so that

|  |  |  |
| --- | --- | --- |
|  | 𝔼yi​[ϕi​(y<i,yi)∣y<i]=𝔼yi∼πθ(⋅∣x,y<i)​[ϕi​(y<i,yi)]=ψi​(y<i).\mathbb{E}\_{y\_{i}}[\phi\_{i}(y\_{<i},y\_{i})\mid y\_{<i}]=\mathbb{E}\_{y\_{i}\sim\pi\_{\theta}(\cdot\mid x,y\_{<i})}[\phi\_{i}(y\_{<i},y\_{i})]=\psi\_{i}(y\_{<i}). |  |

Taking expectation and using the tower property,

|  |  |  |
| --- | --- | --- |
|  | 𝔼y<i,yi​[ϕi​(y<i,yi)]=𝔼y<i​[ψi​(y<i)].\mathbb{E}\_{y\_{<i},y\_{i}}[\phi\_{i}(y\_{<i},y\_{i})]=\mathbb{E}\_{y\_{<i}}[\psi\_{i}(y\_{<i})]. |  |

Finally, by linearity of expectation,

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[g^2​(y)]=∑i=1T𝔼​[ϕi​(y<i,yi)]=∑i=1T𝔼​[ψi​(y<i)]=𝔼​[g^1​(y)].\mathbb{E}[\widehat{g}\_{2}(y)]=\sum\_{i=1}^{T}\mathbb{E}[\phi\_{i}(y\_{<i},y\_{i})]=\sum\_{i=1}^{T}\mathbb{E}[\psi\_{i}(y\_{<i})]=\mathbb{E}[\widehat{g}\_{1}(y)]. |  |

∎

## Appendix C Experimental Details

### Hyperparameters

We report the hyperparameters for SDPO across all experiments in [Table˜5](#A3.T5 "In Hyperparameters ‣ Appendix C Experimental Details ‣ Aligning Language Models from User Interactions").

Table 5: Hyperparameters used for SDPO in each setup. Note that the hyperparameters of SDPO in [Section˜4.1](#S4.SS1 "General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions") were kept the same across all models. The learning rate was chosen by sweeping over {1,2,3,5}×10−6\smash{\{1,2,3,5\}\times 10^{-6}} for Qwen3-4B and then fixing the setup for all models. For the SFT checkpoint in [Table˜4](#S4.T4 "In How important is the quality of user conversations? ‣ General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions"), we similarly swept over {1,2,3,5}×10−6\smash{\{1,2,3,5\}\times 10^{-6}} with best results for 2×10−6\smash{2\times 10^{-6}}. In [Section˜4.2](#S4.SS2 "Continual Personalization and Adaptation from User Interactions ‣ Experimental Results ‣ Aligning Language Models from User Interactions"), SDPO appeared insensitive to hyperparameter choices in early experiments (especially, learning rate), and were fixed to the setup below without additional systematic tuning.

|  |  |  |  |
| --- | --- | --- | --- |
| Hyperparameter | Section 4.1  ([Figures˜3](#S3.F3 "In Self-Distillation as an Alignment Objective. ‣ Directly Learning from User Interactions via Self-Distillation ‣ Aligning Language Models from User Interactions"), [2](#S4.T2 "Table 2 ‣ General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions"), [3](#S4.T3 "Table 3 ‣ How important is the quality of user conversations? ‣ General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions") and [4](#S4.T4 "Table 4 ‣ How important is the quality of user conversations? ‣ General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions")) | Section 4.2  ([Figures˜5](#S4.F5 "In Continual Personalization and Adaptation from User Interactions ‣ Experimental Results ‣ Aligning Language Models from User Interactions") and [4](#S4.F4 "Figure 4 ‣ Continual Personalization and Adaptation from User Interactions ‣ Experimental Results ‣ Aligning Language Models from User Interactions")) | Section 4.2  ([Figure˜6](#S4.F6 "In Continual Personalization and Adaptation from User Interactions ‣ Experimental Results ‣ Aligning Language Models from User Interactions")) |
| Models | Qwen3-4B, Qwen3-8B,  Olmo3-7B-Instruct-SFT,  Olmo3-7B-Instruct-DPO | Qwen3-4B | Qwen3-8B |
| Max prompt length | 2048 | 1024 | 2048 |
| Max compl. length | 2048 | 258 | 2048 |
| Learning rate | 2×10−62\times 10^{-6} | 5×10−65\times 10^{-6} | 5×10−65\times 10^{-6} |
| Batch size | 32 | 16 | 32 |
| Epochs | 2 | 1 | 1 |
| Warm-up ratio | 5% | 0 | 0 |
| LR schedule | Cosine | Constant | Constant |
| Optimizer | AdamW (8-bit) | AdamW | AdamW (8-bit) |
| Temperature | 1.0 | 1.0 | 1.0 |

##### Benchmarks.

For all reported benchmarks, we used the default settings. For AlpacaEval 2.0 and ArenaHard-v2, completions were judged using the defaults “Weighted Alpaca Eval GPT-4 Turbo” and “GPT-4.1”, respectively. IFEval results are reported for prompt-level loose. MMLU-Pro is evaluated with the recommended chain-of-thought 5-shot settings.

## Appendix D Additional Experimental Results

### Additional Results from [Section˜4.1](#S4.SS1 "General Alignment from Real-World User Conversations ‣ Experimental Results ‣ Aligning Language Models from User Interactions")

We evaluate the SDPO results for Qwen3-8B on pre-training benchmarks in [Table˜6](#A4.T6 "In Additional Results from Section˜4.1 ‣ Appendix D Additional Experimental Results ‣ Aligning Language Models from User Interactions"). Overall, we observe no changes in performance.

|  |  |  |  |
| --- | --- | --- | --- |
|  | TruthfulQA (MC1) Acc ±\pm StdErr | HellaSwag Acc ±\pm StdErr | CommonsenseQA Acc ±\pm StdErr |
| Qwen3-8B | 0.366 ±\pm 0.0169 | 0.5717 ±\pm 0.0049 | 0.7846 ±\pm 0.0118 |
| SDPO | 0.3647 ±\pm 0.0169 | 0.5710 ±\pm 0.0049 | 0.7871 ±\pm 0.0117 |

Table 6: SDPO preserves performance on pre-training benchmarks. We additionally evaluate SDPO for Qwen3-8B on the standard pre-training benchmarks TruthfulQA (Lin et al., [2022](#bib.bib56 "Truthfulqa: measuring how models mimic human falsehoods")), HellaSwag (Zellers et al., [2019](#bib.bib57 "HellaSwag: can a machine really finish your sentence?")), and CommonsenseQA (Talmor et al., [2019](#bib.bib58 "Commonsenseqa: a question answering challenge targeting commonsense knowledge")).

### Additional Results from [Section˜4.2](#S4.SS2 "Continual Personalization and Adaptation from User Interactions ‣ Experimental Results ‣ Aligning Language Models from User Interactions")

[Figure˜9](#A4.F9 "In Additional Results from Section˜4.2 ‣ Appendix D Additional Experimental Results ‣ Aligning Language Models from User Interactions") includes the additional results for the personalization results from [Section˜4.2](#S4.SS2 "Continual Personalization and Adaptation from User Interactions ‣ Experimental Results ‣ Aligning Language Models from User Interactions"). Similarly to [Figure˜5](#S4.F5 "In Continual Personalization and Adaptation from User Interactions ‣ Experimental Results ‣ Aligning Language Models from User Interactions") in the main text, we here consider the adaptation of SDPO for Qwen3-4B to a user with a preference profile across three dimensions detailed/concise, casual/professional, beginner/expert. Across all user profiles, we observe that SDPO is able to quickly adapt from only a handful of user interactions, sometimes even exceeding the performance of the in-context oracle that is queried with the user preferences in context.

!(/html/2603.12273/assets/images/personalization/pure_winrate/detailed_professional_expert.png)

!(/html/2603.12273/assets/images/personalization/pure_winrate/concise_professional_expert.png)

!(/html/2603.12273/assets/images/personalization/pure_winrate/detailed_casual_beginner.png)

!(/html/2603.12273/assets/images/personalization/pure_winrate/concise_casual_expert.png)

!(/html/2603.12273/assets/images/personalization/pure_winrate/detailed_professional_beginner.png)

!(/html/2603.12273/assets/images/personalization/pure_winrate/detailed_casual_expert.png)

Figure 9: Additional personalization results from [Section˜4.2](#S4.SS2 "Continual Personalization and Adaptation from User Interactions ‣ Experimental Results ‣ Aligning Language Models from User Interactions") with Qwen3-4B. The win rate is computed against the base model and judged by Qwen3-8B. The In-Context Oracle baseline is obtained by prompting Qwen3-4B directly with the desired writing style.

### User Profiles, Prompts, and Judging in [Section˜4.2](#S4.SS2 "Continual Personalization and Adaptation from User Interactions ‣ Experimental Results ‣ Aligning Language Models from User Interactions")

To generate user responses to the assistant’s completions in [Section˜4.2](#S4.SS2 "Continual Personalization and Adaptation from User Interactions ‣ Experimental Results ‣ Aligning Language Models from User Interactions"), we use the user profiles below as system prompts and then query the user model (Qwen3-8B or Claude Haiku 4.5) to generate a response with this persona. For the experiments in [Figure˜6](#S4.F6 "In Continual Personalization and Adaptation from User Interactions ‣ Experimental Results ‣ Aligning Language Models from User Interactions"), smaller models, such as Qwen3-8B and Qwen3-14B, became too unreliable to act as the user simulator and the judge for the prompts from HelpSteer2 and the more complex user profiles such as *Less Filler Praise & Sycophancy*. We therefore used Claude Haiku 4.5 instead.

For the evaluation of the win rate against the base model, we again add the personas to the system prompt and judge the outputs. Here, each pair of responses is judged twice with flipped positions to remove the position bias from the evaluation, and we evaluate the win rate on 256 held-out prompts in each of the experiments.

#### User Profiles

The user profiles used to simulate user responses in [Section˜4.2](#S4.SS2 "Continual Personalization and Adaptation from User Interactions ‣ Experimental Results ‣ Aligning Language Models from User Interactions"):

Don’t Like Emojis and Icons
:   USER PROFILE: You are playing the role of a user who dislikes emojis and icons in assistant responses.

Less Filler Praise & Sycophancy
:   USER PROFILE: You are playing the role of a user that specifically dislikes when the assistant responses include filler praise at the beginning (such as ’Good question.’ or ’Perfect!’) or fillers at the end (such as I hope this helps!’ or Let me know if there is anything else I can help you with.’). You strongly prefer when the assistant responses directly without unnecessary additions at the beginning or end.

Answer Directly, Reduce Formatting
:   USER PROFILE: You are playing the role of a user who prefers concise responses and dislikes long lists and excessive markdown formatting (such as \*\* \*\* and ###). You prefer plain text that is short and gets to the point quickly.

Concise/Casual/Beginner
:   USER PROFILE: You are playing the role of a user who specifically prefers CONCISE, CASUAL, and BEGINNER-FRIENDLY responses. You want brief context and clear explanations, avoiding long, formal, or technically dense answers.

Detailed/Professional/Expert
:   USER PROFILE: You are playing the role of a user who specifically prefers DETAILED, PROFESSIONAL, and EXPERT-LEVEL responses. You want a structured, impersonal, and analytical presentation with complex sentence structures and sophisticated wording.

#### User Model Prompt and Judge Prompt

The prompts used to simulate user responses with Qwen3-8B and Claude Haiku 4.5. Further below, the prompt used to evaluate the reference completions from the base model against the trained models.

User Model Prompt

[SYSTEM MESSAGE]
USER PROFILE: ⟨USER\_PROFILE⟩
You are simulating the user’s next message in a chat with an AI assistant.
Rules:
- Respond as the user described in the USER PROFILE, using straightforward language.
- Evaluate ONLY the assistant’s response with respect to the preferences in the USER PROFILE.
- Do NOT answer the original user request yourself.
- Respond very briefly.
- Do NOT give feedback unrelated to the USER PROFILE.
- Output ONLY the user message text (no labels, no preface).
- If the assistant’s response matches the USER PROFILE well, you may say so briefly.
[USER MESSAGE]
Original user request:
⟨USER\_PROMPT⟩
Assistant response:
⟨ASSISTANT\_RESPONSE⟩
Write the user’s next message.

Judge Prompt

[SYSTEM MESSAGE]
USER PROFILE: ⟨USER\_PROFILE⟩
You are acting as a strict evaluator of WHICH response better matches your preference described in the USER PROFILE.
You must follow these rules:
- Judge ONLY style, tone, formatting, verbosity, and complexity relative to the persona.
- Do NOT judge factual correctness.
- Do NOT rewrite responses.
- Output exactly ONE character: A, B, or C.
- C means tie/uncertain.
[USER MESSAGE]
User prompt:
⟨USER\_PROMPT⟩
Response A:
⟨RESPONSE\_A⟩
Response B:
⟨RESPONSE\_B⟩
Which response do you prefer as this user? Output only A, B, or C.
