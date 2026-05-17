---
arxiv: '2512.03442'
authors:
- Xingrun Xing
- Zhiyuan Fan
- Jie Lou
- Guoqi Li
- Jiajun Zhang
- Debing Zhang
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'PretrainZero: Reinforcement Active Pretraining'
url: https://arxiv.org/abs/2512.03442
year: 2025
---

# PretrainZero: Reinforcement Active Pretraining

Xingrun Xing1,2, Zhiyuan Fan2, Jie Lou2🖂, Guoqi Li1, Jiajun Zhang1🖂, Debing Zhang2
  
1 Institute of Automation, Chinese Academy of Sciences
  
2 Xiaohongshu Inc.
  
loujie0822@aliyun.com, jjzhang@nlpr.ia.ac.cn

###### Abstract

Mimicking human behavior to actively learning from general experience and achieve artificial general intelligence has always been a human dream.
Recent reinforcement learning (RL) based large-thinking models demonstrate impressive expert-level abilities, i.e., software and math, but still rely heavily on verifiable rewards in specific domains, placing a significant bottleneck to extend the performance boundary of general reasoning capabilities.
In this work, we propose PretrainZero, a reinforcement active learning framework built on the pretraining corpus to extend RL from domain-specific post-training to general pretraining. PretrainZero features the following characteristics:
1) Active pretraining:
inspired by the active learning ability of humans, PretrainZero learns a unified reasoning policy to actively identify reasonable and informative contents from pretraining corpus, and reason to predict these contents by RL.
2) Self-supervised learning:
without any verifiable labels, pretrained reward models, or supervised fine-tuning, we directly pretrain reasoners from 3∼303\sim 30B base models on the general Wikipedia corpus using RL, significantly breaking the verification data-wall for general reasoning.
3) Verification scaling:
by tackling increasingly challenging masked spans, PretrainZero substantially enhances the general reasoning abilities of pretrained base models. In reinforcement pretraining, PretrainZero improves Qwen3-4B-Base for 8.43, 5.96 and 10.60 on MMLU-Pro, SuperGPQA and math average benchmarks. In post-training, the pretrained models can also serve as reasoning foundation models for downstream RLVR tasks.

!(/html/2512.03442/assets/fig1.png)

Figure 1: 
Reinforcement Pre-Training (RLPT) performance in pre-training and post-training stages.

## 1 Introduction

Recent large language models (LLMs) have achieved human-level expertise in specific domains, particularly through large-scale self-supervised learning in pretraining [scaling:law, achiam2023gpt-4] and Reinforcement Learning (RL) [deepseekr1, dapo, chu2025sft] in post-training.
During pretraining, self-supervised learning with a fixed next-token prediction paradigm allows models to leverage large-scale, low-cost data to improve general capabilities effectively.
In contrast, the post-training RL faces a severe data-wall: Reinforcement Learning with Verifiable Rewards (RLVR) [deepseekr1, yue2504does] requires domain-specific verifiers to label training samples, and Reinforcement Learning from Human Feedback (RLHF) [instructgpt, bai2022training], relying on reward models and humans, can only train limited steps to avoid reward hacking.
This motivates a natural direction—performing reinforcement learning [dong2025reinforcement, li2025reinforcement] in a self-supervised pretraining manner [gpt3], in order to use inexpensive pretraining data to extend RLVR and overcome this data-wall.

However, formulating the self-supervised pretraining as RLVR tasks is non-trivial. Towards this goal, this work first investigates stand-alone Reinforcement Learning Pre-Training (RLPT) [dong2025reinforcement]
according to three principles: 1) Comprehensiveness: we establish both baselines including masked token prediction and next token prediction as the reasoning objective [dong2025reinforcement]. 2) Full self-supervision: we exclude any additional Supervised Fine-Tuning (SFT) cold-start and reward models [li2025reinforcement]. 3) Generalization: we avoid Question–Answer formats or synthetic chain-of-thought (CoT) datasets, like OmniMath [omnimath], and train directly on general-domain Wikipedia [bert].
Experimental results demonstrate that the vanilla RLPT fails to emerge high-quality CoT: the low information density of pretraining corpus leads to inefficient learning, and the presence of noisy or incorrect tokens often causes training collapse.

This work proposes the first stand-alone RLPT method to extend RLVR on real-world pretraining corpus, termed PretrainZero. This is achieved by mimicking the human active-learning behavior [yang2025active, settles2009active]:
humans can actively learn from a broad range of experiences, selectively focusing on informative elements and unfamiliar concepts. This allows effective learning even when the underlying experiences are noisy or low in information density. In contrast, current large language models—whether through supervised or reinforcement pretraining—rely on fixed prediction patterns, such as next-token or off-policy selected masked-token prediction. These rigid learning patterns limit their efficiency [berglund2309reversal] and prevent them from leveraging data as flexibly as humans do.

Inspired by this, this work proposes a reinforcement active learning framework in order to learn from real-world pretraining distributions.
Specifically, we introduce an on-policy mask generation task as an auxiliary active-learning objective for the masked-span prediction task. The mask generation policy actively explores informative, verifiable, and not-yet-mastered content within the pretraining data and selects it as learning targets. Consequently, the masked-span prediction policy learns to produce CoT reasoning and recover these informative spans.
For optimization, we formulate a min–max bilevel reinforcement learning objective, where each batch is jointly optimized using GRPO [grpo] over both the mask generation and masked-span prediction tasks.
Different from unsupervised RL methods such as self-play [huang2025r] and test-time scaling [snell2024scaling], PretrainZero provides a verifiable RL scaling mechanism grounded in real data in a self-supervising manner. This avoids the severe hallucination issues in self-play and test-time scaling, where majority voting from model-generated answers serves as supervision and ultimately leads to collapse in prolonged RL training [liu2025prorl].

As shown in Fig. [1](#S0.F1 "Figure 1 ‣ PretrainZero: Reinforcement Active Pretraining"), we evaluate RLPT for 2000 steps in pretraining and add general RLVR in post-training. For a Qwen3-4B-Base [yang2025qwen3] model, PretrainZero consistently improves 8.43, 5.96, and 10.60 on MMLU-Pro [mmlupro], SuperGPQA [supergpqa], and math average benchmarks during reinforcement pretraining.
After general RLVR [ma2025general] in the post-training stage, these improvements remain substantial, with final improvements of 2.35, 3.04, and 2.81 on MMLU-Pro, SuperGPQA, and math average respectively.

Our contributions are summarized as follows:

* •

  We introduce PretrainZero, the first stand-alone RLPT method to operate RLVR on real-world pretraining corpus, enabling general-domain and large-scale reinforcement learning trained directly from base models using only pretraining data as grounding.
* •

  We propose the reinforcement active pretraining mechanism inspired by human active learning. The introduced mask-generation objective enables the model to anticipate what information should be learned actively, ensuring effective training under low–information-density pretraining corpus.
* •

  We evaluate PretrainZero in both the pretraining and post-training stages, showing that it effectively mitigates the general reasoning data-wall with pretraining data, and finally the pretrained reasoning models can serve as the reasoning foundation models for general downstream RLVR tasks.

!(/html/2512.03442/assets/x1.png)

Figure 2: 
An overview of Reinforcement Active Pretraining. Compared with vanilla RLPT, PretrainZero actively explores and learns from the informative contexts on the pretraining corpus.

## 2 Preliminary

In order to learn from the pretraining corpus, traditional self-supervised pretraining adopts language modeling objectives to capture linguistic patterns and contextual dependencies. Recently, the emerging reinforcement pretraining constructs verifiable data through token prediction to learn the reasoning process within concepts. We briefly review different learning patterns in this section.

### 2.1 Self-Supervised Pretraining

Given the context, traditional self-supervised pretraining tasks include masked token prediction (MTP) [gpt2] and next token prediction (NTP) [bert].
As shown in Eq. [1](#S2.E1 "In 2.1 Self-Supervised Pretraining ‣ 2 Preliminary ‣ PretrainZero: Reinforcement Active Pretraining"), the NTP task predict the identity tokens xtx\_{t} in each location given their preceding context x<tx\_{<t} under an auto-regressive pattern:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒥NTP​(θ)=∑t=1Tlog⁡πθ​(xt∣x<t),\mathcal{J}\_{\text{NTP}}(\theta)=\sum\_{t=1}^{T}\log\pi\_{\theta}(x\_{t}\mid x\_{<t}), |  | (1) |

where xx is the token sequence with length TT and θ\theta is the pretrained model parameters. As the counterpart, masked token prediction task jointly leverages both the preceding and succeeding contexts xm<t,t>nx\_{m<t,t>n} to predict the masked tokens xm≤t≤nx\_{m\leq t\leq n}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒥MTP​(θ)=∑t=mnlog⁡πθ​(xm≤t≤n∣xm<t,t>n).\mathcal{J}\_{\text{MTP}}(\theta)=\sum\_{t=m}^{n}\log\pi\_{\theta}(x\_{m\leq t\leq n}\mid x\_{m<t,t>n}). |  | (2) |

Supported by self-supervised pretraining, modern LLMs [gpt4o, liu2024deepseekv3] successfully scale up pretraining on massive Internet data. In this work, we simulate both masked token prediction and next token prediction as reinforcement reasoning tasks to explore more general RL approach [ma2025general].

### 2.2 Reinforcement Pretraining

Recent Reinforcement Pre-Training (RPT) [dong2025reinforcement] extends reinforcement learning into the pretraining corpus, constructing verifiable training data directly from the pretraining corpus and thereby alleviating the reliance on costly annotations and specific environments for verification.
Specifically, RPT introduces the next-token reasoning task: given a sequence xx, one token xtx\_{t} is treated as ground-truth and its preceding tokens x<tx\_{<t} as context for the generated output, oto\_{t}.
Unlike the self-supervised NTP task, where the model directly predicts the next token, RPT [dong2025reinforcement] first produces a chain-of-thought reasoning process [deepseekr1] before generating the final predicted token. In optimization, RPT applies GRPO algorithm with group size GG, and uses the exact match verifiable reward rtir^{i}\_{t} between prediction x^ti\hat{x}^{i}\_{t} and ground-truth xtix\_{t}^{i}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | rti={1if ​x^ti​(x<ti)=xti0otherwise,r^{i}\_{t}=\begin{cases}1&\text{if }\hat{x}^{i}\_{t}(x\_{<t}^{i})=x\_{t}^{i}\\ 0&\text{otherwise}\end{cases}, |  | (3) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒥RPT​(θ)=𝔼(x≤t)∼𝒟,{oti}i=1G∼πθ(⋅∣x<t)​[rti].\mathcal{J}\_{\text{RPT}}(\theta)=\mathbb{E}\_{(x\_{\leq t})\sim\mathcal{D},\ \{o^{i}\_{t}\}\_{i=1}^{G}\sim\pi\_{\theta}(\cdot\mid x\_{<t})}\left[r^{i}\_{t}\right]. |  | (4) |

Discussion on the weakness of vanilla reinforcement pretraining.
Despite the simplicity of RPT and its potential to extend the RLVR-style method into pretraining, several significant concerns also emerge, making vanilla RPT unsuitable for practical pretraining settings:

* •

  Robustness on real-world corpus: although RPT demonstrates improvements on synthetic dataset OmniMath, real-world pretraining data with more noise [bert] often causes training collapsion.
* •

  Training from base models: vanilla RPT depends on post-training distillation; Other explorations usually rely on SFT cold-start, external reward models, significantly increasing the complexity.
* •

  Learning effectiveness: due to the low information density in pretraining corpus, simple token selection methods fails to identify informative content, hindering effective optimization.
* •

  Training efficiency: unlike self-supervised NTP that predicts all tokens in parallel, RPT predicts one single token in each sample, yielding limited learning information per step.

## 3 Reinforcement Active Pretraining

To solve these questions, this work first establishes a Reinforcement Learning Pre-Training (RLPT) baseline on the widely used general domain WikiPedia dataset building upon the Qwen3-4B-Base model.
Based on the empirical observations, we then propose a unified and active pretraining task to confirm the general and practical reinforcement pretraining.

### 3.1 Reinforcement Pretraining Baselines

!(/html/2512.03442/assets/fig3.png)

Figure 3: 
MMLU-Pro performance for foundational RLPT methods. (a) Reinforcement next token prediction and reinforcement masked token prediction. (b) Reinforcement next token prediction with entropy and random token selection.

We establish an RLPT baseline with three masking strategies for training corpus. The model is required to predict masked tokens xtx\_{t} through CoT reasoning, receiving binary rewards from exact match with ground truth:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒥RLPT​(θ)=𝔼x∼𝒟,t,{oti}i=1G∼πθ(⋅∣x\t)​[𝕀​[x^ti=xti]].\mathcal{J}\_{\text{RLPT}}(\theta)=\mathbb{E}\_{x\sim\mathcal{D},\ t,\ \{o^{i}\_{t}\}\_{i=1}^{G}\sim\pi\_{\theta}(\cdot\mid x\_{\backslash t})}\left[\mathbb{I}[\hat{x}\_{t}^{i}=x\_{t}^{i}]\right]. |  | (5) |

Three mask prediction strategies for RLPT are investigated:

* •

  Random Next Token Reasoning. The sequence is randomly truncated and the last token before truncation is masked for prediction. For each sample, the model first generates a CoT and only one selected token is predicted according to the generated CoT.
* •

  Random Masked Span Reasoning. A word span containing several tokens in the sequence is randomly selected and masked [joshi2020spanbert], allowing the CoT to predict more than one tokens.
* •

  Entropy-based Next Token Reasoning. The token with the top 20% entropy in the sequence is randomly selected and masked, with all subsequent tokens truncated, which consists with RPT.

#### Empirical Observation.

Preliminary experiments are conducted to evaluate these masking strategies on the Wikipedia corpus, with performance measured by MMLU-Pro. As shown in Figure [3](#S3.F3 "Figure 3 ‣ 3.1 Reinforcement Pretraining Baselines ‣ 3 Reinforcement Active Pretraining ‣ PretrainZero: Reinforcement Active Pretraining") (left), Random NPT RLPT outperforms Random Mask RLPT with more stable training dynamics:

Findings 1. Although Mask RLPT increases the predicted tokens, the vanilla random word-span selection strategy cannot effectively capture richer semantics in pretraining.

To further investigate the effect of token selection, Random NPT is compared with Entropy NPT, where the token with higher entropy is selected for masking. As shown in Figure [3](#S3.F3 "Figure 3 ‣ 3.1 Reinforcement Pretraining Baselines ‣ 3 Reinforcement Active Pretraining ‣ PretrainZero: Reinforcement Active Pretraining") (right), Entropy NPT leads to training collapse and rapid reward degradation. At the position marked stop optimization, the reward signal becomes degenerate—all samples within a group yield either 0 or 1 accuracy.
The reason is the data quality discrepancy between the synthetic and raw corpus. While entropy-based selection performs well on OmniMath (in RPT), a high-quality synthetic dataset where high-entropy tokens consistently represent challenging but learnable patterns, the same strategy fails on Wikipedia. Raw Wikipedia data contains noise and inconsistencies, causing high-entropy tokens to be either genuinely difficult or simply noisy and unpredictable, which creates unstable learning signals:

Findings 2. In real-world pretraining data distributions, selecting high-entropy tokens is no more effective than a random selection strategy, and learning actively from noisy data is necessary.

### 3.2 Active Pretraining Tasks

!(/html/2512.03442/assets/x2.png)

Figure 4: 
Pretraining Mask Prediction and Mask Generation tasks with GRPO.

The limited performance of these passive masking strategies motivates a more informative and effective learning approach. Consider how humans learn: students focus on informative and valuable content in their experience that maximizes their improvement, rather than randomly selecting practice materials.
Inspired by the active learning behavior, we propose an active masking strategy where the model learns to identify beneficial masking positions during training. Rather than relying on fixed heuristics like random sampling or entropy thresholding, the model discovers which tokens provide the strongest learning signals. The training process consists of two tasks for the shared LLM, as shown in Fig. [4](#S3.F4 "Figure 4 ‣ 3.2 Active Pretraining Tasks ‣ 3 Reinforcement Active Pretraining ‣ PretrainZero: Reinforcement Active Pretraining"):

Mask Generation: given a text sequence ss from pretraining data 𝒟\mathcal{D}, the pretraining LLM πω\pi\_{\omega} first generates a thinking process and then generates a word span, m∼πω(⋅∣s)m\sim\pi\_{\omega}(\cdot\mid s), to mask in this sequence.
As shown in Fig. [5](#S3.F5 "Figure 5 ‣ 3.2 Active Pretraining Tasks ‣ 3 Reinforcement Active Pretraining ‣ PretrainZero: Reinforcement Active Pretraining"), we initially prompt the policy πω(⋅∣s)\pi\_{\omega}(\cdot\mid s) to generate a span mask with one or several words verifiable for reasoning.
During pretraining, the policy πω\pi\_{\omega} continuously learns to explore and capture semantic contents from the noisy pretraining corpus by RL.
During the early training stage, the mask prediction policy is relatively weak and requires explicit clues, while in later stages, it needs to focus on the harder and unsolved words and domains.

Mask Prediction: We introduce the masked span prediction as a verifiable reinforcement learning task. Different from next token reasoning, a single CoT process predicts multiple masked tokens in a continuous span, s[p:q]s\_{[p:q]}. Given the generated mask m∼πω(⋅∣s)m\sim\pi\_{\omega}(\cdot\mid s), we replace the word span s[p:q]s\_{[p:q]} with the mark [mask] in the sequence, and then recover the masked content through CoT reasoning.
As shown in Fig. [5](#S3.F5 "Figure 5 ‣ 3.2 Active Pretraining Tasks ‣ 3 Reinforcement Active Pretraining ‣ PretrainZero: Reinforcement Active Pretraining"), we prompt the policy ψω(⋅∣m,s)\psi\_{\omega}(\cdot\mid m,s) to directly generate a CoT at the initial stage before the final mask prediction x^∼ψω(⋅∣m,s)\hat{x}\sim\psi\_{\omega}(\cdot\mid m,s).
During optimization, the CoT reasons verifiable and semantic targets from the prefixed mask generation task.

Prompt Example: Mask Generation / Prediction

Generate a mask to mask important words in the following paragraph, satisfying the requirements below:
1) The mask should mask one or more entities in the passage. The masked words should be continuous.
2) The masked words should exactly match words in the original passage.
3) The masked words could be predicted according to the context. The difficulty to predict should be moderately challenging for you, so the answer would be short and as unique as possible.
Paragraph: <paragraph>
The final generated masked words must be placed inside `\mask{}`.

There is a passage with masked words by [mask]:
<paragraph with mask>
Please reason step by step, and put the predicted masked words within `\boxed{}`.

Figure 5: Prompt for Mask Generation and Prediction.

### 3.3 Reinforcement Active Learning

Active learning objective. We cast mask generation and mask prediction as a coupled adversarial process, implemented with a shared LLM parameterized by ω\omega. The generator πω′(⋅∣s)\pi\_{\omega^{\prime}}(\cdot\mid s) proposes masking patterns, while the predictor ψω(⋅∣m,s)\psi\_{\omega}(\cdot\mid m,s) seeks to recover the masked content. Based on the final mask prediction rewards R​(s,m,x^)R(s,m,\hat{x}), this interaction is governed by the objective:

|  |  |  |  |
| --- | --- | --- | --- |
|  | J​(ω):=𝔼s∼𝒟,m∼πω′(⋅∣s)​[𝔼x^∼ψω(⋅∣m,s)​[R​(s,m,x^)]],J(\omega)\;:=\;\mathbb{E}\_{s\sim\mathcal{D},\;m\sim\pi\_{\omega^{\prime}}(\cdot\mid s)}\bigg[\,\mathbb{E}\_{\hat{x}\sim\psi\_{\omega}(\cdot\mid m,s)}\big[\,R(s,m,\hat{x})\,\big]\bigg], |  | (6) |

which evaluates the predictor’s performance under the generator’s masking strategy.
To encourage increasingly informative and challenging masks, we define the generator’s
objective as V​(ω)=minω⁡J​(ω)V(\omega)=\min\_{\omega}J(\omega), while the predictor optimizes in the
opposite direction, i.e., arg⁡maxω⁡J​(ω)\arg\max\_{\omega}J(\omega), thereby forming a coupled
min–max formulation:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ω⋆=a​r​g​maxω⁡V​(ω)=arg⁡minω′∈Ω⁡maxω∈Ω⁡𝔼s∼𝒟​[𝔼m∼πω′(⋅∣s),x^∼ψω(⋅∣m,s)​[R​(s,m,x^)]].\omega^{\star}=arg\max\_{\omega}V(\omega)=\arg\min\_{\omega^{\prime}\in\Omega}\max\_{\omega\in\Omega}\mathbb{E}\_{s\sim\mathcal{D}}\Big[\mathbb{E}\_{m\sim\pi\_{\omega^{\prime}}(\cdot\mid s),\,\hat{x}\sim\psi\_{\omega}(\cdot\mid m,s)}[R(s,m,\hat{x})]\Big]. |  | (7) |

This adversarial min–max structure naturally mirrors the principle of active learning, where the generator actively selects reasonable and informative masks to probe the model’s weaknesses, thereby driving the predictor toward improved robustness and generalization.

Reinforcement optimization.
To optimize the min–max active-learning objective in Eq. ([7](#S3.E7 "In 3.3 Reinforcement Active Learning ‣ 3 Reinforcement Active Pretraining ‣ PretrainZero: Reinforcement Active Pretraining")), we implement both the mask prediction and mask generation as RL problems. For the prediction policy ψω​(x^∣m,s)\psi\_{\omega}(\hat{x}\mid m,s), the reward is simply defined as an exact match between the predicted token span x^i\hat{x}^{i} and ground-truth xix^{i}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | rpredi=R​(s,m,x^)=𝕀​[x^i=xi],r^{i}\_{\mathrm{pred}}=R(s,m,\hat{x})=\mathbb{I}[\hat{x}^{i}=x^{i}], |  | (8) |

which directly optimizes the inner maximization 𝔼x^∼ψω(⋅∣m,s)​[R​(s,m,x^)]\mathbb{E}\_{\hat{x}\sim\psi\_{\omega}(\cdot\mid m,s)}[R(s,m,\hat{x})] in Eq. ([6](#S3.E6 "In 3.3 Reinforcement Active Learning ‣ 3 Reinforcement Active Pretraining ‣ PretrainZero: Reinforcement Active Pretraining")).
For the generation policy πω′​(m∣s)\pi\_{\omega^{\prime}}(m\mid s), the reward is defined as the negative prediction accuracy under its own masks:

|  |  |  |  |
| --- | --- | --- | --- |
|  | rgenj=1−𝔼x∼ψω(⋅∣m,s)​[R​(s,m,x)]=1−1G​∑i=1G𝕀​[x^i,j=xi,j],r^{j}\_{\mathrm{gen}}=1-\mathbb{E}\_{x\sim\psi\_{\omega}(\cdot\mid m,s)}[R(s,m,x)]=1-\frac{1}{G}\sum\_{i=1}^{G}\mathbb{I}[\hat{x}^{i,j}=x^{i,j}], |  | (9) |

which aligns with the outer minimization 𝔼s∼𝒟,m∼πω′(⋅∣s)​[𝔼x^∼ψω(⋅∣m,s)​[R​(s,m,x^)]]\mathbb{E}\_{s\sim\mathcal{D},\;m\sim\pi\_{\omega^{\prime}}(\cdot\mid s)}\bigg[\,\mathbb{E}\_{\hat{x}\sim\psi\_{\omega}(\cdot\mid m,s)}\big[\,R(s,m,\hat{x})\,\big]\bigg] in Eq. ([6](#S3.E6 "In 3.3 Reinforcement Active Learning ‣ 3 Reinforcement Active Pretraining ‣ PretrainZero: Reinforcement Active Pretraining")). The mask generator is rewarded when its masks lead to lower prediction accuracy, indicating that the induced masks contain higher information content for the model.
In addition, when the mask prediction accuracy is zero, we further define the generator’s reward to be rgeni=0r^{i}\_{\mathrm{gen}}=0, in order to avoid rewarding the noisy masks that are not predictable for ψω(⋅∣m,s)\psi\_{\omega}(\cdot\mid m,s).

Given the reword definations, rpredir^{i}\_{\mathrm{pred}} and rgenjr^{j}\_{\mathrm{gen}}, we optimize Eq. ([7](#S3.E7 "In 3.3 Reinforcement Active Learning ‣ 3 Reinforcement Active Pretraining ‣ PretrainZero: Reinforcement Active Pretraining")) using GRPO.
By substituting rgenjr^{j}\_{\mathrm{gen}} into the generator’s advantage, we obtain
Agenj=−𝔼​[Apred:,j]A\_{\mathrm{gen}}^{j}=-\,\mathbb{E}\!\left[A\_{\mathrm{pred}}^{:,j}\right],
which proves that the GRPO update is fully consistent with the min–max objective in Eq. ([7](#S3.E7 "In 3.3 Reinforcement Active Learning ‣ 3 Reinforcement Active Pretraining ‣ PretrainZero: Reinforcement Active Pretraining")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | A^genj=rgenj−mean⁡(r1,…,rG)std⁡(r1,…,rG)=−𝔼​[A^pred:,j].\hat{A}\_{\mathrm{gen}}^{j}=\frac{r^{j}\_{\mathrm{gen}}-\operatorname{mean}(r^{1},\dots,r^{G})}{\operatorname{std}(r^{1},\dots,r^{G})}=-\mathbb{E}[\hat{A}\_{\mathrm{pred}}^{:,j}]. |  | (10) |

We directly concatenate and uniformly optimize the mask generation and prediction batches in each step:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒGRPO​(ω)=−1G​∑i=1Gmin⁡(πω​(xi)πωold​(xi)​A^i,clip⁡(πω​(xi)πωold​(xi), 1−ϵ, 1+ϵ)​A^i).\mathcal{L}\_{\text{GRPO}}(\omega)=-\frac{1}{G}\sum\_{i=1}^{G}\min\!\Bigl(\tfrac{\pi\_{\omega}(x\_{i})}{\pi\_{\omega\_{\text{old}}}(x\_{i})}\,\hat{A}\_{i},\;\operatorname{clip}\!\bigl(\tfrac{\pi\_{\omega}(x\_{i})}{\pi\_{\omega\_{\text{old}}}(x\_{i})},\,1-\epsilon,\,1+\epsilon\bigr)\,\hat{A}\_{i}\Bigr). |  | (11) |

## 4 Experimental Results

### 4.1 Implementation Details

Model. To evaluate stand-alone reinforcement pretraining, we directly continue pretraining the base models using reinforcement learning without introducing any intermediate supervised finetuning (SFT) cold start. Specifically, we pretrain base models in 3 ∼\sim 30 billion parameters, including the Qwen3-4B-Base, Qwen3-8B-Base, Qwen3-30B-A3B-MoE-Base, and SmolLM3-3B-Base.

Dataset. To evaluate on real-world distributed pretraining corpus, we use only the most general Wikipedia dataset.
Notice that existing RLPT often includes explicit Question-Answer pairs or synthetic datasets such as OmniMath that contain strong reasoning CoTs; this risks allowing the RL objective to copy these reasoning traces directly, implicitly degrading to supervised learning.

Training. For RLPT, we train 2000 steps using GRPO without KL regularization [ppo]. Following DAPO [dapo], we filter samples whose accuracies are exactly 0.0 or 1.0, and we adopt the clip-higher strategy for stability. For Qwen-Base models, we directly perform the PretrainZero strategy; for SmolLM3-3B-Base, we first use random RLPT for 100 steps as RL cold-start, and then perform PretrainZero for the remaining 1900 steps.
During reasoning, the max length of the prompt and response is limited to 1536 and 4096 tokens, respectively. We adopt the 5×10−75\times 10^{-7} learning rate and the cosine scheme.
In the mask-generation task, each batch contains 32 pretraining paragraphs, with 8 rollouts for each to produce masks. In the mask-prediction task, we evaluate 256 masks from the prefixed mask generation task (32×\times8), and each mask is also paired with 8 rollouts for prediction. Consequently, the overall prompt batch size becomes 288 for RL (32 + 32×\times8).

Evaluation. We evaluate on both general-domin and math-domin reasoning benchmarks. For general domin reasoning, we evaluate on the MMLU-Pro [mmlupro], SuperGPQA [supergpqa], and BBEH [kazemi2025big]; for math domin reasoning, we evaluate on 6 widely used benchmarks, including Math 500 [math500], Olympiad [he2024olympiadbench], Minerva [minerva], GSM8K [gsm8k], AMC23, and AIME24. For AMC23 and AIME24, we evaluate 32 times and report the mean@32 accuracy. We use the Qwen-Math-eval [yang2024qwen2] as the math verifier.

### 4.2 Pretraining Results

!(/html/2512.03442/assets/fig6.png)

  

Figure 6: 
Training dynamic comparisons between PretrainZero and Random RLPT on Qwen3-4B-Base: (a) entropy of model outputs; (b) response length of overall samples; (c) the overall reward.

!(/html/2512.03442/assets/fig7.png)

Figure 7: 
Evaluation comparisons between PretrainZero and Random RLPT on Qwen3-4B-Base: (a) the average accuracy on 3 general reasoning benchmarks; (b) the average accuracy on 6 math reasoning benchmarks; (c) response length on a fixed subset from MMLU-Pro.

Table 1: Results on general-domain reasoning benchmarks. We compare the Base Model, Continue Pre-Training, Supervised Fine-Tuning, our Random RLPT baseline and PretrainZero. We highlight the best performance in bold and the second performance in underline.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Model Name | Overall AVG | MATH AVG | SuperGPQA | BBEH | MMLU-Pro |
| Qwen3-4B-Base | | | | | |
| Base Model | 32.36 | 42.53 | 26.32 | 8.67 | 51.94 |
| Continue PT | 15.89 | 24.65 | 9.67 | 0.04 | 29.21 |
| Supervised FT | 24.27 | 15.55 | 26.38 | 12.28 | 42.88 |
| Random RLPT | 35.41 | 47.87 | 29.10 | 9.45 | 55.21 |
| PretrainZero | 39.61 | 53.13 | 32.28 | 12.68 | 60.37 |
| Qwen3-8B-Base | | | | | |
| Base Model | 37.07 | 47.48 | 31.12 | 10.49 | 59.19 |
| Continue PT | 12.32 | 27.78 | 9.94 | 0.04 | 11.51 |
| Supervised FT | 26.80 | 19.23 | 29.02 | 11.17 | 47.78 |
| Random RLPT | 40.96 | 55.08 | 34.19 | 12.96 | 61.59 |
| PretrainZero | 42.78 | 57.72 | 34.46 | 14.67 | 64.28 |
| SmolLM3-3B-Base | | | | | |
| Base Model | 16.23 | 32.31 | 12.62 | 3.32 | 16.66 |
| Random RLPT | 20.25 | 35.95 | 14.48 | 7.85 | 22.74 |
| PretrainZero | 23.41 | 38.03 | 19.44 | 3.78 | 32.41 |
| Qwen3-30B-A3B-MoE-Base | | | | | |
| Base Model | 38.88 | 52.49 | 33.73 | 10.51 | 58.79 |
| Random RLPT | 40.38 | 52.62 | 36.33 | 12.99 | 59.57 |
| PretrainZero | 43.55 | 58.12 | 36.58 | 14.91 | 64.59 |

Table 2: Results on math-domain reasoning benchmarks. We highlight the best performance in bold and the second performance in underline.

|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model Name | AVG | MATH-500 | Olympiad | Minerva | GSM8K | AMC | AIME24 |
| Qwen3-4B-Base | | | | | | | |
| Base Model | 42.53 | 73.30 | 37.30 | 22.10 | 86.30 | 36.17 | 0.00 |
| Continue PT | 24.65 | 38.00 | 13.60 | 11.00 | 67.00 | 15.00 | 3.30 |
| Supervised FT | 15.55 | 28.50 | 8.10 | 14.30 | 27.40 | 15.00 | 0.00 |
| Random RLPT | 47.87 | 74.80 | 38.50 | 22.10 | 87.50 | 54.30 | 10.00 |
| PretrainZero | 53.13 | 79.10 | 42.70 | 33.80 | 92.90 | 56.95 | 13.30 |
| Qwen3-8B-Base | | | | | | | |
| Base Model | 47.48 | 70.10 | 35.30 | 25.40 | 91.50 | 52.58 | 10.00 |
| Continue PT | 27.78 | 42.70 | 16.90 | 11.80 | 55.30 | 33.28 | 6.70 |
| Supervised FT | 19.23 | 30.50 | 11.70 | 15.40 | 32.80 | 25.00 | 0.00 |
| Random RLPT | 55.08 | 79.20 | 42.70 | 39.00 | 93.80 | 62.50 | 13.30 |
| PretrainZero | 57.72 | 81.90 | 42.50 | 43.40 | 93.50 | 65.00 | 20.00 |
| SmolLM3-3B-Base | | | | | | | |
| Base Model | 32.31 | 53.80 | 20.40 | 14.00 | 81.20 | 22.81 | 1.65 |
| Random RLPT | 35.95 | 59.00 | 21.50 | 20.20 | 82.50 | 32.50 | 0.00 |
| PretrainZero | 38.03 | 62.60 | 25.60 | 22.10 | 83.70 | 27.50 | 6.70 |
| Qwen3-30B-A3B-MoE-Base | | | | | | | |
| Base Model | 52.49 | 74.70 | 43.00 | 22.80 | 91.10 | 66.95 | 16.36 |
| Random RLPT | 52.62 | 79.20 | 41.20 | 38.60 | 82.40 | 59.77 | 14.58 |
| PretrainZero | 58.12 | 81.70 | 43.40 | 41.20 | 94.40 | 70.62 | 17.40 |

Baselines. To compare Reinforcement Learning Pre-Training (RLPT) with conventional training patterns, we primarily establish the following baselines: 1) the base model as the initial baseline. 2) Continue Pre-Training: We continuously pretrain with the self-supervised next token prediction on the same Wikipedia data. 3) Supervised Fine-Tuning: we formulate the masked token prediction task as question-answer pairs as Fig. [5](#S3.F5 "Figure 5 ‣ 3.2 Active Pretraining Tasks ‣ 3 Reinforcement Active Pretraining ‣ PretrainZero: Reinforcement Active Pretraining"), and remove the CoT. 4) Random RLPT: We use the introduced random masked span prediction introduced in Sec. 3.1 as the strong RL baseline.

Comparison with Supervised Pretraining. We summarize the overall and detailed math performance in Table [1](#S4.T1 "Table 1 ‣ 4.2 Pretraining Results ‣ 4 Experimental Results ‣ PretrainZero: Reinforcement Active Pretraining") and [2](#S4.T2 "Table 2 ‣ 4.2 Pretraining Results ‣ 4 Experimental Results ‣ PretrainZero: Reinforcement Active Pretraining") respectively. Compared with the base model, Continued Pre-Training and Supervised Fine-Tuning lead to performance drops of 16.47 and 8.09 on Qwen3-4B, and 24.75 and 10.27 on Qwen3-8B, respectively. This occurs because, for highly optimized models, supervised learning on low-quality Wikipedia passages offers limited meaningful information and can even disrupt the model’s pretrained distribution.
For RLPT, Random RLPT and PretrainZero improve 3.05 and 7.25 on Qwen3-4B, and 3.89 and 5.71 on Qwen3-8B. These results suggest that reinforcement learning is capable of extracting meaningful supervision from relatively low-quality data.

Reasoning Patterns: Random RLPT / PretrainZero

Paragraph: In addition to his 1983 Triple Crown wins, Ralph Hanover won seventeen additional [mask] events, including the very important Adios and Meadowlands Pace …

Random RLPT: Here is the passage with the masked words filled in:
In addition to his 1983 Triple Crown wins, Ralph Hanover won seventeen additional stakes …
`\boxed{stakes}` …

PretrainZero: To determine the masked word in the sentence:
"In addition to his 1983 Triple Crown wins, Ralph Hanover won seventeen …"
1. \*\*Analyze the sentence structure\*\*: The sentence is: "Ralph Hanover won seventeen …"
2. \*\*Identify the missing word\*\*: The sentence structure suggests that the missing word should complete the phrase "seventeen additional [mask] events." Common phrases include "seventeen additional stakes events" or "seventeen additional championship events."
3. \*\*Consider the context\*\*: The sentence mentions specific races like … These are all racing events, particularly in harness racing. The mention of world records and specific track distances further indicates that the missing word is related to racing events.
4. \*\*Determine the most appropriate word\*\*: The most common completion … in this context is … However, the most relevant term in the context of racing achievements is "seventeen additional stakes events."
5. \*\*Conclusion\*\*: The missing word is "stakes."
Therefore, the predicted masked word is:
In addition to his 1983 Triple Crown wins, Ralph Hanover won seventeen additional `\boxed{stakes}` events …

Figure 8: Reasoning pattern comparison between Random RLPT and PretrainZero.

Comparison with Reinforcement Pretraining. As mentioned in Sec. 3.1, previous RPT training on high-entropy tokens quickly stops optimization when applied to real-world pretraining corpus. We compare the training dynamics between Random RLPT and PretrainZero in Fig. [6](#S4.F6 "Figure 6 ‣ 4.2 Pretraining Results ‣ 4 Experimental Results ‣ PretrainZero: Reinforcement Active Pretraining") and [7](#S4.F7 "Figure 7 ‣ 4.2 Pretraining Results ‣ 4 Experimental Results ‣ PretrainZero: Reinforcement Active Pretraining").
As training steps increase, we observe that both the reasoning length of PretrainZero and its performance on general- and math-reasoning benchmarks improve consistently. This indicates that PretrainZero’s reasoning ability is gradually strengthened, similar to RLVR in DeepSeek-R1.
Compared with Random RLPT, the active-learning strategy arouses longer CoT trajectories and noticeably stronger reasoning performance. As shown in Table [1](#S4.T1 "Table 1 ‣ 4.2 Pretraining Results ‣ 4 Experimental Results ‣ PretrainZero: Reinforcement Active Pretraining"), PretrainZero consistently outperforms Random RLPT by 4.20, 1.82, 3.17, and 3.16 points on the Qwen3-4B, Qwen3-8B, Qwen3-30B-A3B-MoE, and SmolLM3-3B base models, respectively.

Reasoning Efficiency.
As shown in Fig. [6](#S4.F6 "Figure 6 ‣ 4.2 Pretraining Results ‣ 4 Experimental Results ‣ PretrainZero: Reinforcement Active Pretraining") (b), although the growth of CoT length in training, we need not worry about the inference efficiency of the reasoning process. The growth mainly comes from improvements in the mask-prediction capability. To verify this, we sample 10% of the MMLU-Pro prompts and evaluate the reasoning length. As shown in Fig. [7](#S4.F7 "Figure 7 ‣ 4.2 Pretraining Results ‣ 4 Experimental Results ‣ PretrainZero: Reinforcement Active Pretraining") (c), for the same questions, the reasoning length remains similar during RLPT. Moreover, compared with the base model, RLPT actually improves the efficiency of CoT reasoning.

Reasoning Pattern.
As shown in Fig. [8](#S4.F8 "Figure 8 ‣ 4.2 Pretraining Results ‣ 4 Experimental Results ‣ PretrainZero: Reinforcement Active Pretraining"), we compare the reasoning patterns between Random RLPT and PretrainZero from the Qwen3-8B-Base. Given the same masked target, Random RLPT directly outputs the answer without any explicit reasoning. In contrast, PretrainZero first explores multiple possibilities, analyzes and verifies them step by step, and finally summarizes to reach a conclusion.
Since the mask-prediction objective does not appear in downstream RL tasks, the emergence of such reasoning behavior during pretraining provides a stronger reasoning ability for generalization.

!(/html/2512.03442/assets/fig9.png)

  

Figure 9: 
RLPT performance after the same RLVR post-training.
(a) Comparison of Qwen3-4B-base, Random RLPT, and PretrainZero.
(b) Comparison of Qwen3-4B-base, PretrainZero with 1000 and 2000 steps RLPT.
(c) Response length comparison in the same MMLU-Pro subset.

### 4.3 Post-Training Results

To investigate whether PretrainZero can improve the general reasoning capabilities of the foundation model for efficient RL finetuning, we apply RLVR as a post-training stage on PretrainZero.
For the general RLVR task, we follow the General Reasoner recipe [ma2025general]. Specifically, we apply the Web-Instruct dataset [ma2025general] in a Question–Answer format, and the same pretrained reward model as the verifier.
For efficient RL finetuning, we train 400 steps on Qwen3-4B series models with a single node with 8×8\times H800 GPUs, which supports at most the 128 batchsize, 1/8 compared with General Reasoner.

We evaluate the training process in Fig. [9](#S4.F9 "Figure 9 ‣ 4.2 Pretraining Results ‣ 4 Experimental Results ‣ PretrainZero: Reinforcement Active Pretraining"), and report the final general-domain and math-domain performance in Table [3](#S4.T3 "Table 3 ‣ 4.3 Post-Training Results ‣ 4 Experimental Results ‣ PretrainZero: Reinforcement Active Pretraining") and Table [4](#S4.T4 "Table 4 ‣ 4.3 Post-Training Results ‣ 4 Experimental Results ‣ PretrainZero: Reinforcement Active Pretraining"), respectively.
As shown in Fig. [9](#S4.F9 "Figure 9 ‣ 4.2 Pretraining Results ‣ 4 Experimental Results ‣ PretrainZero: Reinforcement Active Pretraining") (b), the performance consistently improves as the training starts progressing from the base model to PretrainZero at 1000 RLPT steps and further to PretrainZero at 2000 RLPT steps on MMLU-Pro.
As shown in Fig. [9](#S4.F9 "Figure 9 ‣ 4.2 Pretraining Results ‣ 4 Experimental Results ‣ PretrainZero: Reinforcement Active Pretraining") (c), PretrainZero has more stable and efficient CoT in downstream RLVR.
Compared with the base model, PretrainZero significantly improves the math average and overall accuracy by 2.18 and 2.56 points end-to-end.

Table 3: Results on general-domain reasoning benchmarks after the RLVR post-training. We perform the general RLVR post-training [ma2025general] from the Qwen3-4B-Base model, Random RLPT, and PretrainZero with 1000 / 2000 step RLPT. RLPT / RLVR indicates RL steps in RLPT and RLVR stages respectively. We highlight the best performance in bold.

| Model Name | RLPT / RLVR | Overall AVG | MATH AVG | SuperGPQA | BBEH | MMLU-Pro |
| --- | --- | --- | --- | --- | --- | --- |
| Base Model | – / 400 | 37.90 | 50.96 | 30.26 | 11.59 | 58.80 |
| Random RLPT | 2000 / 400 | 38.43 | 51.49 | 30.77 | 12.83 | 58.62 |
| PretrainZero | 1000 / 400 | 39.15 | 51.84 | 32.32 | 12.39 | 60.03 |
| PretrainZero | 2000 / 400 | 40.46 | 53.77 | 33.30 | 13.61 | 61.15 |

Table 4: Results on math-domain reasoning benchmarks after the RLVR post-training. RLPT / RLVR indicates RL steps in RLPT and RLVR stages respectively.

| Model Name | RLPT / RLVR | AVG | MATH-500 | Olympiad | Minerva | GSM8K | AMC | AIME24 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base Model | – / 400 | 50.96 | 75.70 | 41.80 | 31.60 | 91.30 | 52.03 | 13.30 |
| Random RLPT | 2000 / 400 | 47.87 | 74.80 | 38.50 | 22.10 | 87.50 | 54.30 | 10.00 |
| PretrainZero | 1000 / 400 | 51.84 | 77.00 | 43.00 | 32.40 | 92.50 | 55.00 | 11.13 |
| PretrainZero | 2000 / 400 | 53.77 | 78.80 | 43.00 | 39.70 | 93.00 | 54.84 | 13.30 |

### 4.4 Ablation Studies

Specific Domain.
To explore the impact of data domain on RLPT, we compare RLPT performance on the Wikipedia corpus in the general-domain versus the MathPile [wang2024mathpile] dataset in the math-domain.
As shown in Fig. [10](#S4.F10 "Figure 10 ‣ 4.4 Ablation Studies ‣ 4 Experimental Results ‣ PretrainZero: Reinforcement Active Pretraining") (a), directly using general-domain Wikipedia data yields better performance. Since curating high-quality mathematical data requires substantial expert effort, we recommend using general-domain data for a much lower cost of data acquisition.

Training Robustness.
To confirm RLPT over 2000 steps, we evaluate different mask regularization strategies:
1) PretrainZero: For the generated masks, we retain only those whose underlying spans appear fewer than eight times within the paragraph.
2) PretrainZero-OneMask: Based on PretrainZero, if a generated mask appears multiple times in the paragraph, we randomly replace only one occurrence with [mask] and make prediction.
3) PretrainZero-Words: Since PretrainZero may produce masks that cover incomplete words—reducing interpretability—we filter masks that keep only complete word spans.
As shown in Fig. [10](#S4.F10 "Figure 10 ‣ 4.4 Ablation Studies ‣ 4 Experimental Results ‣ PretrainZero: Reinforcement Active Pretraining") (b), three recipes can be trained stably, and PretrainZero consistently achieves better performance.

!(/html/2512.03442/assets/fig10.png)

  

Figure 10: 
Comparisons for data domain and mask regularization. (a) MMLU-Pro performance on MathPile and Wikipedia. (b) MMLU-Pro performance with different mask regularization strategies.

## 5 Related Works and Discussion

Self-Supervised Pretraining for LLMs.
Scalable self-supervised pretraining [scaling:law] formulates the foundation of advanced large language models. Under the simple and fixed learning pattern, the next token prediction, autoregressive LLMs [transformer, team2025kimi] can be trained on massive corpus at the Internet-scale, establishing strong general-purpose capabilities. Beyond this pattern, token-masked prediction objectives [bert, spanbert] continue to play an important role in the pretraining of language models, such as in BERT-style embedding models [chen2024bge], diffusion language models [nie2025large], and code-focused pretraining [hui2024qwen2]. The reliability, scalability, and broad applicability of self-supervised learning offer key insights for reinforcement pretraining and highlight its potential as a fundamental training strategy.

Reinforcement Learning for LLMs.
Recent large reasoning models are largely driven by post-training reinforcement learning, enabling human-expert performance in specialized domains such as web agents [team2025tongyi], tool use [patilberkeley], software development [jimenez2023swe], and mathematics [deepseekr1]. Despite this progress, existing RLHF [instructgpt, bai2022training] and RLVR [tulu3] approaches rely heavily on human annotation and domain-specific verification environments, leading to a severe data bottleneck in general domains [ma2025general, zhou2025reinforcing]. For RLHF, reward models must be continuously updated with human-labeled data to avoid reward hacking. For RLVR, training data must come from domains with verifiable ground-truth answers, and the construction of verifiable environments fundamentally limits its scalability for general reasoning tasks [ma2025general].

Reinforcement Pretraining.
To overcome the substantial verification data-wall, Reinforcement Learning Pre-Training (RLPT) has recently emerged as a promising direction, which constructs general-purpose RLVR directly on pretraining corpus using self-supervised objectives.
Early works including Quiet-STaR [zelikman2024quiet] and Fast Quiet-STaR [huang2025fast] focus on token-level reasoning.
Reinforcement Pre-Training (RPT) [dong2025reinforcement] is the first to apply the next-token–prediction as the RLVR objective, demonstrating the feasibility of general-purpose RL. However, RPT remains limitations, such as relying on synthetic OmniMath data with CoT annotations rather than real pretraining distributions, and training from a post-trained model instead of a base model, which prevents RPT from being practical and prolonged [liu2025prorl] RLPT.

Recently, PRT [hatamizadeh2025rlp] and RLPT1 [li2025reinforcement] are proposed around the similar period as this work. PRT incorpustes reinforcement learning as an auxiliary objective to the self-supervised pretraining and does not exclude some QA-style training data. RLPT1 employs an additional reward model as a verifier for the sentence-level prediction objective and further introduces a high-quality SFT cold-start.
Despite these advantages, the foundational questions in RLPT remain unexplored: under fully self-supervised conditions—removing reward models, SFT cold start, and supervised cross-entropy losses—can stand-alone RLPT be effectively trained on noisy, real-world pretraining corpus? And how to improve learning efficiency in low-information-density pretraining data? Addressing these fundamental questions becomes the primary focus of this work.

## 6 Conclusion

This work introduces the stand-alone reinforcement pretraining method in a real-world pretraining corpus, named PretrainZero. Coupled with PretrainZero, a new reinforcement active pretraining framework is proposed to explore informative, verifiable, and not-yet-mastered content in noisy pretraining data. Thanks to active learning ability, PretrainZero significantly surpasses previous fixed learning patterns, such as continued pretraining, supervised fine-tuning, and random or entropy-based reinforcement pretraining. We reveal that even Wikipedia, which has already been trained during base model pretraining, can successfully improve end-task performance with reinforcement and active learning methods. We believe that there would be great potential to explore more efficient learning patterns to discover latent information from the pretraining corpus in the future.
