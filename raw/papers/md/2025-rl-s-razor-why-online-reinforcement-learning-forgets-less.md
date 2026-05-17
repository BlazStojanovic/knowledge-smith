---
arxiv: '2509.04259'
authors:
- Idan Shenfeld
- Jyothish Pari
- Pulkit Agrawal
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'RL''s Razor: Why Online Reinforcement Learning Forgets Less'
url: https://arxiv.org/abs/2509.04259
year: 2025
---

# RL’s Razor: Why Online Reinforcement Learning Forgets Less

Idan Shenfeld∗  Jyothish Pari∗  Pulkit Agrawal
  
 Improbable AI Lab, MIT
  
{idanshen, jyop, pulkitag}@mit.edu

###### Abstract

Comparison of fine-tuning models with reinforcement learning (RL) and supervised fine-tuning (SFT) reveals that, despite similar performance at a new task, RL preserves prior knowledge and capabilities significantly better. We find that the degree of forgetting is determined by the distributional shift, measured as the KL-divergence between the fine-tuned and base policy evaluated on the new task. Our analysis reveals that on-policy RL is implicitly biased towards KL-minimal solutions among the many that solve the new task, whereas SFT can converge to distributions arbitrarily far from the base model. We validate these findings through experiments with large language models and robotic foundation models and further provide theoretical justification for why on-policy RL updates lead to a smaller KL change. We term this principle RL’s Razor: among all ways to solve a new task, RL prefers those closest in KL to the original model. Our website is available at <http://jyopari.github.io/posts/rl_razor>.

!(/html/2509.04259/assets/x1.png)

Figure 1: Bias toward KL-minimal solutions reduces forgetting. *Left:* Among policies that solve the new task, RL converges to those closest in KL to the base model. *Right:* This KL bias yields higher prior-task retention at matched new-task performance compared to SFT.

## 1 Introduction

Foundation models have rapidly become the backbone of modern AI, powering applications in language, vision, robotics, and beyond. Despite their remarkable capabilities, today’s models are largely *static* once deployed: they excel at tasks learned during pre-training or post-training, but are not designed to self-improve and continually acquire new capabilities. We imagine a future where deployed models are long-lived *agents* assisting humans in the long-term and continuously adapting to new needs. As such, models must improve and adapt to new data, environments, and objectives Gao et al. ([2025](#bib.bib21)); Dao & Le ([2025](#bib.bib14)); Moradi et al. ([2025](#bib.bib50)); Li et al. ([2025b](#bib.bib44)); Simonds & Yoshiyama ([2025](#bib.bib63)); Zweiger et al. ([2025](#bib.bib78)).

A central challenge to this vision is *catastrophic forgetting*—the tendency for models to lose previously acquired capabilities when trained on new tasks McCloskey & Cohen ([1989](#bib.bib48)); French ([1999](#bib.bib20)); Kirkpatrick et al. ([2017](#bib.bib37)); Luo et al. ([2023](#bib.bib47)).
Although scaling model size and pre-training data improves robustness Ramasesh et al. ([2021](#bib.bib57)); Luo et al. ([2023](#bib.bib47)); Cossu et al. ([2024](#bib.bib12)), catastrophic forgetting remains a persistent obstacle, undermining the promise of continual improvement Bommasani ([2021](#bib.bib4)); Guo et al. ([2025b](#bib.bib26)); Zweiger et al. ([2025](#bib.bib78)). To enable foundation models to serve as long-term agents, we need to develop post-training methods that allow models to acquire new skills without erasing old ones.

To further this goal, we analyze the performance of two widely used post-training schemes of supervised fine-tuning (SFT) and reinforcement learning (RL). Our experiments reveal a surprising finding: even when SFT and RL achieve the same performance on the new task, we observe that SFT often achieves new-task gains by erasing prior knowledge, while RL better preserves old skills. Figure [1](#S0.F1 "Figure 1 ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less") (right) illustrates this tradeoff: although both methods can reach high performance on the new task, RL maintains substantially higher performance on prior tasks compared to SFT.

This striking empirical gap raises the question: what underlying mechanism allows RL to improve on new tasks, but unlike SFT, minimally impacts the model’s prior knowledge?

Previous approaches to catastrophic forgetting targeted specific factors such as constraining weight updates (Kirkpatrick et al., [2017](#bib.bib37); Aljundi et al., [2018](#bib.bib2); Zenke et al., [2017](#bib.bib74)), preserving learned features (Rannen et al., [2017](#bib.bib58); Hou et al., [2019](#bib.bib29)), or regularizing shift in output distribution (Li & Hoiem, [2017](#bib.bib43); Stiennon et al., [2020](#bib.bib64)). While these methods can reduce forgetting, they focus on its effects rather than its underlying cause. Consequently, it remains unclear what truly governs forgetting or why different training algorithms behave so differently.
Some prior work claimed that forgetting can be determined by how much the model’s distribution shifts on past tasks (Rebuffi et al., [2017](#bib.bib59); Castro et al., [2018](#bib.bib7); Chaudhry et al., [2018](#bib.bib8); Wu et al., [2019](#bib.bib71)). Yet in practice, this is infeasible to measure in foundation models, where the set of prior tasks is vast or even unbounded.
To search for a more useful principle, we systematically ablated many candidate variables. Surprisingly, we find that forgetting can instead be predicted using only the *new* task distribution. Specifically, we uncover an empirical forgetting law: When fine-tuning a model π\bm{\pi} on a new task τ\bm{\tau}, the degree of forgetting is accurately predicted by 𝔼x∼τ[KL(π𝟎||π)]\bm{\mathbb{E}\_{x\sim\tau}\big[\text{KL}(\pi\_{0}||\pi)\big]}, the KL divergence between the fine-tuned and base policy evaluated on the new task.
This law is practically useful since it can be measured, and even influenced, during fine-tuning, without requiring access to past-task data.
Although the mechanism remains to be fully understood, the consistency of this law across models and domains suggests it reflects a fundamental property of forgetting.

This law also clarifies the surprising difference between SFT and RL. Our analysis reveals a simple but powerful principle we call *RL’s Razor*: among the many high-reward solutions for a new task, on-policy methods such as RL are inherently biased toward solutions that remain closer to the original policy in KL divergence. Figure [1](#S0.F1 "Figure 1 ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less") (left) highlights this effect: among the many policies that reach a high success rate on the new task, RL is biased toward KL-minimal solutions, while SFT can converge to distant ones.
This bias arises directly from RL’s *on-policy training*: by sampling from the model’s own distribution at every step, RL constrains learning to outputs already given non-negligible probability by the base model. To improve reward, these samples are reweighted and used to update the model, which gradually shifts the policy rather than pulling it toward an arbitrary distribution. Thus, when multiple equally good solutions exist for a new task, RL tends to find solutions close to the original policy, while SFT can converge to solutions much farther away, depending on the provided labels. Theoretical analysis in a simplified setting confirms this view, showing that policy gradient methods converge to KL-minimal solutions even without explicit regularization.

Finally, to validate the KL hypothesis, we construct an “oracle SFT” distribution that provably minimizes KL divergence while achieving perfect accuracy. Training on this oracle distribution produces even less forgetting than RL itself. This demonstrates that RL’s advantage does not stem from being inherently different, but from its implicit KL minimization. Whenever training is biased toward KL-minimal solutions, forgetting is reduced.

Our main contributions are:

* •

  We show that RL fine-tuning forgets less than SFT, even when both reach the same performance on new tasks.
* •

  We uncover an empirical forgetting law: the KL divergence to the base policy, measured on the new task, as a strong predictor of catastrophic forgetting across objectives and hyperparameters.
* •

  We provide empirical and theoretical evidence that the on-policy nature of policy gradient methods leads to smaller KL shifts and explains RL’s advantage.

Together, these findings suggest a new perspective on post-training: to achieve continual adaptation without forgetting, algorithms should explicitly aim to minimize KL divergence from the base model. This principle opens the door to designing future training methods that combine RL’s ability to preserve prior knowledge with the efficiency of SFT, enabling foundation models that can truly *learn for life*.

## 2 Related work

#### Foundation Models and Post-training

In modern deep learning, large-scale models pre-trained on broad, diverse datasets (usually termed Foundation models) serve as general-purpose backbones (Radford et al., [2021](#bib.bib55); Achiam et al., [2023](#bib.bib1); Touvron et al., [2023](#bib.bib67); Hu et al., [2023](#bib.bib32); Li et al., [2024a](#bib.bib40)) with broad domain knowledge and some zero-shot learning abilities (Radford et al., [2018](#bib.bib54); Brown et al., [2020](#bib.bib6)).
However, pre-trained models may not directly meet the requirements of specific applications or align with domain-specific constraints. Post-training methods address this gap by adapting foundation models to downstream tasks through supervised fine-tuning on curated datasets (Howard & Ruder, [2018](#bib.bib30); Dodge et al., [2020](#bib.bib18); Wei et al., [2021](#bib.bib69); Chung et al., [2024](#bib.bib11)), reinforcement learning from human or automated feedback (Ziegler et al., [2019](#bib.bib77); Ouyang et al., [2022](#bib.bib52); Guo et al., [2025a](#bib.bib25); Zhai et al., [2024](#bib.bib75)), and other techniques (Rafailov et al., [2023](#bib.bib56)). In this work, we study how different post-training methods affect forgetting, focusing on supervised fine-tuning and reinforcement learning.

#### Catastrophic Forgetting.

While fine-tuning primarily aims to improve performance on a new specific task, preserving the model’s pre-existing general capabilities is equally critical. Unfortunately, fine-tuning often leads to catastrophic forgetting—a phenomenon where learning new information significantly deteriorates previously acquired knowledge McCloskey & Cohen ([1989](#bib.bib48)); French ([1999](#bib.bib20)); Kirkpatrick et al. ([2017](#bib.bib37)); Ouyang et al. ([2022](#bib.bib52)); Luo et al. ([2023](#bib.bib47)). Many works have sought to reduce forgetting by constraining updates, for example, by penalizing the magnitude of change in the model parameters, features, or matching the output on previous tasks/datasets (Wang et al., [2024](#bib.bib68)). These methods are effective heuristics, but they address the symptoms of forgetting rather than explaining its cause. Our aim is to identify a simple and predictive metric that explains when and why forgetting occurs across different training algorithms.

We do not introduce a new training algorithm, but instead identify a simple *empirical forgetting law*: the KL divergence between the fine-tuned and base policy, measured *on the new task*, reliably predicts the degree of forgetting. The law also sheds light on why some mitigation strategies work. For example, methods like Elastic Weight Consolidation (Kirkpatrick et al., [2017](#bib.bib37)) can be seen as approximations to KL minimization (Chaudhry et al., [2018](#bib.bib8)). Interestingly, practitioners have also observed that KL regularization used in RL fine-tuning of LLMs as a heuristic for stabilizing optimization or preventing reward hacking Stiennon et al. ([2020](#bib.bib64)); Gao et al. ([2023](#bib.bib22)), also helps reduce catastrophic forgetting (Ouyang et al., [2022](#bib.bib52)). Our contribution is to show that KL divergence is not merely a useful heuristic, but a reliable predictor of forgetting across settings.

#### SFT versus RL.

Prior comparisons between SFT and RL have focused on new task performance. A seminal result in sequential decision making is that on-policy learning can achieve stronger performance even when the expert providing supervision is the same one used to generate the offline dataset (Ross et al., [2011](#bib.bib60)). Recent empirical studies have also found that RL fine-tuned models often exhibit superior generalization beyond the training distribution Han et al. ([2025](#bib.bib27)); Chu et al. ([2025](#bib.bib10)); Li et al. ([2025a](#bib.bib41)) and transfer more effectively to related tasks Huan et al. ([2025](#bib.bib33)) compared to SFT. However, prior works haven’t examined the relative susceptibility of RL and SFT to catastrophic forgetting, which is the focus of our study.

Concurrently, Lai et al. ([2025](#bib.bib39)) reports that RL forgets less than SFT, but ascribes RL’s advantage to learning from negative examples and not to the on-policy nature of RL. Results in Section [5](#S5 "5 On-policy methods leads to smaller KL divergence ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less") contradict their explanation of why RL forgets less, showing that the on-policy nature of RL is key. We also contribute the empirical forgetting law, the RL Razor, and its theoretical justification.

## 3 Reinforcement Learning Forgets Less than SFT

!(/html/2509.04259/assets/x2.png)

Figure 2: Pareto frontiers of RL and SFT.
Comparing the performance of a fine-tuned model on the new task (x-axis) and prior task (y-axis). Each point corresponds to a model trained with a different set of hyperparameters, and the curves trace the Pareto frontiers for the two methods. RL achieves new-task improvements while maintaining prior knowledge, whereas SFT improves new-task performance at the expense of forgetting the prior task.

We report results comparing the degree of catastrophic forgetting against new-task performance induced by RL and SFT on various large language model (LLM) and simulated robotic tasks.

### 3.1 Performance Trade-offs

#### Experimental Setup.

For each new task, we fine-tuned models using the same set of prompts. One group of models was trained with SFT, and another with RL using GRPO Shao et al. ([2024](#bib.bib62)). In RL training, we used only a binary success indicator as the reward, *without explicit KL regularization*. Evaluation was performed along two axes:

* •

  New task Performance: We measured performance on the held-out test set of the newly introduced task to assess the performance gain from the training.
* •

  Previous tasks Performance: We measured performance on a diverse set of unrelated benchmarks. A drop in these benchmarks was taken as a measure of catastrophic forgetting.

Since different hyperparameters can lead to varying trade-offs between learning and forgetting, we trained dozens of models under diverse hyperparameter settings for both SFT and RL. To compare methods fairly, we identify the Pareto frontier in the two-dimensional plane of new-task performance versus previous-task performance. The Pareto frontier represents the set of models for which no further improvement on the new task is possible without incurring greater forgetting. Figure [2](#S3.F2 "Figure 2 ‣ 3 Reinforcement Learning Forgets Less than SFT ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less") (right) reports these frontiers: each point corresponds to a trained model with a different set of hyperparameters, and the Pareto-frontier curve indicates the best achievable trade-off for each method.

#### Tasks and Datasets.

We perform experiments across three LLM and a single robotic tasks:

* •

  LLM, Math reasoning: Qwen 2.5 3B-Instruct (Qwen et al., [2025](#bib.bib53)) trained on math questions from the Open-Reasoner-Zero dataset (Hu et al., [2025](#bib.bib31)).
* •

  LLM, Science Q&A: Qwen 2.5 3B-Instruct trained on Chemistry L-3 subset of SciKnowEval (Feng et al., [2024](#bib.bib19)).
* •

  LLM, Tool use: Qwen 2.5 3B-Instruct trained on ToolAlpaca dataset (Tang et al., [2023](#bib.bib66)).
* •

  Robotics, Pick and Place: OpenVLA 7B (Kim et al., [2024](#bib.bib36)) trained in the SimplerEnv environment (Li et al., [2024b](#bib.bib42)) on the task of picking up a can.

To measure forgetting, we evaluated the finetuned models on established benchmarks covering diverse prior capabilities. For LLMs, we used Hellaswag (Zellers et al., [2019](#bib.bib73)), TruthfulQA (Lin et al., [2021](#bib.bib45)), MMLU (Hendrycks et al., [2020](#bib.bib28)), IFEval (Zhou et al., [2023](#bib.bib76)), Winogrande (Sakaguchi et al., [2021](#bib.bib61)), and HumanEval (Chen et al., [2021](#bib.bib9)). For robotic policies, we evaluated on the open/close drawer SimplerEnv tasks, excluding the one used for fine-tuning. These benchmarks act as proxies for prior skills that should be preserved during adaptation. Full details on SFT data sources, hyperparameters, and training/evaluation protocols are provided in Appendix [B](#A2 "Appendix B Training and Evaluation Details ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less").

#### Results.

Figure [2](#S3.F2 "Figure 2 ‣ 3 Reinforcement Learning Forgets Less than SFT ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less") reports the trade-off between new-task performance and retention of prior abilities. For RL, as accuracy on the new task increases, performance on previous benchmarks remains nearly unchanged. In contrast, SFT improvements on the new task consistently come at the cost of substantial forgetting. This difference is most pronounced in Math, where even small gains on the fine-tuned task correspond to a sharp reduction in prior-task performance. In Science Q&A and Tool Use, SFT retains some ability on prior tasks at lower accuracy levels for the new task, but performance deteriorates rapidly as the model approaches higher accuracy on the new task.

Takeaway 1

RL is able to learn new tasks while incurring minimal forgetting, whereas SFT reaches similar new-task performance only by sacrificing prior knowledge.

## 4 Smaller KL divergences lead to less forgetting

!(/html/2509.04259/assets/x3.png)

Figure 3: KL divergence predicts catastrophic forgetting.
(Left) Learning-Forgetting Trade-offs. SFT outperform RL only when an oracle distribution is used as a source of annotation.
(Middle) Forgetting aligns to a single curve when plotted against KL divergence, showing KL as a strong predictor across methods.
(Right) RL improves new-task accuracy with much smaller KL shifts than SFT, highlighting the conservativeness of on-policy updates.

As shown in Section [3](#S3 "3 Reinforcement Learning Forgets Less than SFT ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less"), RL fine-tuning achieves comparable new-task performance to SFT while consistently forgetting less. Explaining this gap requires identifying a variable that determines the degree of forgetting across methods. We therefore searched for a predictor that could account for forgetting independently of the training algorithm or hyperparameters. Such a predictor would both explain the empirical difference between RL and SFT and offer a unifying principle for catastrophic forgetting. Prior work has proposed candidates such as the magnitude of weight changes, sparsity of updates, or gradient rank. Across our experiments, however, none of these variables consistently aligned with the observed forgetting behavior (see Section [6](#S6 "6 Alternative Hypothesis ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less")). What did emerge was an empirical forgetting law: the KL divergence between the fine-tuned model and the base model, measured on the new task, reliably predicts the degree of forgetting.

Testing this hypothesis in large LLMs is challenging, since RL training is computationally expensive and cannot easily be run to convergence. Moreover, the search for predictors requires repeating fine-tuning many times under diverse conditions. To address these limitations, we designed a controlled toy setting, ParityMNIST, that allows us to replicate the RL–SFT gap under full convergence and perform systematic ablations.

ParityMNIST is derived from MNIST (Deng, [2012](#bib.bib16)), but reframes the task as predicting parity (even vs. odd). An image of an even digit is correctly classified if the model predicts *any* even digit label, and likewise for odd digits. Multiple output distributions are thus equally valid, mirroring a key property of the generative tasks we studied in section [3](#S3 "3 Reinforcement Learning Forgets Less than SFT ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less"): many distinct policies can achieve the same performance.

We pretrained a 3-layer MLP jointly on a subset of ParityMNIST and FashionMNIST (Xiao et al., [2017](#bib.bib72)), then fine-tuned only on ParityMNIST while measuring forgetting on FashionMNIST. This design provides a minimal, tractable setting for investigating predictors of forgetting. To parallel the main experiments:

* •

  In the SFT setting, the model was trained on labels sampled from a single arbitrary distribution out of the many possible correct ones.
* •

  In the RL setting, the reward was correctness with respect to parity, leaving the model free to converge to any valid distribution.

For more details, see Appendix [B.3](#A2.SS3 "B.3 MNIST Experiments ‣ Appendix B Training and Evaluation Details ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less"). This design allowed us to replicate the phenomenon where RL reached high accuracy on the new task with substantially slower degradation of prior knowledge, while SFT exhibited a steeper trade-off (Figure [3](#S4.F3 "Figure 3 ‣ 4 Smaller KL divergences lead to less forgetting ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less"), left). Importantly, *reproducing the effect in this simple MLP setting shows that it is not specific to large scale transformers, but a more general property of fine-tuning deep generative models*.

#### KL as Predictor.

Plotting forgetting against the KL divergence from the base model on ParityMNIST reveals a single functional relationship across both RL and SFT (Figure [3](#S4.F3 "Figure 3 ‣ 4 Smaller KL divergences lead to less forgetting ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less"), middle). This indicates that forgetting is determined by KL divergence, not by the choice of training algorithm. A quadratic fit achieves R2=0.96R^{2}=0.96 in this setting, underscoring the strength of the relationship. To test robustness, we repeated the experiment with two different arbitrary SFT labelings. Although their Pareto frontiers differed, the forgetting–KL curves coincided, confirming that KL consistently predicts forgetting irrespective of training method or label distribution. The same correlation appears in our LLM experiments, with a quadratic fit achieving R2=0.71R^{2}=0.71 (Figure [11](#A3.F11 "Figure 11 ‣ C.3 Optimization Dynamics ‣ Appendix C Additional Results ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less")). While weaker, the residuals are mean-zero and can be attributed to noise from approximate KL and accuracy estimation.

#### Optimal SFT Distribution.

To validate that KL divergence is the predictor variable, we constructed an oracle SFT distribution. In ParityMNIST, the simplicity of the task allows us to analytically identify the labeling that minimizes KL divergence to the base model among all distributions achieving 100% accuracy (Appendix [B.3](#A2.SS3 "B.3 MNIST Experiments ‣ Appendix B Training and Evaluation Details ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less")).
If KL divergence fully determines forgetting, then training SFT on this oracle distribution should yield the optimal accuracy–forgetting trade-off. The results in Figure [3](#S4.F3 "Figure 3 ‣ 4 Smaller KL divergences lead to less forgetting ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less") confirm this prediction—SFT trained on the oracle distribution retained more prior knowledge than RL, achieving the best trade-off observed. RL performs well because its on-policy updates bias the solution toward low-KL regions, but when SFT is explicitly guided to the KL-minimal distribution, it can surpass RL.
As an additional validation, we trained an SFT model on data generated by an RL-trained model. The distilled SFT matched RL’s accuracy–forgetting trade-off (Figure [9](#A3.F9 "Figure 9 ‣ C.3 Optimization Dynamics ‣ Appendix C Additional Results ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less")), reinforcing that the distribution learned, rather than the optimization algorithm, governs forgetting.

Takeaway 2

Catastrophic forgetting in both SFT and RL is predicted by the KL divergence between the fine-tuned and base models on the new task.

## 5 On-policy methods leads to smaller KL divergence

!(/html/2509.04259/assets/x4.png)

Figure 4: Comparison of algorithm classes. (Left) The four quadrants illustrate algorithm types, defined by whether they are on-policy or offline and whether they incorporate negative gradients.
(Middle) On-policy methods retain prior knowledge more effectively.
(Right) Both GRPO and 1-0 Reinforce achieve higher new-task accuracy while incurring smaller KL shifts from the base model, showing that on-policy methods consistently induce more conservative KL updates.

Having established that the KL divergence between the trained model and its base distribution on the new task predicts catastrophic forgetting, we now ask: why are RL fine-tuned models able to achieve strong task performance while moving less in KL than SFT models?

### 5.1 Experimental Evidence

To understand the difference in KL behavior, it is useful to contrast the training objectives of SFT and RL. For discrete outputs, SFT minimizes cross-entropy against a supervision distribution πβ\pi\_{\beta} over a distribution of inputs 𝒟\mathcal{D}:

|  |  |  |
| --- | --- | --- |
|  | ℒSFT​(π)=−𝔼x∼𝒟,y∼πβ​[log⁡π​(y|x)]\mathcal{L}\_{\text{SFT}}(\pi)=-\mathbb{E}\_{x\sim\mathcal{D},{\color[rgb]{0,.5,.5}\definecolor[named]{pgfstrokecolor}{rgb}{0,.5,.5}y\sim\pi\_{\beta}}}[\log\pi(y|x)] |  |

In contrast, RL with policy gradients optimizes\*\*\*Notice that in practice, the policy gradient trick (Sutton et al., [1998](#bib.bib65)) ensures gradients are taken only through the log-probability term, not through the sampling distribution inside the expectation.:

|  |  |  |
| --- | --- | --- |
|  | ℒRL​(π)=−𝔼x∼𝒟,y∼π​[A​(x,y)​log⁡π​(y|x)]\mathcal{L}\_{\text{RL}}(\pi)=-\mathbb{E}\_{x\sim\mathcal{D},{\color[rgb]{0,.5,.5}\definecolor[named]{pgfstrokecolor}{rgb}{0,.5,.5}y\sim\pi}}\left[{\color[rgb]{.75,0,.25}\definecolor[named]{pgfstrokecolor}{rgb}{.75,0,.25}A(x,y)}\log\pi(y|x)\right] |  |

where A​(x,y)A(x,y) is an Advantage function, which is the reward of yy normalized with respect to other rewards for the same xx. Two features distinguish this from SFT:

1. 1.

   Sampling Distribution. While in RL the training was done on outputs drawn from the model’s own distribution, in SFT they come from fixed external annotations.
2. 2.

   Negative Examples. While sampling from π\pi, some of the responses will be incorrect. These are usually assigned a negative coefficient A​(x,y)A(x,y). This pushes probability mass away from poor outputs, a mechanism absent in SFT.

Our hypothesis is that one of these two differences is what causes RL’s resistance to forgetting. To examine our hypothesis, we perform experiments with four different objectives:

* •

  GRPO. An on-policy objective that utilizes negative examples. Here, A​(x,y)A(x,y) is the normalized reward.
* •

  1–0 Reinforce. An on-policy algorithm that does not use negative examples. Here, A​(x,y)=1A(x,y)=1 for correct responses and 0 for incorrect ones. This is equivalent to sampling from the model and performing SFT on correct answers only.
* •

  SFT. An offline objective that does not use negative examples.
* •

  SimPO. An offline objective that utilizes negative examples. We create negative examples by sampling incorrect responses from an external model, and use the SFT data for positive examples. The SimPO (Meng et al., [2024](#bib.bib49)) loss compares correct and incorrect outputs via a logistic term:

  |  |  |  |
  | --- | --- | --- |
  |  | ℒSIMPO​(π)=−𝔼x∼𝒟,yw∼πβ+,yl∼πβ−​[log⁡σ​(log⁡π​(yw|x)−log⁡π​(yl|x)−1)]\mathcal{L}\_{\text{SIMPO}}(\pi)=-\mathbb{E}\_{x\sim\mathcal{D},y\_{w}\sim\pi\_{\beta^{+}},y\_{l}\sim\pi\_{\beta^{-}}}\left[\log\sigma\left(\log\pi(y\_{w}|x)-\log\pi(y\_{l}|x)-1\right)\right] |  |

  where πβ+\pi\_{\beta^{+}} and πβ−\pi\_{\beta^{-}} denote distributions for correct and incorrect responses, respectively. We used SimPO rather than naïve likelihood/negative likelihood because the latter was unstable to train.

We compared the four objectives on the Science Q&A task, measuring their learning–forgetting trade-offs as in Section 4. The results, shown in Figure [4](#S5.F4 "Figure 4 ‣ 5 On-policy methods leads to smaller KL divergence ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less"), reveal that 1–0 Reinforce behaves similarly to GRPO, while SimPO resembles SFT. Thus, the critical factor is not the presence of negative gradients but the use of on-policy data.
Plotting KL divergence confirms this conclusion: on-policy methods (GRPO and 1–0 Reinforce) reach the same task performance with significantly smaller KL divergence from the base model than offline methods (SFT and SimPO).

### 5.2 Theoretical Perspective

!(/html/2509.04259/assets/x5.png)

Figure 5: KL-minimal path to optimality. Alternating I-projection into the set of optimal policies and M-projection into Π\Pi carries π0\pi\_{0} into P∗P^{\*} while preferring the closest solution in KL.

Beyond the empirical results, it is useful to ask why on-policy methods naturally induce smaller KL shifts. One way to see this is through the lens of projection in probability space: policy gradient methods can be understood as a conservative projection that keeps the policy close to its starting point while reweighting toward higher-reward outcomes. At each step, the policy samples outputs it already finds likely, then re-weights those samples according to reward, shifting probability mass toward higher-reward outcomes while suppressing lower-reward ones. Crucially, because updates are defined relative to the model’s own distribution, they nudge the policy toward a nearby re-weighted distribution, rather than pulling it toward a potentially distant external distribution (as in SFT). This explains why policy gradient methods tend to remain close to the base model in KL divergence.

This perspective can be formalized by observing that, in the binary-reward case, the re-weighted distribution targeted by policy gradient is exactly the minimum-KL projection of the current policy onto the set of optimal ones.

###### Lemma 5.1.

Let pp be a distribution over a finite set YY, and let R:Y→{0,1}R:Y\to\{0,1\} be a reward function. Rejection sampling from pp with acceptance condition R​(y)=1R(y)=1 yields a distribution qRSq\_{\mathrm{RS}}. This distribution can be equivalently characterized as the solution to:

|  |  |  |
| --- | --- | --- |
|  | qRS=arg​minqDKL(q||p)s.t𝔼y∼q[R(y)]=1q\_{\text{RS}}=\operatorname\*{arg\,min}\_{q}D\_{\text{KL}}(q||p)\quad s.t\quad\mathbb{E}\_{y\sim q}[R(y)]=1 |  |

Building on this, we show that policy gradient converges to the KL-minimal optimal policy within the representable family.

###### Theorem 5.2.

Let YY be a finite set and let Π⊆Δ​(Y)\Pi\subseteq\Delta(Y) be a convex family of feasible policies (e.g., an exponential family). Let R:Y→{0,1}R:Y\to\{0,1\} be a binary reward function and P∗={q:𝔼q​[R]=1}P^{\*}=\{q:\mathbb{E}\_{q}[R]=1\} the set of optimal policies. Then, under suitable regularity conditions, solving the reinforcement learning objective with policy gradient converges to

|  |  |  |
| --- | --- | --- |
|  | π†=arg⁡minπ∈P∗∩Π⁡DKL​(π∥π0),\pi^{\dagger}=\arg\min\_{\pi\in P^{\*}\cap\Pi}D\_{\mathrm{KL}}(\pi\,\|\,\pi\_{0}), |  |

where π0\pi\_{0} is the initialization.
In other words, policy gradient selects, among all optimal representable policies, the one closest in KL-divergence to the starting policy.

A detailed version with proofs is provided in Appendix [A](#A1 "Appendix A Theory ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less").

Takeaway 3

On-policy training explains why RL maintains smaller KL divergence than SFT. Sampling from the model’s own distribution keeps it close to the base model, while SFT pushes it toward arbitrary external distributions.

## 6 Alternative Hypothesis

Science advances not only by identifying the right explanations, but also by eliminating incorrect ones. To this end, we systematically evaluated alternative variables as potential predictors of catastrophic forgetting, grouped into four categories:

* •

  Weight-level changes. Many prior work tried to mitigate forgetting by constraining the change in parameter space (Kirkpatrick et al., [2017](#bib.bib37); Aljundi et al., [2018](#bib.bib2); Zenke et al., [2017](#bib.bib74)). We measured parameter changes under L1L\_{1}, Fisher-weighted L2L\_{2}, and spectral norm metrics. The Fisher matrix was computed on the basis of the model parameters, with expectation over inputs from the previous task. These metrics correlated only weakly with forgetting: large parameter shifts could occur without forgetting, and conversely, forgetting sometimes occurred despite small parameter movement.
* •

  Representation-level changes. Some other papers focused on maintaining the previous features (Jung et al., [2018](#bib.bib35); Hou et al., [2019](#bib.bib29); Dhar et al., [2019](#bib.bib17)). We examined hidden activation shifts (L1 and L2 distances) as proxies for changes in internal representations. Although we found that there is representation drift during training (see Appendix [C.1](#A3.SS1 "C.1 Representation Preservation ‣ Appendix C Additional Results ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less")), the curves were distinct between training objectives, meaning that it is not a good predictor.
* •

  Sparsity and rank of updates. Motivated by Mukherjee et al. ([2025](#bib.bib51)), who argue that RL updates are sparse while SFT weight updates are dense, we explicitly tested this hypothesis. In our setting, however, we found that the reason for the observed sparse updates was the use of bfloat16 for model training. Since bfloat16 has a limited mantissa, small parameter updates (such as those produced by RL) can fail to cross the representational threshold, effectively causing no update at all. Performing the same training with float32 resulted in models with identical performance but without any sparsity in their weight updates. Checking the rank of the weight changes, we found that all algorithms lead to full rank weight updates.
* •

  Distributional distances. We considered multiple measures of output distribution change, all measured over inputs from the new task τ\tau: Forward KL (𝔼x∼τ[KL(π0||π)]\mathbb{E}\_{x\sim\tau}\big[\text{KL}(\pi\_{0}||\pi)\big]), Reverse KL (𝔼x∼τ[KL(π||π0)]\mathbb{E}\_{x\sim\tau}\big[\text{KL}(\pi||\pi\_{0})\big]), Total Variation, and L2L\_{2} distance between distributions. While reverse KL showed a good signal, and TV moderately correlated with forgetting, none approached the predictive power of forward KL.

Table [1](#S6.T1 "Table 1 ‣ 6 Alternative Hypothesis ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less") summarizes these results for the MNIST task. Across all candidates, forward KL divergence between the fine-tuned and base model evaluated on the new task emerges as the only consistent and high-fidelity predictor of catastrophic forgetting.

|  |  |
| --- | --- |
| Variable | R2R^{2} (2nd deg. polynomial) |
| KL, forward | 0.96 ±\pm\,0.01 |
| KL, reverse | 0.93±0.010.93\pm 0.01 |
| TV | 0.80±0.010.80\pm 0.01 |
| Distribution change, L2 | 0.56±0.020.56\pm 0.02 |
| Weight change, L1 | 0.34±0.020.34\pm 0.02 |
| Weight change, Fisher Weighted L2 | 0.58±0.020.58\pm 0.02 |
| Weight change, spectral norm | 0.58±0.020.58\pm 0.02 |
| Sparsity of weight change | N/A |
| Rank of weight change | N/A |
| Activation change, L1 | 0.52±0.020.52\pm 0.02 |
| Activation change, L2 | 0.55±0.020.55\pm 0.02 |

Table 1: Predictive power of alternative variables compared to forward KL. None approaches the explanatory strength of forward KL divergence.

## 7 Discussion and Conclusion

Our study reveals that catastrophic forgetting is governed not by the choice of training algorithm, but by the KL divergence from the base policy evaluated on the new task. This explains why RL forgets less than SFT, as on-policy training naturally biases updates toward KL-minimal solutions, preserving prior knowledge while acquiring new skills.

However, we still lack a mechanistic account of why larger KL shifts on the new task disrupt prior knowledge—whether through representational interference, implicit capacity limits, or other dynamics. Moreover, while we demonstrate the KL–forgetting link across moderate-scale LLMs and toy models, its behavior at frontier scales and in more diverse generative domains remains unknown. In addition, we didn’t study online but off-policy algorithms, which are popular in RL. Addressing these gaps will be essential for grounding the principle and extending it to real-world deployment.

Taken together, our results motivate a new design axis for post-training research: algorithms should be judged not only by how well they optimize new tasks, but also by how conservatively they move in KL relative to the base model. Importantly, this does not mean offline data cannot help, but that continual learning requires updates to keep learning close to the KL-minimal path. Embracing this principle may allow us to build agents that not only learn new skills, but also truly learn for life.

## Acknowledgment

We want to express our gratitude to Nitish Dashora, Seungwook Han, Moritz Reuss, Zhang-Wei Hong, Leshem Choshen, Ahmad Beirami, Mehul Damani, Akarsh Kumar, and members of the Improbable AI lab for the helpful discussion on the paper. We are grateful to MIT Supercloud
and the Lincoln Laboratory Supercomputing Center for providing HPC resources. The research was supported in part by Hyundai Motor Company, Qualcomm Innovation Fellowship, Google, and Amazon. The research was sponsored
by the Army Research Office and was accomplished under
Grant Number W911NF-21-1-0328. The research was also sponsored by the Office of Naval Research and was accomplished under Grant Number N00014-22-1-2740. Research was also sponsored by the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein. The views and conclusions contained in this document are those of the authors
and should not be interpreted as representing the official
policies, either expressed or implied, of the Army Research
Office, Naval Research Office, Air Force, or the U.S. Government.

## Author Contributions

Jyothish Pari co-developed the project and contributed in all aspects of experiments and writing.

Idan Shenfeld co-developed the project and contributed in all aspects of experiments and writing.

Pulkit Agrawal co-developed the project direction, advised IS and JP, and played a significant role in paper writing.

## References

* Achiam et al. (2023)

  Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al.
  Gpt-4 technical report.
  *arXiv preprint arXiv:2303.08774*, 2023.
* Aljundi et al. (2018)

  Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, and Tinne Tuytelaars.
  Memory aware synapses: Learning what (not) to forget.
  In *Proceedings of the European conference on computer vision (ECCV)*, pp. 139–154, 2018.
* Amari & Nagaoka (2000)

  Shun-ichi Amari and Hiroshi Nagaoka.
  *Methods of information geometry*, volume 191.
  American Mathematical Soc., 2000.
* Bommasani (2021)

  Rishi Bommasani.
  On the opportunities and risks of foundation models.
  *arXiv preprint arXiv:2108.07258*, 2021.
* Brohan et al. (2022)

  Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al.
  Rt-1: Robotics transformer for real-world control at scale.
  *arXiv preprint arXiv:2212.06817*, 2022.
* Brown et al. (2020)

  Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al.
  Language models are few-shot learners.
  *Advances in neural information processing systems*, 33:1877–1901, 2020.
* Castro et al. (2018)

  Francisco M Castro, Manuel J Marín-Jiménez, Nicolás Guil, Cordelia Schmid, and Karteek Alahari.
  End-to-end incremental learning.
  In *Proceedings of the European conference on computer vision (ECCV)*, pp. 233–248, 2018.
* Chaudhry et al. (2018)

  Arslan Chaudhry, Puneet K Dokania, Thalaiyasingam Ajanthan, and Philip HS Torr.
  Riemannian walk for incremental learning: Understanding forgetting and intransigence.
  In *Proceedings of the European conference on computer vision (ECCV)*, pp. 532–547, 2018.
* Chen et al. (2021)

  Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al.
  Evaluating large language models trained on code.
  *arXiv preprint arXiv:2107.03374*, 2021.
* Chu et al. (2025)

  Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V Le, Sergey Levine, and Yi Ma.
  Sft memorizes, rl generalizes: A comparative study of foundation model post-training.
  *arXiv preprint arXiv:2501.17161*, 2025.
* Chung et al. (2024)

  Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al.
  Scaling instruction-finetuned language models.
  *Journal of Machine Learning Research*, 25(70):1–53, 2024.
* Cossu et al. (2024)

  Andrea Cossu, Antonio Carta, Lucia Passaro, Vincenzo Lomonaco, Tinne Tuytelaars, and Davide Bacciu.
  Continual pre-training mitigates forgetting in language and vision.
  *Neural Networks*, 179:106492, 2024.
* Csiszár (1984)

  Imre Csiszár.
  Information geometry and alternating minimization procedures.
  *Statistics and Decisions, Dedewicz*, 1:205–237, 1984.
* Dao & Le (2025)

  Alan Dao and Thinh Le.
  Rezero: Enhancing llm search ability by trying one-more-time.
  *arXiv preprint arXiv:2504.11001*, 2025.
* Dempster et al. (1977)

  Arthur P Dempster, Nan M Laird, and Donald B Rubin.
  Maximum likelihood from incomplete data via the em algorithm.
  *Journal of the royal statistical society: series B (methodological)*, 39(1):1–22, 1977.
* Deng (2012)

  Li Deng.
  The mnist database of handwritten digit images for machine learning research [best of the web].
  *IEEE signal processing magazine*, 29(6):141–142, 2012.
* Dhar et al. (2019)

  Prithviraj Dhar, Rajat Vikram Singh, Kuan-Chuan Peng, Ziyan Wu, and Rama Chellappa.
  Learning without memorizing.
  In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 5138–5146, 2019.
* Dodge et al. (2020)

  Jesse Dodge, Gabriel Ilharco, Roy Schwartz, Ali Farhadi, Hannaneh Hajishirzi, and Noah Smith.
  Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping.
  *arXiv preprint arXiv:2002.06305*, 2020.
* Feng et al. (2024)

  Kehua Feng, Keyan Ding, Weijie Wang, Xiang Zhuang, Zeyuan Wang, Ming Qin, Yu Zhao, Jianhua Yao, Qiang Zhang, and Huajun Chen.
  Sciknoweval: Evaluating multi-level scientific knowledge of large language models.
  *arXiv preprint arXiv:2406.09098*, 2024.
* French (1999)

  Robert M French.
  Catastrophic forgetting in connectionist networks.
  *Trends in cognitive sciences*, 3(4):128–135, 1999.
* Gao et al. (2025)

  Huan-ang Gao, Jiayi Geng, Wenyue Hua, Mengkang Hu, Xinzhe Juan, Hongzhang Liu, Shilong Liu, Jiahao Qiu, Xuan Qi, Yiran Wu, et al.
  A survey of self-evolving agents: On path to artificial super intelligence.
  *arXiv preprint arXiv:2507.21046*, 2025.
* Gao et al. (2023)

  Leo Gao, John Schulman, and Jacob Hilton.
  Scaling laws for reward model overoptimization.
  In *International Conference on Machine Learning*, pp. 10835–10866. PMLR, 2023.
* Gao et al. (2024)

  Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou.
  The language model evaluation harness, 07 2024.
  URL <https://zenodo.org/records/12608602>.
* Gunawardana et al. (2005)

  Asela Gunawardana, William Byrne, and Michael I Jordan.
  Convergence theorems for generalized alternating minimization procedures.
  *Journal of machine learning research*, 6(12), 2005.
* Guo et al. (2025a)

  Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al.
  Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.
  *arXiv preprint arXiv:2501.12948*, 2025a.
* Guo et al. (2025b)

  Haiyang Guo, Fanhu Zeng, Fei Zhu, Jiayi Wang, Xukai Wang, Jingang Zhou, Hongbo Zhao, Wenzhuo Liu, Shijie Ma, Da-Han Wang, et al.
  A comprehensive survey on continual learning in generative models.
  *arXiv preprint arXiv:2506.13045*, 2025b.
* Han et al. (2025)

  Seungwook Han, Jyothish Pari, Samuel J Gershman, and Pulkit Agrawal.
  General reasoning requires learning to reason from the get-go.
  *arXiv preprint arXiv:2502.19402*, 2025.
* Hendrycks et al. (2020)

  Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
  Measuring massive multitask language understanding.
  *arXiv preprint arXiv:2009.03300*, 2020.
* Hou et al. (2019)

  Saihui Hou, Xinyu Pan, Chen Change Loy, Zilei Wang, and Dahua Lin.
  Learning a unified classifier incrementally via rebalancing.
  In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 831–839, 2019.
* Howard & Ruder (2018)

  Jeremy Howard and Sebastian Ruder.
  Universal language model fine-tuning for text classification.
  *arXiv preprint arXiv:1801.06146*, 2018.
* Hu et al. (2025)

  Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum.
  Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model.
  *arXiv preprint arXiv:2503.24290*, 2025.
* Hu et al. (2023)

  Yafei Hu, Quanting Xie, Vidhi Jain, Jonathan Francis, Jay Patrikar, Nikhil Keetha, Seungchan Kim, Yaqi Xie, Tianyi Zhang, Hao-Shu Fang, et al.
  Toward general-purpose robots via foundation models: A survey and meta-analysis.
  *arXiv preprint arXiv:2312.08782*, 2023.
* Huan et al. (2025)

  Maggie Huan, Yuetai Li, Tuney Zheng, Xiaoyu Xu, Seungone Kim, Minxin Du, Radha Poovendran, Graham Neubig, and Xiang Yue.
  Does math reasoning improve general llm capabilities? understanding transferability of llm reasoning.
  *arXiv preprint arXiv:2507.00432*, 2025.
* Huh et al. (2024)

  Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola.
  The platonic representation hypothesis.
  *arXiv preprint arXiv:2405.07987*, 2024.
* Jung et al. (2018)

  Heechul Jung, Jeongwoo Ju, Minju Jung, and Junmo Kim.
  Less-forgetful learning for domain expansion in deep neural networks.
  In *Proceedings of the AAAI conference on artificial intelligence*, volume 32, 2018.
* Kim et al. (2024)

  Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al.
  Openvla: An open-source vision-language-action model.
  *arXiv preprint arXiv:2406.09246*, 2024.
* Kirkpatrick et al. (2017)

  James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al.
  Overcoming catastrophic forgetting in neural networks.
  *Proceedings of the national academy of sciences*, 114(13):3521–3526, 2017.
* Kornblith et al. (2019)

  Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton.
  Similarity of neural network representations revisited.
  In *International conference on machine learning*, pp. 3519–3529. PMlR, 2019.
* Lai et al. (2025)

  Song Lai, Haohan Zhao, Rong Feng, Changyi Ma, Wenzhuo Liu, Hongbo Zhao, Xi Lin, Dong Yi, Min Xie, Qingfu Zhang, et al.
  Reinforcement fine-tuning naturally mitigates forgetting in continual post-training.
  *arXiv preprint arXiv:2507.05386*, 2025.
* Li et al. (2024a)

  Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, Jianfeng Gao, et al.
  Multimodal foundation models: From specialists to general-purpose assistants.
  *Foundations and Trends® in Computer Graphics and Vision*, 16(1-2):1–214, 2024a.
* Li et al. (2025a)

  Tianle Li, Jihai Zhang, Yongming Rao, and Yu Cheng.
  Unveiling the compositional ability gap in vision-language reasoning model.
  *arXiv preprint arXiv:2505.19406*, 2025a.
* Li et al. (2024b)

  Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, Sergey Levine, Jiajun Wu, Chelsea Finn, Hao Su, Quan Vuong, and Ted Xiao.
  Evaluating real-world robot manipulation policies in simulation.
  *arXiv preprint arXiv:2405.05941*, 2024b.
* Li & Hoiem (2017)

  Zhizhong Li and Derek Hoiem.
  Learning without forgetting.
  *IEEE transactions on pattern analysis and machine intelligence*, 40(12):2935–2947, 2017.
* Li et al. (2025b)

  Zhongyang Li, Ziyue Li, and Tianyi Zhou.
  C3po: Critical-layer, core-expert, collaborative pathway optimization for test-time expert re-mixing.
  *ArXiv*, abs/2504.07964, 2025b.
  URL <https://api.semanticscholar.org/CorpusID:277667633>.
* Lin et al. (2021)

  Stephanie Lin, Jacob Hilton, and Owain Evans.
  Truthfulqa: Measuring how models mimic human falsehoods.
  *arXiv preprint arXiv:2109.07958*, 2021.
* Liu et al. (2025)

  Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin.
  Understanding r1-zero-like training: A critical perspective.
  *arXiv preprint arXiv:2503.20783*, 2025.
* Luo et al. (2023)

  Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang.
  An empirical study of catastrophic forgetting in large language models during continual fine-tuning.
  *arXiv preprint arXiv:2308.08747*, 2023.
* McCloskey & Cohen (1989)

  Michael McCloskey and Neal J Cohen.
  Catastrophic interference in connectionist networks: The sequential learning problem.
  In *Psychology of learning and motivation*, volume 24, pp. 109–165. Elsevier, 1989.
* Meng et al. (2024)

  Yu Meng, Mengzhou Xia, and Danqi Chen.
  Simpo: Simple preference optimization with a reference-free reward.
  *Advances in Neural Information Processing Systems*, 37:124198–124235, 2024.
* Moradi et al. (2025)

  Mohammad Mahdi Moradi, Hossam Amer, Sudhir Mudur, Weiwei Zhang, Yang Liu, and Walid Ahmed.
  Continuous self-improvement of large language models by test-time training with verifier-driven sample selection.
  *ArXiv*, abs/2505.19475, 2025.
  URL <https://api.semanticscholar.org/CorpusID:278905330>.
* Mukherjee et al. (2025)

  Sagnik Mukherjee, Lifan Yuan, Dilek Hakkani-Tur, and Hao Peng.
  Reinforcement learning finetunes small subnetworks in large language models.
  *arXiv preprint arXiv:2505.11711*, 2025.
* Ouyang et al. (2022)

  Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al.
  Training language models to follow instructions with human feedback.
  *Advances in neural information processing systems*, 35:27730–27744, 2022.
* Qwen et al. (2025)

  Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu.
  Qwen2.5 technical report, 2025.
  URL <https://arxiv.org/abs/2412.15115>.
* Radford et al. (2018)

  Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al.
  Improving language understanding by generative pre-training.
  *arXiv preprint arXiv:2303.08774*, 2018.
* Radford et al. (2021)

  Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al.
  Learning transferable visual models from natural language supervision.
  In *International conference on machine learning*, pp. 8748–8763. PmLR, 2021.
* Rafailov et al. (2023)

  Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn.
  Direct preference optimization: Your language model is secretly a reward model.
  *Advances in neural information processing systems*, 36:53728–53741, 2023.
* Ramasesh et al. (2021)

  Vinay Venkatesh Ramasesh, Aitor Lewkowycz, and Ethan Dyer.
  Effect of scale on catastrophic forgetting in neural networks.
  In *International conference on learning representations*, 2021.
* Rannen et al. (2017)

  Amal Rannen, Rahaf Aljundi, Matthew B Blaschko, and Tinne Tuytelaars.
  Encoder based lifelong learning.
  In *Proceedings of the IEEE international conference on computer vision*, pp. 1320–1328, 2017.
* Rebuffi et al. (2017)

  Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert.
  icarl: Incremental classifier and representation learning.
  In *Proceedings of the IEEE conference on Computer Vision and Pattern Recognition*, pp. 2001–2010, 2017.
* Ross et al. (2011)

  Stéphane Ross, Geoffrey Gordon, and Drew Bagnell.
  A reduction of imitation learning and structured prediction to no-regret online learning.
  In *Proceedings of the fourteenth international conference on artificial intelligence and statistics*, pp. 627–635. JMLR Workshop and Conference Proceedings, 2011.
* Sakaguchi et al. (2021)

  Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi.
  Winogrande: An adversarial winograd schema challenge at scale.
  *Communications of the ACM*, 64(9):99–106, 2021.
* Shao et al. (2024)

  Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al.
  Deepseekmath: Pushing the limits of mathematical reasoning in open language models.
  *arXiv preprint arXiv:2402.03300*, 2024.
* Simonds & Yoshiyama (2025)

  Toby Simonds and Akira Yoshiyama.
  Ladder: Self-improving llms through recursive problem decomposition.
  *arXiv preprint arXiv:2503.00735*, 2025.
* Stiennon et al. (2020)

  Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano.
  Learning to summarize with human feedback.
  *Advances in neural information processing systems*, 33:3008–3021, 2020.
* Sutton et al. (1998)

  Richard S Sutton, Andrew G Barto, et al.
  *Reinforcement learning: An introduction*, volume 1.
  MIT press Cambridge, 1998.
* Tang et al. (2023)

  Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, Boxi Cao, and Le Sun.
  Toolalpaca: Generalized tool learning for language models with 3000 simulated cases.
  *arXiv preprint arXiv:2306.05301*, 2023.
* Touvron et al. (2023)

  Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al.
  Llama: Open and efficient foundation language models.
  *arXiv preprint arXiv:2302.13971*, 2023.
* Wang et al. (2024)

  Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu.
  A comprehensive survey of continual learning: Theory, method and application.
  *IEEE transactions on pattern analysis and machine intelligence*, 46(8):5362–5383, 2024.
* Wei et al. (2021)

  Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le.
  Finetuned language models are zero-shot learners.
  *arXiv preprint arXiv:2109.01652*, 2021.
* Wu (1983)

  CF Jeff Wu.
  On the convergence properties of the em algorithm.
  *The Annals of statistics*, pp. 95–103, 1983.
* Wu et al. (2019)

  Yue Wu, Yinpeng Chen, Lijuan Wang, Yuancheng Ye, Zicheng Liu, Yandong Guo, and Yun Fu.
  Large scale incremental learning.
  In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 374–382, 2019.
* Xiao et al. (2017)

  Han Xiao, Kashif Rasul, and Roland Vollgraf.
  Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms.
  *arXiv preprint arXiv:1708.07747*, 2017.
* Zellers et al. (2019)

  Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi.
  Hellaswag: Can a machine really finish your sentence?
  *arXiv preprint arXiv:1905.07830*, 2019.
* Zenke et al. (2017)

  Friedemann Zenke, Ben Poole, and Surya Ganguli.
  Continual learning through synaptic intelligence.
  In *International conference on machine learning*, pp. 3987–3995. PMLR, 2017.
* Zhai et al. (2024)

  Simon Zhai, Hao Bai, Zipeng Lin, Jiayi Pan, Peter Tong, Yifei Zhou, Alane Suhr, Saining Xie, Yann LeCun, Yi Ma, et al.
  Fine-tuning large vision-language models as decision-making agents via reinforcement learning.
  *Advances in neural information processing systems*, 37:110935–110971, 2024.
* Zhou et al. (2023)

  Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou.
  Instruction-following evaluation for large language models.
  *arXiv preprint arXiv:2311.07911*, 2023.
* Ziegler et al. (2019)

  Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving.
  Fine-tuning language models from human preferences.
  *arXiv preprint arXiv:1909.08593*, 2019.
* Zweiger et al. (2025)

  Adam Zweiger, Jyothish Pari, Han Guo, Ekin Akyürek, Yoon Kim, and Pulkit Agrawal.
  Self-adapting language models.
  *ArXiv*, abs/2506.10943, 2025.
  URL <https://api.semanticscholar.org/CorpusID:279318966>.

## Appendix A Theory

###### Lemma A.1 (Rejection sampling as an I-projection).

Let pp be a distribution over a finite set YY, and let R:Y→{0,1}R:Y\to\{0,1\} be a reward function. Rejection sampling from pp with acceptance condition R​(y)=1R(y)=1 yields a distribution qRSq\_{\mathrm{RS}}. This distribution can be equivalently characterized as the solution to:

|  |  |  |
| --- | --- | --- |
|  | qRS=arg​minqDKL(q||p)s.t𝔼y∼q[R(y)]=1q\_{\text{RS}}=\operatorname\*{arg\,min}\_{q}D\_{\text{KL}}(q||p)\quad s.t\quad\mathbb{E}\_{y\sim q}[R(y)]=1 |  |

Equivalently, qRSq\_{\text{RS}} is the I-projection of pp onto the set {q:𝔼q​[R]=1}\{q:\mathbb{E}\_{q}[R]=1\}

###### Proof.

Let S={y∈Y:R​(y)=1}S=\{y\in Y:R(y)=1\}. Rejection sampling produces the conditional distribution

|  |  |  |
| --- | --- | --- |
|  | qRS​(y)={p​(y)p​(S)y∈S,0y∉S,q\_{\mathrm{RS}}(y)=\begin{cases}\tfrac{p(y)}{p(S)}&y\in S,\\ 0&y\notin S,\end{cases} |  |

where p​(S)=∑y∈Sp​(y)p(S)=\sum\_{y\in S}p(y) and we assume P​(S)>0P(S)>0.

Now consider the optimization problem. The constraint 𝔼q​[R]=1\mathbb{E}\_{q}[R]=1 means

|  |  |  |
| --- | --- | --- |
|  | ∑y∈Yq​(y)​R​(y)=∑y∈Sq​(y)=1\sum\_{y\in Y}q(y)R(y)=\sum\_{y\in S}q(y)=1 |  |

so qq must put all of its mass on SS. Thus the feasible set is exactly all distributions supported on SS.

For any qq supported on SS, we can write p​(y)=p​(S)​p​(y|S)p(y)=p(S)\,p(y|S) for y∈Sy\in S, and then

|  |  |  |  |
| --- | --- | --- | --- |
|  | DKL​(q∥p)\displaystyle D\_{\mathrm{KL}}(q\|p) | =∑y∈Sq​(y)​log⁡q​(y)p​(y)=∑y∈Sq​(y)​log⁡q​(y)p​(y∣S)−log⁡p​(S)​∑y∈Sq​(y)\displaystyle=\sum\_{y\in S}q(y)\log\frac{q(y)}{p(y)}=\sum\_{y\in S}q(y)\log\frac{q(y)}{p(y\mid S)}-\log p(S)\sum\_{y\in S}q(y) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =DKL(q∥p(⋅∣S))−logp(S)\displaystyle=D\_{\mathrm{KL}}\bigl(q\|p(\cdot\mid S)\bigr)-\log p(S) |  |

where we used ∑y∈Sq​(y)=1\sum\_{y\in S}q(y)=1 in the last step. The second term is constant in qq, so minimizing DKL(q||p)D\_{\mathrm{KL}}(q||p) is the same as minimizing DKL(q||p(⋅|S))D\_{\mathrm{KL}}(q||p(\cdot|S)).
By strict convexity of
DKL(⋅∥⋅)D\_{\mathrm{KL}}(\cdot\|\cdot) in its first argument, the unique minimizer is
q=p(⋅∣S)=qRSq=p(\cdot\mid S)=q\_{\mathrm{RS}}.
∎

###### Lemma A.2 (Policy gradient as an M-projection).

Let YY be a finite set and let Π⊆Δ​(Y)\Pi\subseteq\Delta(Y) be a set of admissible policies (distributions over YY).
Consider the single-step reinforcement learning objective

|  |  |  |
| --- | --- | --- |
|  | maxπ⁡𝔼y∼π​[R​(y)]\max\_{\pi}\mathbb{E}\_{y\sim\pi}[R(y)] |  |

where R:Y→ℝ≥0R:Y\to\mathbb{R}\_{\geq 0} is a reward function.
By the policy gradient theorem, this objective is equivalently optimized by

|  |  |  |
| --- | --- | --- |
|  | maxπ⁡𝔼y∼π¯​[R​(y)​log⁡π​(y)]\max\_{\pi}\mathbb{E}\_{y\sim\bar{\pi}}\bigl[R(y)\log\pi(y)\bigr] |  |

where π¯\bar{\pi} indicates that gradients are not propagated through the sampling distribution.
Define the distribution

|  |  |  |
| --- | --- | --- |
|  | q​(y)=π​(y)​R​(y)Z,Z=∑y∈Yπ​(y)​R​(y)q(y)=\frac{\pi(y)R(y)}{Z},\qquad Z=\sum\_{y\in Y}\pi(y)R(y) |  |

Then taking a policy gradient step is equivalent to taking a gradient step on the following objective:

|  |  |  |
| --- | --- | --- |
|  | minπ−𝔼y∼q​[log⁡π​(y)]\min\_{\pi}-\mathbb{E}\_{y\sim q}[\log\pi(y)] |  |

In other words, optimizing the RL objective using policy gradient is equivalent to finding the MM-projection of qq onto the set of feasible policies π\pi using gradient descent.

###### Proof.

Expanding the policy gradient objective gives

|  |  |  |
| --- | --- | --- |
|  | 𝔼y∼π¯​[R​(y)​log⁡π​(y)]=∑y∈Yπ​(y)​R​(y)​log⁡π​(y)\mathbb{E}\_{y\sim\bar{\pi}}[R(y)\log\pi(y)]=\sum\_{y\in Y}\pi(y)R(y)\log\pi(y) |  |

Let Z=∑y∈Yπ​(y)​R​(y)Z=\sum\_{y\in Y}\pi(y)R(y). Define q​(y)=π​(y)​R​(y)/Zq(y)=\pi(y)R(y)/Z. Then the above becomes

|  |  |  |
| --- | --- | --- |
|  | ∑y∈Yπ​(y)​R​(y)​log⁡π​(y)=Z​∑y∈Yq​(y)​log⁡π​(y)=Z​𝔼y∼q​[log⁡π​(y)]\sum\_{y\in Y}\pi(y)R(y)\log\pi(y)=Z\sum\_{y\in Y}q(y)\log\pi(y)=Z\,\mathbb{E}\_{y\sim q}[\log\pi(y)] |  |

Since ZZ does not depend on π\pi in the gradient computation (it is treated as a constant in the π¯\bar{\pi} sense), maximizing the original objective is equivalent to maximizing 𝔼y∼q​[log⁡π​(y)]\mathbb{E}\_{y\sim q}[\log\pi(y)].

Finally, recall that the MM-projection of a distribution qq onto a set of distributions Π\Pi is given by

|  |  |  |
| --- | --- | --- |
|  | minπ∈Π⁡KL​(q∥π)=𝔼q​[log⁡qπ]=𝔼q​[log⁡q]−𝔼q​[log⁡π]\min\_{\pi\in\Pi}\mathrm{KL}(q\|\pi)=\mathbb{E}\_{q}[\log\frac{q}{\pi}]\ =\mathbb{E}\_{q}[\log q]\;-\;\mathbb{E}\_{q}[\log\pi] |  |

since 𝔼q​[log⁡q]\mathbb{E}\_{q}[\log q] does not depend on π\pi, the maximizer of 𝔼π¯​[R​log⁡π]\mathbb{E}\_{\bar{\pi}}[R\log\pi] over Π\Pi coincides with arg⁡minπ∈Π⁡KL​(q∥π)\arg\min\_{\pi\in\Pi}\mathrm{KL}(q\|\pi).
Thus, the policy gradient update corresponds to the MM-projection of qq onto the policy class.
∎

###### Theorem A.3 (RL with binary reward as an EM algorithm).

Let YY be a finite set and let Π⊆Δ​(Y)\Pi\subseteq\Delta(Y) be a set of feasible policies. Let R:Y→{0,1}R:Y\to\{0,1\} be a binary reward function and P∗P^{\*} the set of all optimal policies P∗={q:𝔼q​[R]=1}P^{\*}=\{q:\mathbb{E}\_{q}[R]=1\}. Then, solving the Single-step reinforcement learning objective using policy gradients is equivalent to performing the following optimization procedure:

|  |  |  |
| --- | --- | --- |
|  | qt=arg⁡minq∈P∗⁡KL​(q∥πt),πt+1=arg⁡minπ∈Π⁡KL​(qt∥π)q\_{t}=\arg\min\_{q\in P^{\*}}\mathrm{KL}(q\|\pi\_{t}),\qquad\pi\_{t+1}=\arg\min\_{\pi\in\Pi}\mathrm{KL}(q\_{t}\|\pi) |  |

This procedure is also known as EM with information projection.

###### Proof.

Sampling y∼πy\sim\pi and accepting iff R​(y)=1R(y)=1 is exactly rejection sampling onto the event S={y∈Y:R​(y)=1}S=\{y\in Y:R(y)=1\}. The resulting distribution is π(⋅|S)\pi(\cdot|S).
By Lemma A.1 with p←πp\leftarrow\pi, this π(⋅|S)\pi(\cdot|S) solves

|  |  |  |
| --- | --- | --- |
|  | minq⁡DKL​(q∥π)s.t.𝔼q​[R]=1\min\_{q}D\_{\mathrm{KL}}(q\|\pi)\quad\text{s.t.}\quad\mathbb{E}\_{q}[R]=1 |  |

establishing the I-projection. Applying Lemma A.2 on the RL objective gives us the M-projection.
∎

###### Proposition A.4 (Convergence to minimum KL solution).

Under the setting appear in theorem [A.3](#A1.Thmtheorem3 "Theorem A.3 (RL with binary reward as an EM algorithm). ‣ Appendix A Theory ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less") and assume Π\Pi is an e-flat (exponential-family) model with full support, the optimal set P∗P^{\*} is nonempty and realizable (i.e., Π∩P∗≠∅\Pi\cap P^{\*}\neq\varnothing). Then:

(1) If the M-projection is exact at every step, then (πt)(\pi\_{t}) converges to

|  |  |  |
| --- | --- | --- |
|  | π†=arg⁡minπ∈P∗∩Π⁡DKL​(π∥π0)\pi^{\dagger}=\arg\min\_{\pi\in P^{\*}\cap\Pi}D\_{\mathrm{KL}}(\pi\,\|\,\pi\_{0}) |  |

(2) If the M-projection is inexact but, for some errors εt≥0\varepsilon\_{t}\geq 0, it holds that

|  |  |  |
| --- | --- | --- |
|  | DKL​(qt∥πt+1)≤minπ∈Π⁡DKL​(qt∥π)+εtwith∑t=0∞εt<∞D\_{\mathrm{KL}}(q\_{t}\|\pi\_{t+1})\;\leq\;\min\_{\pi\in\Pi}D\_{\mathrm{KL}}(q\_{t}\|\pi)\;+\;\varepsilon\_{t}\quad\text{with}\quad\sum\_{t=0}^{\infty}\varepsilon\_{t}<\infty |  |

then πt\pi\_{t} also converges to the same limit π†\pi^{\dagger}.

###### Proof.

The I-step is always an exact I-projection (Lemma A.1). In the case of an exact M-step, the iterative process is EM with information projections. The e-/m-flat geometry yields the Pythagorean identities implying convergence to π†\pi^{\dagger} (Dempster et al., [1977](#bib.bib15); Csiszár, [1984](#bib.bib13); Amari & Nagaoka, [2000](#bib.bib3)). When the M-step only ensures a (near-)minimization up to summable errors, the iteration is GEM: monotone improvement and convergence follow from the GEM theory of Wu ([1983](#bib.bib70)) together with generalized alternating minimization for Bregman divergences (Gunawardana et al., [2005](#bib.bib24)), which, under the same e-/m-flat assumptions, selects the same minimum-KL limit π†\pi^{\dagger}.
∎

#### Practical considerations.

Our theoretical equivalence should be interpreted with the following caveats:

* •

  Beyond REINFORCE. In practice, many policy gradient algorithms such as GRPO and PPO replace the raw reward R​(y)R(y) with an advantage estimate A​(y)A(y). Since this substitution is a control variate technique, it leaves the expected gradient direction unchanged while reducing its variance. Thus, our projection-based interpretation continues to hold.
* •

  The optimal policy set P∗P^{\*} defined by the linear constraint 𝔼q​[R]=1\mathbb{E}\_{q}[R]=1 is an mm-flat family, but the representable policy set Π\Pi induced by a neural network parametrization is not in general ee-flat. This may prevent exact convergence to the minimum-KL solution described above. Nevertheless, our theorem provides a principled explanation for the bias observed in practical RL algorithms.

## Appendix B Training and Evaluation Details

### B.1 LLM experiments

Unless otherwise stated, all reinforcement learning experiments were conducted using GRPO (Shao et al., [2024](#bib.bib62)).

For the Math reasoning task, the training set provided final answers but lacked reasoning chains required for SFT training. To obtain these, we queried DeepSeek R1 (Guo et al., [2025a](#bib.bib25)), sampling up to 16 responses per prompt and retaining a single response that matched the correct final answer. This yielded valid annotations for 96% of the dataset. For the Science Q&A task, we applied the same procedure with GPT-4o, obtaining correct annotations for the entire dataset.

To construct the learning–forgetting trade-off curves (e.g., Figure [2](#S3.F2 "Figure 2 ‣ 3 Reinforcement Learning Forgets Less than SFT ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less")), we followed the protocol below:

1. 1.

   Hyperparameter sweep. We trained multiple models under a broad sweep of hyperparameters (see Table [2](#A2.T2 "Table 2 ‣ B.1 LLM experiments ‣ Appendix B Training and Evaluation Details ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less")).
2. 2.

   New-task evaluation. For Math and Science Q&A, accuracy was measured by comparing the model’s final answer to the ground truth, ignoring intermediate reasoning chains. For Tool Use, we extracted API calls from the output and matched them against ground-truth calls via regular expressions.
3. 3.

   Previous-task evaluation. We assessed performance on unrelated benchmarks as described in Section [3.1](#S3.SS1 "3.1 Performance Trade-offs ‣ 3 Reinforcement Learning Forgets Less than SFT ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less"), using the Language Model Evaluation Harness (Gao et al., [2024](#bib.bib23)).
4. 4.

   Pareto filtering. From the trained models, we retained only those lying within 2 accuracy points of the Pareto frontier.
5. 5.

   Curve fitting. An exponential function was fit to the filtered points to produce the trade-off curves.

!(/html/2509.04259/assets/x6.png)

Figure 6: Example for the process of creating the pareto frontier plots

|  |  |  |
| --- | --- | --- |
| Hyperparameter | SFT / SIMPO | RL |
| Base Model | Qwen2.5 3B-Instruct | Qwen2.5 3B-Instruct |
| Learning Rate | {1e-5, 3e-5, 5e-5, 7e-5, 9e-5} | {1e-5, 2e-5, 3e-5, 4e-5, 5e-5} |
| Optimizer | adamw | adamw |
| LR Scheduler | {constant w. warmup, cosine w. warmup} | constant w. warmup |
| Warmup steps | 50 | 50 |
| Epochs | {1,2} | 1 |
| Batch Size | {16,32,64,128} | See Below |
| Max Grad Norm | 1 | 1 |
| bfloat16 | True | True |
| Weight Decay | 0 | 0 |
| GRPO-only hyperparameters | | |
| KL reg. |  | 0 |
| Group Size |  | 64 |
| Prompts per generation |  | 8 |
| num iterations (μ\mu) |  | {1,2} |
| Loss type |  | Dr. GRPO (Liu et al., [2025](#bib.bib46)) |

Table 2: Hyperparameters used for the LLM experiments. Curly braces {} indicate a sweep over the specified values. Additional parameters such as weight decay and max gradient norm were manually ablated; since they showed no significant effect on results, they were not included in the final sweep.]

### B.2 Robotic Experiments

We evaluated the RL–SFT forgetting gap in a robotic control setting using the OpenVLA-7B model (Kim et al., [2024](#bib.bib36)) as our base policy in the SimplerEnv environment (Li et al., [2024b](#bib.bib42)). The fine-tuning task was a pick-and-place scenario requiring the robot to grasp and lift a can, while forgetting was measured on a distinct manipulation task of drawer opening/closing. This setting complements our LLM results by probing whether the KL–forgetting relationship generalizes to embodied policies. To construct the pareto-frontier, we follow the same protocol as in the LLM experiments.

#### Data Collection.

Training data were collected by varying object placement over a 10×1010\times 10 grid of initial positions:
obj-init-x ∈[−0.35−0.12]\in[-0.35-0.12], obj-init-y ∈[−0.02,0.42]\in[-0.02,0.42].
For evaluation, we sampled 100 random object locations uniformly in this area.

#### Supervised Fine-Tuning (SFT).

For each grid point, we collected 10 successful trajectories using the RT-1 (Brohan et al., [2022](#bib.bib5)) model and filtered for successful trajectories. We trained models with batch sizes {16,32,64}\{16,32,64\} and learning rates {1×10−6,3×10−6,5×10−6,7×10−6,9×10−6,1×10−5,3×10−5}\{1\!\times\!10^{-6},3\!\times\!10^{-6},5\!\times\!10^{-6},7\!\times\!10^{-6},9\!\times\!10^{-6},1\!\times\!10^{-5},3\!\times\!10^{-5}\}. Other hyperparameters were: AdamW optimizer, 11 training epoch, max gradient norm of 11, weight decay of 0, warmup of 1010 steps, constant-with-warmup scheduler, and bfloat16 precision.

#### Reinforcement Learning (RL).

For RL, we trained using REINFORCE with an reward normalization baseline, without explicit KL regularization. At each iteration, 5 trajectories were collected per grid point. Rewards were binary success indicators of task completion. RL training used the same training config as SFT.

### B.3 MNIST Experiments

All MNIST experiments were conducted using a 3-layer MLP with input dimension 785785, hidden layers of sizes 512512 and 256256, and output dimension 1010. The input consisted of a flattened 28×2828\times 28 image concatenated with a binary indicator: +1+1 for ParityMNIST and −1-1 for FashionMNIST.

#### Pretraining.

We pretrained the network jointly on ParityMNIST and FashionMNIST using small subsets of the original datasets (500 images from each). For ParityMNIST, the label was chosen uniformly at random among all digit labels with the correct parity.

#### Fine-tuning methods.

In our experiments, we evaluated five fine-tuning strategies:

1. 1.

   GRPO.
2. 2.

   GRPO + KL regularization with coefficient 0.10.1.
3. 3.

   SFT 1: all even digits mapped to label 0, all odd digits to label 1.
4. 4.

   SFT 2: even digits randomly mapped to {0,4}\{0,4\}, odd digits to {1,5}\{1,5\}.
5. 5.

   SFT with oracle distribution: annotations drawn from the minimum-KL distribution consistent with task correctness.

#### Oracle distribution.

Motivated by the KL–forgetting connection, we define the oracle distribution as the one that achieves perfect task accuracy while remaining closest (in KL divergence) to the pretraining distribution π0\pi\_{0}. Concretely, for an input image xx we compute π0(⋅|x)∈ℝ10\pi\_{0}(\cdot|x)\in\mathbb{R}^{10} and the binary indicator vector R∈{0,1}10R\in\{0,1\}^{10} encoding which labels are correct given the digit’s parity. The oracle distribution q∗q^{\*} is the solution to:

|  |  |  |
| --- | --- | --- |
|  | q∗=arg⁡minq⁡DKL​(π0∥q)s.t.q⊤​R=1.q^{\*}=\arg\min\_{q}D\_{\mathrm{KL}}(\pi\_{0}\|q)\quad\text{s.t.}\quad q^{\top}R=1. |  |

Since KL is convex and the constraint is linear, we can calculate a closed-form solution for every image. We then sample from q∗q^{\*} to produce SFT annotations.

#### Hyperparameter sweep.

For each method we trained models across a sweep of 15 learning rates logarithmically spaced between 3​e−63e-6 and 1​e−31e-3, using either a constant-with-warmup or cosine-with-warmup scheduler, and training for 1 or 2 epochs. Including mid-training checkpoints, this produced approximately 500 runs per method.

### B.4 Centered Kernel Alignmen

#### Centered Kernel Alignment (CKA) (Kornblith et al., [2019](#bib.bib38))

Given representations X,Y∈ℝn×dX,Y\in\mathbb{R}^{n\times d}, define kernels
K=X​X⊤K=XX^{\top}, L=Y​Y⊤L=YY^{\top}.
Let H=I−1n​𝟏𝟏⊤H=I-\tfrac{1}{n}\mathbf{1}\mathbf{1}^{\top} be the centering matrix.
The centered kernels are

|  |  |  |
| --- | --- | --- |
|  | K¯=H​K​H,L¯=H​L​H.\bar{K}=HKH,\quad\bar{L}=HLH. |  |

CKA is then computed as

|  |  |  |
| --- | --- | --- |
|  | CKA​(K,L)=⟨K¯,L¯⟩F‖K¯‖F​‖L¯‖F,\mathrm{CKA}(K,L)\;=\;\frac{\langle\bar{K},\bar{L}\rangle\_{F}}{\|\bar{K}\|\_{F}\,\|\bar{L}\|\_{F}}, |  |

where ⟨A,B⟩F=tr​(A⊤​B)\langle A,B\rangle\_{F}=\mathrm{tr}(A^{\top}B).

#### CKA with kk-NN Alignment (CKNNA) (Huh et al., [2024](#bib.bib34))

Let α​(i,j)∈{0,1}\alpha(i,j)\in\{0,1\} indicate whether i,ji,j are
mutual kk-nearest neighbors in both XX and YY.
Define the masked inner product

|  |  |  |
| --- | --- | --- |
|  | ⟨A,B⟩α=∑i=1n∑j=1nα​(i,j)​Ai​j​Bi​j.\langle A,B\rangle\_{\alpha}=\sum\_{i=1}^{n}\sum\_{j=1}^{n}\alpha(i,j)\,A\_{ij}B\_{ij}. |  |

CKNNA is then given by

|  |  |  |
| --- | --- | --- |
|  | CKNNA​(K,L)=⟨K¯,L¯⟩α⟨K¯,K¯⟩α​⟨L¯,L¯⟩α.\mathrm{CKNNA}(K,L)\;=\;\frac{\langle\bar{K},\bar{L}\rangle\_{\alpha}}{\sqrt{\langle\bar{K},\bar{K}\rangle\_{\alpha}\,\langle\bar{L},\bar{L}\rangle\_{\alpha}}}. |  |

When α​(i,j)=1\alpha(i,j)=1 for all i≠ji\neq j, CKNNA reduces to standard CKA.

## Appendix C Additional Results

### C.1 Representation Preservation

While benchmark accuracy provides an external measure of forgetting, it may conflate genuine loss of capability with superficial effects such as formatting mismatch between tasks.
To assess whether fine-tuning alters the model more fundamentally, we analyzed changes to the model’s representations.

#### Experimental Setup.

To study how representations change between models, we compare their embeddings on a shared dataset.
Following prior work, we compare the relative geometry of the embeddings—that is, how different inputs relate to each other. This geometry can be summarized by a kernel (similarity) matrix, which encodes pairwise relationships among input embeddings.
Centered Kernel Alignment (CKA) (Kornblith et al., [2019](#bib.bib38)) is a standard measure for comparing such kernels, providing a way to quantify representational similarity between models.

!(/html/2509.04259/assets/x7.png)

Figure 7: CKA similarity to the base model during training. Although SFT and RL achieve comparable task performance, SFT models diverge substantially in their representations, whereas RL models remain more closely aligned with the base model.

For this analysis, we constructed kernels from random Wikipedia paragraphs, ensuring that the probe data are unrelated to the fine-tuning tasks. We then compared the kernels of the base model and its fine-tuned variants using CKNNA (Huh et al., [2024](#bib.bib34)), a local-neighborhood variant of CKA (see Appendix [B.4](#A2.SS4 "B.4 Centered Kernel Alignmen ‣ Appendix B Training and Evaluation Details ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less") for details). Comparisons were made between SFT and RL models that achieved similar final accuracy on the new task, isolating representational differences due to training method rather than task performance.

#### Results.

Figure [7](#A3.F7 "Figure 7 ‣ Experimental Setup. ‣ C.1 Representation Preservation ‣ Appendix C Additional Results ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less") shows that RL-trained models retain high representational similarity (CKNNA=0.94) to the base model, with CKNNA scores remaining close to one even after fine-tuning on the new task. In contrast, SFT-trained models exhibit substantial representational drift (CKNNA=0.56). These results indicate that RL fine-tuning integrates new abilities while leaving the overall representation space largely intact, whereas SFT alters the geometry more extensively. Together with the benchmark results, this suggests that RL is able to integrate new abilities without disturbing the underlying representational structure, while SFT incurs representational shifts that manifest as catastrophic forgetting.

### C.2 Scaling and Forgetting

Prior work has suggested that catastrophic forgetting diminishes as model size increases (Ramasesh et al., [2021](#bib.bib57); Luo et al., [2023](#bib.bib47); Cossu et al., [2024](#bib.bib12)). To evaluate this claim in our setting, we repeated the SFT experiments from Section [3](#S3 "3 Reinforcement Learning Forgets Less than SFT ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less") using Qwen 2.5 models with 3B, 7B, and 14B parameters on the Science Q&A task.

!(/html/2509.04259/assets/x8.png)

Figure 8: Pareto frontiers for SFT on Qwen 2.5 Instruct models of size 3B, 7B, and 14B on the Science Q&A task. All sizes exhibit the same fundamental trade-off—gains on the new task require forgetting prior capabilities.

The results, shown in Figure [8](#A3.F8 "Figure 8 ‣ C.2 Scaling and Forgetting ‣ Appendix C Additional Results ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less"), demonstrate that although larger models start with better general capabilities, the trade-off between new-task performance and prior-task retention remains unchanged: across all model sizes, SFT improves new-task accuracy at the expense of forgetting. In particular, to reach high accuracy on the Science Q&A task, substantial degradation occurs in performance on prior benchmarks regardless of model scale.

### C.3 Optimization Dynamics

To examine the link between parameter updates and forgetting, we analyzed the optimization trajectory at the level of individual training steps. For each update, we computed two quantities:

1. 1.

   Forgetting direction. Using the FashionMNIST evaluation set, we calculated the gradient of the loss with respect to model parameters. We then measured the cosine similarity between this gradient and the actual parameter update from the training step. A positive cosine indicates that the update increases FashionMNIST loss (catastrophic forgetting), while a negative cosine indicates an update that reduces it.
2. 2.

   KL shift. We measured the change in KL divergence between the model’s output distributions on the ParityMNIST test set before and after the update.

Plotting per-step KL change against the cosine similarity (Figure [10](#A3.F10 "Figure 10 ‣ C.3 Optimization Dynamics ‣ Appendix C Additional Results ‣ RL’s Razor: Why Online Reinforcement Learning Forgets Less")) revealed a strong correlation: steps producing larger KL shifts tended to align more with the forgetting gradient. This analysis demonstrates that at the level of optimization dynamics, catastrophic forgetting is driven by updates that induce larger distributional shifts on the new task.

!(/html/2509.04259/assets/x9.png)

Figure 9: SFT distillation from an RL teacher.
Accuracy trade-off between the new task (MNIST) and the prior task (FashionMNIST). Sweeping student hyperparameters shows that SFT can match the teacher within noise on both tasks. This suggests that what matters is not the optimization path, but the distribution of the final model.

!(/html/2509.04259/assets/x10.png)

!(/html/2509.04259/assets/x11.png)

Figure 10: Gradient similarity versus KL change.
(Left) On the new training task (ParityMNIST), gradient cosine similarity and KL change per step remain anti-correlated.
(Right) On the prior task (FashionMNIST), the gradient similarity is more correlated with the KL change per step on the training task (ParityMNIST).
Together, these plots show that taking a larger step on the current task induces gradients that are more similar in direction to the

!(/html/2509.04259/assets/x12.png)

Figure 11: We plot the KL divergence between the base and fine-tuned model on the new task, alongside the corresponding forgetting performance across methods.
