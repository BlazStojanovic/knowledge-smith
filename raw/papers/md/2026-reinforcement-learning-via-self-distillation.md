---
arxiv: '2601.20802'
authors:
- Jonas Hübotter
- Frederike Lübeck
- Lejs Behric
- Anton Baumann
- Marco Bagatella
- Daniel Marta
- Ido Hakimi
- Idan Shenfeld
- Thomas Kleine Buening
- Carlos Guestrin
- Andreas Krause
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Reinforcement Learning via Self-Distillation
url: https://arxiv.org/abs/2601.20802
year: 2026
---

# Reinforcement Learning via Self-Distillation

Jonas Hübotter11   Frederike Lübeck  ,1,2,1,2   Lejs Behric111In standard RLVR implementations a rollout group contains multiple simultaneous attempts for xx.  ,1,1   Anton Baumann111In standard RLVR implementations a rollout group contains multiple simultaneous attempts for xx.  ,1,1
  
Marco Bagatella1,21,2   Daniel Marta11   Ido Hakimi11   Idan Shenfeld33
  
Thomas Kleine Buening11   Carlos Guestrin44   Andreas Krause11
  
11ETH Zurich   22Max Planck Institute for Intelligent Systems   33MIT   44Stanford
  
<https://github.com/lasgroup/SDPO>
Equal second authorship. Correspondence to jonas.huebotter@inf.ethz.ch.

###### Abstract

Large language models are increasingly post-trained with reinforcement learning in verifiable domains such as code and math.
Yet, current methods for reinforcement learning with verifiable rewards (RLVR) learn only from a scalar outcome reward per attempt, creating a severe credit-assignment bottleneck.
Many verifiable environments actually provide rich textual feedback, such as runtime errors or judge evaluations, that explain *why* an attempt failed.
We formalize this setting as reinforcement learning with rich feedback and introduce Self-Distillation Policy Optimization (SDPO), which converts tokenized feedback into a dense learning signal without any external teacher or explicit reward model.
SDPO treats the current model conditioned on feedback as a self-teacher and distills its feedback-informed next-token predictions back into the policy.
In this way, SDPO leverages the model’s ability to retrospectively identify its own mistakes in-context.
Across scientific reasoning, tool use, and competitive programming on LiveCodeBench v6, SDPO improves sample efficiency and final accuracy over strong RLVR baselines.
Notably, SDPO also outperforms baselines in standard RLVR environments that only return scalar feedback by using successful rollouts as implicit feedback for failed attempts.
Finally, applying SDPO to individual questions at test time accelerates discovery on difficult binary-reward tasks, achieving the same discovery probability as best-of-kk sampling or multi-turn conversations with 3×3\times fewer attempts.

## 1 Introduction

!(/html/2601.20802/assets/x1.png)

Figure 1: 
SDPO substantially outperforms an improved version of Group Relative Policy Optimization (GRPO) on LCB v6 with Qwen3-8B. Further, SDPO achieves GRPO’s final accuracy in 4×4\times fewer generations.
Claude Sonnet 4 is the strongest instruct model on the public LCBv6 leaderboard.
Shaded regions show the standard deviation across 3 seeds.

Progress in deep reinforcement learning has shown that iterating on experience—acting, receiving feedback, and updating a policy—can unlock capabilities that are difficult to obtain from static supervision alone (Mnih et al., [2015](#bib.bib64); Silver et al., [2016](#bib.bib86); [2017](#bib.bib87); Berner et al., [2019](#bib.bib7)).
The same theme now appears in large language models (LLMs): large-scale post-training with reinforcement learning (RL) has substantially improved performance on reasoning-heavy tasks, especially in settings with programmatic or otherwise verifiable evaluation (Jaech et al., [2024](#bib.bib41); Guo et al., [2025](#bib.bib32); Kimi et al., [2025](#bib.bib47); Olmo et al., [2025](#bib.bib68)).

Nevertheless, the dominant RL recipe for LLM post-training remains bottlenecked by credit assignment.
Most current approaches operate in the setting of reinforcement learning with verifiable rewards (RLVR): given a question xx, the model samples an answer y∼πθ(⋅∣x){y\sim\pi\_{\theta}(\cdot\mid x)} and receives a scalar reward r∈ℝr\in\mathbb{R}, often binary (e.g., unit-tests pass/fail in code generation).
Modern policy gradient RLVR methods such as Group Relative Policy Optimization (GRPO; Shao et al., [2024](#bib.bib82)) estimate advantages from these sparse outcome rewards.
Furthermore, when all rollouts in a group receive the same (often zero) reward, GRPO advantages collapse to zero and learning stalls.
To overcome this sparsity, one might prefer distillation from a strong teacher (Guo et al., [2025](#bib.bib32); Yang et al., [2025](#bib.bib106); Lu & Thinking Machines Lab, [2025](#bib.bib60); Guha et al., [2026](#bib.bib31)), which provides dense, token-level supervision.
However, strong teachers are often unavailable in online learning, where the goal is to raise the capability ceiling beyond existing models.

In this work, we argue that the key limitation is not RL per se, but the information bottleneck imposed by scalar outcome rewards.
Many verifiable environments expose *rich tokenized feedback* beyond scalar rewards rr, such as runtime errors, failing unit tests, or evaluations from an LLM judge.
This feedback not only reveals whether a rollout was wrong, but also what went wrong.
We formalize this more general setting as Reinforcement Learning with Rich Feedback (RLRF) and illustrate its difference to RLVR in [Figure˜2](#S1.F2 "In 1 Introduction ‣ Reinforcement Learning via Self-Distillation").
Here, feedback can be any tokenized representation of any state reached by an agentic system.
The central question becomes: how can we convert rich feedback into effective credit assignment without requiring external supervision from a strong teacher?

!(/html/2601.20802/assets/x2.png)

Figure 2: 
Comparison of RLVR and RLRF settings.
In Reinforcement Learning with Verifiable Rewards (RLVR), the agent learns from a scalar reward rr, which often acts as an information bottleneck by masking the underlying environment state.
In contrast, Reinforcement Learning with Rich Feedback (RLRF) utilizes tokenized feedback.
This provides a significantly richer signal than a scalar reward, as the feedback can encapsulate both the reward as well as detailed observations of the state (such as runtime errors from a code environment or feedback from an LLM judge).

[⬇](data:text/plain;base64,ClJ1bnRpbWUgRXJyb3IKWmVyb0RpdmlzaW9uRXJyb3I6IGRpdmlzaW9uIGJ5IHplcm8KTGluZSA3MyBpbiBzZXBhcmF0ZVNxdWFyZXMgKFNvbHV0aW9uLnB5KQpccGFyTGFzdCBFeGVjdXRlZCBJbnB1dApbWzI2LDMwLDJdLFsxMSwyMywxXV0K)

Runtime Error
ZeroDivisionError: division by zero
Line 73 in separateSquares (Solution.py)
\parLast Executed Input
[[26,30,2],[11,23,1]]

Figure 3: Example of feedback from our code environment, inspired by LeetCode. Listings LABEL:lst:feedback\_example\_wrong\_answer, LABEL:lst:memory\_error, and LABEL:lst:index\_error in the appendix show examples of feedback in case of a wrong answer, a memory error, and an index error.

Our starting point is the observation that LLMs already possess a powerful mechanism for using feedback: in-context learning (Brown et al., [2020](#bib.bib9); Wei et al., [2022](#bib.bib102)).
When conditioned on feedback, the same model can often identify plausible mistakes and propose a corrected approach.
A common example of such feedback is the summary of failed test cases on coding platforms like LeetCode ([Figure˜3](#S1.F3 "In 1 Introduction ‣ Reinforcement Learning via Self-Distillation")).
Many recent works leverage this capability to iteratively generate corrections (Chen et al., [2021a](#bib.bib16); Madaan et al., [2023](#bib.bib62); Shinn et al., [2023](#bib.bib85); Yao et al., [2024](#bib.bib108); Yuksekgonul et al., [2025](#bib.bib111); Lee et al., [2025](#bib.bib53)).
In contrast, we use the current policy as a “self-teacher” that, rather than sampling a new response, re-evaluates the *existing* rollout after receiving rich feedback.
Including the feedback in-context transforms the model’s next-token distribution, allowing the self-teacher to agree or disagree with the student’s original choices at specific tokens.
This yields dense, logit-level credit assignment.
For example, when provided with the feedback from [Figure˜3](#S1.F3 "In 1 Introduction ‣ Reinforcement Learning via Self-Distillation"), the self-teacher can identify how the initial attempt should be modified to avoid the runtime error.
Crucially, this mechanism incurs no sampling overhead: we simply re-compute the log-probabilities of the original attempt under the self-teacher’s feedback-augmented context.

Building on this idea, we introduce Self-Distillation Policy Optimization (SDPO), an on-policy algorithm that performs RL via self-distillation.
SDPO samples rollouts from the current policy, obtains rich environment feedback, and then minimizes a logit-level distillation loss that matches the current policy’s next-token distribution to that of the self-teacher.
Conceptually, SDPO addresses the central limitation of applying distillation to online learning: the absence of a stronger external teacher.
Instead of relying on a fixed teacher, SDPO leverages the model’s ability to recognize its own mistakes in hindsight.
By conditioning the current policy on the rich feedback it just received, we construct a self-teacher that provides the dense supervision of distillation while retaining the exploration benefits of on-policy RL.
[Table˜1](#S1.T1 "In 1 Introduction ‣ Reinforcement Learning via Self-Distillation") summarizes how this positions SDPO relative to RLVR and distillation baselines.
We include a comprehensive summary of related work in [Section˜6](#S6 "6 Related Work ‣ Reinforcement Learning via Self-Distillation").

We show that SDPO is a policy gradient algorithm whose advantages are estimated using the self-teacher.
This enables the implementation of SDPO with minor changes to standard RLVR pipelines, simply by swapping out the advantages.

|  |  |  |  |
| --- | --- | --- | --- |
| Method | Sampling | Signal | Feedback |
| SFT / Distillation (Hinton et al., [2015](#bib.bib36)) | ×\boldsymbol{\times} off-policy | ✓ rich | ×\boldsymbol{\times} strong teacher |
| On-Policy Distillation (Agarwal et al., [2024](#bib.bib1)) | ✓ on-policy | ✓ rich | ×\boldsymbol{\times} strong teacher |
| RLVR (such as GRPO) (Lambert et al., [2025](#bib.bib50)) | ✓ on-policy | ×\boldsymbol{\times} weak | ✓ environment |
| RL via Self-Distillation (SDPO) (ours) | ✓ on-policy | ✓ rich | ✓ environment |

Table 1: Comparison of self-distillation to alternative methods for post-training LLMs.

##### Summary of evaluation results.

We evaluate SDPO in three online RL settings:

* •

  Learning without rich feedback (§[3](#S3 "3 Learning without Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation")):
  We evaluate standard RLVR environments that do not return any feedback beyond scalar rewards.
  Here, SDPO treats successful attempts sampled in the current batch as “feedback” for failed attempts on the same question.
  We perform training runs on scientific reasoning and tool use, starting with Qwen3-8B and Olmo3-7B-Instruct.
  We find that SDPO outperforms a strong GRPO baseline that integrates recent improvements: 68.8% vs. 64.1% final accuracy on aggregate.
  SDPO achieves higher accuracy with up to 7×7\times shorter generation lengths compared to GRPO, demonstrating that effective reasoning need not be verbose.
* •

  Learning with rich feedback (§[4](#S4 "4 Learning with Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation")):
  We evaluate competitive programming problems from LiveCodeBench v6 with LeetCode-style feedback.
  As shown in [Figure˜1](#S1.F1 "In 1 Introduction ‣ Reinforcement Learning via Self-Distillation"), SDPO substantially improves over GRPO, reaching a higher final accuracy (48.8% vs. 41.2%) and achieving GRPO’s final accuracy in 4×4\times fewer generations.
  SDPO’s gains grow with model scale, suggesting that the ability for self-teaching emerges as models become stronger in-context learners.
* •

  Discovering novel solutions to hard tasks at test-time (§[5](#S5 "5 Solving Hard Questions via Test-Time Self-Distillation ‣ Reinforcement Learning via Self-Distillation")):
  Finally, we demonstrate that SDPO can accelerate the discovery of solutions to difficult binary-reward questions.
  This contrasts with RLVR methods, which only begin learning once the first solution has been found.
  We leverage SDPO for Test-Time Self-Distillation, a form of test-time training where the model specializes to an individual test question.
  We consider very difficult LiveCodeBench questions, for which the base model’s pass@6464 is below 0.03, and show that SDPO accelerates the discovery of solutions by 3×3\times.

## 2 SDPO: Self-Distillation Policy Optimization

We propose an algorithm that uses the in-context learning ability of the current policy for assigning credit.
Our key object is the *self-teacher*, πθ(⋅∣x,f)\pi\_{\theta}(\cdot\mid x,f), which refers to the current policy (the “student”) prompted with the question xx and the rich feedback ff.
Next to the students’ original attempt yy, ff may incorporate two key kinds of feedback: any environment output (such as runtime errors from a code environment) and a sample solution if xx was already solved with another attempt in the rollout group.111In standard RLVR implementations a rollout group contains multiple simultaneous attempts for xx.
As discussed before, the self-teacher πθ(⋅∣x,f)\pi\_{\theta}(\cdot\mid x,f) should have a higher accuracy than the student πθ(⋅∣x)\pi\_{\theta}(\cdot\mid x) since it sees additional information in-context.
This leads us to observe:

We can use the same policy in two different roles: As the student for the initial attempt and as the teacher to determine the value of actions in hindsight.

We introduce Self-Distillation Policy Optimization (SDPO) which repeatedly distills the self-teacher into the student.
Given a question xx, we first sample rollouts from the student πθ\pi\_{\theta} and obtain corresponding environment feedback.
We then use the KL-divergence, KL​(p∥q)=∑ip​(i)​log⁡p​(i)/q​(i)\smash{\mathrm{KL}\left(p\|q\right)=\sum\_{i}p(i)\log\nicefrac{{p(i)}}{{q(i)}}}, as a distance measure for the next-token distributions of student and teacher, and optimize a standard logit distillation loss:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒSDPO(θ):=∑tKL(πθ(⋅∣x,y<t)∥stopgrad(πθ(⋅∣x,f,y<t)))\mathcal{L}\_{\mathrm{SDPO}}(\theta):=\sum\_{t}\mathrm{KL}(\pi\_{\theta}(\cdot\mid x,y\_{<t})\|\mathrm{stopgrad}(\pi\_{\theta}(\cdot\mid x,f,y\_{<t}))) |  | (1) |

Algorithm 1  SDPO

1:Language model πθ\pi\_{\theta}; dataset with questions xx; number of rollouts GG per question; environment to obtain feedback for attempts.

2:repeat

3:  Sample question xx from dataset.

4:  Sample responses: {yi}i=1G∼πθ(⋅∣x)\smash{\{y\_{i}\}\_{i=1}^{G}\sim\pi\_{\theta}(\cdot\mid x)}.

5:  Evaluate responses to obtain feedback fif\_{i}.
⊳\triangleright Self-distillation:

6:  Compute log-probs of self-teacher 

|  |  |  |
| --- | --- | --- |
|  | log⁡πθ​(yi,t∣x,fi,yi,<t).\log\pi\_{\theta}(y\_{i,t}\mid x,f\_{i},y\_{i,<t}). |  |

7:  Update θ\theta with gradient descent on ℒSDPO​(θ)\mathcal{L}\_{\mathrm{SDPO}}(\theta).

8:until converged

where the stopgrad operator blocks gradients from flowing through the teacher, and thus prevents it from regressing towards the student and ignoring ff.
The intuitive role of the teacher is to determine where and how the students’ original attempt yy was wrong through retrospection based on the feedback ff.
[Figure˜4](#S2.F4 "In 2 SDPO: Self-Distillation Policy Optimization ‣ Reinforcement Learning via Self-Distillation") shows an example of self-teaching with Qwen3-8B as student and self-teacher.
We summarize SDPO in [Algorithm˜1](#alg1 "In 2 SDPO: Self-Distillation Policy Optimization ‣ Reinforcement Learning via Self-Distillation") and display the teachers’ reprompt template in [Table˜2](#S2.T2 "In 2 SDPO: Self-Distillation Policy Optimization ‣ Reinforcement Learning via Self-Distillation").

!(/html/2601.20802/assets/x3.png)

Figure 4: Example of self-teaching with Qwen3-8B. The answer is generated by the model before seeing the feedback. Then, we re-evaluate the log-probs of the original attempt with the *self-teacher* after seeing the feedback. We show the per-token log⁡(ℙ​(self-teacher)/ℙ​(student))\log(\nicefrac{{\mathbb{P}\left(\text{self-teacher}\right)}}{{\mathbb{P}\left(\text{student}\right)}}), with red indicating negative values (self-teacher disagrees)
and white indicating values around zero. Notably, in this example, Qwen3-8B identifies the error through retrospection without an explicit solution. Further, the activation is sparse, identifying where mistakes happen and adjusting to the students’ response distribution.

|  |  |
| --- | --- |
| User: | prompt  Correct solution:  successful\_previous\_rollout  The following is feedback from your unsuccessful earlier attempt:  environment\_output  Correctly solve the original question. |
| Assistant: | original\_response |

Table 2: Template for self-teacher. prompt is replaced with the question. A sample solution previously generated by the student is substituted for successful\_previous\_rollout (if available for this question; otherwise the paragraph is skipped). environment\_output is replaced with the environment output (see, e.g., [Figure˜3](#S1.F3 "In 1 Introduction ‣ Reinforcement Learning via Self-Distillation")) from the models’ original attempt (if it was not successful and there is no solution; otherwise the paragraph is skipped). If the models’ original attempt was successful, this attempt is passed as the correct solution. original\_response is replaced with the models’ original attempt to re-evaluate its log-probabilities under the self-teacher.

We can derive the SDPO gradient as follows (see [Section˜B.1](#A2.SS1 "B.1 Gradient Estimator ‣ Appendix B Theoretical Analysis ‣ Reinforcement Learning via Self-Distillation") for details):

###### Proposition 2.1.

Let 𝒱\mathcal{V} be the set of tokens in the vocabulary.
The gradient of ℒSDPO\mathcal{L}\_{\mathrm{SDPO}} is

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇θℒSDPO​(θ)=𝔼y∼πθ(⋅∣x)​[∑t=1|y|∑y^t∈𝒱∇θlog⁡πθ​(y^t∣x,y<t)⋅log⁡πθ​(y^t∣x,y<t)πθ​(y^t∣x,f,y<t)].\boldsymbol{\nabla}\_{\!\!\theta}\,\mathcal{L}\_{\mathrm{SDPO}}(\theta)=\mathbb{E}\_{y\sim\pi\_{\theta}(\cdot\mid x)}\left[\sum\_{t=1}^{|y|}\sum\_{\hat{y}\_{t}\in\mathcal{V}}\boldsymbol{\nabla}\_{\!\!\theta}\,\log\pi\_{\theta}(\hat{y}\_{t}\mid x,y\_{<t})\cdot\log\frac{\pi\_{\theta}(\hat{y}\_{t}\mid x,y\_{<t})}{\pi\_{\theta}(\hat{y}\_{t}\mid x,f,y\_{<t})}\right].\vskip 4.30554pt |  | (2) |

### 2.1 Comparison to RLVR

Note that the SDPO gradient is a (negated) logit-level policy gradient where the advantages are estimated using the self-teacher.222See [Section˜A.3](#A1.SS3 "A.3 Off-Policy Training: Generalization to Logit-Level Losses ‣ Appendix A Implementation of SDPO ‣ Reinforcement Learning via Self-Distillation") for a detailed comparison of the SDPO gradient to the standard policy gradient.
We can therefore reuse standard RLVR implementations and simply swap out the advantages.
Let yiy\_{i} be the ii-th rollout from a rollout group of size GG for question xx, then we have:

|  |  |  |
| --- | --- | --- |
|  | Ai,tGRPO​(y^i,t):=𝟙​{yi,t=y^i,t}​(ri−mean​{ri}i=1G),Ai,tSDPO​(y^i,t)=log⁡πθ​(y^i,t∣x,fi,yi,<t)πθ​(y^i,t∣x,yi,<t).A\_{i,t}^{\mathrm{GRPO}}(\hat{y}\_{i,t}):=\mathbbm{1}\{{y\_{i,t}=\hat{y}\_{i,t}}\}\left(r\_{i}-\mathrm{mean}\{r\_{i}\}\_{i=1}^{G}\right),\quad A\_{i,t}^{\mathrm{SDPO}}(\hat{y}\_{i,t})=\log\frac{\pi\_{\theta}(\hat{y}\_{i,t}\mid x,f\_{i},y\_{i,<t})}{\pi\_{\theta}(\hat{y}\_{i,t}\mid x,y\_{i,<t})}. |  |

The GRPO advantages are zero on any non-generated token and constant within a rollout yiy\_{i}.333We use the GRPO (Shao et al., [2024](#bib.bib82)) advantage without normalization (Liu et al., [2025b](#bib.bib59)).
In contrast, the SDPO advantages are zero only for tokens where student and teacher perfectly agree.
The SDPO advantage is positive for tokens which are more likely under the teacher while being negative for tokens which are less likely under the teacher.
Thus, SDPO can be seen as a direct extension of standard RLVR methods in two ways:

1. 1.

   from 1-bit feedback to *allowing arbitrary sequences of tokens as feedback*, and
2. 2.

   leveraging this rich feedback to *estimate dense logit-level advantages*.

This tight connection to RLVR methods also enables a straightforward extension of the SDPO gradient from [Equation˜2](#S2.E2 "In Proposition 2.1. ‣ 2 SDPO: Self-Distillation Policy Optimization ‣ Reinforcement Learning via Self-Distillation") to off-policy data via PPO-style clipped importance sampling (Schulman et al., [2017](#bib.bib80)), see [Section˜A.3](#A1.SS3 "A.3 Off-Policy Training: Generalization to Logit-Level Losses ‣ Appendix A Implementation of SDPO ‣ Reinforcement Learning via Self-Distillation").

### 2.2 Compute time & memory

!(/html/2601.20802/assets/x4.png)

Figure 5: Time per step for SDPO vs GRPO (solid: without code environment, light: with code environment).

The only computational overhead of SDPO compared to GRPO is the additional computation of log-probs from the self-teacher, which can be effectively parallelized and is substantially faster than sequential generation.
[Figure˜5](#S2.F5 "In 2.2 Compute time & memory ‣ 2 SDPO: Self-Distillation Policy Optimization ‣ Reinforcement Learning via Self-Distillation") compares the compute time of SDPO and GRPO.
As expected, the compute overhead of SDPO is relatively small.
Here, we use a micro batch size of 2;444The micro batch size corresponds to # rollouts we train on at a time while accumulating gradients. compute time can be further reduced by using larger micro batch sizes.

Naively computing the KL divergence between student and teacher requires holding full logits of both models in memory.
To avoid this, we approximate the KL divergence in the SDPO loss by performing top-KK distillation (i.e., only computing the top-KK logits of the student and the corresponding logits of the teacher alongside a term capturing the tail probability; cf. [Section˜A.2](#A1.SS2 "A.2 Approximate Logit Distillation ‣ Appendix A Implementation of SDPO ‣ Reinforcement Learning via Self-Distillation")). With a reasonable choice of KK (e.g., K=100{K=100}), this avoids virtually any memory overhead while capturing most of the information.

### 2.3 Stability improvements

We find that two practical modifications significantly enhance the training stability of SDPO.
First, we employ a regularized self-teacher, implemented either via an exponential moving average (EMA) of the student parameters or by interpolating the current teacher with the initial teacher (cf. [Section˜A.1](#A1.SS1 "A.1 Regularized teacher ‣ Appendix A Implementation of SDPO ‣ Reinforcement Learning via Self-Distillation")).
As detailed later, both strategies effectively stabilize learning.
Second, we adopt the symmetric Jensen-Shannon divergence for the distillation loss; this formulation has similarly been shown to improve stability in on-policy distillation from external teachers (Agarwal et al., [2024](#bib.bib1)).

## 3 Learning without Rich Environment Feedback

We first evaluate SDPO in standard RLVR environments, where feedback is limited to scalar rewards.
Instead of using the scalar reward, SDPO treats successful attempts sampled in the current batch as “feedback” for failed attempts on the same question.
By comparing the student’s attempt with a correct solution, the self-teacher can identify where the student was wrong and provide dense credit assignment.

!(/html/2601.20802/assets/x5.png)

Figure 6: Training progression of Olmo3-7B-Instruct on Chemistry. We report the average accuracy across 16 samples per question and a rolling average of response lengths over 5 steps. We report GRPO with the optimal hyperparameters for this model and task.

### 3.1 Experimental setting

We evaluate tasks on which the model has not been explicitly fine-tuned:

* •

  Science Q&A (Chemistry, Physics, Biology, Materials science): Undergraduate-level scientific reasoning using reasoning subsets (L3) from SciKnowEval (Feng et al., [2024](#bib.bib27)).
* •

  Tool use: Mapping a tool-API specification and user request to the correct tool call, using ToolAlpaca (Tang et al., [2023](#bib.bib94)).

We perform a train-test split to test in-domain generalization.
We use Qwen3-8B (Yang et al., [2025](#bib.bib106)) and Olmo3-7B-Instruct (Olmo et al., [2025](#bib.bib68)) as initial checkpoints and report avg@16 relative to wall-clock training time, excluding initialization & validation.

##### Baselines.

We compare SDPO to an improved variant of GRPO (Shao et al., [2024](#bib.bib82)), which incorporates several recent modifications (Olmo et al., [2025](#bib.bib68); Khatri et al., [2026](#bib.bib45)) such as asymmetric clipping (Yu et al., [2025](#bib.bib109)), avoiding biased normalization (Liu et al., [2025b](#bib.bib59)), and correcting for off-policy data when using efficient inference frameworks (Yao et al., [2025](#bib.bib107)).
We integrate these modifications into a GRPO implementation that represents a strong baseline, as detailed in [Equation˜10](#A1.E10 "In A.3 Off-Policy Training: Generalization to Logit-Level Losses ‣ Appendix A Implementation of SDPO ‣ Reinforcement Learning via Self-Distillation") in [Section˜A.3](#A1.SS3 "A.3 Off-Policy Training: Generalization to Logit-Level Losses ‣ Appendix A Implementation of SDPO ‣ Reinforcement Learning via Self-Distillation").
GRPO enables off-policy training through PPO’s clipped importance weighting (Schulman et al., [2017](#bib.bib80)).
We additionally report the special case of on-policy GRPO (matching the hyperparameters of vanilla SDPO).
For both baselines, we perform a hyperparameter sweep and report results for the models that achieve the highest validation performance across all target tasks.
Hyperparameters and training details are provided in [Appendix˜E](#A5 "Appendix E Experiment Details ‣ Reinforcement Learning via Self-Distillation").
We use the verl library (Sheng et al., [2025](#bib.bib84)) for fast multi-GPU training.

### 3.2 Results

[Table˜3](#S3.T3 "In 3.2 Results ‣ 3 Learning without Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation") summarizes our results.
We find that SDPO outperforms GRPO across almost all runs, often leading to substantial improvements.
SDPO learns notably faster than GRPO, performing close to 5 hours of GRPO training after only 1 hour of training with SDPO in several cases.
SDPO achieves a particularly substantial improvement over GRPO on the Chemistry task, as is displayed in Figure [6 (left)](#S3.F6 "Figure 6 ‣ 3 Learning without Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation").
With Olmo3-7B-Instruct, *SDPO achieves the 5h GRPO accuracy in 30 minutes of wall-clock training time*, a 10×10\times speedup.
Moreover, SDPO’s 5h accuracy is more than 2020%-points higher than that of GRPO.

|  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Chemistry | | Physics | | Biology | | Materials | | Tool use | |
|  | 1h | 5h | 1h | 5h | 1h | 5h | 1h | 5h | 1h | 5h |
| Qwen3-8B | 35.6 | | 59.2 | | 27.9 | | 58.9 | | 57.5 | |
| + GRPO | 54.754.7 | 60.060.0 | 63.863.8 | 72.772.7 | 34.334.3 | 51.851.8 | 74.3 | 77.177.1 | 64.964.9 | 67.767.7 |
| + GRPO (on-policy) | 54.2 | 69.6 | 63.6 | 63.6 | 44.4 | 44.4 | 73.9 | 74.1 | 60.2 | 65.7 |
| + SDPO (on-policy) | 60.0 | 70.1 | 66.6 | 75.6 | 51.5 | 52.9 | 72.1 | 78.4 | 68.0 | 68.5 |
| Olmo3-7B-Instruct | 18.8 | | 37.7 | | 18.1 | | 36.7 | | 39.3 | |
| + GRPO | 32.7 | 46.846.8 | 55.3 | 63.363.3 | 47.8 | 62.0 | 70.9 | 75.075.0 | 56.4 | 65.0 |
| + GRPO (on-policy) | 48.8 | 54.3 | 62.7 | 62.7 | 54.2 | 63.8 | 73.3 | 73.5 | 56.8 | 60.6 |
| + SDPO (on-policy) | 59.2 | 76.8 | 59.9 | 66.1 | 56.1 | 58.358.3 | 73.7 | 79.1 | 60.8 | 62.162.1 |

Table 3: Comparison of SDPO and GRPO on reasoning-related benchmarks. We report the highest achieved avg@16 within 1 hour and 5 hours of wall-clock training time, respectively. Both SDPO and on-policy GRPO perform one gradient step per generation batch, while GRPO performs 4 off-policy mini batch steps. We select optimal hyperparameters for SDPO and baselines based on 5h accuracy. Each run is performed on a node with 4 NVIDIA GH200 GPUs. Together with initialization and validation, each run takes approximately 6 hours.

We remark that our results with SDPO use strictly on-policy training (i.e., one gradient step per generation batch).
Given the known efficiency gains of off-policy methods that perform multiple gradient updates per generation batch, we believe that studying SDPO with off-policy updates is an exciting direction for future work.

Takeaway 1

We demonstrate that SDPO can learn to reason effectively, generalizing to challenging reasoning tasks.
Without requiring any modification to existing RLVR environments, SDPO outperforms GRPO substantially in several cases.

### 3.3 Self-distillation learns to reason concisely

We consistently observe that SDPO produces substantially shorter generations than GRPO while achieving higher accuracy.
SDPO’s responses are more than 3×3\times shorter on average across tasks (cf. [Table˜8](#A4.T8 "In D.1 Learning without rich environment feedback ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation") in [Appendix˜D](#A4 "Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation")).
On Chemistry with Olmo3-7B-Instruct, SDPO even achieves a 7×7\times reduction in response length relative to GRPO while maintaining higher accuracy (Figure [6 (right)](#S3.F6 "Figure 6 ‣ 3 Learning without Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation")).
While recent progress in RLVR has demonstrated that scaling response length is a powerful driver of emergent reasoning capabilities (Jaech et al., [2024](#bib.bib41); Guo et al., [2025](#bib.bib32); Muennighoff et al., [2025](#bib.bib65)), our results suggest that effective reasoning need not always be verbose. We find that SDPO improves the *efficiency* of reasoning.

Qualitatively, we observe that the longer responses from GRPO often stem from “superficial” reasoning rather than necessary cognitive steps.
GRPO frequently generates filler phrases like “Hmm” and “Wait” or enters circular logical loops that repeat previous steps verbatim.
[Figure˜7](#S3.F7 "In 3.3 Self-distillation learns to reason concisely ‣ 3 Learning without Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation") displays a representative example of this phenomenon.
Remarkably, SDPO’s generations remain concise and avoid these superficial patterns.
This may be explained by SDPO’s dense credit assignment, which assigns a specific advantage to each next-token prediction, leading to sparse advantages (cf. [Figure˜21](#A6.F21 "In F.1 Visualization of Advantages ‣ Appendix F Qualitative Examples ‣ Reinforcement Learning via Self-Distillation") in [Appendix˜F](#A6 "Appendix F Qualitative Examples ‣ Reinforcement Learning via Self-Distillation")).
By improving the efficiency of reasoning, SDPO reduces inference generation time and demonstrates that reasoning performance can be improved by refining *how* the model reasons, not just how *long* it reasons.

…Alternatively…
  
Closer to D? No…
  
Wait I’m going in circles… Wait, perhaps the correct answer is B… 101.85≈69.3\smash{10^{1.85}\approx 69.3}…
  
Ah, this works… Wait I think I messed up…
  
Hmm… 101.85≈69.3\smash{10^{1.85}\approx 69.3}…
  
Thus, the correct answer is likely B: 1.85.
  
<answer>
  
B
  
</answer>

(a) GRPO (5,549 tokens)

…At pH 7.4, all functional groups are neutral… maintaining a balance between hydrophobic and hydrophilic character… [The] overall polarity… keeps logD from being very high… or very low… [typically falling] in the 2.0-3.0 range, with 2.61 (C) being a reasonable estimate…
  
<answer>
  
C
  
</answer>

(b) SDPO (764 tokens)

Figure 7: Example responses from GRPO and SDPO after 50 training steps to the following question: “What is the correct octanol/water distribution coefficient logD under the circumstance of pH 7.4 for the molecule O=C1O[C@@H](COc2ccon2)CN1c1ccc(C2=CCOCC2)c(F)c1?” The answer options are A: 1.32, B: 1.85, C: 2.61, D: 3.76. The correct answer is C.
GRPO’s answer contains 5×5\times “Hmm.”, 9×9\times “No.”, and 25×25\times “Wait”. Further, GRPO’s answer repeats calculations such as “101.85≈69.3\smash{10^{1.85}\approx 69.3}”, which appears four times, and the model even explicitly generates “Wait I’m going in circles”.
SDPO’s answer avoids any circular reasoning and is more than 7×7\times shorter. The base model is Qwen3-8B.

## 4 Learning with Rich Environment Feedback

We next evaluate SDPO on coding tasks.
Coding is a canonical example of an RL environment that provides rich feedback, such as runtime errors and failed unit tests.
Learning to solve these coding problems requires strong credit assignment since the student must identify its precise mistakes to avoid repeating them in the future.
LiveCodeBench (LCB; Jain et al., [2025](#bib.bib42)) provides a set of contest-style coding problems, ranging from simple to competition-level.
We restrict our evaluation to the most recent LCBv6 subset of LCB, which contains 131 questions released between February and May 2025.
We consider a setting with public and private unit tests, common for code contests and coding platforms like LeetCode, where the public tests are used for evaluation during training and the private tests are used for validation (Chen et al., [2022](#bib.bib14); Le et al., [2022](#bib.bib51); El-Kishky et al., [2025](#bib.bib25); Samadi et al., [2025](#bib.bib74)).555We select public tests as a 50% random subset of private tests.

We use the Qwen3 (Yang et al., [2025](#bib.bib106)) model family for our experiments, with Qwen3-8B as default unless otherwise specified.
We report the average accuracy over 4 rollouts and use the same GRPO baseline as outlined in [Section˜3.1](#S3.SS1 "3.1 Experimental setting ‣ 3 Learning without Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation").

##### Results.

[Figure˜1](#S1.F1 "In 1 Introduction ‣ Reinforcement Learning via Self-Distillation") compares the learning curves of SDPO and GRPO on LCBv6.
We find that SDPO achieves a substantially higher final accuracy (48.8%) than GRPO (41.2%) while also outperforming the strongest instruct models on the public LCBv6 leaderboard:666On the public leaderboard, the LCBv6 subset can be obtained by selecting February to May 2025. Claude Sonnet 4 (40.5%) and Claude Opus 4 (39.7%).
Furthermore, SDPO reaches the final accuracy of GRPO in 4×4\times fewer generations.
We include an extended comparison to other RLVR baselines that perform similarly to GRPO in [Table˜9](#A4.T9 "In D.2.3 Baselines ‣ D.2 Learning with rich environment feedback ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation") in the appendix.
Differentiating between the easy, medium, and hard questions of LCB, we find that SDPO particularly improves over GRPO in solving medium and hard questions (cf. [Figure˜15](#A4.F15 "In D.2.1 Additional Results ‣ D.2 Learning with rich environment feedback ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation") in the appendix).

### 4.1 Self-distillation benefits from stronger models

!(/html/2601.20802/assets/x6.png)

Figure 8: SDPO improves with model size. We compare the final LCBv6 validation accuracy of SDPO and GRPO at train step 80, across model sizes from Qwen3.
The ability of SDPO’s teacher to perform accurate retrospection appears to be an emergent phenomenon with scale.
We include an additional scaling study with Qwen2.5-Instruct in the appendix (cf. [Figure˜17](#A4.F17 "In D.2.1 Additional Results ‣ D.2 Learning with rich environment feedback ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation")) which further supports this finding.
Error bars indicate the standard error across 3 seeds.

A central question for our work is whether SDPO is sensitive to the in-context learning ability of the base model.
Intuitively, we expect that SDPO benefits from a strong in-context learner, since this enables the teacher to perform more accurate retrospection.

To answer this question, we perform a scaling study with different model sizes from the Qwen3 (Yang et al., [2025](#bib.bib106)) family.
As shown by extensive prior work, the ability to learn in-context increases with model size (e.g., Brown et al., [2020](#bib.bib9)).
As depicted in [Figure˜8](#S4.F8 "In 4.1 Self-distillation benefits from stronger models ‣ 4 Learning with Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation"), SDPO significantly outperforms GRPO on larger models while only slightly improving over GRPO on smaller models.
To determine whether SDPO can also underperform GRPO on a model weaker than Qwen3-0.6B, we performed an additional scaling study with Qwen2.5-Instruct (Qwen et al., [2024](#bib.bib70)).
While outperforming GRPO with Qwen2.5-7B and performing similarly with Qwen2.5-8B, we find that SDPO underperforms GRPO on Qwen2.5-1.5B, as seen in [Figure˜17](#A4.F17 "In D.2.1 Additional Results ‣ D.2 Learning with rich environment feedback ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation") in [Appendix˜D](#A4 "Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation").

Takeaway 2

Our results suggest that the marginal improvement of SDPO over GRPO is tightly coupled with the strength of the base model, and motivates future study on models stronger than Qwen3-8B.
In the same way that in-context learning is an emergent phenomenon with scale, the self-teacher’s ability to perform accurate retrospection in SDPO appears to be emergent with scale.

### 4.2 Self-distillation performs dense credit assignment

!(/html/2601.20802/assets/x7.png)

Figure 9: Dense credit assignment in SDPO in the example from [Figure˜4](#S2.F4 "In 2 SDPO: Self-Distillation Policy Optimization ‣ Reinforcement Learning via Self-Distillation"). Shown in blue are tokens which become more likely under the self-teacher. The self-teacher identifies how the returned range has to be modified so that it does not contain n.

Whereas GRPO assigns a constant advantage to each generated token, SDPO assigns an individual advantage to *each possible next token* along the generated sequence based on the agreement of student and teacher.
At each position tt in the generated sequence yy, there are |𝒱||\mathcal{V}| possible next tokens where 𝒱\mathcal{V} is the vocabulary.
In distillation, this level is typically called the *logit-level* since it corresponds to the logits of the model.
In practice, we approximate the full next-token distribution by the top-KK tokens, and as such, SDPO assigns |y|⋅K|y|\cdot K unique advantages per sequence.
This is illustrated in [Figure˜9](#S4.F9 "In 4.2 Self-distillation performs dense credit assignment ‣ 4 Learning with Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation") and allows SDPO to perform dense credit assignment.

A natural question is whether the performance gains of SDPO are due to leveraging rich feedback in RLRF or due to the dense credit assignment of SDPO.
To answer this question, we ablate the performance of SDPO in three configurations:

* •

  Logit-level SDPO: credit assignment over the 100 most likely tokens (under the student) at each position.
* •

  Token-level SDPO: credit assignment over the most likely token at each position.
* •

  Sequence-level SDPO: We compute SDPO advantages for all generated tokens and average them to produce a single scalar advantage per sequence (as in GRPO). This does not perform denser credit assignment than GRPO but still leverages the rich feedback ff.

As shown in Figure [10 (left)](#S4.F10 "Figure 10 ‣ 4.3 The self-teacher improves during training ‣ 4 Learning with Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation"), the dense credit assignment of logit-level SDPO leads to significant performance gains over token-level SDPO and sequence-level SDPO.
Nevertheless, even sequence-level SDPO outperforms GRPO, indicating that leveraging rich feedback in RLRF can lead to substantial gains over RLVR methods even without dense credit assignment.

### 4.3 The self-teacher improves during training

!(/html/2601.20802/assets/x8.png)

Figure 10: Left: Rich feedback in RLRF and dense credit assignment of SDPO are complementary. We compare logit-level, token-level, and sequence-level SDPO advantages to GRPO. While denser credit assignment in SDPO is beneficial (logit-level > token-level > sequence-level), even sequence-level SDPO significantly outperforms GRPO due to leveraging the rich feedback. Error bars indicate the standard error across 3 seeds. Right: The self-teacher improves during training. We display the generative accuracy of the self-teacher compared to student on the current training batch (with a rolling average over 5 steps). The final student score is taken at step 80. Notably, the performance of the student significantly surpasses the initial teacher’s accuracy. Error bars indicate the standard deviation across 3 seeds.

|  |  |  |
| --- | --- | --- |
| Teacher | Accuracy | Avg accuracy |
| qθq\_{\theta} | 36.1±1.636.1\pm 1.6 | 29.8±1.329.8\pm 1.3 |
| qθrefq\_{\theta\_{{\mathrm{ref}}}} | 48.8±0.748.8\pm 0.7 | 44.4±0.244.4\pm 0.2 |
| Trust-region | 50.6±0.9\mathbf{50.6}\pm 0.9 | 45.6±0.2\mathbf{45.6}\pm 0.2 |
| EMA | 49.3±0.349.3\pm 0.3 | 45.3±0.2\mathbf{45.3}\pm 0.2 |

Table 4: Best/average accuracy until step 90 of various methods for teacher regularization. Trust-region and EMA teachers use α=0.01\alpha=0.01. Training of the qθq\_{\theta} eventually diverges. Error ranges indicate standard errors across 3 seeds.

Contrary to standard distillation, the self-teacher in SDPO is not frozen, but updated throughout training.
This is a critical component of SDPO, since it enables the teacher to improve over time, which means that the student can learn from a stronger target.
To investigate whether the self-teacher improves during training, we plot the average accuracy when *generating* using the self-teacher in Figure [10 (right)](#S4.F10 "Figure 10 ‣ 4.3 The self-teacher improves during training ‣ 4 Learning with Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation").
We find that the self-teacher improves significantly during training.
Most notably, the student’s accuracy surpasses the initial teacher’s accuracy in later stages of training.
This demonstrates that SDPO enables true bootstrapping of a weak model to a strong model, without the initial self-teacher’s performance limiting the final student.

As described in [Section˜2.3](#S2.SS3 "2.3 Stability improvements ‣ 2 SDPO: Self-Distillation Policy Optimization ‣ Reinforcement Learning via Self-Distillation"), SDPO uses a regularized teacher to stabilize training.
As can be seen in [Table˜4](#S4.T4 "In 4.3 The self-teacher improves during training ‣ 4 Learning with Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation"), a non-regularized teacher significantly underperforms the regularized teachers.
Furthermore, trust-region and EMA teachers outperform the teacher frozen at the initial teacher’s parameters, showing that the teacher improves through parameter sharing with the student.
Yet, SDPO performs well even with a frozen teacher.

### 4.4 On-policy self-distillation avoids catastrophic forgetting

Prior work has shown that a key benefit of on-policy algorithms, such as GRPO, is that models tend not to forget previously obtained capabilities (Shenfeld et al., [2026](#bib.bib83); Chen et al., [2025b](#bib.bib15); Lu & Thinking Machines Lab, [2025](#bib.bib60)).
This is practically desirable since it enables continual training pipelines where a model is trained sequentially on diverse tasks without the need to retrain from scratch.
To evaluate forgetting, we test the final checkpoints of GRPO and SDPO on diverse holdout tasks: IFEval (Zhou et al., [2023](#bib.bib119)), which tests the ability of a model to follow precise format instructions; ArenaHard-v2 (Li et al., [2025](#bib.bib55)), which is an LLM-judged benchmark of real-world instruction-following prompts derived from LMArena (Chiang et al., [2024](#bib.bib19)); and MMLU-Pro (Wang et al., [2024b](#bib.bib101)), which tests broad multi-task knowledge and reasoning.
As displayed in [Table˜5](#S4.T5 "In Off-policy self-distillation baseline. ‣ 4.4 On-policy self-distillation avoids catastrophic forgetting ‣ 4 Learning with Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation"), SDPO learns the new task while mitigating degradation of initial capabilities, overall achieving a better performance–forgetting tradeoff than GRPO.

##### Off-policy self-distillation baseline.

As an additional baseline, we consider training the student via supervised fine-tuning (SFT) on successful generations from the self-teacher (Scheurer et al., [2023](#bib.bib77); Dou et al., [2024](#bib.bib24)).777SFT on a teacher’s predictions is a standard off-policy distillation approach (Kim & Rush, [2016](#bib.bib46)).
This requires 2×2\times the generations of SDPO for the same number of steps, since we have to generate from both the student and the teacher.
We report SFT on the successes of the self-teacher, which achieves a higher accuracy than also including initial successes from the student in the SFT data.
As shown in [Table˜5](#S4.T5 "In Off-policy self-distillation baseline. ‣ 4.4 On-policy self-distillation avoids catastrophic forgetting ‣ 4 Learning with Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation"), SFT on the self-teacher significantly underperforms SDPO on LCBv6, while leading to worse forgetting of prior capabilities.
This mirrors prior findings on the instability of off-policy imitation (see, e.g., Agarwal et al., [2024](#bib.bib1)).

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Task: | Holdout tasks: | | | | |
|  | LCBv6 | IFEval | ArenaHard-v2  (hard prompt) | ArenaHard-v2  (creative writing) | MMLU-Pro | Avg.  (holdout) |
| Base | 27.927.9 | 83.9{83.9} | 14.0{14.0} | 13.7{13.7} | 62.5{62.5} | 43.5{43.5} |
| SFT on self-teacher | 42.7{42.7} | 83.7 | 11.2 | 8.9 | 61.9 | 41.4 |
| GRPO | 41.241.2 | 82.282.2 | 12.012.0 | 10.810.8 | 62.362.3 | 41.841.8 |
| SDPO | 48.8{48.8} | 83.2{83.2} | 12.3{12.3} | 11.1{11.1} | 62.9{62.9} | 42.4{42.4} |

Table 5: On-policy methods do not suffer from catastrophic forgetting. We compare the accuracy of the final checkpoint on the training task LCBv6 and on holdout tasks IFEval, ArenaHard-v2, and MMLU-Pro. We compare to a baseline that trains directly on responses generated by the initial self-teacher with SFT. Overall, SDPO achieves the best performance–forgetting tradeoff. We include additional baseline results in [Table˜9](#A4.T9 "In D.2.3 Baselines ‣ D.2 Learning with rich environment feedback ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation") in the appendix.

### 4.5 Can GRPO and SDPO be combined?

GRPO utilizes Monte Carlo advantages, which are unbiased with respect to the objective of maximizing expected reward J​(θ):=𝔼y∼πθ(⋅∣x)​[r​(y∣x)]J(\theta):=\smash{\mathbb{E}\_{y\sim\pi\_{\theta}(\cdot\mid x)}{}\left[r(y\mid x)\right]}.
In contrast, SDPO advantages are inherently biased with respect to J​(θ)J(\theta) due to being computed from rich feedback and a self-teacher.
This dichotomy parallels the fundamental distinction between Monte Carlo and bootstrapped advantages in RL: while the latter are biased, they typically yield lower variance (Sutton & Barto, [1998](#bib.bib92); Schulman et al., [2016](#bib.bib79)).
This motivates a hybrid approach that combines reward-derived GRPO advantages with feedback-derived SDPO advantages:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Ai,tSDPO+GRPO​(y^i,t):=λ​Ai,tGRPO​(y^i,t)+(1−λ)​Ai,tSDPO​(y^i,t),λ∈[0,1].A\_{i,t}^{\mathrm{SDPO+GRPO}}(\hat{y}\_{i,t}):=\lambda A\_{i,t}^{\mathrm{GRPO}}(\hat{y}\_{i,t})+(1-\lambda)A\_{i,t}^{\mathrm{SDPO}}(\hat{y}\_{i,t}),\quad\lambda\in[0,1]. |  | (3) |

!(/html/2601.20802/assets/x9.png)

  

Figure 11: We compare the LCBv6 validation accuracy at step 80, across model sizes from Qwen3.
SDPO+GRPO significantly outperforms SDPO on the weaker Qwen3-0.6B, while slightly underperforming SDPO on stronger models.
We use λ=0.9\lambda=0.9.
Error bars indicate the standard error across 3 seeds.

As shown in [Figure˜11](#S4.F11 "In 4.5 Can GRPO and SDPO be combined? ‣ 4 Learning with Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation"), SDPO+GRPO appears to be more robust to weaker models than SDPO.
Intuitively, in a weaker model such as Qwen3-0.6B, the SDPO advantages are less reliable, and hence including the GRPO advantage helps to stabilize training.
In contrast, we find that SDPO+GRPO slightly underperforms SDPO on stronger models such as Qwen3-8B.
This suggests that the signal of GRPO, only informed by a scalar reward, can be actively harmful with a strong initial model.

### 4.6 Which feedback is most informative?

To understand which type of rich feedback is most informative, we ablate the three types of feedback present in a verifiable environment like code generation: the sample solution (if a successful rollout is available in the current rollout group), the environment output (such as runtime errors), and the student’s original attempt.

##### Sample solutions.

Including a sample solution from a failed attempt’s rollout group (if available) closely mirrors the group-relative advantages of GRPO.
We emphasize that these sample solutions are always generated by the student, as in GRPO, and do not require an expert model.
They allow for disincentivizing unsuccessful approaches if the model is already able to solve the question.
However, unlike GRPO where all tokens receive the same negative advantage, the self-teacher can identify specific mistakes and provide feedback on how to fix them.

##### Environment output.

The environment output describes the state of the environment after the student’s attempt.
This is complementary to sample solutions since it can provide useful signal even if the student has never solved the question before (a setting we explore extensively in [Section˜5](#S5 "5 Solving Hard Questions via Test-Time Self-Distillation ‣ Reinforcement Learning via Self-Distillation")).
Leveraging environment output is a key differentiating factor between RLRF and RLVR settings.

##### Student’s original attempt.

The student’s original attempt yy does not have to be included in the reprompting template of the teacher.
Indeed, we find that including it biases the teacher towards the student’s attempt (cf. [Table˜6](#S4.T6 "In Student’s original attempt. ‣ 4.6 Which feedback is most informative? ‣ 4 Learning with Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation")).
This reduces the entropy of the student’s distribution (particularly for initially uncertain tokens), thereby reducing exploration.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Teacher before training | | Student trained with SDPO | |
|  | ↑\uparrow Acc. (%) | ↓\downarrow Same output (%) | ↑\uparrow Acc. (%) | Avg. entropy |
| f=f= output | 32.5±0.532.5\pm 0.5 | 13.7±0.613.7\pm 0.6 | 39.8±0.239.8\pm 0.2 | 0.40±0.00.40\pm 0.0 |
| f=f= solution | 42.4±1.0\mathbf{42.4}\pm 1.0 | 12.1±0.712.1\pm 0.7 | 36.8±2.736.8\pm 2.7 | *0.07*±0.0\emph{0.07}\pm 0.0 |
| f=f= output + solution | 42.5±1.2\mathbf{42.5}\pm 1.2 | 10.1±0.2\mathbf{10.1}\pm 0.2 | 48.9±0.9\mathbf{48.9}\pm 0.9 | 0.37±0.00.37\pm 0.0 |
| f=f= yy + output + solution | 39.3±0.839.3\pm 0.8 | 30.0±0.930.0\pm 0.9 | 44.5±1.844.5\pm 1.8 | *0.23*±0.0\emph{0.23}\pm 0.0 |

Table 6: Performance of varying kinds of feedback. We evaluate informativeness of feedback based on SDPO training (until step 70) as well as the direct impact on the self-teacher. “Same output” measures the percentage of cases where the teacher receives the same environment output as the student’s initial attempt (i.e., not exploring alternative approaches). We observe that environment output and sample solutions are complementary and each provide informative feedback. Naively including only solutions or initial attempts yy significantly reduces diversity in the teacher and student. We remark that the sample solutions are generated by the student, enabling similar group-relative advantage estimation to GRPO. Error bars indicate standard deviation across 3 seeds.

We summarize results in [Table˜6](#S4.T6 "In Student’s original attempt. ‣ 4.6 Which feedback is most informative? ‣ 4 Learning with Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation") where we evaluate the effect on SDPO training as well as the direct impact on the self-teacher.
We find that environment output & sample solutions are complementary, each providing informative feedback.
Generally, we observe that performance is not sensitive to syntactic variations of the reprompting template from [Table˜2](#S2.T2 "In 2 SDPO: Self-Distillation Policy Optimization ‣ Reinforcement Learning via Self-Distillation").

## 5 Solving Hard Questions via Test-Time Self-Distillation

In [Sections˜3](#S3 "3 Learning without Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation") and [4](#S4 "4 Learning with Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation"), we have demonstrated that SDPO can substantially improve over RLVR methods when performing “train-time RL” for reasoning tasks.
We now turn to a test-time setting where the model is given only a single hard (binary-reward) question xx and must discover a solution as quickly as possible:

###### Definition 5.1 (Discovery time).

The discovery time is the number of trials needed until a solution is found (i.e., the smallest kk with the kk-th attempt yky\_{k} receiving reward 1).

Based on this notion, we
can define a measure of the efficacy of discovery:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | discovery​@​k:=ℙ​(discovery time ≤k)=ℙ​(r​(y1∣x)=1 or r​(y2∣x)=1 or …or r​(yk∣x)=1),\displaystyle\begin{split}\mathrm{discovery@}k:=&\ \mathbb{P}(\text{discovery time $\leq k$})\\ =&\ \mathbb{P}(\text{$r(y\_{1}\mid x)=1$ or $r(y\_{2}\mid x)=1$ or \ldots or $r(y\_{k}\mid x)=1$}),\end{split} | |  | (4) |

where the probability is over any randomness in the algorithm producing yky\_{k} and the rewards.
Thus, the discovery@kk metric quantifies the probability of
discovering the solution within kk steps.888Our proposed discovery@kk metric is a canonical metric
in the study of runtime speedup (i.e., time until termination, Dolan & Moré ([2002](#bib.bib23))).
While prior work has studied discovery with continuous rewards (e.g., Novikov et al., [2025](#bib.bib67); Yuksekgonul et al., [2026](#bib.bib112)), discovery with language models in sparse or binary-reward settings does not allow “hill-climbing” a continuous reward and has remained less well understood.

The most naive approach to discovery in binary-reward tasks is to sample repeatedly i.i.d. from the base model, also known as best-of-kk.
The canonical pass@kk metric for best-of-kk sampling is exactly the probability of discovering at least one solution within kk independent samples from a fixed model, coinciding with discovery@kk.
The discovery@kk metric generalizes pass@kk to algorithms that sample attempts sequentially.
A common sequential approach re-prompts the base model with additional context from previous attempts (Madaan et al., [2023](#bib.bib62); Shinn et al., [2023](#bib.bib85)).
We refer to this as multi-turn sampling.
Here, the model itself does not change, only its context evolves over time.

Performing RLVR on the question xx does not improve over best-of-kk sampling from the base model, since a binary reward provides no signal until the first solution has already been found.999For this reason, several works consider explicitly constructing curricula of solvable questions (e.g., Zhao et al., [2025](#bib.bib116); Huang et al., [2026](#bib.bib37); Diaz-Bone et al., [2025](#bib.bib22); Hübotter et al., [2025b](#bib.bib40)), which self-distillation avoids. Other work found that RLVR yields limited improvement on hard questions (Yue et al., [2025](#bib.bib110)).
An RLRF method like SDPO does not face the same limitation, as it receives rich feedback from the environment after each attempt.
This rich feedback enables the model to repeatedly “correct” its mistakes as it encounters them and receives feedback, even before ever discovering a solution.
In contrast to multi-turn sampling, SDPO repeatedly compresses context c=(yk,fk)c=(y\_{k},f\_{k}) by distilling πθ(⋅∣x,c)\pi\_{\theta}(\cdot\mid x,c) into a model πθ′(⋅∣x)\pi\_{\theta^{\prime}}(\cdot\mid x) as we illustrate in [Figure˜12](#S5.F12 "In 5 Solving Hard Questions via Test-Time Self-Distillation ‣ Reinforcement Learning via Self-Distillation").
This self-distillation enables SDPO to continually learn over long contexts, whereas the memory bottleneck of transformers inherently limits the context length of multi-turn sampling (Vaswani et al., [2017](#bib.bib96)).
In this section, we seek to answer the question:

Can repeatedly compressing context into model weights via self-distillation

accelerate discovery for hard questions?

!(/html/2601.20802/assets/x10.png)

Figure 12: Compressing context into model weights via self-distillation.
We illustrate the process of distilling the interaction history (context cc) into the model parameters θ\theta.
The model πθ\pi\_{\theta} repeatedly attempts a fixed hard question xx, generating an answer yy and receiving feedback ff.
Rather than appending this history to the context window, the model updates its weights θt→θt+1\theta\_{t}\to\theta\_{t+1} with SDPO (batch size 11) based on the feedback, effectively “fixing” mistakes by encoding πθ(⋅∣x,c)\pi\_{\theta}(\cdot\mid x,c) directly into the policy πθ′(⋅∣x)\pi\_{\theta^{\prime}}(\cdot\mid x).

### 5.1 Experimental setting

We consider a particularly challenging subset of questions from LCBv6 that are at Qwen3-8B’s performance ceiling and require significant test-time sampling to find any solution.
Concretely, we define two groups using Qwen3-8B’s pass@kk: *Hard tasks* with pass@​64<0.5{\text{pass@}64<0.5} and *very hard tasks* with pass@​64<0.03\text{pass@}64<0.03.
Among these, we retain questions for which any of best-of-kk, multi-turn, or SDPO find at least one solution within 512512 steps across 55 seeds.
This results in 19 hard and 9 very hard questions.

For best-of-kk sampling under the base model, we report the standard pass​@​k\text{pass}@k estimate (Chen et al., [2021b](#bib.bib17)) from 2944 independent rollouts.
As multi-turn sampling, we sequentially reprompt the model in-context using the concatenated feedback from previous attempts. To remain within Qwen3-8B’s 40k-token context limit, we employ a first-in, first-out sliding window, discarding the earliest feedback once the maximum prompt length (32k tokens) is reached.
We ablate the multi-turn reprompting strategy in [Figure˜19](#A4.F19 "In D.3 Test-time self-distillation ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation") in [Appendix˜D](#A4 "Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation") and find that retaining only past feedback while forgetting earlier attempts significantly outperforms the baseline that additionally retains past attempts.
We evaluate SDPO with a batch size of 16. We ablate this choice in [Figure˜19](#A4.F19 "In D.3 Test-time self-distillation ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation") in [Appendix˜D](#A4 "Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation") and find that overall performance differences are marginal, yet smaller batch sizes are beneficial for improvements at low generation budgets, while larger batch sizes result in more stable updates that still learn to solve questions at later stages into the run.

### 5.2 Results

!(/html/2601.20802/assets/x11.png)

  

Figure 13: Self-distillation at test-time solves LiveCodeBench questions that neither the base model nor multi-turn conversations can solve.
Left: Very hard questions (9 total) from LCBv6 where the base model achieves pass​@​64<0.03\text{pass}@64<0.03, i.e., in less than 3% cases, sampling 64 responses yields any success.
Right: Hard questions (19 total) from LCBv6 where the base model achieves pass​@​64<0.5\text{pass}@64<0.5.
We report the discovery​@​k\text{discovery}@k metric, representing the probability of discovering at least one solution within kk total generations.
Across both difficulty levels, SDPO achieves higher discovery​@​k\text{discovery}@k rates at almost all generation budgets, compared to the base model and a multi-turn conversation baseline that receives the feedback in-context. We report the mean and bootstrapped 90% confidence intervals of the mean across 5 random seeds per question.

[Figure˜13](#S5.F13 "In 5.2 Results ‣ 5 Solving Hard Questions via Test-Time Self-Distillation ‣ Reinforcement Learning via Self-Distillation") compares discovery​@​k\text{discovery}@k for SDPO, multi-turn sampling, and best-of-kk sampling on very hard (left) and hard (right) questions from LCBv6. Across both difficulty levels, SDPO achieves substantially higher discovery​@​k\text{discovery}@k rates at almost all generation budgets.

On very hard tasks, multi-turn and best-of-kk largely fail to solve questions within the available generation budget, achieving discovery@2750 of only 35.6%35.6\% and 41.5%{41.5}\%, respectively, whereas SDPO discovers a solution in 53.2%{53.2}\% of cases.
SDPO not only solves more questions overall but also does so with substantially fewer attempts.
Notably, to reach a 22%22\% discovery probability on very hard questions, SDPO requires approximately 3×3\times fewer generations than best-of-kk and multi-turn sampling.
On hard tasks, SDPO reaches a 78%{78}\% discovery@2750 probability while achieving a 67%67\% discovery probability with roughly 2.4×2.4\times fewer generations than best-of-kk and multi-turn sampling. Overall, multi-turn and best-of-kk sampling solve only 68.4%{68.4}\% and 72.3%{72.3}\% of questions, respectively.
The context window length for multi-turn sampling is reached after 837 (±466\pm 466) steps for hard questions and after 1007 (±349\pm 349) steps for very hard questions, offering a possible explanation for its diminishing gains at high generation budgets.

##### Question 3 is only solved by SDPO.

SDPO solves all questions that are solved by best-of-kk and multi-turn sampling. Beyond that, SDPO uniquely discovers a solution for Q3, which is neither solvable with multi-turn sampling nor with best-of-kk sampling within 2750 attempts. In contrast, SDPO first discovers a solution for Q3 after 321 attempts, which corresponds to 20 iteration steps of self-distillation based on feedback with a batch size of 16. We include detailed per-question results in [Table˜10](#A4.T10 "In D.3 Test-time self-distillation ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation") in [Appendix˜D](#A4 "Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation").

##### The initial self-teacher does not solve hard questions.

Notably, the self-teacher’s initial accuracy is <1<1% for almost all questions, and even exactly 0% on 7878% of them ([Table˜11](#A4.T11 "In D.3 Test-time self-distillation ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation") in [Appendix˜D](#A4 "Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation")).
This shows that a single turn of in-context feedback is insufficient to solve the problem.
Despite this, the self-teacher’s credit assignment is sufficiently effective for SDPO to iteratively refine the policy and eventually solve these questions.

Takeaway 3

We demonstrate that rich environment feedback enables SDPO to significantly accelerate discovery for hard questions.
This is in contrast to RLVR methods, which only receive a binary reward signal, and therefore only begin learning once the first solution has already been found.

## 6 Related Work

### 6.1 Reinforcement Learning with LLMs

Recently, large-scale RL training on diverse tasks has significantly improved the performance of LLMs on general reasoning tasks (Guo et al., [2025](#bib.bib32); Kimi et al., [2025](#bib.bib47); Olmo et al., [2025](#bib.bib68); Jaech et al., [2024](#bib.bib41); Lambert et al., [2025](#bib.bib50)).
This progress is primarily enabled by RLVR methods that use Monte Carlo estimates of rewards, such as STaR or GRPO (Zelikman et al., [2022](#bib.bib113); Shao et al., [2024](#bib.bib82)), similar to the classical REINFORCE algorithm (Williams, [1992](#bib.bib103)).
While several traditional RLVR algorithms rely on learning separate value networks (Schulman et al., [2017](#bib.bib80)), they incur substantial memory costs and retain the information bottleneck of scalar rewards.

In the RLVR setting, it is common for an (outcome) reward to be given only at the end of a sequence.
To improve credit assignment, several works learn so-called process reward models (PRMs) that estimate rewards for each step in the sequence (Lightman et al., [2023](#bib.bib56); Wang et al., [2024a](#bib.bib99); Setlur et al., [2025](#bib.bib81)).
Unlike our RLRF setting, PRMs are typically trained on scalar rewards, either on value estimates for intermediate states or on outcome rewards (Cui et al., [2025](#bib.bib21)).
Unlike the self-teacher in SDPO, PRMs are a distinct model from the student, introducing significant memory overhead.
Our work shows that *each language model is implicitly a PRM* through retrospection if given rich feedback.

Conceptually, our work is related to “expert iteration” (Anthony et al., [2017](#bib.bib4)) where a student is bootstrapped by repeatedly imitating an improved version of itself (called the “expert”).
Canonically, the expert combines the student with test-time search, such as tree search (Anthony et al., [2017](#bib.bib4)) or majority voting (Zuo et al., [2025](#bib.bib122)).
In contrast, SDPO leverages the student’s ability to learn from rich feedback provided in-context.

### 6.2 Learning from Rich Feedback and through Retrospection

Beyond scalar outcome rewards, recent works have leveraged rich execution or verbal feedback to guide generation (Gehring et al., [2025](#bib.bib28); Yuksekgonul et al., [2025](#bib.bib111)).
A primary line of research focuses on translating verbal feedback into reward functions for RL.
This is often achieved by mapping feedback to discrete token-level rewards using an external frozen model (Wang et al., [2026](#bib.bib98)), or by employing strong external LLMs to explicitly construct state-wise reward functions (Goyal et al., [2019](#bib.bib29); Xie et al., [2024](#bib.bib105); Urcelay et al., [2026](#bib.bib95)).

Alternatively, feedback can be utilized without explicit reward modeling.
Several approaches focus on in-context improvement without integrating the process into the RL optimization loop (Chen et al., [2021a](#bib.bib16); Madaan et al., [2023](#bib.bib62); Shinn et al., [2023](#bib.bib85); Yao et al., [2024](#bib.bib108); Yuksekgonul et al., [2025](#bib.bib111); Lee et al., [2025](#bib.bib53)).
Others manually curate preference datasets by pairing responses before and after feedback to train with direct preference optimization (Stephan et al., [2024](#bib.bib89); Lee et al., [2024](#bib.bib52)), though this requires additional generation and lacks the direct credit assignment of SDPO.
Various recent works bootstrap thinking traces from known answers, using these answers as rich feedback (Zhou et al., [2026](#bib.bib120); Hatamizadeh et al., [2026](#bib.bib35); Zhang et al., [2025](#bib.bib114)).

A central object in several recent works is a feedback-conditioned policy πθ​(y∣x,f)\pi\_{\theta}(y\mid x,f), which learns answers yy that lead to feedback ff (Liu et al., [2023](#bib.bib58); Zhang et al., [2023](#bib.bib115); Luo et al., [2025](#bib.bib61)), typically through supervised objectives.
The idea behind these approaches is to deploy a policy conditioned on desirable (i.e., positive) feedback for deployment.
This approach is conceptually related to goal-conditioned RL (Schaul et al., [2015](#bib.bib76); Liu et al., [2025a](#bib.bib57)), where one can learn from negative examples through goal relabeling (Andrychowicz et al., [2017](#bib.bib3)).
Feedback-conditioned policies view feedback as a goal, whereas RLRF views feedback as a state that can be used to determine whether the goal xx is achieved.
Unlike SDPO, these methods do not use feedback for credit assignment in negative trajectories, but rather as a data transformation for goal relabeling.

### 6.3 Distillation

Distillation is frequently employed as an alternative to supervised fine-tuning (SFT) when a strong teacher model is available.
This approach transfers capabilities by training a student to mimic the output distribution or intermediate representations of the teacher (Hinton et al., [2015](#bib.bib36); Romero et al., [2015](#bib.bib72); Kim & Rush, [2016](#bib.bib46); Sanh et al., [2019](#bib.bib75); Xie et al., [2020](#bib.bib104)).
Distillation is typically performed on fixed off-policy datasets.
To address the distribution shift between training and inference, recent works explore on-policy distillation, where the student learns from feedback of an external teacher on its own generations (Agarwal et al., [2024](#bib.bib1); Gu et al., [2024](#bib.bib30); Yang et al., [2025](#bib.bib106); Lu & Thinking Machines Lab, [2025](#bib.bib60)).
This mitigates the train-test mismatch, which relates closely to earlier work on online imitation learning (Ross et al., [2011](#bib.bib73)).

### 6.4 Self-Distillation

The concept of self-distillation was first proposed by Snell et al. ([2022](#bib.bib88)) in a setting akin to supervised learning, introducing the idea of sampling from a model provided with extra context and training the same model to mimic these predictions without that context.
This mechanism has proven effective for compressing behavior (Bai et al., [2022](#bib.bib5); Choi et al., [2022](#bib.bib20)) and factual information (Eyuboglu et al., [2026](#bib.bib26); Kujanpää et al., [2025](#bib.bib48)) into model weights.
Beyond compressing a fixed context into model weights, recent works have used self-distillation to learn from environment feedback (Scheurer et al., [2023](#bib.bib77); Dou et al., [2024](#bib.bib24); Mitra & Ulukus, [2025](#bib.bib63)).
These approaches use an *off-policy* self-distillation objective, which substantially underperforms SDPO’s on-policy learning.
Off-policy self-distillation trains the student on generations from the teacher, whereas SDPO trains the student to avoid mistakes in its own generations.
In concurrent work, Chen et al. ([2025c](#bib.bib18)) apply on-policy self-distillation to grid world settings where feedback is a scalar reward, and a reflection stage in the self-teacher diagnoses possible mistakes, showing improved credit assignment compared to learning value networks for advantage estimation.

## 7 Conclusion, Limitations, and Future Work

We introduced Reinforcement Learning with Rich Feedback (RLRF), a paradigm where environments provide tokenized feedback beyond scalar rewards, and argued that this removes a key information bottleneck of RLVR.
We then proposed Self-Distillation Policy Optimization (SDPO), which uses the current policy as a feedback-conditioned *self-teacher* and distills its corrected log-probabilities into the student.
This leverages the model’s ability to learn from context for dense credit assignment.
We further demonstrated that SDPO can be implemented as a minimal, drop-in modification to standard RLVR pipelines.

Empirically, SDPO demonstrates superior sample efficiency and wall-clock convergence compared to GRPO on reasoning tasks, even when training in standard RLVR environments without rich feedback.
SDPO’s gains grow with model scale, suggesting that the capacity for self-correction scales with the model’s in-context learning capabilities.
Moreover, we show that performing SDPO at test-time on individual hard binary-reward tasks accelerates the discovery of solutions compared to strong baselines.

SDPO enables learning from rich feedback in a way that parallels human cognition: utilizing precise outcomes rather than just binary rewards.
By allowing the model to determine retrospectively how it should have acted, we demonstrate that language models can convert diverse tokenized feedback into effective self-supervision.

##### Limitations.

Our findings show that SDPO’s performance depends on a model’s in-context learning ability, suggesting that SDPO is primarily applicable for RL-training stronger base models, while it can underperform GRPO on weaker models.
Moreover, performance depends on the quality of the environment feedback. If the environment provides uninformative or misleading feedback, a model may not be able to learn from it through SDPO.
Finally, SDPO adds a small computational overhead compared to GRPO for computing the log-probs of the retrospective model.
While often negligible, this may be a larger overhead for smaller models with shorter generation lengths, where generation time is comparatively small.

##### Future Work.

Our work highlights several exciting directions for future research:

* •

  Long-horizon and agentic settings.
  RLRF is particularly appealing when trajectories are long or expose information about intermediate states.
  Evaluating SDPO in agentic environments is a natural next step.
* •

  Training dynamics at scale.
  Beyond our evaluation on LiveCodeBench, it would be particularly interesting to scale SDPO to large multi-task RL training runs and further study its scaling properties with frontier base models.
* •

  Beyond verifiable rewards.
  While we focused on verifiable code generation, many tasks provide textual feedback without a ground-truth verifier.
  Investigating whether SDPO’s retrospection mechanism can improve alignment in open-ended text generation or continuous-reward tasks remains an open empirical question.
* •

  Behavioral differences in reasoning. We observed that SDPO induces qualitatively different reasoning patterns than GRPO, notably avoiding the latter’s tendency toward verbosity and superficial reasoning.
  Future work should systematically study how individual aspects, such as the reprompt template, influence behavior.

## Author Contributions

Jonas Hübotter conceived of the project in summer 2025 and has been working on it full-time since then, leading the team.
Jonas proposed the conceptual framework of self-distillation for credit assignment with input from Lejs, implemented the algorithm with help from others, led the quantitative experiments on LCBv6, and led the writing of the paper.

Frederike Lübeck led the design of the code environment, led the design and evaluation of the TTT setting in [Section˜5](#S5 "5 Solving Hard Questions via Test-Time Self-Distillation ‣ Reinforcement Learning via Self-Distillation") with input from Jonas, contributed to the project direction in discussions, and contributed significantly to the writing of the paper.

Lejs Behric noted the dense credit assignment of knowledge distillation with strong teacher models in discussions with Jonas, inspiring the idea of self-distillation. Further, Lejs led the evaluation of different teacher templates, co-led the development of a tool for qualitative analysis of runs with Marco and Daniel, helped implement parts of the algorithm, and contributed to the project direction in discussions.

Anton Baumann joined in December 2025 and led the evaluation of SDPO without rich feedback in [Section˜3](#S3 "3 Learning without Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation") with input from Jonas, and contributed to the writing of the paper.

Marco Bagatella and Daniel Marta co-led the development of a tool for qualitative analysis of runs with Lejs, contributed to the training infrastructure, and contributed to the project direction in discussions.

Ido Hakimi significantly contributed to the initial codebase and experimental setup, contributed early algorithmic ideas, and contributed to the project direction in discussions.

Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, and Andreas Krause supported this project, with Idan and Carlos joining in December 2025. They made significant contributions to the project direction in discussions and gave valuable advice on our presentation. Thomas and Idan, in particular, significantly contributed to the development of core algorithmic ideas and design of experiments. Thomas further evaluated checkpoints on holdout benchmarks. Carlos suggested the qualitative analysis of reasoning traces in [Figure˜7](#S3.F7 "In 3.3 Self-distillation learns to reason concisely ‣ 3 Learning without Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation") and the presentation of TTT results in [Section˜5](#S5 "5 Solving Hard Questions via Test-Time Self-Distillation ‣ Reinforcement Learning via Self-Distillation"). Andreas pointed out valuable connections to existing work in RL which shaped the direction of the project.

## Acknowledgments

We would like to thank Akira Yoshiyama, Yassir Akram, Parnian Kassraie, Jonathan Thomm, Roman Vorushin, Afra Amini, Imanol Schlag, Yu Sun, and Moritz Hardt for helpful discussions.
We thank Eduard Durech for helpful conversations regarding the scaling of RL fine-tuning and for his technical guidance on distributed infrastructure and long-context optimization.
Furthermore, we would like to thank Leander Diaz-Bone for supporting dataset generation.

This project was supported through the Swiss AI compute grant a156 and, in part, compute grant infra01.
JH was supported by the Swiss National Science Foundation under NCCR Automation, grant agreement 51NF40 180545.
FL and MB were supported by the ETH-MPI Center for Learning Systems.
TKB and IH were supported by an ETH AI Center Postdoctoral Fellowship.
DM was supported by the Knut and Alice Wallenberg Foundation.

## References

* Agarwal et al. (2024)

  Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem.
  On-policy distillation of language models: Learning from self-generated mistakes.
  In *ICLR*, 2024.
* Akyürek et al. (2025)

  Ekin Akyürek, Mehul Damani, Adam Zweiger, Linlu Qiu, Han Guo, Jyothish Pari, Yoon Kim, and Jacob Andreas.
  The surprising effectiveness of test-time training for few-shot learning.
  In *ICML*, 2025.
* Andrychowicz et al. (2017)

  Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, Pieter Abbeel, and Wojciech Zaremba.
  Hindsight experience replay.
  In *NeurIPS*, 2017.
* Anthony et al. (2017)

  Thomas Anthony, Zheng Tian, and David Barber.
  Thinking fast and slow with deep learning and tree search.
  In *NeurIPS*, 2017.
* Bai et al. (2022)

  Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al.
  Constitutional ai: Harmlessness from ai feedback.
  *arXiv preprint arXiv:2212.08073*, 2022.
* Behrouz et al. (2025)

  Ali Behrouz, Peilin Zhong, and Vahab Mirrokni.
  Titans: Learning to memorize at test time.
  In *NeurIPS*, 2025.
* Berner et al. (2019)

  Christopher Berner, Greg Brockman, Brooke Chan, Vicki Cheung, Przemysław Debiak, Christy Dennison, David Farhi, Quirin Fischer, Shariq Hashme, Chris Hesse, et al.
  Dota 2 with large scale deep reinforcement learning.
  *arXiv preprint arXiv:1912.06680*, 2019.
* Boyd & Vandenberghe (2004)

  Stephen Boyd and Lieven Vandenberghe.
  *Convex optimization*.
  Cambridge university press, 2004.
* Brown et al. (2020)

  Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al.
  Language models are few-shot learners.
  *arXiv preprint ArXiv:2005.14165*, 2020.
* Bubeck (2015)

  Sébastien Bubeck.
  Convex optimization: Algorithms and complexity.
  *Foundations and Trends® in Machine Learning*, 2015.
* Cao et al. (2025)

  Meng Cao, Shuyuan Zhang, Xiao-Wen Chang, and Doina Precup.
  Scar: Shapley credit assignment for more efficient rlhf.
  *arXiv preprint arXiv:2505.20417*, 2025.
* Chan et al. (2024)

  Alex J Chan, Hao Sun, Samuel Holt, and Mihaela Van Der Schaar.
  Dense reward for free in reinforcement learning from human feedback.
  In *ICML*, 2024.
* Chen et al. (2025a)

  Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, et al.
  Minimax-m1: Scaling test-time compute efficiently with lightning attention.
  *arXiv preprint arXiv:2506.13585*, 2025a.
* Chen et al. (2022)

  Bei Chen, Fengji Zhang, Anh Nguyen, Daoguang Zan, Zeqi Lin, Jian-Guang Lou, and Weizhu Chen.
  Codet: Code generation with generated tests.
  In *ICLR*, 2022.
* Chen et al. (2025b)

  Howard Chen, Noam Razin, Karthik Narasimhan, and Danqi Chen.
  Retaining by doing: The role of on-policy data in mitigating forgetting.
  *arXiv preprint arXiv:2510.18874*, 2025b.
* Chen et al. (2021a)

  Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Misha Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch.
  Decision transformer: Reinforcement learning via sequence modeling.
  In *NeurIPS*, 2021a.
* Chen et al. (2021b)

  Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al.
  Evaluating large language models trained on code.
  *arXiv preprint arXiv:2107.03374*, 2021b.
* Chen et al. (2025c)

  Wentse Chen, Jiayu Chen, Fahim Tajwar, Hao Zhu, Xintong Duan, Ruslan Salakhutdinov, and Jeff Schneider.
  Retrospective in-context learning for temporal credit assignment with large language models.
  In *NeurIPS*, 2025c.
* Chiang et al. (2024)

  Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Banghua Zhu, Hao Zhang, Michael Jordan, Joseph E Gonzalez, et al.
  Chatbot arena: An open platform for evaluating llms by human preference.
  In *ICML*, 2024.
* Choi et al. (2022)

  Eunbi Choi, Yongrae Jo, Joel Jang, and Minjoon Seo.
  Prompt injection: Parameterization of fixed inputs.
  *arXiv preprint arXiv:2206.11349*, 2022.
* Cui et al. (2025)

  Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al.
  Process reinforcement through implicit rewards.
  *arXiv preprint arXiv:2502.01456*, 2025.
* Diaz-Bone et al. (2025)

  Leander Diaz-Bone, Marco Bagatella, Jonas Hübotter, and Andreas Krause.
  Discover: Automated curricula for sparse-reward reinforcement learning.
  In *NeurIPS*, 2025.
* Dolan & Moré (2002)

  Elizabeth D Dolan and Jorge J Moré.
  Benchmarking optimization software with performance profiles.
  *Mathematical programming*, 91(2), 2002.
* Dou et al. (2024)

  Zi-Yi Dou, Cheng-Fu Yang, Xueqing Wu, Kai-Wei Chang, and Nanyun Peng.
  Re-rest: Reflection-reinforced self-training for language agents.
  In *EMNLP*, 2024.
* El-Kishky et al. (2025)

  Ahmed El-Kishky, Alexander Wei, Andre Saraiva, Borys Minaiev, Daniel Selsam, David Dohan, Francis Song, Hunter Lightman, Ignasi Clavera, Jakub Pachocki, et al.
  Competitive programming with large reasoning models.
  *arXiv preprint arXiv:2502.06807*, 2025.
* Eyuboglu et al. (2026)

  Sabri Eyuboglu, Ryan Ehrlich, Simran Arora, Neel Guha, Dylan Zinsley, Emily Liu, Will Tennien, Atri Rudra, James Zou, Azalia Mirhoseini, et al.
  Cartridges: Lightweight and general-purpose long context representations via self-study.
  In *ICLR*, 2026.
* Feng et al. (2024)

  Kehua Feng, Keyan Ding, Weijie Wang, Xiang Zhuang, Zeyuan Wang, Ming Qin, Yu Zhao, Jianhua Yao, Qiang Zhang, and Huajun Chen.
  Sciknoweval: Evaluating multi-level scientific knowledge of large language models.
  *arXiv preprint arXiv:2406.09098*, 2024.
* Gehring et al. (2025)

  Jonas Gehring, Kunhao Zheng, Jade Copet, Vegard Mella, Quentin Carbonneaux, Taco Cohen, and Gabriel Synnaeve.
  Rlef: Grounding code llms in execution feedback with reinforcement learning.
  In *ICML*, 2025.
* Goyal et al. (2019)

  Prasoon Goyal, Scott Niekum, and Raymond J Mooney.
  Using natural language for reward shaping in reinforcement learning.
  In *IJCAI*, 2019.
* Gu et al. (2024)

  Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang.
  Minillm: Knowledge distillation of large language models.
  2024.
* Guha et al. (2026)

  Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, et al.
  Openthoughts: Data recipes for reasoning models.
  In *ICLR*, 2026.
* Guo et al. (2025)

  Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al.
  Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.
  *arXiv preprint arXiv:2501.12948*, 2025.
* Haarnoja et al. (2018)

  Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine.
  Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor.
  In *ICML*, 2018.
* Hardt & Sun (2024)

  Moritz Hardt and Yu Sun.
  Test-time training on nearest neighbors for large language models.
  In *ICLR*, 2024.
* Hatamizadeh et al. (2026)

  Ali Hatamizadeh, Syeda Nahida Akter, Shrimai Prabhumoye, Jan Kautz, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, and Yejin Choi.
  Rlp: Reinforcement as a pretraining objective.
  In *ICLR*, 2026.
* Hinton et al. (2015)

  Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.
  Distilling the knowledge in a neural network.
  *arXiv preprint arXiv:1503.02531*, 2015.
* Huang et al. (2026)

  Chengsong Huang, Wenhao Yu, Xiaoyang Wang, Hongming Zhang, Zongxia Li, Ruosen Li, Jiaxin Huang, Haitao Mi, and Dong Yu.
  R-zero: Self-evolving reasoning llm from zero data.
  In *ICLR*, 2026.
* Hübotter et al. (2026)

  Jonas Hübotter, Patrik Wolf, Alexander Shevchenko, Dennis Jüni, Andreas Krause, and Gil Kur.
  Specialization after generalization: Towards understanding test-time training in foundation models.
  In *ICLR*, 2026.
* Hübotter et al. (2025a)

  Jonas Hübotter, Sascha Bongni, Ido Hakimi, and Andreas Krause.
  Efficiently learning at test-time: Active fine-tuning of llms.
  In *ICLR*, 2025a.
* Hübotter et al. (2025b)

  Jonas Hübotter, Leander Diaz-Bone, Ido Hakimi, Andreas Krause, and Moritz Hardt.
  Learning on the job: Test-time curricula for targeted reinforcement learning.
  *arXiv preprint arXiv:2510.04786*, 2025b.
* Jaech et al. (2024)

  Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al.
  Openai o1 system card.
  *arXiv preprint arXiv:2412.16720*, 2024.
* Jain et al. (2025)

  Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica.
  Livecodebench: Holistic and contamination free evaluation of large language models for code.
  In *ICLR*, 2025.
* Kaelbling et al. (1998)

  Leslie Pack Kaelbling, Michael L Littman, and Anthony R Cassandra.
  Planning and acting in partially observable stochastic domains.
  *Artificial intelligence*, 101(1-2), 1998.
* Kazemnejad et al. (2025)

  Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, and Nicolas Le Roux.
  Vineppo: Refining credit assignment in rl training of llms.
  In *ICML*, 2025.
* Khatri et al. (2026)

  Devvrit Khatri, Lovish Madaan, Rishabh Tiwari, Rachit Bansal, Sai Surya Duvvuri, Manzil Zaheer, Inderjit S Dhillon, David Brandfonbrener, and Rishabh Agarwal.
  The art of scaling reinforcement learning compute for llms.
  In *ICLR*, 2026.
* Kim & Rush (2016)

  Yoon Kim and Alexander M Rush.
  Sequence-level knowledge distillation.
  In *EMNLP*, 2016.
* Kimi et al. (2025)

  Kimi, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al.
  Kimi k1.5: Scaling reinforcement learning with llms.
  *arXiv preprint arXiv:2501.12599*, 2025.
* Kujanpää et al. (2025)

  Kalle Kujanpää, Pekka Marttinen, Harri Valpola, and Alexander Ilin.
  Efficient knowledge injection in LLMs via self-distillation.
  *TMLR*, 2025.
* Kwon et al. (2023)

  Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica.
  Efficient memory management for large language model serving with pagedattention.
  In *PSIGOPS*, 2023.
* Lambert et al. (2025)

  Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al.
  Tulu 3: Pushing frontiers in open language model post-training.
  In *COLM*, 2025.
* Le et al. (2022)

  Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven Chu Hong Hoi.
  Coderl: Mastering code generation through pretrained models and deep reinforcement learning.
  In *NeurIPS*, 2022.
* Lee et al. (2024)

  Kyungjae Lee, Dasol Hwang, Sunghyun Park, Youngsoo Jang, and Moontae Lee.
  Reinforcement learning from reflective feedback (rlrf): Aligning and improving llms via fine-grained self-reflection.
  *arXiv preprint arXiv:2403.14238*, 2024.
* Lee et al. (2025)

  Yoonho Lee, Joseph Boen, and Chelsea Finn.
  Feedback descent: Open-ended text optimization via pairwise comparison.
  *arXiv preprint arXiv:2511.07919*, 2025.
* Levine (2018)

  Sergey Levine.
  Reinforcement learning and control as probabilistic inference: Tutorial and review.
  *arXiv preprint arXiv:1805.00909*, 2018.
* Li et al. (2025)

  Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E Gonzalez, and Ion Stoica.
  From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline.
  In *ICML*, 2025.
* Lightman et al. (2023)

  Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe.
  Let’s verify step by step.
  In *ICLR*, 2023.
* Liu et al. (2025a)

  Grace Liu, Michael Tang, and Benjamin Eysenbach.
  A single goal is all you need: Skills and exploration emerge from contrastive rl without rewards, demonstrations, or subgoals.
  In *ICLR*, 2025a.
* Liu et al. (2023)

  Hao Liu, Carmelo Sferrazza, and Pieter Abbeel.
  Chain of hindsight aligns language models with feedback.
  *arXiv preprint arXiv:2302.02676*, 2023.
* Liu et al. (2025b)

  Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin.
  Understanding r1-zero-like training: A critical perspective.
  In *COLM*, 2025b.
* Lu & Thinking Machines Lab (2025)

  Kevin Lu and Thinking Machines Lab.
  On-policy distillation.
  *Thinking Machines Lab: Connectionism*, 2025.
  URL <https://thinkingmachines.ai/blog/on-policy-distillation>.
* Luo et al. (2025)

  Renjie Luo, Zichen Liu, Xiangyan Liu, Chao Du, Min Lin, Wenhu Chen, Wei Lu, and Tianyu Pang.
  Language models can learn from verbal feedback without scalar rewards.
  *arXiv preprint arXiv:2509.22638*, 2025.
* Madaan et al. (2023)

  Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al.
  Self-refine: Iterative refinement with self-feedback.
  In *NeurIPS*, 2023.
* Mitra & Ulukus (2025)

  Purbesh Mitra and Sennur Ulukus.
  Semantic soft bootstrapping: Long context reasoning in llms without reinforcement learning.
  *arXiv preprint arXiv:2512.05105*, 2025.
* Mnih et al. (2015)

  Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, et al.
  Human-level control through deep reinforcement learning.
  *Nature*, 518(7540), 2015.
* Muennighoff et al. (2025)

  Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori B Hashimoto.
  s1: Simple test-time scaling.
  In *EMNLP*, 2025.
* Ng et al. (2000)

  Andrew Y Ng, Stuart Russell, et al.
  Algorithms for inverse reinforcement learning.
  In *ICML*, 2000.
* Novikov et al. (2025)

  Alexander Novikov, Ngân Vũ, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco JR Ruiz, Abbas Mehrabian, et al.
  Alphaevolve: A coding agent for scientific and algorithmic discovery.
  *arXiv preprint arXiv:2506.13131*, 2025.
* Olmo et al. (2025)

  Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, et al.
  Olmo 3.
  *arXiv preprint arXiv:2512.13961*, 2025.
* Peng et al. (2019)

  Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine.
  Advantage-weighted regression: Simple and scalable off-policy reinforcement learning.
  *arXiv preprint arXiv:1910.00177*, 2019.
* Qwen et al. (2024)

  Qwen, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, et al.
  Qwen2.5 technical report.
  *arXiv preprint arXiv:2412.15115*, 2024.
* Rafailov et al. (2023)

  Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn.
  Direct preference optimization: Your language model is secretly a reward model.
  In *NeurIPS*, 2023.
* Romero et al. (2015)

  Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua Bengio.
  Fitnets: Hints for thin deep nets.
  In *ICLR*, 2015.
* Ross et al. (2011)

  Stéphane Ross, Geoffrey Gordon, and Drew Bagnell.
  A reduction of imitation learning and structured prediction to no-regret online learning.
  In *AISTATS*, 2011.
* Samadi et al. (2025)

  Mehrzad Samadi, Aleksander Ficek, Sean Narenthiran, Siddhartha Jain, Wasi Uddin Ahmad, Somshubra Majumdar, Vahid Noroozi, and Boris Ginsburg.
  Scaling test-time compute to achieve ioi gold medal with open-weight models.
  *arXiv preprint arXiv:2510.14232*, 2025.
* Sanh et al. (2019)

  Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf.
  Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter.
  *arXiv preprint arXiv:1910.01108*, 2019.
* Schaul et al. (2015)

  Tom Schaul, Daniel Horgan, Karol Gregor, and David Silver.
  Universal value function approximators.
  In *ICML*, 2015.
* Scheurer et al. (2023)

  Jérémy Scheurer, Jon Ander Campos, Tomasz Korbak, Jun Shern Chan, Angelica Chen, Kyunghyun Cho, and Ethan Perez.
  Training language models with language feedback at scale.
  *arXiv preprint arXiv:2303.16755*, 2023.
* Schulman et al. (2015)

  John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz.
  Trust region policy optimization.
  In *ICML*, 2015.
* Schulman et al. (2016)

  John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel.
  High-dimensional continuous control using generalized advantage estimation.
  In *ICLR*, 2016.
* Schulman et al. (2017)

  John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
  Proximal policy optimization algorithms.
  *arXiv preprint arXiv:1707.06347*, 2017.
* Setlur et al. (2025)

  Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, and Aviral Kumar.
  Rewarding progress: Scaling automated process verifiers for llm reasoning.
  In *ICLR*, 2025.
* Shao et al. (2024)

  Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al.
  Deepseekmath: Pushing the limits of mathematical reasoning in open language models.
  *arXiv preprint arXiv:2402.03300*, 2024.
* Shenfeld et al. (2026)

  Idan Shenfeld, Jyothish Pari, and Pulkit Agrawal.
  Rl’s razor: Why online reinforcement learning forgets less.
  In *ICLR*, 2026.
* Sheng et al. (2025)

  Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu.
  Hybridflow: A flexible and efficient rlhf framework.
  In *EuroSys*, 2025.
* Shinn et al. (2023)

  Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao.
  Reflexion: Language agents with verbal reinforcement learning.
  In *NeurIPS*, 2023.
* Silver et al. (2016)

  David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al.
  Mastering the game of go with deep neural networks and tree search.
  *Nature*, 529(7587), 2016.
* Silver et al. (2017)

  David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, et al.
  Mastering chess and shogi by self-play with a general reinforcement learning algorithm.
  *arXiv preprint arXiv:1712.01815*, 2017.
* Snell et al. (2022)

  Charlie Snell, Dan Klein, and Ruiqi Zhong.
  Learning by distilling context.
  *arXiv preprint arXiv:2209.15189*, 2022.
* Stephan et al. (2024)

  Moritz Stephan, Alexander Khazatsky, Eric Mitchell, Annie S Chen, Sheryl Hsu, Archit Sharma, and Chelsea Finn.
  Rlvf: Learning from verbal feedback without overgeneralization.
  In *ICML*, 2024.
* Sun et al. (2020)

  Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei Efros, and Moritz Hardt.
  Test-time training with self-supervision for generalization under distribution shifts.
  In *ICML*, 2020.
* Sun et al. (2025)

  Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, et al.
  Learning to (learn at test time): Rnns with expressive hidden states.
  In *ICML*, 2025.
* Sutton & Barto (1998)

  Richard S Sutton and Andrew G Barto.
  *Reinforcement learning: An introduction*.
  MIT press, 1998.
* Tandon et al. (2025)

  Arnuv Tandon, Karan Dalal, Xinhao Li, Daniel Koceja, Marcel Rød, Sam Buchanan, Xiaolong Wang, Jure Leskovec, Sanmi Koyejo, Tatsunori Hashimoto, et al.
  End-to-end test-time training for long context.
  *arXiv preprint arXiv:2512.23675*, 2025.
* Tang et al. (2023)

  Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, Boxi Cao, and Le Sun.
  Toolalpaca: Generalized tool learning for language models with 3000 simulated cases.
  *arXiv preprint arXiv:2306.05301*, 2023.
* Urcelay et al. (2026)

  Belen Martin Urcelay, Andreas Krause, and Giorgia Ramponi.
  From words to rewards: Leveraging natural language for reinforcement learning.
  In *TMLR*, 2026.
* Vaswani et al. (2017)

  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.
  Attention is all you need.
  In *NeurIPS*, 2017.
* Wainwright & Jordan (2008)

  Martin J Wainwright and Michael I Jordan.
  Graphical models, exponential families, and variational inference.
  *Foundations and Trends® in Machine Learning*, 2008.
* Wang et al. (2026)

  Hanyang Wang, Lu Wang, Chaoyun Zhang, Tianjun Mao, Si Qin, Qingwei Lin, Saravan Rajmohan, and Dongmei Zhang.
  Text2grad: Reinforcement learning from natural language feedback.
  In *ICLR*, 2026.
* Wang et al. (2024a)

  Peiyi Wang, Lei Li, Zhihong Shao, RX Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui.
  Math-shepherd: Verify and reinforce llms step-by-step without human annotations.
  In *ACL*, 2024a.
* Wang et al. (2025)

  Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, et al.
  Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning.
  In *NeurIPS*, 2025.
* Wang et al. (2024b)

  Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al.
  Mmlu-pro: A more robust and challenging multi-task language understanding benchmark.
  In *NeurIPS*, 2024b.
* Wei et al. (2022)

  Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al.
  Chain-of-thought prompting elicits reasoning in large language models.
  In *NeurIPS*, 2022.
* Williams (1992)

  Ronald J Williams.
  Simple statistical gradient-following algorithms for connectionist reinforcement learning.
  *Machine learning*, 8(3), 1992.
* Xie et al. (2020)

  Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V Le.
  Self-training with noisy student improves imagenet classification.
  In *CVPR*, 2020.
* Xie et al. (2024)

  Tianbao Xie, Siheng Zhao, Chen Henry Wu, Yitao Liu, Qian Luo, Victor Zhong, Yanchao Yang, and Tao Yu.
  Text2reward: Reward shaping with language models for reinforcement learning.
  In *ICLR*, 2024.
* Yang et al. (2025)

  An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al.
  Qwen3 technical report.
  *arXiv preprint arXiv:2505.09388*, 2025.
* Yao et al. (2025)

  Feng Yao, Liyuan Liu, Dinghuai Zhang, Chengyu Dong, Jingbo Shang, and Jianfeng Gao.
  Your efficient rl framework secretly brings you off-policy rl training, 2025.
  URL <https://fengyao.notion.site/off-policy-rl>.
* Yao et al. (2024)

  Weiran Yao, Shelby Heinecke, Juan Carlos Niebles, Zhiwei Liu, Yihao Feng, Le Xue, Rithesh Murthy, Zeyuan Chen, Jianguo Zhang, Devansh Arpit, et al.
  Retroformer: Retrospective large language agents with policy gradient optimization.
  In *ICLR*, 2024.
* Yu et al. (2025)

  Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al.
  Dapo: An open-source llm reinforcement learning system at scale.
  In *NeurIPS*, 2025.
* Yue et al. (2025)

  Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang.
  Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model?
  In *NeurIPS*, 2025.
* Yuksekgonul et al. (2025)

  Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Pan Lu, Zhi Huang, Carlos Guestrin, and James Zou.
  Optimizing generative ai by backpropagating language model feedback.
  *Nature*, 639:609–616, 2025.
* Yuksekgonul et al. (2026)

  Mert Yuksekgonul, Daniel Koceja, Xinhao Li, Federico Bianchi, Jed McCaleb, Xiaolong Wang, Jan Kautz, Yejin Choi, James Zou, Carlos Guestrin, et al.
  Learning to discover at test time.
  *arXiv preprint arXiv:2601.16175*, 2026.
* Zelikman et al. (2022)

  Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D Goodman.
  Star: Bootstrapping reasoning with reasoning.
  In *NeurIPS*, 2022.
* Zhang et al. (2025)

  Kai Zhang, Xiangchao Chen, Bo Liu, Tianci Xue, Zeyi Liao, Zhihan Liu, Xiyao Wang, Yuting Ning, Zhaorun Chen, Xiaohan Fu, et al.
  Agent learning via early experience.
  *arXiv preprint arXiv:2510.08558*, 2025.
* Zhang et al. (2023)

  Tianjun Zhang, Fangchen Liu, Justin Wong, Pieter Abbeel, and Joseph E Gonzalez.
  The wisdom of hindsight makes language models better instruction followers.
  In *ICML*, 2023.
* Zhao et al. (2025)

  Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, and Gao Huang.
  Absolute zero: Reinforced self-play reasoning with zero data.
  In *NeurIPS*, 2025.
* Zheng et al. (2025a)

  Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al.
  Group sequence policy optimization.
  *arXiv preprint arXiv:2507.18071*, 2025a.
* Zheng et al. (2025b)

  Tianyu Zheng, Tianshun Xing, Qingshui Gu, Taoran Liang, Xingwei Qu, Xin Zhou, Yizhi Li, Zhoufutu Wen, Chenghua Lin, Wenhao Huang, et al.
  First return, entropy-eliciting explore.
  *arXiv preprint arXiv:2507.07017*, 2025b.
* Zhou et al. (2023)

  Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou.
  Instruction-following evaluation for large language models.
  *arXiv preprint arXiv:2311.07911*, 2023.
* Zhou et al. (2026)

  Xiangxin Zhou, Zichen Liu, Anya Sims, Haonan Wang, Tianyu Pang, Chongxuan Li, Liang Wang, Min Lin, and Chao Du.
  Reinforcing general reasoning without verifiers.
  In *ICLR*, 2026.
* Ziebart et al. (2008)

  Brian D Ziebart, Andrew L Maas, J Andrew Bagnell, Anind K Dey, et al.
  Maximum entropy inverse reinforcement learning.
  In *AAAI*, 2008.
* Zuo et al. (2025)

  Yuxin Zuo, Kaiyan Zhang, Shang Qu, Li Sheng, Xuekai Zhu, Biqing Qi, Youbang Sun, Ganqu Cui, Ning Ding, and Bowen Zhou.
  Ttrl: Test-time reinforcement learning.
  In *NeurIPS*, 2025.

## Contents

section.1table.caption.4section.2subsection.2.1subsection.2.2subsection.2.3section.3subsection.3.1subsection.3.1subsection.3.2subsection.3.3section.4section.4subsection.4.1subsection.4.2subsection.4.3subsection.4.4subsection.4.4subsection.4.5subsection.4.6subsection.4.6subsection.4.6subsection.4.6section.5subsection.5.1subsection.5.2figure.caption.20figure.caption.20section.6subsection.6.1subsection.6.2subsection.6.3subsection.6.4section.7section.7section.7appendix.Asubsection.A.1subsection.A.2subsection.A.3appendix.Bsubsection.B.1subsection.B.2subsection.B.3appendix.Cappendix.Cappendix.Cappendix.Cappendix.Csubsection.C.1subsection.C.1equation.22appendix.Dsubsection.D.1subsection.D.2subsubsection.D.2.1subsubsection.D.2.2subsubsection.D.2.3subsection.D.3appendix.Esubsection.E.1subsection.E.2subsubsection.E.2.1subsection.E.3appendix.Fsubsection.F.1subsection.F.2subsection.F.3subsection.F.4

## Appendix A Implementation of SDPO

The following pseudocode in [Figure˜14](#A1.F14 "In Appendix A Implementation of SDPO ‣ Reinforcement Learning via Self-Distillation") outlines the implementation of SDPO:

[⬇](data:text/plain;base64,CmRlZiBjb21wdXRlX3NkcG9fbG9zcyhiYXRjaCwgdGVhY2hlcl9jb250ZXh0LCBsb3NzX21hc2spOgoiIiIKQ29tcHV0ZXMgcHJvYmFiaWxpdGllcyBvZiByZXNwb25zZSB5IHVuZGVyIHRoZSBzZWxmLXRlYWNoZXIKYW5kIHRoZSBwZXItbG9naXQgU0RQTyBsb3NzLgoiIiIKIyBDb21wdXRlIG1vZGVsIHByb2JhYmlsaXRpZXMgZm9yIHJlc3BvbnNlIHkKbG9ncHJvYnNfc3R1ZGVudCA9IGNvbXB1dGVfbG9nX3Byb2IoYmF0Y2gpICMgKFQsVikKcHJvYnNfc3R1ZGVudCA9IGxvZ3Byb2JzX3N0dWRlbnQuZXhwKCkgIyAoVCxWKQpccGFyIyBDb21wdXRlIHNlbGYtdGVhY2hlciBwcm9iYWJpbGl0aWVzIGZvciByZXNwb25zZSB5CnRlYWNoZXJfYmF0Y2ggPSByZXByb21wdChiYXRjaCwgdGVhY2hlcl9jb250ZXh0KQpsb2dwcm9ic190ZWFjaGVyID0gY29tcHV0ZV9sb2dfcHJvYih0ZWFjaGVyX2JhdGNoKS5kZXRhY2goKSAjIChULFYpClxwYXIjIENvbXB1dGUgU0RQTyBsb3NzOiBwZXItdG9rZW4gZGl2ZXJnZW5jZQpwZXJfdG9rZW5fbG9zcyA9IGRpdmVyZ2VuY2UobG9ncHJvYnNfc3R1ZGVudCwgbG9ncHJvYnNfdGVhY2hlcikgIyAoVCwpCnJldHVybiBhZ2dfbG9zcyhwZXJfdG9rZW5fbG9zcywgbG9zc19tYXNrLCBsb3NzX2FnZ19tb2RlPSJ0b2tlbi1tZWFuIikK)

def compute\_sdpo\_loss(batch, teacher\_context, loss\_mask):
"""
Computes probabilities of response y under the self-teacher
and the per-logit SDPO loss.
"""
# Compute model probabilities for response y
logprobs\_student = compute\_log\_prob(batch) # (T,V)
probs\_student = logprobs\_student.exp() # (T,V)
\par# Compute self-teacher probabilities for response y
teacher\_batch = reprompt(batch, teacher\_context)
logprobs\_teacher = compute\_log\_prob(teacher\_batch).detach() # (T,V)
\par# Compute SDPO loss: per-token divergence
per\_token\_loss = divergence(logprobs\_student, logprobs\_teacher) # (T,)
return agg\_loss(per\_token\_loss, loss\_mask, loss\_agg\_mode="token-mean")

Figure 14: The pseudo-code of SDPO within a standard RL training pipeline. Omitted here is the filtering to top-KK logprobs for student and teacher (including a tail term) as described in [Section˜A.2](#A1.SS2 "A.2 Approximate Logit Distillation ‣ Appendix A Implementation of SDPO ‣ Reinforcement Learning via Self-Distillation"). Further, we omit here any importance sampling weights to correct for off-policy data. reprompt modifies the batch to incorporate teacher context (i.e., rich feedback). divergence implements any per-token divergence such as reverse-KL, forward-KL, or Jensen-Shannon.

In the following, we provide further details on:

* •

  Teacher regularization ([Section˜A.1](#A1.SS1 "A.1 Regularized teacher ‣ Appendix A Implementation of SDPO ‣ Reinforcement Learning via Self-Distillation"))
* •

  Approximating logit-distillation with the top-KK logits for saving GPU memory ([Section˜A.2](#A1.SS2 "A.2 Approximate Logit Distillation ‣ Appendix A Implementation of SDPO ‣ Reinforcement Learning via Self-Distillation"))
* •

  Generalizing PPO-style policy gradient algorithms to logit-level advantages ([Section˜A.3](#A1.SS3 "A.3 Off-Policy Training: Generalization to Logit-Level Losses ‣ Appendix A Implementation of SDPO ‣ Reinforcement Learning via Self-Distillation"))

To disambiguate the notation of the self-teacher, we use qθ(⋅∣x,f):=πθ(⋅∣reprompt(x,f))q\_{\theta}(\cdot\mid x,f):=\pi\_{\theta}(\cdot\mid\mathrm{reprompt}(x,f)) in the following.
Here, reprompt denotes the reprompt template of the self-teacher.

### A.1 Regularized teacher

In contrast to standard distillation, the teacher in SDPO changes throughout training. This bootstrapping enables the teacher to improve, but it may also lead to training instability.
To stabilize training, we seek to prevent the teacher qq from quickly diverging from the initial teacher qθref\smash{q\_{\theta\_{{\mathrm{ref}}}}}.
We can achieve this by placing an explicit trust-region constraint on qq (Schulman et al., [2015](#bib.bib78); Peng et al., [2019](#bib.bib69)), that is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑tKL(q(yt∣x,f,y<t)∥qθref(yt∣x,f,y<t))≤ϵ,ϵ>0.\sum\_{t}\mathrm{KL}\left(q(y\_{t}\mid x,f,y\_{<t})\|q\_{\theta\_{{\mathrm{ref}}}}(y\_{t}\mid x,f,y\_{<t})\right)\leq\epsilon,\quad\epsilon>0. |  | (5) |

This trust-region can be implemented in two ways:

1. 1.

   Explicit trust-region: We can define the teacher as the policy closest to qθq\_{\theta} while satisfying the trust-region constraint.
   This teacher can be expressed as

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | q​(yt∣x,f,y<t)∝exp⁡((1−α)​log⁡qθref​(yt∣x,f,y<t)+α​log⁡qθ​(yt∣x,f,y<t)),q(y\_{t}\mid x,f,y\_{<t})\propto\exp\!\big((1-\alpha)\log q\_{\theta\_{{\mathrm{ref}}}}(y\_{t}\mid x,f,y\_{<t})+\alpha\log q\_{\theta}(y\_{t}\mid x,f,y\_{<t})\big), |  | (6) |

   with α∈(0,1)\alpha\in(0,1) the inverse Lagrange multiplier for the trust-region constraint.
   We include a full derivation in [Section˜B.2](#A2.SS2 "B.2 Trust-region Teacher ‣ Appendix B Theoretical Analysis ‣ Reinforcement Learning via Self-Distillation").
   We can plug this explicitly constrained teacher directly into the SDPO objective.
2. 2.

   Exponential moving average (EMA): Alternatively, we can stabilize the teacher’s parameters directly; parameterizing qθ′q\_{\theta^{\prime}} by θ′\theta^{\prime} and updating as θ′←(1−α)​θ′+α​θ\theta^{\prime}\leftarrow(1-\alpha)\theta^{\prime}+\alpha\theta with α∈(0,1)\alpha\in(0,1).
   Under mild smoothness assumptions, this EMA teacher remains implicitly within a trust-region around the initial teacher (cf. [Section˜B.3](#A2.SS3 "B.3 EMA Teacher as an Implicit Trust Region ‣ Appendix B Theoretical Analysis ‣ Reinforcement Learning via Self-Distillation")).

Note that each implementation has a different practical advantage:
The EMA teacher requires additional GPU memory for θ′\theta^{\prime} yet does not introduce any runtime overhead.
In contrast, the trust-region teacher requires an additional log-prob computation with qθref\smash{q\_{\theta\_{{\mathrm{ref}}}}} yet does not require additional GPU memory if θref{\theta\_{{\mathrm{ref}}}} is used for explicit KL regularization.

### A.2 Approximate Logit Distillation

To save GPU memory, we perform distillation only on the top-KK tokens predicted by the student:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒSDPO​(θ)\displaystyle\mathcal{L}\_{\mathrm{SDPO}}(\theta) | =𝔼y∼πθ(⋅∣x)∑t=1TKL(πθ(yt∣x,y<t)∥stopgrad(qθ(yt∣x,f,y<t)))\displaystyle=\mathbb{E}\_{y\sim\pi\_{\theta}(\cdot\mid x)}{}\sum\_{t=1}^{T}\mathrm{KL}(\pi\_{\theta}(y\_{t}\mid x,y\_{<t})\|\mathrm{stopgrad}(q\_{\theta}(y\_{t}\mid x,f,y\_{<t}))) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ≈Ey∼πθ(⋅∣x)​∑t=1T∑yt∈topK​(πθ)πθ​(yt∣x,y<t)⋅log⁡πθ​(yt∣x,y<t)stopgrad​(qθ​(yt∣x,f,y<t))+(1−∑yt∈topK​(πθ)πθ​(yt∣x,y<t))⋅log⁡1−∑yt∈topK​(πθ)πθ​(yt∣x,y<t)stopgrad​(1−∑yt∈topK​(πθ)qθ​(yt∣x,f,y<t))⏟tail\displaystyle\approx\begin{multlined}E\_{y\sim\pi\_{\theta}(\cdot\mid x)}\sum\_{t=1}^{T}\sum\_{y\_{t}\in\mathrm{top}\_{K}(\pi\_{\theta})}\pi\_{\theta}(y\_{t}\mid x,y\_{<t})\cdot\log\frac{\pi\_{\theta}(y\_{t}\mid x,y\_{<t})}{\mathrm{stopgrad}(q\_{\theta}(y\_{t}\mid x,f,y\_{<t}))}\\ +\underbrace{\Big(1-\textstyle\sum\_{y\_{t}\in\mathrm{top}\_{K}(\pi\_{\theta})}\pi\_{\theta}(y\_{t}\mid x,y\_{<t})\Big)\cdot\log\frac{1-\textstyle\sum\_{y\_{t}\in\mathrm{top}\_{K}(\pi\_{\theta})}\pi\_{\theta}(y\_{t}\mid x,y\_{<t})}{\mathrm{stopgrad}\Big(1-\textstyle\sum\_{y\_{t}\in\mathrm{top}\_{K}(\pi\_{\theta})}q\_{\theta}(y\_{t}\mid x,f,y\_{<t})\Big)}}\_{\text{tail}}\end{multlined}E\_{y\sim\pi\_{\theta}(\cdot\mid x)}\sum\_{t=1}^{T}\sum\_{y\_{t}\in\mathrm{top}\_{K}(\pi\_{\theta})}\pi\_{\theta}(y\_{t}\mid x,y\_{<t})\cdot\log\frac{\pi\_{\theta}(y\_{t}\mid x,y\_{<t})}{\mathrm{stopgrad}(q\_{\theta}(y\_{t}\mid x,f,y\_{<t}))}\\ +\underbrace{\Big(1-\textstyle\sum\_{y\_{t}\in\mathrm{top}\_{K}(\pi\_{\theta})}\pi\_{\theta}(y\_{t}\mid x,y\_{<t})\Big)\cdot\log\frac{1-\textstyle\sum\_{y\_{t}\in\mathrm{top}\_{K}(\pi\_{\theta})}\pi\_{\theta}(y\_{t}\mid x,y\_{<t})}{\mathrm{stopgrad}\Big(1-\textstyle\sum\_{y\_{t}\in\mathrm{top}\_{K}(\pi\_{\theta})}q\_{\theta}(y\_{t}\mid x,f,y\_{<t})\Big)}}\_{\text{tail}} |  | (9) |

Here, the top-KK is with respect to student.
Without top-KK distillation, we would have to keep two copies of logits in memory: one for teacher and student each.
Top-KK distillation avoids virtually any memory overhead without impacting performance significantly, since most tokens of the vocabulary are not informative at a given time.

### A.3 Off-Policy Training: Generalization to Logit-Level Losses

PPO-style clipping (Schulman et al., [2017](#bib.bib80)) with truncated importance sampling (Yao et al., [2025](#bib.bib107)), clip-higher (Yu et al., [2025](#bib.bib109)), fixed length normalization (Liu et al., [2025b](#bib.bib59)):

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒtoken​(θ):=−1∑i=1G|yi|​∑i=1G∑t=1|yi|min⁡(wi,tTIS,ρ)​min⁡(wi,t​Ai,t,clip​(wi,t,1−εlow,1+εhigh)​Ai,t),\mathcal{L}\_{\mathrm{token}}(\theta):=-{\color[rgb]{1,.5,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,.5,0}\frac{1}{\sum\_{i=1}^{G}|y\_{i}|}}\sum\_{i=1}^{G}\sum\_{t=1}^{|y\_{i}|}{\color[rgb]{0.94921875,0.328125,0.35546875}\definecolor[named]{pgfstrokecolor}{rgb}{0.94921875,0.328125,0.35546875}\min\left(w^{\mathrm{TIS}}\_{i,t},\rho\right)}\min\left(w\_{i,t}A\_{i,t},\text{clip}(w\_{i,t},1-\varepsilon\_{\text{low}},1+{\color[rgb]{0.34765625,0.734375,0.16796875}\definecolor[named]{pgfstrokecolor}{rgb}{0.34765625,0.734375,0.16796875}\varepsilon\_{\text{high}}})A\_{i,t}\right), |  | (10) |

with wi,t:=πθ​(yi,t∣x,yi,<t)πθold​(yi,t∣x,yi,<t)w\_{i,t}:=\frac{\pi\_{\theta}(y\_{i,t}\mid x,y\_{i,<t})}{\pi\_{\theta\_{{\mathrm{old}}}}(y\_{i,t}\mid x,y\_{i,<t})}, wi,tTIS:=πθold​(yi,t∣x,yi,<t)πθoldrollout​(yi,t∣x,yi,<t)w^{\mathrm{TIS}}\_{i,t}:=\frac{\pi\_{\theta\_{{\mathrm{old}}}}(y\_{i,t}\mid x,y\_{i,<t})}{\pi\_{\theta\_{{\mathrm{old}}}}^{\mathrm{rollout}}(y\_{i,t}\mid x,y\_{i,<t})}, and Ai,tA\_{i,t} denotes the per-token advantage.

We extend this to a logit-level loss:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒlogit​(θ):=−1∑i=1G|yi|​∑i=1G∑t=1|yi|∑y^i,tmin⁡(πθold​(y^i,t∣x,yi,<t),ρ​πθoldrollout​(y^i,t∣x,yi,<t))min⁡(wi,t​(y^i,t)​Ai,t​(y^i,t),clip​(wi,t​(y^i,t),1−εlow,1+εhigh)​Ai,t​(y^i,t)),\begin{multlined}\mathcal{L}\_{\mathrm{logit}}(\theta):=-{\color[rgb]{1,.5,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,.5,0}\frac{1}{\sum\_{i=1}^{G}|y\_{i}|}}\sum\_{i=1}^{G}\sum\_{t=1}^{|y\_{i}|}{\color[rgb]{0.16796875,0.3125,0.66796875}\definecolor[named]{pgfstrokecolor}{rgb}{0.16796875,0.3125,0.66796875}\sum\_{\hat{y}\_{i,t}}}\ {\color[rgb]{0.94921875,0.328125,0.35546875}\definecolor[named]{pgfstrokecolor}{rgb}{0.94921875,0.328125,0.35546875}\min\left(\pi\_{\theta\_{{\mathrm{old}}}}(\hat{y}\_{i,t}\mid x,y\_{i,<t}),\rho\pi\_{\theta\_{{\mathrm{old}}}}^{\mathrm{rollout}}(\hat{y}\_{i,t}\mid x,y\_{i,<t})\right)}\\ \min\left(w\_{i,t}(\hat{y}\_{i,t})A\_{i,t}(\hat{y}\_{i,t}),\text{clip}(w\_{i,t}(\hat{y}\_{i,t}),1-\varepsilon\_{\text{low}},1+{\color[rgb]{0.34765625,0.734375,0.16796875}\definecolor[named]{pgfstrokecolor}{rgb}{0.34765625,0.734375,0.16796875}\varepsilon\_{\text{high}}})A\_{i,t}(\hat{y}\_{i,t})\right),\end{multlined}\mathcal{L}\_{\mathrm{logit}}(\theta):=-{\color[rgb]{1,.5,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,.5,0}\frac{1}{\sum\_{i=1}^{G}|y\_{i}|}}\sum\_{i=1}^{G}\sum\_{t=1}^{|y\_{i}|}{\color[rgb]{0.16796875,0.3125,0.66796875}\definecolor[named]{pgfstrokecolor}{rgb}{0.16796875,0.3125,0.66796875}\sum\_{\hat{y}\_{i,t}}}\ {\color[rgb]{0.94921875,0.328125,0.35546875}\definecolor[named]{pgfstrokecolor}{rgb}{0.94921875,0.328125,0.35546875}\min\left(\pi\_{\theta\_{{\mathrm{old}}}}(\hat{y}\_{i,t}\mid x,y\_{i,<t}),\rho\pi\_{\theta\_{{\mathrm{old}}}}^{\mathrm{rollout}}(\hat{y}\_{i,t}\mid x,y\_{i,<t})\right)}\\ \min\left(w\_{i,t}(\hat{y}\_{i,t})A\_{i,t}(\hat{y}\_{i,t}),\text{clip}(w\_{i,t}(\hat{y}\_{i,t}),1-\varepsilon\_{\text{low}},1+{\color[rgb]{0.34765625,0.734375,0.16796875}\definecolor[named]{pgfstrokecolor}{rgb}{0.34765625,0.734375,0.16796875}\varepsilon\_{\text{high}}})A\_{i,t}(\hat{y}\_{i,t})\right), |  | (11) |

where y^i,t\hat{y}\_{i,t} sums over all possible tokens at position tt for rollout ii (or the KK most likely under πθold\pi\_{\theta\_{{\mathrm{old}}}}, cf. [Section˜A.2](#A1.SS2 "A.2 Approximate Logit Distillation ‣ Appendix A Implementation of SDPO ‣ Reinforcement Learning via Self-Distillation")).
The TIS changes since we explicitly weight each logit by its probability under πθold\pi\_{\theta\_{{\mathrm{old}}}} rather than relying on a Monte Carlo estimate of the expectation over next-token predictions.
Here, Ai,t​(y^i,t)A\_{i,t}(\hat{y}\_{i,t}) is a per-logit advantage.

In our experiments for SDPO, we apply the TIS term on a token-level rather than logit-level.

## Appendix B Theoretical Analysis

This section is organized as follows:

* •

  [Section˜B.1](#A2.SS1 "B.1 Gradient Estimator ‣ Appendix B Theoretical Analysis ‣ Reinforcement Learning via Self-Distillation") derives the SDPO gradient from [Proposition˜2.1](#S2.Thmtheorem1 "Proposition 2.1. ‣ 2 SDPO: Self-Distillation Policy Optimization ‣ Reinforcement Learning via Self-Distillation").
* •

  [Section˜B.2](#A2.SS2 "B.2 Trust-region Teacher ‣ Appendix B Theoretical Analysis ‣ Reinforcement Learning via Self-Distillation") derives the trust-region regularized teacher discussed in [Section˜A.1](#A1.SS1 "A.1 Regularized teacher ‣ Appendix A Implementation of SDPO ‣ Reinforcement Learning via Self-Distillation").
* •

  [Section˜B.3](#A2.SS3 "B.3 EMA Teacher as an Implicit Trust Region ‣ Appendix B Theoretical Analysis ‣ Reinforcement Learning via Self-Distillation") shows that the EMA teacher, as discussed in [Section˜A.1](#A1.SS1 "A.1 Regularized teacher ‣ Appendix A Implementation of SDPO ‣ Reinforcement Learning via Self-Distillation"), implements a trust-region constraint.

To disambiguate the notation of the self-teacher, we use qθ(⋅∣x,f):=πθ(⋅∣reprompt(x,f))q\_{\theta}(\cdot\mid x,f):=\pi\_{\theta}(\cdot\mid\mathrm{reprompt}(x,f)) in the following.
Here, reprompt denotes the reprompt template of the self-teacher.

### B.1 Gradient Estimator

###### Proof of Proposition [2.1](#S2.Thmtheorem1 "Proposition 2.1. ‣ 2 SDPO: Self-Distillation Policy Optimization ‣ Reinforcement Learning via Self-Distillation").

In the following, we derive the gradient of ℒSDPO\mathcal{L}\_{\mathrm{SDPO}}.

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇θℒSDPO​(θ)\displaystyle\boldsymbol{\nabla}\_{\!\!\theta}\,\mathcal{L}\_{\mathrm{SDPO}}(\theta) | =∇θ∑t=1TKL(πθ(yt∣x,y<t)∥stopgrad(qθ(yt∣x,f,y<t)))\displaystyle=\boldsymbol{\nabla}\_{\!\!\theta}\,\sum\_{t=1}^{T}\mathrm{KL}(\pi\_{\theta}(y\_{t}\mid x,y\_{<t})\|\mathrm{stopgrad}(q\_{\theta}(y\_{t}\mid x,f,y\_{<t}))) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∇θ​∑t=1T∑ytπθ​(yt∣x,y<t)​log⁡(πθ​(yt∣x,y<t)stopgrad​(qθ​(yt∣x,f,y<t)))\displaystyle=\boldsymbol{\nabla}\_{\!\!\theta}\,\sum\_{t=1}^{T}\sum\_{y\_{t}}\pi\_{\theta}(y\_{t}\mid x,y\_{<t})\log\left(\frac{\pi\_{\theta}(y\_{t}\mid x,y\_{<t})}{\mathrm{stopgrad}(q\_{\theta}(y\_{t}\mid x,f,y\_{<t}))}\right) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Let At,k:=log⁡(stopgrad​(qθ​(yt∣x,f,y<t))πθ​(yt∣x,y<t))A\_{t,k}:=\log\left(\frac{\mathrm{stopgrad}(q\_{\theta}(y\_{t}\mid x,f,y\_{<t}))}{\pi\_{\theta}(y\_{t}\mid x,y\_{<t})}\right). Then, | | | | |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =−∇θ​∑t=1T∑ytπθ​(yt∣x,y<t)​At,k\displaystyle=-\boldsymbol{\nabla}\_{\!\!\theta}\,\sum\_{t=1}^{T}\sum\_{y\_{t}}\pi\_{\theta}(y\_{t}\mid x,y\_{<t})A\_{t,k} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =−∑t=1T∑ytπθ​(yt∣x,y<t)​∇θAt,k+At,k​∇θπθ​(yt∣x,y<t).\displaystyle=-\sum\_{t=1}^{T}\sum\_{y\_{t}}\pi\_{\theta}(y\_{t}\mid x,y\_{<t})\boldsymbol{\nabla}\_{\!\!\theta}\,A\_{t,k}+A\_{t,k}\boldsymbol{\nabla}\_{\!\!\theta}\,\pi\_{\theta}(y\_{t}\mid x,y\_{<t}). |  |

We have that ∇θAt,k=−∇θlog⁡πθ​(yt∣x,y<t)\boldsymbol{\nabla}\_{\!\!\theta}\,A\_{t,k}=-\boldsymbol{\nabla}\_{\!\!\theta}\,\log\pi\_{\theta}(y\_{t}\mid x,y\_{<t}) is the negative score function. Using the score trick, πθ​(yt∣x,y<t)​∇θlog⁡πθ​(yt∣x,y<t)=∇θπθ​(yt∣x,y<t)\pi\_{\theta}(y\_{t}\mid x,y\_{<t})\boldsymbol{\nabla}\_{\!\!\theta}\,\log\pi\_{\theta}(y\_{t}\mid x,y\_{<t})=\boldsymbol{\nabla}\_{\!\!\theta}\,\pi\_{\theta}(y\_{t}\mid x,y\_{<t}). Hence, the first term simplifies to

|  |  |  |  |
| --- | --- | --- | --- |
|  | −∑t=1T∑ytπθ​(yt∣x,y<t)​∇θAt,k\displaystyle-\sum\_{t=1}^{T}\sum\_{y\_{t}}\pi\_{\theta}(y\_{t}\mid x,y\_{<t})\boldsymbol{\nabla}\_{\!\!\theta}\,A\_{t,k} | =∑t=1T∑yt∇θπθ​(yt∣x,y<t)=∑t=1T∇θ∑ytπθ​(yt∣x,y<t)⏟=1=0.\displaystyle=\sum\_{t=1}^{T}\sum\_{y\_{t}}\boldsymbol{\nabla}\_{\!\!\theta}\,\pi\_{\theta}(y\_{t}\mid x,y\_{<t})=\sum\_{t=1}^{T}\boldsymbol{\nabla}\_{\!\!\theta}\,\underbrace{\sum\_{y\_{t}}\pi\_{\theta}(y\_{t}\mid x,y\_{<t})}\_{=1}=0. |  |

Thus, the gradient of ℒSDPO\mathcal{L}\_{\mathrm{SDPO}} is

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇θℒSDPO\displaystyle\boldsymbol{\nabla}\_{\!\!\theta}\,\mathcal{L}\_{\mathrm{SDPO}} | =−∑t=1T∑ytAt,k​∇θπθ​(yt∣x,y<t)\displaystyle=-\sum\_{t=1}^{T}\sum\_{y\_{t}}A\_{t,k}\boldsymbol{\nabla}\_{\!\!\theta}\,\pi\_{\theta}(y\_{t}\mid x,y\_{<t}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =−∑t=1T∑ytπθ​(yt∣x,y<t)​(At,k​∇θlog⁡πθ​(yt∣x,y<t))\displaystyle=-\sum\_{t=1}^{T}\sum\_{y\_{t}}\pi\_{\theta}(y\_{t}\mid x,y\_{<t})\Big(A\_{t,k}\boldsymbol{\nabla}\_{\!\!\theta}\,\log\pi\_{\theta}(y\_{t}\mid x,y\_{<t})\Big) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =−𝔼y∼πθ(⋅∣x)​[∑t=1|y|∑yt∇θlog⁡πθ​(yt∣x,y<t)⋅At,k]\displaystyle=-\mathbb{E}\_{y\sim\pi\_{\theta}(\cdot\mid x)}\left[\sum\_{t=1}^{|y|}\sum\_{y\_{t}}\boldsymbol{\nabla}\_{\!\!\theta}\,\log\pi\_{\theta}(y\_{t}\mid x,y\_{<t})\cdot A\_{t,k}\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =𝔼y∼πθ(⋅∣x)​[∑t=1|y|∑yt∇θlog⁡πθ​(yt∣x,y<t)⋅log⁡πθ​(yt∣x,y<t)qθ​(yt∣x,f,y<t)].\displaystyle=\mathbb{E}\_{y\sim\pi\_{\theta}(\cdot\mid x)}\left[\sum\_{t=1}^{|y|}\sum\_{y\_{t}}\boldsymbol{\nabla}\_{\!\!\theta}\,\log\pi\_{\theta}(y\_{t}\mid x,y\_{<t})\cdot\log\frac{\pi\_{\theta}(y\_{t}\mid x,y\_{<t})}{q\_{\theta}(y\_{t}\mid x,f,y\_{<t})}\right]. |  |

∎

Notably, the above implies that the gradient of ℒSDPO\mathcal{L}\_{\mathrm{SDPO}} is equivalent to the gradient of the loss if −At,k=stopgrad​(log⁡πθ​(yt∣x,y<t)qθ​(yt∣x,f,y<t))-A\_{t,k}=\mathrm{stopgrad}\left(\log\frac{\pi\_{\theta}(y\_{t}\mid x,y\_{<t})}{q\_{\theta}(y\_{t}\mid x,f,y\_{<t})}\right).

### B.2 Trust-region Teacher

To stabilize training, we seek to prevent the teacher qq from diverging from the initial teacher qθrefq\_{\theta\_{{\mathrm{ref}}}}.
We can achieve this by placing an explicit trust-region constraint on the teacher qq (Schulman et al., [2015](#bib.bib78); Peng et al., [2019](#bib.bib69)), that is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑tKL(q(yt∣x,f,y<t)∥qθref(yt∣x,f,y<t))≤ϵ,ϵ>0.\sum\_{t}\mathrm{KL}\left(q(y\_{t}\mid x,f,y\_{<t})\|q\_{\theta\_{{\mathrm{ref}}}}(y\_{t}\mid x,f,y\_{<t})\right)\leq\epsilon,\quad\epsilon>0. |  | (12) |

In the following, we derive a teacher qq which satisfies the trust-region constraint while staying close to the target qθq\_{\theta}.
The following optimization problem characterizes such a qq (Peng et al., [2019](#bib.bib69)):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | arg​maxq∈Δ∑t∑ytq​(yt∣x,f,y<t)​log⁡qθ​(yt∣x,f,y<t)qθref​(yt∣x,f,y<t)s.t.∑tKL(q(yt∣x,f,y<t)∥qθref(yt∣x,f,y<t))≤ϵ,\displaystyle\begin{split}\operatorname\*{arg\,max}\_{q\in\Delta}\ &\sum\_{t}\sum\_{y\_{t}}q(y\_{t}\mid x,f,y\_{<t})\log\frac{q\_{\theta}(y\_{t}\mid x,f,y\_{<t})}{q\_{\theta\_{{\mathrm{ref}}}}(y\_{t}\mid x,f,y\_{<t})}\\ \text{s.t.}\ &\sum\_{t}\mathrm{KL}\left(q(y\_{t}\mid x,f,y\_{<t})\|q\_{\theta\_{{\mathrm{ref}}}}(y\_{t}\mid x,f,y\_{<t})\right)\leq\epsilon,\end{split} | |  | (13) |

where Δ\Delta denotes the probability simplex.
Intuitively, the solution is the qq satisfying the trust-region constraint, which is closest to qθq\_{\theta} (i.e., has minimal cross-entropy to qθq\_{\theta}) while being farthest from qθrefq\_{\theta\_{{\mathrm{ref}}}} (i.e., has maximal cross-entropy to qθrefq\_{\theta\_{{\mathrm{ref}}}}).

###### Proposition B.1.

The solution to [Equation˜13](#A2.E13 "In B.2 Trust-region Teacher ‣ Appendix B Theoretical Analysis ‣ Reinforcement Learning via Self-Distillation") can be expressed in closed form as

|  |  |  |  |
| --- | --- | --- | --- |
|  | q∗​(yt∣x,f,y<t)∝exp⁡((1−α)​log⁡qθref​(yt∣x,f,y<t)+α​log⁡qθ​(yt∣x,f,y<t)).\displaystyle q^{\*}(y\_{t}\mid x,f,y\_{<t})\propto\exp\!\big((1-\alpha)\log q\_{\theta\_{{\mathrm{ref}}}}(y\_{t}\mid x,f,y\_{<t})+\alpha\log q\_{\theta}(y\_{t}\mid x,f,y\_{<t})\big). |  | (14) |

###### Proof.

To simplify notation, we omit the conditioning in the following.
The Lagrangian (with λ≥0\lambda\geq 0 for the KL constraint and ν\nu for normalization) is

|  |  |  |
| --- | --- | --- |
|  | ℒ​(q,λ,ν)=∑t∑ytq​(yt)​log⁡qθ​(yt)qθref​(yt)−λ​(∑ytq​(yt)​log⁡q​(yt)qθref​(yt)−ϵ)+ν​(∑ytq​(yt)−1).\displaystyle\mathcal{L}(q,\lambda,\nu)=\sum\_{t}\sum\_{y\_{t}}q({y\_{t}})\log\frac{q\_{\theta}({y\_{t}})}{q\_{\theta\_{{\mathrm{ref}}}}({y\_{t}})}-\lambda\Big(\sum\_{y\_{t}}q({y\_{t}})\log\frac{q({y\_{t}})}{q\_{\theta\_{{\mathrm{ref}}}}({y\_{t}})}-\epsilon\Big)+\nu\Big(\sum\_{y\_{t}}q({y\_{t}})-1\Big). |  |

Stationarity gives, for all yty\_{t},

|  |  |  |
| --- | --- | --- |
|  | 0=∂ℒ∂q​(yt)=log⁡qθ​(yt)qθref​(yt)−λ​(log⁡q​(yt)qθref​(yt)+1)+ν.\displaystyle 0=\frac{\partial\mathcal{L}}{\partial q(y\_{t})}=\log\frac{q\_{\theta}(y\_{t})}{q\_{\theta\_{{\mathrm{ref}}}}(y\_{t})}-\lambda\Big(\log\frac{q(y\_{t})}{q\_{\theta\_{{\mathrm{ref}}}}(y\_{t})}+1\Big)+\nu. |  |

Let α:=1/λ\alpha:=1/\lambda. Then, the solution to [Equation˜13](#A2.E13 "In B.2 Trust-region Teacher ‣ Appendix B Theoretical Analysis ‣ Reinforcement Learning via Self-Distillation") can be characterized in closed form as

|  |  |  |  |
| --- | --- | --- | --- |
|  | q∗​(yt)\displaystyle q^{\*}(y\_{t}) | ∝qθref​(yt)​exp⁡(α​log⁡qθ​(yt)qθref​(yt))\displaystyle\propto q\_{\theta\_{{\mathrm{ref}}}}(y\_{t})\exp\!\Big(\alpha\log\tfrac{q\_{\theta}(y\_{t})}{q\_{\theta\_{{\mathrm{ref}}}}(y\_{t})}\Big) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ∝exp⁡((1−α)​log⁡qθref​(yt)+α​log⁡qθ​(yt)).\displaystyle\propto\exp\!\big((1-\alpha)\log q\_{\theta\_{{\mathrm{ref}}}}(y\_{t})+\alpha\log q\_{\theta}(y\_{t})\big). |  |

∎

Chen et al. ([2025c](#bib.bib18)) perform a similar derivation, but use reference πθref\pi\_{{\theta\_{{\mathrm{ref}}}}}, which we observe to underperform compared to the reference qθrefq\_{\theta\_{{\mathrm{ref}}}}.

### B.3 EMA Teacher as an Implicit Trust Region

To stabilize training, an alternative to the explicit trust-region teacher in Appendix [B.2](#A2.SS2 "B.2 Trust-region Teacher ‣ Appendix B Theoretical Analysis ‣ Reinforcement Learning via Self-Distillation")
is to parameterize the teacher as an exponential moving average (EMA) of the student’s parameters θk\theta\_{k}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | θk′=(1−α)​θk−1′+α​θk,α∈(0,1),\theta^{\prime}\_{k}=(1-\alpha)\theta^{\prime}\_{k-1}+\alpha\theta\_{k},\qquad\alpha\in(0,1), |  | (15) |

with initialization θ0=θ0′=θref\theta\_{0}=\theta^{\prime}\_{0}={\theta\_{{\mathrm{ref}}}}. While
[Equation˜15](#A2.E15 "In B.3 EMA Teacher as an Implicit Trust Region ‣ Appendix B Theoretical Analysis ‣ Reinforcement Learning via Self-Distillation") constrains the teacher in parameter space (and thus does not, in general, impose an explicit KL trust
region in distribution space), under a mild smoothness condition we can show that the EMA teacher qθ′q\_{\theta^{\prime}} remains within an
*approximate* trust region relative to the reference teacher qθrefq\_{\theta\_{\mathrm{ref}}}.

Let qθ​(yt∣x,f,y<t)q\_{\theta}(y\_{t}\mid x,f,y\_{<t}) be a softmax distribution with logits
zθ​(x,f,y<t)∈ℝ|𝒱|z\_{\theta}(x,f,y\_{<t})\in\mathbb{R}^{|\mathcal{V}|}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | qθ​(yt∣x,f,y<t)=softmax​(zθ​(x,f,y<t))yt.q\_{\theta}(y\_{t}\mid x,f,y\_{<t})=\mathrm{softmax}\big(z\_{\theta}(x,f,y\_{<t})\big)\_{y\_{t}}. |  | (16) |

Assume the logits are LL-Lipschitz in parameters:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖zθ1​(x,f,y<t)−zθ2​(x,f,y<t)‖2≤L​‖θ1−θ2‖2,∀(x,f,t,θ1,θ2).\big\|z\_{\theta\_{1}}(x,f,y\_{<t})-z\_{\theta\_{2}}(x,f,y\_{<t})\big\|\_{2}\leq L\,\big\|\theta\_{1}-\theta\_{2}\big\|\_{2},\qquad\forall(x,f,t,\theta\_{1},\theta\_{2}). |  | (17) |

###### Proposition B.2 (EMA yields an approximate trust region).

Let θ0=θ0′=θref\theta\_{0}=\theta^{\prime}\_{0}={\theta\_{{\mathrm{ref}}}}, k≥1k\geq 1, and update θk′\theta^{\prime}\_{k} by [Equation˜15](#A2.E15 "In B.3 EMA Teacher as an Implicit Trust Region ‣ Appendix B Theoretical Analysis ‣ Reinforcement Learning via Self-Distillation"). We assume:

* •

  The logits are LL-Lipschitz in parameters (cf. [Equation˜17](#A2.E17 "In B.3 EMA Teacher as an Implicit Trust Region ‣ Appendix B Theoretical Analysis ‣ Reinforcement Learning via Self-Distillation")).
* •

  The student does not diverge from the initial model, i.e., ‖θi−θ0‖2≤Rref\|\theta\_{i}-\theta\_{0}\|\_{2}\leq R\_{\mathrm{ref}} for all i≤ki\leq k.

Then, for any (x,c)(x,c) and any sequence length TT, the EMA teacher qθk′\smash{q\_{\theta^{\prime}\_{k}}} implicitly satisfies the trust-region constraint

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑t=1TKL(qθk′(⋅∣x,f,y<t)∥qθref(⋅∣x,f,y<t))≤ϵk,ϵk=L24TRref2(1−(1−α)k)2.\sum\_{t=1}^{T}\mathrm{KL}\left(q\_{\theta^{\prime}\_{k}}(\cdot\mid x,f,y\_{<t})\|q\_{\theta\_{\mathrm{ref}}}(\cdot\mid x,f,y\_{<t})\right)\leq\epsilon\_{k},\qquad\epsilon\_{k}=\frac{L^{2}}{4}\,T\,R\_{\mathrm{ref}}^{2}\,(1-(1-\alpha)^{k})^{2}. |  | (18) |

###### Proof.

Fix (x,f,t)(x,f,t) and define p=softmax​(a)p=\mathrm{softmax}(a) and q=softmax​(b)q=\mathrm{softmax}(b) with
a=zθk′​(x,f,y<t)a=z\_{\theta^{\prime}\_{k}}(x,f,y\_{<t}) and b=zθref​(x,f,y<t)b=z\_{\theta\_{\mathrm{ref}}}(x,f,y\_{<t}).
A standard smoothness bound for softmax distributions101010Let p=softmax​(a)p=\mathrm{softmax}(a) and q=softmax​(b)q=\mathrm{softmax}(b), and define the log-partition
A​(z)=log​∑iexp⁡(zi)A(z)=\log\sum\_{i}\exp(z\_{i}). For the categorical exponential family, the KL divergence admits the
(primal) Bregman form
KL​(p∥q)=A​(b)−A​(a)−⟨∇A​(a),b−a⟩\mathrm{KL}\left(p\|q\right)=A(b)-A(a)-\langle\boldsymbol{\nabla}A(a),\,b-a\rangle
(Wainwright & Jordan, [2008](#bib.bib97), Eq. (5.10)).
Moreover, ∇A​(z)=softmax​(z)\boldsymbol{\nabla}A(z)=\mathrm{softmax}(z) and
∇2A​(z)=diag​(p)−p​p⊤\boldsymbol{\nabla}^{2}A(z)=\mathrm{diag}(p)-pp^{\top} (Boyd & Vandenberghe, [2004](#bib.bib8)).
For any v∈ℝ|𝒱|v\in\mathbb{R}^{|\mathcal{V}|},

v⊤​∇2A​(z)​v=∑ipi​vi2−(∑ipi​vi)2=Vari∼p​(vi)≤(maxi⁡vi−mini⁡vi)24≤‖v‖222,v^{\top}\boldsymbol{\nabla}^{2}A(z)v=\sum\_{i}p\_{i}v\_{i}^{2}-\Big(\sum\_{i}p\_{i}v\_{i}\Big)^{2}=\mathrm{Var}\_{i\sim p}(v\_{i})\leq\frac{(\max\_{i}v\_{i}-\min\_{i}v\_{i})^{2}}{4}\leq\frac{\|v\|\_{2}^{2}}{2},
so ‖∇2A​(z)‖op≤12\|\boldsymbol{\nabla}^{2}A(z)\|\_{\mathrm{op}}\leq\tfrac{1}{2} and hence AA is (1/2)(1/2)-smooth in ∥⋅∥2\|\cdot\|\_{2}.
Applying the standard smoothness inequality for β\beta-smooth functions
(Bubeck, [2015](#bib.bib10), Lemma 3.4) yields
A​(b)≤A​(a)+⟨∇A​(a),b−a⟩+β2​‖b−a‖22A(b)\leq A(a)+\langle\boldsymbol{\nabla}A(a),b-a\rangle+\tfrac{\beta}{2}\|b-a\|\_{2}^{2}; setting β=12\beta=\tfrac{1}{2}
gives KL​(p∥q)≤14​‖a−b‖22\mathrm{KL}\left(p\|q\right)\leq\tfrac{1}{4}\|a-b\|\_{2}^{2}.
 implies

|  |  |  |  |
| --- | --- | --- | --- |
|  | KL​(p∥q)≤14​‖a−b‖22.\mathrm{KL}\left(p\|q\right)\leq\frac{1}{4}\,\|a-b\|\_{2}^{2}. |  | (19) |

By [Equation˜17](#A2.E17 "In B.3 EMA Teacher as an Implicit Trust Region ‣ Appendix B Theoretical Analysis ‣ Reinforcement Learning via Self-Distillation"), this gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | KL(qθk′(⋅∣x,f,y<t)∥qθref(⋅∣x,f,y<t))≤L24∥θk′−θref∥22.\mathrm{KL}\left(q\_{\theta^{\prime}\_{k}}(\cdot\mid x,f,y\_{<t})\|q\_{{\theta\_{{\mathrm{ref}}}}}(\cdot\mid x,f,y\_{<t})\right)\leq\frac{L^{2}}{4}\,\|\theta^{\prime}\_{k}-{\theta\_{{\mathrm{ref}}}}\|\_{2}^{2}. |  | (20) |

Unrolling the EMA recursion with θ0′=θref\theta^{\prime}\_{0}={\theta\_{{\mathrm{ref}}}} yields

|  |  |  |
| --- | --- | --- |
|  | θk′−θref=∑i=1kwi​(θi−θref),wi=α​(1−α)k−i,∑i=1kwi=1−(1−α)k.\theta^{\prime}\_{k}-{\theta\_{{\mathrm{ref}}}}=\sum\_{i=1}^{k}w\_{i}(\theta\_{i}-{\theta\_{{\mathrm{ref}}}}),\qquad w\_{i}=\alpha(1-\alpha)^{k-i},\qquad\sum\_{i=1}^{k}w\_{i}=1-(1-\alpha)^{k}. |  |

By Cauchy–Schwarz,

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖∑i=1kwi​(θi−θref)‖22≤(∑i=1kwi)​(∑i=1kwi​‖θi−θref‖22)=(1−(1−α)k)​∑i=1kwi​‖θi−θref‖22.\Big\|\sum\_{i=1}^{k}w\_{i}(\theta\_{i}-{\theta\_{{\mathrm{ref}}}})\Big\|\_{2}^{2}\leq\Big(\sum\_{i=1}^{k}w\_{i}\Big)\Big(\sum\_{i=1}^{k}w\_{i}\|\theta\_{i}-{\theta\_{{\mathrm{ref}}}}\|\_{2}^{2}\Big)=(1-(1-\alpha)^{k})\sum\_{i=1}^{k}w\_{i}\|\theta\_{i}-{\theta\_{{\mathrm{ref}}}}\|\_{2}^{2}. |  | (21) |

Combining [Equations˜20](#A2.E20 "In Proof. ‣ B.3 EMA Teacher as an Implicit Trust Region ‣ Appendix B Theoretical Analysis ‣ Reinforcement Learning via Self-Distillation") and [21](#A2.E21 "Equation 21 ‣ Proof. ‣ B.3 EMA Teacher as an Implicit Trust Region ‣ Appendix B Theoretical Analysis ‣ Reinforcement Learning via Self-Distillation") and summing over t=1,…,Tt=1,\dots,T gives

|  |  |  |
| --- | --- | --- |
|  | ∑t=1TKL(qθk′(⋅∣x,f,y<t)∥qθref(⋅∣x,f,y<t))≤L24T(1−(1−α)k)∑i=1kα(1−α)k−i∥θi−θref∥22.\sum\_{t=1}^{T}\mathrm{KL}\left(q\_{\theta^{\prime}\_{k}}(\cdot\mid x,f,y\_{<t})\|q\_{{\theta\_{{\mathrm{ref}}}}}(\cdot\mid x,f,y\_{<t})\right)\leq\frac{L^{2}}{4}\,T\,(1-(1-\alpha)^{k})\,\sum\_{i=1}^{k}\alpha(1-\alpha)^{k-i}\,\big\|\theta\_{i}-{\theta\_{{\mathrm{ref}}}}\big\|\_{2}^{2}. |  |

Finally, if ‖θi−θref‖2≤Rref\|\theta\_{i}-{\theta\_{{\mathrm{ref}}}}\|\_{2}\leq R\_{\mathrm{ref}} for all i≤ki\leq k, then
∑i=1kwi​‖θi−θref‖22≤Rref2​∑i=1kwi=Rref2​(1−(1−α)k)\sum\_{i=1}^{k}w\_{i}\|\theta\_{i}-{\theta\_{{\mathrm{ref}}}}\|\_{2}^{2}\leq R\_{\mathrm{ref}}^{2}\sum\_{i=1}^{k}w\_{i}=R\_{\mathrm{ref}}^{2}(1-(1-\alpha)^{k}),
which yields [Equation˜18](#A2.E18 "In Proposition B.2 (EMA yields an approximate trust region). ‣ B.3 EMA Teacher as an Implicit Trust Region ‣ Appendix B Theoretical Analysis ‣ Reinforcement Learning via Self-Distillation").
∎

To summarize, [Equation˜18](#A2.E18 "In Proposition B.2 (EMA yields an approximate trust region). ‣ B.3 EMA Teacher as an Implicit Trust Region ‣ Appendix B Theoretical Analysis ‣ Reinforcement Learning via Self-Distillation") recovers the same form as the explicit trust-region constraint
∑tKL(⋅∥qθref)≤ϵ\smash{\sum\_{t}\mathrm{KL}\left(\cdot\|q\_{{\theta\_{{\mathrm{ref}}}}}\right)\leq\epsilon} (cf. [Section˜B.2](#A2.SS2 "B.2 Trust-region Teacher ‣ Appendix B Theoretical Analysis ‣ Reinforcement Learning via Self-Distillation")), with an
effective radius ϵk\epsilon\_{k} controlled by the EMA rate α\alpha and the cumulative deviation from the reference.
For fixed kk and small α\alpha, 1−(1−α)k≈α​k\smash{1-(1-\alpha)^{k}\approx\alpha k}, so the reference divergence scales as
ϵk=O​(α2​k2)\epsilon\_{k}=O(\alpha^{2}k^{2}) when θi\theta\_{i} stays within a bounded neighborhood of θref{\theta\_{{\mathrm{ref}}}}.

## Appendix C Additional Related Work

##### Value networks and Monte Carlo advantage estimation.

Several prior approaches aim to improve credit assignment but face the same information bottleneck as GRPO. Classical RL frequently trains value networks which provide token-level advantages, but themselves are learned from scalar rewards (Schulman et al., [2016](#bib.bib79); [2017](#bib.bib80)). Furthermore, value networks incur significant computational and memory overhead and are therefore typically not used to train LLMs.
Other recent work estimates token-level advantages by performing additional generations starting from various positions in the original attempt (Kazemnejad et al., [2025](#bib.bib44); Zheng et al., [2025b](#bib.bib118)).
While this can learn with fewer gradient steps than GRPO it still uses only scalar rewards as signal and requires costly additional generations.

##### Dense credit assignment with a reward model.

Several recent works have explored assigning dense (per-token) rewards given access to an external reward model, leveraging internal structure of the reward model (Chan et al., [2024](#bib.bib12); Cao et al., [2025](#bib.bib11)).

##### Partial observability.

From the perspective of classical RL, many verifiable domains for LLMs are naturally *partially observable*:
executing a proposed solution induces a latent environment state (e.g., failing tests or states of an agentic system) that is revealed only through rich feedback.
This aligns with the formalism of partially observable Markov decision processes (POMDPs), where agents must act under incomplete observations of state (Kaelbling et al., [1998](#bib.bib43); Sutton & Barto, [1998](#bib.bib92)).
By contrast, RLVR and RLHF pipelines typically discard this observation channel and learn only from terminal scalar rewards or pairwise preferences.

##### Relation to test-time training.

Our setting from [Section˜5](#S5 "5 Solving Hard Questions via Test-Time Self-Distillation ‣ Reinforcement Learning via Self-Distillation") can be seen as a special case of test-time training where the model itself is updated at test-time using self-distillation.
Updating the model at test-time is known as test-time training (Sun et al., [2020](#bib.bib90); [2025](#bib.bib91); Hardt & Sun, [2024](#bib.bib34); Hübotter et al., [2025a](#bib.bib39); [b](#bib.bib40); Akyürek et al., [2025](#bib.bib2); Behrouz et al., [2025](#bib.bib6); Tandon et al., [2025](#bib.bib93); Hübotter et al., [2026](#bib.bib38)).
Unlike prior work, self-distillation uses the in-context learning ability of the current model to attribute credit after receiving feedback.
This can be seen as simulating long-context reasoning with periodic compression of context into the model weights.

### C.1 SDPO as Maximum Entropy RL

The SDPO objective resembles the objective in maximum entropy RL (e.g., Levine, [2018](#bib.bib54); Haarnoja et al., [2018](#bib.bib33)) with a particular choice of reward function.

##### Maximum Entropy RL

Consider optimizing

|  |  |  |  |
| --- | --- | --- | --- |
|  | arg​maxθ𝔼y∼πθ(⋅∣x)[∑tr(yt∣x,y<t)]+λH[πθ(⋅∣x)],λ>0\operatorname\*{arg\,max}\_{\theta}\ \mathbb{E}\_{y\sim\pi\_{\theta}(\cdot\mid x)}{}\left[\sum\_{t}r(y\_{t}\mid x,y\_{<t})\right]+\lambda\mathrm{H}\left[\pi\_{\theta}(\cdot\mid x)\right],\quad\lambda>0 |  | (22) |

where πθ​(y∣x)=∏t=1Tπθ​(yt∣x,y<t)\smash{\pi\_{\theta}(y\mid x)=\prod\_{t=1}^{T}\pi\_{\theta}(y\_{t}\mid x,y\_{<t})} and H[πθ(⋅∣x)]=𝔼y∼πθ(⋅∣x)[−logπθ(y∣x)]\smash{\mathrm{H}\left[\pi\_{\theta}(\cdot\mid x)\right]=\mathbb{E}\_{y\sim\pi\_{\theta}(\cdot\mid x)}{}\left[-\log\pi\_{\theta}(y\mid x)\right]} is the entropy of the policy.
Here, r​(yt∣x,y<t)r(y\_{t}\mid x,y\_{<t}) is an arbitrary reward function, possibly “dense” (i.e., per-token).
[Equation˜22](#A3.E22 "In Maximum Entropy RL ‣ C.1 SDPO as Maximum Entropy RL ‣ Appendix C Additional Related Work ‣ Reinforcement Learning via Self-Distillation") is known as maximum entropy RL.
It is known that this objective is equivalent to solving a variational inference problem which discuss next.

To this end, we define a Bernoulli random variable 𝒞\mathcal{C} which is 11 if the attempt yy is correct and 0 otherwise.
We then define its distribution as p​(𝒞=1∣x,y)∝exp⁡(1λ​∑tr​(yt∣x,y<t))\smash{p(\mathcal{C}=1\mid x,y)\propto\exp(\tfrac{1}{\lambda}\sum\_{t}r(y\_{t}\mid x,y\_{<t}))}.
Further assuming w.l.o.g. that the “prior” over responses is uniform, we can express the posterior conditioned on the event of correctness as

|  |  |  |  |
| --- | --- | --- | --- |
|  | π⋆​(y∣x):=p​(y∣x,𝒞=1)∝p​(𝒞=1∣x,y)∝exp⁡(1λ​∑tr​(yt∣x,y<t)).\pi^{\star}(y\mid x):=p(y\mid x,\mathcal{C}=1)\propto p(\mathcal{C}=1\mid x,y)\propto\exp\!\left(\frac{1}{\lambda}\sum\_{t}r(y\_{t}\mid x,y\_{<t})\right). |  | (23) |

Then, [Equation˜22](#A3.E22 "In Maximum Entropy RL ‣ C.1 SDPO as Maximum Entropy RL ‣ Appendix C Additional Related Work ‣ Reinforcement Learning via Self-Distillation") is equivalent to minimizing the KL divergence with respect to π⋆\pi^{\star}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | arg​minθ∑tKL(πθ(yt∣x,y<t)∥π⋆(yt∣x,y<t)).\operatorname\*{arg\,min}\_{\theta}\ \sum\_{t}\mathrm{KL}\left(\pi\_{\theta}(y\_{t}\mid x,y\_{<t})\|\pi^{\star}(y\_{t}\mid x,y\_{<t})\right). |  | (24) |

##### SDPO optimizes an implicit reward defined by the teacher

Note that [Equation˜24](#A3.E24 "In Maximum Entropy RL ‣ C.1 SDPO as Maximum Entropy RL ‣ Appendix C Additional Related Work ‣ Reinforcement Learning via Self-Distillation") is equivalent to the SDPO objective ([Equation˜1](#S2.E1 "In 2 SDPO: Self-Distillation Policy Optimization ‣ Reinforcement Learning via Self-Distillation")) with implicit reward r​(yt∣x,y<t)=log⁡q​(yt∣x,f,y<t)r(y\_{t}\mid x,y\_{<t})=\log q(y\_{t}\mid x,f,y\_{<t}) and λ=1\lambda=1.
In this sense, SDPO can be seen as a maximum entropy RL algorithm with dense rewards constructed implicitly through the retrospective model.

This also points to a connection of SDPO to inverse RL (Ng et al., [2000](#bib.bib66); Ziebart et al., [2008](#bib.bib121); Rafailov et al., [2023](#bib.bib71)), where the goal is to recover an unknown reward function.
In SDPO, the student learns an implicit reward function defined by the retrospective model.

## Appendix D Additional Results & Ablations

This section is organized as follows:

* •

  [Section˜D.1](#A4.SS1 "D.1 Learning without rich environment feedback ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation") contains results and ablations for [Section˜3](#S3 "3 Learning without Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation").
* •

  [Section˜D.2](#A4.SS2 "D.2 Learning with rich environment feedback ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation") contains results and ablations for [Section˜4](#S4 "4 Learning with Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation").
* •

  [Section˜D.3](#A4.SS3 "D.3 Test-time self-distillation ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation") contains results and ablations for [Section˜5](#S5 "5 Solving Hard Questions via Test-Time Self-Distillation ‣ Reinforcement Learning via Self-Distillation").

### D.1 Learning without rich environment feedback

* •

  [Table˜7](#A4.T7 "In D.1 Learning without rich environment feedback ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation") reports results when optimal hyperparameters are selected for each model/task combination.
* •

  [Table˜8](#A4.T8 "In D.1 Learning without rich environment feedback ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation") compares average response lengths of SDPO and GRPO.

|  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Chemistry | | Physics | | Biology | | Materials | | Tool use | |
|  | 1h | 5h | 1h | 5h | 1h | 5h | 1h | 5h | 1h | 5h |
| Qwen3-8B | 35.6 | | 59.2 | | 27.9 | | 58.9 | | 57.5 | |
| + GRPO | 54.2 | 69.6 | 62.9 | 74.5 | 34.3 | 51.8 | 74.3 | 77.1 | 61.7 | 68.1 |
| + GRPO (on-policy) | 54.2 | 69.6 | 62.9 | 74.8 | 30.3 | 49.4 | 73.3 | 75.8 | 61.7 | 68.1 |
| + SDPO (on-policy) | 59.9 | 70.1 | 70.6 | 80.6 | 53.1 | 53.1 | 72.1 | 78.3 | 56.4 | 68.5 |
| Olmo3-7B-Instruct | 18.8 | | 37.7 | | 18.1 | | 36.7 | | 39.3 | |
| + GRPO | 42.7 | 54.3 | 55.3 | 63.3 | 54.2 | 63.8 | 73.8 | 78.1 | 56.4 | 65.0 |
| + GRPO (on-policy) | 48.8 | 54.3 | 62.7 | 62.7 | 54.2 | 63.8 | 67.9 | 74.4 | 56.0 | 61.3 |
| + SDPO (on-policy) | 59.2 | 76.8 | 60.3 | 71.4 | 56.1 | 58.3 | 75.3 | 79.2 | 57.3 | 62.5 |

Table 7: Comparison of SDPO and GRPO on reasoning-related benchmarks. We report the highest achieved avg@16 within 1 hour and 5 hours of wall-clock training time, respectively. Both SDPO and on-policy GRPO perform one gradient step per generation batch, while GRPO performs 4 off-policy mini batch steps. We select optimal hyperparameters for SDPO and baselines based on 5h accuracy. We perform this selection independently for each model and dataset. Each run is performed on a node with 4 NVIDIA GH200 GPUs. Together with initialization and validation, each run takes approximately 6 hours. *As opposed to [Table˜3](#S3.T3 "In 3.2 Results ‣ 3 Learning without Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation") which selects globally optimal hyperparameters per method, this table selects optimal hyperparameters individually for each model/task combination based on 5h accuracy.* The hyperparameter grid is described in [Section˜E.2.1](#A5.SS2.SSS1 "E.2.1 Details on Hyperparameter Selection (Section˜3) ‣ E.2 Hyperparameters ‣ Appendix E Experiment Details ‣ Reinforcement Learning via Self-Distillation").

|  |  |  |  |
| --- | --- | --- | --- |
| Model | GRPO | SDPO | Reduction of SDPO |
| Qwen3-8B | 820.8 | 255.8 | 3.2×3.2\times |
| Olmo3-7B-Instruct | 1095.4 | 343.9 | 3.2×3.2\times |

Table 8: Average response lengths of SDPO and GRPO (averaged across tasks from [Section˜3](#S3 "3 Learning without Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation")). Both algorithms are evaluated in the on-policy setting.

### D.2 Learning with rich environment feedback

#### D.2.1 Additional Results

!(/html/2601.20802/assets/x12.png)

Figure 15: Average accuracy during training until step 80, stratified by difficulty. Error bars show standard deviation across 3 seeds.

[Figure˜15](#A4.F15 "In D.2.1 Additional Results ‣ D.2 Learning with rich environment feedback ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation") shows the average accuracy of SDPO and GRPO stratified by question difficulty. LCB differentiates between easy, medium, and hard questions.
As displayed, SDPO significantly improves over GRPO in solving medium and hard questions, highlighting the importance of rich feedback for challenging tasks. Note that this categorization of questions is different from the one in [Section˜5](#S5 "5 Solving Hard Questions via Test-Time Self-Distillation ‣ Reinforcement Learning via Self-Distillation").

In [Figure˜16](#A4.F16 "In D.2.1 Additional Results ‣ D.2 Learning with rich environment feedback ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation"), we compare different train batch sizes and number of rollouts for training GRPO and SDPO on LCBv6.

!(/html/2601.20802/assets/x13.png)

Figure 16: Accuracy (pass@1) for varying train batch sizes (4, 8, 16, 32) and number of rollouts (4, 8) for training SDPO and GRPO with Qwen3-8B (Yang et al., [2025](#bib.bib106)) on LCBv6, ±\pm stderr across 3 seeds. Different shades of the same color correspond to different runs.

Complementing the results shown in [Figure˜8](#S4.F8 "In 4.1 Self-distillation benefits from stronger models ‣ 4 Learning with Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation"), we show additional results using Qwen2.5-Instruct (Qwen et al., [2024](#bib.bib70)) in [Figure˜17](#A4.F17 "In D.2.1 Additional Results ‣ D.2 Learning with rich environment feedback ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation").

!(/html/2601.20802/assets/x14.png)

Figure 17: Average validation accuracy by model size, ±\pm std across 3 seeds. With Qwen2.5-Instruct (Qwen et al., [2024](#bib.bib70)) and Qwen3 (Yang et al., [2025](#bib.bib106)) on LCBv6. Until step 65 for Qwen2.5 and until step 80 for Qwen3.

#### D.2.2 Training Stability

[Figure˜18](#A4.F18 "In D.2.2 Training Stability ‣ D.2 Learning with rich environment feedback ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation") shows diverse metrics logged during training, including the loss, entropy, average gradient norm, and average response length.

!(/html/2601.20802/assets/x15.png)

Figure 18: Loss, entropy, avg. gradient norm and avg. response length during training of SDPO on LCBv6 ([Section˜4](#S4 "4 Learning with Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation")

.

#### D.2.3 Baselines

[Table˜9](#A4.T9 "In D.2.3 Baselines ‣ D.2 Learning with rich environment feedback ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation") compares the performance on LCBv6 of various baselines, including two variants of GRPO, GSPO, and CISPO to SDPO.

|  |  |  |
| --- | --- | --- |
|  | Accuracy | Avg accuracy |
| GRPO | 41.2±0.841.2\pm 0.8 | 38.2±0.038.2\pm 0.0 |
| + only high-entropy tokens (Wang et al., [2025](#bib.bib100)) | 37.8±2.237.8\pm 2.2 | 35.9±0.135.9\pm 0.1 |
| GSPO (Zheng et al., [2025a](#bib.bib117)) | 40.1±2.340.1\pm 2.3 | 37.7±0.137.7\pm 0.1 |
| CISPO (Chen et al., [2025a](#bib.bib13)) | 41.2±1.841.2\pm 1.8 | 37.8±0.137.8\pm 0.1 |
| SDPO | 48.8±0.6\mathbf{48.8}\pm 0.6 | 43.8±0.0\mathbf{43.8}\pm 0.0 |

Table 9: Performance on LCBv6 at/until training step 80 with std over 3 seeds. We compare to GSPO (Zheng et al., [2025a](#bib.bib117)) and CISPO (Chen et al., [2025a](#bib.bib13)). With Qwen3-8B.

### D.3 Test-time self-distillation

Complementing the results shown in [Section˜5](#S5 "5 Solving Hard Questions via Test-Time Self-Distillation ‣ Reinforcement Learning via Self-Distillation"), we show the discovery@kk curves for all hard question in [Figure˜20](#A4.F20 "In D.3 Test-time self-distillation ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation"), and report the mean number of generations until the first discovery in [Table˜10](#A4.T10 "In D.3 Test-time self-distillation ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation"). Further, [Table˜11](#A4.T11 "In D.3 Test-time self-distillation ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation") shows the per-question accuracy of the self-teacher at the initial training step of SDPO. In [Figure˜19](#A4.F19 "In D.3 Test-time self-distillation ‣ Appendix D Additional Results & Ablations ‣ Reinforcement Learning via Self-Distillation"), we ablate the choice of batch size for SDPO and the in-context reprompting strategy for multi-turn sampling.

In the selection of hard questions, we have discarded one malformed question (Q9) where the coding environment did not correctly validate the solution due to rounding inaccuracies, which led to failures even with correct logic.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Question | SDPO | Best-of-kk | Multi-turn | Speedup |
|  |  |  |  | Best-of-kk →\rightarrow SDPO |
| 1 | 104 | 98 | 59 | 0.9×\times |
| 3\* | 1987 | ≥2750\geq 2750 | ≥2750\geq 2750 | 1.4×\times |
| 10\* | 938 | ≥2750\geq 2750 | 1706 | 2.9×\times |
| 43 | 111 | 109 | 111 | 1.0×\times |
| 46\* | 1852 | 1466 | 1315 | 0.8×\times |
| 59 | 172 | 123 | 76 | 0.7×\times |
| 69 | 280 | 134 | 134 | 0.5×\times |
| 74\* | 1948 | 1466 | 2405 | 0.8×\times |
| 86 | 85 | 421 | 335 | 5.0×\times |
| 91\* | 1360 | ≥2750\geq 2750 | 2384 | 2.0×\times |
| 92\* | 1575 | ≥2750\geq 2750 | 2203 | 1.8×\times |
| 95\* | 1948 | 1466 | 1794 | 0.8×\times |
| 100 | 277 | 294 | 1596 | 1.1×\times |
| 103\* | 2246 | ≥2750\geq 2750 | 2210 | 1.2×\times |
| 111 | 85 | 95 | 39 | 1.1×\times |
| 120 | 24 | 327 | 70 | 13.6×\times |
| 125\* | 1795 | 1466 | 2320 | 0.8×\times |
| 127 | 28 | 368 | 61 | 13.1×\times |
| 129 | 168 | 173 | 104 | 1.0×\times |
| Hard tasks | 894 | 1145 | 1141 | 1.3×\times |
| Very hard tasks | 1739 | 2180 | 2121 | 1.2×\times |

Table 10: Mean number of generations until first success per question for SDPO, best-of-kk sampling, and the multi-turn sampling. For the mean calculation, values are truncated at the maximum budget of 2750 generations. Very hard tasks (pass​@​64<0.03\text{pass}@64<0.03) are marked with an asterisk (\*). Averaged over all questions, SDPO achieves successes faster than the baselines, reaching a speedup of up to 13.6×13.6\times on individual questions compared to best-of-kk sampling.

|  |  |
| --- | --- |
| Question | Initial Teacher Accuracy (%) |
| 1 | 0.00 |
| 3 | 0.00 |
| 10 | 0.00 |
| 43 | 6.25 |
| 46 | 0.00 |
| 59 | 0.00 |
| 69 | 3.12 |
| 74 | 0.00 |
| 86 | 0.00 |
| 91 | 0.00 |
| 92 | 0.00 |
| 95 | 0.00 |
| 100 | 0.00 |
| 103 | 0.00 |
| 111 | 0.00 |
| 120 | 0.00 |
| 125 | 0.00 |
| 127 | 1.23 |
| 129 | 0.06 |

Table 11: Average accuracy of the retrospective teacher at the first step for each question. These scores represent the percentage of successful solutions generated when the base model is reprompted with feedback in a single-turn interaction. For the majority of these hard and very hard tasks, the teacher accuracy is near or exactly 0%. Despite this, the self-distilled token-level advantages are sufficiently rich for SDPO to iteratively refine its policy and solve these questions over successive updates.

!(/html/2601.20802/assets/x16.png)

Figure 19: Ablations self-distillation at test-time on hard tasks. Left: Impact of SDPO batch size on pass​@​k\text{pass}@k curves. While smaller batch sizes (8 and 16) can lead to slightly earlier discoveries at very low generation budgets (k<26k<2^{6}), larger batch sizes (16, 32) result in more stable updates that significantly improve the discovery rate as the budget scales.
Right: Comparison of multi-turn reprompting templates on a subset of hard questions. The “Only feedback” template concatenates the feedback from previous attempts using a first-in, first-out sliding window. The “Attempts + Feedback” template concatenates the full turn, also using a sliding window. Including only the feedback substantially outperforms concatenating full conversations.

!(/html/2601.20802/assets/x17.png)

Figure 20: Individual task results self-distillation at test-time. Discovery​@​k\text{Discovery}@k for each of the 19 questions evaluated in [Section˜5](#S5 "5 Solving Hard Questions via Test-Time Self-Distillation ‣ Reinforcement Learning via Self-Distillation"). In most cases, SDPO finds a successful solution significantly earlier than both the base model and the multi-turn baseline. Notably, for one question (Q3) where the base model and the multi-turn baseline maintain a discovery​@​k\text{discovery}@k of zero for the entire budget up to 2750 , SDPO discovers a solution after 321 attempts. Curves represent the mean and 90% confidence intervals across 5 random seeds per question.

## Appendix E Experiment Details

### E.1 Technical setup

All experiments were conducted on a single node equipped with four NVIDIA
GH200 GPUs, for a total of 378GB VRAM. Our environment is built on top of the NVIDIA PyTorch container nvcr.io/nvidia/pytorch:25.02-py3, with CUDA 12.8 and PyTorch v2.7.0.

Our implementation is based on the verl library (Sheng et al., [2025](#bib.bib84)). We use PyTorch Fully Sharded Data Parallel (FSDP2) for distributed training. For rollout generation, we employ vLLM (Kwon et al., [2023](#bib.bib49)), which enables efficient batched inference on the multi-GPU node.

### E.2 Hyperparameters

We summarize hyperparameters used for SDPO in [Table˜12](#A5.T12 "In E.2 Hyperparameters ‣ Appendix E Experiment Details ‣ Reinforcement Learning via Self-Distillation") and those used for GRPO in [Table˜13](#A5.T13 "In E.2 Hyperparameters ‣ Appendix E Experiment Details ‣ Reinforcement Learning via Self-Distillation").

|  |  |  |  |
| --- | --- | --- | --- |
| Parameters | Without Feedback | With Feedback | TTT |
|  | Section [3](#S3 "3 Learning without Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation") | Section [4](#S4 "4 Learning with Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation") | Section [5](#S5 "5 Solving Hard Questions via Test-Time Self-Distillation ‣ Reinforcement Learning via Self-Distillation") |
| General |  |  |  |
| Model | Qwen/Qwen3-8B | Qwen/Qwen3-8B | Qwen/Qwen3-8B |
|  | allenai/Olmo3-7B-Instruct |  |  |
| Thinking | False | False | False |
| Data |  |  |  |
| Max. prompt length | 2048 | 2048 | 2048 |
| Max. response length | 8192 | 8192 | 8192 |
| Batching |  |  |  |
| Question batch size | 32 | 32 | 1 |
| Mini batch size | 32 | 1 | 1 |
| Number of rollouts | 8 | 8 | 16 |
| Rollout |  |  |  |
| Inference engine | vllm | vllm | vllm |
| Temperature | 1.0 | 1.0 | 1.0 |
| Validation |  |  |  |
| Number of rollouts | 16 | 4 | - |
| Temperature | 0.6 | 0.6 | - |
| Top-pp | 0.95 | 0.95 | - |
| SDPO loss |  |  |  |
| Top-KK distillation | 100 | 20 | 20 |
| Distillation divergence | Jensen–Shannon | Reverse-KL | Reverse-KL |
| Clip advantages | – | – | 5.0 |
| Teacher-EMA update rate | 0.05 | 0.01 | 0.01 |
| Rollout importance sampling clip | 2 | 2 | 2 |
| Training |  |  |  |
| Optimizer | AdamW | AdamW | AdamW |
| Learning rate | 1×10−51\times 10^{-5} (constant) | 1×10−61\times 10^{-6} (constant) | 1×10−61\times 10^{-6} (constant) |
| Warmup steps | 10 | 0 | 0 |
| Weight decay | 0.01 | 0.01 | 0.01 |
| Gradient Clip Norm | 1.0 | 1.0 | 1.0 |

Table 12: Hyperparameters used for SDPO for each experimental setup.

|  |  |
| --- | --- |
| Parameters | Experiment 1 |
|  | Section [3](#S3 "3 Learning without Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation") |
| General |  |
| Model | Qwen/Qwen3-8B |
|  | allenai/Olmo3-7B-Instruct |
| Thinking | False |
| Data |  |
| Max. prompt length | 2048 |
| Max. response length | 8192 |
| Batching |  |
| Question batch size | 32 |
| Mini batch size | 8 (default) / 32 (on-policy) |
| Number of rollouts | 8 |
| Rollout |  |
| Inference engine | vllm |
| Temperature | 1.0 |
| Validation |  |
| Temperature | 0.6 |
| Top-pp | 0.95 |
| Number of rollouts | 16 |
| Loss |  |
| ϵ\epsilon-high | 0.28 |
| Rollout importance sampling clip | 2 |
| KL coefficient (λ\lambda) | 0.0 |
| Training |  |
| Optimizer | AdamW |
| Learning rate | 1×10−61\times 10^{-6} (default) / 1×10−51\times 10^{-5} (on-policy) |
| Warmup steps | 10 |
| Weight decay | 0.01 |
| Gradient Clip Norm | 1.0 |

Table 13: Hyperparameters used for GRPO.

#### E.2.1 Details on Hyperparameter Selection ([Section˜3](#S3 "3 Learning without Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation"))

For GRPO in the experiments in [Section˜3](#S3 "3 Learning without Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation"), we perform a grid search over learning rates {10−5,10−6}\{10^{-5},10^{-6}\} and minibatch sizes {8,32}\{8,32\}. For on-policy GRPO, we search over the same learning rates while fixing the minibatch size to 32. For SDPO, we grid-search over KL variants (forward KL, Jensen–Shannon), learning rates {10−5,10−6}\{10^{-5},10^{-6}\}, and minibatch sizes {8,32}\{8,32\}.
For each method (GRPO, on-policy GRPO, and SDPO), we select a *single* hyperparameter configuration that achieves the highest validation accuracy within the first 5 hours of training, evaluated across all datasets and models used in [Section˜3](#S3 "3 Learning without Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation").
We further report results obtained by selecting the optimal hyperparameter configuration separately for each model and dataset in [Table˜3](#S3.T3 "In 3.2 Results ‣ 3 Learning without Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation").

### E.3 User Templates

For multiple-choice questions and tool use, the model must be prompted in a task-specific manner. We therefore provide the prompt templates used for these settings below.

[⬇](data:text/plain;base64,R2l2ZW4gYSBxdWVzdGlvbiBhbmQgZm91ciBvcHRpb25zLCBwbGVhc2Ugc2VsZWN0IHRoZSByaWdodCBhbnN3ZXIuIFJlc3BvbmQgaW4gdGhlIGZvbGxvd2luZyBmb3JtYXQ6CjxyZWFzb25pbmc+Ci4uLgo8L3JlYXNvbmluZz4KPGFuc3dlcj4KLi4uCjwvYW5zd2VyPgoKRm9yIHRoZSBhbnN3ZXIsIG9ubHkgb3V0cHV0IHRoZSBsZXR0ZXIgY29ycmVzcG9uZGluZyB0byB0aGUgY29ycmVjdCBvcHRpb24gKEEsIEIsIEMsIG9yIEQpLCBhbmQgbm90aGluZyBlbHNlLiBEbyBub3QgcmVzdGF0ZSB0aGUgYW5zd2VyIHRleHQuIEZvciBleGFtcGxlLCBpZiB0aGUgYW5zd2VyIGlzICJBIiwganVzdCBvdXRwdXQ6CjxhbnN3ZXI+CkEKPC9hbnN3ZXI+)

Given a question and four options, please select the right answer. Respond in the following format:

<reasoning>

...

</reasoning>

<answer>

...

</answer>

For the answer, only output the letter corresponding to the correct option (A, B, C, or D), and nothing else. Do not restate the answer text. For example, if the answer is "A", just output:

<answer>

A

</answer>

Listing 1: System prompt: Multiple Choice Questions

[⬇](data:text/plain;base64,e3F1ZXN0aW9ufQpQbGVhc2UgcmVhc29uIHN0ZXAgYnkgc3RlcC4=)

{question}

Please reason step by step.

Listing 2: User prompt: Multiple Choice Questions

[⬇](data:text/plain;base64,WW91IGFyZSBhIGhlbHBmdWwgZnVuY3Rpb24tY2FsbGluZyBBSSBhc3Npc3RhbnQuIFlvdSBhcmUgcHJvdmlkZWQgd2l0aCBmdW5jdGlvbiBzaWduYXR1cmVzIHdpdGhpbiA8ZnVuY3Rpb25zPjwvZnVuY3Rpb25zPiBYTUwgdGFncy4gWW91IG1heSBjYWxsIG9uZSBvciBtb3JlIGZ1bmN0aW9ucyB0byBhc3Npc3Qgd2l0aCB0aGUgdXNlciBxdWVyeS4gT3V0cHV0IGFueSBmdW5jdGlvbiBjYWxscyB3aXRoaW4gPGZ1bmN0aW9uX2NhbGxzPjwvZnVuY3Rpb25fY2FsbHM+IFhNTCB0YWdzLiBEbyBub3QgbWFrZSBhc3N1bXB0aW9ucyBhYm91dCB3aGF0IHZhbHVlcyB0byBwbHVnIGludG8gZnVuY3Rpb25zLg==)

You are a helpful function-calling AI assistant. You are provided with function signatures within <functions></functions> XML tags. You may call one or more functions to assist with the user query. Output any function calls within <function\_calls></function\_calls> XML tags. Do not make assumptions about what values to plug into functions.

Listing 3: System prompt: Tool use

[⬇](data:text/plain;base64,WW91ciB0YXNrIGlzIHRvIGFuc3dlciB0aGUgdXNlcidzIHF1ZXN0aW9uIHVzaW5nIGF2YWlsYWJsZSB0b29scy4KWW91IGhhdmUgYWNjZXNzIHRvIHRoZSBmb2xsb3dpbmcgdG9vbHM6Ck5hbWU6IEF4b2xvdGwKRGVzY3JpcHRpb246IENvbGxlY3Rpb24gb2YgYXhvbG90bCBwaWN0dXJlcyBhbmQgZmFjdHMKRG9jdW1lbnRhdGlvbjoKZ2V0UmFuZG9tQXhvbG90bEltYWdlOiBSZXRyaWV2ZSBhIHJhbmRvbSBheG9sb3RsIGltYWdlIHdpdGggaW5mb3JtYXRpb24gb24gdGhlIGltYWdlIHNvdXJjZS4KUGFyYW1ldGVyczoge30KT3V0cHV0OiBTdWNjZXNzZnVsIHJlc3BvbnNlLgogLSBGb3JtYXQ6IGFwcGxpY2F0aW9uL2pzb24KIC0gU3RydWN0dXJlOiBPYmplY3R7dXJsLCBzb3VyY2UsIGRlc2NyaXB0aW9ufQpzZWFyY2hBeG9sb3RsSW1hZ2VzOiBTZWFyY2ggZm9yIGF4b2xvdGwgaW1hZ2VzIGJhc2VkIG9uIHNwZWNpZmljIGNyaXRlcmlhIHN1Y2ggYXMgY29sb3IsIGdlbmRlciwgYW5kIHNpemUuClBhcmFtZXRlcnM6IHsiY29sb3IiOiAic3RyaW5nLiBPbmUgb2Y6IFt3aWxkLCBsZXVjaXN0aWMsIGFsYmlub10uIFRoZSBjb2xvciBvZiB0aGUgYXhvbG90bCAoZS5nLiwgJ3dpbGQnLCAnbGV1Y2lzdGljJywgJ2FsYmlubycsIGV0Yy4pLiIsICJnZW5kZXIiOiAic3RyaW5nLiBPbmUgb2Y6IFttYWxlLCBmZW1hbGVdLiBUaGUgZ2VuZGVyIG9mIHRoZSBheG9sb3RsICgnbWFsZScsICdmZW1hbGUnKS4iLCAic2l6ZSI6ICJzdHJpbmcuIE9uZSBvZjogW3NtYWxsLCBtZWRpdW0sIGxhcmdlXS4gVGhlIHNpemUgb2YgdGhlIGF4b2xvdGwgKCdzbWFsbCcsICdtZWRpdW0nLCAnbGFyZ2UnKS4iLCAicGFnZSI6ICJpbnRlZ2VyLiBUaGUgcGFnZSBudW1iZXIgZm9yIHBhZ2luYXRpb24gcHVycG9zZXMuIn0KT3V0cHV0OiBTdWNjZXNzZnVsIHJlc3BvbnNlLgogLSBGb3JtYXQ6IGFwcGxpY2F0aW9uL2pzb24KIC0gU3RydWN0dXJlOiBPYmplY3R7cmVzdWx0czogQXJyYXlbT2JqZWN0e3VybCwgc291cmNlLCBkZXNjcmlwdGlvbn1dLCBwYWdpbmF0aW9uOiBPYmplY3R7Y3VycmVudF9wYWdlLCB0b3RhbF9wYWdlcywgdG90YWxfcmVzdWx0c319CmdldEF4b2xvdGxGYWN0czogUmV0cmlldmUgaW50ZXJlc3RpbmcgZmFjdHMgYWJvdXQgYXhvbG90bHMgc3VjaCBhcyB0aGVpciBoYWJpdHMsIGhhYml0YXRzLCBhbmQgcGh5c2ljYWwgY2hhcmFjdGVyaXN0aWNzLgpQYXJhbWV0ZXJzOiB7ImNhdGVnb3J5IjogInN0cmluZy4gT25lIG9mOiBbaGFiaXRzLCBoYWJpdGF0LCBwaHlzaWNhbCBjaGFyYWN0ZXJpc3RpY3NdLiBUaGUgY2F0ZWdvcnkgb2YgZmFjdHMgdG8gcmV0cmlldmUgKGUuZy4sICdoYWJpdHMnLCAnaGFiaXRhdCcsICdwaHlzaWNhbCBjaGFyYWN0ZXJpc3RpY3MnKS4iLCAibGltaXQiOiAiaW50ZWdlci4gVGhlIG1heGltdW0gbnVtYmVyIG9mIGZhY3RzIHRvIHJldHVybi4ifQpPdXRwdXQ6IFN1Y2Nlc3NmdWwgcmVzcG9uc2UuCiAtIEZvcm1hdDogYXBwbGljYXRpb24vanNvbgogLSBTdHJ1Y3R1cmU6IEFycmF5W09iamVjdHtmYWN0LCBzb3VyY2V9XQoKVXNlIHRoZSBmb2xsb3dpbmcgZm9ybWF0OgpUaG91Z2h0OiB5b3Ugc2hvdWxkIGFsd2F5cyB0aGluayBhYm91dCB3aGF0IHRvIGRvCkFjdGlvbjogdGhlIGFjdGlvbiB0byB0YWtlLCBzaG91bGQgYmUgb25lIG9mIHRoZSB0b29sIG5hbWVzLgpBY3Rpb24gSW5wdXQ6IHRoZSBpbnB1dCB0byB0aGUgYWN0aW9uLCBtdXN0IGJlIGluIEpTT04gZm9ybWF0LiBBbGwgb2YgdGhlIGFjdGlvbiBpbnB1dCBtdXN0IGJlIHJlYWxpc3RpYyBhbmQgZnJvbSB0aGUgdXNlci4KCkJlZ2luIQpRdWVzdGlvbjogSGV5LCBjYW4geW91IHNob3cgbWUgYSByYW5kb20gcGljdHVyZSBvZiBhbiBheG9sb3RsPw==)

Your task is to answer the user’s question using available tools.

You have access to the following tools:

Name: Axolotl

Description: Collection of axolotl pictures and facts

Documentation:

getRandomAxolotlImage: Retrieve a random axolotl image with information on the image source.

Parameters: {}

Output: Successful response.

- Format: application/json

- Structure: Object{url, source, description}

searchAxolotlImages: Search for axolotl images based on specific criteria such as color, gender, and size.

Parameters: {"color": "string. One of: [wild, leucistic, albino]. The color of the axolotl (e.g., ’wild’, ’leucistic’, ’albino’, etc.).", "gender": "string. One of: [male, female]. The gender of the axolotl (’male’, ’female’).", "size": "string. One of: [small, medium, large]. The size of the axolotl (’small’, ’medium’, ’large’).", "page": "integer. The page number for pagination purposes."}

Output: Successful response.

- Format: application/json

- Structure: Object{results: Array[Object{url, source, description}], pagination: Object{current\_page, total\_pages, total\_results}}

getAxolotlFacts: Retrieve interesting facts about axolotls such as their habits, habitats, and physical characteristics.

Parameters: {"category": "string. One of: [habits, habitat, physical characteristics]. The category of facts to retrieve (e.g., ’habits’, ’habitat’, ’physical characteristics’).", "limit": "integer. The maximum number of facts to return."}

Output: Successful response.

- Format: application/json

- Structure: Array[Object{fact, source}]

Use the following format:

Thought: you should always think about what to do

Action: the action to take, should be one of the tool names.

Action Input: the input to the action, must be in JSON format. All of the action input must be realistic and from the user.

Begin!

Question: Hey, can you show me a random picture of an axolotl?

Listing 4: Example user prompt: Tool use

## Appendix F Qualitative Examples

### F.1 Visualization of Advantages

[Figure˜21](#A6.F21 "In F.1 Visualization of Advantages ‣ Appendix F Qualitative Examples ‣ Reinforcement Learning via Self-Distillation") compares the advantages of SDPO and GRPO in a representative example.

!(/html/2601.20802/assets/x18.png)

Figure 21: Visualization of advantages in SDPO and GRPO with Olmo3-7B-Instruct in a batch from the Chemistry task of [Section˜3](#S3 "3 Learning without Rich Environment Feedback ‣ Reinforcement Learning via Self-Distillation"). Each row corresponds to the beginning of a response. The color indicates the advantage value at that token position, with positive advantages shown in blue and negative advantages shown in red.

### F.2 Examples

Below, we show an example from training SDPO on LCBv6 using Qwen3-8B.

[⬇](data:text/plain;base64,dXNlcgpbUHJvbXB0XQoKWW91IGFyZSBhIGNvZGluZyBleHBlcnQuIFlvdSB3aWxsIGJlIGdpdmVuIGEgY29kaW5nIHByb2JsZW0sIGFuZCB5b3UgbmVlZCB0byB3cml0ZSBhIGNvcnJlY3QgUHl0aG9uIHByb2dyYW0gdGhhdCBtYXRjaGVzIHRoZSBzcGVjaWZpY2F0aW9uIGFuZCBwYXNzZXMgYWxsIHRlc3RzLiBUaGUgdGltZSBsaW1pdCBpcyAxIHNlY29uZC4gWW91IG1heSBzdGFydCBieSBvdXRsaW5pbmcgeW91ciB0aG91Z2h0IHByb2Nlc3MuIEluIHRoZSBlbmQsIHBsZWFzZSBwcm92aWRlIHRoZSBjb21wbGV0ZSBjb2RlIGluIGEgY29kZSBibG9jayBlbmNsb3NlZCB3aXRoIGBgYCBgYGAuCgpZb3UgYXJlIGdpdmVuIGEgYmluYXJ5IHN0cmluZyBzIG9mIGxlbmd0aCBuLCB3aGVyZToKCicxJyByZXByZXNlbnRzIGFuIGFjdGl2ZSBzZWN0aW9uLgonMCcgcmVwcmVzZW50cyBhbiBpbmFjdGl2ZSBzZWN0aW9uLgoKWW91IGNhbiBwZXJmb3JtIGF0IG1vc3Qgb25lIHRyYWRlIHRvIG1heGltaXplIHRoZSBudW1iZXIgb2YgYWN0aXZlIHNlY3Rpb25zIGluIHMuIEluIGEgdHJhZGUsIHlvdToKCkNvbnZlcnQgYSBjb250aWd1b3VzIGJsb2NrIG9mICcxJ3MgdGhhdCBpcyBzdXJyb3VuZGVkIGJ5ICcwJ3MgdG8gYWxsICcwJ3MuCkFmdGVyd2FyZCwgY29udmVydCBhIGNvbnRpZ3VvdXMgYmxvY2sgb2YgJzAncyB0aGF0IGlzIHN1cnJvdW5kZWQgYnkgJzEncyB0byBhbGwgJzEncy4KClJldHVybiB0aGUgbWF4aW11bSBudW1iZXIgb2YgYWN0aXZlIHNlY3Rpb25zIGluIHMgYWZ0ZXIgbWFraW5nIHRoZSBvcHRpbWFsIHRyYWRlLgpOb3RlOiBUcmVhdCBzIGFzIGlmIGl0IGlzIGF1Z21lbnRlZCB3aXRoIGEgJzEnIGF0IGJvdGggZW5kcywgZm9ybWluZyB0ID0gJzEnICsgcyArICcxJy4gVGhlIGF1Z21lbnRlZCAnMSdzIGRvIG5vdCBjb250cmlidXRlIHRvIHRoZSBmaW5hbCBjb3VudC4KCkV4YW1wbGUgMToKCklucHV0OiBzID0gIjAxIgpPdXRwdXQ6IDEKRXhwbGFuYXRpb246CkJlY2F1c2UgdGhlcmUgaXMgbm8gYmxvY2sgb2YgJzEncyBzdXJyb3VuZGVkIGJ5ICcwJ3MsIG5vIHZhbGlkIHRyYWRlIGlzIHBvc3NpYmxlLiBUaGUgbWF4aW11bSBudW1iZXIgb2YgYWN0aXZlIHNlY3Rpb25zIGlzIDEuCgpFeGFtcGxlIDI6CgpJbnB1dDogcyA9ICIwMTAwIgpPdXRwdXQ6IDQKRXhwbGFuYXRpb246CgpTdHJpbmcgIjAxMDAiIC0+IEF1Z21lbnRlZCB0byAiMTAxMDAxIi4KQ2hvb3NlICIwMTAwIiwgY29udmVydCAiMTAxMDAxIiAtPiAiMTAwMDAxIiAtPiAiMTExMTExIi4KVGhlIGZpbmFsIHN0cmluZyB3aXRob3V0IGF1Z21lbnRhdGlvbiBpcyAiMTExMSIuIFRoZSBtYXhpbXVtIG51bWJlciBvZiBhY3RpdmUgc2VjdGlvbnMgaXMgNC4KCgpFeGFtcGxlIDM6CgpJbnB1dDogcyA9ICIxMDAwMTAwIgpPdXRwdXQ6IDcKRXhwbGFuYXRpb246CgpTdHJpbmcgIjEwMDAxMDAiIC0+IEF1Z21lbnRlZCB0byAiMTEwMDAxMDAxIi4KQ2hvb3NlICIwMDAxMDAiLCBjb252ZXJ0ICIxMTAwMDEwMDEiIC0+ICIxMTAwMDAwMDEiIC0+ICIxMTExMTExMTEiLgpUaGUgZmluYWwgc3RyaW5nIHdpdGhvdXQgYXVnbWVudGF0aW9uIGlzICIxMTExMTExIi4gVGhlIG1heGltdW0gbnVtYmVyIG9mIGFjdGl2ZSBzZWN0aW9ucyBpcyA3LgoKCkV4YW1wbGUgNDoKCklucHV0OiBzID0gIjAxMDEwIgpPdXRwdXQ6IDQKRXhwbGFuYXRpb246CgpTdHJpbmcgIjAxMDEwIiAtPiBBdWdtZW50ZWQgdG8gIjEwMTAxMDEiLgpDaG9vc2UgIjAxMCIsIGNvbnZlcnQgIjEwMTAxMDEiIC0+ICIxMDAwMTAxIiAtPiAiMTExMTEwMSIuClRoZSBmaW5hbCBzdHJpbmcgd2l0aG91dCBhdWdtZW50YXRpb24gaXMgIjExMTEwIi4gVGhlIG1heGltdW0gbnVtYmVyIG9mIGFjdGl2ZSBzZWN0aW9ucyBpcyA0LgoKCgpDb25zdHJhaW50czoKCjEgPD0gbiA9PSBzLmxlbmd0aCA8PSAxMF41CnNbaV0gaXMgZWl0aGVyICcwJyBvciAnMScKCllvdXIgc29sdXRpb24gc2hvdWxkIGhhdmUgdGhlIGZvbGxvd2luZyBzaWduYXR1cmU6IGBgYHB5dGhvbgpkZWYgbWF4QWN0aXZlU2VjdGlvbnNBZnRlclRyYWRlKHM6IHN0cikgLT4gaW50OgpgYGAKCmFzc2lzdGFudApbUmVzcG9uc2VdCgo8dGhpbms+Cgo8L3RoaW5rPgoKVG8gc29sdmUgdGhpcyBwcm9ibGVtLCB3ZSBuZWVkIHRvIHVuZGVyc3RhbmQgdGhlIHN0cnVjdHVyZSBvZiB0aGUgc3RyaW5nIGFuZCB0aGUgdHJhZGUgb3BlcmF0aW9uLiBIZXJlJ3MgdGhlIHBsYW46CgojIyMgS2V5IEluc2lnaHRzOgoxLiBXZSBhcmUgZ2l2ZW4gYSBiaW5hcnkgc3RyaW5nIGBzYCwgYW5kIHdlIGFyZSBhbGxvd2VkICoqYXQgbW9zdCBvbmUgdHJhZGUqKi4KMi4gQSAqKnRyYWRlKiogY29uc2lzdHMgb2YgdHdvIHN0ZXBzOgogICAtIENvbnZlcnQgYSBjb250aWd1b3VzIGJsb2NrIG9mIGAnMSdgIHRoYXQgaXMgKipzdXJyb3VuZGVkIGJ5IGAnMCdgIGludG8gYCcwJ2AgKGkuZS4sIGZsaXAgaXQpLgogICAtIFRoZW4gY29udmVydCBhIGNvbnRpZ3VvdXMgYmxvY2sgb2YgYCcwJ2AgdGhhdCBpcyAqKnN1cnJvdW5kZWQgYnkgYCcxJ2AgaW50byBgJzEnYCAoaS5lLiwgZmxpcCBpdCkuCjMuIFRoZSBhY3R1YWwgY291bnQgb2YgYWN0aXZlIHNlY3Rpb25zIGlzIHRoZSBudW1iZXIgb2YgYCcxJ2AgY2hhcmFjdGVycyAqKmFmdGVyKiogdGhlIHRyYWRlLiBUaGUgYXVnbWVudGVkIGAnMSdgIGF0IHRoZSBiZWdpbm5pbmcgYW5kIGVuZCBkbyBub3QgY291bnQgaW4gdGhlIGZpbmFsIHJlc3VsdC4KCiMjIyBTdHJhdGVneToKLSBXZSBzaW11bGF0ZSB0aGUgdHJhZGUgYnkgY29uc2lkZXJpbmcgZWFjaCBwb3NzaWJsZSB3YXkgdG8gZmxpcCBhIGJsb2NrIG9mIGAnMSdgIChib3VuZGVkIGJ5IGAnMCdgKSBhbmQgdGhlbiBhIGJsb2NrIG9mIGAnMCdgIChib3VuZGVkIGJ5IGAnMSdgKS4KLSBGb3IgZWFjaCBwb3NzaWJsZSB2YWxpZCBibG9jaywgc2ltdWxhdGUgdGhlIHRyYWRlIGFuZCBjb21wdXRlIHRoZSByZXN1bHRpbmcgbnVtYmVyIG9mIGAnMSdgIGNoYXJhY3RlcnMuCi0gV2Ugb25seSBuZWVkIHRvIGNvbnNpZGVyIHRoZSBvcmlnaW5hbCBzdHJpbmcgYHNgLCBub3QgdGhlIGF1Z21lbnRlZCBvbmUgKHNpbmNlIHdlIGNhbiBzaW11bGF0ZSB0aGUgdHJhZGUgZGlyZWN0bHkgb24gYHNgIGJ5IGNoZWNraW5nIGJvdW5kYXJpZXMpLgoKV2UnbGwgdXNlIGEgc2xpZGluZyB3aW5kb3cgdGVjaG5pcXVlIHRvIGZpbmQgY29udGlndW91cyBibG9ja3Mgb2YgYCcxJ2AgYW5kIGAnMCdgIGFuZCBzaW11bGF0ZSB0aGUgdHJhZGUuCgotLS0KCiMjIyBJbXBsZW1lbnRhdGlvbiBTdGVwczoKMS4gU2ltdWxhdGUgdGhlIHRyYWRlIGJ5IGNoZWNraW5nIGZvciBhIGJsb2NrIG9mIGAnMSdgIHN1cnJvdW5kZWQgYnkgYCcwJ2Agb3IgYXQgdGhlIGVkZ2VzICh3ZSBzaW11bGF0ZSB0aGUgYXVnbWVudGVkIGAxYCBhdCBib3RoIHNpZGVzKS4KMi4gRm9yIGVhY2ggc3VjaCBibG9jaywgc2ltdWxhdGUgY29udmVydGluZyBpdCB0byBgJzAnYCwgdGhlbiBjb252ZXJ0IGEgYCcwJ2AgYmxvY2sgKGFnYWluLCBzdXJyb3VuZGVkIGJ5IGAnMSdgKSB0byBgJzEnYC4KMy4gQ291bnQgdGhlIG51bWJlciBvZiBgJzEnYCBpbiB0aGUgZmluYWwgc3RyaW5nIGFuZCB0cmFjayB0aGUgbWF4aW11bS4KCi0tLQoKIyMjIFB5dGhvbiBDb2RlOgpgYGBweXRob24KZGVmIG1heEFjdGl2ZVNlY3Rpb25zQWZ0ZXJUcmFkZShzOiBzdHIpIC0+IGludDoKICAgIGlmIG5vdCBzOgogICAgICAgIHJldHVybiAwCgogICAgbWF4X29uZXMgPSAwCiAgICBuID0gbGVuKHMpCgogICAgIyBXZSBhcmUgc2ltdWxhdGluZyB0aGUgYXVnbWVudGVkICcxJ3Mgb24gYm90aCBlbmRzLCBzbyB3ZSB0cmVhdCB0aGUgYm91bmRhcmllcyBhcyAnMSdzCiAgICAjIFNvICcwJyBhdCBpbmRleCAwIG9yIG4tMSBpcyBjb25zaWRlcmVkIHN1cnJvdW5kZWQgYnkgJzEncwoKICAgICMgU3RlcCAxOiBGaW5kIGFsbCBjYW5kaWRhdGUgcG9zaXRpb25zIGZvciB0aGUgZmlyc3QgcGFydCBvZiB0aGUgdHJhZGUKICAgICMgKGNvbnZlcnQgYSBibG9jayBvZiAxcyBzdXJyb3VuZGVkIGJ5IDBzIGludG8gMHMpCiAgICBmb3IgaSBpbiByYW5nZShuKToKICAgICAgICBpZiBzW2ldID09ICcxJzoKICAgICAgICAgICAgc3RhcnQgPSBpCiAgICAgICAgICAgIHdoaWxlIGkgPCBuIGFuZCBzW2ldID09ICcxJzoKICAgICAgICAgICAgICAgIGkgKz0gMQogICAgICAgICAgICBlbmQgPSBpIC0gMQogICAgICAgICAgICAjIENoZWNrIGlmIHRoaXMgYmxvY2sgaXMgc3Vycm91bmRlZCBieSAnMCdzCiAgICAgICAgICAgIGxlZnRfYm91bmRhcnkgPSAoc3RhcnQgPT0gMCBvciBzW3N0YXJ0IC0gMV0gPT0gJzAnKQogICAgICAgICAgICByaWdodF9ib3VuZGFyeSA9IChlbmQgPT0gbiAtIDEgb3Igc1tlbmQgKyAxXSA9PSAnMCcpCiAgICAgICAgICAgIGlmIGxlZnRfYm91bmRhcnkgYW5kIHJpZ2h0X2JvdW5kYXJ5OgogICAgICAgICAgICAgICAgIyBUaGlzIGlzIGEgY2FuZGlkYXRlIGJsb2NrIG9mICcxJ3MgdG8gZmxpcAogICAgICAgICAgICAgICAgIyBOb3csIGxvb2sgZm9yIGEgYmxvY2sgb2YgJzAncyBzdXJyb3VuZGVkIGJ5ICcxJ3MgaW4gdGhlIHJlbWFpbmluZyBzdHJpbmcKICAgICAgICAgICAgICAgICMgUmVwbGFjZSB0aGUgJzEnIGJsb2NrIHdpdGggJzAncwogICAgICAgICAgICAgICAgIyBOb3cgZmluZCBhICcwJyBibG9jayBzdXJyb3VuZGVkIGJ5ICcxJ3MKICAgICAgICAgICAgICAgICMgV2UgY2FuIHNpbXVsYXRlIHRoZSByZW1haW5pbmcgc3RyaW5nCiAgICAgICAgICAgICAgICB0ZW1wID0gbGlzdChzKQogICAgICAgICAgICAgICAgZm9yIGogaW4gcmFuZ2Uoc3RhcnQsIGVuZCArIDEpOgogICAgICAgICAgICAgICAgICAgIHRlbXBbal0gPSAnMCcKCiAgICAgICAgICAgICAgICAjIE5vdyBmaW5kIGEgY29udGlndW91cyBibG9jayBvZiAnMCdzIHN1cnJvdW5kZWQgYnkgJzEncwogICAgICAgICAgICAgICAgbWF4X3RyYWRlX29uZXMgPSAwCiAgICAgICAgICAgICAgICBmb3IgayBpbiByYW5nZShuKToKICAgICAgICAgICAgICAgICAgICBpZiB0ZW1wW2tdID09ICcwJzoKICAgICAgICAgICAgICAgICAgICAgICAgc3RhcnQwID0gawogICAgICAgICAgICAgICAgICAgICAgICB3aGlsZSBrIDwgbiBhbmQgdGVtcFtrXSA9PSAnMCc6CiAgICAgICAgICAgICAgICAgICAgICAgICAgICBrICs9IDEKICAgICAgICAgICAgICAgICAgICAgICAgZW5kMCA9IGsgLSAxCiAgICAgICAgICAgICAgICAgICAgICAgICMgQ2hlY2sgaWYgdGhpcyBibG9jayBpcyBzdXJyb3VuZGVkIGJ5ICcxJ3MKICAgICAgICAgICAgICAgICAgICAgICAgbGVmdF8wID0gKHN0YXJ0MCA9PSAwIG9yIHRlbXBbc3RhcnQwIC0gMV0gPT0gJzEnKQogICAgICAgICAgICAgICAgICAgICAgICByaWdodF8wID0gKGVuZDAgPT0gbiAtIDEgb3IgdGVtcFtlbmQwICsgMV0gPT0gJzEnKQogICAgICAgICAgICAgICAgICAgICAgICBpZiBsZWZ0XzAgYW5kIHJpZ2h0XzA6CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAjIEZsaXAgdGhpcyBibG9jayBvZiAnMCdzIHRvICcxJ3MKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGZvciBqIGluIHJhbmdlKHN0YXJ0MCwgZW5kMCArIDEpOgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRlbXBbal0gPSAnMScKICAgICAgICAgICAgICAgICAgICAgICAgICAgICMgQ291bnQgdGhlIG51bWJlciBvZiAnMSdzCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBjb3VudCA9IHN1bSh0ZW1wKQogICAgICAgICAgICAgICAgICAgICAgICAgICAgbWF4X29uZXMgPSBtYXgobWF4X29uZXMsIGNvdW50KQogICAgICAgICAgICAgICAgICAgICAgICAgICAgYnJlYWsgICMgT25seSB0YWtlIHRoZSBmaXJzdCB2YWxpZCBibG9jayBmb3IgdGhpcyB0cmFkZQogICAgICAgICAgICAgICAgIyBBZnRlciB0cnlpbmcgdGhpcyAnMScgYmxvY2ssIHJlc3RvcmUgdGhlIG9yaWdpbmFsIHN0cmluZwogICAgICAgICAgICAgICAgIyBObyBuZWVkLCB3ZSBqdXN0IHVzZWQgYSBjb3B5CiAgICByZXR1cm4gbWF4X29uZXMKYGBgCgotLS0KCiMjIyBPcHRpbWl6YXRpb24gTm90ZToKVGhlIGFib3ZlIGFwcHJvYWNoIGlzICoqbm90IG9wdGltYWwqKiBmb3IgdmVyeSBsYXJnZSBzdHJpbmdzLiBBIG1vcmUgZWZmaWNpZW50IG1ldGhvZCB3b3VsZCBpbnZvbHZlIHByZWNvbXB1dGluZyB0aGUgcG9zaXRpb25zIG9mIGJsb2NrcyBvZiBgJzEnYCBhbmQgYCcwJ2AsIGFuZCBjaGVja2luZyB2YWxpZCB0cmFkZXMgYmV0d2VlbiB0aGVzZSBibG9ja3MgaW4gbGluZWFyIHRpbWUuCgpCdXQgd2l0aCB0aGUgY29uc3RyYWludHMgZ2l2ZW4gKHVwIHRvIDEwXjUpLCB3ZSBuZWVkIHRvIGVuc3VyZSBsaW5lYXIgb3IgbmVhci1saW5lYXIgcGVyZm9ybWFuY2UuIFRoZSBjdXJyZW50IGNvZGUgaXMgY29ycmVjdCBidXQgKipub3QgZWZmaWNpZW50KiogZm9yIGxhcmdlIGlucHV0cy4KCldvdWxkIHlvdSBsaWtlIG1lIHRvIHByb3ZpZGUgYSBtb3JlIG9wdGltaXplZCB2ZXJzaW9uIHRoYXQgcnVucyBpbiAqKk8obikqKiB0aW1lIGFuZCB1c2VzIG9ubHkgb25lIHBhc3MgdGhyb3VnaCB0aGUgc3RyaW5nPwoKdXNlcgpbRmVlZGJhY2tdCgpiJ1J1bnRpbWUgRXJyb3JcblR5cGVFcnJvcjogdW5zdXBwb3J0ZWQgb3BlcmFuZCB0eXBlKHMpIGZvciArOiBcJ2ludFwnIGFuZCBcJ3N0clwnXG5MaW5lIDQ4IGluIG1heEFjdGl2ZVNlY3Rpb25zQWZ0ZXJUcmFkZSAoU29sdXRpb24ucHkpXG5cbkxhc3QgRXhlY3V0ZWQgSW5wdXRcbiIxMTAwMCInCg==)

[Prompt]

You are a coding expert. You will be given a coding problem, and you need to write a correct Python program that matches the specification and passes all tests. The time limit is 1 second. You may start by outlining your thought process. In the end, please provide the complete code in a code block enclosed with ‘‘‘ ‘‘‘.

You are given a binary string s of length n, where:

’1’ represents an active section.

’0’ represents an inactive section.

You can perform at most one trade to maximize the number of active sections in s. In a trade, you:

Convert a contiguous block of ’1’s that is surrounded by ’0’s to all ’0’s.

Afterward, convert a contiguous block of ’0’s that is surrounded by ’1’s to all ’1’s.

Return the maximum number of active sections in s after making the optimal trade.

Note: Treat s as if it is augmented with a ’1’ at both ends, forming t = ’1’ + s + ’1’. The augmented ’1’s do not contribute to the final count.

Example 1:

Input: s = "01"

Output: 1

Explanation:

Because there is no block of ’1’s surrounded by ’0’s, no valid trade is possible. The maximum number of active sections is 1.

Example 2:

Input: s = "0100"

Output: 4

Explanation:

String "0100" -> Augmented to "101001".

Choose "0100", convert "101001" -> "100001" -> "111111".

The final string without augmentation is "1111". The maximum number of active sections is 4.

Example 3:

Input: s = "1000100"

Output: 7

Explanation:

String "1000100" -> Augmented to "110001001".

Choose "000100", convert "110001001" -> "110000001" -> "111111111".

The final string without augmentation is "1111111". The maximum number of active sections is 7.

Example 4:

Input: s = "01010"

Output: 4

Explanation:

String "01010" -> Augmented to "1010101".

Choose "010", convert "1010101" -> "1000101" -> "1111101".

The final string without augmentation is "11110". The maximum number of active sections is 4.

Constraints:

1 <= n == s.length <= 10^5

s[i] is either ’0’ or ’1’

Your solution should have the following signature: ‘‘‘python

def maxActiveSectionsAfterTrade(s: str) -> int:

‘‘‘

[Response]

<think>

</think>

To solve this problem, we need to understand the structure of the string and the trade operation. Here’s the plan:

### Key Insights:

1. We are given a binary string ‘s‘, and we are allowed \*\*at most one trade\*\*.

2. A \*\*trade\*\* consists of two steps:

- Convert a contiguous block of ‘’1’‘ that is \*\*surrounded by ‘’0’‘ into ‘’0’‘ (i.e., flip it).

- Then convert a contiguous block of ‘’0’‘ that is \*\*surrounded by ‘’1’‘ into ‘’1’‘ (i.e., flip it).

3. The actual count of active sections is the number of ‘’1’‘ characters \*\*after\*\* the trade. The augmented ‘’1’‘ at the beginning and end do not count in the final result.

### Strategy:

- We simulate the trade by considering each possible way to flip a block of ‘’1’‘ (bounded by ‘’0’‘) and then a block of ‘’0’‘ (bounded by ‘’1’‘).

- For each possible valid block, simulate the trade and compute the resulting number of ‘’1’‘ characters.

- We only need to consider the original string ‘s‘, not the augmented one (since we can simulate the trade directly on ‘s‘ by checking boundaries).

We’ll use a sliding window technique to find contiguous blocks of ‘’1’‘ and ‘’0’‘ and simulate the trade.

---

### Implementation Steps:

1. Simulate the trade by checking for a block of ‘’1’‘ surrounded by ‘’0’‘ or at the edges (we simulate the augmented ‘1‘ at both sides).

2. For each such block, simulate converting it to ‘’0’‘, then convert a ‘’0’‘ block (again, surrounded by ‘’1’‘) to ‘’1’‘.

3. Count the number of ‘’1’‘ in the final string and track the maximum.

---

### Python Code:

‘‘‘python

def maxActiveSectionsAfterTrade(s: str) -> int:

if not s:

return 0

max\_ones = 0

n = len(s)

# We are simulating the augmented ’1’s on both ends, so we treat the boundaries as ’1’s

# So ’0’ at index 0 or n-1 is considered surrounded by ’1’s

# Step 1: Find all candidate positions for the first part of the trade

# (convert a block of 1s surrounded by 0s into 0s)

for i in range(n):

if s[i] == ’1’:

start = i

while i < n and s[i] == ’1’:

i += 1

end = i - 1

# Check if this block is surrounded by ’0’s

left\_boundary = (start == 0 or s[start - 1] == ’0’)

right\_boundary = (end == n - 1 or s[end + 1] == ’0’)

if left\_boundary and right\_boundary:

# This is a candidate block of ’1’s to flip

# Now, look for a block of ’0’s surrounded by ’1’s in the remaining string

# Replace the ’1’ block with ’0’s

# Now find a ’0’ block surrounded by ’1’s

# We can simulate the remaining string

temp = list(s)

for j in range(start, end + 1):

temp[j] = ’0’

# Now find a contiguous block of ’0’s surrounded by ’1’s

max\_trade\_ones = 0

for k in range(n):

if temp[k] == ’0’:

start0 = k

while k < n and temp[k] == ’0’:

k += 1

end0 = k - 1

# Check if this block is surrounded by ’1’s

left\_0 = (start0 == 0 or temp[start0 - 1] == ’1’)

right\_0 = (end0 == n - 1 or temp[end0 + 1] == ’1’)

if left\_0 and right\_0:

# Flip this block of ’0’s to ’1’s

for j in range(start0, end0 + 1):

temp[j] = ’1’

# Count the number of ’1’s

count = sum(temp)

max\_ones = max(max\_ones, count)

break # Only take the first valid block for this trade

# After trying this ’1’ block, restore the original string

# No need, we just used a copy

return max\_ones

‘‘‘

---

### Optimization Note:

The above approach is \*\*not optimal\*\* for very large strings. A more efficient method would involve precomputing the positions of blocks of ‘’1’‘ and ‘’0’‘, and checking valid trades between these blocks in linear time.

But with the constraints given (up to 10^5), we need to ensure linear or near-linear performance. The current code is correct but \*\*not efficient\*\* for large inputs.

Would you like me to provide a more optimized version that runs in \*\*O(n)\*\* time and uses only one pass through the string?

[Feedback]

b’Runtime Error\nTypeError: unsupported operand type(s) for +: \’int\’ and \’str\’\nLine 48 in maxActiveSectionsAfterTrade (Solution.py)\n\nLast Executed Input\n"11000"’

### F.3 Environment Feedback

We show three examples of feedback in our coding environment, inspired by LeetCode.

[⬇](data:text/plain;base64,VGVzdCBDYXNlIDM6IFdyb25nIEFuc3dlcgoKSW5wdXQKW1szLDYsMV0sWzQsMjEsNF1dCgpPdXRwdXQKNgoKRXhwZWN0ZWQKMjIuODc1MDAKClRlc3QgQ2FzZSA2OiBXcm9uZyBBbnN3ZXIKCklucHV0CltbMTIsMjUsM10sWzMsMTQsMl1dCgpPdXRwdXQKMTQKCkV4cGVjdGVkCjI1LjgzMzMzCg==)

Test Case 3: Wrong Answer

Input

[[3,6,1],[4,21,4]]

Output

6

Expected

22.87500

Test Case 6: Wrong Answer

Input

[[12,25,3],[3,14,2]]

Output

14

Expected

25.83333

Listing 5: Example of feedback “Wrong Answer” from our code environment in case of a wrong answer,
inspired by LeetCode

[⬇](data:text/plain;base64,UnVudGltZSBFcnJvcgpNZW1vcnlFcnJvcjoKTGluZSA5MSBpbiA8bW9kdWxlPiAoU29sdXRpb24ucHkpCkxpbmUgMjUgaW4gc29sdmUgKFNvbHV0aW9uLnB5KQoKTGFzdCBFeGVjdXRlZCBJbnB1dAoxMAo2MzMgOTMxMgoxMzE0IDg1NDgKODg1NyAxMDYyCjY0MTAgMzI4OQo4NTk0IDEyNjMKODU0OSA3MzMKMzg1OCA1OTczCi4uLiAoMyBtb3JlIGxpbmVzKQ==)

Runtime Error

MemoryError:

Line 91 in <module> (Solution.py)

Line 25 in solve (Solution.py)

Last Executed Input

10

633 9312

1314 8548

8857 1062

6410 3289

8594 1263

8549 733

3858 5973

... (3 more lines)

Listing 6: Example of feedback “Memory Error” from our code environment in case of a wrong answer,
inspired by LeetCode

[⬇](data:text/plain;base64,UnVudGltZSBFcnJvcgpJbmRleEVycm9yOiBsaXN0IGluZGV4IG91dCBvZiByYW5nZQpMaW5lIDI4IGluIHNvcnRNYXRyaXggKFNvbHV0aW9uLnB5KQoKTGFzdCBFeGVjdXRlZCBJbnB1dApbWy0xLC0xLC0xLC0xLC0xLC0xLC0xLC0xLC4uLg==)

Runtime Error

IndexError: list index out of range

Line 28 in sortMatrix (Solution.py)

Last Executed Input

[[-1,-1,-1,-1,-1,-1,-1,-1,...

Listing 7: Example of feedback “Index Error” from our code environment in case of a wrong answer,
inspired by LeetCode

### F.4 Illustrative Example

Figure [22](#A6.F22 "Figure 22 ‣ F.4 Illustrative Example ‣ Appendix F Qualitative Examples ‣ Reinforcement Learning via Self-Distillation") shows an illustrative example of the dense credit assignment in SDPO.

!(/html/2601.20802/assets/x19.png)

Figure 22: Dense credit assignment through self-teaching in SDPO. The answer is generated by then model (Qwen3-8B) before seeing the feedback. Then, we re-evaluate the log-probs of the original attempt with the self-teacher after seeing the feedback. We show the per-token log⁡(ℙ​(self-teacher)/ℙ​(student))\log(\nicefrac{{\mathbb{P}\left(\text{self-teacher}\right)}}{{\mathbb{P}\left(\text{student}\right)}}), with red indicating negative values (self-teacher disagrees), blue indicating positive values (teacher reinforces), and white indicating values around zero. Using binary rewards, GRPO would assign the same, negative advantage to all tokens in the sequence. In contrast, SDPO turns the feedback into dense credit assignment across the sequence. The first row shows the tokens of the generated response. The 3 other rows show the top-kk logits of the self-teacher that are used during self-distillation, suggesting alternative tokens. Notably, in this example, the self-teacher identifies the error through retrospection without an explicit solution. The credit assignment on the generated sequence, and the alternative top-kk logits correctly show that replacing set with dict maintains the order of elements. Further, in the seventh shown position, the model also identifies an alternative solution path which starts with the seen token, instead of directly returning the output. The activation is sparse, identifying where mistakes happen and adjusting to the students’ response distribution for specifically these few tokens.
