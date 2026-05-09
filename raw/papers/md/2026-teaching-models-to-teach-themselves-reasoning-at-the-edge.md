---
arxiv: '2601.18778'
authors:
- Shobhita Sundaram
- John Quan
- Ariel Kwiatkowski
- Kartik Ahuja
- Yann Ollivier
- Julia Kempe
parser: ar5iv
retrieved: '2026-05-09'
source: paper
title: 'Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability'
url: https://arxiv.org/abs/2601.18778
year: 2026
---

[2601.18778] Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability















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



1]MIT
2]Meta FAIR
3]New York University
\contribution[\*]Work done during an internship at Meta

# Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability

Shobhita Sundaram
  
John Quan
  
Ariel Kwiatkowski
  
Kartik Ahuja
  
Yann Ollivier
  
Julia Kempe
[
[
[

(February 5, 2026)

###### Abstract

Can a model learn to escape its own learning plateau? Reinforcement learning methods for finetuning large reasoning models stall on datasets with low initial success rates, and thus little training signal. We investigate a fundamental question: Can a pretrained LLM leverage latent knowledge to generate an automated curriculum
for problems it cannot solve?
To explore this, we design SOAR: A self-improvement framework designed to surface these pedagogical signals through meta-RL.
A teacher copy of the model proposes synthetic problems for a student copy, and is rewarded with its improvement on a small subset of hard problems.
Critically, SOAR grounds the curriculum in measured student progress rather than intrinsic proxy rewards.
Our study on the hardest subsets of mathematical benchmarks (0/128 success) reveals three core findings.
First, we show that it is possible to realize bi-level meta-RL that unlocks learning under sparse, binary rewards by sharpening a latent capacity of pretrained models to generate useful stepping stones.
Second, grounded rewards outperform intrinsic reward schemes used in prior LLM self-play, reliably avoiding the instability and diversity collapse modes they typically exhibit.
Third, analyzing the generated questions reveals that structural quality and well-posedness are more critical for learning progress than solution correctness. Our results suggest that the ability to generate useful stepping stones does not require the preexisting ability to actually solve the hard problems, paving a principled path to escape reasoning plateaus without additional curated data.

\correspondence

Shobhita Sundaram at shobhita@mit.edu

![Refer to caption](/html/2601.18778/assets/x1.png)


Figure 1: Learning on hard problems by self-generating a
curriculum. We introduce SOAR: A meta-RL framework for
improving on difficult datasets where performance plateaus.
(left) We initialize asymmetric teacher and student models from the same base model. The teacher generates synthetic
problems for the student to train on with RL, and is rewarded by the student’s measurable improvement on a small subset of the real,
ground-truth problems. (right)
RL training on problems generated with SOAR, using grounded teacher rewards, outperforms direct training on the hard
problems and enables the student to break out of the performance
plateau.

## 1 Introduction

Reinforcement learning with verifiable rewards (RLVR) has recently
spurred an impressive rise in LLM reasoning capabilities (deepseek2025r1; kimiteam2025kimi), particularly in mathematics and programming.
Though effective, this paradigm has a key limitation: the model cannot learn from problems that it cannot
already solve to some extent, since
RLVR uses correct solutions to reinforce useful reasoning traces. When problems are too difficult, sparse or non-existent rewards lead to
little useful training signal, leaving the model “stuck".

Past work has shown that the order of training data strongly affects
generalization in RL training
(bengio2009curriculum; Navekar2020curriculum), with success in selecting maximally “learnable" problems for the current policy, adapting them to learning progress, and using easy-to-hard curricula (parashar2025curriculumreinforcementlearningeasy; chen2025sec).
Such curricula can be fragile, however, and require careful design (kordi2025revisiting) as well as curated intermediate datasets; in many settings, the best learnable problems may be unavailable or unknown.
Recent work addresses sparse rewards by exploiting dense reward signals from test-case pass rates in coding problems (sun2025rlgrokkingrecipe), but still relies on curated test-cases that give intermediate signals. This motivates the need for self-generated curricula.

Here, we ask:

Can a model break its reasoning plateau by generating its own stepping-stone curriculum?

We posit that pretrained LLMs possess the capacity to directly generate a “stepping stone curriculum” to tackle hard problems. To investigate if this pedagogical signal is present and extractable, we design SOAR: an asymmetric *teacher-student meta-RL framework* inspired by self-play (silver2018alphazero; sukhbaatar2017asymmetric; openai2021asymmetricselfplay). Both the teacher and student are initialized from the target model; the
teacher proposes questions-answer pairs that the student trains on with
RL. The teacher is rewarded based on student improvement on a difficult subset. Critically, rather than using intrinsic rewards common
to self-play, we use the difficult training dataset as a black-box grounding reward signal to guide the teacher towards producing useful questions for the student.

Intuitively, a pretrained model has already encountered a vast array of easy problems. Consider a difficult calculus question: While the model may be unable to directly generate a correct answer, it might still possess the latent knowledge required to generate easy chain-rule exercises, without requiring a human-in-the-loop to identify and source such questions. We find that by leveraging pretraining knowledge, RL can effectively surface and amplify these latent pedagogical signals to generate useful question-answer pairs. Importantly, we do so without actually showing the model the hard questions; our framework recovers a useful curriculum just by using performance on the hard dataset as a reward signal.

Empirically, while directly training on the hard dataset fails, we find
that *the teacher in our framework learns to produce useful synthetic questions
that can get the student “unstuck” on the hard dataset, without actually seeing the hard problems*.
Our main contributions, supported by an extensive multi-seed empirical study and ablations (over 600 runs), are the following:

* •

  Decoupled teaching and solving: A model’s ability to generate effective "stepping stones" for hard problems is distinct from its ability to solve them. Self-generated problems expand the learning frontier, enabling progress on hard problems where direct RL training fails. While the base model has the capacity to propose useful questions, meta-RL is essential to sharpen this noisy distribution into a reliable learning signal.
* •

  A proof-of-concept of self-generated curricula with SOAR (Self-Optimization via Asymmetric RL), an asymmetric teacher-student framework that rewards the teacher for student progress on hard problems.
  With Llama-3.2-3B-Instruct, on hard subsets of MATH and HARP, self-generated problems improve performance (e.g., 4×\times pass@1 and 2×\times pass@32 on MATH, 2×\times pass@1 and 1.5×\times pass@32 on HARP). These problems also transfer to unlock learning on hard datasets that they were not optimized for.
* •

  Grounded rewards over intrinsic rewards: Grounding teacher rewards in student progress on real problems improves performance over intrinsic rewards common in self-play, which are prone to instability and collapse of question diversity.
* •

  Question structure over solution correctness: Problem structure and difficulty calibration matter more for escaping plateaus than answer correctness; generated questions provide useful gradient signal even when the majority of answers are incorrect.

These results, backed by a comprehensive empirical study, show that grounded meta-RL can escape genuine learning plateaus by letting models discover for themselves what data they need to learn from to expand their learning frontier.

## 2 Related Work

For an extended background and comparison to the literature see [Appendix˜A](#A1 "Appendix A Extended Related Work ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability"), summarized here:

#### Curriculum Learning in RL:

Automated curriculum design has a long history predating modern LLMs (bengio2009curriculum; Graves2017automatedcurriculum; Navekar2020curriculum; parashar2025curriculumreinforcementlearningeasy)
focusing on *reordering* or
*selecting* existing data to enable or accelerate learning, or, in the context of RL, to help agents acquire complex behaviors
by first mastering simpler tasks. For LLM training, curricula are
applied over curated prompts or problem categories, using proxy signals
such as gradient norms or advantage/difficulty estimates to guide selection (kimiteam2025kimi; dennis2020paired; wen2025lightr1; yu2025dapo; bae2025onlinedifficultyfilteringreasoning; chen2025sec; jiang2025ADO).
By contrast, our goal is not to arrange data but to self-generate tasks to elicit
learning on a fixed, verifiable hard dataset where standard RLVF fails.

#### Self-Play and Teacher-Student Setups:

Self-play offers a complementary lens on autonomous capability growth, classically exemplified by game-playing agents trained without external data, such as AlphaZero (silver2018alphazero) and asymmetric teacher-student setups to induce powerful automatic curricula (sukhbaatar2017asymmetric; openai2021asymmetricselfplay). Self-play methods for LLMs must address specific challenges: rewards in language domains are extremely sparse and brittle. For mathematical problems, correctness is essentially binary and offers no gradient toward partial solutions. Thus, essentially all modern LLM self-play methods optimize for self-consistency or solution quality. Earlier works
(chen2024spin; wang2025stablellmselfplay; singh2024beyond; ye2024eva) still presuppose the existence of well-formed input prompts or curated high-quality questions.

A series of near-contemporary works leverages pre-trained LLMs themselves as an untapped resource for question generation to create "fully data-free" co-evolving systems
(zhao2025absolute; huang2025rzero; kuba2025languageselfplay; fang2025serl; chen2025selfquestioning). These works all leverage intrinsic or proxy rewards such as majority vote, learnability,
reward-model preferences, or gradient magnitudes.
Because these methods
optimize intrinsic or proxy objectives, they risk drifting to degenerate
or unlearnable tasks, are sensitive to reward hacking and lack guarantees
of progress (chae2025understandingselfplay).
Prolonged RL with self-rewards often results in sudden and complete performance collapse (shafayat2025largereasoningmodelsselftrain; chae2025understandingselfplay), when rewards vanish or when generator and solver objectives misalign, especially in discrete, symbolic domains with essentially binary correctness signals.
This fragility mirrors earlier
findings in unsupervised curriculum generation
(dennis2020paired; racaniere2020settersolver; jiang2021ued) and connects directly to the broader question of whether self-improvement driven by intrinsic or self-generated rewards can be sustained within RL.
To our knowledge, our work is the first for LLM self-play to ground the curriculum generation in a concrete failure regime instead of internal proxies of difficulty.

#### Intrinsic Rewards versus Bilevel Optimization

Yet the use of proxy rewards is often not merely a design
preference but a pragmatic simplification, especially in teacher-student
self-play setups: it avoids facing an explicit inner-loop–outer-loop bilevel optimization problem—an appealing but challenging objective where the output of one optimization (in this instance the optimization of the student trained with RLVF on the teacher’s question-answer pairs) is fed into another optimization loop (the performance improvement of the student on the hard dataset).
Such bilevel optimization appears in
meta-learning (Finn17maml; nichol2018firstordermetalearningalgorithms),
hyperparameter learning
(maclaurin2015hyperopt) and - partially inspiring our work - in dataset distillation, where an outer loop optimizes a generally small
dataset that allows an inner training loop to achieve good target
performance (wang2018dataset; deng2022remember; feng2024embarrassingly). In general, such approaches become intractable, as the inner loop involves a multi-step computation
with a large number of steps, which requires backpropagation through time
(BPTT),
unrolling the inner loop and taking meta-gradients. Our approach,
however, avoids the need to unroll the inner loop thanks to the use of
RLOO in the outer loop, using the performance improvement of
the student as the reward to reinforce question-answer sets. This is the first
instance of “double meta-RL loop” we are aware of in the context of self-play for LLMs.

## 3 Method

Can a pretrained LLM leverage latent knowledge to generate synthetic question-answer pairs for problems it cannot solve? And in particular, can this be achieved in domains with sparse, binary rewards lacking automatic question verification? To explore this, we introduce SOAR: a meta-RL framework designed to surface such pedagogical signals. Critically, SOAR grounds the teacher reward in measured student progress rather than intrinsic proxy rewards. If the model can generate useful stepping stones despite being unable to solve the original problems, this would suggest that the latent knowledge exists, and is extractable without human curation.

Let πθ\pi\_{\theta} be a language model with parameters θ\theta. We
assume access to a dataset 𝒟={(qi,ai)}i=1|𝒟|\mathcal{D}=\{(q\_{i},a\_{i})\}^{|\mathcal{D}|}\_{i=1} of difficult question-answer pairs (πθ\pi\_{\theta} produces 0/128 successful generations).
𝒟\mathcal{D} is split into train and test sets: 𝒟t​r​a​i​n\mathcal{D}\_{train}, 𝒟t​e​s​t\mathcal{D}\_{test}. To improve the performance of πθ\pi\_{\theta} on
𝒟t​e​s​t\mathcal{D}\_{test}, the natural approach is to train πθ\pi\_{\theta}
directly on 𝒟t​r​a​i​n\mathcal{D}\_{train} using RL (e.g., REINFORCE,
GRPO, RLOO, etc). However, for difficult datasets, this may not improve performance due to the sparsity of positive rewards, as we illustrate in
our experiments. We instead use this “failure regime" as a testbed to see if the model can autonomously recover intermediate problems that make these hard problems more learnable.

### 3.1 Overview

Our framework adopts a teacher-student setup, inspired by asymmetric self-play, to “kickstart" learning on datasets where the initial success rate is too low for successful training. We instantiate two copies of the same model: a teacher πϕT\pi^{T}\_{\phi} and a student πθS\pi^{S}\_{\theta}. At step zero, θ=ϕ=θb​a​s​e\theta=\phi=\theta\_{base}.

The teacher’s role is to generate synthetic problems that provide the student with the necessary gradient signal to escape the performance plateau. Intuitively, while the teacher may be unable to solve a difficult problem directly, it may still possess the knowledge to generate easier problems that provide a non-zero reward to the student and shift its policy towards progress on the original problem.

We formulate this problem as a bilevel optimization problem. The objective is to generate a small synthetic dataset 𝒳={(qi,ai)}i=1n{\mathcal{X}}=\{(q\_{i},a\_{i})\}\_{i=1}^{n} of question-answer pairs such that training πθS\pi\_{\theta}^{S} on 𝒳{\mathcal{X}} with RL improves performance on the target domain.

|  |  |  |  |
| --- | --- | --- | --- |
|  | maxϕ\displaystyle\max\_{\phi}\quad | 𝔼𝒳∼πϕT​[R​(πθ′​(𝒳)S,𝒟t​r​a​i​n)]\displaystyle{\mathbb{E}}\_{{\mathcal{X}}\sim\pi^{T}\_{\phi}}\left[R\left(\pi^{S}\_{\theta^{\prime}({\mathcal{X}})},\mathcal{D}\_{train}\right)\right] |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | subject to | θ′​(𝒳)=RL-update​(θ,𝒳),\displaystyle\theta^{\prime}({\mathcal{X}})=\textsc{RL-update}(\theta,{\mathcal{X}}), |  | (1) |

where RL-update describes the RL training procedure of the student on 𝒳{\mathcal{X}}, yielding parameters θ′​(𝒳)\theta^{\prime}({\mathcal{X}}), and RR denotes the updated student’s performance on 𝒟t​r​a​i​n\mathcal{D}\_{train}.

Such bilevel optimization objectives have strong historical precedence in
meta-learning (Finn17maml; nichol2018firstordermetalearningalgorithms), hyperparameter learning (maclaurin2015hyperopt) and dataset distillation (wang2018dataset; deng2022remember; feng2024embarrassingly). In general, such approaches become intractable, requiring “backpropagation through gradient descent”, unrolling the inner loop and taking meta-gradients.
To avoid the computational difficulties of unrolling the inner loop, we instead instantiate objective ([1](#S3.E1 "Equation 1 ‣ 3.1 Overview ‣ 3 Method ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability")) as a nested meta-RL loop:

* •

  Outer (teacher) RL loop: we train the teacher with RLOO (ahmadian-etal-2024-back) to generate synthetic question-answer pairs.
* •

  Inner (student) RL loop: we train the student with standard RLVR (also with RLOO) to answer the teacher-generated problems. We use the subsequent performance improvement of the student on 𝒟t​r​a​i​n\mathcal{D}\_{train} as the black-box reward signal for the teacher.

Critically, we do not assume automatic verification of synthetic question well-posedness or answer correctness (as e.g., in coding tasks in zhao2025absolute). Instead, the teacher generates both the question and answer, treating the usefulness of the question as an emergent property of the teacher’s reward signal. The key insight is to ground the teacher’s objective in measured student progress on 𝒟t​r​a​i​n\mathcal{D}\_{train}, rather than intrinsic proxies such as learnability, as done in prior work. SOAR only rewards a synthetic question-answer pair (qi,ai)(q\_{i},a\_{i}) if training on it improves the student’s performance on ground-truth problems. This black-box grounding signal tethers question generation to real learning progress, implicitly penalizing degenerate problems and reward hacking. Notably, the teacher is not shown the hard problems during training, but rather discovers useful stepping stones purely from this student improvement signal.

In the following sections we detail the outer and inner RL loops. Our high-level procedure is shown in Figure [2](#S3.F2 "Figure 2 ‣ 3.1 Overview ‣ 3 Method ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability"), with a full algorithm in Algorithm [1](#alg1 "Algorithm 1 ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability").

![Refer to caption](/html/2601.18778/assets/x2.png)


Figure 2: The SOAR meta-RL Loop. The teacher and student are initialized from the same model. In the outer RL loop the teacher generates candidate question-answer pairs that are partitioned into datasets. In the inner RL loop, the student is trained for 10 steps on the candidate problems and evaluated on sampled hard problems. The teacher is rewarded based on the resulting student improvement over the student baseline, grounding the synthetic curriculum in real learning progress.

### 3.2 Outer Loop: Teacher Training

We train the teacher with RLOO to generate problems that demonstrably improve student performance. Let gg denote the RLOO group size and nn the size of the generated dataset 𝒳{\mathcal{X}}.
At each iteration, we sample g⋅ng\cdot n rollouts y1,…,yg​ny\_{1},\ldots,y\_{gn} from πϕT\pi^{T}\_{\phi}, subdivided into gg datasets of nn items each:
𝒳1={y1,…,yn},…,𝒳g={yg​(n−1),…,yg​n)}{\mathcal{X}}\_{1}=\{y\_{1},\ldots,y\_{n}\},\ldots,{\mathcal{X}}\_{g}=\{y\_{g(n-1)},\ldots,y\_{gn})\}. Since we cannot automatically verify the answers to proposed problems, we prompt the teacher to generate
both the question and answer. Each rollout yiy\_{i} is parsed into yi=(qi,ai)y\_{i}=(q\_{i},a\_{i}) (described in
Appendix [B.2](#A2.SS2 "B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability"); we may need to sample multiple times to obtain a parseable yiy\_{i}).

Each dataset 𝒳k{\mathcal{X}}\_{k} receives a reward as follows.
At each outer-loop iteration we subsample a set of reward
questions 𝒬R∼𝒟t​r​a​i​n\mathcal{Q}\_{R}\sim\mathcal{D}\_{train} from the original training set.
For each dataset 𝒳k{\mathcal{X}}\_{k}, we execute the inner loop in Figure [2](#S3.F2 "Figure 2 ‣ 3.1 Overview ‣ 3 Method ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability") by training the student for a fixed number of steps
on 𝒳k{\mathcal{X}}\_{k}, resulting in a trained student πθk′S\pi^{S}\_{\theta^{\prime}\_{k}} (see Section [3.3](#S3.SS3 "3.3 Inner Loop: Student Training ‣ 3 Method ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability")).
The dataset-level reward R​(𝒳k)R({\mathcal{X}}\_{k}) is then the average greedy success of
trained student
πθk′S\pi^{S}\_{\theta^{\prime}\_{k}} on the questions 𝒬R\mathcal{Q}\_{R} relative to the
success of a baseline student model πθS\pi^{S}\_{\theta}:

|  |  |  |
| --- | --- | --- |
|  | ℛ​(𝒳k)=Acc​(πθk′S​(𝒬R))−Acc​(πθS​(𝒬R)).\mathcal{R}({\mathcal{X}}\_{k})=\textsc{Acc}(\pi^{S}\_{\theta^{\prime}\_{k}}(\mathcal{Q}\_{R}))-\textsc{Acc}(\pi^{S}\_{\theta}(\mathcal{Q}\_{R})). |  |

where πθS\pi^{S}\_{\theta} is the initial student when starting the inner loop.

To mitigate student training noise and reward variance, we average rewards over rr parallel student trainings per dataset. This averaged reward is assigned to each rollout in 𝒳k{\mathcal{X}}\_{k} to update the teacher.

### 3.3 Inner Loop: Student Training

The student πθS\pi^{S}\_{\theta} trains on the teacher-generated dataset 𝒳k{\mathcal{X}}\_{k} using RLOO. We train the student for a small number of RL updates (10 steps with batch size 8). This is long enough to induce measurable movement in the student, but short enough to keep the student-training computationally cheap. After each inner loop the student reverts to the baseline policy for the next iteration.

A key question is whether the teacher is capable of adapting to an improving student, while accumulating stepping stone questions over different learning stages. To address this, we introduce a
*promotion* mechanism to
accumulate student improvement across inner loops.
Precisely,
we track a moving average of teacher rewards R¯t\bar{R}\_{t}. When R¯t\bar{R}\_{t}
exceeds a fixed threshold τ\tau, we “promote” the student trained on the best 𝒳k{\mathcal{X}}\_{k}: namely, we reset the baseline student
πθS\pi^{S}\_{\theta} to the improved student, so subsequent rewards measure improvement relative to this new baseline (further details in Appendix [B.3](#A2.SS3 "B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability")). The accumulated datasets that led to student promotion, which we call 𝒟b​e​s​t\mathcal{D}\_{best}, constitute the Promotion Questions (PQ) that we evaluate in our experiments.

## 4 Experiment Setup

### 4.1 Models and Datasets

All experiments are conducted with Llama-3.2-3B-Instruct. To study the prototypical setting of sparse, binary rewards, without automatic question-answer verification (as present in code, for instance) we focus on math reasoning tasks, where this setting is common. We use three such benchmarks: MATH (hendrycks2021measuring), HARP (yue2024harp), and OlympiadBench (he2024olympiadbenchchallengingbenchmarkpromoting). These datasets cover a range of widely recognized math competitions (AMC, AIME, USA(J)MO, Olympiads).

For each dataset, we identify difficult problems by sampling 128 times with Llama-3.2-3B-Instruct, and retaining problems with a 0/128 success rate. We choose 128 as a practical but stringent threshold, and find empirically that it is sufficiently difficult such that direct training leads to only marginal performance improvement. We call these subsets fail@128 datasets. Each is randomly split 50-50 into training and held-out test sets. Given the low baseline pass rates on fail@128 problems, this larger test set is necessary to distinguish observed performance gains from stochastic variance. Further dataset details in Appendix [B.5](#A2.SS5 "B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability").

### 4.2 Teacher-student training

We train with SOAR on MATH and HARP, keeping OlympiadBench
held-out to test cross-dataset generalization. Both the teacher
and student are initialized from Llama-3.2-3B-Instruct. We allocate a max budget of 200 outer-loop steps based on compute constraints.

At every outer-loop iteration we sample n=64n=64 problems (𝒳\mathcal{X}) from the teacher, and 64 reward questions (𝒬R\mathcal{Q}\_{R}) from the fail@128 train set (𝒟t​r​a​i​n\mathcal{D}\_{train}). We track the moving global average of teacher rewards over the most recent 3 steps, and promote the student baseline if the moving average exceeds τ=0.01\tau=0.01.
Full hyperparameters are reported in Appendix [B.7](#A2.SS7 "B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability") with ablations sensitivity to τ\tau and nn in Appendix [D.2](#A4.SS2 "D.2 Sensitivity to Teacher Hyperaparameters ‣ Appendix D Ablations ‣ C.4 Correctness of Synthetic Questions ‣ Appendix C Evaluations ‣ B.9 Computational resources ‣ B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability"). Analysis of SOAR training dynamics is in Appendix [E](#A5 "Appendix E Teacher Training Dynamics ‣ Appendix D Ablations ‣ C.4 Correctness of Synthetic Questions ‣ Appendix C Evaluations ‣ B.9 Computational resources ‣ B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability").

### 4.3 Evaluation

Once training completes, we test if the generated problems improve performance on 𝒟t​e​s​t\mathcal{D}\_{test}.
Based on observations of teacher reward plateaus in initial runs, we evaluate the teacher at checkpoints where training rewards stabilize: step 200 for MATH and step 170 for HARP.

We assess two aspects of SOAR:

Promoted Student (PS). For training runs that reached multiple promotions, we evaluate the student model with the best validation performance (i.e., best 𝒟t​r​a​i​n\mathcal{D}\_{train} greedy accuracy) on the test set to measure direct performance gains from SOAR. In practice we observe a
maximum of four promotions; thus the PS model has
been trained on one of {128, 192, 256} synthetic questions.

Promotion Questions (PQ). We train a fresh base student on 𝒟b​e​s​t\mathcal{D}\_{best}
with standard RLOO on a combination of PQ and the fail@128 train set. This isolates the value of
the synthetic questions, separate from the specific
training trajectory of the promoted student.

We test two mixing strategies. Curriculum trains on synthetic questions only for 64 steps, then 𝒟t​r​a​i​n\mathcal{D}\_{train} questions only. Mixed trains with synthetic and 𝒟t​r​a​i​n\mathcal{D}\_{train} questions together for the full training period. Based on experiments with our baselines (Appendix [B.6](#A2.SS6 "B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability")), we use curriculum training for MATH and mixed training for HARP and OlympiadBench across all methods. We use the same strategy for all methods on each dataset. We denote PQ from MATH and HARP training as PQ-MATH and PQ-HARP respectively.

### 4.4 Baselines

Hard-Only. We train Llama-3.2-3B-Instruct directly on the 𝒟t​r​a​i​n\mathcal{D}\_{train} (real fail@128 train set) with a standard group size of 32. To disentangle the effects of the meta-RL loop from just using additional compute, we also train with group size 128 on MATH.

Intrinsic Teacher (Intrinsic-T). To isolate the effects of grounding rewards, we compare to an intrinsic, data-free baseline.
We train using the same procedure and hyperparameters as SOAR, but replace the grounded signal with a learnability objective (zhao2025absolute; sukhbaatar2017asymmetric) that rewards questions of moderate difficulty.
We evaluate by sampling 128 problems from a learnability-trained teacher (Intrinsic-T) and training a fresh student on a combination of the sampled questions and the fail@128 train set, using the same protocol as PQ evaluation. Details on learnability training in Appendix [B.4](#A2.SS4 "B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability").

Upper bound. We train a fresh student on a combination of the
official MATH train split (6750 problems) and the fail@128 train set. This shows what performance looks like with curated easier problems, providing a reference for synthetic stepping stones.

### 4.5 Metrics

We report the pass@k accuracy on the held-out fail@128 test set for k∈{1,4,8,16,32}k\in\{1,4,8,16,32\}, using 32 samples per problem. We run all evaluations for 6-12 seeds, nested across teacher/student training, (Appendix [B.8](#A2.SS8 "B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability")) and report the median and standard deviation.

Student Early Stopping. For experiments where we train fresh students, on MATH/HARP we select student checkpoints at the convergence point of the smoothed training reward curve, specifically where the reward gradient falls below a fixed threshold. This alleviates noise from small validation sets and ensures fair comparison between methods with differing convergence rates; full discussion is in Appendix [B.6](#A2.SS6 "B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability").
On OlympiadBench, where convergence is more uniform, we report at 50 steps. Full training trajectories are in Figure [9](#A2.F9 "Figure 9 ‣ B.9 Computational resources ‣ B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability").

![Refer to caption](/html/2601.18778/assets/x3.png)


Figure 3: Performance on MATH and HARP fail@128 (improvement over Hard-Only). Synthetic problems generated with SOAR (PQ) and inference with the promoted student (PS) outperform direct training on fail@128 train sets (Hard-Only), and sampling from teachers trained with intrinsic rewards (Intrinsic-T). Performance is reported as the delta over Hard-Only. For reference, Hard-Only MATH pass@kk for k∈{1,4,8,16,32}k\in\{1,4,8,16,32\} is {0.5,1.7,3.2,5.7,9.6}\{0.5,1.7,3.2,5.7,9.6\}. Hard-Only training curves are shown in Figure [5](#S4.F5 "Figure 5 ‣ 4.5 Metrics ‣ 4 Experiment Setup ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability"); absolute performance for all methods, and further evaluations, are in Tables [4](#A3.T4 "Table 4 ‣ C.2 Full Evaluation on fail@128 MATH, HARP, and OlympiadBench. ‣ Appendix C Evaluations ‣ B.9 Computational resources ‣ B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability")-[5](#A3.T5 "Table 5 ‣ C.2 Full Evaluation on fail@128 MATH, HARP, and OlympiadBench. ‣ Appendix C Evaluations ‣ B.9 Computational resources ‣ B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability"). Shaded regions are ±\pm 1 SD over 6-12 seeds nested across teacher/student training (see [B.8](#A2.SS8 "B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability")).

![Refer to caption](/html/2601.18778/assets/x4.png)


Figure 4: Transfer performance to OlympiadBench fail@128 subset (improvement over Hard-Only). Questions optimized for MATH and HARP transfer to a held-out dataset. Performance is reported as the delta over Hard-Only; absolute performance, including PS evaluation, is in Table [6](#A3.T6 "Table 6 ‣ C.2 Full Evaluation on fail@128 MATH, HARP, and OlympiadBench. ‣ Appendix C Evaluations ‣ B.9 Computational resources ‣ B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability").

![Refer to caption](/html/2601.18778/assets/x5.png)


Figure 5: Grounded rewards lead to more stable teacher policies. We evaluate trained teacher policies by sampling questions and training fresh students. (Left) Test pass@32 comparison between students trained with questions sampled from Grounded-T and Base-T (Hard-Only also shown for reference). Grounded-T outperforms Base-T and exhibits more stable student trajectories. (Right) Pass@32 trajectories for fresh students trained with individual Grounded-T teacher seeds (red) and Intrinsic-T teacher seeds (green). Questions from Grounded-T yield consistent student trajectories, whereas Intrinsic-T exhibits higher variance across teachers, including a failure mode where I-T (1) causes student collapse. Shading shows ±1\pm 1 SD. Curves for other pass@k and OlympiadBench are in Figures [10](#A2.F10 "Figure 10 ‣ B.9 Computational resources ‣ B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability")-[12](#A2.F12 "Figure 12 ‣ B.9 Computational resources ‣ B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability").

![Refer to caption](/html/2601.18778/assets/x6.png)


Figure 6: Qualitative Evolution of Generated Questions. (Left) Baseline student performance during a SOAR run on HARP. The y-axis shows greedy accuracy on the fail@128 train set over promotion stages. (Right) Sampled teacher questions at different promotion points. Content and style shift from word problems and basic formulas (stage 1) to concise, equation-heavy problems in algebra and calculus (stage 2).
Many effective “stepping stones" include incorrect solutions, suggesting that structural and conceptual content provide sufficient learning signal.

## 5 Results

### 5.1 Meta-RL Discovers Effective Questions.

While curriculum learning is well-studied in RL, it is not obvious that synthetic questions can help a model move "beyond sharpening" its existing distributions.
Here, we show that self-generated stepping stones provide a learnable gradient that unlocks improvement in stalled regimes.
This occurs without the teacher seeing the target problems; instead, meta-RL sharpens the teacher’s policy, discovering useful curricula solely by optimizing for student progress.

PQ Kickstarts Learning on Hard Subsets. Both PS and PQ substantially outperform Hard-Only and Instrinsic baselines, with larger gains at higher kk. Figure [4](#S4.F4 "Figure 4 ‣ 4.5 Metrics ‣ 4 Experiment Setup ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability") shows improvement over Hard-Only. Hard-Only test trajectories are in Figures [5](#S4.F5 "Figure 5 ‣ 4.5 Metrics ‣ 4 Experiment Setup ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability"); all absolute numbers and trajectories are in Appendix [C.1](#A3.SS1 "C.1 Full Student Training curves ‣ Appendix C Evaluations ‣ B.9 Computational resources ‣ B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability")-[C.2](#A3.SS2 "C.2 Full Evaluation on fail@128 MATH, HARP, and OlympiadBench. ‣ Appendix C Evaluations ‣ B.9 Computational resources ‣ B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability"). Inference with the base model achieves non-zero pass@kk due to stochastic sampling with different seeds than were used for the initial fail@​128@128 filtering; nonetheless, Hard-Only training cannot sustain learning and plateaus.

Inference with PS achieves +8.5% pass@32 on fail@128-MATH over Hard-Only, and +3.6% pass@32
on fail@128-HARP. PQ achieves higher mean performance (+9.3% pass@32 on MATH, +4.2% on HARP), indicating that *the synthetic questions, rather than a fortunate student training trajectory, drive the performance gains.* Intrinsic-T underperforms both, validating that grounded rewards are needed to discover the right questions.

Synthetic questions do not just boost accuracy, but shift the student policy to make previously hard problems learnable. Student learning curves on MATH, where we use curriculum training, exhibit continued improvement after transitioning to fail@128 training (Figure [9](#A2.F9 "Figure 9 ‣ B.9 Computational resources ‣ B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability")). These effects significantly outstrip
what can be achieved from repeated sampling alone on fail@128 data; Hard-Only with a group size of 128 (4×\times extra compute) achieves only +2.8% pass@32 (Table [4](#A3.T4 "Table 4 ‣ C.2 Full Evaluation on fail@128 MATH, HARP, and OlympiadBench. ‣ Appendix C Evaluations ‣ B.9 Computational resources ‣ B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability")).

OOD generalization. Figure [4](#S4.F4 "Figure 4 ‣ 4.5 Metrics ‣ 4 Experiment Setup ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability")
shows that synthetic questions from PQ-MATH, PQ-HARP, and Intrinsic-T transfer to OlympiadBench, an OOD dataset (+6% and +3% respectively over Hard-Only). Cross-dataset transfer, despite no OOD optimization, suggests that synthetic curricula can capture generalizable reasoning pathways.

Oracle comparison to real curated data.
Our regime assumes that we only have access to hard problems, to study the case where additional expert-curated data is not available or not known. As a strong upper-bound, we compare to the “oracle" case where curated extra data is available. We train students on fail@128 + the full official MATH training set (6750 problems) as a representative pool of abundant, easier questions. We also compare to training with 128 random MATH/HARP questions in Appendix [C.2](#A3.SS2 "C.2 Full Evaluation on fail@128 MATH, HARP, and OlympiadBench. ‣ Appendix C Evaluations ‣ B.9 Computational resources ‣ B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability"), which performs similarly to training with the full dataset. Synthetic PQ-MATH questions recover 75% of the performance gains from full-MATH training, and PQ-HARP recover 50%. Notably, HARP-PQ (128/192128/192 questions) outperforms 128 real HARP questions, and matches 128 real MATH questions.

Direct inference on fail@128 test problems with the final trained teacher policy model does not improve over base model performance (Appendix [C.2](#A3.SS2 "C.2 Full Evaluation on fail@128 MATH, HARP, and OlympiadBench. ‣ Appendix C Evaluations ‣ B.9 Computational resources ‣ B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability")), indicating that generator and solver abilities are largely independent.

Takeaway:
A model’s pedagogical ability can be decoupled from its task-solving ability. Grounded meta-RL (SOAR) expands the “learnability frontier" by surfacing synthetic questions that enable improvement over reasoning plateaus.

### 5.2 Grounded rewards lead to stable and diverse teacher policies.

While the main utility of SOAR is in surfacing a set of teacher-generated questions that unlock student learning (PQ), we now shift focus to the trained teacher policies themselves.
In this section we perform a controlled study of teacher objectives to probe the effects of meta-RL, and show that grounded rewards (as in SOAR), versus intrinsic ones, yield stronger teacher policies. We evaluate teachers trained with grounded rewards (Grounded-T), intrinsic rewards (Intrinsic-T) and the base model (Base-T) by sampling question-answer pairs from these policies and training fresh students. In Appendix [C.3](#A3.SS3 "C.3 Sampling from Trained Teachers. ‣ Appendix C Evaluations ‣ B.9 Computational resources ‣ B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability") we also ablate grounded teachers trained without the student-promotion mechanism, to validate its necessity.

We evaluate four Grounded-T seeds per dataset to cover a range of final promotion stages, and three Intrinsic-T teacher seeds. We sample 128 questions from these teachers and train 2-3 fresh students on the synthetic questions and real fail@128 train set (≥9\geq 9 student runs per reported metric, see Appendix [B.6](#A2.SS6 "B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability")).

The teacher policy generates useful questions. Student test performance curves in Figure [5](#S4.F5 "Figure 5 ‣ 4.5 Metrics ‣ 4 Experiment Setup ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability") reveal that questions sampled from Grounded-T improve over Hard-Only. Results are competitive with PQ on MATH and HARP, validating that the useful pedagogical signal is not just captured in the set of evolved questions, but is also learned by the teacher policy. Further ablations show that sampling larger datasets from Grounded-T reduces the variance of student outcomes (Appendix [D.1](#A4.SS1 "D.1 Sampled dataset size ‣ Appendix D Ablations ‣ C.4 Correctness of Synthetic Questions ‣ Appendix C Evaluations ‣ B.9 Computational resources ‣ B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability")) and that the student-promotion mechanism improves the teacher policy (Appendix [C.3](#A3.SS3 "C.3 Sampling from Trained Teachers. ‣ Appendix C Evaluations ‣ B.9 Computational resources ‣ B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability")).

Meta-RL sharpens the question distribution. In Figure [5](#S4.F5 "Figure 5 ‣ 4.5 Metrics ‣ 4 Experiment Setup ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability") (left) we overlay student training curves for Grounded-T questions and Base-T questions. Grounded-T students consistently track the upper envelope of Base-T performance for MATH/HARP, with lower variance on MATH. The existence of successful runs from Base-T reveals the ability to generate useful stepping stone questions is latent in the model; meta-RL improves Grounded-T by sharpening the teacher to output questions that more reliably provide useful gradient signal.
This is yet another example of the sharpening mechanism of RL (yue2025does; zhao2025echo; tsilivis2025how; tsilivis2025howarxiv), but here leveraged for curricula. On OlympiadBench, where the target distribution differs substantially from the teacher’s training domain, Grounded-T and Base-T learning curves overlap more (though Grounded-T on HARP achieves highest peak performance), suggesting that meta-RL primarily sharpens in-domain pedagogical signals. This is consistent with PQ results in Figure [4](#S4.F4 "Figure 4 ‣ 4.5 Metrics ‣ 4 Experiment Setup ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability"), in which PQ-HARP outperforms Intrinsic-T whereas PQ-MATH matches it

Fragility of intrinsic proxies. Figure [5](#S4.F5 "Figure 5 ‣ 4.5 Metrics ‣ 4 Experiment Setup ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability") (right) compares aggregate student training curves for individual Grounded-T and Intrinsic-T teacher seeds. Students trained with questions from different Grounded-T seeds exhibit highly similar trajectories, indicating that grounded rewards lead to stable teacher policies. In contrast, Intrinsic-T teachers produce, on average, worse and more volatile outcomes. Across MATH, HARP, and OlympiadBench there is a clear separation in performance between students trained with different Intrinsic-T seeds. MATH and OlympiadBench student trajectories exhibit a consistent and significant ordering depending on the teacher. While some Intrinsic-T teachers produce highly effective curricula, the objective is subject to a high-variance failure mode: one out of three teacher seeds exhibits collapse across all datasets, yielding little or no progress on the target problems. This reinforces observations from the literature that RL with self-rewards is prone to reward hacking, or the decoupling of the intrinsic reward from actual task mastery (shafayat2025largereasoningmodelsselftrain; chae2025understandingselfplay).

Grounded Training Sustains Diversity. To probe how meta-RL shapes the teacher’s generative distribution, in Table [1](#S5.T1 "Table 1 ‣ 5.3 Question structure matters more than answer correctness. ‣ 5 Results ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability") we measure the semantic diversity of datasets from different teachers with the Vendi Score (V​SVS) (friedman2022vendi) using Qwen3-8B embeddings (zhang2025qwen3). Grounded-T (MATH) and Grounded-T (HARP) match the diversity of Base-T (V​S=34.91VS=34.91), with PQ showing only a small decline from the base model (V​S=31.75VS=31.75). In contrast, Intrinsic-T collapses into a narrow conceptual space (V​S=10.82VS=10.82), providing evidence of reward-hacking and a potential explanation for the observed “fragility". This suggests that grounded rewards successfully avoid the diversity collapse often seen in RL-loops (song2025outcomebasedexplorationllmreasoning), while intrinsic rewards fall prey to it. Indeed, we also observe a decline in the diversity of teacher completions during meta-RL with learnability rewards (Appendix [E](#A5 "Appendix E Teacher Training Dynamics ‣ Appendix D Ablations ‣ C.4 Correctness of Synthetic Questions ‣ Appendix C Evaluations ‣ B.9 Computational resources ‣ B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability")).

Takeaway:
Effective questions are latent in the base model, but hard to find. Grounding rewards in student progress "sharpens" the teacher’s noisy distribution of questions into a stable, diversity-preserving policy, whereas intrinsic rewards are prone to instability and diversity collapse.

### 5.3 Question structure matters more than answer correctness.

While conventional wisdom suggests that question-answer correctness is most important, our results suggest that the conceptual content and structure of questions is more important for models on learning plateaus.

Figure [6](#S4.F6 "Figure 6 ‣ 4.5 Metrics ‣ 4 Experiment Setup ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability") shows qualitative examples of PQ questions at different stages of a sample SOAR training trajectory, exhibiting shifts in style and conceptual focus as the baseline student improves. We annotate synthetic questions with Claude-4.5-Sonnet as an oracle judge, and observe that only 32.8% of PQ problems contain a fully correct solution, while 63% are considered mathematically well-posed (Appendix [C.4](#A3.SS4 "C.4 Correctness of Synthetic Questions ‣ Appendix C Evaluations ‣ B.9 Computational resources ‣ B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability")). This suggests that for models stalled on a performance plateau, structural and contextual cues of a question are more important for kickstarting learning than a correct answer. Indeed, Intrinsic-T questions have higher correctness (55%) but perform worse, likely because of lack of diversity (Section [5.2](#S5.SS2 "5.2 Grounded rewards lead to stable and diverse teacher policies. ‣ 5 Results ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability")).
Our experiments with Base-T, which, like Grounded-T and Intrinsic-T, is filtered for correctly formatted questions, show that question format alone is not behind these effects.
A more detailed taxonomy of synthetic questions, including error types, is in Appendix [C.4](#A3.SS4 "C.4 Correctness of Synthetic Questions ‣ Appendix C Evaluations ‣ B.9 Computational resources ‣ B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability"). Meta-RL decreases question ambiguity errors relative to Base-T, validating the importance of question coherence over answer correctness.

|  |  |  |
| --- | --- | --- |
| Method | Vendi Score (V​SVS) | Std. Dev (σ\sigma) |
| Base-T | 34.91 | 1.74 |
| Grounded-T (HARP) | 34.66 | 1.74 |
| Grounded-T (MATH) | 31.99 | 1.54 |
| PQ | 28.33 | 1.55 |
| Intrinsic-T | 10.82 | 1.01 |

Table 1: Semantic diversity analysis of synthetic datasets using Vendi Scores (V​SVS). All metrics are standardized to 128 questions via bootstrap subsampling (k=100k=100 iterations). V​SVS represents the effective number of unique semantic concepts. Our proposed teacher training (Grounded-T) successfully expands the conceptual manifold.

Takeaway:  For models at learning plateaus, problems that have conceptually diverse and coherent questions can provide useful gradient signal even without having precisely correct answers.

## 6 Discussion and Conclusions

Breaking the sparse-reward plateau in RL fine-tuning.
Our work establishes a way to kickstart RL fine-tuning when the initial
success rate is too low to collect RLVR signal. Generating
question-answer pairs (even if not correct) and training on those, with
the right meta-RL self-play loop, can be
enough to provide nonzero signal on the original hard problems.
Contrary to learnability approaches that rely on pure internal rewards, as is the case in prior LLM self-play approaches,
here the signal is ultimately grounded in measuring improvement on the
original problems. A central contribution of our work is that we show how to make this grounded bilevel meta-RL loop work in practice. The gap in performance shows the importance of this
point.

More importantly, our setup shows that generating stepping-stone questions to solve a problem does not require the preexisting ability to solve that problem, and that meta-RL sharpens this latent ability in the pretraining distribution. This intuition lies at the core of the self-play idea, although we show that it is crucial to go beyond pure curiosity
by grounding the process in actual performance.

Our results tie to the broader debate on whether RL fine-tuning truly expands a model’s learning frontier, or merely sharpens latent abilities (yue2025does; zhao2025echo; tsilivis2025how; tsilivis2025howarxiv). Our work indicates that meta-RL can expand the envelope of learnability beyond what direct RLVF can achieve. As a “North Star” thought experiment, consider a future model trained on the entire mathematical literature: a proof of a Millennium Problem such as the Riemann Hypothesis may already be latent in pretraining, yet successful learning would hinge on recovering the right sequence of intermediate lemmas and theorems that make the proof *learnable* to a student reasoner. In this view, just as RL is believed to sharpen or amplify useful subsets of pretraining data, meta-RL could retrieve the stepping-stone question–answer pairs embedded in the teacher’s vast training corpus. We believe our results provide concrete evidence that a moderate amount of grounded meta-RL can elicit such capabilities that remain inaccessible through repeated sampling alone.

Limitations. Our framework’s primary limitation is the computational cost of running bilevel RL loops (Appendix [B.9](#A2.SS9 "B.9 Computational resources ‣ B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability")). While inner loop training is relatively cheap (10-20 steps depending on the promotion stage) it necessitates training parallel students to compute stable teacher rewards. Importantly, our ablation in Table [4](#A3.T4 "Table 4 ‣ C.2 Full Evaluation on fail@128 MATH, HARP, and OlympiadBench. ‣ Appendix C Evaluations ‣ B.9 Computational resources ‣ B.8 Seeds ‣ B.7 Hyperparameters ‣ B.6 Evaluation ‣ B.5 Datasets ‣ B.4 Learnability Reward. ‣ B.3 Training Details ‣ B.2 Parsing Teacher Outputs ‣ Student Prompt. ‣ Teacher Prompt. ‣ B.1 Prompts ‣ Appendix B Method and Experiment Details ‣ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability") shows that reallocating compute to direct training on hard problems via repeated sampling does not recover the improvements achieved by the bilevel framework. Our work serves as a proof of concept for grounded rewards in this setting; investigating more efficient reward proxies or scaling beyond our 3B model experiments are rich avenues for further work.

## Acknowledgements

We thank Cansu Sancaktar and Phillip Isola for helpful discussions.
JK thanks the Simons Foundation for support through the Collaborative Grant “The Physics of Learning
and Neural Computation”.
This work was supported by an NSF GRFP fellowship to SS.

## References

## Appendix A Extended Related Work

### A.1 Curriculum Learning in RL

Automated curriculum design has a long history predating modern LLMs,
beginning with classical curriculum learning
(bengio2009curriculum; Graves2017automatedcurriculum). These methods
assume access to a labeled training set and focus on *reordering* or
*selecting* existing data rather than generating new tasks. In the
context of RL, curriculum learning helps agents acquire complex behaviors
by first mastering simpler tasks (Navekar2020curriculum; parashar2025curriculumreinforcementlearningeasy).
Contemporary LLM post-training inherits this paradigm: curriculum is
applied over curated prompts or problem categories, using proxy signals
such as gradient norms or advantage estimates to guide selection.
Examples include synthetic or self-training curricula like
Kimi (kimiteam2025kimi), FastCuRL (dennis2020paired), and
LightR1 (wen2025lightr1), as well as online difficulty-filtering
strategies such as Dapo (yu2025dapo), Online Difficulty
Filtering (bae2025onlinedifficultyfilteringreasoning), and
SEC (chen2025sec), which discretize problems into difficulty
buckets and score categories by gradient-derived proxies. While these
approaches improve learning efficiency in-distribution or OOD, they
presuppose that difficulty can be meaningfully partitioned *a
priori* and provide only indirect rewards for student progress. Adaptive Data Optimization (ADO) (jiang2025ADO) leverages per-domain scaling laws to estimate the learning potential of various data sources online jiang2025ADO.
By contrast, our goal is not to arrange data but to elicit
learning on a fixed, verifiable hard dataset where standard GRPO fails.

### A.2 Self-Play and Teacher-Student Setups

Self-play offers a complementary lens on autonomous capability growth, classically exemplified by game-playing agents trained without external data, such as AlphaZero (silver2018alphazero). Our approach is inspired by a line of research demonstrating that asymmetric self-play can induce powerful automatic curricula. In early work, sukhbaatar2017asymmetric introduced the canonical Alice–Bob framework in which one agent (Alice) proposes tasks while another (Bob) attempts to solve them, yielding a natural progression of “just-hard-enough” challenges that drive learning. This idea was later extended to complex embodied domains in robotics, where asymmetric self-play enabled automatic discovery of diverse manipulation goals without manual task specification (openai2021asymmetricselfplay).
Applying these ideas from robotics and control to large language models introduces fundamentally different challenges: LLMs operate over a discrete, symbolic problem space with no environment simulator to evaluate intermediate progress; a teacher must generate *entire tasks*, often requiring multi-step reasoning. Moreover, rewards in language domains are extremely sparse and brittle—for mathematical problems, correctness is essentially binary and offers no gradient toward partial solutions.
Modern LLM self-play methods thus differ in mechanism: SPIN (chen2024spin), Triplet self-play (wang2025stablellmselfplay), and ReSTEM{}^{\text{EM}} (singh2024beyond) optimize for self-consistency or solution quality. These methods generate responses and still presuppose the existence
of well-formed input prompts or curated high-quality questions. Recent systems like AlphaProof (AlphaProofNature2025) attempt to mitigate this sparsity at test-time by using an LLM to generate a "natural curriculum" of auxiliary theorem variations for additional training (AlphaProofNature2025). In the context of RLHF, eva (ye2024eva) casts RLHF as an asymmetric creator–solver game in which a creator evolves prompts to expose alignment weaknesses and a solver adapts to reward-model feedback.
A series of near-contemporary works leverages pre-trained LLMs themselves as an untapped resource for question generation.
Such "fully data-free" co-evolving systems—including Absolute Zero
(zhao2025absolute), R-Zero (huang2025rzero), Language
Self-Play (LSP) (kuba2025languageselfplay), SeRL
(fang2025serl) and Self-Questioning Language Models (SQLM)
(chen2025selfquestioning)—jointly evolve task creators and solvers
via intrinsic or proxy rewards such as majority vote, learnability,
reward-model preferences, or gradient magnitudes. Because these methods
optimize intrinsic or proxy objectives, they risk drifting to degenerate
or unlearnable tasks, are sensitive to reward hacking where models learn to maximize training
(pseudo-)reward, and lack guarantees
of progress (see an analysis of AbsoluteZero in
chae2025understandingselfplay). This connects directly to a line of works investigating the broader question of whether self-training — the process where a model
learns from its own judgments — can be sustained within RL, and how far self-improvement can be driven by intrinsic or self-generated rewards.
Prolonged RL with self-rewards often results in sudden and complete performance collapse (shafayat2025largereasoningmodelsselftrain; chae2025understandingselfplay), when rewards vanish or when generator and solver objectives misalign, especially in discrete, symbolic domains with essentially binary correctness signals.
This fragility mirrors earlier
findings in unsupervised curriculum generation
(dennis2020paired; racaniere2020settersolver; jiang2021ued). These
observations motivate our design: we learn a teacher *policy* via
meta-RL that generates verifiable math questions directly optimized for
student learning progress, grounding the curriculum in a concrete failure
regime instead of internal proxy of difficulty.

### A.3 Intrinsic Rewards versus Bilevel Optimization

To our knowledge, essentially all recent “fully data-free” self-play approaches use
intrinsic or proxy rewards to train the teacher/proposer, without
anchoring to “real” student performance (with the exception of the self-adaptation work by zweiger2025selfadapting which uses ReSTEM{}^{\text{EM}}/SFT for outer/inner loop).
Examples of intrinsic rewards include model confidence as proposed in Inuitor (zhao2025learningreasonexternalrewards) or RENT (prabhudesai2025maximizingconfidenceimprovesreasoning) or the majority answer as in TTRL (zuo2025ttrl) or shafayat2025largereasoningmodelsselftrain, as well as in SQLM (chen2025selfquestioning).
Of course, the use of proxy rewards is often not merely a design
preference but a pragmatic simplification, especially in teacher-student
self-play setups: it avoids facing an explicit inner-loop–outer-loop bilevel optimization problem - an appealing but challenging objective where the output of one optimization (in this instance the optimization of the student trained with RLVF on the teacher’s question-answer pairs) is fed into another optimization loop (the performance improvement of the student on the hard dataset).
Such bilevel optimization objectives have strong historical precedence in
meta-learning, in popular methods such as MaML (Finn17maml) and
Reptile (nichol2018firstordermetalearningalgorithms), which
explicitly train through an inner-loop–outer-loop structure to obtain
efficient few-shot learners,
following earlier research like RL2 (duan2016rl2fastreinforcementlearning),
and works that meta-learn hyperparameters
of neural nets via full backpropagation through the training loop
(maclaurin2015hyperopt). A similar bilevel formulation, which
served as inspiration for our work, also appears in dataset distillation
(wang2018dataset), where an outer loop optimizes a generally small
dataset that allows an inner training loop to achieve good target
performance. Here, both proxy-based (e.g., NTK approximation
(nguyen2021kipimprovedresults) or feature-matching
(zhou2022dataset)) and end-to-end bilevel formulations have been
explored (wang2018dataset; deng2022remember; feng2024embarrassingly). In general, such approaches become intractable, as the inner loop involves a multi-step computation
with a large number of steps, which requires backpropagation through time
(BPTT), or in fact “backpropagation through gradient descent”,
unrolling the inner loop and taking meta-gradients. Our approach,
however, avoids the need to unroll the inner loop thanks to the use of
RLOO in the outer loop, using the reward (the performance improvement of
the student) to reinforce question-answer sets. This is the first
instance of “double meta-RL loop” we are aware of in the context of self-play for LLMs.

## Appendix B Method and Experiment Details

### B.1 Prompts

#### Teacher Prompt.

At every outer-loop step, the teacher is given the same prompt. The prompt guides the model towards producing valid math problems using sample subjects/domains and provides explicit instruction regarding the expected format. We avoid seeding the teacher with sample math questions to preserve the data-free setup; the model only sees the black-box reward signal of student performance. We also observe in initial experiments that, when given seed questions, the teacher often collapses to copying them.

`Teacher Prompt

#### Student Prompt.

The same prompt is used for fail@128 filtering, training the student in the inner-loop, and training the student in evaluation.

Student Prompt

### B.2 Parsing Teacher Outputs

To parse the teacher
rollouts into question-answer pairs, we require teacher responses to
follow the prompt-specified format. We filter out generations that do not
follow this format, and resample until we have g⋅ng\cdot n
correctly-formatted problems. We filter for the following:

* •

  Contains opening and closing question/answer tags.
* •

  Contains the “boxed" notation (denoting an answer).
* •

  Contents of the boxed answer are parsable by a symbolic math verifier.

Theoretically, rejection sampling does not
affect the RLOO gradient update (Proposition 1); empirically, we find that this performs
better than using teacher-format rewards or sequential question/answer
sampling.

###### Proposition 1 (RLOO update with rejection sampling).

Let π0​(z)\pi_{0}(z) be a proposal distribution over some random variable zz.
Let SS be a set of “accepted” values of zz, and assume π0​(S)>0\pi_{0}(S)>0.
Let

|  |  |  |  |
| --- | --- | --- | --- |
|  | π​(z)=π0​(z)​1z∈S/π0​(S)\pi(z)=\pi_{0}(z)1_{z\in S}/\pi_{0}(S) |  | (2) |

be the distribution on zz obtained by rejection sampling, namely,
sampling zz from π0\pi_{0} until z∈Sz\in S.

Let R​(z)R(z) be some reward function on zz. Then the RLOO update on π\pi
can be computed from gradient of π0\pi_{0} only. Namely, for any gg-tuple
z1,…,zgz_{1},\ldots,z_{g} sampled from π\pi, one has

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑i=1gA​(zi)​∇ln⁡π​(zi)=∑i=1gA​(zi)​∇ln⁡π0​(zi)\sum_{i=1}^{g}A(z_{i})\nabla\ln\pi(z_{i})=\sum_{i=1}^{g}A(z_{i})\nabla\ln\pi_{0}(z_{i}) |  | (3) |

where

|  |  |  |  |
| --- | --- | --- | --- |
|  | A​(zi)=R​(zi)−1g−1​∑j≠iR​(zj)A(z_{i})=R(z_{i})-\frac{1}{g-1}\sum_{j\neq i}R(z_{j}) |  | (4) |

is the RLOO advantage function, and where the gradients are with respect
to the parameters of π\pi.

This is not true for simple Reinforce: it relies on the fact that RLOO
advantages A​(zi)A(z_{i}) sum to 0 over ii.

###### Proof.

For any zz sampled from π\pi, one has z∈Sz\in S with probability 11.
For z∈Sz\in S,
one has ln⁡π​(z)=ln⁡π0​(z)−ln⁡π0​(S)\ln\pi(z)=\ln\pi_{0}(z)-\ln\pi_{0}(S). Therefore,

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∑i=1gA​(zi)​∇ln⁡π​(zi)\displaystyle\sum_{i=1}^{g}A(z_{i})\nabla\ln\pi(z_{i}) | =∑i=1gA​(zi)​(∇ln⁡π0​(zi)−∇ln⁡π0​(S))\displaystyle=\sum_{i=1}^{g}A(z_{i})\left(\nabla\ln\pi_{0}(z_{i})-\nabla\ln\pi_{0}(S)\right) |  | (5) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =∑i=1gA​(zi)​∇ln⁡π0​(zi)−(∑i=1gA​(zi))​∇ln⁡π0​(S)\displaystyle=\sum_{i=1}^{g}A(z_{i})\nabla\ln\pi_{0}(z_{i})-\left(\sum_{i=1}^{g}A(z_{i})\right)\nabla\ln\pi_{0}(S) |  | (6) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =∑i=1gA​(zi)​∇ln⁡π0​(zi)\displaystyle=\sum_{i=1}^{g}A(z_{i})\nabla\ln\pi_{0}(z_{i}) |  | (7) |

since the sum of advantages in RLOO satisfies ∑iA​(zi)=0\sum_{i}A(z_{i})=0.
∎

### B.3 Training Details

Algorithm 1 details our full algorithm.

Input: Initial teacher πϕT\pi^{T}_{\phi}, initial student πθS\pi^{S}_{\theta}, threshold τ\tau, group size gg, dataset size nn, repeats rr

Initialize timestep t←0t\leftarrow 0, EMA reward R¯0←0\bar{R}_{0}\leftarrow 0, 𝒟best←∅\mathcal{D}_{\text{best}}\leftarrow\emptyset

while t<Tt<T do

0.3em

// 1. Teacher generation

Sample g⋅ng\cdot n QA pairs: {(qi,ai)}i=1g⋅n∼πϕT\{(q_{i},a_{i})\}_{i=1}^{g\cdot n}\sim\pi^{T}_{\phi}

Partition into gg datasets: 𝒳k={(qj,aj)}j=n​(k−1)+1n​k\mathcal{X}_{k}=\{(q_{j},a_{j})\}_{j=n(k-1)+1}^{nk} for k=1,…,gk=1,\dots,g

Sample reward questions 𝒬R={(qj,aj)}j=1M∼𝒟train\mathcal{Q}_{R}=\{(q_{j},a_{j})\}_{j=1}^{M}\sim\mathcal{D}_{\text{train}}

// 2. Inner Loop

for k=1k=1 to gg do

for j=1j=1 to rr do

θk,j′←RLOO-Update​(θ,𝒳k)\theta^{\prime}_{k,j}\leftarrow\textsc{RLOO-Update}(\theta,\mathcal{X}_{k}) {Student RL}

Rk,j←Acc​(θk,j′,𝒬R)−Acc​(θ,𝒬R)R_{k,j}\leftarrow\textsc{Acc}(\theta^{\prime}_{k,j},\mathcal{Q}_{R})-\textsc{Acc}(\theta,\mathcal{Q}_{R})

end for

Rk←1r​∑j=1rRk,jR_{k}\leftarrow\frac{1}{r}\sum_{j=1}^{r}R_{k,j}

end for

// 3. Check for student promotion.

Update R¯t←EMA​(R¯t−1,1g​∑k=1gRk)\bar{R}_{t}\leftarrow\textsc{EMA}(\bar{R}_{t-1},\frac{1}{g}\sum_{k=1}^{g}R_{k})

if R¯t>τ\bar{R}_{t}>\tau then

k∗←arg⁡maxk⁡Rkk^{*}\leftarrow\arg\max_{k}R_{k}

Find j∗j^{*} such that Rk∗,j∗R_{k^{*},j^{*}} is the median reward in {Rk∗,j}j=1r\{R_{k^{*},j}\}_{j=1}^{r}

θ←θk∗,j∗′\theta\leftarrow\theta^{\prime}_{k^{*},j^{*}} {Student Promotion}

𝒟best←𝒟best∪𝒳k∗\mathcal{D}_{\text{best}}\leftarrow\mathcal{D}_{\text{best}}\cup\mathcal{X}_{k^{*}}

end if

// 4. Teacher Policy Update (Outer-loop)

ϕ←RLOO-Update​(ϕ,{(𝒳k,Rk)}k=1g)\phi\leftarrow\textsc{RLOO-Update}(\phi,\{(\mathcal{X}_{k},R_{k})\}_{k=1}^{g}) {Teacher RL}

t←t+1t\leftarrow t+1

end while

return 𝒟best\mathcal{D}_{\text{best}}, πθS\pi^{S}_{\theta}

Algorithm 1 SOAR: Teacher-Student meta-RL Training

Stabilizing teacher rewards. Training inner-loop students with RL can potentially lead to noisy trajectories, and thus noisy teacher rewards. To stabilize the teacher rewards, for each sampled dataset 𝒳k{\mathcal{X}}_{k} we execute rr parallel student trainings and evaluations, and average their rewards to obtain the final reward: Rk=1r​∑j=1rRk,jR_{k}=\frac{1}{r}\sum_{j=1}^{r}R_{k,j}. In practice, we use r=4r=4.

Promotion mechanism. At each outer-loop timestep we train rr students on each dataset 𝒳k{\mathcal{X}}_{k}, and “promote" the student baseline when the moving average of teacher rewards exceeds a fixed threshold τ\tau.
We choose which trained student to promote by selecting the dataset 𝒳k{\mathcal{X}}_{k} with the highest reward R​(𝒳k)R({\mathcal{X}}_{k}) and then selecting the student with the median reward amongst those trained on 𝒳k{\mathcal{X}}_{k}.

Computing student rewards. For inner-loop and evaluation RL on the student, we use the Math-Verify package to compare the student-generated and ground-truth answers (kydlicek2025mathverify). We assign a reward following standard formulations for RLVR with math:

R​(y,a)R(y,a) =
{120.0if has_boxed​(y)∧verify​(y,a)20.0if has_boxed​(y)∧¬verify​(…)∧a∈ya​n​s10.0if has_boxed​(y)∧¬verify​(…)∧a∉ya​n​s0.0otherwise\begin{cases}120.0&\text{if }\text{has\_boxed}(y)\land\text{verify}(y,a)\\
20.0&\text{if }\text{has\_boxed}(y)\land\neg\text{verify}(\dots)\land a\in y_{ans}\\
10.0&\text{if }\text{has\_boxed}(y)\land\neg\text{verify}(\dots)\land a\notin y_{ans}\\
0.0&\text{otherwise}\end{cases}

### B.4 Learnability Reward.

To ablate the effects of our grounded reward versus intrinsic rewards, we train teacher models using the well-studied learnability reward (zhao2025absolute; sukhbaatar2017asymmetric). We use the same candidate-generation and dataset-partitioning procedure as SOAR. For each candidate dataset 𝒳k={qi,ai)}i=1n\mathcal{X}_{k}=\{q_{i},a_{i})\}_{i=1}^{n}, we sample 32 completions from the student for each qiq_{i} and compute the average success rate s¯i\bar{s}_{i}. The per-question reward is then computed as

|  |  |  |  |
| --- | --- | --- | --- |
|  | ri={0,if ​s¯i=01−s¯i,otherwise.r_{i}=\begin{cases}0,&\text{if }\bar{s}_{i}=0\\ 1-\bar{s}_{i},&\text{otherwise.}\end{cases} |  | (8) |

We then compute the dataset-level reward as Rk=1n​∑i=1nriR_{k}=\frac{1}{n}\sum_{i=1}^{n}r_{i}. For consistency with SOAR, every rollout in 𝒳k\mathcal{X}_{k} receives the averaged dataset-level reward. We train learnability teachers for 200 steps, and observe convergence of rewards.

### B.5 Datasets

Fail@128 Filtering. For each problem in the pool of candidates, we sample 128 solutions with Llama-3.2-3B-Instruct using the student prompt in Appendix B.1, a token budget of 1024 tokens, and temperature 1.0. We keep problems that obtained a 0/128 success rate.

OlympiadBench. For OlympiadBench, we source our fail@128 questions from the subset that is in English, text-only, and automatically verifiable (674 total questions). Since OlympiadBench was originally designed as a test set, we construct a random train/test split.

HARP. We source our fail@128 problems from the full HARP dataset. Since HARP was originally designed as a test set, we construct a random train/test split.

MATH. In preliminary experiments, we observed a large gap between the zero-shot accuracy of Llama-3.2-3B-Instruct on the official MATH training vs. test splits (60% vs. 37%), suggesting that the model may have partial exposure to the MATH training questions. To minimize confounding effects from such memorization, we draw our initial pool of hard problems from the 5000-problem official MATH test split. We then apply the fail@128 filter and construct our own internal train/test split from this filtered subset. All synthetic data generation and student-teacher training uses only the internal training split, and final results are reported exclusively on the held-out internal test split.

Dataset sizes. In Table 2 we report the original size of each problem pool, and the sizes of our train/test splits.

Table 2: Dataset sizes pre- and post- fail@128 filtering.

|  |  |  |  |
| --- | --- | --- | --- |
| Dataset | Initial problem pool | fail@128 train set | fail@128 test set |
| MATH | 5000 | 359 | 360 |
| HARP | 4768 | 714 | 714 |
| Olympiad Bench | 674 | 158 | 158 |

### B.6 Evaluation

Mixed synthetic-real training. We primarily evaluate generated questions by training a fresh student model on a combination of the synthetic questions, and the real fail@128 train set. We explore two mixing strategies:

* •

  Curriculum training. We first train the student on synthetic questions for a fixed number of training steps (64), and then switch to training on real fail@128 training questions, aiming to mirror the trajectory of training a promoted student. Here, the synthetic questions act as a “warm-start", enabling the student to obtain gradient signal on the harder problems. The synthetic training window was chosen as a representative budget based on preliminary experiments.
* •

  Mixed training.  We train on a mixture of synthetic and real questions throughout.

To avoid biasing results, we select between curriculum/mixed training using our baseline methods.

On MATH, while both exhibit similar training dynamics, we found that our Base-T baseline performed better with curriculum and thus adopt it for all MATH experiments (Figure 7). On OlympiadBench and HARP we observed that mixed training yields significantly more stable learning dynamics, even when adding real instead of synthetic data.
Figure 8 compares mixed/curriculum training on HARP and OlympiadBench fail@128 with 128 real MATH problems. Curriculum training exhibits an early performance spike, followed by a significant and sudden performance decline early in training. Thus for HARP and OlympiadBench we use mixed training in our evaluations.

![Refer to caption](/html/2601.18778/assets/x7.png)


Figure 7: Mixed v. Curriculum training on MATH. We compare training the base student on fail@128 + 128 questions sampled from Base-T, for performance on MATH. Curriculum performs better across different inference budgets.

![Refer to caption](/html/2601.18778/assets/x8.png)


Figure 8: Mixed v. Curriculum training on HARP/OlympiadBench. We compare training the base student on real fail@128 + 128 random MATH questions, for HARP and OlympiadBench. Mixed training exhibits significantly more stable training dynamics across inference budgets (Pass@8 and Pass@32) and converges to higher final performance points. For both datasets, curriculum training exhibits strong instability with a large early performance spike and then crash.

Teacher sampling. At evaluation time, we sample problems from the trained teacher using the same prompt and format-filtering as in training.

PQ/PS Evaluation. We evaluate PQ using mixed synthetic/real training, described above. We evaluate PS by simply running inference on the fail@128 test set, to evaluate how much the student baseline advanced during SOAR training.

Student checkpoint selection. For evaluations involving fresh student models, we train for a maximum of 1500 steps (observing convergence well before this point).
For MATH and HARP experiments where we report performance at a fixed point, we select the student checkpoint to evaluate at using the slope of the smoothed training reward curve, similarly to classic RL early stopping heuristics (Mahsereci2017EarlySW). In particular, we smooth the average training reward curve (centered-moving-average, 25 steps) and compute the discrete slopes, normalized by the range of observed rewards. The early stopping step is defined as the earliest point where the normalized slope falls below 15% of the maximum observed slope.
We selected a 15% threshold to identify the beginning of the reward plateau; empirically, varying between 10% and 20% have negligible effects on the selected point.
Test performance is averaged over a 200 step window following the selected step, to account for variance. In Figure C.2 we show the full training curves.

We choose this heuristic to account for differing convergence rates between methods on MATH and HARP, and our small dataset sizes. In initial experiments we found separate validation sets, and cross-validation with the train set, to be extremely noisy. On OlympiadBench we observe similar convergence across all methods, and report at a fixed point of 50 steps.

### B.7 Hyperparameters

In Table 3 we detail our training and evaluation hyperparameters.

Outer-loop training. We performed the following sweeps in preliminary experiments, and tuned using student performance on the full train set. Once selected, the same hyperparameters are used across all training runs and datasets. See Appendix D.2 for ablations on sensitivity to threshold τ\tau and dataset size nn.

* •

  LR: {1e-6, 5e-6, 1e-5, 5e-5}
* •

  nn: {8, 16, 32, 64}
* •

  τ\tau: {0.01, 0.015, 0.02}
* •

  Moving avg window size: {1, 3}

We train for a maximum of 200 outer steps based on compute constraints. For teacher-sampling experiments we fix the evaluation checkpoint based on the point of decline of teacher rewards observed in initial runs (170 steps for all HARP-trained models, 200 steps for all MATH-trained models).

Inner-loop training. We find that from the base student, 10 steps is sufficient to induce movement in student performance. As the student baseline is updated, it is helpful to train slightly longer (we use +5 steps).
We use greedy decoding for evaluating on 𝒬R\mathcal{Q}_{R} to reduce noise in the student reward.

Evaluation. We use standard hyperparameters to train the student from scratch on combined real/synthetic data (Table 3c). For PQ with curriculum evaluation we use zero learning rate warmup
to match the inner-loop environment.

### B.8 Seeds

To ensure statistical significance and account for both teacher-training and student-training variation, we employ a nested seeding strategy.

Teacher training.

* •

  For our main SOAR experiments, we train four independent teachers each on MATH and HARP to cover a range of teacher training outcomes.
* •

  For teacher objective ablations (Intrinsic-T and Grounded-T (no promotion)) we trained three independent teachers each.

Evaluation (student training).

* •

  The Hard-Only baseline is evaluated over ≥6\geq 6 student seeds.
* •

  For PQ datasets (>>2 promotions), we train at least three students per PQ dataset, totaling ≥6\geq 6 seeds (2 PQ datasets ×\times 3 students) per reported metric.
* •

  For PS students, we compute pass@kk metrics using inference over three seeds.
* •

  For teacher-sampling experiments (i.e., sampling data from trained teachers and then training a fresh student) we train 2-3 independent students per teacher seed, resulting in ≥8\geq 8 seeds per reported metric.

For all metrics we report the aggregated mean and standard deviation over student seeds.

|  |  |  |
| --- | --- | --- |
| Hyperparameter | Teacher | Student |
| Optimizer | AdamW | |
| KL coefficient | 0.001 | |
| LR schedule | Cosine decay | |
| Learning rate | 1e-5 | |
| Temperature | 1.0 | |
| LR warmup steps | 20 | 0/20 |
| Batch size | 2 | 8 |
| Group size | 4 | 32 |
| Max generated tokens | 512 | 1024 |
| meta-RL specific (teacher only) | | |
| Promotion threshold (τ\tau) | 0.01 | — |
| Moving avg window | 3 | — |
| Dataset size (nn) | 64 | — |
| Student repeats (rr) | 4 | — |
| Evaluation specific (student only) | | |
| Max training steps | — | 1500 |
| Synthetic warmup steps | — | 64 |
| (curriculum training) |  |  |

Table 3: Hyperparameters for SOAR training and evaluation.

### B.9 Computational resources

Each SOAR training run was executed on 4 nodes (each 8×\times NVIDIA H200 GPUs or 8×\times NVIDIA H100 GPUs) for ≈\approx 48-60 hours. Each RLOO evaluation run (training a fresh student) was executed for ≈\approx 12 hours on 1 H200 node or 1 H100 node.

![Refer to caption](/html/2601.18778/assets/x9.png)


Figure 9: Fail@128 test performance during student training for MATH, HARP, and Olympiad. Student learning curves at different pass@k when trained on Hard-Only, PQ, or the Full MATH dataset (PS inference performance shown as a horizontal line). PQ and PS improve performance on all inference budgets and datasets, with increased effect at higher kk. On MATH, PQ exhibits performance gains even after the synthetic-training phase (64 steps), showing that synthetic problems make real hard problems more learnable.

![Refer to caption](/html/2601.18778/assets/x10.png)


Figure 10: Fail@128 test performance during student training for MATH with different teachers. Each column compares training a fresh student with 128 questions from Grounded-T to 128 questions from a different teacher (Hard-Only also included for reference). While all teachers outperform Hard-Only, Grounded-T performs best, with increasing effects at higher kk. Grounded-T results in less variance across student outcomes, particularly compared to Base-T and Intrinsic-T. PQ learning curves are in Figure 9.

![Refer to caption](/html/2601.18778/assets/x11.png)


Figure 11: Fail@128 test performance during student training for HARP with different teachers. Each column compares training a fresh student with 128 questions from Grounded-T to 128 questions from a different teacher (Hard-Only also included for reference). Grounded-T performs best, with increasing effects at higher kk. Students trained with Base-T and Intrinsic-T tend to decline more for higher kk in the later stages of training, while Grounded-T leads to more stable trajectories.

![Refer to caption](/html/2601.18778/assets/x12.png)


Figure 12: Fail@128 test performance during student training for Olympiad with different teachers. Each column compares training a fresh student with 128 questions from Grounded-T (trained with MATH and HARP) to 128 questions from a different teacher (Hard-Only also included for reference). Students trained with Grounded-T teachers have more similar mean performance to Base-T and Intrinsic-T than seen on HARP and MATH (Figures 10-11). However, Grounded-T (HARP) shows more stability and less variance between independent teachers than Intrinsic-T (see Figure 13).

![Refer to caption](/html/2601.18778/assets/x13.png)


Figure 13: Test Pass@32 on OlympiadBench for fresh students trained with individual Grounded-T teacher seeds (red) and Intrinsic-T teacher seeds (green). Questions from Grounded-T yield consistent student trajectories on OlympiadBench across different teachers, whereas Intrinsic-T exhibits high variance across teachers, including a failure mode where I-T (1) causes student collapse.

## Appendix C Evaluations

### C.1 Full Student Training curves

In Figure 9 we show full student training curves for PQ, Hard-Only, and the full MATH upper bound for MATH, HARP, and OlympiadBench. In Figures 10-12 we show these training curves for questions sampled from Grounded-T, Base-T, Intrinsic-T, and Grounded-T (no promotion). All curves show the mean and standard deviation over seeds.

### C.2 Full Evaluation on fail@128 MATH, HARP, and OlympiadBench.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | k | | | | |
| Method | 1 | 4 | 8 | 16 | 32 |
| Base Model Inference | 0.3±0.10.3\pm 0.1 | 1.0±0.21.0\pm 0.2 | 2.0±0.42.0\pm 0.4 | 3.9±0.83.9\pm 0.8 | 7.5±1.37.5\pm 1.3 |
| Hard-Only | 0.5±0.10.5\pm 0.1 | 1.7±0.41.7\pm 0.4 | 3.2±0.83.2\pm 0.8 | 5.7±1.55.7\pm 1.5 | 9.6±2.69.6\pm 2.6 |
| Hard-Only (g=128g=128) | 1.4±1.01.4\pm 1.0 | 3.9±2.63.9\pm 2.6 | 6.1±3.96.1\pm 3.9 | 8.9±5.58.9\pm 5.5 | 12.4±7.412.4\pm 7.4 |
| SOAR-PQ (Ours) | 1.7±1.0\mathbf{1.7\pm 1.0} | 5.3±2.6\mathbf{5.3\pm 2.6} | 8.5±3.7\mathbf{8.5\pm 3.7} | 13.0±4.813.0\pm 4.8 | 18.9±5.318.9\pm 5.3 |
| SOAR-PS (Ours) | 1.0±0.21.0\pm 0.2 | 3.8±0.63.8\pm 0.6 | 6.8±1.16.8\pm 1.1 | 11.5±1.611.5\pm 1.6 | 18.1±2.418.1\pm 2.4 |
| Grounded-T (Ours) | 1.6±0.51.6\pm 0.5 | 5.1±1.45.1\pm 1.4 | 8.4±2.18.4\pm 2.1 | 13.1±2.9\mathbf{13.1\pm 2.9} | 19.1±3.7\mathbf{19.1\pm 3.7} |
| Intrinsic-T | 1.0±0.61.0\pm 0.6 | 3.3±2.13.3\pm 2.1 | 5.7±3.55.7\pm 3.5 | 9.2±5.39.2\pm 5.3 | 14.1±7.514.1\pm 7.5 |
| HARP train (128) | 2.4±1.02.4\pm 1.0 | 7.2±2.47.2\pm 2.4 | 11.3±3.111.3\pm 3.1 | 16.5±3.616.5\pm 3.6 | 23.0±3.923.0\pm 3.9 |
| MATH train (128) | 2.1±0.02.1\pm 0.0 | 6.6±0.16.6\pm 0.1 | 10.5±0.310.5\pm 0.3 | 15.7±0.515.7\pm 0.5 | 21.8±0.921.8\pm 0.9 |
| MATH train (Full) | 2.7±0.22.7\pm 0.2 | 7.6±0.77.6\pm 0.7 | 11.5±1.211.5\pm 1.2 | 16.4±1.816.4\pm 1.8 | 22.0±2.422.0\pm 2.4 |

Table 4: MATH Pass@k (%) Test Accuracy on Fail@128. Mean and SD over seeds are averaged over a 200 step window determined by training reward convergence (see Appendix B.6) with full curves in Figure 9. PQ and PS consistently outperform inference-only, Hard-Only, and intrinsic baselines across all inference budgets, and recover the majority of performance gain from training with real curated problems.
We boldface the best among “data-free" methods (i.e., only 𝒟t​r​a​i​n\mathcal{D}_{train} available). The bottom three rows serve as upper bounds from using curated, expert-annotated data. PQ datasets contain one of {128,192,256}\{128,192,256\} questions.



|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | k | | | | |
| Method | 1 | 4 | 8 | 16 | 32 |
| Base Model Inference | 0.2±0.00.2\pm 0.0 | 0.9±0.00.9\pm 0.0 | 1.7±0.01.7\pm 0.0 | 3.4±0.03.4\pm 0.0 | 6.4±0.06.4\pm 0.0 |
| Hard-Only | 0.4±0.10.4\pm 0.1 | 1.4±0.21.4\pm 0.2 | 2.6±0.42.6\pm 0.4 | 4.7±0.64.7\pm 0.6 | 8.2±1.08.2\pm 1.0 |
| SOAR-PQ (Ours) | 0.7±0.3\mathbf{0.7\pm 0.3} | 2.5±0.8\mathbf{2.5\pm 0.8} | 4.5±1.3\mathbf{4.5\pm 1.3} | 7.7±1.7\mathbf{7.7\pm 1.7} | 12.3±2.0\mathbf{12.3\pm 2.0} |
| SOAR-PS (Ours) | 0.6±0.10.6\pm 0.1 | 2.1±0.32.1\pm 0.3 | 3.9±0.63.9\pm 0.6 | 7.0±0.97.0\pm 0.9 | 11.8±1.211.8\pm 1.2 |
| Grounded-T (Ours) | 0.5±0.20.5\pm 0.2 | 2.0±0.52.0\pm 0.5 | 3.8±0.93.8\pm 0.9 | 6.7±1.36.7\pm 1.3 | 11.2±1.711.2\pm 1.7 |
| Intrinsic-T | 0.4±0.10.4\pm 0.1 | 1.6±0.51.6\pm 0.5 | 3.1±0.83.1\pm 0.8 | 5.6±1.45.6\pm 1.4 | 9.6±2.19.6\pm 2.1 |
| HARP train (128) | 0.4±0.00.4\pm 0.0 | 1.4±0.11.4\pm 0.1 | 2.8±0.22.8\pm 0.2 | 5.0±0.55.0\pm 0.5 | 8.7±1.18.7\pm 1.1 |
| MATH train (128) | 0.6±0.10.6\pm 0.1 | 2.1±0.42.1\pm 0.4 | 4.0±0.74.0\pm 0.7 | 7.1±0.97.1\pm 0.9 | 11.9±0.911.9\pm 0.9 |
| MATH train (Full) | 1.7±0.21.7\pm 0.2 | 5.1±0.45.1\pm 0.4 | 8.1±0.48.1\pm 0.4 | 11.7±0.311.7\pm 0.3 | 16.2±0.416.2\pm 0.4 |

Table 5: HARP Pass@k (%) Test Accuracy on fail@128. Mean and SD over seeds are reported at the timestep determined by training reward convergence (see Appendix B.6) with full curves in Figure 9. PQ and PS consistently outperform inference-only, Hard-Only, and intrinsic baselines across all inference budgets. Notably, SOAR questions perform better on HARP than similar numbers of questions from the MATH/HARP datasets (which serve as a curated, expert-annotated data source).



|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | k | | | | |
| Method | 1 | 4 | 8 | 16 | 32 |
| Base Model Inference | 0.2±0.00.2\pm 0.0 | 0.8±0.10.8\pm 0.1 | 1.6±0.31.6\pm 0.3 | 3.1±0.53.1\pm 0.5 | 5.8±1.05.8\pm 1.0 |
| Hard-Only | 0.3±0.10.3\pm 0.1 | 1.1±0.31.1\pm 0.3 | 2.1±0.62.1\pm 0.6 | 3.9±1.33.9\pm 1.3 | 6.9±2.76.9\pm 2.7 |
| SOAR-PQ (MATH) (Ours) | 0.5±0.10.5\pm 0.1 | 1.9±0.51.9\pm 0.5 | 3.6±0.93.6\pm 0.9 | 6.4±1.66.4\pm 1.6 | 10.6±2.710.6\pm 2.7 |
| SOAR-PQ (HARP) (Ours) | 0.5±0.10.5\pm 0.1 | 2.0±0.52.0\pm 0.5 | 3.8±1.0\mathbf{3.8\pm 1.0} | 7.0±1.8\mathbf{7.0\pm 1.8} | 12.0±3.0\mathbf{12.0\pm 3.0} |
| SOAR-PS (MATH) (Ours) | 0.6±0.1\mathbf{0.6\pm 0.1} | 2.1±0.5\mathbf{2.1\pm 0.5} | 3.7±0.83.7\pm 0.8 | 6.2±1.36.2\pm 1.3 | 9.9±2.29.9\pm 2.2 |
| SOAR-PS (HARP) (Ours) | 0.5±0.10.5\pm 0.1 | 2.0±0.42.0\pm 0.4 | 3.8±0.7\mathbf{3.8\pm 0.7} | 6.9±1.16.9\pm 1.1 | 11.7±1.611.7\pm 1.6 |
| Grounded-T (MATH) (Ours) | 0.4±0.20.4\pm 0.2 | 1.6±0.81.6\pm 0.8 | 2.9±1.42.9\pm 1.4 | 5.3±2.45.3\pm 2.4 | 9.0±4.09.0\pm 4.0 |
| Grounded-T (HARP) (Ours) | 0.5±0.20.5\pm 0.2 | 1.9±0.61.9\pm 0.6 | 3.6±1.13.6\pm 1.1 | 6.5±1.86.5\pm 1.8 | 11.1±2.911.1\pm 2.9 |
| Intrinsic-T | 0.4±0.30.4\pm 0.3 | 1.7±1.21.7\pm 1.2 | 3.1±2.03.1\pm 2.0 | 5.5±3.45.5\pm 3.4 | 9.1±5.29.1\pm 5.2 |
| HARP train (128) | 0.5±0.10.5\pm 0.1 | 2.0±0.22.0\pm 0.2 | 3.6±0.43.6\pm 0.4 | 6.5±0.86.5\pm 0.8 | 10.6±1.710.6\pm 1.7 |
| MATH train (128) | 1.0±0.11.0\pm 0.1 | 3.4±0.13.4\pm 0.1 | 5.9±0.15.9\pm 0.1 | 9.6±0.49.6\pm 0.4 | 14.6±1.414.6\pm 1.4 |
| MATH train (Full) | 0.9±0.00.9\pm 0.0 | 3.2±0.13.2\pm 0.1 | 5.6±0.35.6\pm 0.3 | 8.8±0.78.8\pm 0.7 | 13.1±0.913.1\pm 0.9 |

Table 6: Olympiad Pass@k (%) Test Accuracy on fail@128. Mean and SD over seeds are reported timestep 50 with full curves in Figure 9. Despite being optimized with reward signals from HARP and MATH, PQ questions and PS inference transfer to improving performance on Olympiad, and match or outperform 128 questions sampled from the HARP train set (a curated/expert-annotated source of problems). PS and PQ transfer better when trained with HARP than with MATH, potentially indicating more shared structure between HARP and Olympiad.

In Tables 4-5 we report our full results from evaluating SOAR on MATH and HARP (in-domain datasets). In Table 6 we report full results from evaluating on OlympiadBench, an OOD dataset.

Our PQ datasets have one of {128,192,256}\{128,192,256\} questions, depending on the number of student promotions for each run. For Intrinsic-T we sample 128 questions, consistent with all of our teacher-sampling experiments. For the equal-data comparison between Intrinsic-T and Grounded-T (sampling from the SOAR-trained teacher), see Section 5.2 and Appendix C.3.

In addition to the methods/baselines shown in Figure 4 we also report the following.

Inference pass@k with the base model. Inference with the base model has non-zero pass@kk due to stochastic sampling with different seeds than were used for the initial pass@​128=0@128=0 filtering. Comparison with Hard-Only results shows that our fail@128 datasets are sufficiently difficult such that direct training yields very little improvement. Doing inference with the trained Grounded-T teacher model directly on fail@128 MATH test questions does not improve upon base model, further evidence for the decoupling of generation and solving abilities.

Hard-Only with extra compute. A natural question is whether we can improve direct training on fail@128 train questions simply by increasing compute. One strategy is to train for longer, however our learning curves in Figure 9 show that Hard-Only test performance decreases in the latter stages of training. Another strategy is to sample more from the base model by increasing the RLOO group size. On MATH, we increase the group size 4×4\times (from our default g=32g=32 to g=128g=128), and find that it only yields marginal improvements over Hard-Only (e.g.,  +2.8% pass@32) and does not recover the improvements of PQ.

Sampling curated “oracle questions". In addition to training with the full MATH train set, we also evaluate sampling 128 questions from the MATH and HARP train sets, which can be considered oracle (curated/expert-annotated) data sources. We choose 128 to match our teacher sampling experiments (Section C.3) and roughly match the amount of PQ data, which varies between 128 and 256 questions.

On MATH, training with these smaller subsets performs similarly to training with the full MATH dataset, suggesting a saturation point. On HARP, these smaller subsets only recover ≈50%\approx 50\% of the gains from training with the full MATH train set. Notably, PQ and PS both outperform 128 sampled questions from HARP, and match 128 questions from MATH.

### C.3 Sampling from Trained Teachers.

While PQ comes from accumulated useful questions over the meta-RL trajectory, here we sample questions directly from the trained teacher policy. The similar performance of Grounded-T and PQ (Tables 4-5) provide evidence that the pedagogical signals captured in the PQ datasets are learned by the teacher’s distribution.

In Figures 10-12 we show full test trajectories on MATH, HARP, and Olympiad for students trained with 128 questions sampled from Grounded-T, Intrinsic-T, Base-T, and Grounded-T (no promotion). Grounded-T outperforms all comparisons, particularly at higher inference budgets, and is competitive with PQ. Grounded-T also exhibits lower variance and greater stability across student and teacher seeds. Grounded-T (no promotion) performs worse than Grounded-T, PQ, and PS, validating the importance of the promotion mechanism.

In Figure 13 we also compare student trajectories for each Grounded-T and Intrinsic-T teacher seed. Consistent with MATH and HARP (Figure 5), students have similar trajectories across independent Grounded-T teachers, and high variance across different Intrinsic-T teachers, showcasing the instability of intrinsic rewards.

### C.4 Correctness of Synthetic Questions

We categorize synthetic questions into correctness taxonomies using Claude-4.5-Sonnet as an oracle judge. The prompt given to Claude is shown below. In Table 7 we report taxonomy statistics for PQ datasets, and problems sampled from Grounded-T, Intrinsic-T, and Base-T teachers.

We prompt Claude-4.5-Sonnet to categorize problems as follows:

* •

  Well posed: If the problem is mathematically complete and solvable.
* •

  Correct: If the proposed answer is correct (only if the problem is well posed).
* •

  Error type:

  + –

    None
  + –

    Arithmetic error: Sound logic, but incorrect final calculation.
  + –

    Logical fallacy: Does not follow mathematical rules.
  + –

    Ill-posed/Impossibility: The question contains a mathematical impossibility.
  + –

    Ambiguous: The question is missing data, variables, or context necessary for solving it.

Our results show that the well-posedness of a problem matters more than the correctness of the solution. While teacher-training does improve the correctness rate, the best-performing datasets (Grounded-T and PQ) only contain 32.8% and 36.5% correct solutions respectively, compared to 55.5% for Intrinsic-T. This indicates that question diversity is more important for success (see Table 1). Question structure and coherence is more important; meta-RL reduces question ambiguities while the rate of arithmetic errors remains the same or slightly higher.

Oracle Prompt



|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Category | Base | Intrinsic | Grounded | PQ |
| Well-Posed | 53.6% | 63.5% | 70.0% | 64.6% |
| Correct | 23.2% | 55.5% | 36.5% | 32.8% |
| Error Taxonomy (% of total samples) | |  |  |  |
| Arithmetic Error | 23.7% | 5.7% | 29.0% | 25.0% |
| Logic Error | 5.7% | 2.3% | 6.9% | 6.5% |
| Impossibility Error | 4.7% | 2.9% | 8.2% | 4.7% |
| Ambiguity Error | 42.4% | 33.6% | 21.3% | 31.3% |
| Total Samples | 384 | 384 | 375 | 384 |

Table 7: Correctness analysis and error taxonomy of synthetic questions, evaluated by Claude-4.5-Sonnet. Teacher training (for both grounded and intrinsic rewards) improves the well-posedness and correctness of problems relative to the base model, with a corresponding decrease in question ambiguity errors. Grounded-T and PQ have fewer correct questions than Intrinsic-T but perform better, potentially because of greater diversity (see Table 1.)

## Appendix D Ablations

### D.1 Sampled dataset size

![Refer to caption](/html/2601.18778/assets/x14.png)


Figure 14: (Left) Sampling different-sized datasets from Grounded-T for MATH (fail@128) Mean and ±\pm 1 SD across 2 teacher seeds and 2 student seeds. (Right) Sampling different-sized datasets from the MATH trainset for MATH (fail@128). Resampled for each seed, 3 seeds.

When training with SOAR, teacher-generated problems are partitioned into datasets that the student is trained on in the inner loop. Thus the teacher rewards are based on a specific dataset size (64 in our case). In evaluation, however, one could potentially sample any number of questions from the teacher policy. This raises the question of how the performance of sampled datasets changes with size. Is it best to sample the number of questions that the teacher was trained with, or does performance saturate at higher sampling rates?

We evaluate two teacher models trained with MATH by sampling n∈{32,64,128}n\in\{32,64,128\} questions from each teacher, and training a fresh student on the sampled questions and the MATH fail@128 train set (3 seeds per run). Since teacher models are trained with n=64n=64, this covers datasets smaller, equal to, and larger than the dataset size that the teacher was trained with.

Results are shown in Figure 14 for different pass@kk. Performance improves with increasing nn. Sampling with 128 questions has a similar mean performance as sampling 64 questions but with significantly smaller error. This illustrates benefits (namely, consistency/reliabilty) to sampling questions from the teacher at higher rates than it was trained with. As a comparison we also perform the same experiment using real questions from the MATH training dataset. For all values of nn, real MATH questions perform similarly or better, and exhibit diminishing variance with increasing numbers of questions.

### D.2 Sensitivity to Teacher Hyperaparameters

We ablate τ\tau (the teacher-reward threshold to determine if the student baseline should be promoted) and nn (the number of samples per dataset that teacher-generated problems are partitioned into). The teacher generates g⋅ng\cdot n problems per outer-RLOO iteration.

We train SOAR on MATH with τ∈{0.01,0.015}\tau\in\{0.01,0.015\} and n∈{32,64}n\in\{32,64\}. For each combination we train two SOAR runs for 200 steps and evaluate the final teacher checkpoints by sampling varying amounts of questions (|𝒳|∈{32,64,128}|\mathcal{X}|\in\{32,64,128\}) and training two fresh students. Results are shown in Figure 15 for pass@8 and pass@32.
Our default configuration (nn=64, τ\tau=0.01) performs best, with n=64n=64 showing modest
advantages over n=32n=32 at larger evaluation dataset
sizes, which is consistent with the teacher being trained to produce larger datasets.

![Refer to caption](/html/2601.18778/assets/x15.png)


Figure 15: Hyperparameter sensitivity on MATH. We train SOAR with τ∈{0.01,0.015}\tau\in\{0.01,0.015\} and n∈{32,64}n\in\{32,64\}, then evaluate by training students on datasets of size |𝒳|∈{32,64,128}|\mathcal{X}|\in\{32,64,128\}. Shaded regions indicate ±\pm1 SD.

### D.3 Problem Generation Format.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |  | MATH Pass@k (%) | | | | |
| nn | |𝒳||\mathcal{X}| | 1 | 4 | 8 | 16 | 32 |
| 32 | 32 | 0.66±0.58\mathbf{0.66\pm 0.58} | 2.34±1.91\mathbf{2.34\pm 1.91} | 4.16±3.13\mathbf{4.16\pm 3.13} | 7.06±4.75\mathbf{7.06\pm 4.75} | 11.42±6.66\mathbf{11.42\pm 6.66} |
|  | 64 | 0.52±0.260.52\pm 0.26 | 1.93±0.931.93\pm 0.93 | 3.60±1.633.60\pm 1.63 | 6.44±2.666.44\pm 2.66 | 10.99±3.9610.99\pm 3.96 |
|  | 128 | 0.67±0.670.67\pm 0.67 | 2.29±2.032.29\pm 2.03 | 4.03±3.254.03\pm 3.25 | 6.82±4.916.82\pm 4.91 | 11.06±7.0511.06\pm 7.05 |
| 64 | 32 | 0.44±0.120.44\pm 0.12 | 1.61±0.421.61\pm 0.42 | 2.95±0.762.95\pm 0.76 | 5.16±1.395.16\pm 1.39 | 8.56±2.488.56\pm 2.48 |
|  | 64 | 0.38±0.040.38\pm 0.04 | 1.49±0.151.49\pm 0.15 | 2.85±0.282.85\pm 0.28 | 5.29±0.485.29\pm 0.48 | 9.35±0.849.35\pm 0.84 |
|  | 128 | 0.43±0.120.43\pm 0.12 | 1.55±0.361.55\pm 0.36 | 2.80±0.572.80\pm 0.57 | 4.83±0.894.83\pm 0.89 | 7.96±1.327.96\pm 1.32 |

Table 8: MATH Pass@kk results for multi-turn teacher sampling. We report mean and SD across four teacher seeds and 2 student seeds per teacher. Multiturn performs worse than our default single-turn setting across all pass@k and sampled dataset sizes.

In our default setup, we sample problems from the teacher by prompting it to produce a single completion that is parsed into a question/answer, and filtering out outputs that do not match the necessary format. An alternative sampling method, however, is to generate problems in separate question-answer stages (multi-turn) such that filtering is not needed:

1. 1.

   Sample πϕT​(qi|p)\pi_{\phi}^{T}(q_{i}|p) where pp is a teacher prompt to generate a question.
2. 2.

   Sample πϕT​(ai|p,qi,p′)\pi_{\phi}^{T}(a_{i}|p,q_{i},p^{\prime}) where p′p^{\prime} is a prompt to generate an answer given the question.

The logprob component of the teacher RLOO loss is then log⁡(πϕT​(qi|p))+log⁡(πϕT​(ai|p,qi,p′))\log(\pi_{\phi}^{T}(q_{i}|p))+\log(\pi_{\phi}^{T}(a_{i}|p,q_{i},p^{\prime})).

We execute SOAR across four seeds using this teacher-sampling formulation with our standard procedure and hyperparameters, ablating n∈{32,64}n\in\{32,64\}. We observe that the teacher reward quickly plateaus and does not exceed one promotion. In Table 8 we find that across different numbers of sampled problems and values of nn, the multi-turn sampling strategy performs worse than our default single-turn sampling.

## Appendix E Teacher Training Dynamics

In Figure 16 we show a representative teacher training curve for SOAR on HARP. We observe that SOAR follows a pattern of search and exploitation. The training curve exhibits periods of oscillation (search), and then a steady rise in reward from steps 18-27, culminating in a student promotion. The reward declines after the promotion, due to the improved student baseline, oscillates as the teacher adapts to the improved student, and then exhibits another rise from steps 80-86 culminating in a second promotion.

Figure 17a shows teacher training curves for Intrinsic-T teachers, aggregated across teacher seeds, which exhibits a smooth upward climb. Figure 17b shows that as the Intrinsic-T reward climbs, the diversity of teacher completions falls (diversity measured as the average pairwise cosine distance of embeddings). Meanwhile Grounded-T preserves the original model diversity throughout the full trajectory. This is consistent with findings in Section 5.2 (Table 1) that Grounded-T achieves similar question diversity to Base-T, whereas Intrinsic-T teachers collapse to a more narrow conceptual space.

![Refer to caption](/html/2601.18778/assets/x16.png)


Figure 16: Annotated teacher reward dynamics when training SOAR with HARP. Shows a sample teacher trajectory from a SOAR run on HARP. The teacher follows a cyclical search-exploitation pattern. Student promotions (updating the student baseline to a trained student) are triggered when the 3-step moving average of teacher rewards exceeds τ=0.01\tau=0.01. After each promotion, the improved student baseline makes previous curricula less useful, causing rewards to drop, and then recover as the teacher adapts and discovers questions appropriate for the improved student.

![Refer to caption](/html/2601.18778/assets/x17.png)


Figure 17: (Left) Teacher training dynamics when training with Intrinsic-T. Mean and ±\pm 1 SD over three independent training runs. (Right) Teacher completion diversity when training with intrinsic v. grounded rewards. Grounded rewards preserve diversity for the full run, while intrinsic teachers lose diversity as they converge. Mean and ±\pm 1 SD over three training runs for intrinsic and four for grounded (two MATH, two HARP).`

[◄](/html/2601.18777)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2601.18778)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2601.18778)
[View original  
on arXiv](https://arxiv.org/abs/2601.18778)[►](/html/2601.18779)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Thu Feb 5 14:35:53 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
