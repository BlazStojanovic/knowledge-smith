---
arxiv: '2602.13949'
authors:
- Taiwei Shi
- Sihao Chen
- Bowen Jiang
- Linxin Song
- Longqi Yang
- Jieyu Zhao
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Experiential Reinforcement Learning
url: https://arxiv.org/abs/2602.13949
year: 2026
---

[2602.13949] Experiential Reinforcement Learning














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



# Experiential Reinforcement Learning

Taiwei Shi1 , Sihao Chen2, Bowen Jiang3\*, Linxin Song1, Longqi Yang2, Jieyu Zhao1
  
1University of Southern California, 2Microsoft, 3University of Pennsylvania
  
{taiweish@usc.edu, sihaochen@microsoft.com}
Work done during internship at Microsoft’s Office of Applied Research

###### Abstract

Reinforcement learning has become the central approach for language models (LMs) to learn from environmental reward or feedback. In practice, the environmental feedback is usually sparse and delayed. Learning from such signals is challenging, as LMs must implicitly infer how observed failures should translate into behavioral changes for future iterations.
We introduce Experiential Reinforcement Learning (ERL), a training paradigm that embeds an explicit experience–reflection–consolidation loop into the reinforcement learning process. Given a task, the model generates an initial attempt, receives environmental feedback, and produces a reflection that guides a refined second attempt, whose success is reinforced and internalized into the base policy. This process converts feedback into structured behavioral revision, improving exploration and stabilizing optimization while preserving gains at deployment without additional inference cost. Across sparse-reward control environments and agentic reasoning benchmarks, ERL consistently improves learning efficiency and final performance over strong reinforcement learning baselines, achieving gains of up to +81% in complex multi-step environments and up to +11% in tool-using reasoning tasks. These results suggest that integrating explicit self-reflection into policy training provides a practical mechanism for transforming feedback into durable behavioral improvement.

![Refer to caption](/html/2602.13949/assets/x1.png)


Figure 1: 
In Experiential Reinforcement Learning (ERL), instead of learning from feedback or outcome directly, an agent learns to (1) verbally reflect on its experience and observed outcome, and (2) internalize the reflections to induce behavioral changes in future iterations.

## 1 Introduction

![Refer to caption](/html/2602.13949/assets/x2.png)


Figure 2: Conceptual comparison of learning dynamics in RLVR and Experiential Reinforcement Learning (ERL). RLVR relies on repeated trial-and-error driven by scalar rewards, leading to back-and-forth exploration without durable correction. ERL augments this process with an experience–reflection–consolidation loop that generates a revised attempt and internalizes successful corrections, enabling persistent behavioral improvement.

Large language models are increasingly deployed as decision-making agents that must act, observe feedback, and adapt their behavior in environments with delayed rewards and partial information (Singh et al., [2025](#bib.bib10 "OpenAI gpt-5 system card"); Yang et al., [2025](#bib.bib46 "Qwen3 technical report"); Song et al., [2025a](#bib.bib45 "CoAct-1: computer-using agents with coding as actions"); Bai et al., [2026](#bib.bib11 "Kimi k2: open agentic intelligence")). Reinforcement learning offers a natural framework for improving such agents.
The task environments typically provide feedback in the form of outcome reward after an agent generates the entire trajectory. In practice, training agents against such sparse and delayed outcome signals remains difficult, as models must implicitly infer
how to translate observed failures into corrective behavior, a process that is often unstable and sample-inefficient (Zhang et al., [2025](#bib.bib42 "Agent learning via early experience"); Shi et al., [2026](#bib.bib14 "Efficient reinforcement finetuning via adaptive curriculum learning")). These challenges become more pronounced in agentic reasoning tasks, where multi-step decisions could amplify small errors and obscure credit assignment.

Humans address similar challenges through a process often described as experiential learning, in which effective adaptation arises from a cycle of experience, reflection, conceptualization, and experimentation (Kolb, [2014](#bib.bib22 "Experiential learning: experience as the source of learning and development")). After observing an outcome, a learner reflects on what occurred, forms revised internal models, and applies those revisions in subsequent attempts. This cycle transforms raw feedback into actionable behavioral corrections before those corrections are consolidated into future behavior. While language models have demonstrated reflection-like capabilities at inference time, standard reinforcement learning pipelines largely reduce feedback to scalar optimization signals, requiring policies to implicitly discover corrective structure through undirected exploration rather than explicit experiential revision.

This perspective highlights a progression in how language models learn from supervision and interaction, illustrated in Figures [1](#S0.F1 "Figure 1 ‣ Experiential Reinforcement Learning") and [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Experiential Reinforcement Learning"). In supervised fine-tuning (SFT), policies imitate fixed examples, enabling strong pattern reproduction but offering no mechanism for revising behavior once deployed. Reinforcement learning with verifiable rewards (RLVR) extends learning into interactive settings by optimizing scalar feedback, allowing agents to improve through trial-and-error; however, corrective structure must still be inferred implicitly from sparse or delayed rewards. As visualized in Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Experiential Reinforcement Learning"), this can lead to repeated exploration without durable behavioral correction. A natural next step is to structure learning around experience itself, transforming feedback into intermediate reasoning that supports explicit revision and consolidation within each episode. Figure [1](#S0.F1 "Figure 1 ‣ Experiential Reinforcement Learning") conceptualizes this shift as moving from learning purely from feedback toward deliberate learning from experience.

In this work, we introduce Experiential Reinforcement Learning (ERL), a training paradigm that embeds an explicit experience–reflection–consolidation loop inside reinforcement learning. Instead of learning solely from outcome rewards, the model first produces an initial attempt, receives environment feedback, and generates a structured reflection describing how the attempt should be improved. This reflection conditions a refined second attempt, whose outcome is reinforced and internalized into the base policy. By converting feedback into intermediate reasoning signals, ERL enables the model to perform targeted behavioral correction before policy consolidation. Over time, these corrections become part of the policy itself, allowing improved behavior to persist even when reflection is absent at deployment. An overview of the algorithm is shown in Figure [3](#S1.F3 "Figure 3 ‣ Contributions. ‣ 1 Introduction ‣ Experiential Reinforcement Learning").

We evaluate ERL across sparse-reward control environments and agentic reasoning benchmarks spanning two model scales. ERL consistently outperforms RLVR in all six evaluated settings, achieving gains of up to +81% in Sokoban, +27% in FrozenLake, and up to +11% in HotpotQA. These results demonstrate that embedding structured experiential revision into training improves learning efficiency and produces stronger final policies across both control and reasoning tasks.

#### Contributions.

Our main contributions are:

* •

  We introduce Experiential Reinforcement Learning (ERL), a reinforcement learning paradigm that incorporates an explicit experience–reflection–consolidation loop, enabling models to transform environment feedback into structured behavioral corrections.
* •

  We propose an internalization mechanism that consolidates reflection-driven improvements into the base policy, preserving gains without requiring reflection at inference time.
* •

  We demonstrate that experiential reinforcement learning improves training efficiency and final performance across agentic reasoning tasks.

![Refer to caption](/html/2602.13949/assets/x3.png)


Figure 3: 
Overview of Experiential Reinforcement Learning (ERL).
Given an input task xx, the language model first produces an initial attempt and receives environment feedback.
The same model then generates a self-reflection conditioned on this attempt, which is used to guide a second attempt.
Both attempts and reflections are optimized with reinforcement learning, while successful second attempts are internalized via self-distillation, so the model learns to reproduce improved behavior directly from the original input without self-reflection.

## 2 Experiential Reinforcement Learning (ERL)

Algorithm 1  Experiential Reinforcement Learning

1: Inputs: Language model πθ\pi\_{\theta}; dataset of questions xx; reward threshold τ\tau;
environment returning feedback ff and reward rr.

2: Initialize: reflection memory m←∅m\leftarrow\emptyset.

3: repeat

4:  Sample question xx from the dataset.

5:  // First attempt

6:  Sample an answer y(1)∼πθ(⋅∣x)y^{(1)}\sim\pi\_{\theta}(\cdot\mid x).

7:  Obtain environment feedback and reward (f(1),r(1))(f^{(1)},r^{(1)}).

8:  // Self-reflection

9:  Sample a reflection Δ∼πθ(⋅∣x,y(1),f(1),r(1),m)\Delta\sim\pi\_{\theta}(\cdot\mid x,y^{(1)},f^{(1)},r^{(1)},m).

10:  // Second attempt

11:  Sample a refined answer y(2)∼πθ(⋅∣x,Δ)y^{(2)}\sim\pi\_{\theta}(\cdot\mid x,\Delta).

12:  Obtain environment feedback and reward (f(2),r(2))(f^{(2)},r^{(2)}).

13:  Set reflection reward r~←r(2)\tilde{r}\leftarrow r^{(2)}.

14:  Store reflection m←Δm\leftarrow\Delta\;\; if r(2)>τ\;\;r^{(2)}>\tau.

15:  // RL update

16:  Update θ\theta via
ℒpolicy​(θ)\mathcal{L}\_{\text{policy}}(\theta) over the first attempt, reflection, and second attempt.

17:  // Internalization

18:  Update θ\theta via ℒdistill​(θ)\mathcal{L}\_{\text{distill}}(\theta)
to internalize reflection, training πθ\pi\_{\theta}
to produce y(2)y^{(2)} from xx only.

19: until converged

We introduce *Experiential Reinforcement Learning* (ERL), a training framework that enables a language model to iteratively improve its behavior through self-generated feedback and internalization. The key idea is to treat reflection as an intermediate reasoning signal that guides a refined second attempt, while reinforcement learning aligns both attempts with reward, and supervised distillation consolidates successful behaviors into the base policy. An overview is shown in [Figure 3](#S1.F3 "In Contributions. ‣ 1 Introduction ‣ Experiential Reinforcement Learning"), and the core training loop appears in [Algorithm 1](#alg1 "In 2 Experiential Reinforcement Learning (ERL) ‣ Experiential Reinforcement Learning"). A detailed implementation, including memory persistence and gating logic, is provided in Appendix [A](#A1 "Appendix A Full Algorithm and Gated Reflection ‣ Experiential Reinforcement Learning").

Given an input task xx, the model πθ\pi\_{\theta} first produces an initial response

|  |  |  |
| --- | --- | --- |
|  | y(1)∼πθ(⋅∣x),y^{(1)}\sim\pi\_{\theta}(\cdot\mid x), |  |

which is evaluated by the environment to produce textual feedback f(1)f^{(1)} and reward r(1)r^{(1)}. Rather than immediately updating the policy, ERL optionally triggers a reflection-and-retry phase when the first attempt underperforms relative to a reward threshold τ\tau. This selective retry mechanism focuses compute on trajectories that are most likely to benefit from revision while avoiding unnecessary refinement when performance is already sufficient. When triggered, the model generates a reflection

|  |  |  |
| --- | --- | --- |
|  | Δ∼πθ(⋅∣x,y(1),f(1),r(1),m),\Delta\sim\pi\_{\theta}(\cdot\mid x,y^{(1)},f^{(1)},r^{(1)},m), |  |

which serves as self-guidance describing how the initial attempt can be improved. Here, mm denotes a cross-episode reflection memory that persists successful corrective patterns discovered during training. This memory provides contextual priors that help stabilize reflection generation and encourage reuse of previously effective strategies. The model then produces a refined response

|  |  |  |
| --- | --- | --- |
|  | y(2)∼πθ(⋅∣x,Δ),y^{(2)}\sim\pi\_{\theta}(\cdot\mid x,\Delta), |  |

and receives new feedback (f(2),r(2))(f^{(2)},r^{(2)}). Reflections that lead to sufficiently improved outcomes are stored back into memory,

|  |  |  |
| --- | --- | --- |
|  | m←Δifr(2)>τ,m\leftarrow\Delta\quad\text{if}\quad r^{(2)}>\tau, |  |

allowing corrective knowledge to accumulate across training episodes. The reflection is assigned reward r~=r(2)\tilde{r}=r^{(2)}, encouraging reflections that lead to improved downstream performance.

Both attempts and reflections are optimized using a reinforcement learning objective

|  |  |  |
| --- | --- | --- |
|  | ℒpolicy​(θ)=−𝔼​[A​log⁡πθ​(y∣x,⋅)],\mathcal{L}\_{\text{policy}}(\theta)=-\mathbb{E}\!\left[A\,\log\pi\_{\theta}(y\mid x,\cdot)\right], |  |

where yy denotes model outputs arising from the first attempt, reflection, or second attempt, and the conditioning context corresponds to the inputs specified in [Algorithm 1](#alg1 "In 2 Experiential Reinforcement Learning (ERL) ‣ Experiential Reinforcement Learning"). The advantage estimate AA is computed from the associated rewards.

While reflection and environment feedback provide strong training signals, such supervision is typically unavailable at deployment time, where the model must operate in a zero-shot setting.
We therefore introduce an internalization step that converts reflection-guided improvements into persistent policy behavior. The goal is to make the model remember corrections discovered during training and avoid repeating the same mistakes when feedback is absent. We implement internalization via selective distillation: we supervise the model to imitate only successful second attempts while removing reflection context from the input. Concretely, given a training example xx, we generate a refined response y(2)y^{(2)} and reward r(2)r^{(2)}, and optimize

|  |  |  |
| --- | --- | --- |
|  | ℒdistill​(θ)=−𝔼​[𝕀​(r(2)>0)​log⁡πθ​(y(2)∣x)],\mathcal{L}\_{\text{distill}}(\theta)=-\mathbb{E}\Big[\mathbb{I}\!\left(r^{(2)}>0\right)\,\log\pi\_{\theta}\!\left(y^{(2)}\mid x\right)\Big], |  |

where 𝕀​(⋅)\mathbb{I}(\cdot) is the indicator function. This trains πθ\pi\_{\theta} to reproduce improved behavior from the original input xx alone (no reflection), ensuring that lessons learned through feedback and self-reflection persist at test time.

By alternating between reinforcement learning, selective reflection, and distillation, ERL bootstraps self-improvement: reflections guide higher-quality retries, memory preserves effective corrective structure, reinforcement learning aligns behavior with reward, and distillation internalizes gains into the core model. Over time, this interaction stabilizes training, concentrates exploration on failure cases, and reduces dependence on explicit reflection at inference.

### 2.1 Comparison to Standard RLVR

Standard reinforcement learning with verifiable rewards (RLVR) optimizes a policy directly from scalar outcome signals. Given an input xx, the model samples a response y∼πθ(⋅∣x)y\sim\pi\_{\theta}(\cdot\mid x) and receives a reward rr, with policy updates derived from trajectory-level credit assignment. In this formulation, feedback influences learning only through reward-driven optimization, requiring the model to implicitly discover how failures should translate into behavioral change. Corrective structure therefore emerges slowly through repeated exploration, with no explicit mechanism for revising behavior within the same learning episode. This learning dynamic corresponds to trial-and-error optimization, as illustrated in Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Experiential Reinforcement Learning").

Experiential Reinforcement Learning (ERL) augments this loop with an explicit experience–reflection–consolidation stage embedded inside each trajectory. Instead of optimizing solely from outcome reward, the model converts environment feedback into a reflection that conditions a refined attempt. This intermediate revision produces a locally improved trajectory that is reinforced and later internalized through selective distillation, allowing the base policy to reproduce corrected behavior without reflection at inference. A cross-episode reflection memory further stabilizes this process by preserving corrective patterns that proved effective, allowing subsequent reflections to reuse prior improvements. Importantly, ERL preserves the underlying RLVR objective: policy gradients remain reward-driven, but operate over a richer trajectory structure that includes explicit behavioral correction. This reframing shifts feedback from a scalar endpoint signal to a catalyst for immediate revision, reducing reliance on undirected exploration while maintaining compatibility with standard reinforcement learning pipelines. This contrast between blind trial-and-error learning and reflection-guided revision is visualized in Figure [1](#S0.F1 "Figure 1 ‣ Experiential Reinforcement Learning") and Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Experiential Reinforcement Learning").

## 3 Experiment

We evaluate Experiential Reinforcement Learning (ERL) against standard RLVR on a set of agentic reasoning tasks.

### 3.1 Task

We evaluate ERL on three agentic reasoning tasks: Frozen Lake, Sokoban, and HotpotQA (Yang et al., [2018](#bib.bib18 "HotpotQA: a dataset for diverse, explainable multi-hop question answering")). Detailed environment descriptions are provided in Appendix [B](#A2 "Appendix B Envrionment Configuration Details ‣ Experiential Reinforcement Learning").

For Frozen Lake and Sokoban, we configure the environments with sparse terminal rewards following Wang et al. ([2025](#bib.bib16 "Cogito, ergo ludo: an agent that learns to play by reasoning and planning")) and Guertler et al. ([2025](#bib.bib33 "TextArena")). The agent receives reward only at episode completion: a reward of +1 is assigned for successfully achieving the objective and 0 otherwise. Crucially, we do not provide explicit game rules or environment dynamics. The model must infer task structure purely through interaction, with access limited to the available action set. This evaluation design is inspired by prior work on learning from experience, where the goal is to measure an agent’s ability to acquire task understanding through trial-and-error rather than relying on human-authored priors embedded in pretraining. The combination of sparse rewards and unknown dynamics therefore creates a challenging setting that emphasizes reasoning, planning, and experiential learning.

HotpotQA is adapted into an agentic multi-hop question-answering task following Search-R1 (Jin et al., [2025](#bib.bib9 "Search-r1: training LLMs to reason and leverage search engines with reinforcement learning")). Given a question, the model performs iterative tool-assisted retrieval before producing a final answer. To maintain consistency with the experiential learning setup, we provide only a default system prompt describing available tools, without additional task-specific guidance. Correctness is evaluated using token-level F1 against ground-truth answers. The reward function assigns 1.0 for exact matches, a proportional reward for partial matches with F1 score ≥0.3\geq 0.3, and 0 otherwise.

### 3.2 Models and Baselines

In our experiments, we train Olmo-3-7B-Instruct (Olmo et al., [2025](#bib.bib17 "Olmo 3")) and Qwen3-4B-Instruct-2507 (Yang et al., [2025](#bib.bib46 "Qwen3 technical report")) using both standard RLVR and our proposed ERL paradigm, with GRPO (Shao et al., [2024](#bib.bib28 "DeepSeekMath: pushing the limits of mathematical reasoning in open language models")) serving as the underlying policy-gradient optimizer in all cases. To ensure stable training, we adopt common reinforcement learning techniques such as clipping, KL regularization, and importance sampling. Notably, the internalization stage in ERL naturally involves off-policy data, which can introduce additional instability. We therefore apply the same stabilization techniques during this phase to maintain consistent optimization behavior. Additionally, because ERL requires two attempts per task along with an additional reflection step, we allocate 10 rollouts per task for RLVR and half as many per task per attempt for ERL to equalize the training compute per task across methods. Full hyperparameters and implementation details are provided in Appendix [C](#A3 "Appendix C Training Configuration Details ‣ Experiential Reinforcement Learning").

## 4 Result and Discussion

![Refer to caption](/html/2602.13949/assets/x4.png)


Figure 4: Validation reward trajectories versus training wall-clock time on FrozenLake, HotpotQA, and Sokoban for Qwen3-4B-Instruct-2507 and Olmo-3-7B-Instruct. ERL consistently achieves higher reward and faster improvement than RLVR across tasks and models.

We evaluate Experiential Reinforcement Learning (ERL) against standard RLVR across three environments spanning sparse-reward control (FrozenLake, Sokoban) and agentic reasoning (HotpotQA). Table [1](#S4.T1 "Table 1 ‣ 4.4 Ablation Study: Memory and Reflection Mechanisms ‣ 4 Result and Discussion ‣ Experiential Reinforcement Learning") summarizes the final performance, while Figures [5](#S4.F5 "Figure 5 ‣ 4.1 Performance Across Tasks ‣ 4 Result and Discussion ‣ Experiential Reinforcement Learning")–[6](#S4.F6 "Figure 6 ‣ 4.3 Mechanistic Role of Reflection ‣ 4 Result and Discussion ‣ Experiential Reinforcement Learning") visualize the performance and learning dynamics. All curves are smoothed with a trailing moving average over 5 points. The same smoothing procedure is applied to all figures unless otherwise noted.

### 4.1 Performance Across Tasks

ERL consistently improves final evaluation performance over RLVR across all tasks and both model backbones. As shown in Table [1](#S4.T1 "Table 1 ‣ 4.4 Ablation Study: Memory and Reflection Mechanisms ‣ 4 Result and Discussion ‣ Experiential Reinforcement Learning") and Figure [5](#S4.F5 "Figure 5 ‣ 4.1 Performance Across Tasks ‣ 4 Result and Discussion ‣ Experiential Reinforcement Learning"), experiential training yields gains ranging from moderate improvements on HotpotQA to substantial improvements on Sokoban and Frozenlake.

The largest effect occurs in Sokoban, where Qwen3-4B-Instruct improves from 0.06 to 0.87 and Olmo3-7B-Instruct from 0.04 to 0.20. Sokoban requires long-horizon planning and recovery from compounding errors, making performance sensitive to how well the agent reasons about environment dynamics. Similarly, FrozenLake demands that the agent infer symbol semantics, action consequences, and terminal conditions purely through interaction under sparse rewards. Importantly, as described in Section [3](#S3 "3 Experiment ‣ Experiential Reinforcement Learning"), unlike many prior evaluation setups that provide explicit rules or environment structure, our environments expose only observations and action interfaces; the agent must infer task dynamics through trial-and-error. This design emphasizes learning from experience rather than relying on pre-specified priors, making structured revision particularly valuable. In these settings, the experience–reflection–consolidation loop enables the model to analyze failures, revise strategies, and internalize corrective behavior within each episode, producing large improvements in exploration efficiency and policy quality.

HotpotQA shows smaller but reliable gains. A likely explanation lies in differences in task structure. Compared to the grid-based control environments, HotpotQA presents a more homogeneous interaction pattern centered on repeated tool invocation and answer synthesis, with denser evaluation feedback and fewer latent dynamics to infer. Because RLVR already receives relatively informative gradients in this regime, the additional benefit of structured experiential revision is reduced. This contrast suggests that ERL yields the greatest advantage in environments where learning requires substantial reasoning about unknown dynamics and long-horizon consequences, rather than primarily optimizing over a stable interaction loop.

Importantly, improvements are observed across both models, indicating that the benefits of ERL arise from enhanced learning dynamics rather than architecture-specific effects.

![Refer to caption](/html/2602.13949/assets/x5.png)


Figure 5: Final evaluation reward on FrozenLake, HotpotQA, and Sokoban. ERL consistently outperforms RLVR for both Qwen3-4B-Instruct-2507 and Olmo-3-7B-Instruct.

### 4.2 Learning Efficiency and Optimization Dynamics

Figure [4](#S4.F4 "Figure 4 ‣ 4 Result and Discussion ‣ Experiential Reinforcement Learning") compares validation reward against wall-clock training time. Across tasks and models, ERL reaches higher reward earlier and maintains a persistent margin over RLVR. This acceleration is especially pronounced in FrozenLake and Sokoban, where RLVR progresses gradually while ERL rapidly approaches high-reward behavior.

These dynamics suggest that reflection introduces an intermediate corrective signal that reshapes exploration. Instead of relying solely on terminal reward propagation, the model conditions on feedback and self-generated critique to revise its behavior. This concentrates training updates on trajectories that are already partially aligned with the objective, reducing inefficient exploration.

Even in HotpotQA, where rewards are denser and the environment is comparatively simpler, ERL maintains a consistent performance advantage over RLVR. Across environments, these results indicate that ERL achieves higher final reward while improving learning efficiency, demonstrating that structured experiential revision leads to faster and more effective policy improvement.

### 4.3 Mechanistic Role of Reflection

Figure [6](#S4.F6 "Figure 6 ‣ 4.3 Mechanistic Role of Reflection ‣ 4 Result and Discussion ‣ Experiential Reinforcement Learning") shows training reward trajectories for ERL before and after the reflection step, alongside RLVR. Across environments and models, post-reflection trajectories consistently achieve higher training reward than pre-reflection trajectories and also exceed RLVR.

This comparison highlights the immediate within-episode effect of reflection. After observing feedback from the first attempt, the model generates a structured revision that guides a second attempt with improved actions. The resulting gain in training reward indicates that reflection produces actionable corrections within the same episode, rather than only shaping behavior over long horizons. The sustained separation between pre- and post-reflection curves throughout training suggests that reflection serves as a systematic revision mechanism. By converting observed outcomes into targeted adjustments, it improves the quality of second attempts, which are subsequently reinforced and contribute to longer-term policy improvement.

![Refer to caption](/html/2602.13949/assets/x6.png)


Figure 6: Training reward trajectories for Qwen3-4B-Instruct-2507 and Olmo-3-7B-Instruct comparing RLVR with ERL before and after reflection. Post-reflection trajectories consistently achieve higher reward than both RLVR and pre-reflection trajectories.

### 4.4 Ablation Study: Memory and Reflection Mechanisms

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Task | RLVR | ERL | ERL w/o Mem. | ERL w/o Refl. |
| Qwen3-4B-Instruct-2507 | | | | |
| FrozenLake | 0.86 | 0.94 | 0.86 (-0.08) | 0.60 (-0.34) |
| HotpotQA | 0.45 | 0.56 | 0.56 (-0.00) | 0.48 (-0.08) |
| Sokoban | 0.06 | 0.87 | 0.87 (-0.00) | 0.59 (-0.28) |
| Olmo3-7B-Instruct | | | | |
| FrozenLake | 0.39 | 0.66 | 0.64 (-0.02) | 0.54 (-0.12) |
| HotpotQA | 0.47 | 0.50 | 0.47 (-0.03) | 0.46 (-0.04) |
| Sokoban | 0.04 | 0.20 | 0.24 (+0.04) | 0.06 (-0.14) |

Table 1: Final evaluation reward on FrozenLake, HotpotQA, and Sokoban. ERL performance is compared against ablation variants, with highlighted drops showing the performance degradation relative to ERL when removing memory reuse (w/o Mem.) or structured reflection (w/o Refl.).

![Refer to caption]()


Figure 7: Ablation study on Qwen3-4B-Instruct-2507 in FrozenLake. We compare full ERL with two variants: (1) no memory, which disables cross-episode reflection reuse, and (2) no reflection, which replaces structured self-reflection with raw first-attempt context and a generic retry instruction.

To understand how structured reflection and cross-episode memory contribute to performance, we conduct ablation studies across tasks and models. The quantitative results are reported in Table [1](#S4.T1 "Table 1 ‣ 4.4 Ablation Study: Memory and Reflection Mechanisms ‣ 4 Result and Discussion ‣ Experiential Reinforcement Learning"), and representative learning dynamics for FrozenLake with Qwen3-4B-Instruct-2507 are shown in Figure [7](#S4.F7 "Figure 7 ‣ 4.4 Ablation Study: Memory and Reflection Mechanisms ‣ 4 Result and Discussion ‣ Experiential Reinforcement Learning"). These experiments isolate individual components of ERL while keeping the overall training setup fixed.

The no-memory variant disables cross-episode reflection storage. Reflections are still generated and used to guide the second attempt within each episode, but they are not retained for reuse in future episodes. As a result, corrective signals remain local to individual trajectories rather than accumulating into persistent behavioral priors.

The no-reflection variant preserves the two-attempt interaction structure but removes explicit structured reflection. Instead, the model receives the full first-attempt interaction history together with a generic instruction encouraging improvement. This design tests whether contextual reuse alone can replicate the benefits of structured reflective reasoning. The prompt template used in this setting is shown in Table [9](#A2.T9 "Table 9 ‣ B.3 HotpotQA ‣ Appendix B Envrionment Configuration Details ‣ Experiential Reinforcement Learning") (Appendix).

The results in Table [1](#S4.T1 "Table 1 ‣ 4.4 Ablation Study: Memory and Reflection Mechanisms ‣ 4 Result and Discussion ‣ Experiential Reinforcement Learning") show a consistent ordering across most tasks and models: full ERL achieves the strongest performance, followed by the no-memory variant, while the no-reflection variant exhibits the largest degradation in most settings. Figure [7](#S4.F7 "Figure 7 ‣ 4.4 Ablation Study: Memory and Reflection Mechanisms ‣ 4 Result and Discussion ‣ Experiential Reinforcement Learning") further illustrates that removing memory slows convergence, whereas removing reflection substantially reduces both learning speed and final reward. These findings support the core design intuition of ERL: reflection generates actionable behavioral corrections, and memory propagates those corrections across episodes to enable cumulative refinement.

At the same time, Table [1](#S4.T1 "Table 1 ‣ 4.4 Ablation Study: Memory and Reflection Mechanisms ‣ 4 Result and Discussion ‣ Experiential Reinforcement Learning") reveals an important caveat. In the Olmo3-7B-Instruct Sokoban setting, the no-memory variant slightly outperforms full ERL. This suggests that when a model’s self-reflective ability is limited, or when the environment is complex and stochastic, persistent memory may propagate early inaccurate reflections, making recovery more difficult. In such cases, disabling cross-episode memory can mitigate the accumulation of erroneous priors. Nevertheless, across the broad set of tasks and models evaluated, ERL consistently delivers the strongest overall performance, demonstrating that structured reflection combined with persistent memory is highly effective in most practical settings.

## 5 Related Work

#### Reinforcement Learning for LLMs.

Reinforcement learning has become a central approach for improving large language models. Early work focused on reinforcement learning from human feedback (RLHF) to align model behavior with human preferences and conversational objectives (Ouyang et al., [2022](#bib.bib12 "Training language models to follow instructions with human feedback"); Christiano et al., [2023](#bib.bib13 "Deep reinforcement learning from human preferences"); Shi et al., [2024](#bib.bib32 "Safer-instruct: aligning language models with automated preference data"); [2025](#bib.bib31 "WildFeedback: aligning llms with in-situ user interactions and feedback")). More recent efforts extend RL to enhance mathematical reasoning, where verifiable or programmatic rewards derived from executable checks or formal answer verification provide structured supervision for reasoning and solution construction (OpenAI et al., [2024](#bib.bib41 "OpenAI o1 system card"); Guo et al., [2025](#bib.bib40 "DeepSeek-r1 incentivizes reasoning in llms through reinforcement learning"); Song et al., [2025b](#bib.bib30 "The hallucination tax of reinforcement finetuning"); Shi et al., [2026](#bib.bib14 "Efficient reinforcement finetuning via adaptive curriculum learning")). In parallel, research on tool-using and agentic LLMs treats the model as a policy that interacts with external environments, alternating between actions and observations under task-dependent rewards to solve multi-step problems (Yao et al., [2023](#bib.bib43 "ReAct: synergizing reasoning and acting in language models"); Jin et al., [2025](#bib.bib9 "Search-r1: training LLMs to reason and leverage search engines with reinforcement learning"); Bai et al., [2026](#bib.bib11 "Kimi k2: open agentic intelligence"); Jiang et al., [2026](#bib.bib24 "One model, all roles: multi-turn, multi-agent self-play reinforcement learning for conversational social intelligence")). Despite their different goals, these approaches primarily treat environment feedback as a scalar optimization signal propagated through policy gradients, requiring the model to implicitly infer corrective structure through exploration. In contrast, our ERL paradigm introduces an explicit experience-reflection-consolidation loop that transforms environment feedback into structured behavioral revision before internalizing improvements into the base policy.

#### Learning from Experience.

A growing body of work argues that the next scaling regime for AI will come not from more static human text, but from agents generating ever-richer data through interaction-i.e., learning predominantly from experience. Silver and Sutton ([2025](#bib.bib15 "Welcome to the era of experience")) emphasizes that continual, agent-generated data streams and long-horizon decision-making as the route beyond imitation of human corpora. This motivates algorithmic mechanisms that convert failures into usable learning signal rather than relying on rare successes. In classic reinforcement learning, Andrychowicz et al. ([2018](#bib.bib34 "Hindsight experience replay")) addresses sparse rewards by relabeling goals so that failed trajectories can still provide informative updates, substantially improving sample efficiency in goal-conditioned tasks. In the LLM-agent setting, Zhang et al. ([2025](#bib.bib42 "Agent learning via early experience")) similarly targets the gap between imitation and full RL by training agents on their own interaction traces even when explicit rewards are unavailable, using the agent’s generated future states as supervision and including self-reflection as a way to learn from suboptimal actions. Meanwhile, inference-time reflection methods demonstrate that LLMs can critique and revise their own outputs to improve success (Zelikman et al., [2022](#bib.bib23 "STar: bootstrapping reasoning with reasoning"); Madaan et al., [2023](#bib.bib39 "Self-refine: iterative refinement with self-feedback"); Shinn et al., [2023](#bib.bib38 "Reflexion: language agents with verbal reinforcement learning")), but typically require reflection or memory at deployment.
Concurrent research explores integrating feedback-conditioned improvement directly into training. hübotter2026reinforcementlearningselfdistillation; Song et al. ([2026](#bib.bib37 "Expanding the capabilities of reinforcement learning via text feedback")) formalize RL with textual feedback by distilling a feedback-conditioned teacher policy into a student policy. ERL is aligned with this direction but emphasizes explicit self-reflection as an intermediate reasoning step embedded inside the RL trajectory, where an initial attempt is followed by reflection and a refined retry. Coupled with selective internalization and cross-episode memory, this design treats reflection as a structured credit-assignment mechanism that transforms raw experience into durable behavioral improvement without requiring reflection at inference time.

## 6 Conclusion

In this work, we presented Experiential Reinforcement Learning (ERL), a training paradigm that incorporates an explicit experience–reflection–consolidation stage into the reinforcement learning loop to convert environment feedback into structured behavioral correction. By pairing reflection-guided revision with selective internalization, ERL enables models to learn corrective strategies during training and consolidate them into a deployable policy that operates without reflection at inference time. Across sparse-reward control and agentic reasoning tasks, ERL improves learning efficiency, stabilizes optimization, and produces stronger final policies relative to standard reinforcement learning baselines. These results demonstrate that embedding structured experiential revision directly into the training process provides an effective mechanism for translating feedback into durable behavioral improvement. Looking forward, this work suggests a path toward reinforcement learning systems that are fundamentally grounded in experience, where explicit reflection and consolidation become core primitives for building agents that continually learn, adapt, and improve from their own interactions.

## Acknowledgements

The authors thank the members of the LIME Lab and Microsoft Office of Applied Research for their helpful discussions, feedback, and resources.

## References

* R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. Ramos Garea, M. Geist, and O. Bachem (2024)
  On-policy distillation of language models: learning from self-generated mistakes.
  In International Conference on Learning Representations, B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (Eds.),
  Vol. 2024,  pp. 21246–21263.
  External Links: [Link](https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf)
  Cited by: [Appendix A](#A1.SS0.SSS0.Px3.p1.6 "On-Policy Distillation. ‣ Appendix A Full Algorithm and Gated Reflection ‣ Experiential Reinforcement Learning").
* M. Andrychowicz, F. Wolski, A. Ray, J. Schneider, R. Fong, P. Welinder, B. McGrew, J. Tobin, P. Abbeel, and W. Zaremba (2018)
  Hindsight experience replay.
  External Links: 1707.01495,
  [Link](https://arxiv.org/abs/1707.01495)
  Cited by: [§5](#S5.SS0.SSS0.Px2.p1.1 "Learning from Experience. ‣ 5 Related Work ‣ Experiential Reinforcement Learning").
* Y. Bai, Y. Bao, Y. Charles, C. Chen, G. Chen, H. Chen, H. Chen, J. Chen, N. Chen, R. Chen, Y. Chen, Y. Chen, Y. Chen, Z. Chen, J. Cui, H. Ding, M. Dong, A. Du, C. Du, D. Du, Y. Du, Y. Fan, Y. Feng, K. Fu, B. Gao, C. Gao, H. Gao, P. Gao, T. Gao, Y. Ge, S. Geng, Q. Gu, X. Gu, L. Guan, H. Guo, J. Guo, X. Hao, T. He, W. He, W. He, Y. He, C. Hong, H. Hu, Y. Hu, Z. Hu, W. Huang, Z. Huang, Z. Huang, T. Jiang, Z. Jiang, X. Jin, Y. Kang, G. Lai, C. Li, F. Li, H. Li, M. Li, W. Li, Y. Li, Y. Li, Y. Li, Z. Li, Z. Li, H. Lin, X. Lin, Z. Lin, C. Liu, C. Liu, H. Liu, J. Liu, J. Liu, L. Liu, S. Liu, T. Y. Liu, T. Liu, W. Liu, Y. Liu, Y. Liu, Y. Liu, Y. Liu, Z. Liu, E. Lu, H. Lu, L. Lu, Y. Luo, S. Ma, X. Ma, Y. Ma, S. Mao, J. Mei, X. Men, Y. Miao, S. Pan, Y. Peng, R. Qin, Z. Qin, B. Qu, Z. Shang, L. Shi, S. Shi, F. Song, J. Su, Z. Su, L. Sui, X. Sun, F. Sung, Y. Tai, H. Tang, J. Tao, Q. Teng, C. Tian, C. Wang, D. Wang, F. Wang, H. Wang, H. Wang, J. Wang, J. Wang, J. Wang, S. Wang, S. Wang, S. Wang, X. Wang, Y. Wang, Y. Wang, Y. Wang, Y. Wang, Y. Wang, Z. Wang, Z. Wang, Z. Wang, Z. Wang, C. Wei, Q. Wei, H. Wu, W. Wu, X. Wu, Y. Wu, C. Xiao, J. Xie, X. Xie, W. Xiong, B. Xu, J. Xu, L. H. Xu, L. Xu, S. Xu, W. Xu, X. Xu, Y. Xu, Z. Xu, J. Xu, J. Xu, J. Yan, Y. Yan, H. Yang, X. Yang, Y. Yang, Y. Yang, Z. Yang, Z. Yang, Z. Yang, H. Yao, X. Yao, W. Ye, Z. Ye, B. Yin, L. Yu, E. Yuan, H. Yuan, M. Yuan, S. Yuan, H. Zhan, D. Zhang, H. Zhang, W. Zhang, X. Zhang, Y. Zhang, Y. Zhang, Y. Zhang, Y. Zhang, Y. Zhang, Y. Zhang, Y. Zhang, Y. Zhang, Z. Zhang, H. Zhao, Y. Zhao, Z. Zhao, H. Zheng, S. Zheng, L. Zhong, J. Zhou, X. Zhou, Z. Zhou, J. Zhu, Z. Zhu, W. Zhuang, and X. Zu (2026)
  Kimi k2: open agentic intelligence.
  External Links: 2507.20534,
  [Link](https://arxiv.org/abs/2507.20534)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Experiential Reinforcement Learning"),
  [§5](#S5.SS0.SSS0.Px1.p1.1 "Reinforcement Learning for LLMs. ‣ 5 Related Work ‣ Experiential Reinforcement Learning").
* P. Christiano, J. Leike, T. B. Brown, M. Martic, S. Legg, and D. Amodei (2023)
  Deep reinforcement learning from human preferences.
  External Links: 1706.03741,
  [Link](https://arxiv.org/abs/1706.03741)
  Cited by: [§5](#S5.SS0.SSS0.Px1.p1.1 "Reinforcement Learning for LLMs. ‣ 5 Related Work ‣ Experiential Reinforcement Learning").
* T. Dao (2024)
  FlashAttention-2: faster attention with better parallelism and work partitioning.
  In International Conference on Learning Representations (ICLR),
  Cited by: [Appendix C](#A3.p1.1 "Appendix C Training Configuration Details ‣ Experiential Reinforcement Learning").
* M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P. Mazaré, M. Lomeli, L. Hosseini, and H. Jégou (2024)
  The faiss library.
  External Links: 2401.08281
  Cited by: [§B.3](#A2.SS3.p3.2 "B.3 HotpotQA ‣ Appendix B Envrionment Configuration Details ‣ Experiential Reinforcement Learning").
* L. Guertler, B. Cheng, S. Yu, B. Liu, L. Choshen, and C. Tan (2025)
  TextArena.
  External Links: 2504.11442,
  [Link](https://arxiv.org/abs/2504.11442)
  Cited by: [§B.1](#A2.SS1.p1.3 "B.1 Frozen Lake ‣ Appendix B Envrionment Configuration Details ‣ Experiential Reinforcement Learning"),
  [§B.2](#A2.SS2.p1.3 "B.2 Sokoban ‣ Appendix B Envrionment Configuration Details ‣ Experiential Reinforcement Learning"),
  [§3.1](#S3.SS1.p2.1 "3.1 Task ‣ 3 Experiment ‣ Experiential Reinforcement Learning").
* D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi, X. Zhang, X. Yu, Y. Wu, Z. F. Wu, Z. Gou, Z. Shao, Z. Li, Z. Gao, A. Liu, B. Xue, B. Wang, B. Wu, B. Feng, C. Lu, C. Zhao, C. Deng, C. Ruan, D. Dai, D. Chen, D. Ji, E. Li, F. Lin, F. Dai, F. Luo, G. Hao, G. Chen, G. Li, H. Zhang, H. Xu, H. Ding, H. Gao, H. Qu, H. Li, J. Guo, J. Li, J. Chen, J. Yuan, J. Tu, J. Qiu, J. Li, J. L. Cai, J. Ni, J. Liang, J. Chen, K. Dong, K. Hu, K. You, K. Gao, K. Guan, K. Huang, K. Yu, L. Wang, L. Zhang, L. Zhao, L. Wang, L. Zhang, L. Xu, L. Xia, M. Zhang, M. Zhang, M. Tang, M. Zhou, M. Li, M. Wang, M. Li, N. Tian, P. Huang, P. Zhang, Q. Wang, Q. Chen, Q. Du, R. Ge, R. Zhang, R. Pan, R. Wang, R. J. Chen, R. L. Jin, R. Chen, S. Lu, S. Zhou, S. Chen, S. Ye, S. Wang, S. Yu, S. Zhou, S. Pan, S. S. Li, S. Zhou, S. Wu, T. Yun, T. Pei, T. Sun, T. Wang, W. Zeng, W. Liu, W. Liang, W. Gao, W. Yu, W. Zhang, W. L. Xiao, W. An, X. Liu, X. Wang, X. Chen, X. Nie, X. Cheng, X. Liu, X. Xie, X. Liu, X. Yang, X. Li, X. Su, X. Lin, X. Q. Li, X. Jin, X. Shen, X. Chen, X. Sun, X. Wang, X. Song, X. Zhou, X. Wang, X. Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. Zhang, Y. Xu, Y. Li, Y. Zhao, Y. Sun, Y. Wang, Y. Yu, Y. Zhang, Y. Shi, Y. Xiong, Y. He, Y. Piao, Y. Wang, Y. Tan, Y. Ma, Y. Liu, Y. Guo, Y. Ou, Y. Wang, Y. Gong, Y. Zou, Y. He, Y. Xiong, Y. Luo, Y. You, Y. Liu, Y. Zhou, Y. X. Zhu, Y. Huang, Y. Li, Y. Zheng, Y. Zhu, Y. Ma, Y. Tang, Y. Zha, Y. Yan, Z. Z. Ren, Z. Ren, Z. Sha, Z. Fu, Z. Xu, Z. Xie, Z. Zhang, Z. Hao, Z. Ma, Z. Yan, Z. Wu, Z. Gu, Z. Zhu, Z. Liu, Z. Li, Z. Xie, Z. Song, Z. Pan, Z. Huang, Z. Xu, Z. Zhang, and Z. Zhang (2025)
  DeepSeek-r1 incentivizes reasoning in llms through reinforcement learning.
  Nature 645 (8081),  pp. 633–638.
  External Links: ISSN 1476-4687,
  [Link](http://dx.doi.org/10.1038/s41586-025-09422-z),
  [Document](https://dx.doi.org/10.1038/s41586-025-09422-z)
  Cited by: [§5](#S5.SS0.SSS0.Px1.p1.1 "Reinforcement Learning for LLMs. ‣ 5 Related Work ‣ Experiential Reinforcement Learning").
* B. Jiang, T. Shi, R. Kamoi, Y. Yuan, C. J. Taylor, L. Yang, P. Zhou, and S. Chen (2026)
  One model, all roles: multi-turn, multi-agent self-play reinforcement learning for conversational social intelligence.
  External Links: 2602.03109,
  [Link](https://arxiv.org/abs/2602.03109)
  Cited by: [§5](#S5.SS0.SSS0.Px1.p1.1 "Reinforcement Learning for LLMs. ‣ 5 Related Work ‣ Experiential Reinforcement Learning").
* B. Jin, H. Zeng, Z. Yue, J. Yoon, S. O. Arik, D. Wang, H. Zamani, and J. Han (2025)
  Search-r1: training LLMs to reason and leverage search engines with reinforcement learning.
  In Second Conference on Language Modeling,
  External Links: [Link](https://openreview.net/forum?id=Rwhi91ideu)
  Cited by: [§B.3](#A2.SS3.p4.1 "B.3 HotpotQA ‣ Appendix B Envrionment Configuration Details ‣ Experiential Reinforcement Learning"),
  [§3.1](#S3.SS1.p3.1 "3.1 Task ‣ 3 Experiment ‣ Experiential Reinforcement Learning"),
  [§5](#S5.SS0.SSS0.Px1.p1.1 "Reinforcement Learning for LLMs. ‣ 5 Related Work ‣ Experiential Reinforcement Learning").
* D. A. Kolb (2014)
  Experiential learning: experience as the source of learning and development.
  2 edition, FT Press, Upper Saddle River, NJ.
  External Links: ISBN 9780133892505
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ Experiential Reinforcement Learning").
* W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica (2023)
  Efficient memory management for large language model serving with pagedattention.
  In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles,
  Cited by: [Appendix C](#A3.p1.1 "Appendix C Training Configuration Details ‣ Experiential Reinforcement Learning").
* A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe, U. Alon, N. Dziri, S. Prabhumoye, Y. Yang, S. Gupta, B. P. Majumder, K. Hermann, S. Welleck, A. Yazdanbakhsh, and P. Clark (2023)
  Self-refine: iterative refinement with self-feedback.
  External Links: 2303.17651,
  [Link](https://arxiv.org/abs/2303.17651)
  Cited by: [§5](#S5.SS0.SSS0.Px2.p1.1 "Learning from Experience. ‣ 5 Related Work ‣ Experiential Reinforcement Learning").
* T. Olmo, :, A. Ettinger, A. Bertsch, B. Kuehl, D. Graham, D. Heineman, D. Groeneveld, F. Brahman, F. Timbers, H. Ivison, J. Morrison, J. Poznanski, K. Lo, L. Soldaini, M. Jordan, M. Chen, M. Noukhovitch, N. Lambert, P. Walsh, P. Dasigi, R. Berry, S. Malik, S. Shah, S. Geng, S. Arora, S. Gupta, T. Anderson, T. Xiao, T. Murray, T. Romero, V. Graf, A. Asai, A. Bhagia, A. Wettig, A. Liu, A. Rangapur, C. Anastasiades, C. Huang, D. Schwenk, H. Trivedi, I. Magnusson, J. Lochner, J. Liu, L. J. V. Miranda, M. Sap, M. Morgan, M. Schmitz, M. Guerquin, M. Wilson, R. Huff, R. L. Bras, R. Xin, R. Shao, S. Skjonsberg, S. Z. Shen, S. S. Li, T. Wilde, V. Pyatkin, W. Merrill, Y. Chang, Y. Gu, Z. Zeng, A. Sabharwal, L. Zettlemoyer, P. W. Koh, A. Farhadi, N. A. Smith, and H. Hajishirzi (2025)
  Olmo 3.
  External Links: 2512.13961,
  [Link](https://arxiv.org/abs/2512.13961)
  Cited by: [§3.2](#S3.SS2.p1.1 "3.2 Models and Baselines ‣ 3 Experiment ‣ Experiential Reinforcement Learning").
* OpenAI, :, A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney, A. Iftimie, A. Karpenko, A. T. Passos, A. Neitz, A. Prokofiev, A. Wei, A. Tam, A. Bennett, A. Kumar, A. Saraiva, A. Vallone, A. Duberstein, A. Kondrich, A. Mishchenko, A. Applebaum, A. Jiang, A. Nair, B. Zoph, B. Ghorbani, B. Rossen, B. Sokolowsky, B. Barak, B. McGrew, B. Minaiev, B. Hao, B. Baker, B. Houghton, B. McKinzie, B. Eastman, C. Lugaresi, C. Bassin, C. Hudson, C. M. Li, C. de Bourcy, C. Voss, C. Shen, C. Zhang, C. Koch, C. Orsinger, C. Hesse, C. Fischer, C. Chan, D. Roberts, D. Kappler, D. Levy, D. Selsam, D. Dohan, D. Farhi, D. Mely, D. Robinson, D. Tsipras, D. Li, D. Oprica, E. Freeman, E. Zhang, E. Wong, E. Proehl, E. Cheung, E. Mitchell, E. Wallace, E. Ritter, E. Mays, F. Wang, F. P. Such, F. Raso, F. Leoni, F. Tsimpourlas, F. Song, F. von Lohmann, F. Sulit, G. Salmon, G. Parascandolo, G. Chabot, G. Zhao, G. Brockman, G. Leclerc, H. Salman, H. Bao, H. Sheng, H. Andrin, H. Bagherinezhad, H. Ren, H. Lightman, H. W. Chung, I. Kivlichan, I. O’Connell, I. Osband, I. C. Gilaberte, I. Akkaya, I. Kostrikov, I. Sutskever, I. Kofman, J. Pachocki, J. Lennon, J. Wei, J. Harb, J. Twore, J. Feng, J. Yu, J. Weng, J. Tang, J. Yu, J. Q. Candela, J. Palermo, J. Parish, J. Heidecke, J. Hallman, J. Rizzo, J. Gordon, J. Uesato, J. Ward, J. Huizinga, J. Wang, K. Chen, K. Xiao, K. Singhal, K. Nguyen, K. Cobbe, K. Shi, K. Wood, K. Rimbach, K. Gu-Lemberg, K. Liu, K. Lu, K. Stone, K. Yu, L. Ahmad, L. Yang, L. Liu, L. Maksin, L. Ho, L. Fedus, L. Weng, L. Li, L. McCallum, L. Held, L. Kuhn, L. Kondraciuk, L. Kaiser, L. Metz, M. Boyd, M. Trebacz, M. Joglekar, M. Chen, M. Tintor, M. Meyer, M. Jones, M. Kaufer, M. Schwarzer, M. Shah, M. Yatbaz, M. Y. Guan, M. Xu, M. Yan, M. Glaese, M. Chen, M. Lampe, M. Malek, M. Wang, M. Fradin, M. McClay, M. Pavlov, M. Wang, M. Wang, M. Murati, M. Bavarian, M. Rohaninejad, N. McAleese, N. Chowdhury, N. Chowdhury, N. Ryder, N. Tezak, N. Brown, O. Nachum, O. Boiko, O. Murk, O. Watkins, P. Chao, P. Ashbourne, P. Izmailov, P. Zhokhov, R. Dias, R. Arora, R. Lin, R. G. Lopes, R. Gaon, R. Miyara, R. Leike, R. Hwang, R. Garg, R. Brown, R. James, R. Shu, R. Cheu, R. Greene, S. Jain, S. Altman, S. Toizer, S. Toyer, S. Miserendino, S. Agarwal, S. Hernandez, S. Baker, S. McKinney, S. Yan, S. Zhao, S. Hu, S. Santurkar, S. R. Chaudhuri, S. Zhang, S. Fu, S. Papay, S. Lin, S. Balaji, S. Sanjeev, S. Sidor, T. Broda, A. Clark, T. Wang, T. Gordon, T. Sanders, T. Patwardhan, T. Sottiaux, T. Degry, T. Dimson, T. Zheng, T. Garipov, T. Stasi, T. Bansal, T. Creech, T. Peterson, T. Eloundou, V. Qi, V. Kosaraju, V. Monaco, V. Pong, V. Fomenko, W. Zheng, W. Zhou, W. McCabe, W. Zaremba, Y. Dubois, Y. Lu, Y. Chen, Y. Cha, Y. Bai, Y. He, Y. Zhang, Y. Wang, Z. Shao, and Z. Li (2024)
  OpenAI o1 system card.
  External Links: 2412.16720,
  [Link](https://arxiv.org/abs/2412.16720)
  Cited by: [§5](#S5.SS0.SSS0.Px1.p1.1 "Reinforcement Learning for LLMs. ‣ 5 Related Work ‣ Experiential Reinforcement Learning").
* L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe (2022)
  Training language models to follow instructions with human feedback.
  External Links: 2203.02155,
  [Link](https://arxiv.org/abs/2203.02155)
  Cited by: [§5](#S5.SS0.SSS0.Px1.p1.1 "Reinforcement Learning for LLMs. ‣ 5 Related Work ‣ Experiential Reinforcement Learning").
* Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo (2024)
  DeepSeekMath: pushing the limits of mathematical reasoning in open language models.
  External Links: 2402.03300,
  [Link](https://arxiv.org/abs/2402.03300)
  Cited by: [Appendix C](#A3.p1.1 "Appendix C Training Configuration Details ‣ Experiential Reinforcement Learning"),
  [§3.2](#S3.SS2.p1.1 "3.2 Models and Baselines ‣ 3 Experiment ‣ Experiential Reinforcement Learning").
* T. Shi, K. Chen, and J. Zhao (2024)
  Safer-instruct: aligning language models with automated preference data.
  External Links: 2311.08685,
  [Link](https://arxiv.org/abs/2311.08685)
  Cited by: [§5](#S5.SS0.SSS0.Px1.p1.1 "Reinforcement Learning for LLMs. ‣ 5 Related Work ‣ Experiential Reinforcement Learning").
* T. Shi, Z. Wang, L. Yang, Y. Lin, Z. He, M. Wan, P. Zhou, S. Jauhar, S. Chen, S. Xia, H. Zhang, J. Zhao, X. Xu, X. Song, and J. Neville (2025)
  WildFeedback: aligning llms with in-situ user interactions and feedback.
  External Links: 2408.15549,
  [Link](https://arxiv.org/abs/2408.15549)
  Cited by: [§5](#S5.SS0.SSS0.Px1.p1.1 "Reinforcement Learning for LLMs. ‣ 5 Related Work ‣ Experiential Reinforcement Learning").
* T. Shi, Y. Wu, L. Song, T. Zhou, and J. Zhao (2026)
  Efficient reinforcement finetuning via adaptive curriculum learning.
  External Links: 2504.05520,
  [Link](https://arxiv.org/abs/2504.05520)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Experiential Reinforcement Learning"),
  [§5](#S5.SS0.SSS0.Px1.p1.1 "Reinforcement Learning for LLMs. ‣ 5 Related Work ‣ Experiential Reinforcement Learning").
* N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan, and S. Yao (2023)
  Reflexion: language agents with verbal reinforcement learning.
  External Links: 2303.11366,
  [Link](https://arxiv.org/abs/2303.11366)
  Cited by: [§5](#S5.SS0.SSS0.Px2.p1.1 "Learning from Experience. ‣ 5 Related Work ‣ Experiential Reinforcement Learning").
* D. Silver and R. S. Sutton (2025)
  Welcome to the era of experience.
  External Links: [Link](https://api.semanticscholar.org/CorpusID:277919528)
  Cited by: [§5](#S5.SS0.SSS0.Px2.p1.1 "Learning from Experience. ‣ 5 Related Work ‣ Experiential Reinforcement Learning").
* A. Singh, A. Fry, A. Perelman, A. Tart, A. Ganesh, A. El-Kishky, A. McLaughlin, A. Low, A. Ostrow, A. Ananthram, A. Nathan, A. Luo, A. Helyar, A. Madry, A. Efremov, A. Spyra, A. Baker-Whitcomb, A. Beutel, A. Karpenko, A. Makelov, A. Neitz, A. Wei, A. Barr, A. Kirchmeyer, A. Ivanov, A. Christakis, A. Gillespie, A. Tam, A. Bennett, A. Wan, A. Huang, A. M. Sandjideh, A. Yang, A. Kumar, A. Saraiva, A. Vallone, A. Gheorghe, A. G. Garcia, A. Braunstein, A. Liu, A. Schmidt, A. Mereskin, A. Mishchenko, A. Applebaum, A. Rogerson, A. Rajan, A. Wei, A. Kotha, A. Srivastava, A. Agrawal, A. Vijayvergiya, A. Tyra, A. Nair, A. Nayak, B. Eggers, B. Ji, B. Hoover, B. Chen, B. Chen, B. Barak, B. Minaiev, B. Hao, B. Baker, B. Lightcap, B. McKinzie, B. Wang, B. Quinn, B. Fioca, B. Hsu, B. Yang, B. Yu, B. Zhang, B. Brenner, C. R. Zetino, C. Raymond, C. Lugaresi, C. Paz, C. Hudson, C. Whitney, C. Li, C. Chen, C. Cole, C. Voss, C. Ding, C. Shen, C. Huang, C. Colby, C. Hallacy, C. Koch, C. Lu, C. Kaplan, C. Kim, C. Minott-Henriques, C. Frey, C. Yu, C. Czarnecki, C. Reid, C. Wei, C. Decareaux, C. Scheau, C. Zhang, C. Forbes, D. Tang, D. Goldberg, D. Roberts, D. Palmie, D. Kappler, D. Levine, D. Wright, D. Leo, D. Lin, D. Robinson, D. Grabb, D. Chen, D. Lim, D. Salama, D. Bhattacharjee, D. Tsipras, D. Li, D. Yu, D. Strouse, D. Williams, D. Hunn, E. Bayes, E. Arbus, E. Akyurek, E. Y. Le, E. Widmann, E. Yani, E. Proehl, E. Sert, E. Cheung, E. Schwartz, E. Han, E. Jiang, E. Mitchell, E. Sigler, E. Wallace, E. Ritter, E. Kavanaugh, E. Mays, E. Nikishin, F. Li, F. P. Such, F. de Avila Belbute Peres, F. Raso, F. Bekerman, F. Tsimpourlas, F. Chantzis, F. Song, F. Zhang, G. Raila, G. McGrath, G. Briggs, G. Yang, G. Parascandolo, G. Chabot, G. Kim, G. Zhao, G. Valiant, G. Leclerc, H. Salman, H. Wang, H. Sheng, H. Jiang, H. Wang, H. Jin, H. Sikchi, H. Schmidt, H. Aspegren, H. Chen, H. Qiu, H. Lightman, I. Covert, I. Kivlichan, I. Silber, I. Sohl, I. Hammoud, I. Clavera, I. Lan, I. Akkaya, I. Kostrikov, I. Kofman, I. Etinger, I. Singal, J. Hehir, J. Huh, J. Pan, J. Wilczynski, J. Pachocki, J. Lee, J. Quinn, J. Kiros, J. Kalra, J. Samaroo, J. Wang, J. Wolfe, J. Chen, J. Wang, J. Harb, J. Han, J. Wang, J. Zhao, J. Chen, J. Yang, J. Tworek, J. Chand, J. Landon, J. Liang, J. Lin, J. Liu, J. Wang, J. Tang, J. Yin, J. Jang, J. Morris, J. Flynn, J. Ferstad, J. Heidecke, J. Fishbein, J. Hallman, J. Grant, J. Chien, J. Gordon, J. Park, J. Liss, J. Kraaijeveld, J. Guay, J. Mo, J. Lawson, J. McGrath, J. Vendrow, J. Jiao, J. Lee, J. Steele, J. Wang, J. Mao, K. Chen, K. Hayashi, K. Xiao, K. Salahi, K. Wu, K. Sekhri, K. Sharma, K. Singhal, K. Li, K. Nguyen, K. Gu-Lemberg, K. King, K. Liu, K. Stone, K. Yu, K. Ying, K. Georgiev, K. Lim, K. Tirumala, K. Miller, L. Ahmad, L. Lv, L. Clare, L. Fauconnet, L. Itow, L. Yang, L. Romaniuk, L. Anise, L. Byron, L. Pathak, L. Maksin, L. Lo, L. Ho, L. Jing, L. Wu, L. Xiong, L. Mamitsuka, L. Yang, L. McCallum, L. Held, L. Bourgeois, L. Engstrom, L. Kuhn, L. Feuvrier, L. Zhang, L. Switzer, L. Kondraciuk, L. Kaiser, M. Joglekar, M. Singh, M. Shah, M. Stratta, M. Williams, M. Chen, M. Sun, M. Cayton, M. Li, M. Zhang, M. Aljubeh, M. Nichols, M. Haines, M. Schwarzer, M. Gupta, M. Shah, M. Huang, M. Dong, M. Wang, M. Glaese, M. Carroll, M. Lampe, M. Malek, M. Sharman, M. Zhang, M. Wang, M. Pokrass, M. Florian, M. Pavlov, M. Wang, M. Chen, M. Wang, M. Feng, M. Bavarian, M. Lin, M. Abdool, M. Rohaninejad, N. Soto, N. Staudacher, N. LaFontaine, N. Marwell, N. Liu, N. Preston, N. Turley, N. Ansman, N. Blades, N. Pancha, N. Mikhaylin, N. Felix, N. Handa, N. Rai, N. Keskar, N. Brown, O. Nachum, O. Boiko, O. Murk, O. Watkins, O. Gleeson, P. Mishkin, P. Lesiewicz, P. Baltescu, P. Belov, P. Zhokhov, P. Pronin, P. Guo, P. Thacker, Q. Liu, Q. Yuan, Q. Liu, R. Dias, R. Puckett, R. Arora, R. T. Mullapudi, R. Gaon, R. Miyara, R. Song, R. Aggarwal, R. Marsan, R. Yemiru, R. Xiong, R. Kshirsagar, R. Nuttall, R. Tsiupa, R. Eldan, R. Wang, R. James, R. Ziv, R. Shu, R. Nigmatullin, S. Jain, S. Talaie, S. Altman, S. Arnesen, S. Toizer, S. Toyer, S. Miserendino, S. Agarwal, S. Yoo, S. Heon, S. Ethersmith, S. Grove, S. Taylor, S. Bubeck, S. Banesiu, S. Amdo, S. Zhao, S. Wu, S. Santurkar, S. Zhao, S. R. Chaudhuri, S. Krishnaswamy, Shuaiqi, Xia, S. Cheng, S. Anadkat, S. P. Fishman, S. Tobin, S. Fu, S. Jain, S. Mei, S. Egoian, S. Kim, S. Golden, S. Mah, S. Lin, S. Imm, S. Sharpe, S. Yadlowsky, S. Choudhry, S. Eum, S. Sanjeev, T. Khan, T. Stramer, T. Wang, T. Xin, T. Gogineni, T. Christianson, T. Sanders, T. Patwardhan, T. Degry, T. Shadwell, T. Fu, T. Gao, T. Garipov, T. Sriskandarajah, T. Sherbakov, T. Kaftan, T. Hiratsuka, T. Wang, T. Song, T. Zhao, T. Peterson, V. Kharitonov, V. Chernova, V. Kosaraju, V. Kuo, V. Pong, V. Verma, V. Petrov, W. Jiang, W. Zhang, W. Zhou, W. Xie, W. Zhan, W. McCabe, W. DePue, W. Ellsworth, W. Bain, W. Thompson, X. Chen, X. Qi, X. Xiang, X. Shi, Y. Dubois, Y. Yu, Y. Khakbaz, Y. Wu, Y. Qian, Y. T. Lee, Y. Chen, Y. Zhang, Y. Xiong, Y. Tian, Y. Cha, Y. Bai, Y. Yang, Y. Yuan, Y. Li, Y. Zhang, Y. Yang, Y. Jin, Y. Jiang, Y. Wang, Y. Wang, Y. Liu, Z. Stubenvoll, Z. Dou, Z. Wu, and Z. Wang (2025)
  OpenAI gpt-5 system card.
  External Links: 2601.03267,
  [Link](https://arxiv.org/abs/2601.03267)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Experiential Reinforcement Learning").
* L. Song, Y. Dai, V. Prabhu, J. Zhang, T. Shi, L. Li, J. Li, S. Savarese, Z. Chen, J. Zhao, R. Xu, and C. Xiong (2025a)
  CoAct-1: computer-using agents with coding as actions.
  External Links: 2508.03923,
  [Link](https://arxiv.org/abs/2508.03923)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Experiential Reinforcement Learning").
* L. Song, T. Shi, and J. Zhao (2025b)
  The hallucination tax of reinforcement finetuning.
  External Links: 2505.13988,
  [Link](https://arxiv.org/abs/2505.13988)
  Cited by: [§5](#S5.SS0.SSS0.Px1.p1.1 "Reinforcement Learning for LLMs. ‣ 5 Related Work ‣ Experiential Reinforcement Learning").
* Y. Song, L. Chen, F. Tajwar, R. Munos, D. Pathak, J. A. Bagnell, A. Singh, and A. Zanette (2026)
  Expanding the capabilities of reinforcement learning via text feedback.
  External Links: 2602.02482,
  [Link](https://arxiv.org/abs/2602.02482)
  Cited by: [Appendix A](#A1.SS0.SSS0.Px3.p1.6 "On-Policy Distillation. ‣ Appendix A Full Algorithm and Gated Reflection ‣ Experiential Reinforcement Learning"),
  [§5](#S5.SS0.SSS0.Px2.p1.1 "Learning from Experience. ‣ 5 Related Work ‣ Experiential Reinforcement Learning").
* S. Tan, M. Luo, C. Cai, T. Venkat, K. Montgomery, A. Hao, T. Wu, A. Balyan, M. Roongta, C. Wang, L. E. Li, R. A. Popa, and I. Stoica (2025)
  RLLM: a framework for post-training language agents.
  Note: <https://pretty-radio-b75.notion.site/rLLM-A-Framework-for-Post-Training-Language-Agents-21b81902c146819db63cd98a54ba5f31>Notion Blog
  Cited by: [Appendix C](#A3.p1.1 "Appendix C Training Configuration Details ‣ Experiential Reinforcement Learning").
* S. Wang, Y. Wu, and Z. Xu (2025)
  Cogito, ergo ludo: an agent that learns to play by reasoning and planning.
  External Links: 2509.25052,
  [Link](https://arxiv.org/abs/2509.25052)
  Cited by: [§B.1](#A2.SS1.p1.3 "B.1 Frozen Lake ‣ Appendix B Envrionment Configuration Details ‣ Experiential Reinforcement Learning"),
  [§B.2](#A2.SS2.p1.3 "B.2 Sokoban ‣ Appendix B Envrionment Configuration Details ‣ Experiential Reinforcement Learning"),
  [§3.1](#S3.SS1.p2.1 "3.1 Task ‣ 3 Experiment ‣ Experiential Reinforcement Learning").
* A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, C. Zheng, D. Liu, F. Zhou, F. Huang, F. Hu, H. Ge, H. Wei, H. Lin, J. Tang, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Zhou, J. Lin, K. Dang, K. Bao, K. Yang, L. Yu, L. Deng, M. Li, M. Xue, M. Li, P. Zhang, P. Wang, Q. Zhu, R. Men, R. Gao, S. Liu, S. Luo, T. Li, T. Tang, W. Yin, X. Ren, X. Wang, X. Zhang, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Zhang, Y. Wan, Y. Liu, Z. Wang, Z. Cui, Z. Zhang, Z. Zhou, and Z. Qiu (2025)
  Qwen3 technical report.
  External Links: 2505.09388,
  [Link](https://arxiv.org/abs/2505.09388)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Experiential Reinforcement Learning"),
  [§3.2](#S3.SS2.p1.1 "3.2 Models and Baselines ‣ 3 Experiment ‣ Experiential Reinforcement Learning").
* Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. Cohen, R. Salakhutdinov, and C. D. Manning (2018)
  HotpotQA: a dataset for diverse, explainable multi-hop question answering.
  In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, E. Riloff, D. Chiang, J. Hockenmaier, and J. Tsujii (Eds.),
  Brussels, Belgium,  pp. 2369–2380.
  External Links: [Link](https://aclanthology.org/D18-1259/),
  [Document](https://dx.doi.org/10.18653/v1/D18-1259)
  Cited by: [§3.1](#S3.SS1.p1.1 "3.1 Task ‣ 3 Experiment ‣ Experiential Reinforcement Learning").
* S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao (2023)
  ReAct: synergizing reasoning and acting in language models.
  External Links: 2210.03629,
  [Link](https://arxiv.org/abs/2210.03629)
  Cited by: [§5](#S5.SS0.SSS0.Px1.p1.1 "Reinforcement Learning for LLMs. ‣ 5 Related Work ‣ Experiential Reinforcement Learning").
* E. Zelikman, Y. Wu, J. Mu, and N. Goodman (2022)
  STar: bootstrapping reasoning with reasoning.
  In Advances in Neural Information Processing Systems, A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho (Eds.),
  External Links: [Link](https://openreview.net/forum?id=_3ELRdg2sgI)
  Cited by: [§5](#S5.SS0.SSS0.Px2.p1.1 "Learning from Experience. ‣ 5 Related Work ‣ Experiential Reinforcement Learning").
* K. Zhang, X. Chen, B. Liu, T. Xue, Z. Liao, Z. Liu, X. Wang, Y. Ning, Z. Chen, X. Fu, J. Xie, Y. Sun, B. Gou, Q. Qi, Z. Meng, J. Yang, N. Zhang, X. Li, A. Shah, D. Huynh, H. Li, Z. Yang, S. Cao, L. Jang, S. Zhou, J. Zhu, H. Sun, J. Weston, Y. Su, and Y. Wu (2025)
  Agent learning via early experience.
  External Links: 2510.08558,
  [Link](https://arxiv.org/abs/2510.08558)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Experiential Reinforcement Learning"),
  [§5](#S5.SS0.SSS0.Px2.p1.1 "Learning from Experience. ‣ 5 Related Work ‣ Experiential Reinforcement Learning").

## Appendix A Full Algorithm and Gated Reflection

#### Gated Reflection.

Algorithm [2](#alg2 "Algorithm 2 ‣ On-Policy Distillation. ‣ Appendix A Full Algorithm and Gated Reflection ‣ Experiential Reinforcement Learning") presents the full version of ERL used in our experiments. Compared to the simplified version in Algorithm [1](#alg1 "Algorithm 1 ‣ 2 Experiential Reinforcement Learning (ERL) ‣ Experiential Reinforcement Learning"), the key difference is a gating mechanism on the second attempt: reflection and refinement are triggered only when the first-attempt reward satisfies r(1)<τr^{(1)}<\tau, where τ=1\tau=1. In other words, reflection is applied only to failed or suboptimal trajectories.
In early experiments, we applied reflection to all trajectories, including successful ones, but this led to unstable training and reduced generalization. First, reflecting on already successful attempts encouraged reward hacking: the model sometimes generated instance-specific shortcuts that guaranteed success for the current sample but did not generalize to future episodes. Second, early in training when first-attempt rewards are typically low, the optimization signal became dominated by the second attempt and reflection, which are inherently off-policy relative to the base policy. This imbalance weakened the on-policy learning signal and destabilized the policy. The gating mechanism mitigates these issues by ensuring that successful trajectories remain purely on-policy, while reflection is reserved for corrective revision on failed attempts. This design also aligns training with deployment: at inference time, the model must generate y∼πθ(⋅∣x)y\sim\pi\_{\theta}(\cdot\mid x) without access to reflection Δ\Delta or feedback signals. By restricting reflection to corrective cases and preserving sufficient on-policy updates in every batch, the gating mechanism improves stability in training.

#### Memory Extensions.

Algorithm [2](#alg2 "Algorithm 2 ‣ On-Policy Distillation. ‣ Appendix A Full Algorithm and Gated Reflection ‣ Experiential Reinforcement Learning") also maintains a simple reflection memory that stores successful reflections as system prompt in plain text. A natural extension is to replace this mechanism with a more sophisticated agentic memory system. For example, before the reflection step (Alg. [2](#alg2 "Algorithm 2 ‣ On-Policy Distillation. ‣ Appendix A Full Algorithm and Gated Reflection ‣ Experiential Reinforcement Learning"), Line 12), the model may retrieve relevant past reflections from a memory base conditioned on the current input xx, and after a successful refinement, update the memory using a structured agentic memory update rule rather than direct overwrite. Such retrieval-and-update schemes would allow ERL to scale to more diverse and long-horizon tasks by enabling selective reuse and continual refinement of past corrective knowledge.

#### On-Policy Distillation.

The internalization step in Algorithm [2](#alg2 "Algorithm 2 ‣ On-Policy Distillation. ‣ Appendix A Full Algorithm and Gated Reflection ‣ Experiential Reinforcement Learning") can also be generalized beyond supervised distillation. Instead of training πθ\pi\_{\theta} to reproduce y(2)y^{(2)} from xx using a standard distillation loss, one may adopt an on-policy reverse KL objective. Let the contextual policy with access to reflection and memory be πθ(⋅∣x,Δ)\pi\_{\theta}(\cdot\mid x,\Delta), and the deployment policy be πθ(⋅∣x)\pi\_{\theta}(\cdot\mid x). An on-policy distillation objective can be written as

|  |  |  |
| --- | --- | --- |
|  | ℒOD(θ):=𝔼x∼𝒟[𝕀(r(2)>0)𝔼y∼πθ(⋅∣x)[KL(πθ(⋅∣x,Δ)∥πθ(⋅∣x))]].\mathcal{L}\_{\text{OD}}(\theta):=\mathbb{E}\_{x\sim\mathcal{D}}\Big[\mathbb{I}\!\left(r^{(2)}>0\right)\,\mathbb{E}\_{y\sim\pi\_{\theta}(\cdot\mid x)}\big[\mathrm{KL}\!\left(\pi\_{\theta}(\cdot\mid x,\Delta)\;\|\;\pi\_{\theta}(\cdot\mid x)\right)\big]\Big]. |  |

which encourages the deployment policy to match the richer contextual policy while remaining on-policy with respect to πθ(⋅∣x)\pi\_{\theta}(\cdot\mid x). This connects ERL to recent reverse-KL and on-policy distillation approaches (Agarwal et al., [2024](#bib.bib19 "On-policy distillation of language models: learning from self-generated mistakes"); hübotter2026reinforcementlearningselfdistillation; Song et al., [2026](#bib.bib37 "Expanding the capabilities of reinforcement learning via text feedback")) and provides a principled alternative to supervised internalization.

Algorithm 2  Reinforcement Learning from Self-Reflection (Full Version)

1: Inputs: Language model πθ\pi\_{\theta}; dataset of questions xx;
environment returning feedback ff and reward rr, reward threshold τ\tau.

2: Initialize: reflection memory m←∅m\leftarrow\emptyset.

3: repeat

4:  Sample question xx from dataset.

5:  // First attempt

6:  Sample answer y(1)∼πθ(⋅∣x)y^{(1)}\sim\pi\_{\theta}(\cdot\mid x).

7:  Obtain feedback and reward (f(1),r(1))(f^{(1)},r^{(1)}).

8:  // RL update on the first attempt

9:  Update θ\theta via ℒpolicy​(θ)\mathcal{L}\_{\text{policy}}(\theta) over the first attempt

10:  // Gated second attempt

11:  if r(1)<τr^{(1)}<\tau then

12:   // Reflection with cross-episode memory

13:   Sample reflection Δ∼πθ(⋅∣x,y(1),f(1),r(1),m).\Delta\sim\pi\_{\theta}(\cdot\mid x,y^{(1)},f^{(1)},r^{(1)},m).

14:   Sample refined answer y(2)∼πθ(⋅∣x,Δ).y^{(2)}\sim\pi\_{\theta}(\cdot\mid x,\Delta).

15:   Obtain feedback and reward (f(2),r(2))(f^{(2)},r^{(2)}).

16:   Set reflection reward r~←r(2)\tilde{r}\leftarrow r^{(2)}.

17:   // Store reflection only if improved beyond threshold

18:   if r(2)>τr^{(2)}>\tau then

19:    Store reflection:
m←Δm\leftarrow\Delta.

20:   end if

21:   // RL update on the second attempt

22:   Update θ\theta via
ℒpolicy​(θ)\mathcal{L}\_{\text{policy}}(\theta) over reflection and second attempt.

23:   // Internalization

24:   Update θ\theta via ℒdistill​(θ)\mathcal{L}\_{\text{distill}}(\theta)
to internalize reflection, training πθ\pi\_{\theta}
to produce y(2)y^{(2)} from xx only.

25:  end if

26: until converged

## Appendix B Envrionment Configuration Details

### B.1 Frozen Lake

Frozen Lake is a grid-based navigation environment in which an agent must move from a start location to a goal location on an n×nn\times n grid. We configure our Frozenlake environment following a setup similar to those used in TextArena (Guertler et al., [2025](#bib.bib33 "TextArena")) and Wang et al. ([2025](#bib.bib16 "Cogito, ergo ludo: an agent that learns to play by reasoning and planning")). The grid size nn is sampled uniformly from [2,9][2,9]. For each instance, the start and goal tiles are randomly selected as distinct positions. The grid layout is generated procedurally to ensure that at least one valid path exists between the start and goal.

Each non-goal tile is assigned as either a safe frozen tile or a hole according to a frozen-tile probability parameter pp, sampled uniformly from [0.6,0.85)[0.6,0.85). Holes represent terminal failure states, while frozen tiles are traversable. Transitions are deterministic: the agent’s chosen action directly determines its next grid position, subject to boundary constraints.

At every step, the agent observes a full textual representation of the grid. To reduce the influence of pretrained symbolic priors, we employ abstract symbols rather than semantically meaningful markers. The default encoding is:

|  |  |  |
| --- | --- | --- |
|  | A=agent position,B=goal tile,C=hole,D=safe frozen tile.\texttt{A}=\text{agent position},\quad\texttt{B}=\text{goal tile},\quad\texttt{C}=\text{hole},\quad\texttt{D}=\text{safe frozen tile}. |  |

This representation encourages the model to infer environment dynamics through interaction rather than relying on prior associations.

In addition to the textual presentation of the grid, the environment also appends structured textual feedback to the end of the interaction history after each action. This feedback communicates the outcome of the most recent transition and serves as the only explicit signal describing terminal or invalid events. The feedback messages are defined as follows:

* •

  The agent reached the goal — issued when the agent successfully enters the goal tile. The episode terminates with reward 1.01.0.
* •

  The agent fell into the hole — issued when the agent enters a hole tile. The episode terminates with reward 0.00.0.
* •

  Hit the max step limit — issued when the agent exhausts the fixed step budget. The episode terminates with reward 0.00.0.
* •

  No valid actions were recorded. — issued when the agent produces an invalid action or when the attempted action results in no state change, such as moving into a boundary. The episode continues unless the step budget is exhausted.

The default system prompt, self-reflection prompt, and example task are shown in Tables [2](#A2.T2 "Table 2 ‣ B.3 HotpotQA ‣ Appendix B Envrionment Configuration Details ‣ Experiential Reinforcement Learning"), [4](#A2.T4 "Table 4 ‣ B.3 HotpotQA ‣ Appendix B Envrionment Configuration Details ‣ Experiential Reinforcement Learning"), and [6](#A2.T6 "Table 6 ‣ B.3 HotpotQA ‣ Appendix B Envrionment Configuration Details ‣ Experiential Reinforcement Learning").

The reward function is sparse. The agent receives a reward of 1.01.0 if it reaches the goal tile and 0.00.0 otherwise. Episodes terminate upon reaching the goal, entering a hole, or exhausting a fixed step budget of 8 actions.

For training, we generate 10,000 procedurally sampled instances. Evaluation is conducted on a disjoint set of 100 instances constructed using the same generation process.

### B.2 Sokoban

Sokoban is a grid-based box-pushing environment in which an agent must place all boxes onto designated goal tiles. We configure our Sokoban environment following a setup similar to those used in TextArena (Guertler et al., [2025](#bib.bib33 "TextArena")) and Wang et al. ([2025](#bib.bib16 "Cogito, ergo ludo: an agent that learns to play by reasoning and planning")). Each instance is represented as an n×nn\times n grid, where nn is sampled uniformly from [6,8][6,8] in our procedural generator. We construct single-box, single-goal layouts with border walls, and randomly sample interior positions for the goal, box, and player, subject to non-overlap constraints.

To control difficulty, each generated layout is accepted only if its shortest valid solution is at most 8 moves (computed by BFS over player–box states). This guarantees solvability while keeping episodes short-horizon. Train and test splits are disjoint at the layout level.

At every step, the agent observes the full textual grid. As in FrozenLake, we use abstract symbols to reduce direct reliance on pretrained semantic priors. The default encoding is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | A | =agent position,a=agent on box,B=box,b=box on goal,\displaystyle=\text{agent position},\quad\texttt{a}=\text{agent on box},\quad\texttt{B}=\text{box},\quad\texttt{b}=\text{box on goal}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | C | =goal tile,E=wall,D=floor.\displaystyle=\text{goal tile},\quad\texttt{E}=\text{wall},\quad\texttt{D}=\text{floor}. |  |

The action space is {Up, Down, Left, Right}. Moves are deterministic. The agent may push exactly one adjacent box only when the cell behind the box is free; it cannot pull boxes, move through walls, or move through boxes. Invalid moves produce no state change.

In addition to the grid observation, the interaction trace includes structured textual transition feedback after each action. The feedback messages are:

* •

  The agent solved the puzzle (all boxes on goals). — issued when all boxes are on goal tiles. The episode terminates with reward 1.01.0.
* •

  The agent moved or pushed a box; puzzle not solved yet. — issued when the action changes the state but the puzzle remains unsolved.
* •

  The agent did not move (likely hit a wall or tried to push into a blocked space). — issued when the chosen move is ineffective (no state change).
* •

  Hit the max step limit — issued when the fixed step budget is exhausted before solving. The episode terminates with reward 0.00.0.

The default system prompt, self-reflection prompt, and example task are shown in Tables [2](#A2.T2 "Table 2 ‣ B.3 HotpotQA ‣ Appendix B Envrionment Configuration Details ‣ Experiential Reinforcement Learning"), [4](#A2.T4 "Table 4 ‣ B.3 HotpotQA ‣ Appendix B Envrionment Configuration Details ‣ Experiential Reinforcement Learning"), and [7](#A2.T7 "Table 7 ‣ B.3 HotpotQA ‣ Appendix B Envrionment Configuration Details ‣ Experiential Reinforcement Learning").

The reward is sparse: 1.01.0 if and only if all boxes are on goals, and 0.00.0 otherwise. Episodes terminate on success or when the step budget is exhausted. In the generated REEX Sokoban dataset, the per-instance step budget is 8.

For training, we generate 10,000 procedurally sampled instances. Evaluation is conducted on a disjoint set of 100 instances built with the same generation process.

### B.3 HotpotQA

HotpotQA is a multi-hop open-domain question answering task in which an agent must answer compositional questions by retrieving and synthesizing evidence across multiple documents. Each instance consists of a natural-language question and a reference answer.

Unlike grid-based control environments such as FrozenLake or Sokoban, HotpotQA does not expose an explicit environment state. Instead, the agent operates through a tool-augmented interaction loop in which it alternates between reasoning, retrieval, and answer generation. The agent may invoke an external retrieval tool and ultimately produce a final textual answer. The solver instruction requires that the final answer be formatted inside \boxed{} to enable reliable extraction.

The retrieval interface is defined as:

|  |  |  |
| --- | --- | --- |
|  | local\_search(query, top\_k),\texttt{local\\_search(query, top\\_k)}, |  |

which queries a local dense-retrieval server built over an indexed Wikipedia corpus and returns ranked text snippets relevant to the query. We use a Wikipedia corpus organized by PeterJinGo/wiki-18-corpus, with prebuilt dense indices from PeterJinGo/wiki-18-e5-index. Embeddings are generated using intfloat/e5-base-v2. Retrieval is powered by FAISS (Douze et al., [2024](#bib.bib35 "The faiss library")) with multi-GPU support. During each episode, the agent is allowed up to 5 interaction turns, which may include reasoning steps, tool calls, and final answer submission.

Following the evaluation protocol of Search-R1 (Jin et al., [2025](#bib.bib9 "Search-r1: training LLMs to reason and leverage search engines with reinforcement learning")), the answer extracted from \boxed{} is normalized prior to scoring by lowercasing and whitespace canonicalization. Correctness is measured using token-level F1 against the ground-truth answer. The reward function assigns a score of 1.0 for exact matches, a proportional reward equal to the F1 score when the F1 is at least 0.3, and 0 otherwise.

The default system prompt, self-reflection prompt, and example task are shown in Tables [3](#A2.T3 "Table 3 ‣ B.3 HotpotQA ‣ Appendix B Envrionment Configuration Details ‣ Experiential Reinforcement Learning"), [5](#A2.T5 "Table 5 ‣ B.3 HotpotQA ‣ Appendix B Envrionment Configuration Details ‣ Experiential Reinforcement Learning"), and [8](#A2.T8 "Table 8 ‣ B.3 HotpotQA ‣ Appendix B Envrionment Configuration Details ‣ Experiential Reinforcement Learning").

Initial System Prompt for Frozenlake and Sokoban

You are an agent playing a game on a grid, acting as a reasoning engine.
Your decisions are based on your current game rules (your best guess of how the game works) and your strategic playbook (your learned strategies). These may be incomplete or incorrect.
Your only way to interact with the environment is by choosing your NEXT ACTION.
## Instructions
1. Analyze State: Summarize the current state.
2. Predict Long-term Value of Outcomes: Evaluate the strategic value and potential of the current state for the future.
3. Predict Immediate Consequences: For the top two candidate actions, predict their consequences using a “result-because” structure.
4. Select the Best Action: Choose the action leading to the most advantageous future state.
## Required response structure
<reason>
  
\*\*1. Analysis of the Current State:\*\* [Summary of the board state.]
  
\*\*2. Prediction of the Value of Current States:\*\* [Assessment] - Value: High / Medium / Low
  
\*\*3. Prediction of Immediate Consequences:\*\* [Top 2 candidate actions]
  
</reason>
  
Then output the NEXT ACTION inside triple backticks, e.g., ‘‘‘Up‘‘‘.
Always remember:
- Valid actions: Up, Down, Left, Right.
- Think step by step, but make the final line only the next action wrapped in triple backticks.


Table 2: Initial System used for Frozenlake and Sokoban.



Initial System Prompt for HotpotQA

You are a helpful assistant who answers questions directly and efficiently.
Provide your final answer in \boxed{} format.
## Available tool

```
[
  {
    "type": "function",
    "function": {
      "name": "local_search",
      "description": "Search for information using a dense retrieval
                      server with Wikipedia corpus",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "Search query to retrieve relevant documents"
          },
          "top_k": {
            "type": "integer",
            "description": "Number of results to return (default: 5)",
            "minimum": 1,
            "maximum": 50
          }
        },
        "required": ["query"]
      }
    }
  }
]
```



Table 3: Initial system prompt used for HotpotQA.



Self-reflection Prompt for Frozen Lake and Sokoban

You are a chief scientific strategist and master tactician.
Your mission is to analyze extensive field data from numerous operations to distill and refine the Master Rulebook of a complex game.
You will be presented with a large collection of highly successful trajectories and critical failure trajectories, collected over a long period.
Your primary task is to perform a deep, comparative analysis to understand the fundamental principles of victory and defeat.
Act as a grand strategist, identifying universal patterns and high-level causal relationships.
Your goal is to synthesize these insights to produce the next generation’s Master Rulebook, making it more robust, accurate, and effective.
Core Principles:
- Think Long-Term: focus on universal, strategic truths that hold across diverse scenarios.
- Learn from Contrast: extract insights by comparing winners and losers.
- Synthesize and Consolidate: produce a single unified theory.
- Be Authoritative and Concise: state rules as definitive principles.
Your output MUST be a single consolidated <prompt> block representing the new Master Rulebook:
<prompt>
  
<game\_rules>
  
\*\*1. Symbol Meanings:\*\* [...]
  
\*\*2. Information & Interpretation:\*\* [...]
  
\*\*3. Gameplay & Actions:\*\* [...]
  
\*\*4. Action Effects:\*\* [...]
  
\*\*5. Game Objective & Termination:\*\* [...]
  
</game\_rules>
  
<strategy>
  
\*\*1. Core Strategies:\*\* [...]
  
\*\*2. Tactical Tips:\*\* [...]
  
</strategy>
  
</prompt>
  


Table 4: Self-reflection prompt used for Frozen Lake and Sokoban.



Self-reflection Prompt for HotpotQA

You are an expert prompt updater.
You will analyze recent trajectories, tool calls, and rewards to improve the solver’s system prompt.
When failures occur, explicitly add rules that prevent repeating them (e.g., missing tool calls, hallucinated facts, or unboxed final answers).
Keep the prompt short, actionable, and reusable.
Output ONLY the improved system prompt wrapped in <prompt>...</prompt> tags.


Table 5: Self-reflection prompt used for HotpotQA.



Example Task for Frozen Lake

## {System Prompt}
Current Observation (0):
  
 D    D    C    D
  
 A    D    D    C
  
 D    C    D    D
  
 D    D    B    D
  
You have not achieved the goal yet. Please give the next action.
## Action space
Up | Down | Left | Right
  
## Output requirement
Return reasoning in <reason>...</reason> and final action in triple backticks, e.g., ‘‘‘Right‘‘‘.


Table 6: Example Frozen Lake task instance.



Example Task for Sokoban

## {System Prompt}
Current Board (0):
  
 E    E    E    E    E    E
  
 E    A    D    B    C    E
  
 E    D    D    D    D    E
  
 E    E    E    E    E    E
  
Puzzle not solved yet. Provide the next move.
## Action space
Up | Down | Left | Right
  
## Output requirement
Return reasoning in <reason>...</reason> and final action in triple backticks, e.g., ‘‘‘Right‘‘‘.


Table 7: Example Sokoban task instance.



Example Task for HotpotQA

## {System Prompt}
Question:
Which university did the author of “The Hobbit” attend?
  


Table 8: Example HotpotQA task instance.



Second-Attempt Prompt Template for the No-Reflection Variant

## {System Prompt}
You are also provided with the model’s past attempt data, including observations, actions, rewards, and feedback.
Use this information as context to make a better next-attempt decision policy.
Follow the action/output format exactly.
{First Attempt’s Trajectory}


Table 9: 
Generic second-attempt system prompt used in the no-reflection ablation. The model is provided with the full first-attempt trajectory (observations, actions, rewards, and feedback) together with a generic instruction encouraging improvement, without any structured reflection signal.

## Appendix C Training Configuration Details

We train all models with the rLLM agent training stack (Tan et al., [2025](#bib.bib27 "RLLM: a framework for post-training language agents")) using GRPO (Shao et al., [2024](#bib.bib28 "DeepSeekMath: pushing the limits of mathematical reasoning in open language models")). Training runs on a single node with 8 H100s and uses vLLM (Kwon et al., [2023](#bib.bib26 "Efficient memory management for large language model serving with pagedattention")) with FlashAttention (Dao, [2024](#bib.bib25 "FlashAttention-2: faster attention with better parallelism and work partitioning")).

We enable hybrid engine training, gradient checkpointing, and remove-padding. The optimizer learning rate is 1e-6. Actor updates use a mini batch size of 64, dynamic batch sizing, and a max token length per GPU of 24,000. FSDP parameter/optimizer offload is enabled for the actor, and parameter offload is enabled for the reference model.

We set the training batch size to 64, with a maximum prompt length of 8,196 tokens and a maximum response length of 8,196 tokens. Rollouts are generated asynchronously using vLLM in async mode with a tensor model parallel size of 1. We use a sampling temperature of 0.7, GPU memory utilization of 0.85. For validation rollouts, we generate 4 samples per prompt with temperature 0.7, top-p sampling set to 0.8, and top-k sampling set to 20.

KL regularization is enabled using a low-variance KL loss with coefficient 0.001, and we use a fixed KL control coefficient of 0.001. The actor clipping ratio upper bound is set to 0.28, and the entropy coefficient is set to 0. Rejection sampling and stepwise advantage estimation are disabled.

For RLVR training, we generate 10 samples per prompt. For ERL training, we generate only 4 samples per prompt for each attempt to match the compute budget of RLVR. Evaluation is performed every 5 iterations, and training is manually early stopped upon convergence.

The design and implementation details of the ERL algorithm can be found in Appendix [A](#A1 "Appendix A Full Algorithm and Gated Reflection ‣ Experiential Reinforcement Learning").

[◄](/html/2602.13948)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2602.13949)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2602.13949)
[View original  
on arXiv](https://arxiv.org/abs/2602.13949)[►](/html/2602.13950)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Thu Mar 5 17:41:14 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
