---
arxiv: '2604.08706'
authors:
- Charles Arnal
- Vivien Cabannes
- Taco Cohen
- Julia Kempe
- Remi Munos
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Efficient RL Training for LLMs with Experience Replay
url: https://arxiv.org/abs/2604.08706
year: 2026
---

# Efficient RL Training for LLMs with Experience Replay

[
Affiliation: [
Affiliation: [
Email: [charlesarnal@meta.com](mailto:charlesarnal@meta.com)

(May 6, 2026)

###### Abstract

While Experience Replay—the practice of storing rollouts and reusing them multiple times during training—is a foundational technique in general RL, it remains largely unexplored in LLM post-training due to the prevailing belief that fresh, on-policy data is essential for high performance.
In this work, we challenge this assumption.
We present a systematic study of replay buffers for LLM post-training, formalizing the optimal design as a trade-off between staleness-induced variance, sample diversity and the high computational cost of generation.
We show that strict on-policy sampling is suboptimal when generation is expensive.
Empirically, we show that a well-designed replay buffer can drastically reduce inference compute without degrading – and in some cases even improving – final model performance, while preserving policy entropy.

## 1 Introduction

Reinforcement Learning (RL) has emerged as the key driver behind the reasoning capabilities of modern Large Language Models (LLMs), enabling breakthroughs in complex tasks such as mathematics and coding (deepseekai2025deepseekr1; openr1\_math220k).
However, this performance comes at a prohibitive computational cost. Unlike pre-training, where data is static, RL requires the continuous generation of new training trajectories. In state-of-the-art pipelines, this inference cost often dominates the training budget, and may consume more than 80% of post-training GPU hours.
Standard approaches exacerbate this issue through extreme sample inefficiency: methods like PPO or GRPO typically operate as on-policy as possible, meaning rollouts are generated, used for a single gradient update, and immediately discarded.

This “generate-then-discard” paradigm stands in stark contrast to classical Reinforcement Learning, where Experience Replay, i.e. storing and reusing past trajectories in a buffer, is a foundational tool for sample efficiency (mnih2015human; lin1992self).
While Experience Replay is standard in sample-limited robotics or gaming environments, it has been largely overlooked in LLM training, where the prevailing consensus suggests that the performance degradation from off-policy data outweighs the computational benefits.

In this work, we challenge this consensus.
We demonstrate that discarding trajectories after a single use is computationally suboptimal.
By incorporating a replay buffer into asynchronous training pipelines, we trade a controlled increase in data off-policiness (staleness) and a decrease in data diversity for a dramatic reduction in inference costs.
We formalize this trade-off through a theoretical analysis of the bias-variance decomposition in stochastic gradient descent, proving that optimal compute efficiency is achieved not by being strictly on-policy, but by balancing the freshness and diversity of data against its generating cost.
Our contributions are as follows:

* •

  Theoretical Analysis: We detail the implementation of replay buffers in asynchronous LLM training and provide a mathematical framework quantifying the trade-off between compute efficiency, sample diversity, and gradient bias. We derive theoretical bounds for the optimal buffer size and replay ratio, showing that as the relative cost of inference increases, the optimal strategy shifts further towards experience replay.
* •

  Empirical Analysis: Through extensive experiments, we provide an in-depth analysis of how buffer hyperparameters influence the training process. We show that while aggressive reuse of samples can degrade performance, a well-sized buffer acts as a regularizer that stabilizes training and preserves model output diversity (improving pass@kk metrics).
* •

  Empirical Gains: We validate those conclusions on larger models and show that simple, easy-to-implement buffer strategies can save up to 40% of the compute budget while maintaining, and sometimes surpassing, the same final accuracy as the on-policy baseline, as shown e.g. in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Efficient RL Training for LLMs with Experience Replay"). We further explore how more sophisticated sampling strategies (e.g., prioritizing positive trajectories) and alternative losses can extend the stability of replay buffers, allowing for even greater efficiency gains.

Through this study, we present a straightforward approach for high-efficiency RL fine-tuning, shifting the focus from maximizing performance per step to *maximizing performance per unit of compute.*

!(/html/2604.08706/assets/x1.png)

Figure 1: Experience Replay improves LLM RL Training.
Accuracy on MATH as a function of compute spent when training Qwen2.5-7B on OpenR1-Math-220k for the no-buffer baseline (orange curve) and a buffer of size 8484 with (W,T)=(5,3)(W,T)=(5,3). We report the median and IQR over 1010 seeds. Compute is calibrated so that a single weight update for the baseline costs 11 unit. Baseline runs display increased instability.

## 2 Related Work

##### Experience Replay in RL.

The use of a replay buffer is a cornerstone of deep RL, famously enabling stability and sample efficiency in algorithms like DQN (mnih2015human), Soft Actor-Critic (haarnoja2018soft), and DDPG (lillicrap2015continuous). Techniques such as Prioritized Experience Replay (schaul2015prioritized) and Hindsight Experience Replay (andrychowicz2017hindsight) further optimized how agents learn from past data.
Despite this rich history, modern LLM reasoning pipelines (deepseekai2025deepseekr1; openr1\_math220k) have largely defaulted to on-policy training (e.g., GRPO, PPO), discarding trajectories immediately after a gradient update to avoid off-policy degradation, though, in practice, implementation constraints typically lead to some unavoidable off-policiness.

##### Replay Buffers for LLMs.

Very recently, several works have re-introduced replay buffers to LLM training, though with different motivations. wang2025eframedeeperreasoningexplorationfilterreplay and bartoldson2025trajectorybalanceasynchronydecoupling utilize buffers primarily to enhance exploration and final model performance, often requiring specialized loss functions or complex filtering. Similarly, lu2025arpoendtoendpolicyoptimizationgui and zhang2025rlepreinforcementlearningexperience propose dynamic sampling or multi-phase training to maximize data quality.
In contrast, our work focuses strictly on *compute efficiency*. We do not propose a new training paradigm to beat state-of-the-art accuracy; rather, we systematically analyze the trade-off between off-policiness and efficiency in standard asynchronous pipelines, demonstrating that simple experience replay can drastically reduce the compute budget while maintaining accuracy.

A more detailed discussion of off-policy algorithms and related theoretical works is provided in Appendix [7](#S7 "7 Extended Related Work ‣ Efficient RL Training for LLMs with Experience Replay").

## 3 Experience Replay for Off-Policy RL

We present how experience replay can be efficiently implemented in an LLM post-training pipeline and discuss the role of various hyperparameters and their impact on compute efficiency.

### 3.1 Reinforcement Learning and Replay Buffers

In modern, compute-efficient RL pipelines for LLMs, the GPUs are often split between WW inference workers and TT trainers (noukhovitch2024asynchronous; gehring2024rlef; wu2025llamarl; bartoldson2025trajectorybalanceasynchronydecoupling; faircodegenteam2025cwmopenweightsllmresearch).
At any given time, each of the two groups maintains its own (possibly stale) copy of the model weights.
The inference workers continuously generate trajectories (also called rollouts) using their set of weights, then pass them to the trainers, usually via a transfer queue.
Concurrently, trainers pull trajectories from the queue, perform forward-backward passes over them and update their weights.
Trajectories are discarded after having been used once (schulman2017proximalpolicyoptimizationalgorithms; shao2024deepseekmath; deepseekai2025deepseekr1).
Every few gradient steps, the inference worker’s weights are updated with the current value of the trainers’ weights.
This setting, which corresponds to our experimental implementation, is sometimes referred to as asynchronous training.
Synchronous setups also exist (vonwerra2022trl; sheng2024hybridflow); we discuss them in Section [4](#S4 "4 Mathematical Analysis ‣ Efficient RL Training for LLMs with Experience Replay").

A replay buffer can be implemented as follows: instead of adding their rollouts to a queue, the inference workers add them to a list of trajectories, the replay buffer.
In parallel, trainers continuously sample from this replay buffer; sampling from the buffer does not remove the sampled trajectories from it.111In our specific implementation, the buffer is sharded across trainers; see Appendix [10](#S10 "10 Experimental details ‣ Efficient RL Training for LLMs with Experience Replay") for details.
This allows for the re-using of samples, which in turn reduces the amount of overall compute needed by amortizing the cost of rollout generation, as detailed further below.
Pseudo-code is provided in Appendix [8](#S8 "8 Pseudo-Code Implementation ‣ Efficient RL Training for LLMs with Experience Replay").

The replay buffer can be sampled by the trainers to assemble their training batches following several strategies that pick samples based on characteristics such as their recency, their associated rewards, the norm of past gradients computed using them, or how many times the rollout has already been sampled. One might also want to define a decay rule for the buffer, e.g. making the buffer a first-in, first-out list by keeping only the NN freshest samples.

!(/html/2604.08706/assets/x2.png)!(/html/2604.08706/assets/x3.png)

Figure 2: Effect of Experimental Design on Off-Policiness and Diversity Statistics.
Top row: Distribution of off-policiness, replay ratio and steps-since-last-use over all samples and uses of samples during a training run for buffer size N∈{84,252,756,2268}N\in\{84,252,756,2268\} and (W,T)=(6,2)(W,T)=(6,2). See also Appendix [10.3](#S10.SS3 "10.3 Metrics ‣ 10 Experimental details ‣ Efficient RL Training for LLMs with Experience Replay") for details on the steps-since-last-use metric.
Bottom row: Same statistics for N=252N=252 and (W,T)∈{(6,2),(5,3),(4,4)}(W,T)\in\{(6,2),(5,3),(4,4)\}.
The average replay ratio is 1.781.78, 3.423.42 and 7.07.0 for (W,T)(W,T) equal to (6,2),(5,3)(6,2),(5,3) and (4,4)(4,4) respectively.

### 3.2 Off-Policiness, Diversity, and Compute Efficiency

The design of the buffer and the ratio W/TW/T of inference workers to trainers directly impact three major aspects of training: the compute efficiency, the degree of off-policiness, and the diversity of the samples.

To illustrate these concepts, we consider throughout this subsection a buffer configuration with T∈{1,2,…,7}T\in\{1,2,\ldots,7\} trainer GPUs, W:=8−TW:=8-T inference worker GPUs, and a first-in, first-out buffer (i.e. that contains the last NN samples generated by the inference workers).
Training samples are drawn uniformly at random from the buffer at each step.
We train Qwen2.5-7B qwen2025qwen25technicalreport model on the OpenR1-Math-220k reasoning dataset openr1\_math220k (see Subsection [5.1](#S5.SS1 "5.1 Experimental setup ‣ 5 Experimental results ‣ Efficient RL Training for LLMs with Experience Replay") for experimental details).

##### Compute Efficiency

The compute spent on an RL training run, which we think of in terms of active GPU seconds222We explain in greater details our simplifying assumptions in Appendix [10](#S10 "10 Experimental details ‣ Efficient RL Training for LLMs with Experience Replay")., can be decomposed roughly as the sum of the trainer compute, spent on forward-backward passes and weight updates, and the inference compute, spent on generating rollouts, i.e.
compute≅trainer compute+inference compute.\text{compute}\cong\text{trainer compute}+\text{inference compute}.
In the asynchronous setting and without a buffer, the ratio W/TW/T of inference worker GPUs to trainer GPUs admits an optimal value μ\mu that minimizes GPU downtime.
Indeed, as a first order approximation, we can assume that the trainer compute CC needed for a step (including forward and backward passes and weight update) depends only on the (fixed) batch size, and not on the number of trainer GPUs.
Let μ>0\mu>0 be the factor such that producing a batch of rollouts of the same size costs C⋅μC\cdot\mu compute for the inference workers.333In our experiments, we find that μ\mu ranges from ∼4\sim\!\!4 to ∼10\sim\!\!10 depending on the model, task and implementation considered. 
The total compute needed for each parameter update is then roughly

|  |  |  |  |
| --- | --- | --- | --- |
|  | compute without buffer≈C​(1+μ).\text{compute without buffer}\approx C(1+\mu). |  | (1) |

In that case, the optimal ratio of inference worker GPUs to trainer GPUs, i.e. the ratio such that trainer GPUs process generated rollouts exactly at the speed at which inference worker GPUs produce them, so that neither have any downtime, is precisely μ\mu: if generating rollouts is μ\mu times more costly than training on them, one needs μ\mu times more inference GPUs than trainer GPUs.

By contrast, *when using a replay buffer, the inference compute is decoupled from the trainer compute*:
inference workers can always continuously add trajectories to the buffer from which the trainers can freely pull, independently from how many inference workers and trainers there are.
As in the case without buffer, each backward pass costs CC trainer compute.
On the other hand, the inference compute spent during a backward pass depends on the number of inference worker GPUs that are concurrently working.
Hence, the total compute spent for each parameter update is roughly equal to

|  |  |  |
| --- | --- | --- |
|  | total compute with buffer≈C​(1+W/T).\text{total compute with buffer}\approx C(1+W/T). |  |

As reflected in this formula, when using a buffer, *increasing the number of trainers relative to the number of inference workers makes each gradient step cheaper*; intuitively, this is simply because rollouts are re-used more times on average, meaning that for a given number of optimization steps, fewer rollouts need to be generated.

We define the compute ratio of a buffer configuration to be

|  |  |  |  |
| --- | --- | --- | --- |
|  | γ:=1+W/T1+μ,\gamma:=\frac{1+W/T}{1+\mu}, |  | (2) |

that is, the ratio of the compute cost of a parameter update with and without a buffer.

| (W, T) | (7,1) | (6,2) | (5,3) | (4,4) | (2,6) | (1,7) |
| --- | --- | --- | --- | --- | --- | --- |
| γ\gamma | 1.291.29 | 0.650.65 | 0.430.43 | 0.320.32 | 0.220.22 | 0.180.18 |

Table 1: γ\gamma for various values of (W,T)(W,T) and an estimated μ=5.28\mu=5.28 for Qwen2.5-7B.

##### Degree of Off-Policiness

The design of the replay buffer and the ratio (W,T)(W,T) directly impact the off-policiness of the training distribution.
We define the off-policiness (or staleness) of a sample used in a gradient update as the difference between the step at which the sample was created and the current step.
The average off-policiness over all samples is influenced by both the size NN of the buffer (the larger the buffer, the greater the average off-policiness of the samples that it contains) and the ratio W/TW/T of inference worker to trainer GPUs: the more trainer GPUs there are, the faster weight updates occur and the faster samples become outdated.
This can be observed on the left of Figure [2](#S3.F2 "Figure 2 ‣ 3.1 Reinforcement Learning and Replay Buffers ‣ 3 Experience Replay for Off-Policy RL ‣ Efficient RL Training for LLMs with Experience Replay"), where the distribution of off-policiness over all the samples used through a training run is represented for various pairs (W,T)(W,T) and buffer sizes NN.

##### Diversity of Samples

The use of a replay buffer may deteriorate training dynamics: as the same samples are reused, the training distribution seen by the policy gradient algorithm becomes less diverse, and less information regarding the true objective function is utilized.
This notion of sample diversity arises at two distinct levels.
First, the global diversity of samples, which we measure using the replay ratio of the samples, defined as the number of times a sample has been used for a gradient step over the entire training run.
The average replay ratio will be chiefly conditioned by the ratio W/TW/T: the more trainer GPUs there are relative to the number of inference worker GPUs, the more passes they will do on average on each data point.
This is illustrated in the middle of Figure [2](#S3.F2 "Figure 2 ‣ 3.1 Reinforcement Learning and Replay Buffers ‣ 3 Experience Replay for Off-Policy RL ‣ Efficient RL Training for LLMs with Experience Replay").

Second, the local diversity of samples which is the degree to which samples are repeatedly used in close succession.
We measure local diversity using the time-since-last-use of the samples in the current trainer batch, i.e. the number of gradient steps since the last gradient update to which they contributed.
We expect a loss in local diversity to be more harmful than a loss in global diversity.
At a fixed ratio W/TW/T, one can trade off-policiness for local diversity: by increasing the size of the buffer, the training distribution’s degree of off-policiness will increase (as discussed earlier), but the empirical training distribution will be locally more diverse: though samples are just as likely to be reused over the entire training run, they are less likely to be reused in close succession (due to the greater number of candidate samples in the buffer). This can be seen on the right side of Figure [2](#S3.F2 "Figure 2 ‣ 3.1 Reinforcement Learning and Replay Buffers ‣ 3 Experience Replay for Off-Policy RL ‣ Efficient RL Training for LLMs with Experience Replay").

##### Goal: Increased Efficiency, Preserved Accuracy

The primary motivation behind the use of a replay buffer is to save inference compute by reusing trajectories.
As explained above, *each gradient step (including the required sampling) can be made computationally cheaper by letting the ratio W/TW/T decrease*.
However, we have also seen that *letting the ratio W/TW/T decrease makes the training distribution more off-policy and less diverse.*
It is usually assumed that high off-policiness and low sample diversity should be avoided (see however tang2025rlfinetuningllmsonoffpolicy; arnal2025asymmetricreinforceoffpolicyreinforcement and charton2024emergentpropertiesrepeatedexamples).
Hence there is a *trade-off*: re-using samples from the buffer makes each gradient step cheaper, but resampling too aggressively might end up hurting the expected accuracy gain from each step.
In our experiments, we explore the efficiency/accuracy optimality curve; in other words, *we want to maximize the accuracy achievable at a given compute cost by selecting the best buffer configuration*.
To ensure our conclusions are readily applicable to production environments, we also deliberately prioritize simple implementations that require only modest departures from current SOTA pipelines.

## 4 Mathematical Analysis

While the previous section and our experiments focus on the more compute-efficient asynchronous RL setting, we choose to conduct our mathematical analysis in the conceptually simpler synchronous setting, in which the training alternates between two clearly distinct modes: a generating phase, in which new trajectories are created, and a training phase, during which a gradient descent step is performed using the new rollouts.
We consider a simple first-in, first-out replay buffer: at each training step tt, we (i) generate RR new rollouts using the current policy and insert them at the beginning of a buffer of capacity NN (evicting the oldest samples), and (ii) sample a minibatch of size BB uniformly from the buffer to form a gradient update

|  |  |  |
| --- | --- | --- |
|  | θt+1=θt−η​gt,gt=1B​∑j=1BG​(θt,zt,ij).\theta\_{t+1}=\theta\_{t}-\eta\,g\_{t},\qquad g\_{t}=\frac{1}{B}\sum\_{j=1}^{B}G(\theta\_{t},z\_{t,i\_{j}}). |  |

Here, θ\theta denotes the policy parameters, zt,iz\_{t,i} denotes the ii-th element of the buffer at step tt, iji\_{j} the jj-th sampled index, and G​(θ,z)G(\theta,z) denotes the corresponding gradient estimate of ∇F​(θ)\nabla F(\theta), where FF is the objective we wish to minimize.
The compute cost of such an update, expressed in arbitrary units, is given by c=B+μ​Rc=B+\mu R, where μ\mu denotes the compute cost ratio between a forward-backward pass and one rollout generation, matching the definition above.

The goal of our theoretical analysis is to characterize how the design of the replay buffer affects learning efficiency from a theoretical standpoint.
We adopt the classical non-convex stochastic optimization framework and study the convergence of the training dynamics toward stationary points, as measured by the decay of the expected squared norm of the gradient.
Unless stated otherwise, all norms are Euclidean.

###### Assumption 4.1 (Target Smoothness).

The function FF is non-negative, differentiable, and LL-smooth, i.e.

|  |  |  |
| --- | --- | --- |
|  | ∀x,y‖∇F​(y)−∇F​(x)‖≤L​‖y−x‖.\forall\,x,y\quad\left\|\nabla F(y)-\nabla F(x)\right\|\leq L\left\|y-x\right\|. |  |

Let ℱt\mathcal{F}\_{t} represent the information available from the parameter iterates up to time tt, i.e. the σ\sigma-field associated to the sequence (θs)s≤t(\theta\_{s})\_{s\leq t}.
Define the per-sample and minibatch gradient noises by

|  |  |  |
| --- | --- | --- |
|  | εt,i=G​(θt,zt,i)−∇F​(θt),andεt=1B​∑j=1Bεt,ij.\varepsilon\_{t,i}=G(\theta\_{t},z\_{t,i})-\nabla F(\theta\_{t}),\quad\text{and}\quad\varepsilon\_{t}=\frac{1}{B}\sum\_{j=1}^{B}\varepsilon\_{t,i\_{j}}. |  |

In contrast to usual SGD analysis, experience replay introduces a bias in the gradient estimate through the correlation introduced by the buffer, even with importance ratio correction.444While importance sampling corrects the marginal distribution mismatch between πθt\pi\_{\theta\_{t}} and πθt−τ\pi\_{\theta\_{t-\tau}}, experience replay forces us to reason about previous distributions conditioned on the current parameters, i.e. the distribution πθt−τ(⋅∣θt)\pi\_{\theta\_{t-\tau}}(\cdot\mid\theta\_{t}) at time θt−τ\theta\_{t-\tau} conditioned on the fact that the training trajectory that followed (which was influenced by the samples drawn at θt−τ\theta\_{t-\tau} through the policy gradient algorithm) ended up at θt\theta\_{t}. The distribution πθt−τ(⋅∣θt)\pi\_{\theta\_{t-\tau}}(\cdot\mid\theta\_{t}) is typically not computable.
We expect this bias to be larger when trajectories presently in the buffer have had a strong influence on the subsequent updates leading to the current parameter θt\theta\_{t}, and to be small when the parameters have moved little over the time span covered by the buffer.
This intuition motivates the following assumption, discussed further in Appendix [9](#S9 "9 Mathematical Details ‣ Efficient RL Training for LLMs with Experience Replay").

###### Assumption 4.2 (Bias).

There exists a constant κ≥0\kappa\geq 0 such that for all (t,i)(t,i),

|  |  |  |
| --- | --- | --- |
|  | ∥𝔼[εt,i∣ℱt]∥≤κ∥θt−θti∥,\left\|{\mathbb{E}}[\varepsilon\_{t,i}\mid{\mathcal{F}}\_{t}]\right\|\leq\kappa\left\|\theta\_{t}-\theta\_{t\_{i}}\right\|, |  |

where ti=t+1−⌈i/R⌉t\_{i}=t+1-\left\lceil i/R\right\rceil is the time at which the ii-th element of the buffer was added to the buffer.

The variance of our gradient estimates depends on both the per-sample variance, and the correlation between different samples drawn within the same minibatch.
The per-sample variance typically increases with off-policiness, reflecting the growing variance of importance ratio as off-policiness increases.
In addition, samples within a batch can be statistically dependent, since some may have influenced the sequence of parameter updates that produced the others.
This coupling is mediated by how strongly any individual rollout can affect subsequent iterates.
At time tit\_{i}, a rollout generated at time tj<tit\_{j}<t\_{i} will have contributed, on average, (ti−tj)⋅B/N(t\_{i}-t\_{j})\cdot B/N times to the gradient updates between tit\_{i} and tjt\_{j}. As each update averages over BB samples, we expect the dependency to scale in O​(|ti−tj|/N)O(\left|t\_{i}-t\_{j}\right|/N).
This motivates the following assumption.

###### Assumption 4.3 (Variance).

There exists a non-decreasing function σ:ℝ→ℝ+\sigma:{\mathbb{R}}\rightarrow{\mathbb{R}}\_{+} and a coefficient ρ∈[0,1]\rho\in[0,1], such that for any (t,i)(t,i),

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[‖εt,i‖2]≤σ2​(t−ti),{\mathbb{E}}[\left\|\varepsilon\_{t,i}\right\|^{2}]\leq\sigma^{2}(t-t\_{i}), |  |

and for j≠ij\neq i,

|  |  |  |
| --- | --- | --- |
|  | correlation⁡(εt,i,εt,j)≤ρ​|ti−tj|N.\operatorname{correlation}(\varepsilon\_{t,i},\varepsilon\_{t,j})\leq\frac{\rho\left|t\_{i}-t\_{j}\right|}{N}. |  |

We are now ready to state the main convergence theorem, proven in Appendix [9](#S9 "9 Mathematical Details ‣ Efficient RL Training for LLMs with Experience Replay").

###### Theorem 4.4.

Under Assumptions [4.1](#S4.Thmtheorem1 "Assumption 4.1 (Target Smoothness). ‣ 4 Mathematical Analysis ‣ Efficient RL Training for LLMs with Experience Replay"), [4.2](#S4.Thmtheorem2 "Assumption 4.2 (Bias). ‣ 4 Mathematical Analysis ‣ Efficient RL Training for LLMs with Experience Replay") and [4.3](#S4.Thmtheorem3 "Assumption 4.3 (Variance). ‣ 4 Mathematical Analysis ‣ Efficient RL Training for LLMs with Experience Replay"),
when the learning rate satisfies η≤min⁡(R/(2​2​κ​N),L/2)\eta\leq\min(R/(2\sqrt{2}\kappa N),L/2)

|  |  |  |
| --- | --- | --- |
|  | 1T​∑t=1T−1‖∇F​(θt)‖2≤12​F​(θ0)η​T+8​η​(4​N2​κ2​ηR2+L)​𝒱\frac{1}{T}\sum\_{t=1}^{T-1}\left\|\nabla F(\theta\_{t})\right\|^{2}\leq\frac{12F(\theta\_{0})}{\eta T}+8\eta\left(\frac{4N^{2}\kappa^{2}\eta}{R^{2}}+L\right){\mathcal{V}} |  |

for any T>1T>1,
where 𝒱{\mathcal{V}} is a variance parameter defined as

|  |  |  |
| --- | --- | --- |
|  | 𝒱=σ¯2​(NR)​(1B+1N+ρR).{\mathcal{V}}=\bar{\sigma}^{2}\left(\frac{N}{R}\right)\left(\frac{1}{B}+\frac{1}{N}+\frac{\rho}{R}\right). |  |

and σ¯​(H)\bar{\sigma}(H) is the average of σ​(1),…,σ​(H)\sigma(1),\ldots,\sigma(H).

###### Theorem 4.5 (Optimal Design).

Given an asymptotically large compute budget CC, related to the number TT of iterations by C=(B+μ​R)​TC=(B+\mu R)T, we optimize over (η,N,R,B)(\eta,N,R,B) the bound in Theorem [4.4](#S4.Thmtheorem4 "Theorem 4.4. ‣ 4 Mathematical Analysis ‣ Efficient RL Training for LLMs with Experience Replay"). Assuming RR divides NN, and relaxing integer constraints, it yields the optimal ratios

|  |  |  |  |
| --- | --- | --- | --- |
|  | N/R\displaystyle N/R | =x∗:=arg​minx>0⁡σ¯2​(x)​(1/μ+ρ+1/x)2,\displaystyle=x\_{\*}:=\operatorname\*{arg\,min}\_{x>0}\bar{\sigma}^{2}(x)(\sqrt{1/\mu}+\sqrt{\rho+1/x})^{2}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | B/R\displaystyle B/R | =r∗:=μ/(ρ+1/x∗).\displaystyle=r\_{\*}:=\sqrt{\mu/(\rho+1/x\_{\*})}. |  |

Here, N/RN/R denotes the off-policiness horizon, i.e. the maximum off-policiness of rollouts in the buffer, and B/RB/R the replay ratio, i.e. the average number of times a sample is replayed over the full run.555Note that R/NR/N is the ratio of fresh samples in the buffer, thus by contraposition N/RN/R is the number of rounds a sample will stay in the buffer.
Moreover, since each sample in the buffer is associated with a sampling probability 1/N1/N, we sample BB of them in a batch, and a sample stays for N/RN/R round in the buffer, their average use over their shelf-life is (1/N)​B​(N/R)=B/R(1/N)B(N/R)=B/R.

We also provide a closed-form expression for x∗x\_{\*} in Appendix [9](#S9 "9 Mathematical Details ‣ Efficient RL Training for LLMs with Experience Replay") under a power-law assumption on σ\sigma, as well as further empirical illustrations in Figure [6](#S9.F6 "Figure 6 ‣ 9.3.2 Closed-Form Solution with Power-Law Variance ‣ 9.3 Design Trade-Off. ‣ 9 Mathematical Details ‣ Efficient RL Training for LLMs with Experience Replay").

Theorem [4.5](#S4.Thmtheorem5 "Theorem 4.5 (Optimal Design). ‣ 4 Mathematical Analysis ‣ Efficient RL Training for LLMs with Experience Replay") characterizes the optimal replay-buffer design in terms of the *staleness horizon* N/RN/R and the *replay ratio* B/RB/R.
These ratios serve as key design levers, allowing practitioners to systematically configure the replay buffer for peak algorithmic performance.
Theorem [4.5](#S4.Thmtheorem5 "Theorem 4.5 (Optimal Design). ‣ 4 Mathematical Analysis ‣ Efficient RL Training for LLMs with Experience Replay") reveals a three-way trade-off between staleness-induced noise growth (σ¯2)\bar{\sigma}^{2}), coupling between replayed samples and the parameter iterates (ρ\rho), and the rollout-vs-training compute imbalance (μ\mu).
When the compute cost of rollouts is small (small μ\mu), or when off-policy induced variance (σ¯\bar{\sigma} increases fast) and correlation (ρ\rho) are high, the optimal staleness horizon x∗x\_{\*} approaches zero. This suggests that in such regimes, it is more effective to remain on-policy than to utilize a replay buffer.
Conversely, when rollout generation is expensive (large μ\mu) or off-policy effects are negligible (σ¯\bar{\sigma} and ρ\rho are small), a replay buffer becomes optimal, characterized by a large staleness horizon and a high replay count.
Overall, our theory formalizes the central trade-off studied in our experiments: replay can substantially reduce inference compute, but only up to the point where staleness-induced variance and samples-iterate correlations begin to dominate the benefit of reusing trajectories.

## 5 Experimental results

We explore how experience replay impacts accuracy and compute efficiency when training small and mid-size models with asynchronous RL fine-tuning on reasoning datasets.

### 5.1 Experimental setup

We evaluate replay buffers in the *asynchronous* setting described in Section [3](#S3 "3 Experience Replay for Off-Policy RL ‣ Efficient RL Training for LLMs with Experience Replay"), with WW inference workers generating rollouts and TT trainers performing optimization steps from a shared buffer. Unless otherwise specified, we sample from the buffer uniformly.
In our primary experiments, we fine-tune Qwen3-0.6B and Qwen2.5-7B (qwen2025qwen25technicalreport)
with GRPO (shao2024deepseekmath) on OpenR1-Math-220k (openr1\_math220k), and evaluate on either OpenR1-Math-220k or MATH (hendrycksmath2021).
Unless stated explicitly, we use a learning rate of 3.37⋅10−73.37\cdot 10^{-7} for Qwen3-0.6B and 6⋅10−86\cdot 10^{-8} for Qwen2.5-7B.
We plot accuracy w.r.t. either the number of gradient steps, the compute spent (estimated with ([2](#S3.E2 "Equation 2 ‣ Compute Efficiency ‣ 3.2 Off-Policiness, Diversity, and Compute Efficiency ‣ 3 Experience Replay for Off-Policy RL ‣ Efficient RL Training for LLMs with Experience Replay"))) or the wall-time.
All our experiments are run with at least 44 random seeds, and we report the median and the interquartile range.
See Appendix [11](#S11 "11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay") for ablations on the learning rate, and Appendix [10](#S10 "10 Experimental details ‣ Efficient RL Training for LLMs with Experience Replay") for additional details on the setup, including the estimation of the optimal ratio μ\mu.

!(/html/2604.08706/assets/x4.png)

!(/html/2604.08706/assets/x5.png)

!(/html/2604.08706/assets/x6.png)
 
!(/html/2604.08706/assets/x7.png)

Figure 3:  Accuracy and Pass@kk with respect to Buffer Size. Left: Test accuracy as a function of compute spent when training Qwen3-0.6B on OpenR1-Math-220k for (W,T)=(6,2)(W,T)=(6,2) and various buffer sizes N∈{64,128,256,512,768,1536,2304,6912,20736}N\in\{64,128,256,512,768,1536,2304,6912,20736\}, as well as for a no-buffer baseline. We report the median and IQR over more than 44 seeds. Compute is normalized so that each weight update costs 11 unit for buffer configurations and 1.961.96 for the baseline.
Middle: Pass@kk increase after training for a representative subset of these buffer configurations, relative to the baseline. Right: Best Accuracy achieved over entire runs for various buffer sizes and W/TW/T ratios.
The compute needed to reach those accuracies is reported in Figure [14](#S11.F14 "Figure 14 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay") in Appendix [11](#S11 "11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay").

### 5.2 Main results

Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Efficient RL Training for LLMs with Experience Replay") summarizes our central finding: for a good choice of buffer configuration, one may save up to 40% of compute to reach a given accuracy.
For all compute budget, the accuracy achievable using experience replay is superior to that achievable with strictly on-policy training, contradicting the current paradigm. Moreover, we observe an additional benefit not predicted by our theory: using a buffer stabilizes training, preventing crashes and sometimes enabling a higher peak accuracy.
These findings are confirmed on other buffer configurations, other models and other tasks in Figures [15](#S11.F15 "Figure 15 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay"), [16](#S11.F16 "Figure 16 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay") and [17](#S11.F17 "Figure 17 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay") of Appendix [11](#S11 "11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay").

We now run more comprehensive experiments on a smaller model to further analyze the impact of various buffer hyperparameters and better understand these phenomena.

##### Buffer Size and Off-Policiness.

The left side of Figure [3](#S5.F3 "Figure 3 ‣ 5.1 Experimental setup ‣ 5 Experimental results ‣ Efficient RL Training for LLMs with Experience Replay") shows the test accuracy of Qwen3-0.6B for (W,T)=(6,2)(W,T)=(6,2) and various buffer sizes as a function of compute.
We first observe that all training trajectories (with or without buffer) culminate in a global maximum accuracy, followed by a decline in performance–this is not an uncommon phenomenon in RL (see, e.g., zheng2025prosperitycollapsefaroffpolicy).666Looking at the training accuracy (Figure [13](#S11.F13 "Figure 13 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay") in Appendix [11](#S11 "11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay")), we see that it peaks later than the test accuracy, then crashes as well, indicating that the models initially overfit before ultimately collapsing into a nonsensical policy.
We further observe that increasing the size of the buffer, hence increasing the average off-policiness of the samples, has two marked effects: it slows down the training, and it stabilizes it, leading to a potentially higher maximal accuracy that is reached later in the training run.
As a secondary exploration, we trained the same model without a replay buffer and introduced various levels of off-policiness in the training distribution. The results, reported in Figure [12](#S11.F12 "Figure 12 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay") in the Appendix, align with our findings and show that moderate levels of off-policiness can have a stabilizing effect on the training (independently from the use of experience replay).
We hypothesize that reusing rollouts sampled from older policies regularizes the (evolving) objective function by increasing the diversity of the training distribution, and thus helps prevent overfitting.
As larger models take much longer to overfit, the same effect is not visible in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Efficient RL Training for LLMs with Experience Replay").

##### Replay ratio.

As the ratio W/TW/T between inference workers and trainers decreases, the compute cost of each gradient update drops (Table [1](#S3.T1 "Table 1 ‣ Compute Efficiency ‣ 3.2 Off-Policiness, Diversity, and Compute Efficiency ‣ 3 Experience Replay for Off-Policy RL ‣ Efficient RL Training for LLMs with Experience Replay")), but the average replay ratio rises, going from 2.22.2 for (W,T)=(6,2)(W,T)=(6,2) to 5.65.6 and 17.617.6 for (W,T)=(5,3)(W,T)=(5,3) and (4,4)(4,4) respectively.
We see on the heatmap in Figure [3](#S5.F3 "Figure 3 ‣ 5.1 Experimental setup ‣ 5 Experimental results ‣ Efficient RL Training for LLMs with Experience Replay") that while moderate replay ratios do not adversely affect the maximal accuracy, aggressive replay eventually degrades performances (most likely due to the associated reduced *local* sample diversity, see Section [3](#S3 "3 Experience Replay for Off-Policy RL ‣ Efficient RL Training for LLMs with Experience Replay")).
As shown on the more exhaustive plots for (W,T)∈{(5,3),(4,4)}(W,T)\in\{(5,3),(4,4)\} in Figure [13](#S11.F13 "Figure 13 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay") of Appendix [11](#S11 "11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay"), more extreme configurations can nonetheless remain attractive due to their high compute efficiency.

##### Output diversity.

One can see in Figure [3](#S5.F3 "Figure 3 ‣ 5.1 Experimental setup ‣ 5 Experimental results ‣ Efficient RL Training for LLMs with Experience Replay") that training with experience replay can also improve the pass@k (for k>1k>1). This is true in absolute terms (i.e. the pass@k is improved), but also comparatively: using a buffer helps the pass@k for large k even more than it helps the pass@1.
This shows that while the loss in diversity of the model’s output distribution is a major concern in RL (cui2025entropymechanismreinforcementlearning; yue2025doesreinforcementlearningreally), experience replay can help preserve it.
We attribute this phenomenon to the increased diversity of the training distribution which results from the use of older samples.

To summarize, our experiments suggest that reducing the ratio W/TW/T improves compute efficiency but worsens learning dynamics, whereas increasing the buffer size slows training while stabilizing it and helping preserve output diversity. Under suitable configurations, these effects combine to yield a net improvement across all metrics relative to strictly on-policy RL.

### 5.3 Wall-time Speed

We found compute (as defined through ([2](#S3.E2 "Equation 2 ‣ Compute Efficiency ‣ 3.2 Off-Policiness, Diversity, and Compute Efficiency ‣ 3 Experience Replay for Off-Policy RL ‣ Efficient RL Training for LLMs with Experience Replay"))), which isolates the algorithmic effect of experience replay (fewer rollouts per update), to be a more informative metric than wall-time, which is influenced by implementation-dependent scheduling and queuing effects.
That said, we have observed that the gains in wall-speed from using a buffer in our particular setup either match or exceed the gains in compute efficiency (see Figures [10](#S11.F10 "Figure 10 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay") and [11](#S11.F11 "Figure 11 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay") in Appendix [11](#S11 "11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay")).

Indeed, in asynchronous settings (described in Section [3](#S3 "3 Experience Replay for Off-Policy RL ‣ Efficient RL Training for LLMs with Experience Replay")), inference workers often stall when the transfer queue is full, and trainers stall when the queue is empty—both effects are exacerbated when reward computation introduces variable latency (as also noted by lu2025arpoendtoendpolicyoptimizationgui).
This can occur even when the optimal ratio μ\mu of trainer GPUs to inference GPUs is achieved, and is exacerbated when it is not.
A replay buffer *attenuates* these stalls by decoupling production from consumption: trainers can continue optimizing even when rollout generation temporarily slows, and inference workers can continue generating rollouts even when trainers are temporarily back-pressured.
This smoothing effect brought by the buffer is independent from the increase in compute efficiency discussed at length above.
It can be leveraged to streamline an asynchronous RL pipeline with a ratio W/TW/T set to precisely μ\mu while keeping the expected replay ratio equal to 11.

### 5.4 Controlling for the learning rate: optimality curves

We performed preliminary ablations (reported in Figures [8](#S11.F8 "Figure 8 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay") and [9](#S11.F9 "Figure 9 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay") in Appendix [11](#S11 "11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay")) to ensure that we selected for each model the optimal learning rate for the baseline, i.e. that which led to the highest maximum accuracy and the greatest training stability.
As experience replay changes the optimization dynamics,777E.g., a (statistically unlikely) scenario where the exact same training batch is reused twice in a row would in fact be equal, up to second order terms, to a single gradient step with a learning rate twice as large. we ran further control experiments to ensure that the efficiency gains reported cannot be attributed to inadequate hyperparameters tuning.
Namely, we performed an extensive sweep across learning rates and buffer configurations. For both buffer and non-buffer setups, we plot for each compute budget the best achievable accuracy (over learning rates and buffer parameters) for that budget, resulting in two *optimality curves* reported in Figure [4](#S5.F4 "Figure 4 ‣ 5.4 Controlling for the learning rate: optimality curves ‣ 5 Experimental results ‣ Efficient RL Training for LLMs with Experience Replay").
We find that the best buffer configurations consistently outperform the best non-buffer configurations.

!(/html/2604.08706/assets/x8.png)

Figure 4: Pareto Frontier across Hyperparameters Sweep.
Test accuracy as a function of compute spent when training Qwen3-0.6B on OpenR1-Math-220k for various learning rates ({1.5i⋅10−7}i=05\{1.5^{i}\cdot 10^{-7}\}\_{i=0}^{5}) and buffer configurations: no buffer (blue curves), buffer of size {64,128,256,512,768,2304,6912,20736}\{64,128,256,512,768,2304,6912,20736\} with (W,T)∈{(6,2),(5,3),(4,4)}(W,T)\in\{(6,2),(5,3),(4,4)\} (orange curves). Each curve is the median over at least 44 seeds. The two boldfaced curves delineate the Pareto frontier of each family of runs. Compute is normalized so that each weight update costs 11 unit for baseline configurations and and 0.510.51 for buffer configurations.

### 5.5 Further optimization: refining replay buffer design

So far, we have intentionally focused on the simplest replay buffer implementation, requiring the least deviation from the standard SOTA pipelines.
We now extend our study to more exotic designs in search of further improvements, and consider two refinements.
Firstly, we replace the basic sampling strategy used hitherto with a modified strategy, that we call positive-bias sampling: instead of keeping the freshest NN generated rollouts in the buffer, we keep the freshest (1−δ)​N(1-\delta)N generated rollouts along with the freshest δ​N\delta N correct rollouts not included in those (1−δ)​N(1-\delta)N trajectories (an example is given in Appendix [10.4](#S10.SS4 "10.4 Buffer-specific aspects ‣ 10 Experimental details ‣ Efficient RL Training for LLMs with Experience Replay")), and uniformly sample from these NN samples.
Our intuition is that the utility of correct rollouts is less affected by off-policiness.
Secondly, we replace GRPO with the AsymRE loss from arnal2025asymmetricreinforceoffpolicyreinforcement, which has shown promises in such settings (see Appendix [10](#S10 "10 Experimental details ‣ Efficient RL Training for LLMs with Experience Replay")).
Unlike GRPO, AsymRE does not feature importance ratio correction, which is known to increase variance when off-policiness is high and does not account for subtle dependency effects when sampling from a buffer.

As showcased in Figure [5](#S5.F5 "Figure 5 ‣ 5.5 Further optimization: refining replay buffer design ‣ 5 Experimental results ‣ Efficient RL Training for LLMs with Experience Replay"), we find that both variants lead to substantial improvements over the basic buffer implementation; larger-scale experiments are now needed to validate the robustness of these findings.

As a third refinement, we also tried sampling from a (standard) buffer uniformly without replacement in order to increase local diversity, but the results were inconclusive (see Figure [18](#S11.F18 "Figure 18 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay")).

!(/html/2604.08706/assets/x9.png)

Figure 5: Alternative Loss, Positive-Bias Sampling Rule.
Test accuracy as a function of training steps when training Qwen3-0.6B on OpenR1-Math-220k with a buffer of size N=4608N=4608 and (W,T)=(6,2)(W,T)=(6,2). We use either GRPO or AsymRE, and apply positive-bias sampling with coefficient δ∈{0,0.2,0.5}\delta\in\{0,0.2,0.5\} (note: δ=0\delta=0 corresponds to standard uniform sampling).

## 6 Conclusion

In this work, we challenged the "generate-then-discard" paradigm that currently dominates LLM reinforcement learning.
Through a combination of theoretical analysis and extensive empirical evaluation, we show that a well-configured replay buffer serves as a powerful lever for compute efficiency.
Our theoretical framework characterizes a fundamental three-way trade-off between staleness, sample diversity, and the relative cost of inference.
We show that as the computational burden of rollout generation grows, the optimal strategy shifts decisively toward experience replay.
Empirically, we find that these gains are not merely theoretical: a simple replay buffer can reduce the compute budget by up to 40% while maintaining or even surpassing the accuracy of on-policy baselines.
These findings suggest that maximizing performance per unit of compute, rather than per gradient step, is a more practical objective for RL pipelines, and that experience replay is a key component in achieving this.

While our results are consistent for the model scales evaluated in this study, further work is needed to validate these efficiency gains on larger frontier models.
Additionally, we believe that the Pareto frontier can be pushed further by moving beyond uniform buffers toward more sophisticated sampling rules and off-policy corrections, as well as other losses.

\beginappendix

## 7 Extended Related Work

We provide here a more comprehensive overview of experience replay in reinforcement learning, ranging from foundational deep RL works to the most recent applications in Large Language Models.

### 7.1 Experience Replay in Classical Deep RL

The concept of improving computational efficiency by storing and reusing past transitions is standard in general RL but has historically been difficult to stabilize.

* •

  Foundations: mnih2015human (DQN) demonstrated that training on samples drawn randomly from a replay buffer breaks temporal correlations in data, stabilizing the training of value functions. This became a standard component of off-policy learning.
* •

  Prioritized Sampling: schaul2015prioritized introduced Prioritized Experience Replay (PER), improving upon uniform sampling by prioritizing transitions with high temporal-difference (TD) error, effectively focusing learning on "surprising" or difficult examples.
* •

  Hindsight Replay: andrychowicz2017hindsight proposed Hindsight Experience Replay (HER) for goal-oriented tasks. By re-labeling failed trajectories as successful attempts towards the state they *did* reach, HER allows agents to learn from failure, significantly boosting sample efficiency in sparse-reward settings.
* •

  Theoretical Analysis: zhang2017deeper provided an early theoretical analysis of experience replay, investigating the relationship between buffer size, replay ratio, and performance, a line of inquiry we extend to the LLM setting in Section [4](#S4 "4 Mathematical Analysis ‣ Efficient RL Training for LLMs with Experience Replay").

### 7.2 Off-Policy Algorithms

Using a replay buffer inherently introduces off-policiness—the discrepancy between the data-generating policy and the current policy. Various algorithms have been designed to handle this:

* •

  Actor-Critic Methods: DDPG (lillicrap2015continuous) and Soft Actor-Critic (SAC) (haarnoja2018soft) are off-policy algorithms that update the policy using samples from a buffer. SAC, in particular, maximizes both expected return and entropy, stabilizing training in complex environments.
* •

  Off-Policy Corrections: The Retrace algorithm (munos2016safe) utilizes truncated importance sampling to safely learn from multi-step returns generated by behavioral policies. Addressing the instability of stale updates in LLMs, zheng2025prosperitycollapsefaroffpolicy propose second-moment constraints (M2PO) to stabilize off-policy training.
* •

  Recent Theoretical Advances: More recent approaches derive consistency conditions from KL-regularized policy optimization problems (rafailov2023direct; richemond2024offline; tang2025rlfinetuningllmsonoffpolicy; cohen2025softpolicyoptimizationonline), analyze the
  role of dataset coverage (song2024importanceonlinedataunderstanding), or propose additive renormalization of baselines (arnal2025asymmetricreinforceoffpolicyreinforcement) to handle distribution shifts mathematically.

### 7.3 Experience Replay in Modern LLM Training

While on-policy methods like PPO (schulman2017proximalpolicyoptimizationalgorithms) and GRPO (shao2024deepseekmath) dominate the current LLM landscape, a wave of very recent works (2025) has begun to explore replay mechanisms. However, their goals differ significantly from ours:

* •

  Improving Performance via Exploration: bartoldson2025trajectorybalanceasynchronydecoupling use a replay buffer combined with a dedicated loss function specifically to increase exploration in sparse reward settings. Similarly, wang2025eframedeeperreasoningexplorationfilterreplay focus on saving successful solutions to challenging prompts ("gold samples") to facilitate reasoning breakthroughs.
* •

  Complex Training Pipelines: zhang2025rlepreinforcementlearningexperience propose a two-phase training procedure where samples from an initial exploration phase are reused, while lu2025arpoendtoendpolicyoptimizationgui use a buffer to implement dynamic sampling strategies.

Unlike these works, which often introduce complex new objectives to maximize final accuracy, our work conducts a rigorous analysis of the *efficiency* trade-offs in standard pipelines with the addition of a simple replay buffer.
We aim to answer how much compute can be saved by reusing data in a standard asynchronous setup without degrading performance?

## 8 Pseudo-Code Implementation

This section provides a peudo-code implementation of the asynchronous Reinforcement Learning pipeline.
This code utilizes Python’s asyncio library to simulate the concurrent execution of inference workers (WW) and trainers (TT).
It highlights the transition from a standard stream-based approach to the replay Buffer architecture discussed in our work.

### 8.1 Queue-based Data Transfer

In baseline asynchronous RL, data typically flows through a last-in, first-out (LIFO) pipe, prioritizing the freshest samples for training.
As noted in Section [3](#S3 "3 Experience Replay for Off-Policy RL ‣ Efficient RL Training for LLMs with Experience Replay"), this structure forces a tight coupling between rollout generation and consumption, where trajectories are discarded after a single update.

[⬇](data:text/plain;base64,aW1wb3J0IGFzeW5jaW8KaW1wb3J0IHJhbmRvbQoKY2xhc3MgUXVldWVTdHJ1Y3R1cmU6CiAgICAiIiJTdGFuZGFyZCBGSUZPIHN0b3JhZ2UgZm9yIG9uLXBvbGljeSByb2xsb3V0cy4iIiIKICAgIGRlZiBfX2luaXRfXyhzZWxmKToKICAgICAgICBzZWxmLnF1ZXVlID0gYXN5bmNpby5MaWZvUXVldWUoKQoKICAgIGFzeW5jIGRlZiBwdXNoKHNlbGYsIGRhdGEpOgogICAgICAgIGF3YWl0IHNlbGYucXVldWUucHV0KGRhdGEpCgogICAgYXN5bmMgZGVmIHNhbXBsZShzZWxmLCBiYXRjaF9zaXplKToKICAgICAgICAjIFN0cmljdGx5IGNvbnN1bWVzIGRhdGE6IGl0ZW1zIGFyZSByZW1vdmVkIG9uY2Ugc2FtcGxlZAogICAgICAgIHJldHVybiBbYXdhaXQgc2VsZi5xdWV1ZS5nZXQoKSBmb3IgXyBpbiByYW5nZShiYXRjaF9zaXplKV0=)

1import asyncio

2import random

3

4class QueueStructure:

5 """Standard FIFO storage for on-policy rollouts."""

6 def \_\_init\_\_(self):

7 self.queue = asyncio.LifoQueue()

8

9 async def push(self, data):

10 await self.queue.put(data)

11

12 async def sample(self, batch\_size):

13 # Strictly consumes data: items are removed once sampled

14 return [await self.queue.get() for \_ in range(batch\_size)]

Listing 1: LIFO Queue implementation for on-policy streaming

### 8.2 Inference Worker

The Sampler represents one of the WW inference workers.
It operates in a loop, generating trajectories to be pushed into the storage structure.
While the pseudo-code suggests a sequential weight update, in efficient implementations, such as the one used for this work, weights are typically updated concurrently to rollout generation to maximize throughput (i.e., the policy may change during a rollout, with later tokens generated under a different set of weights than earlier ones).

[⬇](data:text/plain;base64,Y2xhc3MgU2FtcGxlcjoKICAgIGRlZiBfX2luaXRfXyhzZWxmLCBkdW1wX3N0cnVjdCk6CiAgICAgICAgc2VsZi5kdW1wX3N0cnVjdCA9IGR1bXBfc3RydWN0CgogICAgYXN5bmMgZGVmIHJ1bihzZWxmLCBkYXRhc2V0KToKICAgICAgICBmb3IgZGF0YSBpbiBkYXRhc2V0OgogICAgICAgICAgICBhd2FpdCBzZWxmLnJlY2VpdmVfd2VpZ2h0cygpCiAgICAgICAgICAgIHJvbGxvdXQgPSBhd2FpdCBzZWxmLmdlbmVyYXRlX3JvbGxvdXQoZGF0YSkKICAgICAgICAgICAgYXdhaXQgc2VsZi5kdW1wX3N0cnVjdC5wdXNoKHJvbGxvdXQpCgogICAgICAgICMgU2lnbmFsIGNvbXBsZXRpb24gdG8gdGhlIFRyYWluZXIKICAgICAgICBhd2FpdCBzZWxmLmR1bXBfc3RydWN0LnB1c2goIkRPTkUiKQoKICAgIGFzeW5jIGRlZiByZWNlaXZlX3dlaWdodHMoc2VsZik6CiAgICAgICAgIiIiUHVsbCBsYXRlc3QgcGFyYW1ldGVycyBmcm9tIFRyYWluZXIgdG8gc3RheSBhcyAnb24tcG9saWN5JyBhcyBwb3NzaWJsZS4iIiIKICAgICAgICAuLi4KCiAgICBhc3luYyBkZWYgZ2VuZXJhdGVfcm9sbG91dChzZWxmLCBkYXRhKToKICAgICAgICAiIiJTdGFuZGFyZCBMTE0gaW5mZXJlbmNlIHN0ZXAuIiIiCiAgICAgICAgLi4u)

1class Sampler:

2 def \_\_init\_\_(self, dump\_struct):

3 self.dump\_struct = dump\_struct

4

5 async def run(self, dataset):

6 for data in dataset:

7 await self.receive\_weights()

8 rollout = await self.generate\_rollout(data)

9 await self.dump\_struct.push(rollout)

10

11 # Signal completion to the Trainer

12 await self.dump\_struct.push("DONE")

13

14 async def receive\_weights(self):

15 """Pull latest parameters from Trainer to stay as ’on-policy’ as possible."""

16 ...

17

18 async def generate\_rollout(self, data):

19 """Standard LLM inference step."""

20 ...

Listing 2: Inference Worker (Sampler) logic

### 8.3 The Consumer: Trainer

The Trainer represents one of the TT optimization units.
It pulls batches of size BB and performs gradient updates.
This loop runs concurrently with the Sampler.

[⬇](data:text/plain;base64,Y2xhc3MgVHJhaW5lcjoKICAgIGRlZiBfX2luaXRfXyhzZWxmLCBkdW1wX3N0cnVjdCk6CiAgICAgICAgc2VsZi5kdW1wX3N0cnVjdCA9IGR1bXBfc3RydWN0CiAgICAgICAgc2VsZi5pc19ydW5uaW5nID0gVHJ1ZQoKICAgIGFzeW5jIGRlZiBydW4oc2VsZiwgYmF0Y2hfc2l6ZSk6CiAgICAgICAgd2hpbGUgc2VsZi5pc19ydW5uaW5nOgogICAgICAgICAgICBiYXRjaCA9IGF3YWl0IHNlbGYuZHVtcF9zdHJ1Y3Quc2FtcGxlKGJhdGNoX3NpemUpCgogICAgICAgICAgICBpZiAiRE9ORSIgaW4gYmF0Y2g6CiAgICAgICAgICAgICAgICBzZWxmLmlzX3J1bm5pbmcgPSBGYWxzZQogICAgICAgICAgICAgICAgYnJlYWsKCiAgICAgICAgICAgIGF3YWl0IHNlbGYuZm9yd2FyZF9iYWNrd2FyZChiYXRjaCkKICAgICAgICAgICAgYXdhaXQgc2VsZi51cGRhdGVfd2VpZ2h0cygpCgogICAgYXN5bmMgZGVmIGZvcndhcmRfYmFja3dhcmQoc2VsZiwgYmF0Y2gpOgogICAgICAgICIiIkNvbXB1dGUgR1JQTy9QUE8gbG9zcyBhbmQgZ3JhZGllbnRzLiIiIgogICAgICAgIC4uLgoKICAgIGFzeW5jIGRlZiB1cGRhdGVfd2VpZ2h0cyhzZWxmKToKICAgICAgICAiIiJBcHBseSBvcHRpbWl6ZXIgc3RlcCBhbmQgYnJvYWRjYXN0IG5ldyB3ZWlnaHRzLiIiIgogICAgICAgIC4uLg==)

1class Trainer:

2 def \_\_init\_\_(self, dump\_struct):

3 self.dump\_struct = dump\_struct

4 self.is\_running = True

5

6 async def run(self, batch\_size):

7 while self.is\_running:

8 batch = await self.dump\_struct.sample(batch\_size)

9

10 if "DONE" in batch:

11 self.is\_running = False

12 break

13

14 await self.forward\_backward(batch)

15 await self.update\_weights()

16

17 async def forward\_backward(self, batch):

18 """Compute GRPO/PPO loss and gradients."""

19 ...

20

21 async def update\_weights(self):

22 """Apply optimizer step and broadcast new weights."""

23 ...

Listing 3: Optimization Worker (Trainer) logic

### 8.4 Main Orchestration

The main loop instantiates the workers.
While this pseudo-code implementation uses W=T=1W=T=1 for simplicity, in practice more workers would operate in parallel, with a ratio of worker to trainer GPUs set to maximize GPU utilization by minimizing idle time.

[⬇](data:text/plain;base64,YXN5bmMgZGVmIG1haW4oKToKICAgIGRhdGFzZXQgPSAuLi4KICAgIGJhdGNoX3NpemUgPSAuLi4KICAgIGR1bXBfc3RydWN0ID0gLi4uCgogICAgc2FtcGxlciA9IFNhbXBsZXIoZHVtcF9zdHJ1Y3QpCiAgICB0cmFpbmVyID0gVHJhaW5lcihkdW1wX3N0cnVjdCkKCiAgICAjIExhdW5jaGluZyBpbmZlcmVuY2UgYW5kIHRyYWluaW5nIGNvbmN1cnJlbnRseQogICAgYXdhaXQgYXN5bmNpby5nYXRoZXIoCiAgICAgICAgc2FtcGxlci5ydW4oZGF0YXNldCksCiAgICAgICAgdHJhaW5lci5ydW4oYmF0Y2hfc2l6ZSkKICAgICkKCmlmIF9fbmFtZV9fID09ICJfX21haW5fXyI6CiAgICBhc3luY2lvLnJ1bihtYWluKCkp)

1async def main():

2 dataset = ...

3 batch\_size = ...

4 dump\_struct = ...

5

6 sampler = Sampler(dump\_struct)

7 trainer = Trainer(dump\_struct)

8

9 # Launching inference and training concurrently

10 await asyncio.gather(

11 sampler.run(dataset),

12 trainer.run(batch\_size)

13 )

14

15if \_\_name\_\_ == "\_\_main\_\_":

16 asyncio.run(main())

Listing 4: Asynchronous execution entry point

### 8.5 The Replay Buffer: BufferStructure

A replay buffer can be implemented with minimal changes to the pipeline above.
Indeed, one only needs to replace the transfer queue with a new data structure to implement experience replay.
We present it below as the BufferStructure which will store up to NN buffered trajectories.
Unlike the queue, this structure enables multiple samples of the same trajectory.

[⬇](data:text/plain;base64,Y2xhc3MgQnVmZmVyU3RydWN0dXJlOgogICAgIiIiRXhwZXJpZW5jZSBSZXBsYXkgQnVmZmVyIHN1cHBvcnRpbmcgcmFuZG9tIHNhbXBsaW5nLiIiIgogICAgZGVmIF9faW5pdF9fKHNlbGYsIGJ1ZmZlcl9zaXplKToKICAgICAgICBzZWxmLmJ1ZmZlciA9IFtdCiAgICAgICAgc2VsZi5idWZmZXJfc2l6ZSA9IGJ1ZmZlcl9zaXplCiAgICAgICAgc2VsZi5sb2NrID0gYXN5bmNpby5Mb2NrKCkKCiAgICBhc3luYyBkZWYgcHVzaChzZWxmLCBkYXRhKToKICAgICAgICBhc3luYyB3aXRoIHNlbGYubG9jazoKICAgICAgICAgICAgIyBGSUZPIGV2aWN0aW9uIHBvbGljeSBmb3IgdGhlIGJ1ZmZlcgogICAgICAgICAgICBpZiBsZW4oc2VsZi5idWZmZXIpID49IHNlbGYuYnVmZmVyX3NpemU6CiAgICAgICAgICAgICAgICBzZWxmLmJ1ZmZlci5wb3AoMCkKICAgICAgICAgICAgc2VsZi5idWZmZXIuYXBwZW5kKGRhdGEpCgogICAgYXN5bmMgZGVmIHNhbXBsZShzZWxmLCBiYXRjaF9zaXplKToKICAgICAgICBhc3luYyB3aXRoIHNlbGYubG9jazoKICAgICAgICAgICAgIyBTYW1wbGluZyBkb2VzIG5vdCByZW1vdmUgaXRlbXMgZnJvbSB0aGUgYnVmZmVyCiAgICAgICAgICAgIHJldHVybiByYW5kb20uc2FtcGxlKHNlbGYuYnVmZmVyLCBiYXRjaF9zaXplKQ==)

1class BufferStructure:

2 """Experience Replay Buffer supporting random sampling."""

3 def \_\_init\_\_(self, buffer\_size):

4 self.buffer = []

5 self.buffer\_size = buffer\_size

6 self.lock = asyncio.Lock()

7

8 async def push(self, data):

9 async with self.lock:

10 # FIFO eviction policy for the buffer

11 if len(self.buffer) >= self.buffer\_size:

12 self.buffer.pop(0)

13 self.buffer.append(data)

14

15 async def sample(self, batch\_size):

16 async with self.lock:

17 # Sampling does not remove items from the buffer

18 return random.sample(self.buffer, batch\_size)

Listing 5: Circular Replay Buffer with Random Sampling

## 9 Mathematical Details

We provide additional details regarding the mathematical analysis in Section [4](#S4 "4 Mathematical Analysis ‣ Efficient RL Training for LLMs with Experience Replay").

### 9.1 Modeling Details

##### Bias Assumption.

Assumption [4.2](#S4.Thmtheorem2 "Assumption 4.2 (Bias). ‣ 4 Mathematical Analysis ‣ Efficient RL Training for LLMs with Experience Replay") can be motivated by writing the bias explicitly, using the duality bracket and any dual norms

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[εt,i∣ℱt]\displaystyle{\mathbb{E}}[\varepsilon\_{t,i}\mid{\mathcal{F}}\_{t}] | =𝔼z∼πθt−ti[G(θt,z)∣ℱt]−𝔼z∼πθt[G(θt,z)]=⟨πθt−ti(⋅∣ℱt)−πθt,G(θt,⋅)⟩\displaystyle={\mathbb{E}}\_{z\sim\pi\_{\theta\_{t-t\_{i}}}}[G(\theta\_{t},z)\mid{\mathcal{F}}\_{t}]-{\mathbb{E}}\_{z\sim\pi\_{\theta\_{t}}}[G(\theta\_{t},z)]=\left\langle\pi\_{\theta\_{t-t\_{i}}}(\cdot\mid{\mathcal{F}}\_{t})-\pi\_{\theta\_{t}},G(\theta\_{t},\cdot)\right\rangle |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤∥πθt−ti(⋅∣ℱt)−πθt∥∥G(θt,⋅)∥∗.\displaystyle\leq\left\|\pi\_{\theta\_{t-t\_{i}}}(\cdot\mid{\mathcal{F}}\_{t})-\pi\_{\theta\_{t}}\right\|\left\|G(\theta\_{t},\cdot)\right\|\_{\*}. |  |

Here πθt−ti(⋅∣ℱt)\pi\_{\theta\_{t-t\_{i}}}(\cdot\mid{\mathcal{F}}\_{t}) denote the distribution of the samples under θt−ti\theta\_{t-t\_{i}} knowing that the future iterates up to θt\theta\_{t}.
If zz in position ii in the buffer at time tt was never sampled in the batches leading from θt−ti\theta\_{t-t\_{i}} to θt\theta\_{t}, πθt−ti(⋅∣ℱt)\pi\_{\theta\_{t-t\_{i}}}(\cdot\mid{\mathcal{F}}\_{t}) would be equal to πθt−ti\pi\_{\theta\_{t-t\_{i}}}, as knowing the iterates ℱt{\mathcal{F}}\_{t} would not help us reconstruct that sample.
However, the more the sample was used, the more these distributions would be dissimilar.
With κ0\kappa\_{0} the average repetition of a sample in training batch from time tit\_{i} to tt, one may posit

|  |  |  |
| --- | --- | --- |
|  | ∥πθt−ti(⋅∣ℱt)−πθt∥≤κ0∥πθt−ti−πθt∥,\left\|\pi\_{\theta\_{t-t\_{i}}}(\cdot\mid{\mathcal{F}}\_{t})-\pi\_{\theta\_{t}}\right\|\leq\kappa\_{0}\left\|\pi\_{\theta\_{t-t\_{i}}}-\pi\_{\theta\_{t}}\right\|, |  |

where κ0\kappa\_{0} capture the measure of local diversity discussed in Section [3](#S3 "3 Experience Replay for Off-Policy RL ‣ Efficient RL Training for LLMs with Experience Replay"): the more a sample is reused on average between time tit\_{i} and tt, the bigger κ\kappa.888As such, one may want to refine κ0\kappa\_{0} to be a function of the average number of time a sample was used between time tit\_{i} and tt, which is (min⁡(t−ti,N/R)−1)​B/N(\min(t-t\_{i},N/R)-1)B/N.
Assuming GG is bounded by some constant G∞G\_{\infty}, we get a bound on the bias of the form

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[εt,i∣ℱt]≤κ0​G∞​‖πθt−ti−πθt‖.{\mathbb{E}}[\varepsilon\_{t,i}\mid{\mathcal{F}}\_{t}]\leq\kappa\_{0}G\_{\infty}\left\|\pi\_{\theta\_{t-t\_{i}}}-\pi\_{\theta\_{t}}\right\|. |  |

Finally assuming the policy is parameterized in some Lipschitz way for some constant CC, we get

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[εt,i∣ℱt]≤κ0​G∞​C​‖θt−ti−θt‖.{\mathbb{E}}[\varepsilon\_{t,i}\mid{\mathcal{F}}\_{t}]\leq\kappa\_{0}G\_{\infty}C\left\|\theta\_{t-t\_{i}}-\theta\_{t}\right\|. |  |

This motivates formally Assumption [4.2](#S4.Thmtheorem2 "Assumption 4.2 (Bias). ‣ 4 Mathematical Analysis ‣ Efficient RL Training for LLMs with Experience Replay").

##### Variance Assumption.

When using zz the ii-th element of the buffer to estimate ∇F​(θt)\nabla F(\theta\_{t}), the per-sample estimator typically includes some form of off-policy correction (explicit importance weights, clipped ratios as in PPO-style objectives, or implicit reweighting through an advantage estimator).
Abstractly, one may write the estimator as
G​(θt,zt,i)=wt,ti​(z)​G0​(θt,z),G(\theta\_{t},z\_{t,i})=w\_{t,t\_{i}}(z)G\_{0}(\theta\_{t},z),
where wt,tiw\_{t,t\_{i}} is a (possibly clipped) importance-ratio weighting between πθt\pi\_{\theta\_{t}} and πθti\pi\_{\theta\_{t\_{i}}} (recall that zz was generated zz by πθti\pi\_{\theta\_{t\_{i}}}), and G0G\_{0} is a bounded-variance on-policy quantity (e.g. a score-function term times an advantage).
As τ:=t−ti\tau:=t-t\_{i} grows, the mismatch between πθt\pi\_{\theta\_{t}} and πθti\pi\_{\theta\_{t\_{i}}} typically increases, which in turn increases the variability of importance weight wt,tiw\_{t,t\_{i}} and thus the variance of G​(θt,z)G(\theta\_{t},z).
This motivates an upper bound of the form
𝔼​[‖εt,i‖2]≤σ2​(τ){\mathbb{E}}[\|\varepsilon\_{t,i}\|^{2}]\leq\sigma^{2}(\tau)
for some increasing function σ2\sigma^{2}, which captures (in aggregate) the growth of off-policy noise with staleness.

##### Dependencies Assumption.

In standard SGD analyses, the samples ztz\_{t} are i.i.d., and minibatching yields a 1/B1/B variance reduction.
With experience replay, however, the buffer at time tt is “endogenous”: trajectories currently stored in the buffer may have been used in past updates, and those updates affected the parameters that later generated other trajectories that are now in the buffer.
Concretely, εt,i=G​(θt,zt,i)−∇F​(θt)\varepsilon\_{t,i}=G(\theta\_{t},z\_{t,i})-\nabla F(\theta\_{t}) depends on θt\theta\_{t}, while θt\theta\_{t} itself is a function of past minibatch draws; hence two buffer elements can become statistically coupled through the update trajectory that produced θt\theta\_{t}.
At a given step, a fixed element of a buffer of size NN is selected in expectation B/NB/N times (sampling with replacement).
Over hh steps, it is therefore used about h​B/NhB/N times.
Since each update is an average over BB samples, each occurrence contributes a factor 1/B1/B to the update.
Now consider two distinct buffer elements i≠ji\neq j with insertion times ti≤tjt\_{i}\leq t\_{j}.
The updates in the interval [ti,tj)[t\_{i},t\_{j}) can transmit information from zt,iz\_{t,i} to later iterates that enter εt,j\varepsilon\_{t,j} (since zt,jz\_{t,j} is only generated at time tjt\_{j}).
Thus the strength of the coupling should generally increase with the temporal separation |ti−tj||t\_{i}-t\_{j}|.
Aggregating algorithm-specific constants (e.g. clipping, advantage normalization, optimizer state) into a function ρ\rho, this motivates

|  |  |  |
| --- | --- | --- |
|  | corr​(εt,i,εt,j)≤ρN​|ti−tj|,\mathrm{corr}(\varepsilon\_{t,i},\varepsilon\_{t,j})\leq\frac{\rho}{N}|t\_{i}-t\_{j}|, |  |

for ρ\rho a value in [0,1][0,1].
Note that even when ti=tjt\_{i}=t\_{j}, a residual dependence may remain because both trajectories can jointly influence the subsequent parameter path and hence θt\theta\_{t}. While we omit it for simplicity, adding it would not change much the derivations.

### 9.2 Proof of Convergence

Combining the LL-smoothness Assumption [4.1](#S4.Thmtheorem1 "Assumption 4.1 (Target Smoothness). ‣ 4 Mathematical Analysis ‣ Efficient RL Training for LLMs with Experience Replay") with one of Taylor expansion formulas yields

|  |  |  |
| --- | --- | --- |
|  | F​(y)≤F​(x)+⟨∇F​(x),y−x⟩+L2​‖y−x‖2.F(y)\leq F(x)+\left\langle\nabla F(x),y-x\right\rangle+\frac{L}{2}\left\|y-x\right\|^{2}. |  |

Applied in θt+1\theta\_{t+1} and θt\theta\_{t}

|  |  |  |
| --- | --- | --- |
|  | F​(θt+1)≤F​(θt)+⟨∇F​(θt),θt+1−θt⟩+L2​‖θt+1−θt‖2.F(\theta\_{t+1})\leq F(\theta\_{t})+\left\langle\nabla F(\theta\_{t}),\theta\_{t+1}-\theta\_{t}\right\rangle+\frac{L}{2}\left\|\theta\_{t+1}-\theta\_{t}\right\|^{2}. |  |

With

|  |  |  |
| --- | --- | --- |
|  | θt+1−θt=−η​gt=−η​(∇F​(θt)+εt),\theta\_{t+1}-\theta\_{t}=-\eta g\_{t}=-\eta(\nabla F(\theta\_{t})+\varepsilon\_{t}), |  |

we get

|  |  |  |
| --- | --- | --- |
|  | F​(θt+1)≤F​(θt)−η​⟨∇F​(θt),∇F​(θt)+εt⟩+L​η22​‖∇F​(θt)+εt‖2.F(\theta\_{t+1})\leq F(\theta\_{t})-\eta\left\langle\nabla F(\theta\_{t}),\nabla F(\theta\_{t})+\varepsilon\_{t}\right\rangle+\frac{L\eta^{2}}{2}\left\|\nabla F(\theta\_{t})+\varepsilon\_{t}\right\|^{2}. |  |

Developing and rearranging leads to

|  |  |  |
| --- | --- | --- |
|  | F​(θt+1)≤F​(θt)−(η−L​η22)​‖∇F​(θt)‖2−(η−L​η2)​⟨∇F​(θt),εt⟩+L​η22​‖εt‖2.F(\theta\_{t+1})\leq F(\theta\_{t})-\left(\eta-\frac{L\eta^{2}}{2}\right)\left\|\nabla F(\theta\_{t})\right\|^{2}-\left(\eta-L\eta^{2}\right)\left\langle\nabla F(\theta\_{t}),\varepsilon\_{t}\right\rangle+\frac{L\eta^{2}}{2}\left\|\varepsilon\_{t}\right\|^{2}. |  |

Summing over tt and rearranging with get

|  |  |  |
| --- | --- | --- |
|  | (η−L​η22)​1T​∑t=0T−1‖∇F​(θt)‖2≤F​(θ0)−F​(θT)T−(η−L​η2)​1T​∑t=0T−1⟨∇F​(θt),εt⟩+L​η22​1T​∑t=0T−1‖εt‖2.\left(\eta-\frac{L\eta^{2}}{2}\right)\frac{1}{T}\sum\_{t=0}^{T-1}\left\|\nabla F(\theta\_{t})\right\|^{2}\leq\frac{F(\theta\_{0})-F(\theta\_{T})}{T}-\left(\eta-L\eta^{2}\right)\frac{1}{T}\sum\_{t=0}^{T-1}\left\langle\nabla F(\theta\_{t}),\varepsilon\_{t}\right\rangle+\frac{L\eta^{2}}{2}\frac{1}{T}\sum\_{t=0}^{T-1}\left\|\varepsilon\_{t}\right\|^{2}. |  |

Assuming

|  |  |  |  |
| --- | --- | --- | --- |
|  | L​η<1/2,L\eta<1/2, |  | (3) |

we get, with ξ\xi the sign of ∑⟨F​(θt),εt⟩\sum\left\langle F(\theta\_{t}),\varepsilon\_{t}\right\rangle,

|  |  |  |
| --- | --- | --- |
|  | 34​T​∑t=0T−1‖∇F​(θt)‖2≤F​(θ0)−F​(θT)η​T+ξT​∑t=0T−1⟨∇F​(θt),εt⟩+L​η2​1T​∑t=0T−1‖εt‖2.\frac{3}{4T}\sum\_{t=0}^{T-1}\left\|\nabla F(\theta\_{t})\right\|^{2}\leq\frac{F(\theta\_{0})-F(\theta\_{T})}{\eta T}+\frac{\xi}{T}\sum\_{t=0}^{T-1}\left\langle\nabla F(\theta\_{t}),\varepsilon\_{t}\right\rangle+\frac{L\eta}{2}\frac{1}{T}\sum\_{t=0}^{T-1}\left\|\varepsilon\_{t}\right\|^{2}. |  |

Taking the expectation with respect to ℱT{\mathcal{F}}\_{T}, we bound, using Cauchy-Schwarz and a Young’s inequality,

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[⟨∇f​(θt),εt⟩∣ℱT]\displaystyle{\mathbb{E}}[\left\langle\nabla f(\theta\_{t}),\varepsilon\_{t}\right\rangle\mid{\mathcal{F}}\_{T}] | =𝔼​[⟨∇f​(θt),εt⟩∣ℱt]=⟨∇f​(θt),𝔼​[εt∣ℱt]⟩\displaystyle={\mathbb{E}}[\left\langle\nabla f(\theta\_{t}),\varepsilon\_{t}\right\rangle\mid{\mathcal{F}}\_{t}]=\left\langle\nabla f(\theta\_{t}),{\mathbb{E}}[\varepsilon\_{t}\mid{\mathcal{F}}\_{t}]\right\rangle |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤∥∇f(θt)∥∥𝔼[εt∣ℱt]∥≤14∥∇f(θt)∥2+∥𝔼[εt∣ℱt]∥2.\displaystyle\leq\left\|\nabla f(\theta\_{t})\right\|\left\|{\mathbb{E}}[\varepsilon\_{t}\mid{\mathcal{F}}\_{t}]\right\|\leq\frac{1}{4}\left\|\nabla f(\theta\_{t})\right\|^{2}+\left\|{\mathbb{E}}[\varepsilon\_{t}\mid{\mathcal{F}}\_{t}]\right\|^{2}. |  |

Hence,

|  |  |  |
| --- | --- | --- |
|  | ξT∑t=0T−1⟨∇F(θt),εt⟩≤14​T∑t=0T−1∥∇f(θt)∥2+1T∑t=0T−1∥𝔼[εt∣ℱt]∥2.\frac{\xi}{T}\sum\_{t=0}^{T-1}\left\langle\nabla F(\theta\_{t}),\varepsilon\_{t}\right\rangle\leq\frac{1}{4T}\sum\_{t=0}^{T-1}\left\|\nabla f(\theta\_{t})\right\|^{2}+\frac{1}{T}\sum\_{t=0}^{T-1}\left\|{\mathbb{E}}[\varepsilon\_{t}\mid{\mathcal{F}}\_{t}]\right\|^{2}. |  |

Plugging this into the previous inequality, we get

|  |  |  |
| --- | --- | --- |
|  | 12​T∑t=0T−1𝔼[∥∇F(θt)∥2]≤F​(θ0)−F​(θT)η​T+𝔼[1T∑t=0T−1∥𝔼[εt∣ℱt]∥2]+L​η21T∑t=0T−1𝔼[∥εt∥2].\frac{1}{2T}\sum\_{t=0}^{T-1}{\mathbb{E}}[\left\|\nabla F(\theta\_{t})\right\|^{2}]\leq\frac{F(\theta\_{0})-F(\theta\_{T})}{\eta T}+{\mathbb{E}}\Big[\frac{1}{T}\sum\_{t=0}^{T-1}\left\|{\mathbb{E}}[\varepsilon\_{t}\mid{\mathcal{F}}\_{t}]\right\|^{2}\Big]+\frac{L\eta}{2}\frac{1}{T}\sum\_{t=0}^{T-1}{\mathbb{E}}[\left\|\varepsilon\_{t}\right\|^{2}]. |  |

We need to bound the last two quantities, which we identify as the “bias”, and the “variance” part.

#### 9.2.1 Bound on the Bias.

Under Assumption [4.2](#S4.Thmtheorem2 "Assumption 4.2 (Bias). ‣ 4 Mathematical Analysis ‣ Efficient RL Training for LLMs with Experience Replay"), with uniform sampling over the buffer, assuming RR divides NN for simplicity, with H=N/RH=N/R the staleness horizon,

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∥𝔼[εt∣ℱt]∥2\displaystyle\left\|{\mathbb{E}}[\varepsilon\_{t}\mid{\mathcal{F}}\_{t}]\right\|^{2} | =∥𝔼i[𝔼[εt,i∣ℱt]]∥2≤𝔼i[∥𝔼[εt,i∣ℱt]∥2]≤κ2𝔼i[∥θt−θti∥2]=κ2H∑0≤τ<H∥θt−θt−τ∥2.\displaystyle=\Big\|{\mathbb{E}}\_{i}[{\mathbb{E}}[\varepsilon\_{t,i}\mid{\mathcal{F}}\_{t}]]\Big\|^{2}\leq{\mathbb{E}}\_{i}\Big[\big\|{\mathbb{E}}[\varepsilon\_{t,i}\mid{\mathcal{F}}\_{t}]\big\|^{2}\Big]\leq\kappa^{2}{\mathbb{E}}\_{i}\big[\left\|\theta\_{t}-\theta\_{t\_{i}}\right\|^{2}\big]=\frac{\kappa^{2}}{H}\sum\_{0\leq\tau<H}\left\|\theta\_{t}-\theta\_{t-\tau}\right\|^{2}. |  |

The drift is controlled by the magnitude of the gradient updates,

|  |  |  |
| --- | --- | --- |
|  | θt−θt−τ=η​∑t−τ≤s<tgs=η​∑t−τ≤s<t∇F​(θs)+η​∑t−τ≤s<tεs.\theta\_{t}-\theta\_{t-\tau}=\eta\sum\_{t-\tau\leq s<t}g\_{s}=\eta\sum\_{t-\tau\leq s<t}\nabla F(\theta\_{s})+\eta\sum\_{t-\tau\leq s<t}\varepsilon\_{s}. |  |

We proceed with the following bound

|  |  |  |
| --- | --- | --- |
|  | ‖θt−θt−τ‖2≤2​τ​η2​∑t−τ≤s<t‖∇F​(θs)‖2+‖εs‖2≤2​H​η2​∑t−H≤s<t‖∇F​(θs)‖2+‖εs‖2.\left\|\theta\_{t}-\theta\_{t-\tau}\right\|^{2}\leq 2\tau\eta^{2}\sum\_{t-\tau\leq s<t}\left\|\nabla F(\theta\_{s})\right\|^{2}+\left\|\varepsilon\_{s}\right\|^{2}\leq 2H\eta^{2}\sum\_{t-H\leq s<t}\left\|\nabla F(\theta\_{s})\right\|^{2}+\left\|\varepsilon\_{s}\right\|^{2}. |  |

Summing over tt, we get

|  |  |  |
| --- | --- | --- |
|  | 1T∑t=0T−1∥𝔼[εt∣ℱt]∥2≤2H2η2κ21T∑t=0T−1𝔼[∥∇F(θt)∥2∣ℱt]+𝔼[∥εs∥2∣ℱt].\displaystyle\frac{1}{T}\sum\_{t=0}^{T-1}\left\|{\mathbb{E}}[\varepsilon\_{t}\mid{\mathcal{F}}\_{t}]\right\|^{2}\leq 2H^{2}\eta^{2}\kappa^{2}\frac{1}{T}\sum\_{t=0}^{T-1}{\mathbb{E}}[\left\|\nabla F(\theta\_{t})\right\|^{2}\mid{\mathcal{F}}\_{t}]+{\mathbb{E}}[\left\|\varepsilon\_{s}\right\|^{2}\mid{\mathcal{F}}\_{t}]. |  |

When

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2​H2​κ2​η2≤1/4,2H^{2}\kappa^{2}\eta^{2}\leq 1/4, |  | (4) |

plugging our bound on the bias into the main inequality gives

|  |  |  |
| --- | --- | --- |
|  | 14​T​∑t=0T−1𝔼​[‖∇F​(θt)‖2]≤F​(θ0)−F​(θT)η​T+(2​N2​κ2​η2R2+L​η2)​1T​∑t=0T−1𝔼​[‖εt‖2].\frac{1}{4T}\sum\_{t=0}^{T-1}{\mathbb{E}}[\left\|\nabla F(\theta\_{t})\right\|^{2}]\leq\frac{F(\theta\_{0})-F(\theta\_{T})}{\eta T}+\left(\frac{2N^{2}\kappa^{2}\eta^{2}}{R^{2}}+\frac{L\eta}{2}\right)\frac{1}{T}\sum\_{t=0}^{T-1}{\mathbb{E}}[\left\|\varepsilon\_{t}\right\|^{2}]. |  |

#### 9.2.2 Rearrangement between the Variance and the Second Moment

We need to bound the second-moment 𝔼​[‖εt‖2]{\mathbb{E}}[\left\|\varepsilon\_{t}\right\|^{2}].
Let introduce

|  |  |  |
| --- | --- | --- |
|  | ξt,i=εt,i−𝔼​[εt,i],ξt=1B​∑j∈[B]ξt,ij.\xi\_{t,i}=\varepsilon\_{t,i}-{\mathbb{E}}[\varepsilon\_{t,i}],\qquad\xi\_{t}=\frac{1}{B}\sum\_{j\in[B]}\xi\_{t,i\_{j}}. |  |

We have, reusing the previous bound on the bias, together with Eq. ([4](#S9.E4 "Equation 4 ‣ 9.2.1 Bound on the Bias. ‣ 9.2 Proof of Convergence ‣ 9 Mathematical Details ‣ Efficient RL Training for LLMs with Experience Replay")),

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1T​∑t=0T−1𝔼​[‖εt‖2]\displaystyle\frac{1}{T}\sum\_{t=0}^{T-1}{\mathbb{E}}[\left\|\varepsilon\_{t}\right\|^{2}] | =𝔼[1T∑t=0T−1𝔼[∥εt∥2∣ℱt]]=𝔼[1T∑t=0T−1∥𝔼[εt∣ℱt]∥2]+1T∑t=0T−1𝔼[∥ξt∥2]\displaystyle={\mathbb{E}}\Big[\frac{1}{T}\sum\_{t=0}^{T-1}{\mathbb{E}}[\left\|\varepsilon\_{t}\right\|^{2}\mid{\mathcal{F}}\_{t}]\Big]={\mathbb{E}}\Big[\frac{1}{T}\sum\_{t=0}^{T-1}\left\|{\mathbb{E}}[\varepsilon\_{t}\mid{\mathcal{F}}\_{t}]\right\|^{2}\Big]+\frac{1}{T}\sum\_{t=0}^{T-1}{\mathbb{E}}[\left\|\xi\_{t}\right\|^{2}] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤14​T​∑t=0T−1𝔼​[‖∇F​(θt)‖2∣ℱt]+14​T​∑t=0T−1𝔼​[‖εs‖2∣ℱt]+1T​∑t=0T−1𝔼​[‖ξt‖2]\displaystyle\leq\frac{1}{4T}\sum\_{t=0}^{T-1}{\mathbb{E}}[\left\|\nabla F(\theta\_{t})\right\|^{2}\mid{\mathcal{F}}\_{t}]+\frac{1}{4T}\sum\_{t=0}^{T-1}{\mathbb{E}}[\left\|\varepsilon\_{s}\right\|^{2}\mid{\mathcal{F}}\_{t}]+\frac{1}{T}\sum\_{t=0}^{T-1}{\mathbb{E}}[\left\|\xi\_{t}\right\|^{2}] |  |

Hence,

|  |  |  |
| --- | --- | --- |
|  | 1T​∑t=0T−1𝔼​[‖εt‖2]≤13​T​∑t=0T−1𝔼​[‖∇F​(θt)‖2∣ℱt]+43​T​∑t=0T−1𝔼​[‖ξt‖2]\frac{1}{T}\sum\_{t=0}^{T-1}{\mathbb{E}}[\left\|\varepsilon\_{t}\right\|^{2}]\leq\frac{1}{3T}\sum\_{t=0}^{T-1}{\mathbb{E}}[\left\|\nabla F(\theta\_{t})\right\|^{2}\mid{\mathcal{F}}\_{t}]+\frac{4}{3T}\sum\_{t=0}^{T-1}{\mathbb{E}}[\left\|\xi\_{t}\right\|^{2}] |  |

Plugging this into the main bound, and rearranging,

|  |  |  |
| --- | --- | --- |
|  | 112​T​∑t=0T−1𝔼​[‖∇F​(θt)‖2]≤F​(θ0)−F​(θT)η​T+43​(2​N2​κ2​η2R2+L​η2)​1T​∑t=0T−1𝔼​[‖ξt‖2].\frac{1}{12T}\sum\_{t=0}^{T-1}{\mathbb{E}}[\left\|\nabla F(\theta\_{t})\right\|^{2}]\leq\frac{F(\theta\_{0})-F(\theta\_{T})}{\eta T}+\frac{4}{3}\left(\frac{2N^{2}\kappa^{2}\eta^{2}}{R^{2}}+\frac{L\eta}{2}\right)\frac{1}{T}\sum\_{t=0}^{T-1}{\mathbb{E}}[\left\|\xi\_{t}\right\|^{2}]. |  |

#### 9.2.3 Bound on the Variance

Using Assumption [4.3](#S4.Thmtheorem3 "Assumption 4.3 (Variance). ‣ 4 Mathematical Analysis ‣ Efficient RL Training for LLMs with Experience Replay"), we bound, with γ​(|ti−tj|)\gamma(\left|t\_{i}-t\_{j}\right|) the correlation between εt,i\varepsilon\_{t,i} and εt,j\varepsilon\_{t,j},

|  |  |  |  |
| --- | --- | --- | --- |
|  | ⟨ξt,i,ξt,j⟩]\displaystyle\left\langle\xi\_{t,i},\xi\_{t,j}\right\rangle] | =ℙ​(i=j)​𝔼​[‖ξt,i‖2]+ℙ​(i≠j)​𝔼​[⟨ξt,i,ξt,j⟩∣i≠j]\displaystyle={\mathbb{P}}(i=j){\mathbb{E}}[\left\|\xi\_{t,i}\right\|^{2}]+{\mathbb{P}}(i\neq j){\mathbb{E}}[\left\langle\xi\_{t,i},\xi\_{t,j}\right\rangle\mid i\neq j] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤ℙ​(i=j)​𝔼​[σ​(t−ti)2]+ℙ​(i≠j)​𝔼​[γ​(ti−tj)​σ​(t−ti)​σ​(t−tj)]\displaystyle\leq{\mathbb{P}}(i=j){\mathbb{E}}[\sigma(t-t\_{i})^{2}]+{\mathbb{P}}(i\neq j){\mathbb{E}}\Big[\gamma(t\_{i}-t\_{j})\sigma(t-t\_{i})\sigma(t-t\_{j})\Big] |  |

Assuming RR divides NN for simplicity, we have, with H=N/RH=N/R,

|  |  |  |
| --- | --- | --- |
|  | 𝔼[σ(t−ti)2]=1H∑s=0H−1σ(s)2=:σ¯2(H),{\mathbb{E}}[\sigma(t-t\_{i})^{2}]=\frac{1}{H}\sum\_{s=0}^{H-1}\sigma(s)^{2}=:\bar{\sigma}^{2}(H), |  |

together with

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[γ​(|ti−tj|)​σ​(t−ti)​σ​(t−tj)]\displaystyle{\mathbb{E}}[\gamma(\left|t\_{i}-t\_{j}\right|)\sigma(t-t\_{i})\sigma(t-t\_{j})] | =1H2​∑s,s′=0H−1γ​(|s−s′|)​σ​(s)​σ​(s′)≤12​H2​∑s,s′=0H−1γ​(|s−s′|)​(σ​(s)2+σ​(s′)2)\displaystyle=\frac{1}{H^{2}}\sum\_{s,s^{\prime}=0}^{H-1}\gamma(\left|s-s^{\prime}\right|)\sigma(s)\sigma(s^{\prime})\leq\frac{1}{2H^{2}}\sum\_{s,s^{\prime}=0}^{H-1}\gamma(\left|s-s^{\prime}\right|)(\sigma(s)^{2}+\sigma(s^{\prime})^{2}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1H2​∑s=0H−1σ​(s)2​∑s′=0H−1γ​(|s−s′|)≤1H2​∑s=0H−1σ​(s)2​∑τ=0H−12​γ​(τ)\displaystyle=\frac{1}{H^{2}}\sum\_{s=0}^{H-1}\sigma(s)^{2}\sum\_{s^{\prime}=0}^{H-1}\gamma(\left|s-s^{\prime}\right|)\leq\frac{1}{H^{2}}\sum\_{s=0}^{H-1}\sigma(s)^{2}\sum\_{\tau=0}^{H-1}2\gamma(\tau) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =σ¯2(H)1H∑τ=0H−12γ(τ)=:σ¯2(H)γ¯(H).\displaystyle=\bar{\sigma}^{2}(H)\frac{1}{H}\sum\_{\tau=0}^{H-1}2\gamma(\tau)=:\bar{\sigma}^{2}(H)\bar{\gamma}(H). |  |

We deduce that

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[‖ξt‖2]\displaystyle{\mathbb{E}}[\left\|\xi\_{t}\right\|^{2}] | =1B2​∑j,j′∈[B]𝔼​[⟨ξt,ij,ξij′⟩]=1B2​∑j∈[B]𝔼​[‖ξt,ij‖2]+1B2​∑j≠j′∈[B]𝔼​[⟨ξt,ij,ξij′⟩]\displaystyle=\frac{1}{B^{2}}\sum\_{j,j^{\prime}\in[B]}{\mathbb{E}}\big[\left\langle\xi\_{t,i\_{j}},\xi\_{i\_{j^{\prime}}}\right\rangle\big]=\frac{1}{B^{2}}\sum\_{j\in[B]}{\mathbb{E}}[\left\|\xi\_{t,i\_{j}}\right\|^{2}]+\frac{1}{B^{2}}\sum\_{j\neq j^{\prime}\in[B]}{\mathbb{E}}\big[\left\langle\xi\_{t,i\_{j}},\xi\_{i\_{j}^{\prime}}\right\rangle\big] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1B​𝔼​[‖ξt,i‖2]+(B−1)B​𝔼​[⟨ξt,ij,ξt,ij′⟩]\displaystyle=\frac{1}{B}{\mathbb{E}}[\left\|\xi\_{t,i}\right\|^{2}]+\frac{(B-1)}{B}{\mathbb{E}}\big[\left\langle\xi\_{t,i\_{j}},\xi\_{t,i\_{j^{\prime}}}\right\rangle\big] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤σ¯2​(H)B+(B−1)​σ¯2​(H)B​(ℙ​(ij=ij′)+(1−ℙ​(ij≠ij′))​γ¯​(H))\displaystyle\leq\frac{\bar{\sigma}^{2}(H)}{B}+\frac{(B-1)\bar{\sigma}^{2}(H)}{B}\left({\mathbb{P}}(i\_{j}=i\_{j^{\prime}})+(1-{\mathbb{P}}(i\_{j}\neq i\_{j^{\prime}}))\bar{\gamma}(H)\right) |  |

In the case with replacement, we get

|  |  |  |
| --- | --- | --- |
|  | ℙ​(ij=ij′)=1/N,{\mathbb{P}}(i\_{j}=i\_{j^{\prime}})=1/N, |  |

hence

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[‖ξt‖2]≤σ¯2​(H)​(1B+B−1B​1N+B−1B​N−1N​γ¯​(H))≤σ¯2​(NR)​(1B+1N+γ¯​(NR)).\displaystyle{\mathbb{E}}[\left\|\xi\_{t}\right\|^{2}]\leq\bar{\sigma}^{2}(H)\left(\frac{1}{B}+\frac{B-1}{B}\frac{1}{N}+\frac{B-1}{B}\frac{N-1}{N}\bar{\gamma}(H)\right)\leq\bar{\sigma}^{2}\big(\frac{N}{R}\big)\left(\frac{1}{B}+\frac{1}{N}+\bar{\gamma}\big(\frac{N}{R}\big)\right). |  |

With γ​(|ti−tj|)=ρ​|ti−tj|/N\gamma(\left|t\_{i}-t\_{j}\right|)=\rho\left|t\_{i}-t\_{j}\right|/N, we get

|  |  |  |
| --- | --- | --- |
|  | γ¯​(H)=2H​∑τ=0H−1ρ​τN=2​ρH​N​H​(H−1)2=ρ​(H−1)N≤ρ​HN=ρR\bar{\gamma}(H)=\frac{2}{H}\sum\_{\tau=0}^{H-1}\frac{\rho\tau}{N}=\frac{2\rho}{HN}\frac{H(H-1)}{2}=\frac{\rho(H-1)}{N}\leq\frac{\rho H}{N}=\frac{\rho}{R} |  |

Plugging this into the main bound, we get

|  |  |  |
| --- | --- | --- |
|  | 1T​∑t=0T−1𝔼​[‖∇F​(θt)‖2]≤12​F​(θ0)−F​(θT)η​T+8​η​σ¯2​(NR)​(4​N2​κ2​ηR2+L)​(1B+1N+ρR).\frac{1}{T}\sum\_{t=0}^{T-1}{\mathbb{E}}[\left\|\nabla F(\theta\_{t})\right\|^{2}]\leq 12\frac{F(\theta\_{0})-F(\theta\_{T})}{\eta T}+8\eta\bar{\sigma}^{2}\big(\frac{N}{R}\big)\left(\frac{4N^{2}\kappa^{2}\eta}{R^{2}}+L\right)\left(\frac{1}{B}+\frac{1}{N}+\frac{\rho}{R}\right). |  |

### 9.3 Design Trade-Off.

#### 9.3.1 Solution with Specifying σ\sigma

Using that the number of gradient steps TT is a direct function of the compute C=c​TC=cT, where c=(B+μ​R)c=(B+\mu R), aiming to minimize the right hand-side in convergence bound of Theorem [4.4](#S4.Thmtheorem4 "Theorem 4.4. ‣ 4 Mathematical Analysis ‣ Efficient RL Training for LLMs with Experience Replay") provides design consideration for the buffer parameter RR, BB, NN.
As the compute goes to infinity, we notice that the optimal learning rate (solving a third degree polynomial equation) goes to zero.
As such, the term in κ2​η\kappa^{2}\eta becomes negligible in front of LL asymptotically.
In this case, our analysis suggests to design of the buffer by optimizing for RR, NN and BB in order to minimize

|  |  |  |
| --- | --- | --- |
|  | 𝒥0​(R,B,N,η;μ,σ¯,ρ,κ,C)=12​F​(θ0)​(B+μ​R)C​η+8​L​η​σ¯2​(NR)​(1B+1N+ρR){\mathcal{J}}\_{0}(R,B,N,\eta;\mu,\bar{\sigma},\rho,\kappa,C)=\frac{12F(\theta\_{0})(B+\mu R)}{C\eta}+8L\eta\bar{\sigma}^{2}\big(\frac{N}{R}\big)\left(\frac{1}{B}+\frac{1}{N}+\frac{\rho}{R}\right) |  |

Optimizing in η\eta gives

|  |  |  |
| --- | --- | --- |
|  | 𝒥0​(R,B,N,η∗;μ,σ¯,ρ,κ,C)=4​6​L​F​(θ0)C​(B+μ​R)​σ¯2​(NR)​(1B+1N+ρR).{\mathcal{J}}\_{0}(R,B,N,\eta\_{\*};\mu,\bar{\sigma},\rho,\kappa,C)=\frac{4\sqrt{6}\sqrt{LF(\theta\_{0})}}{\sqrt{C}}\sqrt{(B+\mu R)\bar{\sigma}^{2}\big(\frac{N}{R}\big)\left(\frac{1}{B}+\frac{1}{N}+\frac{\rho}{R}\right)}. |  |

Hence, we can simplify our minimization goal by aiming to minimize

|  |  |  |
| --- | --- | --- |
|  | 𝒥​(R,B,N;μ,σ¯,ρ)=(B+μ​R)​σ¯2​(NR)​(1B+1N+ρR){\mathcal{J}}(R,B,N;\mu,\bar{\sigma},\rho)=(B+\mu R)\bar{\sigma}^{2}\big(\frac{N}{R}\big)\left(\frac{1}{B}+\frac{1}{N}+\frac{\rho}{R}\right) |  |

Let us introduce the staleness horizon x=N/Rx=N/R, which corresponds to the maximum staleness of trajectories in the buffer.

|  |  |  |
| --- | --- | --- |
|  | 𝒥=σ¯2​(x)​(B+μ​R)​(1B+1x​R+ρR){\mathcal{J}}=\bar{\sigma}^{2}(x)(B+\mu R)\Big(\frac{1}{B}+\frac{1}{xR}+\frac{\rho}{R}\Big) |  |

Let us introduce the replay ratio yy, which is the average number of time a sample will be replayed during the SGD trajectory.
Since a sample zz stays for xx iteration in the buffer, that at each iteration BB samples are extracted from the buffer, and that the sampling are independent between steps, the replay ratio is expressed as

|  |  |  |
| --- | --- | --- |
|  | y=NR×𝔼​[∑i∈[B]𝕀​{zt,i=z}]=NR​BN=BR.y=\frac{N}{R}\times{\mathbb{E}}[\sum\_{i\in[B]}\mathbb{I}\{z\_{t,i}=z\}]=\frac{N}{R}\frac{B}{N}=\frac{B}{R}. |  |

Using this ratio, we get

|  |  |  |
| --- | --- | --- |
|  | 𝒥=σ¯2​(x)​(1+y/μ)​(1y+1x+ρ){\mathcal{J}}=\bar{\sigma}^{2}(x)(1+y/\mu)\Big(\frac{1}{y}+\frac{1}{x}+\rho\Big) |  |

We aim to minimize 𝒥​(x,y){\mathcal{J}}(x,y) over the domain 𝒟=(0,+∞)2{\mathcal{D}}=(0,+\infty)^{2}.
Since 𝒥{\mathcal{J}} is continuous, and tends to infinity on the border of the domain 𝒟{\mathcal{D}}, it achieves its minimum for some (x∗,y∗)∈𝒟(x\_{\*},y\_{\*})\in{\mathcal{D}} (not necessarily unique).
Moreover, since 𝒥{\mathcal{J}} is infinitely differentiable on its domain, y∗y\_{\*} is characterized by ∂y𝒥​(y,x∗)=0\partial\_{y}{\mathcal{J}}(y,x\_{\*})=0, which leads to

|  |  |  |
| --- | --- | --- |
|  | 0=(1y∗+1x∗+ρ)/μ−(1+y∗/μ)​1y∗2=(1x∗+ρ)/μ−1y∗2.0=\Big(\frac{1}{y\_{\*}}+\frac{1}{x\_{\*}}+\rho\Big)/\mu-(1+y\_{\*}/\mu)\frac{1}{y\_{\*}^{2}}=\left(\frac{1}{x\_{\*}}+\rho\right)/\mu-\frac{1}{y\_{\*}^{2}}. |  |

Hence,

|  |  |  |
| --- | --- | --- |
|  | y∗=μ(ρ+1x∗).y\_{\*}=\frac{\sqrt{\mu}}{\sqrt{\left(\rho+\frac{1}{x\_{\*}}\right)}}. |  |

Plugging this expression back into 𝒥{\mathcal{J}} gives a one-dimensional objective in xx,

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℐ​(x):=𝒥​(x,μ/(ρ+1/x))\displaystyle{\mathcal{I}}(x):={\mathcal{J}}(x,\sqrt{\mu}/\sqrt{(\rho+1/x)}) | =σ¯2​(x)​(1+1μ​(ρ+1x))​((ρ+1x)/μ+1x+ρ)\displaystyle=\bar{\sigma}^{2}(x)\Big(1+\frac{1}{\sqrt{\mu\left(\rho+\frac{1}{x}\right)}}\Big)\Big(\sqrt{\left(\rho+\frac{1}{x}\right)/\mu}+\frac{1}{x}+\rho\Big) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =σ¯2​(x)​(1μ+ρ+1x)2.\displaystyle=\bar{\sigma}^{2}(x)\left(\frac{1}{\sqrt{\mu}}+\sqrt{\rho+\frac{1}{x}}\right)^{2}. |  |

where the last equality follows from the fact that for a=(ρ+1/x)a=(\rho+1/x),

|  |  |  |
| --- | --- | --- |
|  | (1+1μ​a)​(aμ+a)=aμ+a+1μ+aμ=(1μ+a)2.\big(1+\frac{1}{\sqrt{\mu a}}\big)\big(\sqrt{\frac{a}{\mu}}+a\big)=\sqrt{\frac{a}{\mu}}+a+\frac{1}{\mu}+\sqrt{\frac{a}{\mu}}=\left(\frac{1}{\mu}+\sqrt{a}\right)^{2}. |  |

Hence the remaining design choice is

|  |  |  |
| --- | --- | --- |
|  | x∗∈arg​minx⁡σ¯2​(x)​(1μ+ρ+1x)2.x\_{\*}\in\operatorname\*{arg\,min}\_{x}\;\bar{\sigma}^{2}(x)\Big(\frac{1}{\sqrt{\mu}}+\sqrt{\rho+\frac{1}{x}}\Big)^{2}. |  |

Note that we omitted integers and divisibility constraints, which would leads to a constrained version of the solution provided above.

##### Remark on “Under Specifications”

Note our analysis reduces the parameterization of 𝒥{\mathcal{J}} from three variables (B,N,T)(B,N,T) to only two ratios (x,y)(x,y).
This reduction follows from the homogeneity of 𝒥{\mathcal{J}} under the scaling transformation (B,N,R)↦(α​B,α​N,α​R)(B,N,R)\mapsto(\alpha B,\alpha N,\alpha R) for any α>0\alpha>0.
As a consequence, Theorem [4.5](#S4.Thmtheorem5 "Theorem 4.5 (Optimal Design). ‣ 4 Mathematical Analysis ‣ Efficient RL Training for LLMs with Experience Replay") characterizes optimal ratios rather than prescribing absolute values (e.g., a specific batch size), which may at first appear puzzling.

However, the scale invariance only holds in the asymptotic regime.
Entering this regime requires the number of gradient steps T=C/(B+μ​R)T=C/(B+\mu R) to be sufficiently large, thus imposing an upper bound on B+μ​RB+\mu R for a fixed compute budget CC.
Similarly, integer constraints and divisibility assumptions imposes lower bounds on BB, NN, and RR.
In practice, precise finite-time bound would introduce additional quantities, which would break the homogeneity, and provide clear indications on the optimal batch size.

#### 9.3.2 Closed-Form Solution with Power-Law Variance

Let us specify the variance profile as a power law:

|  |  |  |
| --- | --- | --- |
|  | σ​(x)=(xτ)α\sigma(x)=\left(\frac{x}{\tau}\right)^{\alpha} |  |

for some coefficients τ\tau and α\alpha. Using the integral approximation ∑s=0H−1sp≈Hp+1p+1\sum\_{s=0}^{H-1}s^{p}\approx\frac{H^{p+1}}{p+1}, we compute the average variance σ¯2​(H)\bar{\sigma}^{2}(H):

|  |  |  |
| --- | --- | --- |
|  | σ¯2​(H)=1H​∑s=0H−1(sτ)2​α≈1H​τ2​α​H2​α+12​α+1=12​α+1​(Hτ)2​α.\bar{\sigma}^{2}(H)=\frac{1}{H}\sum\_{s=0}^{H-1}\left(\frac{s}{\tau}\right)^{2\alpha}\approx\frac{1}{H\tau^{2\alpha}}\frac{H^{2\alpha+1}}{2\alpha+1}=\frac{1}{2\alpha+1}\left(\frac{H}{\tau}\right)^{2\alpha}. |  |

Recall that x=N/R=Hx=N/R=H. Substituting this into the design objective ℐ​(x){\mathcal{I}}(x), we aim to minimize

|  |  |  |
| --- | --- | --- |
|  | ℐ​(x)=x2​α(2​α+1)​τ2​α​(1μ+ρ+1x)2.{\mathcal{I}}(x)=\frac{x^{2\alpha}}{(2\alpha+1)\tau^{2\alpha}}\left(\frac{1}{\sqrt{\mu}}+\sqrt{\rho+\frac{1}{x}}\right)^{2}. |  |

Dropping constant multiplicative factors, this is equivalent to minimizing the simplified function

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒦​(x)=xα​(1μ+ρ+1x).{\mathcal{K}}(x)=x^{\alpha}\left(\frac{1}{\sqrt{\mu}}+\sqrt{\rho+\frac{1}{x}}\right). |  | (5) |

The first-order optimality condition 𝒦′​(x)=0{\mathcal{K}}^{\prime}(x)=0 yields

|  |  |  |
| --- | --- | --- |
|  | α​xα−1​(1μ+ρ+1x)+xα​12​ρ+1x​(−1x2)=0.\alpha x^{\alpha-1}\left(\frac{1}{\sqrt{\mu}}+\sqrt{\rho+\frac{1}{x}}\right)+x^{\alpha}\frac{1}{2\sqrt{\rho+\frac{1}{x}}}\left(-\frac{1}{x^{2}}\right)=0. |  |

Multiplying by 2​x2−α​ρ+1/x2x^{2-\alpha}\sqrt{\rho+1/x}, we obtain the algebraic equation

|  |  |  |
| --- | --- | --- |
|  | 2​α​x​(1μ​ρ+1x+ρ+1x)=1⇔2​α​ρ​x2+xμ=1−2​α−2​α​ρ​x.2\alpha x\left(\frac{1}{\sqrt{\mu}}\sqrt{\rho+\frac{1}{x}}+\rho+\frac{1}{x}\right)=1\iff 2\alpha\sqrt{\frac{\rho x^{2}+x}{\mu}}=1-2\alpha-2\alpha\rho x. |  |

Squaring both sides leads to a quadratic equation A​x2+B​x+C=0Ax^{2}+Bx+C=0 governing the optimal staleness x∗x\_{\*}:

|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 4​α2​(ρ​x2+x)/μ=(1−2​α−2​α​ρ​x)2\displaystyle 4\alpha^{2}(\rho x^{2}+x)/\mu=(1-2\alpha-2\alpha\rho x)^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⇔\displaystyle\iff | 4​α2​ρ​(1/μ−ρ)⏟A​x2+4​α​(α/μ+ρ​(1−2​α))⏟B​x​−(1−2​α)2⏟C=0.\displaystyle\underbrace{4\alpha^{2}\rho(1/\mu-\rho)}\_{A}x^{2}+\underbrace{4\alpha(\alpha/\mu+\rho(1-2\alpha))}\_{B}x\underbrace{-(1-2\alpha)^{2}}\_{C}=0. |  |

Solving for the positive root, and noting that the discriminant Δ=B2−4​A​C\Delta=B^{2}-4AC simplifies to Δ=16​α2​μ​(α2/μ+ρ​(1−2​α))\Delta=16\alpha^{2}\mu(\alpha^{2}/\mu+\rho(1-2\alpha)), the optimal staleness horizon is given explicitly by

|  |  |  |
| --- | --- | --- |
|  | x∗=−(α/μ+ρ​(1−2​α))+α2/μ2+ρ​(1−2​α)/μ2​α​ρ​(1/μ−ρ).x\_{\*}=\frac{-(\alpha/\mu+\rho(1-2\alpha))+\sqrt{\alpha^{2}/\mu^{2}+\rho(1-2\alpha)/\mu}}{2\alpha\rho(1/\mu-\rho)}. |  |

To find the optimal replay ratio y∗y\_{\*}, we avoid substituting the complex closed-form of x∗x\_{\*} and instead exploit the optimality conditions directly. Recall the relationship characterizing y∗y\_{\*}:

|  |  |  |
| --- | --- | --- |
|  | y∗=1(ρ+1/x∗)/μ⟹ρ+1x∗=μy∗2.y\_{\*}=\frac{1}{\sqrt{(\rho+1/x\_{\*})/\mu}}\implies\rho+\frac{1}{x\_{\*}}=\frac{\mu}{y\_{\*}^{2}}. |  |

This allows us to express the staleness x∗x\_{\*} strictly as a function of y∗y\_{\*}:

|  |  |  |
| --- | --- | --- |
|  | 1x∗=μy∗2−ρ=μ−ρ​y∗2y∗2⟹x∗=y∗2μ−ρ​y∗2.\frac{1}{x\_{\*}}=\frac{\mu}{y\_{\*}^{2}}-\rho=\frac{\mu-\rho y\_{\*}^{2}}{y\_{\*}^{2}}\implies x\_{\*}=\frac{y\_{\*}^{2}}{\mu-\rho y\_{\*}^{2}}. |  |

Substituting the term ρ+1/x∗=μy∗\sqrt{\rho+1/x\_{\*}}=\frac{\sqrt{\mu}}{y\_{\*}} into the first-order optimality condition derived for x∗x\_{\*}:

|  |  |  |
| --- | --- | --- |
|  | 1=2​α​x∗​(1μ​ρ+1x∗+ρ+1x∗)=2​α​x∗​(1y∗+μy∗2).1=2\alpha x\_{\*}\left(\frac{1}{\sqrt{\mu}}\sqrt{\rho+\frac{1}{x\_{\*}}}+\rho+\frac{1}{x\_{\*}}\right)=2\alpha x\_{\*}(\frac{1}{y\_{\*}}+\frac{\mu}{y\_{\*}^{2}}). |  |

Simplifying the term in the parenthesis yields =1=1, or equivalently:

|  |  |  |
| --- | --- | --- |
|  | x∗=y∗22​α​(μ+y∗).x\_{\*}=\frac{y\_{\*}^{2}}{2\alpha(\mu+y\_{\*})}. |  |

Equating the two characterization of x∗x\_{\*} gives

|  |  |  |
| --- | --- | --- |
|  | y∗2μ−ρ​y∗2=y∗22​α​(μ+y∗)⇔μ−ρ​y∗2=2​α​(μ+y∗).\frac{y\_{\*}^{2}}{\mu-\rho y\_{\*}^{2}}=\frac{y\_{\*}^{2}}{2\alpha(\mu+y\_{\*})}\iff\mu-\rho y\_{\*}^{2}=2\alpha(\mu+y\_{\*}). |  |

Rearranging terms yields a quadratic equation in y∗y\_{\*}:

|  |  |  |
| --- | --- | --- |
|  | ρ​y∗2+2​α​y∗+μ​(2​α−1)=0.\rho y\_{\*}^{2}+2\alpha y\_{\*}+\mu(2\alpha-1)=0. |  |

Assuming α<1/2\alpha<1/2, the constant term μ​(2​α−1)\mu(2\alpha-1) is negative, guaranteeing a unique positive solution:

|  |  |  |
| --- | --- | --- |
|  | y∗=−α+α2+μ​ρ​(1−2​α)ρ.y\_{\*}=\frac{-\alpha+\sqrt{\alpha^{2}+\mu\rho(1-2\alpha)}}{\rho}. |  |

An illustration of the formula for x∗x\_{\*} and y∗y\_{\*} is provided in Figure [6](#S9.F6 "Figure 6 ‣ 9.3.2 Closed-Form Solution with Power-Law Variance ‣ 9.3 Design Trade-Off. ‣ 9 Mathematical Details ‣ Efficient RL Training for LLMs with Experience Replay"), and the function x↦𝒦​(x)x\mapsto{\mathcal{K}}(x) (as well as the optimum value x∗x\_{\*}) is shown in Figure [7](#S9.F7 "Figure 7 ‣ 9.3.2 Closed-Form Solution with Power-Law Variance ‣ 9.3 Design Trade-Off. ‣ 9 Mathematical Details ‣ Efficient RL Training for LLMs with Experience Replay").

!(/html/2604.08706/assets/x10.png)

!(/html/2604.08706/assets/x11.png)

!(/html/2604.08706/assets/x12.png)

Figure 6: Optimal Staleness and Replay Ratio as a function of Rollout Cost (μ\mu).
As μ\mu increase, we see that it is better to increase the staleness horizon x∗=N/Rx\_{\*}=N/R, and the replay ratio (y∗=B/R)y\_{\*}=B/R). This also the case when the variance α\alpha or the correlation ρ\rho decreases.

!(/html/2604.08706/assets/x13.png)

Figure 7: Function x↦𝒦​(x)x\mapsto{\mathcal{K}}(x) (which is defined in Eq. ([5](#S9.E5 "Equation 5 ‣ 9.3.2 Closed-Form Solution with Power-Law Variance ‣ 9.3 Design Trade-Off. ‣ 9 Mathematical Details ‣ Efficient RL Training for LLMs with Experience Replay")) and corresponds to 𝒥{\mathcal{J}} for the specific choice σ​(x)=(x/τ)α\sigma(x)=(x/\tau)^{\alpha}) as a function of the staleness horizon x=N/Rx=N/R, for different values of α∈[0,1/2]\alpha\in[0,1/2], and the corresponding optimal values of x∗x\_{\*}.

## 10 Experimental details

We provide additional details regarding our experimental setup.

### 10.1 Hardware and parallelism

We use Nvidia H100 and H200 GPUs.

Our experiments are run on either 1,21,2 or 44 8-GPUs nodes, with data parallelism and without tensor parallelism. When describing a buffer experiment ran on more than 11 node, we report TT and WW divided by the number of nodes; in other words, we describe an experiment run on 16 GPUs with 44 trainer GPUs and 1212 inference GPUs as (6,2)(6,2) rather than (12,4)(12,4). We do so to simplify notations, and because increasing the number of nodes while keeping the same ratio W/TW/T does not impact any of the relevant quantities (size of the buffer, replay ratio, off-policiness): the training dynamics remain the same (up to essentially random effects linked to inter-nodes communications), and the training is accelerated with respect to wall-time, which we do not take into account when estimating compute (in other words, we consider that the cost of a gradient step is not affected by the number of nodes).

Our non-buffer experiments are run with (W,T)∈{(4,4),(5,3),(6,2)}(W,T)\in\{(4,4),(5,3),(6,2)\}: though we find that the theoretical optimal ratio μ\mu is closer to W/T=5W/T=5, ratios closer to 11 are in practice better when training on a small number of GPUs (e.g. 88 or 1616). This is because letting TT be very small (e.g. T∈{1,2}T\in\{1,2\}) forces the maximum micro-batch size to also be very small, while large micro-batch sizes are needed to leverage parallelism-based optimizations.

### 10.2 Optimization and general hyperparameters

We train using the Adam (kingma2014adam) optimizer with constant learning rates.
We use a batch size of 6060, except in the few runs for which T=7T=7, for which we let the batch size be 6363 (as it must be divisible by the number of trainer GPUs).
Unless otherwise specified, we use a learning rate of 6.8⋅10−86.8\cdot 10^{-8} for Qwen2.5-7B and of 3.37⋅10−​73.37\cdot 10^{-}{7} for Qwen3-0.6B.

We use the following GRPO implementation (see shao2024deepseekmath):

|  |  |  |
| --- | --- | --- |
|  | 𝒥G​R​P​O​(θ)=𝔼q∼𝒟,z​[min⁡(πθ​(z|q)πθo​l​d​(z|q)​A,clip​(πθ​(z|q)πθo​l​d​(z|q),1−εlow,1+εhigh)​A)],\mathcal{J}\_{GRPO}(\theta)=\mathbb{E}\_{q\sim\mathcal{D},z}\Big[\min\left(\frac{\pi\_{\theta}(z|q)}{\pi\_{\theta\_{old}}(z|q)}A,\text{clip}\left(\frac{\pi\_{\theta}(z|q)}{\pi\_{\theta\_{old}}(z|q)},1-\varepsilon\_{\rm{low}},1+\varepsilon\_{\rm{high}}\right)A\right)\Big], |  |

where qq is a prompt sampled from a training distribution 𝒟\mathcal{D} and zz is a rollout sampled from the buffer following the chosen sampling strategy. Both the probability πθo​l​d​(z|q)\pi\_{\theta\_{old}}(z|q) and the advantage AA of zz are computed at the time when zz is first generated. More specifically, a group of GG rollouts z1,…,zGz\_{1},\ldots,z\_{G} is generated by the inference workers for each prompt qq, and the advantage AiA\_{i} of ziz\_{i} is defined as

|  |  |  |  |
| --- | --- | --- | --- |
|  | Ai=ri−mean​({r​(z1,q),r​(z2,q),⋯,r​(zG,q)})std​({r​(z1,q),r​(z2,q),⋯,r​(zG,q)}).A\_{i}=\frac{r\_{i}-{\mathrm{mean}(\{r(z\_{1},q),r(z\_{2},q),\cdots,r(z\_{G},q)\})}}{{\mathrm{std}(\{r(z\_{1},q),r(z\_{2},q),\cdots,r(z\_{G},q)\})}}. |  | (6) |

In other words, the advantage is computed when the rollout is generated (and not when it is used to compose a gradient update).

In particular, we do not include a KL regularization term, as recent research suggests that it does not improve performance (see e.g. yu2025dapoopensourcellmreinforcement). We let εlow=εhigh=0.2\varepsilon\_{\rm{low}}=\varepsilon\_{\rm{high}}=0.2, and we let the group size GG be equal to 1616.
Note that when this loss is combined with a buffer, it can be shown that the joint distribution over the current training batch (which is assembled by sampling from the replay buffer) is not corrected in expectation by the importance sampling factor πθ​(z|q)πθo​l​d​(z|q)\frac{\pi\_{\theta}(z|q)}{\pi\_{\theta\_{old}}(z|q)} (even without taking the clipping into account).

We also consider the AsymRE objective function from arnal2025asymmetricreinforceoffpolicyreinforcement, expressed using the same notations as

|  |  |  |
| --- | --- | --- |
|  | JA​s​y​m​R​E​(θ)=𝔼q∼𝒟,z​[1G​∑i=1G(r​(z,q)−(V^+δ​V))​log⁡(πθ​(z|q))],J\_{AsymRE}(\theta)=\mathbb{E}\_{q\sim\mathcal{D},z}\Big[\frac{1}{G}\sum\_{i=1}^{G}(r(z,q)-(\hat{V}+\delta V))\log(\pi\_{\theta}(z|q))\Big], |  |

where V^:=mean​({r​(z1,q),r​(z2,q),⋯,r​(zG,q)})\hat{V}:={\mathrm{mean}(\{r(z\_{1},q),r(z\_{2},q),\cdots,r(z\_{G},q)\})} if z1,…,zGz\_{1},\ldots,z\_{G} is the group of generated rollouts to which zz belongs (see above) and δ​V=−0.1\delta V=-0.1.

We train Qwen3-0.6B without weight tying.

We use a temperature of 11 when generating training samples, of 0.10.1 when evaluating pass@1 (with top\_p = 0.95), and of 11 when evaluating pass@k with k>1k>1 (with top\_p = 0.95).

### 10.3 Metrics

##### Compute

Our abstract measure of compute is in closest correspondence to the notion of FLOPS, but we make throughout the text the following implicit assumptions, which are never completely realized in practice:

* •

  We are in an optimized settings in which there is a direct correspondence between FLOPS and GPU work time, except when a GPU is idle because it is waiting on the work of other GPUs,
* •

  Tasks can be continuously parallelized; in other words, there are no boundaries effects due to the discrete nature of the number of samples and GPUs, and
* •

  When parallelizing a task between KK GPUs, the total compute spent is not a function of KK.

In particular, we ignore the effects of important implementation details, such as tensor parallelism, data parallelism, sharding, etc.

*Steps-since-last-use*
We define in greater detail the steps-since-last-use metric reported in Figure [2](#S3.F2 "Figure 2 ‣ 3.1 Reinforcement Learning and Replay Buffers ‣ 3 Experience Replay for Off-Policy RL ‣ Efficient RL Training for LLMs with Experience Replay").
In the context of this paragraph, we use the term "rollout" to refer to a given data point, and the term "sample" to refer to a data point as it appears in a gradient descent batch.
Each rollout (a given sequence of tokens) can correspond to zero, one or several samples belonging to one or several batches depending on how often it was sampled from the buffer.

* •

  We order all samples used during a training trajectory:

  + –

    For every batch BB, we pick a random ordering of the samples of BB.
  + –

    If batch BB was processed before batch B′B^{\prime}, then z<z′z<z^{\prime} for any z∈B,z′∈B′z\in B,z^{\prime}\in B^{\prime}.
* •

  Whenever a rollout appears as a sample for the first time according to this global order, we associate the value "new" to the sample.
* •

  If a sample zz corresponds to a rollout that has already given rise to an earlier sample z′z^{\prime}, then we associate to zz the number of gradient steps taken since z′z^{\prime}.

As an illustration, let us assume that a rollout gives rise to exactly four samples: z1∈B3z\_{1}\in B\_{3} and z2,z3,z4∈B5z\_{2},z\_{3},z\_{4}\in B\_{5}, where the numbering of the samples reflect their ordering and the batch BiB\_{i} was used at time ii. Then z1z\_{1} is mapped to "new", z2z\_{2} to 22, z3z\_{3} to 0 and z4z\_{4} to 0.
In Figure [2](#S3.F2 "Figure 2 ‣ 3.1 Reinforcement Learning and Replay Buffers ‣ 3 Experience Replay for Off-Policy RL ‣ Efficient RL Training for LLMs with Experience Replay"), we plot the histogram of the values taken by steps-since-last over all samples of each trajectory considered.

*Pass@k*
The pass@k curve from Figure [3](#S5.F3 "Figure 3 ‣ 5.1 Experimental setup ‣ 5 Experimental results ‣ Efficient RL Training for LLMs with Experience Replay") is computed as follows: for a given kk and a given choice of hyperparameters (with or without buffer, etc.), the median over the random seeds of the pass@k training curve is computed. We then report the maximum of this median curve over the training trajectory, as well as the IQR at the step where the maximum is reached.
In particular, the corresponding training step is in general not the same for distinct choices of kk.

### 10.4 Buffer-specific aspects

*Compute ratio γ\gamma*

To estimate the compute cost of a parameter update in a buffer configuration with TT trainer GPUs and WW inference GPUs, we use the compute ratio

|  |  |  |
| --- | --- | --- |
|  | γ=1+W/T1+μ,\gamma=\frac{1+W/T}{1+\mu}, |  |

defined in Equation ([2](#S3.E2 "Equation 2 ‣ Compute Efficiency ‣ 3.2 Off-Policiness, Diversity, and Compute Efficiency ‣ 3 Experience Replay for Off-Policy RL ‣ Efficient RL Training for LLMs with Experience Replay")).
This quantity depends in turn on the optimal ratio μ\mu, defined as the compute cost of generating a rollout divided by the compute cost of processing it through a gradient update; equivalently, μ\mu is the exact number of inference GPUs for each trainer GPUs required so that there is no downtime.

This quantity depends on the model, dataset, implementation details and hardware, as well as on some parameter choices (such as the batch size).
Consider a training run using a replay buffer, and the following quantities:

* •

  KtrainingK\_{\rm training} the number of non-unique rollouts processed through backpropagation over the entire run, i.e. the number of gradient steps multiplied by the batch size,
* •

  KinferenceK\_{\rm inference} the number of unique rollouts generated by the inference GPUs over the entire run,
* •

  TT the number of trainer GPUs, and
* •

  WW the number of inference GPUs.

Each trainer GPU will have processed Ktraining/TK\_{\rm training}/T rollouts, and each inference GPU will have generated Kinference/WK\_{\rm inference}/W rollouts on average.
As inference and trainer GPUs work independently from each other when using a replay buffer and do not suffer any downtime, we can consider that this is a fair measure of their relative speed (or equivalently of the relative compute cost of training vs inference), and use it to estimate μ\mu:

|  |  |  |
| --- | --- | --- |
|  | μ≅Ktraining/TKinference/W.\mu\cong\frac{K\_{\rm training}/T}{K\_{\rm inference}/W}. |  |

To make this estimate more precise, we use the median value over several random seeds.

We report in the table below our estimates of the coefficients μ\mu for the various models featured in our experiments.

| Model | Median μ\mu | IQR | # Independent runs |
| --- | --- | --- | --- |
| Qwen3-0.6B | 6.846.84 | [6.45,7.07][6.45,7.07] | 242242 |
| Qwen2.5-7B | 5.285.28 | [5.12,5.48][5.12,5.48] | 129129 |

Table 2: Estimates of μ\mu for various models, computed as the median over several independent runs. We also provide the interquartile range over those runs.

We provide in Table [3](#S10.T3 "Table 3 ‣ 10.4 Buffer-specific aspects ‣ 10 Experimental details ‣ Efficient RL Training for LLMs with Experience Replay") the γ\gamma values corresponding to Qwen3-0.6B.

| (W, T) | (7,1) | (6,2) | (5,3) | (4,4) | (2,6) | (1,7) |
| --- | --- | --- | --- | --- | --- | --- |
| γ\gamma | 1.021.02 | 0.410.41 | 0.340.34 | 0.260.26 | 0.170.17 | 0.150.15 |

Table 3: γ\gamma for various values of (W,T)(W,T) and an estimated μ=6.84\mu=6.84 for Qwen3-0.6B.

*Sharded buffers*
In our concrete implementation, each trainer GPU maintains its own separate replay buffer.
In other words, newly generated rollouts are added to the replay buffers of the various trainer GPUs (each a list of trajectories) in a balanced way, with each rollout being added to the replay buffer of a single trainer GPU.
Each trainer GPU, when creating a sub-batch from which it will compute a gradient (which will then be averaged with the gradients of other trainer GPUs), samples only from its own buffer.
We always report the total buffer size NN, which is the sum over the TT trainer GPUs of the sizes N/TN/T of their separate buffers.
Our preliminary experiments suggest that this design choice has little impact.

*Positive-bias sampling* We introduced in Section [5](#S5 "5 Experimental results ‣ Efficient RL Training for LLMs with Experience Replay") an alternative buffer strategy, which we call positive-bias sampling: instead of keeping the freshest NN generated rollouts in the buffer, we keep the freshest (1−δ)​N(1-\delta)N generated rollouts along with the freshest δ​N\delta N correct rollouts not included in those (1−δ)​N(1-\delta)N trajectories.
As an example, if N=8N=8, δ=0.75\delta=0.75 and the last rollouts to be produced are

|  |  |  |
| --- | --- | --- |
|  | …​zt−9​zt−8​zt−7​zt−6​zt−5​zt−4​zt−3​zt−2​zt−1​zt,\ldots{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}z\_{t-9}}\>{\color[rgb]{0,1,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,1,0}z\_{t-8}}\>{\color[rgb]{0,1,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,1,0}z\_{t-7}}\>{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}z\_{t-6}}\>{\color[rgb]{0,1,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,1,0}z\_{t-5}}\>{\color[rgb]{0,1,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,1,0}z\_{t-4}}\>{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}z\_{t-3}}\>{\color[rgb]{0,1,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,1,0}z\_{t-2}}\>{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}z\_{t-1}}\>{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}z\_{t}}, |  |

where incorrect and correct rollouts are shown in red and green respectively, then the buffer at time tt is equal to

|  |  |  |
| --- | --- | --- |
|  | zt−8​zt−7​zt−5​zt−4​zt−3​zt−2​zt−1​ztz\_{t-8}\>z\_{t-7}\>z\_{t-5}\>z\_{t-4}\>z\_{t-3}\>z\_{t-2}\>z\_{t-1}\>z\_{t} |  |

## 11 Additional Experimental Results

We provide various additional experimental results:

* •

  In Figures [8](#S11.F8 "Figure 8 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay") and [9](#S11.F9 "Figure 9 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay"), we run ablations to select the best learning rate for Qwen3-0.6B and Qwen2.5-7B. We see that 3.37⋅10−73.37\cdot 10^{-7}, respectively 6.810−86.810^{-8}, achieve the best balance between speed and stability.
* •

  We report accuracy with respect to wall-time for Qwen3-0.6B and Qwen2.5-7B in Figures [10](#S11.F10 "Figure 10 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay") and [11](#S11.F11 "Figure 11 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay").
* •

  We study the impact of off-policiness in no-buffer configurations in Figure [12](#S11.F12 "Figure 12 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay").
* •

  We report accuracy with respect to compute and training dynamics for Qwen3-0.6B with various buffer sizes and W/TW/T ratios in [13](#S11.F13 "Figure 13 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay"). We also report the best accuracies achieved and the corresponding compute costs in Figure [14](#S11.F14 "Figure 14 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay").
* •

  We test our methods on other models and tasks in Figures[16](#S11.F16 "Figure 16 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay") (Qwen3-8B on Lean coding tasks) and [17](#S11.F17 "Figure 17 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay") (Llama 3.2 3B on OpenR1-Math-220k).
* •

  In complement to Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Efficient RL Training for LLMs with Experience Replay"), we report in Figure [15](#S11.F15 "Figure 15 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay") the results for additional buffer configurations for Qwen2.5-7B.
* •

  We show that alternative sampling strategies based on sampling without replacement do not have a clear impact in Figure [18](#S11.F18 "Figure 18 ‣ 11 Additional Experimental Results ‣ Efficient RL Training for LLMs with Experience Replay").

!(/html/2604.08706/assets/x14.png)

!(/html/2604.08706/assets/x15.png)

!(/html/2604.08706/assets/x16.png)

Figure 8: Learning Rate Ablations for Qwen3-0.6B. Test accuracy as a function of the number of steps when training Qwen3-0.6B on OpenR1-Math-220k with various learning rates LR with at least 44 seeds per configuration. We show the median and IQR over the seeds on the left, and all seeds separately on the right.

!(/html/2604.08706/assets/x17.png)

!(/html/2604.08706/assets/x18.png)

!(/html/2604.08706/assets/x19.png)

Figure 9: Learning Rate Ablations for Qwen2.5-7B. Test accuracy on MATH as a function of the number of steps when training Qwen2.5-7B on OpenR1-Math-220k with various learning rates LR with at least 44 seeds per configuration. We show the median and IQR over the seeds on the left, and all seeds separately on the right. Note the frequent crashes when LR>6.8⋅10−8\text{LR}>6.8\cdot 10^{-8}.

!(/html/2604.08706/assets/x20.png)

Figure 10: Wall-time efficiency for Qwen3-0.6B
Test accuracy as a function of wall-time when training Qwen3-0.6B on OpenR1-Math-220k for the no-buffer baseline (orange curve) and various buffer configurations. We report the median and the IQR over at least 44 seeds per curve.

!(/html/2604.08706/assets/x21.png)

Figure 11: 
Wall-time efficiency for Qwen2.5-7B
Test accuracy on MATH as a function of wall-time when training Qwen2.5-7B on OpenR1-Math-220k for the no-buffer baseline (orange curve) and various buffer configurations. We report the median and the IQR over at least 44 seeds per curve.

!(/html/2604.08706/assets/x22.png)

Figure 12: Impact of off-policiness.
We train Qwen3-0.6B on OpenR1-Math-220k without a buffer and we artificially introduce various levels of off-policiness by reducing the frequency at which the model’s weights used by the inference workers to generate rollouts are updated. We label each curve with the median level of off-policiness over all rollouts used, and plot the median test accuracy and its IQR as a function of the number of training steps over at least 44 seeds per curve.

!(/html/2604.08706/assets/x23.png)

!(/html/2604.08706/assets/x24.png)

!(/html/2604.08706/assets/x25.png)

!(/html/2604.08706/assets/x26.png)

!(/html/2604.08706/assets/x27.png)

!(/html/2604.08706/assets/x28.png)

!(/html/2604.08706/assets/x29.png)

!(/html/2604.08706/assets/x30.png)

!(/html/2604.08706/assets/x31.png)

!(/html/2604.08706/assets/x32.png)

Figure 13: Test, Train and Entropy Dynamics.
We train Qwen3-0.6B on OpenR1-Math-220k with a buffer for (W,T)∈{(6,2),(5,3),(4,4)}(W,T)\in\{(6,2),(5,3),(4,4)\} and various buffer sizes.
We report the test accuracy (top), the training accuracy (middle, smoothed using a sliding window), and the training entropy (bottom) as a function of the number of training steps. Note that the training entropy is computed over the batches used by the trainers to compute gradient updates; as using a buffer implies reusing samples generated by outdated policies, it is expected that the reported entropy would be much higher.
We also report two baseline curves, corresponding to non-buffer configurations: one is plotted with respect to the number of steps, while the other is rescaled to be at compute-parity with the buffer configurations (i.e. so that an x-axis unit represents the same amount of compute).
Each curve is the median (along with its IQR) over at least 44 seeds.

!(/html/2604.08706/assets/x33.png)

Figure 14: Accuracy and Speed with respect to Design Choices.
We train Qwen3-0.6B on OpenR1-Math-220k for (W,T)∈{(6,2),(5,3),(4,4)(W,T)\in\{(6,2),(5,3),(4,4) and various buffer sizes.
We report on the right the median best test accuracy achieved over each training run (in other words, the median over the seeds of the best accuracy achieved for each seed), and on the left the median amount of compute that was needed to first reach 98%98\% of that score.

!(/html/2604.08706/assets/x34.png)

!(/html/2604.08706/assets/x35.png)

Figure 15: Additional results for Qwen2.5-7B. Accuracy on MATH as a function of compute spent when training Qwen2.5-7B on OpenR1-Math-220k for the no-buffer baseline (orange curve) and a buffer of size N∈{84,512,2268,20412}N\in\{84,512,2268,20412\} with (W,T)(W,T) equal to (6,2)(6,2) (left) or (5,3)(5,3) (right). We report the median and IQR over more than 44 seeds. Compute is calibrated so that a single weight update for the baseline costs 11 unit.

!(/html/2604.08706/assets/x36.png)
  

Figure 16: Accuracy with respect to Buffer Size for Qwen3-8B on Lean coding tasks. Test accuracy as a function of compute spent when training Qwen3-8B on miniF2F for (W,T)=(6,2)(W,T)=(6,2) and various buffer sizes N∈{128,10000}N\in\{128,10000\}, as well as for a no-buffer baseline. We report the median and IQR over 44 seeds. Compute is normalized so that each weight update costs 0.550.55 unit for buffer configurations and 11 for the baseline.

!(/html/2604.08706/assets/x37.png)
  

Figure 17: Accuracy with respect to Buffer Size for Llama 3.2 3B. Test accuracy as a function of compute spent when training Llama 3.2 3B on OpenR1-Math-220k for (W,T)=(6,2)(W,T)=(6,2) and various buffer sizes N∈{128,2048,16384}N\in\{128,2048,16384\}, as well as for a no-buffer baseline. We report the median and IQR over 44 seeds. Compute is normalized so that each weight update costs 0.580.58 unit for buffer configurations and 11 for the baseline.

!(/html/2604.08706/assets/x38.png)

Figure 18: 
We compared our standard buffer implementation, in which the buffer is sampled uniformly by the trainer GPUs ("vanilla"), with two variants: one in which the sampling is done uniformly without replacement ("No replacement"), and one in which samples that have never been used are sampled in priority, after what the remainder of the batch is filled without replacement ("No replacement, at least once").
We did not find any strong signal, as exemplified by these two representative buffer configurations (with which we trained Qwen3-0.6B on OpenR1-Math-220k).
