---
arxiv: '2509.19128'
authors:
- Alexandre Piché
- Ehsan Kamalloo
- Rafael Pardinas
- Xiaoyin Chen
- Dzmitry Bahdanau
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation'
url: https://arxiv.org/abs/2509.19128
year: 2025
---

[2509.19128] PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation














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



# PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation

Alexandre Piché alexandrelpiche@gmail.com
  
ServiceNow AI Research
Ehsan Kamalloo
  
ServiceNow AI Research
Rafael Pardinas
  
ServiceNow AI Research
Xiaoyin Chen
  
Mila, Université de Montréal
Dzmitry Bahdanau
  
ServiceNow AI Research
  
Mila, McGill University
  
Canada CIFAR AI Chair
Currently affiliated with another institution.

###### Abstract

Reinforcement Learning (RL) is increasingly utilized to enhance the reasoning capabilities of Large Language Models (LLMs). However, effectively scaling these RL methods presents significant challenges, primarily due to the difficulty in maintaining high AI accelerator utilization without generating stale, off-policy data that harms common RL algorithms. This paper introduces PipelineRL, an approach designed to achieve a superior trade-off between hardware efficiency and data on-policyness for LLM training. PipelineRL employs concurrent asynchronous data generation and model training, distinguished by the novel in-flight weight updates. This mechanism allows the LLM generation engine to receive updated model weights with minimal interruption during the generation of token sequences, thereby maximizing both the accelerator utilization and the freshness of training data. Experiments conducted on long-form reasoning tasks using 128 H100 GPUs demonstrate that PipelineRL achieves approximately ∼2​x\sim 2x faster learning compared to conventional RL baselines while maintaining highly on-policy training data. A scalable and modular open-source implementation of PipelineRL is also released as a key contribution.

## 1 Introduction

Reinforcement Learning (RL) has recently become a popular tool to enhance the reasoning and agentic capabilities of Large Language Models (LLMs) (Guo et al., [2025](#bib.bib5); Wei et al., [2025](#bib.bib26)). While RL expands the range of training signals one can use to enhance LLMs, this advanced learning paradigm comes with extra challenges, including being particularly hard to effectively scale to more compute. The scaling difficulty arises from the fact that AI accelerators (like GPUs and TPUs) deliver high throughput only when generating sequences at a large batch size. Hence, naively adding more accelerators to an on-policy RL setup brings increasingly diminishing learning speed improvements because the per-accelerator throughput decreases, while the overall generation latency reaches a plateau. The common workaround of generating training data for multiple optimizer steps results in a lag between the currently trained policy and the behavior policy that generates the training data. The lagging off-policy data is known to harm the commonly used effective RL algorithms (Noukhovitch et al., [2024](#bib.bib16)), including, REINFORCE (Williams, [1992](#bib.bib27)), PPO (Schulman et al., [2017](#bib.bib23)) and GRPO (Shao et al., [2024](#bib.bib24); Guo et al., [2025](#bib.bib5)), because these algorithms were designed to be trained with on-policy or near on-policy data, with the behavior and current policy being very close.

In this paper, we present the PipelineRL approach to RL for LLMs that achieves a better trade-off between hardware utilization and on-policy learning. Like prior work on efficient RL (Espeholt et al., [2018](#bib.bib2); [2019](#bib.bib3)), PipelineRL features concurrent asynchronous data generation and training. PipelineRL adapts prior asychronous RL ideas to long-sequence generation with LLMs by introducing *in-flight weight updates*. As shown in [Figure˜1](#S2.F1 "In 2 Background ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation"), during an in-flight weight update the LLM generation engine only briefly pauses to receive the model weights via a high-speed inter-accelerator network, and then proceeds to continue the generation of in-progress token sequences. In-flight updates eliminate the wasteful waits for the last sequence to finish, ensure high accelerator utilization at a constant generation batch size, and maximize the policy adherence of the recently generated tokens.

Our experiments on RL training for long-form reasoning show that on 4 DGX-H100 nodes, PipelineRL learns ∼2​x\sim 2x faster than the comparable conventional RL baseline. We also observe that PipelineRL training data stays highly on-policy, and that models trained by PipelineRL perform comparably to similarly trained models from the literature. Lastly, a key contribution of this work is a scalable and modular PipelineRL implementation that we release as open-source software.111<https://github.com/ServiceNow/pipelinerl>

## 2 Background

![Refer to caption](/html/2509.19128/assets/figures/inflight_wide.jpg)


Figure 1: a) Conventional RL alternates between using all the GPUs for generation and then training. b) PipelineRL runs generation and training concurrently, always using the freshest model weights for generations thanks to the in-flight weight updates.

### 2.1 Reinforcement Learning for Large Language Models

Reinforcement learning (RL) is commonly used to train Large Language Models (LLM) to respect human preferences (Ouyang et al., [2022](#bib.bib18)) for the LLM’s outputs or to perform long-form reasoning to solve problems (Guo et al., [2025](#bib.bib5)). One can view LLM’s weights as parameterizing
a multi-step policy that assigns probabilities to the next token yiy\_{i} given the prompt xx and the previously generated tokens y<iy\_{<i}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | π​(y|x)\displaystyle\pi(y|x) | =∏i=1nπ​(yi|x,y<i).\displaystyle=\prod\_{i=1}^{n}\pi(y\_{i}|x,y\_{<i}). |  | (1) |

Recent works have shown that variations of basic policy gradient algorithms such as REINFORCE (Williams, [1992](#bib.bib27)) are as effective for training LLMs as more sophisticated alternatives (Ahmadian et al., [2024](#bib.bib1); Roux et al., [2025](#bib.bib21)). Given a set of prompts x1,…,xmx\_{1},\ldots,x\_{m}, REINFORCE maximizes the expected return J​(π)J(\pi) of the policy π\pi by following an estimate ∇~​J​(π)\tilde{\nabla}J(\pi) of the policy gradient ∇J​(π)\nabla J(\pi):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | J​(π)\displaystyle J(\pi) | =1m​∑j=1m[𝔼y∼π(⋅|xj)​R​(xj,y)]\displaystyle=\frac{1}{m}\sum\_{j=1}^{m}\left[\mathbb{E}\_{y\sim\pi(\cdot|x\_{j})}R(x\_{j},y)\right] |  | (2) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∇J​(π)\displaystyle\nabla J(\pi) | =1m​∑j=1m[𝔼y∼π(⋅|xj)​∇log⁡π​(y∣xj)​R​(xj,y)]\displaystyle=\frac{1}{m}\sum\_{j=1}^{m}\left[\mathbb{E}\_{y\sim\pi(\cdot|x\_{j})}\nabla\log\pi(y\mid x\_{j})R(x\_{j},y)\right] |  | (3) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∇~​J​(π)\displaystyle\tilde{\nabla}J(\pi) | =1m​∑j=1m∑t=1Tj(R​(xj,yj)−vϕ​(xj,yj,≤t))​∇log⁡π​(yj,t∣xj,yj,<t),\displaystyle=\frac{1}{m}\sum\_{j=1}^{m}\sum\_{t=1}^{T\_{j}}\left(R(x\_{j},y\_{j})-v\_{\phi}(x\_{j},y\_{j,\leq t})\right)\nabla\log\pi(y\_{j,t}\mid x\_{j},y\_{j,<t}), |  | (4) |

where R​(xj,y)R(x\_{j},y) is the reward and vϕ​(xj,yj,≤t)v\_{\phi}(x\_{j},y\_{j,\leq t}) is a value function learned by minimizing (R​(xj,yj)−vϕ​(xj,yj,≤t))2\big(R(x\_{j},y\_{j})-v\_{\phi}(x\_{j},y\_{j,\leq t})\big)^{2}.

In most practical RL setups, the *current policy* π\pi will often differ from the behavior policy μ\mu that generates yky\_{k}, due to the weights lagging, quantization or implementation difference between the inference and training softwares. This difference is usually handled by either a trust region constraint (Schulman et al., [2017](#bib.bib23)) or using Importance Sampling (IS). In practice, the importance sampling weights are truncated to reduce the variance of the estimator (Munos et al., [2016](#bib.bib15); Espeholt et al., [2018](#bib.bib2)):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∇~​JI​S​(π)\displaystyle\tilde{\nabla}J\_{IS}(\pi) | =1m​∑j=1m∑t=1Tjmin⁡(c,π​(yk∣xj)μ​(yk∣xj))​(R​(xj,yj)−vϕ​(xj,yj,≤t))​∇log⁡π​(yj,t∣xj,yj,<t)\displaystyle=\frac{1}{m}\sum\_{j=1}^{m}\sum\_{t=1}^{T\_{j}}\min\left(c,\frac{\pi(y\_{k}\mid x\_{j})}{\mu(y\_{k}\mid x\_{j})}\right)\left(R(x\_{j},y\_{j})-v\_{\phi}(x\_{j},y\_{j,\leq t})\right)\nabla\log\pi(y\_{j,t}\mid x\_{j},y\_{j,<t}) |  | (5) |

The Effective Sample Size (ESS) (Kong, [1992](#bib.bib10)) is commonly used to quantify the quality of importance sampling estimators in RL (Schlegel et al., [2019](#bib.bib22); Fakoor et al., [2020](#bib.bib4)). When using off-policy RL, ESS measures how many samples from the current policy π\pi would yield equivalent performance to weighted samples from the behavior policy μ\mu. The (normalized) ESS is defined as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ESS=(∑i=1Nwi)2/N​∑i=1Nwi2\text{ESS}=\left(\sum\limits\_{i=1}^{N}w\_{i}\right)^{2}\bigg/N\sum\limits\_{i=1}^{N}w\_{i}^{2} |  | (6) |

where wiw\_{i} are importance weights for a sample of size NN. This metric effectively ranges between 0 and 1 when normalized, with values closer to 1 indicating more efficient sampling, e.g. the ESS of on-policy data is exactly 1. Small ESS will result in a high variance REINFORCE gradient estimate and might destabilize the learning process.

Algorithm 1  Conventional RL

Current policy π\pi.

Optimizer state opt\_state.

Number of optimizer steps per RL step GG.

Training batch size BB.

while True do

// generation ⊳\triangleright RL step starts

μ←π\mu\leftarrow\pi ⊳\triangleright Initialize behavior policy μ\mu

sequences ←\leftarrow generate B​GBG sequences from μ\mu

batches ←\leftarrow split sequences in G batches of size B

// training

lag ←\leftarrow 0 ⊳\triangleright lag between μ\mu and π\pi

for batch in batches do

π\pi, opt\_state ←\leftarrow
optimizer\_step(π\pi, opt\_state, batch)

lag ←\leftarrow lag + 1

end for⊳\triangleright RL step ends

end while

### 2.2 Conventional RL

Most RL implementations alternate between generating sequences and training the policy on the generated data. We refer to this approach as Conventional RL and describe it in detail in [Algorithm˜1](#alg1 "In 2.1 Reinforcement Learning for Large Language Models ‣ 2 Background ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation"). When training involves doing G>1G>1 optimizer steps, the current policy π\pi gets ahead of the behavior policy μ\mu that was used to generate the data. We adopt the term lag to refer to the number of optimizer steps between μ\mu and π\pi.

![Refer to caption](/html/2509.19128/assets/figures/qwen_throughput.png)


(a) Throughput vs batch size.

![Refer to caption](/html/2509.19128/assets/figures/sequences_in_progress.png)


(b) Inference batch size vs time.

![Refer to caption](/html/2509.19128/assets/figures/time_vs_throughput.png)


(c) Time vs Throughput.

Figure 2: Analysis of generation times and throughput. We perform all measurements using a vLLM engine serving a Qwen 2.5 7B model on a H100 GPU.
(a) Short prompt generation throughput increases up to batch size 256. (b) Generation batch size gradually decreases to suboptimal values as the engine finishes sequences (c) Generation time reaches a plateau and throughput decreases as the number of sequences per GPU goes down. We report the average of 5 runs and 95% CI.




Algorithm 2  PipelineRL: Actor and Trainer Processes

1:Current policy weights π\pi.

2:Generation batch size HH.

3:Training sequence queue Qt​r​a​i​nQ\_{train}.

4:Actor Process:

5:function Actor

6:  sequences in progress Sp​r​o​gS\_{prog} ←\leftarrow []

7:  while True do

8:   Sf​i​nS\_{fin}, Sp​r​o​gS\_{prog} ←\leftarrow pop finished sequences from Sp​r​o​gS\_{prog}

9:   Qt​r​a​i​n.p​u​t​(Sf​i​n)Q\_{train}.put(S\_{fin}) ⊳\triangleright Send finished seqs to the trainer

10:   if l​e​n​(Sp​r​o​g<H)len(S\_{prog}<H) then

11:     add H−l​e​n​(Sp​r​o​g)H-len(S\_{prog}) prompts to Sp​r​o​gS\_{prog}

12:   end if

13:   if Trainer requests weight update then ⊳\triangleright In-flight check for new weights

14:     μ\mu ←\leftarrow receive\_weight\_update()

15:   end if

16:   Sp​r​o​gS\_{prog} ←\leftarrow generate next tokens with μ\mu

17:  end while

18:end function

19:

20:Trainer Process:

21:function Trainer(π\pi, opt\_state)

22:  batch ←\leftarrow []

23:  while True do

24:   request\_actor\_weight\_update(π\pi) ⊳\triangleright In-flight weight update

25:   batch ←\leftarrow get BB sequences from Qt​r​a​i​nQ\_{train}

26:   π\pi, opt\_state ←\leftarrow optimizer\_step(π\pi, opt\_state, batch)

27:  end while

28:end function

### 2.3 Efficient Sequence Generation with LLMs

Transformer models generate sequences one token at a time, left-to-right. To make this process efficient, advanced generation (inference) engines such as vLLM and SGLang process a batch of sequences at a time, while carefully managing their past keys and values in a paged structure called KV cache (Kwon et al., [2023b](#bib.bib12)). All modern generation engines support adding new generation requests in-flight to the ones in progress without stopping the generation process. Based on accelerator specifications, generation engines should achieve the maximum generation throughput at very large batch sizes of several thousand sequences.222<https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html> In practice, at very large batch sizes, the per-sequence latency can become prohibitively high, KV cache may grow too large to fit in accelerator memory, or the request queue management overheads can dominate.

## 3 The learning speed ceiling of Conventional RL

Reinforcement learning for LLMs can be slow when the LLM is trained to generate long sequences of tokens, e.g., long-form reasoning to solve mathematical problems, because each generation can take up to several minutes. Here we explain why it is challenging to effectively scale up long sequence RL, i.e. to effectively use a larger number of accelerators NN to make average reward R​(t)R(t) at time tt grow faster. As a mathematical function, one can view R​(t)R(t) as a composition of the functions R​(S)R(S) and S​(t)S(t), where SS is the number of samples the RL learner will have processed by time tt. A faster RL learner will have a higher *learning speed* Δ​RΔ​t\frac{\Delta R}{\Delta t} which we can express as the product of *learning effectiveness* and *learning throughput* as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​RΔ​t⏟speed=Δ​RΔ​S⏟effectiveness×Δ​SΔ​t⏟throughput.\underbrace{\frac{\Delta R}{\Delta t}}\_{\text{speed}}=\underbrace{\frac{\Delta R}{\Delta S}}\_{\text{effectiveness}}\times\underbrace{\frac{\Delta S}{\Delta t}}\_{\text{throughput}}. |  | (7) |

The Conventional RL algorithm from [Algorithm˜1](#alg1 "In 2.1 Reinforcement Learning for Large Language Models ‣ 2 Background ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation") has the highest Δ​RΔ​S\frac{\Delta R}{\Delta S} when it is fully on-policy, i.e., when one performs only one optimizer step per each RL step. Yet the throughput Δ​SΔ​t\frac{\Delta S}{\Delta t} in the pure on-policy case can be low because the accelerators will be working on at most batch size BB samples at a time. Increasing the number of accelerators NN will yield diminishing returns in increasing Δ​SΔ​t\frac{\Delta S}{\Delta t}, because the throughput of each accelerator will decrease when the number of samples per accelerator BN\frac{B}{N} goes below the optimal range ([Figure˜2(c)](#S2.F2.sf3 "In Figure 2 ‣ 2.2 Conventional RL ‣ 2 Background ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation")). For example, see [Figure˜2(a)](#S2.F2.sf1 "In Figure 2 ‣ 2.2 Conventional RL ‣ 2 Background ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation") for inference throughput for a 7B Qwen model on a single H100 GPU. One can see that the throughput increases almost linearly up to the generation batch size of 128. Hence, e.g. using 2​N2N GPUs to generate 32 samples will not be much faster than using NN GPUs to generate 6464. Furthermore, as the LLM finishes the shorter generations, there will be fewer longer generations still in progress, see Figure [2(b)](#S2.F2.sf2 "Figure 2(b) ‣ Figure 2 ‣ 2.2 Conventional RL ‣ 2 Background ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation") for an illustration. Hence, to make good use of the hardware, one should use each accelerator to generate many times more sequences than the optimal batch size.

Commonly, to increase the throughput, most practitioners perform multiple G>1G>1 optimizer steps per RL step, which entails generating B​GBG rollouts at each generation stage. This way, one can often achieve a higher throughput Δ​SΔ​t\frac{\Delta S}{\Delta t} by increasing NN up to a point when B​GN\frac{BG}{N} becomes too small. It is, however, known from the literature that going too off-policy by using a high value of GG will eventually decrease the learning effectiveness Δ​RΔ​S\frac{\Delta R}{\Delta S} (Noukhovitch et al., [2024](#bib.bib16)). Clearly, at some points, the rollouts from the old policy become too stale and no longer useful as the source of learning signal for the current policy. Hence, given a fixed optimizer batch size BB, one scales up Conventional RL by increasing GG and NN until the product Δ​RΔ​S​Δ​SΔ​t\frac{\Delta R}{\Delta S}\frac{\Delta S}{\Delta t} no longer improves, and the hard ceiling of Δ​RΔ​t\frac{\Delta R}{\Delta t} for the given number of accelerators NN is achieved.

![Refer to caption](/html/2509.19128/assets/x1.png)


(a) Token lags as a function of optimizer steps.

![Refer to caption](/html/2509.19128/assets/x2.png)


(b) Pareto curves.

Figure 3: (a) For Conventional RL, the token lag increases with the number of optimizer steps. In PipelineRL with N accelerators, the token lag varies throughout the sequence, where earlier tokens have higher lag. The lag structure in each batch is the same. Doubling the PipelineRL accelerators, everything else constant, double the lag of early tokens. (b) Schematic illustration of PipelineRL’s throughput-effectiveness trade-off as a function of training accelerators TT and of Conventional RL as a function of lag GG. PipelineRL achieves a higher Δ​RΔ​S​Δ​SΔ​t\frac{\Delta R}{\Delta S}\frac{\Delta S}{\Delta t} for the same number NN of accelerators.

## 4 Pushing the learning speed ceiling with PipelineRL

The Pipeline RL method differs from Conventional RL in two aspects: (1) running training and generation in parallel *asynchronously*, and (2) updating the generation weights after every optimizer step in-flight, i.e. without stopping the sequence generation. [Algorithm˜2](#alg2 "In 2.2 Conventional RL ‣ 2 Background ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation") provides an abstracted formal description of PipelineRL in terms of two concurrent Actor and Trainer processes that communicate via a sample queue and a high-bandwidth weight transfer network.

The effectiveness-throughput trade-off for PipelineRL is the opposite of that of Conventional RL. Namely, adding more accelerators to a PipelineRL setup leads to a linear increase of Δ​SΔ​t\frac{\Delta S}{\Delta t}, but may eventually harm Δ​RΔ​S\frac{\Delta R}{\Delta S}. In Figure [3(a)](#S3.F3.sf1 "Figure 3(a) ‣ Figure 3 ‣ 3 The learning speed ceiling of Conventional RL ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation"), we illustrate how PipelineRL produces mixed-policy sequences in which earlier tokens are more off-policy than the recent ones. Doubling NN will double the lag of the earliest tokens as well as the average lag in the PipelineRL batch. Notably, the off-policyness profile is different for PipelineRL and its conventional counterpart. Taking the average token lag as a proxy for off-policyness, in PipelineRL all batches are equally off-policy, whereas for Conventional RL later batches become progressively more off-policy. This difference makes it hard to analytically reason about the Δ​RΔ​t\frac{\Delta R}{\Delta t} improvement that PipelineRL can bring over the baseline, because Δ​RΔ​S\frac{\Delta R}{\Delta S} can only be estimated empirically by running RL experiments. In supplementary material, we present our simulation of how, for the same maximum lag gm​a​xg\_{max} PipelineRL can learn 1.5x faster than Conventional RL. The empirical gains can be even larger, depending on how frequently one can make weight updates without hurting the learning effectiveness Δ​RΔ​S\frac{\Delta R}{\Delta S}.

##### Configuring PipelineRL vs Conventional RL

For a fixed batch size BB and a number of accelerators NN, one can configure Conventional RL by choosing the number of optimizer steps GG, trading off the learning effectiveness for the throughput. The PipelineRL configuration can likewise be mostly reduced to a single parameter, namely the number of training accelerators TT out of NN available ones. Setting a higher TT will almost linearly decrease the time tt​r​a​i​nt\_{train} that is needed for the trainer to process BB sequences and perform an optimizer step. TT effectively determines the optimal generation batch size HH to be used at all N−TN-T accelerators. Using a lower HH leads to a lower maximum generation latency tg​e​nt\_{gen}, which consequently reduces the maximum lag gm​a​x=⌈tg​e​n/tt​r​a​i​n⌉g\_{max}=\lceil t\_{gen}/t\_{train}\rceil. Hence, it makes sense to use the smallest HH that suffices to produce enough training data. Consequently, the maximum lag gm​a​xg\_{max} for PipelineRL grows with the number of training accelerators TT, as higher TT requires a higher HH and leads to a lower tt​r​a​i​nt\_{train} and a higher tg​e​nt\_{gen}. On the contrary, the sample throughput of PipelineRL grows with TT up to a point when N−TN-T accelerators cannot generate enough data for the over-powered trainer.
We recommend avoiding extreme configurations with TT too high (very high lag GG) and TT too low (bad hardware utilization, one can just as well scale down the compute).
[Figure˜3(b)](#S3.F3.sf2 "In Figure 3 ‣ 3 The learning speed ceiling of Conventional RL ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation") visualizes how different configurations of PipelineRL and Conventional RL achieve different learning effectiveness Δ​RΔ​S\frac{\Delta R}{\Delta S} and throughput Δ​SΔ​t\frac{\Delta S}{\Delta t}, with PipelineRL setups reaching higher Δ​RΔ​t=Δ​SΔ​t​Δ​RΔ​S\frac{\Delta R}{\Delta t}=\frac{\Delta S}{\Delta t}\frac{\Delta R}{\Delta S} isocurves.

![Refer to caption](/html/2509.19128/assets/x3.png)


Figure 4: The three pipeline stages of PipelineRL implementation: actor, preprocessor and trainer. Earlier stages stream the data to the latter ones using Redis as the streaming broker.

##### Architecture and Implementation Details

Our PipelineRL implementation concurrently runs many distributed vLLM generation engines and DeepSpeed training workers in a three stage pipeline that we describe in [Figure˜4](#S4.F4 "In Configuring PipelineRL vs Conventional RL ‣ 4 Pushing the learning speed ceiling with PipelineRL ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation"). The middle Preprocessor stage that we omitted from [Algorithm˜2](#alg2 "In 2.2 Conventional RL ‣ 2 Background ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation") for simplicity, computes reference model log-probabilities often used in Reinforcement Learning from Human Feedback (Ouyang et al., [2022](#bib.bib18)). The PipelineRL architecture is highly modular — any generation software that supports the three HTTP API endpoints that PipelineRL requires can be easily integrated in the future. The three APIs are the popular /v1/chat/completions for generation, /init\_process\_group for creating the weight transfer process group, and /request\_weight\_update for initiating the in-flight weight update. Key optimizations in PipelineRL include online sequence packing for fast training and using ring buffers to minimize the lag when earlier pipeline stages run faster than the later ones, e.g. when the trainer makes a checkpoint.

## 5 Experiments

![Refer to caption](/html/2509.19128/assets/figures/reward_per_time_square.png)


(a) Reward vs time.

![Refer to caption](/html/2509.19128/assets/figures/reward_per_samples_square.png)


(b) Reward vs samples.

![Refer to caption](/html/2509.19128/assets/figures/samples_per_time_square.png)


(c) Samples vs time.

Figure 5: (a)
PipelineRL attains the same average reward faster than the conventional RL baselines. (b) PipelineRL achieves the same sample efficiency as G=8G=8 and G=16G=16. (c) PipelineRL generates samples much faster than the conventional RL baselines.



![Refer to caption](/html/2509.19128/assets/figures/lag_plots.png)


(a) Max lag.

![Refer to caption](/html/2509.19128/assets/figures/ess_plots2.png)


(b) Effective sample size.

Figure 6: In [Figure˜6(a)](#S5.F6.sf1 "In Figure 6 ‣ 5 Experiments ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation"), PipelineRL attains a higher max lag that every conventional RL method, but as observed in [Figure˜6(b)](#S5.F6.sf2 "In Figure 6 ‣ 5 Experiments ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation"), the Effective Sample Size is similar to G=8. This indicates that while the max lag is quite high, PipelineRL stays mostly on-policy as measured by the ESS.

For the experimental validation of PipelineRL’s high learning effectiveness Δ​RΔ​S\frac{\Delta R}{\Delta S} and throughput Δ​SΔ​t\frac{\Delta S}{\Delta t}, we have chosen the challenging task of training a base (i.e. not instruction-tuned) model to perform long-form reasoning to solve mathematical problems. We find this task to be a great testbed for PipelineRL because the policy undergoes rapid changes over the course of training. In particular, the length of generated sequences grows dramatically (Guo et al., [2025](#bib.bib5)), making it essential to stay on-policy for effective learning.

##### Experimental setup.

For each experiment, we train the Qwen 2.5 base model (Yang et al., [2024](#bib.bib28)) with 7B parameters on 17K math problems from the OpenReasoner Zero dataset (Hu et al., [2025](#bib.bib8)) for 1000 optimizer steps with the batch size B=1024B=1024. We use Adam optimizer (Kingma, [2014](#bib.bib9)) with the learning rate 1e-6. We run the PipelineRL experiments on 16 DGX-H100 nodes, using 48 GPUs for generation at batch size H=64H=64 and 80 GPUs for training. We tweak PipelineRL to simulate Conventional RL by accumulating and shuffling a buffer of B​GBG samples at the Preprocessor stage before the GG optimizer steps of each RL step start. To estimate the Conventional RL throughput, we use 2 nodes for generation at batch size H=64H=64 and 2 nodes for training, and then add a correction for training on 8x fewer GPUs than what an efficient Conventional RL implementation with a quick generation-training transition could use. To estimate the inference throughput on 128 GPUS instead of 16 GPUs, we submit 128/16128/16 batches of 16×1024×G128\tfrac{16\times 1024\times G}{128} and take the maximum completion time. We give reward 1 to any generated sequence with the correct answer and 0 otherwise. We also give a soft penalty to the model when it gets close to the max sequence length. We train every model with importance weighted REINFORCE as described in Section [2](#S2 "2 Background ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation") and clamp the importance weights to 5. For our experiment we use vLLM (Kwon et al., [2023a](#bib.bib11)) to generate trajectories and use DeepSpeed (Rasley et al., [2020](#bib.bib20)) through accelerate to train the model.

Table 1: Success rate of models trained with PipelineRL compared to results in the literature.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Method | Math 500 | AIME24 | # samples (⋅106\cdot 10^{6} ) | training data |
| |  | | --- | | Qwen 2.5 base 7b | | 31.6 | 3.3 | - | - |
| |  | | --- | | SimpleRL Zero | | (Zeng et al., [2025](#bib.bib29)) | | 78.2 | 20.0 | 0.82 | Math Level 3-5 |
| |  | | --- | | OpenReasoner Zero | | (Hu et al., [2025](#bib.bib8)) | | ∼\sim 82.0 | ∼\sim 20.0 | 8.2 | OpenReasoner |
| PipelineRL (batch size 1024) | 81 | 17.5 | 2.0 | OpenReasoner |
| PipelineRL (batch size 4096) | 84.6 | 19.8 | 6.2 | OpenReasoner |

##### PipelineRL learns faster due to higher throughput.

We compare the learning speed of PipelineRL to that of Conventional RL with G=32G=32 optimizer steps, as that was the maximum GG for which Conventional RL training was stable. PipelineRL achieves the same reward values approximately ∼2​x\sim 2x faster than this baseline ([Figure˜5(a)](#S5.F5.sf1 "In Figure 5 ‣ 5 Experiments ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation")) due to ∼2​x\sim 2x faster sample throughput ([Figure˜5(c)](#S5.F5.sf3 "In Figure 5 ‣ 5 Experiments ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation")). The main cause of the throughput increase is that GPU utilization for G=32G=32 experiment on 128 GPUs is relatively low for each GPU when it has to generate just 32×1024/128=25632\times 1024/128=256 sequences (see Figure [2(b)](#S2.F2.sf2 "Figure 2(b) ‣ Figure 2 ‣ 2.2 Conventional RL ‣ 2 Background ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation")). Further increasing GG to 64 results in divergence, see [Figure˜10](#A2.F10 "In Appendix B Additional Results ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation").

##### PipelineRL learns effectively.

To better measure learning effectiveness Δ​RΔ​S\frac{\Delta R}{\Delta S} of PipelineRL, we also run Conventional RL experiments with G=8G=8, G=16G=16, and G=32G=32 optimizer steps. Notably, the R​(S)R(S) curves are indistinguishable for all compared methods up to a point where G=32G=32 is slower and unstable, likely because of going too far off-policy. This result validates that PipelineRL’s signature in-flight weight updates do no harm to the sequence generation process.

##### PipelineRL matches comparable results on reasoning tasks.

[Table˜1](#S5.T1 "In Experimental setup. ‣ 5 Experiments ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation") compares the test performance of PipelineRL to similar experiments that start training from the same Qwen 2.5 7B model. In this experiment we used batch size 4096 because we found it leads to a higher performance. On the math reasoning benchmarks MATH500 (Hendrycks et al., [2021](#bib.bib6)) and AIME2024 (Li et al., [2024](#bib.bib13)). PipelineRL matches or exceeds the success rate of Open Reasoner Zero and SimpleRL Zero.

##### PipelineRL stays more on-policy.

To gain a better understanding of which training methods stay more on-policy, we plot the evolution of the max lag and the ESS on-policyness measure throughout the training. [Figure˜6(a)](#S5.F6.sf1 "In Figure 6 ‣ 5 Experiments ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation") shows that PipelineRL obtains a higher max lag than the conventional RL baselines. Notably some tokens have a lag of more than 50k samples. However [Figure˜6(b)](#S5.F6.sf2 "In Figure 6 ‣ 5 Experiments ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation") shows that, in terms of ESS, PipelineRL maintains a similar on-policyness as G=8G=8. We further observe that the ESS of G=16G=16 and in particular G=32G=32 drops throughout training.

### 5.1 Impact of in-flight weight updates on on-policyness

In this section, we compare the sampling distribution of in-flight weight updates to 1) conventional RL with different max lag and 2) in-flight weight update with KV cache recomputation. For this experiment, we save a set of consecutive checkpoints CiC\_{i}, one after every optimizer step. To replicate the in-flight weight update, we start from a checkpoint and update the weights of the behavior policy every Lgmax\frac{L}{g\_{\max}} tokens with the subsequent checkpoint, where L is the maximum sequence length and gmaxg\_{\max} is the maximum lag. Specifically, the PipelineRL behavior policy is defined as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | μ:=μC​(x1:t1)​…​μC+g​(xtg:tg+1∣x^1:t1,…​x^tg−1:tg)\displaystyle\mu:=\mu\_{C}(x\_{1:t\_{1}})\ldots\mu\_{C+g}(x\_{t\_{g}:t\_{g+1}}\mid\hat{x}\_{1:t\_{1}},\ldots\hat{x}\_{t\_{g-1}:t\_{g}}) |  | (8) |

where t1=2​Lgmaxt\_{1}=\tfrac{2L}{g\_{\max}} and tg=tg−1+Lgmaxt\_{g}=t\_{g-1}+\tfrac{L}{g\_{\max}} tokens for lag g>1g>1 since the first weight update takes longer than the next updates due to the bubble at the beginning of training, see [Figure˜1](#S2.F1 "In 2 Background ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation") b). We also use x^\hat{x} to stress that the KV cache for the previous tokens is stale - as it was computed under previous model weights. We then compute the KL between the mixed behavior policy μC:C+g\mu\_{C:C+g} and the on-policy behavior policy μC+g\mu\_{C+g}. We also report the KL with the mixed behavior policy with updated KV cache which we denote as *PipelineRL with KV cache recomputed*. To replicate conventional RL, we sample NN sequences from the behavior policy μ:=μC\mu:=\mu\_{C} and compute the KL with on-policy behavior policy μC+g\mu\_{C+g} for different lag kk.

In this experiment, we fine-tune Qwen 2.5 base 7B on the OpenReasoner Zero (Hu et al., [2025](#bib.bib8)) data for 222 optimizer steps. We consider three stages in training to measure KL-divergence: starting at checkpoint 0, 100, and 190. The maximum lag gmaxg\_{\max} is set to 32 and the maximum sequence length LL is 2048. As presented in [Figure˜7](#S5.F7 "In 5.1 Impact of in-flight weight updates on on-policyness ‣ 5 Experiments ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation"), the distribution of mixed-policy sequences closely aligns with that of fully on-policy sequences across all three stages in the training. In contrast, off-policy sequences exhibit consistently higher divergences as lag increases. Also, using stale KV-cache for mixed policy sequences introduces only slightly higher divergence compared to recomputing the cache. This supports our design choice in Pipeline-RL to opt for the more efficient approach of retaining the KV cache.

![Refer to caption](/html/2509.19128/assets/figures/offpolicyness_kl_combined.png)


Figure 7: 
For three different starting checkpoint, PipelineRL with and without KV cache recomputation stay more on-policy than Conventional RL as measured by the KL divergence.

## 6 Related work

Asynchronous and high-throughput RL has been extensively studied. IMPALA (Espeholt et al., [2018](#bib.bib2)) decoupled acting from learning to maximize GPU utilization. Like PipelineRL, IMPALA used truncated importance weights to estimate the value function from off-policy samples. Furthermore, IMPALA kept the policy weights constant for the length of an episode. SeedRL (Espeholt et al., [2019](#bib.bib3)) proposed to update the model’s parameters during an episode, resulting in trajectories where different actions were sampled by different policies. OpenAI Five (OpenAI et al., [2019](#bib.bib17)) was trained using asynchronous PPO to achieve superhuman performance on Dota 2. These previous works were focused on RL for video games. Closer to our work,
 (Noukhovitch et al., [2024](#bib.bib16)) explores asynchronous RL for LLMs. In their approach, data generation for the next GG optimizer steps is synchronized with training on the previous GG optimizer steps, leading to higher off-policyness than Conventional RL, unlike PipelineRL. The same study shows that offline methods such as DPO (Rafailov et al., [2023](#bib.bib19)) can better tolerate off-policyness.

There exist several other scalable open-source RL implementations. veRL (Sheng et al., [2024](#bib.bib25)) implements Conventional RL efficiently by using a sophisticated hybrid generation-training engine that supports quick transitions between training and generation on the same GPUs. We believe veRL’s throughput would be similar to our Conventional RL baseline. Without the hybrid engine, in OpenRLHF  (Hu et al., [2024](#bib.bib7)) training GPUs idle during generation and vice-versa. Concurrently, Magistral(Mistral-AI et al., [2025](#bib.bib14)) also introduced in-flight weight updates.

## 7 Conclusion and Discussion

We have shown how in-flight weight updates help PipelineRL break the learning speed ceiling of the conventional two-stage RL approach. We believe that for long sequence generation, in particular, this speedup would be very difficult to attain with another asynchronous RL approach, as synchronous waits for generation to finish would hurt the throughput and/or learning effectiveness. The stale KV-cache risk that in-flight updates introduce can be mitigated by recomputing the KV cache after each update, which can be done fast at a high GPU utilization, but will still lower the throughput.

We believe PipelineRL may be particular useful for training LLMs to excel at agentic behaviors that involve multiple LLM generations interspersed with environment interactions.
Another promising direction for future work is to study when the recent low lag tokens in PipelineRL are helpful, and on the contrary, where PipelineRL’s constantly high lag of early tokens in long sequences hurts.

##### Limitations

PipelineRL will only bring a limited throughput increase over Conventional RL if the LLM is asked to generate the exact same number of tokens for the same prompt. In this unlikely scenario, Conventional RL will be likewise capable of maintaining a constant generation batch size. The PipelineRL’s stable average token lag and the low lag of recent tokens in each batch may, however, still affect the learning effectiveness. The PipelineRL throughput advantages will likewise decrease in setups with scarce or extensive compute resources. In the former case, each GPU will get enough generation tasks for the GPU utilization to be high. In the latter, the learning speed will be bounded not by the hardware utilization but by the best possible generation latency and by the environment feedback delay.

#### Acknowledgments

We are grateful to Nicolas Chapados for his thoughtful feedback. We also thank our colleagues at ServiceNow—Perouz Taslakian, Massimo Caccia, Catherine Martin, Étienne Marcotte, Torsten Scholak, and Ghazwa Darwiche—for their support in providing additional compute resources.

## References

* Ahmadian et al. (2024)

  Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker.
  Back to basics: Revisiting REINFORCE style optimization for learning from human feedback in LLMs.
  *arXiv preprint arXiv:2402.14740*, 2024.
* Espeholt et al. (2018)

  Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Vlad Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, et al.
  IMPALA: Scalable distributed deep-RL with importance weighted actor-learner architectures.
  In *International conference on machine learning*, pp. 1407–1416. PMLR, 2018.
* Espeholt et al. (2019)

  Lasse Espeholt, Raphaël Marinier, Piotr Stanczyk, Ke Wang, and Marcin Michalski.
  SEED RL: Scalable and efficient deep-RL with accelerated central inference.
  *arXiv preprint arXiv:1910.06591*, 2019.
* Fakoor et al. (2020)

  Rasool Fakoor, Pratik Chaudhari, and Alexander J Smola.
  P3O: Policy-on policy-off policy optimization.
  In *Uncertainty in artificial intelligence*, pp. 1017–1027. PMLR, 2020.
* Guo et al. (2025)

  Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al.
  DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning.
  *arXiv preprint arXiv:2501.12948*, 2025.
* Hendrycks et al. (2021)

  Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt.
  Measuring mathematical problem solving with the MATH dataset.
  *arXiv preprint arXiv:2103.03874*, 2021.
* Hu et al. (2024)

  Jian Hu, Xibin Wu, Zilin Zhu, Xianyu, Weixun Wang, Dehao Zhang, and Yu Cao.
  OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework, November 2024.
  URL <http://arxiv.org/abs/2405.11143>.
  arXiv:2405.11143 [cs].
* Hu et al. (2025)

  Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum.
  Open-Reasoner-Zero: An open source approach to scaling up reinforcement learning on the base model.
  *arXiv preprint arXiv:2503.24290*, 2025.
* Kingma (2014)

  Diederik P Kingma.
  Adam: A method for stochastic optimization.
  *arXiv preprint arXiv:1412.6980*, 2014.
* Kong (1992)

  Augustine Kong.
  A note on importance sampling using standardized weights.
  *University of Chicago, Dept. of Statistics, Tech. Rep*, 348:14, 1992.
* Kwon et al. (2023a)

  Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica.
  Efficient memory management for large language model serving with pagedattention, 2023a.
  URL <https://arxiv.org/abs/2309.06180>.
* Kwon et al. (2023b)

  Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica.
  Efficient Memory Management for Large Language Model Serving with PagedAttention, September 2023b.
  URL <http://arxiv.org/abs/2309.06180>.
  arXiv:2309.06180 [cs].
* Li et al. (2024)

  Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, et al.
  NuminaMath: The largest public dataset in AI4Maths with 860k pairs of competition math problems and solutions.
  *Hugging Face repository*, 13:9, 2024.
* Mistral-AI et al. (2025)

  Mistral-AI, :, Abhinav Rastogi, Albert Q. Jiang, Andy Lo, Gabrielle Berrada, Guillaume Lample, Jason Rute, Joep Barmentlo, Karmesh Yadav, Kartik Khandelwal, Khyathi Raghavi Chandu, Léonard Blier, Lucile Saulnier, Matthieu Dinot, Maxime Darrin, Neha Gupta, Roman Soletskyi, Sagar Vaze, Teven Le Scao, Yihan Wang, Adam Yang, Alexander H. Liu, Alexandre Sablayrolles, Amélie Héliou, Amélie Martin, Andy Ehrenberg, Anmol Agarwal, Antoine Roux, Arthur Darcet, Arthur Mensch, Baptiste Bout, Baptiste Rozière, Baudouin De Monicault, Chris Bamford, Christian Wallenwein, Christophe Renaudin, Clémence Lanfranchi, Darius Dabert, Devon Mizelle, Diego de las Casas, Elliot Chane-Sane, Emilien Fugier, Emma Bou Hanna, Gauthier Delerce, Gauthier Guinet, Georgii Novikov, Guillaume Martin, Himanshu Jaju, Jan Ludziejewski, Jean-Hadrien Chabran, Jean-Malo Delignon, Joachim Studnia, Jonas Amar, Josselin Somerville Roberts, Julien Denize, Karan Saxena, Kush Jain, Lingxiao Zhao, Louis Martin, Luyu Gao, Lélio Renard Lavaud, Marie
  Pellat, Mathilde Guillaumin, Mathis Felardos, Maximilian Augustin, Mickaël Seznec, Nikhil Raghuraman, Olivier Duchenne, Patricia Wang, Patrick von Platen, Patryk Saffer, Paul Jacob, Paul Wambergue, Paula Kurylowicz, Pavankumar Reddy Muddireddy, Philomène Chagniot, Pierre Stock, Pravesh Agrawal, Romain Sauvestre, Rémi Delacourt, Sanchit Gandhi, Sandeep Subramanian, Shashwat Dalal, Siddharth Gandhi, Soham Ghosh, Srijan Mishra, Sumukh Aithal, Szymon Antoniak, Thibault Schueller, Thibaut Lavril, Thomas Robert, Thomas Wang, Timothée Lacroix, Valeriia Nemychnikova, Victor Paltz, Virgile Richard, Wen-Ding Li, William Marshall, Xuanyu Zhang, and Yunhao Tang.
  Magistral, 2025.
  URL <https://arxiv.org/abs/2506.10910>.
* Munos et al. (2016)

  Rémi Munos, Tom Stepleton, Anna Harutyunyan, and Marc Bellemare.
  Safe and efficient off-policy reinforcement learning.
  *Advances in neural information processing systems*, 29, 2016.
* Noukhovitch et al. (2024)

  Michael Noukhovitch, Shengyi Huang, Sophie Xhonneux, Arian Hosseini, Rishabh Agarwal, and Aaron Courville.
  Asynchronous RLHF: Faster and more efficient off-policy RL for language models.
  *arXiv preprint arXiv:2410.18252*, 2024.
* OpenAI et al. (2019)

  OpenAI, :, Christopher Berner, Greg Brockman, Brooke Chan, Vicki Cheung, Przemysław Dębiak, Christy Dennison, David Farhi, Quirin Fischer, Shariq Hashme, Chris Hesse, Rafal Józefowicz, Scott Gray, Catherine Olsson, Jakub Pachocki, Michael Petrov, Henrique P. d. O. Pinto, Jonathan Raiman, Tim Salimans, Jeremy Schlatter, Jonas Schneider, Szymon Sidor, Ilya Sutskever, Jie Tang, Filip Wolski, and Susan Zhang.
  Dota 2 with large scale deep reinforcement learning, 2019.
  URL <https://arxiv.org/abs/1912.06680>.
* Ouyang et al. (2022)

  Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al.
  Training language models to follow instructions with human feedback.
  *Advances in neural information processing systems*, 35:27730–27744, 2022.
* Rafailov et al. (2023)

  Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn.
  Direct preference optimization: Your language model is secretly a reward model.
  *Advances in Neural Information Processing Systems*, 36:53728–53741, 2023.
* Rasley et al. (2020)

  Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He.
  Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters.
  In *KDD*, pp. 3505–3506, 2020.
  URL <https://doi.org/10.1145/3394486.3406703>.
* Roux et al. (2025)

  Nicolas Le Roux, Marc G Bellemare, Jonathan Lebensold, Arnaud Bergeron, Joshua Greaves, Alex Fréchette, Carolyne Pelletier, Eric Thibodeau-Laufer, Sándor Toth, and Sam Work.
  Tapered off-policy REINFORCE: Stable and efficient reinforcement learning for LLMs.
  *arXiv preprint arXiv:2503.14286*, 2025.
* Schlegel et al. (2019)

  Matthew Schlegel, Wesley Chung, Daniel Graves, Jian Qian, and Martha White.
  Importance resampling for off-policy prediction.
  *Advances in Neural Information Processing Systems*, 32, 2019.
* Schulman et al. (2017)

  John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
  Proximal policy optimization algorithms.
  *arXiv preprint arXiv:1707.06347*, 2017.
* Shao et al. (2024)

  Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al.
  DeepSeekMath: Pushing the limits of mathematical reasoning in open language models.
  *arXiv preprint arXiv:2402.03300*, 2024.
* Sheng et al. (2024)

  Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu.
  HybridFlow: A flexible and efficient RLHF framework.
  *arXiv preprint arXiv:2409.19256*, 2024.
* Wei et al. (2025)

  Yuxiang Wei, Olivier Duchenne, Jade Copet, Quentin Carbonneaux, Lingming Zhang, Daniel Fried, Gabriel Synnaeve, Rishabh Singh, and Sida I Wang.
  SWE-RL: Advancing LLM reasoning via reinforcement learning on open software evolution.
  *arXiv preprint arXiv:2502.18449*, 2025.
* Williams (1992)

  Ronald J Williams.
  Simple statistical gradient-following algorithms for connectionist reinforcement learning.
  *Machine learning*, 8:229–256, 1992.
* Yang et al. (2024)

  An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al.
  Qwen2.5 technical report.
  *arXiv preprint arXiv:2412.15115*, 2024.
* Zeng et al. (2025)

  Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He.
  SimpleRL-Zoo: Investigating and taming zero reinforcement learning for open base models in the wild.
  *arXiv preprint arXiv:2503.18892*, 2025.

## Appendix A Analyical estimate of PipelineRL speedup for fixed max lag

In this additional section we estimate how much faster PipelineRL can be compared to Conventional RL for the same value of maximum token lag gm​a​xg\_{max}. We will be using the following notation, mostly the same as in the main text:

* •

  NN is the number of accelerators
* •

  S=B​GS=BG is the number of sequences that are processed in each Conventional RL step
* •

  LL is the maximum and L¯\overline{L} is the average sequence length for the current policy π\pi
* •

  K=S​L¯K=S\overline{L} is total number of tokens that Conventional RL processes in each optimizer step

We will additionally use U​(h)U(h) to refer to the accelerator’s maximum flops utilization when running typical Transformer kernels at batch size hh.

### A.1 Units

To compare throughputs of different RL approaches it useful to adopt time and throughput units that don’t depend on the particular GPU model and the LLM size. To this end we introduce a time unit called flash:

|  |  |  |  |
| --- | --- | --- | --- |
|  | f=Fg​e​nMf=\frac{F\_{gen}}{M} |  | (9) |

where Fg​e​nF\_{gen} is the number of FLOPs required for one token forward pass for the chosen LLM, and MM is the maximum theoretical FLOPs throughput for the given GPU. The meaning of a flash is the theoretically smallest amortized time that a token generation can take. Thus generating KK tokens will take at least KK flashes, though at a more typical generation utilization of ∼0.1\sim 0.1 rate it will take 10​K10K flashes. For very long sequences Fg​e​nF\_{gen} can vary significantly due to attention FLOPs becoming a large part of total FLOPs, but for simplicity here we will abstract away from this detail.

Having introduced flash ff as the unit, we will measure the system throughput in tokens per flash.

Let τ\tau be the amortized training time per token. τ\tau will be similar at scale for PipelineRL and Conventional RL, because both approaches can benefit from sequence packing.

### A.2 Conventional RL throughput

![Refer to caption](/html/2509.19128/assets/x4.png)


Figure 8: H100 utilization at batch size hh as the ratio of maximum theoretical bf16 FLOPS throughput. We use (4096,h)⋅(h,16384)(4096,h)\cdot(h,16384) matrix multiplications for the measurement. For every hh we consider padding up to h+64h+64 to increase the speed, because empirically we observed large utilization bumps when hh is divisible by a higher power of 2 (up to 128).

We can express Conventional RL throughput as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | rc​o​n​v=Ktc​o​n​vg​e​n+tc​o​n​vt​r​a​i​n,r\_{conv}=\frac{K}{t\_{conv}^{gen}+t\_{conv}^{train}}, |  | (10) |

where tc​o​n​vg​e​nt^{gen}\_{conv} and tc​o​n​vt​r​a​i​nt^{train}\_{conv} are times that generation and training take respectively. Let’s look at these terms closer:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | tc​o​n​vg​e​n\displaystyle t\_{conv}^{gen} | =∑l=1Lh​(l)/N​fU​(h​(l)/N)\displaystyle=\sum\limits\_{l=1}^{L}\frac{h(l)/Nf}{U(h(l)/N)} |  | (11) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | tc​o​n​vt​r​a​i​n\displaystyle t\_{conv}^{train} | =K​τN\displaystyle=\frac{K\tau}{N} |  | (12) |

where h​(l)h(l) is the number of sequences still in progress after ll steps of decoding, and U​(h)U(h) is the GPU utilization at batch size hh. To understand [Equation˜11](#A1.E11 "In A.2 Conventional RL throughput ‣ Appendix A Analyical estimate of PipelineRL speedup for fixed max lag ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation"), recall that generating kk tokens by definitions takes kk flashes under perfect GPU utilization and k/U​(k)k/U(k) at the utilization U​(k)U(k).

We can rewrite this in terms of tokens / flash throughputs:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | rc​o​n​v\displaystyle r\_{conv} | =11rc​o​n​vg​e​n+1rc​o​n​vt​r​a​i​n\displaystyle=\frac{1}{\frac{1}{r\_{conv}^{gen}}+\frac{1}{r\_{conv}^{train}}} |  | (13) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | rc​o​n​vg​e​n\displaystyle r\_{conv}^{gen} | =K∑l=1Lh​(l)/NU​(h​(l)/N)\displaystyle=\frac{K}{\sum\limits\_{l=1}^{L}\frac{h(l)/N}{U(h(l)/N)}} |  | (14) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | rc​o​n​vt​r​a​i​n\displaystyle r\_{conv}^{train} | =Nτ\displaystyle=\frac{N}{\tau} |  | (15) |

At low batch size per GPU at step ll, hN​(l)=h​(l)/Nh\_{N}(l)=h(l)/N, the ratio hN​(l)/U​(hN​(l))h\_{N}(l)/U(h\_{N}(l)) will only decrease very slowly as a function of NN, because for modern GPUs xU​(x)\frac{x}{U(x)} is nearly constant for small xx. This is the formal explanation for Conventional RL’s decreasing efficiency as NN grows.

The maximum token lag in the setup we described above is S−1S-1.

### A.3 PipelineRL throughput

For PipelineRL the system throughput is determined by the slowest pipeline stage. Using the concepts introduced above, the throughput of PipelineRL can be estimated as follows:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | rp​i​p​e​l​i​n​e\displaystyle r\_{pipeline} | =min⁡(rp​i​p​e​l​i​n​eg​e​n,rp​i​p​e​l​i​n​et​r​a​i​n)\displaystyle=\min(r\_{pipeline}^{gen},r\_{pipeline}^{train}) |  | (16) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | rp​i​p​e​l​i​n​eg​e​n\displaystyle r\_{pipeline}^{gen} | =U​(H)​I\displaystyle=U(H)I |  | (17) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | rp​i​p​e​l​i​n​et​r​a​i​n\displaystyle r\_{pipeline}^{train} | =N−Iτ\displaystyle=\frac{N-I}{\tau} |  | (18) |

To understand the maximum lag of Pipeline RL consider the fact that the generation GPUs will produce H​I​LHIL tokens during the time it takes to generate the longest possible sequence of length LL. On average there will be H​I​LL¯\frac{HIL}{\overline{L}} sequences in these tokens. Thus, in the worst case when an optimizer step happened just before the longest sequence generation started, a long sequence will be used for training gm​a​x=⌈H​I​LL¯​B⌉g\_{max}=\lceil\frac{HIL}{\overline{L}{B}}\rceil optimizer steps later than its generation started.

To build a same-lag equivalent for a conventional RL system, one needs to maximize rp​i​p​e​l​i​n​e​(H,I)r\_{pipeline}(H,I) while keeping ⌈H​I​LL¯​B⌉≤S−1\lceil\frac{HIL}{\overline{L}{B}}\rceil\leq S-1. We could found this problem difficult to solve analytically, and performed a straight-forward search of all (H,I)(H,I) configurations for our investigations below.

![Refer to caption](/html/2509.19128/assets/x5.png)


Figure 9: Pipeline RL and Conventional RL throughputs as the function of the maximum lag gm​a​xg\_{max} for a setup with N=128N=128 GPUs and batch size B=NB=N.

### A.4 A PipelineRL speedup case study

To compute the exact throughput boost that PipelineRL brings it is necessary to make assumptions about the sequence length distribution and the hardware that is used for the experiments. For the case-study below, we assume uniform length distribution from 11 to the max length LL and H​100H100 as the GPU. We visualize the GPU utilization table U​(h)U(h) in [Figure˜8](#A1.F8 "In A.2 Conventional RL throughput ‣ Appendix A Analyical estimate of PipelineRL speedup for fixed max lag ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation"). The reader can see that U​(h)U(h) grows almost linearly up to h∼200h\sim 200, which makes it possible to compress the generation on fewer GPUs at a higher utilization. For a setup with N=128N=128 GPUs and training batch B=128B=128 we considered all possible (I,H)(I,H) configurations of PipelineRL and plotted their throughput as a function of the lag gm​a​xg\_{max}. [Figure˜9](#A1.F9 "In A.3 PipelineRL throughput ‣ Appendix A Analyical estimate of PipelineRL speedup for fixed max lag ‣ PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation") shows that PipelineRL can be up to 1.57​x1.57x faster for gm​a​x∼133g\_{max}\sim 133. This lag value can to be too high for many practical setups, but with a higher batch size of e.g. B=2048B=2048 the same number of sequences to be generated by each GPU will correspond to a practical 16x lower lag gm​a​x∼8g\_{max}\sim 8.

The mechanics of how PipelineRL achieved the improvement are as follows:

* •

  rp​i​p​e​l​i​n​eg​e​n=16.9r\_{pipeline}^{gen}=16.9, rp​i​p​e​l​i​n​et​r​a​i​n=17.08r\_{pipeline}^{train}=17.08, 𝐫𝐩𝐢𝐩𝐞𝐥𝐢𝐧𝐞=16.9\mathbf{r\_{pipeline}=16.9}, H=192H=192, I=44I=44
* •

  rc​o​n​vg​e​n=18.3r\_{conv}^{gen}=18.3, rc​o​n​vt​r​a​i​n=26.02r\_{conv}^{train}=26.02, 𝐫𝐜𝐨𝐧𝐯=10.7\mathbf{r\_{conv}=10.7}

Clearly, the root cause of PipelineRL’s speedup is that the 44 generation GPUs can produce 16.9 tokens per flash, that is more efficient than having 128 GPUs produce 18.3 tokens per flash in the Conventional RL case.

## Appendix B Additional Results

![Refer to caption](/html/2509.19128/assets/figures/g64_reward_plot.png)


(a) G=64 reward.

![Refer to caption](/html/2509.19128/assets/figures/g64_ess.png)


(b) G=64 Effective Sample Size.

Figure 10: G=64 diverges.

[◄](/html/2509.19127)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2509.19128)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2509.19128)
[View original  
on arXiv](https://arxiv.org/abs/2509.19128)[►](/html/2509.19129)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Mon Oct 6 23:36:45 2025 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
