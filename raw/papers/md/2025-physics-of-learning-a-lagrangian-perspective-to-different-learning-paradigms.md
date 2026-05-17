---
arxiv: '2509.21049'
authors:
- Siyuan Guo
- Bernhard Schölkopf
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'Physics of Learning: A Lagrangian perspective to different learning paradigms'
url: https://arxiv.org/abs/2509.21049
year: 2025
---

# Physics of Learning: A Lagrangian perspective to different learning paradigms

Siyuan Guo \*
  
Department of Computer Science, University of Cambridge
  
Max Planck Institute for Intelligent Systems
  
United Kingdom & Germany
&Bernhard Schölkopf
  
Max Planck Institute for Intelligent Systems
  
ELLIS Institute Tübingen
  
Germany
  
  

\* Correspondence: siyuan.guo@tuebingen.mpg.de

###### Abstract

We study the problem of building an efficient learning system. Efficient learning processes information in the least time, i.e., building a system that reaches a desired error threshold with the least number of observations. Building upon least action principles from physics, we derive classic learning algorithms, Bellman’s optimality equation in reinforcement learning, and the Adam optimizer in generative models from first principles, i.e., the Learning Lagrangian. We postulate that learning searches for stationary paths in the Lagrangian, and learning algorithms are derivable by seeking the stationary trajectories.

Table 1: Overview of Physics-Inspired Learning Lagrangian. Machine learning encompasses a broad set of paradigms from supervised, unsupervised learning to reinforcement learning and generative models. We postulate that learning also follows a physical law, the principle of least action. We unify different learning paradigms through derivation from the first principles. In particular, we compare the learning Lagrangian with existing physical laws and detail each principle’s suitable application in learning tasks. We derive classical learning algorithms that arise when searching for stationary solutions in the Lagrangian.

|  |  |  |
| --- | --- | --- |
| \columncolorLabelCol\cellcolorLabelCol | \columncolor PhysCol\cellcolorPhysColPhysics | \columncolor LearnCol\cellcolorLearnColLearning |
| \columncolorLabelColFermat’s principle | \columncolor PhysColT=∫AB𝑑tT=\int\_{A}^{B}dt | \columncolor LearnColT=∫ϵ​[∅]ϵ​[𝐬]𝑑tT=\int\_{\epsilon[\emptyset]}^{\epsilon[\mathbf{s}]}dt [\*] |
| \columncolorLabelColHamiltonian | \columncolor PhysColH​(𝐱,𝐩)=𝐩⋅𝐱˙−L​(𝐱,𝐱˙)H(\mathbf{x},\mathbf{p})=\mathbf{p}\cdot\dot{\mathbf{x}}-L(\mathbf{x},\dot{\mathbf{x}}) | \columncolor LearnColH​(𝐬,𝐚,λ)=r​(𝐬,𝐚)+f​(𝐬,𝐚)T​λH(\mathbf{s},\mathbf{a},\lambda)=r(\mathbf{s},\mathbf{a})+f(\mathbf{s},\mathbf{a})^{T}\lambda [†\dagger] |
| \columncolorLabelColthe Lagrangian | \columncolor PhysColL=T−VL=T-V | \columncolor LearnColL​(ℓ,∇θℓ)=12​(∇θℓ)T​F−1​∇θℓ−ℓ​(θ)L(\ell,\nabla\_{\theta}\ell)=\frac{1}{2}(\nabla\_{\theta}\ell)^{T}F^{-1}\nabla\_{\theta}\ell-\ell(\theta) [\*] |
| \columncolorLabelCol | \columncolor PhysColApplications | \columncolor LearnColAlgorithms |
| \columncolorLabelColFermat’s principle | \columncolor PhysColParametric Models | \columncolor LearnColA-optimality (Atkinson et al., [2007](#bib.bib1)) |
| \columncolorLabelColHamiltonian | \columncolor PhysColReinforcement Learning | \columncolor LearnColBellman’s Equation (Bellman, [1958](#bib.bib4)) |
| \columncolorLabelColthe Lagrangian | \columncolor PhysColGenerative Models / Supervised Learning | \columncolor LearnColAdam (Kingma, [2014](#bib.bib18)) / RMSprop (Tieleman, [2012](#bib.bib29)) |

* •

  *Notes:* TT in Fermat’s principle denotes time taken to travel from point AA to point BB; ϵ​[∅],ϵ​[𝐬]\epsilon[\emptyset],\epsilon[\mathbf{s}] is the generalization error after observing zero data to data sequence 𝐬:=s1,s2,…\mathbf{s}:=s\_{1},s\_{2},\ldots; HH is the (physical) Hamiltonian system with position 𝐱\mathbf{x} and momentum 𝐩\mathbf{p} and Lagrangian LL; H​(𝐬,𝐚,λ)H(\mathbf{s},\mathbf{a},\lambda) is the reinforcement learning correspondent with state 𝐬\mathbf{s}, action 𝐚\mathbf{a}, reward r​(𝐬,𝐚)r(\mathbf{s},\mathbf{a}), transition dynamics f​(𝐬,𝐚)f(\mathbf{s},\mathbf{a}) and momentum equivalent λ\lambda; L=T−VL=T-V represents kinetic energy minus potential energy; ℓ\ell denotes some log-likelihood function; ∇θℓ\nabla\_{\theta}\ell is gradient with respect to model parameters θ∈ℝP\theta\in\mathbb{R}^{P}; F−1F^{-1} denotes the inverse Fisher information. Bold symbols are vectors; (⋅)⊤(\cdot)^{\top} is transpose; x˙\dot{x} is derivative with respect to time. The learning Lagrangian indicated via [†\dagger] means it is classic textbook material in control theory (see Todorov ([2006](#bib.bib30))). Learning Lagrangians indicated by [\*] are proposed in this work; to the best of our knowledge, no prior published work exists as of September 2025.

## 1 Introduction

Modern machine learning encompasses a broad set of paradigms — supervised and unsupervised learning, reinforcement learning, and generative models, with deep architectures as the dominant modeling substrate. As momentum built across labs, industry, and policymakers, work shifted toward translating technical advances into products. These efforts have accelerated deployment but also privileged trial-and-error engineering and scale-first heuristics, in part because we still lack a principled understanding of when and why learning emerges, generalizes, and fails. This gap has impeded a systematic methodology for designing sample- and compute-efficient learning systems.

This paper demonstrates a close connection between physics and learning and postulates that learning algorithms arise as stationary trajectories of a learning Lagrangian. This paper presents a first-principles account by casting diverse learning paradigms in a single variational framework. We posit learning Lagrangians and show that algorithms arise as stationary points of their action, thereby providing a unifying perspective to parameter estimation tasks—covering supervised learning and generative modeling—and reinforcement learning. Table [1](#S0.T1 "Table 1 ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms") provides a summary of the paper’s main result. Motivated by physical principles, we postulate the corresponding learning analogy and illustrate its use in suitable learning tasks. By seeking stationary paths of the associated action, we recover classical algorithms.

Related Work. Machine learning and physics have early origins from energy-based models (Hinton, [2025](#bib.bib15); Hopfield, [1982](#bib.bib16)) to their statistical mechanical analysis of memory capacity (Gardner & Derrida, [1988](#bib.bib11)). Kaplan et al. ([2020](#bib.bib17)) show physics-like scaling law emerges as the neural models scale; and recent efforts have begun to analyze this phenomenon using statistical mechanics tools (Cui et al., [2021](#bib.bib8); Sorscher et al., [2022](#bib.bib27); Defilippis et al., [2024](#bib.bib9); Bahri et al., [2024](#bib.bib3); Paquette et al., [2024](#bib.bib24)). Bahri et al. ([2020](#bib.bib2)) give a more recent survey focused on deep models. This paper, on the other hand, studies the relationship between efficient learning and the physics Lagrangian without discussing the choice of model architectures. This work derives algorithms through seeking stationary trajectories, and the commonality shared between different learning paradigms offers a unifying perspective.

Organization of the paper:

* •

  Section [2](#S2 "2 Learning as a deceleration process. ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms") formalizes the connection with kinematic quantities (distance, velocity, acceleration) with Shannon information, deriving the corresponding information-processing velocity and acceleration. Insight [2](#S2 "2 Learning as a deceleration process. ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms") shows that learning is a decelerating process.
* •

  Section [3](#S3 "3 Learning Lagrangians ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms") reviews the relevant physical principles and presents the postulated learning Lagrangians. Solving for stationary trajectories of the associated action recovers classical algorithms in parametric models (Sec. [3.1](#S3.SS1 "3.1 Parametric assumption gives analytical path derivation. ‣ 3 Learning Lagrangians ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms")), reinforcement learning (Sec.[3.2](#S3.SS2 "3.2 Reinforcement Learning as stochastic approximation. ‣ 3 Learning Lagrangians ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms")), and parameter estimation tasks (including supervised learning and generative models)(Sec. [3.3](#S3.SS3 "3.3 Generative Models with postulated Lagrangian ‣ 3 Learning Lagrangians ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms")), thereby offering a unifying perspective across seemingly disparate learning paradigms. We thus hypothesize that learning obeys the Principle of Least Action: searching for stationary paths yields learning algorithms.

## 2 Learning as a deceleration process.

Learning in intelligent systems travels distance not in terms of space but information observed. A data stream until time tt is s1,s2,…,sts\_{1},s\_{2},\ldots,s\_{t}, abbreviated as s≤ts\_{\leq t}. In physics, speed is defined as the rate of change of position with respect to time: v=limΔ​t→0Δ​sΔ​t=d​sd​tv=\lim\_{\Delta t\to 0}\frac{\Delta s}{\Delta t}=\frac{ds}{dt}. In information processing, we define position as the amount of Shannon information (Shannon, [1948](#bib.bib25)) up until time t: I​(s≤t):=log⁡1p​(s≤t)=−log⁡p​(s≤t)I(s\_{\leq t}):=\log\frac{1}{p(s\_{\leq t})}=-\log p(s\_{\leq t}). The rate of change of information content with respect to time, termed as instantaneous velocity in information, is thus derivable as: v=limΔ​t→0I​(s≤t+Δ​t)−I​(s≤t)Δ​tv=\lim\_{\Delta t\to 0}\frac{I(s\_{\leq t+\Delta t})-I(s\_{\leq t})}{\Delta t}.

In discrete information flows (e.g., language tokens) when Δ​t=1\Delta t=1, given a data stream x≤tx\_{\leq{t}}, the velocity at time tt is v​(t)=−log⁡p​(xt∣x<t)v(t)=-\log p(x\_{t}\mid x\_{<t}).
Next token prediction is thus modeling the instantaneous rate of change in information, or instantaneous velocity in information.

To check the consistency between distance and velocity in information processing, we expect it to satisfy basic physics properties, e.g., distance as an integral over velocities.

distance as integral. In discrete time, physical distance satisfies: distance=∑iv​(ti)​Δ​t\text{distance}=\sum\_{i}v(t\_{i})\Delta t. That holds true in information processing too: the total amount of information is the sum of chain-ruled conditional probabilities: I​(x≤t)=−log⁡p​(x1,…,xt)=∑i=1tv​(ti)=−∑i=1tlog⁡p​(xi∣x<i)I(x\_{\leq t})=-\log p(x\_{1},\ldots,x\_{t})=\sum\_{i=1}^{t}v(t\_{i})=-\sum\_{i=1}^{t}\log p(x\_{i}\mid x\_{<i}).

Continuing from understanding kinematic quantities in information processing, acceleration is the instantaneous change in velocity, defined as a=d​vd​t=limΔ​t→0Δ​vΔ​ta=\frac{dv}{dt}=\lim\_{\Delta t\to 0}\frac{\Delta v}{\Delta t}.

acceleration. In discrete information flows, acceleration models the instantaneous change in conditional probability in information processing:

|  |  |  |  |
| --- | --- | --- | --- |
|  | a​(t)=−log⁡p​(xt+2∣x≤t+1)+log⁡p​(xt+1∣x≤t)\displaystyle a(t)=-\log p(x\_{t+2}\mid x\_{\leq{t+1}})+\log p(x\_{t+1}\mid x\_{\leq t}) |  | (1) |

!(/html/2509.21049/assets/figs/icl_loss_iclr.png)

Figure 1: Expected test-time in-context learning velocity and acceleration: (Left) In-context per-token loss ℓt=v​(t)=𝔼​[−log⁡pθ​(xt∣x<t)]\ell\_{t}=v(t)=\mathbb{E}[-\log p\_{\theta}(x\_{t}\mid x\_{<t})]; (Right) In-context per-token difference in loss Δ​ℓt=a​(t)=𝔼​[ℓt+1−ℓt]\Delta\ell\_{t}=a(t)=\mathbb{E}[\ell\_{t+1}-\ell\_{t}]. In-context learning (as shown in the right) is a deceleration process, meaning loss goes down but less quickly as time progresses. A similar phenomenon is expected in training and test loss. Here, in-context loss is evaluated on OpenWebText.

Modelling information processing as kinematics, i.e., movements in physical spaces, prepares to understand the later postulation that learning is searching for stationary trajectories of the action. As trajectories often imply movements in physical space, here we mean movements in information space in the above sense. Considering loss curves, regardless of in-context, train, or test losses, from a kinematics perspective, provides insight [2](#S2 "2 Learning as a deceleration process. ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms"). Figure [1](#S2.F1 "Figure 1 ‣ 2 Learning as a deceleration process. ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms") plots the per-token in-context loss and its discrete first and second differences for small language models, corresponding to the expected test-time in-context learning velocity and acceleration.

Insight No.1 (Learning as a deceleration process: there is a limit infv​(t)\inf v(t).)

Generalization error on the test dataset measuring learning progress is bounded below by 0 or ϵ\epsilon determined by intrinsic uncertainty in data. In-context loss curve, vθ​(t)=−𝔼​[log⁡pθ​(xt∣x<t)]v\_{\theta}(t)=-\mathbb{E}[\log p\_{\theta}(x\_{t}\mid x\_{<t})], vθ​(t)v\_{\theta}(t) is a generally non-increasing function, and thus a generally decelerating process.111Due to the monotone convergence theorem, a bounded below, non-increasing function converges to some limit. We thus hypothesize that learning converges to its infimum.

## 3 Learning Lagrangians

Chollet ([2019](#bib.bib6)) measures intelligence centered around efficiency and generality, namely, when facing new tasks, an intelligent agent should adapt and acquire new skills efficiently. This idea has evolved to community challenges established in ARC-AGI-1, and ARC-AGI-2 (Chollet et al., [2025](#bib.bib7)). The authors believe that intelligence is obtained through efficient learning. This paper is motivated to study the design of an efficient learning system. We present our main postulation below. We first provide a short review of relevant principles in physics and then present the corresponding learning Lagrangians. We then show that searching for the stationary path in the Lagrangians, we recover classic algorithms in different tasks.

Main Postulation (Learning-by-Stationarity)

Learning is searching for the path that makes action governed by the Learning Lagrangian stationary. In particular, learning algorithms (as in equations of motion) are obtained by seeking stationary trajectories.

Review of Principles in Physics.

* •

  Fermat’s Principle / Principle of Least Time (Optics) (Born & Wolf, [2019](#bib.bib5))

  A ray of light travelling from point AA to point BB chooses a path along which the time taken is the least or minimum 222More generally, a ray of light travelling from point AA to point BB choose an optical path that is stationary (i.e., maximum, minimum, extremum), mathematically T=∫AB𝑑t=stationaryT=\int\_{A}^{B}dt=\text{stationary}.. Mathematically,

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | T=mins​∫pathn​𝑑s,\displaystyle T=\min\_{s}\int\_{\text{path}}n\,ds, |  | (2) |

  where n=1vn=\frac{1}{v} is refractive index and vv is the velocity of light in the medium.
* •

  Hamilton’s Principle / Principle of Least Action (Mechanics) (Hamilton, [1834](#bib.bib14))

  The Law states that the actual path ξ​(t)\xi(t) taken by a particle is the path that makes the action SS stationary, where

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | S​[ξ]=∫L​𝑑t=∫T−V​d​t,\displaystyle S[\xi]=\int L\,dt=\int T-V\,dt, |  | (3) |

  where LL is the Lagrangian, with TT kinetic energy and VV potential energy. ξ\xi is the generalized coordinates that specify the configuration of the system.

  A classic example is the Newtonian mechanics for a particle, where ξ\xi is the coordinates of the particle in the system.
  The Lagrangian is L=12​m​|𝐱˙|2−V​(𝐱,t)L=\frac{1}{2}m|\dot{\mathbf{x}}|^{2}-V(\mathbf{x},t).
  Finding the path that makes the action stationary leads to Euler-Lagrangian equation, which gives the equation of motion m​𝐱¨=−∇V=Fm\ddot{\mathbf{x}}=-\nabla V=F.
* •

  Hamiltonian system.
  The Hamiltonian system is the Legendre transform of the Lagrangian:

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | H​(𝐱,𝐩)=𝐩⋅𝐱˙−L​(𝐱,𝐱˙),\displaystyle H(\mathbf{x},\mathbf{p})=\mathbf{p}\cdot\dot{\mathbf{x}}-L(\mathbf{x},\dot{\mathbf{x}}), |  | (4) |

  where 𝐩=∂L∂𝐱˙\mathbf{p}=\frac{\partial L}{\partial\dot{\mathbf{x}}} is the conjugate momentum of 𝐱\mathbf{x}.

Efficient learning is as if designing a physical system’s process of walking along the information path such that it takes the least time to reach the desired error threshold. To make the idea concrete:

In learning, we define a point in space as the generalization error ϵ\epsilon after observing a data sequence 𝐬:={s1,s2,…}\mathbf{s}:=\{s\_{1},s\_{2},\ldots\}. Efficient learning thus means optimizing for a path to reach an error threshold in the shortest time (cf. Fermat’s principle of least time). Mathematically,

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | T​(δ)\displaystyle T(\delta) | =min𝐬​∫0∞Θ​(ϵ​[𝐬]−δ)​𝑑t=min𝐬​∫ϵ​[∅]δd​ϵr​(ϵ,𝐬),\displaystyle=\min\_{\mathbf{s}}\int\_{0}^{\infty}\Theta(\epsilon[\mathbf{s}]-\delta)dt=\min\_{\mathbf{s}}\int\_{\epsilon[\emptyset]}^{\delta}\frac{d\epsilon}{r(\epsilon,\mathbf{s})}, |  | (5) |

where ϵ​[𝐬]\epsilon[\mathbf{s}] is the generalization error after seeing data path 𝐬\mathbf{s} and ϵ​[∅]\epsilon[\emptyset] denotes the generalization error before seeing any data,
and Θ\Theta is an indicator function where Θ​(x)=0,if ​x≤0​ and ​1​ if ​x>0\Theta(x)=0,\text{if }x\leq 0\text{ and }1\text{ if }x>0. Learning velocity333We note that different learning problems with different algorithms have different rates of learning. It is derivable given specific setup and algorithm, though not known a priori., denoted by r​(ϵ,𝐬)r(\epsilon,\mathbf{s}), is the rate of difference in generalization error as information progresses, i.e., rθ​(ϵ,𝐬n)=ϵθ​(𝐬n−1)−ϵθ​(𝐬n)r\_{\theta}(\epsilon,\mathbf{s}\_{n})=\epsilon\_{\theta}(\mathbf{s}\_{n-1})-\epsilon\_{\theta}(\mathbf{s}\_{n}), where the small θ\theta denotes the configuration of the system444Configuration includes but not limited to model parameters, initialization, architecture choice.. The least time is quantified as the least number of observations, assuming similar information content in each observation555Future work can investigate how to quantify time when samples do not contain similar information content.. Thus we propose metrics for evaluation for efficient learning:

* •

  sample-efficient:
  Ts​a​m​p​l​eT\_{sample} = number of samples required to achieve the error threshold.
* •

  compute-efficient: Tc​o​m​p​u​t​eT\_{compute} = computational time taken to achieve the error threshold.

The metrics are proposed based on the learning time of the system indicated from Eq. [5](#S3.E5 "In 3 Learning Lagrangians ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms") and the time in real life to process learning (e.g., parallel processing decreases computational time but does not enable sample-efficient learning). The above makes clear that efficient learning that could increase intrinsic intelligence requires optimization in TsampleT\_{\text{sample}}, and investing in compute only may not be the best solution.

A natural next step is to optimize the given objective. However, we face the technical difficulty of unknown generalization error. The generalization error is derivable given a specific setup and algorithm, but it is not known a priori for optimization.

To address the technical difficulty in optimization with unknown generalization error, we consider the following approaches:

* •

  Parametric assumption. Section [3.1](#S3.SS1 "3.1 Parametric assumption gives analytical path derivation. ‣ 3 Learning Lagrangians ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms") provides a concrete example in linear regression with parametric assumptions. Under suitable assumptions on input standardization, optimizing the Lagrangian given by Fermat’s principle Eq. [5](#S3.E5 "In 3 Learning Lagrangians ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms") yields an analytical optimal solution.

  Remark. Though it is not desirable in practice to constrain model classes with parametric restriction due to model mis-specification, we find it helpful to have an analytical analysis that illustrates some properties for efficient learning (e.g., planning is important).
* •

  Reward Hypothesis. Section [3.2](#S3.SS2 "3.2 Reinforcement Learning as stochastic approximation. ‣ 3 Learning Lagrangians ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms") provides insights on how reinforcement learning circumvents the problem with step-wise progress measured by reward. Writing the Lagrangian in terms of reward gives an equivalent form of Hamiltonian system, and finding the stationary path in the Lagrangian gives rise to Bellman’s optimality equations (Bellman, [1958](#bib.bib4)).

  Remark. Given the reward assumption, we will see in the section the derivation does not give rise to concrete Lagrangian as LL in Eq. [4](#S3.E4 "In 3rd item ‣ 3 Learning Lagrangians ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms") is replaced with reward.
* •

  Postulated Lagrangian. Section [3.3](#S3.SS3 "3.3 Generative Models with postulated Lagrangian ‣ 3 Learning Lagrangians ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms") presents our postulated learning Lagrangian in terms of parameter estimation tasks, covering supervised learning and generative modelling. Operationalizing the learning dynamics of loss field through particle dynamics of the configuration gives rise to θ˙=F−1/2​∇θℓ\dot{\theta}=F^{-1/2}\nabla\_{\theta}\ell that Adam (Kingma, [2014](#bib.bib18)) approximates with diagonalized Fisher for parallel processing.

### 3.1 Parametric assumption gives analytical path derivation.

Consider a linear regression setup: Suppose y=xT​β+ϵy=x^{T}\beta+\epsilon and x∈ℝpx\in\mathbb{R}^{p} and ϵ\epsilon has mean 0 and variance σ2\sigma^{2}.
The generalization error on the standard linear regression is:

|  |  |  |
| --- | --- | --- |
|  | ϵ​(𝐱)=σ2+σ2​tr​((XT​X)−1​𝔼​[x​xT]),\epsilon(\mathbf{x})=\sigma^{2}+\sigma^{2}\text{tr}((X^{T}X)^{-1}\mathbb{E}[xx^{T}]), |  |

where xx is the test data point and 𝐱\mathbf{x} are the sequence of observational points as rows in the data matrix XX. Assuming unit norm assumptions where each observed data point satisfies ‖xi‖2=1,∀i||x\_{i}||\_{2}=1,\forall i and xx is uniformly drawn from the unit sphere 𝕊p−1\mathbb{S}^{p-1}. We work in the classical regime where n≥pn\geq p, so that the data matrix XT​XX^{T}X is invertible and has full rank. Note, by unit norm assumption,

|  |  |  |  |
| --- | --- | --- | --- |
|  | tr​(XT​X)=tr​(∑ixi​xiT)=∑itr​(xi​xiT)=∑i‖xi‖22=n.\displaystyle\text{tr}(X^{T}X)=\text{tr}(\sum\_{i}x\_{i}x\_{i}^{T})=\sum\_{i}\text{tr}(x\_{i}x\_{i}^{T})=\sum\_{i}||x\_{i}||\_{2}^{2}=n. |  | (6) |

Further 𝔼​[x​xT]=1p​Ip\mathbb{E}[xx^{T}]=\frac{1}{p}I\_{p} due to uniform sampling over 𝕊p−1\mathbb{S}^{p-1}.
Optimizing the Lagrangian shown in Eq. [5](#S3.E5 "In 3 Learning Lagrangians ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms"), we would like to choose the observational data path 𝐱\mathbf{x} such that ϵ​(𝐱)\epsilon(\mathbf{x}) is minimized with the least number of observations.
Since S:=XT​XS:=X^{T}X is a real symmetric matrix, by the spectral theorem, there exists an orthogonal QQ and a real diagonal matrix Λ\Lambda such that S=Q​Λ​QTS=Q\Lambda Q^{T}. Then S−1=Q​Λ−1​QTS^{-1}=Q\Lambda^{-1}Q^{T} and tr​(S−1)=tr​(Λ−1​QT​Q)=∑i1λi\text{tr}(S^{-1})=\text{tr}(\Lambda^{-1}Q^{T}Q)=\sum\_{i}\frac{1}{\lambda\_{i}}. The problem of optimizing the data path:

|  |  |  |  |
| --- | --- | --- | --- |
|  | min𝐱:‖xi‖2=1​∫0∞Θ​(ϵ​(𝐱)−δ)​𝑑t\displaystyle\min\_{\mathbf{x}:||x\_{i}||\_{2}=1}\int\_{0}^{\infty}\Theta(\epsilon(\mathbf{x})-\delta)dt |  | (7) |

translates to min⁡1p​∑i=1p1λi\min\frac{1}{p}\sum\_{i=1}^{p}\frac{1}{\lambda\_{i}} subject to ∑i=1pλi=n\sum\_{i=1}^{p}\lambda\_{i}=n.
By convexity function t→1tt\to\frac{1}{t} and Jensen’s inequality, one has

|  |  |  |
| --- | --- | --- |
|  | 1p​∑i1λi≥p∑iλi=pn\frac{1}{p}\sum\_{i}\frac{1}{\lambda\_{i}}\geq\frac{p}{\sum\_{i}\lambda\_{i}}=\frac{p}{n} |  |

The inequality is achieved when λi=np\lambda\_{i}=\frac{n}{p}, thus minimum is attained at 1p​∑i=1p1λi=pn\frac{1}{p}\sum\_{i=1}^{p}\frac{1}{\lambda\_{i}}=\frac{p}{n}.
Then

|  |  |  |
| --- | --- | --- |
|  | min𝐱⁡ϵ​(𝐱)=σ2+σ2​pn\min\_{\mathbf{x}}\epsilon(\mathbf{x})=\sigma^{2}+\sigma^{2}\frac{p}{n} |  |

As noted before in Section [2](#S2 "2 Learning as a deceleration process. ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms"), dependent on specific problem setup, there is an irreducible generalization error (σ2\sigma^{2} in this case), and the generalization error ranges from (σ2,2​σ2](\sigma^{2},2\sigma^{2}] due to n≥pn\geq p.
For example, to reach ϵ​(𝐱)=2​σ2\epsilon(\mathbf{x})=2\sigma^{2}, the minimum sample required is pp and XX could be any orthogonal matrix QQ. To reach ϵ​(𝐱)=1.5​σ2\epsilon(\mathbf{x})=1.5\sigma^{2}, the minimum sample required is 2​p2p and X=2​VX=\sqrt{2}V, where VV could be any (real) Stiefel matrix. The analytical example shows us that given parametric assumptions on function classes and input distribution, it is possible to choose the observation matrix most efficiently for reducing generalization error. This is a special case for A-optimality (Atkinson et al., [2007](#bib.bib1)) in linear regression setting.

A natural follow-up question is whether there is a data solution path such that adding more data points always stays along the optimal path? A short answer is no as XT​X=∑xi​xiTX^{T}X=\sum x\_{i}x\_{i}^{T} and adding one single data point to maintain S=np​IpS=\frac{n}{p}I\_{p} implies the added point has the property xi​xiT=1p​Ipx\_{i}x\_{i}^{T}=\frac{1}{p}I\_{p}, which is impossible due to rank difference between 11 and pp. However, adding blocks of pp new data points is possible, planning pp-steps ahead in this case.

Insight No.2

Planning is needed to learn continuously in the most efficient way.

### 3.2 Reinforcement Learning as stochastic approximation.

This section builds on two insights:

* •

  optimizing action/policy is implicitly optimizing the data path or state path in RL terms, for learning, cf. min𝐬\min\_{\mathbf{s}} in Eq. [5](#S3.E5 "In 3 Learning Lagrangians ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms").
* •

  The reward hypothesis circumvents the problem of unknown generalization error.

In fact, searching the stationary points in the Lagrangian written from a reward perspective derives Bellman’s optimality equation (Bellman, [1958](#bib.bib4)), the backbone of many RL algorithms, e.g., policy iteration, value iteration (Sutton & Barto, [2018](#bib.bib28)), Q-learning (Watkins & Dayan, [1992](#bib.bib31)), Deep Q-learning (Mnih et al., [2013](#bib.bib21)).

Reward Hypothesis. All goals can be represented by rewards (Sutton & Barto, [2018](#bib.bib28)).

Reinforcement learning circumvents the problem of unknown generalization error through measuring step-wise progress through reward r​(𝐬,𝐚)r(\mathbf{s},\mathbf{a}) on its current state 𝐬\mathbf{s} and next action 𝐚\mathbf{a}. In other words, the value function V​(𝐬)V(\mathbf{s}) is the path to maximize reward, and the optimization over min𝐬\min\_{\mathbf{s}} is through finding the optimal policy reaching the optimal path V⋆​(𝐬)V\_{\star}(\mathbf{s}). Greydanus & Olah ([2019](#bib.bib12)) provides an intuitive playground on how value function can be viewed from a path perspective. Note that the exact quantification of optimal can be incorporated appropriately through designing the reward function.

Next, we demonstrate that searching for the stationary points in the Lagrangian defined in the RL setting gives commonly known learning algorithms, i.e., Bellman’s optimality equation. We do not claim novelty in this derivation, as it is textbook material in classic control theory, see Pontryagin’s maximum principle (Kirk, [1970](#bib.bib19)), Hamilton-Jacobi-Bellman equations for the continuous case (Evans, [2010](#bib.bib10)); we include it to demonstrate the support of our main postulation that learning is searching for stationary points in the Lagrangian, and finding stationary points gives rise to classic learning algorithms.

Derivation of Bellman equation from the Lagrangian.
The goal of the learning problem is to find actions (𝐚0,𝐚1,…,𝐚n−1)(\mathbf{a}\_{0},\mathbf{a}\_{1},\ldots,\mathbf{a}\_{n-1}) and states (𝐬0,𝐬1,…,𝐬n)(\mathbf{s}\_{0},\mathbf{s}\_{1},\ldots,\mathbf{s}\_{n}) to maximize the objective function JJ, where

|  |  |  |  |
| --- | --- | --- | --- |
|  | J=h​(𝐬n)+∫0tfr​(𝐬t,𝐚t,t)​𝑑t\displaystyle J=h(\mathbf{s}\_{n})+\int\_{0}^{t\_{f}}r(\mathbf{s}\_{t},\mathbf{a}\_{t},t)dt |  | (8) |

subject to constraints 𝐬k+1=f​(𝐬k,𝐚k)\mathbf{s}\_{k+1}=f(\mathbf{s}\_{k},\mathbf{a}\_{k}) and tft\_{f} is final time. This assumes a deterministic transition where the next state is uniquely determined by its action. And h​(𝐬n)h(\mathbf{s}\_{n}) is the terminal reward. Turning the above problem into a constrained optimization problem with Lagrangians:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ​({𝐬},{𝐚},λ)=h​(𝐬n)+∑k=0n−1(r​(𝐬k,𝐚k,k)+(f​(𝐬k,𝐚k)−𝐬k+1)T​λk+1)\displaystyle\mathcal{L}(\{\mathbf{s}\},\{\mathbf{a}\},\lambda)=h(\mathbf{s}\_{n})+\sum\_{k=0}^{n-1}\big(r(\mathbf{s}\_{k},\mathbf{a}\_{k},k)+(f(\mathbf{s}\_{k},\mathbf{a}\_{k})-\mathbf{s}\_{k+1})^{T}\lambda\_{k+1}\big) |  | (9) |

Learning a stationary solution for the Lagrangian means we search for solutions that satisfy ∂ℒ∂𝐬k=0\frac{\partial\mathcal{L}}{\partial\mathbf{s}\_{k}}=0, ∂ℒ∂𝐚k=0\frac{\partial\mathcal{L}}{\partial\mathbf{a}\_{k}}=0 for all kk and ∂ℒ∂λ=0\frac{\partial\mathcal{L}}{\partial\lambda}=0.
Define discrete-time Hamiltonian:

|  |  |  |  |
| --- | --- | --- | --- |
|  | H(k)​(𝐬,𝐚,λ)=r​(𝐬,𝐚,k)+f​(𝐬,𝐚)T​λ\displaystyle H^{(k)}(\mathbf{s},\mathbf{a},\lambda)=r(\mathbf{s},\mathbf{a},k)+f(\mathbf{s},\mathbf{a)}^{T}\lambda |  | (10) |

Re-writing the Lagrangian in Eq. [9](#S3.E9 "In 3.2 Reinforcement Learning as stochastic approximation. ‣ 3 Learning Lagrangians ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms") gives Eq. [11](#S3.E11 "In 3.2 Reinforcement Learning as stochastic approximation. ‣ 3 Learning Lagrangians ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms") :

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ℒ\displaystyle\mathcal{L} | =h​(𝐬n)−𝐬nT​λn+𝐬0T​λ0+∑k=0n−1(H(k)​(𝐬k,𝐚k,λk+1)−𝐬kT​λk)\displaystyle=h(\mathbf{s}\_{n})-\mathbf{s}\_{n}^{T}\lambda\_{n}+\mathbf{s}\_{0}^{T}\lambda\_{0}+\sum\_{k=0}^{n-1}(H^{(k)}(\mathbf{s}\_{k},\mathbf{a}\_{k},\lambda\_{k+1})-\mathbf{s}\_{k}^{T}\lambda\_{k}\big) |  | (11) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | d​ℒ\displaystyle d\mathcal{L} | =(∇𝐬h​(𝐬n)−λn)T​d​𝐬n+λ0T​d​𝐬0+∑k=0n−1(∂H(k)∂𝐬k−λk)T​d​𝐬k+(∂H(k)∂𝐚k)T​d​𝐚k\displaystyle=(\nabla\_{\mathbf{s}}h(\mathbf{s}\_{n})-\lambda\_{n})^{T}d\mathbf{s}\_{n}+\lambda\_{0}^{T}d\mathbf{s}\_{0}+\sum\_{k=0}^{n-1}\big(\frac{\partial H^{(k)}}{\partial\mathbf{s}\_{k}}-\lambda\_{k}\big)^{T}d\mathbf{s}\_{k}+(\frac{\partial H^{(k)}}{\partial\mathbf{a}\_{k}})^{T}d\mathbf{a}\_{k} |  | (12) |

With the initial position fixed (d​𝐬0=0d\mathbf{s}\_{0}=0), we search for solutions that lead to other terms of variations being 0. This leads to solutions that satisfy constraints below:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | λn\displaystyle\lambda\_{n} | =∇𝐬h​(𝐬n)\displaystyle=\nabla\_{\mathbf{s}}h(\mathbf{s}\_{n}) |  | (13) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | λk\displaystyle\lambda\_{k} | =∂r​(𝐬k,𝐚k,k)∂𝐬k+∂f​(𝐬k,𝐚k)∂𝐬kT​λk+1\displaystyle=\frac{\partial r(\mathbf{s}\_{k},\mathbf{a}\_{k},k)}{\partial\mathbf{s}\_{k}}+\frac{\partial f(\mathbf{s}\_{k},\mathbf{a}\_{k})}{\partial\mathbf{s}\_{k}}^{T}\lambda\_{k+1} |  | (14) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝐚k\displaystyle\mathbf{a}\_{k} | =arg⁡maxu⁡H(k)​(𝐬k,u,λk+1)⟹∂H(k)∂𝐚k=0\displaystyle=\arg\max\_{u}H^{(k)}(\mathbf{s}\_{k},u,\lambda\_{k+1})\implies\frac{\partial H^{(k)}}{\partial\mathbf{a}\_{k}}=0 |  | (15) |

Given h​(𝐬n)h(\mathbf{s}\_{n}) is the terminal reward and λn\lambda\_{n} is the derivative of the terminal return with respect to state. That means in RL terms λn=∇sV​(𝐬n)\lambda\_{n}=\nabla\_{s}V(\mathbf{s}\_{n}). Suppose λk=∇sV​(𝐬k)\lambda\_{k}=\nabla\_{s}V(\mathbf{s}\_{k}). Mathematically, differentiating Eq. [16](#S3.E16 "In 3.2 Reinforcement Learning as stochastic approximation. ‣ 3 Learning Lagrangians ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms") with respect to 𝐬k\mathbf{s}\_{k} gives Eq. [14](#S3.E14 "In 3.2 Reinforcement Learning as stochastic approximation. ‣ 3 Learning Lagrangians ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms"):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | V​(𝐬k)\displaystyle V(\mathbf{s}\_{k}) | =r​(𝐬k,𝐚k,k)+V​(𝐬k+1)=r​(𝐬k,𝐚k,k)+V​(f​(𝐬k,𝐚k))\displaystyle=r(\mathbf{s}\_{k},\mathbf{a}\_{k},k)+V(\mathbf{s}\_{k+1})=r(\mathbf{s}\_{k},\mathbf{a}\_{k},k)+V(f(\mathbf{s}\_{k},\mathbf{a}\_{k})) |  | (16) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∇𝐬V​(𝐬k)\displaystyle\nabla\_{\mathbf{s}}V(\mathbf{s}\_{k}) | =∂r​(𝐬k,𝐚k,k)∂𝐬k+∂f​(𝐬k,𝐚k)∂𝐬kT​∇𝐬V​(𝐬k+1)\displaystyle=\frac{\partial r(\mathbf{s}\_{k},\mathbf{a}\_{k},k)}{\partial\mathbf{s}\_{k}}+\frac{\partial f(\mathbf{s}\_{k},\mathbf{a}\_{k})}{\partial\mathbf{s}\_{k}}^{T}\nabla\_{\mathbf{s}}V(\mathbf{s}\_{k+1}) |  | (17) |

Combining with Eq. [15](#S3.E15 "In 3.2 Reinforcement Learning as stochastic approximation. ‣ 3 Learning Lagrangians ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms"), the solution needs to satisfy constraints:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | V​(𝐬k)\displaystyle V(\mathbf{s}\_{k}) | =maxu⁡{r​(𝐬k,u,k)+V​(f​(𝐬k,u))}\displaystyle=\max\_{u}\{r(\mathbf{s}\_{k},u,k)+V(f(\mathbf{s}\_{k},u))\} |  | (18) |

It is not hard to see, in probabilistic transitions where the Lagrangian involves integral over randomness in the environment, the solution that satisfies being the stationary path gives:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | V​(𝐬k)\displaystyle V(\mathbf{s}\_{k}) | =maxu⁡(r​(𝐬k,u,k)+𝔼​[V​(Sk+1)])\displaystyle=\max\_{u}(r(\mathbf{s}\_{k},u,k)+\mathbb{E}[V(S\_{k+1})]) |  | (19) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | uk\displaystyle u\_{k} | =arg⁡maxu⁡(r​(𝐬k,u,k)+𝔼​[V​(Sk+1)])\displaystyle=\arg\max\_{u}(r(\mathbf{s}\_{k},u,k)+\mathbb{E}[V(S\_{k+1})]) |  | (20) |

This is the classic Bellman optimality equation.

Insight No.3

The stationary path in the Lagrangian, written in terms of rewards, should satisfy Bellman’s optimality equation. Thus, optimizing Bellman’s equation is searching for the stationary path.

Remark. Recall the Hamiltonian system:

|  |  |  |  |
| --- | --- | --- | --- |
|  | H​(𝐱,𝐩)=𝐩⋅𝐱˙−L​(𝐱,𝐱˙),\displaystyle H(\mathbf{x},\mathbf{p})=\mathbf{p}\cdot\dot{\mathbf{x}}-L(\mathbf{x},\dot{\mathbf{x}}), |  | (21) |

where 𝐩\mathbf{p} is the conjugate momentum of 𝐱\mathbf{x} and 𝐩=∂L∂𝐱˙\mathbf{p}=\frac{\partial L}{\partial\dot{\mathbf{x}}}. From the above derivation in discrete-time Hamiltonian, we saw that momentum 𝐩\mathbf{p} is λ\lambda and 𝐱˙\dot{\mathbf{x}} is the transition dynamics f​(𝐬,𝐚)f(\mathbf{s},\mathbf{a}), and as noted the Lagrangian or rate of decrease in generalization error as information progresses is replaced with step-wise reward r​(𝐬,𝐚)r(\mathbf{s},\mathbf{a}). Reinforcement learning thus performs well in settings with well-defined rewards, e.g., games (Mnih et al., [2015](#bib.bib22)), chess (Silver et al., [2017](#bib.bib26)), or verifiable problems like mathematics (Guo et al., [2025](#bib.bib13)) though the lack of intermediate rewards for math problems may lead to inefficiency in search, thus large-scale training. Applying RL in real-world applications without clear rewards thus requires a carefully designed reward model, e.g., reinforcement learning from human feedback (Ouyang et al., [2022](#bib.bib23); Lambert, [2025](#bib.bib20)). However, for our purposes, the above derivation does not show the learning Lagrangian. In the section below, we postulate the learning Lagrangian and provide reasons for our postulation.

### 3.3 Generative Models with postulated Lagrangian

In search of a design of an efficient learning system, we started from the equivalent learning Lagrangian from Fermat’s Principle, to a reward-based Hamiltonian system. Efficient learning transitions from traveling on the path that takes the least time to its more general mechanical form as searching for the stationary path to minimize action.

A naïve understanding from discussions in previous sections (see Fermat’s principle) would lead to the conclusion that supervised learning is less efficient than reinforcement learning, due to a lack of optimization over the data path 𝐬\mathbf{s}. In this section, we show that this is not the case. We present our postulated Lagrangian and posit that reinforcement learning is the Legendre transform of parameter estimation tasks, in the same sense as a Hamiltonian system is the Legendre transform of the Lagrangian, such that they share the same optimal solutions.

In generative models, given a dataset 𝒟:={𝐱1,𝐱2,…,𝐱n}\mathcal{D}:=\{\mathbf{x}\_{1},\mathbf{x}\_{2},\ldots,\mathbf{x}\_{n}\}, we search for parameter θ\theta that learns how the data are distributed pθ​(𝐱)p\_{\theta}(\mathbf{x}). Similarly, in supervised learning, we learn a conditional distribution pθ​(y∣x)p\_{\theta}(y\mid x) from either labelled pairs for classification tasks, or regression tasks. Both learning problems, from generative modelling to supervised learning, are parameter estimation problems.

In statistical estimation tasks, we search for an estimator θ^\hat{\theta} that maximizes the likelihood function. Here, we are not only interested in finding an estimator that best models data, but we are also looking for an efficient statistical estimator. The Cramér-Rao lower bound states

Let θ^\hat{\theta} be an unbiased estimator of the unknown parameter θ\theta. Then under regularity conditions,

|  |  |  |  |
| --- | --- | --- | --- |
|  | Var​(θ^)−I−1​(θ),\displaystyle\text{Var}(\hat{\theta})-I^{-1}(\theta), |  | (22) |

is positive semi-definite. In particular, an unbiased estimator θ^\hat{\theta} attains the lower bound, i.e., Var​(θ^)=I−1​(θ)\text{Var}(\hat{\theta})=I^{-1}(\theta) is an efficient estimator. Here I​(θ)I(\theta) is known as the Fisher information and defined as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | I​(θ)\displaystyle I(\theta) | :=𝔼​[(∇θℓ​(θ;x))​(∇θℓ​(θ;x))T]\displaystyle:=\mathbb{E}[(\nabla\_{\theta}\ell(\theta;x))(\nabla\_{\theta}\ell(\theta;x))^{T}] |  | (23) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =−𝔼​[∂2∂θ​∂θT​ℓ​(θ;x)]\displaystyle=-\mathbb{E}\big[\frac{\partial^{2}}{\partial\theta\partial\theta^{T}}\ell(\theta;x)\big] |  | (24) |

where ℓ​(θ;x)\ell(\theta;x) is the log-likelihood function. From hereon, we state the postulation.

Postulation: Consider the loss function ℓ​(θ,t)\ell(\theta,t) as a field666Here we meant by physical field. defined at every point of configuration (θ,t)(\theta,t). The dynamics of the field is governed by the Lagrangian dynamics:

|  |  |  |  |
| --- | --- | --- | --- |
|  | S=∫t𝑑t​∫θ𝑑θ​∫xp​(x)​𝑑x​ℒ​(ℓ,∂ℓ∂t,∂ℓ∂θ,θ,t)\displaystyle S=\int\_{t}dt\int\_{\theta}d\theta\int\_{x}p(x)dx\mathcal{L}(\ell,\frac{\partial\ell}{\partial t},\frac{\partial\ell}{\partial\theta},\theta,t) |  | (25) |

The integral over xx is due to batched sampling over data. Given the loss function in current machine learning paradigm does not depend on time, and knowing potential energy is a static term corresponding to some intrinsic property of the estimation task, we postulate it to be some log-likelihood function ℓ​(θ;x)\ell(\theta;x); knowing kinetic energy takes a quadratic form and taking into account searching for an efficient estimator, we thus hypothesize that Lagrangian takes the form of:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ​(ℓ,∇θℓ)=T−V=12​P​(∇θℓ)T​F​(θ)−1​(∇θℓ)−ℓ​(θ;x)\displaystyle\mathcal{L}(\ell,\nabla\_{\theta}\ell)=T-V=\frac{1}{2P}(\nabla\_{\theta}\ell)^{T}F(\theta)^{-1}(\nabla\_{\theta}\ell)-\ell(\theta;x) |  | (26) |

where PP is the number of model parameters, i.e., θ∈ℝP\theta\in\mathbb{R}^{P} and FF denotes Fisher information. Given the postulated Lagrangian, we expect the solution at the stationary points to satisfy the Euler-Lagrangian equation for scalar field theory with expectation adjusted:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝔼​[∂ℒ∂ℓ]\displaystyle\mathbb{E}[\frac{\partial\mathcal{L}}{\partial\ell}] | =𝔼​[∂∂t​(∂ℒ∂ℓ˙)+∑i∂∂θi​∂ℒ∂(∂ℓ/∂θi)]\displaystyle=\mathbb{E}\Big[\frac{\partial}{\partial t}(\frac{\partial\mathcal{L}}{\partial\dot{\ell}})+\sum\_{i}\frac{\partial}{\partial\theta\_{i}}\frac{\partial\mathcal{L}}{\partial(\partial\ell/\partial\theta\_{i})}\Big] |  | (27) |

The left-hand side is −1-1 and due to ℒ\mathcal{L} has no ℓ˙\dot{\ell} term, the first term in the right-hand side is 0. The second term in the right-hand side can be re-written as 𝔼​[∇θ⋅∂ℒ∂∇θl]\mathbb{E}[\nabla\_{\theta}\cdot\frac{\partial\mathcal{L}}{\partial\nabla\_{\theta}l}]. Thus,

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | −1\displaystyle-1 | =𝔼​[∇θ⋅∂ℒ∂∇θl]\displaystyle=\mathbb{E}[\nabla\_{\theta}\cdot\frac{\partial\mathcal{L}}{\partial\nabla\_{\theta}l}] |  | (28) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | −1\displaystyle-1 | =1P​𝔼​[∇θ⋅(F−1​∇θl)]due to ​∂ℒ∂∇θℓ=F−1​∇θℓ\displaystyle=\frac{1}{P}\mathbb{E}[\nabla\_{\theta}\cdot(F^{-1}\nabla\_{\theta}l)]\quad\text{due to }\frac{\partial\mathcal{L}}{\partial\nabla\_{\theta}\ell}=F^{-1}\nabla\_{\theta}\ell |  | (29) |

Note that the divergence of a vector is the trace of the gradient of the vector. Note the Fisher does not depend on the randomness of xx as it already takes expectation over xx, we have:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | −1\displaystyle-1 | =1Ptr(∇θ(F(θ)−1)𝔼​[∇θl]⏟=0​ at stationary points+F−1𝔼[∇θ2l]])=1Ptr(F−1𝔼​[∇θ2l]⏟=−F)=−1\displaystyle=\frac{1}{P}\text{tr}(\nabla\_{\theta}(F(\theta)^{-1})\underbrace{\mathbb{E}[\nabla\_{\theta}l]}\_{=0\text{ at stationary points}}+F^{-1}\mathbb{E}[\nabla\_{\theta}^{2}l]])=\frac{1}{P}\text{tr}\big(F^{-1}\underbrace{\mathbb{E}[\nabla\_{\theta}^{2}l]}\_{=-F}\big)=-1 |  | (30) |

We thus observe (unsurprisingly) that the solution at stationary points for the parameter estimation task needs to be a maximum likelihood estimator.

The learning dynamics of loss fields needs to be operationalized through changes in particle dynamics where each parameter in the configuration θ\theta is governed by L=T−V=12​m​θ˙T​θ˙−V​(θ,t)L=T-V=\frac{1}{2}m\dot{\theta}^{T}\dot{\theta}-V(\theta,t). Re-writing the postulated Lagrangian, we have θ˙=F−1/2​∇θl\dot{\theta}=F^{-1/2}\nabla\_{\theta}l where the mass of the system is the inverse number of model parameters m=1Pm=\frac{1}{P} and FF is a symmetric and positive semi-definite matrix. In optimization, given unknown observed Fisher, we approximate using the empirical Fisher. Both RMSprop (Tieleman, [2012](#bib.bib29)) and Adam (Kingma, [2014](#bib.bib18)) have update based on F−1/2​∇θℓF^{-1/2}\nabla\_{\theta}\ell:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | RMSprop: ​θt+1\displaystyle\textit{RMSprop: }\theta\_{t+1} | ←θt−α​gtvt+ϵ,\displaystyle\leftarrow\theta\_{t}-\alpha\frac{g\_{t}}{\sqrt{v\_{t}}+\epsilon}, |  | (31) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Adam: ​θt+1\displaystyle\textit{Adam: }\theta\_{t+1} | ←θt−α​m^tv^t+ϵ,\displaystyle\leftarrow\theta\_{t}-\alpha\frac{\hat{m}\_{t}}{\sqrt{\hat{v}\_{t}}+\epsilon}, |  | (32) |

where gt=∇θtℓg\_{t}=\nabla\_{\theta\_{t}}\ell, vt=β2​vt−1+(1−β2)​gt⊙gtv\_{t}=\beta\_{2}v\_{t-1}+(1-\beta\_{2})g\_{t}\odot g\_{t}, and mt=β1​mt−1+(1−β1)​gtm\_{t}=\beta\_{1}m\_{t-1}+(1-\beta\_{1})g\_{t}, m^t=mt1−β1t\hat{m}\_{t}=\frac{m\_{t}}{1-\beta\_{1}^{t}}, v^t=vt1−β2t\hat{v}\_{t}=\frac{v\_{t}}{1-\beta\_{2}^{t}}, and ϵ\epsilon are added for numerical stability.
From the Lagrangian, one can also predict the inefficiency of SGD, as it does not satisfy the Euler-Lagrange equation. Combining with Section [3.2](#S3.SS2 "3.2 Reinforcement Learning as stochastic approximation. ‣ 3 Learning Lagrangians ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms") on the relationship with reinforcement learning and Hamiltonian system, we thus posit our insight:

Insight No.4

Reinforcement learning is the Legendre transform of parameter estimation tasks under Adam / RMSprop optimization.

## 4 Conclusion

Motivated by the study of efficient learning through physics, we find surprising synergies between different physics principles and different learning paradigms, from active data selection, reinforcement learning, to parameter estimation tasks. We assay the results in Section [3](#S3 "3 Learning Lagrangians ‣ Physics of Learning: A Lagrangian perspective to different learning paradigms") and derive classic learning algorithms from seeking stationary trajectories in the Lagrangian, offering a unifying perspective to seemingly broad and different learning paradigms. As any intriguing hypothesis needs experimental verification, a natural next step is to design verifiable experiments. Though at the current status, we find our insights with mathematical justification provide a diverse range of postulations about synergies across different fields that could require community efforts to test and verify.

#### Ethics statement

The paper aims to understand the fundamentals of learning and intelligence. We demonstrate a close connection between physics and learning and postulate that learning, too, follows physical laws. This work promotes the importance of AI safety and ethics, as machine learning, like other engines or entities, obeys the laws of Nature. This paper presents a principled, promising approach to designing safer AI through understanding the fundamental laws behind learning.

#### Reproducibility Statement

The paper includes theoretical derivations within the paper and experiment results are easily reproducible through public sources.

#### The Use of Large Language Models

Large language models are used to polish academic writing, search for references, and provide hints for mathematical proofs with concrete prompts. Large language models are very helpful as an assisted tool, but it still cannot directly contribute to the paper’s main contribution.

## References

* Atkinson et al. (2007)

  Anthony C. Atkinson, Alexander N. Donev, and Randall D. Tobias.
  *Optimum Experimental Designs, with SAS*.
  Oxford University Press, 2007.
* Bahri et al. (2020)

  Yasaman Bahri, Jonathan Kadmon, Jeffrey Pennington, Sam S. Schoenholz, Jascha Sohl-Dickstein, and Surya Ganguli.
  Statistical mechanics of deep learning.
  11:501–528, 2020.
  ISSN 1947-5462.
  doi: https://doi.org/10.1146/annurev-conmatphys-031119-050745.
  URL <https://www.annualreviews.org/content/journals/10.1146/annurev-conmatphys-031119-050745>.
  Publisher: Annual Reviews Type: Journal Article.
* Bahri et al. (2024)

  Yasaman Bahri, Ethan Dyer, Jared Kaplan, Jaehoon Lee, and Utkarsh Sharma.
  Explaining neural scaling laws.
  *Proceedings of the National Academy of Sciences*, 121(27):e2311878121, 2024.
* Bellman (1958)

  Richard Bellman.
  Dynamic programming and stochastic control processes.
  1(3):228–239, 1958.
  ISSN 0019-9958.
  doi: https://doi.org/10.1016/S0019-9958(58)80003-0.
* Born & Wolf (2019)

  Max Born and Emil Wolf.
  *Principles of Optics: 60th Anniversary Edition*.
  Cambridge University Press, 7 edition, 2019.
* Chollet (2019)

  François Chollet.
  On the measure of intelligence.
  *arXiv preprint arXiv:1911.01547*, 2019.
* Chollet et al. (2025)

  Francois Chollet, Mike Knoop, Gregory Kamradt, Bryan Landers, and Henry Pinkard.
  Arc-agi-2: A new challenge for frontier ai reasoning systems.
  *arXiv preprint arXiv:2505.11831*, 2025.
* Cui et al. (2021)

  Hugo Cui, Bruno Loureiro, Florent Krzakala, and Lenka Zdeborová.
  Generalization error rates in kernel regression: The crossover from the noiseless to noisy regime.
  *Advances in Neural Information Processing Systems*, 34:10131–10143, 2021.
* Defilippis et al. (2024)

  Leonardo Defilippis, Bruno Loureiro, and Theodor Misiakiewicz.
  Dimension-free deterministic equivalents and scaling laws for random feature regression.
  *Advances in Neural Information Processing Systems*, 37:104630–104693, 2024.
* Evans (2010)

  Lawrence C. Evans.
  *Partial Differential Equations*.
  American Mathematical Society, 2nd edition, 2010.
  See Chapter 10 on Hamilton–Jacobi and HJB.
* Gardner & Derrida (1988)

  Elizabeth Gardner and Bernard Derrida.
  Optimal storage properties of neural network models.
  *Journal of Physics A: Mathematical and general*, 21(1):271, 1988.
* Greydanus & Olah (2019)

  Sam Greydanus and Chris Olah.
  The paths perspective on value learning.
  *Distill*, 2019.
  doi: 10.23915/distill.00020.
  https://distill.pub/2019/paths-perspective-on-value-learning.
* Guo et al. (2025)

  Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Honghui Ding, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jingchang Chen, Jingyang Yuan, Jinhao Tu, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaichao You, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J.
  Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying
  Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang.
  DeepSeek-r1 incentivizes reasoning in LLMs through reinforcement learning.
  *Nature*, 2025.
* Hamilton (1834)

  William Rowan Hamilton.
  XV. on a general method in dynamics; by which the study of the motions of all free systems of attracting or repelling points is reduced to the search and differentiation of one central relation, or characteristic function.
  124:247–308, 1834.
  doi: 10.1098/rstl.1834.0017.
* Hinton (2025)

  Geoffrey Hinton.
  Nobel lecture: Boltzmann machines.
  *Rev. Mod. Phys.*, 97:030502, Aug 2025.
  doi: 10.1103/RevModPhys.97.030502.
  URL <https://link.aps.org/doi/10.1103/RevModPhys.97.030502>.
* Hopfield (1982)

  J J Hopfield.
  Neural networks and physical systems with emergent collective computational abilities.
  *Proceedings of the National Academy of Sciences*, 79(8):2554–2558, 1982.
* Kaplan et al. (2020)

  Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei.
  Scaling laws for neural language models.
  *arXiv preprint arXiv:2001.08361*, 2020.
* Kingma (2014)

  Diederik P Kingma.
  Adam: A method for stochastic optimization.
  *arXiv preprint arXiv:1412.6980*, 2014.
* Kirk (1970)

  Donald E. Kirk.
  *Optimal Control Theory: An Introduction*.
  Prentice-Hall, 1970.
  Dover reprint, 2004.
* Lambert (2025)

  Nathan Lambert.
  Reinforcement learning from human feedback.
  *arXiv preprint arXiv:2504.12501*, 2025.
* Mnih et al. (2013)

  Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller.
  Playing atari with deep reinforcement learning.
  *arXiv preprint arXiv:1312.5602*, 2013.
* Mnih et al. (2015)

  Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis.
  Human-level control through deep reinforcement learning.
  518(7540):529–533, 2015.
  ISSN 1476-4687.
  doi: 10.1038/nature14236.
* Ouyang et al. (2022)

  Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al.
  Training language models to follow instructions with human feedback.
  *Advances in neural information processing systems*, 35:27730–27744, 2022.
* Paquette et al. (2024)

  Elliot Paquette, Courtney Paquette, Lechao Xiao, and Jeffrey Pennington.
  4+ 3 phases of compute-optimal neural scaling laws.
  *Advances in Neural Information Processing Systems*, 37:16459–16537, 2024.
* Shannon (1948)

  Claude E Shannon.
  A mathematical theory of communication.
  *The Bell system technical journal*, 27(3):379–423, 1948.
* Silver et al. (2017)

  David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, et al.
  Mastering chess and shogi by self-play with a general reinforcement learning algorithm.
  *arXiv preprint arXiv:1712.01815*, 2017.
* Sorscher et al. (2022)

  Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari S. Morcos.
  Beyond neural scaling laws: beating power law scaling via data pruning.
  In *Proceedings of the 36th International Conference on Neural Information Processing Systems*, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc.
  ISBN 9781713871088.
* Sutton & Barto (2018)

  Richard S. Sutton and Andrew G. Barto.
  *Reinforcement Learning: An Introduction*.
  MIT Press, 2nd edition, 2018.
* Tieleman (2012)

  T. Tieleman.
  Lecture 6.5‐rmsprop: Divide the gradient by a running average of its recent magnitude, 2012.
  URL <https://cir.nii.ac.jp/crid/1370017282431050757>.
* Todorov (2006)

  Emanuel Todorov.
  Optimal control theory.
  In *Bayesian Brain: Probabilistic Approaches to Neural Coding*. The MIT Press, 2006.
  ISBN 978-0-262-29418-8.
  doi: 10.7551/mitpress/1535.003.0018.
* Watkins & Dayan (1992)

  Christopher J. C. H. Watkins and Peter Dayan.
  Q-learning.
  *Machine Learning*, 8:279–292, 1992.
  doi: 10.1007/BF00992698.
