---
arxiv: '2604.27063'
authors:
- Aditya A. Ramesh
- Alex Lewandowski
- Jürgen Schmidhuber
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'Learning to Forget: Continual Learning with Adaptive Weight Decay'
url: https://arxiv.org/abs/2604.27063
year: 2026
---

[2604.27063] Learning to Forget: Continual Learning with Adaptive Weight Decay














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



# Learning to Forget: Continual Learning with Adaptive Weight Decay

Aditya A. Ramesh
Affiliation: The Swiss AI Lab, IDSIA USI-SUPSI, Lugano, Switzerland
  
Alex Lewandowski
Affiliation: University of Alberta, Alberta Machine Intelligence Institute, Edmonton, Canada
  
Jürgen Schmidhuber
Affiliation: The Swiss AI Lab, IDSIA USI-SUPSI, Lugano, Switzerland
Affiliation: Center of Excellence for Generative AI, KAUST, Thuwal, Saudi Arabia{aditya, juergen}@idsia.ch, lewandowski@ualberta.ca

###### Abstract

Continual learning agents with finite capacity must balance acquiring new knowledge with retaining the old.
This requires controlled forgetting of knowledge that is no longer needed, freeing up capacity to learn.
Weight decay, viewed as a mechanism for forgetting, can serve this role by gradually discarding information stored in the weights.
However, a fixed scalar weight decay drives this forgetting uniformly over time and uniformly across all parameters, even when some encode stable knowledge while others track rapidly changing targets.
We introduce Forgetting through Adaptive Decay (FADE), which adapts per-parameter weight decay rates online via approximate meta-gradient descent.
We derive FADE for the online linear setting and apply it to the final layer of neural networks.
Our empirical analysis shows that FADE automatically discovers distinct decay rates for different parameters, complements step-size adaptation, and consistently improves over fixed weight decay across online tracking and streaming classification problems.

## 1 Introduction

Judicious forgetting is essential for continual learning.
An agent with finite capacity cannot retain everything that was previously learned.
Therefore, some degree of forgetting in a controlled way (French, [1999](#bib.bib9); Kumar et al., [2025a](#bib.bib24)) is necessary to successfully navigate the stability-plasticity trade-off (Grossberg, [1987](#bib.bib12); Elsayed & Mahmood, [2024](#bib.bib7)).
For short-term memories stored in activations, gating mechanisms in recurrent neural networks provide learned forgetting (Hochreiter & Schmidhuber, [1997](#bib.bib16); Gers et al., [2000](#bib.bib10)).
But there is no corresponding mechanism for long-term knowledge stored in slowly changing weights.

Weight decay (e.g., Hanson & Pratt, [1988](#bib.bib13); Krogh & Hertz, [1991](#bib.bib23)) is widely used as a regularizer in deep learning (Loshchilov & Hutter, [2019](#bib.bib27)), biasing each weight toward zero at every update.
In typical stationary settings, training data is revisited multiple times and weight decay primarily facilitates regularization (MacKay, [1992b](#bib.bib30)).
Here, we focus on the online non-stationary setting, where data arrives one sample at a time, past samples are never revisited, and task boundaries are unknown.
In such settings, weight decay can serve as a forgetting mechanism that controls how much past information each weight retains.

We explore the complementary interpretation of weight decay as a mechanism for forgetting.
One limitation of weight decay as a forgetting mechanism is that the decay rate is a single scalar, fixed across all parameters and constant over time.
While several works have explored adaptive weight decay for mini-batch training (Ishii & Sato, [2017](#bib.bib19); Nakamura & Hong, [2019](#bib.bib33); Xie et al., [2023](#bib.bib48)), these methods focus on stationary settings with fixed datasets.
In continual learning, a fixed decay rate, even with appropriate tuning, is fundamentally mismatched to the problem.
Some parameters may encode stable knowledge that should be retained, while others track rapidly changing targets that require fast forgetting.
An effective forgetting mechanism must adapt selectively, assigning each parameter its own rate.

To this end, we introduce Forgetting through Adaptive DEcay (FADE), which adapts per-parameter weight decay rates via gradient-based meta-learning.
FADE uses meta-gradients (e.g. Hochreiter et al., [2001](#bib.bib17); Xu et al., [2018](#bib.bib49)) that approximate forward-mode differentiation (e.g., Robinson & Fallside, [1987](#bib.bib37); Williams & Zipser, [1989](#bib.bib47)), enabling online updates.
Specifically, FADE derives from the same meta-gradient approximations as IDBD (Sutton, [1992](#bib.bib44); Degris et al., [2024](#bib.bib5)), which adapts per-parameter step sizes (learning rates) online.
FADE instead focuses on how much to forget.

We derive FADE for the online linear setting and apply it to the final layer of neural networks.
On a linear tracking problem, FADE discovers distinct decay rates for relevant and irrelevant features and complements per-parameter step-size adaptation via IDBD.
On a nonlinear teacher-student tracking problem, FADE with SGD achieves roughly half the error of AdamW (Loshchilov & Hutter, [2019](#bib.bib27)).
On streaming label-permuted EMNIST, FADE outperforms weight clipping, the best prior method (Elsayed et al., [2024](#bib.bib8)).
Across all settings, FADE is robust to the initialization of the decay rate, recovering strong performance even from poor initializations.

## 2 Forgetting through Adaptive Decay (FADE)

Algorithm 1  FADE: Forgetting through Adaptive Decay (online linear regression)

0: step size α\alpha, meta-step size θλ\theta\_{\lambda}, initial decay parameter γ0∈ℝd\gamma\_{0}\in\mathbb{R}^{d}

1: Initialize weights w0∈ℝdw\_{0}\in\mathbb{R}^{d}, traces g0←𝟎∈ℝdg\_{0}\leftarrow\mathbf{0}\in\mathbb{R}^{d}

2: Initialize λ0i←exp⁡(γ0i)\lambda\_{0}^{i}\leftarrow\exp({\gamma\_{0}^{i}}) for all ii

3: for t=0,1,2,…t=0,1,2,\ldots do

4:  Receive input xt∈ℝdx\_{t}\in\mathbb{R}^{d} and target yt∗∈ℝy^{\*}\_{t}\in\mathbb{R}

5:  Predict yt←⟨wt,xt⟩y\_{t}\leftarrow\langle w\_{t},x\_{t}\rangle

6:  Compute error δt←yt∗−yt\delta\_{t}\leftarrow y^{\*}\_{t}-y\_{t}

7:  for each parameter i=1,…,di=1,\ldots,d do

8:   # Adapt decay rate

9:   γt+1i←γti+θλ​δt​xti​gti\gamma\_{t+1}^{i}\leftarrow\gamma\_{t}^{i}+\theta\_{\lambda}\,\delta\_{t}\,x\_{t}^{i}\,g\_{t}^{i}

10:   λt+1i←exp⁡(γt+1i)\lambda\_{t+1}^{i}\leftarrow\exp({\gamma\_{t+1}^{i}})

11:   # Update sensitivity trace

12:   gt+1i←gti​[1−λt+1i−α​(xti)2]+−λt+1i​wtig\_{t+1}^{i}\leftarrow g\_{t}^{i}\big[1-\lambda\_{t+1}^{i}-\alpha(x\_{t}^{i})^{2}\big]^{+}-\lambda\_{t+1}^{i}\,w\_{t}^{i}

13:   # Update weight with adaptive decay

14:   wt+1i←(1−λt+1i)​wti+α​δt​xtiw\_{t+1}^{i}\leftarrow(1-\lambda\_{t+1}^{i})\,w\_{t}^{i}+\alpha\,\delta\_{t}\,x\_{t}^{i}

15:  end for

16: end for

FADE replaces a static, global weight decay hyperparameter with dynamic, per-parameter decay adapted online via meta-gradients.
In this section, we derive FADE for the online linear regression setting using the same forward-mode differentiation techniques and approximations as IDBD (Sutton, [1992](#bib.bib44)).
The key idea is to parameterize the decay rate (λi)\lambda\_{i}) for each parameter wiw^{i} as λi=exp⁡(γi)\lambda^{i}=\exp({\gamma^{i}}) and update the meta-parameter γi\gamma^{i}
by gradient descent on the prediction error.
Since the effect of γi\gamma^{i}
on the error is mediated through the weight update, this requires tracking the sensitivity ∂wi/∂γi\partial w^{i}/\partial\gamma^{i} via an auxiliary trace gig^{i}, which is maintained online.

##### FADE for online linear regression.

At every time step t∈{0,1,2,…}t\in\{0,1,2,\dots\}, the learner receives features xt∈ℝdx\_{t}\in\mathbb{R}^{d} and target yt∗∈ℝy\_{t}^{\*}\in\mathbb{R}.
The learner maintains weights wt∈ℝdw\_{t}\in\mathbb{R}^{d} that are used for predicting yt=⟨wt,xt⟩y\_{t}=\langle w\_{t},x\_{t}\rangle.
Let δt=yt∗−yt\delta\_{t}=y^{\*}\_{t}-y\_{t} be the error, and Jt=δt2/2J\_{t}=\delta\_{t}^{2}/2 be the loss at time step tt.

We update weights with weight decay (Hanson & Pratt, [1988](#bib.bib13)) and the delta-rule, equivalent to SGD for linear regression with squared error (Widrow & Hoff, [1960](#bib.bib46)),

|  |  |  |  |
| --- | --- | --- | --- |
|  | wt+1=(1−λt+1)​wt+α​δt​xt,w\_{t+1}=(1-\lambda\_{t+1})\,w\_{t}+\alpha\,\delta\_{t}\,x\_{t}, |  | (1) |

where α\alpha is a scalar step size and λt+1∈ℝd\lambda\_{t+1}\in\mathbb{R}^{d} controls per-parameter decay, with λt+1i\lambda\_{t+1}^{i} controlling the decay for the ii-th component of the weight wtiw\_{t}^{i}.
Note that unrolling the weight update yields an exponentially weighted sum with an effective memory horizon of ∼1/λi\sim 1/\lambda^{i}.
FADE adapts this horizon per-parameter.
Increasing λi\lambda^{i} shortens the horizon (faster forgetting), decreasing it lengthens the horizon (more retention).

The weight decay coefficients λt\lambda\_{t} are parameterized using γt∈ℝd\gamma\_{t}\in\mathbb{R}^{d}. Specifically, λti=exp⁡(γti),i∈{1,2,…​d}\lambda\_{t}^{i}=\exp(\gamma\_{t}^{i}),\ i\in\{1,2,\dots d\}.
The meta-parameters γ\gamma are updated via gradient descent by differentiating the loss with respect to γ\gamma through the weight update (Equation [1](#S2.E1 "In FADE for online linear regression. ‣ 2 Forgetting through Adaptive Decay (FADE) ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")).
This can be understood as an online form of cross-validation where the weights are updated on the current sample, and the resulting performance on the next sample provides gradients for the meta-parameters (Xu et al., [2018](#bib.bib49)).

The meta-gradients are approximated using forward-mode differentiation (Robinson & Fallside, [1987](#bib.bib37); Williams & Zipser, [1989](#bib.bib47)) with the assumption that changing γi\gamma^{i} primarily affects wiw^{i}, with negligible effect on other weights.
Full derivations are provided in Appendix [A](#A1 "Appendix A Derivations ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay").

Concretely, FADE uses the following updates:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | γt+1i\displaystyle\gamma^{i}\_{t+1} | ←γti+θλ​δt​xti​gti,λt+1i←exp⁡(γt+1i),\displaystyle\leftarrow\gamma^{i}\_{t}+\theta\_{\lambda}\,\delta\_{t}\,x^{i}\_{t}\,g^{i}\_{t},\quad\lambda^{i}\_{t+1}\leftarrow\exp({\gamma^{i}\_{t+1}}), |  | (2) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | gt+1i\displaystyle g^{i}\_{t+1} | ←gti​[1−λt+1i−α​(xti)2]+−λt+1i​wti\displaystyle\leftarrow g^{i}\_{t}\,[1-\lambda^{i}\_{t+1}-\alpha(x^{i}\_{t})^{2}]^{+}-\lambda^{i}\_{t+1}\,w\_{t}^{i} |  | (3) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | wt+1i\displaystyle w\_{t+1}^{i} | ←(1−λt+1i)​wti+α​δt​xti.\displaystyle\leftarrow(1-\lambda\_{t+1}^{i})\,w\_{t}^{i}+\alpha\,\delta\_{t}\,x^{i}\_{t}. |  | (4) |

Here gig^{i} is a trace tracking ∂wi/∂γi\partial w^{i}/\partial\gamma^{i}, the sensitivity of the weight to its decay meta-parameter, and θλ\theta\_{\lambda}
is the meta step size.
The [⋅]+[\cdot]^{+} denotes max⁡(⋅,0)\max(\cdot,0), a positive-bounding
operation for stability.
See Algorithm [1](#alg1 "Algorithm 1 ‣ 2 Forgetting through Adaptive Decay (FADE) ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay") for the pseudocode.

FADE can be combined with strategies that use adaptive coordinate-wise step sizes, like Adam (Kingma & Ba, [2014](#bib.bib21)) or IDBD (Sutton, [1992](#bib.bib44)).
In particular, when combined with IDBD, both the decay rate and step size are adapted per-parameter via meta-gradients.
We provide the derivation and pseudocode for FADE with IDBD in Appendix [A.2](#A1.SS2 "A.2 FADE + IDBD ‣ Appendix A Derivations ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay") and Algorithm [4](#alg4 "Algorithm 4 ‣ Appendix D Algorithms ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay"), respectively.

##### Computational cost.

FADE adds two scalar states per parameter (γi\gamma^{i} and gig^{i})
and one scalar hyperparameter (θλ\theta\_{\lambda}), preserving 𝒪​(d)\mathcal{O}(d) cost per step of online gradient descent.

##### FADE with neural networks.

FADE is derived for the online linear setting, where the prediction is a linear function of the weights.
In a neural network, the final layer is a linear function of the hidden representation, so FADE can be applied directly to the final layer while using any standard optimizer (SGD, Adam) for the hidden layers.
This approach of applying a meta-gradient method only to the final layer has also been used successfully by Javed et al. ([2024](#bib.bib20)).
We follow this approach in our nonlinear experiments (Sections [3.2](#S3.SS2 "3.2 Non-linear tracking ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay") and [3.3](#S3.SS3 "3.3 Streaming image classification with label permutation ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")).
The cross-entropy extension, which replaces the squared-error gradient with the softmax gradient, is derived in Appendix [A.3](#A1.SS3 "A.3 FADE with cross-entropy loss ‣ Appendix A Derivations ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay").

## 3 Experiments

We evaluate FADE on three online non-stationary problems of
increasing complexity222Code is available at <https://github.com/Aditya-Ramesh-10/Fade>.
.
First, we consider a linear tracking task where FADE’s derivation applies exactly (Section [3.1](#S3.SS1 "3.1 Linear Tracking ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")).
Next, we investigate a nonlinear teacher-student tracking task where FADE is applied
to the final layer (Section [3.2](#S3.SS2 "3.2 Non-linear tracking ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")).
Finally, we consider a streaming classification benchmark in which we apply FADE with the cross-entropy loss to the final layer (Section [3.3](#S3.SS3 "3.3 Streaming image classification with label permutation ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")).
Across all three experiments, we examine whether adaptive decay improves performance and robustness to hyperparameter choice compared to fixed decay.

### 3.1 Linear Tracking

![Refer to caption](/html/2604.27063/assets/x1.png)


Figure 1: Evolution of FADE’s average decay rates for relevant and irrelevant weight groups on the linear tracking problem with zero noise, starting from λ0≈0.3​(γ0=−1.2)\lambda\_{0}\approx 0.3(\gamma\_{0}=-1.2), with α=0.1\alpha=0.1
and θλ=0.01\theta\_{\lambda}=0.01.

##### Setup.

We consider a non-stationary tracking task that was previously used to motivate step size adaptation with meta-gradients (Degris et al., [2024](#bib.bib5)).
The learner is presented with a d=20d=20 dimensional input at every time step, sampled i.i.d. xt∈ℝd∼𝒩​(0,Id)x\_{t}\in\mathbb{R}^{d}\sim\mathcal{N}(0,I\_{d}), and target yt∗=𝐰t∗⋅xt+ϵy\_{t}^{\*}=\mathbf{w}^{\*}\_{t}\cdot x\_{t}+\epsilon, where ϵ∼𝒩​(0,σn2)\epsilon\sim\mathcal{N}(0,\sigma\_{n}^{2}).
We consider two values of σn\sigma\_{n}.
Of the 20 weights of 𝐰t∗\mathbf{w}^{\*}\_{t}, 5 are relevant (either ±1\pm 1) and
15 are irrelevant (fixed at 0).
Every 20 steps, one randomly chosen relevant weight flips sign.

##### Evaluation.

We compare stochastic gradient descent (SGD, equivalent to the delta rule), IDBD
(Sutton, [1992](#bib.bib44)) (adaptive step sizes), and FADE
(adaptive decay with SGD), along with fixed weight decay (WD) variants.
In the variant with adaptive step sizes and weight decay (FADE + IDBD), we tie the meta-step sizes θλ=θα=θ\theta\_{\lambda}=\theta\_{\alpha}=\theta.
We report MSE over all 200K steps of interaction for the best
hyperparameters per method, averaged across 10 seeds.
The hyperparameter search configuration is provided in Appendix [B.1](#A2.SS1 "B.1 Linear tracking ‣ Appendix B Implementation details ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay").

##### Results.

Results are presented in Table [1](#S3.T1 "Table 1 ‣ Coupled adaptation of step size and weight decay. ‣ 3.1 Linear Tracking ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay").
SGD uses a single step size for all weights and incurs the largest error.
Adding fixed-weight decay improves SGD by biasing unused weights toward zero.
IDBD adapts step sizes per parameter, gradually driving those for irrelevant inputs toward zero, and substantially outperforms SGD.
Fixed weight decay also benefits IDBD, reducing lifetime MSE from 1.4861.486 to 1.3011.301 in the noiseless setting.
FADE, which adapts decay rates rather than step sizes, improves considerably over SGD + WD (1.6531.653 vs. 2.7262.726).
FADE discovers distinct decay rates.
Weights associated with irrelevant features converge to λ≈0.9\lambda\approx 0.9, decaying toward zero.
Weights for relevant features settle near λ≈0.02\lambda\approx 0.02, maintaining a longer memory (Figure [1](#S3.F1 "Figure 1 ‣ 3.1 Linear Tracking ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")).

As this problem was designed to highlight step-size adaptation, IDBD’s advantage over FADE here is not particularly surprising.
The key challenge is distinguishing relevant from irrelevant features, which IDBD handles by learning to stop updating irrelevant weights.
Crucially, FADE + IDBD achieves the lowest error overall (1.2461.246), outperforming both FADE and IDBD alone.
This indicates that adaptive step sizes and adaptive decay are complementary mechanisms.
IDBD controls how quickly each parameter incorporates new information, while FADE controls how much past information each parameter retains.
Even on a problem tailored to step-size adaptation, dynamically adjusting the forgetting rate via FADE provides an additional benefit.

##### Coupled adaptation of step size and weight decay.

We also compare against a ‘coupled’ variant of step size and weight decay adaptation where the decay term comes from the L2L\_{2} regularized loss, i.e., the decay term is αi​λi\alpha^{i}\lambda^{i} rather than λi\lambda^{i} alone, and the weight update is wt+1i=(1−αt+1i​λt+1i)​wti+αt+1i​δt​xtiw^{i}\_{t+1}=(1-\alpha^{i}\_{t+1}\lambda^{i}\_{t+1})\,w^{i}\_{t}+\alpha^{i}\_{t+1}\,\delta\_{t}\,x^{i}\_{t} (see Appendix [A.5](#A1.SS5 "A.5 Coupled decay and step size adaptation for online linear regression ‣ Appendix A Derivations ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay") for the derivation).
The best coupled variant achieves an MSE of 1.270±0.0061.270\pm 0.006 with σn=0\sigma\_{n}=0 and 2.670±0.0092.670\pm 0.009 with σn=1\sigma\_{n}=1, slightly worse than (decoupled) FADE+IDBD (1.246±0.0061.246\pm 0.006 and 2.646±0.0092.646\pm 0.009).
We observe that the coupled update introduces a shared −αi​λi​wti-\alpha^{i}\lambda^{i}w^{i}\_{t} term into both traces, potentially causing some interference in adaptation.

Table 1: Average MSE ±\pm standard deviation across 10 seeds on the linear tracking task across two noise levels σn\sigma\_{n}.

|  | σn=0\sigma\_{n}=0 | σn=1.0\sigma\_{n}=1.0 |
| --- | --- | --- |
| SGD | 3.628±0.0213.628\pm 0.021 | 5.119±0.0265.119\pm 0.026 |
| SGD + WD | 2.726±0.0082.726\pm 0.008 | 4.087±0.0124.087\pm 0.012 |
| IDBD | 1.486±0.0071.486\pm 0.007 | 2.937±0.0112.937\pm 0.011 |
| IDBD + WD | 1.301±0.0061.301\pm 0.006 | 2.718±0.0102.718\pm 0.010 |
| FADE | 1.653±0.0091.653\pm 0.009 | 3.044±0.0113.044\pm 0.011 |
| FADE + IDBD | 1.246±0.006\mathbf{1.246\pm 0.006} | 2.646±0.009\mathbf{2.646\pm 0.009} |

### 3.2 Non-linear tracking

##### Setup.

Vector targets are obtained from a teacher network.
Concretely, the teacher is a neural network with input dimension d=32d=32 and a hidden layer of size h=256h=256 with ReLU activations.
The output layer (head) is linear and outputs a vector y∗∈ℝ20y^{\*}\in\mathbb{R}^{20}.
The learner is a student network that has the same architecture as the teacher.
At each step, the learner receives a single input xt∼𝒩​(0,Id)x\_{t}\sim\mathcal{N}(0,I\_{d}) and the teacher’s output yt∗y^{\*}\_{t} as the target.
Learning proceeds online from a single sample per step, with no replay buffer or resets.

To introduce non-stationarity, we periodically change weights in the teacher’s final layer.
Of the 2020 output units, 66 are stable, and incoming weights remain fixed throughout the interaction.
A further 7 are fast-changing: every
P=500P=500 steps, each weight into these output units is independently multiplied by a random sign (±1\pm 1 with equal probability).
The remaining 7 output units are slow-changing, undergoing the same random sign perturbation every
15​P=750015P=7500 steps.
This creates three distinct rates of non-stationarity across the output units, while the teacher’s input-to-hidden weights remain fixed throughout.

Table 2: Average MSE ±\pm standard deviation over the final 500K steps across 5 seeds on the non-linear tracking problem.

| Method | MSE (final 500K steps) |
| --- | --- |
| SGD | 0.0168±0.00020.0168\pm 0.0002 |
| SGD + WD | 0.0167±0.00020.0167\pm 0.0002 |
| Adam | 0.0170±0.00020.0170\pm 0.0002 |
| AdamW | 0.0138±0.00010.0138\pm 0.0001 |
| FADE + SGD | 0.0073±0.0001\mathbf{0.0073\pm 0.0001} |
| FADE + Adam | 0.0087±0.00010.0087\pm 0.0001 |

##### Evaluation.

The learner interacts for 2M steps.
We report average MSE over the final 500K steps, averaged across 5 seeds.
Since FADE is derived for the linear setting, we apply it only to the final layer of the student network.
In FADE+SGD, SGD is used throughout the network. In FADE+Adam, Adam is used throughout, with FADE providing adaptive decay on the head.
We compare SGD, SGD with weight decay (SGD+WD), Adam (no weight decay), AdamW, FADE+SGD, and FADE+Adam.
For each approach, we perform a grid search over the step-size and weight decay hyperparameters.
Hyperparameter search details and selected hyperparameters are provided in Appendix [B.2](#A2.SS2 "B.2 Non-linear tracking ‣ Appendix B Implementation details ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay").

##### Results.

![Refer to caption](/html/2604.27063/assets/x2.png)


(a) Fast

![Refer to caption](/html/2604.27063/assets/x3.png)


(b) Stable

![Refer to caption](/html/2604.27063/assets/x4.png)


(c) Slow

![Refer to caption](/html/2604.27063/assets/x5.png)


(d) Slow, zoomed in

Figure 2: MSE by group on nonlinear tracking problem. With γ0=−9.2​(λ0≈0.0001)\gamma\_{0}=-9.2(\lambda\_{0}\approx 0.0001), comparing FADE+SGD (θλ=2\theta\_{\lambda}=2) with its fixed-decay counterpart FADE+SGD (θλ=0\theta\_{\lambda}=0).

![Refer to caption](/html/2604.27063/assets/x6.png)


Figure 3: Evolution of FADE’s average per-group decay rates on the nonlinear tracking problem starting from a shared initialization λ0≈10−4\lambda\_{0}\approx 10^{-4} (dashed line), and θλ=2.0\theta\_{\lambda}=2.0.

Main results are presented in Table [2](#S3.T2 "Table 2 ‣ Setup. ‣ 3.2 Non-linear tracking ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay").
All baseline methods (SGD, SGD+WD, Adam, AdamW) achieve similar MSE over the final 500K steps, ranging from around 0.0140.014 to 0.0170.017.
AdamW is the best performing baseline, achieving an average of 0.01380.0138.
Using FADE helps both SGD and Adam.
FADE+SGD achieves 0.00730.0073, roughly half the error of AdamW, while FADE+Adam achieves 0.00870.0087.
We observe that FADE+SGD outperforms FADE+Adam, likely because FADE’s meta-gradient derivation holds exactly for SGD, but only approximately for Adam (Appendix [A.4](#A1.SS4 "A.4 FADE with Adam ‣ Appendix A Derivations ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")).

To isolate FADE’s contribution, we compare FADE+SGD with adaptation (θλ=2\theta\_{\lambda}=2) against its non-adaptive counterpart (θλ=0\theta\_{\lambda}=0).
With θλ=0\theta\_{\lambda}=0, FADE reduces to SGD with fixed weight decay on the head and no decay on the hidden layer.
Figure [2](#S3.F2 "Figure 2 ‣ Results. ‣ 3.2 Non-linear tracking ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay") shows per-group MSE for γ0=−9.2\gamma\_{0}=-9.2, comparing FADE (θλ=2)(\theta\_{\lambda}=2) with fixed decay (θλ=0)(\theta\_{\lambda}=0) .
Across all three groups (fast, slow, and stable outputs), we see improvements from FADE’s adaptation, suggesting that the adaptation is effective for a wide range of non-stationarity rates.
Figure [3](#S3.F3 "Figure 3 ‣ Results. ‣ 3.2 Non-linear tracking ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay") shows the corresponding evolution of weight decay across the groups starting from a shared initialization.
Here, FADE increases decay for fast-changing outputs and decreases it for stable outputs, with slow outputs receiving an intermediate rate.

We examine sensitivity to the initial decay rate λ0\lambda\_{0} and meta-step-size θλ\theta\_{\lambda} in Table [3](#S3.T3 "Table 3 ‣ Results. ‣ 3.2 Non-linear tracking ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay").
Our results show that adaptive weight decay via FADE consistently improves performance.
With fixed decay (θλ=0\theta\_{\lambda}=0), performance varies substantially across the decay rate, with the best initialization (γ0=−6.9)(\gamma\_{0}=-6.9) achieving 0.00990.0099 and the worst (γ0=−2.3)(\gamma\_{0}=-2.3) achieving 0.03600.0360.
With FADE (θλ>0)(\theta\_{\lambda}>0), the sensitivity-gap narrows considerably.
Even from the worst initialization (γ0=−2.3)(\gamma\_{0}=-2.3), FADE recovers to 0.01250.0125. The FADE+Adam sensitivity (Table [4](#S3.T4 "Table 4 ‣ Results. ‣ 3.2 Non-linear tracking ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")) shows a similar pattern, confirming that the benefits are not specific to SGD.

Additionally, we note that (fixed) weight decay on the head, but not on the hidden layer, can perform reasonably well with a well-tuned decay (see γ0=−6.9\gamma\_{0}=-6.9 with θλ=0\theta\_{\lambda}=0 in Table [3](#S3.T3 "Table 3 ‣ Results. ‣ 3.2 Non-linear tracking ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")).
However, adaptive weight decay via FADE provides benefits in terms of improved MSE and robustness to the choice of the initial decay.

Table 3: FADE + SGD sensitivity to initial decay rate γ0\gamma\_{0} and meta-step-size θλ\theta\_{\lambda} on the non-linear tracking problem. θλ=0\theta\_{\lambda}=0 corresponds to fixed weight decay on the head.
Results are average MSE ±\pm standard deviation over the final 500K steps across 5 seeds.

|  | θλ=0\theta\_{\lambda}=0 | θλ=0.5\theta\_{\lambda}=0.5 | θλ=1.0\theta\_{\lambda}=1.0 | θλ=2.0\theta\_{\lambda}=2.0 |
| --- | --- | --- | --- | --- |
| γ0=−2.3​(λ0≈0.1)\gamma\_{0}=-2.3\ (\lambda\_{0}\approx 0.1) | 0.0360±0.00110.0360\pm 0.0011 | 0.0148±0.00010.0148\pm 0.0001 | 0.0137±0.00010.0137\pm 0.0001 | 0.0125±0.00010.0125\pm 0.0001 |
| γ0=−4.6​(λ0≈0.01)\gamma\_{0}=-4.6\ (\lambda\_{0}\approx 0.01) | 0.0182±0.00030.0182\pm 0.0003 | 0.0098±0.00010.0098\pm 0.0001 | 0.0093±0.00010.0093\pm 0.0001 | 0.0089±0.00010.0089\pm 0.0001 |
| γ0=−6.9​(λ0≈0.001)\gamma\_{0}=-6.9\ (\lambda\_{0}\approx 0.001) | 0.0099±0.00010.0099\pm 0.0001 | 0.0084±0.00010.0084\pm 0.0001 | 0.0082±0.00010.0082\pm 0.0001 | 0.0080±0.00010.0080\pm 0.0001 |
| γ0=−9.2​(λ0≈0.0001)\gamma\_{0}=-9.2\ (\lambda\_{0}\approx 0.0001) | 0.0104±0.00020.0104\pm 0.0002 | 0.0083±0.00010.0083\pm 0.0001 | 0.0075±0.00010.0075\pm 0.0001 | 0.0073±0.0001\mathbf{0.0073\pm 0.0001} |




Table 4: FADE + Adam sensitivity to initial decay rate γ0\gamma\_{0} and meta-step-size θλ\theta\_{\lambda} on the non-linear tracking problem. θλ=0\theta\_{\lambda}=0 corresponds to fixed weight decay on the head.
Results are average MSE ±\pm standard deviation over the final 500K steps across 5 seeds.

|  | θλ=0\theta\_{\lambda}=0 | θλ=0.5\theta\_{\lambda}=0.5 | θλ=1.0\theta\_{\lambda}=1.0 | θλ=2.0\theta\_{\lambda}=2.0 |
| --- | --- | --- | --- | --- |
| γ0=−2.3​(λ0≈0.1)\gamma\_{0}=-2.3\ (\lambda\_{0}\approx 0.1) | 0.0315±0.00080.0315\pm 0.0008 | 0.0126±0.00010.0126\pm 0.0001 | 0.0121±0.00010.0121\pm 0.0001 | 0.0118±0.00010.0118\pm 0.0001 |
| γ0=−4.6​(λ0≈0.01)\gamma\_{0}=-4.6\ (\lambda\_{0}\approx 0.01) | 0.0179±0.00020.0179\pm 0.0002 | 0.0107±0.00010.0107\pm 0.0001 | 0.0103±0.00010.0103\pm 0.0001 | 0.0102±0.00010.0102\pm 0.0001 |
| γ0=−6.9​(λ0≈0.001)\gamma\_{0}=-6.9\ (\lambda\_{0}\approx 0.001) | 0.0109±0.00010.0109\pm 0.0001 | 0.0097±0.00010.0097\pm 0.0001 | 0.0097±0.00010.0097\pm 0.0001 | 0.0097±0.00010.0097\pm 0.0001 |
| γ0=−9.2​(λ0≈0.0001)\gamma\_{0}=-9.2\ (\lambda\_{0}\approx 0.0001) | 0.0128±0.00020.0128\pm 0.0002 | 0.0094±0.00010.0094\pm 0.0001 | 0.0087±0.0001\mathbf{0.0087\pm 0.0001} | 0.0088±0.00010.0088\pm 0.0001 |

### 3.3 Streaming image classification with label permutation

##### Setup

Following the setup of Elsayed et al. ([2024](#bib.bib8)), we evaluate on a streaming label-permuted classification problem using the Extended MNIST dataset (EMNIST; Cohen et al., [2017](#bib.bib4)), consisting of 47 classes.
At each step, the learner receives a single image, makes a prediction, observes the label, and then performs one gradient update.
The prediction is evaluated before the update to measure online performance.
Every 2500 steps, a fresh random permutation is applied to all class labels, changing the target function.
Since the images remain unchanged across tasks, useful features can, in principle, be retained, but the mapping from features to labels must be relearned after each permutation.
The network is a two-hidden-layer MLP (300, 150) with LeakyReLU activations.

##### Evaluation

We train for 5M steps (2000 label permutation tasks), which is 55 times the interaction length considered by Elsayed & Mahmood ([2024](#bib.bib7)).
We report average online accuracy across the full run.
All methods use their best hyperparameters selected by grid search (see Appendix [B.3](#A2.SS3 "B.3 Streaming image classification with label permutation ‣ Appendix B Implementation details ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")).

We compare SGD (with weight decay), Adam, AdamW, and FADE.
Additionally, we compare against SGD with weight clipping (SGD + WClip).
Weight clipping (with SGD) was the best-performing approach on this problem in the experiment conducted by Elsayed et al. ([2024](#bib.bib8)), outperforming other strategies such as L2 init (Kumar et al., [2025b](#bib.bib25)), Shrink&Perturb (Ash & Adams, [2020](#bib.bib2)), etc.
For FADE, we again apply it to the final layer that produces the classification logits.
See Appendix [A.3](#A1.SS3 "A.3 FADE with cross-entropy loss ‣ Appendix A Derivations ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay") for FADE with the cross-entropy loss.

![Refer to caption](/html/2604.27063/assets/x7.png)


Figure 4: Online accuracy on label-permuted EMNIST.
FADE+SGD and FADE+Adam apply FADE to the final layer.

![Refer to caption](/html/2604.27063/assets/x8.png)


Figure 5: Online accuracy on label-permuted EMNIST with FADE + Adam, starting from γ0=−11.5\gamma\_{0}=-11.5 (λ0≈10−5\lambda\_{0}\approx 10^{-5}).
Despite an initial drop, FADE with θλ>0\theta\_{\lambda}>0 recovers strongly, with a final accuracy of over 0.70.7.

##### Results

Results are presented in Table [5](#S3.T5 "Table 5 ‣ Results ‣ 3.3 Streaming image classification with label permutation ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay") and online accuracy across tasks is shown in Figure [5](#S3.F5 "Figure 5 ‣ Evaluation ‣ 3.3 Streaming image classification with label permutation ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay").
Without weight decay, SGD and Adam perform poorly on this problem (with average accuracy of 0.2580.258 and 0.1190.119, respectively).
Adding weight decay to these approaches improves performance.
Weight clipping substantially improves performance, with SGD + WClip achieving an average online accuracy of 0.6120.612.
FADE outperforms the considered baselines, with FADE+SGD achieving an average accuracy of 0.8070.807 and FADE+Adam achieving 0.7500.750.

Table 5: Average online accuracy across 5M interactions (2000 tasks) in the streaming classification problem with label permutation using EMNIST.

| Method | Avg Online Accuracy |
| --- | --- |
| SGD | 0.258±0.0060.258\pm 0.006 |
| SGD + WD | 0.335±0.0020.335\pm 0.002 |
| Adam | 0.119±0.0030.119\pm 0.003 |
| AdamW | 0.372±0.0010.372\pm 0.001 |
| SGD + WClip | 0.612±0.0030.612\pm 0.003 |
| FADE + SGD | 0.807±0.001\mathbf{0.807\pm 0.001} |
| FADE + Adam | 0.750±0.0010.750\pm 0.001 |




Table 6: Impact of FADE’s adaptive decay on label-permuted EMNIST with SGD.
Results are average online accuracy ±\pm standard deviation across 5 seeds, with step size α=0.005\alpha=0.005.
θλ=0\theta\_{\lambda}=0 corresponds to fixed weight decay on the head.

|  | θλ=0\theta\_{\lambda}=0 | θλ=0.1\theta\_{\lambda}=0.1 | θλ=0.5\theta\_{\lambda}=0.5 | θλ=1.0\theta\_{\lambda}=1.0 |
| --- | --- | --- | --- | --- |
| γ0=−4.6​(λ0≈0.01)\gamma\_{0}=-4.6\ (\lambda\_{0}\approx 0.01) | 0.694±0.0010.694\pm 0.001 | 0.780±0.0030.780\pm 0.003 | 0.798±0.0010.798\pm 0.001 | 0.800±0.0010.800\pm 0.001 |
| γ0=−6.9​(λ0≈0.001)\gamma\_{0}=-6.9\ (\lambda\_{0}\approx 0.001) | 0.801±0.0010.801\pm 0.001 | 0.807±0.001\mathbf{0.807\pm 0.001} | 0.806±0.0010.806\pm 0.001 | 0.804±0.0010.804\pm 0.001 |
| γ0=−9.2​(λ0≈0.0001)\gamma\_{0}=-9.2\ (\lambda\_{0}\approx 0.0001) | 0.779±0.0010.779\pm 0.001 | 0.801±0.0010.801\pm 0.001 | 0.801±0.0000.801\pm 0.000 | 0.798±0.0010.798\pm 0.001 |
| γ0=−11.5​(λ0≈0.00001)\gamma\_{0}=-11.5\ (\lambda\_{0}\approx 0.00001) | 0.531±0.0050.531\pm 0.005 | 0.682±0.0030.682\pm 0.003 | 0.743±0.0020.743\pm 0.002 | 0.750±0.0020.750\pm 0.002 |




Table 7: Impact of FADE’s adaptive decay on label-permuted EMNIST with Adam.
Results are average online accuracy ±\pm standard deviation across 5 seeds, with step size α=0.0001\alpha=0.0001.
θλ=0\theta\_{\lambda}=0 corresponds to fixed weight decay on the head.

|  | θλ=0\theta\_{\lambda}=0 | θλ=0.1\theta\_{\lambda}=0.1 | θλ=0.5\theta\_{\lambda}=0.5 | θλ=1.0\theta\_{\lambda}=1.0 |
| --- | --- | --- | --- | --- |
| γ0=−4.6​(λ0≈0.01)\gamma\_{0}=-4.6\ (\lambda\_{0}\approx 0.01) | 0.556±0.0010.556\pm 0.001 | 0.737±0.0000.737\pm 0.000 | 0.737±0.0000.737\pm 0.000 | 0.719±0.0010.719\pm 0.001 |
| γ0=−6.9​(λ0≈0.001)\gamma\_{0}=-6.9\ (\lambda\_{0}\approx 0.001) | 0.746±0.0010.746\pm 0.001 | 0.750±0.001\mathbf{0.750\pm 0.001} | 0.741±0.0010.741\pm 0.001 | 0.721±0.0010.721\pm 0.001 |
| γ0=−9.2​(λ0≈0.0001)\gamma\_{0}=-9.2\ (\lambda\_{0}\approx 0.0001) | 0.588±0.0010.588\pm 0.001 | 0.741±0.0010.741\pm 0.001 | 0.716±0.0020.716\pm 0.002 | 0.649±0.0070.649\pm 0.007 |
| γ0=−11.5​(λ0≈0.00001)\gamma\_{0}=-11.5\ (\lambda\_{0}\approx 0.00001) | 0.269±0.0040.269\pm 0.004 | 0.412±0.0040.412\pm 0.004 | 0.472±0.0050.472\pm 0.005 | 0.521±0.0040.521\pm 0.004 |

We examine sensitivity to the initial decay rate λ0\lambda\_{0} and meta-step-size θλ\theta\_{\lambda} in Tables [6](#S3.T6 "Table 6 ‣ Results ‣ 3.3 Streaming image classification with label permutation ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay") and [7](#S3.T7 "Table 7 ‣ Results ‣ 3.3 Streaming image classification with label permutation ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay").
Notably, SGD (or Adam) with fixed weight decay applied only to the head can itself be a strong approach with the right decay coefficient (e.g., 0.8010.801 at γ0=−6.9\gamma\_{0}=-6.9).
To our knowledge, such a baseline has not been explored in prior continual learning benchmarks.
In label permutation, learned features remain useful across tasks, but the mapping from features to classes must be relearned after each permutation.
Decay on the head could facilitate this relearning by clearing stale class mappings.
This result highlights the importance of applying different decay rates to different parts of the network.
However, this baseline is fragile.
With fixed decay on the head (θλ=0)(\theta\_{\lambda}=0), performance depends heavily on the choice of initial decay rate λ0\lambda\_{0}, ranging from 0.5310.531 to 0.8010.801 with SGD.
FADE reduces this sensitivity and provides substantial benefits when initialization is sub-optimal.
Figure [5](#S3.F5 "Figure 5 ‣ Evaluation ‣ 3.3 Streaming image classification with label permutation ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay") illustrates the benefit of adaptivity from a poor initialization with FADE + Adam.
With γ0=−11.5​(λ0≈10−5)\gamma\_{0}=-11.5(\lambda\_{0}\approx 10^{-5}), the initial decay is small and fixed decay performs poorly. FADE discovers appropriate decay rates during interaction, recovering to a high accuracy of over 0.70.7.

##### FADE on all layers.

We evaluated a naive extension of FADE + SGD to all layers on the EMNIST benchmark, applying the same meta-gradient updates derived for the linear setting to hidden layers.
The best configuration achieves
0.5350.535 average online accuracy, outperforming SGD with fixed weight decay (0.3350.335; Table [5](#S3.T5 "Table 5 ‣ Results ‣ 3.3 Streaming image classification with label permutation ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")) but well below head-only FADE (0.8070.807; Table [5](#S3.T5 "Table 5 ‣ Results ‣ 3.3 Streaming image classification with label permutation ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")).
Furthermore, unlike head-only FADE, this variant is sensitive to initialization, indicating that the linear meta-gradient approximation is insufficient for hidden layers.

##### Analysis with partial label permutations.

We also evaluate a variant in which 24 of the 47 classes retain their labels across permutations, resulting in a mix of stable and changing outputs.
This setting may better reflect realistic continual learning problems where some structure persists across tasks.
The main results are presented in Table [8](#S3.T8 "Table 8 ‣ Analysis with partial label permutations. ‣ 3.3 Streaming image classification with label permutation ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay").
FADE+SGD achieves an average online accuracy of 0.8410.841 with the same hyperparameters as the variant where all classes were permuted, considerably better than the other baselines.
Further details and the sensitivity tables for FADE are provided in Appendix [C](#A3 "Appendix C Streaming classification with partial label permuted EMNIST ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay").
FADE’s benefits over its fixed counterpart are more pronounced in this setting.
The best fixed FADE+SGD (with θλ=0\theta\_{\lambda}=0) achieves 0.8300.830 vs 0.8460.846 for the best adaptive variant (see Table [9](#A3.T9 "Table 9 ‣ Appendix C Streaming classification with partial label permuted EMNIST ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")).
Similarly, with FADE+Adam, the best θλ=0\theta\_{\lambda}=0 setting achieves 0.7630.763 vs 0.8100.810 for the best adaptive (see Table [10](#A3.T10 "Table 10 ‣ Appendix C Streaming classification with partial label permuted EMNIST ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")).

Table 8: Average online accuracy across 5M interactions (2000 tasks) in the streaming classification problem with *partial label permutation* using EMNIST.

| Method | Avg Online Accuracy |
| --- | --- |
| SGD | 0.735±0.0120.735\pm 0.012 |
| SGD + WD | 0.694±0.0070.694\pm 0.007 |
| Adam | 0.448±0.0160.448\pm 0.016 |
| AdamW | 0.541±0.0040.541\pm 0.004 |
| SGD + WClip | 0.719±0.0060.719\pm 0.006 |
| FADE + Adam | 0.808±0.0020.808\pm 0.002 |
| FADE + SGD | 0.841±0.001\mathbf{0.841\pm 0.001} |

## 4 Related Work

Our approach is motivated by the perspective of learned forget gates applied to short-term memory in sequence-processing models (Gers et al., [2000](#bib.bib10); Beck et al., [2024](#bib.bib3); Yang et al., [2024](#bib.bib50)).
The forget gate was introduced to prevent unbounded growth of memory cell states, enabling recurrent neural networks to discard outdated information stored in activations (Gers et al., [2000](#bib.bib10); Van der Westhuizen & Lasenby, [2018](#bib.bib45)).
Earlier, Mozer ([1989](#bib.bib32)) proposed an architecture that learns per-unit decay rates on the short-term memory via gradient descent with an online trace.
FADE can be seen as applying these principles to long-term memories stored in weights, adapting per-parameter decay rates rather than per-activation gates.
Unlike activation-level forget gates, which are typically input-dependent, FADE’s decay rates adapt slowly via meta-gradients, matching the timescale of the environment’s non-stationarity rather than input-level variation.
In summary, FADE involves adapting weight decay via meta-learning for continual learning.

##### Adaptive weight decay.

Hanson & Pratt ([1988](#bib.bib13)) study weight decay as a bias toward simpler networks, noting that uniform decay rates are suboptimal because all weights decay equally regardless of their importance.
They compare two types of non-uniform weight decay at the hidden-unit level, with the general idea of applying stronger decay on small weights while leaving large weights intact.
In a Bayesian framework, MacKay ([1992a](#bib.bib29)) proposes selecting weight decay coefficients via evidence maximization.
Several works have since explored adaptive weight decay for better regularization in problems with fixed, stationary datasets.
To balance the scale of loss gradients and the decay penalty, prior works have proposed adapting weight decay coefficients across layers (Ishii & Sato, [2017](#bib.bib19)) or across training iterations to improve adversarial robustness (Ghiasi et al., [2023](#bib.bib11)).
Taking adaptation to a more granular level, Nakamura & Hong ([2019](#bib.bib33)) propose per-parameter weight decay that scales with layer-wise normalized gradient magnitudes, effectively assigning stronger regularization to parameters with relatively larger gradients.
Xie et al. ([2023](#bib.bib48)) propose a weight decay scheduler based on the gradient norm to mitigate the large gradient norms that hinder convergence and generalization when training deep neural networks.
Most recently, scaling adaptive weight decay to modern heterogeneous architectures, He et al. ([2025](#bib.bib14)) assign distinct weight decay strengths to different modules within large language models.
These methods are designed to improve generalization in mini-batch training with stationary targets.
FADE instead targets online continual learning, where non-stationarity rather than overfitting is the core challenge, and derives its adaptation from meta-gradient descent rather than gradient-norm heuristics.

##### Continual learning.

Agents with bounded capacity that are designed to learn forever must trade-off capacity for stability, or for plasticity, by either retaining what was learned against learning new things (Grossberg, [1987](#bib.bib12); Elsayed & Mahmood, [2024](#bib.bib7)).
With neural networks, continual learning can struggle on either component of this trade-off, leading to two distinct phenomena in the literature.
The first of these is
catastrophic forgetting, where neural networks fail to retain what was previously learned when data changes over time (McCloskey & Cohen, [1989](#bib.bib31); Ratcliff, [1990](#bib.bib36); French, [1999](#bib.bib9)).
Mitigation strategies for catastrophic forgetting involve regularizing parameters towards past solutions (Kirkpatrick et al., [2017](#bib.bib22)).
It has also been recognized that storing historical examples and retraining provides an effective, but computationally demanding, baseline (Prabhu et al., [2020](#bib.bib35)).
More recently, loss of plasticity has been identified as another potential failure mode, where neural networks fail to adapt and learn new things (Ash & Adams, [2020](#bib.bib2); Dohare et al., [2024](#bib.bib6)).
This latter phenomenon is often mitigated by either regularization approaches (Lewandowski et al., [2025](#bib.bib26); Kumar et al., [2025b](#bib.bib25)), or by re-initializing weights (Nikishin et al., [2022](#bib.bib34); Hernandez-Garcia et al., [2025](#bib.bib15)).

##### Meta-learning for continual learning.

The difficulties inherent to continual learning can broadly be addressed by meta-learning the learning algorithm itself, either through gradient-based meta-learning or program search (Schmidhuber, [1987](#bib.bib38); [1992](#bib.bib39); [1993](#bib.bib40); Schmidhuber et al., [1997](#bib.bib41); Hochreiter et al., [2001](#bib.bib17); Andrychowicz et al., [2016](#bib.bib1); Irie et al., [2025](#bib.bib18)).
Unlike these more general approaches, FADE does not replace the entire learning algorithm.
It only adapts per-parameter decay rates, leaving the base optimizer (like SGD or Adam) intact.
Closely related are meta-gradient methods that adjust meta-parameters online via differentiation through the update rule (Xu et al., [2018](#bib.bib49); Zahavy et al., [2020](#bib.bib51); Luketina et al., [2022](#bib.bib28)).
IDBD (Sutton, [1992](#bib.bib44)) uses meta-gradients to adapt per-parameter step sizes for online linear regression, and Schraudolph ([1999](#bib.bib42)) extends it to nonlinear networks. Sharifnassab et al. ([2025](#bib.bib43)) generalize IDBD within a framework that optimizes meta-parameters against a discounted lifetime objective.
FADE uses the same forward-mode meta-gradient derivation as IDBD, but focuses specifically on adapting weight decay rates, targeting how much to forget rather than how fast to learn.

## 5 Conclusion

We introduced FADE to adapt per-parameter weight decay rates online via forward-mode meta-gradients.
Our work was motivated by viewing weight decay as a mechanism for forgetting, rather than purely as a regularizer that controls weight magnitudes.
To enable judicious forgetting in continual learning, FADE automatically discovers distinct decay rates during interaction, adding minimal computational overhead.

Empirically, FADE consistently improves over fixed weight decay across three settings of increasing complexity.
In a linear tracking problem, FADE complements step-size adaptation via IDBD, yielding the lowest error when combined.
On a nonlinear teacher-student problem, applying FADE to the final layer achieves roughly half the error of AdamW.
On the streaming label-permuted EMNIST problem, FADE outperforms weight clipping, the previous best method.
Across all settings, FADE is robust to the initial decay rate, narrowing the performance gap across initializations considerably compared to fixed decay.

Our results reveal that applying a suitable fixed non-zero decay rate only to the final layer proves to be a strong baseline that has not received prior attention.
By default, weight decay applies a fixed decay rate uniformly across all parameters, making such a baseline easy to overlook.
This finding supports the importance of different decay rates across different parts of the network, a principle that FADE automates at the per-parameter level on the final layer.

We derived FADE for the online linear regression setting and applied it to the final layer when using it with neural networks.
Scaling FADE to larger architectures requires addressing some open challenges.
We evaluated a naive extension of FADE to all layers on the EMNIST benchmark, applying the same meta-gradient updates derived for the linear setting to hidden layers as well.
This variant outperforms SGD with fixed weight decay but plateaus below head-only FADE, suggesting that the linear approximation is insufficient for hidden layers.
Therefore, an important direction for future work is to formalize the interaction between the decay rates of the network head and those of the hidden layers, while accounting for nonlinearities between layers.
Another possibility is to design meta-gradient approximations that are invariant to the underlying architecture, making them easier to apply across different depths, nonlinearities, and layer types.
This would allow FADE to be extended to each layer in a network and combined with common architectural modifications, such as layer normalization, enabling its use in settings where non-stationarity arises naturally, such as reinforcement learning.

## Acknowledgments

We thank Kazuki Irie, Vincent Herrmann, Arsalan Sharifnassab, and Nicolau Oliver for valuable discussions and helpful comments on earlier drafts.

## References

* Andrychowicz et al. (2016)

  Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, and Nando De Freitas.
  Learning to learn by gradient descent by gradient descent.
  *Advances in neural information processing systems*, 29, 2016.
* Ash & Adams (2020)

  Jordan Ash and Ryan P Adams.
  On warm-starting neural network training.
  *Advances in neural information processing systems*, 33:3884–3894, 2020.
* Beck et al. (2024)

  Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, and Sepp Hochreiter.
  xlstm: Extended long short-term memory.
  *Advances in Neural Information Processing Systems*, 37:107547–107603, 2024.
* Cohen et al. (2017)

  Gregory Cohen, Saeed Afshar, Jonathan Tapson, and Andre Van Schaik.
  Emnist: Extending mnist to handwritten letters.
  In *2017 international joint conference on neural networks (IJCNN)*, pp. 2921–2926. IEEE, 2017.
* Degris et al. (2024)

  Thomas Degris, Khurram Javed, Arsalan Sharifnassab, Yuxin Liu, and Richard Sutton.
  Step-size optimization for continual learning.
  *arXiv preprint arXiv:2401.17401*, 2024.
* Dohare et al. (2024)

  Shibhansh Dohare, J Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A Rupam Mahmood, and Richard S Sutton.
  Loss of plasticity in deep continual learning.
  *Nature*, 632(8026):768–774, 2024.
* Elsayed & Mahmood (2024)

  Mohamed Elsayed and A. Rupam Mahmood.
  Addressing loss of plasticity and catastrophic forgetting in continual learning.
  In *The Twelfth International Conference on Learning Representations, ICLR*, 2024.
* Elsayed et al. (2024)

  Mohamed Elsayed, Qingfeng Lan, Clare Lyle, and A Rupam Mahmood.
  Weight clipping for deep continual and reinforcement learning.
  *arXiv preprint arXiv:2407.01704*, 2024.
* French (1999)

  Robert M French.
  Catastrophic forgetting in connectionist networks.
  *Trends in cognitive sciences*, 3(4):128–135, 1999.
* Gers et al. (2000)

  Felix A Gers, Jürgen Schmidhuber, and Fred Cummins.
  Learning to forget: Continual prediction with LSTM.
  *Neural computation*, 12(10):2451–2471, 2000.
* Ghiasi et al. (2023)

  Mohammad Amin Ghiasi, Ali Shafahi, and Reza Ardekani.
  Improving robustness with adaptive weight decay.
  *Advances in Neural Information Processing Systems*, 36:79067–79080, 2023.
* Grossberg (1987)

  Stephen Grossberg.
  Competitive learning: From interactive activation to adaptive resonance.
  *Cognitive science*, 11(1):23–63, 1987.
* Hanson & Pratt (1988)

  Stephen Hanson and Lorien Pratt.
  Comparing biases for minimal network construction with back-propagation.
  *Advances in neural information processing systems*, 1, 1988.
* He et al. (2025)

  Di He, Songjun Tu, Ajay Jaiswal, Li Shen, Ganzhao Yuan, Shiwei Liu, and Lu Yin.
  Alphadecay: Module-wise weight decay for heavy-tailed balancing in LLMs.
  In *The Thirty-ninth Annual Conference on Neural Information Processing Systems*, 2025.
* Hernandez-Garcia et al. (2025)

  J Fernando Hernandez-Garcia, Shibhansh Dohare, Jun Luo, and Rich S Sutton.
  Reinitializing weights vs units for maintaining plasticity in neural networks.
  *arXiv preprint arXiv:2508.00212*, 2025.
* Hochreiter & Schmidhuber (1997)

  S. Hochreiter and J. Schmidhuber.
  Long Short-Term Memory.
  *Neural Computation*, 9(8):1735–1780, 1997.
* Hochreiter et al. (2001)

  Sepp Hochreiter, A Steven Younger, and Peter R Conwell.
  Learning to learn using gradient descent.
  In *International conference on artificial neural networks*, pp. 87–94. Springer, 2001.
* Irie et al. (2025)

  Kazuki Irie, Róbert Csordás, and Jürgen Schmidhuber.
  Metalearning continual learning algorithms.
  *Transactions on Machine Learning Research*, 2025.
* Ishii & Sato (2017)

  Masato Ishii and Atsushi Sato.
  Layer-wise weight decay for deep neural networks.
  In *Pacific-Rim Symposium on Image and Video Technology*, pp. 276–289. Springer, 2017.
* Javed et al. (2024)

  Khurram Javed, Arsalan Sharifnassab, and Richard S Sutton.
  Swifttd: A fast and robust algorithm for temporal difference learning.
  In *Reinforcement Learning Conference*, 2024.
* Kingma & Ba (2014)

  Diederik P Kingma and Jimmy Ba.
  Adam: A method for stochastic optimization.
  *arXiv preprint arXiv:1412.6980*, 2014.
* Kirkpatrick et al. (2017)

  James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al.
  Overcoming catastrophic forgetting in neural networks.
  *Proceedings of the National Academy of Sciences*, 114:3521–3526, 2017.
* Krogh & Hertz (1991)

  Anders Krogh and John Hertz.
  A simple weight decay can improve generalization.
  *Advances in neural information processing systems*, 4, 1991.
* Kumar et al. (2025a)

  Saurabh Kumar, Henrik Marklund, Ashish Rao, Yifan Zhu, Hong Jun Jeon, Liu Yueyang, and Benjamin Van Roy.
  Continual learning as computationally constrained reinforcement learning.
  *Foundations and Trends in Machine Learning*, 18(5):913–1053, 2025a.
* Kumar et al. (2025b)

  Saurabh Kumar, Henrik Marklund, and Benjamin Van Roy.
  Maintaining plasticity in continual learning via regenerative regularization.
  In *Conference on Lifelong Learning Agents*, pp. 410–430. PMLR, 2025b.
* Lewandowski et al. (2025)

  Alex Lewandowski, Michal Bortkiewicz, Saurabh Kumar, András György, Dale Schuurmans, Mateusz Ostaszewski, and Marlos C. Machado.
  Learning continually by spectral regularization.
  In *The Thirteenth International Conference on Learning Representations, ICLR*, 2025.
* Loshchilov & Hutter (2019)

  Ilya Loshchilov and Frank Hutter.
  Decoupled weight decay regularization.
  In *International Conference on Learning Representations*, 2019.
* Luketina et al. (2022)

  Jelena Luketina, Sebastian Flennerhag, Yannick Schroecker, David Abel, Tom Zahavy, and Satinder Singh.
  Meta-gradients in non-stationary environments.
  In *Conference on Lifelong Learning Agents*, pp. 886–901. PMLR, 2022.
* MacKay (1992a)

  David J.C. MacKay.
  Bayesian interpolation.
  *Neural computation*, 4(3):415–447, 1992a.
* MacKay (1992b)

  David J.C. MacKay.
  A practical bayesian framework for backpropagation networks.
  *Neural computation*, 4(3):448–472, 1992b.
* McCloskey & Cohen (1989)

  Michael McCloskey and Neal J Cohen.
  Catastrophic interference in connectionist networks: The sequential learning problem.
  In *Psychology of learning and motivation*, volume 24, pp. 109–165. Elsevier, 1989.
* Mozer (1989)

  M.C. Mozer.
  A focused backpropagation algorithm for temporal pattern recognition.
  *Complex Systems*, 3:349–381, 1989.
* Nakamura & Hong (2019)

  Kensuke Nakamura and Byung-Woo Hong.
  Adaptive weight decay for deep neural networks.
  *IEEE Access*, 7:118857–118865, 2019.
* Nikishin et al. (2022)

  Evgenii Nikishin, Max Schwarzer, Pierluca D’Oro, Pierre-Luc Bacon, and Aaron Courville.
  The primacy bias in deep reinforcement learning.
  In *International conference on machine learning*, pp. 16828–16847. PMLR, 2022.
* Prabhu et al. (2020)

  Ameya Prabhu, Philip H.S. Torr, and Puneet K. Dokania.
  GDumb: A simple approach that questions our progress in continual learning.
  In *European Conference on Computer Vision*, 2020.
* Ratcliff (1990)

  Roger Ratcliff.
  Connectionist models of recognition memory: constraints imposed by learning and forgetting functions.
  *Psychological review*, 97(2):285, 1990.
* Robinson & Fallside (1987)

  A. J. Robinson and F. Fallside.
  The utility driven dynamic error propagation network.
  Technical Report CUED/F-INFENG/TR.1, Cambridge University Engineering Department, 1987.
* Schmidhuber (1987)

  J. Schmidhuber.
  Evolutionary principles in self-referential learning, or on learning how to learn: the meta-meta-… hook. Institut für Informatik, Technische Universität München, 1987.
* Schmidhuber (1992)

  J. Schmidhuber.
  Steps towards “self-referential” learning.
  Technical Report CU-CS-627-92, Dept. of Comp. Sci., University of Colorado at Boulder, November 1992.
* Schmidhuber (1993)

  J. Schmidhuber.
  An introspective network that can learn to run its own weight change algorithm.
  In *Proc. of the Intl. Conf. on Artificial Neural Networks, Brighton*, pp. 191–195. IEE, 1993.
* Schmidhuber et al. (1997)

  Jürgen Schmidhuber, Jieyu Zhao, and Marco Wiering.
  Shifting inductive bias with success-story algorithm, adaptive levin search, and incremental self-improvement.
  *Machine Learning*, 28(1):105–130, 1997.
* Schraudolph (1999)

  Nicol N Schraudolph.
  Local gain adaptation in stochastic gradient descent.
  In *1999 Ninth international conference on artificial neural networks ICANN 99.(Conf. Publ. No. 470)*, volume 2, pp. 569–574. IET, 1999.
* Sharifnassab et al. (2025)

  Arsalan Sharifnassab, Saber Salehkaleybar, and Richard S Sutton.
  Metaoptimize: A framework for optimizing step sizes and other meta-parameters.
  In *Forty-second International Conference on Machine Learning*, 2025.
* Sutton (1992)

  Richard S Sutton.
  Adapting bias by gradient descent: An incremental version of delta-bar-delta.
  In *AAAI*, volume 92, pp. 171–176. San Jose, CA, 1992.
* Van der Westhuizen & Lasenby (2018)

  Jos Van der Westhuizen and Joan Lasenby.
  The unreasonable effectiveness of the forget gate.
  *arXiv preprint arXiv:1804.04849*, 2018.
* Widrow & Hoff (1960)

  Bernard Widrow and Marcian E. Hoff.
  Adaptive switching circuits.
  *IRE WESCON Convention Record, New York: IRE, pp. 96-104*, 1960.
* Williams & Zipser (1989)

  Ronald J Williams and David Zipser.
  A learning algorithm for continually running fully recurrent neural networks.
  *Neural computation*, 1(2):270–280, 1989.
* Xie et al. (2023)

  Zeke Xie, Zhiqiang Xu, Jingzhao Zhang, Issei Sato, and Masashi Sugiyama.
  On the overlooked pitfalls of weight decay and how to mitigate them: A gradient-norm perspective.
  *Advances in Neural Information Processing Systems*, 36:1208–1228, 2023.
* Xu et al. (2018)

  Zhongwen Xu, Hado van Hasselt, and David Silver.
  Meta-gradient reinforcement learning.
  *Advances in neural information processing systems*, 31, 2018.
* Yang et al. (2024)

  Songlin Yang, Jan Kautz, and Ali Hatamizadeh.
  Gated delta networks: Improving mamba2 with delta rule.
  *arXiv preprint arXiv:2412.06464*, 2024.
* Zahavy et al. (2020)

  Tom Zahavy, Zhongwen Xu, Vivek Veeriah, Matteo Hessel, Junhyuk Oh, Hado van Hasselt, David Silver, and Satinder Singh.
  A self-tuning actor-critic algorithm.
  *Advances in neural information processing systems*, 33:20913–20924, 2020.

## Appendix A Derivations

### A.1 FADE

The weight update for wiw\_{i} is

|  |  |  |  |
| --- | --- | --- | --- |
|  | wt+1i=(1−λt+1i)​wti+α​δt​xti,w^{i}\_{t+1}=(1-\lambda^{i}\_{t+1})\,w^{i}\_{t}+\alpha\,\delta\_{t}\,x^{i}\_{t}, |  | (5) |

where δt=yt∗−∑jwtj​xtj\delta\_{t}=y^{\*}\_{t}-\sum\_{j}w^{j}\_{t}x^{j}\_{t}.

##### Deriving the γi\gamma^{i} update

Per-parameter weight decay λi=exp⁡(γi)\lambda^{i}=\exp({\gamma^{i}}).

Stochastic gradient descent on loss Jt=(δt)2/2J\_{t}=(\delta\_{t})^{2}/2 with respect to γi\gamma^{i}:

|  |  |  |
| --- | --- | --- |
|  | γt+1i=γti−12​θλ​∂(δt)2∂γi.\gamma^{i}\_{t+1}=\gamma^{i}\_{t}-\tfrac{1}{2}\theta\_{\lambda}\frac{\partial(\delta\_{t})^{2}}{\partial\gamma\_{i}}. |  |

Expanding via the chain rule and applying the approximation from IDBD derivation - that the primary effect of changing a meta-parameter associated with weight ii is on weight ii itself, i.e., ∂wtj/∂γi≈0\partial w^{j}\_{t}/\partial\gamma^{i}\approx 0 for j≠ij\neq i,

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂(δt)2∂γi\displaystyle\frac{\partial(\delta\_{t})^{2}}{\partial\gamma^{i}} | =∑j∂(δt)2∂wtj​∂wtj∂γi≈∂(δt)2∂wti​∂wti∂γi.\displaystyle=\sum\_{j}\frac{\partial(\delta\_{t})^{2}}{\partial w^{j}\_{t}}\frac{\partial w^{j}\_{t}}{\partial\gamma^{i}}\approx\frac{\partial(\delta\_{t})^{2}}{\partial w^{i}\_{t}}\frac{\partial w^{i}\_{t}}{\partial\gamma^{i}}. |  |

We have ∂(δt)2∂wti=−2​δt​xti\frac{\partial(\delta\_{t})^{2}}{\partial w^{i}\_{t}}=-2\delta\_{t}\,x^{i}\_{t}. Defining gti≜∂wti∂γig^{i}\_{t}\triangleq\frac{\partial w^{i}\_{t}}{\partial\gamma^{i}}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | γt+1i=γti+θλ​δt​xti​gti.\gamma^{i}\_{t+1}=\gamma^{i}\_{t}+\theta\_{\lambda}\,\delta\_{t}\,x^{i}\_{t}\,g^{i}\_{t}. |  | (6) |

##### Deriving the gig^{i} trace

We differentiate ([5](#A1.E5 "In A.1 FADE ‣ Appendix A Derivations ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")) with respect to γi\gamma^{i}.
Note that ∂λi∂γi=λi\frac{\partial\lambda^{i}}{\partial\gamma^{i}}=\lambda^{i} since λi=exp⁡γi\lambda^{i}=\exp{\gamma^{i}}.

|  |  |  |  |
| --- | --- | --- | --- |
|  | gt+1i\displaystyle g^{i}\_{t+1} | =∂∂γi​[(1−λi)​wti+α​δt​xti]\displaystyle=\frac{\partial}{\partial\gamma^{i}}\Big[(1-\lambda^{i})w^{i}\_{t}+\alpha\delta\_{t}x^{i}\_{t}\Big] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =−λi​wti+(1−λi)​gti+α​∂δt∂γi​xti.\displaystyle=-\lambda^{i}\,w^{i}\_{t}+(1-\lambda^{i})\,g^{i}\_{t}+\alpha\frac{\partial\delta\_{t}}{\partial\gamma^{i}}x^{i}\_{t}. |  |

Next, ∂δt∂γi=∂∂γi​[−∑jwtj​xtj]≈−gti​xti\frac{\partial\delta\_{t}}{\partial\gamma^{i}}=\frac{\partial}{\partial\gamma^{i}}\left[-\sum\_{j}w^{j}\_{t}x^{j}\_{t}\right]\approx-g^{i}\_{t}x^{i}\_{t} using the same IDBD approximation as earlier.
This gives us

|  |  |  |
| --- | --- | --- |
|  | gt+1i=gti​(1−λi−α​(xti)2)−λi​wti.g^{i}\_{t+1}=g^{i}\_{t}\left(1-\lambda^{i}-\alpha(x^{i}\_{t})^{2}\right)-\lambda^{i}\,w^{i}\_{t}. |  |

Adding the positive-bounding operation for stability:

|  |  |  |  |
| --- | --- | --- | --- |
|  | gt+1i=gti​[1−λt+1i−α​(xti)2]+−λt+1i​wti.g^{i}\_{t+1}=g^{i}\_{t}\big[1-\lambda^{i}\_{t+1}-\alpha(x^{i}\_{t})^{2}\big]^{+}-\lambda^{i}\_{t+1}\,w^{i}\_{t}. |  | (7) |

### A.2 FADE + IDBD

Here, the per-parameter step size αi=exp⁡βi\alpha^{i}=\exp{\beta^{i}}.

The weight update for wiw^{i} is

|  |  |  |  |
| --- | --- | --- | --- |
|  | wt+1i=(1−λt+1i)​wti+αt+1i​δt​xti,w^{i}\_{t+1}=(1-\lambda^{i}\_{t+1})\,w^{i}\_{t}+\alpha^{i}\_{t+1}\,\delta\_{t}\,x^{i}\_{t}, |  | (8) |

where δt=yt∗−∑jwtj​xtj\delta\_{t}=y^{\*}\_{t}-\sum\_{j}w^{j}\_{t}x^{j}\_{t}.

The update for γ\gamma and gg follows from the previous section.
We derive updates for β\beta and its associated trace hh.
The original IDBD trace hti≜∂wti∂βih^{i}\_{t}\triangleq\frac{\partial w^{i}\_{t}}{\partial\beta^{i}} must be re-derived because of the introduction of weight decay.

Using the same approach as earlier,

|  |  |  |  |
| --- | --- | --- | --- |
|  | βt+1i=βti+θα​δt​xti​hti.\beta^{i}\_{t+1}=\beta^{i}\_{t}+\theta\_{\alpha}\,\delta\_{t}\,x^{i}\_{t}\,h^{i}\_{t}. |  | (9) |

Differentiating Equation [8](#A1.E8 "In A.2 FADE + IDBD ‣ Appendix A Derivations ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay") with respect to βi\beta^{i}, noting ∂αi∂βi=αi\frac{\partial\alpha^{i}}{\partial\beta^{i}}=\alpha^{i} and λi\lambda^{i} does not depend on βi\beta^{i},

|  |  |  |  |
| --- | --- | --- | --- |
|  | ht+1i\displaystyle h^{i}\_{t+1} | =∂∂βi​[(1−λi)​wti+αi​δt​xti]\displaystyle=\frac{\partial}{\partial\beta^{i}}\Big[(1-\lambda^{i})w^{i}\_{t}+\alpha^{i}\delta\_{t}x^{i}\_{t}\Big] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(1−λi)​hti+αi​δt​xti+αi​∂δt∂βi​xti.\displaystyle=(1-\lambda^{i})\,h^{i}\_{t}+\alpha^{i}\,\delta\_{t}\,x^{i}\_{t}+\alpha^{i}\frac{\partial\delta\_{t}}{\partial\beta^{i}}x^{i}\_{t}. |  |

Using ∂δt∂βi≈−xti​hti\frac{\partial\delta\_{t}}{\partial\beta^{i}}\approx-x^{i}\_{t}h^{i}\_{t} and collecting terms:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ht+1i=hti​[1−λt+1i−αt+1i​(xti)2]++αt+1i​δt​xti.h^{i}\_{t+1}=h^{i}\_{t}\big[1-\lambda^{i}\_{t+1}-\alpha^{i}\_{t+1}(x^{i}\_{t})^{2}\big]^{+}+\alpha^{i}\_{t+1}\,\delta\_{t}\,x^{i}\_{t}. |  | (10) |

This is identical to IDBD’s original hih^{i} update except that λi\lambda^{i} appears in the decay factor.

### A.3 FADE with cross-entropy loss

Let there be CC classes.
The learner produces logits ztk=∑jWtk​j​xtjz^{k}\_{t}=\sum\_{j}W^{kj}\_{t}\,x^{j}\_{t} for k=1,…,Ck=1,\dots,C,
where xt∈ℝdx\_{t}\in\mathbb{R}^{d} is the input, and weights Wt∈ℝC×dW\_{t}\in\mathbb{R}^{C\times d}.
The target yt∈{1​…​C}y\_{t}\in\{1\dots C\} is the class label.
The softmax probabilities are ptk=exp⁡(ztk)/∑mexp⁡(ztm)p^{k}\_{t}=\exp({z^{k}\_{t}})/\sum\_{m}\exp({z^{m}\_{t}}),
and the cross-entropy loss for true class yty\_{t} is

|  |  |  |
| --- | --- | --- |
|  | Jt=−log⁡ptyt.J\_{t}=-\log p\_{t}^{y\_{t}}. |  |

##### Gradient

The gradient of JJ with respect to the weights is

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂Jt∂Wk​j=(ptk−𝕀​[k=yt])​xtj,\frac{\partial J\_{t}}{\partial W^{kj}}=\bigl(p^{k}\_{t}-\mathbb{I}[k=y\_{t}]\bigr)\,x^{j}\_{t}, |  | (11) |

where 𝕀​[⋅]\mathbb{I}[\cdot] is the indicator function.

##### Weight update

As in the regression case, the weight update with per-parameter decay is

|  |  |  |  |
| --- | --- | --- | --- |
|  | Wt+1k​j=(1−λt+1k​j)​Wtk​j−α​∂Jt∂Wk​j,W^{kj}\_{t+1}=(1-\lambda^{kj}\_{t+1})\,W^{kj}\_{t}-\alpha\frac{\partial J\_{t}}{\partial W^{kj}}, |  | (12) |

where λk​j=exp⁡(γk​j)\lambda^{kj}=\exp({\gamma^{kj}}).

##### Meta-gradient update for γk​j\gamma^{kj}

Following the same derivation as in Appendix [A.1](#A1.SS1 "A.1 FADE ‣ Appendix A Derivations ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay"),

|  |  |  |
| --- | --- | --- |
|  | γt+1k​j=γtk​j−θλ​∂Jt∂γk​j.\gamma^{kj}\_{t+1}=\gamma^{kj}\_{t}-\theta\_{\lambda}\frac{\partial J\_{t}}{\partial\gamma^{kj}}. |  |

Expanding via the chain rule and applying the same approximation
as in the regression case — that the primary effect of changing
γk​j\gamma^{kj} is on Wk​jW^{kj} itself, i.e.,
∂Wtk′​j′/∂γk​j≈0\partial W^{k^{\prime}j^{\prime}}\_{t}/\partial\gamma^{kj}\approx 0 for (k′,j′)≠(k,j)(k^{\prime},j^{\prime})\neq(k,j),

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂Jt∂γk​j\displaystyle\frac{\partial J\_{t}}{\partial\gamma^{kj}} | =∑k′,j′∂Jt∂Wtk′​j′​∂Wtk′​j′∂γk​j≈∂Jt∂Wtk​j​∂Wtk​j∂γk​j.\displaystyle=\sum\_{k^{\prime},j^{\prime}}\frac{\partial J\_{t}}{\partial W^{k^{\prime}j^{\prime}}\_{t}}\frac{\partial W^{k^{\prime}j^{\prime}}\_{t}}{\partial\gamma^{kj}}\approx\frac{\partial J\_{t}}{\partial W^{kj}\_{t}}\frac{\partial W^{kj}\_{t}}{\partial\gamma^{kj}}. |  |

Defining gtk​j≜∂Wtk​j/∂γk​jg^{kj}\_{t}\triangleq\partial W^{kj}\_{t}/\partial\gamma^{kj} and δtk≜(𝕀​[k=yt]−ptk)\delta^{k}\_{t}\triangleq\bigl(\mathbb{I}[k=y\_{t}]-p^{k}\_{t}\bigr):

|  |  |  |  |
| --- | --- | --- | --- |
|  | γt+1k​j=γtk​j+θλ​δtk​xtj​gtk​j\gamma^{kj}\_{t+1}=\gamma^{kj}\_{t}+\theta\_{\lambda}\,\delta^{k}\_{t}\,x^{j}\_{t}\,g^{kj}\_{t} |  | (13) |

So far the derivation is analogous to the regression case, the meta-gradient uses the
loss gradient −∂J/∂Wk​j-\partial J/\partial W^{kj} multiplied by the trace gk​jg^{kj}.

##### Trace update

We differentiate Equation [12](#A1.E12 "In Weight update ‣ A.3 FADE with cross-entropy loss ‣ Appendix A Derivations ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay") with respect to γk​j\gamma^{kj}

|  |  |  |  |
| --- | --- | --- | --- |
|  | gt+1k​j\displaystyle g^{kj}\_{t+1} | =∂∂γk​j​[(1−λk​j)​Wtk​j−α​(ptk−𝕀​[k=yt])​xtj]\displaystyle=\frac{\partial}{\partial\gamma^{kj}}\Big[(1-\lambda^{kj})W^{kj}\_{t}-\alpha(p^{k}\_{t}-\mathbb{I}[k=y\_{t}])x^{j}\_{t}\Big] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =−λk​j​Wtk​j+(1−λk​j)​gtk​j−α​∂ptk∂γk​j​xtj.\displaystyle=-\lambda^{kj}\,W^{kj}\_{t}+(1-\lambda^{kj})\,g^{kj}\_{t}-\alpha\frac{\partial p^{k}\_{t}}{\partial\gamma^{kj}}x^{j}\_{t}. |  |

For the remaining term, we expand using the chain rule through the logits:

|  |  |  |
| --- | --- | --- |
|  | ∂ptk∂γk​j=∑k′∂pk∂zk′⋅∂zk′∂γk​j.\frac{\partial p^{k}\_{t}}{\partial\gamma^{kj}}=\sum\_{k^{\prime}}\frac{\partial p^{k}}{\partial z^{k^{\prime}}}\cdot\frac{\partial z^{k^{\prime}}}{\partial\gamma^{kj}}. |  |

Applying the approximation,
∂Wk′​j′/∂γk​j≈0\partial W^{k^{\prime}j^{\prime}}/\partial\gamma^{kj}\approx 0 for (k′,j′)≠(k,j)(k^{\prime},j^{\prime})\neq(k,j),
only the k′=kk^{\prime}=k term survives:

|  |  |  |
| --- | --- | --- |
|  | ∂zk′∂γk​j=∑j′xj′​∂Wk′​j′∂γk​j≈{xj​gtk​jif ​k′=k0otherwise\frac{\partial z^{k^{\prime}}}{\partial\gamma^{kj}}=\sum\_{j^{\prime}}x^{j^{\prime}}\frac{\partial W^{k^{\prime}j^{\prime}}}{\partial\gamma^{kj}}\approx\begin{cases}x^{j}\,g^{kj}\_{t}&\text{if }k^{\prime}=k\\ 0&\text{otherwise}\end{cases} |  |

Using the standard softmax derivative
∂pk/∂zk=pk​(1−pk)\partial p^{k}/\partial z^{k}=p^{k}(1-p^{k}):

|  |  |  |
| --- | --- | --- |
|  | ∂ptk∂γk​j≈ptk​(1−ptk)⋅xtj⋅gtk​j.\frac{\partial p^{k}\_{t}}{\partial\gamma^{kj}}\approx p^{k}\_{t}(1-p^{k}\_{t})\cdot x^{j}\_{t}\cdot g^{kj}\_{t}. |  |

Substituting:

|  |  |  |
| --- | --- | --- |
|  | gt+1k​j=gtk​j​[1−λk​j−α​ptk​(1−ptk)​(xtj)2]−λk​j​Wtk​j.g^{kj}\_{t+1}=g^{kj}\_{t}\Big[1-\lambda^{kj}-\alpha\,p^{k}\_{t}(1-p^{k}\_{t})\,(x^{j}\_{t})^{2}\Big]-\lambda^{kj}\,W^{kj}\_{t}. |  |

Adding the positive-bounding operation for stability:

|  |  |  |  |
| --- | --- | --- | --- |
|  | gt+1k​j=gtk​j​[1−λt+1k​j−α​ptk​(1−ptk)​(xtj)2]+−λt+1k​j​Wtk​j.g^{kj}\_{t+1}=g^{kj}\_{t}\Big[1-\lambda^{kj}\_{t+1}-\alpha\,p^{k}\_{t}(1-p^{k}\_{t})\,(x^{j}\_{t})^{2}\Big]^{+}-\lambda^{kj}\_{t+1}\,W^{kj}\_{t}. |  | (14) |

Compared to the regression trace (Equation [7](#A1.E7 "In Deriving the 𝑔^𝑖 trace ‣ A.1 FADE ‣ Appendix A Derivations ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")),
the only difference is the factor ptk​(1−ptk)∈[0,0.25]p^{k}\_{t}(1-p^{k}\_{t})\in[0,0.25]
multiplying (xtj)2(x^{j}\_{t})^{2}.

### A.4 FADE with Adam

When combining FADE with Adam, the base optimizer provides an effective per-parameter step size.

At step tt, Adam’s effective step size for parameter ii is

|  |  |  |  |
| --- | --- | --- | --- |
|  | αeff,ti=αv^ti+ϵ,\alpha\_{\text{eff},t}^{i}=\frac{\alpha}{\sqrt{\hat{v}\_{t}^{i}}+\epsilon}, |  | (15) |

where v^ti=vti/(1−β2t)\hat{v}\_{t}^{i}=v\_{t}^{i}/(1-\beta\_{2}^{t}) is the bias-corrected second moment estimate (Kingma & Ba, [2014](#bib.bib21)).
The FADE trace update (Equation [7](#A1.E7 "In Deriving the 𝑔^𝑖 trace ‣ A.1 FADE ‣ Appendix A Derivations ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")) becomes

|  |  |  |  |
| --- | --- | --- | --- |
|  | gt+1i←gti​[1−λt+1i−αeff,ti​(xti)2]+−λt+1i​wti.g\_{t+1}^{i}\leftarrow g\_{t}^{i}\left[1-\lambda\_{t+1}^{i}-\alpha\_{\text{eff},t}^{i}(x\_{t}^{i})^{2}\right]\_{+}-\lambda\_{t+1}^{i}w\_{t}^{i}. |  | (16) |

The γ\gamma update and weight decay application remain unchanged.

Adam handles the gradient step, and FADE applies adaptive weight decay independently, analogous to the relationship between Adam and AdamW.
Note that this approximation does not account for the effect of Adam’s first moment (momentum) on the trace, which would require tracking additional dependencies.

### A.5 Coupled decay and step size adaptation for online linear regression

Here we explore an alternative to FADE+IDBD where the weight update comes from SGD on the L2L\_{2} regularized loss, i.e.

|  |  |  |  |
| --- | --- | --- | --- |
|  | wt+1i=(1−αt+1i​λt+1i)​wti+αt+1i​δt​xti.w^{i}\_{t+1}=(1-\alpha^{i}\_{t+1}\lambda^{i}\_{t+1})\,w^{i}\_{t}+\alpha^{i}\_{t+1}\,\delta\_{t}\,x^{i}\_{t}. |  | (17) |

Note that in this update, the decay term involves a product with the per-parameter step size.
We proceed with the same derivation technique.

##### Updates to γi\gamma^{i} and βi\beta^{i}

The meta-parameter updates follow the same form as in the decoupled case:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | βt+1i\displaystyle\beta^{i}\_{t+1} | =βti+θα​δt​xti​hti,αt+1i=exp⁡(βt+1i),\displaystyle=\beta^{i}\_{t}+\theta\_{\alpha}\,\delta\_{t}\,x^{i}\_{t}\,h^{i}\_{t},\quad\alpha^{i}\_{t+1}=\exp(\beta^{i}\_{t+1}), |  | (18) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | γt+1i\displaystyle\gamma^{i}\_{t+1} | =γti+θλ​δt​xti​gti,λt+1i=exp⁡(γt+1i).\displaystyle=\gamma^{i}\_{t}+\theta\_{\lambda}\,\delta\_{t}\,x^{i}\_{t}\,g^{i}\_{t},\quad\lambda^{i}\_{t+1}=\exp(\gamma^{i}\_{t+1}). |  | (19) |

The traces gti≜∂wti/∂γig^{i}\_{t}\triangleq\partial w^{i}\_{t}/\partial\gamma^{i} and hti≜∂wti/∂βih^{i}\_{t}\triangleq\partial w^{i}\_{t}/\partial\beta^{i} must be re-derived to account for the coupling between αi\alpha^{i} and λi\lambda^{i} in the decay term.

##### Deriving the gig^{i} trace

We differentiate the weight update ([17](#A1.E17 "In A.5 Coupled decay and step size adaptation for online linear regression ‣ Appendix A Derivations ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")) with respect to γi\gamma^{i}.
Note that ∂λi/∂γi=λi\partial\lambda^{i}/\partial\gamma^{i}=\lambda^{i} and αi\alpha^{i} does not depend on γi\gamma^{i}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | gt+1i\displaystyle g^{i}\_{t+1} | =∂∂γi​[(1−αi​λi)​wti+αi​δt​xti]\displaystyle=\frac{\partial}{\partial\gamma^{i}}\Big[(1-\alpha^{i}\lambda^{i})w^{i}\_{t}+\alpha^{i}\delta\_{t}x^{i}\_{t}\Big] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =−αi​λi​wti+(1−αi​λi)​gti+αi​∂δt∂γi​xti.\displaystyle=-\alpha^{i}\lambda^{i}\,w^{i}\_{t}+(1-\alpha^{i}\lambda^{i})\,g^{i}\_{t}+\alpha^{i}\frac{\partial\delta\_{t}}{\partial\gamma^{i}}x^{i}\_{t}. |  |

Using ∂δt/∂γi≈−gti​xti\partial\delta\_{t}/\partial\gamma^{i}\approx-g^{i}\_{t}x^{i}\_{t} and adding the positive-bounding operation:

|  |  |  |  |
| --- | --- | --- | --- |
|  | gt+1i=gti​[1−αt+1i​λt+1i−αt+1i​(xti)2]+−αt+1i​λt+1i​wti.g^{i}\_{t+1}=g^{i}\_{t}\big[1-\alpha^{i}\_{t+1}\lambda^{i}\_{t+1}-\alpha^{i}\_{t+1}(x^{i}\_{t})^{2}\big]^{+}-\alpha^{i}\_{t+1}\lambda^{i}\_{t+1}\,w^{i}\_{t}. |  | (20) |

This has a similar structure to the decoupled case (Equation [7](#A1.E7 "In Deriving the 𝑔^𝑖 trace ‣ A.1 FADE ‣ Appendix A Derivations ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")), but with λi\lambda^{i} replaced by αi​λi\alpha^{i}\lambda^{i} throughout.

##### Deriving the hih^{i} trace

We differentiate the weight update ([17](#A1.E17 "In A.5 Coupled decay and step size adaptation for online linear regression ‣ Appendix A Derivations ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")) with respect to βi\beta^{i}.
Note that ∂αi/∂βi=αi\partial\alpha^{i}/\partial\beta^{i}=\alpha^{i}. Crucially, because the decay term is αi​λi\alpha^{i}\lambda^{i}, it now depends on βi\beta^{i}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ht+1i\displaystyle h^{i}\_{t+1} | =∂∂βi​[(1−αi​λi)​wti+αi​δt​xti]\displaystyle=\frac{\partial}{\partial\beta^{i}}\Big[(1-\alpha^{i}\lambda^{i})w^{i}\_{t}+\alpha^{i}\delta\_{t}x^{i}\_{t}\Big] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =−αi​λi​wti+(1−αi​λi)​hti+αi​δt​xti+αi​∂δt∂βi​xti.\displaystyle=-\alpha^{i}\lambda^{i}\,w^{i}\_{t}+(1-\alpha^{i}\lambda^{i})\,h^{i}\_{t}+\alpha^{i}\delta\_{t}\,x^{i}\_{t}+\alpha^{i}\frac{\partial\delta\_{t}}{\partial\beta^{i}}x^{i}\_{t}. |  |

Using ∂δt/∂βi≈−hti​xti\partial\delta\_{t}/\partial\beta^{i}\approx-h^{i}\_{t}x^{i}\_{t} and adding the positive-bounding operation:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ht+1i=hti​[1−αt+1i​λt+1i−αt+1i​(xti)2]++αt+1i​δt​xti−αt+1i​λt+1i​wti.h^{i}\_{t+1}=h^{i}\_{t}\big[1-\alpha^{i}\_{t+1}\lambda^{i}\_{t+1}-\alpha^{i}\_{t+1}(x^{i}\_{t})^{2}\big]^{+}+\alpha^{i}\_{t+1}\delta\_{t}\,x^{i}\_{t}-\alpha^{i}\_{t+1}\lambda^{i}\_{t+1}\,w^{i}\_{t}. |  | (21) |

Compared to the decoupled hih^{i} trace (Equation [10](#A1.E10 "In A.2 FADE + IDBD ‣ Appendix A Derivations ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")), which has the form

|  |  |  |
| --- | --- | --- |
|  | ht+1i=hti​[1−λt+1i−αt+1i​(xti)2]++αt+1i​δt​xti,h^{i}\_{t+1}=h^{i}\_{t}\big[1-\lambda^{i}\_{t+1}-\alpha^{i}\_{t+1}(x^{i}\_{t})^{2}\big]^{+}+\alpha^{i}\_{t+1}\delta\_{t}\,x^{i}\_{t}, |  |

the coupled trace acquires an additional −αt+1i​λt+1i​wti-\alpha^{i}\_{t+1}\lambda^{i}\_{t+1}\,w^{i}\_{t} term.

## Appendix B Implementation details

### B.1 Linear tracking

##### Hyperparameter search

For each algorithm, we perform a grid search over step size and meta-parameters, evaluating MSE averaged over all 200K steps across 10 seeds.
Selected (best) hyperparameters are indicated in bold.

#### B.1.1 Noise σn=0\sigma\_{n}=0

##### SGD

Step size α∈{0.5,0.1,0.05,0.01,0.005,0.001}\alpha\in\{0.5,0.1,\mathbf{0.05},0.01,0.005,0.001\}.
No weight decay, i.e., λ=0\lambda=0.

##### SGD + WD

Step size α∈{0.5,0.1,0.05,0.01,0.005,0.001}\alpha\in\{0.5,0.1,\mathbf{0.05},0.01,0.005,0.001\}, weight decay λ∈{0.001,0.005,0.01,0.05,0.1,0.5}\lambda\in\{0.001,0.005,0.01,\mathbf{0.05},0.1,0.5\}.

##### IDBD

Meta step size θα∈{0.5,0.1,0.05,0.01,0.005,0.001}\theta\_{\alpha}\in\{0.5,0.1,0.05,0.01,\mathbf{0.005},0.001\}, initial step size β0∈{−0.7,−2.3,−3,−4.6,−5.3}\beta\_{0}\in\{-0.7,-2.3,-3,\mathbf{-4.6},-5.3\}.

##### IDBD + WD

Meta step size θα∈{0.1,0.05,0.01,0.005}\theta\_{\alpha}\in\{0.1,0.05,\mathbf{0.01},0.005\}, initial step size β0∈{−0.7,−2.3,−3,−4.6,−5.3}\beta\_{0}\in\{-0.7,-2.3,-3,\mathbf{-4.6},-5.3\}, weight decay λ∈{0.001,0.005,0.01,0.05,0.1,0.5}\lambda\in\{0.001,0.005,\mathbf{0.01},0.05,0.1,0.5\}.

##### FADE

Step size α∈{0.1,0.05,0.01,0.005}\alpha\in\{\mathbf{0.1},0.05,0.01,0.005\}, meta step size θλ∈{0.1,0.05,0.01,0.005}\theta\_{\lambda}\in\{0.1,0.05,\mathbf{0.01},0.005\}, initial decay γ0∈{0,−0.7,−1.2,−2.3,−4.6,−6.9,−9.2}\gamma\_{0}\in\{0,-0.7,\mathbf{-1.2},-2.3,-4.6,-6.9,-9.2\}.

##### FADE + IDBD

Tied meta step size θα=θλ∈{0.1,0.05,0.01,0.005}\theta\_{\alpha}=\theta\_{\lambda}\in\{0.1,0.05,\mathbf{0.01},0.005\}, initial step size β0∈{−0.7,−2.3,−3,−4.6,−5.3}\beta\_{0}\in\{-0.7,-2.3,-3,\mathbf{-4.6},-5.3\}, initial decay γ0∈{−0.7,−1.2,−2.3,−4.6}\gamma\_{0}\in\{-0.7,-1.2,\mathbf{-2.3},-4.6\}.

##### Coupled WD + SS Adaptation

Tied meta step size θα=θλ∈{0.1,0.05,0.01,0.005}\theta\_{\alpha}=\theta\_{\lambda}\in\{0.1,0.05,0.01,0.005\}, initial step size β0∈{−0.7,−2.3,−3,−4.6,−5.3}\beta\_{0}\in\{-0.7,-2.3,-3,-4.6,-5.3\}, initial decay γ0∈{−0.7,−1.2,−2.3,−4.6}\gamma\_{0}\in\{-0.7,-1.2,-2.3,-4.6\}.

#### B.1.2 Noise σn=1.0\sigma\_{n}=1.0

##### SGD

Step size α∈{0.5,0.1,0.05,0.01,0.005,0.001}\alpha\in\{0.5,0.1,0.05,\mathbf{0.01},0.005,0.001\}.
No weight decay, i.e., λ=0\lambda=0.

##### SGD + WD

Step size α∈{0.5,0.1,0.05,0.01,0.005,0.001}\alpha\in\{0.5,0.1,\mathbf{0.05},0.01,0.005,0.001\}, weight decay λ∈{0.001,0.005,0.01,0.05,0.1,0.5}\lambda\in\{0.001,0.005,0.01,\mathbf{0.05},0.1,0.5\}.

##### IDBD

Meta step size θα∈{0.5,0.1,0.05,0.01,0.005,0.001}\theta\_{\alpha}\in\{0.5,0.1,0.05,\mathbf{0.01},0.005,0.001\}, initial step size β0∈{−0.7,−2.3,−3,−4.6,−5.3,−6.9}\beta\_{0}\in\{-0.7,-2.3,-3,-4.6,\mathbf{-5.3},-6.9\}.
No weight decay, i.e., λ=0\lambda=0.

##### IDBD + WD

Meta step size θα∈{0.1,0.05,0.01,0.005}\theta\_{\alpha}\in\{0.1,0.05,\mathbf{0.01},0.005\}, initial step size β0∈{−0.7,−2.3,−3,−4.6,−5.3}\beta\_{0}\in\{-0.7,-2.3,-3,\mathbf{-4.6},-5.3\}, weight decay λ∈{0.001,0.005,0.01,0.05,0.1,0.5}\lambda\in\{0.001,0.005,\mathbf{0.01},0.05,0.1,0.5\}.

##### FADE

Step size α∈{0.1,0.05,0.01,0.005}\alpha\in\{0.1,\mathbf{0.05},0.01,0.005\}, meta step size θλ∈{0.1,0.05,0.01,0.005}\theta\_{\lambda}\in\{0.1,0.05,\mathbf{0.01},0.005\}, initial decay γ0∈{0,−0.7,−1.2,−2.3,−4.6,−6.9,−9.2}\gamma\_{0}\in\{0,-0.7,\mathbf{-1.2},-2.3,-4.6,-6.9,-9.2\}.

##### FADE + IDBD

Tied meta step size θα=θλ∈{0.1,0.05,0.01,0.005}\theta\_{\alpha}=\theta\_{\lambda}\in\{0.1,0.05,\mathbf{0.01},0.005\}, initial step size β0∈{−0.7,−2.3,−3,−4.6,−5.3}\beta\_{0}\in\{-0.7,-2.3,-3,\mathbf{-4.6},-5.3\}, initial decay γ0∈{−0.7,−1.2,−2.3,−4.6}\gamma\_{0}\in\{-0.7,-1.2,\mathbf{-2.3},-4.6\}.

##### Coupled WD + SS Adaptation

Tied meta step size θα=θλ∈{0.1,0.05,0.01,0.005}\theta\_{\alpha}=\theta\_{\lambda}\in\{0.1,0.05,0.01,0.005\}, initial step size β0∈{−0.7,−2.3,−3,−4.6,−5.3}\beta\_{0}\in\{-0.7,-2.3,-3,-4.6,-5.3\}, initial decay γ0∈{−0.7,−1.2,−2.3,−4.6}\gamma\_{0}\in\{-0.7,-1.2,-2.3,-4.6\}.

### B.2 Non-linear tracking

##### Hyperparameter search

For each algorithm, we perform a grid search over step size and regularization parameters, evaluating MSE averaged over the final 500K of 2M training steps across 5 seeds.
All other unspecified hyperparameters (e.g., β1,β2,ϵ\beta\_{1},\beta\_{2},\epsilon for Adam (Kingma & Ba, [2014](#bib.bib21))) are PyTorch defaults.
Selected (best) hyperparameters are indicated in bold.

##### SGD

Step size α∈{0.1,0.01,0.001,0.0001}\alpha\in\{0.1,\mathbf{0.01},0.001,0.0001\}. No weight decay, i.e., λ=0\lambda=0.

##### Adam

Step size α∈{0.1,0.01,0.001,0.0001}\alpha\in\{0.1,0.01,\mathbf{0.001},0.0001\}.
No weight decay, i.e., λ=0\lambda=0.

##### SGD + WD

Step size α∈{ 0.1,0.01,0.001,0.0001}\alpha\in\{\ 0.1,\mathbf{0.01},0.001,0.0001\} and weight decay λ∈{1​e−5,1​e−4,𝟏​𝐞−𝟑,1​e−2,1​e−1}\lambda\in\{1e-5,1e-4,\mathbf{1e-3},1e-2,1e-1\}.

##### AdamW

Step size α∈{0.1,0.01,0.001,0.0001}\alpha\in\{0.1,0.01,\mathbf{0.001},0.0001\}.
Weight decay λ∈{10,1,0.1,0.01,0.001,0.0001}\lambda\in\{10,1,\mathbf{0.1},0.01,0.001,0.0001\}.

##### FADE + SGD

Step size α∈{0.1,0.01,0.001}\alpha\in\{0.1,\mathbf{0.01},0.001\}, meta step size θλ∈{0,0.5,1,𝟐}\theta\_{\lambda}\in\{0,0.5,1,\mathbf{2}\}.
We also consider (and report results for) different values for γ0∈{−2.3,−4.6,−6.9,−9.2}\gamma\_{0}\in\{-2.3,-4.6,-6.9,\mathbf{-9.2}\}.

##### FADE + Adam

Step size α∈{0.01,0.001,0.0001,0.00001}\alpha\in\{0.01,0.001,\mathbf{0.0001},0.00001\}, meta step size θλ∈{0,0.5,𝟏,2}\theta\_{\lambda}\in\{0,0.5,\mathbf{1},2\}.
We also consider different values for γ0∈{−2.3,−4.6,−6.9,−9.2}\gamma\_{0}\in\{-2.3,-4.6,-6.9,\mathbf{-9.2}\}.

### B.3 Streaming image classification with label permutation

For each algorithm, we perform a grid search over step size and regularization parameters, evaluating online accuracy averaged over all 5M training steps across 5 seeds.
All other unspecified hyperparameters (e.g., β1,β2,ϵ\beta\_{1},\beta\_{2},\epsilon for Adam (Kingma & Ba, [2014](#bib.bib21))) are PyTorch defaults.
Selected (best) hyperparameters are indicated in bold.

##### SGD.

Step size α∈{0.05,0.01,0.005,0.001,0.0005}\alpha\in\{0.05,\mathbf{0.01},0.005,0.001,0.0005\}.

##### SGD + WD.

Step size α∈{0.05,0.01,0.005,0.001}\alpha\in\{0.05,\mathbf{0.01},0.005,0.001\}, and weight decay λ∈{1​e−2,1​e−3,𝟏​𝐞−𝟒,1​e−5,1​e−6}\lambda\in\{1e-2,1e-3,\mathbf{1e-4},1e-5,1e-6\}.

##### Adam.

Step size α∈{0.005,0.001,0.0005,0.0001,0.00005}\alpha\in\{0.005,0.001,\mathbf{0.0005},0.0001,0.00005\}.

##### AdamW.

Step size α∈{0.005,0.001,0.0005,0.0001,0.00005}\alpha\in\{0.005,0.001,\mathbf{0.0005},0.0001,0.00005\}, and weight decay λ∈{1,0.1,0.01,0.001,0.0001}\lambda\in\{1,\mathbf{0.1},0.01,0.001,0.0001\}.

##### SGD + Weight Clipping.

Step size α∈{0.1,0.05,0.01,0.005,0.001}\alpha\in\{0.1,0.05,0.01,\mathbf{0.005},0.001\},
clipping parameter κ∈{1,𝟐,3,4,5}\kappa\in\{1,\mathbf{2},3,4,5\}.

##### FADE + SGD.

Step size α∈{0.05,0.01,0.005,0.001}\alpha\in\{0.05,0.01,\mathbf{0.005},0.001\},
meta-step size θλ∈{0,0.1,0.5,1}\theta\_{\lambda}\in\{0,\mathbf{0.1},0.5,1\}, and γ0∈{−2.3,−4.6,−6.9,−9.2}\gamma\_{0}\in\{-2.3,-4.6,\mathbf{-6.9},-9.2\}

##### FADE + Adam.

Step size α∈{0.001,0.0005,0.0001,0.00005}\alpha\in\{0.001,0.0005,\mathbf{0.0001},0.00005\},
meta-step size θλ∈{0,0.1,0.5,1}\theta\_{\lambda}\in\{0,\mathbf{0.1},0.5,1\}, and γ0∈{−2.3,−4.6,−6.9,−9.2}\gamma\_{0}\in\{-2.3,-4.6,\mathbf{-6.9},-9.2\}

## Appendix C Streaming classification with partial label permuted EMNIST

We use the same selected hyperparameters as the setting with label permutations applied to all classes,
these results are provided in Table [8](#S3.T8 "Table 8 ‣ Analysis with partial label permutations. ‣ 3.3 Streaming image classification with label permutation ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay") and Figure [6](#A3.F6 "Figure 6 ‣ Appendix C Streaming classification with partial label permuted EMNIST ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay").

In this setting, SGD without weight decay outperforms SGD with weight decay
(Table [8](#S3.T8 "Table 8 ‣ Analysis with partial label permutations. ‣ 3.3 Streaming image classification with label permutation ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")).
This further motivates weight-decay adaptation based on the problem.

![Refer to caption](/html/2604.27063/assets/x9.png)


Figure 6: Online accuracy with partial label permutations.

Sensitivity analysis for FADE + SGD and FADE + Adam are provided in Table [9](#A3.T9 "Table 9 ‣ Appendix C Streaming classification with partial label permuted EMNIST ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay") and Table [10](#A3.T10 "Table 10 ‣ Appendix C Streaming classification with partial label permuted EMNIST ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay").
Some alternative hyperparameter settings for FADE are slightly better than the default settings used from the setup where all labels were permuted (e.g. 0.8410.841 vs 0.8460.846 in Table [9](#A3.T9 "Table 9 ‣ Appendix C Streaming classification with partial label permuted EMNIST ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")).

Table 9: Impact of FADE’s adaptive decay on *partial label-permuted* EMNIST (24 stable classes which never change labels) with SGD.
Results are average online accuracy ±\pm standard deviation across 5 seeds, with step size α=0.005\alpha=0.005.
θλ=0\theta\_{\lambda}=0 corresponds to fixed weight decay on the head.

|  | θλ=0\theta\_{\lambda}=0 | θλ=0.1\theta\_{\lambda}=0.1 | θλ=0.5\theta\_{\lambda}=0.5 | θλ=1.0\theta\_{\lambda}=1.0 |
| --- | --- | --- | --- | --- |
| γ0=−4.6​(λ0≈0.01)\gamma\_{0}=-4.6\ (\lambda\_{0}\approx 0.01) | 0.699±0.0000.699\pm 0.000 | 0.811±0.0050.811\pm 0.005 | 0.816±0.0060.816\pm 0.006 | 0.822±0.0060.822\pm 0.006 |
| γ0=−6.9​(λ0≈0.001)\gamma\_{0}=-6.9\ (\lambda\_{0}\approx 0.001) | 0.810±0.0010.810\pm 0.001 | 0.841±0.0010.841\pm 0.001 | 0.845±0.0020.845\pm 0.002 | 0.846±0.0030.846\pm 0.003 |
| γ0=−9.2​(λ0≈0.0001)\gamma\_{0}=-9.2\ (\lambda\_{0}\approx 0.0001) | 0.830±0.0020.830\pm 0.002 | 0.845±0.0020.845\pm 0.002 | 0.846±0.0020.846\pm 0.002 | 0.845±0.0020.845\pm 0.002 |
| γ0=−11.5​(λ0≈0.00001)\gamma\_{0}=-11.5\ (\lambda\_{0}\approx 0.00001) | 0.783±0.0060.783\pm 0.006 | 0.818±0.0030.818\pm 0.003 | 0.833±0.0030.833\pm 0.003 | 0.835±0.0020.835\pm 0.002 |




Table 10: Impact of FADE’s adaptive decay on *partial label-permuted* EMNIST (24 stable classes which never change labels) with Adam.
Results are average online accuracy ±\pm standard deviation across 5 seeds, with step size α=0.0001\alpha=0.0001.
θλ=0\theta\_{\lambda}=0 corresponds to fixed weight decay on the head.

|  | θλ=0\theta\_{\lambda}=0 | θλ=0.1\theta\_{\lambda}=0.1 | θλ=0.5\theta\_{\lambda}=0.5 | θλ=1.0\theta\_{\lambda}=1.0 |
| --- | --- | --- | --- | --- |
| γ0=−4.6​(λ0≈0.01)\gamma\_{0}=-4.6\ (\lambda\_{0}\approx 0.01) | 0.560±0.0010.560\pm 0.001 | 0.799±0.0010.799\pm 0.001 | 0.807±0.0020.807\pm 0.002 | 0.800±0.0020.800\pm 0.002 |
| γ0=−6.9​(λ0≈0.001)\gamma\_{0}=-6.9\ (\lambda\_{0}\approx 0.001) | 0.763±0.0000.763\pm 0.000 | 0.808±0.0020.808\pm 0.002 | 0.808±0.0020.808\pm 0.002 | 0.803±0.0030.803\pm 0.003 |
| γ0=−9.2​(λ0≈0.0001)\gamma\_{0}=-9.2\ (\lambda\_{0}\approx 0.0001) | 0.753±0.0020.753\pm 0.002 | 0.810±0.0010.810\pm 0.001 | 0.810±0.0020.810\pm 0.002 | 0.797±0.0020.797\pm 0.002 |
| γ0=−11.5​(λ0≈0.00001)\gamma\_{0}=-11.5\ (\lambda\_{0}\approx 0.00001) | 0.549±0.0110.549\pm 0.011 | 0.728±0.0050.728\pm 0.005 | 0.732±0.0030.732\pm 0.003 | 0.734±0.0050.734\pm 0.005 |

## Appendix D Algorithms

We present pseudocode for IDBD (Algorithm [2](#alg2 "Algorithm 2 ‣ Appendix D Algorithms ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")), IDBD with fixed weight decay (Algorithm [3](#alg3 "Algorithm 3 ‣ Appendix D Algorithms ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")), and FADE + IDBD (Algorithm [4](#alg4 "Algorithm 4 ‣ Appendix D Algorithms ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")).
These algorithms are used in the linear tracking experiment (Section [3.1](#S3.SS1 "3.1 Linear Tracking ‣ 3 Experiments ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay")).
IDBD + WD is a special case of FADE + IDBD in which the decay rate λ\lambda
is fixed rather than adapted.
Removing the γ\gamma update and
gg trace from Algorithm [4](#alg4 "Algorithm 4 ‣ Appendix D Algorithms ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay") and replacing the adaptive λt+1i\lambda^{i}\_{t+1}
with a fixed scalar λ\lambda recovers Algorithm [3](#alg3 "Algorithm 3 ‣ Appendix D Algorithms ‣ Learning to Forget: Continual Learning with Adaptive Weight Decay").
We tie meta-step sizes θα=θλ=θ\theta\_{\alpha}=\theta\_{\lambda}=\theta in our experiment with FADE+IDBD.

Algorithm 2  IDBD: Incremental Delta-Bar-Delta (Sutton, [1992](#bib.bib44))

0: meta-step size θα\theta\_{\alpha}, initial step-size parameter β0∈ℝd\beta\_{0}\in\mathbb{R}^{d}

1: Initialize weights w0∈ℝdw\_{0}\in\mathbb{R}^{d}, traces h0←𝟎∈ℝdh\_{0}\leftarrow\mathbf{0}\in\mathbb{R}^{d}

2: Initialize α0i←exp⁡(β0i)\alpha\_{0}^{i}\leftarrow\exp({\beta\_{0}^{i}}) for all ii

3: for t=0,1,2,…t=0,1,2,\ldots do

4:  Receive input xt∈ℝdx\_{t}\in\mathbb{R}^{d} and target yt∗∈ℝy^{\*}\_{t}\in\mathbb{R}

5:  Predict yt←⟨wt,xt⟩y\_{t}\leftarrow\langle w\_{t},x\_{t}\rangle

6:  Compute error δt←yt∗−yt\delta\_{t}\leftarrow y^{\*}\_{t}-y\_{t}

7:  for each parameter i=1,…,di=1,\ldots,d do

8:   # Adapt step size

9:   βt+1i←βti+θα​δt​xti​hti\beta\_{t+1}^{i}\leftarrow\beta\_{t}^{i}+\theta\_{\alpha}\,\delta\_{t}\,x\_{t}^{i}\,h\_{t}^{i}

10:   αt+1i←exp⁡(βt+1i)\alpha\_{t+1}^{i}\leftarrow\exp({\beta\_{t+1}^{i}})

11:   # Update weight

12:   wt+1i←wti+αt+1i​δt​xtiw\_{t+1}^{i}\leftarrow w\_{t}^{i}+\alpha\_{t+1}^{i}\,\delta\_{t}\,x\_{t}^{i}

13:   # Update sensitivity trace

14:   ht+1i←hti​[1−αt+1i​(xti)2]++αt+1i​δt​xtih\_{t+1}^{i}\leftarrow h\_{t}^{i}\big[1-\alpha\_{t+1}^{i}(x\_{t}^{i})^{2}\big]^{+}+\alpha\_{t+1}^{i}\,\delta\_{t}\,x\_{t}^{i}

15:  end for

16: end for




Algorithm 3  IDBD + WD: Adaptive Step Size with Fixed Weight Decay (online linear regression)

0: meta-step size θα\theta\_{\alpha}, initial step-size parameter β0∈ℝd\beta\_{0}\in\mathbb{R}^{d}, weight decay λ∈ℝ\lambda\in\mathbb{R}

1: Initialize weights w0∈ℝdw\_{0}\in\mathbb{R}^{d}, traces h0←𝟎∈ℝdh\_{0}\leftarrow\mathbf{0}\in\mathbb{R}^{d}

2: Initialize α0i←exp⁡(β0i)\alpha\_{0}^{i}\leftarrow\exp({\beta\_{0}^{i}}) for all ii

3: for t=0,1,2,…t=0,1,2,\ldots do

4:  Receive input xt∈ℝdx\_{t}\in\mathbb{R}^{d} and target yt∗∈ℝy^{\*}\_{t}\in\mathbb{R}

5:  Predict yt←⟨wt,xt⟩y\_{t}\leftarrow\langle w\_{t},x\_{t}\rangle

6:  Compute error δt←yt∗−yt\delta\_{t}\leftarrow y^{\*}\_{t}-y\_{t}

7:  for each parameter i=1,…,di=1,\ldots,d do

8:   # Adapt step size

9:   βt+1i←βti+θα​δt​xti​hti\beta\_{t+1}^{i}\leftarrow\beta\_{t}^{i}+\theta\_{\alpha}\,\delta\_{t}\,x\_{t}^{i}\,h\_{t}^{i}

10:   αt+1i←exp⁡(βt+1i)\alpha\_{t+1}^{i}\leftarrow\exp({\beta\_{t+1}^{i}})

11:   # Update weight with fixed decay

12:   wt+1i←(1−λ)​wti+αt+1i​δt​xtiw\_{t+1}^{i}\leftarrow(1-\lambda)\,w\_{t}^{i}+\alpha\_{t+1}^{i}\,\delta\_{t}\,x\_{t}^{i}

13:   # Update sensitivity trace

14:   ht+1i←hti​[1−λ−αt+1i​(xti)2]++αt+1i​δt​xtih\_{t+1}^{i}\leftarrow h\_{t}^{i}\big[1-\lambda-\alpha\_{t+1}^{i}(x\_{t}^{i})^{2}\big]^{+}+\alpha\_{t+1}^{i}\,\delta\_{t}\,x\_{t}^{i}

15:  end for

16: end for




Algorithm 4  FADE + IDBD: Adaptive Decay and Step Size (online linear regression)

0: meta-step sizes θα,θλ\theta\_{\alpha},\theta\_{\lambda} (can be tied: θα=θλ=θ\theta\_{\alpha}=\theta\_{\lambda}=\theta), initial β0,γ0∈ℝd\beta\_{0},\gamma\_{0}\in\mathbb{R}^{d}

1: Initialize weights w0∈ℝdw\_{0}\in\mathbb{R}^{d}, traces h0←𝟎h\_{0}\leftarrow\mathbf{0}, g0←𝟎∈ℝdg\_{0}\leftarrow\mathbf{0}\in\mathbb{R}^{d}

2: Initialize α0i←exp⁡(β0i)\alpha\_{0}^{i}\leftarrow\exp({\beta\_{0}^{i}}), λ0i←exp⁡(γ0i)\lambda\_{0}^{i}\leftarrow\exp({\gamma\_{0}^{i}}) for all ii

3: for t=0,1,2,…t=0,1,2,\ldots do

4:  Receive input xt∈ℝdx\_{t}\in\mathbb{R}^{d} and target yt∗∈ℝy^{\*}\_{t}\in\mathbb{R}

5:  Predict yt←⟨wt,xt⟩y\_{t}\leftarrow\langle w\_{t},x\_{t}\rangle, compute error δt←yt∗−yt\delta\_{t}\leftarrow y^{\*}\_{t}-y\_{t}

6:  for each parameter i=1,…,di=1,\ldots,d do

7:   # Adapt step size and decay rate

8:   βt+1i←βti+θα​δt​xti​hti\beta\_{t+1}^{i}\leftarrow\beta\_{t}^{i}+\theta\_{\alpha}\,\delta\_{t}\,x\_{t}^{i}\,h\_{t}^{i},  αt+1i←exp⁡(βt+1i)\alpha\_{t+1}^{i}\leftarrow\exp({\beta\_{t+1}^{i}})

9:   γt+1i←γti+θλ​δt​xti​gti\gamma\_{t+1}^{i}\leftarrow\gamma\_{t}^{i}+\theta\_{\lambda}\,\delta\_{t}\,x\_{t}^{i}\,g\_{t}^{i},  λt+1i←exp⁡(γt+1i)\lambda\_{t+1}^{i}\leftarrow\exp({\gamma\_{t+1}^{i}})

10:   # Update traces

11:   ht+1i←hti​[1−λt+1i−αt+1i​(xti)2]++αt+1i​δt​xtih\_{t+1}^{i}\leftarrow h\_{t}^{i}\big[1-\lambda\_{t+1}^{i}-\alpha\_{t+1}^{i}(x\_{t}^{i})^{2}\big]^{+}+\alpha\_{t+1}^{i}\,\delta\_{t}\,x\_{t}^{i}

12:   gt+1i←gti​[1−λt+1i−αt+1i​(xti)2]+−λt+1i​wtig\_{t+1}^{i}\leftarrow g\_{t}^{i}\big[1-\lambda\_{t+1}^{i}-\alpha\_{t+1}^{i}(x\_{t}^{i})^{2}\big]^{+}-\lambda\_{t+1}^{i}\,w\_{t}^{i}

13:   # Update weight

14:   wt+1i←(1−λt+1i)​wti+αt+1i​δt​xtiw\_{t+1}^{i}\leftarrow(1-\lambda\_{t+1}^{i})\,w\_{t}^{i}+\alpha\_{t+1}^{i}\,\delta\_{t}\,x\_{t}^{i}

15:  end for

16: end for

[◄](/html/2604.27062)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2604.27063)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2604.27063)
[View original  
on arXiv](https://arxiv.org/abs/2604.27063)[►](/html/2604.27064)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Tue May 5 22:29:46 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
