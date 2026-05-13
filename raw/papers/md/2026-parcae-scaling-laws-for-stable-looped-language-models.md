---
arxiv: '2604.12946'
authors:
- Hayden Prairie
- Zachary Novack
- Taylor Berg-Kirkpatrick
- Daniel Y. Fu
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'Parcae: Scaling Laws For Stable Looped Language Models'
url: https://arxiv.org/abs/2604.12946
year: 2026
---

[2604.12946] Parcae: Scaling Laws For Stable Looped Language Models















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



# Parcae: Scaling Laws For Stable Looped Language Models

Hayden Prairie
Affiliation: University of California, San Diego
Affiliation: Together AI
  
Zachary Novack
Affiliation: University of California, San Diego
  
Taylor Berg-Kirkpatrick
Affiliation: University of California, San Diego
  
Daniel Y. Fu
Affiliation: University of California, San Diego
Affiliation: Together AI

###### Abstract

Traditional fixed-depth architectures scale quality by increasing training FLOPs, typically through increased parameterization, at the expense of a higher memory footprint, or data.
A potential alternative is *looped architectures*, which instead increase FLOPs by sending activations through a block of layers in a loop.
While promising, existing recipes for training looped architectures can be unstable, suffering from residual explosion and loss spikes.
We address these challenges by recasting looping as a nonlinear time-variant dynamical system over the residual stream. Via a linear approximation to this system, we find that instability occurs in existing looped architectures as a result of large spectral norms in their injection parameters.
To address these instability issues, we propose *Parcae*, a novel *stable*, looped architecture that constrains the spectral norm of the injection parameters via discretization of a negative diagonal parameterization. As a result, Parcae achieves up to 6.3% lower validation perplexity over prior large-scale looped models.
Using our stable looped architecture, we investigate the scaling properties of looping as a medium to improve quality by increasing FLOPs in training and test-time.
For training, we derive predictable power laws to scale FLOPs while keeping parameter count fixed. Our initial scaling laws suggest that looping and data should be increased in tandem, given a fixed FLOP budget.
At test-time, we find that Parcae can use looping to scale compute, following a predictable, saturating exponential decay.
When scaled up to 1.3B parameters, we find that Parcae improves CORE and Core-Extended quality by 2.99 and 1.18 points when compared to strong Transformer baselines under a fixed parameter and data budget, achieving a relative quality of up to 87.5% a Transformer twice the size.

{hprairie,znovack,tberg,danfu}@ucsd.edu

## 1 Introduction

Scaling laws have established that model performance improves predictably with increased FLOPs [kaplan2020scalinglawsneurallanguage, hoffmann2022trainingcomputeoptimallargelanguage], typically by increasing parameter count or training data.
These scaling laws suggest that FLOP-optimal training increases parameters and training data in tandem following empirical power laws.
As a result, the depth and width of state-of-the-art models have grown in an effort to scale with data, subsequently inflating the memory footprint to deploy these models [dettmers2023case4bitprecisionkbit, lin2024awqactivationawareweightquantization].

However, as inference deployments take on an increasingly large portion of compute [touvron2023llamaopenefficientfoundation], and deployments begin to move to the edge [moon2024lpulatencyoptimizedhighlyscalable, narayan2025minionscostefficientcollaborationondevice], there is increasing interest in scaling model quality without increasing parameters.
One mechanism to do this is layer-looped models, such as looped transformers [dehghaniUniversalTransformers2019, geiping\_scaling\_2025, zhuScalingLatentReasoning2025], which iteratively loop activations through a block of layers.
Initial results have been encouraging, with looped models matching the quality of larger fixed-depth architectures [geiping\_scaling\_2025, zhuScalingLatentReasoning2025]. Moreover, they show potential for latent reasoning [avi\_learn\_algorithm, yangLoopedTransformersAre2023] and per-token adaptive compute [geiping\_scaling\_2025, mcleish\_retrofitted\_recurrence].

![Refer to caption](/html/2604.12946/assets/x1.png)


Figure 1: Parcae and the Scaling Laws of Looping.
(*Left*) Parcae constrains the spectral norm of 𝑨¯\overline{\bm{A}} and normalizes the input injection, stabilizing the residual stream hth\_{t} across loops. (*Right*) We observe looping to be an orthogonal axis of scaling compute which follows a power law.

Unfortunately, prior research [geiping\_scaling\_2025, mcleish\_retrofitted\_recurrence, LoopFormerElasticDepthLooped2025] and our work observe these models’ training to be unstable, exhibiting residual state explosion and loss spikes.
Since these models loop the layers of complex non-linear architectures (e.g., transformer blocks [vaswani2023attentionneed]), the source of instability in looped models can be difficult to understand analytically.
As a result, training requires sensitive hyperparameter selection and residual normalization (e.g., Post-Norm) to correct this instability [geiping\_scaling\_2025].
Furthermore, even in convergent training runs, we observe loss spikes as looped models train on stochastic amounts of depth to induce stronger test-time scaling [anilPathIndependentEquilibrium].
In this paper, we study this instability and ask whether stabilizing these models can unlock looping as a predictable, orthogonal axis for scaling compute.

To analyze instability, we observe that prior looped architectures can be recast as a nonlinear time-variant dynamical system over the residual stream [olsson2022incontextlearninginductionheads], taking the form:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ht+1=𝑨¯​ht+𝑩¯​e+ℛ¯​(ht,e),h\_{t+1}=\overline{\bm{A}}h\_{t}+\overline{\bm{B}}e+\overline{\mathcal{R}}(h\_{t},e), |  | (1) |

where for an input ee, the hidden state hh across the depth of an architecture is modulated by 𝑨¯\overline{\bm{A}}, controlling the balance between prior and current residual states; 𝑩¯\overline{\bm{B}}, conditioning the residual on the input ee; and a non-linear operator ℛ¯\overline{\mathcal{R}}, which subsumes the original transformer modules (e.g., Attention, MLPs).
By linearizing this framework (e.g., removing ℛ¯\overline{\mathcal{R}}), we observe that [Equation 1](#S1.E1 "In 1 Introduction ‣ Parcae: Scaling Laws For Stable Looped Language Models") resolves to a linear time invariant (LTI) system from which classic control theory can be used to infer divergence conditions on the residual stream based on the spectral norm of 𝑨¯\overline{\bm{A}}.
We observe that prior looped architectures can learn unstable parameterizations of 𝑨¯\overline{\bm{A}}, which we empirically find to induce residual stream explosion (see [Table 2](#S3.T2 "In Dynamical System over Residual Stream. ‣ 3 Understanding Instability in Looped Architectures ‣ Parcae: Scaling Laws For Stable Looped Language Models")).

To address these issues, we propose *Parcae*, a novel looped transformer that corrects the parameter instability conditions of [Equation 1](#S1.E1 "In 1 Introduction ‣ Parcae: Scaling Laws For Stable Looped Language Models") and uses algorithmic fixes to reduce loss spikes during training. *Parcae* explicitly uses discretization on a continuous representation 𝑨\bm{A} of [Equation 1](#S1.E1 "In 1 Introduction ‣ Parcae: Scaling Laws For Stable Looped Language Models") and parametrizes 𝑨\bm{A} as a negative diagonal matrix, constraining the spectral norm to prevent residual explosion in looped layers.
Additionally, Parcae introduces a normalization on ee, which empirically prevents loss spikes in late stages of training. Finally, Parcae modifies the training algorithm (which aims to minimize the expected loss over variable depths) by enabling intra-batch per-sequence depth sampling to further reduce loss spikes.

We evaluate Parcae on end-to-end quality, training FLOP scaling, and test-time scaling:

* •

  End-to-End Quality. We compare Parcae against parameter- and data-matched RDMs [geiping\_scaling\_2025] and Transformers. Against RDMs, Parcae reduces val. PPL by 6.3%. When scaled up to 1.3B parameters and 100B tokens, Parcae outperforms parameter-matched Transformers by up to 2.99 and 1.18 points on Core and Core-Extended [li2025datacomplmsearchgenerationtraining] benchmarks, respectively — matching Transformers up to twice the size.
* •

  Training FLOP Scaling. To evaluate FLOP training scaling, we study scaling laws for looping in a parameter-matched isoFLOP setting (i.e., whether to scale FLOPs with increased data or looping).
  We find that looping introduces an orthogonal scaling axis, similar to parameters and data.
  Specifically, FLOP-optimal training increases looping and data following empirical power laws (see [Figure 1](#S1.F1 "In 1 Introduction ‣ Parcae: Scaling Laws For Stable Looped Language Models") [*right*]).
* •

  Test-Time Scaling. We study looping as a mechanism to scale test-time compute, observing that recurrence follows predictable exponential decay with an irreducible loss. We further combine both test-time and training power laws to create a single unifying scaling law for looping in Parcae models.

## 2 Background

We first provide a brief background on looped models ([Section 2.1](#S2.SS1 "2.1 Existing Middle-Looped Architectures ‣ 2 Background ‣ Parcae: Scaling Laws For Stable Looped Language Models")), LTI systems ([Section 2.2](#S2.SS2 "2.2 Linear Time-Invariant Dynamical Systems ‣ 2 Background ‣ Parcae: Scaling Laws For Stable Looped Language Models")), and modeling scaling laws ([Section 2.3](#S2.SS3 "2.3 Modeling Scaling Laws ‣ 2 Background ‣ Parcae: Scaling Laws For Stable Looped Language Models")).
Prior work has studied looped architectures along several design axes: loop placement (pre-, mid-, or post-looping) [saunshiReasoningLatentThoughts2025b], halting mechanism (explicit
routers [baeMixtureofRecursionsLearningDynamic2025, zhuScalingLatentReasoning2025] vs. implicit stochastic depth [geiping\_scaling\_2025, mcleish\_retrofitted\_recurrence]), topology (single
block [geiping\_scaling\_2025] or hierarchical [wangHierarchicalReasoningModel2025b, jolicoeur-martineauLessMoreRecursive2025]) and differentiation (explicit or implicit backpropagation [bai2019deepequilibriummodels]). Our work focuses on implicit-halting middle-looped architectures using explicit differentiation; an extended review is in [Appendix B](#A2 "Appendix B Extended Literature Review ‣ Parcae: Scaling Laws For Stable Looped Language Models").

### 2.1 Existing Middle-Looped Architectures

In this paper, we focus on middle-looped architectures [saunshiReasoningLatentThoughts2025b, geiping\_scaling\_2025].
Middle-looped recurrent depth architecture contains three units: an initial prelude unit 𝒫\mathcal{P}, a middle recurrent unit ℛ\mathcal{R}, and a final coda unit 𝒞\mathcal{C}. Formally, given an input s∈Vns\in V^{n}, where VV is vocabulary and nn is sequence dimension, the outputs p∈ℝn×|V|p\in\mathbb{R}^{n\times|V|} can be computed by the following update rule:
e=𝒫​(s),ht+1=ℛ​(ht,e),p=𝒞​(hT),e=\mathcal{P}(s),~h\_{t+1}=\mathcal{R}(h\_{t},e),~p=\mathcal{C}(h\_{T}),
where h0∼𝒩​(0,σ2​Id×d)h\_{0}\sim\mathcal{N}(0,\sigma^{2}I\_{d\times d}) and dd the embedding dimension.
Intuitively, 𝒫\mathcal{P} embeds inputs into the latent space, conditioning ℛ\mathcal{R} as it recursively updates the hidden state ht∈ℝn×dh\_{t}\in\mathbb{R}^{n\times d} for TT iterations,
which 𝒞\mathcal{C} uses to generate pp. Within ℛ\mathcal{R}, prior work inject ee using addition ht+1=ℛ​(ht+e)h\_{t+1}=\mathcal{R}(h\_{t}+e) [yangLoopedTransformersAre2023] or concatenation with projection ht+1=ℛ​(W​[ht;e])h\_{t+1}=\mathcal{R}(W[h\_{t};e]) [geiping\_scaling\_2025], where W∈ℝd×2​dW\in\mathbb{R}^{d\times 2d}.

While looped models can be viewed as weight-sharing layers, modern variants allow for variable depth.
During training, depth TT is sampled per micro-batch [bansalEndtoendAlgorithmSynthesis2022] from Λ\Lambda (e.g., Poisson with mean μrec\mu\_{\text{rec}}), exposing the model to variable depths for stronger test-time scaling [anilPathIndependentEquilibrium].
The training objective thus minimizes the expectation over the dataset and Λ\Lambda.
Lastly, truncated backpropagation through depth, analogous to BPTT [Hinton2013TrainingRN], limits the backward pass to a constant μbwd\mu\_{\text{bwd}} [geiping\_scaling\_2025].

#### Stability.

geiping\_scaling\_2025 found looped models unstable at scale and adopted a block pattern, combining Pre- and Post-Norm to normalize the residual: x¯(ℓ)=LN​(MHA​(LN​(x(ℓ−1)))+x(ℓ−1)),x(ℓ)=LN​(FFN​(LN​(x¯(ℓ)))+x¯(ℓ))\bar{x}^{(\ell)}=\text{LN}(\text{MHA}(\text{LN}(x^{(\ell-1)}))+x^{(\ell-1)}),\quad x^{(\ell)}=\text{LN}(\text{FFN}(\text{LN}(\bar{x}^{(\ell)}))+\bar{x}^{(\ell)})
where LN​(⋅)\mathrm{LN}(\cdot) denotes layer normalization, MHA​(⋅)\mathrm{MHA}(\cdot) multi-head attention, and FFN​(⋅)\mathrm{FFN}(\cdot) feed-forward networks. We later show that residual normalization is unnecessary when stability is properly controlled.

### 2.2 Linear Time-Invariant Dynamical Systems

To study the instability of looped models, we will use an LTI dynamical system as a tractable linear surrogate for complex non-linear looped models.
In control theory, LTI systems are formalized through first-order differential equations
h˙​(t)=𝑨​h​(t)+𝑩​e​(t),y​(t)=𝑪​h​(t)\dot{h}(t)=\bm{A}h(t)+\bm{B}e(t),~y(t)=\bm{C}h(t)
that describe the evolution of a hidden state h​(t)∈ℝdhh(t)\in\mathbb{R}^{d\_{h}} given an input signal e​(t)∈ℝdee(t)\in\mathbb{R}^{d\_{e}},
where 𝑨∈ℝdh×dh\bm{A}\in\mathbb{R}^{d\_{h}\times d\_{h}} governs the dynamics of the system, 𝑩∈ℝdh×de\bm{B}\in\mathbb{R}^{d\_{h}\times d\_{e}} controls how external inputs influence the state, and 𝑪∈ℝde×dh\bm{C}\in\mathbb{R}^{d\_{e}\times d\_{h}} projects the hidden state to the output y​(t)∈ℝdey(t)\in\mathbb{R}^{d\_{e}}. The continuous system can be discretized to obtain
ht=𝑨¯​ht−1+𝑩¯​et,yt=𝑪​hth\_{t}=\overline{\bm{A}}h\_{t-1}+\overline{\bm{B}}e\_{t},y\_{t}=\bm{C}h\_{t}
using a step size Δ\Delta; for instance, zero-order hold (ZOH) would yield 𝑨¯=exp⁡(Δ​𝑨)\overline{\bm{A}}=\exp(\Delta\bm{A}) and 𝑩¯=(Δ​𝑨)−1​(exp⁡(Δ​𝑨)−I)⋅Δ​𝑩\overline{\bm{B}}=(\Delta\bm{A})^{-1}(\exp(\Delta\bm{A})-I)\cdot\Delta\bm{B}.

LTI systems fall into three regimes: *stable* (bounded and convergent), *marginally stable* (oscillatory), and *unstable* (explosive and divergent).
A fundamental property of LTI systems is that their *stability* is determined by the eigenvalues of 𝑨\bm{A}.
Continuous LTI systems require negative eigenvalues of 𝑨\bm{A}; Discrete LTI systems requires ρ​(𝑨¯)<1\rho(\overline{\bm{A}})<1 [1082819], where ρ\rho computes the spectral norm, with unstable systems having ρ​(𝑨¯)>1\rho(\overline{\bm{A}})>1.

### 2.3 Modeling Scaling Laws

We follow hoffmann2022trainingcomputeoptimallargelanguage, which modeled scaling law behaviors via parabolic and parametric fits for varying model sizes and training tokens with a fixed FLOP budget.
For parabolic fits, a quadratic is fit to several FLOP budgets to estimate the loss-optimal model size or number of training tokens. For parametric fits, a function form of ℒ^​(N,D)=E+X⋅N−x+Y⋅D−y\widehat{\mathcal{L}}(N,D)=E+X\cdot N^{-x}+Y\cdot D^{-y} is fit using the Huber loss [huber] between the predicted and empirical log loss values for varying parameters NN and tokens DD, using L-BFGS [lbfgs] to minimize.

## 3 Understanding Instability in Looped Architectures

![Refer to caption](/html/2604.12946/assets/x2.png)


Figure 2: Training Instability of Looped Architectures. (*left*) Pre-Norm looped models diverge, while residual norm. and Parcae converge. (*right*) Instability stems from an exploding recurrent state norm ‖hT‖2||h\_{T}||\_{2}, the hidden embedding norm after TT recurrences.

In this section, we study the instability of looped architectures. Using an LTI view over the residual, we find that instability stems from an unconstrained residual state explosion ([Figure 2](#S3.F2 "In 3 Understanding Instability in Looped Architectures ‣ Parcae: Scaling Laws For Stable Looped Language Models"); [Table 2](#S3.T2 "In Dynamical System over Residual Stream. ‣ 3 Understanding Instability in Looped Architectures ‣ Parcae: Scaling Laws For Stable Looped Language Models") [*Baseline*]; [Appendix F](#A6 "Appendix F Additional Stability Ablations ‣ Parcae: Scaling Laws For Stable Looped Language Models")).
While residual normalization helps mitigate this issue, it requires sensitive hyperparameter tuning ([Table 2](#S3.T2 "In Dynamical System over Residual Stream. ‣ 3 Understanding Instability in Looped Architectures ‣ Parcae: Scaling Laws For Stable Looped Language Models") [*Res. Norm*]), similar to fixed-depth transformers [xu2019understandingimprovinglayernormalization, xiongLayerNormalizationTransformer2020].
Using this LTI framework, we derive stability conditions for the eigenvalues of 𝑨¯\overline{\bm{A}}. We find that prior work does not satisfy these conditions for 𝑨¯\overline{\bm{A}}, which we empirically verify creates major state explosion ([Table 2](#S3.T2 "In Dynamical System over Residual Stream. ‣ 3 Understanding Instability in Looped Architectures ‣ Parcae: Scaling Laws For Stable Looped Language Models")).

#### Dynamical System over Residual Stream.

Our key insight is to recast the forward pass as a dynamical system over the residual stream. Consider a transformer-based looped model as defined in [Section 2.1](#S2.SS1 "2.1 Existing Middle-Looped Architectures ‣ 2 Background ‣ Parcae: Scaling Laws For Stable Looped Language Models") for language modeling, where 𝒫\mathcal{P} is an embedding layer that maps a sequence of tokens s∈Vns\in V^{n} into embedding space e∈ℝn×dhe\in\mathbb{R}^{n\times d\_{h}}, 𝒞\mathcal{C} is a projection head that maps into probability space g:dh→|V|g:d\_{h}\to|V|, and ℛ\mathcal{R} is parameterized with LL transformer blocks. While several methods of input injection could condition ℛ\mathcal{R} on ee, building on prior work [yang2024loopedtransformersbetterlearning, geiping\_scaling\_2025, mcleish\_retrofitted\_recurrence], we focus on linear methods of injection (e.g., ℛ​(ht,e)=ℛ​(W1​ht+W2​e)\mathcal{R}(h\_{t},e)=\mathcal{R}(W\_{1}h\_{t}+W\_{2}e), where W1∈ℝdh×dhW\_{1}\in\mathbb{R}^{d\_{h}\times d\_{h}} and W2∈ℝdh×deW\_{2}\in\mathbb{R}^{d\_{h}\times d\_{e}}).111Both addition [yang2024loopedtransformersbetterlearning] and concatenation [geiping\_scaling\_2025] fall under this framework.

Recall that ℛ\mathcal{R} denotes the full recurrent update ht+1=ℛ​(ht,e)h\_{t+1}=\mathcal{R}(h\_{t},e), encompassing all transformer operations, including residual connections.
The recurrent update can be exactly formulated as a non-linear time-variant dynamical system of the form ht=𝑨¯​ht−1+𝑩¯​e+ℛ¯​(ht−1,e),yt=𝑪​ht,h\_{t}=\overline{\bm{A}}h\_{t-1}+\overline{\bm{B}}e+\overline{\mathcal{R}}(h\_{t-1},e),~y\_{t}=\bm{C}h\_{t},
where 𝑪∈Rdc×dh\bm{C}\in R^{d\_{c}\times d\_{h}} decouples the 𝒞\mathcal{C} and ℛ\mathcal{R} embedding dimension (i.e. p=𝒞​(𝑪​(hT))p=\mathcal{C}(\bm{C}(h\_{T}))).
This derivation is shown in [Appendix C](#A3 "Appendix C Derivation of Instability Conditions of Prior Methods ‣ Parcae: Scaling Laws For Stable Looped Language Models"). Though this formulation does not immediately elucidate instability,
linearizing of this system (i.e., dropping ℛ¯\overline{\mathcal{R}}) yields a discrete LTI system of the form:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ht+1=𝑨¯​ht+𝑩¯​eh\_{t+1}=\overline{\bm{A}}h\_{t}+\overline{\bm{B}}e |  | (2) |

| Method | 𝑨¯\overline{\bm{A}} | 𝑩¯\overline{\bm{B}} | ρ​(𝑨¯)\rho(\overline{\bm{A}}) | LTI Stability |
| --- | --- | --- | --- | --- |
| Addition | II | II | ρ​(𝑨¯)=1\rho(\overline{\bm{A}})=1 | *marginally-stable* |
| Concatenation | ℝdh×dh\mathbb{R}^{d\_{h}\times d\_{h}} | ℝdh×de\mathbb{R}^{d\_{h}\times d\_{e}} | ρ​(𝑨¯)∈ℝ\rho(\overline{\bm{A}})\in\mathbb{R} | *unstable* |
| Parcae (ours) | ZOH(Diag(−exp(ℝdh))\text{ZOH}(\texttt{Diag}(-\exp(\mathbb{R}^{d\_{h}})) | Euler​(ℝdh×de)\text{Euler}(\mathbb{R}^{d\_{h}\times d\_{e}}) | ρ​(𝑨¯)<1\rho(\overline{\bm{A}})<1 | *stable* |

Table 1: Comparison of Prior Update Rule Stability based on LTI Representation.



| LR | Base | Res. Norm | Parcae |
| --- | --- | --- | --- |
| 2e-4 | ✓ | ✓ | ✓ |
| 4e-4 | ✗ | ✓ | ✓ |
| 6e-4 | ✗ | ✗ | ✓ |
| 8e-4 | ✗ | ✗ | ✓ |
| 1e-3 | ✗ | ✗ | ✓ |

Table 2: Hyperparameter Instability. Convergence across learning rates for baseline RDMs, Res. Norm RDMs, and Parcae. Parcae is more robust to hyperparameter selection. Full logs are in [Appendix F](#A6 "Appendix F Additional Stability Ablations ‣ Parcae: Scaling Laws For Stable Looped Language Models").

![Refer to caption](/html/2604.12946/assets/x3.png)


Figure 3: Spectral Radius of Unconstrained A¯\overline{\bm{A}}. For a Pre-Norm RDM, we plot the ρ​(𝑨¯)\rho(\overline{\bm{A}}) throughout training using different learning rates, observing divergent runs learn ρ​(𝑨¯)>1\rho(\overline{\bm{A}})>1. The state explosion, in [Figure 2](#S3.F2 "In 3 Understanding Instability in Looped Architectures ‣ Parcae: Scaling Laws For Stable Looped Language Models") is thus directly linked to 𝑨¯\overline{\bm{A}}.

#### State Explosion from Unconstrained 𝑨¯\overline{\bm{A}} and 𝑩¯\overline{\bm{B}}.

Analyzing the stability of [Equation 2](#S3.E2 "In Dynamical System over Residual Stream. ‣ 3 Understanding Instability in Looped Architectures ‣ Parcae: Scaling Laws For Stable Looped Language Models") identifies ρ​(𝑨¯)\rho(\overline{\bm{A}}) as a critical factor governing instability.
As shown in [Table 1](#S3.T1 "In Dynamical System over Residual Stream. ‣ 3 Understanding Instability in Looped Architectures ‣ Parcae: Scaling Laws For Stable Looped Language Models"), prior work [geiping\_scaling\_2025, yang2024loopedtransformersbetterlearning] chooses parameterizations of 𝑨¯\overline{\bm{A}} such that ρ​(𝑨¯)=1\rho(\overline{\bm{A}})=1 or ρ​(𝑨¯)\rho(\overline{\bm{A}}) is unconstrained. Critically, these are *marginally-stable* or *unstable parameterizations*.

[Table 2](#S3.T2 "In Dynamical System over Residual Stream. ‣ 3 Understanding Instability in Looped Architectures ‣ Parcae: Scaling Laws For Stable Looped Language Models") and [Table 2](#S3.T2 "In Dynamical System over Residual Stream. ‣ 3 Understanding Instability in Looped Architectures ‣ Parcae: Scaling Laws For Stable Looped Language Models") confirm this empirically:
divergent runs learn a spectral radius of ρ​(𝑨¯)≥1\rho(\overline{\bm{A}})\geq 1, with convergent runs maintaining ρ​(𝑨¯)<1\rho(\overline{\bm{A}})<1, affirming that LTI stability constraints are necessary.
Finally, at scale, we observe loss spikes late in training
(e.g., after 170k steps), which we address by normalizing the input to 𝑩¯\overline{\bm{B}} (see [Appendix J](#A10 "Appendix J Ablation of Prelude Normalization ‣ Parcae: Scaling Laws For Stable Looped Language Models") for ablation).

## 4 Parcae: A Stable Looped Architecture

Using our dynamical systems framework, we create *Parcae*, a looped architecture that explicitly satisfies the stability constraints ([Section 4.1](#S4.SS1 "4.1 Block Design and Stable Parameterization of Parcae ‣ 4 Parcae: A Stable Looped Architecture ‣ Parcae: Scaling Laws For Stable Looped Language Models")).
Additionally, we propose a per-sequence depth sampling method to stabilize variance introduced by variable depth ([Section 4.2](#S4.SS2 "4.2 Stable Training Algorithms for Parcae ‣ 4 Parcae: A Stable Looped Architecture ‣ Parcae: Scaling Laws For Stable Looped Language Models")).

### 4.1 Block Design and Stable Parameterization of Parcae

We parameterize 𝑨\bm{A} and 𝑩\bm{B} in continuous form, and discretize using a learned Δ∈ℝdh\Delta\in\mathbb{R}^{d\_{h}}with ZOH and Euler schemes (i.e., 𝑨¯=exp⁡(Δ​𝑨)\overline{\bm{A}}=\exp(\Delta\bm{A}) and 𝑩¯=Δ​𝑩\overline{\bm{B}}=\Delta\bm{B}),222With abuse of notation, we let Δ​𝑨=Δ⊙𝑨\Delta\bm{A}=\Delta\odot\bm{A} (i.e., elementwise multiplication). following prior sequence modeling work [gu2024mambalineartimesequencemodeling, dao2024transformersssmsgeneralizedmodels].
To achieve our target stability conditions by constraining the eigenvalues of 𝑨\bm{A} to be negative, we parameterize 𝑨:=Diag​(−exp⁡(log\_A))\bm{A}:=\texttt{Diag}(-\exp(\texttt{log\\_A})) as a negative diagonal matrix, where Diag​(−exp⁡(⋅))\texttt{Diag}(-\exp(\cdot)) of a vector enforces negativity and log\_A∈ℝdh\texttt{log\\_A}\in\mathbb{R}^{d\_{h}} is our learnable vector.
While many formulations of 𝑨\bm{A} would work, ensuring negative eigenvalues in the diagonal case is simple and cheap.
𝑩\bm{B} is left unconstrained; however, we introduce a normalization layer to the input ee to further stabilize training (see [Appendix J](#A10 "Appendix J Ablation of Prelude Normalization ‣ Parcae: Scaling Laws For Stable Looped Language Models") for ablation).
With this, our update rule, given an input sequence ss, becomes

|  |  |  |  |
| --- | --- | --- | --- |
|  | e=LN​(𝒫​(s)),ht+1=𝑨¯​ht+𝑩¯​e+ℛ¯​(ht,e),p=𝒞​(𝑪​hT),e=\text{LN}(\mathcal{P}(s)),\qquad h\_{t+1}=\overline{\bm{A}}h\_{t}+\overline{\bm{B}}e+\overline{\mathcal{R}}(h\_{t},e),\qquad p=\mathcal{C}(\bm{C}h\_{T}), |  | (3) |

where h0∼𝒩​(0,σ​Idh×dh)h\_{0}\sim\mathcal{N}(0,~\sigma I\_{d\_{h}\times d\_{h}}) and TT is the number of loops.

We parameterize 𝒫\mathcal{P}, ℛ¯\overline{\mathcal{R}}, and 𝒞\mathcal{C} using L𝒫,LℛL\_{\mathcal{P}},L\_{\mathcal{R}} and L𝒞L\_{\mathcal{C}} transformer bloc:ks respectively. For exact block architecture, we match two different architectural setups: one for prior RDMs [geiping\_scaling\_2025] and one for strong Transformer baselines [nanochat]. Parcae’s architecture matches RDMs, differing only in residual normalization and the dynamical systems parameters
(e.g., 𝑨,𝑩,𝑪,Δ\bm{A},\bm{B},\bm{C},\Delta). Against Transformers, we follow a simplified nanochat [nanochat] setup, where we match exact architecture, except we loop the middle third layers and include our dynamical systems parameters and a prelude norm. Exact model definitions and a forward pass can be found in [Appendix P](#A16 "Appendix P Model Definitions ‣ Parcae: Scaling Laws For Stable Looped Language Models") and [Appendix E](#A5 "Appendix E Parcae Forward Pass and Training Algorithms ‣ Parcae: Scaling Laws For Stable Looped Language Models"), respectively.

### 4.2 Stable Training Algorithms for Parcae

We further stabilize Parcae by adjusting the training objective. Specifically, looped models’ training objective is
θ⋆=arg⁡minθ⁡𝔼(x,y)∼𝒟,T∼Λ​[ℓ​(fθ​(x;T),y)]\theta^{\star}\;=\;\arg\min\_{\theta}\;\mathbb{E}\_{(x,y)\sim\mathcal{D},\,T\sim\Lambda}\!\left[\;\ell\!\big(f\_{\theta}(x;T),\,y\big)\;\right], implying that more depths should be sampled per global batch to more faithfully model the expectation over Λ\Lambda.
Thus, we introduce a per-sequence depth sampling algorithm within a micro-batch, which we empirically observe to reduce loss spikes (ablation in [Appendix G](#A7 "Appendix G Per-sequence Sampling Reduces Loss Spikes ‣ Parcae: Scaling Laws For Stable Looped Language Models")).
Additionally, unlike prior work, we parameterize Λ\Lambda based on μrec\mu\_{\text{rec}} alone, as we find that truncating based on μbwd\mu\_{\text{bwd}} significantly hurts extrapolation to both lower and higher recurrences (ablation in [Appendix H](#A8 "Appendix H Sampling of Truncated Recurrence ‣ Parcae: Scaling Laws For Stable Looped Language Models")).
Finally, we choose μbwd=⌈μrec2⌉\mu\_{\text{bwd}}=\lceil\frac{\mu\_{\text{rec}}}{2}\rceil throughout (see [Appendix I](#A9 "Appendix I Selecting 𝜇_\"rec\" and 𝜇_\"bwd\" ‣ Parcae: Scaling Laws For Stable Looped Language Models") for ablation).
A detailed training algorithm is in [Appendix E](#A5 "Appendix E Parcae Forward Pass and Training Algorithms ‣ Parcae: Scaling Laws For Stable Looped Language Models").

## 5 Results

We evaluate Parcae on end-to-end quality ([Section 5.1](#S5.SS1 "5.1 Parcae Improves End-to-End Quality ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models")),
training FLOP scaling ([Section 5.2](#S5.SS2 "5.2 Looping as an Orthogonal Scaling Axis in Training ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models")), and test-time scaling
([Section 5.3](#S5.SS3 "5.3 Test-Time Scaling Laws of Parcae ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models")). We find that Parcae outperforms both parameter-
and data-matched RDMs and Transformers, optimal looping and data
follow predictable power laws, and test-time looping follows a saturating exponential decay.

|  | Model | 𝐓\mathbf{T} | Val. | WikiText | Hellaswag | ARC-c | ARC-e | PIQA | BoolQ | SciQ | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100M | RDM | 16 | 14.23 | 63.27 | 27.16 | 17.66 | 42.38 | 59.14 | 51.35 | 72.50 | 45.03 |
| Parcae | 16 | 13.59 | 60.33 | 27.18 | 18.09 | 43.10 | 59.30 | 61.83 | 71.50 | 46.83 |
| 350M | RDM | 8 | 10.76 | 41.31 | 28.55 | 20.90 | 47.26 | 61.75 | 61.53 | 76.70 | 49.45 |
| Parcae | 8 | 10.09 | 37.53 | 29.23 | 21.08 | 48.78 | 62.08 | 60.73 | 78.80 | 50.12 |

Table 3: Zero-Shot and Perplexity Results Trained on RDM Setup. Comparison of Parcae and RDM [geiping\_scaling\_2025] on
a variety of open source benchmarks and perplexity held-out validation set and Wikitext [merity2016pointer]. Best results are bolded.



|  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Val Loss (↓\downarrow) | | | Core (↑\uparrow) | | | Core Ext (↑\uparrow) | | |
| Configuration | T=1T\!=\!1 | T=4T\!=\!4 | T=8T\!=\!8 | T=1T\!=\!1 | T=4T\!=\!4 | T=8T\!=\!8 | T=1T\!=\!1 | T=4T\!=\!4 | T=8T\!=\!8 |
| RDM | Divergent Training | | | Divergent Training | | | Divergent Training | | |
| + Constrained 𝑨¯\overline{\bm{A}} | 8.99 | 3.15 | 2.97 | −2.0±0.1-2.0\_{\pm 0.1} | 11.0±0.111.0\_{\pm 0.1} | 13.2±0.213.2\_{\pm 0.2} | 0.5±0.10.5\_{\pm 0.1} | 7.8±0.07.8\_{\pm 0.0} | 9.1±0.59.1\_{\pm 0.5} |
| + Per-Seq. Sampling | 3.38 | 3.01 | 2.98 | 7.6±0.2\mathbf{7.6\_{\pm 0.2}} | 13.4±0.213.4\_{\pm 0.2} | 14.0±0.214.0\_{\pm 0.2} | 5.9±0.4\mathbf{5.9\_{\pm 0.4}} | 9.3±0.29.3\_{\pm 0.2} | 9.9±0.2\mathbf{9.9\_{\pm 0.2}} |
| + Prelude Norm | 3.28 | 2.97 | 2.95 | 7.5±0.37.5\_{\pm 0.3} | 13.5±0.0\mathbf{13.5\_{\pm 0.0}} | 14.0±0.2\mathbf{14.0\_{\pm 0.2}} | 5.8±0.35.8\_{\pm 0.3} | 9.4±0.1\mathbf{9.4\_{\pm 0.1}} | 9.7±0.39.7\_{\pm 0.3} |

Table 4: Stability Results Trained on Transformer Setup. To illustrate stability, we retrofit a baseline 140M Transformer into a RDM and then sequentially add our stability improvements.

### 5.1 Parcae Improves End-to-End Quality

We compare Parcae against parameter- and data-matched RDMs and Transformers, finding that Parcae is more stable than prior looped models and that it outperforms both in quality.

#### Setup.

For RDMs, we follow geiping\_scaling\_2025, using the Huginn dataset and tokenizer for training. For transformers, we follow nanochat and train on FineWeb-Edu [penedo2024finewebdatasetsdecantingweb].
For both RDM and Transformer setups, we perform hyperparameter sweeps for both RDMs and Transformers, and then use them for Parcae (i.e., we perform no hyperparameter sweeps for Parcae models). Extended model definitions, hyperparameter selection, and evaluation setup can be found in [Appendix P](#A16 "Appendix P Model Definitions ‣ Parcae: Scaling Laws For Stable Looped Language Models"), [Appendix Q](#A17 "Appendix Q Hyperparameters and Training Details ‣ Parcae: Scaling Laws For Stable Looped Language Models"), and [Appendix M](#A13 "Appendix M Extended Evaluation Details and Setup ‣ Parcae: Scaling Laws For Stable Looped Language Models"), respectively.

Comparison against RDMs. [Table 3](#S5.T3 "In 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models") shows that Parcae reduces perplexity by up to 6.2 % and 9.1 % on a held-out validation set and WikiText [merity2016pointer] against prior RDMs [geiping\_scaling\_2025], while additionally performing up to 1.8 points better on the average of several downstream benchmarks. [Table 4](#S5.T4 "In 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models") ablates that each modification of Parcae contributes:
constraining 𝑨¯\overline{\bm{A}} enables convergence at high TT (e.g., μrec=T=8\mu\_{\text{rec}}=T\!=\!8),
per-sequence sampling stabilizes lower test-time depths, and the prelude norm
further improves quality across all TT (and late stage stability [Appendix J](#A10 "Appendix J Ablation of Prelude Normalization ‣ Parcae: Scaling Laws For Stable Looped Language Models")).

|  | Model | 𝐓\mathbf{T} | Val. PPL (↓\downarrow) | Lambada PPL (↓\downarrow) | Core (↑\uparrow) | Core-Extended (↑\uparrow) |
| --- | --- | --- | --- | --- | --- | --- |
| 140M | Transformer | – | 21.48 | 127.39 | 13.00 ± 0.15 | 8.80 ± 0.21 |
| Parcae | 8 | 19.06 | 80.64 | 14.04 ± 0.20 | 9.67 ± 0.28 |
| 370M | Transformer | – | 15.79 | 40.77 | 17.46 ± 0.03 | 11.71 ± 0.22 |
| Parcae | 8 | 14.49 | 32.74 | 20.00 ± 0.06 | 12.75 ± 0.31 |
| 770M | Transformer | – | 13.08 | 22.37 | 22.42 ± 0.20 | 14.20 ± 0.63 |
| Parcae | 8 | 12.49 | 19.71 | 25.07 ± 0.33 | 15.19 ± 0.43 |
| 1.3B | Transformer | – | 11.95 | 17.26 | 25.45 ± 0.08 | 15.90 ± 0.23 |
| Parcae | 8 | 11.42 | 14.71 | 28.44 ± 0.28 | 17.08 ± 0.09 |

Table 5: Comparing Parcae to Fixed-Depth Transformers. We pretrain Transformers and Parcae with a nanochat setup at several scales, evaluating on a held-out validation set, Lambada [paperno2016lambada], Core, and Core-Extended [li2025datacomplmsearchgenerationtraining].
Best results are bolded.

Comparison Against Transformers. [Table 5](#S5.T5 "In Setup. ‣ 5.1 Parcae Improves End-to-End Quality ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models") shows that Parcae reduces validation perplexity by 4.3–9.2% and improves Core and Core-Extended Scores by up to 2.99 and 1.18 points, respectively. We find that our 770M Parcae model achieves quality comparable to the 1.3B Transformer on Core [li2025datacomplmsearchgenerationtraining] with roughly half the parameters.
Measured as a fraction of the quality gap to the next larger Transformer (e.g., for 140M Core-Extended: 9.67−8.8011.71−8.80⋅100≈29.9%\frac{9.67-8.80}{11.71-8.80}\cdot 100\approx 29.9\%), Parcae achieves a *23.3-87.5% and 29.9-58.2%* better parameter efficiency for Core and Core-Extended, respectively.

![Refer to caption](/html/2604.12946/assets/x4.png)


Figure 4: Looping Scales Training Compute Optimally. (*Left*) Parametric isoLoss contours over μrec\mu\_{\text{rec}} and data. The efficient frontier (blue line) traces the lowest FLOP budget required to achieve each loss level, showing that optimal training requires increased looping. (*Right*) Parabolic isoFLOP fits for 140M and 370M models reveal a clear optimum μrec\mu\_{\text{rec}} at each FLOP budget, indicating that looping is an orthogonal scaling axis to data.

### 5.2 Looping as an Orthogonal Scaling Axis in Training

In this section, we explore the FLOP efficiency of looping under a fixed FLOP and parameter budgets. We find that looping introduces an orthogonal axis for scaling compute, where compute-optimal training increases μrec\mu\_{\text{rec}} and data in tandem following empirical power laws.

#### Setup.

We train 140M and 370M Parcae models under fixed FLOP and parameter budgets, varying training tokens and mean recursion μrec\mu\_{\text{rec}} using the nanochat setup. Additional training details and FLOP estimates can be found in [Appendix O](#A15 "Appendix O Expanded Setup For Training and Test-Time Scaling Laws ‣ Parcae: Scaling Laws For Stable Looped Language Models") and [Appendix D](#A4 "Appendix D FLOP Estimate of Parcae ‣ Parcae: Scaling Laws For Stable Looped Language Models"), respectively.

Modeling Scaling Laws of Looping. At 140M and 370M scales, isoFLOP curves show that increasing μrec\mu\_{\text{rec}} while proportionally reducing tokens yields lower validation loss than training at low recurrence ([Figure 4](#S5.F4 "In Setup. ‣ 5.1 Parcae Improves End-to-End Quality ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models") [*right*]). Using a parabolic fit, we extract the optimal μrec\mu\_{\text{rec}} and token budget at each FLOP level, finding that both follow predictable power laws ([Figure 5](#S5.F5 "In Setup. ‣ 5.2 Looping as an Orthogonal Scaling Axis in Training ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models")) with consistent exponents (γμ≈0.40\gamma\_{\mu}\approx 0.40, γD≈0.78\gamma\_{D}\approx 0.78).
We also fit a parametric function ℒ^​(μrec,D)=E+X⋅𝐍​(μrec)−x+Y⋅D−y\widehat{\mathcal{L}}(\mu\_{\text{rec}},D)=E+X\cdot\mathbf{N}(\mu\_{\text{rec}})^{-x}+Y\cdot D^{-y} over the effective parameterization 𝐍​(μrec)\mathbf{N}(\mu\_{\text{rec}}) (i.e., parameters of unrolling the looped model) and tokens DD ([Figure 4](#S5.F4 "In Setup. ‣ 5.1 Parcae Improves End-to-End Quality ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models"), [*left*]; details in [Appendix K](#A11 "Appendix K Fitting a Parametric Function for Looping ‣ Parcae: Scaling Laws For Stable Looped Language Models")), enabling predictable extrapolation of loss to unseen budgets. To verify, we predict the validation loss of held-out models in [Section 5.1](#S5.SS1 "5.1 Parcae Improves End-to-End Quality ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models"), achieving 1.3% and 0.8% error at 140M and 370M, respectively.

![Refer to caption](/html/2604.12946/assets/x5.png)


Figure 5: Optimal μrec\mu\_{\text{rec}} and Tokens Follows Predictable Power Laws. We fit a parabola to each isoFLOP budget for both 140M and 370M Parcae models, using its minima to approximate the optimal μrec\mu\_{\text{rec}} and token budget at each scale. We observe that optimal recurrence (*left plots*) and tokens (*right plots*) follow a predictable power law with similar coefficients at both scales.



![Refer to caption](/html/2604.12946/assets/x6.png)


Figure 6: Pareto Frontier of Looping. We observe that looping has a stricter IsoFLOP optimal loss frontier over fixed-depth, non-looped models. Dots are empirical points.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | FLOPs |  | Optimal μrec∗\mu\_{\mathrm{rec}}^{\*} | | Fixed-Depth | |
|  | (×1018)\times 10^{18}) | μrec∗\mu\_{\mathrm{rec}}^{\*} | Core | Core Ext. | Core | Core Ext. |
| 140M | 11 | 2 | 7.67.6 | 5.75.7 | 7.9\mathbf{7.9} | 6.1\mathbf{6.1} |
| 22 | 2 | 9.09.0 | 6.26.2 | 10.5\mathbf{10.5} | 6.4\mathbf{6.4} |
| 44 | 4 | 11.2\mathbf{11.2} | 8.4\mathbf{8.4} | 10.710.7 | 8.18.1 |
| 88 | 6 | 10.510.5 | 7.8\mathbf{7.8} | 11.8\mathbf{11.8} | 7.77.7 |
| 1616 | 8 | 14.6\mathbf{14.6} | 9.8\mathbf{9.8} | 13.013.0 | 8.88.8 |
| 6464 | 10 | 16.2\mathbf{16.2} | 11.0\mathbf{11.0} | 15.015.0 | 9.59.5 |
| 370M | 3232 | 4 | 15.215.2 | 10.110.1 | 16.8\mathbf{16.8} | 11.2\mathbf{11.2} |
| 6464 | 6 | 18.1\mathbf{18.1} | 11.611.6 | 18.1\mathbf{18.1} | 12.1\mathbf{12.1} |
| 128128 | 6 | 20.1\mathbf{20.1} | 13.0\mathbf{13.0} | 18.118.1 | 12.012.0 |

Table 6: Core Scores Comparison of Looping Optimal Frontier over Purely Scaling Data. We evaluate the downstream quality of fixed-depth (μrec\mu\_{\text{rec}} =1) and looped Parcae models trained with fixed parameters and FLOP budgets. At both scales, using the optimal μrec\mu\_{\text{rec}} results in better Core and Core-Extended scores at extended FLOP budgets. Expanded results can be found in [Appendix N](#A14 "Appendix N Expanded Results For Fixed-Depth and Looping IsoFLOP Comparison ‣ Parcae: Scaling Laws For Stable Looped Language Models").

IsoFLOP comparison of Looping with Fixed-Depth [Table 6](#S5.T6 "In Setup. ‣ 5.2 Looping as an Orthogonal Scaling Axis in Training ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models") shows fixed-depth Parcae models without looping at each FLOP budget. The optimal curve achieves a strictly lower loss, which translates to 1.2-2.0 points higher Core scores ([Table 6](#S5.T6 "In Setup. ‣ 5.2 Looping as an Orthogonal Scaling Axis in Training ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models")).

### 5.3 Test-Time Scaling Laws of Parcae

We study looping as a mechanism for scaling test-time compute. We find the test-time compute follows a predictable saturating exponential decay, which can be unified with [Section 5.2](#S5.SS2 "5.2 Looping as an Orthogonal Scaling Axis in Training ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models"), connecting both training and test-time scaling laws.

#### Setup.

We train 140M and 370M Parcae models under a fixed data budget with μrec∈{2,4,6,8,10,12}\mu\_{\text{rec}}\in\{2,4,6,8,10,12\} following our nanochat setup, evaluating up to T=24T=24. We additionally evaluate models from [Section 5.2](#S5.SS2 "5.2 Looping as an Orthogonal Scaling Axis in Training ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models") for the unified scaling laws. See [Appendix O](#A15 "Appendix O Expanded Setup For Training and Test-Time Scaling Laws ‣ Parcae: Scaling Laws For Stable Looped Language Models") for details.

Saturation of Test-Time Compute. While prior works observed test-time generalization in small synthetic tasks [yangLoopedTransformersAre2023, bansalEndtoendAlgorithmSynthesis2022], we find quality to be bounded in large-scale language modeling.
Evaluating models from [Section 5.1](#S5.SS1 "5.1 Parcae Improves End-to-End Quality ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models") at 2×2\times μrec\mu\_{\text{rec}} across all four scales ([Figure 7](#S5.F7 "In Setup. ‣ 5.3 Test-Time Scaling Laws of Parcae ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models")), we observe that gains plateau near μrec\mu\_{\text{rec}}, suggesting training depth determines the test-time scaling ceiling.

![Refer to caption](/html/2604.12946/assets/x7.png)


Figure 7: Test-Time Scaling of Parcae. When evaluating Parcae models from [Table 5](#S5.T5 "In Setup. ‣ 5.1 Parcae Improves End-to-End Quality ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models"), we observe test-time looping follows a predictable saturating trend, consistent across model sizes.

![Refer to caption](/html/2604.12946/assets/x8.png)


Figure 8: Scaling Test-Time Compute follows a Predictable Power Laws. We plot the validation loss with different μrec\mu\_{\text{rec}} as a function of test-time recurrence TT, and find the fitted exponential decay (solid curve for each μrec\mu\_{\text{rec}}) tightly captures the test-time performance of looping.

Modeling Scaling Laws of Test-Time Looping. We find that the test-time scaling curves are well-described by a saturating exponential decay of the form: ℒ​(T)=ℒ∞+Z​e−z⋅T\mathcal{L}(T)=\mathcal{L}\_{\infty}+Ze^{-z\cdot T}. This form tightly captures the saturation dynamics for each model ([Figure 8](#S5.F8 "In Setup. ‣ 5.3 Test-Time Scaling Laws of Parcae ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models"); see [Appendix L](#A12 "Appendix L Fitting Parametric Functions to Test-Time Looping ‣ Parcae: Scaling Laws For Stable Looped Language Models") for details), achieving an average Huber loss of 2.5×10−72.5\times 10^{-7} and 1.8×10−71.8\times 10^{-7} for 140M and 370M, respectively.

Unifying Training and Test-Time Scaling Laws. From the learned fits in [Figure 8](#S5.F8 "In Setup. ‣ 5.3 Test-Time Scaling Laws of Parcae ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models"), we observe that ℒ∞\mathcal{L}\_{\infty} matches the training law prediction at T=μrecT=\mu\_{\text{rec}} ([Section 5.2](#S5.SS2 "5.2 Looping as an Orthogonal Scaling Axis in Training ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models")), and that the per-curve decay rate scales inversely with training depth as z/μrecz/\mu\_{\text{rec}} (see [Appendix L](#A12 "Appendix L Fitting Parametric Functions to Test-Time Looping ‣ Parcae: Scaling Laws For Stable Looped Language Models") for details).
These observations motivate a unified scaling law that connects training and test-time compute:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ^unified​(T∣μrec,D)=E+X⋅𝐍​(μrec)−x+Y⋅D−y⏟Training Law Floor ​ℒ^train​(μrec,D)+Z⋅exp⁡(−z⋅T⋅μrec−1)⏟Test-Time Decay\widehat{\mathcal{L}}\_{\text{unified}}(T\mid\mu\_{\text{rec}},D)=\underbrace{E+X\cdot\mathbf{N}(\mu\_{\text{rec}})^{-x}+Y\cdot D^{-y}}\_{\text{Training Law Floor }\widehat{\mathcal{L}}\_{\text{train}}(\mu\_{\text{rec}},D)}+\underbrace{Z\cdot\exp\!\left(-z\cdot T\cdot\mu\_{\text{rec}}^{-1}\right)}\_{\text{Test-Time Decay}} |  | (4) |

where ℒ^train​(μrec,D)\widehat{\mathcal{L}}\_{\text{train}}(\mu\_{\text{rec}},{D}) is the training law in [Section 5.2](#S5.SS2 "5.2 Looping as an Orthogonal Scaling Axis in Training ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models"), and (Z,z)(Z,z) are two fitted parameters governing the test-time scaling.
The training law sets the irreducible floor, while the decay rate −z⋅T/μrec-z\cdot T/\mu\_{\text{rec}} captures how quickly additional recurrences approach it.
On held-out 140M and 370M Parcae models ([Section 5.1](#S5.SS1 "5.1 Parcae Improves End-to-End Quality ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models")), the unified fit predicts test-time loss within 0.85-1.31% average error, dropping further to 0.1-0.17% average error when the empirical loss at T=μrecT=\mu\_{\text{rec}} is used. This confirms that [Equation 4](#S5.E4 "In Setup. ‣ 5.3 Test-Time Scaling Laws of Parcae ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models") captures saturation dynamics, with residual error attributable to the training law’s ∼1%\sim 1\% extrapolation gap (see [Appendix L](#A12 "Appendix L Fitting Parametric Functions to Test-Time Looping ‣ Parcae: Scaling Laws For Stable Looped Language Models") for extended details).

## 6 Discussion and Future Work

In this section, we briefly discuss limitations and future directions.

#### Looped Architectures.

While several design choices around looped architectures have been guided by small-scale empirical results, a deep investigation of loop-unit placement [jacobs2026blockrecurrentdynamicsvisiontransformers], composition (e.g., number of parameters in the recurrent unit and usage of different architectures), and extreme looping (e.g., increasing mean recurrence to deeper depths) at a larger scale is warranted. Within our dynamical systems framework, the use of different discretizations, full-rank parameterizations, and recurrent update rules warrants investigation to enable recurrence at larger depths.

#### Scaling.

While we find Parcae to induce predictable, optimal scaling laws for layer looping, our observations are limited to small architectures. It remains to be seen if Parcae compares favorably when scaling these observations to large FLOP budgets and parameterizations. We are also interested in the interplay of parameters, data, and recurrence as orthogonal axes, and how they should be efficiently scaled together. Finally, one limitation of looping is that, as μrec\mu\_{\text{rec}} increases, the number of test-time steps required to achieve equivalent quality increases. An investigation of techniques that maintain quality with fewer inference time steps is an interesting future direction.

## 7 Conclusion

In this work, we study the stability of looped models through a dynamical systems framework and propose Parcae, a stable looped architecture that prevents residual explosion by constraining the spectral norm of the injection parameters.
Parcae outperforms data- and parameter-matched prior looped models and baseline Transformers, matching downstream quality of models up to twice its size.
We further establish scaling laws for looping: FLOP-optimal training increases looping and data in tandem following predictable power laws, while test-time looping follows a saturating exponential decay law, yielding a unified scaling law connecting training and inference compute.

## Appendix A Glossary

We include a brief glossary of both notations and common metrics used to define and analyze looped architectures.

### A.1 Notation

|  |  |  |
| --- | --- | --- |
| Notation | Description |  |
| dd | Embedding dimension of the model |  |
| tt | Discrete temporal state axis of ℛ\mathcal{R} on ℕ\mathbb{N} |  |
| bb | Global batch size used during pretraining |  |
| 𝒫\mathcal{P} | Initial prelude block of a recurrent architecture |  |
| ℛ\mathcal{R} | Middle recurrent block of a recurrent architecture |  |
| 𝒞\mathcal{C} | Final coda block of a recurrent architecture |  |
| 𝑨\bm{A} | The linear continuous state transition matrix |  |
| 𝑩\bm{B} | The linear continuous state injection matrix |  |
| 𝑪\bm{C} | The linear state output matrix |  |
| Δ\Delta | Learnable discrete parameter for decay, discretizing our model |  |
| ss | Input sequence to a model |  |
| ee | Output embedding of the prelude block 𝒫\mathcal{P} |  |
| hh | Hidden embedding of the recurrent block ℛ\mathcal{R} |  |
| μrec\mu\_{\text{rec}} | Mean recurrent forward propagation steps during pre-training |  |
| μbwd\mu\_{\text{bwd}} | Mean recurrent backward propagation steps during pre-training |  |
| nn | Sampled number of recurrent steps with no gradient updates |  |
| kk | Sampled number of recurrent steps with gradient updates |  |
| TT | Sampled or fixed number of recurrent steps actually taken |  |
| Λ\Lambda | Distribution that recurrences are sampled from during training |  |

Table 7: Glossary of notation and terminology. (Top) Frequently used dimensions for tensors. (Middle) Definition of Parcae blocks. (Bottom) Tensors and distributions are used to express recurrent depth models.

### A.2 Common Metrics

* •

  Recurrent Residual Metric: ‖hT−hT−1‖2||h\_{T}-h\_{T-1}||\_{2}, where T∼ΛT\sim\Lambda. This metric tells us how much we jump around at the final recurrence. Overly small jumps indicate that ℛ\mathcal{R} isn’t learning anything meaningful, while overly large jumps indicate ℛ\mathcal{R} is suffering from state explosion or is unable to learn fixed-point dynamics.
* •

  Recurrent State Norm: ‖hT‖||h\_{T}||, where T∼ΛT\sim\Lambda. In general, we don’t want an overly large recurrent state norm as it creates numerical instabilities and leads to overly large gradients.

## Appendix B Extended Literature Review

Looping model depth has been well explored by prior work; with a large body of work studying looping within general language modeling [dehghaniUniversalTransformers2019, zhuScalingLatentReasoning2025, geiping\_scaling\_2025, mcleish\_retrofitted\_recurrence, baeMixtureofRecursionsLearningDynamic2025] or small-scale algorithmic problems [avi\_learn\_algorithm, yangLoopedTransformersAre2023, bansalEndtoendAlgorithmSynthesis2022, wangHierarchicalReasoningModel2025b, jolicoeur-martineauLessMoreRecursive2025]. Within looped architectures, the design of training paradigms can be relatively split between architectures with explicit halting mechanisms [dehghaniUniversalTransformers2019, zhuScalingLatentReasoning2025, baeMixtureofRecursionsLearningDynamic2025, jolicoeur-martineauLessMoreRecursive2025, wangHierarchicalReasoningModel2025b] and those with implicit halting mechanisms [geiping\_scaling\_2025, mcleish\_retrofitted\_recurrence, LoopFormerElasticDepthLooped2025, xuExpressivePowerLooped2025].
Looped architectures trained with an explicit halting mechanism use specialized architectures to predict when to early exit tokens, preventing additional computation updates on their recurrent stream [wangHierarchicalReasoningModel2025b, jolicoeur-martineauLessMoreRecursive2025, baeMixtureofRecursionsLearningDynamic2025, dehghaniUniversalTransformers2019, elbayadDepthAdaptiveTransformer2020]. Specifically, wangHierarchicalReasoningModel2025b, jolicoeur-martineauLessMoreRecursive2025 formalize *adaptive-computation-time*, a method that utilizes Q-learning as a means to determine convergence. Similarly, works such as baeMixtureofRecursionsLearningDynamic2025 define an architecture that uses light-weight routers to assign dynamic recursion depths, while zhuScalingLatentReasoning2025 uses a prediction head to dynamically define a probability of exiting after recurrent passes. A majority of these approaches draw on methods of layer skipping [elhoushiLayerSkipEnablingEarly2024, raposoMixtureofDepthsDynamicallyAllocating2024]; however, these methods differ from using a shared parameterization for a recurrent block.

Alternatively, looped architectures with an implicit halting mechanism, such as geiping\_scaling\_2025, mcleish\_retrofitted\_recurrence, avi\_learn\_algorithm, bansalEndtoendAlgorithmSynthesis2022, train models with stochastically sampled recurrent steps during pretraining, and then use the KL-divergence between two successive steps to decide when to exit from the recurrent unit early. Finally, LoopFormerElasticDepthLooped2025 ignores adaptive early exiting altogether, instead pretraining a recurrent unit on a static number of recurrences and enforcing a consistency loss on intermediate recurrences.
Our work focuses solely on implicit recurrent depth models [geiping\_scaling\_2025, mcleish\_retrofitted\_recurrence], which are derived from prior initial work [avi\_learn\_algorithm, bansalEndtoendAlgorithmSynthesis2022].

Beyond training paradigms, there are several differing architectural design choices for looped models [geiping\_scaling\_2025, bansalEndtoendAlgorithmSynthesis2022, saunshiReasoningLatentThoughts2025b]. In simple looped architectures that only place a single recurrent unit, the placement of the looped unit is non-trivial, with certain works looping over all layers [dehghaniUniversalTransformers2019, Csordas2024MoEUTMU, Bae2024RelaxedRT]. Alternatively, saunshiReasoningLatentThoughts2025b find middle-looping recurrent units are the most effective in comparison to other formulations, such as pre-looping and post-looping, which loop the beginning and end of the model. The effectiveness of Middle-looping is consistent with the initial work in synthetic problems by bansalEndtoendAlgorithmSynthesis2022, avi\_learn\_algorithm and with the architecture choices of geiping\_scaling\_2025, mcleish\_retrofitted\_recurrence in large-scale language models before training. Within middle-looping architectures, the number of layers within each unit is mostly chosen ad hoc; however, when bootstrapping from a baseline model, koishekenov2025encodethinkdecodescaling found that you optimize placement by algorithmically selecting layers within a model to loop.

While these prior formulations of looping focus on a single recurrent block, hierarchical [wangHierarchicalReasoningModel2025b, jolicoeur-martineauLessMoreRecursive2025], parallel [wu2025parallellooptransformerefficient], and multi-step [jacobs2026blockrecurrentdynamicsvisiontransformers] formulations of layer looping exist. Furthermore, while not all under the same architectural paradigm, layer looping has been explored in multiple domains (e.g., language [geiping\_scaling\_2025, mcleish\_retrofitted\_recurrence], images [jacobs2026blockrecurrentdynamicsvisiontransformers], multi-modal systems [alabdulmohsin2025recursiveinferencescalingwinning], synthetic algorithmic problems [avi\_learn\_algorithm, bansalEndtoendAlgorithmSynthesis2022, yangLoopedTransformersAre2023]), with the choice of looping style and model architecture design changing based on the specific modality. Where layer looping is introduced, how it is affected by individual modalities, and efficient, FLOP-optimal implementations of layer looping remain open questions.

Finally, layer looping is often deeply tied to deep equilibrium (DEQ) models [bai2019deepequilibriummodels, bai2022neural], due to the fixed-point nature often learned in recurrence. DEQs find the equilibrium points via root-finding to approximate an *infinite depth* network. However, unlike looped architectures trained with truncated backpropagation, a key advantage of DEQ models is their use of implicit differentiation through *infinite depth*, which keeps memory constant and independent of effective depth used to solve the fixed point using a rooting finding algorithm. While the use of implicit differentiation in DEQs enables more efficient training, we focus on work that does explicit backpropagation rollouts [geiping\_scaling\_2025, mcleish\_retrofitted\_recurrence, bansalEndtoendAlgorithmSynthesis2022, yang2024loopedtransformersbetterlearning].
Within looped architectures, geiping\_scaling\_2025, mcleish\_retrofitted\_recurrence adopt the usage of path independence from equilibrium models [anilPathIndependentEquilibrium] to warrant their choice of h0h\_{0} initialization.

## Appendix C Derivation of Instability Conditions of Prior Methods

Recall from [Section 2.1](#S2.SS1 "2.1 Existing Middle-Looped Architectures ‣ 2 Background ‣ Parcae: Scaling Laws For Stable Looped Language Models"), that ℛ\mathcal{R} denotes the full recurrent update ht+1=ℛ​(ht,e)h\_{t+1}=\mathcal{R}(h\_{t},e), encompassing all transformer operations, including residual connections.
A common interpretation views the residual stream as a communication channel where hTh\_{T} is the sum of the relative outputs of all previous layers and the original embedding [olsson2022incontextlearninginductionheads].
Applying this to looped models, let ℛ¯\overline{\mathcal{R}} denote the *relative contribution* of the nonlinear operations (i.e., ℛ¯​(W1​ht+W2​e)=ℛ​(W1​ht+W2​e)−(W1​ht+W2​e)\overline{\mathcal{R}}(W\_{1}h\_{t}+W\_{2}e)=\mathcal{R}(W\_{1}h\_{t}+W\_{2}e)-(W\_{1}h\_{t}+W\_{2}e)).
This gives the recurrent update rule
ht+1=W1​ht+W2​e+ℛ¯​(ht,e)h\_{t+1}=W\_{1}h\_{t}+W\_{2}e+\overline{\mathcal{R}}(h\_{t},e)
where we write ℛ¯​(ht,e)=ℛ¯​(W1​ht+W2​e)\overline{\mathcal{R}}(h\_{t},e)=\overline{\mathcal{R}}(W\_{1}h\_{t}+W\_{2}e) for brevity.
Although ℛ¯\overline{\mathcal{R}} is highly non-linear, the recurrent update can be exactly formulated as a *non-linear time-variant dynamical system* of the form:
ht=𝑨¯​ht−1+𝑩¯​e+ℛ¯​(ht−1,e),yt=𝑪​ht,h\_{t}=\overline{\bm{A}}h\_{t-1}+\overline{\bm{B}}e+\overline{\mathcal{R}}(h\_{t-1},e),~y\_{t}=\bm{C}h\_{t},
where 𝑨¯=W1\overline{\bm{A}}=W\_{1}, 𝑩¯=W2\overline{\bm{B}}=W\_{2}, and 𝑪∈Rdc×dh\bm{C}\in R^{d\_{c}\times d\_{h}} decouples the 𝒞\mathcal{C} and ℛ\mathcal{R} embedding dimension (i.e. p=𝒞​(𝑪​(hT))p=\mathcal{C}(\bm{C}(h\_{T}))).

Using the *relative contribution* representation of looped models above, we can recast
prior mediums of input injection discussed in [Section 2.1](#S2.SS1 "2.1 Existing Middle-Looped Architectures ‣ 2 Background ‣ Parcae: Scaling Laws For Stable Looped Language Models") in a form similar to our framework.
Specifically, for Pre-Norm looped models using addition as injection [yangLoopedTransformersAre2023], the dynamical systems update rule can thus be written in the form ht+1=I​ht+I​e+ℛ¯​(I​ht+I​e)h\_{t+1}=Ih\_{t}+Ie+\overline{\mathcal{R}}(Ih\_{t}+Ie).
When linearized (i.e., dropping the nonlinear ℛ¯\overline{\mathcal{R}} block), 𝑨¯=I\overline{\bm{A}}=I, meaning that the model is a *marginally-stable* system as all eigenvalues are 1. Alternatively, the update rule for Pre-Norm looped models using concatenation as injection [geiping\_scaling\_2025] can be rewritten in the form ht+1=W​[ht;e]+ℛ¯​(W​[ht;e])=W1​ht+W2​e+ℛ¯​(W1​ht+W2​e)h\_{t+1}=W[h\_{t};e]+\overline{\mathcal{R}}(W[h\_{t};e])=W\_{1}h\_{t}+W\_{2}e+\overline{\mathcal{R}}(W\_{1}h\_{t}+W\_{2}e). Here 𝑨¯=W1\overline{\bm{A}}=W\_{1} is unbounded and thus can create an explosion of the state if not carefully maintained during training.

## Appendix D FLOP Estimate of Parcae

In standard, fixed-depth architectures, a common means to approximate the number of FLOPs used in training is C=6​N​DC=6ND from kaplan2020scalinglawsneurallanguage, where NN is the number of parameters and DD is the number of tokens used in training. However, looped architectures differ from traditional models in that they exhibit the notion of *effective parameters* N^\hat{N} (e.g., for a model that is a single layer with NN parameters, if it is looped ten times, then it has an effective parameterization of N^=10​N\hat{N}=10N). Furthermore, as Parcae uses truncated backpropagation through depth, the effective parameters can thus be decoupled into two types: N^1\hat{N}\_{1}, which are effective parameters that *are not backpropagated* through, and N^2\hat{N}\_{2}, which are effective parameters that *are backpropagated* through. Thus, following kaplan2020scalinglawsneurallanguage, we can formulate the effective FLOPs of Parcae as C=(2​N^1+6​N^2)​DC=(2\hat{N}\_{1}+6\hat{N}\_{2})D, which further matches the setup of mcleish\_retrofitted\_recurrence. Like mcleish\_retrofitted\_recurrence, we exclude embedding parameters from N^\hat{N}, however, we do include unembedding parameters in N^\hat{N} similar to nanochat. Lastly, we additionally include an estimate for attention FLOPs following chowdhery2022palmscalinglanguagemodeling, nanochat.

## Appendix E Parcae Forward Pass and Training Algorithms

A full forward pass of Parcae, combining our dynamical systems blocks 𝑨,𝑩,𝑪,Δ\bm{A},\bm{B},\bm{C},\Delta and looped models 𝒫\mathcal{P}, ℛ\mathcal{R}, 𝒞\mathcal{C} blocks can be found in [Algorithm 1](#alg1 "In Appendix E Parcae Forward Pass and Training Algorithms ‣ Parcae: Scaling Laws For Stable Looped Language Models").

Algorithm 1  Parcae Forward Pass

1:Input sequence s∈Vns\in V^{n} and recurrent steps TT.

2:e←LN​(𝒫​(s))e\leftarrow\text{LN}(\mathcal{P}(s))

3:h0∼𝒩​(0,σ2​In×d)h\_{0}\sim\mathcal{N}(0,\sigma^{2}I\_{n\times d})

4:𝑨¯,𝑩¯←𝑨,𝑩,Δ\overline{\bm{A}},\overline{\bm{B}}\leftarrow\bm{A},\bm{B},\Delta

5:for t=1t=1 to TT do

6:  ht←𝑨¯​ht−1+𝑩¯​e+ℛ¯​(ht,e)h\_{t}\leftarrow\overline{\bm{A}}h\_{t-1}+\overline{\bm{B}}e+\overline{\mathcal{R}}(h\_{t},e)

7:end for

8:return 𝒞​(𝑪​hT)\mathcal{C}(\bm{C}h\_{T})

We display our algorithm to sample per-sequence depths during Parcae training while maintaining compute efficiency in Algorithm [2](#alg2 "Algorithm 2 ‣ Appendix E Parcae Forward Pass and Training Algorithms ‣ Parcae: Scaling Laws For Stable Looped Language Models"). We do per-sequence depth sampling, but taking the max depth within a batch and performing no state updates at the *beginning* of the recurrent computation. This allows for batched processing of different depths while maintaining efficient gradient flow.

Algorithm 2  Efficient Per-Sequence Stochastic Depth Training

1:Batch of sequences {si}i=1B\{s\_{i}\}\_{i=1}^{B}, means μrec,μbwd\mu\_{\text{rec}},\mu\_{\text{bwd}}, and sampling distribution Λ\Lambda

2:𝒆(i)←𝒫​(si)\bm{e}^{(i)}\leftarrow\mathcal{P}(s\_{i}) for all ii ⊳\triangleright embed sequences

3:Sample T(i)∼Λ​(μrec)T^{(i)}\sim\Lambda(\mu\_{\text{rec}}) for each i∈[B]i\in[B]

4:Tmax←maxi⁡T(i)T\_{\max}\leftarrow\max\_{i}T^{(i)},  τ(i)←Tmax−T(i)\tau^{(i)}\leftarrow T\_{\max}-T^{(i)}

5:𝒉0(i)∼𝒩​(0,σ​𝑰)\bm{h}\_{0}^{(i)}\sim\mathcal{N}(0,\sigma\bm{I}) for all ii

6:𝑨¯,𝑩¯←Discretize​(𝑨,𝑩,Δ)\overline{\bm{A}},\overline{\bm{B}}\leftarrow\textsc{Discretize}(\bm{A},\bm{B},\Delta)

7:for t=0,…,Tmax−1t=0,\ldots,T\_{\max}-1 do

8:  for all ii where t<τ(i)t<\tau^{(i)}:  𝒉t+1(i)←𝒉t(i)\bm{h}\_{t+1}^{(i)}\leftarrow\bm{h}\_{t}^{(i)} ⊳\triangleright no state update

9:  for all ii where τ(i)≤t<Tmax−μbwd\tau^{(i)}\leq t<T\_{\max}-\mu\_{\text{bwd}}: ⊳\triangleright without gradients

10:   𝒉t+1(i)←𝑨¯​𝒉t(i)+𝑩¯​𝒆(i)+ℛ​(𝒉t(i),𝒆(i))\bm{h}\_{t+1}^{(i)}\leftarrow\overline{\bm{A}}\bm{h}\_{t}^{(i)}+\overline{\bm{B}}\bm{e}^{(i)}+\mathcal{R}(\bm{h}\_{t}^{(i)},\bm{e}^{(i)})

11:  for all ii where t≥Tmax−μbwdt\geq T\_{\max}-\mu\_{\text{bwd}}: ⊳\triangleright with gradients

12:   𝒉t+1(i)←𝑨¯​𝒉t(i)+𝑩¯​𝒆(i)+ℛ​(𝒉t(i),𝒆(i))\bm{h}\_{t+1}^{(i)}\leftarrow\overline{\bm{A}}\bm{h}\_{t}^{(i)}+\overline{\bm{B}}\bm{e}^{(i)}+\mathcal{R}(\bm{h}\_{t}^{(i)},\bm{e}^{(i)})

13:end for

14:return {𝒞​(𝑪​𝒉Tmax(i))}i=1B\{\mathcal{C}(\bm{C}\bm{h}\_{T\_{\max}}^{(i)})\}\_{i=1}^{B}

## Appendix F Additional Stability Ablations

We include all training curves for our hyperparameter sweep experiments in [Appendix Q](#A17 "Appendix Q Hyperparameters and Training Details ‣ Parcae: Scaling Laws For Stable Looped Language Models"). We conduct a learning rate sweep over {2​e−4,4​e−4,6​e−4,8​e−4,1​e−3}\{2e-4,4e-4,6e-4,8e-4,1e-3\} observing that Parcae exhibits stable training over both baseline Pre-Norm RDMs and residual normalized RDMs. The training curves and the accompanying recurrent state norm can be observed in [Figure 9](#A6.F9 "In Appendix F Additional Stability Ablations ‣ Parcae: Scaling Laws For Stable Looped Language Models").

![Refer to caption](/html/2604.12946/assets/x9.png)

![Refer to caption](/html/2604.12946/assets/x10.png)

![Refer to caption](/html/2604.12946/assets/x11.png)

![Refer to caption](/html/2604.12946/assets/x12.png)

![Refer to caption](/html/2604.12946/assets/x13.png)

Figure 9: Training instability of recurrent depth models across different learning rates. We show both training losses and recurrent state norm to understand divergence and state explosion.

## Appendix G Per-sequence Sampling Reduces Loss Spikes

When running our per-sequence sampling experiments, we observed that the training curves of per-sequence sampling helped eliminate loss spikes during training. Specifically, in [Figure 10](#A7.F10 "In Appendix G Per-sequence Sampling Reduces Loss Spikes ‣ Parcae: Scaling Laws For Stable Looped Language Models"), for our 350M parameter Parcae models, per-micro-batch has several loss spikes through training while per-sequence sampling does not. We can observe from [Figure 11](#A7.F11 "In Appendix G Per-sequence Sampling Reduces Loss Spikes ‣ Parcae: Scaling Laws For Stable Looped Language Models"), that these training spikes stem directly from overly large recurrent residual jumps at the final recurrence, implying the model is not learning to converge to a steady-state fixed point solution.
It can then be observed that per-sequence depth helps provide a better estimate for our training objective, enabling convergent fixed-point behavior and preventing loss-spikes during training. The direct benefit of this can be observed in [Table 8](#A7.T8 "In Appendix G Per-sequence Sampling Reduces Loss Spikes ‣ Parcae: Scaling Laws For Stable Looped Language Models"), where per-sequence sampling significantly improves the downstream quality of looped models, especially at low test-time recurrences. Finally, we note that per-sequence sampling adds a minimal amount of training overhead, increasing total wall clock time for pretraining by 1.8%, which we believe can be further optimized away with a cleaner implementation.

![Refer to caption](/html/2604.12946/assets/x14.png)


Figure 10: Training curves showing per-sequence sampling effectively eliminates loss spikes in training over per-micro-batch sampling.

![Refer to caption](/html/2604.12946/assets/x15.png)


Figure 11: Comparison of recurrent residual and state norm metrics (defined in [Section A.1](#A1.SS1 "A.1 Notation ‣ Appendix A Glossary ‣ Parcae: Scaling Laws For Stable Looped Language Models")), which show that per-sequence sampling enables stronger fixed point behavior in training.



|  | Method | T=1 | T=4 | T=8 | T=16 |
| --- | --- | --- | --- | --- | --- |
| 100M | Per-Batch | 300.32 | 36.75 | 16.65 | 13.81 |
| Per-Sequence | 70.47 | 17.15 | 14.08 | 13.59 |
| 350M | Per-Batch | 167.61 | 12.80 | 10.40 | 10.24 |
| Per-Sequence | 17.92 | 10.49 | 10.09 | 10.11 |

Table 8: Per-Microbatch vs. Per-Sequence Comparison. We compare perplexity of Parcae models trained with per-microbatch sampling [geiping\_scaling\_2025] and per-sequence sampling, using different recurrences (TT) on a held-out validation set. Bolded results indicate best at each scale.

## Appendix H Sampling of Truncated Recurrence

Algorithm 3  geiping\_scaling\_2025

1:Input: μrec\mu\_{\text{rec}}, μbwd,Λ\mu\_{\text{bwd}},\Lambda, ee

2:n∼Λ​(μrec−μbwd)n\sim\Lambda(\mu\_{\text{rec}}-\mu\_{\text{bwd}})

3:k←μbwdk\leftarrow\mu\_{\text{bwd}}

4:T=n+kT=n+k

5:h0←𝒩​(0,σ2​I)h\_{0}\leftarrow\mathcal{N}(0,\sigma^{2}I)

6:for t=1t=1 to TT do

7:  if t≤nt\leq n then

8:   ht←ℛ​(ht−1,e)h\_{t}\leftarrow\mathcal{R}(h\_{t-1},e) w/o grad

9:  else

10:   ht←ℛ​(ht−1,e)h\_{t}\leftarrow\mathcal{R}(h\_{t-1},e) w/ grad

11:  end if

12:end for

13:return xTx\_{T}

Algorithm 4  Correction (Ours)

1:Input: μrec\mu\_{\text{rec}}, μbwd\mu\_{\text{bwd}}, Λ\Lambda, ee

2:T∼Λ​(μrec)T\sim\Lambda(\mu\_{\text{rec}})

3:n←max⁡(T−μbwd,0)n\leftarrow\max(T-\mu\_{\text{bwd}},0)

4:k←min⁡(T,μbwd)k\leftarrow\min(T,\mu\_{\text{bwd}})

5:h0←𝒩​(0,σ2​I)h\_{0}\leftarrow\mathcal{N}(0,\sigma^{2}I)

6:for t=1t=1 to TT do

7:  if t≤nt\leq n then

8:   ht←ℛ​(ht−1,e)h\_{t}\leftarrow\mathcal{R}(h\_{t-1},e) w/o grad

9:  else

10:   ht←ℛ​(ht−1,e)h\_{t}\leftarrow\mathcal{R}(h\_{t-1},e) w/ grad

11:  end if

12:end for

13:return xTx\_{T}

Figure 12: Comparison of our sampling method with [geiping\_scaling\_2025]. It can be observed that the actual distribution of forward recurrence for [geiping\_scaling\_2025] is a shifted Poisson distribution. The implications of this sampling strategy can be better visualized in [Figure 13](#A8.F13 "In Appendix H Sampling of Truncated Recurrence ‣ Parcae: Scaling Laws For Stable Looped Language Models").

![Refer to caption](/html/2604.12946/assets/x16.png)


Figure 13: A distributional mismatch can be observed from the recurrent sampling method of [geiping\_scaling\_2025]. Specifically, if our desired pre-training distribution for μrec\mu\_{\text{rec}} is a Poisson distribution, the distribution total recurrence TT of [geiping\_scaling\_2025] is truncated based on μbwd\mu\_{\text{bwd}}. However, our sampling method decouples the effects of μbwd\mu\_{\text{bwd}} on Λ\Lambda, allowing the recurrent distribution to be faithfully sampled from.

In our very initial experiments, we observed that we could make a small change to the sampling algorithm of [geiping\_scaling\_2025], which stems from [avi\_learn\_algorithm], to enhance the training of Parcae333We make the same change to RDMs in the main body, observing that they perform better with it and to make comparison fair.. When given an arbitrary distribution to sample from Λ\Lambda and two hyperparameters μrec\mu\_{\text{rec}} (the desired mean steps of the recurrent blocks in pre-training) and μbwd\mu\_{\text{bwd}} (the desired mean back-propagation steps in pre-training), we observe that previous work by [geiping\_scaling\_2025] had a distributional mismatch. Previously, the sampling method of [geiping\_scaling\_2025] exactly followed [Algorithm 3](#alg3 "In Figure 12 ‣ Appendix H Sampling of Truncated Recurrence ‣ Parcae: Scaling Laws For Stable Looped Language Models") with a poisson log-normal distribution with the following distribution

|  |  |  |  |
| --- | --- | --- | --- |
|  | τ∼𝒩​(log⁡(μrec−μbwd)−12​σ2,σ)n∼𝒫​(eτ)+1k←μbwd\displaystyle\tau\sim\mathcal{N}(\log(\mu\_{\text{rec}}-\mu\_{\text{bwd}})-\frac{1}{2}\sigma^{2},\sigma)\qquad n\sim\mathcal{P}(e^{\tau})+1\qquad k\leftarrow\mu\_{\text{bwd}} |  | (5) |

where σ=12\sigma=\frac{1}{2}. To maintain a fixed computation memory budget, [geiping\_scaling\_2025] sets kk to μbwd\mu\_{\text{bwd}}; however, this minor change significantly impacts the underlying recurrent distribution, truncating and compressing the distribution of recurrence actually observed during pre-training. We propose making a minor algorithmic fix to the sampling method, which can be observed in [Algorithm 4](#alg4 "In Figure 12 ‣ Appendix H Sampling of Truncated Recurrence ‣ Parcae: Scaling Laws For Stable Looped Language Models"). While minor, observe in [Figure 13](#A8.F13 "In Appendix H Sampling of Truncated Recurrence ‣ Parcae: Scaling Laws For Stable Looped Language Models") the impact of improving generalization to other recurrences.

To verify our change, we pretrain several small Parcae models on 10 billion tokens to ablate on our design choice. Specifically, we set μrec=μbwd=8\mu\_{\text{rec}}=\mu\_{\text{bwd}}=8 and use Λ∼Poisson\Lambda\sim\text{Poisson}, and use fixed architecture, hyperparameters, and data stream. We train three models: a baseline Parcae model that performs full backpropagation through recurrences, a Parcae model following [Algorithm 3](#alg3 "In Figure 12 ‣ Appendix H Sampling of Truncated Recurrence ‣ Parcae: Scaling Laws For Stable Looped Language Models") by [geiping\_scaling\_2025], and a Parcae model following [Algorithm 4](#alg4 "In Figure 12 ‣ Appendix H Sampling of Truncated Recurrence ‣ Parcae: Scaling Laws For Stable Looped Language Models"). The results of this ablation can be found in [Figure 14](#A8.F14 "In Appendix H Sampling of Truncated Recurrence ‣ Parcae: Scaling Laws For Stable Looped Language Models").

![Refer to caption](/html/2604.12946/assets/x17.png)


Figure 14: Training and validation curves of three 100 million parameter Parcae models pretrained on 10 billion tokens, comparing different truncated back-propagation methods (baseline is a model with no back-propagation truncation). Each model has identical architecture and hyperparameters, with μrec\mu\_{\text{rec}} and μbwd\mu\_{\text{bwd}} both being set to eight, all using Λ∼Poisson\Lambda\sim\text{Poisson}. It can be observed that even though each model has similar training loss and validation loss when using T=8T=8, our implementation more faithfully follows the validation loss of full back-propagation. Specifically for T=4T=4, our implementation significantly improves validation loss compared to [geiping\_scaling\_2025] sampling method.

From [Figure 14](#A8.F14 "In Appendix H Sampling of Truncated Recurrence ‣ Parcae: Scaling Laws For Stable Looped Language Models"), observe that training trajectories and validation loss at T=μrec=8T=\mu\_{\text{rec}}=8 are almost identical for each run; however, our method significantly improves performance for the validation loss of T∈[4,16,64]T\in[4,16,64]. Simply put, the constricting effect [geiping\_scaling\_2025] observed in [Figure 13](#A8.F13 "In Appendix H Sampling of Truncated Recurrence ‣ Parcae: Scaling Laws For Stable Looped Language Models") reduces the effective range of recurrence seen in pretraining, hurting the validation loss of using more or fewer recurrence at test-time.

## Appendix I Selecting μrec\mu\_{\text{rec}} and μbwd\mu\_{\text{bwd}}

![Refer to caption](/html/2604.12946/assets/x18.png)


Figure 15: Validation curves of six different recurrent depth models, pretrained on 10 billion tokens, with a fixed architecture and hyperparameters. Each model is pretrained with a fixed μbwd\mu\_{\text{bwd}} of 8 and varying μrec\mu\_{\text{rec}} in [4,8,14,20,26,32][4,8,14,20,26,32]. The key observation is that scaling up μrec\mu\_{\text{rec}} while keeping μbwd\mu\_{\text{bwd}} fixed results in models that perform worse than if just pretrained on μrec\mu\_{\text{rec}} of eight.



|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | μrec=4\mu\_{\text{rec}}=4 | μrec=8\mu\_{\text{rec}}=8 | μrec=14\mu\_{\text{rec}}=14 | μrec=20\mu\_{\text{rec}}=20 | μrec=26\mu\_{\text{rec}}=26 | μrec=32\mu\_{\text{rec}}=32 |
| Val Loss | 2.477 | 2.453 | 2.456 | 2.457 | 2.458 | 2.458 |
| Val Perplexity | 11.906 | 11.624 | 11.665 | 11.671 | 11.692 | 11.687 |

Table 9: Validation loss and perplexity for looped models trained with different μrec\mu\_{\text{rec}} and a fixed μbwd=4\mu\_{\text{bwd}}=4. We use T=μrecT=\mu\_{\text{rec}}. Surprisingly, μrec=8\mu\_{\text{rec}}=8 performs the best.

A natural question is what choice of μrec\mu\_{\text{rec}} and μbwd\mu\_{\text{bwd}} is appropriate for pretraining looped models. To answer this question, we conduct an experiment where we scale up μrec\mu\_{\text{rec}}, while keeping μbwd\mu\_{\text{bwd}} fixed. In our very initial experiments, we pretrained several small recurrent depth models [geiping\_scaling\_2025] on 10 billion tokens, with a fixed μbwd=4\mu\_{\text{bwd}}=4 and with μrec∈[4,8,14,20,26,32]\mu\_{\text{rec}}\in[4,8,14,20,26,32]444Note that these experiments were run before the distribution mismatch fix discussed in [Appendix H](#A8 "Appendix H Sampling of Truncated Recurrence ‣ Parcae: Scaling Laws For Stable Looped Language Models"). As the mismatch becomes more drastic as μrec\mu\_{\text{rec}} gets closer to μbwd\mu\_{\text{bwd}}, we expect the model pretrained with μrec=4\mu\_{\text{rec}}=4 to be performing sub-optimally.. The results for each of these models on a held-out set of validation data can be observed in [Figure 15](#A9.F15 "In Appendix I Selecting 𝜇_\"rec\" and 𝜇_\"bwd\" ‣ Parcae: Scaling Laws For Stable Looped Language Models"). We additionally include [Table 9](#A9.T9 "In Appendix I Selecting 𝜇_\"rec\" and 𝜇_\"bwd\" ‣ Parcae: Scaling Laws For Stable Looped Language Models"), which gives the validation loss of each model with μrec∈[4,8,14,20,26,32]\mu\_{\text{rec}}\in[4,8,14,20,26,32], where the recurrence that we use for each model at test-time is T=μrecT=\mu\_{\text{rec}}.

The fascinating observation of [Figure 15](#A9.F15 "In Appendix I Selecting 𝜇_\"rec\" and 𝜇_\"bwd\" ‣ Parcae: Scaling Laws For Stable Looped Language Models") is that, contrary to our initial beliefs, models trained with additional μrec\mu\_{\text{rec}} beyond 8 perform worse at both lower and higher rr used at test-time, though more FLOPs were spent during pretraining. While it is a natural expectation that models trained with lower μrec\mu\_{\text{rec}} perform better than models with larger μrec\mu\_{\text{rec}} at low TT, the fact that a μrec\mu\_{\text{rec}} of eight performs the best at higher TT (i.e., T=16T=16 and T=64T=64) is surprising. To determine if this is an inherent limitation of the capacity of looped models or an artifact of μbwd\mu\_{\text{bwd}}, we ran an additional experiment where we fixed μr​e​c=20\mu\_{rec}=20 and instead varied μbwd∈[4,6,8,10,12]\mu\_{\text{bwd}}\in[4,6,8,10,12], pretraining on 8.5 billion tokens for each model. We keep hyperparameters fixed. The results for each of these models on a held-out set of validation data can be visualized in [Figure 16](#A9.F16 "In Appendix I Selecting 𝜇_\"rec\" and 𝜇_\"bwd\" ‣ Parcae: Scaling Laws For Stable Looped Language Models") and [Table 10](#A9.T10 "In Appendix I Selecting 𝜇_\"rec\" and 𝜇_\"bwd\" ‣ Parcae: Scaling Laws For Stable Looped Language Models").

![Refer to caption](/html/2604.12946/assets/x19.png)


Figure 16: Validation and training curves of looped models, pretrained on 8.5 billion tokens. Each model is trained with a fixed μrec=20\mu\_{\text{rec}}=20 and μb​w​d∈[4,6,8,10,12]\mu\_{bwd}\in[4,6,8,10,12]. Observe that scaling up μbwd\mu\_{\text{bwd}} improves validation performance at higher and lower recurrences monotonically for T=1,16,64T=1,16,64.



|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | μbwd=4\mu\_{\text{bwd}}=4 | μbwd=6\mu\_{\text{bwd}}=6 | μbwd=8\mu\_{\text{bwd}}=8 | μbwd=10\mu\_{\text{bwd}}=10 | μbwd=12\mu\_{\text{bwd}}=12 |
| Val Loss | 2.500 | 2.490 | 2.480 | 2.479 | 2.474 |
| Val Perplexity | 12.09 | 12.06 | 11.94 | 11.93 | 11.86 |

Table 10: Validation loss and perplexity of looped models trained with variable μbwd\mu\_{\text{bwd}}, but fixed μrec\mu\_{\text{rec}}.

While lower μbwd\mu\_{\text{bwd}} (i.e., μbwd=4,6,8\mu\_{\text{bwd}}=4,6,8) appears to perform better with lower validation recurrences than higher μbwd\mu\_{\text{bwd}}, the validation loss using T=16,64T=16,64 improves as μbwd\mu\_{\text{bwd}} increases. This implies that the capabilities of looped models utilizing deeper recurrences are heavily coupled with μbwd\mu\_{\text{bwd}}. However, it can be observed that increasing μbwd\mu\_{\text{bwd}} from ten to twelve has minimal impact on validation performance, at the cost of higher pretraining FLOPs. Using this insight, for our main training runs, we choose to use

|  |  |  |  |
| --- | --- | --- | --- |
|  | μbwd=⌈μrec2⌉\mu\_{\text{bwd}}=\lceil\frac{\mu\_{\text{rec}}}{2}\rceil |  | (6) |

We leave the exploration of FLOP optimal choices of μrec\mu\_{\text{rec}} and μbwd\mu\_{\text{bwd}} to future work.

## Appendix J Ablation of Prelude Normalization

In our initial set of experiments, we found that Parcae was able to train stably on the 140M, 370M, and 770M model configurations. Unfortunately, at the 1.3B scale, training appeared stable for the first 150k optimizer steps, afterwards exhibited state explosion and loss spikes, an observation which can be made in [Figure 17](#A10.F17 "In Appendix J Ablation of Prelude Normalization ‣ Parcae: Scaling Laws For Stable Looped Language Models"). To diagnose and fix these issues, we performed a deep exploration of the weight checkpoints before and during loss spikes, investigating both dynamical systems parameters (e.g., 𝑨,𝑩,𝑪,Δ\bm{A},\bm{B},\bm{C},\Delta) and non-linear parameters ℛ¯\overline{\mathcal{R}}.

![Refer to caption](/html/2604.12946/assets/x20.png)


Figure 17: Late Stage Instability of 1.3B Parcae models. We observe loss spikes and state explosion at the final stages of our large-scale run.

![Refer to caption](/html/2604.12946/assets/x21.png)


Figure 18: Spectral Norms of A¯,B¯,C\overline{\bm{A}},\overline{\bm{B}},\bm{C} throughout training 1.3B Parcae. We find that the spectral norm of 𝑨¯\overline{\bm{A}} and 𝑩¯\overline{\bm{B}} remain stable throughout training, while the spectral norm of 𝑪\bm{C} grows.

We begin by exploring the spectral norm of 𝑨¯\overline{\bm{A}}, 𝑩¯\overline{\bm{B}}, 𝑪\bm{C} to see if our dynamical systems block was creating instability, results of which can be found in [Figure 18](#A10.F18 "In Appendix J Ablation of Prelude Normalization ‣ Parcae: Scaling Laws For Stable Looped Language Models"). While we observe that the spectral norm remains relatively low for 𝑨¯\overline{\bm{A}} and 𝑩¯\overline{\bm{B}}, we observed that the spectral norm of 𝑪\bm{C} grew significantly throughout training. While this could be concerning, we find that when passing real activations to 𝑪\bm{C}, using a subset of the validation set, the empirical expanse ratio ‖C​(x)‖‖x‖\frac{||C(x)||}{||x||} (i.e., how much the norm of the residual xx grew after performing 𝑪​(x)\bm{C}(x)) remained relatively low, as seen in [Figure 19](#A10.F19 "In Appendix J Ablation of Prelude Normalization ‣ Parcae: Scaling Laws For Stable Looped Language Models").

![Refer to caption](/html/2604.12946/assets/x22.png)


Figure 19: Comparison of C\bm{C} Amplification with Spectral Norm. We observe that the actual expansion ratio of 𝑪\bm{C} is small and decreasing slowly throughout training.

![Refer to caption](/html/2604.12946/assets/x23.png)


Figure 20: Empirical Average of Recurrent State Norm over TT iterations. For each checkpoint we have for our failed 1.3B Parcae model run, we evaluate the recurrent norm through T=24T=24 recurrences at test time, on a held out validation set of fineweb-edu [penedo2024finewebdatasetsdecantingweb]. We find that after an initial explosion on the first recurrence, the state remains relatively stable.

These results indicate that the dynamical systems units are likely not causing an explosion, and thus, we turn our exploration of the dynamics of the entire recurrent unit. Specifically, we track the recurrent state norm at test-time after T=24T=24 recurrences, results of which can be found in [Figure 20](#A10.F20 "In Appendix J Ablation of Prelude Normalization ‣ Parcae: Scaling Laws For Stable Looped Language Models"). We found that on the first recurrence, the recurrent state norm jumped drastically, and then remained relatively stable throughout increased recurrences. To determine what caused the initial spike, we perform a fine-grained analysis of the first recurrence (i.e., T=1T=1), tracking the recurrent state norm after injection and through each transformer block, the results of which can be found in [Figure 21](#A10.F21 "In Appendix J Ablation of Prelude Normalization ‣ Parcae: Scaling Laws For Stable Looped Language Models"). The major takeaway from [Figure 21](#A10.F21 "In Appendix J Ablation of Prelude Normalization ‣ Parcae: Scaling Laws For Stable Looped Language Models") is that the non-linear parts of Parcae do not appear to cause the explosion in state and that the initial explosion steps from the input injections of ee, the output of the prelude block 𝒫\mathcal{P}. We confirm that this is the case, and visualization of which can be seen in [Figure 22](#A10.F22 "In Appendix J Ablation of Prelude Normalization ‣ Parcae: Scaling Laws For Stable Looped Language Models").

![Refer to caption](/html/2604.12946/assets/x24.png)


Figure 21: Recurrent State Norm Progression After Each Transformer Block for T=1T=1. For each checkpoint we have for our failed 1.3B Parcae model run, we evaluate the recurrent norm after injection and each non-linear transformer block for only T=1T=1. We find that the non-linear parts of Parcae have little effect on explosion, which instead mainly stems from the initial injection of prelude output ee.

![Refer to caption](/html/2604.12946/assets/x25.png)


Figure 22: State Norm Progression Throughout each Transformer Layer in the Prelude Block. For each checkpoint we have for our failed 1.3B Parcae model run, we evaluate residual norm after each transformer block in the prelude 𝒫\mathcal{P}. We find that a single layer creates an explosion of the residual norm and leads to divergence.

Given this, we propose a simple fix of adding a normalization layer on the output of the prelude block 𝒫\mathcal{P} (i.e., for an input xx then e←LN​(𝒫​(x))e\leftarrow\text{LN}(\mathcal{P}(x)), where LN​(⋅)\text{LN}(\cdot) is some form of normalization). We note that this does two things: (1) normalizes the input to the recurrent unit, which we observe to further stabilize the recurrent dynamics of looping, and (2) stabilizes the gradient flow to the 𝒫\mathcal{P}.555We do not directly prove or show this; however, it can be inferred by prior work on how normalization stabilizes forward and backward passes of transformers [xu2019understandingimprovinglayernormalization, xiongLayerNormalizationTransformer2020]. This simple fix enables our stable training run for the 1.3B Parcae reported in [Section 5](#S5 "5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models").

Empirically, we find that using a prelude norm directly stabilizes the recurrent norm further, preventing the recurrent norm from growing too large (see [Figure 23](#A10.F23 "In Appendix J Ablation of Prelude Normalization ‣ Parcae: Scaling Laws For Stable Looped Language Models")). Additionally, we find that using a prelude norm leads to better convergence in both our 140M and 370M Parcae models (see [Figure 24](#A10.F24 "In Appendix J Ablation of Prelude Normalization ‣ Parcae: Scaling Laws For Stable Looped Language Models")), with only a negligible improvement for our 770M and 1.3B Parcae models.

![Refer to caption](/html/2604.12946/assets/x26.png)


Figure 23: Prelude Norm Stabilizes Recurrent Norm. We find that prelude norm helps stabilize recurrent state norm in Parcae models following the setup in [Section 5.1](#S5.SS1 "5.1 Parcae Improves End-to-End Quality ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models") for Transformers.

![Refer to caption](/html/2604.12946/assets/x27.png)


Figure 24: Prelude Norm Improves Quality. We find that in our 140M and 370M Parcae models trained in the same setup as [Section 5.1](#S5.SS1 "5.1 Parcae Improves End-to-End Quality ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models") for Transformers, normalizing the prelude output leads to better convergence.

## Appendix K Fitting a Parametric Function for Looping

We follow hoffmann2022trainingcomputeoptimallargelanguage setup for fitting a parametric loss function. Specifically, using the models trained with several IsoFLOP budgets in [Section 5.2](#S5.SS2 "5.2 Looping as an Orthogonal Scaling Axis in Training ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models"), we fit a parametric function of the form

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ^train​(μrec,𝒟)=E+A⋅𝐍​(μrec)−a+B⋅𝒟−b\widehat{\mathcal{L}}\_{\text{train}}(\mu\_{\text{rec}},\mathcal{D})=E+A\cdot\mathbf{N}(\mu\_{\text{rec}})^{-a}+B\cdot\mathcal{D}^{-b} |  | (7) |

where 𝐍​(μrec)\mathbf{N}(\mu\_{\text{rec}}) is the *effective parameter count* of the model if you were to unroll all loops into real parameters, 𝒟\mathcal{D} is the number of tokens that were used in training, and A,B,a,bA,B,a,b are learned parameters. We specifically use Huber loss [huber] on the log loss between the prediction of the parametric fit and the validation loss of the models, using L-BFGS [lbfgs] to minimize. We choose the parametric function of this form as it exactly follows [hoffmann2022trainingcomputeoptimallargelanguage], but with parameters 𝐍\mathbf{N} now being a function of μrec\mu\_{\text{rec}}. Finally, we take the best result from 500 random restarts of L-BFGS, each with up to 10,000 iterations, selecting the initialization that achieves the lowest Huber loss. The results of fitting the parametric function can be visualized in [Figure 25](#A11.F25 "In Appendix K Fitting a Parametric Function for Looping ‣ Parcae: Scaling Laws For Stable Looped Language Models"), and the learned values can be observed in [Table 11](#A11.T11 "In Appendix K Fitting a Parametric Function for Looping ‣ Parcae: Scaling Laws For Stable Looped Language Models").

![Refer to caption](/html/2604.12946/assets/x28.png)


Figure 25: Parametric Fit of Looping. Visualization of our parametric function ℒ^train​(μrec,D)\widehat{\mathcal{L}}\_{\text{train}}(\mu\_{\text{rec}},D), which displays the IsoLoss contours for both 140M Parcae (*left*) and 370M Parcae (*right*) models.



| Model | 𝑬\boldsymbol{E} | 𝑨\boldsymbol{A} | 𝒂\boldsymbol{a} | 𝑩\boldsymbol{B} | 𝒃\boldsymbol{b} | Huber (×10−4\times 10^{-4}) |
| --- | --- | --- | --- | --- | --- | --- |
| Small (140M) | 2.662 | 522733.307 | 0.771 | 25420.102 | 0.525 | 0.44 |
| Medium (370M) | 2.439 | 832134.346 | 0.775 | 6386.865 | 0.448 | 0.01 |

Table 11: Optimal Scaling Coefficients for Parametric Fits.

## Appendix L Fitting Parametric Functions to Test-Time Looping

In this section, we provide a more detailed analysis of the test-time scaling laws discussed in [Section 5.3](#S5.SS3 "5.3 Test-Time Scaling Laws of Parcae ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models"). Following the setup discussed in [Appendix O](#A15 "Appendix O Expanded Setup For Training and Test-Time Scaling Laws ‣ Parcae: Scaling Laws For Stable Looped Language Models"), we train several Parcae models on varying μrec\mu\_{\text{rec}}, fixing data and parameter count, and evaluate each at test-time recurrences up to T=24T=24.

#### Choice of Functional Form.

We aim to find a parametric function that captures the saturating relationship between test-time recurrence TT and validation loss. We consider four candidate functional forms, each with an irreducible loss floor ℒ∞\mathcal{L}\_{\infty} (except the pure power law):

1. (a)

   ℒ​(T)=ℒ∞+Z⋅e−z​T\mathcal{L}(T)=\mathcal{L}\_{\infty}+Z\cdot e^{-zT} (exponential decay)
2. (b)

   ℒ​(T)=ℒ∞+Z⋅(1+T)−z\mathcal{L}(T)=\mathcal{L}\_{\infty}+Z\cdot(1+T)^{-z} (shifted power law)
3. (c)

   ℒ​(T)=ℒ∞+Z⋅T−z\mathcal{L}(T)=\mathcal{L}\_{\infty}+Z\cdot T^{-z} (power law)
4. (d)

   ℒ​(T)=Z⋅T−z\mathcal{L}(T)=Z\cdot T^{-z} (power law, no floor)

Each form has 3 free parameters (ℒ∞,Z,z\mathcal{L}\_{\infty},Z,z), except (d), which has 2. We fit each form independently to every test-time curve using least-squares on log-loss, and report the average Huber loss (δ=10−3\delta=10^{-3}) across all curves. To evaluate extrapolation, we additionally fit each form on T≤μrecT\leq\mu\_{\text{rec}} and evaluate on held-out T>μrecT>\mu\_{\text{rec}}.

|  | ℒ∞+Z​e−z​T\mathcal{L}\_{\infty}{+}Ze^{-zT} | ℒ∞+Z​(1+T)−z\mathcal{L}\_{\infty}{+}Z(1{+}T)^{-z} | ℒ∞+Z​T−z\mathcal{L}\_{\infty}{+}ZT^{-z} | Z​T−zZT^{-z} |
| --- | --- | --- | --- | --- |
| In-Distribution | | | | |
| 140M | 2.52 | 5.42 | 11.11 | 112.89 |
| 370M | 1.88 | 5.26 | 10.77 | 104.95 |
| Extrapolation (T>μrecT>\mu\_{\text{rec}}) | | | | |
| 140M | 3.18 | 21.41 | 43.99 | 397.90 |
| 370M | 2.29 | 18.51 | 38.68 | 369.83 |

Table 12: Functional form comparison for test-time scaling. We report average Huber loss (×10−7\times 10^{-7}) across all per-curve fits, both in-distribution (all TT) and in extrapolation (fit T≤μrecT\leq\mu\_{\text{rec}}, evaluate T>μrecT>\mu\_{\text{rec}}). Lower is better.

As shown in [Table 12](#A12.T12 "In Choice of Functional Form. ‣ Appendix L Fitting Parametric Functions to Test-Time Looping ‣ Parcae: Scaling Laws For Stable Looped Language Models"), the exponential decay form achieves the lowest Huber loss both in-distribution (2.3×2.3\times better than the shifted power law) and under extrapolation (7.1×7.1\times better), consistently across both model sizes. Notably, omitting the irreducible floor ℒ∞\mathcal{L}\_{\infty} (form (d)) increases error by over 40×40\times, confirming that test-time scaling saturates to a finite loss determined by training (this is also obvious from looking at [Figure 8](#S5.F8 "In Setup. ‣ 5.3 Test-Time Scaling Laws of Parcae ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models")).

While purely speculative, there is a nice connection between the exponential form and Parcae’s dynamical systems framework. In classical control theory literature, a stable discrete-time linear system with a spectral radius below unity converges exponentially in the state norm. The observed exponential decay in loss is thus consistent with the dynamical system formulation that Parcae uses.

#### Recovery of the training law at T=μrecT=\mu\_{\text{rec}}.

We additionally observe that the fitted irreducible loss ℒ∞\mathcal{L}\_{\infty} closely matches the empirical loss at T=μrecT=\mu\_{\text{rec}} ([Table 13](#A12.T13 "In Recovery of the training law at 𝑇=𝜇_\"rec\". ‣ Appendix L Fitting Parametric Functions to Test-Time Looping ‣ Parcae: Scaling Laws For Stable Looped Language Models")), motivating the use of the training scaling law ℒ^train​(μrec,D)\hat{\mathcal{L}}\_{\mathrm{train}}(\mu\_{\text{rec}},D) as the irreducible floor in a unified law.

| Model | Mean % Err | Max % Err |
| --- | --- | --- |
| 140M | 0.16% | 0.59% |
| 370M | 0.05% | 0.22% |

Table 13: Mean and max absolute percent error between ℒ∞\mathcal{L}\_{\infty} and ℒ​(T=μrec)\mathcal{L}(T{=}\mu\_{\text{rec}}) across all isoFLOP configurations.

#### Conditioning on Training Recurrence.

To model test-time scaling across models trained at different μrec\mu\_{\text{rec}}, the decay rate must depend on the training depth. We compare three forms for the unified test-time law, all using the training scaling law ℒ^train​(μrec,D)\hat{\mathcal{L}}\_{\mathrm{train}}(\mu\_{\text{rec}},D) from [Section 5.2](#S5.SS2 "5.2 Looping as an Orthogonal Scaling Axis in Training ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models") as the irreducible floor:

1. (a)

   ℒ^train+Z⋅exp⁡(−z⋅μrec−γ⋅T)\hat{\mathcal{L}}\_{\mathrm{train}}+Z\cdot\exp\!\bigl(-z\cdot\mu\_{\text{rec}}^{-\gamma}\cdot T\bigr) (learned γ\gamma, 3 params)
2. (b)

   ℒ^train+Z⋅exp⁡(−z/μrec⋅T)\hat{\mathcal{L}}\_{\mathrm{train}}+Z\cdot\exp\!\bigl(-z/\mu\_{\text{rec}}\cdot T\bigr) (γ=1\gamma=1, 2 params)
3. (c)

   ℒ^train+Z⋅exp⁡(−z⋅T)\hat{\mathcal{L}}\_{\mathrm{train}}+Z\cdot\exp\!\bigl(-z\cdot T\bigr) (no conditioning, 2 params)

|  |  |  |  |
| --- | --- | --- | --- |
|  | Z​e−z​μ−γ​TZe^{-z\mu^{-\gamma}T} | Z​e−z​T/μZe^{-zT/\mu} (γ=1\gamma{=}1) | Z​e−z​TZe^{-zT} (no μ\mu) |
| Train (isoFLOP) | | | |
| 140M | 0.001116 | 0.001177 | 0.003253 |
| 370M | 0.000229 | 0.000283 | 0.001438 |
| Test (held-out, μrec=8\mu\_{\text{rec}}{=}8) | | | |
| 140M | 0.000207 | 0.000212 | 0.000266 |
| 370M | 0.000133 | 0.000131 | 0.000189 |

Table 14: Ablation of μrec\mu\_{\text{rec}} conditioning in the unified test-time law. We report total Huber loss on the isoFLOP training set and on held-out Table 5 models (μrec=8\mu\_{\text{rec}}=8, fixed data budget). Lower is better.

As shown in [Table 14](#A12.T14 "In Conditioning on Training Recurrence. ‣ Appendix L Fitting Parametric Functions to Test-Time Looping ‣ Parcae: Scaling Laws For Stable Looped Language Models"), removing μrec\mu\_{\text{rec}} conditioning entirely increases training error by 3.5×3.5\times and held-out error by ∼33%{\sim}33\%, confirming that the decay rate must depend on training depth (also obvious from looking at [Figure 8](#S5.F8 "In Setup. ‣ 5.3 Test-Time Scaling Laws of Parcae ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models")). The learned γ\gamma offers a modest improvement (∼8%{\sim}8\%) over γ=1\gamma=1 on the training set, with fitted values of γ=1.19\gamma=1.19 (140M) and γ=1.17\gamma=1.17 (370M) consistent across scales; on held-out models, the two are indistinguishable. We therefore adopt γ=1\gamma=1 for simplicity, yielding the unified law:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ^unified​(T∣μrec,D)=E+X⋅N​(μrec)−x+Y⋅D−y⏟Training Law Floor ​ℒ^train​(μrec,D)+Z⋅exp⁡(−z⋅Tμrec)⏟Test-Time Decay\hat{\mathcal{L}}\_{\mathrm{unified}}(T\mid\mu\_{\text{rec}},D)=\underbrace{E+X\cdot N(\mu\_{\text{rec}})^{-x}+Y\cdot D^{-y}}\_{\text{Training Law Floor }\hat{\mathcal{L}}\_{\mathrm{train}}(\mu\_{\text{rec}},D)}+\underbrace{Z\cdot\exp\!\left(-\frac{z\cdot T}{\mu\_{\text{rec}}}\right)}\_{\text{Test-Time Decay}} |  | (8) |

where the test-time term depends on the ratio T/μrecT/\mu\_{\text{rec}}, i.e., the fraction of training depth used at inference.

#### Testing the Unified Parametric Fit.

To evaluate generalization, we use the unified law fitted on isoFLOP data to predict the test-time scaling curves of held-out 140M and 370M Parcae models from [Section 5.1](#S5.SS1 "5.1 Parcae Improves End-to-End Quality ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models"), which were trained on fixed data budgets and are a completely out-of-distribution setting. As shown in [Figure 26](#A12.F26 "In Testing the Unified Parametric Fit. ‣ Appendix L Fitting Parametric Functions to Test-Time Looping ‣ Parcae: Scaling Laws For Stable Looped Language Models"), the unified fit (orange) predicts validation loss within 0.85–1.31% average error. When the training law floor is replaced with the empirical loss at T=μrecT=\mu\_{\text{rec}} (oracle, blue), error drops to 0.10–0.17%, confirming that the test-time decay is faithfully captured and the residual error is attributable to the training law’s ∼1%{\sim}1\% extrapolation gap.

![Refer to caption](/html/2604.12946/assets/x29.png)


Figure 26: Out-of-Distribution Prediction of Unified Parametric Fit.
We visualize the prediction of our unified parametric fit (orange) and an oracle fit using the empirical loss at T=μrecT=\mu\_{\text{rec}} for ℒ^train\widehat{\mathcal{L}}\_{\text{train}} (blue) against empirical validation loss with increasing TT for models trained in [Section 5.1](#S5.SS1 "5.1 Parcae Improves End-to-End Quality ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models").

## Appendix M Extended Evaluation Details and Setup

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Category | Task | Type | Shots | Core |
| Understanding | HellaSwag [zellers2019hellaswag] (0-shot) | MC | 0 | ✓ |
| HellaSwag [zellers2019hellaswag] (10-shot) | MC | 10 | ✓ |
| Lambada [paperno2016lambada] | LM | 0 | ✓ |
| Winograd WSC [wsc:2015] | S | 0 | ✓ |
| WinoGrande [sakaguchi2021winogrande] | S | 0 | ✓ |
| BIG-Bench Language ID [srivastava2023imitationgamequantifyingextrapolating] | MC | 10 | ✓ |
| BIG-Bench Conlang Translation [srivastava2023imitationgamequantifyingextrapolating] | LM | 0 |  |
| BIG-Bench Conceptual Comb. [srivastava2023imitationgamequantifyingextrapolating] | MC | 10 |  |
| World Knowl. | Jeopardy [kaggle200000Jeopardy] | LM | 10 | ✓ |
| BIG-Bench QA WikiData [srivastava2023imitationgamequantifyingextrapolating] | LM | 10 | ✓ |
| ARC-Easy [clark2018think] | MC | 10 | ✓ |
| ARC-Challenge [clark2018think] | MC | 10 | ✓ |
| MMLU (0-shot) [hendrycks2021measuringmassivemultitasklanguage] | MC | 0 |  |
| MMLU (5-shot) [hendrycks2021measuringmassivemultitasklanguage] | MC | 5 |  |
| BIG-Bench Misconceptions [srivastava2023imitationgamequantifyingextrapolating] | MC | 10 |  |
| Commonsense | COPA [gordon-etal-2012-semeval] | MC | 0 | ✓ |
| CommonsenseQA [talmor2019commonsenseqaquestionansweringchallenge] | MC | 10 | ✓ |
| PIQA [bisk2020piqa] | MC | 10 | ✓ |
| OpenBookQA [mihaylov2018suitarmorconductelectricity] | MC | 0 | ✓ |
| SIQA [sap2019socialiqacommonsensereasoningsocial] | MC | 10 |  |
| BIG-Bench Novel Concepts [srivastava2023imitationgamequantifyingextrapolating] | MC | 10 |  |
| BIG-Bench Strange Stories [srivastava2023imitationgamequantifyingextrapolating] | MC | 10 |  |
| BIG-Bench Strategy QA [srivastava2023imitationgamequantifyingextrapolating] | MC | 10 |  |
| Symbolic / Math | BIG-Bench Dyck Languages [srivastava2023imitationgamequantifyingextrapolating] | LM | 10 | ✓ |
| AGI Eval LSAT AR [zhong2021arlsat] | MC | 3 | ✓ |
| BIG-Bench CS Algorithms [srivastava2023imitationgamequantifyingextrapolating] | LM | 10 | ✓ |
| BIG-Bench Operators [srivastava2023imitationgamequantifyingextrapolating] | LM | 10 | ✓ |
| BIG-Bench Repeat Copy Logic [srivastava2023imitationgamequantifyingextrapolating] | LM | 10 | ✓ |
| BIG-Bench Elementary Math QA [srivastava2023imitationgamequantifyingextrapolating] | MC | 10 |  |
| BIG-Bench Logical Deduction [srivastava2023imitationgamequantifyingextrapolating] | MC | 10 |  |
| Simple Arithmetic (no spaces) [llmfoundry] | LM | 10 |  |
| Simple Arithmetic (w/ spaces) [llmfoundry] | LM | 10 |  |
| MathQA [amini2019mathqa] | MC | 10 |  |
| LogiQA [liu2020logiqa] | MC | 10 |  |
| Reading Comp. | SQuAD [rajpurkar2016squad100000questionsmachine] | LM | 10 | ✓ |
| CoQA [reddy2019coqaconversationalquestionanswering] | LM | 0 | ✓ |
| BoolQ [clark2019boolq] | MC | 10 | ✓ |
| PubMedQA (labeled) [jin2019pubmedqa] | LM | 10 |  |
| AGI Eval LSAT RC [zhong2023agieval] | MC | 3 |  |
| AGI Eval LSAT LR [wang2022lsat] | MC | 3 |  |
| AGI Eval SAT English [zhong2023agieval] | MC | 3 |  |
| BIG-Bench Understanding Fables [srivastava2023imitationgamequantifyingextrapolating] | MC | 10 |  |
| Safety | Winogender MC (Female) [rudinger2018genderbiascoreferenceresolution] | MC | 10 |  |
| Winogender MC (Male) [rudinger2018genderbiascoreferenceresolution] | MC | 10 |  |
| Enterprise PII Classification | MC | 10 |  |
| BBQ [parrish2022bbqhandbuiltbiasbenchmark] | MC | 3 |  |

Table 15: Full list of downstream evaluation Tasks marked with ✓ are included in the Core [li2025datacomplmsearchgenerationtraining]; all tasks are included in Core-Extended [li2025datacomplmsearchgenerationtraining]. Type indicates the scoring method: MC (multiple choice, lowest mean NLL), S (schema-based NLL), or LM (exact greedy match).

We include a complete list of benchmarks used for evaluation in [Table 15](#A13.T15 "In Appendix M Extended Evaluation Details and Setup ‣ Parcae: Scaling Laws For Stable Looped Language Models"). For our results in [Section 5](#S5 "5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models") where we are comparing against baseline transformers, we run each benchmark with three different seeds, as this changes both the initial recurrent state and the in-context few-shot examples.

## Appendix N Expanded Results For Fixed-Depth and Looping IsoFLOP Comparison

We included an expanded form of [Table 6](#S5.T6 "In Setup. ‣ 5.2 Looping as an Orthogonal Scaling Axis in Training ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models") to ensure reproducibility, which additionally includes error bars in [Table 16](#A14.T16 "In Appendix N Expanded Results For Fixed-Depth and Looping IsoFLOP Comparison ‣ Parcae: Scaling Laws For Stable Looped Language Models").

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | FLOPs |  | Optimal μrec∗\mu\_{\mathrm{rec}}^{\*} | | Fixed-Depth (μrec=1\mu\_{\mathrm{rec}}=1) | |
|  | (×1018\times 10^{18}) | μrec∗\mu\_{\mathrm{rec}}^{\*} | Core | Core Ext. | Core | Core Ext. |
| 140M | 11 | 2 | 7.6±0.37.6\pm 0.3 | 5.7±0.55.7\pm 0.5 | 7.9±0.2\mathbf{7.9\pm 0.2} | 6.1±0.1\mathbf{6.1\pm 0.1} |
| 22 | 2 | 9.0±0.29.0\pm 0.2 | 6.2±0.16.2\pm 0.1 | 10.5±0.1\mathbf{10.5\pm 0.1} | 6.4±0.2\mathbf{6.4\pm 0.2} |
| 44 | 4 | 11.2±0.0\mathbf{11.2\pm 0.0} | 8.4±0.2\mathbf{8.4\pm 0.2} | 10.7±0.110.7\pm 0.1 | 8.1±0.38.1\pm 0.3 |
| 88 | 6 | 10.5±0.110.5\pm 0.1 | 7.8±0.2\mathbf{7.8\pm 0.2} | 11.8±0.2\mathbf{11.8\pm 0.2} | 7.7±0.27.7\pm 0.2 |
| 1616 | 8 | 14.6±0.1\mathbf{14.6\pm 0.1} | 9.8±0.4\mathbf{9.8\pm 0.4} | 13.0±0.213.0\pm 0.2 | 8.8±0.48.8\pm 0.4 |
| 6464 | 10 | 16.2±0.2\mathbf{16.2\pm 0.2} | 11.0±0.1\mathbf{11.0\pm 0.1} | 15.0±0.215.0\pm 0.2 | 9.5±0.49.5\pm 0.4 |
| 370M | 3232 | 4 | 15.2±0.115.2\pm 0.1 | 10.1±0.210.1\pm 0.2 | 16.8±0.1\mathbf{16.8\pm 0.1} | 11.2±0.4\mathbf{11.2\pm 0.4} |
| 6464 | 6 | 18.1±0.2\mathbf{18.1\pm 0.2} | 11.6±0.211.6\pm 0.2 | 18.1±0.1\mathbf{18.1\pm 0.1} | 12.1±0.2\mathbf{12.1\pm 0.2} |
| 128128 | 6 | 20.1±0.1\mathbf{20.1\pm 0.1} | 13.0±0.1\mathbf{13.0\pm 0.1} | 18.1±0.118.1\pm 0.1 | 12.0±0.112.0\pm 0.1 |

Table 16: Expanded Core Scores Comparison of Looping Optimal Frontier over Purely Scaling Data. Including variance bars now.

## Appendix O Expanded Setup For Training and Test-Time Scaling Laws

For our scaling laws experiments, we train models under two setups: (1) an isoFLOP training setup where we train models with variable amounts of μrec\mu\_{\text{rec}}, but with fixed FLOP and parameter budgets, and (2) where we vary μrec\mu\_{\text{rec}}, but keep data and parameters constant. Additionally, for our unified scaling laws experiments, we reuse the models trained in setup (1) and then evaluate them with varying amounts of test-time recurrences. All of the experiments use the same exact experimental setup for Transformers described in [Appendix Q](#A17 "Appendix Q Hyperparameters and Training Details ‣ Parcae: Scaling Laws For Stable Looped Language Models") and [Appendix P](#A16 "Appendix P Model Definitions ‣ Parcae: Scaling Laws For Stable Looped Language Models") (i.e., using a nanochat [nanochat]). We will discuss each experiment in detail below.

#### (1) Setup for IsoFLOP Experiments.

For each parameter count (140M and 370M), we fix the total training FLOP budget and vary μrec∈{2,4,6,8,10,12}\mu\_{\text{rec}}\in\{2,4,6,8,10,12\}, adjusting the number of training tokens to maintain the FLOP budget (i.e., increasing μrec\mu\_{\text{rec}} reduces the token budget proportionally). For 140M models, we use FLOP budgets of {1,2,4,8,16,64}×1018\{1,2,4,8,16,64\}\times 10^{18}; for 370M models, {32,64,128}×1018\{32,64,128\}\times 10^{18}.
This yields 36 and 18 trained models for 140M and 370M, respectively. Each model is evaluated on a held-out validation set at T=μrecT=\mu\_{\text{rec}}.
We use these validation losses to fit the parametric training scaling law ℒ^train​(μrec,𝒟)\widehat{\mathcal{L}}\_{\text{train}}(\mu\_{\text{rec}},\mathcal{D}) and to extract optimal μrec∗\mu\_{\text{rec}}^{\*} at each FLOP budget via parabolic fits.
Additionally, we train fixed-depth (μrec=1\mu\_{\text{rec}}=1) Parcae models at each FLOP budget to serve as baselines for the looping frontier comparison. Expanded details of the predicted frontiers calculation can be found in [Appendix N](#A14 "Appendix N Expanded Results For Fixed-Depth and Looping IsoFLOP Comparison ‣ Parcae: Scaling Laws For Stable Looped Language Models").

#### (2) Setup for Test-Time Saturation and Power Laws.

To study how test-time recurrence scales quality, we train 140M and 370M Parcae models under a fixed data budget of 11.2B tokens with μrec∈{2,4,6,8,10,12}\mu\_{\text{rec}}\in\{2,4,6,8,10,12\}. Each model is then evaluated on a held-out validation set at test-time recurrences T∈{1,2,3,…,24}T\in\{1,2,3,\ldots,24\}, yielding a saturation curve per μrec\mu\_{\text{rec}}. We fit an independent exponential decay law ℒ​(T)=ℒ∞+Z⋅exp⁡(−z⋅T)\mathcal{L}(T)=\mathcal{L}\_{\infty}+Z\cdot\exp(-z\cdot T) to each curve following the procedure in Section [L](#A12 "Appendix L Fitting Parametric Functions to Test-Time Looping ‣ Parcae: Scaling Laws For Stable Looped Language Models"). We additionally evaluate the Parcae models from [Section 5.1](#S5.SS1 "5.1 Parcae Improves End-to-End Quality ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models") (140M–1.3B, trained at μ​rec=8\mu{\text{rec}}=8) at test-time recurrences T∈{1,…,16}T\in\{1,\ldots,16\} to verify that the saturation behavior is consistent across model sizes.

#### (3) Setup for Unified Scaling Law.

To fit the unified scaling law ([Equation 4](#S5.E4 "In Setup. ‣ 5.3 Test-Time Scaling Laws of Parcae ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models")), we reuse the isoFLOP models from setup (1) and evaluate each at test-time recurrences T∈{1,2,4,6,8,10,12,16,20,24}T\in\{1,2,4,6,8,10,12,16,20,24\}, yielding approximately 540 data points per model size. We fit all 8 parameters of [Equation 4](#S5.E4 "In Setup. ‣ 5.3 Test-Time Scaling Laws of Parcae ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models") jointly on this data using Huber loss on the log loss with L-BFGS over 1,000 random restarts. To validate, we evaluate the unified fit on held-out 140M and 370M Parcae models from [Section 5.1](#S5.SS1 "5.1 Parcae Improves End-to-End Quality ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models"), which were trained on fixed data budgets outside the isoFLOP sweep, at test-time recurrences T∈{1,…,16}T\in\{1,\ldots,16\}.

## Appendix P Model Definitions

As we perform experiments in two setups, one following prior work in recurrent depth models [geiping\_scaling\_2025] and one following a strong baseline transformer [nanochat], we separate the model definitions into [Section P.1](#A16.SS1 "P.1 Model Definitions for RDM and Parcae Comparison ‣ Appendix P Model Definitions ‣ Parcae: Scaling Laws For Stable Looped Language Models") and [Section P.2](#A16.SS2 "P.2 Model Definitions for Transformer and Parcae Comparison ‣ Appendix P Model Definitions ‣ Parcae: Scaling Laws For Stable Looped Language Models"), respectively.

### P.1 Model Definitions for RDM and Parcae Comparison

In this section, we will discuss the model configuration used for models in [Section 5.1](#S5.SS1 "5.1 Parcae Improves End-to-End Quality ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models") for RDMs [geiping\_scaling\_2025]. For all 𝒫\mathcal{P}, ℛ\mathcal{R}, and 𝒞\mathcal{C} modules, we follow geiping\_scaling\_2025, and use standard, causal self-attention and gated SwiGLU MLP [shazeer2020gluvariantsimprovetransformer]. For attention, we use RoPE [su2023roformerenhancedtransformerrotary] with θ=50000\theta=50000 and for normalization we use RMSNorm [zhang2019rootmeansquarelayer]. We use Pre-Norm transformer blocks for all modules within Parcae, and follow takase2025spikemorestabilizingpretraining, initializing weights using 𝒩​(0,25​d)\mathcal{N}(0,\frac{2}{5d}), where dd is the model dimension.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Parcae-100M | Parcae-350M | RDM-100M | RDM-350M |
| Parameters | 114,242,560 | 378,558,464 | 114,242,560 | 382,765,056 |
| Layers in 𝒫\mathcal{P} | 1 | 1 | 1 | 1 |
| Layers in 𝒞\mathcal{C} | 1 | 1 | 1 | 1 |
| Layers in ℛ\mathcal{R} | 1 | 2 | 1 | 2 |
| dmodeld\_{\text{model}} | 1,024 | 2,048 | 1,024 | 2,048 |
| dintermediated\_{\text{intermediate}} | 3,520 | 7,040 | 3,520 | 7,040 |
| Attention | Causal Self-Attention [vaswani2023attentionneed] | | | |
| MLP | SwiGLU [elfwing2017sigmoidweightedlinearunitsneural, shazeer2020gluvariantsimprovetransformer] | | | |
| Pos. Embed. | RoPE [su2023roformerenhancedtransformerrotary] | | | |
| Vocab Size | 65,536 | | | |
| Norm | RMS-Norm [zhang2019rootmeansquarelayer] | | | |
| Init | Scaled [takase2025spikemorestabilizingpretraining] | | | |
| Tied Embeddings | Yes | | | |
| State Init. | like-init [geiping\_scaling\_2025] | | | |
| μrec\mu\_{\text{rec}} | 16 | 8 | 16 | 8 |
| Backprop Depth | 8 | 4 | 8 | 4 |
| Sampling | Poisson Distribution | | | |

Table 17: Model definitions of both Parcae and baseline residual-norm RDMs [geiping\_scaling\_2025].

### P.2 Model Definitions for Transformer and Parcae Comparison

In this section, we will discuss the model definitions used for our experiments in [Section 5.1](#S5.SS1 "5.1 Parcae Improves End-to-End Quality ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models") for Transformers. Our architecture is derived from nanochat, while being slightly adapted to fit with GPT2 [radford2019language] style parameter classes. Model definitions of both Parcae and baseline Transformers can be found in [Table 18](#A16.T18 "In P.2 Model Definitions for Transformer and Parcae Comparison ‣ Appendix P Model Definitions ‣ Parcae: Scaling Laws For Stable Looped Language Models"), while the difference in parameter count can be found in [Table 19](#A16.T19 "In P.2 Model Definitions for Transformer and Parcae Comparison ‣ Appendix P Model Definitions ‣ Parcae: Scaling Laws For Stable Looped Language Models").666We note that Parcae does technically introduce additional parameters over baseline Transformers; however, they are negligible in comparison to total parameter counts.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  |  | Small (140M) | Medium (370M) | Large (770M) | XLarge (1.3B) |
| Architecture | Layers (Transformer) | 6 | 12 | 18 | 24 |
| Layers in 𝒫\mathcal{P} (Parcae) | 2 | 4 | 6 | 8 |
| Layers in ℛ\mathcal{R} (Parcae) | 2 | 4 | 6 | 8 |
| Layers in 𝒞\mathcal{C} (Parcae) | 2 | 4 | 6 | 8 |
| dmodeld\_{\text{model}} | 768 | 1,024 | 1,280 | 1,536 |
| dintermediated\_{\text{intermediate}} | 3,072 | 4,096 | 5,120 | 6,144 |
| Attention Heads | 6 | 8 | 10 | 12 |
| Head Dimension | 128 | 128 | 128 | 128 |
| Shared Details | Attention | Causal Self-Attention [vaswani2023attentionneed] w/ QK-Norm [henry2020querykeynormalizationtransformers] | | | |
| MLP | ReLU2\text{ReLU}^{2} [zhang2024relu2winsdiscoveringefficient] | | | |
| Value Embeddings | Gated, alternating layers [tian2023resformerscalingvitsmultiresolution] | | | |
| Pos. Embed. | RoPE (θ=50,000\theta{=}50{,}000) [su2023roformerenhancedtransformerrotary] | | | |
| Vocab Size | 32,768 | | | |
| Norm | RMS-Norm (Pre-Norm) [zhang2019rootmeansquarelayer] | | | |
| Context Length | 2,048 | | | |
| Bias | None | | | |
| Init. | Scaled-zero [takase2025spikemorestabilizingpretraining, nanochat] | | | |
| Tied Embeddings | Yes | | | |
| Parcae | Injection | Diagonal | | | |
| State Init. | like-init [geiping\_scaling\_2025] | | | |
| μrec\mu\_{\text{rec}} | 8 | | | |
| Backprop Depth | 4 | | | |
| Sampling | Poisson (truncated, per-sequence) | | | |

Table 18: Model definitions of both Parcae and baseline Transformers.



|  | Small (140M) | Medium (370M) | Large (770M) | XLarge (1.3B) |
| --- | --- | --- | --- | --- |
| Transformer Parameters | 143,141,184 | 385,903,104 | 773,375,040 | 1,333,868,544 |
| Parcae Parameters | 144,323,136 | 388,003,328 | 776,655,680 | 1,338,591,744 |
| Additional Parameters | 1,181,952 | 2,100,224 | 3,280,640 | 4,723,200 |
| Additional (%) | 0.83% | 0.54% | 0.42% | 0.35% |

Table 19: Comparison of Parcae and Transformer parameter count.

## Appendix Q Hyperparameters and Training Details

Again, as we perform experiments in two setups, one following prior work in recurrent depth models [geiping\_scaling\_2025] and one following a strong baseline transformer [nanochat], we separate the hyperparameter configurations into [Section Q.1](#A17.SS1 "Q.1 Hyperparameters for Parcae and RDM Comparison ‣ Appendix Q Hyperparameters and Training Details ‣ Parcae: Scaling Laws For Stable Looped Language Models") and [Section Q.2](#A17.SS2 "Q.2 Hyperparameters for Parcae and Transformer Comparison ‣ Appendix Q Hyperparameters and Training Details ‣ Parcae: Scaling Laws For Stable Looped Language Models"), respectively.

### Q.1 Hyperparameters for Parcae and RDM Comparison

In this section, we will discuss the hyperparameter configuration used in [Section 5.1](#S5.SS1 "5.1 Parcae Improves End-to-End Quality ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models") for RDMs [geiping\_scaling\_2025]. We train with a warm-up and cool-down (4096 steps following [geiping\_scaling\_2025]) and a constant learning rate (η=4×10−3\eta=4\times 10^{-3} for 100M models and η=2×10−3\eta=2\times 10^{-3} for 350M models) [pmlr-v202-geiping23a, Zhai\_2022\_CVPR]. As our optimizer, we use Adam with decoupled weight regularization (β1=0.9,β2​0.95\beta\_{1}=0.9,\beta\_{2}0.95) [kingma2017adammethodstochasticoptimization, loshchilov2019decoupledweightdecayregularization], using update clipping [wortsman2023stable] and removing the ϵ\epsilon constant [everett2024scalingexponentsparameterizationsoptimizers]. Gradients above 1 are clipped.

For learning rates, we swept our selection of learning rates for RDMs [geiping\_scaling\_2025], over the search space [2​e−4,4​e−4,6​e−4,8​e−4,1​e−3][2e-4,4e-4,6e-4,8e-4,1e-3], approximately using 10 to 1 token to parameter ratio. We then select the best learning rate for each scale (e.g., 4e-4 for 100M and 2e-4 for 350M). We perform no learning rate sweep for Parcae, using the best learning rate for RDMs [geiping\_scaling\_2025]. We do this so that our comparison between Parcae and prior methods is fair, as we observed significant divergence in training for RDMs based on learning rate (see [Appendix F](#A6 "Appendix F Additional Stability Ablations ‣ Parcae: Scaling Laws For Stable Looped Language Models")).
We stipulate that Parcae models would likely perform better with stronger hyperparameter tuning.

### Q.2 Hyperparameters for Parcae and Transformer Comparison

In this section, we will discuss the hyperparameter configuration used in [Section 5.1](#S5.SS1 "5.1 Parcae Improves End-to-End Quality ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models") for Transformers. We use a simplified version of nanochat [nanochat], with the main difference being a simplified learning rate selection. Specifically, in nanochat [nanochat], different parameter groups have different learning rates (e.g., MLP, value-embeddings, and projection head have different learning rates), which we simplify into just two parameter groups, one for AdamW [kingma2017adammethodstochasticoptimization, loshchilov2019decoupledweightdecayregularization] and one for Muon [jordan2024muon]. A breakdown of which parameters are placed with each of these groups follows nanochat [nanochat], and can be found in [Table 20](#A17.T20 "In Q.2 Hyperparameters for Parcae and Transformer Comparison ‣ Appendix Q Hyperparameters and Training Details ‣ Parcae: Scaling Laws For Stable Looped Language Models").

| Optimizer | Parameters |
| --- | --- |
| AdamW [kingma2017adammethodstochasticoptimization] | Token embeddings (wte) |
| LM head (lm\_head) |
| Normalization layers (RMSNorm) |
| Value embedding gates (ve\_gate) |
| All 1D parameters |
| AdamW [kingma2017adammethodstochasticoptimization] (Parcae only) | Injection parameters (𝑨\bm{A}, Δ\Delta, 𝑩\bm{B}) |
| Readout projection (𝑪\bm{C}) |
| Muon [jordan2024muon] | Attention projections (WQW\_{Q}, WKW\_{K}, WVW\_{V}, WOW\_{O}) |
| MLP weights (WfcW\_{\text{fc}}, WprojW\_{\text{proj}}) |

Table 20: Optimizer parameter group assignment for Parcae and baseline Transformers.

As we simplify the learning rate setup used in nanochat [nanochat], we perform a rigorous hyperparameter sweep of baseline Transformers to create the strongest baseline. Specifically, for small and medium models, we form a sweep over {3​e−4,5​e−4,6​e−4,8​e−4,1​e−3,1.5​e−3,2​e−3,3​e−3,4​e−3,8​e−3,1​e−2,1.5​e−2,2​e−2}\{3e-4,5e-4,6e-4,8e-4,1e-3,1.5e-3,2e-3,3e-3,4e-3,8e-3,1e-2,1.5e-2,2e-2\} for AdamW learning rates and a sweep over {3​e−4,5​e−4,1​e−3,2​e−3,4​e−3,8​e−3,1​e−2,1.5​e−2,2​e−2}\{3e-4,5e-4,1e-3,2e-3,4e-3,8e-3,1e-2,1.5e-2,2e-2\} for Muon learning rates using 1:20 param to token ratios for the search, where we find that for both models 8​e−38e-3 works best for both sizes and optimizers. For large and xlarge transformer models, we perform a constrained sweep of learning rate in {2​e−3,3​e−3,4​e−3,6​e−3,8​e−3}\{2e-3,3e-3,4e-3,6e-3,8e-3\} for AdamW [kingma2017adammethodstochasticoptimization], while keeping the Muon learning rate fixed at 8​e−38e{-3}, using a 1:7 parameter to token ratio, where we find that a learning rate of 6​e−36e-3 performs the best. We perform *no learning rate sweeps for Parcae*, to ensure that we are giving the fairest comparison. We expect that there likely exists a more optimal learning rate for Parcae, which could further improve performance.

Following nanochat [nanochat], we use a fixed learning rate, with no warmup and 50% cooldown. For Muon [jordan2024muon], we use five iterations of polar express orthogonalization [amsel2025polarexpressoptimalmatrix], factored variance reductions [si2025adamuonadaptivemuonoptimizer], and cautious weight decay [chen2026cautiousweightdecay]. We train with BF16 mixed precision. For our data pipeline, we use a BOS-aligned dataloader with BestFit-Crop packing [ding2024fewertruncationsimprovelanguage] and training on FineWeb-edu [penedo2024finewebdatasetsdecantingweb]. We clip gradients above 1.
A table of hyperparameter details can be found in [Table 21](#A17.T21 "In Q.2 Hyperparameters for Parcae and Transformer Comparison ‣ Appendix Q Hyperparameters and Training Details ‣ Parcae: Scaling Laws For Stable Looped Language Models").

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Small (140M) | Medium (370M) | Large (770M) | XLarge (1.3B) |
| Training Tokens | 11.2B | 29.6B | 61.6B | 104B |
| Batch Size (sequences) | 256 | 256 | 256 | 256 |
| Sequence Length | 2,048 | 2,048 | 2,048 | 2,048 |
| Precision | bf16-mixed | | | |
| AdamW LR | 8×10−38\times 10^{-3} | 8×10−38\times 10^{-3} | 6×10−36\times 10^{-3} | 6×10−36\times 10^{-3} |
| AdamW (β1,β2)(\beta\_{1},\beta\_{2}) | (0.8,0.95)(0.8,0.95) | | | |
| AdamW Weight Decay | 0.00.0 | | | |
| AdamW ϵ\epsilon | 10−1010^{-10} | | | |
| Muon LR | 8×10−38\times 10^{-3} | | | |
| Muon Momentum | 0.950.95 | | | |
| Muon Weight Decay | 0.20.2 (linear decay to 0) | | | |
| Muon Orthogonalization Steps | 5 | | | |
| LR Schedule | Fixed (0% warmup, 50% cooldown) | | | |
| Gradient Clipping | 1.01.0 | | | |

Table 21: Hyperparameter used from training Parcae and Transformer models in [Section 5.1](#S5.SS1 "5.1 Parcae Improves End-to-End Quality ‣ 5 Results ‣ Parcae: Scaling Laws For Stable Looped Language Models") for Transformers.

Lastly, following nanochat [nanochat], we train our own tokenizer, which we use for all models. Details of the tokenizer training and setup can be found in [Appendix R](#A18 "Appendix R Tokenizer Training ‣ Parcae: Scaling Laws For Stable Looped Language Models").

## Appendix R Tokenizer Training

We train a custom BPE tokenizer with a vocabulary size of 32,768 using the HuggingFace tokenizers library. We follow a GPT-4 style configuration [openai2024gpt4technicalreport]: byte-level BPE with byte fallback, no text normalization, and a GPT-4 style pre-tokenization split pattern. The tokenizer is trained on 2 billion characters from the FineWeb-Edu training set [penedo2024finewebdatasetsdecantingweb], with individual documents capped at 10,000 characters. We define three special tokens: <|bos|>, <|eos|>, and <|pad|>. A small comparison of our tokenizer used in our experiments with others can be found in [Table 22](#A18.T22 "In Appendix R Tokenizer Training ‣ Parcae: Scaling Laws For Stable Looped Language Models").

| Tokenizer | Vocab Size | Bytes/Token ↑\uparrow | |
| --- | --- | --- | --- |
|  |  | Train | Val |
| GPT-2 (gpt2) | 50,257 | 4.67 | 4.63 |
| GPT-4 (cl100k) | 100,277 | 4.81 | 4.76 |
| Ours | 32,768 | 4.72 | 4.65 |

Table 22: Compression ratio (bytes per token) on FineWeb-Edu for tokenizer used in training.

[◄](/html/2604.12945)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2604.12946)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2604.12946)
[View original  
on arXiv](https://arxiv.org/abs/2604.12946)[►](/html/2604.12947)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Tue May 5 13:59:37 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
