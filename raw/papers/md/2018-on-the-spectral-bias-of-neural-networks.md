---
arxiv: '1806.08734'
authors:
- Nasim Rahaman
- Aristide Baratin
- Devansh Arpit
- Felix Draxler
- Min Lin
- Fred A. Hamprecht
- Yoshua Bengio
- Aaron Courville
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: On the Spectral Bias of Neural Networks
url: http://arxiv.org/abs/1806.08734v3
year: 2018
---

[1806.08734] On the Spectral Bias of Neural Networks















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



# On the Spectral Bias of Neural Networks

Nasim Rahaman
  
Aristide Baratin
  
Devansh Arpit
  
Felix Draxler
  
Min Lin
  
Fred A. Hamprecht
  
Yoshua Bengio
  
Aaron Courville

###### Abstract

Neural networks are known to be a class of highly expressive functions able to fit even random input-output mappings with 100%percent100100\% accuracy.
In this work we present properties of neural networks that complement this aspect of expressivity. By using tools from Fourier analysis, we highlight a learning bias of deep networks towards low frequency functions – i.e. functions that vary globally without local fluctuations – which manifests itself as a frequency-dependent learning speed. Intuitively, this property is in line with the observation that over-parameterized networks prioritize learning simple patterns that generalize across data samples. We also investigate the role of the shape of the data manifold by presenting empirical and theoretical evidence that, somewhat counter-intuitively, learning higher frequencies gets *easier* with increasing manifold complexity.

Machine Learning, ICML

## 1 Introduction

The remarkable success of deep neural networks at generalizing to natural data
is at odds with the traditional notions of model complexity and their empirically demonstrated ability to fit arbitrary random data to perfect accuracy (Zhang et al., [2017a](#bib.bib39); Arpit et al., [2017](#bib.bib2)). This has prompted recent investigations of possible implicit regularization mechanisms inherent in the learning process which induce a bias towards low complexity solutions (Neyshabur et al., [2014](#bib.bib25); Soudry et al., [2017](#bib.bib34); Poggio et al., [2018](#bib.bib28); Neyshabur et al., [2017](#bib.bib26)).

In this work, we take a slightly shifted view on implicit regularization by suggesting that
low-complexity functions are *learned faster* during training by gradient descent.
We expose this bias by taking a closer look at neural networks through the lens of Fourier analysis. While they can approximate arbitrary functions, we find that these networks prioritize learning the low frequency modes, a phenomenon we call the *spectral bias*. This bias manifests itself not just in the process of learning, but also in the parameterization of the model itself:
in fact, we show that the lower frequency components of trained networks are more robust to random parameter perturbations. Finally, we also expose and analyze the rather intricate interplay between the spectral bias and the geometry of the data manifold by showing that high frequencies get easier to learn when the data lies on a lower-dimensional manifold of complex shape embedded in the input space of the model.
We focus the discussion on networks with rectified linear unit (ReLU) activations, whose continuous piece-wise linear structure enables an analytic treatment.

### Contributions111Code: https://github.com/nasimrahaman/SpectralBias

1. 1.

   We exploit the continuous piecewise-linear structure of ReLU networks to evaluate
   its Fourier spectrum (Section [2](#S2 "2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks")).
2. 2.

   We find empirical evidence of a *spectral bias*: i.e. lower frequencies are learned first. We also show that lower frequencies are more robust to random perturbations of the network parameters (Section [3](#S3 "3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks")).
3. 3.

   We study the role of the shape of the data manifold: we show how complex manifold shapes can facilitate the learning of higher frequencies and develop a theoretical understanding of this behavior
   (Section [4](#S4 "4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks")).

## 2 Fourier analysis of ReLU networks

### 2.1 Preliminaries

Throughout the paper we call ‘ReLU network’ a scalar function f:ℝd↦ℝ:𝑓maps-tosuperscriptℝ𝑑ℝf:\mathbb{R}^{d}\mapsto\mathbb{R} defined by a neural network with L𝐿L hidden layers of widths d1,⋯​dL

subscript𝑑1⋯subscript𝑑𝐿d\_{1},\cdots d\_{L}
and a single output neuron:

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(𝐱)=(T(L+1)∘σ∘T(L)∘⋯∘σ∘T(1))​(𝐱)𝑓𝐱superscript𝑇𝐿1𝜎superscript𝑇𝐿⋯𝜎superscript𝑇1𝐱f(\mathbf{x})=\left(T^{(L+1)}\circ\sigma\circ T^{(L)}\circ\cdots\circ\sigma\circ T^{(1)}\right)(\mathbf{x}) |  | (1) |

where each T(k):ℝdk−1→ℝdk:superscript𝑇𝑘→superscriptℝsubscript𝑑𝑘1superscriptℝsubscript𝑑𝑘T^{(k)}:\mathbb{R}^{d\_{k-1}}\rightarrow\mathbb{R}^{d\_{k}} is an affine function
(d0=dsubscript𝑑0𝑑d\_{0}=d and dL+1=1subscript𝑑𝐿11d\_{L+1}=1) and σ​(𝐮)i=max⁡(0,ui)𝜎subscript𝐮𝑖0subscript𝑢𝑖\sigma(\mathbf{u})\_{i}=\max(0,u\_{i}) denotes the ReLU activation function acting elementwise on a vector 𝐮=(u1,⋯​un)𝐮subscript𝑢1⋯subscript𝑢𝑛\mathbf{u}=(u\_{1},\cdots u\_{n}). In the standard basis, T(k)​(𝐱)=W(k)​𝐱+𝐛(k)superscript𝑇𝑘𝐱superscript𝑊𝑘𝐱superscript𝐛𝑘T^{(k)}(\mathbf{x})=W^{(k)}\mathbf{x}+\mathbf{b}^{(k)} for some weight matrix W(k)superscript𝑊𝑘W^{(k)} and bias vector 𝐛(k)superscript𝐛𝑘\mathbf{b}^{(k)}.

ReLU networks are known to be continuous piece-wise linear (CPWL) functions, where the linear regions are convex polytopes (Raghu et al., [2016](#bib.bib30); Montufar et al., [2014](#bib.bib24); Zhang et al., [2018](#bib.bib41); Arora et al., [2018](#bib.bib1)).
Remarkably, the converse also holds:
every CPWL function can be represented by a ReLU network (Arora et al., [2018](#bib.bib1), Theorem 2.1), which in turn endows ReLU networks with universal approximation properties.
Given the ReLU network f𝑓f from Eqn. [1](#S2.E1 "In 2.1 Preliminaries ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks"), we
can make the piecewise linearity explicit by writing,

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(𝐱)=∑ϵ1Pϵ​(𝐱)​(Wϵ​𝐱+𝐛ϵ)𝑓𝐱subscriptitalic-ϵsubscript1subscript𝑃italic-ϵ𝐱subscript𝑊italic-ϵ𝐱subscript𝐛italic-ϵf(\mathbf{x})=\sum\_{\epsilon}1\_{P\_{\epsilon}}(\mathbf{x})\,(W\_{\epsilon}\mathbf{x}+\mathbf{b}\_{\epsilon}) |  | (2) |

where ϵitalic-ϵ\epsilon is an index for the linear regions Pϵsubscript𝑃italic-ϵP\_{\epsilon}
and 1Pϵsubscript1subscript𝑃italic-ϵ1\_{P\_{\epsilon}} is the indicator function on Pϵsubscript𝑃italic-ϵP\_{\epsilon}. As shown in Appendix [B](#A2 "Appendix B The Continuous Piecewise Linear Structure of Deep ReLU Networks ‣ On the Spectral Bias of Neural Networks") in more detail, each region corresponds to an *activation pattern*222We adopt the terminology of Raghu et al. ([2016](#bib.bib30)); Montufar et al. ([2014](#bib.bib24)). of all hidden neurons of the network, which is a binary vector with components conditioned on the sign of the input of the respective neuron. The 1×d1𝑑1\times d matrix Wϵsubscript𝑊italic-ϵW\_{\epsilon} is given by

|  |  |  |  |
| --- | --- | --- | --- |
|  | Wϵ=W(L+1)​Wϵ(L)​⋯​Wϵ(1)subscript𝑊italic-ϵsuperscript𝑊𝐿1subscriptsuperscript𝑊𝐿italic-ϵ⋯subscriptsuperscript𝑊1italic-ϵW\_{\epsilon}=W^{(L+1)}W^{(L)}\_{\epsilon}\cdots W^{(1)}\_{\epsilon} |  | (3) |

where Wϵ(k)subscriptsuperscript𝑊𝑘italic-ϵW^{(k)}\_{\epsilon} is obtained from the original weight W(k)superscript𝑊𝑘W^{(k)} by setting its jt​hsuperscript𝑗𝑡ℎj^{th} column to zero whenever the neuron j𝑗j of the kt​hsuperscript𝑘𝑡ℎk^{th} layer is inactive.

### 2.2 Fourier Spectrum

In the following, we study the structure of ReLU networks in terms of their Fourier representation, f​(𝐱):=(2​π)d/2​∫f~​(𝐤)​ei​𝐤⋅𝐱​𝐝𝐤assign𝑓𝐱superscript2𝜋𝑑2~𝑓𝐤superscript𝑒⋅𝑖𝐤𝐱𝐝𝐤f(\mathbf{x}):=(2\pi)^{\nicefrac{{d}}{{2}}}\int\tilde{f}(\mathbf{k})\,e^{i\mathbf{k}\cdot\mathbf{x}}\mathbf{dk}, where f~​(𝐤):=∫f​(𝐱)​e−i​𝐤⋅𝐱​𝐝𝐱assign~𝑓𝐤𝑓𝐱superscript𝑒⋅𝑖𝐤𝐱𝐝𝐱\tilde{f}(\mathbf{k}):=\int f(\mathbf{x})\,e^{-i\mathbf{k}\cdot\mathbf{x}}\mathbf{dx} is the Fourier transform333Note that general ReLU networks need not be squared integrable: for instance, the class of two-layer ReLU networks represent an arrangement of hyperplanes (Montufar et al., [2014](#bib.bib24)) and hence grow linearly as x→∞→𝑥x\rightarrow\infty. In such cases, the Fourier transform is to be understood in the sense of tempered distributions acting on rapidly decaying smooth functions ϕitalic-ϕ\phi as ⟨f~,ϕ⟩=⟨f,ϕ~⟩

~𝑓italic-ϕ

𝑓~italic-ϕ\langle\tilde{f},\phi\rangle=\langle f,\tilde{\phi}\rangle. See Appendix [C](#A3 "Appendix C Fourier Analysis of ReLU Networks ‣ On the Spectral Bias of Neural Networks") for a formal treatment.
. Lemmas [1](#Thmlemma1 "Lemma 1. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks") and [2](#Thmlemma2 "Lemma 2. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks") yield the explicit form of the Fourier components (we refer to Appendix [C](#A3 "Appendix C Fourier Analysis of ReLU Networks ‣ On the Spectral Bias of Neural Networks") for the proofs and technical details).

###### Lemma 1.

The Fourier transform of ReLU networks decomposes as,

|  |  |  |  |
| --- | --- | --- | --- |
|  | f~​(𝐤)=i​∑ϵWϵ​𝐤k2​1~Pϵ​(𝐤)~𝑓𝐤𝑖subscriptitalic-ϵsubscript𝑊italic-ϵ𝐤superscript𝑘2subscript~1subscript𝑃italic-ϵ𝐤\tilde{f}(\mathbf{k})=i\sum\_{\epsilon}\frac{W\_{\epsilon}\mathbf{k}}{k^{2}}\,\tilde{1}\_{P\_{\epsilon}}(\mathbf{k}) |  | (4) |

where k=‖𝐤‖𝑘norm𝐤k=\|\mathbf{k}\| and 1~P​(𝐤)=∫Pe−i​𝐤⋅𝐱​𝐝𝐱subscript~1𝑃𝐤subscript𝑃superscript𝑒⋅𝑖𝐤𝐱𝐝𝐱\tilde{1}\_{P}(\mathbf{k})=\int\_{P}e^{-i\mathbf{k}\cdot\mathbf{x}}\mathbf{dx} is the Fourier transform of the indicator function of P𝑃P.

The Fourier transform of the indicator over linear regions appearing in Eqn. [4](#S2.E4 "In Lemma 1. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks") are fairly intricate mathematical objects. Diaz et al. ([2016](#bib.bib10)) develop an elegant procedure for evaluating it in arbitrary dimensions via a recursive application of Stokes theorem. We describe this procedure in detail444We also generalize the construction to tempered distributions. in Appendix [C.2](#A3.SS2 "C.2 Fourier Transform of Polytopes ‣ Appendix C Fourier Analysis of ReLU Networks ‣ On the Spectral Bias of Neural Networks"), and present here its main corollary.

###### Lemma 2.

Let P𝑃P be a full dimensional polytope in ℝdsuperscriptℝ𝑑\mathbb{R}^{d}. Its Fourier spectrum takes the form:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1~P​(𝐤)=∑n=0dDn​(𝐤)​1Gn​(𝐤)knsubscript~1𝑃𝐤superscriptsubscript𝑛0𝑑subscript𝐷𝑛𝐤subscript1subscript𝐺𝑛𝐤superscript𝑘𝑛\tilde{1}\_{P}(\mathbf{k})=\sum\_{n=0}^{d}\frac{D\_{n}(\mathbf{k})1\_{G\_{n}}(\mathbf{k})}{k^{n}} |  | (5) |

where Gnsubscript𝐺𝑛G\_{n} is the union of n𝑛n-dimensional subspaces that are orthogonal to some n𝑛n-codimensional face of P𝑃P, Dn:ℝd→ℂ:subscript𝐷𝑛→superscriptℝ𝑑ℂD\_{n}:\mathbb{R}^{d}\rightarrow\mathbb{C} is in Θ​(1)​(k→∞)Θ1→𝑘\Theta(1)\,(k\rightarrow\infty) and 1Gnsubscript1subscript𝐺𝑛1\_{G\_{n}} the indicator over Gnsubscript𝐺𝑛G\_{n}.

Lemmas [1](#Thmlemma1 "Lemma 1. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks"), [2](#Thmlemma2 "Lemma 2. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks")
together yield the main result of this section.

###### Theorem 1.

The Fourier components of the ReLU network fθsubscript𝑓𝜃f\_{\theta} with parameters θ𝜃\theta is given by the rational function:

|  |  |  |  |
| --- | --- | --- | --- |
|  | f~θ​(𝐤)=∑n=0dCn​(θ,𝐤)​1Hnθ​(𝐤)kn+1subscript~𝑓𝜃𝐤superscriptsubscript𝑛0𝑑subscript𝐶𝑛𝜃𝐤subscript1subscriptsuperscript𝐻𝜃𝑛𝐤superscript𝑘𝑛1\tilde{f}\_{\theta}(\mathbf{k})=\sum\_{n=0}^{d}\frac{C\_{n}(\theta,\mathbf{k})1\_{H^{\theta}\_{n}}(\mathbf{k})}{k^{n+1}} |  | (6) |

where Hnθsubscriptsuperscript𝐻𝜃𝑛H^{\theta}\_{n} is the union of n𝑛n-dimensional subspaces that are orthogonal to some n𝑛n-codimensional faces of some polytope Pϵsubscript𝑃italic-ϵP\_{\epsilon} and Cn​(⋅,θ):ℝd→ℂ:subscript𝐶𝑛⋅𝜃→superscriptℝ𝑑ℂC\_{n}(\cdot,\theta):\mathbb{R}^{d}\rightarrow\mathbb{C} is
Θ​(1)​(k→∞)Θ1→𝑘\Theta(1)\,(k\rightarrow\infty).

Note that Eqn [6](#S2.E6 "In Theorem 1. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks") applies to general ReLU networks with arbitrary width and depth555Symmetries that might arise due to additional assumptions can be used to further develop Eqn [6](#S2.E6 "In Theorem 1. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks"), see e.g. Eldan & Shamir ([2016](#bib.bib12)) for 2-layer networks..

Discussion. We make the following two observations. First, the spectral decay of ReLU networks is highly anisotropic in large dimensions. In almost all directions of ℝdsuperscriptℝ𝑑\mathbb{R}^{d}, we have a k−d−1superscript𝑘𝑑1k^{-d-1} decay. However, the decay can be as slow as k−2superscript𝑘2k^{-2} in specific directions
orthogonal to the d−1𝑑1d-1 dimensional faces bounding the linear regions666Note that such a rate is *not* guaranteed by piecewise smoothness alone. For instance, the function |x|𝑥\sqrt{|x|} is continuous and smooth everywhere except at x=0𝑥0x=0, yet it decays as k−1.5superscript𝑘1.5k^{-1.5} in the Fourier domain..

Second, the numerator in Eqn [6](#S2.E6 "In Theorem 1. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks") is bounded by Nf​Lfsubscript𝑁𝑓subscript𝐿𝑓N\_{f}L\_{f} (cf. Appendix [C.3](#A3.SS3 "C.3 On Theorem 1 ‣ Appendix C Fourier Analysis of ReLU Networks ‣ On the Spectral Bias of Neural Networks")), where Nfsubscript𝑁𝑓N\_{f} is the number of linear regions and Lf=maxϵ⁡‖Wϵ‖subscript𝐿𝑓subscriptitalic-ϵnormsubscript𝑊italic-ϵL\_{f}=\max\_{\epsilon}\|W\_{\epsilon}\| is the Lipschitz constant of the network.
Further, the Lipschitz constant Lfsubscript𝐿𝑓L\_{f} can be bounded as (cf. Appendix [C.6](#A3.SS6 "C.6 Proof of the Lipschtiz bound ‣ Appendix C Fourier Analysis of ReLU Networks ‣ On the Spectral Bias of Neural Networks")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | Lf≤∏k=1L+1‖W(k)‖≤‖θ‖∞L+1​d​∏k=1Ldksubscript𝐿𝑓superscriptsubscriptproduct𝑘1𝐿1normsuperscript𝑊𝑘superscriptsubscriptnorm𝜃𝐿1𝑑superscriptsubscriptproduct𝑘1𝐿subscript𝑑𝑘L\_{f}\leq\prod\_{k=1}^{L+1}\|W^{(k)}\|\leq\|\theta\|\_{\infty}^{L+1}\sqrt{d}\prod\_{k=1}^{L}d\_{k} |  | (7) |

where ∥⋅∥\|\cdot\| is the spectral norm and ∥⋅∥∞\|\cdot\|\_{\infty} the max norm, and dksubscript𝑑𝑘d\_{k} is the number of units in the k𝑘k-th layer. This makes the bound on Lfsubscript𝐿𝑓L\_{f} scale exponentially in depth and polynomial in width. As for the number Nfsubscript𝑁𝑓N\_{f} of linear regions, Montufar et al. ([2014](#bib.bib24)) and Raghu et al. ([2016](#bib.bib30)) obtain tight bounds that exhibit the same scaling behaviour (Raghu et al., [2016](#bib.bib30), Theorem 1). In Appendix [A.5](#A1.SS5 "A.5 Qualitative Ablation over Architectures ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks"), we qualitatively ablate over the depth and width of the network to expose how this reflects on the Fourier spectrum of the network.

![Refer to caption](/html/1806.08734/assets/x1.png)

![Refer to caption](/html/1806.08734/assets/x2.png)

(a) Equal Amplitudes

![Refer to caption](/html/1806.08734/assets/x3.png)

![Refer to caption](/html/1806.08734/assets/x4.png)

(b) Increasing Amplitudes

Figure 1: Left (a, b): Evolution of the spectrum (x-axis for frequency) during training (y-axis). The colors show the measured amplitude of the network spectrum at the corresponding frequency, normalized by the target amplitude at the same frequency (i.e. |f~ki|/Aisubscript~𝑓subscript𝑘𝑖subscript𝐴𝑖|\tilde{f}\_{k\_{i}}|/A\_{i}) and the colorbar is clipped between 0 and 1. Right (a, b): Evolution of the spectral norm (y-axis) of each layer during training (x-axis). Figure-set (a) shows the setting where all frequency components in the target function have the same amplitude, and (b) where higher frequencies have larger amplitudes. Gist: We find that even when higher frequencies have larger amplitudes, the model prioritizes learning lower frequencies first. We also find that the spectral norm of weights increases as the model fits higher frequency, which is what we expect from Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks").

## 3 Lower Frequencies are Learned First

We now present experiments
showing that networks tend to fit *lower frequencies first* during training. We refer to this phenomenon as the *spectral bias*, and discuss it in light of the results of
Section [2](#S2 "2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks").

![Refer to caption](/html/1806.08734/assets/figures/learntfunc_eq_amp_iter_100.png)


(a) Iteration 100

![Refer to caption](/html/1806.08734/assets/figures/learntfunc_eq_amp_iter_1000.png)


(b) Iteration 1000

![Refer to caption](/html/1806.08734/assets/figures/learntfunc_eq_amp_iter_10000.png)


(c) Iteration 10000

![Refer to caption](/html/1806.08734/assets/figures/learntfunc_eq_amp_iter_80000.png)


(d) Iteration 80000

Figure 2: The learnt function (green) overlayed on the target function (blue) as the training progresses. The target function is a superposition of sinusoids of frequencies κ=(5,10,…,45,50)𝜅510…4550\kappa=(5,10,...,45,50), equal amplitudes and randomly sampled phases.

### 3.1 Synthetic Experiments

###### Experiment 1.

The setup is as follows777More experimental details and additional plots are provided in Appendix [A.1](#A1.SS1 "A.1 Experiment 1 ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks").:
Given frequencies κ=(k1,k2,…)𝜅subscript𝑘1subscript𝑘2…\kappa=(k\_{1},k\_{2},...) with corresponding amplitudes α=(A1,A2,…)𝛼subscript𝐴1subscript𝐴2…\alpha=(A\_{1},A\_{2},...), and phases ϕ=(φ1,φ2,…)italic-ϕsubscript𝜑1subscript𝜑2…\phi=(\varphi\_{1},\varphi\_{2},...), we consider the mapping λ:[0,1]→ℝ:𝜆→01ℝ\lambda:[0,1]\rightarrow\mathbb{R} given by

|  |  |  |  |
| --- | --- | --- | --- |
|  | λ​(z)=∑iAi​sin⁡(2​π​ki​z+φi).𝜆𝑧subscript𝑖subscript𝐴𝑖2𝜋subscript𝑘𝑖𝑧subscript𝜑𝑖\lambda(z)=\sum\_{i}A\_{i}\sin(2\pi k\_{i}z+\varphi\_{i}). |  | (8) |

A 6-layer deep 256-unit wide ReLU network fθsubscript𝑓𝜃f\_{\theta} is trained to regress λ𝜆\lambda with κ=(5,10,…,45,50)𝜅510…4550\kappa=(5,10,...,45,50) and N=200𝑁200N=200 input samples spaced equally over [0,1]01[0,1];
its spectrum f~θ​(k)subscript~𝑓𝜃𝑘\tilde{f}\_{\theta}(k) in expectation over φi∼U​(0,2​π)similar-tosubscript𝜑𝑖𝑈02𝜋\varphi\_{i}\sim U(0,2\pi) is monitored as training progresses. In the first setting, we set equal amplitude Ai=1subscript𝐴𝑖1A\_{i}=1 for all frequencies and in the second setting, the amplitude increases from A1=0.1subscript𝐴10.1A\_{1}=0.1 to A10=1subscript𝐴101A\_{10}=1. Figure [1](#S2.F1 "Figure 1 ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks") shows the normalized magnitudes |f~θ​(ki)|/Aisubscript~𝑓𝜃subscript𝑘𝑖subscript𝐴𝑖|\tilde{f}\_{\theta}(k\_{i})|/A\_{i} at various frequencies, as training progresses with full-batch gradient descent. Further, Figure [2](#S3.F2 "Figure 2 ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks") shows the learned function at intermediate training iterations. The result is that
lower frequencies (i.e. smaller kisubscript𝑘𝑖k\_{i}’s) are regressed first, regardless of their amplitudes.

###### Experiment 2.

![Refer to caption](/html/1806.08734/assets/figures/better_normalized_robustness.png)


Figure 3: Normalized spectrum of the model (x-axis for frequency, colorbar for magnitude) with perturbed parameters as a function of parameter perturbation (y-axis). The colormap is clipped between 0 and 1. We observe that the lower frequencies are more robust to parameter perturbations than the higher frequencies.

Our goal here is to illustrate a phenomenon that complements the one highlighted above: lower frequencies are more *robust* to parameter perturbations. The set up is the same as in Experiment [1](#Thmexperiment1 "Experiment 1. ‣ 3.1 Synthetic Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks"). The network is trained to regress a target function with frequencies κ=(10,15,20,…,45,50)𝜅101520…4550\kappa=(10,15,20,...,45,50) and amplitudes Ai=1​∀isubscript𝐴𝑖1for-all𝑖A\_{i}=1\,\forall\,i.
After convergence to θ∗superscript𝜃\theta^{\*}, we consider random (isotropic) perturbations θ=θ∗+δ​θ^𝜃superscript𝜃𝛿^𝜃\theta=\theta^{\*}+\delta\hat{\theta} of given magnitude δ𝛿\delta,
where θ^^𝜃\hat{\theta} is a random unit vector in parameter space. We evaluate the network function fθsubscript𝑓𝜃f\_{\theta} at the perturbed parameters, and compute the magnitude of its discrete Fourier transform at frequencies kisubscript𝑘𝑖k\_{i} to obtain |f~θ​(ki)|subscript~𝑓𝜃subscript𝑘𝑖|\tilde{f}\_{\theta}({k\_{i}})|. We also average over 100 samples of θ^^𝜃\hat{\theta} to obtain |f~𝔼​θ​(ki)|subscript~𝑓𝔼𝜃subscript𝑘𝑖|\tilde{f}\_{\mathbb{E}\theta}({k\_{i}})|, which we normalize by |f~θ⁣∗​(ki)|subscript~𝑓

𝜃subscript𝑘𝑖|\tilde{f}\_{\theta\*}({k\_{i}})|. Finally, we average over the phases ϕitalic-ϕ\phi (see Eqn [8](#S3.E8 "In Experiment 1. ‣ 3.1 Synthetic Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks")). The result, shown in Figure [3](#S3.F3 "Figure 3 ‣ Experiment 2. ‣ 3.1 Synthetic Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks"), demonstrates that higher frequencies are significantly less robust than the lower ones, guiding the intuition that expressing higher frequencies requires the parameters to be finely-tuned to work together. In other words, parameters that contribute towards expressing high-frequency components occupy a small volume in the parameter space. We formalize this in Appendix [D](#A4 "Appendix D Volume of High-Frequency Parameters in Parameter Space ‣ On the Spectral Bias of Neural Networks").

![Refer to caption](/html/1806.08734/assets/figures/mnist_noise_freq_0_1_amp_var_val_ch_n.png)


(a) k=0.1𝑘0.1k=0.1

![Refer to caption](/html/1806.08734/assets/figures/mnist_noise_freq_1_amp_var_val_ch_n.png)


(b) k=1𝑘1k=1

![Refer to caption](/html/1806.08734/assets/figures/mnist_noise_freq_var_amp_0_5_val_ch_n.png)


(c) β=0.5𝛽0.5\beta=0.5

![Refer to caption](/html/1806.08734/assets/figures/mnist_noise_freq_var_amp_1_val_ch_n.png)


(d) β=1.𝛽1\beta=1.

Figure 4: (a,b,c,d): Validation curves for various settings of noise amplitude β𝛽\beta and frequency k𝑘k. Corresponding training curves can be found in Figure [11](#A1.F11 "Figure 11 ‣ A.3 Experiment 3 ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks") in appendix [A.3](#A1.SS3 "A.3 Experiment 3 ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks").
Gist: Low frequency noise affects the network more than their high-frequency counterparts. Further, for high-frequency noise, one finds that the validation loss dips early in the training. Both these observations are explained by the fact that network readily fit lower frequencies, but learn higher frequencies later in the training.

Discussion . Multiple theoretical aspects may underlie these observations. First, for a fixed architecture, recall that the numerator in Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks") is888The tightness of this bound is verified empirically in appendix [A.5](#A1.SS5 "A.5 Qualitative Ablation over Architectures ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks"). 𝒪​(Lf)𝒪subscript𝐿𝑓\mathcal{O}(L\_{f}) (where Lfsubscript𝐿𝑓L\_{f} is the Lipschitz constant of the function). However, Lfsubscript𝐿𝑓L\_{f} is bounded by the parameter norm, which can only increase gradually during training by gradient descent. This leads to the higher frequencies being learned999This assumes that the Lipschitz constant of the (noisy) target function is larger than that of the network at initialization. late in the optimization process. To confirm that the bound indeed increases as the model fits higher frequencies, we plot in Fig [1](#S2.F1 "Figure 1 ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks") the spectral norm of weights of each layer during training for both cases of constant and increasing amplitudes.

Second (cf. Appendix [C.4](#A3.SS4 "C.4 Spectral Decay Rate of the Parameter Gradient ‣ Appendix C Fourier Analysis of ReLU Networks ‣ On the Spectral Bias of Neural Networks")), the exact form of the Fourier spectrum yields that for a fixed direction 𝐤^^𝐤\hat{\mathbf{k}}, the spectral decay rate of the parameter gradient ∂f~/∂θ~𝑓𝜃\nicefrac{{\partial\tilde{f}}}{{\partial\theta}} is at most one exponent of k𝑘k lower than that of f~~𝑓\tilde{f}. If for a fixed 𝐤^^𝐤\hat{\mathbf{k}} we have f~=𝒪​(k−Δ−1)~𝑓𝒪superscript𝑘Δ1\tilde{f}=\mathcal{O}(k^{-\Delta-1}) where 1≤Δ≤d1Δ𝑑1\leq\Delta\leq d, we obtain for the residual h=f−λℎ𝑓𝜆h=f-\lambda and (continuous) training step t𝑡t:

|  |  |  |  |
| --- | --- | --- | --- |
|  | |d​h~​(𝐤)d​t|=|d​f~​(𝐤)d​t|=|d​f~​(𝐤)d​θ|⏟𝒪​(k−Δ)​|d​θd​t|⏞|η⋅d​ℒ/d​θ|=𝒪​(k−Δ)𝑑~ℎ𝐤𝑑𝑡𝑑~𝑓𝐤𝑑𝑡subscript⏟𝑑~𝑓𝐤𝑑𝜃𝒪superscript𝑘Δsuperscript⏞𝑑𝜃𝑑𝑡⋅𝜂𝑑ℒ𝑑𝜃𝒪superscript𝑘Δ\displaystyle\left|\frac{d\tilde{h}(\mathbf{k})}{dt}\right|=\left|\frac{d\tilde{f}(\mathbf{k})}{dt}\right|=\underbrace{\left|\frac{d\tilde{f}(\mathbf{k})}{d\theta}\right|}\_{\mathcal{O}(k^{-\Delta})}\overbrace{\left|\frac{d\theta}{dt}\right|}^{\left|\eta\cdot\nicefrac{{d\mathcal{L}}}{{d\theta}}\right|}=\mathcal{O}(k^{-\Delta}) |  | (9) |

where we use the fact that d​θ/d​t𝑑𝜃𝑑𝑡\nicefrac{{d\theta}}{{dt}} is just the learning rate times the parameter gradient of the loss which is independent101010Note however that the loss term might involve a sum or an integral over all frequencies, but the summation is over a different variable. of k𝑘k, and assume that the target function λ𝜆\lambda is fixed. Eqn [9](#S3.E9 "In 3.1 Synthetic Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks") shows that the rate of change of the residual decays with increasing frequency, which is what we find in Experiment [1](#Thmexperiment1 "Experiment 1. ‣ 3.1 Synthetic Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks").

### 3.2 Real-Data Experiments

While Experiments [1](#Thmexperiment1 "Experiment 1. ‣ 3.1 Synthetic Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks") and [2](#Thmexperiment2 "Experiment 2. ‣ 3.1 Synthetic Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks") establish the spectral bias by explicitly evaluating the Fourier coefficients, doing so becomes prohibitively expensive for larger d𝑑d (e.g. on MNIST). To tackle this, we propose the following set of experiments to measure the effect of spectral bias indirectly on MNIST.

###### Experiment 3.

In this experiment, we investigate how the validation performance dependent on the frequency of noise added to the training target. We find that the best validation performance on MNIST is particularly insensitive to the magnitude of high-frequency noise, yet it is adversely affected by low-frequency noise. We consider a target (binary) function τ0:X→{0,1}:subscript𝜏0→𝑋01\tau\_{0}:X\rightarrow\{0,1\} defined on the space X=[0,1]784𝑋superscript01784X=[0,1]^{784} of MNIST inputs.
Samples {𝐱i,τ0​(𝐱i)}isubscriptsubscript𝐱𝑖subscript𝜏0subscript𝐱𝑖𝑖\{\mathbf{x}\_{i},\tau\_{0}(\mathbf{x}\_{i})\}\_{i} form a subset of the MNIST dataset comprising samples 𝐱isubscript𝐱𝑖\mathbf{x}\_{i} belonging to two classes. Let ψk​(𝐱)subscript𝜓𝑘𝐱\psi\_{k}(\mathbf{x}) be a *noise function*:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ψk​(𝐱)=sin⁡(k​‖𝐱‖)subscript𝜓𝑘𝐱𝑘norm𝐱\psi\_{k}(\mathbf{x})=\sin(k\|\mathbf{x}\|) |  | (10) |

corresponding to a *radial wave* defined on the 784784784-dimensional input space111111The rationale behind using a radial wave is that it induces oscillations (simultaneously) along all spatial directions. Another viable option is to induce oscillations along the principle axes of the data: we have verified that the key trends of interest are preserved.. The final target function τksubscript𝜏𝑘\tau\_{k} is then given by τk=τ0+β​ψksubscript𝜏𝑘subscript𝜏0𝛽subscript𝜓𝑘\tau\_{k}=\tau\_{0}+\beta\psi\_{k}, where β𝛽\beta is the effective amplitude of the noise. We fit the same network as in Experiment [1](#Thmexperiment1 "Experiment 1. ‣ 3.1 Synthetic Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks") to the target τksubscript𝜏𝑘\tau\_{k} with the MSE loss. In the first set of experiments, we ablate over k𝑘k for a pair of fixed β𝛽\betas, while in the second set we ablate over β𝛽\beta for a pair of fixed k𝑘ks. In Figure [4](#S3.F4 "Figure 4 ‣ 3.1 Synthetic Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks"), we show the respective validation loss curves, where the validation set is obtained by evaluating τ0subscript𝜏0\tau\_{0} on a separate subset of the data, i.e. {𝐱j,τ0​(𝐱j)}jsubscriptsubscript𝐱𝑗subscript𝜏0subscript𝐱𝑗𝑗\{\mathbf{x}\_{j},\tau\_{0}(\mathbf{x}\_{j})\}\_{j}. Figure [11](#A1.F11 "Figure 11 ‣ A.3 Experiment 3 ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks") (in appendix [A.3](#A1.SS3 "A.3 Experiment 3 ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks")) shows the respective training curves.

Discussion.
The profile of the loss curves varies significantly with the frequency of noise added to the target. In Figure [4(a)](#S3.F4.sf1 "In Figure 4 ‣ 3.1 Synthetic Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks"), we see that the validation performance is adversely affected by the amplitude of the low-frequency noise, whereas Figure [4(b)](#S3.F4.sf2 "In Figure 4 ‣ 3.1 Synthetic Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks") shows that the amplitude of high-frequency noise does not significantly affect the best validation score. This is explained by the fact that the network readily fits the noise signal if it is low frequency, whereas the higher frequency noise is only fit later in the training. In the latter case, the dip in validation score early in the training is when the network has learned the low frequency true target function τ0subscript𝜏0\tau\_{0}; the remainder of the training is spent learning the higher-frequencies in the training target τ𝜏\tau, as we shall see in the next experiment. Figures [4(c)](#S3.F4.sf3 "In Figure 4 ‣ 3.1 Synthetic Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks") and [4(d)](#S3.F4.sf4 "In Figure 4 ‣ 3.1 Synthetic Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks") confirm that the dip in validation score exacerbates for increasing frequency of the noise. Further, we observe that for higher frequencies (e.g. k=0.5𝑘0.5k=0.5), increasing the amplitude β𝛽\beta does not significantly degrade the best performance at the dip, confirming that the network is fairly robust to the amplitude of high-frequency noise.

Finally, we note that the dip in validation score was also observed by Arpit et al. ([2017](#bib.bib2)) with i.i.d. noise121212Recall that i.i.d. noise is white-noise, which has a constant Fourier spectrum magnitude in expectation, i.e. it also contains high-frequency components. in a classification setting.

###### Experiment 4.

To investigate the dip observed in Experiment [3](#Thmexperiment3 "Experiment 3. ‣ 3.2 Real-Data Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks"), we now take a more direct approach by considering a generalized notion of frequency. To that end, we project the network function to the space spanned by the orthonormal eigenfunctions φnsubscript𝜑𝑛\varphi\_{n} of the Gaussian RBF kernel (Braun et al., [2006](#bib.bib6)). These eigenfunctions φnsubscript𝜑𝑛\varphi\_{n} (sorted by decreasing eigenvalues) resemble sinusoids (Fasshauer, [2011](#bib.bib13)), and the index n𝑛n can be thought of as being a proxy for the frequency, as can be seen from Figure [6](#S3.F6 "Figure 6 ‣ 3.2 Real-Data Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks") (see Appendix [A.4](#A1.SS4 "A.4 Experiment 4 ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks") for additional details and supporting plots). While we will call f~​[n]~𝑓delimited-[]𝑛\tilde{f}[n] as the spectrum of the function f𝑓f, it should be understood as f~​[n]=⟨fℋ,φn⟩ℋ~𝑓delimited-[]𝑛subscript

subscript𝑓ℋsubscript𝜑𝑛
ℋ\tilde{f}[n]=\langle f\_{\mathcal{H}},\varphi\_{n}\rangle\_{\mathcal{H}}, where fℋ∈span​{φn}nsubscript𝑓ℋspansubscriptsubscript𝜑𝑛𝑛f\_{\mathcal{H}}\in\text{span}\{\varphi\_{n}\}\_{n} and fℋ​(𝐱i)=f​(𝐱i)subscript𝑓ℋsubscript𝐱𝑖𝑓subscript𝐱𝑖f\_{\mathcal{H}}(\mathbf{x}\_{i})=f(\mathbf{x}\_{i}) on the MNIST samples 𝐱i∈Xsubscript𝐱𝑖𝑋\mathbf{x}\_{i}\in X. This allows us to define a noise function as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ψγ​(𝐱)=∑nN(nN)γ​φn​(𝐱)subscript𝜓𝛾𝐱superscriptsubscript𝑛𝑁superscript𝑛𝑁𝛾subscript𝜑𝑛𝐱\psi\_{\gamma}(\mathbf{x})=\sum\_{n}^{N}\left(\frac{n}{N}\right)^{\gamma}\varphi\_{n}(\mathbf{x}) |  | (11) |

where N𝑁N is the number of available samples and γ=2𝛾2\gamma=2. Like in Experiment [3](#Thmexperiment3 "Experiment 3. ‣ 3.2 Real-Data Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks"), the target function is given by τ=τ0+β​ψ𝜏subscript𝜏0𝛽𝜓\tau=\tau\_{0}+\beta\psi, and the same network is trained to regress τ𝜏\tau. Figure [5](#S3.F5 "Figure 5 ‣ 3.2 Real-Data Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks") shows the (generalized) spectrum τ𝜏\tau and τ0subscript𝜏0\tau\_{0}, and that of f𝑓f as training progresses. Figure [13](#A1.F13 "Figure 13 ‣ A.4.1 Loss Curves Accompanying Figure 5 ‣ A.4 Experiment 4 ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks") (in appendix) shows the corresponding dip in validation loss, where the validation set is same as the training set but with true target function τ0subscript𝜏0\tau\_{0} instead of the noised target τ𝜏\tau.

![Refer to caption](/html/1806.08734/assets/figures/kernel_hfn_bias.png)


Figure 5: Spectrum of the network as it is trained on MNIST target with high-frequency noise (*Noised Target*). We see that the network fits the true target at around the 200200200th iteration, which is when the validation score dips (Figure [13](#A1.F13 "Figure 13 ‣ A.4.1 Loss Curves Accompanying Figure 5 ‣ A.4 Experiment 4 ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks") in appendix).

![Refer to caption](/html/1806.08734/assets/figures/rbf_freq_of_eigvecs.png)


Figure 6: Spectrum (x-axis for frequency, colorbar for magnitude) of the n𝑛n-th (y-axis) eigenvector of the Gaussian RBF kernel matrix Ki​j=k​(𝐱i,𝐱j)subscript𝐾𝑖𝑗𝑘subscript𝐱𝑖subscript𝐱𝑗K\_{ij}=k(\mathbf{x}\_{i},\mathbf{x}\_{j}), where the sample set is {xi∈[0,1]}i=150superscriptsubscriptsubscript𝑥𝑖01𝑖150\{x\_{i}\in[0,1]\}\_{i=1}^{50} is N=50𝑁50N=50 uniformly spaced points between 00 and 111 and k𝑘k is the Gaussian RBF kernel function. Gist: The eigenfunctions with increasing n𝑛n roughly correspond to sinusoids of increasing frequency. Refer to Appendix [A.4](#A1.SS4 "A.4 Experiment 4 ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks") for more details.

Discussion. From Figure [5](#S3.F5 "Figure 5 ‣ 3.2 Real-Data Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks"), we learn that the drop in validation score observed in Figure [4](#S3.F4 "Figure 4 ‣ 3.1 Synthetic Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks") is exactly when the higher-frequencies of the noise signal are yet to be learned. As the network gradually learns the higher frequency eigenfunctions, the validation loss increases while the training loss continues to decrease. Thus these experiments show that the phenomenon of spectral bias persists on non-synthetic data and in high dimensional input spaces.

## 4 Not all Manifolds are Learned Equal

In this section, we investigate subtleties that arise when the data lies on a lower dimensional manifold embedded in the higher dimensional input space of the model. We find that the *shape* of the data-manifold impacts the learnability of high frequencies in a non-trivial way. As we shall see, this is because low frequency functions in the input space may have high frequency components when restricted to lower dimensional manifolds of complex shapes.
We demonstrate results in an illustrative minimal setting131313We include additional experiments on MNIST and CIFAR-10 in appendices [A.6](#A1.SS6 "A.6 MNIST: A Proof of Concept ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks") and [A.7](#A1.SS7 "A.7 Cifar-10: It’s All Connected ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks")., free from unwanted confounding factors, and present a theoretical analysis of the phenomenon.

Manifold hypothesis. We consider the case where the data lies on a lower dimensional *data manifold* ℳ⊂ℝdℳsuperscriptℝ𝑑{\mathcal{M}}\subset\mathbb{R}^{d} embedded in input space (Goodfellow et al., [2016](#bib.bib14)), which we assume to be the image
γ​([0,1]m)𝛾superscript01𝑚\gamma([0,1]^{m}) of some injective mapping γ:[0,1]m→ℝd:𝛾→superscript01𝑚superscriptℝ𝑑\gamma:[0,1]^{m}\rightarrow\mathbb{R}^{d} defined on a lower dimensional latent space [0,1]msuperscript01𝑚[0,1]^{m}. Under this hypothesis and in the context of the standard regression problem, a target function τ:ℳ→ℝ:𝜏→ℳℝ\tau:{\mathcal{M}}\rightarrow\mathbb{R} defined on the data manifold can be identified with a function λ=τ∘γ𝜆𝜏𝛾\lambda=\tau\circ\gamma defined on the latent space. Regressing τ𝜏\tau is therefore equivalent to finding f:ℝd→ℝ:𝑓→superscriptℝ𝑑ℝf:\mathbb{R}^{d}\rightarrow\mathbb{R} such that f∘γ𝑓𝛾f\circ\gamma matches λ𝜆\lambda. Further, assuming that the data probability distribution μ𝜇\mu supported on ℳℳ{\mathcal{M}} is induced by γ𝛾\gamma from the uniform distribution U𝑈U in the latent space [0,1]msuperscript01𝑚[0,1]^{m}, the mean square error can be expressed as:

|  |  |  |
| --- | --- | --- |
|  | MSEμ(𝐱)​[f,τ]=𝔼𝐱∼μ​|f​(𝐱)−τ​(𝐱)|2=subscriptsuperscriptMSE𝐱𝜇𝑓𝜏subscript𝔼similar-to𝐱𝜇superscript𝑓𝐱𝜏𝐱2absent\displaystyle\textup{MSE}^{(\mathbf{x})}\_{\mu}[f,\tau]=\mathbb{E}\_{\mathbf{x}\sim\mu}|f(\mathbf{x})-\tau(\mathbf{x})|^{2}= |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼𝐳∼U|(f(γ(𝐳))−λ(𝐳)|2=MSEU(𝐳)[f∘γ,λ]\displaystyle\mathbb{E}\_{\mathbf{z}\sim U}|(f(\gamma(\mathbf{z}))-\lambda(\mathbf{z})|^{2}=\textup{MSE}^{(\mathbf{z})}\_{U}[f\circ\gamma,\lambda] |  | (12) |

Observe that there is a vast space of degenerate solutions f𝑓f that minimize the mean squared error – namely all functions on ℝdsuperscriptℝ𝑑\mathbb{R}^{d} that yield the same function when restricted to the data manifold ℳℳ{\mathcal{M}}.

Our findings from the previous section suggest that neural networks are biased towards expressing a particular subset of such solutions, namely those that are low frequency. It is also worth noting that there exist methods that restrict the space of solutions: notably adversarial training (Goodfellow et al., [2014](#bib.bib15)) and Mixup (Zhang et al., [2017b](#bib.bib40)).

Experimental set up. The experimental setting is designed to afford control over both the shape of the data manifold and the target function defined on it. We will consider the family of curves in ℝ2superscriptℝ2\mathbb{R}^{2} generated by mappings γL:[0,1]→ℝ2:subscript𝛾𝐿→01superscriptℝ2\gamma\_{L}:[0,1]\rightarrow\mathbb{R}^{2} given by

|  |  |  |  |
| --- | --- | --- | --- |
|  | γL​(z)=subscript𝛾𝐿𝑧absent\displaystyle\gamma\_{L}(z)= | RL​(z)​(cos⁡(2​π​z),sin⁡(2​π​z))subscript𝑅𝐿𝑧2𝜋𝑧2𝜋𝑧\displaystyle R\_{L}(z)(\cos(2\pi z),\sin(2\pi z)) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | where | RL​(z)=1+12​sin⁡(2​π​L​z)subscript𝑅𝐿𝑧1122𝜋𝐿𝑧\displaystyle R\_{L}(z)=1+\frac{1}{2}\sin(2\pi Lz) |  | (13) |

![Refer to caption](/html/1806.08734/assets/figures/func_k=200-L=20.png)

![Refer to caption](/html/1806.08734/assets/figures/func_k=200-L=20_2.png)

Figure 7: Functions learned by two identical networks (up to initialization) to classify the binarized value of a sine wave of frequency k=200𝑘200k=200 defined on a γL=20subscript𝛾𝐿20\gamma\_{L=20} manifold. Both yield close to perfect accuracy for the samples defined on the manifold (scatter plot), yet they differ significantly elsewhere. The shaded regions show the predicted class (Red or Blue) whereas contours show the confidence (absolute value of logits).

Here, γL​([0,1])subscript𝛾𝐿01\gamma\_{L}([0,1]) defines the data-manifold and corresponds to a flower-shaped curve with L𝐿L petals, or a unit circle when L=0𝐿0L=0 (see e.g. Fig [7](#S4.F7 "Figure 7 ‣ 4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks")). Given a signal λ:[0,1]→ℝ:𝜆→01ℝ\lambda:[0,1]\rightarrow\mathbb{R} defined on the latent space [0,1]01[0,1], the task entails learning a network f:ℝ2→ℝ:𝑓→superscriptℝ2ℝf:\mathbb{R}^{2}\rightarrow\mathbb{R} such that f∘γL𝑓subscript𝛾𝐿f\circ\gamma\_{L} matches the signal λ𝜆\lambda.

![Refer to caption](/html/1806.08734/assets/figures/spec_evo_eq_amps_L-0.png)


(a) L=0𝐿0L=0

![Refer to caption](/html/1806.08734/assets/figures/spec_evo_eq_amps_L-4.png)


(b) L=4𝐿4L=4

![Refer to caption](/html/1806.08734/assets/figures/spec_evo_eq_amps_L-10.png)


(c) L=10𝐿10L=10

![Refer to caption](/html/1806.08734/assets/figures/spec_evo_eq_amps_L-16.png)


(d) L=16𝐿16L=16

![Refer to caption](/html/1806.08734/assets/figures/loss_curve_eq_amp_L=0-4-10-16.png)


(e) Loss curves

Figure 8: (a,b,c,d): Evolution of the network spectrum (x-axis for frequency, colorbar for magnitude) during training (y-axis) for the same target functions defined on manifolds γLsubscript𝛾𝐿\gamma\_{L} for various L𝐿L. Since the target function has amplitudes Ai=1subscript𝐴𝑖1A\_{i}=1 for all frequencies kisubscript𝑘𝑖k\_{i} plotted, the colorbar is clipped between 0 and 1. (e): Corresponding learning curves.
Gist: Some manifolds (here with larger L𝐿L) make it easier for the network to learn higher frequencies than others.

![Refer to caption](/html/1806.08734/assets/figures/Lvk_classification_4.png)


Figure 9: Heatmap of training accuracies of a network trained to predict the binarized value of a sine wave of given frequency (x-axis) defined on γLsubscript𝛾𝐿\gamma\_{L} for various L𝐿L (y-axis).

###### Experiment 5.

The set-up is similar to that of Experiment [1](#Thmexperiment1 "Experiment 1. ‣ 3.1 Synthetic Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks"), and λ𝜆\lambda is as defined in Eqn. [8](#S3.E8 "In Experiment 1. ‣ 3.1 Synthetic Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks") with frequencies κ=(20,40,…,180,200)𝜅2040…180200\kappa=(20,40,...,180,200), and amplitudes Ai=1​∀isubscript𝐴𝑖1for-all𝑖A\_{i}=1\,\forall\,i. The model f𝑓f is trained on the dataset {γL​(zi),λ​(zi)}i=1Nsuperscriptsubscriptsubscript𝛾𝐿subscript𝑧𝑖𝜆subscript𝑧𝑖𝑖1𝑁\{\gamma\_{L}(z\_{i}),\lambda(z\_{i})\}\_{i=1}^{N} with N=1000𝑁1000N=1000 uniformly spaced samples zisubscript𝑧𝑖z\_{i} between 00 and 111. The spectrum of f∘γL𝑓subscript𝛾𝐿f\circ\gamma\_{L} in expectation over φi∼U​(0,2​π)similar-tosubscript𝜑𝑖𝑈02𝜋\varphi\_{i}\sim U(0,2\pi) is monitored as training progresses, and the result is shown in Fig [8](#S4.F8 "Figure 8 ‣ 4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks") for various L𝐿L. Fig [8(e)](#S4.F8.sf5 "In Figure 8 ‣ 4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks") shows the corresponding mean squared error curves. More experimental details in appendix [A.2](#A1.SS2 "A.2 Experiment 5 ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks").

The results demonstrate a clear attenuation of the spectral bias as L𝐿L grows. Moreover, Fig [8(e)](#S4.F8.sf5 "In Figure 8 ‣ 4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks") suggests that the larger the L𝐿L, the easier the learning task.

###### Experiment 6.

Here, we adapt the setting of Experiment [5](#Thmexperiment5 "Experiment 5. ‣ 4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks") to binary classification by simply thresholding the function λ𝜆\lambda at 0.50.50.5 to obtain a binary target signal. To simplify visualization, we only use signals with a single frequency mode k𝑘k, such that λ​(z)=sin⁡(2​π​k​z+φ)𝜆𝑧2𝜋𝑘𝑧𝜑\lambda(z)=\sin(2\pi kz+\varphi). We train the same network on the resulting classification task with cross-entropy loss141414We use Pytorch’s BCEWithLogitsLoss. Internally, it takes a sigmoid of the network’s output (the logits) before evaluating the cross-entropy. for k∈{50,100,…,350,400}𝑘50100…350400k\in\{50,100,...,350,400\} and L∈{0,2,…,18,20}𝐿02…1820L\in\{0,2,...,18,20\}. The heatmap in Fig [9](#S4.F9 "Figure 9 ‣ 4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks") shows the classification accuracy for each (k,L)𝑘𝐿(k,L) pair. Fig [7](#S4.F7 "Figure 7 ‣ 4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks") shows visualizations of the functions learned by the same network, trained on (k,L)=(200,20)𝑘𝐿20020(k,L)=(200,20) under identical conditions up to random initialization.

Observe that increasing L𝐿L (i.e. going up a column in Fig [9](#S4.F9 "Figure 9 ‣ 4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks")) results in better (classification) performance for the same target signal. This is the same behaviour as we observed in Experiment [5](#Thmexperiment5 "Experiment 5. ‣ 4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks") (Fig [8](#S4.F8 "Figure 8 ‣ 4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks")a-d), but now with binary cross-entropy loss instead of the MSE.

Discussion. These experiments hint towards a rich interaction between the shape of the manifold and the effective difficulty of the learning task. The key mechanism underlying this phenomenon (as we formalize below) is that the relationship between frequency spectrum of the network f𝑓f and that of the fit f∘γL𝑓subscript𝛾𝐿f\circ\gamma\_{L} is mediated by the embedding map γLsubscript𝛾𝐿\gamma\_{L}. In particular, we argue that a given signal defined on the manifold is easier to fit when the coordinate functions of the manifold embedding itself has high frequency components. Thus, in our experimental setting, the same signal embedded in a flower with more petals can be captured with lower frequencies of the network.

To understand this mathematically, we address the following questions: given a target function λ𝜆\lambda, how small can the frequencies of a solution f𝑓f be such that f∘γ=λ𝑓𝛾𝜆f\circ\gamma=\lambda? And further, how does this relate to the geometry of the data-manifold ℳℳ{\mathcal{M}} induced by γ𝛾\gamma?
To find out, we write the Fourier transform of the composite function,

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | (f∘γ)~​(𝐥)~𝑓𝛾𝐥\displaystyle\widetilde{(f\circ\gamma)}(\mathbf{l}) | =∫𝐝𝐤​f~​(𝐤)​Pγ​(𝐥,𝐤)absent𝐝𝐤~𝑓𝐤subscript𝑃𝛾𝐥𝐤\displaystyle=\int\mathbf{dk}\tilde{f}(\mathbf{k})P\_{\gamma}(\mathbf{l},\mathbf{k}) |  | (14) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | where​Pγ​(𝐥,𝐤)wheresubscript𝑃𝛾𝐥𝐤\displaystyle\mathrm{where}\;P\_{\gamma}(\mathbf{l},\mathbf{k}) | =∫[0,1]m𝐝𝐳​ei​(𝐤⋅γ​(𝐳)−𝐥⋅𝐳)absentsubscriptsuperscript01𝑚𝐝𝐳superscript𝑒𝑖⋅𝐤𝛾𝐳⋅𝐥𝐳\displaystyle=\int\_{[0,1]^{m}}\mathbf{dz}\,e^{i(\mathbf{k}\cdot\gamma(\mathbf{z})-\mathbf{l}\cdot\mathbf{z})} |  |

The kernel Pγsubscript𝑃𝛾P\_{\gamma} depends on only γ𝛾\gamma and elegantly encodes the correspondence between frequencies 𝐤∈ℝd𝐤superscriptℝ𝑑\mathbf{k}\in\mathbb{R}^{d} in input space and frequencies 𝐥∈ℝm𝐥superscriptℝ𝑚\mathbf{l}\in\mathbb{R}^{m} in the latent space [0,1]msuperscript01𝑚[0,1]^{m}. Following a procedure from [Bergner et al.](#bib.bib5) , we can further investigate the behaviour of the kernel
in the regime where the stationary phase approximation is applicable, i.e. when l2+k2→∞→superscript𝑙2superscript𝑘2l^{2}+k^{2}\rightarrow\infty (cf. section 3.2. of [Bergner et al.](#bib.bib5) ). In this regime, the integral Pγsubscript𝑃𝛾P\_{\gamma} is dominated
by critical points 𝐳¯¯𝐳\bar{\mathbf{z}} of its phase, which satisfy

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐥=Jγ​(𝐳¯)​𝐤𝐥subscript𝐽𝛾¯𝐳𝐤\mathbf{l}=J\_{\gamma}(\bar{\mathbf{z}})\,\mathbf{k} |  | (15) |

where Jγ​(𝐳)i​j=∇iγj​(𝐳)subscript𝐽𝛾subscript𝐳𝑖𝑗subscript∇𝑖subscript𝛾𝑗𝐳J\_{\gamma}(\mathbf{z})\_{ij}=\nabla\_{i}\gamma\_{j}(\mathbf{z}) is the m×d𝑚𝑑m\times d Jacobian matrix of γ𝛾\gamma. Non-zero values of the kernel correspond to pairs (𝐥,𝐤)𝐥𝐤(\mathbf{l},\mathbf{k}) such that Eqn [15](#S4.E15 "In 4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks") has a solution. Further, given that the components of γ𝛾\gamma (i.e. its coordinate functions) are defined on an interval [0,1]msuperscript01𝑚[0,1]^{m}, one can use their Fourier series representation together with Eqn [15](#S4.E15 "In 4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks") to obtain a condition on their frequencies (shown in appendix [C.7](#A3.SS7 "C.7 The Fourier Transform of a Function Composition ‣ Appendix C Fourier Analysis of ReLU Networks ‣ On the Spectral Bias of Neural Networks")). More precisely, we find that the i𝑖i-th component of the RHS in Eqn [15](#S4.E15 "In 4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks") is proportional to 𝐩​γ~i​[𝐩]​ki𝐩subscript~𝛾𝑖delimited-[]𝐩subscript𝑘𝑖\mathbf{p}\tilde{\gamma}\_{i}[\mathbf{p}]k\_{i} where 𝐩∈ℤm𝐩superscriptℤ𝑚\mathbf{p}\in\mathbb{Z}^{m} is the frequency of the coordinate function γisubscript𝛾𝑖\gamma\_{i}. This yields that we can get arbitrarily large frequencies lisubscript𝑙𝑖l\_{i} if γ~i​[𝐩]subscript~𝛾𝑖delimited-[]𝐩\tilde{\gamma}\_{i}[\mathbf{p}] is large151515Consider that the data-domain is bounded, implying that γ~~𝛾\tilde{\gamma} cannot be arbitrarily scaled. enough for large 𝐩𝐩\mathbf{p}, even when kisubscript𝑘𝑖k\_{i} is fixed.

This is precisely what Experiments [5](#Thmexperiment5 "Experiment 5. ‣ 4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks") and [6](#Thmexperiment6 "Experiment 6. ‣ 4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks") demonstrate in a minimal setting. From Eqn [4](#S4.Ex2 "4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks"), observe that the coordinate functions have a frequency mode at L𝐿L. For increasing L𝐿L, it is apparent that the frequency magnitudes l𝑙l (in the latent space) that can be expressed with the same frequency k𝑘k (in the input space) increases with increasing L𝐿L. This allows the remarkable interpretation that the neural network function can express large frequencies on a manifold (l𝑙l) with smaller frequencies w.r.t its input domain (k𝑘k), provided that the coordinate functions of the data manifold embedding itself has high-frequency components.

## 5 Related Work

A number of works have focused on showing that neural networks are capable of approximating arbitrarily complex functions. Hornik et al. ([1989](#bib.bib17)); Cybenko ([1989](#bib.bib8)); Leshno et al. ([1993](#bib.bib20)) have shown that neural networks can be universal approximators when given sufficient width; more recently, Lu et al. ([2017](#bib.bib21)) proved that this property holds also for width-bounded networks. Montufar et al. ([2014](#bib.bib24)) showed that the number of linear regions of deep ReLU networks
grows polynomially with width and exponentially with depth; Raghu et al. ([2016](#bib.bib30)) generalized this result and provided asymptotically tight bounds. There have been various results of the benefits of depth for efficient approximation (Poole et al., [2016](#bib.bib29); Telgarsky, [2016](#bib.bib36); Eldan & Shamir, [2016](#bib.bib12)).
These analysis on the expressive power of deep neural networks can in part explain why over-parameterized networks can perfectly learn random input-output mappings (Zhang et al., [2017a](#bib.bib39)).

Our work more directly follows the line of research on implicit regularization in neural networks trained by gradient descent (Neyshabur et al., [2014](#bib.bib25); Soudry et al., [2017](#bib.bib34); Poggio et al., [2018](#bib.bib28); Neyshabur et al., [2017](#bib.bib26)).
In fact, while our Fourier analysis of deep ReLU networks also reflects the width and depth dependence of their expressivity, we focused on showing a learning bias of these networks towards simple functions with dominant lower frequency components. We view our results as a first step towards formalizing the findings of Arpit et al. ([2017](#bib.bib2)), where it is empirically shown that deep networks prioritize learning simple patterns of the data during training.

A few other works studied neural networks through the lens of harmonic analysis. For example, Candès ([1999](#bib.bib7)) used the ridgelet transform to build constructive procedures for approximating a given function by neural networks, in the case of oscillatory activation functions. This approach has been recently generalized to unbounded activation functions by
Sonoda & Murata ([2017](#bib.bib33)). Eldan & Shamir ([2016](#bib.bib12)) use insights on the support of the Fourier spectrum of two-layer networks to derive a worse-case depth-separation result. Barron ([1993](#bib.bib3)) makes use of Fourier space properties of the target function to derive an architecture-dependent approximation bound.
In a concurrent and independent work, Xu et al. ([2018](#bib.bib38)) make the same observation that lower frequencies are learned first.
The subsequent work by Xu ([2018](#bib.bib37)) proposes a theoretical analysis of the phenomenon in the case of 2-layer networks with sigmoid activation, based on the spectrum of the sigmoid function.

In light of our findings, it is worth comparing the case of neural networks and other popular algorithms such that kernel machines (KM) and K𝐾K-nearest neighbor classifiers. We refer to the
Appendix [E](#A5 "Appendix E Kernel Machines and KNNs ‣ On the Spectral Bias of Neural Networks") for a detailed discussion and references. In summary, our discussion there suggests that 1. DNNs strike a good balance between function smoothness and expressivity/parameter-efficiency compared with KM; 2. DNNs learn a smoother function compared with K𝐾KNNs since the spectrum of the DNN decays faster compared with K𝐾KNNs in the experiments shown there.

## 6 Conclusion

We studied deep ReLU networks through the lens of Fourier analysis. Several conclusions can be drawn from our analysis.
While neural networks can approximate arbitrary functions, we find that they favour *low frequency* ones – hence they exhibit a bias towards smooth functions – a phenomenon that we called *spectral bias*. We also illustrated how the geometry of the data manifold impacts expressivity in a non-trivial way, as
high frequency functions defined on complex manifolds can be expressed by lower frequency network functions defined in input space.

We view future work that explore the properties of neural networks in Fourier domain as promising. For example, the Fourier transform affords a natural way of measuring how fast a function can change within a small neighborhood in its input domain; as such, it is a strong candidate for quantifying and analyzing the *sensitivity* of a model – which in turn provides a natural measure of complexity (Novak et al., [2018](#bib.bib27)).
We hope to encourage more research in this direction.

## Acknowledgements

The authors would like to thank Joan Bruna, Rémi Le Priol, Vikram Voleti, Ullrich Köthe, Steffen Wolf, Lorenzo Cerrone, Sebastian Damrich, as well as the anonymous reviewers for their valuable feedback.

## References

* Arora et al. (2018)

  Arora, R., Basu, A., Mianjy, P., and Mukherjee, A.
  Understanding deep neural networks with rectified linear units.
  In *International Conference on Learning Representations*, 2018.
  URL <https://openreview.net/forum?id=B1J_rgWRW>.
* Arpit et al. (2017)

  Arpit, D., Jastrzebski, S., Ballas, N., Krueger, D., Bengio, E., Kanwal, M. S.,
  Maharaj, T., Fischer, A., Courville, A., Bengio, Y., et al.
  A closer look at memorization in deep networks.
  *arXiv preprint arXiv:1706.05394*, 2017.
* Barron (1993)

  Barron, A. R.
  Universal approximation bounds for superpositions of a sigmoidal
  function.
  *IEEE Transactions on Information theory*, 39(3):930–945, 1993.
* Bengio et al. (2009)

  Bengio, Y. et al.
  Learning deep architectures for ai.
  *Foundations and trends® in Machine Learning*,
  2(1):1–127, 2009.
* (5)

  Bergner, S., Möller, T., Weiskopf, D., and Muraki, D. J.
  A spectral analysis of function concatenations and its implications
  for sampling in direct volume visualization.
* Braun et al. (2006)

  Braun, M. L., Lange, T., and Buhmann, J. M.
  Model selection in kernel methods based on a spectral analysis of
  label information.
  In *Joint Pattern Recognition Symposium*, pp.  344–353.
  Springer, 2006.
* Candès (1999)

  Candès, E. J.
  Harmonic analysis of neural networks.
  *Applied and Computational Harmonic Analysis*, 6(2):197–218, 1999.
* Cybenko (1989)

  Cybenko, G.
  Approximation by superpositions of a sigmoidal function.
  *Mathematics of Control, Signals, and Systems (MCSS)*,
  2(4):303–314, 1989.
* Devroye et al. (1996)

  Devroye, L., Györfi, L., and Lugosi, G.
  Consistency of the k-nearest neighbor rule.
  In *A Probabilistic Theory of Pattern Recognition*, pp. 169–185. Springer, 1996.
* Diaz et al. (2016)

  Diaz, R., Le, Q.-N., and Robins, S.
  Fourier transforms of polytopes, solid angle sums, and discrete
  volume.
  *arXiv preprint arXiv:1602.08593*, 2016.
* Draxler et al. (2018)

  Draxler, F., Veschgini, K., Salmhofer, M., and Hamprecht, F. A.
  Essentially no barriers in neural network energy landscape.
  *arXiv preprint arXiv:1803.00885*, 2018.
* Eldan & Shamir (2016)

  Eldan, R. and Shamir, O.
  The power of depth for feedforward neural networks.
  In *Conference on Learning Theory*, pp.  907–940, 2016.
* Fasshauer (2011)

  Fasshauer, G. E.
  Positive definite kernels: past, present and future.
  *Dolomite Research Notes on Approximation*, 4:21–63,
  2011.
* Goodfellow et al. (2016)

  Goodfellow, I., Bengio, Y., and Courville, A.
  *Deep Learning*.
  MIT Press, 2016.
  <http://www.deeplearningbook.org>.
* Goodfellow et al. (2014)

  Goodfellow, I. J., Shlens, J., and Szegedy, C.
  Explaining and harnessing adversarial examples.
  *arXiv preprint arXiv:1412.6572*, 2014.
* Hammer & Gersmann (2003)

  Hammer, B. and Gersmann, K.
  A note on the universal approximation capability of support vector
  machines.
  *Neural Processing Letters*, 17(1):43–53,
  2003.
* Hornik et al. (1989)

  Hornik, K., Stinchcombe, M., and White, H.
  Multilayer feedforward networks are universal approximators.
  *Neural networks*, 2(5):359–366, 1989.
* Kingma & Ba (2014)

  Kingma, D. P. and Ba, J.
  Adam: A method for stochastic optimization.
  *arXiv preprint arXiv:1412.6980*, 2014.
* Kolsbjerg et al. (2016)

  Kolsbjerg, E. L., Groves, M. N., and Hammer, B.
  An automated nudged elastic band method.
  *The Journal of chemical physics*, 145(9):094107, 2016.
* Leshno et al. (1993)

  Leshno, M., Lin, V. Y., Pinkus, A., and Schocken, S.
  Multilayer feedforward networks with a nonpolynomial activation
  function can approximate any function.
  *Neural networks*, 6(6):861–867, 1993.
* Lu et al. (2017)

  Lu, Z., Pu, H., Wang, F., Hu, Z., and Wang, L.
  The expressive power of neural networks: A view from the width.
  In *Advances in Neural Information Processing Systems*, pp. 6231–6239, 2017.
* Ma & Belkin (2017)

  Ma, S. and Belkin, M.
  Diving into the shallows: a computational perspective on large-scale
  shallow learning.
  In *Advances in Neural Information Processing Systems*, pp. 3781–3790, 2017.
* Miyato et al. (2018)

  Miyato, T., Kataoka, T., Koyama, M., and Yoshida, Y.
  Spectral normalization for generative adversarial networks.
  In *International Conference on Learning Representations*, 2018.
  URL <https://openreview.net/forum?id=B1QRgziT->.
* Montufar et al. (2014)

  Montufar, G. F., Pascanu, R., Cho, K., and Bengio, Y.
  On the number of linear regions of deep neural networks.
  In *Advances in neural information processing systems*, pp. 2924–2932, 2014.
* Neyshabur et al. (2014)

  Neyshabur, B., Tomioka, R., and Srebro, N.
  In search of the real inductive bias: On the role of implicit
  regularization in deep learning.
  *arXiv preprint arXiv:1412.6614*, 2014.
* Neyshabur et al. (2017)

  Neyshabur, B., Bhojanapalli, S., McAllester, D., and Srebro, N.
  Exploring generalization in deep learning.
  In *Advances in Neural Information Processing Systems*, pp. 5949–5958, 2017.
* Novak et al. (2018)

  Novak, R., Bahri, Y., Abolafia, D. A., Pennington, J., and Sohl-Dickstein, J.
  Sensitivity and generalization in neural networks: an empirical
  study.
  In *International Conference on Learning Representations*, 2018.
  URL <https://openreview.net/forum?id=HJC2SzZCW>.
* Poggio et al. (2018)

  Poggio, T., Kawaguchi, K., Liao, Q., Miranda, B., Rosasco, L., Boix, X.,
  Hidary, J., and Mhaskar, H.
  Theory of deep learning iii: the non-overfitting puzzle.
  Technical report, Technical report, CBMM memo 073, 2018.
* Poole et al. (2016)

  Poole, B., Lahiri, S., Raghu, M., Sohl-Dickstein, J., and Ganguli, S.
  Exponential expressivity in deep neural networks through transient
  chaos.
  In Lee, D. D., Sugiyama, M., Luxburg, U. V., Guyon, I., and Garnett,
  R. (eds.), *Advances in Neural Information Processing Systems 29*, pp. 3360–3368. Curran Associates, Inc., 2016.
* Raghu et al. (2016)

  Raghu, M., Poole, B., Kleinberg, J., Ganguli, S., and Sohl-Dickstein, J.
  On the expressive power of deep neural networks.
  *arXiv preprint arXiv:1606.05336*, 2016.
* Rasmussen (2004)

  Rasmussen, C. E.
  Gaussian processes in machine learning.
  In *Advanced lectures on machine learning*, pp.  63–71.
  Springer, 2004.
* Serov (2017)

  Serov, V.
  *Fourier series, Fourier transform and their applications to
  mathematical physics*.
  Springer, 2017.
* Sonoda & Murata (2017)

  Sonoda, S. and Murata, N.
  Neural network with unbounded activation functions is universal
  approximator.
  *Applied and Computational Harmonic Analysis*, 43(2):233–268, 2017.
* Soudry et al. (2017)

  Soudry, D., Hoffer, E., Nacson, M. S., Gunasekar, S., and Srebro, N.
  The implicit bias of gradient descent on separable data.
  *arXiv preprint arXiv:1710.10345*, 2017.
* Spivak (2018)

  Spivak, M.
  *Calculus On Manifolds: A Modern Approach To Classical Theorems
  Of Advanced Calculus*.
  CRC press, 2018.
* Telgarsky (2016)

  Telgarsky, M.
  Benefits of depth in neural networks.
  *Conference on Learning Theory (COLT), 2016*, 2016.
* Xu (2018)

  Xu, Z. J.
  Understanding training and generalization in deep learning by fourier
  analysis.
  *arXiv preprint arXiv:1808.04295*, 2018.
* Xu et al. (2018)

  Xu, Z.-Q. J., Zhang, Y., and Xiao, Y.
  Training behavior of deep neural network in frequency domain.
  *arXiv preprint arXiv:1807.01251*, 2018.
* Zhang et al. (2017a)

  Zhang, C., Bengio, S., Hardt, M., Recht, B., and Vinyals, O.
  Understanding deep learning requires rethinking generalization.
  *International Conference on Learning Representations (ICLR)*,
  2017a.
* Zhang et al. (2017b)

  Zhang, H., Cisse, M., Dauphin, Y. N., and Lopez-Paz, D.
  mixup: Beyond empirical risk minimization.
  *arXiv preprint arXiv:1710.09412*, 2017b.
* Zhang et al. (2018)

  Zhang, L., Naitzat, G., and Lim, L.-H.
  Tropical geometry of deep neural networks.
  *arXiv preprint arXiv:1805.07091*, 2018.

## Appendix A Experimental Details

### A.1 Experiment [1](#Thmexperiment1 "Experiment 1. ‣ 3.1 Synthetic Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks")

We fit a 6 layer ReLU network with 256 units per layer fθsubscript𝑓𝜃f\_{\theta} to the target function λ𝜆\lambda, which is a superposition of sine waves with increasing frequencies:

|  |  |  |
| --- | --- | --- |
|  | λ:[0,1]→ℝ,λ​(z)=∑iAi​sin⁡(2​π​ki​z+φi):𝜆formulae-sequence→01ℝ𝜆𝑧subscript𝑖subscript𝐴𝑖2𝜋subscript𝑘𝑖𝑧subscript𝜑𝑖\lambda:[0,1]\rightarrow\mathbb{R},\,\lambda(z)=\sum\_{i}A\_{i}\sin(2\pi k\_{i}z+\varphi\_{i}) |  |

where ki=(5,10,15,…,50)subscript𝑘𝑖51015…50k\_{i}=(5,10,15,...,50), and φisubscript𝜑𝑖\varphi\_{i} is sampled from the uniform distribution U​(0,2​π)𝑈02𝜋U(0,2\pi). In the first setting, we set equal amplitude for all frequencies, i.e. Ai=1​∀isubscript𝐴𝑖1for-all𝑖A\_{i}=1\,\forall\,i, while in the second setting we assign larger amplitudes to the higher frequencies, i.e. Ai=(0.1,0.2,…,1)subscript𝐴𝑖0.10.2…1A\_{i}=(0.1,0.2,...,1). We sample λ𝜆\lambda on 200 uniformly spaced points in [0,1]01[0,1] and train the network for 800008000080000 steps of full-batch gradient descent with Adam (Kingma & Ba, [2014](#bib.bib18)). Note that we do not use stochastic gradient descent to avoid the stochasticity in parameter updates as a confounding factor. We evaluate the network on the same 200 point grid every 100 training steps and compute the magnitude of its (single-sided) discrete fourier transform at frequencies kisubscript𝑘𝑖k\_{i} which we denote with |f~ki|subscript~𝑓subscript𝑘𝑖|\tilde{f}\_{k\_{i}}|. Finally, we plot in figure [1](#S2.F1 "Figure 1 ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks") the normalized magnitudes |f~ki|Aisubscript~𝑓subscript𝑘𝑖subscript𝐴𝑖\frac{|\tilde{f}\_{k\_{i}}|}{A\_{i}} averaged over 10 runs (with different sets of sampled phases φisubscript𝜑𝑖\varphi\_{i}). We also record the spectral norms of the weights at each layer as the training progresses, which we plot in figure [1](#S2.F1 "Figure 1 ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks") for both settings (the spectral norm is evaluated with 10 power iterations). In figure [2](#S3.F2 "Figure 2 ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks"), we show an example target function and the predictions of the network trained on it (over the iterations), and in figure [10](#A1.F10 "Figure 10 ‣ A.1 Experiment 1 ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks") we plot the loss curves.

![Refer to caption](/html/1806.08734/assets/figures/loss_curve_eq_amp.png)


(a) Equal Amplitudes.

![Refer to caption](/html/1806.08734/assets/figures/loss_curve_inc_amp.png)


(b) Increasing Amplitudes.

Figure 10: Loss curves averaged over multiple runs. (cf. Experiment [1](#Thmexperiment1 "Experiment 1. ‣ 3.1 Synthetic Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks"))

### A.2 Experiment [5](#Thmexperiment5 "Experiment 5. ‣ 4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks")

We use the same 6-layer deep 256-unit wide network and define the target function

|  |  |  |
| --- | --- | --- |
|  | λ:𝒟→ℝ,z↦λ​(z)=∑iAi​sin⁡(2​π​ki​z+φi):𝜆formulae-sequence→𝒟ℝmaps-to𝑧𝜆𝑧subscript𝑖subscript𝐴𝑖2𝜋subscript𝑘𝑖𝑧subscript𝜑𝑖\lambda:\mathcal{D}\rightarrow\mathbb{R},\;z\mapsto\lambda(z)=\sum\_{i}A\_{i}\sin(2\pi k\_{i}z+\varphi\_{i}) |  |

where ki=(20,40,…,180,200)subscript𝑘𝑖2040…180200k\_{i}=(20,40,...,180,200), Ai=1​∀isubscript𝐴𝑖1for-all𝑖A\_{i}=1\,\forall\,i and φ∼U​(0,2​π)similar-to𝜑𝑈02𝜋\varphi\sim U(0,2\pi). We sample ϕitalic-ϕ\phi on a grid with 1000 uniformly spaced points between 0 and 1 and map it to the input domain via γLsubscript𝛾𝐿\gamma\_{L} to obtain a dataset {(γL​(zj),λ​(zj))}j=0999superscriptsubscriptsubscript𝛾𝐿subscript𝑧𝑗𝜆subscript𝑧𝑗𝑗0999\{(\gamma\_{L}(z\_{j}),\lambda(z\_{j}))\}\_{j=0}^{999}, on which we train the network with 50000 full-batch gradient descent steps of Adam. On the same 1000-point grid, we evaluate the magnitude of the (single-sided) discrete Fourier transform of fθ∘γLsubscript𝑓𝜃subscript𝛾𝐿f\_{\theta}\circ\gamma\_{L} every 100 training steps at frequencies kisubscript𝑘𝑖k\_{i} and average over 10 runs (each with a different set of sampled zisubscript𝑧𝑖z\_{i}’s). Fig [8](#S4.F8 "Figure 8 ‣ 4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks") shows the evolution of the spectrum as training progresses for L=0,4,10,16𝐿

041016L=0,4,10,16, and Fig [8(e)](#S4.F8.sf5 "In Figure 8 ‣ 4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks") shows the corresponding loss curves.

### A.3 Experiment [3](#Thmexperiment3 "Experiment 3. ‣ 3.2 Real-Data Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks")

In Figure [11](#A1.F11 "Figure 11 ‣ A.3 Experiment 3 ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks"), we show the training curves corresponding to Figure [4](#S3.F4 "Figure 4 ‣ 3.1 Synthetic Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks").

![Refer to caption](/html/1806.08734/assets/figures/mnist_noise_freq_0_1_amp_var_tra_ch_n.png)


(a) k=0.1𝑘0.1k=0.1

![Refer to caption](/html/1806.08734/assets/figures/mnist_noise_freq_1_amp_var_tra_ch_n.png)


(b) k=1𝑘1k=1

![Refer to caption](/html/1806.08734/assets/figures/mnist_noise_freq_var_amp_0_5_tra_ch_n.png)


(c) β=0.5𝛽0.5\beta=0.5

![Refer to caption](/html/1806.08734/assets/figures/mnist_noise_freq_var_amp_1_tra_ch_n.png)


(d) β=1.𝛽1\beta=1.

Figure 11: (a,b,c,d): Training curves for various settings of noise amplitude β𝛽\beta and frequency k𝑘k corresponding to Figure [4](#S3.F4 "Figure 4 ‣ 3.1 Synthetic Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks").

### A.4 Experiment [4](#Thmexperiment4 "Experiment 4. ‣ 3.2 Real-Data Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks")

Consider the Gaussian Radial Basis Kernel, given by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | k:X×X→ℝ,kσ​(𝐱,𝐲)↦exp⁡(‖𝐱−𝐲‖σ2):𝑘formulae-sequence→𝑋𝑋ℝmaps-tosubscript𝑘𝜎𝐱𝐲norm𝐱𝐲superscript𝜎2k:X\times X\rightarrow\mathbb{R},\,k\_{\sigma}(\mathbf{x},\mathbf{y})\mapsto\exp\left(\frac{\|\mathbf{x}-\mathbf{y}\|}{\sigma^{2}}\right) |  | (16) |

where X𝑋X is a compact subset of ℝdsuperscriptℝ𝑑\mathbb{R}^{d} and σ∈ℝ+𝜎subscriptℝ\sigma\in\mathbb{R}\_{+} is defined as the width of the kernel161616We drop the subscript σ𝜎\sigma to simplify the notation.. Since k𝑘k is positive definite (Fasshauer, [2011](#bib.bib13)), Mercer’s Theorem can be invoked to express it as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | k​(𝐱,𝐲)=∑n=1∞λi​φn​(𝐱)​φn​(𝐲)𝑘𝐱𝐲superscriptsubscript𝑛1subscript𝜆𝑖subscript𝜑𝑛𝐱subscript𝜑𝑛𝐲k(\mathbf{x},\mathbf{y})=\sum\_{n=1}^{\infty}\lambda\_{i}\varphi\_{n}(\mathbf{x})\varphi\_{n}(\mathbf{y}) |  | (17) |

where φnsubscript𝜑𝑛\varphi\_{n} is the eigenfunction of k𝑘k satisfying:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∫k​(𝐱,𝐲)​φn​(𝐲)​𝐝𝐲=⟨k​(𝐱,⋅),φn⟩=λn​φn​(𝐱)𝑘𝐱𝐲subscript𝜑𝑛𝐲𝐝𝐲  𝑘𝐱⋅subscript𝜑𝑛subscript𝜆𝑛subscript𝜑𝑛𝐱\int k(\mathbf{x},\mathbf{y})\varphi\_{n}(\mathbf{y})\mathbf{dy}=\langle k(\mathbf{x},\cdot),\varphi\_{n}\rangle=\lambda\_{n}\varphi\_{n}(\mathbf{x}) |  | (18) |

Due to positive definiteness of the kernel, the eigenvalues λisubscript𝜆𝑖\lambda\_{i} are non-negative and the eigenfunctions φnsubscript𝜑𝑛\varphi\_{n} form an orthogonal basis of L2​(X)superscript𝐿2𝑋L^{2}(X), i.e. ⟨φi,φj⟩=δi​j

subscript𝜑𝑖subscript𝜑𝑗
subscript𝛿𝑖𝑗\langle\varphi\_{i},\varphi\_{j}\rangle=\delta\_{ij}. The analogy to the final case is easily seen: let X=𝐱ii=1N𝑋superscriptsubscriptsubscript𝐱𝑖𝑖1𝑁X={\mathbf{x}\_{i}}\_{i=1}^{N} be the set of samples, f:X→ℝ:𝑓→𝑋ℝf:X\rightarrow\mathbb{R} a function. One obtains (cf. Chapter 4 (Rasmussen, [2004](#bib.bib31))):

|  |  |  |  |
| --- | --- | --- | --- |
|  | ⟨k​(𝐱,⋅),f⟩=∑i=1Nk​(𝐱,𝐱i)​fi  𝑘𝐱⋅𝑓 superscriptsubscript𝑖1𝑁𝑘𝐱subscript𝐱𝑖subscript𝑓𝑖\langle k(\mathbf{x},\cdot),f\rangle=\sum\_{i=1}^{N}k(\mathbf{x},\mathbf{x}\_{i})f\_{i} |  | (19) |

where fi=f​(𝐱i)subscript𝑓𝑖𝑓subscript𝐱𝑖f\_{i}=f(\mathbf{x}\_{i}). Now, defining K𝐾K as the positive definite kernel matrix with elements Ki​j=k​(𝐱i​𝐱j)subscript𝐾𝑖𝑗𝑘subscript𝐱𝑖subscript𝐱𝑗K\_{ij}=k(\mathbf{x}\_{i}\mathbf{x}\_{j}), we consider it’s eigendecomposition V​Λ​VT𝑉Λsuperscript𝑉𝑇V\Lambda V^{T} where ΛΛ\Lambda is the diagonal matrix of (w.l.o.g sorted) eigenvalues λ1≤…≤λNsubscript𝜆1…subscript𝜆𝑁\lambda\_{1}\leq...\leq\lambda\_{N} and the columns of V𝑉V are the corresponding eigenvectors. This yields:

|  |  |  |  |
| --- | --- | --- | --- |
|  |  | k​(𝐱i,𝐱j)=Ki​j=(V​Λ​VT)i​j=∑n=1Nλn​vn​i​vn​j𝑘subscript𝐱𝑖subscript𝐱𝑗subscript𝐾𝑖𝑗subscript𝑉Λsuperscript𝑉𝑇𝑖𝑗superscriptsubscript𝑛1𝑁subscript𝜆𝑛subscript𝑣𝑛𝑖subscript𝑣𝑛𝑗\displaystyle k(\mathbf{x}\_{i},\mathbf{x}\_{j})=K\_{ij}=(V\Lambda V^{T})\_{ij}=\sum\_{n=1}^{N}\lambda\_{n}v\_{ni}v\_{nj} |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | =\displaystyle= | ∑n=1Nλn​φn​(𝐱i)​φn​(𝐱j)⟹φn​(𝐱i)=vn​isuperscriptsubscript𝑛1𝑁subscript𝜆𝑛subscript𝜑𝑛subscript𝐱𝑖subscript𝜑𝑛subscript𝐱𝑗subscript𝜑𝑛subscript𝐱𝑖subscript𝑣𝑛𝑖\displaystyle\sum\_{n=1}^{N}\lambda\_{n}\varphi\_{n}(\mathbf{x}\_{i})\varphi\_{n}(\mathbf{x}\_{j})\implies\varphi\_{n}(\mathbf{x}\_{i})=v\_{ni} |  | (20) |

Like in (Braun et al., [2006](#bib.bib6)), we define the *spectrum* f~​[n]~𝑓delimited-[]𝑛\tilde{f}[n] of the function f𝑓f as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | f~​[n]=⟨f,φn⟩=𝐟⋅𝐯n~𝑓delimited-[]𝑛  𝑓subscript𝜑𝑛⋅𝐟subscript𝐯𝑛\tilde{f}[n]=\langle f,\varphi\_{n}\rangle=\mathbf{f}\cdot\mathbf{v}\_{n} |  | (21) |

where 𝐟=(f​(𝐱1),…,f​(𝐱N))𝐟𝑓subscript𝐱1…𝑓subscript𝐱𝑁\mathbf{f}=(f(\mathbf{x}\_{1}),...,f(\mathbf{x}\_{N})). The value n𝑛n can be thought of a generalized notion of *frequency*. Indeed, it is known (Fasshauer, [2011](#bib.bib13); Rasmussen, [2004](#bib.bib31)), for instance, that the eigenfunctions φnsubscript𝜑𝑛\varphi\_{n} resemble sinusoids with increasing frequencies (for increasing n𝑛n or decreasing λnsubscript𝜆𝑛\lambda\_{n}). In Figure [6](#S3.F6 "Figure 6 ‣ 3.2 Real-Data Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks"), we plot the eigenvectors 𝐯0subscript𝐯0\mathbf{v}\_{0} and 𝐯Nsubscript𝐯𝑁\mathbf{v}\_{N} for {𝐱i}i=150superscriptsubscriptsubscript𝐱𝑖𝑖150\{\mathbf{x}\_{i}\}\_{i=1}^{50} uniformly spaced between [0,1]01[0,1]. Further, in Figure ? we evaluate the discrete Fourier transform of all N=50𝑁50N=50 eigenvectors, and find that the eigenfunction index n𝑛n does indeed coincide with frequency k𝑘k. Finally, we remark that the link between signal complexity and the spectrum is extensively studied in (Braun et al., [2006](#bib.bib6)).

![Refer to caption](/html/1806.08734/assets/figures/exploratory_ev0.png)


(a) Eigenvector with the largest eigenvalue (n=1𝑛1n=1).

![Refer to caption](/html/1806.08734/assets/figures/exploratory_ev49.png)


(b) Eigenvector with the smallest eigenvalue (n=50𝑛50n=50).

Figure 12: Two extreme eigenvectors of the Gaussian RBF kernel for 505050 uniformly spaced samples between 00 and 111.

#### A.4.1 Loss Curves Accompanying Figure [5](#S3.F5 "Figure 5 ‣ 3.2 Real-Data Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks")

![Refer to caption](/html/1806.08734/assets/figures/kernel_hfn_loss.png)


Figure 13: Loss curves for the Figure [5](#S3.F5 "Figure 5 ‣ 3.2 Real-Data Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks"). We find that the validation loss dips at around the 200200200th iteration.

### A.5 Qualitative Ablation over Architectures

![Refer to caption](/html/1806.08734/assets/figures/delta_fn.png)


(a) Sampled δ𝛿\delta-function at x=0.5𝑥0.5x=0.5.

![Refer to caption](/html/1806.08734/assets/figures/spec_delta_fn.png)


(b) Constant Spectrum of the δ𝛿\delta-function.

Figure 14: The target function used in Experiment [7](#Thmexperiment7 "Experiment 7. ‣ A.5 Qualitative Ablation over Architectures ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks").

Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks") exposes the relationship between the fourier spectrum of a network and its depth, width and max-norm of parameters. The following experiment is a qualitative ablation study over these variables.

###### Experiment 7.

In this experiment, we fit various networks to the δ𝛿\delta-function at x=0.5𝑥0.5x=0.5 (see Fig [14(a)](#A1.F14.sf1 "In Figure 14 ‣ A.5 Qualitative Ablation over Architectures ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks")). Its spectrum is constant for all frequencies (Fig [14(b)](#A1.F14.sf2 "In Figure 14 ‣ A.5 Qualitative Ablation over Architectures ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks")), which makes it particularly useful for testing how well a given network can fit large frequencies. Fig [17](#A1.F17 "Figure 17 ‣ A.5 Qualitative Ablation over Architectures ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks") shows the ablation over weight clip (i.e. max parameter max-norm), Fig [15](#A1.F15 "Figure 15 ‣ A.5 Qualitative Ablation over Architectures ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks") over depth and Fig [16](#A1.F16 "Figure 16 ‣ A.5 Qualitative Ablation over Architectures ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks") over width. Fig [18](#A1.F18 "Figure 18 ‣ A.5 Qualitative Ablation over Architectures ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks") exemplarily shows how the network prediction evolves with training iterations. All networks are trained for 60K iterations of full-batch gradient descent under identical conditions (Adam optimizer with l​r=0.0003𝑙𝑟0.0003lr=0.0003, no weight decay).

![Refer to caption](/html/1806.08734/assets/figures/spec_w=16-d=3-K=10.png)


(a) Depth =3absent3=3.

![Refer to caption](/html/1806.08734/assets/figures/spec_w=16-d=4-K=10.png)


(b) Depth =4absent4=4.

![Refer to caption](/html/1806.08734/assets/figures/spec_w=16-d=5-K=10.png)


(c) Depth =5absent5=5.

![Refer to caption](/html/1806.08734/assets/figures/spec_w=16-d=6-K=10.png)


(d) Depth =6absent6=6.

Figure 15: Evolution with training iterations (y-axis) of the Fourier spectrum (x-axis for frequency, and colormap for magnitude) for a network with varying depth, width =16absent16=16 and weight clip =10absent10=10. The spectrum of the target function is a constant 0.0050.0050.005 for all frequencies.



![Refer to caption](/html/1806.08734/assets/figures/spec_w=16-d=3-K=10.png)


(a) Width =16absent16=16.

![Refer to caption](/html/1806.08734/assets/figures/spec_w=32-d=3-K=10.png)


(b) Width =32absent32=32.

![Refer to caption](/html/1806.08734/assets/figures/spec_w=64-d=3-K=10.png)


(c) Width =64absent64=64.

![Refer to caption](/html/1806.08734/assets/figures/spec_w=128-d=3-K=10.png)


(d) Width =128absent128=128.

Figure 16: Evolution with training iterations (y-axis) of the Fourier spectrum (x-axis for frequency, and colormap for magnitude) for a network with varying width, depth =3absent3=3 and weight clip =10absent10=10. The spectrum of the target function is a constant 0.0050.0050.005 for all frequencies.



![Refer to caption](/html/1806.08734/assets/figures/spec_w=64-d=6-K=0_1.png)


(a) Weight Clip =0.1absent0.1=0.1.

![Refer to caption](/html/1806.08734/assets/figures/spec_w=64-d=6-K=0_15.png)


(b) Weight Clip =0.15absent0.15=0.15.

![Refer to caption](/html/1806.08734/assets/figures/spec_w=64-d=6-K=0_2.png)


(c) Weight Clip =0.2absent0.2=0.2.

![Refer to caption](/html/1806.08734/assets/figures/spec_w=64-d=6-K=2.png)


(d) Weight Clip =2absent2=2.

Figure 17: Evolution with training iterations (y-axis) of the Fourier spectrum (x-axis for frequency, and colormap for magnitude) for a network with varying weight clip, depth =6absent6=6 and width =64absent64=64. The spectrum of the target function is a constant 0.0050.0050.005 for all frequencies.



![Refer to caption](/html/1806.08734/assets/figures/space_w=64-d=6-K=0_1.png)


(a) Weight Clip =0.1absent0.1=0.1.

![Refer to caption](/html/1806.08734/assets/figures/space_w=64-d=6-K=0_15.png)


(b) Weight Clip =0.15absent0.15=0.15.

![Refer to caption](/html/1806.08734/assets/figures/space_w=64-d=6-K=0_2.png)


(c) Weight Clip =0.2absent0.2=0.2.

![Refer to caption](/html/1806.08734/assets/figures/space_w=64-d=6-K=2.png)


(d) Weight Clip =2absent2=2.

Figure 18: Evolution with training iterations (y-axis) of the network prediction (x-axis for input, and colormap for predicted value) for a network with varying weight clip, depth =6absent6=6 and width =64absent64=64. The target function is a δ𝛿\delta peak at x=0.5𝑥0.5x=0.5.

We make the following observations.

1. (a)

   Fig [15](#A1.F15 "Figure 15 ‣ A.5 Qualitative Ablation over Architectures ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks") shows that increasing the depth (for fixed width) significantly improves the network’s ability to fit higher frequencies (note that the depth increases linearly).
2. (b)

   Fig [16](#A1.F16 "Figure 16 ‣ A.5 Qualitative Ablation over Architectures ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks") shows that increasing the width (for fixed depth) also helps, but the effect is considerably weaker (note that the width increases exponentially).
3. (c)

   Fig [17](#A1.F17 "Figure 17 ‣ A.5 Qualitative Ablation over Architectures ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks") shows that increasing the weight clip (or the max parameter max-norm) also helps the network fit higher frequencies.

The above observations are all consistent with Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks"), and further show that lower frequencies are learned first (i.e. the spectral bias, cf. Experiment [1](#Thmexperiment1 "Experiment 1. ‣ 3.1 Synthetic Experiments ‣ 3 Lower Frequencies are Learned First ‣ On the Spectral Bias of Neural Networks")). Further, Figure [17](#A1.F17 "Figure 17 ‣ A.5 Qualitative Ablation over Architectures ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks") shows that constraining the Lipschitz constant (weight clip) prevents the network from learning higher frequencies, furnishing evidence that the 𝒪​(Lf)𝒪subscript𝐿𝑓\mathcal{O}(L\_{f}) bound can be tight.

### A.6 MNIST: A Proof of Concept

In the following experiment, we show that given two manifolds of the same dimension – one flat and the other not – the task of learning random labels is harder to solve if the input samples lie on the same manifold. We demonstrate on MNIST under the assumption that the manifold hypothesis is true, and use the fact that the spectrum of the target function we use (white noise) is constant in expectation, and therefore independent of the underlying coordinate system when defined on the manifold.

###### Experiment 8.

In this experiment, we investigate if it is easier to learn a signal on a more realistic data-manifold like that of MNIST (assuming the manifold hypothesis is true), and compare with a flat manifold of the same dimension. To that end, we use the 646464-dimensional feature-space ℰℰ\mathcal{E} of a denoising171717This experiment yields the same result if variational autoencoders are used instead. autoencoder as a proxy for the real data-manifold of unknown number of dimensions. The decoder functions as an embedding of ℰℰ\mathcal{E} in the input space X=ℝ784𝑋superscriptℝ784X=\mathbb{R}^{784}, which effectively amounts to training a network on the reconstructions of the autoencoder. For comparision, we use an injective embedding181818The xy-plane is ℝ3superscriptℝ3\mathbb{R}^{3} an injective embedding of a subset of ℝ2superscriptℝ2\mathbb{R}^{2} in ℝ3superscriptℝ3\mathbb{R}^{3}. of a 64-dimensional hyperplane in X𝑋X. The latter is equivalent to sampling 784784784-dimensional vectors from U​([0,1])𝑈01U([0,1]) and setting all but the first 64 components to zero. The target function is white-noise, sampled as scalars from the uniform distribution U​([0,1])𝑈01U([0,1]). Two identical networks are trained under identical conditions, and Fig [19](#A1.F19 "Figure 19 ‣ A.6 MNIST: A Proof of Concept ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks") shows the resulting loss curves, each averaged over 10 runs.

This result complements the findings of (Arpit et al., [2017](#bib.bib2)) and (Zhang et al., [2017a](#bib.bib39)), which show that it’s easier to fit random labels to random inputs if the latter is defined on the full dimensional input space (i.e. the dimension of the flat manifold is the same as that of the input space, and not that of the underlying data-manifold being used for comparison).

![Refer to caption](/html/1806.08734/assets/figures/loss_curves_short.png)


Figure 19: Loss curves of two identical networks trained to regress white-noise under identical conditions, one on MNIST reconstructions from a DAE with 64 encoder features (blue), and the other on 64-dimensional random vectors (green).

### A.7 Cifar-10: It’s All Connected

We have seen that deep neural networks are biased towards learning low frequency functions. This should have as a consequence that isolated *bubbles* of constant prediction are rare. This in turn implies that given any two points in the input space and a network function that predicts the same class for the said points, there should be a path connecting them such that the network prediction does not change along the path. In the following, we present an experiment where we use a path finding method to find such a path between all Cifar-10 input samples indeed exist.

![Refer to caption](/html/1806.08734/assets/x5.png)


Figure 20: Path between CIFAR-10 adversarial examples (e.g. “frog” and “automobile”, such that all images are classified as “airplane”).

###### Experiment 9.

Using AutoNEB (Kolsbjerg et al., [2016](#bib.bib19)), we construct paths between (adversarial) Cifar-10 images that are classified by a ResNet20 to be all of the same target class.
AutoNEB bends a linear path between points in some space ℝmsuperscriptℝ𝑚\mathbb{R}^{m} so that some maximum energy along the path is minimal.
Here, the space is the input space of the neural network, i.e. the space of 32×32×33232332\times 32\times 3 images and the logit output of the ResNet20 for a given class is minimized.
We construct paths between the following points in image space:

* •

  From one training image to another,
* •

  from a training image to an adversarial,
* •

  from one adversarial to another.

We only consider pairs of images that belong to the same class c𝑐c (or, for adversarials, that originate from another class ≠cabsent𝑐\neq c, but that the model classifies to be of the specified class c𝑐c).
For each class, we randomly select 50 training images and select a total of 50 random images from all other classes and generate adversarial samples from the latter.
Then, paths between all pairs from the whole set of images are computed.

The AutoNEB parameters are chosen as follows:
We run four NEB iterations with 10 steps of SGD with learning rate 0.0010.0010.001 and momentum 0.90.90.9.
This computational budget is similar to that required to compute the adversarial samples.
The gradient for each NEB step is computed to maximize the logit output of the ResNet-20 for the specified target class c𝑐c.
We use the formulation of NEB without springs (Draxler et al., [2018](#bib.bib11)).

The result is very clear: We can find paths between *all* pairs of images for all CIFAR10 labels that do not cross a single decision boundary.
This means that all paths belong to the same connected component regarding the output of the DNN. This holds for all possible combinations of images in the above list. Figure [21](#A1.F21 "Figure 21 ‣ A.7 Cifar-10: It’s All Connected ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks") shows connecting training to adversarial images and Figure [20](#A1.F20 "Figure 20 ‣ A.7 Cifar-10: It’s All Connected ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks") paths between pairs of adversarial images. Paths between training images are not shown, they provide no further insight. Note that the paths are strikingly simple: Visually, they are hard to distinguish from the linear interpolation. Quantitatively, they are essentially (but not exactly) linear, with an average length (3.0±0.3)%percentplus-or-minus3.00.3(3.0\pm 0.3)\% longer than the linear connection.

![Refer to caption](/html/1806.08734/assets/x6.png)


Figure 21: Each row is a path through the image space from an adversarial sample (right) to a true training image (left). All images are classified by a ResNet-20 to be of the class of the training sample on the right with at least 95% softmax certainty. This experiment shows we can find a path from adversarial examples (right, Eg. ”(cat)”) that are classified as a particular class (”airplane”) are connected to actual training samples from that class (left, ”airplane”) such that all samples along that path are also predicted by the network to be of the same class.

## Appendix B The Continuous Piecewise Linear Structure of Deep ReLU Networks

We consider the class of ReLU network functions f:ℝd↦ℝ:𝑓maps-tosuperscriptℝ𝑑ℝf:\mathbb{R}^{d}\mapsto\mathbb{R} defined by Eqn. [1](#S2.E1 "In 2.1 Preliminaries ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks").
Following the terminology of (Raghu et al., [2016](#bib.bib30); Montufar et al., [2014](#bib.bib24)), each linear region of the network then corresponds to a unique *activation pattern*, wherein each hidden neuron is assigned an activation variable ϵ∈{−1,1}italic-ϵ11\epsilon\in\{-1,1\}, conditioned on whether its input is positive or negative.
ReLU networks can be explictly expressed as a sum over all possible activation patterns, as in the following lemma.

###### Lemma 3.

Given L𝐿L binary vectors ϵ(1),⋯​ϵ(L)

superscriptitalic-ϵ1⋯superscriptitalic-ϵ𝐿\epsilon^{(1)},\cdots\epsilon^{(L)} with ϵ(k)∈{−1,1}dksuperscriptitalic-ϵ𝑘superscript11subscript𝑑𝑘\epsilon^{(k)}\in\{-1,1\}^{d\_{k}}, let Tϵ(k)(k):ℝdk−1→ℝdk:subscriptsuperscript𝑇𝑘superscriptitalic-ϵ𝑘→superscriptℝsubscript𝑑𝑘1superscriptℝsubscript𝑑𝑘{T}^{(k)}\_{\epsilon^{(k)}}:\mathbb{R}^{d\_{k-1}}\rightarrow\mathbb{R}^{d\_{k}} the affine function defined by Tϵ(k)(k)​(𝐮)i=(T(k)​(𝐮))isubscriptsuperscript𝑇𝑘superscriptitalic-ϵ𝑘subscript𝐮𝑖subscriptsuperscript𝑇𝑘𝐮𝑖T^{(k)}\_{\epsilon^{(k)}}(\mathbf{u})\_{i}=(T^{(k)}(\mathbf{u}))\_{i} if (ϵk)i=1subscriptsubscriptitalic-ϵ𝑘𝑖1(\epsilon\_{k})\_{i}=1, and 00 otherwise. ReLU network functions, as defined in Eqn. [1](#S2.E1 "In 2.1 Preliminaries ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks"), can be expressed as

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(𝐱)=∑ϵ(1),⋯​ϵ(L)1Pf,ϵ​(𝐱)​(T(L+1)∘Tϵ(L)(L)∘⋯∘Tϵ(1)(1))​(𝐱)𝑓𝐱subscript  superscriptitalic-ϵ1⋯superscriptitalic-ϵ𝐿subscript1subscript𝑃  𝑓italic-ϵ𝐱superscript𝑇𝐿1subscriptsuperscript𝑇𝐿superscriptitalic-ϵ𝐿⋯subscriptsuperscript𝑇1superscriptitalic-ϵ1𝐱f(\mathbf{x})=\sum\_{\epsilon^{(1)},\cdots\epsilon^{(L)}}1\_{P\_{f,\epsilon}}(\mathbf{x})\,\left(T^{(L+1)}\circ T^{(L)}\_{\epsilon^{(L)}}\circ\cdots\circ T^{(1)}\_{\epsilon^{(1)}}\right)(\mathbf{x}) |  | (22) |

where 1Psubscript1𝑃1\_{P} denotes the indicator function of the subset P⊂ℝd𝑃superscriptℝ𝑑P\subset\mathbb{R}^{d}, and Pf,ϵsubscript𝑃

𝑓italic-ϵP\_{f,\epsilon} is the polytope defined as the set of solutions of the following linear inequalities (for all k=1,⋯,L𝑘

1⋯𝐿k=1,\cdots,L):

|  |  |  |  |
| --- | --- | --- | --- |
|  | (ϵk)i​(T(k)∘Tϵ(k−1)(k−1)∘⋯∘Tϵ(1)(1))​(𝐱)i≥0,i=1,⋯​dkformulae-sequencesubscriptsubscriptitalic-ϵ𝑘𝑖superscript𝑇𝑘subscriptsuperscript𝑇𝑘1superscriptitalic-ϵ𝑘1⋯subscriptsuperscript𝑇1superscriptitalic-ϵ1subscript𝐱𝑖0𝑖  1⋯subscript𝑑𝑘(\epsilon\_{k})\_{i}\,(T^{(k)}\circ T^{(k-1)}\_{\epsilon^{({k-1})}}\circ\cdots\circ T^{(1)}\_{\epsilon^{(1)}})(\mathbf{x})\_{i}\geq 0,\quad i=1,\cdots d\_{k} |  | (23) |

f𝑓f is therefore affine on each of the polytopes Pf,ϵsubscript𝑃

𝑓italic-ϵP\_{f,\epsilon}, which finitely partition the input space ℝdsuperscriptℝ𝑑\mathbb{R}^{d} to convex polytopes. Remarkably, the correspondence between ReLU networks and CPWL functions goes both ways: Arora et al. ([2018](#bib.bib1)) show that every CPWL function can be represented by a ReLU network, which in turn endows ReLU networks with the universal approximation property.

Finally, in the standard basis, each affine map T(k):ℝdk−1→ℝdk:superscript𝑇𝑘→superscriptℝsubscript𝑑𝑘1superscriptℝsubscript𝑑𝑘{T}^{(k)}:\mathbb{R}^{d\_{k-1}}\rightarrow\mathbb{R}^{d\_{k}} is specified by a weight matrix W(k)∈ℝdk−1×ℝdksuperscript𝑊𝑘superscriptℝsubscript𝑑𝑘1superscriptℝsubscript𝑑𝑘W^{(k)}\in\mathbb{R}^{d\_{k-1}}\times\mathbb{R}^{d\_{k}} and a bias vector b(k)∈ℝdksuperscript𝑏𝑘superscriptℝsubscript𝑑𝑘b^{(k)}\in\mathbb{R}^{d\_{k}}.
In the linear region Pf,ϵsubscript𝑃

𝑓italic-ϵP\_{f,\epsilon}, f𝑓f can be expressed as
fϵ​(x)=Wϵ​x+bϵsubscript𝑓italic-ϵ𝑥subscript𝑊italic-ϵ𝑥subscript𝑏italic-ϵf\_{\epsilon}(x)=W\_{\epsilon}x+b\_{\epsilon}, where in particular

|  |  |  |  |
| --- | --- | --- | --- |
|  | Wϵ=W(L+1)​WϵL(L)​⋯​Wϵ1(1)∈ℝ1×d,subscript𝑊italic-ϵsuperscript𝑊𝐿1subscriptsuperscript𝑊𝐿subscriptitalic-ϵ𝐿⋯subscriptsuperscript𝑊1subscriptitalic-ϵ1superscriptℝ1𝑑W\_{\epsilon}=W^{(L+1)}W^{(L)}\_{\epsilon\_{L}}\cdots W^{(1)}\_{\epsilon\_{1}}\in\mathbb{R}^{1\times d}, |  | (24) |

where Wϵ(k)subscriptsuperscript𝑊𝑘italic-ϵW^{(k)}\_{\epsilon} is obtained from W(k)superscript𝑊𝑘W^{(k)} by setting its j𝑗jth column to zero whenever (ϵk)j=−1subscriptsubscriptitalic-ϵ𝑘𝑗1(\epsilon\_{k})\_{j}=-1.

## Appendix C Fourier Analysis of ReLU Networks

### C.1 Proof of Lemma [1](#Thmlemma1 "Lemma 1. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks")

###### Proof.

Case 1: The function f𝑓f has compact support. The vector-valued function 𝐤​f​(𝐱)​ei​𝐤⋅𝐱𝐤𝑓𝐱superscript𝑒⋅𝑖𝐤𝐱\mathbf{k}f(\mathbf{x})e^{i\mathbf{k}\cdot\mathbf{x}} is continuous everywhere and has well-defined and continuous gradients almost everywhere. So by Stokes’ theorem (see e.g Spivak ([2018](#bib.bib35))), the integral of its divergence is a pure boundary term. Since we restricted to functions with compact support, the theorem yields

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∫∇𝐱⋅[𝐤​f​(𝐱)​e−i​𝐤⋅𝐱]​𝐝𝐱=0⋅subscript∇𝐱delimited-[]𝐤𝑓𝐱superscript𝑒⋅𝑖𝐤𝐱𝐝𝐱0\int\nabla\_{\mathbf{x}}\cdot\left[\mathbf{k}f(\mathbf{x})e^{-i\mathbf{k}\cdot\mathbf{x}}\right]\mathbf{dx}=0 |  | (25) |

The integrand is (𝐤⋅(∇𝐱f)​(𝐱)−i​k2​f​(𝐱))​e−i​𝐤⋅𝐱⋅𝐤subscript∇𝐱𝑓𝐱𝑖superscript𝑘2𝑓𝐱superscript𝑒⋅𝑖𝐤𝐱(\mathbf{k}\cdot(\nabla\_{\mathbf{x}}f)(\mathbf{x})-ik^{2}f(\mathbf{x}))e^{-i\mathbf{k}\cdot\mathbf{x}}, so we deduce,

|  |  |  |  |
| --- | --- | --- | --- |
|  | f^​(𝐤)=1−i​k2​𝐤⋅∫(∇𝐱f)​(𝐱)​e−i​𝐤⋅𝐱^𝑓𝐤⋅1𝑖superscript𝑘2𝐤subscript∇𝐱𝑓𝐱superscript𝑒⋅𝑖𝐤𝐱\hat{f}(\mathbf{k})=\frac{1}{-ik^{2}}\mathbf{k}\cdot\!\int(\nabla\_{\mathbf{x}}f)(\mathbf{x})\,e^{-i\mathbf{k}\cdot\mathbf{x}} |  | (26) |

Now, within each polytope of the decomposition ([22](#A2.E22 "In Lemma 3. ‣ Appendix B The Continuous Piecewise Linear Structure of Deep ReLU Networks ‣ On the Spectral Bias of Neural Networks")), f𝑓f is affine so its gradient is a constant vector,
∇𝐱fϵ=WϵTsubscript∇𝐱subscript𝑓italic-ϵsuperscriptsubscript𝑊italic-ϵ𝑇\nabla\_{\mathbf{x}}f\_{\epsilon}=W\_{\epsilon}^{T},
which gives the desired result ([1](#Thmlemma1 "Lemma 1. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks")).

Case 2: The function f𝑓f does not have compact support.
Without the assumption of compact support, the function f𝑓f is not squared-integrable. The Fourier transform therefore only exists in the sense of distributions, as defined below.

Let 𝒮𝒮\mathcal{S} be the Schwartz space over ℝdsuperscriptℝ𝑑\mathbb{R}^{d} of rapidly decaying test functions which together with its derivatives decay to zero as x→∞→𝑥x\rightarrow\infty faster than any power of x𝑥x. A tempered distribution is a continuous linear functional on 𝒮𝒮\mathcal{S}. A function f𝑓f that doesn’t grow faster than a polynomial at infinity can be identified with a tempered distribution Tfsubscript𝑇𝑓T\_{f} as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Tf:𝒮→ℝ,φ↦⟨f,φ⟩=∫ℝdf​(𝐱)​φ​(𝐱)​𝐝𝐱:subscript𝑇𝑓formulae-sequence→𝒮ℝmaps-to𝜑  𝑓𝜑subscriptsuperscriptℝ𝑑𝑓𝐱𝜑𝐱𝐝𝐱T\_{f}:\mathcal{S}\rightarrow\mathbb{R},\,\varphi\mapsto\langle f,\varphi\rangle=\int\_{\mathbb{R}^{d}}f(\mathbf{x})\varphi(\mathbf{x})\mathbf{dx} |  | (27) |

In the following, we shall identify Tfsubscript𝑇𝑓T\_{f} with f𝑓f. The Fourier transform f~~𝑓\tilde{f} of the tempered distribution is defined as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ⟨f~,φ⟩:=⟨f,φ~⟩assign  ~𝑓𝜑  𝑓~𝜑\langle\tilde{f},\varphi\rangle:=\langle f,\tilde{\varphi}\rangle |  | (28) |

where φ~~𝜑\tilde{\varphi} is the Fourier transform of φ𝜑\varphi. In this sense, the standard notion of the Fourier transform is generalized to functions that are not squared-integrable.

Consider the continuous piecewise-linear ReLU network f:ℝd→ℝ:𝑓→superscriptℝ𝑑ℝf:\mathbb{R}^{d}\rightarrow\mathbb{R}. Since it can grow at most linearly, we interpret it as a tempered distribution on ℝdsuperscriptℝ𝑑\mathbb{R}^{d}.
Recall that the linear regions Pϵsubscript𝑃italic-ϵP\_{\epsilon} are enumerated by ϵitalic-ϵ\epsilon. Let fϵsubscript𝑓italic-ϵf\_{\epsilon} be the restriction of f𝑓f to Pϵsubscript𝑃italic-ϵP\_{\epsilon}, making fϵ​(𝐱)=WϵT​𝐱subscript𝑓italic-ϵ𝐱superscriptsubscript𝑊italic-ϵ𝑇𝐱f\_{\epsilon}(\mathbf{x})=W\_{\epsilon}^{T}\mathbf{x}. The distributional derivative of f𝑓f is given by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇𝐱f=∑ϵ∇𝐱fϵ⋅1Pϵ=∑ϵWϵT​1Pϵsubscript∇𝐱𝑓subscriptitalic-ϵsubscript∇𝐱⋅subscript𝑓italic-ϵsubscript1subscript𝑃italic-ϵsubscriptitalic-ϵsuperscriptsubscript𝑊italic-ϵ𝑇subscript1subscript𝑃italic-ϵ\nabla\_{\mathbf{x}}f=\sum\_{\epsilon}\nabla\_{\mathbf{x}}f\_{\epsilon}\cdot 1\_{P\_{\epsilon}}=\sum\_{\epsilon}W\_{\epsilon}^{T}1\_{P\_{\epsilon}} |  | (29) |

where 1Pϵsubscript1subscript𝑃italic-ϵ1\_{P\_{\epsilon}} is the indicator over Pϵsubscript𝑃italic-ϵP\_{\epsilon} and we used ∇𝐱fϵ=WϵTsubscript∇𝐱subscript𝑓italic-ϵsuperscriptsubscript𝑊italic-ϵ𝑇\nabla\_{\mathbf{x}}f\_{\epsilon}=W\_{\epsilon}^{T}.
It then follows from elementary properties of Schwartz spaces (see e.g. Chapter 16 of Serov ([2017](#bib.bib32))) that:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | [∇𝐱f~]​(𝐤)delimited-[]~subscript∇𝐱𝑓𝐤\displaystyle[\widetilde{\nabla\_{\mathbf{x}}f}](\mathbf{k}) | =−i​𝐤​f~​(𝐤)absent𝑖𝐤~𝑓𝐤\displaystyle=-i\mathbf{k}\tilde{f}(\mathbf{k}) |  | (30) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ⟹f~​(𝐤)absent~𝑓𝐤\displaystyle\implies\tilde{f}(\mathbf{k}) | =1−i​k2​𝐤⋅[∇𝐱f~]​(𝐤)absent⋅1𝑖superscript𝑘2𝐤delimited-[]~subscript∇𝐱𝑓𝐤\displaystyle=\frac{1}{-ik^{2}}\mathbf{k}\cdot[\widetilde{\nabla\_{\mathbf{x}}f}](\mathbf{k}) |  | (31) |

Together with Eqn [29](#A3.E29 "In Proof. ‣ C.1 Proof of Lemma 1 ‣ Appendix C Fourier Analysis of ReLU Networks ‣ On the Spectral Bias of Neural Networks") and linearity of the Fourier transform, this gives the desired result ([1](#Thmlemma1 "Lemma 1. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks")).

∎

### C.2 Fourier Transform of Polytopes

#### C.2.1 Theorem 1 of Diaz et al. ([2016](#bib.bib10))

Let F𝐹F be a m𝑚m dimensional polytope in ℝdsuperscriptℝ𝑑\mathbb{R}^{d}, such that 1≤m≤d1𝑚𝑑1\leq m\leq d. Denote by 𝐤∈ℝd𝐤superscriptℝ𝑑\mathbf{k}\in\mathbb{R}^{d} a vector in the Fourier space, by ϕ𝐤​(x)=−𝐤⋅𝐱subscriptitalic-ϕ𝐤𝑥⋅𝐤𝐱\phi\_{\mathbf{k}}(x)=-\mathbf{k}\cdot\mathbf{x} the linear phase function, by F~~𝐹\tilde{F} the Fourier transform of the indicator function on F𝐹F, by ∂F𝐹\partial F the boundary of F𝐹F and by volmsubscriptvol𝑚\text{vol}\_{m} the m𝑚m-dimensional (Hausdorff) measure. Let ProjF​(𝐤)subscriptProj𝐹𝐤\text{Proj}\_{F}(\mathbf{k}) be the orthogonal projection of 𝐤𝐤\mathbf{k} on to F𝐹F (obtained by removing all components of 𝐤𝐤\mathbf{k} orthogonal to F𝐹F). Given a m−1𝑚1m-1 dimensional facet G𝐺G of F𝐹F, let 𝐍F​(G)subscript𝐍𝐹𝐺\mathbf{N}\_{F}(G) be the unit normal vector to G𝐺G that points out of F𝐹F. It then holds:

1. If ProjF​(𝐤)=0subscriptProj𝐹𝐤0\text{Proj}\_{F}(\mathbf{k})=0, then ϕ𝐤​(x)=Φ𝐤subscriptitalic-ϕ𝐤𝑥subscriptΦ𝐤\phi\_{\mathbf{k}}(x)=\Phi\_{\mathbf{k}} is constant on F𝐹F, and we have:

|  |  |  |  |
| --- | --- | --- | --- |
|  | F~=volF​(F)​ei​Φ𝐤~𝐹subscriptvol𝐹𝐹superscript𝑒𝑖subscriptΦ𝐤\tilde{F}=\text{vol}\_{F}(F)e^{i\Phi\_{\mathbf{k}}} |  | (32) |

2. But if ProjF​(𝐤)≠0subscriptProj𝐹𝐤0\text{Proj}\_{F}(\mathbf{k})\neq 0, then:

|  |  |  |  |
| --- | --- | --- | --- |
|  | F~=i​∑G∈∂FProjF​(𝐤)⋅𝐍F​(G)‖ProjF​(𝐤)‖2​G~​(𝐤)~𝐹𝑖subscript𝐺𝐹⋅subscriptProj𝐹𝐤subscript𝐍𝐹𝐺superscriptnormsubscriptProj𝐹𝐤2~𝐺𝐤\tilde{F}=i\sum\_{G\in\partial F}\frac{\text{Proj}\_{F}(\mathbf{k})\cdot\mathbf{N}\_{F}(G)}{\|\text{Proj}\_{F}(\mathbf{k})\|^{2}}\tilde{G}(\mathbf{k}) |  | (33) |

#### C.2.2 Discussion

The above theorem provides a recursive relation for computing the Fourier transform of an arbitrary polytope. More precisely, the Fourier transform of a m𝑚m-dimensional polytope is expressed as a sum of fourier transforms over the m−1𝑚1m-1 dimensional boundaries of the said polytope (which are themselves polytopes) times a 𝒪​(k−1)𝒪superscript𝑘1\mathcal{O}(k^{-1}) *weight* term (with k=‖𝐤‖𝑘norm𝐤k=\|\mathbf{k}\|). The recursion terminates if ProjF​(𝐤)=0subscriptProj𝐹𝐤0\text{Proj}\_{F}(\mathbf{k})=0, which then yields a constant.

To structure this computation, Diaz et al. ([2016](#bib.bib10)) introduce a book-keeping device called the *face poset* of the polytope. It can be understood as a weighted directed acyclic graph (DAG) with polytopes of various dimensions as its nodes. We start at the root node which is the full dimensional polytope P𝑃P (i.e. we initially set m=n𝑚𝑛m=n). For all of the codimension-one boundary faces F𝐹F of P𝑃P, we then draw an edge from the root P𝑃P to node F𝐹F and weight it with a term given by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | WF,G=i​ProjF​(𝐤)⋅𝐍F​(G)‖ProjF​(𝐤)‖2subscript𝑊  𝐹𝐺𝑖⋅subscriptProj𝐹𝐤subscript𝐍𝐹𝐺superscriptnormsubscriptProj𝐹𝐤2W\_{F,G}=i\frac{\text{Proj}\_{F}(\mathbf{k})\cdot\mathbf{N}\_{F}(G)}{\|\text{Proj}\_{F}(\mathbf{k})\|^{2}} |  | (34) |

and repeat the process iteratively for each F𝐹F. Note that the weight term is 𝒪​(k−1)𝒪superscript𝑘1\mathcal{O}(k^{-1}) where ProjF​(𝐤)≠0subscriptProj𝐹𝐤0\text{Proj}\_{F}(\mathbf{k})\neq 0. This process yields tree paths T:F0=P→F1→…→F|T|:𝑇subscript𝐹0𝑃→subscript𝐹1→…→subscript𝐹𝑇T:F\_{0}=P\rightarrow F\_{1}\rightarrow...\rightarrow F\_{|T|} where each Fi+1∈∂Fisubscript𝐹𝑖1subscript𝐹𝑖F\_{i+1}\in\partial F\_{i} has one dimension less than Fisubscript𝐹𝑖F\_{i}. For a given path and 𝐤𝐤\mathbf{k}, the terminal node for this path, FnTsubscript𝐹subscript𝑛𝑇F\_{n\_{T}}, is the first polytope for which ProjFnT​(𝐤)=0subscriptProjsubscript𝐹subscript𝑛𝑇𝐤0\text{Proj}\_{F\_{n\_{T}}}(\mathbf{k})=0.
The final Fourier transform is obtained by multiplying the weights along each path and summing over all tree paths:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1~P​(𝐤)=∑T∏i=0|T|−1WFi,Fi+1​volF|T|​(F|T|)​ei​Φ𝐤subscript~1𝑃𝐤subscript𝑇superscriptsubscriptproduct𝑖0𝑇1subscript𝑊  subscript𝐹𝑖subscript𝐹𝑖1subscriptvolsubscript𝐹𝑇subscript𝐹𝑇superscript𝑒𝑖subscriptΦ𝐤\tilde{1}\_{P}(\mathbf{k})=\sum\_{T}\prod\_{i=0}^{|T|-1}W\_{F\_{i},F\_{i+1}}\text{vol}\_{F\_{|T|}}(F\_{|T|})e^{i\Phi\_{\mathbf{k}}} |  | (35) |

where Φ(T)=𝐤⋅𝐱0TsuperscriptΦ𝑇⋅𝐤superscriptsubscript𝐱0𝑇\Phi^{(T)}=\mathbf{k}\cdot\mathbf{x}\_{0}^{T} for an arbitrary point 𝐱0Tsuperscriptsubscript𝐱0𝑇\mathbf{x}\_{0}^{T} in F|T|subscript𝐹𝑇F\_{|T|}.

To write this as a weighted sum of indicator functions, as in Lemma [2](#Thmlemma2 "Lemma 2. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks"), let 𝒯nsubscript𝒯𝑛\mathcal{T}\_{n} denote the set of all tree paths T𝑇T of length n𝑛n, i.e. |T|=n𝑇𝑛|T|=n. For a tree path T𝑇T, let S​(T)𝑆𝑇S(T) be the orthogonal to the terminal node Fnsubscript𝐹𝑛F\_{n}, i.e the vectors 𝐤𝐤\mathbf{k} such that ProjFn​(𝐤)=0subscriptProjsubscript𝐹𝑛𝐤0\text{Proj}\_{F\_{n}}(\mathbf{k})=0. The sum over T𝑇T in Eqn ([35](#A3.E35 "In C.2.2 Discussion ‣ C.2 Fourier Transform of Polytopes ‣ Appendix C Fourier Analysis of ReLU Networks ‣ On the Spectral Bias of Neural Networks")) can be split as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1~P=∑n=0d1Gnkn​∑T∈𝒯n1S​(T)​∏i=0n−1W¯FiT,Fi+1T​volFnT​(FnT)​ei​Φ𝐤(T)subscript~1𝑃superscriptsubscript𝑛0𝑑subscript1subscript𝐺𝑛superscript𝑘𝑛subscript𝑇subscript𝒯𝑛subscript1𝑆𝑇superscriptsubscriptproduct𝑖0𝑛1subscript¯𝑊  subscriptsuperscript𝐹𝑇𝑖subscriptsuperscript𝐹𝑇𝑖1subscriptvolsubscriptsuperscript𝐹𝑇𝑛subscriptsuperscript𝐹𝑇𝑛superscript𝑒𝑖subscriptsuperscriptΦ𝑇𝐤\tilde{1}\_{P}=\sum\_{n=0}^{d}\frac{1\_{G\_{n}}}{k^{n}}\sum\_{T\in\mathcal{T}\_{n}}1\_{S(T)}\prod\_{i=0}^{n-1}\bar{W}\_{F^{T}\_{i},F^{T}\_{i+1}}\text{vol}\_{F^{T}\_{n}}(F^{T}\_{n})e^{i\Phi^{(T)}\_{\mathbf{k}}} |  | (36) |

where W¯F,G=k​WF,Gsubscript¯𝑊

𝐹𝐺𝑘subscript𝑊

𝐹𝐺\bar{W}\_{F,G}=kW\_{F,G} and
Gn=⋃T∈𝒯nS​(T)subscript𝐺𝑛subscript𝑇subscript𝒯𝑛𝑆𝑇G\_{n}=\bigcup\_{T\in\mathcal{T}\_{n}}S(T). In words, Gnsubscript𝐺𝑛G\_{n} is the set of all vectors 𝐤𝐤\mathbf{k} that are orthogonal to some n𝑛n-codimensional face of the polytope.
We identify:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Dq=∑T∈𝒯n1S​(T)​∏i=0n−1W¯FiT,Fi+1T​volFnT​(FnT)​ei​Φ𝐤(T)subscript𝐷𝑞subscript𝑇subscript𝒯𝑛subscript1𝑆𝑇superscriptsubscriptproduct𝑖0𝑛1subscript¯𝑊  subscriptsuperscript𝐹𝑇𝑖subscriptsuperscript𝐹𝑇𝑖1subscriptvolsubscriptsuperscript𝐹𝑇𝑛subscriptsuperscript𝐹𝑇𝑛superscript𝑒𝑖subscriptsuperscriptΦ𝑇𝐤D\_{q}=\sum\_{T\in\mathcal{T}\_{n}}1\_{S(T)}\prod\_{i=0}^{n-1}\bar{W}\_{F^{T}\_{i},F^{T}\_{i+1}}\text{vol}\_{F^{T}\_{n}}(F^{T}\_{n})e^{i\Phi^{(T)}\_{\mathbf{k}}} |  | (37) |

and D0​(𝐤)=vol​(P)subscript𝐷0𝐤vol𝑃D\_{0}(\mathbf{k})=\text{vol}(P) to obtain Lemma [2](#Thmlemma2 "Lemma 2. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks"). Observe that Dnsubscript𝐷𝑛D\_{n} depends on k𝑘k only via the phase term ei​Φ𝐤(T)superscript𝑒𝑖subscriptsuperscriptΦ𝑇𝐤e^{i\Phi^{(T)}\_{\mathbf{k}}}, implying that Dn=Θ​(1)​(k→∞)subscript𝐷𝑛Θ1→𝑘D\_{n}=\Theta(1)\,(k\rightarrow\infty).

Informally, for a generic vector 𝐤𝐤\mathbf{k}, all paths terminate at the zero-dimensional vertices of the original polytope, i.e. dim​(Fn)=0dimsubscript𝐹𝑛0\text{dim}(F\_{n})=0, implying the length of the path n𝑛n equals the number of dimensions d𝑑d, yielding a 𝒪​(k−d)𝒪superscript𝑘𝑑\mathcal{O}(k^{-d}) spectrum. The exceptions occur if a path terminates prematurely, because 𝐤𝐤\mathbf{k} happens to lie orthogonal to some d−r𝑑𝑟d-r-dimensional face Frsubscript𝐹𝑟F\_{r} in the path, in which case we are left with a 𝒪​(k−r)𝒪superscript𝑘𝑟\mathcal{O}(k^{-r}) term (with r<d𝑟𝑑r<d) which dominates asymptotically. Note that all vectors orthogonal to the d−r𝑑𝑟d-r dimensional face Frsubscript𝐹𝑟F\_{r} lie on a r𝑟r-dimensional subspace of ℝdsuperscriptℝ𝑑\mathbb{R}^{d}. Since a polytope has a finite number of faces (of any dimension), the 𝐤𝐤\mathbf{k}’s for which the Fourier transform is 𝒪​(k−r)𝒪superscript𝑘𝑟\mathcal{O}(k^{-r}) (instead of 𝒪​(k−d)𝒪superscript𝑘𝑑\mathcal{O}(k^{-d})) lies on a finite union of closed subspaces of dimension r𝑟r (with r<d𝑟𝑑r<d). The Lebesgue measure of all such lower dimensional subspaces for all such r𝑟r is 00, leading us to the conclusion that the spectrum decays as 𝒪​(k−d)𝒪superscript𝑘𝑑\mathcal{O}(k^{-d}) for *almost all* 𝐤𝐤\mathbf{k}’s.

### C.3 On Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks")

Equation [6](#S2.E6 "In Theorem 1. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks") can be obtained by swapping the (finite) sum over ϵitalic-ϵ\epsilon in Lemma [1](#Thmlemma1 "Lemma 1. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks") with that over the paths T𝑇T in Eqn [36](#A3.E36 "In C.2.2 Discussion ‣ C.2 Fourier Transform of Polytopes ‣ Appendix C Fourier Analysis of ReLU Networks ‣ On the Spectral Bias of Neural Networks"). In particular, we have:

|  |  |  |  |
| --- | --- | --- | --- |
|  | f~=∑n=0d1Hnkn+1​∑ϵWϵ​Dnϵ​1Gnϵ~𝑓superscriptsubscript𝑛0𝑑subscript1subscript𝐻𝑛superscript𝑘𝑛1subscriptitalic-ϵsubscript𝑊italic-ϵsuperscriptsubscript𝐷𝑛italic-ϵsubscript1superscriptsubscript𝐺𝑛italic-ϵ\tilde{f}=\sum\_{n=0}^{d}\frac{1\_{H\_{n}}}{k^{n+1}}\sum\_{\epsilon}W\_{\epsilon}D\_{n}^{\epsilon}1\_{G\_{n}^{\epsilon}} |  | (38) |

Now, the sum ∑ϵWϵ​Dnϵ​(𝐤^)​IGnϵ​(𝐤)subscriptitalic-ϵsubscript𝑊italic-ϵsuperscriptsubscript𝐷𝑛italic-ϵ^𝐤subscript𝐼superscriptsubscript𝐺𝑛italic-ϵ𝐤\sum\_{\epsilon}W\_{\epsilon}D\_{n}^{\epsilon}(\hat{\mathbf{k}})I\_{G\_{n}^{\epsilon}}(\mathbf{k}) is supported on the union:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Hn=⋃ϵGnϵsubscript𝐻𝑛subscriptitalic-ϵsuperscriptsubscript𝐺𝑛italic-ϵH\_{n}=\bigcup\_{\epsilon}G\_{n}^{\epsilon} |  | (39) |

Identifying:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Cn​(⋅,θ)=∑ϵWϵ​Dnϵ​1Gnϵsubscript𝐶𝑛⋅𝜃subscriptitalic-ϵsubscript𝑊italic-ϵsuperscriptsubscript𝐷𝑛italic-ϵsubscript1superscriptsubscript𝐺𝑛italic-ϵ\displaystyle C\_{n}(\cdot,\theta)=\sum\_{\epsilon}W\_{\epsilon}D\_{n}^{\epsilon}1\_{G\_{n}^{\epsilon}} |  | (40) |

where Cn​(⋅,θ)=𝒪​(1)​(k→∞)subscript𝐶𝑛⋅𝜃𝒪1→𝑘C\_{n}(\cdot,\theta)=\mathcal{O}(1)\,(k\rightarrow\infty), we obtain Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks"). Further, if Nfsubscript𝑁𝑓N\_{f} is the number of linear regions of the network and Lf=maxϵ⁡‖Wϵ‖subscript𝐿𝑓subscriptitalic-ϵnormsubscript𝑊italic-ϵL\_{f}=\max\_{\epsilon}\|W\_{\epsilon}\|, we see that Cn=𝒪​(Lf​Nf)subscript𝐶𝑛𝒪subscript𝐿𝑓subscript𝑁𝑓C\_{n}=\mathcal{O}(L\_{f}N\_{f}). Indeed, in Appendix [A.5](#A1.SS5 "A.5 Qualitative Ablation over Architectures ‣ Appendix A Experimental Details ‣ On the Spectral Bias of Neural Networks"), we empirically find that relaxing the constraint on the weight clip (which can be identified with Lfsubscript𝐿𝑓L\_{f}) enabled the network to fit higher frequencies, implying that the 𝒪​(Lf)𝒪subscript𝐿𝑓\mathcal{O}(L\_{f}) bound can be tight.

### C.4 Spectral Decay Rate of the Parameter Gradient

###### Proposition 1.

Let θ𝜃\theta be a generic parameter of the network function f𝑓f. The spectral decay rate of ∂f~/∂θ~𝑓𝜃\nicefrac{{\partial\tilde{f}}}{{\partial\theta}} is 𝒪​(k​f~)𝒪𝑘~𝑓\mathcal{O}(k\tilde{f}).

###### Proof.

For a fixed 𝐤^^𝐤\hat{\mathbf{k}}, observe from Eqn [38](#A3.E38 "In C.3 On Theorem 1 ‣ Appendix C Fourier Analysis of ReLU Networks ‣ On the Spectral Bias of Neural Networks") and Eqn [37](#A3.E37 "In C.2.2 Discussion ‣ C.2 Fourier Transform of Polytopes ‣ Appendix C Fourier Analysis of ReLU Networks ‣ On the Spectral Bias of Neural Networks") that the only terms dependent on k𝑘k are the pure powers k−n−1superscript𝑘𝑛1k^{-n-1} and the phase terms ei​Φ𝐤(T)superscript𝑒𝑖superscriptsubscriptΦ𝐤𝑇e^{i\Phi\_{\mathbf{k}}^{(T)}}, where Φ𝐤(T)=k​𝐤^⋅𝐱0q​(T)superscriptsubscriptΦ𝐤𝑇⋅𝑘^𝐤superscriptsubscript𝐱0𝑞𝑇\Phi\_{\mathbf{k}}^{(T)}=k\hat{\mathbf{k}}\cdot\mathbf{x}\_{0}^{q(T)}. However, the term 𝐱0q​(T)superscriptsubscript𝐱0𝑞𝑇\mathbf{x}\_{0}^{q(T)} is in general a function of θ𝜃\theta, and consequently the partial derivative of ei​Φ𝐤(T)superscript𝑒𝑖superscriptsubscriptΦ𝐤𝑇e^{i\Phi\_{\mathbf{k}}^{(T)}} w.r.t θ𝜃\theta yields a term that is proportional to k𝑘k. This term now dominates the asymptotic behaviour as k→∞→𝑘k\rightarrow\infty, adding an extra power of k𝑘k to the total spectral decay rate of f~~𝑓\tilde{f}.
∎

Therefore, if f=𝒪​(k−Δ−1)𝑓𝒪superscript𝑘Δ1f=\mathcal{O}(k^{-\Delta-1}) where ΔΔ\Delta is the codimension of the highest dimensional polytope 𝐤^^𝐤\hat{\mathbf{k}} is orthogonal to, we have that ∂f/∂θ=𝒪​(k−Δ)𝑓𝜃𝒪superscript𝑘Δ\nicefrac{{\partial f}}{{\partial\theta}}=\mathcal{O}(k^{-\Delta}).

### C.5 Convergence Rate of a Network Trained on Pure-Frequency Targets

In this section, we derive an asymptotic bound on the convergence rate under the assumption that the target function has only one frequency component.

###### Proposition 2.

Let λ:[0,1]→ℝ:𝜆→01ℝ\lambda:[0,1]\rightarrow\mathbb{R} be a target function sampled in its domain at N𝑁N uniformly spaced points. Suppose that its Fourier transform after sampling takes the form: λ~​(k)=A0​δk,k0~𝜆𝑘subscript𝐴0subscript𝛿

𝑘subscript𝑘0\tilde{\lambda}(k)=A\_{0}\delta\_{k,k\_{0}}, where δ𝛿\delta is the Kronecker delta. Let f𝑓f be a neural network trained with full-batch gradient descent with learning rate η𝜂\eta on the Mean Squared Error, and denote by ftsubscript𝑓𝑡f\_{t} the state of the network at time t𝑡t. Let h​(⋅,t)=ft−λℎ⋅𝑡subscript𝑓𝑡𝜆h(\cdot,t)=f\_{t}-\lambda be the residual at time t𝑡t. We have that:

|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂h~​(k0,t)∂t|=𝒪​(k0−1)~ℎsubscript𝑘0𝑡𝑡𝒪superscriptsubscript𝑘01\left|\frac{\partial\tilde{h}(k\_{0},t)}{\partial t}\right|=\mathcal{O}(k\_{0}^{-1}) |  | (41) |

###### Proof.

Consider that:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | |∂h~​(k0)∂t|~ℎsubscript𝑘0𝑡\displaystyle\left|\frac{\partial\tilde{h}(k\_{0})}{\partial t}\right| | =|∂f~​(k0)∂θ|​|∂θ∂t|absent~𝑓subscript𝑘0𝜃𝜃𝑡\displaystyle=\left|\frac{\partial\tilde{f}(k\_{0})}{\partial\theta}\right|\left|\frac{\partial\theta}{\partial t}\right| |  | (42) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =|η​∂f~∂θ|​|∂ℒ​[f~,λ~]∂θ|absent𝜂~𝑓𝜃ℒ~𝑓~𝜆𝜃\displaystyle=\left|\eta\frac{\partial\tilde{f}}{\partial\theta}\right|\left|\frac{\partial\mathcal{L}[\tilde{f},\tilde{\lambda}]}{\partial\theta}\right| |  | (43) |

where ℒℒ\mathcal{L} is the sampled MSE loss and the first term is 𝒪​(k0−1)𝒪superscriptsubscript𝑘01\mathcal{O}(k\_{0}^{-1}) as can be seen from Proposition [1](#Thmproposition1 "Proposition 1. ‣ C.4 Spectral Decay Rate of the Parameter Gradient ‣ Appendix C Fourier Analysis of ReLU Networks ‣ On the Spectral Bias of Neural Networks"). With Parceval’s Theorem, we obtain:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ​[f,λ]ℒ𝑓𝜆\displaystyle\mathcal{L}[f,\lambda] | =∑x=0N−1|f​(x)−λ​(x)|2=∑k=−N/2N/2−1|f~​(k)−λ~​(k)|2absentsuperscriptsubscript𝑥0𝑁1superscript𝑓𝑥𝜆𝑥2superscriptsubscript𝑘𝑁2𝑁21superscript~𝑓𝑘~𝜆𝑘2\displaystyle=\sum\_{x=0}^{N-1}|f(x)-\lambda(x)|^{2}=\sum\_{k=-\nicefrac{{N}}{{2}}}^{\nicefrac{{N}}{{2}}-1}|\tilde{f}(k)-\tilde{\lambda}(k)|^{2} |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =ℒ​[f~,λ~]absentℒ~𝑓~𝜆\displaystyle=\mathcal{L}[\tilde{f},\tilde{\lambda}] |  | (44) |

For the magnitude of parameter gradient, we obtain:

|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂ℒ​[f~,λ~]∂θ|ℒ~𝑓~𝜆𝜃\displaystyle\left|\frac{\partial\mathcal{L}[\tilde{f},\tilde{\lambda}]}{\partial\theta}\right| | =2​|∑k=−N/2N/2−1Re​[f~​(k)−λ~​(k)]​∂f~​(k)∂θ|absent2superscriptsubscript𝑘𝑁2𝑁21Redelimited-[]~𝑓𝑘~𝜆𝑘~𝑓𝑘𝜃\displaystyle=2\left|\sum\_{k=-\nicefrac{{N}}{{2}}}^{\nicefrac{{N}}{{2}}-1}\text{Re}[\tilde{f}(k)-\tilde{\lambda}(k)]\frac{\partial\tilde{f}(k)}{\partial\theta}\right| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤2​∑k=−N/2N/2−1|f~​(k)−λ~​(k)|​|∂f~​(k)∂θ|absent2superscriptsubscript𝑘𝑁2𝑁21~𝑓𝑘~𝜆𝑘~𝑓𝑘𝜃\displaystyle\leq 2\sum\_{k=-\nicefrac{{N}}{{2}}}^{\nicefrac{{N}}{{2}}-1}|\tilde{f}(k)-\tilde{\lambda}(k)|\left|\frac{\partial\tilde{f}(k)}{\partial\theta}\right| |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ≤2​|A0​∂f~​(k0)∂θ|+2​∑k=−N/2N/2−1|f~​(k)​∂f~​(k)∂θ|absent2subscript𝐴0~𝑓subscript𝑘0𝜃2superscriptsubscript𝑘𝑁2𝑁21~𝑓𝑘~𝑓𝑘𝜃\displaystyle\leq 2\left|A\_{0}\frac{\partial\tilde{f}(k\_{0})}{\partial\theta}\right|+2\sum\_{k=-\nicefrac{{N}}{{2}}}^{\nicefrac{{N}}{{2}}-1}\left|\tilde{f}(k)\frac{\partial\tilde{f}(k)}{\partial\theta}\right| |  | (45) |

where in the last line we used that λ~~𝜆\tilde{\lambda} is a Kronecker-δ𝛿\delta in the Fourier domain. Now, the second summand does not depend on k0subscript𝑘0k\_{0}, but the first summand is again 𝒪​(k0−1)𝒪superscriptsubscript𝑘01\mathcal{O}(k\_{0}^{-1}).
∎

### C.6 Proof of the Lipschtiz bound

###### Proposition 3.

The Lipschitz constant Lfsubscript𝐿𝑓L\_{f} of the ReLU network f𝑓f is bound as follows (for all ϵitalic-ϵ\epsilon):

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖Wϵ‖≤Lf≤∏k=1L+1‖W(k)‖≤‖θ‖∞L+1​d​∏k=1Ldknormsubscript𝑊italic-ϵsubscript𝐿𝑓superscriptsubscriptproduct𝑘1𝐿1normsuperscript𝑊𝑘superscriptsubscriptnorm𝜃𝐿1𝑑superscriptsubscriptproduct𝑘1𝐿subscript𝑑𝑘\|W\_{\epsilon}\|\leq L\_{f}\leq\prod\_{k=1}^{L+1}\|W^{(k)}\|\leq\|\theta\|\_{\infty}^{L+1}\sqrt{d}\prod\_{k=1}^{L}d\_{k} |  | (46) |

###### Proof.

The first equality is simply the fact that Lf=maxϵ⁡‖Wϵ‖subscript𝐿𝑓subscriptitalic-ϵnormsubscript𝑊italic-ϵL\_{f}=\max\_{\epsilon}\|W\_{\epsilon}\|, and the second inequality follows trivially from the parameterization of a ReLU network as a chain of function compositions191919Recall that the Lipschitz constant of a composition of two or more functions is the product of their respective Lipschtiz constants., together with the fact that the Lipschitz constant of the ReLU function is 1 (cf. (Miyato et al., [2018](#bib.bib23)), equation 7). To see the third inequality, consider the definition of the spectral norm of a I×J𝐼𝐽I\times J matrix W𝑊W:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖W‖=max‖𝐡‖=1⁡‖W​𝐡‖norm𝑊subscriptnorm𝐡1norm𝑊𝐡\|W\|=\max\_{\|\mathbf{h}\|=1}\|W\mathbf{h}\| |  | (47) |

Now, ‖W​𝐡‖=∑i|𝐰i⋅𝐡|norm𝑊𝐡subscript𝑖⋅subscript𝐰𝑖𝐡\|W\mathbf{h}\|=\sqrt{\sum\_{i}|\mathbf{w}\_{i}\cdot\mathbf{h}|}, where 𝐰isubscript𝐰𝑖\mathbf{w}\_{i} is the i𝑖i-th row of the weight matrix W𝑊W and i=1,…,I𝑖

1…𝐼i=1,...,I. Further, if ‖𝐡‖=1norm𝐡1\|\mathbf{h}\|=1, we have |𝐰i⋅𝐡|≤‖𝐰i‖​‖𝐡‖=‖𝐰i‖⋅subscript𝐰𝑖𝐡normsubscript𝐰𝑖norm𝐡normsubscript𝐰𝑖|\mathbf{w}\_{i}\cdot\mathbf{h}|\leq\|\mathbf{w}\_{i}\|\|\mathbf{h}\|=\|\mathbf{w}\_{i}\|. Since ‖𝐰i‖=∑j|wi​j|normsubscript𝐰𝑖subscript𝑗subscript𝑤𝑖𝑗\|\mathbf{w}\_{i}\|=\sqrt{\sum\_{j}|w\_{ij}|} (with j=1,…,J𝑗

1…𝐽j=1,...,J) and |wi​j|≤‖θ‖∞subscript𝑤𝑖𝑗subscriptnorm𝜃|w\_{ij}|\leq\|\theta\|\_{\infty}, we find that ‖𝐰i‖≤J​‖θ‖∞normsubscript𝐰𝑖𝐽subscriptnorm𝜃\|\mathbf{w}\_{i}\|\leq\sqrt{J}\|\theta\|\_{\infty}. Consequently, ∑i|𝐰i⋅𝐡|≤I​J​‖θ‖∞subscript𝑖⋅subscript𝐰𝑖𝐡𝐼𝐽subscriptnorm𝜃\sqrt{\sum\_{i}|\mathbf{w}\_{i}\cdot\mathbf{h}|}\leq\sqrt{IJ}\|\theta\|\_{\infty} and we obtain:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖W‖≤I​J​‖θ‖∞norm𝑊𝐼𝐽subscriptnorm𝜃\|W\|\leq\sqrt{IJ}\|\theta\|\_{\infty} |  | (48) |

Now for W=W(k)𝑊superscript𝑊𝑘W=W^{(k)}, we have I=dk−1𝐼subscript𝑑𝑘1I=d\_{k-1} and J=dk𝐽subscript𝑑𝑘J=d\_{k}. In the product over k𝑘k, every dksubscript𝑑𝑘d\_{k} except the first and the last occur in pairs, which cancels the square root. For k=1𝑘1k=1, dk−1=dsubscript𝑑𝑘1𝑑d\_{k-1}=d (for the d𝑑d input neurons) and for k=L+1𝑘𝐿1k=L+1, dk=1subscript𝑑𝑘1d\_{k}=1 (for a single output neuron). The final inequality now follows.
∎

### C.7 The Fourier Transform of a Function Composition

Consider Equation [14](#S4.E14 "In 4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks"). The general idea is to investigate the behaviour of Pγ​(𝐥,𝐤)subscript𝑃𝛾𝐥𝐤P\_{\gamma}(\mathbf{l},\mathbf{k}) for large frequencies 𝐥𝐥\mathbf{l} on manifold but smaller frequencies 𝐤𝐤\mathbf{k} in the input domain. In particular, we are interested in the regime where the stationary phase approximation is applicable to Pγsubscript𝑃𝛾P\_{\gamma}, i.e. when l2+k2→∞→superscript𝑙2superscript𝑘2l^{2}+k^{2}\rightarrow\infty (cf. section 3.2. of ([Bergner et al.,](#bib.bib5) )). In this regime, the integrand in Pγ​(𝐤,𝐥)subscript𝑃𝛾𝐤𝐥P\_{\gamma}(\mathbf{k},\mathbf{l}) oscillates fast enough such that the only constructive contribution originates from where the phase term u​(𝐳)=𝐤⋅γ​(𝐳)−𝐥⋅𝐳𝑢𝐳⋅𝐤𝛾𝐳⋅𝐥𝐳u(\mathbf{z})=\mathbf{k}\cdot\gamma(\mathbf{z})-\mathbf{l}\cdot\mathbf{z} does not change with changing 𝐳𝐳\mathbf{z}. This yields the condition that ∇𝐳u​(𝐳)=0subscript∇𝐳𝑢𝐳0\nabla\_{\mathbf{z}}u(\mathbf{z})=0, which translates to the condition (with Einstein summation convention implied and ∂ν=∂/∂xνsubscript𝜈subscript𝑥𝜈\partial\_{\nu}=\nicefrac{{\partial}}{{\partial x\_{\nu}}}):

|  |  |  |  |
| --- | --- | --- | --- |
|  | lν=kμ​∂νγμ​(𝐳)subscript𝑙𝜈subscript𝑘𝜇subscript𝜈subscript𝛾𝜇𝐳l\_{\nu}=k\_{\mu}\partial\_{\nu}\gamma\_{\mu}(\mathbf{z}) |  | (49) |

Now, we impose periodic boundary conditions202020This is possible whenever γ𝛾\gamma is defined on a bounded domain, e.g. on [0,1]msuperscript01𝑚[0,1]^{m}. on the components of γ𝛾\gamma, and without loss of generality we let the period be 2​π2𝜋2\pi. Further, we require that the manifold be contained in a box212121This is equivalent to assuming that the data lies in a bounded set. of some size in ℝdsuperscriptℝ𝑑\mathbb{R}^{d}. The μ𝜇\mu-th component γμsubscript𝛾𝜇\gamma\_{\mu} can now be expressed as a Fourier series:

|  |  |  |  |
| --- | --- | --- | --- |
|  | γμ​(𝐳)subscript𝛾𝜇𝐳\displaystyle\gamma\_{\mu}(\mathbf{z}) | =∑𝐩∈ℤmγ~μ​[𝐩]​e−i​pρ​zρabsentsubscript𝐩superscriptℤ𝑚subscript~𝛾𝜇delimited-[]𝐩superscript𝑒𝑖subscript𝑝𝜌subscript𝑧𝜌\displaystyle=\sum\_{\mathbf{p}\in\mathbb{Z}^{m}}\tilde{\gamma}\_{\mu}[\mathbf{p}]e^{-ip\_{\rho}z\_{\rho}} |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂νγμ​(𝐳)subscript𝜈subscript𝛾𝜇𝐳\displaystyle\partial\_{\nu}\gamma\_{\mu}(\mathbf{z}) | =∑𝐩∈ℤm−i​pν​γ~μ​[𝐩]​e−i​pρ​zρabsentsubscript𝐩superscriptℤ𝑚𝑖subscript𝑝𝜈subscript~𝛾𝜇delimited-[]𝐩superscript𝑒𝑖subscript𝑝𝜌subscript𝑧𝜌\displaystyle=\sum\_{\mathbf{p}\in\mathbb{Z}^{m}}-ip\_{\nu}\tilde{\gamma}\_{\mu}[\mathbf{p}]e^{-ip\_{\rho}z\_{\rho}} |  | (50) |

Equation [C.7](#A3.Ex10 "C.7 The Fourier Transform of a Function Composition ‣ Appendix C Fourier Analysis of ReLU Networks ‣ On the Spectral Bias of Neural Networks") can be substituted in equation [49](#A3.E49 "In C.7 The Fourier Transform of a Function Composition ‣ Appendix C Fourier Analysis of ReLU Networks ‣ On the Spectral Bias of Neural Networks") to obtain:

|  |  |  |  |
| --- | --- | --- | --- |
|  | l​l^ν=−i​k​∑𝐩∈ℤmpν​k^μ​γ~μ​[𝐩]​e−i​pρ​zρ𝑙subscript^𝑙𝜈𝑖𝑘subscript𝐩superscriptℤ𝑚subscript𝑝𝜈subscript^𝑘𝜇subscript~𝛾𝜇delimited-[]𝐩superscript𝑒𝑖subscript𝑝𝜌subscript𝑧𝜌l\hat{l}\_{\nu}=-ik\sum\_{\mathbf{p}\in\mathbb{Z}^{m}}p\_{\nu}\hat{k}\_{\mu}\tilde{\gamma}\_{\mu}[\mathbf{p}]e^{-ip\_{\rho}z\_{\rho}} |  | (51) |

where we have split kμsubscript𝑘𝜇k\_{\mu} and lνsubscript𝑙𝜈l\_{\nu} in to their magnitudes k𝑘k and l𝑙l and directions k^νsubscript^𝑘𝜈\hat{k}\_{\nu} and l^μsubscript^𝑙𝜇\hat{l}\_{\mu} (respectively). We are now interested in the conditions on γ𝛾\gamma under which the RHS can be large in magnitude, even when k𝑘k is fixed. Recall that γ𝛾\gamma is constrained to a box – consequently, we can not arbitrarily scale up γ~μsubscript~𝛾𝜇\tilde{\gamma}\_{\mu}. However, if γ~μ​[𝐩]subscript~𝛾𝜇delimited-[]𝐩\tilde{\gamma}\_{\mu}[\mathbf{p}] decays slowly enough with increasing 𝐩𝐩\mathbf{p}, the RHS can be made arbitrarily large (for certain conditions on 𝐳𝐳\mathbf{z}, l^μsubscript^𝑙𝜇\hat{l}\_{\mu} and k^νsubscript^𝑘𝜈\hat{k}\_{\nu}).

## Appendix D Volume of *High-Frequency Parameters* in Parameter Space

For a given neural network, we now show that the volume of the parameter space containing parameters that contribute ϵitalic-ϵ\epsilon-non-negligibly to frequency components of magnitude k′superscript𝑘′k^{\prime} above a certain cut-off k𝑘k decays with increasing k𝑘k. For notational simplicity and without loss of generality, we absorb the direction 𝐤^^𝐤\hat{\mathbf{k}} of 𝐤𝐤\mathbf{k} in the respective mappings and only deal with the magnitude k𝑘k.

###### Definition 1.

Given a ReLU network fθsubscript𝑓𝜃f\_{\theta} of fixed depth, width and weight clip K𝐾K with parameter vector θ𝜃\theta, an ϵ>0italic-ϵ0\epsilon>0 and Θ=BK∞​(0)Θsubscriptsuperscript𝐵𝐾0\Theta=B^{\infty}\_{K}(0) a L∞superscript𝐿L^{\infty} ball around 00, we define:

|  |  |  |
| --- | --- | --- |
|  | Ξϵ​(k)={θ∈Θ|∃k′>k,|f~θ​(k′)|>ϵ}subscriptΞitalic-ϵ𝑘conditional-set𝜃Θformulae-sequencesuperscript𝑘′𝑘subscript~𝑓𝜃superscript𝑘′italic-ϵ\Xi\_{\epsilon}(k)=\{\theta\in\Theta|\exists k^{\prime}>k,|\tilde{f}\_{\theta}(k^{\prime})|>\epsilon\} |  |

as the set of all parameters vectors θ∈Ξϵ​(k)𝜃subscriptΞitalic-ϵ𝑘\theta\in\Xi\_{\epsilon}(k) that contribute more than an ϵitalic-ϵ\epsilon in expressing one or more frequencies k′superscript𝑘′k^{\prime} above a cut-off frequency k𝑘k.

###### Remark 1.

If k2≥k1subscript𝑘2subscript𝑘1k\_{2}\geq k\_{1}, we have Ξϵ​(k2)⊆Ξϵ​(k1)subscriptΞitalic-ϵsubscript𝑘2subscriptΞitalic-ϵsubscript𝑘1\Xi\_{\epsilon}(k\_{2})\subseteq\Xi\_{\epsilon}(k\_{1}) and consequently vol​(Ξϵ​(k2))≤vol​(Ξϵ​(k1))volsubscriptΞitalic-ϵsubscript𝑘2volsubscriptΞitalic-ϵsubscript𝑘1\text{vol}(\Xi\_{\epsilon}(k\_{2}))\leq\text{vol}(\Xi\_{\epsilon}(k\_{1})), where vol is the Lebesgue measure.

###### Lemma 4.

Let 1kϵ​(θ)superscriptsubscript1𝑘italic-ϵ𝜃1\_{k}^{\epsilon}(\theta) be the indicator function on Ξϵ​(k)subscriptΞitalic-ϵ𝑘\Xi\_{\epsilon}(k). Then:

|  |  |  |
| --- | --- | --- |
|  | ∃κ>0:∀k≥κ,1kϵ​(θ)=0:𝜅0formulae-sequencefor-all𝑘𝜅superscriptsubscript1𝑘italic-ϵ𝜃0\exists\,\kappa>0:\forall k\geq\kappa,1\_{k}^{\epsilon}(\theta)=0 |  |

###### Proof.

From theorem [1](#Thmtheorem1 "Theorem 1. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks"), we know that222222Note from Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks") that ΔΔ\Delta implicitly depends only on the unit vector 𝐤^^𝐤\hat{\mathbf{k}}. |f~θ​(k)|=𝒪​(k−Δ−1)subscript~𝑓𝜃𝑘𝒪superscript𝑘Δ1|\tilde{f}\_{\theta}(k)|=\mathcal{O}(k^{-\Delta-1}) for an integer 1≤Δ≤d1Δ𝑑1\leq\Delta\leq d. In the worse case where Δ=1Δ1\Delta=1, we have that ∃M<∞:|f~θ​(k)|<Mk2:𝑀subscript~𝑓𝜃𝑘𝑀superscript𝑘2\exists M<\infty:|\tilde{f}\_{\theta}(k)|<\frac{M}{k^{2}}. Now, simply select a κ>Mϵ𝜅𝑀italic-ϵ\kappa>\sqrt{\frac{M}{\epsilon}} such that Mκ2<ϵ𝑀superscript𝜅2italic-ϵ\frac{M}{\kappa^{2}}<\epsilon. This yields that |f~θ​(κ)|<Mκ2<ϵsubscript~𝑓𝜃𝜅𝑀superscript𝜅2italic-ϵ|\tilde{f}\_{\theta}(\kappa)|<\frac{M}{\kappa^{2}}<\epsilon, and given that Mκ2≤Mk2​∀k≥κ𝑀superscript𝜅2𝑀superscript𝑘2for-all𝑘𝜅\frac{M}{\kappa^{2}}\leq\frac{M}{k^{2}}\,\forall\,k\geq\kappa, we find |f~θ​(k)|<ϵ​∀k≥κsubscript~𝑓𝜃𝑘italic-ϵfor-all𝑘𝜅|\tilde{f}\_{\theta}(k)|<\epsilon\,\forall\,k\geq\kappa. Now by definition [1](#Thmdefinition1 "Definition 1. ‣ Appendix D Volume of High-Frequency Parameters in Parameter Space ‣ On the Spectral Bias of Neural Networks"), θ∉Ξϵ​(κ)𝜃subscriptΞitalic-ϵ𝜅\theta\not\in\Xi\_{\epsilon}(\kappa), and since Ξϵ​(k)⊆Ξϵ​(κ)subscriptΞitalic-ϵ𝑘subscriptΞitalic-ϵ𝜅\Xi\_{\epsilon}(k)\subseteq\Xi\_{\epsilon}(\kappa) (see remark [1](#Thmremark1 "Remark 1. ‣ Appendix D Volume of High-Frequency Parameters in Parameter Space ‣ On the Spectral Bias of Neural Networks")), we have θ∉Ξϵ​(k)𝜃subscriptΞitalic-ϵ𝑘\theta\not\in\Xi\_{\epsilon}(k), implying 1kϵ​(θ)=0​∀k≥κsuperscriptsubscript1𝑘italic-ϵ𝜃0for-all𝑘𝜅1\_{k}^{\epsilon}(\theta)=0\,\forall\,k\geq\kappa.
∎

###### Remark 2.

We have 1kϵ​(θ)≤|f~θ​(k)|superscriptsubscript1𝑘italic-ϵ𝜃subscript~𝑓𝜃𝑘1\_{k}^{\epsilon}(\theta)\leq|\tilde{f}\_{\theta}(k)| for large enough k𝑘k (i.e. for k≥κ𝑘𝜅k\geq\kappa), since |f~θ​(k)|≥0subscript~𝑓𝜃𝑘0|\tilde{f}\_{\theta}(k)|\geq 0.

###### Proposition 1.

The relative volume of Ξϵ​(k)subscriptΞitalic-ϵ𝑘\Xi\_{\epsilon}(k) w.r.t. ΘΘ\Theta is 𝒪​(k−Δ−1)𝒪superscript𝑘Δ1\mathcal{O}(k^{-\Delta-1}) where 1≤Δ≤d1Δ𝑑1\leq\Delta\leq d.

###### Proof.

The volume is given by the integral over the indicator function, i.e.

|  |  |  |
| --- | --- | --- |
|  | vol​(Ξϵ​(k))=∫θ∈Θ1kϵ​(θ)​𝑑θvolsubscriptΞitalic-ϵ𝑘subscript𝜃Θsuperscriptsubscript1𝑘italic-ϵ𝜃differential-d𝜃\text{vol}(\Xi\_{\epsilon}(k))=\int\_{\theta\in\Theta}1\_{k}^{\epsilon}(\theta)d\theta |  |

For a large enough k𝑘k, we have from remark [2](#Thmremark2 "Remark 2. ‣ Appendix D Volume of High-Frequency Parameters in Parameter Space ‣ On the Spectral Bias of Neural Networks"), the monotonicity of the Lebesgue integral and theorem [1](#Thmtheorem1 "Theorem 1. ‣ 2.2 Fourier Spectrum ‣ 2 Fourier analysis of ReLU networks ‣ On the Spectral Bias of Neural Networks") that:

|  |  |  |  |
| --- | --- | --- | --- |
|  | vol​(Ξϵ​(k))volsubscriptΞitalic-ϵ𝑘\displaystyle\text{vol}(\Xi\_{\epsilon}(k)) | =∫θ∈Θ1kϵ​(θ)​𝑑θabsentsubscript𝜃Θsuperscriptsubscript1𝑘italic-ϵ𝜃differential-d𝜃\displaystyle=\int\_{\theta\in\Theta}1\_{k}^{\epsilon}(\theta)d\theta |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤∫θ∈Θ|f~θ​(k)|​𝑑θ=𝒪​(k−Δ−1)​vol​(Θ)absentsubscript𝜃Θsubscript~𝑓𝜃𝑘differential-d𝜃𝒪superscript𝑘Δ1volΘ\displaystyle\leq\int\_{\theta\in\Theta}|\tilde{f}\_{\theta}(k)|d\theta=\mathcal{O}(k^{-\Delta-1})\text{vol}(\Theta) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ⟹vol​(Ξϵ​(k))vol​(Θ)=𝒪​(k−Δ−1)absentvolsubscriptΞitalic-ϵ𝑘volΘ𝒪superscript𝑘Δ1\displaystyle\implies\frac{\text{vol}(\Xi\_{\epsilon}(k))}{\text{vol}(\Theta)}=\mathcal{O}(k^{-\Delta-1}) |  |

∎

## Appendix E Kernel Machines and KNNs

In this section, in light of our findings, we want to compare DNNs with K-nearest neighbor (k-NN) classifier and kernel machines which are also popular learning algorithms, but are, in contrast to DNNs, better understood theoretically.

### E.1 Kernel Machines vs DNNs

Given that we study why DNNs are biased towards learning smooth functions, we note that kernel machines (KM) are also highly Lipschitz smooth (Eg. for Gaussian kernels all derivatives are bounded). However there are crutial differences between the two. While kernel machines can approximate any target function in principal (Hammer & Gersmann, [2003](#bib.bib16)), the number of Gaussian kernels needed scales linearly with the number of sign changes in the target function (Bengio et al., [2009](#bib.bib4)). Ma & Belkin ([2017](#bib.bib22)) have further shown that for smooth kernels, a target function cannot be approximated within ϵitalic-ϵ\epsilon precision in any polynomial of 1/ϵ1italic-ϵ1/\epsilon steps by gradient descent.

Deep networks on the other hand are also capable of approximating any target function (as shown by the universal approximation theorems Hornik et al. ([1989](#bib.bib17)); Cybenko ([1989](#bib.bib8))), but they are also parameter efficient in contrast to KM. For instance, we have seen that deep ReLU networks separate the input space into number of linear regions that grow polynomially in width of layers and exponentially in the depth of the network (Montufar et al., [2014](#bib.bib24); Raghu et al., [2016](#bib.bib30)). A similar result on the exponentially growing expressive power of networks in terms of their depth is also shown in (Poole et al., [2016](#bib.bib29)). In this paper we have further shown that DNNs are inherently biased towards lower frequency (smooth) functions over a finite parameter space. This suggests that DNNs strike a good balance between function smoothness and expressibility/parameter-efficiency compared with KM.

### E.2 K-NN Classifier vs. DNN classifier

K𝐾K-nearest neighbor (K𝐾KNN) also has a historical importance as a classification algorithm due to its simplicity. It has been shown to be a consistent approximator (Devroye et al., [1996](#bib.bib9)), i.e., asymptotically its empirical risk goes to zero as K→∞→𝐾K\rightarrow\infty and K/N→0→𝐾𝑁0K/N\rightarrow 0, where N𝑁N is the number of training samples. However, because it is a memory based algorithm, it is prohibitively slow for large datasets. Since the smoothness of a K𝐾KNN prediction function is not well studied, we compare the smoothness between K𝐾KNN and DNN. For various values of K𝐾K, we train a K𝐾KNN classifier on a k=150𝑘150k=150 frequency signal (which is binarized) defined on the L=20𝐿20L=20 manifold (see section [4](#S4 "4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks")), and extract probability predictions on a box interval in ℝ2superscriptℝ2\mathbb{R}^{2}. On this interval, we evaluate the 2D FFT and integrate out the angular components (where the angle is parameterized by φ𝜑\varphi) to obtain ζ​(k)𝜁𝑘\zeta(k):

|  |  |  |  |
| --- | --- | --- | --- |
|  | ζ​(k)=dd​k​∫0k𝑑k′​k′​∫02​π𝑑φ​|f~​(k′,φ)|𝜁𝑘𝑑𝑑𝑘superscriptsubscript0𝑘differential-dsuperscript𝑘′superscript𝑘′superscriptsubscript02𝜋differential-d𝜑~𝑓superscript𝑘′𝜑\zeta(k)=\frac{d}{dk}\int\_{0}^{k}dk^{\prime}k^{\prime}\int\_{0}^{2\pi}d\varphi|\tilde{f}(k^{\prime},\varphi)| |  | (52) |

Finally, we plot ζ​(k)𝜁𝑘\zeta(k) for various K𝐾K in figure [22(e)](#A5.F22.sf5 "In Figure 22 ‣ E.2 K-NN Classifier vs. DNN classifier ‣ Appendix E Kernel Machines and KNNs ‣ On the Spectral Bias of Neural Networks"). Furthermore, we train a DNN on the very same dataset and overlay the radial spectrum of the resulting probability map on the same plot. We find that while DNN’s are as expressive as a K=1𝐾1K=1 KNN classifier at lower (radial) frequencies, the frequency spectrum of DNNs decay faster than KNN classifier for all values of K𝐾K considered, indicating that the DNN is smoother than the K𝐾KNNs considered. We also repeat the experiment corresponding to Fig. [9](#S4.F9 "Figure 9 ‣ 4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks") with KNNs (see Fig. [22(d)](#A5.F22.sf4 "In Figure 22 ‣ E.2 K-NN Classifier vs. DNN classifier ‣ Appendix E Kernel Machines and KNNs ‣ On the Spectral Bias of Neural Networks")) for various K𝐾K’s, to find that unlike DNNs, KNNs do not necessarily perform better for larger L𝐿L’s, suggesting that KNNs do not exploit the geometry of the manifold like DNNs do.

![Refer to caption](/html/1806.08734/assets/figures/lvk_k5.png)


(a) K=5𝐾5K=5.

![Refer to caption](/html/1806.08734/assets/figures/lvk_k10.png)


(b) K=10𝐾10K=10.

![Refer to caption](/html/1806.08734/assets/figures/lvk_k15.png)


(c) K=15𝐾15K=15.

![Refer to caption](/html/1806.08734/assets/figures/lvk_k20.png)


(d) K=20𝐾20K=20.

![Refer to caption](/html/1806.08734/assets/x7.png)


(e) Frequency spectrum

Figure 22: (a,b,c,d): Heatmaps of training accuracies (L𝐿L-vs-k𝑘k) of KNNs for various K𝐾K. When comparing with figure [9](#S4.F9 "Figure 9 ‣ 4 Not all Manifolds are Learned Equal ‣ On the Spectral Bias of Neural Networks"), note that the y-axis is flipped. (e): The frequency spectrum of K𝐾KNNs with different values of K𝐾K, and a DNN. The DNN learns a smoother function compared with the K𝐾KNNs considered since the spectrum of the DNN decays faster compared with K𝐾KNNs.

[◄](/html/1806.08733)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/1806.08734)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+1806.08734)
[View original  
on arXiv](https://arxiv.org/abs/1806.08734)[►](/html/1806.08735)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Fri Mar 1 07:25:49 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
