---
arxiv: '1706.02515'
authors:
- Günter Klambauer
- Thomas Unterthiner
- Andreas Mayr
- Sepp Hochreiter
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: Self-Normalizing Neural Networks
url: http://arxiv.org/abs/1706.02515v5
year: 2017
---

[1706.02515] Self-Normalizing Neural Networks















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



# Self-Normalizing Neural Networks

Günter Klambauer
&Thomas Unterthiner
&Andreas Mayr
&Sepp Hochreiter
  
LIT AI Lab & Institute of Bioinformatics,
  
Johannes Kepler University Linz
  
A-4040 Linz, Austria
  
{klambauer,unterthiner,mayr,hochreit}@bioinf.jku.at

###### Abstract

Deep Learning has revolutionized vision via convolutional neural networks (CNNs)
and natural language processing via recurrent neural networks (RNNs).
However, success stories of Deep Learning with
standard feed-forward neural networks (FNNs) are rare.
FNNs that perform well are typically shallow
and, therefore cannot exploit many levels of abstract representations.
We introduce self-normalizing neural networks (SNNs) to
enable high-level abstract representations.
While batch normalization requires explicit normalization,
neuron activations of SNNs
automatically converge towards zero mean and unit variance.
The activation function of SNNs are “scaled exponential linear units”
(SELUs), which induce self-normalizing properties.
Using the Banach fixed-point theorem,
we prove that activations close to zero mean and unit variance
that are propagated through many network layers will converge
towards zero mean and unit variance —
even under the presence of noise and perturbations.
This convergence property of SNNs allows to (1) train deep networks with many layers, (2) employ strong regularization schemes, and (3) to make learning highly robust.
Furthermore, for activations not close to unit
variance, we prove an upper and lower bound
on the variance, thus, vanishing and exploding gradients are impossible.
We compared SNNs on (a) 121 tasks from the UCI machine learning repository,
on (b) drug discovery benchmarks, and on (c) astronomy tasks
with standard FNNs, and other machine learning methods such as random forests and support vector machines.
For FNNs we considered
(i) ReLU networks without normalization, (ii) batch normalization, (iii) layer normalization, (iv) weight normalization,
(v) highway networks, and (vi) residual networks.
SNNs significantly outperformed all competing FNN methods at
121 UCI tasks, outperformed all competing methods
at the Tox21 dataset, and set a new record at an astronomy data set.
The winning SNN architectures are often very deep. Implementations are available at: [github.com/bioinf-jku/SNNs](https://www.github.com/bioinf-jku/SNNs).

Accepted for publication at NIPS 2017; please cite as:
  
Klambauer, G., Unterthiner, T., Mayr, A., & Hochreiter, S. (2017). Self-Normalizing Neural Networks. In Advances in Neural Information Processing Systems (NIPS).

## Introduction

Deep Learning has set new records at different benchmarks and
led to various commercial applications [[25](#bib.bib25), [33](#bib.bib33)].
Recurrent neural networks (RNNs) [[18](#bib.bib18)]
achieved new levels at speech and natural language processing,
for example at the TIMIT benchmark [[12](#bib.bib12)] or at
language translation [[36](#bib.bib36)], and
are already employed in mobile devices [[31](#bib.bib31)].
RNNs have won handwriting recognition challenges (Chinese and Arabic
handwriting) [[33](#bib.bib33), [13](#bib.bib13), [6](#bib.bib6)]
and Kaggle challenges,
such as the “Grasp-and Lift EEG” competition.
Their counterparts, convolutional neural networks (CNNs) [[24](#bib.bib24)] excel
at vision and video tasks.
CNNs are on par with human dermatologists at the
visual detection of skin cancer [[9](#bib.bib9)].
The visual processing for self-driving cars is based on CNNs [[19](#bib.bib19)],
as is the visual input to AlphaGo which has beaten
one of the best human GO players [[34](#bib.bib34)].
At vision challenges, CNNs are constantly winning, for example at
the large ImageNet competition [[23](#bib.bib23), [16](#bib.bib16)], but
also almost all Kaggle vision challenges, such as the “Diabetic Retinopathy” and
the “Right Whale” challenges [[8](#bib.bib8), [14](#bib.bib14)].

However, looking at Kaggle challenges that are not related to vision or sequential
tasks, gradient boosting, random forests, or support vector machines (SVMs) are winning most of the competitions.
Deep Learning is notably absent, and for the few cases where FNNs won,
they are shallow. For example, the HIGGS challenge,
the Merck Molecular Activity challenge, and the
Tox21 Data challenge were all won by FNNs with at most four hidden layers.
Surprisingly, it is hard to find success stories with FNNs that
have many hidden layers, though they would allow for different levels
of abstract representations
of the input [[3](#bib.bib3)].

To robustly train very deep CNNs, batch normalization evolved into a standard to normalize
neuron activations to zero mean and unit variance [[20](#bib.bib20)].
Layer normalization [[2](#bib.bib2)] also ensures zero mean and unit
variance, while weight normalization [[32](#bib.bib32)] ensures
zero mean and unit variance if in the previous layer the activations have
zero mean and unit variance.
However, training with normalization techniques is perturbed by
stochastic gradient descent (SGD), stochastic
regularization (like dropout),
and the estimation of the normalization parameters.
Both RNNs and CNNs can stabilize learning via weight sharing,
therefore they are less prone to these perturbations.
In contrast, FNNs trained with normalization techniques suffer from
these perturbations and have high variance
in the training error (see Figure [1](#Sx2.F1 "Figure 1 ‣ Normalization and SNNs. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")).
This high variance hinders learning and slows it down.
Furthermore, strong regularization, such as dropout,
is not possible as it would further
increase the variance which in turn would lead to divergence of the
learning process.
We believe that this sensitivity to perturbations
is the reason that FNNs are less
successful than RNNs and CNNs.

Self-normalizing neural networks (SNNs) are robust to perturbations
and do not have high variance in their training errors (see Figure [1](#Sx2.F1 "Figure 1 ‣ Normalization and SNNs. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")).
SNNs push neuron activations to zero mean and unit variance
thereby leading to the same effect as batch normalization,
which enables to robustly learn many layers.
SNNs are based on scaled exponential linear units “SELUs”
which induce self-normalizing properties like variance stabilization
which in turn avoids exploding and vanishing gradients.

## Self-normalizing Neural Networks (SNNs)

##### Normalization and SNNs.

For a neural network with activation function f𝑓f,
we consider two consecutive layers
that are connected by a weight matrix 𝑾𝑾\bm{W}.
Since the input to a neural
network is a random variable,
the activations 𝒙𝒙\bm{x} in the lower
layer, the network inputs 𝒛=𝑾​𝒙𝒛𝑾𝒙\bm{z}=\bm{W}\bm{x}, and the
activations 𝒚=f​(𝒛)𝒚𝑓𝒛\bm{y}=f(\bm{z}) in the higher layer are
random variables as well.
We assume that all activations xisubscript𝑥𝑖x\_{i} of the lower layer
have mean
μ:=E​(xi)assign𝜇Esubscript𝑥𝑖\mu:=\mathbf{\mathrm{E}}(x\_{i}) and variance ν:=Var​(xi)assign𝜈Varsubscript𝑥𝑖\nu:=\mathbf{\mathrm{Var}}(x\_{i}).
An activation y𝑦y in the
higher layer has mean
μ~:=E​(y)assign~𝜇E𝑦{\tilde{\mu}}:=\mathbf{\mathrm{E}}(y) and variance
ν~:=Var​(y)assign~𝜈Var𝑦{\tilde{\nu}}:=\mathbf{\mathrm{Var}}(y).
Here E(.)\mathbf{\mathrm{E}}(.) denotes the expectation and
Var(.)\mathbf{\mathrm{Var}}(.) the variance of a random variable.
A single activation y=f​(z)𝑦𝑓𝑧y=f(z) has net input
z=𝒘T​𝒙𝑧superscript𝒘𝑇𝒙z=\bm{w}^{T}\bm{x}.
For n𝑛n units with activation
xi,1⩽i⩽n

subscript𝑥𝑖1
𝑖𝑛x\_{i},1\leqslant i\leqslant n in the lower layer, we define n𝑛n
times the mean of the
weight vector 𝒘∈ℝn𝒘superscriptℝ𝑛\bm{w}\in\mathbb{R}^{n} as ω:=∑i=1nwiassign𝜔superscriptsubscript𝑖1𝑛subscript𝑤𝑖\omega:=\sum\_{i=1}^{n}w\_{i} and n𝑛n
times the second moment as τ:=∑i=1nwi2assign𝜏superscriptsubscript𝑖1𝑛superscriptsubscript𝑤𝑖2\tau:=\sum\_{i=1}^{n}w\_{i}^{2}.

We consider the mapping g𝑔g that maps mean and variance of the activations from one layer
to mean and variance of the activations in the next layer

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | (μν)matrix𝜇𝜈\displaystyle\begin{pmatrix}\mu\\ \nu\end{pmatrix}\ | ↦(μ~ν~):(μ~ν~)=g(μν).\displaystyle\mapsto\ \begin{pmatrix}{\tilde{\mu}}\\ {\tilde{\nu}}\end{pmatrix}\ :\quad\begin{pmatrix}{\tilde{\mu}}\\ {\tilde{\nu}}\end{pmatrix}\ =\ g\begin{pmatrix}\mu\\ \nu\end{pmatrix}\ . |  | (1) |

Normalization techniques like batch, layer, or weight normalization
ensure a mapping g𝑔g that keeps
(μ,ν)𝜇𝜈(\mu,\nu) and (μ~,ν~)~𝜇~𝜈({\tilde{\mu}},{\tilde{\nu}})
close to predefined values, typically (0,1)01(0,1).

###### Definition 1 (Self-normalizing neural net).

A neural network is self-normalizing if it possesses a mapping
g:Ω↦Ω:𝑔maps-toΩΩg:\Omega\mapsto\Omega for each activation y𝑦y that maps mean and variance from one layer to the next
and has a stable and attracting fixed point depending on (ω,τ)𝜔𝜏(\omega,\tau) in ΩΩ\Omega.
Furthermore, the mean and the variance remain in the domain ΩΩ\Omega, that is g​(Ω)⊆Ω𝑔ΩΩg(\Omega)\subseteq\Omega, where
Ω={(μ,ν)|μ∈[μmin,μmax],ν∈[νmin,νmax]}Ωconditional-set𝜇𝜈formulae-sequence𝜇subscript𝜇subscript𝜇𝜈subscript𝜈subscript𝜈\Omega=\{(\mu,\nu)\ |\ \mu\in[\mu\_{\min},\mu\_{\max}],\nu\in[\nu\_{\min},\nu\_{\max}]\}.
When iteratively applying the mapping g𝑔g, each point within ΩΩ\Omega converges to this fixed point.

Therefore, we consider activations of a neural network to be normalized,
if both their mean and their variance across samples are within predefined intervals.
If mean and variance of 𝒙𝒙\bm{x} are already within these intervals, then also
mean and variance of 𝒚𝒚\bm{y} remain in these intervals, i.e., the
normalization is transitive across layers. Within these intervals,
the mean and variance both converge to a fixed point if the mapping g𝑔g is applied
iteratively.

![Refer to caption](/html/1706.02515/assets/x1.png)

![Refer to caption](/html/1706.02515/assets/x2.png)

Figure 1: The left panel and the right panel show the training error (y-axis) for feed-forward neural networks (FNNs) with batch
normalization (BatchNorm) and self-normalizing networks (SNN) across update steps (x-axis)
on the MNIST dataset the CIFAR10 dataset, respectively.
We tested networks with 8, 16, and 32 layers and learning rate 1e-5. FNNs
with batch normalization exhibit high variance due to perturbations.
In contrast, SNNs do not suffer from high variance as they are
more robust to perturbations and learn faster.

Therefore, SNNs
keep normalization of activations when propagating them
through layers of the network.
The normalization effect is observed across layers of a network:
in each layer the activations are getting closer to the fixed point.
The normalization effect can also observed be for two fixed layers across learning steps: perturbations of lower layer activations or
weights are damped in the higher layer by drawing the activations
towards the fixed point.
If for all y𝑦y in the higher layer, ω𝜔\omega
and τ𝜏\tau of the corresponding weight vector are the same, then
the fixed points are also the same. In this case we have a
unique fixed point for all activations y𝑦y.
Otherwise, in the more general case, ω𝜔\omega
and τ𝜏\tau differ for different y𝑦y but the mean activations are
drawn into [μmin,μmax]subscript𝜇subscript𝜇[\mu\_{\min},\mu\_{\max}] and the variances
are drawn into [νmin,νmax]subscript𝜈subscript𝜈[\nu\_{\min},\nu\_{\max}].

##### Constructing Self-Normalizing Neural Networks.

We aim at constructing self-normalizing
neural networks by adjusting the properties of the function g𝑔g.
Only two design choices are available for the
function g𝑔g:
(1) the activation function and
(2) the initialization of the weights.

For the activation function,
we propose “scaled exponential linear units” (SELUs) to render a FNN
as self-normalizing. The SELU activation function is given by

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | selu⁡(x)selu𝑥\displaystyle\operatorname{selu}(x)\ | =λ​{xif ​x>0α​ex−αif ​x⩽0.absent𝜆cases𝑥if 𝑥0𝛼superscript𝑒𝑥𝛼if 𝑥0\displaystyle=\ \lambda\ \begin{cases}x&\text{if }x>0\\ \alpha e^{x}-\alpha&\text{if }x\leqslant 0\end{cases}\ . |  | (2) |

SELUs allow to construct a mapping g𝑔g with properties that lead to SNNs.
SNNs cannot be derived with (scaled)
rectified linear units (ReLUs), sigmoid units, tanh\tanh units, and leaky
ReLUs.
The activation function is required to have
(1) negative and positive values for controlling the mean,
(2) saturation regions (derivatives approaching zero) to dampen the variance if
it is too large in the lower layer,
(3) a slope larger than one to increase the variance if
it is too small in the lower layer,
(4) a continuous curve.
The latter ensures a fixed point, where variance damping is equalized by variance increasing.
We met these properties of the activation function by multiplying the exponential linear
unit (ELU) [[7](#bib.bib7)] with λ>1𝜆1\lambda>1 to ensure a slope larger than one
for positive net inputs.

For the weight initialization, we
propose ω=0𝜔0\omega=0 and τ=1𝜏1\tau=1 for all units in the higher layer.
The next paragraphs will show the advantages of this initialization.
Of course, during learning these assumptions on the weight vector will
be violated. However, we can prove the self-normalizing property
even for weight vectors that are not normalized, therefore, the
self-normalizing property can be kept during learning and weight changes.

##### Deriving the Mean and Variance Mapping Function g𝑔g.

We assume that the xisubscript𝑥𝑖x\_{i} are independent from each other but share
the same mean μ𝜇\mu and variance ν𝜈\nu.
Of course, the independence assumptions is
not fulfilled in general. We will elaborate on the independence
assumption below.
The network input z𝑧z in the higher layer
is z=𝒘T​𝒙𝑧superscript𝒘𝑇𝒙z=\bm{w}^{T}\bm{x} for which we can infer the following moments
E​(z)=∑i=1nwi​E​(xi)=μ​ωE𝑧superscriptsubscript𝑖1𝑛subscript𝑤𝑖Esubscript𝑥𝑖𝜇𝜔\mathbf{\mathrm{E}}(z)\ =\ \sum\_{i=1}^{n}\ w\_{i}\ \mathbf{\mathrm{E}}(x\_{i})\ =\ \mu\ \omega
and
Var​(z)=Var​(∑i=1nwi​xi)=ν​τVar𝑧Varsuperscriptsubscript𝑖1𝑛subscript𝑤𝑖subscript𝑥𝑖𝜈𝜏\mathbf{\mathrm{Var}}(z)\ =\ \mathbf{\mathrm{Var}}(\sum\_{i=1}^{n}w\_{i}\ x\_{i})\ =\ \nu\ \tau,
where we used the independence of the xisubscript𝑥𝑖x\_{i}.
The net input z𝑧z is a weighted sum of independent,
but not necessarily identically distributed variables xisubscript𝑥𝑖x\_{i},
for which the central limit theorem (CLT) states that z𝑧z approaches a normal distribution:
z∼𝒩​(μ​ω,ν​τ)similar-to𝑧𝒩𝜇𝜔𝜈𝜏z\sim\mathcal{N}(\mu\omega,\sqrt{\nu\tau})
with density pN​(z;μ​ω,ν​τ)subscript𝑝N

𝑧𝜇𝜔𝜈𝜏p\_{\mathrm{N}}(z;\mu\omega,\sqrt{\nu\tau}).
According to the CLT, the larger n𝑛n, the closer is z𝑧z to a normal distribution.
For Deep Learning, broad layers with hundreds of neurons xisubscript𝑥𝑖x\_{i} are common.
Therefore the assumption that z𝑧z is normally distributed is met well for most currently used
neural networks (see Figure [A8](#S4.F8 "Figure A8 ‣ Distribution of network inputs. ‣ A4.3 Tox21 challenge data set: Hyperparameters ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")).
The function g𝑔g maps the mean and variance of activations in the lower layer to the mean
μ~=E​(y)~𝜇E𝑦{\tilde{\mu}}=\mathbf{\mathrm{E}}(y) and variance ν~=Var​(y)~𝜈Var𝑦{\tilde{\nu}}=\mathbf{\mathrm{Var}}(y) of the activations y𝑦y in the next layer:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | g:(μν)↦(μ~ν~)::𝑔maps-tomatrix𝜇𝜈matrix~𝜇~𝜈:absent\displaystyle g:\begin{pmatrix}\mu\\ \nu\end{pmatrix}\mapsto\begin{pmatrix}{\tilde{\mu}}\\ {\tilde{\nu}}\end{pmatrix}:\ \ \quad | μ~​(μ,ω,ν,τ)=∫−∞∞selu⁡(z)​pN​(z;μ​ω,ν​τ)​dz~𝜇𝜇𝜔𝜈𝜏superscriptsubscriptselu𝑧subscript𝑝N  𝑧𝜇𝜔𝜈𝜏 differential-d𝑧\displaystyle{\tilde{\mu}}(\mu,\omega,\nu,\tau)\ =\ \int\_{-\infty}^{\infty}\operatorname{selu}(z)\ p\_{{\mathrm{N}}}(z;\mu\omega,\sqrt{\nu\tau})\ \mathrm{d}z |  | (3) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ν~(μ,ω,ν,τ)=∫−∞∞selu(z)2pN(z;μω,ν​τ)dz−(μ~)2.\displaystyle{\tilde{\nu}}(\mu,\omega,\nu,\tau)\ =\ \int\_{-\infty}^{\infty}\operatorname{selu}(z)^{2}\ p\_{{\mathrm{N}}}(z;\mu\omega,\sqrt{\nu\tau})\ \mathrm{d}z\ -\ ({\tilde{\mu}})^{2}\ . |  |

These integrals can be analytically computed and lead to following
mappings of the moments:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | μ~~𝜇\displaystyle{\tilde{\mu}}\ | =12λ((μω)erf(μ​ω2​ν​τ)+\displaystyle=\ \frac{1}{2}\lambda\left((\mu\omega)\operatorname{erf}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+\right. |  | (4) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | αeμ​ω+ν​τ2erfc(μ​ω+ν​τ2​ν​τ)−αerfc(μ​ω2​ν​τ)+2πν​τe−(μ​ω)22​(ν​τ)+μω)\displaystyle\left.\alpha\ e^{\mu\omega+\frac{\nu\tau}{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-\alpha\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+\sqrt{\frac{2}{\pi}}\sqrt{\nu\tau}e^{-\frac{(\mu\omega)^{2}}{2(\nu\tau)}}+\mu\omega\right) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ν~~𝜈\displaystyle{\tilde{\nu}}\ | =12λ2(((μω)2+ντ)(2−erfc(μ​ω2​ν​τ))+α2(−2eμ​ω+ν​τ2erfc(μ​ω+ν​τ2​ν​τ)\displaystyle=\ \frac{1}{2}\lambda^{2}\left(\left((\mu\omega)^{2}+\nu\tau\right)\left(2-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)+\alpha^{2}\left(-2e^{\mu\omega+\frac{\nu\tau}{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\right.\right. |  | (5) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +e2​(μ​ω+ν​τ)erfc(μ​ω+2​ν​τ2​ν​τ)+erfc(μ​ω2​ν​τ))+2π(μω)ν​τe−(μ​ω)22​(ν​τ))−(μ~)2\displaystyle\left.\left.+e^{2(\mu\omega+\nu\tau)}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)+\sqrt{\frac{2}{\pi}}(\mu\omega)\sqrt{\nu\tau}e^{-\frac{(\mu\omega)^{2}}{2(\nu\tau)}}\right)-\left({\tilde{\mu}}\right)^{2} |  |

##### Stable and Attracting Fixed Point (𝟎,𝟏)01\bm{(0,1)} for Normalized Weights.

We assume a normalized weight
vector 𝒘𝒘\bm{w} with ω=0𝜔0\omega=0 and τ=1𝜏1\tau=1.
Given a fixed point (μ,ν)𝜇𝜈(\mu,\nu),
we can solve equations Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) and Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) for α𝛼\alpha and
λ𝜆\lambda.
We chose the fixed point (μ,ν)=(0,1)𝜇𝜈01(\mu,\nu)=(0,1),
which is typical for activation normalization.
We obtain the fixed point equations μ~=μ=0~𝜇𝜇0{\tilde{\mu}}=\mu=0 and ν~=ν=1~𝜈𝜈1{\tilde{\nu}}=\nu=1 that we solve for α𝛼\alpha and λ𝜆\lambda and obtain the solutions α01≈ 1.6733subscript𝛼011.6733\alpha\_{\mathrm{01}}\approx\ 1.6733 and λ01≈ 1.0507subscript𝜆011.0507\lambda\_{\mathrm{01}}\approx\ 1.0507,
where the subscript 0101{\mathrm{01}} indicates that these are the parameters for fixed point (0,1)01(0,1).
The analytical expressions for α01subscript𝛼01\alpha\_{\mathrm{01}} and λ01subscript𝜆01\lambda\_{\mathrm{01}} are given in Eq. ([14](#S1.E14 "In A1 Background ‣ Self-Normalizing Neural Networks")). We are interested whether the fixed point (μ,ν)=(0,1)𝜇𝜈01(\mu,\nu)=(0,1) is stable
and attracting. If the Jacobian of g𝑔g has a norm smaller than 1 at the
fixed point, then g𝑔g is a contraction mapping and the fixed point is stable.
The (2x2)-Jacobian 𝒥​(μ,ν)𝒥𝜇𝜈\mathcal{J}(\mu,\nu) of g:(μ,ν)↦(μ~,ν~):𝑔maps-to𝜇𝜈~𝜇~𝜈g:(\mu,\nu)\mapsto({\tilde{\mu}},{\tilde{\nu}}) evaluated at the fixed point (0,1)01(0,1) with α01subscript𝛼01\alpha\_{\mathrm{01}} and
λ01subscript𝜆01\lambda\_{\mathrm{01}} is

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | 𝒥​(μ,ν)𝒥𝜇𝜈\displaystyle\mathcal{J}(\mu,\nu)\ | =(∂μnew​(μ,ν)∂μ∂μnew​(μ,ν)∂ν∂νnew​(μ,ν)∂μ∂νnew​(μ,ν)∂ν),absentmatrixsuperscript𝜇new𝜇𝜈𝜇superscript𝜇new𝜇𝜈𝜈missing-subexpressionmissing-subexpressionsuperscript𝜈new𝜇𝜈𝜇superscript𝜈new𝜇𝜈𝜈\displaystyle=\ \begin{pmatrix}\partial\frac{\mu^{\mathrm{new}}(\mu,\nu)}{\partial\mu}&\partial\frac{\mu^{\mathrm{new}}(\mu,\nu)}{\partial\nu}\\ ~{}&~{}\\ \partial\frac{\nu^{\mathrm{new}}(\mu,\nu)}{\partial\mu}&\partial\frac{\nu^{\mathrm{new}}(\mu,\nu)}{\partial\nu}\end{pmatrix},\ | 𝒥​(0,1)=(0.00.0888340.00.782648).𝒥01matrix0.00.0888340.00.782648\displaystyle\mathcal{J}(0,1)\ =\ \begin{pmatrix}0.0&0.088834\\ 0.0&0.782648\end{pmatrix}\ . |  | (6) |

The spectral norm of 𝒥​(0,1)𝒥01\mathcal{J}(0,1) (its largest
singular value) is 0.7877<10.787710.7877<1. That means g𝑔g is a contraction
mapping around the fixed point (0,1)01(0,1) (the mapping is depicted in Figure [2](#Sx2.F2 "Figure 2 ‣ Stable and Attracting Fixed Points for Unnormalized Weights. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")).
Therefore, (0,1)01(0,1) is a stable fixed point of
the mapping g𝑔g.

##### Stable and Attracting Fixed Points for Unnormalized Weights.

A normalized weight vector 𝒘𝒘\bm{w} cannot be ensured during learning.
For SELU parameters
α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01} and λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01},
we show in the next theorem that
if (ω,τ)𝜔𝜏(\omega,\tau) is close to (0,1)01(0,1), then g𝑔g still has an
attracting and stable fixed point
that is close to (0,1)01(0,1).
Thus, in the general case there still exists a stable fixed point
which, however, depends on (ω,τ)𝜔𝜏(\omega,\tau).
If we restrict (μ,ν,ω,τ)𝜇𝜈𝜔𝜏(\mu,\nu,\omega,\tau) to certain intervals, then we
can show that (μ,ν)𝜇𝜈(\mu,\nu) is mapped to the respective intervals.
Next we present the central theorem of this paper,
from which follows that SELU
networks are self-normalizing under mild conditions on the weights.

###### Theorem 1 (Stable and Attracting Fixed Points).

We assume α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01} and λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01}.
We restrict the range of the variables to the following intervals
μ∈[−0.1,0.1]𝜇0.10.1\mu\in[-0.1,0.1],
ω∈[−0.1,0.1]𝜔0.10.1\omega\in[-0.1,0.1],
ν∈[0.8,1.5]𝜈0.81.5\nu\in[0.8,1.5], and
τ∈[0.95,1.1]𝜏0.951.1\tau\in[0.95,1.1],
that define the functions’ domain ΩΩ\Omega.
For ω=0𝜔0\omega=0 and τ=1𝜏1\tau=1, the mapping Eq. ([3](#Sx2.E3 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))
has the stable
fixed point (μ,ν)=(0,1)𝜇𝜈01(\mu,\nu)=(0,1), whereas for other ω𝜔\omega and τ𝜏\tau the mapping Eq. ([3](#Sx2.E3 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))
has a stable and
attracting fixed point depending on (ω,τ)𝜔𝜏(\omega,\tau) in the
(μ,ν)𝜇𝜈(\mu,\nu)-domain: μ∈[−0.03106,0.06773]𝜇0.031060.06773\mu\in[-0.03106,0.06773] and
ν∈[0.80009,1.48617]𝜈0.800091.48617\nu\in[0.80009,1.48617].
All points within the (μ,ν)𝜇𝜈(\mu,\nu)-domain converge when
iteratively applying the mapping Eq. ([3](#Sx2.E3 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) to this fixed point.

![Refer to caption](/html/1706.02515/assets/x3.png)


Figure 2: For ω=0𝜔0\omega=0 and τ=1𝜏1\tau=1,
the mapping g𝑔g of mean μ𝜇\mu (x𝑥x-axis) and variance ν𝜈\nu (y𝑦y-axis)
to the next layer’s mean μ~~𝜇{\tilde{\mu}} and variance ν~~𝜈{\tilde{\nu}}
is depicted.
Arrows show in which direction (μ,ν)𝜇𝜈(\mu,\nu) is mapped by g:(μ,ν)↦(μ~,ν~):𝑔maps-to𝜇𝜈~𝜇~𝜈g:(\mu,\nu)\mapsto({\tilde{\mu}},{\tilde{\nu}}).
The fixed point of the mapping g𝑔g is (0,1)01(0,1).

###### Proof.

We provide a proof sketch (see detailed proof in Appendix Section [A3](#S3 "A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")).
With the Banach fixed point theorem we show that there exists
a unique attracting and stable fixed point.
To this end, we have to prove that a) g𝑔g is a contraction mapping and b) that
the mapping stays in the domain, that is, g​(Ω)⊆Ω𝑔ΩΩg(\Omega)\subseteq\Omega.
The spectral norm of the Jacobian of g𝑔g can be obtained via
an explicit formula for the largest singular value for a 2×2222\times 2 matrix.
g𝑔g is a contraction mapping if its spectral norm is smaller than 111.
We perform a computer-assisted proof to
evaluate the largest singular value on a fine
grid and ensure the precision of the computer
evaluation by an error propagation analysis of the implemented
algorithms on the according hardware.
Singular values between grid points are upper bounded by the
mean value theorem. To this end, we bound the derivatives
of the formula for the largest singular value with respect to
ω,τ,μ,ν

𝜔𝜏𝜇𝜈\omega,\tau,\mu,\nu.
Then we apply the mean value theorem to pairs of points, where one is on the grid and the
other is off the grid. This shows that for all values of
ω,τ,μ,ν

𝜔𝜏𝜇𝜈\omega,\tau,\mu,\nu in the domain ΩΩ\Omega, the spectral norm of
g𝑔g is smaller than one.
Therefore, g𝑔g is a contraction mapping on the domain ΩΩ\Omega.
Finally, we show that the mapping g𝑔g stays in the domain ΩΩ\Omega by
deriving bounds on μ~~𝜇{\tilde{\mu}} and ν~~𝜈{\tilde{\nu}}.
Hence, the Banach fixed-point theorem holds and there exists a unique
fixed point in ΩΩ\Omega that is attained.
∎

Consequently, feed-forward neural networks with many units in each layer
and with the SELU activation function are self-normalizing (see definition [1](#Thmdefinition1 "Definition 1 (Self-normalizing neural net). ‣ Normalization and SNNs. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")), which
readily follows from Theorem [1](#Thmtheorem1 "Theorem 1 (Stable and Attracting Fixed Points). ‣ Stable and Attracting Fixed Points for Unnormalized Weights. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks").
To give an intuition, the main property of SELUs is that they damp the variance for negative
net inputs and increase the variance for positive net inputs.
The variance damping is stronger if net inputs are further away from zero while
the variance increase is stronger if net inputs are close to zero.
Thus, for large variance of the activations in the lower
layer the damping effect is dominant and the variance decreases in the
higher layer.
Vice versa, for small variance the
variance increase is dominant and the variance increases in the higher layer.

However, we cannot guarantee that mean and variance remain in the domain ΩΩ\Omega.
Therefore, we next treat the case where (μ,ν)𝜇𝜈(\mu,\nu) are outside ΩΩ\Omega.
It is especially crucial to consider ν𝜈\nu because this variable has much stronger
influence than μ𝜇\mu. Mapping ν𝜈\nu across layers to a high value corresponds to an
exploding gradient, since the Jacobian of the activation of high layers with respect to activations
in lower layers has large singular values.
Analogously, mapping ν𝜈\nu across layers to a low value corresponds to an
vanishing gradient. Bounding the mapping of ν𝜈\nu from above and below would avoid
both exploding and vanishing gradients.
Theorem [2](#Thmtheorem2 "Theorem 2 (Decreasing 𝜈). ‣ Stable and Attracting Fixed Points for Unnormalized Weights. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks") states that the variance of neuron activations of SNNs
is bounded from above, and therefore ensures that SNNs learn robustly and do not
suffer from exploding gradients.

###### Theorem 2 (Decreasing ν𝜈\nu).

For λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01}, α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01}
and the domain Ω+superscriptΩ\Omega^{+}:
−1⩽μ⩽11𝜇1-1\leqslant\mu\leqslant 1,
−0.1⩽ω⩽0.10.1𝜔0.1-0.1\leqslant\omega\leqslant 0.1,
3⩽ν⩽163𝜈163\leqslant\nu\leqslant 16, and
0.8⩽τ⩽1.250.8𝜏1.250.8\leqslant\tau\leqslant 1.25,
we have for the mapping of the variance
ν~​(μ,ω,ν,τ,λ,α)~𝜈𝜇𝜔𝜈𝜏𝜆𝛼{\tilde{\nu}}(\mu,\omega,\nu,\tau,\lambda,\alpha) given in Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")):
ν~​(μ,ω,ν,τ,λ01,α01)<ν~𝜈𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01𝜈{\tilde{\nu}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})<\nu.

The proof can be found in the Appendix Section [A3](#S3 "A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").
Thus, when mapped across many layers, the variance in the interval [3,16]316[3,16] is mapped to a value below 333. Consequently, all fixed
points (μ,ν)𝜇𝜈(\mu,\nu) of the mapping g𝑔g (Eq. ([3](#Sx2.E3 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))) have ν<3𝜈3\nu<3.
Analogously, Theorem [3](#Thmtheorem3 "Theorem 3 (Increasing 𝜈). ‣ Stable and Attracting Fixed Points for Unnormalized Weights. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks") states that the variance of neuron activations of SNNs
is bounded from below, and therefore ensures that SNNs do not suffer from vanishing gradients.

###### Theorem 3 (Increasing ν𝜈\nu).

We consider λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01}, α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01}
and the domain Ω−superscriptΩ\Omega^{-}:
−0.1⩽μ⩽0.10.1𝜇0.1-0.1\leqslant\mu\leqslant 0.1, and
−0.1⩽ω⩽0.10.1𝜔0.1-0.1\leqslant\omega\leqslant 0.1.
For the domain
0.02⩽ν⩽0.160.02𝜈0.160.02\leqslant\nu\leqslant 0.16
and 0.8⩽τ⩽1.250.8𝜏1.250.8\leqslant\tau\leqslant 1.25 as well as for the domain
0.02⩽ν⩽0.240.02𝜈0.240.02\leqslant\nu\leqslant 0.24
and 0.9⩽τ⩽1.250.9𝜏1.250.9\leqslant\tau\leqslant 1.25,
the mapping of the variance
ν~​(μ,ω,ν,τ,λ,α)~𝜈𝜇𝜔𝜈𝜏𝜆𝛼{\tilde{\nu}}(\mu,\omega,\nu,\tau,\lambda,\alpha) given in Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))
increases:
ν~​(μ,ω,ν,τ,λ01,α01)>ν~𝜈𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01𝜈{\tilde{\nu}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})>\nu.

The proof can be found in the Appendix Section [A3](#S3 "A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").
All fixed
points (μ,ν)𝜇𝜈(\mu,\nu) of the mapping g𝑔g (Eq. ([3](#Sx2.E3 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))) ensure for 0.8⩽τ0.8𝜏0.8\leqslant\tau that
ν~>0.16~𝜈0.16{\tilde{\nu}}>0.16
and for 0.9⩽τ0.9𝜏0.9\leqslant\tau that ν~>0.24~𝜈0.24{\tilde{\nu}}>0.24.
Consequently, the variance mapping Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) ensures a lower bound on the variance ν𝜈\nu.
Therefore SELU networks control the variance of the activations and
push it into an interval, whereafter the mean and variance move toward
the fixed point.
Thus, SELU networks are steadily normalizing the variance and
subsequently normalizing the mean, too.
In all experiments, we observed that
self-normalizing neural networks push the mean and variance of activations into the domain ΩΩ\Omega .

##### Initialization.

Since SNNs have a fixed point at zero mean and unit variance
for normalized weights ω=∑i=1nwi=0𝜔superscriptsubscript𝑖1𝑛subscript𝑤𝑖0\omega=\sum\_{i=1}^{n}w\_{i}=0 and
τ=∑i=1nwi2=1𝜏superscriptsubscript𝑖1𝑛superscriptsubscript𝑤𝑖21\tau=\sum\_{i=1}^{n}w\_{i}^{2}=1 (see above),
we initialize SNNs such that these
constraints are fulfilled in expectation.
We draw the weights from a Gaussian distribution
with E​(wi)=0Esubscript𝑤𝑖0\mathbf{\mathrm{E}}(w\_{i})=0 and variance Var​(wi)=1/nVarsubscript𝑤𝑖1𝑛\mathbf{\mathrm{Var}}(w\_{i})=1/n.
Uniform and truncated Gaussian distributions with these moments
led to networks with similar behavior.
The “MSRA initialization” is similar since
it uses zero mean and variance 2/n2𝑛2/n to initialize the weights [[17](#bib.bib17)].
The additional factor 222 counters the effect of rectified
linear units.

##### New Dropout Technique.

Standard dropout randomly sets an activation x𝑥x to zero with probability 1−q1𝑞1-q for 0<q⩽10𝑞10<q\leqslant 1.
In order to preserve the mean, the activations are scaled by 1/q1𝑞1/q during training.
If x𝑥x has mean E​(x)=μE𝑥𝜇\mathbf{\mathrm{E}}(x)=\mu and variance
Var​(x)=νVar𝑥𝜈\mathbf{\mathrm{Var}}(x)=\nu, and the dropout variable d𝑑d follows
a binomial distribution B​(1,q)𝐵1𝑞B(1,q), then the mean E​(1/q​d​x)=μE1𝑞𝑑𝑥𝜇\mathbf{\mathrm{E}}(1/qdx)=\mu is kept.
Dropout fits well to rectified linear units, since
zero is in the low variance region and corresponds
to the default value.
For scaled exponential linear units, the default and low variance
value is limx→−∞selu⁡(x)=−λ​α=α′subscript→𝑥selu𝑥𝜆𝛼superscript𝛼′\lim\_{x\to-\infty}\operatorname{selu}(x)=-\lambda\alpha=\alpha^{\prime}.
Therefore, we propose “alpha dropout”,
that randomly sets inputs to α′superscript𝛼′\alpha^{\prime}.
The new mean and new variance is
E​(x​d+α′​(1−d))=q​μ+(1−q)​α′E𝑥𝑑superscript𝛼′1𝑑𝑞𝜇1𝑞superscript𝛼′\mathbf{\mathrm{E}}(xd+\alpha^{\prime}(1-d))=q\mu+(1-q)\alpha^{\prime}, and
Var​(x​d+α′​(1−d))=q​((1−q)​(α′−μ)2+ν)Var𝑥𝑑superscript𝛼′1𝑑𝑞1𝑞superscriptsuperscript𝛼′𝜇2𝜈\mathbf{\mathrm{Var}}(xd+\alpha^{\prime}(1-d))=q((1-q)(\alpha^{\prime}-\mu)^{2}+\nu).
We aim at keeping mean and variance to their original values after “alpha
dropout”, in order to ensure the self-normalizing property even for “alpha dropout”.
The affine transformation a​(x​d+α′​(1−d))+b𝑎𝑥𝑑superscript𝛼′1𝑑𝑏a(xd+\alpha^{\prime}(1-d))+b allows to
determine parameters a𝑎a and b𝑏b such that mean and variance are kept to their values:
E​(a​(x​d+α′​(1−d))+b)=μandVar​(a​(x​d+α′​(1−d))+b)=ν.formulae-sequenceE𝑎𝑥𝑑superscript𝛼′1𝑑𝑏

𝜇andVar𝑎𝑥𝑑superscript𝛼′1𝑑𝑏𝜈\mathbf{\mathrm{E}}(a(xd+\alpha^{\prime}(1-d))+b)=\mu\ \ \text{and}\ \ \mathbf{\mathrm{Var}}(a(xd+\alpha^{\prime}(1-d))+b)=\nu\ .
In contrast to dropout, a𝑎a and b𝑏b will depend on μ𝜇\mu and ν𝜈\nu,
however our SNNs converge to activations with
zero mean and unit variance.
With μ=0𝜇0\mu=0 and ν=1𝜈1\nu=1, we obtain a=(q+α′⁣2​q​(1−q))−1/2𝑎superscript𝑞superscript𝛼

′2𝑞1𝑞12a=\left(q+\alpha^{\prime 2}q(1-q)\right)^{-1/2} and b=−(q+α′⁣2​q​(1−q))−1/2​((1−q)​α′)𝑏superscript𝑞superscript𝛼

′2𝑞1𝑞121𝑞superscript𝛼′b=-\left(q+\alpha^{\prime 2}q(1-q)\right)^{-1/2}\left((1-q)\alpha^{\prime}\right).
The parameters a𝑎a and b𝑏b only depend on the dropout rate 1−q1𝑞1-q
and the most negative activation α′superscript𝛼′\alpha^{\prime}.
Empirically, we found that dropout rates 1−q=0.051𝑞0.051-q=0.05 or 0.100.100.10 lead to models with good performance.
“Alpha-dropout” fits well to scaled exponential linear units by randomly setting
activations to the negative saturation value.

##### Applicability of the central limit theorem and independence assumption.

In the derivative of the mapping (Eq. ([3](#Sx2.E3 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))), we used the central limit theorem (CLT)
to approximate the network inputs z=∑i=1nwi​xi𝑧superscriptsubscript𝑖1𝑛subscript𝑤𝑖subscript𝑥𝑖z=\sum\_{i=1}^{n}w\_{i}x\_{i} with a normal distribution.
We justified normality because network inputs represent a weighted sum of the inputs xisubscript𝑥𝑖x\_{i}, where for Deep Learning n𝑛n is typically large.
The Berry-Esseen theorem states that the convergence rate to normality is n−1/2superscript𝑛12n^{-1/2} [[22](#bib.bib22)].
In the classical version of the CLT, the random variables have to be independent and identically
distributed, which typically does not hold for neural networks.
However, the Lyapunov CLT does not require the variable to be identically distributed anymore. Furthermore,
even under weak dependence, sums of random variables converge in distribution to a Gaussian distribution [[5](#bib.bib5)].

## Experiments

We compare SNNs to other deep networks at different
benchmarks.
Hyperparameters such as
number of layers (blocks), neurons per layer, learning rate, and dropout rate,
are adjusted by grid-search for each dataset on a separate validation set
(see Section [A4](#S4 "A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")).
We compare the following FNN methods:

* •

  “MSRAinit”: FNNs without normalization and
  with ReLU activations and “Microsoft weight initialization” [[17](#bib.bib17)].
* •

  “BatchNorm”: FNNs with batch normalization [[20](#bib.bib20)].
* •

  “LayerNorm”: FNNs with layer normalization [[2](#bib.bib2)].
* •

  “WeightNorm”: FNNs with weight normalization [[32](#bib.bib32)].
* •

  “Highway”: Highway networks [[35](#bib.bib35)].
* •

  “ResNet”: Residual networks [[16](#bib.bib16)] adapted to FNNs
  using residual blocks with 2 or 3 layers with rectangular or diavolo shape.
* •

  “SNNs”: Self normalizing networks with SELUs with α=α01𝛼subscript𝛼01\alpha=\alpha\_{\mathrm{01}} and λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\mathrm{01}} and
  the proposed dropout technique and initialization strategy.

##### 121 UCI Machine Learning Repository datasets.

The benchmark comprises 121 classification datasets from the UCI Machine Learning repository [[10](#bib.bib10)]
from diverse application areas, such as physics, geology, or biology.
The size of the datasets ranges between 101010 and 130,000

130000130,000 data points and the
number of features from 444 to 250250250.
In abovementioned work [[10](#bib.bib10)],
there were methodological mistakes [[37](#bib.bib37)] which we avoided here.
Each compared FNN method
was optimized with respect to its architecture and hyperparameters on a validation set that was then
removed from the subsequent analysis.
The selected hyperparameters served to evaluate the methods in terms of accuracy on
the pre-defined test sets (details on the hyperparameter selection are given in Section [A4](#S4 "A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")).
The accuracies are reported in the Table [A11](#S4.T11 "Table A11 ‣ Results of FNN methods for all 121 data sets. ‣ A4.2 121 UCI Machine Learning Repository data sets: detailed results ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks").
We ranked the methods by their accuracy for each
prediction task and compared their average ranks.
SNNs significantly outperform all competing networks in pairwise comparisons (paired
Wilcoxon test across datasets) as reported in Table [1](#Sx3.T1 "Table 1 ‣ 121 UCI Machine Learning Repository datasets. ‣ Experiments ‣ Self-Normalizing Neural Networks") (left panel).

Table 1: Left: Comparison of seven FNNs on 121 UCI tasks.
We consider the average rank difference to rank 444, which is
the average rank of seven methods with random predictions.
The first column gives the method, the second
the average rank difference, and the last the p𝑝p-value
of a paired Wilcoxon test whether the difference to the best performing
method is significant.
SNNs significantly outperform all other methods.
Right: Comparison of 24 machine learning methods (ML) on the UCI datasets
with more than 1000 data points.
The first column gives the method, the second
the average rank difference to rank 12.512.512.5, and the last the p𝑝p-value
of a paired Wilcoxon test whether the difference to the best performing
method is significant. Methods that were significantly worse than
the best method are marked with “\*”.
The full tables can be found in Table [A11](#S4.T11 "Table A11 ‣ Results of FNN methods for all 121 data sets. ‣ A4.2 121 UCI Machine Learning Repository data sets: detailed results ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks"), Table [A12](#S4.T12 "Table A12 ‣ Results. ‣ A4.2 121 UCI Machine Learning Repository data sets: detailed results ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks") and Table [A13](#S4.T13 "Table A13 ‣ Results. ‣ A4.2 121 UCI Machine Learning Repository data sets: detailed results ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks").
SNNs outperform all competing methods.

| FNN method comparison | | | ML method comparison | | |
| --- | --- | --- | --- | --- | --- |
| Method | avg. rank diff. | p𝑝p-value | Method | avg. rank diff. | p𝑝p-value |
| SNN | -0.756 |  | SNN | -6.7 |  |
| MSRAinit | -0.240\* | 2.7e-02 | SVM | -6.4 | 5.8e-01 |
| LayerNorm | -0.198\* | 1.5e-02 | RandomForest | -5.9 | 2.1e-01 |
| Highway | 0.021\* | 1.9e-03 | MSRAinit | -5.4\* | 4.5e-03 |
| ResNet | 0.273\* | 5.4e-04 | LayerNorm | -5.3 | 7.1e-02 |
| WeightNorm | 0.397\* | 7.8e-07 | Highway | -4.6\* | 1.7e-03 |
| BatchNorm | 0.504\* | 3.5e-06 | ……\ldots | ……\ldots | ……\ldots |

We further included 17 machine learning methods representing diverse method groups [[10](#bib.bib10)]
in the comparison and
the grouped the data sets into “small” and “large” data sets (for details see Section [A4](#S4 "A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")).
On 75 small datasets with less than 1000 data points, random forests and SVMs outperform SNNs and other FNNs.
On 46 larger datasets with at least 1000 data points,
SNNs show the highest performance followed by SVMs and random forests (see right panel of Table [1](#Sx3.T1 "Table 1 ‣ 121 UCI Machine Learning Repository datasets. ‣ Experiments ‣ Self-Normalizing Neural Networks"),
for complete results see Tables [A12](#S4.T12 "Table A12 ‣ Results. ‣ A4.2 121 UCI Machine Learning Repository data sets: detailed results ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks") and [A12](#S4.T12 "Table A12 ‣ Results. ‣ A4.2 121 UCI Machine Learning Repository data sets: detailed results ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")).
Overall, SNNs have outperformed state of the art machine learning methods on UCI datasets
with more than 1,000 data points.

Typically, hyperparameter selection chose SNN architectures that were
much deeper than the selected architectures of other FNNs, with an average depth of 10.8 layers,
compared to average depths of 6.0 for BatchNorm, 3.8 WeightNorm, 7.0 LayerNorm, 5.9 Highway,
and 7.1 for MSRAinit networks. For ResNet, the average number of blocks was 6.35.
SNNs with many more than 4 layers often provide the best predictive accuracies across all neural networks.

##### Drug discovery: The Tox21 challenge dataset.

The Tox21 challenge dataset comprises about 12,000 chemical compounds
whose twelve toxic effects have to be predicted based on their chemical structure.
We used the validation sets
of the challenge winners for hyperparameter selection (see Section [A4](#S4 "A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")) and
the challenge test set for performance comparison.
We repeated the whole evaluation procedure 5 times
to obtain error bars.
The results in terms of average AUC are given in Table [2](#Sx3.T2 "Table 2 ‣ Drug discovery: The Tox21 challenge dataset. ‣ Experiments ‣ Self-Normalizing Neural Networks").
In 2015, the challenge organized by the US NIH
was won by an ensemble of shallow ReLU FNNs which achieved an AUC of 0.846 [[28](#bib.bib28)].
Besides FNNs, this ensemble also contained random forests and SVMs.
Single SNNs came close with an AUC of 0.845±plus-or-minus\pm0.003.
The best performing SNNs have 8 layers, compared to the runner-ups ReLU networks with layer normalization with 2 and 3 layers.
Also batchnorm and weightnorm networks, typically perform best with shallow
networks of 2 to 4 layers (Table [2](#Sx3.T2 "Table 2 ‣ Drug discovery: The Tox21 challenge dataset. ‣ Experiments ‣ Self-Normalizing Neural Networks")). The deeper the networks, the
larger the difference in performance between SNNs and other methods (see columns 5–8 of Table [2](#Sx3.T2 "Table 2 ‣ Drug discovery: The Tox21 challenge dataset. ‣ Experiments ‣ Self-Normalizing Neural Networks")).
The best performing method is an SNN with 8 layers.

Table 2: Comparison of FNNs at the Tox21 challenge dataset
in terms of AUC. The rows represent different methods and the columns
different network depth and for ResNets the number of residual blocks
(“na”: 32 blocks were omitted due to computational constraints).
The deeper the networks, the more prominent is the advantage of SNNs.
The best networks are SNNs with 8 layers.

| #layers / #blocks | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| method | 2 | 3 | 4 | 6 | 8 | 16 | 32 |
| SNN | 83.7 ±plus-or-minus\pm 0.3 | 84.4 ±plus-or-minus\pm 0.5 | 84.2 ±plus-or-minus\pm 0.4 | 83.9 ±plus-or-minus\pm 0.5 | 84.5 ±plus-or-minus\pm 0.2 | 83.5 ±plus-or-minus\pm 0.5 | 82.5 ±plus-or-minus\pm 0.7 |
| Batchnorm | 80.0 ±plus-or-minus\pm 0.5 | 79.8 ±plus-or-minus\pm 1.6 | 77.2 ±plus-or-minus\pm 1.1 | 77.0 ±plus-or-minus\pm 1.7 | 75.0 ±plus-or-minus\pm 0.9 | 73.7 ±plus-or-minus\pm 2.0 | 76.0 ±plus-or-minus\pm 1.1 |
| WeightNorm | 83.7 ±plus-or-minus\pm 0.8 | 82.9 ±plus-or-minus\pm 0.8 | 82.2 ±plus-or-minus\pm 0.9 | 82.5 ±plus-or-minus\pm 0.6 | 81.9 ±plus-or-minus\pm 1.2 | 78.1 ±plus-or-minus\pm 1.3 | 56.6 ±plus-or-minus\pm 2.6 |
| LayerNorm | 84.3 ±plus-or-minus\pm 0.3 | 84.3 ±plus-or-minus\pm 0.5 | 84.0 ±plus-or-minus\pm 0.2 | 82.5 ±plus-or-minus\pm 0.8 | 80.9 ±plus-or-minus\pm 1.8 | 78.7 ±plus-or-minus\pm 2.3 | 78.8 ±plus-or-minus\pm 0.8 |
| Highway | 83.3 ±plus-or-minus\pm 0.9 | 83.0 ±plus-or-minus\pm 0.5 | 82.6 ±plus-or-minus\pm 0.9 | 82.4 ±plus-or-minus\pm 0.8 | 80.3 ±plus-or-minus\pm 1.4 | 80.3 ±plus-or-minus\pm 2.4 | 79.6 ±plus-or-minus\pm 0.8 |
| MSRAinit | 82.7 ±plus-or-minus\pm 0.4 | 81.6 ±plus-or-minus\pm 0.9 | 81.1 ±plus-or-minus\pm 1.7 | 80.6 ±plus-or-minus\pm 0.6 | 80.9 ±plus-or-minus\pm 1.1 | 80.2 ±plus-or-minus\pm 1.1 | 80.4 ±plus-or-minus\pm 1.9 |
| ResNet | 82.2 ±plus-or-minus\pm 1.1 | 80.0 ±plus-or-minus\pm 2.0 | 80.5 ±plus-or-minus\pm 1.2 | 81.2 ±plus-or-minus\pm 0.7 | 81.8 ±plus-or-minus\pm 0.6 | 81.2 ±plus-or-minus\pm 0.6 | na |

##### Astronomy: Prediction of pulsars in the HTRU2 dataset.

Since a decade, machine learning methods have been used to identify pulsars in radio wave signals [[27](#bib.bib27)].
Recently, the High Time Resolution Universe Survey (HTRU2) dataset has been released with
1,639 real pulsars and 16,259 spurious signals.
Currently, the highest AUC value of a 10-fold cross-validation is 0.976
which has been achieved by Naive Bayes classifiers followed by decision tree C4.5 with 0.949 and SVMs with 0.929.
We used eight features constructed by the PulsarFeatureLab as used previously [[27](#bib.bib27)].
We assessed the performance of FNNs using 10-fold nested cross-validation,
where the hyperparameters were selected in the inner loop on a validation set (for details on the hyperparameter selection see Section [A4](#S4 "A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")).
Table [3](#Sx3.T3 "Table 3 ‣ Astronomy: Prediction of pulsars in the HTRU2 dataset. ‣ Experiments ‣ Self-Normalizing Neural Networks") reports the results
in terms of AUC. SNNs outperform all other methods and have pushed the state-of-the-art
to an AUC of 0.980.980.98.

Table 3: Comparison of FNNs and reference methods at HTRU2
in terms of AUC.
The first, fourth and seventh column give the method,
the second, fifth and eight column the AUC averaged over 10 cross-validation folds,
and the third and sixth column the p𝑝p-value of a paired Wilcoxon test of the AUCs against
the best performing method across the 10 folds.
FNNs achieve better results than Naive Bayes (NB), C4.5, and SVM.
SNNs exhibit the best performance and set a new record.

| FNN methods | | | FNN methods | | | ref. methods | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| method | AUC | p𝑝p-value | method | AUC | p𝑝p-value | method | AUC |
| SNN | 0.9803     ±plus-or-minus\pm 0.010 |  |  |  |  |  |  |
| MSRAinit | 0.9791     ±plus-or-minus\pm 0.010 | 3.5e-01 | LayerNorm | 0.9762\* ±plus-or-minus\pm 0.011 | 1.4e-02 | NB | 0.976 |
| WeightNorm | 0.9786\* ±plus-or-minus\pm 0.010 | 2.4e-02 | BatchNorm | 0.9760     ±plus-or-minus\pm 0.013 | 6.5e-02 | C4.5 | 0.946 |
| Highway | 0.9766\* ±plus-or-minus\pm 0.009 | 9.8e-03 | ResNet | 0.9753\* ±plus-or-minus\pm 0.010 | 6.8e-03 | SVM | 0.929 |

## Conclusion

We have introduced self-normalizing neural networks for
which we have proved that neuron activations are pushed towards zero mean and unit variance
when propagated through the network.
Additionally, for activations not close to unit
variance, we have proved an upper and lower bound
on the variance mapping. Consequently, SNNs do not face vanishing and exploding gradient
problems. Therefore, SNNs work well for architectures with many layers, allowed us to introduce a
novel regularization scheme, and learn very robustly.
On 121 UCI benchmark datasets, SNNs have outperformed other FNNs with and without normalization techniques,
such as batch, layer, and weight normalization, or specialized architectures, such as Highway or
Residual networks.
SNNs also yielded the best results on drug discovery and astronomy tasks.
The best performing SNN architectures are typically very deep in contrast to other FNNs.

## Acknowledgments

This work was supported by IWT research grant IWT150865 (Exaptation), H2020 project
grant 671555 (ExCAPE), grant IWT135122 (ChemBioBridge),
Zalando SE with Research Agreement 01/2016,
Audi.JKU Deep Learning Center, Audi Electronic Venture GmbH,
and the NVIDIA Corporation.

## References

The references are provided in Section [A7](#S7 "A7 References ‣ Self-Normalizing Neural Networks").

## Appendix

###### Contents

1. [A1 Background](#S1 "In Self-Normalizing Neural Networks")
2. [A2 Theorems](#S2 "In Self-Normalizing Neural Networks")
   1. [A2.1 Theorem 1: Stable and Attracting Fixed Points Close to (0,1)](#S2.SS1 "In A2 Theorems ‣ Self-Normalizing Neural Networks")
   2. [A2.2 Theorem 2: Decreasing Variance from Above](#S2.SS2 "In A2 Theorems ‣ Self-Normalizing Neural Networks")
   3. [A2.3 Theorem 3: Increasing Variance from Below](#S2.SS3 "In A2 Theorems ‣ Self-Normalizing Neural Networks")
3. [A3 Proofs of the Theorems](#S3 "In Self-Normalizing Neural Networks")
   1. [A3.1 Proof of Theorem 1](#S3.SS1 "In A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
   2. [A3.2 Proof of Theorem 2](#S3.SS2 "In A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
   3. [A3.3 Proof of Theorem 3](#S3.SS3 "In A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
   4. [A3.4 Lemmata and Other Tools Required for the Proofs](#S3.SS4 "In A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
      1. [A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one](#S3.SS4.SSS1 "In A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
      2. [A3.4.2 Lemmata for proofing Theorem 1 (part 2): Mapping within domain](#S3.SS4.SSS2 "In A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
      3. [A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting](#S3.SS4.SSS3 "In A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
      4. [A3.4.4 Lemmata for proofing Theorem 3: The variance is expanding](#S3.SS4.SSS4 "In A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
      5. [A3.4.5 Computer-assisted proof details for main Lemma 12 in Section A3.4.1.](#S3.SS4.SSS5 "In A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
      6. [A3.4.6 Intermediate Lemmata and Proofs](#S3.SS4.SSS6 "In A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
4. [A4 Additional information on experiments](#S4 "In Self-Normalizing Neural Networks")
   1. [A4.1 121 UCI Machine Learning Repository data sets: Hyperparameters](#S4.SS1 "In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
   2. [A4.2 121 UCI Machine Learning Repository data sets: detailed results](#S4.SS2 "In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
   3. [A4.3 Tox21 challenge data set: Hyperparameters](#S4.SS3 "In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
   4. [A4.4 HTRU2 data set: Hyperparameters](#S4.SS4 "In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
5. [A5 Other fixed points](#S5 "In Self-Normalizing Neural Networks")
6. [A6 Bounds determined by numerical methods](#S6 "In Self-Normalizing Neural Networks")
7. [A7 References](#S7 "In Self-Normalizing Neural Networks")
8. [References](#bib "In Self-Normalizing Neural Networks")

This appendix is organized as follows: the first section
sets the background, definitions, and formulations.
The main theorems are presented in the next section.
The following section is devoted to the proofs of these theorems.
The next section reports additional results and details on the
performed computational experiments, such as hyperparameter selection.
The last section shows that our theoretical bounds can be
confirmed by numerical methods as a sanity check.

The proof of theorem 1 is based on the Banach’s fixed point theorem
for which we require (1) a contraction mapping, which is proved in Subsection [A3.4.1](#S3.SS4.SSS1 "A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
and (2) that the mapping stays within its domain, which is proved in Subsection [A3.4.2](#S3.SS4.SSS2 "A3.4.2 Lemmata for proofing Theorem 1 (part 2): Mapping within domain ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
For part (1), the proof relies on the main Lemma 12, which is a computer-assisted proof, and can be found
in Subsection [A3.4.1](#S3.SS4.SSS1 "A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"). The validity of the computer-assisted proof is shown in Subsection [A3.4.5](#S3.SS4.SSS5 "A3.4.5 Computer-assisted proof details for main Lemma 12 in Section A3.4.1. ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") by
error analysis and the precision of the functions’ implementation.
The last Subsection [A3.4.6](#S3.SS4.SSS6 "A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") compiles various lemmata with intermediate results that support
the proofs of the main lemmata and theorems.

## A1 Background

We consider a neural network with activation function f𝑓f and
two consecutive layers that are connected by weight matrix 𝑾𝑾\bm{W}.
Since samples that serve as input to the neural network are chosen according to a distribution,
the activations x𝑥\bm{x} in the lower layer,
the network inputs 𝒛=𝑾​𝒙𝒛𝑾𝒙\bm{z}=\bm{W}\bm{x}, and activations y=f​(z)𝑦𝑓𝑧\bm{y}=f(\bm{z}) in the
higher layer are all random variables. We assume that all units xisubscript𝑥𝑖x\_{i} in the lower layer
have mean activation μ:=E⁡(xi)assign𝜇Esubscript𝑥𝑖\mu:=\operatorname{E}(x\_{i}) and variance of the
activation
ν:=Var⁡(xi)assign𝜈Varsubscript𝑥𝑖\nu:=\operatorname{Var}(x\_{i}) and a unit y𝑦y in the
higher layer has mean activation μ~:=E⁡(y)assign~𝜇E𝑦{\tilde{\mu}}:=\operatorname{E}(y) and variance
ν~:=Var⁡(y)assign~𝜈Var𝑦{\tilde{\nu}}:=\operatorname{Var}(y). Here E(.)\operatorname{E}(.) denotes the expectation and
Var(.)\operatorname{Var}(.) the variance of a random variable.
For activation of unit y𝑦y, we have net input z=𝒘T​𝒙𝑧superscript𝒘𝑇𝒙z=\bm{w}^{T}\bm{x} and
the scaled exponential linear unit (SELU)
activation y=selu⁡(z)𝑦selu𝑧y=\operatorname{selu}(z), with

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | selu⁡(x)selu𝑥\displaystyle\operatorname{selu}(x)\ | =λ​{xif ​x>0α​ex−αif ​x⩽0.absent𝜆cases𝑥if 𝑥0𝛼superscript𝑒𝑥𝛼if 𝑥0\displaystyle=\ \lambda\ \begin{cases}x&\text{if }x>0\\ \alpha e^{x}-\alpha&\text{if }x\leqslant 0\end{cases}\ . |  | (7) |

For n𝑛n units xi,1⩽i⩽n

subscript𝑥𝑖1
𝑖𝑛x\_{i},1\leqslant i\leqslant n in the lower layer and
the weight vector 𝒘∈ℝn𝒘superscriptℝ𝑛\bm{w}\in\mathbb{R}^{n}, we define
n𝑛n times the mean by ω:=∑i=1nwiassign𝜔superscriptsubscript𝑖1𝑛subscript𝑤𝑖\omega:=\sum\_{i=1}^{n}w\_{i}
and n𝑛n times the second moment by τ:=∑i=1nwi2assign𝜏superscriptsubscript𝑖1𝑛superscriptsubscript𝑤𝑖2\tau:=\sum\_{i=1}^{n}w\_{i}^{2}.

We define a mapping g𝑔g from mean μ𝜇\mu and
variance ν𝜈\nu of one layer
to the mean μ~~𝜇{\tilde{\mu}} and variance ν~~𝜈{\tilde{\nu}} in the next layer:

|  |  |  |  |
| --- | --- | --- | --- |
|  | g:(μ,ν)↦(μ~,ν~).:𝑔maps-to𝜇𝜈~𝜇~𝜈\displaystyle g:(\mu,\nu)\mapsto({\tilde{\mu}},{\tilde{\nu}})\ . |  | (8) |

For neural networks with scaled exponential linear units,
the mean is of the activations in the next layer computed according to

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | μ~~𝜇\displaystyle{\tilde{\mu}}\ | =∫−∞0λ​α​(exp⁡(z)−1)​pGauss​(z;μ​ω,ν​τ)​𝑑z+∫0∞λ​z​pGauss​(z;μ​ω,ν​τ)​𝑑z,absentsuperscriptsubscript0𝜆𝛼𝑧1subscript𝑝Gauss  𝑧𝜇𝜔𝜈𝜏 differential-d𝑧superscriptsubscript0𝜆𝑧subscript𝑝Gauss  𝑧𝜇𝜔𝜈𝜏 differential-d𝑧\displaystyle=\ \int\_{-\infty}^{0}\lambda\alpha(\exp(z)-1)p\_{\mathrm{Gauss}}(z;\mu\omega,\sqrt{\nu\tau})dz\ +\ \int\_{0}^{\infty}\lambda zp\_{\mathrm{Gauss}}(z;\mu\omega,\sqrt{\nu\tau})dz\ , |  | (9) |

and the second moment of the activations in the next layer is computed according to

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ξ~~𝜉\displaystyle{\tilde{\xi}}\ | =∫−∞0λ2​α2​(exp⁡(z)−1)2​pGauss​(z;μ​ω,ν​τ)​𝑑z+∫0∞λ2​z2​pGauss​(z;μ​ω,ν​τ)​𝑑z.absentsuperscriptsubscript0superscript𝜆2superscript𝛼2superscript𝑧12subscript𝑝Gauss  𝑧𝜇𝜔𝜈𝜏 differential-d𝑧superscriptsubscript0superscript𝜆2superscript𝑧2subscript𝑝Gauss  𝑧𝜇𝜔𝜈𝜏 differential-d𝑧\displaystyle=\ \int\_{-\infty}^{0}\lambda^{2}\alpha^{2}(\exp(z)-1)^{2}p\_{\mathrm{Gauss}}(z;\mu\omega,\sqrt{\nu\tau})dz\ +\ \int\_{0}^{\infty}\lambda^{2}z^{2}p\_{\mathrm{Gauss}}(z;\mu\omega,\sqrt{\nu\tau})dz\ . |  | (10) |

Therefore, the expressions μ~~𝜇{\tilde{\mu}} and ν~~𝜈{\tilde{\nu}} have the following form:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | μ~(μ,ω,ν,τ,λ,α)=12λ(−(α+μω)erfc(μ​ω2​ν​τ)+\displaystyle{\tilde{\mu}}(\mu,\omega,\nu,\tau,\lambda,\alpha)\ =\frac{1}{2}\lambda\left(-(\alpha+\mu\omega)\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+\right. |  | (11) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | αeμ​ω+ν​τ2erfc(μ​ω+ν​τ2​ν​τ)+2πν​τe−μ2​ω22​ν​τ+2μω)\displaystyle\left.\alpha e^{\mu\omega+\frac{\nu\tau}{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+\sqrt{\frac{2}{\pi}}\sqrt{\nu\tau}e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}+2\mu\omega\right) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ν~​(μ,ω,ν,τ,λ,α)=ξ~​(μ,ω,ν,τ,λ,α)−(μ~​(μ,ω,ν,τ,λ,α))2~𝜈𝜇𝜔𝜈𝜏𝜆𝛼~𝜉𝜇𝜔𝜈𝜏𝜆𝛼superscript~𝜇𝜇𝜔𝜈𝜏𝜆𝛼2\displaystyle{\tilde{\nu}}(\mu,\omega,\nu,\tau,\lambda,\alpha)\ ={\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda,\alpha)-\left({\tilde{\mu}}(\mu,\omega,\nu,\tau,\lambda,\alpha)\right)^{2} |  | (12) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ξ~(μ,ω,ν,τ,λ,α)=12λ2(((μω)2+ντ)(erf(μ​ω2​ν​τ)+1)+\displaystyle{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda,\alpha)\ =\ \frac{1}{2}\lambda^{2}\left(\left((\mu\omega)^{2}+\nu\tau\right)\left(\operatorname{erf}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+1\right)+\right. |  | (13) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | α2(−2eμ​ω+ν​τ2erfc(μ​ω+ν​τ2​ν​τ)+e2​(μ​ω+ν​τ)erfc(μ​ω+2​ν​τ2​ν​τ)+\displaystyle\left.\alpha^{2}\left(-2e^{\mu\omega+\frac{\nu\tau}{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+e^{2(\mu\omega+\nu\tau)}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+\right.\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | erfc(μ​ω2​ν​τ))+2π(μω)ν​τe−(μ​ω)22​(ν​τ))\displaystyle\left.\left.\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)+\sqrt{\frac{2}{\pi}}(\mu\omega)\sqrt{\nu\tau}e^{-\frac{(\mu\omega)^{2}}{2(\nu\tau)}}\right) |  |

We solve equations Eq. [4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks") and
Eq. [5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks") for fixed points μ~=μ~𝜇𝜇{\tilde{\mu}}=\mu and ν~=ν~𝜈𝜈{\tilde{\nu}}=\nu.
For a normalized weight vector with ω=0𝜔0\omega=0 and τ=1𝜏1\tau=1 and the
fixed point (μ,ν)=(0,1)𝜇𝜈01(\mu,\nu)=(0,1),
we can solve equations Eq. [4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks") and
Eq. [5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks") for α𝛼\alpha and λ𝜆\lambda.
We denote the solutions to fixed point (μ,ν)=(0,1)𝜇𝜈01(\mu,\nu)=(0,1)
by α01subscript𝛼01\alpha\_{\rm 01} and λ01subscript𝜆01\lambda\_{\rm 01}.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | α01=−2πerfc⁡(12)​exp⁡(12)−1≈1.67326subscript𝛼012𝜋erfc121211.67326\displaystyle\alpha\_{\rm 01}=-\frac{\sqrt{\frac{2}{\pi}}}{\operatorname{erfc}\left(\frac{1}{\sqrt{2}}\right)\exp\left(\frac{1}{2}\right)-1}\approx 1.67326 |  | (14) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | λ01=(1−erfc⁡(12)​e)​2​πsubscript𝜆011erfc12𝑒2𝜋\displaystyle\lambda\_{\rm 01}=\left(1-\operatorname{erfc}\left(\frac{1}{\sqrt{2}}\right)\sqrt{e}\right)\sqrt{2\pi} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (2erfc(2)e2+πerfc(12)2e−2(2+π)erfc(12)e+π+2)−1/2\displaystyle\left(2\operatorname{erfc}\left(\sqrt{2}\right)e^{2}+\pi\operatorname{erfc}\left(\frac{1}{\sqrt{2}}\right)^{2}e-2(2+\pi)\operatorname{erfc}\left(\frac{1}{\sqrt{2}}\right)\sqrt{e}+\pi+2\right)^{-1/2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | λ01≈1.0507.subscript𝜆011.0507\displaystyle\lambda\_{\rm 01}\approx 1.0507\ . |  |

The parameters α01subscript𝛼01\alpha\_{\rm 01} and λ01subscript𝜆01\lambda\_{\rm 01} ensure

|  |  |  |
| --- | --- | --- |
|  | μ~​(0,0,1,1,λ01,α01)=0~𝜇0011subscript𝜆01subscript𝛼010\displaystyle{\tilde{\mu}}(0,0,1,1,\lambda\_{\rm 01},\alpha\_{\rm 01})=0 |  |
|  |  |  |
| --- | --- | --- |
|  | ν~​(0,0,1,1,λ01,α01)=1~𝜈0011subscript𝜆01subscript𝛼011\displaystyle{\tilde{\nu}}(0,0,1,1,\lambda\_{\rm 01},\alpha\_{\rm 01})=1 |  |

Since we focus on the fixed point (μ,ν)=(0,1)𝜇𝜈01(\mu,\nu)=(0,1),
we assume throughout the analysis that α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01} and λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01}.
We consider the functions μ~​(μ,ω,ν,τ,λ01,α01)~𝜇𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01{\tilde{\mu}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01}),
ν~​(μ,ω,ν,τ,λ01,α01)~𝜈𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01{\tilde{\nu}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01}),
and ξ~​(μ,ω,ν,τ,λ01,α01)~𝜉𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})
on the domain
Ω={(μ,ω,ν,τ)|μ∈[μmin,μmax]=[−0.1,0.1],ω∈[ωmin,ωmax]=[−0.1,0.1],ν∈[νmin,νmax]=[0.8,1.5],τ∈[τmin,τmax]=[0.95,1.1]}Ωconditional-set𝜇𝜔𝜈𝜏formulae-sequence𝜇subscript𝜇minsubscript𝜇max0.10.1𝜔subscript𝜔minsubscript𝜔max0.10.1𝜈subscript𝜈minsubscript𝜈max0.81.5𝜏subscript𝜏minsubscript𝜏max0.951.1\Omega=\{(\mu,\omega,\nu,\tau)\ |\ \mu\in[\mu\_{\rm min},\mu\_{\rm max}]=[-0.1,0.1],\omega\in[\omega\_{\rm min},\omega\_{\rm max}]=[-0.1,0.1],\nu\in[\nu\_{\rm min},\nu\_{\rm max}]=[0.8,1.5],\tau\in[\tau\_{\rm min},\tau\_{\rm max}]=[0.95,1.1]\}.

Figure [2](#Sx2.F2 "Figure 2 ‣ Stable and Attracting Fixed Points for Unnormalized Weights. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks") visualizes
the mapping g𝑔g for ω=0𝜔0\omega=0 and τ=1𝜏1\tau=1 and
α01subscript𝛼01\alpha\_{\rm 01} and λ01subscript𝜆01\lambda\_{\rm 01} at few pre-selected points.
It can be seen that (0,1)01(0,1) is an attracting
fixed point of the mapping g𝑔g.

## A2 Theorems

### A2.1 Theorem 1: Stable and Attracting Fixed Points Close to (0,1)

Theorem [1](#Thmtheorem1 "Theorem 1 (Stable and Attracting Fixed Points). ‣ Stable and Attracting Fixed Points for Unnormalized Weights. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")
shows that the mapping g𝑔g defined by Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))
and Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))
exhibits a stable and attracting fixed point close to zero mean and
unit variance.
Theorem [1](#Thmtheorem1 "Theorem 1 (Stable and Attracting Fixed Points). ‣ Stable and Attracting Fixed Points for Unnormalized Weights. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks") establishes the self-normalizing property of self-normalizing
neural networks (SNNs). The stable and
attracting fixed point leads to robust learning through many layers.

###### Theorem 1 (Stable and Attracting Fixed Points).

We assume α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01} and λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01}.
We restrict the range of the variables to the domain
μ∈[−0.1,0.1]𝜇0.10.1\mu\in[-0.1,0.1],
ω∈[−0.1,0.1]𝜔0.10.1\omega\in[-0.1,0.1],
ν∈[0.8,1.5]𝜈0.81.5\nu\in[0.8,1.5], and
τ∈[0.95,1.1]𝜏0.951.1\tau\in[0.95,1.1].
For ω=0𝜔0\omega=0 and τ=1𝜏1\tau=1, the mapping Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))
and Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) has the stable
fixed point (μ,ν)=(0,1)𝜇𝜈01(\mu,\nu)=(0,1).
For other ω𝜔\omega and τ𝜏\tau the mapping Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))
and Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) has a stable and
attracting fixed point depending on (ω,τ)𝜔𝜏(\omega,\tau) in the
(μ,ν)𝜇𝜈(\mu,\nu)-domain: μ∈[−0.03106,0.06773]𝜇0.031060.06773\mu\in[-0.03106,0.06773] and
ν∈[0.80009,1.48617]𝜈0.800091.48617\nu\in[0.80009,1.48617].
All points within the (μ,ν)𝜇𝜈(\mu,\nu)-domain converge when
iteratively applying the mapping Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))
and Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) to this fixed point.

### A2.2 Theorem 2: Decreasing Variance from Above

The next Theorem [2](#Thmtheorem2a "Theorem 2 (Decreasing 𝜈). ‣ A2.2 Theorem 2: Decreasing Variance from Above ‣ A2 Theorems ‣ Self-Normalizing Neural Networks") states
that the variance of unit activations
does not explode through
consecutive layers of self-normalizing networks.
Even more, a large variance of unit activations decreases when
propagated through the network.
In particular this ensures that exploding gradients will never be
observed.
In contrast to the domain in previous subsection,
in which ν∈[0.8,1.5]𝜈0.81.5\nu\in[0.8,1.5], we now consider a domain
in which the variance of the inputs is higher ν∈[3,16]𝜈316\nu\in[3,16] and even the
range of the mean is increased μ∈[−1,1]𝜇11\mu\in[-1,1]. We denote this new domain with
the symbol Ω++superscriptΩabsent\Omega^{++} to indicate that the variance lies above the variance of the original domain ΩΩ\Omega.
In Ω++superscriptΩabsent\Omega^{++}, we can show that the variance ν~~𝜈{\tilde{\nu}} in the next layer is always smaller
then the original variance ν𝜈\nu.
Concretely, this theorem states that:

###### Theorem 2 (Decreasing ν𝜈\nu).

For λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01}, α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01}
and the domain Ω++superscriptΩabsent\Omega^{++}:
−1⩽μ⩽11𝜇1-1\leqslant\mu\leqslant 1,
−0.1⩽ω⩽0.10.1𝜔0.1-0.1\leqslant\omega\leqslant 0.1,
3⩽ν⩽163𝜈163\leqslant\nu\leqslant 16, and
0.8⩽τ⩽1.250.8𝜏1.250.8\leqslant\tau\leqslant 1.25 we have for
the mapping of the variance
ν~​(μ,ω,ν,τ,λ,α)~𝜈𝜇𝜔𝜈𝜏𝜆𝛼{\tilde{\nu}}(\mu,\omega,\nu,\tau,\lambda,\alpha) given in Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ν~​(μ,ω,ν,τ,λ01,α01)~𝜈𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01\displaystyle{\tilde{\nu}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ | <ν.absent𝜈\displaystyle<\ \nu\ . |  | (15) |

The variance decreases in [3,16]316[3,16] and all fixed
points (μ,ν)𝜇𝜈(\mu,\nu) of mapping Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) and Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) have ν<3𝜈3\nu<3.

### A2.3 Theorem 3: Increasing Variance from Below

The next Theorem [3](#Thmtheorem3 "Theorem 3 (Increasing 𝜈). ‣ Stable and Attracting Fixed Points for Unnormalized Weights. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks") states
that the variance of unit activations
does not vanish through
consecutive layers of self-normalizing networks.
Even more, a small variance of unit activations increases when
propagated through the network.
In particular this ensures that vanishing gradients will never be
observed.
In contrast to the first domain,
in which ν∈[0.8,1.5]𝜈0.81.5\nu\in[0.8,1.5], we now consider two domains Ω1−superscriptsubscriptΩ1\Omega\_{1}^{-} and
Ω2−superscriptsubscriptΩ2\Omega\_{2}^{-} in which the variance of the inputs is lower 0.05⩽ν⩽0.160.05𝜈0.160.05\leqslant\nu\leqslant 0.16 and 0.05⩽ν⩽0.240.05𝜈0.240.05\leqslant\nu\leqslant 0.24,
and even the parameter τ𝜏\tau is different 0.9⩽τ⩽1.250.9𝜏1.250.9\leqslant\tau\leqslant 1.25 to the original ΩΩ\Omega.
We denote this new domain with
the symbol Ωi−subscriptsuperscriptΩ𝑖\Omega^{-}\_{i} to indicate that the variance lies below the variance of the original domain ΩΩ\Omega.
In Ω1−superscriptsubscriptΩ1\Omega\_{1}^{-} and Ω2−superscriptsubscriptΩ2\Omega\_{2}^{-},
we can show that the variance ν~~𝜈{\tilde{\nu}} in the next layer is always larger
then the original variance ν𝜈\nu, which means that the variance does not vanish through
consecutive layers of self-normalizing networks.
Concretely, this theorem states that:

###### Theorem 3 (Increasing ν𝜈\nu).

We consider λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01}, α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01}
and the two domains
Ω1−={(μ,ω,ν,τ)|−0.1⩽μ⩽0.1,−0.1⩽ω⩽0.1,0.05⩽ν⩽0.16,0.8⩽τ⩽1.25}superscriptsubscriptΩ1conditional-set𝜇𝜔𝜈𝜏formulae-sequence0.1𝜇0.10.1𝜔0.10.05𝜈0.160.8𝜏1.25\Omega\_{1}^{-}=\{(\mu,\omega,\nu,\tau)\ |\ -0.1\leqslant\mu\leqslant 0.1,-0.1\leqslant\omega\leqslant 0.1,0.05\leqslant\nu\leqslant 0.16,0.8\leqslant\tau\leqslant 1.25\}
and
Ω2−={(μ,ω,ν,τ)|−0.1⩽μ⩽0.1,−0.1⩽ω⩽0.1,0.05⩽ν⩽0.24,0.9⩽τ⩽1.25}superscriptsubscriptΩ2conditional-set𝜇𝜔𝜈𝜏formulae-sequence0.1𝜇0.10.1𝜔0.10.05𝜈0.240.9𝜏1.25\Omega\_{2}^{-}=\{(\mu,\omega,\nu,\tau)\ |\ -0.1\leqslant\mu\leqslant 0.1,-0.1\leqslant\omega\leqslant 0.1,0.05\leqslant\nu\leqslant 0.24,0.9\leqslant\tau\leqslant 1.25\}.

The mapping of the variance
ν~​(μ,ω,ν,τ,λ,α)~𝜈𝜇𝜔𝜈𝜏𝜆𝛼{\tilde{\nu}}(\mu,\omega,\nu,\tau,\lambda,\alpha) given in Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) increases

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ν~​(μ,ω,ν,τ,λ01,α01)~𝜈𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01\displaystyle{\tilde{\nu}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ | >νabsent𝜈\displaystyle>\ \nu |  | (16) |

in both Ω1−superscriptsubscriptΩ1\Omega\_{1}^{-} and Ω2−superscriptsubscriptΩ2\Omega\_{2}^{-}.
All fixed
points (μ,ν)𝜇𝜈(\mu,\nu) of mapping Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) and
Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) ensure for 0.8⩽τ0.8𝜏0.8\leqslant\tau that
ν~>0.16~𝜈0.16{\tilde{\nu}}>0.16
and for 0.9⩽τ0.9𝜏0.9\leqslant\tau that ν~>0.24~𝜈0.24{\tilde{\nu}}>0.24.
Consequently, the variance mapping Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) and
Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) ensures a lower bound on the variance ν𝜈\nu.

## A3 Proofs of the Theorems

### A3.1 Proof of Theorem 1

We have to show that the mapping g𝑔g defined by Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))
and Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))
has a stable and attracting fixed point close to (0,1)01(0,1).
To proof this statement and Theorem [1](#Thmtheorem1 "Theorem 1 (Stable and Attracting Fixed Points). ‣ Stable and Attracting Fixed Points for Unnormalized Weights. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"),
we apply the Banach fixed point theorem which
requires (1) that g𝑔g is a contraction mapping and (2)
that g𝑔g does not map outside the function’s
domain, concretely:

###### Theorem 4 (Banach Fixed Point Theorem).

Let (X,d)𝑋𝑑(X,d) be a non-empty complete metric space with a
contraction mapping f:X→X:𝑓→𝑋𝑋f:X\to X. Then f𝑓f has
a unique fixed-point xf∈Xsubscript𝑥𝑓𝑋x\_{f}\in X with f​(xf)=xf𝑓subscript𝑥𝑓subscript𝑥𝑓f(x\_{f})=x\_{f}.
Every sequence xn=f​(xn−1)subscript𝑥𝑛𝑓subscript𝑥𝑛1x\_{n}=f(x\_{n-1})
with starting element x0∈Xsubscript𝑥0𝑋x\_{0}\in X converges to the fixed point:
xn→n→∞xf→𝑛→subscript𝑥𝑛subscript𝑥𝑓x\_{n}\xrightarrow[n\to\infty]{\ }x\_{f}.

Contraction mappings are functions that map two points such that their distance is decreasing:

###### Definition 2 (Contraction mapping).

A function f:X→X:𝑓→𝑋𝑋f:X\to X on a metric space X𝑋X with distance d𝑑d is a contraction mapping, if there
is a 0⩽δ<10𝛿10\leqslant\delta<1, such that for all points 𝐮𝐮\bm{u} and 𝐯𝐯\bm{v} in X𝑋X:
d​(f​(𝐮),f​(𝐯))⩽δ​d​(𝐮,𝐯)𝑑𝑓𝐮𝑓𝐯𝛿𝑑𝐮𝐯d(f(\bm{u}),f(\bm{v}))\leqslant\delta d(\bm{u},\bm{v}).

To show that g𝑔g is a contraction mapping in ΩΩ\Omega with distance ∥.∥2\|.\|\_{2}, we use the Mean Value
Theorem for u,v∈Ω

𝑢𝑣
Ωu,v\in\Omega

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖g​(𝒖)−g​(𝒗)‖2⩽M​‖𝒖−𝒗‖2,subscriptnorm𝑔𝒖𝑔𝒗2𝑀subscriptnorm𝒖𝒗2\displaystyle\|g(\bm{u})-g(\bm{v})\|\_{2}\leqslant M\ \|\bm{u}-\bm{v}\|\_{2}, |  | (17) |

in which M𝑀M is an upper bound on the spectral norm the Jacobian ℋℋ\mathcal{H} of g𝑔g.
The spectral norm is given by the largest singular value of the Jacobian of g𝑔g.
If the largest singular value of the Jacobian is smaller than 1,
the mapping g𝑔g of the mean and variance to the mean and variance in the next layer is contracting.
We show that the largest singular value is smaller than 1 by
evaluating the function for the singular value
S​(μ,ω,ν,τ,λ,α)𝑆𝜇𝜔𝜈𝜏𝜆𝛼S(\mu,\omega,\nu,\tau,\lambda,\alpha) on a grid.
Then we use the Mean Value Theorem to bound the deviation of the
function S𝑆S between grid points.
To this end, we have to bound the gradient of S𝑆S with respect to
(μ,ω,ν,τ)𝜇𝜔𝜈𝜏(\mu,\omega,\nu,\tau). If all function values plus
gradient times the deltas (differences between grid points and evaluated
points) is still smaller than 1, then we have proofed that the
function is below 1 (Lemma [12](#Thmtheorem12 "Lemma 12 (Largest Singular Value Smaller Than One). ‣ Upper Bounds on the Largest Singular Value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")). To show that the mapping does not map outside the function’s domain, we
derive bounds on the expressions for the mean and the variance (Lemma [13](#Thmtheorem13 "Lemma 13 (Mapping into the domain). ‣ A3.4.2 Lemmata for proofing Theorem 1 (part 2): Mapping within domain ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")).
Section [A3.4.1](#S3.SS4.SSS1 "A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") and Section [A3.4.2](#S3.SS4.SSS2 "A3.4.2 Lemmata for proofing Theorem 1 (part 2): Mapping within domain ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") are concerned with the contraction mapping and
the image of the function domain of g𝑔g, respectively.

With the results that the largest singular value of the Jacobian is smaller than
one (Lemma [12](#Thmtheorem12 "Lemma 12 (Largest Singular Value Smaller Than One). ‣ Upper Bounds on the Largest Singular Value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) and that the mapping stays in the domain ΩΩ\Omega
(Lemma [13](#Thmtheorem13 "Lemma 13 (Mapping into the domain). ‣ A3.4.2 Lemmata for proofing Theorem 1 (part 2): Mapping within domain ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")), we can prove Theorem [1](#Thmtheorem1 "Theorem 1 (Stable and Attracting Fixed Points). ‣ Stable and Attracting Fixed Points for Unnormalized Weights. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks").
We first recall Theorem [1](#Thmtheorem1 "Theorem 1 (Stable and Attracting Fixed Points). ‣ Stable and Attracting Fixed Points for Unnormalized Weights. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"):

###### Theorem (Stable and Attracting Fixed Points).

We assume α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01} and λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01}.
We restrict the range of the variables to the domain
μ∈[−0.1,0.1]𝜇0.10.1\mu\in[-0.1,0.1],
ω∈[−0.1,0.1]𝜔0.10.1\omega\in[-0.1,0.1],
ν∈[0.8,1.5]𝜈0.81.5\nu\in[0.8,1.5], and
τ∈[0.95,1.1]𝜏0.951.1\tau\in[0.95,1.1].
For ω=0𝜔0\omega=0 and τ=1𝜏1\tau=1, the mapping Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))
and Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) has the stable
fixed point (μ,ν)=(0,1)𝜇𝜈01(\mu,\nu)=(0,1).
For other ω𝜔\omega and τ𝜏\tau the mapping Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))
and Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) has a stable and
attracting fixed point depending on (ω,τ)𝜔𝜏(\omega,\tau) in the
(μ,ν)𝜇𝜈(\mu,\nu)-domain: μ∈[−0.03106,0.06773]𝜇0.031060.06773\mu\in[-0.03106,0.06773] and
ν∈[0.80009,1.48617]𝜈0.800091.48617\nu\in[0.80009,1.48617].
All points within the (μ,ν)𝜇𝜈(\mu,\nu)-domain converge when
iteratively applying the mapping Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))
and Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) to this fixed point.

###### Proof.

According to Lemma [12](#Thmtheorem12 "Lemma 12 (Largest Singular Value Smaller Than One). ‣ Upper Bounds on the Largest Singular Value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") the mapping g𝑔g (Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) and Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")))
is a contraction mapping in the given
domain, that is, it has a Lipschitz constant smaller than one.
We showed that (μ,ν)=(0,1)𝜇𝜈01(\mu,\nu)=(0,1) is a fixed point of the
mapping for (ω,τ)=(0,1)𝜔𝜏01(\omega,\tau)=(0,1).

The domain is compact (bounded and closed), therefore it is a
complete metric space.
We further have to make sure the mapping g𝑔g does not map outside its domain ΩΩ\Omega.
According to Lemma [13](#Thmtheorem13 "Lemma 13 (Mapping into the domain). ‣ A3.4.2 Lemmata for proofing Theorem 1 (part 2): Mapping within domain ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), the mapping maps into the domain μ∈[−0.03106,0.06773]𝜇0.031060.06773\mu\in[-0.03106,0.06773] and
ν∈[0.80009,1.48617]𝜈0.800091.48617\nu\in[0.80009,1.48617].

Now we can apply the Banach fixed point theorem
given in Theorem [4](#Thmtheorem4 "Theorem 4 (Banach Fixed Point Theorem). ‣ A3.1 Proof of Theorem 1 ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") from which the statement of the
theorem follows.
∎

### A3.2 Proof of Theorem 2

First we recall Theorem [2](#Thmtheorem2a "Theorem 2 (Decreasing 𝜈). ‣ A2.2 Theorem 2: Decreasing Variance from Above ‣ A2 Theorems ‣ Self-Normalizing Neural Networks"):

###### Theorem (Decreasing ν𝜈\nu).

For λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01}, α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01}
and the domain Ω++superscriptΩabsent\Omega^{++}:
−1⩽μ⩽11𝜇1-1\leqslant\mu\leqslant 1,
−0.1⩽ω⩽0.10.1𝜔0.1-0.1\leqslant\omega\leqslant 0.1,
3⩽ν⩽163𝜈163\leqslant\nu\leqslant 16, and
0.8⩽τ⩽1.250.8𝜏1.250.8\leqslant\tau\leqslant 1.25 we have for
the mapping of the variance
ν~​(μ,ω,ν,τ,λ,α)~𝜈𝜇𝜔𝜈𝜏𝜆𝛼{\tilde{\nu}}(\mu,\omega,\nu,\tau,\lambda,\alpha) given in Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ν~​(μ,ω,ν,τ,λ01,α01)~𝜈𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01\displaystyle{\tilde{\nu}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ | <ν.absent𝜈\displaystyle<\ \nu\ . |  | (18) |

The variance decreases in [3,16]316[3,16] and all fixed
points (μ,ν)𝜇𝜈(\mu,\nu) of mapping Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) and Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) have ν<3𝜈3\nu<3.

###### Proof.

We start to consider an even larger domain
−1⩽μ⩽11𝜇1-1\leqslant\mu\leqslant 1,
−0.1⩽ω⩽0.10.1𝜔0.1-0.1\leqslant\omega\leqslant 0.1,
1.5⩽ν⩽161.5𝜈161.5\leqslant\nu\leqslant 16, and
0.8⩽τ⩽1.250.8𝜏1.250.8\leqslant\tau\leqslant 1.25.
We prove facts for this domain and later restrict to
3⩽ν⩽163𝜈163\leqslant\nu\leqslant 16, i.e. Ω++superscriptΩabsent\Omega^{++}.
We consider the function g𝑔g of the difference between the second moment ξ~~𝜉{\tilde{\xi}} in the next layer
and the variance ν𝜈\nu in the lower layer:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | g​(μ,ω,ν,τ,λ01,α01)𝑔𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01\displaystyle g(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ | =ξ~​(μ,ω,ν,τ,λ01,α01)−ν.absent~𝜉𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01𝜈\displaystyle=\ {\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ -\ \nu\ . |  | (19) |

If we can show that g​(μ,ω,ν,τ,λ01,α01)<0𝑔𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼010g(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})<0 for
all (μ,ω,ν,τ)∈Ω++𝜇𝜔𝜈𝜏superscriptΩabsent(\mu,\omega,\nu,\tau)\in\Omega^{++}, then
we would obtain our desired result ν~⩽ξ~<ν~𝜈~𝜉𝜈{\tilde{\nu}}\leqslant{\tilde{\xi}}<\nu.
The derivative with respect to ν𝜈\nu is according to Theorem [16](#Thmtheorem16 "Theorem 16 (Contraction 𝜈-mapping). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂∂ν​g​(μ,ω,ν,τ,λ01,α01)𝜈𝑔𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01\displaystyle\frac{\partial}{\partial\nu}g(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ | =∂∂ν​ξ~​(μ,ω,ν,τ,λ01,α​01)− 1< 0.absent𝜈~𝜉𝜇𝜔𝜈𝜏subscript𝜆01𝛼011 0\displaystyle=\ \frac{\partial}{\partial\nu}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha{\rm 01})\ -\ 1\ <\ 0\ . |  | (20) |

Therefore g𝑔g is strictly monotonically decreasing in ν𝜈\nu.
Since ξ~~𝜉{\tilde{\xi}} is a function in ν​τ𝜈𝜏\nu\tau
(these variables only appear as this product), we
have for x=ν​τ𝑥𝜈𝜏x=\nu\tau

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂∂ν​ξ~=∂∂x​ξ~​∂x∂ν=∂∂x​ξ~​τ𝜈~𝜉𝑥~𝜉𝑥𝜈𝑥~𝜉𝜏\displaystyle\frac{\partial}{\partial\nu}{\tilde{\xi}}\ =\ \frac{\partial}{\partial x}{\tilde{\xi}}\ \frac{\partial x}{\partial\nu}\ =\ \frac{\partial}{\partial x}{\tilde{\xi}}\ \tau |  | (21) |

and

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂∂τ​ξ~=∂∂x​ξ~​∂x∂τ=∂∂x​ξ~​ν.𝜏~𝜉𝑥~𝜉𝑥𝜏𝑥~𝜉𝜈\displaystyle\frac{\partial}{\partial\tau}{\tilde{\xi}}\ =\ \frac{\partial}{\partial x}{\tilde{\xi}}\ \frac{\partial x}{\partial\tau}\ =\ \frac{\partial}{\partial x}{\tilde{\xi}}\ \nu\ . |  | (22) |

Therefore we have according to Theorem [16](#Thmtheorem16 "Theorem 16 (Contraction 𝜈-mapping). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂∂τ​ξ~​(μ,ω,ν,τ,λ01,α​01)𝜏~𝜉𝜇𝜔𝜈𝜏subscript𝜆01𝛼01\displaystyle\frac{\partial}{\partial\tau}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha{\rm 01})\ | =ντ​∂∂ν​ξ~​(μ,ω,ν,τ,λ01,α​01)> 0.absent𝜈𝜏𝜈~𝜉𝜇𝜔𝜈𝜏subscript𝜆01𝛼01 0\displaystyle=\ \frac{\nu}{\tau}\ \frac{\partial}{\partial\nu}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha{\rm 01})\ >\ 0\ . |  | (23) |

Therefore

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂∂τ​g​(μ,ω,ν,τ,λ01,α​01)𝜏𝑔𝜇𝜔𝜈𝜏subscript𝜆01𝛼01\displaystyle\frac{\partial}{\partial\tau}g(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha{\rm 01})\ | =∂∂τ​ξ~​(μ,ω,ν,τ,λ01,α​01)> 0.absent𝜏~𝜉𝜇𝜔𝜈𝜏subscript𝜆01𝛼01 0\displaystyle=\ \frac{\partial}{\partial\tau}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha{\rm 01})\ >\ 0\ . |  | (24) |

Consequently, g𝑔g is strictly monotonically increasing in τ𝜏\tau.
Now we consider the derivative with respect to μ𝜇\mu and ω𝜔\omega. We start with ∂∂μ​ξ~​(μ,ω,ν,τ,λ,α)𝜇~𝜉𝜇𝜔𝜈𝜏𝜆𝛼\frac{\partial}{\partial\mu}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda,\alpha),
which is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ∂∂μ​ξ~​(μ,ω,ν,τ,λ,α)=𝜇~𝜉𝜇𝜔𝜈𝜏𝜆𝛼absent\displaystyle\frac{\partial}{\partial\mu}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda,\alpha)\ = |  | (25) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | λ2ω(α2(−eμ​ω+ν​τ2)erfc(μ​ω+ν​τ2​ν​τ)+\displaystyle\lambda^{2}\omega\left(\alpha^{2}\left(-e^{\mu\omega+\frac{\nu\tau}{2}}\right)\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | α2e2​μ​ω+2​ν​τerfc(μ​ω+2​ν​τ2​ν​τ)+μω(2−erfc(μ​ω2​ν​τ))+2πν​τe−μ2​ω22​ν​τ).\displaystyle\left.\alpha^{2}e^{2\mu\omega+2\nu\tau}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+\mu\omega\left(2-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)+\sqrt{\frac{2}{\pi}}\sqrt{\nu\tau}e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\right)\ . |  |

We consider the sub-function

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2π​ν​τ−α2​(e(μ​ω+ν​τ2​ν​τ)2​erfc⁡(μ​ω+ν​τ2​ν​τ)−e(μ​ω+2​ν​τ2​ν​τ)2​erfc⁡(μ​ω+2​ν​τ2​ν​τ)).2𝜋𝜈𝜏superscript𝛼2superscript𝑒superscript𝜇𝜔𝜈𝜏2𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏superscript𝑒superscript𝜇𝜔2𝜈𝜏2𝜈𝜏2erfc𝜇𝜔2𝜈𝜏2𝜈𝜏\displaystyle\sqrt{\frac{2}{\pi}}\sqrt{\nu\tau}-\alpha^{2}\left(e^{\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-e^{\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)\ . |  | (26) |

We set x=ν​τ𝑥𝜈𝜏x=\nu\tau and y=μ​ω𝑦𝜇𝜔y=\mu\omega and obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2π​x−α2​(e(x+y2​x)2​erfc⁡(x+y2​x)−e(2​x+y2​x)2​erfc⁡(2​x+y2​x)).2𝜋𝑥superscript𝛼2superscript𝑒superscript𝑥𝑦2𝑥2erfc𝑥𝑦2𝑥superscript𝑒superscript2𝑥𝑦2𝑥2erfc2𝑥𝑦2𝑥\displaystyle\sqrt{\frac{2}{\pi}}\sqrt{x}-\alpha^{2}\left(e^{\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)^{2}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)-e^{\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)^{2}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)\right)\ . |  | (27) |

The derivative to this sub-function with respect to y𝑦y is

|  |  |  |  |
| --- | --- | --- | --- |
|  | α2​(e(2​x+y)22​x​(2​x+y)​erfc⁡(2​x+y2​x)−e(x+y)22​x​(x+y)​erfc⁡(x+y2​x))x=superscript𝛼2superscript𝑒superscript2𝑥𝑦22𝑥2𝑥𝑦erfc2𝑥𝑦2𝑥superscript𝑒superscript𝑥𝑦22𝑥𝑥𝑦erfc𝑥𝑦2𝑥𝑥absent\displaystyle\frac{\alpha^{2}\left(e^{\frac{(2x+y)^{2}}{2x}}(2x+y)\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)-e^{\frac{(x+y)^{2}}{2x}}(x+y)\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)\right)}{x}\ = |  | (28) |
|  |  |  |
| --- | --- | --- |
|  | 2​α2​x​(e(2​x+y)22​x​(2​x+y)​erfc⁡(2​x+y2​x)2​x−e(x+y)22​x​(x+y)​erfc⁡(x+y2​x)2​x)x> 0.2superscript𝛼2𝑥superscript𝑒superscript2𝑥𝑦22𝑥2𝑥𝑦erfc2𝑥𝑦2𝑥2𝑥superscript𝑒superscript𝑥𝑦22𝑥𝑥𝑦erfc𝑥𝑦2𝑥2𝑥𝑥 0\displaystyle\frac{\sqrt{2}\alpha^{2}\sqrt{x}\left(\frac{e^{\frac{(2x+y)^{2}}{2x}}(2x+y)\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)}{\sqrt{2}\sqrt{x}}-\frac{e^{\frac{(x+y)^{2}}{2x}}(x+y)\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)}{\sqrt{2}\sqrt{x}}\right)}{x}\ >\ 0\ . |  |

The inequality follows from Lemma [24](#Thmtheorem24 "Lemma 24 (Properties of 𝑥⁢𝑒^𝑥²⁢erfc(𝑥)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), which states that
z​ez2​erfc⁡(z)𝑧superscript𝑒superscript𝑧2erfc𝑧ze^{z^{2}}\operatorname{erfc}(z) is monotonically increasing in z𝑧z.
Therefore the sub-function is increasing in y𝑦y. The derivative to this sub-function with respect to x𝑥x is

|  |  |  |  |
| --- | --- | --- | --- |
|  | 12​π​x2πα2(e(2​x+y)22​x(4x2−y2)erfc(2​x+y2​x)\displaystyle\frac{1}{2\sqrt{\pi}x^{2}}\sqrt{\pi}\alpha^{2}\left(e^{\frac{(2x+y)^{2}}{2x}}\left(4x^{2}-y^{2}\right)\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)\right. |  | (29) |
|  |  |  |
| --- | --- | --- |
|  | −e(x+y)22​x(x−y)(x+y)erfc(x+y2​x))−2(α2−1)x3/2.\displaystyle\left.-e^{\frac{(x+y)^{2}}{2x}}(x-y)(x+y)\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)\right)-\sqrt{2}\left(\alpha^{2}-1\right)x^{3/2}. |  |

The sub-function is increasing in x𝑥x, since the
derivative is larger than zero:

|  |  |  |  |
| --- | --- | --- | --- |
|  | π​α2​(e(2​x+y)22​x​(4​x2−y2)​erfc⁡(2​x+y2​x)−e(x+y)22​x​(x−y)​(x+y)​erfc⁡(x+y2​x))−2​x3/2​(α2−1)2​π​x2⩾𝜋superscript𝛼2superscript𝑒superscript2𝑥𝑦22𝑥4superscript𝑥2superscript𝑦2erfc2𝑥𝑦2𝑥superscript𝑒superscript𝑥𝑦22𝑥𝑥𝑦𝑥𝑦erfc𝑥𝑦2𝑥2superscript𝑥32superscript𝛼212𝜋superscript𝑥2absent\displaystyle\frac{\sqrt{\pi}\alpha^{2}\left(e^{\frac{(2x+y)^{2}}{2x}}\left(4x^{2}-y^{2}\right)\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)-e^{\frac{(x+y)^{2}}{2x}}(x-y)(x+y)\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)\right)-\sqrt{2}x^{3/2}\left(\alpha^{2}-1\right)}{2\sqrt{\pi}x^{2}}\ \geqslant |  | (30) |
|  |  |  |
| --- | --- | --- |
|  | π​α2​((2​x−y)​(2​x+y)​2π​(2​x+y2​x+(2​x+y2​x)2+2)−(x−y)​(x+y)​2π​(x+y2​x+(x+y2​x)2+4π))−2​x3/2​(α2−1)2​π​x2=𝜋superscript𝛼22𝑥𝑦2𝑥𝑦2𝜋2𝑥𝑦2𝑥superscript2𝑥𝑦2𝑥22𝑥𝑦𝑥𝑦2𝜋𝑥𝑦2𝑥superscript𝑥𝑦2𝑥24𝜋2superscript𝑥32superscript𝛼212𝜋superscript𝑥2absent\displaystyle\frac{\sqrt{\pi}\alpha^{2}\left(\frac{(2x-y)(2x+y)2}{\sqrt{\pi}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}+\sqrt{\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)^{2}+2}\right)}-\frac{(x-y)(x+y)2}{\sqrt{\pi}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}+\sqrt{\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)^{2}+\frac{4}{\pi}}\right)}\right)-\sqrt{2}x^{3/2}\left(\alpha^{2}-1\right)}{2\sqrt{\pi}x^{2}}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | π​α2​((2​x−y)​(2​x+y)​2​(2​x)π​(2​x+y+(2​x+y)2+4​x)−(x−y)​(x+y)​2​(2​x)π​(x+y+(x+y)2+8​xπ))−2​x3/2​(α2−1)2​π​x2=𝜋superscript𝛼22𝑥𝑦2𝑥𝑦22𝑥𝜋2𝑥𝑦superscript2𝑥𝑦24𝑥𝑥𝑦𝑥𝑦22𝑥𝜋𝑥𝑦superscript𝑥𝑦28𝑥𝜋2superscript𝑥32superscript𝛼212𝜋superscript𝑥2absent\displaystyle\frac{\sqrt{\pi}\alpha^{2}\left(\frac{(2x-y)(2x+y)2\left(\sqrt{2}\sqrt{x}\right)}{\sqrt{\pi}\left(2x+y+\sqrt{(2x+y)^{2}+4x}\right)}-\frac{(x-y)(x+y)2\left(\sqrt{2}\sqrt{x}\right)}{\sqrt{\pi}\left(x+y+\sqrt{(x+y)^{2}+\frac{8x}{\pi}}\right)}\right)-\sqrt{2}x^{3/2}\left(\alpha^{2}-1\right)}{2\sqrt{\pi}x^{2}}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | π​α2​((2​x−y)​(2​x+y)​2π​(2​x+y+(2​x+y)2+4​x)−(x−y)​(x+y)​2π​(x+y+(x+y)2+8​xπ))−x​(α2−1)2​π​x3/2>𝜋superscript𝛼22𝑥𝑦2𝑥𝑦2𝜋2𝑥𝑦superscript2𝑥𝑦24𝑥𝑥𝑦𝑥𝑦2𝜋𝑥𝑦superscript𝑥𝑦28𝑥𝜋𝑥superscript𝛼212𝜋superscript𝑥32absent\displaystyle\frac{\sqrt{\pi}\alpha^{2}\left(\frac{(2x-y)(2x+y)2}{\sqrt{\pi}\left(2x+y+\sqrt{(2x+y)^{2}+4x}\right)}-\frac{(x-y)(x+y)2}{\sqrt{\pi}\left(x+y+\sqrt{(x+y)^{2}+\frac{8x}{\pi}}\right)}\right)-x\left(\alpha^{2}-1\right)}{\sqrt{2}\sqrt{\pi}x^{3/2}}\ > |  |
|  |  |  |
| --- | --- | --- |
|  | π​α2​((2​x−y)​(2​x+y)​2π​(2​x+y+(2​x+y)2+2​(2​x+y)+1)−(x−y)​(x+y)​2π​(x+y+(x+y)2+0.878⋅2​(x+y)+0.8782))−x​(α2−1)2​π​x3/2=𝜋superscript𝛼22𝑥𝑦2𝑥𝑦2𝜋2𝑥𝑦superscript2𝑥𝑦222𝑥𝑦1𝑥𝑦𝑥𝑦2𝜋𝑥𝑦superscript𝑥𝑦2⋅0.8782𝑥𝑦superscript0.8782𝑥superscript𝛼212𝜋superscript𝑥32absent\displaystyle\frac{\sqrt{\pi}\alpha^{2}\left(\frac{(2x-y)(2x+y)2}{\sqrt{\pi}\left(2x+y+\sqrt{(2x+y)^{2}+2(2x+y)+1}\right)}-\frac{(x-y)(x+y)2}{\sqrt{\pi}\left(x+y+\sqrt{(x+y)^{2}+0.878\cdot 2(x+y)+0.878^{2}}\right)}\right)-x\left(\alpha^{2}-1\right)}{\sqrt{2}\sqrt{\pi}x^{3/2}}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | π​α2​((2​x−y)​(2​x+y)​2π​(2​x+y+(2​x+y+1)2)−(x−y)​(x+y)​2π​(x+y+(x+y+0.878)2))−x​(α2−1)2​π​x3/2=𝜋superscript𝛼22𝑥𝑦2𝑥𝑦2𝜋2𝑥𝑦superscript2𝑥𝑦12𝑥𝑦𝑥𝑦2𝜋𝑥𝑦superscript𝑥𝑦0.8782𝑥superscript𝛼212𝜋superscript𝑥32absent\displaystyle\frac{\sqrt{\pi}\alpha^{2}\left(\frac{(2x-y)(2x+y)2}{\sqrt{\pi}\left(2x+y+\sqrt{(2x+y+1)^{2}}\right)}-\frac{(x-y)(x+y)2}{\sqrt{\pi}\left(x+y+\sqrt{(x+y+0.878)^{2}}\right)}\right)-x\left(\alpha^{2}-1\right)}{\sqrt{2}\sqrt{\pi}x^{3/2}}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | π​α2​((2​x−y)​(2​x+y)​2π​(2​(2​x+y)+1)−(x−y)​(x+y)​2π​(2​(x+y)+0.878))−x​(α2−1)2​π​x3/2=𝜋superscript𝛼22𝑥𝑦2𝑥𝑦2𝜋22𝑥𝑦1𝑥𝑦𝑥𝑦2𝜋2𝑥𝑦0.878𝑥superscript𝛼212𝜋superscript𝑥32absent\displaystyle\frac{\sqrt{\pi}\alpha^{2}\left(\frac{(2x-y)(2x+y)2}{\sqrt{\pi}(2(2x+y)+1)}-\frac{(x-y)(x+y)2}{\sqrt{\pi}(2(x+y)+0.878)}\right)-x\left(\alpha^{2}-1\right)}{\sqrt{2}\sqrt{\pi}x^{3/2}}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | π​α2​((2​(x+y)+0.878)​(2​x−y)​(2​x+y)​2π−(x−y)​(x+y)​(2​(2​x+y)+1)​2π)(2​(2​x+y)+1)​(2​(x+y)+0.878)​2​π​x3/2+limit-from𝜋superscript𝛼22𝑥𝑦0.8782𝑥𝑦2𝑥𝑦2𝜋𝑥𝑦𝑥𝑦22𝑥𝑦12𝜋22𝑥𝑦12𝑥𝑦0.8782𝜋superscript𝑥32\displaystyle\frac{\sqrt{\pi}\alpha^{2}\left(\frac{(2(x+y)+0.878)(2x-y)(2x+y)2}{\sqrt{\pi}}-\frac{(x-y)(x+y)(2(2x+y)+1)2}{\sqrt{\pi}}\right)}{(2(2x+y)+1)(2(x+y)+0.878)\sqrt{2}\sqrt{\pi}x^{3/2}}\ + |  |
|  |  |  |
| --- | --- | --- |
|  | π​α2​(−x​(α2−1)​(2​(2​x+y)+1)​(2​(x+y)+0.878))(2​(2​x+y)+1)​(2​(x+y)+0.878)​2​π​x3/2=𝜋superscript𝛼2𝑥superscript𝛼2122𝑥𝑦12𝑥𝑦0.87822𝑥𝑦12𝑥𝑦0.8782𝜋superscript𝑥32absent\displaystyle\frac{\sqrt{\pi}\alpha^{2}\left(-x\left(\alpha^{2}-1\right)(2(2x+y)+1)(2(x+y)+0.878)\right)}{(2(2x+y)+1)(2(x+y)+0.878)\sqrt{2}\sqrt{\pi}x^{3/2}}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | 8​x3+12​x2​y+4.14569​x2+4​x​y2−6.76009​x​y−1.58023​x+0.683154​y2(2​(2​x+y)+1)​(2​(x+y)+0.878)​2​π​x3/2>8superscript𝑥312superscript𝑥2𝑦4.14569superscript𝑥24𝑥superscript𝑦26.76009𝑥𝑦1.58023𝑥0.683154superscript𝑦222𝑥𝑦12𝑥𝑦0.8782𝜋superscript𝑥32absent\displaystyle\frac{8x^{3}+12x^{2}y+4.14569x^{2}+4xy^{2}-6.76009xy-1.58023x+0.683154y^{2}}{(2(2x+y)+1)(2(x+y)+0.878)\sqrt{2}\sqrt{\pi}x^{3/2}}\ > |  |
|  |  |  |
| --- | --- | --- |
|  | 8​x3−0.1⋅12​x2+4.14569​x2+4⋅(0.0)2​x−6.76009⋅0.1​x−1.58023​x+0.683154⋅(0.0)2(2​(2​x+y)+1)​(2​(x+y)+0.878)​2​π​x3/2=8superscript𝑥3⋅0.112superscript𝑥24.14569superscript𝑥2⋅4superscript0.02𝑥⋅6.760090.1𝑥1.58023𝑥⋅0.683154superscript0.0222𝑥𝑦12𝑥𝑦0.8782𝜋superscript𝑥32absent\displaystyle\frac{8x^{3}-0.1\cdot 12x^{2}+4.14569x^{2}+4\cdot(0.0)^{2}x-6.76009\cdot 0.1x-1.58023x+0.683154\cdot(0.0)^{2}}{(2(2x+y)+1)(2(x+y)+0.878)\sqrt{2}\sqrt{\pi}x^{3/2}}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | 8​x2+2.94569​x−2.25624(2​(2​x+y)+1)​(2​(x+y)+0.878)​2​π​x=8superscript𝑥22.94569𝑥2.2562422𝑥𝑦12𝑥𝑦0.8782𝜋𝑥absent\displaystyle\frac{8x^{2}+2.94569x-2.25624}{(2(2x+y)+1)(2(x+y)+0.878)\sqrt{2}\sqrt{\pi}\sqrt{x}}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | 8​(x−0.377966)​(x+0.746178)(2​(2​x+y)+1)​(2​(x+y)+0.878)​2​π​x> 0.8𝑥0.377966𝑥0.74617822𝑥𝑦12𝑥𝑦0.8782𝜋𝑥 0\displaystyle\frac{8(x-0.377966)(x+0.746178)}{(2(2x+y)+1)(2(x+y)+0.878)\sqrt{2}\sqrt{\pi}\sqrt{x}}\ >\ 0\ . |  |

We explain this chain of inequalities:

* •

  First inequality: We applied Lemma [22](#Thmtheorem22 "Lemma 22 (Erfc bound from Abramowitz). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") two times.
* •

  Equalities factor out 2​x2𝑥\sqrt{2}\sqrt{x} and reformulate.
* •

  Second inequality part 1: we applied

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 0<2​y⟹(2​x+y)2+4​x+1<(2​x+y)2+2​(2​x+y)+1=(2​x+y+1)2.02𝑦⟹superscript2𝑥𝑦24𝑥1superscript2𝑥𝑦222𝑥𝑦1superscript2𝑥𝑦12\displaystyle 0<2y\Longrightarrow(2x+y)^{2}+4x+1<(2x+y)^{2}+2(2x+y)+1=(2x+y+1)^{2}\ . |  | (31) |
* •

  Second inequality part 2: we show that for a=110​(960+169​ππ−13)𝑎110960169𝜋𝜋13a=\frac{1}{10}\left(\sqrt{\frac{960+169\pi}{\pi}}-13\right) following holds:
  8​xπ−(a2+2​a​(x+y))⩾08𝑥𝜋superscript𝑎22𝑎𝑥𝑦0\frac{8x}{\pi}-\left(a^{2}+2a(x+y)\right)\geqslant 0.
  We have ∂∂x​8​xπ−(a2+2​a​(x+y))=8π−2​a>0𝑥8𝑥𝜋superscript𝑎22𝑎𝑥𝑦8𝜋2𝑎0\frac{\partial}{\partial x}\frac{8x}{\pi}-\left(a^{2}+2a(x+y)\right)=\frac{8}{\pi}-2a>0 and
  ∂∂y​8​xπ−(a2+2​a​(x+y))=−2​a<0𝑦8𝑥𝜋superscript𝑎22𝑎𝑥𝑦2𝑎0\frac{\partial}{\partial y}\frac{8x}{\pi}-\left(a^{2}+2a(x+y)\right)=-2a<0.
  Therefore the minimum is at border for minimal x𝑥x and maximal y𝑦y:

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 8⋅1.2π−(210​(960+169​ππ−13)​(1.2+0.1)+(110​(960+169​ππ−13))2)= 0.⋅81.2𝜋210960169𝜋𝜋131.20.1superscript110960169𝜋𝜋132 0\displaystyle\frac{8\cdot 1.2}{\pi}-\left(\frac{2}{10}\left(\sqrt{\frac{960+169\pi}{\pi}}-13\right)(1.2+0.1)+\left(\frac{1}{10}\left(\sqrt{\frac{960+169\pi}{\pi}}-13\right)\right)^{2}\right)\ =\ 0\ . |  | (32) |

  Thus

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 8​xπ⩾a2+2​a​(x+y).8𝑥𝜋superscript𝑎22𝑎𝑥𝑦\displaystyle\frac{8x}{\pi}\ \geqslant\ a^{2}+2a(x+y)\ . |  | (33) |

  for a=110​(960+169​ππ−13)>0.878𝑎110960169𝜋𝜋130.878a=\frac{1}{10}\left(\sqrt{\frac{960+169\pi}{\pi}}-13\right)>0.878.
* •

  Equalities only solve square root and factor out the resulting
  terms (2​(2​x+y)+1)22𝑥𝑦1(2(2x+y)+1) and (2​(x+y)+0.878)2𝑥𝑦0.878(2(x+y)+0.878).
* •

  We set α=α01𝛼subscript𝛼01\alpha=\alpha\_{01} and multiplied out. Thereafter we
  also factored out x𝑥x in the numerator. Finally a quadratic
  equations was solved.

The sub-function has its minimal value for
minimal
x=ν​τ=1.5⋅0.8=1.2𝑥𝜈𝜏⋅1.50.81.2x=\nu\tau=1.5\cdot 0.8=1.2 and minimal
y=μ​ω=−1⋅0.1=−0.1𝑦𝜇𝜔⋅10.10.1y=\mu\omega=-1\cdot 0.1=-0.1.
We further minimize the function

|  |  |  |  |
| --- | --- | --- | --- |
|  | μ​ω​eμ2​ω22​ν​τ​(2−erfc⁡(μ​ω2​ν​τ))>−0.1​e0.122⋅1.2​(2−erfc⁡(0.12​1.2)).𝜇𝜔superscript𝑒superscript𝜇2superscript𝜔22𝜈𝜏2erfc𝜇𝜔2𝜈𝜏0.1superscript𝑒superscript0.12⋅21.22erfc0.121.2\displaystyle\mu\omega e^{\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\left(2-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)\ >\ -0.1e^{\frac{0.1^{2}}{2\cdot 1.2}}\left(2-\operatorname{erfc}\left(\frac{0.1}{\sqrt{2}\sqrt{1.2}}\right)\right)\ . |  | (34) |

We compute the minimum of the term in brackets of ∂∂μ​ξ~​(μ,ω,ν,τ,λ,α)𝜇~𝜉𝜇𝜔𝜈𝜏𝜆𝛼\frac{\partial}{\partial\mu}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda,\alpha)
in Eq. ([25](#S3.E25 "In Proof. ‣ A3.2 Proof of Theorem 2 ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | μ​ω​eμ2​ω22​ν​τ​(2−erfc⁡(μ​ω2​ν​τ))+limit-from𝜇𝜔superscript𝑒superscript𝜇2superscript𝜔22𝜈𝜏2erfc𝜇𝜔2𝜈𝜏\displaystyle\mu\omega e^{\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\left(2-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)+ |  | (35) |
|  |  |  |
| --- | --- | --- |
|  | α012​(−(e(μ​ω+ν​τ2​ν​τ)2​erfc⁡(μ​ω+ν​τ2​ν​τ)−e(μ​ω+2​ν​τ2​ν​τ)2​erfc⁡(μ​ω+2​ν​τ2​ν​τ)))+2π​ν​τ>superscriptsubscript𝛼012superscript𝑒superscript𝜇𝜔𝜈𝜏2𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏superscript𝑒superscript𝜇𝜔2𝜈𝜏2𝜈𝜏2erfc𝜇𝜔2𝜈𝜏2𝜈𝜏2𝜋𝜈𝜏absent\displaystyle\alpha\_{\rm 01}^{2}\left(-\left(e^{\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-e^{\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)\right)+\sqrt{\frac{2}{\pi}}\sqrt{\nu\tau}\ > |  |
|  |  |  |
| --- | --- | --- |
|  | α012​(−(e(1.2−0.12​1.2)2​erfc⁡(1.2−0.12​1.2)−e(2⋅1.2−0.12​1.2)2​erfc⁡(2⋅1.2−0.12​1.2)))−limit-fromsuperscriptsubscript𝛼012superscript𝑒superscript1.20.121.22erfc1.20.121.2superscript𝑒superscript⋅21.20.121.22erfc⋅21.20.121.2\displaystyle\alpha\_{\rm 01}^{2}\left(-\left(e^{\left(\frac{1.2-0.1}{\sqrt{2}\sqrt{1.2}}\right)^{2}}\operatorname{erfc}\left(\frac{1.2-0.1}{\sqrt{2}\sqrt{1.2}}\right)-e^{\left(\frac{2\cdot 1.2-0.1}{\sqrt{2}\sqrt{1.2}}\right)^{2}}\operatorname{erfc}\left(\frac{2\cdot 1.2-0.1}{\sqrt{2}\sqrt{1.2}}\right)\right)\right)- |  |
|  |  |  |
| --- | --- | --- |
|  | 0.1​e0.122⋅1.2​(2−erfc⁡(0.12​1.2))+1.2​2π= 0.212234.0.1superscript𝑒superscript0.12⋅21.22erfc0.121.21.22𝜋0.212234\displaystyle 0.1e^{\frac{0.1^{2}}{2\cdot 1.2}}\left(2-\operatorname{erfc}\left(\frac{0.1}{\sqrt{2}\sqrt{1.2}}\right)\right)+\sqrt{1.2}\sqrt{\frac{2}{\pi}}\ =\ 0.212234\ . |  |

Therefore the term in brackets of Eq. ([25](#S3.E25 "In Proof. ‣ A3.2 Proof of Theorem 2 ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"))
is larger than zero.
Thus, ∂∂μ​ξ~​(μ,ω,ν,τ,λ,α)𝜇~𝜉𝜇𝜔𝜈𝜏𝜆𝛼\frac{\partial}{\partial\mu}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda,\alpha)
has the sign of ω𝜔\omega.
Since ξ~~𝜉{\tilde{\xi}} is a function in μ​ω𝜇𝜔\mu\omega
(these variables only appear as this product), we
have for x=μ​ω𝑥𝜇𝜔x=\mu\omega

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂∂ν​ξ~=∂∂x​ξ~​∂x∂μ=∂∂x​ξ~​ω𝜈~𝜉𝑥~𝜉𝑥𝜇𝑥~𝜉𝜔\displaystyle\frac{\partial}{\partial\nu}{\tilde{\xi}}\ =\ \frac{\partial}{\partial x}{\tilde{\xi}}\ \frac{\partial x}{\partial\mu}\ =\ \frac{\partial}{\partial x}{\tilde{\xi}}\ \omega |  | (36) |

and

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂∂ω​ξ~=∂∂x​ξ~​∂x∂ω=∂∂x​ξ~​μ.𝜔~𝜉𝑥~𝜉𝑥𝜔𝑥~𝜉𝜇\displaystyle\frac{\partial}{\partial\omega}{\tilde{\xi}}\ =\ \frac{\partial}{\partial x}{\tilde{\xi}}\ \frac{\partial x}{\partial\omega}\ =\ \frac{\partial}{\partial x}{\tilde{\xi}}\ \mu\ . |  | (37) |

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂∂ω​ξ~​(μ,ω,ν,τ,λ01,α​01)𝜔~𝜉𝜇𝜔𝜈𝜏subscript𝜆01𝛼01\displaystyle\frac{\partial}{\partial\omega}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha{\rm 01})\ | =μω​∂∂μ​ξ~​(μ,ω,ν,τ,λ01,α​01).absent𝜇𝜔𝜇~𝜉𝜇𝜔𝜈𝜏subscript𝜆01𝛼01\displaystyle=\ \frac{\mu}{\omega}\ \frac{\partial}{\partial\mu}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha{\rm 01})\ . |  | (38) |

Since ∂∂μ​ξ~𝜇~𝜉\frac{\partial}{\partial\mu}{\tilde{\xi}} has the sign of ω𝜔\omega,
∂∂μ​ξ~𝜇~𝜉\frac{\partial}{\partial\mu}{\tilde{\xi}} has the sign of μ𝜇\mu.
Therefore

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂∂ω​g​(μ,ω,ν,τ,λ01,α​01)𝜔𝑔𝜇𝜔𝜈𝜏subscript𝜆01𝛼01\displaystyle\frac{\partial}{\partial\omega}g(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha{\rm 01})\ | =∂∂ω​ξ~​(μ,ω,ν,τ,λ01,α​01)absent𝜔~𝜉𝜇𝜔𝜈𝜏subscript𝜆01𝛼01\displaystyle=\ \frac{\partial}{\partial\omega}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha{\rm 01}) |  | (39) |

has the sign of μ𝜇\mu.

We now divide the μ𝜇\mu-domain into
−1⩽μ⩽01𝜇0-1\leqslant\mu\leqslant 0 and 0⩽μ⩽10𝜇10\leqslant\mu\leqslant 1.
Analogously we divide the ω𝜔\omega-domain into
−0.1⩽ω⩽00.1𝜔0-0.1\leqslant\omega\leqslant 0 and 0⩽ω⩽0.10𝜔0.10\leqslant\omega\leqslant 0.1.
In this domains g𝑔g is strictly monotonically.

For all domains
g𝑔g is strictly monotonically decreasing in ν𝜈\nu
and strictly monotonically increasing in τ𝜏\tau.
Note that we now consider the range 3⩽ν⩽163𝜈163\leqslant\nu\leqslant 16.
For the maximal value of g𝑔g we set ν=3𝜈3\nu=3 (we set it to 3!)
and τ=1.25𝜏1.25\tau=1.25.

We consider now all combination of these domains:

* •

  −1⩽μ⩽01𝜇0-1\leqslant\mu\leqslant 0 and −0.1⩽ω⩽00.1𝜔0-0.1\leqslant\omega\leqslant 0:

  g𝑔g is decreasing in μ𝜇\mu and decreasing in ω𝜔\omega.
  We set μ=−1𝜇1\mu=-1 and ω=−0.1𝜔0.1\omega=-0.1.

  |  |  |  |  |  |
  | --- | --- | --- | --- | --- |
  |  | g​(−1,−0.1,3,1.25,λ01,α01)𝑔10.131.25subscript𝜆01subscript𝛼01\displaystyle g(-1,-0.1,3,1.25,\lambda\_{\rm 01},\alpha\_{\rm 01})\ | =−0.0180173.absent0.0180173\displaystyle=\ -0.0180173\ . |  | (40) |
* •

  −1⩽μ⩽01𝜇0-1\leqslant\mu\leqslant 0 and 0⩽ω⩽0.10𝜔0.10\leqslant\omega\leqslant 0.1:

  g𝑔g is increasing in μ𝜇\mu and decreasing in ω𝜔\omega.
  We set μ=0𝜇0\mu=0 and ω=0𝜔0\omega=0.

  |  |  |  |  |  |
  | --- | --- | --- | --- | --- |
  |  | g​(0,0,3,1.25,λ01,α01)𝑔0031.25subscript𝜆01subscript𝛼01\displaystyle g(0,0,3,1.25,\lambda\_{\rm 01},\alpha\_{\rm 01})\ | =−0.148532.absent0.148532\displaystyle=\ -0.148532\ . |  | (41) |
* •

  0⩽μ⩽10𝜇10\leqslant\mu\leqslant 1 and −0.1⩽ω⩽00.1𝜔0-0.1\leqslant\omega\leqslant 0:

  g𝑔g is decreasing in μ𝜇\mu and increasing in ω𝜔\omega.
  We set μ=0𝜇0\mu=0 and ω=0𝜔0\omega=0.

  |  |  |  |  |  |
  | --- | --- | --- | --- | --- |
  |  | g​(0,0,3,1.25,λ01,α01)𝑔0031.25subscript𝜆01subscript𝛼01\displaystyle g(0,0,3,1.25,\lambda\_{\rm 01},\alpha\_{\rm 01})\ | =−0.148532.absent0.148532\displaystyle=\ -0.148532\ . |  | (42) |
* •

  0⩽μ⩽10𝜇10\leqslant\mu\leqslant 1 and 0⩽ω⩽0.10𝜔0.10\leqslant\omega\leqslant 0.1:

  g𝑔g is increasing in μ𝜇\mu and increasing in ω𝜔\omega.
  We set μ=1𝜇1\mu=1 and ω=0.1𝜔0.1\omega=0.1.

  |  |  |  |  |  |
  | --- | --- | --- | --- | --- |
  |  | g​(1,0.1,3,1.25,λ01,α01)𝑔10.131.25subscript𝜆01subscript𝛼01\displaystyle g(1,0.1,3,1.25,\lambda\_{\rm 01},\alpha\_{\rm 01})\ | =−0.0180173.absent0.0180173\displaystyle=\ -0.0180173\ . |  | (43) |

  Therefore the maximal value of g𝑔g is −0.01801730.0180173-0.0180173.

∎

### A3.3 Proof of Theorem 3

First we recall Theorem [3](#Thmtheorem3 "Theorem 3 (Increasing 𝜈). ‣ Stable and Attracting Fixed Points for Unnormalized Weights. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"):

###### Theorem (Increasing ν𝜈\nu).

We consider λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01}, α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01}
and the two domains
Ω1−={(μ,ω,ν,τ)|−0.1⩽μ⩽0.1,−0.1⩽ω⩽0.1,0.05⩽ν⩽0.16,0.8⩽τ⩽1.25}superscriptsubscriptΩ1conditional-set𝜇𝜔𝜈𝜏formulae-sequence0.1𝜇0.10.1𝜔0.10.05𝜈0.160.8𝜏1.25\Omega\_{1}^{-}=\{(\mu,\omega,\nu,\tau)\ |\ -0.1\leqslant\mu\leqslant 0.1,-0.1\leqslant\omega\leqslant 0.1,0.05\leqslant\nu\leqslant 0.16,0.8\leqslant\tau\leqslant 1.25\}
and
Ω2−={(μ,ω,ν,τ)|−0.1⩽μ⩽0.1,−0.1⩽ω⩽0.1,0.05⩽ν⩽0.24,0.9⩽τ⩽1.25}superscriptsubscriptΩ2conditional-set𝜇𝜔𝜈𝜏formulae-sequence0.1𝜇0.10.1𝜔0.10.05𝜈0.240.9𝜏1.25\Omega\_{2}^{-}=\{(\mu,\omega,\nu,\tau)\ |\ -0.1\leqslant\mu\leqslant 0.1,-0.1\leqslant\omega\leqslant 0.1,0.05\leqslant\nu\leqslant 0.24,0.9\leqslant\tau\leqslant 1.25\} .

The mapping of the variance
ν~​(μ,ω,ν,τ,λ,α)~𝜈𝜇𝜔𝜈𝜏𝜆𝛼{\tilde{\nu}}(\mu,\omega,\nu,\tau,\lambda,\alpha) given in Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) increases

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ν~​(μ,ω,ν,τ,λ01,α01)~𝜈𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01\displaystyle{\tilde{\nu}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ | >νabsent𝜈\displaystyle>\ \nu |  | (44) |

in both Ω1−superscriptsubscriptΩ1\Omega\_{1}^{-} and Ω2−superscriptsubscriptΩ2\Omega\_{2}^{-}.
All fixed
points (μ,ν)𝜇𝜈(\mu,\nu) of mapping Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) and
Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) ensure for 0.8⩽τ0.8𝜏0.8\leqslant\tau that
ν~>0.16~𝜈0.16{\tilde{\nu}}>0.16
and for 0.9⩽τ0.9𝜏0.9\leqslant\tau that ν~>0.24~𝜈0.24{\tilde{\nu}}>0.24.
Consequently, the variance mapping Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) and
Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) ensures a lower bound on the variance ν𝜈\nu.

###### Proof.

The mean value theorem states that there exists a t∈[0,1]𝑡01t\in[0,1] for which

|  |  |  |  |
| --- | --- | --- | --- |
|  | ξ~​(μ,ω,ν,τ,λ01,α01)−ξ~​(μ,ω,νmin,τ,λ01,α01)=~𝜉𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01~𝜉𝜇𝜔subscript𝜈min𝜏subscript𝜆01subscript𝛼01absent\displaystyle{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ -\ {\tilde{\xi}}(\mu,\omega,\nu\_{\mathrm{min}},\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ = |  | (45) |
|  |  |  |
| --- | --- | --- |
|  | ∂∂ν​ξ~​(μ,ω,ν+t​(νmin−ν),τ,λ01,α01)​(ν−νmin).𝜈~𝜉𝜇𝜔𝜈𝑡subscript𝜈min𝜈𝜏subscript𝜆01subscript𝛼01𝜈subscript𝜈min\displaystyle\frac{\partial}{\partial\nu}{\tilde{\xi}}(\mu,\omega,\nu+t(\nu\_{\mathrm{min}}-\nu),\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ (\nu-\nu\_{\mathrm{min}})\ . |  |

Therefore

|  |  |  |  |
| --- | --- | --- | --- |
|  | ξ~​(μ,ω,ν,τ,λ01,α01)=ξ~​(μ,ω,νmin,τ,λ01,α01)+~𝜉𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01limit-from~𝜉𝜇𝜔subscript𝜈min𝜏subscript𝜆01subscript𝛼01\displaystyle{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ =\ {\tilde{\xi}}(\mu,\omega,\nu\_{\mathrm{min}},\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ + |  | (46) |
|  |  |  |
| --- | --- | --- |
|  | ∂∂ν​ξ~​(μ,ω,ν+t​(νmin−ν),τ,λ01,α01)​(ν−νmin).𝜈~𝜉𝜇𝜔𝜈𝑡subscript𝜈min𝜈𝜏subscript𝜆01subscript𝛼01𝜈subscript𝜈min\displaystyle\frac{\partial}{\partial\nu}{\tilde{\xi}}(\mu,\omega,\nu+t(\nu\_{\mathrm{min}}-\nu),\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ (\nu-\nu\_{\mathrm{min}})\ . |  |

Therefore we are interested to bound the derivative of the ξ𝜉\xi-mapping
Eq. ([13](#S1.E13 "In A1 Background ‣ Self-Normalizing Neural Networks")) with respect to ν𝜈\nu:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ∂∂ν​ξ~​(μ,ω,ν,τ,λ01,α01)=𝜈~𝜉𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01absent\displaystyle\frac{\partial}{\partial\nu}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ = |  | (47) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 12λ2τe−μ2​ω22​ν​τ(α2(−(e(μ​ω+ν​τ2​ν​τ)2erfc(μ​ω+ν​τ2​ν​τ)−2e(μ​ω+2​ν​τ2​ν​τ)2erfc(μ​ω+2​ν​τ2​ν​τ)))−\displaystyle\frac{1}{2}\lambda^{2}\tau e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\left(\alpha^{2}\left(-\left(e^{\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-2e^{\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)\right)-\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | erfc(μ​ω2​ν​τ)+2).\displaystyle\left.\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+2\right)\ . |  |

The sub-term Eq. ([322](#S3.E322 "In Proof. ‣ Behavior of the main subfunction with respect to 𝑦 at minimal 𝑥. ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) enters the derivative
Eq. ([47](#S3.E47 "In Proof. ‣ A3.3 Proof of Theorem 3 ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) with a negative sign!
According to Lemma [18](#Thmtheorem18 "Lemma 18 (Monotone Derivative). ‣ Main Sub-Function From Below. ‣ A3.4.4 Lemmata for proofing Theorem 3: The variance is expanding ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"),
the minimal value of sub-term Eq. ([322](#S3.E322 "In Proof. ‣ Behavior of the main subfunction with respect to 𝑦 at minimal 𝑥. ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"))
is obtained by the largest largest ν𝜈\nu,
by the smallest τ𝜏\tau, and the largest y=μ​ω=0.01𝑦𝜇𝜔0.01y=\mu\omega=0.01.
Also the positive term
erfc⁡(μ​ω2​ν​τ)+2erfc𝜇𝜔2𝜈𝜏2\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+2 is multiplied by τ𝜏\tau, which is minimized
by using the smallest τ𝜏\tau.
Therefore we can use the smallest τ𝜏\tau in whole formula
Eq. ([47](#S3.E47 "In Proof. ‣ A3.3 Proof of Theorem 3 ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) to lower bound it.

First we consider the domain
0.05⩽ν⩽0.160.05𝜈0.160.05\leqslant\nu\leqslant 0.16 and 0.8⩽τ⩽1.250.8𝜏1.250.8\leqslant\tau\leqslant 1.25.
The factor consisting of the exponential in front of the brackets has
its smallest value for e−0.01⋅0.012⋅0.05⋅0.8superscript𝑒⋅0.010.01⋅20.050.8e^{-\frac{0.01\cdot 0.01}{2\cdot 0.05\cdot 0.8}}.
Since erfcerfc\operatorname{erfc} is monotonically decreasing we inserted the
smallest argument via erfc⁡(−0.012​0.05⋅0.8)erfc0.012⋅0.050.8\operatorname{erfc}\left(-\frac{0.01}{\sqrt{2}\sqrt{0.05\cdot 0.8}}\right) in order to obtain the maximal negative contribution.
Thus, applying Lemma [18](#Thmtheorem18 "Lemma 18 (Monotone Derivative). ‣ Main Sub-Function From Below. ‣ A3.4.4 Lemmata for proofing Theorem 3: The variance is expanding ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), we obtain the lower bound on the derivative:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 12λ2τe−μ2​ω22​ν​τ(α2(−(e(μ​ω+ν​τ2​ν​τ)2erfc(μ​ω+ν​τ2​ν​τ)−2e(μ​ω+2​ν​τ2​ν​τ)2erfc(μ​ω+2​ν​τ2​ν​τ)))−\displaystyle\frac{1}{2}\lambda^{2}\tau e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\left(\alpha^{2}\left(-\left(e^{\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-2e^{\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)\right)-\right. |  | (48) |
|  |  |  |
| --- | --- | --- |
|  | erfc(μ​ω2​ν​τ)+2)>\displaystyle\left.\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+2\right)\ > |  |
|  |  |  |
| --- | --- | --- |
|  | 120.8e−0.01⋅0.012⋅0.05⋅0.8λ012(α012(−(e(0.16⋅0.8+0.012​0.16⋅0.8)2erfc(0.16⋅0.8+0.012​0.16⋅0.8)−\displaystyle\frac{1}{2}0.8e^{-\frac{0.01\cdot 0.01}{2\cdot 0.05\cdot 0.8}}\lambda\_{\rm 01}^{2}\left(\alpha\_{\rm 01}^{2}\left(-\left(e^{\left(\frac{0.16\cdot 0.8+0.01}{\sqrt{2}\sqrt{0.16\cdot 0.8}}\right)^{2}}\operatorname{erfc}\left(\frac{0.16\cdot 0.8+0.01}{\sqrt{2}\sqrt{0.16\cdot 0.8}}\right)-\right.\right.\right. |  |
|  |  |  |
| --- | --- | --- |
|  | 2e(2⋅0.16⋅0.8+0.012​0.16⋅0.8)2erfc(2⋅0.16⋅0.8+0.012​0.16⋅0.8)))−erfc(−0.012​0.05⋅0.8)+2))> 0.969231.\displaystyle\left.\left.\left.2e^{\left(\frac{2\cdot 0.16\cdot 0.8+0.01}{\sqrt{2}\sqrt{0.16\cdot 0.8}}\right)^{2}}\operatorname{erfc}\left(\frac{2\cdot 0.16\cdot 0.8+0.01}{\sqrt{2}\sqrt{0.16\cdot 0.8}}\right)\right)\right)-\operatorname{erfc}\left(-\frac{0.01}{\sqrt{2}\sqrt{0.05\cdot 0.8}}\right)+2\right))\ >\ 0.969231\ . |  |

For applying the mean value theorem, we require the smallest ν~​(ν)~𝜈𝜈{\tilde{\nu}}(\nu).
We follow the proof of Lemma [8](#Thmtheorem8 "Lemma 8 (Derivatives of the Mapping). ‣ Bounds on mean, variance and second moment. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), which shows
that at the minimum y=μ​ω𝑦𝜇𝜔y=\mu\omega must be maximal
and x=ν​τ𝑥𝜈𝜏x=\nu\tau must be minimal.
Thus, the smallest
ξ~​(μ,ω,ν,τ,λ01,α01)~𝜉𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})
is
ξ~​(0.01,0.01,0.05,0.8,λ01,α01)=0.0662727~𝜉0.010.010.050.8subscript𝜆01subscript𝛼010.0662727{\tilde{\xi}}(0.01,0.01,0.05,0.8,\lambda\_{\rm 01},\alpha\_{\rm 01})=0.0662727
for 0.05⩽ν0.05𝜈0.05\leqslant\nu and 0.8⩽τ0.8𝜏0.8\leqslant\tau.

Therefore the mean value theorem and the bound on (μ~)2superscript~𝜇2({\tilde{\mu}})^{2} (Lemma [43](#Thmtheorem43 "Lemma 43 (Tight bound on 𝜇̃² in Ω⁻). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) provide

|  |  |  |  |
| --- | --- | --- | --- |
|  | ν~=ξ~​(μ,ω,ν,τ,λ01,α01)−(μ~​(μ,ω,ν,τ,λ01,α01))2>~𝜈~𝜉𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01superscript~𝜇𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼012absent\displaystyle{\tilde{\nu}}={\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})-\left({\tilde{\mu}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\right)^{2}> |  | (49) |
|  |  |  |
| --- | --- | --- |
|  | 0.0662727+ 0.969231​(ν−0.05)−0.005= 0.01281115+0.969231​ν>0.06627270.969231𝜈0.050.0050.012811150.969231𝜈absent\displaystyle 0.0662727\ +\ 0.969231(\nu-0.05)-0.005\ =\ 0.01281115+0.969231\nu\ > |  |
|  |  |  |
| --- | --- | --- |
|  | 0.08006969⋅0.16+0.969231​ν⩾1.049301​ν>ν.⋅0.080069690.160.969231𝜈1.049301𝜈𝜈\displaystyle 0.08006969\cdot 0.16+0.969231\nu\geqslant 1.049301\nu\ >\ \nu\ . |  |

Next we consider the domain
0.05⩽ν⩽0.240.05𝜈0.240.05\leqslant\nu\leqslant 0.24
and 0.9⩽τ⩽1.250.9𝜏1.250.9\leqslant\tau\leqslant 1.25.
The factor consisting of the exponential in front of the brackets has
its smallest value for e−0.01⋅0.012⋅0.05⋅0.9superscript𝑒⋅0.010.01⋅20.050.9e^{-\frac{0.01\cdot 0.01}{2\cdot 0.05\cdot 0.9}}.
Since erfcerfc\operatorname{erfc} is monotonically decreasing we inserted the
smallest argument via erfc⁡(−0.012​0.05⋅0.9)erfc0.012⋅0.050.9\operatorname{erfc}\left(-\frac{0.01}{\sqrt{2}\sqrt{0.05\cdot 0.9}}\right) in order to obtain the maximal negative contribution.

Thus, applying Lemma [18](#Thmtheorem18 "Lemma 18 (Monotone Derivative). ‣ Main Sub-Function From Below. ‣ A3.4.4 Lemmata for proofing Theorem 3: The variance is expanding ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), we obtain the lower bound on the derivative:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 12λ2τe−μ2​ω22​ν​τ(α2(−(e(μ​ω+ν​τ2​ν​τ)2erfc(μ​ω+ν​τ2​ν​τ)−2e(μ​ω+2​ν​τ2​ν​τ)2erfc(μ​ω+2​ν​τ2​ν​τ)))−\displaystyle\frac{1}{2}\lambda^{2}\tau e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\left(\alpha^{2}\left(-\left(e^{\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-2e^{\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)\right)-\right. |  | (50) |
|  |  |  |
| --- | --- | --- |
|  | erfc(μ​ω2​ν​τ)+2)>\displaystyle\left.\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+2\right)\ > |  |
|  |  |  |
| --- | --- | --- |
|  | 120.9e−0.01⋅0.012⋅0.05⋅0.9λ012(α012(−(e(0.24⋅0.9+0.012​0.24⋅0.9)2erfc(0.24⋅0.9+0.012​0.24⋅0.9)−\displaystyle\frac{1}{2}0.9e^{-\frac{0.01\cdot 0.01}{2\cdot 0.05\cdot 0.9}}\lambda\_{\rm 01}^{2}\left(\alpha\_{\rm 01}^{2}\left(-\left(e^{\left(\frac{0.24\cdot 0.9+0.01}{\sqrt{2}\sqrt{0.24\cdot 0.9}}\right)^{2}}\operatorname{erfc}\left(\frac{0.24\cdot 0.9+0.01}{\sqrt{2}\sqrt{0.24\cdot 0.9}}\right)-\right.\right.\right. |  |
|  |  |  |
| --- | --- | --- |
|  | 2e(2⋅0.24⋅0.9+0.012​0.24⋅0.9)2erfc(2⋅0.24⋅0.9+0.012​0.24⋅0.9)))−erfc(−0.012​0.05⋅0.9)+2))> 0.976952.\displaystyle\left.\left.\left.2e^{\left(\frac{2\cdot 0.24\cdot 0.9+0.01}{\sqrt{2}\sqrt{0.24\cdot 0.9}}\right)^{2}}\operatorname{erfc}\left(\frac{2\cdot 0.24\cdot 0.9+0.01}{\sqrt{2}\sqrt{0.24\cdot 0.9}}\right)\right)\right)-\operatorname{erfc}\left(-\frac{0.01}{\sqrt{2}\sqrt{0.05\cdot 0.9}}\right)+2\right))\ >\ 0.976952\ . |  |

For applying the mean value theorem, we require the smallest ν~​(ν)~𝜈𝜈{\tilde{\nu}}(\nu).
We follow the proof of Lemma [8](#Thmtheorem8 "Lemma 8 (Derivatives of the Mapping). ‣ Bounds on mean, variance and second moment. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), which shows
that at the minimum y=μ​ω𝑦𝜇𝜔y=\mu\omega must be maximal
and x=ν​τ𝑥𝜈𝜏x=\nu\tau must be minimal.
Thus, the smallest
ξ~​(μ,ω,ν,τ,λ01,α01)~𝜉𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})
is
ξ~​(0.01,0.01,0.05,0.9,λ01,α01)=0.0738404~𝜉0.010.010.050.9subscript𝜆01subscript𝛼010.0738404{\tilde{\xi}}(0.01,0.01,0.05,0.9,\lambda\_{\rm 01},\alpha\_{\rm 01})=0.0738404
for 0.05⩽ν0.05𝜈0.05\leqslant\nu and 0.9⩽τ0.9𝜏0.9\leqslant\tau.
Therefore the mean value theorem and the bound on (μ~)2superscript~𝜇2({\tilde{\mu}})^{2} (Lemma [43](#Thmtheorem43 "Lemma 43 (Tight bound on 𝜇̃² in Ω⁻). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | ν~=ξ~​(μ,ω,ν,τ,λ01,α01)−(μ~​(μ,ω,ν,τ,λ01,α01))2>~𝜈~𝜉𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01superscript~𝜇𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼012absent\displaystyle{\tilde{\nu}}={\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})-\left({\tilde{\mu}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\right)^{2}> |  | (51) |
|  |  |  |
| --- | --- | --- |
|  | 0.0738404+ 0.976952​(ν−0.05)−0.005= 0.0199928+0.976952​ν>0.07384040.976952𝜈0.050.0050.01999280.976952𝜈absent\displaystyle\ 0.0738404\ +\ 0.976952(\nu-0.05)-0.005\ =\ 0.0199928+0.976952\nu\ > |  |
|  |  |  |
| --- | --- | --- |
|  | 0.08330333⋅0.24+0.976952​ν⩾1.060255​ν>ν.⋅0.083303330.240.976952𝜈1.060255𝜈𝜈\displaystyle 0.08330333\cdot 0.24+0.976952\nu\geqslant 1.060255\nu\ >\ \nu\ . |  |

∎

### A3.4 Lemmata and Other Tools Required for the Proofs

#### A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one

In this section, we show that the largest singular value of the Jacobian of the
mapping g𝑔g is smaller than one. Therefore, g𝑔g is a contraction mapping.
This is even true in a larger domain than the original ΩΩ\Omega. We
do not need to restrict τ∈[0.95,1.1]𝜏0.951.1\tau\in[0.95,1.1], but we can extend to
τ∈[0.8,1.25]𝜏0.81.25\tau\in[0.8,1.25]. The range of the other variables is unchanged such that
we consider the following domain throughout this section: μ∈[−0.1,0.1]𝜇0.10.1\mu\in[-0.1,0.1],
ω∈[−0.1,0.1]𝜔0.10.1\omega\in[-0.1,0.1],
ν∈[0.8,1.5]𝜈0.81.5\nu\in[0.8,1.5], and
τ∈[0.8,1.25]𝜏0.81.25\tau\in[0.8,1.25].

##### Jacobian of the mapping.

In the following, we denote two Jacobians:
(1) the Jacobian 𝒥𝒥\mathcal{J} of the mapping h:(μ,ν)↦(μ~,ξ~):ℎmaps-to𝜇𝜈~𝜇~𝜉h:(\mu,\nu)\mapsto({\tilde{\mu}},{\tilde{\xi}}), and
(2) the Jacobian ℋℋ\mathcal{H} of the mapping g:(μ,ν)↦(μ~,ν~):𝑔maps-to𝜇𝜈~𝜇~𝜈g:(\mu,\nu)\mapsto({\tilde{\mu}},{\tilde{\nu}})
because the influence of μ~~𝜇{\tilde{\mu}} on ν~~𝜈{\tilde{\nu}} is small,
and many properties of the system can already be seen on 𝒥𝒥\mathcal{J}.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒥𝒥\displaystyle\mathcal{J}\ | =(𝒥11𝒥12𝒥21𝒥22)=(∂∂μ​μ~∂∂ν​μ~∂∂μ​ξ~∂∂ν​ξ~)absentsubscript𝒥11subscript𝒥12subscript𝒥21subscript𝒥22𝜇~𝜇𝜈~𝜇𝜇~𝜉𝜈~𝜉\displaystyle=\ \left(\begin{array}[]{cc}{\mathcal{J}}\_{11}&{\mathcal{J}}\_{12}\\ {\mathcal{J}}\_{21}&{\mathcal{J}}\_{22}\\ \end{array}\right)=\ \left(\begin{array}[]{cc}\frac{\partial}{\partial\mu}{\tilde{\mu}}&\frac{\partial}{\partial\nu}{\tilde{\mu}}\\ \frac{\partial}{\partial\mu}{\tilde{\xi}}&\frac{\partial}{\partial\nu}{\tilde{\xi}}\\ \end{array}\right) |  | (56) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ℋℋ\displaystyle\mathcal{H}\ | =(ℋ11ℋ12ℋ21ℋ22)=(𝒥11𝒥12𝒥21−2​μ~​𝒥11𝒥22−2​μ~​𝒥12)absentsubscriptℋ11subscriptℋ12subscriptℋ21subscriptℋ22subscript𝒥11subscript𝒥12subscript𝒥212~𝜇subscript𝒥11subscript𝒥222~𝜇subscript𝒥12\displaystyle=\ \left(\begin{array}[]{cc}{\mathcal{H}}\_{11}&{\mathcal{H}}\_{12}\\ {\mathcal{H}}\_{21}&{\mathcal{H}}\_{22}\\ \end{array}\right)=\ \left(\begin{array}[]{cc}{\mathcal{J}}\_{11}&{\mathcal{J}}\_{12}\\ {\mathcal{J}}\_{21}-2{\tilde{\mu}}{\mathcal{J}}\_{11}&{\mathcal{J}}\_{22}-2{\tilde{\mu}}{\mathcal{J}}\_{12}\\ \end{array}\right) |  | (61) |

The definition of the entries of the Jacobian 𝒥𝒥\mathcal{J} is:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 𝒥11​(μ,ω,ν,τ,λ,α)=∂∂μ​μ~​(μ,ω,ν,τ,λ,α)=subscript𝒥11𝜇𝜔𝜈𝜏𝜆𝛼𝜇~𝜇𝜇𝜔𝜈𝜏𝜆𝛼absent\displaystyle{\mathcal{J}}\_{11}(\mu,\omega,\nu,\tau,\lambda,\alpha)\ =\frac{\partial}{\partial\mu}{\tilde{\mu}}(\mu,\omega,\nu,\tau,\lambda,\alpha)\ = |  | (62) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 12​λ​ω​(α​eμ​ω+ν​τ2​erfc⁡(μ​ω+ν​τ2​ν​τ)−erfc⁡(μ​ω2​ν​τ)+2)12𝜆𝜔𝛼superscript𝑒𝜇𝜔𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏erfc𝜇𝜔2𝜈𝜏2\displaystyle\frac{1}{2}\lambda\omega\left(\alpha e^{\mu\omega+\frac{\nu\tau}{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+2\right) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 𝒥12​(μ,ω,ν,τ,λ,α)=∂∂ν​μ~​(μ,ω,ν,τ,λ,α)=subscript𝒥12𝜇𝜔𝜈𝜏𝜆𝛼𝜈~𝜇𝜇𝜔𝜈𝜏𝜆𝛼absent\displaystyle{\mathcal{J}}\_{12}(\mu,\omega,\nu,\tau,\lambda,\alpha)\ =\frac{\partial}{\partial\nu}{\tilde{\mu}}(\mu,\omega,\nu,\tau,\lambda,\alpha)\ = |  | (63) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 14​λ​τ​(α​eμ​ω+ν​τ2​erfc⁡(μ​ω+ν​τ2​ν​τ)−(α−1)​2π​ν​τ​e−μ2​ω22​ν​τ)14𝜆𝜏𝛼superscript𝑒𝜇𝜔𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏𝛼12𝜋𝜈𝜏superscript𝑒superscript𝜇2superscript𝜔22𝜈𝜏\displaystyle\frac{1}{4}\lambda\tau\left(\alpha e^{\mu\omega+\frac{\nu\tau}{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-(\alpha-1)\sqrt{\frac{2}{\pi\nu\tau}}e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\right) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 𝒥21​(μ,ω,ν,τ,λ,α)=∂∂μ​ξ~​(μ,ω,ν,τ,λ,α)=subscript𝒥21𝜇𝜔𝜈𝜏𝜆𝛼𝜇~𝜉𝜇𝜔𝜈𝜏𝜆𝛼absent\displaystyle{\mathcal{J}}\_{21}(\mu,\omega,\nu,\tau,\lambda,\alpha)\ =\ \frac{\partial}{\partial\mu}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda,\alpha)\ = |  | (64) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | λ2ω(α2(−eμ​ω+ν​τ2)erfc(μ​ω+ν​τ2​ν​τ)+\displaystyle\lambda^{2}\omega\left(\alpha^{2}\left(-e^{\mu\omega+\frac{\nu\tau}{2}}\right)\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | α2e2​μ​ω+2​ν​τerfc(μ​ω+2​ν​τ2​ν​τ)+μω(2−erfc(μ​ω2​ν​τ))+2πν​τe−μ2​ω22​ν​τ)\displaystyle\left.\alpha^{2}e^{2\mu\omega+2\nu\tau}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+\mu\omega\left(2-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)+\sqrt{\frac{2}{\pi}}\sqrt{\nu\tau}e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\right) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 𝒥22​(μ,ω,ν,τ,λ,α)=∂∂ν​ξ~​(μ,ω,ν,τ,λ,α)=subscript𝒥22𝜇𝜔𝜈𝜏𝜆𝛼𝜈~𝜉𝜇𝜔𝜈𝜏𝜆𝛼absent\displaystyle{\mathcal{J}}\_{22}(\mu,\omega,\nu,\tau,\lambda,\alpha)\ =\frac{\partial}{\partial\nu}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda,\alpha)\ = |  | (65) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 12λ2τ(α2(−eμ​ω+ν​τ2)erfc(μ​ω+ν​τ2​ν​τ)+\displaystyle\frac{1}{2}\lambda^{2}\tau\left(\alpha^{2}\left(-e^{\mu\omega+\frac{\nu\tau}{2}}\right)\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2α2e2​μ​ω+2​ν​τerfc(μ​ω+2​ν​τ2​ν​τ)−erfc(μ​ω2​ν​τ)+2)\displaystyle\left.2\alpha^{2}e^{2\mu\omega+2\nu\tau}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+2\right) |  |

##### Proof sketch: Bounding the largest singular value of the Jacobian.

If the largest singular value of the Jacobian is smaller than 1, then
the spectral norm of the Jacobian is smaller than 1.
Then the mapping Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))
and Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))
of the mean and variance to the mean and variance in the next layer is contracting.

We show that the largest singular value is smaller than 1 by
evaluating the function
S​(μ,ω,ν,τ,λ,α)𝑆𝜇𝜔𝜈𝜏𝜆𝛼S(\mu,\omega,\nu,\tau,\lambda,\alpha) on a grid.
Then we use the Mean Value Theorem to bound the deviation of the
function S𝑆S between grid points.
Toward this end we have to bound the gradient of S𝑆S with respect to
(μ,ω,ν,τ)𝜇𝜔𝜈𝜏(\mu,\omega,\nu,\tau). If all function values plus
gradient times the deltas (differences between grid points and evaluated
points) is still smaller than 1, then we have proofed that the
function is below 1.

The singular values of the 2×2222\times 2 matrix

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝑨𝑨\displaystyle\bm{A}\ | =(a11a12a21a22)absentsubscript𝑎11subscript𝑎12subscript𝑎21subscript𝑎22\displaystyle=\ \left(\begin{array}[]{cc}a\_{11}&a\_{12}\\ a\_{21}&a\_{22}\\ \end{array}\right) |  | (68) |

are

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | s1subscript𝑠1\displaystyle s\_{1}\ | =12​((a11+a22)2+(a21−a12)2+(a11−a22)2+(a12+a21)2)absent12superscriptsubscript𝑎11subscript𝑎222superscriptsubscript𝑎21subscript𝑎122superscriptsubscript𝑎11subscript𝑎222superscriptsubscript𝑎12subscript𝑎212\displaystyle=\ \frac{1}{2}\ \left(\sqrt{(a\_{11}+a\_{22})^{2}+(a\_{21}-a\_{12})^{2}}\ +\ \sqrt{(a\_{11}-a\_{22})^{2}+(a\_{12}+a\_{21})^{2}}\right) |  | (69) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | s2subscript𝑠2\displaystyle s\_{2} | =12​((a11+a22)2+(a21−a12)2−(a11−a22)2+(a12+a21)2).absent12superscriptsubscript𝑎11subscript𝑎222superscriptsubscript𝑎21subscript𝑎122superscriptsubscript𝑎11subscript𝑎222superscriptsubscript𝑎12subscript𝑎212\displaystyle=\frac{1}{2}\ \left(\sqrt{(a\_{11}+a\_{22})^{2}+(a\_{21}-a\_{12})^{2}}\ -\ \sqrt{(a\_{11}-a\_{22})^{2}+(a\_{12}+a\_{21})^{2}}\right). |  | (70) |

We used an explicit formula for the singular values [[4](#bib.bib4)]. We now set
ℋ11=a11,ℋ12=a12,ℋ21=a21,ℋ22=a22formulae-sequencesubscriptℋ11subscript𝑎11formulae-sequencesubscriptℋ12subscript𝑎12formulae-sequencesubscriptℋ21subscript𝑎21subscriptℋ22subscript𝑎22{\mathcal{H}}\_{11}=a\_{11},{\mathcal{H}}\_{12}=a\_{12},{\mathcal{H}}\_{21}=a\_{21},{\mathcal{H}}\_{22}=a\_{22}
to obtain a formula for the largest singular value of the Jacobian
depending on (μ,ω,ν,τ,λ,α)𝜇𝜔𝜈𝜏𝜆𝛼(\mu,\omega,\nu,\tau,\lambda,\alpha).
The formula for the largest singular value for the Jacobian is:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | S​(μ,ω,ν,τ,λ,α)𝑆𝜇𝜔𝜈𝜏𝜆𝛼\displaystyle S(\mu,\omega,\nu,\tau,\lambda,\alpha)\ | =((ℋ11+ℋ22)2+(ℋ21−ℋ12)2+(ℋ11−ℋ22)2+(ℋ12+ℋ21)2)=absentsuperscriptsubscriptℋ11subscriptℋ222superscriptsubscriptℋ21subscriptℋ122superscriptsubscriptℋ11subscriptℋ222superscriptsubscriptℋ12subscriptℋ212absent\displaystyle=\left(\sqrt{({\mathcal{H}}\_{11}+{\mathcal{H}}\_{22})^{2}+({\mathcal{H}}\_{21}-{\mathcal{H}}\_{12})^{2}}\ +\ \sqrt{({\mathcal{H}}\_{11}-{\mathcal{H}}\_{22})^{2}+({\mathcal{H}}\_{12}+{\mathcal{H}}\_{21})^{2}}\right)= |  | (71) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =12((𝒥11+𝒥22−2​μ~​𝒥12)2+(𝒥21−2​μ~​𝒥11−𝒥12)2+\displaystyle=\ \frac{1}{2}\ \left(\sqrt{({\mathcal{J}}\_{11}+{\mathcal{J}}\_{22}-2{\tilde{\mu}}{\mathcal{J}}\_{12})^{2}+({\mathcal{J}}\_{21}-2{\tilde{\mu}}{\mathcal{J}}\_{11}-{\mathcal{J}}\_{12})^{2}}\ +\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (𝒥11−𝒥22+2​μ~​𝒥12)2+(𝒥12+𝒥21−2​μ~​𝒥11)2),\displaystyle\left.\sqrt{({\mathcal{J}}\_{11}-{\mathcal{J}}\_{22}+2{\tilde{\mu}}{\mathcal{J}}\_{12})^{2}+({\mathcal{J}}\_{12}+{\mathcal{J}}\_{21}-2{\tilde{\mu}}{\mathcal{J}}\_{11})^{2}}\right), |  |

where 𝒥𝒥\mathcal{J} are defined in Eq. ([62](#S3.E62 "In Jacobian of the mapping. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) and we left out the dependencies on
(μ,ω,ν,τ,λ,α)𝜇𝜔𝜈𝜏𝜆𝛼(\mu,\omega,\nu,\tau,\lambda,\alpha) in order to keep the notation uncluttered, e.g. we
wrote 𝒥11subscript𝒥11{\mathcal{J}}\_{11} instead of 𝒥11​(μ,ω,ν,τ,λ,α)subscript𝒥11𝜇𝜔𝜈𝜏𝜆𝛼{\mathcal{J}}\_{11}(\mu,\omega,\nu,\tau,\lambda,\alpha).

##### Bounds on the derivatives of the Jacobian entries.

In order to bound the gradient of the singular value, we have to bound
the derivatives of the Jacobian entries
𝒥11​(μ,ω,ν,τ,λ,α)subscript𝒥11𝜇𝜔𝜈𝜏𝜆𝛼{\mathcal{J}}\_{11}(\mu,\omega,\nu,\tau,\lambda,\alpha),
𝒥12​(μ,ω,ν,τ,λ,α)subscript𝒥12𝜇𝜔𝜈𝜏𝜆𝛼{\mathcal{J}}\_{12}(\mu,\omega,\nu,\tau,\lambda,\alpha),
𝒥21​(μ,ω,ν,τ,λ,α)subscript𝒥21𝜇𝜔𝜈𝜏𝜆𝛼{\mathcal{J}}\_{21}(\mu,\omega,\nu,\tau,\lambda,\alpha), and
𝒥22​(μ,ω,ν,τ,λ,α)subscript𝒥22𝜇𝜔𝜈𝜏𝜆𝛼{\mathcal{J}}\_{22}(\mu,\omega,\nu,\tau,\lambda,\alpha)
with respect to
μ𝜇\mu, ω𝜔\omega, ν𝜈\nu, and τ𝜏\tau. The values
λ𝜆\lambda and α𝛼\alpha are fixed to λ01subscript𝜆01\lambda\_{\rm 01} and α01subscript𝛼01\alpha\_{\rm 01}.
The 16 derivatives of the 4 Jacobian entries with respect to the 4
variables are:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂𝒥11∂μsubscript𝒥11𝜇\displaystyle\frac{\partial{\mathcal{J}}\_{11}}{\partial\mu}\ | =12​λ​ω2​e−μ2​ω22​ν​τ​(α​e(μ​ω+ν​τ)22​ν​τ​erfc⁡(μ​ω+ν​τ2​ν​τ)−2π​(α−1)ν​τ)absent12𝜆superscript𝜔2superscript𝑒superscript𝜇2superscript𝜔22𝜈𝜏𝛼superscript𝑒superscript𝜇𝜔𝜈𝜏22𝜈𝜏erfc𝜇𝜔𝜈𝜏2𝜈𝜏2𝜋𝛼1𝜈𝜏\displaystyle=\ \frac{1}{2}\lambda\omega^{2}e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\left(\alpha e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-\frac{\sqrt{\frac{2}{\pi}}(\alpha-1)}{\sqrt{\nu\tau}}\right) |  | (72) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂𝒥11∂ωsubscript𝒥11𝜔\displaystyle\frac{\partial{\mathcal{J}}\_{11}}{\partial\omega}\ | =12λ(−e−μ2​ω22​ν​τ(2π​(α−1)​μ​ων​τ−α(μω+1)e(μ​ω+ν​τ)22​ν​τerfc(μ​ω+ν​τ2​ν​τ))−\displaystyle=\ \frac{1}{2}\lambda\left(-e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\left(\frac{\sqrt{\frac{2}{\pi}}(\alpha-1)\mu\omega}{\sqrt{\nu\tau}}-\alpha(\mu\omega+1)e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)\ -\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | erfc(μ​ω2​ν​τ)+2)\displaystyle\left.\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+2\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂𝒥11∂νsubscript𝒥11𝜈\displaystyle\frac{\partial{\mathcal{J}}\_{11}}{\partial\nu}\ | =14​λ​τ​ω​e−μ2​ω22​ν​τ​(α​e(μ​ω+ν​τ)22​ν​τ​erfc⁡(μ​ω+ν​τ2​ν​τ)+2π​((α−1)​μ​ω(ν​τ)3/2−αν​τ))absent14𝜆𝜏𝜔superscript𝑒superscript𝜇2superscript𝜔22𝜈𝜏𝛼superscript𝑒superscript𝜇𝜔𝜈𝜏22𝜈𝜏erfc𝜇𝜔𝜈𝜏2𝜈𝜏2𝜋𝛼1𝜇𝜔superscript𝜈𝜏32𝛼𝜈𝜏\displaystyle=\ \frac{1}{4}\lambda\tau\omega e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\left(\alpha e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+\sqrt{\frac{2}{\pi}}\left(\frac{(\alpha-1)\mu\omega}{(\nu\tau)^{3/2}}-\frac{\alpha}{\sqrt{\nu\tau}}\right)\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂𝒥11∂τsubscript𝒥11𝜏\displaystyle\frac{\partial{\mathcal{J}}\_{11}}{\partial\tau}\ | =14​λ​ν​ω​e−μ2​ω22​ν​τ​(α​e(μ​ω+ν​τ)22​ν​τ​erfc⁡(μ​ω+ν​τ2​ν​τ)+2π​((α−1)​μ​ω(ν​τ)3/2−αν​τ))absent14𝜆𝜈𝜔superscript𝑒superscript𝜇2superscript𝜔22𝜈𝜏𝛼superscript𝑒superscript𝜇𝜔𝜈𝜏22𝜈𝜏erfc𝜇𝜔𝜈𝜏2𝜈𝜏2𝜋𝛼1𝜇𝜔superscript𝜈𝜏32𝛼𝜈𝜏\displaystyle=\ \frac{1}{4}\lambda\nu\omega e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\left(\alpha e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+\sqrt{\frac{2}{\pi}}\left(\frac{(\alpha-1)\mu\omega}{(\nu\tau)^{3/2}}-\frac{\alpha}{\sqrt{\nu\tau}}\right)\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂𝒥12∂μsubscript𝒥12𝜇\displaystyle\frac{\partial{\mathcal{J}}\_{12}}{\partial\mu}\ | =∂𝒥11∂νabsentsubscript𝒥11𝜈\displaystyle=\ \frac{\partial{\mathcal{J}}\_{11}}{\partial\nu} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂𝒥12∂ωsubscript𝒥12𝜔\displaystyle\frac{\partial{\mathcal{J}}\_{12}}{\partial\omega}\ | =14​λ​μ​τ​e−μ2​ω22​ν​τ​(α​e(μ​ω+ν​τ)22​ν​τ​erfc⁡(μ​ω+ν​τ2​ν​τ)+2π​((α−1)​μ​ω(ν​τ)3/2−αν​τ))absent14𝜆𝜇𝜏superscript𝑒superscript𝜇2superscript𝜔22𝜈𝜏𝛼superscript𝑒superscript𝜇𝜔𝜈𝜏22𝜈𝜏erfc𝜇𝜔𝜈𝜏2𝜈𝜏2𝜋𝛼1𝜇𝜔superscript𝜈𝜏32𝛼𝜈𝜏\displaystyle=\ \frac{1}{4}\lambda\mu\tau e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\left(\alpha e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+\sqrt{\frac{2}{\pi}}\left(\frac{(\alpha-1)\mu\omega}{(\nu\tau)^{3/2}}-\frac{\alpha}{\sqrt{\nu\tau}}\right)\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂𝒥12∂νsubscript𝒥12𝜈\displaystyle\frac{\partial{\mathcal{J}}\_{12}}{\partial\nu}\ | =18λe−μ2​ω22​ν​τ(ατ2e(μ​ω+ν​τ)22​ν​τerfc(μ​ω+ν​τ2​ν​τ)+\displaystyle=\ \frac{1}{8}\lambda e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\left(\alpha\tau^{2}e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\ +\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2π((−1)​(α−1)​μ2​ω2ν5/2​τ+τ​(α+α​μ​ω−1)ν3/2−α​τ3/2ν))\displaystyle\left.\sqrt{\frac{2}{\pi}}\left(\frac{(-1)(\alpha-1)\mu^{2}\omega^{2}}{\nu^{5/2}\sqrt{\tau}}+\frac{\sqrt{\tau}(\alpha+\alpha\mu\omega-1)}{\nu^{3/2}}-\frac{\alpha\tau^{3/2}}{\sqrt{\nu}}\right)\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂𝒥12∂τsubscript𝒥12𝜏\displaystyle\frac{\partial{\mathcal{J}}\_{12}}{\partial\tau}\ | =18λe−μ2​ω22​ν​τ(2αe(μ​ω+ν​τ)22​ν​τerfc(μ​ω+ν​τ2​ν​τ)+αντe(μ​ω+ν​τ)22​ν​τerfc(μ​ω+ν​τ2​ν​τ)+\displaystyle=\ \frac{1}{8}\lambda e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\left(2\alpha e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+\alpha\nu\tau e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\ +\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2π((−1)​(α−1)​μ2​ω2(ν​τ)3/2+−α+α​μ​ω+1ν​τ−αν​τ))\displaystyle\left.\sqrt{\frac{2}{\pi}}\left(\frac{(-1)(\alpha-1)\mu^{2}\omega^{2}}{(\nu\tau)^{3/2}}+\frac{-\alpha+\alpha\mu\omega+1}{\sqrt{\nu\tau}}-\alpha\sqrt{\nu\tau}\right)\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂𝒥21∂μsubscript𝒥21𝜇\displaystyle\frac{\partial{\mathcal{J}}\_{21}}{\partial\mu}\ | =λ2ω2(α2(−e−μ2​ω22​ν​τ)e(μ​ω+ν​τ)22​ν​τerfc(μ​ω+ν​τ2​ν​τ)+\displaystyle=\ \lambda^{2}\omega^{2}\left(\alpha^{2}\left(-e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\right)e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\ +\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2α2e(μ​ω+2​ν​τ)22​ν​τe−μ2​ω22​ν​τerfc(μ​ω+2​ν​τ2​ν​τ)−erfc(μ​ω2​ν​τ)+2)\displaystyle\left.2\alpha^{2}e^{\frac{(\mu\omega+2\nu\tau)^{2}}{2\nu\tau}}e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+2\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂𝒥21∂ωsubscript𝒥21𝜔\displaystyle\frac{\partial{\mathcal{J}}\_{21}}{\partial\omega}\ | =λ2(α2(μω+1)(−e−μ2​ω22​ν​τ)e(μ​ω+ν​τ)22​ν​τerfc(μ​ω+ν​τ2​ν​τ)+\displaystyle=\ \lambda^{2}\left(\alpha^{2}(\mu\omega+1)\left(-e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\right)e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\ +\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | α2​(2​μ​ω+1)​e(μ​ω+2​ν​τ)22​ν​τ​e−μ2​ω22​ν​τ​erfc⁡(μ​ω+2​ν​τ2​ν​τ)+limit-fromsuperscript𝛼22𝜇𝜔1superscript𝑒superscript𝜇𝜔2𝜈𝜏22𝜈𝜏superscript𝑒superscript𝜇2superscript𝜔22𝜈𝜏erfc𝜇𝜔2𝜈𝜏2𝜈𝜏\displaystyle\left.\alpha^{2}(2\mu\omega+1)e^{\frac{(\mu\omega+2\nu\tau)^{2}}{2\nu\tau}}e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\ +\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2μω(2−erfc(μ​ω2​ν​τ))+2πν​τe−μ2​ω22​ν​τ)\displaystyle\left.2\mu\omega\left(2-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)+\sqrt{\frac{2}{\pi}}\sqrt{\nu\tau}e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂𝒥21∂νsubscript𝒥21𝜈\displaystyle\frac{\partial{\mathcal{J}}\_{21}}{\partial\nu}\ | =12λ2τωe−μ2​ω22​ν​τ(α2(−e(μ​ω+ν​τ)22​ν​τ)erfc(μ​ω+ν​τ2​ν​τ)+\displaystyle=\ \frac{1}{2}\lambda^{2}\tau\omega e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\left(\alpha^{2}\left(-e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\right)\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\ +\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 4α2e(μ​ω+2​ν​τ)22​ν​τerfc(μ​ω+2​ν​τ2​ν​τ)+2π​(−1)​(α2−1)ν​τ)\displaystyle\left.4\alpha^{2}e^{\frac{(\mu\omega+2\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+\frac{\sqrt{\frac{2}{\pi}}(-1)\left(\alpha^{2}-1\right)}{\sqrt{\nu\tau}}\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂𝒥21∂τsubscript𝒥21𝜏\displaystyle\frac{\partial{\mathcal{J}}\_{21}}{\partial\tau}\ | =12λ2νωe−μ2​ω22​ν​τ(α2(−e(μ​ω+ν​τ)22​ν​τ)erfc(μ​ω+ν​τ2​ν​τ)+\displaystyle=\ \frac{1}{2}\lambda^{2}\nu\omega e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\left(\alpha^{2}\left(-e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\right)\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\ +\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 4α2e(μ​ω+2​ν​τ)22​ν​τerfc(μ​ω+2​ν​τ2​ν​τ)+2π​(−1)​(α2−1)ν​τ)\displaystyle\left.4\alpha^{2}e^{\frac{(\mu\omega+2\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+\frac{\sqrt{\frac{2}{\pi}}(-1)\left(\alpha^{2}-1\right)}{\sqrt{\nu\tau}}\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂𝒥22∂μsubscript𝒥22𝜇\displaystyle\frac{\partial{\mathcal{J}}\_{22}}{\partial\mu}\ | =∂𝒥21∂νabsentsubscript𝒥21𝜈\displaystyle=\ \frac{\partial{\mathcal{J}}\_{21}}{\partial\nu} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂𝒥22∂ωsubscript𝒥22𝜔\displaystyle\frac{\partial{\mathcal{J}}\_{22}}{\partial\omega}\ | =12λ2μτe−μ2​ω22​ν​τ(α2(−e(μ​ω+ν​τ)22​ν​τ)erfc(μ​ω+ν​τ2​ν​τ)+\displaystyle=\ \frac{1}{2}\lambda^{2}\mu\tau e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\left(\alpha^{2}\left(-e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\right)\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\ +\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 4α2e(μ​ω+2​ν​τ)22​ν​τerfc(μ​ω+2​ν​τ2​ν​τ)+2π​(−1)​(α2−1)ν​τ)\displaystyle\left.4\alpha^{2}e^{\frac{(\mu\omega+2\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+\frac{\sqrt{\frac{2}{\pi}}(-1)\left(\alpha^{2}-1\right)}{\sqrt{\nu\tau}}\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂𝒥22∂νsubscript𝒥22𝜈\displaystyle\frac{\partial{\mathcal{J}}\_{22}}{\partial\nu}\ | =14λ2τ2e−μ2​ω22​ν​τ(α2(−e(μ​ω+ν​τ)22​ν​τ)erfc(μ​ω+ν​τ2​ν​τ)+\displaystyle=\ \frac{1}{4}\lambda^{2}\tau^{2}e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\left(\alpha^{2}\left(-e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\right)\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\ +\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 8α2e(μ​ω+2​ν​τ)22​ν​τerfc(μ​ω+2​ν​τ2​ν​τ)+2π((α2−1)​μ​ω(ν​τ)3/2−3​α2ν​τ))\displaystyle\left.8\alpha^{2}e^{\frac{(\mu\omega+2\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+\sqrt{\frac{2}{\pi}}\left(\frac{\left(\alpha^{2}-1\right)\mu\omega}{(\nu\tau)^{3/2}}-\frac{3\alpha^{2}}{\sqrt{\nu\tau}}\right)\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂𝒥22∂τsubscript𝒥22𝜏\displaystyle\frac{\partial{\mathcal{J}}\_{22}}{\partial\tau}\ | =14λ2(−2α2e−μ2​ω22​ν​τe(μ​ω+ν​τ)22​ν​τerfc(μ​ω+ν​τ2​ν​τ)−\displaystyle=\ \frac{1}{4}\lambda^{2}\left(-2\alpha^{2}e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\ -\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | α2​ν​τ​e−μ2​ω22​ν​τ​e(μ​ω+ν​τ)22​ν​τ​erfc⁡(μ​ω+ν​τ2​ν​τ)+4​α2​e(μ​ω+2​ν​τ)22​ν​τ​e−μ2​ω22​ν​τ​erfc⁡(μ​ω+2​ν​τ2​ν​τ)+superscript𝛼2𝜈𝜏superscript𝑒superscript𝜇2superscript𝜔22𝜈𝜏superscript𝑒superscript𝜇𝜔𝜈𝜏22𝜈𝜏erfc𝜇𝜔𝜈𝜏2𝜈𝜏limit-from4superscript𝛼2superscript𝑒superscript𝜇𝜔2𝜈𝜏22𝜈𝜏superscript𝑒superscript𝜇2superscript𝜔22𝜈𝜏erfc𝜇𝜔2𝜈𝜏2𝜈𝜏\displaystyle\left.\alpha^{2}\nu\tau e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+4\alpha^{2}e^{\frac{(\mu\omega+2\nu\tau)^{2}}{2\nu\tau}}e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 8​α2​ν​τ​e(μ​ω+2​ν​τ)22​ν​τ​e−μ2​ω22​ν​τ​erfc⁡(μ​ω+2​ν​τ2​ν​τ)+2​(2−erfc⁡(μ​ω2​ν​τ))+8superscript𝛼2𝜈𝜏superscript𝑒superscript𝜇𝜔2𝜈𝜏22𝜈𝜏superscript𝑒superscript𝜇2superscript𝜔22𝜈𝜏erfc𝜇𝜔2𝜈𝜏2𝜈𝜏limit-from22erfc𝜇𝜔2𝜈𝜏\displaystyle\left.8\alpha^{2}\nu\tau e^{\frac{(\mu\omega+2\nu\tau)^{2}}{2\nu\tau}}e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+2\left(2-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)+\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2πe−μ2​ω22​ν​τ((α2−1)​μ​ων​τ−3α2ν​τ))\displaystyle\left.\sqrt{\frac{2}{\pi}}e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\left(\frac{\left(\alpha^{2}-1\right)\mu\omega}{\sqrt{\nu\tau}}-3\alpha^{2}\sqrt{\nu\tau}\right)\right) |  |

###### Lemma 5 (Bounds on the Derivatives).

The following bounds on the absolute values of the
derivatives of the Jacobian entries 𝒥11​(μ,ω,ν,τ,λ,α)subscript𝒥11𝜇𝜔𝜈𝜏𝜆𝛼{\mathcal{J}}\_{11}(\mu,\omega,\nu,\tau,\lambda,\alpha),
𝒥12​(μ,ω,ν,τ,λ,α)subscript𝒥12𝜇𝜔𝜈𝜏𝜆𝛼{\mathcal{J}}\_{12}(\mu,\omega,\nu,\tau,\lambda,\alpha),
𝒥21​(μ,ω,ν,τ,λ,α)subscript𝒥21𝜇𝜔𝜈𝜏𝜆𝛼{\mathcal{J}}\_{21}(\mu,\omega,\nu,\tau,\lambda,\alpha), and
𝒥22​(μ,ω,ν,τ,λ,α)subscript𝒥22𝜇𝜔𝜈𝜏𝜆𝛼{\mathcal{J}}\_{22}(\mu,\omega,\nu,\tau,\lambda,\alpha)
with respect to
μ𝜇\mu, ω𝜔\omega, ν𝜈\nu, and τ𝜏\tau hold:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | |∂𝒥11∂μ|subscript𝒥11𝜇\displaystyle\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\mu}\right|\ | < 0.0031049101995398316absent0.0031049101995398316\displaystyle<\ 0.0031049101995398316 |  | (73) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥11∂ω|subscript𝒥11𝜔\displaystyle\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\omega}\right|\ | < 1.055872374194189absent1.055872374194189\displaystyle<\ 1.055872374194189 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥11∂ν|subscript𝒥11𝜈\displaystyle\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\nu}\right|\ | < 0.031242911235461816absent0.031242911235461816\displaystyle<\ 0.031242911235461816 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥11∂τ|subscript𝒥11𝜏\displaystyle\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\tau}\right|\ | < 0.03749149348255419absent0.03749149348255419\displaystyle<\ 0.03749149348255419 |  |

|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥12∂μ|subscript𝒥12𝜇\displaystyle\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\mu}\right|\ | < 0.031242911235461816absent0.031242911235461816\displaystyle<\ 0.031242911235461816 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥12∂ω|subscript𝒥12𝜔\displaystyle\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\omega}\right|\ | < 0.031242911235461816absent0.031242911235461816\displaystyle<\ 0.031242911235461816 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥12∂ν|subscript𝒥12𝜈\displaystyle\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\nu}\right|\ | < 0.21232788238624354absent0.21232788238624354\displaystyle<\ 0.21232788238624354 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥12∂τ|subscript𝒥12𝜏\displaystyle\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\tau}\right|\ | < 0.2124377655377270absent0.2124377655377270\displaystyle<\ 0.2124377655377270 |  |

|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥21∂μ|subscript𝒥21𝜇\displaystyle\left|\frac{\partial{\mathcal{J}}\_{21}}{\partial\mu}\right|\ | < 0.02220441024325437absent0.02220441024325437\displaystyle<\ 0.02220441024325437 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥21∂ω|subscript𝒥21𝜔\displaystyle\left|\frac{\partial{\mathcal{J}}\_{21}}{\partial\omega}\right|\ | < 1.146955401845684absent1.146955401845684\displaystyle<\ 1.146955401845684 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥21∂ν|subscript𝒥21𝜈\displaystyle\left|\frac{\partial{\mathcal{J}}\_{21}}{\partial\nu}\right|\ | < 0.14983446469110305absent0.14983446469110305\displaystyle<\ 0.14983446469110305 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥21∂τ|subscript𝒥21𝜏\displaystyle\left|\frac{\partial{\mathcal{J}}\_{21}}{\partial\tau}\right|\ | < 0.17980135762932363absent0.17980135762932363\displaystyle<\ 0.17980135762932363 |  |

|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥22∂μ|subscript𝒥22𝜇\displaystyle\left|\frac{\partial{\mathcal{J}}\_{22}}{\partial\mu}\right|\ | < 0.14983446469110305absent0.14983446469110305\displaystyle<\ 0.14983446469110305 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥22∂ω|subscript𝒥22𝜔\displaystyle\left|\frac{\partial{\mathcal{J}}\_{22}}{\partial\omega}\right|\ | < 0.14983446469110305absent0.14983446469110305\displaystyle<\ 0.14983446469110305 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥22∂ν|subscript𝒥22𝜈\displaystyle\left|\frac{\partial{\mathcal{J}}\_{22}}{\partial\nu}\right|\ | < 1.805740052651535absent1.805740052651535\displaystyle<\ 1.805740052651535 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥22∂τ|subscript𝒥22𝜏\displaystyle\left|\frac{\partial{\mathcal{J}}\_{22}}{\partial\tau}\right|\ | < 2.396685907216327absent2.396685907216327\displaystyle<\ 2.396685907216327 |  |

###### Proof.

See proof [39](#Thmtheorem39 "Lemma 39 (Bounds on the Derivatives). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").
∎

##### Bounds on the entries of the Jacobian.

###### Lemma 6 (Bound on J11).

The absolute value of the function
  
𝒥11=12​λ​ω​(α​eμ​ω+ν​τ2​erfc⁡(μ​ω+ν​τ2​ν​τ)−erfc⁡(μ​ω2​ν​τ)+2)subscript𝒥1112𝜆𝜔𝛼superscript𝑒𝜇𝜔𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏erfc𝜇𝜔2𝜈𝜏2\mathcal{J}\_{11}=\frac{1}{2}\lambda\omega\left(\alpha e^{\mu\omega+\frac{\nu\tau}{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+2\right) is bounded by
|𝒥11|⩽0.104497subscript𝒥110.104497\left|\mathcal{J}\_{11}\right|\leqslant 0.104497 in the domain −0.1⩽μ⩽0.10.1𝜇0.1-0.1\leqslant\mu\leqslant 0.1, −0.1⩽ω⩽0.10.1𝜔0.1-0.1\leqslant\omega\leqslant 0.1, 0.8⩽ν⩽1.50.8𝜈1.50.8\leqslant\nu\leqslant 1.5,
and 0.8⩽τ⩽1.250.8𝜏1.250.8\leqslant\tau\leqslant 1.25 for α=α01𝛼subscript𝛼01\alpha=\alpha\_{\mathrm{01}} and λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\mathrm{01}}.

###### Proof.

|  |  |  |
| --- | --- | --- |
|  | |𝒥11|=|12​λ​ω​(α​eμ​ω+ν​τ2​erfc⁡(μ​ω+ν​τ2​ν​τ)+2−erfc⁡(μ​ω2​ν​τ))|subscript𝒥1112𝜆𝜔𝛼superscript𝑒𝜇𝜔𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏2erfc𝜇𝜔2𝜈𝜏\displaystyle\left|\mathcal{J}\_{11}\right|=\left|\frac{1}{2}\lambda\omega\left(\alpha e^{\mu\omega+\frac{\nu\tau}{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+2-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)\right| |  |
|  |  |  |
| --- | --- | --- |
|  | ⩽|12|​|λ|​|ω|​(|α|​0.587622+1.00584)⩽0.104497,absent12𝜆𝜔𝛼0.5876221.005840.104497\displaystyle\leqslant|\frac{1}{2}||\lambda||\omega|\left(|\alpha|0.587622+1.00584\right)\leqslant 0.104497, |  |

where we used that (a) J11subscript𝐽11J\_{11} is strictly monotonically increasing in μ​ω𝜇𝜔\mu\omega and |2−erfc⁡(0.012​ν​τ)|⩽1.005842erfc0.012𝜈𝜏1.00584|2-\operatorname{erfc}\left(\frac{0.01}{\sqrt{2}\sqrt{\nu\tau}}\right)|\leqslant 1.00584
and (b) Lemma [47](#Thmtheorem47 "Lemma 47. ‣ Behavior of the main subfunction with respect to 𝑦 at minimal 𝑥. ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") that
|eμ​ω+ν​τ2​erfc⁡(μ​ω+ν​τ2​ν​τ)|⩽e0.01+0.642​erfc⁡(0.01+0.642​0.64)=0.587622superscript𝑒𝜇𝜔𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏superscript𝑒0.010.642erfc0.010.6420.640.587622|e^{\mu\omega+\frac{\nu\tau}{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)|\leqslant e^{0.01+\frac{0.64}{2}}\operatorname{erfc}\left(\frac{0.01+0.64}{\sqrt{2}\sqrt{0.64}}\right)=0.587622
∎

###### Lemma 7 (Bound on J12).

The absolute value of the function
  
𝒥12=14​λ​τ​(α​eμ​ω+ν​τ2​erfc⁡(μ​ω+ν​τ2​ν​τ)−(α−1)​2π​ν​τ​e−μ2​ω22​ν​τ)subscript𝒥1214𝜆𝜏𝛼superscript𝑒𝜇𝜔𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏𝛼12𝜋𝜈𝜏superscript𝑒superscript𝜇2superscript𝜔22𝜈𝜏\mathcal{J}\_{12}=\frac{1}{4}\lambda\tau\left(\alpha e^{\mu\omega+\frac{\nu\tau}{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-(\alpha-1)\sqrt{\frac{2}{\pi\nu\tau}}e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\right) is bounded by
|𝒥12|⩽0.194145subscript𝒥120.194145\left|\mathcal{J}\_{12}\right|\leqslant 0.194145 in the domain −0.1⩽μ⩽0.10.1𝜇0.1-0.1\leqslant\mu\leqslant 0.1, −0.1⩽ω⩽0.10.1𝜔0.1-0.1\leqslant\omega\leqslant 0.1, 0.8⩽ν⩽1.50.8𝜈1.50.8\leqslant\nu\leqslant 1.5,
and 0.8⩽τ⩽1.250.8𝜏1.250.8\leqslant\tau\leqslant 1.25 for α=α01𝛼subscript𝛼01\alpha=\alpha\_{\mathrm{01}} and λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\mathrm{01}}.

###### Proof.

|  |  |  |
| --- | --- | --- |
|  | |J12|⩽14​|λ|​|τ|​|(α​eμ​ω+ν​τ2​erfc⁡(μ​ω+ν​τ2​ν​τ)−(α−1)​2π​ν​τ​e−μ2​ω22​ν​τ)|⩽subscript𝐽1214𝜆𝜏𝛼superscript𝑒𝜇𝜔𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏𝛼12𝜋𝜈𝜏superscript𝑒superscript𝜇2superscript𝜔22𝜈𝜏absent\displaystyle|J\_{12}|\leqslant\frac{1}{4}|\lambda||\tau|\left|\left(\alpha e^{\mu\omega+\frac{\nu\tau}{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-(\alpha-1)\sqrt{\frac{2}{\pi\nu\tau}}e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\right)\right|\leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | 14​|λ|​|τ|​|0.983247−0.392294|⩽14𝜆𝜏0.9832470.392294absent\displaystyle\frac{1}{4}|\lambda||\tau|\left|0.983247-0.392294\right|\leqslant |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 0.1940350.194035\displaystyle 0.194035 |  | (75) |

For the first term we have 0.434947⩽eμ​ω+ν​τ2​erfc⁡(μ​ω+ν​τ2​ν​τ)⩽0.5876220.434947superscript𝑒𝜇𝜔𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏0.5876220.434947\leqslant e^{\mu\omega+\frac{\nu\tau}{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\leqslant 0.587622 after
Lemma [47](#Thmtheorem47 "Lemma 47. ‣ Behavior of the main subfunction with respect to 𝑦 at minimal 𝑥. ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") and for the second term 0.582677⩽2π​ν​τ​e−μ2​ω22​ν​τ⩽0.9973560.5826772𝜋𝜈𝜏superscript𝑒superscript𝜇2superscript𝜔22𝜈𝜏0.9973560.582677\leqslant\sqrt{\frac{2}{\pi\nu\tau}}e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\leqslant 0.997356, which can easily be seen
by maximizing or minimizing the arguments of the exponential or the square root function. The first term scaled by α𝛼\alpha is
0.727780⩽α​eμ​ω+ν​τ2​erfc⁡(μ​ω+ν​τ2​ν​τ)⩽0.9832470.727780𝛼superscript𝑒𝜇𝜔𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏0.9832470.727780\leqslant\alpha e^{\mu\omega+\frac{\nu\tau}{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\leqslant 0.983247
and the second term scaled by α−1𝛼1\alpha-1 is
0.392294⩽(α−1)​2π​ν​τ​e−μ2​ω22​ν​τ⩽0.6714840.392294𝛼12𝜋𝜈𝜏superscript𝑒superscript𝜇2superscript𝜔22𝜈𝜏0.6714840.392294\leqslant(\alpha-1)\sqrt{\frac{2}{\pi\nu\tau}}e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\leqslant 0.671484.
Therefore, the absolute difference between these terms is at most 0.983247−0.3922940.9832470.3922940.983247-0.392294
leading to the derived bound.

∎

##### Bounds on mean, variance and second moment.

For deriving bounds on μ~~𝜇{\tilde{\mu}}, ξ~~𝜉{\tilde{\xi}}, and ν~~𝜈{\tilde{\nu}}, we need
the following lemma.

###### Lemma 8 (Derivatives of the Mapping).

We assume α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01} and λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01}.
We restrict the range of the variables to the domain
μ∈[−0.1,0.1]𝜇0.10.1\mu\in[-0.1,0.1],
ω∈[−0.1,0.1]𝜔0.10.1\omega\in[-0.1,0.1],
ν∈[0.8,1.5]𝜈0.81.5\nu\in[0.8,1.5], and
τ∈[0.8,1.25]𝜏0.81.25\tau\in[0.8,1.25].

The derivative ∂∂μ​μ~​(μ,ω,ν,τ,λ,α)𝜇~𝜇𝜇𝜔𝜈𝜏𝜆𝛼\frac{\partial}{\partial\mu}{\tilde{\mu}}(\mu,\omega,\nu,\tau,\lambda,\alpha)
has the sign of ω𝜔\omega.

The derivative ∂∂ν​μ~​(μ,ω,ν,τ,λ,α)𝜈~𝜇𝜇𝜔𝜈𝜏𝜆𝛼\frac{\partial}{\partial\nu}{\tilde{\mu}}(\mu,\omega,\nu,\tau,\lambda,\alpha)
is positive.

The derivative ∂∂μ​ξ~​(μ,ω,ν,τ,λ,α)𝜇~𝜉𝜇𝜔𝜈𝜏𝜆𝛼\frac{\partial}{\partial\mu}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda,\alpha)
has the sign of ω𝜔\omega.

The derivative
∂∂ν​ξ~​(μ,ω,ν,τ,λ,α)𝜈~𝜉𝜇𝜔𝜈𝜏𝜆𝛼\frac{\partial}{\partial\nu}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda,\alpha)
is positive.

###### Proof.

See [40](#Thmtheorem40 "Lemma 40 (Derivatives of the Mapping). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").
∎

###### Lemma 9 (Bounds on mean, variance and second moment).

The expressions μ~~𝜇{\tilde{\mu}}, ξ~~𝜉{\tilde{\xi}}, and ν~~𝜈{\tilde{\nu}}
for
α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01} and λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01}
are bounded by
−0.041160<μ~<0.0876530.041160~𝜇0.087653-0.041160<{\tilde{\mu}}<0.087653,
0.703257<ξ~<1.6437050.703257~𝜉1.6437050.703257<{\tilde{\xi}}<1.643705
and
0.695574<ν~<1.6360230.695574~𝜈1.6360230.695574<{\tilde{\nu}}<1.636023
in the domain μ∈[−0.1,0.1]𝜇0.10.1\mu\in[-0.1,0.1],
ν∈[0.8,15]𝜈0.815\nu\in[0.8,15], ω∈[−0.1,0.1]𝜔0.10.1\omega\in[-0.1,0.1], τ∈[0.8,1.25]𝜏0.81.25\tau\in[0.8,1.25].

###### Proof.

We use Lemma [8](#Thmtheorem8 "Lemma 8 (Derivatives of the Mapping). ‣ Bounds on mean, variance and second moment. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") which states that with given
sign the derivatives of the mapping Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))
and Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) with respect to ν𝜈\nu
and μ𝜇\mu are either positive or have the sign of
ω𝜔\omega.
Therefore with given sign of ω𝜔\omega the mappings are strict monotonic and
the their maxima and minima are found at the borders. The minimum of μ~~𝜇{\tilde{\mu}} is obtained at
μ​ω=−0.01𝜇𝜔0.01\mu\omega=-0.01 and its maximum at μ​ω=0.01𝜇𝜔0.01\mu\omega=0.01 and σ𝜎\sigma and τ𝜏\tau at minimal or maximal values, respectively.
It follows that

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | −0.041160<μ~​(−0.1,0.1,0.8,0.8,λ01,α01)⩽0.041160~𝜇0.10.10.80.8subscript𝜆01subscript𝛼01absent\displaystyle-0.041160<{\tilde{\mu}}(-0.1,0.1,0.8,0.8,\lambda\_{\rm 01},\alpha\_{\rm 01})\leqslant | μ~⩽μ~​(0.1,0.1,1.5,1.25,λ01,α01)<0.087653.~𝜇~𝜇0.10.11.51.25subscript𝜆01subscript𝛼010.087653\displaystyle{\tilde{\mu}}\leqslant{\tilde{\mu}}(0.1,0.1,1.5,1.25,\lambda\_{\rm 01},\alpha\_{\rm 01})<0.087653. |  | (76) |

Similarly, the maximum and minimum of ξ~~𝜉{\tilde{\xi}} is obtained at the values mentioned above:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 0.703257<ξ~​(−0.1,0.1,0.8,0.8,λ01,α01)⩽0.703257~𝜉0.10.10.80.8subscript𝜆01subscript𝛼01absent\displaystyle 0.703257<{\tilde{\xi}}(-0.1,0.1,0.8,0.8,\lambda\_{\rm 01},\alpha\_{\rm 01})\leqslant | ξ~⩽ξ~​(0.1,0.1,1.5,1.25,λ01,α01)<1.643705.~𝜉~𝜉0.10.11.51.25subscript𝜆01subscript𝛼011.643705\displaystyle{\tilde{\xi}}\leqslant{\tilde{\xi}}(0.1,0.1,1.5,1.25,\lambda\_{\rm 01},\alpha\_{\rm 01})<1.643705. |  | (77) |

Hence we obtain the following bounds on ν~~𝜈{\tilde{\nu}}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 0.703257−μ~2<ξ~−μ~20.703257superscript~𝜇2~𝜉superscript~𝜇2\displaystyle 0.703257-{\tilde{\mu}}^{2}<{\tilde{\xi}}-{\tilde{\mu}}^{2} | <1.643705−μ~2absent1.643705superscript~𝜇2\displaystyle<1.643705-{\tilde{\mu}}^{2} |  | (78) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 0.703257−0.007683<ν~0.7032570.007683~𝜈\displaystyle 0.703257-0.007683<{\tilde{\nu}} | <1.643705−0.007682absent1.6437050.007682\displaystyle<1.643705-0.007682 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 0.695574<ν~0.695574~𝜈\displaystyle 0.695574<{\tilde{\nu}} | <1.636023.absent1.636023\displaystyle<1.636023. |  |

∎

##### Upper Bounds on the Largest Singular Value of the Jacobian.

###### Lemma 10 (Upper Bounds on Absolute Derivatives of Largest Singular Value).

We set
α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01} and λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01} and
restrict the range of the variables to
μ∈[μmin,μmax]=[−0.1,0.1]𝜇subscript𝜇minsubscript𝜇max0.10.1\mu\in[\mu\_{\rm min},\mu\_{\rm max}]=[-0.1,0.1],
ω∈[ωmin,ωmax]=[−0.1,0.1]𝜔subscript𝜔minsubscript𝜔max0.10.1\omega\in[\omega\_{\rm min},\omega\_{\rm max}]=[-0.1,0.1],
ν∈[νmin,νmax]=[0.8,1.5]𝜈subscript𝜈minsubscript𝜈max0.81.5\nu\in[\nu\_{\rm min},\nu\_{\rm max}]=[0.8,1.5], and
τ∈[τmin,τmax]=[0.8,1.25]𝜏subscript𝜏minsubscript𝜏max0.81.25\tau\in[\tau\_{\rm min},\tau\_{\rm max}]=[0.8,1.25].

The absolute values of derivatives of the largest singular value
S​(μ,ω,ν,τ,λ,α)𝑆𝜇𝜔𝜈𝜏𝜆𝛼S(\mu,\omega,\nu,\tau,\lambda,\alpha)
given in Eq. ([71](#S3.E71 "In Proof sketch: Bounding the largest singular value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) with respect to
(μ,ω,ν,τ)𝜇𝜔𝜈𝜏(\mu,\omega,\nu,\tau) are bounded as follows:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | |∂S∂μ|𝑆𝜇\displaystyle\left|\frac{\partial S}{\partial\mu}\right|\ | < 0.32112,absent0.32112\displaystyle<\ 0.32112\ , |  | (79) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | |∂S∂ω|𝑆𝜔\displaystyle\left|\frac{\partial S}{\partial\omega}\right|\ | < 2.63690,absent2.63690\displaystyle<\ 2.63690\ , |  | (80) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | |∂S∂ν|𝑆𝜈\displaystyle\left|\frac{\partial S}{\partial\nu}\right|\ | < 2.28242,absent2.28242\displaystyle<\ 2.28242\ , |  | (81) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | |∂S∂τ|𝑆𝜏\displaystyle\left|\frac{\partial S}{\partial\tau}\right|\ | < 2.98610.absent2.98610\displaystyle<\ 2.98610\ . |  | (82) |

###### Proof.

The Jacobian of our mapping Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) and
Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) is defined as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝑯𝑯\displaystyle\bm{H}\ | =(ℋ11ℋ12ℋ21ℋ22)=(𝒥11𝒥12𝒥21−2​μ~​𝒥11𝒥22−2​μ~​𝒥12)absentsubscriptℋ11subscriptℋ12subscriptℋ21subscriptℋ22subscript𝒥11subscript𝒥12subscript𝒥212~𝜇subscript𝒥11subscript𝒥222~𝜇subscript𝒥12\displaystyle=\ \left(\begin{array}[]{cc}{\mathcal{H}}\_{11}&{\mathcal{H}}\_{12}\\ {\mathcal{H}}\_{21}&{\mathcal{H}}\_{22}\\ \end{array}\right)=\ \left(\begin{array}[]{cc}{\mathcal{J}}\_{11}&{\mathcal{J}}\_{12}\\ {\mathcal{J}}\_{21}-2{\tilde{\mu}}{\mathcal{J}}\_{11}&{\mathcal{J}}\_{22}-2{\tilde{\mu}}{\mathcal{J}}\_{12}\\ \end{array}\right) |  | (87) |

and has the largest singular value

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | S​(μ,ω,ν,τ,λ,α)𝑆𝜇𝜔𝜈𝜏𝜆𝛼\displaystyle S(\mu,\omega,\nu,\tau,\lambda,\alpha)\ | =12​((ℋ11−ℋ22)2+(ℋ12+ℋ21)2+(ℋ11+ℋ22)2+(ℋ12−ℋ21)2),absent12superscriptsubscriptℋ11subscriptℋ222superscriptsubscriptℋ12subscriptℋ212superscriptsubscriptℋ11subscriptℋ222superscriptsubscriptℋ12subscriptℋ212\displaystyle=\ \frac{1}{2}\left(\sqrt{({\mathcal{H}}\_{11}-{\mathcal{H}}\_{22})^{2}+({\mathcal{H}}\_{12}+{\mathcal{H}}\_{21})^{2}}+\sqrt{({\mathcal{H}}\_{11}+{\mathcal{H}}\_{22})^{2}+({\mathcal{H}}\_{12}-{\mathcal{H}}\_{21})^{2}}\right), |  | (88) |

according to the formula of Blinn, [[4](#bib.bib4)].

We obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂S∂ℋ11|=|12​(ℋ11−ℋ22(ℋ11−ℋ22)2+(ℋ12+ℋ21)2+ℋ11+ℋ22(ℋ11+ℋ22)2+(ℋ21−ℋ12)2)|<𝑆subscriptℋ1112subscriptℋ11subscriptℋ22superscriptsubscriptℋ11subscriptℋ222superscriptsubscriptℋ12subscriptℋ212subscriptℋ11subscriptℋ22superscriptsubscriptℋ11subscriptℋ222superscriptsubscriptℋ21subscriptℋ122absent\displaystyle\left|\frac{\partial S}{\partial{\mathcal{H}}\_{11}}\right|\ =\ \left|\frac{1}{2}\left(\frac{{\mathcal{H}}\_{11}-{\mathcal{H}}\_{22}}{\sqrt{({\mathcal{H}}\_{11}-{\mathcal{H}}\_{22})^{2}+({\mathcal{H}}\_{12}+{\mathcal{H}}\_{21})^{2}}}+\frac{{\mathcal{H}}\_{11}+{\mathcal{H}}\_{22}}{\sqrt{({\mathcal{H}}\_{11}+{\mathcal{H}}\_{22})^{2}+({\mathcal{H}}\_{21}-{\mathcal{H}}\_{12})^{2}}}\right)\right|\ < |  | (89) |
|  |  |  |
| --- | --- | --- |
|  | 12​(|1(ℋ12+ℋ21)2(ℋ11−ℋ22)2+1|+|1(ℋ21−ℋ12)2(ℋ11+ℋ22)2+1|)<1+12= 1121superscriptsubscriptℋ12subscriptℋ212superscriptsubscriptℋ11subscriptℋ22211superscriptsubscriptℋ21subscriptℋ122superscriptsubscriptℋ11subscriptℋ22211121\displaystyle\frac{1}{2}\left(\left|\frac{1}{\sqrt{\frac{({\mathcal{H}}\_{12}+{\mathcal{H}}\_{21})^{2}}{({\mathcal{H}}\_{11}-{\mathcal{H}}\_{22})^{2}}+1}}\right|+\left|\frac{1}{\sqrt{\frac{({\mathcal{H}}\_{21}-{\mathcal{H}}\_{12})^{2}}{({\mathcal{H}}\_{11}+{\mathcal{H}}\_{22})^{2}}+1}}\right|\right)\ <\ \frac{1+1}{2}\ =\ 1 |  |

and analogously

|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂S∂ℋ12|=|12​(ℋ12+ℋ21(ℋ11−ℋ22)2+(ℋ12+ℋ21)2−ℋ21−ℋ12(ℋ11+ℋ22)2+(ℋ21−ℋ12)2)|< 1𝑆subscriptℋ1212subscriptℋ12subscriptℋ21superscriptsubscriptℋ11subscriptℋ222superscriptsubscriptℋ12subscriptℋ212subscriptℋ21subscriptℋ12superscriptsubscriptℋ11subscriptℋ222superscriptsubscriptℋ21subscriptℋ1221\displaystyle\left|\frac{\partial S}{\partial{\mathcal{H}}\_{12}}\right|\ =\ \left|\frac{1}{2}\left(\frac{{\mathcal{H}}\_{12}+{\mathcal{H}}\_{21}}{\sqrt{({\mathcal{H}}\_{11}-{\mathcal{H}}\_{22})^{2}+({\mathcal{H}}\_{12}+{\mathcal{H}}\_{21})^{2}}}-\frac{{\mathcal{H}}\_{21}-{\mathcal{H}}\_{12}}{\sqrt{({\mathcal{H}}\_{11}+{\mathcal{H}}\_{22})^{2}+({\mathcal{H}}\_{21}-{\mathcal{H}}\_{12})^{2}}}\right)\right|\ <\ 1 |  | (90) |

and

|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂S∂ℋ21|=|12​(ℋ21−ℋ12(ℋ11+ℋ22)2+(ℋ21−ℋ12)2+ℋ12+ℋ21(ℋ11−ℋ22)2+(ℋ12+ℋ21)2)|< 1𝑆subscriptℋ2112subscriptℋ21subscriptℋ12superscriptsubscriptℋ11subscriptℋ222superscriptsubscriptℋ21subscriptℋ122subscriptℋ12subscriptℋ21superscriptsubscriptℋ11subscriptℋ222superscriptsubscriptℋ12subscriptℋ2121\displaystyle\left|\frac{\partial S}{\partial{\mathcal{H}}\_{21}}\right|\ =\ \left|\frac{1}{2}\left(\frac{{\mathcal{H}}\_{21}-{\mathcal{H}}\_{12}}{\sqrt{({\mathcal{H}}\_{11}+{\mathcal{H}}\_{22})^{2}+({\mathcal{H}}\_{21}-{\mathcal{H}}\_{12})^{2}}}+\frac{{\mathcal{H}}\_{12}+{\mathcal{H}}\_{21}}{\sqrt{({\mathcal{H}}\_{11}-{\mathcal{H}}\_{22})^{2}+({\mathcal{H}}\_{12}+{\mathcal{H}}\_{21})^{2}}}\right)\right|\ <\ 1 |  | (91) |

and

|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂S∂ℋ22|=|12​(ℋ11+ℋ22(ℋ11+ℋ22)2+(ℋ21−ℋ12)2−ℋ11−ℋ22(ℋ11−ℋ22)2+(ℋ12+ℋ21)2)|< 1.𝑆subscriptℋ2212subscriptℋ11subscriptℋ22superscriptsubscriptℋ11subscriptℋ222superscriptsubscriptℋ21subscriptℋ122subscriptℋ11subscriptℋ22superscriptsubscriptℋ11subscriptℋ222superscriptsubscriptℋ12subscriptℋ2121\displaystyle\left|\frac{\partial S}{\partial{\mathcal{H}}\_{22}}\right|\ =\ \left|\frac{1}{2}\left(\frac{{\mathcal{H}}\_{11}+{\mathcal{H}}\_{22}}{\sqrt{({\mathcal{H}}\_{11}+{\mathcal{H}}\_{22})^{2}+({\mathcal{H}}\_{21}-{\mathcal{H}}\_{12})^{2}}}-\frac{{\mathcal{H}}\_{11}-{\mathcal{H}}\_{22}}{\sqrt{({\mathcal{H}}\_{11}-{\mathcal{H}}\_{22})^{2}+({\mathcal{H}}\_{12}+{\mathcal{H}}\_{21})^{2}}}\right)\right|\ <\ 1\ . |  | (92) |

We have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂S∂μ𝑆𝜇\displaystyle\frac{\partial S}{\partial\mu}\ | =∂S∂ℋ11​∂ℋ11∂μ+∂S∂ℋ12​∂ℋ12∂μ+∂S∂ℋ21​∂ℋ21∂μ+∂S∂ℋ22​∂ℋ22∂μabsent𝑆subscriptℋ11subscriptℋ11𝜇𝑆subscriptℋ12subscriptℋ12𝜇𝑆subscriptℋ21subscriptℋ21𝜇𝑆subscriptℋ22subscriptℋ22𝜇\displaystyle=\ \frac{\partial S}{\partial{\mathcal{H}}\_{11}}\frac{\partial{\mathcal{H}}\_{11}}{\partial\mu}\ +\ \frac{\partial S}{\partial{\mathcal{H}}\_{12}}\frac{\partial{\mathcal{H}}\_{12}}{\partial\mu}\ +\ \frac{\partial S}{\partial{\mathcal{H}}\_{21}}\frac{\partial{\mathcal{H}}\_{21}}{\partial\mu}\ +\ \frac{\partial S}{\partial{\mathcal{H}}\_{22}}\frac{\partial{\mathcal{H}}\_{22}}{\partial\mu} |  | (93) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂S∂ω𝑆𝜔\displaystyle\frac{\partial S}{\partial\omega}\ | =∂S∂ℋ11​∂ℋ11∂ω+∂S∂ℋ12​∂ℋ12∂ω+∂S∂ℋ21​∂ℋ21∂ω+∂S∂ℋ22​∂ℋ22∂ωabsent𝑆subscriptℋ11subscriptℋ11𝜔𝑆subscriptℋ12subscriptℋ12𝜔𝑆subscriptℋ21subscriptℋ21𝜔𝑆subscriptℋ22subscriptℋ22𝜔\displaystyle=\ \frac{\partial S}{\partial{\mathcal{H}}\_{11}}\frac{\partial{\mathcal{H}}\_{11}}{\partial\omega}\ +\ \frac{\partial S}{\partial{\mathcal{H}}\_{12}}\frac{\partial{\mathcal{H}}\_{12}}{\partial\omega}\ +\ \frac{\partial S}{\partial{\mathcal{H}}\_{21}}\frac{\partial{\mathcal{H}}\_{21}}{\partial\omega}\ +\ \frac{\partial S}{\partial{\mathcal{H}}\_{22}}\frac{\partial{\mathcal{H}}\_{22}}{\partial\omega} |  | (94) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂S∂ν𝑆𝜈\displaystyle\frac{\partial S}{\partial\nu}\ | =∂S∂ℋ11​∂ℋ11∂ν+∂S∂ℋ12​∂ℋ12∂ν+∂S∂ℋ21​∂ℋ21∂ν+∂S∂ℋ22​∂ℋ22∂νabsent𝑆subscriptℋ11subscriptℋ11𝜈𝑆subscriptℋ12subscriptℋ12𝜈𝑆subscriptℋ21subscriptℋ21𝜈𝑆subscriptℋ22subscriptℋ22𝜈\displaystyle=\ \frac{\partial S}{\partial{\mathcal{H}}\_{11}}\frac{\partial{\mathcal{H}}\_{11}}{\partial\nu}\ +\ \frac{\partial S}{\partial{\mathcal{H}}\_{12}}\frac{\partial{\mathcal{H}}\_{12}}{\partial\nu}\ +\ \frac{\partial S}{\partial{\mathcal{H}}\_{21}}\frac{\partial{\mathcal{H}}\_{21}}{\partial\nu}\ +\ \frac{\partial S}{\partial{\mathcal{H}}\_{22}}\frac{\partial{\mathcal{H}}\_{22}}{\partial\nu} |  | (95) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂S∂τ𝑆𝜏\displaystyle\frac{\partial S}{\partial\tau}\ | =∂S∂ℋ11​∂ℋ11∂τ+∂S∂ℋ12​∂ℋ12∂τ+∂S∂ℋ21​∂ℋ21∂τ+∂S∂ℋ22​∂ℋ22∂τabsent𝑆subscriptℋ11subscriptℋ11𝜏𝑆subscriptℋ12subscriptℋ12𝜏𝑆subscriptℋ21subscriptℋ21𝜏𝑆subscriptℋ22subscriptℋ22𝜏\displaystyle=\ \frac{\partial S}{\partial{\mathcal{H}}\_{11}}\frac{\partial{\mathcal{H}}\_{11}}{\partial\tau}\ +\ \frac{\partial S}{\partial{\mathcal{H}}\_{12}}\frac{\partial{\mathcal{H}}\_{12}}{\partial\tau}\ +\ \frac{\partial S}{\partial{\mathcal{H}}\_{21}}\frac{\partial{\mathcal{H}}\_{21}}{\partial\tau}\ +\ \frac{\partial S}{\partial{\mathcal{H}}\_{22}}\frac{\partial{\mathcal{H}}\_{22}}{\partial\tau} |  | (96) |

from which follows using the bounds from Lemma [5](#Thmtheorem5 "Lemma 5 (Bounds on the Derivatives). ‣ Bounds on the derivatives of the Jacobian entries. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"):

Derivative of the singular value w.r.t. μ𝜇\mu:

|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂S∂μ|⩽𝑆𝜇absent\displaystyle\left|\frac{\partial S}{\partial\mu}\right|\ \leqslant |  | (98) |
|  |  |  |
| --- | --- | --- |
|  | |∂S∂ℋ11|​|∂ℋ11∂μ|+|∂S∂ℋ12|​|∂ℋ12∂μ|+|∂S∂ℋ21|​|∂ℋ21∂μ|+|∂S∂ℋ22|​|∂ℋ22∂μ|⩽𝑆subscriptℋ11subscriptℋ11𝜇𝑆subscriptℋ12subscriptℋ12𝜇𝑆subscriptℋ21subscriptℋ21𝜇𝑆subscriptℋ22subscriptℋ22𝜇absent\displaystyle\left|\frac{\partial S}{\partial{\mathcal{H}}\_{11}}\right|\left|\frac{\partial{\mathcal{H}}\_{11}}{\partial\mu}\right|+\left|\frac{\partial S}{\partial{\mathcal{H}}\_{12}}\right|\left|\frac{\partial{\mathcal{H}}\_{12}}{\partial\mu}\right|+\left|\frac{\partial S}{\partial{\mathcal{H}}\_{21}}\right|\left|\frac{\partial{\mathcal{H}}\_{21}}{\partial\mu}\right|+\left|\frac{\partial S}{\partial{\mathcal{H}}\_{22}}\right|\left|\frac{\partial{\mathcal{H}}\_{22}}{\partial\mu}\right|\ \leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | |∂ℋ11∂μ|+|∂ℋ12∂μ|+|∂ℋ21∂μ|+|∂ℋ22∂μ|⩽subscriptℋ11𝜇subscriptℋ12𝜇subscriptℋ21𝜇subscriptℋ22𝜇absent\displaystyle\left|\frac{\partial{\mathcal{H}}\_{11}}{\partial\mu}\right|+\left|\frac{\partial{\mathcal{H}}\_{12}}{\partial\mu}\right|+\left|\frac{\partial{\mathcal{H}}\_{21}}{\partial\mu}\right|+\left|\frac{\partial{\mathcal{H}}\_{22}}{\partial\mu}\right|\ \leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | |∂𝒥11∂μ|+|∂𝒥12∂μ|+|∂𝒥21−2​μ~​𝒥11∂μ|+|∂𝒥22−2​μ~​𝒥12∂μ|⩽subscript𝒥11𝜇subscript𝒥12𝜇subscript𝒥212~𝜇subscript𝒥11𝜇subscript𝒥222~𝜇subscript𝒥12𝜇absent\displaystyle\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\mu}\right|+\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\mu}\right|+\left|\frac{\partial{\mathcal{J}}\_{21}-2{\tilde{\mu}}{\mathcal{J}}\_{11}}{\partial\mu}\right|+\left|\frac{\partial{\mathcal{J}}\_{22}-2{\tilde{\mu}}{\mathcal{J}}\_{12}}{\partial\mu}\right|\ \leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | |∂𝒥11∂μ|+|∂𝒥12∂μ|+|∂𝒥21∂μ|+|∂𝒥22∂μ|+2​|∂𝒥11∂μ|​|μ~|+2​|𝒥11|2+2​|∂𝒥12∂μ|​|μ~|+2​|𝒥12|​|𝒥11|⩽subscript𝒥11𝜇subscript𝒥12𝜇subscript𝒥21𝜇subscript𝒥22𝜇2subscript𝒥11𝜇~𝜇2superscriptsubscript𝒥1122subscript𝒥12𝜇~𝜇2subscript𝒥12subscript𝒥11absent\displaystyle\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\mu}\right|+\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\mu}\right|+\left|\frac{\partial{\mathcal{J}}\_{21}}{\partial\mu}\right|+\left|\frac{\partial{\mathcal{J}}\_{22}}{\partial\mu}\right|+2\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\mu}\right|\left|{\tilde{\mu}}\right|+2\left|\mathcal{J}\_{11}\right|^{2}+2\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\mu}\right|\left|{\tilde{\mu}}\right|+2\left|\mathcal{J}\_{12}\right|\left|\mathcal{J}\_{11}\right|\ \leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | 0.0031049101995398316+0.031242911235461816+0.02220441024325437+0.14983446469110305+0.00310491019953983160.0312429112354618160.02220441024325437limit-from0.14983446469110305\displaystyle 0.0031049101995398316+0.031242911235461816+0.02220441024325437+0.14983446469110305+ |  |
|  |  |  |
| --- | --- | --- |
|  | 2⋅0.104497⋅0.087653+2⋅0.1044972+⋅20.1044970.087653limit-from⋅2superscript0.1044972\displaystyle 2\cdot 0.104497\cdot 0.087653+2\cdot 0.104497^{2}+ |  |
|  |  |  |
| --- | --- | --- |
|  | 2⋅0.194035⋅0.087653+2⋅0.104497⋅0.194035< 0.32112,⋅20.1940350.087653⋅20.1044970.1940350.32112\displaystyle 2\cdot 0.194035\cdot 0.087653+2\cdot 0.104497\cdot 0.194035<\ 0.32112, |  |

where we used the results from the lemmata [5](#Thmtheorem5 "Lemma 5 (Bounds on the Derivatives). ‣ Bounds on the derivatives of the Jacobian entries. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), [6](#Thmtheorem6 "Lemma 6 (Bound on J11). ‣ Bounds on the entries of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), [7](#Thmtheorem7 "Lemma 7 (Bound on J12). ‣ Bounds on the entries of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), and [9](#Thmtheorem9 "Lemma 9 (Bounds on mean, variance and second moment). ‣ Bounds on mean, variance and second moment. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").

Derivative of the singular value w.r.t. ω𝜔\omega:

|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂S∂ω|⩽𝑆𝜔absent\displaystyle\left|\frac{\partial S}{\partial\omega}\right|\ \leqslant |  | (99) |
|  |  |  |
| --- | --- | --- |
|  | |∂S∂ℋ11|​|∂ℋ11∂ω|+|∂S∂ℋ12|​|∂ℋ12∂ω|+|∂S∂ℋ21|​|∂ℋ21∂ω|+|∂S∂ℋ22|​|∂ℋ22∂ω|⩽𝑆subscriptℋ11subscriptℋ11𝜔𝑆subscriptℋ12subscriptℋ12𝜔𝑆subscriptℋ21subscriptℋ21𝜔𝑆subscriptℋ22subscriptℋ22𝜔absent\displaystyle\left|\frac{\partial S}{\partial{\mathcal{H}}\_{11}}\right|\left|\frac{\partial{\mathcal{H}}\_{11}}{\partial\omega}\right|+\left|\frac{\partial S}{\partial{\mathcal{H}}\_{12}}\right|\left|\frac{\partial{\mathcal{H}}\_{12}}{\partial\omega}\right|+\left|\frac{\partial S}{\partial{\mathcal{H}}\_{21}}\right|\left|\frac{\partial{\mathcal{H}}\_{21}}{\partial\omega}\right|+\left|\frac{\partial S}{\partial{\mathcal{H}}\_{22}}\right|\left|\frac{\partial{\mathcal{H}}\_{22}}{\partial\omega}\right|\ \leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | |∂ℋ11∂ω|+|∂ℋ12∂ω|+|∂ℋ21∂ω|+|∂ℋ22∂ω|⩽subscriptℋ11𝜔subscriptℋ12𝜔subscriptℋ21𝜔subscriptℋ22𝜔absent\displaystyle\left|\frac{\partial{\mathcal{H}}\_{11}}{\partial\omega}\right|+\left|\frac{\partial{\mathcal{H}}\_{12}}{\partial\omega}\right|+\left|\frac{\partial{\mathcal{H}}\_{21}}{\partial\omega}\right|+\left|\frac{\partial{\mathcal{H}}\_{22}}{\partial\omega}\right|\ \leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | |∂𝒥11∂ω|+|∂𝒥12∂ω|+|∂𝒥21−2​μ~​𝒥11∂ω|+|∂𝒥22−2​μ~​𝒥12∂ω|⩽subscript𝒥11𝜔subscript𝒥12𝜔subscript𝒥212~𝜇subscript𝒥11𝜔subscript𝒥222~𝜇subscript𝒥12𝜔absent\displaystyle\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\omega}\right|+\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\omega}\right|+\left|\frac{\partial{\mathcal{J}}\_{21}-2{\tilde{\mu}}{\mathcal{J}}\_{11}}{\partial\omega}\right|+\left|\frac{\partial{\mathcal{J}}\_{22}-2{\tilde{\mu}}{\mathcal{J}}\_{12}}{\partial\omega}\right|\ \leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | |∂𝒥11∂ω|+|∂𝒥12∂ω|+|∂𝒥21∂ω|+|∂𝒥22∂ω|+2​|∂𝒥11∂ω|​|μ~|+2​|𝒥11|​|∂μ~∂ω|+subscript𝒥11𝜔subscript𝒥12𝜔subscript𝒥21𝜔subscript𝒥22𝜔2subscript𝒥11𝜔~𝜇limit-from2subscript𝒥11~𝜇𝜔\displaystyle\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\omega}\right|+\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\omega}\right|+\left|\frac{\partial{\mathcal{J}}\_{21}}{\partial\omega}\right|+\left|\frac{\partial{\mathcal{J}}\_{22}}{\partial\omega}\right|+2\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\omega}\right|\left|{\tilde{\mu}}\right|+2\left|\mathcal{J}\_{11}\right|\left|\frac{\partial{\tilde{\mu}}}{\partial\omega}\right|+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 2​|∂𝒥12∂ω|​|μ~|+2​|𝒥12|​|∂μ~∂ω|⩽2subscript𝒥12𝜔~𝜇2subscript𝒥12~𝜇𝜔absent\displaystyle\left.2\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\omega}\right|\left|{\tilde{\mu}}\right|+2\left|\mathcal{J}\_{12}\right|\left|\frac{\partial{\tilde{\mu}}}{\partial\omega}\right|\right.\ \leqslant |  | (100) |
|  |  |  |
| --- | --- | --- |
|  | 2.38392+2⋅1.055872374194189⋅0.087653+2⋅0.1044972+2⋅0.031242911235461816⋅0.0876532.38392⋅21.0558723741941890.087653⋅2superscript0.1044972⋅20.0312429112354618160.087653\displaystyle 2.38392+2\cdot 1.055872374194189\cdot 0.087653+2\cdot 0.104497^{2}+2\cdot 0.031242911235461816\cdot 0.087653 |  |
|  |  |  |
| --- | --- | --- |
|  | +2⋅0.194035⋅0.104497< 2.63690,⋅20.1940350.1044972.63690\displaystyle+2\cdot 0.194035\cdot 0.104497\ <\ 2.63690\ , |  |

where we used the results from the lemmata [5](#Thmtheorem5 "Lemma 5 (Bounds on the Derivatives). ‣ Bounds on the derivatives of the Jacobian entries. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), [6](#Thmtheorem6 "Lemma 6 (Bound on J11). ‣ Bounds on the entries of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), [7](#Thmtheorem7 "Lemma 7 (Bound on J12). ‣ Bounds on the entries of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), and [9](#Thmtheorem9 "Lemma 9 (Bounds on mean, variance and second moment). ‣ Bounds on mean, variance and second moment. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") and that
μ~~𝜇{\tilde{\mu}} is symmetric for μ,ω

𝜇𝜔\mu,\omega.

Derivative of the singular value w.r.t. ν𝜈\nu:

|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂S∂ν|⩽𝑆𝜈absent\displaystyle\left|\frac{\partial S}{\partial\nu}\right|\ \leqslant |  | (101) |
|  |  |  |
| --- | --- | --- |
|  | |∂S∂ℋ11|​|∂ℋ11∂ν|+|∂S∂ℋ12|​|∂ℋ12∂ν|+|∂S∂ℋ21|​|∂ℋ21∂ν|+|∂S∂ℋ22|​|∂ℋ22∂ν|⩽𝑆subscriptℋ11subscriptℋ11𝜈𝑆subscriptℋ12subscriptℋ12𝜈𝑆subscriptℋ21subscriptℋ21𝜈𝑆subscriptℋ22subscriptℋ22𝜈absent\displaystyle\left|\frac{\partial S}{\partial{\mathcal{H}}\_{11}}\right|\left|\frac{\partial{\mathcal{H}}\_{11}}{\partial\nu}\right|+\left|\frac{\partial S}{\partial{\mathcal{H}}\_{12}}\right|\left|\frac{\partial{\mathcal{H}}\_{12}}{\partial\nu}\right|+\left|\frac{\partial S}{\partial{\mathcal{H}}\_{21}}\right|\left|\frac{\partial{\mathcal{H}}\_{21}}{\partial\nu}\right|+\left|\frac{\partial S}{\partial{\mathcal{H}}\_{22}}\right|\left|\frac{\partial{\mathcal{H}}\_{22}}{\partial\nu}\right|\ \leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | |∂ℋ11∂ν|+|∂ℋ12∂ν|+|∂ℋ21∂ν|+|∂ℋ22∂ν|⩽subscriptℋ11𝜈subscriptℋ12𝜈subscriptℋ21𝜈subscriptℋ22𝜈absent\displaystyle\left|\frac{\partial{\mathcal{H}}\_{11}}{\partial\nu}\right|+\left|\frac{\partial{\mathcal{H}}\_{12}}{\partial\nu}\right|+\left|\frac{\partial{\mathcal{H}}\_{21}}{\partial\nu}\right|+\left|\frac{\partial{\mathcal{H}}\_{22}}{\partial\nu}\right|\ \leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | |∂𝒥11∂ν|+|∂𝒥12∂ν|+|∂𝒥21−2​μ~​𝒥11∂ν|+|∂𝒥22−2​μ~​𝒥12∂ν|⩽subscript𝒥11𝜈subscript𝒥12𝜈subscript𝒥212~𝜇subscript𝒥11𝜈subscript𝒥222~𝜇subscript𝒥12𝜈absent\displaystyle\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\nu}\right|+\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\nu}\right|+\left|\frac{\partial{\mathcal{J}}\_{21}-2{\tilde{\mu}}{\mathcal{J}}\_{11}}{\partial\nu}\right|+\left|\frac{\partial{\mathcal{J}}\_{22}-2{\tilde{\mu}}{\mathcal{J}}\_{12}}{\partial\nu}\right|\ \leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | |∂𝒥11∂ν|+|∂𝒥12∂ν|+|∂𝒥21∂ν|+|∂𝒥22∂ν|+2​|∂𝒥11∂ν|​|μ~|+2​|𝒥11|​|𝒥12|+2​|∂𝒥12∂ν|​|μ~|+2​|𝒥12|2⩽subscript𝒥11𝜈subscript𝒥12𝜈subscript𝒥21𝜈subscript𝒥22𝜈2subscript𝒥11𝜈~𝜇2subscript𝒥11subscript𝒥122subscript𝒥12𝜈~𝜇2superscriptsubscript𝒥122absent\displaystyle\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\nu}\right|+\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\nu}\right|+\left|\frac{\partial{\mathcal{J}}\_{21}}{\partial\nu}\right|+\left|\frac{\partial{\mathcal{J}}\_{22}}{\partial\nu}\right|+2\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\nu}\right|\left|{\tilde{\mu}}\right|+2\left|\mathcal{J}\_{11}\right|\left|\mathcal{J}\_{12}\right|+2\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\nu}\right|\left|{\tilde{\mu}}\right|+2\left|\mathcal{J}\_{12}\right|^{2}\ \leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | 2.19916+2⋅0.031242911235461816⋅0.087653+2⋅0.104497⋅0.194035+2.19916⋅20.0312429112354618160.087653limit-from⋅20.1044970.194035\displaystyle 2.19916+2\cdot 0.031242911235461816\cdot 0.087653+2\cdot 0.104497\cdot 0.194035+ |  |
|  |  |  |
| --- | --- | --- |
|  | 2⋅0.21232788238624354⋅0.087653+2⋅0.1940352< 2.28242,⋅20.212327882386243540.087653⋅2superscript0.19403522.28242\displaystyle 2\cdot 0.21232788238624354\cdot 0.087653+2\cdot 0.194035^{2}\ <\ 2.28242\ , |  |

where we used the results from the lemmata [5](#Thmtheorem5 "Lemma 5 (Bounds on the Derivatives). ‣ Bounds on the derivatives of the Jacobian entries. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), [6](#Thmtheorem6 "Lemma 6 (Bound on J11). ‣ Bounds on the entries of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), [7](#Thmtheorem7 "Lemma 7 (Bound on J12). ‣ Bounds on the entries of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), and [9](#Thmtheorem9 "Lemma 9 (Bounds on mean, variance and second moment). ‣ Bounds on mean, variance and second moment. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").

Derivative of the singular value w.r.t. τ𝜏\tau:

|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂S∂τ|⩽𝑆𝜏absent\displaystyle\left|\frac{\partial S}{\partial\tau}\right|\ \leqslant |  | (102) |
|  |  |  |
| --- | --- | --- |
|  | |∂S∂ℋ11|​|∂ℋ11∂τ|+|∂S∂ℋ12|​|∂ℋ12∂τ|+|∂S∂ℋ21|​|∂ℋ21∂τ|+|∂S∂ℋ22|​|∂ℋ22∂τ|⩽𝑆subscriptℋ11subscriptℋ11𝜏𝑆subscriptℋ12subscriptℋ12𝜏𝑆subscriptℋ21subscriptℋ21𝜏𝑆subscriptℋ22subscriptℋ22𝜏absent\displaystyle\left|\frac{\partial S}{\partial{\mathcal{H}}\_{11}}\right|\left|\frac{\partial{\mathcal{H}}\_{11}}{\partial\tau}\right|+\left|\frac{\partial S}{\partial{\mathcal{H}}\_{12}}\right|\left|\frac{\partial{\mathcal{H}}\_{12}}{\partial\tau}\right|+\left|\frac{\partial S}{\partial{\mathcal{H}}\_{21}}\right|\left|\frac{\partial{\mathcal{H}}\_{21}}{\partial\tau}\right|+\left|\frac{\partial S}{\partial{\mathcal{H}}\_{22}}\right|\left|\frac{\partial{\mathcal{H}}\_{22}}{\partial\tau}\right|\ \leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | |∂ℋ11∂τ|+|∂ℋ12∂τ|+|∂ℋ21∂τ|+|∂ℋ22∂τ|⩽subscriptℋ11𝜏subscriptℋ12𝜏subscriptℋ21𝜏subscriptℋ22𝜏absent\displaystyle\left|\frac{\partial{\mathcal{H}}\_{11}}{\partial\tau}\right|+\left|\frac{\partial{\mathcal{H}}\_{12}}{\partial\tau}\right|+\left|\frac{\partial{\mathcal{H}}\_{21}}{\partial\tau}\right|+\left|\frac{\partial{\mathcal{H}}\_{22}}{\partial\tau}\right|\ \leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | |∂𝒥11∂τ|+|∂𝒥12∂τ|+|∂𝒥21−2​μ~​𝒥11∂τ|+|∂𝒥22−2​μ~​𝒥12∂τ|⩽subscript𝒥11𝜏subscript𝒥12𝜏subscript𝒥212~𝜇subscript𝒥11𝜏subscript𝒥222~𝜇subscript𝒥12𝜏absent\displaystyle\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\tau}\right|+\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\tau}\right|+\left|\frac{\partial{\mathcal{J}}\_{21}-2{\tilde{\mu}}{\mathcal{J}}\_{11}}{\partial\tau}\right|+\left|\frac{\partial{\mathcal{J}}\_{22}-2{\tilde{\mu}}{\mathcal{J}}\_{12}}{\partial\tau}\right|\ \leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | |∂𝒥11∂τ|+|∂𝒥12∂τ|+|∂𝒥21∂τ|+|∂𝒥22∂τ|+2​|∂𝒥11∂τ|​|μ~|+2​|𝒥11|​|∂μ~∂τ|+subscript𝒥11𝜏subscript𝒥12𝜏subscript𝒥21𝜏subscript𝒥22𝜏2subscript𝒥11𝜏~𝜇limit-from2subscript𝒥11~𝜇𝜏\displaystyle\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\tau}\right|+\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\tau}\right|+\left|\frac{\partial{\mathcal{J}}\_{21}}{\partial\tau}\right|+\left|\frac{\partial{\mathcal{J}}\_{22}}{\partial\tau}\right|+2\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\tau}\right|\left|{\tilde{\mu}}\right|+2\left|\mathcal{J}\_{11}\right|\left|\frac{\partial{\tilde{\mu}}}{\partial\tau}\right|+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 2​|∂𝒥12∂τ|​|μ~|+2​|𝒥12|​|∂μ~∂τ|⩽2subscript𝒥12𝜏~𝜇2subscript𝒥12~𝜇𝜏absent\displaystyle\left.2\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\tau}\right|\left|{\tilde{\mu}}\right|+2\left|\mathcal{J}\_{12}\right|\left|\frac{\partial{\tilde{\mu}}}{\partial\tau}\right|\right.\ \leqslant |  | (103) |
|  |  |  |
| --- | --- | --- |
|  | 2.82643+2⋅0.03749149348255419⋅0.087653+2⋅0.104497⋅0.194035+2.82643⋅20.037491493482554190.087653limit-from⋅20.1044970.194035\displaystyle 2.82643+2\cdot 0.03749149348255419\cdot 0.087653+2\cdot 0.104497\cdot 0.194035+ |  |
|  |  |  |
| --- | --- | --- |
|  | 2⋅0.2124377655377270⋅0.087653+2⋅0.1940352< 2.98610,⋅20.21243776553772700.087653⋅2superscript0.19403522.98610\displaystyle 2\cdot 0.2124377655377270\cdot 0.087653+2\cdot 0.194035^{2}\ <\ 2.98610\ , |  |

where we used the results from the lemmata [5](#Thmtheorem5 "Lemma 5 (Bounds on the Derivatives). ‣ Bounds on the derivatives of the Jacobian entries. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), [6](#Thmtheorem6 "Lemma 6 (Bound on J11). ‣ Bounds on the entries of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), [7](#Thmtheorem7 "Lemma 7 (Bound on J12). ‣ Bounds on the entries of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), and [9](#Thmtheorem9 "Lemma 9 (Bounds on mean, variance and second moment). ‣ Bounds on mean, variance and second moment. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") and that
μ~~𝜇{\tilde{\mu}} is symmetric for ν,τ

𝜈𝜏\nu,\tau.

∎

###### Lemma 11 (Mean Value Theorem Bound on Deviation from Largest Singular Value).

We set
α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01} and λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01} and
restrict the range of the variables to
μ∈[μmin,μmax]=[−0.1,0.1]𝜇subscript𝜇minsubscript𝜇max0.10.1\mu\in[\mu\_{\rm min},\mu\_{\rm max}]=[-0.1,0.1],
ω∈[ωmin,ωmax]=[−0.1,0.1]𝜔subscript𝜔minsubscript𝜔max0.10.1\omega\in[\omega\_{\rm min},\omega\_{\rm max}]=[-0.1,0.1],
ν∈[νmin,νmax]=[0.8,1.5]𝜈subscript𝜈minsubscript𝜈max0.81.5\nu\in[\nu\_{\rm min},\nu\_{\rm max}]=[0.8,1.5], and
τ∈[τmin,τmax]=[0.8,1.25]𝜏subscript𝜏minsubscript𝜏max0.81.25\tau\in[\tau\_{\rm min},\tau\_{\rm max}]=[0.8,1.25].

The distance of the singular value at
S​(μ,ω,ν,τ,λ01,α01)𝑆𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01S(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})
and that at
S​(μ+Δ​μ,ω+Δ​ω,ν+Δ​ν,τ+Δ​τ,λ01,α01)𝑆𝜇Δ𝜇𝜔Δ𝜔𝜈Δ𝜈𝜏Δ𝜏subscript𝜆01subscript𝛼01S(\mu+\Delta\mu,\omega+\Delta\omega,\nu+\Delta\nu,\tau+\Delta\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})
is bounded as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | |S​(μ+Δ​μ,ω+Δ​ω,ν+Δ​ν,τ+Δ​τ,λ01,α01)−S​(μ,ω,ν,τ,λ01,α01)|<𝑆𝜇Δ𝜇𝜔Δ𝜔𝜈Δ𝜈𝜏Δ𝜏subscript𝜆01subscript𝛼01𝑆𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01absent\displaystyle\left|S(\mu+\Delta\mu,\omega+\Delta\omega,\nu+\Delta\nu,\tau+\Delta\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ -\ S(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\right|\ < |  | (104) |
|  |  |  |
| --- | --- | --- |
|  | 0.32112​|Δ​μ|+2.63690​|Δ​ω|+2.28242​|Δ​ν|+2.98610​|Δ​τ|.0.32112Δ𝜇2.63690Δ𝜔2.28242Δ𝜈2.98610Δ𝜏\displaystyle 0.32112\left|\Delta\mu\right|+2.63690\left|\Delta\omega\right|+2.28242\left|\Delta\nu\right|+2.98610\left|\Delta\tau\right|\ . |  |

###### Proof.

The mean value theorem states that a t∈[0,1]𝑡01t\in[0,1] exists for
which

|  |  |  |  |
| --- | --- | --- | --- |
|  | S​(μ+Δ​μ,ω+Δ​ω,ν+Δ​ν,τ+Δ​τ,λ01,α01)−S​(μ,ω,ν,τ,λ01,α01)=𝑆𝜇Δ𝜇𝜔Δ𝜔𝜈Δ𝜈𝜏Δ𝜏subscript𝜆01subscript𝛼01𝑆𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01absent\displaystyle S(\mu+\Delta\mu,\omega+\Delta\omega,\nu+\Delta\nu,\tau+\Delta\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ -\ S(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ = |  | (105) |
|  |  |  |
| --- | --- | --- |
|  | ∂S∂μ​(μ+t​Δ​μ,ω+t​Δ​ω,ν+t​Δ​ν,τ+t​Δ​τ,λ01,α01)​Δ​μ+limit-from𝑆𝜇𝜇𝑡Δ𝜇𝜔𝑡Δ𝜔𝜈𝑡Δ𝜈𝜏𝑡Δ𝜏subscript𝜆01subscript𝛼01Δ𝜇\displaystyle\frac{\partial S}{\partial\mu}(\mu+t\Delta\mu,\omega+t\Delta\omega,\nu+t\Delta\nu,\tau+t\Delta\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ \Delta\mu\ + |  |
|  |  |  |
| --- | --- | --- |
|  | ∂S∂ω​(μ+t​Δ​μ,ω+t​Δ​ω,ν+t​Δ​ν,τ+t​Δ​τ,λ01,α01)​Δ​ω+limit-from𝑆𝜔𝜇𝑡Δ𝜇𝜔𝑡Δ𝜔𝜈𝑡Δ𝜈𝜏𝑡Δ𝜏subscript𝜆01subscript𝛼01Δ𝜔\displaystyle\frac{\partial S}{\partial\omega}(\mu+t\Delta\mu,\omega+t\Delta\omega,\nu+t\Delta\nu,\tau+t\Delta\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ \Delta\omega\ + |  |
|  |  |  |
| --- | --- | --- |
|  | ∂S∂ν​(μ+t​Δ​μ,ω+t​Δ​ω,ν+t​Δ​ν,τ+t​Δ​τ,λ01,α01)​Δ​ν+limit-from𝑆𝜈𝜇𝑡Δ𝜇𝜔𝑡Δ𝜔𝜈𝑡Δ𝜈𝜏𝑡Δ𝜏subscript𝜆01subscript𝛼01Δ𝜈\displaystyle\frac{\partial S}{\partial\nu}(\mu+t\Delta\mu,\omega+t\Delta\omega,\nu+t\Delta\nu,\tau+t\Delta\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ \Delta\nu\ + |  |
|  |  |  |
| --- | --- | --- |
|  | ∂S∂τ​(μ+t​Δ​μ,ω+t​Δ​ω,ν+t​Δ​ν,τ+t​Δ​τ,λ01,α01)​Δ​τ𝑆𝜏𝜇𝑡Δ𝜇𝜔𝑡Δ𝜔𝜈𝑡Δ𝜈𝜏𝑡Δ𝜏subscript𝜆01subscript𝛼01Δ𝜏\displaystyle\frac{\partial S}{\partial\tau}(\mu+t\Delta\mu,\omega+t\Delta\omega,\nu+t\Delta\nu,\tau+t\Delta\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ \Delta\tau |  |

from which immediately follows that

|  |  |  |  |
| --- | --- | --- | --- |
|  | |S​(μ+Δ​μ,ω+Δ​ω,ν+Δ​ν,τ+Δ​τ,λ01,α01)−S​(μ,ω,ν,τ,λ01,α01)|⩽𝑆𝜇Δ𝜇𝜔Δ𝜔𝜈Δ𝜈𝜏Δ𝜏subscript𝜆01subscript𝛼01𝑆𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01absent\displaystyle\left|S(\mu+\Delta\mu,\omega+\Delta\omega,\nu+\Delta\nu,\tau+\Delta\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ -\ S(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\right|\ \leqslant |  | (106) |
|  |  |  |
| --- | --- | --- |
|  | |∂S∂μ​(μ+t​Δ​μ,ω+t​Δ​ω,ν+t​Δ​ν,τ+t​Δ​τ,λ01,α01)|​|Δ​μ|+limit-from𝑆𝜇𝜇𝑡Δ𝜇𝜔𝑡Δ𝜔𝜈𝑡Δ𝜈𝜏𝑡Δ𝜏subscript𝜆01subscript𝛼01Δ𝜇\displaystyle\left|\frac{\partial S}{\partial\mu}(\mu+t\Delta\mu,\omega+t\Delta\omega,\nu+t\Delta\nu,\tau+t\Delta\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\right|\ \left|\Delta\mu\right|\ + |  |
|  |  |  |
| --- | --- | --- |
|  | |∂S∂ω​(μ+t​Δ​μ,ω+t​Δ​ω,ν+t​Δ​ν,τ+t​Δ​τ,λ01,α01)|​|Δ​ω|+limit-from𝑆𝜔𝜇𝑡Δ𝜇𝜔𝑡Δ𝜔𝜈𝑡Δ𝜈𝜏𝑡Δ𝜏subscript𝜆01subscript𝛼01Δ𝜔\displaystyle\left|\frac{\partial S}{\partial\omega}(\mu+t\Delta\mu,\omega+t\Delta\omega,\nu+t\Delta\nu,\tau+t\Delta\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\right|\ \left|\Delta\omega\right|\ + |  |
|  |  |  |
| --- | --- | --- |
|  | |∂S∂ν​(μ+t​Δ​μ,ω+t​Δ​ω,ν+t​Δ​ν,τ+t​Δ​τ,λ01,α01)|​|Δ​ν|+limit-from𝑆𝜈𝜇𝑡Δ𝜇𝜔𝑡Δ𝜔𝜈𝑡Δ𝜈𝜏𝑡Δ𝜏subscript𝜆01subscript𝛼01Δ𝜈\displaystyle\left|\frac{\partial S}{\partial\nu}(\mu+t\Delta\mu,\omega+t\Delta\omega,\nu+t\Delta\nu,\tau+t\Delta\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\right|\ \left|\Delta\nu\right|\ + |  |
|  |  |  |
| --- | --- | --- |
|  | |∂S∂τ​(μ+t​Δ​μ,ω+t​Δ​ω,ν+t​Δ​ν,τ+t​Δ​τ,λ01,α01)|​|Δ​τ|.𝑆𝜏𝜇𝑡Δ𝜇𝜔𝑡Δ𝜔𝜈𝑡Δ𝜈𝜏𝑡Δ𝜏subscript𝜆01subscript𝛼01Δ𝜏\displaystyle\left|\frac{\partial S}{\partial\tau}(\mu+t\Delta\mu,\omega+t\Delta\omega,\nu+t\Delta\nu,\tau+t\Delta\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\right|\ \left|\Delta\tau\right|\ . |  |

We now apply Lemma [10](#Thmtheorem10 "Lemma 10 (Upper Bounds on Absolute Derivatives of Largest Singular Value). ‣ Upper Bounds on the Largest Singular Value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") which gives bounds on the
derivatives, which immediately gives the statement of the lemma.
∎

###### Lemma 12 (Largest Singular Value Smaller Than One).

We set
α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01} and λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01} and
restrict the range of the variables to
μ∈[−0.1,0.1]𝜇0.10.1\mu\in[-0.1,0.1],
ω∈[−0.1,0.1]𝜔0.10.1\omega\in[-0.1,0.1],
ν∈[0.8,1.5]𝜈0.81.5\nu\in[0.8,1.5], and
τ∈[0.8,1.25]𝜏0.81.25\tau\in[0.8,1.25].

The the largest singular value of the Jacobian is smaller than 1:

|  |  |  |  |
| --- | --- | --- | --- |
|  | S​(μ,ω,ν,τ,λ01,α01)< 1.𝑆𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼011\displaystyle S(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ <\ 1\ . |  | (107) |

Therefore the mapping Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))
and Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) is a contraction mapping.

###### Proof.

We set
Δ​μ=0.0068097371Δ𝜇0.0068097371\Delta\mu=0.0068097371,
Δ​ω=0.0008292885Δ𝜔0.0008292885\Delta\omega=0.0008292885,
Δ​ν=0.0009580840Δ𝜈0.0009580840\Delta\nu=0.0009580840, and
Δ​τ=0.0007323095Δ𝜏0.0007323095\Delta\tau=0.0007323095.

According to Lemma [11](#Thmtheorem11 "Lemma 11 (Mean Value Theorem Bound on Deviation from Largest Singular Value). ‣ Upper Bounds on the Largest Singular Value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | |S​(μ+Δ​μ,ω+Δ​ω,ν+Δ​ν,τ+Δ​τ,λ01,α01)−S​(μ,ω,ν,τ,λ01,α01)|<𝑆𝜇Δ𝜇𝜔Δ𝜔𝜈Δ𝜈𝜏Δ𝜏subscript𝜆01subscript𝛼01𝑆𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01absent\displaystyle\left|S(\mu+\Delta\mu,\omega+\Delta\omega,\nu+\Delta\nu,\tau+\Delta\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ -\ S(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\right|\ < |  | (108) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 0.32112⋅0.0068097371+2.63690⋅0.0008292885+⋅0.321120.0068097371limit-from⋅2.636900.0008292885\displaystyle 0.32112\cdot 0.0068097371+2.63690\cdot 0.0008292885+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2.28242⋅0.0009580840+2.98610⋅0.0007323095< 0.008747.⋅2.282420.0009580840⋅2.986100.00073230950.008747\displaystyle 2.28242\cdot 0.0009580840+2.98610\cdot 0.0007323095\ <\ 0.008747\ . |  |

For a grid with grid length
Δ​μ=0.0068097371Δ𝜇0.0068097371\Delta\mu=0.0068097371,
Δ​ω=0.0008292885Δ𝜔0.0008292885\Delta\omega=0.0008292885,
Δ​ν=0.0009580840Δ𝜈0.0009580840\Delta\nu=0.0009580840, and
Δ​τ=0.0007323095Δ𝜏0.0007323095\Delta\tau=0.0007323095,
we evaluated the function Eq. ([71](#S3.E71 "In Proof sketch: Bounding the largest singular value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"))
for the largest singular value
in the domain
μ∈[−0.1,0.1]𝜇0.10.1\mu\in[-0.1,0.1],
ω∈[−0.1,0.1]𝜔0.10.1\omega\in[-0.1,0.1],
ν∈[0.8,1.5]𝜈0.81.5\nu\in[0.8,1.5], and
τ∈[0.8,1.25]𝜏0.81.25\tau\in[0.8,1.25].
We did this using a computer.
According to Subsection [A3.4.5](#S3.SS4.SSS5 "A3.4.5 Computer-assisted proof details for main Lemma 12 in Section A3.4.1. ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
the precision if regarding error propagation
and precision of the implemented functions is larger than
10−13superscript101310^{-13}.
We performed the evaluation on different operating systems and
different hardware architectures including CPUs and GPUs.
In all cases the function Eq. ([71](#S3.E71 "In Proof sketch: Bounding the largest singular value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) for the largest singular
value of the Jacobian is bounded by 0.99125241710587720.99125241710587720.9912524171058772.

We obtain from Eq. ([108](#S3.E108 "In Proof. ‣ Upper Bounds on the Largest Singular Value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | S​(μ+Δ​μ,ω+Δ​ω,ν+Δ​ν,τ+Δ​τ,λ01,α01)⩽ 0.9912524171058772+ 0.008747< 1.𝑆𝜇Δ𝜇𝜔Δ𝜔𝜈Δ𝜈𝜏Δ𝜏subscript𝜆01subscript𝛼010.99125241710587720.0087471\displaystyle S(\mu+\Delta\mu,\omega+\Delta\omega,\nu+\Delta\nu,\tau+\Delta\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ \leqslant\ 0.9912524171058772\ +\ 0.008747\ <\ 1\ . |  | (109) |

∎

#### A3.4.2 Lemmata for proofing Theorem 1 (part 2): Mapping within domain

We further have to investigate whether the the mapping Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))
and Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) maps into a predefined domains.

###### Lemma 13 (Mapping into the domain).

The mapping Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))
and Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) map for
α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01} and λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01}
into the domain
μ∈[−0.03106,0.06773]𝜇0.031060.06773\mu\in[-0.03106,0.06773] and
ν∈[0.80009,1.48617]𝜈0.800091.48617\nu\in[0.80009,1.48617] with ω∈[−0.1,0.1]𝜔0.10.1\omega\in[-0.1,0.1] and τ∈[0.95,1.1]𝜏0.951.1\tau\in[0.95,1.1].

###### Proof.

We use Lemma [8](#Thmtheorem8 "Lemma 8 (Derivatives of the Mapping). ‣ Bounds on mean, variance and second moment. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") which states that with given
sign the derivatives of the mapping Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))
and Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")) with respect to α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01}
and λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01} are either positive or have the sign of
ω𝜔\omega.
Therefore with given sign of ω𝜔\omega the mappings are strict monotonic and
the their maxima and minima are found at the borders. The minimum of μ~~𝜇{\tilde{\mu}} is obtained at
μ​ω=−0.01𝜇𝜔0.01\mu\omega=-0.01 and its maximum at μ​ω=0.01𝜇𝜔0.01\mu\omega=0.01 and σ𝜎\sigma and τ𝜏\tau at their
minimal and maximal values, respectively. It follows that:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | −0.03106<μ~​(−0.1,0.1,0.8,0.95,λ01,α01)⩽0.03106~𝜇0.10.10.80.95subscript𝜆01subscript𝛼01absent\displaystyle-0.03106<{\tilde{\mu}}(-0.1,0.1,0.8,0.95,\lambda\_{\rm 01},\alpha\_{\rm 01})\leqslant | μ~⩽μ~​(0.1,0.1,1.5,1.1,λ01,α01)<0.06773,~𝜇~𝜇0.10.11.51.1subscript𝜆01subscript𝛼010.06773\displaystyle{\tilde{\mu}}\leqslant{\tilde{\mu}}(0.1,0.1,1.5,1.1,\lambda\_{\rm 01},\alpha\_{\rm 01})<0.06773, |  | (110) |

and that μ~∈[−0.1,0.1]~𝜇0.10.1{\tilde{\mu}}\in[-0.1,0.1].

Similarly, the maximum and minimum of ξ~({\tilde{\xi}}( is obtained at the values mentioned above:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 0.80467<ξ~​(−0.1,0.1,0.8,0.95,λ01,α01)⩽0.80467~𝜉0.10.10.80.95subscript𝜆01subscript𝛼01absent\displaystyle 0.80467<{\tilde{\xi}}(-0.1,0.1,0.8,0.95,\lambda\_{\rm 01},\alpha\_{\rm 01})\leqslant | ξ~⩽ξ~​(0.1,0.1,1.5,1.1,λ01,α01)<1.48617.~𝜉~𝜉0.10.11.51.1subscript𝜆01subscript𝛼011.48617\displaystyle{\tilde{\xi}}\leqslant{\tilde{\xi}}(0.1,0.1,1.5,1.1,\lambda\_{\rm 01},\alpha\_{\rm 01})<1.48617. |  | (111) |

Since |ξ~−ν~|=|μ~2|<0.004597~𝜉~𝜈superscript~𝜇20.004597|{\tilde{\xi}}-{\tilde{\nu}}|=|{\tilde{\mu}}^{2}|<0.004597, we can conclude that
0.80009<ν~<1.486170.80009~𝜈1.486170.80009<{\tilde{\nu}}<1.48617 and the variance remains in [0.8,1.5]0.81.5[0.8,1.5].
∎

###### Corollary 14.

The image g​(Ω′)𝑔superscriptΩ′g(\Omega^{\prime}) of the mapping g:(μ,ν)↦(μ~,ν~):𝑔maps-to𝜇𝜈~𝜇~𝜈g:(\mu,\nu)\mapsto({\tilde{\mu}},{\tilde{\nu}}) (Eq. ([8](#S1.E8 "In A1 Background ‣ Self-Normalizing Neural Networks")))
and the domain Ω′={(μ,ν)|−0.1⩽μ⩽0.1,0.8⩽μ⩽1.5}superscriptΩ′conditional-set𝜇𝜈formulae-sequence0.1𝜇0.10.8𝜇1.5\Omega^{\prime}=\{(\mu,\nu)|-0.1\leqslant\mu\leqslant 0.1,0.8\leqslant\mu\leqslant 1.5\} is
a subset of Ω′superscriptΩ′\Omega^{\prime}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | g​(Ω′)⊆Ω′,𝑔superscriptΩ′superscriptΩ′\displaystyle g(\Omega^{\prime})\subseteq\Omega^{\prime}, |  | (112) |

for all ω∈[−0.1,0.1]𝜔0.10.1\omega\in[-0.1,0.1] and τ∈[0.95,1.1]𝜏0.951.1\tau\in[0.95,1.1].

###### Proof.

Directly follows from Lemma [13](#Thmtheorem13 "Lemma 13 (Mapping into the domain). ‣ A3.4.2 Lemmata for proofing Theorem 1 (part 2): Mapping within domain ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").
∎

#### A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting

##### Main Sub-Function.

We consider the main sub-function of the derivate of second moment, J​22𝐽22J22 (Eq. ([62](#S3.E62 "In Jacobian of the mapping. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"))):

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂∂ν​ξ~=12​λ2​τ​(−α2​eμ​ω+ν​τ2​erfc⁡(μ​ω+ν​τ2​ν​τ)+2​α2​e2​μ​ω+2​ν​τ​erfc⁡(μ​ω+2​ν​τ2​ν​τ)−erfc⁡(μ​ω2​ν​τ)+2)𝜈~𝜉12superscript𝜆2𝜏superscript𝛼2superscript𝑒𝜇𝜔𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏2superscript𝛼2superscript𝑒2𝜇𝜔2𝜈𝜏erfc𝜇𝜔2𝜈𝜏2𝜈𝜏erfc𝜇𝜔2𝜈𝜏2\displaystyle\frac{\partial}{\partial\nu}{\tilde{\xi}}=\frac{1}{2}\lambda^{2}\tau\left(-\alpha^{2}e^{\mu\omega+\frac{\nu\tau}{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+2\alpha^{2}e^{2\mu\omega+2\nu\tau}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+2\right) |  | (113) |

that depends on μ​ω𝜇𝜔\mu\omega and ν​τ𝜈𝜏\nu\tau, therefore we
set x=ν​τ𝑥𝜈𝜏x=\nu\tau and y=μ​ω𝑦𝜇𝜔y=\mu\omega. Algebraic reformulations provide the
formula in the following form:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂∂ν​ξ~=12​λ2​τ​(α2​(−e−y22​x)​(e(x+y)22​x​erfc⁡(y+x2​x)−2​e(2​x+y)22​x​erfc⁡(y+2​x2​x))−erfc⁡(y2​x)+2)𝜈~𝜉12superscript𝜆2𝜏superscript𝛼2superscript𝑒superscript𝑦22𝑥superscript𝑒superscript𝑥𝑦22𝑥erfc𝑦𝑥2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥erfc𝑦2𝑥2𝑥erfc𝑦2𝑥2\displaystyle\frac{\partial}{\partial\nu}{\tilde{\xi}}\ =\frac{1}{2}\lambda^{2}\tau\left(\alpha^{2}\left(-e^{-\frac{y^{2}}{2x}}\right)\left(e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{y+x}{\sqrt{2}\sqrt{x}}\right)-2e^{\frac{(2x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{y+2x}{\sqrt{2}\sqrt{x}}\right)\right)-\operatorname{erfc}\left(\frac{y}{\sqrt{2}\sqrt{x}}\right)+2\right) |  | (114) |

For λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01} and
α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01},
we consider the domain
−1⩽μ⩽11𝜇1-1\leqslant\mu\leqslant 1,
−0.1⩽ω⩽0.10.1𝜔0.1-0.1\leqslant\omega\leqslant 0.1,
1.5⩽ν⩽161.5𝜈161.5\leqslant\nu\leqslant 16, and,
0.8⩽τ⩽1.250.8𝜏1.250.8\leqslant\tau\leqslant 1.25.

For x𝑥x and y𝑦y we obtain: 0.8⋅1.5=1.2⩽x⩽20=1.25⋅16⋅0.81.51.2𝑥20⋅1.25160.8\cdot 1.5=1.2\leqslant x\leqslant 20=1.25\cdot 16 and
0.1⋅(−1)=−0.1⩽y⩽0.1=0.1⋅1⋅0.110.1𝑦0.1⋅0.110.1\cdot(-1)=-0.1\leqslant y\leqslant 0.1=0.1\cdot 1.
In the following we assume to remain within this domain.

###### Lemma 15 (Main subfunction).

For 1.2⩽x⩽201.2𝑥201.2\leqslant x\leqslant 20 and −0.1⩽y⩽0.10.1𝑦0.1-0.1\leqslant y\leqslant 0.1,

the function

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(x+y)22​x​erfc⁡(x+y2​x)−2​e(2​x+y)22​x​erfc⁡(2​x+y2​x)superscript𝑒superscript𝑥𝑦22𝑥erfc𝑥𝑦2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥erfc2𝑥𝑦2𝑥\displaystyle e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)-2e^{\frac{(2x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right) |  | (115) |

is smaller than zero, is strictly monotonically increasing in x𝑥x,
and strictly monotonically decreasing in y𝑦y for the minimal x=12/10=1.2𝑥12101.2x=12/10=1.2.

###### Proof.

See proof [44](#Thmtheorem44 "Lemma 44 (Main subfunction). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").
∎

The graph of the subfunction in the specified domain is displayed in Figure [A3](#S3.F3 "Figure A3 ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").

![Refer to caption](/html/1706.02515/assets/x4.png)

![Refer to caption](/html/1706.02515/assets/x5.png)

Figure A3: Left panel: Graphs of the main subfunction f​(x,y)=e(x+y)22​x​erfc⁡(x+y2​x)−2​e(2​x+y)22​x​erfc⁡(2​x+y2​x)𝑓𝑥𝑦superscript𝑒superscript𝑥𝑦22𝑥erfc𝑥𝑦2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥erfc2𝑥𝑦2𝑥f(x,y)=e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)-2e^{\frac{(2x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)
treated in Lemma [15](#Thmtheorem15 "Lemma 15 (Main subfunction). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"). The function is negative and monotonically increasing with x𝑥x independent of y𝑦y.
Right panel: Graphs of the main subfunction at minimal x=1.2𝑥1.2x=1.2. The graph shows that the function f​(1.2,y)𝑓1.2𝑦f(1.2,y) is strictly monotonically decreasing in y𝑦y.

###### Theorem 16 (Contraction ν𝜈\nu-mapping).

The mapping of the variance ν~​(μ,ω,ν,τ,λ,α)~𝜈𝜇𝜔𝜈𝜏𝜆𝛼{\tilde{\nu}}(\mu,\omega,\nu,\tau,\lambda,\alpha) given in Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))
is contracting for
λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01}, α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01}
and the domain Ω+superscriptΩ\Omega^{+}:
−0.1⩽μ⩽0.10.1𝜇0.1-0.1\leqslant\mu\leqslant 0.1,
−0.1⩽ω⩽0.10.1𝜔0.1-0.1\leqslant\omega\leqslant 0.1,
1.5⩽ν⩽161.5𝜈161.5\leqslant\nu\leqslant 16, and
0.8⩽τ⩽1.250.8𝜏1.250.8\leqslant\tau\leqslant 1.25, that is,

|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂∂ν​ν~​(μ,ω,ν,τ,λ01,α01)|< 1.𝜈~𝜈𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼011\displaystyle\left|\frac{\partial}{\partial\nu}{\tilde{\nu}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\right|\ <\ 1\ . |  | (116) |

###### Proof.

In this domain Ω+superscriptΩ\Omega^{+} we have the following three properties (see further below):
∂∂ν​ξ~<1𝜈~𝜉1\frac{\partial}{\partial\nu}{\tilde{\xi}}<1, μ~>0~𝜇0{\tilde{\mu}}>0,
and ∂∂ν​μ~>0𝜈~𝜇0\frac{\partial}{\partial\nu}{\tilde{\mu}}>0. Therefore, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂∂ν​ν~|=|∂∂ν​ξ~−2​μ~​∂∂ν​μ~|<|∂∂ν​ξ~|<1𝜈~𝜈𝜈~𝜉2~𝜇𝜈~𝜇𝜈~𝜉1\displaystyle\left|\frac{\partial}{\partial\nu}{\tilde{\nu}}\right|=\left|\frac{\partial}{\partial\nu}{\tilde{\xi}}-2{\tilde{\mu}}\frac{\partial}{\partial\nu}{\tilde{\mu}}\right|<\left|\frac{\partial}{\partial\nu}{\tilde{\xi}}\right|<1 |  | (117) |

* •

  We first proof that ∂∂ν​ξ~<1𝜈~𝜉1\frac{\partial}{\partial\nu}{\tilde{\xi}}<1 in an even larger domain that fully contains Ω+superscriptΩ\Omega^{+}.
  According to Eq. ([62](#S3.E62 "In Jacobian of the mapping. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")),
  the derivative of the mapping Eq. ([5](#Sx2.E5 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"))
  with respect to the variance ν𝜈\nu is

  |  |  |  |  |  |
  | --- | --- | --- | --- | --- |
  |  |  | ∂∂ν​ξ~​(μ,ω,ν,τ,λ01,α01)=𝜈~𝜉𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01absent\displaystyle\frac{\partial}{\partial\nu}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda\_{\rm 01},\alpha\_{\rm 01})\ = |  | (118) |
  |  |  |  |  |
  | --- | --- | --- | --- |
  |  |  | 12λ2τ(α2(−eμ​ω+ν​τ2)erfc(μ​ω+ν​τ2​ν​τ)+\displaystyle\frac{1}{2}\lambda^{2}\tau\left(\alpha^{2}\left(-e^{\mu\omega+\frac{\nu\tau}{2}}\right)\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+\right. |  |
  |  |  |  |  |
  | --- | --- | --- | --- |
  |  |  | 2α2e2​μ​ω+2​ν​τerfc(μ​ω+2​ν​τ2​ν​τ)−erfc(μ​ω2​ν​τ)+2).\displaystyle\left.2\alpha^{2}e^{2\mu\omega+2\nu\tau}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+2\right)\ . |  |

  For
  λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01}, α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01},
  −1⩽μ⩽11𝜇1-1\leqslant\mu\leqslant 1,
  −0.1⩽ω⩽0.10.1𝜔0.1-0.1\leqslant\omega\leqslant 0.1
  1.5⩽ν⩽161.5𝜈161.5\leqslant\nu\leqslant 16, and
  0.8⩽τ⩽1.250.8𝜏1.250.8\leqslant\tau\leqslant 1.25, we first show that the derivative is positive
  and then upper bound it.

  According to Lemma [15](#Thmtheorem15 "Lemma 15 (Main subfunction). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), the expression

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | e(μ​ω+ν​τ)22​ν​τ​erfc⁡(μ​ω+ν​τ2​ν​τ)−2​e(μ​ω+2​ν​τ)22​ν​τ​erfc⁡(μ​ω+2​ν​τ2​ν​τ)superscript𝑒superscript𝜇𝜔𝜈𝜏22𝜈𝜏erfc𝜇𝜔𝜈𝜏2𝜈𝜏2superscript𝑒superscript𝜇𝜔2𝜈𝜏22𝜈𝜏erfc𝜇𝜔2𝜈𝜏2𝜈𝜏\displaystyle e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-2e^{\frac{(\mu\omega+2\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right) |  | (119) |

  is negative. This expression multiplied by positive factors is
  subtracted in the derivative Eq. ([118](#S3.E118 "In 1st item ‣ Proof. ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")), therefore, the
  whole term is positive.
  The remaining term

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 2−erfc⁡(μ​ω2​ν​τ)2erfc𝜇𝜔2𝜈𝜏\displaystyle 2-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right) |  | (120) |

  of the derivative Eq. ([118](#S3.E118 "In 1st item ‣ Proof. ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"))
  is also positive according to Lemma [21](#Thmtheorem21 "Lemma 21 (Basic functions). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").
  All factors outside the brackets in Eq. ([118](#S3.E118 "In 1st item ‣ Proof. ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) are
  positive. Hence, the derivative Eq. ([118](#S3.E118 "In 1st item ‣ Proof. ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) is positive.

  The upper bound of the derivative is:

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 12λ012τ(α012(−eμ​ω+ν​τ2)erfc(μ​ω+ν​τ2​ν​τ)+\displaystyle\frac{1}{2}\lambda\_{\rm 01}^{2}\tau\left(\alpha\_{\rm 01}^{2}\left(-e^{\mu\omega+\frac{\nu\tau}{2}}\right)\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+\right. |  | (121) |
  |  |  |  |
  | --- | --- | --- |
  |  | 2α012e2​μ​ω+2​ν​τerfc(μ​ω+2​ν​τ2​ν​τ)−erfc(μ​ω2​ν​τ)+2)=\displaystyle\left.2\alpha\_{\rm 01}^{2}e^{2\mu\omega+2\nu\tau}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+2\right)\ = |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 12λ012τ(α012(−e−μ2​ω22​ν​τ)(e(μ​ω+ν​τ)22​ν​τerfc(μ​ω+ν​τ2​ν​τ)−\displaystyle\frac{1}{2}\lambda\_{\rm 01}^{2}\tau\left(\alpha\_{\rm 01}^{2}\left(-e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\right)\left(e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-\right.\right. |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 2e(μ​ω+2​ν​τ)22​ν​τerfc(μ​ω+2​ν​τ2​ν​τ))−erfc(μ​ω2​ν​τ)+2)⩽\displaystyle\left.\left.2e^{\frac{(\mu\omega+2\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+2\right)\ \leqslant |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 121.25λ012(α012(−e−μ2​ω22​ν​τ)(e(μ​ω+ν​τ)22​ν​τerfc(μ​ω+ν​τ2​ν​τ)−\displaystyle\frac{1}{2}1.25\lambda\_{\rm 01}^{2}\left(\alpha\_{\rm 01}^{2}\left(-e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\right)\left(e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-\right.\right. |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 2e(μ​ω+2​ν​τ)22​ν​τerfc(μ​ω+2​ν​τ2​ν​τ))−erfc(μ​ω2​ν​τ)+2)⩽\displaystyle\left.\left.2e^{\frac{(\mu\omega+2\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+2\right)\ \leqslant |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 121.25λ012(α012(e(1.2+0.12​1.2)2erfc(1.2+0.12​1.2)−\displaystyle\frac{1}{2}1.25\lambda\_{\rm 01}^{2}\left(\alpha\_{\rm 01}^{2}\left(e^{\left(\frac{1.2+0.1}{\sqrt{2}\sqrt{1.2}}\right)^{2}}\operatorname{erfc}\left(\frac{1.2+0.1}{\sqrt{2}\sqrt{1.2}}\right)-\right.\right. |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 2e(2⋅1.2+0.12​1.2)2erfc(2⋅1.2+0.12​1.2))(−e−μ2​ω22​ν​τ)−erfc(μ​ω2​ν​τ)+2)⩽\displaystyle\left.\left.2e^{\left(\frac{2\cdot 1.2+0.1}{\sqrt{2}\sqrt{1.2}}\right)^{2}}\operatorname{erfc}\left(\frac{2\cdot 1.2+0.1}{\sqrt{2}\sqrt{1.2}}\right)\right)\left(-e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\right)-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+2\right)\ \leqslant |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 121.25λ012(−e0.0α012(e(1.2+0.12​1.2)2erfc(1.2+0.12​1.2)−\displaystyle\frac{1}{2}1.25\lambda\_{\rm 01}^{2}\left(-e^{0.0}\alpha\_{\rm 01}^{2}\left(e^{\left(\frac{1.2+0.1}{\sqrt{2}\sqrt{1.2}}\right)^{2}}\operatorname{erfc}\left(\frac{1.2+0.1}{\sqrt{2}\sqrt{1.2}}\right)-\right.\right. |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 2e(2⋅1.2+0.12​1.2)2erfc(2⋅1.2+0.12​1.2))−erfc(μ​ω2​ν​τ)+2)⩽\displaystyle\left.\left.2e^{\left(\frac{2\cdot 1.2+0.1}{\sqrt{2}\sqrt{1.2}}\right)^{2}}\operatorname{erfc}\left(\frac{2\cdot 1.2+0.1}{\sqrt{2}\sqrt{1.2}}\right)\right)-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+2\right)\ \leqslant |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 121.25λ012(−e0.0α012(e(1.2+0.12​1.2)2erfc(1.2+0.12​1.2)−\displaystyle\frac{1}{2}1.25\lambda\_{\rm 01}^{2}\left(-e^{0.0}\alpha\_{\rm 01}^{2}\left(e^{\left(\frac{1.2+0.1}{\sqrt{2}\sqrt{1.2}}\right)^{2}}\operatorname{erfc}\left(\frac{1.2+0.1}{\sqrt{2}\sqrt{1.2}}\right)-\right.\right. |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 2e(2⋅1.2+0.12​1.2)2erfc(2⋅1.2+0.12​1.2))−erfc(0.12​1.2)+2)⩽\displaystyle\left.\left.2e^{\left(\frac{2\cdot 1.2+0.1}{\sqrt{2}\sqrt{1.2}}\right)^{2}}\operatorname{erfc}\left(\frac{2\cdot 1.2+0.1}{\sqrt{2}\sqrt{1.2}}\right)\right)-\operatorname{erfc}\left(\frac{0.1}{\sqrt{2}\sqrt{1.2}}\right)+2\right)\ \leqslant |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 0.995063< 1.0.9950631\displaystyle 0.995063\ <\ 1\ . |  |

  We explain the chain of inequalities:

  + –

    First equality brings the expression
    into a shape where we can apply Lemma [15](#Thmtheorem15 "Lemma 15 (Main subfunction). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") for the
    the function Eq. ([115](#S3.E115 "In Lemma 15 (Main subfunction). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")).
  + –

    First inequality: The overall factor τ𝜏\tau is bounded by 1.25.
  + –

    Second inequality: We apply Lemma [15](#Thmtheorem15 "Lemma 15 (Main subfunction). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").
    According to Lemma [15](#Thmtheorem15 "Lemma 15 (Main subfunction). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") the function Eq. ([115](#S3.E115 "In Lemma 15 (Main subfunction). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) is negative.
    The largest contribution is to subtract the most negative value of
    the function Eq. ([115](#S3.E115 "In Lemma 15 (Main subfunction). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")), that is, the minimum of
    function Eq. ([115](#S3.E115 "In Lemma 15 (Main subfunction). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")).
    According to Lemma [15](#Thmtheorem15 "Lemma 15 (Main subfunction). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") the function
    Eq. ([115](#S3.E115 "In Lemma 15 (Main subfunction). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) is strictly monotonically increasing in x𝑥x
    and strictly monotonically decreasing in y𝑦y for x=1.2𝑥1.2x=1.2.
    Therefore the function Eq. ([115](#S3.E115 "In Lemma 15 (Main subfunction). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) has its minimum
    at minimal x=ν​τ=1.5⋅0.8=1.2𝑥𝜈𝜏⋅1.50.81.2x=\nu\tau=1.5\cdot 0.8=1.2
    and maximal y=μ​ω=1.0⋅0.1=0.1𝑦𝜇𝜔⋅1.00.10.1y=\mu\omega=1.0\cdot 0.1=0.1. We insert these values into
    the expression.
  + –

    Third inequality: We use for the whole expression
    the maximal factor
    e−μ2​ω22​ν​τ<1superscript𝑒superscript𝜇2superscript𝜔22𝜈𝜏1e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}<1 by setting this
    factor to 1.
  + –

    Fourth inequality: erfcerfc\operatorname{erfc} is strictly monotonically
    decreasing. Therefore we maximize its argument to obtain the least
    value which is subtracted. We use the minimal x=ν​τ=1.5⋅0.8=1.2𝑥𝜈𝜏⋅1.50.81.2x=\nu\tau=1.5\cdot 0.8=1.2 and the maximal y=μ​ω=1.0⋅0.1=0.1𝑦𝜇𝜔⋅1.00.10.1y=\mu\omega=1.0\cdot 0.1=0.1.
  + –

    Sixth inequality: evaluation of the terms.
* •

  We now show that μ~>0~𝜇0{\tilde{\mu}}>0. The expression
  μ~​(μ,ω,ν,τ)~𝜇𝜇𝜔𝜈𝜏{\tilde{\mu}}(\mu,\omega,\nu,\tau) (Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")))
  is strictly monotonically increasing im μ​ω𝜇𝜔\mu\omega and ν​τ𝜈𝜏\nu\tau. Therefore,
  the minimal value in Ω+superscriptΩ\Omega^{+} is obtained at
  μ~​(0.01,0.01,1.5,0.8)=0.008293>0~𝜇0.010.011.50.80.0082930{\tilde{\mu}}(0.01,0.01,1.5,0.8)=0.008293>0.
* •

  Last we show that ∂∂ν​μ~>0𝜈~𝜇0\frac{\partial}{\partial\nu}{\tilde{\mu}}>0.
  The expression
  ∂∂ν​μ~​(μ,ω,ν,τ)=𝒥12​(μ,ω,ν,τ)𝜈~𝜇𝜇𝜔𝜈𝜏subscript𝒥12𝜇𝜔𝜈𝜏\frac{\partial}{\partial\nu}{\tilde{\mu}}(\mu,\omega,\nu,\tau)={\mathcal{J}}\_{12}(\mu,\omega,\nu,\tau) (Eq. ([62](#S3.E62 "In Jacobian of the mapping. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")))
  can we reformulated as follows:

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 𝒥12​(μ,ω,ν,τ,λ,α)=λ​τ​e−μ2​ω22​ν​τ​(π​α​e(μ​ω+ν​τ)22​ν​τ​erfc⁡(μ​ω+ν​τ2​ν​τ)−2​(α−1)ν​τ)4​πsubscript𝒥12𝜇𝜔𝜈𝜏𝜆𝛼𝜆𝜏superscript𝑒superscript𝜇2superscript𝜔22𝜈𝜏𝜋𝛼superscript𝑒superscript𝜇𝜔𝜈𝜏22𝜈𝜏erfc𝜇𝜔𝜈𝜏2𝜈𝜏2𝛼1𝜈𝜏4𝜋\displaystyle{\mathcal{J}}\_{12}(\mu,\omega,\nu,\tau,\lambda,\alpha)\ =\frac{\lambda\tau e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\left(\sqrt{\pi}\alpha e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-\frac{\sqrt{2}(\alpha-1)}{\sqrt{\nu\tau}}\right)}{4\sqrt{\pi}} |  | (122) |

  is larger than zero when the term π​α​e(μ​ω+ν​τ)22​ν​τ​erfc⁡(μ​ω+ν​τ2​ν​τ)−2​(α−1)ν​τ𝜋𝛼superscript𝑒superscript𝜇𝜔𝜈𝜏22𝜈𝜏erfc𝜇𝜔𝜈𝜏2𝜈𝜏2𝛼1𝜈𝜏\sqrt{\pi}\alpha e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-\frac{\sqrt{2}(\alpha-1)}{\sqrt{\nu\tau}}
  is larger than zero. This term obtains its minimal value
  at μ​ω=0.01𝜇𝜔0.01\mu\omega=0.01 and ν​τ=16⋅1.25𝜈𝜏⋅161.25\nu\tau=16\cdot 1.25, which can easily be shown using the
  Abramowitz bounds (Lemma [22](#Thmtheorem22 "Lemma 22 (Erfc bound from Abramowitz). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"))
  and evaluates to 0.160.160.16, therefore 𝒥12>0subscript𝒥120{\mathcal{J}}\_{12}>0 in Ω+superscriptΩ\Omega^{+}.

∎

#### A3.4.4 Lemmata for proofing Theorem 3: The variance is expanding

##### Main Sub-Function From Below.

We consider functions in
μ​ω𝜇𝜔\mu\omega and ν​τ𝜈𝜏\nu\tau, therefore we
set x=μ​ω𝑥𝜇𝜔x=\mu\omega and y=ν​τ𝑦𝜈𝜏y=\nu\tau.

For λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01} and
α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01},
we consider the domain
−0.1⩽μ⩽0.10.1𝜇0.1-0.1\leqslant\mu\leqslant 0.1,
−0.1⩽ω⩽0.10.1𝜔0.1-0.1\leqslant\omega\leqslant 0.1
0.00875⩽ν⩽0.70.00875𝜈0.70.00875\leqslant\nu\leqslant 0.7, and
0.8⩽τ⩽1.250.8𝜏1.250.8\leqslant\tau\leqslant 1.25.

For x𝑥x and y𝑦y we obtain: 0.8⋅0.00875=0.007⩽x⩽0.875=1.25⋅0.7⋅0.80.008750.007𝑥0.875⋅1.250.70.8\cdot 0.00875=0.007\leqslant x\leqslant 0.875=1.25\cdot 0.7 and
0.1⋅(−0.1)=−0.01⩽y⩽0.01=0.1⋅0.1⋅0.10.10.01𝑦0.01⋅0.10.10.1\cdot(-0.1)=-0.01\leqslant y\leqslant 0.01=0.1\cdot 0.1.
In the following we assume to be within this domain.

In this domain, we consider the main sub-function of the derivate of second moment in the next layer, J​22𝐽22J22 (Eq. ([62](#S3.E62 "In Jacobian of the mapping. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"))):

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂∂ν​ξ~=12​λ2​τ​(−α2​eμ​ω+ν​τ2​erfc⁡(μ​ω+ν​τ2​ν​τ)+2​α2​e2​μ​ω+2​ν​τ​erfc⁡(μ​ω+2​ν​τ2​ν​τ)−erfc⁡(μ​ω2​ν​τ)+2)𝜈~𝜉12superscript𝜆2𝜏superscript𝛼2superscript𝑒𝜇𝜔𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏2superscript𝛼2superscript𝑒2𝜇𝜔2𝜈𝜏erfc𝜇𝜔2𝜈𝜏2𝜈𝜏erfc𝜇𝜔2𝜈𝜏2\displaystyle\frac{\partial}{\partial\nu}{\tilde{\xi}}=\frac{1}{2}\lambda^{2}\tau\left(-\alpha^{2}e^{\mu\omega+\frac{\nu\tau}{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+2\alpha^{2}e^{2\mu\omega+2\nu\tau}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+2\right) |  | (123) |

that depends on μ​ω𝜇𝜔\mu\omega and ν​τ𝜈𝜏\nu\tau, therefore we
set x=ν​τ𝑥𝜈𝜏x=\nu\tau and y=μ​ω𝑦𝜇𝜔y=\mu\omega. Algebraic reformulations provide the
formula in the following form:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂∂ν​ξ~=𝜈~𝜉absent\displaystyle\frac{\partial}{\partial\nu}{\tilde{\xi}}\ = |  | (124) |
|  |  |  |
| --- | --- | --- |
|  | 12​λ2​τ​(α2​(−e−y22​x)​(e(x+y)22​x​erfc⁡(y+x2​x)−2​e(2​x+y)22​x​erfc⁡(y+2​x2​x))−erfc⁡(y2​x)+2)12superscript𝜆2𝜏superscript𝛼2superscript𝑒superscript𝑦22𝑥superscript𝑒superscript𝑥𝑦22𝑥erfc𝑦𝑥2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥erfc𝑦2𝑥2𝑥erfc𝑦2𝑥2\displaystyle\frac{1}{2}\lambda^{2}\tau\left(\alpha^{2}\left(-e^{-\frac{y^{2}}{2x}}\right)\left(e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{y+x}{\sqrt{2}\sqrt{x}}\right)-2e^{\frac{(2x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{y+2x}{\sqrt{2}\sqrt{x}}\right)\right)-\operatorname{erfc}\left(\frac{y}{\sqrt{2}\sqrt{x}}\right)+2\right) |  |

###### Lemma 17 (Main subfunction Below).

For 0.007⩽x⩽0.8750.007𝑥0.8750.007\leqslant x\leqslant 0.875 and −0.01⩽y⩽0.010.01𝑦0.01-0.01\leqslant y\leqslant 0.01,
the function

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(x+y)22​x​erfc⁡(x+y2​x)−2​e(2​x+y)22​x​erfc⁡(2​x+y2​x)superscript𝑒superscript𝑥𝑦22𝑥erfc𝑥𝑦2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥erfc2𝑥𝑦2𝑥\displaystyle e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)-2e^{\frac{(2x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right) |  | (125) |

smaller than zero, is strictly monotonically increasing in x𝑥x
and strictly monotonically increasing in y𝑦y for the minimal x=0.007=0.00875⋅0.8𝑥0.007⋅0.008750.8x=0.007=0.00875\cdot 0.8,
x=0.56=0.7⋅0.8𝑥0.56⋅0.70.8x=0.56=0.7\cdot 0.8, x=0.128=0.16⋅0.8𝑥0.128⋅0.160.8x=0.128=0.16\cdot 0.8, and x=0.216=0.24⋅0.9𝑥0.216⋅0.240.9x=0.216=0.24\cdot 0.9 (lower
bound of 0.90.90.9 on τ𝜏\tau).

###### Proof.

See proof [45](#Thmtheorem45 "Lemma 45 (Main subfunction below). ‣ Behavior of the main subfunction with respect to 𝑦 at minimal 𝑥. ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").
∎

###### Lemma 18 (Monotone Derivative).

For λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01}, α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01}
and the domain
−0.1⩽μ⩽0.10.1𝜇0.1-0.1\leqslant\mu\leqslant 0.1,
−0.1⩽ω⩽0.10.1𝜔0.1-0.1\leqslant\omega\leqslant 0.1,
0.00875⩽ν⩽0.70.00875𝜈0.70.00875\leqslant\nu\leqslant 0.7, and
0.8⩽τ⩽1.250.8𝜏1.250.8\leqslant\tau\leqslant 1.25.
We are interested of the derivative of

|  |  |  |  |
| --- | --- | --- | --- |
|  | τ​(e(μ​ω+ν​τ2​ν​τ)2​erfc⁡(μ​ω+ν​τ2​ν​τ)−2​e(μ​ω+2⋅ν​τ2​ν​τ)2​erfc⁡(μ​ω+2​ν​τ2​ν​τ)).𝜏superscript𝑒superscript𝜇𝜔𝜈𝜏2𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏2superscript𝑒superscript𝜇𝜔⋅2𝜈𝜏2𝜈𝜏2erfc𝜇𝜔2𝜈𝜏2𝜈𝜏\displaystyle\tau\left(e^{\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-2e^{\left(\frac{\mu\omega+2\cdot\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)\ . |  | (126) |

The derivative of the equation above with
respect to

* •

  ν𝜈\nu is larger than zero;
* •

  τ𝜏\tau is smaller than zero for maximal
  ν=0.7𝜈0.7\nu=0.7, ν=0.16𝜈0.16\nu=0.16, and ν=0.24𝜈0.24\nu=0.24 (with
  0.9⩽τ0.9𝜏0.9\leqslant\tau);
* •

  y=μ​ω𝑦𝜇𝜔y=\mu\omega is larger than zero for ν​τ=0.008750.8=0.007𝜈𝜏0.008750.80.007\nu\tau=0.008750.8=0.007, ν​τ=0.70.8=0.56𝜈𝜏0.70.80.56\nu\tau=0.70.8=0.56, ν​τ=0.160.8=0.128𝜈𝜏0.160.80.128\nu\tau=0.160.8=0.128, and ν​τ=0.24⋅0.9=0.216𝜈𝜏⋅0.240.90.216\nu\tau=0.24\cdot 0.9=0.216.

###### Proof.

See proof [46](#Thmtheorem46 "Lemma 46 (Monotone Derivative). ‣ Behavior of the main subfunction with respect to 𝑦 at minimal 𝑥. ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").
∎

#### A3.4.5 Computer-assisted proof details for main Lemma 12 in Section A3.4.1.

##### Error Analysis.

We investigate the error propagation for the
singular value (Eq. ([71](#S3.E71 "In Proof sketch: Bounding the largest singular value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"))) if the function arguments μ,ω,ν,τ

𝜇𝜔𝜈𝜏\mu,\omega,\nu,\tau
suffer from numerical imprecisions up to ϵitalic-ϵ\epsilon. To this end, we first
derive error propagation rules based on the mean value theorem and then
we apply these rules to the formula for the singular value.

###### Lemma 19 (Mean value theorem).

For a real-valued function f𝑓f which is differentiable in the closed interval [a,b]𝑎𝑏[a,b],
there exists t∈[0,1]𝑡01t\in[0,1] with

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | f​(𝒂)−f​(𝒃)𝑓𝒂𝑓𝒃\displaystyle f(\bm{a})\ -\ f(\bm{b})\ | =∇f​(𝒂+t​(𝒃−𝒂))⋅(𝒂−𝒃).absent⋅∇𝑓𝒂𝑡𝒃𝒂𝒂𝒃\displaystyle=\ \nabla f(\bm{a}+t(\bm{b}-\bm{a}))\ \cdot\ (\bm{a}\ -\ \bm{b})\ . |  | (127) |

It follows that
for computation with error Δ​xΔ𝑥\Delta x, there exists a t∈[0,1]𝑡01t\in[0,1] with

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | |f​(𝒙+Δ​𝒙)−f​(𝒙)|𝑓𝒙Δ𝒙𝑓𝒙\displaystyle\left|f(\bm{x}+\Delta\bm{x})\ -\ f(\bm{x})\right|\ | ⩽‖∇f​(𝒙+t​Δ​𝒙)‖​‖Δ​𝒙‖.absentnorm∇𝑓𝒙𝑡Δ𝒙normΔ𝒙\displaystyle\leqslant\ \left\|\nabla f(\bm{x}+t\Delta\bm{x})\right\|\ \left\|\Delta\bm{x}\right\|\ . |  | (128) |

Therefore the increase of the norm of the error after applying
function f𝑓f is bounded by the norm of the gradient
‖∇f​(𝒙+t​Δ​𝒙)‖norm∇𝑓𝒙𝑡Δ𝒙\left\|\nabla f(\bm{x}+t\Delta\bm{x})\right\|.

We now compute for the functions, that we consider their gradient and
its 2-norm:

* •

  addition:

  f​(𝒙)=x1+x2𝑓𝒙subscript𝑥1subscript𝑥2f(\bm{x})=x\_{1}+x\_{2} and ∇f​(𝒙)=(1,1)∇𝑓𝒙11\nabla f(\bm{x})=(1,1), which gives
  ‖∇f​(𝒙)‖=2norm∇𝑓𝒙2\left\|\nabla f(\bm{x})\right\|=\sqrt{2}.

  We further know that

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | |f​(𝒙+Δ​𝒙)−f​(𝒙)|=|x1+x2+Δ​x1+Δ​x2−x1−x2|⩽|Δ​x1|+|Δ​x2|.𝑓𝒙Δ𝒙𝑓𝒙subscript𝑥1subscript𝑥2Δsubscript𝑥1Δsubscript𝑥2subscript𝑥1subscript𝑥2Δsubscript𝑥1Δsubscript𝑥2\displaystyle\left|f(\bm{x}+\Delta\bm{x})-f(\bm{x})\right|\ =\ \left|x\_{1}+x\_{2}+\Delta x\_{1}+\Delta x\_{2}-x\_{1}-x\_{2}\right|\ \leqslant\ \left|\Delta x\_{1}\right|+\left|\Delta x\_{2}\right|\ . |  | (129) |

  Adding n𝑛n terms gives:

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | |∑i=1nxi+Δ​xi−∑i=1nxi|⩽∑i=1n|Δ​xi|⩽n​|Δ​xi|max.superscriptsubscript𝑖1𝑛subscript𝑥𝑖Δsubscript𝑥𝑖superscriptsubscript𝑖1𝑛subscript𝑥𝑖superscriptsubscript𝑖1𝑛Δsubscript𝑥𝑖𝑛subscriptΔsubscript𝑥𝑖max\displaystyle\left|\sum\_{i=1}^{n}x\_{i}+\Delta x\_{i}\ -\ \sum\_{i=1}^{n}x\_{i}\right|\ \leqslant\ \sum\_{i=1}^{n}\left|\Delta x\_{i}\right|\ \leqslant\ n\left|\Delta x\_{i}\right|\_{\mathrm{max}}\ . |  | (130) |
* •

  subtraction:

  f​(𝒙)=x1−x2𝑓𝒙subscript𝑥1subscript𝑥2f(\bm{x})=x\_{1}-x\_{2} and ∇f​(𝒙)=(1,−1)∇𝑓𝒙11\nabla f(\bm{x})=(1,-1), which gives
  ‖∇f​(𝒙)‖=2norm∇𝑓𝒙2\left\|\nabla f(\bm{x})\right\|=\sqrt{2}.

  We further know that

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | |f​(𝒙+Δ​𝒙)−f​(𝒙)|=|x1−x2+Δ​x1−Δ​x2−x1+x2|⩽|Δ​x1|+|Δ​x2|.𝑓𝒙Δ𝒙𝑓𝒙subscript𝑥1subscript𝑥2Δsubscript𝑥1Δsubscript𝑥2subscript𝑥1subscript𝑥2Δsubscript𝑥1Δsubscript𝑥2\displaystyle\left|f(\bm{x}+\Delta\bm{x})-f(\bm{x})\right|\ =\ \left|x\_{1}-x\_{2}+\Delta x\_{1}-\Delta x\_{2}-x\_{1}+x\_{2}\right|\ \leqslant\ \left|\Delta x\_{1}\right|+\left|\Delta x\_{2}\right|\ . |  | (131) |

  Subtracting n𝑛n terms gives:

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | |∑i=1n−(xi+Δ​xi)+∑i=1nxi|⩽∑i=1n|Δ​xi|⩽n​|Δ​xi|max.superscriptsubscript𝑖1𝑛subscript𝑥𝑖Δsubscript𝑥𝑖superscriptsubscript𝑖1𝑛subscript𝑥𝑖superscriptsubscript𝑖1𝑛Δsubscript𝑥𝑖𝑛subscriptΔsubscript𝑥𝑖max\displaystyle\left|\sum\_{i=1}^{n}-(x\_{i}+\Delta x\_{i})\ +\ \sum\_{i=1}^{n}x\_{i}\right|\ \leqslant\ \sum\_{i=1}^{n}\left|\Delta x\_{i}\right|\ \leqslant\ n\left|\Delta x\_{i}\right|\_{\mathrm{max}}\ . |  | (132) |
* •

  multiplication:

  f​(𝒙)=x1​x2𝑓𝒙subscript𝑥1subscript𝑥2f(\bm{x})=x\_{1}x\_{2} and ∇f​(𝒙)=(x2,x1)∇𝑓𝒙subscript𝑥2subscript𝑥1\nabla f(\bm{x})=(x\_{2},x\_{1}), which gives
  ‖∇f​(𝒙)‖=‖𝒙‖norm∇𝑓𝒙norm𝒙\left\|\nabla f(\bm{x})\right\|=\left\|\bm{x}\right\|.

  We further know that

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | |f​(𝒙+Δ​𝒙)−f​(𝒙)|=|x1⋅x2+Δ​x1⋅x2+Δ​x2⋅x1+Δ​x1⋅Δ​xs−x1⋅x2|⩽𝑓𝒙Δ𝒙𝑓𝒙⋅subscript𝑥1subscript𝑥2⋅Δsubscript𝑥1subscript𝑥2⋅Δsubscript𝑥2subscript𝑥1⋅Δsubscript𝑥1Δsubscript𝑥𝑠⋅subscript𝑥1subscript𝑥2absent\displaystyle\left|f(\bm{x}+\Delta\bm{x})-f(\bm{x})\right|\ =\ \left|x\_{1}\cdot x\_{2}+\Delta x\_{1}\cdot x\_{2}+\Delta x\_{2}\cdot x\_{1}+\Delta x\_{1}\cdot\Delta x\_{s}-x\_{1}\cdot x\_{2}\right|\ \leqslant |  | (133) |
  |  |  |  |
  | --- | --- | --- |
  |  | |Δ​x1|​|x2|+|Δ​x2|​|x1|+O​(Δ2).Δsubscript𝑥1subscript𝑥2Δsubscript𝑥2subscript𝑥1𝑂superscriptΔ2\displaystyle\left|\Delta x\_{1}\right|\left|x\_{2}\right|+\left|\Delta x\_{2}\right|\left|x\_{1}\right|+O(\Delta^{2})\ . |  |

  Multiplying n𝑛n terms gives:

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | |∏i=1n(xi+Δ​xi)−∏i=1nxi|=|∏i=1nxi​∑i=1nΔ​xixi+O​(Δ2)|⩽superscriptsubscriptproduct𝑖1𝑛subscript𝑥𝑖Δsubscript𝑥𝑖superscriptsubscriptproduct𝑖1𝑛subscript𝑥𝑖superscriptsubscriptproduct𝑖1𝑛subscript𝑥𝑖superscriptsubscript𝑖1𝑛Δsubscript𝑥𝑖subscript𝑥𝑖𝑂superscriptΔ2absent\displaystyle\left|\prod\_{i=1}^{n}(x\_{i}+\Delta x\_{i})\ -\ \prod\_{i=1}^{n}x\_{i}\right|\ =\ \left|\prod\_{i=1}^{n}x\_{i}\sum\_{i=1}^{n}\frac{\Delta x\_{i}}{x\_{i}}\ +\ O(\Delta^{2})\right|\ \leqslant |  | (134) |
  |  |  |  |
  | --- | --- | --- |
  |  | ∏i=1n|xi|​∑i=1n|Δ​xixi|+O​(Δ2)⩽n​∏i=1n|xi|​|Δ​xixi|max+O​(Δ2).superscriptsubscriptproduct𝑖1𝑛subscript𝑥𝑖superscriptsubscript𝑖1𝑛Δsubscript𝑥𝑖subscript𝑥𝑖𝑂superscriptΔ2𝑛superscriptsubscriptproduct𝑖1𝑛subscript𝑥𝑖subscriptΔsubscript𝑥𝑖subscript𝑥𝑖max𝑂superscriptΔ2\displaystyle\prod\_{i=1}^{n}\left|x\_{i}\right|\sum\_{i=1}^{n}\left|\frac{\Delta x\_{i}}{x\_{i}}\right|\ +\ O(\Delta^{2})\ \leqslant\ n\ \prod\_{i=1}^{n}\left|x\_{i}\right|\ \left|\frac{\Delta x\_{i}}{x\_{i}}\right|\_{\mathrm{max}}\ +\ O(\Delta^{2})\ . |  |
* •

  division:

  f​(𝒙)=x1x2𝑓𝒙subscript𝑥1subscript𝑥2f(\bm{x})=\frac{x\_{1}}{x\_{2}} and ∇f​(𝒙)=(1x2,−x1x22)∇𝑓𝒙1subscript𝑥2subscript𝑥1superscriptsubscript𝑥22\nabla f(\bm{x})=\left(\frac{1}{x\_{2}},-\frac{x\_{1}}{x\_{2}^{2}}\right), which gives
  ‖∇f​(𝒙)‖=‖𝒙‖x22norm∇𝑓𝒙norm𝒙superscriptsubscript𝑥22\left\|\nabla f(\bm{x})\right\|=\frac{\left\|\bm{x}\right\|}{x\_{2}^{2}}.

  We further know that

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | |f​(𝒙+Δ​𝒙)−f​(𝒙)|=|x1+Δ​x1x2+Δ​x2−x1x2|=|(x1+Δ​x1)​x2−x1​(x2+Δ​x2)(x2+Δ​x2)​x2|=𝑓𝒙Δ𝒙𝑓𝒙subscript𝑥1Δsubscript𝑥1subscript𝑥2Δsubscript𝑥2subscript𝑥1subscript𝑥2subscript𝑥1Δsubscript𝑥1subscript𝑥2subscript𝑥1subscript𝑥2Δsubscript𝑥2subscript𝑥2Δsubscript𝑥2subscript𝑥2absent\displaystyle\left|f(\bm{x}+\Delta\bm{x})-f(\bm{x})\right|\ =\ \left|\frac{x\_{1}+\Delta x\_{1}}{x\_{2}+\Delta x\_{2}}-\frac{x\_{1}}{x\_{2}}\right|\ =\ \left|\frac{(x\_{1}+\Delta x\_{1})x\_{2}-x\_{1}(x\_{2}+\Delta x\_{2})}{(x\_{2}+\Delta x\_{2})x\_{2}}\right|\ = |  | (135) |
  |  |  |  |
  | --- | --- | --- |
  |  | |Δ​x1⋅x2−Δ​x2⋅x1x22+Δ​x2⋅x2|=|Δ​x1x2−Δ​x2⋅x1x22|+O​(Δ2).⋅Δsubscript𝑥1subscript𝑥2⋅Δsubscript𝑥2subscript𝑥1superscriptsubscript𝑥22⋅Δsubscript𝑥2subscript𝑥2Δsubscript𝑥1subscript𝑥2⋅Δsubscript𝑥2subscript𝑥1superscriptsubscript𝑥22𝑂superscriptΔ2\displaystyle\left|\frac{\Delta x\_{1}\cdot x\_{2}-\Delta x\_{2}\cdot x\_{1}}{x\_{2}^{2}+\Delta x\_{2}\cdot x\_{2}}\right|\ =\ \left|\frac{\Delta x\_{1}}{x\_{2}}-\frac{\Delta x\_{2}\cdot x\_{1}}{x\_{2}^{2}}\right|+O(\Delta^{2})\ . |  |
* •

  square root:

  f​(x)=x𝑓𝑥𝑥f(x)=\sqrt{x} and f′​(x)=12​xsuperscript𝑓′𝑥12𝑥f^{\prime}(x)=\frac{1}{2\sqrt{x}}, which gives
  |f′​(x)|=12​xsuperscript𝑓′𝑥12𝑥\left|f^{\prime}(x)\right|=\frac{1}{2\sqrt{x}}.
* •

  exponential function:

  f​(x)=exp⁡(x)𝑓𝑥𝑥f(x)=\exp(x) and f′​(x)=exp⁡(x)superscript𝑓′𝑥𝑥f^{\prime}(x)=\exp(x), which gives
  |f′​(x)|=exp⁡(x)superscript𝑓′𝑥𝑥\left|f^{\prime}(x)\right|=\exp(x).
* •

  error function:

  f​(x)=erf​(x)𝑓𝑥erf𝑥f(x)=\mathrm{erf}(x) and f′​(x)=2π​exp⁡(−x2)superscript𝑓′𝑥2𝜋superscript𝑥2f^{\prime}(x)=\frac{2}{\sqrt{\pi}}\exp(-x^{2}), which gives
  |f′​(x)|=2π​exp⁡(−x2)superscript𝑓′𝑥2𝜋superscript𝑥2\left|f^{\prime}(x)\right|=\frac{2}{\sqrt{\pi}}\exp(-x^{2}).
* •

  complementary error function:

  f​(x)=erfc​(x)𝑓𝑥erfc𝑥f(x)=\mathrm{erfc}(x) and f′​(x)=−2π​exp⁡(−x2)superscript𝑓′𝑥2𝜋superscript𝑥2f^{\prime}(x)=-\frac{2}{\sqrt{\pi}}\exp(-x^{2}), which gives
  |f′​(x)|=2π​exp⁡(−x2)superscript𝑓′𝑥2𝜋superscript𝑥2\left|f^{\prime}(x)\right|=\frac{2}{\sqrt{\pi}}\exp(-x^{2}).

###### Lemma 20.

If the values μ,ω,ν,τ

𝜇𝜔𝜈𝜏\mu,\omega,\nu,\tau have a precision of ϵitalic-ϵ\epsilon,
the singular value (Eq. ([71](#S3.E71 "In Proof sketch: Bounding the largest singular value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"))) evaluated with the formulas
given in Eq. ([62](#S3.E62 "In Jacobian of the mapping. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) and Eq. ([71](#S3.E71 "In Proof sketch: Bounding the largest singular value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) has
a precision better than 292​ϵ292italic-ϵ292\epsilon.

This means for a machine with a typical precision of 2−52=2.220446⋅10−16superscript252⋅2.220446superscript10162^{-52}=2.220446\cdot 10^{-16}, we have the rounding error ϵ≈10−16italic-ϵsuperscript1016\epsilon\approx 10^{-16}, the evaluation
of the singular value (Eq. ([71](#S3.E71 "In Proof sketch: Bounding the largest singular value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"))) with the formulas given in Eq. ([62](#S3.E62 "In Jacobian of the mapping. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) and Eq. ([71](#S3.E71 "In Proof sketch: Bounding the largest singular value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) has
a precision better than 10−13>292​ϵsuperscript1013292italic-ϵ10^{-13}>292\epsilon.

###### Proof.

We have the numerical precision ϵitalic-ϵ\epsilon of the parameters μ,ω,ν,τ

𝜇𝜔𝜈𝜏\mu,\omega,\nu,\tau, that we denote by
Δ​μ,Δ​ω,Δ​ν,Δ​τ

Δ𝜇Δ𝜔Δ𝜈Δ𝜏\Delta\mu,\Delta\omega,\Delta\nu,\Delta\tau together with our domain ΩΩ\Omega.

With the error propagation rules that we derived in Subsection [A3.4.5](#S3.SS4.SSS5 "A3.4.5 Computer-assisted proof details for main Lemma 12 in Section A3.4.1. ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), we
can obtain bounds for the numerical errors on the following simple expressions:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δ​(μ​ω)Δ𝜇𝜔\displaystyle\Delta\left(\mu\omega\right) | ⩽Δ​μ​|ω|+Δ​ω​|μ|⩽0.2​ϵabsentΔ𝜇𝜔Δ𝜔𝜇0.2italic-ϵ\displaystyle\leqslant\Delta\mu\left|\omega\right|+\Delta\omega\left|\mu\right|\leqslant 0.2\epsilon |  | (136) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​(ν​τ)Δ𝜈𝜏\displaystyle\Delta\left(\nu\tau\right) | ⩽Δ​ν​|τ|+Δ​τ​|ν|⩽1.5​ϵ+1.5​ϵ=3​ϵabsentΔ𝜈𝜏Δ𝜏𝜈1.5italic-ϵ1.5italic-ϵ3italic-ϵ\displaystyle\leqslant\Delta\nu\left|\tau\right|+\Delta\tau\left|\nu\right|\leqslant 1.5\epsilon+1.5\epsilon=3\epsilon |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​(ν​τ2)Δ𝜈𝜏2\displaystyle\Delta\left(\frac{\nu\tau}{2}\right) | ⩽(Δ​(ν​τ)​2+Δ​2​|ν​τ|)​122⩽(6​ϵ+1.25⋅1.5​ϵ)/4<2​ϵabsentΔ𝜈𝜏2Δ2𝜈𝜏1superscript226italic-ϵ⋅1.251.5italic-ϵ42italic-ϵ\displaystyle\leqslant\left(\Delta(\nu\tau)2+\Delta 2\left|\nu\tau\right|\right)\frac{1}{2^{2}}\leqslant(6\epsilon+1.25\cdot 1.5\epsilon)/4<2\epsilon |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​(μ​ω+ν​τ)Δ𝜇𝜔𝜈𝜏\displaystyle\Delta\left(\mu\omega+\nu\tau\right) | ⩽Δ​(μ​ω)+Δ​(ν​τ)=3.2​ϵabsentΔ𝜇𝜔Δ𝜈𝜏3.2italic-ϵ\displaystyle\leqslant\Delta\left(\mu\omega\right)+\Delta\left(\nu\tau\right)=3.2\epsilon |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​(μ​ω+ν​τ2)Δ𝜇𝜔𝜈𝜏2\displaystyle\Delta\left(\mu\omega+\frac{\nu\tau}{2}\right) | ⩽Δ​(μ​ω)+Δ​(ν​τ2)<2.2​ϵabsentΔ𝜇𝜔Δ𝜈𝜏22.2italic-ϵ\displaystyle\leqslant\Delta\left(\mu\omega\right)+\Delta\left(\frac{\nu\tau}{2}\right)<2.2\epsilon |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​(ν​τ)Δ𝜈𝜏\displaystyle\Delta\left(\sqrt{\nu\tau}\right) | ⩽Δ​(ν​τ)2​ν​τ⩽3​ϵ2​0.64=1.875​ϵabsentΔ𝜈𝜏2𝜈𝜏3italic-ϵ20.641.875italic-ϵ\displaystyle\leqslant\frac{\Delta\left(\nu\tau\right)}{2\sqrt{\nu\tau}}\leqslant\frac{3\epsilon}{2\sqrt{0.64}}=1.875\epsilon |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​(2)Δ2\displaystyle\Delta\left(\sqrt{2}\right) | ⩽Δ​22​2⩽12​2​ϵabsentΔ222122italic-ϵ\displaystyle\leqslant\frac{\Delta 2}{2\sqrt{2}}\leqslant\frac{1}{2\sqrt{2}}\epsilon |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​(2​ν​τ)Δ2𝜈𝜏\displaystyle\Delta\left(\sqrt{2}\sqrt{\nu\tau}\right) | ⩽2​Δ​(ν​τ)+ν​τ​Δ​(2)⩽2⋅1.875​ϵ+1.5⋅1.25⋅12​2​ϵ<3.5​ϵabsent2Δ𝜈𝜏𝜈𝜏Δ2⋅21.875italic-ϵ⋅1.51.25122italic-ϵ3.5italic-ϵ\displaystyle\leqslant\sqrt{2}\Delta\left(\sqrt{\nu\tau}\right)+\nu\tau\Delta\left(\sqrt{2}\right)\leqslant\sqrt{2}\cdot 1.875\epsilon+1.5\cdot 1.25\cdot\frac{1}{2\sqrt{2}}\epsilon<3.5\epsilon |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​(μ​ω2​ν​τ)Δ𝜇𝜔2𝜈𝜏\displaystyle\Delta\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right) | ⩽(Δ​(μ​ω)​2​ν​τ+|μ​ω|​Δ​(2​ν​τ))​1(2​ν​τ)2⩽absentΔ𝜇𝜔2𝜈𝜏𝜇𝜔Δ2𝜈𝜏1superscript2𝜈𝜏2absent\displaystyle\leqslant\left(\Delta\left(\mu\omega\right)\sqrt{2}\sqrt{\nu\tau}+\left|\mu\omega\right|\Delta\left(\sqrt{2}\sqrt{\nu\tau}\right)\right)\frac{1}{\left(\sqrt{2}\sqrt{\nu\tau}\right)^{2}}\leqslant |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (0.2​ϵ​2​0.64+0.01⋅3.5​ϵ)​12⋅0.64<0.25​ϵ0.2italic-ϵ20.64⋅0.013.5italic-ϵ1⋅20.640.25italic-ϵ\displaystyle\left(0.2\epsilon\sqrt{2}\sqrt{0.64}+0.01\cdot 3.5\epsilon\right)\frac{1}{2\cdot 0.64}<0.25\epsilon |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​(μ​ω+ν​τ2​ν​τ)Δ𝜇𝜔𝜈𝜏2𝜈𝜏\displaystyle\Delta\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right) | ⩽(Δ​(μ​ω+ν​τ)​2​ν​τ+|μ​ω+ν​τ|​Δ​(2​ν​τ))​1(2​ν​τ)2⩽absentΔ𝜇𝜔𝜈𝜏2𝜈𝜏𝜇𝜔𝜈𝜏Δ2𝜈𝜏1superscript2𝜈𝜏2absent\displaystyle\leqslant\left(\Delta\left(\mu\omega+\nu\tau\right)\sqrt{2}\sqrt{\nu\tau}+\left|\mu\omega+\nu\tau\right|\Delta\left(\sqrt{2}\sqrt{\nu\tau}\right)\right)\frac{1}{\left(\sqrt{2}\sqrt{\nu\tau}\right)^{2}}\leqslant |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (3.2​ϵ​2​0.64+1.885⋅3.5​ϵ)​12⋅0.64<8​ϵ.3.2italic-ϵ20.64⋅1.8853.5italic-ϵ1⋅20.648italic-ϵ\displaystyle\left(3.2\epsilon\sqrt{2}\sqrt{0.64}+1.885\cdot 3.5\epsilon\right)\frac{1}{2\cdot 0.64}<8\epsilon. |  |

Using these bounds on the simple expressions, we can now calculate bounds on the numerical errors of compound expressions:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δ​(erfc⁡(μ​ω2​ν​τ))Δerfc𝜇𝜔2𝜈𝜏\displaystyle\Delta\left(\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)\right) | ⩽2π​e−(μ​ω2​ν​τ)2​Δ​(μ​ω2​ν​τ)<absent2𝜋superscript𝑒superscript𝜇𝜔2𝜈𝜏2Δ𝜇𝜔2𝜈𝜏absent\displaystyle\leqslant\frac{2}{\sqrt{\pi}}e^{-\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\Delta\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)< |  | (137) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2π​0.25​ϵ<0.3​ϵ2𝜋0.25italic-ϵ0.3italic-ϵ\displaystyle\frac{2}{\sqrt{\pi}}0.25\epsilon<0.3\epsilon |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δ​(erfc⁡(μ​ω+ν​τ2​ν​τ))Δerfc𝜇𝜔𝜈𝜏2𝜈𝜏\displaystyle\Delta\left(\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\right) | ⩽2π​e−(μ​ω+ν​τ2​ν​τ)2​Δ​(μ​ω+ν​τ2​ν​τ)<absent2𝜋superscript𝑒superscript𝜇𝜔𝜈𝜏2𝜈𝜏2Δ𝜇𝜔𝜈𝜏2𝜈𝜏absent\displaystyle\leqslant\frac{2}{\sqrt{\pi}}e^{-\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\Delta\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)< |  | (138) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2π​8​ϵ<10​ϵ2𝜋8italic-ϵ10italic-ϵ\displaystyle\frac{2}{\sqrt{\pi}}8\epsilon<10\epsilon |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δ​(eμ​ω+ν​τ2)Δsuperscript𝑒𝜇𝜔𝜈𝜏2\displaystyle\Delta\left(e^{\mu\omega+\frac{\nu\tau}{2}}\right) | ⩽(eμ​ω+ν​τ2)​Δ​(eμ​ω+ν​τ2)<absentsuperscript𝑒𝜇𝜔𝜈𝜏2Δsuperscript𝑒𝜇𝜔𝜈𝜏2absent\displaystyle\leqslant\left(e^{\mu\omega+\frac{\nu\tau}{2}}\right)\Delta\left(e^{\mu\omega+\frac{\nu\tau}{2}}\right)< |  | (139) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | e0.9475​2.2​ϵ<5.7​ϵsuperscript𝑒0.94752.2italic-ϵ5.7italic-ϵ\displaystyle e^{0.9475}2.2\epsilon<5.7\epsilon |  | (140) |

Subsequently, we can use the above results to get bounds for the numerical errors on the Jacobian entries (Eq. ([62](#S3.E62 "In Jacobian of the mapping. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"))),
applying the rules from Subsection [A3.4.5](#S3.SS4.SSS5 "A3.4.5 Computer-assisted proof details for main Lemma 12 in Section A3.4.1. ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") again:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​(𝒥11)=Δ​(12​λ​ω​(α​eμ​ω+ν​τ2​erfc⁡(μ​ω+ν​τ2​ν​τ)−erfc⁡(μ​ω2​ν​τ)+2))<6​ϵ,Δsubscript𝒥11Δ12𝜆𝜔𝛼superscript𝑒𝜇𝜔𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏erfc𝜇𝜔2𝜈𝜏26italic-ϵ\displaystyle\Delta\left({\mathcal{J}}\_{11}\right)\ =\ \Delta\left(\frac{1}{2}\lambda\omega\left(\alpha e^{\mu\omega+\frac{\nu\tau}{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+2\right)\right)<6\epsilon, |  | (141) |

and we obtain Δ​(𝒥12)<78​ϵΔsubscript𝒥1278italic-ϵ\Delta\left({\mathcal{J}}\_{12}\right)<78\epsilon, Δ​(𝒥21)<189​ϵΔsubscript𝒥21189italic-ϵ\Delta\left({\mathcal{J}}\_{21}\right)<189\epsilon, Δ​(𝒥22)<405​ϵΔsubscript𝒥22405italic-ϵ\Delta\left({\mathcal{J}}\_{22}\right)<405\epsilon
and Δ​(μ~)<52​ϵΔ~𝜇52italic-ϵ\Delta\left({\tilde{\mu}}\right)<52\epsilon.
We also have bounds on the absolute values on 𝒥i​jsubscript𝒥𝑖𝑗\mathcal{J}\_{ij} and μ~~𝜇{\tilde{\mu}} (see Lemma [6](#Thmtheorem6 "Lemma 6 (Bound on J11). ‣ Bounds on the entries of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"),
Lemma [7](#Thmtheorem7 "Lemma 7 (Bound on J12). ‣ Bounds on the entries of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), and Lemma [9](#Thmtheorem9 "Lemma 9 (Bounds on mean, variance and second moment). ‣ Bounds on mean, variance and second moment. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")), therefore we can
propagate the error also through the function that calculates the singular value (Eq. ([71](#S3.E71 "In Proof sketch: Bounding the largest singular value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"))).

|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​(S​(μ,ω,ν,τ,λ,α))=Δ𝑆𝜇𝜔𝜈𝜏𝜆𝛼absent\displaystyle\Delta\left(S(\mu,\omega,\nu,\tau,\lambda,\alpha)\right)\ = |  | (142) |
|  |  |  |
| --- | --- | --- |
|  | Δ(12((𝒥11+𝒥22−2​μ~​𝒥12)2+(𝒥21−2​μ~​𝒥11−𝒥12)2+\displaystyle\Delta\left(\frac{1}{2}\ \left(\sqrt{({\mathcal{J}}\_{11}+{\mathcal{J}}\_{22}-2{\tilde{\mu}}{\mathcal{J}}\_{12})^{2}+({\mathcal{J}}\_{21}-2{\tilde{\mu}}{\mathcal{J}}\_{11}-{\mathcal{J}}\_{12})^{2}}\ +\right.\right. |  |
|  |  |  |
| --- | --- | --- |
|  | (𝒥11−𝒥22+2​μ~​𝒥12)2+(𝒥12+𝒥21−2​μ~​𝒥11)2))<292ϵ.\displaystyle\left.\left.\sqrt{({\mathcal{J}}\_{11}-{\mathcal{J}}\_{22}+2{\tilde{\mu}}{\mathcal{J}}\_{12})^{2}+({\mathcal{J}}\_{12}+{\mathcal{J}}\_{21}-2{\tilde{\mu}}{\mathcal{J}}\_{11})^{2}}\right)\right)<292\epsilon. |  |

∎

##### Precision of Implementations.

We will show that our computations are correct up to 3 ulps. For
our implementation in GNU C library and the hardware architectures
that we used, the precision of all mathematical functions that we used
is at least one ulp.
The term “ulp” (acronym for “unit in the last place”) was coined
by W. Kahan in 1960. It is the highest precision (up to some factor
smaller 1), which can be
achieved for the given hardware and floating point representation.

Kahan defined ulp as [[21](#bib.bib21)]:

> “Ulp(x)𝑥(x) is the gap between the two finite floating-point numbers
> nearest x𝑥x, even if x𝑥x is one of them. (But ulp(NaN) is NaN.)”

Harrison defined ulp as [[15](#bib.bib15)]:

> “an ulp in x𝑥x is the distance
> between the two closest straddling floating point numbers a𝑎a and b𝑏b, i.e. those with
> a⩽x⩽b𝑎𝑥𝑏a\leqslant x\leqslant b and a≠b𝑎𝑏a\not=b assuming an unbounded exponent range.”

In the literature we find also slightly different definitions
[[29](#bib.bib29)].

According to [[29](#bib.bib29)] who refers to [[11](#bib.bib11)]:

> “IEEE-754 mandates four standard rounding modes:”
>
> “Round-to-nearest: r​(x)𝑟𝑥r(x) is the floating-point value closest to x𝑥x with the
> usual distance; if two floating-point value are equally close to x𝑥x, then r​(x)𝑟𝑥r(x)
> is the one whose least significant bit is equal to zero.”
>
> “IEEE-754 standardises 5 operations: addition (which we shall note ⊕direct-sum\oplus in order to
> distinguish it from the operation over the reals), subtraction (⊖symmetric-difference\ominus), multiplication
> (⊗tensor-product\otimes), division (⊘⊘\oslash), and also square root.”
>
> “IEEE-754 specifies em exact rounding [Goldberg, 1991, §1.5]: the result of a
> floating-point operation is the same as if the operation were performed on the
> real numbers with the given inputs, then rounded according to the rules in the
> preceding section. Thus, x⊕ydirect-sum𝑥𝑦x\oplus y is defined as r​(x+y)𝑟𝑥𝑦r(x+y), with x𝑥x and y𝑦y taken as
> elements of ℝ∪{−∞,+∞}ℝ\mathbb{R}\cup\{-\infty,+\infty\}; the same applies for the other operators.”

Consequently, the IEEE-754 standard guarantees that addition,
subtraction, multiplication, division, and squared root is precise up
to one ulp.

We have to consider transcendental functions. First the is the
exponential function, and then the complementary error
function erfc​(x)erfc𝑥\mathrm{erfc}(x),
which can be computed via the error function erf​(x)erf𝑥\mathrm{erf}(x).

Intel states [[29](#bib.bib29)]:

> “With the Intel486 processor and Intel 387 math coprocessor, the worst-
> case, transcendental function error is typically 333 or 3.53.53.5 ulps, but is some-
> times as large as 4.54.54.5 ulps.”

According to <https://www.mirbsd.org/htman/i386/man3/exp.htm> and
<http://man.openbsd.org/OpenBSD-current/man3/exp.3>:

> “exp(x)𝑥(x), log(x)𝑥(x), expm1(x)𝑥(x) and log1p(x)𝑥(x) are accurate to within an ulp”

which is the same for freebsd <https://www.freebsd.org/cgi/man.cgi?query=exp&sektion=3&apropos=0&manpath=freebsd>:

> “The values of exp(0), expm1(0), exp2(integer), and pow(integer, integer)
> are exact provided that they are representable. Otherwise the error in
> these functions is generally below one ulp.”

The same holds for “FDLIBM” <http://www.netlib.org/fdlibm/readme>:

> “FDLIBM is intended to provide a reasonably portable (see
> assumptions below), reference quality (below one ulp for
> major functions like sin,cos,exp,log) math library
> (libm.a).”

In
<http://www.gnu.org/software/libc/manual/html_node/Errors-in-Math-Functions.html>
we find that both expexp\mathrm{exp} and
erferf\mathrm{erf} have an error of 1 ulp while erfcerfc\mathrm{erfc} has an
error up to 3 ulps depending on the architecture.
For the most common architectures as used by us, however, the error of
erfcerfc\mathrm{erfc} is 1 ulp.

We implemented the function in the programming language C.
We rely on the GNU C Library [[26](#bib.bib26)].
According to the GNU C Library manual which can be obtained from
<http://www.gnu.org/software/libc/manual/pdf/libc.pdf>,
the errors of the math functions exp\exp, erferf\mathrm{erf}, and
erfcerfc\mathrm{erfc}
are not larger than 3 ulps for all architectures
[[26](#bib.bib26), pp. 528].
For the architectures ix86, i386/i686/fpu, and m68k/fpmu68k/m680x0/fpu
that we used the error are at least one ulp
[[26](#bib.bib26), pp. 528].

#### A3.4.6 Intermediate Lemmata and Proofs

Since we focus on the fixed point
(μ,ν)=(0,1)𝜇𝜈01(\mu,\nu)=(0,1),
we assume for our whole analysis
that α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01} and λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01}.
Furthermore, we restrict the range of the variables
μ∈[μmin,μmax]=[−0.1,0.1]𝜇subscript𝜇minsubscript𝜇max0.10.1\mu\in[\mu\_{\rm min},\mu\_{\rm max}]=[-0.1,0.1],
ω∈[ωmin,ωmax]=[−0.1,0.1]𝜔subscript𝜔minsubscript𝜔max0.10.1\omega\in[\omega\_{\rm min},\omega\_{\rm max}]=[-0.1,0.1],
ν∈[νmin,νmax]=[0.8,1.5]𝜈subscript𝜈minsubscript𝜈max0.81.5\nu\in[\nu\_{\rm min},\nu\_{\rm max}]=[0.8,1.5], and
τ∈[τmin,τmax]=[0.8,1.25]𝜏subscript𝜏minsubscript𝜏max0.81.25\tau\in[\tau\_{\rm min},\tau\_{\rm max}]=[0.8,1.25].

For bounding different partial derivatives we need properties of
different functions.
We will bound a the absolute value of a function by computing an upper
bound on its maximum and a lower bound on its minimum. These bounds
are computed by upper or lower bounding terms. The bounds get tighter
if we can combine terms to a more complex function and bound this
function. The following lemmata give some properties of functions that
we will use in bounding complex functions.

Throughout this work, we use the error function erf⁡(x):=1π​∫−xxe−t2assignerf𝑥1𝜋superscriptsubscript𝑥𝑥superscript𝑒superscript𝑡2\operatorname{erf}(x):=\frac{1}{\sqrt{\pi}}\int\_{-x}^{x}e^{-t^{2}} and the complementary
error function erfc⁡(x)=1−erf⁡(x)erfc𝑥1erf𝑥\operatorname{erfc}(x)=1-\operatorname{erf}(x).

###### Lemma 21 (Basic functions).

exp⁡(x)𝑥\exp(x) is strictly monotonically increasing from 00 at −∞-\infty to
∞\infty at ∞\infty and has positive curvature.

According to its definition erfc⁡(x)erfc𝑥\operatorname{erfc}(x) is strictly monotonically decreasing from 2 at −∞-\infty to 0 at ∞\infty.

Next we introduce a bound on erfcerfc\operatorname{erfc}:

###### Lemma 22 (Erfc bound from Abramowitz).

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 2​e−x2π​(x2+2+x)2superscript𝑒superscript𝑥2𝜋superscript𝑥22𝑥\displaystyle\frac{2e^{-x^{2}}}{\sqrt{\pi}\left(\sqrt{x^{2}+2}+x\right)}\ | <erfc⁡(x)⩽2​e−x2π​(x2+4π+x),absenterfc𝑥2superscript𝑒superscript𝑥2𝜋superscript𝑥24𝜋𝑥\displaystyle<\ \operatorname{erfc}(x)\ \leqslant\ \frac{2e^{-x^{2}}}{\sqrt{\pi}\left(\sqrt{x^{2}+\frac{4}{\pi}}+x\right)}, |  | (143) |

for x>0𝑥0x>0.

###### Proof.

The statement follows immediately from
[[1](#bib.bib1)] (page 298, formula 7.1.13).
∎

These bounds are displayed in figure [A4](#S3.F4 "Figure A4 ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").

![Refer to caption](/html/1706.02515/assets/x6.png)


Figure A4: Graphs of the upper and lower bounds on erfcerfc\operatorname{erfc}. The lower bound 2​e−x2π​(x2+2+x)2superscript𝑒superscript𝑥2𝜋superscript𝑥22𝑥\frac{2e^{-x^{2}}}{\sqrt{\pi}\left(\sqrt{x^{2}+2}+x\right)} (red),
the upper bound 2​e−x2π​(x2+4π+x)2superscript𝑒superscript𝑥2𝜋superscript𝑥24𝜋𝑥\frac{2e^{-x^{2}}}{\sqrt{\pi}\left(\sqrt{x^{2}+\frac{4}{\pi}}+x\right)} (green) and the function erfc⁡(x)erfc𝑥\operatorname{erfc}(x) (blue) as
treated in Lemma [22](#Thmtheorem22 "Lemma 22 (Erfc bound from Abramowitz). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").

###### Lemma 23 (Function ex2​erfc⁡(x)superscript𝑒superscript𝑥2erfc𝑥e^{x^{2}}\operatorname{erfc}(x)).

ex2​erfc⁡(x)superscript𝑒superscript𝑥2erfc𝑥e^{x^{2}}\operatorname{erfc}(x) is strictly monotonically decreasing for x>0𝑥0x>0
and has positive curvature
(positive 2nd order derivative), that is, the decreasing slowes down.

![Refer to caption](/html/1706.02515/assets/x7.png)

![Refer to caption](/html/1706.02515/assets/x8.png)

Figure A5: Graphs of the functions ex2​erfc⁡(x)superscript𝑒superscript𝑥2erfc𝑥e^{x^{2}}\operatorname{erfc}(x) (left) and x​ex2​erfc⁡(x)𝑥superscript𝑒superscript𝑥2erfc𝑥xe^{x^{2}}\operatorname{erfc}(x) (right) treated in Lemma [23](#Thmtheorem23 "Lemma 23 (Function 𝑒^𝑥²⁢erfc(𝑥)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") and Lemma [24](#Thmtheorem24 "Lemma 24 (Properties of 𝑥⁢𝑒^𝑥²⁢erfc(𝑥)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"),
respectively.

A graph of the function is displayed in Figure [A5](#S3.F5 "Figure A5 ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").

###### Proof.

The derivative of ex2​erfc⁡(x)superscript𝑒superscript𝑥2erfc𝑥e^{x^{2}}\operatorname{erfc}(x) is

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂ex2​erfc⁡(x)∂x= 2​ex2​x​erfc⁡(x)−2π.superscript𝑒superscript𝑥2erfc𝑥𝑥2superscript𝑒superscript𝑥2𝑥erfc𝑥2𝜋\displaystyle\frac{\partial e^{x^{2}}\operatorname{erfc}(x)}{\partial x}\ =\ 2e^{x^{2}}x\operatorname{erfc}(x)-\frac{2}{\sqrt{\pi}}\ . |  | (144) |

Using Lemma [22](#Thmtheorem22 "Lemma 22 (Erfc bound from Abramowitz). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), we get

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂ex2​erfc⁡(x)∂x= 2​ex2​x​erfc⁡(x)−2π<superscript𝑒superscript𝑥2erfc𝑥𝑥2superscript𝑒superscript𝑥2𝑥erfc𝑥2𝜋absent\displaystyle\frac{\partial e^{x^{2}}\operatorname{erfc}(x)}{\partial x}\ =\ 2e^{x^{2}}x\operatorname{erfc}(x)-\frac{2}{\sqrt{\pi}}\ < | 4​xπ​(x2+4π+x)−2π=2​(24π​x2+1+1−1)π< 04𝑥𝜋superscript𝑥24𝜋𝑥2𝜋224𝜋superscript𝑥2111𝜋 0\displaystyle\frac{4x}{\sqrt{\pi}\left(\sqrt{x^{2}+\frac{4}{\pi}}+x\right)}-\frac{2}{\sqrt{\pi}}=\frac{2\left(\frac{2}{\sqrt{\frac{4}{\pi x^{2}}+1}+1}-1\right)}{\sqrt{\pi}}\ <\ 0 |  | (145) |

Thus ex2​erfc⁡(x)superscript𝑒superscript𝑥2erfc𝑥e^{x^{2}}\operatorname{erfc}(x)
is strictly monotonically decreasing for x>0𝑥0x>0.

The second order derivative of ex2​erfc⁡(x)superscript𝑒superscript𝑥2erfc𝑥e^{x^{2}}\operatorname{erfc}(x) is

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂2ex2​erfc⁡(x)∂x2= 4​ex2​x2​erfc⁡(x)+2​ex2​erfc⁡(x)−4​xπ.superscript2superscript𝑒superscript𝑥2erfc𝑥superscript𝑥24superscript𝑒superscript𝑥2superscript𝑥2erfc𝑥2superscript𝑒superscript𝑥2erfc𝑥4𝑥𝜋\displaystyle\frac{\partial^{2}e^{x^{2}}\operatorname{erfc}(x)}{\partial x^{2}}\ =\ 4e^{x^{2}}x^{2}\operatorname{erfc}(x)+2e^{x^{2}}\operatorname{erfc}(x)-\frac{4x}{\sqrt{\pi}}\ . |  | (146) |

Again using Lemma [22](#Thmtheorem22 "Lemma 22 (Erfc bound from Abramowitz). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") (first inequality), we get

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2​((2​x2+1)​ex2​erfc⁡(x)−2​xπ)>22superscript𝑥21superscript𝑒superscript𝑥2erfc𝑥2𝑥𝜋absent\displaystyle 2\left(\left(2x^{2}+1\right)e^{x^{2}}\operatorname{erfc}(x)-\frac{2x}{\sqrt{\pi}}\right)\ > |  | (147) |
|  |  |  |
| --- | --- | --- |
|  | 4​(2​x2+1)π​(x2+2+x)−4​xπ=42superscript𝑥21𝜋superscript𝑥22𝑥4𝑥𝜋absent\displaystyle\frac{4\left(2x^{2}+1\right)}{\sqrt{\pi}\left(\sqrt{x^{2}+2}+x\right)}-\frac{4x}{\sqrt{\pi}}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | 4​(x2−x2+2​x+1)π​(x2+2+x)=4superscript𝑥2superscript𝑥22𝑥1𝜋superscript𝑥22𝑥absent\displaystyle\frac{4\left(x^{2}-\sqrt{x^{2}+2}x+1\right)}{\sqrt{\pi}\left(\sqrt{x^{2}+2}+x\right)}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | 4​(x2−x4+2​x2+1)π​(x2+2+x)>4superscript𝑥2superscript𝑥42superscript𝑥21𝜋superscript𝑥22𝑥absent\displaystyle\frac{4\left(x^{2}-\sqrt{x^{4}+2x^{2}}+1\right)}{\sqrt{\pi}\left(\sqrt{x^{2}+2}+x\right)}\ > |  |
|  |  |  |
| --- | --- | --- |
|  | 4​(x2−x4+2​x2+1+1)π​(x2+2+x)= 04superscript𝑥2superscript𝑥42superscript𝑥211𝜋superscript𝑥22𝑥 0\displaystyle\frac{4\left(x^{2}-\sqrt{x^{4}+2x^{2}+1}+1\right)}{\sqrt{\pi}\left(\sqrt{x^{2}+2}+x\right)}\ =\ 0 |  |

For the last inequality we added 1 in the numerator in the square root
which is subtracted, that is, making a larger negative term in the
numerator.
∎

###### Lemma 24 (Properties of x​ex2​erfc⁡(x)𝑥superscript𝑒superscript𝑥2erfc𝑥xe^{x^{2}}\operatorname{erfc}(x)).

The function x​ex2​erfc⁡(x)𝑥superscript𝑒superscript𝑥2erfc𝑥xe^{x^{2}}\operatorname{erfc}(x) has the sign of x𝑥x and is
monotonically increasing to 1π1𝜋\frac{1}{\sqrt{\pi}}.

###### Proof.

The derivative of x​ex2​erfc⁡(x)𝑥superscript𝑒superscript𝑥2erfc𝑥xe^{x^{2}}\operatorname{erfc}(x) is

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2​ex2​x2​erfc⁡(x)+ex2​erfc⁡(x)−2​xπ.2superscript𝑒superscript𝑥2superscript𝑥2erfc𝑥superscript𝑒superscript𝑥2erfc𝑥2𝑥𝜋\displaystyle 2e^{x^{2}}x^{2}\operatorname{erfc}(x)+e^{x^{2}}\operatorname{erfc}(x)-\frac{2x}{\sqrt{\pi}}\ . |  | (148) |

This derivative is positive since

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2​ex2​x2​erfc⁡(x)+ex2​erfc⁡(x)−2​xπ=2superscript𝑒superscript𝑥2superscript𝑥2erfc𝑥superscript𝑒superscript𝑥2erfc𝑥2𝑥𝜋absent\displaystyle 2e^{x^{2}}x^{2}\operatorname{erfc}(x)+e^{x^{2}}\operatorname{erfc}(x)-\frac{2x}{\sqrt{\pi}}\ = |  | (149) |
|  |  |  |
| --- | --- | --- |
|  | ex2​(2​x2+1)​erfc⁡(x)−2​xπ>2​(2​x2+1)π​(x2+2+x)−2​xπ=2​((2​x2+1)−x​(x2+2+x))π​(x2+2+x)=superscript𝑒superscript𝑥22superscript𝑥21erfc𝑥2𝑥𝜋22superscript𝑥21𝜋superscript𝑥22𝑥2𝑥𝜋22superscript𝑥21𝑥superscript𝑥22𝑥𝜋superscript𝑥22𝑥absent\displaystyle e^{x^{2}}\left(2x^{2}+1\right)\operatorname{erfc}(x)-\frac{2x}{\sqrt{\pi}}>\frac{2\left(2x^{2}+1\right)}{\sqrt{\pi}\left(\sqrt{x^{2}+2}+x\right)}-\frac{2x}{\sqrt{\pi}}=\frac{2\left(\left(2x^{2}+1\right)-x\left(\sqrt{x^{2}+2}+x\right)\right)}{\sqrt{\pi}\left(\sqrt{x^{2}+2}+x\right)}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | 2​(x2−x​x2+2+1)π​(x2+2+x)=2​(x2−x​x2+2+1)π​(x2+2+x)>2​(x2−x​x2+1x2+2+1)π​(x2+2+x)=2superscript𝑥2𝑥superscript𝑥221𝜋superscript𝑥22𝑥2superscript𝑥2𝑥superscript𝑥221𝜋superscript𝑥22𝑥2superscript𝑥2𝑥superscript𝑥21superscript𝑥221𝜋superscript𝑥22𝑥absent\displaystyle\frac{2\left(x^{2}-x\sqrt{x^{2}+2}+1\right)}{\sqrt{\pi}\left(\sqrt{x^{2}+2}+x\right)}=\frac{2\left(x^{2}-x\sqrt{x^{2}+2}+1\right)}{\sqrt{\pi}\left(\sqrt{x^{2}+2}+x\right)}>\frac{2\left(x^{2}-x\sqrt{x^{2}+\frac{1}{x^{2}}+2}+1\right)}{\sqrt{\pi}\left(\sqrt{x^{2}+2}+x\right)}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | 2​(x2−x4+2​x2+1+1)π​(x2+2+x)=2​(x2−(x2+1)2+1)π​(x2+2+x)=0.2superscript𝑥2superscript𝑥42superscript𝑥211𝜋superscript𝑥22𝑥2superscript𝑥2superscriptsuperscript𝑥2121𝜋superscript𝑥22𝑥0\displaystyle\frac{2\left(x^{2}-\sqrt{x^{4}+2x^{2}+1}+1\right)}{\sqrt{\pi}\left(\sqrt{x^{2}+2}+x\right)}=\frac{2\left(x^{2}-\sqrt{\left(x^{2}+1\right)^{2}}+1\right)}{\sqrt{\pi}\left(\sqrt{x^{2}+2}+x\right)}=0\ . |  |

We apply Lemma [22](#Thmtheorem22 "Lemma 22 (Erfc bound from Abramowitz). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
to x​erfc⁡(x)​ex2𝑥erfc𝑥superscript𝑒superscript𝑥2x\operatorname{erfc}(x)e^{x^{2}} and divide the terms of the lemma by x𝑥x,
which gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2π​(2x2+1+1)<x​erfc⁡(x)​ex2⩽2π​(4π​x2+1+1).2𝜋2superscript𝑥211𝑥erfc𝑥superscript𝑒superscript𝑥22𝜋4𝜋superscript𝑥211\displaystyle\frac{2}{\sqrt{\pi}\left(\sqrt{\frac{2}{x^{2}}+1}+1\right)}<x\operatorname{erfc}(x)e^{x^{2}}\leqslant\frac{2}{\sqrt{\pi}\left(\sqrt{\frac{4}{\pi x^{2}}+1}+1\right)}\ . |  | (150) |

For limx→∞subscript→𝑥\lim\_{x\to\infty} both the upper and the lower bound go to
1π1𝜋\frac{1}{\sqrt{\pi}}.
∎

###### Lemma 25 (Function μ​ω𝜇𝜔\mu\omega).

h11​(μ,ω)=μ​ωsubscriptℎ11𝜇𝜔𝜇𝜔h\_{11}(\mu,\omega)=\mu\omega is monotonically increasing in μ​ω𝜇𝜔\mu\omega.
It has minimal value t11=−0.01subscript𝑡110.01t\_{11}=-0.01 and maximal value
T11=0.01subscript𝑇110.01T\_{11}=0.01.

###### Proof.

Obvious.
∎

###### Lemma 26 (Function ν​τ𝜈𝜏\nu\tau).

h22​(ν,τ)=ν​τsubscriptℎ22𝜈𝜏𝜈𝜏h\_{22}(\nu,\tau)=\nu\tau is
monotonically increasing in ν​τ𝜈𝜏\nu\tau and is positive.
It has minimal value t22=0.64subscript𝑡220.64t\_{22}=0.64 and maximal value
T22=1.875subscript𝑇221.875T\_{22}=1.875.

###### Proof.

Obvious.
∎

###### Lemma 27 (Function μ​ω+ν​τ2​ν​τ𝜇𝜔𝜈𝜏2𝜈𝜏\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}).

h1​(μ,ω,ν,τ)=μ​ω+ν​τ2​ν​τsubscriptℎ1𝜇𝜔𝜈𝜏𝜇𝜔𝜈𝜏2𝜈𝜏h\_{1}(\mu,\omega,\nu,\tau)=\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}
is larger than zero and increasing in both ν​τ𝜈𝜏\nu\tau and μ​ω𝜇𝜔\mu\omega.
It has minimal value t1=0.5568subscript𝑡10.5568t\_{1}=0.5568 and maximal value
T1=0.9734subscript𝑇10.9734T\_{1}=0.9734.

###### Proof.

The derivative of the function
μ​ω+x2​x𝜇𝜔𝑥2𝑥\frac{\mu\omega+x}{\sqrt{2}\sqrt{x}}
with respect to x𝑥x is

|  |  |  |  |
| --- | --- | --- | --- |
|  | 12​x−μ​ω+x2​2​x3/2=2​x−(μ​ω+x)2​2​x3/2=x−μ​ω2​2​x3/2> 0,12𝑥𝜇𝜔𝑥22superscript𝑥322𝑥𝜇𝜔𝑥22superscript𝑥32𝑥𝜇𝜔22superscript𝑥32 0\displaystyle\frac{1}{\sqrt{2}\sqrt{x}}-\frac{\mu\omega+x}{2\sqrt{2}x^{3/2}}\ =\frac{2x-(\mu\omega+x)}{2\sqrt{2}x^{3/2}}\ =\ \frac{x-\mu\omega}{2\sqrt{2}x^{3/2}}\ >\ 0\ , |  | (151) |

since x>0.8⋅0.8𝑥⋅0.80.8x>0.8\cdot 0.8 and μ​ω<0.1⋅0.1𝜇𝜔⋅0.10.1\mu\omega<0.1\cdot 0.1.
∎

###### Lemma 28 (Function μ​ω+2​ν​τ2​ν​τ𝜇𝜔2𝜈𝜏2𝜈𝜏\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}).

h2​(μ,ω,ν,τ)=μ​ω+2​ν​τ2​ν​τsubscriptℎ2𝜇𝜔𝜈𝜏𝜇𝜔2𝜈𝜏2𝜈𝜏h\_{2}(\mu,\omega,\nu,\tau)=\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}
is larger than zero and increasing in both ν​τ𝜈𝜏\nu\tau and μ​ω𝜇𝜔\mu\omega.
It has minimal value t2=1.1225subscript𝑡21.1225t\_{2}=1.1225 and maximal value
T2=1.9417subscript𝑇21.9417T\_{2}=1.9417.

###### Proof.

The derivative of the function
μ​ω+2​x2​x𝜇𝜔2𝑥2𝑥\frac{\mu\omega+2x}{\sqrt{2}\sqrt{x}}
with respect to x𝑥x is

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2x−μ​ω+2​x2​2​x3/2=4​x−(μ​ω+2​x)2​2​x3/2=2​x−μ​ω2​2​x3/2> 0.2𝑥𝜇𝜔2𝑥22superscript𝑥324𝑥𝜇𝜔2𝑥22superscript𝑥322𝑥𝜇𝜔22superscript𝑥32 0\displaystyle\frac{\sqrt{2}}{\sqrt{x}}-\frac{\mu\omega+2x}{2\sqrt{2}x^{3/2}}=\frac{4x-(\mu\omega+2x)}{2\sqrt{2}x^{3/2}}=\frac{2x-\mu\omega}{2\sqrt{2}x^{3/2}}\ >\ 0\ . |  | (152) |

∎

###### Lemma 29 (Function μ​ω2​ν​τ𝜇𝜔2𝜈𝜏\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}).

h3​(μ,ω,ν,τ)=μ​ω2​ν​τsubscriptℎ3𝜇𝜔𝜈𝜏𝜇𝜔2𝜈𝜏h\_{3}(\mu,\omega,\nu,\tau)=\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}
monotonically decreasing in ν​τ𝜈𝜏\nu\tau and monotonically increasing in μ​ω𝜇𝜔\mu\omega.
It has minimal value t3=−0.0088388subscript𝑡30.0088388t\_{3}=-0.0088388 and maximal value
T3=0.0088388subscript𝑇30.0088388T\_{3}=0.0088388.

###### Proof.

Obvious.
∎

###### Lemma 30 (Function (μ​ω2​ν​τ)2superscript𝜇𝜔2𝜈𝜏2\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}).

h4​(μ,ω,ν,τ)=(μ​ω2​ν​τ)2subscriptℎ4𝜇𝜔𝜈𝜏superscript𝜇𝜔2𝜈𝜏2h\_{4}(\mu,\omega,\nu,\tau)=\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}
has a minimum at 0 for μ=0𝜇0\mu=0 or ω=0𝜔0\omega=0 and has a maximum for
the smallest ν​τ𝜈𝜏\nu\tau and largest |μ​ω|𝜇𝜔\left|\mu\omega\right| and is larger or equal to zero.
It has minimal value t4=0subscript𝑡40t\_{4}=0 and maximal value
T4=0.000078126subscript𝑇40.000078126T\_{4}=0.000078126.

###### Proof.

Obvious.
∎

###### Lemma 31 (Function 2π​(α−1)ν​τ2𝜋𝛼1𝜈𝜏\frac{\sqrt{\frac{2}{\pi}}(\alpha-1)}{\sqrt{\nu\tau}}).

2π​(α−1)ν​τ>02𝜋𝛼1𝜈𝜏0\frac{\sqrt{\frac{2}{\pi}}(\alpha-1)}{\sqrt{\nu\tau}}>0
and decreasing in ν​τ𝜈𝜏\nu\tau.

###### Proof.

Statements follow directly from elementary functions square root and
division.
∎

###### Lemma 32 (Function 2−erfc⁡(μ​ω2​ν​τ)2erfc𝜇𝜔2𝜈𝜏2-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)).

2−erfc⁡(μ​ω2​ν​τ)>02erfc𝜇𝜔2𝜈𝜏02-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)>0
and decreasing in ν​τ𝜈𝜏\nu\tau and increasing in
μ​ω𝜇𝜔\mu\omega.

###### Proof.

Statements follow directly from Lemma [21](#Thmtheorem21 "Lemma 21 (Basic functions). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") and erfcerfc\operatorname{erfc}.
∎

###### Lemma 33 (Function 2π​((α−1)​μ​ω(ν​τ)3/2−αν​τ)2𝜋𝛼1𝜇𝜔superscript𝜈𝜏32𝛼𝜈𝜏\sqrt{\frac{2}{\pi}}\left(\frac{(\alpha-1)\mu\omega}{(\nu\tau)^{3/2}}-\frac{\alpha}{\sqrt{\nu\tau}}\right)).

For λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01} and α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01},
2π​((α−1)​μ​ω(ν​τ)3/2−αν​τ)<02𝜋𝛼1𝜇𝜔superscript𝜈𝜏32𝛼𝜈𝜏0\sqrt{\frac{2}{\pi}}\left(\frac{(\alpha-1)\mu\omega}{(\nu\tau)^{3/2}}-\frac{\alpha}{\sqrt{\nu\tau}}\right)<0
and increasing in both ν​τ𝜈𝜏\nu\tau and μ​ω𝜇𝜔\mu\omega.

###### Proof.

We consider the function
2π​((α−1)​μ​ωx3/2−αx)2𝜋𝛼1𝜇𝜔superscript𝑥32𝛼𝑥\sqrt{\frac{2}{\pi}}\left(\frac{(\alpha-1)\mu\omega}{x^{3/2}}-\frac{\alpha}{\sqrt{x}}\right),
which has the derivative with respect to x𝑥x:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2π​(α2​x3/2−3​(α−1)​μ​ω2​x5/2).2𝜋𝛼2superscript𝑥323𝛼1𝜇𝜔2superscript𝑥52\displaystyle\sqrt{\frac{2}{\pi}}\left(\frac{\alpha}{2x^{3/2}}-\frac{3(\alpha-1)\mu\omega}{2x^{5/2}}\right)\ . |  | (153) |

This derivative is larger than zero, since

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2π​(α2​(ν​τ)3/2−3​(α−1)​μ​ω2​(ν​τ)5/2)>2π​(α−3​(α−1)​μ​ων​τ)2​(ν​τ)3/2> 0.2𝜋𝛼2superscript𝜈𝜏323𝛼1𝜇𝜔2superscript𝜈𝜏522𝜋𝛼3𝛼1𝜇𝜔𝜈𝜏2superscript𝜈𝜏32 0\displaystyle\sqrt{\frac{2}{\pi}}\left(\frac{\alpha}{2(\nu\tau)^{3/2}}-\frac{3(\alpha-1)\mu\omega}{2(\nu\tau)^{5/2}}\right)\ >\frac{\sqrt{\frac{2}{\pi}}\left(\alpha-\frac{3(\alpha-1)\mu\omega}{\nu\tau}\right)}{2(\nu\tau)^{3/2}}\ >\ 0\ . |  | (154) |

The last inequality follows from
α−3⋅0.1⋅0.1​(α−1)0.8⋅0.8>0𝛼⋅30.10.1𝛼1⋅0.80.80\alpha-\frac{3\cdot 0.1\cdot 0.1(\alpha-1)}{0.8\cdot 0.8}>0 for α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01}.

We next consider the function
2π​((α−1)​x(ν​τ)3/2−αν​τ)2𝜋𝛼1𝑥superscript𝜈𝜏32𝛼𝜈𝜏\sqrt{\frac{2}{\pi}}\left(\frac{(\alpha-1)x}{(\nu\tau)^{3/2}}-\frac{\alpha}{\sqrt{\nu\tau}}\right),
which has the derivative with respect to x𝑥x:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2π​(α−1)(ν​τ)3/2> 0.2𝜋𝛼1superscript𝜈𝜏32 0\displaystyle\frac{\sqrt{\frac{2}{\pi}}(\alpha-1)}{(\nu\tau)^{3/2}}\ >\ 0\ . |  | (155) |

∎

###### Lemma 34 (Function 2π​((−1)​(α−1)​μ2​ω2(ν​τ)3/2+−α+α​μ​ω+1ν​τ−α​ν​τ)2𝜋1𝛼1superscript𝜇2superscript𝜔2superscript𝜈𝜏32𝛼𝛼𝜇𝜔1𝜈𝜏𝛼𝜈𝜏\sqrt{\frac{2}{\pi}}\left(\frac{(-1)(\alpha-1)\mu^{2}\omega^{2}}{(\nu\tau)^{3/2}}+\frac{-\alpha+\alpha\mu\omega+1}{\sqrt{\nu\tau}}-\alpha\sqrt{\nu\tau}\right)).

The function
  
2π​((−1)​(α−1)​μ2​ω2(ν​τ)3/2+−α+α​μ​ω+1ν​τ−α​ν​τ)<02𝜋1𝛼1superscript𝜇2superscript𝜔2superscript𝜈𝜏32𝛼𝛼𝜇𝜔1𝜈𝜏𝛼𝜈𝜏0\sqrt{\frac{2}{\pi}}\left(\frac{(-1)(\alpha-1)\mu^{2}\omega^{2}}{(\nu\tau)^{3/2}}+\frac{-\alpha+\alpha\mu\omega+1}{\sqrt{\nu\tau}}-\alpha\sqrt{\nu\tau}\right)<0
is decreasing in ν​τ𝜈𝜏\nu\tau and increasing in μ​ω𝜇𝜔\mu\omega.

###### Proof.

We define the function

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2π​((−1)​(α−1)​μ2​ω2x3/2+−α+α​μ​ω+1x−α​x)2𝜋1𝛼1superscript𝜇2superscript𝜔2superscript𝑥32𝛼𝛼𝜇𝜔1𝑥𝛼𝑥\displaystyle\sqrt{\frac{2}{\pi}}\left(\frac{(-1)(\alpha-1)\mu^{2}\omega^{2}}{x^{3/2}}+\frac{-\alpha+\alpha\mu\omega+1}{\sqrt{x}}-\alpha\sqrt{x}\right) |  | (156) |

which has as derivative with respect to x𝑥x:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2π​(3​(α−1)​μ2​ω22​x5/2−−α+α​μ​ω+12​x3/2−α2​x)=2𝜋3𝛼1superscript𝜇2superscript𝜔22superscript𝑥52𝛼𝛼𝜇𝜔12superscript𝑥32𝛼2𝑥absent\displaystyle\sqrt{\frac{2}{\pi}}\left(\frac{3(\alpha-1)\mu^{2}\omega^{2}}{2x^{5/2}}-\frac{-\alpha+\alpha\mu\omega+1}{2x^{3/2}}-\frac{\alpha}{2\sqrt{x}}\right)\ = |  | (157) |
|  |  |  |
| --- | --- | --- |
|  | 12​π​x5/2​(3​(α−1)​μ2​ω2−x​(−α+α​μ​ω+1)−α​x2).12𝜋superscript𝑥523𝛼1superscript𝜇2superscript𝜔2𝑥𝛼𝛼𝜇𝜔1𝛼superscript𝑥2\displaystyle\frac{1}{\sqrt{2\pi}x^{5/2}}\left(3(\alpha-1)\mu^{2}\omega^{2}-x(-\alpha+\alpha\mu\omega+1)-\alpha x^{2}\right)\ . |  |

The derivative of the term
3​(α−1)​μ2​ω2−x​(−α+α​μ​ω+1)−α​x23𝛼1superscript𝜇2superscript𝜔2𝑥𝛼𝛼𝜇𝜔1𝛼superscript𝑥23(\alpha-1)\mu^{2}\omega^{2}-x(-\alpha+\alpha\mu\omega+1)-\alpha x^{2}
with respect to x𝑥x is
−1+α−μ​ω​α−2​α​x<01𝛼𝜇𝜔𝛼2𝛼𝑥0-1+\alpha-\mu\omega\alpha-2\alpha x<0, since
2​α​x>1.6​α2𝛼𝑥1.6𝛼2\alpha x>1.6\alpha.
Therefore the term is maximized with the smallest value for x𝑥x, which
is x=ν​τ=0.8⋅0.8𝑥𝜈𝜏⋅0.80.8x=\nu\tau=0.8\cdot 0.8.
For μ​ω𝜇𝜔\mu\omega we use for each term the value which gives maximal
contribution. We obtain an upper bound for the term:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 3​(−0.1⋅0.1)2​(α01−1)−(0.8⋅0.8)2​α01−0.8⋅0.8​((−0.1⋅0.1)​α01−α01+1)=−0.243569.3superscript⋅0.10.12subscript𝛼011superscript⋅0.80.82subscript𝛼01⋅0.80.8⋅0.10.1subscript𝛼01subscript𝛼0110.243569\displaystyle 3(-0.1\cdot 0.1)^{2}(\alpha\_{\rm 01}-1)-(0.8\cdot 0.8)^{2}\alpha\_{\rm 01}-0.8\cdot 0.8((-0.1\cdot 0.1)\alpha\_{\rm 01}-\alpha\_{\rm 01}+1)\ =\ -0.243569\ . |  | (158) |

Therefore the derivative with respect to x=ν​τ𝑥𝜈𝜏x=\nu\tau
is smaller than zero and the original function is decreasing in ν​τ𝜈𝜏\nu\tau

We now consider the derivative with respect to x=μ​ω𝑥𝜇𝜔x=\mu\omega.
The derivative with respect to x𝑥x of the function

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2π​(−α​ν​τ−(α−1)​x2(ν​τ)3/2+−α+α​x+1ν​τ)2𝜋𝛼𝜈𝜏𝛼1superscript𝑥2superscript𝜈𝜏32𝛼𝛼𝑥1𝜈𝜏\displaystyle\sqrt{\frac{2}{\pi}}\left(-\alpha\sqrt{\nu\tau}-\frac{(\alpha-1)x^{2}}{(\nu\tau)^{3/2}}+\frac{-\alpha+\alpha x+1}{\sqrt{\nu\tau}}\right) |  | (159) |

is

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2π​(α​ν​τ−2​(α−1)​x)(ν​τ)3/2.2𝜋𝛼𝜈𝜏2𝛼1𝑥superscript𝜈𝜏32\displaystyle\frac{\sqrt{\frac{2}{\pi}}(\alpha\nu\tau-2(\alpha-1)x)}{(\nu\tau)^{3/2}}\ . |  | (160) |

Since
−2​x​(−1+α)+ν​τ​α>−2⋅0.01⋅(−1+α01)+0.8⋅0.8​α01>1.0574>02𝑥1𝛼𝜈𝜏𝛼⋅20.011subscript𝛼01⋅0.80.8subscript𝛼011.05740-2x(-1+\alpha)+\nu\tau\alpha>-2\cdot 0.01\cdot(-1+\alpha\_{\rm 01})+0.8\cdot 0.8\alpha\_{\rm 01}>1.0574>0, the derivative is larger than zero.
Consequently, the original function is increasing in μ​ω𝜇𝜔\mu\omega.

The maximal value is obtained with the minimal ν​τ=0.8⋅0.8𝜈𝜏⋅0.80.8\nu\tau=0.8\cdot 0.8 and the maximal μ​ω=0.1⋅0.1𝜇𝜔⋅0.10.1\mu\omega=0.1\cdot 0.1.
The maximal value is

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2π​(0.1⋅0.1​α01−α01+10.8⋅0.8+0.12​0.12​(−1)​(α01−1)(0.8⋅0.8)3/2−0.8⋅0.8​α01)=−1.72296.2𝜋⋅0.10.1subscript𝛼01subscript𝛼011⋅0.80.8superscript0.12superscript0.121subscript𝛼011superscript⋅0.80.832⋅0.80.8subscript𝛼011.72296\displaystyle\sqrt{\frac{2}{\pi}}\left(\frac{0.1\cdot 0.1\alpha\_{\rm 01}-\alpha\_{\rm 01}+1}{\sqrt{0.8\cdot 0.8}}+\frac{0.1^{2}0.1^{2}(-1)(\alpha\_{\rm 01}-1)}{(0.8\cdot 0.8)^{3/2}}-\sqrt{0.8\cdot 0.8}\alpha\_{\rm 01}\right)\ =\ -1.72296\ . |  | (161) |

Therefore the original function is smaller than zero.
∎

###### Lemma 35 (Function 2π​((α2−1)​μ​ω(ν​τ)3/2−3​α2ν​τ)2𝜋superscript𝛼21𝜇𝜔superscript𝜈𝜏323superscript𝛼2𝜈𝜏\sqrt{\frac{2}{\pi}}\left(\frac{\left(\alpha^{2}-1\right)\mu\omega}{(\nu\tau)^{3/2}}-\frac{3\alpha^{2}}{\sqrt{\nu\tau}}\right)).

For λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01} and α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01},
  
2π​((α2−1)​μ​ω(ν​τ)3/2−3​α2ν​τ)<02𝜋superscript𝛼21𝜇𝜔superscript𝜈𝜏323superscript𝛼2𝜈𝜏0\sqrt{\frac{2}{\pi}}\left(\frac{\left(\alpha^{2}-1\right)\mu\omega}{(\nu\tau)^{3/2}}-\frac{3\alpha^{2}}{\sqrt{\nu\tau}}\right)<0
and increasing in both ν​τ𝜈𝜏\nu\tau and μ​ω𝜇𝜔\mu\omega.

###### Proof.

The derivative of the function

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2π​((α2−1)​μ​ωx3/2−3​α2x)2𝜋superscript𝛼21𝜇𝜔superscript𝑥323superscript𝛼2𝑥\displaystyle\sqrt{\frac{2}{\pi}}\left(\frac{\left(\alpha^{2}-1\right)\mu\omega}{x^{3/2}}-\frac{3\alpha^{2}}{\sqrt{x}}\right) |  | (162) |

with respect to x𝑥x is

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2π​(3​α22​x3/2−3​(α2−1)​μ​ω2​x5/2)=3​(α2​x−(α2−1)​μ​ω)2​π​x5/2> 0,2𝜋3superscript𝛼22superscript𝑥323superscript𝛼21𝜇𝜔2superscript𝑥523superscript𝛼2𝑥superscript𝛼21𝜇𝜔2𝜋superscript𝑥52 0\displaystyle\sqrt{\frac{2}{\pi}}\left(\frac{3\alpha^{2}}{2x^{3/2}}-\frac{3\left(\alpha^{2}-1\right)\mu\omega}{2x^{5/2}}\right)\ =\ \frac{3\left(\alpha^{2}x-\left(\alpha^{2}-1\right)\mu\omega\right)}{\sqrt{2\pi}x^{5/2}}\ >\ 0\ , |  | (163) |

since
α2​x−μ​ω​(−1+α2)>α012​0.8⋅0.8−0.1⋅0.1⋅(−1+α012)>1.77387superscript𝛼2𝑥𝜇𝜔1superscript𝛼2⋅superscriptsubscript𝛼0120.80.8⋅0.10.11superscriptsubscript𝛼0121.77387\alpha^{2}x-\mu\omega(-1+\alpha^{2})>\alpha\_{\rm 01}^{2}0.8\cdot 0.8-0.1\cdot 0.1\cdot(-1+\alpha\_{\rm 01}^{2})>1.77387

The derivative of the function

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2π​((α2−1)​x(ν​τ)3/2−3​α2ν​τ)2𝜋superscript𝛼21𝑥superscript𝜈𝜏323superscript𝛼2𝜈𝜏\displaystyle\sqrt{\frac{2}{\pi}}\left(\frac{\left(\alpha^{2}-1\right)x}{(\nu\tau)^{3/2}}-\frac{3\alpha^{2}}{\sqrt{\nu\tau}}\right) |  | (164) |

with respect to x𝑥x is

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2π​(α2−1)(ν​τ)3/2> 0.2𝜋superscript𝛼21superscript𝜈𝜏32 0\displaystyle\frac{\sqrt{\frac{2}{\pi}}\left(\alpha^{2}-1\right)}{(\nu\tau)^{3/2}}\ >\ 0\ . |  | (165) |

The maximal function value is obtained by maximal ν​τ=1.5⋅1.25𝜈𝜏⋅1.51.25\nu\tau=1.5\cdot 1.25 and the maximal μ​ω=0.1⋅0.1𝜇𝜔⋅0.10.1\mu\omega=0.1\cdot 0.1.
The maximal value is
2π​(0.1⋅0.1​(α012−1)(1.5⋅1.25)3/2−3​α0121.5⋅1.25)=−4.888692𝜋⋅0.10.1superscriptsubscript𝛼0121superscript⋅1.51.25323superscriptsubscript𝛼012⋅1.51.254.88869\sqrt{\frac{2}{\pi}}\left(\frac{0.1\cdot 0.1\left(\alpha\_{\rm 01}^{2}-1\right)}{(1.5\cdot 1.25)^{3/2}}-\frac{3\alpha\_{\rm 01}^{2}}{\sqrt{1.5\cdot 1.25}}\right)\ =\ -4.88869.
Therefore the function is negative.
∎

###### Lemma 36 (Function 2π​((α2−1)​μ​ων​τ−3​α2​ν​τ)2𝜋superscript𝛼21𝜇𝜔𝜈𝜏3superscript𝛼2𝜈𝜏\sqrt{\frac{2}{\pi}}\left(\frac{\left(\alpha^{2}-1\right)\mu\omega}{\sqrt{\nu\tau}}-3\alpha^{2}\sqrt{\nu\tau}\right)).

The function
2π​((α2−1)​μ​ων​τ−3​α2​ν​τ)<02𝜋superscript𝛼21𝜇𝜔𝜈𝜏3superscript𝛼2𝜈𝜏0\sqrt{\frac{2}{\pi}}\left(\frac{\left(\alpha^{2}-1\right)\mu\omega}{\sqrt{\nu\tau}}-3\alpha^{2}\sqrt{\nu\tau}\right)<0
is decreasing in ν​τ𝜈𝜏\nu\tau and increasing in μ​ω𝜇𝜔\mu\omega.

###### Proof.

The derivative of the function

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2π​((α2−1)​μ​ωx−3​α2​x)2𝜋superscript𝛼21𝜇𝜔𝑥3superscript𝛼2𝑥\displaystyle\sqrt{\frac{2}{\pi}}\left(\frac{\left(\alpha^{2}-1\right)\mu\omega}{\sqrt{x}}-3\alpha^{2}\sqrt{x}\right) |  | (166) |

with respect to x𝑥x is

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2π​(−(α2−1)​μ​ω2​x3/2−3​α22​x)=−(α2−1)​μ​ω−3​α2​x2​π​x3/2< 0,2𝜋superscript𝛼21𝜇𝜔2superscript𝑥323superscript𝛼22𝑥superscript𝛼21𝜇𝜔3superscript𝛼2𝑥2𝜋superscript𝑥32 0\displaystyle\sqrt{\frac{2}{\pi}}\left(-\frac{\left(\alpha^{2}-1\right)\mu\omega}{2x^{3/2}}-\frac{3\alpha^{2}}{2\sqrt{x}}\right)\ =\ \frac{-\left(\alpha^{2}-1\right)\mu\omega-3\alpha^{2}x}{\sqrt{2\pi}x^{3/2}}\ <\ 0\ , |  | (167) |

since
−3​α2​x−μ​ω​(−1+α2)<−3​α012​0.8⋅0.8+0.1⋅0.1​(−1+α012)<−5.357643superscript𝛼2𝑥𝜇𝜔1superscript𝛼2⋅3superscriptsubscript𝛼0120.80.8⋅0.10.11superscriptsubscript𝛼0125.35764-3\alpha^{2}x-\mu\omega(-1+\alpha^{2})<-3\alpha\_{\rm 01}^{2}0.8\cdot 0.8+0.1\cdot 0.1(-1+\alpha\_{\rm 01}^{2})<-5.35764.

The derivative of the function

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2π​((α2−1)​xν​τ−3​α2​ν​τ)2𝜋superscript𝛼21𝑥𝜈𝜏3superscript𝛼2𝜈𝜏\displaystyle\sqrt{\frac{2}{\pi}}\left(\frac{\left(\alpha^{2}-1\right)x}{\sqrt{\nu\tau}}-3\alpha^{2}\sqrt{\nu\tau}\right) |  | (168) |

with respect to x𝑥x is

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2π​(α2−1)ν​τ> 0.2𝜋superscript𝛼21𝜈𝜏 0\displaystyle\frac{\sqrt{\frac{2}{\pi}}\left(\alpha^{2}-1\right)}{\sqrt{\nu\tau}}\ >\ 0\ . |  | (169) |

The maximal function value is obtained for
minimal ν​τ=0.8⋅0.8𝜈𝜏⋅0.80.8\nu\tau=0.8\cdot 0.8 and the maximal μ​ω=0.1⋅0.1𝜇𝜔⋅0.10.1\mu\omega=0.1\cdot 0.1.
The value is
2π​(0.1⋅0.1​(α012−1)0.8⋅0.8−3​0.8⋅0.8​α012)=−5.343472𝜋⋅0.10.1superscriptsubscript𝛼0121⋅0.80.83⋅0.80.8superscriptsubscript𝛼0125.34347\sqrt{\frac{2}{\pi}}\left(\frac{0.1\cdot 0.1\left(\alpha\_{\rm 01}^{2}-1\right)}{\sqrt{0.8\cdot 0.8}}-3\sqrt{0.8\cdot 0.8}\alpha\_{\rm 01}^{2}\right)\ =\ -5.34347.
Thus, the function is negative.
∎

###### Lemma 37 (Function ν​τ​e(μ​ω+ν​τ)22​ν​τ​erfc⁡(μ​ω+ν​τ2​ν​τ)𝜈𝜏superscript𝑒superscript𝜇𝜔𝜈𝜏22𝜈𝜏erfc𝜇𝜔𝜈𝜏2𝜈𝜏\nu\tau e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)).

The function
ν​τ​e(μ​ω+ν​τ)22​ν​τ​erfc⁡(μ​ω+ν​τ2​ν​τ)>0𝜈𝜏superscript𝑒superscript𝜇𝜔𝜈𝜏22𝜈𝜏erfc𝜇𝜔𝜈𝜏2𝜈𝜏0\nu\tau e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)>0
is increasing in ν​τ𝜈𝜏\nu\tau and decreasing in μ​ω𝜇𝜔\mu\omega.

###### Proof.

The derivative of the function

|  |  |  |  |
| --- | --- | --- | --- |
|  | x​e(μ​ω+x)22​x​erfc⁡(μ​ω+x2​x)𝑥superscript𝑒superscript𝜇𝜔𝑥22𝑥erfc𝜇𝜔𝑥2𝑥\displaystyle xe^{\frac{(\mu\omega+x)^{2}}{2x}}\operatorname{erfc}\left(\frac{\mu\omega+x}{\sqrt{2}\sqrt{x}}\right) |  | (170) |

with respect to x𝑥x is

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(μ​ω+x)22​x​(x​(x+2)−μ2​ω2)​erfc⁡(μ​ω+x2​x)2​x+μ​ω−x2​π​x.superscript𝑒superscript𝜇𝜔𝑥22𝑥𝑥𝑥2superscript𝜇2superscript𝜔2erfc𝜇𝜔𝑥2𝑥2𝑥𝜇𝜔𝑥2𝜋𝑥\displaystyle\frac{e^{\frac{(\mu\omega+x)^{2}}{2x}}\left(x(x+2)-\mu^{2}\omega^{2}\right)\operatorname{erfc}\left(\frac{\mu\omega+x}{\sqrt{2}\sqrt{x}}\right)}{2x}+\frac{\mu\omega-x}{\sqrt{2\pi}\sqrt{x}}\ . |  | (171) |

This derivative is larger than zero, since

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(μ​ω+ν​τ)22​ν​τ​(ν​τ​(ν​τ+2)−μ2​ω2)​erfc⁡(μ​ω+ν​τ2​ν​τ)2​ν​τ+μ​ω−ν​τ2​π​ν​τ>superscript𝑒superscript𝜇𝜔𝜈𝜏22𝜈𝜏𝜈𝜏𝜈𝜏2superscript𝜇2superscript𝜔2erfc𝜇𝜔𝜈𝜏2𝜈𝜏2𝜈𝜏𝜇𝜔𝜈𝜏2𝜋𝜈𝜏absent\displaystyle\frac{e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\left(\nu\tau(\nu\tau+2)-\mu^{2}\omega^{2}\right)\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)}{2\nu\tau}+\frac{\mu\omega-\nu\tau}{\sqrt{2\pi}\sqrt{\nu\tau}}\ > |  | (172) |
|  |  |  |
| --- | --- | --- |
|  | 0.4349​(ν​τ​(ν​τ+2)−μ2​ω2)2​ν​τ+μ​ω−ν​τ2​π​ν​τ>0.4349𝜈𝜏𝜈𝜏2superscript𝜇2superscript𝜔22𝜈𝜏𝜇𝜔𝜈𝜏2𝜋𝜈𝜏absent\displaystyle\frac{0.4349\left(\nu\tau(\nu\tau+2)-\mu^{2}\omega^{2}\right)}{2\nu\tau}+\frac{\mu\omega-\nu\tau}{\sqrt{2\pi}\sqrt{\nu\tau}}\ > |  |
|  |  |  |
| --- | --- | --- |
|  | 0.5​(ν​τ​(ν​τ+2)−μ2​ω2)2​π​ν​τ+μ​ω−ν​τ2​π​ν​τ=0.5𝜈𝜏𝜈𝜏2superscript𝜇2superscript𝜔22𝜋𝜈𝜏𝜇𝜔𝜈𝜏2𝜋𝜈𝜏absent\displaystyle\frac{0.5\left(\nu\tau(\nu\tau+2)-\mu^{2}\omega^{2}\right)}{\sqrt{2\pi}\nu\tau}+\frac{\mu\omega-\nu\tau}{\sqrt{2\pi}\sqrt{\nu\tau}}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | 0.5​(ν​τ​(ν​τ+2)−μ2​ω2)+ν​τ​(μ​ω−ν​τ)2​π​ν​τ=0.5𝜈𝜏𝜈𝜏2superscript𝜇2superscript𝜔2𝜈𝜏𝜇𝜔𝜈𝜏2𝜋𝜈𝜏absent\displaystyle\frac{0.5\left(\nu\tau(\nu\tau+2)-\mu^{2}\omega^{2}\right)+\sqrt{\nu\tau}(\mu\omega-\nu\tau)}{\sqrt{2\pi}\nu\tau}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | −0.5​μ2​ω2+μ​ω​ν​τ+0.5​(ν​τ)2−ν​τ​ν​τ+ν​τ2​π​ν​τ=0.5superscript𝜇2superscript𝜔2𝜇𝜔𝜈𝜏0.5superscript𝜈𝜏2𝜈𝜏𝜈𝜏𝜈𝜏2𝜋𝜈𝜏absent\displaystyle\frac{-0.5\mu^{2}\omega^{2}+\mu\omega\sqrt{\nu\tau}+0.5(\nu\tau)^{2}-\nu\tau\sqrt{\nu\tau}+\nu\tau}{\sqrt{2\pi}\nu\tau}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | −0.5​μ2​ω2+μ​ω​ν​τ+(0.5​ν​τ−ν​τ)2+0.25​(ν​τ)22​π​ν​τ> 0.0.5superscript𝜇2superscript𝜔2𝜇𝜔𝜈𝜏superscript0.5𝜈𝜏𝜈𝜏20.25superscript𝜈𝜏22𝜋𝜈𝜏 0\displaystyle\frac{-0.5\mu^{2}\omega^{2}+\mu\omega\sqrt{\nu\tau}+\left(0.5\nu\tau-\sqrt{\nu\tau}\right)^{2}+0.25(\nu\tau)^{2}}{\sqrt{2\pi}\nu\tau}\ >\ 0\ . |  |

We explain this chain of inequalities:

* •

  The first inequality follows by applying Lemma [23](#Thmtheorem23 "Lemma 23 (Function 𝑒^𝑥²⁢erfc(𝑥)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
  which says that e(μ​ω+ν​τ)22​ν​τ​erfc⁡(μ​ω+ν​τ2​ν​τ)superscript𝑒superscript𝜇𝜔𝜈𝜏22𝜈𝜏erfc𝜇𝜔𝜈𝜏2𝜈𝜏e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)
  is strictly monotonically decreasing. The minimal value that is larger
  than 0.4349 is taken on at
  the maximal values ν​τ=1.5⋅1.25𝜈𝜏⋅1.51.25\nu\tau=1.5\cdot 1.25 and μ​ω=0.1⋅0.1𝜇𝜔⋅0.10.1\mu\omega=0.1\cdot 0.1.
* •

  The second inequality uses
  12​0.4349​2​π=0.545066>0.5120.43492𝜋0.5450660.5\frac{1}{2}0.4349\sqrt{2\pi}=0.545066>0.5.
* •

  The equalities are just algebraic reformulations.
* •

  The last inequality follows from
  −0.5​μ2​ω2+μ​ω​ν​τ+0.25​(ν​τ)2>0.25​(0.8⋅0.8)2−0.5⋅(0.1)2​(0.1)2−0.1⋅0.1⋅0.8⋅0.8=0.09435>00.5superscript𝜇2superscript𝜔2𝜇𝜔𝜈𝜏0.25superscript𝜈𝜏20.25superscript⋅0.80.82⋅0.5superscript0.12superscript0.12⋅0.10.1⋅0.80.80.094350-0.5\mu^{2}\omega^{2}+\mu\omega\sqrt{\nu\tau}+0.25(\nu\tau)^{2}>0.25(0.8\cdot 0.8)^{2}-0.5\cdot(0.1)^{2}(0.1)^{2}-0.1\cdot 0.1\cdot\sqrt{0.8\cdot 0.8}=0.09435>0.

Therefore the function is increasing in ν​τ𝜈𝜏\nu\tau.

Decreasing in μ​ω𝜇𝜔\mu\omega follows from decreasing of ex2​erfc⁡(x)superscript𝑒superscript𝑥2erfc𝑥e^{x^{2}}\operatorname{erfc}(x)
according to Lemma [23](#Thmtheorem23 "Lemma 23 (Function 𝑒^𝑥²⁢erfc(𝑥)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").
Positivity follows form the fact that erfcerfc\operatorname{erfc} and the
exponential function are positive and that ν​τ>0𝜈𝜏0\nu\tau>0.
∎

###### Lemma 38 (Function ν​τ​e(μ​ω+2​ν​τ)22​ν​τ​erfc⁡(μ​ω+2​ν​τ2​ν​τ)𝜈𝜏superscript𝑒superscript𝜇𝜔2𝜈𝜏22𝜈𝜏erfc𝜇𝜔2𝜈𝜏2𝜈𝜏\nu\tau e^{\frac{(\mu\omega+2\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)).

The function
ν​τ​e(μ​ω+2​ν​τ)22​ν​τ​erfc⁡(μ​ω+2​ν​τ2​ν​τ)>0𝜈𝜏superscript𝑒superscript𝜇𝜔2𝜈𝜏22𝜈𝜏erfc𝜇𝜔2𝜈𝜏2𝜈𝜏0\nu\tau e^{\frac{(\mu\omega+2\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)>0
is increasing in ν​τ𝜈𝜏\nu\tau and decreasing in μ​ω𝜇𝜔\mu\omega.

###### Proof.

The derivative of the function

|  |  |  |  |
| --- | --- | --- | --- |
|  | x​e(μ​ω+2​x)22​x​erfc⁡(μ​ω+2​x2​2​x)𝑥superscript𝑒superscript𝜇𝜔2𝑥22𝑥erfc𝜇𝜔2𝑥22𝑥\displaystyle xe^{\frac{(\mu\omega+2x)^{2}}{2x}}\operatorname{erfc}\left(\frac{\mu\omega+2x}{\sqrt{2}\sqrt{2x}}\right) |  | (173) |

is

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(μ​ω+2​x)24​x​(π​e(μ​ω+2​x)24​x​(2​x​(2​x+1)−μ2​ω2)​erfc⁡(μ​ω+2​x2​x)+x​(μ​ω−2​x))2​π​x.superscript𝑒superscript𝜇𝜔2𝑥24𝑥𝜋superscript𝑒superscript𝜇𝜔2𝑥24𝑥2𝑥2𝑥1superscript𝜇2superscript𝜔2erfc𝜇𝜔2𝑥2𝑥𝑥𝜇𝜔2𝑥2𝜋𝑥\displaystyle\frac{e^{\frac{(\mu\omega+2x)^{2}}{4x}}\left(\sqrt{\pi}e^{\frac{(\mu\omega+2x)^{2}}{4x}}\left(2x(2x+1)-\mu^{2}\omega^{2}\right)\operatorname{erfc}\left(\frac{\mu\omega+2x}{2\sqrt{x}}\right)+\sqrt{x}(\mu\omega-2x)\right)}{2\sqrt{\pi}x}\ . |  | (174) |

We only have to determine the sign of
π​e(μ​ω+2​x)24​x​(2​x​(2​x+1)−μ2​ω2)​erfc⁡(μ​ω+2​x2​x)+x​(μ​ω−2​x)𝜋superscript𝑒superscript𝜇𝜔2𝑥24𝑥2𝑥2𝑥1superscript𝜇2superscript𝜔2erfc𝜇𝜔2𝑥2𝑥𝑥𝜇𝜔2𝑥\sqrt{\pi}e^{\frac{(\mu\omega+2x)^{2}}{4x}}\left(2x(2x+1)-\mu^{2}\omega^{2}\right)\operatorname{erfc}\left(\frac{\mu\omega+2x}{2\sqrt{x}}\right)+\sqrt{x}(\mu\omega-2x)
since all other factors are obviously larger than zero.

This derivative is larger than zero, since

|  |  |  |  |
| --- | --- | --- | --- |
|  | π​e(μ​ω+2​ν​τ)24​ν​τ​(2​ν​τ​(2​ν​τ+1)−μ2​ω2)​erfc⁡(μ​ω+2​ν​τ2​ν​τ)+ν​τ​(μ​ω−2​ν​τ)>𝜋superscript𝑒superscript𝜇𝜔2𝜈𝜏24𝜈𝜏2𝜈𝜏2𝜈𝜏1superscript𝜇2superscript𝜔2erfc𝜇𝜔2𝜈𝜏2𝜈𝜏𝜈𝜏𝜇𝜔2𝜈𝜏absent\displaystyle\sqrt{\pi}e^{\frac{(\mu\omega+2\nu\tau)^{2}}{4\nu\tau}}\left(2\nu\tau(2\nu\tau+1)-\mu^{2}\omega^{2}\right)\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{2\sqrt{\nu\tau}}\right)+\sqrt{\nu\tau}(\mu\omega-2\nu\tau)\ > |  | (175) |
|  |  |  |
| --- | --- | --- |
|  | 0.463979​(2​ν​τ​(2​ν​τ+1)−μ2​ω2)+ν​τ​(μ​ω−2​ν​τ)=0.4639792𝜈𝜏2𝜈𝜏1superscript𝜇2superscript𝜔2𝜈𝜏𝜇𝜔2𝜈𝜏absent\displaystyle 0.463979\left(2\nu\tau(2\nu\tau+1)-\mu^{2}\omega^{2}\right)+\sqrt{\nu\tau}(\mu\omega-2\nu\tau)\ = |  |
|  |  |  |
| --- | --- | --- |
|  | −0.463979​μ2​ω2+μ​ω​ν​τ+1.85592​(ν​τ)2+0.927958​ν​τ−2​ν​τ​ν​τ=0.463979superscript𝜇2superscript𝜔2𝜇𝜔𝜈𝜏1.85592superscript𝜈𝜏20.927958𝜈𝜏2𝜈𝜏𝜈𝜏absent\displaystyle-0.463979\mu^{2}\omega^{2}+\mu\omega\sqrt{\nu\tau}+1.85592(\nu\tau)^{2}+0.927958\nu\tau-2\nu\tau\sqrt{\nu\tau}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | μ​ω​(ν​τ−0.463979​μ​ω)+0.85592​(ν​τ)2+(ν​τ−ν​τ)2−0.0720421​ν​τ> 0.𝜇𝜔𝜈𝜏0.463979𝜇𝜔0.85592superscript𝜈𝜏2superscript𝜈𝜏𝜈𝜏20.0720421𝜈𝜏 0\displaystyle\mu\omega\left(\sqrt{\nu\tau}-0.463979\mu\omega\right)+0.85592(\nu\tau)^{2}+\left(\nu\tau-\sqrt{\nu\tau}\right)^{2}-0.0720421\nu\tau\ >\ 0\ . |  |

We explain this chain of inequalities:

* •

  The first inequality follows by applying Lemma [23](#Thmtheorem23 "Lemma 23 (Function 𝑒^𝑥²⁢erfc(𝑥)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
  which says that e(μ​ω+2​ν​τ)22​ν​τ​erfc⁡(μ​ω+2​ν​τ2​ν​τ)superscript𝑒superscript𝜇𝜔2𝜈𝜏22𝜈𝜏erfc𝜇𝜔2𝜈𝜏2𝜈𝜏e^{\frac{(\mu\omega+2\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)
  is strictly monotonically decreasing. The minimal value that is larger
  than 0.261772 is taken on at
  the maximal values ν​τ=1.5⋅1.25𝜈𝜏⋅1.51.25\nu\tau=1.5\cdot 1.25 and μ​ω=0.1⋅0.1𝜇𝜔⋅0.10.1\mu\omega=0.1\cdot 0.1.
  0.261772​π>0.4639790.261772𝜋0.4639790.261772\sqrt{\pi}>0.463979.
* •

  The equalities are just algebraic reformulations.
* •

  The last inequality follows from
  μ​ω​(ν​τ−0.463979​μ​ω)+0.85592​(ν​τ)2−0.0720421​ν​τ>0.85592⋅(0.8⋅0.8)2−0.1⋅0.1​(1.5⋅1.25+0.1⋅0.1⋅0.463979)−0.0720421⋅1.5⋅1.25>0.201766𝜇𝜔𝜈𝜏0.463979𝜇𝜔0.85592superscript𝜈𝜏20.0720421𝜈𝜏⋅0.85592superscript⋅0.80.82⋅0.10.1⋅1.51.25⋅0.10.10.463979⋅0.07204211.51.250.201766\mu\omega\left(\sqrt{\nu\tau}-0.463979\mu\omega\right)+0.85592(\nu\tau)^{2}-0.0720421\nu\tau>0.85592\cdot(0.8\cdot 0.8)^{2}-0.1\cdot 0.1\left(\sqrt{1.5\cdot 1.25}+0.1\cdot 0.1\cdot 0.463979\right)-0.0720421\cdot 1.5\cdot 1.25>0.201766.

Therefore the function is increasing in ν​τ𝜈𝜏\nu\tau.

Decreasing in μ​ω𝜇𝜔\mu\omega follows from decreasing of ex2​erfc⁡(x)superscript𝑒superscript𝑥2erfc𝑥e^{x^{2}}\operatorname{erfc}(x)
according to Lemma [23](#Thmtheorem23 "Lemma 23 (Function 𝑒^𝑥²⁢erfc(𝑥)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").
Positivity follows from the fact that erfcerfc\operatorname{erfc} and the
exponential function are positive and that ν​τ>0𝜈𝜏0\nu\tau>0.
∎

###### Lemma 39 (Bounds on the Derivatives).

The following bounds on the absolute values of the
derivatives of the Jacobian entries 𝒥11​(μ,ω,ν,τ,λ,α)subscript𝒥11𝜇𝜔𝜈𝜏𝜆𝛼{\mathcal{J}}\_{11}(\mu,\omega,\nu,\tau,\lambda,\alpha),
𝒥12​(μ,ω,ν,τ,λ,α)subscript𝒥12𝜇𝜔𝜈𝜏𝜆𝛼{\mathcal{J}}\_{12}(\mu,\omega,\nu,\tau,\lambda,\alpha),
𝒥21​(μ,ω,ν,τ,λ,α)subscript𝒥21𝜇𝜔𝜈𝜏𝜆𝛼{\mathcal{J}}\_{21}(\mu,\omega,\nu,\tau,\lambda,\alpha), and
𝒥22​(μ,ω,ν,τ,λ,α)subscript𝒥22𝜇𝜔𝜈𝜏𝜆𝛼{\mathcal{J}}\_{22}(\mu,\omega,\nu,\tau,\lambda,\alpha)
with respect to
μ𝜇\mu, ω𝜔\omega, ν𝜈\nu, and τ𝜏\tau hold:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | |∂𝒥11∂μ|subscript𝒥11𝜇\displaystyle\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\mu}\right|\ | < 0.0031049101995398316absent0.0031049101995398316\displaystyle<\ 0.0031049101995398316 |  | (176) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥11∂ω|subscript𝒥11𝜔\displaystyle\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\omega}\right|\ | < 1.055872374194189absent1.055872374194189\displaystyle<\ 1.055872374194189 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥11∂ν|subscript𝒥11𝜈\displaystyle\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\nu}\right|\ | < 0.031242911235461816absent0.031242911235461816\displaystyle<\ 0.031242911235461816 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥11∂τ|subscript𝒥11𝜏\displaystyle\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\tau}\right|\ | < 0.03749149348255419absent0.03749149348255419\displaystyle<\ 0.03749149348255419 |  |

|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥12∂μ|subscript𝒥12𝜇\displaystyle\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\mu}\right|\ | < 0.031242911235461816absent0.031242911235461816\displaystyle<\ 0.031242911235461816 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥12∂ω|subscript𝒥12𝜔\displaystyle\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\omega}\right|\ | < 0.031242911235461816absent0.031242911235461816\displaystyle<\ 0.031242911235461816 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥12∂ν|subscript𝒥12𝜈\displaystyle\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\nu}\right|\ | < 0.21232788238624354absent0.21232788238624354\displaystyle<\ 0.21232788238624354 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥12∂τ|subscript𝒥12𝜏\displaystyle\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\tau}\right|\ | < 0.2124377655377270absent0.2124377655377270\displaystyle<\ 0.2124377655377270 |  |

|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥21∂μ|subscript𝒥21𝜇\displaystyle\left|\frac{\partial{\mathcal{J}}\_{21}}{\partial\mu}\right|\ | < 0.02220441024325437absent0.02220441024325437\displaystyle<\ 0.02220441024325437 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥21∂ω|subscript𝒥21𝜔\displaystyle\left|\frac{\partial{\mathcal{J}}\_{21}}{\partial\omega}\right|\ | < 1.146955401845684absent1.146955401845684\displaystyle<\ 1.146955401845684 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥21∂ν|subscript𝒥21𝜈\displaystyle\left|\frac{\partial{\mathcal{J}}\_{21}}{\partial\nu}\right|\ | < 0.14983446469110305absent0.14983446469110305\displaystyle<\ 0.14983446469110305 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥21∂τ|subscript𝒥21𝜏\displaystyle\left|\frac{\partial{\mathcal{J}}\_{21}}{\partial\tau}\right|\ | < 0.17980135762932363absent0.17980135762932363\displaystyle<\ 0.17980135762932363 |  |

|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥22∂μ|subscript𝒥22𝜇\displaystyle\left|\frac{\partial{\mathcal{J}}\_{22}}{\partial\mu}\right|\ | < 0.14983446469110305absent0.14983446469110305\displaystyle<\ 0.14983446469110305 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥22∂ω|subscript𝒥22𝜔\displaystyle\left|\frac{\partial{\mathcal{J}}\_{22}}{\partial\omega}\right|\ | < 0.14983446469110305absent0.14983446469110305\displaystyle<\ 0.14983446469110305 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥22∂ν|subscript𝒥22𝜈\displaystyle\left|\frac{\partial{\mathcal{J}}\_{22}}{\partial\nu}\right|\ | < 1.805740052651535absent1.805740052651535\displaystyle<\ 1.805740052651535 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥22∂τ|subscript𝒥22𝜏\displaystyle\left|\frac{\partial{\mathcal{J}}\_{22}}{\partial\tau}\right|\ | < 2.396685907216327absent2.396685907216327\displaystyle<\ 2.396685907216327 |  |

###### Proof.

For each derivative we compute a lower and an upper bound and take the
maximum of the absolute value.
A lower bound is determined by minimizing the single terms of the
functions that represents the derivative. An upper bound is determined
by maximizing the single terms of the functions that represent the
derivative. Terms can be combined to larger terms for which
the maximum and the minimum must be known. We apply many previous lemmata
which state properties of functions representing single or combined
terms. The more terms are combined, the tighter the bounds can be
made.

Next we go through all the derivatives, where we use
Lemma [25](#Thmtheorem25 "Lemma 25 (Function 𝜇⁢𝜔). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"),
Lemma [26](#Thmtheorem26 "Lemma 26 (Function 𝜈⁢𝜏). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"),
Lemma [27](#Thmtheorem27 "Lemma 27 (Function {𝜇⁢𝜔+𝜈⁢𝜏}/√2⁢√𝜈⁢𝜏). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"),
Lemma [28](#Thmtheorem28 "Lemma 28 (Function {𝜇⁢𝜔+2⁢𝜈⁢𝜏}/√2⁢√𝜈⁢𝜏). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"),
Lemma [29](#Thmtheorem29 "Lemma 29 (Function 𝜇⁢𝜔/√2⁢√𝜈⁢𝜏). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"),
Lemma [30](#Thmtheorem30 "Lemma 30 (Function (𝜇⁢𝜔/√2⁢√𝜈⁢𝜏)²). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"),
Lemma [21](#Thmtheorem21 "Lemma 21 (Basic functions). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), and
Lemma [23](#Thmtheorem23 "Lemma 23 (Function 𝑒^𝑥²⁢erfc(𝑥)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") without citing. Furthermore, we use the bounds on the simple
expressions t11subscript𝑡11t\_{11},t22subscript𝑡22t\_{22}, …, and T4subscript𝑇4T\_{4} as defined the aforementioned lemmata:

* •

  ∂𝒥11∂μsubscript𝒥11𝜇\frac{\partial{\mathcal{J}}\_{11}}{\partial\mu}

  We use Lemma [31](#Thmtheorem31 "Lemma 31 (Function √{2/𝜋}⁢(𝛼-1)/√𝜈⁢𝜏). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") and
  consider the expression
  α​e(μ​ω+ν​τ)22​ν​τ​erfc⁡(μ​ω+ν​τ2​ν​τ)−2π​(α−1)ν​τ𝛼superscript𝑒superscript𝜇𝜔𝜈𝜏22𝜈𝜏erfc𝜇𝜔𝜈𝜏2𝜈𝜏2𝜋𝛼1𝜈𝜏\alpha e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-\frac{\sqrt{\frac{2}{\pi}}(\alpha-1)}{\sqrt{\nu\tau}}
  in brackets.
  An upper bound on the maximum of is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | α01​et12​erfc⁡(t1)−2π​(α01−1)T22= 0.591017.subscript𝛼01superscript𝑒superscriptsubscript𝑡12erfcsubscript𝑡12𝜋subscript𝛼011subscript𝑇220.591017\displaystyle\alpha\_{\rm 01}e^{t\_{1}^{2}}\operatorname{erfc}(t\_{1})-\frac{\sqrt{\frac{2}{\pi}}(\alpha\_{\rm 01}-1)}{\sqrt{T\_{22}}}\ =\ 0.591017\ . |  | (177) |

  A lower bound on the minimum is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | α01​eT12​erfc⁡(T1)−2π​(α01−1)t22= 0.056318.subscript𝛼01superscript𝑒superscriptsubscript𝑇12erfcsubscript𝑇12𝜋subscript𝛼011subscript𝑡220.056318\displaystyle\alpha\_{\rm 01}e^{T\_{1}^{2}}\operatorname{erfc}(T\_{1})-\frac{\sqrt{\frac{2}{\pi}}(\alpha\_{\rm 01}-1)}{\sqrt{t\_{22}}}\ =\ 0.056318\ . |  | (178) |

  Thus, an upper bound on the maximal absolute value is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 12​λ01​ωmax2​et4​(α01​et12​erfc⁡(t1)−2π​(α01−1)T22)= 0.0031049101995398316.12subscript𝜆01superscriptsubscript𝜔max2superscript𝑒subscript𝑡4subscript𝛼01superscript𝑒superscriptsubscript𝑡12erfcsubscript𝑡12𝜋subscript𝛼011subscript𝑇220.0031049101995398316\displaystyle\frac{1}{2}\lambda\_{\rm 01}\omega\_{\rm max}^{2}e^{t\_{4}}\left(\alpha\_{\rm 01}e^{t\_{1}^{2}}\operatorname{erfc}(t\_{1})-\frac{\sqrt{\frac{2}{\pi}}(\alpha\_{\rm 01}-1)}{\sqrt{T\_{22}}}\right)\ =\ 0.0031049101995398316\ . |  | (179) |
* •

  ∂𝒥11∂ωsubscript𝒥11𝜔\frac{\partial{\mathcal{J}}\_{11}}{\partial\omega}

  We use Lemma [31](#Thmtheorem31 "Lemma 31 (Function √{2/𝜋}⁢(𝛼-1)/√𝜈⁢𝜏). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") and
  consider the expression
  2π​(α−1)​μ​ων​τ−α​(μ​ω+1)​e(μ​ω+ν​τ)22​ν​τ​erfc⁡(μ​ω+ν​τ2​ν​τ)2𝜋𝛼1𝜇𝜔𝜈𝜏𝛼𝜇𝜔1superscript𝑒superscript𝜇𝜔𝜈𝜏22𝜈𝜏erfc𝜇𝜔𝜈𝜏2𝜈𝜏\frac{\sqrt{\frac{2}{\pi}}(\alpha-1)\mu\omega}{\sqrt{\nu\tau}}-\alpha(\mu\omega+1)e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)
  in brackets.

  An upper bound on the maximum is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 2π​(α01−1)​T11t22−α01​(t11+1)​eT12​erfc⁡(T1)=−0.713808.2𝜋subscript𝛼011subscript𝑇11subscript𝑡22subscript𝛼01subscript𝑡111superscript𝑒superscriptsubscript𝑇12erfcsubscript𝑇10.713808\displaystyle\frac{\sqrt{\frac{2}{\pi}}(\alpha\_{\rm 01}-1)T\_{11}}{\sqrt{t\_{22}}}-\alpha\_{\rm 01}(t\_{11}+1)e^{T\_{1}^{2}}\operatorname{erfc}(T\_{1})\ =\ -0.713808\ . |  | (180) |

  A lower bound on the minimum is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 2π​(α01−1)​t11t22−α01​(T11+1)​et12​erfc⁡(t1)=−0.99987.2𝜋subscript𝛼011subscript𝑡11subscript𝑡22subscript𝛼01subscript𝑇111superscript𝑒superscriptsubscript𝑡12erfcsubscript𝑡10.99987\displaystyle\frac{\sqrt{\frac{2}{\pi}}(\alpha\_{\rm 01}-1)t\_{11}}{\sqrt{t\_{22}}}-\alpha\_{\rm 01}(T\_{11}+1)e^{t\_{1}^{2}}\operatorname{erfc}(t\_{1})\ =\ -0.99987\ . |  | (181) |

  This term is subtracted, and 2−erfc⁡(x)>02erfc𝑥02-\operatorname{erfc}(x)>0, therefore we have
  to use the minimum and the maximum for the argument of erfcerfc\operatorname{erfc}.

  Thus, an upper bound on the maximal absolute value is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 12​λ01​(−et4​(2π​(α01−1)​t11t22−α01​(T11+1)​et12​erfc⁡(t1))−erfc⁡(T3)+2)=12subscript𝜆01superscript𝑒subscript𝑡42𝜋subscript𝛼011subscript𝑡11subscript𝑡22subscript𝛼01subscript𝑇111superscript𝑒superscriptsubscript𝑡12erfcsubscript𝑡1erfcsubscript𝑇32absent\displaystyle\frac{1}{2}\lambda\_{\rm 01}\left(-e^{t\_{4}}\left(\frac{\sqrt{\frac{2}{\pi}}(\alpha\_{\rm 01}-1)t\_{11}}{\sqrt{t\_{22}}}-\alpha\_{\rm 01}(T\_{11}+1)e^{t\_{1}^{2}}\operatorname{erfc}(t\_{1})\right)\ -\operatorname{erfc}(T\_{3})+2\right)\ =\ |  | (182) |
  |  |  |  |
  | --- | --- | --- |
  |  | 1.055872374194189.1.055872374194189\displaystyle 1.055872374194189\ . |  |
* •

  ∂𝒥11∂νsubscript𝒥11𝜈\frac{\partial{\mathcal{J}}\_{11}}{\partial\nu}

  We consider the term in brackets

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | α​e(μ​ω+ν​τ)22​ν​τ​erfc⁡(μ​ω+ν​τ2​ν​τ)+2π​((α−1)​μ​ω(ν​τ)3/2−αν​τ).𝛼superscript𝑒superscript𝜇𝜔𝜈𝜏22𝜈𝜏erfc𝜇𝜔𝜈𝜏2𝜈𝜏2𝜋𝛼1𝜇𝜔superscript𝜈𝜏32𝛼𝜈𝜏\displaystyle\alpha e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+\sqrt{\frac{2}{\pi}}\left(\frac{(\alpha-1)\mu\omega}{(\nu\tau)^{3/2}}-\frac{\alpha}{\sqrt{\nu\tau}}\right)\ . |  | (183) |

  We apply Lemma [33](#Thmtheorem33 "Lemma 33 (Function √{2/𝜋}⁢({(𝛼-1)⁢𝜇⁢𝜔/(𝜈⁢𝜏)^{3/2}}-{𝛼/√𝜈⁢𝜏})). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") for the first sub-term.
  An upper bound on the maximum is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | α01​et12​erfc⁡(t1)+2π​((α01−1)​T11T223/2−α01T22)= 0.0104167.subscript𝛼01superscript𝑒superscriptsubscript𝑡12erfcsubscript𝑡12𝜋subscript𝛼011subscript𝑇11superscriptsubscript𝑇2232subscript𝛼01subscript𝑇220.0104167\displaystyle\alpha\_{\rm 01}e^{t\_{1}^{2}}\operatorname{erfc}(t\_{1})+\sqrt{\frac{2}{\pi}}\left(\frac{(\alpha\_{\rm 01}-1)T\_{11}}{T\_{22}^{3/2}}-\frac{\alpha\_{\rm 01}}{\sqrt{T\_{22}}}\right)\ =\ 0.0104167\ . |  | (184) |

  A lower bound on the minimum is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | α01​eT12​erfc⁡(T1)+2π​((α01−1)​t11t223/2−α01t22)=−0.95153.subscript𝛼01superscript𝑒superscriptsubscript𝑇12erfcsubscript𝑇12𝜋subscript𝛼011subscript𝑡11superscriptsubscript𝑡2232subscript𝛼01subscript𝑡220.95153\displaystyle\alpha\_{\rm 01}e^{T\_{1}^{2}}\operatorname{erfc}(T\_{1})+\sqrt{\frac{2}{\pi}}\left(\frac{(\alpha\_{\rm 01}-1)t\_{11}}{t\_{22}^{3/2}}-\frac{\alpha\_{\rm 01}}{\sqrt{t\_{22}}}\right)\ =\ -0.95153\ . |  | (185) |

  Thus, an upper bound on the maximal absolute value is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | −14​λ01​τmax​ωmax​et4​(α01​eT12​erfc⁡(T1)+2π​((α01−1)​t11t223/2−α01t22))=14subscript𝜆01subscript𝜏maxsubscript𝜔maxsuperscript𝑒subscript𝑡4subscript𝛼01superscript𝑒superscriptsubscript𝑇12erfcsubscript𝑇12𝜋subscript𝛼011subscript𝑡11superscriptsubscript𝑡2232subscript𝛼01subscript𝑡22absent\displaystyle-\frac{1}{4}\lambda\_{\rm 01}\tau\_{\rm max}\omega\_{\rm max}e^{t\_{4}}\left(\alpha\_{\rm 01}e^{T\_{1}^{2}}\operatorname{erfc}(T\_{1})+\sqrt{\frac{2}{\pi}}\left(\frac{(\alpha\_{\rm 01}-1)t\_{11}}{t\_{22}^{3/2}}-\frac{\alpha\_{\rm 01}}{\sqrt{t\_{22}}}\right)\right)\ = |  | (186) |
  |  |  |  |
  | --- | --- | --- |
  |  | 0.031242911235461816.0.031242911235461816\displaystyle 0.031242911235461816\ . |  |
* •

  ∂𝒥11∂τsubscript𝒥11𝜏\frac{\partial{\mathcal{J}}\_{11}}{\partial\tau}

  We use the results of item ∂𝒥11∂νsubscript𝒥11𝜈\frac{\partial{\mathcal{J}}\_{11}}{\partial\nu}
  were the brackets are only differently scaled.
  Thus, an upper bound on the maximal absolute value is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | −14​λ01​νmax​ωmax​et4​(α01​eT12​erfc⁡(T1)+2π​((α01−1)​t11t223/2−α01t22))=14subscript𝜆01subscript𝜈maxsubscript𝜔maxsuperscript𝑒subscript𝑡4subscript𝛼01superscript𝑒superscriptsubscript𝑇12erfcsubscript𝑇12𝜋subscript𝛼011subscript𝑡11superscriptsubscript𝑡2232subscript𝛼01subscript𝑡22absent\displaystyle-\frac{1}{4}\lambda\_{\rm 01}\nu\_{\rm max}\omega\_{\rm max}e^{t\_{4}}\left(\alpha\_{\rm 01}e^{T\_{1}^{2}}\operatorname{erfc}(T\_{1})+\sqrt{\frac{2}{\pi}}\left(\frac{(\alpha\_{\rm 01}-1)t\_{11}}{t\_{22}^{3/2}}-\frac{\alpha\_{\rm 01}}{\sqrt{t\_{22}}}\right)\right)\ = |  | (187) |
  |  |  |  |
  | --- | --- | --- |
  |  | 0.03749149348255419.0.03749149348255419\displaystyle 0.03749149348255419\ . |  |
* •

  ∂𝒥12∂μsubscript𝒥12𝜇\frac{\partial{\mathcal{J}}\_{12}}{\partial\mu}

  Since ∂𝒥12∂μ=∂𝒥11∂νsubscript𝒥12𝜇subscript𝒥11𝜈\frac{\partial{\mathcal{J}}\_{12}}{\partial\mu}=\frac{\partial{\mathcal{J}}\_{11}}{\partial\nu},
  an upper bound on the maximal absolute value is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | −14​λ01​τmax​ωmax​et4​(α01​eT12​erfc⁡(T1)+2π​((α01−1)​t11t223/2−α01t22))=14subscript𝜆01subscript𝜏maxsubscript𝜔maxsuperscript𝑒subscript𝑡4subscript𝛼01superscript𝑒superscriptsubscript𝑇12erfcsubscript𝑇12𝜋subscript𝛼011subscript𝑡11superscriptsubscript𝑡2232subscript𝛼01subscript𝑡22absent\displaystyle-\frac{1}{4}\lambda\_{\rm 01}\tau\_{\rm max}\omega\_{\rm max}e^{t\_{4}}\left(\alpha\_{\rm 01}e^{T\_{1}^{2}}\operatorname{erfc}(T\_{1})+\sqrt{\frac{2}{\pi}}\left(\frac{(\alpha\_{\rm 01}-1)t\_{11}}{t\_{22}^{3/2}}-\frac{\alpha\_{\rm 01}}{\sqrt{t\_{22}}}\right)\right)\ = |  | (188) |
  |  |  |  |
  | --- | --- | --- |
  |  | 0.031242911235461816.0.031242911235461816\displaystyle 0.031242911235461816\ . |  |
* •

  ∂𝒥12∂ωsubscript𝒥12𝜔\frac{\partial{\mathcal{J}}\_{12}}{\partial\omega}

  We use the results of item ∂𝒥11∂νsubscript𝒥11𝜈\frac{\partial{\mathcal{J}}\_{11}}{\partial\nu}
  were the brackets are only differently scaled.
  Thus, an upper bound on the maximal absolute value is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | −14​λ01​μmax​τmax​et4​(α01​eT12​erfc⁡(T1)+2π​((α01−1)​t11t223/2−α01t22))=14subscript𝜆01subscript𝜇maxsubscript𝜏maxsuperscript𝑒subscript𝑡4subscript𝛼01superscript𝑒superscriptsubscript𝑇12erfcsubscript𝑇12𝜋subscript𝛼011subscript𝑡11superscriptsubscript𝑡2232subscript𝛼01subscript𝑡22absent\displaystyle-\frac{1}{4}\lambda\_{\rm 01}\mu\_{\rm max}\tau\_{\rm max}e^{t\_{4}}\left(\alpha\_{\rm 01}e^{T\_{1}^{2}}\operatorname{erfc}(T\_{1})+\sqrt{\frac{2}{\pi}}\left(\frac{(\alpha\_{\rm 01}-1)t\_{11}}{t\_{22}^{3/2}}-\frac{\alpha\_{\rm 01}}{\sqrt{t\_{22}}}\right)\right)\ = |  | (189) |
  |  |  |  |
  | --- | --- | --- |
  |  | 0.031242911235461816.0.031242911235461816\displaystyle 0.031242911235461816\ . |  |
* •

  ∂𝒥12∂νsubscript𝒥12𝜈\frac{\partial{\mathcal{J}}\_{12}}{\partial\nu}

  For the second term in brackets, we see that
  α01​τmin2​eT12​erfc⁡(T1)=0.465793subscript𝛼01superscriptsubscript𝜏min2superscript𝑒superscriptsubscript𝑇12erfcsubscript𝑇10.465793\alpha\_{\rm 01}\tau\_{\rm min}^{2}e^{T\_{1}^{2}}\operatorname{erfc}(T\_{1})=0.465793 and α01​τmax2​et12​erfc⁡(t1)=1.53644subscript𝛼01superscriptsubscript𝜏max2superscript𝑒superscriptsubscript𝑡12erfcsubscript𝑡11.53644\alpha\_{\rm 01}\tau\_{\rm max}^{2}e^{t\_{1}^{2}}\operatorname{erfc}(t\_{1})=1.53644.

  We now check different values for

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 2π​((−1)​(α−1)​μ2​ω2ν5/2​τ+τ​(α+α​μ​ω−1)ν3/2−α​τ3/2ν),2𝜋1𝛼1superscript𝜇2superscript𝜔2superscript𝜈52𝜏𝜏𝛼𝛼𝜇𝜔1superscript𝜈32𝛼superscript𝜏32𝜈\displaystyle\sqrt{\frac{2}{\pi}}\left(\frac{(-1)(\alpha-1)\mu^{2}\omega^{2}}{\nu^{5/2}\sqrt{\tau}}+\frac{\sqrt{\tau}(\alpha+\alpha\mu\omega-1)}{\nu^{3/2}}-\frac{\alpha\tau^{3/2}}{\sqrt{\nu}}\right)\ , |  | (190) |

  where we maximize or minimize all single terms.

  A lower bound on the minimum of this expression is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 2π​((−1)​(α01−1)​μmax2​ωmax2νmin5/2​τmin+τmin​(α01+α01​t11−1)νmax3/2−α01​τmax3/2νmin)=2𝜋1subscript𝛼011superscriptsubscript𝜇max2superscriptsubscript𝜔max2superscriptsubscript𝜈min52subscript𝜏minsubscript𝜏minsubscript𝛼01subscript𝛼01subscript𝑡111superscriptsubscript𝜈max32subscript𝛼01superscriptsubscript𝜏max32subscript𝜈minabsent\displaystyle\sqrt{\frac{2}{\pi}}\left(\frac{(-1)(\alpha\_{\rm 01}-1)\mu\_{\rm max}^{2}\omega\_{\rm max}^{2}}{\nu\_{\rm min}^{5/2}\sqrt{\tau\_{\rm min}}}+\frac{\sqrt{\tau\_{\rm min}}(\alpha\_{\rm 01}+\alpha\_{\rm 01}t\_{11}-1)}{\nu\_{\rm max}^{3/2}}-\frac{\alpha\_{\rm 01}\tau\_{\rm max}^{3/2}}{\sqrt{\nu\_{\rm min}}}\right)\ = |  | (191) |
  |  |  |  |
  | --- | --- | --- |
  |  | −1.83112.1.83112\displaystyle-1.83112\ . |  |

  An upper bound on the maximum of this expression is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 2π​((−1)​(α01−1)​μmin2​ωmin2νmax5/2​τmax+τmax​(α01+α01​T11−1)νmin3/2−α01​τmin3/2νmax)=2𝜋1subscript𝛼011superscriptsubscript𝜇min2superscriptsubscript𝜔min2superscriptsubscript𝜈max52subscript𝜏maxsubscript𝜏maxsubscript𝛼01subscript𝛼01subscript𝑇111superscriptsubscript𝜈min32subscript𝛼01superscriptsubscript𝜏min32subscript𝜈maxabsent\displaystyle\sqrt{\frac{2}{\pi}}\left(\frac{(-1)(\alpha\_{\rm 01}-1)\mu\_{\rm min}^{2}\omega\_{\rm min}^{2}}{\nu\_{\rm max}^{5/2}\sqrt{\tau\_{\rm max}}}+\frac{\sqrt{\tau\_{\rm max}}(\alpha\_{\rm 01}+\alpha\_{\rm 01}T\_{11}-1)}{\nu\_{\rm min}^{3/2}}-\frac{\alpha\_{\rm 01}\tau\_{\rm min}^{3/2}}{\sqrt{\nu\_{\rm max}}}\right)\ = |  | (192) |
  |  |  |  |
  | --- | --- | --- |
  |  | 0.0802158.0.0802158\displaystyle 0.0802158\ . |  |

  An upper bound on the maximum is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 18λ01et4(2π((−1)​(α01−1)​μmin2​ωmin2νmax5/2​τmax−α01​τmin3/2νmax+\displaystyle\frac{1}{8}\lambda\_{\rm 01}e^{t\_{4}}\left(\sqrt{\frac{2}{\pi}}\left(\frac{(-1)(\alpha\_{\rm 01}-1)\mu\_{\rm min}^{2}\omega\_{\rm min}^{2}}{\nu\_{\rm max}^{5/2}\sqrt{\tau\_{\rm max}}}-\frac{\alpha\_{\rm 01}\tau\_{\rm min}^{3/2}}{\sqrt{\nu\_{\rm max}}}\ +\right.\right. |  | (193) |
  |  |  |  |
  | --- | --- | --- |
  |  | τmax​(α01+α01​T11−1)νmin3/2)+α01τmax2et12erfc(t1))= 0.212328.\displaystyle\left.\left.\frac{\sqrt{\tau\_{\rm max}}(\alpha\_{\rm 01}+\alpha\_{\rm 01}T\_{11}-1)}{\nu\_{\rm min}^{3/2}}\right)+\alpha\_{\rm 01}\tau\_{\rm max}^{2}e^{t\_{1}^{2}}\operatorname{erfc}(t\_{1})\right)\ =\ 0.212328\ . |  |

  A lower bound on the minimum is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 18λ01et4(α01τmin2eT12erfc(T1)+\displaystyle\frac{1}{8}\lambda\_{\rm 01}e^{t\_{4}}\left(\alpha\_{\rm 01}\tau\_{\rm min}^{2}e^{T\_{1}^{2}}\operatorname{erfc}(T\_{1})\ +\right. |  | (194) |
  |  |  |  |
  | --- | --- | --- |
  |  | 2π((−1)​(α01−1)​μmax2​ωmax2νmin5/2​τmin+τmin​(α01+α01​t11−1)νmax3/2−α01​τmax3/2νmin))=\displaystyle\left.\sqrt{\frac{2}{\pi}}\left(\frac{(-1)(\alpha\_{\rm 01}-1)\mu\_{\rm max}^{2}\omega\_{\rm max}^{2}}{\nu\_{\rm min}^{5/2}\sqrt{\tau\_{\rm min}}}+\frac{\sqrt{\tau\_{\rm min}}(\alpha\_{\rm 01}+\alpha\_{\rm 01}t\_{11}-1)}{\nu\_{\rm max}^{3/2}}-\frac{\alpha\_{\rm 01}\tau\_{\rm max}^{3/2}}{\sqrt{\nu\_{\rm min}}}\right)\right)\ = |  |
  |  |  |  |
  | --- | --- | --- |
  |  | −0.179318.0.179318\displaystyle-0.179318\ . |  |

  Thus, an upper bound on the maximal absolute value is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 18λ01et4(2π((−1)​(α01−1)​μmin2​ωmin2νmax5/2​τmax−α01​τmin3/2νmax+\displaystyle\frac{1}{8}\lambda\_{\rm 01}e^{t\_{4}}\left(\sqrt{\frac{2}{\pi}}\left(\frac{(-1)(\alpha\_{\rm 01}-1)\mu\_{\rm min}^{2}\omega\_{\rm min}^{2}}{\nu\_{\rm max}^{5/2}\sqrt{\tau\_{\rm max}}}-\frac{\alpha\_{\rm 01}\tau\_{\rm min}^{3/2}}{\sqrt{\nu\_{\rm max}}}\ +\right.\right. |  | (195) |
  |  |  |  |
  | --- | --- | --- |
  |  | τmax​(α01+α01​T11−1)νmin3/2)+α01τmax2et12erfc(t1))= 0.21232788238624354.\displaystyle\left.\left.\frac{\sqrt{\tau\_{\rm max}}(\alpha\_{\rm 01}+\alpha\_{\rm 01}T\_{11}-1)}{\nu\_{\rm min}^{3/2}}\right)+\alpha\_{\rm 01}\tau\_{\rm max}^{2}e^{t\_{1}^{2}}\operatorname{erfc}(t\_{1})\right)\ =\ 0.21232788238624354\ . |  |
* •

  ∂𝒥12∂τsubscript𝒥12𝜏\frac{\partial{\mathcal{J}}\_{12}}{\partial\tau}

  We use Lemma [34](#Thmtheorem34 "Lemma 34 (Function √{2/𝜋}⁢({(-1)⁢(𝛼-1)⁢𝜇²⁢𝜔²/(𝜈⁢𝜏)^{3/2}}+{{-𝛼+𝛼⁢𝜇⁢𝜔+1}/√𝜈⁢𝜏}-𝛼⁢√𝜈⁢𝜏)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") to obtain
  an upper bound on the maximum of the expression of the lemma:

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 2π​(0.12⋅0.12​(−1)​(α01−1)(0.8⋅0.8)3/2−0.8⋅0.8​α01+(0.1⋅0.1)​α01−α01+10.8⋅0.8)=−1.72296.2𝜋⋅superscript0.12superscript0.121subscript𝛼011superscript⋅0.80.832⋅0.80.8subscript𝛼01⋅0.10.1subscript𝛼01subscript𝛼011⋅0.80.81.72296\displaystyle\sqrt{\frac{2}{\pi}}\left(\frac{0.1^{2}\cdot 0.1^{2}(-1)(\alpha\_{\rm 01}-1)}{(0.8\cdot 0.8)^{3/2}}-\sqrt{0.8\cdot 0.8}\alpha\_{\rm 01}+\frac{(0.1\cdot 0.1)\alpha\_{\rm 01}-\alpha\_{\rm 01}+1}{\sqrt{0.8\cdot 0.8}}\right)\ =\ -1.72296\ . |  | (196) |

  We use Lemma [34](#Thmtheorem34 "Lemma 34 (Function √{2/𝜋}⁢({(-1)⁢(𝛼-1)⁢𝜇²⁢𝜔²/(𝜈⁢𝜏)^{3/2}}+{{-𝛼+𝛼⁢𝜇⁢𝜔+1}/√𝜈⁢𝜏}-𝛼⁢√𝜈⁢𝜏)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") to obtain
  an lower bound on the minimum of the expression of the lemma:

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 2π​(0.12⋅0.12​(−1)​(α01−1)(1.5⋅1.25)3/2−1.5⋅1.25​α01+(−0.1⋅0.1)​α01−α01+11.5⋅1.25)=−2.2302.2𝜋⋅superscript0.12superscript0.121subscript𝛼011superscript⋅1.51.2532⋅1.51.25subscript𝛼01⋅0.10.1subscript𝛼01subscript𝛼011⋅1.51.252.2302\displaystyle\sqrt{\frac{2}{\pi}}\left(\frac{0.1^{2}\cdot 0.1^{2}(-1)(\alpha\_{\rm 01}-1)}{(1.5\cdot 1.25)^{3/2}}-\sqrt{1.5\cdot 1.25}\alpha\_{\rm 01}+\frac{(-0.1\cdot 0.1)\alpha\_{\rm 01}-\alpha\_{\rm 01}+1}{\sqrt{1.5\cdot 1.25}}\right)\ =\ -2.2302\ . |  | (197) |

  Next we apply Lemma [37](#Thmtheorem37 "Lemma 37 (Function 𝜈⁢𝜏⁢𝑒^{(𝜇⁢𝜔+𝜈⁢𝜏)²/2⁢𝜈⁢𝜏}⁢erfc({𝜇⁢𝜔+𝜈⁢𝜏}/√2⁢√𝜈⁢𝜏)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") for the expression ν​τ​e(μ​ω+ν​τ)22​ν​τ​erfc⁡(μ​ω+ν​τ2​ν​τ)𝜈𝜏superscript𝑒superscript𝜇𝜔𝜈𝜏22𝜈𝜏erfc𝜇𝜔𝜈𝜏2𝜈𝜏\nu\tau e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right).
  We use Lemma [37](#Thmtheorem37 "Lemma 37 (Function 𝜈⁢𝜏⁢𝑒^{(𝜇⁢𝜔+𝜈⁢𝜏)²/2⁢𝜈⁢𝜏}⁢erfc({𝜇⁢𝜔+𝜈⁢𝜏}/√2⁢√𝜈⁢𝜏)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") to obtain
  an upper bound on the maximum of this expression:

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 1.5⋅1.25​e(1.5⋅1.25−0.1⋅0.1)22⋅1.5⋅1.25​α01​erfc⁡(1.5⋅1.25−0.1⋅0.12​1.5⋅1.25)= 1.37381.⋅1.51.25superscript𝑒superscript⋅1.51.25⋅0.10.12⋅21.51.25subscript𝛼01erfc⋅1.51.25⋅0.10.12⋅1.51.251.37381\displaystyle 1.5\cdot 1.25e^{\frac{(1.5\cdot 1.25-0.1\cdot 0.1)^{2}}{2\cdot 1.5\cdot 1.25}}\alpha\_{\rm 01}\operatorname{erfc}\left(\frac{1.5\cdot 1.25-0.1\cdot 0.1}{\sqrt{2}\sqrt{1.5\cdot 1.25}}\right)\ =\ 1.37381\ . |  | (198) |

  We use Lemma [37](#Thmtheorem37 "Lemma 37 (Function 𝜈⁢𝜏⁢𝑒^{(𝜇⁢𝜔+𝜈⁢𝜏)²/2⁢𝜈⁢𝜏}⁢erfc({𝜇⁢𝜔+𝜈⁢𝜏}/√2⁢√𝜈⁢𝜏)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") to obtain
  an lower bound on the minimum of this expression:

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 0.8⋅0.8​e(0.8⋅0.8+0.1⋅0.1)22⋅0.8⋅0.8​α01​erfc⁡(0.8⋅0.8+0.1⋅0.12​0.8⋅0.8)= 0.620462.⋅0.80.8superscript𝑒superscript⋅0.80.8⋅0.10.12⋅20.80.8subscript𝛼01erfc⋅0.80.8⋅0.10.12⋅0.80.80.620462\displaystyle 0.8\cdot 0.8e^{\frac{(0.8\cdot 0.8+0.1\cdot 0.1)^{2}}{2\cdot 0.8\cdot 0.8}}\alpha\_{\rm 01}\operatorname{erfc}\left(\frac{0.8\cdot 0.8+0.1\cdot 0.1}{\sqrt{2}\sqrt{0.8\cdot 0.8}}\right)\ =\ 0.620462\ . |  | (199) |

  Next we apply Lemma [23](#Thmtheorem23 "Lemma 23 (Function 𝑒^𝑥²⁢erfc(𝑥)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") for 2​α​e(μ​ω+ν​τ)22​ν​τ​erfc⁡(μ​ω+ν​τ2​ν​τ)2𝛼superscript𝑒superscript𝜇𝜔𝜈𝜏22𝜈𝜏erfc𝜇𝜔𝜈𝜏2𝜈𝜏2\alpha e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right).
  An upper bound on this expression is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 2​e(0.8⋅0.8−0.1⋅0.1)220.8⋅0.8​α01​erfc⁡(0.8⋅0.8−0.1⋅0.12​0.8⋅0.8)= 1.96664.2superscript𝑒superscript⋅0.80.8⋅0.10.12⋅20.80.8subscript𝛼01erfc⋅0.80.8⋅0.10.12⋅0.80.81.96664\displaystyle 2e^{\frac{(0.8\cdot 0.8-0.1\cdot 0.1)^{2}}{20.8\cdot 0.8}}\alpha\_{\rm 01}\operatorname{erfc}\left(\frac{0.8\cdot 0.8-0.1\cdot 0.1}{\sqrt{2}\sqrt{0.8\cdot 0.8}}\right)\ =\ 1.96664\ . |  | (200) |

  A lower bound on this expression is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 2​e(1.5⋅1.25+0.1⋅0.1)22⋅1.5⋅1.25​α01​erfc⁡(1.5⋅1.25+0.1⋅0.12​1.5⋅1.25)= 1.4556.2superscript𝑒superscript⋅1.51.25⋅0.10.12⋅21.51.25subscript𝛼01erfc⋅1.51.25⋅0.10.12⋅1.51.251.4556\displaystyle 2e^{\frac{(1.5\cdot 1.25+0.1\cdot 0.1)^{2}}{2\cdot 1.5\cdot 1.25}}\alpha\_{\rm 01}\operatorname{erfc}\left(\frac{1.5\cdot 1.25+0.1\cdot 0.1}{\sqrt{2}\sqrt{1.5\cdot 1.25}}\right)\ =\ 1.4556\ . |  | (201) |

  The sum of the minimal values of the terms is
  −2.23019+0.62046+1.45560=−0.1541332.230190.620461.455600.154133-2.23019+0.62046+1.45560=-0.154133.

  The sum of the maximal values of the terms is
  −1.72295+1.37380+1.96664=1.617491.722951.373801.966641.61749-1.72295+1.37380+1.96664=1.61749.

  Thus, an upper bound on the maximal absolute value is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 18λ01et4(α01T22e(t11+T22)22​T22erfc(t11+T222​T22)+\displaystyle\frac{1}{8}\lambda\_{\rm 01}e^{t\_{4}}\left(\alpha\_{\rm 01}T\_{22}e^{\frac{(t\_{11}+T\_{22})^{2}}{2T\_{22}}}\operatorname{erfc}\left(\frac{t\_{11}+T\_{22}}{\sqrt{2}\sqrt{T\_{22}}}\right)\ +\right. |  | (202) |
  |  |  |  |
  | --- | --- | --- |
  |  | 2α01et12erfc(t1)+2π(−(α01−1)​T112t223/2+−α01+α01​T11+1t22−\displaystyle\left.2\alpha\_{\rm 01}e^{t\_{1}^{2}}\operatorname{erfc}(t\_{1})+\sqrt{\frac{2}{\pi}}\left(-\frac{(\alpha\_{\rm 01}-1)T\_{11}^{2}}{t\_{22}^{3/2}}+\frac{-\alpha\_{\rm 01}+\alpha\_{\rm 01}T\_{11}+1}{\sqrt{t\_{22}}}\ -\right.\right. |  |
  |  |  |  |
  | --- | --- | --- |
  |  | α01t22))= 0.2124377655377270.\displaystyle\left.\left.\alpha\_{\rm 01}\sqrt{t\_{22}}\right)\right)\ =\ 0.2124377655377270\ . |  |
* •

  ∂𝒥21∂μsubscript𝒥21𝜇\frac{\partial{\mathcal{J}}\_{21}}{\partial\mu}

  An upper bound on the maximum is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | λ012​ωmax2​(α012​eT12​(−e−T4)​erfc⁡(T1)+2​α012​et22​et4​erfc⁡(t2)−erfc⁡(T3)+2)=superscriptsubscript𝜆012superscriptsubscript𝜔max2superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑇12superscript𝑒subscript𝑇4erfcsubscript𝑇12superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑡22superscript𝑒subscript𝑡4erfcsubscript𝑡2erfcsubscript𝑇32absent\displaystyle\lambda\_{\rm 01}^{2}\omega\_{\rm max}^{2}\left(\alpha\_{\rm 01}^{2}e^{T\_{1}^{2}}\left(-e^{-T\_{4}}\right)\operatorname{erfc}(T\_{1})+2\alpha\_{\rm 01}^{2}e^{t\_{2}^{2}}e^{t\_{4}}\operatorname{erfc}(t\_{2})\ -\operatorname{erfc}(T\_{3})+2\right)\ =\ |  | (203) |
  |  |  |  |
  | --- | --- | --- |
  |  | 0.0222044.0.0222044\displaystyle 0.0222044\ . |  |

  A upper bound on the absolute minimum is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | λ012​ωmax2​(α012​et12​(−e−t4)​erfc⁡(t1)+2​α012​eT22​eT4​erfc⁡(T2)−erfc⁡(t3)+2)=superscriptsubscript𝜆012superscriptsubscript𝜔max2superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑡12superscript𝑒subscript𝑡4erfcsubscript𝑡12superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑇22superscript𝑒subscript𝑇4erfcsubscript𝑇2erfcsubscript𝑡32absent\displaystyle\lambda\_{\rm 01}^{2}\omega\_{\rm max}^{2}\left(\alpha\_{\rm 01}^{2}e^{t\_{1}^{2}}\left(-e^{-t\_{4}}\right)\operatorname{erfc}(t\_{1})+2\alpha\_{\rm 01}^{2}e^{T\_{2}^{2}}e^{T\_{4}}\operatorname{erfc}(T\_{2})\ -\operatorname{erfc}(t\_{3})+2\right)\ =\ |  | (204) |
  |  |  |  |
  | --- | --- | --- |
  |  | 0.00894889.0.00894889\displaystyle 0.00894889\ . |  |

  Thus, an upper bound on the maximal absolute value is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | λ012​ωmax2​(α012​eT12​(−e−T4)​erfc⁡(T1)+2​α012​et22​et4​erfc⁡(t2)−erfc⁡(T3)+2)=superscriptsubscript𝜆012superscriptsubscript𝜔max2superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑇12superscript𝑒subscript𝑇4erfcsubscript𝑇12superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑡22superscript𝑒subscript𝑡4erfcsubscript𝑡2erfcsubscript𝑇32absent\displaystyle\lambda\_{\rm 01}^{2}\omega\_{\rm max}^{2}\left(\alpha\_{\rm 01}^{2}e^{T\_{1}^{2}}\left(-e^{-T\_{4}}\right)\operatorname{erfc}(T\_{1})+2\alpha\_{\rm 01}^{2}e^{t\_{2}^{2}}e^{t\_{4}}\operatorname{erfc}(t\_{2})\ -\operatorname{erfc}(T\_{3})+2\right)\ =\ |  | (205) |
  |  |  |  |
  | --- | --- | --- |
  |  | 0.02220441024325437.0.02220441024325437\displaystyle 0.02220441024325437\ . |  |
* •

  ∂𝒥21∂ωsubscript𝒥21𝜔\frac{\partial{\mathcal{J}}\_{21}}{\partial\omega}

  An upper bound on the maximum is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | λ012(α012(2T11+1)et22e−t4erfc(t2)+2T11(2−erfc(T3))+\displaystyle\lambda\_{\rm 01}^{2}\left(\alpha\_{\rm 01}^{2}(2T\_{11}+1)e^{t\_{2}^{2}}e^{-t\_{4}}\operatorname{erfc}(t\_{2})+2T\_{11}(2-\operatorname{erfc}(T\_{3}))\ +\right. |  | (206) |
  |  |  |  |
  | --- | --- | --- |
  |  | α012(t11+1)eT12(−e−T4)erfc(T1)+2πT22e−t4)= 1.14696.\displaystyle\left.\alpha\_{\rm 01}^{2}(t\_{11}+1)e^{T\_{1}^{2}}\left(-e^{-T\_{4}}\right)\operatorname{erfc}(T\_{1})+\sqrt{\frac{2}{\pi}}\sqrt{T\_{22}}e^{-t\_{4}}\right)\ =\ 1.14696\ . |  |

  A lower bound on the minimum is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | λ012(α012(T11+1)et12(−e−t4)erfc(t1)+\displaystyle\lambda\_{\rm 01}^{2}\left(\alpha\_{\rm 01}^{2}(T\_{11}+1)e^{t\_{1}^{2}}\left(-e^{-t\_{4}}\right)\operatorname{erfc}(t\_{1})\ +\right. |  | (207) |
  |  |  |  |
  | --- | --- | --- |
  |  | α012​(2​t11+1)​eT22​e−T4​erfc⁡(T2)+2​t11​(2−erfc⁡(T3))+superscriptsubscript𝛼0122subscript𝑡111superscript𝑒superscriptsubscript𝑇22superscript𝑒subscript𝑇4erfcsubscript𝑇2limit-from2subscript𝑡112erfcsubscript𝑇3\displaystyle\left.\alpha\_{\rm 01}^{2}(2t\_{11}+1)e^{T\_{2}^{2}}e^{-T\_{4}}\operatorname{erfc}(T\_{2})+2t\_{11}(2-\operatorname{erfc}(T\_{3}))+\right. |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 2πt22e−T4)=−0.359403.\displaystyle\left.\sqrt{\frac{2}{\pi}}\sqrt{t\_{22}}e^{-T\_{4}}\right)\ =\ -0.359403\ . |  |

  Thus, an upper bound on the maximal absolute value is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | λ012(α012(2T11+1)et22e−t4erfc(t2)+2T11(2−erfc(T3))+\displaystyle\lambda\_{\rm 01}^{2}\left(\alpha\_{\rm 01}^{2}(2T\_{11}+1)e^{t\_{2}^{2}}e^{-t\_{4}}\operatorname{erfc}(t\_{2})+2T\_{11}(2-\operatorname{erfc}(T\_{3}))\ +\right. |  | (208) |
  |  |  |  |
  | --- | --- | --- |
  |  | α012(t11+1)eT12(−e−T4)erfc(T1)+2πT22e−t4)= 1.146955401845684.\displaystyle\left.\alpha\_{\rm 01}^{2}(t\_{11}+1)e^{T\_{1}^{2}}\left(-e^{-T\_{4}}\right)\operatorname{erfc}(T\_{1})+\sqrt{\frac{2}{\pi}}\sqrt{T\_{22}}e^{-t\_{4}}\right)\ =\ 1.146955401845684\ . |  |
* •

  ∂𝒥21∂νsubscript𝒥21𝜈\frac{\partial{\mathcal{J}}\_{21}}{\partial\nu}

  An upper bound on the maximum is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 12​λ012​τmax​ωmax​e−t4​(α012​(−eT12)​erfc⁡(T1)+4​α012​et22​erfc⁡(t2)+2π​(−1)​(α012−1)T22)=12superscriptsubscript𝜆012subscript𝜏maxsubscript𝜔maxsuperscript𝑒subscript𝑡4superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑇12erfcsubscript𝑇14superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑡22erfcsubscript𝑡22𝜋1superscriptsubscript𝛼0121subscript𝑇22absent\displaystyle\frac{1}{2}\lambda\_{\rm 01}^{2}\tau\_{\rm max}\omega\_{\rm max}e^{-t\_{4}}\left(\alpha\_{\rm 01}^{2}\left(-e^{T\_{1}^{2}}\right)\operatorname{erfc}(T\_{1})+4\alpha\_{\rm 01}^{2}e^{t\_{2}^{2}}\operatorname{erfc}(t\_{2})\ +\frac{\sqrt{\frac{2}{\pi}}(-1)\left(\alpha\_{\rm 01}^{2}-1\right)}{\sqrt{T\_{22}}}\right)=\ |  | (209) |
  |  |  |  |
  | --- | --- | --- |
  |  | 0.149834.0.149834\displaystyle 0.149834\ . |  |

  A lower bound on the minimum is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 12​λ012​τmax​ωmax​e−t4​(α012​(−et12)​erfc⁡(t1)+4​α012​eT22​erfc⁡(T2)+2π​(−1)​(α012−1)t22)=12superscriptsubscript𝜆012subscript𝜏maxsubscript𝜔maxsuperscript𝑒subscript𝑡4superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑡12erfcsubscript𝑡14superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑇22erfcsubscript𝑇22𝜋1superscriptsubscript𝛼0121subscript𝑡22absent\displaystyle\frac{1}{2}\lambda\_{\rm 01}^{2}\tau\_{\rm max}\omega\_{\rm max}e^{-t\_{4}}\left(\alpha\_{\rm 01}^{2}\left(-e^{t\_{1}^{2}}\right)\operatorname{erfc}(t\_{1})+4\alpha\_{\rm 01}^{2}e^{T\_{2}^{2}}\operatorname{erfc}(T\_{2})\ +\frac{\sqrt{\frac{2}{\pi}}(-1)\left(\alpha\_{\rm 01}^{2}-1\right)}{\sqrt{t\_{22}}}\right)= |  | (210) |
  |  |  |  |
  | --- | --- | --- |
  |  | −0.0351035.0.0351035\displaystyle-0.0351035\ . |  |

  Thus, an upper bound on the maximal absolute value is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 12​λ012​τmax​ωmax​e−t4​(α012​(−eT12)​erfc⁡(T1)+4​α012​et22​erfc⁡(t2)+2π​(−1)​(α012−1)T22)=12superscriptsubscript𝜆012subscript𝜏maxsubscript𝜔maxsuperscript𝑒subscript𝑡4superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑇12erfcsubscript𝑇14superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑡22erfcsubscript𝑡22𝜋1superscriptsubscript𝛼0121subscript𝑇22absent\displaystyle\frac{1}{2}\lambda\_{\rm 01}^{2}\tau\_{\rm max}\omega\_{\rm max}e^{-t\_{4}}\left(\alpha\_{\rm 01}^{2}\left(-e^{T\_{1}^{2}}\right)\operatorname{erfc}(T\_{1})+4\alpha\_{\rm 01}^{2}e^{t\_{2}^{2}}\operatorname{erfc}(t\_{2})\ +\frac{\sqrt{\frac{2}{\pi}}(-1)\left(\alpha\_{\rm 01}^{2}-1\right)}{\sqrt{T\_{22}}}\right)\ =\ |  | (211) |
  |  |  |  |
  | --- | --- | --- |
  |  | 0.14983446469110305.0.14983446469110305\displaystyle 0.14983446469110305\ . |  |
* •

  ∂𝒥21∂τsubscript𝒥21𝜏\frac{\partial{\mathcal{J}}\_{21}}{\partial\tau}

  An upper bound on the maximum is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 12​λ012​νmax​ωmax​e−t4​(α012​(−eT12)​erfc⁡(T1)+4​α012​et22​erfc⁡(t2)+2π​(−1)​(α012−1)T22)=12superscriptsubscript𝜆012subscript𝜈maxsubscript𝜔maxsuperscript𝑒subscript𝑡4superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑇12erfcsubscript𝑇14superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑡22erfcsubscript𝑡22𝜋1superscriptsubscript𝛼0121subscript𝑇22absent\displaystyle\frac{1}{2}\lambda\_{\rm 01}^{2}\nu\_{\rm max}\omega\_{\rm max}e^{-t\_{4}}\left(\alpha\_{\rm 01}^{2}\left(-e^{T\_{1}^{2}}\right)\operatorname{erfc}(T\_{1})+4\alpha\_{\rm 01}^{2}e^{t\_{2}^{2}}\operatorname{erfc}(t\_{2})\ +\frac{\sqrt{\frac{2}{\pi}}(-1)\left(\alpha\_{\rm 01}^{2}-1\right)}{\sqrt{T\_{22}}}\right)\ =\ |  | (212) |
  |  |  |  |
  | --- | --- | --- |
  |  | 0.179801.0.179801\displaystyle 0.179801\ . |  |

  A lower bound on the minimum is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 12​λ012​νmax​ωmax​e−t4​(α012​(−et12)​erfc⁡(t1)+4​α012​eT22​erfc⁡(T2)+2π​(−1)​(α012−1)t22)=12superscriptsubscript𝜆012subscript𝜈maxsubscript𝜔maxsuperscript𝑒subscript𝑡4superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑡12erfcsubscript𝑡14superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑇22erfcsubscript𝑇22𝜋1superscriptsubscript𝛼0121subscript𝑡22absent\displaystyle\frac{1}{2}\lambda\_{\rm 01}^{2}\nu\_{\rm max}\omega\_{\rm max}e^{-t\_{4}}\left(\alpha\_{\rm 01}^{2}\left(-e^{t\_{1}^{2}}\right)\operatorname{erfc}(t\_{1})+4\alpha\_{\rm 01}^{2}e^{T\_{2}^{2}}\operatorname{erfc}(T\_{2})\ +\frac{\sqrt{\frac{2}{\pi}}(-1)\left(\alpha\_{\rm 01}^{2}-1\right)}{\sqrt{t\_{22}}}\right)\ =\ |  | (213) |
  |  |  |  |
  | --- | --- | --- |
  |  | −0.0421242.0.0421242\displaystyle-0.0421242\ . |  |

  Thus, an upper bound on the maximal absolute value is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 12​λ012​νmax​ωmax​e−t4​(α012​(−eT12)​erfc⁡(T1)+4​α012​et22​erfc⁡(t2)+2π​(−1)​(α012−1)T22)=12superscriptsubscript𝜆012subscript𝜈maxsubscript𝜔maxsuperscript𝑒subscript𝑡4superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑇12erfcsubscript𝑇14superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑡22erfcsubscript𝑡22𝜋1superscriptsubscript𝛼0121subscript𝑇22absent\displaystyle\frac{1}{2}\lambda\_{\rm 01}^{2}\nu\_{\rm max}\omega\_{\rm max}e^{-t\_{4}}\left(\alpha\_{\rm 01}^{2}\left(-e^{T\_{1}^{2}}\right)\operatorname{erfc}(T\_{1})+4\alpha\_{\rm 01}^{2}e^{t\_{2}^{2}}\operatorname{erfc}(t\_{2})\ +\frac{\sqrt{\frac{2}{\pi}}(-1)\left(\alpha\_{\rm 01}^{2}-1\right)}{\sqrt{T\_{22}}}\right)\ =\ |  | (214) |
  |  |  |  |
  | --- | --- | --- |
  |  | 0.17980135762932363.0.17980135762932363\displaystyle 0.17980135762932363\ . |  |
* •

  ∂𝒥22∂μsubscript𝒥22𝜇\frac{\partial{\mathcal{J}}\_{22}}{\partial\mu}

  We use the fact that ∂𝒥22∂μ=∂𝒥21∂νsubscript𝒥22𝜇subscript𝒥21𝜈\frac{\partial{\mathcal{J}}\_{22}}{\partial\mu}=\frac{\partial{\mathcal{J}}\_{21}}{\partial\nu}.
  Thus, an upper bound on the maximal absolute value is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 12​λ012​τmax​ωmax​e−t4​(α012​(−eT12)​erfc⁡(T1)+4​α012​et22​erfc⁡(t2)+2π​(−1)​(α012−1)T22)=12superscriptsubscript𝜆012subscript𝜏maxsubscript𝜔maxsuperscript𝑒subscript𝑡4superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑇12erfcsubscript𝑇14superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑡22erfcsubscript𝑡22𝜋1superscriptsubscript𝛼0121subscript𝑇22absent\displaystyle\frac{1}{2}\lambda\_{\rm 01}^{2}\tau\_{\rm max}\omega\_{\rm max}e^{-t\_{4}}\left(\alpha\_{\rm 01}^{2}\left(-e^{T\_{1}^{2}}\right)\operatorname{erfc}(T\_{1})+4\alpha\_{\rm 01}^{2}e^{t\_{2}^{2}}\operatorname{erfc}(t\_{2})\ +\frac{\sqrt{\frac{2}{\pi}}(-1)\left(\alpha\_{\rm 01}^{2}-1\right)}{\sqrt{T\_{22}}}\right)\ =\ |  | (215) |
  |  |  |  |
  | --- | --- | --- |
  |  | 0.14983446469110305.0.14983446469110305\displaystyle 0.14983446469110305\ . |  |
* •

  ∂𝒥22∂ωsubscript𝒥22𝜔\frac{\partial{\mathcal{J}}\_{22}}{\partial\omega}

  An upper bound on the maximum is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 12​λ012​μmax​τmax​e−t4​(α012​(−eT12)​erfc⁡(T1)+4​α012​et22​erfc⁡(t2)+2π​(−1)​(α012−1)T22)=12superscriptsubscript𝜆012subscript𝜇maxsubscript𝜏maxsuperscript𝑒subscript𝑡4superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑇12erfcsubscript𝑇14superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑡22erfcsubscript𝑡22𝜋1superscriptsubscript𝛼0121subscript𝑇22absent\displaystyle\frac{1}{2}\lambda\_{\rm 01}^{2}\mu\_{\rm max}\tau\_{\rm max}e^{-t\_{4}}\left(\alpha\_{\rm 01}^{2}\left(-e^{T\_{1}^{2}}\right)\operatorname{erfc}(T\_{1})+4\alpha\_{\rm 01}^{2}e^{t\_{2}^{2}}\operatorname{erfc}(t\_{2})\ +\frac{\sqrt{\frac{2}{\pi}}(-1)\left(\alpha\_{\rm 01}^{2}-1\right)}{\sqrt{T\_{22}}}\right)\ =\ |  | (216) |
  |  |  |  |
  | --- | --- | --- |
  |  | 0.149834.0.149834\displaystyle 0.149834\ . |  |

  A lower bound on the minimum is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 12​λ012​μmax​τmax​e−t4​(α012​(−et12)​erfc⁡(t1)+4​α012​eT22​erfc⁡(T2)+2π​(−1)​(α012−1)t22)=12superscriptsubscript𝜆012subscript𝜇maxsubscript𝜏maxsuperscript𝑒subscript𝑡4superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑡12erfcsubscript𝑡14superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑇22erfcsubscript𝑇22𝜋1superscriptsubscript𝛼0121subscript𝑡22absent\displaystyle\frac{1}{2}\lambda\_{\rm 01}^{2}\mu\_{\rm max}\tau\_{\rm max}e^{-t\_{4}}\left(\alpha\_{\rm 01}^{2}\left(-e^{t\_{1}^{2}}\right)\operatorname{erfc}(t\_{1})+4\alpha\_{\rm 01}^{2}e^{T\_{2}^{2}}\operatorname{erfc}(T\_{2})\ +\frac{\sqrt{\frac{2}{\pi}}(-1)\left(\alpha\_{\rm 01}^{2}-1\right)}{\sqrt{t\_{22}}}\right)\ =\ |  | (217) |
  |  |  |  |
  | --- | --- | --- |
  |  | −0.0351035.0.0351035\displaystyle-0.0351035\ . |  |

  Thus, an upper bound on the maximal absolute value is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 12​λ012​μmax​τmax​e−t4​(α012​(−eT12)​erfc⁡(T1)+4​α012​et22​erfc⁡(t2)+2π​(−1)​(α012−1)T22)=12superscriptsubscript𝜆012subscript𝜇maxsubscript𝜏maxsuperscript𝑒subscript𝑡4superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑇12erfcsubscript𝑇14superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑡22erfcsubscript𝑡22𝜋1superscriptsubscript𝛼0121subscript𝑇22absent\displaystyle\frac{1}{2}\lambda\_{\rm 01}^{2}\mu\_{\rm max}\tau\_{\rm max}e^{-t\_{4}}\left(\alpha\_{\rm 01}^{2}\left(-e^{T\_{1}^{2}}\right)\operatorname{erfc}(T\_{1})+4\alpha\_{\rm 01}^{2}e^{t\_{2}^{2}}\operatorname{erfc}(t\_{2})\ +\frac{\sqrt{\frac{2}{\pi}}(-1)\left(\alpha\_{\rm 01}^{2}-1\right)}{\sqrt{T\_{22}}}\right)\ =\ |  | (218) |
  |  |  |  |
  | --- | --- | --- |
  |  | 0.14983446469110305.0.14983446469110305\displaystyle 0.14983446469110305\ . |  |
* •

  ∂𝒥22∂νsubscript𝒥22𝜈\frac{\partial{\mathcal{J}}\_{22}}{\partial\nu}

  We apply Lemma [35](#Thmtheorem35 "Lemma 35 (Function √{2/𝜋}⁢({(𝛼²-1)⁢𝜇⁢𝜔/(𝜈⁢𝜏)^{3/2}}-{3⁢𝛼²/√𝜈⁢𝜏})). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") to the expression
  2π​((α2−1)​μ​ω(ν​τ)3/2−3​α2ν​τ)2𝜋superscript𝛼21𝜇𝜔superscript𝜈𝜏323superscript𝛼2𝜈𝜏\sqrt{\frac{2}{\pi}}\left(\frac{\left(\alpha^{2}-1\right)\mu\omega}{(\nu\tau)^{3/2}}-\frac{3\alpha^{2}}{\sqrt{\nu\tau}}\right).
  Using Lemma [35](#Thmtheorem35 "Lemma 35 (Function √{2/𝜋}⁢({(𝛼²-1)⁢𝜇⁢𝜔/(𝜈⁢𝜏)^{3/2}}-{3⁢𝛼²/√𝜈⁢𝜏})). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), an upper bound on the maximum is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 14λ012τmax2e−t4(α012(−eT12)erfc(T1)+8α012et22erfc(t2)+\displaystyle\frac{1}{4}\lambda\_{\rm 01}^{2}\tau\_{\rm max}^{2}e^{-t\_{4}}\left(\alpha\_{\rm 01}^{2}\left(-e^{T\_{1}^{2}}\right)\operatorname{erfc}(T\_{1})+8\alpha\_{\rm 01}^{2}e^{t\_{2}^{2}}\operatorname{erfc}(t\_{2})\ +\right. |  | (219) |
  |  |  |  |
  | --- | --- | --- |
  |  | 2π((α012−1)​T11T223/2−3​α012T22))= 1.19441.\displaystyle\left.\sqrt{\frac{2}{\pi}}\left(\frac{\left(\alpha\_{\rm 01}^{2}-1\right)T\_{11}}{T\_{22}^{3/2}}-\frac{3\alpha\_{\rm 01}^{2}}{\sqrt{T\_{22}}}\right)\right)\ =\ 1.19441\ . |  |

  Using Lemma [35](#Thmtheorem35 "Lemma 35 (Function √{2/𝜋}⁢({(𝛼²-1)⁢𝜇⁢𝜔/(𝜈⁢𝜏)^{3/2}}-{3⁢𝛼²/√𝜈⁢𝜏})). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), a lower bound on the minimum is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 14λ012τmax2e−t4(α012(−et12)erfc(t1)+8α012eT22erfc(T2)+\displaystyle\frac{1}{4}\lambda\_{\rm 01}^{2}\tau\_{\rm max}^{2}e^{-t\_{4}}\left(\alpha\_{\rm 01}^{2}\left(-e^{t\_{1}^{2}}\right)\operatorname{erfc}(t\_{1})+8\alpha\_{\rm 01}^{2}e^{T\_{2}^{2}}\operatorname{erfc}(T\_{2})\ +\right. |  | (220) |
  |  |  |  |
  | --- | --- | --- |
  |  | 2π((α012−1)​t11t223/2−3​α012t22))=−1.80574.\displaystyle\left.\sqrt{\frac{2}{\pi}}\left(\frac{\left(\alpha\_{\rm 01}^{2}-1\right)t\_{11}}{t\_{22}^{3/2}}-\frac{3\alpha\_{\rm 01}^{2}}{\sqrt{t\_{22}}}\right)\right)\ =\ -1.80574\ . |  |

  Thus, an upper bound on the maximal absolute value is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | −14λ012τmax2e−t4(α012(−et12)erfc(t1)+8α012eT22erfc(T2)+\displaystyle-\frac{1}{4}\lambda\_{\rm 01}^{2}\tau\_{\rm max}^{2}e^{-t\_{4}}\left(\alpha\_{\rm 01}^{2}\left(-e^{t\_{1}^{2}}\right)\operatorname{erfc}(t\_{1})+8\alpha\_{\rm 01}^{2}e^{T\_{2}^{2}}\operatorname{erfc}(T\_{2})\ +\right. |  | (221) |
  |  |  |  |
  | --- | --- | --- |
  |  | 2π((α012−1)​t11t223/2−3​α012t22))= 1.805740052651535.\displaystyle\left.\sqrt{\frac{2}{\pi}}\left(\frac{\left(\alpha\_{\rm 01}^{2}-1\right)t\_{11}}{t\_{22}^{3/2}}-\frac{3\alpha\_{\rm 01}^{2}}{\sqrt{t\_{22}}}\right)\right)\ =\ 1.805740052651535\ . |  |
* •

  ∂𝒥22∂τsubscript𝒥22𝜏\frac{\partial{\mathcal{J}}\_{22}}{\partial\tau}

  We apply Lemma [36](#Thmtheorem36 "Lemma 36 (Function √{2/𝜋}⁢({(𝛼²-1)⁢𝜇⁢𝜔/√𝜈⁢𝜏}-3⁢𝛼²⁢√𝜈⁢𝜏)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") to the expression
  2π​((α2−1)​μ​ων​τ−3​α2​ν​τ)2𝜋superscript𝛼21𝜇𝜔𝜈𝜏3superscript𝛼2𝜈𝜏\sqrt{\frac{2}{\pi}}\left(\frac{\left(\alpha^{2}-1\right)\mu\omega}{\sqrt{\nu\tau}}-3\alpha^{2}\sqrt{\nu\tau}\right).
    
  We apply Lemma [37](#Thmtheorem37 "Lemma 37 (Function 𝜈⁢𝜏⁢𝑒^{(𝜇⁢𝜔+𝜈⁢𝜏)²/2⁢𝜈⁢𝜏}⁢erfc({𝜇⁢𝜔+𝜈⁢𝜏}/√2⁢√𝜈⁢𝜏)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") to the expression
  ν​τ​e(μ​ω+ν​τ)22​ν​τ​erfc⁡(μ​ω+ν​τ2​ν​τ)𝜈𝜏superscript𝑒superscript𝜇𝜔𝜈𝜏22𝜈𝜏erfc𝜇𝜔𝜈𝜏2𝜈𝜏\nu\tau e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right).
  We apply Lemma [38](#Thmtheorem38 "Lemma 38 (Function 𝜈⁢𝜏⁢𝑒^{(𝜇⁢𝜔+2⁢𝜈⁢𝜏)²/2⁢𝜈⁢𝜏}⁢erfc({𝜇⁢𝜔+2⁢𝜈⁢𝜏}/√2⁢√𝜈⁢𝜏)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") to the expression
  ν​τ​e(μ​ω+2​ν​τ)22​ν​τ​erfc⁡(μ​ω+2​ν​τ2​ν​τ)𝜈𝜏superscript𝑒superscript𝜇𝜔2𝜈𝜏22𝜈𝜏erfc𝜇𝜔2𝜈𝜏2𝜈𝜏\nu\tau e^{\frac{(\mu\omega+2\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right).

  We combine the results of these lemmata to obtain
  an upper bound on the maximum:

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 14λ012(−α012t22e−T4e(T11+t22)22​t22erfc(T11+t222​t22)+\displaystyle\frac{1}{4}\lambda\_{\rm 01}^{2}\left(-\alpha\_{\rm 01}^{2}t\_{22}e^{-T\_{4}}e^{\frac{(T\_{11}+t\_{22})^{2}}{2t\_{22}}}\operatorname{erfc}\left(\frac{T\_{11}+t\_{22}}{\sqrt{2}\sqrt{t\_{22}}}\right)\ +\right. |  | (222) |
  |  |  |  |
  | --- | --- | --- |
  |  | 8​α012​T22​e−t4​e(t11+2​T22)22​T22​erfc⁡(t11+2​T222​T22)−limit-from8superscriptsubscript𝛼012subscript𝑇22superscript𝑒subscript𝑡4superscript𝑒superscriptsubscript𝑡112subscript𝑇2222subscript𝑇22erfcsubscript𝑡112subscript𝑇222subscript𝑇22\displaystyle\left.8\alpha\_{\rm 01}^{2}T\_{22}e^{-t\_{4}}e^{\frac{(t\_{11}+2T\_{22})^{2}}{2T\_{22}}}\operatorname{erfc}\left(\frac{t\_{11}+2T\_{22}}{\sqrt{2}\sqrt{T\_{22}}}\right)\ -\right. |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 2​α012​eT12​e−T4​erfc⁡(T1)+4​α012​et22​e−t4​erfc⁡(t2)+2​(2−erfc⁡(T3))+2superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑇12superscript𝑒subscript𝑇4erfcsubscript𝑇14superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑡22superscript𝑒subscript𝑡4erfcsubscript𝑡2limit-from22erfcsubscript𝑇3\displaystyle\left.2\alpha\_{\rm 01}^{2}e^{T\_{1}^{2}}e^{-T\_{4}}\operatorname{erfc}(T\_{1})+4\alpha\_{\rm 01}^{2}e^{t\_{2}^{2}}e^{-t\_{4}}\operatorname{erfc}(t\_{2})+2(2-\operatorname{erfc}(T\_{3}))\ +\right. |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 2πe−T4((α012−1)​T11t22−3α012t22))= 2.39669.\displaystyle\left.\sqrt{\frac{2}{\pi}}e^{-T\_{4}}\left(\frac{\left(\alpha\_{\rm 01}^{2}-1\right)T\_{11}}{\sqrt{t\_{22}}}-3\alpha\_{\rm 01}^{2}\sqrt{t\_{22}}\right)\right)\ =\ 2.39669\ . |  |

  We combine the results of these lemmata to obtain
  an lower bound on the minimum:

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 14λ012(8α012t22e−T4e(T11+2​t22)22​t22erfc(T11+2​t222​t22)+\displaystyle\frac{1}{4}\lambda\_{\rm 01}^{2}\left(8\alpha\_{\rm 01}^{2}t\_{22}e^{-T\_{4}}e^{\frac{(T\_{11}+2t\_{22})^{2}}{2t\_{22}}}\operatorname{erfc}\left(\frac{T\_{11}+2t\_{22}}{\sqrt{2}\sqrt{t\_{22}}}\right)\ +\right. |  | (223) |
  |  |  |  |
  | --- | --- | --- |
  |  | α012​T22​e−t4​e(t11+T22)22​T22​erfc⁡(t11+T222​T22)−limit-fromsuperscriptsubscript𝛼012subscript𝑇22superscript𝑒subscript𝑡4superscript𝑒superscriptsubscript𝑡11subscript𝑇2222subscript𝑇22erfcsubscript𝑡11subscript𝑇222subscript𝑇22\displaystyle\left.\alpha\_{\rm 01}^{2}T\_{22}e^{-t\_{4}}e^{\frac{(t\_{11}+T\_{22})^{2}}{2T\_{22}}}\operatorname{erfc}\left(\frac{t\_{11}+T\_{22}}{\sqrt{2}\sqrt{T\_{22}}}\right)\ -\right. |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 2​α012​et12​e−t4​erfc⁡(t1)+4​α012​eT22​e−T4​erfc⁡(T2)+2superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑡12superscript𝑒subscript𝑡4erfcsubscript𝑡1limit-from4superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑇22superscript𝑒subscript𝑇4erfcsubscript𝑇2\displaystyle\left.2\alpha\_{\rm 01}^{2}e^{t\_{1}^{2}}e^{-t\_{4}}\operatorname{erfc}(t\_{1})+4\alpha\_{\rm 01}^{2}e^{T\_{2}^{2}}e^{-T\_{4}}\operatorname{erfc}(T\_{2})\ +\right. |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 2(2−erfc(t3))+2πe−t4((α012−1)​t11T22−3α012T22))=−1.17154.\displaystyle\left.2(2-\operatorname{erfc}(t\_{3}))+\sqrt{\frac{2}{\pi}}e^{-t\_{4}}\left(\frac{\left(\alpha\_{\rm 01}^{2}-1\right)t\_{11}}{\sqrt{T\_{22}}}-3\alpha\_{\rm 01}^{2}\sqrt{T\_{22}}\right)\right)\ =\ -1.17154\ . |  |

  Thus, an upper bound on the maximal absolute value is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 14λ012(−α012t22e−T4e(T11+t22)22​t22erfc(T11+t222​t22)+\displaystyle\frac{1}{4}\lambda\_{\rm 01}^{2}\left(-\alpha\_{\rm 01}^{2}t\_{22}e^{-T\_{4}}e^{\frac{(T\_{11}+t\_{22})^{2}}{2t\_{22}}}\operatorname{erfc}\left(\frac{T\_{11}+t\_{22}}{\sqrt{2}\sqrt{t\_{22}}}\right)\ +\right. |  | (224) |
  |  |  |  |
  | --- | --- | --- |
  |  | 8​α012​T22​e−t4​e(t11+2​T22)22​T22​erfc⁡(t11+2​T222​T22)−limit-from8superscriptsubscript𝛼012subscript𝑇22superscript𝑒subscript𝑡4superscript𝑒superscriptsubscript𝑡112subscript𝑇2222subscript𝑇22erfcsubscript𝑡112subscript𝑇222subscript𝑇22\displaystyle\left.8\alpha\_{\rm 01}^{2}T\_{22}e^{-t\_{4}}e^{\frac{(t\_{11}+2T\_{22})^{2}}{2T\_{22}}}\operatorname{erfc}\left(\frac{t\_{11}+2T\_{22}}{\sqrt{2}\sqrt{T\_{22}}}\right)\ -\right. |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 2​α012​eT12​e−T4​erfc⁡(T1)+4​α012​et22​e−t4​erfc⁡(t2)+2​(2−erfc⁡(T3))+2superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑇12superscript𝑒subscript𝑇4erfcsubscript𝑇14superscriptsubscript𝛼012superscript𝑒superscriptsubscript𝑡22superscript𝑒subscript𝑡4erfcsubscript𝑡2limit-from22erfcsubscript𝑇3\displaystyle\left.2\alpha\_{\rm 01}^{2}e^{T\_{1}^{2}}e^{-T\_{4}}\operatorname{erfc}(T\_{1})+4\alpha\_{\rm 01}^{2}e^{t\_{2}^{2}}e^{-t\_{4}}\operatorname{erfc}(t\_{2})+2(2-\operatorname{erfc}(T\_{3}))\ +\right. |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 2πe−T4((α012−1)​T11t22−3α012t22))= 2.396685907216327.\displaystyle\left.\sqrt{\frac{2}{\pi}}e^{-T\_{4}}\left(\frac{\left(\alpha\_{\rm 01}^{2}-1\right)T\_{11}}{\sqrt{t\_{22}}}-3\alpha\_{\rm 01}^{2}\sqrt{t\_{22}}\right)\right)\ =\ 2.396685907216327\ . |  |

∎

###### Lemma 40 (Derivatives of the Mapping).

We assume α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01} and λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01}.
We restrict the range of the variables to the domain
μ∈[−0.1,0.1]𝜇0.10.1\mu\in[-0.1,0.1],
ω∈[−0.1,0.1]𝜔0.10.1\omega\in[-0.1,0.1],
ν∈[0.8,1.5]𝜈0.81.5\nu\in[0.8,1.5], and
τ∈[0.8,1.25]𝜏0.81.25\tau\in[0.8,1.25].

The derivative ∂∂μ​μ~​(μ,ω,ν,τ,λ,α)𝜇~𝜇𝜇𝜔𝜈𝜏𝜆𝛼\frac{\partial}{\partial\mu}{\tilde{\mu}}(\mu,\omega,\nu,\tau,\lambda,\alpha)
has the sign of ω𝜔\omega.

The derivative ∂∂ν​μ~​(μ,ω,ν,τ,λ,α)𝜈~𝜇𝜇𝜔𝜈𝜏𝜆𝛼\frac{\partial}{\partial\nu}{\tilde{\mu}}(\mu,\omega,\nu,\tau,\lambda,\alpha)
is positive.

The derivative ∂∂μ​ξ~​(μ,ω,ν,τ,λ,α)𝜇~𝜉𝜇𝜔𝜈𝜏𝜆𝛼\frac{\partial}{\partial\mu}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda,\alpha)
has the sign of ω𝜔\omega.

The derivative
∂∂ν​ξ~​(μ,ω,ν,τ,λ,α)𝜈~𝜉𝜇𝜔𝜈𝜏𝜆𝛼\frac{\partial}{\partial\nu}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda,\alpha)
is positive.

###### Proof.

* •

  ∂∂μ​μ~​(μ,ω,ν,τ,λ,α)𝜇~𝜇𝜇𝜔𝜈𝜏𝜆𝛼\frac{\partial}{\partial\mu}{\tilde{\mu}}(\mu,\omega,\nu,\tau,\lambda,\alpha)

  (2−erfc(x)>0(2-\operatorname{erfc}(x)>0 according to
  Lemma [21](#Thmtheorem21 "Lemma 21 (Basic functions). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") and ex2​erfc⁡(x)superscript𝑒superscript𝑥2erfc𝑥e^{x^{2}}\operatorname{erfc}(x)
  is also larger than zero according to Lemma [23](#Thmtheorem23 "Lemma 23 (Function 𝑒^𝑥²⁢erfc(𝑥)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").
  Consequently, has ∂∂μ​μ~​(μ,ω,ν,τ,λ,α)𝜇~𝜇𝜇𝜔𝜈𝜏𝜆𝛼\frac{\partial}{\partial\mu}{\tilde{\mu}}(\mu,\omega,\nu,\tau,\lambda,\alpha)
  the sign of ω𝜔\omega.
* •

  ∂∂ν​μ~​(μ,ω,ν,τ,λ,α)𝜈~𝜇𝜇𝜔𝜈𝜏𝜆𝛼\frac{\partial}{\partial\nu}{\tilde{\mu}}(\mu,\omega,\nu,\tau,\lambda,\alpha)

  Lemma [23](#Thmtheorem23 "Lemma 23 (Function 𝑒^𝑥²⁢erfc(𝑥)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") says
  ex2​erfc⁡(x)superscript𝑒superscript𝑥2erfc𝑥e^{x^{2}}\operatorname{erfc}(x) is decreasing in μ​ω+ν​τ2​ν​τ𝜇𝜔𝜈𝜏2𝜈𝜏\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}.
  The first term (negative) is increasing in ν​τ𝜈𝜏\nu\tau since it is
  proportional to minus
  one over the squared root of ν​τ𝜈𝜏\nu\tau.

  We obtain a lower bound by
  setting μ​ω+ν​τ2​ν​τ=1.5⋅1.25+0.1⋅0.12​1.5⋅1.25𝜇𝜔𝜈𝜏2𝜈𝜏⋅1.51.25⋅0.10.12⋅1.51.25\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}=\frac{1.5\cdot 1.25+0.1\cdot 0.1}{\sqrt{2}\sqrt{1.5\cdot 1.25}} for the ex2​erfc⁡(x)superscript𝑒superscript𝑥2erfc𝑥e^{x^{2}}\operatorname{erfc}(x) term.
  The term in brackets is larger than
  e(1.5⋅1.25+0.1⋅0.12​1.5⋅1.25)2​α01​erfc⁡(1.5⋅1.25+0.1⋅0.12​1.5⋅1.25)−2π​0.8⋅0.8​(α01−1)=0.056superscript𝑒superscript⋅1.51.25⋅0.10.12⋅1.51.252subscript𝛼01erfc⋅1.51.25⋅0.10.12⋅1.51.252⋅𝜋0.80.8subscript𝛼0110.056e^{\left(\frac{1.5\cdot 1.25+0.1\cdot 0.1}{\sqrt{2}\sqrt{1.5\cdot 1.25}}\right)^{2}}\alpha\_{\rm 01}\ \operatorname{erfc}\left(\frac{1.5\cdot 1.25+0.1\cdot 0.1}{\sqrt{2}\sqrt{1.5\cdot 1.25}}\right)-\sqrt{\frac{2}{\pi 0.8\cdot 0.8}}(\alpha\_{\rm 01}-1)=0.056
  Consequently, the function is larger than zero.
* •

  ∂∂μ​ξ~​(μ,ω,ν,τ,λ,α)𝜇~𝜉𝜇𝜔𝜈𝜏𝜆𝛼\frac{\partial}{\partial\mu}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda,\alpha)

  We consider the sub-function

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 2π​ν​τ−α2​(e(μ​ω+ν​τ2​ν​τ)2​erfc⁡(μ​ω+ν​τ2​ν​τ)−e(μ​ω+2​ν​τ2​ν​τ)2​erfc⁡(μ​ω+2​ν​τ2​ν​τ)).2𝜋𝜈𝜏superscript𝛼2superscript𝑒superscript𝜇𝜔𝜈𝜏2𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏superscript𝑒superscript𝜇𝜔2𝜈𝜏2𝜈𝜏2erfc𝜇𝜔2𝜈𝜏2𝜈𝜏\displaystyle\sqrt{\frac{2}{\pi}}\sqrt{\nu\tau}-\alpha^{2}\left(e^{\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-e^{\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)\ . |  | (225) |

  We set x=ν​τ𝑥𝜈𝜏x=\nu\tau and y=μ​ω𝑦𝜇𝜔y=\mu\omega and obtain

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 2π​x−α2​(e(x+y2​x)2​erfc⁡(x+y2​x)−e(2​x+y2​x)2​erfc⁡(2​x+y2​x)).2𝜋𝑥superscript𝛼2superscript𝑒superscript𝑥𝑦2𝑥2erfc𝑥𝑦2𝑥superscript𝑒superscript2𝑥𝑦2𝑥2erfc2𝑥𝑦2𝑥\displaystyle\sqrt{\frac{2}{\pi}}\sqrt{x}-\alpha^{2}\left(e^{\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)^{2}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)-e^{\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)^{2}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)\right)\ . |  | (226) |

  The derivative of this sub-function with respect to y𝑦y is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | α2​(e(2​x+y)22​x​(2​x+y)​erfc⁡(2​x+y2​x)−e(x+y)22​x​(x+y)​erfc⁡(x+y2​x))x=superscript𝛼2superscript𝑒superscript2𝑥𝑦22𝑥2𝑥𝑦erfc2𝑥𝑦2𝑥superscript𝑒superscript𝑥𝑦22𝑥𝑥𝑦erfc𝑥𝑦2𝑥𝑥absent\displaystyle\frac{\alpha^{2}\left(e^{\frac{(2x+y)^{2}}{2x}}(2x+y)\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)-e^{\frac{(x+y)^{2}}{2x}}(x+y)\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)\right)}{x}\ = |  | (227) |
  |  |  |  |
  | --- | --- | --- |
  |  | 2​α2​x​(e(2​x+y)22​x​(x+y)​erfc⁡(x+y2​x)2​x−e(x+y)22​x​(x+y)​erfc⁡(x+y2​x)2​x)x> 0.2superscript𝛼2𝑥superscript𝑒superscript2𝑥𝑦22𝑥𝑥𝑦erfc𝑥𝑦2𝑥2𝑥superscript𝑒superscript𝑥𝑦22𝑥𝑥𝑦erfc𝑥𝑦2𝑥2𝑥𝑥 0\displaystyle\frac{\sqrt{2}\alpha^{2}\sqrt{x}\left(\frac{e^{\frac{(2x+y)^{2}}{2x}}(x+y)\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)}{\sqrt{2}\sqrt{x}}-\frac{e^{\frac{(x+y)^{2}}{2x}}(x+y)\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)}{\sqrt{2}\sqrt{x}}\right)}{x}\ >\ 0\ . |  |

  The inequality follows from Lemma [24](#Thmtheorem24 "Lemma 24 (Properties of 𝑥⁢𝑒^𝑥²⁢erfc(𝑥)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), which states that
  z​ez2​erfc⁡(z)𝑧superscript𝑒superscript𝑧2erfc𝑧ze^{z^{2}}\operatorname{erfc}(z) is monotonically increasing in z𝑧z.
  Therefore the sub-function is increasing in y𝑦y.

  The derivative of this sub-function with respect to x𝑥x is

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | π​α2​(e(2​x+y)22​x​(4​x2−y2)​erfc⁡(2​x+y2​x)−e(x+y)22​x​(x−y)​(x+y)​erfc⁡(x+y2​x))−2​(α2−1)​x3/22​π​x2.𝜋superscript𝛼2superscript𝑒superscript2𝑥𝑦22𝑥4superscript𝑥2superscript𝑦2erfc2𝑥𝑦2𝑥superscript𝑒superscript𝑥𝑦22𝑥𝑥𝑦𝑥𝑦erfc𝑥𝑦2𝑥2superscript𝛼21superscript𝑥322𝜋superscript𝑥2\displaystyle\frac{\sqrt{\pi}\alpha^{2}\left(e^{\frac{(2x+y)^{2}}{2x}}\left(4x^{2}-y^{2}\right)\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)-e^{\frac{(x+y)^{2}}{2x}}(x-y)(x+y)\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)\right)-\sqrt{2}\left(\alpha^{2}-1\right)x^{3/2}}{2\sqrt{\pi}x^{2}}\ . |  | (228) |

  The sub-function is increasing in x𝑥x, since the
  derivative is larger than zero:

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | π​α2​(e(2​x+y)22​x​(4​x2−y2)​erfc⁡(2​x+y2​x)−e(x+y)22​x​(x−y)​(x+y)​erfc⁡(x+y2​x))−2​x3/2​(α2−1)2​π​x2⩾𝜋superscript𝛼2superscript𝑒superscript2𝑥𝑦22𝑥4superscript𝑥2superscript𝑦2erfc2𝑥𝑦2𝑥superscript𝑒superscript𝑥𝑦22𝑥𝑥𝑦𝑥𝑦erfc𝑥𝑦2𝑥2superscript𝑥32superscript𝛼212𝜋superscript𝑥2absent\displaystyle\frac{\sqrt{\pi}\alpha^{2}\left(e^{\frac{(2x+y)^{2}}{2x}}\left(4x^{2}-y^{2}\right)\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)-e^{\frac{(x+y)^{2}}{2x}}(x-y)(x+y)\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)\right)-\sqrt{2}x^{3/2}\left(\alpha^{2}-1\right)}{2\sqrt{\pi}x^{2}}\ \geqslant |  | (229) |
  |  |  |  |
  | --- | --- | --- |
  |  | π​α2​((2​x−y)​(2​x+y)​2π​(2​x+y2​x+(2​x+y2​x)2+2)−(x−y)​(x+y)​2π​(x+y2​x+(x+y2​x)2+4π))−2​x3/2​(α2−1)2​π​x2=𝜋superscript𝛼22𝑥𝑦2𝑥𝑦2𝜋2𝑥𝑦2𝑥superscript2𝑥𝑦2𝑥22𝑥𝑦𝑥𝑦2𝜋𝑥𝑦2𝑥superscript𝑥𝑦2𝑥24𝜋2superscript𝑥32superscript𝛼212𝜋superscript𝑥2absent\displaystyle\frac{\sqrt{\pi}\alpha^{2}\left(\frac{(2x-y)(2x+y)2}{\sqrt{\pi}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}+\sqrt{\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)^{2}+2}\right)}-\frac{(x-y)(x+y)2}{\sqrt{\pi}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}+\sqrt{\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)^{2}+\frac{4}{\pi}}\right)}\right)-\sqrt{2}x^{3/2}\left(\alpha^{2}-1\right)}{2\sqrt{\pi}x^{2}}\ = |  |
  |  |  |  |
  | --- | --- | --- |
  |  | π​α2​((2​x−y)​(2​x+y)​2​(2​x)π​(2​x+y+(2​x+y)2+4​x)−(x−y)​(x+y)​2​(2​x)π​(x+y+(x+y)2+8​xπ))−2​x3/2​(α2−1)2​π​x2=𝜋superscript𝛼22𝑥𝑦2𝑥𝑦22𝑥𝜋2𝑥𝑦superscript2𝑥𝑦24𝑥𝑥𝑦𝑥𝑦22𝑥𝜋𝑥𝑦superscript𝑥𝑦28𝑥𝜋2superscript𝑥32superscript𝛼212𝜋superscript𝑥2absent\displaystyle\frac{\sqrt{\pi}\alpha^{2}\left(\frac{(2x-y)(2x+y)2\left(\sqrt{2}\sqrt{x}\right)}{\sqrt{\pi}\left(2x+y+\sqrt{(2x+y)^{2}+4x}\right)}-\frac{(x-y)(x+y)2\left(\sqrt{2}\sqrt{x}\right)}{\sqrt{\pi}\left(x+y+\sqrt{(x+y)^{2}+\frac{8x}{\pi}}\right)}\right)-\sqrt{2}x^{3/2}\left(\alpha^{2}-1\right)}{2\sqrt{\pi}x^{2}}\ = |  |
  |  |  |  |
  | --- | --- | --- |
  |  | π​α2​((2​x−y)​(2​x+y)​2π​(2​x+y+(2​x+y)2+4​x)−(x−y)​(x+y)​2π​(x+y+(x+y)2+8​xπ))−x​(α2−1)2​π​x3/2>𝜋superscript𝛼22𝑥𝑦2𝑥𝑦2𝜋2𝑥𝑦superscript2𝑥𝑦24𝑥𝑥𝑦𝑥𝑦2𝜋𝑥𝑦superscript𝑥𝑦28𝑥𝜋𝑥superscript𝛼212𝜋superscript𝑥32absent\displaystyle\frac{\sqrt{\pi}\alpha^{2}\left(\frac{(2x-y)(2x+y)2}{\sqrt{\pi}\left(2x+y+\sqrt{(2x+y)^{2}+4x}\right)}-\frac{(x-y)(x+y)2}{\sqrt{\pi}\left(x+y+\sqrt{(x+y)^{2}+\frac{8x}{\pi}}\right)}\right)-x\left(\alpha^{2}-1\right)}{\sqrt{2}\sqrt{\pi}x^{3/2}}\ > |  |
  |  |  |  |
  | --- | --- | --- |
  |  | π​α2​((2​x−y)​(2​x+y)​2π​(2​x+y+(2​x+y)2+2​(2​x+y)+1)−(x−y)​(x+y)​2π​(x+y+(x+y)2+0.782⋅2​(x+y)+0.7822))−x​(α2−1)2​π​x3/2=𝜋superscript𝛼22𝑥𝑦2𝑥𝑦2𝜋2𝑥𝑦superscript2𝑥𝑦222𝑥𝑦1𝑥𝑦𝑥𝑦2𝜋𝑥𝑦superscript𝑥𝑦2⋅0.7822𝑥𝑦superscript0.7822𝑥superscript𝛼212𝜋superscript𝑥32absent\displaystyle\frac{\sqrt{\pi}\alpha^{2}\left(\frac{(2x-y)(2x+y)2}{\sqrt{\pi}\left(2x+y+\sqrt{(2x+y)^{2}+2(2x+y)+1}\right)}-\frac{(x-y)(x+y)2}{\sqrt{\pi}\left(x+y+\sqrt{(x+y)^{2}+0.782\cdot 2(x+y)+0.782^{2}}\right)}\right)-x\left(\alpha^{2}-1\right)}{\sqrt{2}\sqrt{\pi}x^{3/2}}\ = |  |
  |  |  |  |
  | --- | --- | --- |
  |  | π​α2​((2​x−y)​(2​x+y)​2π​(2​x+y+(2​x+y+1)2)−(x−y)​(x+y)​2π​(x+y+(x+y+0.782)2))−x​(α2−1)2​π​x3/2=𝜋superscript𝛼22𝑥𝑦2𝑥𝑦2𝜋2𝑥𝑦superscript2𝑥𝑦12𝑥𝑦𝑥𝑦2𝜋𝑥𝑦superscript𝑥𝑦0.7822𝑥superscript𝛼212𝜋superscript𝑥32absent\displaystyle\frac{\sqrt{\pi}\alpha^{2}\left(\frac{(2x-y)(2x+y)2}{\sqrt{\pi}\left(2x+y+\sqrt{(2x+y+1)^{2}}\right)}-\frac{(x-y)(x+y)2}{\sqrt{\pi}\left(x+y+\sqrt{(x+y+0.782)^{2}}\right)}\right)-x\left(\alpha^{2}-1\right)}{\sqrt{2}\sqrt{\pi}x^{3/2}}\ = |  |
  |  |  |  |
  | --- | --- | --- |
  |  | π​α2​((2​x−y)​(2​x+y)​2π​(2​(2​x+y)+1)−(x−y)​(x+y)​2π​(2​(x+y)+0.782))−x​(α2−1)2​π​x3/2=𝜋superscript𝛼22𝑥𝑦2𝑥𝑦2𝜋22𝑥𝑦1𝑥𝑦𝑥𝑦2𝜋2𝑥𝑦0.782𝑥superscript𝛼212𝜋superscript𝑥32absent\displaystyle\frac{\sqrt{\pi}\alpha^{2}\left(\frac{(2x-y)(2x+y)2}{\sqrt{\pi}(2(2x+y)+1)}-\frac{(x-y)(x+y)2}{\sqrt{\pi}(2(x+y)+0.782)}\right)-x\left(\alpha^{2}-1\right)}{\sqrt{2}\sqrt{\pi}x^{3/2}}\ = |  |
  |  |  |  |
  | --- | --- | --- |
  |  | π​α2​((2​(x+y)+0.782)​(2​x−y)​(2​x+y)​2π−(x−y)​(x+y)​(2​(2​x+y)+1)​2π)(2​(2​x+y)+1)​(2​(x+y)+0.782)​2​π​x3/2+limit-from𝜋superscript𝛼22𝑥𝑦0.7822𝑥𝑦2𝑥𝑦2𝜋𝑥𝑦𝑥𝑦22𝑥𝑦12𝜋22𝑥𝑦12𝑥𝑦0.7822𝜋superscript𝑥32\displaystyle\frac{\sqrt{\pi}\alpha^{2}\left(\frac{(2(x+y)+0.782)(2x-y)(2x+y)2}{\sqrt{\pi}}-\frac{(x-y)(x+y)(2(2x+y)+1)2}{\sqrt{\pi}}\right)}{(2(2x+y)+1)(2(x+y)+0.782)\sqrt{2}\sqrt{\pi}x^{3/2}}\ + |  |
  |  |  |  |
  | --- | --- | --- |
  |  | π​α2​(−x​(α2−1)​(2​(2​x+y)+1)​(2​(x+y)+0.782))(2​(2​x+y)+1)​(2​(x+y)+0.782)​2​π​x3/2=𝜋superscript𝛼2𝑥superscript𝛼2122𝑥𝑦12𝑥𝑦0.78222𝑥𝑦12𝑥𝑦0.7822𝜋superscript𝑥32absent\displaystyle\frac{\sqrt{\pi}\alpha^{2}\left(-x\left(\alpha^{2}-1\right)(2(2x+y)+1)(2(x+y)+0.782)\right)}{(2(2x+y)+1)(2(x+y)+0.782)\sqrt{2}\sqrt{\pi}x^{3/2}}\ = |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 8​x3+(12​y+2.68657)​x2+(y​(4​y−6.41452)−1.40745)​x+1.22072​y2(2​(2​x+y)+1)​(2​(x+y)+0.782)​2​π​x3/2>8superscript𝑥312𝑦2.68657superscript𝑥2𝑦4𝑦6.414521.40745𝑥1.22072superscript𝑦222𝑥𝑦12𝑥𝑦0.7822𝜋superscript𝑥32absent\displaystyle\frac{8x^{3}+(12y+2.68657)x^{2}+(y(4y-6.41452)-1.40745)x+1.22072y^{2}}{(2(2x+y)+1)(2(x+y)+0.782)\sqrt{2}\sqrt{\pi}x^{3/2}}\ > |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 8​x3+(2.68657−120.01)​x2+(0.01​(−6.41452−40.01)−1.40745)​x+1.22072​(0.0)2(2​(2​x+y)+1)​(2​(x+y)+0.782)​2​π​x3/2=8superscript𝑥32.68657120.01superscript𝑥20.016.4145240.011.40745𝑥1.22072superscript0.0222𝑥𝑦12𝑥𝑦0.7822𝜋superscript𝑥32absent\displaystyle\frac{8x^{3}+(2.68657-120.01)x^{2}+(0.01(-6.41452-40.01)-1.40745)x+1.22072(0.0)^{2}}{(2(2x+y)+1)(2(x+y)+0.782)\sqrt{2}\sqrt{\pi}x^{3/2}}\ = |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 8​x2+2.56657​x−1.472(2​(2​x+y)+1)​(2​(x+y)+0.782)​2​π​x=8superscript𝑥22.56657𝑥1.47222𝑥𝑦12𝑥𝑦0.7822𝜋𝑥absent\displaystyle\frac{8x^{2}+2.56657x-1.472}{(2(2x+y)+1)(2(x+y)+0.782)\sqrt{2}\sqrt{\pi}\sqrt{x}}\ = |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 8​x2+2.56657​x−1.472(2​(2​x+y)+1)​(2​(x+y)+0.782)​2​π​x=8superscript𝑥22.56657𝑥1.47222𝑥𝑦12𝑥𝑦0.7822𝜋𝑥absent\displaystyle\frac{8x^{2}+2.56657x-1.472}{(2(2x+y)+1)(2(x+y)+0.782)\sqrt{2}\sqrt{\pi}\sqrt{x}}\ = |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 8​(x+0.618374)​(x−0.297553)(2​(2​x+y)+1)​(2​(x+y)+0.782)​2​π​x> 0.8𝑥0.618374𝑥0.29755322𝑥𝑦12𝑥𝑦0.7822𝜋𝑥 0\displaystyle\frac{8(x+0.618374)(x-0.297553)}{(2(2x+y)+1)(2(x+y)+0.782)\sqrt{2}\sqrt{\pi}\sqrt{x}}\ >\ 0\ . |  |

  We explain this chain of inequalities:

  + –

    First inequality: We applied Lemma [22](#Thmtheorem22 "Lemma 22 (Erfc bound from Abramowitz). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") two times.
  + –

    Equalities factor out 2​x2𝑥\sqrt{2}\sqrt{x} and reformulate.
  + –

    Second inequality part 1: we applied

    |  |  |  |  |
    | --- | --- | --- | --- |
    |  | 0<2​y⟹(2​x+y)2+4​x+1<(2​x+y)2+2​(2​x+y)+1=(2​x+y+1)2.02𝑦⟹superscript2𝑥𝑦24𝑥1superscript2𝑥𝑦222𝑥𝑦1superscript2𝑥𝑦12\displaystyle 0<2y\Longrightarrow(2x+y)^{2}+4x+1<(2x+y)^{2}+2(2x+y)+1=(2x+y+1)^{2}\ . |  | (230) |
  + –

    Second inequality part 2: we show that for a=120​(2048+169​ππ−13)𝑎1202048169𝜋𝜋13a=\frac{1}{20}\left(\sqrt{\frac{2048+169\pi}{\pi}}-13\right) following holds:
    8​xπ−(a2+2​a​(x+y))⩾08𝑥𝜋superscript𝑎22𝑎𝑥𝑦0\frac{8x}{\pi}-\left(a^{2}+2a(x+y)\right)\geqslant 0.
    We have ∂∂x​8​xπ−(a2+2​a​(x+y))=8π−2​a>0𝑥8𝑥𝜋superscript𝑎22𝑎𝑥𝑦8𝜋2𝑎0\frac{\partial}{\partial x}\frac{8x}{\pi}-\left(a^{2}+2a(x+y)\right)=\frac{8}{\pi}-2a>0 and
    ∂∂y​8​xπ−(a2+2​a​(x+y))=−2​a>0𝑦8𝑥𝜋superscript𝑎22𝑎𝑥𝑦2𝑎0\frac{\partial}{\partial y}\frac{8x}{\pi}-\left(a^{2}+2a(x+y)\right)=-2a>0.
    Therefore the minimum is at border for minimal x𝑥x and maximal y𝑦y:

    |  |  |  |  |
    | --- | --- | --- | --- |
    |  | 8⋅0.64π−(220​(2048+169​ππ−13)​(0.64+0.01)+(120​(2048+169​ππ−13))2)= 0.⋅80.64𝜋2202048169𝜋𝜋130.640.01superscript1202048169𝜋𝜋132 0\displaystyle\frac{8\cdot 0.64}{\pi}-\left(\frac{2}{20}\left(\sqrt{\frac{2048+169\pi}{\pi}}-13\right)(0.64+0.01)+\left(\frac{1}{20}\left(\sqrt{\frac{2048+169\pi}{\pi}}-13\right)\right)^{2}\right)\ =\ 0\ . |  | (231) |

    Thus

    |  |  |  |  |
    | --- | --- | --- | --- |
    |  | 8​xπ⩾a2+2​a​(x+y).8𝑥𝜋superscript𝑎22𝑎𝑥𝑦\displaystyle\frac{8x}{\pi}\ \geqslant\ a^{2}+2a(x+y)\ . |  | (232) |

    for a=120​(2048+169​ππ−13)>0.782𝑎1202048169𝜋𝜋130.782a=\frac{1}{20}\left(\sqrt{\frac{2048+169\pi}{\pi}}-13\right)>0.782.
  + –

    Equalities only solve square root and factor out the resulting
    terms (2​(2​x+y)+1)22𝑥𝑦1(2(2x+y)+1) and (2​(x+y)+0.782)2𝑥𝑦0.782(2(x+y)+0.782).
  + –

    We set α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01} and multiplied out. Thereafter we
    also factored out x𝑥x in the numerator. Finally a quadratic
    equations was solved.

  The sub-function has its minimal value for
  minimal x𝑥x and minimal y𝑦y
  x=ν​τ=0.8⋅0.8=0.64𝑥𝜈𝜏⋅0.80.80.64x=\nu\tau=0.8\cdot 0.8=0.64 and y=μ​ω=−0.1⋅0.1=−0.01𝑦𝜇𝜔⋅0.10.10.01y=\mu\omega=-0.1\cdot 0.1=-0.01.
  We further minimize the function

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | μ​ω​eμ2​ω22​ν​τ​(2−erfc⁡(μ​ω2​ν​τ))>−0.01​e0.01220.64​(2−erfc⁡(0.012​0.64)).𝜇𝜔superscript𝑒superscript𝜇2superscript𝜔22𝜈𝜏2erfc𝜇𝜔2𝜈𝜏0.01superscript𝑒superscript0.01220.642erfc0.0120.64\displaystyle\mu\omega e^{\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\left(2-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)>-0.01e^{\frac{0.01^{2}}{20.64}}\left(2-\operatorname{erfc}\left(\frac{0.01}{\sqrt{2}\sqrt{0.64}}\right)\right)\ . |  | (233) |

  We compute the minimum of the term in brackets of ∂∂μ​ξ~​(μ,ω,ν,τ,λ,α)𝜇~𝜉𝜇𝜔𝜈𝜏𝜆𝛼\frac{\partial}{\partial\mu}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda,\alpha):

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | μ​ω​eμ2​ω22​ν​τ​(2−erfc⁡(μ​ω2​ν​τ))+limit-from𝜇𝜔superscript𝑒superscript𝜇2superscript𝜔22𝜈𝜏2erfc𝜇𝜔2𝜈𝜏\displaystyle\mu\omega e^{\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\left(2-\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)+ |  | (234) |
  |  |  |  |
  | --- | --- | --- |
  |  | α012​(−(e(μ​ω+ν​τ2​ν​τ)2​erfc⁡(μ​ω+ν​τ2​ν​τ)−e(μ​ω+2​ν​τ2​ν​τ)2​erfc⁡(μ​ω+2​ν​τ2​ν​τ)))+2π​ν​τ>superscriptsubscript𝛼012superscript𝑒superscript𝜇𝜔𝜈𝜏2𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏superscript𝑒superscript𝜇𝜔2𝜈𝜏2𝜈𝜏2erfc𝜇𝜔2𝜈𝜏2𝜈𝜏2𝜋𝜈𝜏absent\displaystyle\alpha\_{\rm 01}^{2}\left(-\left(e^{\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-e^{\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)\right)+\sqrt{\frac{2}{\pi}}\sqrt{\nu\tau}\ > |  |
  |  |  |  |
  | --- | --- | --- |
  |  | α012​(−(e(0.64−0.012​0.64)2​erfc⁡(0.64−0.012​0.64)−e(20.64−0.012​0.64)2​erfc⁡(2⋅0.64−0.012​0.64)))−limit-fromsuperscriptsubscript𝛼012superscript𝑒superscript0.640.0120.642erfc0.640.0120.64superscript𝑒superscript20.640.0120.642erfc⋅20.640.0120.64\displaystyle\alpha\_{\rm 01}^{2}\left(-\left(e^{\left(\frac{0.64-0.01}{\sqrt{2}\sqrt{0.64}}\right)^{2}}\operatorname{erfc}\left(\frac{0.64-0.01}{\sqrt{2}\sqrt{0.64}}\right)-e^{\left(\frac{20.64-0.01}{\sqrt{2}\sqrt{0.64}}\right)^{2}}\operatorname{erfc}\left(\frac{2\cdot 0.64-0.01}{\sqrt{2}\sqrt{0.64}}\right)\right)\right)- |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 0.01​e0.01220.64​(2−erfc⁡(0.012​0.64))+0.64​2π= 0.0923765.0.01superscript𝑒superscript0.01220.642erfc0.0120.640.642𝜋0.0923765\displaystyle 0.01e^{\frac{0.01^{2}}{20.64}}\left(2-\operatorname{erfc}\left(\frac{0.01}{\sqrt{2}\sqrt{0.64}}\right)\right)+\sqrt{0.64}\sqrt{\frac{2}{\pi}}\ =\ 0.0923765\ . |  |

  Therefore the term in brackets is larger than zero.

  Thus, ∂∂μ​ξ~​(μ,ω,ν,τ,λ,α)𝜇~𝜉𝜇𝜔𝜈𝜏𝜆𝛼\frac{\partial}{\partial\mu}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda,\alpha)
  has the sign of ω𝜔\omega.
* •

  ∂∂ν​ξ~​(μ,ω,ν,τ,λ,α)𝜈~𝜉𝜇𝜔𝜈𝜏𝜆𝛼\frac{\partial}{\partial\nu}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda,\alpha)

  We look at the sub-term

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 2​e(2​x+y2​x)2​erfc⁡(2​x+y2​x)−e(x+y2​x)2​erfc⁡(x+y2​x).2superscript𝑒superscript2𝑥𝑦2𝑥2erfc2𝑥𝑦2𝑥superscript𝑒superscript𝑥𝑦2𝑥2erfc𝑥𝑦2𝑥\displaystyle 2e^{\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)^{2}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)-e^{\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)^{2}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)\ . |  | (235) |

  We obtain a chain of inequalities:

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 2​e(2​x+y2​x)2​erfc⁡(2​x+y2​x)−e(x+y2​x)2​erfc⁡(x+y2​x)>2superscript𝑒superscript2𝑥𝑦2𝑥2erfc2𝑥𝑦2𝑥superscript𝑒superscript𝑥𝑦2𝑥2erfc𝑥𝑦2𝑥absent\displaystyle 2e^{\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)^{2}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)-e^{\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)^{2}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)\ > |  | (236) |
  |  |  |  |
  | --- | --- | --- |
  |  | 2⋅2π​(2​x+y2​x+(2​x+y2​x)2+2)−2π​(x+y2​x+(x+y2​x)2+4π)=⋅22𝜋2𝑥𝑦2𝑥superscript2𝑥𝑦2𝑥222𝜋𝑥𝑦2𝑥superscript𝑥𝑦2𝑥24𝜋absent\displaystyle\frac{2\cdot 2}{\sqrt{\pi}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}+\sqrt{\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)^{2}+2}\right)}-\frac{2}{\sqrt{\pi}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}+\sqrt{\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)^{2}+\frac{4}{\pi}}\right)}\ = |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 2​2​x​(2(2​x+y)2+4​x+2​x+y−1(x+y)2+8​xπ+x+y)π>22𝑥2superscript2𝑥𝑦24𝑥2𝑥𝑦1superscript𝑥𝑦28𝑥𝜋𝑥𝑦𝜋absent\displaystyle\frac{2\sqrt{2}\sqrt{x}\left(\frac{2}{\sqrt{(2x+y)^{2}+4x}+2x+y}-\frac{1}{\sqrt{(x+y)^{2}+\frac{8x}{\pi}}+x+y}\right)}{\sqrt{\pi}}\ > |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 2​2​x​(2(2​x+y)2+2​(2​x+y)+1+2​x+y−1(x+y)2+0.782⋅2​(x+y)+0.7822+x+y)π=22𝑥2superscript2𝑥𝑦222𝑥𝑦12𝑥𝑦1superscript𝑥𝑦2⋅0.7822𝑥𝑦superscript0.7822𝑥𝑦𝜋absent\displaystyle\frac{2\sqrt{2}\sqrt{x}\left(\frac{2}{\sqrt{(2x+y)^{2}+2(2x+y)+1}+2x+y}-\frac{1}{\sqrt{(x+y)^{2}+0.782\cdot 2(x+y)+0.782^{2}}+x+y}\right)}{\sqrt{\pi}}\ = |  |
  |  |  |  |
  | --- | --- | --- |
  |  | 2​2​x​(22​(2​x+y)+1−12​(x+y)+0.782)π=22𝑥222𝑥𝑦112𝑥𝑦0.782𝜋absent\displaystyle\frac{2\sqrt{2}\sqrt{x}\left(\frac{2}{2(2x+y)+1}-\frac{1}{2(x+y)+0.782}\right)}{\sqrt{\pi}}\ = |  |
  |  |  |  |
  | --- | --- | --- |
  |  | (2​2​x)​(2​(2​(x+y)+0.782)−(2​(2​x+y)+1))π​((2​(x+y)+0.782)​(2​(2​x+y)+1))=22𝑥22𝑥𝑦0.78222𝑥𝑦1𝜋2𝑥𝑦0.78222𝑥𝑦1absent\displaystyle\frac{\left(2\sqrt{2}\sqrt{x}\right)(2(2(x+y)+0.782)-(2(2x+y)+1))}{\sqrt{\pi}((2(x+y)+0.782)(2(2x+y)+1))}\ = |  |
  |  |  |  |
  | --- | --- | --- |
  |  | (2​2​x)​(2​y+0.782⋅2−1)π​((2​(x+y)+0.782)​(2​(2​x+y)+1))> 0.22𝑥2𝑦⋅0.78221𝜋2𝑥𝑦0.78222𝑥𝑦1 0\displaystyle\frac{\left(2\sqrt{2}\sqrt{x}\right)(2y+0.782\cdot 2-1)}{\sqrt{\pi}((2(x+y)+0.782)(2(2x+y)+1))}\ >\ 0\ . |  |

  We explain this chain of inequalities:

  + –

    First inequality: We applied Lemma [22](#Thmtheorem22 "Lemma 22 (Erfc bound from Abramowitz). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") two times.
  + –

    Equalities factor out 2​x2𝑥\sqrt{2}\sqrt{x} and reformulate.
  + –

    Second inequality part 1: we applied

    |  |  |  |  |
    | --- | --- | --- | --- |
    |  | 0<2​y⟹(2​x+y)2+4​x+1<(2​x+y)2+2​(2​x+y)+1=(2​x+y+1)2.02𝑦⟹superscript2𝑥𝑦24𝑥1superscript2𝑥𝑦222𝑥𝑦1superscript2𝑥𝑦12\displaystyle 0<2y\Longrightarrow(2x+y)^{2}+4x+1<(2x+y)^{2}+2(2x+y)+1=(2x+y+1)^{2}\ . |  | (237) |
  + –

    Second inequality part 2: we show that for a=120​(2048+169​ππ−13)𝑎1202048169𝜋𝜋13a=\frac{1}{20}\left(\sqrt{\frac{2048+169\pi}{\pi}}-13\right) following holds:
    8​xπ−(a2+2​a​(x+y))⩾08𝑥𝜋superscript𝑎22𝑎𝑥𝑦0\frac{8x}{\pi}-\left(a^{2}+2a(x+y)\right)\geqslant 0.
    We have ∂∂x​8​xπ−(a2+2​a​(x+y))=8π−2​a>0𝑥8𝑥𝜋superscript𝑎22𝑎𝑥𝑦8𝜋2𝑎0\frac{\partial}{\partial x}\frac{8x}{\pi}-\left(a^{2}+2a(x+y)\right)=\frac{8}{\pi}-2a>0 and
    ∂∂y​8​xπ−(a2+2​a​(x+y))=−2​a<0𝑦8𝑥𝜋superscript𝑎22𝑎𝑥𝑦2𝑎0\frac{\partial}{\partial y}\frac{8x}{\pi}-\left(a^{2}+2a(x+y)\right)=-2a<0.
    Therefore the minimum is at border for minimal x𝑥x and maximal y𝑦y:

    |  |  |  |  |
    | --- | --- | --- | --- |
    |  | 8⋅0.64π−(220​(2048+169​ππ−13)​(0.64+0.01)+(120​(2048+169​ππ−13))2)= 0.⋅80.64𝜋2202048169𝜋𝜋130.640.01superscript1202048169𝜋𝜋132 0\displaystyle\frac{8\cdot 0.64}{\pi}-\left(\frac{2}{20}\left(\sqrt{\frac{2048+169\pi}{\pi}}-13\right)(0.64+0.01)+\left(\frac{1}{20}\left(\sqrt{\frac{2048+169\pi}{\pi}}-13\right)\right)^{2}\right)\ =\ 0\ . |  | (238) |

    Thus

    |  |  |  |  |
    | --- | --- | --- | --- |
    |  | 8​xπ⩾a2+2​a​(x+y).8𝑥𝜋superscript𝑎22𝑎𝑥𝑦\displaystyle\frac{8x}{\pi}\ \geqslant\ a^{2}+2a(x+y)\ . |  | (239) |

    for a=120​(2048+169​ππ−13)>0.782𝑎1202048169𝜋𝜋130.782a=\frac{1}{20}\left(\sqrt{\frac{2048+169\pi}{\pi}}-13\right)>0.782.
  + –

    Equalities only solve square root and factor out the resulting
    terms (2​(2​x+y)+1)22𝑥𝑦1(2(2x+y)+1) and (2​(x+y)+0.782)2𝑥𝑦0.782(2(x+y)+0.782).

  We know that (2−erfc(x)>0(2-\operatorname{erfc}(x)>0 according to
  Lemma [21](#Thmtheorem21 "Lemma 21 (Basic functions). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").
  For the sub-term we derived

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 2​e(2​x+y2​x)2​erfc⁡(2​x+y2​x)−e(x+y2​x)2​erfc⁡(x+y2​x)> 0.2superscript𝑒superscript2𝑥𝑦2𝑥2erfc2𝑥𝑦2𝑥superscript𝑒superscript𝑥𝑦2𝑥2erfc𝑥𝑦2𝑥 0\displaystyle 2e^{\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)^{2}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)-e^{\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)^{2}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)\ >\ 0\ . |  | (240) |

  Consequently, both terms in the brackets of ∂∂ν​ξ~​(μ,ω,ν,τ,λ,α)𝜈~𝜉𝜇𝜔𝜈𝜏𝜆𝛼\frac{\partial}{\partial\nu}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda,\alpha)
  are larger than zero.
  Therefore ∂∂ν​ξ~​(μ,ω,ν,τ,λ,α)𝜈~𝜉𝜇𝜔𝜈𝜏𝜆𝛼\frac{\partial}{\partial\nu}{\tilde{\xi}}(\mu,\omega,\nu,\tau,\lambda,\alpha)
  is larger than zero.

∎

###### Lemma 41 (Mean at low variance).

The mapping of the mean μ~~𝜇{\tilde{\mu}} (Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")))

|  |  |  |  |
| --- | --- | --- | --- |
|  | μ~(μ,ω,ν,τ,λ,α)=12λ(−(α+μω)erfc(μ​ω2​ν​τ)+\displaystyle{\tilde{\mu}}(\mu,\omega,\nu,\tau,\lambda,\alpha)\ =\frac{1}{2}\lambda\left(-(\alpha+\mu\omega)\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+\right. |  | (241) |
|  |  |  |
| --- | --- | --- |
|  | αeμ​ω+ν​τ2erfc(μ​ω+ν​τ2​ν​τ)+2πν​τe−μ2​ω22​ν​τ+2μω)\displaystyle\left.\alpha e^{\mu\omega+\frac{\nu\tau}{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+\sqrt{\frac{2}{\pi}}\sqrt{\nu\tau}e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}+2\mu\omega\right) |  |

in the domain −0.1⩽μ⩽−0.10.1𝜇0.1-0.1\leqslant\mu\leqslant-0.1, −0.1⩽ω⩽−0.10.1𝜔0.1-0.1\leqslant\omega\leqslant-0.1,
and 0.02⩽ν​τ⩽0.50.02𝜈𝜏0.50.02\leqslant\nu\tau\leqslant 0.5 is bounded by

|  |  |  |  |
| --- | --- | --- | --- |
|  | |μ~​(μ,ω,ν,τ,λ01,α01)|<0.289324~𝜇𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼010.289324\displaystyle|{\tilde{\mu}}(\mu,\omega,\nu,\tau,\lambda\_{\mathrm{01}},\alpha\_{\mathrm{01}})|<0.289324 |  | (242) |

and

|  |  |  |  |
| --- | --- | --- | --- |
|  | limν→0|μ~​(μ,ω,ν,τ,λ01,α01)|=λ​μ​ω.subscript→𝜈0~𝜇𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01𝜆𝜇𝜔\displaystyle\lim\_{\nu\rightarrow 0}|{\tilde{\mu}}(\mu,\omega,\nu,\tau,\lambda\_{\mathrm{01}},\alpha\_{\mathrm{01}})|=\lambda\mu\omega. |  | (243) |

We can consider μ~~𝜇{\tilde{\mu}} with given μ​ω𝜇𝜔\mu\omega as a function in x=ν​τ𝑥𝜈𝜏x=\nu\tau. We show the graph of this function at the
maximal μ​ω=0.01𝜇𝜔0.01\mu\omega=0.01 in the interval x∈[0,1]𝑥01x\in[0,1] in Figure [A6](#S3.F6 "Figure A6 ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").

![Refer to caption](/html/1706.02515/assets/x9.png)


Figure A6: The graph of function μ~~𝜇{\tilde{\mu}} for low variances x=ν​τ𝑥𝜈𝜏x=\nu\tau for μ​ω=0.01𝜇𝜔0.01\mu\omega=0.01, where x∈[0,3]𝑥03x\in[0,3], is
displayed in yellow.
Lower and upper bounds based on the Abramowitz bounds (Lemma [22](#Thmtheorem22 "Lemma 22 (Erfc bound from Abramowitz). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) are displayed in green and blue, respectively.

###### Proof.

Since μ~~𝜇{\tilde{\mu}} is strictly monotonically increasing with μ​ω𝜇𝜔\mu\omega

|  |  |  |  |
| --- | --- | --- | --- |
|  | μ~​(μ,ω,ν,τ,λ,α)⩽~𝜇𝜇𝜔𝜈𝜏𝜆𝛼absent\displaystyle{\tilde{\mu}}(\mu,\omega,\nu,\tau,\lambda,\alpha)\leqslant |  | (244) |
|  |  |  |
| --- | --- | --- |
|  | μ~​(0.1,0.1,ν,τ,λ,α)⩽~𝜇0.10.1𝜈𝜏𝜆𝛼absent\displaystyle{\tilde{\mu}}(0.1,0.1,\nu,\tau,\lambda,\alpha)\leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | 12​λ​(−(α+0.01)​erfc⁡(0.012​ν​τ)+α​e0.01+ν​τ2​erfc⁡(0.01+ν​τ2​ν​τ)+2π​ν​τ​e−0.0122​ν​τ+2⋅0.01)⩽12𝜆𝛼0.01erfc0.012𝜈𝜏𝛼superscript𝑒0.01𝜈𝜏2erfc0.01𝜈𝜏2𝜈𝜏2𝜋𝜈𝜏superscript𝑒superscript0.0122𝜈𝜏⋅20.01absent\displaystyle\frac{1}{2}\lambda\left(-(\alpha+0.01)\operatorname{erfc}\left(\frac{0.01}{\sqrt{2}\sqrt{\nu\tau}}\right)+\alpha e^{0.01+\frac{\nu\tau}{2}}\operatorname{erfc}\left(\frac{0.01+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)+\sqrt{\frac{2}{\pi}}\sqrt{\nu\tau}e^{-\frac{0.01^{2}}{2\nu\tau}}+2\cdot 0.01\right)\leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | 12​λ01​(e0.052+0.01​α01​erfc⁡(0.02+0.012​0.02)−(α01+0.01)​erfc⁡(0.012​0.02)+e−0.0122⋅0.5​0.5​2π+0.01⋅2)12subscript𝜆01superscript𝑒0.0520.01subscript𝛼01erfc0.020.0120.02subscript𝛼010.01erfc0.0120.02superscript𝑒superscript0.012⋅20.50.52𝜋⋅0.012\displaystyle\frac{1}{2}\lambda\_{\mathrm{01}}\left(e^{\frac{0.05}{2}+0.01}\alpha\_{\mathrm{01}}\operatorname{erfc}\left(\frac{0.02+0.01}{\sqrt{2}\sqrt{0.02}}\right)-(\alpha\_{\mathrm{01}}+0.01)\operatorname{erfc}\left(\frac{0.01}{\sqrt{2}\sqrt{0.02}}\right)+e^{-\frac{0.01^{2}}{2\cdot 0.5}}\sqrt{0.5}\sqrt{\frac{2}{\pi}}+0.01\cdot 2\right) |  |
|  |  |  |
| --- | --- | --- |
|  | <0.21857,absent0.21857\displaystyle<0.21857, |  |

where we have used the monotonicity of the terms in ν​τ𝜈𝜏\nu\tau.

Similarly, we can use the monotonicity of the terms in ν​τ𝜈𝜏\nu\tau to show that

|  |  |  |  |
| --- | --- | --- | --- |
|  | μ~​(μ,ω,ν,τ,λ,α)⩾μ~​(0.1,−0.1,ν,τ,λ,α)>−0.289324,~𝜇𝜇𝜔𝜈𝜏𝜆𝛼~𝜇0.10.1𝜈𝜏𝜆𝛼0.289324\displaystyle{\tilde{\mu}}(\mu,\omega,\nu,\tau,\lambda,\alpha)\geqslant{\tilde{\mu}}(0.1,-0.1,\nu,\tau,\lambda,\alpha)>-0.289324, |  | (245) |

such that |μ~|<0.289324~𝜇0.289324\left|{\tilde{\mu}}\right|<0.289324 at low variances.

Furthermore, when (ν​τ)→0→𝜈𝜏0(\nu\tau)\rightarrow 0, the terms with the arguments of the complementary error functions erfcerfc\operatorname{erfc} and the exponential function
go to infinity, therefore these three terms converge to zero. Hence, the remaining terms are only 2​μ​ω​12​λ2𝜇𝜔12𝜆2\mu\omega\frac{1}{2}\lambda.
∎

###### Lemma 42 (Bounds on derivatives of μ~~𝜇{\tilde{\mu}} in Ω−superscriptΩ\Omega^{-}).

The derivatives of the function μ~(μ,ω,ν,τ,λ01,α01{\tilde{\mu}}(\mu,\omega,\nu,\tau,\lambda\_{\mathrm{0}1},\alpha\_{\mathrm{0}1}
(Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")))
with respect to μ,ω,ν,τ

𝜇𝜔𝜈𝜏\mu,\omega,\nu,\tau in the domain
Ω−={μ,ω,ν,τ|−0.1⩽μ⩽0.1,−0.1⩽ω⩽0.1,0.05⩽ν⩽0.24,0.8⩽τ⩽1.25}superscriptΩconditional-set

𝜇𝜔𝜈𝜏
formulae-sequence0.1𝜇0.10.1𝜔0.10.05𝜈0.240.8𝜏1.25\Omega^{-}=\{\mu,\omega,\nu,\tau\ |\ -0.1\leqslant\mu\leqslant 0.1,-0.1\leqslant\omega\leqslant 0.1,0.05\leqslant\nu\leqslant 0.24,0.8\leqslant\tau\leqslant 1.25\}
can be bounded as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂∂μ​μ~|<0.14𝜇~𝜇0.14\displaystyle\left|\frac{\partial}{\partial\mu}{\tilde{\mu}}\right|<0.14 |  | (246) |
|  |  |  |
| --- | --- | --- |
|  | |∂∂ω​μ~|<0.14𝜔~𝜇0.14\displaystyle\left|\frac{\partial}{\partial\omega}{\tilde{\mu}}\right|<0.14 |  |
|  |  |  |
| --- | --- | --- |
|  | |∂∂ν​μ~|<0.52𝜈~𝜇0.52\displaystyle\left|\frac{\partial}{\partial\nu}{\tilde{\mu}}\right|<0.52 |  |
|  |  |  |
| --- | --- | --- |
|  | |∂∂τ​μ~|<0.11.𝜏~𝜇0.11\displaystyle\left|\frac{\partial}{\partial\tau}{\tilde{\mu}}\right|<0.11. |  |

###### Proof.

The expression

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂∂μ​μ~=J11=12​λ​ω​e−(μ​ω)22​ν​τ​(2​e(μ​ω)22​ν​τ−e(μ​ω)22​ν​τ​erfc⁡(μ​ω2​ν​τ)+α​e(μ​ω+ν​τ)22​ν​τ​erfc⁡(μ​ω+ν​τ2​ν​τ))𝜇~𝜇subscript𝐽1112𝜆𝜔superscript𝑒superscript𝜇𝜔22𝜈𝜏2superscript𝑒superscript𝜇𝜔22𝜈𝜏superscript𝑒superscript𝜇𝜔22𝜈𝜏erfc𝜇𝜔2𝜈𝜏𝛼superscript𝑒superscript𝜇𝜔𝜈𝜏22𝜈𝜏erfc𝜇𝜔𝜈𝜏2𝜈𝜏\displaystyle\frac{\partial}{\partial\mu}{\tilde{\mu}}=J\_{11}=\frac{1}{2}\lambda\omega e^{\frac{-(\mu\omega)^{2}}{2\nu\tau}}\left(2e^{\frac{(\mu\omega)^{2}}{2\nu\tau}}-e^{\frac{(\mu\omega)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)+\alpha e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\right) |  | (247) |

contains the terms e(μ​ω)22​ν​τ​erfc⁡(μ​ω2​ν​τ)superscript𝑒superscript𝜇𝜔22𝜈𝜏erfc𝜇𝜔2𝜈𝜏e^{\frac{(\mu\omega)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega}{\sqrt{2}\sqrt{\nu\tau}}\right)
and e(μ​ω+ν​τ)22​ν​τ​erfc⁡(μ​ω+ν​τ2​ν​τ)superscript𝑒superscript𝜇𝜔𝜈𝜏22𝜈𝜏erfc𝜇𝜔𝜈𝜏2𝜈𝜏e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)
which are monotonically decreasing in their arguments (Lemma [23](#Thmtheorem23 "Lemma 23 (Function 𝑒^𝑥²⁢erfc(𝑥)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")). We can therefore obtain their
minima and maximal at the minimal and maximal arguments. Since the first term has a negative sign in the expression, both terms
reach their maximal value at μ​ω=−0.01𝜇𝜔0.01\mu\omega=-0.01, ν=0.05𝜈0.05\nu=0.05, and τ=0.8𝜏0.8\tau=0.8.

|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂∂μ​μ~|⩽12​|λ​ω|​|(2−e0.03535532​erfc⁡(0.0353553)+α​e0.1060662​erfc⁡(0.106066))|<0.133𝜇~𝜇12𝜆𝜔2superscript𝑒superscript0.03535532erfc0.0353553𝛼superscript𝑒superscript0.1060662erfc0.1060660.133\displaystyle\left|\frac{\partial}{\partial\mu}{\tilde{\mu}}\right|\leqslant\frac{1}{2}\left|\lambda\omega\right|\left|\left(2-e^{0.0353553^{2}}\operatorname{erfc}\left(0.0353553\right)+\alpha e^{0.106066^{2}}\operatorname{erfc}\left(0.106066\right)\right)\right|<0.133 |  | (248) |

Since, μ~~𝜇{\tilde{\mu}} is symmetric in μ𝜇\mu and ω𝜔\omega, these bounds also hold for the derivate to ω𝜔\omega.

We use the argumentation that the term with the error function is monotonically decreasing (Lemma [23](#Thmtheorem23 "Lemma 23 (Function 𝑒^𝑥²⁢erfc(𝑥)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"))
again for the expression

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂∂ν​μ~=J12=𝜈~𝜇subscript𝐽12absent\displaystyle\frac{\partial}{\partial\nu}{\tilde{\mu}}=J\_{12}= |  | (249) |
|  |  |  |
| --- | --- | --- |
|  | =14​λ​τ​e−μ2​ω22​ν​τ​(α​e(μ​ω+ν​τ)22​ν​τ​erfc⁡(μ​ω+ν​τ2​ν​τ)−(α−1)​2π​ν​τ)⩽absent14𝜆𝜏superscript𝑒superscript𝜇2superscript𝜔22𝜈𝜏𝛼superscript𝑒superscript𝜇𝜔𝜈𝜏22𝜈𝜏erfc𝜇𝜔𝜈𝜏2𝜈𝜏𝛼12𝜋𝜈𝜏absent\displaystyle=\frac{1}{4}\lambda\tau e^{-\frac{\mu^{2}\omega^{2}}{2\nu\tau}}\left(\alpha e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-(\alpha-1)\sqrt{\frac{2}{\pi\nu\tau}}\ \right)\leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | |14​λ​τ|​(|1.1072−2.68593|)<0.52.14𝜆𝜏1.10722.685930.52\displaystyle\left|\frac{1}{4}\lambda\tau\right|\left(\left|1.1072-2.68593\right|\right)<0.52. |  |

We have used that the term 1.1072⩽α01​e(μ​ω+ν​τ)22​ν​τ​erfc⁡(μ​ω+ν​τ2​ν​τ)⩽1.490421.1072subscript𝛼01superscript𝑒superscript𝜇𝜔𝜈𝜏22𝜈𝜏erfc𝜇𝜔𝜈𝜏2𝜈𝜏1.490421.1072\leqslant\alpha\_{\mathrm{01}}e^{\frac{(\mu\omega+\nu\tau)^{2}}{2\nu\tau}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\leqslant 1.49042
and the term 0.942286⩽(α−1)​2π​ν​τ⩽2.685930.942286𝛼12𝜋𝜈𝜏2.685930.942286\leqslant(\alpha-1)\sqrt{\frac{2}{\pi\nu\tau}}\ \leqslant 2.68593.
Since μ~~𝜇{\tilde{\mu}} is symmetric in ν𝜈\nu and τ𝜏\tau, we only have to chance outermost
term |14​λ​τ|14𝜆𝜏\left|\frac{1}{4}\lambda\tau\right| to |14​λ​ν|14𝜆𝜈\left|\frac{1}{4}\lambda\nu\right| to
obtain the estimate |∂∂τ​μ~|<0.11𝜏~𝜇0.11\left|\frac{\partial}{\partial\tau}{\tilde{\mu}}\right|<0.11.

∎

###### Lemma 43 (Tight bound on μ~2superscript~𝜇2{{\tilde{\mu}}}^{2} in Ω−superscriptΩ\Omega^{-}).

The function μ~2​(μ,ω,ν,τ,λ01,α01)superscript~𝜇2𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01{{\tilde{\mu}}}^{2}(\mu,\omega,\nu,\tau,\lambda\_{\mathrm{0}1},\alpha\_{\mathrm{0}1})
(Eq. ([4](#Sx2.E4 "In Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")))
is bounded by

|  |  |  |  |
| --- | --- | --- | --- |
|  | |μ~2|<0.005superscript~𝜇20.005\displaystyle\left|{{\tilde{\mu}}}^{2}\right|<0.005 |  | (250) |

in the domain
Ω−={μ,ω,ν,τ|−0.1⩽μ⩽0.1,−0.1⩽ω⩽0.1,0.05⩽ν⩽0.24,0.8⩽τ⩽1.25}superscriptΩconditional-set

𝜇𝜔𝜈𝜏
formulae-sequence0.1𝜇0.10.1𝜔0.10.05𝜈0.240.8𝜏1.25\Omega^{-}=\{\mu,\omega,\nu,\tau\ |\ -0.1\leqslant\mu\leqslant 0.1,-0.1\leqslant\omega\leqslant 0.1,0.05\leqslant\nu\leqslant 0.24,0.8\leqslant\tau\leqslant 1.25\}.

We visualize the function μ~2superscript~𝜇2{{\tilde{\mu}}}^{2} at its maximal μ​ν=−0.01𝜇𝜈0.01\mu\nu=-0.01 and for x=ν​τ𝑥𝜈𝜏x=\nu\tau in the form
h​(x)=μ~2​(0.1,−0.1,x,1,λ01,α01)ℎ𝑥superscript~𝜇20.10.1𝑥1subscript𝜆01subscript𝛼01h(x)={{\tilde{\mu}}}^{2}(0.1,-0.1,x,1,\lambda\_{\mathrm{0}1},\alpha\_{\mathrm{0}1}) in Figure [A7](#S3.F7 "Figure A7 ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").

![Refer to caption](/html/1706.02515/assets/x10.png)


Figure A7: The graph of the function h​(x)=μ~2​(0.1,−0.1,x,1,λ01,α01)ℎ𝑥superscript~𝜇20.10.1𝑥1subscript𝜆01subscript𝛼01h(x)={{\tilde{\mu}}}^{2}(0.1,-0.1,x,1,\lambda\_{\mathrm{0}1},\alpha\_{\mathrm{0}1}) is displayed. It has a local
maximum at x=ν​τ≈0.187342𝑥𝜈𝜏0.187342x=\nu\tau\approx 0.187342 and h​(x)≈0.00451457ℎ𝑥0.00451457h(x)\approx 0.00451457 in the domain x∈[0,1]𝑥01x\in[0,1].

###### Proof.

We use a similar strategy to the one we have used to show the bound on the singular value (Lemmata [10](#Thmtheorem10 "Lemma 10 (Upper Bounds on Absolute Derivatives of Largest Singular Value). ‣ Upper Bounds on the Largest Singular Value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), [11](#Thmtheorem11 "Lemma 11 (Mean Value Theorem Bound on Deviation from Largest Singular Value). ‣ Upper Bounds on the Largest Singular Value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), and [12](#Thmtheorem12 "Lemma 12 (Largest Singular Value Smaller Than One). ‣ Upper Bounds on the Largest Singular Value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")), where
we evaluted the function on a grid and used bounds on the derivatives together with the mean value theorem.
Here we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | |μ~2​(μ,ω,ν,τ,λ01,α01)−μ~2​(μ+Δ​μ,ω+Δ​ω,ν+Δ​ν,τ+Δ​τ,λ01,α01)|⩽superscript~𝜇2𝜇𝜔𝜈𝜏subscript𝜆01subscript𝛼01superscript~𝜇2𝜇Δ𝜇𝜔Δ𝜔𝜈Δ𝜈𝜏Δ𝜏subscript𝜆01subscript𝛼01absent\displaystyle\left|{{\tilde{\mu}}}^{2}(\mu,\omega,\nu,\tau,\lambda\_{\mathrm{0}1},\alpha\_{\mathrm{0}1})-{{\tilde{\mu}}}^{2}(\mu+\Delta\mu,\omega+\Delta\omega,\nu+\Delta\nu,\tau+\Delta\tau,\lambda\_{\mathrm{0}1},\alpha\_{\mathrm{0}1})\right|\leqslant |  | (252) |
|  |  |  |
| --- | --- | --- |
|  | |∂∂μ​μ~2|​|Δ​μ|+|∂∂ω​μ~2|​|Δ​ω|+|∂∂ν​μ~2|​|Δ​ν|+|∂∂τ​μ~2|​|Δ​τ|.𝜇superscript~𝜇2Δ𝜇𝜔superscript~𝜇2Δ𝜔𝜈superscript~𝜇2Δ𝜈𝜏superscript~𝜇2Δ𝜏\displaystyle\left|\frac{\partial}{\partial\mu}{{\tilde{\mu}}}^{2}\right||\Delta\mu|+\left|\frac{\partial}{\partial\omega}{{\tilde{\mu}}}^{2}\right||\Delta\omega|+\left|\frac{\partial}{\partial\nu}{{\tilde{\mu}}}^{2}\right||\Delta\nu|+\left|\frac{\partial}{\partial\tau}{{\tilde{\mu}}}^{2}\right||\Delta\tau|. |  |

We use Lemma [42](#Thmtheorem42 "Lemma 42 (Bounds on derivatives of 𝜇̃ in Ω⁻). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") and Lemma [41](#Thmtheorem41 "Lemma 41 (Mean at low variance). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), to obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂∂μ​μ~2|=2​|μ~|​|∂∂μ​μ~|<2⋅0.289324⋅0.14=0.08101072𝜇superscript~𝜇22~𝜇𝜇~𝜇⋅20.2893240.140.08101072\displaystyle\left|\frac{\partial}{\partial\mu}{{\tilde{\mu}}}^{2}\right|=2\left|{\tilde{\mu}}\right|\left|\frac{\partial}{\partial\mu}{\tilde{\mu}}\right|<2\cdot 0.289324\cdot 0.14=0.08101072 |  | (253) |
|  |  |  |
| --- | --- | --- |
|  | |∂∂ω​μ~2|=2​|μ~|​|∂∂ω​μ~|<2⋅0.289324⋅0.14=0.08101072𝜔superscript~𝜇22~𝜇𝜔~𝜇⋅20.2893240.140.08101072\displaystyle\left|\frac{\partial}{\partial\omega}{{\tilde{\mu}}}^{2}\right|=2\left|{\tilde{\mu}}\right|\left|\frac{\partial}{\partial\omega}{\tilde{\mu}}\right|<2\cdot 0.289324\cdot 0.14=0.08101072 |  |
|  |  |  |
| --- | --- | --- |
|  | |∂∂ν​μ~2|=2​|μ~|​|∂∂ν​μ~|<2⋅0.289324⋅0.52=0.30089696𝜈superscript~𝜇22~𝜇𝜈~𝜇⋅20.2893240.520.30089696\displaystyle\left|\frac{\partial}{\partial\nu}{{\tilde{\mu}}}^{2}\right|=2\left|{\tilde{\mu}}\right|\left|\frac{\partial}{\partial\nu}{\tilde{\mu}}\right|<2\cdot 0.289324\cdot 0.52=0.30089696 |  |
|  |  |  |
| --- | --- | --- |
|  | |∂∂τ​μ~2|=2​|μ~|​|∂∂τ​μ~|<2⋅0.289324⋅0.11=0.06365128𝜏superscript~𝜇22~𝜇𝜏~𝜇⋅20.2893240.110.06365128\displaystyle\left|\frac{\partial}{\partial\tau}{{\tilde{\mu}}}^{2}\right|=2\left|{\tilde{\mu}}\right|\left|\frac{\partial}{\partial\tau}{\tilde{\mu}}\right|<2\cdot 0.289324\cdot 0.11=0.06365128 |  |

We evaluated the function μ~2superscript~𝜇2{{\tilde{\mu}}}^{2} in a grid G𝐺G of Ω−superscriptΩ\Omega^{-} with Δ​μ=0.001498041Δ𝜇0.001498041\Delta\mu=0.001498041,
Δ​ω=0.001498041Δ𝜔0.001498041\Delta\omega=0.001498041,
Δ​ν=0.0004033190Δ𝜈0.0004033190\Delta\nu=0.0004033190, and
Δ​τ=0.0019065994Δ𝜏0.0019065994\Delta\tau=0.0019065994 using a computer and obtained the maximal value maxG(μ~)2=0.00451457\max\_{G}({\tilde{\mu}})^{2}=0.00451457, therefore
the maximal value of μ~2superscript~𝜇2{{\tilde{\mu}}}^{2} is bounded by

|  |  |  |  |
| --- | --- | --- | --- |
|  | max(μ,ω,ν,τ)∈Ω−(μ~)2⩽\displaystyle\max\_{(\mu,\omega,\nu,\tau)\in\Omega^{-}}({\tilde{\mu}})^{2}\leqslant |  | (254) |
|  |  |  |
| --- | --- | --- |
|  | 0.00451457+0.001498041⋅0.08101072+0.001498041⋅0.08101072+0.00451457⋅0.0014980410.08101072limit-from⋅0.0014980410.08101072\displaystyle 0.00451457+0.001498041\cdot 0.08101072+0.001498041\cdot 0.08101072+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 0.0004033190⋅0.30089696+0.0019065994⋅0.06365128<0.005.⋅0.00040331900.30089696⋅0.00190659940.063651280.005\displaystyle 0.0004033190\cdot 0.30089696+0.0019065994\cdot 0.06365128<0.005. |  | (255) |

Furthermore we used error propagation to estimate the numerical error on the function evaluation. Using the error propagation rules
derived in Subsection [A3.4.5](#S3.SS4.SSS5 "A3.4.5 Computer-assisted proof details for main Lemma 12 in Section A3.4.1. ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), we found that the numerical error is smaller than 10−13superscript101310^{-13} in the worst case.
∎

###### Lemma 44 (Main subfunction).

For 1.2⩽x⩽201.2𝑥201.2\leqslant x\leqslant 20 and −0.1⩽y⩽0.10.1𝑦0.1-0.1\leqslant y\leqslant 0.1,

the function

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(x+y)22​x​erfc⁡(x+y2​x)−2​e(2​x+y)22​x​erfc⁡(2​x+y2​x)superscript𝑒superscript𝑥𝑦22𝑥erfc𝑥𝑦2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥erfc2𝑥𝑦2𝑥\displaystyle e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)-2e^{\frac{(2x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right) |  | (256) |

is smaller than zero, is strictly monotonically increasing in x𝑥x,
and strictly monotonically decreasing in y𝑦y for the minimal x=12/10=1.2𝑥12101.2x=12/10=1.2.

###### Proof.

We first consider the derivative of sub-function
Eq. ([115](#S3.E115 "In Lemma 15 (Main subfunction). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) with respect to x𝑥x.
The derivative of the function

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(x+y)22​x​erfc⁡(x+y2​x)−2​e(2​x+y)22​x​erfc⁡(2​x+y2​x)superscript𝑒superscript𝑥𝑦22𝑥erfc𝑥𝑦2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥erfc2𝑥𝑦2𝑥\displaystyle e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)-2e^{\frac{(2x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right) |  | (257) |

with respect to x𝑥x is

|  |  |  |  |
| --- | --- | --- | --- |
|  | π​(e(x+y)22​x​(x−y)​(x+y)​erfc⁡(x+y2​x)−2​e(2​x+y)22​x​(4​x2−y2)​erfc⁡(2​x+y2​x))+2​x​(3​x−y)2​π​x2=𝜋superscript𝑒superscript𝑥𝑦22𝑥𝑥𝑦𝑥𝑦erfc𝑥𝑦2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥4superscript𝑥2superscript𝑦2erfc2𝑥𝑦2𝑥2𝑥3𝑥𝑦2𝜋superscript𝑥2absent\displaystyle\frac{\sqrt{\pi}\left(e^{\frac{(x+y)^{2}}{2x}}(x-y)(x+y)\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)-2e^{\frac{(2x+y)^{2}}{2x}}\left(4x^{2}-y^{2}\right)\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)\right)+\sqrt{2}\sqrt{x}(3x-y)}{2\sqrt{\pi}x^{2}}\ = |  | (258) |
|  |  |  |
| --- | --- | --- |
|  | π​(e(x+y)22​x​(x−y)​(x+y)​erfc⁡(x+y2​x)−2​e(2​x+y)22​x​(2​x+y)​(2​x−y)​erfc⁡(2​x+y2​x))+2​x​(3​x−y)2​π​x2=𝜋superscript𝑒superscript𝑥𝑦22𝑥𝑥𝑦𝑥𝑦erfc𝑥𝑦2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥2𝑥𝑦2𝑥𝑦erfc2𝑥𝑦2𝑥2𝑥3𝑥𝑦2𝜋superscript𝑥2absent\displaystyle\frac{\sqrt{\pi}\left(e^{\frac{(x+y)^{2}}{2x}}(x-y)(x+y)\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)-2e^{\frac{(2x+y)^{2}}{2x}}(2x+y)(2x-y)\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)\right)+\sqrt{2}\sqrt{x}(3x-y)}{2\sqrt{\pi}x^{2}}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | π​(e(x+y)22​x​(x−y)​(x+y)​erfc⁡(x+y2​x)2​x−2​e(2​x+y)22​x​(2​x+y)​(2​x−y)​erfc⁡(2​x+y2​x)2​x)+(3​x−y)2​2​π​x2​x.𝜋superscript𝑒superscript𝑥𝑦22𝑥𝑥𝑦𝑥𝑦erfc𝑥𝑦2𝑥2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥2𝑥𝑦2𝑥𝑦erfc2𝑥𝑦2𝑥2𝑥3𝑥𝑦22𝜋superscript𝑥2𝑥\displaystyle\frac{\sqrt{\pi}\left(\frac{e^{\frac{(x+y)^{2}}{2x}}(x-y)(x+y)\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)}{\sqrt{2}\sqrt{x}}-\frac{2e^{\frac{(2x+y)^{2}}{2x}}(2x+y)(2x-y)\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)}{\sqrt{2}\sqrt{x}}\right)+(3x-y)}{2\sqrt{2}\sqrt{\pi}x^{2}\sqrt{x}}\ . |  |

We consider the numerator

|  |  |  |  |
| --- | --- | --- | --- |
|  | π​(e(x+y)22​x​(x−y)​(x+y)​erfc⁡(x+y2​x)2​x−2​e(2​x+y)22​x​(2​x+y)​(2​x−y)​erfc⁡(2​x+y2​x)2​x)+(3​x−y).𝜋superscript𝑒superscript𝑥𝑦22𝑥𝑥𝑦𝑥𝑦erfc𝑥𝑦2𝑥2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥2𝑥𝑦2𝑥𝑦erfc2𝑥𝑦2𝑥2𝑥3𝑥𝑦\displaystyle\sqrt{\pi}\left(\frac{e^{\frac{(x+y)^{2}}{2x}}(x-y)(x+y)\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)}{\sqrt{2}\sqrt{x}}-\frac{2e^{\frac{(2x+y)^{2}}{2x}}(2x+y)(2x-y)\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)}{\sqrt{2}\sqrt{x}}\right)+(3x-y)\ . |  | (259) |

For bounding this value, we use the approximation

|  |  |  |  |
| --- | --- | --- | --- |
|  | ez2​erfc⁡(z)≈2.911π​(2.911−1)​z+π​z2+2.9112.superscript𝑒superscript𝑧2erfc𝑧2.911𝜋2.9111𝑧𝜋superscript𝑧2superscript2.9112\displaystyle e^{z^{2}}\operatorname{erfc}(z)\ \approx\ \frac{2.911}{\sqrt{\pi}(2.911-1)z+\sqrt{\pi z^{2}+2.911^{2}}}\ . |  | (260) |

from Ren and MacKenzie, [[30](#bib.bib30)].
We start with an error analysis of this approximation.
According to Ren and MacKenzie, [[30](#bib.bib30)] (Figure 1), the approximation
error is positive in the range
[0.7,3.2]0.73.2[0.7,3.2]. This range contains all possible
arguments of erfcerfc\operatorname{erfc} that we consider.
Numerically we maximized and minimized the approximation error of the
whole expression

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | E​(x,y)𝐸𝑥𝑦\displaystyle E(x,y)\ | =(e(x+y)22​x​(x−y)​(x+y)​erfc⁡(x+y2​x)2​x−2​e(2​x+y)22​x​(2​x−y)​(2​x+y)​erfc⁡(2​x+y2​x)2​x)−absentlimit-fromsuperscript𝑒superscript𝑥𝑦22𝑥𝑥𝑦𝑥𝑦erfc𝑥𝑦2𝑥2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥2𝑥𝑦2𝑥𝑦erfc2𝑥𝑦2𝑥2𝑥\displaystyle=\ \left(\frac{e^{\frac{(x+y)^{2}}{2x}}(x-y)(x+y)\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)}{\sqrt{2}\sqrt{x}}-\frac{2e^{\frac{(2x+y)^{2}}{2x}}(2x-y)(2x+y)\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)}{\sqrt{2}\sqrt{x}}\right)\ - |  | (261) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (2.911​(x−y)​(x+y)(2​x)​(π​(2.911−1)​(x+y)2​x+π​(x+y2​x)2+2.9112)−\displaystyle\left(\frac{2.911(x-y)(x+y)}{\left(\sqrt{2}\sqrt{x}\right)\left(\frac{\sqrt{\pi}(2.911-1)(x+y)}{\sqrt{2}\sqrt{x}}+\sqrt{\pi\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)^{2}+2.911^{2}}\right)}\ -\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2⋅2.911​(2​x−y)​(2​x+y)(2​x)​(π​(2.911−1)​(2​x+y)2​x+π​(2​x+y2​x)2+2.9112)).\displaystyle\left.\frac{2\cdot 2.911(2x-y)(2x+y)}{\left(\sqrt{2}\sqrt{x}\right)\left(\frac{\sqrt{\pi}(2.911-1)(2x+y)}{\sqrt{2}\sqrt{x}}+\sqrt{\pi\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)^{2}+2.911^{2}}\right)}\right)\ . |  |

We numerically determined 0.0113556⩽E​(x,y)⩽0.01695510.0113556𝐸𝑥𝑦0.01695510.0113556\leqslant E(x,y)\leqslant 0.0169551 for
1.2⩽x⩽201.2𝑥201.2\leqslant x\leqslant 20 and −0.1⩽y⩽0.10.1𝑦0.1-0.1\leqslant y\leqslant 0.1.
We used different numerical optimization techniques like
gradient based constraint BFGS algorithms and
non-gradient-based Nelder-Mead methods with different start points.
Therefore our approximation is smaller than the function that we
approximate.
We subtract an additional safety gap of 0.0131259 from our
approximation to ensure that the inequality via the approximation
holds true. With this safety gap the inequality would hold true even
for negative x𝑥x, where the approximation error becomes negative and
the safety gap would compensate.
Of course, the safety gap of 0.0131259 is not necessary for our
analysis but may help or future investigations.

We have the sequences of inequalities using the approximation of Ren and MacKenzie, [[30](#bib.bib30)]:

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  |  | (3​x−y)+(e(x+y)22​x​(x−y)​(x+y)​erfc⁡(x+y2​x)2​x−2​e(2​x+y)22​x​(2​x−y)​(2​x+y)​erfc⁡(2​x+y2​x)2​x)​π⩾3𝑥𝑦superscript𝑒superscript𝑥𝑦22𝑥𝑥𝑦𝑥𝑦erfc𝑥𝑦2𝑥2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥2𝑥𝑦2𝑥𝑦erfc2𝑥𝑦2𝑥2𝑥𝜋absent\displaystyle(3x-y)+\left(\frac{e^{\frac{(x+y)^{2}}{2x}}(x-y)(x+y)\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)}{\sqrt{2}\sqrt{x}}-\frac{2e^{\frac{(2x+y)^{2}}{2x}}(2x-y)(2x+y)\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)}{\sqrt{2}\sqrt{x}}\right)\sqrt{\pi}\ \geqslant |  | | (262) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | (3x−y)+(2.911​(x−y)​(x+y)(π​(x+y2​x)2+2.9112+(2.911−1)​π​(x+y)2​x)​(2​x)−\displaystyle(3x-y)+\left(\frac{2.911(x-y)(x+y)}{\left(\sqrt{\pi\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)^{2}+2.911^{2}}+\frac{(2.911-1)\sqrt{\pi}(x+y)}{\sqrt{2}\sqrt{x}}\right)\left(\sqrt{2}\sqrt{x}\right)}\ -\right. |  | |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 2​(2​x−y)​(2​x+y)​2.911(2​x)​(π​(2​x+y2​x)2+2.9112+(2.911−1)​π​(2​x+y)2​x))π−0.0131259=\displaystyle\left.\frac{2(2x-y)(2x+y)2.911}{\left(\sqrt{2}\sqrt{x}\right)\left(\sqrt{\pi\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)^{2}+2.911^{2}}+\frac{(2.911-1)\sqrt{\pi}(2x+y)}{\sqrt{2}\sqrt{x}}\right)}\right)\sqrt{\pi}-0.0131259\ = |  | |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | (3x−y)+((2​x​2.911)​(x−y)​(x+y)(π​(x+y)2+2⋅2.9112​x+(2.911−1)​(x+y)​π)​(2​x)−\displaystyle(3x-y)+\left(\frac{\left(\sqrt{2}\sqrt{x}2.911\right)(x-y)(x+y)}{\left(\sqrt{\pi(x+y)^{2}+2\cdot 2.911^{2}x}+(2.911-1)(x+y)\sqrt{\pi}\right)\left(\sqrt{2}\sqrt{x}\right)}\ -\right. |  | |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 2​(2​x−y)​(2​x+y)​(2​x​2.911)(2​x)​(π​(2​x+y)2+2⋅2.9112​x+(2.911−1)​(2​x+y)​π))π−0.0131259=\displaystyle\left.\frac{2(2x-y)(2x+y)\left(\sqrt{2}\sqrt{x}2.911\right)}{\left(\sqrt{2}\sqrt{x}\right)\left(\sqrt{\pi(2x+y)^{2}+2\cdot 2.911^{2}x}+(2.911-1)(2x+y)\sqrt{\pi}\right)}\right)\sqrt{\pi}-0.0131259\ = |  | |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | (3x−y)+2.911((x−y)​(x+y)(2.911−1)​(x+y)+(x+y)2+2⋅2.9112​xπ−\displaystyle(3x-y)+2.911\left(\frac{(x-y)(x+y)}{(2.911-1)(x+y)+\sqrt{(x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}}\ -\right. |  | |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 2​(2​x−y)​(2​x+y)(2.911−1)​(2​x+y)+(2​x+y)2+2⋅2.9112​xπ)−0.0131259⩾\displaystyle\left.\frac{2(2x-y)(2x+y)}{(2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}}\right)-0.0131259\ \geqslant |  | |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | (3x−y)+2.911((x−y)​(x+y)(2.911−1)​(x+y)+(2.9112π)2+(x+y)2+2⋅2.9112​xπ+2⋅2.9112​yπ−\displaystyle(3x-y)+2.911\left(\frac{(x-y)(x+y)}{(2.911-1)(x+y)+\sqrt{\left(\frac{2.911^{2}}{\pi}\right)^{2}+(x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}+\frac{2\cdot 2.911^{2}y}{\pi}}}\ -\right. |  | |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 2​(2​x−y)​(2​x+y)(2.911−1)​(2​x+y)+(2​x+y)2+2⋅2.9112​xπ)−0.0131259=\displaystyle\left.\frac{2(2x-y)(2x+y)}{(2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}}\right)-0.0131259\ = |  | |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | (3x−y)+2.911((x−y)​(x+y)(2.911−1)​(x+y)+(x+y+2.9112π)2−\displaystyle(3x-y)+2.911\left(\frac{(x-y)(x+y)}{(2.911-1)(x+y)+\sqrt{\left(x+y+\frac{2.911^{2}}{\pi}\right)^{2}}}\ -\right. |  | |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 2​(2​x−y)​(2​x+y)(2.911−1)​(2​x+y)+(2​x+y)2+2⋅2.9112​xπ)−0.0131259=\displaystyle\left.\frac{2(2x-y)(2x+y)}{(2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}}\right)-0.0131259\ = |  | |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | (3​x−y)+2.911​((x−y)​(x+y)2.911​(x+y)+2.9112π−2​(2​x−y)​(2​x+y)(2.911−1)​(2​x+y)+(2​x+y)2+2⋅2.9112​xπ)−0.0131259=3𝑥𝑦2.911𝑥𝑦𝑥𝑦2.911𝑥𝑦superscript2.9112𝜋22𝑥𝑦2𝑥𝑦2.91112𝑥𝑦superscript2𝑥𝑦2⋅2superscript2.9112𝑥𝜋0.0131259absent\displaystyle(3x-y)+2.911\left(\frac{(x-y)(x+y)}{2.911(x+y)+\frac{2.911^{2}}{\pi}}-\frac{2(2x-y)(2x+y)}{(2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}}\right)-0.0131259\ = |  | |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | (3​x−y)+(x−y)​(x+y)(x+y)+2.911π−2​(2​x−y)​(2​x+y)​2.911(2.911−1)​(2​x+y)+(2​x+y)2+2⋅2.9112​xπ−0.0131259=3𝑥𝑦𝑥𝑦𝑥𝑦𝑥𝑦2.911𝜋22𝑥𝑦2𝑥𝑦2.9112.91112𝑥𝑦superscript2𝑥𝑦2⋅2superscript2.9112𝑥𝜋0.0131259absent\displaystyle(3x-y)+\frac{(x-y)(x+y)}{(x+y)+\frac{2.911}{\pi}}-\frac{2(2x-y)(2x+y)2.911}{(2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}}-0.0131259\ = | (3​x−y)+(x−y)​(x+y)(x+y)+2.911π−2​(2​x−y)​(2​x+y)​2.911(2.911−1)​(2​x+y)+(2​x+y)2+2⋅2.9112​xπ−0.0131259=3𝑥𝑦𝑥𝑦𝑥𝑦𝑥𝑦2.911𝜋22𝑥𝑦2𝑥𝑦2.9112.91112𝑥𝑦superscript2𝑥𝑦2⋅2superscript2.9112𝑥𝜋0.0131259absent\displaystyle(3x-y)+\frac{(x-y)(x+y)}{(x+y)+\frac{2.911}{\pi}}-\frac{2(2x-y)(2x+y)2.911}{(2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}}-0.0131259\ = |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | (−2(2x−y)2.911((x+y)+2.911π)(2x+y)+\displaystyle\left(-2(2x-y)2.911\left((x+y)+\frac{2.911}{\pi}\right)(2x+y)\right.\ + |  | |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ((x+y)+2.911π)​(3​x−y−0.0131259)​((2.911−1)​(2​x+y)+(2​x+y)2+2⋅2.9112​xπ)+limit-from𝑥𝑦2.911𝜋3𝑥𝑦0.01312592.91112𝑥𝑦superscript2𝑥𝑦2⋅2superscript2.9112𝑥𝜋\displaystyle\left.\left((x+y)+\frac{2.911}{\pi}\right)(3x-y-0.0131259)\left((2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}\right)\right.\ + |  | |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | (x−y)(x+y)((2.911−1)(2x+y)+(2​x+y)2+2⋅2.9112​xπ))\displaystyle\left.(x-y)(x+y)\left((2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}\right)\right) |  | |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | (((x+y)+2.911π)​((2.911−1)​(2​x+y)+(2​x+y)2+2⋅2.9112​xπ))−1=superscript𝑥𝑦2.911𝜋2.91112𝑥𝑦superscript2𝑥𝑦2⋅2superscript2.9112𝑥𝜋1absent\displaystyle\left(\left((x+y)+\frac{2.911}{\pi}\right)\left((2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}\right)\right)^{-1}\ = |  | |
|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  |  | (((x−y)(x+y)+(3x−y−0.0131259)(x+y+0.9266))((2​x+y)2+5.39467​x+3.822x+1.911y)−\displaystyle\left(((x-y)(x+y)+(3x-y-0.0131259)(x+y+0.9266))\left(\sqrt{(2x+y)^{2}+5.39467x}+3.822x+1.911y\right)\right.\ - |  | | (263) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 5.822(2x−y)(x+y+0.9266)(2x+y))\displaystyle\left.5.822(2x-y)(x+y+0.9266)(2x+y)\right) |  | |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | (((x+y)+2.911π)​((2.911−1)​(2​x+y)+(2​x+y)2+22.9112​xπ))−1> 0.superscript𝑥𝑦2.911𝜋2.91112𝑥𝑦superscript2𝑥𝑦2superscript22.9112𝑥𝜋1 0\displaystyle\left(\left((x+y)+\frac{2.911}{\pi}\right)\left((2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{22.911^{2}x}{\pi}}\right)\right)^{-1}\ >\ 0\ . |  | |

We explain this sequence of inequalities:

* •

  First inequality: The approximation of Ren and MacKenzie, [[30](#bib.bib30)]
  and then subtracting a safety gap (which would not be necessary for the
  current analysis).
* •

  Equalities: The factor 2​x2𝑥\sqrt{2}\sqrt{x} is factored out and
  canceled.
* •

  Second inequality: adds a positive term in the first root to
  obtain a binomial form. The term containing the root
  is positive and the root is in the denominator,
  therefore the whole term becomes smaller.

* •

  Equalities: solve for the term and factor out.
* •

  Bringing all terms to the denominator
  ((x+y)+2.911π)​((2.911−1)​(2​x+y)+(2​x+y)2+2⋅2.9112​xπ)𝑥𝑦2.911𝜋2.91112𝑥𝑦superscript2𝑥𝑦2⋅2superscript2.9112𝑥𝜋\left((x+y)+\frac{2.911}{\pi}\right)\left((2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}\right).
* •

  Equalities: Multiplying out and expanding terms.
* •

  Last inequality >0absent0>0 is proofed in the following sequence of
  inequalities.

We look at the numerator of the last expression of
Eq. ([262](#S3.E262 "In Proof. ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")), which we show to be
positive in order to show >0absent0>0 in
Eq. ([262](#S3.E262 "In Proof. ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")). The numerator is

|  |  |  |  |
| --- | --- | --- | --- |
|  | ((x−y)​(x+y)+(3​x−y−0.0131259)​(x+y+0.9266))​((2​x+y)2+5.39467​x+3.822​x+1.911​y)−limit-from𝑥𝑦𝑥𝑦3𝑥𝑦0.0131259𝑥𝑦0.9266superscript2𝑥𝑦25.39467𝑥3.822𝑥1.911𝑦\displaystyle((x-y)(x+y)+(3x-y-0.0131259)(x+y+0.9266))\left(\sqrt{(2x+y)^{2}+5.39467x}+3.822x+1.911y\right)- |  | (264) |
|  |  |  |
| --- | --- | --- |
|  | 5.822​(2​x−y)​(x+y+0.9266)​(2​x+y)=5.8222𝑥𝑦𝑥𝑦0.92662𝑥𝑦absent\displaystyle 5.822(2x-y)(x+y+0.9266)(2x+y)\ = |  |
|  |  |  |
| --- | --- | --- |
|  | −5.822(2x−y)(x+y+0.9266)(2x+y)+(3.822x+1.911y)((x−y)(x+y)+\displaystyle-5.822(2x-y)(x+y+0.9266)(2x+y)+(3.822x+1.911y)((x-y)(x+y)+ |  |
|  |  |  |
| --- | --- | --- |
|  | (3x−y−0.0131259)(x+y+0.9266))+((x−y)(x+y)+\displaystyle(3x-y-0.0131259)(x+y+0.9266))+((x-y)(x+y)+ |  |
|  |  |  |
| --- | --- | --- |
|  | (3x−y−0.0131259)(x+y+0.9266))(2​x+y)2+5.39467​x=\displaystyle(3x-y-0.0131259)(x+y+0.9266))\sqrt{(2x+y)^{2}+5.39467x}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | −8.0​x3+(4​x2+2​x​y+2.76667​x−2​y2−0.939726​y−0.0121625)​(2​x+y)2+5.39467​x−8.0superscript𝑥3limit-from4superscript𝑥22𝑥𝑦2.76667𝑥2superscript𝑦20.939726𝑦0.0121625superscript2𝑥𝑦25.39467𝑥\displaystyle-8.0x^{3}+\left(4x^{2}+2xy+2.76667x-2y^{2}-0.939726y-0.0121625\right)\sqrt{(2x+y)^{2}+5.39467x}- |  |
|  |  |  |
| --- | --- | --- |
|  | 8.0​x2​y−11.0044​x2+2.0​x​y2+1.69548​x​y−0.0464849​x+2.0​y3+3.59885​y2−0.0232425​y=8.0superscript𝑥2𝑦11.0044superscript𝑥22.0𝑥superscript𝑦21.69548𝑥𝑦0.0464849𝑥2.0superscript𝑦33.59885superscript𝑦20.0232425𝑦absent\displaystyle 8.0x^{2}y-11.0044x^{2}+2.0xy^{2}+1.69548xy-0.0464849x+2.0y^{3}+3.59885y^{2}-0.0232425y\ = |  |
|  |  |  |
| --- | --- | --- |
|  | −8.0​x3+(4​x2+2​x​y+2.76667​x−2​y2−0.939726​y−0.0121625)​(2​x+y)2+5.39467​x−8.0superscript𝑥3limit-from4superscript𝑥22𝑥𝑦2.76667𝑥2superscript𝑦20.939726𝑦0.0121625superscript2𝑥𝑦25.39467𝑥\displaystyle-8.0x^{3}+\left(4x^{2}+2xy+2.76667x-2y^{2}-0.939726y-0.0121625\right)\sqrt{(2x+y)^{2}+5.39467x}- |  |
|  |  |  |
| --- | --- | --- |
|  | 8.0​x2​y−11.0044​x2+2.0​x​y2+1.69548​x​y−0.0464849​x+2.0​y3+3.59885​y2−0.0232425​y.8.0superscript𝑥2𝑦11.0044superscript𝑥22.0𝑥superscript𝑦21.69548𝑥𝑦0.0464849𝑥2.0superscript𝑦33.59885superscript𝑦20.0232425𝑦\displaystyle 8.0x^{2}y-11.0044x^{2}+2.0xy^{2}+1.69548xy-0.0464849x+2.0y^{3}+3.59885y^{2}-0.0232425y\ . |  |

The factor in front of the root is positive.
If the term, that does not contain the root, was positive, then the whole expression would be positive and
we would have proofed that the numerator is positive.
Therefore we consider the case that the term, that does not contain the root, is negative.
The term that contains the root must be larger than the other term in absolute values.

|  |  |  |  |
| --- | --- | --- | --- |
|  | −(−8.0x3−8.0x2y−11.0044x2+2.xy2+1.69548xy−0.0464849x+2.y3+3.59885y2−0.0232425y)<\displaystyle-\left(-8.0x^{3}-8.0x^{2}y-11.0044x^{2}+2.xy^{2}+1.69548xy-0.0464849x+2.y^{3}+3.59885y^{2}-0.0232425y\right)\ < |  | (265) |
|  |  |  |
| --- | --- | --- |
|  | (4​x2+2​x​y+2.76667​x−2​y2−0.939726​y−0.0121625)​(2​x+y)2+5.39467​x.4superscript𝑥22𝑥𝑦2.76667𝑥2superscript𝑦20.939726𝑦0.0121625superscript2𝑥𝑦25.39467𝑥\displaystyle\left(4x^{2}+2xy+2.76667x-2y^{2}-0.939726y-0.0121625\right)\sqrt{(2x+y)^{2}+5.39467x}\ . |  |

Therefore the squares of the root term have to be larger
than the square of the other term to show >0absent0>0 in
Eq. ([262](#S3.E262 "In Proof. ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")).
Thus, we have the inequality:

|  |  |  |  |
| --- | --- | --- | --- |
|  | (−8.0x3−8.0x2y−11.0044x2+2.xy2+1.69548xy−0.0464849x+2.y3+3.59885y2−0.0232425y)2<\displaystyle\left(-8.0x^{3}-8.0x^{2}y-11.0044x^{2}+2.xy^{2}+1.69548xy-0.0464849x+2.y^{3}+3.59885y^{2}-0.0232425y\right)^{2}\ < |  | (266) |
|  |  |  |
| --- | --- | --- |
|  | (4​x2+2​x​y+2.76667​x−2​y2−0.939726​y−0.0121625)2​((2​x+y)2+5.39467​x).superscript4superscript𝑥22𝑥𝑦2.76667𝑥2superscript𝑦20.939726𝑦0.01216252superscript2𝑥𝑦25.39467𝑥\displaystyle\left(4x^{2}+2xy+2.76667x-2y^{2}-0.939726y-0.0121625\right)^{2}\left((2x+y)^{2}+5.39467x\right)\ . |  |

This is equivalent to

|  |  |  |  |
| --- | --- | --- | --- |
|  | 0<(4​x2+2​x​y+2.76667​x−2​y2−0.939726​y−0.0121625)2​((2​x+y)2+5.39467​x)−0limit-fromsuperscript4superscript𝑥22𝑥𝑦2.76667𝑥2superscript𝑦20.939726𝑦0.01216252superscript2𝑥𝑦25.39467𝑥\displaystyle 0\ <\ \left(4x^{2}+2xy+2.76667x-2y^{2}-0.939726y-0.0121625\right)^{2}\left((2x+y)^{2}+5.39467x\right)- |  | (267) |
|  |  |  |
| --- | --- | --- |
|  | (−8.0​x3−8.0​x2​y−11.0044​x2+2.0​x​y2+1.69548​x​y−0.0464849​x+2.0​y3+3.59885​y2−0.0232425​y)2=superscript8.0superscript𝑥38.0superscript𝑥2𝑦11.0044superscript𝑥22.0𝑥superscript𝑦21.69548𝑥𝑦0.0464849𝑥2.0superscript𝑦33.59885superscript𝑦20.0232425𝑦2absent\displaystyle\left(-8.0x^{3}-8.0x^{2}y-11.0044x^{2}+2.0xy^{2}+1.69548xy-0.0464849x+2.0y^{3}+3.59885y^{2}-0.0232425y\right)^{2}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | −1.2227​x5+40.1006​x4​y+27.7897​x4+41.0176​x3​y2+64.5799​x3​y+39.4762​x3+10.9422​x2​y3−1.2227superscript𝑥540.1006superscript𝑥4𝑦27.7897superscript𝑥441.0176superscript𝑥3superscript𝑦264.5799superscript𝑥3𝑦39.4762superscript𝑥3limit-from10.9422superscript𝑥2superscript𝑦3\displaystyle-1.2227x^{5}+40.1006x^{4}y+27.7897x^{4}+41.0176x^{3}y^{2}+64.5799x^{3}y+39.4762x^{3}+10.9422x^{2}y^{3}- |  |
|  |  |  |
| --- | --- | --- |
|  | 13.543​x2​y2−28.8455​x2​y−0.364625​x2+0.611352​x​y4+6.83183​x​y3+5.46393​x​y2+13.543superscript𝑥2superscript𝑦228.8455superscript𝑥2𝑦0.364625superscript𝑥20.611352𝑥superscript𝑦46.83183𝑥superscript𝑦3limit-from5.46393𝑥superscript𝑦2\displaystyle 13.543x^{2}y^{2}-28.8455x^{2}y-0.364625x^{2}+0.611352xy^{4}+6.83183xy^{3}+5.46393xy^{2}+ |  |
|  |  |  |
| --- | --- | --- |
|  | 0.121746​x​y+0.000798008​x−10.6365​y5−11.927​y4+0.190151​y3−0.000392287​y2.0.121746𝑥𝑦0.000798008𝑥10.6365superscript𝑦511.927superscript𝑦40.190151superscript𝑦30.000392287superscript𝑦2\displaystyle 0.121746xy+0.000798008x-10.6365y^{5}-11.927y^{4}+0.190151y^{3}-0.000392287y^{2}\ . |  |

We obtain the inequalities:

|  |  |  |  |
| --- | --- | --- | --- |
|  | −1.2227​x5+40.1006​x4​y+27.7897​x4+41.0176​x3​y2+64.5799​x3​y+39.4762​x3+10.9422​x2​y3−1.2227superscript𝑥540.1006superscript𝑥4𝑦27.7897superscript𝑥441.0176superscript𝑥3superscript𝑦264.5799superscript𝑥3𝑦39.4762superscript𝑥3limit-from10.9422superscript𝑥2superscript𝑦3\displaystyle-1.2227x^{5}+40.1006x^{4}y+27.7897x^{4}+41.0176x^{3}y^{2}+64.5799x^{3}y+39.4762x^{3}+10.9422x^{2}y^{3}- |  | (268) |
|  |  |  |
| --- | --- | --- |
|  | 13.543​x2​y2−28.8455​x2​y−0.364625​x2+0.611352​x​y4+6.83183​x​y3+5.46393​x​y2+13.543superscript𝑥2superscript𝑦228.8455superscript𝑥2𝑦0.364625superscript𝑥20.611352𝑥superscript𝑦46.83183𝑥superscript𝑦3limit-from5.46393𝑥superscript𝑦2\displaystyle 13.543x^{2}y^{2}-28.8455x^{2}y-0.364625x^{2}+0.611352xy^{4}+6.83183xy^{3}+5.46393xy^{2}+ |  |
|  |  |  |
| --- | --- | --- |
|  | 0.121746​x​y+0.000798008​x−10.6365​y5−11.927​y4+0.190151​y3−0.000392287​y2=0.121746𝑥𝑦0.000798008𝑥10.6365superscript𝑦511.927superscript𝑦40.190151superscript𝑦30.000392287superscript𝑦2absent\displaystyle 0.121746xy+0.000798008x-10.6365y^{5}-11.927y^{4}+0.190151y^{3}-0.000392287y^{2}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | −1.2227​x5+27.7897​x4+41.0176​x3​y2+39.4762​x3−13.543​x2​y2−0.364625​x2+1.2227superscript𝑥527.7897superscript𝑥441.0176superscript𝑥3superscript𝑦239.4762superscript𝑥313.543superscript𝑥2superscript𝑦2limit-from0.364625superscript𝑥2\displaystyle-1.2227x^{5}+27.7897x^{4}+41.0176x^{3}y^{2}+39.4762x^{3}-13.543x^{2}y^{2}-0.364625x^{2}+ |  |
|  |  |  |
| --- | --- | --- |
|  | y(40.1006x4+64.5799x3+10.9422x2y2−28.8455x2+6.83183xy2+0.121746x−\displaystyle y\left(40.1006x^{4}+64.5799x^{3}+10.9422x^{2}y^{2}-28.8455x^{2}+6.83183xy^{2}+0.121746x\ -\right. |  |
|  |  |  |
| --- | --- | --- |
|  | 10.6365y4+0.190151y2)+0.611352xy4+5.46393xy2+0.000798008x−11.927y4−0.000392287y2>\displaystyle\left.10.6365y^{4}+0.190151y^{2}\right)+0.611352xy^{4}+5.46393xy^{2}+0.000798008x-11.927y^{4}-0.000392287y^{2}\ > |  |
|  |  |  |
| --- | --- | --- |
|  | −1.2227​x5+27.7897​x4+41.0176⋅(0.0)2​x3+39.4762​x3−13.543⋅(0.1)2​x2−0.364625​x2−1.2227superscript𝑥527.7897superscript𝑥4⋅41.0176superscript0.02superscript𝑥339.4762superscript𝑥3⋅13.543superscript0.12superscript𝑥2limit-from0.364625superscript𝑥2\displaystyle-1.2227x^{5}+27.7897x^{4}+41.0176\cdot(0.0)^{2}x^{3}+39.4762x^{3}-13.543\cdot(0.1)^{2}x^{2}-0.364625x^{2}- |  |
|  |  |  |
| --- | --- | --- |
|  | 0.1⋅(40.1006x4+64.5799x3+10.9422⋅(0.1)2x2−28.8455x2+6.83183⋅(0.1)2x+0.121746x+\displaystyle 0.1\cdot\left(40.1006x^{4}+64.5799x^{3}+10.9422\cdot(0.1)^{2}x^{2}-28.8455x^{2}+6.83183\cdot(0.1)^{2}x+0.121746x\ +\right. |  |
|  |  |  |
| --- | --- | --- |
|  | 10.6365⋅(0.1)4+0.190151⋅(0.1)2)+\displaystyle\left.10.6365\cdot(0.1)^{4}+0.190151\cdot(0.1)^{2}\right)+ |  |
|  |  |  |
| --- | --- | --- |
|  | 0.611352⋅(0.0)4​x+5.46393⋅(0.0)2​x+0.000798008​x−11.927⋅(0.1)4−0.000392287⋅(0.1)2=⋅0.611352superscript0.04𝑥⋅5.46393superscript0.02𝑥0.000798008𝑥⋅11.927superscript0.14⋅0.000392287superscript0.12absent\displaystyle 0.611352\cdot(0.0)^{4}x+5.46393\cdot(0.0)^{2}x+0.000798008x-11.927\cdot(0.1)^{4}-0.000392287\cdot(0.1)^{2}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | −1.2227​x5+23.7796​x4+(20+13.0182)​x3+2.37355​x2−0.0182084​x−0.000194074⩾1.2227superscript𝑥523.7796superscript𝑥42013.0182superscript𝑥32.37355superscript𝑥20.0182084𝑥0.000194074absent\displaystyle-1.2227x^{5}+23.7796x^{4}+(20+13.0182)x^{3}+2.37355x^{2}-0.0182084x-0.000194074\ \geqslant |  |
|  |  |  |
| --- | --- | --- |
|  | −1.2227​x5+24.7796​x4+13.0182​x3+2.37355​x2−0.0182084​x−0.000194074>1.2227superscript𝑥524.7796superscript𝑥413.0182superscript𝑥32.37355superscript𝑥20.0182084𝑥0.000194074absent\displaystyle-1.2227x^{5}+24.7796x^{4}+13.0182x^{3}+2.37355x^{2}-0.0182084x-0.000194074\ > |  |
|  |  |  |
| --- | --- | --- |
|  | 13.0182​x3+2.37355​x2−0.0182084​x−0.000194074> 0.13.0182superscript𝑥32.37355superscript𝑥20.0182084𝑥0.000194074 0\displaystyle 13.0182x^{3}+2.37355x^{2}-0.0182084x-0.000194074\ >\ 0\ . |  |

We used 24.7796⋅(20)4−1.2227⋅(20)5=52090.9>0⋅24.7796superscript204⋅1.2227superscript20552090.9024.7796\cdot(20)^{4}-1.2227\cdot(20)^{5}=52090.9>0 and x⩽20𝑥20x\leqslant 20.
We have proofed the last inequality >0absent0>0 of Eq. ([262](#S3.E262 "In Proof. ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")).

Consequently the derivative is always positive independent of y𝑦y,
thus

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(x+y)22​x​erfc⁡(x+y2​x)−2​e(2​x+y)22​x​erfc⁡(2​x+y2​x)superscript𝑒superscript𝑥𝑦22𝑥erfc𝑥𝑦2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥erfc2𝑥𝑦2𝑥\displaystyle e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)-2e^{\frac{(2x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right) |  | (269) |

is strictly monotonically increasing in x𝑥x.

##### The main subfunction is smaller than zero.

Next we show that the
sub-function Eq. ([115](#S3.E115 "In Lemma 15 (Main subfunction). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) is smaller
than zero.
We consider the limit:

|  |  |  |  |
| --- | --- | --- | --- |
|  | limx→∞e(x+y)22​x​erfc⁡(x+y2​x)− 2​e(2​x+y)22​x​erfc⁡(2​x+y2​x)= 0subscript→𝑥superscript𝑒superscript𝑥𝑦22𝑥erfc𝑥𝑦2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥erfc2𝑥𝑦2𝑥 0\displaystyle\lim\_{x\to\infty}e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)\ -\ 2e^{\frac{(2x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)\ =\ 0 |  | (270) |

The limit follows from Lemma [22](#Thmtheorem22 "Lemma 22 (Erfc bound from Abramowitz). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").
Since the function is monotonic increasing in x𝑥x, it has to approach
00 from below. Thus,

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(x+y)22​x​erfc⁡(x+y2​x)−2​e(2​x+y)22​x​erfc⁡(2​x+y2​x)superscript𝑒superscript𝑥𝑦22𝑥erfc𝑥𝑦2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥erfc2𝑥𝑦2𝑥\displaystyle e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)-2e^{\frac{(2x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right) |  | (271) |

is smaller than zero.

##### Behavior of the main subfunction with respect to y𝑦y at minimal x𝑥x.

We now consider the derivative of sub-function
Eq. ([115](#S3.E115 "In Lemma 15 (Main subfunction). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) with respect to y𝑦y.
We proofed that sub-function
Eq. ([115](#S3.E115 "In Lemma 15 (Main subfunction). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) is strictly monotonically increasing
independent of y𝑦y.
In the proof of Theorem [16](#Thmtheorem16 "Theorem 16 (Contraction 𝜈-mapping). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), we need the minimum
of sub-function
Eq. ([115](#S3.E115 "In Lemma 15 (Main subfunction). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")). Therefore we are only interested in the
derivative of sub-function
Eq. ([115](#S3.E115 "In Lemma 15 (Main subfunction). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) with respect to y𝑦y
for the minimum x=12/10=1.2𝑥12101.2x=12/10=1.2

Consequently, we insert the minimum x=12/10=1.2𝑥12101.2x=12/10=1.2 into the sub-function
Eq. ([115](#S3.E115 "In Lemma 15 (Main subfunction). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")). The main terms become

|  |  |  |  |
| --- | --- | --- | --- |
|  | x+y2​x=y+1.22​1.2=y2​1.2+1.22=5​y+62​15𝑥𝑦2𝑥𝑦1.221.2𝑦21.21.225𝑦6215\displaystyle\frac{x+y}{\sqrt{2}\sqrt{x}}\ =\ \frac{y+1.2}{\sqrt{2}\sqrt{1.2}}\ =\ \frac{y}{\sqrt{2}\sqrt{1.2}}+\frac{\sqrt{1.2}}{\sqrt{2}}\ =\ \frac{5y+6}{2\sqrt{15}} |  | (272) |

and

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2​x+y2​x=y+1.2⋅22​1.2=y2​1.2+1.2​2=5​y+122​15.2𝑥𝑦2𝑥𝑦⋅1.2221.2𝑦21.21.225𝑦12215\displaystyle\frac{2x+y}{\sqrt{2}\sqrt{x}}\ =\ \frac{y+1.2\cdot 2}{\sqrt{2}\sqrt{1.2}}\ =\ \frac{y}{\sqrt{2}\sqrt{1.2}}+\sqrt{1.2}\sqrt{2}\ =\ \frac{5y+12}{2\sqrt{15}}\ . |  | (273) |

Sub-function
Eq. ([115](#S3.E115 "In Lemma 15 (Main subfunction). ‣ Main Sub-Function. ‣ A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) becomes:

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(y2​1210+12102)2​erfc⁡(y2​1210+12102)−2​e(y2​1210+2​1210)2​erfc⁡(y2​1210+2​1210).superscript𝑒superscript𝑦21210121022erfc𝑦21210121022superscript𝑒superscript𝑦21210212102erfc𝑦2121021210\displaystyle e^{\left(\frac{y}{\sqrt{2}\sqrt{\frac{12}{10}}}+\frac{\sqrt{\frac{12}{10}}}{\sqrt{2}}\right)^{2}}\operatorname{erfc}\left(\frac{y}{\sqrt{2}\sqrt{\frac{12}{10}}}+\frac{\sqrt{\frac{12}{10}}}{\sqrt{2}}\right)-2e^{\left(\frac{y}{\sqrt{2}\sqrt{\frac{12}{10}}}+\sqrt{2}\sqrt{\frac{12}{10}}\right)^{2}}\operatorname{erfc}\left(\frac{y}{\sqrt{2}\sqrt{\frac{12}{10}}}+\sqrt{2}\sqrt{\frac{12}{10}}\right)\ . |  | (274) |

The derivative of this function with respect to y𝑦y is

|  |  |  |  |
| --- | --- | --- | --- |
|  | 15​π​(e160​(5​y+6)2​(5​y+6)​erfc⁡(5​y+62​15)−2​e160​(5​y+12)2​(5​y+12)​erfc⁡(5​y+122​15))+306​15​π.15𝜋superscript𝑒160superscript5𝑦625𝑦6erfc5𝑦62152superscript𝑒160superscript5𝑦1225𝑦12erfc5𝑦1221530615𝜋\displaystyle\frac{\sqrt{15\pi}\left(e^{\frac{1}{60}(5y+6)^{2}}(5y+6)\operatorname{erfc}\left(\frac{5y+6}{2\sqrt{15}}\right)-2e^{\frac{1}{60}(5y+12)^{2}}(5y+12)\operatorname{erfc}\left(\frac{5y+12}{2\sqrt{15}}\right)\right)+30}{6\sqrt{15\pi}}\ . |  | (275) |

We again will use the approximation of Ren and MacKenzie, [[30](#bib.bib30)]

|  |  |  |  |
| --- | --- | --- | --- |
|  | ez2​erfc⁡(z)=2.911π​(2.911−1)​z+π​z2+2.9112.superscript𝑒superscript𝑧2erfc𝑧2.911𝜋2.9111𝑧𝜋superscript𝑧2superscript2.9112\displaystyle e^{z^{2}}\operatorname{erfc}(z)\ =\ \frac{2.911}{\sqrt{\pi}(2.911-1)z+\sqrt{\pi z^{2}+2.911^{2}}}\ . |  | (276) |

Therefore we first perform an error analysis.
We estimated the maximum and minimum of

|  |  |  |  |
| --- | --- | --- | --- |
|  | 15​π​(2⋅2.911​(5​y+12)π​(2.911−1)​(5​y+12)2​15+π​(5​y+122​15)2+2.9112−2.911​(5​y+6)π​(2.911−1)​(5​y+6)2​15+π​(5​y+62​15)2+2.9112)+30+15𝜋⋅22.9115𝑦12𝜋2.91115𝑦12215𝜋superscript5𝑦122152superscript2.91122.9115𝑦6𝜋2.91115𝑦6215𝜋superscript5𝑦62152superscript2.9112limit-from30\displaystyle\sqrt{15\pi}\left(\frac{2\cdot 2.911(5y+12)}{\frac{\sqrt{\pi}(2.911-1)(5y+12)}{2\sqrt{15}}+\sqrt{\pi\left(\frac{5y+12}{2\sqrt{15}}\right)^{2}+2.911^{2}}}-\frac{2.911(5y+6)}{\frac{\sqrt{\pi}(2.911-1)(5y+6)}{2\sqrt{15}}+\sqrt{\pi\left(\frac{5y+6}{2\sqrt{15}}\right)^{2}+2.911^{2}}}\right)+30\ + |  | (277) |
|  |  |  |
| --- | --- | --- |
|  | 15​π​(e160​(5​y+6)2​(5​y+6)​erfc⁡(5​y+62​15)−2​e160​(5​y+12)2​(5​y+12)​erfc⁡(5​y+122​15))+30.15𝜋superscript𝑒160superscript5𝑦625𝑦6erfc5𝑦62152superscript𝑒160superscript5𝑦1225𝑦12erfc5𝑦1221530\displaystyle\sqrt{15\pi}\left(e^{\frac{1}{60}(5y+6)^{2}}(5y+6)\operatorname{erfc}\left(\frac{5y+6}{2\sqrt{15}}\right)-2e^{\frac{1}{60}(5y+12)^{2}}(5y+12)\operatorname{erfc}\left(\frac{5y+12}{2\sqrt{15}}\right)\right)+30\ . |  |

We obtained for the maximal absolute error the value 0.1630520.1630520.163052.
We added an approximation
error of 0.20.20.2 to the approximation of the derivative.
Since we want to show that the approximation upper bounds the true
expression, the addition of the approximation error is required here.
We get a sequence of inequalities:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 15​π​(e160​(5​y+6)2​(5​y+6)​erfc⁡(5​y+62​15)−2​e160​(5​y+12)2​(5​y+12)​erfc⁡(5​y+122​15))+30⩽15𝜋superscript𝑒160superscript5𝑦625𝑦6erfc5𝑦62152superscript𝑒160superscript5𝑦1225𝑦12erfc5𝑦1221530absent\displaystyle\sqrt{15\pi}\left(e^{\frac{1}{60}(5y+6)^{2}}(5y+6)\operatorname{erfc}\left(\frac{5y+6}{2\sqrt{15}}\right)-2e^{\frac{1}{60}(5y+12)^{2}}(5y+12)\operatorname{erfc}\left(\frac{5y+12}{2\sqrt{15}}\right)\right)+30\ \leqslant |  | (278) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 15​π​(2.911​(5​y+6)π​(2.911−1)​(5​y+6)2​15+π​(5​y+62​15)2+2.9112−2⋅2.911​(5​y+12)π​(2.911−1)​(5​y+12)2​15+π​(5​y+122​15)2+2.9112)+limit-from15𝜋2.9115𝑦6𝜋2.91115𝑦6215𝜋superscript5𝑦62152superscript2.9112⋅22.9115𝑦12𝜋2.91115𝑦12215𝜋superscript5𝑦122152superscript2.9112\displaystyle\sqrt{15\pi}\left(\frac{2.911(5y+6)}{\frac{\sqrt{\pi}(2.911-1)(5y+6)}{2\sqrt{15}}+\sqrt{\pi\left(\frac{5y+6}{2\sqrt{15}}\right)^{2}+2.911^{2}}}-\frac{2\cdot 2.911(5y+12)}{\frac{\sqrt{\pi}(2.911-1)(5y+12)}{2\sqrt{15}}+\sqrt{\pi\left(\frac{5y+12}{2\sqrt{15}}\right)^{2}+2.911^{2}}}\right)+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 30+0.2=300.2absent\displaystyle 30+0.2\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (30⋅2.911)​(5​y+6)(2.911−1)​(5​y+6)+(5​y+6)2+(2​15⋅2.911π)2−2​(30⋅2.911)​(5​y+12)(2.911−1)​(5​y+12)+(5​y+12)2+(2​15⋅2.911π)2+⋅302.9115𝑦62.91115𝑦6superscript5𝑦62superscript⋅2152.911𝜋2limit-from2⋅302.9115𝑦122.91115𝑦12superscript5𝑦122superscript⋅2152.911𝜋2\displaystyle\frac{(30\cdot 2.911)(5y+6)}{(2.911-1)(5y+6)+\sqrt{(5y+6)^{2}+\left(\frac{2\sqrt{15}\cdot 2.911}{\sqrt{\pi}}\right)^{2}}}-\frac{2(30\cdot 2.911)(5y+12)}{(2.911-1)(5y+12)+\sqrt{(5y+12)^{2}+\left(\frac{2\sqrt{15}\cdot 2.911}{\sqrt{\pi}}\right)^{2}}}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 30+0.2=300.2absent\displaystyle 30+0.2\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ((0.2+30)((2.911−1)(5y+12)+(5​y+12)2+(2​15⋅2.911π)2)\displaystyle\left((0.2+30)\left((2.911-1)(5y+12)+\sqrt{(5y+12)^{2}+\left(\frac{2\sqrt{15}\cdot 2.911}{\sqrt{\pi}}\right)^{2}}\right)\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ((2.911−1)​(5​y+6)+(5​y+6)2+(2​15⋅2.911π)2)−limit-from2.91115𝑦6superscript5𝑦62superscript⋅2152.911𝜋2\displaystyle\left.\left((2.911-1)(5y+6)+\sqrt{(5y+6)^{2}+\left(\frac{2\sqrt{15}\cdot 2.911}{\sqrt{\pi}}\right)^{2}}\right)-\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2⋅30⋅2.911​(5​y+12)​((2.911−1)​(5​y+6)+(5​y+6)2+(2​15⋅2.911π)2)+limit-from⋅2302.9115𝑦122.91115𝑦6superscript5𝑦62superscript⋅2152.911𝜋2\displaystyle\left.2\cdot 30\cdot 2.911(5y+12)\left((2.911-1)(5y+6)+\sqrt{(5y+6)^{2}+\left(\frac{2\sqrt{15}\cdot 2.911}{\sqrt{\pi}}\right)^{2}}\right)+\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2.911⋅30(5y+6)((2.911−1)(5y+12)+(5​y+12)2+(2​15⋅2.911π)2))\displaystyle\left.2.911\cdot 30(5y+6)\left((2.911-1)(5y+12)+\sqrt{(5y+12)^{2}+\left(\frac{2\sqrt{15}\cdot 2.911}{\sqrt{\pi}}\right)^{2}}\right)\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (((2.911−1)(5y+6)+(5​y+6)2+(2​15⋅2.911π)2)\displaystyle\left(\left((2.911-1)(5y+6)+\sqrt{(5y+6)^{2}+\left(\frac{2\sqrt{15}\cdot 2.911}{\sqrt{\pi}}\right)^{2}}\right)\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ((2.911−1)(5y+12)+(5​y+12)2+(2​15⋅2.911π)2))−1< 0.\displaystyle\left.\left((2.911-1)(5y+12)+\sqrt{(5y+12)^{2}+\left(\frac{2\sqrt{15}\cdot 2.911}{\sqrt{\pi}}\right)^{2}}\right)\right)^{-1}\ <\ 0\ . |  |

We explain this sequence of inequalities.

* •

  First inequality: The approximation of Ren and MacKenzie, [[30](#bib.bib30)]
  and then adding the error bound to ensure that the approximation
  is larger than the true value.
* •

  First equality: The factor 2​152152\sqrt{15} and 2​π2𝜋2\sqrt{\pi}
  are factored out and canceled.
* •

  Second equality: Bringing all terms to the denominator

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | ((2.911−1)​(5​y+6)+(5​y+6)2+(2​15​2.911π)2)2.91115𝑦6superscript5𝑦62superscript2152.911𝜋2\displaystyle\left((2.911-1)(5y+6)+\sqrt{(5y+6)^{2}+\left(\frac{2\sqrt{15}2.911}{\sqrt{\pi}}\right)^{2}}\right) |  | (279) |
  |  |  |  |
  | --- | --- | --- |
  |  | ((2.911−1)​(5​y+12)+(5​y+12)2+(2​15⋅2.911π)2).2.91115𝑦12superscript5𝑦122superscript⋅2152.911𝜋2\displaystyle\left((2.911-1)(5y+12)+\sqrt{(5y+12)^{2}+\left(\frac{2\sqrt{15}\cdot 2.911}{\sqrt{\pi}}\right)^{2}}\right)\ . |  |
* •

  Last inequality <0absent0<0 is proofed in the following sequence of
  inequalities.

We look at the numerator of the last term in Eq. ([278](#S3.E278 "In Behavior of the main subfunction with respect to 𝑦 at minimal 𝑥. ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")). We
have to proof that this numerator is smaller than zero in order to
proof the last inequality of Eq. ([278](#S3.E278 "In Behavior of the main subfunction with respect to 𝑦 at minimal 𝑥. ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")).
The numerator is

|  |  |  |  |
| --- | --- | --- | --- |
|  | (0.2+30)​((2.911−1)​(5​y+12)+(5​y+12)2+(2​15⋅2.911π)2)0.2302.91115𝑦12superscript5𝑦122superscript⋅2152.911𝜋2\displaystyle(0.2+30)\left((2.911-1)(5y+12)+\sqrt{(5y+12)^{2}+\left(\frac{2\sqrt{15}\cdot 2.911}{\sqrt{\pi}}\right)^{2}}\right) |  | (280) |
|  |  |  |
| --- | --- | --- |
|  | ((2.911−1)​(5​y+6)+(5​y+6)2+(2​15⋅2.911π)2)−limit-from2.91115𝑦6superscript5𝑦62superscript⋅2152.911𝜋2\displaystyle\left((2.911-1)(5y+6)+\sqrt{(5y+6)^{2}+\left(\frac{2\sqrt{15}\cdot 2.911}{\sqrt{\pi}}\right)^{2}}\right)\ - |  |
|  |  |  |
| --- | --- | --- |
|  | 2⋅30⋅2.911​(5​y+12)​((2.911−1)​(5​y+6)+(5​y+6)2+(2​15⋅2.911π)2)+limit-from⋅2302.9115𝑦122.91115𝑦6superscript5𝑦62superscript⋅2152.911𝜋2\displaystyle 2\cdot 30\cdot 2.911(5y+12)\left((2.911-1)(5y+6)+\sqrt{(5y+6)^{2}+\left(\frac{2\sqrt{15}\cdot 2.911}{\sqrt{\pi}}\right)^{2}}\right)+ |  |
|  |  |  |
| --- | --- | --- |
|  | 2.911⋅30​(5​y+6)​((2.911−1)​(5​y+12)+(5​y+12)2+(2​15​ .2.911π)2).⋅2.911305𝑦62.91115𝑦12superscript5𝑦122superscript215.2.911𝜋2\displaystyle 2.911\cdot 30(5y+6)\left((2.911-1)(5y+12)+\sqrt{(5y+12)^{2}+\left(\frac{2\sqrt{15}\ .2.911}{\sqrt{\pi}}\right)^{2}}\right)\ . |  |

We now compute upper bounds for this numerator:

|  |  |  |  |
| --- | --- | --- | --- |
|  | (0.2+30)​((2.911−1)​(5​y+12)+(5​y+12)2+(2​15⋅2.911π)2)0.2302.91115𝑦12superscript5𝑦122superscript⋅2152.911𝜋2\displaystyle(0.2+30)\left((2.911-1)(5y+12)+\sqrt{(5y+12)^{2}+\left(\frac{2\sqrt{15}\cdot 2.911}{\sqrt{\pi}}\right)^{2}}\right) |  | (281) |
|  |  |  |
| --- | --- | --- |
|  | ((2.911−1)​(5​y+6)+(5​y+6)2+(2​15⋅2.911π)2)−limit-from2.91115𝑦6superscript5𝑦62superscript⋅2152.911𝜋2\displaystyle\left((2.911-1)(5y+6)+\sqrt{(5y+6)^{2}+\left(\frac{2\sqrt{15}\cdot 2.911}{\sqrt{\pi}}\right)^{2}}\right)- |  |
|  |  |  |
| --- | --- | --- |
|  | 2⋅30⋅2.911​(5​y+12)​((2.911−1)​(5​y+6)+(5​y+6)2+(2​15⋅2.911π)2)+limit-from⋅2302.9115𝑦122.91115𝑦6superscript5𝑦62superscript⋅2152.911𝜋2\displaystyle 2\cdot 30\cdot 2.911(5y+12)\left((2.911-1)(5y+6)+\sqrt{(5y+6)^{2}+\left(\frac{2\sqrt{15}\cdot 2.911}{\sqrt{\pi}}\right)^{2}}\right)+ |  |
|  |  |  |
| --- | --- | --- |
|  | 2.911⋅30​(5​y+6)​((2.911−1)​(5​y+12)+(5​y+12)2+(2​15⋅2.911π)2)=⋅2.911305𝑦62.91115𝑦12superscript5𝑦122superscript⋅2152.911𝜋2absent\displaystyle 2.911\cdot 30(5y+6)\left((2.911-1)(5y+12)+\sqrt{(5y+12)^{2}+\left(\frac{2\sqrt{15}\cdot 2.911}{\sqrt{\pi}}\right)^{2}}\right)\ = |  |
|  |  |  |
| --- | --- | --- |
|  | −1414.99​y2−584.739​(5​y+6)2+161.84​y+725.211​(5​y+12)2+161.84​y−1414.99superscript𝑦2584.739superscript5𝑦62161.84𝑦limit-from725.211superscript5𝑦122161.84𝑦\displaystyle-1414.99y^{2}-584.739\sqrt{(5y+6)^{2}+161.84}y+725.211\sqrt{(5y+12)^{2}+161.84}y- |  |
|  |  |  |
| --- | --- | --- |
|  | 5093.97​y−1403.37​(5​y+6)2+161.84+30.2​(5​y+6)2+161.84​(5​y+12)2+161.84+5093.97𝑦1403.37superscript5𝑦62161.84limit-from30.2superscript5𝑦62161.84superscript5𝑦122161.84\displaystyle 5093.97y-1403.37\sqrt{(5y+6)^{2}+161.84}+30.2\sqrt{(5y+6)^{2}+161.84}\sqrt{(5y+12)^{2}+161.84}+ |  |
|  |  |  |
| --- | --- | --- |
|  | 870.253​(5​y+12)2+161.84−4075.17<870.253superscript5𝑦122161.844075.17absent\displaystyle 870.253\sqrt{(5y+12)^{2}+161.84}-4075.17\ < |  |
|  |  |  |
| --- | --- | --- |
|  | −1414.99​y2−584.739​(5​y+6)2+161.84​y+725.211​(5​y+12)2+161.84​y−1414.99superscript𝑦2584.739superscript5𝑦62161.84𝑦limit-from725.211superscript5𝑦122161.84𝑦\displaystyle-1414.99y^{2}-584.739\sqrt{(5y+6)^{2}+161.84}y+725.211\sqrt{(5y+12)^{2}+161.84}y- |  |
|  |  |  |
| --- | --- | --- |
|  | 5093.97​y−1403.37​(6+5⋅(−0.1))2+161.84+30.2​(6+5⋅0.1)2+161.84​(12+5⋅0.1)2+161.84+5093.97𝑦1403.37superscript6⋅50.12161.84limit-from30.2superscript6⋅50.12161.84superscript12⋅50.12161.84\displaystyle 5093.97y-1403.37\sqrt{(6+5\cdot(-0.1))^{2}+161.84}+30.2\sqrt{(6+5\cdot 0.1)^{2}+161.84}\sqrt{(12+5\cdot 0.1)^{2}+161.84}+ |  |
|  |  |  |
| --- | --- | --- |
|  | 870.253​(12+5⋅0.1)2+161.84−4075.17=870.253superscript12⋅50.12161.844075.17absent\displaystyle 870.253\sqrt{(12+5\cdot 0.1)^{2}+161.84}-4075.17\ = |  |
|  |  |  |
| --- | --- | --- |
|  | −1414.99​y2−584.739​(5​y+6)2+161.84​y+725.211​(5​y+12)2+161.84​y−5093.97​y−309.691<1414.99superscript𝑦2584.739superscript5𝑦62161.84𝑦725.211superscript5𝑦122161.84𝑦5093.97𝑦309.691absent\displaystyle-1414.99y^{2}-584.739\sqrt{(5y+6)^{2}+161.84}y+725.211\sqrt{(5y+12)^{2}+161.84}y-5093.97y-309.691\ < |  |
|  |  |  |
| --- | --- | --- |
|  | y​(−584.739​(5​y+6)2+161.84+725.211​(5​y+12)2+161.84−5093.97)−309.691<𝑦584.739superscript5𝑦62161.84725.211superscript5𝑦122161.845093.97309.691absent\displaystyle y\left(-584.739\sqrt{(5y+6)^{2}+161.84}+725.211\sqrt{(5y+12)^{2}+161.84}-5093.97\right)-309.691\ < |  |
|  |  |  |
| --- | --- | --- |
|  | −0.1​(725.211​(12+5⋅(−0.1))2+161.84−584.739​(6+5⋅0.1)2+161.84−5093.97)−309.691=0.1725.211superscript12⋅50.12161.84584.739superscript6⋅50.12161.845093.97309.691absent\displaystyle-0.1\left(725.211\sqrt{(12+5\cdot(-0.1))^{2}+161.84}-584.739\sqrt{(6+5\cdot 0.1)^{2}+161.84}-5093.97\right)-309.691\ = |  |
|  |  |  |
| --- | --- | --- |
|  | −208.604.208.604\displaystyle-208.604\ . |  |

For the first inequality we choose y𝑦y in the roots, so that
positive terms maximally increase and negative terms maximally decrease.
The second inequality just removed the y2superscript𝑦2y^{2} term which is always
negative, therefore increased the expression.
For the last inequality, the term in brackets
is negative for all settings of y𝑦y.
Therefore we make the brackets as negative as possible
and make the whole term positive by multiplying with y=−0.1𝑦0.1y=-0.1.

Consequently

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(x+y)22​x​erfc⁡(x+y2​x)−2​e(2​x+y)22​x​erfc⁡(2​x+y2​x)superscript𝑒superscript𝑥𝑦22𝑥erfc𝑥𝑦2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥erfc2𝑥𝑦2𝑥\displaystyle e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)-2e^{\frac{(2x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right) |  | (282) |

is strictly monotonically decreasing in y𝑦y for the minimal
x=1.2𝑥1.2x=1.2.
∎

###### Lemma 45 (Main subfunction below).

For 0.007⩽x⩽0.8750.007𝑥0.8750.007\leqslant x\leqslant 0.875 and −0.01⩽y⩽0.010.01𝑦0.01-0.01\leqslant y\leqslant 0.01,
the function

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(x+y)22​x​erfc⁡(x+y2​x)−2​e(2​x+y)22​x​erfc⁡(2​x+y2​x)superscript𝑒superscript𝑥𝑦22𝑥erfc𝑥𝑦2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥erfc2𝑥𝑦2𝑥\displaystyle e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)-2e^{\frac{(2x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right) |  | (283) |

smaller than zero, is strictly monotonically increasing in x𝑥x
and strictly monotonically increasing in y𝑦y for the minimal x=0.007=0.00875⋅0.8𝑥0.007⋅0.008750.8x=0.007=0.00875\cdot 0.8,
x=0.56=0.7⋅0.8𝑥0.56⋅0.70.8x=0.56=0.7\cdot 0.8, x=0.128=0.16⋅0.8𝑥0.128⋅0.160.8x=0.128=0.16\cdot 0.8, and x=0.216=0.24⋅0.9𝑥0.216⋅0.240.9x=0.216=0.24\cdot 0.9 (lower
bound of 0.90.90.9 on τ𝜏\tau).

###### Proof.

We first consider the derivative of sub-function
Eq. ([125](#S3.E125 "In Lemma 17 (Main subfunction Below). ‣ Main Sub-Function From Below. ‣ A3.4.4 Lemmata for proofing Theorem 3: The variance is expanding ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) with respect to x𝑥x.
The derivative of the function

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(x+y)22​x​erfc⁡(x+y2​x)−2​e(2​x+y)22​x​erfc⁡(2​x+y2​x)superscript𝑒superscript𝑥𝑦22𝑥erfc𝑥𝑦2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥erfc2𝑥𝑦2𝑥\displaystyle e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)-2e^{\frac{(2x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right) |  | (284) |

with respect to x𝑥x is

|  |  |  |  |
| --- | --- | --- | --- |
|  | π​(e(x+y)22​x​(x−y)​(x+y)​erfc⁡(x+y2​x)−2​e(2​x+y)22​x​(4​x2−y2)​erfc⁡(2​x+y2​x))+2​x​(3​x−y)2​π​x2=𝜋superscript𝑒superscript𝑥𝑦22𝑥𝑥𝑦𝑥𝑦erfc𝑥𝑦2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥4superscript𝑥2superscript𝑦2erfc2𝑥𝑦2𝑥2𝑥3𝑥𝑦2𝜋superscript𝑥2absent\displaystyle\frac{\sqrt{\pi}\left(e^{\frac{(x+y)^{2}}{2x}}(x-y)(x+y)\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)-2e^{\frac{(2x+y)^{2}}{2x}}\left(4x^{2}-y^{2}\right)\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)\right)+\sqrt{2}\sqrt{x}(3x-y)}{2\sqrt{\pi}x^{2}}\ = |  | (285) |
|  |  |  |
| --- | --- | --- |
|  | π​(e(x+y)22​x​(x−y)​(x+y)​erfc⁡(x+y2​x)−2​e(2​x+y)22​x​(2​x+y)​(2​x−y)​erfc⁡(2​x+y2​x))+2​x​(3​x−y)2​π​x2=𝜋superscript𝑒superscript𝑥𝑦22𝑥𝑥𝑦𝑥𝑦erfc𝑥𝑦2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥2𝑥𝑦2𝑥𝑦erfc2𝑥𝑦2𝑥2𝑥3𝑥𝑦2𝜋superscript𝑥2absent\displaystyle\frac{\sqrt{\pi}\left(e^{\frac{(x+y)^{2}}{2x}}(x-y)(x+y)\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)-2e^{\frac{(2x+y)^{2}}{2x}}(2x+y)(2x-y)\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)\right)+\sqrt{2}\sqrt{x}(3x-y)}{2\sqrt{\pi}x^{2}}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | π​(e(x+y)22​x​(x−y)​(x+y)​erfc⁡(x+y2​x)2​x−2​e(2​x+y)22​x​(2​x+y)​(2​x−y)​erfc⁡(2​x+y2​x)2​x)+(3​x−y)2​2​π​x​x2.𝜋superscript𝑒superscript𝑥𝑦22𝑥𝑥𝑦𝑥𝑦erfc𝑥𝑦2𝑥2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥2𝑥𝑦2𝑥𝑦erfc2𝑥𝑦2𝑥2𝑥3𝑥𝑦22𝜋𝑥superscript𝑥2\displaystyle\frac{\sqrt{\pi}\left(\frac{e^{\frac{(x+y)^{2}}{2x}}(x-y)(x+y)\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)}{\sqrt{2}\sqrt{x}}-\frac{2e^{\frac{(2x+y)^{2}}{2x}}(2x+y)(2x-y)\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)}{\sqrt{2}\sqrt{x}}\right)+(3x-y)}{\sqrt{2}2\sqrt{\pi}\sqrt{x}x^{2}}\ . |  |

We consider the numerator

|  |  |  |  |
| --- | --- | --- | --- |
|  | π​(e(x+y)22​x​(x−y)​(x+y)​erfc⁡(x+y2​x)2​x−2​e(2​x+y)22​x​(2​x+y)​(2​x−y)​erfc⁡(2​x+y2​x)2​x)+(3​x−y).𝜋superscript𝑒superscript𝑥𝑦22𝑥𝑥𝑦𝑥𝑦erfc𝑥𝑦2𝑥2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥2𝑥𝑦2𝑥𝑦erfc2𝑥𝑦2𝑥2𝑥3𝑥𝑦\displaystyle\sqrt{\pi}\left(\frac{e^{\frac{(x+y)^{2}}{2x}}(x-y)(x+y)\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)}{\sqrt{2}\sqrt{x}}-\frac{2e^{\frac{(2x+y)^{2}}{2x}}(2x+y)(2x-y)\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)}{\sqrt{2}\sqrt{x}}\right)+(3x-y)\ . |  | (286) |

For bounding this value, we use the approximation

|  |  |  |  |
| --- | --- | --- | --- |
|  | ez2​erfc⁡(z)≈2.911π​(2.911−1)​z+π​z2+2.9112.superscript𝑒superscript𝑧2erfc𝑧2.911𝜋2.9111𝑧𝜋superscript𝑧2superscript2.9112\displaystyle e^{z^{2}}\operatorname{erfc}(z)\ \approx\ \frac{2.911}{\sqrt{\pi}(2.911-1)z+\sqrt{\pi z^{2}+2.911^{2}}}\ . |  | (287) |

from Ren and MacKenzie, [[30](#bib.bib30)].
We start with an error analysis of this approximation.
According to Ren and MacKenzie, [[30](#bib.bib30)] (Figure 1), the approximation
error is both positive and negative in the range
[0.175,1.33]0.1751.33[0.175,1.33]. This range contains all possible
arguments of erfcerfc\operatorname{erfc} that we consider in this subsection.
Numerically we maximized and minimized the approximation error of the
whole expression

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | E​(x,y)𝐸𝑥𝑦\displaystyle E(x,y)\ | =(e(x+y)22​x​(x−y)​(x+y)​erfc⁡(x+y2​x)2​x−2​e(2​x+y)22​x​(2​x−y)​(2​x+y)​erfc⁡(2​x+y2​x)2​x)−absentlimit-fromsuperscript𝑒superscript𝑥𝑦22𝑥𝑥𝑦𝑥𝑦erfc𝑥𝑦2𝑥2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥2𝑥𝑦2𝑥𝑦erfc2𝑥𝑦2𝑥2𝑥\displaystyle=\ \left(\frac{e^{\frac{(x+y)^{2}}{2x}}(x-y)(x+y)\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)}{\sqrt{2}\sqrt{x}}-\frac{2e^{\frac{(2x+y)^{2}}{2x}}(2x-y)(2x+y)\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)}{\sqrt{2}\sqrt{x}}\right)\ - |  | (288) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (2.911​(x−y)​(x+y)(2​x)​(π​(2.911−1)​(x+y)2​x+π​(x+y2​x)2+2.9112)−\displaystyle\left(\frac{2.911(x-y)(x+y)}{\left(\sqrt{2}\sqrt{x}\right)\left(\frac{\sqrt{\pi}(2.911-1)(x+y)}{\sqrt{2}\sqrt{x}}+\sqrt{\pi\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)^{2}+2.911^{2}}\right)}\ -\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2⋅2.911​(2​x−y)​(2​x+y)(2​x)​(π​(2.911−1)​(2​x+y)2​x+π​(2​x+y2​x)2+2.9112)).\displaystyle\left.\frac{2\cdot 2.911(2x-y)(2x+y)}{\left(\sqrt{2}\sqrt{x}\right)\left(\frac{\sqrt{\pi}(2.911-1)(2x+y)}{\sqrt{2}\sqrt{x}}+\sqrt{\pi\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)^{2}+2.911^{2}}\right)}\right)\ . |  |

We numerically determined −0.000228141⩽E​(x,y)⩽0.004956880.000228141𝐸𝑥𝑦0.00495688-0.000228141\leqslant E(x,y)\leqslant 0.00495688 for
0.08⩽x⩽0.8750.08𝑥0.8750.08\leqslant x\leqslant 0.875 and −0.01⩽y⩽0.010.01𝑦0.01-0.01\leqslant y\leqslant 0.01.
We used different numerical optimization techniques like
gradient based constraint BFGS algorithms and
non-gradient-based Nelder-Mead methods with different start points.
Therefore our approximation is smaller than the function that we
approximate.

We use an error gap of −0.00030.0003-0.0003 to countermand the error due to the
approximation. We have the sequences of inequalities using the approximation of
Ren and MacKenzie, [[30](#bib.bib30)]:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | (3​x−y)+(e(x+y)22​x​(x−y)​(x+y)​erfc⁡(x+y2​x)2​x−2​e(2​x+y)22​x​(2​x−y)​(2​x+y)​erfc⁡(2​x+y2​x)2​x)​π⩾3𝑥𝑦superscript𝑒superscript𝑥𝑦22𝑥𝑥𝑦𝑥𝑦erfc𝑥𝑦2𝑥2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥2𝑥𝑦2𝑥𝑦erfc2𝑥𝑦2𝑥2𝑥𝜋absent\displaystyle(3x-y)+\left(\frac{e^{\frac{(x+y)^{2}}{2x}}(x-y)(x+y)\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)}{\sqrt{2}\sqrt{x}}-\frac{2e^{\frac{(2x+y)^{2}}{2x}}(2x-y)(2x+y)\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)}{\sqrt{2}\sqrt{x}}\right)\sqrt{\pi}\ \geqslant |  | (289) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (3x−y)+(2.911​(x−y)​(x+y)(π​(x+y2​x)2+2.9112+(2.911−1)​π​(x+y)2​x)​(2​x)−\displaystyle(3x-y)+\left(\frac{2.911(x-y)(x+y)}{\left(\sqrt{\pi\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)^{2}+2.911^{2}}+\frac{(2.911-1)\sqrt{\pi}(x+y)}{\sqrt{2}\sqrt{x}}\right)\left(\sqrt{2}\sqrt{x}\right)}\ -\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2​(2​x−y)​(2​x+y)​2.911(2​x)​(π​(2​x+y2​x)2+2.9112+(2.911−1)​π​(2​x+y)2​x))π−0.0003=\displaystyle\left.\frac{2(2x-y)(2x+y)2.911}{\left(\sqrt{2}\sqrt{x}\right)\left(\sqrt{\pi\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)^{2}+2.911^{2}}+\frac{(2.911-1)\sqrt{\pi}(2x+y)}{\sqrt{2}\sqrt{x}}\right)}\right)\sqrt{\pi}-0.0003\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (3x−y)+((2​x​2.911)​(x−y)​(x+y)(π​(x+y)2+2⋅2.9112​x+(2.911−1)​(x+y)​π)​(2​x)−\displaystyle(3x-y)+\left(\frac{\left(\sqrt{2}\sqrt{x}2.911\right)(x-y)(x+y)}{\left(\sqrt{\pi(x+y)^{2}+2\cdot 2.911^{2}x}+(2.911-1)(x+y)\sqrt{\pi}\right)\left(\sqrt{2}\sqrt{x}\right)}\ -\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2​(2​x−y)​(2​x+y)​(2​x​2.911)(2​x)​(π​(2​x+y)2+2⋅2.9112​x+(2.911−1)​(2​x+y)​π))π−0.0003=\displaystyle\left.\frac{2(2x-y)(2x+y)\left(\sqrt{2}\sqrt{x}2.911\right)}{\left(\sqrt{2}\sqrt{x}\right)\left(\sqrt{\pi(2x+y)^{2}+2\cdot 2.911^{2}x}+(2.911-1)(2x+y)\sqrt{\pi}\right)}\right)\sqrt{\pi}-0.0003\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (3x−y)+2.911((x−y)​(x+y)(2.911−1)​(x+y)+(x+y)2+2⋅2.9112​xπ−\displaystyle(3x-y)+2.911\left(\frac{(x-y)(x+y)}{(2.911-1)(x+y)+\sqrt{(x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}}\ -\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2​(2​x−y)​(2​x+y)(2.911−1)​(2​x+y)+(2​x+y)2+2⋅2.9112​xπ)−0.0003⩾\displaystyle\left.\frac{2(2x-y)(2x+y)}{(2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}}\right)-0.0003\ \geqslant |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (3x−y)+2.911((x−y)​(x+y)(2.911−1)​(x+y)+(2.9112π)2+(x+y)2+2⋅2.9112​xπ+2⋅2.9112​yπ−\displaystyle(3x-y)+2.911\left(\frac{(x-y)(x+y)}{(2.911-1)(x+y)+\sqrt{\left(\frac{2.911^{2}}{\pi}\right)^{2}+(x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}+\frac{2\cdot 2.911^{2}y}{\pi}}}\ -\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2​(2​x−y)​(2​x+y)(2.911−1)​(2​x+y)+(2​x+y)2+2⋅2.9112​xπ)−0.0003=\displaystyle\left.\frac{2(2x-y)(2x+y)}{(2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}}\right)-0.0003\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (3x−y)+2.911((x−y)​(x+y)(2.911−1)​(x+y)+(x+y+2.9112π)2−\displaystyle(3x-y)+2.911\left(\frac{(x-y)(x+y)}{(2.911-1)(x+y)+\sqrt{\left(x+y+\frac{2.911^{2}}{\pi}\right)^{2}}}\ -\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2​(2​x−y)​(2​x+y)(2.911−1)​(2​x+y)+(2​x+y)2+2⋅2.9112​xπ)−0.0003=\displaystyle\left.\frac{2(2x-y)(2x+y)}{(2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}}\right)-0.0003\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (3​x−y)+2.911​((x−y)​(x+y)2.911​(x+y)+2.9112π−2​(2​x−y)​(2​x+y)(2.911−1)​(2​x+y)+(2​x+y)2+2⋅2.9112​xπ)−0.0003=3𝑥𝑦2.911𝑥𝑦𝑥𝑦2.911𝑥𝑦superscript2.9112𝜋22𝑥𝑦2𝑥𝑦2.91112𝑥𝑦superscript2𝑥𝑦2⋅2superscript2.9112𝑥𝜋0.0003absent\displaystyle(3x-y)+2.911\left(\frac{(x-y)(x+y)}{2.911(x+y)+\frac{2.911^{2}}{\pi}}-\frac{2(2x-y)(2x+y)}{(2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}}\right)-0.0003\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (3​x−y)+(x−y)​(x+y)(x+y)+2.911π−2​(2​x−y)​(2​x+y)​2.911(2.911−1)​(2​x+y)+(2​x+y)2+2⋅2.9112​xπ−0.0003=3𝑥𝑦𝑥𝑦𝑥𝑦𝑥𝑦2.911𝜋22𝑥𝑦2𝑥𝑦2.9112.91112𝑥𝑦superscript2𝑥𝑦2⋅2superscript2.9112𝑥𝜋0.0003absent\displaystyle(3x-y)+\frac{(x-y)(x+y)}{(x+y)+\frac{2.911}{\pi}}-\frac{2(2x-y)(2x+y)2.911}{(2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}}-0.0003\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (3​x−y)+(x−y)​(x+y)(x+y)+2.911π−2​(2​x−y)​(2​x+y)​2.911(2.911−1)​(2​x+y)+(2​x+y)2+2⋅2.9112​xπ−0.0003=3𝑥𝑦𝑥𝑦𝑥𝑦𝑥𝑦2.911𝜋22𝑥𝑦2𝑥𝑦2.9112.91112𝑥𝑦superscript2𝑥𝑦2⋅2superscript2.9112𝑥𝜋0.0003absent\displaystyle(3x-y)+\frac{(x-y)(x+y)}{(x+y)+\frac{2.911}{\pi}}-\frac{2(2x-y)(2x+y)2.911}{(2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}}-0.0003\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−2(2x−y)2.911((x+y)+2.911π)(2x+y)+\displaystyle\left(-2(2x-y)2.911\left((x+y)+\frac{2.911}{\pi}\right)(2x+y)\right.\ + |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ((x+y)+2.911π)​(3​x−y−0.0003)​((2.911−1)​(2​x+y)+(2​x+y)2+2⋅2.9112​xπ)+limit-from𝑥𝑦2.911𝜋3𝑥𝑦0.00032.91112𝑥𝑦superscript2𝑥𝑦2⋅2superscript2.9112𝑥𝜋\displaystyle\left.\left((x+y)+\frac{2.911}{\pi}\right)(3x-y-0.0003)\left((2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}\right)\right.\ + |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (x−y)(x+y)((2.911−1)(2x+y)+(2​x+y)2+2⋅2.9112​xπ))\displaystyle\left.(x-y)(x+y)\left((2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}\right)\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (((x+y)+2.911π)​((2.911−1)​(2​x+y)+(2​x+y)2+2⋅2.9112​xπ))−1=superscript𝑥𝑦2.911𝜋2.91112𝑥𝑦superscript2𝑥𝑦2⋅2superscript2.9112𝑥𝜋1absent\displaystyle\left(\left((x+y)+\frac{2.911}{\pi}\right)\left((2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}\right)\right)^{-1}\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−8x3−8x2y+4x2(2​x+y)2+5.39467​x−10.9554x2+2xy2−2y2(2​x+y)2+5.39467​x+\displaystyle\left(-8x^{3}-8x^{2}y+4x^{2}\sqrt{(2x+y)^{2}+5.39467x}-10.9554x^{2}+2xy^{2}-2y^{2}\sqrt{(2x+y)^{2}+5.39467x}\ +\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 1.76901​x​y+2​x​y​(2​x+y)2+5.39467​x+2.7795​x​(2​x+y)2+5.39467​x−1.76901𝑥𝑦2𝑥𝑦superscript2𝑥𝑦25.39467𝑥limit-from2.7795𝑥superscript2𝑥𝑦25.39467𝑥\displaystyle\left.1.76901xy+2xy\sqrt{(2x+y)^{2}+5.39467x}+2.7795x\sqrt{(2x+y)^{2}+5.39467x}\ -\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 0.9269​y​(2​x+y)2+5.39467​x−0.00027798​(2​x+y)2+5.39467​x−0.00106244​x+0.9269𝑦superscript2𝑥𝑦25.39467𝑥0.00027798superscript2𝑥𝑦25.39467𝑥limit-from0.00106244𝑥\displaystyle\left.0.9269y\sqrt{(2x+y)^{2}+5.39467x}-0.00027798\sqrt{(2x+y)^{2}+5.39467x}-0.00106244x\ +\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2y3+3.62336y2−0.00053122y)\displaystyle\left.2y^{3}+3.62336y^{2}-0.00053122y\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (((x+y)+2.911π)​((2.911−1)​(2​x+y)+(2​x+y)2+2⋅2.9112​xπ))−1=superscript𝑥𝑦2.911𝜋2.91112𝑥𝑦superscript2𝑥𝑦2⋅2superscript2.9112𝑥𝜋1absent\displaystyle\left(\left((x+y)+\frac{2.911}{\pi}\right)\left((2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}\right)\right)^{-1}\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−8x3+(4x2+2xy+2.7795x−2y2−0.9269y−0.00027798)(2​x+y)2+5.39467​x−\displaystyle\left(-8x^{3}+\left(4x^{2}+2xy+2.7795x-2y^{2}-0.9269y-0.00027798\right)\sqrt{(2x+y)^{2}+5.39467x}\ -\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 8x2y−10.9554x2+2xy2+1.76901xy−0.00106244x+2y3+3.62336y2−0.00053122y)\displaystyle\left.8x^{2}y-10.9554x^{2}+2xy^{2}+1.76901xy-0.00106244x+2y^{3}+3.62336y^{2}-0.00053122y\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (((x+y)+2.911π)​((2.911−1)​(2​x+y)+(2​x+y)2+2⋅2.9112​xπ))−1> 0.superscript𝑥𝑦2.911𝜋2.91112𝑥𝑦superscript2𝑥𝑦2⋅2superscript2.9112𝑥𝜋1 0\displaystyle\left(\left((x+y)+\frac{2.911}{\pi}\right)\left((2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}\right)\right)^{-1}\ >\ 0\ . |  |

We explain this sequence of inequalities:

* •

  First inequality: The approximation of Ren and MacKenzie, [[30](#bib.bib30)]
  and then subtracting an error gap of 0.00030.00030.0003.
* •

  Equalities: The factor 2​x2𝑥\sqrt{2}\sqrt{x} is factored out and
  canceled.
* •

  Second inequality: adds a positive term in the first root to
  obtain a binomial form. The term containing the root
  is positive and the root is in the denominator,
  therefore the whole term becomes smaller.
* •

  Equalities: solve for the term and factor out.
* •

  Bringing all terms to the denominator
  ((x+y)+2.911π)​((2.911−1)​(2​x+y)+(2​x+y)2+2⋅2.9112​xπ)𝑥𝑦2.911𝜋2.91112𝑥𝑦superscript2𝑥𝑦2⋅2superscript2.9112𝑥𝜋\left((x+y)+\frac{2.911}{\pi}\right)\left((2.911-1)(2x+y)+\sqrt{(2x+y)^{2}+\frac{2\cdot 2.911^{2}x}{\pi}}\right).
* •

  Equalities: Multiplying out and expanding terms.
* •

  Last inequality >0absent0>0 is proofed in the following sequence of
  inequalities.

We look at the numerator of the last expression of
Eq. ([289](#S3.E289 "In Proof. ‣ Behavior of the main subfunction with respect to 𝑦 at minimal 𝑥. ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")), which we show to be
positive in order to show >0absent0>0 in
Eq. ([289](#S3.E289 "In Proof. ‣ Behavior of the main subfunction with respect to 𝑦 at minimal 𝑥. ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")). The numerator is

|  |  |  |  |
| --- | --- | --- | --- |
|  | −8​x3+(4​x2+2​x​y+2.7795​x−2​y2−0.9269​y−0.00027798)​(2​x+y)2+5.39467​x−8superscript𝑥3limit-from4superscript𝑥22𝑥𝑦2.7795𝑥2superscript𝑦20.9269𝑦0.00027798superscript2𝑥𝑦25.39467𝑥\displaystyle-8x^{3}+\left(4x^{2}+2xy+2.7795x-2y^{2}-0.9269y-0.00027798\right)\sqrt{(2x+y)^{2}+5.39467x}\ - |  | (290) |
|  |  |  |
| --- | --- | --- |
|  | 8​x2​y−10.9554​x2+2​x​y2+1.76901​x​y−0.00106244​x+2​y3+3.62336​y2−0.00053122​y.8superscript𝑥2𝑦10.9554superscript𝑥22𝑥superscript𝑦21.76901𝑥𝑦0.00106244𝑥2superscript𝑦33.62336superscript𝑦20.00053122𝑦\displaystyle 8x^{2}y-10.9554x^{2}+2xy^{2}+1.76901xy-0.00106244x+2y^{3}+3.62336y^{2}-0.00053122y\ . |  |

The factor 4​x2+2​x​y+2.7795​x−2​y2−0.9269​y−0.000277984superscript𝑥22𝑥𝑦2.7795𝑥2superscript𝑦20.9269𝑦0.000277984x^{2}+2xy+2.7795x-2y^{2}-0.9269y-0.00027798
in front of the root is positive:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 4​x2+2​x​y+2.7795​x−2​y2−0.9269​y−0.00027798>4superscript𝑥22𝑥𝑦2.7795𝑥2superscript𝑦20.9269𝑦0.00027798absent\displaystyle 4x^{2}+2xy+2.7795x-2y^{2}-0.9269y-0.00027798\ > |  | (291) |
|  |  |  |
| --- | --- | --- |
|  | −2​y2+0.007⋅2​y−0.9269​y+4⋅0.0072+2.7795⋅0.007−0.00027798=2superscript𝑦2⋅0.0072𝑦0.9269𝑦⋅4superscript0.0072⋅2.77950.0070.00027798absent\displaystyle-2y^{2}+0.007\cdot 2y-0.9269y+4\cdot 0.007^{2}+2.7795\cdot 0.007-0.00027798\ = |  |
|  |  |  |
| --- | --- | --- |
|  | −2​y2−0.9129​y+2.77942=−2​(y+1.42897)​(y−0.972523)>0.2superscript𝑦20.9129𝑦2.779422𝑦1.42897𝑦0.9725230\displaystyle-2y^{2}-0.9129y+2.77942\ =-2(y+1.42897)(y-0.972523)\ >0\ . |  |

If the term that does not contain the root would be positive,
then everything is positive and we have proofed the the numerator is
positive. Therefore we consider the case that the term that does
not contain the root is negative.
The term that contains the root must be larger than
the other term in absolute values.

|  |  |  |  |
| --- | --- | --- | --- |
|  | −(−8​x3−8​x2​y−10.9554​x2+2​x​y2+1.76901​x​y−0.00106244​x+2​y3+3.62336​y2−0.00053122​y)<8superscript𝑥38superscript𝑥2𝑦10.9554superscript𝑥22𝑥superscript𝑦21.76901𝑥𝑦0.00106244𝑥2superscript𝑦33.62336superscript𝑦20.00053122𝑦absent\displaystyle-\left(-8x^{3}-8x^{2}y-10.9554x^{2}+2xy^{2}+1.76901xy-0.00106244x+2y^{3}+3.62336y^{2}-0.00053122y\right)\ < |  | (292) |
|  |  |  |
| --- | --- | --- |
|  | (4​x2+2​x​y+2.7795​x−2​y2−0.9269​y−0.00027798)​(2​x+y)2+5.39467​x.4superscript𝑥22𝑥𝑦2.7795𝑥2superscript𝑦20.9269𝑦0.00027798superscript2𝑥𝑦25.39467𝑥\displaystyle\left(4x^{2}+2xy+2.7795x-2y^{2}-0.9269y-0.00027798\right)\sqrt{(2x+y)^{2}+5.39467x}\ . |  |

Therefore the squares of the root term have to be larger
than the square of the other term to show >0absent0>0 in
Eq. ([289](#S3.E289 "In Proof. ‣ Behavior of the main subfunction with respect to 𝑦 at minimal 𝑥. ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")).
Thus, we have the inequality:

|  |  |  |  |
| --- | --- | --- | --- |
|  | (−8​x3−8​x2​y−10.9554​x2+2​x​y2+1.76901​x​y−0.00106244​x+2​y3+3.62336​y2−0.00053122​y)2<superscript8superscript𝑥38superscript𝑥2𝑦10.9554superscript𝑥22𝑥superscript𝑦21.76901𝑥𝑦0.00106244𝑥2superscript𝑦33.62336superscript𝑦20.00053122𝑦2absent\displaystyle\left(-8x^{3}-8x^{2}y-10.9554x^{2}+2xy^{2}+1.76901xy-0.00106244x+2y^{3}+3.62336y^{2}-0.00053122y\right)^{2}\ < |  | (293) |
|  |  |  |
| --- | --- | --- |
|  | (4​x2+2​x​y+2.7795​x−2​y2−0.9269​y−0.00027798)2​((2​x+y)2+5.39467​x).superscript4superscript𝑥22𝑥𝑦2.7795𝑥2superscript𝑦20.9269𝑦0.000277982superscript2𝑥𝑦25.39467𝑥\displaystyle\left(4x^{2}+2xy+2.7795x-2y^{2}-0.9269y-0.00027798\right)^{2}\left((2x+y)^{2}+5.39467x\right)\ . |  |

This is equivalent to

|  |  |  |  |
| --- | --- | --- | --- |
|  | 0<(4​x2+2​x​y+2.7795​x−2​y2−0.9269​y−0.00027798)2​((2​x+y)2+5.39467​x)−0limit-fromsuperscript4superscript𝑥22𝑥𝑦2.7795𝑥2superscript𝑦20.9269𝑦0.000277982superscript2𝑥𝑦25.39467𝑥\displaystyle 0\ <\ \left(4x^{2}+2xy+2.7795x-2y^{2}-0.9269y-0.00027798\right)^{2}\left((2x+y)^{2}+5.39467x\right)- |  | (294) |
|  |  |  |
| --- | --- | --- |
|  | (−8​x3−8​x2​y−10.9554​x2+2​x​y2+1.76901​x​y−0.00106244​x+2​y3+3.62336​y2−0.00053122​y)2=superscript8superscript𝑥38superscript𝑥2𝑦10.9554superscript𝑥22𝑥superscript𝑦21.76901𝑥𝑦0.00106244𝑥2superscript𝑦33.62336superscript𝑦20.00053122𝑦2absent\displaystyle\left(-8x^{3}-8x^{2}y-10.9554x^{2}+2xy^{2}+1.76901xy-0.00106244x+2y^{3}+3.62336y^{2}-0.00053122y\right)^{2}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | x⋅4.168614250⋅10−7−y2​2.049216091⋅10−7−0.0279456​x5+⋅𝑥4.168614250superscript107⋅superscript𝑦22.049216091superscript107limit-from0.0279456superscript𝑥5\displaystyle x\cdot 4.168614250\cdot 10^{-7}-y^{2}2.049216091\cdot 10^{-7}-0.0279456x^{5}+ |  |
|  |  |  |
| --- | --- | --- |
|  | 43.0875​x4​y+30.8113​x4+43.1084​x3​y2+68.989​x3​y+41.6357​x3+10.7928​x2​y3−13.1726​x2​y2−43.0875superscript𝑥4𝑦30.8113superscript𝑥443.1084superscript𝑥3superscript𝑦268.989superscript𝑥3𝑦41.6357superscript𝑥310.7928superscript𝑥2superscript𝑦3limit-from13.1726superscript𝑥2superscript𝑦2\displaystyle 43.0875x^{4}y+30.8113x^{4}+43.1084x^{3}y^{2}+68.989x^{3}y+41.6357x^{3}+10.7928x^{2}y^{3}-13.1726x^{2}y^{2}- |  |
|  |  |  |
| --- | --- | --- |
|  | 27.8148​x2​y−0.00833715​x2+0.0139728​x​y4+5.47537​x​y3+27.8148superscript𝑥2𝑦0.00833715superscript𝑥20.0139728𝑥superscript𝑦4limit-from5.47537𝑥superscript𝑦3\displaystyle 27.8148x^{2}y-0.00833715x^{2}+0.0139728xy^{4}+5.47537xy^{3}+ |  |
|  |  |  |
| --- | --- | --- |
|  | 4.65089​x​y2+0.00277916​x​y−10.7858​y5−12.2664​y4+0.00436492​y3.4.65089𝑥superscript𝑦20.00277916𝑥𝑦10.7858superscript𝑦512.2664superscript𝑦40.00436492superscript𝑦3\displaystyle 4.65089xy^{2}+0.00277916xy-10.7858y^{5}-12.2664y^{4}+0.00436492y^{3}\ . |  |

We obtain the inequalities:

|  |  |  |  |
| --- | --- | --- | --- |
|  | x⋅4.168614250⋅10−7−y2​2.049216091⋅10−7−0.0279456​x5+⋅𝑥4.168614250superscript107⋅superscript𝑦22.049216091superscript107limit-from0.0279456superscript𝑥5\displaystyle x\cdot 4.168614250\cdot 10^{-7}-y^{2}2.049216091\cdot 10^{-7}-0.0279456x^{5}+ |  | (295) |
|  |  |  |
| --- | --- | --- |
|  | 43.0875​x4​y+30.8113​x4+43.1084​x3​y2+68.989​x3​y+41.6357​x3+10.7928​x2​y3−43.0875superscript𝑥4𝑦30.8113superscript𝑥443.1084superscript𝑥3superscript𝑦268.989superscript𝑥3𝑦41.6357superscript𝑥3limit-from10.7928superscript𝑥2superscript𝑦3\displaystyle 43.0875x^{4}y+30.8113x^{4}+43.1084x^{3}y^{2}+68.989x^{3}y+41.6357x^{3}+10.7928x^{2}y^{3}- |  |
|  |  |  |
| --- | --- | --- |
|  | 13.1726​x2​y2−27.8148​x2​y−0.00833715​x2+13.1726superscript𝑥2superscript𝑦227.8148superscript𝑥2𝑦limit-from0.00833715superscript𝑥2\displaystyle 13.1726x^{2}y^{2}-27.8148x^{2}y-0.00833715x^{2}+ |  |
|  |  |  |
| --- | --- | --- |
|  | 0.0139728​x​y4+5.47537​x​y3+4.65089​x​y2+0.00277916​x​y−10.7858​y5−12.2664​y4+0.00436492​y3>0.0139728𝑥superscript𝑦45.47537𝑥superscript𝑦34.65089𝑥superscript𝑦20.00277916𝑥𝑦10.7858superscript𝑦512.2664superscript𝑦40.00436492superscript𝑦3absent\displaystyle 0.0139728xy^{4}+5.47537xy^{3}+4.65089xy^{2}+0.00277916xy-10.7858y^{5}-12.2664y^{4}+0.00436492y^{3}\ > |  |
|  |  |  |
| --- | --- | --- |
|  | x⋅4.168614250⋅10−7−(0.01)2​2.049216091⋅10−7−0.0279456​x5+⋅𝑥4.168614250superscript107⋅superscript0.0122.049216091superscript107limit-from0.0279456superscript𝑥5\displaystyle x\cdot 4.168614250\cdot 10^{-7}-(0.01)^{2}2.049216091\cdot 10^{-7}-0.0279456x^{5}+ |  |
|  |  |  |
| --- | --- | --- |
|  | 0.0⋅43.0875​x4+30.8113​x4+43.1084​(0.0)2​x3+0.0⋅68.989​x3+41.6357​x3+⋅0.043.0875superscript𝑥430.8113superscript𝑥443.1084superscript0.02superscript𝑥3⋅0.068.989superscript𝑥3limit-from41.6357superscript𝑥3\displaystyle 0.0\cdot 43.0875x^{4}+30.8113x^{4}+43.1084(0.0)^{2}x^{3}+0.0\cdot 68.989x^{3}+41.6357x^{3}+ |  |
|  |  |  |
| --- | --- | --- |
|  | 10.7928​(0.0)3​x2−13.1726​(0.01)2​x2−27.8148​(0.01)​x2−0.00833715​x2+10.7928superscript0.03superscript𝑥213.1726superscript0.012superscript𝑥227.81480.01superscript𝑥2limit-from0.00833715superscript𝑥2\displaystyle 10.7928(0.0)^{3}x^{2}-13.1726(0.01)^{2}x^{2}-27.8148(0.01)x^{2}-0.00833715x^{2}+ |  |
|  |  |  |
| --- | --- | --- |
|  | 0.0139728​(0.0)4​x+5.47537​(0.0)3​x+4.65089​(0.0)2​x+0.0139728superscript0.04𝑥5.47537superscript0.03𝑥limit-from4.65089superscript0.02𝑥\displaystyle 0.0139728(0.0)^{4}x+5.47537(0.0)^{3}x+4.65089(0.0)^{2}x+ |  |
|  |  |  |
| --- | --- | --- |
|  | 0.0⋅0.00277916​x−10.7858​(0.01)5−12.2664​(0.01)4+0.00436492​(0.0)3=⋅0.00.00277916𝑥10.7858superscript0.01512.2664superscript0.0140.00436492superscript0.03absent\displaystyle 0.0\cdot 0.00277916x-10.7858(0.01)^{5}-12.2664(0.01)^{4}+0.00436492(0.0)^{3}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | x⋅4.168614250⋅10−7−1.237626189⋅10−7−0.0279456​x5+30.8113​x4+41.6357​x3−0.287802​x2>⋅𝑥4.168614250superscript107⋅1.237626189superscript1070.0279456superscript𝑥530.8113superscript𝑥441.6357superscript𝑥30.287802superscript𝑥2absent\displaystyle x\cdot 4.168614250\cdot 10^{-7}-1.237626189\cdot 10^{-7}-0.0279456x^{5}+30.8113x^{4}+41.6357x^{3}-0.287802x^{2}\ > |  |
|  |  |  |
| --- | --- | --- |
|  | −(x0.007)3​1.237626189⋅10−7+30.8113​x4−(0.875)⋅0.0279456​x4+41.6357​x3−(0.287802​x)​x20.007=⋅superscript𝑥0.00731.237626189superscript10730.8113superscript𝑥4⋅0.8750.0279456superscript𝑥441.6357superscript𝑥30.287802𝑥superscript𝑥20.007absent\displaystyle-\left(\frac{x}{0.007}\right)^{3}1.237626189\cdot 10^{-7}+30.8113x^{4}-(0.875)\cdot 0.0279456x^{4}+41.6357x^{3}-\frac{(0.287802x)x^{2}}{0.007}\ = |  |
|  |  |  |
| --- | --- | --- |
|  | 30.7869​x4+0.160295​x3> 0.30.7869superscript𝑥40.160295superscript𝑥3 0\displaystyle 30.7869x^{4}+0.160295x^{3}\ >\ 0\ . |  |

We used x⩾0.007𝑥0.007x\geqslant 0.007 and x⩽0.875𝑥0.875x\leqslant 0.875 (reducing the negative x4superscript𝑥4x^{4}-term to a
x3superscript𝑥3x^{3}-term).
We have proofed the last inequality >0absent0>0 of Eq. ([289](#S3.E289 "In Proof. ‣ Behavior of the main subfunction with respect to 𝑦 at minimal 𝑥. ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")).

Consequently the derivative is always positive independent of y𝑦y,
thus

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(x+y)22​x​erfc⁡(x+y2​x)−2​e(2​x+y)22​x​erfc⁡(2​x+y2​x)superscript𝑒superscript𝑥𝑦22𝑥erfc𝑥𝑦2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥erfc2𝑥𝑦2𝑥\displaystyle e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)-2e^{\frac{(2x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right) |  | (296) |

is strictly monotonically increasing in x𝑥x.

Next we show that the
sub-function Eq. ([125](#S3.E125 "In Lemma 17 (Main subfunction Below). ‣ Main Sub-Function From Below. ‣ A3.4.4 Lemmata for proofing Theorem 3: The variance is expanding ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) is smaller
than zero.
We consider the limit:

|  |  |  |  |
| --- | --- | --- | --- |
|  | limx→∞e(x+y)22​x​erfc⁡(x+y2​x)− 2​e(2​x+y)22​x​erfc⁡(2​x+y2​x)= 0subscript→𝑥superscript𝑒superscript𝑥𝑦22𝑥erfc𝑥𝑦2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥erfc2𝑥𝑦2𝑥 0\displaystyle\lim\_{x\to\infty}e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)\ -\ 2e^{\frac{(2x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right)\ =\ 0 |  | (297) |

The limit follows from Lemma [22](#Thmtheorem22 "Lemma 22 (Erfc bound from Abramowitz). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").
Since the function is monotonic increasing in x𝑥x, it has to approach
00 from below. Thus,

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(x+y)22​x​erfc⁡(x+y2​x)−2​e(2​x+y)22​x​erfc⁡(2​x+y2​x)superscript𝑒superscript𝑥𝑦22𝑥erfc𝑥𝑦2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥erfc2𝑥𝑦2𝑥\displaystyle e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)-2e^{\frac{(2x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right) |  | (298) |

is smaller than zero.

We now consider the derivative of sub-function
Eq. ([125](#S3.E125 "In Lemma 17 (Main subfunction Below). ‣ Main Sub-Function From Below. ‣ A3.4.4 Lemmata for proofing Theorem 3: The variance is expanding ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) with respect to y𝑦y.
We proofed that sub-function
Eq. ([125](#S3.E125 "In Lemma 17 (Main subfunction Below). ‣ Main Sub-Function From Below. ‣ A3.4.4 Lemmata for proofing Theorem 3: The variance is expanding ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) is strictly monotonically increasing
independent of y𝑦y.
In the proof of Theorem [3](#Thmtheorem3 "Theorem 3 (Increasing 𝜈). ‣ Stable and Attracting Fixed Points for Unnormalized Weights. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"), we need the minimum
of sub-function
Eq. ([125](#S3.E125 "In Lemma 17 (Main subfunction Below). ‣ Main Sub-Function From Below. ‣ A3.4.4 Lemmata for proofing Theorem 3: The variance is expanding ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")). First, we are interested in the
derivative of sub-function
Eq. ([125](#S3.E125 "In Lemma 17 (Main subfunction Below). ‣ Main Sub-Function From Below. ‣ A3.4.4 Lemmata for proofing Theorem 3: The variance is expanding ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) with respect to y𝑦y
for the minimum x=0.007=7/1000𝑥0.00771000x=0.007=7/1000.

Consequently, we insert the minimum x=0.007=7/1000𝑥0.00771000x=0.007=7/1000 into the sub-function
Eq. ([125](#S3.E125 "In Lemma 17 (Main subfunction Below). ‣ Main Sub-Function From Below. ‣ A3.4.4 Lemmata for proofing Theorem 3: The variance is expanding ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(y2​71000+710002)2​erfc⁡(y2​71000+710002)−limit-fromsuperscript𝑒superscript𝑦2710007100022erfc𝑦271000710002\displaystyle e^{\left(\frac{y}{\sqrt{2}\sqrt{\frac{7}{1000}}}+\frac{\sqrt{\frac{7}{1000}}}{\sqrt{2}}\right)^{2}}\operatorname{erfc}\left(\frac{y}{\sqrt{2}\sqrt{\frac{7}{1000}}}+\frac{\sqrt{\frac{7}{1000}}}{\sqrt{2}}\right)- |  | (299) |
|  |  |  |
| --- | --- | --- |
|  | 2​e(y2​71000+2​71000)2​erfc⁡(y2​71000+2​71000)=2superscript𝑒superscript𝑦2710002710002erfc𝑦271000271000absent\displaystyle 2e^{\left(\frac{y}{\sqrt{2}\sqrt{\frac{7}{1000}}}+\sqrt{2}\sqrt{\frac{7}{1000}}\right)^{2}}\operatorname{erfc}\left(\frac{y}{\sqrt{2}\sqrt{\frac{7}{1000}}}+\sqrt{2}\sqrt{\frac{7}{1000}}\right)\ = |  |
|  |  |  |
| --- | --- | --- |
|  | e500​y27+y+72000​erfc⁡(1000​y+720​35)−2​e(500​y+7)23500​erfc⁡(500​y+710​35).superscript𝑒500superscript𝑦27𝑦72000erfc1000𝑦720352superscript𝑒superscript500𝑦723500erfc500𝑦71035\displaystyle e^{\frac{500y^{2}}{7}+y+\frac{7}{2000}}\operatorname{erfc}\left(\frac{1000y+7}{20\sqrt{35}}\right)-2e^{\frac{(500y+7)^{2}}{3500}}\operatorname{erfc}\left(\frac{500y+7}{10\sqrt{35}}\right)\ . |  |

The derivative of this function with respect to y𝑦y is

|  |  |  |  |
| --- | --- | --- | --- |
|  | (1000​y7+1)​e500​y27+y+72000​erfc⁡(1000​y+720​35)−limit-from1000𝑦71superscript𝑒500superscript𝑦27𝑦72000erfc1000𝑦72035\displaystyle\left(\frac{1000y}{7}+1\right)e^{\frac{500y^{2}}{7}+y+\frac{7}{2000}}\operatorname{erfc}\left(\frac{1000y+7}{20\sqrt{35}}\right)- |  | (300) |
|  |  |  |
| --- | --- | --- |
|  | 17​4​e(500​y+7)23500​(500​y+7)​erfc⁡(500​y+710​35)+20​57​π>174superscript𝑒superscript500𝑦723500500𝑦7erfc500𝑦710352057𝜋absent\displaystyle\frac{1}{7}4e^{\frac{(500y+7)^{2}}{3500}}(500y+7)\operatorname{erfc}\left(\frac{500y+7}{10\sqrt{35}}\right)+20\sqrt{\frac{5}{7\pi}}\ > |  |
|  |  |  |
| --- | --- | --- |
|  | (1+1000⋅(−0.01)7)​e−0.01+72000+500⋅(−0.01)27​erfc⁡(7+1000+(−0.01)20​35)−limit-from1⋅10000.017superscript𝑒0.0172000⋅500superscript0.0127erfc710000.012035\displaystyle\left(1+\frac{1000\cdot(-0.01)}{7}\right)e^{-0.01+\frac{7}{2000}+\frac{500\cdot(-0.01)^{2}}{7}}\operatorname{erfc}\left(\frac{7+1000+(-0.01)}{20\sqrt{35}}\right)- |  |
|  |  |  |
| --- | --- | --- |
|  | 17​4​e(7+500⋅0.01)23500​(7+500⋅0.01)​erfc⁡(7+500⋅0.0110​35)+20​57​π> 3.56.174superscript𝑒superscript7⋅5000.01235007⋅5000.01erfc7⋅5000.0110352057𝜋3.56\displaystyle\frac{1}{7}4e^{\frac{(7+500\cdot 0.01)^{2}}{3500}}(7+500\cdot 0.01)\operatorname{erfc}\left(\frac{7+500\cdot 0.01}{10\sqrt{35}}\right)+20\sqrt{\frac{5}{7\pi}}\ >\ 3.56\ . |  |

For the first inequality, we use Lemma [24](#Thmtheorem24 "Lemma 24 (Properties of 𝑥⁢𝑒^𝑥²⁢erfc(𝑥)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").
Lemma [24](#Thmtheorem24 "Lemma 24 (Properties of 𝑥⁢𝑒^𝑥²⁢erfc(𝑥)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") says that
the function x​ex2​erfc⁡(x)𝑥superscript𝑒superscript𝑥2erfc𝑥xe^{x^{2}}\operatorname{erfc}(x) has the sign of x𝑥x and is
monotonically increasing to 1π1𝜋\frac{1}{\sqrt{\pi}}.
Consequently, we inserted the maximal y=0.01𝑦0.01y=0.01 to
make the negative term more negative and the minimal y=−0.01𝑦0.01y=-0.01
to make the positive term less positive.

Consequently

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(x+y)22​x​erfc⁡(x+y2​x)−2​e(2​x+y)22​x​erfc⁡(2​x+y2​x)superscript𝑒superscript𝑥𝑦22𝑥erfc𝑥𝑦2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥erfc2𝑥𝑦2𝑥\displaystyle e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)-2e^{\frac{(2x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right) |  | (301) |

is strictly monotonically increasing in y𝑦y for the minimal
x=0.007𝑥0.007x=0.007.

Next, we consider x=0.7⋅0.8=0.56𝑥⋅0.70.80.56x=0.7\cdot 0.8=0.56, which is the maximal ν=0.7𝜈0.7\nu=0.7
and minimal τ=0.8𝜏0.8\tau=0.8.
We insert the minimum x=0.56=56/100𝑥0.5656100x=0.56=56/100 into the sub-function
Eq. ([125](#S3.E125 "In Lemma 17 (Main subfunction Below). ‣ Main Sub-Function From Below. ‣ A3.4.4 Lemmata for proofing Theorem 3: The variance is expanding ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(y2​56100+561002)2​erfc⁡(y2​56100+561002)−limit-fromsuperscript𝑒superscript𝑦2561005610022erfc𝑦256100561002\displaystyle e^{\left(\frac{y}{\sqrt{2}\sqrt{\frac{56}{100}}}+\frac{\sqrt{\frac{56}{100}}}{\sqrt{2}}\right)^{2}}\operatorname{erfc}\left(\frac{y}{\sqrt{2}\sqrt{\frac{56}{100}}}+\frac{\sqrt{\frac{56}{100}}}{\sqrt{2}}\right)- |  | (302) |
|  |  |  |
| --- | --- | --- |
|  | 2​e(y2​56100+2​56100)2​erfc⁡(y2​56100+2​56100).2superscript𝑒superscript𝑦2561002561002erfc𝑦256100256100\displaystyle 2e^{\left(\frac{y}{\sqrt{2}\sqrt{\frac{56}{100}}}+\sqrt{2}\sqrt{\frac{56}{100}}\right)^{2}}\operatorname{erfc}\left(\frac{y}{\sqrt{2}\sqrt{\frac{56}{100}}}+\sqrt{2}\sqrt{\frac{56}{100}}\right)\ . |  |

The derivative with respect to y𝑦y is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 5​e(5​y2​7+75)2​(5​y2​7+75)​erfc⁡(5​y2​7+75)7−limit-from5superscript𝑒superscript5𝑦277525𝑦2775erfc5𝑦27757\displaystyle\frac{5e^{\left(\frac{5y}{2\sqrt{7}}+\frac{\sqrt{7}}{5}\right)^{2}}\left(\frac{5y}{2\sqrt{7}}+\frac{\sqrt{7}}{5}\right)\operatorname{erfc}\left(\frac{5y}{2\sqrt{7}}+\frac{\sqrt{7}}{5}\right)}{\sqrt{7}}- |  | (303) |
|  |  |  |
| --- | --- | --- |
|  | 10​e(5​y2​7+2​75)2​(5​y2​7+2​75)​erfc⁡(5​y2​7+2​75)7+57​π>10superscript𝑒superscript5𝑦2727525𝑦27275erfc5𝑦27275757𝜋absent\displaystyle\frac{10e^{\left(\frac{5y}{2\sqrt{7}}+\frac{2\sqrt{7}}{5}\right)^{2}}\left(\frac{5y}{2\sqrt{7}}+\frac{2\sqrt{7}}{5}\right)\operatorname{erfc}\left(\frac{5y}{2\sqrt{7}}+\frac{2\sqrt{7}}{5}\right)}{\sqrt{7}}+\frac{5}{\sqrt{7\pi}}\ >\ |  |
|  |  |  |
| --- | --- | --- |
|  | 5​e(75−0.01⋅52​7)2​(75−0.01⋅52​7)​erfc⁡(75−0.01⋅52​7)7−limit-from5superscript𝑒superscript75⋅0.01527275⋅0.01527erfc75⋅0.015277\displaystyle\frac{5e^{\left(\frac{\sqrt{7}}{5}-\frac{0.01\cdot 5}{2\sqrt{7}}\right)^{2}}\left(\frac{\sqrt{7}}{5}-\frac{0.01\cdot 5}{2\sqrt{7}}\right)\operatorname{erfc}\left(\frac{\sqrt{7}}{5}-\frac{0.01\cdot 5}{2\sqrt{7}}\right)}{\sqrt{7}}- |  |
|  |  |  |
| --- | --- | --- |
|  | 10​e(2​75+0.01⋅52​7)2​(2​75+0.01⋅52​7)​erfc⁡(2​75+0.01⋅52​7)7+57​π> 0.00746.10superscript𝑒superscript275⋅0.015272275⋅0.01527erfc275⋅0.01527757𝜋0.00746\displaystyle\frac{10e^{\left(\frac{2\sqrt{7}}{5}+\frac{0.01\cdot 5}{2\sqrt{7}}\right)^{2}}\left(\frac{2\sqrt{7}}{5}+\frac{0.01\cdot 5}{2\sqrt{7}}\right)\operatorname{erfc}\left(\frac{2\sqrt{7}}{5}+\frac{0.01\cdot 5}{2\sqrt{7}}\right)}{\sqrt{7}}+\frac{5}{\sqrt{7\pi}}\ >\ 0.00746\ . |  |

For the first inequality we applied Lemma [24](#Thmtheorem24 "Lemma 24 (Properties of 𝑥⁢𝑒^𝑥²⁢erfc(𝑥)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
which states that the function x​ex2​erfc⁡(x)𝑥superscript𝑒superscript𝑥2erfc𝑥xe^{x^{2}}\operatorname{erfc}(x) is
monotonically increasing.
Consequently, we inserted the maximal y=0.01𝑦0.01y=0.01 to
make the negative term more negative and the minimal y=−0.01𝑦0.01y=-0.01
to make the positive term less positive.

Consequently

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(x+y)22​x​erfc⁡(x+y2​x)−2​e(2​x+y)22​x​erfc⁡(2​x+y2​x)superscript𝑒superscript𝑥𝑦22𝑥erfc𝑥𝑦2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥erfc2𝑥𝑦2𝑥\displaystyle e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)-2e^{\frac{(2x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right) |  | (304) |

is strictly monotonically increasing in y𝑦y for x=0.56𝑥0.56x=0.56.

Next, we consider x=0.16⋅0.8=0.128𝑥⋅0.160.80.128x=0.16\cdot 0.8=0.128, which is the minimal τ=0.8𝜏0.8\tau=0.8.
We insert the minimum x=0.128=128/1000𝑥0.1281281000x=0.128=128/1000 into the sub-function
Eq. ([125](#S3.E125 "In Lemma 17 (Main subfunction Below). ‣ Main Sub-Function From Below. ‣ A3.4.4 Lemmata for proofing Theorem 3: The variance is expanding ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(y2​1281000+12810002)2​erfc⁡(y2​1281000+12810002)−limit-fromsuperscript𝑒superscript𝑦21281000128100022erfc𝑦2128100012810002\displaystyle e^{\left(\frac{y}{\sqrt{2}\sqrt{\frac{128}{1000}}}+\frac{\sqrt{\frac{128}{1000}}}{\sqrt{2}}\right)^{2}}\operatorname{erfc}\left(\frac{y}{\sqrt{2}\sqrt{\frac{128}{1000}}}+\frac{\sqrt{\frac{128}{1000}}}{\sqrt{2}}\right)- |  | (305) |
|  |  |  |
| --- | --- | --- |
|  | 2​e(y2​1281000+2​1281000)2​erfc⁡(y2​1281000+2​1281000)=2superscript𝑒superscript𝑦21281000212810002erfc𝑦2128100021281000absent\displaystyle 2e^{\left(\frac{y}{\sqrt{2}\sqrt{\frac{128}{1000}}}+\sqrt{2}\sqrt{\frac{128}{1000}}\right)^{2}}\operatorname{erfc}\left(\frac{y}{\sqrt{2}\sqrt{\frac{128}{1000}}}+\sqrt{2}\sqrt{\frac{128}{1000}}\right)= |  |
|  |  |  |
| --- | --- | --- |
|  | e125​y232+y+8125​erfc⁡(125​y+1620​10)−2​e(125​y+32)24000​erfc⁡(125​y+3220​10).superscript𝑒125superscript𝑦232𝑦8125erfc125𝑦1620102superscript𝑒superscript125𝑦3224000erfc125𝑦322010\displaystyle e^{\frac{125y^{2}}{32}+y+\frac{8}{125}}\operatorname{erfc}\left(\frac{125y+16}{20\sqrt{10}}\right)-2e^{\frac{(125y+32)^{2}}{4000}}\operatorname{erfc}\left(\frac{125y+32}{20\sqrt{10}}\right)\ . |  |

The derivative with respect to y𝑦y is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 116(e125​y232+y+8125(125y+16)erfc(125​y+1620​10)−\displaystyle\frac{1}{16}\left(e^{\frac{125y^{2}}{32}+y+\frac{8}{125}}(125y+16)\operatorname{erfc}\left(\frac{125y+16}{20\sqrt{10}}\right)-\right. |  | (306) |
|  |  |  |
| --- | --- | --- |
|  | 2e(125​y+32)24000(125y+32)erfc(125​y+3220​10)+2010π)>\displaystyle\left.2e^{\frac{(125y+32)^{2}}{4000}}(125y+32)\operatorname{erfc}\left(\frac{125y+32}{20\sqrt{10}}\right)+20\sqrt{\frac{10}{\pi}}\right)\ >\ |  |
|  |  |  |
| --- | --- | --- |
|  | 116((16+125(−0.01))e−0.01+8125+125​(−0.01)232erfc(16+125​(−0.01)20​10)−\displaystyle\frac{1}{16}\left((16+125(-0.01))e^{-0.01+\frac{8}{125}+\frac{125(-0.01)^{2}}{32}}\operatorname{erfc}\left(\frac{16+125(-0.01)}{20\sqrt{10}}\right)-\right. |  |
|  |  |  |
| --- | --- | --- |
|  | 2e(32+1250.01)24000(32+1250.01)erfc(32+1250.0120​10)+2010π)> 0.4468.\displaystyle\left.2e^{\frac{(32+1250.01)^{2}}{4000}}(32+1250.01)\operatorname{erfc}\left(\frac{32+1250.01}{20\sqrt{10}}\right)+20\sqrt{\frac{10}{\pi}}\right)\ >\ 0.4468\ . |  |

For the first inequality we applied Lemma [24](#Thmtheorem24 "Lemma 24 (Properties of 𝑥⁢𝑒^𝑥²⁢erfc(𝑥)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
which states that the function x​ex2​erfc⁡(x)𝑥superscript𝑒superscript𝑥2erfc𝑥xe^{x^{2}}\operatorname{erfc}(x) is
monotonically increasing.
Consequently, we inserted the maximal y=0.01𝑦0.01y=0.01 to
make the negative term more negative and the minimal y=−0.01𝑦0.01y=-0.01
to make the positive term less positive.

Consequently

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(x+y)22​x​erfc⁡(x+y2​x)−2​e(2​x+y)22​x​erfc⁡(2​x+y2​x)superscript𝑒superscript𝑥𝑦22𝑥erfc𝑥𝑦2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥erfc2𝑥𝑦2𝑥\displaystyle e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)-2e^{\frac{(2x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right) |  | (307) |

is strictly monotonically increasing in y𝑦y for x=0.128𝑥0.128x=0.128.

Next, we consider x=0.24⋅0.9=0.216𝑥⋅0.240.90.216x=0.24\cdot 0.9=0.216, which is the minimal
τ=0.9𝜏0.9\tau=0.9 (here we consider 0.90.90.9 as lower bound for τ𝜏\tau).
We insert the minimum x=0.216=216/1000𝑥0.2162161000x=0.216=216/1000 into the sub-function
Eq. ([125](#S3.E125 "In Lemma 17 (Main subfunction Below). ‣ Main Sub-Function From Below. ‣ A3.4.4 Lemmata for proofing Theorem 3: The variance is expanding ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(y2​2161000+21610002)2​erfc⁡(y2​2161000+21610002)−limit-fromsuperscript𝑒superscript𝑦22161000216100022erfc𝑦2216100021610002\displaystyle e^{\left(\frac{y}{\sqrt{2}\sqrt{\frac{216}{1000}}}+\frac{\sqrt{\frac{216}{1000}}}{\sqrt{2}}\right)^{2}}\operatorname{erfc}\left(\frac{y}{\sqrt{2}\sqrt{\frac{216}{1000}}}+\frac{\sqrt{\frac{216}{1000}}}{\sqrt{2}}\right)- |  | (308) |
|  |  |  |
| --- | --- | --- |
|  | 2​e(y2​2161000+2​2161000)2​erfc⁡(y2​2161000+2​2161000)=2superscript𝑒superscript𝑦22161000221610002erfc𝑦2216100022161000absent\displaystyle 2e^{\left(\frac{y}{\sqrt{2}\sqrt{\frac{216}{1000}}}+\sqrt{2}\sqrt{\frac{216}{1000}}\right)^{2}}\operatorname{erfc}\left(\frac{y}{\sqrt{2}\sqrt{\frac{216}{1000}}}+\sqrt{2}\sqrt{\frac{216}{1000}}\right)= |  |
|  |  |  |
| --- | --- | --- |
|  | e(125​y+27)26750​erfc⁡(125​y+2715​30)−2​e(125​y+54)26750​erfc⁡(125​y+5415​30)superscript𝑒superscript125𝑦2726750erfc125𝑦2715302superscript𝑒superscript125𝑦5426750erfc125𝑦541530\displaystyle e^{\frac{(125y+27)^{2}}{6750}}\operatorname{erfc}\left(\frac{125y+27}{15\sqrt{30}}\right)-2e^{\frac{(125y+54)^{2}}{6750}}\operatorname{erfc}\left(\frac{125y+54}{15\sqrt{30}}\right) |  |

The derivative with respect to y𝑦y is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 127(e(125​y+27)26750(125y+27)erfc(125​y+2715​30)−\displaystyle\frac{1}{27}\left(e^{\frac{(125y+27)^{2}}{6750}}(125y+27)\operatorname{erfc}\left(\frac{125y+27}{15\sqrt{30}}\right)-\right. |  | (309) |
|  |  |  |
| --- | --- | --- |
|  | 2e(125​y+54)26750(125y+54)erfc(125​y+5415​30)+1530π)>\displaystyle\left.2e^{\frac{(125y+54)^{2}}{6750}}(125y+54)\operatorname{erfc}\left(\frac{125y+54}{15\sqrt{30}}\right)+15\sqrt{\frac{30}{\pi}}\right)\ >\ |  |
|  |  |  |
| --- | --- | --- |
|  | 127((27+125(−0.01))e(27+125​(−0.01))26750erfc(27+125​(−0.01)15​30)−\displaystyle\frac{1}{27}\left((27+125(-0.01))e^{\frac{(27+125(-0.01))^{2}}{6750}}\operatorname{erfc}\left(\frac{27+125(-0.01)}{15\sqrt{30}}\right)-\right. |  |
|  |  |  |
| --- | --- | --- |
|  | 2e(54+1250.01)26750(54+1250.01)erfc(54+1250.0115​30)+1530π))> 0.211288.\displaystyle\left.2e^{\frac{(54+1250.01)^{2}}{6750}}(54+1250.01)\operatorname{erfc}\left(\frac{54+1250.01}{15\sqrt{30}}\right)+15\sqrt{\frac{30}{\pi}}\right))\ >\ 0.211288\ . |  |

For the first inequality we applied Lemma [24](#Thmtheorem24 "Lemma 24 (Properties of 𝑥⁢𝑒^𝑥²⁢erfc(𝑥)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
which states that the function x​ex2​erfc⁡(x)𝑥superscript𝑒superscript𝑥2erfc𝑥xe^{x^{2}}\operatorname{erfc}(x) is
monotonically increasing.
Consequently, we inserted the maximal y=0.01𝑦0.01y=0.01 to
make the negative term more negative and the minimal y=−0.01𝑦0.01y=-0.01
to make the positive term less positive.

Consequently

|  |  |  |  |
| --- | --- | --- | --- |
|  | e(x+y)22​x​erfc⁡(x+y2​x)−2​e(2​x+y)22​x​erfc⁡(2​x+y2​x)superscript𝑒superscript𝑥𝑦22𝑥erfc𝑥𝑦2𝑥2superscript𝑒superscript2𝑥𝑦22𝑥erfc2𝑥𝑦2𝑥\displaystyle e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)-2e^{\frac{(2x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{2x+y}{\sqrt{2}\sqrt{x}}\right) |  | (310) |

is strictly monotonically increasing in y𝑦y for x=0.216𝑥0.216x=0.216.
∎

###### Lemma 46 (Monotone Derivative).

For λ=λ01𝜆subscript𝜆01\lambda=\lambda\_{\rm 01}, α=α01𝛼subscript𝛼01\alpha=\alpha\_{\rm 01}
and the domain
−0.1⩽μ⩽0.10.1𝜇0.1-0.1\leqslant\mu\leqslant 0.1,
−0.1⩽ω⩽0.10.1𝜔0.1-0.1\leqslant\omega\leqslant 0.1,
0.00875⩽ν⩽0.70.00875𝜈0.70.00875\leqslant\nu\leqslant 0.7, and
0.8⩽τ⩽1.250.8𝜏1.250.8\leqslant\tau\leqslant 1.25.
We are interested of the derivative of

|  |  |  |  |
| --- | --- | --- | --- |
|  | τ​(e(μ​ω+ν​τ2​ν​τ)2​erfc⁡(μ​ω+ν​τ2​ν​τ)−2​e(μ​ω+2⋅ν​τ2​ν​τ)2​erfc⁡(μ​ω+2⋅ν​τ2​ν​τ)).𝜏superscript𝑒superscript𝜇𝜔𝜈𝜏2𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏2superscript𝑒superscript𝜇𝜔⋅2𝜈𝜏2𝜈𝜏2erfc𝜇𝜔⋅2𝜈𝜏2𝜈𝜏\displaystyle\tau\left(e^{\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-2e^{\left(\frac{\mu\omega+2\cdot\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+2\cdot\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\right)\ . |  | (311) |

The derivative of the equation above with
respect to

* •

  ν𝜈\nu is larger than zero;
* •

  τ𝜏\tau is smaller than zero for maximal
  ν=0.7𝜈0.7\nu=0.7, ν=0.16𝜈0.16\nu=0.16, and ν=0.24𝜈0.24\nu=0.24 (with
  0.9⩽τ0.9𝜏0.9\leqslant\tau);
* •

  y=μ​ω𝑦𝜇𝜔y=\mu\omega is larger than zero for ν​τ=0.00875⋅0.8=0.007𝜈𝜏⋅0.008750.80.007\nu\tau=0.00875\cdot 0.8=0.007, ν​τ=0.7⋅0.8=0.56𝜈𝜏⋅0.70.80.56\nu\tau=0.7\cdot 0.8=0.56, ν​τ=0.16⋅0.8=0.128𝜈𝜏⋅0.160.80.128\nu\tau=0.16\cdot 0.8=0.128, and ν​τ=0.24⋅0.9=0.216𝜈𝜏⋅0.240.90.216\nu\tau=0.24\cdot 0.9=0.216.

###### Proof.

We consider the domain:
−0.1⩽μ⩽0.10.1𝜇0.1-0.1\leqslant\mu\leqslant 0.1,
−0.1⩽ω⩽0.10.1𝜔0.1-0.1\leqslant\omega\leqslant 0.1,
0.00875⩽ν⩽0.70.00875𝜈0.70.00875\leqslant\nu\leqslant 0.7, and
0.8⩽τ⩽1.250.8𝜏1.250.8\leqslant\tau\leqslant 1.25.

We use Lemma [17](#Thmtheorem17 "Lemma 17 (Main subfunction Below). ‣ Main Sub-Function From Below. ‣ A3.4.4 Lemmata for proofing Theorem 3: The variance is expanding ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") to determine the derivatives.
Consequently, the derivative of

|  |  |  |  |
| --- | --- | --- | --- |
|  | τ​(e(μ​ω+ν​τ2​ν​τ)2​erfc⁡(μ​ω+ν​τ2​ν​τ)−2​e(μ​ω+2​ν​τ2​ν​τ)2​erfc⁡(μ​ω+2​ν​τ2​ν​τ))𝜏superscript𝑒superscript𝜇𝜔𝜈𝜏2𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏2superscript𝑒superscript𝜇𝜔2𝜈𝜏2𝜈𝜏2erfc𝜇𝜔2𝜈𝜏2𝜈𝜏\displaystyle\tau\left(e^{\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-2e^{\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\right) |  | (312) |

with respect to ν𝜈\nu is larger than zero, which follows
directly from Lemma [17](#Thmtheorem17 "Lemma 17 (Main subfunction Below). ‣ Main Sub-Function From Below. ‣ A3.4.4 Lemmata for proofing Theorem 3: The variance is expanding ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") using the chain rule.

Consequently, the derivative of

|  |  |  |  |
| --- | --- | --- | --- |
|  | τ​(e(μ​ω+ν​τ2​ν​τ)2​erfc⁡(μ​ω+ν​τ2​ν​τ)−2​e(μ​ω+2​ν​τ2​ν​τ)2​erfc⁡(μ​ω+2​ν​τ2​ν​τ))𝜏superscript𝑒superscript𝜇𝜔𝜈𝜏2𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏2superscript𝑒superscript𝜇𝜔2𝜈𝜏2𝜈𝜏2erfc𝜇𝜔2𝜈𝜏2𝜈𝜏\displaystyle\tau\left(e^{\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-2e^{\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\right) |  | (313) |

with respect to y=μ​ω𝑦𝜇𝜔y=\mu\omega is larger than zero for ν​τ=0.00875⋅0.8=0.007𝜈𝜏⋅0.008750.80.007\nu\tau=0.00875\cdot 0.8=0.007, ν​τ=0.7⋅0.8=0.56𝜈𝜏⋅0.70.80.56\nu\tau=0.7\cdot 0.8=0.56, ν​τ=0.16⋅0.8=0.128𝜈𝜏⋅0.160.80.128\nu\tau=0.16\cdot 0.8=0.128, and ν​τ=0.24⋅0.9=0.216𝜈𝜏⋅0.240.90.216\nu\tau=0.24\cdot 0.9=0.216,
which also follows
directly from Lemma [17](#Thmtheorem17 "Lemma 17 (Main subfunction Below). ‣ Main Sub-Function From Below. ‣ A3.4.4 Lemmata for proofing Theorem 3: The variance is expanding ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks").

We now consider the derivative with respect to τ𝜏\tau,
which is not trivial since τ𝜏\tau is a factor of the whole expression.
The sub-expression should be maximized as it appears with
negative sign in the mapping for ν𝜈\nu.

First,
we consider the function for
the largest ν=0.7𝜈0.7\nu=0.7 and the largest y=μ​ω=0.01𝑦𝜇𝜔0.01y=\mu\omega=0.01
for determining the derivative with respect to τ𝜏\tau.

The expression becomes

|  |  |  |  |
| --- | --- | --- | --- |
|  | τ​(e(7​τ10+11002​7​τ10)2​erfc⁡(7​τ10+11002​7​τ10)−2​e(2⋅7​τ10+11002​7​τ10)2​erfc⁡(2⋅7​τ10+11002​7​τ10)).𝜏superscript𝑒superscript7𝜏10110027𝜏102erfc7𝜏10110027𝜏102superscript𝑒superscript⋅27𝜏10110027𝜏102erfc⋅27𝜏10110027𝜏10\displaystyle\tau\left(e^{\left(\frac{\frac{7\tau}{10}+\frac{1}{100}}{\sqrt{2}\sqrt{\frac{7\tau}{10}}}\right)^{2}}\operatorname{erfc}\left(\frac{\frac{7\tau}{10}+\frac{1}{100}}{\sqrt{2}\sqrt{\frac{7\tau}{10}}}\right)-2e^{\left(\frac{\frac{2\cdot 7\tau}{10}+\frac{1}{100}}{\sqrt{2}\sqrt{\frac{7\tau}{10}}}\right)^{2}}\operatorname{erfc}\left(\frac{\frac{2\cdot 7\tau}{10}+\frac{1}{100}}{\sqrt{2}\sqrt{\frac{7\tau}{10}}}\right)\right)\ . |  | (314) |

The derivative with respect to τ𝜏\tau is

|  |  |  |  |
| --- | --- | --- | --- |
|  | (π(e(70​τ+1)214000​τ(700τ(7τ+20)−1)erfc(70​τ+120​35​τ)−\displaystyle\left(\sqrt{\pi}\left(e^{\frac{(70\tau+1)^{2}}{14000\tau}}(700\tau(7\tau+20)-1)\operatorname{erfc}\left(\frac{70\tau+1}{20\sqrt{35}\sqrt{\tau}}\right)\ -\right.\right. |  | (315) |
|  |  |  |
| --- | --- | --- |
|  | 2e(140​τ+1)214000​τ(2800τ(7τ+5)−1)erfc(140​τ+120​35​τ))+2035(210τ−1)τ)\displaystyle\left.\left.2e^{\frac{(140\tau+1)^{2}}{14000\tau}}(2800\tau(7\tau+5)-1)\operatorname{erfc}\left(\frac{140\tau+1}{20\sqrt{35}\sqrt{\tau}}\right)\right)+20\sqrt{35}(210\tau-1)\sqrt{\tau}\right) |  |
|  |  |  |
| --- | --- | --- |
|  | (14000​π​τ)−1.superscript14000𝜋𝜏1\displaystyle\left(14000\sqrt{\pi}\tau\right)^{-1}\ . |  |

We are considering only the numerator and use again the approximation
of Ren and MacKenzie, [[30](#bib.bib30)].
The error analysis on the whole numerator gives an approximation error 97<E<18697𝐸18697<E<186. Therefore
we add 200 to the numerator when we use the approximation Ren and MacKenzie, [[30](#bib.bib30)].
We obtain the inequalities:

|  |  |  |  |
| --- | --- | --- | --- |
|  | π(e(70​τ+1)214000​τ(700τ(7τ+20)−1)erfc(70​τ+120​35​τ)−\displaystyle\sqrt{\pi}\left(e^{\frac{(70\tau+1)^{2}}{14000\tau}}(700\tau(7\tau+20)-1)\operatorname{erfc}\left(\frac{70\tau+1}{20\sqrt{35}\sqrt{\tau}}\right)\ -\right. |  | (316) |
|  |  |  |
| --- | --- | --- |
|  | 2e(140​τ+1)214000​τ(2800τ(7τ+5)−1)erfc(140​τ+120​35​τ))+2035(210τ−1)τ⩽\displaystyle\left.2e^{\frac{(140\tau+1)^{2}}{14000\tau}}(2800\tau(7\tau+5)-1)\operatorname{erfc}\left(\frac{140\tau+1}{20\sqrt{35}\sqrt{\tau}}\right)\right)+20\sqrt{35}(210\tau-1)\sqrt{\tau}\ \leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | π(2.911​(700​τ​(7​τ+20)−1)π​(2.911−1)​(70​τ+1)20​35​τ+π​(70​τ+120​35​τ)2+2.9112−\displaystyle\sqrt{\pi}\left(\frac{2.911(700\tau(7\tau+20)-1)}{\frac{\sqrt{\pi}(2.911-1)(70\tau+1)}{20\sqrt{35}\sqrt{\tau}}+\sqrt{\pi\left(\frac{70\tau+1}{20\sqrt{35}\sqrt{\tau}}\right)^{2}+2.911^{2}}}\ -\right. |  |
|  |  |  |
| --- | --- | --- |
|  | 2⋅2.911​(2800​τ​(7​τ+5)−1)π​(2.911−1)​(140​τ+1)20​35​τ+π​(140​τ+120​35​τ)2+2.9112)\displaystyle\left.\frac{2\cdot 2.911(2800\tau(7\tau+5)-1)}{\frac{\sqrt{\pi}(2.911-1)(140\tau+1)}{20\sqrt{35}\sqrt{\tau}}+\sqrt{\pi\left(\frac{140\tau+1}{20\sqrt{35}\sqrt{\tau}}\right)^{2}+2.911^{2}}}\right) |  |
|  |  |  |
| --- | --- | --- |
|  | +20​35​(210​τ−1)​τ+200=2035210𝜏1𝜏200absent\displaystyle\ +20\sqrt{35}(210\tau-1)\sqrt{\tau}+200\ = |  |
|  |  |  |
| --- | --- | --- |
|  | π((700​τ​(7​τ+20)−1)​(20⋅35⋅2.911​τ)π​(2.911−1)​(70​τ+1)+(20⋅2.911​35​τ)2+π​(70​τ+1)2−\displaystyle\sqrt{\pi}\left(\frac{(700\tau(7\tau+20)-1)\left(20\cdot\sqrt{35}\cdot 2.911\sqrt{\tau}\right)}{\sqrt{\pi}(2.911-1)(70\tau+1)+\sqrt{\left(20\cdot 2.911\sqrt{35}\sqrt{\tau}\right)^{2}+\pi(70\tau+1)^{2}}}\ -\right. |  |
|  |  |  |
| --- | --- | --- |
|  | 2​(2800​τ​(7​τ+5)−1)​(20⋅35⋅2.911​τ)π​(2.911−1)​(140​τ+1)+(20⋅35⋅2.911​τ)2+π​(140​τ+1)2)+\displaystyle\left.\frac{2(2800\tau(7\tau+5)-1)\left(20\cdot\sqrt{35}\cdot 2.911\sqrt{\tau}\right)}{\sqrt{\pi}(2.911-1)(140\tau+1)+\sqrt{\left(20\cdot\sqrt{35}\cdot 2.911\sqrt{\tau}\right)^{2}+\pi(140\tau+1)^{2}}}\right)+ |  |
|  |  |  |
| --- | --- | --- |
|  | (20​35​(210​τ−1)​τ+200)=2035210𝜏1𝜏200absent\displaystyle\left(20\sqrt{35}(210\tau-1)\sqrt{\tau}+200\right)\ = |  |
|  |  |  |
| --- | --- | --- |
|  | ((2035(210τ−1)τ+200)(π(2.911−1)(70τ+1)+(20⋅35⋅2.911​τ)2+π​(70​τ+1)2)\displaystyle\left(\left(20\sqrt{35}(210\tau-1)\sqrt{\tau}+200\right)\left(\sqrt{\pi}(2.911-1)(70\tau+1)+\sqrt{\left(20\cdot\sqrt{35}\cdot 2.911\sqrt{\tau}\right)^{2}+\pi(70\tau+1)^{2}}\right)\right. |  |
|  |  |  |
| --- | --- | --- |
|  | (π​(2.911−1)​(140​τ+1)+(20⋅35⋅2.911​τ)2+π​(140​τ+1)2)+limit-from𝜋2.9111140𝜏1superscript⋅20352.911𝜏2𝜋superscript140𝜏12\displaystyle\left.\left(\sqrt{\pi}(2.911-1)(140\tau+1)+\sqrt{\left(20\cdot\sqrt{35}\cdot 2.911\sqrt{\tau}\right)^{2}+\pi(140\tau+1)^{2}}\right)+\right. |  |
|  |  |  |
| --- | --- | --- |
|  | 2.911⋅20​35​π​(700​τ​(7​τ+20)−1)​τ⋅2.9112035𝜋700𝜏7𝜏201𝜏\displaystyle\left.2.911\cdot 20\sqrt{35}\sqrt{\pi}(700\tau(7\tau+20)-1)\sqrt{\tau}\right. |  |
|  |  |  |
| --- | --- | --- |
|  | (π​(2.911−1)​(140​τ+1)+(20⋅35⋅2.911​τ)2+π​(140​τ+1)2)−limit-from𝜋2.9111140𝜏1superscript⋅20352.911𝜏2𝜋superscript140𝜏12\displaystyle\left.\left(\sqrt{\pi}(2.911-1)(140\tau+1)+\sqrt{\left(20\cdot\sqrt{35}\cdot 2.911\sqrt{\tau}\right)^{2}+\pi(140\tau+1)^{2}}\right)-\right. |  |
|  |  |  |
| --- | --- | --- |
|  | π​2⋅20⋅35⋅2.911​(2800​τ​(7​τ+5)−1)⋅𝜋220352.9112800𝜏7𝜏51\displaystyle\left.\sqrt{\pi}2\cdot 20\cdot\sqrt{35}\cdot 2.911(2800\tau(7\tau+5)-1)\right. |  |
|  |  |  |
| --- | --- | --- |
|  | τ(π(2.911−1)(70τ+1)+(20⋅35⋅2.911​τ)2+π​(70​τ+1)2))\displaystyle\left.\sqrt{\tau}\left(\sqrt{\pi}(2.911-1)(70\tau+1)+\sqrt{\left(20\cdot\sqrt{35}\cdot 2.911\sqrt{\tau}\right)^{2}+\pi(70\tau+1)^{2}}\right)\right) |  |
|  |  |  |
| --- | --- | --- |
|  | ((π(2.911−1)(70τ+1)+(20​35⋅2.911⋅τ)2+π​(70​τ+1)2)\displaystyle\left(\left(\sqrt{\pi}(2.911-1)(70\tau+1)+\sqrt{\left(20\sqrt{35}\cdot 2.911\cdot\sqrt{\tau}\right)^{2}+\pi(70\tau+1)^{2}}\right)\right. |  |
|  |  |  |
| --- | --- | --- |
|  | (π(2.911−1)(140τ+1)+(20​35⋅2.911⋅τ)2+π​(140​τ+1)2))−1.\displaystyle\left.\left(\sqrt{\pi}(2.911-1)(140\tau+1)+\sqrt{\left(20\sqrt{35}\cdot 2.911\cdot\sqrt{\tau}\right)^{2}+\pi(140\tau+1)^{2}}\right)\right)^{-1}\ . |  |

After applying the approximation
of Ren and MacKenzie, [[30](#bib.bib30)] and adding 200,
we first factored out 20​35​τ2035𝜏20\sqrt{35}\sqrt{\tau}.
Then we brought all terms to the same denominator.

We now consider the numerator:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | (20​35​(210​τ−1)​τ+200)​(π​(2.911−1)​(70​τ+1)+(20⋅35⋅2.911​τ)2+π​(70​τ+1)2)2035210𝜏1𝜏200𝜋2.911170𝜏1superscript⋅20352.911𝜏2𝜋superscript70𝜏12\displaystyle\left(20\sqrt{35}(210\tau-1)\sqrt{\tau}+200\right)\left(\sqrt{\pi}(2.911-1)(70\tau+1)+\sqrt{\left(20\cdot\sqrt{35}\cdot 2.911\sqrt{\tau}\right)^{2}+\pi(70\tau+1)^{2}}\right) |  | (317) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (π​(2.911−1)​(140​τ+1)+(20⋅35⋅2.911​τ)2+π​(140​τ+1)2)+limit-from𝜋2.9111140𝜏1superscript⋅20352.911𝜏2𝜋superscript140𝜏12\displaystyle\left(\sqrt{\pi}(2.911-1)(140\tau+1)+\sqrt{\left(20\cdot\sqrt{35}\cdot 2.911\sqrt{\tau}\right)^{2}+\pi(140\tau+1)^{2}}\right)\ + |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2.911⋅20​35​π​(700​τ​(7​τ+20)−1)​τ⋅2.9112035𝜋700𝜏7𝜏201𝜏\displaystyle 2.911\cdot 20\sqrt{35}\sqrt{\pi}(700\tau(7\tau+20)-1)\sqrt{\tau} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (π​(2.911−1)​(140​τ+1)+(20⋅35⋅2.911​τ)2+π​(140​τ+1)2)−limit-from𝜋2.9111140𝜏1superscript⋅20352.911𝜏2𝜋superscript140𝜏12\displaystyle\left(\sqrt{\pi}(2.911-1)(140\tau+1)+\sqrt{\left(20\cdot\sqrt{35}\cdot 2.911\sqrt{\tau}\right)^{2}+\pi(140\tau+1)^{2}}\right)- |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | π​2⋅20⋅35⋅2.911​(2800​τ​(7​τ+5)−1)​τ⋅𝜋220352.9112800𝜏7𝜏51𝜏\displaystyle\sqrt{\pi}2\cdot 20\cdot\sqrt{35}\cdot 2.911(2800\tau(7\tau+5)-1)\sqrt{\tau} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (π​(2.911−1)​(70​τ+1)+(20⋅35⋅2.911​τ)2+π​(70​τ+1)2)=𝜋2.911170𝜏1superscript⋅20352.911𝜏2𝜋superscript70𝜏12absent\displaystyle\left(\sqrt{\pi}(2.911-1)(70\tau+1)+\sqrt{\left(20\cdot\sqrt{35}\cdot 2.911\sqrt{\tau}\right)^{2}+\pi(70\tau+1)^{2}}\right)\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −1.70658×107​π​(70​τ+1)2+118635​τ​τ3/2+limit-from1.70658superscript107𝜋superscript70𝜏12118635𝜏superscript𝜏32\displaystyle-1.70658\times 10^{7}\sqrt{\pi(70\tau+1)^{2}+118635\tau}\tau^{3/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 4200​35​π​(70​τ+1)2+118635​τ​π​(140​τ+1)2+118635​τ​τ3/2+limit-from420035𝜋superscript70𝜏12118635𝜏𝜋superscript140𝜏12118635𝜏superscript𝜏32\displaystyle 4200\sqrt{35}\sqrt{\pi(70\tau+1)^{2}+118635\tau}\sqrt{\pi(140\tau+1)^{2}+118635\tau}\tau^{3/2}\ + |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 8.60302×106​π​(140​τ+1)2+118635​τ​τ3/2−2.89498×107​τ3/2−8.60302superscript106𝜋superscript140𝜏12118635𝜏superscript𝜏32limit-from2.89498superscript107superscript𝜏32\displaystyle 8.60302\times 10^{6}\sqrt{\pi(140\tau+1)^{2}+118635\tau}\tau^{3/2}-2.89498\times 10^{7}\tau^{3/2}\ - |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 1.21486×107​π​(70​τ+1)2+118635​τ​τ5/2+8.8828×106​π​(140​τ+1)2+118635​τ​τ5/2−1.21486superscript107𝜋superscript70𝜏12118635𝜏superscript𝜏52limit-from8.8828superscript106𝜋superscript140𝜏12118635𝜏superscript𝜏52\displaystyle 1.21486\times 10^{7}\sqrt{\pi(70\tau+1)^{2}+118635\tau}\tau^{5/2}+8.8828\times 10^{6}\sqrt{\pi(140\tau+1)^{2}+118635\tau}\tau^{5/2}\ - |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2.43651×107​τ5/2−1.46191×109​τ7/2+2.24868×107​τ2+94840.5​π​(70​τ+1)2+118635​τ​τ+2.43651superscript107superscript𝜏521.46191superscript109superscript𝜏722.24868superscript107superscript𝜏2limit-from94840.5𝜋superscript70𝜏12118635𝜏𝜏\displaystyle 2.43651\times 10^{7}\tau^{5/2}-1.46191\times 10^{9}\tau^{7/2}+2.24868\times 10^{7}\tau^{2}+94840.5\sqrt{\pi(70\tau+1)^{2}+118635\tau}\tau\ + |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 47420.2​π​(140​τ+1)2+118635​τ​τ+481860​τ+710.354​τ+47420.2𝜋superscript140𝜏12118635𝜏𝜏481860𝜏limit-from710.354𝜏\displaystyle 47420.2\sqrt{\pi(140\tau+1)^{2}+118635\tau}\tau+481860\tau+710.354\sqrt{\tau}\ + |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 820.213​τ​π​(70​τ+1)2+118635​τ+677.432​π​(70​τ+1)2+118635​τ−820.213𝜏𝜋superscript70𝜏12118635𝜏limit-from677.432𝜋superscript70𝜏12118635𝜏\displaystyle 820.213\sqrt{\tau}\sqrt{\pi(70\tau+1)^{2}+118635\tau}+677.432\sqrt{\pi(70\tau+1)^{2}+118635\tau}\ - |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 1011.27​τ​π​(140​τ+1)2+118635​τ−limit-from1011.27𝜏𝜋superscript140𝜏12118635𝜏\displaystyle 1011.27\sqrt{\tau}\sqrt{\pi(140\tau+1)^{2}+118635\tau}\ - |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 20​35​τ​π​(70​τ+1)2+118635​τ​π​(140​τ+1)2+118635​τ+limit-from2035𝜏𝜋superscript70𝜏12118635𝜏𝜋superscript140𝜏12118635𝜏\displaystyle 20\sqrt{35}\sqrt{\tau}\sqrt{\pi(70\tau+1)^{2}+118635\tau}\sqrt{\pi(140\tau+1)^{2}+118635\tau}\ + |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 200​π​(70​τ+1)2+118635​τ​π​(140​τ+1)2+118635​τ+limit-from200𝜋superscript70𝜏12118635𝜏𝜋superscript140𝜏12118635𝜏\displaystyle 200\sqrt{\pi(70\tau+1)^{2}+118635\tau}\sqrt{\pi(140\tau+1)^{2}+118635\tau}\ + |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 677.432​π​(140​τ+1)2+118635​τ+2294.57=677.432𝜋superscript140𝜏12118635𝜏2294.57absent\displaystyle 677.432\sqrt{\pi(140\tau+1)^{2}+118635\tau}+2294.57\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −2.89498×107​τ3/2−2.43651×107​τ5/2−1.46191×109​τ7/2+2.89498superscript107superscript𝜏322.43651superscript107superscript𝜏52limit-from1.46191superscript109superscript𝜏72\displaystyle-2.89498\times 10^{7}\tau^{3/2}-2.43651\times 10^{7}\tau^{5/2}-1.46191\times 10^{9}\tau^{7/2}\ + |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−1.70658×107​τ3/2−1.21486×107​τ5/2+94840.5​τ+820.213​τ+677.432)1.70658superscript107superscript𝜏321.21486superscript107superscript𝜏5294840.5𝜏820.213𝜏677.432\displaystyle\left(-1.70658\times 10^{7}\tau^{3/2}-1.21486\times 10^{7}\tau^{5/2}+94840.5\tau+820.213\sqrt{\tau}+677.432\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | π​(70​τ+1)2+118635​τ+limit-from𝜋superscript70𝜏12118635𝜏\displaystyle\sqrt{\pi(70\tau+1)^{2}+118635\tau}\ + |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (8.60302×106​τ3/2+8.8828×106​τ5/2+47420.2​τ−1011.27​τ+677.432)8.60302superscript106superscript𝜏328.8828superscript106superscript𝜏5247420.2𝜏1011.27𝜏677.432\displaystyle\left(8.60302\times 10^{6}\tau^{3/2}+8.8828\times 10^{6}\tau^{5/2}+47420.2\tau-1011.27\sqrt{\tau}+677.432\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | π​(140​τ+1)2+118635​τ+limit-from𝜋superscript140𝜏12118635𝜏\displaystyle\sqrt{\pi(140\tau+1)^{2}+118635\tau}\ + |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (4200​35​τ3/2−20​35​τ+200)​π​(70​τ+1)2+118635​τ​π​(140​τ+1)2+118635​τ+limit-from420035superscript𝜏322035𝜏200𝜋superscript70𝜏12118635𝜏𝜋superscript140𝜏12118635𝜏\displaystyle\left(4200\sqrt{35}\tau^{3/2}-20\sqrt{35}\sqrt{\tau}+200\right)\sqrt{\pi(70\tau+1)^{2}+118635\tau}\sqrt{\pi(140\tau+1)^{2}+118635\tau}\ + |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2.24868×107​τ2+481860.τ+710.354​τ+2294.57⩽formulae-sequence2.24868superscript107superscript𝜏2481860𝜏710.354𝜏2294.57absent\displaystyle 2.24868\times 10^{7}\tau^{2}+481860.\tau+710.354\sqrt{\tau}+2294.57\ \leqslant |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −2.89498×107​τ3/2−2.43651×107​τ5/2−1.46191×109​τ7/2+2.89498superscript107superscript𝜏322.43651superscript107superscript𝜏52limit-from1.46191superscript109superscript𝜏72\displaystyle-2.89498\times 10^{7}\tau^{3/2}-2.43651\times 10^{7}\tau^{5/2}-1.46191\times 10^{9}\tau^{7/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−1.70658×107​τ3/2−1.21486×107​τ5/2+820.213​1.25+1.25⋅94840.5+677.432)1.70658superscript107superscript𝜏321.21486superscript107superscript𝜏52820.2131.25⋅1.2594840.5677.432\displaystyle\left(-1.70658\times 10^{7}\tau^{3/2}-1.21486\times 10^{7}\tau^{5/2}+820.213\sqrt{1.25}+1.25\cdot 94840.5+677.432\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | π​(70​τ+1)2+118635​τ+limit-from𝜋superscript70𝜏12118635𝜏\displaystyle\sqrt{\pi(70\tau+1)^{2}+118635\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (8.60302×106​τ3/2+8.8828×106​τ5/2−1011.27​0.8+1.25⋅47420.2+677.432)8.60302superscript106superscript𝜏328.8828superscript106superscript𝜏521011.270.8⋅1.2547420.2677.432\displaystyle\left(8.60302\times 10^{6}\tau^{3/2}+8.8828\times 10^{6}\tau^{5/2}-1011.27\sqrt{0.8}+1.25\cdot 47420.2+677.432\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | π​(140​τ+1)2+118635​τ+limit-from𝜋superscript140𝜏12118635𝜏\displaystyle\sqrt{\pi(140\tau+1)^{2}+118635\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (4200​35​τ3/2−20​35​τ+200)420035superscript𝜏322035𝜏200\displaystyle\left(4200\sqrt{35}\tau^{3/2}-20\sqrt{35}\sqrt{\tau}+200\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | π​(70​τ+1)2+118635​τ​π​(140​τ+1)2+118635​τ+limit-from𝜋superscript70𝜏12118635𝜏𝜋superscript140𝜏12118635𝜏\displaystyle\sqrt{\pi(70\tau+1)^{2}+118635\tau}\sqrt{\pi(140\tau+1)^{2}+118635\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2.24868×107​τ2+710.354​1.25+1.25⋅481860+2294.57=2.24868superscript107superscript𝜏2710.3541.25⋅1.254818602294.57absent\displaystyle 2.24868\times 10^{7}\tau^{2}+710.354\sqrt{1.25}+1.25\cdot 481860+2294.57\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −2.89498×107​τ3/2−2.43651×107​τ5/2−1.46191×109​τ7/2+2.89498superscript107superscript𝜏322.43651superscript107superscript𝜏52limit-from1.46191superscript109superscript𝜏72\displaystyle-2.89498\times 10^{7}\tau^{3/2}-2.43651\times 10^{7}\tau^{5/2}-1.46191\times 10^{9}\tau^{7/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−1.70658×107τ3/2−1.21486×107τ5/2+120145.)π​(70​τ+1)2+118635​τ+\displaystyle\left(-1.70658\times 10^{7}\tau^{3/2}-1.21486\times 10^{7}\tau^{5/2}+120145.\right)\sqrt{\pi(70\tau+1)^{2}+118635\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (8.60302×106​τ3/2+8.8828×106​τ5/2+59048.2)​π​(140​τ+1)2+118635​τ+limit-from8.60302superscript106superscript𝜏328.8828superscript106superscript𝜏5259048.2𝜋superscript140𝜏12118635𝜏\displaystyle\left(8.60302\times 10^{6}\tau^{3/2}+8.8828\times 10^{6}\tau^{5/2}+59048.2\right)\sqrt{\pi(140\tau+1)^{2}+118635\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (4200​35​τ3/2−20​35​τ+200)​π​(70​τ+1)2+118635​τ​π​(140​τ+1)2+118635​τ+limit-from420035superscript𝜏322035𝜏200𝜋superscript70𝜏12118635𝜏𝜋superscript140𝜏12118635𝜏\displaystyle\left(4200\sqrt{35}\tau^{3/2}-20\sqrt{35}\sqrt{\tau}+200\right)\sqrt{\pi(70\tau+1)^{2}+118635\tau}\sqrt{\pi(140\tau+1)^{2}+118635\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2.24868×107​τ2+605413=2.24868superscript107superscript𝜏2605413absent\displaystyle 2.24868\times 10^{7}\tau^{2}+605413\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −2.89498×107​τ3/2−2.43651×107​τ5/2−1.46191×109​τ7/2+2.89498superscript107superscript𝜏322.43651superscript107superscript𝜏52limit-from1.46191superscript109superscript𝜏72\displaystyle-2.89498\times 10^{7}\tau^{3/2}-2.43651\times 10^{7}\tau^{5/2}-1.46191\times 10^{9}\tau^{7/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (8.60302×106​τ3/2+8.8828×106​τ5/2+59048.2)​19600​π​(τ+1.94093)​(τ+0.0000262866)+limit-from8.60302superscript106superscript𝜏328.8828superscript106superscript𝜏5259048.219600𝜋𝜏1.94093𝜏0.0000262866\displaystyle\left(8.60302\times 10^{6}\tau^{3/2}+8.8828\times 10^{6}\tau^{5/2}+59048.2\right)\sqrt{19600\pi(\tau+1.94093)(\tau+0.0000262866)}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−1.70658×107τ3/2−1.21486×107τ5/2+120145.)4900​π​(τ+7.73521)​(τ+0.0000263835)+\displaystyle\left(-1.70658\times 10^{7}\tau^{3/2}-1.21486\times 10^{7}\tau^{5/2}+120145.\right)\sqrt{4900\pi(\tau+7.73521)(\tau+0.0000263835)}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (4200​35​τ3/2−20​35​τ+200)420035superscript𝜏322035𝜏200\displaystyle\left(4200\sqrt{35}\tau^{3/2}-20\sqrt{35}\sqrt{\tau}+200\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 19600​π​(τ+1.94093)​(τ+0.0000262866)​4900​π​(τ+7.73521)​(τ+0.0000263835)+limit-from19600𝜋𝜏1.94093𝜏0.00002628664900𝜋𝜏7.73521𝜏0.0000263835\displaystyle\sqrt{19600\pi(\tau+1.94093)(\tau+0.0000262866)}\sqrt{4900\pi(\tau+7.73521)(\tau+0.0000263835)}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2.24868×107​τ2+605413⩽2.24868superscript107superscript𝜏2605413absent\displaystyle 2.24868\times 10^{7}\tau^{2}+605413\ \leqslant |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −2.89498×107​τ3/2−2.43651×107​τ5/2−1.46191×109​τ7/2+2.89498superscript107superscript𝜏322.43651superscript107superscript𝜏52limit-from1.46191superscript109superscript𝜏72\displaystyle-2.89498\times 10^{7}\tau^{3/2}-2.43651\times 10^{7}\tau^{5/2}-1.46191\times 10^{9}\tau^{7/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (8.60302×106​τ3/2+8.8828×106​τ5/2+59048.2)​19600​π​(τ+1.94093)​τ+limit-from8.60302superscript106superscript𝜏328.8828superscript106superscript𝜏5259048.219600𝜋𝜏1.94093𝜏\displaystyle\left(8.60302\times 10^{6}\tau^{3/2}+8.8828\times 10^{6}\tau^{5/2}+59048.2\right)\sqrt{19600\pi(\tau+1.94093)\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−1.70658×107τ3/2−1.21486×107τ5/2+120145.)4900​π​1.00003​(τ+7.73521)​τ+\displaystyle\left(-1.70658\times 10^{7}\tau^{3/2}-1.21486\times 10^{7}\tau^{5/2}+120145.\right)\sqrt{4900\pi 1.00003(\tau+7.73521)\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (4200​35​τ3/2−20​35​τ+200)​19600​π​1.00003​(τ+1.94093)​τ420035superscript𝜏322035𝜏20019600𝜋1.00003𝜏1.94093𝜏\displaystyle\left(4200\sqrt{35}\tau^{3/2}-20\sqrt{35}\sqrt{\tau}+200\right)\sqrt{19600\pi 1.00003(\tau+1.94093)\tau} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 4900​π​1.00003​(τ+7.73521)​τ+limit-from4900𝜋1.00003𝜏7.73521𝜏\displaystyle\sqrt{4900\pi 1.00003(\tau+7.73521)\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2.24868×107​τ2+605413=2.24868superscript107superscript𝜏2605413absent\displaystyle 2.24868\times 10^{7}\tau^{2}+605413\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −2.89498×107​τ3/2−2.43651×107​τ5/2−1.46191×109​τ7/2+2.89498superscript107superscript𝜏322.43651superscript107superscript𝜏52limit-from1.46191superscript109superscript𝜏72\displaystyle-2.89498\times 10^{7}\tau^{3/2}-2.43651\times 10^{7}\tau^{5/2}-1.46191\times 10^{9}\tau^{7/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−3.64296×106​τ3/2+7.65021×108​τ5/2+6.15772×106​τ)3.64296superscript106superscript𝜏327.65021superscript108superscript𝜏526.15772superscript106𝜏\displaystyle\left(-3.64296\times 10^{6}\tau^{3/2}+7.65021\times 10^{8}\tau^{5/2}+6.15772\times 10^{6}\tau\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | τ+1.94093​τ+7.73521+2.24868×107​τ2+𝜏1.94093𝜏7.73521limit-from2.24868superscript107superscript𝜏2\displaystyle\sqrt{\tau+1.94093}\sqrt{\tau+7.73521}+2.24868\times 10^{7}\tau^{2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (2.20425×109​τ3+2.13482×109​τ2+1.46527×107​τ)​τ+1.94093+limit-from2.20425superscript109superscript𝜏32.13482superscript109superscript𝜏21.46527superscript107𝜏𝜏1.94093\displaystyle\left(2.20425\times 10^{9}\tau^{3}+2.13482\times 10^{9}\tau^{2}+1.46527\times 10^{7}\sqrt{\tau}\right)\sqrt{\tau+1.94093}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−1.5073×109​τ3−2.11738×109​τ2+1.49066×107​τ)​τ+7.73521+605413⩽1.5073superscript109superscript𝜏32.11738superscript109superscript𝜏21.49066superscript107𝜏𝜏7.73521605413absent\displaystyle\left(-1.5073\times 10^{9}\tau^{3}-2.11738\times 10^{9}\tau^{2}+1.49066\times 10^{7}\sqrt{\tau}\right)\sqrt{\tau+7.73521}+605413\ \leqslant |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 1.25+1.94093​1.25+7.73521​(−3.64296×106​τ3/2+7.65021×108​τ5/2+6.15772×106​τ)+limit-from1.251.940931.257.735213.64296superscript106superscript𝜏327.65021superscript108superscript𝜏526.15772superscript106𝜏\displaystyle\sqrt{1.25+1.94093}\sqrt{1.25+7.73521}\left(-3.64296\times 10^{6}\tau^{3/2}+7.65021\times 10^{8}\tau^{5/2}+6.15772\times 10^{6}\tau\right)+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 1.25+1.94093​(2.20425×109​τ3+2.13482×109​τ2+1.46527×107​τ)+limit-from1.251.940932.20425superscript109superscript𝜏32.13482superscript109superscript𝜏21.46527superscript107𝜏\displaystyle\sqrt{1.25+1.94093}\left(2.20425\times 10^{9}\tau^{3}+2.13482\times 10^{9}\tau^{2}+1.46527\times 10^{7}\sqrt{\tau}\right)+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 0.8+7.73521​(−1.5073×109​τ3−2.11738×109​τ2+1.49066×107​τ)−limit-from0.87.735211.5073superscript109superscript𝜏32.11738superscript109superscript𝜏21.49066superscript107𝜏\displaystyle\sqrt{0.8+7.73521}\left(-1.5073\times 10^{9}\tau^{3}-2.11738\times 10^{9}\tau^{2}+1.49066\times 10^{7}\sqrt{\tau}\right)- |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2.89498×107​τ3/2−2.43651×107​τ5/2−1.46191×109​τ7/2+2.24868×107​τ2+605413=2.89498superscript107superscript𝜏322.43651superscript107superscript𝜏521.46191superscript109superscript𝜏722.24868superscript107superscript𝜏2605413absent\displaystyle 2.89498\times 10^{7}\tau^{3/2}-2.43651\times 10^{7}\tau^{5/2}-1.46191\times 10^{9}\tau^{7/2}+2.24868\times 10^{7}\tau^{2}+605413\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −4.84561×107​τ3/2+4.07198×109​τ5/2−1.46191×109​τ7/2−4.84561superscript107superscript𝜏324.07198superscript109superscript𝜏52limit-from1.46191superscript109superscript𝜏72\displaystyle-4.84561\times 10^{7}\tau^{3/2}+4.07198\times 10^{9}\tau^{5/2}-1.46191\times 10^{9}\tau^{7/2}- |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 4.66103×108​τ3−2.34999×109​τ2+4.66103superscript108superscript𝜏3limit-from2.34999superscript109superscript𝜏2\displaystyle 4.66103\times 10^{8}\tau^{3}-2.34999\times 10^{9}\tau^{2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 3.29718×107​τ+6.97241×107​τ+605413⩽3.29718superscript107𝜏6.97241superscript107𝜏605413absent\displaystyle 3.29718\times 10^{7}\tau+6.97241\times 10^{7}\sqrt{\tau}+605413\ \leqslant |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 605413​τ3/20.83/2−4.84561×107​τ3/2+605413superscript𝜏32superscript0.832limit-from4.84561superscript107superscript𝜏32\displaystyle\frac{605413\tau^{3/2}}{0.8^{3/2}}-4.84561\times 10^{7}\tau^{3/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 4.07198×109​τ5/2−1.46191×109​τ7/2−4.07198superscript109superscript𝜏52limit-from1.46191superscript109superscript𝜏72\displaystyle 4.07198\times 10^{9}\tau^{5/2}-1.46191\times 10^{9}\tau^{7/2}- |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 4.66103×108​τ3−2.34999×109​τ2+3.29718×107​τ​τ0.8+6.97241×107​τ​τ0.8=4.66103superscript108superscript𝜏32.34999superscript109superscript𝜏23.29718superscript107𝜏𝜏0.86.97241superscript107𝜏𝜏0.8absent\displaystyle 4.66103\times 10^{8}\tau^{3}-2.34999\times 10^{9}\tau^{2}+\frac{3.29718\times 10^{7}\sqrt{\tau}\tau}{\sqrt{0.8}}+\frac{6.97241\times 10^{7}\tau\sqrt{\tau}}{0.8}\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | τ3/2(−4.66103×108τ3/2−1.46191×109τ2−2.34999×109τ+\displaystyle\tau^{3/2}\left(-4.66103\times 10^{8}\tau^{3/2}-1.46191\times 10^{9}\tau^{2}-2.34999\times 10^{9}\sqrt{\tau}+\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 4.07198×109τ+7.64087×107)⩽\displaystyle\left.4.07198\times 10^{9}\tau+7.64087\times 10^{7}\right)\ \leqslant |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | τ3/2(−4.66103×108τ3/2−1.46191×109τ2+7.64087×107​τ0.8−\displaystyle\tau^{3/2}\left(-4.66103\times 10^{8}\tau^{3/2}-1.46191\times 10^{9}\tau^{2}+\frac{7.64087\times 10^{7}\sqrt{\tau}}{\sqrt{0.8}}-\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2.34999×109τ+4.07198×109τ)=\displaystyle\left.2.34999\times 10^{9}\sqrt{\tau}+4.07198\times 10^{9}\tau\right)\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | τ2​(−1.46191×109​τ3/2+4.07198×109​τ−4.66103×108​τ−2.26457×109)⩽superscript𝜏21.46191superscript109superscript𝜏324.07198superscript109𝜏4.66103superscript108𝜏2.26457superscript109absent\displaystyle\tau^{2}\left(-1.46191\times 10^{9}\tau^{3/2}+4.07198\times 10^{9}\sqrt{\tau}-4.66103\times 10^{8}\tau-2.26457\times 10^{9}\right)\ \leqslant |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−2.26457×109+4.07198×109​0.8−4.66103×108​0.8−1.46191×109​0.83/2)​τ2=2.26457superscript1094.07198superscript1090.84.66103superscript1080.81.46191superscript109superscript0.832superscript𝜏2absent\displaystyle\left(-2.26457\times 10^{9}+4.07198\times 10^{9}\sqrt{0.8}-4.66103\times 10^{8}0.8-1.46191\times 10^{9}0.8^{3/2}\right)\tau^{2}\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −4.14199×107​τ2< 0.4.14199superscript107superscript𝜏2 0\displaystyle-4.14199\times 10^{7}\tau^{2}\ <\ 0\ . |  |

First we expanded the term (multiplied it out).
The we put the terms multiplied by the same square root into brackets.
The next inequality sign stems from inserting the maximal value of 1.251.251.25 for τ𝜏\tau for
some positive terms and value of 0.80.80.8 for negative terms.
These terms are then expanded at the ==-sign.
The next equality factors the terms under the squared root.
We decreased the negative term by setting
τ=τ+0.0000263835𝜏𝜏0.0000263835\tau=\tau+0.0000263835 under the root.
We increased positive terms by setting
τ+0.000026286=1.00003​τ𝜏0.0000262861.00003𝜏\tau+0.000026286=1.00003\tau and
τ+0.000026383=1.00003​τ𝜏0.0000263831.00003𝜏\tau+0.000026383=1.00003\tau
under the root for positive terms.
The positive terms are increase, since
0.8+0.0000263830.8=1.000030.80.0000263830.81.00003\frac{0.8+0.000026383}{0.8}=1.00003, thus
τ+0.000026286<τ+0.000026383⩽1.00003​τ𝜏0.000026286𝜏0.0000263831.00003𝜏\tau+0.000026286<\tau+0.000026383\leqslant 1.00003\tau.
For the next inequality we decreased negative terms by inserting
τ=0.8𝜏0.8\tau=0.8 and increased positive terms by inserting
τ=1.25𝜏1.25\tau=1.25. The next equality expands the terms.
We use upper bound of 1.251.251.25 and lower bound of 0.80.80.8 to obtain terms with
corresponding exponents of τ𝜏\tau.

For the last ⩽\leqslant-sign we used the function

|  |  |  |  |
| --- | --- | --- | --- |
|  | −1.46191×109​τ3/2+4.07198×109​τ−4.66103×108​τ−2.26457×1091.46191superscript109superscript𝜏324.07198superscript109𝜏4.66103superscript108𝜏2.26457superscript109\displaystyle-1.46191\times 10^{9}\tau^{3/2}+4.07198\times 10^{9}\sqrt{\tau}-4.66103\times 10^{8}\tau-2.26457\times 10^{9} |  | (318) |

The derivative of this function is

|  |  |  |  |
| --- | --- | --- | --- |
|  | −2.19286×109​τ+2.03599×109τ−4.66103×1082.19286superscript109𝜏2.03599superscript109𝜏4.66103superscript108\displaystyle-2.19286\times 10^{9}\sqrt{\tau}+\frac{2.03599\times 10^{9}}{\sqrt{\tau}}-4.66103\times 10^{8} |  | (319) |

and the second order derivative is

|  |  |  |  |
| --- | --- | --- | --- |
|  | −1.01799×109τ3/2−1.09643×109τ< 0.1.01799superscript109superscript𝜏321.09643superscript109𝜏 0\displaystyle-\frac{1.01799\times 10^{9}}{\tau^{3/2}}-\frac{1.09643\times 10^{9}}{\sqrt{\tau}}\ <\ 0\ . |  | (320) |

The derivative at 0.8 is smaller than zero:

|  |  |  |  |
| --- | --- | --- | --- |
|  | −2.19286×109​0.8−4.66103×108+2.03599×1090.8=2.19286superscript1090.84.66103superscript1082.03599superscript1090.8absent\displaystyle-2.19286\times 10^{9}\sqrt{0.8}-4.66103\times 10^{8}+\frac{2.03599\times 10^{9}}{\sqrt{0.8}}= |  | (321) |
|  |  |  |
| --- | --- | --- |
|  | −1.51154×108< 0.1.51154superscript108 0\displaystyle-1.51154\times 10^{8}\ <\ 0\ . |  |

Since the second order derivative is negative, the derivative
decreases with increasing τ𝜏\tau. Therefore the derivative is
negative for all values of τ𝜏\tau that we consider, that is, the
function Eq. ([318](#S3.E318 "In Proof. ‣ Behavior of the main subfunction with respect to 𝑦 at minimal 𝑥. ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) is strictly monotonically decreasing.
The maximum of the function Eq. ([318](#S3.E318 "In Proof. ‣ Behavior of the main subfunction with respect to 𝑦 at minimal 𝑥. ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")) is therefore at 0.80.80.8.
We inserted 0.80.80.8 to obtain the maximum.

Consequently, the derivative of

|  |  |  |  |
| --- | --- | --- | --- |
|  | τ​(e(μ​ω+ν​τ2​ν​τ)2​erfc⁡(μ​ω+ν​τ2​ν​τ)−2​e(μ​ω+2​ν​τ2​ν​τ)2​erfc⁡(μ​ω+2​ν​τ2​ν​τ))𝜏superscript𝑒superscript𝜇𝜔𝜈𝜏2𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏2superscript𝑒superscript𝜇𝜔2𝜈𝜏2𝜈𝜏2erfc𝜇𝜔2𝜈𝜏2𝜈𝜏\displaystyle\tau\left(e^{\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-2e^{\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\right) |  | (322) |

with respect to τ𝜏\tau is smaller than zero for maximal ν=0.7𝜈0.7\nu=0.7.

Next,
we consider the function for
the largest ν=0.16𝜈0.16\nu=0.16 and the largest y=μ​ω=0.01𝑦𝜇𝜔0.01y=\mu\omega=0.01
for determining the derivative with respect to τ𝜏\tau.

The expression becomes

|  |  |  |  |
| --- | --- | --- | --- |
|  | τ​(e(16​τ100+11002​16​τ100)2​erfc⁡(16​τ100+11002​16​τ100)−e(2 16​τ100+11002​16​τ100)2​erfc⁡(2 16​τ100+11002​16​τ100)).𝜏superscript𝑒superscript16𝜏1001100216𝜏1002erfc16𝜏1001100216𝜏100superscript𝑒superscript216𝜏1001100216𝜏1002erfc216𝜏1001100216𝜏100\displaystyle\tau\left(e^{\left(\frac{\frac{16\tau}{100}+\frac{1}{100}}{\sqrt{2}\sqrt{\frac{16\tau}{100}}}\right)^{2}}\operatorname{erfc}\left(\frac{\frac{16\tau}{100}+\frac{1}{100}}{\sqrt{2}\sqrt{\frac{16\tau}{100}}}\right)-e^{\left(\frac{\frac{2\ 16\tau}{100}+\frac{1}{100}}{\sqrt{2}\sqrt{\frac{16\tau}{100}}}\right)^{2}}\operatorname{erfc}\left(\frac{\frac{2\ 16\tau}{100}+\frac{1}{100}}{\sqrt{2}\sqrt{\frac{16\tau}{100}}}\right)\right)\ . |  | (323) |

The derivative with respect to τ𝜏\tau is

|  |  |  |  |
| --- | --- | --- | --- |
|  | (π(e(16​τ+1)23200​τ(128τ(2τ+25)−1)erfc(16​τ+140​2​τ)−\displaystyle\left(\sqrt{\pi}\left(e^{\frac{(16\tau+1)^{2}}{3200\tau}}(128\tau(2\tau+25)-1)\operatorname{erfc}\left(\frac{16\tau+1}{40\sqrt{2}\sqrt{\tau}}\right)-\right.\right. |  | (324) |
|  |  |  |
| --- | --- | --- |
|  | 2e(32​τ+1)23200​τ(128τ(8τ+25)−1)erfc(32​τ+140​2​τ))+402(48τ−1)τ)\displaystyle\left.\left.2e^{\frac{(32\tau+1)^{2}}{3200\tau}}(128\tau(8\tau+25)-1)\operatorname{erfc}\left(\frac{32\tau+1}{40\sqrt{2}\sqrt{\tau}}\right)\right)+40\sqrt{2}(48\tau-1)\sqrt{\tau}\right) |  |
|  |  |  |
| --- | --- | --- |
|  | (3200​π​τ)−1.superscript3200𝜋𝜏1\displaystyle\left(3200\sqrt{\pi}\tau\right)^{-1}\ . |  |

We are considering only the numerator and use again the approximation
of Ren and MacKenzie, [[30](#bib.bib30)].
The error analysis on the whole numerator gives an approximation error 1.1<E<121.1𝐸121.1<E<12. Therefore
we add 20 to the numerator when we use the approximation of Ren and MacKenzie, [[30](#bib.bib30)].
We obtain the inequalities:

|  |  |  |  |
| --- | --- | --- | --- |
|  | π(e(16​τ+1)23200​τ(128τ(2τ+25)−1)erfc(16​τ+140​2​τ)−\displaystyle\sqrt{\pi}\left(e^{\frac{(16\tau+1)^{2}}{3200\tau}}(128\tau(2\tau+25)-1)\operatorname{erfc}\left(\frac{16\tau+1}{40\sqrt{2}\sqrt{\tau}}\right)-\right. |  | (325) |
|  |  |  |
| --- | --- | --- |
|  | 2e(32​τ+1)23200​τ(128τ(8τ+25)−1)erfc(32​τ+140​2​τ))+402(48τ−1)τ⩽\displaystyle\left.2e^{\frac{(32\tau+1)^{2}}{3200\tau}}(128\tau(8\tau+25)-1)\operatorname{erfc}\left(\frac{32\tau+1}{40\sqrt{2}\sqrt{\tau}}\right)\right)+40\sqrt{2}(48\tau-1)\sqrt{\tau}\ \leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | π(2.911​(128​τ​(2​τ+25)−1)π​(2.911−1)​(16​τ+1)40​2​τ+π​(16​τ+140​2​τ)2+2.9112−\displaystyle\sqrt{\pi}\left(\frac{2.911(128\tau(2\tau+25)-1)}{\frac{\sqrt{\pi}(2.911-1)(16\tau+1)}{40\sqrt{2}\sqrt{\tau}}+\sqrt{\pi\left(\frac{16\tau+1}{40\sqrt{2}\sqrt{\tau}}\right)^{2}+2.911^{2}}}\ -\right. |  |
|  |  |  |
| --- | --- | --- |
|  | 2⋅2.911​(128​τ​(8​τ+25)−1)π​(2.911−1)​(32​τ+1)40​2​τ+π​(32​τ+140​2​τ)2+2.9112)\displaystyle\left.\frac{2\cdot 2.911(128\tau(8\tau+25)-1)}{\frac{\sqrt{\pi}(2.911-1)(32\tau+1)}{40\sqrt{2}\sqrt{\tau}}+\sqrt{\pi\left(\frac{32\tau+1}{40\sqrt{2}\sqrt{\tau}}\right)^{2}+2.911^{2}}}\right) |  |
|  |  |  |
| --- | --- | --- |
|  | +40​2​(48​τ−1)​τ+20=40248𝜏1𝜏20absent\displaystyle\ +40\sqrt{2}(48\tau-1)\sqrt{\tau}+20\ = |  |
|  |  |  |
| --- | --- | --- |
|  | π((128​τ​(2​τ+25)−1)​(40​2​2.911​τ)π​(2.911−1)​(16​τ+1)+(40​2​2.911​τ)2+π​(16​τ+1)2−\displaystyle\sqrt{\pi}\left(\frac{(128\tau(2\tau+25)-1)\left(40\sqrt{2}2.911\sqrt{\tau}\right)}{\sqrt{\pi}(2.911-1)(16\tau+1)+\sqrt{\left(40\sqrt{2}2.911\sqrt{\tau}\right)^{2}+\pi(16\tau+1)^{2}}}\ -\right. |  |
|  |  |  |
| --- | --- | --- |
|  | 2​(128​τ​(8​τ+25)−1)​(40​2​2.911​τ)π​(2.911−1)​(32​τ+1)+(40​2​2.911​τ)2+π​(32​τ+1)2)+\displaystyle\left.\frac{2(128\tau(8\tau+25)-1)\left(40\sqrt{2}2.911\sqrt{\tau}\right)}{\sqrt{\pi}(2.911-1)(32\tau+1)+\sqrt{\left(40\sqrt{2}2.911\sqrt{\tau}\right)^{2}+\pi(32\tau+1)^{2}}}\right)+ |  |
|  |  |  |
| --- | --- | --- |
|  | 40​2​(48​τ−1)​τ+20=40248𝜏1𝜏20absent\displaystyle 40\sqrt{2}(48\tau-1)\sqrt{\tau}+20\ = |  |
|  |  |  |
| --- | --- | --- |
|  | ((402(48τ−1)τ+20)(π(2.911−1)(16τ+1)+(40​2​2.911​τ)2+π​(16​τ+1)2)\displaystyle\left(\left(40\sqrt{2}(48\tau-1)\sqrt{\tau}+20\right)\left(\sqrt{\pi}(2.911-1)(16\tau+1)+\sqrt{\left(40\sqrt{2}2.911\sqrt{\tau}\right)^{2}+\pi(16\tau+1)^{2}}\right)\right. |  |
|  |  |  |
| --- | --- | --- |
|  | (π(2.911−1)(32τ+1)+(40​2​2.911​τ)2+π​(32​τ+1)2)++\displaystyle\left.\left(\sqrt{\pi}(2.911-1)(32\tau+1)+\sqrt{\left(40\sqrt{2}2.911\sqrt{\tau}\right)^{2}+\pi(32\tau+1)^{2}}\right)++\right. |  |
|  |  |  |
| --- | --- | --- |
|  | 2.911⋅40​2​π​(128​τ​(2​τ+25)−1)​τ⋅2.911402𝜋128𝜏2𝜏251𝜏\displaystyle\left.2.911\cdot 40\sqrt{2}\sqrt{\pi}(128\tau(2\tau+25)-1)\sqrt{\tau}\right. |  |
|  |  |  |
| --- | --- | --- |
|  | (π​(2.911−1)​(32​τ+1)+(40​2​2.911​τ)2+π​(32​τ+1)2)−limit-from𝜋2.911132𝜏1superscript4022.911𝜏2𝜋superscript32𝜏12\displaystyle\left.\left(\sqrt{\pi}(2.911-1)(32\tau+1)+\sqrt{\left(40\sqrt{2}2.911\sqrt{\tau}\right)^{2}+\pi(32\tau+1)^{2}}\right)-\right. |  |
|  |  |  |
| --- | --- | --- |
|  | 2​π​40​2​2.911​(128​τ​(8​τ+25)−1)2𝜋4022.911128𝜏8𝜏251\displaystyle\left.2\sqrt{\pi}40\sqrt{2}2.911(128\tau(8\tau+25)-1)\right. |  |
|  |  |  |
| --- | --- | --- |
|  | τ(π(2.911−1)(16τ+1)+(40​2​2.911​τ)2+π​(16​τ+1)2))\displaystyle\left.\sqrt{\tau}\left(\sqrt{\pi}(2.911-1)(16\tau+1)+\sqrt{\left(40\sqrt{2}2.911\sqrt{\tau}\right)^{2}+\pi(16\tau+1)^{2}}\right)\right) |  |
|  |  |  |
| --- | --- | --- |
|  | ((π(2.911−1)(32τ+1)+(40​2​2.911​τ)2+π​(32​τ+1)2)\displaystyle\left(\left(\sqrt{\pi}(2.911-1)(32\tau+1)+\sqrt{\left(40\sqrt{2}2.911\sqrt{\tau}\right)^{2}+\pi(32\tau+1)^{2}}\right)\right. |  |
|  |  |  |
| --- | --- | --- |
|  | (π(2.911−1)(32τ+1)+(40​2​2.911​τ)2+π​(32​τ+1)2))−1.\displaystyle\left.\left(\sqrt{\pi}(2.911-1)(32\tau+1)+\sqrt{\left(40\sqrt{2}2.911\sqrt{\tau}\right)^{2}+\pi(32\tau+1)^{2}}\right)\right)^{-1}\ . |  |

After applying the approximation
of Ren and MacKenzie, [[30](#bib.bib30)] and adding 20,
we first factored out 40​2​τ402𝜏40\sqrt{2}\sqrt{\tau}.
Then we brought all terms to the same denominator.

We now consider the numerator:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | (40​2​(48​τ−1)​τ+20)​(π​(2.911−1)​(16​τ+1)+(40​2​2.911​τ)2+π​(16​τ+1)2)40248𝜏1𝜏20𝜋2.911116𝜏1superscript4022.911𝜏2𝜋superscript16𝜏12\displaystyle\left(40\sqrt{2}(48\tau-1)\sqrt{\tau}+20\right)\left(\sqrt{\pi}(2.911-1)(16\tau+1)+\sqrt{\left(40\sqrt{2}2.911\sqrt{\tau}\right)^{2}+\pi(16\tau+1)^{2}}\right) |  | (326) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (π​(2.911−1)​(32​τ+1)+(40​2​2.911​τ)2+π​(32​τ+1)2)+limit-from𝜋2.911132𝜏1superscript4022.911𝜏2𝜋superscript32𝜏12\displaystyle\left(\sqrt{\pi}(2.911-1)(32\tau+1)+\sqrt{\left(40\sqrt{2}2.911\sqrt{\tau}\right)^{2}+\pi(32\tau+1)^{2}}\right)\ + |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2.911⋅40​2​π​(128​τ​(2​τ+25)−1)​τ⋅2.911402𝜋128𝜏2𝜏251𝜏\displaystyle 2.911\cdot 40\sqrt{2}\sqrt{\pi}(128\tau(2\tau+25)-1)\sqrt{\tau} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (π​(2.911−1)​(32​τ+1)+(40​2​2.911​τ)2+π​(32​τ+1)2)−limit-from𝜋2.911132𝜏1superscript4022.911𝜏2𝜋superscript32𝜏12\displaystyle\left(\sqrt{\pi}(2.911-1)(32\tau+1)+\sqrt{\left(40\sqrt{2}2.911\sqrt{\tau}\right)^{2}+\pi(32\tau+1)^{2}}\right)- |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2​π​40​2​2.911​(128​τ​(8​τ+25)−1)​τ2𝜋4022.911128𝜏8𝜏251𝜏\displaystyle 2\sqrt{\pi}40\sqrt{2}2.911(128\tau(8\tau+25)-1)\sqrt{\tau} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (π​(2.911−1)​(16​τ+1)+(40​2​2.911​τ)2+π​(16​τ+1)2)=𝜋2.911116𝜏1superscript4022.911𝜏2𝜋superscript16𝜏12absent\displaystyle\left(\sqrt{\pi}(2.911-1)(16\tau+1)+\sqrt{\left(40\sqrt{2}2.911\sqrt{\tau}\right)^{2}+\pi(16\tau+1)^{2}}\right)\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −1.86491×106​π​(16​τ+1)2+27116.5​τ​τ3/2+limit-from1.86491superscript106𝜋superscript16𝜏1227116.5𝜏superscript𝜏32\displaystyle-1.86491\times 10^{6}\sqrt{\pi(16\tau+1)^{2}+27116.5\tau}\tau^{3/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 1920​2​π​(16​τ+1)2+27116.5​τ​π​(32​τ+1)2+27116.5​τ​τ3/2+limit-from19202𝜋superscript16𝜏1227116.5𝜏𝜋superscript32𝜏1227116.5𝜏superscript𝜏32\displaystyle 1920\sqrt{2}\sqrt{\pi(16\tau+1)^{2}+27116.5\tau}\sqrt{\pi(32\tau+1)^{2}+27116.5\tau}\tau^{3/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 940121​π​(32​τ+1)2+27116.5​τ​τ3/2−3.16357×106​τ3/2−940121𝜋superscript32𝜏1227116.5𝜏superscript𝜏32limit-from3.16357superscript106superscript𝜏32\displaystyle 940121\sqrt{\pi(32\tau+1)^{2}+27116.5\tau}\tau^{3/2}-3.16357\times 10^{6}\tau^{3/2}- |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 303446​π​(16​τ+1)2+27116.5​τ​τ5/2+221873​π​(32​τ+1)2+27116.5​τ​τ5/2−608588​τ5/2−303446𝜋superscript16𝜏1227116.5𝜏superscript𝜏52221873𝜋superscript32𝜏1227116.5𝜏superscript𝜏52limit-from608588superscript𝜏52\displaystyle 303446\sqrt{\pi(16\tau+1)^{2}+27116.5\tau}\tau^{5/2}+221873\sqrt{\pi(32\tau+1)^{2}+27116.5\tau}\tau^{5/2}-608588\tau^{5/2}- |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 8.34635×106​τ7/2+117482.τ2+2167.78​π​(16​τ+1)2+27116.5​τ​τ+formulae-sequence8.34635superscript106superscript𝜏72117482superscript𝜏2limit-from2167.78𝜋superscript16𝜏1227116.5𝜏𝜏\displaystyle 8.34635\times 10^{6}\tau^{7/2}+117482.\tau^{2}+2167.78\sqrt{\pi(16\tau+1)^{2}+27116.5\tau}\tau+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 1083.89​π​(32​τ+1)2+27116.5​τ​τ+limit-from1083.89𝜋superscript32𝜏1227116.5𝜏𝜏\displaystyle 1083.89\sqrt{\pi(32\tau+1)^{2}+27116.5\tau}\tau+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 11013.9​τ+339.614​τ+392.137​τ​π​(16​τ+1)2+27116.5​τ+11013.9𝜏339.614𝜏limit-from392.137𝜏𝜋superscript16𝜏1227116.5𝜏\displaystyle 11013.9\tau+339.614\sqrt{\tau}+392.137\sqrt{\tau}\sqrt{\pi(16\tau+1)^{2}+27116.5\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 67.7432​π​(16​τ+1)2+27116.5​τ−483.478​τ​π​(32​τ+1)2+27116.5​τ−67.7432𝜋superscript16𝜏1227116.5𝜏limit-from483.478𝜏𝜋superscript32𝜏1227116.5𝜏\displaystyle 67.7432\sqrt{\pi(16\tau+1)^{2}+27116.5\tau}-483.478\sqrt{\tau}\sqrt{\pi(32\tau+1)^{2}+27116.5\tau}- |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 40​2​τ​π​(16​τ+1)2+27116.5​τ​π​(32​τ+1)2+27116.5​τ+limit-from402𝜏𝜋superscript16𝜏1227116.5𝜏𝜋superscript32𝜏1227116.5𝜏\displaystyle 40\sqrt{2}\sqrt{\tau}\sqrt{\pi(16\tau+1)^{2}+27116.5\tau}\sqrt{\pi(32\tau+1)^{2}+27116.5\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 20​π​(16​τ+1)2+27116.5​τ​π​(32​τ+1)2+27116.5​τ+limit-from20𝜋superscript16𝜏1227116.5𝜏𝜋superscript32𝜏1227116.5𝜏\displaystyle 20\sqrt{\pi(16\tau+1)^{2}+27116.5\tau}\sqrt{\pi(32\tau+1)^{2}+27116.5\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 67.7432​π​(32​τ+1)2+27116.5​τ+229.457=67.7432𝜋superscript32𝜏1227116.5𝜏229.457absent\displaystyle 67.7432\sqrt{\pi(32\tau+1)^{2}+27116.5\tau}+229.457\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −3.16357×106​τ3/2−608588​τ5/2−8.34635×106​τ7/2+3.16357superscript106superscript𝜏32608588superscript𝜏52limit-from8.34635superscript106superscript𝜏72\displaystyle-3.16357\times 10^{6}\tau^{3/2}-608588\tau^{5/2}-8.34635\times 10^{6}\tau^{7/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−1.86491×106​τ3/2−303446​τ5/2+2167.78​τ+392.137​τ+67.7432)1.86491superscript106superscript𝜏32303446superscript𝜏522167.78𝜏392.137𝜏67.7432\displaystyle\left(-1.86491\times 10^{6}\tau^{3/2}-303446\tau^{5/2}+2167.78\tau+392.137\sqrt{\tau}+67.7432\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | π​(16​τ+1)2+27116.5​τ+limit-from𝜋superscript16𝜏1227116.5𝜏\displaystyle\sqrt{\pi(16\tau+1)^{2}+27116.5\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (940121​τ3/2+221873​τ5/2+1083.89​τ−483.478​τ+67.7432)940121superscript𝜏32221873superscript𝜏521083.89𝜏483.478𝜏67.7432\displaystyle\left(940121\tau^{3/2}+221873\tau^{5/2}+1083.89\tau-483.478\sqrt{\tau}+67.7432\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | π​(32​τ+1)2+27116.5​τ+limit-from𝜋superscript32𝜏1227116.5𝜏\displaystyle\sqrt{\pi(32\tau+1)^{2}+27116.5\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (1920​2​τ3/2−40​2​τ+20)​π​(16​τ+1)2+27116.5​τ​π​(32​τ+1)2+27116.5​τ+limit-from19202superscript𝜏32402𝜏20𝜋superscript16𝜏1227116.5𝜏𝜋superscript32𝜏1227116.5𝜏\displaystyle\left(1920\sqrt{2}\tau^{3/2}-40\sqrt{2}\sqrt{\tau}+20\right)\sqrt{\pi(16\tau+1)^{2}+27116.5\tau}\sqrt{\pi(32\tau+1)^{2}+27116.5\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 117482.τ2+11013.9​τ+339.614​τ+229.457⩽formulae-sequence117482superscript𝜏211013.9𝜏339.614𝜏229.457absent\displaystyle 117482.\tau^{2}+11013.9\tau+339.614\sqrt{\tau}+229.457\ \leqslant |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −3.16357×106​τ3/2−608588​τ5/2−8.34635×106​τ7/2+3.16357superscript106superscript𝜏32608588superscript𝜏52limit-from8.34635superscript106superscript𝜏72\displaystyle-3.16357\times 10^{6}\tau^{3/2}-608588\tau^{5/2}-8.34635\times 10^{6}\tau^{7/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−1.86491×106​τ3/2−303446​τ5/2+392.137​1.25+1.252167.78+67.7432)1.86491superscript106superscript𝜏32303446superscript𝜏52392.1371.251.252167.7867.7432\displaystyle\left(-1.86491\times 10^{6}\tau^{3/2}-303446\tau^{5/2}+392.137\sqrt{1.25}+1.252167.78+67.7432\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | π​(16​τ+1)2+27116.5​τ+limit-from𝜋superscript16𝜏1227116.5𝜏\displaystyle\sqrt{\pi(16\tau+1)^{2}+27116.5\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (940121​τ3/2+221873​τ5/2−483.478​0.8+1.251083.89+67.7432)940121superscript𝜏32221873superscript𝜏52483.4780.81.251083.8967.7432\displaystyle\left(940121\tau^{3/2}+221873\tau^{5/2}-483.478\sqrt{0.8}+1.251083.89+67.7432\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | π​(32​τ+1)2+27116.5​τ+limit-from𝜋superscript32𝜏1227116.5𝜏\displaystyle\sqrt{\pi(32\tau+1)^{2}+27116.5\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (1920​2​τ3/2−40​2​τ+20)​π​(16​τ+1)2+27116.5​τ​π​(32​τ+1)2+27116.5​τ+limit-from19202superscript𝜏32402𝜏20𝜋superscript16𝜏1227116.5𝜏𝜋superscript32𝜏1227116.5𝜏\displaystyle\left(1920\sqrt{2}\tau^{3/2}-40\sqrt{2}\sqrt{\tau}+20\right)\sqrt{\pi(16\tau+1)^{2}+27116.5\tau}\sqrt{\pi(32\tau+1)^{2}+27116.5\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 117482.τ2+339.614​1.25+1.2511013.9+229.457=formulae-sequence117482superscript𝜏2339.6141.251.2511013.9229.457absent\displaystyle 117482.\tau^{2}+339.614\sqrt{1.25}+1.2511013.9+229.457\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −3.16357×106​τ3/2−608588​τ5/2−8.34635×106​τ7/2+3.16357superscript106superscript𝜏32608588superscript𝜏52limit-from8.34635superscript106superscript𝜏72\displaystyle-3.16357\times 10^{6}\tau^{3/2}-608588\tau^{5/2}-8.34635\times 10^{6}\tau^{7/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−1.86491×106​τ3/2−303446​τ5/2+3215.89)​π​(16​τ+1)2+27116.5​τ+limit-from1.86491superscript106superscript𝜏32303446superscript𝜏523215.89𝜋superscript16𝜏1227116.5𝜏\displaystyle\left(-1.86491\times 10^{6}\tau^{3/2}-303446\tau^{5/2}+3215.89\right)\sqrt{\pi(16\tau+1)^{2}+27116.5\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (940121​τ3/2+221873​τ5/2+990.171)​π​(32​τ+1)2+27116.5​τ+limit-from940121superscript𝜏32221873superscript𝜏52990.171𝜋superscript32𝜏1227116.5𝜏\displaystyle\left(940121\tau^{3/2}+221873\tau^{5/2}+990.171\right)\sqrt{\pi(32\tau+1)^{2}+27116.5\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (1920​2​τ3/2−40​2​τ+20)​π​(16​τ+1)2+27116.5​τ​π​(32​τ+1)2+27116.5​τ+limit-from19202superscript𝜏32402𝜏20𝜋superscript16𝜏1227116.5𝜏𝜋superscript32𝜏1227116.5𝜏\displaystyle\left(1920\sqrt{2}\tau^{3/2}-40\sqrt{2}\sqrt{\tau}+20\right)\sqrt{\pi(16\tau+1)^{2}+27116.5\tau}\sqrt{\pi(32\tau+1)^{2}+27116.5\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 117482​τ2+14376.6=117482superscript𝜏214376.6absent\displaystyle 117482\tau^{2}+14376.6\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −3.16357×106​τ3/2−608588​τ5/2−8.34635×106​τ7/2+3.16357superscript106superscript𝜏32608588superscript𝜏52limit-from8.34635superscript106superscript𝜏72\displaystyle-3.16357\times 10^{6}\tau^{3/2}-608588\tau^{5/2}-8.34635\times 10^{6}\tau^{7/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (940121​τ3/2+221873​τ5/2+990.171)​1024​π​(τ+8.49155)​(τ+0.000115004)+limit-from940121superscript𝜏32221873superscript𝜏52990.1711024𝜋𝜏8.49155𝜏0.000115004\displaystyle\left(940121\tau^{3/2}+221873\tau^{5/2}+990.171\right)\sqrt{1024\pi(\tau+8.49155)(\tau+0.000115004)}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−1.86491×106​τ3/2−303446​τ5/2+3215.89)​256​π​(τ+33.8415)​(τ+0.000115428)+limit-from1.86491superscript106superscript𝜏32303446superscript𝜏523215.89256𝜋𝜏33.8415𝜏0.000115428\displaystyle\left(-1.86491\times 10^{6}\tau^{3/2}-303446\tau^{5/2}+3215.89\right)\sqrt{256\pi(\tau+33.8415)(\tau+0.000115428)}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (1920​2​τ3/2−40​2​τ+20)​1024​π​(τ+8.49155)​(τ+0.000115004)19202superscript𝜏32402𝜏201024𝜋𝜏8.49155𝜏0.000115004\displaystyle\left(1920\sqrt{2}\tau^{3/2}-40\sqrt{2}\sqrt{\tau}+20\right)\sqrt{1024\pi(\tau+8.49155)(\tau+0.000115004)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 256​π​(τ+33.8415)​(τ+0.000115428)+limit-from256𝜋𝜏33.8415𝜏0.000115428\displaystyle\sqrt{256\pi(\tau+33.8415)(\tau+0.000115428)}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 117482.τ2+14376.6⩽formulae-sequence117482superscript𝜏214376.6absent\displaystyle 117482.\tau^{2}+14376.6\ \leqslant |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −3.16357×106​τ3/2−608588​τ5/2−8.34635×106​τ7/2+3.16357superscript106superscript𝜏32608588superscript𝜏52limit-from8.34635superscript106superscript𝜏72\displaystyle-3.16357\times 10^{6}\tau^{3/2}-608588\tau^{5/2}-8.34635\times 10^{6}\tau^{7/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (940121​τ3/2+221873​τ5/2+990.171)​1024​π​1.00014​(τ+8.49155)​τ+limit-from940121superscript𝜏32221873superscript𝜏52990.1711024𝜋1.00014𝜏8.49155𝜏\displaystyle\left(940121\tau^{3/2}+221873\tau^{5/2}+990.171\right)\sqrt{1024\pi 1.00014(\tau+8.49155)\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (1920​2​τ3/2−40​2​τ+20)​256​π​1.00014​(τ+33.8415)​τ​1024​π​1.00014​(τ+8.49155)​τ+limit-from19202superscript𝜏32402𝜏20256𝜋1.00014𝜏33.8415𝜏1024𝜋1.00014𝜏8.49155𝜏\displaystyle\left(1920\sqrt{2}\tau^{3/2}-40\sqrt{2}\sqrt{\tau}+20\right)\sqrt{256\pi 1.00014(\tau+33.8415)\tau}\sqrt{1024\pi 1.00014(\tau+8.49155)\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−1.86491×106​τ3/2−303446​τ5/2+3215.89)​256​π​(τ+33.8415)​τ+limit-from1.86491superscript106superscript𝜏32303446superscript𝜏523215.89256𝜋𝜏33.8415𝜏\displaystyle\left(-1.86491\times 10^{6}\tau^{3/2}-303446\tau^{5/2}+3215.89\right)\sqrt{256\pi(\tau+33.8415)\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 117482.τ2+14376.6=formulae-sequence117482superscript𝜏214376.6absent\displaystyle 117482.\tau^{2}+14376.6\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −3.16357×106​τ3/2−608588​τ5/2−8.34635×106​τ7/2+3.16357superscript106superscript𝜏32608588superscript𝜏52limit-from8.34635superscript106superscript𝜏72\displaystyle-3.16357\times 10^{6}\tau^{3/2}-608588\tau^{5/2}-8.34635\times 10^{6}\tau^{7/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−91003​τ3/2+4.36814×106​τ5/2+32174.4​τ)​τ+8.49155​τ+33.8415+117482.τ2+formulae-sequence91003superscript𝜏324.36814superscript106superscript𝜏5232174.4𝜏𝜏8.49155𝜏33.8415117482limit-fromsuperscript𝜏2\displaystyle\left(-91003\tau^{3/2}+4.36814\times 10^{6}\tau^{5/2}+32174.4\tau\right)\sqrt{\tau+8.49155}\sqrt{\tau+33.8415}+117482.\tau^{2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (1.25852×107​τ3+5.33261×107​τ2+56165.1​τ)​τ+8.49155+limit-from1.25852superscript107superscript𝜏35.33261superscript107superscript𝜏256165.1𝜏𝜏8.49155\displaystyle\left(1.25852\times 10^{7}\tau^{3}+5.33261\times 10^{7}\tau^{2}+56165.1\sqrt{\tau}\right)\sqrt{\tau+8.49155}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−8.60549×106​τ3−5.28876×107​τ2+91200.4​τ)​τ+33.8415+14376.6⩽8.60549superscript106superscript𝜏35.28876superscript107superscript𝜏291200.4𝜏𝜏33.841514376.6absent\displaystyle\left(-8.60549\times 10^{6}\tau^{3}-5.28876\times 10^{7}\tau^{2}+91200.4\sqrt{\tau}\right)\sqrt{\tau+33.8415}+14376.6\ \leqslant |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 1.25+8.49155​1.25+33.8415​(−91003​τ3/2+4.36814×106​τ5/2+32174.4​τ)+limit-from1.258.491551.2533.841591003superscript𝜏324.36814superscript106superscript𝜏5232174.4𝜏\displaystyle\sqrt{1.25+8.49155}\sqrt{1.25+33.8415}\left(-91003\tau^{3/2}+4.36814\times 10^{6}\tau^{5/2}+32174.4\tau\right)+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 1.25+8.49155​(1.25852×107​τ3+5.33261×107​τ2+56165.1​τ)+limit-from1.258.491551.25852superscript107superscript𝜏35.33261superscript107superscript𝜏256165.1𝜏\displaystyle\sqrt{1.25+8.49155}\left(1.25852\times 10^{7}\tau^{3}+5.33261\times 10^{7}\tau^{2}+56165.1\sqrt{\tau}\right)+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 0.8+33.8415​(−8.60549×106​τ3−5.28876×107​τ2+91200.4​τ)−limit-from0.833.84158.60549superscript106superscript𝜏35.28876superscript107superscript𝜏291200.4𝜏\displaystyle\sqrt{0.8+33.8415}\left(-8.60549\times 10^{6}\tau^{3}-5.28876\times 10^{7}\tau^{2}+91200.4\sqrt{\tau}\right)- |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 3.16357×106​τ3/2−608588​τ5/2−8.34635×106​τ7/2+117482.τ2+14376.6=formulae-sequence3.16357superscript106superscript𝜏32608588superscript𝜏528.34635superscript106superscript𝜏72117482superscript𝜏214376.6absent\displaystyle 3.16357\times 10^{6}\tau^{3/2}-608588\tau^{5/2}-8.34635\times 10^{6}\tau^{7/2}+117482.\tau^{2}+14376.6\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −4.84613×106​τ3/2+8.01543×107​τ5/2−8.34635×106​τ7/2−4.84613superscript106superscript𝜏328.01543superscript107superscript𝜏52limit-from8.34635superscript106superscript𝜏72\displaystyle-4.84613\times 10^{6}\tau^{3/2}+8.01543\times 10^{7}\tau^{5/2}-8.34635\times 10^{6}\tau^{7/2}- |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 1.13691×107​τ3−1.44725×108​τ2+1.13691superscript107superscript𝜏3limit-from1.44725superscript108superscript𝜏2\displaystyle 1.13691\times 10^{7}\tau^{3}-1.44725\times 10^{8}\tau^{2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 594875.τ+712078.τ+14376.6⩽formulae-sequence594875𝜏712078𝜏14376.6absent\displaystyle 594875.\tau+712078.\sqrt{\tau}+14376.6\ \leqslant |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 14376.6​τ3/20.83/2−4.84613×106​τ3/2+14376.6superscript𝜏32superscript0.832limit-from4.84613superscript106superscript𝜏32\displaystyle\frac{14376.6\tau^{3/2}}{0.8^{3/2}}-4.84613\times 10^{6}\tau^{3/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 8.01543×107​τ5/2−8.34635×106​τ7/2−8.01543superscript107superscript𝜏52limit-from8.34635superscript106superscript𝜏72\displaystyle 8.01543\times 10^{7}\tau^{5/2}-8.34635\times 10^{6}\tau^{7/2}- |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 1.13691×107​τ3−1.44725×108​τ2+594875.τ​τ0.8+712078.τ​τ0.8=1.13691superscript107superscript𝜏31.44725superscript108superscript𝜏2formulae-sequence594875𝜏𝜏0.8formulae-sequence712078𝜏𝜏0.8absent\displaystyle 1.13691\times 10^{7}\tau^{3}-1.44725\times 10^{8}\tau^{2}+\frac{594875.\sqrt{\tau}\tau}{\sqrt{0.8}}+\frac{712078.\tau\sqrt{\tau}}{0.8}\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −3.1311⋅106​τ3/2−1.44725⋅108​τ2+8.01543⋅107​τ5/2−1.13691⋅107​τ3−⋅3.1311superscript106superscript𝜏32⋅1.44725superscript108superscript𝜏2⋅8.01543superscript107superscript𝜏52limit-from⋅1.13691superscript107superscript𝜏3\displaystyle-3.1311\cdot 10^{6}\tau^{3/2}-1.44725\cdot 10^{8}\tau^{2}+8.01543\cdot 10^{7}\tau^{5/2}-1.13691\cdot 10^{7}\tau^{3}- |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 8.34635⋅106​τ7/2⩽⋅8.34635superscript106superscript𝜏72absent\displaystyle 8.34635\cdot 10^{6}\tau^{7/2}\ \leqslant |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −3.1311×106​τ3/2+8.01543×107​1.25​τ5/2τ−3.1311superscript106superscript𝜏32limit-from8.01543superscript1071.25superscript𝜏52𝜏\displaystyle-3.1311\times 10^{6}\tau^{3/2}+\frac{8.01543\times 10^{7}\sqrt{1.25}\tau^{5/2}}{\sqrt{\tau}}- |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 8.34635×106​τ7/2−1.13691×107​τ3−1.44725×108​τ2=8.34635superscript106superscript𝜏721.13691superscript107superscript𝜏31.44725superscript108superscript𝜏2absent\displaystyle 8.34635\times 10^{6}\tau^{7/2}-1.13691\times 10^{7}\tau^{3}-1.44725\times 10^{8}\tau^{2}\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −3.1311×106​τ3/2−8.34635×106​τ7/2−1.13691×107​τ3−5.51094×107​τ2​2< 0.3.1311superscript106superscript𝜏328.34635superscript106superscript𝜏721.13691superscript107superscript𝜏35.51094superscript107superscript𝜏22 0\displaystyle-3.1311\times 10^{6}\tau^{3/2}-8.34635\times 10^{6}\tau^{7/2}-1.13691\times 10^{7}\tau^{3}-5.51094\times 10^{7}\tau^{2}2\ <\ 0\ . |  |

First we expanded the term (multiplied it out).
The we put the terms multiplied by the same square root into brackets.
The next inequality sign stems from inserting the maximal value of 1.251.251.25 for τ𝜏\tau for
some positive terms and value of 0.80.80.8 for negative terms.
These terms are then expanded at the ==-sign.
The next equality factors the terms under the squared root.
We decreased the negative term by setting
τ=τ+0.00011542𝜏𝜏0.00011542\tau=\tau+0.00011542 under the root.
We increased positive terms by setting
τ+0.00011542=1.00014​τ𝜏0.000115421.00014𝜏\tau+0.00011542=1.00014\tau and
τ+0.000115004=1.00014​τ𝜏0.0001150041.00014𝜏\tau+0.000115004=1.00014\tau
under the root for positive terms.
The positive terms are increase, since
0.8+0.000115420.8<1.0001420.80.000115420.81.000142\frac{0.8+0.00011542}{0.8}<1.000142, thus
τ+0.000115004<τ+0.00011542⩽1.00014​τ𝜏0.000115004𝜏0.000115421.00014𝜏\tau+0.000115004<\tau+0.00011542\leqslant 1.00014\tau.
For the next inequality we decreased negative terms by inserting
τ=0.8𝜏0.8\tau=0.8 and increased positive terms by inserting
τ=1.25𝜏1.25\tau=1.25. The next equality expands the terms.
We use upper bound of 1.251.251.25 and lower bound of 0.80.80.8 to obtain terms with
corresponding exponents of τ𝜏\tau.

Consequently, the derivative of

|  |  |  |  |
| --- | --- | --- | --- |
|  | τ​(e(μ​ω+ν​τ2​ν​τ)2​erfc⁡(μ​ω+ν​τ2​ν​τ)−2​e(μ​ω+2​ν​τ2​ν​τ)2​erfc⁡(μ​ω+2​ν​τ2​ν​τ))𝜏superscript𝑒superscript𝜇𝜔𝜈𝜏2𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏2superscript𝑒superscript𝜇𝜔2𝜈𝜏2𝜈𝜏2erfc𝜇𝜔2𝜈𝜏2𝜈𝜏\displaystyle\tau\left(e^{\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-2e^{\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\right) |  | (327) |

with respect to τ𝜏\tau is smaller than zero for maximal ν=0.16𝜈0.16\nu=0.16.

Next,
we consider the function for
the largest ν=0.24𝜈0.24\nu=0.24 and the largest y=μ​ω=0.01𝑦𝜇𝜔0.01y=\mu\omega=0.01
for determining the derivative with respect to τ𝜏\tau.
However we assume 0.9⩽τ0.9𝜏0.9\leqslant\tau, in order to restrict the
domain of τ𝜏\tau.

The expression becomes

|  |  |  |  |
| --- | --- | --- | --- |
|  | τ​(e(24​τ100+11002​24​τ100)2​erfc⁡(24​τ100+11002​24​τ100)−e(2 24​τ100+11002​24​τ100)2​erfc⁡(2 24​τ100+11002​24​τ100)).𝜏superscript𝑒superscript24𝜏1001100224𝜏1002erfc24𝜏1001100224𝜏100superscript𝑒superscript224𝜏1001100224𝜏1002erfc224𝜏1001100224𝜏100\displaystyle\tau\left(e^{\left(\frac{\frac{24\tau}{100}+\frac{1}{100}}{\sqrt{2}\sqrt{\frac{24\tau}{100}}}\right)^{2}}\operatorname{erfc}\left(\frac{\frac{24\tau}{100}+\frac{1}{100}}{\sqrt{2}\sqrt{\frac{24\tau}{100}}}\right)-e^{\left(\frac{\frac{2\ 24\tau}{100}+\frac{1}{100}}{\sqrt{2}\sqrt{\frac{24\tau}{100}}}\right)^{2}}\operatorname{erfc}\left(\frac{\frac{2\ 24\tau}{100}+\frac{1}{100}}{\sqrt{2}\sqrt{\frac{24\tau}{100}}}\right)\right)\ . |  | (328) |

The derivative with respect to τ𝜏\tau is

|  |  |  |  |
| --- | --- | --- | --- |
|  | (π(e(24​τ+1)24800​τ(192τ(3τ+25)−1)erfc(24​τ+140​3​τ)−\displaystyle\left(\sqrt{\pi}\left(e^{\frac{(24\tau+1)^{2}}{4800\tau}}(192\tau(3\tau+25)-1)\operatorname{erfc}\left(\frac{24\tau+1}{40\sqrt{3}\sqrt{\tau}}\right)-\right.\right. |  | (329) |
|  |  |  |
| --- | --- | --- |
|  | 2e(48​τ+1)24800​τ(192τ(12τ+25)−1)erfc(48​τ+140​3​τ))+403(72τ−1)τ)\displaystyle\left.\left.2e^{\frac{(48\tau+1)^{2}}{4800\tau}}(192\tau(12\tau+25)-1)\operatorname{erfc}\left(\frac{48\tau+1}{40\sqrt{3}\sqrt{\tau}}\right)\right)+40\sqrt{3}(72\tau-1)\sqrt{\tau}\right) |  |
|  |  |  |
| --- | --- | --- |
|  | (4800​π​τ)−1.superscript4800𝜋𝜏1\displaystyle\left(4800\sqrt{\pi}\tau\right)^{-1}\ . |  |

We are considering only the numerator and use again the approximation
of Ren and MacKenzie, [[30](#bib.bib30)].
The error analysis on the whole numerator gives an approximation error 14<E<3214𝐸3214<E<32. Therefore
we add 32 to the numerator when we use the approximation of Ren and MacKenzie, [[30](#bib.bib30)].
We obtain the inequalities:

|  |  |  |  |
| --- | --- | --- | --- |
|  | π(e(24​τ+1)24800​τ(192τ(3τ+25)−1)erfc(24​τ+140​3​τ)−\displaystyle\sqrt{\pi}\left(e^{\frac{(24\tau+1)^{2}}{4800\tau}}(192\tau(3\tau+25)-1)\operatorname{erfc}\left(\frac{24\tau+1}{40\sqrt{3}\sqrt{\tau}}\right)\ -\right. |  | (330) |
|  |  |  |
| --- | --- | --- |
|  | 2e(48​τ+1)24800​τ(192τ(12τ+25)−1)erfc(48​τ+140​3​τ))+403(72τ−1)τ⩽\displaystyle\left.2e^{\frac{(48\tau+1)^{2}}{4800\tau}}(192\tau(12\tau+25)-1)\operatorname{erfc}\left(\frac{48\tau+1}{40\sqrt{3}\sqrt{\tau}}\right)\right)+40\sqrt{3}(72\tau-1)\sqrt{\tau}\ \leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | π(2.911​(192​τ​(3​τ+25)−1)π​(2.911−1)​(24​τ+1)40​3​τ+π​(24​τ+140​3​τ)2+2.9112−\displaystyle\sqrt{\pi}\left(\frac{2.911(192\tau(3\tau+25)-1)}{\frac{\sqrt{\pi}(2.911-1)(24\tau+1)}{40\sqrt{3}\sqrt{\tau}}+\sqrt{\pi\left(\frac{24\tau+1}{40\sqrt{3}\sqrt{\tau}}\right)^{2}+2.911^{2}}}\ -\right. |  |
|  |  |  |
| --- | --- | --- |
|  | 2⋅2.911​(192​τ​(12​τ+25)−1)π​(2.911−1)​(48​τ+1)40​3​τ+π​(48​τ+140​3​τ)2+2.9112)+\displaystyle\left.\frac{2\cdot 2.911(192\tau(12\tau+25)-1)}{\frac{\sqrt{\pi}(2.911-1)(48\tau+1)}{40\sqrt{3}\sqrt{\tau}}+\sqrt{\pi\left(\frac{48\tau+1}{40\sqrt{3}\sqrt{\tau}}\right)^{2}+2.911^{2}}}\right)+ |  |
|  |  |  |
| --- | --- | --- |
|  | 40​3​(72​τ−1)​τ+32=40372𝜏1𝜏32absent\displaystyle 40\sqrt{3}(72\tau-1)\sqrt{\tau}+32\ = |  |
|  |  |  |
| --- | --- | --- |
|  | π((192​τ​(3​τ+25)−1)​(40​3​2.911​τ)π​(2.911−1)​(24​τ+1)+(40​3​2.911​τ)2+π​(24​τ+1)2−\displaystyle\sqrt{\pi}\left(\frac{(192\tau(3\tau+25)-1)\left(40\sqrt{3}2.911\sqrt{\tau}\right)}{\sqrt{\pi}(2.911-1)(24\tau+1)+\sqrt{\left(40\sqrt{3}2.911\sqrt{\tau}\right)^{2}+\pi(24\tau+1)^{2}}}\ -\right. |  |
|  |  |  |
| --- | --- | --- |
|  | 2​(192​τ​(12​τ+25)−1)​(40​3​2.911​τ)π​(2.911−1)​(48​τ+1)+(40​3​2.911​τ)2+π​(48​τ+1)2)+\displaystyle\left.\frac{2(192\tau(12\tau+25)-1)\left(40\sqrt{3}2.911\sqrt{\tau}\right)}{\sqrt{\pi}(2.911-1)(48\tau+1)+\sqrt{\left(40\sqrt{3}2.911\sqrt{\tau}\right)^{2}+\pi(48\tau+1)^{2}}}\right)+ |  |
|  |  |  |
| --- | --- | --- |
|  | 40​3​(72​τ−1)​τ+32=40372𝜏1𝜏32absent\displaystyle 40\sqrt{3}(72\tau-1)\sqrt{\tau}+32\ = |  |
|  |  |  |
| --- | --- | --- |
|  | ((403(72τ−1)τ+32)(π(2.911−1)(24τ+1)+(40​3​2.911​τ)2+π​(24​τ+1)2)\displaystyle\left(\left(40\sqrt{3}(72\tau-1)\sqrt{\tau}+32\right)\left(\sqrt{\pi}(2.911-1)(24\tau+1)+\sqrt{\left(40\sqrt{3}2.911\sqrt{\tau}\right)^{2}+\pi(24\tau+1)^{2}}\right)\right. |  |
|  |  |  |
| --- | --- | --- |
|  | (π​(2.911−1)​(48​τ+1)+(40​3​2.911​τ)2+π​(48​τ+1)2)+limit-from𝜋2.911148𝜏1superscript4032.911𝜏2𝜋superscript48𝜏12\displaystyle\left.\left(\sqrt{\pi}(2.911-1)(48\tau+1)+\sqrt{\left(40\sqrt{3}2.911\sqrt{\tau}\right)^{2}+\pi(48\tau+1)^{2}}\right)+\right. |  |
|  |  |  |
| --- | --- | --- |
|  | 2.911⋅40​3​π​(192​τ​(3​τ+25)−1)​τ⋅2.911403𝜋192𝜏3𝜏251𝜏\displaystyle\left.2.911\cdot 40\sqrt{3}\sqrt{\pi}(192\tau(3\tau+25)-1)\sqrt{\tau}\right. |  |
|  |  |  |
| --- | --- | --- |
|  | (π​(2.911−1)​(48​τ+1)+(40​3​2.911​τ)2+π​(48​τ+1)2)−limit-from𝜋2.911148𝜏1superscript4032.911𝜏2𝜋superscript48𝜏12\displaystyle\left.\left(\sqrt{\pi}(2.911-1)(48\tau+1)+\sqrt{\left(40\sqrt{3}2.911\sqrt{\tau}\right)^{2}+\pi(48\tau+1)^{2}}\right)-\right. |  |
|  |  |  |
| --- | --- | --- |
|  | 2​π​40​3​2.911​(192​τ​(12​τ+25)−1)2𝜋4032.911192𝜏12𝜏251\displaystyle\left.2\sqrt{\pi}40\sqrt{3}2.911(192\tau(12\tau+25)-1)\right. |  |
|  |  |  |
| --- | --- | --- |
|  | τ(π(2.911−1)(24τ+1)+(40​3​2.911​τ)2+π​(24​τ+1)2))\displaystyle\left.\sqrt{\tau}\left(\sqrt{\pi}(2.911-1)(24\tau+1)+\sqrt{\left(40\sqrt{3}2.911\sqrt{\tau}\right)^{2}+\pi(24\tau+1)^{2}}\right)\right) |  |
|  |  |  |
| --- | --- | --- |
|  | ((π(2.911−1)(24τ+1)+(40​3​2.911​τ)2+π​(24​τ+1)2)\displaystyle\left(\left(\sqrt{\pi}(2.911-1)(24\tau+1)+\sqrt{\left(40\sqrt{3}2.911\sqrt{\tau}\right)^{2}+\pi(24\tau+1)^{2}}\right)\right. |  |
|  |  |  |
| --- | --- | --- |
|  | (π(2.911−1)(48τ+1)+(40​3​2.911​τ)2+π​(48​τ+1)2))−1.\displaystyle\left.\left(\sqrt{\pi}(2.911-1)(48\tau+1)+\sqrt{\left(40\sqrt{3}2.911\sqrt{\tau}\right)^{2}+\pi(48\tau+1)^{2}}\right)\right)^{-1}\ . |  |

After applying the approximation
of Ren and MacKenzie, [[30](#bib.bib30)] and adding 200,
we first factored out 40​3​τ403𝜏40\sqrt{3}\sqrt{\tau}.
Then we brought all terms to the same denominator.

We now consider the numerator:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | (40​3​(72​τ−1)​τ+32)​(π​(2.911−1)​(24​τ+1)+(40​3​2.911​τ)2+π​(24​τ+1)2)40372𝜏1𝜏32𝜋2.911124𝜏1superscript4032.911𝜏2𝜋superscript24𝜏12\displaystyle\left(40\sqrt{3}(72\tau-1)\sqrt{\tau}+32\right)\left(\sqrt{\pi}(2.911-1)(24\tau+1)+\sqrt{\left(40\sqrt{3}2.911\sqrt{\tau}\right)^{2}+\pi(24\tau+1)^{2}}\right) |  | (331) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (π​(2.911−1)​(48​τ+1)+(40​3​2.911​τ)2+π​(48​τ+1)2)+limit-from𝜋2.911148𝜏1superscript4032.911𝜏2𝜋superscript48𝜏12\displaystyle\left(\sqrt{\pi}(2.911-1)(48\tau+1)+\sqrt{\left(40\sqrt{3}2.911\sqrt{\tau}\right)^{2}+\pi(48\tau+1)^{2}}\right)\ + |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2.911⋅40​3​π​(192​τ​(3​τ+25)−1)​τ⋅2.911403𝜋192𝜏3𝜏251𝜏\displaystyle 2.911\cdot 40\sqrt{3}\sqrt{\pi}(192\tau(3\tau+25)-1)\sqrt{\tau} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (π​(2.911−1)​(48​τ+1)+(40​3​2.911​τ)2+π​(48​τ+1)2)−limit-from𝜋2.911148𝜏1superscript4032.911𝜏2𝜋superscript48𝜏12\displaystyle\left(\sqrt{\pi}(2.911-1)(48\tau+1)+\sqrt{\left(40\sqrt{3}2.911\sqrt{\tau}\right)^{2}+\pi(48\tau+1)^{2}}\right)- |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2​π​40​3​2.911​(192​τ​(12​τ+25)−1)​τ2𝜋4032.911192𝜏12𝜏251𝜏\displaystyle 2\sqrt{\pi}40\sqrt{3}2.911(192\tau(12\tau+25)-1)\sqrt{\tau} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (π​(2.911−1)​(24​τ+1)+(40​3​2.911​τ)2+π​(24​τ+1)2)=𝜋2.911124𝜏1superscript4032.911𝜏2𝜋superscript24𝜏12absent\displaystyle\left(\sqrt{\pi}(2.911-1)(24\tau+1)+\sqrt{\left(40\sqrt{3}2.911\sqrt{\tau}\right)^{2}+\pi(24\tau+1)^{2}}\right)\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −3.42607×106​π​(24​τ+1)2+40674.8​τ​τ3/2+limit-from3.42607superscript106𝜋superscript24𝜏1240674.8𝜏superscript𝜏32\displaystyle-3.42607\times 10^{6}\sqrt{\pi(24\tau+1)^{2}+40674.8\tau}\tau^{3/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2880​3​π​(24​τ+1)2+40674.8​τ​π​(48​τ+1)2+40674.8​τ​τ3/2+limit-from28803𝜋superscript24𝜏1240674.8𝜏𝜋superscript48𝜏1240674.8𝜏superscript𝜏32\displaystyle 2880\sqrt{3}\sqrt{\pi(24\tau+1)^{2}+40674.8\tau}\sqrt{\pi(48\tau+1)^{2}+40674.8\tau}\tau^{3/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 1.72711×106​π​(48​τ+1)2+40674.8​τ​τ3/2−5.81185×106​τ3/2−1.72711superscript106𝜋superscript48𝜏1240674.8𝜏superscript𝜏32limit-from5.81185superscript106superscript𝜏32\displaystyle 1.72711\times 10^{6}\sqrt{\pi(48\tau+1)^{2}+40674.8\tau}\tau^{3/2}-5.81185\times 10^{6}\tau^{3/2}\ - |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 836198​π​(24​τ+1)2+40674.8​τ​τ5/2+611410​π​(48​τ+1)2+40674.8​τ​τ5/2−836198𝜋superscript24𝜏1240674.8𝜏superscript𝜏52limit-from611410𝜋superscript48𝜏1240674.8𝜏superscript𝜏52\displaystyle 836198\sqrt{\pi(24\tau+1)^{2}+40674.8\tau}\tau^{5/2}+611410\sqrt{\pi(48\tau+1)^{2}+40674.8\tau}\tau^{5/2}- |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 1.67707×106​τ5/2−limit-from1.67707superscript106superscript𝜏52\displaystyle 1.67707\times 10^{6}\tau^{5/2}\ - |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 3.44998×107​τ7/2+422935.τ2+5202.68​π​(24​τ+1)2+40674.8​τ​τ+formulae-sequence3.44998superscript107superscript𝜏72422935superscript𝜏2limit-from5202.68𝜋superscript24𝜏1240674.8𝜏𝜏\displaystyle 3.44998\times 10^{7}\tau^{7/2}+422935.\tau^{2}+5202.68\sqrt{\pi(24\tau+1)^{2}+40674.8\tau}\tau+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 2601.34​π​(48​τ+1)2+40674.8​τ​τ+limit-from2601.34𝜋superscript48𝜏1240674.8𝜏𝜏\displaystyle 2601.34\sqrt{\pi(48\tau+1)^{2}+40674.8\tau}\tau\ + |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 26433.4​τ+415.94​τ+480.268​τ​π​(24​τ+1)2+40674.8​τ+26433.4𝜏415.94𝜏limit-from480.268𝜏𝜋superscript24𝜏1240674.8𝜏\displaystyle 26433.4\tau+415.94\sqrt{\tau}+480.268\sqrt{\tau}\sqrt{\pi(24\tau+1)^{2}+40674.8\tau}\ + |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 108.389​π​(24​τ+1)2+40674.8​τ−592.138​τ​π​(48​τ+1)2+40674.8​τ−108.389𝜋superscript24𝜏1240674.8𝜏limit-from592.138𝜏𝜋superscript48𝜏1240674.8𝜏\displaystyle 108.389\sqrt{\pi(24\tau+1)^{2}+40674.8\tau}-592.138\sqrt{\tau}\sqrt{\pi(48\tau+1)^{2}+40674.8\tau}- |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 40​3​τ​π​(24​τ+1)2+40674.8​τ​π​(48​τ+1)2+40674.8​τ+limit-from403𝜏𝜋superscript24𝜏1240674.8𝜏𝜋superscript48𝜏1240674.8𝜏\displaystyle 40\sqrt{3}\sqrt{\tau}\sqrt{\pi(24\tau+1)^{2}+40674.8\tau}\sqrt{\pi(48\tau+1)^{2}+40674.8\tau}\ + |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 32​π​(24​τ+1)2+40674.8​τ​π​(48​τ+1)2+40674.8​τ+limit-from32𝜋superscript24𝜏1240674.8𝜏𝜋superscript48𝜏1240674.8𝜏\displaystyle 32\sqrt{\pi(24\tau+1)^{2}+40674.8\tau}\sqrt{\pi(48\tau+1)^{2}+40674.8\tau}\ + |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 108.389​π​(48​τ+1)2+40674.8​τ+367.131=108.389𝜋superscript48𝜏1240674.8𝜏367.131absent\displaystyle 108.389\sqrt{\pi(48\tau+1)^{2}+40674.8\tau}+367.131\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −5.81185×106​τ3/2−1.67707×106​τ5/2−3.44998×107​τ7/2+5.81185superscript106superscript𝜏321.67707superscript106superscript𝜏52limit-from3.44998superscript107superscript𝜏72\displaystyle-5.81185\times 10^{6}\tau^{3/2}-1.67707\times 10^{6}\tau^{5/2}-3.44998\times 10^{7}\tau^{7/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−3.42607×106​τ3/2−836198​τ5/2+5202.68​τ+480.268​τ+108.389)3.42607superscript106superscript𝜏32836198superscript𝜏525202.68𝜏480.268𝜏108.389\displaystyle\left(-3.42607\times 10^{6}\tau^{3/2}-836198\tau^{5/2}+5202.68\tau+480.268\sqrt{\tau}+108.389\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | π​(24​τ+1)2+40674.8​τ+limit-from𝜋superscript24𝜏1240674.8𝜏\displaystyle\sqrt{\pi(24\tau+1)^{2}+40674.8\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (1.72711×106​τ3/2+611410​τ5/2+2601.34​τ−592.138​τ+108.389)1.72711superscript106superscript𝜏32611410superscript𝜏522601.34𝜏592.138𝜏108.389\displaystyle\left(1.72711\times 10^{6}\tau^{3/2}+611410\tau^{5/2}+2601.34\tau-592.138\sqrt{\tau}+108.389\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | π​(48​τ+1)2+40674.8​τ+limit-from𝜋superscript48𝜏1240674.8𝜏\displaystyle\sqrt{\pi(48\tau+1)^{2}+40674.8\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (2880​3​τ3/2−40​3​τ+32)​π​(24​τ+1)2+40674.8​τ​π​(48​τ+1)2+40674.8​τ+limit-from28803superscript𝜏32403𝜏32𝜋superscript24𝜏1240674.8𝜏𝜋superscript48𝜏1240674.8𝜏\displaystyle\left(2880\sqrt{3}\tau^{3/2}-40\sqrt{3}\sqrt{\tau}+32\right)\sqrt{\pi(24\tau+1)^{2}+40674.8\tau}\sqrt{\pi(48\tau+1)^{2}+40674.8\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 422935.τ2+26433.4​τ+415.94​τ+367.131⩽formulae-sequence422935superscript𝜏226433.4𝜏415.94𝜏367.131absent\displaystyle 422935.\tau^{2}+26433.4\tau+415.94\sqrt{\tau}+367.131\ \leqslant |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −5.81185×106​τ3/2−1.67707×106​τ5/2−3.44998×107​τ7/2+5.81185superscript106superscript𝜏321.67707superscript106superscript𝜏52limit-from3.44998superscript107superscript𝜏72\displaystyle-5.81185\times 10^{6}\tau^{3/2}-1.67707\times 10^{6}\tau^{5/2}-3.44998\times 10^{7}\tau^{7/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−3.42607×106​τ3/2−836198​τ5/2+480.268​1.25+1.255202.68+108.389)3.42607superscript106superscript𝜏32836198superscript𝜏52480.2681.251.255202.68108.389\displaystyle\left(-3.42607\times 10^{6}\tau^{3/2}-836198\tau^{5/2}+480.268\sqrt{1.25}+1.255202.68+108.389\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | π​(24​τ+1)2+40674.8​τ+limit-from𝜋superscript24𝜏1240674.8𝜏\displaystyle\sqrt{\pi(24\tau+1)^{2}+40674.8\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (1.72711×106​τ3/2+611410​τ5/2−592.138​0.9+1.252601.34+108.389)1.72711superscript106superscript𝜏32611410superscript𝜏52592.1380.91.252601.34108.389\displaystyle\left(1.72711\times 10^{6}\tau^{3/2}+611410\tau^{5/2}-592.138\sqrt{0.9}+1.252601.34+108.389\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | π​(48​τ+1)2+40674.8​τ+limit-from𝜋superscript48𝜏1240674.8𝜏\displaystyle\sqrt{\pi(48\tau+1)^{2}+40674.8\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (2880​3​τ3/2−40​3​τ+32)​π​(24​τ+1)2+40674.8​τ​π​(48​τ+1)2+40674.8​τ+limit-from28803superscript𝜏32403𝜏32𝜋superscript24𝜏1240674.8𝜏𝜋superscript48𝜏1240674.8𝜏\displaystyle\left(2880\sqrt{3}\tau^{3/2}-40\sqrt{3}\sqrt{\tau}+32\right)\sqrt{\pi(24\tau+1)^{2}+40674.8\tau}\sqrt{\pi(48\tau+1)^{2}+40674.8\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 422935​τ2+415.94​1.25+1.2526433.4+367.131=422935superscript𝜏2415.941.251.2526433.4367.131absent\displaystyle 422935\tau^{2}+415.94\sqrt{1.25}+1.2526433.4+367.131\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −5.81185×106​τ3/2−1.67707×106​τ5/2−3.44998×107​τ7/2+5.81185superscript106superscript𝜏321.67707superscript106superscript𝜏52limit-from3.44998superscript107superscript𝜏72\displaystyle-5.81185\times 10^{6}\tau^{3/2}-1.67707\times 10^{6}\tau^{5/2}-3.44998\times 10^{7}\tau^{7/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−3.42607×106​τ3/2−836198​τ5/2+7148.69)​π​(24​τ+1)2+40674.8​τ+limit-from3.42607superscript106superscript𝜏32836198superscript𝜏527148.69𝜋superscript24𝜏1240674.8𝜏\displaystyle\left(-3.42607\times 10^{6}\tau^{3/2}-836198\tau^{5/2}+7148.69\right)\sqrt{\pi(24\tau+1)^{2}+40674.8\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (1.72711×106​τ3/2+611410​τ5/2+2798.31)​π​(48​τ+1)2+40674.8​τ+limit-from1.72711superscript106superscript𝜏32611410superscript𝜏522798.31𝜋superscript48𝜏1240674.8𝜏\displaystyle\left(1.72711\times 10^{6}\tau^{3/2}+611410\tau^{5/2}+2798.31\right)\sqrt{\pi(48\tau+1)^{2}+40674.8\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (2880​3​τ3/2−40​3​τ+32)​π​(24​τ+1)2+40674.8​τ​π​(48​τ+1)2+40674.8​τ+limit-from28803superscript𝜏32403𝜏32𝜋superscript24𝜏1240674.8𝜏𝜋superscript48𝜏1240674.8𝜏\displaystyle\left(2880\sqrt{3}\tau^{3/2}-40\sqrt{3}\sqrt{\tau}+32\right)\sqrt{\pi(24\tau+1)^{2}+40674.8\tau}\sqrt{\pi(48\tau+1)^{2}+40674.8\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 422935​τ2+33874=422935superscript𝜏233874absent\displaystyle 422935\tau^{2}+33874\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −5.81185×106​τ3/2−1.67707×106​τ5/2−3.44998×107​τ7/2+5.81185superscript106superscript𝜏321.67707superscript106superscript𝜏52limit-from3.44998superscript107superscript𝜏72\displaystyle-5.81185\times 10^{6}\tau^{3/2}-1.67707\times 10^{6}\tau^{5/2}-3.44998\times 10^{7}\tau^{7/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (1.72711×106​τ3/2+611410​τ5/2+2798.31)​2304​π​(τ+5.66103)​(τ+0.0000766694)+limit-from1.72711superscript106superscript𝜏32611410superscript𝜏522798.312304𝜋𝜏5.66103𝜏0.0000766694\displaystyle\left(1.72711\times 10^{6}\tau^{3/2}+611410\tau^{5/2}+2798.31\right)\sqrt{2304\pi(\tau+5.66103)(\tau+0.0000766694)}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−3.42607×106​τ3/2−836198​τ5/2+7148.69)​576​π​(τ+22.561)​(τ+0.0000769518)+limit-from3.42607superscript106superscript𝜏32836198superscript𝜏527148.69576𝜋𝜏22.561𝜏0.0000769518\displaystyle\left(-3.42607\times 10^{6}\tau^{3/2}-836198\tau^{5/2}+7148.69\right)\sqrt{576\pi(\tau+22.561)(\tau+0.0000769518)}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (2880​3​τ3/2−40​3​τ+32)​2304​π​(τ+5.66103)​(τ+0.0000766694)28803superscript𝜏32403𝜏322304𝜋𝜏5.66103𝜏0.0000766694\displaystyle\left(2880\sqrt{3}\tau^{3/2}-40\sqrt{3}\sqrt{\tau}+32\right)\sqrt{2304\pi(\tau+5.66103)(\tau+0.0000766694)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 576​π​(τ+22.561)​(τ+0.0000769518)+limit-from576𝜋𝜏22.561𝜏0.0000769518\displaystyle\sqrt{576\pi(\tau+22.561)(\tau+0.0000769518)}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 422935​τ2+33874⩽422935superscript𝜏233874absent\displaystyle 422935\tau^{2}+33874\ \leqslant |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −5.81185106​τ3/2−1.67707×106​τ5/2−3.44998×107​τ7/2+superscript5.81185106superscript𝜏321.67707superscript106superscript𝜏52limit-from3.44998superscript107superscript𝜏72\displaystyle-5.8118510^{6}\tau^{3/2}-1.67707\times 10^{6}\tau^{5/2}-3.44998\times 10^{7}\tau^{7/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (1.72711×106​τ3/2+611410​τ5/2+2798.31)​2304​π​1.0001​(τ+5.66103)​τ+limit-from1.72711superscript106superscript𝜏32611410superscript𝜏522798.312304𝜋1.0001𝜏5.66103𝜏\displaystyle\left(1.72711\times 10^{6}\tau^{3/2}+611410\tau^{5/2}+2798.31\right)\sqrt{2304\pi 1.0001(\tau+5.66103)\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (2880​3​τ3/2−40​3​τ+32)​2304​π​1.0001​(τ+5.66103)​τ​576​π​1.0001​(τ+22.561)​τ+limit-from28803superscript𝜏32403𝜏322304𝜋1.0001𝜏5.66103𝜏576𝜋1.0001𝜏22.561𝜏\displaystyle\left(2880\sqrt{3}\tau^{3/2}-40\sqrt{3}\sqrt{\tau}+32\right)\sqrt{2304\pi 1.0001(\tau+5.66103)\tau}\sqrt{576\pi 1.0001(\tau+22.561)\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−3.42607×106​τ3/2−836198​τ5/2+7148.69)3.42607superscript106superscript𝜏32836198superscript𝜏527148.69\displaystyle\left(-3.42607\times 10^{6}\tau^{3/2}-836198\tau^{5/2}+7148.69\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 576​π​(τ+22.561)​τ+limit-from576𝜋𝜏22.561𝜏\displaystyle\sqrt{576\pi(\tau+22.561)\tau}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 422935​τ2+33874.=formulae-sequence422935superscript𝜏233874\displaystyle 422935\tau^{2}+33874.\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −5.81185106​τ3/2−1.67707×106​τ5/2−3.44998×107​τ7/2+superscript5.81185106superscript𝜏321.67707superscript106superscript𝜏52limit-from3.44998superscript107superscript𝜏72\displaystyle-5.8118510^{6}\tau^{3/2}-1.67707\times 10^{6}\tau^{5/2}-3.44998\times 10^{7}\tau^{7/2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−250764.τ3/2+1.8055×107​τ5/2+115823.τ)formulae-sequence250764superscript𝜏321.8055superscript107superscript𝜏52115823𝜏\displaystyle\left(-250764.\tau^{3/2}+1.8055\times 10^{7}\tau^{5/2}+115823.\tau\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | τ+5.66103​τ+22.561+422935.τ2+formulae-sequence𝜏5.66103𝜏22.561422935limit-fromsuperscript𝜏2\displaystyle\sqrt{\tau+5.66103}\sqrt{\tau+22.561}+422935.\tau^{2}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (5.20199×107τ3+1.46946×108τ2+238086.τ)τ+5.66103+\displaystyle\left(5.20199\times 10^{7}\tau^{3}+1.46946\times 10^{8}\tau^{2}+238086.\sqrt{\tau}\right)\sqrt{\tau+5.66103}+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | (−3.55709×107τ3−1.45741×108τ2+304097.τ)τ+22.561+33874.⩽\displaystyle\left(-3.55709\times 10^{7}\tau^{3}-1.45741\times 10^{8}\tau^{2}+304097.\sqrt{\tau}\right)\sqrt{\tau+22.561}+33874.\ \leqslant |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 1.25+5.661031.25+22.561(−250764.τ3/2+1.8055×107τ5/2+115823.τ)+\displaystyle\sqrt{1.25+5.66103}\sqrt{1.25+22.561}\left(-250764.\tau^{3/2}+1.8055\times 10^{7}\tau^{5/2}+115823.\tau\right)+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 1.25+5.66103(5.20199×107τ3+1.46946×108τ2+238086.τ)+\displaystyle\sqrt{1.25+5.66103}\left(5.20199\times 10^{7}\tau^{3}+1.46946\times 10^{8}\tau^{2}+238086.\sqrt{\tau}\right)+ |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 0.9+22.561(−3.55709×107τ3−1.45741×108τ2+304097.τ)−\displaystyle\sqrt{0.9+22.561}\left(-3.55709\times 10^{7}\tau^{3}-1.45741\times 10^{8}\tau^{2}+304097.\sqrt{\tau}\right)- |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 5.81185106​τ3/2−1.67707×106​τ5/2−3.44998×107​τ7/2+422935.τ2+33874.⩽formulae-sequencesuperscript5.81185106superscript𝜏321.67707superscript106superscript𝜏523.44998superscript107superscript𝜏72422935superscript𝜏233874\displaystyle 5.8118510^{6}\tau^{3/2}-1.67707\times 10^{6}\tau^{5/2}-3.44998\times 10^{7}\tau^{7/2}+422935.\tau^{2}+33874.\ \leqslant |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 33874.τ3/20.93/2−9.02866×106​τ3/2+2.29933×108​τ5/2−3.44998×107​τ7/2−formulae-sequence33874superscript𝜏32superscript0.9329.02866superscript106superscript𝜏322.29933superscript108superscript𝜏52limit-from3.44998superscript107superscript𝜏72\displaystyle\frac{33874.\tau^{3/2}}{0.9^{3/2}}-9.02866\times 10^{6}\tau^{3/2}+2.29933\times 10^{8}\tau^{5/2}-3.44998\times 10^{7}\tau^{7/2}- |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 3.5539×107​τ3−3.19193×108​τ2+1.48578×106​τ​τ0.9+2.09884×106​τ​τ0.9=3.5539superscript107superscript𝜏33.19193superscript108superscript𝜏21.48578superscript106𝜏𝜏0.92.09884superscript106𝜏𝜏0.9absent\displaystyle 3.5539\times 10^{7}\tau^{3}-3.19193\times 10^{8}\tau^{2}+\frac{1.48578\times 10^{6}\sqrt{\tau}\tau}{\sqrt{0.9}}+\frac{2.09884\times 10^{6}\tau\sqrt{\tau}}{0.9}\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −5.09079×106​τ3/2+2.29933×108​τ5/2−5.09079superscript106superscript𝜏32limit-from2.29933superscript108superscript𝜏52\displaystyle-5.09079\times 10^{6}\tau^{3/2}+2.29933\times 10^{8}\tau^{5/2}- |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 3.44998×107​τ7/2−3.5539×107​τ3−3.19193×108​τ2⩽3.44998superscript107superscript𝜏723.5539superscript107superscript𝜏33.19193superscript108superscript𝜏2absent\displaystyle 3.44998\times 10^{7}\tau^{7/2}-3.5539\times 10^{7}\tau^{3}-3.19193\times 10^{8}\tau^{2}\ \leqslant |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −5.09079×106​τ3/2+2.29933×108​1.25​τ5/2τ−3.44998×107​τ7/2−5.09079superscript106superscript𝜏322.29933superscript1081.25superscript𝜏52𝜏limit-from3.44998superscript107superscript𝜏72\displaystyle-5.09079\times 10^{6}\tau^{3/2}+\frac{2.29933\times 10^{8}\sqrt{1.25}\tau^{5/2}}{\sqrt{\tau}}-3.44998\times 10^{7}\tau^{7/2}- |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 3.5539×107​τ3−3.19193×108​τ2=3.5539superscript107superscript𝜏33.19193superscript108superscript𝜏2absent\displaystyle 3.5539\times 10^{7}\tau^{3}-3.19193\times 10^{8}\tau^{2}\ = |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −5.09079×106​τ3/2−3.44998×107​τ7/2−3.5539×107​τ3−6.21197×107​τ2< 0.5.09079superscript106superscript𝜏323.44998superscript107superscript𝜏723.5539superscript107superscript𝜏36.21197superscript107superscript𝜏2 0\displaystyle-5.09079\times 10^{6}\tau^{3/2}-3.44998\times 10^{7}\tau^{7/2}-3.5539\times 10^{7}\tau^{3}-6.21197\times 10^{7}\tau^{2}\ <\ 0\ . |  |

First we expanded the term (multiplied it out).
The we put the terms multiplied by the same square root into brackets.
The next inequality sign stems from inserting the maximal value of 1.251.251.25 for τ𝜏\tau for
some positive terms and value of 0.90.90.9 for negative terms.
These terms are then expanded at the ==-sign.
The next equality factors the terms under the squared root.
We decreased the negative term by setting
τ=τ+0.0000769518𝜏𝜏0.0000769518\tau=\tau+0.0000769518 under the root.
We increased positive terms by setting
τ+0.0000769518=1.0000962​τ𝜏0.00007695181.0000962𝜏\tau+0.0000769518=1.0000962\tau and
τ+0.0000766694=1.0000962​τ𝜏0.00007666941.0000962𝜏\tau+0.0000766694=1.0000962\tau
under the root for positive terms.
The positive terms are increase, since
0.8+0.00007695180.8<1.00009620.80.00007695180.81.0000962\frac{0.8+0.0000769518}{0.8}<1.0000962, thus
τ+0.0000766694<τ+0.0000769518⩽1.0000962​τ𝜏0.0000766694𝜏0.00007695181.0000962𝜏\tau+0.0000766694<\tau+0.0000769518\leqslant 1.0000962\tau.
For the next inequality we decreased negative terms by inserting
τ=0.9𝜏0.9\tau=0.9 and increased positive terms by inserting
τ=1.25𝜏1.25\tau=1.25. The next equality expands the terms.
We use upper bound of 1.251.251.25 and lower bound of 0.90.90.9 to obtain terms with
corresponding exponents of τ𝜏\tau.

Consequently, the derivative of

|  |  |  |  |
| --- | --- | --- | --- |
|  | τ​(e(μ​ω+ν​τ2​ν​τ)2​erfc⁡(μ​ω+ν​τ2​ν​τ)−2​e(μ​ω+2​ν​τ2​ν​τ)2​erfc⁡(μ​ω+2​ν​τ2​ν​τ))𝜏superscript𝑒superscript𝜇𝜔𝜈𝜏2𝜈𝜏2erfc𝜇𝜔𝜈𝜏2𝜈𝜏2superscript𝑒superscript𝜇𝜔2𝜈𝜏2𝜈𝜏2erfc𝜇𝜔2𝜈𝜏2𝜈𝜏\displaystyle\tau\left(e^{\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)-2e^{\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)^{2}}\operatorname{erfc}\left(\frac{\mu\omega+2\nu\tau}{\sqrt{2}\sqrt{\nu\tau}}\right)\right) |  | (332) |

with respect to τ𝜏\tau is smaller than zero for maximal
ν=0.24𝜈0.24\nu=0.24 and the domain 0.9⩽τ⩽1.250.9𝜏1.250.9\leqslant\tau\leqslant 1.25.
∎

###### Lemma 47.

In the domain −0.01⩽y⩽0.010.01𝑦0.01-0.01\leqslant y\leqslant 0.01 and 0.64⩽x⩽1.8750.64𝑥1.8750.64\leqslant x\leqslant 1.875,
the function f​(x,y)=e12​(2​y+x)​erfc⁡(x+y2​x)𝑓𝑥𝑦superscript𝑒122𝑦𝑥erfc𝑥𝑦2𝑥f(x,y)=e^{\frac{1}{2}(2y+x)}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2x}}\right) has a global
maximum at y=0.64𝑦0.64y=0.64 and x=−0.01𝑥0.01x=-0.01 and a global minimum at y=1.875𝑦1.875y=1.875 and x=0.01𝑥0.01x=0.01.

###### Proof.

f​(x,y)=e12​(2​y+x)​erfc⁡(x+y2​x)𝑓𝑥𝑦superscript𝑒122𝑦𝑥erfc𝑥𝑦2𝑥f(x,y)=e^{\frac{1}{2}(2y+x)}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2x}}\right) is strictly monotonically decreasing
in x𝑥x, since its derivative with respect to x𝑥x is negative:

|  |  |  |
| --- | --- | --- |
|  | e−y22​x​(π​x3/2​e(x+y)22​x​erfc⁡(x+y2​x)+2​(y−x))2​π​x3/2<0superscript𝑒superscript𝑦22𝑥𝜋superscript𝑥32superscript𝑒superscript𝑥𝑦22𝑥erfc𝑥𝑦2𝑥2𝑦𝑥2𝜋superscript𝑥320\displaystyle\frac{e^{-\frac{y^{2}}{2x}}\left(\sqrt{\pi}x^{3/2}e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)+\sqrt{2}(y-x)\right)}{2\sqrt{\pi}x^{3/2}}<0 |  |
|  |  |  |
| --- | --- | --- |
|  | ⇔π​x3/2​e(x+y)22​x​erfc⁡(x+y2​x)+2​(y−x)<0iffabsent𝜋superscript𝑥32superscript𝑒superscript𝑥𝑦22𝑥erfc𝑥𝑦2𝑥2𝑦𝑥0\displaystyle\iff\sqrt{\pi}x^{3/2}e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)+\sqrt{2}(y-x)<0 |  |
|  |  |  |
| --- | --- | --- |
|  | π​x3/2​e(x+y)22​x​erfc⁡(x+y2​x)+2​(y−x)⩽𝜋superscript𝑥32superscript𝑒superscript𝑥𝑦22𝑥erfc𝑥𝑦2𝑥2𝑦𝑥absent\displaystyle\sqrt{\pi}x^{3/2}e^{\frac{(x+y)^{2}}{2x}}\operatorname{erfc}\left(\frac{x+y}{\sqrt{2}\sqrt{x}}\right)+\sqrt{2}(y-x)\leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | 2​x3/2x+y2​x+(x+y)22​x+4π+y​2−x​2⩽2superscript𝑥32𝑥𝑦2𝑥superscript𝑥𝑦22𝑥4𝜋𝑦2𝑥2absent\displaystyle\frac{2x^{3/2}}{\frac{x+y}{\sqrt{2}\sqrt{x}}+\sqrt{\frac{(x+y)^{2}}{2x}+\frac{4}{\pi}}}+y\sqrt{2}-x\sqrt{2}\leqslant |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 2⋅0.643/20.01+0.642​0.64+(0.01+0.64)22⋅0.64+4π+0.01​2−0.64​2=−0.334658<0.⋅2superscript0.64320.010.6420.64superscript0.010.642⋅20.644𝜋0.0120.6420.3346580\displaystyle\frac{2\cdot 0.64^{3/2}}{\frac{0.01+0.64}{\sqrt{2}\sqrt{0.64}}+\sqrt{\frac{(0.01+0.64)^{2}}{2\cdot 0.64}+\frac{4}{\pi}}}+0.01\sqrt{2}-0.64\sqrt{2}=-0.334658<0. |  | (333) |

The two last inqualities come from applying Abramowitz bounds [22](#Thmtheorem22 "Lemma 22 (Erfc bound from Abramowitz). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks") and from the fact that the expression
2​x3/2x+y2​x+(x+y)22​x+4π+y​2−x​22superscript𝑥32𝑥𝑦2𝑥superscript𝑥𝑦22𝑥4𝜋𝑦2𝑥2\frac{2x^{3/2}}{\frac{x+y}{\sqrt{2}\sqrt{x}}+\sqrt{\frac{(x+y)^{2}}{2x}+\frac{4}{\pi}}}+y\sqrt{2}-x\sqrt{2}
does not change monotonicity in the
domain and hence the maximum must be found at the border. For x=0.64𝑥0.64x=0.64 that maximizes the function f​(x,y)𝑓𝑥𝑦f(x,y) is monotonically in y𝑦y, because
its derivative w.r.t. y𝑦y at x=0.64𝑥0.64x=0.64 is

|  |  |  |
| --- | --- | --- |
|  | ey​(1.37713​erfc⁡(0.883883​y+0.565685)−1.37349​e−0.78125​(y+0.64)2)<0superscript𝑒𝑦1.37713erfc0.883883𝑦0.5656851.37349superscript𝑒0.78125superscript𝑦0.6420\displaystyle e^{y}\left(1.37713\operatorname{erfc}(0.883883y+0.565685)-1.37349e^{-0.78125(y+0.64)^{2}}\right)<0 |  |
|  |  |  |
| --- | --- | --- |
|  | ⇔(1.37713​erfc⁡(0.883883​y+0.565685)−1.37349​e−0.78125​(y+0.64)2)<0iffabsent1.37713erfc0.883883𝑦0.5656851.37349superscript𝑒0.78125superscript𝑦0.6420\displaystyle\iff\left(1.37713\operatorname{erfc}(0.883883y+0.565685)-1.37349e^{-0.78125(y+0.64)^{2}}\right)<0 |  |
|  |  |  |
| --- | --- | --- |
|  | (1.37713​erfc⁡(0.883883​y+0.565685)−1.37349​e−0.78125​(y+0.64)2)⩽1.37713erfc0.883883𝑦0.5656851.37349superscript𝑒0.78125superscript𝑦0.642absent\displaystyle\left(1.37713\operatorname{erfc}(0.883883y+0.565685)-1.37349e^{-0.78125(y+0.64)^{2}}\right)\leqslant |  |
|  |  |  |
| --- | --- | --- |
|  | (1.37713erfc(0.883883⋅−0.01+0.565685)−1.37349e−0.78125​(0.01+0.64)2)=\displaystyle\left(1.37713\operatorname{erfc}(0.883883\cdot-0.01+0.565685)-1.37349e^{-0.78125(0.01+0.64)^{2}}\right)= |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 0.5935272325870631−0.987354705867739<0.0.59352723258706310.9873547058677390\displaystyle 0.5935272325870631-0.987354705867739<0. |  | (334) |

Therefore, the values y=0.64𝑦0.64y=0.64 and x=−0.01𝑥0.01x=-0.01 give
a global maximum of the function f​(x,y)𝑓𝑥𝑦f(x,y) in the domain −0.01⩽y⩽0.010.01𝑦0.01-0.01\leqslant y\leqslant 0.01 and 0.64⩽x⩽1.8750.64𝑥1.8750.64\leqslant x\leqslant 1.875 and
the values y=1.875𝑦1.875y=1.875 and x=0.01𝑥0.01x=0.01 give the global minimum.
∎

## A4 Additional information on experiments

In this section, we report the hyperparameters that were considered for each method and
data set and give details on the processing of the data sets.

### A4.1 121 UCI Machine Learning Repository data sets: Hyperparameters

For the UCI data sets, the best hyperparameter setting was determined by a grid-search over all
hyperparameter combinations using 15% of the training data as validation set.
The early stopping parameter was determined on the smoothed learning curves of 100 epochs
of the validation set. Smoothing was done using moving averages of 10 consecutive
values. We tested “rectangular” and “conic” layers – rectangular layers have
constant number of hidden units in each layer, conic layers start with the given
number of hidden units in the first layer and then decrease the number of hidden units
to the size of the output layer according to the geometric progession.
If multiple hyperparameters provided identical performance on the validation
set, we preferred settings with a higher number of layers, lower learning rates and higher dropout rates.
All methods had the chance to adjust their hyperparameters to the data set at hand.

Table A4: Hyperparameters considered for self-normalizing networks in the UCI data sets.

| Hyperparameter | Considered values |
| --- | --- |
| Number of hidden units | {1024, 512, 256} |
| Number of hidden layers | {2, 3, 4, 8, 16, 32} |
| Learning rate | {0.01, 0.1, 1} |
| Dropout rate | {0.05, 0} |
| Layer form | {rectangular, conic} |




Table A5: Hyperparameters considered for ReLU networks with MS initialization in the UCI data sets.

| Hyperparameter | Considered values |
| --- | --- |
| Number of hidden units | {1024, 512, 256} |
| Number of hidden layers | {2,3,4,8,16,32} |
| Learning rate | {0.01, 0.1, 1} |
| Dropout rate | {0.5, 0} |
| Layer form | {rectangular, conic} |




Table A6: Hyperparameters considered for batch normalized networks in the UCI data sets.

| Hyperparameter | Considered values |
| --- | --- |
| Number of hidden units | {1024, 512, 256} |
| Number of hidden layers | {2, 3, 4, 8, 16, 32} |
| Learning rate | {0.01, 0.1, 1} |
| Normalization | {Batchnorm} |
| Layer form | {rectangular, conic} |




Table A7: Hyperparameters considered for weight normalized networks in the UCI data sets.

| Hyperparameter | Considered values |
| --- | --- |
| Number of hidden units | {1024, 512, 256} |
| Number of hidden layers | {2, 3, 4, 8, 16, 32} |
| Learning rate | {0.01, 0.1, 1} |
| Normalization | {Weightnorm} |
| Layer form | {rectangular, conic} |




Table A8: Hyperparameters considered for layer normalized networks in the UCI data sets.

| Hyperparameter | Considered values |
| --- | --- |
| Number of hidden units | {1024, 512, 256} |
| Number of hidden layers | {2, 3, 4, 8, 16, 32} |
| Learning rate | {0.01, 0.1, 1} |
| Normalization | {Layernorm} |
| Layer form | {rectangular, conic} |




Table A9: Hyperparameters considered for Highway networks in the UCI data sets.

| Hyperparameter | Considered values |
| --- | --- |
| Number of hidden layers | {2, 3, 4, 8, 16, 32} |
| Learning rate | {0.01, 0.1, 1} |
| Dropout rate | {0, 0.5} |




Table A10: Hyperparameters considered for Residual networks in the UCI data sets.

| Hyperparameter | Considered values |
| --- | --- |
| Number of blocks | {2, 3, 4, 8, 16} |
| Number of neurons per blocks | {1024, 512, 256} |
| Block form | {rectangular, diavolo} |
| Bottleneck | {25%, 50%} |
| Learning rate | {0.01, 0.1, 1} |

### A4.2 121 UCI Machine Learning Repository data sets: detailed results

##### Methods compared.

We used data sets and preprocessing scripts by Fernández-Delgado et al., [[10](#bib.bib10)] for data preparation and
defining training and test sets. With several flaws in the method comparison[[37](#bib.bib37)] that we avoided,
the authors compared 179 machine learning methods of 17 groups in their experiments.
The method groups were defined by Fernández-Delgado et al., [[10](#bib.bib10)] as follows:
Support Vector Machines, RandomForest, Multivariate adaptive regression splines (MARS),
Boosting, Rule-based, logistic and multinomial regression,
Discriminant Analysis (DA), Bagging,
Nearest Neighbour, DecisionTree, other Ensembles, Neural Networks, Bayesian, Other Methods,
generalized linear models (GLM), Partial least squares and principal component regression (PLSR), and Stacking.
However, many of methods assigned to those groups were merely different implementations of the
same method. Therefore, we selected one representative of each of the 17 groups for method
comparison. The representative method was chosen as the group’s method with the median performance
across all tasks. Finally, we included 17 other machine learning methods of Fernández-Delgado et al., [[10](#bib.bib10)],
and 6 FNNs, BatchNorm, WeightNorm, LayerNorm,
Highway, Residual and MSRAinit networks, and self-normalizing neural networks (SNNs) giving a total of 24 compared methods.

##### Results of FNN methods for all 121 data sets.

The results of the compared FNN methods can be found in Table [A11](#S4.T11 "Table A11 ‣ Results of FNN methods for all 121 data sets. ‣ A4.2 121 UCI Machine Learning Repository data sets: detailed results ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks").

Table A11: Comparison of FNN methods on all 121 UCI data sets.. The table reports the accuracy of FNN methods at each individual
task of the 121 UCI data sets. The first column gives the name of the data set, the second the number
of training data points N𝑁N, the third the number of features M𝑀M and the consecutive columns the accuracy values of
self-normalizing networks (SNNs), ReLU networks without normalization and with MSRA initialization (MS),
Highway networks (HW), Residual Networks (ResNet), networks with batch normalization (BN), weight
normalization (WN), and layer normalization (LN).

| dataset | N𝑁N | M𝑀M | SNN | MS | HW | ResNet | BN | WN | LN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abalone | 4177 | 9 | 0.6657 | 0.6284 | 0.6427 | 0.6466 | 0.6303 | 0.6351 | 0.6178 |
| acute-inflammation | 120 | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9000 |
| acute-nephritis | 120 | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| adult | 48842 | 15 | 0.8476 | 0.8487 | 0.8453 | 0.8484 | 0.8499 | 0.8453 | 0.8517 |
| annealing | 898 | 32 | 0.7600 | 0.7300 | 0.3600 | 0.2600 | 0.1200 | 0.6500 | 0.5000 |
| arrhythmia | 452 | 263 | 0.6549 | 0.6372 | 0.6283 | 0.6460 | 0.5929 | 0.6018 | 0.5752 |
| audiology-std | 196 | 60 | 0.8000 | 0.6800 | 0.7200 | 0.8000 | 0.6400 | 0.7200 | 0.8000 |
| balance-scale | 625 | 5 | 0.9231 | 0.9231 | 0.9103 | 0.9167 | 0.9231 | 0.9551 | 0.9872 |
| balloons | 16 | 5 | 1.0000 | 0.5000 | 0.2500 | 1.0000 | 1.0000 | 0.0000 | 0.7500 |
| bank | 4521 | 17 | 0.8903 | 0.8876 | 0.8885 | 0.8796 | 0.8823 | 0.8850 | 0.8920 |
| blood | 748 | 5 | 0.7701 | 0.7754 | 0.7968 | 0.8021 | 0.7647 | 0.7594 | 0.7112 |
| breast-cancer | 286 | 10 | 0.7183 | 0.6901 | 0.7465 | 0.7465 | 0.7324 | 0.6197 | 0.6620 |
| breast-cancer-wisc | 699 | 10 | 0.9714 | 0.9714 | 0.9771 | 0.9714 | 0.9829 | 0.9657 | 0.9714 |
| breast-cancer-wisc-diag | 569 | 31 | 0.9789 | 0.9718 | 0.9789 | 0.9507 | 0.9789 | 0.9718 | 0.9648 |
| breast-cancer-wisc-prog | 198 | 34 | 0.6735 | 0.7347 | 0.8367 | 0.8163 | 0.7755 | 0.8367 | 0.7959 |
| breast-tissue | 106 | 10 | 0.7308 | 0.4615 | 0.6154 | 0.4231 | 0.4615 | 0.5385 | 0.5769 |
| car | 1728 | 7 | 0.9838 | 0.9861 | 0.9560 | 0.9282 | 0.9606 | 0.9769 | 0.9907 |
| cardiotocography-10clases | 2126 | 22 | 0.8399 | 0.8418 | 0.8456 | 0.8173 | 0.7910 | 0.8606 | 0.8362 |
| cardiotocography-3clases | 2126 | 22 | 0.9153 | 0.8964 | 0.9171 | 0.9021 | 0.9096 | 0.8945 | 0.9021 |
| chess-krvk | 28056 | 7 | 0.8805 | 0.8606 | 0.5255 | 0.8543 | 0.8781 | 0.7673 | 0.8938 |
| chess-krvkp | 3196 | 37 | 0.9912 | 0.9900 | 0.9900 | 0.9912 | 0.9862 | 0.9912 | 0.9875 |
| congressional-voting | 435 | 17 | 0.6147 | 0.6055 | 0.5872 | 0.5963 | 0.5872 | 0.5872 | 0.5780 |
| conn-bench-sonar-mines-rocks | 208 | 61 | 0.7885 | 0.8269 | 0.8462 | 0.8077 | 0.7115 | 0.8269 | 0.6731 |
| conn-bench-vowel-deterding | 990 | 12 | 0.9957 | 0.9935 | 0.9784 | 0.9935 | 0.9610 | 0.9524 | 0.9935 |
| connect-4 | 67557 | 43 | 0.8807 | 0.8831 | 0.8599 | 0.8716 | 0.8729 | 0.8833 | 0.8856 |
| contrac | 1473 | 10 | 0.5190 | 0.5136 | 0.5054 | 0.5136 | 0.4538 | 0.4755 | 0.4592 |
| credit-approval | 690 | 16 | 0.8430 | 0.8430 | 0.8547 | 0.8430 | 0.8721 | 0.9070 | 0.8547 |
| cylinder-bands | 512 | 36 | 0.7266 | 0.7656 | 0.7969 | 0.7734 | 0.7500 | 0.7578 | 0.7578 |
| dermatology | 366 | 35 | 0.9231 | 0.9121 | 0.9780 | 0.9231 | 0.9341 | 0.9451 | 0.9451 |
| echocardiogram | 131 | 11 | 0.8182 | 0.8485 | 0.6061 | 0.8485 | 0.8485 | 0.7879 | 0.8182 |
| ecoli | 336 | 8 | 0.8929 | 0.8333 | 0.8690 | 0.8214 | 0.8214 | 0.8452 | 0.8571 |
| energy-y1 | 768 | 9 | 0.9583 | 0.9583 | 0.8802 | 0.8177 | 0.8646 | 0.9010 | 0.9479 |
| energy-y2 | 768 | 9 | 0.9063 | 0.8958 | 0.9010 | 0.8750 | 0.8750 | 0.8906 | 0.8802 |
| fertility | 100 | 10 | 0.9200 | 0.8800 | 0.8800 | 0.8400 | 0.6800 | 0.6800 | 0.8800 |
| flags | 194 | 29 | 0.4583 | 0.4583 | 0.4375 | 0.3750 | 0.4167 | 0.4167 | 0.3542 |
| glass | 214 | 10 | 0.7358 | 0.6038 | 0.6415 | 0.6415 | 0.5849 | 0.6792 | 0.6981 |
| haberman-survival | 306 | 4 | 0.7368 | 0.7237 | 0.6447 | 0.6842 | 0.7368 | 0.7500 | 0.6842 |
| hayes-roth | 160 | 4 | 0.6786 | 0.4643 | 0.7857 | 0.7143 | 0.7500 | 0.5714 | 0.8929 |
| heart-cleveland | 303 | 14 | 0.6184 | 0.6053 | 0.6316 | 0.5658 | 0.5789 | 0.5658 | 0.5789 |
| heart-hungarian | 294 | 13 | 0.7945 | 0.8356 | 0.7945 | 0.8082 | 0.8493 | 0.7534 | 0.8493 |
| heart-switzerland | 123 | 13 | 0.3548 | 0.3871 | 0.5806 | 0.3226 | 0.3871 | 0.2581 | 0.5161 |
| heart-va | 200 | 13 | 0.3600 | 0.2600 | 0.4000 | 0.2600 | 0.2800 | 0.2200 | 0.2400 |
| hepatitis | 155 | 20 | 0.7692 | 0.7692 | 0.6667 | 0.7692 | 0.8718 | 0.8462 | 0.7436 |
| hill-valley | 1212 | 101 | 0.5248 | 0.5116 | 0.5000 | 0.5396 | 0.5050 | 0.4934 | 0.5050 |
| horse-colic | 368 | 26 | 0.8088 | 0.8529 | 0.7794 | 0.8088 | 0.8529 | 0.7059 | 0.7941 |
| ilpd-indian-liver | 583 | 10 | 0.6986 | 0.6644 | 0.6781 | 0.6712 | 0.5959 | 0.6918 | 0.6986 |



|  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image-segmentation | 2310 | 19 | 0.9114 | 0.9090 | 0.9024 | 0.8919 | 0.8481 | 0.8938 | 0.8838 |
| ionosphere | 351 | 34 | 0.8864 | 0.9091 | 0.9432 | 0.9545 | 0.9432 | 0.9318 | 0.9432 |
| iris | 150 | 5 | 0.9730 | 0.9189 | 0.8378 | 0.9730 | 0.9189 | 1.0000 | 0.9730 |
| led-display | 1000 | 8 | 0.7640 | 0.7200 | 0.7040 | 0.7160 | 0.6280 | 0.6920 | 0.6480 |
| lenses | 24 | 5 | 0.6667 | 1.0000 | 1.0000 | 0.6667 | 0.8333 | 0.8333 | 0.6667 |
| letter | 20000 | 17 | 0.9726 | 0.9712 | 0.8984 | 0.9762 | 0.9796 | 0.9580 | 0.9742 |
| libras | 360 | 91 | 0.7889 | 0.8667 | 0.8222 | 0.7111 | 0.7444 | 0.8000 | 0.8333 |
| low-res-spect | 531 | 101 | 0.8571 | 0.8496 | 0.9023 | 0.8647 | 0.8571 | 0.8872 | 0.8947 |
| lung-cancer | 32 | 57 | 0.6250 | 0.3750 | 0.1250 | 0.2500 | 0.5000 | 0.5000 | 0.2500 |
| lymphography | 148 | 19 | 0.9189 | 0.7297 | 0.7297 | 0.6757 | 0.7568 | 0.7568 | 0.7838 |
| magic | 19020 | 11 | 0.8692 | 0.8629 | 0.8673 | 0.8723 | 0.8713 | 0.8690 | 0.8620 |
| mammographic | 961 | 6 | 0.8250 | 0.8083 | 0.7917 | 0.7833 | 0.8167 | 0.8292 | 0.8208 |
| miniboone | 130064 | 51 | 0.9307 | 0.9250 | 0.9270 | 0.9254 | 0.9262 | 0.9272 | 0.9313 |
| molec-biol-promoter | 106 | 58 | 0.8462 | 0.7692 | 0.6923 | 0.7692 | 0.7692 | 0.6923 | 0.4615 |
| molec-biol-splice | 3190 | 61 | 0.9009 | 0.8482 | 0.8833 | 0.8557 | 0.8519 | 0.8494 | 0.8607 |
| monks-1 | 556 | 7 | 0.7523 | 0.6551 | 0.5833 | 0.7546 | 0.9074 | 0.5000 | 0.7014 |
| monks-2 | 601 | 7 | 0.5926 | 0.6343 | 0.6389 | 0.6273 | 0.3287 | 0.6644 | 0.5162 |
| monks-3 | 554 | 7 | 0.6042 | 0.7454 | 0.5880 | 0.5833 | 0.5278 | 0.5231 | 0.6991 |
| mushroom | 8124 | 22 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9990 | 0.9995 | 0.9995 |
| musk-1 | 476 | 167 | 0.8739 | 0.8655 | 0.8992 | 0.8739 | 0.8235 | 0.8992 | 0.8992 |
| musk-2 | 6598 | 167 | 0.9891 | 0.9945 | 0.9915 | 0.9964 | 0.9982 | 0.9927 | 0.9951 |
| nursery | 12960 | 9 | 0.9978 | 0.9988 | 1.0000 | 0.9994 | 0.9994 | 0.9966 | 0.9966 |
| oocytes\_merluccius\_nucleus\_4d | 1022 | 42 | 0.8235 | 0.8196 | 0.7176 | 0.8000 | 0.8078 | 0.8078 | 0.7686 |
| oocytes\_merluccius\_states\_2f | 1022 | 26 | 0.9529 | 0.9490 | 0.9490 | 0.9373 | 0.9333 | 0.9020 | 0.9412 |
| oocytes\_trisopterus\_nucleus\_2f | 912 | 26 | 0.7982 | 0.8728 | 0.8289 | 0.7719 | 0.7456 | 0.7939 | 0.8202 |
| oocytes\_trisopterus\_states\_5b | 912 | 33 | 0.9342 | 0.9430 | 0.9342 | 0.8947 | 0.8947 | 0.9254 | 0.8991 |
| optical | 5620 | 63 | 0.9711 | 0.9666 | 0.9644 | 0.9627 | 0.9716 | 0.9638 | 0.9755 |
| ozone | 2536 | 73 | 0.9700 | 0.9732 | 0.9716 | 0.9669 | 0.9669 | 0.9748 | 0.9716 |
| page-blocks | 5473 | 11 | 0.9583 | 0.9708 | 0.9656 | 0.9605 | 0.9613 | 0.9730 | 0.9708 |
| parkinsons | 195 | 23 | 0.8980 | 0.9184 | 0.8367 | 0.9184 | 0.8571 | 0.8163 | 0.8571 |
| pendigits | 10992 | 17 | 0.9706 | 0.9714 | 0.9671 | 0.9708 | 0.9734 | 0.9620 | 0.9657 |
| pima | 768 | 9 | 0.7552 | 0.7656 | 0.7188 | 0.7135 | 0.7188 | 0.6979 | 0.6927 |
| pittsburg-bridges-MATERIAL | 106 | 8 | 0.8846 | 0.8462 | 0.9231 | 0.9231 | 0.8846 | 0.8077 | 0.9231 |
| pittsburg-bridges-REL-L | 103 | 8 | 0.6923 | 0.7692 | 0.6923 | 0.8462 | 0.7692 | 0.6538 | 0.7308 |
| pittsburg-bridges-SPAN | 92 | 8 | 0.6957 | 0.5217 | 0.5652 | 0.5652 | 0.5652 | 0.6522 | 0.6087 |
| pittsburg-bridges-T-OR-D | 102 | 8 | 0.8400 | 0.8800 | 0.8800 | 0.8800 | 0.8800 | 0.8800 | 0.8800 |
| pittsburg-bridges-TYPE | 105 | 8 | 0.6538 | 0.6538 | 0.5385 | 0.6538 | 0.1154 | 0.4615 | 0.6538 |
| planning | 182 | 13 | 0.6889 | 0.6667 | 0.6000 | 0.7111 | 0.6222 | 0.6444 | 0.6889 |
| plant-margin | 1600 | 65 | 0.8125 | 0.8125 | 0.8375 | 0.7975 | 0.7600 | 0.8175 | 0.8425 |
| plant-shape | 1600 | 65 | 0.7275 | 0.6350 | 0.6325 | 0.5150 | 0.2850 | 0.6575 | 0.6775 |
| plant-texture | 1599 | 65 | 0.8125 | 0.7900 | 0.7900 | 0.8000 | 0.8200 | 0.8175 | 0.8350 |
| post-operative | 90 | 9 | 0.7273 | 0.7273 | 0.5909 | 0.7273 | 0.5909 | 0.5455 | 0.7727 |
| primary-tumor | 330 | 18 | 0.5244 | 0.5000 | 0.4512 | 0.3902 | 0.5122 | 0.5000 | 0.4512 |
| ringnorm | 7400 | 21 | 0.9751 | 0.9843 | 0.9692 | 0.9811 | 0.9843 | 0.9719 | 0.9827 |
| seeds | 210 | 8 | 0.8846 | 0.8654 | 0.9423 | 0.8654 | 0.8654 | 0.8846 | 0.8846 |
| semeion | 1593 | 257 | 0.9196 | 0.9296 | 0.9447 | 0.9146 | 0.9372 | 0.9322 | 0.9447 |
| soybean | 683 | 36 | 0.8511 | 0.8723 | 0.8617 | 0.8670 | 0.8883 | 0.8537 | 0.8484 |
| spambase | 4601 | 58 | 0.9409 | 0.9461 | 0.9435 | 0.9461 | 0.9426 | 0.9504 | 0.9513 |
| spect | 265 | 23 | 0.6398 | 0.6183 | 0.6022 | 0.6667 | 0.6344 | 0.6398 | 0.6720 |
| spectf | 267 | 45 | 0.4973 | 0.6043 | 0.8930 | 0.7005 | 0.2299 | 0.4545 | 0.5561 |
| statlog-australian-credit | 690 | 15 | 0.5988 | 0.6802 | 0.6802 | 0.6395 | 0.6802 | 0.6860 | 0.6279 |
| statlog-german-credit | 1000 | 25 | 0.7560 | 0.7280 | 0.7760 | 0.7720 | 0.7520 | 0.7400 | 0.7400 |



|  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| statlog-heart | 270 | 14 | 0.9254 | 0.8358 | 0.7761 | 0.8657 | 0.7910 | 0.8657 | 0.7910 |
| statlog-image | 2310 | 19 | 0.9549 | 0.9757 | 0.9584 | 0.9584 | 0.9671 | 0.9515 | 0.9757 |
| statlog-landsat | 6435 | 37 | 0.9100 | 0.9075 | 0.9110 | 0.9055 | 0.9040 | 0.8925 | 0.9040 |
| statlog-shuttle | 58000 | 10 | 0.9990 | 0.9983 | 0.9977 | 0.9992 | 0.9988 | 0.9988 | 0.9987 |
| statlog-vehicle | 846 | 19 | 0.8009 | 0.8294 | 0.7962 | 0.7583 | 0.7583 | 0.8009 | 0.7915 |
| steel-plates | 1941 | 28 | 0.7835 | 0.7567 | 0.7608 | 0.7629 | 0.7031 | 0.7856 | 0.7588 |
| synthetic-control | 600 | 61 | 0.9867 | 0.9800 | 0.9867 | 0.9600 | 0.9733 | 0.9867 | 0.9733 |
| teaching | 151 | 6 | 0.5000 | 0.6053 | 0.5263 | 0.5526 | 0.5000 | 0.3158 | 0.6316 |
| thyroid | 7200 | 22 | 0.9816 | 0.9770 | 0.9708 | 0.9799 | 0.9778 | 0.9807 | 0.9752 |
| tic-tac-toe | 958 | 10 | 0.9665 | 0.9833 | 0.9749 | 0.9623 | 0.9833 | 0.9707 | 0.9791 |
| titanic | 2201 | 4 | 0.7836 | 0.7909 | 0.7927 | 0.7727 | 0.7800 | 0.7818 | 0.7891 |
| trains | 10 | 30 | NA | NA | NA | NA | 0.5000 | 0.5000 | 1.0000 |
| twonorm | 7400 | 21 | 0.9805 | 0.9778 | 0.9708 | 0.9735 | 0.9757 | 0.9730 | 0.9724 |
| vertebral-column-2clases | 310 | 7 | 0.8312 | 0.8701 | 0.8571 | 0.8312 | 0.8312 | 0.6623 | 0.8442 |
| vertebral-column-3clases | 310 | 7 | 0.8312 | 0.8052 | 0.7922 | 0.7532 | 0.7792 | 0.7403 | 0.8312 |
| wall-following | 5456 | 25 | 0.9098 | 0.9076 | 0.9230 | 0.9223 | 0.9333 | 0.9274 | 0.9128 |
| waveform | 5000 | 22 | 0.8480 | 0.8312 | 0.8320 | 0.8360 | 0.8360 | 0.8376 | 0.8448 |
| waveform-noise | 5000 | 41 | 0.8608 | 0.8328 | 0.8696 | 0.8584 | 0.8480 | 0.8640 | 0.8504 |
| wine | 178 | 14 | 0.9773 | 0.9318 | 0.9091 | 0.9773 | 0.9773 | 0.9773 | 0.9773 |
| wine-quality-red | 1599 | 12 | 0.6300 | 0.6250 | 0.5625 | 0.6150 | 0.5450 | 0.5575 | 0.6100 |
| wine-quality-white | 4898 | 12 | 0.6373 | 0.6479 | 0.5564 | 0.6307 | 0.5335 | 0.5482 | 0.6544 |
| yeast | 1484 | 9 | 0.6307 | 0.6173 | 0.6065 | 0.5499 | 0.4906 | 0.5876 | 0.6092 |
| zoo | 101 | 17 | 0.9200 | 1.0000 | 0.8800 | 1.0000 | 0.7200 | 0.9600 | 0.9600 |

##### Small and large data sets.

We assigned each of the 121 UCI data sets into the group “large datasets” or
“small datasets” if the had more than 1,000 data points or less, respectively.
We expected that Deep Learning methods require large data sets to competitive to other machine learning methods.
This resulted in 75 small and 46 large data sets.

##### Results.

The results of the method comparison are given in Tables [A12](#S4.T12 "Table A12 ‣ Results. ‣ A4.2 121 UCI Machine Learning Repository data sets: detailed results ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks") and [A13](#S4.T13 "Table A13 ‣ Results. ‣ A4.2 121 UCI Machine Learning Repository data sets: detailed results ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks") for
small and large data sets, respectively. On small data sets, SVMs performed best followed
by RandomForest and SNNs. On large data sets, SNNs are the best method followed by SVMs and
Random Forest.

Table A12: UCI comparison reporting the average rank
of a method on 75 classification task of the
UCI machine learning repository with less than 1000 data points.
For each dataset, the 24 compared methods,
were ranked by their
accuracy and the ranks were averaged across the tasks.
The first column gives the method group, the second the
method, the third
the average rank , and the last the p𝑝p-value
of a paired Wilcoxon test whether the difference to the best performing
method is significant.
SNNs are ranked third having been outperformed by Random Forests and SVMs.

| methodGroup | method | avg. rank | p𝑝p-value |
| --- | --- | --- | --- |
| SVM | LibSVM\_weka | 9.3 |  |
| RandomForest | RRFglobal\_caret | 9.6 | 2.5e-01 |
| SNN | SNN | 9.6 | 3.8e-01 |
| LMR | SimpleLogistic\_weka | 9.9 | 1.5e-01 |
| NeuralNetworks | lvq\_caret | 10.1 | 1.0e-01 |
| MARS | gcvEarth\_caret | 10.7 | 3.6e-02 |
| MSRAinit | MSRAinit | 11.0 | 4.0e-02 |
| LayerNorm | LayerNorm | 11.3 | 7.2e-02 |
| Highway | Highway | 11.5 | 8.9e-03 |
| DiscriminantAnalysis | mda\_R | 11.8 | 2.6e-03 |
| Boosting | LogitBoost\_weka | 11.9 | 2.4e-02 |
| Bagging | ctreeBag\_R | 12.1 | 1.8e-03 |
| ResNet | ResNet | 12.3 | 3.5e-03 |
| BatchNorm | BatchNorm | 12.6 | 4.9e-04 |
| Rule-based | JRip\_caret | 12.9 | 1.7e-04 |
| WeightNorm | WeightNorm | 13.0 | 8.3e-05 |
| DecisionTree | rpart2\_caret | 13.6 | 7.0e-04 |
| OtherEnsembles | Dagging\_weka | 13.9 | 3.0e-05 |
| Nearest Neighbour | NNge\_weka | 14.0 | 7.7e-04 |
| OtherMethods | pam\_caret | 14.2 | 1.5e-04 |
| PLSR | simpls\_R | 14.3 | 4.6e-05 |
| Bayesian | NaiveBayes\_weka | 14.6 | 1.2e-04 |
| GLM | bayesglm\_caret | 15.0 | 1.6e-06 |
| Stacking | Stacking\_weka | 20.9 | 2.2e-12 |




Table A13: UCI comparison reporting the average rank
of a method on 46 classification task of the
UCI machine learning repository with more than 1000 data points.
For each dataset, the 24 compared methods,
were ranked by their
accuracy and the ranks were averaged across the tasks.
The first column gives the method group, the second the
method, the third
the average rank , and the last the p𝑝p-value
of a paired Wilcoxon test whether the difference to the best performing
method is significant.
SNNs are ranked first having outperformed diverse machine learning methods and
other FNNs.

| methodGroup | method | avg. rank | p𝑝p-value |
| --- | --- | --- | --- |
| SNN | SNN | 5.8 |  |
| SVM | LibSVM\_weka | 6.1 | 5.8e-01 |
| RandomForest | RRFglobal\_caret | 6.6 | 2.1e-01 |
| MSRAinit | MSRAinit | 7.1 | 4.5e-03 |
| LayerNorm | LayerNorm | 7.2 | 7.1e-02 |
| Highway | Highway | 7.9 | 1.7e-03 |
| ResNet | ResNet | 8.4 | 1.7e-04 |
| WeightNorm | WeightNorm | 8.7 | 5.5e-04 |
| BatchNorm | BatchNorm | 9.7 | 1.8e-04 |
| MARS | gcvEarth\_caret | 9.9 | 8.2e-05 |
| Boosting | LogitBoost\_weka | 12.1 | 2.2e-07 |
| LMR | SimpleLogistic\_weka | 12.4 | 3.8e-09 |
| Rule-based | JRip\_caret | 12.4 | 9.0e-08 |
| Bagging | ctreeBag\_R | 13.5 | 1.6e-05 |
| DiscriminantAnalysis | mda\_R | 13.9 | 1.4e-10 |
| Nearest Neighbour | NNge\_weka | 14.1 | 1.6e-10 |
| DecisionTree | rpart2\_caret | 15.5 | 2.3e-08 |
| OtherEnsembles | Dagging\_weka | 16.1 | 4.4e-12 |
| NeuralNetworks | lvq\_caret | 16.3 | 1.6e-12 |
| Bayesian | NaiveBayes\_weka | 17.9 | 1.6e-12 |
| OtherMethods | pam\_caret | 18.3 | 2.8e-14 |
| GLM | bayesglm\_caret | 18.7 | 1.5e-11 |
| PLSR | simpls\_R | 19.0 | 3.4e-11 |
| Stacking | Stacking\_weka | 22.5 | 2.8e-14 |

### A4.3 Tox21 challenge data set: Hyperparameters

For the Tox21 data set, the best hyperparameter setting was determined by a grid-search over all
hyperparameter combinations using the validation set defined by the challenge winners [[28](#bib.bib28)].
The hyperparameter space was chosen to be similar to the hyperparameters that were tested by Mayr et al., [[28](#bib.bib28)].
The early stopping parameter was determined on the smoothed learning curves of 100 epochs
of the validation set. Smoothing was done using moving averages of 10 consecutive
values. We tested “rectangular” and “conic” layers – rectangular layers have
constant number of hidden units in each layer, conic layers start with the given
number of hidden units in the first layer and then decrease the number of hidden units
to the size of the output layer according to the geometric progession.
All methods had the chance to adjust their hyperparameters to the data set at hand.

Table A14: Hyperparameters considered for self-normalizing networks in the Tox21 data set.

| Hyperparameter | Considered values |
| --- | --- |
| Number of hidden units | {1024, 2048} |
| Number of hidden layers | {2,3,4,6,8,16,32} |
| Learning rate | {0.01, 0.05, 0.1} |
| Dropout rate | {0.05, 0.10} |
| Layer form | {rectangular, conic} |
| L2 regularization parameter | {0.001,0.0001,0.00001} |




Table A15: Hyperparameters considered for ReLU networks with MS initialization in the Tox21 data set.

| Hyperparameter | Considered values |
| --- | --- |
| Number of hidden units | {1024, 2048} |
| Number of hidden layers | {2,3,4,6,8,16,32} |
| Learning rate | {0.01, 0.05, 0.1} |
| Dropout rate | {0.5, 0} |
| Layer form | {rectangular, conic} |
| L2 regularization parameter | {0.001,0.0001,0.00001} |




Table A16: Hyperparameters considered for batch normalized networks in the Tox21 data set.

| Hyperparameter | Considered values |
| --- | --- |
| Number of hidden units | {1024, 2048} |
| Number of hidden layers | {2, 3, 4, 6, 8, 16, 32} |
| Learning rate | {0.01, 0.05, 0.1} |
| Normalization | {Batchnorm} |
| Layer form | {rectangular, conic} |
| L2 regularization parameter | {0.001,0.0001,0.00001} |




Table A17: Hyperparameters considered for weight normalized networks in the Tox21 data set.

| Hyperparameter | Considered values |
| --- | --- |
| Number of hidden units | {1024, 2048} |
| Number of hidden layers | {2, 3, 4, 6, 8, 16, 32} |
| Learning rate | {0.01, 0.05, 0.1} |
| Normalization | {Weightnorm} |
| Dropout rate | {0, 0.5} |
| Layer form | {rectangular, conic} |
| L2 regularization parameter | {0.001,0.0001,0.00001} |




Table A18: Hyperparameters considered for layer normalized networks in the Tox21 data set.

| Hyperparameter | Considered values |
| --- | --- |
| Number of hidden units | {1024, 2048} |
| Number of hidden layers | {2, 3, 4, 6, 8, 16, 32} |
| Learning rate | {0.01, 0.05, 0.1} |
| Normalization | {Layernorm} |
| Dropout rate | {0, 0.5} |
| Layer form | {rectangular, conic} |
| L2 regularization parameter | {0.001,0.0001,0.00001} |




Table A19: Hyperparameters considered for Highway networks in the Tox21 data set.

| Hyperparameter | Considered values |
| --- | --- |
| Number of hidden layers | {2, 3, 4, 6, 8, 16, 32} |
| Learning rate | {0.01, 0.05, 0.1} |
| Dropout rate | {0, 0.5} |
| L2 regularization parameter | {0.001,0.0001,0.00001} |




Table A20: Hyperparameters considered for Residual networks in the Tox21 data set.

| Hyperparameter | Considered values |
| --- | --- |
| Number of blocks | {2, 3, 4, 6, 8, 16} |
| Number of neurons per blocks | {1024, 2048} |
| Block form | {rectangular, diavolo} |
| Bottleneck | {25%, 50%} |
| Learning rate | {0.01, 0.05, 0.1} |
| L2 regularization parameter | {0.001,0.0001,0.00001} |

##### Distribution of network inputs.

We empirically checked the assumption that the distribution of network inputs can
well be approximated by a normal distribution. To this end, we investigated the
density of the network inputs before and during learning and found that
these density are close to normal distributions (see Figure [A8](#S4.F8 "Figure A8 ‣ Distribution of network inputs. ‣ A4.3 Tox21 challenge data set: Hyperparameters ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")).

![Refer to caption](/html/1706.02515/assets/x11.png)

![Refer to caption](/html/1706.02515/assets/x12.png)

Figure A8: Distribution of network inputs of an SNN for the Tox21 data set.
The plots show the distribution of network inputs z𝑧z of the second layer of a typical Tox21 network.
The red curves display a kernel density estimator of the network inputs and the black curve is the
density of a standard normal distribution.
Left panel: At initialization time before learning. The distribution of network inputs is close to a standard
normal distribution.
Right panel: After 40 epochs of learning. The distributions of network inputs is close to a normal distribution.

### A4.4 HTRU2 data set: Hyperparameters

For the HTRU2 data set, the best hyperparameter setting was determined by a grid-search over all
hyperparameter combinations using one of the 9 non-testing folds as validation fold in a nested
cross-validation procedure. Concretely,
if M𝑀M was the testing fold, we used M−1𝑀1M-1 as validation fold, and for M=1𝑀1M=1 we used fold 101010
for validation. The early stopping parameter was determined on the smoothed learning curves of 100 epochs
of the validation set. Smoothing was done using moving averages of 10 consecutive
values. We tested “rectangular” and “conic” layers – rectangular layers have
constant number of hidden units in each layer, conic layers start with the given
number of hidden units in the first layer and then decrease the number of hidden units
to the size of the output layer according to the geometric progession.
All methods had the chance to adjust their hyperparameters to the data set at hand.

Table A21: Hyperparameters considered for self-normalizing networks on the HTRU2 data set.

| Hyperparameter | Considered values |
| --- | --- |
| Number of hidden units | {256, 512, 1024} |
| Number of hidden layers | {2, 4, 8, 16, 32} |
| Learning rate | {0.1, 0.01, 1} |
| Dropout rate | { 0, 0.05} |
| Layer form | {rectangular, conic} |




Table A22: Hyperparameters considered for ReLU networks with Microsoft initialization on the HTRU2 data set.

| Hyperparameter | Considered values |
| --- | --- |
| Number of hidden units | {256, 512, 1024} |
| Number of hidden layers | {2, 4, 8, 16, 32} |
| Learning rate | {0.1, 0.01, 1} |
| Dropout rate | {0, 0.5} |
| Layer form | {rectangular, conic} |




Table A23: Hyperparameters considered for BatchNorm networks on the HTRU2 data set.

| Hyperparameter | Considered values |
| --- | --- |
| Number of hidden units | {256, 512, 1024} |
| Number of hidden layers | {2, 4, 8, 16, 32} |
| Learning rate | {0.1, 0.01, 1} |
| Normalization | {Batchnorm} |
| Layer form | {rectangular, conic} |




Table A24: Hyperparameters considered for WeightNorm networks on the HTRU2 data set.

| Hyperparameter | Considered values |
| --- | --- |
| Number of hidden units | {256, 512, 1024} |
| Number of hidden layers | {2, 4, 8, 16, 32} |
| Learning rate | {0.1, 0.01, 1} |
| Normalization | {Weightnorm} |
| Layer form | {rectangular, conic} |




Table A25: Hyperparameters considered for LayerNorm networks on the HTRU2 data set.

| Hyperparameter | Considered values |
| --- | --- |
| Number of hidden units | {256, 512, 1024} |
| Number of hidden layers | {2, 4, 8, 16, 32} |
| Learning rate | {0.1, 0.01, 1} |
| Normalization | {Layernorm} |
| Layer form | {rectangular, conic} |




Table A26: Hyperparameters considered for Highway networks on the HTRU2 data set.

| Hyperparameter | Considered values |
| --- | --- |
| Number of hidden layers | {2, 4, 8, 16, 32} |
| Learning rate | {0.1, 0.01, 1} |
| Dropout rate | {0, 0.5} |




Table A27: Hyperparameters considered for Residual networks on the HTRU2 data set.

| Hyperparameter | Considered values |
| --- | --- |
| Number of hidden units | {256, 512, 1024} |
| Number of residual blocks | {2, 3, 4, 8, 16} |
| Learning rate | {0.1, 0.01, 1} |
| Block form | {rectangular, diavolo} |
| Bottleneck | {0.25, 0.5} |

## A5 Other fixed points

A similar analysis with corresponding function domains can be performed for other fixed points, for example for μ=μ~=0𝜇~𝜇0\mu={\tilde{\mu}}=0 and ν=ν~=2𝜈~𝜈2\nu={\tilde{\nu}}=2, which leads
to a SELU activation function with parameters α02=1.97126subscript𝛼021.97126\alpha\_{\mathrm{02}}=1.97126 and λ02=1.06071subscript𝜆021.06071\lambda\_{\mathrm{02}}=1.06071.

## A6 Bounds determined by numerical methods

In this section we report bounds on previously discussed expressions as determined by numerical methods (min and max have been
computed).

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 0(μ=0.06,ω=0,ν=1.35,τ=1.12)subscript0formulae-sequence𝜇0.06formulae-sequence𝜔0formulae-sequence𝜈1.35𝜏1.12\displaystyle 0\_{(\mu=0.06,\omega=0,\nu=1.35,\tau=1.12)}\ | <∂𝒥11∂μ< .00182415(μ=−0.1,ω=0.1,ν=1.47845,τ=0.883374)absentsubscript𝒥11𝜇subscript.00182415formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈1.47845𝜏0.883374\displaystyle<\ \frac{\partial{\mathcal{J}}\_{11}}{\partial\mu}\ <\ .00182415\_{(\mu=-0.1,\omega=0.1,\nu=1.47845,\tau=0.883374)} |  | (335) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 0.905413(μ=0.1,ω=−0.1,ν=1.5,τ=1.25)subscript0.905413formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈1.5𝜏1.25\displaystyle 0.905413\_{(\mu=0.1,\omega=-0.1,\nu=1.5,\tau=1.25)}\ | <∂𝒥11∂ω< 1.04143(μ=0.1,ω=0.1,ν=0.8,τ=0.8)absentsubscript𝒥11𝜔subscript1.04143formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈0.8𝜏0.8\displaystyle<\ \frac{\partial{\mathcal{J}}\_{11}}{\partial\omega}\ <\ 1.04143\_{(\mu=0.1,\omega=0.1,\nu=0.8,\tau=0.8)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | −0.0151177(μ=−0.1,ω=0.1,ν=0.8,τ=1.25)subscript0.0151177formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈0.8𝜏1.25\displaystyle-0.0151177\_{(\mu=-0.1,\omega=0.1,\nu=0.8,\tau=1.25)}\ | <∂𝒥11∂ν< 0.0151177(μ=0.1,ω=−0.1,ν=0.8,τ=1.25)absentsubscript𝒥11𝜈subscript0.0151177formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈0.8𝜏1.25\displaystyle<\ \frac{\partial{\mathcal{J}}\_{11}}{\partial\nu}\ <\ 0.0151177\_{(\mu=0.1,\omega=-0.1,\nu=0.8,\tau=1.25)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | −0.015194(μ=−0.1,ω=0.1,ν=0.8,τ=1.25)subscript0.015194formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈0.8𝜏1.25\displaystyle-0.015194\_{(\mu=-0.1,\omega=0.1,\nu=0.8,\tau=1.25)}\ | <∂𝒥11∂τ< 0.015194(μ=0.1,ω=−0.1,ν=0.8,τ=1.25)absentsubscript𝒥11𝜏subscript0.015194formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈0.8𝜏1.25\displaystyle<\ \frac{\partial{\mathcal{J}}\_{11}}{\partial\tau}\ <\ 0.015194\_{(\mu=0.1,\omega=-0.1,\nu=0.8,\tau=1.25)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | −0.0151177(μ=−0.1,ω=0.1,ν=0.8,τ=1.25)subscript0.0151177formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈0.8𝜏1.25\displaystyle-0.0151177\_{(\mu=-0.1,\omega=0.1,\nu=0.8,\tau=1.25)}\ | <∂𝒥12∂μ< 0.0151177(μ=0.1,ω=−0.1,ν=0.8,τ=1.25)absentsubscript𝒥12𝜇subscript0.0151177formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈0.8𝜏1.25\displaystyle<\ \frac{\partial{\mathcal{J}}\_{12}}{\partial\mu}\ <\ 0.0151177\_{(\mu=0.1,\omega=-0.1,\nu=0.8,\tau=1.25)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | −0.0151177(μ=0.1,ω=−0.1,ν=0.8,τ=1.25)subscript0.0151177formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈0.8𝜏1.25\displaystyle-0.0151177\_{(\mu=0.1,\omega=-0.1,\nu=0.8,\tau=1.25)}\ | <∂𝒥12∂ω< 0.0151177(μ=0.1,ω=−0.1,ν=0.8,τ=1.25)absentsubscript𝒥12𝜔subscript0.0151177formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈0.8𝜏1.25\displaystyle<\ \frac{\partial{\mathcal{J}}\_{12}}{\partial\omega}\ <\ 0.0151177\_{(\mu=0.1,\omega=-0.1,\nu=0.8,\tau=1.25)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | −0.00785613(μ=0.1,ω=−0.1,ν=1.5,τ=1.25)subscript0.00785613formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈1.5𝜏1.25\displaystyle-0.00785613\_{(\mu=0.1,\omega=-0.1,\nu=1.5,\tau=1.25)}\ | <∂𝒥12∂ν< 0.0315805(μ=0.1,ω=0.1,ν=0.8,τ=0.8)absentsubscript𝒥12𝜈subscript0.0315805formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈0.8𝜏0.8\displaystyle<\ \frac{\partial{\mathcal{J}}\_{12}}{\partial\nu}\ <\ 0.0315805\_{(\mu=0.1,\omega=0.1,\nu=0.8,\tau=0.8)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 0.0799824(μ=0.1,ω=−0.1,ν=1.5,τ=1.25)subscript0.0799824formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈1.5𝜏1.25\displaystyle 0.0799824\_{(\mu=0.1,\omega=-0.1,\nu=1.5,\tau=1.25)}\ | <∂𝒥12∂τ< 0.110267(μ=−0.1,ω=0.1,ν=0.8,τ=0.8)absentsubscript𝒥12𝜏subscript0.110267formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈0.8𝜏0.8\displaystyle<\ \frac{\partial{\mathcal{J}}\_{12}}{\partial\tau}\ <\ 0.110267\_{(\mu=-0.1,\omega=0.1,\nu=0.8,\tau=0.8)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 0(μ=0.06,ω=0,ν=1.35,τ=1.12)subscript0formulae-sequence𝜇0.06formulae-sequence𝜔0formulae-sequence𝜈1.35𝜏1.12\displaystyle 0\_{(\mu=0.06,\omega=0,\nu=1.35,\tau=1.12)}\ | <∂𝒥21∂μ< 0.0174802(μ=0.1,ω=0.1,ν=0.8,τ=0.8)absentsubscript𝒥21𝜇subscript0.0174802formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈0.8𝜏0.8\displaystyle<\ \frac{\partial{\mathcal{J}}\_{21}}{\partial\mu}\ <\ 0.0174802\_{(\mu=0.1,\omega=0.1,\nu=0.8,\tau=0.8)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 0.0849308(μ=0.1,ω=−0.1,ν=0.8,τ=0.8)subscript0.0849308formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈0.8𝜏0.8\displaystyle 0.0849308\_{(\mu=0.1,\omega=-0.1,\nu=0.8,\tau=0.8)}\ | <∂𝒥21∂ω< 0.695766(μ=0.1,ω=0.1,ν=1.5,τ=1.25)absentsubscript𝒥21𝜔subscript0.695766formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈1.5𝜏1.25\displaystyle<\ \frac{\partial{\mathcal{J}}\_{21}}{\partial\omega}\ <\ 0.695766\_{(\mu=0.1,\omega=0.1,\nu=1.5,\tau=1.25)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | −0.0600823(μ=0.1,ω=−0.1,ν=0.8,τ=1.25)subscript0.0600823formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈0.8𝜏1.25\displaystyle-0.0600823\_{(\mu=0.1,\omega=-0.1,\nu=0.8,\tau=1.25)}\ | <∂𝒥21∂ν< 0.0600823(μ=−0.1,ω=0.1,ν=0.8,τ=1.25)absentsubscript𝒥21𝜈subscript0.0600823formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈0.8𝜏1.25\displaystyle<\ \frac{\partial{\mathcal{J}}\_{21}}{\partial\nu}\ <\ 0.0600823\_{(\mu=-0.1,\omega=0.1,\nu=0.8,\tau=1.25)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | −0.0673083(μ=0.1,ω=−0.1,ν=1.5,τ=0.8)subscript0.0673083formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈1.5𝜏0.8\displaystyle-0.0673083\_{(\mu=0.1,\omega=-0.1,\nu=1.5,\tau=0.8)}\ | <∂𝒥21∂τ< 0.0673083(μ=−0.1,ω=0.1,ν=1.5,τ=0.8)absentsubscript𝒥21𝜏subscript0.0673083formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈1.5𝜏0.8\displaystyle<\ \frac{\partial{\mathcal{J}}\_{21}}{\partial\tau}\ <\ 0.0673083\_{(\mu=-0.1,\omega=0.1,\nu=1.5,\tau=0.8)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | −0.0600823(μ=0.1,ω=−0.1,ν=0.8,τ=1.25)subscript0.0600823formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈0.8𝜏1.25\displaystyle-0.0600823\_{(\mu=0.1,\omega=-0.1,\nu=0.8,\tau=1.25)}\ | <∂𝒥22∂μ< 0.0600823(μ=−0.1,ω=0.1,ν=0.8,τ=1.25)absentsubscript𝒥22𝜇subscript0.0600823formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈0.8𝜏1.25\displaystyle<\ \frac{\partial{\mathcal{J}}\_{22}}{\partial\mu}\ <\ 0.0600823\_{(\mu=-0.1,\omega=0.1,\nu=0.8,\tau=1.25)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | −0.0600823(μ=0.1,ω=−0.1,ν=0.8,τ=1.25)subscript0.0600823formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈0.8𝜏1.25\displaystyle-0.0600823\_{(\mu=0.1,\omega=-0.1,\nu=0.8,\tau=1.25)}\ | <∂𝒥22∂ω< 0.0600823(μ=−0.1,ω=0.1,ν=0.8,τ=1.25)absentsubscript𝒥22𝜔subscript0.0600823formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈0.8𝜏1.25\displaystyle<\ \frac{\partial{\mathcal{J}}\_{22}}{\partial\omega}\ <\ 0.0600823\_{(\mu=-0.1,\omega=0.1,\nu=0.8,\tau=1.25)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | −0.276862(μ=−0.01,ω=−0.01,ν=0.8,τ=1.25)subscript0.276862formulae-sequence𝜇0.01formulae-sequence𝜔0.01formulae-sequence𝜈0.8𝜏1.25\displaystyle-0.276862\_{(\mu=-0.01,\omega=-0.01,\nu=0.8,\tau=1.25)}\ | <∂𝒥22∂ν<−0.084813(μ=−0.1,ω=0.1,ν=1.5,τ=0.8)absentsubscript𝒥22𝜈subscript0.084813formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈1.5𝜏0.8\displaystyle<\ \frac{\partial{\mathcal{J}}\_{22}}{\partial\nu}\ <\ -0.084813\_{(\mu=-0.1,\omega=0.1,\nu=1.5,\tau=0.8)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 0.562302(μ=0.1,ω=−0.1,ν=1.5,τ=1.25)subscript0.562302formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈1.5𝜏1.25\displaystyle 0.562302\_{(\mu=0.1,\omega=-0.1,\nu=1.5,\tau=1.25)}\ | <∂𝒥22∂τ< 0.664051(μ=0.1,ω=0.1,ν=0.8,τ=0.8)absentsubscript𝒥22𝜏subscript0.664051formulae-sequence𝜇0.1formulae-sequence𝜔0.1formulae-sequence𝜈0.8𝜏0.8\displaystyle<\ \frac{\partial{\mathcal{J}}\_{22}}{\partial\tau}\ <\ 0.664051\_{(\mu=0.1,\omega=0.1,\nu=0.8,\tau=0.8)} |  |

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | |∂𝒥11∂μ|subscript𝒥11𝜇\displaystyle\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\mu}\right|\ | < 0.00182415​(0.0031049101995398316)absent0.001824150.0031049101995398316\displaystyle<\ 0.00182415(0.0031049101995398316) |  | (336) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥11∂ω|subscript𝒥11𝜔\displaystyle\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\omega}\right|\ | < 1.04143​(1.055872374194189)absent1.041431.055872374194189\displaystyle<\ 1.04143(1.055872374194189) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥11∂ν|subscript𝒥11𝜈\displaystyle\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\nu}\right|\ | < 0.0151177​(0.031242911235461816)absent0.01511770.031242911235461816\displaystyle<\ 0.0151177(0.031242911235461816) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥11∂τ|subscript𝒥11𝜏\displaystyle\left|\frac{\partial{\mathcal{J}}\_{11}}{\partial\tau}\right|\ | < 0.015194​(0.03749149348255419)absent0.0151940.03749149348255419\displaystyle<\ 0.015194(0.03749149348255419) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥12∂μ|subscript𝒥12𝜇\displaystyle\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\mu}\right|\ | < 0.0151177​(0.031242911235461816)absent0.01511770.031242911235461816\displaystyle<\ 0.0151177(0.031242911235461816) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥12∂ω|subscript𝒥12𝜔\displaystyle\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\omega}\right|\ | < 0.0151177​(0.031242911235461816)absent0.01511770.031242911235461816\displaystyle<\ 0.0151177(0.031242911235461816) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥12∂ν|subscript𝒥12𝜈\displaystyle\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\nu}\right|\ | < 0.0315805​(0.21232788238624354)absent0.03158050.21232788238624354\displaystyle<\ 0.0315805(0.21232788238624354) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥12∂τ|subscript𝒥12𝜏\displaystyle\left|\frac{\partial{\mathcal{J}}\_{12}}{\partial\tau}\right|\ | < 0.110267​(0.2124377655377270)absent0.1102670.2124377655377270\displaystyle<\ 0.110267(0.2124377655377270) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥21∂μ|subscript𝒥21𝜇\displaystyle\left|\frac{\partial{\mathcal{J}}\_{21}}{\partial\mu}\right|\ | < 0.0174802​(0.02220441024325437)absent0.01748020.02220441024325437\displaystyle<\ 0.0174802(0.02220441024325437) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥21∂ω|subscript𝒥21𝜔\displaystyle\left|\frac{\partial{\mathcal{J}}\_{21}}{\partial\omega}\right|\ | < 0.695766​(1.146955401845684)absent0.6957661.146955401845684\displaystyle<\ 0.695766(1.146955401845684) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥21∂ν|subscript𝒥21𝜈\displaystyle\left|\frac{\partial{\mathcal{J}}\_{21}}{\partial\nu}\right|\ | < 0.0600823​(0.14983446469110305)absent0.06008230.14983446469110305\displaystyle<\ 0.0600823(0.14983446469110305) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥21∂τ|subscript𝒥21𝜏\displaystyle\left|\frac{\partial{\mathcal{J}}\_{21}}{\partial\tau}\right|\ | < 0.0673083​(0.17980135762932363)absent0.06730830.17980135762932363\displaystyle<\ 0.0673083(0.17980135762932363) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥22∂μ|subscript𝒥22𝜇\displaystyle\left|\frac{\partial{\mathcal{J}}\_{22}}{\partial\mu}\right|\ | < 0.0600823​(0.14983446469110305)absent0.06008230.14983446469110305\displaystyle<\ 0.0600823(0.14983446469110305) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥22∂ω|subscript𝒥22𝜔\displaystyle\left|\frac{\partial{\mathcal{J}}\_{22}}{\partial\omega}\right|\ | < 0.0600823​(0.14983446469110305)absent0.06008230.14983446469110305\displaystyle<\ 0.0600823(0.14983446469110305) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥22∂ν|subscript𝒥22𝜈\displaystyle\left|\frac{\partial{\mathcal{J}}\_{22}}{\partial\nu}\right|\ | < 0.562302​(1.805740052651535)absent0.5623021.805740052651535\displaystyle<\ 0.562302(1.805740052651535) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |∂𝒥22∂τ|subscript𝒥22𝜏\displaystyle\left|\frac{\partial{\mathcal{J}}\_{22}}{\partial\tau}\right|\ | < 0.664051​(2.396685907216327)absent0.6640512.396685907216327\displaystyle<\ 0.664051(2.396685907216327) |  |

## A7 References

## References

* Abramowitz and Stegun, [1964]

  Abramowitz, M. and Stegun, I. (1964).
  Handbook of Mathematical Functions, volume 55 of Applied
  Mathematics Series.
  National Bureau of Standards, 10th edition.
* Ba et al., [2016]

  Ba, J. L., Kiros, J. R., and Hinton, G. (2016).
  Layer normalization.
  arXiv preprint arXiv:1607.06450.
* Bengio, [2013]

  Bengio, Y. (2013).
  Deep learning of representations: Looking forward.
  In Proceedings of the First International Conference on
  Statistical Language and Speech Processing, pages 1–37, Berlin, Heidelberg.
* Blinn, [1996]

  Blinn, J. (1996).
  Consider the lowly 2×\times2 matrix.
  IEEE Computer Graphics and Applications, pages 82–88.
* Bradley, [1981]

  Bradley, R. C. (1981).
  Central limit theorems under weak dependence.
  Journal of Multivariate Analysis, 11(1):1–16.
* Cireşan and Meier, [2015]

  Cireşan, D. and Meier, U. (2015).
  Multi-column deep neural networks for offline handwritten chinese
  character classification.
  In 2015 International Joint Conference on Neural Networks
  (IJCNN), pages 1–6. IEEE.
* Clevert et al., [2015]

  Clevert, D.-A., Unterthiner, T., and Hochreiter, S. (2015).
  Fast and accurate deep network learning by exponential linear units
  (ELUs).
  5th International Conference on Learning Representations,
  arXiv:1511.07289.
* Dugan et al., [2016]

  Dugan, P., Clark, C., LeCun, Y., and Van Parijs, S. (2016).
  Phase 4: Dcl system using deep learning approaches for land-based or
  ship-based real-time recognition and localization of marine
  mammals-distributed processing and big data applications.
  arXiv preprint arXiv:1605.00982.
* Esteva et al., [2017]

  Esteva, A., Kuprel, B., Novoa, R., Ko, J., Swetter, S., Blau, H., and Thrun, S.
  (2017).
  Dermatologist-level classification of skin cancer with deep neural
  networks.
  Nature, 542(7639):115–118.
* Fernández-Delgado et al., [2014]

  Fernández-Delgado, M., Cernadas, E., Barro, S., and Amorim, D. (2014).
  Do we need hundreds of classifiers to solve real world classification
  problems.
  Journal of Machine Learning Research, 15(1):3133–3181.
* Goldberg, [1991]

  Goldberg, D. (1991).
  What every computer scientist should know about floating-point
  arithmetic.
  ACM Comput. Surv., 223(1):5–48.
* Graves et al., [2013]

  Graves, A., Mohamed, A., and Hinton, G. (2013).
  Speech recognition with deep recurrent neural networks.
  In IEEE International conference on acoustics, speech and
  signal processing (ICASSP), pages 6645–6649.
* Graves and Schmidhuber, [2009]

  Graves, A. and Schmidhuber, J. (2009).
  Offline handwriting recognition with multidimensional recurrent
  neural networks.
  In Advances in neural information processing systems, pages
  545–552.
* Gulshan et al., [2016]

  Gulshan, V., Peng, L., Coram, M., Stumpe, M. C., Wu, D., Narayanaswamy, A.,
  Venugopalan, S., Widner, K., Madams, T., Cuadros, J., et al. (2016).
  Development and validation of a deep learning algorithm for detection
  of diabetic retinopathy in retinal fundus photographs.
  JAMA, 316(22):2402–2410.
* Harrison, [1999]

  Harrison, J. (1999).
  A machine-checked theory of floating point arithmetic.
  In Bertot, Y., Dowek, G., Hirschowitz, A., Paulin, C., and Théry,
  L., editors, Theorem Proving in Higher Order Logics: 12th International
  Conference, TPHOLs’99, volume 1690 of Lecture Notes in Computer
  Science, pages 113–130. Springer-Verlag.
* [16]

  He, K., Zhang, X., Ren, S., and Sun, J. (2015a).
  Deep residual learning for image recognition.
  In IEEE Conference on Computer Vision and Pattern Recognition
  (CVPR).
* [17]

  He, K., Zhang, X., Ren, S., and Sun, J. (2015b).
  Delving deep into rectifiers: Surpassing human-level performance on
  imagenet classification.
  In Proceedings of the IEEE International Conference on
  Computer Vision (ICCV), pages 1026–1034.
* Hochreiter and Schmidhuber, [1997]

  Hochreiter, S. and Schmidhuber, J. (1997).
  Long short-term memory.
  Neural Computation, 9(8):1735–1780.
* Huval et al., [2015]

  Huval, B., Wang, T., Tandon, S., et al. (2015).
  An empirical evaluation of deep learning on highway driving.
  arXiv preprint arXiv:1504.01716.
* Ioffe and Szegedy, [2015]

  Ioffe, S. and Szegedy, C. (2015).
  Batch normalization: Accelerating deep network training by reducing
  internal covariate shift.
  In Proceedings of The 32nd International Conference on Machine
  Learning, pages 448–456.
* Kahan, [2004]

  Kahan, W. (2004).
  A logarithm too clever by half.
  Technical report, University of California, Berkeley.
* Korolev and Shevtsova, [2012]

  Korolev, V. and Shevtsova, I. (2012).
  An improvement of the Berry–Esseen inequality with applications to
  Poisson and mixed Poisson random sums.
  Scandinavian Actuarial Journal, 2012(2):81–105.
* Krizhevsky et al., [2012]

  Krizhevsky, A., Sutskever, I., and Hinton, G. (2012).
  Imagenet classification with deep convolutional neural networks.
  In Advances in Neural Information Processing Systems, pages
  1097–1105.
* LeCun and Bengio, [1995]

  LeCun, Y. and Bengio, Y. (1995).
  Convolutional networks for images, speech, and time series.
  The handbook of brain theory and neural networks,
  3361(10):1995.
* LeCun et al., [2015]

  LeCun, Y., Bengio, Y., and Hinton, G. (2015).
  Deep learning.
  Nature, 521(7553):436–444.
* Loosemore et al., [2016]

  Loosemore, S., Stallman, R. M., McGrath, R., Oram, A., and Drepper, U. (2016).
  The GNU C Library: Application Fundamentals.
  GNU Press, Free Software Foundation, 51 Franklin St, Fifth Floor,
  Boston, MA 02110-1301, USA, 2.24 edition.
* Lyon et al., [2016]

  Lyon, R., Stappers, B., Cooper, S., Brooke, J., and Knowles, J. (2016).
  Fifty years of pulsar candidate selection: From simple filters to a
  new principled real-time classification approach.
  Monthly Notices of the Royal Astronomical Society,
  459(1):1104–1123.
* Mayr et al., [2016]

  Mayr, A., Klambauer, G., Unterthiner, T., and Hochreiter, S. (2016).
  DeepTox: Toxicity prediction using deep learning.
  Frontiers in Environmental Science, 3:80.
* Muller, [2005]

  Muller, J.-M. (2005).
  On the definition of ulp(x)𝑥(x).
  Technical Report Research report RR2005-09, Laboratoire de
  l’Informatique du Parallélisme.
* Ren and MacKenzie, [2007]

  Ren, C. and MacKenzie, A. R. (2007).
  Closed-form approximations to the error and complementary error
  functions and their applications in atmospheric science.
  Atmos. Sci. Let., pages 70–73.
* Sak et al., [2015]

  Sak, H., Senior, A., Rao, K., and Beaufays, F. (2015).
  Fast and accurate recurrent neural network acoustic models for speech
  recognition.
  arXiv preprint arXiv:1507.06947.
* Salimans and Kingma, [2016]

  Salimans, T. and Kingma, D. P. (2016).
  Weight normalization: A simple reparameterization to accelerate
  training of deep neural networks.
  In Advances in Neural Information Processing Systems, pages
  901–909.
* Schmidhuber, [2015]

  Schmidhuber, J. (2015).
  Deep learning in neural networks: An overview.
  Neural Networks, 61:85–117.
* Silver et al., [2016]

  Silver, D., Huang, A., Maddison, C., et al. (2016).
  Mastering the game of Go with deep neural networks and tree search.
  Nature, 529(7587):484–489.
* Srivastava et al., [2015]

  Srivastava, R. K., Greff, K., and Schmidhuber, J. (2015).
  Training very deep networks.
  In Advances in Neural Information Processing Systems, pages
  2377–2385.
* Sutskever et al., [2014]

  Sutskever, I., Vinyals, O., and Le, Q. V. (2014).
  Sequence to sequence learning with neural networks.
  In Advances in Neural Information Processing Systems, pages
  3104–3112.
* Wainberg et al., [2016]

  Wainberg, M., Alipanahi, B., and Frey, B. J. (2016).
  Are random forests truly the best classifiers?
  Journal of Machine Learning Research, 17(110):1–5.

###### List of Figures

1. [1 FNN and SNN trainin error curves](#Sx2.F1 "Figure 1In Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")
2. [2 Visualization of the mapping g𝑔g](#Sx2.F2 "Figure 2In Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")
3. [A3 Graph of the main subfunction of the derivative of the second moment](#S3.F3 "Figure A3In A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
4. [A4 Graph of the Abramowitz bound for the complementary error function.](#S3.F4 "Figure A4In A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
5. [A5 Graphs of the functions ex2​erfc⁡(x)superscript𝑒superscript𝑥2erfc𝑥e^{x^{2}}\operatorname{erfc}(x) and x​ex2​erfc⁡(x)𝑥superscript𝑒superscript𝑥2erfc𝑥xe^{x^{2}}\operatorname{erfc}(x).](#S3.F5 "Figure A5In A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
6. [A6 The graph of function μ~~𝜇{\tilde{\mu}} for low variances](#S3.F6 "Figure A6In A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
7. [A7 Graph of the function h​(x)=μ~2​(0.1,−0.1,x,1,λ01,α01)ℎ𝑥superscript~𝜇20.10.1𝑥1subscript𝜆01subscript𝛼01h(x)={{\tilde{\mu}}}^{2}(0.1,-0.1,x,1,\lambda\_{\mathrm{0}1},\alpha\_{\mathrm{0}1})](#S3.F7 "Figure A7In A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
8. [A8 Distribution of network inputs in Tox21 SNNs.](#S4.F8 "Figure A8In A4.3 Tox21 challenge data set: Hyperparameters ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")


###### List of Tables

1. [1 Comparison of seven FNNs on 121 UCI tasks](#Sx3.T1 "Table 1In Experiments ‣ Self-Normalizing Neural Networks")
2. [2 Comparison of FNNs at the Tox21 challenge dataset](#Sx3.T2 "Table 2In Experiments ‣ Self-Normalizing Neural Networks")
3. [3 Comparison of FNNs and reference methods at HTRU2](#Sx3.T3 "Table 3In Experiments ‣ Self-Normalizing Neural Networks")
4. [A4 Hyperparameters considered for self-normalizing networks in the UCI data sets.](#S4.T4 "Table A4In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
5. [A5 Hyperparameters considered for ReLU networks in the UCI data sets.](#S4.T5 "Table A5In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
6. [A6 Hyperparameters considered for batch normalized networks in the UCI data sets.](#S4.T6 "Table A6In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
7. [A7 Hyperparameters considered for weight normalized networks in the UCI data sets.](#S4.T7 "Table A7In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
8. [A8 Hyperparameters considered for layer normalized networks in the UCI data sets.](#S4.T8 "Table A8In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
9. [A9 Hyperparameters considered for Highway networks in the UCI data sets.](#S4.T9 "Table A9In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
10. [A10 Hyperparameters considered for Residual networks in the UCI data sets.](#S4.T10 "Table A10In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
11. [A11 Comparison of FNN methods on all 121 UCI data sets.](#S4.T11 "Table A11In A4.2 121 UCI Machine Learning Repository data sets: detailed results ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
12. [A12 Method comparison on small UCI data sets](#S4.T12 "Table A12In A4.2 121 UCI Machine Learning Repository data sets: detailed results ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
13. [A13 Method comparison on large UCI data sets](#S4.T13 "Table A13In A4.2 121 UCI Machine Learning Repository data sets: detailed results ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
14. [A14 Hyperparameters considered for self-normalizing networks in the Tox21 data set.](#S4.T14 "Table A14In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
15. [A15 Hyperparameters considered for ReLU networks in the Tox21 data set.](#S4.T15 "Table A15In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
16. [A16 Hyperparameters considered for batch normalized networks in the Tox21 data set.](#S4.T16 "Table A16In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
17. [A17 Hyperparameters considered for weight normalized networks in the Tox21 data set.](#S4.T17 "Table A17In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
18. [A18 Hyperparameters considered for layer normalized networks in the Tox21 data set.](#S4.T18 "Table A18In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
19. [A19 Hyperparameters considered for Highway networks in the Tox21 data set.](#S4.T19 "Table A19In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
20. [A20 Hyperparameters considered for Residual networks in the Tox21 data set.](#S4.T20 "Table A20In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
21. [A21 Hyperparameters considered for self-normalizing networks on the HTRU2 data set.](#S4.T21 "Table A21In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
22. [A22 Hyperparameters considered for ReLU networks on the HTRU2 data set.](#S4.T22 "Table A22In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
23. [A23 Hyperparameters considered for BatchNorm networks on the HTRU2 data set.](#S4.T23 "Table A23In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
24. [A24 Hyperparameters considered for WeightNorm networks on the HTRU2 data set.](#S4.T24 "Table A24In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
25. [A25 Hyperparameters considered for LayerNorm networks on the HTRU2 data set.](#S4.T25 "Table A25In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
26. [A26 Hyperparameters considered for Highway networks on the HTRU2 data set.](#S4.T26 "Table A26In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
27. [A27 Hyperparameters considered for Residual networks on the HTRU2 data set.](#S4.T27 "Table A27In A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")

## Brief index

* Abramowitz bounds [§A3.4.6](#S3.SS4.SSS6.p5.1 "A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), [Lemma 22](#Thmtheorem22 "Lemma 22 (Erfc bound from Abramowitz). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
* Banach Fixed Point Theorem [Theorem 4](#Thmtheorem4 "Theorem 4 (Banach Fixed Point Theorem). ‣ A3.1 Proof of Theorem 1 ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
* bounds
  + derivatives of Jacobian entries [§A3.4.1](#S3.SS4.SSS1.Px3 "Bounds on the derivatives of the Jacobian entries. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
  + Jacobian entries [§A3.4.1](#S3.SS4.SSS1.Px4 "Bounds on the entries of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
  + mean and variance [§A3.4.1](#S3.SS4.SSS1.Px5 "Bounds on mean, variance and second moment. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
  + singular value [§A3.4.1](#S3.SS4.SSS1.Px6 "Upper Bounds on the Largest Singular Value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), [Lemma 11](#Thmtheorem11 "Lemma 11 (Mean Value Theorem Bound on Deviation from Largest Singular Value). ‣ Upper Bounds on the Largest Singular Value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
* central limit theorem [Applicability of the central limit theorem and independence assumption.](#Sx2.SS0.SSS0.Px8 "Applicability of the central limit theorem and independence assumption. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")
* complementary error function
  + bounds [§A3.4.6](#S3.SS4.SSS6.p5.1 "A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
  + definition [§A3.4.6](#S3.SS4.SSS6.p3.2 "A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
* computer-assisted proof [§A3.4.5](#S3.SS4.SSS5 "A3.4.5 Computer-assisted proof details for main Lemma 12 in Section A3.4.1. ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
* contracting variance [§A3.4.3](#S3.SS4.SSS3 "A3.4.3 Lemmata for proofing Theorem 2: The variance is contracting ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
* definitions [Normalization and SNNs.](#Sx2.SS0.SSS0.Px1 "Normalization and SNNs. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")
* domain
  + singular value [§A3.4.1](#S3.SS4.SSS1.p1.9 "A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
  + Theorem 1 [Theorem 1](#Thmtheorem1a.p1.16.16 "Theorem 1 (Stable and Attracting Fixed Points). ‣ A2.1 Theorem 1: Stable and Attracting Fixed Points Close to (0,1) ‣ A2 Theorems ‣ Self-Normalizing Neural Networks")
  + Theorem 2 [Theorem 2](#Thmtheorem2a.p1.8.8 "Theorem 2 (Decreasing 𝜈). ‣ A2.2 Theorem 2: Decreasing Variance from Above ‣ A2 Theorems ‣ Self-Normalizing Neural Networks")
  + Theorem 3 [Theorem 3](#Thmtheorem3a.p1.4.4 "Theorem 3 (Increasing 𝜈). ‣ A2.3 Theorem 3: Increasing Variance from Below ‣ A2 Theorems ‣ Self-Normalizing Neural Networks")
* dropout [New Dropout Technique.](#Sx2.SS0.SSS0.Px7 "New Dropout Technique. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")
* erf [§A3.4.6](#S3.SS4.SSS6.p3.2 "A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), [§A3.4.6](#S3.SS4.SSS6.p5.1 "A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
* erfc [§A3.4.6](#S3.SS4.SSS6.p3.2 "A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), [§A3.4.6](#S3.SS4.SSS6.p5.1 "A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
* error function
  + bounds [§A3.4.6](#S3.SS4.SSS6.p5.1 "A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
  + definition [§A3.4.6](#S3.SS4.SSS6.p3.2 "A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
  + properties [Lemma 24](#Thmtheorem24 "Lemma 24 (Properties of 𝑥⁢𝑒^𝑥²⁢erfc(𝑥)). ‣ A3.4.6 Intermediate Lemmata and Proofs ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
* expanding variance [§A3.4.4](#S3.SS4.SSS4 "A3.4.4 Lemmata for proofing Theorem 3: The variance is expanding ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
* experiments [§A4](#S4 "A4 Additional information on experiments ‣ Self-Normalizing Neural Networks"), [Experiments](#Sx3 "Experiments ‣ Self-Normalizing Neural Networks")
  + astronomy [Astronomy: Prediction of pulsars in the HTRU2 dataset.](#Sx3.SS0.SSS0.Px3 "Astronomy: Prediction of pulsars in the HTRU2 dataset. ‣ Experiments ‣ Self-Normalizing Neural Networks")
  + HTRU2 [§A4.4](#S4.SS4 "A4.4 HTRU2 data set: Hyperparameters ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks"), [Astronomy: Prediction of pulsars in the HTRU2 dataset.](#Sx3.SS0.SSS0.Px3 "Astronomy: Prediction of pulsars in the HTRU2 dataset. ‣ Experiments ‣ Self-Normalizing Neural Networks")
    - hyperparameters [§A4.4](#S4.SS4.p1.4 "A4.4 HTRU2 data set: Hyperparameters ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
  + methods compared [Experiments](#Sx3.p1.1 "Experiments ‣ Self-Normalizing Neural Networks")
  + Tox21 [§A4.3](#S4.SS3 "A4.3 Tox21 challenge data set: Hyperparameters ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks"), [Drug discovery: The Tox21 challenge dataset.](#Sx3.SS0.SSS0.Px2 "Drug discovery: The Tox21 challenge dataset. ‣ Experiments ‣ Self-Normalizing Neural Networks")
    - hyperparameters [§A4.3](#S4.SS3 "A4.3 Tox21 challenge data set: Hyperparameters ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks"), [Drug discovery: The Tox21 challenge dataset.](#Sx3.SS0.SSS0.Px2 "Drug discovery: The Tox21 challenge dataset. ‣ Experiments ‣ Self-Normalizing Neural Networks")
  + UCI [§A4.1](#S4.SS1 "A4.1 121 UCI Machine Learning Repository data sets: Hyperparameters ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks"), [121 UCI Machine Learning Repository datasets.](#Sx3.SS0.SSS0.Px1 "121 UCI Machine Learning Repository datasets. ‣ Experiments ‣ Self-Normalizing Neural Networks")
    - details [§A4.2](#S4.SS2 "A4.2 121 UCI Machine Learning Repository data sets: detailed results ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
    - hyperparameters [§A4.1](#S4.SS1 "A4.1 121 UCI Machine Learning Repository data sets: Hyperparameters ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
    - results [§A4.2](#S4.SS2.SSS0.Px4.p1.1 "Results. ‣ A4.2 121 UCI Machine Learning Repository data sets: detailed results ‣ A4 Additional information on experiments ‣ Self-Normalizing Neural Networks")
* initialization [Initialization.](#Sx2.SS0.SSS0.Px6 "Initialization. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")
* Jacobian [§A3.4.1](#S3.SS4.SSS1.Px1 "Jacobian of the mapping. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), [§A3.4.1](#S3.SS4.SSS1.Px2 "Proof sketch: Bounding the largest singular value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
  + bounds [§A3.4.1](#S3.SS4.SSS1.Px4 "Bounds on the entries of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
  + definition [§A3.4.1](#S3.SS4.SSS1.Px1.p1.7 "Jacobian of the mapping. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
  + derivatives [§A3.4.1](#S3.SS4.SSS1.Px3 "Bounds on the derivatives of the Jacobian entries. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
  + entries [§A3.4.1](#S3.SS4.SSS1.Px1.p3.1 "Jacobian of the mapping. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), [§A3.4.1](#S3.SS4.SSS1.Px4 "Bounds on the entries of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
  + singular value [§A3.4.1](#S3.SS4.SSS1.Px2.p3.3 "Proof sketch: Bounding the largest singular value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
  + singular value bound [§A3.4.1](#S3.SS4.SSS1.Px6 "Upper Bounds on the Largest Singular Value of the Jacobian. ‣ A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
* lemmata [§A3.4](#S3.SS4 "A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
  + Jacobian bound [§A3.4.1](#S3.SS4.SSS1 "A3.4.1 Lemmata for proofing Theorem 1 (part 1): Jacobian norm smaller than one ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
* mapping g𝑔g [Normalization and SNNs.](#Sx2.SS0.SSS0.Px1.p2.2 "Normalization and SNNs. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks"), [Deriving the Mean and Variance Mapping Function g𝑔g.](#Sx2.SS0.SSS0.Px3.p1.22 "Deriving the Mean and Variance Mapping Function 𝑔. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")
  + definition [§A1](#S1.p3.3 "A1 Background ‣ Self-Normalizing Neural Networks")
* mapping in domain [§A3.4.2](#S3.SS4.SSS2 "A3.4.2 Lemmata for proofing Theorem 1 (part 2): Mapping within domain ‣ A3.4 Lemmata and Other Tools Required for the Proofs ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
* self-normalizing neural networks [Definition 1](#Thmdefinition1 "Definition 1 (Self-normalizing neural net). ‣ Normalization and SNNs. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")
* SELU
  + definition [Constructing Self-Normalizing Neural Networks.](#Sx2.SS0.SSS0.Px2.p2.4 "Constructing Self-Normalizing Neural Networks. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")
  + parameters [§A1](#S1.p4.12 "A1 Background ‣ Self-Normalizing Neural Networks"), [Stable and Attracting Fixed Point (𝟎,𝟏)01\bm{(0,1)} for Normalized Weights.](#Sx2.SS0.SSS0.Px4.p1.25 "Stable and Attracting Fixed Point (𝟎,𝟏) for Normalized Weights. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")
* Theorem 1 [§A2.1](#S2.SS1 "A2.1 Theorem 1: Stable and Attracting Fixed Points Close to (0,1) ‣ A2 Theorems ‣ Self-Normalizing Neural Networks"), [Theorem 1](#Thmtheorem1 "Theorem 1 (Stable and Attracting Fixed Points). ‣ Stable and Attracting Fixed Points for Unnormalized Weights. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")
  + proof [§A3.1](#S3.SS1 "A3.1 Proof of Theorem 1 ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks"), [§A3.1](#S3.SS1.3 "Proof. ‣ A3.1 Proof of Theorem 1 ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
  + proof sketch [Proof.](#Sx2.SS0.SSS0.Px5.1 "Proof. ‣ Stable and Attracting Fixed Points for Unnormalized Weights. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")
* Theorem 2 [§A2.2](#S2.SS2.p1.8 "A2.2 Theorem 2: Decreasing Variance from Above ‣ A2 Theorems ‣ Self-Normalizing Neural Networks"), [Theorem 2](#Thmtheorem2 "Theorem 2 (Decreasing 𝜈). ‣ Stable and Attracting Fixed Points for Unnormalized Weights. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")
  + proof [§A3.2](#S3.SS2 "A3.2 Proof of Theorem 2 ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")
* Theorem 3 [§A2.3](#S2.SS3.p1.14 "A2.3 Theorem 3: Increasing Variance from Below ‣ A2 Theorems ‣ Self-Normalizing Neural Networks"), [Theorem 3](#Thmtheorem3 "Theorem 3 (Increasing 𝜈). ‣ Stable and Attracting Fixed Points for Unnormalized Weights. ‣ Self-normalizing Neural Networks (SNNs) ‣ Self-Normalizing Neural Networks")
  + proof [§A3.3](#S3.SS3 "A3.3 Proof of Theorem 3 ‣ A3 Proofs of the Theorems ‣ Self-Normalizing Neural Networks")

[◄](/html/1706.02513)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/1706.02515)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+1706.02515)
[View original  
on arXiv](https://arxiv.org/abs/1706.02515)[►](/html/1706.02516)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Tue Mar 5 05:27:07 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
