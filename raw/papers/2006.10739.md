---
arxiv: '2006.10739'
authors:
- Matthew Tancik
- Pratul P. Srinivasan
- Ben Mildenhall
- Sara Fridovich-Keil
- Nithin Raghavan
- Utkarsh Singhal
- Ravi Ramamoorthi
- Jonathan T. Barron
- Ren Ng
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional
  Domains
url: http://arxiv.org/abs/2006.10739v1
year: 2020
---

[2006.10739] Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains














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



\doparttoc\faketableofcontents

# Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains

Matthew Tancik1\*
&Pratul P. Srinivasan1,2\*
&Ben Mildenhall1\*
&Sara Fridovich-Keil1
Nithin Raghavan1
&Utkarsh Singhal1
&Ravi Ramamoorthi3
&Jonathan T. Barron2
&Ren Ng1
&
  
1University of California, Berkeley   2Google Research   3University of California, San Diego

###### Abstract

We show that passing input points through a simple Fourier feature mapping enables a multilayer perceptron (MLP) to learn high-frequency functions in low-dimensional problem domains. These results shed light on recent advances in computer vision and graphics that achieve state-of-the-art results by using MLPs to represent complex 3D objects and scenes. Using tools from the neural tangent kernel (NTK) literature, we show that a standard MLP fails to learn high frequencies both in theory and in practice. To overcome this spectral bias, we use a Fourier feature mapping to transform the effective NTK into a stationary kernel with a tunable bandwidth. We suggest an approach for selecting problem-specific Fourier features that greatly improves the performance of MLPs for low-dimensional regression tasks relevant to the computer vision and graphics communities.

## 1 Introduction

A recent line of research in computer vision and graphics replaces traditional discrete representations of objects, scene geometry, and appearance (e.g. meshes and voxel grids) with continuous functions parameterized by deep fully-connected networks (also called multilayer perceptrons or MLPs).
These MLPs, which we will call “coordinate-based” MLPs, take low-dimensional coordinates as inputs (typically points in ℝ3superscriptℝ3\mathbb{R}^{3}) and are trained to output a representation of shape, density, and/or color at each input location (see Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains")).
This strategy is compelling since coordinate-based MLPs are amenable to gradient-based optimization and machine learning, and can be orders of magnitude more compact than grid-sampled representations.
Coordinate-based MLPs have been used to represent images [[28](#bib.bib28), [38](#bib.bib38)] (referred to as “compositional pattern producing networks”), volume density [[27](#bib.bib27)], occupancy [[24](#bib.bib24)], and signed distance [[32](#bib.bib32)], and have achieved state-of-the-art results across a variety of tasks such as shape representation [[9](#bib.bib9), [10](#bib.bib10), [12](#bib.bib12), [13](#bib.bib13), [17](#bib.bib17), [26](#bib.bib26), [32](#bib.bib32)], texture synthesis [[15](#bib.bib15), [31](#bib.bib31)], shape inference from images [[22](#bib.bib22), [23](#bib.bib23)], and novel view synthesis [[27](#bib.bib27), [29](#bib.bib29), [35](#bib.bib35), [37](#bib.bib37)].

We leverage recent progress in modeling the behavior of deep networks using kernel regression with a neural tangent kernel (NTK) [[16](#bib.bib16)] to theoretically and experimentally show that standard MLPs are poorly suited for these low-dimensional coordinate-based vision and graphics tasks. In particular, MLPs have difficulty learning high frequency functions, a phenomenon referred to in the literature as “spectral bias” [[3](#bib.bib3), [33](#bib.bib33)].
NTK theory suggests that this is because standard coordinate-based MLPs correspond to kernels with a rapid frequency falloff, which effectively prevents them from being able to represent the high-frequency content present in natural images and scenes.

A few recent works [[27](#bib.bib27), [44](#bib.bib44)] have experimentally found that a heuristic sinusoidal mapping of input coordinates (called a “positional encoding”) allows MLPs to represent higher frequency content. We observe that this is a special case of Fourier features [[34](#bib.bib34)]: mapping input coordinates 𝐯𝐯\mathbf{v} to γ​(𝐯)=[a1​cos⁡(2​π​𝐛1T​𝐯),a1​sin⁡(2​π​𝐛1T​𝐯),…,am​cos⁡(2​π​𝐛mT​𝐯),am​sin⁡(2​π​𝐛mT​𝐯)]T𝛾𝐯superscript

subscript𝑎12𝜋superscriptsubscript𝐛1T𝐯subscript𝑎12𝜋superscriptsubscript𝐛1T𝐯…subscript𝑎𝑚2𝜋superscriptsubscript𝐛𝑚T𝐯subscript𝑎𝑚2𝜋superscriptsubscript𝐛𝑚T𝐯
T\gamma(\mathbf{v})=\left[a\_{1}\cos(2\pi\mathbf{b}\_{1}^{\mathrm{T}}\mathbf{v}),a\_{1}\sin(2\pi\mathbf{b}\_{1}^{\mathrm{T}}\mathbf{v}),\ldots,a\_{m}\cos(2\pi\mathbf{b}\_{m}^{\mathrm{T}}\mathbf{v}),a\_{m}\sin(2\pi\mathbf{b}\_{m}^{\mathrm{T}}\mathbf{v})\right]^{\mathrm{T}}
before passing them into an MLP.
We show that this mapping transforms the NTK into a stationary (shift-invariant) kernel and enables tuning the NTK’s spectrum by modifying the frequency vectors 𝐛jsubscript𝐛𝑗\mathbf{b}\_{j},
thereby controlling the range of frequencies that can be learned by the corresponding MLP.
We show that the simple strategy of setting aj=1subscript𝑎𝑗1a\_{j}=1 and randomly sampling 𝐛jsubscript𝐛𝑗\mathbf{b}\_{j}
from an isotropic distribution achieves good performance, and that the scale (standard deviation) of this distribution matters much more than its specific shape.
We train MLPs with this Fourier feature input mapping across a range of tasks relevant to the computer vision and graphics communities. As highlighted in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"), our proposed mapping dramatically improves the performance of coordinate-based MLPs. In summary, we make the following contributions:

* •

  We leverage NTK theory and simple experiments to show that a Fourier feature mapping can be used to overcome the spectral bias of coordinate-based MLPs towards low frequencies by allowing them to learn much higher frequencies (Section [4](#S4 "4 Fourier Features for a Tunable Stationary Neural Tangent Kernel ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains")).
* •

  We demonstrate that a random Fourier feature mapping with an appropriately chosen scale can dramatically improve the performance of coordinate-based MLPs across many low-dimensional tasks in computer vision and graphics (Section [5](#S5 "5 Manipulating the Fourier Feature Mapping ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains")).

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Refer to caption  (a) Coordinate-based MLP | |  |  |  |  |  |  | | --- | --- | --- | --- | --- | --- | | No Fourier features | γ​(𝐯)=𝐯𝛾𝐯𝐯\gamma(\mathbf{v})=\mathbf{v} | Refer to caption | Refer to caption | Refer to caption | Refer to caption | | With Fourier features | γ​(𝐯)=FF⁡(𝐯)𝛾𝐯FF𝐯\gamma(\mathbf{v})=\operatorname{FF}(\mathbf{v}) | Refer to caption | Refer to caption | Refer to caption | Refer to caption | |  |  | (b) Image regression | (c) 3D shape regression | (d) MRI reconstruction | (e) Inverse rendering | |  |  | (x,y)→RGB→𝑥𝑦RGB\!(x,\!y)\!\rightarrow\textrm{RGB} | (x,y,z)→occupancy→𝑥𝑦𝑧occupancy\!(x,\!y,\!z)\!\rightarrow\textrm{occupancy} | (x,y,z)→density→𝑥𝑦𝑧density\!(x,\!y,\!z)\!\rightarrow\textrm{density} | (x,y,z)→RGB, density→𝑥𝑦𝑧RGB, density\!(x,\!y,\!z)\!\rightarrow\!\textrm{RGB, density}\! | |

Figure 1: Fourier features improve the results of coordinate-based MLPs for a variety of high-frequency low-dimensional regression tasks, both with direct (b, c) and indirect (d, e) supervision. We visualize an example MLP (a) for an image regression task (b), where the input to the network is a pixel coordinate and the output is that pixel’s color. Passing coordinates directly into the network (top) produces blurry images, whereas preprocessing the input with a Fourier feature mapping (bottom) enables the MLP to represent higher frequency details.

## 2 Related Work

Our work is motivated by the widespread use of coordinate-based MLPs to represent a variety of visual signals, including images [[38](#bib.bib38)] and 3D scenes [[24](#bib.bib24), [27](#bib.bib27), [32](#bib.bib32)]. In particular, our analysis is intended to clarify experimental results demonstrating that an input mapping of coordinates (which they called a “positional encoding”) using sinusoids with logarithmically-spaced axis-aligned frequencies improves the performance of coordinate-based MLPs on the tasks of novel view synthesis from 2D images [[27](#bib.bib27)] and protein structure modeling from cryo-electron microscopy [[44](#bib.bib44)]. We analyze this technique to show that it corresponds to a modification of the MLP’s NTK, and we show that other non-axis-aligned frequency distributions can outperform this positional encoding.

Prior works in natural language processing and time series analysis [[18](#bib.bib18), [39](#bib.bib39), [42](#bib.bib42)] have used a similar positional encoding to represent time or 1D position. In particular, Xu et al.  [[42](#bib.bib42)] use random Fourier features (RFF) [[34](#bib.bib34)] to approximate stationary kernels with a sinusoidal input mapping and propose techniques to tune the mapping parameters. Our work extends this by directly explaining such mappings as a modification of the resulting network’s NTK. Additionally, we address the embedding of multidimensional coordinates, which is necessary for vision and graphics tasks.

To analyze the effects of applying a Fourier feature mapping to input coordinates before passing them through an MLP, we rely on recent theoretical work that models neural networks in the limits of infinite width and infinitesimal learning rate as kernel regression using the NTK [[2](#bib.bib2), [5](#bib.bib5), [11](#bib.bib11), [16](#bib.bib16), [20](#bib.bib20)]. In particular, we use the analyses from Lee et al.  [[20](#bib.bib20)] and Arora et al.  [[2](#bib.bib2)], which show that the outputs of a network throughout gradient descent remain close to those of a linear dynamical system whose convergence rate is governed by the eigenvalues of the NTK matrix [[2](#bib.bib2), [3](#bib.bib3), [5](#bib.bib5), [20](#bib.bib20), [43](#bib.bib43)]. Analysis of the NTK’s eigendecomposition shows that its eigenvalue spectrum decays rapidly as a function of frequency, which explains the widely-observed “spectral bias” of deep networks towards learning low-frequency functions [[3](#bib.bib3), [4](#bib.bib4), [33](#bib.bib33)].

We leverage this analysis to consider the implications of adding a Fourier feature mapping before the network, and we show that this mapping has a significant effect on the NTK’s eigenvalue spectrum and on the corresponding network’s convergence properties in practice.

## 3 Background and Notation

To lay the foundation for our theoretical analysis, we first review classic kernel regression and its connection to recent results that analyze the training dynamics and generalization behavior of deep fully-connected networks. In later sections, we use these tools to analyze the effects of training coordinate-based MLPs with Fourier feature mappings.

Kernel regression.
Kernel regression is a classic nonlinear regression algorithm [[40](#bib.bib40)]. Given a training dataset (𝐗,𝐲)={(𝐱i,yi)}i=1n𝐗𝐲superscriptsubscriptsubscript𝐱𝑖subscript𝑦𝑖𝑖1𝑛(\mathbf{X},\mathbf{y})=\{(\mathbf{x}\_{i},y\_{i})\}\_{i=1}^{n}, where 𝐱isubscript𝐱𝑖\mathbf{x}\_{i} are input points and yi=f​(𝐱i)subscript𝑦𝑖𝑓subscript𝐱𝑖y\_{i}=f(\mathbf{x}\_{i}) are the corresponding scalar output labels, kernel regression constructs an estimate f^^𝑓\hat{f} of the underlying function at any point 𝐱𝐱\mathbf{x} as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | f^​(𝐱)=∑i=1n(𝐊−1​𝐲)i​k​(𝐱i,𝐱),^𝑓𝐱superscriptsubscript𝑖1𝑛subscriptsuperscript𝐊1𝐲𝑖𝑘subscript𝐱𝑖𝐱\hat{f}(\mathbf{x})=\sum\_{i=1}^{n}\left(\mathbf{K}^{-1}\mathbf{y}\right)\_{i}k(\mathbf{x}\_{i},\mathbf{x})\,, |  | (1) |

where 𝐊𝐊\mathbf{K} is an n×n𝑛𝑛n\times n kernel (Gram) matrix with entries 𝐊i​j=k​(𝐱i,𝐱j)subscript𝐊𝑖𝑗𝑘subscript𝐱𝑖subscript𝐱𝑗\mathbf{K}\_{ij}=k(\mathbf{x}\_{i},\mathbf{x}\_{j}) and k𝑘k is a symmetric positive semidefinite (PSD) kernel function which represents the “similarity” between two input vectors. Intuitively, the kernel regression estimate at any point 𝐱𝐱\mathbf{x} can be thought of as a weighted sum of training labels yisubscript𝑦𝑖y\_{i} using the similarity between the corresponding 𝐱isubscript𝐱𝑖\mathbf{x}\_{i} and 𝐱𝐱\mathbf{x}.

Approximating deep networks with kernel regression. Let f𝑓f be a fully-connected deep network with weights θ𝜃\theta initialized from a Gaussian distribution 𝒩𝒩\mathcal{N}.
Theory proposed by Jacot et al.  [[16](#bib.bib16)] and extended by others [[2](#bib.bib2), [3](#bib.bib3), [20](#bib.bib20)] shows that when the width of the layers in f𝑓f tends to infinity and the learning rate for SGD tends to zero, the function f​(𝐱;θ)𝑓

𝐱𝜃f(\mathbf{x};\theta) converges over the course of training to the kernel regression solution using the *neural tangent kernel* (NTK), defined as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | kNTK​(𝐱i,𝐱j)=𝔼θ∼𝒩​⟨∂f​(𝐱i;θ)∂θ,∂f​(𝐱j;θ)∂θ⟩.subscript𝑘NTKsubscript𝐱𝑖subscript𝐱𝑗subscript𝔼similar-to𝜃𝒩  𝑓  subscript𝐱𝑖𝜃𝜃𝑓  subscript𝐱𝑗𝜃𝜃k\_{\mathrm{NTK}}(\mathbf{x}\_{i},\mathbf{x}\_{j})=\mathbb{E}\_{\theta\sim\mathcal{N}}\left\langle\frac{\partial f(\mathbf{x}\_{i};\theta)}{\partial\theta},\frac{\partial f(\mathbf{x}\_{j};\theta)}{\partial\theta}\right\rangle\,. |  | (2) |

When the inputs are restricted to a hypersphere, the NTK for an MLP can be written as a dot product kernel (a kernel in the form hNTK​(𝐱iT​𝐱j)subscriptℎNTKsuperscriptsubscript𝐱𝑖Tsubscript𝐱𝑗h\_{\mathrm{NTK}}(\mathbf{x}\_{i}^{\mathrm{T}}\mathbf{x}\_{j}) for a scalar function hNTK:ℝ→ℝ:subscriptℎNTK→ℝℝh\_{\mathrm{NTK}}:\mathbb{R}\to\mathbb{R}).

Prior work [[2](#bib.bib2), [3](#bib.bib3), [16](#bib.bib16), [20](#bib.bib20)] shows that an NTK linear system model can be used to approximate the dynamics of a deep network during training. We consider a network trained with an L2 loss and a learning rate η𝜂\eta, where the network’s weights are initialized such that the output of the network at initialization is close to zero. Under asymptotic conditions stated in Lee et al.  [[20](#bib.bib20)], the network’s output for any data 𝐗testsubscript𝐗test\mathbf{X}\_{\mathrm{test}} after t𝑡t training iterations can be approximated as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐲^(t)≈𝐊test​𝐊−1​(𝐈−e−η​𝐊​t)​𝐲,superscript^𝐲𝑡subscript𝐊testsuperscript𝐊1𝐈superscript𝑒𝜂𝐊𝑡𝐲\hat{\mathbf{y}}^{(t)}\approx\mathbf{K}\_{\mathrm{test}}\mathbf{K}^{-1}\left(\mathbf{I}-e^{-\eta\mathbf{K}t}\right)\mathbf{y}\,, |  | (3) |

where 𝐲^(t)=f​(𝐗test;θ)superscript^𝐲𝑡𝑓

subscript𝐗test𝜃\hat{\mathbf{y}}^{(t)}=f(\mathbf{X}\_{\mathrm{test}};\theta) are the network’s predictions on input points 𝐗testsubscript𝐗test\mathbf{X}\_{\mathrm{test}} at training iteration t𝑡t, 𝐊𝐊\mathbf{K} is the NTK matrix between all pairs of training points in 𝐗𝐗\mathbf{X}, and 𝐊testsubscript𝐊test\mathbf{K}\_{\mathrm{test}} is the NTK matrix between all points in 𝐗testsubscript𝐗test\mathbf{X}\_{\mathrm{test}} and all points in the training dataset 𝐗𝐗\mathbf{X}.

Spectral bias when training neural networks.
Let us consider the training error 𝐲^train(t)−𝐲subscriptsuperscript^𝐲𝑡train𝐲\mathbf{\hat{y}}^{(t)}\_{\textrm{train}}-\mathbf{y}, where 𝐲^train(t)subscriptsuperscript^𝐲𝑡train\mathbf{\hat{y}}^{(t)}\_{\textrm{train}} are the network’s predictions on the training dataset at iteration t𝑡t. Since the NTK matrix 𝐊𝐊\mathbf{K} must be PSD, we can take its eigendecomposition 𝐊=𝐐​𝚲​𝐐T𝐊𝐐𝚲superscript𝐐T\mathbf{K}=\mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^{\mathrm{T}}, where 𝐐𝐐\mathbf{Q} is orthogonal and 𝚲𝚲\mathbf{\Lambda} is a diagonal matrix whose entries are the eigenvalues λi≥0subscript𝜆𝑖0\lambda\_{i}\geq 0 of 𝐊𝐊\mathbf{K}.
Then, since e−η​𝐊​t=𝐐​e−η​𝚲​t​𝐐Tsuperscript𝑒𝜂𝐊𝑡𝐐superscript𝑒𝜂𝚲𝑡superscript𝐐Te^{-\eta\mathbf{K}t}=\mathbf{Q}e^{-\eta\mathbf{\Lambda}t}\mathbf{Q}^{\mathrm{T}}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐐T​(𝐲^train(t)−𝐲)≈𝐐T​((𝐈−e−η​𝐊​t)​𝐲−𝐲)=−e−η​𝚲​t​𝐐T​𝐲.superscript𝐐Tsubscriptsuperscript^𝐲𝑡train𝐲superscript𝐐T𝐈superscript𝑒𝜂𝐊𝑡𝐲𝐲superscript𝑒𝜂𝚲𝑡superscript𝐐T𝐲\mathbf{Q}^{\mathrm{T}}(\mathbf{\hat{y}}^{(t)}\_{\textrm{train}}-\mathbf{y})\approx\mathbf{Q}^{\mathrm{T}}\left(\left(\mathbf{I}-e^{-\eta\mathbf{K}t}\right)\mathbf{y}-\mathbf{y}\right)=-e^{-\eta\mathbf{\Lambda}t}\mathbf{Q}^{\mathrm{T}}\mathbf{y}\,. |  | (4) |

This means that if we consider training convergence in the eigenbasis of the NTK, the ithsuperscript𝑖thi^{\textrm{th}} component of the absolute error |𝐐T​(𝐲^train(t)−𝐲)|isubscriptsuperscript𝐐Tsubscriptsuperscript^𝐲𝑡train𝐲𝑖|\mathbf{Q}^{\mathrm{T}}(\mathbf{\hat{y}}^{(t)}\_{\textrm{train}}-\mathbf{y})|\_{i} will decay approximately exponentially at the rate η​λi𝜂subscript𝜆𝑖\eta\lambda\_{i}. In other words, components of the target function that correspond to kernel eigenvectors with larger eigenvalues will be learned faster. For a conventional MLP, the eigenvalues of the NTK decay rapidly [[4](#bib.bib4), [5](#bib.bib5), [14](#bib.bib14)]. This results in extremely slow convergence to the high frequency components of the target function, to the point where standard MLPs are effectively unable to learn these components, as visualized in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"). Next, we describe a technique to address this slow convergence by using a Fourier feature mapping of input coordinates before passing them to the MLP.

## 4 Fourier Features for a Tunable Stationary Neural Tangent Kernel

Machine learning analysis typically addresses the case in which inputs are high dimensional points (e.g. the pixels of an image reshaped into a vector) and training examples are sparsely distributed.
In contrast, in this work we consider *low-dimensional regression* tasks, wherein inputs are assumed to be dense coordinates in a subset of ℝdsuperscriptℝ𝑑\mathbb{R}^{d} for small values of d𝑑d (e.g. pixel coordinates).
This setting has two significant implications when viewing deep networks through the lens of kernel regression:

1. 1.

   We would like the composed NTK to be shift-invariant over the input domain, since the training points are distributed with uniform density. In problems where the inputs are normalized to the surface of a hypersphere (common in machine learning), a dot product kernel (such as the regular NTK) corresponds to spherical convolution. However, inputs in our setting are dense in Euclidean space. A Fourier feature mapping of input coordinates makes the composed NTK stationary (shift-invariant), acting as a convolution kernel over the input domain (see Appendix [C](#A3 "Appendix C Stationary kernels ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") for additional discussion on stationary kernels).
2. 2.

   We would like to control the bandwidth of the NTK to improve training speed and generalization. As we see from Eqn. [4](#S3.E4 "In 3 Background and Notation ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"), a “wider” kernel with a slower spectral falloff achieves faster training convergence for high frequency components. However, we know from signal processing that reconstructing a signal using a kernel whose spectrum is *too* wide causes high frequency aliasing artifacts. We show in Section [5](#S5 "5 Manipulating the Fourier Feature Mapping ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") that a Fourier feature input mapping can be tuned to lie between these “underfitting’ and “overfitting” extremes, enabling both fast convergence and low test error.

Fourier features and the composed neural tangent kernel.
Fourier feature mappings have been used in many applications since their introduction in the seminal work of Rahimi and Recht [[34](#bib.bib34)], which used random Fourier features to approximate an arbitrary stationary kernel function by applying Bochner’s theorem. Extending this technique, we use a Fourier feature mapping γ𝛾\gamma to featurize input coordinates before passing them through a coordinate-based MLP, and investigate the theoretical and practical effect this has on convergence speed and generalization. The function γ𝛾\gamma maps input points 𝐯∈[0,1)d𝐯superscript01𝑑\mathbf{v}\in[0,1)^{d} to the surface of a higher dimensional hypersphere with a set of sinusoids:

|  |  |  |  |
| --- | --- | --- | --- |
|  | γ​(𝐯)=[a1​cos⁡(2​π​𝐛1T​𝐯),a1​sin⁡(2​π​𝐛1T​𝐯),…,am​cos⁡(2​π​𝐛mT​𝐯),am​sin⁡(2​π​𝐛mT​𝐯)]T.𝛾𝐯superscript  subscript𝑎12𝜋superscriptsubscript𝐛1T𝐯subscript𝑎12𝜋superscriptsubscript𝐛1T𝐯…subscript𝑎𝑚2𝜋superscriptsubscript𝐛𝑚T𝐯subscript𝑎𝑚2𝜋superscriptsubscript𝐛𝑚T𝐯 T\gamma(\mathbf{v})=\left[a\_{1}\cos(2\pi\mathbf{b}\_{1}^{\mathrm{T}}\mathbf{v}),a\_{1}\sin(2\pi\mathbf{b}\_{1}^{\mathrm{T}}\mathbf{v}),\ldots,a\_{m}\cos(2\pi\mathbf{b}\_{m}^{\mathrm{T}}\mathbf{v}),a\_{m}\sin(2\pi\mathbf{b}\_{m}^{\mathrm{T}}\mathbf{v})\right]^{\mathrm{T}}\,. |  | (5) |

Because cos⁡(α−β)=cos⁡α​cos⁡β+sin⁡α​sin⁡β𝛼𝛽𝛼𝛽𝛼𝛽\cos(\alpha-\beta)=\cos\alpha\cos\beta+\sin\alpha\sin\beta, the kernel function induced by this mapping is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | kγ​(𝐯1,𝐯2)=γ​(𝐯1)T​γ​(𝐯2)=∑j=1maj2​cos⁡(2​π​𝐛jT​(𝐯1−𝐯2))=hγ​(𝐯1−𝐯2),subscript𝑘𝛾subscript𝐯1subscript𝐯2𝛾superscriptsubscript𝐯1T𝛾subscript𝐯2superscriptsubscript𝑗1𝑚superscriptsubscript𝑎𝑗22𝜋superscriptsubscript𝐛𝑗Tsubscript𝐯1subscript𝐯2subscriptℎ𝛾subscript𝐯1subscript𝐯2\displaystyle k\_{\gamma}(\mathbf{v}\_{1},\mathbf{v}\_{2})=\gamma(\mathbf{v}\_{1})^{\mathrm{T}}\gamma(\mathbf{v}\_{2})=\sum\_{j=1}^{m}a\_{j}^{2}\cos\left(2\pi\mathbf{b}\_{j}^{\mathrm{T}}\left(\mathbf{v}\_{1}-\mathbf{v}\_{2}\right)\right)=h\_{\gamma}(\mathbf{v}\_{1}-\mathbf{v}\_{2})\,, |  | (6) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | where ​hγ​(𝐯Δ)≜∑j=1maj2​cos⁡(2​π​𝐛jT​𝐯Δ).≜where subscriptℎ𝛾subscript𝐯Δsuperscriptsubscript𝑗1𝑚superscriptsubscript𝑎𝑗22𝜋superscriptsubscript𝐛𝑗Tsubscript𝐯Δ\displaystyle\textrm{where }h\_{\gamma}(\mathbf{v}\_{\Delta})\triangleq\sum\_{j=1}^{m}a\_{j}^{2}\cos(2\pi\mathbf{b}\_{j}^{\mathrm{T}}\mathbf{v}\_{\Delta})\,. |  | (7) |

Note that this kernel is stationary (a function of only the difference between points). We can think of the mapping as a Fourier approximation of a kernel function: 𝐛jsubscript𝐛𝑗\mathbf{b}\_{j} are the Fourier basis frequencies used to approximate the kernel, and aj2superscriptsubscript𝑎𝑗2a\_{j}^{2} are the corresponding Fourier series coefficients.

After computing the Fourier features for our input points, we pass them through an MLP to get f​(γ​(𝐯);θ)𝑓

𝛾𝐯𝜃f(\gamma(\mathbf{v});\theta). As discussed previously, the result of training a network can be approximated by kernel regression using the kernel hNTK​(𝐱iT​𝐱j)subscriptℎNTKsuperscriptsubscript𝐱𝑖Tsubscript𝐱𝑗h\_{\mathrm{NTK}}(\mathbf{x}\_{i}^{\mathrm{T}}\mathbf{x}\_{j}). In our case, 𝐱i=γ​(𝐯i)subscript𝐱𝑖𝛾subscript𝐯𝑖\mathbf{x}\_{i}=\gamma(\mathbf{v}\_{i}) so the composed kernel becomes:

|  |  |  |  |
| --- | --- | --- | --- |
|  | hNTK​(𝐱iT​𝐱j)=hNTK​(γ​(𝐯i)T​γ​(𝐯j))=hNTK​(hγ​(𝐯i−𝐯j)).subscriptℎNTKsuperscriptsubscript𝐱𝑖Tsubscript𝐱𝑗subscriptℎNTK𝛾superscriptsubscript𝐯𝑖T𝛾subscript𝐯𝑗subscriptℎNTKsubscriptℎ𝛾subscript𝐯𝑖subscript𝐯𝑗h\_{\mathrm{NTK}}(\mathbf{x}\_{i}^{\mathrm{T}}\mathbf{x}\_{j})=h\_{\mathrm{NTK}}\left(\gamma\left(\mathbf{v}\_{i}\right)^{\mathrm{T}}\gamma\left(\mathbf{v}\_{j}\right)\right)=h\_{\mathrm{NTK}}\left(h\_{\gamma}\left(\mathbf{v}\_{i}-\mathbf{v}\_{j}\right)\right). |  | (8) |

Thus, training a network on these embedded input points corresponds to kernel regression with the *stationary* composed NTK function hNTK∘hγsubscriptℎNTKsubscriptℎ𝛾h\_{\mathrm{NTK}}\circ h\_{\gamma} . The MLP function approximates a convolution of the composed NTK with a weighted Dirac delta at each input training point 𝐯isubscript𝐯𝑖\mathbf{v}\_{i}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | f^=(hNTK∘hγ)∗∑i=1nwi​δ𝐯i^𝑓subscriptℎNTKsubscriptℎ𝛾superscriptsubscript𝑖1𝑛subscript𝑤𝑖subscript𝛿subscript𝐯𝑖\displaystyle\hat{f}=\left(h\_{\mathrm{NTK}}\circ h\_{\gamma}\right)\*\sum\_{i=1}^{n}w\_{i}\delta\_{\mathbf{v}\_{i}} |  | (9) |

where 𝐰=𝐊−1​𝐲𝐰superscript𝐊1𝐲\mathbf{w}=\mathbf{K}^{-1}\mathbf{y} (from Eqn. [1](#S3.E1 "In 3 Background and Notation ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains")).
This allows us to draw analogies to signal processing, where the composed NTK acts similarly to a reconstruction filter. In the next section, we show that the frequency decay of the composed NTK determines the behavior of the reconstructed signal.

## 5 Manipulating the Fourier Feature Mapping

Preprocessing the inputs to a coordinate-based MLP with a Fourier feature mapping creates a composed NTK that is not only stationary but also *tunable*. By manipulating the settings of the ajsubscript𝑎𝑗a\_{j} and 𝐛jsubscript𝐛𝑗\mathbf{b}\_{j} parameters in Eqn. [5](#S4.E5 "In 4 Fourier Features for a Tunable Stationary Neural Tangent Kernel ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"), it is possible to dramatically change both the rate of convergence and the generalization behavior of the resulting network.
In this section, we investigate the effects of the Fourier feature mapping in the setting of 1D function regression.

We train MLPs to learn signals f𝑓f defined on the interval [0,1)01[0,1). We sample c​n𝑐𝑛cn linearly spaced points on the interval,
using every cthsuperscript𝑐thc^{\textrm{th}} point as the training set and the remaining points as the test set.
Since our composed kernel function is stationary, evaluating it at linearly spaced points on a periodic domain makes the resulting kernel matrix circulant: it represents a convolution and is diagonalizable by the Fourier transform. Thus, we can compute the eigenvalues of the composed NTK matrix by simply taking the Fourier transform of a single row. All experiments are implemented in JAX [[8](#bib.bib8)] and the NTK functions are calculated automatically using the Neural Tangents library [[30](#bib.bib30)].

Visualizing the composed NTK.
We first visualize how modifying the Fourier feature mapping changes the composed NTK. We set bj=jsubscript𝑏𝑗𝑗b\_{j}=j (full Fourier basis in 1D) and aj=1/jpsubscript𝑎𝑗1superscript𝑗𝑝a\_{j}=1/j^{p} for j=1,…,n/2𝑗

1…𝑛2j=1,\ldots,n/2.
We use p=∞𝑝p=\infty to denote the mapping γ​(v)=[cos⁡2​π​v,sin⁡2​π​v]T𝛾𝑣superscript2𝜋𝑣2𝜋𝑣T\gamma(v)=\left[\cos 2\pi v,\sin 2\pi v\right]^{\mathrm{T}} that simply wraps [0,1)01[0,1) around the unit circle (this is referred to as the “basic” mapping in later experiments).
Figure [2](#S5.F2 "Figure 2 ‣ 5 Manipulating the Fourier Feature Mapping ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") demonstrates the effect of varying p𝑝p on the composed NTK. By construction, lower p𝑝p values result in a slower falloff in the frequency domain and a correspondingly narrower kernel in the spatial domain.

![Refer to caption](/html/2006.10739/assets/x5.png)


Figure 2: 
Adding a Fourier feature mapping can improve the poor conditioning of a coordinate-based MLP’s neural tangent kernel (NTK).
(a) We visualize the NTK function kNTK​(xi,xj)subscript𝑘NTKsubscript𝑥𝑖subscript𝑥𝑗k\_{\mathrm{NTK}}(x\_{i},x\_{j}) (Eqn. [2](#S3.E2 "In 3 Background and Notation ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains")) for a 4-layer ReLU MLP with one scalar input.
This kernel is not shift-invariant and does not have a strong diagonal, making it poorly suited for kernel regression in low-dimensional problems.
(b) A basic input mapping γ​(v)=[cos⁡2​π​v,sin⁡2​π​v]T𝛾𝑣superscript2𝜋𝑣2𝜋𝑣T\gamma(v)=\left[\cos 2\pi v,\sin 2\pi v\right]^{\mathrm{T}}
makes the composed NTK kNTK​(γ​(vi),γ​(vj))subscript𝑘NTK𝛾subscript𝑣𝑖𝛾subscript𝑣𝑗k\_{\mathrm{NTK}}(\gamma(v\_{i}),\gamma(v\_{j})) shift-invariant (stationary).
(c) A Fourier feature input mapping (Eqn. [5](#S4.E5 "In 4 Fourier Features for a Tunable Stationary Neural Tangent Kernel ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains")) can be used to tune the composed kernel’s width, where we set aj=1/jpsubscript𝑎𝑗1superscript𝑗𝑝a\_{j}=1/j^{p} and bj=jsubscript𝑏𝑗𝑗b\_{j}=j for j=1,…,n/2𝑗

1…𝑛2j=1,\ldots,n/2. (d) Higher frequency mappings (lower p𝑝p) result in composed kernels with wider spectra, which enables faster convergence for high-frequency components (see Figure [3](#S5.F3 "Figure 3 ‣ 5 Manipulating the Fourier Feature Mapping ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains")).

Effects of Fourier features on network convergence.
We generate ground truth 1D functions by sampling c​n𝑐𝑛cn values from a family with parameter α𝛼\alpha as follows: we sample a standard i.i.d. Gaussian vector of length c​n𝑐𝑛cn, scale its ithsuperscript𝑖thi^{\textrm{th}} entry by 1/iα1superscript𝑖𝛼1/i^{\alpha}, then return the real component of its inverse Fourier transform. We will refer to this as a “1/fα1superscript𝑓𝛼1/f^{\alpha} noise” signal.

In Figure [3](#S5.F3 "Figure 3 ‣ 5 Manipulating the Fourier Feature Mapping ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"), we train MLPs (4 layers, 1024 channels, ReLU activations) to fit a bandlimited 1/f11superscript𝑓11/f^{1} noise signal (c=8,n=32formulae-sequence𝑐8𝑛32c=8,n=32) using Fourier feature mappings with different p𝑝p values. Figures [3](#S5.F3 "Figure 3 ‣ 5 Manipulating the Fourier Feature Mapping ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains")b and [3](#S5.F3 "Figure 3 ‣ 5 Manipulating the Fourier Feature Mapping ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains")d show that the NTK linear dynamics model accurately predict the effects of modifying the Fourier feature mapping parameters. Separating different frequency components of the training error in Figure [3](#S5.F3 "Figure 3 ‣ 5 Manipulating the Fourier Feature Mapping ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains")c reveals that networks with narrower NTK spectra converge faster for low frequency components but essentially never converge for high frequency components, whereas networks with wider NTK spectra successfully converge across all components.
The Fourier feature mapping p=1𝑝1p=1 has adequate power across frequencies present in the target signal (so the network converges rapidly during training) but limited power in higher frequencies (preventing overfitting or aliasing).

![Refer to caption](/html/2006.10739/assets/x6.png)


Figure 3: Combining a network with a Fourier feature mapping has dramatic effects on convergence and generalization.
Here we train a network on 32 sampled points from a 1D function (a) using mappings shown in Fig. [2](#S5.F2 "Figure 2 ‣ 5 Manipulating the Fourier Feature Mapping ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"). A mapping with a smaller p𝑝p value yields a composed NTK with more power in higher frequencies, enabling the corresponding network to learn a higher frequency function.
The theoretical and experimental training loss improves monotonically with higher frequency kernels (d), but the test-set loss is lowest at p=1𝑝1p=1 and falls as the network starts to overfit (b). As predicted by Eqn. [4](#S3.E4 "In 3 Background and Notation ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"), we see roughly log-linear convergence of the training loss frequency components (c). Higher frequency kernels result in faster convergence for high frequency loss components, thereby overcoming the “spectral bias” observed when training networks with no input mapping.

Tuning Fourier features in practice.
Eqn. [3](#S3.E3 "In 3 Background and Notation ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") allows us to estimate a trained network’s theoretical loss on a validation set using the composed kernel. For small 1D problems, we can minimize this loss with gradient-based optimization to choose mapping parameters ajsubscript𝑎𝑗a\_{j} (given a dense sampling of bjsubscript𝑏𝑗b\_{j}).
In this carefully controlled setting (1D signals, small training dataset, gradient descent with small learning rate, very wide networks), we find that this optimized mapping also achieves the best performance when training networks.
Please refer to Appendix [A.1](#A1.SS1 "A.1 Optimizing validation error through the NTK linear dynamics ‣ Appendix A Further experiments ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") for details and experiments.

In real-world problems, especially in multiple dimensions, it is not feasible to use a feature mapping that densely samples Fourier basis functions; the number of Fourier basis functions scales with the number of training data points, which grows exponentially with dimension. Instead, we sample a set of random Fourier features [[34](#bib.bib34)] from a parametric distribution. We find that the exact sampling distribution family is much less important than the distribution’s scale (standard deviation).

Figure [4](#S5.F4 "Figure 4 ‣ 5 Manipulating the Fourier Feature Mapping ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") demonstrates this point using hyperparameter sweeps for a variety of sampling distributions. In each subfigure, we draw 1D target signals (c=2,n=1024formulae-sequence𝑐2𝑛1024c=2,n=1024) from a fixed 1/fα1superscript𝑓𝛼1/f^{\alpha} distribution and train networks to learn them. We use random Fourier feature mappings (of length 16) sampled from different distribution families (Gaussian, uniform, uniform in log space, and Laplacian) and sweep over each distribution’s scale.
Perhaps surprisingly, the standard deviation of the sampled frequencies alone is enough to predict test set performance, regardless of the underlying distribution’s shape. We show that this holds for higher-dimensional tasks in Appendix [A.4](#A1.SS4 "A.4 Visualizing underfitting and overfitting in 2D ‣ Appendix A Further experiments ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"). We also observe that passing this sparse sampling of Fourier features through an MLP matches the performance of using a dense set of Fourier features with the same MLP, suggesting a strategy for scaling to higher dimensions. We proceed with a Gaussian distribution for our higher-dimensional experiments in Section [6](#S6 "6 Experiments ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") and treat the scale as a hyperparameter to tune on a validation dataset.

![Refer to caption](/html/2006.10739/assets/x7.png)


Figure 4: 
We find that a sparse random sampling of Fourier features can perform as well as a dense set of features and that the width of the distribution matters more than the shape.
Here, we generate random 1D signals from 1/fα1superscript𝑓𝛼1/f^{\alpha} noise and report the test-set accuracy of different trained models that use a sparse set (16 out of 1024) of random Fourier features sampled from different distributions. Each subplot represents a different family of 1D signals.
Each dot represents a trained network, where the color indicates which Fourier feature sampling distribution is used. We plot the test error of each model versus the empirical standard deviation of its sampled frequencies.
The best models using sparsely sampled features are able to match the performance of a model trained with dense Fourier features (dashed lines with error bars).
All sampling distributions trace out the same curve, exhibiting underfitting (slow convergence) when the standard deviation of sampled frequencies is too low and overfitting when it is too high.
This implies that the precise shape of the distribution used to sample frequencies does not have a significant impact on performance.

## 6 Experiments

We validate the benefits of using Fourier feature mappings for coordinate-based MLPs with experiments on a variety of regression tasks relevant to the computer vision and graphics communities.

### 6.1 Compared mappings

In Table [1](#S6.T1 "Table 1 ‣ 6.2 Tasks ‣ 6 Experiments ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"), we compare the performance of coordinate-based MLPs with no input mapping and with the following Fourier feature mappings (cos,sin

\cos,\sin are applied elementwise):

Basic:
γ​(𝐯)=[cos⁡(2​π​𝐯​v),sin⁡(2​π​𝐯)]T𝛾𝐯superscript2𝜋𝐯𝑣2𝜋𝐯T\gamma(\mathbf{v})=\left[\cos(2\pi\mathbf{v}{v}),\sin(2\pi\mathbf{v})\right]^{\mathrm{T}}.
Simply wraps input coordinates around the circle.

Positional encoding:
γ​(𝐯)=[…,cos⁡(2​π​σj/m​𝐯),sin⁡(2​π​σj/m​𝐯),…]T𝛾𝐯superscript

…2𝜋superscript𝜎𝑗𝑚𝐯2𝜋superscript𝜎𝑗𝑚𝐯…
T\gamma(\mathbf{v})=\left[\ldots,\cos(2\pi\sigma^{j/m}\mathbf{v}),\sin(2\pi\sigma^{j/m}\mathbf{v}),\ldots\right]^{\mathrm{T}} for j=0,…,m−1𝑗

0…𝑚1j=0,\ldots,m-1.
Uses log-linear spaced frequencies for each dimension, where the scale σ𝜎\sigma is chosen for each task and dataset by a hyperparameter sweep. This is a generalization of the “positional encoding” used by prior work [[27](#bib.bib27), [39](#bib.bib39), [44](#bib.bib44)]. Note that this mapping is deterministic and only contains on-axis frequencies, making it naturally biased towards data that has more frequency content along the axes.

Gaussian:
γ​(𝐯)=[cos⁡(2​π​𝐁𝐯),sin⁡(2​π​𝐁𝐯)]T𝛾𝐯superscript2𝜋𝐁𝐯2𝜋𝐁𝐯T\gamma(\mathbf{v})=\left[\cos(2\pi\mathbf{B}\mathbf{v}),\sin(2\pi\mathbf{B}\mathbf{v})\right]^{\mathrm{T}},
where each entry in 𝐁∈ℝm×d𝐁superscriptℝ𝑚𝑑\mathbf{B}\in\mathbb{R}^{m\times d} is sampled from 𝒩​(0,σ2)𝒩0superscript𝜎2\mathcal{N}(0,\sigma^{2}), and σ𝜎\sigma is chosen for each task and dataset with a hyperparameter sweep.
In the absence of any strong prior on the frequency spectrum of the signal, we use an isotropic Gaussian distribution.

Our experiments show that all of the Fourier feature mappings improve the performance of coordinate-based MLPs over using no mapping and that the Gaussian RFF mapping performs best.

### 6.2 Tasks

We conduct experiments with direct regression, where supervision labels are in the same space as the network outputs, as well as indirect regression, where the network outputs are passed through a forward model to produce observations in the same space as the supervision labels (Appendix [D](#A4 "Appendix D Indirect supervision through a linear map ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") contains a theoretical analysis of indirect regression through a linear forward model). For each task and dataset, we tune Fourier feature scales on a held-out set of signals.
For each target signal, we train an MLP on a training subset of the signal and compute error over the remaining test subset.
All tasks (except 3D shape regression) use L2 loss and a ReLU MLP with 4 layers and 256 channels. The 3D shape regression task uses cross-entropy loss and a ReLU MLP with 8 layers and 256 channels. We apply a sigmoid activation to the output for each task (except the view synthesis density prediction). We use 256 frequencies for the feature mapping in all experiments (see Appendix [A.2](#A1.SS2 "A.2 Feature sparsity and network depth ‣ Appendix A Further experiments ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") for experiments that investigate the effects of network depth and feature mapping sparsity). Appendix [E](#A5 "Appendix E Task details ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") provides additional details on each task and our implementations, and Appendix [F](#A6 "Appendix F Additional results figures ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") shows more result figures.

2D image regression.
In this task, we train an MLP to regress from a 2D input pixel coordinate to the corresponding RGB value of an image.
For each test image, we train an MLP on a regularly-spaced grid containing 1/414\nicefrac{{1}}{{4}} of the pixels and report test error on the remaining pixels.
We compare input mappings over a dataset of natural images and a dataset of text images.

3D shape regression.
Occupancy Networks [[24](#bib.bib24)] implicitly represent a 3D shape as the “decision boundary” of an MLP, which is trained to output 0 for points outside the shape and 1 for points inside the shape. Each batch of training data is generated by sampling points uniformly at random from the bounding box of the shape and calculating their labels using the ground truth mesh. Test error is calculated using intersection-over-union versus ground truth on a set of points randomly sampled near the mesh surface to better highlight the different mappings’ abilities to resolve fine details.

2D computed tomography (CT).
In CT, we observe integral projections of a density field instead of direct measurements. In our 2D CT experiments, we train an MLP that takes in a 2D pixel coordinate and predicts the corresponding volume density at that location. The network is indirectly supervised by the loss between a sparse set of ground-truth integral projections and integral projections computed from the network’s output. We conduct experiments using two datasets: procedurally-generated Shepp-Logan phantoms [[36](#bib.bib36)] and 2D brain images from the ATLAS dataset [[21](#bib.bib21)].

3D magnetic resonance imaging (MRI).
In MRI, we observe Fourier transform coefficients of atomic response to radio waves under a magnetic field. In our 3D MRI experiments, we train an MLP that takes in a 3D voxel coordinate and predicts the corresponding response at that location. The network is indirectly supervised by the loss between a sparse set of ground-truth Fourier transform coefficients and Fourier transform coefficients computed from discretely querying the MLP on a voxel grid. We conduct experiments using the ATLAS dataset [[21](#bib.bib21)].

3D inverse rendering for view synthesis.
In view synthesis, we observe 2D photographs of a 3D scene, reconstruct a representation of that scene, then render images from new viewpoints. To perform this task, we train a coordinate-based MLP that takes in a 3D location and outputs a color and volume density. This MLP is indirectly supervised by the loss between the set of 2D image observations and the same viewpoints re-rendered from the predicted scene representation. We use a simplified version of the method described in NeRF [[27](#bib.bib27)], where we remove hierarchical sampling and view dependence and replace the original positional encoding with our compared input mappings.

|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | Direct supervision | | | Indirect supervision | | | |
|  | 2D image | | 3D shape [[24](#bib.bib24)] | 2D CT | | 3D MRI | 3D NeRF [[27](#bib.bib27)] |
|  | Natural | Text |  | Shepp | ATLAS | ATLAS |  |
| No mapping | 19.32 | 18.40 | 0.864 | 16.75 | 15.44 | 26.14 | 22.41 |
| Basic | 21.71 | 20.48 | 0.892 | 23.31 | 16.95 | 28.58 | 23.16 |
| Positional enc. | 24.95 | 27.57 | 0.960 | 26.89 | 19.55 | 32.23 | 25.28 |
| Gaussian | 25.57 | 30.47 | 0.973 | 28.33 | 19.88 | 34.51 | 25.48 |

Table 1: We compare four different input mappings on a variety of low-dimensional regression tasks. All results are reported in PSNR except 3D shape, which uses IoU (higher is better for all).
No mapping represents using a standard MLP with no feature mapping.
Basic, Positional encoding, and Gaussian are different variants of Fourier feature maps.
For the Direct supervision tasks, the network is supervised using ground truth labels for each input coordinate. For the Indirect supervision tasks, the network outputs are passed through a forward model before the loss is applied (integral projection for CT, the Fourier transform for MRI, and nonlinear volume rendering for NeRF).
Fourier feature mappings improve results across all tasks, with random Gaussian features performing best.

## 7 Conclusion

We leverage NTK theory to show that a Fourier feature mapping can make coordinate-based MLPs better suited for modeling functions in low dimensions, thereby overcoming the spectral bias inherent in coordinate-based MLPs. We experimentally show that tuning the Fourier feature parameters offers control over the frequency falloff of the combined NTK and significantly improves performance across a range of graphics and imaging tasks.
These findings shed light on the burgeoning technique of using coordinate-based MLPs to represent 3D shapes in computer vision and graphics pipelines, and provide a simple strategy for practitioners to improve results in these domains.

## Acknowledgements

We thank Ben Recht for advice, and Cecilia Zhang and Tim Brooks for their comments on the text.
BM is funded by a Hertz Foundation Fellowship and acknowledges support from the Google BAIR Commons program.
MT, PS and SFK are funded by NSF Graduate Fellowships.
RR was supported in part by ONR grants N000141712687 and
N000142012529 and the Ronald L. Graham Chair.
RN was supported in part by an FHL Vive Center Seed Grant.
Google University Relations provided a generous donation of compute credits.

## References

* [1]

  Eirikur Agustsson and Radu Timofte.
  NTIRE 2017 challenge on single image super-resolution: Dataset and
  study.
  CVPR Workshops, 2017.
* [2]

  Sanjeev Arora, Simon Du, Wei Hu, Zhiyuan Li, and Ruosong Wang.
  Fine-grained analysis of optimization and generalization for
  overparameterized two-layer neural networks.
  ICML, 2019.
* [3]

  Ronen Basri, Meirav Galun, Amnon Geifman, David Jacobs, Yoni Kasten, and Shira
  Kritchman.
  Frequency bias in neural networks for input of non-uniform density.
  arXiv preprint arXiv:2003.04560, 2020.
* [4]

  Ronen Basri, David Jacobs, Yoni Kasten, and Shira Kritchman.
  The convergence rate of neural networks for learned functions of
  different frequencies.
  NeurIPS, 2019.
* [5]

  Alberto Bietti and Julien Mairal.
  On the inductive bias of neural tangent kernels.
  NeurIPS, 2019.
* [6]

  Blake Bordelon, Abdulkadir Canatar, and Cengiz Pehlevan.
  Spectrum dependent learning curves in kernel regression and wide
  neural networks.
  arXiv preprint arXiv:2002.02561, 2020.
* [7]

  R. N. Bracewell.
  Strip integration in radio astronomy.
  Australian Journal of Physics, 1956.
* [8]

  James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary,
  Dougal Maclaurin, and Skye Wanderman-Milne.
  JAX: composable transformations of Python+NumPy programs,
  2018.
  <http://github.com/google/jax>.
* [9]

  Zhiqin Chen and Hao Zhang.
  Learning implicit fields for generative shape modeling.
  CVPR, 2019.
* [10]

  Boyang Deng, JP Lewis, Timothy Jeruzalski, Gerard Pons-Moll, Geoffrey Hinton,
  Mohammad Norouzi, and Andrea Tagliasacchi.
  Neural articulated shape approximation.
  arXiv preprint arXiv:1912.03207, 2019.
* [11]

  Simon S. Du, Xiyu Zhai, Barnabas Poczos, and Aarti Singh.
  Gradient descent provably optimizes over-parameterized neural
  networks.
  ICLR, 2019.
* [12]

  Kyle Genova, Forrester Cole, Aaron Sarna Daniel Vlasic, William T. Freeman, and
  Thomas Funkhouser.
  Learning shape templates with structured implicit functions.
  ICCV, 2019.
* [13]

  Kyle Genova, Forrester Cole, Avneesh Sud, Aaron Sarna, and Thomas Funkhouser.
  Local deep implicit functions for 3D shape.
  CVPR, 2020.
* [14]

  Reinhard Heckel and Mahdi Soltanolkotabi.
  Compressive sensing with un-trained neural networks: Gradient descent
  finds the smoothest approximation.
  arXiv preprint arXiv:2005.03991, 2020.
* [15]

  Philipp Henzler, Niloy J Mitra, and Tobias Ritschel.
  Learning a neural 3d texture space from 2d exemplars.
  CVPR, 2020.
* [16]

  Arthur Jacot, Franck Gabriel, and Clément Hongler.
  Neural Tangent Kernel: Convergence and generalization in
  neural networks.
  NeurIPS, 2018.
* [17]

  Chiyu Jiang, Avneesh Sud, Ameesh Makadia, Jingwei Huang, Matthias Nießner,
  and Thomas Funkhouser.
  Local implicit grid representations for 3D scenes.
  CVPR, 2020.
* [18]

  Seyed Mehran Kazemi, Rishab Goel, Sepehr Eghbali, Janahan Ramanan, Jaspreet
  Sahota, Sanjay Thakur, Stella Wu, Cathal Smyth, Pascal Poupart, and Marcus
  Brubaker.
  Time2vec: Learning a vector representation of time.
  arXiv preprint arXiv:1907.05321, 2019.
* [19]

  Diederik P. Kingma and Jimmy Ba.
  Adam: A method for stochastic optimization.
  ICLR, 2015.
* [20]

  Jaehoon Lee, Lechao Xiao, Samuel Schoenholz, Yasaman Bahri, Roman Novak, Jascha
  Sohl-Dickstein, and Jeffrey Pennington.
  Wide neural networks of any depth evolve as linear models under
  gradient descent.
  NeurIPS, 2019.
* [21]

  Sook-Lei Liew, Julia M. Anglin, Nick W. Banks, Matt Sondag, Kaori L. Ito, Kim,
  et al.
  A large, open source dataset of stroke anatomical brain images and
  manual lesion segmentations.
  Scientific Data, 2018.
* [22]

  Shaohui Liu, Yinda Zhang, Songyou Peng, Boxin Shi, Marc Pollefeys, and Zhaopeng
  Cui.
  Dist: Rendering deep implicit signed distance function with
  differentiable sphere tracing.
  CVPR, 2020.
* [23]

  Shichen Liu, Shunsuke Saito, Weikai Chen, and Hao Li.
  Learning to infer implicit surfaces without 3D supervision.
  NeurIPS, 2019.
* [24]

  Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and
  Andreas Geiger.
  Occupancy networks: Learning 3D reconstruction in function space.
  CVPR, 2019.
* [25]

  Michael Dawson-Haggerty et al.
  trimesh, 2019.
  <https://trimsh.org/>.
* [26]

  Mateusz Michalkiewicz, Jhony K Pontes, Dominic Jack, Mahsa Baktashmotlagh, and
  Anders Eriksson.
  Implicit surface representations as layers in neural networks.
  ICCV, 2019.
* [27]

  Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi
  Ramamoorthi, and Ren Ng.
  NeRF: Representing scenes as neural radiance fields for view
  synthesis.
  arXiv preprint arXiv:2003.08934, 2020.
* [28]

  Anh Nguyen, Jason Yosinski, and Jeff Clune.
  Deep neural networks are easily fooled: High confidence predictions
  for unrecognizable images.
  CVPR, 2015.
* [29]

  Michael Niemeyer, Lars Mescheder, Michael Oechsle, and Andreas Geiger.
  Differentiable volumetric rendering: Learning implicit 3D
  representations without 3D supervision.
  CVPR, 2020.
* [30]

  Roman Novak, Lechao Xiao, Jiri Hron, Jaehoon Lee, Alexander A. Alemi, Jascha
  Sohl-Dickstein, and Samuel S. Schoenholz.
  Neural tangents: Fast and easy infinite neural networks in Python.
  ICLR, 2020.
* [31]

  Michael Oechsle, Lars Mescheder, Michael Niemeyer, Thilo Strauss, and Andreas
  Geiger.
  Texture fields: Learning texture representations in function space.
  ICCV, 2019.
* [32]

  Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven
  Lovegrove.
  DeepSDF: Learning continuous signed distance functions for shape
  representation.
  CVPR, 2019.
* [33]

  Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred A.
  Hamprecht, Yoshua Bengio, and Aaron Courville.
  On the spectral bias of neural networks.
  ICML, 2019.
* [34]

  Ali Rahimi and Benjamin Recht.
  Random features for large-scale kernel machines.
  NeurIPS, 2007.
* [35]

  Shunsuke Saito, , Zeng Huang, Ryota Natsume, Shigeo Morishima, Angjoo Kanazawa,
  and Hao Li.
  PIFu: Pixel-aligned implicit function for high-resolution clothed
  human digitization.
  ICCV, 2019.
* [36]

  Lawrence A. Shepp and Benjamin F. Logan.
  The Fourier reconstruction of a head section.
  IEEE Transactions on nuclear science, 1974.
* [37]

  Vincent Sitzmann, Michael Zollhoefer, and Gordon Wetzstein.
  Scene representation networks: Continuous 3D-structure-aware neural
  scene representations.
  NeurIPS, 2019.
* [38]

  Kenneth O. Stanley.
  Compositional pattern producing networks: A novel abstraction of
  development.
  Genetic Programming and Evolvable Machines, 2007.
* [39]

  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
  Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.
  Attention is all you need.
  NeurIPS, 2017.
* [40]

  Martin J. Wainwright.
  Reproducing Kernel Hilbert Spaces, page 383–415.
  Cambridge Series in Statistical and Probabilistic Mathematics.
  Cambridge University Press, 2019.
* [41]

  Ingo Wald, Sven Woop, Carsten Benthin, Gregory S Johnson, and Manfred Ernst.
  Embree: a kernel framework for efficient CPU ray tracing.
  ACM Transactions on Graphics (TOG), 2014.
* [42]

  Da Xu, Chuanwei Ruan, Evren Korpeoglu, Sushant Kumar, and Kannan Achan.
  Self-attention with functional time representation learning.
  NeurIPS, 2019.
* [43]

  Greg Yang and Hadi Salman.
  A fine-grained spectral perspective on neural networks.
  arXiv preprint arXiv:1907.10599, 2019.
* [44]

  Ellen D. Zhong, Tristan Bepler, Joseph H. Davis, and Bonnie Berger.
  Reconstructing continuous distributions of 3D protein structure
  from cryo-EM images.
  ICLR, 2020.

## Appendix A Further experiments

### A.1 Optimizing validation error through the NTK linear dynamics

Using Eqn. [3](#S3.E3 "In 3 Background and Notation ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") in the main paper, we can predict what error a trained network will achieve on a set of testing points. Since this equation depends on the composed NTK, we can directly relate predicted test set loss to the Fourier feature mapping parameters a𝑎a and b𝑏b for a validation set of signals 𝐲v​a​lsubscript𝐲𝑣𝑎𝑙\mathbf{y}\_{val}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒopt=∥𝐮(t)−𝐲val∥22≈∥𝐊val​𝐊−1​(𝐈−e−η​𝐊​t)​𝐲−𝐲val∥22,subscriptℒoptsuperscriptsubscriptdelimited-∥∥superscript𝐮𝑡subscript𝐲val22superscriptsubscriptdelimited-∥∥subscript𝐊valsuperscript𝐊1𝐈superscript𝑒𝜂𝐊𝑡𝐲superscriptsubscript𝐲valabsent22\mathcal{L}\_{\mathrm{opt}}=\left\lVert\mathbf{u}^{(t)}-\mathbf{y}\_{\mathrm{val}}\right\rVert\_{2}^{2}\approx\left\lVert\mathbf{K}\_{\mathrm{val}}\mathbf{K}^{-1}\left(\mathbf{I}-e^{-\eta\mathbf{K}t}\right)\mathbf{y}-\mathbf{y}\_{\mathrm{val}}^{\phantom{{}^{(t)}}}\right\rVert\_{2}^{2}, |  | (10) |

where 𝐊valsubscript𝐊val\mathbf{K}\_{\mathrm{val}} is the composed NTK evaluated between points in a validation dataset 𝐗valsubscript𝐗val\mathbf{X}\_{\mathrm{val}} and training dataset 𝐗𝐗\mathbf{X}, and η𝜂\eta and t𝑡t are the learning rate and number of iterations that will be used when training the actual network.

In Figure [5](#A1.F5 "Figure 5 ‣ A.1 Optimizing validation error through the NTK linear dynamics ‣ Appendix A Further experiments ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"), we show the results of minimizing Eqn. [10](#A1.E10 "In A.1 Optimizing validation error through the NTK linear dynamics ‣ Appendix A Further experiments ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") by gradient descent on ajsubscript𝑎𝑗a\_{j} values (with fixed corresponding “densely sampled” bj=jsubscript𝑏𝑗𝑗b\_{j}=j) for validation sets sampled from three different 1/fα1superscript𝑓𝛼1/f^{\alpha} noise families. Note that gradient descent on this theoretical loss approximation produces ajsubscript𝑎𝑗a\_{j} values which are able to perform as well as the best “power law” ajsubscript𝑎𝑗a\_{j} values for each respective signal class (compared dashed lines versus ×\times markers in Figure [5](#A1.F5 "Figure 5 ‣ A.1 Optimizing validation error through the NTK linear dynamics ‣ Appendix A Further experiments ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains")b). As mentioned in the main text, we find that this optimization strategy is only viable for small 1D regression problems. In our multidimensional tasks, using densely sampled 𝐛jsubscript𝐛𝑗\mathbf{b}\_{j} values is not tractable due to memory constraints. In addition, the theoretical approximation only holds when training the network using SGD, and in practice we train using the Adam optimizer [[19](#bib.bib19)].

![Refer to caption](/html/2006.10739/assets/x8.png)


Figure 5: The Fourier feature mappings can be optimized for better performance on a class of target signals by using the linearized network approximation. Here we consider target signals sampled from three different power law distributions. In (a) we show the spectrum for composed kernels corresponding to different optimized feature mappings, where the feature mappings are initialized to match the “Power ∞\infty” distribution. In (b) we take an alternative approach where we sweep over "power law" settings for our Fourier features. We find that tuning this simple parameterization is able to perform on par with the optimized feature maps.

### A.2 Feature sparsity and network depth

In our experiments, we observe that deeper networks need fewer Fourier features than shallow networks. As the depth of the MLP increases, we observe that a sparser set of frequencies can achieve similar performance; Figure [6](#A1.F6.3 "Figure 6 ‣ A.2 Feature sparsity and network depth ‣ Appendix A Further experiments ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") illustrates this effect in the context of 2D image regression.

Again drawing on NTK theory, we understand this tradeoff as an effect of frequency “spreading,” as illustrated in Figure [7](#A1.F7 "Figure 7 ‣ A.2 Feature sparsity and network depth ‣ Appendix A Further experiments ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"). A Fourier featurization consists of only discrete frequencies,
but when composed with the NTK, the influence of each discrete frequency “spreads” over its local neighborhood in the final spectrum. We find that the “spread” around each frequency feature increases for deeper networks.
For an MLP to learn all of the frequency components in the target signal, its corresponding composed NTK must contain adequate power across the frequency support of the target signal. This is accomplished either by including more frequencies in the Fourier features or by spreading those frequencies through sufficient NTK depth.

![Refer to caption](/html/2006.10739/assets/x9.png)

Figure 6: In a 2D image regression task (explained in Section [E.1](#A5.SS1 "E.1 2D image ‣ Appendix E Task details ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains")) we find that shallower networks require more Fourier features than deeper networks. This is explained by the frequency spreading effect shown in Figure [7](#A1.F7 "Figure 7 ‣ A.2 Feature sparsity and network depth ‣ Appendix A Further experiments ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"). In this experiment we use the *Natural* image dataset and a Gaussian mapping. All of the network layers have 256 channels, and the networks are trained using an Adam [[19](#bib.bib19)] optimizer with a learning rate of 10−3superscript10310^{-3}.

![Refer to caption](/html/2006.10739/assets/x10.png)


Figure 7: Each frequency included in a Fourier embedding is “spread” by the NTK, with deeper NTKs causing more frequency spreading. We posit that this frequency spreading is what enables an MLP with a sparse set of Fourier features to faithfully reconstruct a complex signal, which would be poorly reconstructed by either sparse Fourier feature regression or a plain coordinate-based MLP.

### A.3 Gradient descent does not optimize Fourier features

One may wonder if the Fourier feature mapping parameters ajsubscript𝑎𝑗a\_{j} and 𝐛jsubscript𝐛𝑗\mathbf{b}\_{j} can be optimized alongside network weights using gradient descent, which may circumvent the need for careful initialization.
We performed an experiment in which the aj,𝐛j

subscript𝑎𝑗subscript𝐛𝑗a\_{j},\mathbf{b}\_{j} values are treated as trainable variables (along with the weights of the network) and optimize all variables with Adam to minimize training loss.
Figure [8](#A1.F8 "Figure 8 ‣ A.3 Gradient descent does not optimize Fourier features ‣ Appendix A Further experiments ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") shows that jointly optimizing these parameters does not improve performance compared to leaving them fixed.

![Refer to caption](/html/2006.10739/assets/x11.png)


Figure 8: “Training” the Fourier feature mapping parameters ajsubscript𝑎𝑗a\_{j} and 𝐛jsubscript𝐛𝑗\mathbf{b}\_{j} along with the network weights using Adam does not improve performance, as the 𝐛jsubscript𝐛𝑗\mathbf{b}\_{j} values do not deviate significantly from their initial values. We show that this holds when 𝐛jsubscript𝐛𝑗\mathbf{b}\_{j} are initialized at three different scales of Gaussian Fourier features in the case of the 2D image task (ajsubscript𝑎𝑗a\_{j} are always initialized as 111).

### A.4 Visualizing underfitting and overfitting in 2D

Figure [4](#S5.F4 "Figure 4 ‣ 5 Manipulating the Fourier Feature Mapping ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") in the main text shows (in a 1D setting) that as the scale of the Fourier feature sampling distribution increases, the trained network’s error traces out a curve that starts in an underfitting regime (only low frequencies are learned) and ends in an overfitting regime (the learned function includes high-frequency detail not present in the training data). In Figure [9](#A1.F9 "Figure 9 ‣ A.4 Visualizing underfitting and overfitting in 2D ‣ Appendix A Further experiments ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"), we show analogous behavior for 2D image regression, demonstrating that the same phenomenon holds in a multidimensional problem.
In Figure [10](#A1.F10 "Figure 10 ‣ A.4 Visualizing underfitting and overfitting in 2D ‣ Appendix A Further experiments ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"), we show how changing the scale for Gaussian Fourier features qualitatively affects the final result in the 2D image regression task.

![Refer to caption](/html/2006.10739/assets/x12.png)


(a) Test error for 2D image task

![Refer to caption](/html/2006.10739/assets/x13.png)


(b) Train and test error for 2D image task

Figure 9: An alternate version of Figure [4](#S5.F4 "Figure 4 ‣ 5 Manipulating the Fourier Feature Mapping ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") from the main text where the underlying signal is a 2D image (see 2D image task details in Section [E.1](#A5.SS1 "E.1 2D image ‣ Appendix E Task details ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains")) instead of 1D signal.
This multi-dimensional case exhibits the same behavior as was seen in the 1D case: we see the same underfitting/overfitting pattern for four different isotropic Fourier feature distributions, and the distribution shape matters less than the scale of sampled bisubscript𝑏𝑖b\_{i} values.



|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Refer to caption | Refer to caption | Refer to caption | Refer to caption | Refer to caption |
| σ=1𝜎1\sigma=1 | σ=2𝜎2\sigma=2 | σ=10𝜎10\sigma=10 | σ=32𝜎32\sigma=32 | σ=64𝜎64\sigma=64 |

Figure 10: A visualization of the 2D image regression task with different Gaussian scales (corresponding to points along the curve shown in Figure [9](#A1.F9 "Figure 9 ‣ A.4 Visualizing underfitting and overfitting in 2D ‣ Appendix A Further experiments ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains")).
Low values of σ𝜎\sigma underfit, resulting in oversmoothed interpolation, and large values of σ𝜎\sigma overfit, resulting in noisy interpolation. We find that σ=10𝜎10\sigma=10 performs best for our *Natural* image dataset.

### A.5 Failures of positional encoding (axis-aligned bias)

![Refer to caption](/html/2006.10739/assets/x14.png)


Figure 11: We train a coordinate-based MLP to fit target 2D images consisting of simple sinusoids at different frequencies and angles. The positional encoding mapping performs well at on-axis angles and performs worse on off-axis angles, while the Gaussian RFF mapping performs similarly well across all angles (results are averaged over radii). Error bars are plotted over runs with different randomly-sampled frequencies for the Gaussian RFF mapping, while positional encoding is deterministic.

Here we present a simple experiment to directly showcase the benefits of using an isotropic frequency distribution, such as Gaussian RFF, compared to the axis-aligned “positional encoding” used in prior work [[27](#bib.bib27), [44](#bib.bib44)]. As discussed in the main paper, the positional encoding mapping only uses on-axis frequencies. This approach is well-suited to data that has more frequency content along the coordinate axes, but is not as effective for more natural signals.

In Figure [11](#A1.F11 "Figure 11 ‣ A.5 Failures of positional encoding (axis-aligned bias) ‣ Appendix A Further experiments ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"), we conduct a simple 2D image experiment where we train a coordinate-based MLP (2 layers, 256 channels) to fit target 2D sinusoid images (512×512512512512\times 512 resolution). We sample 64 such 2D sinusoid images (regularly-sampled in polar coordinates, with 16 angles and 4 radii) and train a 2D coordinate-based MLP to fit each, using the same setup as the 2D image experiments described in Section [E.1](#A5.SS1 "E.1 2D image ‣ Appendix E Task details ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"). The isotropic Gaussian RFF mapping performs well across all angles, while the positional encoding mapping performs worse for frequencies that are not axis-aligned.

## Appendix B Additional details for main text figures

### B.1 Main text Figure [3](#S5.F3 "Figure 3 ‣ 5 Manipulating the Fourier Feature Mapping ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") (effect of feature mapping on convergence speed)

In Figure [12](#A2.F12 "Figure 12 ‣ B.1 Main text Figure 3 (effect of feature mapping on convergence speed) ‣ Appendix B Additional details for main text figures ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"), we present an alternate version of Figure 3 from the main text showing a denser sampling of p𝑝p values to better visualize the effect of changing Fourier feature falloff on the resulting trained network. Again, the feature mapping used here is aj=1/jp,bj=jformulae-sequencesubscript𝑎𝑗1superscript𝑗𝑝subscript𝑏𝑗𝑗a\_{j}=1/j^{p},b\_{j}=j for j=1,…,n/2𝑗

1…𝑛2j=1,\ldots,n/2.

![Refer to caption](/html/2006.10739/assets/x15.png)


Figure 12: An extension of Figure [3](#S5.F3 "Figure 3 ‣ 5 Manipulating the Fourier Feature Mapping ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") from the main paper, showing more values of p𝑝p. In (c) we see that mappings with more gradual frequency falloff (lower p𝑝p) converge significantly faster in mid and high frequencies, resulting in faster overall training convergence (d). In (b) we see that p=1𝑝1p=1 achieves a lower test error than the other mappings.

### B.2 Main text Figure [4](#S5.F4 "Figure 4 ‣ 5 Manipulating the Fourier Feature Mapping ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") (different random feature distributions in 1D)

Exact details for the sampling distributions used to generate bjsubscript𝑏𝑗b\_{j} values for Figure [4](#S5.F4 "Figure 4 ‣ 5 Manipulating the Fourier Feature Mapping ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") in the main text are shown in Table [2](#A2.T2 "Table 2 ‣ Uniform log distribution ‣ B.2 Main text Figure 4 (different random feature distributions in 1D) ‣ Appendix B Additional details for main text figures ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"). In Figure [13](#A2.F13 "Figure 13 ‣ Uniform log distribution ‣ B.2 Main text Figure 4 (different random feature distributions in 1D) ‣ Appendix B Additional details for main text figures ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"), we present an alternate version showing both train and test performance, emphasizing the underfitting/overfitting regimes created by manipulating the scale of the Fourier features.

#### Uniform log distribution

We include the *Uniform log* distribution because it is the random equivalent of the “positional encoding” sometimes used in prior work. One observation is that the sampling for uniform-log variables (X′=σu​lXsuperscript𝑋′superscriptsubscript𝜎𝑢𝑙𝑋X^{\prime}=\sigma\_{ul}^{X} where X∼𝒰​[0,1)similar-to𝑋𝒰01X\sim\mathcal{U}[0,1)) corresponds to the following CDF:

|  |  |  |  |
| --- | --- | --- | --- |
|  | P​(X′≤x)=log⁡xlog⁡σu​l,for ​x∈[1,σu​l),formulae-sequence𝑃superscript𝑋′𝑥𝑥subscript𝜎𝑢𝑙for 𝑥1subscript𝜎𝑢𝑙\displaystyle P(X^{\prime}\leq x)=\frac{\log x}{\log\sigma\_{ul}},\quad\textrm{for }x\in[1,\sigma\_{ul})\,, |  | (11) |

which has the following PDF:

|  |  |  |  |
| --- | --- | --- | --- |
|  | p​(x)=dd​x​P​(X′≤x)=1x​log⁡σu​l.𝑝𝑥𝑑𝑑𝑥𝑃superscript𝑋′𝑥1𝑥subscript𝜎𝑢𝑙\displaystyle p(x)=\frac{d}{dx}P(X^{\prime}\leq x)=\frac{1}{x\log\sigma\_{ul}}\,. |  | (12) |

This shows that the randomized equivalent of positional encoding is sampling from a distribution proportional to a 1/f1𝑓1/f falloff power law.

|  |  |
| --- | --- |
| Name | Sampled bjsubscript𝑏𝑗b\_{j} values |
| Gaussian | σg​Xsubscript𝜎𝑔𝑋\sigma\_{g}X for X∼𝒩​(0,1)similar-to𝑋𝒩01X\sim\mathcal{N}(0,1) |
| Uniform | σu​Xsubscript𝜎𝑢𝑋\sigma\_{u}X for X∼𝒰​[0,1)similar-to𝑋𝒰01X\sim\mathcal{U}[0,1) |
| Uniform log | σu​lXsuperscriptsubscript𝜎𝑢𝑙𝑋\sigma\_{ul}^{X} for X∼𝒰​[0,1)similar-to𝑋𝒰01X\sim\mathcal{U}[0,1) |
| Laplacian | σl​Xsubscript𝜎𝑙𝑋\sigma\_{l}X for X∼Laplace​(0,1)similar-to𝑋Laplace01X\sim\mathrm{Laplace}(0,1) |
| Positional Enc. | 2σp​Xsuperscript2subscript𝜎𝑝𝑋2^{\sigma\_{p}X} for X∈linspace​(0,1)𝑋linspace01X\in\mathrm{linspace}(0,1) (deterministic) |

Table 2: Different distributions used for sampling frequencies, where σ𝜎\sigma is each distribution’s “scale”.

![Refer to caption](/html/2006.10739/assets/x16.png)


Figure 13: An alternate version of Figure [4](#S5.F4 "Figure 4 ‣ 5 Manipulating the Fourier Feature Mapping ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") from the main text showing both training error and test error for a variety of different Fourier feature sampling distributions. Adding training error to the plot clearly distinguishes between the underfitting regime with low frequency bisubscript𝑏𝑖b\_{i} (where train and test error are similar) versus the overfitting regime with high frequency bisubscript𝑏𝑖b\_{i} (where the test error increases but training error approaches machine precision).

## Appendix C Stationary kernels

One of the primary benefits of our Fourier feature mapping is that it results in a *stationary* composed NTK function. In this section, we offer some intuition for why stationarity is desirable for our low-dimensional graphics and imaging problems.

First, let us consider the implications of using an MLP applied directly to a low-dimensional input (without any Fourier feature mapping). In this setting, the NTK is a function of the dot product between its inputs and of their norms [[3](#bib.bib3), [5](#bib.bib5), [6](#bib.bib6), [16](#bib.bib16)].
This makes the NTK *rotation*-invariant, but not *translation*-invariant. For our graphics and imaging applications, we want to be able to model an object or scene equally well regardless of its location, so translation-invariance or *stationarity* is a crucial property. We can then add approximate rotation invariance back by using an isotropic frequency sampling distribution.

This aligns with standard practice in signal processing, in which k​(𝐮,𝐯)=h~​(𝐮−𝐯)=h~​(𝐯−𝐮)𝑘𝐮𝐯~ℎ𝐮𝐯~ℎ𝐯𝐮k(\mathbf{u},\mathbf{v})=\tilde{h}(\mathbf{u}-\mathbf{v})=\tilde{h}(\mathbf{v}-\mathbf{u}) (e.g. the Gaussian or radial basis function kernel, or the sinc reconstruction filter kernel). This Euclidean notion of similarity based on difference vectors is better suited to the low-dimensional regime, in which we expect (and can afford) dense and nearly uniform sampling. Regression with a stationary kernel corresponds to reconstruction with a convolution filter: new predictions are sums of training points, weighted by a function of Euclidean distance.

One of the most important features of our sinusoidal input mapping is that it translates between these two regimes. If 𝐮,𝐯∈ℝd

𝐮𝐯
superscriptℝ𝑑\mathbf{u},\mathbf{v}\in\mathbb{R}^{d} for small d𝑑d, γ𝛾\gamma is our Fourier feature embedding function, and k𝑘k is a dot product kernel function, then k​(γ​(𝐮),γ​(𝐯))=h​(γ​(𝐮)T​γ​(𝐯))=h~​(𝐮−𝐯)𝑘𝛾𝐮𝛾𝐯ℎ𝛾superscript𝐮T𝛾𝐯~ℎ𝐮𝐯k(\gamma(\mathbf{u}),\gamma(\mathbf{v}))=h(\gamma(\mathbf{u})^{\mathrm{T}}\gamma(\mathbf{v}))=\tilde{h}(\mathbf{u}-\mathbf{v}). In words, our sinusoidal input mapping transforms a dot product kernel into a stationary one, making it better suited to the low-dimensional regime.

This effect is illustrated in a simple 1D example in Figure [14](#A3.F14 "Figure 14 ‣ Appendix C Stationary kernels ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"), which shows that the benefits of a stationary composed NTK indeed appear in the MLP setting with a basic Fourier featurization (using a single frequency). We train MLPs with and without this basic Fourier embedding to learn a set of shifted 1D Gaussian probability density functions. The plain MLP successfully fits a zero-centered function but struggles to fit shifted functions, while the MLP with basic Fourier embedding exhibits stationary behavior, with good performance regardless of shifts.

![Refer to caption](/html/2006.10739/assets/x17.png)


Figure 14: A plain coordinate-based MLP can learn a centered function (in this case a Gaussian density) but struggles to model shifts of the same function. Adding a basic Fourier embedding (with a single frequency) enables the MLP to fit the target function equally well regardless of shifts. The NTK corresponding to the plain MLP is based on dot products between inputs, whereas the NTK corresponding to the NTK with Fourier embedding is based on Euclidean distances between inputs, making it shift-invariant. In this experiment we train an MLP (4 layers, 256 channels, ReLU activation) for 500 iterations using the Adam [[19](#bib.bib19)] optimizer with a learning rate of 10−4superscript10410^{-4}. We report mean and standard deviation performance over 20 random network initializations.

## Appendix D Indirect supervision through a linear map

In some of the tasks we explore in this work, such as image regression or 3D shape regression, optimization is performed by minimizing a loss between the output of a network and a directly observed quantity, such as the color of a pixel or the occupancy of a voxel.
But in many graphics and imaging applications of interest, measurements are *indirect*, and the loss must be computed on the output of a network after it has been processed by some physical forward model. In NeRF [[27](#bib.bib27)], measurements are taken by sampling and compositing along rays in each viewing direction. In MRI, measurements are taken along various curves through the frequency domain. In CT, measurements are integral projections of the subject at various angles, which correspond to measuring lines through the origin in the frequency domain.
Although the measurement transformation for NeRF is nonlinear (in density, although it is linear in color), those for both CT and MRI are linear.
In this section, we extend the linearized training dynamics of Lee et al. [[20](#bib.bib20)] to the setting of training through a linear operator denoted by a matrix 𝐀𝐀\mathbf{A}. This allows us to modify Eqn. 3  to incorporate 𝐀𝐀\mathbf{A}, thereby demonstrating that the conclusions drawn in this work for the “direct” regression case also apply to the “indirect” case.

Our derivation closely follows Lee et al. [[20](#bib.bib20)], and begins by replacing the neural network f𝑓f with its linearization around the initial parameters θ0subscript𝜃0\theta\_{0}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ftlin​(𝐱)≜f0​(𝐱)+∇θf0​(𝐱)|θ=θ0​ωt,≜superscriptsubscript𝑓𝑡lin𝐱subscript𝑓0𝐱evaluated-atsubscript∇𝜃subscript𝑓0𝐱𝜃subscript𝜃0subscript𝜔𝑡f\_{t}^{\mathrm{lin}}(\mathbf{x})\triangleq f\_{0}(\mathbf{x})+\nabla\_{\theta}f\_{0}(\mathbf{x})|\_{\theta=\theta\_{0}}\omega\_{t}\,, |  | (13) |

where ωt≜θt−θ0≜subscript𝜔𝑡subscript𝜃𝑡subscript𝜃0\omega\_{t}\triangleq\theta\_{t}-\theta\_{0} denotes the change in network parameters since initialization and t𝑡t denotes time in continuous-time gradient flow dynamics. Then [[20](#bib.bib20)] describes the dynamics of gradient flow:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | f˙tlin​(𝐱)superscriptsubscript˙𝑓𝑡lin𝐱\displaystyle\dot{f}\_{t}^{\mathrm{lin}}(\mathbf{x}) | =−η​Θ^0​(𝐱,𝐗)​∇ftlin​(𝐗)ℒ,absent𝜂subscript^Θ0𝐱𝐗subscript∇superscriptsubscript𝑓𝑡lin𝐗ℒ\displaystyle=-\eta\hat{\Theta}\_{0}(\mathbf{x},\mathbf{X})\nabla\_{f\_{t}^{\mathrm{lin}}(\mathbf{X})}\mathcal{L}\,, |  | (14) |

where Θ^t​(⋅,⋅)=∇θft​(⋅)​∇θft​(⋅)Tsubscript^Θ𝑡⋅⋅subscript∇𝜃subscript𝑓𝑡⋅subscript∇𝜃subscript𝑓𝑡superscript⋅T\hat{\Theta}\_{t}(\cdot,\cdot)=\nabla\_{\theta}f\_{t}(\cdot)\nabla\_{\theta}f\_{t}(\cdot)^{\mathrm{T}} is the NTK matrix at time t𝑡t (Θ^tsubscript^Θ𝑡\hat{\Theta}\_{t} is shorthand for Θ^t​(𝐗,𝐗)subscript^Θ𝑡𝐗𝐗\hat{\Theta}\_{t}(\mathbf{X},\mathbf{X})) and ℒℒ\mathcal{L} is the training loss.
At this point, we depart slightly from the analysis of [[20](#bib.bib20)]: instead of ℒ=∑(𝐱,y)∈𝒟ℓ​(ftlin​(𝐱),y)ℒsubscript𝐱𝑦𝒟ℓsuperscriptsubscript𝑓𝑡lin𝐱𝑦\mathcal{L}=\sum\_{(\mathbf{x},y)\in\mathcal{D}}\ell(f\_{t}^{\mathrm{lin}}(\mathbf{x}),y) we have ℒ=12​∥𝐀​(ftlin​(𝐗)−𝐲)∥22ℒ12superscriptsubscriptdelimited-∥∥𝐀superscriptsubscript𝑓𝑡lin𝐗𝐲22\mathcal{L}=\frac{1}{2}\left\lVert\mathbf{A}(f\_{t}^{\mathrm{lin}}(\mathbf{X})-\mathbf{y})\right\rVert\_{2}^{2}, where 𝐲𝐲\mathbf{y} denotes the vector of training labels. The gradient of the loss is then

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∇ftlin​(𝐗)ℒsubscript∇superscriptsubscript𝑓𝑡lin𝐗ℒ\displaystyle\nabla\_{f\_{t}^{\mathrm{lin}}(\mathbf{X})}\mathcal{L} | =∇ftlin​(𝐗)12​∥𝐀​(ftlin​(𝐗)−𝐲)∥22absentsubscript∇superscriptsubscript𝑓𝑡lin𝐗12superscriptsubscriptdelimited-∥∥𝐀superscriptsubscript𝑓𝑡lin𝐗𝐲22\displaystyle=\nabla\_{f\_{t}^{\mathrm{lin}}(\mathbf{X})}\frac{1}{2}\left\lVert\mathbf{A}\left(f\_{t}^{\mathrm{lin}}(\mathbf{X})-\mathbf{y}\right)\right\rVert\_{2}^{2}\, |  | (15) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =𝐀T​𝐀​(ftlin​(𝐗)−𝐲).absentsuperscript𝐀T𝐀superscriptsubscript𝑓𝑡lin𝐗𝐲\displaystyle=\mathbf{A}^{\mathrm{T}}\mathbf{A}\left(f\_{t}^{\mathrm{lin}}(\mathbf{X})-\mathbf{y}\right)\,. |  | (16) |

Substituting this into the gradient flow dynamics of Eqn. [14](#A4.E14 "In Appendix D Indirect supervision through a linear map ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") gives us:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | f˙tlin​(𝐱)superscriptsubscript˙𝑓𝑡lin𝐱\displaystyle\dot{f}\_{t}^{\mathrm{lin}}(\mathbf{x}) | =−η​Θ^0​(𝐱,𝐗)​𝐀T​𝐀​(ftlin​(𝐗)−𝐲),absent𝜂subscript^Θ0𝐱𝐗superscript𝐀T𝐀superscriptsubscript𝑓𝑡lin𝐗𝐲\displaystyle=-\eta\hat{\Theta}\_{0}(\mathbf{x},\mathbf{X})\mathbf{A}^{\mathrm{T}}\mathbf{A}\left(f\_{t}^{\mathrm{lin}}(\mathbf{X})-\mathbf{y}\right)\,, |  | (17) |

with corresponding solution:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ftlin​(𝐗)superscriptsubscript𝑓𝑡lin𝐗\displaystyle f\_{t}^{\mathrm{lin}}(\mathbf{X}) | =(𝐈−e−η​Θ^0​𝐀T​𝐀​t)​𝐲+e−η​Θ^0​𝐀T​𝐀​t​f0​(𝐗).absent𝐈superscript𝑒𝜂subscript^Θ0superscript𝐀T𝐀𝑡𝐲superscript𝑒𝜂subscript^Θ0superscript𝐀T𝐀𝑡subscript𝑓0𝐗\displaystyle=\left(\mathbf{I}-e^{-\eta\hat{\Theta}\_{0}\mathbf{A}^{\mathrm{T}}\mathbf{A}t}\right)\mathbf{y}+e^{-\eta\hat{\Theta}\_{0}\mathbf{A}^{\mathrm{T}}\mathbf{A}t}f\_{0}(\mathbf{X})\,. |  | (18) |

Finally, again following [[20](#bib.bib20)], we can decompose ftlin​(𝐱)=μt​(𝐱)+γt​(𝐱)superscriptsubscript𝑓𝑡lin𝐱subscript𝜇𝑡𝐱subscript𝛾𝑡𝐱f\_{t}^{\mathrm{lin}}(\mathbf{x})=\mu\_{t}(\mathbf{x})+\gamma\_{t}(\mathbf{x}) at any test point 𝐱𝐱\mathbf{x}, where

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | μt​(𝐱)subscript𝜇𝑡𝐱\displaystyle\mu\_{t}(\mathbf{x}) | =Θ^0​(𝐱,𝐗)​Θ^0−1​(𝐈−e−η​Θ^0​𝐀T​𝐀​t)​𝐲,absentsubscript^Θ0𝐱𝐗superscriptsubscript^Θ01𝐈superscript𝑒𝜂subscript^Θ0superscript𝐀T𝐀𝑡𝐲\displaystyle=\hat{\Theta}\_{0}(\mathbf{x},\mathbf{X})\hat{\Theta}\_{0}^{-1}\left(\mathbf{I}-e^{-\eta\hat{\Theta}\_{0}\mathbf{A}^{\mathrm{T}}\mathbf{A}t}\right)\mathbf{y}\,, |  | (19) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | γt​(𝐱)subscript𝛾𝑡𝐱\displaystyle\gamma\_{t}(\mathbf{x}) | =f0​(𝐱)−Θ^0​(𝐱,𝐗)​Θ^0−1​(𝐈−e−η​Θ^0​𝐀T​𝐀​t)​f0​(𝐗).absentsubscript𝑓0𝐱subscript^Θ0𝐱𝐗superscriptsubscript^Θ01𝐈superscript𝑒𝜂subscript^Θ0superscript𝐀T𝐀𝑡subscript𝑓0𝐗\displaystyle=f\_{0}(\mathbf{x})-\hat{\Theta}\_{0}(\mathbf{x},\mathbf{X})\hat{\Theta}\_{0}^{-1}\left(\mathbf{I}-e^{-\eta\hat{\Theta}\_{0}\mathbf{A}^{\mathrm{T}}\mathbf{A}t}\right)f\_{0}(\mathbf{X})\,. |  | (20) |

Assuming our initialization is small, i.e., f0​(𝐱)≈0​∀𝐱subscript𝑓0𝐱0for-all𝐱f\_{0}(\mathbf{x})\approx 0~{}\forall\mathbf{x}, we can write our approximate linearized network output as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ftlin​(𝐱)≈Θ^0​(𝐱,𝐗)​Θ^0−1​(𝐈−e−η​Θ^0​𝐀T​𝐀​t)​𝐲.superscriptsubscript𝑓𝑡lin𝐱subscript^Θ0𝐱𝐗superscriptsubscript^Θ01𝐈superscript𝑒𝜂subscript^Θ0superscript𝐀T𝐀𝑡𝐲f\_{t}^{\mathrm{lin}}(\mathbf{x})\approx\hat{\Theta}\_{0}(\mathbf{x},\mathbf{X})\hat{\Theta}\_{0}^{-1}\left(\mathbf{I}-e^{-\eta\hat{\Theta}\_{0}\mathbf{A}^{\mathrm{T}}\mathbf{A}t}\right)\mathbf{y}\,. |  | (21) |

In our previous analysis, we work instead with the expected or infinite-width NTK matrix 𝐊𝐊\mathbf{K}, which is fixed throughout training. Using this notation, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐲^(t)≈ftlin​(𝐗test)≈𝐊test​𝐊−1​(𝐈−e−η​𝐊𝐀T​𝐀​t)​𝐲.superscript^𝐲𝑡superscriptsubscript𝑓𝑡linsubscript𝐗testsubscript𝐊testsuperscript𝐊1𝐈superscript𝑒𝜂superscript𝐊𝐀T𝐀𝑡𝐲\hat{\mathbf{y}}^{(t)}\approx f\_{t}^{\mathrm{lin}}(\mathbf{X}\_{\mathrm{test}})\approx\mathbf{K}\_{\mathrm{test}}\mathbf{K}^{-1}\left(\mathbf{I}-e^{-\eta\mathbf{K}\mathbf{A}^{\mathrm{T}}\mathbf{A}t}\right)\mathbf{y}\,. |  | (22) |

This is nearly identical to Eqn. 3in the main paper, except that the convergence is governed by the spectrum of 𝐊𝐀T​𝐀superscript𝐊𝐀T𝐀\mathbf{K}\mathbf{A}^{\mathrm{T}}\mathbf{A} rather than 𝐊𝐊\mathbf{K} alone. If 𝐀𝐀\mathbf{A} is unitary, such as the Fourier transform matrix used in (densely sampled) MRI, then training should behave exactly as if we were training on direct measurements. However, if 𝐀𝐀\mathbf{A} is not full rank, then training will only affect the components with nonzero eigenvalues in 𝐊𝐀T​𝐀superscript𝐊𝐀T𝐀\mathbf{K}\mathbf{A}^{\mathrm{T}}\mathbf{A}. In this more common scenario, we want to design a kernel that will provide large eigenvalues in the components that 𝐀𝐀\mathbf{A} can represent, so that the learnable components will converge quickly, and provide reasonable priors for the components we cannot learn.

In our two tasks that supervise through a linear map, CT and MRI, the 𝐀T​𝐀superscript𝐀T𝐀\mathbf{A}^{\mathrm{T}}\mathbf{A} has a structure that illuminates how the linear map interacts with the composed NTK.
The 𝐀T​𝐀superscript𝐀T𝐀\mathbf{A}^{\mathrm{T}}\mathbf{A} matrices for both these tasks are diagonalizable by the DFT matrix, where the diagonal entries are simply the number of times the corresponding frequency is measured by the MRI or CT sampling patterns. This follows from the fact that CT and MRI measurements can both be formulated as Fourier space sampling: CT samples rotated slices in Fourier space through the origin [[7](#bib.bib7)] and MRI samples operator-chosen Fourier trajectories. This means that frequencies not observed by the MRI or CT sampling patterns will never be supervised during training. Therefore, it is crucial to choose a Fourier feature mapping that results in a composed NTK with a good prior on these frequencies.

## Appendix E Task details

We present additional details for each task from Section [6](#S6 "6 Experiments ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") in the main text, including training parameters, forward models, datasets, etc. All experiments are implemented using JAX [[8](#bib.bib8)] and trained on a single K80 or RTX2080Ti GPU. Training a single MLP took between 10 seconds (for the 2D image task) and 30 minutes (for the inverse rendering task).

### E.1 2D image

The 2D image regression tasks presented in the main text all use 512×512512512512\times 512 resolution images. A subsampled grid of 256×256256256256\times 256 pixels is used as training data, and an offset grid of 256×256256256256\times 256 pixels is used for testing. We use two image datasets: *Natural* and *Text*, each consisting of 32 images. The *Natural* images are generated by taking center crops of randomly sampled images from the Div2K dataset [[1](#bib.bib1)]. The *Text* images are generated by placing random strings of text with random sizes and colors on a white background (examples can be seen in Figure [15](#A6.F15 "Figure 15 ‣ Appendix F Additional results figures ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains")). For each dataset we perform a hyperparameter sweep over feature mapping scales on 16 images. We find that scales σg=10subscript𝜎𝑔10\sigma\_{g}=10 and σp=6subscript𝜎𝑝6\sigma\_{p}=6 work best for the *Natural* dataset and σg=14subscript𝜎𝑔14\sigma\_{g}=14 and σp=5subscript𝜎𝑝5\sigma\_{p}=5 work best for the *Text* dataset (see Table [2](#A2.T2 "Table 2 ‣ Uniform log distribution ‣ B.2 Main text Figure 4 (different random feature distributions in 1D) ‣ Appendix B Additional details for main text figures ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") for mapping definitions). In Table [3](#A5.T3 "Table 3 ‣ E.1 2D image ‣ Appendix E Task details ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"), we report model performance using the optimal mapping scale on the remaining 16 images.

|  |  |  |
| --- | --- | --- |
|  | Natural | Text |
| No mapping | 19.32±2.48plus-or-minus19.322.4819.32\pm 2.48 | 18.40±2.23plus-or-minus18.402.2318.40\pm 2.23 |
| Basic | 21.71±2.71plus-or-minus21.712.7121.71\pm 2.71 | 20.48±1.96plus-or-minus20.481.9620.48\pm 1.96 |
| Positional enc. | 24.95±3.72plus-or-minus24.953.7224.95\pm 3.72 | 27.57±3.07plus-or-minus27.573.0727.57\pm 3.07 |
| Gaussian | 25.57±4.19plus-or-minus25.574.19\mathbf{25.57\pm 4.19} | 30.47±2.11plus-or-minus30.472.11\mathbf{30.47\pm 2.11} |

Table 3: 2D image results (mean ±plus-or-minus\pm standard deviation of PSNR)

Each model (MLP with 4 layers, 256 channels, ReLU activation, sigmoid output) is trained for 2000 iterations using the Adam [[19](#bib.bib19)] optimizer with default settings (β1=0.9subscript𝛽10.9\beta\_{1}=0.9, β2=0.999subscript𝛽20.999\beta\_{2}=0.999, ϵ=10−8italic-ϵsuperscript108\epsilon=10^{-8}). Learning rates are manually tuned for each dataset and method. For *Natural* images a learning rate of 10−3superscript10310^{-3} is used for the Gaussian RFF and the positional encoding, and a learning rate of 10−2superscript10210^{-2} is used for the basic mapping and “no mapping” methods. For the *Text* images a learning rate of 10−3superscript10310^{-3} is used for all methods.

### E.2 3D shape

We evaluate the 3D shape regression task (similar to Occupancy Networks [[24](#bib.bib24)]) on four complex triangle meshes commonly used in computer graphics applications (*Dragon*, *Armadillo*, *Buddha*, and *Lucy*, shown in Figure [16](#A6.F16 "Figure 16 ‣ Appendix F Additional results figures ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains")), each containing hundreds of thousands of vertices. We train one coordinate-based MLP network to represent a single mesh rather than trying to generalize one network to encode multiple objects, since our goal is to demonstrate that a network with no mapping or the low frequency “basic” mapping cannot accurately represent even a *single* shape, let alone a whole class of objects.

We use a network with 8 layers of 256 channels each and a ReLU nonlinearity between each layer. We apply a sigmoid activation to the output. Our batch size is 323superscript32332^{3} points, and we use the Adam optimizer [[19](#bib.bib19)] with a learning rate starting at 5×10−45superscript1045\times 10^{-4} and exponentially decaying by a factor of 0.010.010.01 over the course of 10000 total training iterations. At each training iteration, we sample a batch of 3D points uniformly at random from the bounding box of the mesh, and then calculate ground truth labels (using the point-in-mesh method implemented in the Trimesh library [[25](#bib.bib25)], which relies on the Embree kernel for acceleration [[41](#bib.bib41)]). We use cross-entropy loss to train the network to match these classification labels (0 for points outside the mesh, 1 for points inside).

The meshes are scaled to fit inside the unit cube [0,1]3superscript013[0,1]^{3} such that the centroid of the mesh is (0.5,0.5,0.5)0.50.50.5(0.5,0.5,0.5). We use the *Lucy* statue mesh as a validation object to find optimal scale values for the positional encoding and Gaussian feature mapping. As described in the caption for Table [4](#A5.T4 "Table 4 ‣ E.2 3D shape ‣ Appendix E Task details ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"), we calculate error on both a uniformly random test set and a test set that is close to the mesh surface (randomly chosen mesh vertices that have been perturbed by a random Gaussian vector with standard deviation 0.010.010.01) in order to illustrate that Fourier feature mappings provide a large benefit in resolving fine surface details. Both test sets have 643superscript64364^{3} points.

|  |  |  |
| --- | --- | --- |
|  | Uniform points | Boundary points |
| No mapping | 0.959±0.006plus-or-minus0.9590.0060.959\pm 0.006 | 0.864±0.014plus-or-minus0.8640.0140.864\pm 0.014 |
| Basic | 0.966±0.007plus-or-minus0.9660.0070.966\pm 0.007 | 0.892±0.017plus-or-minus0.8920.0170.892\pm 0.017 |
| Positional enc. | 0.987±0.005plus-or-minus0.9870.0050.987\pm 0.005 | 0.960±0.011plus-or-minus0.9600.0110.960\pm 0.011 |
| Gaussian | 0.988±0.007plus-or-minus0.9880.007\mathbf{0.988\pm 0.007} | 0.973±0.010plus-or-minus0.9730.010\mathbf{0.973\pm 0.010} |

Table 4: 3D shape results (mean ±plus-or-minus\pm standard deviation of intersection-over-union). *Uniform points* is an “easy” test set where points are sampled uniformly at random from the bounding box of the ground truth mesh, while *Boundary points* is a “hard” test set where points are sampled near the boundary of the ground truth mesh.

In Figure [16](#A6.F16 "Figure 16 ‣ Appendix F Additional results figures ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"), we visualize additional results on all four meshes mentioned above (including the validation mesh *Lucy*). We render normal maps, which are computed by taking the cross product of the numerical horizontal and vertical derivatives of the depth map. The original depth map is generated by intersecting camera rays with the first 0.50.50.5 isosurface of the network. We select the Fourier feature scales for (d) and (e) by doing a hyperparameter search based on validation loss for the *Lucy* mesh in the last row and report test loss over the other three meshes (Table [4](#A5.T4 "Table 4 ‣ E.2 3D shape ‣ Appendix E Task details ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains")). Note that the weights for each trained MLP are only 2MB, while the triangle mesh files for the objects shown are 61MB, 7MB, 79MB, and 32MB respectively.

### E.3 2D CT

In computed tomography (CT), we observe measurements that are integral projections (integrals along parallel lines) of a density field.
We construct a 2D CT task by using ground truth 512×512512512512\times 512 resolution images, and computing 20 synthetic integral projections at evenly-spaced angles. For each of these images, the supervision data is the set of integral projections, and the test PSNR is evaluated over the original image.

We use two datasets for our 2D CT task: randomized Shepp-Logan phantoms [[36](#bib.bib36)], and the ATLAS brain dataset [[21](#bib.bib21)].
For each dataset, we perform a hyperparameter sweep over mapping scales on 8 examples. We found that scales σg=4subscript𝜎𝑔4\sigma\_{g}=4 and σp=3subscript𝜎𝑝3\sigma\_{p}=3 work best for the *Shepp* dataset and σg=5subscript𝜎𝑔5\sigma\_{g}=5 and σp=5subscript𝜎𝑝5\sigma\_{p}=5 work best for the *ATLAS* dataset. In Table [5](#A5.T5 "Table 5 ‣ E.3 2D CT ‣ Appendix E Task details ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"), we report model performance using the optimal mapping scale on a distinct set of 8 images.

|  |  |  |
| --- | --- | --- |
|  | Shepp | ATLAS |
| No mapping | 16.75±3.64plus-or-minus16.753.6416.75\pm 3.64 | 15.44±1.28plus-or-minus15.441.2815.44\pm 1.28 |
| Basic | 23.31±4.66plus-or-minus23.314.6623.31\pm 4.66 | 16.95±0.72plus-or-minus16.950.7216.95\pm 0.72 |
| Positional enc. | 26.89±1.46plus-or-minus26.891.4626.89\pm 1.46 | 19.55±1.09plus-or-minus19.551.0919.55\pm 1.09 |
| Gaussian | 28.33±1.15plus-or-minus28.331.15\mathbf{28.33\pm 1.15} | 19.88±1.23plus-or-minus19.881.23\mathbf{19.88\pm 1.23} |

Table 5: 2D CT results (mean ±plus-or-minus\pm standard deviation of PSNR).

Each model (MLP with 4 layers, 256 channels, ReLU activation, sigmoid output) is trained for 1000 iterations using the Adam [[19](#bib.bib19)] optimizer with default settings (β1=0.9subscript𝛽10.9\beta\_{1}=0.9, β2=0.999subscript𝛽20.999\beta\_{2}=0.999, ϵ=10−8italic-ϵsuperscript108\epsilon=10^{-8}). The learning rate is manually tuned for each method. Gaussian RFF and positional encoding use a learning rate of 10−3superscript10310^{-3}, and the basic and “no mapping” method use a learning rate of 10−2superscript10210^{-2}.

### E.4 3D MRI

In magnetic resonance imaging (MRI), we observe measurements that are Fourier coefficients of the atomic response to radio waves under a magnetic field.
We construct a toy 3D MRI task by using ground truth 96×96×9696969696\times 96\times 96 resolution volumes and randomly sampling ∼13%similar-toabsentpercent13\sim\!13\% of the Fourier coefficients for each volume from an isotropic Gaussian. For each of these volumes, the supervision data is the set of sampled Fourier coefficients, and the test PSNR is evaluated over the original volume.

We use the ATLAS brain dataset [[21](#bib.bib21)] for our 3D MRI experiments.
We perform a hyperparameter sweep over mapping scales on 6 examples. We find that scales σg=5subscript𝜎𝑔5\sigma\_{g}=5 and σp=4subscript𝜎𝑝4\sigma\_{p}=4 perform best. In Table [6](#A5.T6 "Table 6 ‣ E.4 3D MRI ‣ Appendix E Task details ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"), we report model performance using the optimal mapping scale on a distinct set of 6 images.
Each model (MLP with 4 layers, 256 channels, ReLU activation, sigmoid output) is trained for 1000 iterations using the Adam [[19](#bib.bib19)] optimizer with default settings (β1=0.9subscript𝛽10.9\beta\_{1}=0.9, β2=0.999subscript𝛽20.999\beta\_{2}=0.999, ϵ=10−8italic-ϵsuperscript108\epsilon=10^{-8}). We use a manually-tuned learning rate of 2×10−32superscript1032\times 10^{-3} for each method. Results are visualized in Figure [18](#A6.F18 "Figure 18 ‣ Appendix F Additional results figures ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains").

|  |  |
| --- | --- |
|  | ATLAS |
| No mapping | 26.14±1.45plus-or-minus26.141.4526.14\pm 1.45 |
| Basic | 28.58±2.45plus-or-minus28.582.4528.58\pm 2.45 |
| Positional enc. | 32.23±3.08plus-or-minus32.233.0832.23\pm 3.08 |
| Gaussian | 34.51±2.72plus-or-minus34.512.72\mathbf{34.51\pm 2.72} |

Table 6: 3D MRI results (mean ±plus-or-minus\pm standard deviation of PSNR).

### E.5 3D inverse rendering for view synthesis

In this task we use the “tiny NeRF” simplified version of the view synthesis method NeRF [[27](#bib.bib27)] where hierarchical sampling and view dependence have been removed. The model is trained to predict the color and volume density at an input 3D point. Volumetric rendering is used to render novel viewpoints of the object. The loss is calculated between the rendered views and ground truth renders. In our experiments we use the NeRF *Lego* dataset of 120 images downsampled to 400×400400400400\times 400 pixel resolution. The dataset is split into 100 training images, 7 validation images, and 13 test images. The reconstruction quality on the validation images is used to determine the best mapping scale; for this scene we find σg=6.05subscript𝜎𝑔6.05\sigma\_{g}=6.05 and σp=1.27subscript𝜎𝑝1.27\sigma\_{p}=1.27 perform best.

The model (MLP with 4 layers, 256 channels, ReLU activation, sigmoid on RGB output) is trained for 5×1055superscript1055\times 10^{5} iterations using the Adam [[19](#bib.bib19)] optimizer with default settings (β1=0.9subscript𝛽10.9\beta\_{1}=0.9, β2=0.999subscript𝛽20.999\beta\_{2}=0.999, ϵ=10−8italic-ϵsuperscript108\epsilon=10^{-8}). The learning rate is manually tuned for each mapping: 10−2superscript10210^{-2} for no mapping, 5×10−35superscript1035\times 10^{-3} for basic, 5×10−45superscript1045\times 10^{-4} for positional encoding, and 5×10−45superscript1045\times 10^{-4} for Gaussian. During training we use batches of 1024 rays.

The original NeRF method [[27](#bib.bib27)] uses an input mapping similar to the *Positional encoding* we compare against. The original NeRF mapping is smaller than our mappings (8 vs. 256 frequencies). We include metrics for this mapping in Table [7](#A5.T7 "Table 7 ‣ E.5 3D inverse rendering for view synthesis ‣ Appendix E Task details ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") under *Original pos. enc*. The positional encoding mappings only contain frequencies on the axes, and are therefore biased towards signals with on-axis frequency content (as demonstrated in Section [A.5](#A1.SS5 "A.5 Failures of positional encoding (axis-aligned bias) ‣ Appendix A Further experiments ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains")). In our experiments we rotate the *Lego* scene, which was manually axis-aligned in the original dataset, for a more equitable comparison. Table [7](#A5.T7 "Table 7 ‣ E.5 3D inverse rendering for view synthesis ‣ Appendix E Task details ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") also reports metrics for positional encodings on the original axis-aligned scene. Results are visualized in Figure [19](#A6.F19 "Figure 19 ‣ Appendix F Additional results figures ‣ Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains").

|  |  |
| --- | --- |
|  | 3D NeRF |
| No mapping | 22.41±0.92plus-or-minus22.410.9222.41\pm 0.92 |
| Basic | 23.16±0.90plus-or-minus23.160.9023.16\pm 0.90 |
| Original pos. enc. | 24.81±0.88plus-or-minus24.810.8824.81\pm 0.88 |
| Positional enc. | 25.28±0.83plus-or-minus25.280.8325.28\pm 0.83 |
| Gaussian | 25.48±0.89plus-or-minus25.480.89\mathbf{25.48\pm 0.89} |
| Original pos. enc. (axis-aligned) | 25.60±0.76plus-or-minus25.600.7625.60\pm 0.76 |
| Positional enc. (axis-aligned) | 26.27±0.91plus-or-minus26.270.9126.27\pm 0.91 |

Table 7: 3D NeRF results (mean and standard deviation of PSNR). Error is calculated based on held-out images of the scene since the ground truth radiance field is not known.

## Appendix F Additional results figures

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Refer to caption | Refer to caption | Refer to caption | Refer to caption | Refer to caption |
| Refer to caption | Refer to caption | Refer to caption | Refer to caption | Refer to caption |
| Refer to caption | Refer to caption | Refer to caption | Refer to caption | Refer to caption |
| Refer to caption | Refer to caption | Refer to caption | Refer to caption | Refer to caption |
| Refer to caption | Refer to caption | Refer to caption | Refer to caption | Refer to caption |
| (a) Ground Truth | (b) No mapping | (c) Basic | (d) Positional enc. | (e) Gaussian |

Figure 15: Additional results for the 2D image regression task, for three images from our *Natural* dataset (top) and two images from our *Text* dataset (bottom).



|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Refer to caption | Refer to caption | Refer to caption | Refer to caption | Refer to caption |
| Refer to caption | Refer to caption | Refer to caption | Refer to caption | Refer to caption |
| Refer to caption | Refer to caption | Refer to caption | Refer to caption | Refer to caption |
| Refer to caption | Refer to caption | Refer to caption | Refer to caption | Refer to caption |
| (a) Ground Truth | (b) No mapping | (c) Basic | (d) Positional enc. | (e) Gaussian |

Figure 16: Additional results for the 3D shape occupancy task [[24](#bib.bib24)].



|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Refer to caption | Refer to caption | Refer to caption | Refer to caption | Refer to caption |
| Refer to caption | Refer to caption | Refer to caption | Refer to caption | Refer to caption |
| (a) Ground Truth | (b) No mapping | (c) Basic | (d) Positional enc. | (e) Gaussian |

Figure 17: Results for the 2D CT task.



|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Refer to caption | Refer to caption | Refer to caption | Refer to caption | Refer to caption |
| (a) Ground Truth | (b) No mapping | (c) Basic | (d) Positional enc. | (e) Gaussian |

Figure 18: Additional results for the 3D MRI task.



|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Refer to caption | Refer to caption | Refer to caption | Refer to caption | Refer to caption |
| Refer to caption | Refer to caption | Refer to caption | Refer to caption | Refer to caption |
| Refer to caption | Refer to caption | Refer to caption | Refer to caption | Refer to caption |
| (a) Ground Truth | (b) No mapping | (c) Basic | (d) Positional enc. | (e) Gaussian |

Figure 19: Additional results for the inverse rendering task [[27](#bib.bib27)].

[◄](/html/2006.10738)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2006.10739)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2006.10739)
[View original  
on arXiv](https://arxiv.org/abs/2006.10739)[►](/html/2006.10740)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Wed Mar 6 12:38:52 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
