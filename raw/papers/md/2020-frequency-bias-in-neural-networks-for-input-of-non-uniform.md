---
arxiv: '2003.04560'
authors:
- Ronen Basri
- Meirav Galun
- Amnon Geifman
- David Jacobs
- Yoni Kasten
- Shira Kritchman
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: Frequency Bias in Neural Networks for Input of Non-Uniform Density
url: http://arxiv.org/abs/2003.04560v1
year: 2020
---

# Frequency Bias in Neural Networks for Input of Non-Uniform Density

Ronen Basri
  
Meirav Galun
  
Amnon Geifman
  
David Jacobs
  
Yoni Kasten
  
Shira Kritchman

###### Abstract

Recent works have partly attributed the generalization ability of over-parameterized neural networks to frequency bias – networks trained with gradient descent on data drawn from a uniform distribution find a low frequency fit before high frequency ones. As realistic training sets are not drawn from a uniform distribution, we here use the Neural Tangent Kernel (NTK) model to explore the effect of variable density on training dynamics. Our results, which combine analytic and empirical observations, show that when learning a pure harmonic function of frequency κ𝜅\kappa, convergence at a point 𝐱∈𝕊d−1𝐱superscript𝕊𝑑1\mathbf{x}\in\mathbb{S}^{d-1} occurs in time O​(κd/p​(𝐱))𝑂superscript𝜅𝑑𝑝𝐱O(\kappa^{d}/p(\mathbf{x})) where p​(𝐱)𝑝𝐱p(\mathbf{x}) denotes the local density at 𝐱𝐱\mathbf{x}. Specifically, for data in 𝕊1superscript𝕊1\mathbb{S}^{1} we analytically derive the eigenfunctions of the kernel associated with the NTK for two-layer networks. We further prove convergence results for deep, fully connected networks with respect to the spectral decomposition of the NTK. Our empirical study highlights similarities and differences between deep and shallow networks in this model.

Machine Learning, ICML

## 1 Introduction

A key question in understanding the success of neural networks is: what makes over-parameterized networks generalize so well, avoiding solutions that overfit the training data? In search of an explanation, a number of recent papers (Farnia et al., [2018](#bib.bib12); Rahaman et al., [2019](#bib.bib19); Xu et al., [2019](#bib.bib25)) have suggested that training with gradient descent (GD) (as well as SGD) yields a frequency bias – in early epochs training a neural net yields a low frequency fit to the target function, while high frequencies are learned only in later epochs, if they are needed to fit the data (see Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")(top)).

!(/html/2003.04560/assets/x1.png)

!(/html/2003.04560/assets/x2.png)

!(/html/2003.04560/assets/x3.png)

!(/html/2003.04560/assets/x4.png)

Figure 1: Frequency bias under uniform (top) and non-uniform (bottom) distributions. The light cyan line represents the target function which is composed of the sum of a low and high frequency functions. The thin black line represents the network output. Top: when training data is distributed uniformly, low frequency (left) is learned before high frequency (right). Bottom: with non-uniform distribution (positive region is dense, negative is sparse), a good low frequency fit for the low density region is obtained only after 40 epochs, but by then the network fits most of the high frequency component of the target function at the dense region.

This frequency bias has been carefully analyzed in the case of over-parameterized, two-layer networks with Rectified Linear Unit (ReLU) activation, when only the first layer is trained. The dynamics of GD in this case was shown to match the dynamics of GD for the corresponding Neural Tangent Kernel (NTK) (Arora et al., [2019b](#bib.bib4); Du et al., [2019](#bib.bib11); Jacot et al., [2018](#bib.bib15)). Assuming the training data is distributed uniformly on a hypersphere, the NTK matrix forms a convolution on the sphere. Its eigenvectors consist of the spherical harmonic functions (Basri et al., [2019](#bib.bib6); Xie et al., [2017](#bib.bib24)), and its eigenvalues shrink monotonically with frequency, yielding longer convergence times for high frequency components. Specifically, for training data on the circle, high frequencies are learned quadratically slower than low frequencies, and this frequency-dependent gap increases exponentially with dimension (Basri et al., [2019](#bib.bib6); Bietti & Mairal, [2019](#bib.bib7); Cao et al., [2019](#bib.bib9)).

All this previous work assumed that training data is distributed uniformly. However, realistic training datasets are distributed with a non-uniform density. A natural question therefore is to what extent frequency bias is exhibited for such datasets? Below we provide evidence that frequency bias interacts with density. We show that in any region of the input space with locally constant density, low frequencies are still learned much faster than high frequencies, but the rate of learning also depends linearly on the density. This phenomenon is demonstrated in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")(bottom).

Our paper contains both theoretical and empirical results. We first focus on analyzing the NTK model for two-layer networks with ReLU activation and 2D input, normalized to lie on the unit circle, allowing for input drawn from a non-uniform density that is piecewise constant. For this model we derive closed form expressions for its eigenfunctions and eigenvalues. These eigenfunctions contain functions of piecewise constant local frequency, with higher frequencies where the density of the training data is higher. This implies that we learn high frequency components of a target function faster in regions of higher density. This also allows us to prove that a pure 1-dimensional sine function of frequency κ𝜅\kappa is learned in time O​(κ2/p∗)𝑂superscript𝜅2superscript𝑝O(\kappa^{2}/p^{\*}), where p∗superscript𝑝p^{\*} denotes the minimum density in the input space. Our experiments illustrate these results and further suggest that for input on a d−1𝑑1d-1-dimensional hypersphere, spherical harmonics are learned in time O​(κd/p∗)𝑂superscript𝜅𝑑superscript𝑝O(\kappa^{d}/p^{\*}).

We next examine the NTK for deep, fully connected (FC) networks. We first prove that given a target function y​(𝐱)𝑦𝐱y(\mathbf{x}), training networks of finite width with GD converges to y𝑦y at a speed that depends on the projection of y𝑦y over the eigenvectors of the NTK, extending previous results proved for two-layer networks (Arora et al., [2019b](#bib.bib4); Cao et al., [2019](#bib.bib9)). We further show that for uniform data the eigenfunctions of NTK consist of the spherical harmonics. We complement these observations with several empirical findings. (1) We show that for uniformly distributed data the eigenvalues decay with frequency, suggesting that frequency bias exists also in deep FC networks. Moreover, similar to two-layer networks, a pure harmonic function of frequency κ𝜅\kappa is learned in time O​(κd)𝑂superscript𝜅𝑑O(\kappa^{d}) asymptotically in κ𝜅\kappa. However, deeper networks appear to learn frequencies of lower k𝑘k faster than shallow ones. (2) For training data drawn from non-uniform densities the eigenfunctions of NTK appear indistinguishable from those obtained for two-layer networks, indicating that with deep nets learning a harmonic of frequency κ𝜅\kappa should also require O​(κd/p∗)𝑂superscript𝜅𝑑superscript𝑝O(\kappa^{d}/p^{\*}) iterations.

Our results have several implications. First, we extend results that have been proven for training data with a uniform density to the more realistic case of non-uniform density, also extending results for shallow networks to deep, fully connected networks. These results support the idea that real neural networks have a frequency bias that can explain their ability to avoid overfitting. Second, while it is not surprising that networks fit functions of all frequencies more slowly in regions with low data density, we demonstrate that this is the case and quantify this effect. Our results have an interesting implication for training that uses early stopping to regularize the solution. Suppose the signal one wishes to fit is low frequency, and it is corrupted by high frequency noise. Because a network learns low frequency signals more slowly in regions of low density, by the time the signal is learned in these regions, the network will also have learned high frequency components of the noise in regions of high density. This is illustrated in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")(bottom).

\comment

——-
Figures:

1. 1.

   Motivating example: Target function made of low frequency + high frequency noise (of small amplitude). Data is composed, say, of two constant densities. Show fit at different epochs. Essentially we want to show that network cannot fit the low frequency in all space at any given time. i.e. to get the low frequency at the sparse part you already fit the noise in the dense part [Ronen: Should we show in contrast that this is possible with uniform density? Maybe not..]
2. 2.

   2 layers, NTK: the eigenfunctions obtained for a certain density. Also for 2D data.
3. 3.

   2 layers, NTK: a plot of the eigenvalues + normalized by Z to show they all coincide.. Also for 2D data.
4. 4.

   2 layers network: convergence times for several densities + normalized so that all parabolas coincide
5. 5.

   Inverse linear relation of convergence time and density. Also for 2D data.
6. 6.

   Deep NTK: eigenfunctions for uniform and non-uniform densities. (Maybe unnecessary because they are the same as the shallow case, or maybe plot just the ones for a deep net?)
7. 7.

   Deep, NTK: exponent as a function of number of layers for 1D, 2D and maybe 3D.
8. 8.

   Deep network: actual convergence times for 3, 5 and 7 layers. Show a fit to the previous figure. Also for 2D input.

FC deep nets (NTK):

1. 1.

   Proof of convergence with fine grained.
2. 2.

   Eigen functions under the uniform density are spherical harmonics and are ordered according to frequency.
3. 3.

   Complexity (exponent) of convergence rate under the uniform density as a function of depth and dimension (graph).
4. 4.

   Run experiments to fit this graph (using Yoni’s code).
5. 5.

   Is there degeneracy with very deep FC?
6. 6.

   Non-uniform for deep FC + experiment. Also Resnet?
7. 7.

   Verify theory for non-uniform. Are the eigenvalues identical to the uniform case?
8. 8.

   Probably not in this round: NTK for Resnet

## 2 Prior work

Many recent papers attempt to explain the generalization ability of overparameterized nets. Perhaps the most convincing relate overparameterized networks to kernel methods. (Jacot et al., [2018](#bib.bib15)) identified a family of kernels, termed Neural Tangent Kernels, and showed that neural networks behave like these kernels, in the limit of infinite widths.
Related work investigated variants of these kernels, showing that networks of finite, albeit very large widths converge to zero training error almost always and deriving generalization bounds for such networks. These analyses were applied to two-layer networks (Bach, [2017](#bib.bib5); Bietti & Mairal, [2019](#bib.bib7); Du et al., [2019](#bib.bib11); Vempala & Wilmes, [2018](#bib.bib21); Xie et al., [2017](#bib.bib24)), multilayer perceptrons (i,e. fully connected), residual and convolutional networks (Allen-Zhu et al., [2018](#bib.bib1), [2019](#bib.bib2); Arora et al., [2019a](#bib.bib3); Huang & Yau, [2019](#bib.bib14); Lee et al., [2019](#bib.bib16)).

However, these kernel models have been criticised for requiring unrealistically wide networks. Additionally, it is still debated if such linear dynamics (referred to as “lazy training”) fully explain the performance of neural networks. Recent theoretical and empirical results suggest that NTK models still somewhat underperform common nonlinear networks (Arora et al., [2019a](#bib.bib3); Chizat et al., [2019](#bib.bib10); Novak et al., [2019](#bib.bib18); Woodworth et al., [2019](#bib.bib23)).

Other work suggested that networks are biased to learn simple functions, and in particular that GD proceeds by first fitting a low frequency function to the target function, and only fits the higher frequencies in later epochs
(Rahaman et al., [2019](#bib.bib19); Xu et al., [2019](#bib.bib25); Farnia et al., [2018](#bib.bib12)). Additional work (Bach, [2017](#bib.bib5); Basri et al., [2019](#bib.bib6); Bietti & Mairal, [2019](#bib.bib7); Cao et al., [2019](#bib.bib9)) proved the existence of frequency bias in NTK models of two-layer networks and derived convergence rates of training as a function of target frequency. All of these works assumed that training data is distributed uniformly. (Canu & Elisseef, [1999](#bib.bib8)) proposed loss functions that allow higher frequency fit in regions where training data is dense, and only low frequency fit in the sparse regions. Our results suggest that such a penalization may be implicitly enforced in NTK models.

Classical work on kernel methods acknowledged the importance of understanding the eigenfunctions and eigenvalues of kernels for non-uniform data distributions, but focused mainly on bounding the difference between the empirical kernel matrix and the theoretical kernel for the given distribution (e.g., (Shawe-Taylor et al., [2005](#bib.bib20); Williams & Seeger, [2000](#bib.bib22))). (Liang & Lee, [2013](#bib.bib17)) derived analytic expressions for the eigenfunctions of polynomial kernels. (Goel & Klivans, [2017](#bib.bib13)) investigated the gram matrix of the data distribution and showed that sufficiently fast decay of its eigenvalues allows learnability by neural networks. We are unaware of works that derive analytic expressions for the eigenfunctions of NTK under non-uniform distributions.

\comment

Left outs:
Papers showing implicit regularization (https://arxiv.org/abs/1810.01075).
  
Convergence results for two-layers (like Arora fine grained) + spherical harmonics for uniform distribution
(https://arxiv.org/pdf/1912.01198.pdf)
  
The dynamics of gradient descent bias the model towards simple solutions - initialization separates deep and shallow
(https://arxiv.org/pdf/1909.12051.pdf)

## 3 Preliminaries

We consider in this work NTK models for fully connected neural networks with rectified linear unit (ReLU) activations. These kernels are defined through the following formula

|  |  |  |  |
| --- | --- | --- | --- |
|  | k​(𝐱i,𝐱j)=𝔼𝐰∼ℐ​⟨∂f​(𝐱i,𝐰)∂𝐰,∂f​(𝐱j,𝐰)∂𝐰⟩,𝑘subscript𝐱𝑖subscript𝐱𝑗subscript𝔼similar-to𝐰ℐ  𝑓subscript𝐱𝑖𝐰𝐰𝑓subscript𝐱𝑗𝐰𝐰k(\mathbf{x}\_{i},\mathbf{x}\_{j})=\mathbb{E}\_{\mathbf{w}\sim{\cal I}}\left<\frac{\partial f(\mathbf{x}\_{i},\mathbf{w})}{\partial\mathbf{w}},\frac{\partial f(\mathbf{x}\_{j},\mathbf{w})}{\partial\mathbf{w}}\right>, |  | (1) |

where f​(𝐱,𝐰)𝑓𝐱𝐰f(\mathbf{x},\mathbf{w}) is the network output for point 𝐱∈ℝd𝐱superscriptℝ𝑑\mathbf{x}\in\mathbb{R}^{d} with parameters 𝐰𝐰\mathbf{w}, 𝐱isubscript𝐱𝑖\mathbf{x}\_{i} and 𝐱jsubscript𝐱𝑗\mathbf{x}\_{j} are any two training points, and the expectation is over the possible initializations of 𝐰𝐰\mathbf{w}, denoted ℐℐ{\cal I} (usually normal distribution).

We first consider a two layer network with bias:

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(𝐱;𝐰)=1m​∑r=1mar​σ​(𝐰rT​𝐱+br),𝑓  𝐱𝐰1𝑚superscriptsubscript𝑟1𝑚subscript𝑎𝑟𝜎superscriptsubscript𝐰𝑟𝑇𝐱subscript𝑏𝑟f(\mathbf{x};\mathbf{w})=\frac{1}{\sqrt{m}}\sum\_{r=1}^{m}a\_{r}\sigma(\mathbf{w}\_{r}^{T}\mathbf{x}+b\_{r}), |  | (2) |

where ‖𝐱‖=1norm𝐱1\|\mathbf{x}\|=1 (denoted 𝐱∈𝕊d−1𝐱superscript𝕊𝑑1\mathbf{x}\in\mathbb{S}^{d-1}) is the input, the vector 𝐰𝐰\mathbf{w} includes the weights and bias terms of the first layer, denoted respectively W=[𝐰1,…,𝐰m]∈ℝd×m𝑊

subscript𝐰1…subscript𝐰𝑚superscriptℝ𝑑𝑚W=[\mathbf{w}\_{1},...,\mathbf{w}\_{m}]\in\mathbb{R}^{d\times m} and 𝐛=[b1,…,bm]T∈ℝm𝐛superscript

subscript𝑏1…subscript𝑏𝑚
𝑇superscriptℝ𝑚\mathbf{b}=[b\_{1},...,b\_{m}]^{T}\in\mathbb{R}^{m}, as well as the weights of the second layer, denoted 𝐚=[a1,…,am]T∈ℝm𝐚superscript

subscript𝑎1…subscript𝑎𝑚
𝑇superscriptℝ𝑚\mathbf{a}=[a\_{1},...,a\_{m}]^{T}\in\mathbb{R}^{m}. σ𝜎\sigma denotes the ReLU function, σ​(x)=max⁡(x,0)𝜎𝑥𝑥0\sigma(x)=\max(x,0). Bias is important in the case of two-layer networks since (Basri et al., [2019](#bib.bib6)) without bias such networks are non-universal and cannot express harmonic functions of odd frequencies except frequency 1.

We then consider deep fully-connected networks with L+1>2𝐿12L+1>2 layers. For such networks we forgo the bias since our empirical results (Section [5](#S5 "5 Deep networks ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) indicate that they are universal even without bias. These networks are expressed as

|  |  |  |  |
| --- | --- | --- | --- |
|  |  | f(𝐱;𝐰)=W(L+1)⋅cσdLσ(W(L)⋅\displaystyle f(\mathbf{x};\mathbf{w})=W^{(L+1)}\cdot\sqrt{\frac{c\_{\sigma}}{d\_{L}}}\sigma\left(W^{(L)}\cdot\right. |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | cσdL−1σ(W(L−1)⋯cσd1σ(W(1)𝐱))),\displaystyle\left.\sqrt{\frac{c\_{\sigma}}{d\_{L-1}}}\sigma\left(W^{(L-1)}\cdots\sqrt{\frac{c\_{\sigma}}{d\_{1}}}\sigma\left(W^{(1)}\mathbf{x}\right)\right)\right), |  | (3) |

where 𝐱∈ℝd1𝐱superscriptℝsubscript𝑑1\mathbf{x}\in\mathbb{R}^{d\_{1}}, ‖𝐱‖=1norm𝐱1\|\mathbf{x}\|=1, the parameters 𝐰𝐰\mathbf{w} include W(L+1),W(L),…,W(1)

superscript𝑊𝐿1superscript𝑊𝐿…superscript𝑊1W^{(L+1)},W^{(L)},...,W^{(1)}, where W(l)∈ℝdl×dl−1superscript𝑊𝑙superscriptℝsubscript𝑑𝑙subscript𝑑𝑙1W^{(l)}\in\mathbb{R}^{d\_{l}\times d\_{l-1}}, W(L+1)∈ℝ1×dLsuperscript𝑊𝐿1superscriptℝ1subscript𝑑𝐿W^{(L+1)}\in\mathbb{R}^{1\times d\_{L}}, and cσ=1/(𝔼z∼𝒩​(0,1)​[σ​(z)2])=2subscript𝑐𝜎1subscript𝔼similar-to𝑧𝒩01delimited-[]𝜎superscript𝑧22c\_{\sigma}=1/\left(\mathbb{E}\_{z\sim\mathcal{N}(0,1)}[\sigma(z)^{2}]\right)=2.

We assume that n𝑛n training points are sampled i.i.d. from an arbitrary distribution p​(𝐱)𝑝𝐱p(\mathbf{x}) on the hypersphere and that each sample 𝐱isubscript𝐱𝑖\mathbf{x}\_{i} is supplied with a target value yi∈ℝsubscript𝑦𝑖ℝy\_{i}\in\mathbb{R} from an unknown function yi=g​(𝐱i)subscript𝑦𝑖𝑔subscript𝐱𝑖y\_{i}=g(\mathbf{x}\_{i}). Our theoretical derivations further assume that p​(𝐱)𝑝𝐱p(\mathbf{x}) is piecewise constant. The network is trained to minimize the ℓ2subscriptℓ2\ell\_{2} loss

|  |  |  |  |
| --- | --- | --- | --- |
|  | Φ​(𝐰)=12​∑i=1n(yi−f​(𝐱i;𝐰))2.Φ𝐰12superscriptsubscript𝑖1𝑛superscriptsubscript𝑦𝑖𝑓  subscript𝐱𝑖𝐰2\Phi(\mathbf{w})=\frac{1}{2}\sum\_{i=1}^{n}(y\_{i}-f(\mathbf{x}\_{i};\mathbf{w}))^{2}. |  | (4) |

using gradient descent (GD).

For our analysis, to simplify the NTK expressions, in the case of a two-layer network we only train the weights and bias of the first layer (as in (Arora et al., [2019b](#bib.bib4); Du et al., [2019](#bib.bib11))). We initialize these weights from a normal distribution 𝐰r(0),br(0)∼𝒩​(0,τ2​I)similar-to

superscriptsubscript𝐰𝑟0superscriptsubscript𝑏𝑟0
𝒩0superscript𝜏2𝐼\mathbf{w}\_{r}^{(0)},b\_{r}^{(0)}\sim{\cal N}(0,\tau^{2}I). We further initialize arsubscript𝑎𝑟a\_{r} from a uniform distribution on {−1,1}11\{-1,1\} and keep those weights fixed. In the case of deep networks we train all the weights, initializing by 𝐰∼𝒩​(0,I)similar-to𝐰𝒩0𝐼\mathbf{w}\sim{\cal N}(0,I).

We next provide expressions for the corresponding neural tangent kernels. For a two-layer network with bias where only the first layer weights are trained the corresponding NTK takes the form (Basri et al., [2019](#bib.bib6))

|  |  |  |  |
| --- | --- | --- | --- |
|  | k​(𝐱i,𝐱j)=14​π​(𝐱iT​𝐱j+1)​(π−arccos⁡(𝐱iT​𝐱j)).𝑘subscript𝐱𝑖subscript𝐱𝑗14𝜋superscriptsubscript𝐱𝑖𝑇subscript𝐱𝑗1𝜋superscriptsubscript𝐱𝑖𝑇subscript𝐱𝑗k(\mathbf{x}\_{i},\mathbf{x}\_{j})=\frac{1}{4\pi}(\mathbf{x}\_{i}^{T}\mathbf{x}\_{j}+1)(\pi-\arccos(\mathbf{x}\_{i}^{T}\mathbf{x}\_{j})). |  | (5) |

When the training data is distributed uniformly, this kernel forms a convolution operator, and so its eigenfunctions are the spherical harmonics on the hypersphere 𝕊d−1superscript𝕊𝑑1\mathbb{S}^{d-1} (or Fourier series when d=2𝑑2d=2). The eigenvalues shrink at the rate of O​(1/κd)𝑂1superscript𝜅𝑑O(1/\kappa^{d}), where κ𝜅\kappa denotes the frequency of the spherical harmonic functions. Gradient descent training of a target function composed of a pure harmonic requires a number of iterations that is inversely proportional to the corresponding eigenvalue, i.e., O​(κd)𝑂superscript𝜅𝑑O(\kappa^{d}). (Bach, [2017](#bib.bib5); Basri et al., [2019](#bib.bib6); Bietti & Mairal, [2019](#bib.bib7); Cao et al., [2019](#bib.bib9); Xie et al., [2017](#bib.bib24))

For a deep FC network the NTK is expressed by the following recursion (Arora et al., [2019a](#bib.bib3); Jacot et al., [2018](#bib.bib15))

|  |  |  |  |
| --- | --- | --- | --- |
|  | Θ∞(L)​(𝐱i,𝐱j)=Θ∞(L−1)​(𝐱i,𝐱j)​Σ˙(L)​(𝐱i,𝐱j)+Σ(L)​(𝐱i,𝐱j),superscriptsubscriptΘ𝐿subscript𝐱𝑖subscript𝐱𝑗superscriptsubscriptΘ𝐿1subscript𝐱𝑖subscript𝐱𝑗superscript˙Σ𝐿subscript𝐱𝑖subscript𝐱𝑗superscriptΣ𝐿subscript𝐱𝑖subscript𝐱𝑗\Theta\_{\infty}^{(L)}(\mathbf{x}\_{i},\mathbf{x}\_{j})=\Theta\_{\infty}^{(L-1)}(\mathbf{x}\_{i},\mathbf{x}\_{j})\dot{\Sigma}^{(L)}(\mathbf{x}\_{i},\mathbf{x}\_{j})+\Sigma^{(L)}(\mathbf{x}\_{i},\mathbf{x}\_{j}), |  | (6) |

where for h∈[L]ℎdelimited-[]𝐿h\in[L]

|  |  |  |  |
| --- | --- | --- | --- |
|  | Σ(0)​(𝐱i,𝐱j)superscriptΣ0subscript𝐱𝑖subscript𝐱𝑗\displaystyle\Sigma^{(0)}(\mathbf{x}\_{i},\mathbf{x}\_{j}) | =𝐱iT​𝐱jabsentsuperscriptsubscript𝐱𝑖𝑇subscript𝐱𝑗\displaystyle=\mathbf{x}\_{i}^{T}\mathbf{x}\_{j} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Λ(h)​(𝐱i,𝐱j)superscriptΛℎsubscript𝐱𝑖subscript𝐱𝑗\displaystyle\Lambda^{(h)}(\mathbf{x}\_{i},\mathbf{x}\_{j}) | =[Σ(h−1)​(𝐱i,𝐱i)Σ(h−1)​(𝐱i,𝐱j)Σ(h−1)​(𝐱j,𝐱i)Σ(h−1)​(𝐱j,𝐱j)]absentmatrixsuperscriptΣℎ1subscript𝐱𝑖subscript𝐱𝑖superscriptΣℎ1subscript𝐱𝑖subscript𝐱𝑗superscriptΣℎ1subscript𝐱𝑗subscript𝐱𝑖superscriptΣℎ1subscript𝐱𝑗subscript𝐱𝑗\displaystyle=\begin{bmatrix}\Sigma^{(h-1)}(\mathbf{x}\_{i},\mathbf{x}\_{i})&\Sigma^{(h-1)}(\mathbf{x}\_{i},\mathbf{x}\_{j})\\ \Sigma^{(h-1)}(\mathbf{x}\_{j},\mathbf{x}\_{i})&\Sigma^{(h-1)}(\mathbf{x}\_{j},\mathbf{x}\_{j})\end{bmatrix} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Σ(h)​(𝐱i,𝐱j)superscriptΣℎsubscript𝐱𝑖subscript𝐱𝑗\displaystyle\Sigma^{(h)}(\mathbf{x}\_{i},\mathbf{x}\_{j}) | =cσ​𝔼(u,v)∼𝒩​(0,Λ(h))​[σ​(u)​σ​(v)]absentsubscript𝑐𝜎subscript𝔼similar-to𝑢𝑣𝒩0superscriptΛℎdelimited-[]𝜎𝑢𝜎𝑣\displaystyle=c\_{\sigma}\mathbb{E}\_{(u,v)\sim\mathcal{N}(0,\Lambda^{(h)})}[\sigma(u)\sigma(v)] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Σ˙(h)​(𝐱i,𝐱j)superscript˙Σℎsubscript𝐱𝑖subscript𝐱𝑗\displaystyle\dot{\Sigma}^{(h)}(\mathbf{x}\_{i},\mathbf{x}\_{j}) | =cσ​𝔼(u,v)∼𝒩​(0,Λ(h))​[σ˙​(u)​σ˙​(v)].absentsubscript𝑐𝜎subscript𝔼similar-to𝑢𝑣𝒩0superscriptΛℎdelimited-[]˙𝜎𝑢˙𝜎𝑣\displaystyle=c\_{\sigma}\mathbb{E}\_{(u,v)\sim\mathcal{N}(0,\Lambda^{(h)})}[\dot{\sigma}(u)\dot{\sigma}(v)]. |  |

Here σ˙​(⋅)˙𝜎⋅\dot{\sigma}(\cdot) denotes the step function (i.e., the derivative of the ReLU function).
The covariance matrices have the form Λ=[1ρρ1]Λmatrix1𝜌𝜌1\Lambda=\begin{bmatrix}1&\rho\\
\rho&1\end{bmatrix} with |ρ|≤1𝜌1|\rho|\leq 1, and the expectations have the following closed form expressions

|  |  |  |
| --- | --- | --- |
|  | 𝔼(u,v)∼𝒩​(0,Λ(h))​[σ​(u)​σ​(v)]=ρ​(π−arccos⁡(ρ))+1−ρ22​πsubscript𝔼similar-to𝑢𝑣𝒩0superscriptΛℎdelimited-[]𝜎𝑢𝜎𝑣𝜌𝜋𝜌1superscript𝜌22𝜋\displaystyle\mathbb{E}\_{(u,v)\sim\mathcal{N}(0,\Lambda^{(h)})}[\sigma(u)\sigma(v)]=\frac{\rho(\pi-\arccos(\rho))+\sqrt{1-\rho^{2}}}{2\pi} |  |
|  |  |  |
| --- | --- | --- |
|  | 𝔼(u,v)∼𝒩​(0,Λ(h))​[σ˙​(u)​σ˙​(v)]=ρ​(π−arccos⁡(ρ))2​π.subscript𝔼similar-to𝑢𝑣𝒩0superscriptΛℎdelimited-[]˙𝜎𝑢˙𝜎𝑣𝜌𝜋𝜌2𝜋\displaystyle\mathbb{E}\_{(u,v)\sim\mathcal{N}(0,\Lambda^{(h)})}[\dot{\sigma}(u)\dot{\sigma}(v)]=\frac{\rho(\pi-\arccos(\rho))}{2\pi}. |  |

## 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions

We begin by investigating the NTK model for two-layer networks when the training is drawn from a non-uniform distribution. Focusing first on 1D target functions y​(𝐱):𝕊1→ℝ:𝑦𝐱→superscript𝕊1ℝy(\mathbf{x}):\mathbb{S}^{1}\rightarrow\mathbb{R} and a piecewise constant data distribution p​(𝐱)𝑝𝐱p(\mathbf{x}), we derive explicit expressions for the eigenfunctions and eigenvalues of NTK. This allows us to prove that learning a one-dimensional function of frequency κ𝜅\kappa requires O​(κ2/p∗)𝑂superscript𝜅2superscript𝑝O(\kappa^{2}/p^{\*}) iterations, where p∗superscript𝑝p^{\*} denotes the minimal density in p​(x)𝑝𝑥p(x). We complement these theoretical derivations with experiments with functions in higher dimensions, which indicate that learning functions of frequency κ𝜅\kappa in 𝕊d−1superscript𝕊𝑑1\mathbb{S}^{d-1} requires
O​(κd/p∗)𝑂superscript𝜅𝑑superscript𝑝O(\kappa^{d}/p^{\*}) iterations.

Consider the NTK model described in ([5](#S3.E5 "In 3 Preliminaries ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")), which corresponds to an infinitely wide, two-layer network for which only the first layer is trained. Suppose that n𝑛n training data points are sampled from a non-uniform, piecewise constant distribution p​(𝐱)𝑝𝐱p(\mathbf{x}) on the circle, 𝐱∈𝕊1𝐱superscript𝕊1\mathbf{x}\in\mathbb{S}^{1}. We then form an n×n𝑛𝑛n\times n matrix Hpsuperscript𝐻𝑝H^{p} whose entries for samples 𝐱isubscript𝐱𝑖\mathbf{x}\_{i} and 𝐱jsubscript𝐱𝑗\mathbf{x}\_{j} consist of Hi​jp=k​(𝐱i,𝐱j)subscriptsuperscript𝐻𝑝𝑖𝑗𝑘subscript𝐱𝑖subscript𝐱𝑗H^{p}\_{ij}=k(\mathbf{x}\_{i},\mathbf{x}\_{j}), with k𝑘k as defined in ([5](#S3.E5 "In 3 Preliminaries ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")). Following (Arora et al., [2019b](#bib.bib4)), the convergence rates of GD for such a network will depend on the eigen-system of Hpsuperscript𝐻𝑝H^{p}. To analyze this eigen-system, we consider the limit of Hpsuperscript𝐻𝑝H^{p} as the number of points goes to infinity. In this limit the eigen-system of Hpsuperscript𝐻𝑝H^{p} approaches the eigen-system of the kernel k​(𝐱i,𝐱j)​p​(𝐱j)𝑘subscript𝐱𝑖subscript𝐱𝑗𝑝subscript𝐱𝑗k(\mathbf{x}\_{i},\mathbf{x}\_{j})p(\mathbf{x}\_{j}), where the eigenfunctions f​(x)𝑓𝑥f(x) satisfy the following equation (Shawe-Taylor et al., [2005](#bib.bib20); Williams & Seeger, [2000](#bib.bib22)),

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∫𝕊1k​(𝐱i,𝐱j)​p​(𝐱j)​f​(𝐱j)​𝑑𝐱j=λ​f​(𝐱i).subscriptsuperscript𝕊1𝑘subscript𝐱𝑖subscript𝐱𝑗𝑝subscript𝐱𝑗𝑓subscript𝐱𝑗differential-dsubscript𝐱𝑗𝜆𝑓subscript𝐱𝑖\int\_{\mathbb{S}^{1}}k(\mathbf{x}\_{i},\mathbf{x}\_{j})p(\mathbf{x}\_{j})f(\mathbf{x}\_{j})d\mathbf{x}\_{j}=\lambda f(\mathbf{x}\_{i}). |  | (7) |

This is a homogeneous Fredholm Equation of the second kind with the non-symmetric polar kernel k​(𝐱i,𝐱j)​p​(𝐱j)𝑘subscript𝐱𝑖subscript𝐱𝑗𝑝subscript𝐱𝑗k(\mathbf{x}\_{i},\mathbf{x}\_{j})p(\mathbf{x}\_{j}). The existence of the eigenfunctions with real eigenvalues is established by symmetrizing the kernel. Let k~​(𝐱i,𝐱j)=p1/2​(𝐱i)​k​(𝐱i,𝐱j)​p1/2​(𝐱j)~𝑘subscript𝐱𝑖subscript𝐱𝑗superscript𝑝12subscript𝐱𝑖𝑘subscript𝐱𝑖subscript𝐱𝑗superscript𝑝12subscript𝐱𝑗\tilde{k}(\mathbf{x}\_{i},\mathbf{x}\_{j})=p^{1/2}(\mathbf{x}\_{i})k(\mathbf{x}\_{i},\mathbf{x}\_{j})p^{1/2}(\mathbf{x}\_{j}) and g​(𝐱)=p1/2​(𝐱)​f​(𝐱)𝑔𝐱superscript𝑝12𝐱𝑓𝐱g(\mathbf{x})=p^{1/2}(\mathbf{x})f(\mathbf{x}). Multiplying ([7](#S4.E7 "In 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) by p1/2​(𝐱i)superscript𝑝12subscript𝐱𝑖p^{1/2}(\mathbf{x}\_{i}) yields

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∫𝕊dk~​(𝐱i,𝐱j)​g​(𝐱j)​𝑑𝐱j=λ​g​(𝐱i),subscriptsuperscript𝕊𝑑~𝑘subscript𝐱𝑖subscript𝐱𝑗𝑔subscript𝐱𝑗differential-dsubscript𝐱𝑗𝜆𝑔subscript𝐱𝑖\int\_{\mathbb{S}^{d}}\tilde{k}(\mathbf{x}\_{i},\mathbf{x}\_{j})g(\mathbf{x}\_{j})d\mathbf{x}\_{j}=\lambda g(\mathbf{x}\_{i}), |  | (8) |

implying the eigenfunctions exist and λ𝜆\lambda is real.

We next parameterize the unit circle by angles, and denote by x,z

𝑥𝑧x,z any two angles. We can therefore express ([7](#S4.E7 "In 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) as

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∫x−πx+πk​(x,z)​p​(z)​f​(z)​𝑑z=λ​f​(x),superscriptsubscript𝑥𝜋𝑥𝜋𝑘𝑥𝑧𝑝𝑧𝑓𝑧differential-d𝑧𝜆𝑓𝑥\int\_{x-\pi}^{x+\pi}k(x,z)p(z)f(z)dz=\lambda f(x), |  | (9) |

where the kernel in ([5](#S3.E5 "In 3 Preliminaries ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) expressed in terms of angles reads

|  |  |  |  |
| --- | --- | --- | --- |
|  | k​(x,z)=14​π​(cos⁡(x−z)+1)​(π−|x−z|).𝑘𝑥𝑧14𝜋𝑥𝑧1𝜋𝑥𝑧k(x,z)=\frac{1}{4\pi}(\cos(x-z)+1)(\pi-|x-z|). |  | (10) |

Both p​(x)𝑝𝑥p(x) and f​(x)𝑓𝑥f(x) are periodic with a period of 2​π2𝜋2\pi since x𝑥x lies on the unit circle.

### 4.1 Explicit expressions for the eigenfunctions

Below we solve ([9](#S4.E9 "In 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) and derive an explicit expression for the eigenfunctions f​(x)𝑓𝑥f(x). Our derivation assumes that p​(x)𝑝𝑥p(x) is piecewise constant. While this assumption limits the scope of our solution, empirical results suggest that when p​(x)𝑝𝑥p(x) changes continuously the eigenfunctions are modulated continuously, consistently with our solution. We summarize:

###### Proposition 1.

Let p​(x)𝑝𝑥p(x) be a piecewise constant density function on 𝕊1superscript𝕊1\mathbb{S}^{1}. Then the eigenfunctions in ([9](#S4.E9 "In 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) take the general form

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(x)=a​(p​(x))​cos⁡(qZ​Ψ​(x)+b​(p​(x))),𝑓𝑥𝑎𝑝𝑥𝑞𝑍Ψ𝑥𝑏𝑝𝑥f(x)=a(p(x))\cos\left(\frac{q}{Z}\Psi(x)+b(p(x))\right), |  | (11) |

where q𝑞q is integer, Ψ​(x)=∫−πxp​(x~)​𝑑x~Ψ𝑥superscriptsubscript𝜋𝑥𝑝~𝑥differential-d~𝑥\Psi(x)=\int\_{-\pi}^{x}\sqrt{p(\tilde{x})}d\tilde{x} and Z=12​π​Ψ​(π)𝑍12𝜋Ψ𝜋Z=\frac{1}{2\pi}\Psi(\pi).

Note that if p​(x)=pj𝑝𝑥subscript𝑝𝑗p(x)=p\_{j} is constant in a connected region Rj⊆𝕊1subscript𝑅𝑗superscript𝕊1R\_{j}\subseteq\mathbb{S}^{1}, then ([11](#S4.E11 "In Proposition 1. ‣ 4.1 Explicit expressions for the eigenfunctions ‣ 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) can be written as

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(x)=aj​cos⁡(q​pj​xZ+bj),∀x∈Rj.formulae-sequence𝑓𝑥subscript𝑎𝑗𝑞subscript𝑝𝑗𝑥𝑍subscript𝑏𝑗for-all𝑥subscript𝑅𝑗f(x)=a\_{j}\cos\left(\frac{q\sqrt{p\_{j}}x}{Z}+b\_{j}\right),\forall x\in R\_{j}. |  | (12) |

In other words, over the region Rjsubscript𝑅𝑗R\_{j}, this is a cosine function with frequency proportional to pjsubscript𝑝𝑗\sqrt{p\_{j}}. A plot of eigenfunctions for a piecewise constant distribution is shown in Fig. [3](#S4.F3 "Figure 3 ‣ 4.1 Explicit expressions for the eigenfunctions ‣ 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density").

!(/html/2003.04560/assets/x5.png)

Figure 2: For the NTK of a two-layer network with bias we plot its eigenfunctions (in a decreasing order of eigenvalues) under a non-uniform data distribution in 𝕊1superscript𝕊1\mathbb{S}^{1}. Here we used a density composed of three constant regions with p​(x)∈3/(2​π)​{1/7,2/7,4/7}𝑝𝑥32𝜋172747p(x)\in 3/(2\pi)\{1/7,2/7,4/7\} (bottom right plot).

!(/html/2003.04560/assets/x6.png)

Figure 3: The local frequency in the eigenfunctions within each of the three constant region densities in Figure [3](#S4.F3 "Figure 3 ‣ 4.1 Explicit expressions for the eigenfunctions ‣ 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"), plotted for both a two-layer and deep (depth=10) networks (marked respectively by squares and plus signs). Measurements are obtained by applying FFT to each region. The measurements are in close match to our formula ([12](#S4.E12 "In 4.1 Explicit expressions for the eigenfunctions ‣ 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) (solid line).

The proof of the proposition relies on a lemma, proved in supplementary material, stating that the solution to ([9](#S4.E9 "In 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) satisfies the following second order ordinary differential equation (ODE)

|  |  |  |  |
| --- | --- | --- | --- |
|  | f′′​(x)=−p​(x)π​λ​f​(x).superscript𝑓′′𝑥𝑝𝑥𝜋𝜆𝑓𝑥f^{\prime\prime}(x)=-\frac{p(x)}{\pi\lambda}f(x). |  | (13) |

In a nutshell, the lemma proved by applying a sequence of six derivatives to ([9](#S4.E9 "In 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) with respect to x𝑥x, along with some algebraic manipulations, yielding a sixth order ODE for f​(x)𝑓𝑥f(x). Assuming that p​(x)𝑝𝑥p(x) is piecewise constant simplifies the ODE. Then ([13](#S4.E13 "In 4.1 Explicit expressions for the eigenfunctions ‣ 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) is obtained by restricting p​(x)𝑝𝑥p(x) to have a period of π𝜋\pi, but this restriction can be lifted by preprocessing the data in a straightforward way without changing the function that needs to be learned.

Eq. ([13](#S4.E13 "In 4.1 Explicit expressions for the eigenfunctions ‣ 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) has the following general solutions

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(x)=A​ei​Ψ​(x)π​λ​x+B​e−i​Ψ​(x)π​λ​x,𝑓𝑥𝐴superscript𝑒𝑖Ψ𝑥𝜋𝜆𝑥𝐵superscript𝑒𝑖Ψ𝑥𝜋𝜆𝑥f(x)=Ae^{i\frac{\Psi(x)}{\sqrt{\pi\lambda}}x}+Be^{-i\frac{\Psi(x)}{\sqrt{\pi\lambda}}x}, |  | (14) |

such that the derivative of ΨΨ\Psi is Ψ′​(x)=p​(x)superscriptΨ′𝑥𝑝𝑥\Psi^{\prime}(x)=\sqrt{p(x)}, resulting in real eigenfunctions of the form

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(x)=a​(p​(x))​cos⁡(Ψ​(x)π​λ​x+b​(p​(x))).𝑓𝑥𝑎𝑝𝑥Ψ𝑥𝜋𝜆𝑥𝑏𝑝𝑥f(x)=a(p(x))\cos\left(\frac{\Psi(x)}{\sqrt{\pi\lambda}}x+b(p(x))\right). |  | (15) |

As with the uniform distribution, due to periodic boundary conditions there is a countable number of eigenvalues, and those can be determined (up to scale) using the known eigenvalues for the uniform case (Basri et al., [2019](#bib.bib6)). With this we obtain

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | λ𝜆\displaystyle\lambda | =\displaystyle= | {Z2​(12​π2+18)q=0Z2​(1π2+18)q=1Z2​(q2+1)π2​(q2−1)2q≥2​evenZ2π2​q2q≥2​odd.casessuperscript𝑍212superscript𝜋218𝑞0superscript𝑍21superscript𝜋218𝑞1superscript𝑍2superscript𝑞21superscript𝜋2superscriptsuperscript𝑞212𝑞2evensuperscript𝑍2superscript𝜋2superscript𝑞2𝑞2odd\displaystyle\left\{\begin{array}[]{ll}Z^{2}\left(\frac{1}{2\pi^{2}}+\frac{1}{8}\right)&q=0\\[1.70709pt] Z^{2}\left(\frac{1}{\pi^{2}}+\frac{1}{8}\right)&q=1\\[1.70709pt] \frac{Z^{2}(q^{2}+1)}{\pi^{2}(q^{2}-1)^{2}}&q\geq 2~{}~{}\rm{even}\\[1.70709pt] \frac{Z^{2}}{\pi^{2}q^{2}}&q\geq 2~{}~{}\rm{odd.}\end{array}\right. |  | (20) |

q𝑞q is integer, and there is one eigenfunction for q=0𝑞0q=0 and two eigenfunctions for every q>0𝑞0q>0. Figure [4](#S4.F4 "Figure 4 ‣ 4.1 Explicit expressions for the eigenfunctions ‣ 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density") shows a plot of the eigenvalues computed for various densities.

!(/html/2003.04560/assets/x7.png)

Figure 4: The kernel eigenvalues for several distributions. The formula (marked by the solid lines) closely matches the eigenvalues Hpsuperscript𝐻𝑝H^{p} computed numerically using 50​K50𝐾50K points.

!(/html/2003.04560/assets/x8.png)

Figure 5: For the NTK of a two-layer network we plot the eigenvectors of Hpsuperscript𝐻𝑝H^{p} for a continuous distribution, p​(x)=3​cos⁡(2​x+π)+4.59​π𝑝𝑥32𝑥𝜋4.59𝜋p(x)=\frac{3\cos(2x+\pi)+4.5}{9\pi} (bottom right).

The amplitudes and phase shifts are determined by requiring the eigenfunctions to be continuous and differentiable everywhere. We show in supplementary material that for two neighboring regions, j,j+1

𝑗𝑗1j,j+1 it holds that if pj≤pj+1subscript𝑝𝑗subscript𝑝𝑗1p\_{j}\leq p\_{j+1} then the ratio of the amplitudes is bounded (tightly) for different values of pjsubscript𝑝𝑗p\_{j} and pj+1subscript𝑝𝑗1p\_{j+1} as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1≤ajaj+1≤pj+1pj.1subscript𝑎𝑗subscript𝑎𝑗1subscript𝑝𝑗1subscript𝑝𝑗1\leq\frac{a\_{j}}{a\_{j+1}}\leq\sqrt{\frac{p\_{j+1}}{p\_{j}}}. |  | (21) |

Figure [3](#S4.F3 "Figure 3 ‣ 4.1 Explicit expressions for the eigenfunctions ‣ 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density") shows the eigenvectors and eigenvalues for an example of a piecewise constant distribution. It can be seen that each eigenfunction consists of a piecewise sine function; i.e., the eigenfunctions in every region where p​(x)𝑝𝑥p(x) is constant form pure sine functions with frequency that changes from one region to the next. As we inspect eigenfunctions with decreasing eigenvalues we find, as our theory shows (see Figure [3](#S4.F3 "Figure 3 ‣ 4.1 Explicit expressions for the eigenfunctions ‣ 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")), that the frequencies increase in all regions, but for all eigenfunctions they maintain constant ratios that are equal to the ratios between the square roots of the corresponding densities. Finally, Figure [5](#S4.F5 "Figure 5 ‣ 4.1 Explicit expressions for the eigenfunctions ‣ 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density") shows the eigenvectors of Hpsuperscript𝐻𝑝H^{p} for a continuous distribution, showing similar behaviour to our analytic expressions.

### 4.2 Time to convergence

Determining the eigenfunctions and eigenvalues of the NTK allows us to predict the number of iterations needed to learn target functions and to understand effects due to varying densities. To understand this we consider target functions of the form g​(x)=cos⁡(κ​x)𝑔𝑥𝜅𝑥g(x)=\cos(\kappa x) where x𝑥x is drawn from a piecewise constant distribution p​(x)𝑝𝑥p(x) on 𝕊1superscript𝕊1\mathbb{S}^{1}. Denote by Rj⊆𝕊1subscript𝑅𝑗superscript𝕊1R\_{j}\subseteq\mathbb{S}^{1}, 1≤j≤l1𝑗𝑙1\leq j\leq l the regions of constant density. Loosely speaking (see Figure [7](#S4.F7 "Figure 7 ‣ 4.2 Time to convergence ‣ 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")), for each region Rjsubscript𝑅𝑗R\_{j} we expect g​(x)𝑔𝑥g(x) to correlate well with one eigenfunction (and perhaps to additional ones, but with less energy). Of these, the region corresponding to the lowest density should correlate with an eigenfunction with the smallest eigenvalue. This eigenvalue, which depends on both the target frequency κ𝜅\kappa and the density p​(x)𝑝𝑥p(x) within that region, will determine the number of iterations to convergence. This is summarized in the following theorem.

###### Theorem 1.

Let p​(x)𝑝𝑥p(x) be a piecewise constant distribution on 𝕊1superscript𝕊1\mathbb{S}^{1}. Denote by u(t)​(x)superscript𝑢𝑡𝑥u^{(t)}(x) the prediction of the network at iteration t𝑡t of GD. For any δ>0𝛿0\delta>0 the number of iterations t𝑡t needed to achieve ‖g​(x)−u(t)​(x)‖<δnorm𝑔𝑥superscript𝑢𝑡𝑥𝛿\|g(x)-u^{(t)}(x)\|<\delta is O~​(κ2/p∗)~𝑂superscript𝜅2superscript𝑝\tilde{O}(\kappa^{2}/p^{\*}), where p∗superscript𝑝p^{\*} denotes the minimal density of p​(x)𝑝𝑥p(x) in 𝕊1superscript𝕊1\mathbb{S}^{1} and O~(.)\tilde{O}(.) hides logarithmic terms.

Proving this theorem is complicated by the fact that (1) the frequency of the target function may not be exactly represented in the eigenfunctions of the kernel, due to the discrete number of eigenfunctions, and (2) the eigenfunctions restricted to any given region Rjsubscript𝑅𝑗R\_{j} are not orthogonal. These two properties may result in non-negligible correlations of g​(x)𝑔𝑥g(x) with eigenfunctions of yet smaller eigenvalues. Therefore, to prove Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 4.2 Time to convergence ‣ 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density") we first inspect the projections of g​(x)𝑔𝑥g(x) onto the eigenfunctions corresponding to such small eigenvalues and prove a bound on this tail. Subsequently we use this bound to prove the convergence rate in the theorem. The proofs are provided in the supplementary material.

!(/html/2003.04560/assets/x9.png)

!(/html/2003.04560/assets/x10.png)

!(/html/2003.04560/assets/x11.png)

!(/html/2003.04560/assets/x12.png)

!(/html/2003.04560/assets/x13.png)

Figure 6: Illustration of Thm. [1](#Thmtheorem1 "Theorem 1. ‣ 4.2 Time to convergence ‣ 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"). For a piecewise constant density with three regions (top right), a function g​(x)=sin⁡(14​x)𝑔𝑥14𝑥g(x)=\sin(14x) (in green, bottom plots) is projected onto the eigenfunctions of k𝑘k (three of which are shown with black curves in the bottom plots), producing coefficients gqsubscript𝑔𝑞g\_{q} (top left). This produces three peaks around the points predicted by our theory (marked by the dotted vertical lines), which correspond to high correlation of g​(x)𝑔𝑥g(x) with one of the three regions for the appropriate three basis functions (bottom row).

In Figure [7](#S4.F7 "Figure 7 ‣ 4.2 Time to convergence ‣ 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density") we used the target function g​(x)=sin⁡(κ​x)𝑔𝑥𝜅𝑥g(x)=\sin(\kappa x) for different values of κ𝜅\kappa to train a 2-layer network. The data was sampled from a non-uniform distribution with three constant regions of densities 3/(2​π)​(1/7,2/7,4/7)32𝜋1727473/(2\pi)(1/7,2/7,4/7). It can be seen that runtime increases for each region in proportion to κ2superscript𝜅2\kappa^{2}, and the network converged faster at denser regions (in proportion to p​(x)𝑝𝑥p(x)).

!(/html/2003.04560/assets/x14.png)

Figure 7: Convergence times as a function of the target frequency κ𝜅\kappa for a two-layer network trained with data drawn from a non-uniform distribution in 𝕊1superscript𝕊1\mathbb{S}^{1}. We used the distribution of Figure [3](#S4.F3 "Figure 3 ‣ 4.1 Explicit expressions for the eigenfunctions ‣ 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"), which is composed of three regions of constant density with a ratio of 1:2:4. For each region Rjsubscript𝑅𝑗R\_{j} the network converges at time proportional to κ2/pjsuperscript𝜅2subscript𝑝𝑗\kappa^{2}/p\_{j}, as is indicated by the three quadratic curves fit to the data points. In addition, the median ratios between our measurements for the three regions are 1:1.96:3.89, in close fit to the distribution.

### 4.3 Higher dimension

Deriving analytic expressions for data drawn from a non-uniform distribution in higher dimension, i.e., in Sd−1superscript𝑆𝑑1S^{d-1}, d>2𝑑2d>2 is challenging and is left for future work. However, simulation experiments lead us to conjecture that the main properties in the 𝕊1superscript𝕊1\mathbb{S}^{1} hold also in higher dimension, i.e., (1) the eigenfunctions for piecewise constant distributions resemble concatenated patches of spherical harmonics, (2) the frequencies of these harmonics change with density, and increase monotonically as the respective eigenvalues become smaller, and (3) learning a harmonic function of frequency k𝑘k should require O​(kd/p∗)𝑂superscript𝑘𝑑superscript𝑝O(k^{d}/p^{\*}) iterations.

Figure [8](#S5.F8 "Figure 8 ‣ 5 Deep networks ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density") shows an example plot of eigenfunctions in 𝕊2superscript𝕊2\mathbb{S}^{2} with a density function that is constant in each hemisphere. We further used harmonic functions of different frequencies to train a two-layer network with bias. Figure [9](#S5.F9 "Figure 9 ‣ 5 Deep networks ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density") shows convergence time as a function of frequency. As conjectured, for each region convergence time increases roughly in proportion to k3superscript𝑘3k^{3}, and convergence in different regions is linearly faster with density.

## 5 Deep networks

We next extend our discussion to NTK models of deep, fully connected networks. We first prove that the eigenvectors of NTK indeed characterize the convergence of GD of highly overparmeterized networks of finite width. We then empirically investigate the eigenvectors and eigenvalues of NTK for data drawn from either uniform or non-uniform distributions and show convergence times for pure sine and harmonic target functions.

We begin by showing that the eigenvectors of NTK characterize the dynamics of overparameterized FC networks of finite width. Our theorem extends Thm. 4.1 in (Arora et al., [2019b](#bib.bib4)) (see also (Cao et al., [2019](#bib.bib9))), which has dealt with two-layer networks, to deep nets. Consider a FC network of depth L𝐿L and width m𝑚m in each layer, and suppose the network is trained with n𝑛n pairs {(𝐱i,yi)}i=1nsuperscriptsubscriptsubscript𝐱𝑖subscript𝑦𝑖𝑖1𝑛\{(\mathbf{x}\_{i},y\_{i})\}\_{i=1}^{n}. Denote the vector of target values by 𝐲=(y1,…,yn)𝐲subscript𝑦1…subscript𝑦𝑛\mathbf{y}=(y\_{1},...,y\_{n}) and the network predictions for these values at time t𝑡t by 𝐮(t)superscript𝐮𝑡\mathbf{u}^{(t)}. In our theorem, Thm. [2](#Thmtheorem2 "Theorem 2. ‣ 5 Deep networks ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"), we use a slightly different model than the model stated above ([3](#S3.Ex1 "3 Preliminaries ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")). First, we assume that the first and last layers are initialized and then held fixed throughout training, and the last layer is initialized randomly ∼𝒩​(0,τ2​I)similar-toabsent𝒩0superscript𝜏2𝐼\sim\mathcal{N}(0,\tau^{2}I). The NTK for this training data is summarized in an n×n𝑛𝑛n\times n matrix H∞superscript𝐻H^{\infty}, whose entries are set to Hi​j∞=k​(𝐱i,𝐱j)subscriptsuperscript𝐻𝑖𝑗𝑘subscript𝐱𝑖subscript𝐱𝑗H^{\infty}\_{ij}=k(\mathbf{x}\_{i},\mathbf{x}\_{j}) where k𝑘k is defined in ([1](#S3.E1 "In 3 Preliminaries ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")). Let 𝐯isubscript𝐯𝑖\mathbf{v}\_{i} and λisubscript𝜆𝑖\lambda\_{i} respectively denote the eigenvectors of H∞superscript𝐻H^{\infty} and their corresponding eigenvalues. The next Theorem establishes that the convergence rate of training this deep (finite width) network depends on the decomposition of the target values 𝐲𝐲\mathbf{y} over the eigenvectors of H∞superscript𝐻H^{\infty}.

###### Theorem 2.

For any ϵ∈(0,1]italic-ϵ01\epsilon\in(0,1] and δ∈(0,O​(1L)]𝛿0𝑂1𝐿\delta\in(0,O(\frac{1}{L})], let τ=Θ​(ϵ​δ^n)𝜏Θitalic-ϵ^𝛿𝑛\tau=\Theta(\frac{\epsilon\hat{\delta}}{n}), m≥Ω​(n24​L12​log5⁡mδ8​τ6)𝑚Ωsuperscript𝑛24superscript𝐿12superscript5𝑚superscript𝛿8superscript𝜏6m\geq\Omega\left(\frac{n^{24}L^{12}\log^{5}m}{\delta^{8}\tau^{6}}\right), η=Θ​(δn4​L2​m​τ2)𝜂Θ𝛿superscript𝑛4superscript𝐿2𝑚superscript𝜏2\eta=\Theta\left(\frac{\delta}{n^{4}L^{2}m\tau^{2}}\right). Then, with probability of at least 1−δ^1^𝛿1-\hat{\delta} over the random initialization after t𝑡t GD iterations we have that

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖𝐲−𝐮(t)‖=∑i=1n(1−η​λi)2​t​(𝐯iT​𝐲)2±ϵ.norm𝐲superscript𝐮𝑡plus-or-minussuperscriptsubscript𝑖1𝑛superscript1𝜂subscript𝜆𝑖2𝑡superscriptsuperscriptsubscript𝐯𝑖𝑇𝐲2italic-ϵ\|\mathbf{y}-\mathbf{u}^{(t)}\|=\sqrt{\sum\_{i=1}^{n}(1-\eta\lambda\_{i})^{2t}(\mathbf{v}\_{i}^{T}\mathbf{y})^{2}}\,\pm\epsilon. |  | (22) |

The proof is provided in the supplementary material. Below, we give a brief proof sketch.
First, we show that for any number of layers and at any iteration t𝑡t the following relation holds

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐮(t+1)−𝐲=(I−η​H​(t))​(𝐮(t)−𝐲)+ϵ​(t),superscript𝐮𝑡1𝐲𝐼𝜂𝐻𝑡superscript𝐮𝑡𝐲italic-ϵ𝑡\displaystyle\mathbf{u}^{(t+1)}-\mathbf{y}=(I-\eta H(t))(\mathbf{u}^{(t)}-\mathbf{y})+\epsilon(t), |  | (23) |

where
Hi​j​(t)=⟨∂f​(𝐱i,𝐰​(t))∂𝐰,∂f​(𝐱j,𝐰​(t))∂𝐰⟩,subscript𝐻𝑖𝑗𝑡

𝑓subscript𝐱𝑖𝐰𝑡𝐰𝑓subscript𝐱𝑗𝐰𝑡𝐰H\_{ij}(t)=\left<\frac{\partial f(\mathbf{x}\_{i},\mathbf{w}(t))}{\partial\mathbf{w}},\frac{\partial f(\mathbf{x}\_{j},\mathbf{w}(t))}{\partial\mathbf{w}}\right>,
and the residual ϵ​(t)italic-ϵ𝑡\epsilon(t) due to the GD steps is relatively small. Then, based on several results due to (Allen-Zhu et al., [2019](#bib.bib2); Arora et al., [2019a](#bib.bib3)), we show that H​(t)𝐻𝑡H(t) can be approximated by H∞superscript𝐻H^{\infty}, yielding, by applying recursion to ([23](#S5.E23 "In 5 Deep networks ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"))

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐮(t)−𝐲=(I−η​H∞)t​(𝐮(0)−𝐲)+ξ​(t).superscript𝐮𝑡𝐲superscript𝐼𝜂superscript𝐻𝑡superscript𝐮0𝐲𝜉𝑡\mathbf{u}^{(t)}-\mathbf{y}=(I-\eta H^{\infty})^{t}(\mathbf{u}^{(0)}-\mathbf{y})+\xi(t).\\ |  | (24) |

where ‖ξ​(t)‖≤O​(ϵ)norm𝜉𝑡𝑂italic-ϵ\|\xi(t)\|\leq O(\epsilon). Next we show that under the setting of τ𝜏\tau, ‖𝐮(0)‖≤O​(ϵ)normsuperscript𝐮0𝑂italic-ϵ\|\mathbf{u}^{(0)}\|\leq O(\epsilon). Finally, by applying the spectral decomposition to H∞superscript𝐻H^{\infty} we obtain
([50](#A4.E50 "In Theorem 4. ‣ D.1 The network model ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")).

!(/html/2003.04560/assets/figures/2D_eigenfunctions.png)

Figure 8: The eigenfunctions of NTK for a two-layer network with bias for data drawn from a non-uniform distribution from 𝕊2superscript𝕊2\mathbb{S}^{2}. The left and right hemispheres each have constant density with a ratio of 12:1.

!(/html/2003.04560/assets/x15.png)

!(/html/2003.04560/assets/x16.png)

!(/html/2003.04560/assets/x17.png)

Figure 9: Convergence times as a function of the target harmonic frequency κ𝜅\kappa for a two-layer network trained with data drawn from a non-uniform distribution in 𝕊2superscript𝕊2\mathbb{S}^{2}. In each plot the sphere was divided into 2 halves, with density ratios (from left to right) of 1:2, 1:3, 1:4. The plot shows a cubic fit to the measurements. The median ratios between our measurements for the three subplots are 1.76, 2.45 and 2.99, undershooting our conjectured ratios. We believe this is due to sensitivity of experiments on 𝕊2superscript𝕊2\mathbb{S}^{2} to sampling.

!(/html/2003.04560/assets/x18.png)

Figure 10: The eigenfunctions of NTK for a deep network (depth 10) for the uniform distribution in 𝕊1superscript𝕊1\mathbb{S}^{1}. The eigenvectors are arranged according to a descending order of their corresponding eigenvalues.

Our next aim is to compute the eigenvectors and eigenvalues of NTK matrices for deep networks. This, together with Theorem [2](#Thmtheorem2 "Theorem 2. ‣ 5 Deep networks ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"), will allow us to derive convergence rates for different target functions. Toward that aim we observe that the NTK kernel k​(𝐱i,𝐱j)𝑘subscript𝐱𝑖subscript𝐱𝑗k(\mathbf{x}\_{i},\mathbf{x}\_{j}) is a function of the inner product of its arguments. This can be concluded from its recursive definition in ([6](#S3.E6 "In 3 Preliminaries ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")), since Σ(0)​(𝐱i,𝐱j)=𝐱iT​𝐱jsuperscriptΣ0subscript𝐱𝑖subscript𝐱𝑗superscriptsubscript𝐱𝑖𝑇subscript𝐱𝑗\Sigma^{(0)}(\mathbf{x}\_{i},\mathbf{x}\_{j})=\mathbf{x}\_{i}^{T}\mathbf{x}\_{j}; both Σ(h)​(𝐱i,𝐱j)superscriptΣℎsubscript𝐱𝑖subscript𝐱𝑗\Sigma^{(h)}(\mathbf{x}\_{i},\mathbf{x}\_{j}) and Σ˙(h)​(𝐱i,𝐱j)superscript˙Σℎsubscript𝐱𝑖subscript𝐱𝑗\dot{\Sigma}^{(h)}(\mathbf{x}\_{i},\mathbf{x}\_{j}) are (scaled) expectations over random variables drawn from a zero normal distribution and whose covariance, by recursion, is a function of the inner product 𝐱iT​𝐱jsuperscriptsubscript𝐱𝑖𝑇subscript𝐱𝑗\mathbf{x}\_{i}^{T}\mathbf{x}\_{j}. Consequently, the kernel decomposes over the zonal spherical harmonics in 𝕊d−1superscript𝕊𝑑1\mathbb{S}^{d-1} (or Fourier series in 𝕊1superscript𝕊1\mathbb{S}^{1}), and for training data drawn from the uniform distribution the corresponding kernel matrix forms a convolution.

Figure [10](#S5.F10 "Figure 10 ‣ 5 Deep networks ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density") shows for the NTK of depth 10 that indeed the eigenvectors in 𝕊1superscript𝕊1\mathbb{S}^{1} is the Fourier series. We note that despite the lack of bias terms all the Fourier components are included. The eigenvalues decrease monotonically with frequency, indicating that the network should learn low frequency functions faster than high frequency ones. Moreover, as Figures [12](#S5.F12 "Figure 12 ‣ 5 Deep networks ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density") and [12](#S5.F12 "Figure 12 ‣ 5 Deep networks ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density") show, regardless of depth, when trained with a function of frequency κ𝜅\kappa overparameterized networks converge respectively at the asymptotic speed of O​(κ2)𝑂superscript𝜅2O(\kappa^{2}) and O​(κ3)𝑂superscript𝜅3O(\kappa^{3}) for uniform data in 𝕊1superscript𝕊1\mathbb{S}^{1} and 𝕊2superscript𝕊2\mathbb{S}^{2}. Interestingly, however, the eigenvalues of NTK reveal a difference in the way deep and shallow networks treat low frequencies in the target function, as is refelcted by the plots in Figure [12](#S5.F12 "Figure 12 ‣ 5 Deep networks ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"). Each line of one color represents the log of the eigenvalues for one network and the lines are ordered from shallow to deep in ascending order. The local slope of these lines indicate the speed of convergence for the corresponding frequencies. Asymptotically all the lines become parallel as the frequency κ𝜅\kappa increases, implying that the asymptotic convergence times should be equal for all depths. However, for the low frequencies the lines corresponding to deeper networks are flatter than those corresponding to shallow networks. This flatter slope indicates that the frequency bias for such frequencies is smaller, implying that deep networks learn frequencies, e.g., 6-10, almost as fast as 1-5, while this is not true for shallow networks.

!(/html/2003.04560/assets/x19.png)

!(/html/2003.04560/assets/x20.png)

Figure 11: For deep networks (3 and 7 layers) and data drawn from the uniform distribution in 𝕊1superscript𝕊1\mathbb{S}^{1} (left) and 𝕊2superscript𝕊2\mathbb{S}^{2} (right) we plot training times as a function of target frequency (marked by the solid blue circles). This is compared to the times predicted by the eigenvalues of the corresponding NTK model (red circles).

!(/html/2003.04560/assets/x21.png)

!(/html/2003.04560/assets/x22.png)

Figure 12: This figure shows a plot of the eigenvalues of NTK for FC networks of different depths with points drawn from a uniform density in 𝕊1superscript𝕊1\mathbb{S}^{1} (left) and 𝕊2superscript𝕊2\mathbb{S}^{2} (right). The plot is given in log-log scale. Networks of different depths are colored differently. Plots for deeper networks appear higher due to scaling. It can be seen that all curves decrease monotonically, indicating that the eigenvalues decay with frequency. In addition they all become parallel as the frequency κ𝜅\kappa grows, converging to a slope of -2 for 𝕊1superscript𝕊1\mathbb{S}^{1} and -3 for 𝕊2superscript𝕊2\mathbb{S}^{2} (fitting the curves in the left plot starting at κ=50𝜅50\kappa=50 yields a slope of 1.94; fitting the right plot starting at κ=10𝜅10\kappa=10 yields a slope of 2.80). This indicates that asymptotically the rate of learning a frequency κ𝜅\kappa is O​(κ2)𝑂superscript𝜅2O(\kappa^{2}) and O​(κ3)𝑂superscript𝜅3O(\kappa^{3}) respectively regardless of depth. The shallower slope of deep networks on the left part of each plot indicates that middle frequencies are learned faster with deep networks than with shallow ones.

Finally, for data drawn from a non-uniform distribution the eigenfunctions of NTK for deep networks appear to be indistinguishable from those obtained for two-layer networks. Figure [3](#S4.F3 "Figure 3 ‣ 4.1 Explicit expressions for the eigenfunctions ‣ 4 The eigenfunctions of NTK for two-layer networks for non-uniform distributions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density") shows a plot of the local frequencies obtained with NTK for a network of depth 10. It can be seen that the local frequencies are identical to those obtained with NTK for a two-layer network. The eigenvalues are similar to those obtained with the uniform density, up to a normalizing scale which depends on the distribution. Similarly to the two-layer case, learning a harmonic function of frequency κ𝜅\kappa is therefore expected to require O​(κd/p∗)𝑂superscript𝜅𝑑superscript𝑝O(\kappa^{d}/p^{\*}) iterations.

## 6 Conclusion

The main contribution of our work is to show that insights about neural networks that have been derived with the assumption of uniformly distributed training data also apply, in interesting ways, to more realistic, non-uniform data. Prior work has shown that the Neural Tangent Kernel provides a model of real, overparameterized neural networks that is tractable to analyze and that matches real experiments. Our work shows that NTK has a frequency bias for non-uniform data distributions as well as for uniform ones. This strengthens the case that this frequency bias may play an important role in real neural networks.

We also quantify this frequency bias. We derive an expression for the eigenfunctions of NTK, showing that for piecewise constant data distributions the eigenfunctions consist of piecewise harmonic functions. The frequency of these piecewise functions increases linearly with the square root of the local density of the data. As a consequence, for 1D inputs, networks modeled by NTK learn harmonic functions with a speed that increases quadratically in their frequency and decreases linearly with the local density. Experiments indicate that these results generalize naturally to higher dimensions. These results support the idea that overparameterized networks avoid overfitting because they fit target functions with smooth functions, and are slow to add high frequency components that could overfit.

## Acknowledgements

This material is based partly upon work supported by the National Science Foundation under Grant No. DMS1439786 while the authors were in residence at the Institute for Computational and Experimental
Research in Mathematics in Providence, RI, during the Computer Vision program. We would like to thank the Quantifying Ensemble Diversity for Robust Machine Learning (QED for RML) program from DARPA for their support of this project.

## References

* Allen-Zhu et al. (2018)

  Allen-Zhu, Z., Li, Y., and Song, Z.
  On the convergence rate of training recurrent neural networks.
  In *33rd Conference on Neural Information Processing Systems
  (NeurIPS 2019)*, 2018.
* Allen-Zhu et al. (2019)

  Allen-Zhu, Z., Li, Y., and Song, Z.
  A convergence theory for deep learning via over-parameterization.
  In Chaudhuri, K. and Salakhutdinov, R. (eds.), *Proceedings of
  the 36th International Conference on Machine Learning*, volume 97 of
  *Proceedings of Machine Learning Research*, pp.  242–252, 2019.
* Arora et al. (2019a)

  Arora, S., Du, S. S., Hu, W., Li, Z., Salakhutdinov, R., and Wang, R.
  On exact computation with an infinitely wide neural net.
  In *NeurIPS*, 2019a.
* Arora et al. (2019b)

  Arora, S., Du, S. S., Hu, W., Li, Z., and Wang, R.
  Fine-grained analysis of optimization and generalization for
  overparameterized two-layer neural networks.
  *arXiv preprint arXiv:1901.08584*, 2019b.
* Bach (2017)

  Bach, F.
  Breaking the curse of dimensionality with convex neural networks.
  *Journal of Machine Learning Research*, 18:1–53,
  2017.
* Basri et al. (2019)

  Basri, R., Jacobs, D., Kasten, Y., and Kritchman, S.
  The convergence rate of neural networks for learned functions of
  different frequencies.
  In *NeurIPS*, 2019.
* Bietti & Mairal (2019)

  Bietti, A. and Mairal, J.
  On the inductive bias of neural tangent kernels.
  In *NeurIPS*, 2019.
* Canu & Elisseef (1999)

  Canu, M. F. and Elisseef, A.
  Regularization , kernels and sigmoid netst.
  In *INSA, Rouen*, 1999.
* Cao et al. (2019)

  Cao, Y., Fang, Z., Wu, Y., Zhou, D.-X., and Gu, Q.
  Towards understanding the spectral bias of deep learning, 2019.
* Chizat et al. (2019)

  Chizat, L., Oyallon, E., and Bach, F.
  On lazy training in differentiable programming.
  In *Advances in Neural Information Processing Systems*, 2019.
* Du et al. (2019)

  Du, S. S., Zhai, X., Poczos, B., and Singh, A.
  Gradient descent provably optimizes over-parameterized neural
  networks.
  *International Conference on Learning Representations (ICLR)*,
  2019.
* Farnia et al. (2018)

  Farnia, F., Zhang, J., and Tse, D.
  A spectral approach to generalization and optimization in neural
  networks.
  2018.
* Goel & Klivans (2017)

  Goel, S. and Klivans, A. R.
  Eigenvalue decay implies polynomial-time learnability for neural
  networks.
  In *NIPS*, 2017.
* Huang & Yau (2019)

  Huang, J. and Yau, H.-T.
  Dynamics of deep neural networks and neural tangent hierarchy.
  *arXiv preprint arXiv:1909.08156*, 2019.
* Jacot et al. (2018)

  Jacot, A., Gabriel, F., and Hongler, C.
  Neural tangent kernel: Convergence and generalization in neural
  networks.
  In *Proceedings of the 32nd International Conference on Neural
  Information Processing Systems*, pp.  8580–8589, 2018.
* Lee et al. (2019)

  Lee, J., Xiao, L., Schoenholz, S., Bahri, Y., Novak, R., Sohl-Dickstein, J.,
  and Pennington, J.
  Wide neural networks of any depth evolve as linear models under
  gradient descent.
  In *Advances in Neural Information Processing Systems*, pp. 8570–8581, 2019.
* Liang & Lee (2013)

  Liang, Z. and Lee, Y.
  Eigen-analysis of nonlinear pca with polynomial kernels.
  *Statistical Analysis and Data Mining: The ASA Data Science
  Journal*, 6(6):529–544, 2013.
* Novak et al. (2019)

  Novak, R., Xiao, L., Bahri, Y., Lee, J., Yang, G., Hron, J., Abolafia, D. A.,
  Pennington, J., and Sohl-Dickstein, J.
  Bayesian deep convolutional networks with many channels are gaussian
  processes.
  In *7th International Conference on Learning Representations,
  ICLR 2019, New Orleans, LA, USA, May 6-9, 2019*, 2019.
* Rahaman et al. (2019)

  Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F.,
  Bengio, Y., and Courville, A.
  On the spectral bias of neural networks.
  In Chaudhuri, K. and Salakhutdinov, R. (eds.), *Proceedings of
  the 36th International Conference on Machine Learning*, volume 97 of
  *Proceedings of Machine Learning Research*, pp.  5301–5310. PMLR,
  2019.
* Shawe-Taylor et al. (2005)

  Shawe-Taylor, J., Williams, C. K., Cristianini, N., and Kandola, J.
  On the eigenspectrum of the gram matrix and the generalization error
  of kernel-pca.
  *IEEE Trans. Inf. Theor.*, 51(7):2510–2522, 2005.
* Vempala & Wilmes (2018)

  Vempala, S. S. and Wilmes, J.
  Gradient descent for one-hidden-layer neural networks: Polynomial
  convergence and sq lower bounds.
  In *COLT*, 2018.
* Williams & Seeger (2000)

  Williams, C. and Seeger, M.
  The effect of the input density distribution on kernel-based
  classifiers.
  In *Proceedings of the 17th International Conference on Machine
  Learning*, pp.  1159–1166, 2000.
* Woodworth et al. (2019)

  Woodworth, B. E., Gunasekar, S., Lee, J. D., Soudry, D., and Srebro, N.
  Kernel and deep regimes in overparametrized models.
  *CoRR*, abs/1906.05827, 2019.
* Xie et al. (2017)

  Xie, B., Liang, Y., and Song, L.
  Diverse neural network learns true target functions.
  In *International Conference on Artificial Intelligence and
  Statistics (AISTATS), Fort Lauderdale, Florida*, pp.  1216–1224, 2017.
* Xu et al. (2019)

  Xu, Z. J., Zhang, Y., Luo, T., Xiao, Y., and Ma, Z.
  Frequency principle: Fourier analysis sheds light on deep neural
  networks.
  *CoRR*, abs/1901.06523, 2019.

## Appendix A Eigenfunctions of NTK for a two layer-network for data drawn from a piecewise constant distribution

###### Lemma 1.

Let p​(x)𝑝𝑥p(x) be a piecewise constant density function on 𝕊1superscript𝕊1\mathbb{S}^{1}. Then the eigenfunctions in Eq. (9) in the paper satisfy the following ordinary differential equation

|  |  |  |  |
| --- | --- | --- | --- |
|  | f′′​(x)=−p​(x)π​λ​f​(x).superscript𝑓′′𝑥𝑝𝑥𝜋𝜆𝑓𝑥f^{\prime\prime}(x)=-\frac{p(x)}{\pi\lambda}f(x). |  | (25) |

###### Proof.

Combining Eqs. (9) and (10) in the paper we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∫x−πx+π(1+cos⁡(z−x))​(π−|z−x|)​f​(z)​p​(z)​𝑑z=4​π​λ​f​(x)superscriptsubscript𝑥𝜋𝑥𝜋1𝑧𝑥𝜋𝑧𝑥𝑓𝑧𝑝𝑧differential-d𝑧4𝜋𝜆𝑓𝑥\int\_{x-\pi}^{x+\pi}(1+\cos(z-x))(\pi-|z-x|)f(z)p(z)dz=4\pi\lambda f(x) |  | (26) |

Below we take six derivatives of ([26](#A1.E26 "In Proof. ‣ Appendix A Eigenfunctions of NTK for a two layer-network for data drawn from a piecewise constant distribution ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) with respect to x𝑥x. We use parenthesized superscripts f(n)​(x)superscript𝑓𝑛𝑥f^{(n)}(x) to denote the nthsuperscript𝑛thn^{\mathrm{th}} derivative of f𝑓f at x𝑥x. First derivative

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 4​π​λ​f(1)​(x)4𝜋𝜆superscript𝑓1𝑥\displaystyle 4\pi\lambda f^{(1)}(x) | =\displaystyle= | −∫x−πx(1+cos⁡(z−x)−(π+z−x)​sin⁡(z−x))​f​(z)​p​(z)​𝑑zsuperscriptsubscript𝑥𝜋𝑥1𝑧𝑥𝜋𝑧𝑥𝑧𝑥𝑓𝑧𝑝𝑧differential-d𝑧\displaystyle-\int\_{x-\pi}^{x}\left(1+\cos(z-x)-(\pi+z-x)\sin(z-x)\right)f(z)p(z)dz |  |
|  |  |  | +∫xx+π(1+cos⁡(z−x)+(π−z+x)​sin⁡(z−x))​f​(z)​p​(z)​𝑑zsuperscriptsubscript𝑥𝑥𝜋1𝑧𝑥𝜋𝑧𝑥𝑧𝑥𝑓𝑧𝑝𝑧differential-d𝑧\displaystyle+\int\_{x}^{x+\pi}\left(1+\cos(z-x)+(\pi-z+x)\sin(z-x)\right)f(z)p(z)dz |  |

Second derivative

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 4​π​λ​f(2)​(x)+4​f​(x)​p​(x)4𝜋𝜆superscript𝑓2𝑥4𝑓𝑥𝑝𝑥\displaystyle 4\pi\lambda f^{(2)}(x)+4f(x)p(x) | =\displaystyle= | −∫x−πx(2​sin⁡(z−x)+(π+z−x)​cos⁡(z−x))​f​(z)​p​(z)​𝑑zsuperscriptsubscript𝑥𝜋𝑥2𝑧𝑥𝜋𝑧𝑥𝑧𝑥𝑓𝑧𝑝𝑧differential-d𝑧\displaystyle-\int\_{x-\pi}^{x}\left(2\sin(z-x)+(\pi+z-x)\cos(z-x)\right)f(z)p(z)dz |  |
|  |  |  | +∫xx+π(2​sin⁡(z−x)−(π−z+x)​cos⁡(z−x))​f​(z)​p​(z)​𝑑zsuperscriptsubscript𝑥𝑥𝜋2𝑧𝑥𝜋𝑧𝑥𝑧𝑥𝑓𝑧𝑝𝑧differential-d𝑧\displaystyle+\int\_{x}^{x+\pi}\left(2\sin(z-x)-(\pi-z+x)\cos(z-x)\right)f(z)p(z)dz |  |

Adding this to ([26](#A1.E26 "In Proof. ‣ Appendix A Eigenfunctions of NTK for a two layer-network for data drawn from a piecewise constant distribution ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"))

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | 4​π​λ​f(2)​(x)+4​f​(x)​p​(x)+4​π​λ​f​(x)4𝜋𝜆superscript𝑓2𝑥4𝑓𝑥𝑝𝑥4𝜋𝜆𝑓𝑥\displaystyle 4\pi\lambda f^{(2)}(x)+4f(x)p(x)+4\pi\lambda f(x) | =\displaystyle= | ∫x−πx(π+z−x−2​sin⁡(z−x))​f​(z)​p​(z)​𝑑zsuperscriptsubscript𝑥𝜋𝑥𝜋𝑧𝑥2𝑧𝑥𝑓𝑧𝑝𝑧differential-d𝑧\displaystyle\int\_{x-\pi}^{x}\left(\pi+z-x-2\sin(z-x)\right)f(z)p(z)dz |  | (27) |
|  |  |  | +∫xx+π(π−z+x+2​sin⁡(z−x))​f​(z)​p​(z)​𝑑zsuperscriptsubscript𝑥𝑥𝜋𝜋𝑧𝑥2𝑧𝑥𝑓𝑧𝑝𝑧differential-d𝑧\displaystyle+\int\_{x}^{x+\pi}\left(\pi-z+x+2\sin(z-x)\right)f(z)p(z)dz |  |

Third derivative

|  |  |  |
| --- | --- | --- |
|  | 4​π​λ​f(3)​(x)+4​π​λ​f(1)​(x)+4​f(1)​(x)​p​(x)+4​f​(x)​p(1)​(x)=4𝜋𝜆superscript𝑓3𝑥4𝜋𝜆superscript𝑓1𝑥4superscript𝑓1𝑥𝑝𝑥4𝑓𝑥superscript𝑝1𝑥absent\displaystyle 4\pi\lambda f^{(3)}(x)+4\pi\lambda f^{(1)}(x)+4f^{(1)}(x)p(x)+4f(x)p^{(1)}(x)= |  |
|  |  |  |
| --- | --- | --- |
|  | ∫x−πx(2​cos⁡(z−x)−1)​f​(z)​p​(z)​𝑑z−∫xx+π(2​cos⁡(z−x)−1)​f​(z)​p​(z)​𝑑zsuperscriptsubscript𝑥𝜋𝑥2𝑧𝑥1𝑓𝑧𝑝𝑧differential-d𝑧superscriptsubscript𝑥𝑥𝜋2𝑧𝑥1𝑓𝑧𝑝𝑧differential-d𝑧\displaystyle\int\_{x-\pi}^{x}\left(2\cos(z-x)-1\right)f(z)p(z)dz-\int\_{x}^{x+\pi}\left(2\,\cos(z-x)-1\right)f(z)p(z)dz |  |

Fourth derivative

|  |  |  |
| --- | --- | --- |
|  | 4​π​λ​f(4)​(x)+4​π​λ​f(2)​(x)+4​f(2)​(x)​p​(x)+8​f(1)​(x)​p(1)​(x)+4​f​(x)​p(2)​(x)−2​f​(x)​p​(x)=4𝜋𝜆superscript𝑓4𝑥4𝜋𝜆superscript𝑓2𝑥4superscript𝑓2𝑥𝑝𝑥8superscript𝑓1𝑥superscript𝑝1𝑥4𝑓𝑥superscript𝑝2𝑥2𝑓𝑥𝑝𝑥absent\displaystyle 4\pi\lambda f^{(4)}(x)+4\pi\lambda f^{(2)}(x)+4f^{(2)}(x)p(x)+8f^{(1)}(x)p^{(1)}(x)+4f(x)p^{(2)}(x)-2f(x)p(x)= |  |
|  |  |  |
| --- | --- | --- |
|  | 3​f​(x−π)​p​(x−π)+3​f​(x+π)​p​(x+π)−∫xx+π2​sin⁡(z−x)​f​(z)​p​(z)​𝑑z+∫x−πx2​sin⁡(z−x)​f​(z)​p​(z)​𝑑z3𝑓𝑥𝜋𝑝𝑥𝜋3𝑓𝑥𝜋𝑝𝑥𝜋superscriptsubscript𝑥𝑥𝜋2𝑧𝑥𝑓𝑧𝑝𝑧differential-d𝑧superscriptsubscript𝑥𝜋𝑥2𝑧𝑥𝑓𝑧𝑝𝑧differential-d𝑧\displaystyle 3f(x-\pi)p(x-\pi)+3f(x+\pi)p(x+\pi)-\int\_{x}^{x+\pi}2\sin(z-x)f(z)p(z)dz+\int\_{x-\pi}^{x}2\sin(z-x)f(z)p(z)dz |  |

Adding this to ([27](#A1.E27 "In Proof. ‣ Appendix A Eigenfunctions of NTK for a two layer-network for data drawn from a piecewise constant distribution ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"))

|  |  |  |
| --- | --- | --- |
|  | 4​π​λ​f(4)​(x)+8​π​λ​f(2)​(x)+4​π​λ​f​(x)+2​f​(x)​p​(x)+4​p​(x)​f(2)​(x)+8​f(1)​(x)​p(1)​(x)+4​f​(x)​p(2)​(x)=4𝜋𝜆superscript𝑓4𝑥8𝜋𝜆superscript𝑓2𝑥4𝜋𝜆𝑓𝑥2𝑓𝑥𝑝𝑥4𝑝𝑥superscript𝑓2𝑥8superscript𝑓1𝑥superscript𝑝1𝑥4𝑓𝑥superscript𝑝2𝑥absent\displaystyle 4\pi\lambda f^{(4)}(x)+8\pi\lambda f^{(2)}(x)+4\pi\lambda f(x)+2f(x)p(x)+4p(x)f^{(2)}(x)+8f^{(1)}(x)p^{(1)}(x)+4f(x)p^{(2)}(x)= |  |
|  |  |  |
| --- | --- | --- |
|  | 3​f​(x−π)​p​(x−π)+3​f​(x+π)​p​(x+π)+∫xx+π(π−z+x)​f​(z)​p​(z)​𝑑z+∫x−πx(π+z−x)​f​(z)​p​(z)​𝑑z3𝑓𝑥𝜋𝑝𝑥𝜋3𝑓𝑥𝜋𝑝𝑥𝜋superscriptsubscript𝑥𝑥𝜋𝜋𝑧𝑥𝑓𝑧𝑝𝑧differential-d𝑧superscriptsubscript𝑥𝜋𝑥𝜋𝑧𝑥𝑓𝑧𝑝𝑧differential-d𝑧\displaystyle 3f(x-\pi)p(x-\pi)+3f(x+\pi)p(x+\pi)+\int\_{x}^{x+\pi}(\pi-z+x)f(z)p(z)dz+\int\_{x-\pi}^{x}(\pi+z-x)f(z)p(z)dz |  |

Fifth derivative

|  |  |  |
| --- | --- | --- |
|  | 4​π​λ​f(5)​(x)+8​π​λ​f(3)​(x)+4​π​λ​f(1)​(x)+4​f(3)​(x)​p​(x)+f(2)​(x)​p(1)​(x)+12​f(1)​(x)+p(2)​(x)4𝜋𝜆superscript𝑓5𝑥8𝜋𝜆superscript𝑓3𝑥4𝜋𝜆superscript𝑓1𝑥4superscript𝑓3𝑥𝑝𝑥superscript𝑓2𝑥superscript𝑝1𝑥12superscript𝑓1𝑥superscript𝑝2𝑥\displaystyle 4\pi\lambda f^{(5)}(x)+8\pi\lambda f^{(3)}\left(x\right)+4\pi\lambda f^{(1)}(x)+4f^{(3)}(x)p(x)+f^{(2)}(x)p^{(1)}(x)+12f^{(1)}(x)+p^{(2)}(x) |  |
|  |  |  |
| --- | --- | --- |
|  | +2​f(1)​(x)​p​(x)+4​f​(x)​p(3)​(x)=−2​f​(x)​p(2)​(x)+3​f(1)​(x−π)​p​(x−π)+3​f​(x−π)​p(1)​(x−π)2superscript𝑓1𝑥𝑝𝑥4𝑓𝑥superscript𝑝3𝑥2𝑓𝑥superscript𝑝2𝑥3superscript𝑓1𝑥𝜋𝑝𝑥𝜋3𝑓𝑥𝜋superscript𝑝1𝑥𝜋\displaystyle+2f^{(1)}(x)p(x)+4f(x)p^{(3)}(x)=-2f(x)p^{(2)}(x)+3f^{(1)}(x-\pi)p(x-\pi)+3f(x-\pi)p^{(1)}(x-\pi) |  |
|  |  |  |
| --- | --- | --- |
|  | +3​f(1)​(x+π)​p​(x+π)+3​f​(x+π)​p(1)​(x+π)−∫x−πxf​(z)​p​(z)​𝑑z+∫xx+πf​(z)​p​(z)​𝑑z3superscript𝑓1𝑥𝜋𝑝𝑥𝜋3𝑓𝑥𝜋superscript𝑝1𝑥𝜋superscriptsubscript𝑥𝜋𝑥𝑓𝑧𝑝𝑧differential-d𝑧superscriptsubscript𝑥𝑥𝜋𝑓𝑧𝑝𝑧differential-d𝑧\displaystyle+3f^{(1)}(x+\pi)p(x+\pi)+3f(x+\pi)p^{(1)}(x+\pi)-\int\_{x-\pi}^{x}f(z)p(z)dz+\int\_{x}^{x+\pi}f(z)p(z)dz |  |

Sixth derivative

|  |  |  |
| --- | --- | --- |
|  | 4​π​λ​f(6)​(x)+8​π​λ​f(4)​(x)+4​π​λ​f(2)​(x)=3​f(2)​(x+π)​p​(x+π)+3​p(2)​(x+π)​f​(x+π)4𝜋𝜆superscript𝑓6𝑥8𝜋𝜆superscript𝑓4𝑥4𝜋𝜆superscript𝑓2𝑥3superscript𝑓2𝑥𝜋𝑝𝑥𝜋3superscript𝑝2𝑥𝜋𝑓𝑥𝜋\displaystyle 4\pi\lambda f^{(6)}(x)+8\pi\lambda f^{(4)}(x)+4\pi\lambda f^{(2)}(x)=3f^{(2)}(x+\pi)p\left(x+\pi\right)+3p^{(2)}(x+\pi)f(x+\pi) |  |
|  |  |  |
| --- | --- | --- |
|  | +6​f(1)​(x+π)​p(1)​(x+π)−2​f​(x)​p​(x)+f​(x−π)​p​(x−π)−4​f​(x)​p(4)​(x)−4​p​(x)​f(4)​(x)6superscript𝑓1𝑥𝜋superscript𝑝1𝑥𝜋2𝑓𝑥𝑝𝑥𝑓𝑥𝜋𝑝𝑥𝜋4𝑓𝑥superscript𝑝4𝑥4𝑝𝑥superscript𝑓4𝑥\displaystyle+6f^{(1)}(x+\pi)p^{(1)}(x+\pi)-2f(x)p(x)+f(x-\pi)p(x-\pi)-4f(x)p^{(4)}(x)-4p(x)f^{(4)}\left(x\right) |  |
|  |  |  |
| --- | --- | --- |
|  | −2​f​(x)​p(2)​(x)−2​p​(x)​f(2)​(x)+f​(x+π)​p​(x+π)+6​f(1)​(x−π)​p(1)​(x−π)+3​f(2)​(x−π)​p​(x−π)2𝑓𝑥superscript𝑝2𝑥2𝑝𝑥superscript𝑓2𝑥𝑓𝑥𝜋𝑝𝑥𝜋6superscript𝑓1𝑥𝜋superscript𝑝1𝑥𝜋3superscript𝑓2𝑥𝜋𝑝𝑥𝜋\displaystyle-2f(x)p^{(2)}(x)-2\,p(x)\,f^{(2)}(x)+f(x+\pi)p(x+\pi)+6f^{(1)}(x-\pi)p^{(1)}(x-\pi)+3f^{(2)}(x-\pi)p(x-\pi) |  |
|  |  |  |
| --- | --- | --- |
|  | +3​p(2)​(x−π)​f​(x−π)−16​f(1)​(x)​p(3)​(x)−16​f(3)​(x)​p(1)​(x)−24​p(2)​(x)​f(2)​(x)−4​f(1)​(x)​p(1)​(x)3superscript𝑝2𝑥𝜋𝑓𝑥𝜋16superscript𝑓1𝑥superscript𝑝3𝑥16superscript𝑓3𝑥superscript𝑝1𝑥24superscript𝑝2𝑥superscript𝑓2𝑥4superscript𝑓1𝑥superscript𝑝1𝑥\displaystyle+3p^{(2)}(x-\pi)f(x-\pi)-16f^{(1)}(x)p^{(3)}(x)-16f^{(3)}(x)p^{(1)}(x)-24p^{(2)}(x)f^{(2)}(x)-4f^{(1)}(x)p^{(1)}(x) |  |

Next, we simplify and rearrange. We omit dependence on x𝑥x, note that f​(x−π)=f​(x+π)𝑓𝑥𝜋𝑓𝑥𝜋f(x-\pi)=f(x+\pi) and p​(x−π)=p​(x+π)𝑝𝑥𝜋𝑝𝑥𝜋p(x-\pi)=p(x+\pi) and respectively denote them by f¯¯𝑓\bar{f} and p¯¯𝑝\bar{p}.

|  |  |  |
| --- | --- | --- |
|  | 2​π​λ​f(6)+2​(p+2​π​λ)​f(4)+8​p(1)​f(3)+(p+12​p(2)+2​π​λ)​f(2)+2𝜋𝜆superscript𝑓62𝑝2𝜋𝜆superscript𝑓48superscript𝑝1superscript𝑓3limit-from𝑝12superscript𝑝22𝜋𝜆superscript𝑓2\displaystyle 2\pi\lambda f^{(6)}+2(p+2\pi\lambda)f^{(4)}+8p^{(1)}f^{(3)}+(p+12p^{(2)}+2\pi\lambda)f^{(2)}+ |  |
|  |  |  |
| --- | --- | --- |
|  | 2​(p(1)+4​p(3))​f(1)+(p+p(2)+2​p(4))​f=(p¯+3​p¯(2))​f¯+6​p¯(1)​f¯(1)+3​p¯​f¯(2)2superscript𝑝14superscript𝑝3superscript𝑓1𝑝superscript𝑝22superscript𝑝4𝑓¯𝑝3superscript¯𝑝2¯𝑓6superscript¯𝑝1superscript¯𝑓13¯𝑝superscript¯𝑓2\displaystyle 2(p^{(1)}+4p^{(3)})f^{(1)}+(p+p^{(2)}+2p^{(4)})f=(\bar{p}+3\bar{p}^{(2)})\bar{f}+6\bar{p}^{(1)}\bar{f}^{(1)}+3\bar{p}\bar{f}^{(2)} |  |

Assume next that p​(x)𝑝𝑥p(x) is constant around x𝑥x and x−π𝑥𝜋x-\pi, so its derivatives at these points vanish. Then,

|  |  |  |
| --- | --- | --- |
|  | 2​π​λ​f(6)+(2​p+4​π​λ)​f(4)+(p+2​π​λ)​f(2)+p​f=p¯​f¯+3​p¯​f¯(2)2𝜋𝜆superscript𝑓62𝑝4𝜋𝜆superscript𝑓4𝑝2𝜋𝜆superscript𝑓2𝑝𝑓¯𝑝¯𝑓3¯𝑝superscript¯𝑓2\displaystyle 2\pi\lambda f^{(6)}+(2p+4\pi\lambda)f^{(4)}+(p+2\pi\lambda)f^{(2)}+pf=\bar{p}\bar{f}+3\bar{p}\bar{f}^{(2)} |  |

We next make the assumption that p​(x)𝑝𝑥p(x) has a period of π𝜋\pi (so p=p¯𝑝¯𝑝p=\bar{p}) in which case f​(x+π)=−f​(x)𝑓𝑥𝜋𝑓𝑥f(x+\pi)=-f(x) (i.e., f¯=−f¯𝑓𝑓\bar{f}=-f). These assumptions will be removed later. With these assumptions we have

|  |  |  |
| --- | --- | --- |
|  | 2​π​λ​f(6)+(2​p+4​π​λ)​f(4)+(4​p+2​π​λ)​f(2)+2​p​f=02𝜋𝜆superscript𝑓62𝑝4𝜋𝜆superscript𝑓44𝑝2𝜋𝜆superscript𝑓22𝑝𝑓0\displaystyle 2\pi\lambda f^{(6)}+(2p+4\pi\lambda)f^{(4)}+(4p+2\pi\lambda)f^{(2)}+2pf=0 |  |

It can be readily verified that this equation is solved by ([25](#A1.E25 "In Lemma 1. ‣ Appendix A Eigenfunctions of NTK for a two layer-network for data drawn from a piecewise constant distribution ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")).

Finally, if p​(x)𝑝𝑥p(x) does not have a period of π𝜋\pi we can preprocess the data in a straightforward way to make p𝑝p have a period of π𝜋\pi (by mapping the interval [0,4​π)04𝜋[0,4\pi) to [0,2​π)02𝜋[0,2\pi)) without changing the function that needs to be learned.
∎

## Appendix B The amplitudes of the eigenfunctions in different regions

In this section for the NTK of a 2-layer network for which only the first layer is trained we compute bounds on the amplitudes of its eigenfunctions. We first bound the ratios between the amplitudes in two neighboring regions, and use this in the following section to bound the amplitude in any one region.

### B.1 Ratios between the amplitudes of neighboring regions

If p​(x)=pj𝑝𝑥subscript𝑝𝑗p(x)=p\_{j} is constant in each region Rj⊆𝕊1subscript𝑅𝑗superscript𝕊1R\_{j}\subseteq\mathbb{S}^{1}, 1≤j≤l1𝑗𝑙1\leq j\leq l, then the eigenfunction or order q𝑞q fq​(x)subscript𝑓𝑞𝑥f\_{q}(x) for x∈Rj𝑥subscript𝑅𝑗x\in R\_{j} can be written as

|  |  |  |
| --- | --- | --- |
|  | fq​(x)=aj​cos⁡(q​pj​xZ+bj)subscript𝑓𝑞𝑥subscript𝑎𝑗𝑞subscript𝑝𝑗𝑥𝑍subscript𝑏𝑗f\_{q}(x)=a\_{j}\cos\left(\frac{q\sqrt{p\_{j}}x}{Z}+b\_{j}\right) |  |

where aj≥0subscript𝑎𝑗0a\_{j}\geq 0. In this part we characterize the amplitudes the different regions ajsubscript𝑎𝑗a\_{j} for j=1,…,l𝑗

1…𝑙j=1,...,l.

We notice that the eigenfunctions appear to be continuous and differentiable. Without loss of generality, assume that the boundary between region j𝑗j to region j+1𝑗1j+1 happens at x=0𝑥0x=0. Then the eigenfunction in the vicinity of 0 is defined as follows:

|  |  |  |
| --- | --- | --- |
|  | fq​(x)={aj​cos⁡(q​pjZ​x+bj)x≤0aj+1​cos⁡(q​pj+1Z​x+bj+1)x≥0subscript𝑓𝑞𝑥casessubscript𝑎𝑗𝑞subscript𝑝𝑗𝑍𝑥subscript𝑏𝑗𝑥0subscript𝑎𝑗1𝑞subscript𝑝𝑗1𝑍𝑥subscript𝑏𝑗1𝑥0f\_{q}(x)=\begin{cases}a\_{j}\cos(q\frac{\sqrt{p\_{j}}}{Z}x+b\_{j})&x\leq 0\\ a\_{j+1}\cos(q\frac{\sqrt{p\_{j+1}}}{Z}x+b\_{j+1})&x\geq 0\end{cases} |  |

Continuity at x=0𝑥0x=0 implies that

|  |  |  |  |
| --- | --- | --- | --- |
|  | aj​cos⁡(bj)=aj+1​cos⁡(bj+1)⇒ajaj+1=cos⁡(bj+1)cos⁡(bj)subscript𝑎𝑗subscript𝑏𝑗subscript𝑎𝑗1subscript𝑏𝑗1⇒subscript𝑎𝑗subscript𝑎𝑗1subscript𝑏𝑗1subscript𝑏𝑗a\_{j}\cos(b\_{j})=a\_{j+1}\cos(b\_{j+1})\,\Rightarrow\,\frac{a\_{j}}{a\_{j+1}}=\frac{\cos(b\_{j+1})}{\cos(b\_{j})} |  | (28) |

Differentiability at x=0𝑥0x=0 implies

|  |  |  |
| --- | --- | --- |
|  | aj​pj​sin⁡(bj)=aj+1​pj+1​sin⁡(bj+1)⇔ajaj+1=pj+1​sin⁡(bj+1)pj​sin⁡(bj)⇔subscript𝑎𝑗subscript𝑝𝑗subscript𝑏𝑗subscript𝑎𝑗1subscript𝑝𝑗1subscript𝑏𝑗1subscript𝑎𝑗subscript𝑎𝑗1subscript𝑝𝑗1subscript𝑏𝑗1subscript𝑝𝑗subscript𝑏𝑗a\_{j}\sqrt{p\_{j}}\sin(b\_{j})=a\_{j+1}\sqrt{p\_{j+1}}\sin(b\_{j+1})\,\Leftrightarrow\,\frac{a\_{j}}{a\_{j+1}}=\frac{\sqrt{p\_{j+1}}\sin(b\_{j+1})}{\sqrt{p\_{j}}\sin(b\_{j})} |  |

These allow us to bound the ratio aj/aj+1subscript𝑎𝑗subscript𝑎𝑗1a\_{j}/a\_{j+1}. We have

|  |  |  |  |
| --- | --- | --- | --- |
|  | ajaj+1=pj+1​sin⁡(bj+1)pj​sin⁡(bj)⇒(ajaj+1)2=pj+1​sin2⁡(bj+1)pj​sin2⁡(bj)=pj+1​(1−cos2⁡(bj+1))pj​(1−cos2⁡(bj))subscript𝑎𝑗subscript𝑎𝑗1subscript𝑝𝑗1subscript𝑏𝑗1subscript𝑝𝑗subscript𝑏𝑗⇒superscriptsubscript𝑎𝑗subscript𝑎𝑗12subscript𝑝𝑗1superscript2subscript𝑏𝑗1subscript𝑝𝑗superscript2subscript𝑏𝑗subscript𝑝𝑗11superscript2subscript𝑏𝑗1subscript𝑝𝑗1superscript2subscript𝑏𝑗\frac{a\_{j}}{a\_{j+1}}=\frac{\sqrt{p\_{j+1}}\sin(b\_{j+1})}{\sqrt{p\_{j}}\sin(b\_{j})}\,\Rightarrow\,\left(\frac{a\_{j}}{a\_{j+1}}\right)^{2}=\frac{p\_{j+1}\sin^{2}(b\_{j+1})}{p\_{j}\sin^{2}(b\_{j})}=\frac{p\_{j+1}(1-\cos^{2}(b\_{j+1}))}{p\_{j}(1-\cos^{2}(b\_{j}))} |  | (29) |

On the other hand, from ([28](#A2.E28 "In B.1 Ratios between the amplitudes of neighboring regions ‣ Appendix B The amplitudes of the eigenfunctions in different regions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) we know that

|  |  |  |  |
| --- | --- | --- | --- |
|  | ajaj+1=cos⁡(bj+1)cos⁡(bj+1)⇒(ajaj+1)2=cos2⁡(bj+1)cos2⁡(bj)⇒cos2⁡(bj+1)=cos2⁡(bj)​(ajaj+1)2subscript𝑎𝑗subscript𝑎𝑗1subscript𝑏𝑗1subscript𝑏𝑗1⇒superscriptsubscript𝑎𝑗subscript𝑎𝑗12superscript2subscript𝑏𝑗1superscript2subscript𝑏𝑗⇒superscript2subscript𝑏𝑗1superscript2subscript𝑏𝑗superscriptsubscript𝑎𝑗subscript𝑎𝑗12\frac{a\_{j}}{a\_{j+1}}=\frac{\cos(b\_{j+1})}{\cos(b\_{j+1})}\,\Rightarrow\,\left(\frac{a\_{j}}{a\_{j+1}}\right)^{2}=\frac{\cos^{2}(b\_{j+1})}{\cos^{2}(b\_{j})}\,\Rightarrow\,\cos^{2}(b\_{j+1})=\cos^{2}(b\_{j})\left(\frac{a\_{j}}{a\_{j+1}}\right)^{2} |  | (30) |

Substitute ([30](#A2.E30 "In B.1 Ratios between the amplitudes of neighboring regions ‣ Appendix B The amplitudes of the eigenfunctions in different regions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) in ([29](#A2.E29 "In B.1 Ratios between the amplitudes of neighboring regions ‣ Appendix B The amplitudes of the eigenfunctions in different regions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) we get

|  |  |  |
| --- | --- | --- |
|  | (ajaj+1)2=pj+1pj​1−cos2⁡(bj)​(ajaj+1)21−cos2⁡(bj)⇒(ajaj+1)2​(1−cos2⁡(bj))=pj+1pj​(1−cos2⁡(bj)​(ajaj+1)2)superscriptsubscript𝑎𝑗subscript𝑎𝑗12subscript𝑝𝑗1subscript𝑝𝑗1superscript2subscript𝑏𝑗superscriptsubscript𝑎𝑗subscript𝑎𝑗121superscript2subscript𝑏𝑗⇒superscriptsubscript𝑎𝑗subscript𝑎𝑗121superscript2subscript𝑏𝑗subscript𝑝𝑗1subscript𝑝𝑗1superscript2subscript𝑏𝑗superscriptsubscript𝑎𝑗subscript𝑎𝑗12\left(\frac{a\_{j}}{a\_{j+1}}\right)^{2}=\frac{p\_{j+1}}{p\_{j}}\frac{1-\cos^{2}(b\_{j})(\frac{a\_{j}}{a\_{j+1}})^{2}}{1-\cos^{2}(b\_{j})}\,\Rightarrow\,\left(\frac{a\_{j}}{a\_{j+1}}\right)^{2}(1-\cos^{2}(b\_{j}))=\frac{p\_{j+1}}{p\_{j}}\left(1-\cos^{2}(b\_{j})\left(\frac{a\_{j}}{a\_{j+1}}\right)^{2}\right) |  |

And we have

|  |  |  |
| --- | --- | --- |
|  | (ajaj+1)2​(1−cos2⁡(bj)+pj+1pj​cos2⁡(bj))=pj+1pjsuperscriptsubscript𝑎𝑗subscript𝑎𝑗121superscript2subscript𝑏𝑗subscript𝑝𝑗1subscript𝑝𝑗superscript2subscript𝑏𝑗subscript𝑝𝑗1subscript𝑝𝑗\left(\frac{a\_{j}}{a\_{j+1}}\right)^{2}(1-\cos^{2}(b\_{j})+\frac{p\_{j+1}}{p\_{j}}\cos^{2}(b\_{j}))=\frac{p\_{j+1}}{p\_{j}} |  |

implying that

|  |  |  |  |
| --- | --- | --- | --- |
|  | (ajaj+1)2=pj+1pj1−cos2⁡(bj)​(1−pj+1pj)superscriptsubscript𝑎𝑗subscript𝑎𝑗12subscript𝑝𝑗1subscript𝑝𝑗1superscript2subscript𝑏𝑗1subscript𝑝𝑗1subscript𝑝𝑗\left(\frac{a\_{j}}{a\_{j+1}}\right)^{2}=\frac{\frac{p\_{j+1}}{p\_{j}}}{1-\cos^{2}(b\_{j})\left(1-\frac{p\_{j+1}}{p\_{j}}\right)} |  | (31) |

WLOG assume that pj+1/pj≥1subscript𝑝𝑗1subscript𝑝𝑗1p\_{j+1}/p\_{j}\geq 1 then

|  |  |  |
| --- | --- | --- |
|  | cos2⁡(bj)​(1−pj+1pj)≤0⇒11−cos2⁡(bj)​(1−pj+1pj)≤1superscript2subscript𝑏𝑗1subscript𝑝𝑗1subscript𝑝𝑗0⇒11superscript2subscript𝑏𝑗1subscript𝑝𝑗1subscript𝑝𝑗1\cos^{2}(b\_{j})\left(1-\frac{p\_{j+1}}{p\_{j}}\right)\leq 0\,\Rightarrow\,\frac{1}{1-\cos^{2}(b\_{j})\left(1-\frac{p\_{j+1}}{p\_{j}}\right)}\leq 1 |  |

As a result we get

|  |  |  |
| --- | --- | --- |
|  | (ajaj+1)2=pj+1pj1−cos2⁡(bj)​(1−pj+1pj)≤pj+1pj⇒ajaj+1≤pj+1pjsuperscriptsubscript𝑎𝑗subscript𝑎𝑗12subscript𝑝𝑗1subscript𝑝𝑗1superscript2subscript𝑏𝑗1subscript𝑝𝑗1subscript𝑝𝑗subscript𝑝𝑗1subscript𝑝𝑗⇒subscript𝑎𝑗subscript𝑎𝑗1subscript𝑝𝑗1subscript𝑝𝑗\left(\frac{a\_{j}}{a\_{j+1}}\right)^{2}=\frac{\frac{p\_{j+1}}{p\_{j}}}{1-\cos^{2}(b\_{j})(1-\frac{p\_{j+1}}{p\_{j}})}\leq\frac{p\_{j+1}}{p\_{j}}\,\Rightarrow\,\frac{a\_{j}}{a\_{j+1}}\leq\sqrt{\frac{p\_{j+1}}{p\_{j}}} |  |

For a lower bound note that the denominator in ([31](#A2.E31 "In B.1 Ratios between the amplitudes of neighboring regions ‣ Appendix B The amplitudes of the eigenfunctions in different regions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) satisfies

|  |  |  |
| --- | --- | --- |
|  | 1−cos2⁡(bj)​(1−pj+1pj)=sin2⁡(bj)+pj+1pj​cos2⁡(bj)≤pj+1pj1superscript2subscript𝑏𝑗1subscript𝑝𝑗1subscript𝑝𝑗superscript2subscript𝑏𝑗subscript𝑝𝑗1subscript𝑝𝑗superscript2subscript𝑏𝑗subscript𝑝𝑗1subscript𝑝𝑗1-\cos^{2}(b\_{j})(1-\frac{p\_{j+1}}{p\_{j}})=\sin^{2}(b\_{j})+\frac{p\_{j+1}}{p\_{j}}\cos^{2}(b\_{j})\leq\frac{p\_{j+1}}{p\_{j}} |  |

where the inequality is due to the assumption that pj+1≥pjsubscript𝑝𝑗1subscript𝑝𝑗p\_{j+1}\geq p\_{j}. Consequently, (aj+1/aj)2≥1superscriptsubscript𝑎𝑗1subscript𝑎𝑗21(a\_{j+1}/a\_{j})^{2}\geq 1. In summary, we have bounded the ratios between the amplitudes of neighboring regions by

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1≤ajaj+1≤pj+1pj1subscript𝑎𝑗subscript𝑎𝑗1subscript𝑝𝑗1subscript𝑝𝑗1\leq\frac{a\_{j}}{a\_{j+1}}\leq\sqrt{\frac{p\_{j+1}}{p\_{j}}} |  | (32) |

We next note that these bounds are tight and are obtained in the following setup. Assume we have an even number of regions of constant density l𝑙l each with equal size. Suppose that in each region the eigenfunction includes an integer number of cycles. For each q𝑞q we construct an eigenfunction, by choosing a phase bj=0subscript𝑏𝑗0b\_{j}=0 for j=1,…,l𝑗

1…𝑙j=1,...,l, and it holds that the border between region l/2𝑙2l/2 and l/2+1𝑙21l/2+1 lies at x=0𝑥0x=0. As a result, at this point we have

|  |  |  |
| --- | --- | --- |
|  | al2​cos⁡(q​pl2​ 0Z)=al2+1​cos⁡(q​pl2+1​ 0Z)⇒al2=al2+1subscript𝑎𝑙2𝑞subscript𝑝𝑙2 0𝑍subscript𝑎𝑙21𝑞subscript𝑝𝑙21 0𝑍⇒subscript𝑎𝑙2subscript𝑎𝑙21a\_{\frac{l}{2}}\cos\left(\frac{q\sqrt{p\_{\frac{l}{2}}}\,0}{Z}\right)=a\_{\frac{l}{2}+1}\cos\left(\frac{q\sqrt{p\_{\frac{l}{2}+1}}\,0}{Z}\right)\,\Rightarrow\,a\_{\frac{l}{2}}=a\_{\frac{l}{2}+1} |  |

But since each region contains an integer number of cycles we get for j=1,…,l𝑗

1…𝑙j=1,...,l

|  |  |  |  |
| --- | --- | --- | --- |
|  | cos⁡(q​pl2​ 0Z)=cos⁡(q​pjZ​(2​πl​j−π))=1𝑞subscript𝑝𝑙2 0𝑍𝑞subscript𝑝𝑗𝑍2𝜋𝑙𝑗𝜋1\cos\left(\frac{q\sqrt{p\_{\frac{l}{2}}}\,0}{Z}\right)=\cos\left(\frac{q\sqrt{p\_{j}}}{Z}\left(\frac{2\pi}{l}j-\pi\right)\right)=1 |  | (33) |

Continuity implies for j=2,…,l𝑗

2…𝑙j=2,...,l

|  |  |  |
| --- | --- | --- |
|  | aj−1​cos⁡(q​pj−1Z​(2​π​(j−1)l−π))=aj​cos⁡(q​pjZ​(2​π​(j−1)l−π))⇒aj−1=ajsubscript𝑎𝑗1𝑞subscript𝑝𝑗1𝑍2𝜋𝑗1𝑙𝜋subscript𝑎𝑗𝑞subscript𝑝𝑗𝑍2𝜋𝑗1𝑙𝜋⇒subscript𝑎𝑗1subscript𝑎𝑗a\_{j-1}\cos\left(\frac{q\sqrt{p\_{j-1}}}{Z}\left(\frac{2\pi(j-1)}{l}-\pi\right)\right)=a\_{j}\cos\left(\frac{q\sqrt{p\_{j}}}{Z}\left(\frac{2\pi(j-1)}{l}-\pi\right)\right)\Rightarrow a\_{j-1}=a\_{j} |  |

As a result, for each q𝑞q we get one eigenfunction (up to a global scale)

|  |  |  |  |
| --- | --- | --- | --- |
|  | fq1​(x)=cos⁡(q​pj​xZ), forx∈[2​π​(j−1)l−π,2​π​jl−π]formulae-sequencesuperscriptsubscript𝑓𝑞1𝑥  𝑞subscript𝑝𝑗𝑥𝑍, for𝑥2𝜋𝑗1𝑙𝜋2𝜋𝑗𝑙𝜋f\_{q}^{1}(x)=\cos\left(\frac{q\sqrt{p\_{j}}x}{Z}\right)\ \ \text{, for}\ \ x\in\left[\frac{2\pi(j-1)}{l}-\pi,\frac{2\pi j}{l}-\pi\right] |  | (34) |

We next construct a second eigenfunction for each q𝑞q. Since there is an integer number of cycles in each region, to keep the second eigenfunction of each q𝑞q orthogonal to the first one, we choose a phase of −π/2𝜋2-\pi/2:

|  |  |  |
| --- | --- | --- |
|  | fq1​(x)=aj​sin⁡(q​pj​xZ), forx∈[2​π​(j−1)l−π,2​π​jl−π]formulae-sequencesuperscriptsubscript𝑓𝑞1𝑥  subscript𝑎𝑗𝑞subscript𝑝𝑗𝑥𝑍, for𝑥2𝜋𝑗1𝑙𝜋2𝜋𝑗𝑙𝜋f\_{q}^{1}(x)=a\_{j}\sin\left(\frac{q\sqrt{p\_{j}}x}{Z}\right)\ \ \text{, for}\ \ x\in\left[\frac{2\pi(j-1)}{l}-\pi,\frac{2\pi j}{l}-\pi\right] |  |

Next, to maintain differentiability, the derivative at the border between regions Rjsubscript𝑅𝑗R\_{j} and Rj+1subscript𝑅𝑗1R\_{j+1} must be equal. So at x=2​π​j/l−π𝑥2𝜋𝑗𝑙𝜋x=2\pi j/l-\pi we have for j=1,…,l−1𝑗

1…𝑙1j=1,...,l-1

|  |  |  |
| --- | --- | --- |
|  | dd​x​(aj​sin⁡(q​pj​xZ))=dd​x​(aj+1​sin⁡(q​pj+1​xZ))⇒𝑑𝑑𝑥subscript𝑎𝑗𝑞subscript𝑝𝑗𝑥𝑍𝑑𝑑𝑥subscript𝑎𝑗1𝑞subscript𝑝𝑗1𝑥𝑍⇒absent\frac{d}{dx}\left(a\_{j}\sin\left(\frac{q\sqrt{p\_{j}}x}{Z}\right)\right)=\frac{d}{dx}\left(a\_{j+1}\sin\left(\frac{q\sqrt{p\_{j+1}}x}{Z}\right)\right)\Rightarrow |  |

|  |  |  |
| --- | --- | --- |
|  | −aj​q​pjZ​cos⁡(q​pj​xZ)=−aj+1​q​pj+1Z​cos⁡(q​pj+1​xZ)⇒subscript𝑎𝑗𝑞subscript𝑝𝑗𝑍𝑞subscript𝑝𝑗𝑥𝑍subscript𝑎𝑗1𝑞subscript𝑝𝑗1𝑍𝑞subscript𝑝𝑗1𝑥𝑍⇒absent-\frac{a\_{j}q\sqrt{p\_{j}}}{Z}\cos\left(\frac{q\sqrt{p\_{j}}x}{Z}\right)=-\frac{a\_{j+1}q\sqrt{p\_{j+1}}}{Z}\cos\left(\frac{q\sqrt{p\_{j+1}}x}{Z}\right)\Rightarrow |  |

|  |  |  |
| --- | --- | --- |
|  | aj​pj​cos⁡(q​pj​xZ)=aj+1​pj+1​cos⁡(q​pj+1​xZ)subscript𝑎𝑗subscript𝑝𝑗𝑞subscript𝑝𝑗𝑥𝑍subscript𝑎𝑗1subscript𝑝𝑗1𝑞subscript𝑝𝑗1𝑥𝑍a\_{j}\sqrt{p\_{j}}\cos\left(\frac{q\sqrt{p\_{j}}x}{Z}\right)=a\_{j+1}\sqrt{p\_{j+1}}\cos\left(\frac{q\sqrt{p\_{j+1}}x}{Z}\right) |  |

From ([33](#A2.E33 "In B.1 Ratios between the amplitudes of neighboring regions ‣ Appendix B The amplitudes of the eigenfunctions in different regions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) we have

|  |  |  |
| --- | --- | --- |
|  | aj​pj=aj+1​pj+1⇒ajaj+1=pj+1pjsubscript𝑎𝑗subscript𝑝𝑗subscript𝑎𝑗1subscript𝑝𝑗1⇒subscript𝑎𝑗subscript𝑎𝑗1subscript𝑝𝑗1subscript𝑝𝑗a\_{j}\sqrt{p\_{j}}=a\_{j+1}\sqrt{p\_{j+1}}\Rightarrow\frac{a\_{j}}{a\_{j+1}}=\frac{\sqrt{p\_{j+1}}}{\sqrt{p\_{j}}} |  |

And we can choose for the second eigenfunction for each q𝑞q (up to a global scale)

|  |  |  |  |
| --- | --- | --- | --- |
|  | fq2​(x)=1pj​sin⁡(q​pj​xZ), forx∈[2​π​(j−1)l−π,2​π​jl−π]formulae-sequencesuperscriptsubscript𝑓𝑞2𝑥  1subscript𝑝𝑗𝑞subscript𝑝𝑗𝑥𝑍, for𝑥2𝜋𝑗1𝑙𝜋2𝜋𝑗𝑙𝜋f\_{q}^{2}(x)=\frac{1}{\sqrt{p\_{j}}}\sin\left(\frac{q\sqrt{p\_{j}}x}{Z}\right)\ \ \text{, for}\ \ x\in\left[\frac{2\pi(j-1)}{l}-\pi,\frac{2\pi j}{l}-\pi\right] |  | (35) |

In Figure [13](#A2.F13 "Figure 13 ‣ B.1 Ratios between the amplitudes of neighboring regions ‣ Appendix B The amplitudes of the eigenfunctions in different regions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density") we show an example for this setup.

!(/html/2003.04560/assets/x23.png)

Figure 13: For the NTK of a two-layer network with bias we plot in each of the four columns four of its eigenfunction pairs (each of the same eigenvalue) under a non-uniform data distribution of p​(x)∈1/π​{4/5,1/5}𝑝𝑥1𝜋4515p(x)\in 1/\pi\{4/5,1/5\} in 𝕊1superscript𝕊1\mathbb{S}^{1}. For this distribution whenever mod​(q,3)=0mod𝑞30\mathrm{mod}(q,3)=0 there is an integer number of cycles in each region. As a result, for each q𝑞q we obtain two eigenfunctions of the form of ([34](#A2.E34 "In B.1 Ratios between the amplitudes of neighboring regions ‣ Appendix B The amplitudes of the eigenfunctions in different regions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) and ([35](#A2.E35 "In B.1 Ratios between the amplitudes of neighboring regions ‣ Appendix B The amplitudes of the eigenfunctions in different regions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")).

### B.2 Bounding ajsubscript𝑎𝑗a\_{j}

Assuming p​(x)𝑝𝑥p(x) is constant in l𝑙l regions and that WLOG up to a global scale, the minimal amplitude is amin=1subscript𝑎1a\_{\min}=1. Then for two neighboring regions Rjsubscript𝑅𝑗R\_{j} and Rj+1subscript𝑅𝑗1R\_{j+1} if pj≥pj+1⇒aj+1aj≤pjpj+1≤pmaxpminsubscript𝑝𝑗subscript𝑝𝑗1⇒subscript𝑎𝑗1subscript𝑎𝑗subscript𝑝𝑗subscript𝑝𝑗1subscript𝑝subscript𝑝p\_{j}\geq p\_{j+1}\Rightarrow\frac{a\_{j+1}}{a\_{j}}\leq\sqrt{\frac{p\_{j}}{p\_{j+1}}}\leq\sqrt{\frac{p\_{\max}}{p\_{\min}}}
and if pj+1≥pjsubscript𝑝𝑗1subscript𝑝𝑗p\_{j+1}\geq p\_{j} ⇒ajaj+1≥1⇒aj+1aj≤1≤pmaxpmin⇒absentsubscript𝑎𝑗subscript𝑎𝑗11⇒subscript𝑎𝑗1subscript𝑎𝑗1subscript𝑝subscript𝑝\Rightarrow\frac{a\_{j}}{a\_{j+1}}\geq 1\Rightarrow\frac{a\_{j+1}}{a\_{j}}\leq 1\leq\sqrt{\frac{p\_{\max}}{p\_{\min}}}. As a result in each transition between two regions we have

|  |  |  |
| --- | --- | --- |
|  | ai+1ai≤pmaxpminsubscript𝑎𝑖1subscript𝑎𝑖subscript𝑝subscript𝑝\frac{a\_{i+1}}{a\_{i}}\leq\sqrt{\frac{p\_{\max}}{p\_{\min}}} |  |

Starting from a minimal amplitude of magnitude 111. For l𝑙l regions there are no more than l𝑙l transitions so each amplitude is (loosely) bounded as follows

|  |  |  |
| --- | --- | --- |
|  | aj≤am​i​n​(pmaxpmin)l=(pmaxpmin)l2subscript𝑎𝑗subscript𝑎𝑚𝑖𝑛superscriptsubscript𝑝subscript𝑝𝑙superscriptsubscript𝑝subscript𝑝𝑙2a\_{j}\leq a\_{min}\left(\sqrt{\frac{p\_{\max}}{p\_{\min}}}\right)^{l}=\left(\frac{p\_{\max}}{p\_{\min}}\right)^{\frac{l}{2}} |  |

Next we bound the global scale factor. Let s=∫−ππ(f​(x))2​𝑑x𝑠superscriptsubscript𝜋𝜋superscript𝑓𝑥2differential-d𝑥s=\int\_{-\pi}^{\pi}(f(x))^{2}dx. Then we have that after normalizing the global scale factor

|  |  |  |
| --- | --- | --- |
|  | aj≤1s​(pmaxpmin)l2subscript𝑎𝑗1𝑠superscriptsubscript𝑝subscript𝑝𝑙2a\_{j}\leq\frac{1}{\sqrt{s}}\left(\frac{p\_{\max}}{p\_{\min}}\right)^{\frac{l}{2}} |  |

To simplify notation we denote the frequency of each region by qj=pj​qZsubscript𝑞𝑗subscript𝑝𝑗𝑞𝑍q\_{j}=\frac{\sqrt{p\_{j}}q}{Z}. Then for s𝑠s we have:

|  |  |  |
| --- | --- | --- |
|  | s=∫−ππ(f​(x))2​𝑑x=∑j=1laj2​∫Rjcos2⁡(qj​x+bj)​𝑑x≥∑j=1lamin2​∫Rjcos2⁡(qj​x+bj)​𝑑x=∑j=1l∫Rjcos2⁡(qj​x+bj)​𝑑x𝑠superscriptsubscript𝜋𝜋superscript𝑓𝑥2differential-d𝑥superscriptsubscript𝑗1𝑙superscriptsubscript𝑎𝑗2subscriptsubscript𝑅𝑗superscript2subscript𝑞𝑗𝑥subscript𝑏𝑗differential-d𝑥superscriptsubscript𝑗1𝑙superscriptsubscript𝑎2subscriptsubscript𝑅𝑗superscript2subscript𝑞𝑗𝑥subscript𝑏𝑗differential-d𝑥superscriptsubscript𝑗1𝑙subscriptsubscript𝑅𝑗superscript2subscript𝑞𝑗𝑥subscript𝑏𝑗differential-d𝑥s=\int\_{-\pi}^{\pi}(f(x))^{2}dx=\sum\_{j=1}^{l}a\_{j}^{2}\int\_{R\_{j}}\cos^{2}(q\_{j}x+b\_{j})dx\geq\sum\_{j=1}^{l}a\_{\min}^{2}\int\_{R\_{j}}\cos^{2}(q\_{j}x+b\_{j})dx=\sum\_{j=1}^{l}\int\_{R\_{j}}\cos^{2}(q\_{j}x+b\_{j})dx |  |

For each region we have

|  |  |  |
| --- | --- | --- |
|  | ∫Rjcos2⁡(qj​x+bj)​𝑑x=∫−π+2​πl​(j−1)−π+2​πl​jcos2⁡(qj​x+bj)​𝑑x=subscriptsubscript𝑅𝑗superscript2subscript𝑞𝑗𝑥subscript𝑏𝑗differential-d𝑥superscriptsubscript𝜋2𝜋𝑙𝑗1𝜋2𝜋𝑙𝑗superscript2subscript𝑞𝑗𝑥subscript𝑏𝑗differential-d𝑥absent\int\_{R\_{j}}\cos^{2}(q\_{j}x+b\_{j})dx=\int\_{-\pi+\frac{2\pi}{l}(j-1)}^{-\pi+\frac{2\pi}{l}j}\cos^{2}(q\_{j}x+b\_{j})dx= |  |

|  |  |  |
| --- | --- | --- |
|  | 12​∫−π+2​πl​(j−1)−π+2​πl​j(1+cos⁡(2​qj​x+2​bj))​𝑑x=12​(x+sin⁡(2​qj​x+2​bj)2​qj)−π+2​πl​(j−1)−π+2​πl​j=12superscriptsubscript𝜋2𝜋𝑙𝑗1𝜋2𝜋𝑙𝑗12subscript𝑞𝑗𝑥2subscript𝑏𝑗differential-d𝑥12superscriptsubscript𝑥2subscript𝑞𝑗𝑥2subscript𝑏𝑗2subscript𝑞𝑗𝜋2𝜋𝑙𝑗1𝜋2𝜋𝑙𝑗absent\frac{1}{2}\int\_{-\pi+\frac{2\pi}{l}(j-1)}^{-\pi+\frac{2\pi}{l}j}(1+\cos{}(2q\_{j}x+2b\_{j}))dx=\frac{1}{2}\left(x+\frac{\sin(2q\_{j}x+2b\_{j})}{2q\_{j}}\right)\_{-\pi+\frac{2\pi}{l}(j-1)}^{-\pi+\frac{2\pi}{l}j}= |  |

|  |  |  |
| --- | --- | --- |
|  | 12​(−π+2​πl​j+sin⁡(2​qj​(−π+2​πl​j)+2​bj)2​qj−(−π+2​πl​(j−1))−sin⁡(2​qj​(−π+2​πl​(j−1))+2​bj)2​qj)=12𝜋2𝜋𝑙𝑗2subscript𝑞𝑗𝜋2𝜋𝑙𝑗2subscript𝑏𝑗2subscript𝑞𝑗𝜋2𝜋𝑙𝑗12subscript𝑞𝑗𝜋2𝜋𝑙𝑗12subscript𝑏𝑗2subscript𝑞𝑗absent\frac{1}{2}\left(-\pi+\frac{2\pi}{l}j+\frac{\sin(2q\_{j}(-\pi+\frac{2\pi}{l}j)+2b\_{j})}{2q\_{j}}-(-\pi+\frac{2\pi}{l}(j-1))-\frac{\sin(2q\_{j}(-\pi+\frac{2\pi}{l}(j-1))+2b\_{j})}{2q\_{j}}\right)= |  |

|  |  |  |
| --- | --- | --- |
|  | 12​(2​πl+sin⁡(2​qj​(−π+2​πl​j)+2​bj)2​qj−sin⁡(2​qj​(−π+2​πl​(j−1))+2​bj)2​qj)≥πl−12​qj122𝜋𝑙2subscript𝑞𝑗𝜋2𝜋𝑙𝑗2subscript𝑏𝑗2subscript𝑞𝑗2subscript𝑞𝑗𝜋2𝜋𝑙𝑗12subscript𝑏𝑗2subscript𝑞𝑗𝜋𝑙12subscript𝑞𝑗\frac{1}{2}\left(\frac{2\pi}{l}+\frac{\sin(2q\_{j}(-\pi+\frac{2\pi}{l}j)+2b\_{j})}{2q\_{j}}-\frac{\sin(2q\_{j}(-\pi+\frac{2\pi}{l}(j-1))+2b\_{j})}{2q\_{j}}\right)\geq\frac{\pi}{l}-\frac{1}{2q\_{j}} |  |

So we get s≥∑j=1lπl−12​qj=π−12​∑j=1l1qj=π−12​∑j=1lZpj​q𝑠superscriptsubscript𝑗1𝑙𝜋𝑙12subscript𝑞𝑗𝜋12superscriptsubscript𝑗1𝑙1subscript𝑞𝑗𝜋12superscriptsubscript𝑗1𝑙𝑍subscript𝑝𝑗𝑞s\geq\sum\_{j=1}^{l}\frac{\pi}{l}-\frac{1}{2q\_{j}}=\pi-\frac{1}{2}\sum\_{j=1}^{l}\frac{1}{q\_{j}}=\pi-\frac{1}{2}\sum\_{j=1}^{l}\frac{Z}{\sqrt{p\_{j}}q}.

And we get:

|  |  |  |
| --- | --- | --- |
|  | s≥π−12​∑j=1lZpj​q=π−Z2​q​∑j=1l1pj𝑠𝜋12superscriptsubscript𝑗1𝑙𝑍subscript𝑝𝑗𝑞𝜋𝑍2𝑞superscriptsubscript𝑗1𝑙1subscript𝑝𝑗s\geq\pi-\frac{1}{2}\sum\_{j=1}^{l}\frac{Z}{\sqrt{p\_{j}}q}=\pi-\frac{Z}{2q}\sum\_{j=1}^{l}\frac{1}{\sqrt{p\_{j}}} |  |

As a result all the amplitudes in an eigenfunction of order q𝑞q are bounded by

|  |  |  |  |
| --- | --- | --- | --- |
|  | ai≤1π−Z2​q​∑j=1l1pj​(pmaxpmin)l2​for​all​1≤i≤lsubscript𝑎𝑖1𝜋𝑍2𝑞superscriptsubscript𝑗1𝑙1subscript𝑝𝑗superscriptsubscript𝑝subscript𝑝𝑙2forall1𝑖𝑙a\_{i}\leq\frac{1}{\sqrt{\pi-\frac{Z}{2q}\sum\_{j=1}^{l}\frac{1}{\sqrt{p\_{j}}}}}\left(\frac{p\_{\max}}{p\_{\min}}\right)^{\frac{l}{2}}~{}~{}~{}\mathrm{for~{}all~{}}1\leq i\leq l |  | (36) |

## Appendix C Local convergence rate as a function of frequency

To derive the rate of convergence as a function of frequency and density we assume that p​(x)𝑝𝑥p(x) forms a piecewise-constant distribution (PCD) with a fixed number of pieces l𝑙l of equal sizes, p​(x)=pj𝑝𝑥subscript𝑝𝑗p(x)=p\_{j} in Rjsubscript𝑅𝑗R\_{j}, 1≤j≤l1𝑗𝑙{1\leq j\leq l}. Our proof will rely on a lemma that states informally that not too many eigenfunctions need to be taken into account for convergence – more precisely, only a number linear in k𝑘k and inversely linear in p∗superscript𝑝\sqrt{p^{\*}}, where p∗>0superscript𝑝0p^{\*}>0 denotes the minimal density. Convergence rate is then determined by the eigenfunction with highest eigenvalue included in the approximation for g​(x)𝑔𝑥g(x).

###### Lemma 2.

Let p​(x)𝑝𝑥p(x) be PCD. For any ϵ>0italic-ϵ0\epsilon>0, there exist nksubscript𝑛𝑘n\_{k} such that ∑j=nk+1∞gi2<ϵ2superscriptsubscript𝑗subscript𝑛𝑘1superscriptsubscript𝑔𝑖2superscriptitalic-ϵ2\sum\_{j={n\_{k}+1}}^{\infty}g\_{i}^{2}<\epsilon^{2}, where gi=∫−ππvi​(x)​g​(x)​p​(x)​𝑑xsubscript𝑔𝑖superscriptsubscript𝜋𝜋subscript𝑣𝑖𝑥𝑔𝑥𝑝𝑥differential-d𝑥g\_{i}=\int\_{-\pi}^{\pi}v\_{i}(x)g(x)p(x)dx and nksubscript𝑛𝑘n\_{k} is bound as in ([39](#A3.E39 "In Proof. ‣ Appendix C Local convergence rate as a function of frequency ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) below.

###### Proof.

Given a target function g​(x)=cos⁡(k​x)𝑔𝑥𝑘𝑥g(x)=\cos(kx) and a basis function vi​(x)=a​(x)​cos⁡(qi​p​(x)​xZ+b​(x))subscript𝑣𝑖𝑥𝑎𝑥subscript𝑞𝑖𝑝𝑥𝑥𝑍𝑏𝑥v\_{i}(x)=a(x)\cos(\frac{q\_{i}\sqrt{p(x)}x}{Z}+b(x)) where qi=⌊i/2⌋subscript𝑞𝑖𝑖2q\_{i}=\lfloor i/2\rfloor. (We will assume a=1𝑎1a=1 for now.) Their inner product can be written as

|  |  |  |  |
| --- | --- | --- | --- |
|  | gi=∑j=1laj​pj​∫Rjcos⁡(k​x)​cos⁡(qi​j​x+bj)​𝑑xsubscript𝑔𝑖superscriptsubscript𝑗1𝑙subscript𝑎𝑗subscript𝑝𝑗subscriptsubscript𝑅𝑗𝑘𝑥subscript𝑞𝑖𝑗𝑥subscript𝑏𝑗differential-d𝑥g\_{i}=\sum\_{j=1}^{l}a\_{j}p\_{j}\int\_{R\_{j}}\cos(kx)\cos(q\_{ij}x+b\_{j})dx |  | (37) |

where qi​j=qi​pj/Zsubscript𝑞𝑖𝑗subscript𝑞𝑖subscript𝑝𝑗𝑍q\_{ij}=q\_{i}\sqrt{p\_{j}}/Z denotes the local frequency of vi​(x)subscript𝑣𝑖𝑥v\_{i}(x) at Rjsubscript𝑅𝑗R\_{j}. Next, to derive a bound we will restrict our treatment to qi​j≥2​ksubscript𝑞𝑖𝑗2𝑘q\_{ij}\geq 2k (and by that bound nksubscript𝑛𝑘n\_{k} from below). With this assumption we obtain

|  |  |  |
| --- | --- | --- |
|  | |∫Rjcos⁡(k​x)​cos⁡(qi​j​x+bj)​𝑑x|≤|∫−πlπlcos⁡(k​x)​cos⁡(qi​j​x)​𝑑x|=subscriptsubscript𝑅𝑗𝑘𝑥subscript𝑞𝑖𝑗𝑥subscript𝑏𝑗differential-d𝑥superscriptsubscript𝜋𝑙𝜋𝑙𝑘𝑥subscript𝑞𝑖𝑗𝑥differential-d𝑥absent\displaystyle\left|\int\_{R\_{j}}\cos(kx)\cos(q\_{ij}x+b\_{j})dx\right|\leq\left|\int\_{-\frac{\pi}{l}}^{\frac{\pi}{l}}\cos(kx)\cos(q\_{ij}x)dx\right|= |  |
|  |  |  |
| --- | --- | --- |
|  | |sin⁡(π​(qi​j+k)l)qi​j+k+sin⁡(π​(qi​j−k)l)qi​j−k|≤1qi​j+k+1qi​j−k=2​qi​jqi​j2−k2≤83​qi​j𝜋subscript𝑞𝑖𝑗𝑘𝑙subscript𝑞𝑖𝑗𝑘𝜋subscript𝑞𝑖𝑗𝑘𝑙subscript𝑞𝑖𝑗𝑘1subscript𝑞𝑖𝑗𝑘1subscript𝑞𝑖𝑗𝑘2subscript𝑞𝑖𝑗superscriptsubscript𝑞𝑖𝑗2superscript𝑘283subscript𝑞𝑖𝑗\displaystyle\left|\dfrac{\sin\left(\frac{\pi\left(q\_{ij}+k\right)}{l}\right)}{q\_{ij}+k}+\dfrac{\sin\left(\frac{\pi\left(q\_{ij}-k\right)}{l}\right)}{q\_{ij}-k}\right|\leq\frac{1}{q\_{ij}+k}+\frac{1}{q\_{ij}-k}=\dfrac{2q\_{ij}}{q\_{ij}^{2}-k^{2}}\leq\frac{8}{3q\_{ij}} |  |

Let p∗=minj⁡pjsuperscript𝑝subscript𝑗subscript𝑝𝑗p^{\*}=\min\_{j}p\_{j} and let qi∗=qi​p∗/Zsuperscriptsubscript𝑞𝑖subscript𝑞𝑖superscript𝑝𝑍q\_{i}^{\*}=q\_{i}\sqrt{p^{\*}}/Z, qi∗superscriptsubscript𝑞𝑖q\_{i}^{\*} denotes the frequency associated with the corresponding region (which is the lowest within visubscript𝑣𝑖v\_{i}). Our requirement that qi​j>2​ksubscript𝑞𝑖𝑗2𝑘q\_{ij}>2k for all 1≤j≤l1𝑗𝑙1\leq j\leq l implies that qi∗>2​ksuperscriptsubscript𝑞𝑖2𝑘q\_{i}^{\*}>2k, and therefore

|  |  |  |  |
| --- | --- | --- | --- |
|  | qi>2​Z​kp∗subscript𝑞𝑖2𝑍𝑘superscript𝑝q\_{i}>\frac{2Zk}{\sqrt{p^{\*}}} |  | (38) |

Additionally, using ([37](#A3.E37 "In Proof. ‣ Appendix C Local convergence rate as a function of frequency ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"))

|  |  |  |
| --- | --- | --- |
|  | |gi|≤83​∑j=1laj​pjqi​j≤83​qi∗​∑j=1laj​pj=8​B3​qi∗=8​B​Z3​qi​p∗subscript𝑔𝑖83superscriptsubscript𝑗1𝑙subscript𝑎𝑗subscript𝑝𝑗subscript𝑞𝑖𝑗83subscriptsuperscript𝑞𝑖superscriptsubscript𝑗1𝑙subscript𝑎𝑗subscript𝑝𝑗8𝐵3subscriptsuperscript𝑞𝑖8𝐵𝑍3subscript𝑞𝑖superscript𝑝|g\_{i}|\leq\frac{8}{3}\sum\_{j=1}^{l}\frac{a\_{j}p\_{j}}{q\_{ij}}\leq\frac{8}{3q^{\*}\_{i}}\sum\_{j=1}^{l}a\_{j}p\_{j}=\frac{8B}{3q^{\*}\_{i}}=\frac{8BZ}{3q\_{i}\sqrt{p^{\*}}} |  |

where we denote by B=∑j=1laj​pj𝐵superscriptsubscript𝑗1𝑙subscript𝑎𝑗subscript𝑝𝑗B=\sum\_{j=1}^{l}a\_{j}p\_{j} and the equality on the right is obtained by plugging in the definition of qi∗superscriptsubscript𝑞𝑖q\_{i}^{\*}. Note that ∑j=1lpj=l/(2​π)superscriptsubscript𝑗1𝑙subscript𝑝𝑗𝑙2𝜋\sum\_{j=1}^{l}p\_{j}=l/(2\pi) (since 1=∫−ππp​(x)​𝑑x=∑j=1l2​π​pj/l1superscriptsubscript𝜋𝜋𝑝𝑥differential-d𝑥superscriptsubscript𝑗1𝑙2𝜋subscript𝑝𝑗𝑙1=\int\_{-\pi}^{\pi}p(x)dx=\sum\_{j=1}^{l}2\pi p\_{j}/l), implying that B≤l​a∗/(2​π)𝐵𝑙superscript𝑎2𝜋B\leq la^{\*}/(2\pi), where a∗=maxj⁡ajsuperscript𝑎subscript𝑗subscript𝑎𝑗a^{\*}=\max\_{j}a\_{j} and a∗superscript𝑎a^{\*} is bounded by ([36](#A2.E36 "In B.2 Bounding 𝑎_𝑗 ‣ Appendix B The amplitudes of the eigenfunctions in different regions ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")).

Next, for a given ϵ>0italic-ϵ0\epsilon>0 we wish to bound the sum ∑i=nk∞gi2superscriptsubscript𝑖subscript𝑛𝑘superscriptsubscript𝑔𝑖2\sum\_{i=n\_{k}}^{\infty}g\_{i}^{2} by starting from a sufficiently high index nksubscript𝑛𝑘n\_{k}, i.e.,

|  |  |  |
| --- | --- | --- |
|  | ∑i=nk+1∞gi2≤(8​B​Z3​p∗)2​∑i=nk+1∞1qi2<1qnk​(8​B​Z3​p∗)2<ϵ2superscriptsubscript𝑖subscript𝑛𝑘1superscriptsubscript𝑔𝑖2superscript8𝐵𝑍3superscript𝑝2superscriptsubscript𝑖subscript𝑛𝑘11superscriptsubscript𝑞𝑖21subscript𝑞subscript𝑛𝑘superscript8𝐵𝑍3superscript𝑝2superscriptitalic-ϵ2\sum\_{i={n\_{k}+1}}^{\infty}g\_{i}^{2}\leq\left(\frac{8BZ}{3\sqrt{p^{\*}}}\right)^{2}\sum\_{i={n\_{k}+1}}^{\infty}\frac{1}{q\_{i}^{2}}<\frac{1}{q\_{n\_{k}}}\left(\frac{8BZ}{3\sqrt{p^{\*}}}\right)^{2}<\epsilon^{2} |  |

By the definition of qisubscript𝑞𝑖q\_{i}, nk≥2​qnksubscript𝑛𝑘2subscript𝑞subscript𝑛𝑘n\_{k}\geq 2q\_{n\_{k}}, so

|  |  |  |
| --- | --- | --- |
|  | nk>2ϵ2​(8​B​Z3​p∗)2=128​B2​Z29​ϵ2​p∗subscript𝑛𝑘2superscriptitalic-ϵ2superscript8𝐵𝑍3superscript𝑝2128superscript𝐵2superscript𝑍29superscriptitalic-ϵ2superscript𝑝n\_{k}>\frac{2}{\epsilon^{2}}\left(\frac{8BZ}{3\sqrt{p^{\*}}}\right)^{2}=\frac{128B^{2}Z^{2}}{9\epsilon^{2}p^{\*}} |  |

So in conclusion,

|  |  |  |  |
| --- | --- | --- | --- |
|  | nk>max⁡{4​Z​kp∗,128​B2​Z29​ϵ2​p∗}subscript𝑛𝑘4𝑍𝑘superscript𝑝128superscript𝐵2superscript𝑍29superscriptitalic-ϵ2superscript𝑝n\_{k}>\max\left\{\frac{4Zk}{\sqrt{p^{\*}}},\,\frac{128B^{2}Z^{2}}{9\epsilon^{2}p^{\*}}\right\} |  | (39) |

∎

###### Theorem 3.

Let p​(x)𝑝𝑥p(x) be a PCD, for any δ>0𝛿0\delta>0 the number of iterations t𝑡t needed to achieve ‖g​(x)−u(t)​(x)‖<δnorm𝑔𝑥superscript𝑢𝑡𝑥𝛿\|g(x)-u^{(t)}(x)\|<\delta is O~​(k2/p∗)~𝑂superscript𝑘2superscript𝑝\tilde{O}(k^{2}/p^{\*}), where O~~𝑂\tilde{O} hides logarithmic terms.

###### Proof.

Let nksubscript𝑛𝑘n\_{k} be chosen as in Lemma [2](#Thmlemma2 "Lemma 2. ‣ Appendix C Local convergence rate as a function of frequency ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density") with ϵ=δ/2italic-ϵ𝛿2\epsilon=\delta/2, i.e.

|  |  |  |
| --- | --- | --- |
|  | nk=max⁡{4​Z​kp∗,256​B2​Z29​δ2​p∗}subscript𝑛𝑘4𝑍𝑘superscript𝑝256superscript𝐵2superscript𝑍29superscript𝛿2superscript𝑝n\_{k}=\max\left\{\frac{4Zk}{\sqrt{p^{\*}}},\,\frac{256B^{2}Z^{2}}{9\delta^{2}p^{\*}}\right\} |  |

Let

|  |  |  |
| --- | --- | --- |
|  | g^​(x)=∑i=1nkgi​v​(i)^𝑔𝑥superscriptsubscript𝑖1subscript𝑛𝑘subscript𝑔𝑖𝑣𝑖\hat{g}(x)=\sum\_{i=1}^{n\_{k}}g\_{i}v(i) |  |

Then,

|  |  |  |
| --- | --- | --- |
|  | ‖g​(x)−g^​(x)‖2=∑i=nk+1∞gi2<(δ2)2superscriptnorm𝑔𝑥^𝑔𝑥2superscriptsubscript𝑖subscript𝑛𝑘1superscriptsubscript𝑔𝑖2superscript𝛿22\|g(x)-\hat{g}(x)\|^{2}=\sum\_{i=n\_{k}+1}^{\infty}g\_{i}^{2}<\left(\frac{\delta}{2}\right)^{2} |  |

and due to triangle inequality

|  |  |  |
| --- | --- | --- |
|  | ‖g​(x)−u(t)​(x)‖≤‖g​(x)−g^​(x)‖+‖g^​(x)−u(t)​(x)‖norm𝑔𝑥superscript𝑢𝑡𝑥norm𝑔𝑥^𝑔𝑥norm^𝑔𝑥superscript𝑢𝑡𝑥\|g(x)-u^{(t)}(x)\|\leq\|g(x)-\hat{g}(x)\|+\|\hat{g}(x)-u^{(t)}(x)\| |  |

it suffices to find t𝑡t such that

|  |  |  |
| --- | --- | --- |
|  | ‖g^​(x)−u(t)​(x)‖<δ2=δ~norm^𝑔𝑥superscript𝑢𝑡𝑥𝛿2~𝛿\|\hat{g}(x)-u^{(t)}(x)\|<\frac{\delta}{2}=\tilde{\delta} |  |

Using (Arora et al., 2019b)’s Theorem 4.1 adapted to continuous operators

|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ2=‖g^−u(t)‖2≈∑i=1nk(1−η​λi)2​t​gi2≤π​∑i=1nk(1−η​λi)2​t≤π​nk​(1−η​λnk)2​tsuperscriptΔ2superscriptnorm^𝑔superscript𝑢𝑡2superscriptsubscript𝑖1subscript𝑛𝑘superscript1𝜂subscript𝜆𝑖2𝑡superscriptsubscript𝑔𝑖2𝜋superscriptsubscript𝑖1subscript𝑛𝑘superscript1𝜂subscript𝜆𝑖2𝑡𝜋subscript𝑛𝑘superscript1𝜂subscript𝜆subscript𝑛𝑘2𝑡\Delta^{2}=\|\hat{g}-u^{(t)}\|^{2}\approx\sum\_{i=1}^{n\_{k}}(1-\eta\lambda\_{i})^{2t}g\_{i}^{2}\leq\pi\sum\_{i=1}^{n\_{k}}(1-\eta\lambda\_{i})^{2t}\leq\pi n\_{k}(1-\eta\lambda\_{n\_{k}})^{2t} |  | (40) |

where the left inequality is due to |gi|2≤‖cos2⁡(k​x)‖=πsuperscriptsubscript𝑔𝑖2normsuperscript2𝑘𝑥𝜋|g\_{i}|^{2}\leq\|\cos^{2}(kx)\|=\pi and the right inequality is because λisubscript𝜆𝑖\lambda\_{i} are arranged in a descending order. Now for a fixed distribution p​(x)𝑝𝑥p(x), and since we are interested in the asymptotic rate of convergence (i.e., as k→∞→𝑘k\rightarrow\infty), as soon as k>64​B2​Z/(9​δ~2​p∗)𝑘64superscript𝐵2𝑍9superscript~𝛿2superscript𝑝k>64B^{2}Z/(9\tilde{\delta}^{2}\sqrt{p^{\*}}) it suffices to only consider the case qnk=2​Z​k/p∗subscript𝑞subscript𝑛𝑘2𝑍𝑘superscript𝑝q\_{n\_{k}}=2Zk/\sqrt{p^{\*}}, as in ([38](#A3.E38 "In Proof. ‣ Appendix C Local convergence rate as a function of frequency ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")). The eigenvalue λnksubscript𝜆subscript𝑛𝑘\lambda\_{n\_{k}} is determined according to

|  |  |  |
| --- | --- | --- |
|  | λnk=Z2π2​qnk2=p∗4​π2​k2subscript𝜆subscript𝑛𝑘superscript𝑍2superscript𝜋2superscriptsubscript𝑞subscript𝑛𝑘2superscript𝑝4superscript𝜋2superscript𝑘2\lambda\_{n\_{k}}=\frac{Z^{2}}{\pi^{2}q\_{n\_{k}}^{2}}=\frac{p^{\*}}{4\pi^{2}k^{2}} |  |

(Here we used the expression for λnksubscript𝜆subscript𝑛𝑘\lambda\_{n\_{k}} assuming nksubscript𝑛𝑘n\_{k} is odd. A similar expression of the same order is obtained for even nksubscript𝑛𝑘n\_{k}.) Consequently, to bound Δ2<δ~superscriptΔ2~𝛿\Delta^{2}<\tilde{\delta} in ([40](#A3.E40 "In Proof. ‣ Appendix C Local convergence rate as a function of frequency ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) and substituting for nksubscript𝑛𝑘n\_{k} and λnksubscript𝜆subscript𝑛𝑘\lambda\_{n\_{k}} we have

|  |  |  |
| --- | --- | --- |
|  | 4​Z​kp∗​(1−η​p∗4​π2​k2)2​t<δ~4𝑍𝑘superscript𝑝superscript1𝜂superscript𝑝4superscript𝜋2superscript𝑘22𝑡~𝛿\frac{4Zk}{\sqrt{p^{\*}}}\left(1-\frac{\eta p^{\*}}{4\pi^{2}k^{2}}\right)^{2t}<\tilde{\delta} |  |

Taking log

|  |  |  |
| --- | --- | --- |
|  | 2​t​log⁡(1−η​p∗4​π2​k2)>log⁡(δ​p∗4​Z​k)2𝑡1𝜂superscript𝑝4superscript𝜋2superscript𝑘2𝛿superscript𝑝4𝑍𝑘2t\log\left(1-\frac{\eta p^{\*}}{4\pi^{2}k^{2}}\right)>\log\left(\frac{\delta\sqrt{p^{\*}}}{4Zk}\right) |  |

from which we obtain

|  |  |  |
| --- | --- | --- |
|  | t>log⁡(δ​p∗4​Z​k)2​log⁡(1−η​p∗4​π2​k2)≈−2​π2​k2η​p∗​log⁡(δ​p∗4​Z​k)=O~​(k2p∗)𝑡𝛿superscript𝑝4𝑍𝑘21𝜂superscript𝑝4superscript𝜋2superscript𝑘22superscript𝜋2superscript𝑘2𝜂superscript𝑝𝛿superscript𝑝4𝑍𝑘~𝑂superscript𝑘2superscript𝑝t>\frac{\log\left(\frac{\delta\sqrt{p^{\*}}}{4Zk}\right)}{2\log\left(1-\frac{\eta p^{\*}}{4\pi^{2}k^{2}}\right)}\approx-\frac{2\pi^{2}k^{2}}{\eta p^{\*}}\log\left(\frac{\delta\sqrt{p^{\*}}}{4Zk}\right)=\tilde{O}\left(\frac{k^{2}}{p^{\*}}\right) |  |

where O~~𝑂\tilde{O} hides logarithmic terms.

∎

\comment

————-

Maybe this can be useful..

|  |  |  |  |
| --- | --- | --- | --- |
|  | A=A​(k,d,l)=∫−πlπlcos⁡(k​x)​cos⁡((k+d)​x)​𝑑x=sin⁡(π​(2​k+d)l)2​k+d+sin⁡(π​dl)d𝐴𝐴𝑘𝑑𝑙superscriptsubscript𝜋𝑙𝜋𝑙𝑘𝑥𝑘𝑑𝑥differential-d𝑥𝜋2𝑘𝑑𝑙2𝑘𝑑𝜋𝑑𝑙𝑑A=A(k,d,l)=\int\_{-\frac{\pi}{l}}^{\frac{\pi}{l}}\cos(kx)\cos((k+d)x)dx=\dfrac{\sin\left(\frac{{\pi}\left(2k+d\right)}{l}\right)}{2k+d}+\dfrac{\sin\left(\frac{{\pi}d}{l}\right)}{d} |  | (41) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | sin⁡(π​(2​k+d)l)=sin⁡(2​π​kl)​cos⁡(π​dl)+cos⁡(2​π​kl)​sin⁡(π​dl)𝜋2𝑘𝑑𝑙2𝜋𝑘𝑙𝜋𝑑𝑙2𝜋𝑘𝑙𝜋𝑑𝑙\sin\left(\frac{{\pi}\left(2k+d\right)}{l}\right)=\sin\left(\frac{2\pi k}{l}\right)\cos\left(\frac{\pi d}{l}\right)+\cos\left(\frac{2\pi k}{l}\right)\sin\left(\frac{\pi d}{l}\right) |  | (42) |

Assuming k/l𝑘𝑙k/l is integer then sin⁡(2​π​kl)=02𝜋𝑘𝑙0\sin\left(\frac{2\pi k}{l}\right)=0 and cos⁡(2​π​kl)=12𝜋𝑘𝑙1\cos\left(\frac{2\pi k}{l}\right)=1, so

|  |  |  |  |
| --- | --- | --- | --- |
|  | sin⁡(π​(2​k+d)l)=sin⁡(π​dl)𝜋2𝑘𝑑𝑙𝜋𝑑𝑙\sin\left(\frac{{\pi}\left(2k+d\right)}{l}\right)=\sin\left(\frac{\pi d}{l}\right) |  | (43) |

Therefore,

|  |  |  |  |
| --- | --- | --- | --- |
|  | A=(12​k+d+1d)​sin⁡(π​dl)=2​(k+d)d​(2​k+d)​sin⁡(π​dl)=(1+d2​k+d)​1d​sin⁡(π​dl)𝐴12𝑘𝑑1𝑑𝜋𝑑𝑙2𝑘𝑑𝑑2𝑘𝑑𝜋𝑑𝑙1𝑑2𝑘𝑑1𝑑𝜋𝑑𝑙A=\left(\dfrac{1}{2k+d}+\dfrac{1}{d}\right)\sin\left(\frac{{\pi}d}{l}\right)=\dfrac{2(k+d)}{d(2k+d)}\sin\left(\frac{{\pi}d}{l}\right)=\left(1+\dfrac{d}{2k+d}\right)\dfrac{1}{d}\sin\left(\frac{{\pi}d}{l}\right) |  | (44) |

Assuming |d|≤0.5𝑑0.5|d|\leq 0.5, we note that

|  |  |  |  |
| --- | --- | --- | --- |
|  | d2​k+d⪅14​k𝑑2𝑘𝑑14𝑘\dfrac{d}{2k+d}\lessapprox\dfrac{1}{4k} |  | (45) |

and

|  |  |  |  |
| --- | --- | --- | --- |
|  | 0.9​πl≤1d​sin⁡(π​dl)≤πl0.9𝜋𝑙1𝑑𝜋𝑑𝑙𝜋𝑙0.9\dfrac{\pi}{l}\leq\dfrac{1}{d}\sin\left(\frac{{\pi}d}{l}\right)\leq\dfrac{\pi}{l} |  | (46) |

In principle in every region Rjsubscript𝑅𝑗R\_{j} we should have |dj|≤pj/(2​Z)subscript𝑑𝑗subscript𝑝𝑗2𝑍|d\_{j}|\leq\sqrt{p\_{j}}/(2Z). Since

|  |  |  |  |
| --- | --- | --- | --- |
|  | Z=12​π​∫−ππp​(x)=1l​∑j=1lpj𝑍12𝜋superscriptsubscript𝜋𝜋𝑝𝑥1𝑙superscriptsubscript𝑗1𝑙subscript𝑝𝑗Z=\frac{1}{2\pi}\int\_{-\pi}^{\pi}\sqrt{p(x)}=\frac{1}{l}\sum\_{j=1}^{l}\sqrt{p\_{j}} |  | (47) |

we obtain that

|  |  |  |  |
| --- | --- | --- | --- |
|  | |dj|≤l​pj2​∑j=1lpjsubscript𝑑𝑗𝑙subscript𝑝𝑗2superscriptsubscript𝑗1𝑙subscript𝑝𝑗|d\_{j}|\leq\frac{l\sqrt{p\_{j}}}{2\sum\_{j=1}^{l}\sqrt{p\_{j}}} |  | (48) |

## Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2

### D.1 The network model

The parameters of the network are W=(W1,…,WL)𝑊subscript𝑊1…subscript𝑊𝐿W=(W\_{1},...,W\_{L}) where Wl∈ℝm×msubscript𝑊𝑙superscriptℝ𝑚𝑚W\_{l}\in\mathbb{R}^{m\times m} and also A∈ℝm×d𝐴superscriptℝ𝑚𝑑A\in\mathbb{R}^{m\times d} and B∈ℝ1×m𝐵superscriptℝ1𝑚B\in\mathbb{R}^{1\times m}. The network function over input 𝐱i∈ℝdsubscript𝐱𝑖superscriptℝ𝑑\mathbf{x}\_{i}\in\mathbb{R}^{d} (i∈[n]𝑖delimited-[]𝑛i\in\left[n\right]) is given by

|  |  |  |
| --- | --- | --- |
|  | ui=f(𝐱i;W)=Bσ(WLσ(WL−1σ(….(W1σ(Axi))..))u\_{i}=f(\mathbf{x}\_{i};W)=B\sigma(W\_{L}\sigma(W\_{L-1}\sigma(....(W\_{1}\sigma(Ax\_{i}))..)) |  |

where σ𝜎\sigma stands for element wise RELU activation function. For a tuple W=(W1,…,WL)𝑊subscript𝑊1…subscript𝑊𝐿W=(W\_{1},...,W\_{L}) of matrices, we let ∥W∥2=maxl∈[L]∥Wl∥2\left\lVert W\right\rVert\_{2}=\max\_{l\in[L]}\left\lVert W\_{l}\right\rVert\_{2} and ∥W∥F=(∑l=1L∥Wl∥F2)1/2subscriptdelimited-∥∥𝑊𝐹superscriptsuperscriptsubscript𝑙1𝐿superscriptsubscriptdelimited-∥∥subscript𝑊𝑙𝐹212\left\lVert W\right\rVert\_{F}=(\sum\_{l=1}^{L}\left\lVert W\_{l}\right\rVert\_{F}^{2})^{1/2}.

The parameters are initialized randomly from a normal distribution according to

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | [Wl]i​jsubscriptdelimited-[]subscript𝑊𝑙𝑖𝑗\displaystyle[W\_{l}]\_{ij} | ∼𝒩​(0,2m),l∈[L]formulae-sequencesimilar-toabsent𝒩02𝑚𝑙delimited-[]𝐿\displaystyle\sim\mathcal{N}(0,\frac{2}{m}),\,l\in[L] |  | (49) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Ai​jsubscript𝐴𝑖𝑗\displaystyle A\_{ij} | ∼𝒩​(0,2m)similar-toabsent𝒩02𝑚\displaystyle\sim\mathcal{N}(0,\frac{2}{m}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Bi​jsubscript𝐵𝑖𝑗\displaystyle B\_{ij} | ∼𝒩​(0,τ2)similar-toabsent𝒩0superscript𝜏2\displaystyle\sim\mathcal{N}(0,\tau^{2}) |  |

where similarly to (Allen-Zhu et al., [2019](#bib.bib2)) the layers A𝐴A and B𝐵B are initialized and held fixed.

The network functionality is summarized as follows

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐡i,0subscript𝐡  𝑖0\displaystyle\mathbf{h}\_{i,0} | =σ​(A​𝐱i)absent𝜎𝐴subscript𝐱𝑖\displaystyle=\sigma(A\mathbf{x}\_{i}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐡i,l(t)superscriptsubscript𝐡  𝑖𝑙𝑡\displaystyle\mathbf{h}\_{i,l}^{(t)} | =σ​(Wl(t)​𝐡i,l−1(t))absent𝜎superscriptsubscript𝑊𝑙𝑡superscriptsubscript𝐡  𝑖𝑙1𝑡\displaystyle=\sigma(W\_{l}^{(t)}\mathbf{h}\_{i,l-1}^{(t)}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐮i(t)superscriptsubscript𝐮𝑖𝑡\displaystyle\mathbf{u}\_{i}^{(t)} | =B​𝐡i,L(t)absent𝐵superscriptsubscript𝐡  𝑖𝐿𝑡\displaystyle=B\mathbf{h}\_{i,L}^{(t)} |  |

where i∈[n]𝑖delimited-[]𝑛i\in\left[n\right], l∈[L]𝑙delimited-[]𝐿l\in\left[L\right] and t𝑡t denotes iteration number. In addition, for each input vector i∈[n]𝑖delimited-[]𝑛i\in\left[n\right] and layer l∈{0,1,…,L}𝑙01…𝐿l\in\{0,1,...,L\}, we associate a diagonal matrix Di,lsubscript𝐷

𝑖𝑙D\_{i,l} such that for j∈[m]𝑗delimited-[]𝑚j\in\left[m\right], (Di,l)j,j=𝕀(Wl​𝐡i,l−1)j≥0subscriptsubscript𝐷

𝑖𝑙

𝑗𝑗subscript𝕀subscriptsubscript𝑊𝑙subscript𝐡

𝑖𝑙1𝑗0(D\_{i,l})\_{j,j}=\mathbb{I}\_{(W\_{l}\mathbf{h}\_{i,l-1})\_{j}\geq 0}, where we use the convention 𝐡i,−1=𝐱isubscript𝐡

𝑖1subscript𝐱𝑖\mathbf{h}\_{i,-1}=\mathbf{x}\_{i}.
The network is trained to minimize the ℓ2subscriptℓ2\ell\_{2} loss

|  |  |  |
| --- | --- | --- |
|  | Φ​(W)=12​∑i=1n(yi−f​(𝐱i;W))2Φ𝑊12superscriptsubscript𝑖1𝑛superscriptsubscript𝑦𝑖𝑓  subscript𝐱𝑖𝑊2\Phi(W)=\frac{1}{2}\sum\_{i=1}^{n}(y\_{i}-f(\mathbf{x}\_{i};W))^{2} |  |

We will analyze the properties of the matrices H,H∞∈ℝn×n

𝐻superscript𝐻
superscriptℝ𝑛𝑛H,H^{\infty}\in\mathbb{R}^{n\times n}, comprised of the following entries

|  |  |  |
| --- | --- | --- |
|  | Hi​j​(t)=⟨∂ui(t)∂W,∂uj(t)∂W⟩subscript𝐻𝑖𝑗𝑡  superscriptsubscript𝑢𝑖𝑡𝑊superscriptsubscript𝑢𝑗𝑡𝑊H\_{ij}(t)=\left\langle\frac{\partial u\_{i}^{(t)}}{\partial W},\frac{\partial u\_{j}^{(t)}}{\partial W}\right\rangle |  |

|  |  |  |
| --- | --- | --- |
|  | Hi​j∞=𝔼W​⟨∂ui(0)∂W,∂uj(0)∂W⟩.superscriptsubscript𝐻𝑖𝑗subscript𝔼𝑊  superscriptsubscript𝑢𝑖0𝑊superscriptsubscript𝑢𝑗0𝑊H\_{ij}^{\infty}=\mathbb{E}\_{W}\left\langle\frac{\partial u\_{i}^{(0)}}{\partial W},\frac{\partial u\_{j}^{(0)}}{\partial W}\right\rangle. |  |

We write the eigen-decomposition of H∞=∑i=1nλi​𝐯i​𝐯iTsuperscript𝐻superscriptsubscript𝑖1𝑛subscript𝜆𝑖subscript𝐯𝑖superscriptsubscript𝐯𝑖𝑇H^{\infty}=\sum\_{i=1}^{n}\lambda\_{i}\mathbf{v}\_{i}\mathbf{v}\_{i}^{T}, where 𝐯1,…,𝐯n

subscript𝐯1…subscript𝐯𝑛\mathbf{v}\_{1},\ldots,\mathbf{v}\_{n} are the eigenvectors of H∞superscript𝐻H^{\infty} and λ1,…,λn

subscript𝜆1…subscript𝜆𝑛\lambda\_{1},\ldots,\lambda\_{n} are their corresponding eigenvalues. The minimal eigenvalue is denoted by λ0=min⁡(λ​(H∞))subscript𝜆0𝜆superscript𝐻\lambda\_{0}=\min(\lambda(H^{\infty})).

###### Theorem 4.

For any ϵ∈(0,1]italic-ϵ01\epsilon\in(0,1] and δ∈(0,O​(1L)]𝛿0𝑂1𝐿\delta\in(0,O(\frac{1}{L})], let τ=Θ​(ϵ​δ^n)𝜏Θitalic-ϵ^𝛿𝑛\tau=\Theta(\frac{\epsilon\hat{\delta}}{n}), m≥Ω​(n24​L12​log5⁡mδ8​τ6)𝑚Ωsuperscript𝑛24superscript𝐿12superscript5𝑚superscript𝛿8superscript𝜏6m\geq\Omega\left(\frac{n^{24}L^{12}\log^{5}m}{\delta^{8}\tau^{6}}\right), η=Θ​(δn4​L2​m​τ2)𝜂Θ𝛿superscript𝑛4superscript𝐿2𝑚superscript𝜏2\eta=\Theta\left(\frac{\delta}{n^{4}L^{2}m\tau^{2}}\right). Then, with probability of at least 1−δ^1^𝛿1-\hat{\delta} over the random initialization after t𝑡t iterations of GD we have that

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖𝐲−𝐮(t)‖=∑i=1n(1−η​λi)2​t​(𝐯iT​𝐲)2±ϵ.norm𝐲superscript𝐮𝑡plus-or-minussuperscriptsubscript𝑖1𝑛superscript1𝜂subscript𝜆𝑖2𝑡superscriptsuperscriptsubscript𝐯𝑖𝑇𝐲2italic-ϵ\|\mathbf{y}-\mathbf{u}^{(t)}\|=\sqrt{\sum\_{i=1}^{n}(1-\eta\lambda\_{i})^{2t}(\mathbf{v}\_{i}^{T}\mathbf{y})^{2}}\,\pm\epsilon. |  | (50) |

### D.2 Proof strategy

The proof of Thm. [4](#Thmtheorem4 "Theorem 4. ‣ D.1 The network model ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density") relies on a theorem, provided by (Allen-Zhu et al., [2019](#bib.bib2)), stated in Thm. [5](#Thmtheorem5 "Theorem 5. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"), and an observation, based the on the derivation of the proof to that theorem, which we state in Lemma [4](#Thmlemma4 "Lemma 4. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density").

Thm. [5](#Thmtheorem5 "Theorem 5. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density") assumes that the data is normalized, so that ∥𝐱i∥=1delimited-∥∥subscript𝐱𝑖1\left\lVert\mathbf{x}\_{i}\right\rVert=1, and there exists δ∈(0,O​(1L)]𝛿0𝑂1𝐿\delta\in(0,O(\frac{1}{L})] such that for every pair i,j∈[n]

𝑖𝑗
delimited-[]𝑛i,j\in[n], we have ∥𝐱i−𝐱j∥≥δdelimited-∥∥subscript𝐱𝑖subscript𝐱𝑗𝛿\left\lVert\mathbf{x}\_{i}-\mathbf{x}\_{j}\right\rVert\geq\delta and also it holds that |yi|≤O(1)\left|y\_{i}\right\rvert\leq O(1).

In addition, we prove Lemma [3](#Thmlemma3 "Lemma 3. ‣ D.2 Proof strategy ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"), which is the basis for the proof of our Theorem.

###### Lemma 3.

Suppose δ∈(0,O​(1L)]𝛿0𝑂1𝐿\delta\in(0,O(\frac{1}{L})], m≥Ω​(n24​L12​log5⁡mδ8​τ2)𝑚Ωsuperscript𝑛24superscript𝐿12superscript5𝑚superscript𝛿8superscript𝜏2m\geq\Omega\left(\frac{n^{24}L^{12}\log^{5}m}{\delta^{8}\tau^{2}}\right), η=Θ​(δn4​L2​m​τ2)𝜂Θ𝛿superscript𝑛4superscript𝐿2𝑚superscript𝜏2\eta=\Theta\left(\frac{\delta}{n^{4}L^{2}m\tau^{2}}\right) and also let ω=O​(n3​log⁡mδ​τ​m)𝜔𝑂superscript𝑛3𝑚𝛿𝜏𝑚\omega=O(\frac{n^{3}\log m}{\delta\tau\sqrt{m}}). Then, with probability at least 1−e−Ω​(m​ω2/3​L)1superscript𝑒Ω𝑚superscript𝜔23𝐿1-e^{-\Omega(m\omega^{2/3}L)} over the randomness of A,B

𝐴𝐵A,B and W(0)superscript𝑊0W^{(0)} we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐮​(t+1)−𝐲=(I−η​H​(t))​(𝐮​(t)−𝐲)+ϵ​(t)𝐮𝑡1𝐲𝐼𝜂𝐻𝑡𝐮𝑡𝐲italic-ϵ𝑡\displaystyle\mathbf{u}(t+1)-\mathbf{y}=(I-\eta H(t))(\mathbf{u}(t)-\mathbf{y})+\epsilon(t) |  | (51) |

with

|  |  |  |
| --- | --- | --- |
|  | ∥ϵ​(t)∥≤O​(L​log4/3⁡mτ1/3​m1/6​n1.5)​Φ​(W(t))+O​(δ2τ​n6​m0.5​L1.5)​Φ​(W(t))delimited-∥∥italic-ϵ𝑡𝑂𝐿superscript43𝑚superscript𝜏13superscript𝑚16superscript𝑛1.5Φsuperscript𝑊𝑡𝑂superscript𝛿2𝜏superscript𝑛6superscript𝑚0.5superscript𝐿1.5Φsuperscript𝑊𝑡\left\lVert\epsilon(t)\right\rVert\leq O\left(\frac{L\log^{4/3}m}{\tau^{1/3}m^{1/6}n^{1.5}}\right)\sqrt{\Phi(W^{(t)})}+O\left(\frac{\delta^{2}}{\tau n^{6}m^{0.5}L^{1.5}}\right)\Phi(W^{(t)}) |  |

The proof of the Lemma is deferred, and will be given after the proof of the theorem.

### D.3 Proof of Thm [4](#Thmtheorem4 "Theorem 4. ‣ D.1 The network model ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")

###### Proof.

By Lemma [3](#Thmlemma3 "Lemma 3. ‣ D.2 Proof strategy ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density") we have the following relation

|  |  |  |
| --- | --- | --- |
|  | 𝐮​(t)−𝐲=(I−η​H​(t−1))​(𝐮​(t−1)−𝐲)+ϵ​(t−1)𝐮𝑡𝐲𝐼𝜂𝐻𝑡1𝐮𝑡1𝐲italic-ϵ𝑡1\mathbf{u}(t)-\mathbf{y}=(I-\eta H(t-1))(\mathbf{u}(t-1)-\mathbf{y})+\epsilon(t-1) |  |

Adding and subtracting η​H∞​(𝐮​(t−1)−𝐲)𝜂superscript𝐻𝐮𝑡1𝐲\eta H^{\infty}(\mathbf{u}(t-1)-\mathbf{y}) we have

|  |  |  |
| --- | --- | --- |
|  | 𝐮​(t)−𝐲=(I−η​H∞)​(𝐮​(t−1)−𝐲)+η​(H∞−H​(t−1))​(𝐮​(t−1)−𝐲)+ϵ​(t−1)𝐮𝑡𝐲𝐼𝜂superscript𝐻𝐮𝑡1𝐲𝜂superscript𝐻𝐻𝑡1𝐮𝑡1𝐲italic-ϵ𝑡1\mathbf{u}(t)-\mathbf{y}=(I-\eta H^{\infty})(\mathbf{u}(t-1)-\mathbf{y})+\eta(H^{\infty}-H(t-1))(\mathbf{u}(t-1)-\mathbf{y})+\epsilon(t-1) |  |

and this is equivalent to

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐮​(t)−𝐲=(I−η​H∞)​(𝐮​(t−1)−𝐲)+ξ​(t−1).𝐮𝑡𝐲𝐼𝜂superscript𝐻𝐮𝑡1𝐲𝜉𝑡1\mathbf{u}(t)-\mathbf{y}=(I-\eta H^{\infty})(\mathbf{u}(t-1)-\mathbf{y})+\xi(t-1). |  | (52) |

where we denote ξ​(t)=η​(H∞−H​(t))​(𝐮​(t)−𝐲)+ϵ​(t)𝜉𝑡𝜂superscript𝐻𝐻𝑡𝐮𝑡𝐲italic-ϵ𝑡\xi(t)=\eta(H^{\infty}-H(t))(\mathbf{u}(t)-\mathbf{y})+\epsilon(t). Then, by applying ([52](#A4.E52 "In Proof. ‣ D.3 Proof of Thm 4 ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) recursively, we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐮​(t)−𝐲=(I−η​H∞)t​(𝐮​(0)−𝐲)+∑i=0t−1(I−η​H∞)i​ξ​(t−1−i)𝐮𝑡𝐲superscript𝐼𝜂superscript𝐻𝑡𝐮0𝐲superscriptsubscript𝑖0𝑡1superscript𝐼𝜂superscript𝐻𝑖𝜉𝑡1𝑖\mathbf{u}(t)-\mathbf{y}=(I-\eta H^{\infty})^{t}(\mathbf{u}(0)-\mathbf{y})+\sum\_{i=0}^{t-1}(I-\eta H^{\infty})^{i}\xi(t-1-i)\\ |  | (53) |

We first bound the quantity ∥ξ​(t−1−i)∥2subscriptdelimited-∥∥𝜉𝑡1𝑖2\left\lVert\xi(t-1-i)\right\rVert\_{2}

|  |  |  |
| --- | --- | --- |
|  | ∥ξ​(t−1−i)∥2=∥η​(H​(t−1−i)−H∞)​(y−u​(t−1−i))+ϵ​(t−1−i)∥2subscriptdelimited-∥∥𝜉𝑡1𝑖2subscriptdelimited-∥∥𝜂𝐻𝑡1𝑖superscript𝐻𝑦𝑢𝑡1𝑖italic-ϵ𝑡1𝑖2\displaystyle\left\lVert\xi(t-1-i)\right\rVert\_{2}=\left\lVert\eta(H(t-1-i)-H^{\infty})(y-u(t-1-i))+\epsilon(t-1-i)\right\rVert\_{2} |  |
|  |  |  |
| --- | --- | --- |
|  | ≤∥η​(H​(t−1−i)−H∞)∥2​∥(y−u​(t−1−i))∥2+∥ϵ​(t−1−i)∥2absentsubscriptdelimited-∥∥𝜂𝐻𝑡1𝑖superscript𝐻2subscriptdelimited-∥∥𝑦𝑢𝑡1𝑖2subscriptdelimited-∥∥italic-ϵ𝑡1𝑖2\displaystyle\leq\left\lVert\eta(H(t-1-i)-H^{\infty})\right\rVert\_{2}\left\lVert(y-u(t-1-i))\right\rVert\_{2}+\left\lVert\epsilon(t-1-i)\right\rVert\_{2} |  |
|  |  |  |
| --- | --- | --- |
|  | η≤1,2O(δ2​m​τ3n6)Φ​(W(t−1−i)))+O(δ2τ​n6​m0.5​L1.5)Φ(W(t−1−i))+O(L​log4/3⁡mτ1/3​m1/6​n1.5)Φ​(W(t−1−i))\displaystyle\eta\leq^{{}^{1,2}}{O\left(\frac{\delta^{2}m\tau^{3}}{n^{6}}\right)}\sqrt{\Phi(W^{(t-1-i)})})+O\left(\frac{\delta^{2}}{\tau n^{6}m^{0.5}L^{1.5}}\right)\Phi(W^{(t-1-i)})+O\left(\frac{L\log^{4/3}m}{\tau^{1/3}m^{1/6}n^{1.5}}\right)\sqrt{\Phi(W^{(t-1-i)})} |  |
|  |  |  |
| --- | --- | --- |
|  | ≤3(1−Ω(τ2​η​δ​mn2))t−1−i2(ηO(δ2​m​τ3n6)Φ(W(0))+O(δ2τ​n6​m0.5​L1.5)Φ(W(0))+O(L​log4/3⁡mτ1/3​m1/6​n1.5)Φ​(W(0)))\displaystyle\leq^{{}^{3}}\left(1-\Omega\left(\frac{\tau^{2}\eta\delta m}{n^{2}}\right)\right)^{\frac{t-1-i}{2}}\left({\eta O\left(\frac{\delta^{2}m\tau^{3}}{n^{6}}\right)}\sqrt{\Phi(W^{(0)}})+O\left(\frac{\delta^{2}}{\tau n^{6}m^{0.5}L^{1.5}}\right)\Phi(W^{(0)})+O\left(\frac{L\log^{4/3}m}{\tau^{1/3}m^{1/6}n^{1.5}}\right)\sqrt{\Phi(W^{(0)})}\right) |  |
|  |  |  |
| --- | --- | --- |
|  | ≤4(1−Ω​(τ2​η​δ​mn2))t−1−i2​(η​O​(n)​O​(δ2​m​τ3n6)+O​(δ2τ​n6​m0.5​L1.5)​O​(n)+O​(L​log4/3⁡mτ1/3​m1/6​n1.5)​O​(n))superscript4absentsuperscript1Ωsuperscript𝜏2𝜂𝛿𝑚superscript𝑛2𝑡1𝑖2𝜂𝑂𝑛𝑂superscript𝛿2𝑚superscript𝜏3superscript𝑛6𝑂superscript𝛿2𝜏superscript𝑛6superscript𝑚0.5superscript𝐿1.5𝑂𝑛𝑂𝐿superscript43𝑚superscript𝜏13superscript𝑚16superscript𝑛1.5𝑂𝑛\displaystyle\leq^{{}^{4}}\left(1-\Omega\left(\frac{\tau^{2}\eta\delta m}{n^{2}}\right)\right)^{\frac{t-1-i}{2}}\left(\eta O\left(\sqrt{n}\right){O\left(\frac{\delta^{2}m\tau^{3}}{n^{6}}\right)}+O\left(\frac{\delta^{2}}{\tau n^{6}m^{0.5}L^{1.5}}\right)O\left(n\right)+O\left(\frac{L\log^{4/3}m}{\tau^{1/3}m^{1/6}n^{1.5}}\right)O\left(\sqrt{n}\right)\right) |  |
|  |  |  |
| --- | --- | --- |
|  | =(1−Ω​(τ2​η​δ​mn2))(t−1−i)2​(η​O​(δ2​m​τ3n5.5)+O​(δ2τ​n5​m0.5​L1.5)+O​(L​log4/3⁡mτ1/3​m1/6​n))absentsuperscript1Ωsuperscript𝜏2𝜂𝛿𝑚superscript𝑛2𝑡1𝑖2𝜂𝑂superscript𝛿2𝑚superscript𝜏3superscript𝑛5.5𝑂superscript𝛿2𝜏superscript𝑛5superscript𝑚0.5superscript𝐿1.5𝑂𝐿superscript43𝑚superscript𝜏13superscript𝑚16𝑛\displaystyle=\left(1-\Omega\left(\frac{\tau^{2}\eta\delta m}{n^{2}}\right)\right)^{\frac{(t-1-i)}{2}}\left({\eta O\left(\frac{\delta^{2}m\tau^{3}}{n^{5.5}}\right)}+O\left(\frac{\delta^{2}}{\tau n^{5}m^{0.5}L^{1.5}}\right)+O\left(\frac{L\log^{4/3}m}{\tau^{1/3}m^{1/6}n}\right)\right) |  |

where we make the following derivations

1. 1.

   Using Lemma [14](#Thmlemma14 "Lemma 14. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density") which states that ∥H​(t)−H∞∥2≤O​(δ2​m​τ3n6)subscriptdelimited-∥∥𝐻𝑡superscript𝐻2𝑂superscript𝛿2𝑚superscript𝜏3superscript𝑛6\left\lVert H(t)-H^{\infty}\right\rVert\_{2}\leq O(\frac{\delta^{2}m\tau^{3}}{n^{6}}).
2. 2.

   Using the bound in Lemma [3](#Thmlemma3 "Lemma 3. ‣ D.2 Proof strategy ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"), for ϵ​(t−1−i)italic-ϵ𝑡1𝑖\epsilon(t-1-i)
3. 3.

   Using bound over the loss by, Lemma [4](#Thmlemma4 "Lemma 4. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density") (b).
4. 4.

   By Lemma [11](#Thmlemma11 "Lemma 11. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density") the loss at initialization is bounded by O​(n)𝑂𝑛O(n).

Using the bound, derived above, ([53](#A4.E53 "In Proof. ‣ D.3 Proof of Thm 4 ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) yields

|  |  |  |
| --- | --- | --- |
|  | ∥𝐮​(t)−𝐲∥=∥(I−η​H∞)t​(𝐮​(0)−𝐲)+∑i=0t−1((I−η​H∞)i​ξ​(t−1−i))∥delimited-∥∥𝐮𝑡𝐲delimited-∥∥superscript𝐼𝜂superscript𝐻𝑡𝐮0𝐲superscriptsubscript𝑖0𝑡1superscript𝐼𝜂superscript𝐻𝑖𝜉𝑡1𝑖\displaystyle\left\lVert\mathbf{u}(t)-\mathbf{y}\right\rVert=\left\lVert(I-\eta H^{\infty})^{t}(\mathbf{u}(0)-\mathbf{y})+\sum\_{i=0}^{t-1}((I-\eta H^{\infty})^{i}\xi(t-1-i))\right\rVert |  |
|  |  |  |
| --- | --- | --- |
|  | ≤1∥(I−η​H∞)t​(𝐮​(0)−𝐲)∥superscript1absentdelimited-∥∥superscript𝐼𝜂superscript𝐻𝑡𝐮0𝐲\displaystyle\leq^{{}^{1}}\left\lVert(I-\eta H^{\infty})^{t}(\mathbf{u}(0)-\mathbf{y})\right\rVert |  |
|  |  |  |
| --- | --- | --- |
|  | +∑i=0t−1(1−η​λ0)i​(1−Ω​(τ2​η​δ​mn2))(t−1−i)2​(η​O​(δ2​m​τ3n5.5)+O​(δ2τ​n5​m0.5​L1.5)+O​(L​log4/3⁡mτ1/3​m1/6​n))superscriptsubscript𝑖0𝑡1superscript1𝜂subscript𝜆0𝑖superscript1Ωsuperscript𝜏2𝜂𝛿𝑚superscript𝑛2𝑡1𝑖2𝜂𝑂superscript𝛿2𝑚superscript𝜏3superscript𝑛5.5𝑂superscript𝛿2𝜏superscript𝑛5superscript𝑚0.5superscript𝐿1.5𝑂𝐿superscript43𝑚superscript𝜏13superscript𝑚16𝑛\displaystyle+\sum\_{i=0}^{t-1}(1-\eta\lambda\_{0})^{i}\left(1-\Omega\left(\frac{\tau^{2}\eta\delta m}{n^{2}}\right)\right)^{\frac{(t-1-i)}{2}}\left({\eta O\left(\frac{\delta^{2}m\tau^{3}}{n^{5.5}}\right)}+O\left(\frac{\delta^{2}}{\tau n^{5}m^{0.5}L^{1.5}}\right)+O\left(\frac{L\log^{4/3}m}{\tau^{1/3}m^{1/6}n}\right)\right) |  |
|  |  |  |
| --- | --- | --- |
|  | ≤2∥(I−η​H∞)t​(𝐮​(0)−𝐲)∥+t​(η​O​(δ2​m​τ3n5.5)+O​(δ2τ​n5​m0.5​L1.5)+O​(L​log4/3⁡mτ1/3​m1/6​n))superscript2absentdelimited-∥∥superscript𝐼𝜂superscript𝐻𝑡𝐮0𝐲𝑡𝜂𝑂superscript𝛿2𝑚superscript𝜏3superscript𝑛5.5𝑂superscript𝛿2𝜏superscript𝑛5superscript𝑚0.5superscript𝐿1.5𝑂𝐿superscript43𝑚superscript𝜏13superscript𝑚16𝑛\displaystyle\leq^{{}^{2}}\left\lVert(I-\eta H^{\infty})^{t}(\mathbf{u}(0)-\mathbf{y})\right\rVert+t\left({\eta O\left(\frac{\delta^{2}m\tau^{3}}{n^{5.5}}\right)}+O\left(\frac{\delta^{2}}{\tau n^{5}m^{0.5}L^{1.5}}\right)+O\left(\frac{L\log^{4/3}m}{\tau^{1/3}m^{1/6}n}\right)\right) |  |
|  |  |  |
| --- | --- | --- |
|  | ≤3∥(I−η​H∞)t​(𝐮​(0)−𝐲)∥+O​(n6​L2δ2)​(η​O​(δ2​m​τ3n5.5)+O​(δ2τ​n5​m0.5​L1.5)+O​(L​log4/3⁡mτ1/3​m1/6​n))superscript3absentdelimited-∥∥superscript𝐼𝜂superscript𝐻𝑡𝐮0𝐲𝑂superscript𝑛6superscript𝐿2superscript𝛿2𝜂𝑂superscript𝛿2𝑚superscript𝜏3superscript𝑛5.5𝑂superscript𝛿2𝜏superscript𝑛5superscript𝑚0.5superscript𝐿1.5𝑂𝐿superscript43𝑚superscript𝜏13superscript𝑚16𝑛\displaystyle\leq^{{}^{3}}\left\lVert(I-\eta H^{\infty})^{t}(\mathbf{u}(0)-\mathbf{y})\right\rVert+O\left(\frac{n^{6}L^{2}}{\delta^{2}}\right)\left({\eta O\left(\frac{\delta^{2}m\tau^{3}}{n^{5.5}}\right)}+O\left(\frac{\delta^{2}}{\tau n^{5}m^{0.5}L^{1.5}}\right)+O\left(\frac{L\log^{4/3}m}{\tau^{1/3}m^{1/6}n}\right)\right) |  |
|  |  |  |
| --- | --- | --- |
|  | ≤∥(I−η​H∞)t∥​∥𝐮​(0)∥+∥(I−η​H∞)t​𝐲∥+O​(n6​L2δ2)​(η​O​(δ2​m​τ3n5.5)+O​(δ2τ​n5​m0.5​L1.5)+O​(L​log4/3⁡mτ1/3​m1/6​n))absentdelimited-∥∥superscript𝐼𝜂superscript𝐻𝑡delimited-∥∥𝐮0delimited-∥∥superscript𝐼𝜂superscript𝐻𝑡𝐲𝑂superscript𝑛6superscript𝐿2superscript𝛿2𝜂𝑂superscript𝛿2𝑚superscript𝜏3superscript𝑛5.5𝑂superscript𝛿2𝜏superscript𝑛5superscript𝑚0.5superscript𝐿1.5𝑂𝐿superscript43𝑚superscript𝜏13superscript𝑚16𝑛\displaystyle\leq\left\lVert(I-\eta H^{\infty})^{t}\right\rVert\left\lVert\mathbf{u}(0)\right\rVert+\left\lVert(I-\eta H^{\infty})^{t}\mathbf{y}\right\rVert+O\left(\frac{n^{6}L^{2}}{\delta^{2}}\right)\left({\eta O\left(\frac{\delta^{2}m\tau^{3}}{n^{5.5}}\right)}+O\left(\frac{\delta^{2}}{\tau n^{5}m^{0.5}L^{1.5}}\right)+O\left(\frac{L\log^{4/3}m}{\tau^{1/3}m^{1/6}n}\right)\right) |  |

where we make the following derivations

1. 1.

   ∥I−η​H∞∥2subscriptdelimited-∥∥𝐼𝜂superscript𝐻2\left\lVert I-\eta H^{\infty}\right\rVert\_{2} is bounded by the maximal eigenvalue of the positive definite matrix (I−η​H∞)𝐼𝜂superscript𝐻(I-\eta H^{\infty}), i.e, (1−η​λ0)1𝜂subscript𝜆0(1-\eta\lambda\_{0}).
2. 2.

   (1−η​λ0)i​(1−Ω​(τ2​η​δ​mn2))(t−1−i)2≤1superscript1𝜂subscript𝜆0𝑖superscript1Ωsuperscript𝜏2𝜂𝛿𝑚superscript𝑛2𝑡1𝑖21(1-\eta\lambda\_{0})^{i}\left(1-\Omega\left(\frac{\tau^{2}\eta\delta m}{n^{2}}\right)\right)^{\frac{(t-1-i)}{2}}\leq 1
3. 3.

   By Theorem [5](#Thmtheorem5 "Theorem 5. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"), t≤O​(n6​L2δ2)𝑡𝑂superscript𝑛6superscript𝐿2superscript𝛿2t\leq O(\frac{n^{6}L^{2}}{\delta^{2}})

Next, it is straightforward to show that

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∥(I−η​H∞)t​𝐲∥=∑i=1n(1−η​λi)2​t​(𝐯iT​𝐲)2delimited-∥∥superscript𝐼𝜂superscript𝐻𝑡𝐲superscriptsubscript𝑖1𝑛superscript1𝜂subscript𝜆𝑖2𝑡superscriptsuperscriptsubscript𝐯𝑖𝑇𝐲2\left\lVert(I-\eta H^{\infty})^{t}\mathbf{y}\right\rVert=\sqrt{\sum\_{i=1}^{n}(1-\eta\lambda\_{i})^{2t}(\mathbf{v}\_{i}^{T}\mathbf{y})^{2}} |  | (54) |

where λi,𝐯i

subscript𝜆𝑖subscript𝐯𝑖\lambda\_{i},\mathbf{v}\_{i} are the eigenvalues and eigenvectors of H∞superscript𝐻H^{\infty}, respectively.

For the first term we use lemma [11](#Thmlemma11 "Lemma 11. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density") which states that ∥𝐮​(0)∥≤n​τδ^delimited-∥∥𝐮0𝑛𝜏^𝛿\left\lVert\mathbf{u}(0)\right\rVert\leq\frac{\sqrt{n}\tau}{\hat{\delta}}, and by our choice of τ𝜏\tau we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∥(I−η​H∞)t∥​∥𝐮​(0)∥≤(1−η​λ0)t​O​(n​τδ^)≤ϵdelimited-∥∥superscript𝐼𝜂superscript𝐻𝑡delimited-∥∥𝐮0superscript1𝜂subscript𝜆0𝑡𝑂𝑛𝜏^𝛿italic-ϵ\left\lVert(I-\eta H^{\infty})^{t}\right\rVert\left\lVert\mathbf{u}(0)\right\rVert\leq(1-\eta\lambda\_{0})^{t}O\left(\frac{\sqrt{n}\tau}{\hat{\delta}}\right)\leq\epsilon |  | (55) |

Finally, by our choice of η,m,τ

𝜂𝑚𝜏\eta,m,\tau it holds that

|  |  |  |  |
| --- | --- | --- | --- |
|  | O​(n6​L2δ2)​(O​(δ2​m​τ3n5.5)​η+O​(δ2τ​n5​m0.5​L1.5)+O​(L​log4/3⁡mτ1/3​m1/6​n))≤ϵ𝑂superscript𝑛6superscript𝐿2superscript𝛿2𝑂superscript𝛿2𝑚superscript𝜏3superscript𝑛5.5𝜂𝑂superscript𝛿2𝜏superscript𝑛5superscript𝑚0.5superscript𝐿1.5𝑂𝐿superscript43𝑚superscript𝜏13superscript𝑚16𝑛italic-ϵO\left(\frac{n^{6}L^{2}}{\delta^{2}}\right)\left({O\left(\frac{\delta^{2}m\tau^{3}}{n^{5.5}}\right)}\eta+O\left(\frac{\delta^{2}}{\tau n^{5}m^{0.5}L^{1.5}}\right)+O\left(\frac{L\log^{4/3}m}{\tau^{1/3}m^{1/6}n}\right)\right)\leq\epsilon |  | (56) |

Combining ([54](#A4.E54 "In Proof. ‣ D.3 Proof of Thm 4 ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")), ([55](#A4.E55 "In Proof. ‣ D.3 Proof of Thm 4 ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) and ([56](#A4.E56 "In Proof. ‣ D.3 Proof of Thm 4 ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")) yields

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∥𝐲−𝐮​(t)∥=∑i=1n(1−η​λi)2​k​(𝐯iT​𝐲)2±ϵdelimited-∥∥𝐲𝐮𝑡plus-or-minussuperscriptsubscript𝑖1𝑛superscript1𝜂subscript𝜆𝑖2𝑘superscriptsuperscriptsubscript𝐯𝑖𝑇𝐲2italic-ϵ\left\lVert\mathbf{y}-\mathbf{u}(t)\right\rVert=\sqrt{\sum\_{i=1}^{n}(1-\eta\lambda\_{i})^{2k}(\mathbf{v}\_{i}^{T}\mathbf{y})^{2}}\pm{\epsilon} |  | (57) |

∎

### D.4 Supporting Lemmas

###### Proof.

Proof of Lemma [3](#Thmlemma3 "Lemma 3. ‣ D.2 Proof strategy ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density").

By construction

|  |  |  |  |
| --- | --- | --- | --- |
|  | ϵi​(t)subscriptitalic-ϵ𝑖𝑡\displaystyle\epsilon\_{i}(t) | =ui​(t+1)−ui​(t)+[η​H​(t)​(𝐮​(t)−𝐲)]iabsentsubscript𝑢𝑖𝑡1subscript𝑢𝑖𝑡subscriptdelimited-[]𝜂𝐻𝑡𝐮𝑡𝐲𝑖\displaystyle=u\_{{}\_{i}}(t+1)-u\_{i}(t)+\left[\eta H(t)(\mathbf{u}(t)-\mathbf{y})\right]\_{i} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =ui​(t+1)−ui​(t)+η​∑j=1n(uj​(t)−yj)​Hi​j​(t)absentsubscript𝑢𝑖𝑡1subscript𝑢𝑖𝑡𝜂superscriptsubscript𝑗1𝑛subscript𝑢𝑗𝑡subscript𝑦𝑗subscript𝐻𝑖𝑗𝑡\displaystyle=u\_{i}(t+1)-u\_{i}(t)+\eta\sum\_{j=1}^{n}(u\_{j}(t)-y\_{j})H\_{ij}(t) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =ui​(t+1)−ui​(t)+η​∑j=1n(uj​(t)−yj)​⟨∂ui​(t)∂W,∂uj​(t)∂W⟩absentsubscript𝑢𝑖𝑡1subscript𝑢𝑖𝑡𝜂superscriptsubscript𝑗1𝑛subscript𝑢𝑗𝑡subscript𝑦𝑗  subscript𝑢𝑖𝑡𝑊subscript𝑢𝑗𝑡𝑊\displaystyle=u\_{i}(t+1)-u\_{i}(t)+\eta\sum\_{j=1}^{n}(u\_{j}(t)-y\_{j})\left\langle\frac{\partial u\_{i}(t)}{\partial W},\frac{\partial u\_{j}(t)}{\partial W}\right\rangle |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =ui​(t+1)−ui​(t)+η​⟨∂ui​(t)∂W,∑j=1n(uj​(t)−yj)​∂uj​(t)∂W⟩absentsubscript𝑢𝑖𝑡1subscript𝑢𝑖𝑡𝜂  subscript𝑢𝑖𝑡𝑊superscriptsubscript𝑗1𝑛subscript𝑢𝑗𝑡subscript𝑦𝑗subscript𝑢𝑗𝑡𝑊\displaystyle=u\_{i}(t+1)-u\_{i}(t)+\eta\left\langle\frac{\partial u\_{i}(t)}{\partial W},\sum\_{j=1}^{n}(u\_{j}(t)-y\_{j})\frac{\partial u\_{j}(t)}{\partial W}\right\rangle |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =ui​(t+1)−ui​(t)+η​⟨∂ui∂W,∇Φ​(W(t))⟩.absentsubscript𝑢𝑖𝑡1subscript𝑢𝑖𝑡𝜂  subscript𝑢𝑖𝑊∇Φsuperscript𝑊𝑡\displaystyle=u\_{i}(t+1)-u\_{i}(t)+\eta\left\langle\frac{\partial u\_{i}}{\partial W},\nabla\Phi(W^{(t)})\right\rangle. |  |

We denote −η​∇Φ​(W(t))𝜂∇Φsuperscript𝑊𝑡-\eta\nabla\Phi(W^{(t)}) by W′=(W1′,…,WL′)superscript𝑊′superscriptsubscript𝑊1′…superscriptsubscript𝑊𝐿′W^{\prime}=(W\_{1}^{{}^{\prime}},...,W\_{L}^{{}^{\prime}}), yielding

|  |  |  |  |
| --- | --- | --- | --- |
|  | ϵi​(t)subscriptitalic-ϵ𝑖𝑡\displaystyle\epsilon\_{i}(t) | =ui​(t+1)−ui​(t)−⟨∂ui​(t)∂W,W′⟩absentsubscript𝑢𝑖𝑡1subscript𝑢𝑖𝑡  subscript𝑢𝑖𝑡𝑊superscript𝑊′\displaystyle=u\_{i}{(t+1)}-u\_{i}{(t)}-\left\langle\frac{\partial u\_{i}(t)}{\partial W},W^{\prime}\right\rangle |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =B​(hi,L(t+1)−hi,L(t))−⟨∂ui​(t)∂W,W′⟩absent𝐵superscriptsubscriptℎ  𝑖𝐿𝑡1superscriptsubscriptℎ  𝑖𝐿𝑡  subscript𝑢𝑖𝑡𝑊superscript𝑊′\displaystyle=B(h\_{i,L}^{(t+1)}-h\_{i,L}^{(t)})-\left\langle\frac{\partial u\_{i}(t)}{\partial W},W^{\prime}\right\rangle |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =B​(hi,L(t+1)−hi,L(t)−∑l=1LDi,L(t)​WL(t)​Di,L−1(t)​WL−1(t)​⋯​Di,L+1(t)​Wl+1(t)​Di,l(t)​Wl′​hi,l−1(t))absent𝐵superscriptsubscriptℎ  𝑖𝐿𝑡1superscriptsubscriptℎ  𝑖𝐿𝑡superscriptsubscript𝑙1𝐿superscriptsubscript𝐷  𝑖𝐿𝑡superscriptsubscript𝑊𝐿𝑡superscriptsubscript𝐷  𝑖𝐿1𝑡superscriptsubscript𝑊𝐿1𝑡⋯superscriptsubscript𝐷  𝑖𝐿1𝑡superscriptsubscript𝑊𝑙1𝑡superscriptsubscript𝐷  𝑖𝑙𝑡superscriptsubscript𝑊𝑙′superscriptsubscriptℎ  𝑖𝑙1𝑡\displaystyle=B(h\_{i,L}^{(t+1)}-h\_{i,L}^{(t)}-\sum\_{l=1}^{L}D\_{i,L}^{(t)}W\_{L}^{(t)}D\_{i,L-1}^{(t)}W\_{L-1}^{(t)}\cdots D\_{i,L+1}^{(t)}W\_{l+1}^{(t)}D\_{i,l}^{(t)}W\_{l}^{\prime}h\_{i,l-1}^{(t)}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =B​(∑l=1L(Di,L(t)+Di,L′′)​WL(t)​⋯​Wl+1(t)​(Di,l(t)+Di,l′′)​Wl′​hi,l−1(t+1)−∑l=1LDi,L(t)​WL(t)​⋯​Wl+1(t)​Di,l(t)​Wl′​hi,l−1(t))absent𝐵superscriptsubscript𝑙1𝐿superscriptsubscript𝐷  𝑖𝐿𝑡superscriptsubscript𝐷  𝑖𝐿′′superscriptsubscript𝑊𝐿𝑡⋯superscriptsubscript𝑊𝑙1𝑡superscriptsubscript𝐷  𝑖𝑙𝑡superscriptsubscript𝐷  𝑖𝑙′′superscriptsubscript𝑊𝑙′superscriptsubscriptℎ  𝑖𝑙1𝑡1superscriptsubscript𝑙1𝐿superscriptsubscript𝐷  𝑖𝐿𝑡superscriptsubscript𝑊𝐿𝑡⋯superscriptsubscript𝑊𝑙1𝑡superscriptsubscript𝐷  𝑖𝑙𝑡superscriptsubscript𝑊𝑙′superscriptsubscriptℎ  𝑖𝑙1𝑡\displaystyle=B\left(\sum\_{l=1}^{L}(D\_{i,L}^{(t)}+D\_{i,L}^{\prime\prime})W\_{L}^{(t)}\cdots W\_{l+1}^{(t)}(D\_{i,l}^{(t)}+D\_{i,l}^{\prime\prime})W\_{l}^{\prime}h\_{i,l-1}^{(t+1)}-\sum\_{l=1}^{L}D\_{i,L}^{(t)}W\_{L}^{(t)}\cdots W\_{l+1}^{(t)}D\_{i,l}^{(t)}W\_{l}^{\prime}h\_{i,l-1}^{(t)}\right) |  |

where the last equality is obtained by replacing hi,L(t+1)−hi,L(t)superscriptsubscriptℎ

𝑖𝐿𝑡1superscriptsubscriptℎ

𝑖𝐿𝑡h\_{i,L}^{(t+1)}-h\_{i,L}^{(t)} by the term provided in Lemma [5](#Thmlemma5 "Lemma 5. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"), where Di,l′′∈ℝm×msuperscriptsubscript𝐷

𝑖𝑙′′superscriptℝ𝑚𝑚D\_{i,l}^{\prime\prime}\in\mathbb{R}^{m\times m} are diagonal matrices with entries in [−1,1]11[-1,1].

Now, we derive a bound for |ϵi(t)|\left|\epsilon\_{i}(t)\right\rvert.
We start by subtracting and adding the same term, yielding

|  |  |  |  |
| --- | --- | --- | --- |
|  | |ϵi(t)|\displaystyle\left|\epsilon\_{i}(t)\right\rvert | =|B(∑l=1L(Di,L(t)+Di,L′′)WL(t)⋯Wl+1(t)(Di,l(t)+Di,l′′)Wl′hi,l−1(t+1)−Di,L(t)WL(t)⋯Wl+1(t)Di,l(t)Wl′hi,l−1(t+1)\displaystyle=|B(\sum\_{l=1}^{L}(D\_{i,L}^{(t)}+D\_{i,L}^{\prime\prime})W\_{L}^{(t)}\cdots W\_{l+1}^{(t)}(D\_{i,l}^{(t)}+D\_{i,l}^{\prime\prime})W\_{l}^{\prime}h\_{i,l-1}^{(t+1)}-D\_{i,L}^{(t)}W\_{L}^{(t)}\cdots W\_{l+1}^{(t)}D\_{i,l}^{(t)}W\_{l}^{\prime}h\_{i,l-1}^{(t+1)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑l=1LDi,L(t)WL(t)⋯Wl+1(t)Di,l(t)Wl′hi,l−1(t+1)−Di,L(t)WL(t)⋯Wl+1(t)Di,l(t)Wl′hi,l−1(t))|\displaystyle+\sum\_{l=1}^{L}D\_{i,L}^{(t)}W\_{L}^{(t)}\cdots W\_{l+1}^{(t)}D\_{i,l}^{(t)}W\_{l}^{\prime}h\_{i,l-1}^{(t+1)}-D\_{i,L}^{(t)}W\_{L}^{(t)}\cdots W\_{l+1}^{(t)}D\_{i,l}^{(t)}W\_{l}^{\prime}h\_{i,l-1}^{(t)})| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤∑l=1L|B((Di,L(t)+Di,L′′)WL(t)…Wl+1(t)(Di,l(t)+Di,l′′)Wl′hi,l−1(t+1)−Di,L(t)WL(t)⋯Wl+1(t)Di,l(t)Wl′hi,l−1(t+1))|\displaystyle\leq\sum\_{l=1}^{L}\left|B\left((D\_{i,L}^{(t)}+D\_{i,L}^{\prime\prime})W\_{L}^{(t)}...W\_{l+1}^{(t)}(D\_{i,l}^{(t)}+D\_{i,l}^{\prime\prime})W\_{l}^{\prime}h\_{i,l-1}^{(t+1)}-D\_{i,L}^{(t)}W\_{L}^{(t)}\cdots W\_{l+1}^{(t)}D\_{i,l}^{(t)}W\_{l}^{\prime}h\_{i,l-1}^{(t+1)}\right)\right\rvert |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑l=1L|B(Di,L(t)WL(t)⋯Wl+1(t)Di,l(t)Wl′hi,l−1(t+1)−Di,L(t)WL(t)⋯Wl+1(t)Di,l(t)Wl′hi,l−1(t))|.\displaystyle+\sum\_{l=1}^{L}\left|B\left(D\_{i,L}^{(t)}W\_{L}^{(t)}\cdots W\_{l+1}^{(t)}D\_{i,l}^{(t)}W\_{l}^{\prime}h\_{i,l-1}^{(t+1)}-D\_{i,L}^{(t)}W\_{L}^{(t)}\cdots W\_{l+1}^{(t)}D\_{i,l}^{(t)}W\_{l}^{\prime}h\_{i,l-1}^{(t)}\right)\right\rvert. |  |

To construct the bound for |ϵi(t)|\left|\epsilon\_{i}(t)\right\rvert, we separately bound each of the above two terms. For the first term

|  |  |  |
| --- | --- | --- |
|  | |B((Di,L(t)+Di,L′′)WL(t)…Wl+1(t)(Di,l(t)+Di,l′′)Wl′hi,l−1(t+1)−Di,L(t)WL(t)…Wl+1(t)Di,l(t)Wl′hi,l−1(t+1))|\displaystyle\left|B\left((D\_{i,L}^{(t)}+D\_{i,L}^{\prime\prime})W\_{L}^{(t)}...W\_{l+1}^{(t)}(D\_{i,l}^{(t)}+D\_{i,l}^{\prime\prime})W\_{l}^{\prime}h\_{i,l-1}^{(t+1)}-D\_{i,L}^{(t)}W\_{L}^{(t)}...W\_{l+1}^{(t)}D\_{i,l}^{(t)}W\_{l}^{\prime}h\_{i,l-1}^{(t+1)}\right)\right\rvert |  |
|  |  |  |
| --- | --- | --- |
|  | ≤∥B​((Di,L(t)+Di,L′′)​WL(t)​…​Wl+1(t)​(Di,l(t)+Di,l′′)−Di,L(t)​WL(t)​…​Wl+1(t)​Di,l(t))∥2​∥Wl′​hi,l−1(t+1)∥2absentsubscriptdelimited-∥∥𝐵superscriptsubscript𝐷  𝑖𝐿𝑡superscriptsubscript𝐷  𝑖𝐿′′superscriptsubscript𝑊𝐿𝑡…superscriptsubscript𝑊𝑙1𝑡superscriptsubscript𝐷  𝑖𝑙𝑡superscriptsubscript𝐷  𝑖𝑙′′superscriptsubscript𝐷  𝑖𝐿𝑡superscriptsubscript𝑊𝐿𝑡…superscriptsubscript𝑊𝑙1𝑡superscriptsubscript𝐷  𝑖𝑙𝑡2subscriptdelimited-∥∥superscriptsubscript𝑊𝑙′superscriptsubscriptℎ  𝑖𝑙1𝑡12\displaystyle\leq\left\lVert B\left((D\_{i,L}^{(t)}+D\_{i,L}^{\prime\prime})W\_{L}^{(t)}...W\_{l+1}^{(t)}(D\_{i,l}^{(t)}+D\_{i,l}^{\prime\prime})-D\_{i,L}^{(t)}W\_{L}^{(t)}...W\_{l+1}^{(t)}D\_{i,l}^{(t)}\right)\right\rVert\_{2}\left\lVert W\_{l}^{\prime}h\_{i,l-1}^{(t+1)}\right\rVert\_{2} |  |
|  |  |  |
| --- | --- | --- |
|  | ≤1∥B​((Di,L(t)+Di,L′′)​WL(t)​…​Wl+1(t)​(Di,l(t)+Di,l′′)−Di,L(0)​WL(0)​…​Wl+1(0)​Di,l(0))∥2​O​(∥Wl′∥2)superscript1absentsubscriptdelimited-∥∥𝐵superscriptsubscript𝐷  𝑖𝐿𝑡superscriptsubscript𝐷  𝑖𝐿′′superscriptsubscript𝑊𝐿𝑡…superscriptsubscript𝑊𝑙1𝑡superscriptsubscript𝐷  𝑖𝑙𝑡superscriptsubscript𝐷  𝑖𝑙′′superscriptsubscript𝐷  𝑖𝐿0superscriptsubscript𝑊𝐿0…superscriptsubscript𝑊𝑙10superscriptsubscript𝐷  𝑖𝑙02𝑂subscriptdelimited-∥∥subscriptsuperscript𝑊′𝑙2\displaystyle\leq^{{}^{1}}\left\lVert B\left((D\_{i,L}^{(t)}+D\_{i,L}^{\prime\prime})W\_{L}^{(t)}...W\_{l+1}^{(t)}(D\_{i,l}^{(t)}+D\_{i,l}^{\prime\prime})-D\_{i,L}^{(0)}W\_{L}^{(0)}...W\_{l+1}^{(0)}D\_{i,l}^{(0)}\right)\right\rVert\_{2}O(\left\lVert W^{\prime}\_{l}\right\rVert\_{2}) |  |
|  |  |  |
| --- | --- | --- |
|  | +∥B(Di,L(0)WL(0)…Wl+1(0)Di,l(0)−Di,L(t)WL(t)…Wl+1(t)Di,l(t))∥2O(∥Wl′∥2))\displaystyle+\left\lVert B\left(D\_{i,L}^{(0)}W\_{L}^{(0)}...W\_{l+1}^{(0)}D\_{i,l}^{(0)}-D\_{i,L}^{(t)}W\_{L}^{(t)}...W\_{l+1}^{(t)}D\_{i,l}^{(t)}\right)\right\rVert\_{2}O(\left\lVert W^{\prime}\_{l}\right\rVert\_{2})) |  |
|  |  |  |
| --- | --- | --- |
|  | =2∥B(Di,L(0)−Di,L(0)+Di,L(t)+Di,L′′)WL(t)…Wl+1(t)(Di,l(0)−Di,l(0)+Di,l(t)+Di,l′′)−Di,L(0)WL(0)…Wl+1(0)Di,l(0))∥2O(∥Wl′∥2)\displaystyle=^{{}^{2}}\left\lVert B\left(D\_{i,L}^{(0)}-D\_{i,L}^{(0)}+D\_{i,L}^{(t)}+D\_{i,L}^{\prime\prime})W\_{L}^{(t)}...W\_{l+1}^{(t)}(D\_{i,l}^{(0)}-D\_{i,l}^{(0)}+D\_{i,l}^{(t)}+D\_{i,l}^{\prime\prime})-D\_{i,L}^{(0)}W\_{L}^{(0)}...W\_{l+1}^{(0)}D\_{i,l}^{(0)}\right)\right\rVert\_{2}O(\left\lVert W^{\prime}\_{l}\right\rVert\_{2}) |  |
|  |  |  |
| --- | --- | --- |
|  | +∥B​(Di,L(0)​WL(0)​…​Wl+1(0)​Di,l(0)−(Di,L(0)−Di,L(0)+Di,L(t))​WL(t)​…​Wl+1(t)​(Di,l(0)−Di,l(0)+Di,l(t)))∥2​O​(∥Wl′∥2)subscriptdelimited-∥∥𝐵superscriptsubscript𝐷  𝑖𝐿0superscriptsubscript𝑊𝐿0…superscriptsubscript𝑊𝑙10superscriptsubscript𝐷  𝑖𝑙0superscriptsubscript𝐷  𝑖𝐿0superscriptsubscript𝐷  𝑖𝐿0superscriptsubscript𝐷  𝑖𝐿𝑡superscriptsubscript𝑊𝐿𝑡…superscriptsubscript𝑊𝑙1𝑡superscriptsubscript𝐷  𝑖𝑙0superscriptsubscript𝐷  𝑖𝑙0superscriptsubscript𝐷  𝑖𝑙𝑡2𝑂subscriptdelimited-∥∥subscriptsuperscript𝑊′𝑙2\displaystyle+\left\lVert B\left(D\_{i,L}^{(0)}W\_{L}^{(0)}...W\_{l+1}^{(0)}D\_{i,l}^{(0)}-(D\_{i,L}^{(0)}-D\_{i,L}^{(0)}+D\_{i,L}^{(t)})W\_{L}^{(t)}...W\_{l+1}^{(t)}(D\_{i,l}^{(0)}-D\_{i,l}^{(0)}+D\_{i,l}^{(t)})\right)\right\rVert\_{2}O(\left\lVert W^{\prime}\_{l}\right\rVert\_{2}) |  |
|  |  |  |
| --- | --- | --- |
|  | ≤3O​(τ​ω1/3​L2​m​log⁡m)​O​(∥Wl′∥2)superscript3absent𝑂𝜏superscript𝜔13superscript𝐿2𝑚𝑚𝑂subscriptdelimited-∥∥subscriptsuperscript𝑊′𝑙2\displaystyle\leq^{{}^{3}}O(\tau\omega^{1/3}L^{2}\sqrt{m\log m})O(\left\lVert W^{\prime}\_{l}\right\rVert\_{2}) |  |

where we apply the following derivations

1. 1.

   We subtract and add the same term, use triangle inequality and the result provided in Lemma [10](#Thmlemma10 "Lemma 10. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"), ∥hi,l−1(t+1)∥=O​(1)delimited-∥∥superscriptsubscriptℎ
   𝑖𝑙1𝑡1𝑂1\left\lVert h\_{i,l-1}^{(t+1)}\right\rVert=O(1).
2. 2.

   Subtract and add Di,l(0)superscriptsubscript𝐷
   𝑖𝑙0D\_{i,l}^{(0)} from each coefficient that multiply Wl(t)superscriptsubscript𝑊𝑙𝑡W\_{l}^{(t)}.
3. 3.

   Due to Lemma [4](#Thmlemma4 "Lemma 4. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"), it holds that ‖W(t)−W(0)‖≤ωnormsuperscript𝑊𝑡superscript𝑊0𝜔||W^{(t)}-W^{(0)}||\leq\omega. This enables us to use Lemma [6](#Thmlemma6 "Lemma 6. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"), implying that ‖Di,l(t)−Di,l(0)‖0≤s=O​(m​ω2/3​L)subscriptnormsuperscriptsubscript𝐷
   𝑖𝑙𝑡superscriptsubscript𝐷
   𝑖𝑙00𝑠𝑂𝑚superscript𝜔23𝐿\|D\_{i,l}^{(t)}-D\_{i,l}^{(0)}\|\_{0}\leq s=O(m\omega^{2/3}L). Moreover, in conjunction with Lemma [5](#Thmlemma5 "Lemma 5. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"), this yields ∥Di,l(t)+Di,l′′−Di,l(0)∥0≤ssubscriptdelimited-∥∥superscriptsubscript𝐷
   𝑖𝑙𝑡superscriptsubscript𝐷
   𝑖𝑙′′superscriptsubscript𝐷
   𝑖𝑙00𝑠\left\lVert D\_{i,l}^{(t)}+D\_{i,l}^{\prime\prime}-D\_{i,l}^{(0)}\right\rVert\_{0}\leq s. Having that, we can apply Lemma [7](#Thmlemma7 "Lemma 7. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"), to obtain a bound for the first term.

For the second term we have that:

|  |  |  |
| --- | --- | --- |
|  | |B(Di,L(t)WL(t)…Wl+1(t)Di,l(t)Wl′hi,l−1(t+1)−Di,L(t)WL(t)…Wl+1(t)Di,l(t)Wl′hi,l−1(t))|\displaystyle\left|B(D\_{i,L}^{(t)}W\_{L}^{(t)}...W\_{l+1}^{(t)}D\_{i,l}^{(t)}W\_{l}^{\prime}h\_{i,l-1}^{(t+1)}-D\_{i,L}^{(t)}W\_{L}^{(t)}...W\_{l+1}^{(t)}D\_{i,l}^{(t)}W\_{l}^{\prime}h\_{i,l-1}^{(t)})\right\rvert |  |
|  |  |  |
| --- | --- | --- |
|  | =|B(Di,L(t)WL(t)…Wl+1(t)Di,l(t)Wl′(hi,l−1(t+1)−hi,l−1(t)))|\displaystyle=\left|B(D\_{i,L}^{(t)}W\_{L}^{(t)}...W\_{l+1}^{(t)}D\_{i,l}^{(t)}W\_{l}^{\prime}(h\_{i,l-1}^{(t+1)}-h\_{i,l-1}^{(t)}))\right\rvert |  |
|  |  |  |
| --- | --- | --- |
|  | ≤(∥B​(Di,L(t)​WL(t)​…​Wl+1(t)​Di,l(t)−Di,L(0)​WL(0)​…​Wl+1(0)​Di,l(0))∥+∥B​Di,L(0)​WL(0)​…​Wl+1(0)​Di,l(0)∥)​∥Wl′∥​∥hi,l−1(t+1)−hi,l−1(t)∥absentdelimited-∥∥𝐵superscriptsubscript𝐷  𝑖𝐿𝑡superscriptsubscript𝑊𝐿𝑡…superscriptsubscript𝑊𝑙1𝑡superscriptsubscript𝐷  𝑖𝑙𝑡superscriptsubscript𝐷  𝑖𝐿0superscriptsubscript𝑊𝐿0…superscriptsubscript𝑊𝑙10superscriptsubscript𝐷  𝑖𝑙0delimited-∥∥𝐵superscriptsubscript𝐷  𝑖𝐿0superscriptsubscript𝑊𝐿0…superscriptsubscript𝑊𝑙10superscriptsubscript𝐷  𝑖𝑙0delimited-∥∥subscriptsuperscript𝑊′𝑙delimited-∥∥superscriptsubscriptℎ  𝑖𝑙1𝑡1superscriptsubscriptℎ  𝑖𝑙1𝑡\displaystyle\leq\left(\left\lVert B(D\_{i,L}^{(t)}W\_{L}^{(t)}...W\_{l+1}^{(t)}D\_{i,l}^{(t)}-D\_{i,L}^{(0)}W\_{L}^{(0)}...W\_{l+1}^{(0)}D\_{i,l}^{(0)})\right\rVert+\left\lVert BD\_{i,L}^{(0)}W\_{L}^{(0)}...W\_{l+1}^{(0)}D\_{i,l}^{(0)}\right\rVert\right)\left\lVert W^{\prime}\_{l}\right\rVert\left\lVert h\_{i,l-1}^{(t+1)}-h\_{i,l-1}^{(t)}\right\rVert |  |
|  |  |  |
| --- | --- | --- |
|  | ≤1(O​(τ​ω1/3​L2​m​log⁡m)+∥B​Di,L(0)​WL(0)​…​Wl+1(0)​Di,l(0)∥)​∥Wl′∥​∥hi,l−1(t+1)−hi,l−1(t)∥superscript1absent𝑂𝜏superscript𝜔13superscript𝐿2𝑚𝑚delimited-∥∥𝐵superscriptsubscript𝐷  𝑖𝐿0superscriptsubscript𝑊𝐿0…superscriptsubscript𝑊𝑙10superscriptsubscript𝐷  𝑖𝑙0delimited-∥∥subscriptsuperscript𝑊′𝑙delimited-∥∥superscriptsubscriptℎ  𝑖𝑙1𝑡1superscriptsubscriptℎ  𝑖𝑙1𝑡\displaystyle\leq^{{}^{1}}\left(O(\tau\omega^{1/3}L^{2}\sqrt{m\log m})+\left\lVert BD\_{i,L}^{(0)}W\_{L}^{(0)}...W\_{l+1}^{(0)}D\_{i,l}^{(0)}\right\rVert\right)\left\lVert W^{\prime}\_{l}\right\rVert\left\lVert h\_{i,l-1}^{(t+1)}-h\_{i,l-1}^{(t)}\right\rVert |  |
|  |  |  |
| --- | --- | --- |
|  | ≤2τ​O​(m+ω1/3​L2​m​log⁡m)​∥Wl′∥​∥hi,l−1(t+1)−hi,l−1(t)∥≤3τ​O​(m+ω1/3​L2​m​log⁡m)​L1.5​∥W′∥2superscript2absent𝜏𝑂𝑚superscript𝜔13superscript𝐿2𝑚𝑚delimited-∥∥subscriptsuperscript𝑊′𝑙delimited-∥∥superscriptsubscriptℎ  𝑖𝑙1𝑡1superscriptsubscriptℎ  𝑖𝑙1𝑡superscript3𝜏𝑂𝑚superscript𝜔13superscript𝐿2𝑚𝑚superscript𝐿1.5superscriptdelimited-∥∥superscript𝑊′2\displaystyle\leq^{{}^{2}}\tau O(\sqrt{m}+\omega^{1/3}L^{2}\sqrt{m\log m})\left\lVert W^{\prime}\_{l}\right\rVert\left\lVert h\_{i,l-1}^{(t+1)}-h\_{i,l-1}^{(t)}\right\rVert\leq^{{}^{3}}\tau O(\sqrt{m}+\omega^{1/3}L^{2}\sqrt{m\log m})L^{1.5}\left\lVert W^{\prime}\right\rVert^{2} |  |
|  |  |  |
| --- | --- | --- |
|  | ≤4O​(τ​m)​L1.5​∥W′∥2superscript4absent𝑂𝜏𝑚superscript𝐿1.5superscriptdelimited-∥∥superscript𝑊′2\displaystyle\leq^{{}^{4}}O(\tau\sqrt{m})L^{1.5}\left\lVert W^{\prime}\right\rVert^{2} |  |

where we apply the following derivations

1. 1.

   As in the previous derivation, using Lemma [7](#Thmlemma7 "Lemma 7. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density").
2. 2.

   Applying Lemma [8](#Thmlemma8 "Lemma 8. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density").
3. 3.

   Using Lemma [5](#Thmlemma5 "Lemma 5. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density").
4. 4.

   Plug in ω=n3​log⁡mδ​τ​m𝜔superscript𝑛3𝑚𝛿𝜏𝑚\omega=\frac{n^{3}\log m}{\delta\tau\sqrt{m}}.

Since W′=−η​∇Φ​(W(t))superscript𝑊′𝜂∇Φsuperscript𝑊𝑡W^{\prime}=-\eta\nabla\Phi(W^{(t)}), we can get a bound for ∥W′∥2subscriptdelimited-∥∥superscript𝑊′2\left\lVert W^{\prime}\right\rVert\_{2} using Lemma [9](#Thmlemma9 "Lemma 9. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"), yielding ∥W′∥2≤η​O​(τ​n​m​Φ​(W(t)))subscriptdelimited-∥∥superscript𝑊′2𝜂𝑂𝜏𝑛𝑚Φsuperscript𝑊𝑡\left\lVert W^{\prime}\right\rVert\_{2}\leq\eta O(\tau\sqrt{nm}\sqrt{\Phi(W^{(t)})}).

Taking into account the two bounds, and summing over the all layers and data points we obtain that

|  |  |  |
| --- | --- | --- |
|  | ∥ϵ​(t)∥≤n​L​O​(τ​w1/3​L2​m​log⁡m)​O​(η​τ​n​m​Φ​(W(t)))+n​L​O​(τ​m)​L1.5​O​(η2​τ2​n​m​Φ​(W(t)))delimited-∥∥italic-ϵ𝑡𝑛𝐿𝑂𝜏superscript𝑤13superscript𝐿2𝑚𝑚𝑂𝜂𝜏𝑛𝑚Φsuperscript𝑊𝑡𝑛𝐿𝑂𝜏𝑚superscript𝐿1.5𝑂superscript𝜂2superscript𝜏2𝑛𝑚Φsuperscript𝑊𝑡\left\lVert\epsilon(t)\right\rVert\leq nLO(\tau w^{1/3}L^{2}\sqrt{m\log m})O(\eta\tau\sqrt{nm}\sqrt{\Phi(W^{(t)})})+nLO(\tau\sqrt{m})L^{1.5}O(\eta^{2}\tau^{2}nm\Phi(W^{(t)})) |  |

Using our choice of η𝜂\eta and the value of ω𝜔\omega, we finally get

|  |  |  |
| --- | --- | --- |
|  | ∥ϵ​(t)∥≤O​(L​log4/3⁡mτ1/3​m1/6​n1.5)​Φ​(W(t))+O​(δ2τ​n6​m0.5​L1.5)​Φ​(W(t))delimited-∥∥italic-ϵ𝑡𝑂𝐿superscript43𝑚superscript𝜏13superscript𝑚16superscript𝑛1.5Φsuperscript𝑊𝑡𝑂superscript𝛿2𝜏superscript𝑛6superscript𝑚0.5superscript𝐿1.5Φsuperscript𝑊𝑡\left\lVert\epsilon(t)\right\rVert\leq O\left(\frac{L\log^{4/3}m}{\tau^{1/3}m^{1/6}n^{1.5}}\right)\sqrt{\Phi(W^{(t)})}+O\left(\frac{\delta^{2}}{\tau n^{6}m^{0.5}L^{1.5}}\right)\Phi(W^{(t)}) |  |

∎

###### Theorem 5.

111This theorem was proved in (Allen-Zhu et al., [2019](#bib.bib2)), for τ=1𝜏1\tau=1. However, it is straightforward to generalize it for τ∈(0,1]𝜏01\tau\in(0,1] at the price of modifying m𝑚m and η𝜂\eta by a factor of 1τ21superscript𝜏2\frac{1}{\tau^{2}}

For any ϵ∈(0,1]italic-ϵ01\epsilon\in(0,1] and δ∈(0,O​(1L)]𝛿0𝑂1𝐿\delta\in(0,O(\frac{1}{L})], let m≥Ω​(n24​L12​log5⁡mδ8​τ2)𝑚Ωsuperscript𝑛24superscript𝐿12superscript5𝑚superscript𝛿8superscript𝜏2m\geq\Omega\left(\frac{n^{24}L^{12}\log^{5}m}{\delta^{8}\tau^{2}}\right), η=Θ​(δn4​L2​m​τ2)𝜂Θ𝛿superscript𝑛4superscript𝐿2𝑚superscript𝜏2\eta=\Theta\left(\frac{\delta}{n^{4}L^{2}m\tau^{2}}\right) and W(0),A,B

superscript𝑊0𝐴𝐵W^{(0)},A,B are at random initialization ([49](#A4.E49 "In D.1 The network model ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")). Then, starting from Gaussian initialization, with probability at least 1−e−Ω​(l​o​g2​m)1superscript𝑒Ω𝑙𝑜superscript𝑔2𝑚1-e^{-\Omega(log^{2}m)}, gradient descent with learning rate η𝜂\eta achieves

|  |  |  |
| --- | --- | --- |
|  | Φ​(W)≤ϵ​in​T=Θ​(n6​L2δ2​log⁡1ϵ)Φ𝑊italic-ϵin𝑇Θsuperscript𝑛6superscript𝐿2superscript𝛿21italic-ϵ\Phi(W)\leq\epsilon~{}~{}\text{in}~{}~{}T=\Theta\left(\frac{n^{6}L^{2}}{\delta^{2}}\log\frac{1}{\epsilon}\right) |  |

###### Lemma 4.

Under the assumptions of Thm. [5](#Thmtheorem5 "Theorem 5. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"),
it holds that for every t=0,1,..,T−1t=0,1,..,T-1

|  |  |  |  |
| --- | --- | --- | --- |
|  | (a)𝑎\displaystyle(a)~{}~{}~{}~{}~{} | ∥W(t)−W(0)∥F≤ω:=O​(n3δ​τ​m​log⁡m)subscriptdelimited-∥∥superscript𝑊𝑡superscript𝑊0𝐹𝜔assign𝑂superscript𝑛3𝛿𝜏𝑚𝑚\displaystyle\left\lVert W^{(t)}-W^{(0)}\right\rVert\_{F}\leq\omega:=O\left(\frac{n^{3}}{\delta\tau\sqrt{m}}\log m\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | (b)𝑏\displaystyle(b)~{}~{}~{}~{}~{} | Φ​(W(t))≤(1−Ω​(τ2​η​δ​mn2))t​Φ​(W(0))Φsuperscript𝑊𝑡superscript1Ωsuperscript𝜏2𝜂𝛿𝑚superscript𝑛2𝑡Φsuperscript𝑊0\displaystyle\Phi(W^{(t)})\leq\left(1-\Omega\left(\frac{\tau^{2}\eta\delta m}{n^{2}}\right)\right)^{t}\Phi(W^{(0)}) |  |

###### Lemma 5.

(This Lemma follows Claim 11.2 from (Allen-Zhu et al., [2019](#bib.bib2)))
Let ω∈[Ω​(1τ3​m3/2​L3/2​log3/2⁡m),O​(1L4.5​log3⁡m)]𝜔Ω1superscript𝜏3superscript𝑚32superscript𝐿32superscript32𝑚𝑂1superscript𝐿4.5superscript3𝑚\omega\in[\Omega(\frac{1}{\tau^{3}m^{3/2}L^{3/2}\log^{3/2}m}),O(\frac{1}{L^{4.5}\log^{3}m})], then under the following assumptions ∥W(t)−W(0)∥2≤ωsubscriptdelimited-∥∥superscript𝑊𝑡superscript𝑊02𝜔\left\lVert W^{(t)}-W^{(0)}\right\rVert\_{2}\leq\omega and ∥W′∥2≤wsubscriptdelimited-∥∥superscript𝑊′2𝑤\left\lVert W^{\prime}\right\rVert\_{2}\leq w it holds that
there exist diagonal matrices Di,l′′∈ℝm×msubscriptsuperscript𝐷′′

𝑖𝑙superscriptℝ𝑚𝑚D^{\prime\prime}\_{i,l}\in\mathbb{R}^{m\times m} with entries in [-1,1] such that

|  |  |  |
| --- | --- | --- |
|  | ∀i∈[n],∀l∈[L]:hi,l(t+1)−hi,l(t)=∑a=1l(Di,l(t)+Di,l′′)​Wl(t)​…​Wa+1(t)​(Di,a(t)+Di,a′′)​Wa′​hi,a−1(t+1):formulae-sequencefor-all𝑖delimited-[]𝑛for-all𝑙delimited-[]𝐿subscriptsuperscriptℎ𝑡1  𝑖𝑙subscriptsuperscriptℎ𝑡  𝑖𝑙superscriptsubscript𝑎1𝑙superscriptsubscript𝐷  𝑖𝑙𝑡subscriptsuperscript𝐷′′  𝑖𝑙superscriptsubscript𝑊𝑙𝑡…superscriptsubscript𝑊𝑎1𝑡superscriptsubscript𝐷  𝑖𝑎𝑡superscriptsubscript𝐷  𝑖𝑎′′subscriptsuperscript𝑊′𝑎subscriptsuperscriptℎ𝑡1  𝑖𝑎1\displaystyle\forall i\in[n],\forall l\in[L]:h^{(t+1)}\_{i,l}-h^{(t)}\_{i,l}=\sum\_{a=1}^{l}(D\_{i,l}^{(t)}+D^{\prime\prime}\_{i,l})W\_{l}^{(t)}...W\_{a+1}^{(t)}(D\_{i,a}^{(t)}+D\_{i,a}^{\prime\prime})W^{\prime}\_{a}h^{(t+1)}\_{i,a-1} |  |

Furthermore we have ∥hi,l(t+1)−hi,l(t)∥≤O​(L1.5)​∥W′∥2delimited-∥∥subscriptsuperscriptℎ𝑡1

𝑖𝑙subscriptsuperscriptℎ𝑡

𝑖𝑙𝑂superscript𝐿1.5subscriptdelimited-∥∥superscript𝑊′2\left\lVert h^{(t+1)}\_{i,l}-h^{(t)}\_{i,l}\right\rVert\leq O(L^{1.5})\left\lVert W^{\prime}\right\rVert\_{2} and ∥B​hi,l(t+1)−B​hi,l(t)∥≤O​(L​τ​m)​∥W′∥2delimited-∥∥𝐵subscriptsuperscriptℎ𝑡1

𝑖𝑙𝐵subscriptsuperscriptℎ𝑡

𝑖𝑙𝑂𝐿𝜏𝑚subscriptdelimited-∥∥superscript𝑊′2\left\lVert Bh^{(t+1)}\_{i,l}-Bh^{(t)}\_{i,l}\right\rVert\leq O(L\tau\sqrt{m})\left\lVert W^{\prime}\right\rVert\_{2} and ∥Di,l′′∥0≤O​(m​ω2/3​L)subscriptdelimited-∥∥superscriptsubscript𝐷

𝑖𝑙′′0𝑂𝑚superscript𝜔23𝐿\left\lVert D\_{i,l}^{\prime\prime}\right\rVert\_{0}\leq O(m\omega^{2/3}L)

###### Lemma 6.

(This Lemma follows Lemma 8.2 from (Allen-Zhu et al., [2019](#bib.bib2))) Suppose ω≤1C​L9/2​l​o​g3​m𝜔1𝐶superscript𝐿92𝑙𝑜superscript𝑔3𝑚\omega\leq\frac{1}{CL^{9/2}log^{3}m} for some sufficiently large constant C>1𝐶1C>1. With probability at least 1−e−Ω​(m​ω2/3​L)1superscript𝑒Ω𝑚superscript𝜔23𝐿1-e^{-\Omega(m\omega^{2/3}L)} for every (W(t)−W(0))superscript𝑊𝑡superscript𝑊0(W^{(t)}-W^{(0)}) satisfying ∥W(t)−W(0)∥2≤ωsubscriptdelimited-∥∥superscript𝑊𝑡superscript𝑊02𝜔\left\lVert W^{(t)}-W^{(0)}\right\rVert\_{2}\leq\omega,

|  |  |  |
| --- | --- | --- |
|  | ∥Di,l(t)−Di,l(0)∥0≤O​(m​ω2/3​L)subscriptdelimited-∥∥superscriptsubscript𝐷  𝑖𝑙𝑡superscriptsubscript𝐷  𝑖𝑙00𝑂𝑚superscript𝜔23𝐿\left\lVert D\_{i,l}^{(t)}-D\_{i,l}^{(0)}\right\rVert\_{0}\leq O(m\omega^{2/3}L) |  |

###### Lemma 7.

(This Lemma follows Lemma 8.7 from (Allen-Zhu et al., [2019](#bib.bib2))) For s=O​(m​w2/3​L)𝑠𝑂𝑚superscript𝑤23𝐿s=O(mw^{2/3}L), with probability at least 1−e−Ω​(s​log⁡m)1superscript𝑒Ω𝑠𝑚1-e^{-\Omega(s\log m)} over the randomness of W(0),A,B

superscript𝑊0𝐴𝐵W^{(0)},A,B

* •

  for all i∈[n],a∈[L+1]formulae-sequence𝑖delimited-[]𝑛𝑎delimited-[]𝐿1i\in[n],a\in[L+1]
* •

  for every diagonal matrices Di,0′′′,⋯,Di,L′′′∈[−3,3]m×m
  superscriptsubscript𝐷
  𝑖0′′′⋯superscriptsubscript𝐷
  𝑖𝐿′′′superscript33𝑚𝑚D\_{i,0}^{\prime\prime\prime},\cdots,D\_{i,L}^{\prime\prime\prime}\in[-3,3]^{m\times m} with at most s non-zero entries
* •

  for every perturbation with respect to the initialization W1′′​⋯​WL′′∈ℝm×msubscriptsuperscript𝑊′′1⋯subscriptsuperscript𝑊′′𝐿superscriptℝ𝑚𝑚W^{\prime\prime}\_{1}\cdots W^{\prime\prime}\_{L}\in\mathbb{R}^{m\times m} with ∥W′′∥2≤ω=O​(1/L1.5)subscriptdelimited-∥∥superscript𝑊′′2𝜔𝑂1superscript𝐿1.5\left\lVert W^{\prime\prime}\right\rVert\_{2}\leq\omega=O(1/L^{1.5})

it holds ∥B​(Di,L(0)+Di,L′′′)​(WL(0)+WL′′)​⋯​(Wa+1(0)+Wa+1′′)​(Di,a(0)+Di,a′′′)−B​Di,L(0)​WL(0)​⋯​Wa+1(0)​Di,a(0)∥2≤O​(τ​ω1/3​L2​m​log⁡m)subscriptdelimited-∥∥𝐵superscriptsubscript𝐷

𝑖𝐿0subscriptsuperscript𝐷′′′

𝑖𝐿superscriptsubscript𝑊𝐿0superscriptsubscript𝑊𝐿′′⋯superscriptsubscript𝑊𝑎10superscriptsubscript𝑊𝑎1′′superscriptsubscript𝐷

𝑖𝑎0subscriptsuperscript𝐷′′′

𝑖𝑎𝐵superscriptsubscript𝐷

𝑖𝐿0superscriptsubscript𝑊𝐿0⋯superscriptsubscript𝑊𝑎10superscriptsubscript𝐷

𝑖𝑎02𝑂𝜏superscript𝜔13superscript𝐿2𝑚𝑚\left\lVert B(D\_{i,L}^{(0)}+D^{\prime\prime\prime}\_{i,L})(W\_{L}^{(0)}+W\_{L}^{\prime\prime})\cdots(W\_{a+1}^{(0)}+W\_{a+1}^{\prime\prime})(D\_{i,a}^{(0)}+D^{\prime\prime\prime}\_{i,a})-BD\_{i,L}^{(0)}W\_{L}^{(0)}\cdots W\_{a+1}^{(0)}D\_{i,a}^{(0)}\right\rVert\_{2}\leq O(\tau\omega^{1/3}L^{2}\sqrt{m\log m})

###### Lemma 8.

(This Lemma follows Lemma 7.4b from (Allen-Zhu et al., [2019](#bib.bib2)))
Suppose m≥Ω​(n​L​log⁡(n​L)).𝑚Ω𝑛𝐿𝑛𝐿m\geq\Omega(nL\log(nL)). If s=O​(m​ω2/3​L)𝑠𝑂𝑚superscript𝜔23𝐿s=O(m\omega^{2/3}L) then with probability at least 1−e−Ω​(s​log⁡m)1superscript𝑒Ω𝑠𝑚1-e^{-\Omega(s\log m)} for all i∈[n],a∈[L+1]formulae-sequence𝑖delimited-[]𝑛𝑎delimited-[]𝐿1i\in[n],a\in[L+1] it holds that ∥vT​B​Di,L(0)​WL(0)​⋯​Di,a(0)​Wa(0)∥≤O​(τ​m)​∥v∥delimited-∥∥superscript𝑣𝑇𝐵superscriptsubscript𝐷

𝑖𝐿0superscriptsubscript𝑊𝐿0⋯superscriptsubscript𝐷

𝑖𝑎0superscriptsubscript𝑊𝑎0𝑂𝜏𝑚delimited-∥∥𝑣\left\lVert v^{T}BD\_{i,L}^{(0)}W\_{L}^{(0)}\cdots D\_{i,a}^{(0)}W\_{a}^{(0)}\right\rVert\leq O(\tau\sqrt{m})\left\lVert v\right\rVert.

###### Lemma 9.

(This Lemma follows Theorem 3 from (Allen-Zhu et al., [2019](#bib.bib2)))
Let ω=O​(δ3/2n9/2​L6​log3⁡m)𝜔𝑂superscript𝛿32superscript𝑛92superscript𝐿6superscript3𝑚\omega=O(\frac{\delta^{3/2}}{n^{9/2}L^{6}\log^{3}m}). With probability at least 1−e−Ω​(m​ω2/3​L)1superscript𝑒Ω𝑚superscript𝜔23𝐿1-e^{-\Omega(m\omega^{2/3}L)} over the randomness of W0,A,B

superscript𝑊0𝐴𝐵W^{0},A,B, it satisfies for every l∈[L]𝑙delimited-[]𝐿l\in[L] and W𝑊W with ∥W−W(0)∥2≤ωsubscriptdelimited-∥∥𝑊superscript𝑊02𝜔\left\lVert W-W^{(0)}\right\rVert\_{2}\leq\omega that

|  |  |  |
| --- | --- | --- |
|  | ‖∇WlΦ​(W)‖F2≤O​(τ2​Φ​(W)⋅n⋅m)subscriptsuperscriptnormsubscript∇subscript𝑊𝑙Φ𝑊2𝐹𝑂⋅superscript𝜏2Φ𝑊𝑛𝑚\|\nabla\_{W\_{l}}\Phi(W)\|^{2}\_{F}\leq O(\tau^{2}\Phi(W)\cdot n\cdot m) |  |

###### Lemma 10.

(This Lemma is based on Lemma 7.1 and Lemma 8.2c from (Allen-Zhu et al., [2019](#bib.bib2)))
With high probability over the randomness of A,W

𝐴𝑊A,W we have

|  |  |  |
| --- | --- | --- |
|  | ∀i∈[n],l∈{0,1,..,L}:∥hi,l∥=O(1)\forall i\in[n],l\in\{0,1,..,L\}:\|h\_{i,l}\|=O(1) |  |

###### Lemma 11.

Let δ>0𝛿0\delta>0 and m≥Ω(Llog(nL/δ)m\geq\Omega(L\log(nL/\delta) then with probability at least 1−δ1𝛿1-\delta it holds that ‖u​(0)‖≤n​τ/δnorm𝑢0𝑛𝜏𝛿||u(0)||\leq\sqrt{n}\tau/\delta and as a consequence by using the triangle inequality Φ​(W​(0))=12​∥𝐲−𝐮​(0)∥2≤O​(n)Φ𝑊012superscriptdelimited-∥∥𝐲𝐮02𝑂𝑛\Phi(W(0))=\frac{1}{2}\left\lVert\mathbf{y}-\mathbf{u}(0)\right\rVert^{2}\leq O(n)

###### Proof.

Conditioned on W,A

𝑊𝐴W,A it holds that ui​(0)∽N​(0,τ2​∥hi,L∥2)∽subscript𝑢𝑖0𝑁0superscript𝜏2superscriptdelimited-∥∥subscriptℎ

𝑖𝐿2u\_{i}(0)\backsim N(0,\tau^{2}\left\lVert h\_{i,L}\right\rVert^{2}) and since by Lemma [10](#Thmlemma10 "Lemma 10. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density") we have that ∥hi,L∥=O​(1)delimited-∥∥subscriptℎ

𝑖𝐿𝑂1\left\lVert h\_{i,L}\right\rVert=O(1), this yields E​(∥𝐮​(0)∥2)=O​(n​τ2)𝐸superscriptdelimited-∥∥𝐮02𝑂𝑛superscript𝜏2E(\left\lVert\mathbf{u}(0)\right\rVert^{2})=O\left(n\tau^{2}\right). Then by Markov’s inequality, ∥𝐮​(0)∥2≤n​τ2/δ2superscriptdelimited-∥∥𝐮02𝑛superscript𝜏2superscript𝛿2\left\lVert\mathbf{u}(0)\right\rVert^{2}\leq n\tau^{2}/\delta^{2} with probability 1−δ1𝛿1-\delta.
∎

###### Lemma 12.

(Based on Theorem 3.1 (Arora et al., [2019a](#bib.bib3)))222The formulation given in (Arora et al., [2019a](#bib.bib3)) considers training w.r.t all layers. The proof can be extended trivially to the case where the first and last layers are held fixed. 
Fix ϵ>0italic-ϵ0\epsilon>0 and δ∈(0,1)𝛿01\delta\in(0,1) and assume m≥Ω​(L6ϵ4​l​o​g​(Lδ))𝑚Ωsuperscript𝐿6superscriptitalic-ϵ4𝑙𝑜𝑔𝐿𝛿m\geq\Omega(\frac{L^{6}}{\epsilon^{4}}log(\frac{L}{\delta})). Then for any pair of inputs 𝐱i,𝐱j

subscript𝐱𝑖subscript𝐱𝑗\mathbf{x}\_{i},\mathbf{x}\_{j} such that ‖𝐱i‖≤1,‖𝐱j‖≤1formulae-sequencenormsubscript𝐱𝑖1normsubscript𝐱𝑗1\|\mathbf{x}\_{i}\|\leq 1,\|\mathbf{x}\_{j}\|\leq 1 with probability 1−δ1𝛿1-\delta we have

|  |  |  |
| --- | --- | --- |
|  | |1mHi​j(0)−1mHi​j∞|≤(L+1)ϵ\left|\frac{1}{m}H\_{ij}(0)-\frac{1}{m}H^{\infty}\_{ij}\right\rvert\leq(L+1)\epsilon |  |

###### Lemma 13.

(Based on Theorem 5c (Allen-Zhu et al., [2019](#bib.bib2)))
Let W(0),A,B

superscript𝑊0𝐴𝐵W^{(0)},A,B be at random initialization. For any pair of inputs 𝐱i,𝐱j

subscript𝐱𝑖subscript𝐱𝑗\mathbf{x}\_{i},\mathbf{x}\_{j} and parameter ω≤O​(1L9​l​o​g3/2​m)𝜔𝑂1superscript𝐿9𝑙𝑜superscript𝑔32𝑚\omega\leq O(\frac{1}{L^{9}log^{3/2}m}) with probability at least 1−e−Ω​(m​ω2/3​L)1superscript𝑒Ω𝑚superscript𝜔23𝐿1-e^{-\Omega(m\omega^{2/3}L)} over W(0),A,B

superscript𝑊0𝐴𝐵W^{(0)},A,B with ∥W(0)−W(t)∥2≤ωsubscriptdelimited-∥∥superscript𝑊0superscript𝑊𝑡2𝜔\left\lVert W^{(0)}-W^{(t)}\right\rVert\_{2}\leq\omega we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | |Hi​j(t)−Hi​j(0)|≤O(log⁡m⋅ω1/3L3)Hi,i​(0)​Hj,j​(0)\displaystyle\left|H\_{ij}(t)-H\_{ij}(0)\right\rvert\leq O(\sqrt{\log m}\cdot\omega^{1/3}L^{3})\sqrt{H\_{i,i}(0)H\_{j,j}(0)} |  | (58) |

###### Lemma 14.

Let δ^∈(0,1]^𝛿01\hat{\delta}\in(0,1] and W(0),A,B

superscript𝑊0𝐴𝐵W^{(0)},A,B be at random initialization. Then, for m≥Ω​(n24​L12​log5⁡mδ8​τ6)𝑚Ωsuperscript𝑛24superscript𝐿12superscript5𝑚superscript𝛿8superscript𝜏6m\geq\Omega\left(\frac{n^{24}L^{12}\log^{5}m}{\delta^{8}\tau^{6}}\right) and parameter ω=O​(n3δ​τ​m​log⁡m)𝜔𝑂superscript𝑛3𝛿𝜏𝑚𝑚\omega=O\left(\frac{n^{3}}{\delta\tau\sqrt{m}}\log m\right) with probability of at least 1−δ^1^𝛿1-\hat{\delta} over W(0),A,B

superscript𝑊0𝐴𝐵W^{(0)},A,B with ∥W(0)−W(t)∥2≤ωsubscriptdelimited-∥∥superscript𝑊0superscript𝑊𝑡2𝜔\left\lVert W^{(0)}-W^{(t)}\right\rVert\_{2}\leq\omega it holds that

1. 1.

   ∥H​(t)−H​(0)∥2≤O​(n3​l​o​g5/6​mδ​τ)​m5/6subscriptdelimited-∥∥𝐻𝑡𝐻02𝑂superscript𝑛3𝑙𝑜superscript𝑔56𝑚𝛿𝜏superscript𝑚56\left\lVert H(t)-H(0)\right\rVert\_{2}\leq O(\frac{n^{3}log^{5/6}m}{\delta\tau})m^{5/6}
2. 2.

   ∥H​(0)−H∞∥2≤O​(δ2​m​τ3n6)subscriptdelimited-∥∥𝐻0superscript𝐻2𝑂superscript𝛿2𝑚superscript𝜏3superscript𝑛6\left\lVert H(0)-H^{\infty}\right\rVert\_{2}\leq O(\frac{\delta^{2}m\tau^{3}}{n^{6}})
3. 3.

   ∥H∞−H​(t)∥2≤O​(n3​l​o​g5/6​mδ​τ)​m5/6+O​(δ2​m​τ3n6)≤O​(δ2​m​τ3n6)subscriptdelimited-∥∥superscript𝐻𝐻𝑡2𝑂superscript𝑛3𝑙𝑜superscript𝑔56𝑚𝛿𝜏superscript𝑚56𝑂superscript𝛿2𝑚superscript𝜏3superscript𝑛6𝑂superscript𝛿2𝑚superscript𝜏3superscript𝑛6\left\lVert H^{\infty}-H(t)\right\rVert\_{2}\leq O(\frac{n^{3}log^{5/6}m}{\delta\tau})m^{5/6}+O(\frac{\delta^{2}m\tau^{3}}{n^{6}})\leq O(\frac{\delta^{2}m\tau^{3}}{n^{6}})

###### Proof.

We prove the first claim. Then, the second claim is obtained by plugging m𝑚m into Lemma [12](#Thmlemma12 "Lemma 12. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density"). The third claim is a direct consequence of the two claims using triangle inequality.

By the definition of Hi​j​(0)subscript𝐻𝑖𝑗0H\_{ij}(0) we have that

|  |  |  |  |
| --- | --- | --- | --- |
|  | Hi​i​(0)subscript𝐻𝑖𝑖0\displaystyle\sqrt{H\_{ii}(0)} | =⟨∂ui​(0)∂W,∂ui​(0)∂W⟩absent  subscript𝑢𝑖0𝑊subscript𝑢𝑖0𝑊\displaystyle=\sqrt{\left\langle\frac{\partial u\_{i}(0)}{\partial W},\frac{\partial u\_{i}(0)}{\partial W}\right\rangle} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤∑l=1L∥∂ui​(0)∂Wl∥=∑l=1L∥hi,l−1​B​Di,L(0)​WL(0)​Di,L−1(0)​WL−1(0)​⋯​Di,L+1(0)​Wl+1(0)​Di,l(0)∥absentsuperscriptsubscript𝑙1𝐿delimited-∥∥subscript𝑢𝑖0subscript𝑊𝑙superscriptsubscript𝑙1𝐿delimited-∥∥subscriptℎ  𝑖𝑙1𝐵subscriptsuperscript𝐷0  𝑖𝐿subscriptsuperscript𝑊0𝐿subscriptsuperscript𝐷0  𝑖𝐿1subscriptsuperscript𝑊0𝐿1⋯subscriptsuperscript𝐷0  𝑖𝐿1subscriptsuperscript𝑊0𝑙1subscriptsuperscript𝐷0  𝑖𝑙\displaystyle\leq\sum\_{l=1}^{L}\left\lVert\frac{\partial u\_{i}(0)}{\partial W\_{l}}\right\rVert=\sum\_{l=1}^{L}\left\lVert h\_{i,l-1}BD^{(0)}\_{i,L}W^{(0)}\_{L}D^{(0)}\_{i,L-1}W^{(0)}\_{L-1}\cdots D^{(0)}\_{i,L+1}W^{(0)}\_{l+1}D^{(0)}\_{i,l}\right\rVert |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤∑l=1L∥hi,l−1∥​∥B​Di,L(0)​WL(0)​Di,L−1(0)​WL−1(0)​⋯​Di,L+1(0)​Wl+1(0)​Di,l(0)∥≤O​(L​m​τ)absentsuperscriptsubscript𝑙1𝐿delimited-∥∥subscriptℎ  𝑖𝑙1delimited-∥∥𝐵subscriptsuperscript𝐷0  𝑖𝐿subscriptsuperscript𝑊0𝐿subscriptsuperscript𝐷0  𝑖𝐿1subscriptsuperscript𝑊0𝐿1⋯subscriptsuperscript𝐷0  𝑖𝐿1subscriptsuperscript𝑊0𝑙1subscriptsuperscript𝐷0  𝑖𝑙𝑂𝐿𝑚𝜏\displaystyle\leq\sum\_{l=1}^{L}\left\lVert h\_{i,l-1}\right\rVert\left\lVert BD^{(0)}\_{i,L}W^{(0)}\_{L}D^{(0)}\_{i,L-1}W^{(0)}\_{L-1}\cdots D^{(0)}\_{i,L+1}W^{(0)}\_{l+1}D^{(0)}\_{i,l}\right\rVert\leq O(L\sqrt{m}\tau) |  |

where the last inequality is obtained by applying Lemma [8](#Thmlemma8 "Lemma 8. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density") and Lemma [10](#Thmlemma10 "Lemma 10. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density").
Applying the obtained bound for Hi​i​(0)subscript𝐻𝑖𝑖0H\_{ii}(0) and Hj​j​(0)subscript𝐻𝑗𝑗0H\_{jj}(0) yields a bound for |Hi​j(t)−Hi​j(0)|\left|H\_{ij}(t)-H\_{ij}(0)\right\rvert, using ([58](#A4.E58 "In Lemma 13. ‣ D.4 Supporting Lemmas ‣ Appendix D Spectral convergence analysis for deep networks - proof of Theorem 2 ‣ Frequency Bias in Neural Networks for Input of Non-Uniform Density")). Finally, ∥H​(t)−H​(0)∥≤O​(n3​l​o​g5/6​mδ​τ)​m5/6delimited-∥∥𝐻𝑡𝐻0𝑂superscript𝑛3𝑙𝑜superscript𝑔56𝑚𝛿𝜏superscript𝑚56\left\lVert H(t)-H(0)\right\rVert\leq O(\frac{n^{3}log^{5/6}m}{\delta\tau})m^{5/6}.
∎

## Appendix E Experiment setup

Below we provide our experimental setup for all the figures in the paper.

Figure 1.
Experiments are run with input data in 𝕊1superscript𝕊1\mathbb{S}^{1} drawn from a uniform (top plots) and non-uniform (bottom plots) distributions, where the latter densities are of ratio 1:40:1401:40. The target function is y​(x)=0.4​cos⁡(16​x)+cos⁡(x)𝑦𝑥0.416𝑥𝑥y(x)=0.4\cos(16x)+\cos(x). The number of training points is n=10000𝑛10000n=10000 and batch size is 100. The network includes L=10𝐿10L=10 fully connected layers, each with m=256𝑚256m=256 hidden units. The weights are initialized with normal distribution with standard deviation τ=0.1𝜏0.1\tau=0.1, and the learning rate is η=0.001𝜂0.001\eta=0.001.

Figure 2. Eigenfunctions are computed with n=2,933𝑛

2933n=2,933 data points in 𝕊1superscript𝕊1\mathbb{S}^{1}.

Figure 3. Local frequencies are computed with n=1,467𝑛

1467n=1,467 data points in 𝕊1superscript𝕊1\mathbb{S}^{1}.

Figure 4. Eigenvalues are computed with n=50,000𝑛

50000n=50,000 data points in 𝕊1superscript𝕊1\mathbb{S}^{1}.

Figure 5. Eigenvalues are computed with n=12,567𝑛

12567n=12,567 data points in 𝕊1superscript𝕊1\mathbb{S}^{1}.

Figure 6.
Eigenvectors are computed numerically using n=10,000𝑛

10000n=10,000 data points in 𝕊1superscript𝕊1\mathbb{S}^{1} drawn from a piecewise constant distribution with densities proportional to (11,1,3)1113(11,1,3).

Figure 7.
Convergence times are measured by training a two-layer network with bias. The weights of the second layer are set randomly to −11-1 or 111 (with probability 0.50.50.5) and remain fixed throughout training. The bias is initialized to zero. The network parameters are set to m=4000𝑚4000m=4000, η=0.004𝜂0.004\eta=0.004, n=734𝑛734n=734, and τ=0.2𝜏0.2\tau=0.2. Convergence for region Rjsubscript𝑅𝑗R\_{j} is declared when 12|Rj|​∑i∈Rjn(f​(xi;w)−ui)2<δn\frac{1}{2\left|R\_{j}\right\rvert}\sum\_{i\in R\_{j}}^{n}\left(f(x\_{i};w)-u\_{i}\right)^{2}<\frac{\delta}{n} with δ=0.05𝛿0.05\delta=0.05.

Figure 8.
Eigenvectors are computed with n=9,926𝑛

9926n=9,926 data points in 𝕊2superscript𝕊2\mathbb{S}^{2}.

Figure 9.
We used the same setup as in Figure 7 with the parameters: m=8000𝑚8000m=8000, t​a​u=0.2𝑡𝑎𝑢0.2tau=0.2, and η=0.004𝜂0.004\eta=0.004. Here n𝑛n varies between the three plots. We sampled 300 points from a uniform distribution on one hemisphere, and 300​p2/p1300subscript𝑝2subscript𝑝1300p\_{2}/p\_{1} points on the other hemisphere, where p2/p1∈{2,3,4}subscript𝑝2subscript𝑝1234p\_{2}/p\_{1}\in\{2,3,4\}.

Figure 10.
Eigenvectors are computed with n=1257𝑛1257n=1257 data points in 𝕊1superscript𝕊1\mathbb{S}^{1}.

Figure 11. Here we compare the number of iterations needed for a deep FC network to converge the number of iterations predicted by the eigenvalue of the corresponding NTK. We used m=256𝑚256m=256, η=0.05𝜂0.05\eta=0.05 and δ=0.05𝛿0.05\delta=0.05. The corresponding NTK was calculated in the 𝕊1superscript𝕊1\mathbb{S}^{1} with n=630𝑛630n=630 points and in 𝕊2superscript𝕊2\mathbb{S}^{2} with n=1,000𝑛

1000n=1,000 points, both drawn from a uniform distribution. Note that the plot for 𝕊2superscript𝕊2\mathbb{S}^{2} appears on the left and the one for 𝕊1superscript𝕊1\mathbb{S}^{1} on the right.

Figure 12. Here, we calculate the eigenvalues of NTK for FC networks with 3≤L≤503𝐿503\leq L\leq 50 layers for data distributed uniformly in 𝕊1superscript𝕊1\mathbb{S}^{1} (left) and 𝕊2superscript𝕊2\mathbb{S}^{2} (right). The NTK was calculated with n=16,383𝑛

16383n=16,383 and n=20,000𝑛

20000n=20,000 data points in 𝕊1superscript𝕊1\mathbb{S}^{1} and 𝕊2superscript𝕊2\mathbb{S}^{2}, respectively.
