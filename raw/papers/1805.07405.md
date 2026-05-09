---
arxiv: '1805.07405'
authors:
- Marek Śmieja marek.smieja@uj.edu.pl &Łukasz Struski lukasz.struski@uj.edu.pl &Jacek
  Tabor jacek.tabor@uj.edu.pl &Bartosz Zieliński bartosz.zielinski@uj.edu.pl &Przemysław
  Spurek przemyslaw.spurek@uj.edu.pl
parser: ar5iv
retrieved: '2026-05-09'
source: paper
title: Processing of missing data by neural networks
url: https://arxiv.org/abs/1805.07405
year: 2018
---

[1805.07405] Processing of missing data by neural networks














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



# Processing of missing data by neural networks

Marek Śmieja
  
marek.smieja@uj.edu.pl
  
&Łukasz Struski
  
lukasz.struski@uj.edu.pl
  
&Jacek Tabor
  
jacek.tabor@uj.edu.pl
  
&Bartosz Zieliński
  
bartosz.zielinski@uj.edu.pl
  
&Przemysław Spurek
  
przemyslaw.spurek@uj.edu.pl

###### Abstract

We propose a general, theoretically justified mechanism for processing missing data by neural networks. Our idea is to replace typical neuron’s response in the first hidden layer by its expected value. This approach can be applied for various types of networks at minimal cost in their modification. Moreover, in contrast to recent approaches, it does not require complete data for training. Experimental results performed on different types of architectures show that our method gives better results than typical imputation strategies and other methods dedicated for incomplete data.

Faculty of Mathematics and Computer Science

Jagiellonian University

Łojasiewicza 6, 30-348 Kraków, Poland

## 1 Introduction

Learning from incomplete data has been recognized as one of the fundamental challenges in machine learning [[1](#bib.bib1)]. Due to the great interest in deep learning in the last decade, it is especially important to establish unified tools for practitioners to process missing data with arbitrary neural networks.

In this paper, we introduce a general, theoretically justified methodology for feeding neural networks with missing data. Our idea is to model the uncertainty on missing attributes by probability density functions, which eliminates the need of direct completion (imputation) by single values. In consequence, every missing data point is identified with parametric density, e.g. GMM, which is trained together with remaining network parameters. To process this probabilistic representation by neural network, we generalize the neuron’s response at the first hidden layer by taking its expected value (Section [3](#S3 "3 Layer for processing missing data ‣ Processing of missing data by neural networks")). This strategy can be understand as calculating the average neuron’s activation over the imputations drawn from missing data density (see Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Processing of missing data by neural networks") for the illustration).

The main advantage of the proposed approach is the ability to train neural network on data sets containing only incomplete samples (without a single fully observable data). This distinguishes our approach from recent models like context encoder [[2](#bib.bib2), [3](#bib.bib3)], denoising autoencoder [[4](#bib.bib4)] or modified generative adversarial network [[5](#bib.bib5)], which require complete data as an output of the network in training. Moreover, our approach can be applied to various types of neural networks what requires only minimal modification in their architectures. Our main theoretical result shows that this generalization does not lead to loss of information when processing the input (Section [4](#S4 "4 Theoretical analysis ‣ Processing of missing data by neural networks")). Experimental results performed on several types of networks demonstrate practical usefulness of the method (see Section [5](#S5 "5 Experiments ‣ Processing of missing data by neural networks") and Figure [4](#A2.F4 "Figure 4 ‣ B.2 General identification property ‣ Appendix B Theoretical analysis ‣ Processing of missing data by neural networks") for sample results) .

∫ϕ​(wT​x+b)​FS​(x)​𝑑xitalic-ϕsuperscript𝑤𝑇𝑥𝑏subscript𝐹𝑆𝑥differential-d𝑥\int\phi(w^{T}x+b)F\_{S}(x)dxx1subscript𝑥1x\_{1} ⋆⋆\starx3subscript𝑥3x\_{3} ⋆⋆\starx5subscript𝑥5x\_{5} x6subscript𝑥6x\_{6} x7subscript𝑥7x\_{7} w1subscript𝑤1w\_{1}w2subscript𝑤2w\_{2}w3subscript𝑤3w\_{3}w4subscript𝑤4w\_{4}w5subscript𝑤5w\_{5}w6subscript𝑤6w\_{6}w7subscript𝑤7w\_{7}GMM params: (pi,mi,Σi)isubscriptsubscript𝑝𝑖subscript𝑚𝑖subscriptΣ𝑖𝑖(p\_{i},m\_{i},\Sigma\_{i})\_{i} ![Refer to caption](/html/1805.07405/assets/plotnn.png)
INPUTOUTPUT

Figure 1: Missing data point (x,J)𝑥𝐽(x,J), where x∈ℝD𝑥superscriptℝ𝐷x\in\mathbb{R}^{D} and J⊂{1,…,D}𝐽1…𝐷J\subset\{1,\ldots,D\} denotes absent attributes, is represented as a conditional density FSsubscript𝐹𝑆F\_{S} (data density restricted to the affine subspace S=Aff​[x,J]𝑆Aff𝑥𝐽S=\mathrm{Aff}[x,J] identified with (x,J)𝑥𝐽(x,J)). Instead of calculating the activation function ϕitalic-ϕ\phi on a single data point (as for complete data points), the first hidden layer computes the expected activation of neurons. Parameters of missing data density (pi,μi,Σi)isubscriptsubscript𝑝𝑖subscript𝜇𝑖subscriptΣ𝑖𝑖(p\_{i},\mu\_{i},\Sigma\_{i})\_{i} are tuned jointly with remaining network parameters.

## 2 Related work

Typical strategy for using machine learning methods with incomplete inputs relies on filling absent attributes based on observable ones [[6](#bib.bib6)], e.g. mean or k-NN imputation. One can also train separate models, e.g. neural networks [[7](#bib.bib7)], extreme learning machines (ELM) [[8](#bib.bib8)], k𝑘k-nearest neighbors [[9](#bib.bib9)], etc., for predicting the unobserved features. Iterative filling of missing attributes is one of the most popular technique in this class [[10](#bib.bib10), [11](#bib.bib11)]. Recently, a modified generative adversarial net (GAN) was adapted to fill in absent attributes with realistic values [[12](#bib.bib12)]. A supervised imputation, which learns a replacement value for each missing attribute jointly with remaining network parameters, was proposed in [[13](#bib.bib13)].

Instead of generating candidates for filling missing attributes, one can build a probabilistic model of incomplete data (under certain assumptions on missing mechanism) [[14](#bib.bib14), [15](#bib.bib15)], which is subsequently fed into particular learning model [[16](#bib.bib16), [17](#bib.bib17), [18](#bib.bib18), [19](#bib.bib19), [20](#bib.bib20), [21](#bib.bib21), [22](#bib.bib22), [23](#bib.bib23)].
Decision function can also be learned based on the visible inputs alone [[24](#bib.bib24), [25](#bib.bib25)], see [[26](#bib.bib26), [27](#bib.bib27)] for SVM and random forest cases. Pelckmans et. al. [[28](#bib.bib28)] modeled the expected risk under the uncertainty of the predicted outputs. The authors of [[29](#bib.bib29)] designed an algorithm for kernel classification under low-rank assumption, while Goldberg et. al. [[30](#bib.bib30)] used matrix completion strategy to solve missing data problem.

The paper [[31](#bib.bib31)] used recurrent neural networks with feedback into the input units, which fills absent attributes for the sole purpose of minimizing a learning criterion. By applying the rough set theory, the authors of [[32](#bib.bib32)] presented a feedforward neural network which gives an imprecise answer as the result of input data imperfection. Goodfellow et. al. [[33](#bib.bib33)] introduced the multi-prediction deep Boltzmann machine, which is capable of solving different inference problems, including classification with missing inputs.

Alternatively, missing data can be processed using the popular context encoder (CE) [[2](#bib.bib2), [3](#bib.bib3)] or modified GAN [[5](#bib.bib5)], which were proposed for filling missing regions in natural images. The other possibility would be to use denoising autoencoder [[4](#bib.bib4)], which was used e.g. for removing complex patterns like superimposed text from an image. Both approaches, however, require complete data as an output of the network in training phase, which is in contradiction with many real data sets (such us medical ones).

## 3 Layer for processing missing data

In this section, we present our methodology for feeding neural networks with missing data. We show how to represent incomplete data by probability density functions and how to generalize neuron’s activation function to process them.

Missing data representation. A missing data point is denoted by (x,J)𝑥𝐽(x,J), where x∈ℝD𝑥superscriptℝ𝐷x\in\mathbb{R}^{D} and J⊂{1,…,D}𝐽1…𝐷J\subset\{1,\ldots,D\} is a set of attributes with missing values. With each missing point (x,J)𝑥𝐽(x,J) we associate the affine subspace consisting of all points which coincide with x𝑥x on known coordinates J′={1,…,N}∖Jsuperscript𝐽′1…𝑁𝐽J^{\prime}=\{1,\ldots,N\}\setminus J:

|  |  |  |
| --- | --- | --- |
|  | S=Aff​[x,J]=x+span​(eJ),𝑆Aff𝑥𝐽𝑥spansubscript𝑒𝐽S=\mathrm{Aff}[x,J]=x+\mathrm{span}(e\_{J}), |  |

where eJ=[ej]j∈Jsubscript𝑒𝐽subscriptdelimited-[]subscript𝑒𝑗𝑗𝐽e\_{J}=[e\_{j}]\_{j\in J} and ejsubscript𝑒𝑗e\_{j} is j𝑗j-th canonical vector in ℝDsuperscriptℝ𝐷\mathbb{R}^{D}.

Let us assume that the values at missing attributes come from the unknown D𝐷D-dimensional probability distribution F𝐹F. Then we can model the unobserved values of (x,J)𝑥𝐽(x,J) by restricting F𝐹F to the affine subspace S=Aff​[x,J]𝑆Aff𝑥𝐽S=\mathrm{Aff}[x,J]. In consequence, possible values of incomplete data point (x,J)𝑥𝐽(x,J) are described by a conditional density111More precisely, FSsubscript𝐹𝑆F\_{S} equals a density F𝐹F conditioned on the observed attributes. FS:S→ℝ:subscript𝐹𝑆→𝑆ℝF\_{S}:S\to\mathbb{R} given by (see Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Processing of missing data by neural networks")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | FS​(x)={1∫SF​(s)​𝑑s​F​(x)​, for ​x∈S,0​, otherwise.subscript𝐹𝑆𝑥cases1subscript𝑆𝐹𝑠differential-d𝑠𝐹𝑥, for 𝑥𝑆otherwise0, otherwise.otherwiseF\_{S}(x)=\begin{cases}\frac{1}{\int\_{S}F(s)ds}F(x)\mbox{, for }x\in S,\\ 0\mbox{, otherwise.}\end{cases} |  | (1) |

Notice that FSsubscript𝐹𝑆F\_{S} is a degenerate density defined on the whole ℝDsuperscriptℝ𝐷\mathbb{R}^{D} space222An example of degenerate density is a degenerate Gaussian N​(m,Σ)𝑁𝑚ΣN(m,\Sigma), for which ΣΣ\Sigma is not invertible. A degenerate Gaussian is defined on affine subspace (given by image of ΣΣ\Sigma), see [[34](#bib.bib34)] for details. For simplicity we use the same notation N​(m,Σ)𝑁𝑚ΣN(m,\Sigma) to denote both standard and degenerate Gaussians., which allows to interpret it as a probabilistic representation of missing data point (x,J)𝑥𝐽(x,J).

In our approach, we use the mixture of Gaussians (GMM) with diagonal covariance matrices as a missing data density F𝐹F. The choice of diagonal covariance reduces the number of model parameters, which is crucial in high dimensional problems. Clearly, a conditional density for the mixture of Gaussians is a (degenerate) mixture of Gaussians with a support in the subspace. Moreover, we apply an additional regularization in the calculation of conditional density ([6](#A1.E6 "In A.1 Regularized conditional density ‣ Appendix A Missing data representation ‣ Processing of missing data by neural networks")) to avoid some artifacts when Gaussian densities are used333One can show that the conditional density of a missing point (x,J)𝑥𝐽(x,J) sufficiently distant from the data reduces to only one Gaussian, which center is nearest in the Mahalanobis distance to Aff​[x,J]Aff𝑥𝐽\mathrm{Aff}[x,J]. This regularization allows to move from typical conditional density given by ([6](#A1.E6 "In A.1 Regularized conditional density ‣ Appendix A Missing data representation ‣ Processing of missing data by neural networks")) to marginal density in the limiting case. Precise formulas for a regularized density for GMM with detailed explanations are presented in Supplementary Materials (section 1).

Generalized neuron’s response. To process probability density functions (representing missing data points) by neural networks, we generalize the neuron’s activation function. For a probability density function FSsubscript𝐹𝑆F\_{S}, we define the generalized response (activation) of a neuron n:ℝD→ℝ:𝑛→superscriptℝ𝐷ℝn:\mathbb{R}^{D}\to\mathbb{R} on FSsubscript𝐹𝑆F\_{S} as the mean output:

|  |  |  |
| --- | --- | --- |
|  | n​(FS)=E​[n​(x)|x∼FS]=∫n​(x)​FS​(x)​𝑑x.𝑛subscript𝐹𝑆𝐸delimited-[]similar-toconditional𝑛𝑥𝑥subscript𝐹𝑆𝑛𝑥subscript𝐹𝑆𝑥differential-d𝑥n(F\_{S})=E[n(x)|x\sim F\_{S}]=\int n(x)F\_{S}(x)dx. |  |

Observe that it is sufficient to generalize neuron’s response at the first layer only, while the rest of network architecture can remain unchanged. Basic requirement is the ability of computing expected value with respect to FSsubscript𝐹𝑆F\_{S}. We demonstrate that the generalized response of ReLU and RBF neurons with respect to the mixture of diagonal Gaussians can be calculated efficiently.

Let us recall that the ReLU neuron is given by

|  |  |  |
| --- | --- | --- |
|  | ReLUw,b​(x)=max⁡(wT​x+b,0),subscriptReLU  𝑤𝑏𝑥superscript𝑤𝑇𝑥𝑏0\mathrm{ReLU}\_{w,b}(x)=\max(w^{T}x+b,0), |  |

where w∈ℝD𝑤superscriptℝ𝐷w\in\mathbb{R}^{D} and b∈ℝ𝑏ℝb\in\mathbb{R} is the bias. Given 1-dimensional Gaussian density N​(m,σ2)𝑁𝑚superscript𝜎2N(m,\sigma^{2}), we first evaluate ReLU​[N​(m,σ2)]ReLUdelimited-[]𝑁𝑚superscript𝜎2\mathrm{ReLU}[N(m,\sigma^{2})], where ReLU=max⁡(0,x)ReLU0𝑥\mathrm{ReLU}=\max(0,x). If we define an auxiliary function:

|  |  |  |
| --- | --- | --- |
|  | N​R​(w)=ReLU​[N​(w,1)],NR𝑤ReLUdelimited-[]𝑁𝑤1\mbox{$\mathrm{N\!R}$}\!(w)=\mathrm{ReLU}[N(w,1)], |  |

then the generalized response equals:

|  |  |  |
| --- | --- | --- |
|  | ReLU​[N​(m,σ2)]=σ​N​R​(mσ).ReLUdelimited-[]𝑁𝑚superscript𝜎2𝜎NR𝑚𝜎\mathrm{ReLU}[N(m,\sigma^{2})]=\sigma\mbox{$\mathrm{N\!R}$}\!(\frac{m}{\sigma}). |  |

Elementary calculation gives:

|  |  |  |  |
| --- | --- | --- | --- |
|  | N​R​(w)=12​π​exp⁡(−w22)+w2​(1+erf​(w2)),NR𝑤12𝜋superscript𝑤22𝑤21erf𝑤2\mbox{$\mathrm{N\!R}$}\!(w)=\frac{1}{\sqrt{2\pi}}\exp(-\frac{w^{2}}{2})+\frac{w}{2}(1+\mathrm{erf}(\frac{w}{\sqrt{2}})), |  | (2) |

where erf​(z)=2p​i​∫0zexp⁡(−t2)​𝑑terf𝑧2𝑝𝑖superscriptsubscript0𝑧superscript𝑡2differential-d𝑡\mathrm{erf}(z)=\frac{2}{\sqrt{pi}}\int\_{0}^{z}\exp(-t^{2})dt.

We proceed with a general case, where an input data point x𝑥x is generated from the mixture of (degenerate) Gaussians. The following observation shows how to calculate the generalized response of ReLUw,b​(x)subscriptReLU

𝑤𝑏𝑥\mathrm{ReLU}\_{w,b}(x), where w∈ℝD,b∈ℝformulae-sequence𝑤superscriptℝ𝐷𝑏ℝw\in\mathbb{R}^{D},b\in\mathbb{R} are neuron weights.

###### Theorem 3.1.

Let F=∑ipi​N​(mi,Σi)𝐹subscript𝑖subscript𝑝𝑖𝑁subscript𝑚𝑖subscriptΣ𝑖F=\sum\_{i}p\_{i}N(m\_{i},\Sigma\_{i}) be the mixture of (possibly degenerate) Gaussians. Given weights w=(w1,…,wD)∈ℝD,b∈ℝformulae-sequence𝑤subscript𝑤1…subscript𝑤𝐷superscriptℝ𝐷𝑏ℝw=(w\_{1},\ldots,w\_{D})\in\mathbb{R}^{D},b\in\mathbb{R}, we have:

|  |  |  |
| --- | --- | --- |
|  | ReLUw,b​(F)=∑ipi​wT​Σi​w​N​R​(wT​mi+bwT​Σi​w).subscriptReLU  𝑤𝑏𝐹subscript𝑖subscript𝑝𝑖superscript𝑤𝑇subscriptΣ𝑖𝑤NRsuperscript𝑤𝑇subscript𝑚𝑖𝑏superscript𝑤𝑇subscriptΣ𝑖𝑤\mathrm{ReLU}\_{w,b}(F)=\sum\_{i}p\_{i}\sqrt{w^{T}\Sigma\_{i}w}\mbox{$\mathrm{N\!R}$}\!\big{(}\frac{w^{T}m\_{i}+b}{\sqrt{w^{T}\Sigma\_{i}w}}\big{)}. |  |

###### Proof.

If x∼N​(m,Σ)similar-to𝑥𝑁𝑚Σx\sim N(m,\Sigma) then wT​x+b∼N​(wT​x+b,wT​Σ​w)similar-tosuperscript𝑤𝑇𝑥𝑏𝑁superscript𝑤𝑇𝑥𝑏superscript𝑤𝑇Σ𝑤w^{T}x+b\sim N(w^{T}x+b,w^{T}\Sigma w). Consequently, if x∼∑ipi​N​(mi,Σi)similar-to𝑥subscript𝑖subscript𝑝𝑖𝑁subscript𝑚𝑖subscriptΣ𝑖x\sim\sum\_{i}p\_{i}N(m\_{i},\Sigma\_{i}), then wT​x+b∼∑ipi​N​(wT​mi+b,wT​Σi​w)similar-tosuperscript𝑤𝑇𝑥𝑏subscript𝑖subscript𝑝𝑖𝑁superscript𝑤𝑇subscript𝑚𝑖𝑏superscript𝑤𝑇subscriptΣ𝑖𝑤w^{T}x+b\sim\sum\_{i}p\_{i}N(w^{T}m\_{i}+b,w^{T}\Sigma\_{i}w).

Making use of ([2](#S3.E2 "In 3 Layer for processing missing data ‣ Processing of missing data by neural networks")), we get:

|  |  |  |
| --- | --- | --- |
|  | ReLUw,b​(F)=∫ℝReLU​(x)​∑ipi​N​(wT​mi+b,wT​Σi​w)​(x)​d​x=∑ipi​∫0∞x​N​(wT​mi+b,wT​Σi​w)​(x)​𝑑x=∑ipi​wT​Σi​w​N​R​(wT​mi+bwT​Σi​w).subscriptReLU  𝑤𝑏𝐹subscriptℝReLU𝑥subscript𝑖subscript𝑝𝑖𝑁superscript𝑤𝑇subscript𝑚𝑖𝑏superscript𝑤𝑇subscriptΣ𝑖𝑤𝑥𝑑𝑥subscript𝑖subscript𝑝𝑖superscriptsubscript0𝑥𝑁superscript𝑤𝑇subscript𝑚𝑖𝑏superscript𝑤𝑇subscriptΣ𝑖𝑤𝑥differential-d𝑥subscript𝑖subscript𝑝𝑖superscript𝑤𝑇subscriptΣ𝑖𝑤NRsuperscript𝑤𝑇subscript𝑚𝑖𝑏superscript𝑤𝑇subscriptΣ𝑖𝑤\mathrm{ReLU}\_{w,b}(F)=\int\_{\mathbb{R}}\mathrm{ReLU}(x)\sum\_{i}p\_{i}N(w^{T}m\_{i}+b,w^{T}\Sigma\_{i}w)(x)dx\\ =\sum\_{i}p\_{i}\int\_{0}^{\infty}xN(w^{T}m\_{i}+b,w^{T}\Sigma\_{i}w)(x)dx=\sum\_{i}p\_{i}\sqrt{w^{T}\Sigma\_{i}w}\mbox{$\mathrm{N\!R}$}\!\big{(}\frac{w^{T}m\_{i}+b}{\sqrt{w^{T}\Sigma\_{i}w}}\big{)}. |  |

∎

We show the formula for a generalized RBF neuron’s activation. Let us recall that RBF function is given by RBFc,Γ​(x)=N​(c,Γ)​(x)subscriptRBF

𝑐Γ𝑥𝑁𝑐Γ𝑥\mathrm{RBF}\_{c,\Gamma}(x)=N(c,\Gamma)(x).

###### Theorem 3.2.

Let F=∑ipi​N​(mi,Σi)𝐹subscript𝑖subscript𝑝𝑖𝑁subscript𝑚𝑖subscriptΣ𝑖F=\sum\_{i}p\_{i}N(m\_{i},\Sigma\_{i}) be the mixture of (possibly degenerate) Gaussians and let RBF unit be parametrized by N​(c,Γ)𝑁𝑐ΓN(c,\Gamma). We have:

|  |  |  |
| --- | --- | --- |
|  | RBFc,Γ​(F)=∑i=1kpi​N​(mi−c,Γ+Σi)​(0).subscriptRBF  𝑐Γ𝐹superscriptsubscript𝑖1𝑘subscript𝑝𝑖𝑁subscript𝑚𝑖𝑐ΓsubscriptΣ𝑖0\mathrm{RBF}\_{c,\Gamma}(F)=\sum\_{i=1}^{k}p\_{i}N(m\_{i}-c,\Gamma+\Sigma\_{i})(0). |  |

###### Proof.

We have:

|  |  |  |  |
| --- | --- | --- | --- |
|  | RBFc,Γ​(F)=∫ℝDRBFc,Γ​(x)​F​(x)​𝑑x=∑i=1kpi​∫ℝDN​(c,Γ)​(x)​N​(mi,Σi)​(x)​𝑑x=∑i=1kpi​⟨N​(c,Γ),N​(mi,Σi)⟩=∑i=1kpi​N​(mi−c,Γ+Σi)​(0).subscriptRBF  𝑐Γ𝐹subscriptsuperscriptℝ𝐷subscriptRBF  𝑐Γ𝑥𝐹𝑥differential-d𝑥superscriptsubscript𝑖1𝑘subscript𝑝𝑖subscriptsuperscriptℝ𝐷𝑁𝑐Γ𝑥𝑁superscript𝑚𝑖superscriptΣ𝑖𝑥differential-d𝑥superscriptsubscript𝑖1𝑘subscript𝑝𝑖  𝑁𝑐Γ𝑁superscript𝑚𝑖superscriptΣ𝑖superscriptsubscript𝑖1𝑘subscript𝑝𝑖𝑁subscript𝑚𝑖𝑐ΓsuperscriptΣ𝑖0\mathrm{RBF}\_{c,\Gamma}(F)=\int\_{\mathbb{R}^{D}}\mathrm{RBF}\_{c,\Gamma}(x)F(x)dx=\sum\_{i=1}^{k}p\_{i}\int\_{\mathbb{R}^{D}}N(c,\Gamma)(x)N(m^{i},\Sigma^{i})(x)dx\\[3.44444pt] =\sum\_{i=1}^{k}p\_{i}\langle N(c,\Gamma),N(m^{i},\Sigma^{i})\rangle=\sum\_{i=1}^{k}p\_{i}N(m\_{i}-c,\Gamma+\Sigma^{i})(0). |  | (3) |

∎

Network architecture. Adaptation of a given neural network to incomplete data relies on the following steps:

1. 1.

   Estimation of missing data density with the use of mixture of diagonal Gaussians. If data satisfy missing at random assumption (MAR), then we can adapt EM algorithm to estimate incomplete data density with the use of GMM. In more general case, we can let the network to learn optimal parameters of GMM with respect to its cost function444If huge amount of complete data is available during training, one should use variants of EM algorithm to estimate data density. It could be either used directly as a missing data density or tuned by neural networks with small amount of missing data.. The later case was examined in the experiment.
2. 2.

   Generalization of neuron’s response. A missing data point (x,J)𝑥𝐽(x,J) is interpreted as the mixture of degenerate Gaussians FSsubscript𝐹𝑆F\_{S} on S=Aff​[x,J]𝑆Aff𝑥𝐽S=\mathrm{Aff}[x,J]. Thus we need to generalize the activation functions of all neurons in the first hidden layer of the network to process probability measures. In consequence, the response of n​(⋅)𝑛⋅n(\cdot) on (x,J)𝑥𝐽(x,J) is given by n​(FS)𝑛subscript𝐹𝑆n(F\_{S}).

The rest of the architecture does not change, i.e. the modification is only required on the first hidden layer.

Observe that our generalized network can also process classical points, which do not contain any missing values. In this case, generalized neurons reduce to classical ones, because missing data density F𝐹F is only used to estimate possible values at absent attributes. If all attributes are complete then this density is simply not used. In consequence, if we want to use missing data in testing stage, we need to feed the network with incomplete data in training to fit accurate density model.

## 4 Theoretical analysis

There appears a natural question: how much information we lose using generalized neuron’s activation at the first layer? Our main theoretical result shows that our approach does not lead to the lose of information, which justifies our reasoning from a theoretical perspective. For a transparency, we will work with general probability measures instead of density functions. The generalized response of neuron n:ℝD→ℝ:𝑛→superscriptℝ𝐷ℝn:\mathbb{R}^{D}\to\mathbb{R} evaluated on a probability measure μ𝜇\mu is given by:

|  |  |  |
| --- | --- | --- |
|  | n​(μ):=∫n​(x)​𝑑μ​(x).assign𝑛𝜇𝑛𝑥differential-d𝜇𝑥n(\mu):=\int n(x)d\mu(x). |  |

The following theorem shows that a neural network with generalized ReLU units is able to identify any two probability measures. The proof is a natural modification of the respective standard proofs of Universal Approximation Property (UAP), and therefore we present only its sketch. Observe that all generalized ReLU return finite values iff a probability measure μ𝜇\mu satisfies the condition

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∫‖x‖​𝑑μ​(x)<∞.norm𝑥differential-d𝜇𝑥\int\|x\|d\mu(x)<\infty. |  | (4) |

That is the reason why we reduce to such measures in the following theorem.

###### Theorem 4.1.

Let μ,ν

𝜇𝜈\mu,\nu be probabilistic measures satisfying condition ([4](#S4.E4 "In 4 Theoretical analysis ‣ Processing of missing data by neural networks")).
If

|  |  |  |  |
| --- | --- | --- | --- |
|  | ReLUw,b​(μ)=ReLUw,b​(ν)​ for ​w∈ℝD,b∈ℝ,formulae-sequencesubscriptReLU  𝑤𝑏𝜇subscriptReLU  𝑤𝑏𝜈 for 𝑤superscriptℝ𝐷𝑏ℝ\mathrm{ReLU}\_{w,b}(\mu)=\mathrm{ReLU}\_{w,b}(\nu)\mbox{ for }w\in\mathbb{R}^{D},b\in\mathbb{R}, |  | (5) |

then ν=μ𝜈𝜇\nu=\mu.

###### Proof.

Let us fix an arbitrary w∈ℝD𝑤superscriptℝ𝐷w\in\mathbb{R}^{D} and define the set

|  |  |  |
| --- | --- | --- |
|  | ℱw={p:ℝ→ℝ:∫p​(wT​x)​𝑑μ​(x)=∫p​(wT​x)​𝑑ν​(x)}.subscriptℱ𝑤conditional-set𝑝:→ℝℝ𝑝superscript𝑤𝑇𝑥differential-d𝜇𝑥𝑝superscript𝑤𝑇𝑥differential-d𝜈𝑥\mathcal{F}\_{w}=\big{\{}p:\mathbb{R}\to\mathbb{R}:\int p(w^{T}x)d\mu(x)=\int p(w^{T}x)d\nu(x)\big{\}}. |  |

Our main step in the proof lies in showing that ℱwsubscriptℱ𝑤\mathcal{F}\_{w} contains all continuous bounded functions.

Let ri∈ℝsubscript𝑟𝑖ℝr\_{i}\in\mathbb{R} such that −∞=r0<r1<…<rl−1<rl=∞subscript𝑟0subscript𝑟1…subscript𝑟𝑙1subscript𝑟𝑙-\infty=r\_{0}<r\_{1}<\ldots<r\_{l-1}<r\_{l}=\infty and qi∈ℝsubscript𝑞𝑖ℝq\_{i}\in\mathbb{R} such that q0=q1=0=ql−1=qlsubscript𝑞0subscript𝑞10subscript𝑞𝑙1subscript𝑞𝑙q\_{0}=q\_{1}=0=q\_{l-1}=q\_{l}, be given. Let
Q:ℝ→ℝ:𝑄→ℝℝQ:\mathbb{R}\to\mathbb{R} be a piecewise linear continuous function which is affine linear on intervals [ri,ri+1]subscript𝑟𝑖subscript𝑟𝑖1[r\_{i},r\_{i+1}] and such that Q​(ri)=qi𝑄subscript𝑟𝑖subscript𝑞𝑖Q(r\_{i})=q\_{i}.
We show that Q∈ℱw𝑄subscriptℱ𝑤Q\in\mathcal{F}\_{w}.
Since

|  |  |  |
| --- | --- | --- |
|  | Q=∑i=1l−1qi⋅Tri−1,ri,ri+1,𝑄superscriptsubscript𝑖1𝑙1⋅subscript𝑞𝑖subscript𝑇  subscript𝑟𝑖1subscript𝑟𝑖subscript𝑟𝑖1Q=\sum\_{i=1}^{l-1}q\_{i}\cdot T\_{r\_{i-1},r\_{i},r\_{i+1}}, |  |

where the tent-like piecewise linear function T𝑇T is defined by

|  |  |  |
| --- | --- | --- |
|  | Tp0,p1,p2​(r)={0​ for ​r≤p0,r−p0p1−p0​ for ​r∈[p0,p1],p2−rp2−p1​ for ​r∈[p1,p2],0​ for ​r≥p2,subscript𝑇  subscript𝑝0subscript𝑝1subscript𝑝2𝑟cases0 for 𝑟subscript𝑝0otherwise𝑟subscript𝑝0subscript𝑝1subscript𝑝0 for 𝑟subscript𝑝0subscript𝑝1otherwisesubscript𝑝2𝑟subscript𝑝2subscript𝑝1 for 𝑟subscript𝑝1subscript𝑝2otherwise0 for 𝑟subscript𝑝2otherwiseT\_{p\_{0},p\_{1},p\_{2}}(r)=\begin{cases}0\mbox{ for }r\leq p\_{0},\\ \frac{r-p\_{0}}{p\_{1}-p\_{0}}\mbox{ for }r\in[p\_{0},p\_{1}],\\ \frac{p\_{2}-r}{p\_{2}-p\_{1}}\mbox{ for }r\in[p\_{1},p\_{2}],\\ 0\mbox{ for }r\geq p\_{2},\end{cases} |  |

it is sufficient to prove that T∈ℱw𝑇subscriptℱ𝑤T\in\mathcal{F}\_{w}. Let Mp​(r)=max⁡(0,r−p)subscript𝑀𝑝𝑟0𝑟𝑝M\_{p}(r)=\max(0,r-p). Clearly

|  |  |  |
| --- | --- | --- |
|  | Tp0,p1,p2=1p1−p0⋅(Mp0−Mp1)−1p2−p1⋅(Mp2−Mp1).subscript𝑇  subscript𝑝0subscript𝑝1subscript𝑝2⋅1subscript𝑝1subscript𝑝0subscript𝑀subscript𝑝0subscript𝑀subscript𝑝1⋅1subscript𝑝2subscript𝑝1subscript𝑀subscript𝑝2subscript𝑀subscript𝑝1T\_{p\_{0},p\_{1},p\_{2}}=\frac{1}{p\_{1}-p\_{0}}\cdot(M\_{p\_{0}}-M\_{p\_{1}})-\frac{1}{p\_{2}-p\_{1}}\cdot(M\_{p\_{2}}-M\_{p\_{1}}). |  |

However, directly from ([5](#S4.E5 "In Theorem 4.1. ‣ 4 Theoretical analysis ‣ Processing of missing data by neural networks")) we see that Mp∈ℱwsubscript𝑀𝑝subscriptℱ𝑤M\_{p}\in\mathcal{F}\_{w} for every p𝑝p, and consequently T𝑇T and Q𝑄Q are also in ℱwsubscriptℱ𝑤\mathcal{F}\_{w}.

Now let us fix an arbitrary bounded continuous function G𝐺G. We show that G∈ℱw𝐺subscriptℱ𝑤G\in\mathcal{F}\_{w}. To observe this, take an arbitrary uniformly bounded sequence of piecewise linear functions described before which is convergent pointwise to G𝐺G. By the Lebesgue dominated
convergence theorem we obtain that G∈ℱw𝐺subscriptℱ𝑤G\in\mathcal{F}\_{w}.

Therefore cos⁡(⋅),sin⁡(⋅)∈ℱw

⋅⋅
subscriptℱ𝑤\cos(\cdot),\sin(\cdot)\in\mathcal{F}\_{w} holds consequently also for the function ei​r=cos⁡r+i​sin⁡rsuperscript𝑒𝑖𝑟𝑟𝑖𝑟e^{ir}=\cos r+i\sin r we have the equality

|  |  |  |
| --- | --- | --- |
|  | ∫exp⁡(i​wT​x)​𝑑μ​(x)=∫exp⁡(i​wT​x)​𝑑ν​(x).𝑖superscript𝑤𝑇𝑥differential-d𝜇𝑥𝑖superscript𝑤𝑇𝑥differential-d𝜈𝑥\int\exp(iw^{T}x)d\mu(x)=\int\exp(iw^{T}x)d\nu(x). |  |

Since w∈ℝD𝑤superscriptℝ𝐷w\in\mathbb{R}^{D} was chosen arbitrarily, this means that the characteristic functions of two measures coincide, and therefore μ=ν𝜇𝜈\mu=\nu.
∎

It is possible to obtain an analogical result for RBF activation function. Moreover, we can also get more general result under stronger assumptions on considered probability measures. More precisely, if a given family of neurons satisfies UAP, then their generalization is also capable of identifying any probability measure with compact support. Complete analysis of both cases is presented in Supplementary Material (section 2).

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| original | mask | k-nn | mean | dropout | our | CE |

​ ![Refer to caption](/html/1805.07405/assets/mnist_4_merged.png)

Figure 2: Reconstructions of partially incomplete images using the autoencoder. From left: (1) original image, (2) image with missing pixels passed to autoencooder; the output produced by autoencoder when unknown pixels were initially filled by (3) k-nn imputation and (4) mean imputation; (5) the results obtained by autoencoder with dropout, (6) our method and (7) context encoder. All columns except the last one were obtained with loss function computed based on pixels from outside the mask (no fully observable data available in training phase). It can be noticed that our method gives much sharper images than the competitive methods.

## 5 Experiments

We evaluated our model on three types of architectures. First, as a proof of concept, we verified the proposed approach in the context of autoencoder (AE). Next we applied multilayer perceptron (MLP) to multiclass classification problem and finally we used shallow radial basis function network (RBFN) in binary classification. For a comparison we only considered methods with publicly available codes and thus many methods described in the related work section have not been taken into account. The code implementing the proposed method is available at <https://github.com/lstruski/Processing-of-missing-data-by-neural-networks>.

Autoencoder. Autoencoder (AE) is usually used for generating compressed representation of data. However, in this experiment, we were interested in restoring corrupted images, where part of data was hidden.

As a data set, we used grayscale handwritten digits retrieved from MNIST database. For each image of the size 28×28=784282878428\times 28=784 pixels, we removed a square patch of the
size555In the case when the removed patch size was smaller, all considered methods performed very well and cannot be visibly distinguished. 13×13131313\times 13. The location of the patch was uniformly sampled for each image. AE used in the experiments consists of 5 hidden layers with 256, 128, 64, 128, 256 neurons in subsequent layers. The first layer was parametrized by ReLU activation functions, while the remaining units used sigmoids666We also experimented with ReLU in remaining layers (except the last one), however the results we have obtained were less plausible..

As describe in Section [1](#S1 "1 Introduction ‣ Processing of missing data by neural networks"), our model assumes that there is no complete data in training phase. Therefore, the loss function was computed based only on pixels from outside the mask.

As a baseline, we considered combination of analogical architecture with popular imputation techniques:

k-nn: Missing features were replaced with mean values of those features computed from the K𝐾K nearest training samples (we used K=5𝐾5K=5). Neighborhood was measured using Euclidean distance in the subspace of observed features.

mean: Missing features were replaced with mean values of those features computed for all (incomplete) training samples.

dropout: Input neurons with missing values were dropped777Values of the remaining neurons were divided by 1−d​r​o​p​o​u​t​r​a​t​e1𝑑𝑟𝑜𝑝𝑜𝑢𝑡𝑟𝑎𝑡𝑒1-dropout\ rate.

Table 1: Mean square error of reconstruction on MNIST incomplete images (we report the errors calculated over the whole area, inside and outside the mask). Described errors are obtained for images with intensities scaled to [0,1]01[0,1].

|  | only missing data | | | | complete data |
| --- | --- | --- | --- | --- | --- |
|  | k-nn | mean | dropout | our | CE |
| Total error | 0.011890.011890.01189 | 0.017270.017270.01727 | 0.013790.013790.01379 | 0.010560.01056{\bf 0.01056} | 0.013260.013260.01326 |
| Error inside the mask | 0.007220.007220.00722 | 0.008980.008980.00898 | 0.008820.008820.00882 | 0.008100.008100.00810 | 0.007100.00710{\bf 0.00710} |
| Error outside the mask | 0.004680.004680.00468 | 0.008290.008290.00829 | 0.004980.004980.00498 | 0.002460.00246{\bf 0.00246} | 0.006170.006170.00617 |

Additionally, we used a type of context encoder (CE), where missing features were replaced with mean values, however in contrast to mean imputation, the complete data were used as an output of the network in training phase. This model was expected to perform better, because it used complete data in computing the network loss function.

Incomplete inputs and their reconstructions obtained with various approaches are presented in Figure [4](#A2.F4 "Figure 4 ‣ B.2 General identification property ‣ Appendix B Theoretical analysis ‣ Processing of missing data by neural networks") (more examples are included in Supplementary Material, section 3). It can be observed that our method gives sharper images then the competitive methods. In order to support the qualitative results, we calculated mean square error of reconstruction (see Table [1](#S5.T1 "Table 1 ‣ 5 Experiments ‣ Processing of missing data by neural networks")). Quantitative results confirm that our method has lower error than imputation methods, both inside and outside the mask. Moreover, it overcomes CE in case of the whole area and the area outside the mask. In case of the area inside the mask, CE error is only slightly better than ours, however CE requires complete data in training.

Multilayer perceptron. In this experiment, we considered a typical MLP architecture with 3 ReLU hidden layers. It was applied to multiclass classification problem on Epileptic Seizure Recognition data set (ESR) taken from [[35](#bib.bib35)]. Each 178-dimensional vector (out of 11500 samples) is EEG recording of a given person for 1 second, categorized into one of 5 classes. To generate missing attributes, we randomly removed 25%, 50%, 75% and 90% of values.

Table 2: Classification results on ESR data obtained using MLP (the results of CE are not bolded, because it had access to complete examples).

|  | only missing data | | | | | | complete data |
| --- | --- | --- | --- | --- | --- | --- | --- |
| % of missing | k-nn | mice | mean | gmm | dropout | our | CE |
| 25% | 0.7730.7730.773 | 0.8230.823{\bf 0.823} | 0.7990.7990.799 | 0.8230.823{\bf 0.823} | 0.7960.7960.796 | 0.8150.8150.815 | 0.8120.8120.812 |
| 50% | 0.7730.7730.773 | 0.8160.8160.816 | 0.7030.7030.703 | 0.8010.8010.801 | 0.7800.7800.780 | 0.8170.817{\bf 0.817} | 0.8130.8130.813 |
| 75% | 0.6280.6280.628 | 0.7860.7860.786 | 0.6240.6240.624 | 0.7480.7480.748 | 0.7550.7550.755 | 0.7870.787{\bf 0.787} | 0.7920.7920.792 |
| 90% | 0.6150.6150.615 | 0.6700.6700.670 | 0.5960.5960.596 | 0.6970.6970.697 | 0.7490.7490.749 | 0.7600.760{\bf 0.760} | 0.7710.7710.771 |

In addition to the imputation methods described in the previous experiment, we also used iterative filling of missing attributes using Multiple Imputation by Chained Equation (mice), where several imputations are drawing from the conditional distribution of data by Markov chain Monte Carlo techniques [[10](#bib.bib10), [11](#bib.bib11)]. Moreover, we considered the mixture of Gaussians (gmm), where missing features were replaced with values sampled from GMM estimated from incomplete data using EM algorithm888Due to the high-dimensionality of MNIST data, mice was not able to construct imputations in previous experiment. Analogically, EM algorithm was not able to fit GMM because of singularity of covariance matrices..

We applied double 5-fold cross-validation procedure to report classification results and we tuned required hyper-parameters.
The number of the mixture components for our method was selected in the inner cross-validation from the possible values {2,3,5}235\{2,3,5\}. Initial mixture of Gaussians was selected using classical GMM with diagonal matrices. The results were assessed using classical accuracy measure.

The results presented in Table [2](#S5.T2 "Table 2 ‣ 5 Experiments ‣ Processing of missing data by neural networks") show the advantage of our model over classical imputation methods, which give reasonable results only for low number of missing values. It is also slightly better than dropout, which is more robust to the number of absent attributes than typical imputations. It can be seen that our method gives comparable scores to CE, even though CE had access to complete training data. We also ran MLP on complete ESR data (with no missing attributes), which gave 0.8360.8360.836 of accuracy.

Table 3: Summary of data sets with internally absent attributes.

| Data set | #Instances | #Attributes | #Missing |
| --- | --- | --- | --- |
| bands | 539 | 19 | 5.38% |
| kidney disease | 400 | 24 | 10.54% |
| hepatitis | 155 | 19 | 5.67% |
| horse | 368 | 22 | 23.80% |
| mammographics | 961 | 5 | 3.37% |
| pima | 768 | 8 | 12.24% |
| winconsin | 699 | 9 | 0.25% |

Radial basis function network. RBFN can be considered as a minimal architecture implementing our model, which contains only one hidden layer. We used cross-entropy function applied on a softmax in the output layer. This network suits well for small low-dimensional data.

For the evaluation, we considered two-class data sets retrieved from UCI repository [[36](#bib.bib36)] with internally missing attributes, see Table [3](#S5.T3 "Table 3 ‣ 5 Experiments ‣ Processing of missing data by neural networks") (more data sets are included in Supplementary Materials, section 4). Since the classification is binary, we extended baseline with two additional SVM kernel models which work directly with incomplete data without performing any imputations:

geom: Its objective function is based on the geometric interpretation of the margin and aims to maximize the margin of each sample in its own relevant subspace [[26](#bib.bib26)].

karma: This algorithm iteratively tunes kernel classifier under low-rank assumptions [[29](#bib.bib29)].

The above SVM methods were combined with RBF kernel function.

We applied analogical cross-validation procedure as before. The number of RBF units was selected in the inner cross-validation from the range {25,50,75,100}255075100\{25,50,75,100\}. Initial centers of RBFNs were randomly selected from training data while variances were samples from N​(0,1)𝑁01N(0,1). For SVM methods, the margin parameter C𝐶C and kernel radius γ𝛾\gamma were selected from {2k:k=−5,−3,…,9}conditional-setsuperscript2𝑘𝑘

53…9\{2^{k}:k=-5,-3,\ldots,9\} for both parameters. For karma, additional parameter γk​a​r​m​asubscript𝛾𝑘𝑎𝑟𝑚𝑎\gamma\_{karma} was selected from the set {1,2}12\{1,2\}.

The results, presented in Table [4](#S5.T4 "Table 4 ‣ 5 Experiments ‣ Processing of missing data by neural networks"), indicate that our model outperformed imputation techniques in almost all cases. It partially confirms that the use of raw incomplete data in neural networks is usually better approach than filling missing attributes before learning process. Moreover, it obtained more accurate results than modified kernel methods, which directly work on incomplete data.

Table 4: Classification results obtained using RBFN (the results of CE are not bolded, because it had access to complete examples).

|  | only missing data | | | | | | | | complete data |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data | karma | geom | k-nn | mice | mean | gmm | dropout | our | CE |
| bands | 0.5800.5800.580 | 0.5710.5710.571 | 0.5200.5200.520 | 0.5440.5440.544 | 0.5450.5450.545 | 0.5770.5770.577 | 0.6160.616{\bf 0.616} | 0.5980.5980.598 | 0.6210.6210.621 |
| kidney | 0.9950.995{\bf 0.995} | 0.9860.9860.986 | 0.9920.9920.992 | 0.9920.9920.992 | 0.9850.9850.985 | 0.9800.9800.980 | 0.9830.9830.983 | 0.9930.9930.993 | 0.9960.9960.996 |
| hepatitis | 0.6650.6650.665 | 0.8170.8170.817 | 0.8250.8250.825 | 0.7920.7920.792 | 0.8250.8250.825 | 0.8200.8200.820 | 0.7800.7800.780 | 0.8460.846{\bf 0.846} | 0.8430.8430.843 |
| horse | 0.8260.8260.826 | 0.8220.8220.822 | 0.8070.8070.807 | 0.8200.8200.820 | 0.7930.7930.793 | 0.8180.8180.818 | 0.8230.8230.823 | 0.8640.864{\bf 0.864} | 0.8580.8580.858 |
| mammogr. | 0.7730.7730.773 | 0.8150.8150.815 | 0.8220.8220.822 | 0.8250.8250.825 | 0.8190.8190.819 | 0.8030.8030.803 | 0.8140.8140.814 | 0.8310.831{\bf 0.831} | 0.8220.8220.822 |
| pima | 0.7680.7680.768 | 0.7660.7660.766 | 0.7670.7670.767 | 0.7690.769{\bf 0.769} | 0.7600.7600.760 | 0.7420.7420.742 | 0.7540.7540.754 | 0.7470.7470.747 | 0.7430.7430.743 |
| winconsin | 0.9580.9580.958 | 0.9580.9580.958 | 0.9670.9670.967 | 0.9700.970{\bf 0.970} | 0.9650.9650.965 | 0.9570.9570.957 | 0.9640.9640.964 | 0.9700.970{\bf 0.970} | 0.9680.9680.968 |

## 6 Conclusion

In this paper, we proposed a general approach for adapting neural networks to process incomplete data, which is able to train on data set containing only incomplete samples. Our strategy introduces input layer for processing missing data, which can be used for a wide range of networks and does not require their extensive modifications. Thanks to representing incomplete data with probability density function, it is possible to determine more generalized and accurate response (activation) of the neuron. We showed that this generalization is justified from a theoretical perspective. The experiments confirm its practical usefulness in various tasks and for diverse network architectures. In particular, it gives comparable results to the methods, which require complete data in training.

## Acknowledgement

This work was partially supported by National Science Centre, Poland (grants no. 2016/21/D/ST6/00980, 2015/19/B/ST6/01819, 2015/19/D/ST6/01215, 2015/19/D/ST6/01472). We would like to thank the anonymous reviewers for their valuable comments on our paper.

## References

* [1]

  Ian Goodfellow, Yoshua Bengio, and Aaron Courville.
  Deep learning.
  MIT press, 2016.
* [2]

  Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A
  Efros.
  Context encoders: Feature learning by inpainting.
  In Proceedings of the IEEE Conference on Computer Vision and
  Pattern Recognition, pages 2536–2544, 2016.
* [3]

  Chao Yang, Xin Lu, Zhe Lin, Eli Shechtman, Oliver Wang, and Hao Li.
  High-resolution image inpainting using multi-scale neural patch
  synthesis.
  In The IEEE Conference on Computer Vision and Pattern
  Recognition (CVPR), volume 1, page 3, 2017.
* [4]

  Junyuan Xie, Linli Xu, and Enhong Chen.
  Image denoising and inpainting with deep neural networks.
  In Advances in neural information processing systems, pages
  341–349, 2012.
* [5]

  Raymond A Yeh, Chen Chen, Teck Yian Lim, Alexander G Schwing, Mark
  Hasegawa-Johnson, and Minh N Do.
  Semantic image inpainting with deep generative models.
  In Proceedings of the IEEE Conference on Computer Vision and
  Pattern Recognition, pages 5485–5493, 2017.
* [6]

  Patrick E McKnight, Katherine M McKnight, Souraya Sidani, and Aurelio Jose
  Figueredo.
  Missing data: A gentle introduction.
  Guilford Press, 2007.
* [7]

  Peter K Sharpe and RJ Solly.
  Dealing with missing values in neural network-based diagnostic
  systems.
  Neural Computing & Applications, 3(2):73–77, 1995.
* [8]

  Dušan Sovilj, Emil Eirola, Yoan Miche, Kaj-Mikael Björk, Rui Nian,
  Anton Akusok, and Amaury Lendasse.
  Extreme learning machine for missing data using multiple imputations.
  Neurocomputing, 174:220–231, 2016.
* [9]

  Gustavo EAPA Batista, Maria Carolina Monard, et al.
  A study of k-nearest neighbour as an imputation method.
  HIS, 87(251-260):48, 2002.
* [10]

  Stef Buuren and Karin Groothuis-Oudshoorn.
  mice: Multivariate imputation by chained equations in r.
  Journal of statistical software, 45(3), 2011.
* [11]

  Melissa J Azur, Elizabeth A Stuart, Constantine Frangakis, and Philip J Leaf.
  Multiple imputation by chained equations: what is it and how does it
  work?
  International journal of methods in psychiatric research,
  20(1):40–49, 2011.
* [12]

  Jinsung Yoon, James Jordon, and Mihaela van der Schaar.
  Gain: Missing data imputation using generative adversarial nets.
  pages 5689–5698, 2018.
* [13]

  Maya Gupta, Andrew Cotter, Jan Pfeifer, Konstantin Voevodski, Kevin Canini,
  Alexander Mangylov, Wojciech Moczydlowski, and Alexander Van Esbroeck.
  Monotonic calibrated interpolated look-up tables.
  The Journal of Machine Learning Research, 17(1):3790–3836,
  2016.
* [14]

  Zoubin Ghahramani and Michael I Jordan.
  Supervised learning from incomplete data via an EM approach.
  In Advances in Neural Information Processing Systems, pages
  120–127. Citeseer, 1994.
* [15]

  Volker Tresp, Subutai Ahmad, and Ralph Neuneier.
  Training neural networks with deficient data.
  In Advances in neural information processing systems, pages
  128–135, 1994.
* [16]

  Marek Śmieja, Łukasz Struski, and Jacek Tabor.
  Generalized rbf kernel for incomplete data.
  arXiv preprint arXiv:1612.01480, 2016.
* [17]

  David Williams, Xuejun Liao, Ya Xue, and Lawrence Carin.
  Incomplete-data classification using logistic regression.
  In Proceedings of the International Conference on Machine
  Learning, pages 972–979. ACM, 2005.
* [18]

  Alexander J Smola, SVN Vishwanathan, and Thomas Hofmann.
  Kernel methods for missing variables.
  In Proceedings of the International Conference on Artificial
  Intelligence and Statistics. Citeseer, 2005.
* [19]

  David Williams and Lawrence Carin.
  Analytical kernel matrix completion with incomplete multi-view data.
  In Proceedings of the ICML Workshop on Learning With Multiple
  Views, 2005.
* [20]

  Pannagadatta K Shivaswamy, Chiranjib Bhattacharyya, and Alexander J Smola.
  Second order cone programming approaches for handling missing and
  uncertain data.
  Journal of Machine Learning Research, 7:1283–1314, 2006.
* [21]

  Diego PP Mesquita, João PP Gomes, and Leonardo R Rodrigues.
  Extreme learning machines for datasets with missing values using the
  unscented transform.
  In Intelligent Systems (BRACIS), 2016 5th Brazilian Conference
  on, pages 85–90. IEEE, 2016.
* [22]

  Xuejun Liao, Hui Li, and Lawrence Carin.
  Quadratically gated mixture of experts for incomplete data
  classification.
  In Proceedings of the International Conference on Machine
  Learning, pages 553–560. ACM, 2007.
* [23]

  Uwe Dick, Peter Haider, and Tobias Scheffer.
  Learning from incomplete data with infinite imputations.
  In Proceedings of the International Conference on Machine
  Learning, pages 232–239. ACM, 2008.
* [24]

  Ofer Dekel, Ohad Shamir, and Lin Xiao.
  Learning to classify with missing and corrupted features.
  Machine Learning, 81(2):149–178, 2010.
* [25]

  Amir Globerson and Sam Roweis.
  Nightmare at test time: robust learning by feature deletion.
  In Proceedings of the International Conference on Machine
  Learning, pages 353–360. ACM, 2006.
* [26]

  Gal Chechik, Geremy Heitz, Gal Elidan, Pieter Abbeel, and Daphne Koller.
  Max-margin classification of data with absent features.
  Journal of Machine Learning Research, 9:1–21, 2008.
* [27]

  Jing Xia, Shengyu Zhang, Guolong Cai, Li Li, Qing Pan, Jing Yan, and Gangmin
  Ning.
  Adjusted weight voting algorithm for random forests in handling
  missing values.
  Pattern Recognition, 69:52–60, 2017.
* [28]

  Kristiaan Pelckmans, Jos De Brabanter, Johan AK Suykens, and Bart De Moor.
  Handling missing values in support vector machine classifiers.
  Neural Networks, 18(5):684–692, 2005.
* [29]

  Elad Hazan, Roi Livni, and Yishay Mansour.
  Classification with low rank and missing data.
  In Proceedings of The 32nd International Conference on Machine
  Learning, pages 257–266, 2015.
* [30]

  Andrew Goldberg, Ben Recht, Junming Xu, Robert Nowak, and Xiaojin Zhu.
  Transduction with matrix completion: Three birds with one stone.
  In Advances in neural information processing systems, pages
  757–765, 2010.
* [31]

  Yoshua Bengio and Francois Gingras.
  Recurrent neural networks for missing or asynchronous data.
  In Advances in neural information processing systems, pages
  395–401, 1996.
* [32]

  Robert K Nowicki, Rafal Scherer, and Leszek Rutkowski.
  Novel rough neural network for classification with missing data.
  In Methods and Models in Automation and Robotics (MMAR), 2016
  21st International Conference on, pages 820–825. IEEE, 2016.
* [33]

  Ian Goodfellow, Mehdi Mirza, Aaron Courville, and Yoshua Bengio.
  Multi-prediction deep boltzmann machines.
  In Advances in Neural Information Processing Systems, pages
  548–556, 2013.
* [34]

  Calyampudi Radhakrishna Rao, Calyampudi Radhakrishna Rao, Mathematischer
  Statistiker, Calyampudi Radhakrishna Rao, and Calyampudi Radhakrishna Rao.
  Linear statistical inference and its applications, volume 2.
  Wiley New York, 1973.
* [35]

  Ralph G Andrzejak, Klaus Lehnertz, Florian Mormann, Christoph Rieke, Peter
  David, and Christian E Elger.
  Indications of nonlinear deterministic and finite-dimensional
  structures in time series of brain electrical activity: Dependence on
  recording region and brain state.
  Physical Review E, 64(6):061907, 2001.
* [36]

  Arthur Asuncion and David J. Newman.
  UCI Machine Learning Repository, 2007.

## Appendix A Missing data representation

In this section, we show how to regularize typical conditional probability density function. Next, we present complete formulas for a conditional density in the case of the mixture of Gaussians.

### A.1 Regularized conditional density

Let us recall a definition of conditional density representing missing data points formulated in the paper. We assume that F𝐹F is a probability density function on data space ℝDsuperscriptℝ𝐷\mathbb{R}^{D}. A missing data point (x,J)𝑥𝐽(x,J) can be represented by restricting F𝐹F to the affine subspace S=Aff​[x,J]𝑆Aff𝑥𝐽S=\mathrm{Aff}[x,J], which gives a conditional density FS:S→ℝ:subscript𝐹𝑆→𝑆ℝF\_{S}:S\to\mathbb{R} given by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | FS​(x)={1∫SF​(s)​𝑑s​F​(x)​, for ​x∈S,0​, otherwise,subscript𝐹𝑆𝑥cases1subscript𝑆𝐹𝑠differential-d𝑠𝐹𝑥, for 𝑥𝑆otherwise0, otherwise,otherwiseF\_{S}(x)=\begin{cases}\frac{1}{\int\_{S}F(s)ds}F(x)\mbox{, for }x\in S,\\ 0\mbox{, otherwise,}\end{cases} |  | (6) |

The natural choice for missing data density F𝐹F is to apply GMM. However, the straightforward application of GMM may lead to some practical problems with taking the conditional density ([6](#A1.E6 "In A.1 Regularized conditional density ‣ Appendix A Missing data representation ‣ Processing of missing data by neural networks")). Thus, to provide better representation of missing data points we introduce additional regularization described in this section.

Let us observe that the formula ([6](#A1.E6 "In A.1 Regularized conditional density ‣ Appendix A Missing data representation ‣ Processing of missing data by neural networks")) is not well-defined in the case when the density function F𝐹F is identically zero on the affine space S=Aff​[x,J]𝑆Aff𝑥𝐽S=\mathrm{Aff}[x,J]. In practice, the same problem appears numerically for the mixture of gaussians, because every component has exponentially fast decrease. In consequence, ([6](#A1.E6 "In A.1 Regularized conditional density ‣ Appendix A Missing data representation ‣ Processing of missing data by neural networks")) either gives no sense, or trivializes999One can show that the conditional density of a missing point (x,J)𝑥𝐽(x,J) sufficiently distant from the data reduces to only one gaussian, which center is nearest in the Mahalanobis distance to Aff​[x,J]Aff𝑥𝐽\mathrm{Aff}[x,J] (reduces to only one gaussian) for points sufficiently far from the main clusters. To some extent we can also explain this problem with the fact that the real density can have much slower decrease to infinity than gaussians, and therefore the estimation of conditional density based on gaussian mixture becomes unreliable.

To overcome this problem, which occurs in the case of gaussian distributions, we introduce the regularized γ𝛾\gamma-conditional densities, where γ>0𝛾0\gamma>0 is a regularization parameter. Intuitively, the regularization allows to control the influence of F𝐹F outside S=Aff​[x,J]𝑆Aff𝑥𝐽S=\mathrm{Aff}[x,J] on conditional density FSsubscript𝐹𝑆F\_{S}. In consequence, the mixture components (in the case of GMM) could have higher impact on the final conditional density even if they are located far from S𝑆S. The Figure [3](#A1.F3 "Figure 3 ‣ A.1 Regularized conditional density ‣ Appendix A Missing data representation ‣ Processing of missing data by neural networks") illustrates the regularization effect for different values of γ𝛾\gamma. We are indebted to the classical idea behind the definition of conditional probability.

![Refer to caption](/html/1805.07405/assets/gamma_0_sigma_1.png)


(a) Conditional density (γ→0→𝛾0\gamma\to 0)

![Refer to caption](/html/1805.07405/assets/gamma_1_sigma_1.png)


(b) Intermediate case (γ=1𝛾1\gamma=1)

![Refer to caption](/html/1805.07405/assets/gamma_1000_sigma_1.png)


(c) Marginal density (γ→∞→𝛾\gamma\to\infty)

Figure 3: Illustration of probabilistic representation FSsubscript𝐹𝑆F\_{S} of missing data point (∗,−1)∈ℝ21superscriptℝ2(\*,-1)\in\mathbb{R}^{2} for different regularization parameters γ𝛾\gamma when data density is given by the mixture of two Gaussians.

Let γ>0𝛾0\gamma>0 be a regularization parameter. By regularized γ𝛾\gamma-restriction of
F𝐹F to the affine subspace S𝑆S of ℝDsuperscriptℝ𝐷\mathbb{R}^{D} we understand

|  |  |  |
| --- | --- | --- |
|  | Fγ|S​(x)={∫S⟂xF​(s)⋅N​(x,γ​IS−x)​(s)​𝑑s​, if ​x∈S,0​ otherwise,evaluated-atsuperscript𝐹𝛾𝑆𝑥casessubscriptsuperscriptsubscript𝑆perpendicular-to𝑥⋅𝐹𝑠𝑁𝑥𝛾subscript𝐼𝑆𝑥𝑠differential-d𝑠, if 𝑥𝑆otherwise0 otherwise,otherwiseF^{\gamma}|\_{S}(x)=\begin{cases}\int\_{S\_{\perp}^{x}}F(s)\cdot N(x,\gamma I\_{S-x})(s)ds\mbox{, if }x\in S,\\ 0\mbox{ otherwise,}\end{cases} |  |

where S⟂x={w:(w−x)⟂(S−x)}superscriptsubscript𝑆perpendicular-to𝑥conditional-set𝑤perpendicular-to𝑤𝑥𝑆𝑥S\_{\perp}^{x}=\{w:(w-x)\perp(S-x)\} is the affine space consisting of all
points which are at x𝑥x perpendicular to S𝑆S, and N​(x,γ​IS−x)𝑁𝑥𝛾subscript𝐼𝑆𝑥N(x,\gamma I\_{S-x}) is the degenerate normal density which has mean at x𝑥x, is supported on S𝑆S and its covariance matrix is a rescaled identity (restricted to S−x𝑆𝑥S-x).
Then the regularized γ𝛾\gamma-conditional density FSγsubscriptsuperscript𝐹𝛾𝑆F^{\gamma}\_{S} is defined as the normalization of Fγ|Sevaluated-atsuperscript𝐹𝛾𝑆F^{\gamma}|\_{S}:

|  |  |  |
| --- | --- | --- |
|  | FSγ={1∫SFγ|S​(s)​d​s​Fγ|S​ for ​s∈S,0​ otherwise.subscriptsuperscript𝐹𝛾𝑆casesevaluated-at1evaluated-atsubscript𝑆superscript𝐹𝛾𝑆𝑠𝑑𝑠superscript𝐹𝛾𝑆 for 𝑠𝑆otherwise0 otherwise.otherwiseF^{\gamma}\_{S}=\begin{cases}\frac{1}{\int\_{S}F^{\gamma}|\_{S}(s)ds}F^{\gamma}|\_{S}\mbox{ for }s\in S,\\ 0\mbox{ otherwise.}\end{cases} |  |

The regularized density FSγsubscriptsuperscript𝐹𝛾𝑆F^{\gamma}\_{S} has the following properties:

1. 1.

   FSγsubscriptsuperscript𝐹𝛾𝑆F^{\gamma}\_{S} is well-defined degenerate density on S𝑆S for every γ𝛾\gamma,
2. 2.

   FSγsubscriptsuperscript𝐹𝛾𝑆F^{\gamma}\_{S} converges to the conditional density FSsubscript𝐹𝑆F\_{S} with γ→0→𝛾0\gamma\to 0, see Figure [3(a)](#A1.F3.sf1 "In Figure 3 ‣ A.1 Regularized conditional density ‣ Appendix A Missing data representation ‣ Processing of missing data by neural networks")
3. 3.

   FSγsubscriptsuperscript𝐹𝛾𝑆F^{\gamma}\_{S} converges to the marginal density as γ→∞→𝛾\gamma\to\infty, see Figure [3(c)](#A1.F3.sf3 "In Figure 3 ‣ A.1 Regularized conditional density ‣ Appendix A Missing data representation ‣ Processing of missing data by neural networks").

One can easily see that the first point follows directly from the fact that F𝐹F is integrable.
Since for an arbitrary function g𝑔g we have ∫g​(s)​N​(x,γ​I)​(s)​𝑑s→f​(x)→𝑔𝑠𝑁𝑥𝛾𝐼𝑠differential-d𝑠𝑓𝑥\int g(s)N(x,\gamma I)(s)ds\to f(x), as γ→0→𝛾0\gamma\to 0, we obtain that

|  |  |  |
| --- | --- | --- |
|  | limγ→0Fγ|S​(x)=F​(x)​ for ​x∈S.evaluated-atsubscript→𝛾0superscript𝐹𝛾𝑆𝑥𝐹𝑥 for 𝑥𝑆\lim\_{\gamma\to 0}F^{\gamma}|\_{S}(x)=F(x)\mbox{ for }x\in S. |  |

Thus Fγ|S→F|S→evaluated-atsuperscript𝐹𝛾𝑆evaluated-at𝐹𝑆F^{\gamma}|\_{S}\to F|\_{S}, as γ→0→𝛾0\gamma\to 0.
Analogously

|  |  |  |
| --- | --- | --- |
|  | limγ→∞∫1N​(0,γ​I)​(0)​g​(s)​N​(x,γ​I)​(s)​𝑑ssubscript→𝛾1𝑁0𝛾𝐼0𝑔𝑠𝑁𝑥𝛾𝐼𝑠differential-d𝑠\lim\_{\gamma\to\infty}\int\frac{1}{N(0,\gamma I)(0)}g(s)N(x,\gamma I)(s)ds |  |

|  |  |  |
| --- | --- | --- |
|  | =limγ→∞∫g​(s)​exp⁡(−12​γ​‖x−s‖2)​𝑑s=∫g​(s)​𝑑s,absentsubscript→𝛾𝑔𝑠12𝛾superscriptnorm𝑥𝑠2differential-d𝑠𝑔𝑠differential-d𝑠=\lim\_{\gamma\to\infty}\int g(s)\exp(-\frac{1}{2\gamma}\|x-s\|^{2})ds=\int g(s)ds, |  |

which implies that for large γ𝛾\gamma the function FSγsubscriptsuperscript𝐹𝛾𝑆F^{\gamma}\_{S} as a renormalization of Fγ|Sevaluated-atsuperscript𝐹𝛾𝑆F^{\gamma}|\_{S} at point x𝑥x converges to

|  |  |  |
| --- | --- | --- |
|  | ∫S⟂xF​(s)​𝑑s,subscriptsuperscriptsubscript𝑆perpendicular-to𝑥𝐹𝑠differential-d𝑠\int\_{S\_{\perp}^{x}}F(s)ds, |  |

which is exactly the value of marginal density at point x𝑥x.

### A.2 Gaussian model for missing data density

We consider the case of F𝐹F given by GMM and calculate analytical formula for the regularized γ𝛾\gamma-conditional density. To reduce the number of parameters and provide more reliable estimation in high dimensional space, we use diagonal covariance matrix for each mixture component.

We will need the following notation: given a point x∈ℝD𝑥superscriptℝ𝐷x\in\mathbb{R}^{D} and a set of indexes K⊂{1,…,D}𝐾1…𝐷K\subset\{1,\ldots,D\} by xKsubscript𝑥𝐾x\_{K} we denote the restriction of x𝑥x to the set of indexes K𝐾K. The complementary set to K𝐾K is denoted by K′superscript𝐾′K^{\prime}. Given x,y

𝑥𝑦x,y, by [xK′,yK]subscript𝑥superscript𝐾′subscript𝑦𝐾[x\_{K^{\prime}},y\_{K}] we denote a point in ℝDsuperscriptℝ𝐷\mathbb{R}^{D} which coordinates equal x𝑥x on K′superscript𝐾′K^{\prime} and y𝑦y on K𝐾K. We use analogous notation for matrices.

One obtains the following exact formula for regularized restriction of gaussian density with diagonal covariance:

###### Proposition A.1.

Let N​(m,Σ)𝑁𝑚ΣN(m,\Sigma) be non-degenerate normal density with a diagonal covariance Σ=diag​(σ1,…,σD)Σdiagsubscript𝜎1…subscript𝜎𝐷\Sigma=\mathrm{diag}(\sigma\_{1},\ldots,\sigma\_{D}). We consider a missing data point (x,J)𝑥𝐽(x,J) represented by the affine subspace S=Aff​[x,J]𝑆Aff𝑥𝐽S=\mathrm{Aff}[x,J]. Let γ>0𝛾0\gamma>0 be a regularization parameter.

The γ𝛾\gamma-regularized restriction of F𝐹F to S𝑆S at point s=[xJ′,yJ]∈S𝑠subscript𝑥superscript𝐽′subscript𝑦𝐽𝑆s=[x\_{J^{\prime}},y\_{J}]\in S equals:

|  |  |  |
| --- | --- | --- |
|  | Fγ|S​(s)=Cm,Σ,Sγ​N​(mS,ΣS)​(s),evaluated-atsuperscript𝐹𝛾𝑆𝑠subscriptsuperscript𝐶𝛾  𝑚Σ𝑆𝑁subscript𝑚𝑆subscriptΣ𝑆𝑠F^{\gamma}|\_{S}(s)=C^{\gamma}\_{m,\Sigma,S}N(m\_{S},\Sigma\_{S})(s), |  |

where

|  |  |  |
| --- | --- | --- |
|  | mS=[xJ′,mJ],ΣS=[0J′​J′,ΣJ​J],formulae-sequencesubscript𝑚𝑆subscript𝑥superscript𝐽′subscript𝑚𝐽subscriptΣ𝑆subscript0superscript𝐽′superscript𝐽′subscriptΣ𝐽𝐽m\_{S}=[x\_{J^{\prime}},m\_{J}],\Sigma\_{S}=[0\_{J^{\prime}J^{\prime}},\Sigma\_{JJ}], |  |

|  |  |  |
| --- | --- | --- |
|  | Cm,Σ,Sγ=1(2​π)(D−|J|)/2​∏l∈J′(γ+σl)1/2⋅exp⁡(−12​∑l∈J′1γ+σl​(ml−xl)2).subscriptsuperscript𝐶𝛾  𝑚Σ𝑆absent1superscript2𝜋𝐷𝐽2subscriptproduct𝑙superscript𝐽′superscript𝛾subscript𝜎𝑙12missing-subexpression⋅absent12subscript𝑙superscript𝐽′1𝛾subscript𝜎𝑙superscriptsubscript𝑚𝑙subscript𝑥𝑙2\begin{array}[]{ll}C^{\gamma}\_{m,\Sigma,S}=&\displaystyle{\frac{1}{(2\pi)^{(D-|J|)/2}\prod\_{l\in J^{\prime}}(\gamma+\sigma\_{l})^{1/2}}}\\[12.91663pt] &\cdot\exp(-\frac{1}{2}\sum\_{l\in J^{\prime}}\frac{1}{\gamma+\sigma\_{l}}(m\_{l}-x\_{l})^{2}).\end{array} |  |

Finally, by using the above proposition (after normalization) we get the formula for the regularized conditional density in the case of the mixture of gaussians:

###### Corollary A.1.

Let F𝐹F be the mixture of nondegenerate gaussians

|  |  |  |
| --- | --- | --- |
|  | F=∑ipi​N​(mi,Σi),𝐹subscript𝑖subscript𝑝𝑖𝑁subscript𝑚𝑖subscriptΣ𝑖F=\sum\_{i}p\_{i}N(m\_{i},\Sigma\_{i}), |  |

where all Σi=diag​(σ1i,…,σDi)subscriptΣ𝑖diagsubscriptsuperscript𝜎𝑖1…subscriptsuperscript𝜎𝑖𝐷\Sigma\_{i}=\mathrm{diag}(\sigma^{i}\_{1},\ldots,\sigma^{i}\_{D}) and let S=Aff​[x,J]𝑆Aff𝑥𝐽S=\mathrm{Aff}[x,J].

Then

|  |  |  |
| --- | --- | --- |
|  | FSγ=∑iri​N​(mSi,ΣSi),subscriptsuperscript𝐹𝛾𝑆subscript𝑖subscript𝑟𝑖𝑁subscriptsuperscript𝑚𝑖𝑆subscriptsuperscriptΣ𝑖𝑆F^{\gamma}\_{S}=\sum\_{i}r\_{i}N(m^{i}\_{S},\Sigma^{i}\_{S}), |  |

where

|  |  |  |
| --- | --- | --- |
|  | mSi=[xJ′,(mi)J],ΣSi=[0J′​J′,(Σi)J​J],ri=qi∑jqj,qi=Cmi,Σi,Sγ⋅pi,formulae-sequencesubscriptsuperscript𝑚𝑖𝑆subscript𝑥superscript𝐽′subscriptsubscript𝑚𝑖𝐽subscriptsuperscriptΣ𝑖𝑆subscript0superscript𝐽′superscript𝐽′subscriptsubscriptΣ𝑖𝐽𝐽formulae-sequencesubscript𝑟𝑖subscript𝑞𝑖subscript𝑗subscript𝑞𝑗subscript𝑞𝑖⋅subscriptsuperscript𝐶𝛾  subscript𝑚𝑖subscriptΣ𝑖𝑆subscript𝑝𝑖\begin{array}[]{l}\displaystyle{m^{i}\_{S}=[x\_{J^{\prime}},(m\_{i})\_{J}],\Sigma^{i}\_{S}=[0\_{J^{\prime}J^{\prime}},(\Sigma\_{i})\_{JJ}]},\\[4.30554pt] \displaystyle{r\_{i}=\frac{q\_{i}}{\sum\_{j}q\_{j}},q\_{i}=C^{\gamma}\_{m\_{i},\Sigma\_{i},S}\cdot p\_{i}},\\ \end{array} |  |

|  |  |  |
| --- | --- | --- |
|  | Cm,Σ,Sγ=1(2​π)(D−|J|)/2​∏l∈J′(γ+σl)1/2⋅exp(−12∑l∈J′1γ+σl(ml−xl2).\begin{array}[]{ll}C^{\gamma}\_{m,\Sigma,S}=&\displaystyle{\frac{1}{(2\pi)^{(D-|J|)/2}\prod\_{l\in J^{\prime}}(\gamma+\sigma\_{l})^{1/2}}}\\[12.91663pt] &\cdot\exp(-\frac{1}{2}\sum\_{l\in J^{\prime}}\frac{1}{\gamma+\sigma\_{l}}(m\_{l}-x\_{l}^{2}).\\[8.61108pt] \end{array} |  |

## Appendix B Theoretical analysis

In this section, we continue a theoretical analysis of our model. First, we consider a special case of RBF neurons for arbitrary probability measures. Next, we restrict our attention to the measures with compact supports and show that the identification property holds for neurons satisfying UAP

### B.1 Identification property for RBF

RBF function is given by

|  |  |  |
| --- | --- | --- |
|  | RBFm,Σ​(x)=N​(m,Σ)​(x),subscriptRBF  𝑚Σ𝑥𝑁𝑚Σ𝑥\mathrm{RBF}\_{m,\Sigma}(x)=N(m,\Sigma)(x), |  |

where m𝑚m is an arbitrary point and ΣΣ\Sigma is positively defined symmetric matrix. In some cases one often restricts to either diagonal or rescaled identities Σ=α​IΣ𝛼𝐼\Sigma=\alpha I, where α>0𝛼0\alpha>0. In the last case we use the notation RBFm,αsubscriptRBF

𝑚𝛼\mathrm{RBF}\_{m,\alpha} for RBFm,α​IsubscriptRBF

𝑚𝛼𝐼\mathrm{RBF}\_{m,\alpha I}.

###### Theorem B.1.

Let μ,ν

𝜇𝜈\mu,\nu be probabilistic measures.
If

|  |  |  |
| --- | --- | --- |
|  | RBFm,α​(μ)=RBFm,α​(ν)​ for every ​m∈ℝD,α>0,formulae-sequencesubscriptRBF  𝑚𝛼𝜇subscriptRBF  𝑚𝛼𝜈 for every 𝑚superscriptℝ𝐷𝛼0\mathrm{RBF}\_{m,\alpha}(\mu)=\mathrm{RBF}\_{m,\alpha}(\nu)\mbox{ for every }m\in\mathbb{R}^{D},\alpha>0, |  |

then ν=μ𝜈𝜇\nu=\mu.

###### Proof.

We will show that μ𝜇\mu and ν𝜈\nu coincide on every cube. Recall that
(η∗f)​(x)=∫η​(y)⋅f​(x−y)​𝑑λN​(x)𝜂𝑓𝑥⋅𝜂𝑦𝑓𝑥𝑦differential-dsubscript𝜆𝑁𝑥(\eta\*f)(x)=\int\eta(y)\cdot f(x-y)d\lambda\_{N}(x).

Let us first observe that for an arbitrary cube K=a+[0,h]D𝐾𝑎superscript0ℎ𝐷K=a+[0,h]^{D}

|  |  |  |
| --- | --- | --- |
|  | ∫𝟙K∗N​(0,α)​𝑑μ​(x)=∫𝟙K∗N​(0,α)​𝑑ν​(x),subscript1𝐾𝑁0𝛼differential-d𝜇𝑥subscript1𝐾𝑁0𝛼differential-d𝜈𝑥\int\mathds{1}\_{K}\*N(0,\alpha)d\mu(x)=\int\mathds{1}\_{K}\*N(0,\alpha)d\nu(x), |  |

where h>0ℎ0h>0 is arbitrary.
This follows from the obvious observation that

|  |  |  |
| --- | --- | --- |
|  | 1nK​∑i∈ℤD∩[0,n]DN​(a+in​h,α​I)1superscript𝑛𝐾subscript𝑖superscriptℤ𝐷superscript0𝑛𝐷𝑁𝑎𝑖𝑛ℎ𝛼𝐼\frac{1}{n^{K}}\sum\_{i\in\mathbb{Z}^{D}\cap[0,n]^{D}}N(a+\tfrac{i}{n}h,\alpha I) |  |

converges uniformly to 𝟙K∗N​(0,α​I)subscript1𝐾𝑁0𝛼𝐼\mathds{1}\_{K}\*N(0,\alpha I), as n𝑛n goes to ∞\infty.

Since 𝟙K+1n​[0,1]n∗N​(0,1n4​I)subscript1𝐾1𝑛superscript01𝑛𝑁01superscript𝑛4𝐼\mathds{1}\_{K+\frac{1}{n}[0,1]^{n}}\*N(0,\frac{1}{n^{4}}I) converges pointwise to 𝟙Ksubscript1𝐾\mathds{1}\_{K}, analogously as before by applying Lebesgue dominated convergence theorem we obtain the assertion.
∎

### B.2 General identification property

We begin with recalling the UAP (universal approximation property). We say that a family of neurons 𝒩𝒩\mathcal{N} has UAP if for every compact set K⊂ℝD𝐾superscriptℝ𝐷K\subset\mathbb{R}^{D} and a continuous function f:K→ℝ:𝑓→𝐾ℝf:K\to\mathbb{R} the function f𝑓f can be arbitrarily close approximated with respect to supremum norm by span​(𝒩)span𝒩\mathrm{span}(\mathcal{N}) (linear combinations of elements of 𝒩𝒩\mathcal{N}).

Our result shows that if a given family of neurons satisfies UAP, then their generalization allows to distinguish any two probability measures with compact support:

###### Theorem B.2.

Let μ,ν

𝜇𝜈\mu,\nu be probabilistic measures with compact support.
Let 𝒩𝒩\mathcal{N} be a family of functions having UAP.

If

|  |  |  |  |
| --- | --- | --- | --- |
|  | n​(μ)=n​(ν)​ for every ​n∈𝒩,𝑛𝜇𝑛𝜈 for every 𝑛𝒩n(\mu)=n(\nu)\mbox{ for every }n\in\mathcal{N}, |  | (7) |

then ν=μ𝜈𝜇\nu=\mu.

###### Proof.

Since μ,ν

𝜇𝜈\mu,\nu have compact support, we can take R>1𝑅1R>1 such that suppμ,suppν⊂B​(0,R−1)

supp𝜇supp𝜈
𝐵0𝑅1\operatorname\*{supp}\mu,\operatorname\*{supp}\nu\subset B(0,R-1), where B​(a,r)𝐵𝑎𝑟B(a,r) denotes the closed ball centered at a𝑎a and with radius r𝑟r. To prove that measures μ,ν

𝜇𝜈\mu,\nu are equal it is obviously sufficient to prove that they coincide on each ball B​(a,r)𝐵𝑎𝑟B(a,r) with arbitrary a∈B​(0,R−1)𝑎𝐵0𝑅1a\in B(0,R-1) and radius r<1𝑟1r<1.

Let ϕnsubscriptitalic-ϕ𝑛\phi\_{n} be defined by

|  |  |  |
| --- | --- | --- |
|  | ϕn​(x)=1−n⋅d​(x,B​(a,r))​ for ​x∈ℝD,subscriptitalic-ϕ𝑛𝑥1⋅𝑛𝑑𝑥𝐵𝑎𝑟 for 𝑥superscriptℝ𝐷\phi\_{n}(x)=1-n\cdot d(x,B(a,r))\mbox{ for }x\in\mathbb{R}^{D}, |  |

where d​(x,U)𝑑𝑥𝑈d(x,U) denotes the distance of point x𝑥x from the set U𝑈U.
Observe that ϕnsubscriptitalic-ϕ𝑛\phi\_{n} is a continuous function which is one on B​(a,r)𝐵𝑎𝑟B(a,r) an and zero on ℝD∖B​(a,r+1/n)superscriptℝ𝐷𝐵𝑎𝑟1𝑛\mathbb{R}^{D}\setminus B(a,r+1/n), and therefore ϕnsubscriptitalic-ϕ𝑛\phi\_{n} is a uniformly bounded sequence of functions which converges pointwise to the characteristic funtion 𝟙B​(a,r)subscript1𝐵𝑎𝑟\mathds{1}\_{B(a,r)} of the set B​(a,r)𝐵𝑎𝑟B(a,r).

By the UAP property we choose ψn∈span​(𝒩)subscript𝜓𝑛span𝒩\psi\_{n}\in\mathrm{span}(\mathcal{N}) such that

|  |  |  |
| --- | --- | --- |
|  | suppx∈B​(0,R)|ϕn​(x)−ψn​(x)|≤1/n.subscriptsupp𝑥𝐵0𝑅subscriptitalic-ϕ𝑛𝑥subscript𝜓𝑛𝑥1𝑛\operatorname\*{supp}\_{x\in B(0,R)}|\phi\_{n}(x)-\psi\_{n}(x)|\leq 1/n. |  |

By the above also ψnsubscript𝜓𝑛\psi\_{n} restricted to B​(0,R)𝐵0𝑅B(0,R) is a uniformly bounded sequence of functions which converges pointwise to 𝟙B​(a,r)subscript1𝐵𝑎𝑟\mathds{1}\_{B(a,r)}.
Since ψn∈𝒩subscript𝜓𝑛𝒩\psi\_{n}\in\mathcal{N}, by ([7](#A2.E7 "In Theorem B.2. ‣ B.2 General identification property ‣ Appendix B Theoretical analysis ‣ Processing of missing data by neural networks")) we get

|  |  |  |
| --- | --- | --- |
|  | ∫ψn​(x)​𝑑μ​(x)=∫ψn​(x)​𝑑ν​(x).subscript𝜓𝑛𝑥differential-d𝜇𝑥subscript𝜓𝑛𝑥differential-d𝜈𝑥\int\psi\_{n}(x)d\mu(x)=\int\psi\_{n}(x)d\nu(x). |  |

Now by the Lebesgue dominated convergence theorem
we trivially get

|  |  |  |
| --- | --- | --- |
|  | ∫ψn​(x)​𝑑μ​(x)=∫B​(0,R)ψn​(x)​𝑑μ​(x)→μ​(B​(a,r)),∫ψn​(x)​𝑑ν​(x)=∫B​(0,R)ψn​(x)​𝑑ν​(x)→ν​(B​(a,r)),subscript𝜓𝑛𝑥differential-d𝜇𝑥subscript𝐵0𝑅subscript𝜓𝑛𝑥differential-d𝜇𝑥→𝜇𝐵𝑎𝑟subscript𝜓𝑛𝑥differential-d𝜈𝑥subscript𝐵0𝑅subscript𝜓𝑛𝑥differential-d𝜈𝑥→𝜈𝐵𝑎𝑟\begin{array}[]{l}\displaystyle{\int\psi\_{n}(x)d\mu(x)=\int\_{B(0,R)}\psi\_{n}(x)d\mu(x)\to\mu(B(a,r)),}\\[4.30554pt] \displaystyle{\int\psi\_{n}(x)d\nu(x)=\int\_{B(0,R)}\psi\_{n}(x)d\nu(x)\to\nu(B(a,r)),}\end{array} |  |

which makes the proof complete.
∎

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| original | mask | k-nn | mean | dropout | our | CE |

​ ![Refer to caption](/html/1805.07405/assets/mnist_10_merged.png)

Figure 4: More reconstructions of partially incomplete images using the autoencoder. From left: (1) original image, (2) image with missing pixels passed to autoencooder; the output produced by autoencoder when absent pixels were initially filled by (3) k-nn imputation and (4) mean imputation; (5) the results obtained by autoencoder with (5) dropout, (6) our method and (7) context encoder. All columns except the last one were obtained with loss function computed based on pixels from outside the mask (no fully observable data available in training phase). It can be noticed that our method gives much sharper images then the competitive methods.




Table 5: Summary of data sets, where 50% of values were removed randomly.

| Data set | #Instances | #Attributes |
| --- | --- | --- |
| australian | 690 | 14 |
| bank | 1372 | 4 |
| breast cancer | 699 | 8 |
| crashes | 540 | 20 |
| diabetes | 768 | 8 |
| fourclass | 862 | 2 |
| heart | 270 | 13 |
| liver disorders | 345 | 6 |




Table 6: Classification results measured by accuracy on UCI data sets with 50% of removed attributes.

| data | karma | geom | k-nn | mice | mean | dropout | our |
| --- | --- | --- | --- | --- | --- | --- | --- |
| australian | 0.8330.833{\bf 0.833} | 0.8020.8020.802 | 0.8200.8200.820 | 0.8260.8260.826 | 0.8080.8080.808 | 0.8120.8120.812 | 0.8330.833{\bf 0.833} |
| bank | 0.7990.799{\bf 0.799} | 0.7400.7400.740 | 0.7630.7630.763 | 0.7930.7930.793 | 0.7880.7880.788 | 0.7220.7220.722 | 0.7950.7950.795 |
| breast cancer | 0.9380.9380.938 | 0.8740.8740.874 | 0.9020.9020.902 | 0.9420.9420.942 | 0.9380.9380.938 | 0.9110.9110.911 | 0.9510.951{\bf 0.951} |
| crashes | 0.9200.920{\bf 0.920} | 0.9140.9140.914 | 0.8980.8980.898 | 0.8940.8940.894 | 0.8920.8920.892 | 0.9000.9000.900 | 0.9200.920{\bf 0.920} |
| diabetes | 0.6950.6950.695 | 0.6440.6440.644 | 0.6730.6730.673 | 0.7080.708{\bf 0.708} | 0.6990.6990.699 | 0.6750.6750.675 | 0.6900.6900.690 |
| fourclass | 0.8080.808{\bf 0.808} | 0.6530.6530.653 | 0.7660.7660.766 | 0.7760.7760.776 | 0.7660.7660.766 | 0.7310.7310.731 | 0.7370.7370.737 |
| heart | 0.7550.7550.755 | 0.7380.7380.738 | 0.7250.7250.725 | 0.7510.7510.751 | 0.7250.7250.725 | 0.7220.7220.722 | 0.7700.770{\bf 0.770} |
| liver disorders | 0.5300.5300.530 | 0.5910.5910.591 | 0.5650.5650.565 | 0.5760.5760.576 | 0.5620.5620.562 | 0.5710.5710.571 | 0.6080.608{\bf 0.608} |

## Appendix C Reconstruction of incomplete MNIST images

Due to the limited space in the paper, we could only present 4 sample images from MNIST experiment. In Figure [4](#A2.F4 "Figure 4 ‣ B.2 General identification property ‣ Appendix B Theoretical analysis ‣ Processing of missing data by neural networks"), we present more examples from this experiment.

## Appendix D Additional RBFN experiment

In addition to data sets reported in the paper, we also ran RBFN on 8 examples retrieved from UCI repository, see Table [5](#A2.T5 "Table 5 ‣ B.2 General identification property ‣ Appendix B Theoretical analysis ‣ Processing of missing data by neural networks"). These are complete data sets (with no missing attributes). To generate missing samples, we randomly removed 50% of values.

The results presented in Table [6](#A2.T6 "Table 6 ‣ B.2 General identification property ‣ Appendix B Theoretical analysis ‣ Processing of missing data by neural networks") confirm the effects reported in the paper. Our method outperformed imputation techniques in almost all case and was slightly better than karma algorithm.

## Appendix E Computational complexity

We analyze the computational complexity of applying a layer for missing data processing with k𝑘k Gaussians for modeling missing data density. Given an incomplete data point S=Aff​[x,J]𝑆Aff𝑥𝐽S=\mathrm{Aff}[x,J], where x∈ℝD𝑥superscriptℝ𝐷x\in\mathbb{R}^{D} and J⊂{1,…,N}𝐽1…𝑁J\subset\{1,\ldots,N\}, the cost of calculation of regularized (degenerate) density FSγsuperscriptsubscript𝐹𝑆𝛾F\_{S}^{\gamma} is O​(k​|J′|)𝑂𝑘superscript𝐽′O(k|J^{\prime}|), where J′={1,…,N}∖Jsuperscript𝐽′1…𝑁𝐽J^{\prime}=\{1,\ldots,N\}\setminus J (see Corollary 1.1. in supplementary material). Computation of a generalized ReLU activation (Theorem 3.1) takes O​(k​D+k​|J|)𝑂𝑘𝐷𝑘𝐽O(kD+k|J|). If we have t𝑡t neurons in the first layer, then a total cost of applying our layer is O​(k​|J′|+t​k​(D+|J|))𝑂𝑘superscript𝐽′𝑡𝑘𝐷𝐽O(k|J^{\prime}|+tk(D+|J|)).

In contrast, for a complete data point we need to compute t𝑡t ReLU activations, which is O​(t​D)𝑂𝑡𝐷O(tD). In consequence, generalized activations can be about 2​k2𝑘2k times slower than working on complete data.

## Appendix F Learning missing data density

![Refer to caption](/html/1805.07405/assets/x1.png)


(a) Reference classification

![Refer to caption](/html/1805.07405/assets/x2.png)


(b) Initial Gaussians

![Refer to caption](/html/1805.07405/assets/x3.png)


(c) Final Gaussians and resulted classification

Figure 5: Toy example of learning missing data density by the network.

To run our model we need to define initial mixture of Gaussians. This distribution is passed to the network and its parameters are tuned jointly with remaining network weights to minimize the overall cost of the network.

We illustrate this behavior on the following toy example. We generated a data set from the mixture of four Gaussians; two of them were labeled as class 1 (green) while the remaining two were labeled as class 2 (blue), see Figure [5(a)](#A6.F5.sf1 "In Figure 5 ‣ Appendix F Learning missing data density ‣ Processing of missing data by neural networks"). We removed one of the attributes from randomly selected data points x=(x1,x2)𝑥subscript𝑥1subscript𝑥2x=(x\_{1},x\_{2}) with x1<0subscript𝑥10x\_{1}<0. In other words, we generated missing samples only from two Gaussian on the left. Figure [5(b)](#A6.F5.sf2 "In Figure 5 ‣ Appendix F Learning missing data density ‣ Processing of missing data by neural networks") shows initial GMM passed to the network. As can be seen this GMM matches neither a data density nor a density of missing samples. After training, we get a GMM, where its first component estimates a density of class 1, while the second component matches class 2, see Figure [5(c)](#A6.F5.sf3 "In Figure 5 ‣ Appendix F Learning missing data density ‣ Processing of missing data by neural networks"). In consequence, learning missing data density by the network helped to perform better classification than estimating GMM directly by EM algorithm.

[◄](/html/1805.07404)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/1805.07405)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+1805.07405)
[View original  
on arXiv](https://arxiv.org/abs/1805.07405)[►](/html/1805.07406)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Mon Mar 11 05:09:34 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
