---
arxiv: '2410.21265'
authors:
- Jeremy Bernstein
- Laker Newhouse
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Modular Duality in Deep Learning
url: https://arxiv.org/abs/2410.21265
year: 2024
---

# Modular Duality in Deep Learning

Jeremy Bernstein jbernstein@mit.edu
  
Laker Newhouse lakern@mit.edu
  
MIT CSAIL

###### Abstract

An old idea in optimization theory says that since the gradient is a dual vector it may not be subtracted from the weights without first being mapped to the primal space where the weights reside. We take this idea seriously in this paper and construct such a duality map for general neural networks. Our map, which we call modular dualization, forms a unifying theoretical basis for training algorithms that are a) fast and b) scalable. Modular dualization involves first assigning operator norms to layers based on the semantics of each layer, and then using these layerwise norms to recursively induce a duality map on the weight space of the full neural architecture. We conclude by deriving GPU-friendly algorithms for dualizing 𝖤𝗆𝖻𝖾𝖽𝖤𝗆𝖻𝖾𝖽\mathsf{Embed}, 𝖫𝗂𝗇𝖾𝖺𝗋𝖫𝗂𝗇𝖾𝖺𝗋\mathsf{Linear} and 𝖢𝗈𝗇𝗏𝟤𝖣𝖢𝗈𝗇𝗏𝟤𝖣\mathsf{Conv2D} layers—the latter two methods are based on a new rectangular Newton-Schulz iteration that we propose. Our iteration was recently used to set new speed records for training NanoGPT. Overall, we hope that our theory of modular duality will yield a next generation of fast and scalable optimizers for general neural architectures.

## 1 Introduction

In this paper, we pursue a rigorous and first-principles theoretical framework for designing neural network training algorithms. We hope that building such a framework will facilitate the design of a next generation of fast and scalable optimizers that are automatically tailored to different neural architectures.

While gradient descent is the workhorse of modern machine learning, the most vanilla form of the algorithm does not, in our view, pass a basic type check. For a gradient update to type check, we insist that the gradient must be passed through a duality map before being multiplied by a learning rate and applied to the weights:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝚠𝚎𝚒𝚐𝚑𝚝−𝙻𝚁∗𝚠𝚎𝚒𝚐𝚑𝚝.𝚐𝚛𝚊𝚍formulae-sequence𝚠𝚎𝚒𝚐𝚑𝚝𝙻𝚁𝚠𝚎𝚒𝚐𝚑𝚝𝚐𝚛𝚊𝚍\displaystyle\mathtt{weight}-\mathtt{LR}\,\mathtt{\*}\,\mathtt{weight.grad} | type check: failed! |  | (1) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝚠𝚎𝚒𝚐𝚑𝚝−𝙻𝚁∗𝚍𝚞𝚊𝚕𝚒𝚣𝚎(𝚠𝚎𝚒𝚐𝚑𝚝.𝚐𝚛𝚊𝚍)\displaystyle\mathtt{weight}-\mathtt{LR}\,\mathtt{\*}\,\mathtt{dualize}(\mathtt{weight.grad}) | type check: passed! |  | (2) |

Why? The reason is that the loss function may not be equally smooth in all directions in weight space, and there is no reason for the sizes of different components of the raw gradient vector to respect this heterogeneity. In other words, the geometry of the loss function may be non-isotropic. Insisting on a type check should force the user to become cognizant of this issue and to find a suitable duality map. A good duality map should adjust the size and direction of the gradient to respect the smoothness structure of the loss function.

Duality maps on vector spaces are commonplace in physics and applied math. Examples include the musical isomorphism in differential geometry (Grosse, [2022](#bib.bib17)), raising and lowering indices in general relativity (Carroll, [2019](#bib.bib9)) and the bra-ket notation in quantum mechanics (Sakurai & Napolitano, [2020](#bib.bib30)). Duality maps are also central to several optimization theories including mirror descent (Nemirovsky & Yudin, [1983](#bib.bib28)), natural gradient descent (Amari, [2016](#bib.bib1)) and steepest descent on a normed space (Boyd & Vandenberghe, [2004](#bib.bib5)). Despite the efforts of some prescient papers (Carlson et al., [2015b](#bib.bib8); Flynn, [2017](#bib.bib14)), the latter kind of duality map involving normed vector spaces is yet to puncture the deep learning mainstream.

We believe that duality is a key theoretical concept that will help in building performant large-scale machine learning systems. To support this belief, we show in this paper that two important and seemingly disparate methods in contemporary optimization research may be seen as approximations to a single duality map. These methods are maximal update parameterization (Yang & Hu, [2021](#bib.bib36)), which is aimed at scalable training, and Shampoo (Shi et al., [2023](#bib.bib31)), which is targeted at fast training. We show in [Section 4.1](#S4.SS1 "4.1 Duality Maps for Atomic Modules ‣ 4 Modular Dualization ‣ Modular Duality in Deep Learning") that both methods emerge as partial approximations to a single duality map induced by the RMS–RMS operator norm.

The main contribution of this paper is to describe a procedure for constructing duality maps for general neural architectures. The procedure, which we call modular dualization, works in three steps:

* Step 1:

  Operator norms are assigned to individual layers based on the input-output semantics of each layer;
* Step 2:

  Based on these operator norms, duality maps are constructed for individual layers;
* Step 3:

  Given the layerwise duality maps and the structure of the neural architecture, a single duality map is recursively induced on the full weight space of the architecture.

To instantiate this procedure for a rich family of neural architectures—including convolutional networks and transformers—we write down duality maps for 𝖫𝗂𝗇𝖾𝖺𝗋𝖫𝗂𝗇𝖾𝖺𝗋\mathsf{Linear}, 𝖤𝗆𝖻𝖾𝖽𝖤𝗆𝖻𝖾𝖽\mathsf{Embed} and 𝖢𝗈𝗇𝗏𝟤𝖣𝖢𝗈𝗇𝗏𝟤𝖣\mathsf{Conv2D} layers. We also provide novel, GPU-friendly algorithms for computing these duality maps. Overall, we hope that modular dualization will help in the principled design of the machine learning systems of the future.

## 2 Related Work

This paper constructs a duality map for general neural architectures. Our approach is based on assigning operator norms to individual network layers and using these layerwise norms to recursively induce a duality map on the full neural architecture. The most closely related prior work is a series of papers on spectral descent (Carlson et al., [2015a](#bib.bib6); [b](#bib.bib8); [2016](#bib.bib7)) and a paper on duality structure gradient descent (Flynn, [2017](#bib.bib14)).

Spectral descent has been applied to restricted Boltzmann machines (Carlson et al., [2015a](#bib.bib6)) and discrete graphical models (Carlson et al., [2016](#bib.bib7)), but let us focus on the more closely related paper on spectral descent for deep learning (Carlson et al., [2015b](#bib.bib8)). In that paper, the authors propose assigning the Schatten-∞\infty norm (a.k.a. spectral norm) to individual linear layers. This assignment is based on the observation that neural networks admit natural majorization bounds in the Schatten-∞\infty norm. The authors call the corresponding duality map for linear layers the “#-operator”—a name presumably inspired by the musical isomorphism (Grosse, [2022](#bib.bib17)). The authors propose a cheap approximation to the #-operator based on sketching (Martinsson & Tropp, [2020](#bib.bib27)), and they also propose a way to mix RMSprop-style pre-conditioning information (Tieleman & Hinton, [2012](#bib.bib34)) into the weight updates. In contrast to our work, the authors only derive duality maps for single linear layers, and these maps are then heuristically extended to all-layer updates. Nonetheless, the authors achieve substantial wallclock speedups using variants of spectral descent to train small networks.

Now, let us turn our attention to duality structure gradient descent (Flynn, [2017](#bib.bib14)), which constructs a duality map on the full weight space of the neural architecture based on identifying a Finsler structure (Deimling, [1985](#bib.bib11)) inherent to neural networks. Similar to modular dualization, Flynn ([2017](#bib.bib14))’s duality map works by assigning duality maps to each layer and then inducing a duality map on the full weight space. The substantial difference to our approach is that Flynn ([2017](#bib.bib14)) leverages a weighted sum (L1subscript𝐿1L\_{1} combination) of layerwise norms to construct his full duality map. This leads to optimization methods that only update a single layer at each iteration, and the methods need to be heuristically extended to achieve all-layer updates. In contrast, we leverage the modular norm (Large et al., [2024](#bib.bib25)), which takes a weighted max (L∞subscript𝐿L\_{\infty} combination) of layerwise norms. In turn, our duality map leads directly to more conventional
all-layer optimizers.

Another important difference between our work on modular duality and prior work on duality structure gradient descent is that we fully “modularize” our theory—meaning that our construction is explicitly recursive—and as such it is easy to code up into a software package. In this regard, we are inspired by a line of work that attempts to build optimization algorithms that automatically adapt to the structure of general computation graphs. The earliest work we know of in this category is the PhD thesis of Grant ([2004](#bib.bib16)) on disciplined convex programming, which aims to infer the convexity properties of general functions by breaking them up into subexpressions and applying composition theorems from convex analysis. More recent progress in this vein includes work on universal majorization-minimization algorithms (Streeter & Dillon, [2022](#bib.bib33); Streeter, [2023](#bib.bib32)) and related papers on automatic majorization (Tran et al., [2015](#bib.bib35); Bernstein et al., [2023](#bib.bib4)).

## 3 Theoretical Preliminaries

In this section, we introduce duality maps, a means of constructing duality maps based on norms, and finally a norm called the modular norm that is well-suited to describe the geometry of general neural architectures.

### 3.1 Duality Maps

Given a vector space 𝒱𝒱\mathcal{V}, we say that a function f:𝒱→ℝ:𝑓→𝒱ℝf:\mathcal{V}\to\mathbb{R} is a linear functional on 𝒱𝒱\mathcal{V} if f𝑓f is linear. We define the dual space 𝒱∗superscript𝒱\mathcal{V}^{\*} to be the set of linear functionals on the vector space 𝒱𝒱\mathcal{V}. The dual space is itself a vector space provided that addition is defined pointwise (f+g)​(x):=f​(x)+g​(x)assign𝑓𝑔𝑥𝑓𝑥𝑔𝑥(f+g)(x)\vcentcolon=f(x)+g(x) and scalar multiplication is defined pointwise (α​f)​(x):=α​f​(x)assign𝛼𝑓𝑥𝛼𝑓𝑥(\alpha f)(x)\vcentcolon=\alpha f(x) for any scalar α𝛼\alpha. By duality map we simply mean any rule for identifying members of the dual vector space 𝒱∗superscript𝒱\mathcal{V}^{\*} with members of the primal vector space 𝒱𝒱\mathcal{V}, or potentially vice versa.

Let ℒ:𝒲→ℝ:ℒ→𝒲ℝ\mathcal{L}:\mathcal{W}\to\mathbb{R} denote the loss of a differentiable machine learning model with weight space 𝒲=ℝn𝒲superscriptℝ𝑛\mathcal{W}=\mathbb{R}^{n}. The Taylor expansion of the loss at weight setting 𝒘∈𝒲𝒘𝒲{\bm{w}}\in\mathcal{W} is given by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ​(𝒘+Δ​𝒘)=ℒ​(𝒘)+∇𝒘ℒ​(𝒘)⊤​Δ​𝒘+higher-order terms.ℒ𝒘Δ𝒘ℒ𝒘subscript∇𝒘ℒsuperscript𝒘topΔ𝒘higher-order terms\mathcal{L}({\bm{w}}+\Delta{\bm{w}})=\mathcal{L}({\bm{w}})+\nabla\_{\bm{w}}\mathcal{L}({\bm{w}})^{\top}\Delta{\bm{w}}+\text{higher-order terms}. |  | (3) |

Observe that, in the first-order term, the gradient ∇𝒘ℒ​(𝒘)subscript∇𝒘ℒ𝒘\nabla\_{\bm{w}}\mathcal{L}({\bm{w}}) is acting as a linear functional: it is pairing with the weight vector Δ​𝒘∈𝒲Δ𝒘𝒲\Delta{\bm{w}}\in\mathcal{W} in a linear way to produce a real number. As such, we shall say that the gradient belongs to the dual weight space: ∇𝒘ℒ​(𝒘)∈𝒲∗subscript∇𝒘ℒ𝒘superscript𝒲\nabla\_{\bm{w}}\mathcal{L}({\bm{w}})\in\mathcal{W}^{\*}. We shall forbid ourselves from directly subtracting a member of the dual weight space 𝒲∗superscript𝒲\mathcal{W}^{\*} from the weight space 𝒲𝒲\mathcal{W}. If we would like to conduct a gradient descent update, then we had better find a duality map to send the gradient back to the primal space 𝒲𝒲\mathcal{W}.

This restriction may seem absurd! After all, here the weight space 𝒲𝒲\mathcal{W} and its dual 𝒲∗superscript𝒲\mathcal{W}^{\*} are both just ℝnsuperscriptℝ𝑛\mathbb{R}^{n}. However, insisting upon this type check serves to remind us that the curvature of the loss function may be highly heterogeneous. The next section will show one way to construct duality maps to account for this.

### 3.2 Steepest Descent on a Normed Space

Suppose that we have found a norm ∥⋅∥:𝒲→ℝ\|{\cdot}\|:\mathcal{W}\to\mathbb{R} and a sharpness parameter λ>0𝜆0\lambda>0 that serve as a good model of the higher-order terms in the Taylor expansion of the loss function given in [Equation 3](#S3.E3 "In 3.1 Duality Maps ‣ 3 Theoretical Preliminaries ‣ Modular Duality in Deep Learning"):

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ​(𝒘+Δ​𝒘)≈ℒ​(𝒘)+∇𝒘ℒ​(𝒘)⊤​Δ​𝒘+λ2⋅‖Δ​𝒘‖2.ℒ𝒘Δ𝒘ℒ𝒘subscript∇𝒘ℒsuperscript𝒘topΔ𝒘⋅𝜆2superscriptnormΔ𝒘2\mathcal{L}({\bm{w}}+\Delta{\bm{w}})\approx\mathcal{L}({\bm{w}})+\nabla\_{\bm{w}}\mathcal{L}({\bm{w}})^{\top}\Delta{\bm{w}}+\frac{\lambda}{2}\cdot\|{\Delta{\bm{w}}}\|^{2}. |  | (4) |

In other words, the norm provides a good characterization of the heterogeneity in curvature of the loss function. Then it makes sense to solve for a weight update Δ​𝒘Δ𝒘\Delta{\bm{w}} by minimizing the right-hand side of [Equation 4](#S3.E4 "In 3.2 Steepest Descent on a Normed Space ‣ 3 Theoretical Preliminaries ‣ Modular Duality in Deep Learning"). We will show that the minimizer can be expressed in terms of a dual norm and a duality map:

###### Definition 1 (Dual norm).

Given a norm ∥⋅∥:ℝn→ℝ\|{\cdot}\|:\mathbb{R}^{n}\to\mathbb{R}, the dual norm ∥⋅∥†\|{\cdot}\|^{\dagger} of a vector 𝐠∈ℝn𝐠superscriptℝ𝑛{\bm{g}}\in\mathbb{R}^{n} is given by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖𝒈‖†:=max𝒕∈ℝn:‖𝒕‖=1⁡𝒈⊤​𝒕.assignsuperscriptnorm𝒈†subscript:𝒕superscriptℝ𝑛norm𝒕1superscript𝒈top𝒕\|{{\bm{g}}}\|^{\dagger}\vcentcolon=\max\_{{\bm{t}}\in\mathbb{R}^{n}:\|{{\bm{t}}}\|=1}{\bm{g}}^{\top}{\bm{t}}. |  | (5) |

###### Definition 2 (Duality map based on a norm).

Given a norm ∥⋅∥:ℝn→ℝ\|{\cdot}\|:\mathbb{R}^{n}\to\mathbb{R}, we consider the duality map:

|  |  |  |  |
| --- | --- | --- | --- |
|  | dualize∥⋅∥⁡𝒈:=arg​max𝒕∈ℝn:‖𝒕‖=1⁡𝒈⊤​𝒕,\operatorname{dualize}\_{\|{\cdot}\|}{\bm{g}}\vcentcolon=\operatorname\*{arg\,max}\_{{\bm{t}}\in\mathbb{R}^{n}:\|{{\bm{t}}}\|=1}{\bm{g}}^{\top}{\bm{t}}, |  | (6) |

where, if the arg​maxargmax\operatorname\*{arg\,max} is not unique, dualize∥⋅∥\operatorname{dualize}\_{\|{\cdot}\|} returns any maximizer.

Given these definitions, minimizing the expression in the right-hand side of [Equation 4](#S3.E4 "In 3.2 Steepest Descent on a Normed Space ‣ 3 Theoretical Preliminaries ‣ Modular Duality in Deep Learning") can be done using the following standard proposition, for which Bernstein & Newhouse ([2024](#bib.bib3)) provide a proof:

###### Proposition 1 (Steepest descent under a norm).

For any 𝐠∈ℝn𝐠superscriptℝ𝑛{\bm{g}}\in\mathbb{R}^{n} thought of as “the gradient”, any λ≥0𝜆0\lambda\geq 0 thought of as “the sharpness”, and any norm ∥⋅∥:ℝn→ℝ\|{\cdot}\|:\mathbb{R}^{n}\to\mathbb{R} with dual norm ∥⋅∥†\|{\cdot}\|^{\dagger} and duality map dualize∥⋅∥\operatorname{dualize}\_{\|{\cdot}\|}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | arg​minΔ​𝒘∈ℝn⁡[𝒈⊤​Δ​𝒘+λ2​‖Δ​𝒘‖2]=−‖𝒈‖†λ×dualize∥⋅∥⁡𝒈.\displaystyle\operatorname\*{arg\,min}\_{\Delta{\bm{w}}\in\mathbb{R}^{n}}\left[{\bm{g}}^{\top}\Delta{\bm{w}}+\frac{\lambda}{2}\,\|{\Delta{\bm{w}}}\|^{2}\right]=-\frac{\|{{\bm{g}}}\|^{\dagger}}{\lambda}\times\operatorname{dualize}\_{\|{\cdot}\|}{\bm{g}}. |  | (7) |

In words: to find the minimizer of a linear term penalized by a squared norm, we need only evaluate the dual norm and a duality map. In this paper, we focus on constructing a duality map for the modular norm, which is defined on general neural architectures. The next section reviews duality maps for more standard norms.

### 3.3 Basic Norms and Duality Maps

Many basic norms and duality maps are already covered in prior work (Carlson et al., [2016](#bib.bib7); [2015a](#bib.bib6); [2015b](#bib.bib8); Flynn, [2017](#bib.bib14)). For some warmup examples, the following duality maps for vector norms are standard:

###### Example 1 (Duality map for the Euclidean norm).

For a vector 𝐠∈ℝd𝐠superscriptℝ𝑑{\bm{g}}\in\mathbb{R}^{d}, we have dualize∥⋅∥2⁡𝐠=𝐠/‖𝐠‖2\operatorname{dualize}\_{\|{\cdot}\|\_{2}}{\bm{g}}={\bm{g}}/\|{{\bm{g}}}\|\_{2}.

###### Example 2 (Duality map for the infinity norm).

For a vector 𝐠∈ℝd𝐠superscriptℝ𝑑{\bm{g}}\in\mathbb{R}^{d}, we have dualize∥⋅∥∞⁡𝐠=sign⁡(𝐠)\smash{\operatorname{dualize}\_{\|{\cdot}\|\_{\infty}}{\bm{g}}=\operatorname{sign}({\bm{g}})}, where the sign function is applied entrywise and we are free to take sign⁡(0)=0sign00\operatorname{sign}(0)=0.

In neural networks, the weight spaces of individual layers tend to have matrix structure. And layers with the same shape weight matrix may have semantically different input and output spaces—think embedding versus linear layers in a transformer. As such, we will need duality maps for different induced operator norms:

###### Definition 3 (Induced operator norm).

Given a matrix 𝐌∈ℝdout×din𝐌superscriptℝsubscript𝑑outsubscript𝑑in{\bm{M}}\in\mathbb{R}^{d\_{\mathrm{out}}\times d\_{\mathrm{in}}} and two normed vector spaces (ℝdin,∥⋅∥α)(\mathbb{R}^{d\_{\mathrm{in}}},\|{\cdot}\|\_{\alpha}) and (ℝdout,∥⋅∥β)(\mathbb{R}^{d\_{\mathrm{out}}},\|{\cdot}\|\_{\beta}), the “α𝛼\alpha to β𝛽\beta” induced operator norm is given by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖𝑴‖α→β=max𝒙∈ℝdin⁡‖𝑴​𝒙‖β‖𝒙‖α.subscriptnorm𝑴→𝛼𝛽subscript  𝒙superscriptℝsubscript𝑑insubscriptnorm𝑴𝒙𝛽subscriptnorm𝒙𝛼\|{{\bm{M}}}\|\_{\alpha\to\beta}=\max\_{\begin{subarray}{c}{\bm{x}}\in\mathbb{R}^{d\_{\mathrm{in}}}\end{subarray}}\frac{\|{{\bm{M}}{\bm{x}}}\|\_{\beta}}{\|{{\bm{x}}}\|\_{\alpha}}. |  | (8) |

For tensors, we define the duality map via dualize∥⋅∥𝑮:=arg​max‖𝑻‖=1flatten(𝑮)⊤flatten(𝑻)\operatorname{dualize}\_{\|{\cdot}\|}{\bm{G}}\vcentcolon=\operatorname\*{arg\,max}\_{\|{{\bm{T}}}\|=1}\operatorname{flatten}({\bm{G}})^{\top}\operatorname{flatten}({\bm{T}}). For linear layers, we will need the duality map for the RMS→RMS→RMSRMS\mathrm{RMS}\to\mathrm{RMS} induced operator norm. This ends up as a rescaled version of the spectral norm duality map from prior work (Carlson et al., [2015b](#bib.bib8); Flynn, [2017](#bib.bib14)).

###### Example 3 (Duality map for the RMS→RMS→RMSRMS\mathrm{RMS}\to\mathrm{RMS} operator norm).

For a vector 𝐯∈ℝd𝐯superscriptℝ𝑑{\bm{v}}\in\mathbb{R}^{d}, we define the RMS norm to be the normalized Euclidean norm: ‖𝐯‖RMS=‖𝐯‖2/dsubscriptnorm𝐯RMSsubscriptnorm𝐯2𝑑\|{{\bm{v}}}\|\_{\mathrm{RMS}}=\|{{\bm{v}}}\|\_{2}/\sqrt{d}. Given a matrix 𝐖∈ℝdout×din𝐖superscriptℝsubscript𝑑outsubscript𝑑in{\bm{W}}\in\mathbb{R}^{d\_{\mathrm{out}}\times d\_{\mathrm{in}}}, the RMS→RMS→RMSRMS\mathrm{RMS}\to\mathrm{RMS} induced operator norm resolves to a rescaled spectral norm: ‖𝐖‖RMS→RMS=din/dout×‖𝐖‖∗subscriptnorm𝐖→RMSRMSsubscript𝑑insubscript𝑑outsubscriptnorm𝐖\|{{\bm{W}}}\|\_{\mathrm{RMS}\to\mathrm{RMS}}=\sqrt{d\_{\mathrm{in}}/d\_{\mathrm{out}}}\times\|{{\bm{W}}}\|\_{\*}, where ∥⋅∥∗\|{\cdot}\|\_{\*} denotes the standard spectral norm. For a matrix 𝐆∈ℝdout×din𝐆superscriptℝsubscript𝑑outsubscript𝑑in{\bm{G}}\in\mathbb{R}^{d\_{\mathrm{out}}\times d\_{\mathrm{in}}} with reduced singular value decomposition 𝐆=𝐔​𝚺​𝐕⊤𝐆𝐔𝚺superscript𝐕top{\bm{G}}={\bm{U}}{\bm{\Sigma}}{\bm{V}}^{\top}, the corresponding duality map is given by dualize∥⋅∥RMS→RMS⁡𝐆=dout/din×𝐔​𝐕⊤.\operatorname{dualize}\_{\|{\cdot}\|\_{\mathrm{RMS}\to\mathrm{RMS}}}{\bm{G}}=\sqrt{d\_{\mathrm{out}}/d\_{\mathrm{in}}}\times{\bm{U}}{\bm{V}}^{\top}.

And for embedding layers, we will need the duality map for the ℓ1→RMS→subscriptℓ1RMS\ell\_{1}\to\mathrm{RMS} operator norm:

###### Example 4 (Duality map for the ℓ1→RMS→subscriptℓ1RMS\ell\_{1}\to\mathrm{RMS} operator norm).

Given a matrix 𝐖∈ℝdout×din𝐖superscriptℝsubscript𝑑outsubscript𝑑in{\bm{W}}\in\mathbb{R}^{d\_{\mathrm{out}}\times d\_{\mathrm{in}}}, the ℓ1→RMS→subscriptℓ1RMS\ell\_{1}\to\mathrm{RMS} induced operator norm resolves to the max RMSRMS\mathrm{RMS} norm of the columns: ‖𝐖‖ℓ1→RMS=maxi⁡‖coli​(𝐖)‖RMSsubscriptnorm𝐖→subscriptℓ1RMSsubscript𝑖subscriptnormsubscriptcol𝑖𝐖RMS\|{{\bm{W}}}\|\_{\ell\_{1}\to\mathrm{RMS}}=\max\_{i}\|{\mathrm{col}\_{i}({\bm{W}})}\|\_{\mathrm{RMS}}. For a matrix 𝐆∈ℝdout×din𝐆superscriptℝsubscript𝑑outsubscript𝑑in{\bm{G}}\in\mathbb{R}^{d\_{\mathrm{out}}\times d\_{\mathrm{in}}}, the corresponding duality map dualize∥⋅∥ℓ1→RMS⁡𝐆\operatorname{dualize}\_{\|{\cdot}\|\_{\ell\_{1}\to\mathrm{RMS}}}{\bm{G}} simply normalizes each column of 𝐆𝐆{\bm{G}} to have unit RMS norm: coli​(𝐆)↦coli​(𝐆)/‖coli​(𝐆)‖RMSmaps-tosubscriptcol𝑖𝐆subscriptcol𝑖𝐆subscriptnormsubscriptcol𝑖𝐆RMS\mathrm{col}\_{i}({\bm{G}})\mapsto\mathrm{col}\_{i}({\bm{G}})/\|{\mathrm{col}\_{i}({\bm{G}})}\|\_{\mathrm{RMS}} for each i=1,…,din𝑖

1…subscript𝑑ini=1,...,d\_{\mathrm{in}}.

### 3.4 The Modular Norm

The modular norm (Large et al., [2024](#bib.bib25)) is intended to help characterize the heterogeneous curvature of general neural architectures. The construction first defines an abstract module type along with a notion of what is a good, or well-normed, module. Then combination rules are given for constructing new well-normed modules from a library of existing well-normed modules. So modules are a special case of combinator pattern from functional programming (Haskell Wiki Contributors, [2007](#bib.bib19)). Modules are also related to the monoidal category from category theory (Fong & Spivak, [2019](#bib.bib15)). We begin by defining the abstract notion of a module:

###### Definition 4 (Module).

Given input vector space 𝒳𝒳\mathcal{X}, output vector space 𝒴𝒴\mathcal{Y} and weight vector space 𝒲𝒲\mathcal{W}, a module 𝖬𝖬\mathsf{M} is an object with the following four attributes:

1. (a)

   a function, 𝖬.𝖿𝗈𝗋𝗐𝖺𝗋𝖽:𝒲×𝒳→𝒴\mathsf{M}\mathsf{.forward}:\mathcal{W}\times\mathcal{X}\to\mathcal{Y}, which maps an input and a weight vector to an output;
2. (b)

   a number, 𝖬.𝗆𝖺𝗌𝗌≥0formulae-sequence𝖬𝗆𝖺𝗌𝗌0\mathsf{M}\mathsf{.mass}\geq 0, which is used to set the proportion of feature learning that this module contributes to any supermodule;
3. (c)

   a number, 𝖬.𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒≥0formulae-sequence𝖬𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒0\mathsf{M}\mathsf{.sensitivity}\geq 0, which estimates the module’s sensitivity to input perturbations;
4. (d)

   a norm over the weight space, 𝖬.𝗇𝗈𝗋𝗆:𝒲→ℝ≥0\mathsf{M}\mathsf{.norm}:\mathcal{W}\to\mathbb{R}\_{\geq 0}, sometimes abbreviated to just ∥⋅∥𝖬\|{\cdot}\|\_{\mathsf{M}}.

We shall care most about modules that are well-normed, which amounts to requiring that the forward function is Lipschitz-continuous in the weights with constant 1 and in the inputs with constant 𝖬.𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒formulae-sequence𝖬𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒\mathsf{M}\mathsf{.sensitivity}:

###### Definition 5 (Well-normed module).

Let 𝖬𝖬\mathsf{M} be a module on (𝒳,𝒴,𝒲)𝒳𝒴𝒲(\mathcal{X},\mathcal{Y},\mathcal{W}), where the input and output spaces have respective norms ∥⋅∥𝒳\|{\cdot}\|\_{\mathcal{X}} and ∥⋅∥𝒴\|{\cdot}\|\_{\mathcal{Y}}. 𝖬𝖬\mathsf{M} is well-normed if for all inputs 𝐱∈𝒳𝐱𝒳{\bm{x}}\in\mathcal{X} and weights 𝐰∈𝒲𝐰𝒲{\bm{w}}\in\mathcal{W}:

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | ∥∇𝒘𝖬.𝖿𝗈𝗋𝗐𝖺𝗋𝖽(𝒘,𝒙)⋄Δ𝒘∥𝒴\displaystyle\|{\nabla\_{\bm{w}}\mathsf{M}\mathsf{.forward}({\bm{w}},{\bm{x}})\diamond\Delta{\bm{w}}}\|\_{\mathcal{Y}} | ≤𝖬.𝗇𝗈𝗋𝗆​(Δ​𝒘)formulae-sequenceabsent𝖬𝗇𝗈𝗋𝗆Δ𝒘\displaystyle\leq\mathsf{M}\mathsf{.norm}(\Delta{\bm{w}}) | for all ​Δ​𝒘∈𝒲;for all Δ𝒘𝒲\displaystyle\text{for all }\Delta{\bm{w}}\in\mathcal{W}; |  | (9) |
|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | ∥∇𝒙𝖬.𝖿𝗈𝗋𝗐𝖺𝗋𝖽(𝒘,𝒙)⋄Δ𝒙∥𝒴\displaystyle\|{\nabla\_{\bm{x}}\mathsf{M}\mathsf{.forward}({\bm{w}},{\bm{x}})\diamond\Delta{\bm{x}}}\|\_{\mathcal{Y}} | ≤𝖬.𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒∗‖Δ​𝒙‖𝒳formulae-sequenceabsent𝖬𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒subscriptnormΔ𝒙𝒳\displaystyle\leq\mathsf{M}\mathsf{.sensitivity}\*\|{\Delta{\bm{x}}}\|\_{\mathcal{X}} | for all ​Δ​𝒙∈𝒳.for all Δ𝒙𝒳\displaystyle\text{for all }\Delta{\bm{x}}\in\mathcal{X}. |  | (10) |

The ⋄⋄\diamond operator denotes summation over any shared tensor indices. This definition of well-normed-ness can be used as a guiding principle in the design of a library of atomic (i.e. handwritten) modules. First, norms should be assigned to the input and output space of each module based on the semantics of 𝖬.𝖿𝗈𝗋𝗐𝖺𝗋𝖽formulae-sequence𝖬𝖿𝗈𝗋𝗐𝖺𝗋𝖽\mathsf{M}\mathsf{.forward}. Then a norm 𝖬.𝗇𝗈𝗋𝗆formulae-sequence𝖬𝗇𝗈𝗋𝗆\mathsf{M}\mathsf{.norm} should be assigned to the module’s weight space and a number 𝖬.𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒formulae-sequence𝖬𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒\mathsf{M}\mathsf{.sensitivity} should be chosen to make the module well-normed. Examples are given in [Section 4.1](#S4.SS1 "4.1 Duality Maps for Atomic Modules ‣ 4 Modular Dualization ‣ Modular Duality in Deep Learning").

Given such a library of well-normed atomic modules, a compound module built through any arbitrary sequence of module compositions and module concatenations is automatically well-normed (Large et al., [2024](#bib.bib25)). And if the atomic modules in the library are not only well-normed but are also smooth in an appropriate sense, then Large et al. ([2024](#bib.bib25)) give an automatic procedure for computing sharpness coefficients for any compound module built from the library. The relevant definition of module composition is as follows:

###### Definition 6 (Module composition).

Consider module 𝖬1subscript𝖬1\mathsf{M}\_{1} with input, output and weight space (𝒳1,𝒴1,𝒲1)subscript𝒳1subscript𝒴1subscript𝒲1(\mathcal{X}\_{1},\mathcal{Y}\_{1},\mathcal{W}\_{1}) and module 𝖬2subscript𝖬2\mathsf{M}\_{2} with input, output and weight space (𝒳2,𝒴2,𝒲2)subscript𝒳2subscript𝒴2subscript𝒲2(\mathcal{X}\_{2},\mathcal{Y}\_{2},\mathcal{W}\_{2}). 𝖬1subscript𝖬1\mathsf{M}\_{1} and 𝖬2subscript𝖬2\mathsf{M}\_{2} are composable if 𝒳2=𝒴1subscript𝒳2subscript𝒴1\mathcal{X}\_{2}=\mathcal{Y}\_{1}. Their composite module 𝖬=𝖬2∘𝖬1𝖬subscript𝖬2subscript𝖬1\mathsf{M}=\mathsf{M}\_{2}\circ\mathsf{M}\_{1} has input, output and weight space (𝒳1,𝒴2,𝒲1×𝒲2)subscript𝒳1subscript𝒴2subscript𝒲1subscript𝒲2(\mathcal{X}\_{1},\mathcal{Y}\_{2},\mathcal{W}\_{1}\times\mathcal{W}\_{2}) and attributes:

1. (a)

   𝖬.𝖿𝗈𝗋𝗐𝖺𝗋𝖽((𝒘1,𝒘2),𝒙))=𝖬2.𝖿𝗈𝗋𝗐𝖺𝗋𝖽(𝒘2,𝖬1.𝖿𝗈𝗋𝗐𝖺𝗋𝖽(𝒘1,𝒙))\mathsf{M}\mathsf{.forward}(({\bm{w}}\_{1},{\bm{w}}\_{2}),{\bm{x}}))=\mathsf{M}\_{2}\mathsf{.forward}({\bm{w}}\_{2},\mathsf{M}\_{1}\mathsf{.forward}({\bm{w}}\_{1},{\bm{x}}));
2. (b)

   𝖬.𝗆𝖺𝗌𝗌=𝖬1.𝗆𝖺𝗌𝗌+𝖬2.𝗆𝖺𝗌𝗌formulae-sequence𝖬𝗆𝖺𝗌𝗌subscript𝖬1𝗆𝖺𝗌𝗌subscript𝖬2𝗆𝖺𝗌𝗌\mathsf{M}\mathsf{.mass}=\mathsf{M}\_{1}\mathsf{.mass}+\mathsf{M}\_{2}\mathsf{.mass};
3. (c)

   𝖬.𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒=𝖬1.𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒∗𝖬2.𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒formulae-sequence𝖬𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒subscript𝖬1𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒subscript𝖬2𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒\mathsf{M}\mathsf{.sensitivity}=\mathsf{M}\_{1}\mathsf{.sensitivity}\*\mathsf{M}\_{2}\mathsf{.sensitivity};
4. (d)

   𝖬.𝗇𝗈𝗋𝗆​((𝒘1,𝒘2))formulae-sequence𝖬𝗇𝗈𝗋𝗆subscript𝒘1subscript𝒘2\mathsf{M}\mathsf{.norm}(({\bm{w}}\_{1},{\bm{w}}\_{2})) given by:

   |  |  |  |
   | --- | --- | --- |
   |  | max(𝖬2.𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒∗𝖬.𝗆𝖺𝗌𝗌𝖬1.𝗆𝖺𝗌𝗌∗𝖬1.𝗇𝗈𝗋𝗆(𝒘1),𝖬.𝗆𝖺𝗌𝗌𝖬2.𝗆𝖺𝗌𝗌∗𝖬2.𝗇𝗈𝗋𝗆(𝒘2)),\max\left(\mathsf{M}\_{2}\mathsf{.sensitivity}\*\frac{\mathsf{M}\mathsf{.mass}}{\mathsf{M}\_{1}\mathsf{.mass}}\*\mathsf{M}\_{1}\mathsf{.norm}({\bm{w}}\_{1}),\frac{\mathsf{M}\mathsf{.mass}}{\mathsf{M}\_{2}\mathsf{.mass}}\*\mathsf{M}\_{2}\mathsf{.norm}({\bm{w}}\_{2})\right), |  |

   where if 𝖬1.𝗆𝖺𝗌𝗌formulae-sequencesubscript𝖬1𝗆𝖺𝗌𝗌\mathsf{M}\_{1}\mathsf{.mass} or 𝖬2.𝗆𝖺𝗌𝗌formulae-sequencesubscript𝖬2𝗆𝖺𝗌𝗌\mathsf{M}\_{2}\mathsf{.mass} is zero, the corresponding term in the max\max is set to zero.

So the composite norm is taken to be a weighted max over the norms of the two sub-modules, where the weight space of the first module is coupled to the input sensitivity of the second module. The module masses provide freedom to tune the importance of each sub-module in the norm, and Large et al. ([2024](#bib.bib25)) prove that module mass provides precise control over the amount of feature learning that can happen in each sub-module.

Module concatenation is defined in a similar way to module composition:

###### Definition 7 (Module concatenation).

Consider module 𝖬1subscript𝖬1\mathsf{M}\_{1} with input, output and weight space (𝒳1,𝒴1,𝒲1)subscript𝒳1subscript𝒴1subscript𝒲1(\mathcal{X}\_{1},\mathcal{Y}\_{1},\mathcal{W}\_{1}) and module 𝖬2subscript𝖬2\mathsf{M}\_{2} with input, output and weight space (𝒳2,𝒴2,𝒲2)subscript𝒳2subscript𝒴2subscript𝒲2(\mathcal{X}\_{2},\mathcal{Y}\_{2},\mathcal{W}\_{2}). We say that 𝖬1subscript𝖬1\mathsf{M}\_{1} and 𝖬2subscript𝖬2\mathsf{M}\_{2} are concatenatable if their input spaces match: 𝒳1=𝒳2subscript𝒳1subscript𝒳2\mathcal{X}\_{1}=\mathcal{X}\_{2}. The tuple module 𝖬=(𝖬1,𝖬2)𝖬subscript𝖬1subscript𝖬2\mathsf{M}=(\mathsf{M}\_{1},\mathsf{M}\_{2}) has input, output and weight space (𝒳1,𝒴1×𝒴2,𝒲1×𝒲2)subscript𝒳1subscript𝒴1subscript𝒴2subscript𝒲1subscript𝒲2(\mathcal{X}\_{1},\mathcal{Y}\_{1}\times\mathcal{Y}\_{2},\mathcal{W}\_{1}\times\mathcal{W}\_{2}) and the following list of attributes:

1. (a)

   𝖬.𝖿𝗈𝗋𝗐𝖺𝗋𝖽((𝒘1,𝒘2),𝒙))=(𝖬1.𝖿𝗈𝗋𝗐𝖺𝗋𝖽(𝒘1,𝒙),𝖬2.𝖿𝗈𝗋𝗐𝖺𝗋𝖽(𝒘2,𝒙))\mathsf{M}\mathsf{.forward}(({\bm{w}}\_{1},{\bm{w}}\_{2}),{\bm{x}}))=(\mathsf{M}\_{1}\mathsf{.forward}({\bm{w}}\_{1},{\bm{x}}),\mathsf{M}\_{2}\mathsf{.forward}({\bm{w}}\_{2},{\bm{x}}));
2. (b)

   𝖬.𝗆𝖺𝗌𝗌=𝖬1.𝗆𝖺𝗌𝗌+𝖬2.𝗆𝖺𝗌𝗌formulae-sequence𝖬𝗆𝖺𝗌𝗌subscript𝖬1𝗆𝖺𝗌𝗌subscript𝖬2𝗆𝖺𝗌𝗌\mathsf{M}\mathsf{.mass}=\mathsf{M}\_{1}\mathsf{.mass}+\mathsf{M}\_{2}\mathsf{.mass};
3. (c)

   𝖬.𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒=𝖬1.𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒+𝖬2.𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒formulae-sequence𝖬𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒subscript𝖬1𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒subscript𝖬2𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒\mathsf{M}\mathsf{.sensitivity}=\mathsf{M}\_{1}\mathsf{.sensitivity}+\mathsf{M}\_{2}\mathsf{.sensitivity};
4. (d)

   𝖬.𝗇𝗈𝗋𝗆​(𝒘1,𝒘2)formulae-sequence𝖬𝗇𝗈𝗋𝗆subscript𝒘1subscript𝒘2\mathsf{M}\mathsf{.norm}({\bm{w}}\_{1},{\bm{w}}\_{2}) given by:

   |  |  |  |
   | --- | --- | --- |
   |  | max(𝖬.𝗆𝖺𝗌𝗌𝖬1.𝗆𝖺𝗌𝗌∗𝖬1.𝗇𝗈𝗋𝗆(𝒘1),𝖬.𝗆𝖺𝗌𝗌𝖬2.𝗆𝖺𝗌𝗌∗𝖬2.𝗇𝗈𝗋𝗆(𝒘2)),\max\left(\frac{\mathsf{M}\mathsf{.mass}}{\mathsf{M}\_{1}\mathsf{.mass}}\*\mathsf{M}\_{1}\mathsf{.norm}({\bm{w}}\_{1}),\frac{\mathsf{M}\mathsf{.mass}}{\mathsf{M}\_{2}\mathsf{.mass}}\*\mathsf{M}\_{2}\mathsf{.norm}({\bm{w}}\_{2})\right), |  |

   where if 𝖬1.𝗆𝖺𝗌𝗌formulae-sequencesubscript𝖬1𝗆𝖺𝗌𝗌\mathsf{M}\_{1}\mathsf{.mass} or 𝖬2.𝗆𝖺𝗌𝗌formulae-sequencesubscript𝖬2𝗆𝖺𝗌𝗌\mathsf{M}\_{2}\mathsf{.mass} is zero, the corresponding term in the max\max is set to zero.

A shortcoming of the paper by Large et al. ([2024](#bib.bib25)) is that the power of the modular norm is not fully leveraged. In particular, the authors do modular normalization of training, where weight updates to modules are sometimes just naïvely divided by their norm. In this paper we make fuller use of the geometry implied by the modular norm by constructing the corresponding duality map, which we call modular dualization.

## 4 Modular Dualization

In this section, we construct a duality map for general neural architectures. Our strategy is to first write down duality maps for atomic modules, i.e. individual layers. We then extend to arbitrary compound modules, i.e. full neural networks, by showing how duality maps should pass through composition and concatenation.

### 4.1 Duality Maps for Atomic Modules

To construct a duality map for an atomic module 𝖠𝖠\mathsf{A}, the idea is to first fix norms on the input and output spaces that respect the semantics of 𝖠.𝖿𝗈𝗋𝗐𝖺𝗋𝖽formulae-sequence𝖠𝖿𝗈𝗋𝗐𝖺𝗋𝖽\mathsf{A}\mathsf{.forward}. We should select norms that describe both how large we would like the inputs and outputs to be, and in what geometry we would like the outputs to evolve. Then we place a norm on the weight space such that 𝖠𝖠\mathsf{A} is well-normed: this is typically the operator norm ([Definition 3](#Thmmydefinition3 "Definition 3 (Induced operator norm). ‣ 3.3 Basic Norms and Duality Maps ‣ 3 Theoretical Preliminaries ‣ Modular Duality in Deep Learning")) induced by the input and output norms. Finally we are in position to solve for the duality map, which we shall call 𝖠.𝖽𝗎𝖺𝗅𝗂𝗓𝖾formulae-sequence𝖠𝖽𝗎𝖺𝗅𝗂𝗓𝖾\mathsf{A}\mathsf{.dualize}. We now give some examples of this procedure for the basic layer types of 𝖫𝗂𝗇𝖾𝖺𝗋𝖫𝗂𝗇𝖾𝖺𝗋\mathsf{Linear}, 𝖤𝗆𝖻𝖾𝖽𝖤𝗆𝖻𝖾𝖽\mathsf{Embed} and 𝖢𝗈𝗇𝗏𝟤𝖣𝖢𝗈𝗇𝗏𝟤𝖣\mathsf{Conv2D}. The results are summarized in [Table 1](#S4.T1 "In 4.1 Duality Maps for Atomic Modules ‣ 4 Modular Dualization ‣ Modular Duality in Deep Learning").

We start with the canonical example of an atomic module:

###### Example 5 (The 𝖫𝗂𝗇𝖾𝖺𝗋𝖫𝗂𝗇𝖾𝖺𝗋\mathsf{Linear} module).

The 𝖫𝗂𝗇𝖾𝖺𝗋𝖫𝗂𝗇𝖾𝖺𝗋\mathsf{Linear} module sends inputs from 𝒳=ℝdin𝒳superscriptℝsubscript𝑑in\mathcal{X}=\mathbb{R}^{d\_{\mathrm{in}}} to outputs in 𝒴=ℝdout𝒴superscriptℝsubscript𝑑out\mathcal{Y}=\mathbb{R}^{d\_{\mathrm{out}}}. The weight space is given by the matrix space 𝒲=ℝdout×din𝒲superscriptℝsubscript𝑑outsubscript𝑑in\mathcal{W}=\mathbb{R}^{d\_{\mathrm{out}}\times d\_{\mathrm{in}}}. We endow the 𝖫𝗂𝗇𝖾𝖺𝗋𝖫𝗂𝗇𝖾𝖺𝗋\mathsf{Linear} module with attributes:

1. 1.

   𝖫𝗂𝗇𝖾𝖺𝗋.𝖿𝗈𝗋𝗐𝖺𝗋𝖽​(𝑾,𝒙)=𝑾​𝒙formulae-sequence𝖫𝗂𝗇𝖾𝖺𝗋𝖿𝗈𝗋𝗐𝖺𝗋𝖽𝑾𝒙𝑾𝒙\mathsf{Linear}\mathsf{.forward}({\bm{W}},{\bm{x}})={\bm{W}}{\bm{x}}, the matrix-vector product;
2. 2.

   𝖫𝗂𝗇𝖾𝖺𝗋.𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒=1formulae-sequence𝖫𝗂𝗇𝖾𝖺𝗋𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒1\mathsf{Linear}\mathsf{.sensitivity}=1;
3. 3.

   𝖫𝗂𝗇𝖾𝖺𝗋.𝗆𝖺𝗌𝗌=μformulae-sequence𝖫𝗂𝗇𝖾𝖺𝗋𝗆𝖺𝗌𝗌𝜇\mathsf{Linear}\mathsf{.mass}=\mu, where μ≥0𝜇0\mu\geq 0 is a hyperparameter;
4. 4.

   𝖫𝗂𝗇𝖾𝖺𝗋.𝗇𝗈𝗋𝗆​(𝑾)=‖𝑾‖RMS→RMSformulae-sequence𝖫𝗂𝗇𝖾𝖺𝗋𝗇𝗈𝗋𝗆𝑾subscriptnorm𝑾→RMSRMS\mathsf{Linear}\mathsf{.norm}({\bm{W}})=\|{{\bm{W}}}\|\_{\mathrm{RMS}\to\mathrm{RMS}}, the RMS→RMS→RMSRMS\mathrm{RMS}\to\mathrm{RMS} induced operator norm.

Since the 𝖫𝗂𝗇𝖾𝖺𝗋𝖫𝗂𝗇𝖾𝖺𝗋\mathsf{Linear} module is intended to map to and from vectors of roughly unit RMSRMS\mathrm{RMS} norm, we place the RMSRMS\mathrm{RMS} norm on both the input and output space: ∥⋅∥𝒳=∥⋅∥RMS\|{\cdot}\|\_{\mathcal{X}}=\|{\cdot}\|\_{\mathrm{RMS}} and ∥⋅∥𝒴=∥⋅∥RMS\|{\cdot}\|\_{\mathcal{Y}}=\|{\cdot}\|\_{\mathrm{RMS}}. Then 𝖫𝗂𝗇𝖾𝖺𝗋𝖫𝗂𝗇𝖾𝖺𝗋\mathsf{Linear} is well-normed if the inputs and weights belong to the unit balls {𝐱∈ℝdin:‖𝐱‖𝒳≤1}conditional-set𝐱superscriptℝsubscript𝑑insubscriptnorm𝐱𝒳1\left\{{\bm{x}}\in\mathbb{R}^{d\_{\mathrm{in}}}:\|{{\bm{x}}}\|\_{\mathcal{X}}\leq 1\right\} and {𝐖∈ℝdout×din:𝖫𝗂𝗇𝖾𝖺𝗋.𝗇𝗈𝗋𝗆​(𝐖)≤1}conditional-set𝐖superscriptℝsubscript𝑑outsubscript𝑑informulae-sequence𝖫𝗂𝗇𝖾𝖺𝗋𝗇𝗈𝗋𝗆𝐖1\left\{{\bm{W}}\in\mathbb{R}^{d\_{\mathrm{out}}\times d\_{\mathrm{in}}}:\mathsf{Linear}\mathsf{.norm}({\bm{W}})\leq 1\right\}. Referring back to [Section 3.3](#S3.SS3 "3.3 Basic Norms and Duality Maps ‣ 3 Theoretical Preliminaries ‣ Modular Duality in Deep Learning"), the duality map corresponding to 𝖫𝗂𝗇𝖾𝖺𝗋.𝗇𝗈𝗋𝗆formulae-sequence𝖫𝗂𝗇𝖾𝖺𝗋𝗇𝗈𝗋𝗆\mathsf{Linear}\mathsf{.norm} is then given by:

1. 5.

   𝖫𝗂𝗇𝖾𝖺𝗋.𝖽𝗎𝖺𝗅𝗂𝗓𝖾​(𝑮)=doutdin×𝑼​𝑽⊤formulae-sequence𝖫𝗂𝗇𝖾𝖺𝗋𝖽𝗎𝖺𝗅𝗂𝗓𝖾𝑮subscript𝑑outsubscript𝑑in𝑼superscript𝑽top\mathsf{Linear}\mathsf{.dualize}({\bm{G}})=\sqrt{\frac{d\_{\mathrm{out}}}{d\_{\mathrm{in}}}}\times{\bm{U}}{\bm{V}}^{\top}, where the gradient 𝑮∈ℝdout×din𝑮superscriptℝsubscript𝑑outsubscript𝑑in{\bm{G}}\in\mathbb{R}^{d\_{\mathrm{out}}\times d\_{\mathrm{in}}} has reduced SVD 𝑮=𝑼​𝚺​𝑽⊤𝑮𝑼𝚺superscript𝑽top{\bm{G}}={\bm{U}}{\bm{\Sigma}}{\bm{V}}^{\top}.

This single duality map recovers essential features of both maximal update parameterization (Yang & Hu, [2021](#bib.bib36), μ𝜇\muP) and Shampoo (Gupta et al., [2018](#bib.bib18)). In particular, the factor of dout/dinsubscript𝑑outsubscript𝑑in\sqrt{d\_{\mathrm{out}}/d\_{\mathrm{in}}} in 𝖫𝗂𝗇𝖾𝖺𝗋.𝖽𝗎𝖺𝗅𝗂𝗓𝖾formulae-sequence𝖫𝗂𝗇𝖾𝖺𝗋𝖽𝗎𝖺𝗅𝗂𝗓𝖾\mathsf{Linear}\mathsf{.dualize} recovers spectral update scaling (Yang et al., [2023](#bib.bib37)) that leads to μ𝜇\muP. (Initializing such that 𝖫𝗂𝗇𝖾𝖺𝗋.𝗇𝗈𝗋𝗆​(𝑾)=1formulae-sequence𝖫𝗂𝗇𝖾𝖺𝗋𝗇𝗈𝗋𝗆𝑾1\mathsf{Linear}\mathsf{.norm}({\bm{W}})=1 also recovers μ𝜇\muP initialization scaling.) And the mapping 𝑮↦𝑼​𝑽⊤maps-to𝑮𝑼superscript𝑽top{\bm{G}}\mapsto{\bm{U}}{\bm{V}}^{\top} is equivalent to Shampoo without accumulation (Bernstein & Newhouse, [2024](#bib.bib3)). As such, we believe that duality maps may help reconcile different strands of deep learning research and provide a unifying basis for fast and scalable training algorithms.

The 𝖤𝗆𝖻𝖾𝖽𝖤𝗆𝖻𝖾𝖽\mathsf{Embed} module provides a useful counterpoint to the 𝖫𝗂𝗇𝖾𝖺𝗋𝖫𝗂𝗇𝖾𝖺𝗋\mathsf{Linear} module. The difference between the two modules stems from the fact that the input spaces of 𝖤𝗆𝖻𝖾𝖽𝖤𝗆𝖻𝖾𝖽\mathsf{Embed} and 𝖫𝗂𝗇𝖾𝖺𝗋𝖫𝗂𝗇𝖾𝖺𝗋\mathsf{Linear} have different semantics.

###### Example 6 (The 𝖤𝗆𝖻𝖾𝖽𝖤𝗆𝖻𝖾𝖽\mathsf{Embed} module).

The 𝖤𝗆𝖻𝖾𝖽𝖤𝗆𝖻𝖾𝖽\mathsf{Embed} module sends inputs from 𝒳=ℝdin𝒳superscriptℝsubscript𝑑in\mathcal{X}=\mathbb{R}^{d\_{\mathrm{in}}} to outputs in 𝒴=ℝdout𝒴superscriptℝsubscript𝑑out\mathcal{Y}=\mathbb{R}^{d\_{\mathrm{out}}}. The weight space is given by the matrix space 𝒲=ℝdout×din𝒲superscriptℝsubscript𝑑outsubscript𝑑in\mathcal{W}=\mathbb{R}^{d\_{\mathrm{out}}\times d\_{\mathrm{in}}}. We endow the 𝖤𝗆𝖻𝖾𝖽𝖤𝗆𝖻𝖾𝖽\mathsf{Embed} module with attributes:

1. 1.

   𝖤𝗆𝖻𝖾𝖽.𝖿𝗈𝗋𝗐𝖺𝗋𝖽​(𝑾,𝒙)=𝑾​𝒙formulae-sequence𝖤𝗆𝖻𝖾𝖽𝖿𝗈𝗋𝗐𝖺𝗋𝖽𝑾𝒙𝑾𝒙\mathsf{Embed}\mathsf{.forward}({\bm{W}},{\bm{x}})={\bm{W}}{\bm{x}}, the matrix-vector product;
2. 2.

   𝖤𝗆𝖻𝖾𝖽.𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒=1formulae-sequence𝖤𝗆𝖻𝖾𝖽𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒1\mathsf{Embed}\mathsf{.sensitivity}=1;
3. 3.

   𝖤𝗆𝖻𝖾𝖽.𝗆𝖺𝗌𝗌=μformulae-sequence𝖤𝗆𝖻𝖾𝖽𝗆𝖺𝗌𝗌𝜇\mathsf{Embed}\mathsf{.mass}=\mu, where μ≥0𝜇0\mu\geq 0 is a hyperparameter;
4. 4.

   𝖤𝗆𝖻𝖾𝖽.𝗇𝗈𝗋𝗆​(𝑾)=‖𝑾‖ℓ1→RMSformulae-sequence𝖤𝗆𝖻𝖾𝖽𝗇𝗈𝗋𝗆𝑾subscriptnorm𝑾→subscriptℓ1RMS\mathsf{Embed}\mathsf{.norm}({\bm{W}})=\|{{\bm{W}}}\|\_{\ell\_{1}\to\mathrm{RMS}}, the ℓ1→RMS→subscriptℓ1RMS\ell\_{1}\to\mathrm{RMS} induced operator norm.

𝖤𝗆𝖻𝖾𝖽𝖤𝗆𝖻𝖾𝖽\mathsf{Embed} is intended to map from one-hot vectors to vectors of roughly unit RMSRMS\mathrm{RMS} norm, so we place the ℓ1subscriptℓ1\ell\_{1} norm on the input space and the RMSRMS\mathrm{RMS} norm on the output space: ∥⋅∥𝒳=∥⋅∥ℓ1\|{\cdot}\|\_{\mathcal{X}}=\|{\cdot}\|\_{\ell\_{1}} and ∥⋅∥𝒴=∥⋅∥RMS\|{\cdot}\|\_{\mathcal{Y}}=\|{\cdot}\|\_{\mathrm{RMS}}. Then 𝖤𝗆𝖻𝖾𝖽𝖤𝗆𝖻𝖾𝖽\mathsf{Embed} is well-normed if the inputs and weights belong to the unit balls {𝐱∈ℝdin:‖𝐱‖𝒳≤1}conditional-set𝐱superscriptℝsubscript𝑑insubscriptnorm𝐱𝒳1\left\{{\bm{x}}\in\mathbb{R}^{d\_{\mathrm{in}}}:\|{{\bm{x}}}\|\_{\mathcal{X}}\leq 1\right\} and {𝐖∈ℝdout×din:𝖤𝗆𝖻𝖾𝖽.𝗇𝗈𝗋𝗆​(𝐖)≤1}conditional-set𝐖superscriptℝsubscript𝑑outsubscript𝑑informulae-sequence𝖤𝗆𝖻𝖾𝖽𝗇𝗈𝗋𝗆𝐖1\left\{{\bm{W}}\in\mathbb{R}^{d\_{\mathrm{out}}\times d\_{\mathrm{in}}}:\mathsf{Embed}\mathsf{.norm}({\bm{W}})\leq 1\right\}. Referring back to [Section 3.3](#S3.SS3 "3.3 Basic Norms and Duality Maps ‣ 3 Theoretical Preliminaries ‣ Modular Duality in Deep Learning"), the duality map for 𝖤𝗆𝖻𝖾𝖽.𝗇𝗈𝗋𝗆formulae-sequence𝖤𝗆𝖻𝖾𝖽𝗇𝗈𝗋𝗆\mathsf{Embed}\mathsf{.norm} is:

1. 5.

   𝖤𝗆𝖻𝖾𝖽.𝖽𝗎𝖺𝗅𝗂𝗓𝖾​(𝑮)formulae-sequence𝖤𝗆𝖻𝖾𝖽𝖽𝗎𝖺𝗅𝗂𝗓𝖾𝑮\mathsf{Embed}\mathsf{.dualize}({\bm{G}}) performs the mapping colj​(𝑮)↦colj​(𝑮)‖colj​(𝑮)‖RMSmaps-tosubscriptcol𝑗𝑮subscriptcol𝑗𝑮subscriptnormsubscriptcol𝑗𝑮RMS\mathrm{col}\_{j}({\bm{G}})\mapsto\frac{\mathrm{col}\_{j}({\bm{G}})}{\|{\mathrm{col}\_{j}({\bm{G}})}\|\_{\mathrm{RMS}}} for each column index j=1,…,din𝑗
   1…subscript𝑑inj=1,...,d\_{\mathrm{in}}.

Finally, we consider a 𝖢𝗈𝗇𝗏𝟤𝖣𝖢𝗈𝗇𝗏𝟤𝖣\mathsf{Conv2D} module with a k×k𝑘𝑘k\times k kernel. 𝖢𝗈𝗇𝗏𝟤𝖣𝖢𝗈𝗇𝗏𝟤𝖣\mathsf{Conv2D} has a more involved tensor structure than 𝖫𝗂𝗇𝖾𝖺𝗋𝖫𝗂𝗇𝖾𝖺𝗋\mathsf{Linear} and 𝖤𝗆𝖻𝖾𝖽𝖤𝗆𝖻𝖾𝖽\mathsf{Embed}. The calculations work by slicing up the weight tensor into a collection of k2superscript𝑘2k^{2} matrices.

###### Example 7 (The 𝖢𝗈𝗇𝗏𝟤𝖣𝖢𝗈𝗇𝗏𝟤𝖣\mathsf{Conv2D} module).

The 𝖢𝗈𝗇𝗏𝟤𝖣𝖢𝗈𝗇𝗏𝟤𝖣\mathsf{Conv2D} module sends inputs from 𝒳=ℝWin×Hin×din𝒳superscriptℝsubscript𝑊insubscript𝐻insubscript𝑑in\mathcal{X}=\mathbb{R}^{W\_{\mathrm{in}}\times H\_{\mathrm{in}}\times d\_{\mathrm{in}}} to outputs in 𝒴=ℝWout×Hout×dout𝒴superscriptℝsubscript𝑊outsubscript𝐻outsubscript𝑑out\mathcal{Y}=\mathbb{R}^{W\_{\mathrm{out}}\times H\_{\mathrm{out}}\times d\_{\mathrm{out}}}. We think of this as mapping an input image of width Winsubscript𝑊inW\_{\mathrm{in}}, height Hinsubscript𝐻inH\_{\mathrm{in}} and with dinsubscript𝑑ind\_{\mathrm{in}} color channels to an output image of width Woutsubscript𝑊outW\_{\mathrm{out}}, height Houtsubscript𝐻outH\_{\mathrm{out}} and with doutsubscript𝑑outd\_{\mathrm{out}} color channels. The weight space is given by the tensor space 𝒲=ℝdout×din×k×k𝒲superscriptℝsubscript𝑑outsubscript𝑑in𝑘𝑘\mathcal{W}=\mathbb{R}^{d\_{\mathrm{out}}\times d\_{\mathrm{in}}\times k\times k}, where k𝑘k is the kernel size. We endow 𝖢𝗈𝗇𝗏𝟤𝖣𝖢𝗈𝗇𝗏𝟤𝖣\mathsf{Conv2D} with attributes:

1. 1.

   𝖢𝗈𝗇𝗏𝟤𝖣.𝖿𝗈𝗋𝗐𝖺𝗋𝖽​(𝑾,𝒙)=𝑾⊛𝒙formulae-sequence𝖢𝗈𝗇𝗏𝟤𝖣𝖿𝗈𝗋𝗐𝖺𝗋𝖽𝑾𝒙⊛𝑾𝒙\mathsf{Conv2D}\mathsf{.forward}({\bm{W}},{\bm{x}})={\bm{W}}\circledast{\bm{x}}, where ⊛⊛\circledast denotes 2D convolution;
2. 2.

   𝖢𝗈𝗇𝗏𝟤𝖣.𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒=1formulae-sequence𝖢𝗈𝗇𝗏𝟤𝖣𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒1\mathsf{Conv2D}\mathsf{.sensitivity}=1;
3. 3.

   𝖢𝗈𝗇𝗏𝟤𝖣.𝗆𝖺𝗌𝗌=μformulae-sequence𝖢𝗈𝗇𝗏𝟤𝖣𝗆𝖺𝗌𝗌𝜇\mathsf{Conv2D}\mathsf{.mass}=\mu, where μ≥0𝜇0\mu\geq 0 is a hyperparameter;
4. 4.

   𝖢𝗈𝗇𝗏𝟤𝖣.𝗇𝗈𝗋𝗆​(𝑾)=k2​maxi,j=1k⁡‖𝑾⋅⁣⋅i​j‖RMS→RMSformulae-sequence𝖢𝗈𝗇𝗏𝟤𝖣𝗇𝗈𝗋𝗆𝑾superscript𝑘2superscriptsubscript
   𝑖𝑗1𝑘subscriptnormsubscript𝑾
   ⋅⋅absent𝑖𝑗→RMSRMS\mathsf{Conv2D}\mathsf{.norm}({\bm{W}})=k^{2}\max\_{i,j=1}^{k}\|{{\bm{W}}\_{\cdot\cdot ij}}\|\_{\mathrm{RMS}\to\mathrm{RMS}}, the max RMS→RMS→RMSRMS\mathrm{RMS}\to\mathrm{RMS} norm over kernel indices.

We would like pixel intensities in the inputs and outputs to be order one and undergo order one change. We formalize this by taking the input and output norms to be the spatial maximum of the RMS norms of all the color channel vectors: ‖𝐱‖𝒳=maxw=1Win⁡maxh=1Hin⁡‖𝐱w​h⁣⋅‖RMSsubscriptnorm𝐱𝒳superscriptsubscript𝑤1subscript𝑊insuperscriptsubscriptℎ1subscript𝐻insubscriptnormsubscript𝐱

𝑤ℎ⋅RMS\|{{\bm{x}}}\|\_{\mathcal{X}}=\max\_{w=1}^{W\_{\mathrm{in}}}\max\_{h=1}^{H\_{\mathrm{in}}}\|{{\bm{x}}\_{wh\cdot}}\|\_{\mathrm{RMS}} and ‖𝐲‖𝒴=maxw=1Wout⁡maxh=1Hout⁡‖𝐲w​h⁣⋅‖RMSsubscriptnorm𝐲𝒴superscriptsubscript𝑤1subscript𝑊outsuperscriptsubscriptℎ1subscript𝐻outsubscriptnormsubscript𝐲

𝑤ℎ⋅RMS\|{{\bm{y}}}\|\_{\mathcal{Y}}=\max\_{w=1}^{W\_{\mathrm{out}}}\max\_{h=1}^{H\_{\mathrm{out}}}\|{{\bm{y}}\_{wh\cdot}}\|\_{\mathrm{RMS}}. Then 𝖢𝗈𝗇𝗏𝟤𝖣𝖢𝗈𝗇𝗏𝟤𝖣\mathsf{Conv2D} is well-normed if the inputs and weights belong to the unit balls {𝐱∈ℝWin×Hin×din:‖𝐱‖𝒳≤1}conditional-set𝐱superscriptℝsubscript𝑊insubscript𝐻insubscript𝑑insubscriptnorm𝐱𝒳1\left\{{\bm{x}}\in\mathbb{R}^{W\_{\mathrm{in}}\times H\_{\mathrm{in}}\times d\_{\mathrm{in}}}:\|{{\bm{x}}}\|\_{\mathcal{X}}\leq 1\right\} and {𝐖∈ℝdout×din×k×k:𝖢𝗈𝗇𝗏𝟤𝖣.𝗇𝗈𝗋𝗆​(𝐖)≤1}conditional-set𝐖superscriptℝsubscript𝑑outsubscript𝑑in𝑘𝑘formulae-sequence𝖢𝗈𝗇𝗏𝟤𝖣𝗇𝗈𝗋𝗆𝐖1\left\{{\bm{W}}\in\mathbb{R}^{d\_{\mathrm{out}}\times d\_{\mathrm{in}}\times k\times k}:\mathsf{Conv2D}\mathsf{.norm}({\bm{W}})\leq 1\right\}. Since the duality map for a max of norms decouples into one duality map per sub-norm, the duality map corresponding to 𝖢𝗈𝗇𝗏𝟤𝖣.𝗇𝗈𝗋𝗆formulae-sequence𝖢𝗈𝗇𝗏𝟤𝖣𝗇𝗈𝗋𝗆\mathsf{Conv2D}\mathsf{.norm} is given by:

1. 5.

   𝖢𝗈𝗇𝗏𝟤𝖣.𝖽𝗎𝖺𝗅𝗂𝗓𝖾​(𝑮)formulae-sequence𝖢𝗈𝗇𝗏𝟤𝖣𝖽𝗎𝖺𝗅𝗂𝗓𝖾𝑮\mathsf{Conv2D}\mathsf{.dualize}({\bm{G}}) does 𝑮⋅⁣⋅i​j↦1k2​doutdin×𝑼i​j​𝑽i​j⊤maps-tosubscript𝑮
   ⋅⋅absent𝑖𝑗1superscript𝑘2subscript𝑑outsubscript𝑑insubscript𝑼𝑖𝑗superscriptsubscript𝑽𝑖𝑗top{\bm{G}}\_{\cdot\cdot ij}\mapsto\frac{1}{k^{2}}\sqrt{\frac{d\_{\mathrm{out}}}{d\_{\mathrm{in}}}}\times{\bm{U}}\_{ij}{\bm{V}}\_{ij}^{\top}, where 𝑮⋅⁣⋅i​jsubscript𝑮
   ⋅⋅absent𝑖𝑗{\bm{G}}\_{\cdot\cdot ij} has reduced SVD 𝑼i​j​𝚺i​j​𝑽i​j⊤subscript𝑼𝑖𝑗subscript𝚺𝑖𝑗superscriptsubscript𝑽𝑖𝑗top{\bm{U}}\_{ij}{\bm{\Sigma}}\_{ij}{\bm{V}}\_{ij}^{\top}.

|  |  |  |  |
| --- | --- | --- | --- |
| Module | Weight Space 𝓦𝓦\bm{\mathcal{W}} | 𝗠𝗼𝗱𝘂𝗹𝗲.𝗻𝗼𝗿𝗺formulae-sequence𝗠𝗼𝗱𝘂𝗹𝗲𝗻𝗼𝗿𝗺\bm{\mathsf{Module}\mathsf{.norm}} | 𝗠𝗼𝗱𝘂𝗹𝗲.𝗱𝘂𝗮𝗹𝗶𝘇𝗲formulae-sequence𝗠𝗼𝗱𝘂𝗹𝗲𝗱𝘂𝗮𝗹𝗶𝘇𝗲\bm{\mathsf{Module}\mathsf{.dualize}} |
| 𝖫𝗂𝗇𝖾𝖺𝗋𝖫𝗂𝗇𝖾𝖺𝗋\mathsf{Linear} | ℝdout×dinsuperscriptℝsubscript𝑑outsubscript𝑑in\mathbb{R}^{d\_{\mathrm{out}}\times d\_{\mathrm{in}}} | 𝑾↦‖𝑾‖RMS→RMSmaps-to𝑾subscriptnorm𝑾→RMSRMS{\bm{W}}\mapsto\|{{\bm{W}}}\|\_{\mathrm{RMS}\to\mathrm{RMS}} | 𝑮↦doutdin×𝑼​𝑽⊤maps-to𝑮subscript𝑑outsubscript𝑑in𝑼superscript𝑽top{\bm{G}}\mapsto\sqrt{\frac{d\_{\mathrm{out}}}{d\_{\mathrm{in}}}}\times{\bm{U}}{\bm{V}}^{\top} |
| 𝖤𝗆𝖻𝖾𝖽𝖤𝗆𝖻𝖾𝖽\mathsf{Embed} | ℝdout×dinsuperscriptℝsubscript𝑑outsubscript𝑑in\mathbb{R}^{d\_{\mathrm{out}}\times d\_{\mathrm{in}}} | 𝑾↦‖𝑾‖ℓ1→RMSmaps-to𝑾subscriptnorm𝑾→subscriptℓ1RMS{\bm{W}}\mapsto\|{{\bm{W}}}\|\_{\ell\_{1}\to\mathrm{RMS}} | colj​(𝑮)↦colj​(𝑮)‖colj​(𝑮)‖RMSmaps-tosubscriptcol𝑗𝑮subscriptcol𝑗𝑮subscriptnormsubscriptcol𝑗𝑮RMS\mathrm{col}\_{j}({\bm{G}})\mapsto\frac{\mathrm{col}\_{j}({\bm{G}})}{\|{\mathrm{col}\_{j}({\bm{G}})}\|\_{\mathrm{RMS}}} |
| 𝖢𝗈𝗇𝗏𝟤𝖣𝖢𝗈𝗇𝗏𝟤𝖣\mathsf{Conv2D} | ℝdout×din×k×ksuperscriptℝsubscript𝑑outsubscript𝑑in𝑘𝑘\mathbb{R}^{d\_{\mathrm{out}}\times d\_{\mathrm{in}}\times k\times k} | 𝑾↦k2​maxi,j=1k⁡‖𝑾⋅⁣⋅i​j‖RMS→RMSmaps-to𝑾superscript𝑘2superscriptsubscript  𝑖𝑗 1𝑘subscriptnormsubscript𝑾  ⋅⋅absent𝑖𝑗→RMSRMS{\bm{W}}\mapsto k^{2}\max\_{i,j=1}^{k}\|{{\bm{W}}\_{\cdot\cdot ij}}\|\_{\mathrm{RMS}\to\mathrm{RMS}} | 𝑮⋅⁣⋅i​j↦1k2​doutdin×𝑼i​j​𝑽i​j⊤maps-tosubscript𝑮  ⋅⋅absent𝑖𝑗1superscript𝑘2subscript𝑑outsubscript𝑑insubscript𝑼𝑖𝑗superscriptsubscript𝑽𝑖𝑗top{\bm{G}}\_{\cdot\cdot ij}\mapsto\frac{1}{k^{2}}\sqrt{\frac{d\_{\mathrm{out}}}{d\_{\mathrm{in}}}}\times{\bm{U}}\_{ij}{\bm{V}}\_{ij}^{\top} |

Table 1: Duality maps for three atomic modules: 𝗟𝗶𝗻𝗲𝗮𝗿𝗟𝗶𝗻𝗲𝗮𝗿\bm{\mathsf{Linear}}, 𝗘𝗺𝗯𝗲𝗱𝗘𝗺𝗯𝗲𝗱\bm{\mathsf{Embed}}, and 𝗖𝗼𝗻𝘃𝟮𝗗𝗖𝗼𝗻𝘃𝟮𝗗\bm{\mathsf{Conv2D}}. These atomic modules are sufficient to build convolutional neural networks and transformers. In 𝖫𝗂𝗇𝖾𝖺𝗋.𝖽𝗎𝖺𝗅𝗂𝗓𝖾formulae-sequence𝖫𝗂𝗇𝖾𝖺𝗋𝖽𝗎𝖺𝗅𝗂𝗓𝖾\mathsf{Linear}\mathsf{.dualize}, we let 𝑼​𝚺​𝑽⊤𝑼𝚺superscript𝑽top{\bm{U}}{\bm{\Sigma}}{\bm{V}}^{\top} denote the reduced SVD of the gradient matrix 𝑮𝑮{\bm{G}}. In 𝖢𝗈𝗇𝗏𝟤𝖣.𝖽𝗎𝖺𝗅𝗂𝗓𝖾formulae-sequence𝖢𝗈𝗇𝗏𝟤𝖣𝖽𝗎𝖺𝗅𝗂𝗓𝖾\mathsf{Conv2D}\mathsf{.dualize}, we let 𝑼i​j​𝚺i​j​𝑽i​j⊤subscript𝑼𝑖𝑗subscript𝚺𝑖𝑗superscriptsubscript𝑽𝑖𝑗top\smash{{\bm{U}}\_{ij}{\bm{\Sigma}}\_{ij}{\bm{V}}\_{ij}^{\top}} denote the reduced SVD of the slice of the gradient tensor 𝑮⋅⁣⋅i​jsubscript𝑮

⋅⋅absent𝑖𝑗{\bm{G}}\_{\cdot\cdot ij} at kernel indices i𝑖i and j𝑗j. [Section 5](#S5 "5 Fast Duality Maps ‣ Modular Duality in Deep Learning") provides GPU-friendly algorithms for computing these duality maps based on a new family of Newton-Schulz iterations that we propose.

### 4.2 Duality Maps for Bond Modules

Large et al. ([2024](#bib.bib25)) define another class of basic modules: bond modules. Bonds are handwritten modules without weights. An example of a bond is the 𝖱𝖾𝖫𝖴𝖱𝖾𝖫𝖴\mathsf{ReLU} nonlinearity. For a bond 𝖡𝖡\mathsf{B}, the weight space is the zero vector space 𝒲={0}𝒲0\mathcal{W}=\{0\} and the modular norm 𝖡.𝗇𝗈𝗋𝗆=0↦0formulae-sequence𝖡𝗇𝗈𝗋𝗆0maps-to0\mathsf{B}\mathsf{.norm}=0\mapsto 0. As such, the corresponding duality map is also 𝖡.𝖽𝗎𝖺𝗅𝗂𝗓𝖾=0↦0formulae-sequence𝖡𝖽𝗎𝖺𝗅𝗂𝗓𝖾0maps-to0\mathsf{B}\mathsf{.dualize}=0\mapsto 0. In a software package, one need not write norms or duality maps for bond modules.

### 4.3 Duality Maps for Compound Modules

First, given two composable modules 𝖬1subscript𝖬1\mathsf{M}\_{1} and 𝖬2subscript𝖬2\mathsf{M}\_{2}, the duality map for the composite 𝖬=𝖬2∘𝖬1𝖬subscript𝖬2subscript𝖬1\mathsf{M}=\mathsf{M}\_{2}\circ\mathsf{M}\_{1} is given by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝖬.𝖽𝗎𝖺𝗅𝗂𝗓𝖾(𝒈1,𝒈2)=(1𝖬2.𝗌𝖾𝗇𝗌𝗂𝗍𝗂𝗏𝗂𝗍𝗒∗𝖬1.𝗆𝖺𝗌𝗌𝖬.𝗆𝖺𝗌𝗌∗𝖬1.𝖽𝗎𝖺𝗅𝗂𝗓𝖾(𝒈1),𝖬2.𝗆𝖺𝗌𝗌𝖬.𝗆𝖺𝗌𝗌∗𝖬2.𝖽𝗎𝖺𝗅𝗂𝗓𝖾(𝒈2)).\mathsf{M}\mathsf{.dualize}({\bm{g}}\_{1},{\bm{g}}\_{2})=\left(\frac{1}{\mathsf{M}\_{2}\mathsf{.sensitivity}}\*\frac{\mathsf{M}\_{1}\mathsf{.mass}}{\mathsf{M}\mathsf{.mass}}\*\mathsf{M}\_{1}\mathsf{.dualize}({\bm{g}}\_{1}),\frac{\mathsf{M}\_{2}\mathsf{.mass}}{\mathsf{M}\mathsf{.mass}}\*\mathsf{M}\_{2}\mathsf{.dualize}({\bm{g}}\_{2})\right). |  | (11) |

And second, given two concatenatable modules 𝖬1subscript𝖬1\mathsf{M}\_{1} and 𝖬2subscript𝖬2\mathsf{M}\_{2}, the duality map for the tuple 𝖬=(𝖬1,𝖬2)𝖬subscript𝖬1subscript𝖬2\mathsf{M}=(\mathsf{M}\_{1},\mathsf{M}\_{2}) is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝖬.𝖽𝗎𝖺𝗅𝗂𝗓𝖾(𝒈1,𝒈2)=(𝖬1.𝗆𝖺𝗌𝗌𝖬.𝗆𝖺𝗌𝗌∗𝖬1.𝖽𝗎𝖺𝗅𝗂𝗓𝖾(𝒈1),𝖬2.𝗆𝖺𝗌𝗌𝖬.𝗆𝖺𝗌𝗌∗𝖬2.𝖽𝗎𝖺𝗅𝗂𝗓𝖾(𝒈2)).\mathsf{M}\mathsf{.dualize}({\bm{g}}\_{1},{\bm{g}}\_{2})=\left(\frac{\mathsf{M}\_{1}\mathsf{.mass}}{\mathsf{M}\mathsf{.mass}}\*\mathsf{M}\_{1}\mathsf{.dualize}({\bm{g}}\_{1}),\frac{\mathsf{M}\_{2}\mathsf{.mass}}{\mathsf{M}\mathsf{.mass}}\*\mathsf{M}\_{2}\mathsf{.dualize}({\bm{g}}\_{2})\right). |  | (12) |

The proofs of [Equations 11](#S4.E11 "In 4.3 Duality Maps for Compound Modules ‣ 4 Modular Dualization ‣ Modular Duality in Deep Learning") and [12](#S4.E12 "Equation 12 ‣ 4.3 Duality Maps for Compound Modules ‣ 4 Modular Dualization ‣ Modular Duality in Deep Learning") follow in a straightforward manner from
[Definitions 6](#Thmmydefinition6 "Definition 6 (Module composition). ‣ 3.4 The Modular Norm ‣ 3 Theoretical Preliminaries ‣ Modular Duality in Deep Learning") and [7](#Thmmydefinition7 "Definition 7 (Module concatenation). ‣ 3.4 The Modular Norm ‣ 3 Theoretical Preliminaries ‣ Modular Duality in Deep Learning").

## 5 Fast Duality Maps

For modular dualization to be practically feasible, we need ways of computing duality maps quickly. Inspecting the duality maps listed in [Table 1](#S4.T1 "In 4.1 Duality Maps for Atomic Modules ‣ 4 Modular Dualization ‣ Modular Duality in Deep Learning"), we see that 𝖤𝗆𝖻𝖾𝖽.𝖽𝗎𝖺𝗅𝗂𝗓𝖾formulae-sequence𝖤𝗆𝖻𝖾𝖽𝖽𝗎𝖺𝗅𝗂𝗓𝖾\mathsf{Embed}\mathsf{.dualize} is easy to implement since it just involves computing vector norms of matrix columns. But 𝖫𝗂𝗇𝖾𝖺𝗋.𝖽𝗎𝖺𝗅𝗂𝗓𝖾formulae-sequence𝖫𝗂𝗇𝖾𝖺𝗋𝖽𝗎𝖺𝗅𝗂𝗓𝖾\mathsf{Linear}\mathsf{.dualize} and 𝖢𝗈𝗇𝗏𝟤𝖣.𝖽𝗎𝖺𝗅𝗂𝗓𝖾formulae-sequence𝖢𝗈𝗇𝗏𝟤𝖣𝖽𝗎𝖺𝗅𝗂𝗓𝖾\mathsf{Conv2D}\mathsf{.dualize} involve the projection:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝑮=𝑼​𝚺​𝑽⊤↦𝑼​𝑽⊤,𝑮𝑼𝚺superscript𝑽topmaps-to𝑼superscript𝑽top{\bm{G}}={\bm{U}}{\bm{\Sigma}}{\bm{V}}^{\top}\mapsto{\bm{U}}{\bm{V}}^{\top}, |  | (13) |

where 𝑼​𝚺​𝑽⊤𝑼𝚺superscript𝑽top{\bm{U}}{\bm{\Sigma}}{\bm{V}}^{\top} is the reduced SVD of the matrix 𝑮𝑮{\bm{G}}. Since computing SVDs can be slow (Carlson et al., [2015b](#bib.bib8); Flynn, [2017](#bib.bib14)), here we discuss three fast approximations to [Equation 13](#S5.E13 "In 5 Fast Duality Maps ‣ Modular Duality in Deep Learning") via sketching, iterations for inverse matrix roots, and a new family of rectangular Newton-Schulz iterations that we propose. Which method works best may depend on the condition number of the matrix 𝑮𝑮{\bm{G}} or the available computational resources.

### 5.1 Sketching

Sketching is a randomized method (Martinsson & Tropp, [2020](#bib.bib27)) that can be used to build low-rank approximations to the SVD. Carlson et al. ([2015b](#bib.bib8)) already used sketching to provide a fast approximation to their ##\#-operator. More recent papers have experimented with sketching in the context of Shampoo-type algorithms (Feinberg et al., [2023](#bib.bib13)). A potential downside of approximating [Equation 13](#S5.E13 "In 5 Fast Duality Maps ‣ Modular Duality in Deep Learning") via sketching is that randomized SVD methods usually try to accurately approximate the largest singular values of a matrix (Martinsson & Tropp, [2020](#bib.bib27), Section 11.2) while the value of [Equation 13](#S5.E13 "In 5 Fast Duality Maps ‣ Modular Duality in Deep Learning") may lie in its action on the small singular values.

### 5.2 Iterations for Inverse Matrix Roots

Given a full rank matrix 𝑮𝑮{\bm{G}} with reduced SVD 𝑼​𝚺​𝑽⊤𝑼𝚺superscript𝑽top{\bm{U}}{\bm{\Sigma}}{\bm{V}}^{\top}, we have that:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝑼​𝑽⊤=(𝑮​𝑮⊤)−1/4​𝑮​(𝑮⊤​𝑮)−1/4=(𝑮​𝑮⊤)−1/2​𝑮=𝑮​(𝑮⊤​𝑮)−1/2.𝑼superscript𝑽topsuperscript𝑮superscript𝑮top14𝑮superscriptsuperscript𝑮top𝑮14superscript𝑮superscript𝑮top12𝑮𝑮superscriptsuperscript𝑮top𝑮12{\bm{U}}{\bm{V}}^{\top}=({\bm{G}}{\bm{G}}^{\top})^{-\nicefrac{{1}}{{4}}}\,{\bm{G}}\,({\bm{G}}^{\top}{\bm{G}})^{-\nicefrac{{1}}{{4}}}=({\bm{G}}{\bm{G}}^{\top})^{-\nicefrac{{1}}{{2}}}\,{\bm{G}}={\bm{G}}\,({\bm{G}}^{\top}{\bm{G}})^{-\nicefrac{{1}}{{2}}}. |  | (14) |

This provides a route to approximating [Equation 13](#S5.E13 "In 5 Fast Duality Maps ‣ Modular Duality in Deep Learning") since one can compute inverse matrix roots such as (𝑮​𝑮⊤)−1/2superscript𝑮superscript𝑮top12({\bm{G}}{\bm{G}}^{\top})^{-\nicefrac{{1}}{{2}}} via Newton iteration (Lakić, [1998](#bib.bib24)). This is discussed in Chapter 7 of Higham ([2008](#bib.bib20))’s book and also see Anil et al. ([2020](#bib.bib2))’s paper. Care must be taken with inverses whenever the matrix 𝑮𝑮{\bm{G}} is ill-conditioned.

### 5.3 Rectangular Newton-Schulz Iteration

We developed a novel “rectangular Newton-Schulz iteration” for computing 𝑼​𝑽⊤𝑼superscript𝑽top{\bm{U}}{\bm{V}}^{\top}. In short, if we first normalize the matrix 𝑮𝑮{\bm{G}} according to 𝑿0=𝑮/‖𝑮‖ℓ2→ℓ2subscript𝑿0𝑮subscriptnorm𝑮→subscriptℓ2subscriptℓ2{\bm{X}}\_{0}={\bm{G}}/\|{{\bm{G}}}\|\_{\ell\_{2}\to\ell\_{2}} (or alternatively 𝑿0=𝑮/‖𝑮‖Fsubscript𝑿0𝑮subscriptnorm𝑮𝐹{\bm{X}}\_{0}={\bm{G}}/\|{{\bm{G}}}\|\_{F}) and then iterate:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝑿t+1=32⋅𝑿t−12⋅𝑿t​𝑿t⊤​𝑿t,subscript𝑿𝑡1⋅32subscript𝑿𝑡⋅12subscript𝑿𝑡superscriptsubscript𝑿𝑡topsubscript𝑿𝑡{\bm{X}}\_{t+1}=\frac{3}{2}\cdot{\bm{X}}\_{t}-\frac{1}{2}\cdot{\bm{X}}\_{t}{\bm{X}}\_{t}^{\top}{\bm{X}}\_{t}, |  | (15) |

then as t→∞→𝑡t\to\infty, the sequence 𝑿t→𝑼​𝑽⊤→subscript𝑿𝑡𝑼superscript𝑽top{\bm{X}}\_{t}\to{\bm{U}}{\bm{V}}^{\top}. To see this, one can plot the univariate cubic function f​(x):=32⋅x−12⋅x3assign𝑓𝑥⋅32𝑥⋅12superscript𝑥3f(x)\vcentcolon=\tfrac{3}{2}\cdot x-\tfrac{1}{2}\cdot x^{3} and see that, for 0<x<30𝑥30<x<\sqrt{3}, iterating this cubic will push x𝑥x closer and closer to +11+1. The final step is to realize that the effect of the iteration in [Equation 15](#S5.E15 "In 5.3 Rectangular Newton-Schulz Iteration ‣ 5 Fast Duality Maps ‣ Modular Duality in Deep Learning") is to apply this cubic f​(x)𝑓𝑥f(x) to each singular value of 𝑿tsubscript𝑿𝑡{\bm{X}}\_{t}. This shows that the spectral normalization 𝑿0=𝑮/‖𝑮‖ℓ2→ℓ2subscript𝑿0𝑮subscriptnorm𝑮→subscriptℓ2subscriptℓ2{\bm{X}}\_{0}={\bm{G}}/\|{{\bm{G}}}\|\_{\ell\_{2}\to\ell\_{2}} is stronger than what is required: we need only ensure that 𝑿0subscript𝑿0{\bm{X}}\_{0} has singular values no greater than 33\sqrt{3} for the iteration to converge.

The iteration in [Equation 15](#S5.E15 "In 5.3 Rectangular Newton-Schulz Iteration ‣ 5 Fast Duality Maps ‣ Modular Duality in Deep Learning") has the advantage over sketching that it always works on all singular values, and since the iteration does not compute inverse matrix roots it is well-behaved even on low-rank matrices.

Finally, there are in fact a family of degree 2​n+12𝑛12n+1 polynomial iterations of the form

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝑿t+1=a⋅𝑿t+b⋅𝑿t​𝑿t⊤​𝑿t+c⋅(𝑿t​𝑿t⊤)2​𝑿t+…+z⋅(𝑿t​𝑿t⊤)n​𝑿tsubscript𝑿𝑡1⋅𝑎subscript𝑿𝑡⋅𝑏subscript𝑿𝑡superscriptsubscript𝑿𝑡topsubscript𝑿𝑡⋅𝑐superscriptsubscript𝑿𝑡superscriptsubscript𝑿𝑡top2subscript𝑿𝑡…⋅𝑧superscriptsubscript𝑿𝑡superscriptsubscript𝑿𝑡top𝑛subscript𝑿𝑡{\bm{X}}\_{t+1}=a\cdot{\bm{X}}\_{t}+b\cdot{\bm{X}}\_{t}{\bm{X}}\_{t}^{\top}{\bm{X}}\_{t}+c\cdot({\bm{X}}\_{t}{\bm{X}}\_{t}^{\top})^{2}{\bm{X}}\_{t}+...+z\cdot({\bm{X}}\_{t}{\bm{X}}\_{t}^{\top})^{n}{\bm{X}}\_{t} |  | (16) |

for suitable a,b,c,…,z

𝑎𝑏𝑐…𝑧a,b,c,...,z that could be used instead of [Equation 15](#S5.E15 "In 5.3 Rectangular Newton-Schulz Iteration ‣ 5 Fast Duality Maps ‣ Modular Duality in Deep Learning"). One should choose coefficients a,b,c,…,z

𝑎𝑏𝑐…𝑧a,b,c,...,z so that the univariate polynomial g​(x)=a⋅x+b⋅x3+c⋅x5+…+z⋅x2​n+1𝑔𝑥⋅𝑎𝑥⋅𝑏superscript𝑥3⋅𝑐superscript𝑥5…⋅𝑧superscript𝑥2𝑛1g(x)=a\cdot x+b\cdot x^{3}+c\cdot x^{5}+...+z\cdot x^{2n+1} is a suitable approximation to sign⁡(x)sign𝑥\operatorname{sign}(x). One may try to further accelerate the iteration by “tuning” the coefficients a,b,c,…,z

𝑎𝑏𝑐…𝑧a,b,c,...,z empirically.

We came up with [Equation 15](#S5.E15 "In 5.3 Rectangular Newton-Schulz Iteration ‣ 5 Fast Duality Maps ‣ Modular Duality in Deep Learning") by inspecting Equation 5.22 in Higham ([2008](#bib.bib20))’s book, which provides a related iteration for computing the “matrix sign function” for square matrices. We developed the graphical understanding ourselves and used this as the basis for proposing the higher-order polynomial iterations.

## 6 Discussion

This paper develops the theory of modular duality and the procedure of modular dualization as means to construct duality maps for general neural architectures. Here, we comment on implications and connections.

### 6.1 A Type System for Deep Learning

Part of the inspiration for this work is the idea of building a fully-fledged type system for deep learning. We think that activation spaces should be typed by their intended norm and the intended size of activations in that norm. This information would help in the construction of well-normed modules (see [Section 4.1](#S4.SS1 "4.1 Duality Maps for Atomic Modules ‣ 4 Modular Dualization ‣ Modular Duality in Deep Learning")). Modules should be typed according to [Definition 4](#Thmmydefinition4 "Definition 4 (Module). ‣ 3.4 The Modular Norm ‣ 3 Theoretical Preliminaries ‣ Modular Duality in Deep Learning"). And, as suggested in the introduction, gradients should be explicitly typed as dual vectors. A duality map should flip the type of a dual vector to a primal vector. We plan to use the Modula deep learning package (Large et al., [2024](#bib.bib25)) as a testbed for these ideas.

### 6.2 Neural Network Speedrunning

We believe that the ideas in this paper can help in the design of faster training methods. In fact, a new NanoGPT training speed record was recently set (Jordan, [2024](#bib.bib23)) using our rectangular Newton-Schulz iteration. We communicated the iteration to Keller Jordan through our workshop paper (Bernstein & Newhouse, [2024](#bib.bib3)).

### 6.3 Modular Duality: A Unifying Theoretical Framework for Fast and Scalable Training

An important topic in contemporary optimization research is the design of fast and scalable training methods for neural networks. In fact, the theme of the Optimization for Machine Learning workshop at this year’s NeurIPS conference is “scaling up optimization” (OPT, [2024](#bib.bib29)). Two popular methods in this research space are maximal update parameterization (Yang & Hu, [2021](#bib.bib36), μ𝜇\muP), which allows for increasing network width without changing the optimal learning rate, and Shampoo (Gupta et al., [2018](#bib.bib18)), a variant of which (Shi et al., [2023](#bib.bib31)) won a speed challenge at the inaugural AlgoPerf optimization competition (Dahl et al., [2023](#bib.bib10)).

We showed in [Section 4.1](#S4.SS1 "4.1 Duality Maps for Atomic Modules ‣ 4 Modular Dualization ‣ Modular Duality in Deep Learning") that essential features of both μ𝜇\muP and Shampoo are recovered from the single duality map 𝖫𝗂𝗇𝖾𝖺𝗋.𝖽𝗎𝖺𝗅𝗂𝗓𝖾formulae-sequence𝖫𝗂𝗇𝖾𝖺𝗋𝖽𝗎𝖺𝗅𝗂𝗓𝖾\mathsf{Linear}\mathsf{.dualize}. We think that, on a basic theoretical level, μ𝜇\muP and Shampoo should be viewed as partial approximations to this duality map. This observation helps put μ𝜇\muP and Shampoo on a consistent theoretical footing, orients the methods with respect to overlooked prior work on spectral descent (Carlson et al., [2015b](#bib.bib8)) and duality structure gradient descent (Flynn, [2017](#bib.bib14)), and suggests new ways to generalize these methods to arbitrary layer types and network architectures via the modular norm and modular dualization.

### 6.4 On the Alignment of Activations and Updates

Recent work (Yang et al., [2023](#bib.bib37); Everett et al., [2024](#bib.bib12); Large et al., [2024](#bib.bib25)) has singled out the following question as important to the design of scalable deep learning systems: to what extent do gradient updates to neural network layers align with incoming activation vectors? This question is important since it helps inform how large weight updates need to be to induce a certain amount of change in layer outputs. Duality maps such as 𝖫𝗂𝗇𝖾𝖺𝗋.𝖽𝗎𝖺𝗅𝗂𝗓𝖾formulae-sequence𝖫𝗂𝗇𝖾𝖺𝗋𝖽𝗎𝖺𝗅𝗂𝗓𝖾\mathsf{Linear}\mathsf{.dualize} and 𝖢𝗈𝗇𝗏𝟤𝖣.𝖽𝗎𝖺𝗅𝗂𝗓𝖾formulae-sequence𝖢𝗈𝗇𝗏𝟤𝖣𝖽𝗎𝖺𝗅𝗂𝗓𝖾\mathsf{Conv2D}\mathsf{.dualize} may help simplify the answer to this question, since they project gradients to scaled semi-orthogonal matrices for which all singular values have the same magnitude.

### 6.5 A Numerical Paradox: The Weights Don’t Change!

Past work (Lee et al., [2019](#bib.bib26); Jesus et al., [2021](#bib.bib22)) has pointed out an apparent paradox in deep learning: the weights seem to move a vanishing amount from initialization in the limit of large network width. This finding has motivated a substantial amount of work on linearized training dynamics (Jacot et al., [2018](#bib.bib21)). We attempted to resolve this paradox in prior work by showing that the weights move a roughly constant amount at any width when the change is measured in spectral norm (Yang et al., [2023](#bib.bib37)). But duality maps change the story again: 𝖫𝗂𝗇𝖾𝖺𝗋.𝖽𝗎𝖺𝗅𝗂𝗓𝖾formulae-sequence𝖫𝗂𝗇𝖾𝖺𝗋𝖽𝗎𝖺𝗅𝗂𝗓𝖾\mathsf{Linear}\mathsf{.dualize} ramps up the stable rank of updates, so the weights should move a non-trivial relative amount at large width even in the Frobenius norm—provided the batch size is not too small.

## 7 Conclusion

This paper has proposed a recursive procedure called modular dualization for building duality maps for general neural architectures. The procedure unifies past strands of optimization research on Shampoo (Gupta et al., [2018](#bib.bib18)) and μ𝜇\muP (Yang & Hu, [2021](#bib.bib36)). Partial implementations have already led to significant wall-clock speedups in transformer training (Jordan, [2024](#bib.bib23)). Our rectangular Newton-Schulz iteration provides a GPU-friendly and numerically stable means of dualizing under the RMS→RMS→RMSRMS\mathrm{RMS}\to\mathrm{RMS} operator norm, while avoiding some of the downsides of sketching-based approaches (Carlson et al., [2015b](#bib.bib8)). Overall, we hope that our theory of modular duality provides a clarifying toolkit for the design and analysis of deep learning systems.

## Acknowledgements

Many ideas in this paper were developed jointly with Tim Large before he left to work at a tech company.
We are grateful to Phillip Isola for invaluable discussions. We also thank Jack Gallagher, Keller Jordan, Simo Ryu, Rogier Brussee, Tongzhou Wang, Victor Butoi, Jeffrey Cider and Volkan Cevher for helpful conversations.

## References

* Amari (2016)

  Shun-ichi Amari.
  *Information Geometry and Its Applications*.
  Springer, 2016.
* Anil et al. (2020)

  Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, and Yoram Singer.
  Scalable second order optimization for deep learning.
  *arXiv:2002.09018*, 2020.
* Bernstein & Newhouse (2024)

  Jeremy Bernstein and Laker Newhouse.
  Old optimizer, new norm: An anthology.
  In *Workshop on Optimization for Machine Learning*, 2024.
* Bernstein et al. (2023)

  Jeremy Bernstein, Chris Mingard, Kevin Huang, Navid Azizan, and Yisong Yue.
  Automatic Gradient Descent: Deep Learning without Hyperparameters.
  *arXiv:2304.05187*, 2023.
* Boyd & Vandenberghe (2004)

  Stephen Boyd and Lieven Vandenberghe.
  *Convex Optimization*.
  Cambridge University Press, 2004.
* Carlson et al. (2015a)

  David Carlson, Volkan Cevher, and Lawrence Carin.
  Stochastic spectral descent for restricted Boltzmann machines.
  In *International Conference on Artificial Intelligence and Statistics*, 2015a.
* Carlson et al. (2016)

  David Carlson, Ya-Ping Hsieh, Edo Collins, Lawrence Carin, and Volkan Cevher.
  Stochastic spectral descent for discrete graphical models.
  *Selected Topics in Signal Processing*, 2016.
* Carlson et al. (2015b)

  David E. Carlson, Edo Collins, Ya-Ping Hsieh, Lawrence Carin, and Volkan Cevher.
  Preconditioned spectral descent for deep learning.
  In *Neural Information Processing Systems*, 2015b.
* Carroll (2019)

  Sean M. Carroll.
  *Spacetime and Geometry: An Introduction to General Relativity*.
  Cambridge University Press, 2019.
* Dahl et al. (2023)

  George E. Dahl, Frank Schneider, Zachary Nado, Naman Agarwal, Chandramouli Shama Sastry, Philipp Hennig, Sourabh Medapati, Runa Eschenhagen, Priya Kasimbeg, Daniel Suo, Juhan Bae, Justin Gilmer, Abel L. Peirson, Bilal Khan, Rohan Anil, Mike Rabbat, Shankar Krishnan, Daniel Snider, Ehsan Amid, Kongtao Chen, Chris J. Maddison, Rakshith Vasudev, Michal Badura, Ankush Garg, and Peter Mattson.
  Benchmarking neural network training algorithms.
  *arXiv:2306.07179*, 2023.
* Deimling (1985)

  Klaus Deimling.
  *Nonlinear Functional Analysis*.
  Springer Berlin, Heidelberg, 1985.
* Everett et al. (2024)

  Katie E. Everett, Lechao Xiao, Mitchell Wortsman, Alexander A. Alemi, Roman Novak, Peter J. Liu, Izzeddin Gur, Jascha Sohl-Dickstein, Leslie Pack Kaelbling, Jaehoon Lee, and Jeffrey Pennington.
  Scaling exponents across parameterizations and optimizers.
  In *International Conference on Machine Learning*, 2024.
* Feinberg et al. (2023)

  Vladimir Feinberg, Xinyi Chen, Y. Jennifer Sun, Rohan Anil, and Elad Hazan.
  Sketchy: Memory-efficient adaptive regularization with frequent directions.
  In *Neural Information Processing Systems*, 2023.
* Flynn (2017)

  Thomas Flynn.
  The duality structure gradient descent algorithm: Analysis and applications to neural networks.
  *arXiv:1708.00523*, 2017.
* Fong & Spivak (2019)

  Brendan Fong and David I. Spivak.
  *An Invitation to Applied Category Theory: Seven Sketches in Compositionality*.
  Cambridge University Press, 2019.
* Grant (2004)

  Michael Charles Grant.
  *Disciplined Convex Programming*.
  PhD dissertation, Stanford University, 2004.
* Grosse (2022)

  Roger Grosse.
  Metrics.
  Lecture 3 of CSC2541: Neural Net Training Dynamics, 2022.
* Gupta et al. (2018)

  Vineet Gupta, Tomer Koren, and Yoram Singer.
  Shampoo: Preconditioned stochastic tensor optimization.
  In *International Conference on Machine Learning*, 2018.
* Haskell Wiki Contributors (2007)

  Haskell Wiki Contributors.
  Combinator pattern.
  Haskell Wiki, 2007.
  URL <https://wiki.haskell.org/Combinator_pattern>.
* Higham (2008)

  Nicholas J. Higham.
  *Functions of Matrices*.
  Society for Industrial and Applied Mathematics, 2008.
* Jacot et al. (2018)

  Arthur Jacot, Franck Gabriel, and Clement Hongler.
  Neural tangent kernel: Convergence and generalization in neural networks.
  In *Neural Information Processing Systems*, 2018.
* Jesus et al. (2021)

  Ricardo J. Jesus, Mário L. Antunes, Rui A. da Costa, Sergey N. Dorogovtsev, José F. F. Mendes, and Rui L. Aguiar.
  Effect of initial configuration of weights on training and function of artificial neural networks.
  *Mathematics*, 2021.
* Jordan (2024)

  Keller Jordan.
  New training speed record for @karpathy’s 124M-parameter NanoGPT setup: 3.28 Fineweb validation loss in 3.7B training tokens.
  <https://x.com/kellerjordan0/status/1842300916864844014>, 2024.
* Lakić (1998)

  Slobodan Lakić.
  On the computation of the matrix k-th root.
  *Journal of Applied Mathematics and Mechanics*, 1998.
* Large et al. (2024)

  Tim Large, Yang Liu, Minyoung Huh, Hyojin Bahng, Phillip Isola, and Jeremy Bernstein.
  Scalable optimization in the modular norm.
  In *Neural Information Processing Systems*, 2024.
* Lee et al. (2019)

  Jaehoon Lee, Lechao Xiao, Samuel Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, and Jeffrey Pennington.
  Wide neural networks of any depth evolve as linear models under gradient descent.
  In *Neural Information Processing Systems*, 2019.
* Martinsson & Tropp (2020)

  Per-Gunnar Martinsson and Joel A. Tropp.
  Randomized numerical linear algebra: Foundations and algorithms.
  *Acta Numerica*, 2020.
* Nemirovsky & Yudin (1983)

  Arkady S. Nemirovsky and David B. Yudin.
  *Problem complexity and method efficiency in optimization*.
  Wiley, 1983.
* OPT (2024)

  OPT.
  Optimization for Machine Learning, 2024.
  URL <https://opt-ml.org/>.
* Sakurai & Napolitano (2020)

  J. J. Sakurai and Jim Napolitano.
  *Modern Quantum Mechanics*.
  Cambridge University Press, 2020.
* Shi et al. (2023)

  Hao-Jun Michael Shi, Tsung-Hsien Lee, Shintaro Iwasaki, Jose Gallego-Posada, Zhijing Li, Kaushik Rangadurai, Dheevatsa Mudigere, and Michael Rabbat.
  A distributed data-parallel PyTorch implementation of the distributed Shampoo optimizer for training neural networks at-scale.
  *arXiv:2309.06497*, 2023.
* Streeter (2023)

  Matthew Streeter.
  Universal majorization-minimization algorithms.
  *arXiv:2308.00190*, 2023.
* Streeter & Dillon (2022)

  Matthew J. Streeter and Joshua V. Dillon.
  Automatically bounding the Taylor remainder series: Tighter bounds and new applications.
  *arXiv:2212.11429*, 2022.
* Tieleman & Hinton (2012)

  Tijmen Tieleman and Geoffrey Hinton.
  RMSprop.
  *Coursera: Neural Networks for Machine Learning*, Lecture 6.5, 2012.
* Tran et al. (2015)

  Dung T. Tran, Nobutaka Ono, and Emmanuel Vincent.
  Fast DNN training based on auxiliary function technique.
  *International Conference on Acoustics, Speech and Signal Processing*, 2015.
* Yang & Hu (2021)

  Greg Yang and Edward J. Hu.
  Tensor programs IV: Feature learning in infinite-width neural networks.
  In *International Conference on Machine Learning*, 2021.
* Yang et al. (2023)

  Greg Yang, James B. Simon, and Jeremy Bernstein.
  A spectral condition for feature learning.
  *arXiv:2310.17813*, 2023.
