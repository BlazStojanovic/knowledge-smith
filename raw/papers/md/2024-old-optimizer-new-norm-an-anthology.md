---
arxiv: '2409.20325'
authors:
- Jeremy Bernstein
- Laker Newhouse
parser: ar5iv
retrieved: '2026-05-09'
source: paper
title: 'Old Optimizer, New Norm: An Anthology'
url: https://arxiv.org/abs/2409.20325
year: 2024
---

[2409.20325] Old Optimizer, New Norm: An Anthology














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



\SetAlCapSkip

1em
\SetKwCommentComment##\# 
\optauthor\NameJeremy Bernstein \Emailjbernstein@mit.edu
  
\NameLaker Newhouse \Emaillakern@mit.edu
  
\addrMIT CSAIL, United States

# Old Optimizer, New Norm: An Anthology

###### Abstract

Deep learning optimizers are often motivated through a mix of convex and approximate second-order theory. We select three such methods—Adam, Shampoo and Prodigy—and argue that each method can instead be understood as a squarely first-order method without convexity assumptions. In fact, after switching off exponential moving averages, each method is equivalent to steepest descent under a particular norm. By generalizing this observation, we chart a new design space for training algorithms. Different operator norms should be assigned to different tensors based on the role that the tensor plays within the network. For example, while linear and embedding layers may have the same weight space of ℝm×nsuperscriptℝ𝑚𝑛\mathbb{R}^{m\times n}, these layers play different roles and should be assigned different norms. We hope that this idea of carefully metrizing the neural architecture might lead to more stable, scalable and indeed faster training.

## Prologue

Deep learning optimizers are often motivated from the perspectives of convex and approximate second-order theory. These theoretical frameworks have been used to inspire algorithmic ideas, as well as providing means to analyse the convergence of various optimizers. However, we believe—and will attempt to demonstrate—that there is a wealth of untapped algorithmic opportunity in the simpler realm of exact first-order theory without convexity assumptions.

To make our case, we choose three optimizers that were originally analysed under convex or approximate second-order theory: Adam, Shampoo and Prodigy. After disabling their exponential moving averages (EMA), we show that each algorithm admits a parsimonious theoretical explanation as a variant of steepest descent under a certain norm. EMA can then be thought of as “smoothing out” the algorithm, or making it more robust to mini-batch noise, although nailing down the precise role of EMA is perhaps still an open problem.

By steepest descent, we mean the procedure of choosing a weight update Δ​𝒘Δ𝒘\Delta{\bm{w}} to minimise a local quadratic model of the loss function ℒℒ\mathcal{L} of the form ℒ​(𝒘)+∇𝒘ℒ​(𝒘)⊤​Δ​𝒘+λ2⋅‖Δ​𝒘‖2ℒ𝒘subscript∇𝒘ℒsuperscript𝒘topΔ𝒘⋅𝜆2superscriptnormΔ𝒘2\mathcal{L}({\bm{w}})+\nabla\_{\bm{w}}\mathcal{L}({\bm{w}})^{\top}\Delta{\bm{w}}+\frac{\lambda}{2}\cdot\|{\Delta{\bm{w}}}\|^{2}, visualized in [Figure 1](#Sx1.F1 "In Prologue ‣ Old Optimizer, New Norm: An Anthology"). Crucially, the sharpness parameter λ𝜆\lambda and norm ∥⋅∥\|{\cdot}\| are chosen a priori, without touching an (approximate) Hessian during training. As such, we consider steepest descent to be a squarely first-order method and not an (approximate) second-order method.

Throughout the anthology, we rely on a dual description of steepest descent:

###### Proposition 1 (Steepest descent)

For any 𝐠∈ℝn𝐠superscriptℝ𝑛{\bm{g}}\in\mathbb{R}^{n} thought of as “the gradient” and any λ≥0𝜆0\lambda\geq 0 thought of as “the sharpness”, and for any norm ∥⋅∥:ℝn→ℝ\|{\cdot}\|:\mathbb{R}^{n}\to\mathbb{R} with dual norm ∥⋅∥†\|{\cdot}\|^{\dagger}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | arg​minΔ​𝒘∈ℝn⁡[𝒈⊤​Δ​𝒘+λ2​‖Δ​𝒘‖2]=−‖𝒈‖†λ⋅arg​max‖𝒕‖=1⁡𝒈⊤​𝒕.subscriptargminΔ𝒘superscriptℝ𝑛superscript𝒈topΔ𝒘𝜆2superscriptnormΔ𝒘2⋅superscriptnorm𝒈†𝜆subscriptargmaxnorm𝒕1superscript𝒈top𝒕\displaystyle\operatorname\*{arg\,min}\_{\Delta{\bm{w}}\in\mathbb{R}^{n}}\left[{\bm{g}}^{\top}\Delta{\bm{w}}+\frac{\lambda}{2}\,\|{\Delta{\bm{w}}}\|^{2}\right]=-\frac{\|{{\bm{g}}}\|^{\dagger}}{\lambda}\cdot\operatorname\*{arg\,max}\_{\|{{\bm{t}}}\|=1}{\bm{g}}^{\top}{\bm{t}}. |  | (1) |

[Equation 1](#Sx1.E1 "In Proposition 1 (Steepest descent) ‣ Prologue ‣ Old Optimizer, New Norm: An Anthology") separates the solution of the steepest descent problem into two pieces: first computing the step size as the dual norm of the gradient divided by the sharpness, and second solving for the step direction as the unit vector that maximizes the inner product with the gradient. The proof of this proposition is given in [Appendix B](#A2.SSx1 "1: \nameref*prop:steepest ‣ Appendix B Proofs ‣ Old Optimizer, New Norm: An Anthology").

![Refer to caption](/html/2409.20325/assets/x1.png)


Figure 1: Steepest descent considers the problem of minimizing a linear functional under a quadratic penalty: arg​minΔ​𝒘∈ℝn⁡[𝒈⊤​Δ​𝒘+λ2​‖Δ​𝒘‖2]subscriptargminΔ𝒘superscriptℝ𝑛superscript𝒈topΔ𝒘𝜆2superscriptnormΔ𝒘2\operatorname\*{arg\,min}\_{\Delta{\bm{w}}\in\mathbb{R}^{n}}\left[{\bm{g}}^{\top}\Delta{\bm{w}}+\frac{\lambda}{2}\,\|{\Delta{\bm{w}}}\|^{2}\right] for 𝒈∈ℝn𝒈superscriptℝ𝑛{\bm{g}}\in\mathbb{R}^{n}. Here we show how the solution varies with the sharpness λ>0𝜆0\lambda>0 and the choice of norm ∥⋅∥\|{\cdot}\|. We overlay different norm balls on top of a linear color gradient, and use arrows to denote the solution, meaning the member of the norm ball that “minimizes the color”. a) Increasing the sharpness decreases the size of the solution vector. b) Changing the norm can change the direction of the solution vector. For different ℓpsubscriptℓ𝑝\ell\_{p} norms, the solution direction changes because the gradient is not axis-aligned. In practice, we should pick the sharpness and norm to fit the geometry of our loss.

Of course, the art of steepest descent lies in choosing a norm ∥⋅∥\|{\cdot}\| and a sharpness λ𝜆\lambda suited to the optimization problem at hand. While it may be possible to turn this art into a science (Large et al., [2024](#bib.bib24)), that ambition is beyond the scope of this anthology. Here we point out that past methods do implicitly make decisions about norms, and in a somewhat haphazard manner. In fact, they implicitly assign different induced matrix norms to the network layers:

###### Definition 1 (Induced operator norm)

Given a matrix 𝐌∈ℝdout×din𝐌superscriptℝsubscript𝑑outsubscript𝑑in{\bm{M}}\in\mathbb{R}^{d\_{\mathrm{out}}\times d\_{\mathrm{in}}} and two normed vector spaces (ℝdin,∥⋅∥α)(\mathbb{R}^{d\_{\mathrm{in}}},\|{\cdot}\|\_{\alpha}) and (ℝdout,∥⋅∥β)(\mathbb{R}^{d\_{\mathrm{out}}},\|{\cdot}\|\_{\beta}), the “α𝛼\alpha to β𝛽\beta” induced operator norm is given by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖𝑴‖α→β=max𝒙∈ℝdin⁡‖𝑴​𝒙‖β‖𝒙‖α.subscriptnorm𝑴→𝛼𝛽subscript  𝒙superscriptℝsubscript𝑑insubscriptnorm𝑴𝒙𝛽subscriptnorm𝒙𝛼\|{{\bm{M}}}\|\_{\alpha\to\beta}=\max\_{\begin{subarray}{c}{\bm{x}}\in\mathbb{R}^{d\_{\mathrm{in}}}\end{subarray}}\frac{\|{{\bm{M}}{\bm{x}}}\|\_{\beta}}{\|{{\bm{x}}}\|\_{\alpha}}. |  | (2) |

[1](#Thmmydefinition1 "Definition 1 (Induced operator norm) ‣ Prologue ‣ Old Optimizer, New Norm: An Anthology") tells us that by varying the choice of vector norms ∥⋅∥α\|{\cdot}\|\_{\alpha} and ∥⋅∥β\|{\cdot}\|\_{\beta}, we can induce a large family of matrix norms. In turn, this implies a correspondingly large family of steepest descent optimizers. By foregrounding this issue, we hope that algorithm designers may develop more suitable optimizers by becoming more intentional about their choice of norm.

## Story I Adam as Steepest Descent under the Max-of-Max Norm

Adam is a widely used deep learning optimizer: the original paper of Kingma and Ba ([2015](#bib.bib21)) now has well over 100,000 citations. Adam has been motivated in various ways, including through convex analysis (Kingma and Ba, [2015](#bib.bib21)) and as an approximate second-order method (Sun and Spall, [2021](#bib.bib31)). However, there have been efforts to build a more direct understanding of Adam: for instance, with exponential moving averages (EMA) switched off, Adam is just sign gradient descent (Balles and Hennig, [2018](#bib.bib3); Bernstein et al., [2018](#bib.bib4)), which is equivalent to steepest descent under the infinity norm (Carlson et al., [2015](#bib.bib6)). In this story, we connect Adam to a certain “max-of-max” norm, showing how Adam respects the tensor structure of a neural network in a very particular way.

To begin, we review how Adam connects to sign gradient descent. Ignoring bias corrections and numerical stabilizations, Adam is given by the following system of updates:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒎tsubscript𝒎𝑡\displaystyle{\bm{m}}\_{t} | =β1⋅𝒎t−1+(1−β1)⋅𝒈t,absent⋅subscript𝛽1subscript𝒎𝑡1⋅1subscript𝛽1subscript𝒈𝑡\displaystyle=\beta\_{1}\cdot{\bm{m}}\_{t-1}+(1-\beta\_{1})\cdot{\bm{g}}\_{t}, |  | (3) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒗tsubscript𝒗𝑡\displaystyle{\bm{v}}\_{t} | =β2⋅𝒗t−1+(1−β2)⋅𝒈t2,absent⋅subscript𝛽2subscript𝒗𝑡1⋅1subscript𝛽2superscriptsubscript𝒈𝑡2\displaystyle=\beta\_{2}\cdot{\bm{v}}\_{t-1}+(1-\beta\_{2})\cdot{\bm{g}}\_{t}^{2}, |  | (4) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒘t+1subscript𝒘𝑡1\displaystyle{\bm{w}}\_{t+1} | =𝒘t−η⋅𝒎t/𝒗t,absentsubscript𝒘𝑡⋅𝜂subscript𝒎𝑡subscript𝒗𝑡\displaystyle={\bm{w}}\_{t}-\eta\cdot{\bm{m}}\_{t}/\sqrt{{\bm{v}}\_{t}}, |  | (5) |

where t𝑡t denotes the time step, 𝒈t∈ℝnsubscript𝒈𝑡superscriptℝ𝑛{\bm{g}}\_{t}\in\mathbb{R}^{n} the gradient vector and η>0𝜂0\eta>0 the step size. The EMA time scales of the first gradient moment 𝒎tsubscript𝒎𝑡{\bm{m}}\_{t} and second moment 𝒗tsubscript𝒗𝑡{\bm{v}}\_{t} are set by 0≤β1,β2<1formulae-sequence0subscript𝛽1subscript𝛽210\leq\beta\_{1},\beta\_{2}<1. All operations are conducted entry-wise. If we switch off EMA by setting β1=β2=0subscript𝛽1subscript𝛽20\beta\_{1}=\beta\_{2}=0, the Adam updates reduce to just sign gradient descent:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒘t+1subscript𝒘𝑡1\displaystyle{\bm{w}}\_{t+1} | =𝒘t−η⋅𝒈t/𝒈t2absentsubscript𝒘𝑡⋅𝜂subscript𝒈𝑡superscriptsubscript𝒈𝑡2\displaystyle={\bm{w}}\_{t}-\eta\cdot{\bm{g}}\_{t}/\sqrt{{\bm{g}}\_{t}^{2}} |  | (6) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =𝒘t−η⋅sign⁡(𝒈t).absentsubscript𝒘𝑡⋅𝜂signsubscript𝒈𝑡\displaystyle={\bm{w}}\_{t}-\eta\cdot\operatorname{sign}({\bm{g}}\_{t}). |  | (7) |

This connection to sign descent should not be surprising since Adam, published in 2015, builds on the RMSprop optimizer that Tieleman and Hinton ([2012](#bib.bib32)) already called “the mini-batch version of just using the sign of the gradient”. And RMSprop itself built on the RPROP optimizer (Riedmiller and Braun, [1993](#bib.bib28)), which also uses gradient signs.

Still, why should using the sign of the gradient be a good idea in deep learning? In search of a motivation, it is interesting to consider that sign descent solves the problem of steepest descent under the vector ℓ∞subscriptℓ\ell\_{\infty} norm, ‖𝒗‖∞:=maxi⁡|𝒗i|assignsubscriptnorm𝒗subscript𝑖subscript𝒗𝑖\|{{\bm{v}}}\|\_{\infty}\vcentcolon=\max\_{i}|{{\bm{v}}\_{i}}| (Carlson et al., [2015](#bib.bib6), [2016](#bib.bib7); Fan, [2017](#bib.bib13)):

###### Proposition 2 (Sign descent as steepest descent under the infinity norm)

For any gradient vector 𝐠∈ℝn𝐠superscriptℝ𝑛{\bm{g}}\in\mathbb{R}^{n} and sharpness λ>0𝜆0\lambda>0, it holds that:

|  |  |  |  |
| --- | --- | --- | --- |
|  | arg​minΔ​𝒘∈ℝn⁡[𝒈⊤​Δ​𝒘+λ2​‖Δ​𝒘‖∞2]=−‖𝒈‖1λ​sign⁡(𝒈).subscriptargminΔ𝒘superscriptℝ𝑛superscript𝒈topΔ𝒘𝜆2superscriptsubscriptnormΔ𝒘2subscriptnorm𝒈1𝜆sign𝒈\operatorname\*{arg\,min}\_{\Delta{\bm{w}}\in\mathbb{R}^{n}}\left[{\bm{g}}^{\top}\Delta{\bm{w}}+\frac{\lambda}{2}\,\|{\Delta{\bm{w}}}\|\_{\infty}^{2}\right]=-\frac{\|{{\bm{g}}}\|\_{1}}{\lambda}\,\operatorname{sign}({\bm{g}}). |  | (8) |

In words, the vector that minimizes a linear functional under an infinity norm penalty is a scalar multiple of a sign vector. The proof is given in [Appendix B](#A2.SSx2 "2: \nameref*prop:sign-descent ‣ Appendix B Proofs ‣ Old Optimizer, New Norm: An Anthology").

While this connection between Adam, sign descent and steepest descent is perhaps cute, it does not answer a basic question: Why does the vector ℓ∞subscriptℓ\ell\_{\infty} norm have anything to do with neural network training? In particular, taking the weight space to be ℝnsuperscriptℝ𝑛\mathbb{R}^{n} equipped with the simple infinity norm seems to “throw away” the fact that the weight space of a neural network is built in a structured way out of layers of matrices (and perhaps other tensors).

To resolve this conundrum, we suggest that in fact the vector ℓ∞subscriptℓ\ell\_{\infty} norm on the flattened weight space doesn’t have anything to do with deep learning. Instead, there is a coincidence at play. The ℓ∞subscriptℓ\ell\_{\infty} norm enjoys a special property summarized by the slogan “a max of a max is a max”. To see this, consider a neural network with a list of L𝐿L weight matrices 𝑾1,…,𝑾L

subscript𝑾1…subscript𝑾𝐿{\bm{W}}\_{1},\dots,{\bm{W}}\_{L}. Let rowr​(𝑾l)subscriptrow𝑟subscript𝑾𝑙\mathrm{row}\_{r}({\bm{W}}\_{l}) denote the r𝑟rth row of the l𝑙lth weight matrix, and let 𝒘=flatten⁡(𝑾1,…,𝑾L)∈ℝn𝒘flattensubscript𝑾1…subscript𝑾𝐿superscriptℝ𝑛{\bm{w}}=\operatorname{flatten}({\bm{W}}\_{1},\dots,{\bm{W}}\_{L})\in\mathbb{R}^{n} denote the full flattened weight vector. Then we have that:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖𝒘‖∞=maxl⁡maxr⁡‖rowr​(𝑾l)‖∞=maxl⁡‖𝑾l‖ℓ1→ℓ∞,subscriptnorm𝒘subscript𝑙subscript𝑟subscriptnormsubscriptrow𝑟subscript𝑾𝑙subscript𝑙subscriptnormsubscript𝑾𝑙→subscriptℓ1subscriptℓ\|{{\bm{w}}}\|\_{\infty}=\max\_{l}\max\_{r}\|{\mathrm{row}\_{r}({\bm{W}}\_{l})}\|\_{\infty}=\max\_{l}\|{{\bm{W}}\_{l}}\|\_{\ell\_{1}\to\ell\_{\infty}}, |  | (9) |

where the second equality follows via [8](#Thmmyproposition8 "Proposition 8 (ℓ₁→ℓ_𝑝 and ℓ_𝑝→ℓ_∞ induced operator norms are tractable) ‣ Epilogue ‣ Old Optimizer, New Norm: An Anthology"). In words, the infinity norm of the flattened weight vector coincides with the largest ℓ1subscriptℓ1\ell\_{1} to ℓ∞subscriptℓ\ell\_{\infty} operator norm of the layers. So [Equation 9](#S1.E9 "In Story I Adam as Steepest Descent under the Max-of-Max Norm ‣ Old Optimizer, New Norm: An Anthology") connects the unstructured space of the flattened weight vector to the structured space of the list of weight matrices. We refer to the object maxl⁡‖𝑾l‖ℓ1→ℓ∞subscript𝑙subscriptnormsubscript𝑾𝑙→subscriptℓ1subscriptℓ\max\_{l}\|{{\bm{W}}\_{l}}\|\_{\ell\_{1}\to\ell\_{\infty}} as the “max-of-max norm”. And sign descent emerges as steepest descent under this norm:

###### Proposition 3 (Sign descent as steepest descent under the max-of-max norm)

For any list of gradient matrices 𝐆1,…,𝐆L

subscript𝐆1…subscript𝐆𝐿{\bm{G}}\_{1},...,{\bm{G}}\_{L} and any sharpness λ>0𝜆0\lambda>0, consider the problem:

|  |  |  |  |
| --- | --- | --- | --- |
|  | arg​minΔ​𝑾1,…,Δ​𝑾L⁡[∑l=1L⟨𝑮l,Δ​𝑾l⟩+λ2​maxl=1L⁡‖Δ​𝑾l‖ℓ1→ℓ∞2],subscriptargmin  Δsubscript𝑾1…Δsubscript𝑾𝐿superscriptsubscript𝑙1𝐿  subscript𝑮𝑙Δsubscript𝑾𝑙𝜆2superscriptsubscript𝑙1𝐿superscriptsubscriptnormΔsubscript𝑾𝑙→subscriptℓ1subscriptℓ2\operatorname\*{arg\,min}\_{\Delta{\bm{W}}\_{1},...,\Delta{\bm{W}}\_{L}}\left[\sum\_{l=1}^{L}\langle{\bm{G}}\_{l},\Delta{\bm{W}}\_{l}\rangle+\frac{\lambda}{2}\max\_{l=1}^{L}\|{\Delta{\bm{W}}\_{l}}\|\_{\ell\_{1}\to\ell\_{\infty}}^{2}\right], |  | (10) |

where ⟨⋅,⋅⟩

⋅⋅\langle\cdot,\cdot\rangle denotes the Frobenius inner product, and Δ​𝐖lΔsubscript𝐖𝑙\Delta{\bm{W}}\_{l} has the same shape as 𝐆lsubscript𝐆𝑙{\bm{G}}\_{l}. For step size η=1λ​∑l=1L‖𝐆l‖ℓ1→ℓ∞†𝜂1𝜆superscriptsubscript𝑙1𝐿superscriptsubscriptnormsubscript𝐆𝑙→subscriptℓ1subscriptℓ†\eta=\frac{1}{\lambda}\sum\_{l=1}^{L}\|{{\bm{G}}\_{l}}\|\_{\ell\_{1}\to\ell\_{\infty}}^{\dagger}, where ††\dagger denotes the dual norm, [Equation 10](#S1.E10 "In Proposition 3 (Sign descent as steepest descent under the max-of-max norm) ‣ Story I Adam as Steepest Descent under the Max-of-Max Norm ‣ Old Optimizer, New Norm: An Anthology") is solved by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​𝑾l=−η⋅sign⁡(𝑮l) for each layer ​l=1,…,L.formulae-sequenceΔsubscript𝑾𝑙⋅𝜂signsubscript𝑮𝑙 for each layer 𝑙  1…𝐿\Delta{\bm{W}}\_{l}=-\eta\cdot\operatorname{sign}({\bm{G}}\_{l})\qquad\text{ for each layer }l=1,...,L. |  | (11) |

In words, the matrix-aware steepest descent problem of [Equation 10](#S1.E10 "In Proposition 3 (Sign descent as steepest descent under the max-of-max norm) ‣ Story I Adam as Steepest Descent under the Max-of-Max Norm ‣ Old Optimizer, New Norm: An Anthology") is solved by layerwise sign descent as given in [Equation 11](#S1.E11 "In Proposition 3 (Sign descent as steepest descent under the max-of-max norm) ‣ Story I Adam as Steepest Descent under the Max-of-Max Norm ‣ Old Optimizer, New Norm: An Anthology"). This observation—that sign descent updates are implicitly doing per-matrix gradient normalization—may be a major reason that Adam, sign descent and Lion (Chen et al., [2023](#bib.bib9)) outperform vanilla gradient descent in large language model training (Zhao et al., [2024](#bib.bib34); Large et al., [2024](#bib.bib24)). The proof is given in [Appendix B](#A2.SSx3 "3: \nameref*prop:structural-sign-descent ‣ Appendix B Proofs ‣ Old Optimizer, New Norm: An Anthology").

\pgfornament

[width=0.3]82

All told, this story has shown that Adam without EMA is sign descent and that, coincidentally, sign descent solves two different steepest descent problems: one on the flattened weight space, and one that is aware of the matrix structure of neural architecture. But, at the end of this story, questions linger. Why does the ℓ1subscriptℓ1\ell\_{1} to ℓ∞subscriptℓ\ell\_{\infty} induced operator norm rear its head? What does it have to do with deep learning? Aren’t there other induced operator norms on matrices we could equally well consider? For answers to these questions, dear reader, you’ll have to wait for our next story… a story about Shampoo!

## Story II Shampoo as Steepest Descent under the Spectral Norm

Now, dear reader, we turn our attention to Shampoo (Gupta et al., [2017](#bib.bib15), [2018](#bib.bib16)). A variant of the Shampoo optimizer won the external tuning track of the 2024 AlgoPerf: Training Algorithms competition (Dahl et al., [2023](#bib.bib10)). While the method was originally motivated as a generalization of the AdaGrad convex optimizer (Duchi et al., [2011](#bib.bib12)) to tensor spaces, more recent work casts Shampoo as an approximate second-order method (Anil et al., [2020](#bib.bib1); Morwani et al., [2024](#bib.bib27)). We will show that Shampoo—with accumulation disabled—is steepest descent under the max spectral norm over layers.

To begin, we show that Shampoo updates, without accumulation, are semi-orthogonal matrices. At time step t𝑡t and for each layer, Shampoo collects the gradient matrix 𝑮tsubscript𝑮𝑡{\bm{G}}\_{t} and makes the following update to the weight matrix 𝑾tsubscript𝑾𝑡{\bm{W}}\_{t}:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝑳tsubscript𝑳𝑡\displaystyle{\bm{L}}\_{t} | =𝑳t−1+𝑮t​𝑮tT,absentsubscript𝑳𝑡1subscript𝑮𝑡superscriptsubscript𝑮𝑡𝑇\displaystyle={\bm{L}}\_{t-1}+{\bm{G}}\_{t}{\bm{G}}\_{t}^{T}, |  | (12) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝑹tsubscript𝑹𝑡\displaystyle{\bm{R}}\_{t} | =𝑹t−1+𝑮tT​𝑮t,absentsubscript𝑹𝑡1superscriptsubscript𝑮𝑡𝑇subscript𝑮𝑡\displaystyle=\smash{{\bm{R}}\_{t-1}+{\bm{G}}\_{t}^{T}{\bm{G}}\_{t}}, |  | (13) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝑾t+1subscript𝑾𝑡1\displaystyle{\bm{W}}\_{t+1} | =𝑾t−η⋅𝑳t−1/4​𝑮t​𝑹t−1/4.absentsubscript𝑾𝑡⋅𝜂superscriptsubscript𝑳𝑡14subscript𝑮𝑡superscriptsubscript𝑹𝑡14\displaystyle=\smash{{\bm{W}}\_{t}-\eta\cdot{\bm{L}}\_{t}^{-\nicefrac{{1}}{{4}}}{\bm{G}}\_{t}{\bm{R}}\_{t}^{-\nicefrac{{1}}{{4}}}}. |  | (14) |

All operations, including the inverse fourth roots, are matrix operations. The accumulators 𝑳tsubscript𝑳𝑡{\bm{L}}\_{t} and 𝑹tsubscript𝑹𝑡{\bm{R}}\_{t} are referred to as the “left and right pre-conditioners”. Practitioners usually replace the simple sums in [Equations 12](#S2.E12 "In Story II Shampoo as Steepest Descent under the Spectral Norm ‣ Old Optimizer, New Norm: An Anthology") and [13](#S2.E13 "Equation 13 ‣ Story II Shampoo as Steepest Descent under the Spectral Norm ‣ Old Optimizer, New Norm: An Anthology") with EMAs (Shi et al., [2023](#bib.bib29)). If we disable the accumulation, setting 𝑳t=𝑮t​𝑮t⊤subscript𝑳𝑡subscript𝑮𝑡superscriptsubscript𝑮𝑡top{\bm{L}}\_{t}={\bm{G}}\_{t}{\bm{G}}\_{t}^{\top} and 𝑹t=𝑮t⊤​𝑮tsubscript𝑹𝑡superscriptsubscript𝑮𝑡topsubscript𝑮𝑡{\bm{R}}\_{t}={\bm{G}}\_{t}^{\top}{\bm{G}}\_{t}, Shampoo reduces to:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝑾t+1subscript𝑾𝑡1\displaystyle{\bm{W}}\_{t+1} | =𝑾t−η⋅(𝑮t​𝑮t⊤)−1/4​𝑮t​(𝑮t⊤​𝑮t)−1/4absentsubscript𝑾𝑡⋅𝜂superscriptsubscript𝑮𝑡superscriptsubscript𝑮𝑡top14subscript𝑮𝑡superscriptsuperscriptsubscript𝑮𝑡topsubscript𝑮𝑡14\displaystyle=\smash{{\bm{W}}\_{t}-\eta\cdot({\bm{G}}\_{t}{\bm{G}}\_{t}^{\top})^{-\nicefrac{{1}}{{4}}}\,{\bm{G}}\_{t}\,({\bm{G}}\_{t}^{\top}{\bm{G}}\_{t})^{-\nicefrac{{1}}{{4}}}} |  | (15) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =𝑾t−η⋅𝑼t​𝑽t⊤,absentsubscript𝑾𝑡⋅𝜂subscript𝑼𝑡superscriptsubscript𝑽𝑡top\displaystyle=\smash{{\bm{W}}\_{t}-\eta\cdot{\bm{U}}\_{t}{\bm{V}}\_{t}^{\top}}, |  | (16) |

where [Equation 16](#S2.E16 "In Story II Shampoo as Steepest Descent under the Spectral Norm ‣ Old Optimizer, New Norm: An Anthology") is reached by substituting the reduced singular value decomposition (SVD) of the gradient 𝑮t=𝑼t​𝚺t​𝑽t⊤subscript𝑮𝑡subscript𝑼𝑡subscript𝚺𝑡superscriptsubscript𝑽𝑡top{\bm{G}}\_{t}={\bm{U}}\_{t}{\bm{\Sigma}}\_{t}{\bm{V}}\_{t}^{\top} into [Equation 15](#S2.E15 "In Story II Shampoo as Steepest Descent under the Spectral Norm ‣ Old Optimizer, New Norm: An Anthology"). Notice that there is a direct parallel between [Equations 6](#S1.E6 "In Story I Adam as Steepest Descent under the Max-of-Max Norm ‣ Old Optimizer, New Norm: An Anthology") and [7](#S1.E7 "Equation 7 ‣ Story I Adam as Steepest Descent under the Max-of-Max Norm ‣ Old Optimizer, New Norm: An Anthology") for Adam and [Equations 15](#S2.E15 "In Story II Shampoo as Steepest Descent under the Spectral Norm ‣ Old Optimizer, New Norm: An Anthology") and [16](#S2.E16 "Equation 16 ‣ Story II Shampoo as Steepest Descent under the Spectral Norm ‣ Old Optimizer, New Norm: An Anthology") for Shampoo. So, Shampoo without accumulation makes a semi-orthogonal weight update. In fact:

###### Proposition 4 (Projection to the closest semi-orthogonal matrix)

Consider the
  
semi-orthogonal matrices 𝒪m×n:={𝐀∈ℝm×n:𝐀​𝐀⊤=𝐈m​ or ​𝐀⊤​𝐀=𝐈n}assignsubscript𝒪𝑚𝑛conditional-set𝐀superscriptℝ𝑚𝑛𝐀superscript𝐀topsubscript𝐈𝑚 or superscript𝐀top𝐀subscript𝐈𝑛\mathcal{O}\_{m\times n}\vcentcolon=\left\{{\bm{A}}\in\mathbb{R}^{m\times n}:{\bm{A}}{\bm{A}}^{\top}=\mathbf{I}\_{m}\text{ or }{\bm{A}}^{\top}{\bm{A}}=\mathbf{I}\_{n}\right\} and let ∥⋅∥F\|{\cdot}\|\_{F} denote the Frobenius norm. For any matrix 𝐆∈ℝm×n𝐆superscriptℝ𝑚𝑛{\bm{G}}\in\mathbb{R}^{m\times n} with reduced SVD 𝐆=𝐔​𝚺​𝐕⊤𝐆𝐔𝚺superscript𝐕top{\bm{G}}={\bm{U}}{\bm{\Sigma}}{\bm{V}}^{\top}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | arg​min𝑨∈𝒪m×n⁡‖𝑨−𝑮‖F=𝑼​𝑽⊤,subscriptargmin𝑨subscript𝒪𝑚𝑛subscriptnorm𝑨𝑮𝐹𝑼superscript𝑽top\operatorname\*{arg\,min}\_{{\bm{A}}\in\mathcal{O}\_{m\times n}}\|{{\bm{A}}-{\bm{G}}}\|\_{F}={\bm{U}}{\bm{V}}^{\top}, |  | (17) |

where the minimizer 𝐔​𝐕⊤𝐔superscript𝐕top{\bm{U}}{\bm{V}}^{\top} is unique if and only if the matrix 𝐆𝐆{\bm{G}} has full rank.

So, Shampoo without accumulation projects the gradient matrix to the closest semi-orthogonal matrix in Frobenius norm. The proof is in [Appendix B](#A2.SSx4 "4: \nameref*prop:projection ‣ Appendix B Proofs ‣ Old Optimizer, New Norm: An Anthology"). Why might this be a good idea, you ask? Well, for one thing, it’s steepest descent—this time under the maximum spectral norm ∥⋅∥ℓ2→ℓ2\|{\cdot}\|\_{\ell\_{2}\to\ell\_{2}} ([1](#Thmmydefinition1 "Definition 1 (Induced operator norm) ‣ Prologue ‣ Old Optimizer, New Norm: An Anthology")) over all the matrices in the network:

###### Proposition 5 (Shampoo as steepest descent under the spectral norm)

For any list of gradient matrices 𝐆1,…,𝐆L

subscript𝐆1…subscript𝐆𝐿{\bm{G}}\_{1},...,{\bm{G}}\_{L} and any sharpness λ>0𝜆0\lambda>0, consider the problem:

|  |  |  |  |
| --- | --- | --- | --- |
|  | arg​minΔ​𝑾1,…,Δ​𝑾L⁡[∑l=1L⟨𝑮l,Δ​𝑾l⟩+λ2​maxl=1L⁡‖Δ​𝑾l‖ℓ2→ℓ22],subscriptargmin  Δsubscript𝑾1…Δsubscript𝑾𝐿superscriptsubscript𝑙1𝐿  subscript𝑮𝑙Δsubscript𝑾𝑙𝜆2superscriptsubscript𝑙1𝐿superscriptsubscriptnormΔsubscript𝑾𝑙→subscriptℓ2subscriptℓ22\operatorname\*{arg\,min}\_{\Delta{\bm{W}}\_{1},...,\Delta{\bm{W}}\_{L}}\left[\sum\_{l=1}^{L}\langle{\bm{G}}\_{l},\Delta{\bm{W}}\_{l}\rangle+\frac{\lambda}{2}\,\max\_{l=1}^{L}\|{\Delta{\bm{W}}\_{l}}\|\_{\ell\_{2}\to\ell\_{2}}^{2}\right], |  | (18) |

where ⟨⋅,⋅⟩

⋅⋅\langle\cdot,\cdot\rangle denotes the Frobenius inner product and Δ​𝐖lΔsubscript𝐖𝑙\Delta{\bm{W}}\_{l} has the same shape as 𝐆lsubscript𝐆𝑙{\bm{G}}\_{l}. Suppose that 𝐆lsubscript𝐆𝑙{\bm{G}}\_{l} has reduced SVD given by 𝐆l=𝐔l​𝚺l​𝐕l⊤subscript𝐆𝑙subscript𝐔𝑙subscript𝚺𝑙superscriptsubscript𝐕𝑙top{\bm{G}}\_{l}={\bm{U}}\_{l}{\bm{\Sigma}}\_{l}{\bm{V}}\_{l}^{\top} for each l=1,…,L𝑙

1…𝐿l=1,...,L. Then [Equation 18](#S2.E18 "In Proposition 5 (Shampoo as steepest descent under the spectral norm) ‣ Story II Shampoo as Steepest Descent under the Spectral Norm ‣ Old Optimizer, New Norm: An Anthology") is solved with a step size η=1λ​∑l=1Ltr⁡𝚺l𝜂1𝜆superscriptsubscript𝑙1𝐿trsubscript𝚺𝑙\eta=\frac{1}{\lambda}\sum\_{l=1}^{L}\operatorname{tr}{\bm{\Sigma}}\_{l} and an update:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​𝑾l=−η⋅𝑼l​𝑽l⊤ for each ​l=1,…,L.formulae-sequenceΔsubscript𝑾𝑙⋅𝜂subscript𝑼𝑙superscriptsubscript𝑽𝑙top for each 𝑙  1…𝐿\Delta{\bm{W}}\_{l}=-\eta\cdot{\bm{U}}\_{l}{\bm{V}}\_{l}^{\top}\quad\text{ for each }l=1,...,L. |  | (19) |

This solution for Δ​𝐖lΔsubscript𝐖𝑙\Delta{\bm{W}}\_{l} is unique if and only if the matrix 𝐆lsubscript𝐆𝑙{\bm{G}}\_{l} is of full rank.

The proof is given in [Appendix B](#A2.SSx5 "5: \nameref*prop:shampoo-steepest ‣ Appendix B Proofs ‣ Old Optimizer, New Norm: An Anthology"). A novelty of this proposition in contrast to prior work on stochastic spectral descent (Carlson et al., [2015](#bib.bib6), [2016](#bib.bib7)) is our use of a max norm over layers to handle the multi-layer case. However, our main contribution here is to draw the connection between [5](#Thmmyproposition5 "Proposition 5 (Shampoo as steepest descent under the spectral norm) ‣ Story II Shampoo as Steepest Descent under the Spectral Norm ‣ Old Optimizer, New Norm: An Anthology") and Shampoo as in [Equations 15](#S2.E15 "In Story II Shampoo as Steepest Descent under the Spectral Norm ‣ Old Optimizer, New Norm: An Anthology") and [16](#S2.E16 "Equation 16 ‣ Story II Shampoo as Steepest Descent under the Spectral Norm ‣ Old Optimizer, New Norm: An Anthology").

So, Shampoo without accumulation is steepest descent under the spectral norm. Why might this be a good idea in deep learning? The idea that we wish to advance is that one can derive upper bounds on the loss of machine learning models in terms of spectral norms. Here we present the simplest possible example: a linear model and the square loss.

###### Proposition 6 (Bounding the square loss of a linear predictor)

Consider a matrix 𝐖∈ℝdout×din𝐖superscriptℝsubscript𝑑outsubscript𝑑in{\bm{W}}\in\mathbb{R}^{d\_{\mathrm{out}}\times d\_{\mathrm{in}}} that we shall think of as a linear predictor mapping an input 𝐱∈ℝdin𝐱superscriptℝsubscript𝑑in{\bm{x}}\in\mathbb{R}^{d\_{\mathrm{in}}} to an output 𝐲=𝐖​𝐱∈ℝdout𝐲𝐖𝐱superscriptℝsubscript𝑑out{\bm{y}}={\bm{W}}{\bm{x}}\in\mathbb{R}^{d\_{\mathrm{out}}}. Given a dataset of n𝑛n samples 𝒟={(𝐱1,𝐲1),…,(𝐱n,𝐲n)}𝒟subscript𝐱1subscript𝐲1…subscript𝐱𝑛subscript𝐲𝑛\mathcal{D}=\{({\bm{x}}\_{1},{\bm{y}}\_{1}),...,({\bm{x}}\_{n},{\bm{y}}\_{n})\}, where the i𝑖ith input is normalized such that ‖𝐱i‖2=dinsubscriptnormsubscript𝐱𝑖2subscript𝑑in\|{\bm{x}}\_{i}\|\_{2}=\sqrt{d\_{\mathrm{in}}}, we can construct the “square loss”:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ​(𝑾):=12​n​∑i=1n1dout​‖𝒚i−𝑾​𝒙i‖22.assignℒ𝑾12𝑛superscriptsubscript𝑖1𝑛1subscript𝑑outsuperscriptsubscriptnormsubscript𝒚𝑖𝑾subscript𝒙𝑖22\mathcal{L}({\bm{W}})\vcentcolon=\frac{1}{2n}\sum\_{i=1}^{n}\frac{1}{d\_{\mathrm{out}}}\|{{\bm{y}}\_{i}-{\bm{W}}{\bm{x}}\_{i}}\|\_{2}^{2}. |  | (20) |

Then, for any matrix Δ​𝐖∈ℝdout×dinΔ𝐖superscriptℝsubscript𝑑outsubscript𝑑in\Delta{\bm{W}}\in\mathbb{R}^{d\_{\mathrm{out}}\times d\_{\mathrm{in}}} thought of as a weight update, it holds that:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ​(𝑾+Δ​𝑾)≤ℒ​(𝑾)+⟨∇𝑾ℒ​(𝑾),Δ​𝑾⟩+12⋅dindout⋅‖Δ​𝑾‖ℓ2→ℓ22,ℒ𝑾Δ𝑾ℒ𝑾  subscript∇𝑾ℒ𝑾Δ𝑾 ⋅12subscript𝑑insubscript𝑑outsuperscriptsubscriptnormΔ𝑾→subscriptℓ2subscriptℓ22\mathcal{L}({\bm{W}}+\Delta{\bm{W}})\leq\mathcal{L}({\bm{W}})+\langle\nabla\_{\bm{W}}\mathcal{L}({\bm{W}}),\Delta{\bm{W}}\rangle+\tfrac{1}{2}\cdot\tfrac{d\_{\mathrm{in}}}{d\_{\mathrm{out}}}\cdot\|{\Delta{\bm{W}}}\|\_{\mathrm{\ell\_{2}\to\ell\_{2}}}^{2}, |  | (21) |

where ⟨⋅,⋅⟩

⋅⋅\langle\cdot,\cdot\rangle is the Frobenius inner product.

In words: the square loss of a linear predictor admits an upper bound that is quadratic in the spectral norm of the weight perturbation. Choosing the weight perturbation to minimize this upper bound is precisely steepest descent under the spectral norm! The proof is given in [Appendix B](#A2.SSx6 "6: \nameref*prop:majorization ‣ Appendix B Proofs ‣ Old Optimizer, New Norm: An Anthology"). This optimizer design pattern, which starts by deriving an upper bound on the loss (as in [6](#Thmmyproposition6 "Proposition 6 (Bounding the square loss of a linear predictor) ‣ Story II Shampoo as Steepest Descent under the Spectral Norm ‣ Old Optimizer, New Norm: An Anthology")) and then minimizes it (as in [5](#Thmmyproposition5 "Proposition 5 (Shampoo as steepest descent under the spectral norm) ‣ Story II Shampoo as Steepest Descent under the Spectral Norm ‣ Old Optimizer, New Norm: An Anthology")), is known generally as majorization-minimization (Lange, [2016](#bib.bib23)). It is an exact and first-principles design pattern, without Hessian approximations or appeals to convex theory. This design pattern is used extensively by Carlson et al. ([2015](#bib.bib6), [2016](#bib.bib7)) to design optimizers for restricted Boltzmann machines and discrete graphical models. Generalizing the pattern to arbitrary network architectures and loss functions requires more advanced machinery (Bernstein et al., [2023](#bib.bib5); Streeter, [2023](#bib.bib30); Large et al., [2024](#bib.bib24)).

\pgfornament

[width=0.3]82

And so, dear reader, we have reached the end of our second story. We have shown that Shampoo without accumulation corresponds to projecting the gradient matrix to the closest semi-orthogonal matrix, which solves the problem of steepest descent under the spectral norm. And we showed how steepest descent under the spectral norm emerges from upper bounding the square loss of a linear predictor. This perspective, of viewing Shampoo as a (smoothed out) projection to the space of semi-orthogonal matrices, grounds the algorithm in a prior literature on spectral descent (Carlson et al., [2015](#bib.bib6), [2016](#bib.bib7); Fan, [2017](#bib.bib13)). And in [Appendix A](#A1 "Appendix A Computational Strategies for Shampoo ‣ Old Optimizer, New Norm: An Anthology"), we discuss how it might unlock new means for computing the Shampoo updates.

We summarize our first two stories in [Table 1](#S2.T1 "In Story II Shampoo as Steepest Descent under the Spectral Norm ‣ Old Optimizer, New Norm: An Anthology"). And we still have one more left to tell…

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Domain | Norm | Solution | Optimizer | Cousin |
| ℝnsuperscriptℝ𝑛\mathbb{R}^{n} | Euclidean ℓ2subscriptℓ2\ell\_{2} | Δ​𝒘=−‖𝒈‖2λ​𝒈‖𝒈‖2Δ𝒘subscriptnorm𝒈2𝜆𝒈subscriptnorm𝒈2\displaystyle\Delta{\bm{w}}=-\tfrac{\|{{\bm{g}}}\|\_{2}}{\lambda}\,\tfrac{{\bm{g}}}{\|{{\bm{g}}}\|\_{2}} | vanilla gradient descent | SGD |
| ℝnsuperscriptℝ𝑛\mathbb{R}^{n} | infinity ℓ∞subscriptℓ\ell\_{\infty} | Δ​𝒘=−‖𝒈‖1λ​sign⁡(𝒈)Δ𝒘subscriptnorm𝒈1𝜆sign𝒈\displaystyle\Delta{\bm{w}}=-\tfrac{\|{{\bm{g}}}\|\_{1}}{\lambda}\operatorname{sign}({\bm{g}}) | sign descent | Adam |
| ℝm×nsuperscriptℝ𝑚𝑛\mathbb{R}^{m\times n} | Frobenius S2subscript𝑆2S\_{2} | Δ​𝑾=−‖𝑮‖Fλ​𝑮‖𝑮‖FΔ𝑾subscriptnorm𝑮𝐹𝜆𝑮subscriptnorm𝑮𝐹\displaystyle\Delta{\bm{W}}=-\tfrac{\|{{\bm{G}}}\|\_{F}}{\lambda}\,\tfrac{{\bm{G}}}{\|{{\bm{G}}}\|\_{F}} | vanilla gradient descent | SGD |
| ℝm×nsuperscriptℝ𝑚𝑛\mathbb{R}^{m\times n} | spectral S∞subscript𝑆S\_{\infty} | Δ​𝑾=−tr⁡𝚺λ​𝑼​𝑽⊤Δ𝑾tr𝚺𝜆𝑼superscript𝑽top\displaystyle\Delta{\bm{W}}=-\tfrac{\operatorname{tr}{\bm{\Sigma}}}{\lambda}\,{\bm{U}}{\bm{V}}^{\top} | spectral descent | Shampoo |

Table 1: Popular optimizers are related to steepest descent under different norms. For vector-valued optimization problems, we consider the steepest descent problem arg​minΔ​𝒘⁡𝒈⊤​Δ​𝒘+λ2⋅‖Δ​𝒘‖2subscriptargminΔ𝒘superscript𝒈topΔ𝒘⋅𝜆2superscriptnormΔ𝒘2\smash{\operatorname\*{arg\,min}\_{\Delta{\bm{w}}}{\bm{g}}^{\top}\Delta{\bm{w}}+\frac{\lambda}{2}\cdot\|{\Delta{\bm{w}}}\|^{2}}. For matrix-valued problems, we consider arg​minΔ​𝑾⁡⟨𝑮,Δ​𝑾⟩+λ2⋅‖Δ​𝑾‖2subscriptargminΔ𝑾𝑮Δ𝑾⋅𝜆2superscriptnormΔ𝑾2\smash{\operatorname\*{arg\,min}\_{\Delta{\bm{W}}}\,\langle{\bm{G}},\Delta{\bm{W}}\rangle+\frac{\lambda}{2}\cdot\|{\Delta{\bm{W}}}\|^{2}}, where ⟨⋅,⋅⟩

⋅⋅\langle\cdot,\cdot\rangle is the Frobenius inner product. We list the solution for different vector ℓpsubscriptℓ𝑝\ell\_{p} norms and Schatten Spsubscript𝑆𝑝S\_{p} norms. The Schatten Spsubscript𝑆𝑝S\_{p} norm of a matrix returns the ℓpsubscriptℓ𝑝\ell\_{p} norm of its vector of singular values. Finally, 𝑮=𝑼​𝚺​𝑽⊤𝑮𝑼𝚺superscript𝑽top{\bm{G}}={\bm{U}}{\bm{\Sigma}}{\bm{V}}^{\top} is the reduced singular value decomposition of the gradient.

## Story III Prodigy: Automatically Computing the Escape Velocity

For our final story, we speak of Prodigy (Mishchenko and Defazio, [2023](#bib.bib26)). The Prodigy optimizer falls amid a series of recent works (Defazio and Mishchenko, [2023](#bib.bib11); Khaled et al., [2023](#bib.bib20); Ivgi et al., [2023](#bib.bib18)) that attempt to apply convex theory to design and analyse deep learning optimizers that do not require tuning. In contrast, we argue that Prodigy (without EMA) is but another example of steepest descent, where instead of using the step size η=‖𝒈‖†/λ𝜂superscriptnorm𝒈†𝜆\eta=\|{{\bm{g}}}\|^{\dagger}/\lambda from [1](#Thmmyproposition1 "Proposition 1 (Steepest descent) ‣ Prologue ‣ Old Optimizer, New Norm: An Anthology"), Prodigy uses a heuristic to automatically warm up to a good step size. This demonstrates the value of [1](#Thmmyproposition1 "Proposition 1 (Steepest descent) ‣ Prologue ‣ Old Optimizer, New Norm: An Anthology") for disentangling the optimizer design problem. If one knows a good norm ∥⋅∥\|{\cdot}\| but is ignorant of the sharpness parameter λ𝜆\lambda, then one may obtain the step direction by solving arg​max‖𝒕‖=1⁡𝒈⊤​𝒕subscriptargmaxnorm𝒕1superscript𝒈top𝒕\operatorname\*{arg\,max}\_{\|{{\bm{t}}}\|=1}{\bm{g}}^{\top}{\bm{t}} from [1](#Thmmyproposition1 "Proposition 1 (Steepest descent) ‣ Prologue ‣ Old Optimizer, New Norm: An Anthology"), while using another means to find a good step size.

Then let us make our case. We focus on Algorithm 3 in the Prodigy paper, since this is the version used in their experiments. We first show that with EMA switched off, Prodigy implements sign gradient descent with a step size that warms up automatically. Ignoring the numerical stabilization and learning rate schedule, Prodigy is given by:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒎tsubscript𝒎𝑡\displaystyle{\bm{m}}\_{t} | =β1⋅𝒎t−1+(1−β1)⋅ηt​𝒈t,absent⋅subscript𝛽1subscript𝒎𝑡1⋅1subscript𝛽1subscript𝜂𝑡subscript𝒈𝑡\displaystyle=\beta\_{1}\cdot{\bm{m}}\_{t-1}+(1-\beta\_{1})\cdot\eta\_{t}\,{\bm{g}}\_{t}, |  | (22) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒗tsubscript𝒗𝑡\displaystyle{\bm{v}}\_{t} | =β2⋅𝒗t−1+(1−β2)⋅ηt2​𝒈t2,absent⋅subscript𝛽2subscript𝒗𝑡1⋅1subscript𝛽2superscriptsubscript𝜂𝑡2superscriptsubscript𝒈𝑡2\displaystyle=\beta\_{2}\cdot{\bm{v}}\_{t-1}+(1-\beta\_{2})\cdot\eta\_{t}^{2}\,{\bm{g}}\_{t}^{2}, |  | (23) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | rtsubscript𝑟𝑡\displaystyle r\_{t} | =β2⋅rt−1+(1−β2)⋅ηt2​𝒈t⊤​(𝒘0−𝒘t),absent⋅subscript𝛽2subscript𝑟𝑡1⋅1subscript𝛽2superscriptsubscript𝜂𝑡2superscriptsubscript𝒈𝑡topsubscript𝒘0subscript𝒘𝑡\displaystyle=\sqrt{\beta\_{2}}\cdot r\_{t-1}+(1-\sqrt{\beta\_{2}})\cdot\eta\_{t}^{2}\,{\bm{g}}\_{t}^{\top}({\bm{w}}\_{0}-{\bm{w}}\_{t}), |  | (24) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒔tsubscript𝒔𝑡\displaystyle{\bm{s}}\_{t} | =β2⋅𝒔t−1+(1−β2)⋅ηt2​𝒈t,absent⋅subscript𝛽2subscript𝒔𝑡1⋅1subscript𝛽2superscriptsubscript𝜂𝑡2subscript𝒈𝑡\displaystyle=\sqrt{\beta\_{2}}\cdot{\bm{s}}\_{t-1}+(1-\sqrt{\beta\_{2}})\cdot\eta\_{t}^{2}\,{\bm{g}}\_{t}, |  | (25) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ηt+1subscript𝜂𝑡1\displaystyle\eta\_{t+1} | =max⁡(ηt,rt‖𝒔t‖1),absentsubscript𝜂𝑡subscript𝑟𝑡subscriptnormsubscript𝒔𝑡1\displaystyle=\max\left(\eta\_{t},\tfrac{r\_{t}}{\|{{\bm{s}}\_{t}}\|\_{1}}\right), |  | (26) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒘t+1subscript𝒘𝑡1\displaystyle{\bm{w}}\_{t+1} | =𝒘t−ηt⋅𝒎t/𝒗t,absentsubscript𝒘𝑡⋅subscript𝜂𝑡subscript𝒎𝑡subscript𝒗𝑡\displaystyle={\bm{w}}\_{t}-\eta\_{t}\cdot{\bm{m}}\_{t}/\sqrt{{\bm{v}}\_{t}}, |  | (27) |

where t𝑡t denotes the time step and 𝒈t∈ℝnsubscript𝒈𝑡superscriptℝ𝑛{\bm{g}}\_{t}\in\mathbb{R}^{n} the gradient vector. While this system of updates may seem intimidating, if we switch off EMA by setting β1=β2=0subscript𝛽1subscript𝛽20\beta\_{1}=\beta\_{2}=0, the Prodigy updates simplify dramatically to just sign gradient descent with a dynamical step size as follows:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ηt+1subscript𝜂𝑡1\displaystyle\eta\_{t+1} | =max⁡(ηt,𝒈t⊤​(𝒘0−𝒘t)‖𝒈t‖1),absentsubscript𝜂𝑡superscriptsubscript𝒈𝑡topsubscript𝒘0subscript𝒘𝑡subscriptnormsubscript𝒈𝑡1\displaystyle=\max\left(\eta\_{t},\tfrac{{\bm{g}}\_{t}^{\top}({\bm{w}}\_{0}-{\bm{w}}\_{t})}{\|{{\bm{g}}\_{t}}\|\_{1}}\right), |  | (28) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒘t+1subscript𝒘𝑡1\displaystyle{\bm{w}}\_{t+1} | =𝒘t−ηt⋅sign⁡(𝒈t).absentsubscript𝒘𝑡⋅subscript𝜂𝑡signsubscript𝒈𝑡\displaystyle={\bm{w}}\_{t}-\eta\_{t}\cdot\operatorname{sign}({\bm{g}}\_{t}). |  | (29) |

But [2](#Thmmyproposition2 "Proposition 2 (Sign descent as steepest descent under the infinity norm) ‣ Story I Adam as Steepest Descent under the Max-of-Max Norm ‣ Old Optimizer, New Norm: An Anthology") showed that sign descent is steepest descent under the infinity norm. Therefore [Equations 28](#S3.E28 "In Story III Prodigy: Automatically Computing the Escape Velocity ‣ Old Optimizer, New Norm: An Anthology") and [29](#S3.E29 "Equation 29 ‣ Story III Prodigy: Automatically Computing the Escape Velocity ‣ Old Optimizer, New Norm: An Anthology") prove our claim that Prodigy without EMA is steepest descent, although with a dynamically chosen step size denoted ηtsubscript𝜂𝑡\eta\_{t}.

All that remains is to understand the dynamical rule, given by [Equation 28](#S3.E28 "In Story III Prodigy: Automatically Computing the Escape Velocity ‣ Old Optimizer, New Norm: An Anthology"), for choosing the step size ηtsubscript𝜂𝑡\eta\_{t}. We shall argue that this dynamical rule can be understood to approximate a heuristic algorithm for achieving, but not exceeding, what we shall call escape velocity:

* •

  Choose a very small initial step size η0subscript𝜂0\eta\_{0}—small enough to be a priori sure that η0≪η⋆much-less-thansubscript𝜂0subscript𝜂⋆\eta\_{0}\ll\eta\_{\star}, where η⋆subscript𝜂⋆\eta\_{\star} denotes escape velocity: the unknown but optimal initial step size;
* •

  At each step, check if the weights 𝒘tsubscript𝒘𝑡{\bm{w}}\_{t} have escaped the linearization of the loss around the initial weights 𝒘0subscript𝒘0{\bm{w}}\_{0}—if not, double the step size according to ηt+1=2×ηtsubscript𝜂𝑡12subscript𝜂𝑡\eta\_{t+1}=2\times\eta\_{t};
* •

  Once the weights 𝒘tsubscript𝒘𝑡{\bm{w}}\_{t} have escaped the initial linearization, stop increasing the step size. We say that the step size ηtsubscript𝜂𝑡\eta\_{t} has reached escape velocity η⋆subscript𝜂⋆\eta\_{\star}.

The rationale behind this procedure is that if we knew the optimal initial step size η⋆subscript𝜂⋆\eta\_{\star}, then the weights should escape the initial linearization of the loss in a single step. Formally, the directional derivative (𝒘1−𝒘0)⊤​𝒈1superscriptsubscript𝒘1subscript𝒘0topsubscript𝒈1({\bm{w}}\_{1}-{\bm{w}}\_{0})^{\top}{\bm{g}}\_{1} must vanish if the step size is chosen optimally (Cauchy, [1847](#bib.bib8)). If the directional derivative in the direction of the first weight update is still negative (𝒘1−𝒘0)⊤​𝒈1<0superscriptsubscript𝒘1subscript𝒘0topsubscript𝒈10({\bm{w}}\_{1}-{\bm{w}}\_{0})^{\top}{\bm{g}}\_{1}<0, then we could have taken a larger step. Said another way, we can use the angle that the gradient 𝒈1subscript𝒈1{\bm{g}}\_{1} makes with the change in weights 𝒘1−𝒘0subscript𝒘1subscript𝒘0{\bm{w}}\_{1}-{\bm{w}}\_{0} to tell us whether or not we should increase the step size. Notice that procedure has no reliance on convexity.

With this in mind, let us massage Prodigy’s step size update ([Equation 28](#S3.E28 "In Story III Prodigy: Automatically Computing the Escape Velocity ‣ Old Optimizer, New Norm: An Anthology")) as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ηt+1=max⁡(ηt,𝒈t⊤​(𝒘0−𝒘t)‖𝒈t‖1)=max⁡(ηt,‖𝒈t‖2‖𝒈t‖1×‖𝒘t−𝒘0‖2×cos⁡θ),subscript𝜂𝑡1subscript𝜂𝑡superscriptsubscript𝒈𝑡topsubscript𝒘0subscript𝒘𝑡subscriptnormsubscript𝒈𝑡1subscript𝜂𝑡subscriptnormsubscript𝒈𝑡2subscriptnormsubscript𝒈𝑡1subscriptnormsubscript𝒘𝑡subscript𝒘02𝜃\displaystyle\eta\_{t+1}=\max\left(\eta\_{t},\tfrac{{\bm{g}}\_{t}^{\top}({\bm{w}}\_{0}-{\bm{w}}\_{t})}{\|{{\bm{g}}\_{t}}\|\_{1}}\right)=\max\left(\eta\_{t},\tfrac{\|{{\bm{g}}\_{t}}\|\_{2}}{\|{{\bm{g}}\_{t}}\|\_{1}}\times\|{{\bm{w}}\_{t}-{\bm{w}}\_{0}}\|\_{2}\times\cos\theta\right), |  | (30) |

where θ𝜃\theta denotes the angle between the gradient 𝒈tsubscript𝒈𝑡{\bm{g}}\_{t} and the difference in weights 𝒘0−𝒘tsubscript𝒘0subscript𝒘𝑡{\bm{w}}\_{0}-{\bm{w}}\_{t}. To help make sense of this expression, we make two assumptions:

1. 1.

   The gradient is a “dense” vector in ℝnsuperscriptℝ𝑛\mathbb{R}^{n}, meaning that ‖𝒈t‖2/‖𝒈t‖1≈1/nsubscriptnormsubscript𝒈𝑡2subscriptnormsubscript𝒈𝑡11𝑛\|{{\bm{g}}\_{t}}\|\_{2}/\|{{\bm{g}}\_{t}}\|\_{1}\approx 1/\sqrt{n};
2. 2.

   𝒘tsubscript𝒘𝑡{\bm{w}}\_{t} is still close enough to the initialization 𝒘0subscript𝒘0{\bm{w}}\_{0} that cos⁡θ≈1𝜃1\cos\theta\approx 1.

Under these assumptions, [Equation 30](#S3.E30 "In Story III Prodigy: Automatically Computing the Escape Velocity ‣ Old Optimizer, New Norm: An Anthology") becomes just ηt+1≈max⁡(ηt,‖𝒘t−𝒘0‖RMS)subscript𝜂𝑡1subscript𝜂𝑡subscriptnormsubscript𝒘𝑡subscript𝒘0RMS\eta\_{t+1}\approx\max\left(\eta\_{t},\|{{\bm{w}}\_{t}-{\bm{w}}\_{0}}\|\_{\mathrm{RMS}}\right), where the root mean square (RMS) norm is defined via ∥⋅∥RMS:=1n∥⋅∥2\|{\cdot}\|\_{\mathrm{RMS}}\vcentcolon=\tfrac{1}{\sqrt{n}}\,\|{\cdot}\|\_{2}. Combined with [Equation 29](#S3.E29 "In Story III Prodigy: Automatically Computing the Escape Velocity ‣ Old Optimizer, New Norm: An Anthology"), this allows us to estimate the size of the weight change at step t+1𝑡1t+1:

|  |  |  |
| --- | --- | --- |
|  | ‖𝒘t+2−𝒘t+1‖RMS=ηt+1⋅‖sign⁡(𝒈t)‖RMS≈max⁡(ηt,‖𝒘t−𝒘0‖RMS)≥‖𝒘t−𝒘0‖RMS,subscriptnormsubscript𝒘𝑡2subscript𝒘𝑡1RMS⋅subscript𝜂𝑡1subscriptnormsignsubscript𝒈𝑡RMSsubscript𝜂𝑡subscriptnormsubscript𝒘𝑡subscript𝒘0RMSsubscriptnormsubscript𝒘𝑡subscript𝒘0RMS\|{{\bm{w}}\_{t+2}-{\bm{w}}\_{t+1}}\|\_{\mathrm{RMS}}=\eta\_{t+1}\cdot\|{\operatorname{sign}({\bm{g}}\_{t})}\|\_{\mathrm{RMS}}\approx\max\left(\eta\_{t},\|{{\bm{w}}\_{t}-{\bm{w}}\_{0}}\|\_{\mathrm{RMS}}\right)\geq\|{{\bm{w}}\_{t}-{\bm{w}}\_{0}}\|\_{\mathrm{RMS}}, |  |

where we have used the fact that a sign vector has unit RMS norm. In words, while assumptions ([1](#S3.I2.i1 "Item 1 ‣ Story III Prodigy: Automatically Computing the Escape Velocity ‣ Old Optimizer, New Norm: An Anthology")) and ([2](#S3.I2.i2 "Item 2 ‣ Story III Prodigy: Automatically Computing the Escape Velocity ‣ Old Optimizer, New Norm: An Anthology")) hold, the step size at time t+1𝑡1t+1 is equivalent to the whole progress up to step t𝑡t. This suggests exponential growth in the step size that continues until assumption ([2](#S3.I2.i2 "Item 2 ‣ Story III Prodigy: Automatically Computing the Escape Velocity ‣ Old Optimizer, New Norm: An Anthology")) breaks, which we think of as the step size reaching the escape velocity η⋆subscript𝜂⋆\eta\_{\star}

Now we wish to point out that this procedure is just one amongst a space of line search methods that one might consider (Armijo, [1966](#bib.bib2); Riedmiller and Braun, [1993](#bib.bib28); Kenneweg et al., [2024](#bib.bib19)). For instance, Prodigy’s decision to only let ηtsubscript𝜂𝑡\eta\_{t} increase and never decrease could be sub-optimal. And the decision to measure the angle between the gradient and the weight difference 𝒘t−𝒘0subscript𝒘𝑡subscript𝒘0{\bm{w}}\_{t}-{\bm{w}}\_{0} has alternatives. One could instead use the most recent weight difference 𝒘t−𝒘t−1subscript𝒘𝑡subscript𝒘𝑡1{\bm{w}}\_{t}-{\bm{w}}\_{t-1}. Lastly, in place of relying on the norm ratio ‖𝒈‖2/‖𝒈1‖subscriptnorm𝒈2normsubscript𝒈1\|{{\bm{g}}}\|\_{2}/\|{{\bm{g}}\_{1}}\| to implicitly convert the ℓ2subscriptℓ2\ell\_{2} norm ‖𝒘t−𝒘0‖2subscriptnormsubscript𝒘𝑡subscript𝒘02\|{{\bm{w}}\_{t}-{\bm{w}}\_{0}}\|\_{2} into the RMS norm ‖𝒘t−𝒘0‖RMSsubscriptnormsubscript𝒘𝑡subscript𝒘0RMS\|{{\bm{w}}\_{t}-{\bm{w}}\_{0}}\|\_{\mathrm{RMS}}, one could consider a more explicit method. For instance, we found a rule akin to ηt+1=ηt×(1+cos⁡θ)subscript𝜂𝑡1subscript𝜂𝑡1𝜃\eta\_{t+1}=\eta\_{t}\times(1+\cos\theta) to work well in some preliminary experiments.

\pgfornament

[width=0.3]82

Our time grows short, dear reader, and our third story draws to an end. We have argued that Prodigy without EMA is sign descent—an example of steepest descent—with a particular mechanism for warming up the step size. Starting with a tiny initial step size, Prodigy multiplicatively increases the step size until the weights escape the initial locally linear region of the loss. Prodigy’s step size adjustment is based on the angle between the gradient and the total weight change. This is a form of online line search. This highlights that once one has a chosen a norm, the steepest descent framework allows freedom to estimate the step size in various different ways.

## Epilogue

This anthology has presented new ways of understanding old optimizers. [1](#Thmmyproposition1 "Proposition 1 (Steepest descent) ‣ Prologue ‣ Old Optimizer, New Norm: An Anthology") decouples the optimizer design problem into two pieces: first choosing a norm and second finding a step size. This design space is already broad. We have argued that Adam chooses the infinity norm ([2](#Thmmyproposition2 "Proposition 2 (Sign descent as steepest descent under the infinity norm) ‣ Story I Adam as Steepest Descent under the Max-of-Max Norm ‣ Old Optimizer, New Norm: An Anthology")) or equivalently the max-of-max norm ([3](#Thmmyproposition3 "Proposition 3 (Sign descent as steepest descent under the max-of-max norm) ‣ Story I Adam as Steepest Descent under the Max-of-Max Norm ‣ Old Optimizer, New Norm: An Anthology")), which respects a layered matrix structure. Shampoo chooses the spectral norm ([5](#Thmmyproposition5 "Proposition 5 (Shampoo as steepest descent under the spectral norm) ‣ Story II Shampoo as Steepest Descent under the Spectral Norm ‣ Old Optimizer, New Norm: An Anthology")). Prodigy chooses the same norm as Adam, and then uses a heuristic to automatically warm up to a good step size, as in [Equation 28](#S3.E28 "In Story III Prodigy: Automatically Computing the Escape Velocity ‣ Old Optimizer, New Norm: An Anthology"), which we term reaching escape velocity.

Through the lens of steepest descent, the decisions that Adam, Shampoo and Prodigy make may seem arbitrary. In fact, we think that they are somewhat arbitrary. And there may be more principled ways to make these decisions. To demonstrate this point, we now introduce a tool called the modular norm (Large et al., [2024](#bib.bib24)) and its corresponding steepest descent algorithm. The modular norm generalizes the norms that appeared in [3](#Thmmyproposition3 "Proposition 3 (Sign descent as steepest descent under the max-of-max norm) ‣ Story I Adam as Steepest Descent under the Max-of-Max Norm ‣ Old Optimizer, New Norm: An Anthology") for Adam and [5](#Thmmyproposition5 "Proposition 5 (Shampoo as steepest descent under the spectral norm) ‣ Story II Shampoo as Steepest Descent under the Spectral Norm ‣ Old Optimizer, New Norm: An Anthology") for Shampoo. Formally:

###### Proposition 7 (Steepest descent under the modular norm)

Given scalar coefficients s1,…,sL>0

subscript𝑠1…subscript𝑠𝐿
0s\_{1},\dots,s\_{L}>0 and norms ∥⋅∥1,…,∥⋅∥L\|{\cdot}\|\_{1},\dots,\|{\cdot}\|\_{L}, we define the modular norm as the mapping:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝑾1,…,𝑾L↦max⁡{s1​‖𝑾1‖1,…,sL​‖𝑾L‖L}.maps-to  subscript𝑾1…subscript𝑾𝐿 subscript𝑠1subscriptnormsubscript𝑾11…subscript𝑠𝐿subscriptnormsubscript𝑾𝐿𝐿{\bm{W}}\_{1},\dots,{\bm{W}}\_{L}\mapsto\max\left\{s\_{1}\|{{\bm{W}}\_{1}}\|\_{1},\dots,s\_{L}\|{{\bm{W}}\_{L}}\|\_{L}\right\}. |  | (31) |

The corresponding steepest descent problem is given by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | arg​minΔ​𝑾1,…,Δ​𝑾L⁡[∑l=1L⟨𝑮l,Δ​𝑾l⟩+λ2​maxl=1L⁡sl2​‖Δ​𝑾l‖l2],subscriptargmin  Δsubscript𝑾1…Δsubscript𝑾𝐿superscriptsubscript𝑙1𝐿  subscript𝑮𝑙Δsubscript𝑾𝑙𝜆2superscriptsubscript𝑙1𝐿superscriptsubscript𝑠𝑙2superscriptsubscriptnormΔsubscript𝑾𝑙𝑙2\operatorname\*{arg\,min}\_{\Delta{\bm{W}}\_{1},\dots,\Delta{\bm{W}}\_{L}}\left[\sum\_{l=1}^{L}\langle{\bm{G}}\_{l},\Delta{\bm{W}}\_{l}\rangle+\frac{\lambda}{2}\max\_{l=1}^{L}s\_{l}^{2}\|{\Delta{\bm{W}}\_{l}}\|\_{l}^{2}\right], |  | (32) |

where ⟨⋅,⋅⟩

⋅⋅\langle\cdot,\cdot\rangle denotes the Frobenius inner product, and for each l=1,…,L𝑙

1…𝐿l=1,...,L the two matrices Δ​𝐖lΔsubscript𝐖𝑙\Delta{\bm{W}}\_{l} and 𝐆lsubscript𝐆𝑙{\bm{G}}\_{l} are of the same shape. If we define the global step size η=1λ​∑k=1L1sk​‖𝐆k‖k†𝜂1𝜆superscriptsubscript𝑘1𝐿1subscript𝑠𝑘superscriptsubscriptnormsubscript𝐆𝑘𝑘†\eta=\frac{1}{\lambda}\sum\_{k=1}^{L}\frac{1}{s\_{k}}\|{{\bm{G}}\_{k}}\|\_{k}^{\dagger}, then the solution to [Equation 32](#Sx2.E32 "In Proposition 7 (Steepest descent under the modular norm) ‣ Epilogue ‣ Old Optimizer, New Norm: An Anthology") is given by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​𝑾l=−ηsl⋅arg​max‖𝑻l‖l=1⁡⟨𝑮l,𝑻l⟩ for each layer ​l=1,…,L.formulae-sequenceΔsubscript𝑾𝑙⋅𝜂subscript𝑠𝑙subscriptargmaxsubscriptnormsubscript𝑻𝑙𝑙1subscript𝑮𝑙subscript𝑻𝑙 for each layer 𝑙  1…𝐿\displaystyle\Delta{\bm{W}}\_{l}=-\frac{\eta}{s\_{l}}\cdot\operatorname\*{arg\,max}\_{\|{{\bm{T}}\_{l}}\|\_{l}=1}\,\langle{\bm{G}}\_{l},{\bm{T}}\_{l}\rangle\quad\text{ for each layer }l=1,...,L. |  | (33) |

In words, steepest descent under the modular norm updates each layer in a direction informed by that layer’s norm and with a global step size computed as a weighted sum of the dual norms of the gradients over layers. The proof of this proposition is given in [Appendix B](#A2.SSx7 "7: \nameref*prop:steepest-modular ‣ Appendix B Proofs ‣ Old Optimizer, New Norm: An Anthology").

When confronted with the modular norm, it’s natural to ask how one should assign norms to layers. And there are so many norms to choose from! Beyond the familiar ℓ2→ℓ2→subscriptℓ2subscriptℓ2\ell\_{2}\to\ell\_{2} spectral norm, many other induced operator norms are computationally tractable:

###### Proposition 8 (ℓ1→ℓp→subscriptℓ1subscriptℓ𝑝\ell\_{1}\to\ell\_{p} and ℓp→ℓ∞→subscriptℓ𝑝subscriptℓ\ell\_{p}\to\ell\_{\infty} induced operator norms are tractable)

For a matrix 𝐌∈ℝm×n𝐌superscriptℝ𝑚𝑛{\bm{M}}\in\mathbb{R}^{m\times n} with m𝑚m rows {rowi​(𝐌)}i=1msuperscriptsubscriptsubscriptrow𝑖𝐌𝑖1𝑚\{\mathrm{row}\_{i}({\bm{M}})\}\_{i=1}^{m} and n𝑛n columns {colj​(𝐌)}j=1nsuperscriptsubscriptsubscriptcol𝑗𝐌𝑗1𝑛\{\mathrm{col}\_{j}({\bm{M}})\}\_{j=1}^{n}, and 1≤p≤∞1𝑝1\leq p\leq\infty:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖𝑴‖ℓ1→ℓp=maxj⁡‖colj​(𝑴)‖p;‖𝑴‖ℓp→ℓ∞=maxi⁡‖rowi​(𝑴)‖pp−1.formulae-sequencesubscriptnorm𝑴→subscriptℓ1subscriptℓ𝑝subscript𝑗subscriptnormsubscriptcol𝑗𝑴𝑝subscriptnorm𝑴→subscriptℓ𝑝subscriptℓsubscript𝑖subscriptnormsubscriptrow𝑖𝑴𝑝𝑝1\displaystyle\|{{\bm{M}}}\|\_{\ell\_{1}\to\ell\_{p}}=\max\_{j}\|{\mathrm{col}\_{j}({\bm{M}})}\|\_{p};\qquad\|{{\bm{M}}}\|\_{\ell\_{p}\to\ell\_{\infty}}=\max\_{i}\|{\mathrm{row}\_{i}({\bm{M}})}\|\_{\frac{p}{p-1}}. |  | (34) |

In words, the ℓ1→ℓp→subscriptℓ1subscriptℓ𝑝\ell\_{1}\to\ell\_{p} operator norm is the largest ℓpsubscriptℓ𝑝\ell\_{p} norm of the columns; the ℓp→ℓ∞→subscriptℓ𝑝subscriptℓ\ell\_{p}\to\ell\_{\infty} operator norm is the largest dual ℓpsubscriptℓ𝑝\ell\_{p} norm over the rows. The proof is given in [Appendix B](#A2.SSx8 "8: \nameref*prop:tractable-norms ‣ Appendix B Proofs ‣ Old Optimizer, New Norm: An Anthology").

To assign a norm to a layer, we believe that one should consider the role that layer plays in the neural network. For instance, since linear layers are typically used to map to and from vectors with roughly unit RMS norm, it is appropriate to equip linear layers with the induced RMS to RMS operator norm (Yang et al., [2023](#bib.bib33)), which resolves to a rescaled spectral norm. And since embedding layers map from one-hot vectors to vectors with roughly unit RMS norm, it is appropriate to equip embedding layers with the ℓ1subscriptℓ1\ell\_{1} to RMS operator norm, which resolves to a rescaled ℓ1subscriptℓ1\ell\_{1} to ℓ2subscriptℓ2\ell\_{2} operator norm. So embedding layers and linear layers should be equipped with different norms despite the weight space being a matrix space in both cases. In short, the algorithm designer has freedom to choose input and output norms for layers that capture differences in how the layers are used; inducing the corresponding operator norm on the layer’s weights provides control over how the optimizer learns representations.

We believe that picking the right norms could improve the speed and scalability of neural network training. We are seeing evidence that equipping neural network layers with better norms can lead to learning rate transfer across scale (Yang et al., [2023](#bib.bib33); Large et al., [2024](#bib.bib24)). And since Shampoo won the external tuning track of the 2024 AlgoPerf competition (Dahl et al., [2023](#bib.bib10)), it is garnering interest as a fast training method. The second story in our anthology shows that Shampoo is closely connected to the spectral norm.

In conclusion, this work highlights a perspective on optimizer design as choosing two things: a norm and a step size. We have shown that three popular methods—Adam, Shampoo and Prodigy—fit within this perspective. We hope that researchers can design improved training algorithms by choosing norms and step sizes more intentionally.

“Though this be madness, yet there is method in’t.”
  
Hamlet

## Acknowledgements

We are grateful to Tim Large and Phillip Isola for invaluable discussions on the stories in this anthology. We also thank Jack Gallagher, Keller Jordan and Victor Butoi for very helpful conversations.

## References

* Anil et al. (2020)

  Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, and Yoram Singer.
  Scalable second order optimization for deep learning.
  *arXiv:2002.09018*, 2020.
* Armijo (1966)

  Larry Armijo.
  Minimization of functions having Lipschitz continuous first partial derivatives.
  *Pacific Journal of Mathematics*, 1966.
* Balles and Hennig (2018)

  Lukas Balles and Philipp Hennig.
  Dissecting Adam: The sign, magnitude and variance of stochastic gradients.
  In *International Conference on Machine Learning*, 2018.
* Bernstein et al. (2018)

  Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, and Animashree Anandkumar.
  signSGD: Compressed optimisation for non-convex problems.
  In *International Conference on Machine Learning*, 2018.
* Bernstein et al. (2023)

  Jeremy Bernstein, Chris Mingard, Kevin Huang, Navid Azizan, and Yisong Yue.
  Automatic Gradient Descent: Deep Learning without Hyperparameters.
  *arXiv:2304.05187*, 2023.
* Carlson et al. (2015)

  David Carlson, Volkan Cevher, and Lawrence Carin.
  Stochastic spectral descent for Restricted Boltzmann Machines.
  In *International Conference on Artificial Intelligence and Statistics*, 2015.
* Carlson et al. (2016)

  David Carlson, Ya-Ping Hsieh, Edo Collins, Lawrence Carin, and Volkan Cevher.
  Stochastic spectral descent for discrete graphical models.
  *Selected Topics in Signal Processing*, 2016.
* Cauchy (1847)

  Augustin-Louis Cauchy.
  Méthode générale pour la résolution des systèmes d’équations simultanées.
  *Comptes Rendus Hebdomadaires des Séances de l’Académie des Sciences*, 1847.
* Chen et al. (2023)

  Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, and Quoc V Le.
  Symbolic discovery of optimization algorithms.
  In *Neural Information Processing Systems*, 2023.
* Dahl et al. (2023)

  George E. Dahl, Frank Schneider, Zachary Nado, Naman Agarwal, Chandramouli Shama Sastry, Philipp Hennig, Sourabh Medapati, Runa Eschenhagen, Priya Kasimbeg, Daniel Suo, Juhan Bae, Justin Gilmer, Abel L. Peirson, Bilal Khan, Rohan Anil, Mike Rabbat, Shankar Krishnan, Daniel Snider, Ehsan Amid, Kongtao Chen, Chris J. Maddison, Rakshith Vasudev, Michal Badura, Ankush Garg, and Peter Mattson.
  Benchmarking neural network training algorithms.
  *arXiv:2306.07179*, 2023.
* Defazio and Mishchenko (2023)

  Aaron Defazio and Konstantin Mishchenko.
  Learning-rate-free learning by D-adaptation.
  In *International Conference on Machine Learning*, 2023.
* Duchi et al. (2011)

  John C. Duchi, Elad Hazan, and Yoram Singer.
  Adaptive subgradient methods for online learning and stochastic optimization.
  *Journal Machine Learning Research*, 2011.
* Fan (2017)

  Kai Fan.
  Unifying the stochastic spectral descent for Restricted Boltzmann Machines with Bernoulli or Gaussian inputs.
  *arXiv:1703.09766*, 2017.
* Feinberg et al. (2023)

  Vladimir Feinberg, Xinyi Chen, Y. Jennifer Sun, Rohan Anil, and Elad Hazan.
  Sketchy: Memory-efficient adaptive regularization with frequent directions.
  In *Neural Information Processing Systems*, 2023.
* Gupta et al. (2017)

  Vineet Gupta, Tomer Koren, and Yoram Singer.
  A unified approach to adaptive regularization in online and stochastic optimization.
  Technical report, Google Brain, 2017.
* Gupta et al. (2018)

  Vineet Gupta, Tomer Koren, and Yoram Singer.
  Shampoo: Preconditioned stochastic tensor optimization.
  In *International Conference on Machine Learning*, 2018.
* Higham (2008)

  Nicholas J. Higham.
  *Functions of Matrices*.
  Society for Industrial and Applied Mathematics, 2008.
* Ivgi et al. (2023)

  Maor Ivgi, Oliver Hinder, and Yair Carmon.
  DoG is SGD’s best friend: A parameter-free dynamic step size schedule.
  In *International Conference on Machine Learning*, 2023.
* Kenneweg et al. (2024)

  Philip Kenneweg, Tristan Kenneweg, and Barbara Hammer.
  Improving line search methods for large scale neural network training.
  In *International Conference on Artificial Intelligence, Computer, Data Sciences and Applications*, 2024.
* Khaled et al. (2023)

  Ahmed Khaled, Konstantin Mishchenko, and Chi Jin.
  DoWG unleashed: An efficient universal parameter-free gradient descent method.
  In *Neural Information Processing Systems*, 2023.
* Kingma and Ba (2015)

  Diederik P. Kingma and Jimmy Ba.
  Adam: A method for stochastic optimization.
  In *International Conference on Learning Representations*, 2015.
* Lakić (1998)

  Slobodan Lakić.
  On the computation of the matrix k-th root.
  *Journal of Applied Mathematics and Mechanics*, 1998.
* Lange (2016)

  Kenneth Lange.
  *MM Optimization Algorithms*.
  Society for Industrial and Applied Mathematics, 2016.
* Large et al. (2024)

  Tim Large, Yang Liu, Minyoung Huh, Hyojin Bahng, Phillip Isola, and Jeremy Bernstein.
  Scalable optimization in the modular norm.
  *arXiv:2405.14813*, 2024.
* Martinsson and Tropp (2020)

  Per-Gunnar Martinsson and Joel A. Tropp.
  Randomized numerical linear algebra: Foundations and algorithms.
  *Acta Numerica*, 2020.
* Mishchenko and Defazio (2023)

  Konstantin Mishchenko and Aaron Defazio.
  Prodigy: An expeditiously adaptive parameter-free learner.
  *arXiv:2306.06101*, 2023.
* Morwani et al. (2024)

  Depen Morwani, Itai Shapira, Nikhil Vyas, Eran Malach, Sham Kakade, and Lucas Janson.
  A new perspective on Shampoo’s preconditioner.
  *arXiv:2406.17748*, 2024.
* Riedmiller and Braun (1993)

  Martin Riedmiller and Heinrich Braun.
  A direct adaptive method for faster backpropagation learning: The RPROP algorithm.
  In *International Conference on Neural Networks*, 1993.
* Shi et al. (2023)

  Hao-Jun Michael Shi, Tsung-Hsien Lee, Shintaro Iwasaki, Jose Gallego-Posada, Zhijing Li, Kaushik Rangadurai, Dheevatsa Mudigere, and Michael Rabbat.
  A distributed data-parallel PyTorch implementation of the distributed Shampoo optimizer for training neural networks at-scale.
  *arXiv:2309.06497*, 2023.
* Streeter (2023)

  Matthew Streeter.
  Universal majorization-minimization algorithms.
  *arXiv:2308.00190*, 2023.
* Sun and Spall (2021)

  Shiqing Sun and James C. Spall.
  Connection of diagonal Hessian estimates to natural gradients in stochastic optimization.
  In *Information Sciences and Systems*, 2021.
* Tieleman and Hinton (2012)

  Tijmen Tieleman and Geoffrey Hinton.
  RMSprop.
  *Coursera: Neural Networks for Machine Learning*, Lecture 6.5, 2012.
* Yang et al. (2023)

  Greg Yang, James B. Simon, and Jeremy Bernstein.
  A spectral condition for feature learning.
  *arXiv:2310.17813*, 2023.
* Zhao et al. (2024)

  Rosie Zhao, Depen Morwani, David Brandfonbrener, Nikhil Vyas, and Sham Kakade.
  Deconstructing what makes a good optimizer for language models.
  *arXiv:2407.07972*, 2024.

## Appendix A Computational Strategies for Shampoo

Let 𝑮∈ℝm×n𝑮superscriptℝ𝑚𝑛{\bm{G}}\in\mathbb{R}^{m\times n} be a gradient matrix with reduced SVD 𝑮=𝑼​𝚺​𝑽⊤𝑮𝑼𝚺superscript𝑽top{\bm{G}}={\bm{U}}{\bm{\Sigma}}{\bm{V}}^{\top}. By [Equations 15](#S2.E15 "In Story II Shampoo as Steepest Descent under the Spectral Norm ‣ Old Optimizer, New Norm: An Anthology") and [16](#S2.E16 "Equation 16 ‣ Story II Shampoo as Steepest Descent under the Spectral Norm ‣ Old Optimizer, New Norm: An Anthology"), the corresponding Shampoo update (with EMA disabled) is given by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ​𝑾=−η⋅(𝑮​𝑮⊤)−1/4​𝑮​(𝑮⊤​𝑮)−1/4=−η⋅𝑼​𝑽⊤.Δ𝑾⋅𝜂superscript𝑮superscript𝑮top14𝑮superscriptsuperscript𝑮top𝑮14⋅𝜂𝑼superscript𝑽top\Delta{\bm{W}}=-\eta\cdot({\bm{G}}{\bm{G}}^{\top})^{-\nicefrac{{1}}{{4}}}\,{\bm{G}}\,({\bm{G}}^{\top}{\bm{G}})^{-\nicefrac{{1}}{{4}}}=-\eta\cdot{\bm{U}}{\bm{V}}^{\top}. |  | (35) |

Here we list every means we know of computing or approximating this equation. First, we mention that (𝑮​𝑮⊤)−1/4​𝑮​(𝑮⊤​𝑮)−1/4=(𝑮​𝑮⊤)−1/2​𝑮=𝑮​(𝑮⊤​𝑮)−1/2superscript𝑮superscript𝑮top14𝑮superscriptsuperscript𝑮top𝑮14superscript𝑮superscript𝑮top12𝑮𝑮superscriptsuperscript𝑮top𝑮12({\bm{G}}{\bm{G}}^{\top})^{-\nicefrac{{1}}{{4}}}\,{\bm{G}}\,({\bm{G}}^{\top}{\bm{G}})^{-\nicefrac{{1}}{{4}}}=({\bm{G}}{\bm{G}}^{\top})^{-\nicefrac{{1}}{{2}}}\,{\bm{G}}={\bm{G}}\,({\bm{G}}^{\top}{\bm{G}})^{-\nicefrac{{1}}{{2}}}, so if one is willing to compute inverse matrix roots, one need only compute either (𝑮​𝑮⊤)−1/2superscript𝑮superscript𝑮top12({\bm{G}}{\bm{G}}^{\top})^{-\nicefrac{{1}}{{2}}} or (𝑮⊤​𝑮)−1/2superscriptsuperscript𝑮top𝑮12({\bm{G}}^{\top}{\bm{G}})^{-\nicefrac{{1}}{{2}}}, whichever has smaller dimension. With that said, to compute [Equation 35](#A1.E35 "In Appendix A Computational Strategies for Shampoo ‣ Old Optimizer, New Norm: An Anthology"), one may:

1. 1.

   Do the SVD. Apply an SVD routine to compute 𝑼𝑼{\bm{U}}, 𝚺𝚺{\bm{\Sigma}} and 𝑽⊤superscript𝑽top{\bm{V}}^{\top} and just discard 𝚺𝚺{\bm{\Sigma}}.
2. 2.

   Do sketching. Sketching is a randomized method (Martinsson and Tropp, [2020](#bib.bib25)) that can be used to approximate the SVD. See, for instance, Sketchy (Feinberg et al., [2023](#bib.bib14)).
3. 3.

   Do Newton iteration for inverse p𝑝pth roots. Inverse matrix roots such as (𝑮​𝑮⊤)−1/2superscript𝑮superscript𝑮top12({\bm{G}}{\bm{G}}^{\top})^{-\nicefrac{{1}}{{2}}} can be computed via Newton iteration (Lakić, [1998](#bib.bib22)). This is discussed in Chapter 7 of Higham ([2008](#bib.bib17))’s book. And see Anil et al. ([2020](#bib.bib1))’s paper.
4. 4.

   Do Newton-Schulz iteration. We developed a “Newton-Schulz iteration” for computing 𝑼​𝑽⊤𝑼superscript𝑽top{\bm{U}}{\bm{V}}^{\top}, adapted from Equation 5.22 in Higham ([2008](#bib.bib17))’s book. In short, if we set 𝑿0=𝑮/‖𝑮‖ℓ2→ℓ2subscript𝑿0𝑮subscriptnorm𝑮→subscriptℓ2subscriptℓ2{\bm{X}}\_{0}={\bm{G}}/\|{{\bm{G}}}\|\_{\ell\_{2}\to\ell\_{2}} (or alternatively 𝑿0=𝑮/‖𝑮‖Fsubscript𝑿0𝑮subscriptnorm𝑮𝐹{\bm{X}}\_{0}={\bm{G}}/\|{{\bm{G}}}\|\_{F}) and iterate:

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | 𝑿t+1=32⋅𝑿t−12⋅𝑿t​𝑿t⊤​𝑿t,subscript𝑿𝑡1⋅32subscript𝑿𝑡⋅12subscript𝑿𝑡superscriptsubscript𝑿𝑡topsubscript𝑿𝑡{\bm{X}}\_{t+1}=\frac{3}{2}\cdot{\bm{X}}\_{t}-\frac{1}{2}\cdot{\bm{X}}\_{t}{\bm{X}}\_{t}^{\top}{\bm{X}}\_{t}, |  | (36) |

   then as t→∞→𝑡t\to\infty, the sequence 𝑿t→𝑼​𝑽⊤→subscript𝑿𝑡𝑼superscript𝑽top{\bm{X}}\_{t}\to{\bm{U}}{\bm{V}}^{\top}. To see this, one should plot the univariate cubic function f​(x):=32⋅x−12⋅x3assign𝑓𝑥⋅32𝑥⋅12superscript𝑥3f(x)\vcentcolon=\tfrac{3}{2}\cdot x-\tfrac{1}{2}\cdot x^{3} and see that, for 0<x<30𝑥30<x<\sqrt{3}, iterating this cubic will push x𝑥x closer and closer to +11+1. The final step is to realize that the effect of the iteration in [Equation 36](#A1.E36 "In Item 4 ‣ Appendix A Computational Strategies for Shampoo ‣ Old Optimizer, New Norm: An Anthology") is to apply this cubic f​(x)𝑓𝑥f(x) to each singular value of 𝑿tsubscript𝑿𝑡{\bm{X}}\_{t}. This also shows that the spectral normalization 𝑿0=𝑮/‖𝑮‖ℓ2→ℓ2subscript𝑿0𝑮subscriptnorm𝑮→subscriptℓ2subscriptℓ2{\bm{X}}\_{0}={\bm{G}}/\|{{\bm{G}}}\|\_{\ell\_{2}\to\ell\_{2}} is stronger than what is required: we need only ensure that 𝑿0subscript𝑿0{\bm{X}}\_{0} has all singular values greater than zero and less than 33\sqrt{3} in order for the iteration to converge.

   Finally, there are in fact a family of degree 2​n+12𝑛12n+1 polynomial iterations of the form

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | 𝑿t+1=a⋅𝑿t+b⋅𝑿t​𝑿t⊤​𝑿t+c⋅(𝑿t​𝑿t⊤)2​𝑿t+…+z⋅(𝑿t​𝑿t⊤)n​𝑿tsubscript𝑿𝑡1⋅𝑎subscript𝑿𝑡⋅𝑏subscript𝑿𝑡superscriptsubscript𝑿𝑡topsubscript𝑿𝑡⋅𝑐superscriptsubscript𝑿𝑡superscriptsubscript𝑿𝑡top2subscript𝑿𝑡…⋅𝑧superscriptsubscript𝑿𝑡superscriptsubscript𝑿𝑡top𝑛subscript𝑿𝑡{\bm{X}}\_{t+1}=a\cdot{\bm{X}}\_{t}+b\cdot{\bm{X}}\_{t}{\bm{X}}\_{t}^{\top}{\bm{X}}\_{t}+c\cdot({\bm{X}}\_{t}{\bm{X}}\_{t}^{\top})^{2}{\bm{X}}\_{t}+...+z\cdot({\bm{X}}\_{t}{\bm{X}}\_{t}^{\top})^{n}{\bm{X}}\_{t} |  | (37) |

   for suitable a,b,c,…,z
   𝑎𝑏𝑐…𝑧a,b,c,...,z that could be used instead of [Equation 36](#A1.E36 "In Item 4 ‣ Appendix A Computational Strategies for Shampoo ‣ Old Optimizer, New Norm: An Anthology"). One should choose coefficients a,b,c,…,z
   𝑎𝑏𝑐…𝑧a,b,c,...,z so that the univariate polynomial g​(x)=a⋅x+b⋅x3+c⋅x5+…+z⋅x2​n+1𝑔𝑥⋅𝑎𝑥⋅𝑏superscript𝑥3⋅𝑐superscript𝑥5…⋅𝑧superscript𝑥2𝑛1g(x)=a\cdot x+b\cdot x^{3}+c\cdot x^{5}+...+z\cdot x^{2n+1} is a suitable approximation to sign⁡(x)sign𝑥\operatorname{sign}(x).

Which of these methods is most useful in practice may depend on factors such as the condition number of the matrix 𝑮𝑮{\bm{G}} or the nature of the available computational resources.

## Appendix B Proofs

### [1](#Thmmyproposition1 "Proposition 1 (Steepest descent) ‣ Prologue ‣ Old Optimizer, New Norm: An Anthology"): \nameref\*prop:steepest

###### Proof B.1.

First, let’s study the minimization under the change of variables Δ​𝐰=c⋅𝐭Δ𝐰⋅𝑐𝐭\Delta{\bm{w}}=c\cdot{\bm{t}}, where c≥0𝑐0c\geq 0 encodes the “magnitude” and 𝐭𝐭{\bm{t}} is a unit vector (‖𝐭‖=1norm𝐭1\|{{\bm{t}}}\|=1) encoding the “direction”:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | minΔ​𝒘∈ℝn⁡[𝒈⊤​Δ​𝒘+λ2​‖Δ​𝒘‖2]subscriptΔ𝒘superscriptℝ𝑛superscript𝒈topΔ𝒘𝜆2superscriptnormΔ𝒘2\displaystyle\min\_{\Delta{\bm{w}}\in\mathbb{R}^{n}}\left[{\bm{g}}^{\top}\Delta{\bm{w}}+\frac{\lambda}{2}\,\|{\Delta{\bm{w}}}\|^{2}\right] | =minc≥0⁡min𝒕∈ℝn:‖𝒕‖=1⁡[c⋅𝒈⊤​𝒕+λ2​c2​‖𝒕‖2]absentsubscript𝑐0subscript:𝒕superscriptℝ𝑛norm𝒕1⋅𝑐superscript𝒈top𝒕𝜆2superscript𝑐2superscriptnorm𝒕2\displaystyle=\min\_{c\geq 0}\min\_{{\bm{t}}\in\mathbb{R}^{n}:\|{{\bm{t}}}\|=1}\left[c\cdot{\bm{g}}^{\top}{\bm{t}}+\frac{\lambda}{2}c^{2}\|{{\bm{t}}}\|^{2}\right] |  | (38) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =minc≥0⁡[c⋅min𝒕∈ℝn:‖𝒕‖=1⁡[𝒈⊤​𝒕]+λ2​c2]absentsubscript𝑐0⋅𝑐subscript:𝒕superscriptℝ𝑛norm𝒕1superscript𝒈top𝒕𝜆2superscript𝑐2\displaystyle=\min\_{c\geq 0}\left[c\cdot\min\_{{\bm{t}}\in\mathbb{R}^{n}:\|{{\bm{t}}}\|=1}\left[{\bm{g}}^{\top}{\bm{t}}\right]+\frac{\lambda}{2}c^{2}\right] |  | (39) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =minc≥0⁡[−c⋅‖𝒈‖†+λ2​c2],absentsubscript𝑐0⋅𝑐superscriptnorm𝒈†𝜆2superscript𝑐2\displaystyle=\min\_{c\geq 0}\left[-c\cdot\|{{\bm{g}}}\|^{\dagger}+\frac{\lambda}{2}c^{2}\right], |  | (40) |

Inspecting [Equation 39](#A2.E39 "In Proof B.1. ‣ 1: \nameref*prop:steepest ‣ Appendix B Proofs ‣ Old Optimizer, New Norm: An Anthology"), we see that the minimizer for the direction 𝐭𝐭{\bm{t}} is given by:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒕𝒕\displaystyle{\bm{t}} | =arg​min𝒕∈ℝn:‖𝒕‖=1⁡[𝒈⊤​𝒕]=−arg​max𝒕∈ℝn:‖𝒕‖=1⁡[𝒈⊤​𝒕]absentsubscriptargmin:𝒕superscriptℝ𝑛norm𝒕1superscript𝒈top𝒕subscriptargmax:𝒕superscriptℝ𝑛norm𝒕1superscript𝒈top𝒕\displaystyle=\operatorname\*{arg\,min}\_{{\bm{t}}\in\mathbb{R}^{n}:\|{{\bm{t}}}\|=1}\left[{\bm{g}}^{\top}{\bm{t}}\right]=-\operatorname\*{arg\,max}\_{{\bm{t}}\in\mathbb{R}^{n}:\|{{\bm{t}}}\|=1}\left[{\bm{g}}^{\top}{\bm{t}}\right] |  | (41) |

And similarly, by inspecting [Equation 40](#A2.E40 "In Proof B.1. ‣ 1: \nameref*prop:steepest ‣ Appendix B Proofs ‣ Old Optimizer, New Norm: An Anthology"), the minimizer for the magnitude c𝑐c is given by:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | c𝑐\displaystyle c | =arg​minc≥0⁡[−c⋅‖𝒈‖†+λ2​c2]=‖𝒈‖†λ.absentsubscriptargmin𝑐0⋅𝑐superscriptnorm𝒈†𝜆2superscript𝑐2superscriptnorm𝒈†𝜆\displaystyle=\operatorname\*{arg\,min}\_{c\geq 0}\left[-c\cdot\|{{\bm{g}}}\|^{\dagger}+\frac{\lambda}{2}c^{2}\right]=\frac{\|{{\bm{g}}}\|^{\dagger}}{\lambda}. |  | (42) |

Multiplying these expressions, we obtain the minimizer for Δ​𝐰Δ𝐰\Delta{\bm{w}}, yielding the result.

### [2](#Thmmyproposition2 "Proposition 2 (Sign descent as steepest descent under the infinity norm) ‣ Story I Adam as Steepest Descent under the Max-of-Max Norm ‣ Old Optimizer, New Norm: An Anthology"): \nameref\*prop:sign-descent

###### Proof B.2.

The result follows by applying [1](#Thmmyproposition1 "Proposition 1 (Steepest descent) ‣ Prologue ‣ Old Optimizer, New Norm: An Anthology"). We just need that arg​max‖𝐭‖∞=1⁡𝐠⊤​𝐭=sign⁡(𝐠)subscriptargmaxsubscriptnorm𝐭1superscript𝐠top𝐭sign𝐠\operatorname\*{arg\,max}\_{\|{{\bm{t}}}\|\_{\infty}=1}{\bm{g}}^{\top}{\bm{t}}=\operatorname{sign}({\bm{g}}), and also that the dual norm ‖𝐠‖∞†:=max‖𝐭‖∞=1⁡𝐠⊤​𝐭=𝐠⊤​sign⁡(𝐠)=‖𝐠‖1assignsuperscriptsubscriptnorm𝐠†subscriptsubscriptnorm𝐭1superscript𝐠top𝐭superscript𝐠topsign𝐠subscriptnorm𝐠1\|{{\bm{g}}}\|\_{\infty}^{\dagger}\vcentcolon=\max\_{\|{{\bm{t}}}\|\_{\infty}=1}{\bm{g}}^{\top}{\bm{t}}={\bm{g}}^{\top}\operatorname{sign}({\bm{g}})=\|{{\bm{g}}}\|\_{1}.

### [3](#Thmmyproposition3 "Proposition 3 (Sign descent as steepest descent under the max-of-max norm) ‣ Story I Adam as Steepest Descent under the Max-of-Max Norm ‣ Old Optimizer, New Norm: An Anthology"): \nameref\*prop:structural-sign-descent

###### Proof B.3.

The result follows from [7](#Thmmyproposition7 "Proposition 7 (Steepest descent under the modular norm) ‣ Epilogue ‣ Old Optimizer, New Norm: An Anthology") by setting all the scalars s1,…,sL

subscript𝑠1…subscript𝑠𝐿s\_{1},...,s\_{L} to one and all the norms ∥⋅∥1,…,∥⋅∥L\|{\cdot}\|\_{1},...,\|{\cdot}\|\_{L} to the ℓ1subscriptℓ1\ell\_{1} to ℓ∞subscriptℓ\ell\_{\infty} operator norm. All we need is to show that the argmax at each matrix space l=1,…,L𝑙

1…𝐿l=1,...,L satisfies:

|  |  |  |  |
| --- | --- | --- | --- |
|  | arg​max‖𝑻l‖ℓ1→ℓ∞=1⁡tr⁡(𝑮l⊤​𝑻l)=sign⁡(𝑮l).subscriptargmaxsubscriptnormsubscript𝑻𝑙→subscriptℓ1subscriptℓ1trsuperscriptsubscript𝑮𝑙topsubscript𝑻𝑙signsubscript𝑮𝑙\operatorname\*{arg\,max}\_{\|{{\bm{T}}\_{l}}\|\_{\ell\_{1}\to\ell\_{\infty}}=1}\operatorname{tr}({\bm{G}}\_{l}^{\top}{\bm{T}}\_{l})=\operatorname{sign}({\bm{G}}\_{l}). |  | (43) |

But this holds because, by [8](#Thmmyproposition8 "Proposition 8 (ℓ₁→ℓ_𝑝 and ℓ_𝑝→ℓ_∞ induced operator norms are tractable) ‣ Epilogue ‣ Old Optimizer, New Norm: An Anthology"), ‖𝐓‖ℓ1→ℓ∞=maxi⁡‖coli​(𝐓)‖∞=maxi​j⁡|𝐓i​j|subscriptnorm𝐓→subscriptℓ1subscriptℓsubscript𝑖subscriptnormsubscriptcol𝑖𝐓subscript𝑖𝑗subscript𝐓𝑖𝑗\|{{\bm{T}}}\|\_{\ell\_{1}\to\ell\_{\infty}}=\max\_{i}\|{\mathrm{col}\_{i}({\bm{T}})}\|\_{\infty}=\max\_{ij}|{{\bm{T}}\_{ij}}|, and therefore all components in the argmax must be of unit size and gradient aligned.

### [4](#Thmmyproposition4 "Proposition 4 (Projection to the closest semi-orthogonal matrix) ‣ Story II Shampoo as Steepest Descent under the Spectral Norm ‣ Old Optimizer, New Norm: An Anthology"): \nameref\*prop:projection

###### Proof B.4.

To begin, we observe that the minimizer over semi-orthogonal matrices of the “distance” ‖𝐀−𝐆‖Fsubscriptnorm𝐀𝐆𝐹\|{{\bm{A}}-{\bm{G}}}\|\_{F} is the same as the maximizer over semi-orthogonal matrices of the “alignment” ⟨𝐀,𝐆⟩

𝐀𝐆\langle{\bm{A}},{\bm{G}}\rangle, where ⟨⋅,⋅⟩

⋅⋅\langle\cdot,\cdot\rangle denotes the Frobenius inner product. This is because:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝑨−𝑮‖F2superscriptsubscriptnorm𝑨𝑮𝐹2\displaystyle\|{{\bm{A}}-{\bm{G}}}\|\_{F}^{2} | =‖𝑨‖F2−2⋅⟨𝑨,𝑮⟩+‖𝑮‖F2,absentsuperscriptsubscriptnorm𝑨𝐹2⋅2  𝑨𝑮superscriptsubscriptnorm𝑮𝐹2\displaystyle=\|{{\bm{A}}}\|\_{F}^{2}-2\cdot\langle{\bm{A}},{\bm{G}}\rangle+\|{{\bm{G}}}\|\_{F}^{2}, |  | (44) |

and the term ‖𝐀‖F2superscriptsubscriptnorm𝐀𝐹2\|{{\bm{A}}}\|\_{F}^{2} is fixed at ‖𝐀‖F2=min⁡(m,n)superscriptsubscriptnorm𝐀𝐹2𝑚𝑛\|{{\bm{A}}}\|\_{F}^{2}=\min(m,n) for a semi-orthogonal matrix 𝐀∈𝒪m×n𝐀subscript𝒪𝑚𝑛{\bm{A}}\in\mathcal{O}\_{m\times n}.

Now, let 𝐆=∑iσi​𝐮i​𝐯i⊤𝐆subscript𝑖subscript𝜎𝑖subscript𝐮𝑖superscriptsubscript𝐯𝑖top{\bm{G}}=\sum\_{i}\sigma\_{i}\,{\bm{u}}\_{i}{\bm{v}}\_{i}^{\top} denote the SVD of 𝐆𝐆{\bm{G}}. Then the alignment satisfies:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ⟨𝑨,𝑮⟩=tr​∑iσi​𝒗i​𝒖i⊤​𝑨=∑iσi​𝒖i⊤​𝑨​𝒗i≤∑iσi,  𝑨𝑮 trsubscript𝑖subscript𝜎𝑖subscript𝒗𝑖superscriptsubscript𝒖𝑖top𝑨subscript𝑖subscript𝜎𝑖superscriptsubscript𝒖𝑖top𝑨subscript𝒗𝑖subscript𝑖subscript𝜎𝑖\displaystyle\langle{\bm{A}},{\bm{G}}\rangle=\operatorname{tr}\sum\_{i}\sigma\_{i}\,{\bm{v}}\_{i}{\bm{u}}\_{i}^{\top}{\bm{A}}=\sum\_{i}\sigma\_{i}\,{\bm{u}}\_{i}^{\top}{\bm{A}}{\bm{v}}\_{i}\leq\sum\_{i}\sigma\_{i}, |  | (45) |

where the second equality follows by the cyclic property of the trace, and the inequality is since 𝐀𝐀{\bm{A}} being semi-orthogonal means that 𝐮⊤​𝐀​𝐯≤1superscript𝐮top𝐀𝐯1{\bm{u}}^{\top}{\bm{A}}{\bm{v}}\leq 1 for any two unit vectors 𝐮𝐮{\bm{u}} and 𝐯𝐯{\bm{v}}.

Next, observe that for the semi-orthogonal matrix 𝐀⋆=∑i𝐮i​𝐯i⊤subscript𝐀⋆subscript𝑖subscript𝐮𝑖superscriptsubscript𝐯𝑖top{\bm{A}}\_{\star}=\sum\_{i}{\bm{u}}\_{i}{\bm{v}}\_{i}^{\top}, we have that:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ⟨𝑨⋆,𝑮⟩=∑iσi​∑j𝒖i⊤​𝒖j​𝒗j⊤​𝒗i=∑iσi,  subscript𝑨⋆𝑮 subscript𝑖subscript𝜎𝑖subscript𝑗superscriptsubscript𝒖𝑖topsubscript𝒖𝑗superscriptsubscript𝒗𝑗topsubscript𝒗𝑖subscript𝑖subscript𝜎𝑖\langle{\bm{A}}\_{\star},{\bm{G}}\rangle=\sum\_{i}\sigma\_{i}\sum\_{j}{\bm{u}}\_{i}^{\top}{\bm{u}}\_{j}{\bm{v}}\_{j}^{\top}{\bm{v}}\_{i}=\sum\_{i}\sigma\_{i}, |  | (46) |

since the {𝐮i}subscript𝐮𝑖\{{\bm{u}}\_{i}\} and {𝐯i}subscript𝐯𝑖\{{\bm{v}}\_{i}\} are orthonormal. Comparing against [Equation 45](#A2.E45 "In Proof B.4. ‣ 4: \nameref*prop:projection ‣ Appendix B Proofs ‣ Old Optimizer, New Norm: An Anthology"), we see that 𝐀⋆subscript𝐀⋆{\bm{A}}\_{\star} indeed maximizes the alignment, since it achieves the upper bound of ∑iσisubscript𝑖subscript𝜎𝑖\sum\_{i}\sigma\_{i}. And 𝐀⋆subscript𝐀⋆{\bm{A}}\_{\star} therefore also minimizes the distance ‖𝐀−𝐆‖Fsubscriptnorm𝐀𝐆𝐹\|{{\bm{A}}-{\bm{G}}}\|\_{F} amongst semi-orthogonal matrices 𝐀𝐀{\bm{A}}. Note that if 𝐔𝐔{\bm{U}} is the matrix that has the {𝐮i}subscript𝐮𝑖\{{\bm{u}}\_{i}\} as columns, and likewise for 𝐕𝐕{\bm{V}} and the {𝐯i}subscript𝐯𝑖\{{\bm{v}}\_{i}\}, then this solution may equivalently be expressed as 𝐀⋆=𝐔​𝐕⊤subscript𝐀⋆𝐔superscript𝐕top{\bm{A}}\_{\star}={\bm{U}}{\bm{V}}^{\top}.

All that remains is to explore the uniqueness of this solution:

* •

  If 𝑮𝑮{\bm{G}} is full rank, the solution 𝑨⋆subscript𝑨⋆{\bm{A}}\_{\star} is unique. 𝑮𝑮{\bm{G}} being full rank means that all the singular values σisubscript𝜎𝑖\sigma\_{i} are positive. In this case, we see from [Equation 45](#A2.E45 "In Proof B.4. ‣ 4: \nameref*prop:projection ‣ Appendix B Proofs ‣ Old Optimizer, New Norm: An Anthology") that to maximize the alignment the semi-orthogonal matrix 𝑨𝑨{\bm{A}} must satisfy 𝒖i⊤​𝑨​𝒗i=1superscriptsubscript𝒖𝑖top𝑨subscript𝒗𝑖1{\bm{u}}\_{i}^{\top}{\bm{A}}{\bm{v}}\_{i}=1 for all i𝑖i. Since 𝑨𝑨{\bm{A}} has spectral norm one, in turn this requires that
  𝑨​𝒗i=𝒖i𝑨subscript𝒗𝑖subscript𝒖𝑖{\bm{A}}{\bm{v}}\_{i}={\bm{u}}\_{i} and 𝑨⊤​𝒖i=𝒗isuperscript𝑨topsubscript𝒖𝑖subscript𝒗𝑖{\bm{A}}^{\top}{\bm{u}}\_{i}={\bm{v}}\_{i} for all i𝑖i. These conditions uniquely pick out 𝑨=∑i𝒖i​𝒗i⊤𝑨subscript𝑖subscript𝒖𝑖superscriptsubscript𝒗𝑖top{\bm{A}}=\sum\_{i}{\bm{u}}\_{i}{\bm{v}}\_{i}^{\top}.
* •

  If 𝑮𝑮{\bm{G}} is not full rank then the solution 𝑨⋆subscript𝑨⋆{\bm{A}}\_{\star} is not unique. This solution is just as good:

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | 𝑨†=∑i:σi>0𝒖i​𝒗i⊤+∑i:σi=0𝒖i​(−𝒗i)⊤.subscript𝑨†subscript:𝑖subscript𝜎𝑖0subscript𝒖𝑖superscriptsubscript𝒗𝑖topsubscript:𝑖subscript𝜎𝑖0subscript𝒖𝑖superscriptsubscript𝒗𝑖top{\bm{A}}\_{\dagger}=\sum\_{i:\sigma\_{i}>0}{\bm{u}}\_{i}{\bm{v}}\_{i}^{\top}+\sum\_{i:\sigma\_{i}=0}{\bm{u}}\_{i}(-{\bm{v}}\_{i})^{\top}. |  | (47) |

This completes the proof.

### [5](#Thmmyproposition5 "Proposition 5 (Shampoo as steepest descent under the spectral norm) ‣ Story II Shampoo as Steepest Descent under the Spectral Norm ‣ Old Optimizer, New Norm: An Anthology"): \nameref\*prop:shampoo-steepest

###### Proof B.5.

First, we apply [7](#Thmmyproposition7 "Proposition 7 (Steepest descent under the modular norm) ‣ Epilogue ‣ Old Optimizer, New Norm: An Anthology") with scalars s1,…,sL

subscript𝑠1…subscript𝑠𝐿s\_{1},...,s\_{L} set to one and all norms set to ∥⋅∥ℓ2→ℓ2\|{\cdot}\|\_{\ell\_{2}\to\ell\_{2}}. This tells us that the solution is given by Δ​𝐖l=−η⋅arg​max‖𝐓l‖l=1⁡tr⁡(𝐆l⊤​𝐓l)Δsubscript𝐖𝑙⋅𝜂subscriptargmaxsubscriptnormsubscript𝐓𝑙𝑙1trsuperscriptsubscript𝐆𝑙topsubscript𝐓𝑙\Delta{\bm{W}}\_{l}=-\eta\cdot\operatorname\*{arg\,max}\_{\|{{\bm{T}}\_{l}}\|\_{l}=1}\operatorname{tr}({\bm{G}}\_{l}^{\top}{\bm{T}}\_{l}) for each l=1,…,L𝑙

1…𝐿l=1,...,L and with η=1λ​∑k=1L‖𝐆k‖ℓ2→ℓ2†𝜂1𝜆superscriptsubscript𝑘1𝐿superscriptsubscriptnormsubscript𝐆𝑘→subscriptℓ2subscriptℓ2†\eta=\frac{1}{\lambda}\sum\_{k=1}^{L}\|{{\bm{G}}\_{k}}\|\_{\ell\_{2}\to\ell\_{2}}^{\dagger}. We just need to resolve the dual norm and evaluate the argmax.

Let’s start with the dual norm. For a matrix 𝐆𝐆{\bm{G}} with SVD ∑iσi​𝐮i​𝐯i⊤=𝐔​𝚺​𝐕⊤subscript𝑖subscript𝜎𝑖subscript𝐮𝑖superscriptsubscript𝐯𝑖top𝐔𝚺superscript𝐕top\sum\_{i}\sigma\_{i}\,{\bm{u}}\_{i}{\bm{v}}\_{i}^{\top}={\bm{U}}{\bm{\Sigma}}{\bm{V}}^{\top} we have:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖𝑮‖ℓ2→ℓ2†:=max‖𝑻‖ℓ2→ℓ2=1⁡tr⁡𝑮⊤​𝑻=tr​∑iσi​𝒗i​𝒖i⊤​𝑻=∑iσi​𝒖i⊤​𝑻​𝒗i≤∑iσi=tr⁡𝚺,assignsuperscriptsubscriptnorm𝑮→subscriptℓ2subscriptℓ2†subscriptsubscriptnorm𝑻→subscriptℓ2subscriptℓ21trsuperscript𝑮top𝑻trsubscript𝑖subscript𝜎𝑖subscript𝒗𝑖superscriptsubscript𝒖𝑖top𝑻subscript𝑖subscript𝜎𝑖superscriptsubscript𝒖𝑖top𝑻subscript𝒗𝑖subscript𝑖subscript𝜎𝑖tr𝚺\|{{\bm{G}}}\|\_{\ell\_{2}\to\ell\_{2}}^{\dagger}\vcentcolon=\max\_{\|{{\bm{T}}}\|\_{\ell\_{2}\to\ell\_{2}}=1}\operatorname{tr}{\bm{G}}^{\top}{\bm{T}}=\operatorname{tr}\sum\_{i}\sigma\_{i}\,{\bm{v}}\_{i}{\bm{u}}\_{i}^{\top}{\bm{T}}=\sum\_{i}\sigma\_{i}\,{\bm{u}}\_{i}^{\top}{\bm{T}}{\bm{v}}\_{i}\leq\sum\_{i}\sigma\_{i}=\operatorname{tr}{\bm{\Sigma}}, |  | (48) |

where the upper bound follows from the spectral norm constraint on 𝐓𝐓{\bm{T}}. But this upper bound is attained by setting 𝐓=𝐔​𝐕⊤𝐓𝐔superscript𝐕top{\bm{T}}={\bm{U}}{\bm{V}}^{\top} (also resolving the argmax) and so ‖𝐆‖ℓ2→ℓ2†=tr⁡𝚺superscriptsubscriptnorm𝐆→subscriptℓ2subscriptℓ2†tr𝚺\|{{\bm{G}}}\|\_{\ell\_{2}\to\ell\_{2}}^{\dagger}=\operatorname{tr}{\bm{\Sigma}}.

The uniqueness claim follows by the same argument as for [4](#Thmmyproposition4 "Proposition 4 (Projection to the closest semi-orthogonal matrix) ‣ Story II Shampoo as Steepest Descent under the Spectral Norm ‣ Old Optimizer, New Norm: An Anthology").

### [6](#Thmmyproposition6 "Proposition 6 (Bounding the square loss of a linear predictor) ‣ Story II Shampoo as Steepest Descent under the Spectral Norm ‣ Old Optimizer, New Norm: An Anthology"): \nameref\*prop:majorization

###### Proof B.6.

First observe that the square loss is quadratic in 𝐖𝐖{\bm{W}} so there are no cubic terms or higher. The bound must agree to first-order with the first-order Taylor expansion of ℒ​(𝐖+Δ​𝐖)ℒ𝐖Δ𝐖\mathcal{L}({\bm{W}}+\Delta{\bm{W}}), which is precisely ℒ​(𝐖)+⟨∇𝐖ℒ​(𝐖),Δ​𝐖⟩ℒ𝐖

subscript∇𝐖ℒ𝐖Δ𝐖\mathcal{L}({\bm{W}})+\langle\nabla\_{\bm{W}}\mathcal{L}({\bm{W}}),\Delta{\bm{W}}\rangle, since otherwise the bound would be violated for sufficiently small Δ​𝐖Δ𝐖\Delta{\bm{W}}. To obtain the second-order piece of the bound, it’s easiest just to multiply out ℒ​(𝐖+Δ​𝐖)ℒ𝐖Δ𝐖\mathcal{L}({\bm{W}}+\Delta{\bm{W}}) and see that the second-order piece of ℒ​(𝐖+Δ​𝐖)ℒ𝐖Δ𝐖\mathcal{L}({\bm{W}}+\Delta{\bm{W}}) satisfies:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 12​n​∑i=1n1dout​‖Δ​𝑾​𝒙(i)‖22≤12​n​∑i=1n1dout​‖Δ​𝑾‖ℓ2→ℓ22⋅‖𝒙(i)‖22=12​dindout​‖Δ​𝑾‖ℓ2→ℓ22,12𝑛superscriptsubscript𝑖1𝑛1subscript𝑑outsuperscriptsubscriptnormΔ𝑾superscript𝒙𝑖2212𝑛superscriptsubscript𝑖1𝑛⋅1subscript𝑑outsuperscriptsubscriptnormΔ𝑾→subscriptℓ2subscriptℓ22superscriptsubscriptnormsuperscript𝒙𝑖2212subscript𝑑insubscript𝑑outsuperscriptsubscriptnormΔ𝑾→subscriptℓ2subscriptℓ22\displaystyle\frac{1}{2n}\sum\_{i=1}^{n}\frac{1}{d\_{\mathrm{out}}}\|{\Delta{\bm{W}}{\bm{x}}^{(i)}}\|\_{2}^{2}\leq\frac{1}{2n}\sum\_{i=1}^{n}\frac{1}{d\_{\mathrm{out}}}\|{\Delta{\bm{W}}}\|\_{\ell\_{2}\to\ell\_{2}}^{2}\cdot\|{{\bm{x}}^{(i)}}\|\_{2}^{2}=\frac{1}{2}\frac{d\_{\mathrm{in}}}{d\_{\mathrm{out}}}\|{\Delta{\bm{W}}}\|\_{\ell\_{2}\to\ell\_{2}}^{2}, |  | (49) |

where the last equality uses the input normalization ‖𝐱(i)‖2=dinsubscriptnormsuperscript𝐱𝑖2subscript𝑑in\|{{\bm{x}}^{(i)}}\|\_{2}=\sqrt{d\_{\mathrm{in}}}. We are done.

### [7](#Thmmyproposition7 "Proposition 7 (Steepest descent under the modular norm) ‣ Epilogue ‣ Old Optimizer, New Norm: An Anthology"): \nameref\*prop:steepest-modular

###### Proof B.7.

For each layer l=1,…,L𝑙

1…𝐿l=1,...,L, we decompose Δ​𝐖lΔsubscript𝐖𝑙\Delta{\bm{W}}\_{l} into its magnitude and direction: Δ​𝐖l=cl⋅𝐓lΔsubscript𝐖𝑙⋅subscript𝑐𝑙subscript𝐓𝑙\Delta{\bm{W}}\_{l}=c\_{l}\cdot{\bm{T}}\_{l}, for cl≥0subscript𝑐𝑙0c\_{l}\geq 0 and ‖𝐓l‖l=1subscriptnormsubscript𝐓𝑙𝑙1\|{{\bm{T}}\_{l}}\|\_{l}=1. Under this change of variables, the minimization becomes:

|  |  |  |  |
| --- | --- | --- | --- |
|  | minΔ​𝑾1,…,Δ​𝑾L⁡[∑l=1L⟨𝑮l,Δ​𝑾l⟩+λ2​maxl=1L⁡sl2​‖Δ​𝑾l‖l2]subscript  Δsubscript𝑾1…Δsubscript𝑾𝐿superscriptsubscript𝑙1𝐿  subscript𝑮𝑙Δsubscript𝑾𝑙𝜆2superscriptsubscript𝑙1𝐿superscriptsubscript𝑠𝑙2superscriptsubscriptnormΔsubscript𝑾𝑙𝑙2\displaystyle\min\_{\Delta{\bm{W}}\_{1},\dots,\Delta{\bm{W}}\_{L}}\left[\sum\_{l=1}^{L}\langle{\bm{G}}\_{l},\Delta{\bm{W}}\_{l}\rangle+\frac{\lambda}{2}\max\_{l=1}^{L}s\_{l}^{2}\|{\Delta{\bm{W}}\_{l}}\|\_{l}^{2}\right] |  | (50) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =minc1,…,cL≥0⁡[∑l=1Lcl​min‖𝑻l‖l=1⁡⟨𝑮l,𝑻l⟩+λ2​maxl=1L⁡sl2​cl2]absentsubscript  subscript𝑐1…subscript𝑐𝐿 0superscriptsubscript𝑙1𝐿subscript𝑐𝑙subscriptsubscriptnormsubscript𝑻𝑙𝑙1subscript𝑮𝑙subscript𝑻𝑙𝜆2superscriptsubscript𝑙1𝐿superscriptsubscript𝑠𝑙2superscriptsubscript𝑐𝑙2\displaystyle\qquad=\min\_{c\_{1},\dots,c\_{L}\geq 0}\left[\sum\_{l=1}^{L}c\_{l}\min\_{\|{{\bm{T}}\_{l}}\|\_{l}=1}\langle{\bm{G}}\_{l},{\bm{T}}\_{l}\rangle+\frac{\lambda}{2}\max\_{l=1}^{L}s\_{l}^{2}c\_{l}^{2}\right] |  | (51) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =minc1,…,cL≥0⁡[−∑l=1Lcl​‖𝑮l‖l†+λ2​maxl=1L⁡sl2​cl2]absentsubscript  subscript𝑐1…subscript𝑐𝐿 0superscriptsubscript𝑙1𝐿subscript𝑐𝑙superscriptsubscriptnormsubscript𝑮𝑙𝑙†𝜆2superscriptsubscript𝑙1𝐿superscriptsubscript𝑠𝑙2superscriptsubscript𝑐𝑙2\displaystyle\qquad=\min\_{c\_{1},\dots,c\_{L}\geq 0}\left[-\sum\_{l=1}^{L}c\_{l}\|{{\bm{G}}\_{l}}\|\_{l}^{\dagger}+\frac{\lambda}{2}\max\_{l=1}^{L}s\_{l}^{2}c\_{l}^{2}\right] |  | (52) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =minη≥0⁡[−∑l=1Lηsl​‖𝑮l‖l†+λ2​η2],absentsubscript𝜂0superscriptsubscript𝑙1𝐿𝜂subscript𝑠𝑙superscriptsubscriptnormsubscript𝑮𝑙𝑙†𝜆2superscript𝜂2\displaystyle\qquad=\min\_{\eta\geq 0}\left[-\sum\_{l=1}^{L}\frac{\eta}{s\_{l}}\|{{\bm{G}}\_{l}}\|\_{l}^{\dagger}+\frac{\lambda}{2}\eta^{2}\right], |  | (53) |

where [Equation 53](#A2.E53 "In Proof B.7. ‣ 7: \nameref*prop:steepest-modular ‣ Appendix B Proofs ‣ Old Optimizer, New Norm: An Anthology") follows by observing that at the minimum we must have s1​c1,…,sL​cL

subscript𝑠1subscript𝑐1…subscript𝑠𝐿subscript𝑐𝐿s\_{1}c\_{1},...,s\_{L}c\_{L} all taking the same value of η≥0𝜂0\eta\geq 0 (still to be determined), since otherwise we could increase the sum ∑lcl​‖𝐆l‖l†subscript𝑙subscript𝑐𝑙superscriptsubscriptnormsubscript𝐆𝑙𝑙†\sum\_{l}c\_{l}\|{{\bm{G}}\_{l}}\|\_{l}^{\dagger} by increasing any of the slack clsubscript𝑐𝑙c\_{l} without paying a penalty in terms of the max. We can now read off the minimizers from [Equations 51](#A2.E51 "In Proof B.7. ‣ 7: \nameref*prop:steepest-modular ‣ Appendix B Proofs ‣ Old Optimizer, New Norm: An Anthology"), [52](#A2.E52 "Equation 52 ‣ Proof B.7. ‣ 7: \nameref*prop:steepest-modular ‣ Appendix B Proofs ‣ Old Optimizer, New Norm: An Anthology") and [53](#A2.E53 "Equation 53 ‣ Proof B.7. ‣ 7: \nameref*prop:steepest-modular ‣ Appendix B Proofs ‣ Old Optimizer, New Norm: An Anthology"):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝑻lsubscript𝑻𝑙\displaystyle{\bm{T}}\_{l} | =arg​min‖𝑻l‖l=1⁡⟨𝑮l,𝑻l⟩=−arg​max‖𝑻l‖l=1⁡⟨𝑮l,𝑻l⟩;absentsubscriptargminsubscriptnormsubscript𝑻𝑙𝑙1subscript𝑮𝑙subscript𝑻𝑙subscriptargmaxsubscriptnormsubscript𝑻𝑙𝑙1subscript𝑮𝑙subscript𝑻𝑙\displaystyle=\operatorname\*{arg\,min}\_{\|{{\bm{T}}\_{l}}\|\_{l}=1}\,\langle{\bm{G}}\_{l},{\bm{T}}\_{l}\rangle=-\operatorname\*{arg\,max}\_{\|{{\bm{T}}\_{l}}\|\_{l}=1}\,\langle{\bm{G}}\_{l},{\bm{T}}\_{l}\rangle; |  | (54) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | clsubscript𝑐𝑙\displaystyle c\_{l} | =ηsl;absent𝜂subscript𝑠𝑙\displaystyle=\frac{\eta}{s\_{l}}; |  | (55) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | η𝜂\displaystyle\eta | =1λ​∑k=1L1sk​‖𝑮k‖k†.absent1𝜆superscriptsubscript𝑘1𝐿1subscript𝑠𝑘superscriptsubscriptnormsubscript𝑮𝑘𝑘†\displaystyle=\frac{1}{\lambda}\sum\_{k=1}^{L}\frac{1}{s\_{k}}\|{{\bm{G}}\_{k}}\|\_{k}^{\dagger}. |  | (56) |

Combining, we obtain the overall minimizer for each l=1,…,L𝑙

1…𝐿l=1,...,L via Δ​𝐖l=cl⋅𝐓l=−ηsl​arg​max⁡⟨𝐆l,𝐓l⟩Δsubscript𝐖𝑙⋅subscript𝑐𝑙subscript𝐓𝑙𝜂subscript𝑠𝑙argmaxsubscript𝐆𝑙subscript𝐓𝑙\Delta{\bm{W}}\_{l}=c\_{l}\cdot{\bm{T}}\_{l}=-\frac{\eta}{s\_{l}}\operatorname\*{arg\,max}\,\langle{\bm{G}}\_{l},{\bm{T}}\_{l}\rangle, where η𝜂\eta is given by [Equation 56](#A2.E56 "In Proof B.7. ‣ 7: \nameref*prop:steepest-modular ‣ Appendix B Proofs ‣ Old Optimizer, New Norm: An Anthology"), proving the result.

### [8](#Thmmyproposition8 "Proposition 8 (ℓ₁→ℓ_𝑝 and ℓ_𝑝→ℓ_∞ induced operator norms are tractable) ‣ Epilogue ‣ Old Optimizer, New Norm: An Anthology"): \nameref\*prop:tractable-norms

###### Proof B.8.

Let’s start with the ℓ1→ℓp→subscriptℓ1subscriptℓ𝑝\ell\_{1}\to\ell\_{p} operator norm. Here we observe that, in matrix-vector multiplication, each component of an input vector selects and scales a column of the matrix:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝑴‖ℓ1→ℓp=max‖𝒙‖1=1⁡‖𝑴​𝒙‖p=max‖𝒙‖1=1⁡‖∑jcolj​(𝑴)​𝒙j‖psubscriptnorm𝑴→subscriptℓ1subscriptℓ𝑝subscriptsubscriptnorm𝒙11subscriptnorm𝑴𝒙𝑝subscriptsubscriptnorm𝒙11subscriptnormsubscript𝑗subscriptcol𝑗𝑴subscript𝒙𝑗𝑝\displaystyle\|{{\bm{M}}}\|\_{\ell\_{1}\to\ell\_{p}}=\max\_{\|{{\bm{x}}}\|\_{1}=1}\|{{\bm{M}}{\bm{x}}}\|\_{p}=\max\_{\|{{\bm{x}}}\|\_{1}=1}\Big{\|}\sum\_{j}\mathrm{col}\_{j}({\bm{M}}){\bm{x}}\_{j}\Big{\|}\_{p} | ≤max‖𝒙‖1=1​∑j|𝒙j|⋅‖colj​(𝑴)‖pabsentsubscriptsubscriptnorm𝒙11subscript𝑗⋅subscript𝒙𝑗subscriptnormsubscriptcol𝑗𝑴𝑝\displaystyle\leq\max\_{\|{{\bm{x}}}\|\_{1}=1}\sum\_{j}|{{\bm{x}}\_{j}}|\cdot\|{\mathrm{col}\_{j}({\bm{M}})}\|\_{p} |  | (57) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ≤max‖𝒙‖1=1⁡‖𝒙‖1⋅maxj⁡‖colj​(𝑴)‖pabsentsubscriptsubscriptnorm𝒙11⋅subscriptnorm𝒙1subscript𝑗subscriptnormsubscriptcol𝑗𝑴𝑝\displaystyle\leq\max\_{\|{{\bm{x}}}\|\_{1}=1}\|{{\bm{x}}}\|\_{1}\cdot\max\_{j}\|{\mathrm{col}\_{j}({\bm{M}})}\|\_{p} |  | (58) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =maxj⁡‖colj​(𝑴)‖p,absentsubscript𝑗subscriptnormsubscriptcol𝑗𝑴𝑝\displaystyle=\max\_{j}\|{\mathrm{col}\_{j}({\bm{M}})}\|\_{p}, |  | (59) |

by the triangle inequality and Hölder’s inequality. But the upper bound in [Equation 59](#A2.E59 "In Proof B.8. ‣ 8: \nameref*prop:tractable-norms ‣ Appendix B Proofs ‣ Old Optimizer, New Norm: An Anthology") is attained by selecting the column index j⋆=arg​maxj⁡‖colj​(𝐌)‖psubscript𝑗⋆subscriptargmax𝑗subscriptnormsubscriptcol𝑗𝐌𝑝j\_{\star}=\operatorname\*{arg\,max}\_{j}\|{\mathrm{col}\_{j}({\bm{M}})}\|\_{p} with the largest norm, then setting 𝐱j⋆=1subscript𝐱subscript𝑗⋆1{\bm{x}}\_{j\_{\star}}=1 and the other input components to zero. So ‖𝐌‖ℓ1→ℓp=maxj⁡‖colj​(𝐌)‖p.subscriptnorm𝐌→subscriptℓ1subscriptℓ𝑝subscript𝑗subscriptnormsubscriptcol𝑗𝐌𝑝\|{{\bm{M}}}\|\_{\ell\_{1}\to\ell\_{p}}=\max\_{j}\|{\mathrm{col}\_{j}({\bm{M}})}\|\_{p}.

Next, let’s deal with the ℓp→ℓ∞→subscriptℓ𝑝subscriptℓ\ell\_{p}\to\ell\_{\infty} operator norm. Here we break up a matrix-vector product in terms of the dot product between the vector and the matrix rows:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖𝑴‖ℓp→ℓ∞=max‖𝒙‖p=1⁡‖𝑴​𝒙‖∞subscriptnorm𝑴→subscriptℓ𝑝subscriptℓsubscriptsubscriptnorm𝒙𝑝1subscriptnorm𝑴𝒙\displaystyle\|{{\bm{M}}}\|\_{\ell\_{p}\to\ell\_{\infty}}=\max\_{\|{{\bm{x}}}\|\_{p}=1}\|{{\bm{M}}{\bm{x}}}\|\_{\infty} | =max‖𝒙‖p=1⁡maxi⁡|𝒙⊤​rowi​(𝑴)|absentsubscriptsubscriptnorm𝒙𝑝1subscript𝑖superscript𝒙topsubscriptrow𝑖𝑴\displaystyle=\max\_{\|{{\bm{x}}}\|\_{p}=1}\max\_{i}|{{\bm{x}}^{\top}\mathrm{row}\_{i}({\bm{M}})}| |  | (60) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =maxi⁡max‖𝒙‖p=1⁡|𝒙⊤​rowi​(𝑴)|absentsubscript𝑖subscriptsubscriptnorm𝒙𝑝1superscript𝒙topsubscriptrow𝑖𝑴\displaystyle=\max\_{i}\max\_{\|{{\bm{x}}}\|\_{p}=1}|{{\bm{x}}^{\top}\mathrm{row}\_{i}({\bm{M}})}| |  | (61) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =maxi⁡‖rowi​(𝑴)‖p†.absentsubscript𝑖superscriptsubscriptnormsubscriptrow𝑖𝑴𝑝†\displaystyle=\max\_{i}\|{\mathrm{row}\_{i}({\bm{M}})}\|\_{p}^{\dagger}. |  | (62) |

The proof is completed by recalling that the vector ℓpsubscriptℓ𝑝\ell\_{p} norm is dual to the vector ℓqsubscriptℓ𝑞\ell\_{q} norm for 1/p+1/q=11𝑝1𝑞11/p+1/q=1. In other words, ∥⋅∥p†=∥⋅∥pp−1\|{\cdot}\|\_{p}^{\dagger}=\|{\cdot}\|\_{\frac{p}{p-1}}.

[◄](/html/2409.20324)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2409.20325)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2409.20325)
[View original  
on arXiv](https://arxiv.org/abs/2409.20325)[►](/html/2409.20326)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Sat Oct 5 20:05:52 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
