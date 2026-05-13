---
arxiv: '2604.20446'
authors:
- Elon Litman
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: The Origin of Edge of Stability
url: https://arxiv.org/abs/2604.20446
year: 2026
---

[2604.20446] The Origin of Edge of Stability














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



# The Origin of Edge of Stability

Elon Litman
Affiliation: Stanford University
Affiliation: elonlit@stanford.edu

###### Abstract

Full-batch gradient descent on neural networks drives the largest Hessian eigenvalue to the threshold 2/η2/\eta, where η\eta is the learning rate. This phenomenon, the *Edge of Stability*, has resisted a unified explanation: existing accounts establish self-regulation near the edge but do not explain why the trajectory is forced toward 2/η2/\eta from arbitrary initialization. We introduce the *edge coupling*, a functional on consecutive iterate pairs whose coefficient is uniquely fixed by the gradient-descent update. Differencing its criticality condition yields a step recurrence with stability boundary 2/η2/\eta, and a second-order expansion yields a loss-change formula whose telescoping sum forces curvature toward 2/η2/\eta. The two formulas involve different Hessian averages, but the mean value theorem localizes each to the true Hessian at an interior point of the step segment, yielding exact forcing of the Hessian eigenvalue with no gap. Setting both gradients of the edge coupling to zero classifies fixed points and period-two orbits; near a fixed point, the problem reduces to a function of the half-amplitude alone, which determines which directions support period-two orbits and on which side of the critical learning rate they appear.

## 1 Introduction

#### The Edge of Stability.

Classical optimization theory guarantees monotonic loss decrease whenever η<2/λmax​(∇2L)\eta<2/\lambda\_{\max}(\nabla^{2}L), where λmax​(∇2L)\lambda\_{\max}(\nabla^{2}L) is the largest Hessian eigenvalue, called the *sharpness* (nesterov2004introductory; nocedal2006numerical). cohen2021gradient discovered that in practice the opposite occurs: when a fixed learning rate is used, the sharpness rises during training until it reaches the value 2/η2/\eta, at which point it saturates and the training loss begins to oscillate on short timescales while continuing to decrease on longer ones. They termed this the *Edge of Stability* (EoS) and documented it across architectures, datasets, and loss functions. In the closely related *catapult phase* (lewkowycz2020large), large learning rates initially drive sharpness down before progressive sharpening returns it to 2/η2/\eta. Progressive sharpening had been studied by jastrzebski2017three, wu2018sgd, and ghorbani2019investigation, who connected it to the Hessian spectrum and implicit selection of flat minima; lyu2020gradient established a related bias toward margin maximization. Together, these observations established 2/η2/\eta as a universal threshold of full-batch gradient descent. [Figure 1](#S1.F1 "Figure 1 ‣ The Edge of Stability. ‣ 1 Introduction ‣ The Origin of Edge of Stability") illustrates both phases.

![Refer to caption](/html/2604.20446/assets/x1.png)

![Refer to caption](/html/2604.20446/assets/x2.png)

Figure 1: Edge of Stability on a 3-layer MLP (CIFAR-10).
Full-batch GD, η=0.5\eta=0.5, GELU activations, MSE loss.
Dotted line marks tct\_{c}, the first step at which r~k≈2/η\widetilde{r}\_{k}\approx 2/\eta.
a, Effective curvature r~k\widetilde{r}\_{k} (blue) and sharpness λmax\lambda\_{\max} (green); both saturate near 2/η=42/\eta=4 (dashed).
b, Training loss (solid) and 5-step running mean (dashed). Inset: detrended loss Lk−L¯kL\_{k}-\bar{L}\_{k} showing oscillation.

#### Prior work.

Several complementary lines of work have sought to explain why gradient descent saturates at 2/η2/\eta. One line shows that discrete gradient descent implicitly regularizes toward flat minima through a gradient-norm penalty (barrett2021implicit; smith2021origin; keskar2017large; hochreiter1997flat), with related implicit biases in the stochastic, classification, and large-learning-rate settings (blanc2020implicit; lyu2020gradient; li2019towards). A second line addresses local dynamics at the edge: damian2023selfstabilization proved that cubic Taylor terms create a feedback loop reducing sharpness near 2/η2/\eta, and agarwala2022secondorder and even2023sgd obtained analogous results under spectral conditions. Further work has analyzed unstable convergence beyond the classical threshold (ahn2022understanding; arora2022understanding), connected curvature to large-scale training instabilities (gilmer2022loss; lyu2022understanding), and related convergence rate to sharpness (ma2022multiscale).

#### The edge coupling.

Despite this progress, existing results (damian2023selfstabilization; agarwala2022secondorder; even2023sgd) are inherently local: they establish self-regulation near the edge but do not explain why the trajectory is forced toward 2/η2/\eta from arbitrary initialization. We take a different approach. Rather than analyzing the dynamics step by step, we ask: is there a scalar functional on consecutive iterate pairs whose criticality conditions encode gradient descent? Consider the family of symmetric couplings

|  |  |  |  |
| --- | --- | --- | --- |
|  | L​(x)+L​(y)−α2​‖x−y‖2.\displaystyle L(x)+L(y)-\frac{\alpha}{2}\|x-y\|^{2}. |  | (1) |

Setting the xx-gradient to zero gives y=x−α−1​∇L​(x)y=x-\alpha^{-1}\nabla L(x), which is the gradient-descent update if and only if α=η−1\alpha=\eta^{-1}. We call the resulting functional

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒜η​(x,y)≜L​(x)+L​(y)−12​η​‖x−y‖22\displaystyle\mathcal{A}\_{\eta}(x,y)\;\triangleq\;L(x)+L(y)-\frac{1}{2\eta}\|x-y\|\_{2}^{2} |  | (2) |

the *edge coupling*: it is attached to one edge of the discrete trajectory, and the threshold it produces governs the Edge of Stability. Viewed through the lens of Hamiltonian mechanics, this functional can be considered a discrete generating function whose criticality conditions encode gradient descent in boundary-value form (marsden2001discrete; arnol2013mathematical).

#### Contributions.

Every result in this paper follows from the criticality conditions of 𝒜η\mathcal{A}\_{\eta}. Setting the xx-gradient to zero recovers the gradient-descent update; differencing this condition between consecutive steps yields a recurrence with stability boundary 2/η2/\eta, and a second-order expansion yields a loss-change formula ([Section˜2](#S2 "2 The Edge Coupling ‣ The Origin of Edge of Stability")). These two formulas involve different Hessian averages, but we show that the mean value theorem localizes each to the true Hessian at an interior point of the step segment. Summing the loss-change formula then forces the localized Hessian eigenvalue toward 2/η2/\eta exactly ([Section˜4](#S4 "4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability")). Setting both partial gradients to zero classifies all fixed points and period-two orbits; near a fixed point, the problem reduces to a function of the half-amplitude alone, which determines which directions support period-two orbits and on which side of the critical learning rate they appear ([Section˜2](#S2 "2 The Edge Coupling ‣ The Origin of Edge of Stability")). For two-layer linear networks, this construction is width-invariant and the period-doubling branch appears continuously on the large-learning-rate side of the threshold ([Proposition˜2.6](#S2.Thmtheorem6 "Proposition 2.6 (Transverse edge normal form for two-layer linear networks). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability")). [Section˜5](#S5 "5 Stability Mechanisms at the Edge ‣ The Origin of Edge of Stability") explains why the system remains stable at the edge. The appendices extend the forcing theorems to mini-batch SGD ([appendix˜D](#A4 "Appendix D Curvature Concentration under Mini-Batch SGD ‣ The Origin of Edge of Stability")), to pairs of trajectories ([appendix˜E](#A5 "Appendix E Discrete-Time Kelvin–Voigt Dynamics ‣ The Origin of Edge of Stability")), and to continuous time ([appendix˜F](#A6 "Appendix F Continuous-Time Kelvin–Voigt Framework ‣ The Origin of Edge of Stability")).

## 2 The Edge Coupling

We now develop the consequences of the edge coupling 𝒜η\mathcal{A}\_{\eta}, referring to consecutive pairs (wk,wk+1)(w\_{k},w\_{k+1}) as the *edges* of the trajectory. Since a period-two orbit alternates symmetrically about a center, it is natural to write (x,y)=(m−a,m+a)(x,y)=(m-a,\,m+a), which gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒜η​(m−a,m+a)=2​Ψη​(m,a),Ψη​(m,a)≜12​(L​(m+a)+L​(m−a))−1η​‖a‖2,\displaystyle\mathcal{A}\_{\eta}(m-a,m+a)=2\Psi\_{\eta}(m,a),\qquad\Psi\_{\eta}(m,a)\triangleq\tfrac{1}{2}\bigl(L(m+a)+L(m-a)\bigr)-\tfrac{1}{\eta}\|a\|^{2}, |  | (3) |

This reparametrization separates the center mm from the half-amplitude aa of the oscillation and will be central to the period-two analysis in [Theorem˜2.3](#S2.Thmtheorem3 "Theorem 2.3 (Center reduction and the edge eigenproblem). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability"). The first theorem identifies the critical points of 𝒜η\mathcal{A}\_{\eta}, and the second derives the dynamical consequences.

###### Theorem 2.1 (Variational characterization of gradient descent).

Let L:ℝd→ℝL:\mathbb{R}^{d}\to\mathbb{R} be C2C^{2}, let η>0\eta>0, and define

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒜η​(x,y)≜L​(x)+L​(y)−12​η​‖x−y‖22.\displaystyle\mathcal{A}\_{\eta}(x,y)\;\triangleq\;L(x)+L(y)-\frac{1}{2\eta}\|x-y\|\_{2}^{2}. |  | (4) |

Write Γη≜{(x,y)∈ℝd×ℝd:∇x𝒜η​(x,y)=0}\Gamma\_{\eta}\triangleq\{(x,y)\in\mathbb{R}^{d}\times\mathbb{R}^{d}:\nabla\_{x}\mathcal{A}\_{\eta}(x,y)=0\} for the set of partial critical points in xx, and for a gradient-descent trajectory wk+1=wk−η​∇L​(wk)w\_{k+1}=w\_{k}-\eta\nabla L(w\_{k}) set dk≜wk+1−wkd\_{k}\triangleq w\_{k+1}-w\_{k}.

(i) The edge manifold Γη\Gamma\_{\eta} coincides with the graph of the gradient-descent map:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Γη={(x,y):y=x−η​∇L​(x)}.\displaystyle\Gamma\_{\eta}=\{(x,y):y=x-\eta\nabla L(x)\}. |  | (5) |

In particular, every consecutive pair (wk,wk+1)(w\_{k},w\_{k+1}) lies on Γη\Gamma\_{\eta}.

(ii) The yy-gradient measures how far the current edge is from closing a period-two orbit. On a GD edge it gives the two-step displacement:

|  |  |  |  |
| --- | --- | --- | --- |
|  | −η​∇y𝒜η​(wk,wk+1)=wk+2−wk=dk+dk+1.\displaystyle-\eta\,\nabla\_{y}\mathcal{A}\_{\eta}(w\_{k},w\_{k+1})=w\_{k+2}-w\_{k}=d\_{k}+d\_{k+1}. |  | (6) |

Setting both partial gradients to zero therefore imposes the pair of update equations

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇𝒜η​(x,y)=0⇔{y=x−η​∇L​(x),x=y−η​∇L​(y),\displaystyle\nabla\mathcal{A}\_{\eta}(x,y)=0\quad\iff\quad\begin{cases}y=x-\eta\nabla L(x),\\ x=y-\eta\nabla L(y),\end{cases} |  | (7) |

so the full critical points of 𝒜η\mathcal{A}\_{\eta} are the fixed points (x=y)(x=y)
and the period-two orbits (x≠y)(x\neq y) of gradient descent.

*Proof sketch.* The result follows by direct differentiation: ∇x𝒜η=0\nabla\_{x}\mathcal{A}\_{\eta}=0 rearranges to the GD update, and evaluating ∇y𝒜η\nabla\_{y}\mathcal{A}\_{\eta} on a GD edge yields the two-step displacement. See [appendix˜A](#A1 "Appendix A Proofs from the Main Text ‣ The Origin of Edge of Stability").

We now extract the content of [Theorem˜2.1](#S2.Thmtheorem1 "Theorem 2.1 (Variational characterization of gradient descent). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability"). Along the step segment wk+τ​dkw\_{k}+\tau d\_{k} for τ∈[0,1]\tau\in[0,1], two Hessian averages arise naturally: a uniform average H¯k\bar{H}\_{k} from differencing gradients, and a triangularly weighted average H~k\widetilde{H}\_{k} from expanding the loss.

###### Theorem 2.2 (Propagator and One-Step Loss Change).

In the setting of [Theorem˜2.1](#S2.Thmtheorem1 "Theorem 2.1 (Variational characterization of gradient descent). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability"), define the step-averaged Hessians and effective curvature

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | H¯k≜∫01∇2L​(wk+τ​dk)​𝑑τ\displaystyle\bar{H}\_{k}\triangleq\int\_{0}^{1}\nabla^{2}L(w\_{k}+\tau d\_{k})\,d\tau | ,H~k≜2∫01(1−τ)∇2L(wk+τdk)dτ,\displaystyle,\quad\widetilde{H}\_{k}\triangleq 2\int\_{0}^{1}(1-\tau)\nabla^{2}L(w\_{k}+\tau d\_{k})\,d\tau, |  | (8) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | r~k≜dk⊤​H~k​dk‖dk‖22​(dk≠0).\displaystyle\widetilde{r}\_{k}\triangleq\frac{d\_{k}^{\top}\widetilde{H}\_{k}d\_{k}}{\|d\_{k}\|\_{2}^{2}}\;\;(d\_{k}\neq 0). |  | (9) |

(i) Differencing the partial criticality condition ∇x𝒜η=0\nabla\_{x}\mathcal{A}\_{\eta}=0 between consecutive edges yields the step-increment propagator:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | dk+1\displaystyle d\_{k+1} | =(I−η​H¯k)​dk,\displaystyle=(I-\eta\bar{H}\_{k})\,d\_{k}, |  | (10) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | wk+2−wk\displaystyle w\_{k+2}-w\_{k} | =(2​I−η​H¯k)​dk.\displaystyle=(2I-\eta\bar{H}\_{k})\,d\_{k}. |  | (11) |

Two-step return wk+2=wkw\_{k+2}=w\_{k} therefore occurs if and only if H¯k​dk=(2/η)​dk\bar{H}\_{k}d\_{k}=(2/\eta)\,d\_{k}.

(ii) A second-order expansion of 𝒜η\mathcal{A}\_{\eta} along a partial critical edge gives the one-step loss change:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | L​(wk+1)−L​(wk)\displaystyle L(w\_{k+1})-L(w\_{k}) | =−1η​‖dk‖22+12​dk⊤​H~k​dk=−‖dk‖222​η​(2−η​r~k).\displaystyle=-\frac{1}{\eta}\|d\_{k}\|\_{2}^{2}+\frac{1}{2}\,d\_{k}^{\top}\widetilde{H}\_{k}d\_{k}=-\frac{\|d\_{k}\|\_{2}^{2}}{2\eta}\Bigl(2-\eta\widetilde{r}\_{k}\Bigr). |  | (12) |

Summing over k=0,…,K−1k=0,\ldots,K{-}1 telescopes to

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑k=0K−1‖dk‖22​(2η−r~k)=2​(L​(w0)−L​(wK)).\displaystyle\sum\_{k=0}^{K-1}\|d\_{k}\|\_{2}^{2}\!\left(\frac{2}{\eta}-\widetilde{r}\_{k}\right)=2\bigl(L(w\_{0})-L(w\_{K})\bigr). |  | (13) |

*Proof sketch.* Part (i) differences ∇x𝒜η=0\nabla\_{x}\mathcal{A}\_{\eta}=0 between consecutive edges and applies the fundamental theorem of calculus. Part (ii) uses the exact Taylor formula with integral remainder and vanishing linear term. See [appendix˜A](#A1 "Appendix A Proofs from the Main Text ‣ The Origin of Edge of Stability").

At the Edge of Stability, gradient descent approximately reverses its step at each iteration, producing near-periodic oscillations. We now analyze the period-two orbits that organize this behavior. A period-two orbit near a critical point w¯\bar{w} has the form (w¯−a,w¯+a)(\bar{w}-a,\,\bar{w}+a), but as aa grows the center shifts away from w¯\bar{w}. The implicit function theorem lets us solve for the true center m​(a)m(a) and rewrite the problem in terms of the half-amplitude aa alone:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Φη​(a)=𝒫​(a)−1η​‖a‖2,\displaystyle\Phi\_{\eta}(a)=\mathcal{P}(a)-\frac{1}{\eta}\|a\|^{2}, |  | (14) |

where 𝒫\mathcal{P} depends only on LL and not on the learning rate.

###### Theorem 2.3 (Center reduction and the edge eigenproblem).

Let LL be C4C^{4} near a nondegenerate critical point w¯\bar{w}, and write
H≜∇2L​(w¯)H\triangleq\nabla^{2}L(\bar{w}).
There exist neighborhoods U∋0U\ni 0 and V∋w¯V\ni\bar{w} and a unique smooth map
m:U→Vm:U\to V such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | m​(0)=w¯,∇L​(m​(a)+a)+∇L​(m​(a)−a)=0(a∈U).\displaystyle m(0)=\bar{w},\qquad\nabla L(m(a)+a)+\nabla L(m(a)-a)=0\qquad(a\in U). |  | (15) |

The map mm is even: m​(−a)=m​(a)m(-a)=m(a).

Define

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒫​(a)≜12​(L​(m​(a)+a)+L​(m​(a)−a)),a∈U.\displaystyle\mathcal{P}(a)\;\triangleq\;\frac{1}{2}\Bigl(L(m(a)+a)+L(m(a)-a)\Bigr),\qquad a\in U. |  | (16) |

Then, for every η>0\eta>0,

|  |  |  |  |
| --- | --- | --- | --- |
|  | Φη​(a)≜Ψη​(m​(a),a)=𝒫​(a)−1η​‖a‖2\displaystyle\Phi\_{\eta}(a)\;\triangleq\;\Psi\_{\eta}(m(a),a)=\mathcal{P}(a)-\frac{1}{\eta}\|a\|^{2} |  | (17) |

is an even function of aa. The gradient of 𝒫\mathcal{P} satisfies

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇𝒫​(a)=12​(∇L​(m​(a)+a)−∇L​(m​(a)−a)),\displaystyle\nabla\mathcal{P}(a)=\frac{1}{2}\Bigl(\nabla L(m(a)+a)-\nabla L(m(a)-a)\Bigr), |  | (18) |

so the critical points of Φη\Phi\_{\eta} satisfy a nonlinear eigenvalue equation: a≠0a\neq 0 is a critical point of Φη\Phi\_{\eta} if and only if

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇𝒫​(a)=2η​a.\displaystyle\nabla\mathcal{P}(a)=\frac{2}{\eta}\,a. |  | (19) |

Full criticality of 𝒜η\mathcal{A}\_{\eta} at (m​(a)−a,m​(a)+a)(m(a)-a,\,m(a)+a) is equivalent to this same equation. Nearby critical points of 𝒜η\mathcal{A}\_{\eta} are therefore in bijection with critical points of Φη\Phi\_{\eta}: a=0a=0 gives the fixed point (w¯,w¯)(\bar{w},\bar{w}), and a≠0a\neq 0 gives a period-two orbit xη=m​(a)−ax\_{\eta}=m(a)-a, yη=m​(a)+ay\_{\eta}=m(a)+a.

The Hessian of 𝒫\mathcal{P} at the origin is ∇2𝒫​(0)=H\nabla^{2}\mathcal{P}(0)=H, and therefore

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇2Φη​(0)=H−2η​I.\displaystyle\nabla^{2}\Phi\_{\eta}(0)=H-\frac{2}{\eta}I. |  | (20) |

This becomes singular when 2/η2/\eta enters the spectrum of HH, which is therefore the spectral threshold for the birth of nontrivial period-two orbits.

*Proof sketch.* The implicit function theorem applied to the center-balance equation yields the center map m​(a)m(a); the marginal formula and nonlinear eigenvalue equation follow from differentiating the reduced functional. See [appendix˜A](#A1 "Appendix A Proofs from the Main Text ‣ The Origin of Edge of Stability").

*Remark on nondegeneracy.* Deep learning landscapes typically have degenerate minima forming Morse–Bott manifolds (due to overparameterization symmetries). The theorem applies to such settings by restricting to the normal space of the minimum manifold, as we demonstrate for two-layer linear networks in [Proposition˜2.6](#S2.Thmtheorem6 "Proposition 2.6 (Transverse edge normal form for two-layer linear networks). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability") and [appendix˜G](#A7 "Appendix G Transverse edge normal form for two-layer linear networks ‣ The Origin of Edge of Stability").

To use [Theorem˜2.3](#S2.Thmtheorem3 "Theorem 2.3 (Center reduction and the edge eigenproblem). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability") we need the Taylor expansion of 𝒫\mathcal{P}. Its quartic term 𝒬\mathcal{Q} determines whether period-two orbits appear.

###### Proposition 2.4 (Quartic expansion near a fixed point).

In the setting of [Theorem˜2.3](#S2.Thmtheorem3 "Theorem 2.3 (Center reduction and the edge eigenproblem). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability"), the center map has expansion

|  |  |  |  |
| --- | --- | --- | --- |
|  | m​(a)=w¯−12​H−1​∇3L​(w¯)​[a,a,⋅]+O​(‖a‖4),\displaystyle m(a)=\bar{w}-\frac{1}{2}\,H^{-1}\nabla^{3}L(\bar{w})[a,a,\cdot]+O(\|a\|^{4}), |  | (21) |

and 𝒫\mathcal{P} has expansion

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒫​(a)=L​(w¯)+12​⟨H​a,a⟩+14​𝒬​(a)+o​(‖a‖4),\displaystyle\mathcal{P}(a)=L(\bar{w})+\frac{1}{2}\langle Ha,a\rangle+\frac{1}{4}\,\mathcal{Q}(a)+o(\|a\|^{4}), |  | (22) |

where

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒬​(a)≜16​∇4L​(w¯)​[a,a,a,a]−12​⟨∇3L​(w¯)​[a,a,⋅],H−1​∇3L​(w¯)​[a,a,⋅]⟩.\displaystyle\mathcal{Q}(a)\;\triangleq\;\frac{1}{6}\,\nabla^{4}L(\bar{w})[a,a,a,a]-\frac{1}{2}\Bigl\langle\nabla^{3}L(\bar{w})[a,a,\cdot],\,H^{-1}\nabla^{3}L(\bar{w})[a,a,\cdot]\Bigr\rangle. |  | (23) |

Consequently,

|  |  |  |  |
| --- | --- | --- | --- |
|  | Φη​(a)=L​(w¯)+12​⟨(H−2η​I)​a,a⟩+14​𝒬​(a)+o​(‖a‖4).\displaystyle\Phi\_{\eta}(a)=L(\bar{w})+\frac{1}{2}\Bigl\langle\Bigl(H-\frac{2}{\eta}I\Bigr)a,a\Bigr\rangle+\frac{1}{4}\,\mathcal{Q}(a)+o(\|a\|^{4}). |  | (24) |

*Proof sketch.* Differentiating the center-balance equation twice at a=0a=0 determines the leading term of m​(a)m(a), and substituting into the Taylor expansion of 𝒫\mathcal{P} produces 𝒬\mathcal{Q}. See [appendix˜A](#A1 "Appendix A Proofs from the Main Text ‣ The Origin of Edge of Stability").

With this expansion in hand, the bifurcation problem reduces to finding nontrivial solutions of ∇𝒫​(a)=(2/η)​a\nabla\mathcal{P}(a)=(2/\eta)a. Near the critical learning rate ηc\eta\_{c}, any small solution aa must lie close to the kernel Ec=ker⁡(H−(2/ηc)​I)E\_{c}=\ker(H-(2/\eta\_{c})I). The quartic term 𝒬\mathcal{Q}, restricted to the unit sphere S​(Ec)S(E\_{c}), then determines which directions in EcE\_{c} support bifurcating branches and on which side of ηc\eta\_{c} they appear (kielhofer2012bifurcation; golubitsky1985singularities).

###### Corollary 2.5 (Generic branching at the edge).

Fix ηc>0\eta\_{c}>0 and let

|  |  |  |  |
| --- | --- | --- | --- |
|  | Ec≜ker⁡(H−2ηc​I).\displaystyle E\_{c}\triangleq\ker\!\Bigl(H-\frac{2}{\eta\_{c}}I\Bigr). |  | (25) |

Assume u∈S​(Ec)u\in S(E\_{c}) is a nondegenerate critical point of
𝒬|S​(Ec)\mathcal{Q}|\_{S(E\_{c})} and that 𝒬​(u)≠0\mathcal{Q}(u)\neq 0.
Then there exists a unique local branch of nontrivial period-two
orbits, unique up to swapping the two points, with amplitude

|  |  |  |  |
| --- | --- | --- | --- |
|  | a​(η)=α​(η)​u+o​(|η−ηc|),\displaystyle a(\eta)=\alpha(\eta)\,u+o\!\bigl(\sqrt{|\eta-\eta\_{c}|}\bigr), |  | (26) |

where

|  |  |  |  |
| --- | --- | --- | --- |
|  | α​(η)2=2η−2ηc𝒬​(u)+o​(|η−ηc|).\displaystyle\alpha(\eta)^{2}=\frac{\frac{2}{\eta}-\frac{2}{\eta\_{c}}}{\mathcal{Q}(u)}+o(|\eta-\eta\_{c}|). |  | (27) |

The branch exists on the side

|  |  |  |  |
| --- | --- | --- | --- |
|  | (2η−2ηc)​𝒬​(u)>0.\displaystyle\Bigl(\frac{2}{\eta}-\frac{2}{\eta\_{c}}\Bigr)\mathcal{Q}(u)>0. |  | (28) |

Moreover, every sufficiently small period-two orbit is
tangent to a critical direction of 𝒬|S​(Ec)\mathcal{Q}|\_{S(E\_{c})}.

In particular, when dimEc=1\dim E\_{c}=1, this reduces to the scalar
amplitude scaling.

*Proof sketch.* A Lyapunov–Schmidt reduction decomposes the amplitude into EcE\_{c} and Ec⟂E\_{c}^{\perp} components; the radial projection of the reduced gradient equation yields the amplitude scaling. See [appendix˜A](#A1 "Appendix A Proofs from the Main Text ‣ The Origin of Edge of Stability").

###### Proposition 2.6 (Transverse edge normal form for two-layer linear networks).

For Lh​(W1,W2)=12​‖W2​W1−M‖F2L\_{h}(W\_{1},W\_{2})=\frac{1}{2}\|W\_{2}W\_{1}-M\|\_{F}^{2} with W1∈ℝh×dW\_{1}\in\mathbb{R}^{h\times d}, W2∈ℝp×hW\_{2}\in\mathbb{R}^{p\times h}, and rank⁡(M)=r≤h\operatorname{rank}(M)=r\leq h, the minimum set is a Morse–Bott manifold. The Hessian has a nontrivial kernel corresponding to reparametrization symmetries, so we restrict to the orthogonal complement 𝒩=(ker​∇2Lh​(w¯))⟂\mathcal{N}=(\ker\nabla^{2}L\_{h}(\bar{w}))^{\perp}. The resulting transverse edge theory is *width-invariant*: overparameterization adds only flat directions and leaves the restricted loss unchanged. If σ1>σ2\sigma\_{1}>\sigma\_{2}, then

|  |  |  |  |
| --- | --- | --- | --- |
|  | Φη⟂​(t​uc)=Lh​(w¯)+(σ1−1η)​t2−t4+o​(t4),\displaystyle\Phi\_{\eta}^{\perp}(tu\_{c})\;=\;L\_{h}(\bar{w})+\bigl(\sigma\_{1}-\tfrac{1}{\eta}\bigr)t^{2}-t^{4}+o(t^{4}), |  | (29) |

so the first period-doubling occurs for η>ηc=1/σ1\eta>\eta\_{c}=1/\sigma\_{1} and the branch emerges continuously from zero at ηc\eta\_{c} for every h≥rh\geq r ([Figure 2](#S2.F2 "Figure 2 ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability"); proof in [appendix˜G](#A7 "Appendix G Transverse edge normal form for two-layer linear networks ‣ The Origin of Edge of Stability")).

![Refer to caption](/html/2604.20446/assets/x3.png)

![Refer to caption](/html/2604.20446/assets/x4.png)

Figure 2: Continuous onset of period-doubling in a two-layer linear network ([Proposition˜2.6](#S2.Thmtheorem6 "Proposition 2.6 (Transverse edge normal form for two-layer linear networks). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability")).
p=5p\!=\!5, h=3h\!=\!3, d=10d\!=\!10, rank-3 target, n=200n=200 samples.
a, Period-two amplitude 2​‖a​(η)‖2\|a(\eta)\| vs. η−ηc\eta-\eta\_{c} (log-log).
Observed amplitude (dots) tracks the η−ηc\sqrt{\eta-\eta\_{c}} scaling predicted by [Corollary˜2.5](#S2.Thmtheorem5 "Corollary 2.5 (Generic branching at the edge). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability") (dashed);
𝒬⟂​(uc)<0\mathcal{Q}^{\perp}(u\_{c})<0 shows that the branch appears for η>ηc\eta>\eta\_{c}.
b, Pitchfork diagram: projection ⟨wk−w¯,uc⟩\langle w\_{k}-\bar{w},u\_{c}\rangle vs. η\eta. Branches emerge continuously at ηc\eta\_{c}, confirming that period-doubling begins for η>ηc\eta>\eta\_{c}.

## 3 Why 𝟐/𝜼2/\eta Appears Everywhere

#### Two step-averaged Hessians.

[Theorem˜2.2](#S2.Thmtheorem2 "Theorem 2.2 (Propagator and One-Step Loss Change). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability") introduced two Hessian averages along each step segment: a uniform average H¯k\bar{H}\_{k} and a triangularly weighted average H~k\widetilde{H}\_{k}. These arise inevitably from the fundamental theorem of calculus applied to gradients and losses, respectively:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∇L​(wk+1)−∇L​(wk)\displaystyle\nabla L(w\_{k+1})-\nabla L(w\_{k}) | =H¯k​dk,\displaystyle=\bar{H}\_{k}\,d\_{k}, |  | (30) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | L​(wk+1)−L​(wk)\displaystyle L(w\_{k+1})-L(w\_{k}) | =∇L​(wk)⊤​dk+12​dk⊤​H~k​dk.\displaystyle=\nabla L(w\_{k})^{\top}d\_{k}+\tfrac{1}{2}\,d\_{k}^{\top}\widetilde{H}\_{k}\,d\_{k}. |  | (31) |

#### The roles of 𝟐/𝜼2/\eta.

Restating [Theorem˜2.2](#S2.Thmtheorem2 "Theorem 2.2 (Propagator and One-Step Loss Change). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability") in terms of these averages reveals 2/η2/\eta as a single threshold governing both propagation and descent.

###### Corollary 3.1 (Roles of 𝟐/𝜼2/\eta).

Fix kk with dk≠0d\_{k}\neq 0. Step reversal dk+1=−dkd\_{k+1}=-d\_{k} and two-step return wk+2=wkw\_{k+2}=w\_{k} are both equivalent to

|  |  |  |  |
| --- | --- | --- | --- |
|  | H¯k​dk=2η​dk.\displaystyle\bar{H}\_{k}d\_{k}=\frac{2}{\eta}\,d\_{k}. |  | (32) |

Recall the effective curvature r~k\widetilde{r}\_{k} from [Theorem˜2.2](#S2.Thmtheorem2 "Theorem 2.2 (Propagator and One-Step Loss Change). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability"). The loss decreases if and only if r~k≤2/η\widetilde{r}\_{k}\leq 2/\eta:

|  |  |  |  |
| --- | --- | --- | --- |
|  | L​(wk+1)−L​(wk)=−‖dk‖222​η​(2−η​r~k).\displaystyle L(w\_{k+1})-L(w\_{k})=-\frac{\|d\_{k}\|\_{2}^{2}}{2\eta}\Bigl(2-\eta\widetilde{r}\_{k}\Bigr). |  | (33) |

#### Curvature along the step.

Two different Hessian averages appear, but they are two views of the same underlying object. To make this precise, we restrict attention to the step direction uk≜dk/‖dk‖2u\_{k}\triangleq d\_{k}/\|d\_{k}\|\_{2} and define

|  |  |  |  |
| --- | --- | --- | --- |
|  | qk​(τ)≜uk⊤​∇2L​(wk+τ​dk)​uk,τ∈[0,1].\displaystyle q\_{k}(\tau)\triangleq u\_{k}^{\top}\nabla^{2}L(w\_{k}+\tau d\_{k})\,u\_{k},\qquad\tau\in[0,1]. |  | (34) |

With r¯k≜dk⊤​H¯k​dk/‖dk‖22\bar{r}\_{k}\triangleq d\_{k}^{\top}\bar{H}\_{k}d\_{k}/\|d\_{k}\|\_{2}^{2}, both r¯k\bar{r}\_{k} and r~k\widetilde{r}\_{k} are weighted integrals of qkq\_{k}. Since qkq\_{k} is continuous, each is attained as an exact pointwise value at some interior point of the step segment ([Theorem˜4.3](#S4.Thmtheorem3 "Theorem 4.3 (Localization to the true Hessian). ‣ Exact sharpness forcing on each edge. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability")), and the loss change is well approximated by a proxy that uses only trajectory data:

|  |  |  |  |
| --- | --- | --- | --- |
|  | L​(wk+1)−L​(wk)≈−12​η​dk⊤​(wk+2−wk).\displaystyle L(w\_{k+1})-L(w\_{k})\;\approx\;-\frac{1}{2\eta}\,d\_{k}^{\top}(w\_{k+2}-w\_{k}). |  | (35) |

## 4 The Origin of Edge of Stability

#### Why the dynamics is forced toward 𝟐/𝜼2/\eta.

The previous section established 2/η2/\eta as the threshold governing both step propagation and loss descent. We now show that this threshold is a global attractor: the trajectory is forced to visit it. The argument is a conservation law. The loss-change formula ([Corollary˜3.1](#S3.Thmtheorem1 "Corollary 3.1 (Roles of 𝟐/𝜼). ‣ The roles of 𝟐/𝜼. ‣ 3 Why 𝟐/𝜼 Appears Everywhere ‣ The Origin of Edge of Stability")) writes each per-step loss change as the product of ‖dk‖22\|d\_{k}\|\_{2}^{2} and the deviation 2/η−r~k2/\eta-\widetilde{r}\_{k}. Summing over the trajectory, the loss side telescopes to a bounded quantity while the curvature deviations accumulate. Because the total loss drop is finite, the curvature r~k\widetilde{r}\_{k} cannot stay far from 2/η2/\eta.

###### Theorem 4.1 (Curvature concentration at 𝟐/𝜼2/\eta).

Let L:ℝd→ℝL:\mathbb{R}^{d}\to\mathbb{R} be C2C^{2} and bounded below by
Linf≜infwL​(w)>−∞L\_{\inf}\triangleq\inf\_{w}L(w)>-\infty.
Run gradient descent wk+1=wk−η​∇L​(wk)w\_{k+1}=w\_{k}-\eta\nabla L(w\_{k}) with step size η>0\eta>0.
Use the notation dkd\_{k}, H~k\widetilde{H}\_{k}, r~k\widetilde{r}\_{k} from
[Theorems˜2.2](#S2.Thmtheorem2 "Theorem 2.2 (Propagator and One-Step Loss Change). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability") and [3.1](#S3.Thmtheorem1 "Corollary 3.1 (Roles of 𝟐/𝜼). ‣ The roles of 𝟐/𝜼. ‣ 3 Why 𝟐/𝜼 Appears Everywhere ‣ The Origin of Edge of Stability"), and set
EK≜∑k=0K−1‖dk‖22E\_{K}\triangleq\sum\_{k=0}^{K-1}\|d\_{k}\|\_{2}^{2}.

For every K≥1K\geq 1, the ‖dk‖2\|d\_{k}\|^{2}-weighted deviations from 2/η2/\eta telescope to the total loss change:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑k=0K−1‖dk‖22​(2η−r~k)=2​(L​(w0)−L​(wK)).\displaystyle\sum\_{k=0}^{K-1}\|d\_{k}\|\_{2}^{2}\!\left(\frac{2}{\eta}-\widetilde{r}\_{k}\right)=2\bigl(L(w\_{0})-L(w\_{K})\bigr). |  | (36) |

Dividing by EK>0E\_{K}>0 rewrites this as a weighted average:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑k=0K−1‖dk‖22​r~kEK=2η−2​(L​(w0)−L​(wK))EK.\displaystyle\frac{\sum\_{k=0}^{K-1}\|d\_{k}\|\_{2}^{2}\,\widetilde{r}\_{k}}{E\_{K}}=\frac{2}{\eta}-\frac{2\bigl(L(w\_{0})-L(w\_{K})\bigr)}{E\_{K}}. |  | (37) |

Since the left-hand side is bounded above by maxk<K⁡r~k\max\_{k<K}\widetilde{r}\_{k} and below by mink<K⁡r~k\min\_{k<K}\widetilde{r}\_{k}, we obtain a lower bound on the maximum curvature encountered along the trajectory:

|  |  |  |  |
| --- | --- | --- | --- |
|  | max0≤k≤K−1⁡r~k≥2η−2​(L​(w0)−Linf)EK.\displaystyle\max\_{0\leq k\leq K-1}\widetilde{r}\_{k}\;\geq\;\frac{2}{\eta}-\frac{2\bigl(L(w\_{0})-L\_{\inf}\bigr)}{E\_{K}}. |  | (38) |

As EKE\_{K} grows, the right-hand side approaches 2/η2/\eta, so the trajectory is forced to visit steps with effective curvature arbitrarily close to the threshold. If LL is additionally bounded above along the trajectory and EK→∞E\_{K}\to\infty, then the weighted average converges:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑k=0K−1‖dk‖22​r~kEK⟶2η.\displaystyle\frac{\sum\_{k=0}^{K-1}\|d\_{k}\|\_{2}^{2}\,\widetilde{r}\_{k}}{E\_{K}}\longrightarrow\frac{2}{\eta}. |  | (39) |

*Proof sketch.* Telescope the one-step loss-change formula over KK steps. See [appendix˜A](#A1 "Appendix A Proofs from the Main Text ‣ The Origin of Edge of Stability").

So far we know that r~k\widetilde{r}\_{k} concentrates near 2/η2/\eta in a weighted average sense. The next theorem strengthens this to step-level concentration by decomposing deviations into positive and negative parts.

###### Theorem 4.2 (Concentration near 𝟐/𝜼2/\eta).

In the setting of [Theorem˜4.1](#S4.Thmtheorem1 "Theorem 4.1 (Curvature concentration at 𝟐/𝜼). ‣ Why the dynamics is forced toward 𝟐/𝜼. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability"), define

|  |  |  |  |
| --- | --- | --- | --- |
|  | BK−≜∑k=0K−1‖dk‖22​(2η−r~k)+,BK+≜∑k=0K−1‖dk‖22​(r~k−2η)+.\displaystyle B\_{K}^{-}\;\triangleq\;\sum\_{k=0}^{K-1}\|d\_{k}\|\_{2}^{2}\Bigl(\tfrac{2}{\eta}-\widetilde{r}\_{k}\Bigr)\_{+},\qquad B\_{K}^{+}\;\triangleq\;\sum\_{k=0}^{K-1}\|d\_{k}\|\_{2}^{2}\Bigl(\widetilde{r}\_{k}-\tfrac{2}{\eta}\Bigr)\_{+}. |  | (40) |

Then for every K≥1K\geq 1,

|  |  |  |  |
| --- | --- | --- | --- |
|  | BK−−BK+=2​(L​(w0)−L​(wK)).\displaystyle B\_{K}^{-}-B\_{K}^{+}=2\bigl(L(w\_{0})-L(w\_{K})\bigr). |  | (41) |

For any δ>0\delta>0,

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∑r~k≤2/η−δ‖dk‖22\displaystyle\sum\_{\widetilde{r}\_{k}\leq 2/\eta-\delta}\|d\_{k}\|\_{2}^{2} | ≤2​(L​(w0)−Linf)+BK+δ,\displaystyle\;\leq\;\frac{2(L(w\_{0})-L\_{\inf})+B\_{K}^{+}}{\delta}, |  | (42) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∑r~k≥2/η+δ‖dk‖22\displaystyle\sum\_{\widetilde{r}\_{k}\geq 2/\eta+\delta}\|d\_{k}\|\_{2}^{2} | ≤BK+δ.\displaystyle\;\leq\;\frac{B\_{K}^{+}}{\delta}. |  | (43) |

Hence if B∞+≜limK→∞BK+<∞B\_{\infty}^{+}\triangleq\lim\_{K\to\infty}B\_{K}^{+}<\infty, then for every δ>0\delta>0,

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑k=0∞𝟏{|r~k−2/η|≥δ}​‖dk‖22<∞,\displaystyle\sum\_{k=0}^{\infty}\mathbf{1}\_{\{|\widetilde{r}\_{k}-2/\eta|\geq\delta\}}\|d\_{k}\|\_{2}^{2}<\infty, |  | (44) |

and if moreover EK→∞E\_{K}\to\infty, the weighted curvature concentrates at 2/η2/\eta:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑k=0K−1𝟏{|r~k−2/η|<δ}​‖dk‖22EK⟶1.\displaystyle\frac{\sum\_{k=0}^{K-1}\mathbf{1}\_{\{|\widetilde{r}\_{k}-2/\eta|<\delta\}}\|d\_{k}\|\_{2}^{2}}{E\_{K}}\longrightarrow 1. |  | (45) |

No monotone-descent assumption is needed; the controlling quantity is finiteness of B∞+B\_{\infty}^{+}.

*Proof sketch.* Decompose 2/η−r~k2/\eta-\widetilde{r}\_{k} into positive and negative parts; Markov-type estimates yield the window bounds. See [appendix˜A](#A1 "Appendix A Proofs from the Main Text ‣ The Origin of Edge of Stability"). Panel (b) of [Figure 3](#S4.F3 "Figure 3 ‣ Why the dynamics is forced toward 𝟐/𝜼. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability") validates the loss-change formula step by step: the scatter of actual Δ​Lk\Delta L\_{k} against the prediction from r¯k\bar{r}\_{k} lies tightly along y=xy=x, confirming both [Theorem˜2.2](#S2.Thmtheorem2 "Theorem 2.2 (Propagator and One-Step Loss Change). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability")(ii) and that r¯k\bar{r}\_{k} and r~k\widetilde{r}\_{k} nearly coincide ([Theorem˜4.3](#S4.Thmtheorem3 "Theorem 4.3 (Localization to the true Hessian). ‣ Exact sharpness forcing on each edge. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability")).

![Refer to caption](/html/2604.20446/assets/x5.png)

![Refer to caption](/html/2604.20446/assets/x6.png)

Figure 3: Validation of [Theorems˜4.1](#S4.Thmtheorem1 "Theorem 4.1 (Curvature concentration at 𝟐/𝜼). ‣ Why the dynamics is forced toward 𝟐/𝜼. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability") and [2.2](#S2.Thmtheorem2 "Theorem 2.2 (Propagator and One-Step Loss Change). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability").
Four learning rates, 4,0004{,}000 steps, shared initialization.
a, Weighted average curvature converges to 2/η2/\eta (dashed).
b, Actual Δ​Lk\Delta L\_{k} vs. the proxy −12​η​dk⊤​(wk+2−wk)-\frac{1}{2\eta}d\_{k}^{\top}(w\_{k+2}-w\_{k}).
Its tightness confirms that r¯k\bar{r}\_{k} and r~k\widetilde{r}\_{k} nearly coincide on this run.

#### Exact sharpness forcing on each edge.

So far the concentration theorems control r~k\widetilde{r}\_{k} and r¯k\bar{r}\_{k}, which are weighted averages of the Hessian along each step segment. A natural question is whether this forcing extends to the true Hessian eigenvalue at an actual point. The answer is affirmative and requires nothing beyond the mean value theorem: along each edge, r~k\widetilde{r}\_{k} and r¯k\bar{r}\_{k} are realized as exact pointwise values of the directional curvature at specific interior points, so the forcing transfers to the true Hessian eigenvalue at those points with no residual gap.

###### Theorem 4.3 (Localization to the true Hessian).

Let uk=dk/‖dk‖2u\_{k}=d\_{k}/\|d\_{k}\|\_{2} when dk≠0d\_{k}\neq 0. For each such kk, there exist ξk,ζk∈(0,1)\xi\_{k},\zeta\_{k}\in(0,1) such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | r~k=uk⊤​∇2L​(wk+ξk​dk)​uk,r¯k=uk⊤​∇2L​(wk+ζk​dk)​uk.\displaystyle\widetilde{r}\_{k}=u\_{k}^{\top}\nabla^{2}L(w\_{k}+\xi\_{k}d\_{k})\,u\_{k},\qquad\bar{r}\_{k}=u\_{k}^{\top}\nabla^{2}L(w\_{k}+\zeta\_{k}d\_{k})\,u\_{k}. |  | (46) |

Consequently, λmax​(∇2L​(wk+ξk​dk))≥r~k\lambda\_{\max}(\nabla^{2}L(w\_{k}+\xi\_{k}d\_{k}))\geq\widetilde{r}\_{k} and λmax​(∇2L​(wk+ζk​dk))≥r¯k\lambda\_{\max}(\nabla^{2}L(w\_{k}+\zeta\_{k}d\_{k}))\geq\bar{r}\_{k}. Combined with [Theorem˜4.1](#S4.Thmtheorem1 "Theorem 4.1 (Curvature concentration at 𝟐/𝜼). ‣ Why the dynamics is forced toward 𝟐/𝜼. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability"):

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑k=0K−1‖dk‖22​(2η−λmax​(∇2L​(wk+ξk​dk)))≤2​(L​(w0)−L​(wK)),\displaystyle\sum\_{k=0}^{K-1}\|d\_{k}\|\_{2}^{2}\!\left(\frac{2}{\eta}-\lambda\_{\max}\!\bigl(\nabla^{2}L(w\_{k}+\xi\_{k}d\_{k})\bigr)\right)\leq 2\bigl(L(w\_{0})-L(w\_{K})\bigr), |  | (47) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | max0≤k<K⁡λmax​(∇2L​(wk+ξk​dk))≥2η−2​(L​(w0)−Linf)EK.\displaystyle\max\_{0\leq k<K}\lambda\_{\max}\!\bigl(\nabla^{2}L(w\_{k}+\xi\_{k}d\_{k})\bigr)\geq\frac{2}{\eta}-\frac{2(L(w\_{0})-L\_{\inf})}{E\_{K}}. |  | (48) |

From the propagator ([Corollary˜4.4](#S4.Thmtheorem4 "Corollary 4.4 (Near-Periodicity Implies Near-Critical Sharpness). ‣ Discrete stability and near-periodicity. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | λmax​(∇2L​(wk+ζk​dk))≥2η−‖wk+2−wk‖2η​‖dk‖2.\displaystyle\lambda\_{\max}\!\bigl(\nabla^{2}L(w\_{k}+\zeta\_{k}d\_{k})\bigr)\geq\frac{2}{\eta}-\frac{\|w\_{k+2}-w\_{k}\|\_{2}}{\eta\,\|d\_{k}\|\_{2}}. |  | (49) |

*Proof sketch.* Define the scalar restriction gk​(t)=L​(wk+t​dk)g\_{k}(t)=L(w\_{k}+t\,d\_{k}). Taylor’s theorem with Lagrange remainder gives gk​(1)=gk​(0)+gk′​(0)+12​gk′′​(ξk)g\_{k}(1)=g\_{k}(0)+g\_{k}^{\prime}(0)+\frac{1}{2}g\_{k}^{\prime\prime}(\xi\_{k}); comparing with the integral form ([Theorem˜2.2](#S2.Thmtheorem2 "Theorem 2.2 (Propagator and One-Step Loss Change). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability")(ii)) identifies r~k=qk​(ξk)\widetilde{r}\_{k}=q\_{k}(\xi\_{k}). The mean value theorem applied to gk′g\_{k}^{\prime} gives r¯k=qk​(ζk)\bar{r}\_{k}=q\_{k}(\zeta\_{k}). The forcing and propagator bounds follow from qk≤λmaxq\_{k}\leq\lambda\_{\max} combined with [Theorems˜4.1](#S4.Thmtheorem1 "Theorem 4.1 (Curvature concentration at 𝟐/𝜼). ‣ Why the dynamics is forced toward 𝟐/𝜼. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability") and [4.4](#S4.Thmtheorem4 "Corollary 4.4 (Near-Periodicity Implies Near-Critical Sharpness). ‣ Discrete stability and near-periodicity. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability"). See [appendix˜B](#A2 "Appendix B From Hessian Averages to True Sharpness ‣ The Origin of Edge of Stability").

#### Discrete stability and near-periodicity.

The recurrence dk+1=(I−η​H¯k)​dkd\_{k+1}=(I-\eta\bar{H}\_{k})d\_{k} is stable when the eigenvalues of H¯k\bar{H}\_{k} lie in [0,2/η][0,2/\eta]: outside this interval the multiplier |1−η​λ||1-\eta\lambda| exceeds 11 and the step norm grows. At the Edge of Stability the sharpest eigenvalue sits near 2/η2/\eta, giving a multiplier near −1-1; the step direction reverses at each iteration and the loss oscillates (formal bound in [Theorem˜A.3](#A1.Thmtheorem3 "Theorem A.3 (Discrete Stability Bound). ‣ Proof of Theorem˜4.2 (Concentration within a window of 𝟐/𝜼). ‣ Appendix A Proofs from the Main Text ‣ The Origin of Edge of Stability")).

In practice, gradient descent does not settle into a period-two orbit but exhibits approximate periodicity (cohen2021gradient): the iterate nearly returns after two steps, so ‖wk+2−wk‖≪‖wk+1−wk‖\|w\_{k+2}-w\_{k}\|\ll\|w\_{k+1}-w\_{k}\| ([Figure 4](#A1.F4 "Figure 4 ‣ Proof of Corollary˜4.4 (Near-Periodicity Implies Near-Critical Sharpness). ‣ Appendix A Proofs from the Main Text ‣ The Origin of Edge of Stability") in [appendix˜A](#A1 "Appendix A Proofs from the Main Text ‣ The Origin of Edge of Stability")). This condition alone, without any Hessian computation, ensures that the directional curvature of H¯k\bar{H}\_{k} along the step is close to 2/η2/\eta.

###### Corollary 4.4 (Near-Periodicity Implies Near-Critical Sharpness).

The directional curvature of H¯k\bar{H}\_{k} along the step direction satisfies

|  |  |  |  |
| --- | --- | --- | --- |
|  | |dk⊤​H¯k​dk‖dk‖22−2η|≤‖wk+2−wk‖2η​‖dk‖2.\displaystyle\left|\frac{d\_{k}^{\top}\bar{H}\_{k}\,d\_{k}}{\|d\_{k}\|\_{2}^{2}}-\frac{2}{\eta}\right|\leq\frac{\|w\_{k+2}-w\_{k}\|\_{2}}{\eta\,\|d\_{k}\|\_{2}}. |  | (50) |

*Proof sketch.* Express the two-step displacement in terms of gradients; Cauchy–Schwarz completes the bound. See [appendix˜A](#A1 "Appendix A Proofs from the Main Text ‣ The Origin of Edge of Stability").

## 5 Stability Mechanisms at the Edge

[Section˜4](#S4 "4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability") explains where 2/η2/\eta comes from. Two questions remain: why does the system not diverge at the edge, and why does it stay there? The answer splits into two regimes (proofs in [appendix˜C](#A3 "Appendix C Growth above the threshold and oscillatory cancellation ‣ The Origin of Edge of Stability")).

#### Growth above the threshold.

Write r¯k≜dk⊤​H¯k​dk/‖dk‖22\bar{r}\_{k}\triangleq d\_{k}^{\top}\bar{H}\_{k}d\_{k}/\|d\_{k}\|\_{2}^{2}.

###### Proposition 5.1 (Growth above the threshold).

⟨dk+1,dk⟩=(1−η​r¯k)​‖dk‖22\langle d\_{k+1},d\_{k}\rangle=(1-\eta\bar{r}\_{k})\|d\_{k}\|\_{2}^{2}. Hence if r¯k≥2/η+δ\bar{r}\_{k}\geq 2/\eta+\delta, then ‖dk+1‖2≥(1+η​δ)​‖dk‖2\|d\_{k+1}\|\_{2}\geq(1+\eta\delta)\|d\_{k}\|\_{2}, and a sustained above-threshold run of length t−st-s yields geometric growth: ‖dt‖2≥(1+η​δ)t−s​‖ds‖2\|d\_{t}\|\_{2}\geq(1+\eta\delta)^{t-s}\|d\_{s}\|\_{2}.

On any bounded trajectory, step-norm growth is impossible, so the dynamics cannot remain above the threshold.

#### Oscillatory cancellation.

The recoil explains why the system cannot stay above 2/η2/\eta. Inside the stability window, the question is whether the near-critical multiplier mk≈−1m\_{k}\approx-1 causes secular drift.

###### Theorem 5.2 (Oscillatory cancellation).

Consider xk+1=mk​xk−η​ukx\_{k+1}=m\_{k}x\_{k}-\eta u\_{k}, x0=0x\_{0}=0, with mk∈[−1,0]m\_{k}\in[-1,0] for all kk. Then for every T≥1T\geq 1,

|  |  |  |  |
| --- | --- | --- | --- |
|  | |xT|≤η​(|uT−1|+∑k=0T−2|uk+1−uk|).\displaystyle|x\_{T}|\leq\eta\left(|u\_{T-1}|+\sum\_{k=0}^{T-2}|u\_{k+1}-u\_{k}|\right). |  | (51) |

The bound holds for arbitrary time-varying multipliers across the entire oscillatory stability window. When the curvature rises above 2/η2/\eta, [Proposition˜5.1](#S5.Thmtheorem1 "Proposition 5.1 (Growth above the threshold). ‣ Growth above the threshold. ‣ 5 Stability Mechanisms at the Edge ‣ The Origin of Edge of Stability") forces the dynamics back through expanding sign reversals; inside the window, [Theorem˜5.2](#S5.Thmtheorem2 "Theorem 5.2 (Oscillatory cancellation). ‣ Oscillatory cancellation. ‣ 5 Stability Mechanisms at the Edge ‣ The Origin of Edge of Stability") prevents secular growth. Together these explain the empirical observation (cohen2021gradient; damian2023selfstabilization) that sharpness hovers near 2/η2/\eta: excursions above the threshold drive parameters into regions of lower curvature, while near-threshold oscillations cancel rather than drift.

## 6 Conclusion

The edge coupling 𝒜η\mathcal{A}\_{\eta} provides a unified account of the Edge of Stability. Its criticality conditions yield the step recurrence and the loss-change formula; summing the latter forces curvature to 2/η2/\eta globally ([Theorem˜4.1](#S4.Thmtheorem1 "Theorem 4.1 (Curvature concentration at 𝟐/𝜼). ‣ Why the dynamics is forced toward 𝟐/𝜼. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability")), resolving the locality limitation of prior work (damian2023selfstabilization). The mean value theorem then localizes each average to the true Hessian eigenvalue at an interior point of each step segment, yielding exact forcing with no residual gap. The center reduction in [Theorem˜2.3](#S2.Thmtheorem3 "Theorem 2.3 (Center reduction and the edge eigenproblem). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability") reduces period-two bifurcation to a nonlinear eigenproblem; for two-layer linear networks this reduction is width-invariant and the period-doubling branch appears continuously on the large-learning-rate side of the threshold ([Proposition˜2.6](#S2.Thmtheorem6 "Proposition 2.6 (Transverse edge normal form for two-layer linear networks). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability")).

#### Limitations and future directions.

The global theorems require C2C^{2} smoothness and the bifurcation analysis requires C4C^{4}, which excludes losses with ReLU activations unless smoothed. The forcing theorems guarantee that curvature visits 2/η2/\eta but do not predict how long the system spends there or how the loss decreases in the EoS phase. Connecting the two-trajectory stability bound ([appendix˜E](#A5 "Appendix E Discrete-Time Kelvin–Voigt Dynamics ‣ The Origin of Edge of Stability")) to curvature-aware generalization bounds in the spirit of hardt2016train is a natural next step.

## Appendix A Proofs from the Main Text

This appendix collects the proofs of all results stated in [Sections˜2](#S2 "2 The Edge Coupling ‣ The Origin of Edge of Stability"), [3](#S3 "3 Why 𝟐/𝜼 Appears Everywhere ‣ The Origin of Edge of Stability"), [4](#S4 "4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability") and [5](#S5 "5 Stability Mechanisms at the Edge ‣ The Origin of Edge of Stability"). We also develop two results referenced by the main text: a spectral analysis of the edge coupling at a fixed point ([Proposition˜A.1](#A1.Thmtheorem1 "Proposition A.1 (Spectral threshold of the edge coupling). ‣ Proof of Theorem˜2.2 (Propagator and One-Step Loss Change). ‣ Appendix A Proofs from the Main Text ‣ The Origin of Edge of Stability")), which shows that the Hessian of 𝒜η\mathcal{A}\_{\eta} at a diagonal critical point (w¯,w¯)(\bar{w},\bar{w}) becomes indefinite when an eigenvalue of ∇2L​(w¯)\nabla^{2}L(\bar{w}) exceeds 2/η2/\eta; and a discrete stability bound ([Theorem˜A.3](#A1.Thmtheorem3 "Theorem A.3 (Discrete Stability Bound). ‣ Proof of Theorem˜4.2 (Concentration within a window of 𝟐/𝜼). ‣ Appendix A Proofs from the Main Text ‣ The Origin of Edge of Stability")), which controls how far the step norm can grow when the curvature temporarily leaves the interval [0,2/η][0,2/\eta].

#### Proof of [Theorem˜2.1](#S2.Thmtheorem1 "Theorem 2.1 (Variational characterization of gradient descent). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability").

###### Proof.

*(i)* Since ∇x𝒜η​(x,y)=∇L​(x)−1η​(x−y)\nabla\_{x}\mathcal{A}\_{\eta}(x,y)=\nabla L(x)-\frac{1}{\eta}(x-y), the condition
∇x𝒜η​(x,y)=0\nabla\_{x}\mathcal{A}\_{\eta}(x,y)=0 is equivalent to y=x−η​∇L​(x)y=x-\eta\nabla L(x).

*(ii)* We have ∇y𝒜η​(x,y)=∇L​(y)+1η​(x−y)\nabla\_{y}\mathcal{A}\_{\eta}(x,y)=\nabla L(y)+\frac{1}{\eta}(x-y).
Evaluating at (x,y)=(wk,wk+1)(x,y)=(w\_{k},w\_{k+1}) and using wk+1−wk=−η​∇L​(wk)w\_{k+1}-w\_{k}=-\eta\nabla L(w\_{k}) gives

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | −η​∇y𝒜η​(wk,wk+1)\displaystyle-\eta\nabla\_{y}\mathcal{A}\_{\eta}(w\_{k},w\_{k+1}) | =−η​∇L​(wk+1)−(wk−wk+1)\displaystyle=-\eta\nabla L(w\_{k+1})-(w\_{k}-w\_{k+1}) |  | (52) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =(wk+2−wk+1)+(wk+1−wk)\displaystyle=(w\_{k+2}-w\_{k+1})+(w\_{k+1}-w\_{k}) |  | (53) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =wk+2−wk.\displaystyle=w\_{k+2}-w\_{k}. |  | (54) |

The critical-point characterization follows by imposing both
∇x𝒜η=0\nabla\_{x}\mathcal{A}\_{\eta}=0 and ∇y𝒜η=0\nabla\_{y}\mathcal{A}\_{\eta}=0.
∎

#### Proof of [Theorem˜2.2](#S2.Thmtheorem2 "Theorem 2.2 (Propagator and One-Step Loss Change). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability") (Propagator and One-Step Loss Change).

The strategy for part (i) is to difference the criticality condition ∇x𝒜η=0\nabla\_{x}\mathcal{A}\_{\eta}=0 between two consecutive edges, which converts a gradient identity into a step-increment recurrence. For part (ii), we slide one endpoint of the edge coupling from wkw\_{k} to wk+1w\_{k+1} along the step segment; the vanishing of the first derivative at τ=0\tau=0 (which is the criticality condition itself) turns the Taylor expansion into a pure second-order expression.

###### Proof.

*(i)* Both consecutive pairs (wk,wk+1)(w\_{k},w\_{k+1}) and (wk+1,wk+2)(w\_{k+1},w\_{k+2}) lie on Γη\Gamma\_{\eta}, so differencing the partial criticality condition between them and substituting ∇L​(wk+1)−∇L​(wk)=H¯k​dk\nabla L(w\_{k+1})-\nabla L(w\_{k})=\bar{H}\_{k}d\_{k} gives

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 0\displaystyle 0 | =∇x𝒜η​(wk+1,wk+2)−∇x𝒜η​(wk,wk+1)\displaystyle=\nabla\_{x}\mathcal{A}\_{\eta}(w\_{k+1},w\_{k+2})-\nabla\_{x}\mathcal{A}\_{\eta}(w\_{k},w\_{k+1}) |  | (55) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =H¯k​dk−1η​(dk−dk+1),\displaystyle=\bar{H}\_{k}d\_{k}-\frac{1}{\eta}(d\_{k}-d\_{k+1}), |  | (56) |

which rearranges to dk+1=(I−η​H¯k)​dkd\_{k+1}=(I-\eta\bar{H}\_{k})d\_{k}. The two-step displacement follows by adding dkd\_{k} to both sides.

*(ii)* Fix kk and consider sliding the first argument of 𝒜η\mathcal{A}\_{\eta} along the step: define fk​(τ)≜𝒜η​(wk+τ​dk,wk+1)f\_{k}(\tau)\triangleq\mathcal{A}\_{\eta}(w\_{k}+\tau d\_{k},w\_{k+1}) for τ∈[0,1]\tau\in[0,1].
At τ=0\tau=0 we are at the actual edge (wk,wk+1)(w\_{k},w\_{k+1}), and at τ=1\tau=1 both arguments coincide at wk+1w\_{k+1}. The partial criticality condition ensures that the linear term vanishes: fk′​(0)=0f\_{k}^{\prime}(0)=0. The boundary values and second derivative are

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | fk​(0)\displaystyle f\_{k}(0) | =L​(wk)+L​(wk+1)−12​η​‖dk‖22,\displaystyle=L(w\_{k})+L(w\_{k+1})-\frac{1}{2\eta}\|d\_{k}\|\_{2}^{2}, |  | (57) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | fk​(1)\displaystyle f\_{k}(1) | =𝒜η​(wk+1,wk+1)=2​L​(wk+1),\displaystyle=\mathcal{A}\_{\eta}(w\_{k+1},w\_{k+1})=2L(w\_{k+1}), |  | (58) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | fk′′​(τ)\displaystyle f\_{k}^{\prime\prime}(\tau) | =dk⊤​∇2L​(wk+τ​dk)​dk−1η​‖dk‖22.\displaystyle=d\_{k}^{\top}\nabla^{2}L(w\_{k}+\tau d\_{k})d\_{k}-\frac{1}{\eta}\|d\_{k}\|\_{2}^{2}. |  | (59) |

The Taylor formula with integral remainder, fk​(1)−fk​(0)=∫01(1−τ)​fk′′​(τ)​𝑑τf\_{k}(1)-f\_{k}(0)=\int\_{0}^{1}(1-\tau)f\_{k}^{\prime\prime}(\tau)\,d\tau, then gives after substitution

|  |  |  |  |
| --- | --- | --- | --- |
|  | L​(wk+1)−L​(wk)=−1η​‖dk‖22+12​dk⊤​H~k​dk.\displaystyle L(w\_{k+1})-L(w\_{k})=-\frac{1}{\eta}\|d\_{k}\|\_{2}^{2}+\frac{1}{2}\,d\_{k}^{\top}\widetilde{H}\_{k}d\_{k}. |  | (60) |

The telescoping identity follows by summing over kk.
∎

The forcing and concentration theorems in the main text describe the trajectory globally, but the edge coupling also contains local information at a fixed point. The Hessian of 𝒜η\mathcal{A}\_{\eta} at a diagonal critical point (w¯,w¯)(\bar{w},\bar{w}) acts on the product space ℝd×ℝd\mathbb{R}^{d}\times\mathbb{R}^{d}, and its eigendirections split into two natural families: diagonal directions (u,u)(u,u) that move both iterates together, and anti-diagonal directions (u,−u)(u,-u) that push them apart. The following proposition shows that the diagonal directions are always stable (they see only the curvature HH), while the anti-diagonal directions become unstable when any eigenvalue of HH crosses 2/η2/\eta. This provides a spectral interpretation of the Edge of Stability directly from the second-order structure of the edge coupling.

###### Proposition A.1 (Spectral threshold of the edge coupling).

Let w¯\bar{w} be a critical point of LL with H=∇2L​(w¯)H=\nabla^{2}L(\bar{w}).
Then (w¯,w¯)(\bar{w},\bar{w}) is a critical point of 𝒜η\mathcal{A}\_{\eta}, with Hessian

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇2𝒜η​(w¯,w¯)=(H−1η​I1η​I1η​IH−1η​I).\displaystyle\nabla^{2}\mathcal{A}\_{\eta}(\bar{w},\bar{w})=\begin{pmatrix}H-\frac{1}{\eta}I&\frac{1}{\eta}I\\ \frac{1}{\eta}I&H-\frac{1}{\eta}I\end{pmatrix}. |  | (61) |

Along the diagonal and anti-diagonal directions, this reduces to

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∇2𝒜η​(w¯,w¯)​[(u,u),(u,u)]\displaystyle\nabla^{2}\mathcal{A}\_{\eta}(\bar{w},\bar{w})[(u,u),(u,u)] | =2​u⊤​H​u,\displaystyle=2\,u^{\top}Hu, |  | (62) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∇2𝒜η​(w¯,w¯)​[(u,−u),(u,−u)]\displaystyle\nabla^{2}\mathcal{A}\_{\eta}(\bar{w},\bar{w})[(u,-u),(u,-u)] | =2​u⊤​(H−2η​I)​u.\displaystyle=2\,u^{\top}\!\left(H-\frac{2}{\eta}I\right)u. |  | (63) |

The first expression is always nonnegative at a local minimum, so diagonal perturbations are stable. The second changes sign when λmax​(H)\lambda\_{\max}(H) exceeds 2/η2/\eta: at that threshold, the fixed point of the edge coupling loses stability to perturbations that split the two iterates apart, which is the onset of period-two oscillation.

###### Proof.

Differentiating 𝒜η\mathcal{A}\_{\eta} twice gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇2𝒜η​(x,y)=(∇2L​(x)−1η​I1η​I1η​I∇2L​(y)−1η​I).\displaystyle\nabla^{2}\mathcal{A}\_{\eta}(x,y)=\begin{pmatrix}\nabla^{2}L(x)-\frac{1}{\eta}I&\frac{1}{\eta}I\\ \frac{1}{\eta}I&\nabla^{2}L(y)-\frac{1}{\eta}I\end{pmatrix}. |  | (64) |

Evaluating at (w¯,w¯)(\bar{w},\bar{w}) gives the stated formula, and the diagonal and
anti-diagonal restrictions follow by direct substitution.
∎

#### Proof of [Theorem˜2.3](#S2.Thmtheorem3 "Theorem 2.3 (Center reduction and the edge eigenproblem). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability").

###### Proof.

Define the center-balance map and the symmetrized loss by

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | F​(m,a)\displaystyle F(m,a) | ≜12​(∇L​(m+a)+∇L​(m−a)),\displaystyle\triangleq\frac{1}{2}\bigl(\nabla L(m+a)+\nabla L(m-a)\bigr), |  | (65) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | G​(m,a)\displaystyle G(m,a) | ≜12​(L​(m+a)+L​(m−a)).\displaystyle\triangleq\frac{1}{2}\bigl(L(m+a)+L(m-a)\bigr). |  | (66) |

Since F​(w¯,0)=0F(\bar{w},0)=0 and Dm​F​(w¯,0)=HD\_{m}F(\bar{w},0)=H is invertible, the implicit function theorem gives a unique smooth map
m​(a)m(a) near a=0a=0 with m​(0)=w¯m(0)=\bar{w} and F​(m​(a),a)=0F(m(a),a)=0, proving
([15](#S2.E15 "Equation 15 ‣ Theorem 2.3 (Center reduction and the edge eigenproblem). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability")). Because F​(m,−a)=F​(m,a)F(m,-a)=F(m,a), uniqueness implies
m​(−a)=m​(a)m(-a)=m(a).

Setting 𝒫​(a)≜G​(m​(a),a)\mathcal{P}(a)\triangleq G(m(a),a) gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | Φη​(a)=Ψη​(m​(a),a)=𝒫​(a)−1η​‖a‖2,\displaystyle\Phi\_{\eta}(a)=\Psi\_{\eta}(m(a),a)=\mathcal{P}(a)-\frac{1}{\eta}\|a\|^{2}, |  | (67) |

which is ([17](#S2.E17 "Equation 17 ‣ Theorem 2.3 (Center reduction and the edge eigenproblem). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability")). Since Dm​G​(m​(a),a)=F​(m​(a),a)=0D\_{m}G(m(a),a)=F(m(a),a)=0, the mm-dependence drops out when differentiating 𝒫\mathcal{P}, and the chain rule gives

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | D​𝒫​(a)​[h]\displaystyle D\mathcal{P}(a)[h] | =Da​G​(m​(a),a)​[h]=12​⟨∇L​(m​(a)+a)−∇L​(m​(a)−a),h⟩,\displaystyle=D\_{a}G(m(a),a)[h]=\frac{1}{2}\Bigl\langle\nabla L(m(a)+a)-\nabla L(m(a)-a),\,h\Bigr\rangle, |  | (68) |

which is ([18](#S2.E18 "Equation 18 ‣ Theorem 2.3 (Center reduction and the edge eigenproblem). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability")). It follows that ∇Φη​(a)=∇𝒫​(a)−2η​a\nabla\Phi\_{\eta}(a)=\nabla\mathcal{P}(a)-\frac{2}{\eta}a, proving ([19](#S2.E19 "Equation 19 ‣ Theorem 2.3 (Center reduction and the edge eigenproblem). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability")). Since F​(m​(a),a)=0F(m(a),a)=0 already
imposes the mm-criticality equation, ∇Φη​(a)=0\nabla\Phi\_{\eta}(a)=0 is equivalent
to full criticality of Ψη\Psi\_{\eta}, hence of 𝒜η\mathcal{A}\_{\eta} under the
change of variables (x,y)=(m−a,m+a)(x,y)=(m-a,m+a).

Differentiating ([18](#S2.E18 "Equation 18 ‣ Theorem 2.3 (Center reduction and the edge eigenproblem). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability")) at a=0a=0 and using D​m​(0)=0Dm(0)=0 (since mm is even) gives ∇2𝒫​(0)=H\nabla^{2}\mathcal{P}(0)=H, and therefore
∇2Φη​(0)=H−2η​I\nabla^{2}\Phi\_{\eta}(0)=H-\frac{2}{\eta}I.
∎

#### Proof of [Proposition˜2.4](#S2.Thmtheorem4 "Proposition 2.4 (Quartic expansion near a fixed point). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability").

###### Proof.

Because mm is even, D​m​(0)=0Dm(0)=0. Differentiating the identity
F​(m​(a),a)=0F(m(a),a)=0 twice at a=0a=0 and solving for the second derivative of the center map gives

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | H​D2​m​(0)​[h,h]+∇3L​(w¯)​[h,h,⋅]\displaystyle H\,D^{2}m(0)[h,h]+\nabla^{3}L(\bar{w})[h,h,\cdot] | =0,\displaystyle=0, |  | (69) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | D2​m​(0)​[h,h]\displaystyle D^{2}m(0)[h,h] | =−H−1​∇3L​(w¯)​[h,h,⋅].\displaystyle=-H^{-1}\nabla^{3}L(\bar{w})[h,h,\cdot]. |  | (70) |

Taylor expanding the even map mm then yields ([21](#S2.E21 "Equation 21 ‣ Proposition 2.4 (Quartic expansion near a fixed point). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | m​(a)=w¯−12​H−1​∇3L​(w¯)​[a,a,⋅]+O​(‖a‖4).\displaystyle m(a)=\bar{w}-\frac{1}{2}H^{-1}\nabla^{3}L(\bar{w})[a,a,\cdot]+O(\|a\|^{4}). |  | (71) |

To obtain the quartic jet, set p​(a)≜m​(a)−w¯=O​(‖a‖2)p(a)\triangleq m(a)-\bar{w}=O(\|a\|^{2}).
Taylor expanding L​(w¯+p±a)L(\bar{w}+p\pm a) through quartic order and averaging gives

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒫​(a)\displaystyle\mathcal{P}(a) | =L​(w¯)+12​⟨H​a,a⟩+12​⟨H​p,p⟩\displaystyle=L(\bar{w})+\frac{1}{2}\langle Ha,a\rangle+\frac{1}{2}\langle Hp,p\rangle |  | (72) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +12​⟨∇3L​(w¯)​[a,a,⋅],p⟩+124​∇4L​(w¯)​[a,a,a,a]+o​(‖a‖4).\displaystyle+\frac{1}{2}\Bigl\langle\nabla^{3}L(\bar{w})[a,a,\cdot],\,p\Bigr\rangle+\frac{1}{24}\nabla^{4}L(\bar{w})[a,a,a,a]+o(\|a\|^{4}). |  | (73) |

Substituting p​(a)=−12​H−1​∇3L​(w¯)​[a,a,⋅]+o​(‖a‖2)p(a)=-\frac{1}{2}H^{-1}\nabla^{3}L(\bar{w})[a,a,\cdot]+o(\|a\|^{2}) and collecting quartic terms yields 14​𝒬​(a)\frac{1}{4}\,\mathcal{Q}(a), proving
([22](#S2.E22 "Equation 22 ‣ Proposition 2.4 (Quartic expansion near a fixed point). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability")) and hence ([24](#S2.E24 "Equation 24 ‣ Proposition 2.4 (Quartic expansion near a fixed point). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability")).
∎

#### Proof of [Corollary˜2.5](#S2.Thmtheorem5 "Corollary 2.5 (Generic branching at the edge). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability") (Generic branching at the edge).

###### Proof.

Write μ≜2ηc−2η\mu\triangleq\frac{2}{\eta\_{c}}-\frac{2}{\eta} for the bifurcation parameter. By [Theorem˜2.3](#S2.Thmtheorem3 "Theorem 2.3 (Center reduction and the edge eigenproblem). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability"), the reduced functional has the expansion

|  |  |  |  |
| --- | --- | --- | --- |
|  | Φη​(a)=L​(w¯)+12​⟨(H−2η​I)​a,a⟩+14​𝒬​(a)+o​(‖a‖4+|μ|​‖a‖2).\displaystyle\Phi\_{\eta}(a)=L(\bar{w})+\frac{1}{2}\langle(H-\tfrac{2}{\eta}I)a,a\rangle+\frac{1}{4}\mathcal{Q}(a)+o(\|a\|^{4}+|\mu|\,\|a\|^{2}). |  | (74) |

We perform a Lyapunov–Schmidt reduction. Decompose a=ξ+ba=\xi+b with ξ∈Ec\xi\in E\_{c} and b∈Ec⟂b\in E\_{c}^{\perp}.
Since H−2ηc​IH-\frac{2}{\eta\_{c}}I is invertible on Ec⟂E\_{c}^{\perp},
the implicit function theorem solves the Ec⟂E\_{c}^{\perp}-component of
∇Φη​(ξ+b)=0\nabla\Phi\_{\eta}(\xi+b)=0 uniquely as b=b​(ξ,η)=O​(‖ξ‖3+|μ|​‖ξ‖)b=b(\xi,\eta)=O(\|\xi\|^{3}+|\mu|\,\|\xi\|). Substituting back reduces the problem to a potential on EcE\_{c} alone:

|  |  |  |  |
| --- | --- | --- | --- |
|  | φη​(ξ)=L​(w¯)+12​μ​‖ξ‖2+14​𝒬​(ξ)+o​(‖ξ‖4+|μ|​‖ξ‖2).\displaystyle\varphi\_{\eta}(\xi)=L(\bar{w})+\frac{1}{2}\mu\|\xi\|^{2}+\frac{1}{4}\mathcal{Q}(\xi)+o(\|\xi\|^{4}+|\mu|\,\|\xi\|^{2}). |  | (75) |

Writing ξ=α​v\xi=\alpha v with α≥0\alpha\geq 0 and v∈S​(Ec)v\in S(E\_{c}) separates the amplitude from the direction. The gradient of the reduced potential is

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇φη​(α​v)=μ​α​v+14​α3​∇𝒬​(v)+o​(α3+|μ|​α).\displaystyle\nabla\varphi\_{\eta}(\alpha v)=\mu\alpha v+\frac{1}{4}\alpha^{3}\nabla\mathcal{Q}(v)+o(\alpha^{3}+|\mu|\alpha). |  | (76) |

The tangential projection to S​(Ec)S(E\_{c}) forces ∇S(𝒬|S​(Ec))⁡(v)=o​(1)\nabla\_{S}(\mathcal{Q}|\_{S(E\_{c})})(v)=o(1),
so every small nontrivial critical point has its direction tending to a critical point of
𝒬|S​(Ec)\mathcal{Q}|\_{S(E\_{c})}. If uu is nondegenerate, the implicit
function theorem gives a unique nearby direction branch v​(η)→uv(\eta)\to u.

For the amplitude, we project radially and use Euler’s identity ⟨v,∇𝒬​(v)⟩=4​𝒬​(v)\langle v,\nabla\mathcal{Q}(v)\rangle=4\mathcal{Q}(v) for the homogeneous quartic to obtain

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 0\displaystyle 0 | =μ+α2​𝒬​(u)+o​(|μ|),\displaystyle=\mu+\alpha^{2}\mathcal{Q}(u)+o(|\mu|), |  | (77) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | α​(η)2\displaystyle\alpha(\eta)^{2} | =2η−2ηc𝒬​(u)+o​(|η−ηc|).\displaystyle=\frac{\frac{2}{\eta}-\frac{2}{\eta\_{c}}}{\mathcal{Q}(u)}+o(|\eta-\eta\_{c}|). |  | (78) |

This is real and positive on the side where (2η−2ηc)​𝒬​(u)>0\bigl(\frac{2}{\eta}-\frac{2}{\eta\_{c}}\bigr)\mathcal{Q}(u)>0.
∎

#### Proof of [Corollary˜3.1](#S3.Thmtheorem1 "Corollary 3.1 (Roles of 𝟐/𝜼). ‣ The roles of 𝟐/𝜼. ‣ 3 Why 𝟐/𝜼 Appears Everywhere ‣ The Origin of Edge of Stability") (Roles of 𝟐/𝜼2/\eta).

###### Proof.

[Theorem˜2.2](#S2.Thmtheorem2 "Theorem 2.2 (Propagator and One-Step Loss Change). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability") gives the propagator ([32](#S3.E32 "Equation 32 ‣ Corollary 3.1 (Roles of 𝟐/𝜼). ‣ The roles of 𝟐/𝜼. ‣ 3 Why 𝟐/𝜼 Appears Everywhere ‣ The Origin of Edge of Stability")) and the loss-change formula ([33](#S3.E33 "Equation 33 ‣ Corollary 3.1 (Roles of 𝟐/𝜼). ‣ The roles of 𝟐/𝜼. ‣ 3 Why 𝟐/𝜼 Appears Everywhere ‣ The Origin of Edge of Stability")). For the two-step return, we compute the displacement directly:

|  |  |  |  |
| --- | --- | --- | --- |
|  | wk+2−wk=dk+1+dk=(I−η​H¯k)​dk+dk=(2​I−η​H¯k)​dk.\displaystyle w\_{k+2}-w\_{k}=d\_{k+1}+d\_{k}=(I-\eta\bar{H}\_{k})d\_{k}+d\_{k}=(2I-\eta\bar{H}\_{k})d\_{k}. |  | (79) |

Since dk≠0d\_{k}\neq 0, we have wk+2=wkw\_{k+2}=w\_{k} if and only if H¯k​dk=(2/η)​dk\bar{H}\_{k}d\_{k}=(2/\eta)d\_{k}. The descent/ascent boundary follows from ‖dk‖22/(2​η)>0\|d\_{k}\|\_{2}^{2}/(2\eta)>0.
∎

#### Proof of [Theorem˜4.1](#S4.Thmtheorem1 "Theorem 4.1 (Curvature concentration at 𝟐/𝜼). ‣ Why the dynamics is forced toward 𝟐/𝜼. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability") (Curvature concentration at 𝟐/𝜼2/\eta).

###### Proof.

The loss-change formula ([Corollary˜3.1](#S3.Thmtheorem1 "Corollary 3.1 (Roles of 𝟐/𝜼). ‣ The roles of 𝟐/𝜼. ‣ 3 Why 𝟐/𝜼 Appears Everywhere ‣ The Origin of Edge of Stability")) writes each step as L​(wk)−L​(wk+1)=‖dk‖222​(2η−r~k)L(w\_{k})-L(w\_{k+1})=\frac{\|d\_{k}\|\_{2}^{2}}{2}(\frac{2}{\eta}-\widetilde{r}\_{k}). Summing from k=0k=0 to K−1K-1, the left side telescopes while the right side accumulates:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | L​(w0)−L​(wK)\displaystyle L(w\_{0})-L(w\_{K}) | =12​∑k=0K−1‖dk‖22​(2η−r~k),\displaystyle=\frac{1}{2}\sum\_{k=0}^{K-1}\|d\_{k}\|\_{2}^{2}\!\left(\frac{2}{\eta}-\widetilde{r}\_{k}\right), |  | (80) |

which is [Equation˜36](#S4.E36 "In Theorem 4.1 (Curvature concentration at 𝟐/𝜼). ‣ Why the dynamics is forced toward 𝟐/𝜼. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability"). Dividing by EK>0E\_{K}>0 yields
[Equation˜37](#S4.E37 "In Theorem 4.1 (Curvature concentration at 𝟐/𝜼). ‣ Why the dynamics is forced toward 𝟐/𝜼. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability"). The left side of [Equation˜37](#S4.E37 "In Theorem 4.1 (Curvature concentration at 𝟐/𝜼). ‣ Why the dynamics is forced toward 𝟐/𝜼. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability") is a weighted average of
r~k\widetilde{r}\_{k}, so it is bounded above by maxk<K⁡r~k\max\_{k<K}\widetilde{r}\_{k}. Using L​(wK)≥LinfL(w\_{K})\geq L\_{\inf} then gives the forcing bound and the asymptotic concentration:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | maxk<K⁡r~k\displaystyle\max\_{k<K}\widetilde{r}\_{k} | ≥2η−2​(L​(w0)−Linf)EK,\displaystyle\;\geq\;\frac{2}{\eta}-\frac{2(L(w\_{0})-L\_{\inf})}{E\_{K}}, |  | (81) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | |∑k<K‖dk‖22​r~kEK−2η|\displaystyle\left|\frac{\sum\_{k<K}\|d\_{k}\|\_{2}^{2}\,\widetilde{r}\_{k}}{E\_{K}}-\frac{2}{\eta}\right| | =2​|L​(w0)−L​(wK)|EK\displaystyle=\frac{2|L(w\_{0})-L(w\_{K})|}{E\_{K}} |  | (82) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ≤2​max⁡{|L​(w0)−Linf|,|Lsup−L​(w0)|}EK→0.\displaystyle\;\leq\;\frac{2\max\{|L(w\_{0})-L\_{\inf}|,\,|L\_{\sup}-L(w\_{0})|\}}{E\_{K}}\to 0. |  | (83) |

∎

#### Proof of [Theorem˜4.2](#S4.Thmtheorem2 "Theorem 4.2 (Concentration near 𝟐/𝜼). ‣ Why the dynamics is forced toward 𝟐/𝜼. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability") (Concentration within a window of 𝟐/𝜼2/\eta).

###### Proof.

Write xk≜2/η−r~kx\_{k}\triangleq 2/\eta-\widetilde{r}\_{k} and decompose x=x+−(−x)+x=x\_{+}-(-x)\_{+}. Combined with the telescoping identity ([Theorem˜4.1](#S4.Thmtheorem1 "Theorem 4.1 (Curvature concentration at 𝟐/𝜼). ‣ Why the dynamics is forced toward 𝟐/𝜼. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability")), this gives [Equation˜41](#S4.E41 "In Theorem 4.2 (Concentration near 𝟐/𝜼). ‣ Why the dynamics is forced toward 𝟐/𝜼. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability"):

|  |  |  |  |
| --- | --- | --- | --- |
|  | BK−−BK+=∑k=0K−1‖dk‖22​xk=2​(L​(w0)−L​(wK)).\displaystyle B\_{K}^{-}-B\_{K}^{+}=\sum\_{k=0}^{K-1}\|d\_{k}\|\_{2}^{2}x\_{k}=2\bigl(L(w\_{0})-L(w\_{K})\bigr). |  | (84) |

For the subcritical bound [Equation˜42](#S4.E42 "In Theorem 4.2 (Concentration near 𝟐/𝜼). ‣ Why the dynamics is forced toward 𝟐/𝜼. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability"), when r~k≤2/η−δ\widetilde{r}\_{k}\leq 2/\eta-\delta we have (2/η−r~k)+≥δ(2/\eta-\widetilde{r}\_{k})\_{+}\geq\delta; for the supercritical bound [Equation˜43](#S4.E43 "In Theorem 4.2 (Concentration near 𝟐/𝜼). ‣ Why the dynamics is forced toward 𝟐/𝜼. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability"), when r~k≥2/η+δ\widetilde{r}\_{k}\geq 2/\eta+\delta we have (r~k−2/η)+≥δ(\widetilde{r}\_{k}-2/\eta)\_{+}\geq\delta. Markov-type estimates then give

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | δ​∑r~k≤2/η−δ‖dk‖22\displaystyle\delta\sum\_{\widetilde{r}\_{k}\leq 2/\eta-\delta}\|d\_{k}\|\_{2}^{2} | ≤BK−=2​(L​(w0)−L​(wK))+BK+≤ 2​(L​(w0)−Linf)+BK+,\displaystyle\;\leq\;B\_{K}^{-}=2\bigl(L(w\_{0})-L(w\_{K})\bigr)+B\_{K}^{+}\;\leq\;2\bigl(L(w\_{0})-L\_{\inf}\bigr)+B\_{K}^{+}, |  | (85) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | δ​∑r~k≥2/η+δ‖dk‖22\displaystyle\delta\sum\_{\widetilde{r}\_{k}\geq 2/\eta+\delta}\|d\_{k}\|\_{2}^{2} | ≤BK+.\displaystyle\;\leq\;B\_{K}^{+}. |  | (86) |

Summing both inequalities and passing to K→∞K\to\infty under B∞+<∞B\_{\infty}^{+}<\infty gives
[Equation˜44](#S4.E44 "In Theorem 4.2 (Concentration near 𝟐/𝜼). ‣ Why the dynamics is forced toward 𝟐/𝜼. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability"). Dividing by EK→∞E\_{K}\to\infty yields [Equation˜45](#S4.E45 "In Theorem 4.2 (Concentration near 𝟐/𝜼). ‣ Why the dynamics is forced toward 𝟐/𝜼. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability").
∎

The step recurrence dk+1=(I−η​H¯k)​dkd\_{k+1}=(I-\eta\bar{H}\_{k})d\_{k} from [Theorem˜2.2](#S2.Thmtheorem2 "Theorem 2.2 (Propagator and One-Step Loss Change). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability") has the form of a discrete propagator with time-varying coefficient. When all eigenvalues of H¯k\bar{H}\_{k} lie in the interval [0,2/η][0,2/\eta], each one-step multiplier |1−η​λ||1-\eta\lambda| is at most one, so the step norm cannot grow. Outside this interval, however, some multiplier exceeds one and the step norm inflates. The question is: by how much? The following definition measures the degree of instability at each step, and the theorem shows that the cumulative effect of transient excursions outside [0,2/η][0,2/\eta] is controlled by their sum.

###### Definition A.2 (Excursion beyond the stability window).

For the recurrence dk+1=(I−η​Ak)​dkd\_{k+1}=(I-\eta A\_{k})d\_{k}, define

|  |  |  |  |
| --- | --- | --- | --- |
|  | κk≜max⁡{0,η​λmax​(Ak)−2,−η​λmin​(Ak)}.\displaystyle\kappa\_{k}\triangleq\max\bigl\{0,\;\eta\lambda\_{\max}(A\_{k})-2,\;-\eta\lambda\_{\min}(A\_{k})\bigr\}. |  | (87) |

When κk=0\kappa\_{k}=0, the spectrum lies entirely within [0,2/η][0,2/\eta] and the one-step map is a contraction.

###### Theorem A.3 (Discrete Stability Bound).

Assume AkA\_{k} is symmetric for all kk. Then the discrete propagator satisfies

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖𝒯​[k,s]‖op≤exp⁡(∑r=sk−1κr),\displaystyle\|\mathcal{T}[k,s]\|\_{\mathrm{op}}\leq\exp\!\left(\sum\_{r=s}^{k-1}\kappa\_{r}\right), |  | (88) |

and consequently the discrete strain satisfies

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖δk‖2≤η​∑s=0k−1exp⁡(∑r=s+1k−1κr)​‖fs‖2.\displaystyle\|\delta\_{k}\|\_{2}\leq\eta\sum\_{s=0}^{k-1}\exp\!\left(\sum\_{r=s+1}^{k-1}\kappa\_{r}\right)\|f\_{s}\|\_{2}. |  | (89) |

###### Proof.

For a symmetric matrix AA with eigenvalues {λi}\{\lambda\_{i}\}, the operator norm of the one-step propagator is ‖I−η​A‖op=maxi⁡|1−η​λi|\|I-\eta A\|\_{\mathrm{op}}=\max\_{i}|1-\eta\lambda\_{i}|. When η​λi∈[0,2]\eta\lambda\_{i}\in[0,2] the multiplier satisfies |1−η​λi|≤1|1-\eta\lambda\_{i}|\leq 1; when η​λi>2\eta\lambda\_{i}>2 we have |1−η​λi|=η​λi−1≤1+κ|1-\eta\lambda\_{i}|=\eta\lambda\_{i}-1\leq 1+\kappa; and when λi<0\lambda\_{i}<0 we have |1−η​λi|=1+η​|λi|≤1+κ|1-\eta\lambda\_{i}|=1+\eta|\lambda\_{i}|\leq 1+\kappa. In every case ‖I−η​Ak‖op≤1+κk\|I-\eta A\_{k}\|\_{\mathrm{op}}\leq 1+\kappa\_{k}. Applying this bound stepwise:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖𝒯​[k,s]‖op≤∏r=sk−1(1+κr)≤exp⁡(∑r=sk−1κr),\displaystyle\|\mathcal{T}[k,s]\|\_{\mathrm{op}}\leq\prod\_{r=s}^{k-1}(1+\kappa\_{r})\leq\exp\!\left(\sum\_{r=s}^{k-1}\kappa\_{r}\right), |  | (90) |

using log⁡(1+x)≤x\log(1+x)\leq x. The strain bound follows from iterating the recurrence

|  |  |  |  |
| --- | --- | --- | --- |
|  | δk+1=(I−η​Ak)​δk−η​fk\displaystyle\delta\_{k+1}=(I-\eta A\_{k})\delta\_{k}-\eta f\_{k} |  | (91) |

to obtain the variation-of-constants representation

|  |  |  |  |
| --- | --- | --- | --- |
|  | δk=−η​∑s=0k−1𝒯​[k,s+1]​fs,\displaystyle\delta\_{k}=-\eta\sum\_{s=0}^{k-1}\mathcal{T}[k,s{+}1]\,f\_{s}, |  | (92) |

then applying the propagator bound [Equation˜88](#A1.E88 "In Theorem A.3 (Discrete Stability Bound). ‣ Proof of Theorem˜4.2 (Concentration within a window of 𝟐/𝜼). ‣ Appendix A Proofs from the Main Text ‣ The Origin of Edge of Stability") together with the triangle inequality.
∎

#### Proof of [Corollary˜4.4](#S4.Thmtheorem4 "Corollary 4.4 (Near-Periodicity Implies Near-Critical Sharpness). ‣ Discrete stability and near-periodicity. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability") (Near-Periodicity Implies Near-Critical Sharpness).

###### Proof.

By the mean value theorem, H¯k​dk=∇L​(wk+1)−∇L​(wk)\bar{H}\_{k}d\_{k}=\nabla L(w\_{k+1})-\nabla L(w\_{k}). We rewrite the gradient difference in terms of the gradient sum to extract the 2/η2/\eta threshold:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | H¯k​dk\displaystyle\bar{H}\_{k}d\_{k} | =∇L​(wk+1)−∇L​(wk)\displaystyle=\nabla L(w\_{k+1})-\nabla L(w\_{k}) |  | (93) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =(∇L​(wk+1)+∇L​(wk))−2​∇L​(wk)\displaystyle=\bigl(\nabla L(w\_{k+1})+\nabla L(w\_{k})\bigr)-2\nabla L(w\_{k}) |  | (94) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =−1η​(wk+2−wk)+2η​dk.\displaystyle=-\frac{1}{\eta}(w\_{k+2}-w\_{k})+\frac{2}{\eta}\,d\_{k}. |  | (95) |

Taking the inner product with dkd\_{k}, dividing by ‖dk‖22\|d\_{k}\|\_{2}^{2}, and applying Cauchy–Schwarz to the displacement term gives [Equation˜50](#S4.E50 "In Corollary 4.4 (Near-Periodicity Implies Near-Critical Sharpness). ‣ Discrete stability and near-periodicity. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability").
∎

![Refer to caption](/html/2604.20446/assets/x7.png)


Figure 4: Two-step return ratio.
‖wk+2−wk‖/‖dk‖\|w\_{k+2}-w\_{k}\|/\|d\_{k}\| vs. training step for two learning rates (solid: rolling median; faint: raw). Before EoS onset the ratio exceeds 11, reflecting the progressive sharpening phase in which consecutive steps reinforce rather than reverse. At EoS onset, the ratio drops from O​(1)O(1) toward ∼0.15\sim 0.15, indicating
approximate (but not exact) period-two behavior; by
[Corollary˜4.4](#S4.Thmtheorem4 "Corollary 4.4 (Near-Periodicity Implies Near-Critical Sharpness). ‣ Discrete stability and near-periodicity. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability"), this directly bounds how close the directional
curvature of H¯k\bar{H}\_{k} is to 2/η2/\eta.

## Appendix B From Hessian Averages to True Sharpness

The concentration theorems of [Section˜4](#S4 "4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability") show that the step-averaged curvatures r~k\widetilde{r}\_{k} and r¯k\bar{r}\_{k} are forced toward 2/η2/\eta, but the Edge of Stability as observed experimentally involves the largest Hessian eigenvalue at actual points in parameter space. This appendix shows that each averaged curvature is in fact the exact directional curvature of the true Hessian at a specific interior point of the step segment, so the forcing transfers with no residual error.

###### Definition B.1 (Curvature along a step).

Fix a step kk with dk≠0d\_{k}\neq 0 and set uk≜dk/‖dk‖2u\_{k}\triangleq d\_{k}/\|d\_{k}\|\_{2}. Define

|  |  |  |  |
| --- | --- | --- | --- |
|  | qk​(τ)≜uk⊤​∇2L​(wk+τ​dk)​uk,τ∈[0,1].\displaystyle q\_{k}(\tau)\triangleq u\_{k}^{\top}\nabla^{2}L(w\_{k}+\tau d\_{k})\,u\_{k},\qquad\tau\in[0,1]. |  | (96) |

The uniform average of qkq\_{k} gives r¯k\bar{r}\_{k} and the triangularly weighted average gives r~k\widetilde{r}\_{k}. Since these are averages of a continuous function over [0,1][0,1], the mean value theorem guarantees that each is attained as a pointwise value of qkq\_{k}.

###### Proof of [Theorem˜4.3](#S4.Thmtheorem3 "Theorem 4.3 (Localization to the true Hessian). ‣ Exact sharpness forcing on each edge. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability").

Define the scalar edge restriction gk​(t)≜L​(wk+t​dk)g\_{k}(t)\triangleq L(w\_{k}+t\,d\_{k}) for t∈[0,1]t\in[0,1]. Its derivatives are

|  |  |  |  |
| --- | --- | --- | --- |
|  | gk′​(t)=⟨∇L​(wk+t​dk),dk⟩,gk′′​(t)=dk⊤​∇2L​(wk+t​dk)​dk=‖dk‖22​qk​(t).\displaystyle g\_{k}^{\prime}(t)=\langle\nabla L(w\_{k}+t\,d\_{k}),\,d\_{k}\rangle,\qquad g\_{k}^{\prime\prime}(t)=d\_{k}^{\top}\nabla^{2}L(w\_{k}+t\,d\_{k})\,d\_{k}=\|d\_{k}\|\_{2}^{2}\,q\_{k}(t). |  | (97) |

The gradient-descent update gives gk′​(0)=⟨∇L​(wk),dk⟩=−‖dk‖22/ηg\_{k}^{\prime}(0)=\langle\nabla L(w\_{k}),d\_{k}\rangle=-\|d\_{k}\|\_{2}^{2}/\eta.

*Localization of r~k\widetilde{r}\_{k}.* Taylor’s theorem with Lagrange remainder gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | gk​(1)=gk​(0)+gk′​(0)+12​gk′′​(ξk)\displaystyle g\_{k}(1)=g\_{k}(0)+g\_{k}^{\prime}(0)+\tfrac{1}{2}\,g\_{k}^{\prime\prime}(\xi\_{k}) |  | (98) |

for some ξk∈(0,1)\xi\_{k}\in(0,1). The integral form of the same expansion ([Theorem˜2.2](#S2.Thmtheorem2 "Theorem 2.2 (Propagator and One-Step Loss Change). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability")(ii)) gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | gk​(1)−gk​(0)=gk′​(0)+12​‖dk‖22​r~k.\displaystyle g\_{k}(1)-g\_{k}(0)=g\_{k}^{\prime}(0)+\tfrac{1}{2}\|d\_{k}\|\_{2}^{2}\,\widetilde{r}\_{k}. |  | (99) |

Comparing the two and dividing by 12​‖dk‖22\frac{1}{2}\|d\_{k}\|\_{2}^{2} yields r~k=qk​(ξk)=uk⊤​∇2L​(wk+ξk​dk)​uk\widetilde{r}\_{k}=q\_{k}(\xi\_{k})=u\_{k}^{\top}\nabla^{2}L(w\_{k}+\xi\_{k}d\_{k})\,u\_{k}.

*Localization of r¯k\bar{r}\_{k}.* The ordinary mean value theorem applied to hk​(t)≜gk′​(t)h\_{k}(t)\triangleq g\_{k}^{\prime}(t) gives hk​(1)−hk​(0)=hk′​(ζk)h\_{k}(1)-h\_{k}(0)=h\_{k}^{\prime}(\zeta\_{k})
for some ζk∈(0,1)\zeta\_{k}\in(0,1). The left side equals ⟨∇L​(wk+1)−∇L​(wk),dk⟩=‖dk‖22​r¯k\langle\nabla L(w\_{k+1})-\nabla L(w\_{k}),\,d\_{k}\rangle=\|d\_{k}\|\_{2}^{2}\,\bar{r}\_{k}; the right side equals gk′′​(ζk)=‖dk‖22​qk​(ζk)g\_{k}^{\prime\prime}(\zeta\_{k})=\|d\_{k}\|\_{2}^{2}\,q\_{k}(\zeta\_{k}). Dividing gives r¯k=qk​(ζk)=uk⊤​∇2L​(wk+ζk​dk)​uk\bar{r}\_{k}=q\_{k}(\zeta\_{k})=u\_{k}^{\top}\nabla^{2}L(w\_{k}+\zeta\_{k}d\_{k})\,u\_{k}.

*Eigenvalue bounds.* Since uku\_{k} is a unit vector, uk⊤​∇2L​(z)​uk≤λmax​(∇2L​(z))u\_{k}^{\top}\nabla^{2}L(z)\,u\_{k}\leq\lambda\_{\max}(\nabla^{2}L(z)) for any zz.

*Forcing inequality.* Substituting r~k≤λmax​(∇2L​(wk+ξk​dk))\widetilde{r}\_{k}\leq\lambda\_{\max}(\nabla^{2}L(w\_{k}+\xi\_{k}d\_{k})) into [Equation˜36](#S4.E36 "In Theorem 4.1 (Curvature concentration at 𝟐/𝜼). ‣ Why the dynamics is forced toward 𝟐/𝜼. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability") gives [Equation˜47](#S4.E47 "In Theorem 4.3 (Localization to the true Hessian). ‣ Exact sharpness forcing on each edge. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability"). Using L​(wK)≥LinfL(w\_{K})\geq L\_{\inf} gives [Equation˜48](#S4.E48 "In Theorem 4.3 (Localization to the true Hessian). ‣ Exact sharpness forcing on each edge. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability").

*Propagator bound.* From [Corollary˜4.4](#S4.Thmtheorem4 "Corollary 4.4 (Near-Periodicity Implies Near-Critical Sharpness). ‣ Discrete stability and near-periodicity. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability"), r¯k≥2/η−‖wk+2−wk‖2/(η​‖dk‖2)\bar{r}\_{k}\geq 2/\eta-\|w\_{k+2}-w\_{k}\|\_{2}/(\eta\|d\_{k}\|\_{2}). Since r¯k≤λmax​(∇2L​(wk+ζk​dk))\bar{r}\_{k}\leq\lambda\_{\max}(\nabla^{2}L(w\_{k}+\zeta\_{k}d\_{k})), the bound [Equation˜49](#S4.E49 "In Theorem 4.3 (Localization to the true Hessian). ‣ Exact sharpness forcing on each edge. ‣ 4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability") follows.
∎

## Appendix C Growth above the threshold and oscillatory cancellation

This appendix proves the two stability mechanisms from [Section˜5](#S5 "5 Stability Mechanisms at the Edge ‣ The Origin of Edge of Stability"), which together explain why the dynamics remains bounded at the edge.

The first mechanism is growth above the threshold ([Proposition˜5.1](#S5.Thmtheorem1 "Proposition 5.1 (Growth above the threshold). ‣ Growth above the threshold. ‣ 5 Stability Mechanisms at the Edge ‣ The Origin of Edge of Stability")). When the curvature exceeds 2/η2/\eta, the one-step multiplier along the step direction has magnitude greater than one, so the step norm grows geometrically. On any bounded trajectory this growth is unsustainable, which means the dynamics cannot remain above the threshold for more than a few steps before being forced back below it.

The second mechanism is oscillatory cancellation ([Theorem˜5.2](#S5.Thmtheorem2 "Theorem 5.2 (Oscillatory cancellation). ‣ Oscillatory cancellation. ‣ 5 Stability Mechanisms at the Edge ‣ The Origin of Edge of Stability")). Once the curvature is back inside the stability window [0,2/η][0,2/\eta], the multiplier is close to −1-1, so the step direction alternates sign at each iteration. One might worry that small asymmetries in this alternation could accumulate into secular drift. The cancellation theorem shows that this does not happen: the total displacement is controlled by the last forcing value plus the total variation of the forcing sequence, regardless of how the multipliers vary within the window.

###### Proof of [Proposition˜5.1](#S5.Thmtheorem1 "Proposition 5.1 (Growth above the threshold). ‣ Growth above the threshold. ‣ 5 Stability Mechanisms at the Edge ‣ The Origin of Edge of Stability").

The propagator identity gives dk+1=(I−η​H¯k)​dkd\_{k+1}=(I-\eta\bar{H}\_{k})d\_{k}. Taking the inner product with dkd\_{k}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ⟨dk+1,dk⟩=‖dk‖22−η​dk⊤​H¯k​dk=(1−η​r¯k)​‖dk‖22.\displaystyle\langle d\_{k+1},d\_{k}\rangle=\|d\_{k}\|\_{2}^{2}-\eta\,d\_{k}^{\top}\bar{H}\_{k}d\_{k}=(1-\eta\bar{r}\_{k})\|d\_{k}\|\_{2}^{2}. |  | (100) |

If r¯k≥2/η+δ\bar{r}\_{k}\geq 2/\eta+\delta, then 1−η​r¯k≤−(1+η​δ)1-\eta\bar{r}\_{k}\leq-(1+\eta\delta), so

|  |  |  |  |
| --- | --- | --- | --- |
|  | ⟨dk+1,dk⟩≤−(1+η​δ)​‖dk‖22.\displaystyle\langle d\_{k+1},d\_{k}\rangle\leq-(1+\eta\delta)\|d\_{k}\|\_{2}^{2}. |  | (101) |

Cauchy–Schwarz then gives ‖dk+1‖2≥(1+η​δ)​‖dk‖2\|d\_{k+1}\|\_{2}\geq(1+\eta\delta)\|d\_{k}\|\_{2}.
Iterating from j=sj=s to j=t−1j=t-1 yields ‖dt‖2≥(1+η​δ)t−s​‖ds‖2\|d\_{t}\|\_{2}\geq(1+\eta\delta)^{t-s}\|d\_{s}\|\_{2}.
∎

###### Proof of [Theorem˜5.2](#S5.Thmtheorem2 "Theorem 5.2 (Oscillatory cancellation). ‣ Oscillatory cancellation. ‣ 5 Stability Mechanisms at the Edge ‣ The Origin of Edge of Stability").

Unrolling the recursion gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | xT=−η​∑s=0T−1as​us,as≜∏r=s+1T−1mr.\displaystyle x\_{T}=-\eta\sum\_{s=0}^{T-1}a\_{s}u\_{s},\qquad a\_{s}\triangleq\prod\_{r=s+1}^{T-1}m\_{r}. |  | (102) |

To exploit the alternating signs, write as=(−1)T−1−s​bsa\_{s}=(-1)^{T-1-s}b\_{s}, where

|  |  |  |  |
| --- | --- | --- | --- |
|  | bs≜∏r=s+1T−1|mr|.\displaystyle b\_{s}\triangleq\prod\_{r=s+1}^{T-1}|m\_{r}|. |  | (103) |

Since |mr|≤1|m\_{r}|\leq 1, the sequence b0≤b1≤⋯≤bT−1=1b\_{0}\leq b\_{1}\leq\cdots\leq b\_{T-1}=1 is nondecreasing.
Define the partial sums Aj≜∑s=0jasA\_{j}\triangleq\sum\_{s=0}^{j}a\_{s}. The alternating signs and the monotonicity of the bsb\_{s} sequence give |Aj|≤1|A\_{j}|\leq 1 for every jj.
Applying Abel summation:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑s=0T−1as​us=AT−1​uT−1+∑s=0T−2As​(us−us+1).\displaystyle\sum\_{s=0}^{T-1}a\_{s}u\_{s}=A\_{T-1}u\_{T-1}+\sum\_{s=0}^{T-2}A\_{s}(u\_{s}-u\_{s+1}). |  | (104) |

Taking absolute values and using |Aj|≤1|A\_{j}|\leq 1 gives ([51](#S5.E51 "Equation 51 ‣ Theorem 5.2 (Oscillatory cancellation). ‣ Oscillatory cancellation. ‣ 5 Stability Mechanisms at the Edge ‣ The Origin of Edge of Stability")).
∎

## Appendix D Curvature Concentration under Mini-Batch SGD

The forcing theorems of [Section˜4](#S4 "4 The Origin of Edge of Stability ‣ The Origin of Edge of Stability") were stated for full-batch gradient descent, but the telescoping structure that drives them does not require exact gradients. This appendix extends the edge-balance identity to mini-batch SGD.

When the gradient estimate at step kk is ∇L​(wk)+ξk\nabla L(w\_{k})+\xi\_{k} rather than ∇L​(wk)\nabla L(w\_{k}), the one-step loss change picks up two additional terms beyond the deterministic formula. The first is a cross term between the true gradient and the noise, which vanishes in expectation under the standard martingale-difference assumption. The second is a variance term proportional to ‖ξk‖22\|\xi\_{k}\|\_{2}^{2}, which persists in expectation and shifts the balance identity by the cumulative squared noise magnitude. The telescoping structure otherwise survives intact.

###### Theorem D.1 (Stochastic curvature concentration).

Let

|  |  |  |  |
| --- | --- | --- | --- |
|  | wk+1=wk−η​(∇L​(wk)+ξk),\displaystyle w\_{k+1}=w\_{k}-\eta\bigl(\nabla L(w\_{k})+\xi\_{k}\bigr), |  | (105) |

where ξk∈ℝd\xi\_{k}\in\mathbb{R}^{d} is an arbitrary noise sequence. Set
sk≜wk+1−wks\_{k}\triangleq w\_{k+1}-w\_{k}, and define H¯k\bar{H}\_{k}, H~k\widetilde{H}\_{k}, and

|  |  |  |  |
| --- | --- | --- | --- |
|  | r~k≜sk⊤​H~k​sk‖sk‖22(sk≠0)\displaystyle\widetilde{r}\_{k}\triangleq\frac{s\_{k}^{\top}\widetilde{H}\_{k}s\_{k}}{\|s\_{k}\|\_{2}^{2}}\qquad(s\_{k}\neq 0) |  | (106) |

along the stochastic step segment wk+τ​skw\_{k}+\tau s\_{k} exactly as in [Theorem˜2.2](#S2.Thmtheorem2 "Theorem 2.2 (Propagator and One-Step Loss Change). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability").
Then:

(i) The step increments satisfy the exact forced propagator

|  |  |  |  |
| --- | --- | --- | --- |
|  | sk+1=(I−η​H¯k)​sk−η​(ξk+1−ξk).\displaystyle s\_{k+1}=(I-\eta\bar{H}\_{k})s\_{k}-\eta(\xi\_{k+1}-\xi\_{k}). |  | (107) |

(ii) The one-step loss change satisfies

|  |  |  |  |
| --- | --- | --- | --- |
|  | L​(wk+1)−L​(wk)=−‖sk‖222​η​(2−η​r~k)+η​⟨∇L​(wk),ξk⟩+η​‖ξk‖22.\displaystyle L(w\_{k+1})-L(w\_{k})=-\frac{\|s\_{k}\|\_{2}^{2}}{2\eta}\Bigl(2-\eta\widetilde{r}\_{k}\Bigr)+\eta\langle\nabla L(w\_{k}),\xi\_{k}\rangle+\eta\|\xi\_{k}\|\_{2}^{2}. |  | (108) |

(iii) Summing over k=0,…,K−1k=0,\dots,K-1 gives the stochastic telescoping identity

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑k=0K−1‖sk‖22​(2η−r~k)=2​(L​(w0)−L​(wK))+2​η​∑k=0K−1⟨∇L​(wk),ξk⟩+2​η​∑k=0K−1‖ξk‖22.\displaystyle\sum\_{k=0}^{K-1}\|s\_{k}\|\_{2}^{2}\Bigl(\frac{2}{\eta}-\widetilde{r}\_{k}\Bigr)=2\bigl(L(w\_{0})-L(w\_{K})\bigr)+2\eta\sum\_{k=0}^{K-1}\langle\nabla L(w\_{k}),\xi\_{k}\rangle+2\eta\sum\_{k=0}^{K-1}\|\xi\_{k}\|\_{2}^{2}. |  | (109) |

If {ξk}\{\xi\_{k}\} is a square-integrable martingale-difference sequence with respect to the SGD filtration, then

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[∑k=0K−1‖sk‖22​(2η−r~k)]=2​𝔼​[L​(w0)−L​(wK)]+2​η​∑k=0K−1𝔼​‖ξk‖22.\displaystyle\mathbb{E}\!\left[\sum\_{k=0}^{K-1}\|s\_{k}\|\_{2}^{2}\Bigl(\frac{2}{\eta}-\widetilde{r}\_{k}\Bigr)\right]=2\mathbb{E}\bigl[L(w\_{0})-L(w\_{K})\bigr]+2\eta\sum\_{k=0}^{K-1}\mathbb{E}\|\xi\_{k}\|\_{2}^{2}. |  | (110) |

###### Proof.

For ([107](#A4.E107 "Equation 107 ‣ Theorem D.1 (Stochastic curvature concentration). ‣ Appendix D Curvature Concentration under Mini-Batch SGD ‣ The Origin of Edge of Stability")), write

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | sk+1\displaystyle s\_{k+1} | =−η​(∇L​(wk+1)+ξk+1)\displaystyle=-\eta\bigl(\nabla L(w\_{k+1})+\xi\_{k+1}\bigr) |  | (111) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =−η​(∇L​(wk)+ξk)−η​(∇L​(wk+1)−∇L​(wk))−η​(ξk+1−ξk)\displaystyle=-\eta\bigl(\nabla L(w\_{k})+\xi\_{k}\bigr)-\eta\bigl(\nabla L(w\_{k+1})-\nabla L(w\_{k})\bigr)-\eta(\xi\_{k+1}-\xi\_{k}) |  | (112) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =sk−η​H¯k​sk−η​(ξk+1−ξk).\displaystyle=s\_{k}-\eta\bar{H}\_{k}s\_{k}-\eta(\xi\_{k+1}-\xi\_{k}). |  | (113) |

For ([108](#A4.E108 "Equation 108 ‣ Theorem D.1 (Stochastic curvature concentration). ‣ Appendix D Curvature Concentration under Mini-Batch SGD ‣ The Origin of Edge of Stability")), the Taylor expansion with integral remainder and the substitution sk=−η​(∇L​(wk)+ξk)s\_{k}=-\eta(\nabla L(w\_{k})+\xi\_{k}) give

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | L​(wk+1)−L​(wk)\displaystyle L(w\_{k+1})-L(w\_{k}) | =∇L​(wk)⊤​sk+12​sk⊤​H~k​sk,\displaystyle=\nabla L(w\_{k})^{\top}s\_{k}+\frac{1}{2}s\_{k}^{\top}\widetilde{H}\_{k}s\_{k}, |  | (114) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∇L​(wk)⊤​sk\displaystyle\nabla L(w\_{k})^{\top}s\_{k} | =−η​‖∇L​(wk)‖22−η​⟨∇L​(wk),ξk⟩\displaystyle=-\eta\|\nabla L(w\_{k})\|\_{2}^{2}-\eta\langle\nabla L(w\_{k}),\xi\_{k}\rangle |  | (115) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =−1η​‖sk‖22+η​⟨∇L​(wk),ξk⟩+η​‖ξk‖22,\displaystyle=-\frac{1}{\eta}\|s\_{k}\|\_{2}^{2}+\eta\langle\nabla L(w\_{k}),\xi\_{k}\rangle+\eta\|\xi\_{k}\|\_{2}^{2}, |  | (116) |

where the last line uses ‖sk‖22=η2​‖∇L​(wk)+ξk‖22\|s\_{k}\|\_{2}^{2}=\eta^{2}\|\nabla L(w\_{k})+\xi\_{k}\|\_{2}^{2}.
Substituting yields ([108](#A4.E108 "Equation 108 ‣ Theorem D.1 (Stochastic curvature concentration). ‣ Appendix D Curvature Concentration under Mini-Batch SGD ‣ The Origin of Edge of Stability")); summing yields ([109](#A4.E109 "Equation 109 ‣ Theorem D.1 (Stochastic curvature concentration). ‣ Appendix D Curvature Concentration under Mini-Batch SGD ‣ The Origin of Edge of Stability")). The expectation identity follows from 𝔼​[⟨∇L​(wk),ξk⟩∣ℱk]=0\mathbb{E}[\langle\nabla L(w\_{k}),\xi\_{k}\rangle\mid\mathcal{F}\_{k}]=0.
∎

## Appendix E Discrete-Time Kelvin–Voigt Dynamics

The step recurrence dk+1=(I−η​H¯k)​dkd\_{k+1}=(I-\eta\bar{H}\_{k})d\_{k} from [Theorem˜2.2](#S2.Thmtheorem2 "Theorem 2.2 (Propagator and One-Step Loss Change). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability") governs how the step size of a single trajectory evolves. A natural next question is: how does the *difference* between two trajectories evolve? This is the question of algorithmic stability [bousquet2002stability, hardt2016train], and the edge coupling provides a natural framework for answering it.

The idea is to replace the homogeneous coupling 𝒜η\mathcal{A}\_{\eta}, which uses the same loss at both iterates, with a heterogeneous variant that assigns different losses to the two positions:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒜ηS,S′​(x,y)≜LS​(x)+LS′​(y)−12​η​‖x−y‖2.\displaystyle\mathcal{A}\_{\eta}^{S,S^{\prime}}(x,y)\;\triangleq\;L\_{S}(x)+L\_{S^{\prime}}(y)-\frac{1}{2\eta}\|x-y\|^{2}. |  | (117) |

Setting ∇x𝒜ηS,S′=0\nabla\_{x}\mathcal{A}\_{\eta}^{S,S^{\prime}}=0 recovers the gradient-descent step on dataset SS, while ∇y𝒜ηS,S′=0\nabla\_{y}\mathcal{A}\_{\eta}^{S,S^{\prime}}=0 recovers the step on dataset S′S^{\prime}. The parameter deviation δk=wk−wk′\delta\_{k}=w\_{k}-w\_{k}^{\prime} between the two trajectories then satisfies a forced linear recurrence whose stability boundary is again 2/η2/\eta.

The rest of this appendix makes this precise. We first define the quantities that measure the deviation ([Definition˜E.1](#A5.Thmtheorem1 "Definition E.1 (Discrete Strain and Stress). ‣ E.1 Discrete Algorithmic Strain and Data Stress ‣ Appendix E Discrete-Time Kelvin–Voigt Dynamics ‣ The Origin of Edge of Stability")), then derive the recurrence they satisfy ([Theorem˜E.2](#A5.Thmtheorem2 "Theorem E.2 (Discrete Kelvin–Voigt Variational Equation). ‣ E.2 The Discrete Kelvin–Voigt Equation ‣ Appendix E Discrete-Time Kelvin–Voigt Dynamics ‣ The Origin of Edge of Stability")), and finally solve it via a discrete propagator ([Theorem˜E.5](#A5.Thmtheorem5 "Theorem E.5 (Discrete Strain Propagation Formula). ‣ E.3 The Discrete Propagator and Strain Integral ‣ Appendix E Discrete-Time Kelvin–Voigt Dynamics ‣ The Origin of Edge of Stability")).

### E.1 Discrete Algorithmic Strain and Data Stress

When two copies of gradient descent are run on different datasets from a common initialization, the parameter deviation grows due to two distinct effects: the curvature of the loss landscape amplifies existing displacement, and the gradient mismatch between the two datasets injects new displacement at each step. To disentangle these, we introduce the following quantities. Consider gradient descent with step size η>0\eta>0 applied to two datasets:

|  |  |  |  |
| --- | --- | --- | --- |
|  | wk+1=wk−η​∇LS​(wk),wk+1′=wk′−η​∇LS′​(wk′),\displaystyle w\_{k+1}=w\_{k}-\eta\nabla L\_{S}(w\_{k}),\qquad w^{\prime}\_{k+1}=w^{\prime}\_{k}-\eta\nabla L\_{S^{\prime}}(w^{\prime}\_{k}), |  | (118) |

with common initialization w0=w0′w\_{0}=w^{\prime}\_{0}.

###### Definition E.1 (Discrete Strain and Stress).

The discrete algorithmic strain and data stress are

|  |  |  |  |
| --- | --- | --- | --- |
|  | δk≜wk−wk′,fk≜∇LS​(wk′)−∇LS′​(wk′).\displaystyle\delta\_{k}\triangleq w\_{k}-w^{\prime}\_{k},\qquad f\_{k}\triangleq\nabla L\_{S}(w^{\prime}\_{k})-\nabla L\_{S^{\prime}}(w^{\prime}\_{k}). |  | (119) |

The discrete stability matrix is the segment-averaged Hessian

|  |  |  |  |
| --- | --- | --- | --- |
|  | Ak≜∫01∇2LS​(wk′+τ​δk)​𝑑τ.\displaystyle A\_{k}\triangleq\int\_{0}^{1}\nabla^{2}L\_{S}\bigl(w^{\prime}\_{k}+\tau\delta\_{k}\bigr)\,d\tau. |  | (120) |

### E.2 The Discrete Kelvin–Voigt Equation

The strain δk\delta\_{k} evolves through two contributions at each step: the gradient of LSL\_{S} at wkw\_{k} versus at wk′w\_{k}^{\prime} (which depends on both the curvature and the current displacement), and the gradient mismatch fkf\_{k} between the two datasets at the reference point wk′w\_{k}^{\prime}. The mean value theorem along the segment from wk′w\_{k}^{\prime} to wkw\_{k} absorbs the displacement dependence into the stability matrix AkA\_{k}, separating curvature amplification from data-driven forcing.

###### Theorem E.2 (Discrete Kelvin–Voigt Variational Equation).

The discrete algorithmic strain satisfies the recurrence

|  |  |  |  |
| --- | --- | --- | --- |
|  | δk+1=(I−η​Ak)​δk−η​fk,δ0=0.\displaystyle\delta\_{k+1}=(I-\eta A\_{k})\,\delta\_{k}-\eta f\_{k},\qquad\delta\_{0}=0. |  | (121) |

This is precisely the explicit Euler discretization with time step η\eta of the continuous Kelvin–Voigt equation [Equation˜135](#A6.E135 "In Theorem F.3 (Kelvin–Voigt Variational Equation). ‣ F.2 The Kelvin–Voigt Constitutive Equation ‣ Appendix F Continuous-Time Kelvin–Voigt Framework ‣ The Origin of Edge of Stability").

###### Proof.

Subtracting the GD updates [Equation˜118](#A5.E118 "In E.1 Discrete Algorithmic Strain and Data Stress ‣ Appendix E Discrete-Time Kelvin–Voigt Dynamics ‣ The Origin of Edge of Stability"):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | δk+1\displaystyle\delta\_{k+1} | =wk+1−wk+1′=δk−η​(∇LS​(wk)−∇LS′​(wk′)).\displaystyle=w\_{k+1}-w^{\prime}\_{k+1}=\delta\_{k}-\eta\bigl(\nabla L\_{S}(w\_{k})-\nabla L\_{S^{\prime}}(w^{\prime}\_{k})\bigr). |  | (122) |

Adding and subtracting ∇LS​(wk′)\nabla L\_{S}(w^{\prime}\_{k}):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | δk+1\displaystyle\delta\_{k+1} | =δk−η​(∇LS​(wk)−∇LS​(wk′))−η​(∇LS​(wk′)−∇LS′​(wk′))⏟=fk.\displaystyle=\delta\_{k}-\eta\bigl(\nabla L\_{S}(w\_{k})-\nabla L\_{S}(w^{\prime}\_{k})\bigr)-\eta\underbrace{\bigl(\nabla L\_{S}(w^{\prime}\_{k})-\nabla L\_{S^{\prime}}(w^{\prime}\_{k})\bigr)}\_{=f\_{k}}. |  | (123) |

Applying the mean value theorem to the first bracketed term:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇LS​(wk)−∇LS​(wk′)=(∫01∇2LS​(wk′+τ​δk)​𝑑τ)​δk=Ak​δk.\displaystyle\nabla L\_{S}(w\_{k})-\nabla L\_{S}(w^{\prime}\_{k})=\Bigl(\int\_{0}^{1}\nabla^{2}L\_{S}(w^{\prime}\_{k}+\tau\delta\_{k})\,d\tau\Bigr)\delta\_{k}=A\_{k}\,\delta\_{k}. |  | (124) |

Substituting yields δk+1=(I−η​Ak)​δk−η​fk\delta\_{k+1}=(I-\eta A\_{k})\delta\_{k}-\eta f\_{k}.
∎

###### Remark E.3 (From Continuous to Discrete).

Forward Euler discretization of ∂tδ+A​(t)​δ=−f​(t)\partial\_{t}\delta+A(t)\delta=-f(t) gives δk+1=(I−η​Ak)​δk−η​fk\delta\_{k+1}=(I-\eta A\_{k})\delta\_{k}-\eta f\_{k}. [Theorem˜E.2](#A5.Thmtheorem2 "Theorem E.2 (Discrete Kelvin–Voigt Variational Equation). ‣ E.2 The Discrete Kelvin–Voigt Equation ‣ Appendix E Discrete-Time Kelvin–Voigt Dynamics ‣ The Origin of Edge of Stability") shows that the discrete gradient-descent strain satisfies this equation with no approximation, so the stability boundaries of explicit Euler are inherited by the actual dynamics.

### E.3 The Discrete Propagator and Strain Integral

The recurrence in [Theorem˜E.2](#A5.Thmtheorem2 "Theorem E.2 (Discrete Kelvin–Voigt Variational Equation). ‣ E.2 The Discrete Kelvin–Voigt Equation ‣ Appendix E Discrete-Time Kelvin–Voigt Dynamics ‣ The Origin of Edge of Stability") is a forced linear system, and such systems can be solved by variation of constants. The idea is to first understand the homogeneous part (how a displacement at step ss is transported forward to step kk by the curvature alone), and then sum up the contributions of each past stress fsf\_{s} after it has been transported forward. The object that encodes this forward transport is the discrete propagator.

###### Definition E.4 (Discrete Propagator).

The discrete propagator 𝒯​[k,s]\mathcal{T}[k,s] for the system [Equation˜121](#A5.E121 "In Theorem E.2 (Discrete Kelvin–Voigt Variational Equation). ‣ E.2 The Discrete Kelvin–Voigt Equation ‣ Appendix E Discrete-Time Kelvin–Voigt Dynamics ‣ The Origin of Edge of Stability") is

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒯​[k,s]≜{(I−η​Ak−1)​(I−η​Ak−2)​⋯​(I−η​As),k>s,I,k=s.\displaystyle\mathcal{T}[k,s]\triangleq\begin{cases}(I-\eta A\_{k-1})(I-\eta A\_{k-2})\cdots(I-\eta A\_{s}),&k>s,\\ I,&k=s.\end{cases} |  | (125) |

###### Theorem E.5 (Discrete Strain Propagation Formula).

The discrete algorithmic strain admits the representation

|  |  |  |  |
| --- | --- | --- | --- |
|  | δk=−η​∑s=0k−1𝒯​[k,s+1]​fs.\displaystyle\delta\_{k}=-\eta\sum\_{s=0}^{k-1}\mathcal{T}[k,s+1]\,f\_{s}. |  | (126) |

###### Proof.

By induction on kk. For k=0k=0, δ0=0\delta\_{0}=0 and the sum is empty. Assuming the formula holds for kk, the recurrence [Equation˜121](#A5.E121 "In Theorem E.2 (Discrete Kelvin–Voigt Variational Equation). ‣ E.2 The Discrete Kelvin–Voigt Equation ‣ Appendix E Discrete-Time Kelvin–Voigt Dynamics ‣ The Origin of Edge of Stability") gives

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | δk+1\displaystyle\delta\_{k+1} | =(I−η​Ak)​δk−η​fk\displaystyle=(I-\eta A\_{k})\,\delta\_{k}-\eta f\_{k} |  | (127) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =(I−η​Ak)​(−η​∑s=0k−1𝒯​[k,s+1]​fs)−η​fk\displaystyle=(I-\eta A\_{k})\Bigl(-\eta\sum\_{s=0}^{k-1}\mathcal{T}[k,s+1]\,f\_{s}\Bigr)-\eta f\_{k} |  | (128) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =−η​∑s=0k−1(I−η​Ak)​𝒯​[k,s+1]⏟=𝒯​[k+1,s+1]​fs−η​I⏟=𝒯​[k+1,k+1]​fk\displaystyle=-\eta\sum\_{s=0}^{k-1}\underbrace{(I-\eta A\_{k})\mathcal{T}[k,s+1]}\_{=\mathcal{T}[k+1,s+1]}f\_{s}-\eta\underbrace{I}\_{=\mathcal{T}[k+1,k+1]}f\_{k} |  | (129) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =−η​∑s=0k𝒯​[k+1,s+1]​fs,\displaystyle=-\eta\sum\_{s=0}^{k}\mathcal{T}[k+1,s+1]\,f\_{s}, |  | (130) |

which is the formula at k+1k+1.
∎

## Appendix F Continuous-Time Kelvin–Voigt Framework

The Edge of Stability is inherently discrete [cohen2021gradient], but developing the framework first in continuous time makes the qualitative change introduced by discretization transparent. In that setting, the difference between two nearby training trajectories satisfies a first-order linear variational equation, which we interpret as a Kelvin-Voigt-type constitutive law for algorithmic strain.

In continuous time (gradient flow), positive curvature is always stabilizing: larger eigenvalues of the Hessian cause faster contraction of perturbations, and only negative eigenvalues drive instability. Discretization breaks this monotonicity. With a finite step size η\eta, the stability window becomes [0,2/η][0,2/\eta]: eigenvalues above 2/η2/\eta flip the sign of the one-step multiplier past −1-1 and cause oscillatory instability, even though they are positive. It is this upper boundary that gives rise to the Edge of Stability. The continuous-time derivations in this appendix parallel those of [appendix˜E](#A5 "Appendix E Discrete-Time Kelvin–Voigt Dynamics ‣ The Origin of Edge of Stability"), with the stability matrix A​(t)A(t) now defined as the path-averaged Hessian between two gradient-flow trajectories and the stress f​(t)f(t) as the gradient mismatch at the reference trajectory.

### F.1 Algorithmic Strain and Data Stress

Consider two trajectories wS​(t)w\_{S}(t) and wS′​(t)w\_{S^{\prime}}(t), initialized at the same point w0w\_{0} and evolving under gradient flow on datasets SS and S′S^{\prime}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂twS​(t)=−∇LS​(wS​(t)),∂twS′​(t)=−∇LS′​(wS′​(t)).\displaystyle\partial\_{t}w\_{S}(t)=-\nabla L\_{S}(w\_{S}(t)),\qquad\partial\_{t}w\_{S^{\prime}}(t)=-\nabla L\_{S^{\prime}}(w\_{S^{\prime}}(t)). |  | (131) |

###### Definition F.1 (Algorithmic Strain and Data Stress).

Let wS,wS′:[0,T]→ℝdw\_{S},w\_{S^{\prime}}:[0,T]\to\mathbb{R}^{d} solve [Equation˜131](#A6.E131 "In F.1 Algorithmic Strain and Data Stress ‣ Appendix F Continuous-Time Kelvin–Voigt Framework ‣ The Origin of Edge of Stability") with wS​(0)=wS′​(0)=w0w\_{S}(0)=w\_{S^{\prime}}(0)=w\_{0}. The algorithmic strain and data stress are

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | δ​(t)\displaystyle\delta(t) | ≜wS​(t)−wS′​(t),\displaystyle\triangleq w\_{S}(t)-w\_{S^{\prime}}(t), |  | (132) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | f​(t)\displaystyle f(t) | ≜∇LS​(wS′​(t))−∇LS′​(wS′​(t)).\displaystyle\triangleq\nabla L\_{S}(w\_{S^{\prime}}(t))-\nabla L\_{S^{\prime}}(w\_{S^{\prime}}(t)). |  | (133) |

The strain measures the displacement between trajectories; the stress isolates the direct effect of the dataset change from the indirect effect of parameter displacement.

### F.2 The Kelvin–Voigt Constitutive Equation

The strain δ​(t)\delta(t) satisfies a first-order linear ODE that arises by subtracting the two gradient flow equations. The gradient difference between the same loss LSL\_{S} evaluated at two nearby points is handled by the mean value theorem, which absorbs the displacement dependence into the stability matrix A​(t)A(t) and leaves the stress f​(t)f(t) as a pure forcing term.

###### Definition F.2 (Stability Matrix).

The stability matrix is the integral mean of the Hessian along the segment connecting the two trajectories:

|  |  |  |  |
| --- | --- | --- | --- |
|  | A​(t)≜∫01∇2LS​(wS′​(t)+τ​(wS​(t)−wS′​(t)))​𝑑τ.\displaystyle A(t)\triangleq\int\_{0}^{1}\nabla^{2}L\_{S}\bigl(w\_{S^{\prime}}(t)+\tau(w\_{S}(t)-w\_{S^{\prime}}(t))\bigr)\,d\tau. |  | (134) |

###### Theorem F.3 (Kelvin–Voigt Variational Equation).

The algorithmic strain δ​(t)\delta(t) satisfies the variational equation

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂tδ​(t)+A​(t)​δ​(t)=−f​(t),δ​(0)=0.\displaystyle\partial\_{t}\delta(t)+A(t)\delta(t)=-f(t),\quad\delta(0)=0. |  | (135) |

This has the form of the constitutive equation for an anisotropic Kelvin–Voigt viscoelastic material, with A​(t)A(t) as the elastic modulus and f​(t)f(t) as the applied stress. The equation is state-dependent: A​(t)A(t) depends on δ​(t)\delta(t) through the mean-value integration path.

###### Proof.

Subtracting the gradient flow equations and decomposing gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂tδ\displaystyle\partial\_{t}\delta | =−∇LS​(wS)+∇LS′​(wS′)\displaystyle=-\nabla L\_{S}(w\_{S})+\nabla L\_{S^{\prime}}(w\_{S^{\prime}}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =−∇LS​(wS)+∇LS​(wS′)−∇LS​(wS′)+∇LS′​(wS′)\displaystyle=-\nabla L\_{S}(w\_{S})+\nabla L\_{S}(w\_{S^{\prime}})-\nabla L\_{S}(w\_{S^{\prime}})+\nabla L\_{S^{\prime}}(w\_{S^{\prime}}) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =−[∇LS​(wS)−∇LS​(wS′)]−[∇LS​(wS′)−∇LS′​(wS′)]⏟=f​(t).\displaystyle=-\bigl[\nabla L\_{S}(w\_{S})-\nabla L\_{S}(w\_{S^{\prime}})\bigr]-\underbrace{\bigl[\nabla L\_{S}(w\_{S^{\prime}})-\nabla L\_{S^{\prime}}(w\_{S^{\prime}})\bigr]}\_{=f(t)}. |  | (136) |

The first bracket is the gradient difference of the same objective at two points. By the mean value theorem for C2C^{2} functions,

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇LS​(wS)−∇LS​(wS′)=(∫01∇2LS​(wS′+τ​δ)​𝑑τ)​δ=A​(t)​δ​(t).\displaystyle\nabla L\_{S}(w\_{S})-\nabla L\_{S}(w\_{S^{\prime}})=\Bigl(\int\_{0}^{1}\nabla^{2}L\_{S}\bigl(w\_{S^{\prime}}+\tau\,\delta\bigr)\,d\tau\Bigr)\delta=A(t)\,\delta(t). |  | (137) |

Substituting yields

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂tδ+A​(t)​δ=−f​(t).\displaystyle\partial\_{t}\delta+A(t)\delta=-f(t). |  | (138) |

The initial condition δ​(0)=0\delta(0)=0 is immediate from wS​(0)=wS′​(0)w\_{S}(0)=w\_{S^{\prime}}(0).
∎

###### Remark F.4 (Scope of the Kelvin–Voigt Correspondence).

The classical Kelvin–Voigt model assumes A​(t)≻0A(t)\succ 0, so all perturbations decay. Here A​(t)A(t) can be indefinite during saddle-point traversal: negative eigenvalues drive strain growth, while the viscous term ∂tδ\partial\_{t}\delta provides dissipative competition. Instability requires sustained negative curvature, quantified by ∫α−​(t)​𝑑t\int\alpha\_{-}(t)\,dt in [Theorem˜F.8](#A6.Thmtheorem8 "Theorem F.8 (Effective Stability Bound). ‣ F.3 KV Stability: The Role of Curvature ‣ Appendix F Continuous-Time Kelvin–Voigt Framework ‣ The Origin of Edge of Stability"). In discrete time ([appendix˜E](#A5 "Appendix E Discrete-Time Kelvin–Voigt Dynamics ‣ The Origin of Edge of Stability")), positive curvature also has a stability boundary: eigenvalues exceeding 2/η2/\eta cause oscillatory instability, a phenomenon absent in continuous time.

The solution of the Kelvin–Voigt equation is expressed through the propagator 𝒯​(t,s)\mathcal{T}(t,s), which maps the state at time ss to the state at time tt under the homogeneous dynamics.

###### Definition F.5 (Parameter-Space Propagator).

The propagator 𝒯​(t,s)\mathcal{T}(t,s) is the unique solution to

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂∂t​𝒯​(t,s)=−A​(t)​𝒯​(t,s),for ​t≥s,𝒯​(s,s)=Id.\displaystyle\begin{aligned} \frac{\partial}{\partial t}\mathcal{T}(t,s)&=-A(t)\mathcal{T}(t,s),\quad\text{for }t\geq s,\\ \mathcal{T}(s,s)&=I\_{d}.\end{aligned} |  | (139) |

The propagator satisfies the semigroup property 𝒯​(t,r)=𝒯​(t,s)​𝒯​(s,r)\mathcal{T}(t,r)=\mathcal{T}(t,s)\mathcal{T}(s,r) for any t≥s≥rt\geq s\geq r, and the inverse relation 𝒯​(t,s)−1=𝒯​(s,t)\mathcal{T}(t,s)^{-1}=\mathcal{T}(s,t).

The variation-of-constants formula then expresses δ​(t)\delta(t) as a convolution of past stresses with the propagator.

###### Theorem F.6 (Strain Propagation Integral).

The algorithmic strain δ​(t)\delta(t) admits the integral representation

|  |  |  |  |
| --- | --- | --- | --- |
|  | δ​(t)=−∫0t𝒯​(t,s)​f​(s)​𝑑s.\displaystyle\delta(t)=-\int\_{0}^{t}\mathcal{T}(t,s)f(s)\,ds. |  | (140) |

###### Proof.

Differentiating [Equation˜140](#A6.E140 "In Theorem F.6 (Strain Propagation Integral). ‣ F.2 The Kelvin–Voigt Constitutive Equation ‣ Appendix F Continuous-Time Kelvin–Voigt Framework ‣ The Origin of Edge of Stability") via the Leibniz rule:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | dd​t​δ​(t)\displaystyle\frac{d}{dt}\delta(t) | =dd​t​(−∫0t𝒯​(t,s)​f​(s)​𝑑s)\displaystyle=\frac{d}{dt}\left(-\int\_{0}^{t}\mathcal{T}(t,s)f(s)\,ds\right) |  | (141) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =−𝒯​(t,t)​f​(t)−∫0t∂∂t​𝒯​(t,s)​f​(s)​𝑑s.\displaystyle=-\mathcal{T}(t,t)f(t)-\int\_{0}^{t}\frac{\partial}{\partial t}\mathcal{T}(t,s)f(s)\,ds. |  | (142) |

The boundary term is −f​(t)-f(t). Substituting ∂t𝒯​(t,s)=−A​(t)​𝒯​(t,s)\partial\_{t}\mathcal{T}(t,s)=-A(t)\mathcal{T}(t,s) into the integral term:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | −∫0t∂∂t​𝒯​(t,s)​f​(s)​𝑑s\displaystyle-\int\_{0}^{t}\frac{\partial}{\partial t}\mathcal{T}(t,s)f(s)\,ds | =−∫0t(−A​(t)​𝒯​(t,s))​f​(s)​𝑑s\displaystyle=-\int\_{0}^{t}\bigl(-A(t)\mathcal{T}(t,s)\bigr)f(s)\,ds |  | (143) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =A​(t)​∫0t𝒯​(t,s)​f​(s)​𝑑s\displaystyle=A(t)\int\_{0}^{t}\mathcal{T}(t,s)f(s)\,ds |  | (144) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =−A​(t)​δ​(t).\displaystyle=-A(t)\delta(t). |  | (145) |

Combining gives δ˙=−f​(t)−A​(t)​δ​(t)\dot{\delta}=-f(t)-A(t)\delta(t), which is [Equation˜135](#A6.E135 "In Theorem F.3 (Kelvin–Voigt Variational Equation). ‣ F.2 The Kelvin–Voigt Constitutive Equation ‣ Appendix F Continuous-Time Kelvin–Voigt Framework ‣ The Origin of Edge of Stability"). The initial condition δ​(0)=0\delta(0)=0 follows from the vanishing domain at t=0t=0, and uniqueness from Picard–Lindelöf.
∎

### F.3 KV Stability: The Role of Curvature

With the strain integral in hand ([Theorem˜F.6](#A6.Thmtheorem6 "Theorem F.6 (Strain Propagation Integral). ‣ F.2 The Kelvin–Voigt Constitutive Equation ‣ Appendix F Continuous-Time Kelvin–Voigt Framework ‣ The Origin of Edge of Stability")), bounding ‖δ​(t)‖\|\delta(t)\| reduces to bounding the propagator norm ‖𝒯​(t,s)‖op\|\mathcal{T}(t,s)\|\_{\mathrm{op}}. A naive Grönwall bound would use ‖A​(τ)‖op\|A(\tau)\|\_{\mathrm{op}}, but this is unnecessarily pessimistic: it treats large positive eigenvalues as destabilizing, when in continuous time they cause faster contraction. The correct bound uses only the negative part of the spectrum. This is the fundamental difference between continuous and discrete time: in continuous time, instability requires negative curvature; in discrete time ([appendix˜E](#A5 "Appendix E Discrete-Time Kelvin–Voigt Dynamics ‣ The Origin of Edge of Stability")), curvature above 2/η2/\eta is equally destabilizing.

###### Definition F.7 (Instantaneous Negative Curvature).

The instantaneous negative curvature is

|  |  |  |  |
| --- | --- | --- | --- |
|  | α−​(t)≜max⁡{0,−λmin​(A​(t)+A​(t)⊤2)}.\displaystyle\alpha\_{-}(t)\triangleq\max\left\{0,\,-\lambda\_{\min}\left(\frac{A(t)+A(t)^{\top}}{2}\right)\right\}. |  | (146) |

When A​(t)A(t) is symmetric, this simplifies to α−​(t)=max⁡{0,−λmin​(A​(t))}\alpha\_{-}(t)=\max\{0,-\lambda\_{\min}(A(t))\}.

###### Theorem F.8 (Effective Stability Bound).

For any 0≤s≤t0\leq s\leq t, the operator norm of the propagator is controlled by the cumulative negative curvature:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖𝒯​(t,s)‖op≤exp⁡(∫stα−​(τ)​𝑑τ).\displaystyle\|\mathcal{T}(t,s)\|\_{\mathrm{op}}\leq\exp\left(\int\_{s}^{t}\alpha\_{-}(\tau)\,d\tau\right). |  | (147) |

Consequently, the algorithmic strain satisfies the bound

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖δ​(t)‖2≤∫0texp⁡(∫stα−​(τ)​𝑑τ)​‖f​(s)‖2​𝑑s.\displaystyle\|\delta(t)\|\_{2}\leq\int\_{0}^{t}\exp\left(\int\_{s}^{t}\alpha\_{-}(\tau)\,d\tau\right)\|f(s)\|\_{2}\,ds. |  | (148) |

In particular, if the loss landscape is locally convex along the trajectory (so that α−​(t)=0\alpha\_{-}(t)=0 for all tt), then the propagator is a contraction and ‖δ​(t)‖2≤∫0t‖f​(s)‖2​𝑑s\|\delta(t)\|\_{2}\leq\int\_{0}^{t}\|f(s)\|\_{2}\,ds.

###### Proof.

Fix a unit vector vv and set u​(τ)=𝒯​(τ,s)​vu(\tau)=\mathcal{T}(\tau,s)v, ψ​(τ)=‖u​(τ)‖22\psi(\tau)=\|u(\tau)\|\_{2}^{2}. The propagator equation ∂τu=−A​(τ)​u\partial\_{\tau}u=-A(\tau)u gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | d​ψd​τ=−2​u​(τ)⊤​A​(τ)​u​(τ).\displaystyle\frac{d\psi}{d\tau}=-2\,u(\tau)^{\top}A(\tau)\,u(\tau). |  | (149) |

Writing A=H+SA=H+S with HH symmetric and SS skew-symmetric, the skew part drops out of the quadratic form, so

|  |  |  |  |
| --- | --- | --- | --- |
|  | u⊤​A​u=u⊤​H​u≥λmin​(H)​‖u‖22,\displaystyle u^{\top}A\,u=u^{\top}H\,u\geq\lambda\_{\min}(H)\,\|u\|\_{2}^{2}, |  | (150) |

and therefore ψ˙≤2​α−​(τ)​ψ​(τ)\dot{\psi}\leq 2\alpha\_{-}(\tau)\,\psi(\tau). Grönwall’s inequality with ψ​(s)=1\psi(s)=1 gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | ψ​(t)≤exp⁡(2​∫stα−​(τ)​𝑑τ).\displaystyle\psi(t)\leq\exp\!\Bigl(2\int\_{s}^{t}\alpha\_{-}(\tau)\,d\tau\Bigr). |  | (151) |

Taking square roots and supremizing over unit vv yields [Equation˜147](#A6.E147 "In Theorem F.8 (Effective Stability Bound). ‣ F.3 KV Stability: The Role of Curvature ‣ Appendix F Continuous-Time Kelvin–Voigt Framework ‣ The Origin of Edge of Stability"). The strain bound follows from the triangle inequality applied to [Equation˜140](#A6.E140 "In Theorem F.6 (Strain Propagation Integral). ‣ F.2 The Kelvin–Voigt Constitutive Equation ‣ Appendix F Continuous-Time Kelvin–Voigt Framework ‣ The Origin of Edge of Stability"):

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖δ​(t)‖2≤∫0t‖𝒯​(t,s)‖op​‖f​(s)‖2​𝑑s≤∫0texp⁡(∫stα−​(τ)​𝑑τ)​‖f​(s)‖2​𝑑s.\displaystyle\|\delta(t)\|\_{2}\leq\int\_{0}^{t}\|\mathcal{T}(t,s)\|\_{\mathrm{op}}\,\|f(s)\|\_{2}\,ds\leq\int\_{0}^{t}\exp\!\Bigl(\int\_{s}^{t}\alpha\_{-}(\tau)\,d\tau\Bigr)\|f(s)\|\_{2}\,ds. |  | (152) |

∎

In convex regions (A​(t)⪰0A(t)\succeq 0), the propagator is a contraction and instability arises only from non-convex portions of the trajectory. This is where continuous and discrete time diverge. In the discrete setting of [appendix˜E](#A5 "Appendix E Discrete-Time Kelvin–Voigt Dynamics ‣ The Origin of Edge of Stability"), explicit time-stepping introduces a finite stability window [0,2/η][0,2/\eta]: eigenvalues above 2/η2/\eta cause oscillatory instability even though they are positive. It is this upper boundary that gives rise to the Edge of Stability.

## Appendix G Transverse edge normal form for two-layer linear networks

This appendix proves [Proposition˜2.6](#S2.Thmtheorem6 "Proposition 2.6 (Transverse edge normal form for two-layer linear networks). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability") by computing 𝒫\mathcal{P} for the two-layer linear network loss

|  |  |  |  |
| --- | --- | --- | --- |
|  | Lh​(W1,W2)=12​‖W2​W1−M‖F2\displaystyle L\_{h}(W\_{1},W\_{2})=\tfrac{1}{2}\|W\_{2}W\_{1}-M\|\_{F}^{2} |  | (153) |

at a balanced global minimizer. Because the network is linear, all derivatives of LhL\_{h} can be computed in closed form, which makes this a concrete test case for the general bifurcation theory of [Theorems˜2.3](#S2.Thmtheorem3 "Theorem 2.3 (Center reduction and the edge eigenproblem). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability") and [2.5](#S2.Thmtheorem5 "Corollary 2.5 (Generic branching at the edge). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability").

The argument proceeds in four steps. First, we identify a canonical minimizer using the SVD of the target matrix MM. Second, we compute the Hessian kernel, which consists of reparametrization symmetries that do not change the product W2​W1W\_{2}W\_{1}. Third, we restrict the edge coupling theory to the orthogonal complement of this kernel (the normal space 𝒩\mathcal{N}), and show that this restriction is independent of the hidden width hh. Fourth, we evaluate the branch form 𝒬\mathcal{Q} along the leading eigenvector and find that it is negative, giving a supercritical bifurcation.

#### Canonical balanced minimizer.

The starting point is the SVD of the target matrix. Let

|  |  |  |  |
| --- | --- | --- | --- |
|  | M=Ur​Σ​Vr⊤,Σ=diag⁡(σ1,…,σr),σ1≥⋯≥σr>0.\displaystyle M=U\_{r}\Sigma V\_{r}^{\top},\qquad\Sigma=\operatorname{diag}(\sigma\_{1},\dots,\sigma\_{r}),\qquad\sigma\_{1}\geq\cdots\geq\sigma\_{r}>0. |  | (154) |

Every balanced global minimizer has the form

|  |  |  |  |
| --- | --- | --- | --- |
|  | W¯2=Ur​Σ1/2​R⊤,W¯1=R​Σ1/2​Vr⊤,R∈ℝh×r,R⊤​R=Ir.\displaystyle\bar{W}\_{2}=U\_{r}\Sigma^{1/2}R^{\top},\qquad\bar{W}\_{1}=R\Sigma^{1/2}V\_{r}^{\top},\qquad R\in\mathbb{R}^{h\times r},\quad R^{\top}R=I\_{r}. |  | (155) |

Since the loss and the Frobenius metric are invariant under orthogonal changes of output, input, and hidden coordinates, we may choose coordinates so that

|  |  |  |  |
| --- | --- | --- | --- |
|  | M=(Σ000),W¯1(h)=(Σ1/2000),W¯2(h)=(Σ1/2000).\displaystyle M=\begin{pmatrix}\Sigma&0\\ 0&0\end{pmatrix},\qquad\bar{W}\_{1}^{(h)}=\begin{pmatrix}\Sigma^{1/2}&0\\ 0&0\end{pmatrix},\qquad\bar{W}\_{2}^{(h)}=\begin{pmatrix}\Sigma^{1/2}&0\\ 0&0\end{pmatrix}. |  | (156) |

#### Kernel and normal slice.

The Hessian at w¯\bar{w} has a nontrivial kernel because many different factorizations W2​W1=MW\_{2}W\_{1}=M achieve the same product and hence the same loss. These are the reparametrization symmetries of the network. The center-reduction [Theorem˜2.3](#S2.Thmtheorem3 "Theorem 2.3 (Center reduction and the edge eigenproblem). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability") requires a nondegenerate Hessian, so we must identify the kernel 𝒦\mathcal{K} and restrict to its orthogonal complement 𝒩\mathcal{N}, where the curvature is nonzero and the bifurcation analysis applies. Write perturbations as

|  |  |  |  |
| --- | --- | --- | --- |
|  | δ​W1=(ABCD),δ​W2=(EFGH),\displaystyle\delta W\_{1}=\begin{pmatrix}A&B\\ C&D\end{pmatrix},\qquad\delta W\_{2}=\begin{pmatrix}E&F\\ G&H\end{pmatrix}, |  | (157) |

where A∈ℝr×rA\in\mathbb{R}^{r\times r}, B∈ℝr×(d−r)B\in\mathbb{R}^{r\times(d-r)}, etc., matching the block structure of W¯1(h)\bar{W}\_{1}^{(h)}. Define the product map Fh​(W1,W2)≜W2​W1F\_{h}(W\_{1},W\_{2})\triangleq W\_{2}W\_{1}. At a global minimizer the residual vanishes, so the Hessian factors as

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇2Lh​(w¯)=D​Fh​(w¯)∗​D​Fh​(w¯).\displaystyle\nabla^{2}L\_{h}(\bar{w})=DF\_{h}(\bar{w})^{\ast}DF\_{h}(\bar{w}). |  | (158) |

A direct computation gives the linearization and its kernel:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | D​Fh​(w¯)​[δ​W1,δ​W2]\displaystyle DF\_{h}(\bar{w})[\delta W\_{1},\delta W\_{2}] | =W¯2​δ​W1+δ​W2​W¯1=(Σ1/2​A+E​Σ1/2Σ1/2​BG​Σ1/20),\displaystyle=\bar{W}\_{2}\delta W\_{1}+\delta W\_{2}\bar{W}\_{1}=\begin{pmatrix}\Sigma^{1/2}A+E\Sigma^{1/2}&\Sigma^{1/2}B\\ G\Sigma^{1/2}&0\end{pmatrix}, |  | (159) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒦\displaystyle\mathcal{K} | =ker​∇2Lh​(w¯)=ker⁡D​Fh​(w¯)\displaystyle=\ker\nabla^{2}L\_{h}(\bar{w})=\ker DF\_{h}(\bar{w}) |  | (160) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ={B=0,G=0,Σ1/2​A+E​Σ1/2=0,C,D,F,H​arbitrary}.\displaystyle=\left\{\begin{array}[]{l}B=0,\quad G=0,\quad\Sigma^{1/2}A+E\Sigma^{1/2}=0,\\[1.29167pt] C,D,F,H\ \text{arbitrary}\end{array}\right\}. |  | (163) |

The image of the linearization determines what directions in the output space the factorization can explore:

|  |  |  |  |
| --- | --- | --- | --- |
|  | im⁡D​Fh​(w¯)={(ZXY0):Z∈ℝr×r,X∈ℝr×(d−r),Y∈ℝ(p−r)×r},\displaystyle\operatorname{im}DF\_{h}(\bar{w})=\left\{\begin{pmatrix}Z&X\\ Y&0\end{pmatrix}:Z\in\mathbb{R}^{r\times r},\ X\in\mathbb{R}^{r\times(d-r)},\ Y\in\mathbb{R}^{(p-r)\times r}\right\}, |  | (164) |

which is the tangent space TM​ℛrT\_{M}\mathcal{R}\_{r} of the rank-rr matrix manifold at MM. Therefore D​Fh​(w¯)DF\_{h}(\bar{w}) has rank r​(p+d−r)r(p+d-r), so by the constant-rank theorem the minimum set

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℳh​(M)≜{(W1,W2):W2​W1=M}\displaystyle\mathcal{M}\_{h}(M)\triangleq\{(W\_{1},W\_{2}):W\_{2}W\_{1}=M\} |  | (165) |

is a smooth manifold near w¯\bar{w} with tangent space Tw¯​ℳh​(M)=𝒦T\_{\bar{w}}\mathcal{M}\_{h}(M)=\mathcal{K}. Thus w¯\bar{w} is Morse–Bott.

The kernel 𝒦\mathcal{K} consists of flat directions that do not affect the loss. To apply the edge coupling theory, we restrict to the orthogonal complement 𝒩=𝒦⟂\mathcal{N}=\mathcal{K}^{\perp}, which captures the directions with nonzero curvature. Define

|  |  |  |  |
| --- | --- | --- | --- |
|  | T​(A,E)≜Σ1/2​A+E​Σ1/2.\displaystyle T(A,E)\triangleq\Sigma^{1/2}A+E\Sigma^{1/2}. |  | (166) |

Its adjoint is

|  |  |  |  |
| --- | --- | --- | --- |
|  | T∗​(Y)=(Σ1/2​Y,Y​Σ1/2),\displaystyle T^{\ast}(Y)=\bigl(\Sigma^{1/2}Y,\;Y\Sigma^{1/2}\bigr), |  | (167) |

so

|  |  |  |  |
| --- | --- | --- | --- |
|  | (ker⁡T)⟂=ran⁡T∗={(Σ1/2​Y,Y​Σ1/2):Y∈ℝr×r}.\displaystyle(\ker T)^{\perp}=\operatorname{ran}T^{\ast}=\left\{(\Sigma^{1/2}Y,\;Y\Sigma^{1/2}):Y\in\mathbb{R}^{r\times r}\right\}. |  | (168) |

Combining the (ker⁡T)⟂(\ker T)^{\perp} constraint on the (A,E)(A,E)-block with unconstrained BB and GG blocks (and the zero blocks from the kernel conditions C=D=F=H=0C=D=F=H=0) gives the full normal space:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒩={δ​W1=(Σ1/2​YB00),δ​W2=(Y​Σ1/20G0):Y,B,G}.\displaystyle\mathcal{N}=\left\{\delta W\_{1}=\begin{pmatrix}\Sigma^{1/2}Y&B\\ 0&0\end{pmatrix},\qquad\delta W\_{2}=\begin{pmatrix}Y\Sigma^{1/2}&0\\ G&0\end{pmatrix}:Y,B,G\right\}. |  | (169) |

#### Exact width-invariance of the restricted loss.

The central observation is that the restricted loss on 𝒩\mathcal{N} is independent of the hidden width hh. Increasing hh beyond the rank rr adds only flat directions in 𝒦\mathcal{K}, so the loss restricted to 𝒩\mathcal{N} is the same for any h≥rh\geq r. We now establish this by constructing an explicit isometry between the normal slices at different widths. For a=(Y,B,G)a=(Y,B,G), define

|  |  |  |  |
| --- | --- | --- | --- |
|  | Th​(a)≜((Σ1/2​YB00),(Y​Σ1/20G0))∈𝒩.\displaystyle T\_{h}(a)\triangleq\left(\begin{pmatrix}\Sigma^{1/2}Y&B\\ 0&0\end{pmatrix},\begin{pmatrix}Y\Sigma^{1/2}&0\\ G&0\end{pmatrix}\right)\in\mathcal{N}. |  | (170) |

The linearization and quadratic residual along this normal slice are

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Ph​(a)\displaystyle P\_{h}(a) | ≜D​Fh​(w¯)​Th​(a)=(Σ​Y+Y​ΣΣ1/2​BG​Σ1/20),\displaystyle\triangleq DF\_{h}(\bar{w})T\_{h}(a)=\begin{pmatrix}\Sigma Y+Y\Sigma&\Sigma^{1/2}B\\ G\Sigma^{1/2}&0\end{pmatrix}, |  | (171) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Qh​(a)\displaystyle Q\_{h}(a) | ≜δ​W2​δ​W1=(Y​Σ​YY​Σ1/2​BG​Σ1/2​YG​B),\displaystyle\triangleq\delta W\_{2}\,\delta W\_{1}=\begin{pmatrix}Y\Sigma Y&Y\Sigma^{1/2}B\\ G\Sigma^{1/2}Y&GB\end{pmatrix}, |  | (172) |

so the restricted loss takes the form

|  |  |  |  |
| --- | --- | --- | --- |
|  | Lh​(w¯+Th​(a))=12​‖Ph​(a)+Qh​(a)‖F2.\displaystyle L\_{h}(\bar{w}+T\_{h}(a))=\frac{1}{2}\|P\_{h}(a)+Q\_{h}(a)\|\_{F}^{2}. |  | (173) |

The key point is that neither Ph​(a)P\_{h}(a) nor Qh​(a)Q\_{h}(a) depends on hh: the extra rows and columns added by increasing the hidden width are all zero. To make this formal, consider the minimal-width model (h=rh=r) at the balanced minimizer

|  |  |  |  |
| --- | --- | --- | --- |
|  | W¯1(r)=(Σ1/2​ 0),W¯2(r)=(Σ1/20),\displaystyle\bar{W}\_{1}^{(r)}=(\Sigma^{1/2}\ \ 0),\qquad\bar{W}\_{2}^{(r)}=\binom{\Sigma^{1/2}}{0}, |  | (174) |

with normal-slice embedding

|  |  |  |  |
| --- | --- | --- | --- |
|  | Tr​(a)≜((Σ1/2​YB),(Y​Σ1/2G)).\displaystyle T\_{r}(a)\triangleq\left((\Sigma^{1/2}Y\ \ B),\binom{Y\Sigma^{1/2}}{G}\right). |  | (175) |

The restricted loss at width rr produces the same PhP\_{h} and QhQ\_{h}, so

|  |  |  |  |
| --- | --- | --- | --- |
|  | Lr​(w¯(r)+Tr​(a))=12​‖Ph​(a)+Qh​(a)‖F2=Lh​(w¯+Th​(a)).\displaystyle L\_{r}(\bar{w}^{(r)}+T\_{r}(a))=\frac{1}{2}\|P\_{h}(a)+Q\_{h}(a)\|\_{F}^{2}=L\_{h}(\bar{w}+T\_{h}(a)). |  | (176) |

The zero-padding map Zh≜Th​Tr−1:𝒩(r)→𝒩(h)Z\_{h}\triangleq T\_{h}T\_{r}^{-1}:\mathcal{N}^{(r)}\to\mathcal{N}^{(h)} is therefore an isometric isomorphism, and

|  |  |  |  |
| --- | --- | --- | --- |
|  | Lh​(w¯(h)+Zh​ξ)=Lr​(w¯(r)+ξ)(ξ∈𝒩(r)).\displaystyle L\_{h}(\bar{w}^{(h)}+Z\_{h}\xi)=L\_{r}(\bar{w}^{(r)}+\xi)\qquad(\xi\in\mathcal{N}^{(r)}). |  | (177) |

The restricted losses coincide, and so do all their derivatives. Width-invariance of 𝒫\mathcal{P} and its quartic term follows.

#### Transverse spectrum.

To determine the critical learning rate at which period-doubling first occurs, we need to know which eigenvalue of the restricted Hessian is largest. Since the residual vanishes at the minimizer, the Hessian factors as ∇2Lh​(w¯)=D​Fh​(w¯)∗​D​Fh​(w¯)\nabla^{2}L\_{h}(\bar{w})=DF\_{h}(\bar{w})^{\ast}DF\_{h}(\bar{w}). The transverse Hessian quadratic form then decomposes into three orthogonal blocks corresponding to the YY, BB, and GG components of the normal space:

|  |  |  |  |
| --- | --- | --- | --- |
|  | q​(Y,B,G)=‖Σ​Y+Y​Σ‖F2+‖Σ1/2​B‖F2+‖G​Σ1/2‖F2.\displaystyle q(Y,B,G)=\|\Sigma Y+Y\Sigma\|\_{F}^{2}+\|\Sigma^{1/2}B\|\_{F}^{2}+\|G\Sigma^{1/2}\|\_{F}^{2}. |  | (178) |

Evaluating on the matrix units Ei​jE\_{ij} in the YY-block,
Ei​βE\_{i\beta} in the BB-block, and Eα​jE\_{\alpha j} in the GG-block gives the eigenvalues and full spectrum:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | λ​(Ei​j)\displaystyle\lambda(E\_{ij}) | =σi+σj,λ​(Ei​β)=σi,λ​(Eα​j)=σj,\displaystyle=\sigma\_{i}+\sigma\_{j},\qquad\lambda(E\_{i\beta})=\sigma\_{i},\qquad\lambda(E\_{\alpha j})=\sigma\_{j}, |  | (179) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | spec⁡(H⟂)\displaystyle\operatorname{spec}(H^{\perp}) | ={σi+σj:1≤i,j≤r}∪{σi:1≤i≤r}×(p+d−2​r),\displaystyle=\{\sigma\_{i}+\sigma\_{j}:1\leq i,j\leq r\}\cup\{\sigma\_{i}:1\leq i\leq r\}^{\times(p+d-2r)}, |  | (180) |

independent of hh.

#### Exact sharp-mode normal form.

As η\eta increases from zero, the reduced Hessian H⟂−(2/η)​IH^{\perp}-(2/\eta)I first becomes singular when 2/η2/\eta hits the largest eigenvalue 2​σ12\sigma\_{1}, which occurs at ηc=1/σ1\eta\_{c}=1/\sigma\_{1}. This is the critical learning rate for the first period-doubling bifurcation. Assuming σ1>σ2\sigma\_{1}>\sigma\_{2} so that this eigenvalue is simple, the corresponding unit eigenvector is

|  |  |  |  |
| --- | --- | --- | --- |
|  | uc=Th​(E112​σ1, 0, 0),\displaystyle u\_{c}=T\_{h}\!\left(\frac{E\_{11}}{\sqrt{2\sigma\_{1}}},\,0,\,0\right), |  | (181) |

which corresponds to the perturbation δ​W1=δ​W2=12​E11\delta W\_{1}=\delta W\_{2}=\frac{1}{\sqrt{2}}E\_{11} on the active block with zeros elsewhere. Restricting the loss to this line and expanding gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | Lh​(w¯+t​uc)=12​(2​σ1​t+12​t2)2=Lh​(w¯)+σ1​t2+σ12​t3+18​t4.\displaystyle L\_{h}(\bar{w}+tu\_{c})=\frac{1}{2}\Bigl(\sqrt{2\sigma\_{1}}\,t+\frac{1}{2}t^{2}\Bigr)^{2}=L\_{h}(\bar{w})+\sigma\_{1}t^{2}+\frac{\sqrt{\sigma\_{1}}}{\sqrt{2}}t^{3}+\frac{1}{8}t^{4}. |  | (182) |

Reading off the third and fourth derivatives and applying [Proposition˜2.4](#S2.Thmtheorem4 "Proposition 2.4 (Quartic expansion near a fixed point). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability") to this one-dimensional restricted loss yields the quartic term:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒬⟂​(uc)=36−(3​2​σ1)22​(2​σ1)=−4.\displaystyle\mathcal{Q}^{\perp}(u\_{c})=\frac{3}{6}-\frac{(3\sqrt{2\sigma\_{1}})^{2}}{2(2\sigma\_{1})}=-4. |  | (183) |

Substituting into the reduced functional gives

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Φη⟂​(t​uc)\displaystyle\Phi\_{\eta}^{\perp}(tu\_{c}) | =Lh​(w¯)+12​(2​σ1−2η)​t2+14​𝒬⟂​(uc)​t4+o​(t4)\displaystyle=L\_{h}(\bar{w})+\frac{1}{2}\Bigl(2\sigma\_{1}-\frac{2}{\eta}\Bigr)t^{2}+\frac{1}{4}\,\mathcal{Q}^{\perp}(u\_{c})\,t^{4}+o(t^{4}) |  | (184) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =Lh​(w¯)+(σ1−1η)​t2−t4+o​(t4).\displaystyle=L\_{h}(\bar{w})+\Bigl(\sigma\_{1}-\frac{1}{\eta}\Bigr)t^{2}-t^{4}+o(t^{4}). |  | (185) |

The quartic coefficient is −1<0-1<0, so by [Corollary˜2.5](#S2.Thmtheorem5 "Corollary 2.5 (Generic branching at the edge). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability") the period-two orbit exists for η>ηc=1/σ1\eta>\eta\_{c}=1/\sigma\_{1} and grows continuously from zero amplitude at ηc\eta\_{c}.
∎

#### Experimental validation.

[Figure 2](#S2.F2 "Figure 2 ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability") validates the prediction for a two-layer linear network (p=5p=5, h=3h=3, d=10d=10, rank-3 target, n=200n=200 samples). At the global minimizer w¯\bar{w}, the computed quartic term satisfies 𝒬⟂​(uc)<0\mathcal{Q}^{\perp}(u\_{c})<0, predicting that the period-doubling branch appears for η>ηc\eta>\eta\_{c} and emerges continuously from zero at ηc\eta\_{c}. Full-batch GD at learning rates η\eta near ηc=1/σ1\eta\_{c}=1/\sigma\_{1} confirms this: the oscillation amplitude grows continuously from zero for η>ηc\eta>\eta\_{c} and tracks the η−ηc\sqrt{\eta-\eta\_{c}} scaling of [Corollary˜2.5](#S2.Thmtheorem5 "Corollary 2.5 (Generic branching at the edge). ‣ 2 The Edge Coupling ‣ The Origin of Edge of Stability"), with oscillations concentrated along ucu\_{c}.

[◄](/html/2604.20445)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2604.20446)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2604.20446)
[View original  
on arXiv](https://arxiv.org/abs/2604.20446)[►](/html/2604.20447)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Tue May 5 22:41:55 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
