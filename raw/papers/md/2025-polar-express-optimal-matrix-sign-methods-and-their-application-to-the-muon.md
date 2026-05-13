---
arxiv: '2505.16932'
authors:
- Noah Amsel
- David Persson
- Christopher Musco
- Robert M. Gower
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'The Polar Express: Optimal Matrix Sign Methods and Their Application to the
  Muon Algorithm'
url: https://arxiv.org/abs/2505.16932
year: 2025
---

[2505.16932] The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm














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



# The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm

Noah Amsel
New York University. noah.amsel@nyu.edu
  
David Persson
New York University and Flatiron Institute. dup210@nyu.edu, dpersson@flatironinstitute.org
  
Christopher Musco
New York University. cmusco@nyu.edu
  
Robert M. Gower
Flatiron Institute. rgower@flatironinstitute.org

###### Abstract

Computing the polar decomposition and the related matrix sign function has been a well-studied problem in numerical analysis for decades. Recently, it has emerged as an important subroutine within the Muon algorithm for training deep neural networks.
However, the requirements of this application differ sharply from classical settings: deep learning demands GPU-friendly algorithms that prioritize high throughput over high precision. We introduce Polar Express, a new method for computing the polar decomposition. Like Newton–Schulz and other classical polynomial methods, our approach uses only matrix-matrix multiplications, making it
very efficient on GPUs.
Inspired by earlier work of Chen & Chow and Nakatsukasa & Freund, Polar Express adapts the update rule at each iteration by solving a minimax optimization problem.
We prove that this strategy minimizes error in a worst-case sense, allowing Polar Express to converge as rapidly as possible both in the early iterations and asymptotically.
We also address finite-precision issues, making it practical to use in bfloat16. When integrated into the Muon training framework, our method leads to consistent improvements in validation loss when training a GPT-2 model on one billion tokens from the FineWeb dataset, outperforming recent alternatives across a range of learning rates.

###### Contents

1. [1 Introduction](#S1 "In The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")
   1. [1.1 The Muon Method](#S1.SS1 "In 1 Introduction ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")
   2. [1.2 Computing the Polar Factor](#S1.SS2 "In 1 Introduction ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")
   3. [1.3 Contributions](#S1.SS3 "In 1 Introduction ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")
2. [2 Related Work](#S2 "In The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")
3. [3 Approximations by Compositions of Polynomials](#S3 "In The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")
4. [4 The Polar Express](#S4 "In The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")
   1. [4.1 Greedy is optimal](#S4.SS1 "In 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")
   2. [4.2 Finding the optimal polynomial for each iteration](#S4.SS2 "In 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")
   3. [4.3 Upper and lower bounds on the singular values](#S4.SS3 "In 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")
   4. [4.4 Finite precision considerations](#S4.SS4 "In 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")
   5. [4.5 The algorithm](#S4.SS5 "In 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")
5. [5 Numerical Experiments](#S5 "In The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")
   1. [5.1 Convergence of Polar Express](#S5.SS1 "In 5 Numerical Experiments ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")
   2. [5.2 Training GPT-2](#S5.SS2 "In 5 Numerical Experiments ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")
6. [A Proof of Theorem 4.1](#A1 "In The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")
7. [B Proof of Theorem 4.3](#A2 "In The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")
8. [C Proof of equivalence between (6) and (7)](#A3 "In The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")
9. [D Remez algorithm](#A4 "In The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")
10. [E Initialization for Matrices with Large Spectral Gaps](#A5 "In The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")
11. [F Fast Polynomial Iteration for Rectangular Matrices](#A6 "In The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")
12. [G Code for Constructing Polynomials of Polar Express](#A7 "In The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")

## 1  Introduction

Advanced linear algebra is making its way into deep learning. Efficient algorithms for computing *matrix functions* have found exciting new applications in training neural networks. In particular, approximations to the matrix-inverse are used in the full Adagrad method [[13](#bib.bib13)], the matrix square-root and quarter-root appear as subroutines in the Shampoo and Soap optimizers [[17](#bib.bib17), [43](#bib.bib43), [47](#bib.bib47)], and most recently, the matrix sign function has become a key ingredient of the Muon optimizer [[5](#bib.bib5), [4](#bib.bib4), [22](#bib.bib22)].

While the problem of computing these matrix functions has been studied by numerical analysts for decades, applications in deep learning come with different requirements than those in computational science. For deep learning, it is critical to take maximum advantage of GPU-friendly operations like matrix-matrix products and to avoid less parallel operations. Moreover, memory overhead must be small to handle large models. On the other hand, high accuracy is typically less important; the gold standard of sixteen digits of accuracy is overkill in deep learning.

Given these considerations, there is a need to develop new matrix function methods that are tailor-made for deep learning applications.
We take on this challenge by designing a state-of-the-art, GPU-friendly algorithm for computing the matrix sign function, or more generally, for computing the *polar decomposition* of a rectangular matrix. We apply our new Polar Express method ([Algorithm 1](#alg1 "In 1 Introduction ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")) to compute the descent direction in the increasingly popular Muon optimizer.
In [Figure 1](#S1.F1 "In 1 Introduction ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"), we show that using Polar Express within Muon consistently results in lower validation loss across all learning rates when training a GPT-2 model, as compared to other matrix sign methods [[9](#bib.bib9), [44](#bib.bib44), [22](#bib.bib22)].

Algorithm 1  Python code for the Polar Express of degree = 5.

[⬇](data:text/plain;base64,ZnJvbSBpdGVydG9vbHMgaW1wb3J0IHJlcGVhdAppbXBvcnQgdG9yY2gKCmNvZWZmc19saXN0ID0gWwogICAgKDguMjg3MjEyMDE4MTQ1NjMsIC0yMy41OTU4ODY1MTkwOTg4MzcsIDE3LjMwMDM4NzMxMjUzMDkzMyksCiAgICAoNC4xMDcwNTkxMTE1NDIyMDMsIC0yLjk0Nzg0OTkxNjczNzkxMDYsIDAuNTQ0ODQzMTA4MjkyNjYwMSksCiAgICAoMy45NDg2OTA4NTM0ODIyOTQ2LCAtMi45MDg5MDIxMTU5NjI5NDksIDAuNTUxODE5MTM5NDM3MDEzNyksCiAgICAoMy4zMTg0MTk2NTczNzA2MDE1LCAtMi40ODg0ODgwMjQzMTQ4NzQsIDAuNTEwMDQ4OTQwMTIzNzIpLAogICAgKDIuMzAwNjUyMDE5OTU0ODE3LCAtMS42Njg5MDM5ODQ1NzQ3NDkzLCAwLjQxODgwNzMxMTk1MjU2NzMpLAogICAgKDEuODkxMzAxNDA3Nzg3Mzk4LCAtMS4yNjc5OTU4MjcxOTQ1ODY4LCAwLjM3NjgwNDA4OTQ4NTI0ODM1KSwKICAgICgxLjg3NTAwMTQ4MDg1MzQ0NzksIC0xLjI1MDAwMTY0NTM5OTk0ODcsIDAuMzc1MDAwMTY0NTQ3NDI0OCksCiAgICAoMS44NzUsIC0xLjI1LCAwLjM3NSksICAjIHN1YnNlcXVlbnQgY29lZmZzIGVxdWFsIHRoaXMgbnVtZXJpY2FsbHkKXQojIHNhZmV0eSBmYWN0b3IgZm9yIG51bWVyaWNhbCBzdGFiaWxpdHkgKGJ1dCBleGNsdWRlIGxhc3QgcG9seW5vbWlhbCkKY29lZmZzX2xpc3QgPSBbKGEgLyAxLjAxLCBiIC8gMS4wMSoqMywgYyAvIDEuMDEqKjUpCiAgICAgICAgICAgICAgICBmb3IgKGEsIGIsIGMpIGluIGNvZWZmc19saXN0WzotMV1dICsgW2NvZWZmc19saXN0Wy0xXV0KCkB0b3JjaC5jb21waWxlCmRlZiBQb2xhckV4cHJlc3MoRzogdG9yY2guVGVuc29yLCBzdGVwczogaW50KSAtPiB0b3JjaC5UZW5zb3I6CiAgICBhc3NlcnQgRy5uZGltID49IDIKICAgIFggPSBHLmJmbG9hdDE2KCkgICMgZm9yIHNwZWVkCiAgICBpZiBHLnNpemUoLTIpID4gRy5zaXplKC0xKTogWCA9IFgubVQgICMgdGhpcyByZWR1Y2VzIEZMT1BzCiAgICBYID0gWCAvIChYLm5vcm0oZGltPSgtMiwgLTEpLCBrZWVwZGltPVRydWUpICogMS4wMSkKICAgIGhzID0gY29lZmZzX2xpc3RbOnN0ZXBzXSArIGxpc3QoCiAgICAgICAgcmVwZWF0KGNvZWZmc19saXN0Wy0xXSwgc3RlcHMgLSBsZW4oY29lZmZzX2xpc3QpKSkKICAgIGZvciBhLCBiLCBjIGluIGhzOgogICAgICAgIEEgPSBYIEAgWC5tVAogICAgICAgIEIgPSBiICogQSArIGMgKiBBIEAgQQogICAgICAgIFggPSBhICogWCArIEIgQCBYICAjIFggPC0gYVggKyBiWF4zICsgY1heNQogICAgaWYgRy5zaXplKC0yKSA+IEcuc2l6ZSgtMSk6IFggPSBYLm1UCiAgICByZXR1cm4gWA==)

from itertools import repeat

import torch

coeffs\_list = [

(8.28721201814563, -23.595886519098837, 17.300387312530933),

(4.107059111542203, -2.9478499167379106, 0.5448431082926601),

(3.9486908534822946, -2.908902115962949, 0.5518191394370137),

(3.3184196573706015, -2.488488024314874, 0.51004894012372),

(2.300652019954817, -1.6689039845747493, 0.4188073119525673),

(1.891301407787398, -1.2679958271945868, 0.37680408948524835),

(1.8750014808534479, -1.2500016453999487, 0.3750001645474248),

(1.875, -1.25, 0.375), # subsequent coeffs equal this numerically

]

# safety factor for numerical stability (but exclude last polynomial)

coeffs\_list = [(a / 1.01, b / 1.01\*\*3, c / 1.01\*\*5)

for (a, b, c) in coeffs\_list[:-1]] + [coeffs\_list[-1]]

@torch.compile

def PolarExpress(G: torch.Tensor, steps: int) -> torch.Tensor:

assert G.ndim >= 2

X = G.bfloat16() # for speed

if G.size(-2) > G.size(-1): X = X.mT # this reduces FLOPs

X = X / (X.norm(dim=(-2, -1), keepdim=True) \* 1.01)

hs = coeffs\_list[:steps] + list(

repeat(coeffs\_list[-1], steps - len(coeffs\_list)))

for a, b, c in hs:

A = X @ X.mT

B = b \* A + c \* A @ A

X = a \* X + B @ X # X <- aX + bX^3 + cX^5

if G.size(-2) > G.size(-1): X = X.mT

return X



![Refer to caption](/html/2505.16932/assets/x1.png)

![Refer to caption](/html/2505.16932/assets/x2.png)

Figure 1: Training a GPT-2-Large
model (774M params) on 1 billion tokens from the FineWeb dataset [[2](#bib.bib2)]. The label muon-<name> refers to implementing Muon using <name> to compute the polar factor. Left: final validation loss across learning rates. Right: validation loss across epochs using the best learning rate. The best learning rate (l​rlr) and final validation loss for each method was adamw (l​r=0.0001)(lr=0.0001): 4.1724.172, muon-You (l​r=0.02)(lr=0.02): 3.4003.400, muon-Jordan (l​r=0.02)(lr=0.02): 3.3983.398 and muon-PolarExp (l​r=0.02)(lr=0.02): 3.3403.340.

### 1.1  The Muon Method

The Muon optimizer has recently gained popularity for training large language models, often outperforming state-of-the-art adaptive gradient methods like Adam and AdamW [[26](#bib.bib26), [31](#bib.bib31)]. Muon has been used to set records for the NanoGPT speedrun [[22](#bib.bib22)] and to expand the Pareto frontier of performance versus training FLOPs for large language models [[30](#bib.bib30), [42](#bib.bib42)].

The Muon update rule [[5](#bib.bib5)] is defined as follows.
Let λ,β>0\lambda,\beta>0 be the learning rate and momentum coefficient hyperparameters. (By default, β=0.9\beta=0.9.)
Let 𝑾t∈ℝm×n\bm{W}\_{t}\in\mathbb{R}^{m\times n} be the weight matrix of a given neural network layer at iteration tt, and let 𝑮t∈ℝm×n\bm{G}\_{t}\in\mathbb{R}^{m\times n} be its (stochastic) gradient.
Let 𝑴t∈ℝm×n\bm{M}\_{t}\in\mathbb{R}^{m\times n} be the running momentum estimate of the gradient, where 𝑴0=𝟎\bm{M}\_{0}=\bm{0}.
The Muon update is given by

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝑴t\displaystyle\bm{M}\_{t} | =β​𝑴t−1+(1−β)​𝑮t\displaystyle=\beta\bm{M}\_{t-1}+(1-\beta)\bm{G}\_{t} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝑾t+1\displaystyle\bm{W}\_{t+1} | =𝑾t−λ​polar(𝑴t).\displaystyle=\bm{W}\_{t}-\lambda\operatorname\*{polar}(\bm{M}\_{t}). |  |

Whereas standard stochastic gradient descent (SGD) with momentum updates the weight matrix by taking a step in the direction −𝑴t-\bm{M}\_{t}, the Muon method steps in the direction −polar(𝑴t)-\operatorname\*{polar}(\bm{M}\_{t}), where polar(𝑴)\operatorname\*{polar}(\bm{M}) denotes the closest semi-orthogonal matrix to 𝑴\bm{M} [[19](#bib.bib19), Chapter 8].
Concretely, if 𝑴=𝑼​𝚺​𝑽𝖳\bm{M}=\bm{U}\bm{\Sigma}\bm{V}^{\mathsf{T}} is the singular value decomposition (SVD) of 𝑴\bm{M}, then

|  |  |  |  |
| --- | --- | --- | --- |
|  | polar(𝑴):=𝑼​𝑽𝖳.\operatorname\*{polar}(\bm{M}):=\bm{U}\bm{V}^{\mathsf{T}}. |  | (1) |

The matrix polar(𝑴)\operatorname\*{polar}(\bm{M}) can be seen as a generalization of the matrix sign function to rectangular matrices [[3](#bib.bib3)]. Indeed, when 𝑴\bm{M} is square symmetric with eigendecomposition 𝑴=𝑽​𝚲​𝑽𝖳\bm{M}=\bm{V}\bm{\Lambda}\bm{V}^{\mathsf{T}}, polar(𝑴)\operatorname\*{polar}(\bm{M}) exactly coincides with the matrix sign function sign(𝑴)=𝑽​sign(𝚲)⁡𝑽𝖳\operatorname\*{sign}(\bm{M})=\bm{V}\operatorname\*{sign}(\bm{\Lambda})\bm{V}^{\mathsf{T}}
[[19](#bib.bib19), Chapter 5]. Equivalently, polar(𝑴)\operatorname\*{polar}(\bm{M}) is the left orthogonal factor of the polar decomposition of 𝑴\bm{M} [[19](#bib.bib19), Chapter 8].
The motivation for Muon is that −polar(𝑴)-\operatorname\*{polar}(\bm{M}) gives the steepest-descent direction with respect to the *spectral norm* (instead of the Frobenius norm, as in standard SGD).

Recent work [[40](#bib.bib40)] shows that Muon can be viewed as a conditional gradient (Frank-Wolfe) method with a trust region defined by the spectral norm. In the same work, the authors also provide a convergence theory for the smooth and non-convex setting, as well as for the stochastic non-convex case. The analysis of Muon was further refined in [[41](#bib.bib41)], which proves convergence under a layerwise (L0,L1)(L\_{0},L\_{1})-smoothness assumption, in both the stochastic non-convex and stochastic Polyak–Łojasiewicz settings. We refer the reader to [[22](#bib.bib22)] and [[5](#bib.bib5)] for further background. In this paper, we take the Muon update rule as given and focus on the problem of efficiently computing the polar decomposition polar(𝑴)\operatorname\*{polar}(\bm{M}).

### 1.2  Computing the Polar Factor

Although polar(𝑴)\operatorname\*{polar}(\bm{M}) can be computed directly via an SVD in O​(min⁡(m​n2,n​m2))O(\min(mn^{2},nm^{2})) time, doing so is prohibitively expensive in deep learning applications, especially as standard SVD algorithms fail to take full advantage of the parallelism available on GPUs. There has been significant work on highly-parallel methods for the SVD, but the most common approaches actually require computing the matrix-sign function as a subroutine [[33](#bib.bib33), [35](#bib.bib35)].
Numerical analysts have spent decades developing iterative methods for computing polar(𝑴)\operatorname\*{polar}(\bm{M}).
This rich line of work includes Newton–Schulz [[19](#bib.bib19), Chapter 8], Padé iteration [[25](#bib.bib25), [18](#bib.bib18)], the Newton and scaled Newton iterations [[19](#bib.bib19), Chapter 8], the QWHD iteration [[32](#bib.bib32), [35](#bib.bib35)], and *Zolo-pd* (Zolotarev polar decomposition) [[33](#bib.bib33)].
Unfortunately, as discussed in [Section 2](#S2 "2 Related Work ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"), most of these methods are based on rational approximations to the function sign(x)\operatorname\*{sign}(x) and require computing matrix inverses or QR decompositions.
Such methods are ill-suited to GPU acceleration and deep learning applications.
In contrast, the older Newton-Schulz method is based on *polynomial* approximation of sign(x)\operatorname\*{sign}(x) and uses only matrix-matrix products.
Thus, Muon initially used Newton-Schulz [[4](#bib.bib4)]. Indeed, Muon stands for “MomentUm Orthogonalized by Newton-Schulz” [[22](#bib.bib22)].

#### The Newton-Schulz methods.

Newton-Schulz constructs a sequence of approximations 𝑿t≈polar(𝑴)\bm{X}\_{t}\approx\operatorname\*{polar}(\bm{M}) as follows:

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  |  | 𝑿0=𝑴/‖𝑴‖F\displaystyle\bm{X}\_{0}=\bm{M}/\|\bm{M}\|\_{\text{F}} | 𝑿t+1=32​𝑿t−12​𝑿t​𝑿t⊤​𝑿t\displaystyle\bm{X}\_{t+1}=\frac{3}{2}\bm{X}\_{t}-\frac{1}{2}\bm{X}\_{t}\bm{X}\_{t}^{\top}\bm{X}\_{t} |  | (2) |

At each iteration, this rule effectively applies the cubic polynomial p​(x)=32​x−12​x3p(x)=\frac{3}{2}x-\frac{1}{2}x^{3} to each singular value of 𝑿t\bm{X}\_{t}.
The scalar fixed-point iteration xt+1=p​(xt)x\_{t+1}=p(x\_{t}) converges to sign(x0)\operatorname\*{sign}(x\_{0}) as t→∞t\to\infty, provided |x0|≤1|x\_{0}|\leq 1.
As a result, the matrix iteration satisfies limt→∞𝑿t=𝑼​𝑽⊤=polar(𝑿0)\lim\limits\_{t\to\infty}\bm{X}\_{t}=\bm{U}\bm{V}^{\top}=\operatorname\*{polar}(\bm{X}\_{0}).
Higher-degree versions of Newton-Schulz follow the same principle. For example, the degree-5 polynomial p​(x)=(15​x−10​x3+3​x5)/8p(x)=(15x-10x^{3}+3x^{5})/8 converges even faster. The Newton-Schulz iterations converge super-exponentially when 𝑿t\bm{X}\_{t} is sufficiently close to polar(𝑴)\operatorname\*{polar}(\bm{M}), but they suffer from slow initial convergence; when 𝑿0\bm{X}\_{0} is far from polar(𝑴)\operatorname\*{polar}(\bm{M}), the approximation improves slowly over the first few iterations.

#### The Jordan and You methods.

In Muon, high accuracy approximations to polar(𝑴)\operatorname\*{polar}(\bm{M}) are usually not necessary. The primary goal is instead to compute a coarse approximation in as few iterations as possible. To accelerate convergence in the low-accuracy regime, Jordan recently proposed a fixed-point iteration based on the polynomial p​(x)=3.4445​x−4.7750​x3+2.0315​x5p(x)=3.4445x-4.7750x^{3}+2.0315x^{5} [[22](#bib.bib22)],
which was found using a heuristic numerical search.
Unlike Newton-Schulz, the scheme that Jordan proposed does not converge to polar(𝑴)\operatorname\*{polar}(\bm{M}). Instead, it plateaus at an error of ≈0.3\approx 0.3. However, it reaches this level of accuracy rapidly.
As a result, when the number of iterations is smaller than about 1010, Jordan’s method outperforms the Newton-Schulz iteration.
Building on this idea, You [[9](#bib.bib9)] proposed a method that applies six different polynomial updates in succession, which were again found by heuristic search.
This method achieves better accuracy than Jordan’s but still fails to converge.

We introduce a new method.
In particular, we derive polynomial update rules that are *optimal* at every iteration, outperforming all previous polynomial methods in our setting.

### 1.3  Contributions

We present Polar Express, an iterative method for approximating polar(𝑴)\operatorname\*{polar}(\bm{M}).
Our method dynamically adapts the polynomial update rule at each iteration, prioritizing rapid progress in the initial stage and high accuracy in the later stage.
Polar Express constructs polynomials p1,…,pTp\_{1},\ldots,p\_{T} so that the resulting composition is the optimal approximation to the sign function with respect to the supremum (L∞L^{\infty}) norm ([Theorem 4.1](#S4.Thmdefinition1 "Theorem 4.1. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")).
By iteratively applying these polynomials to 𝑴\bm{M}, Polar Express computes an approximation to polar(𝑴)\operatorname\*{polar}(\bm{M}) that is optimal in the worst-case at every iteration.
Our method converges to polar(𝑴)\operatorname\*{polar}(\bm{M}) super-exponentially ([Theorem 4.3](#S4.Thmdefinition3 "Theorem 4.3. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")), and it quickly reaches a good approximation within just five to ten iterations.
This early-stage acceleration is especially valuable in deep learning applications, where runtime efficiency takes precedence over high accuracy.
In contrast, classical methods like Newton-Schulz suffer from a slow initial convergence, while recent heuristic proposals [[22](#bib.bib22), [9](#bib.bib9)] fail to converge.
Our method is efficient to run on GPUs, using only a few matrix-matrix products per iteration.111In [Appendices E](#A5 "Appendix E Initialization for Matrices with Large Spectral Gaps ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") and [F](#A6 "Appendix F Fast Polynomial Iteration for Rectangular Matrices ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"), we describe two further algorithmic ideas that can be incorporated into Polar Express. They are not used in our Muon experiments but they may be beneficial in other settings, and we believe they merit further study.

We give an explicit instantiation of Polar Express in [Algorithm 1](#alg1 "In 1 Introduction ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"), which incorporates minor modifications to make it compatible with half-precision arithmetic (see [Section 4.4](#S4.SS4 "4.4 Finite precision considerations ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")).
[Algorithm 1](#alg1 "In 1 Introduction ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") can be used as a drop-in replacement for previous methods.
In numerical experiments, Polar Express outperforms previous methods on synthetic matrices and gradient matrices from a GPT-2 transformer ([Figure 4](#S5.F4 "In 5.1 Convergence of Polar Express ‣ 5 Numerical Experiments ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")).
We demonstrate the effectiveness of using Polar Express within the Muon optimizer in [Figure 1](#S1.F1 "In 1 Introduction ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"), showing that it consistently improves the training of GPT-2 language models on 1 billion tokens of the FineWeb dataset [[2](#bib.bib2)].

#### Notation.

We let ‖𝑴‖F\|\bm{M}\|\_{\text{F}} and ‖𝑴‖2\|\bm{M}\|\_{2} denote the Frobenius norm and spectral norm (largest singular value) of a matrix 𝑴\bm{M}, respectively. We denote the spectrum (set of singular values) by σ​(𝑴)\sigma(\bm{M}).

Let ℙd\mathbb{P}\_{d} be the set of polynomials of degree at most dd. For odd dd, ℙdodd\mathbb{P}\_{d}^{\operatorname\*{odd}} denotes the set of polynomials of degree at most dd containing only odd-degree monomials. For a polynomial pp, deg⁡(p)\deg(p) is its degree. Let sign(x)\operatorname\*{sign}(x) be the scalar sign function, which satisfies sign(0)=0\operatorname\*{sign}(0)=0, sign(x)=1\operatorname\*{sign}(x)=1 if x>0x>0 and sign(x)=−1\operatorname\*{sign}(x)=-1 if x<0x<0.

For a polynomial p∈ℙdoddp\in\mathbb{P}\_{d}^{\operatorname\*{odd}} and a matrix 𝑴\bm{M} with rank reduced SVD given by 𝑴=𝑼​𝚺​𝑽𝖳\bm{M}=\bm{U}\bm{\Sigma}\bm{V}^{\mathsf{T}} and positive singular values σ1≥⋯≥σrank(𝑴)>0\sigma\_{1}\geq\cdots\geq\sigma\_{\operatorname\*{rank}(\bm{M})}>0, we define p​(𝑴):=𝑼​p​(𝚺)​𝑽𝖳p(\bm{M}):=\bm{U}p(\bm{\Sigma})\bm{V}^{\mathsf{T}}, where p​(𝚺)p(\bm{\Sigma}) is the diagonal matrix with diagonal entries p​(σi)p(\sigma\_{i}) for i=1,…,rank(𝑴)i=1,\ldots,\operatorname\*{rank}(\bm{M}).

## 2  Related Work

Computing polar(𝑴)\operatorname\*{polar}(\bm{M}) is an important and longstanding problem in numerical linear algebra, with applications spanning electronic structure calculations, lattice quantum chromodynamics, orthogonal Procrustes analysis, parallel algorithms for computing the SVD, and beyond; see e.g. [[18](#bib.bib18), [23](#bib.bib23), [8](#bib.bib8), [16](#bib.bib16), [36](#bib.bib36), [45](#bib.bib45)].

#### Newton-Schulz and polynomial Padé methods.

The earliest methods in the literature are polynomial iterations like ([2](#S1.E2 "Equation 2 ‣ The Newton-Schulz methods. ‣ 1.2 Computing the Polar Factor ‣ 1 Introduction ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")). Several nearly simultaneous papers introduced the family of polynomial Padé iterations, comprising Newton-Schulz and its higher-degree analogues [[27](#bib.bib27), [6](#bib.bib6), [18](#bib.bib18), [29](#bib.bib29)].
These higher-degree methods are also sometimes called “Newton-Schulz”; when doing so, we will specify the degree for clarity.
In these methods, each iteration refines the current approximation 𝑿t\bm{X}\_{t} by applying a low-degree odd matrix polynomial, where any odd monomial x↦x2​q+1x\mapsto x^{2q+1} is defined for rectangular matrices by the formula 𝑿t↦𝑿t​(𝑿t⊤​𝑿t)q\bm{X}\_{t}\mapsto\bm{X}\_{t}\left(\bm{X}\_{t}^{\top}\bm{X}\_{t}\right)^{q}.
Our Polar Express method also takes this form, though unlike Newton-Schulz, it changes the polynomial at each iteration.

The polynomials used in Padé methods are chosen to match the value and first few derivatives of sign(x)\operatorname\*{sign}(x) at the points x=±1x=\pm 1. For instance, the update rule of the third method in this family is defined by p​(x)=116​(35​x−35​x3+21​x5−5​x7)p(x)=\frac{1}{16}\left(35x-35x^{3}+21x^{5}-5x^{7}\right), which is the unique degree-7 polynomial satisfying p​(±1)=±1p(\pm 1)=\pm 1 and p′​(±1)=p′′​(±1)=p′′′​(±1)=0p^{\prime}(\pm 1)=p^{\prime\prime}(\pm 1)=p^{\prime\prime\prime}(\pm 1)=0. These methods converge so long as all singular values of 𝑿0\bm{X}\_{0} lie in (0,1](0,1], a condition guaranteed by the initialization of ([2](#S1.E2 "Equation 2 ‣ The Newton-Schulz methods. ‣ 1.2 Computing the Polar Factor ‣ 1 Introduction ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")).
Furthermore, the order of convergence of the degree 2​q+12q+1 method is q+1q+1 [[6](#bib.bib6)]. In particular, the Newton-Schulz method (q=1q=1) converges quadratically.

#### Newton’s method and rational Padé.

In the numerical analysis literature, polynomial methods were succeeded by rational iterations like Newton’s method [[18](#bib.bib18)], defined as follows222Our description of Newton’s method and other rational methods assumes square non-singular 𝑴\bm{M}. Non-square problems can be reduced to the square case by an initial QR decomposition, but this is not an option for purely polynomial methods like ours.:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝑿0=𝑴\displaystyle\bm{X}\_{0}=\bm{M} | 𝑿t+1=12​(𝑿t+𝑿t−⊤)\displaystyle\bm{X}\_{t+1}=\frac{1}{2}\left(\bm{X}\_{t}+\bm{X}\_{t}^{-\top}\right) |  | (3) |

Newton’s method also converges quadratically. Like Newton-Schulz, it works because the rational function r​(x)=12​(x+x−1)r(x)=\frac{1}{2}(x+x^{-1}) has a stable fixed point at 11; unlike for Newton-Schulz, this point is a global attractor for the whole positive real line.
At first glance, Newton’s method has nothing to do with the Padé iterations discussed above.
However, after a change of variables 𝒀t=𝑿t−1\bm{Y}\_{t}=\bm{X}\_{t}^{-1}, it can be reinterpreted as 𝒀t+1=2​𝒀t​(𝑰+𝒀t⊤​𝒀t)−1\bm{Y}\_{t+1}=2\bm{Y}\_{t}(\bm{I}+\bm{Y}\_{t}^{\top}\bm{Y}\_{t})^{-1}, which is sometimes called inverse Newton.
Observing that r​(x)=2​x1+x2r(x)=\frac{2x}{1+x^{2}} satisfies r​(±1)=±1r(\pm 1)=\pm 1 and r′​(±1)=0r^{\prime}(\pm 1)=0, we see that (inverse) Newton is also a Padé method, though a rational rather than polynomial one.
In fact, given a odd degree 2​qn+12q\_{n}+1 for the numerator and an even degree 2​qd2q\_{d} for the denominator, there is a unique rational function that matches the value and first qn+qdq\_{n}+q\_{d} derivatives of sign(x)\operatorname\*{sign}(x) at x=±1x=\pm 1.
This directly yields a Padé method for computing polar(𝑴)\operatorname\*{polar}(\bm{M}) whose order of convergence is qn+qd+1q\_{n}+q\_{d}+1.
For instance, r​(x)=3​x+x31+3​x2r(x)=\frac{3x+x^{3}}{1+3x^{2}} is called Halley’s method, which converges cubically.
When qd=0q\_{d}=0, we recover the polynomial Padé methods.

There are two main weakness of Newton’s method and the Padé iterations: slow convergence in the initial phase and the need to compute explicit inverses.
To accelerate initial convergence, Higham popularized the technique of rescaling the matrix after every Newton iteration [[18](#bib.bib18)].
Intuitively, rescaling 𝑿t\bm{X}\_{t} so that σmax=1/σmin\sigma\_{\max}=1/\sigma\_{\min} centers the spectrum around 11, where convergence is fastest.
Several easily-computable choices of scaling factor exist to accomplish this approximately.
Note that this rescaling scheme would fail for Newton-Schulz, which likewise suffers from slow initial convergence but which would diverge if σmax≫1\sigma\_{\max}\gg 1.

Computing matrix inverses is difficult to parallelize and to implement stably in low precision arithmetic.
However, a trick was developed for stably computing many rational methods *without* explicit inverses; QR decompositions can be used instead [[32](#bib.bib32), [49](#bib.bib49)].
Applying this trick to Halley’s method and combining with a special rescaling scheme yields the QDWH (QR-based dynamically weighted Halley) method, which converges in just six iterations for any reasonably conditioned matrix [[32](#bib.bib32)].

#### Adaptive rational methods from optimal approximations.

A landmark 2016 paper introduced a new paradigm to design iterative methods for computing polar(𝑴)\operatorname\*{polar}(\bm{M}) [[33](#bib.bib33)].
We describe this paradigm in more detail in [Section 4](#S4 "4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"), but the main insight is as follows.
Padé methods choose the update rule to be an approximation to sign(x)\operatorname\*{sign}(x) of a given degree that is optimally accurate in the neighborhood of x=1x=1.
Instead, we should choose the approximation to sign(x)\operatorname\*{sign}(x) that is optimal over an *interval* [ℓ,1]⊂ℝ≥0[\ell,1]\subset\mathbb{R}\_{\geq 0} that contains the singular values.
Moreover, after each step of the algorithm, the range of the singular values changes; therefore, we adapt the update rule at each iteration to match the new interval.
When the range of the singular values is large, this approach ensures that the update rule shrinks it as quickly as possible.
As the algorithm proceeds and the interval shrinks to a small neighborhood of 11, the update rule approaches that of a Padé method, maintaining the same high order of convergence as it has.

Within the class of odd rational functions whose numerators and denominators have degree 2​q+12q+1 and 2​q2q, respectively, an explicit formula for this optimal approximation to sign(x)\operatorname\*{sign}(x) on any interval [ℓ,1][\ell,1] was found by Zolotarev.
It was shown that these rationals have remarkable convergence properties for any qq [[33](#bib.bib33)].
For q=1q=1, this optimal approximation coincides exactly with the dynamically weighted Halley’s method (QDWH) referenced above.
For even faster convergence than QDWH, [[33](#bib.bib33)] proposed the zolo-pd method, which uses q=17q=17.
Finally, these methods all admit the same QR-based implementation trick as QDWH.

#### Adaptive polynomial methods.

In this paper, we adopt the paradigm of zolo-pd [[33](#bib.bib33)] but with polynomials rather than rationals of degree (2​q+1,2​q)(2q+1,2q). This choice avoids the need for QR factorizations, relying solely on GPU-friendly matrix-matrix multiplications in low-precision arithmetic. While this class of methods has not been fully developed in the numerical analysis literature, similar ideas have been rediscovered in different guises.
In an unpublished manuscript that predates zolo-pd, Chen and Chow [[11](#bib.bib11)] describe a rescaling strategy for Newton-Schulz.
Though motivated differently, their method is equivalent to ours for degree-3 polynomials.
They also observe numerical instability that prevents the method from converging to all the way to machine precision.
Using the insights of [[34](#bib.bib34)], they propose a simple mitigation for this issue that we adopt in [Section 4.4](#S4.SS4 "4.4 Finite precision considerations ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm").
Our work gives this method a theoretical foundation that connects it to the paradigm of zolo-pd, and we prove its optimality in the sense of ([6](#S3.E6 "Equation 6 ‣ 3 Approximations by Compositions of Polynomials ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")).
In addition, we study odd polynomials of arbitrary degree and focus particularly on the degree-5 case.
Independently, a group of cryptographers developed a similar method for approximating the scalar function sign(x)\operatorname\*{sign}(x) in the context of homomorphic encryption schemes [[28](#bib.bib28)]. Their focus is mainly on tuning the analogues in their setting of the polynomial degree and number of iterations, whereas we focus on demonstrating optimality and efficiently constructing the update polynomials for degree 33 and 55.
In addition, we consider matrix-valued inputs in low-precision arithmetic—not scalars in exact arithmetic—and we demonstrate our method’s effectiveness within the Muon algorithm for training deep neural networks.

#### Application within Muon.

The designers of Muon realized that, due to the extreme efficiency requirements and lax accuracy requirements of their setting, rational-based methods from the numerical analysis literature are inapplicable.
However, polynomial-based iteration schemes can take full advantage of GPUs because they use only matrix-matrix products in half-precision arithmetic, not inverses or QR decompositions.
The preference for speed over accuracy motivates methods that aim to quickly produce coarse approximations, even at the cost of asymptotic convergence.
Examples include the proposals of Jordan [[22](#bib.bib22)] and You [[44](#bib.bib44), [9](#bib.bib9)], as discussed in [Section 1.2](#S1.SS2 "1.2 Computing the Polar Factor ‣ 1 Introduction ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"). Like Chen and Chow [[11](#bib.bib11)], Jordan found that convergence in the initial phase can be accelerated by choosing update rules that have a large derivative near zero, so as to increase the small singular values as much as possible at each iteration.
You furthermore chose to use different update rules at each iteration, allowing extra flexibility to tune the trade-off between speed and accuracy.
Both used degree-5 polynomials that were found through gradient descent on heuristic objective functions.
These proposals were previously compared to Newton-Schultz333Jordan [[22](#bib.bib22)] actually compares to 2​x−32​x3+12​x52x-\frac{3}{2}x^{3}+\frac{1}{2}x^{5}, whereas the true degree-5 Newton-Schulz polynomial is (15​x−10​x3+3​x5)/8(15x-10x^{3}+3x^{5})/8. However, the difference in performance is negligible for the first few iterations., but never to Chen and Chow’s method.
We find that our method outperforms them all.

## 3  Approximations by Compositions of Polynomials

To design a GPU-friendly method for computing polar(𝑴)\operatorname\*{polar}(\bm{M}), we limit ourselves to the following GPU-friendly operations:

1. i)

   Linear combinations: given scalars β,γ∈ℝ\beta,\gamma\in\mathbb{R} and matrices 𝑩\bm{B} and 𝑪\bm{C}, compute β​𝑩+γ​𝑪\beta\bm{B}+\gamma\bm{C},
2. ii)

   Matrix-matrix products: compute 𝑩​𝑪\bm{B}\bm{C}.

While both these computational primitives are well-suited for parallel computing environments, matrix-matrix products come at a higher computational cost than linear combinations. Therefore, our method attempts to minimize the number of matrix-matrix products.
A key observation is that we can compute *odd* monomials of 𝑴=𝑼​𝚺​𝑽𝖳\bm{M}=\bm{U}\bm{\Sigma}\bm{V}^{\mathsf{T}} using the following formula:

|  |  |  |
| --- | --- | --- |
|  | 𝑴2​q+1:=𝑼​𝚺2​q+1​𝑽𝖳=𝑴​(𝑴𝖳​𝑴)q.\bm{M}^{2q+1}:=\bm{U}\bm{\Sigma}^{2q+1}\bm{V}^{\mathsf{T}}=\bm{M}(\bm{M}^{\mathsf{T}}\bm{M})^{q}. |  |

Hence, for an odd polynomial p​(x)=a0​x+a1​x3+⋯+aq​x2​q+1p(x)=a\_{0}x+a\_{1}x^{3}+\cdots+a\_{q}x^{2q+1} we can compute

|  |  |  |
| --- | --- | --- |
|  | p​(𝑴):=a0​𝑴+a1​𝑴​(𝑴𝖳​𝑴)+⋯+aq​𝑴​(𝑴𝖳​𝑴)q.p(\bm{M}):=a\_{0}\bm{M}+a\_{1}\bm{M}(\bm{M}^{\mathsf{T}}\bm{M})+\cdots+a\_{q}\bm{M}(\bm{M}^{\mathsf{T}}\bm{M})^{q}. |  |

It has been shown that for an arbitrary polynomial pp, one requires Θ(deg(p)1/2)\Theta(\deg(p)^{1/2}) products to compute p​(𝑴)p(\bm{M}) [[39](#bib.bib39)]; see also [[20](#bib.bib20)] for related work.
This compares favorably to the naive approach that forms all monomials in pp and then sums them together, which requires Ω​(deg⁡(p))\Omega(\deg(p)) products.
However, if pp can be expressed as a composition of TT polynomials, each of degree dd

|  |  |  |  |
| --- | --- | --- | --- |
|  | p=pT∘pT−1∘⋯∘p1,p=p\_{T}\circ p\_{T-1}\circ\cdots\circ p\_{1}, |  | (4) |

then the degree of pp is dTd^{T}, and p​(𝑴)p(\bm{M}) can be efficiently computed recursively by

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝑿0=𝑴,𝑿t=pt​(𝑿t−1)​ for ​t=1,2,…,T.\bm{X}\_{0}=\bm{M},\quad\bm{X}\_{t}=p\_{t}(\bm{X}\_{t-1})\text{ for }t=1,2,\ldots,T. |  | (5) |

The final iterate is 𝑿T=p​(𝑴)\bm{X}\_{T}=p(\bm{M}), which we compute with just O​(T​d)O(Td) matrix-matrix products.

Iterative methods for polar(𝑴)\operatorname\*{polar}(\bm{M}) can be seen in this light.
For instance, the degree-5 Newton-Schulz method uses the polynomial update pt​(x)=158​x−108​x3+38​x5p\_{t}(x)=\frac{15}{8}x-\frac{10}{8}x^{3}+\frac{3}{8}x^{5} for each t=1,…,Tt=1,\ldots,T.
The composition p=pT∘⋯∘p1p=p\_{T}\circ\cdots\circ p\_{1} approximates sign(x)\operatorname\*{sign}(x), and the approximation error goes to 0 as TT grows.
In this paper, we ask the following question: what choice of pT∘⋯∘p1p\_{T}\circ\cdots\circ p\_{1} gives the *best* approximation to sign(x)\operatorname\*{sign}(x)?

The method we will present is optimal in the following sense: given lower and upper bounds ℓ\ell and uu on the singular values of 𝑴\bm{M}, an odd degree d∈ℕd\in\mathbb{N}, and the number of iterations T∈ℕT\in\mathbb{N}, our method computes the composition p⋆​(𝑴)p^{\star}(\bm{M}) that minimizes the worst-case error in the spectral norm. That is,

|  |  |  |  |
| --- | --- | --- | --- |
|  | p⋆=argminp=pT∘pT−1∘⋯∘p1pt∈ℙdoddmax𝑴∈ℝm×nσ​(𝑴)⊂[ℓ,u]⁡‖polar(𝑴)−p​(𝑴)‖2.p^{\star}=\operatorname\*{argmin}\_{\begin{subarray}{c}p=p\_{T}\circ p\_{T-1}\circ\cdots\circ p\_{1}\\ p\_{t}\in\mathbb{P}\_{d}^{\operatorname\*{odd}}\end{subarray}}\max\_{\begin{subarray}{c}\bm{M}\in\mathbb{R}^{m\times n}\\ \sigma(\bm{M})\subset[\ell,u]\end{subarray}}\left\|\operatorname\*{polar}(\bm{M})-p(\bm{M})\right\|\_{2}. |  | (6) |

Given that polar(𝑴)−p​(𝑴)=𝑼​(𝑰−p​(𝚺))​𝑽𝖳\operatorname\*{polar}(\bm{M})-p(\bm{M})=\bm{U}(\bm{I}-p(\bm{\Sigma}))\bm{V}^{\mathsf{T}}, and by the unitary invariance of the spectral norm, we have that ([6](#S3.E6 "Equation 6 ‣ 3 Approximations by Compositions of Polynomials ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")) is equivalent to

|  |  |  |  |
| --- | --- | --- | --- |
|  | p⋆=argminp=pT∘pT−1∘⋯∘p1pt∈ℙdoddmaxx∈[ℓ,u]⁡|1−p​(x)|.p^{\star}\;=\;\operatorname\*{argmin}\_{\begin{subarray}{c}p=p\_{T}\circ p\_{T-1}\circ\cdots\circ p\_{1}\\ p\_{t}\in\mathbb{P}\_{d}^{\operatorname\*{odd}}\end{subarray}}\,\max\_{x\in[\ell,u]}\left|1-p(x)\right|. |  | (7) |

For completeness, the equivalence between ([6](#S3.E6 "Equation 6 ‣ 3 Approximations by Compositions of Polynomials ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")) and ([7](#S3.E7 "Equation 7 ‣ 3 Approximations by Compositions of Polynomials ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")) is proven in [Appendix C](#A3 "Appendix C Proof of equivalence between (6) and (7) ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm").

![Refer to caption](/html/2505.16932/assets/img/demonstration.png)


(a) 
The left figure compares the composition (for T=6T=6 and d=5d=5) of polynomials given by Polar Express (ℓ=0.001\ell=0.001), You’s method (which is defined up to 6 iterations), Newton-Schulz, and Jordan’s method for approximating sign(x)\operatorname\*{sign}(x). The right figure demonstrates the convergence of the methods on [0.001,1][0.001,1]. Note the slow initial convergence of Newton-Schulz.

![Refer to caption](/html/2505.16932/assets/img/steps.png)


(b) 
The evolution of the first three optimal polynomials p1p\_{1}, p2p\_{2}, and p3p\_{3} and the corresponding lower bounds ℓt+1=pt​(ℓt)\ell\_{t+1}=p\_{t}(\ell\_{t}) and upper bounds ut+1=2−ℓt+1u\_{t+1}=2-\ell\_{t+1}, as described in [Theorem 4.1](#S4.Thmdefinition1 "Theorem 4.1. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"). The horizontal black line indicates y=1y=1. The polynomial degree is d=5d=5 and the number of iterations is T=3T=3. We set ℓ1=0.03\ell\_{1}=0.03 and u1=1u\_{1}=1.

Figure 2:

In other words, the problem given in ([6](#S3.E6 "Equation 6 ‣ 3 Approximations by Compositions of Polynomials ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")) reduces to that of finding a “uniform” or “minimax” approximation to the constant function x↦1x\mapsto 1 over the interval [ℓ,u][\ell,u], as given in ([7](#S3.E7 "Equation 7 ‣ 3 Approximations by Compositions of Polynomials ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")).
Uniform approximation on an interval by polynomials or rational functions of a given degree is a central topic in approximation theory; see e.g. [[46](#bib.bib46)].
Here, we seek an approximation of a particular form—a *composition* of odd polynomials of fixed degrees.
In the next section, we solve the optimization problem of ([7](#S3.E7 "Equation 7 ‣ 3 Approximations by Compositions of Polynomials ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")) and use the solution to create Polar Express. [Figure 2](#S3.F2 "In 3 Approximations by Compositions of Polynomials ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") (a) shows the resulting p∗p^{\*} polynomial labeled as PolarExp, as compared to the Jordan’s method in [[22](#bib.bib22)], and the six iterations of You’s method in [[9](#bib.bib9)].

## 4  The Polar Express

### 4.1  Greedy is optimal

The key observation is that the polynomial used in each iteration can be chosen greedily, given the choice of polynomials from the previous iterations.
For the first iteration, we choose p1p\_{1} so as to map the interval [ℓ,u][\ell,u] as close to 11 as possible.
That is, it minimizes maxx∈[ℓ,u]⁡|1−p1​(x)|\max\_{x\in[\ell,u]}|1-p\_{1}(x)|.
The image of p1p\_{1} will be a new interval [ℓ2,u2][\ell\_{2},u\_{2}], where

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℓ2=minx∈[ℓ,u]⁡p1​(x)u2=maxx∈[ℓ,u]⁡p1​(x)\ell\_{2}=\min\_{x\in[\ell,u]}p\_{1}(x)\qquad\qquad u\_{2}=\max\_{x\in[\ell,u]}p\_{1}(x) |  | (8) |

We now pick p2p\_{2} to map the interval [ℓ2,u2][\ell\_{2},u\_{2}] as close to 11 as possible, obtaining a new interval [ℓ3,u3][\ell\_{3},u\_{3}] that is the image of [ℓ,u][\ell,u] through p2∘p1p\_{2}\circ p\_{1}.
We continue this process for as many iterations as desired.

The following theorem guarantees that this process finds the solution to ([7](#S3.E7 "Equation 7 ‣ 3 Approximations by Compositions of Polynomials ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")), and thereby also ([6](#S3.E6 "Equation 6 ‣ 3 Approximations by Compositions of Polynomials ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")).
The scheme is also outlined in [Figure 2](#S3.F2 "In 3 Approximations by Compositions of Polynomials ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") (b), which demonstrates the evolution of the lower bounds ℓt\ell\_{t}, the upper bounds utu\_{t}, and the polynomials ptp\_{t} across iterations.

###### Theorem 4.1.

Let dd be odd and define ℓ1=ℓ\ell\_{1}=\ell and u1=uu\_{1}=u. For t=1,…,Tt=1,\ldots,T define

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | pt\displaystyle p\_{t} | =argminp∈ℙdoddmaxx∈[ℓt,ut]⁡|1−p​(x)|\displaystyle=\;\operatorname\*{argmin}\_{\begin{subarray}{c}p\in\mathbb{P}\_{d}^{\operatorname\*{odd}}\end{subarray}}\,\max\_{x\in[\ell\_{t},u\_{t}]}|1-p(x)| |  | (9) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ℓt+1\displaystyle\ell\_{t+1} | =minx∈[ℓt,ut]⁡pt​(x)\displaystyle=\;\min\_{x\in[\ell\_{t},u\_{t}]}p\_{t}(x) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ut+1\displaystyle u\_{t+1} | =maxx∈[ℓt,ut]⁡pt​(x)\displaystyle=\;\max\_{x\in[\ell\_{t},u\_{t}]}p\_{t}(x) |  |

Then the new error, lower and upper bounds can be computed through

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℓt+1=pt​(ℓt),ut+1=2−ℓt+1, and maxx∈[ℓt,ut]⁡|1−pt​(x)|=1−ℓt+1.\ell\_{t+1}=p\_{t}(\ell\_{t}),\quad u\_{t+1}=2-\ell\_{t+1},\quad\text{ and }\quad\max\limits\_{x\in[\ell\_{t},u\_{t}]}|1-p\_{t}(x)|=1-\ell\_{t+1}. |  | (10) |

Furthermore, the composition p⋆:=pT∘pT−1∘⋯∘p1p^{\star}:=p\_{T}\circ p\_{T-1}\circ\cdots\circ p\_{1} is optimal and the error is given by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | maxx∈[ℓ,u]⁡|1−p⋆​(x)|=minp=pT∘pT−1∘⋯∘p1pt∈ℙdodd⁡maxx∈[ℓ,u]⁡|1−p​(x)|=1−ℓT+1.\max\limits\_{x\in[\ell,u]}|1-p^{\star}(x)|\quad=\quad\min\_{\begin{subarray}{c}p=p\_{T}\circ p\_{T-1}\circ\cdots\circ p\_{1}\\ p\_{t}\in\mathbb{P}\_{d}^{\operatorname\*{odd}}\end{subarray}}\,\max\_{x\in[\ell,u]}\left|1-p(x)\right|=1-\ell\_{T+1}. |  | (11) |

###### Proof.

See [Appendix A](#A1 "Appendix A Proof of Theorem 4.1 ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm").
∎

###### Remark 4.2 (Why a fixed degree?).

We note that choice of the degree of each p1,p2,…,pTp\_{1},p\_{2},\ldots,p\_{T} need not be the same for [Theorem 4.1](#S4.Thmdefinition1 "Theorem 4.1. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") to hold. More generally, one may specify a sequence of degrees d1,…,dTd\_{1},\ldots,d\_{T} and define each ptp\_{t} as

|  |  |  |
| --- | --- | --- |
|  | pt=argminp∈ℙdtoddmaxx∈[ℓt,ut]⁡|p​(x)−1|,for ​t=1,…,T.p\_{t}=\operatorname\*{argmin}\_{\begin{subarray}{c}p\in\mathbb{P}\_{d\_{t}}^{\operatorname\*{odd}}\end{subarray}}\,\max\_{x\in[\ell\_{t},u\_{t}]}|p(x)-1|,\qquad\mbox{for }t=1,\ldots,T. |  |

Our theory translates entirely to this more general case. However, for simplicity we assume d=dtd=d\_{t} for all t=1,…,Tt=1,\ldots,T. Our setting is similar to that of [[28](#bib.bib28)], which considers the closely related problem of choosing the depth TT and degrees d1,…,dTd\_{1},\ldots,d\_{T} such that pp approximates sign\operatorname\*{sign} up to a prescribed error tolerance while minimizing the number of scalar multiplications. Interestingly, from [[28](#bib.bib28), Table 2] the optimal choice of degrees is dt=5d\_{t}=5 for *almost* all iterations. This justifies choosing dd to be a constant and our use of d=5d=5 in particular.

Fortunately, ([10](#S4.E10 "Equation 10 ‣ Theorem 4.1. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")) shows that once ptp\_{t} has been found, we can compute the new lower and upper bounds ℓt+1\ell\_{t+1} and ut+1u\_{t+1} and the approximation error simply by evaluating pt​(ℓt)p\_{t}(\ell\_{t}). Hence, for any *fixed* upper and lower bounds on the singular values of 𝑴\bm{M}, we can *precompute* the polynomials p1,…,pTp\_{1},\ldots,p\_{T} and the bounds [ℓ1,u1],…,[ℓT+1,uT+1][\ell\_{1},u\_{1}],\ldots,[\ell\_{T+1},u\_{T+1}].
Then, applying the iterative procedure of ([5](#S3.E5 "Equation 5 ‣ 3 Approximations by Compositions of Polynomials ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")), the final iterate 𝑿T\bm{X}\_{T} will satisfy the following error bound

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖polar(𝑴)−𝑿T‖2=‖polar(𝑴)−p⋆​(𝑴)‖2≤1−ℓT+1.\|\operatorname\*{polar}(\bm{M})-\bm{X}\_{T}\|\_{2}=\|\operatorname\*{polar}(\bm{M})-p^{\star}(\bm{M})\|\_{2}\leq 1-\ell\_{T+1}. |  | (12) |

From the optimality guarantee of [Theorem 4.1](#S4.Thmdefinition1 "Theorem 4.1. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"), we know that our method converges at least as fast as the Newton-Schulz iteration of the same degree.
Combining this fact with an existing analysis of Newton-Schulz, we immediately get the following convergence guarantee showing that our method enjoys faster than exponential convergence.

###### Theorem 4.3.

Let 𝑴\bm{M} be a matrix normalized so that σ​(𝑴)⊂[ℓ,1]\sigma(\bm{M})\subset[\ell,1]. Let 𝑿T=p⋆​(𝑴)\bm{X}\_{T}=p^{\star}(\bm{M}), where p⋆p^{\star} is the polynomial from [Theorem 4.1](#S4.Thmdefinition1 "Theorem 4.1. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") with d=2​q+1d=2q+1. Then, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖polar(𝑴)−𝑿T‖2≤|1−ℓ2|(q+1)T.\|\operatorname\*{polar}(\bm{M})-\bm{X}\_{T}\|\_{2}\leq|1-\ell^{2}|^{(q+1)^{T}}. |  | (13) |

Hence, for d=3d=3 the method converges quadratically and for d=5d=5 the method converges cubically.

###### Proof.

See [Appendix B](#A2 "Appendix B Proof of Theorem 4.3 ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm").
∎

In fact, [Theorem 4.3](#S4.Thmdefinition3 "Theorem 4.3. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") underestimates how fast our method converges.
For degree d=5d=5, our method converges about twice as fast as Newton-Schulz (compare with [[11](#bib.bib11), Section 3.1]).
Furthermore, the same analysis applies even if p∗p^{\*} is constructed using a “lower bound” ℓ\ell that was too high. That is, replacing ℓ\ell on the right-hand side of ([13](#S4.E13 "Equation 13 ‣ Theorem 4.3. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")) by σmin\sigma\_{\min}, the theorem holds even if p∗p^{\*} is constructed to be optimal on the interval [ℓ,1][\ell,1] for ℓ>σmin\ell>\sigma\_{\min}.
Intuitively, when ℓ=u=1\ell=u=1, the polynomial p∗p^{\*} coincides exactly with the Newton-Schulz method.
Mistakenly setting ℓ>σmin\ell>\sigma\_{\min}, we obtain a polynomial that converges slower than the optimal polynomial but faster than Newton-Schulz, so the guarantee of [Theorem 4.3](#S4.Thmdefinition3 "Theorem 4.3. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") still holds (cf. [[11](#bib.bib11), Theorem 3.3]).

### 4.2  Finding the optimal polynomial for each iteration

[Theorem 4.1](#S4.Thmdefinition1 "Theorem 4.1. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") shows that we can solve ([7](#S3.E7 "Equation 7 ‣ 3 Approximations by Compositions of Polynomials ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")) by greedily choosing the optimal approximation pt∈ℙdoddp\_{t}\in\mathbb{P}\_{d}^{\operatorname\*{odd}} for each interval [ℓt,ut][\ell\_{t},u\_{t}] for t=1,…,Tt=1,\ldots,T.
In this section, we show how to find each ptp\_{t}.
Since we are now focused on just one iteration, we drop the subscripts. Given ℓ\ell and uu, we wish to solve the following optimization problem:

|  |  |  |  |
| --- | --- | --- | --- |
|  | argminp∈ℙdoddmaxx∈[ℓ,u]⁡|1−p​(x)|\operatorname\*{argmin}\_{\begin{subarray}{c}p\in\mathbb{P}\_{d}^{\operatorname\*{odd}}\end{subarray}}\,\max\_{x\in[\ell,u]}|1-p(x)| |  | (14) |

That is, we seek a minimax or uniform approximation of the function x↦1x\mapsto 1 on [ℓ,u][\ell,u] from the set of odd polynomials. (Equivalently, we seek a minimax optimal approximation to sign(x)\operatorname\*{sign}(x) on [−u,−ℓ]∪[ℓ,u][-u,-\ell]\cup[\ell,u].)

Problems of the form ([14](#S4.E14 "Equation 14 ‣ 4.2 Finding the optimal polynomial for each iteration ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")) are well-studied in approximation theory and numerical analysis.
The key mathematical insight underlying the solution is the Equioscillation Theorem, which we state formally for our setting in [Lemma A.1](#A1.Thmdefinition1 "Lemma A.1. ‣ Appendix A Proof of Theorem 4.1 ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm").
This theorem gives a surprising characterization of the optimal solution of ([14](#S4.E14 "Equation 14 ‣ 4.2 Finding the optimal polynomial for each iteration ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")): an odd pp is optimal for degree 2​q+12q+1 if and only if there is a set of q+2q+2 equioscillating points.
This is a set of points at which pp achieves its maximum approximation error ±E\pm E, and for which the sign of the error alternates.
Even if the optimal approximation error EE is not known in advance, finding a set of q+2q+2 equioscillating points for a given EE serves as a certificate that no better approximation error is achievable.
The Equioscillation Theorem is the basis of the Remez algorithm [[37](#bib.bib37), [38](#bib.bib38)], a general tool that can be used to find (nearly) optimal polynomial approximations of a given degree to *any* function on any interval.
With very minor modifications to handle the constraint that pp be odd, Remez can be used to directly solve ([14](#S4.E14 "Equation 14 ‣ 4.2 Finding the optimal polynomial for each iteration ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")).

However, the Remez algorithm is opaque, complex, and difficult to implement correctly.
Fortunately, we do not need the Remez algorithm in its full generality to solve our problem. We seek only low degree polynomials, and the function we wish to approximate is the constant function f​(x)≡1f(x)\equiv 1.
For d=3d=3, we can derive an explicit, closed form solution to ([14](#S4.E14 "Equation 14 ‣ 4.2 Finding the optimal polynomial for each iteration ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")) using the Equioscillation Theorem.
Up to rescaling, the optimal polynomial turns out to be the same one derived in Chen and Chow by different means [[11](#bib.bib11)].
For degree d=5d=5, we present [Algorithm 3](#alg3 "In Appendix D Remez algorithm ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"), a much simpler way of solving ([14](#S4.E14 "Equation 14 ‣ 4.2 Finding the optimal polynomial for each iteration ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")) that is mathematically equivalent to Remez in our setting. This algorithm is implemented in its entirety in [Appendix G](#A7 "Appendix G Code for Constructing Polynomials of Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm").

We briefly describe the solution for d=3d=3. We seek a polynomial of the form p​(x)=a​x+b​x3p(x)=ax+bx^{3}. The Equioscillation Theorem stipulates that pp must have an equioscillating set of size 3. For pp to achieve its maximum error at a point xx, xx must be a local extremum of p​(x)−1p(x)-1 on the interval [ℓ,u][\ell,u]. Thus, for xx to be eligible for membership in the equioscillating set, it must either be a true local extremum of p​(x)−1p(x)-1 that happens to lie in [ℓ,u][\ell,u], or else one of the endpoints ℓ,u\ell,u.
However, because pp is an odd cubic, it has at most one true local extremum on ℝ≥0\mathbb{R}\_{\geq 0}. Thus, to build an equioscillating set of three points, we must include pp’s unique positive local extremum *and* both endpoints. This local extremum of pp occurs at −a3​b\sqrt{\frac{-a}{3b}}.
Therefore, we seek a,ba,b such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | p​(ℓ)=1−E,p​(−a3​b)=1+E,p​(u)=1−Ep(\ell)=1-E,\qquad\qquad p\left(\sqrt{\frac{-a}{3b}}\right)=1+E,\qquad\qquad p(u)=1-E |  | (15) |

for some EE.
This is a system of three equations in three variables.
The solution p​(x)=a​x+b​x3p(x)=ax+bx^{3} is most easily expressed as follows.
Let pNS​(x)=32​x−12​x3p\_{\operatorname\*{NS}}(x)=\frac{3}{2}x-\frac{1}{2}x^{3}. Then

|  |  |  |
| --- | --- | --- |
|  | p​(x)=β​pNS​(α​x), where ​α=3u2+l​u+ℓ2 and β=42+ℓ​u​(ℓ+u)​α3.p(x)=\beta p\_{\operatorname\*{NS}}(\alpha x),\quad\text{ where }\alpha=\sqrt{\frac{3}{u^{2}+lu+\ell^{2}}}\quad\text{ and }\quad\beta=\frac{4}{2+\ell u(\ell+u)\alpha^{3}}. |  |

We now turn to the degree-5 case. The intuition of [Algorithm 3](#alg3 "In Appendix D Remez algorithm ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") is as follows. For any fixed set of four points ℓ<q<r<u\ell<q<r<u, we can find an degree-5 odd polynomial p​(x)=a​x+b​x3+c​x5p(x)=ax+bx^{3}+cx^{5} that satisfies

|  |  |  |
| --- | --- | --- |
|  | p​(ℓ)=1−E,p​(q)=1+E,p​(r)=1−E,p​(u)=1+Ep(\ell)=1-E,\qquad p(q)=1+E,\qquad p(r)=1-E,\qquad p(u)=1+E |  |

for some EE by solving a 4×44\times 4 linear system in a,b,ca,b,c and EE.
Likewise, for any fixed degree-5 odd pp, we can find its four (or fewer) local extrema on [ℓ,u][\ell,u] as follows: they occur at ℓ,u\ell,u and the roots of p′p^{\prime}, which is an even degree-4 polynomial whose roots can easily be found by the *quadratic* formula.
[Algorithm 3](#alg3 "In Appendix D Remez algorithm ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") simply alternates between these two steps (solving for a,b,c,Ea,b,c,E and solving for q,rq,r) until the points q,rq,r converge.
Once they have converged, {ℓ,q,r,u}\{\ell,q,r,u\} forms an equioscillating set, so pp is the optimal polynomial.
For more details, please see [Appendix D](#A4 "Appendix D Remez algorithm ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm").

### 4.3  Upper and lower bounds on the singular values

To instantiate our method, we need upper and lower bounds uu and ℓ\ell on the singular values of the input matrix 𝑴\bm{M}.
A trivial upper bound is given by ‖𝑴‖F\|\bm{M}\|\_{\text{F}}.
For 𝑴∈ℝm×n\bm{M}\in\mathbb{R}^{m\times n} with n≤mn\leq m, this can overestimate σmax​(𝑴)\sigma\_{\max}(\bm{M}) by a factor of n\sqrt{n} in the worst case.
However in practice, the gradient matrices of the weights of dense linear layers in neural networks tend to have small effective rank [[48](#bib.bib48)].
Consequently, the Frobenius norm tends to be a reasonably good bound on the spectral norm that is loose only by a small constant factor.
We rescale the input matrix by setting 𝑿0=𝑴/‖𝑴‖F\bm{X}\_{0}=\bm{M}/\|\bm{M}\|\_{\text{F}} so that u=1u=1.

It is difficult to efficiently find a good lower bound on the smallest singular value, so we are forced to guess.
Fortunately, the consequences of a bad guess are not severe.
As discussed above, the method will eventually converge for any ℓ∈(0,u]\ell\in(0,u], and even an order of magnitude error in our guess of ℓ\ell only delays convergence by a few iterations.
For matrices stored in floating point arithmetic, the singular values are usually larger than machine precision ϵmach\epsilon\_{\text{mach}} [[7](#bib.bib7)], so a good guess is to set ℓ≈ϵmach\ell\approx\epsilon\_{\text{mach}}.
In our numerical experiments we work in bfloat16 where
ϵmach=2−7=0.0078125.\epsilon\_{\text{mach}}=2^{-7}=0.0078125.
Hence we set ℓ=10−3\ell=10^{-3} and u=1u=1.
Since we use these bounds for all input matrices, we can precompute the optimal polynomials once and apply them to as many inputs as we want.

### 4.4  Finite precision considerations

![Refer to caption](/html/2505.16932/assets/x3.png)


Figure 3: Effects of stabilizing the update rules with a safety factor and cushioning, as described in [Section 4.4](#S4.SS4 "4.4 Finite precision considerations ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm").
The blue curve is the optimal degree-5 polynomial for the interval [0.005,1][0.005,1].
It is has numerical issues because it maps singular values near 0.80.8 down to almost zero and maps 1+ϵ1+\epsilon to ≈ut+1+25​ϵ\approx u\_{t+1}+25\epsilon.
The stabilized version is better because it ensures pt​(x)x≥0.236\frac{p\_{t}(x)}{x}\geq 0.236 and maps all x≤1.01x\leq 1.01 to at most ut+1u\_{t+1}.

When working in finite-precision arithmetic, especially the half-precision bfloat16 format used in deep learning, we must take some care to avoid blowups and other problems due to numerical error.
To this end, we make three small changes to the method.
These adjustments stabilize the algorithm with a negligible effect on accuracy.
Furthermore, these adjustments can be made in the offline stage by modifying the coefficients of our optimal polynomials.

The first issue arises when numerical round-off creates singular values that are slightly larger than our current upper bound utu\_{t}.
Our optimal polynomials converge only when the singular values of 𝑿t\bm{X}\_{t} are less than utu\_{t}.
In some cases we have

|  |  |  |
| --- | --- | --- |
|  | pt​(ut+ϵ)>ut+1+ϵ,p\_{t}(u\_{t}+\epsilon)>u\_{t+1}+\epsilon, |  |

so over many iterations, a singular value that is slightly larger than utu\_{t} large could grow to ∞\infty instead of converging to 11.

To fix this issue, we simply replace each polynomial x↦pt​(x)x\mapsto p\_{t}(x) by x↦pt​(x/1.01)x\mapsto p\_{t}(x/1.01).
This safety factor corrects for round-off errors in previous iterations while only slightly changing the behavior of the polynomial on the interval [ℓt,ut][\ell\_{t},u\_{t}], though it does cause the singular values to converge to 0.9999980.999998 instead of to 11. To correct for this, the safety factor can be omitted in the final iteration.

The second issue was identified in [[34](#bib.bib34)] and addressed in the context of polynomial iterations by Chen and Chow [[11](#bib.bib11)].
In general, iterative methods for polar(𝑴)\operatorname\*{polar}(\bm{M}) aim to increase each singular value relative to the largest singular value; while σmin​(𝑿0)≪σmax​(𝑿0)\sigma\_{\min}(\bm{X}\_{0})\ll\sigma\_{\max}(\bm{X}\_{0}), after enough iterations, σmin​(𝑿t)≈σmax​(𝑿t)≈1\sigma\_{\min}(\bm{X}\_{t})\approx\sigma\_{\max}(\bm{X}\_{t})\approx 1.
However, the convergence of each singular value to σmax\sigma\_{\max} may not be monotonic.
Over the domain [ℓt,ut][\ell\_{t},u\_{t}], our optimal polynomial ptp\_{t} oscillates repeatedly between ℓt+1\ell\_{t+1} and ut+1u\_{t+1}, so some singular values that are near utu\_{t} may get mapped down to ℓt+1\ell\_{t+1}.
It so happens that this non-monotonicity—even at a single iteration—can cause loss of precision.
That is, problems occur if

|  |  |  |
| --- | --- | --- |
|  | pt​(σi)σi≪maxx∈[σmin,σmax]⁡pt​(x)σmax,\frac{p\_{t}(\sigma\_{i})}{\sigma\_{i}}\ll\frac{\max\limits\_{x\in[\sigma\_{\min},\sigma\_{\max}]}p\_{t}(x)}{\sigma\_{\max}}, |  |

where 0≤σmin≤σi≤σmax0\leq\sigma\_{\min}\leq\sigma\_{i}\leq\sigma\_{\max} are singular values of 𝑿t\bm{X}\_{t} [[34](#bib.bib34)]. In the extreme case pt​(σi)<0p\_{t}(\sigma\_{i})<0, the iith singular vector will change sign, casuing the method to converge to the polar factor of the wrong matrix.
Unlike Newton-Schulz, unscaled Newton, or QDWH, our method is affected by this loss of precision.

To mitigate this issue, [[11](#bib.bib11)] propose modifying their update polynomials to enforce a lower bound on the ratio pt​(σi)σi\frac{p\_{t}(\sigma\_{i})}{\sigma\_{i}}.
This issue only occurs when ℓt≪ut\ell\_{t}\ll u\_{t}; as ℓt→ut\ell\_{t}\to u\_{t}, our optimal polynomial approaches the Padé approximant and so pt​(x)x≥1\frac{p\_{t}(x)}{x}\geq 1 for all x∈[0,ut]x\in[0,u\_{t}].
We could fully solve the problem by using the Padé approximant instead of our optimal polynomial, but this would significantly slow down convergence.
Instead we compromise.
When ℓt≥ut/10\ell\_{t}\geq u\_{t}/10, we find that pt​(x)x≥0.236\frac{p\_{t}(x)}{x}\geq 0.236.
Therefore, whenever ℓt<ut/10\ell\_{t}<u\_{t}/10 we select the update rule as though ℓt=ut/10\ell\_{t}=u\_{t}/10.
This change slows convergence, but only very slightly.
(The choice of 10 is somewhat arbitrary. In [Appendix G](#A7 "Appendix G Code for Constructing Polynomials of Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"), we use a different factor.)

The third change is copied from the original Muon implementation: normalize 𝑴\bm{M} by ‖𝑴‖F+10−2\|\bm{M}\|\_{\text{F}}+10^{-2} instead of by ‖𝑴‖F\|\bm{M}\|\_{\text{F}}. As before, we set u1=1u\_{1}=1.

### 4.5  The algorithm

We give the complete pseudocode for our proposed method for any degree in [Algorithm 2](#alg2 "In 4.5 The algorithm ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"). We give the specfic version of the Polar Express with degree d=5d=5 and ℓ=10−3\ell=10^{-3} used in our GPT experiments in [Algorithm 1](#alg1 "In 1 Introduction ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm").
Our algorithm first computes the polynomials p1,…,pTp\_{1},\ldots,p\_{T} of [Theorem 4.1](#S4.Thmdefinition1 "Theorem 4.1. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") in full precision using the results of [Section 4.2](#S4.SS2 "4.2 Finding the optimal polynomial for each iteration ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") (or the Remez algorithm for d>5d>5).
This stage is offline because the coefficients of the polynomials are only computed and stored once.
For every subsequent call to the algorithm, these coefficients are reused and the offline stage is skipped. For instance, in [Algorithm 1](#alg1 "In 1 Introduction ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") these polynomials have been precomputed and stored in the variable coeffs\_list.

Algorithm 2  The General Polar Express

input: Matrix 𝑴\bm{M}, iteration count TT, degree dd, approximate lower bound ℓ\ell.
  
output: An approximation 𝑿T\bm{X}\_{T} to polar(𝑴)\operatorname\*{polar}(\bm{M}).

1:Offline stage: ⊳\triangleright In float64

2:ℓ1=ℓ\ell\_{1}=\ell, u1=1u\_{1}=1.

3:for t=1,2,…,Tt=1,2,\ldots,T do

4:  Solve using Remez [Appendix D](#A4 "Appendix D Remez algorithm ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"):

5:  pt=argminp∈ℙdoddmaxx∈[max⁡(ℓt,ut/10),ut]⁡|1−p​(x)|p\_{t}=\operatorname\*{argmin}\limits\_{p\in\mathbb{P}\_{d}^{\operatorname\*{odd}}}\max\limits\_{x\in\left[\max(\ell\_{t},u\_{t}/10),\,u\_{t}\right]}|1-p(x)|.

6:  pt←pt(⋅/1.01)p\_{t}\leftarrow p\_{t}(\cdot/1.01).

7:  ℓt+1=pt​(ℓt),ut+1=2−ℓt+1\ell\_{t+1}=p\_{t}(\ell\_{t}),u\_{t+1}=2-\ell\_{t+1}.

8:end for

9:Online stage: ⊳\triangleright In float16

10:Let 𝑿0=𝑴/(‖𝑴‖F+10−7)\bm{X}\_{0}=\bm{M}/(\|\bm{M}\|\_{\text{F}}+10^{-7}).

11:for t=1,2,…,Tt=1,2,\ldots,T do

12:  𝑿t=pt​(𝑿t−1)\bm{X}\_{t}=p\_{t}(\bm{X}\_{t-1}).

13:end for

14:return 𝑿T\bm{X}\_{T}.

The polynomial p⋆:=pT∘⋯∘p1p^{\star}:=p\_{T}\circ\cdots\circ p\_{1} is then applied to the input matrix 𝑴\bm{M} in the online stage.
The online stage can be performed in lower precision (bfloat16) for greater speed on a GPU.
Horner’s rule can be used to carry out each iteration. For instance, if pt=a​x+b​x3+c​x5p\_{t}=ax+bx^{3}+cx^{5}, then

|  |  |  |
| --- | --- | --- |
|  | 𝑿t=𝑿t−1​(a​𝑰+𝒀t−1​(b​𝑰+c​𝒀t−1))\bm{X}\_{t}=\bm{X}\_{t-1}\left(a\bm{I}+\bm{Y}\_{t-1}\left(b\bm{I}+c\bm{Y}\_{t-1}\right)\right) |  |

where 𝒀t−1=𝑿t−1⊤​𝑿t−1\bm{Y}\_{t-1}=\bm{X}\_{t-1}^{\top}\bm{X}\_{t-1}.

A simple implementation of the offline stage of [Algorithm 2](#alg2 "In 4.5 The algorithm ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") is given in [Appendix G](#A7 "Appendix G Code for Constructing Polynomials of Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm").
For deep learning applications, we recommend using d=5d=5 and T=5T=5 or 66 with ℓ1=10−3\ell\_{1}=10^{-3}.
With these parameters, the offline stage as implemented in [Appendix G](#A7 "Appendix G Code for Constructing Polynomials of Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") gives the polynomials
encoded in coeffs\_list in [Algorithm 1](#alg1 "In 1 Introduction ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm").
All told, our proposal for Muon is to apply the composition of these polynomials to 𝑴/(‖𝑴‖F+10−2)\bm{M}/(\|\bm{M}\|\_{F}+10^{-2}).

## 5  Numerical Experiments

### 5.1  Convergence of Polar Express

![Refer to caption](/html/2505.16932/assets/x4.png)


Figure 4: Convergence of various degree-5 polynomial methods in the spectral norm. When tuned properly, Polar Express attains outperforms the other methods at every iteration. Left panel: synthetic matrix with σmax=1\sigma\_{\max}=1, σmin=10−6\sigma\_{\min}=10^{-6}. Right panel: gradient of a certain weight matrix of a randomly-initialized GPT-2 architecture on a batch of language modeling data, normalized by the Frobenius norm.

We compare the performance of Polar Express against degree-5 Newton-Schulz and the methods of Chen and Chow [[11](#bib.bib11)], Jordan [[22](#bib.bib22)], and You [[9](#bib.bib9)].

We first study an idealized scenario where the spectrum of the input matrix is known exactly. We generate a random matrix whose singular values are evenly spaced on a logarithmic scale between 10−610^{-6} and 11. The right and left singular vectors are chosen at random.
The left panel of [Figure 4](#S5.F4 "In 5.1 Convergence of Polar Express ‣ 5 Numerical Experiments ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") shows the results.
Since all the methods in this plot use degree-5 polynomials, their computational and runtime costs are all proportional to the number of iterations.
As expected, Newton-Schulz converges but makes almost no progress for the first 17 iterations. Jordan’s method rapidly achieves an error of ≈0.3\approx 0.3 after just 11 iterations, but ceases to converge further.
You’s method, which is difficult to see on the plot because it is only defined for six iterations, converges at a similar rate as Jordan’s method.
When Polar Express is instantiated with ℓ=σmin\ell=\sigma\_{\min}, it dominates the other methods at every iteration, achieving excellent accuracy after just 11 iterations and converging about twice as fast as Newton-Schulz to any given error.
Even when the lower bound on σmin\sigma\_{\min} is wrong by two orders of magnitude in either direction, the method remains competitive, though it does not actually outperform Keller until iteration 13 or 14.

![Refer to caption](/html/2505.16932/assets/x5.png)


Figure 5: Convergence of polynomial methods in the Frobenius norm on GPT-2 gradient matrices. The number of matrix-matrix products is T​(d+1)/2T(d+1)/2, where dd is the degree (33 for Chen & Chow; 55 for all others) and TT is the number of iterations.

Next we test the methods’ performance on a matrix from a real-world application, namely, the gradient of a weight matrix from the fourth transformer block of a GPT-2 architecture with respect to a language modeling objective on a batch of text from the Tiny Shakespeare dataset [[24](#bib.bib24)].
The right panel of [Figure 4](#S5.F4 "In 5.1 Convergence of Polar Express ‣ 5 Numerical Experiments ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") shows the results.
Once again, the best-tuned version of Polar Express outperforms the other methods.
This time, we see that setting ℓ\ell to be many orders of magnitude too small can delay convergence significantly, and make Polar Express less competitive as compared to Jordan’s method.

For most other weight matrices in this GPT-2 model, the methods all take more than 10 iterations to converge in the spectral norm.
The spectral error is large if there is even one outlying singular value that is far from 11.
However, for some applications, we may be satisfied with a weaker notion of convergence, like the relative Frobenius norm.
[Figure 5](#S5.F5 "In 5.1 Convergence of Polar Express ‣ 5 Numerical Experiments ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") shows the performance of various methods on this metric.
We use gradient matrices of the same model, but from two different layers.
In addition, we compare the degree-5 methods to Chen and Chow’s degree-3 method.
To make this comparison fair, we measure the number of matrix-matrix products performed by each method instead the number of iterations.
We find that Polar Express can once again dominate the other methods across iterations. Chen and Chow’s method is also quite competitive, and the remaining methods behave much as in [Figure 4](#S5.F4 "In 5.1 Convergence of Polar Express ‣ 5 Numerical Experiments ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm").

### 5.2  Training GPT-2

In our final experiment, we compare the performance of using our Polar Express method given in [Algorithm 1](#alg1 "In 1 Introduction ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") inside the Muon algorithm versus Jordan’s [[22](#bib.bib22)] and You’s [[9](#bib.bib9)] methods.

Our experimental setup is based on the modified nanogpt code of Jordan [[21](#bib.bib21)].
We train two different GPT-2 models:

|  |  |  |  |
| --- | --- | --- | --- |
|  | GPT-Small:\displaystyle\texttt{GPT-Small}: | nembd=768,nlayer=12,nhead=12\displaystyle\quad n\_{\text{embd}}=768,\quad n\_{\text{layer}}=12,\quad n\_{\text{head}}=12 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | GPT-Large:\displaystyle\texttt{GPT-Large}: | nembd=1280,nlayer=36,nhead=20\displaystyle\quad n\_{\text{embd}}=1280,\quad n\_{\text{layer}}=36,\quad n\_{\text{head}}=20 |  |

and a vocabulary size of 50,25750{,}257, using a context length of 10241024. Training is performed on 1B tokens from the FineWeb dataset [[2](#bib.bib2)], using a batch size of 32 and a single epoch. All models are trained with mixed precision (bfloat16) on 4 H100 GPUs.
For all methods we use the learning rate schedule proposed in [[21](#bib.bib21)], consisting of a constant phase for the first 40% of training steps followed by a linear decay.
All methods for the matrix sign computations are performed in float16b precision and use five iterations.

We apply Muon selectively to certain layers of the model. Following the nano-gpt implementation [[21](#bib.bib21)], we assign Muon to all parameters with at least two dimensions (typically weight matrices, and excluding RMS norm parameters), excluding the embeddings, unembeddings, and the attention head. These excluded parameters are instead optimized with AdamW.

![Refer to caption](/html/2505.16932/assets/x6.png)

![Refer to caption](/html/2505.16932/assets/x7.png)

![Refer to caption](/html/2505.16932/assets/x8.png)

![Refer to caption](/html/2505.16932/assets/x9.png)

Figure 6: Training a GPT-2 (124M) model on 1 Billion tokens of the Fineweb data set [[2](#bib.bib2)]. The Legend muon-<name> refers to using muon with the <name> method for computing polar(𝑴)\operatorname\*{polar}(\bm{M}). Top Left: The final validation loss vs. the learning rate. The final best validation losses for each method were, in reverse order, adamw: 4.1974.197, muon-Jordan: 3.6393.639, muon-You: 3.6293.629 and muon-PolarExp: 3.5883.588. Bottom Left: The final training loss vs the learning rate. Top Right: Validation loss vs. number of iterations. Bottom Left: validation loss vs. time, plotting each method with its best learning rate.

[Figure 1](#S1.F1 "In 1 Introduction ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") and [Figure 6](#S5.F6 "In 5.2 Training GPT-2 ‣ 5 Numerical Experiments ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") shows the resulting runs of each method in terms of validation loss and training loss on the GPT-Large and GPT-Small models, respectively. In both figures we can see that muon-PolarExp achieves a better validation and training loss than muon-Jordan or muon-You for every learning rate. Since each iteration of the different matrix sign methods are equally expensive (since they all apply a degree 5 polynomial), improved validation loss in terms of epochs also translates to an improved loss in terms of wall clock time (see bottom right of [Figure 6](#S5.F6 "In 5.2 Training GPT-2 ‣ 5 Numerical Experiments ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")).
The advantage is remarkably consistent across all learning rates and epochs.

## References

* [1]

  N. I. Achieser.
  Theory of approximation.
  Dover Publications, Inc., New York, 1992.
  Translated from the Russian and with a preface by Charles J. Hyman, Reprint of the 1956 English translation.
* [2]

  Samuel Aroca-Ouellette, Philippe Beaudoin, Guillaume Lajoie, Liam Paull, Joelle Pineau, Pascal Vincent, and Anirudh Goyal.
  Fineweb: Learning language models with high quality web data.
  In NeurIPS Datasets and Benchmarks Track, 2023.
  URL: <https://arxiv.org/abs/2306.03061>.
* [3]

  Michele Benzi and Ru Huang.
  Some matrix properties preserved by generalized matrix functions.
  Spec. Matrices, 7:27–37, 2019.
  [doi:10.1515/spma-2019-0003](https://doi.org/10.1515/spma-2019-0003).
* [4]

  Jeremy Bernstein and Laker Newhouse.
  Modular duality in deep learning.
  arXiv preprint arXiv:2410.21265, 2024.
  URL: <https://arxiv.org/abs/2410.21265>.
* [5]

  Jeremy Bernstein and Laker Newhouse.
  Old optimizer, new norm: An anthology.
  arXiv preprint arXiv:2409.20325, 2024.
  URL: <https://arxiv.org/abs/2409.20325>.
* [6]

  Ȧ. Björck and C. Bowie.
  An iterative algorithm for computing the best estimate of an orthogonal matrix.
  SIAM J. Numer. Anal., 8:358–364, 1971.
  [doi:10.1137/0708036](https://doi.org/10.1137/0708036).
* [7]

  Christos Boutsikas, Petros Drineas, and Ilse C. F. Ipsen.
  Small singular values can increase in lower precision.
  SIAM J. Matrix Anal. Appl., 45(3):1518–1540, 2024.
  [doi:10.1137/23M1557209](https://doi.org/10.1137/23M1557209).
* [8]

  J Douglas Carroll and Phipps Arabie.
  Multidimensional scaling.
  pages 179–250, 1998.
  URL: <https://www.sciencedirect.com/science/article/pii/B9780120999750500051>, [doi:10.1016/B978-012099975-0.50005-1](https://doi.org/10.1016/B978-012099975-0.50005-1).
* [9]

  Franz Louis Cesista, You Jiacheng, and Keller Jordan.
  Squeezing 1-2% efficiency gains out of muon by optimizing the newton-schulz coefficients, 2025.
  URL: <http://leloykun.github.io/ponder/muon-opt-coeffs/>.
* [10]

  PL Chebyshev.
  Questions on smallest quantities connected with the approximate representation of functions (1859).
  Collected works, 2:151–235, 1947.
* [11]

  Jie Chen and Edmond Chow.
  A stable scaling of newton-schulz for improving the sign function computation of a hermitian matrix.
  Preprint ANL/MCS-P5059-0114, 2014.
  URL: <https://www.mcs.anl.gov/papers/P5059-0114.pdf>.
* [12]

  E. W. Cheney.
  Introduction to approximation theory.
  McGraw-Hill Book Co., New York-Toronto-London, 1966.
* [13]

  John Duchi, Elad Hazan, and Yoram Singer.
  Adaptive subgradient methods for online learning and stochastic optimization.
  J. Mach. Learn. Res., 12:2121–2159, 2011.
* [14]

  Alexandre Eremenko and Peter Yuditskii.
  Uniform approximation of sgn​x{\rm sgn}\,x by polynomials and entire functions.
  J. Anal. Math., 101:313–324, 2007.
  [doi:10.1007/s11854-007-0011-3](https://doi.org/10.1007/s11854-007-0011-3).
* [15]

  Gene H. Golub and Charles F. Van Loan.
  Matrix computations.
  Johns Hopkins Studies in the Mathematical Sciences. Johns Hopkins University Press, Baltimore, MD, fourth edition, 2013.
* [16]

  J. C. Gower and G. B. Dijksterhuis.
  Procrustes problems, volume 30 of Oxford Statistical Science Series.
  Oxford University Press, Oxford, 2004.
  [doi:10.1093/acprof:oso/9780198510581.001.0001](https://doi.org/10.1093/acprof:oso/9780198510581.001.0001).
* [17]

  Vineet Gupta, Tomer Koren, and Yoram Singer.
  Shampoo: Preconditioned stochastic tensor optimization.
  In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 1842–1850. PMLR, 10–15 Jul 2018.
  URL: <https://proceedings.mlr.press/v80/gupta18a.html>.
* [18]

  Nicholas J. Higham.
  Computing the polar decomposition—with applications.
  SIAM J. Sci. Statist. Comput., 7(4):1160–1174, 1986.
  [doi:10.1137/0907079](https://doi.org/10.1137/0907079).
* [19]

  Nicholas J. Higham.
  Functions of matrices.
  SIAM, Philadelphia, PA, 2008.
  [doi:10.1137/1.9780898717778](https://doi.org/10.1137/1.9780898717778).
* [20]

  Elias Jarlebring and Gustaf Lorentzon.
  The polynomial set associated with a fixed number of matrix-matrix multiplications.
  arXiv preprint arXiv:2504.01500, 2025.
  URL: <https://arxiv.org/abs/2504.01500>.
* [21]

  Keller Jordan, Jeremy Bernstein, Brendan Rappazzo, @fernbear.bsky.social, Boza Vlado, You Jiacheng, Franz Cesista, Braden Koszarsky, and @Grad62304977.
  modded-nanogpt: Speedrunning the nanogpt baseline, 2024.
  URL: <https://github.com/KellerJordan/modded-nanogpt>.
* [22]

  Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein.
  Muon: An optimizer for hidden layers in neural networks, 2024.
  URL: <https://kellerjordan.github.io/posts/muon/>.
* [23]

  Tetsuya Kaneko, Simone Fiori, and Toshihisa Tanaka.
  Empirical arithmetic averaging over the compact Stiefel manifold.
  IEEE Trans. Signal Process., 61(4):883–894, 2013.
  [doi:10.1109/TSP.2012.2226167](https://doi.org/10.1109/TSP.2012.2226167).
* [24]

  Andrej Karpathy.
  char-rnn.
  <https://github.com/karpathy/char-rnn>, 2015.
* [25]

  Charles Kenney and Alan J. Laub.
  Rational iterative methods for the matrix sign function.
  SIAM J. Matrix Anal. Appl., 12(2):273–291, 1991.
  [doi:10.1137/0612020](https://doi.org/10.1137/0612020).
* [26]

  Diederik P. Kingma and Jimmy Ba.
  Adam: A method for stochastic optimization.
  In International Conference on Learning Representations, 2015.
  URL: <http://arxiv.org/abs/1412.6980>.
* [27]

  Zdislav Kovářík.
  Some iterative methods for improving orthonormality.
  SIAM J. Numer. Anal., 7:386–389, 1970.
  [doi:10.1137/0707031](https://doi.org/10.1137/0707031).
* [28]

  Eunsang Lee, Joon-Woo Lee, Jong-Seon No, and Young-Sik Kim.
  Minimax approximation of sign function by composite polynomial for homomorphic comparison.
  IEEE Transactions on Dependable and Secure Computing, 19(6):3711–3727, 2022.
  [doi:10.1109/TDSC.2021.3105111](https://doi.org/10.1109/TDSC.2021.3105111).
* [29]

  R. B. Leipnik.
  Rapidly convergent recursive solution of quadratic operator equations.
  Numer. Math., 17:1–16, 1971.
  [doi:10.1007/BF01395861](https://doi.org/10.1007/BF01395861).
* [30]

  Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, et al.
  Muon is scalable for LLM training.
  arXiv preprint arXiv:2502.16982, 2025.
  URL: <https://arxiv.org/abs/2502.16982>.
* [31]

  Ilya Loshchilov and Frank Hutter.
  Decoupled weight decay regularization.
  In International Conference on Learning Representations, 2019.
  URL: <https://openreview.net/forum?id=Bkg6RiCqY7>.
* [32]

  Yuji Nakatsukasa, Zhaojun Bai, and François Gygi.
  Optimizing Halley’s iteration for computing the matrix polar decomposition.
  SIAM J. Matrix Anal. Appl., 31(5):2700–2720, 2010.
  [doi:10.1137/090774999](https://doi.org/10.1137/090774999).
* [33]

  Yuji Nakatsukasa and Roland W. Freund.
  Computing fundamental matrix decompositions accurately via the matrix sign function in two iterations: the power of Zolotarev’s functions.
  SIAM Rev., 58(3):461–493, 2016.
  [doi:10.1137/140990334](https://doi.org/10.1137/140990334).
* [34]

  Yuji Nakatsukasa and Nicholas J. Higham.
  Backward stability of iterations for computing the polar decomposition.
  SIAM J. Matrix Anal. Appl., 33(2):460–479, 2012.
  [doi:10.1137/110857544](https://doi.org/10.1137/110857544).
* [35]

  Yuji Nakatsukasa and Nicholas J. Higham.
  Stable and efficient spectral divide and conquer algorithms for the symmetric eigenvalue decomposition and the SVD.
  SIAM J. Sci. Comput., 35(3):A1325–A1349, 2013.
  [doi:10.1137/120876605](https://doi.org/10.1137/120876605).
* [36]

  Herbert Neuberger.
  Exactly massless quarks on the lattice.
  Phys. Lett. B, 417(1-2):141–144, 1998.
  [doi:10.1016/S0370-2693(97)01368-3](https://doi.org/10.1016/S0370-2693(97)01368-3).
* [37]

  Ricardo Pachón and Lloyd N. Trefethen.
  Barycentric-Remez algorithms for best polynomial approximation in the chebfun system.
  BIT, 49(4):721–741, 2009.
  [doi:10.1007/s10543-009-0240-1](https://doi.org/10.1007/s10543-009-0240-1).
* [38]

  T Parks and James McClellan.
  Chebyshev approximation for nonrecursive digital filters with linear phase.
  IEEE Transactions on circuit theory, 19(2):189–194, 1972.
  [doi:10.1109/TCT.1972.1083419](https://doi.org/10.1109/TCT.1972.1083419).
* [39]

  Michael S. Paterson and Larry J. Stockmeyer.
  On the number of nonscalar multiplications necessary to evaluate polynomials.
  SIAM J. Comput., 2:60–66, 1973.
  [doi:10.1137/0202007](https://doi.org/10.1137/0202007).
* [40]

  Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, Zhenyu Zhu, Antonio Silveti-Falls, and Volkan Cevher.
  Training deep learning models with norm-constrained lmos, 2025.
  URL: <https://arxiv.org/abs/2502.07529>, [arXiv:2502.07529](https://arxiv.org/abs/2502.07529).
* [41]

  Artem Riabinin, Egor Shulgin, Kaja Gruntkowska, and Peter Richtárik.
  Gluon: Making muon & scion great again! (bridging theory and practice of lmo-based optimizers for llms), 2025.
  URL: <https://arxiv.org/abs/2505.13416>, [arXiv:2505.13416](https://arxiv.org/abs/2505.13416).
* [42]

  Ishaan Shah, Anthony M Polloreno, Karl Stratos, Philip Monk, Adarsh Chaluvaraju, Andrew Hojel, Andrew Ma, Anil Thomas, Ashish Tanwer, Darsh J Shah, et al.
  Practical efficiency of muon for pretraining.
  arXiv preprint arXiv:2505.02222, 2025.
  URL: <https://arxiv.org/abs/2505.02222>.
* [43]

  Hao-Jun Michael Shi, Tsung-Hsien Lee, Shintaro Iwasaki, Jose Gallego-Posada, Zhijing Li, Kaushik Rangadurai, Dheevatsa Mudigere, and Michael Rabbat.
  A distributed data-parallel PyTorch implementation of the distributed Shampoo optimizer for training neural networks at-scale.
  arXiv preprint arXiv:2309.06497, 2023.
  URL: <https://arxiv.org/abs/2309.06497>.
* [44]

  Modula Systems.
  Newton-schulz algorithm — jiacheng’s six-step method.
  <https://docs.modula.systems/algorithms/newton-schulz/#jiacheng-s-six-step>, 2024.
  Accessed: 2025-05-19.
* [45]

  Attila Szabo and Neil S Ostlund.
  Modern quantum chemistry: introduction to advanced electronic structure theory.
  Courier Corporation, 1996.
* [46]

  Lloyd N. Trefethen.
  Approximation theory and approximation practice.
  Society for Industrial and Applied Mathematics (SIAM), Philadelphia, PA, extended edition, 2020.
* [47]

  Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, and Sham M. Kakade.
  SOAP: Improving and stabilizing shampoo using adam for language modeling.
  In The Thirteenth International Conference on Learning Representations, 2025.
  URL: <https://openreview.net/forum?id=IDxZhXrpNf>.
* [48]

  Greg Yang, James B. Simon, and Jeremy Bernstein.
  A spectral condition for feature learning, 2024.
  URL: <https://arxiv.org/abs/2310.17813>, [arXiv:2310.17813](https://arxiv.org/abs/2310.17813).
* [49]

  Zhenyue Zhang, Hongyuan Zha, and Wenlong Ying.
  Fast parallelizable methods for computing invariant subspaces of Hermitian matrices.
  J. Comput. Math., 25(5):583–594, 2007.
  URL: <http://www.jstor.org/stable/43693395>.

## Appendix A Proof of [Theorem 4.1](#S4.Thmdefinition1 "Theorem 4.1. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")

The aim of this section is to prove [Theorem 4.1](#S4.Thmdefinition1 "Theorem 4.1. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"). We begin with a result that provides a few essential properties for the the polynomial solving ([7](#S3.E7 "Equation 7 ‣ 3 Approximations by Compositions of Polynomials ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")) when T=1T=1. This result is known as Chebyshev’s theorem [[10](#bib.bib10)] or the equioscillation theorem [[46](#bib.bib46), Chapter 10].

###### Lemma A.1.

Let d=2​q+1d=2q+1 and u,ℓ>0u,\ell>0. Consider the problem

|  |  |  |  |
| --- | --- | --- | --- |
|  | minp∈ℙdodd⁡maxx∈[ℓ,u]⁡|1−p​(x)|.\min\limits\_{p\in\mathbb{P}\_{d}^{\operatorname\*{odd}}}\max\limits\_{x\in[\ell,u]}|1-p(x)|. |  | (16) |

There exists a unique polynomial p⋆∈ℙdoddp^{\star}\in\mathbb{P}\_{d}^{\operatorname\*{odd}} solving ([16](#A1.E16 "Equation 16 ‣ Lemma A.1. ‣ Appendix A Proof of Theorem 4.1 ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")). Furthermore, p⋆p^{\star} is the unique solution to the above problem if and only if there exist q+2q+2 distinct points {x0,…,xq+1}⊂[ℓ,u]\{x\_{0},\ldots,x\_{q+1}\}\subset[\ell,u] such that

|  |  |  |
| --- | --- | --- |
|  | 1−p⋆​(xi)=η​(−1)i​maxx∈[ℓ,u]⁡|1−p⋆​(x)|,for​i=0,…,q+1,1-p^{\star}(x\_{i})\;=\;\eta(-1)^{i}\max\limits\_{x\in[\ell,u]}|1-p^{\star}(x)|,\quad\mbox{for}\;i=0,\ldots,q+1, |  |

for η=1\eta=1 or η=−1\eta=-1.

###### Proof.

A discussion can be found in [[14](#bib.bib14)]. Here we include a formal proof for completeness.

By Chebyshev’s Theorem [[1](#bib.bib1), [10](#bib.bib10), [12](#bib.bib12)] it is sufficient to show that ℙdodd\mathbb{P}\_{d}^{\operatorname\*{odd}} satisfies the Haar condition: any non-zero p∈ℙdodd=span​{x,…,x3,…,x2​q+1}p\in\mathbb{P}\_{d}^{\operatorname\*{odd}}=\mbox{span}\{x,\ldots,x^{3},\ldots,x^{2q+1}\} can have at most qq roots in [ℓ,u][\ell,u].

Since deg⁡(p)=d=2​q+1\deg(p)=d=2q+1 we know that pp can have at most 2​q+12q+1 roots in ℝ\mathbb{R}. However, since p​(0)=0p(0)=0 and p​(x)=−p​(−x)p(x)=-p(-x) we know that pp has one root at zero, and the remaining roots come in symmetric pairs (x,−x)(x,-x). Because of this, pp can have at most qq roots in the positive orthant, and thus it can have at most qq roots in [ℓ,u]⊂(0,∞)[\ell,u]\subset(0,\infty). Hence, ℙdodd\mathbb{P}\_{d}^{\operatorname\*{odd}} satisfies the Haar condition, which yields the desired result.

∎

The proof of [Theorem 4.1](#S4.Thmdefinition1 "Theorem 4.1. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") will be by induction on TT. We begin by establishing the base case, T=1T=1, which is handled by the following result.

###### Lemma A.2.

Let u,ℓ>0u,\ell>0 and define

|  |  |  |
| --- | --- | --- |
|  | p⋆:=argminp∈ℙd∗maxx∈[ℓ,u]⁡|1−p​(x)|.p^{\star}:=\operatorname\*{argmin}\limits\_{p\in\mathbb{P}\_{d}^{\*}}\max\limits\_{x\in[\ell,u]}|1-p(x)|. |  |

Then

|  |  |  |
| --- | --- | --- |
|  | p⋆​(ℓ)=minx∈[ℓ,u]⁡p⋆​(x),maxx∈[ℓ,u]⁡p⋆​(x)=2−p⋆​(ℓ), and ​maxx∈[ℓ,u]⁡|1−p⋆​(x)|=1−p⋆​(ℓ).p^{\star}(\ell)=\min\limits\_{x\in[\ell,u]}p^{\star}(x),\quad\max\limits\_{x\in[\ell,u]}p^{\star}(x)=2-p^{\star}(\ell),\text{ and }\max\limits\_{x\in[\ell,u]}|1-p^{\star}(x)|=1-p^{\star}(\ell). |  |

###### Proof.

Throughout the proof we assume d=2​q+1d=2q+1. We begin with proving

|  |  |  |
| --- | --- | --- |
|  | p⋆​(ℓ)=minx∈[ℓ,u]⁡p⋆​(x).p^{\star}(\ell)=\min\limits\_{x\in[\ell,u]}p^{\star}(x). |  |

Consider the polynomial e​(x):=1−p⋆​(x)e(x):=1-p^{\star}(x). The proof will contain three steps. We first rule out the trivial case that p⋆≠0p^{\star}\neq 0, since p​(x)=2ℓ+u​xp(x)=\frac{2}{\ell+u}x would then be a better approximation. Hence, p⋆p^{\star} cannot be the zero polynomial.

*Step 1: e​(x)e(x) has exactly qq stationary points inside the open interval (ℓ,u)(\ell,u).*

Note that e​(x)e(x) has at most 2​q2q stationary points in ℝ\mathbb{R}, since its derivative e′​(x)e^{\prime}(x) is a polynomial of degree 2​q2q. Furthermore, since p⋆p^{\star} is odd, we have that e′​(x)=−p′​(x)e^{\prime}(x)=-p^{\prime}(x) is even of degree 2​q2q, and thus can have at most qq stationary points contained in (0,+∞)(0,+\infty). Hence, there can be at *most* qq stationary points of e​(x)e(x) inside the interval [ℓ,u][\ell,u].

By [Lemma A.1](#A1.Thmdefinition1 "Lemma A.1. ‣ Appendix A Proof of Theorem 4.1 ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") there are q+2q+2 points x0,…,xq+1∈[ℓ,u]x\_{0},\ldots,x\_{q+1}\in[\ell,u] where e​(x)e(x) is maximized or minimized in [ℓ,u][\ell,u]. These points are either stationary points or they are endpoints of the interval [ℓ,u][\ell,u]. Let nextn\_{\text{ext}} be the number of stationary points and nstatn\_{\text{stat}} be the number of endpoints in the set {x0,…,xq+1}\{x\_{0},\ldots,x\_{q+1}\}. Since a point can be both a stationary point and an endpoint we have q+2≤nend+nstatq+2\leq n\_{\text{end}}+n\_{\text{stat}}. However, nend≤2n\_{\text{end}}\leq 2 and nstat≤qn\_{\text{stat}}\leq q, which follows from the previous paragraph where we showed that there are at most qq stationary points of e​(x)e(x) in [ℓ,u][\ell,u]. So nend+nstat≤q+2n\_{\text{end}}+n\_{\text{stat}}\leq q+2, and consequently we must have nend=2n\_{\text{end}}=2 and nstat=qn\_{\text{stat}}=q, as required.

*Step 2: x=ℓx=\ell is a maximum of e​(x)e(x) on the interval [ℓ,u][\ell,u]*

By [Lemma A.1](#A1.Thmdefinition1 "Lemma A.1. ‣ Appendix A Proof of Theorem 4.1 ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") and the discussion from Step 1, we know that |e​(x)||e(x)| is maximized at q+2q+2 points inside [ℓ,u][\ell,u] and qq of these points are contained inside the open interval (ℓ,u)(\ell,u). Hence, x=ℓx=\ell must either be a maximum or a minimum of e​(x)e(x). We will show that x=ℓx=\ell must be a maximum by contradiction.

Suppose x=ℓx=\ell was a minimum of e​(x)e(x) on [ℓ,u][\ell,u]. First note that p⋆p^{\star} is trivially non-negative on [ℓ,u][\ell,u], or else p​(x)=0p(x)=0 would be a better polynomial. Hence, since p⋆​(0)=0p^{\star}(0)=0 we must have p∗′​(δ)>0{p^{\*}}^{\prime}(\delta)>0 for some δ∈[0,ℓ]\delta\in[0,\ell], or else the zero polynomial p​(x)=0p(x)=0 would be a better approximation. Hence, for some δ∈[0,ℓ]\delta\in[0,\ell] we have e′​(δ)<0e^{\prime}(\delta)<0.

We must also have e′​(ℓ)≥0e^{\prime}(\ell)\geq 0 or else x=ℓx=\ell is not a minimum of e​(x)e(x). Since e′​(δ)<0e^{\prime}(\delta)<0 for some δ∈[0,ℓ]\delta\in[0,\ell] and e′​(ℓ)≥0e^{\prime}(\ell)\geq 0, by the intermediate value theorem there exists a point x∗∈[0,ℓ]x^{\*}\in[0,\ell] such that e′​(x∗)=0e^{\prime}(x^{\*})=0. However, by the discussion above we know that all stationary points of ee are contained inside the open interval (ℓ,u)(\ell,u). Hence, x=ℓx=\ell cannot be a minimum of e​(x)e(x) on [ℓ,u][\ell,u]. However, by Step 1 we know that the endpoints of [ℓ,u][\ell,u] must be either minima or maxima of e​(x)e(x). Hence, x=ℓx=\ell is a maximum of e​(x)e(x) on [ℓ,u][\ell,u].

*Step 3: Obtaining the desired equalities*

Since e​(x)e(x) has a maximum in [ℓ,u][\ell,u] at x=ℓx=\ell, we have p⋆​(ℓ)=minx∈[ℓ,u]⁡p⋆​(x)p^{\star}(\ell)=\min\limits\_{x\in[\ell,u]}p^{\star}(x). The other two equalities are immediate consequences of the equioscillation property of p⋆p^{\star} [Lemma A.1](#A1.Thmdefinition1 "Lemma A.1. ‣ Appendix A Proof of Theorem 4.1 ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") and that x=ℓx=\ell is a minimum of p⋆p^{\star} over the set [ℓ,u][\ell,u].
∎

With the above-mentioned result in hand, we are ready to prove [Theorem 4.1](#S4.Thmdefinition1 "Theorem 4.1. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm").

See [4.1](#S4.Thmdefinition1 "Theorem 4.1. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")

###### Proof.

The proof of ([10](#S4.E10 "Equation 10 ‣ Theorem 4.1. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")) is an immediate consequence of [Lemma A.2](#A1.Thmdefinition2 "Lemma A.2. ‣ Appendix A Proof of Theorem 4.1 ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"), since for each t=1,…,Tt=1,\ldots,T, ptp\_{t} is the optimal approximation in ℙdodd\mathbb{P}\_{d}^{\operatorname\*{odd}} to x↦1x\mapsto 1.

We now proceed with the proof of ([11](#S4.E11 "Equation 11 ‣ Theorem 4.1. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")), which will be by induction. The proof for T=1T=1 is an immediate consequence of [Lemma A.2](#A1.Thmdefinition2 "Lemma A.2. ‣ Appendix A Proof of Theorem 4.1 ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") and we also have p⋆​(ℓ)=ℓ2p^{\star}(\ell)=\ell\_{2} by ([10](#S4.E10 "Equation 10 ‣ Theorem 4.1. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")). Now suppose the result is true for all t≤T−1t\leq T-1. For t=1,…,T−1t=1,\ldots,T-1, note that the image of ptp\_{t} on [ℓt,ut][\ell\_{t},u\_{t}] is exactly [ℓt+1,ut+1][\ell\_{t+1},u\_{t+1}] by i). Hence, if we define g​(x):=pT−1∘⋯∘p1​(x)g(x):=p\_{T-1}\circ\cdots\circ p\_{1}(x), then the image of gg on [ℓ,u][\ell,u] is [ℓT,uT][\ell\_{T},u\_{T}]. Furthermore, by i) we also have g​(ℓ)=ℓTg(\ell)=\ell\_{T}. Pick any ff such that f≠gf\neq g and

|  |  |  |
| --- | --- | --- |
|  | f=p~T−1∘⋯∘p~1,f=\widetilde{p}\_{T-1}\circ\cdots\circ\widetilde{p}\_{1}, |  |

for some p~1,…,p~T−1∈ℙdodd\widetilde{p}\_{1},\ldots,\widetilde{p}\_{T-1}\in\mathbb{P}\_{d}^{\operatorname\*{odd}}. Let the image of ff on [ℓ,u][\ell,u] be [a,b][a,b]. We will prove that ab≤ℓTuT\frac{a}{b}\leq\frac{\ell\_{T}}{u\_{T}} by contradiction.

Suppose ab>ℓTuT\frac{a}{b}>\frac{\ell\_{T}}{u\_{T}}.
Define c=2a+bc=\frac{2}{a+b}. Then, the image of the scaled function c​fcf on [ℓ,u][\ell,u] is [c​a,c​b][ca,cb] and c​fcf satisfies

|  |  |  |
| --- | --- | --- |
|  | maxx∈[ℓ,u]⁡|1−c​f​(x)|=max⁡{1−c​a,c​b−1}=b−aa+b.\max\limits\_{x\in[\ell,u]}|1-cf(x)|=\max\left\{1-ca,cb-1\right\}=\frac{b-a}{a+b}. |  |

Recall by our inductive hypothesis, we have maxx∈[ℓ,u]⁡|1−g​(x)|=1−ℓT=uT−1\max\limits\_{x\in[\ell,u]}|1-g(x)|=1-\ell\_{T}=u\_{T}-1 where the second equality holds by ([10](#S4.E10 "Equation 10 ‣ Theorem 4.1. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")). It follows that

|  |  |  |  |
| --- | --- | --- | --- |
|  | ab\displaystyle\frac{a}{b} | >ℓTuT\displaystyle>\frac{\ell\_{T}}{u\_{T}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⇔ab\displaystyle\Leftrightarrow\frac{a}{b} | >ℓT2−ℓT\displaystyle>\frac{\ell\_{T}}{2-\ell\_{T}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⇔ℓT\displaystyle\Leftrightarrow\ell\_{T} | <2​aa+b\displaystyle<\frac{2a}{a+b} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⇔1−ℓT\displaystyle\Leftrightarrow 1-\ell\_{T} | >b−aa+b\displaystyle>\frac{b-a}{a+b} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⇔maxx∈[ℓ,u]⁡|1−g​(x)|\displaystyle\Leftrightarrow\max\limits\_{x\in[\ell,u]}|1-g(x)| | >maxx∈[ℓ,u]⁡|1−c​f​(x)|,\displaystyle>\max\limits\_{x\in[\ell,u]}|1-cf(x)|, |  |

which leads to a contradiction to our inductive hypothesis that gg is optimal. Hence, we must have ab≤ℓTuT\frac{a}{b}\leq\frac{\ell\_{T}}{u\_{T}}.

Consequently, using that ab≤ℓTuT\frac{a}{b}\leq\frac{\ell\_{T}}{u\_{T}}, we will show that for any p~T∈ℙdodd\widetilde{p}\_{T}\in\mathbb{P}\_{d}^{\operatorname\*{odd}} and for any f=p~T−1∘⋯∘p~1f=\widetilde{p}\_{T-1}\circ\cdots\circ\widetilde{p}\_{1} p~T∘f\widetilde{p}\_{T}\circ f cannot be a better approximation than pT∘gp\_{T}\circ g. In particular, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | maxx∈[ℓ,u]⁡|1−p~T​(f​(x))|\displaystyle\max\limits\_{x\in[\ell,u]}|1-\widetilde{p}\_{T}(f(x))| | ≥minp∈ℙd∗⁡maxx∈[ℓ,u]⁡|1−p​(f​(x))|\displaystyle\geq\min\limits\_{p\in\mathbb{P}\_{d}^{\*}}\max\limits\_{x\in[\ell,u]}|1-p(f(x))| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =minp∈ℙd∗⁡maxx∈[a,b]⁡|1−p​(x)|\displaystyle=\min\limits\_{p\in\mathbb{P}\_{d}^{\*}}\max\limits\_{x\in[a,b]}|1-p(x)| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =minp∈ℙd∗⁡maxx∈[a/b,1]⁡|1−p​(x)|\displaystyle=\min\limits\_{p\in\mathbb{P}\_{d}^{\*}}\max\limits\_{x\in[a/b,1]}|1-p(x)| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≥minp∈ℙd∗⁡maxx∈[ℓT/uT,1]⁡|1−p​(x)|\displaystyle\geq\min\limits\_{p\in\mathbb{P}\_{d}^{\*}}\max\limits\_{x\in[\ell\_{T}/u\_{T},1]}|1-p(x)| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =minp∈ℙd∗⁡maxx∈[ℓT,uT]⁡|1−p​(x)|\displaystyle=\min\limits\_{p\in\mathbb{P}\_{d}^{\*}}\max\limits\_{x\in[\ell\_{T},u\_{T}]}|1-p(x)| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =minp∈ℙd∗⁡maxx∈[ℓ,u]⁡|1−p​(g​(x))|\displaystyle=\min\limits\_{p\in\mathbb{P}\_{d}^{\*}}\max\limits\_{x\in[\ell,u]}|1-p(g(x))| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =maxx∈[ℓT,uT]⁡|1−pT​(g​(x))|=1−pT​(ℓT)=1−ℓT+1,\displaystyle=\max\limits\_{x\in[\ell\_{T},u\_{T}]}|1-p\_{T}(g(x))|=1-p\_{T}(\ell\_{T})=1-\ell\_{T+1}, |  |

where the second and third equality follow by changing variables y=x/by=x/b so that

|  |  |  |
| --- | --- | --- |
|  | minp∈ℙd∗⁡maxx∈[a,b]⁡|1−p​(x)|=minp∈ℙd∗⁡maxy∈[a/b,1]⁡|1−p​(b​y)|=minp∈ℙd∗⁡maxy∈[a/b,1]⁡|1−p​(y)|\min\limits\_{p\in\mathbb{P}\_{d}^{\*}}\max\limits\_{x\in[a,b]}|1-p(x)|=\min\limits\_{p\in\mathbb{P}\_{d}^{\*}}\max\limits\_{y\in[a/b,1]}|1-p(by)|=\min\limits\_{p\in\mathbb{P}\_{d}^{\*}}\max\limits\_{y\in[a/b,1]}|1-p(y)| |  |

and this last equality follows
because the space ℙd∗\mathbb{P}\_{d}^{\*} is invariant under input rescaling; that is, for any b≠0b\neq 0, the map x↦b​xx\mapsto bx preserves the space span​{x,x3,…,xd}\mathrm{span}\{x,x^{3},\dots,x^{d}\}. This concludes the proof.
∎

## Appendix B Proof of [Theorem 4.3](#S4.Thmdefinition3 "Theorem 4.3. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")

In this section we provide the proof of the convergence guarantee stated in [Theorem 4.3](#S4.Thmdefinition3 "Theorem 4.3. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm").

See [4.3](#S4.Thmdefinition3 "Theorem 4.3. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")

###### Proof.

Define

|  |  |  |
| --- | --- | --- |
|  | p⋆=argminp=pT∘pT−1∘⋯∘p1pt∈ℙd∗maxx∈[ℓ,u]⁡|1−p​(x)|.p^{\star}=\operatorname\*{argmin}\_{\begin{subarray}{c}p=p\_{T}\circ p\_{T-1}\circ\cdots\circ p\_{1}\\ p\_{t}\in\mathbb{P}\_{d}^{\*}\end{subarray}}\,\max\_{x\in[\ell,u]}\left|1-p(x)\right|. |  |

Then [Algorithm 2](#alg2 "In 4.5 The algorithm ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") returns 𝑿T=p⋆​(𝑴)\bm{X}\_{T}=p^{\star}(\bm{M}). Let h∈ℙqh\in\mathbb{P}\_{q} be [q/0][q/0] Padé-approximant to (1−x)−1/2(1-x)^{-1/2} [[25](#bib.bib25), Section 3] and define p​(x)=x​h​(1−x2)∈ℙdoddp(x)=xh(1-x^{2})\in\mathbb{P}\_{d}^{\operatorname\*{odd}}. Define f=p∘⋯∘pf=p\circ\cdots\circ p as the composition of pp with itself TT times. Then, by [Theorem 4.1](#S4.Thmdefinition1 "Theorem 4.1. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") and [[25](#bib.bib25), Theorem 3.1] we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖sign(𝑴)−𝑿T‖2\displaystyle\|\operatorname\*{sign}(\bm{M})-\bm{X}\_{T}\|\_{2} | ≤maxx∈[ℓ,1]⁡|1−p⋆​(x)|\displaystyle\leq\max\limits\_{x\in[\ell,1]}|1-p^{\star}(x)| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤maxx∈[ℓ,1]⁡|1−f​(x)|\displaystyle\leq\max\limits\_{x\in[\ell,1]}|1-f(x)| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤maxx∈[ℓ,1]⁡[|1−x2|(d+1)T1+f​(x)]\displaystyle\leq\max\limits\_{x\in[\ell,1]}\left[\frac{|1-x^{2}|^{(d+1)^{T}}}{1+f(x)}\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤|1−ℓ2|(d+1)T,\displaystyle\leq|1-\ell^{2}|^{(d+1)^{T}}, |  |

as required.
∎

## Appendix C Proof of equivalence between ([6](#S3.E6 "Equation 6 ‣ 3 Approximations by Compositions of Polynomials ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")) and ([7](#S3.E7 "Equation 7 ‣ 3 Approximations by Compositions of Polynomials ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"))

In this section we provide a proof for the equivalence between ([6](#S3.E6 "Equation 6 ‣ 3 Approximations by Compositions of Polynomials ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")) and ([7](#S3.E7 "Equation 7 ‣ 3 Approximations by Compositions of Polynomials ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")). It is sufficient to show that for any fixed polynomial pp we have

|  |  |  |
| --- | --- | --- |
|  | ε1:=max𝑴∈ℝm×nσ​(𝑴)⊂[ℓ,u]⁡‖polar(𝑴)−p​(𝑴)‖2=maxx∈[ℓ,u]⁡|1−p​(x)|:=ε2.\varepsilon\_{1}:=\max\_{\begin{subarray}{c}\bm{M}\in\mathbb{R}^{m\times n}\\ \sigma(\bm{M})\subset[\ell,u]\end{subarray}}\left\|\operatorname\*{polar}(\bm{M})-p(\bm{M})\right\|\_{2}=\max\_{x\in[\ell,u]}\left|1-p(x)\right|:=\varepsilon\_{2}. |  |

For any fixed 𝑴\bm{M}, by the unitary invariance of the spectral norm we immediately have

|  |  |  |
| --- | --- | --- |
|  | ‖polar(𝑴)−p​(𝑴)‖2=maxσi∈σ​(𝑴)⁡|1−p​(σi)|≤maxx∈[ℓ,u]⁡|1−p​(x)|.\left\|\operatorname\*{polar}(\bm{M})-p(\bm{M})\right\|\_{2}=\max\limits\_{\sigma\_{i}\in\sigma(\bm{M})}|1-p(\sigma\_{i})|\leq\max\limits\_{x\in[\ell,u]}\left|1-p(x)\right|. |  |

Consequently, ε1≤ε2\varepsilon\_{1}\leq\varepsilon\_{2}.

Suppose that x∗∈[ℓ,u]x^{\*}\in[\ell,u] is chosen so that |1−p​(x∗)|=maxx∈[ℓ,u]⁡|1−p​(x)|.|1-p(x^{\*})|=\max\_{x\in[\ell,u]}\left|1-p(x)\right|. Without loss of generality, assume m≥nm\geq n. Letting 𝑴=x∗​𝑼​𝑽𝖳\bm{M}=x^{\*}\bm{U}\bm{V}^{\mathsf{T}}, for any matrix 𝑼∈ℝm×n\bm{U}\in\mathbb{R}^{m\times n} and 𝑽∈ℝn×n\bm{V}\in\mathbb{R}^{n\times n} with orthonormal columns, and noting polar(𝑴)=𝑼​𝑽𝖳\operatorname\*{polar}(\bm{M})=\bm{U}\bm{V}^{\mathsf{T}} yields

|  |  |  |  |
| --- | --- | --- | --- |
|  | ε1\displaystyle\varepsilon\_{1} | ≥‖polar(𝑴)−p​(𝑴)‖2\displaystyle\geq\|\operatorname\*{polar}(\bm{M})-p(\bm{M})\|\_{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =‖𝑰n−p​(x∗)​𝑰n‖2\displaystyle=\|\bm{I}\_{n}-p(x^{\*})\bm{I}\_{n}\|\_{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =|1−p​(x∗)|\displaystyle=|1-p(x^{\*})| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =maxx∈[ℓ,u]⁡|1−p​(x)|=ε2\displaystyle=\max\_{x\in[\ell,u]}\left|1-p(x)\right|\;=\varepsilon\_{2} |  |

Consequently, ε1≥ε2\varepsilon\_{1}\geq\varepsilon\_{2}. Hence, ε1=ε2\varepsilon\_{1}=\varepsilon\_{2}, as desired.

## Appendix D Remez algorithm

[Theorem 4.1](#S4.Thmdefinition1 "Theorem 4.1. ‣ 4.1 Greedy is optimal ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") shows that we can solve ([7](#S3.E7 "Equation 7 ‣ 3 Approximations by Compositions of Polynomials ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")) by greedily choosing the optimal approximation pt∈ℙdoddp\_{t}\in\mathbb{P}\_{d}^{\operatorname\*{odd}} for each interval [ℓt,ut][\ell\_{t},u\_{t}] for t=1,…,Tt=1,\ldots,T. In this section, we outline how the Remez algorithm [[37](#bib.bib37), [38](#bib.bib38)] can be used to approximate ptp\_{t}.

We begin with the case when d=3d=3. In this case, there is a simple closed form for the optimal odd polynomial p⋆∈ℙ3oddp^{\star}\in\mathbb{P}\_{3}^{\operatorname\*{odd}}; see [[11](#bib.bib11)]. On a given interval [ℓ,u][\ell,u], the optimal approximation to the constant function x↦1x\mapsto 1 is given by the scaled and shifted Newton-Schulz polynomial pNS​(x)=32​x−12​x3p\_{\operatorname\*{NS}}(x)=\frac{3}{2}x-\frac{1}{2}x^{3}:

|  |  |  |
| --- | --- | --- |
|  | p⋆​(x)=β​pNS​(α​x), where ​α=3u2+l​u+ℓ2​ and ​β=42+ℓ​u​(ℓ+u)​α3.p^{\star}(x)=\beta p\_{\operatorname\*{NS}}(\alpha x),\text{ where }\alpha=\sqrt{\frac{3}{u^{2}+lu+\ell^{2}}}\text{ and }\beta=\frac{4}{2+\ell u(\ell+u)\alpha^{3}}. |  |

One can verify that this polynomial satisfies the equioscillation condition from [Lemma A.1](#A1.Thmdefinition1 "Lemma A.1. ‣ Appendix A Proof of Theorem 4.1 ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") at x=ℓ,1α,ux=\ell,\frac{1}{\alpha},u and therefore necessarily has to be the optimal approximation from ℙ3odd\mathbb{P}\_{3}^{\operatorname\*{odd}}. Unfortunately, for larger dd finding closed form expressions for optimal approximations from ℙdodd\mathbb{P}\_{d}^{\operatorname\*{odd}} becomes challenging.
In fact, to the best of our knowledge the optimal approximation is not known for d≥5d\geq 5.
However, we can approximate the optimal polynomial using the Remez algorithm.

Let d=2​q+1d=2q+1.
Recall that from [Lemma A.1](#A1.Thmdefinition1 "Lemma A.1. ‣ Appendix A Proof of Theorem 4.1 ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") the optimal polynomial must satisfy the equioscillation property q+2q+2 times.
The Remez algorithm is an iterative algorithm that finds the equioscillation points A={x0,…,xq+1}A=\{x\_{0},\ldots,x\_{q+1}\} from [Lemma A.1](#A1.Thmdefinition1 "Lemma A.1. ‣ Appendix A Proof of Theorem 4.1 ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") by iteratively refining a sequence of trial points A(k)={x0(k),…,xq+1(k)}A^{(k)}=\{x\_{0}^{(k)},\ldots,x\_{q+1}^{(k)}\} so that A(k)A^{(k)} converges to AA.
From the sequence of trial points A(k)A^{(k)} the algorithm also finds a sequence of polynomials p(k)p^{(k)} so that p(k)p^{(k)} converges to the optimal polynomial. The convergence is extremely fast, and usually 10 iterations is sufficient to converge to the optimal polynomial up to double precision machine epsilon [[37](#bib.bib37)]. More commonly, the Remez algorithm is used to find optimal polynomial approximations to general continuous functions where d≈100d\approx 100 or even d≈1000d\approx 1000. However, because the polynomial we build to approximate sign(x)\operatorname\*{sign}(x) is a composition of polynomials, each of which has a low degree, in our setting the degree dd is small, usually d=5d=5.
For d=5d=5 the Remez algorithm simplifies significantly. We now describe this simplified algorithm.

Recall that as stated in [Lemma A.1](#A1.Thmdefinition1 "Lemma A.1. ‣ Appendix A Proof of Theorem 4.1 ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"), the unique optimal approximation p⋆∈ℙ5oddp^{\star}\in\mathbb{P}\_{5}^{\operatorname\*{odd}} satisfies the equioscillation property at four points. The Remez algorithm first starts with a trial set A(1)={x0(1),x1(1),x2(1),x3(1)}⊂[ℓ,u]A^{(1)}=\{x\_{0}^{(1)},x\_{1}^{(1)},x\_{2}^{(1)},x\_{3}^{(1)}\}\subset[\ell,u] which *ideally* should come close to satisfying the equioscillation property. Since we know that ℓ\ell and uu must be equioscillation points we always set x0(k)=ℓx\_{0}^{(k)}=\ell and x3(k)=ux\_{3}^{(k)}=u for all kk. x2(1)x\_{2}^{(1)} and x3(1)x\_{3}^{(1)} are chosen to be 14​ℓ+34​u\frac{1}{4}\ell+\frac{3}{4}u and 34​ℓ+14​u\frac{3}{4}\ell+\frac{1}{4}u, since we observe that as ℓ→u\ell\to u these are approximately the other equioscillation points. Next, the algorithm solves the following system of equations

|  |  |  |  |
| --- | --- | --- | --- |
|  | a1​xi(1)+b1​(xi(1))3+c1​(xi(1))5+(−1)i​E1=sign(x)=1,i=0,1,2,3,a\_{1}x\_{i}^{(1)}+b\_{1}(x\_{i}^{(1)})^{3}+c\_{1}(x\_{i}^{(1)})^{5}+(-1)^{i}E\_{1}=\operatorname\*{sign}(x)=1,\quad i=0,1,2,3, |  | (17) |

for the unknowns a1,b1,c1,a\_{1},b\_{1},c\_{1}, and E1E\_{1}. Recalling that x0(1)=ℓx\_{0}^{(1)}=\ell and x3(1)=ux\_{3}^{(1)}=u, ([17](#A4.E17 "Equation 17 ‣ Appendix D Remez algorithm ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")) can equivalently be written as a system of linear equations

|  |  |  |  |
| --- | --- | --- | --- |
|  | [ℓℓ3ℓ51x1(1)(x1(1))3(x1(1))5−1x2(1)(x2(1))3(x2(1))51uu3u5−1]​[abcE]=[1111].\begin{bmatrix}\ell&\ell^{3}&\ell^{5}&1\\ x\_{1}^{(1)}&(x\_{1}^{(1)})^{3}&(x\_{1}^{(1)})^{5}&-1\\ x\_{2}^{(1)}&(x\_{2}^{(1)})^{3}&(x\_{2}^{(1)})^{5}&1\\ u&u^{3}&u^{5}&-1\end{bmatrix}\begin{bmatrix}a\\ b\\ c\\ E\end{bmatrix}=\begin{bmatrix}1\\ 1\\ 1\\ 1\end{bmatrix}. |  | (18) |

Once we have solved for a1,b1,c1,a\_{1},b\_{1},c\_{1}, and E1E\_{1} we set p1​(x)=a1​x+b1​x3+b1​x5p\_{1}(x)=a\_{1}x+b\_{1}x^{3}+b\_{1}x^{5}. Now we want to find the error in the L∞L^{\infty} norm of using p1p\_{1} to approximate x↦1x\mapsto 1. We therefore find local maxima of the error function e1​(x)=1−p1​(x)e\_{1}(x)=1-p\_{1}(x) on (ℓ,u)(\ell,u) by setting the derivative of e1​(x)e\_{1}(x) to zero and solving for xx. This results in the quartic equation 5​b1​x4+3​b1​x2+a=05b\_{1}x^{4}+3b\_{1}x^{2}+a=0, which has closed form solutions given by the quadratic formula after the substitution y=x2y=x^{2}. We now let x1(2)x\_{1}^{(2)} and x2(2)x\_{2}^{(2)} be the solutions to this equation and let A(2)={ℓ,x1(2),x2(2),u}A^{(2)}=\{\ell,x\_{1}^{(2)},x\_{2}^{(2)},u\}. We repeat the procedure until |Ek|:=maxx∈[ℓ,u]⁡|1−pk​(x)|≈maxx∈[ℓ,u]⁡|1−pk+1​(x)|:=|Ek+1||E\_{k}|:=\max\limits\_{x\in[\ell,u]}|1-p\_{k}(x)|\approx\max\limits\_{x\in[\ell,u]}|1-p\_{k+1}(x)|:=|E\_{k+1}|.

We note that the matrix appearing in ([18](#A4.E18 "Equation 18 ‣ Appendix D Remez algorithm ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")) is a Vandermonde matrix. Vandermonde matrices become notoriously ill-conditioned as the degree grows large [[15](#bib.bib15), Section 4.6]. However, since in our setting we choose dd to be small, there is no ill-conditioning. Instead, we observe ill-conditioning when ℓ≈u\ell\approx u. However, as ℓ/u→1\ell/u\to 1 the optimal polynomial will converge to the polynomial x/u8​(15−10​(x/u)2+3​(x/u)4)\frac{x/u}{8}\left(15-10(x/u)^{2}+3(x/u)^{4}\right), which can be verified by noting that as ℓ/u→1\ell/u\to 1 all equioscillation points x0,x1,x2,x3x\_{0},x\_{1},x\_{2},x\_{3} must converge to uu. For general d=2​q+1d=2q+1, the polynomial will converge to (x/ℓ)​h​(1−(x/ℓ)2)(x/\ell)h(1-(x/\ell)^{2}) where h∈ℙqh\in\mathbb{P}\_{q} is the [q/0][q/0] Padé approximant to (1−x)1/2(1-x)^{1/2} [[25](#bib.bib25)]. In fact, this polynomial is extremely close to the optimal polynomial for sufficiently large ℓ\ell. To see this, let p⋆p^{\star} be the optimal approximation from ℙ5odd\mathbb{P}\_{5}^{\operatorname\*{odd}} and let p​(x)=x/u8​(15−10​(x/u)2+3​(x/u)4)p(x)=\frac{x/u}{8}\left(15-10(x/u)^{2}+3(x/u)^{4}\right). Then,

|  |  |  |  |
| --- | --- | --- | --- |
|  | maxx∈[ℓ,u]⁡|p⋆​(x)−p​(x)|\displaystyle\max\limits\_{x\in[\ell,u]}|p^{\star}(x)-p(x)| | ≤maxx∈[ℓ,u]⁡|1−p​(x)|+maxx∈[ℓ,u]⁡|1−p⋆​(x)|\displaystyle\leq\max\limits\_{x\in[\ell,u]}|1-p(x)|+\max\limits\_{x\in[\ell,u]}|1-p^{\star}(x)| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤2​maxx∈[ℓ,u]⁡|1−p​(x)|\displaystyle\leq 2\max\limits\_{x\in[\ell,u]}|1-p(x)| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤2​(1−ℓ/u)3.\displaystyle\leq 2\left(1-\ell/u\right)^{3}. |  |

where we invoked [[25](#bib.bib25), Theorem 3.1] and the fact that p⋆p^{\star} is the optimal approximation to x↦1x\mapsto 1 from ℙ5odd\mathbb{P}\_{5}^{\operatorname\*{odd}}. Hence, when ℓ/u≥1−ϵd1/3\ell/u\geq 1-\epsilon\_{d}^{1/3}, where ϵdouble≈1.1×10−16\epsilon\_{\text{double}}\approx 1.1\times 10^{-16} is the double precision machine epsilon, then |p⋆​(x)−p​(x)|≤2​ϵdouble|p^{\star}(x)-p(x)|\leq 2\epsilon\_{\text{double}}. In other words, up to double precision machine epsilon, p⋆p^{\star} is equal to pp. Therefore, whenever ℓ/u≥1−ϵdouble1/3\ell/u\geq 1-\epsilon\_{\text{double}}^{1/3} the algorithm simply returns pp as the optimal polynomial.

The algorithm is outlined in [Algorithm 3](#alg3 "In Appendix D Remez algorithm ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm").
In our experiments, we never observed [Algorithm 3](#alg3 "In Appendix D Remez algorithm ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") taking more than five iterations to converge.

Algorithm 3  Remez algorithm (degree 5 approximation for sign(x)\operatorname\*{sign}(x))

input: interval [ℓ,u][\ell,u] for ℓ>0\ell>0, initial trial points x1(1),x2(1)∈[ℓ,u]x\_{1}^{(1)},x\_{2}^{(1)}\in[\ell,u].
  
output: Approximation p∈ℙ5oddp\in\mathbb{P}\_{5}^{\operatorname\*{odd}} to p⋆=argminp∈ℙ5oddmaxx∈[ℓ,u]⁡|1−p​(x)|p^{\star}=\operatorname\*{argmin}\limits\_{p\in\mathbb{P}\_{5}^{\operatorname\*{odd}}}\max\limits\_{x\in[\ell,u]}|1-p(x)|.

define  ϵdouble=1.11×10−16\epsilon\_{\text{double}}=1.11\times 10^{-16}

if ℓ/u≥1−ϵdouble1/3\ell/u\geq 1-\epsilon\_{\text{double}}^{1/3}  then

Return p​(x)=x/u8​(15−10​(x/u)2+3​(x/u)4)p(x)=\frac{x/u}{8}\left(15-10(x/u)^{2}+3(x/u)^{4}\right)

end if

E0=∞E\_{0}=\infty, E−1=−∞E\_{-1}=-\infty

k←0k\leftarrow 0

while ||Ek|−|Ek−1||>ϵdouble||E\_{k}|-|E\_{k-1}||>\epsilon\_{\text{double}}  do

k←k+1k\leftarrow k+1

[akbkckEk]=[ℓℓ3ℓ51x1(k)(x1(k))3(x1(k))5−1x2(k)(x2(k))3(x2(1))51uu3u5−1]−1​[1111]\begin{bmatrix}a\_{k}\\
b\_{k}\\
c\_{k}\\
E\_{k}\end{bmatrix}=\begin{bmatrix}\ell&\ell^{3}&\ell^{5}&1\\
x\_{1}^{(k)}&(x\_{1}^{(k)})^{3}&(x\_{1}^{(k)})^{5}&-1\\
x\_{2}^{(k)}&(x\_{2}^{(k)})^{3}&(x\_{2}^{(1)})^{5}&1\\
u&u^{3}&u^{5}&-1\end{bmatrix}^{-1}\begin{bmatrix}1\\
1\\
1\\
1\end{bmatrix}

x1(k+1)=−3​bk−9​bk2−20​ak​ck10​ck,x2(k+1)=−3​bk+9​bk2−20​ak​ck10​ckx\_{1}^{(k+1)}=\sqrt{\frac{-3b\_{k}-\sqrt{9b\_{k}^{2}-20a\_{k}c\_{k}}}{10c\_{k}}},x\_{2}^{(k+1)}=\sqrt{\frac{-3b\_{k}+\sqrt{9b\_{k}^{2}-20a\_{k}c\_{k}}}{10c\_{k}}}

end while

Return p​(x)=ak​x+bk​x3+ck​x5p(x)=a\_{k}x+b\_{k}x^{3}+c\_{k}x^{5}

## Appendix E Initialization for Matrices with Large Spectral Gaps

In [Section 4](#S4 "4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"), we constructed a sequence of polynomials that is adapted to the range of the singular values [ℓ,u][\ell,u].
Assuming nothing else about the input, these polynomials are optimal because they provide a good approximation to 11 across the entire interval.
However, in many applications, the spectrum has large gaps; that is, there are several large outlying singular values that are well-separated from the rest.
For these matrices, it is not necessary for the polynomial to be accurate on the entire interval [ℓ,u][\ell,u], only on the range of the small singular values plus a few other isolated points.
In this section, we take advantage of this structure to accelerate our method by preprocessing the matrix to eliminate the largest singular values.

The first step is to find small intervals containing each of these large singular values.
To find lower bounds, we use subspace iteration, which is a generalization of the power method that approximates multiple singular values simultaneously.
Fix kk, the number of singular values we wish to eliminate.
Letting σ1≥⋯≥σn\sigma\_{1}\geq\cdots\geq\sigma\_{n} denote the singular values of 𝑴\bm{M}, subspace iteration produces estimates σ~1≥⋯≥σ~k\tilde{\sigma}\_{1}\geq\cdots\geq\tilde{\sigma}\_{k} satisfying σi≥σ~i\sigma\_{i}\geq\tilde{\sigma}\_{i} for all i∈1,…,ki\in 1,\ldots,k.444Let 𝑸0∈ℝn×k\bm{Q}\_{0}\in\mathbb{R}^{n\times k} be a random matrix with orthonormal columns and define 𝑸t+1,𝑹t+1=𝚚𝚛​(𝑴⊤​𝑴​𝑸t)\bm{Q}\_{t+1},\bm{R}\_{t+1}=\mathtt{qr}\left(\bm{M}^{\top}\bm{M}\bm{Q}\_{t}\right), where 𝚚𝚛\mathtt{qr} is the QR decomposition. Subspace iteration outputs the singular values σ~1,…,σ~k\tilde{\sigma}\_{1},\ldots,\tilde{\sigma}\_{k} of 𝑴​𝑸T\bm{M}\bm{Q}\_{T},
σ~1,…,σ~k\tilde{\sigma}\_{1},\ldots,\tilde{\sigma}\_{k}. By the Cauchy interlacing theorem, σ~k≤σk\tilde{\sigma}\_{k}\leq\sigma\_{k}.
To find upper bounds on each σi\sigma\_{i}, we can use the fact that ‖𝑴‖F2=∑j=1nσj2\|\bm{M}\|\_{\text{F}}^{2}=\sum\_{j=1}^{n}\sigma\_{j}^{2} as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | σi2=‖𝑴‖F2−∑j=1j≠inσj2≤‖𝑴‖F2−∑j=1j≠ikσj2≤‖𝑴‖F2−∑j=1j≠ikσ~j2\sigma\_{i}^{2}=\|\bm{M}\|\_{\text{F}}^{2}-\sum\limits\_{\begin{subarray}{c}j=1\\ j\neq i\end{subarray}}^{n}\sigma\_{j}^{2}\leq\|\bm{M}\|\_{\text{F}}^{2}-\sum\limits\_{\begin{subarray}{c}j=1\\ j\neq i\end{subarray}}^{k}\sigma\_{j}^{2}\leq\|\bm{M}\|\_{\text{F}}^{2}-\sum\limits\_{\begin{subarray}{c}j=1\\ j\neq i\end{subarray}}^{k}\tilde{\sigma}\_{j}^{2} |  | (19) |

That is, for each i∈[n]i\in[n],

|  |  |  |
| --- | --- | --- |
|  | σi∈[σ~i,‖𝑴‖F2−∑j=1j≠ikσ~j2]\sigma\_{i}\in\left[\tilde{\sigma}\_{i},\,\,\sqrt{\|\bm{M}\|\_{\text{F}}^{2}-\sum\limits\_{\begin{subarray}{c}j=1\\ j\neq i\end{subarray}}^{k}\tilde{\sigma}\_{j}^{2}}\right] |  |

Setting i=k+1i=k+1, the above also provides an upper bound for the tail of the spectrum, σk+1,…,σn\sigma\_{k+1},\ldots,\sigma\_{n}.

The second step is to find an odd polynomial that well-approximates the constant function on each of these intervals and on the tail simultaneously.
For simplicity, we treat only the k=1k=1 case here.
Assume that 𝑴\bm{M} is normalized to ‖𝑴‖F=1\|\bm{M}\|\_{\text{F}}=1 and let z=σ~1z=\tilde{\sigma}\_{1} be the lower bound produced by subspace iteration (which reduces to the power method in this case).
Then ([19](#A5.E19 "Equation 19 ‣ Appendix E Initialization for Matrices with Large Spectral Gaps ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")) gives σ1∈[z,1]\sigma\_{1}\in[z,1] and σ2,…,σn≤1−z2\sigma\_{2},\ldots,\sigma\_{n}\leq\sqrt{1-z^{2}}.
Assume that these intervals do not overlap, that is, 1−z2≤z⇔z≥1/2\sqrt{1-z^{2}}\leq z\iff z\geq 1/\sqrt{2}.
Then we construct the unique odd cubic polynomial p​(x)=a​x+b​x3p(x)=ax+bx^{3} that satisfies p​(1−z2)=1p(\sqrt{1-z^{2}})=1 and p​(z)=1p(z)=1 by setting

|  |  |  |  |
| --- | --- | --- | --- |
|  | a=z2​(z+1−z2)−1−z2z​1−z2​(2​z2−1)b=1−z2−zz​1−z2​(2​z2−1)a=\frac{z^{2}(z+\sqrt{1-z^{2}})-\sqrt{1-z^{2}}}{z\sqrt{1-z^{2}}(2z^{2}-1)}\qquad b=\frac{\sqrt{1-z^{2}}-z}{z\sqrt{1-z^{2}}(2z^{2}-1)} |  | (20) |

Because p​(0)=0p(0)=0 and pp has at most one local extremum on ℝ≥0\mathbb{R}\_{\geq 0}, these conditions immediately guarantee that pp is concave-increasing on [0,1−z2][0,\sqrt{1-z^{2}}], so it must lie above the line x↦x/1−z2x\mapsto x/\sqrt{1-z^{2}}. Furthermore, pp is decreasing on [σ1,1][\sigma\_{1},1], so it maps σ1∈[z,1]\sigma\_{1}\in[z,1] to [p​(1),1][p(1),1].
By minimizing p​(1)p(1) over all valid zz (that is, over the interval z∈[1/2,1]z\in[1/\sqrt{2},1]), one can further show that p​(1)>1/2p(1)>1/\sqrt{2}, so σ1\sigma\_{1} cannot be decreased very much by applying pp.
Thus, the largest singular value of p​(𝑴)p(\bm{M}) is still at most 11, while the smaller singular values have increased by a potentially large factor of 1/1−z21/\sqrt{1-z^{2}}.
When there is a large outlying singular value, zz is close to 11 and this initialization scheme makes much more progress than a standard iteration of PolarExpress would have.

In [Figure 7](#A5.F7 "In Appendix E Initialization for Matrices with Large Spectral Gaps ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"), we demonstrate the benefit of using the pp given by ([20](#A5.E20 "Equation 20 ‣ Appendix E Initialization for Matrices with Large Spectral Gaps ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")) on a synthetic matrix whose spectrum follows a power law decay. That is, σj​(𝑴)=j−5\sigma\_{j}(\bm{M})=j^{-5}, so this matrix has a large outlying singular value σ1≫σ2\sigma\_{1}\gg\sigma\_{2}.
Applying ([20](#A5.E20 "Equation 20 ‣ Appendix E Initialization for Matrices with Large Spectral Gaps ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")) costs almost as much as performing an iteration of a degree-5 polynomial method, so for fair comparison, we count it as an additional iteration in this plot.
For both Newton-Schulz and Polar Express, performing the extra spectrum-aware initialization step described in this section leads to significant speedups in convergence.

![Refer to caption](/html/2505.16932/assets/x10.png)


Figure 7: Benefits of the spectrum-aware initialization scheme of [Appendix E](#A5 "Appendix E Initialization for Matrices with Large Spectral Gaps ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"). Using this scheme improves convergence of both Newton-Schulz and Polar Express on a synthetic 32×3232\times 32 matrix with σj​(𝑴)=j−5\sigma\_{j}(\bm{M})=j^{-5}. Note that we count the spectrum-aware initialization as an additional iteration.

## Appendix F Fast Polynomial Iteration for Rectangular Matrices

In this section, we describe a simple method for applying an iterative polynomial method to a rectangular matrix.
For matrices with a large aspect ratio, this method yields significant computational savings.
We emphasize that this method is applicable to *any* computation of the form (pT∘⋯∘p1)​(𝑿)(p\_{T}\circ\cdots\circ p\_{1})(\bm{X}), where each ptp\_{t} is an odd polynomial.
Thus, it can be used to apply Newton-Schulz or Jordan’s polynomials in addition to our own.

As a preliminary, we first describe the baseline approach.
Let 𝑿∈ℝm×n\bm{X}\in\mathbb{R}^{m\times n} with m≥nm\geq n, where α:=m/n≥1\alpha:=m/n\geq 1 is called the aspect ratio.
Any odd polynomial pp of degree d=2​q+1d=2q+1 can be represented as p​(x)=x​h​(x2)p(x)=xh(x^{2}), where hh is a polynomial of degree qq.
Thus, p​(𝑿)=𝑿​h​(𝑿⊤​𝑿)p(\bm{X})=\bm{X}h(\bm{X}^{\top}\bm{X}).
Furthermore, hh can be written in a factored form called Horner’s rule to reduce the number of multiplications.
For instance, if h​(y)=a+b​y+c​y2+d​y3h(y)=a+by+cy^{2}+dy^{3}, Horner’s rule gives h​(y)=a+y​(b+y​(c+d​y))h(y)=a+y\left(b+y\left(c+dy\right)\right).
For a matrix, h​(𝒀)=a​𝑰+𝒀​(b​𝑰+𝒀​(c​𝑰+d​𝒀))h(\bm{Y})=a\bm{I}+\bm{Y}\left(b\bm{I}+\bm{Y}\left(c\bm{I}+d\bm{Y}\right)\right).
Thus for 𝒀∈ℝn×n\bm{Y}\in\mathbb{R}^{n\times n}, computing h​(𝒀)h(\bm{Y}) costs about (deg⁡(h)−1)⋅n3\left(\deg(h)-1\right)\cdot n^{3} operations, and computing p​(𝑿)=𝑿​h​(𝑿⊤​𝑿)p(\bm{X})=\bm{X}h(\bm{X}^{\top}\bm{X}) costs 2​m​n2+(d−12−1)⋅n3=(d−32+2​α)⋅n32mn^{2}+\left(\frac{d-1}{2}-1\right)\cdot n^{3}=\left(\frac{d-3}{2}+2\alpha\right)\cdot n^{3} operations.
This process could be repeated for each iteration p1,…,pTp\_{1},\ldots,p\_{T}.
Notice that if we instead computed h​(𝑿​𝑿⊤)​𝑿h(\bm{X}\bm{X}^{\top})\bm{X}, the result would be the same but the cost would be higher.

A major drawback of this naive approach is that it has a strong dependence on α\alpha, since two rectangular matrix multiplications must be performed in *each* of the TT iterations.
When m≫nm\gg n, these two multiplications dominate the cost.
In [Algorithm 4](#alg4 "In Appendix F Fast Polynomial Iteration for Rectangular Matrices ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"), we introduce a simple trick that dramatically reduces this cost, using just two rectangular matrix multiplications to compute *all* TT iterations.

Algorithm 4  Fast Polynomial Iteration for Rectangular Matrices

input: 𝑿∈ℝm×n\bm{X}\in\mathbb{R}^{m\times n} with m>1.5​nm>1.5n, odd polynomials p1​(x)=x​h1​(x2),…,pT​(x)=x​hT​(x2)p\_{1}(x)=xh\_{1}(x^{2}),\ldots,p\_{T}(x)=xh\_{T}(x^{2}).
  
output: The matrix (pT∘⋯∘p1)​(𝑿)(p\_{T}\circ\cdots\circ p\_{1})(\bm{X}).

𝒀=𝑿⊤​𝑿\bm{Y}=\bm{X}^{\top}\bm{X} ⊳\triangleright m​n2mn^{2}

Let 𝑸0=𝑰\bm{Q}\_{0}=\bm{I}

for t=1,2,…,Tt=1,2,\ldots,T do

𝑹t=𝑸t−1⊤​𝒀​𝑸t−1\bm{R}\_{t}=\bm{Q}\_{t-1}^{\top}\bm{Y}\bm{Q}\_{t-1} ⊳\triangleright 2​n32n^{3}

𝑸t=𝑸t−1​ht​(𝑹t)\bm{Q}\_{t}=\bm{Q}\_{t-1}h\_{t}(\bm{R}\_{t}) ⊳\triangleright Horner’s rule: deg⁡(ht)⋅n3\deg(h\_{t})\cdot n^{3}

end for

return 𝑿​𝑸T\bm{X}\bm{Q}\_{T} ⊳\triangleright m​n2mn^{2}

To see why this works, define q0​(x)=xq\_{0}(x)=x,

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | qt​(x)\displaystyle q\_{t}(x) | =(pt∘⋯∘p1)​(x)x=pt​((pt−1∘⋯∘p1)​(x))x=pt​(x​qt−1​(x))x\displaystyle=\frac{(p\_{t}\circ\cdots\circ p\_{1})(x)}{x}=\frac{p\_{t}\left((p\_{t-1}\circ\cdots\circ p\_{1})(x)\right)}{x}=\frac{p\_{t}\left(xq\_{t-1}(x)\right)}{x} |  | (21) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =x​qt−1​(x)⋅ht​((x​qt−1​(x))2)x=qt−1​(x)⋅ht​(x2⋅qt−1​(x)2)\displaystyle=\frac{xq\_{t-1}(x)\cdot h\_{t}\left((xq\_{t-1}(x))^{2}\right)}{x}=q\_{t-1}(x)\cdot h\_{t}\left(x^{2}\cdot q\_{t-1}(x)^{2}\right) |  | (22) |

and rt​(x)=x2⋅qt−1​(x)2r\_{t}(x)=x^{2}\cdot q\_{t-1}(x)^{2}.
It is clear by induction that 𝑹t=rt​(𝑿),𝑸t=qt​(𝑿)\bm{R}\_{t}=r\_{t}(\bm{X}),\bm{Q}\_{t}=q\_{t}(\bm{X}), and 𝑿​𝑸T=(pt∘⋯∘p1)​(𝑿)\bm{X}\bm{Q}\_{T}=(p\_{t}\circ\cdots\circ p\_{1})(\bm{X}).
As promised, this algorithm uses no rectangular multiplications in the for-loop.
If each ptp\_{t} is degree dd, then the total cost is (d+32​T+2​α)⋅n3\left(\frac{d+3}{2}T+2\alpha\right)\cdot n^{3}.
When α>1.5​TT−1\alpha>1.5\frac{T}{T-1}, this is smaller than the naive method.
We can use this criterion to select either [Algorithm 4](#alg4 "In Appendix F Fast Polynomial Iteration for Rectangular Matrices ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") or the baseline method at runtime.

There is one significant weakness of [Algorithm 4](#alg4 "In Appendix F Fast Polynomial Iteration for Rectangular Matrices ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"). In bfloat16 precision, it can introduce numerical errors.
Our intuition for why this happens is as follows. Let 𝑿=𝑼​𝚺​𝑽⊤\bm{X}=\bm{U}\bm{\Sigma}\bm{V}^{\top} be the SVD. For large TT, (pT∘⋯​p1)​(𝑿)=𝑿​𝑸T≈polar(𝑿)=𝑼​𝑽⊤(p\_{T}\circ\cdots p\_{1})(\bm{X})=\bm{X}\bm{Q}\_{T}\approx\operatorname\*{polar}(\bm{X})=\bm{U}\bm{V}^{\top}. Thus, 𝑸T≈𝑽⊤​𝚺−1​𝑽\bm{Q}\_{T}\approx\bm{V}^{\top}\bm{\Sigma}^{-1}\bm{V}. When 𝑿\bm{X} has very small singular values and the floating point precision is very low, instantiating 𝑸T\bm{Q}\_{T} may be unstable.
To mitigate this issue, we use a restarting strategy.
Notice that the issue arises only for large TT, for which (pT∘⋯∘p1)​(ϵ)≈1(p\_{T}\circ\cdots\circ p\_{1})(\epsilon)\approx 1.
Limiting ourselves to T=3T=3 iterations improves the conditioning of 𝑸T\bm{Q}\_{T} because (pT∘⋯∘p1)​(ϵ)≪1(p\_{T}\circ\cdots\circ p\_{1})(\epsilon)\ll 1
Thus, to compute T=6T=6 iterations, we can apply [Algorithm 4](#alg4 "In Appendix F Fast Polynomial Iteration for Rectangular Matrices ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") first to the first three polynomials, then again to the last three polynomials.
Note that restarting [Algorithm 4](#alg4 "In Appendix F Fast Polynomial Iteration for Rectangular Matrices ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") after every iteration is exactly the same as the baseline method.
This approach provides a tunable hyperparameter—the number of iterations we apply before restarting [Algorithm 4](#alg4 "In Appendix F Fast Polynomial Iteration for Rectangular Matrices ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")—allowing us to find a balance between the fully stable but slow baseline on the one hand, and a very fast but numerically risky method on the other hand.

[Figure 8](#A6.F8 "In Appendix F Fast Polynomial Iteration for Rectangular Matrices ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") shows that using [Algorithm 4](#alg4 "In Appendix F Fast Polynomial Iteration for Rectangular Matrices ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") can dramatically improve runtime on the GPU when the aspect ratio is large enough.
As expected, using [Algorithm 4](#alg4 "In Appendix F Fast Polynomial Iteration for Rectangular Matrices ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") for many iterations significantly reduces the dependence of the runtime on the aspect ratio.
Running six iterations of a degree-5 polynomial method when α=4\alpha=4 (as with the linear transformations in each MLP block of a transformer) we obtain almost a 2x speedup, and when α=32\alpha=32, we obtain nearly a 10x speedup.
If we restart every three iterations, the trend is the same but the runtime savings are somewhat smaller.

![Refer to caption](/html/2505.16932/assets/x11.png)


Figure 8: Effects of using [Algorithm 4](#alg4 "In Appendix F Fast Polynomial Iteration for Rectangular Matrices ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") on runtime on a GPU. We run T=6T=6 iterations of a degree-5 polynomial method on matrices with various dimensions nn and aspect ratios α\alpha. Restart interval =6=6 is [Algorithm 4](#alg4 "In Appendix F Fast Polynomial Iteration for Rectangular Matrices ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"), restart interval =1=1 is equivalent to the baseline (that is, not using [Algorithm 4](#alg4 "In Appendix F Fast Polynomial Iteration for Rectangular Matrices ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")), and restart interval =3=3 is an intermediate method that calls [Algorithm 4](#alg4 "In Appendix F Fast Polynomial Iteration for Rectangular Matrices ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") once to do the first three iterations and again to do the last three iterations for greater stability.
When α≫1\alpha\gg 1, increasing the restart interval *significantly* reduces the runtime.

Preliminary experiments using [Algorithm 4](#alg4 "In Appendix F Fast Polynomial Iteration for Rectangular Matrices ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") with Muon in bfloat16 were not successful. The training and validation losses were sometimes significantly higher than before, even accounting for the runtime savings of [Algorithm 4](#alg4 "In Appendix F Fast Polynomial Iteration for Rectangular Matrices ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"). While this technique may already be applicable in float32 or for architectures whose weight matrices have larger aspect rations, more work is needed to make it practical for general deep learning applications.

If these problems can be mitigated, the speed afforded by [Algorithm 4](#alg4 "In Appendix F Fast Polynomial Iteration for Rectangular Matrices ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") suggests a potentially beneficial change in the way Muon is applied to transformers.
Each multi-head attention layer contains four square weight matrices 𝑾Q,𝑾K,𝑾V\bm{W}\_{Q},\bm{W}\_{K},\bm{W}\_{V} and 𝑾O∈ℝd×d\bm{W}\_{O}\in\mathbb{R}^{d\times d}.
The orthogonalization step of Muon is either applied separately to these four matrices or else to [𝑾Q​∣𝑾K∣​𝑾V][\bm{W}\_{Q}\mid\bm{W}\_{K}\mid\bm{W}\_{V}] and 𝑾O\bm{W}\_{O}, since typical implementations of multi-head attention store the weights in this concatenated form.
However, we believe it is natural to consider each of these four weight matrices to be a concatenation of many smaller linear transformations, each corresponding to a single attention head.
If HH is the number of heads, each of these smaller matrices has size d×dHd\times\frac{d}{H}; that is, they have aspect ratio α=H\alpha=H.
Since typical transformers like GPT-3 can have as many as 9696 heads, this version of Muon can yield huge savings in the runtime of each step.
We leave it to future work to examine whether this version of Muon also enjoys fast convergence.

## Appendix G Code for Constructing Polynomials of Polar Express

The following code gives a Python implementation of the offline stage of [Algorithm 2](#alg2 "In 4.5 The algorithm ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"). This code was used to construct the coefficients of the polynomials given in (LABEL:eq:coeffs), which in turn were used in our Muon experiments ([Section 5.2](#S5.SS2 "5.2 Training GPT-2 ‣ 5 Numerical Experiments ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")). It uses ℓ=10−3\ell=10^{-3} and u=1u=1 by default.
It incorporates [Algorithm 3](#alg3 "In Appendix D Remez algorithm ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm") and the numerical tweaks described in [Section 4.4](#S4.SS4 "4.4 Finite precision considerations ‣ 4 The Polar Express ‣ The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm")

[⬇](data:text/plain;base64,ZnJvbSBtYXRoIGltcG9ydCBpbmYsIHNxcnQKaW1wb3J0IG51bXB5IGFzIG5wCgoKZGVmIG9wdGltYWxfcXVpbnRpYyhsLCB1KToKICAgIGFzc2VydCAwIDw9IGwgPD0gdQogICAgaWYgMSAtIDVlLTYgPD0gbCAvIHU6CiAgICAgICAgIyBBYm92ZSB0aGlzIHRocmVzaG9sZCwgdGhlIGVxdWlvc2NpbGxhdGluZyBwb2x5bm9taWFscwogICAgICAgICMgaXMgbnVtZXJpY2FsbHkgZXF1YWwgdG8uLi4KICAgICAgICByZXR1cm4gKDE1LzgpL3UsICgtMTAvOCkvKHUqKjMpLCAoMy84KS8odSoqNSkKICAgICMgVGhpcyBpbml0aWFsaXphdGlvbiBiZWNvbWVzIGV4YWN0IGFzIGwgLT4gdQogICAgcSA9ICgzKmwgKyAxKSAvIDQKICAgIHIgPSAobCArIDMpIC8gNAogICAgRSwgb2xkX0UgPSBpbmYsIE5vbmUKICAgIHdoaWxlIG5vdCBvbGRfRSBvciBhYnMob2xkX0UgLSBFKSA+IDFlLTE1OgogICAgICAgIG9sZF9FID0gRQogICAgICAgIExIUyA9IG5wLmFycmF5KFsKICAgICAgICAgICAgW2wsIGwqKjMsIGwqKjUsIDFdLAogICAgICAgICAgICBbcSwgcSoqMywgcSoqNSwgLTFdLAogICAgICAgICAgICBbciwgcioqMywgcioqNSwgMV0sCiAgICAgICAgICAgIFt1LCB1KiozLCB1Kio1LCAtMV0sCiAgICAgICAgXSkKICAgICAgICBhLCBiLCBjLCBFID0gbnAubGluYWxnLnNvbHZlKExIUywgbnAub25lcyg0KSkKICAgICAgICBxLCByID0gbnAuc3FydCgoLTMqYiArIG5wLmFycmF5KFstMSwgMV0pICoKICAgICAgICAgICAgICAgICAgICAgICAgc3FydCg5KmIqKjIgLSAyMCphKmMpKSAvICgxMCpjKSkKICAgIHJldHVybiBmbG9hdChhKSwgZmxvYXQoYiksIGZsb2F0KGMpCgoKZGVmIG9wdGltYWxfY29tcG9zaXRpb24obCwgbnVtX2l0ZXJzLCBjdXNoaW9uPTAuMDI0MDczMjc0MjQxODI3NjEpOgogICAgdSA9IDEKICAgIGNvZWZmaWNpZW50cyA9IFtdCiAgICBmb3IgXyBpbiByYW5nZShudW1faXRlcnMpOgogICAgICAgIGEsIGIsIGMgPSBvcHRpbWFsX3F1aW50aWMobWF4KGwsIGN1c2hpb24qdSksIHUpCiAgICAgICAgIyBEdWUgdG8gY3VzaGlvbmluZywgdGhpcyBtYXkgYmUgY2VudGVyZWQgYXJvdW5kIDEgd2l0aAogICAgICAgICMgcmVzcGVjdCB0byAwLjAyNCp1LCB1LiBSZWNlbnRlciBpdCBhcm91bmQgMSB3aXRoIHJlc3BlY3QKICAgICAgICAjIHRvIGwsIHUsIG1lYW5pbmcgZmluZCBjIHNvIHRoYXQgMSAtIGMqcChsKSA9IGMqcCh1KSAtIDE6CiAgICAgICAgcGwgPSBhKmwgKyBiKmwqKjMgKyBjKmwqKjUKICAgICAgICBwdSA9IGEqdSArIGIqdSoqMyArIGMqdSoqNQogICAgICAgIHJlc2NhbGFyID0gMi8ocGwgKyBwdSkKICAgICAgICBhICo9IHJlc2NhbGFyOyBiICo9IHJlc2NhbGFyOyBjICo9IHJlc2NhbGFyCiAgICAgICAgIyBPcHRpb25hbGx5IGluY29ycG9yYXRlIHNhZmV0eSBmYWN0b3IgaGVyZToKICAgICAgICAjIGEgLz0gMS4wMTsgYiAvPSAxLjAxKiozOyBjIC89IDEuMDEqKjUKICAgICAgICBjb2VmZmljaWVudHMuYXBwZW5kKChhLCBiLCBjKSkKICAgICAgICBsID0gYSpsICsgYipsKiozICsgYypsKio1CiAgICAgICAgdSA9IDIgLSBsCiAgICByZXR1cm4gY29lZmZpY2llbnRzCgoKcHJpbnQoKm9wdGltYWxfY29tcG9zaXRpb24oMWUtMywgMTApLCBzZXA9IlxuIikK)

from math import inf, sqrt

import numpy as np

def optimal\_quintic(l, u):

assert 0 <= l <= u

if 1 - 5e-6 <= l / u:

# Above this threshold, the equioscillating polynomials

# is numerically equal to...

return (15/8)/u, (-10/8)/(u\*\*3), (3/8)/(u\*\*5)

# This initialization becomes exact as l -> u

q = (3\*l + 1) / 4

r = (l + 3) / 4

E, old\_E = inf, None

while not old\_E or abs(old\_E - E) > 1e-15:

old\_E = E

LHS = np.array([

[l, l\*\*3, l\*\*5, 1],

[q, q\*\*3, q\*\*5, -1],

[r, r\*\*3, r\*\*5, 1],

[u, u\*\*3, u\*\*5, -1],

])

a, b, c, E = np.linalg.solve(LHS, np.ones(4))

q, r = np.sqrt((-3\*b + np.array([-1, 1]) \*

sqrt(9\*b\*\*2 - 20\*a\*c)) / (10\*c))

return float(a), float(b), float(c)

def optimal\_composition(l, num\_iters, cushion=0.02407327424182761):

u = 1

coefficients = []

for \_ in range(num\_iters):

a, b, c = optimal\_quintic(max(l, cushion\*u), u)

# Due to cushioning, this may be centered around 1 with

# respect to 0.024\*u, u. Recenter it around 1 with respect

# to l, u, meaning find c so that 1 - c\*p(l) = c\*p(u) - 1:

pl = a\*l + b\*l\*\*3 + c\*l\*\*5

pu = a\*u + b\*u\*\*3 + c\*u\*\*5

rescalar = 2/(pl + pu)

a \*= rescalar; b \*= rescalar; c \*= rescalar

# Optionally incorporate safety factor here:

# a /= 1.01; b /= 1.01\*\*3; c /= 1.01\*\*5

coefficients.append((a, b, c))

l = a\*l + b\*l\*\*3 + c\*l\*\*5

u = 2 - l

return coefficients

print(\*optimal\_composition(1e-3, 10), sep="\n")

[◄](/html/2505.16931)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2505.16932)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2505.16932)
[View original  
on arXiv](https://arxiv.org/abs/2505.16932)[►](/html/2505.16933)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Thu Jun 5 21:06:42 2025 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
