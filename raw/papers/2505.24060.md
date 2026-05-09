---
arxiv: '2505.24060'
authors:
- Chris Mingard
- Lukas Seier
- Niclas Göring
- Andrei-Vlad Badelita
- Charles London
- Ard Louis
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: Characterising the Inductive Biases of Neural Networks on Boolean Data
url: http://arxiv.org/abs/2505.24060v1
year: 2025
---

[2505.24060] Characterising the Inductive Biases of Neural Networks on Boolean Data














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



# Characterising the Inductive Biases of Neural Networks on Boolean Data

Chris Mingard
  
Lukas Seier
  
Niclas Göring
  
Andrei-Vlad Badelita
  
Charles London
  
Ard Louis

###### Abstract

Deep neural networks are renowned for their ability to generalise well across diverse tasks, even when heavily overparameterized. Existing works offer only partial explanations (for example, the NTK-based task-model alignment explanation neglects feature learning). Here, we provide an end-to-end, analytically tractable case study that links a network’s inductive prior, its training dynamics including feature learning, and its eventual generalisation. Specifically, we exploit the one-to-one correspondence between depth-2 discrete fully connected networks and disjunctive normal form (DNF) formulas by training on Boolean functions. Under a Monte Carlo learning algorithm, our model exhibits predictable training dynamics and the emergence of interpretable features. This framework allows us to trace, in detail, how inductive bias and feature formation drive generalisation.

## 1 Introduction

Deep neural networks have achieved remarkable success despite being vastly overparameterized (Brown et al., [2020](#bib.bib12); Jumper et al., [2021](#bib.bib35)), challenging classical learning theory predictions that such models should overfit due to their capacity to learn highly complex functions (Zhang et al., [2016](#bib.bib70)). As a result, much work has gone into studying the inductive biases of neural networks as a means for explaining their ability to generalise (Chizat & Bach, [2020](#bib.bib15); Belkin, [2021](#bib.bib6); Delétang et al., [2023](#bib.bib18)). While several partial explanations exist, we lack a complete framework that connects architectural design, learning dynamics, feature emergence, and generalisation in an analytically tractable manner.

Understanding neural network generalisation can broadly be organised into three lines of work. *(i) Kernel-based theories* treat a network’s training dynamics in the infinite-width limit as linearised gradient descent most prominently through the Neural Tangent Kernel (NTK) (Jacot et al., [2018](#bib.bib34); Ortiz-Jiménez et al., [2021](#bib.bib53)). These analyses show that the network first fits functions aligned with the top eigenfunctions of the kernel (Bowman & Montufar, [2022](#bib.bib10)). However, this framework cannot capture feature learning as weights do not evolve during training. *(ii) Finite-width feature-learning* analyses the training dynamics but mostly in mean-field settings or for linear neural networks for specific data distributions (Chizat et al., [2019](#bib.bib16); Mei et al., [2018](#bib.bib42); Dominé et al., [2025](#bib.bib22)). Lastly, *(iii) Mechanistic interpretability* reveals meaningful structures in trained networks like feature detectors in vision models (Elhage et al., [2021](#bib.bib24)) or induction heads in language models (Nanda et al., [2023](#bib.bib47)). It identifies what representations emerge without explaining why or how these features develop through the interplay of architecture, data, and training dynamics.

Despite deep neural networks’ impressive generalisation in many domains and numerous insightful partial theories, there is still no end-to-end understanding of generalisation in neural networks from first principles.

### 1.1 Our contribution

The aim of this paper is to introduce a toy model that lets us
follow, step by step, how inductive bias (architecture and weight initialization), training dynamics and feature
learning combine to yield generalisation. Concretely, we study
*depth-2  discrete fully-connected networks* (DFCNs) on Boolean functions and show:

1. 1.

   Interpretable complexity measure:
   We prove a one-to-one correspondence between DFCNs and disjunctive
   normal forms (DNFs) ([Proposition˜2.7](#S2.Thmtheorem7 "Proposition 2.7 (DNF-DFCN bijection). ‣ 2.2 DFCN–DNF correspondence ‣ 2 Preliminaries and intuition ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")). This allows us to
   translate geometric notions such as weight norm into a
   function-level complexity measure K(f)K{(f)}.
2. 2.

   Analytic characterisation of implicit bias:
   Randomly initialised DFCNs induce a tractable prior
   P(f)P(f) over Boolean functions. We derive bounds on P(f)P(f) in terms of K(f)K{(f)} that show strong simplicity bias in a range of function families ([Table˜2](#S3.T2 "In 3.2 Simplicity bias in 𝑃(𝑓) ‣ 3 Untrained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")).
3. 3.

   Feature learning under a Bayesian lens: Using both Metropolis sampling and a greedy stochastic gradient descent (SGD)-like algorithm, we demonstrate that generalisation correlates with the function’s prior probability P(f)P(f). Functions occupying larger volumes in parameter space (which have low DNF complexity) have low sample complexity, while those with small prior probability (like parity) are unlearnable – more data harms generalisation.
4. 4.

   Weight decay induces stronger simplicity bias and improves feature learning:
   For DFCNs, ℓ1\ell\_{1}-regularisation translates into a simple multiplicative factor e−λK(f)e^{-\lambda K{(f)}} in the posterior
   ([4.2](#S4.Ex4 "4.2 Weight decay adds an additional bias in the posterior ‣ 4 Trained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")). This lets us quantify how
   weight decay sharpens the native simplicity bias in P(f)P(f). This improves generalisation on
   “easy” targets (small K(f)K{(f)}) but not on inherently complex ones
   (e.g. high-order parity). We also quantify how this optimiser-induced bias can lead to learning better features.

### 1.2 Related work

For a full discussion of related work refer to Appendix [A](#A1 "Appendix A A brief review of generalisation in neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"). (Mingard et al., [2021](#bib.bib45); Smith & Le, [2017](#bib.bib62); Neal, [2012](#bib.bib49)) argued that generalisation is best understood in a Bayesian framework, viewing SGD’s convergence as approximating a Bayesian posterior. (Valle-Pérez et al., [2018](#bib.bib68); Mingard et al., [2025](#bib.bib46), [2019](#bib.bib44)) empirically demonstrated that the prior over functions in randomly initialized neural networks has a strong simplicity bias towards Lemple-Ziv(LZ)-simple Boolean functions, see also e.g. (Palma et al., [2019](#bib.bib54); Teney et al., [2025b](#bib.bib65)). For Boolean function learning, see also (Abbe et al., [2025](#bib.bib3), [2023](#bib.bib1)).

![Refer to caption](/html/2505.24060/assets/x1.png)


Figure 1: 
Representing Boolean functions Here we show the three ways of representing ff. The green panel shows the string representation and truth table. The left red panel shows how we can extract the DNF representation from the truth table. The right red panel shows the minimum DNF representation of ff – when the complexity K(f)K{(f)} is minimised. The grey panels show how we can represent ff by copying the clauses from the red panels into a DFCN (with ’+’s meaning 1 and ’-’s meaning −1-1).
W(1)W^{(1)} and b(1)b^{(1)} are the weights and biases of the first layer. The combination of the ReLU activation and the bias term (set with [Definition˜2.6](#S2.Thmtheorem6 "Definition 2.6 (DFCN). ‣ 2.2 DFCN–DNF correspondence ‣ 2 Preliminaries and intuition ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")) ensures that each neuron’s output is only 1 when the clause is True, and 0 otherwise. W(2)W^{(2)} and b(2)b^{(2)} act as the OR operators (plus a global function negation β\beta). The example in the figure uses β=1\beta=1. Note that to guarantee full expressivity, W(1)W^{(1)} has dimensions (2n−1×n)(2^{n-1}\times n).

## 2 Preliminaries and intuition

In [Section˜2.1](#S2.SS1 "2.1 Boolean functions and their disjunctive normal form ‣ 2 Preliminaries and intuition ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"), we introduce Boolean functions along with two canonical representations: the string-representation and the DNF-representation. In [Section˜2.2](#S2.SS2 "2.2 DFCN–DNF correspondence ‣ 2 Preliminaries and intuition ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") we introduce a novel DFCN-representation, which provides a one-to-one correspondence between Boolean functions and DFCNs.

### 2.1 Boolean functions and their disjunctive normal form

###### Definition 2.1 (Boolean function ff).

For a fixed input dimension nn, a Boolean function maps the 2n2^{n} vertices of the nn-dimensional hypercube to {0,1}\{0,1\}

|  |  |  |  |
| --- | --- | --- | --- |
|  | f:{0,1}n→{0,1}.\displaystyle f:\{0,1\}^{n}\rightarrow\{0,1\}. |  | (1) |

There are 2n2^{n} inputs 𝒙∈{0,1}n\bm{x}\in\{0,1\}^{n} and 2 outputs y∈{0,1}y\in\{0,1\} and therefore 22n2^{2^{n}} possible functions.

###### Definition 2.2 (string-representation).

We define the *string*-representation of a function ff as an output string of 0s and 1s where the order is given by an ascending concatenated binary representation of the inputs. See [Figure˜1](#S1.F1 "In 1.2 Related work ‣ 1 Introduction ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data").

There is a second canonical way to represent a Boolean function: with the *disjunctive normal form* O’Donnell ([2014](#bib.bib50)).
Any Boolean function with nn variables can be described by a truth table with 2n2^{n} rows (see [Figure˜1](#S1.F1 "In 1.2 Related work ‣ 1 Introduction ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")). Each row represents a complete assignment of the variables and therefore corresponds to a conjunction of literals (a clause). To obtain a symbolic description from this table, retain only the rows whose output entry is 11 and take the disjunction (logical OR) of their corresponding clauses. The resulting formula is the function’s DNF, defined formally below.

###### Definition 2.3 (Literal, clause, DNF).

Let 𝒙∈{0,1}n\bm{x}\in\{0,1\}^{n} denote a Boolean input vector.
A *literal* is either a variable xix\_{i} or its negation
¬xi\neg x\_{i} for some index i∈{1,…,n}i\in\{1,\dots,n\}.
A *clause* CC is a conjunction (AND) of one or more literals: C(𝒙)=∧i∈Svi,S⊆{1,…,n},vi∈{xi,¬xi}C(\bm{x})=\wedge\_{i\in S}v\_{i},\ S\subseteq\{1,\dots,n\},\ v\_{i}\in\{x\_{i},\neg x\_{i}\}. A Boolean function f(𝒙)f(\bm{x}) can be described by a DNF with tt clauses if there exist clauses C1,…,CtC\_{1},...,C\_{t}, and a global negation
β∈{+1,−1}\beta\in\{+1,-1\} such that

|  |  |  |
| --- | --- | --- |
|  | Φf(𝒙)=β[C1(𝒙)∨C2(𝒙)∨⋯∨Ct(𝒙)].\Phi\_{f}(\bm{x})\;=\;\beta\bigl{[}C\_{1}(\bm{x})\;\lor\;C\_{2}(\bm{x})\;\lor\;\cdots\lor\;C\_{t}(\bm{x})\bigr{]}. |  |

We note a slight abuse of notation in using xx to denote both input vectors and literals; the intended meaning should be clear from context.

###### Definition 2.4 (Length of a DNF).

For a DNF Φf\Phi\_{f} whose jj-th clause
contains kj=|Sj|k\_{j}=|S\_{j}| literals,
its *length* is the total number of literals

|  |  |  |  |
| --- | --- | --- | --- |
|  | L(Φf)=∑j=1tkj.\displaystyle L(\Phi\_{f})\;=\;\sum\_{j=1}^{t}k\_{j}. |  | (2) |

When β=−1\beta=-1, we obtain the logical complement ¬DNF\neg\mathrm{DNF}.
With this additional global negation, every Boolean function admits a DNF with at most 2n−12^{n-1} clauses (Mingard et al., [2019](#bib.bib44)).
This representation of length L(Φf)≤n⋅2n−1L(\Phi\_{f})\leq n\cdot 2^{n-1} is obtained by enumerating all truth-table rows whose output equals 1 (or 0 when β=−1\beta=-1, which may be required when t>2n−1t>2^{n-1}); it is called the *canonical expansion* and, up to lexicographic ordering of the clauses, is unique (Kohavi & Jha, [2009](#bib.bib37); Quine, [1952](#bib.bib55)).
See [Figure˜1](#S1.F1 "In 1.2 Related work ‣ 1 Introduction ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") for an example.
In practice, however, a Boolean function often admits a far shorter DNF Φf\Phi\_{f}, and finding such a minimal representation is NP-hard (Allender et al., [2005](#bib.bib4)).
Conversely, the length can be artificially increased beyond n⋅2n−1n\cdot 2^{n-1} by padding the formula with tautologically false clauses or duplicating existing ones, yielding DNFs of arbitrarily large length that still compute ff. Because the raw length can therefore grow without bound, a meaningful complexity measure should refer to the *shortest* realisation of L(Φf)L(\Phi\_{f})
(see [Section˜B.2](#A2.SS2 "B.2 Relating DNF complexity to weight norm ‣ Appendix B Notes on complexity measures ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") for further justification).

###### Definition 2.5 (DNF complexity).

We define the DNF complexity K(f)K{(f)} as the shortest possible DNF expressing ff,

|  |  |  |  |
| --- | --- | --- | --- |
|  | K(f)=minΦfL(Φf).\displaystyle K{(f)}\;=\;\min\_{\Phi\_{f}}L(\Phi\_{f}). |  | (3) |

### 2.2 DFCN–DNF correspondence

A central concept of this paper is to link the first two well-known representations of a Boolean function to one in terms of DFCNs with a fixed hidden layer width of 0pt2n−10pt2^{n-1}, where 0pt∈ℕ0pt\in\mathbb{N}. A DFCN with this structure is fully expressive (Mingard et al., [2019](#bib.bib44)).

###### Definition 2.6 (DFCN).

A DFCN is a depth-two network

|  |  |  |  |
| --- | --- | --- | --- |
|  | fθ(𝒙)\displaystyle f\_{\theta}(\bm{x}) | =𝟙[W(2)σ(W(1)𝒙+b(1))+b(2)>0],\displaystyle=\mathbbm{1}\left[W^{(2)}\sigma\!\bigl{(}W^{(1)}\bm{x}+b^{(1)}\bigr{)}+b^{(2)}>0\right], |  |

with ReLU activation σ\sigma and the following weight structure:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| first-layer weights W(1)W^{(1)} | W(1)ijW^{(1)}\_{ij} | ∈\in | {−1,0,1}\{-1,0,1\} |  |
| first-layer bias b(1)b^{(1)} | b(1)ib^{(1)}\_{i} | == | 1−∑j[W(1)ij=+1]1-\!\sum\_{j}[W^{(1)}\_{ij}=+1] |  |
| second-layer weights W(2)W^{(2)} | W(2)iW^{(2)}\_{i} | ∈\in | β⋅{0,1}\beta\cdot\{0,1\} |  |
| second-layer bias b(2)b^{(2)} | b(2)b^{(2)} | == | (1−β)/2(1-\beta)/{2} |  |
| global negation β\beta | β\beta | ∈\in | {−1,1}\{-1,1\} |  |

Table 1: DFCN construction.

The central reason we construct a discrete neural network with [Definition˜2.6](#S2.Thmtheorem6 "Definition 2.6 (DFCN). ‣ 2.2 DFCN–DNF correspondence ‣ 2 Preliminaries and intuition ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") is that DFCNs are in one-to-one correspondence with DNF expressions.

###### Proposition 2.7 (DNF-DFCN bijection).

For fixed nn, there is a bijection between *(i)* parameter vectors θ\theta satisfying the restrictions in [Table˜1](#S2.T1 "In Definition 2.6 (DFCN). ‣ 2.2 DFCN–DNF correspondence ‣ 2 Preliminaries and intuition ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"), up to row permutations of W(1)W^{(1)}, and
*(ii)* DNF formulas over nn variables, up to clause permutations and allowing for a global β\beta negation.

###### Proof.

For the proof see Appendix [B.7](#A2.SS7.SSS0.Px1 "Proof of proposition 2.7 ‣ B.7 Proofs from Mingard et al. (2019) ‣ Appendix B Notes on complexity measures ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data").∎

[Figure˜1](#S1.F1 "In 1.2 Related work ‣ 1 Introduction ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") illustrates how the DFCN construction intuitively works. A tt-clause DNF can represent any Boolean function with tt 1s. Each clause can be represented in a single row of the first layer of the DFCN, with the ReLU activation and correctly set bias term returning 1 if and only if the clause is satisfied. The second layer of 11s and 0s then acts as the OR operator (with a 0 ignoring the clause).
The parameter β\beta is there to ensure symmetry between a function and its complement.
As DFCNs and DNFs are in bijection, given a sufficient width, there also exist multiple DFCNs expressing the same Boolean function ff. We define 𝒲(1)f\mathcal{W}^{(1)}\_{f} as the set of all matrices W(1)W^{(1)} that express ff. We now relate the weight norm to the DNF complexity.

###### Definition 2.8.

We set the norm of the weight matrices (W(1),W(2))(W^{(1)},W^{(2)}) to

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∥W(k)∥1=∑ij|W(k)ij|=∑ij𝟙[W(k)ij≠0],\displaystyle\|W^{(k)}\|\_{1}=\sum\_{ij}|W^{(k)}\_{ij}|=\sum\_{ij}\mathbbm{1}[{W^{(k)}\_{ij}\neq 0}], |  | (4) |

k∈{1,2}k\in\{1,2\}.
∥θ∥1=∥W(1)∥1+∥W(2)∥1\|\theta\|\_{1}=\|W^{(1)}\|\_{1}+\|W^{(2)}\|\_{1} denotes the overall norm of the DFCN.

∥W(1)∥1\|W^{(1)}\|\_{1} corresponds to the number of non-zero entries in W(1)W^{(1)}, which is equivalent to the number of literals in the DNF representation, allowing us to relate this quantity to K(f)K{(f)}.

###### Proposition 2.9.

For ff represented as a DFCN fθf\_{\theta}, the complexity is given by

|  |  |  |  |
| --- | --- | --- | --- |
|  | K(f)=minW(1)∈𝒲(1)f∥W(1)∥1.\displaystyle K{(f)}=\underset{W^{(1)}\in\mathcal{W}^{(1)}\_{f}}{\min}\|W^{(1)}\|\_{1}. |  | (5) |

###### Proof.

This directly follows from [Proposition˜2.7](#S2.Thmtheorem7 "Proposition 2.7 (DNF-DFCN bijection). ‣ 2.2 DFCN–DNF correspondence ‣ 2 Preliminaries and intuition ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"), see [Section˜B.2](#A2.SS2 "B.2 Relating DNF complexity to weight norm ‣ Appendix B Notes on complexity measures ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") for more details.
∎

The “minimum representation” DFCN (right grey panel of [Figure˜1](#S1.F1 "In 1.2 Related work ‣ 1 Introduction ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")) has the lowest K(f)K{(f)}; equivalently, ∥W(1)∥1\|W^{(1)}\|\_{1} is minimized.
This construction lets us directly link low weight norm to simple functions.
[Proposition˜2.7](#S2.Thmtheorem7 "Proposition 2.7 (DNF-DFCN bijection). ‣ 2.2 DFCN–DNF correspondence ‣ 2 Preliminaries and intuition ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") completes the picture of the following equivalent ways to express any Boolean function ff:

* •

  String representation: We can represent ff using a binary string of output values, ordered by input.
* •

  DNF representation: We can represent ff using a DNF Φf\Phi\_{f}.
* •

  DFCN representation: We can represent ff by a DFCN of width ≥2n−1\geq 2^{n-1}.

## 3 Untrained neural networks

In this section we provide empirical results indicating that individual functions with low DNF complexity occupy a much larger fraction of parameter space than complex functions do. This bias towards simple functions influences training ([Section˜4](#S4 "4 Trained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")).

### 3.1 A DFCN induced prior over Boolean functions

Our goal is to understand which Boolean functions a *randomly
initialised* DFCN is most likely to compute.
Because, by Proposition [2.7](#S2.Thmtheorem7 "Proposition 2.7 (DNF-DFCN bijection). ‣ 2.2 DFCN–DNF correspondence ‣ 2 Preliminaries and intuition ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"), a depth-two DFCN
is fully specified once its first-layer weight matrix
W(1)W^{(1)} is fixed, the most agnostic prior is to draw each entry of W(1)W^{(1)} independently and uniformly from the ternary set {−1,0,1}\{-1,0,1\}. After W(1)W^{(1)} is chosen we flip an unbiased coin for a global sign β∈{−1,1}\beta\in\{-1,1\}. If at least one hidden unit in a row in W(1)W^{(1)} is non-zero, we set W(2)i=βW^{(2)}\_{i}=\beta, else W(2)i=0W^{(2)}\_{i}=0. The two bias vectors are deterministic functions of
W(1)W^{(1)} and β\beta, see Definition [2.6](#S2.Thmtheorem6 "Definition 2.6 (DFCN). ‣ 2.2 DFCN–DNF correspondence ‣ 2 Preliminaries and intuition ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data").

###### Definition 3.1 (Prior probability).

Let {θ}\{\theta\} denote the finite set of weight vectors θ\theta produced by the sampling procedure above, its size is |{θ}|=2⋅3n2n−1|\{\theta\}|=2\cdot 3^{n2^{n-1}}. For a Boolean function ff we define

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f)=|{θ∈{θ}:fθ=f}||{θ}|,\displaystyle P(f)\;=\;\frac{\bigl{|}\{\theta\in\{\theta\}:f\_{\theta}=f\}\bigr{|}}{|\{\theta\}|}, |  | (6) |

i.e. the fraction of all admissible parameters that implement
ff.

Because each weight setting occupies a unit cell of equal size,
P(f)P(f) is proportional to the volume of weight space
assigned to ff. Boolean functions ff whose string representation can be implemented by many different weight configurations naturally claim a larger share of this volume.

### 3.2 Simplicity bias in P(f)P(f)

P(f)P(f) has emerged as a strong predictor of generalisation (Mingard et al., [2021](#bib.bib45); Valle-Pérez & Louis, [2020](#bib.bib67)). Under a Bayesian update with 01-likelihood on mm samples, the posterior weight of any interpolating function ff is exactly proportional to P(f)P(f). Moreover, the PAC–Bayesian bound

|  |  |  |
| --- | --- | --- |
|  | ϵ(f)≤1−exp(−lnP(f)+ln(δ/2m)m−1)\epsilon(f)\leq 1-\exp\!\Bigl{(}\frac{-\ln P(f)+\ln(\delta/2m)}{m-1}\Bigr{)} |  |

implies that larger P(f)P(f) yields tighter expected generalisation error.

Empirical studies showed that the equivalent prior of continuous FCNs is heavily biased toward simple functions.
Motivated by general arguments on overparameterised learners from (Dingle et al., [2018](#bib.bib19)), Valle-Pérez et al. ([2018](#bib.bib68)) observed that
P(f)≲ 2−KLZ(f)+𝒪(1)P(f)\;\lesssim\;2^{-K\_{\mathrm{LZ}}(f)+\mathcal{O}(1)}, where KLZ(f)K\_{\mathrm{LZ}}(f) is the Lempel-Ziv complexity of the string representation of ff.
While these findings suggested a fundamental connection between function probability and complexity, they relied on a complexity metric that lacks a connection to network architecture.
In contrast, DNF complexity K(f)K{(f)} provides a more interpretable measure with explicit ties to the network, as it directly counts the minimal literals needed to express the function (see [Section˜B.3](#A2.SS3 "B.3 Important differences between 𝐾(𝑓) and 𝐾_{𝐿𝑍}(𝑓) ‣ Appendix B Notes on complexity measures ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") for a further discussion on complexity metrics). This connection to the DFCN allows us to derive analytical bounds on P(f)P(f) in the next section.

| Function class | Scaling | Complexity | log2(P)/K\log\_{2}{(P)}/K ratio |
| --- | --- | --- | --- |
| Constant | 0≤1−P(f)≤2−O(0pt(4/3)n)\displaystyle 0\leq 1-P(f)\leq 2^{-O(0pt(4/3)^{n})} | O(1) | 0pt(4/3)n\displaystyle 0pt(4/3)^{n} |
| tt-entropy (t=O(n))(t=O(n)) | 2−O(0ptt(4/3)n)\displaystyle 2^{-O(0ptt(4/3)^{n})} | O(nt) | 0pt(4/3)n/n0pt(4/3)^{n}/n |
| kk-parity | 2−Θ(0ptk2n−1)2^{-\Theta(0ptk2^{n-1})} | k2k−1\displaystyle k2^{k-1} | 0pt2n−k\displaystyle 0pt2^{n-k} |

Table 2: Scaling laws in P(f)P(f) as a function of complexity. We only show the leading order terms, valid when 0pt≫(3/4)n0pt\gg(3/4)^{n}.

### 3.3 Understanding P(f)P(f) v.s. K(f)K{(f)}

[Figure˜2](#S3.F2 "In 3.3 Understanding 𝑃(𝑓) v.s. 𝐾(𝑓) ‣ 3 Untrained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")(a) compares the empirical prior probability
P(f)P(f) obtained from 10810^{8} i.i.d. parameter draws to the DNF complexity K(f)K{(f)} for n=4n=4 (see [Appendix˜C](#A3 "Appendix C Approximating 𝑃(𝑓) by sampling ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") for full details).
Only 631 of the total 6553665536 functions were not found, indicating that P(f)≲10−8P(f)\lesssim 10^{-8} for these functions. Each datapoint is a function. The minimum complexity constant function (blue) is the most frequent function, with the random tt-entropy functions (reds) occupying the upper part of the envelope, and kk-parity (greens) the lower.
[Figure˜2](#S3.F2 "In 3.3 Understanding 𝑃(𝑓) v.s. 𝐾(𝑓) ‣ 3 Untrained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")(b) shows the dependence of P(f)P(f) on nn for some function types. kk-parity fall much faster with nn than the tt-entropy with t=1,2,4t=1,2,4. Can we predict P(f)P(f) v.s. K(f)K{(f)} at large nn?

###### Theorem 3.2.

We require αw≥1\alpha\_{w}\geq 1 to satisfy full expressivity. To leading order, for the three function classes defined in this section, P(f)P(f) scales as [Table˜2](#S3.T2 "In 3.2 Simplicity bias in 𝑃(𝑓) ‣ 3 Untrained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")

###### Proof.

See [Section˜D.2](#A4.SS2 "D.2 Function class: constant ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") for bounds on the constant function, [Section˜D.3](#A4.SS3 "D.3 Function class: entropy ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") for tt-entropy, [Section˜D.4](#A4.SS4 "D.4 Function class: parity ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") for kk-parity.
∎

We study three canonical families of functions (full descriptions in [Appendix˜D](#A4 "Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")). Constant functions: ff outputs the same label on every input – K(f)=0K(f)=0 (minimum representation has W(1)=0W^{(1)}=0). kk-parity: Sparse parity on the first kk bits – K(f)=k2k−1K{(f)}=k2^{k-1}. tt-entropy: Exactly tt ones and 2n−t2^{n}\!-\!t zeros – K(f)≤nmin(t,2n−t)K{(f)}\;\leq\;n\min\!\bigl{(}t,2^{n}\!-\!t\bigr{)}. These classes of functions allow us to explore a broad range of complexities. We summarise the most relevant bounds and scaling laws in [Table˜2](#S3.T2 "In 3.2 Simplicity bias in 𝑃(𝑓) ‣ 3 Untrained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") (for a full discussion of bounds and assumptions, see [Appendix˜D](#A4 "Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")).

Valle-Pérez et al. ([2018](#bib.bib68)) predicted that log2(P(f))≤−aK+b\log\_{2}(P(f))\leq-aK+b for some constants a,ba,b (independent of KK), and empirically observed that for a large class of functions, the bound was an equality.
[Table˜2](#S3.T2 "In 3.2 Simplicity bias in 𝑃(𝑓) ‣ 3 Untrained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") provides theoretical results showing that for the typical tt-entropy function, this scaling is a function not of K(f)K{(f)}, but 0ptt(4/3)n0ptt(4/3)^{n}, multiplying the complexity term by 0pt(4/3)n/n0pt(4/3)^{n}/n. However, for kk-parity, P(f)P(f) scales as 0ptk2n−10ptk2^{n-1}, suppressing the complexity term by a significantly larger factor, 0pt2n−k0pt2^{n-k} (which is dependent on complexity). This extra suppression predicts that P(f)P(f) will occupy a wide envelope – with some functions of low complexity but also low probability (a concrete example of predictions are presented in (Dingle et al., [2020](#bib.bib20))). This is visualised in [Figure˜2](#S3.F2 "In 3.3 Understanding 𝑃(𝑓) v.s. 𝐾(𝑓) ‣ 3 Untrained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")(c).

![Refer to caption](/html/2505.24060/assets/fig/main/plot_priors.png)


Figure 2: 
Prior probability P(f)P(f) vs. DNF complexity K(f)K{(f)} for n=4n=4.
The hard cutoff at P(f)=10−8P(f)=10^{-8} reflects sampling constraints from 10810^{8} parameter draws. (a) Each point represents a Boolean function, with constant functions (blue, K=0K=0) dominating the parameter space. Low-complexity functions occupy exponentially larger volumes, with kk-parity (greens) suppressed compared to tt-entropy (reds) of equal complexity. (b) Function probability scaling with input dimension nn shows kk-parity probability decreasing much faster than tt-entropy, matching theoretical bounds in Table [2](#S3.T2 "Table 2 ‣ 3.2 Simplicity bias in 𝑃(𝑓) ‣ 3 Untrained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"). (c) Asymptotic bounds on P(f)P(f) for large nn values. The black point is parity.

## 4 Trained neural networks

We use n=7n=7. All models are trained on random subsets
S⊂{0,1}nS\subset\{0,1\}^{n} of size m∈{16,32,64,96}m\in\{16,32,64,96\}, where the rest of the set {0,1}n\{0,1\}^{n} is used as a test set. Results are averaged over ten independent draws. All runs use the DFCN of [Definition˜2.6](#S2.Thmtheorem6 "Definition 2.6 (DFCN). ‣ 2.2 DFCN–DNF correspondence ‣ 2 Preliminaries and intuition ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") with
αw=2\alpha\_{w}=2 to ensure overparameterisation.

### 4.1 Training algorithms

As DFCNs have a discrete weight space, they cannot be straightforwardly trained using SGD. While SGD variants adapted to discrete weights exist (Hubara et al., [2016](#bib.bib33)), we instead employ a fully Bayesian MCMC algorithm, due to its easier interpretability, and an oracle algorithm as described below. We also implement a steepest descent random search algorithm. See Appendix [E](#A5 "Appendix E Trained networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") for full details.

Metropolis–Hastings (Alg. [1](#alg1 "Algorithm 1 ‣ Proposal distribution. ‣ E.2 Metropolis-Hastings algorithm ‣ Appendix E Trained networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")).
:   A Metropolis-Hastings algorithm based on the acceptance
    probability
    α=min{1,e−κΔℒ(θ)e−λΔ∥θ∥1}\alpha=\min\!\bigl{\{}1,e^{-\kappa\Delta\mathcal{L}(\theta)}e^{-\lambda\Delta\|\theta\|\_{1}}\bigr{\}},
    where Δ\Delta denotes the difference between steps, ℒ\mathcal{L} is the MSE error, κ\kappa is an inverse-temperature hyperparameter and λ\lambda is the weight–decay coefficient.

Min norm oracle (Alg. [2](#alg2 "Algorithm 2 ‣ E.3 Min norm Oracle algorithm ‣ Appendix E Trained networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")).
:   This is an oracle that returns the minimal complexity DNF compatible with the training set SS
    obtained by exhaustive search.

Greedy SGD–like (Alg. [3](#alg3 "Algorithm 3 ‣ E.4 SGD-like algorithm ‣ Appendix E Trained networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")).
:   This performs greedy local optimisation. At every step θt\theta\_{t}, we evaluate the loss of every possible neighbour of
    (W(1),W(2))(W^{(1)},W^{(2)}) with Hamming distance one to θt−1\theta\_{t-1}. We find the set of neighbours which maximally improve the batch error (minimising the loss), picking uniformly from the lowest-norm neighbours with probability pp, otherwise picking uniformly from the entire set with probability 1−p1-p. The hyperparameter pp acts like a weight decay parameter: larger pp results in a larger bias towards minimum norm functions and hence towards functions of small K(f)K{(f)}.

### 4.2 Weight decay adds an additional bias in the posterior

For [Algorithm˜1](#alg1 "In Proposal distribution. ‣ E.2 Metropolis-Hastings algorithm ‣ Appendix E Trained networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") we choose κ=1000\kappa=1000, which approximates the likelihood term e−κΔℒ(θ)e^{-\kappa\Delta\mathcal{L}(\theta)} in the MCMC sampling as a 01-likelihood 𝟙[fθ(S)=f∗]\mathbbm{1}[f\_{\theta}(S)=f^{\!\*}]. With the uniform prior over θ\theta defined in Definition [3.1](#S3.Thmtheorem1 "Definition 3.1 (Prior probability). ‣ 3.1 A DFCN induced prior over Boolean functions ‣ 3 Untrained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"), the posterior over Boolean functions ff is obtained by marginalising:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Pλ(f∣S)\displaystyle P\_{\lambda}(f\mid S) | =∑{θ:fθ=f}𝟙[fθ(S)=f∗]e−λ∥θ∥1P(θ)∑θ𝟙[fθ(S)=f∗]e−λ∥θ∥1P(θ)\displaystyle=\frac{\displaystyle\sum\nolimits\_{\{\theta:f\_{\theta}=f\}}\mathbbm{1}[f\_{\theta}(S)=f^{\!\*}]e^{-\lambda\|\theta\|\_{1}}P(\theta)}{\displaystyle\sum\nolimits\_{\theta}\mathbbm{1}[f\_{\theta}(S)=f^{\!\*}]e^{-\lambda\|\theta\|\_{1}}P(\theta)} |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ≃e−λK(f)Pλ=0(f∣S)𝔼f∼Pλ=0(⋅∣S)[e−λK(f)],\displaystyle\simeq{\displaystyle\frac{e^{-\lambda K{(f)}}P\_{\lambda=0}(f\mid S)}{\mathbb{E}\_{f\sim P\_{\lambda=0}(\cdot\mid S)}[e^{-\lambda K{(f)}}]}}, |  | (7) |

where Pλ=0(f∣S)P\_{\lambda=0}(f\mid S) is just the posterior induced by a 01-likelihood, P(f∣S)∝𝟙[fθ(S)=f∗]P(f)P(f\mid S)\propto\mathbbm{1}[f\_{\theta}(S)=f^{\!\*}]P(f).
We have assumed that
∑{θ:fθ=f}e−λ∥θ∥1\sum\_{\{\theta:f\_{\theta}=f\}}e^{-\lambda\|\theta\|\_{1}}
is dominated by the smallest attainable norm
∥θ∥1\|\theta\|\_{1} for a given ff, and that ∥θ∥1≃∥W(1)∥1\|\theta\|\_{1}\simeq\|W^{(1)}\|\_{1}, which gets more accurate for larger nn since the parameter space is largely dominated by W(1)W^{(1)}. As the norm is directly related to the complexity (Proposition [2.9](#S2.Thmtheorem9 "Proposition 2.9. ‣ 2.2 DFCN–DNF correspondence ‣ 2 Preliminaries and intuition ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")), weight decay approximately acts as a multiplicative bias e−λK(f)e^{-\lambda K{(f)}} that further sharpens the simplicity bias in P(f)P(f).

![Refer to caption](/html/2505.24060/assets/x2.png)


Figure 3: 
Inductive biases of trained DFCNs (n=7n=7) with [Algorithm˜1](#alg1 "In Proposal distribution. ‣ E.2 Metropolis-Hastings algorithm ‣ Appendix E Trained networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") (a) shows how the inductive bias of DFCNs towards lower complexity functions allows them to find such functions more easily than higher complexity functions. It also shows that weight decay increases these biases, being able to achieve 100% test accuracy on some functions. See [Appendix˜E](#A5 "Appendix E Trained networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") for a list of the functions used. (b) and (c) show heatmaps of W(1)W^{(1)} during training at different test accuracy checkpoints with a target function of 4-parity for no weight decay (λ=0\lambda=0) and with weight decay (λ=0.01\lambda=0.01), respectively. Both panels show how the network learns simple representations of the target function, with weight decay managing to learning the exact DNF representation.

The results of training DFCNs with [Algorithm˜1](#alg1 "In Proposal distribution. ‣ E.2 Metropolis-Hastings algorithm ‣ Appendix E Trained networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") are shown in [Figure˜3](#S4.F3 "In 4.2 Weight decay adds an additional bias in the posterior ‣ 4 Trained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"). [Figure˜3](#S4.F3 "In 4.2 Weight decay adds an additional bias in the posterior ‣ 4 Trained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")(a) shows the inductive bias of the network: higher complexity functions are harder for the network to learn due to the large bias towards low complexity functions. Hence, at zero training error, the network only achieves good test accuracy for simple target functions. Figures [3](#S4.F3 "Figure 3 ‣ 4.2 Weight decay adds an additional bias in the posterior ‣ 4 Trained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")(b) and (c) show heatmaps of W(1)W^{(1)} during training at different test accuracy checkpoints for a 4-parity target function. We see that for no weight decay (λ=0\lambda=0), the network trains towards a simple representation of the target function (resulting in a sparse W(1)W^{(1)} with many zeros), but adding weight decay (λ=0.01\lambda=0.01) greatly improves the generalisation of the network by further increasing the bias towards low complexity functions. At 100% test accuracy ([Figure˜3](#S4.F3 "In 4.2 Weight decay adds an additional bias in the posterior ‣ 4 Trained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")(c)), we see that the network has learned the exact minimal DNF representation – i.e. optimal features – of the target function, with 4-parity requiring eight clauses in its minimal DNF.

This gives an intuitive explanation for the general empirical observation that weight-decay improves test performance and the general hypothesis that flatter minima (higher volume or prior weight P(f)P(f)) generalise better (He et al., [2020](#bib.bib32); Li et al., [2018](#bib.bib41); Tessier et al., [2022](#bib.bib66)).

### 4.3 The special case of parity ([Figure˜4](#S4.F4 "In Greedy SGD-like. ‣ 4.3 The special case of parity (Figure˜4) ‣ 4 Trained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"))

##### MCMC without weight-decay (λ=0\lambda=0).

As κ≫1\kappa\gg 1, the posterior is dominated by the prior Pλ=0P\_{\lambda=0}.
Simple parities (k≤4k\leq 4) are eventually learned, while
k=6,7k=6,7 never exceed chance level despite more data.
Their posterior mass is simply too small for the algorithm to find them
within the allotted budget, mirroring the extreme rarity observed in [Figure˜2](#S3.F2 "In 3.3 Understanding 𝑃(𝑓) v.s. 𝐾(𝑓) ‣ 3 Untrained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"). Weight norms stay high and almost
kk-independent. See [Section˜B.6](#A2.SS6 "B.6 Why does parity generalise badly? ‣ Appendix B Notes on complexity measures ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") for a discussion on why more data actually harms generalisation for high-parity target functions.

##### MCMC with weight-decay (λ>0\lambda>0).

Penalising the norm dramatically improves generalisation when a
low-norm representation exists.
For k=1k=1 the network reaches perfect accuracy after m=64m=64 examples and its weight norm drops to a much lower value than without weight-decay. These performance gains from weight-decay persist up to k=4k=4, beyond that the minimum representation is so large that the decay term can no longer offset the small prior probability ([4.2](#S4.Ex4 "4.2 Weight decay adds an additional bias in the posterior ‣ 4 Trained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")).

##### Min norm Oracle.

The oracle provides the Bayes-optimal
accuracy achievable if the learner always selects the
lowest-norm DNF that interpolates SS. The accuracy gap between the weight-decay sampler and the oracle is small for k≤4k\leq 4, demonstrating that the algorithm actually discovers the minimal complexity DNF in practice even though it is capable of expressing much more complicated DNFs for the given string representation.

##### Greedy SGD-like.

We train on parity as well as the other function families with the greedy SGD-like algorithm, see Fig. LABEL:fig:app:discrete\_trained\_plot in the Appendix. The learning curves look qualitatively similar to MCMC, showing that SGD behaves Bayesian in our DFCN setting, as also claimed in (Mingard et al., [2021](#bib.bib45)).

![Refer to caption](/html/2505.24060/assets/x3.png)


Figure 4: 
Training statistics for kk-parity target functions
(a), (b) and (c) show training and test accuracies for various kk-parity target functions for the MCMC algorithm ([Algorithm˜1](#alg1 "In Proposal distribution. ‣ E.2 Metropolis-Hastings algorithm ‣ Appendix E Trained networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")) without weight decay (λ=0\lambda=0), with weight decay (λ=0.01\lambda=0.01) and an oracle algorithm ([Algorithm˜2](#alg2 "In E.3 Min norm Oracle algorithm ‣ Appendix E Trained networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")), respectively. Weight decay improves test accuracy for all k<5k<5 functions. For 77-parity (the most complex function for an n=7n=7 input DFCN), the model is strongly biased against this function; the more data we give it, the worse the test accuracy will be. (d), (e) and (f) show the weight norms for each of the training algorithms, clearly showing that weight decay greatly lowers the norm compared to no weight decay.

## 5 Discussion

State-of-the-art neural networks are so large and the data they ingest so heterogeneous that we can usually only fully understand curated sub-problems – for instance, modular arithmetic (Nanda et al., [2023](#bib.bib47)). To make causal statements about representation learning we need a small, exactly-solvable test-bed in which (i) the target function’s complexity is tunable, model-independent and human-interpretable, (ii) learned representations live in an easily interpretable space and (iii) the inductive bias and learned functions can be precisely understood.

Our DFCN offers precisely that – because it represents a DNF, the complexity of the learning problem is controlled directly by the number and size of clauses of the target function ff – the complexity K(f)K{(f)}. K(f)K{(f)} is therefore simultaneously meaningful for the data, the hypothesis class, and the parameters. Furthermore, the one-to-one mapping between weights and logical clauses lets us derive analytic expressions for the prior P(f)P(f), turning qualitative notions of “simplicity bias” from (Valle-Pérez et al., [2018](#bib.bib68)) into quantitative, testable predictions for generalisation.
In this sense, the DFCN plays the same role in deep-learning theory that the Ising model plays in statistical physics: it is the minimal, exactly computable system that still exhibits the phenomena we care about.
Sliding along the complexity axis within a single framework demonstrates how complexity, inductive bias, training dynamics, and generalisation interact.

Specifically, we analytically demonstrated that DFCNs with uniform sampled weights (i) induce a strong simplicity bias in the distribution over Boolean function P(f)P(f). (ii) This bias in the prior directly determines generalisation with a Bayesian learning algorithm, as best seen for high-parity Boolean functions where the bias against complex functions is so strong that it cannot overcome the prior, even with additional data points. (iii) Weight decay amplifies this simplicity bias, learning the minimum representation, inducing feature learning. This explains why it improves performance on simple target functions but not on inherently complex ones.

##### Limitations.

Our work focuses on discrete networks with Boolean inputs, which provides analytical tractability but leaves a gap between our theory and typical continuous deep learning applications. The sampling algorithms become computationally intractable for large nn, even though this is the interesting regime concerning the bounds in [Table˜2](#S3.T2 "In 3.2 Simplicity bias in 𝑃(𝑓) ‣ 3 Untrained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"). Furthermore, our training algorithms do not capture all properties of continuous optimisation with SGD or other optimisers like Adam.

##### Future directions.

One strength of this model is its ability to directly study the effect of optimiser hyperparameters. Exploiting this to study other phenomena observed in continuous neural networks such as grokking and neural collapse, via tuning of the weight decay parameter or increasing the width of the DFCN, would be interesting to explore.
The most difficult task will be to develop suitable and interpretable optimiser measures that allow for bounds on the prior P(f)P(f). This research direction should also better understand how different optimisers and training schemes influence the posterior, or whether a Bayesian formulation is even possible at all.
Furthermore, improving the bounds on P(f)P(f) as a function of K(f)K{(f)}, and understanding the properties of complexity measures (see [Appendix˜B](#A2 "Appendix B Notes on complexity measures ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") for some preliminary work), may provide additional insight. This would allow for more precise generalisation bounds as a function of K(f)K{(f)}. Finally, providing generalisation bounds on the oracle algorithm would shed more light on the effect of weight decay on generalisation.

##### Acknowledgements

The authors would like to thank Yoonsoo Nam, Sofia Fausone, Richie Yeung and Marta Bielinska for fruitful discussions of the core concepts in this paper.

## References

* Abbe et al. (2023)

  Abbe, E., Boix-Adsera, E., and Misiakiewicz, T.
  Sgd learning on neural networks: leap complexity and saddle-to-saddle dynamics, 2023.
  URL <https://arxiv.org/abs/2302.11055>.
* Abbe et al. (2024)

  Abbe, E., Boix-Adsera, E., and Misiakiewicz, T.
  The merged-staircase property: a necessary and nearly sufficient condition for SGD learning of sparse functions on two-layer neural networks, 2024.
  URL <http://arxiv.org/abs/2202.08658>.
* Abbe et al. (2025)

  Abbe, E., Cornacchia, E., Hązła, J., and Kougang-Yombi, D.
  Learning high-degree parities: The crucial role of the initialization, 2025.
  URL <http://arxiv.org/abs/2412.04910>.
* Allender et al. (2005)

  Allender, E., Hellerstein, L., McCabe, P., Pitassi, T., and Saks, M.
  Minimizing dnf formulas and ac0 circuits given a truth table.
  *Electronic Colloquium on Computational Complexity (ECCC)*, 01 2005.
* Basri et al. (2020)

  Basri, R., Galun, M., Geifman, A., Jacobs, D., Kasten, Y., and Kritchman, S.
  Frequency bias in neural networks for input of non-uniform density.
  In *Proceedings of the 37th International Conference on Machine Learning*, pp.  685–694. PMLR, 2020.
  URL <https://proceedings.mlr.press/v119/basri20a.html>.
  ISSN: 2640-3498.
* Belkin (2021)

  Belkin, M.
  Fit without fear: remarkable mathematical phenomena of deep learning through the prism of interpolation, 2021.
  URL <https://arxiv.org/abs/2105.14368>.
* Bhattamishra et al. (2022)

  Bhattamishra, S., Patel, A., Kanade, V., and Blunsom, P.
  Simplicity bias in transformers and their ability to learn sparse boolean functions.
  *arXiv preprint arXiv:2211.12316*, 2022.
* Blais & Tan (2015)

  Blais, E. and Tan, L.-Y.
  Approximating boolean functions with depth-2 circuits.
  *SIAM Journal on Computing*, 44(6):1583–1600, 2015.
* Bordelon et al. (2024)

  Bordelon, B., Atanasov, A., and Pehlevan, C.
  How feature learning can improve neural scaling laws, 2024.
  URL <http://arxiv.org/abs/2409.17858>.
* Bowman & Montufar (2022)

  Bowman, B. and Montufar, G.
  Spectral bias outside the training set for deep networks in the kernel regime.
  In *Proceedings of the 36th International Conference on Neural Information Processing Systems*, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc.
  ISBN 9781713871088.
* Bozoukov (2025)

  Bozoukov, M.
  Uncovering branch specialization in inceptionv1 using k sparse autoencoders.
  *arXiv preprint arXiv:2504.11489*, 2025.
* Brown et al. (2020)

  Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D.
  Language models are few-shot learners, 2020.
  URL <https://arxiv.org/abs/2005.14165>.
* Cammarata et al. (2020)

  Cammarata, N., Goh, G., Carter, S., Schubert, L., Petrov, M., and Olah, C.
  Curve detectors.
  *Distill*, 2020.
  doi: 10.23915/distill.00024.003.
  https://distill.pub/2020/circuits/curve-detectors.
* Canatar et al. (2021)

  Canatar, A., Bordelon, B., and Pehlevan, C.
  Spectral bias and task-model alignment explain generalization in kernel regression and infinitely wide neural networks.
  *Nature communications*, 12(1):1–12, 2021.
* Chizat & Bach (2020)

  Chizat, L. and Bach, F.
  Implicit bias of gradient descent for wide two-layer neural networks trained with the logistic loss.
  In Abernethy, J. and Agarwal, S. (eds.), *Proceedings of Thirty Third Conference on Learning Theory*, volume 125 of *Proceedings of Machine Learning Research*, pp.  1305–1338. PMLR, 09–12 Jul 2020.
  URL <https://proceedings.mlr.press/v125/chizat20a.html>.
* Chizat et al. (2019)

  Chizat, L., Oyallon, E., and Bach, F.
  On lazy training in differentiable programming.
  *Advances in neural information processing systems*, 32, 2019.
* Cohen et al. (2021)

  Cohen, O., Malka, O., and Ringel, Z.
  Learning curves for deep neural networks: A gaussian field theory perspective.
  *arxiv*, 3(2):023034, 2021.
  ISSN 2643-1564.
  doi: 10.1103/PhysRevResearch.3.023034.
  URL <http://arxiv.org/abs/1906.05301>.
* Delétang et al. (2023)

  Delétang, G., Ruoss, A., Grau-Moya, J., Genewein, T., Wenliang, L. K., Catt, E., Cundy, C., Hutter, M., Legg, S., Veness, J., and Ortega, P. A.
  Neural networks and the chomsky hierarchy, 2023.
  URL <https://arxiv.org/abs/2207.02098>.
* Dingle et al. (2018)

  Dingle, K., Camargo, C. Q., and Louis, A. A.
  Input–output maps are strongly biased towards simple outputs.
  *Nature Communications*, 9(1):1–7, 2018.
* Dingle et al. (2020)

  Dingle, K., Pérez, G. V., and Louis, A. A.
  Generic predictions of output probability based on complexities of inputs and outputs.
  *Scientific Reports*, 10(1):1–9, 2020.
* Dingle et al. (2024)

  Dingle, K., Alaskandarani, M., Hamzi, B., and Louis, A. A.
  Exploring simplicity bias in 1d dynamical systems.
  *Entropy*, 26(5):426, 2024.
* Dominé et al. (2025)

  Dominé, C. C. J., Anguita, N., Proca, A. M., Braun, L., Kunin, D., Mediano, P. A. M., and Saxe, A. M.
  From lazy to rich: Exact learning dynamics in deep linear networks, 2025.
  URL <https://arxiv.org/abs/2409.14623>.
* Donhauser et al. (2021)

  Donhauser, K., Wu, M., and Yang, F.
  How rotational invariance of common kernels prevents generalization in high dimensions.
  In *Proceedings of the 38th International Conference on Machine Learning*, pp.  2804–2814. PMLR, 2021.
  URL <https://proceedings.mlr.press/v139/donhauser21a.html>.
  ISSN: 2640-3498.
* Elhage et al. (2021)

  Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., et al.
  A mathematical framework for transformer circuits.
  *Transformer Circuits Thread*, 1(1):12, 2021.
* Everett et al. (2024)

  Everett, K., Xiao, L., Wortsman, M., Alemi, A. A., Novak, R., Liu, P. J., Gur, I., Sohl-Dickstein, J., Kaelbling, L. P., Lee, J., and Pennington, J.
  Scaling exponents across parameterizations and optimizers, 2024.
  URL <http://arxiv.org/abs/2407.05872>.
* Fel et al. (2024)

  Fel, T., Béthune, L., and Lampinen, A. K.
  Understanding visual feature reliance through the lens of complexity.
  *arxiv*, 2024.
* Geifman et al. (2020)

  Geifman, A., Yadav, A., Kasten, Y., Galun, M., Jacobs, D., and Ronen, B.
  On the similarity between the laplace and neural tangent kernels.
  In *Advances in Neural Information Processing Systems*, volume 33, pp.  1451–1461. Curran Associates, Inc., 2020.
  URL <https://proceedings.neurips.cc/paper/2020/hash/1006ff12c465532f8c574aeaa4461b16-Abstract.html>.
* Geifman et al. (2022)

  Geifman, A., Galun, M., Jacobs, D., and Basri, R.
  On the spectral bias of convolutional neural tangent and gaussian process kernels, 2022.
  URL <http://arxiv.org/abs/2203.09255>.
* Geirhos et al. (2018)

  Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., and Brendel, W.
  Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness.
  In *International conference on learning representations*, 2018.
* Ghorbani et al. (2020)

  Ghorbani, B., Mei, S., Misiakiewicz, T., and Montanari, A.
  When do neural networks outperform kernel methods?
  In *Advances in Neural Information Processing Systems*, volume 33, pp.  14820–14830. Curran Associates, Inc., 2020.
  URL <https://proceedings.neurips.cc/paper/2020/hash/a9df2255ad642b923d95503b9a7958d8-Abstract.html>.
* Goldblum et al. (2023)

  Goldblum, M., Finzi, M., Rowan, K., and Wilson, A. G.
  The no free lunch theorem, kolmogorov complexity, and the role of inductive biases in machine learning.
  *arXiv preprint arXiv:2304.05366*, 2023.
* He et al. (2020)

  He, F., Liu, T., and Tao, D.
  Why resnet works? residuals generalize.
  *IEEE Transactions on Neural Networks and Learning Systems*, PP:1–14, 02 2020.
  doi: 10.1109/TNNLS.2020.2966319.
* Hubara et al. (2016)

  Hubara, I., Courbariaux, M., Soudry, D., El-Yaniv, R., and Bengio, Y.
  Binarized neural networks.
  In *Advances in Neural Information Processing Systems*, volume 29. Curran Associates, Inc., 2016.
  URL <https://proceedings.neurips.cc/paper_files/paper/2016/hash/d8330f857a17c53d217014ee776bfd50-Abstract.html>.
* Jacot et al. (2018)

  Jacot, A., Gabriel, F., and Hongler, C.
  Neural tangent kernel: Convergence and generalization in neural networks.
  In *Advances in neural information processing systems*, pp.  8571–8580, 2018.
* Jumper et al. (2021)

  Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., Tunyasuvunakool, K., Bates, R., Žídek, A., Potapenko, A., Bridgland, A., Meyer, C., Kohl, S., Ballard, A., Cowie, A., Romera-Paredes, B., Nikolov, S., Jain, R., Adler, J., and Hassabis, D.
  Highly accurate protein structure prediction with alphafold.
  *Nature*, 596:583–589, 07 2021.
  doi: 10.1038/s41586-021-03819-2.
* Kalimeris et al. (2019)

  Kalimeris, D., Kaplun, G., Nakkiran, P., Edelman, B., Yang, T., Barak, B., and Zhang, H.
  Sgd on neural networks learns functions of increasing complexity.
  *Advances in neural information processing systems*, 32, 2019.
* Kohavi & Jha (2009)

  Kohavi, Z. and Jha, N.
  *Switching and finite automata theory, third edition*, volume 9780521857482.
  Cambridge University Press, United Kingdom, January 2009.
  ISBN 9780521857482.
  doi: 10.1017/CBO9780511816239.
  Publisher Copyright: © The McGraw-Hill Companies Inc.and © Z. Kohavi and N. Jha 2010.
* Lee et al. (2020)

  Lee, J., Xiao, L., Schoenholz, S. S., Bahri, Y., Novak, R., Sohl-Dickstein, J., and Pennington, J.
  Wide neural networks of any depth evolve as linear models under gradient descent.
  *arxiv*, 2020(12):124002, 2020.
  ISSN 1742-5468.
  doi: 10.1088/1742-5468/abc62b.
  URL <http://arxiv.org/abs/1902.06720>.
* Lempel & Ziv (1976)

  Lempel, A. and Ziv, J.
  On the complexity of finite sequences.
  *IEEE Transactions on information theory*, 22(1):75–81, 1976.
* Li et al. (2024)

  Li, C., Liang, Y., Shi, Z., Song, Z., and Zhou, T.
  Fourier circuits in neural networks and transformers: A case study of modular arithmetic with multiple inputs.
  *arXiv preprint arXiv:2402.09469*, 2024.
* Li et al. (2018)

  Li, H., Xu, Z., Taylor, G., Studer, C., and Goldstein, T.
  Visualizing the loss landscape of neural nets, 2018.
  URL <https://arxiv.org/abs/1712.09913>.
* Mei et al. (2018)

  Mei, S., Montanari, A., and Nguyen, P.-M.
  A mean field view of the landscape of two-layer neural networks.
  *Proceedings of the National Academy of Sciences*, 115(33), July 2018.
  ISSN 1091-6490.
  doi: 10.1073/pnas.1806579115.
  URL <http://dx.doi.org/10.1073/pnas.1806579115>.
* Meurer et al. (2017)

  Meurer, A., Smith, C. P., Paprocki, M., Čertík, O., Kirpichev, S. B., Rocklin, M., Kumar, A., Ivanov, S., Moore, J. K., Singh, S., Rathnayake, T., Vig, S., Granger, B. E., Muller, R. P., Bonazzi, F., Gupta, H., Vats, S., Johansson, F., Pedregosa, F., Curry, M. J., Terrel, A. R., Roučka, v., Saboo, A., Fernando, I., Kulal, S., Cimrman, R., and Scopatz, A.
  Sympy: symbolic computing in python.
  *PeerJ Computer Science*, 3:e103, January 2017.
  ISSN 2376-5992.
  doi: 10.7717/peerj-cs.103.
  URL <https://doi.org/10.7717/peerj-cs.103>.
* Mingard et al. (2019)

  Mingard, C., Skalse, J., Valle-Pérez, G., Martínez-Rubio, D., Mikulik, V., and Louis, A. A.
  Neural networks are a priori biased towards boolean functions with low entropy.
  *arXiv preprint arXiv:1909.11522*, 2019.
* Mingard et al. (2021)

  Mingard, C., Valle-Pérez, G., Skalse, J., and Louis, A. A.
  Is sgd a bayesian sampler? well, almost.
  *Journal of Machine Learning Research*, 22(79):1–64, 2021.
* Mingard et al. (2025)

  Mingard, C., Rees, H., Valle-Pérez, G., and Louis, A. A.
  Deep neural networks have an inbuilt occam’s razor.
  *Nature Communications*, 16(1):220, 2025.
* Nanda et al. (2023)

  Nanda, N., Chan, L., Lieberum, T., Smith, J., and Steinhardt, J.
  Progress measures for grokking via mechanistic interpretability, 2023.
  *URL https://arxiv. org/abs/2301.05217*, 2023.
* Naveh et al. (2021)

  Naveh, G., David, O. B., Sompolinsky, H., and Ringel, Z.
  Predicting the outputs of finite deep neural networks trained with noisy gradients.
  *Physical Review E*, 104(6):064301, 2021.
* Neal (2012)

  Neal, R. M.
  *Bayesian learning for neural networks*, volume 118.
  Springer Science & Business Media, 2012.
* O’Donnell (2014)

  O’Donnell, R.
  *Analysis of boolean functions*.
  Cambridge University Press, 2014.
* Olah et al. (2020)

  Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., and Carter, S.
  Zoom in: An introduction to circuits.
  *Distill*, 5(3):e00024–001, 2020.
* Olsson et al. (2022)

  Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Askell, A., Bai, Y., Chen, A., et al.
  In-context learning and induction heads.
  *arXiv preprint arXiv:2209.11895*, 2022.
* Ortiz-Jiménez et al. (2021)

  Ortiz-Jiménez, G., Moosavi-Dezfooli, S.-M., and Frossard, P.
  What can linearized neural networks actually say about generalization?, 2021.
  URL <https://arxiv.org/abs/2106.06770>.
* Palma et al. (2019)

  Palma, G. D., Kiani, B. T., and Lloyd, S.
  Random deep neural networks are biased towards simple functions, 2019.
  URL <https://arxiv.org/abs/1812.10156>.
* Quine (1952)

  Quine, W. V.
  The problem of simplifying truth functions.
  *American Mathematical Monthly*, 59:521–531, 1952.
  URL <https://api.semanticscholar.org/CorpusID:124965557>.
* Rahaman et al. (2019)

  Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F., Bengio, Y., and Courville, A.
  On the spectral bias of neural networks.
  In *International Conference on Machine Learning*, pp.  5301–5310. PMLR, 2019.
* Refinetti et al. (2021)

  Refinetti, M., Goldt, S., Krzakala, F., and Zdeborová, L.
  Classifying high-dimensional gaussian mixtures: Where kernel methods fail and neural networks succeed, 2021.
  URL <http://arxiv.org/abs/2102.11742>.
* Ren et al. (2024)

  Ren, J., Guo, Q., Yan, H., Liu, D., Zhang, Q., Qiu, X., and Lin, D.
  Identifying semantic induction heads to understand in-context learning.
  *arXiv preprint arXiv:2402.13055*, 2024.
* Ridout et al. (2024)

  Ridout, S., Nemenman, I., Louis, A., Mingard, C., Grabarczyk, R., Dingle, K., Valle Pérez, G., and London, C.
  Bounds on learning with power-law priors.
  In *APS March Meeting Abstracts*, volume 2024, pp.  T28–006, 2024.
* Schoenholz et al. (2017)

  Schoenholz, S. S., Gilmer, J., Ganguli, S., and Sohl-Dickstein, J.
  DEEP INFORMATION PROPAGATION.
  *arxiv*, 2017.
* Simon et al. (2021)

  Simon, J. B., Dickens, M., and DeWeese, M. R.
  Neural tangent kernel eigenvalues accurately predict generalization.
  *arXiv preprint arXiv:2110.03922*, 2021.
* Smith & Le (2017)

  Smith, S. L. and Le, Q. V.
  A bayesian perspective on generalization and stochastic gradient descent.
  *CoRR*, abs/1710.06451, 2017.
  URL <http://arxiv.org/abs/1710.06451>.
* Teney et al. (2024)

  Teney, D., Nicolicioiu, A. M., Hartmann, V., and Abbasnejad, E.
  Neural redshift: Random networks are not random functions.
  In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp.  4786–4796, 2024.
* Teney et al. (2025a)

  Teney, D., Jiang, L., Gogianu, F., and Abbasnejad, E.
  Do we always need the simplicity bias? looking for optimal inductive biases in the wild, 2025a.
  URL <http://arxiv.org/abs/2503.10065>.
* Teney et al. (2025b)

  Teney, D., Nicolicioiu, A., Hartmann, V., and Abbasnejad, E.
  Neural redshift: Random networks are not random functions, 2025b.
  URL <https://arxiv.org/abs/2403.02241>.
* Tessier et al. (2022)

  Tessier, H., Gripon, V., Léonardon, M., Arzel, M., Hannagan, T., and Bertrand, D.
  Rethinking weight decay for efficient neural network pruning.
  *Journal of Imaging*, 8(3):64, March 2022.
  ISSN 2313-433X.
  doi: 10.3390/jimaging8030064.
  URL <http://dx.doi.org/10.3390/jimaging8030064>.
* Valle-Pérez & Louis (2020)

  Valle-Pérez, G. and Louis, A. A.
  Generalization bounds for deep learning.
  *arXiv preprint arXiv:2012.04115*, 2020.
* Valle-Pérez et al. (2018)

  Valle-Pérez, G., Camargo, C. Q., and Louis, A. A.
  Deep learning generalizes because the parameter-function map is biased towards simple functions.
  *arXiv preprint arXiv:1805.08522*, 2018.
* Yang & Salman (2020)

  Yang, G. and Salman, H.
  A fine-grained spectral perspective on neural networks, 2020.
  URL <http://arxiv.org/abs/1907.10599>.
* Zhang et al. (2016)

  Zhang, C., Bengio, S., Hardt, M., Recht, B., and Vinyals, O.
  Understanding deep learning requires rethinking generalization.
  *arXiv preprint arXiv:1611.03530*, 2016.

## Appendix A A brief review of generalisation in neural networks

### A.1 Empirical studies of simplicity bias in neural networks

An increasing amount of work shows that randomly initialised standard neural network architectures possess an implicit “simplicity bias”: that is, they prefer to learn functions that are algorithmically simple or low in complexity far more frequently than complex ones.

##### Bias toward low-complexity functions in untrained neural networks

Valle-Pérez et al. ([2018](#bib.bib68)) first quantified the bias of fully-connected neural networks on Boolean data by sampling functions from randomly initialised fully-connected networks and measuring their complexity via Lempel-Ziv compression as a proxy for Kolmogorov complexity.
Using a bound derived from algorithmic information theory (Dingle et al., [2018](#bib.bib19)), they found that the upper bound, P(f)≤2−KLZ(f)+𝒪(1)P(f)\leq 2^{-K\_{LZ}{(f)}+\mathcal{O}(1)}, was tight for small FCNs on boolean data (KLZ(f)K\_{LZ}{(f)} is the Lempel-Ziv complexity of ff, explained in [Section˜B.1](#A2.SS1 "B.1 A primer on Lempel-Ziv complexity ‣ Appendix B Notes on complexity measures ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")). This means simple functions are exponentially more likely to be realised by a random network than ones with a high Lempel-Ziv complexity (Dingle et al., [2024](#bib.bib21)).
Mingard et al. ([2019](#bib.bib44)) built on this observation, proving that, at initialisation, perceptrons are biased towards Boolean functions with low output entropy, meaning functions that output mostly 0’s or 1’s (which tend to be simpler) are favoured. This bias closely follows a Zipf-law or power-law distribution in function probability, indicating a heavy skew toward a few simple functions (see also (Valle-Pérez et al., [2018](#bib.bib68))).

##### Inductive bias of different architectures

Different architectures, initialisation schemes and activation functions lead to different inductive biases in randomly initialised neural networks (Teney et al., [2025a](#bib.bib64)). Schoenholz et al. ([2017](#bib.bib60)) showed a phase transition from ordered to chaotic information propagation in randomly initialised neural networks, depending on the variance of the random Gaussian weights. Teney et al. ([2024](#bib.bib63)) systematically examined random untrained networks with various activation functions, measuring their complexity in terms of their Fourier spectrum, polynomial expansion and (LZ) compressibility.
They showed that standard multilayer perceptrons with ReLU or GELU activations strongly prefer low-frequency functions – effectively smooth input-output mappings – across different depths and weight scales. The opposite is true for Gaussian or sinusoidal activation functions, producing a very different prior in function space, where the inductive bias prefers higher-frequency (generally more complex) functions.
See (Yang & Salman, [2020](#bib.bib69)) for a similar kernel-based analysis.

##### Inductive bias and real world data

Different neural networks have different kinds of inductive biases. How much these biases improve or worsen generalisation depends on the structure of the target function.
A common example is training an FCN and a CNN on an image dataset like CIFAR10: both will perform better than random, but the CNN will reach a significantly higher test accuracy.
Convolutional networks trained on images have been found to latch onto simple, low-level cues (like textures or colours) rather than more complex global structures. Geirhos et al. ([2018](#bib.bib29)) showed that ImageNet-trained CNNs are strongly biased toward texture recognition rather than object shape.
When presented with images where texture and shape cause conflict, CNNs predominantly follow the texture. Texture cues are, in a sense “simpler” or more immediate statistical features of images (requiring only local filtering), whereas global shape integration is more complex. This bias towards easy-to-pick-up features can hurt robustness, but it is an instance of the network’s preference for a simple explanation of the data (here, classifying by texture) when one exists. See (Fel et al., [2024](#bib.bib26)) for a further discussion of the complexity of features in CNNs and how they arise during training.

Another line of work has examined transformer architectures for algorithmic or logical tasks. Bhattamishra et al. ([2022](#bib.bib7)) found that Transformers have a simplicity bias analogous to deep networks: they more readily learn low-sensitivity (sparse) Boolean functions than complex ones.
For instance, a Transformer trained on a Boolean function that depends only on a small subset of input bits (with the rest being irrelevant noise) will generalise well, whereas learning a highly “entangled” function like parity (which depends on all bits) is notably difficult without special measures. This suggests the inductive biases of modern sequence models also favour functions with simple structures (e.g. ones that can be decomposed into a few salient features or rules), even if the models in principle have the capacity to implement very complex mappings
(Fel et al., [2024](#bib.bib26)).

Inductive bias only improves the performance of the model when it aligns with the target. Having well-performing models relies on real-world data being highly structured. Indeed, Goldblum et al. ([2023](#bib.bib31)) argue that this bias is one key reason we can have “general-purpose” models: real-world tasks themselves produce data that are far from fully random and instead have low underlying complexity, benefiting neural networks, which innately favour such low-complexity patterns.
In their experiments, architectures specialised for one domain can often compress or model data from another domain if it shares a low-complexity structure, and even large pretrained language models with random weights preferentially generate low-complexity (compressible) sequences rather than arbitrary complex ones (Goldblum et al., [2023](#bib.bib31)).
This surprising observation – that an untrained GPT-style model already favours simplistic output patterns – reinforces that a great deal of inductive bias comes from architecture alone, not just from gradient descent (Teney et al., [2024](#bib.bib63)).

##### Bias towards low-complexity functions after training

(Zhang et al., [2016](#bib.bib70)) showed that neural networks are expressive enough to fit randomly labelled data. This raised the question of why neural networks trained on non-random labels generalise at all, since they would be perfectly capable of interpolating the training data while having random chance accuracy on the test data. This highlights that, beyond random initialisation, the neural network training process itself introduces an inductive bias.
Mingard et al. ([2021](#bib.bib45)) argue that the posterior distribution over functions retains a simplicity bias – among all functions consistent with the training set, SGD-trained networks tend to land on ones of relatively low complexity. They offer empirical evidence on non-Boolean data that SGD is more likely to learn functions with larger P(f)P(f) (see also (Naveh et al., [2021](#bib.bib48)) for a theoretical explanation using infinitesimal GD with weight decay).
Kalimeris et al. ([2019](#bib.bib36)) provided complementary evidence by examining training dynamics: they observed that SGD learns functions of increasing complexity over time, effectively learning a simple, approximately linear decision boundary in the early epochs and then gradually fitting more complex aspects of the target function in later epochs
Early in training, almost all of the network’s performance can be attributed to a “simple classifier” component, and only with more iterations does the model incorporate higher-order or more complex features. See [Section˜A.3](#A1.SS3 "A.3 Kernel Methods and Spectral Perspectives on Generalization ‣ Appendix A A brief review of generalisation in neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") for the kernel explanation of this phenomenon (or see Canatar et al. ([2021](#bib.bib14))).

### A.2 Mechanistic Interpretability

While the studies discussed in [Section˜A.1](#A1.SS1 "A.1 Empirical studies of simplicity bias in neural networks ‣ Appendix A A brief review of generalisation in neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") treat neural networks mostly as black-box functions, a complementary line of research asks how exactly neural networks internally represent data – in effect, opening the black box to find mechanistic explanations for a network’s behaviour. Mechanistic interpretability seeks to reverse-engineer the specific circuits and algorithms encoded in a trained network’s weights. The goal is to move beyond coarse measures (like complexity or norm-based capacity) and identify neurons, attention heads, layers, or combinations thereof that correspond to meaningful functions or subroutines within the network.

##### Interpreting CNNs – Circuits and feature visualisation

In CNNs, interpretability research has progressed from identifying individual neurons that detect human-interpretable concepts to mapping out multi-neuron interactions, or circuits, that represent higher-level features. Early work used feature visualisation to synthesise an input image that maximally activates a given neuron or layer, revealing the feature that neuron represents (e.g. a neuron might fire for textures like “striped pattern” or specific objects like “dog faces”).

In a series of articles (see e.g. (Olah et al., [2020](#bib.bib51); Elhage et al., [2021](#bib.bib24))), Olah et al. demonstrated that CNNS learn circuits for meaningful visual concepts. One notable example is a “curve detector” circuit (Cammarata et al., [2020](#bib.bib13)): lower-layer neurons detect short curves at various positions; a mid-layer neuron sums these to detect longer curves; that neuron in turn feeds into a higher-layer neuron that detects round objects (like wheels or pupils), together forming a hierarchical circuit for round shapes.
Such precise visual explanations illustrate how a network builds complex features (like detecting an animal face) by composing simpler ones (edges, textures, etc.). However, purely correlational methods face the challenge of polysemantic neurons (neurons that activate on multiple features).

Bozoukov ([2025](#bib.bib11)) used sparse autoencoders on InceptionV1’s mixed layers to isolate more interpretable “feature vectors”, which enabled tracing connections between features across layers to reconstruct circuits. They uncovered branch-specific circuits in the network – e.g. a chain of features detecting animal faces: early layers detect oriented animal parts (faces facing left or right), a next-layer feature combines them into a generic animal face detector, which then branches into specialised detectors for dog faces and dog legs in later layers. This kind of mechanistic story, where one can follow the activation flow through a sequence of feature detectors, represents a state-of-the-art understanding of how CNNs implement complex visual recognition. It provides a satisfying mechanistic explanation (complete with visual evidence) for tasks that the network learned – essentially reverse-engineering a portion of the network’s computation graph in human-understandable terms.

##### Transformer models – Induction heads and algorithms

In Transformers, recent work has identified small-scale circuits inside large models that correspond to specific algorithms the model has learned. A prime example is the discovery of induction heads in Transformers.
Induction heads are particular attention heads that implement a simple copy-and-paste algorithm: given a sequence pattern “[A][B] … [A]”, an induction head learns to attend from the second “[A]” back to the token “[B]” that followed the first “[A]”, effectively retrieving “B” as the predicted continuation. Olsson et al. ([2022](#bib.bib52)) provided multiple lines of evidence that induction heads are the mechanistic basis of in-context learning in Transformers.
They observed that at a certain training stage, models undergo a sudden jump in their ability to do in-context prediction of sequences (for example, continue a list in the style of the prompt), and this coincides with the emergence of one or two attention heads that reliably implement the above copy mechanism. By ablating those heads, the in-context learning ability drops, confirming a causal role. This finding is remarkable because it isolates a transparent algorithm within the black-box: the model learns to implement a memory lookup and retrieval operation entirely through a couple of attention heads (which are simple matrix-weighted operations). It’s a rare case where one can point to specific weights in a large language model and say “this is performing task X via mechanism Y.” Subsequent work has extended this analysis to larger models and more complex behaviours. For instance, Ren et al. ([2024](#bib.bib58)) identified “semantic induction heads” that not only copy tokens but do so in a way that respects word semantics (copying the next word of a repeated sequence even with intervening synonyms), showing the versatility of these circuits.

Beyond induction heads, interpretability researchers have attempted to reverse-engineer entire algorithms learned by Transformers on small tasks. For example, Li et al. ([2024](#bib.bib40)) fully explained how a tiny Transformer performs modular arithmetic. Nanda et al. ([2023](#bib.bib47)) studied grokking in transformers performing modular arithmetic, by reverse-engineering the algorithm the network learned for modular addition. They discovered that the 1-layer Transformer had learned to implement addition by internally converting numbers to a discrete Fourier representation (essentially representing integers as complex phases on a circle) and performing rotations. Accordingly, they defined progress measures for each component of the algorithm (e.g. how well the Fourier conversion sub-circuit was formed) and tracked them during training. They found that training proceeded in three phases: (1) memorization – the model first purely memorizes many training examples, (2) circuit formation – gradually the Fourier addition circuit emerges and gains strength, and (3) cleanup – finally the model prunes away the now-unneeded memorized solutions, relying purely on the general algorithm. What appeared as a sudden “grokking” jump in test accuracy was explained mechanistically as the point when the algorithmic circuit surpassed memorisation in importance.

### A.3 Kernel Methods and Spectral Perspectives on Generalization

Another avenue to understand inductive bias and generalisation is through the lens of kernel methods and spectral analysis. In the infinite-width limit (given the correct parameterisation of the neural network (Everett et al., [2024](#bib.bib25))), training the neural network with SGD corresponds to kernel regression with the so-called Neural Tangent Kernel (NTK) (Jacot et al., [2018](#bib.bib34); Lee et al., [2020](#bib.bib38)). Analysing the eigenfunctions and eigenvalues of these kernels gives insight into what functions the network can learn easily – revealing the network’s bias in function space in more mathematical terms.

##### Spectral bias on the hypersphere

For fully-connected ReLU networks, there are several works that provide a full description of the NTK eigenbasis. For fully-connected ReLU networks with input drawn from the dd-dimensional hypersphere, the NTK is a dot-product kernel. Its eigenfunctions are the spherical harmonics on the hypersphere. The symmetry of the architecture together with the data symmetry leads to a polynomial decay in the eigenfunction kdk^{d} with the degree kk of the spherical harmonics (Basri et al., [2020](#bib.bib5); Geifman et al., [2020](#bib.bib27)).

This means the kernel deems low-frequency spherical harmonics as “important” functions (high eigenvalue), and high-frequency as “hard” functions (low eigenvalue). Intuitively, this quantifies the simplicity bias of the network in a basis of functions: smooth, low-order functions lie in top-eigenvalue eigenspaces, whereas highly oscillatory or complex dependences lie in low-eigenvalue eigenspaces. A direct consequence is a spectral bias in learning. When one trains a neural network (in the kernel regime) or does kernel regression on data, the component of the target function along high-eigenvalue eigenfunctions is learned first and with few samples, while components along low-eigenvalue eigenfunctions require many more samples to fit.

Empirical studies have demonstrated this frequency bias: for instance, when fitting a target function on the unit circle that is a mixture of sines and cosines, neural nets quickly learn the low-frequency modes and only later fit the high-frequency components
(Simon et al., [2021](#bib.bib61)). Rahaman et al. ([2019](#bib.bib56)) showed that standard deep networks trained on regression tasks exhibit a Fourier frequency bias: the error in fitting different Fourier modes is controlled by the mode frequency, with low-frequency signals learned much faster (Canatar et al., [2021](#bib.bib14)).

##### Spectral bias on the hypercube

These arguments transfer to neural networks trained with data from the hypercube. The NTK eigenfunctions on the hypercube are parity functions on subsets of kk input bits, and eigenvalues decrease as kk grows. Thus, a network can learn any function that is, say, an XOR of a few bits relatively easily (those correspond to eigenfunctions with high eigenvalue), but learning the parity of all nn bits (the hardest, k=nk=n eigenfunction) is extremely slow and sample-inefficient – essentially requiring memorization since that function lies in a low-eigenvalue subspace (Simon et al., [2021](#bib.bib61)).
This quantitatively explains why neural nets struggle with high-order parity (a well-known “hard” function class in theory) unless aided by exponential data or special architectural tweaks (Abbe et al., [2025](#bib.bib3)): the inductive bias is simply misaligned with that function. On the other hand, a function like a single-bit identity or a simple conjunction of a few bits has most of its variance in low-order parity components and is learned readily. These insights bridge the gap between the empirical simplicity bias and a theoretical characterisation: networks have an implicit bias toward functions expressible by low-degree polynomials or low-frequency Fourier components, which are precisely the “simple” patterns in the input space.

The spectral bias argument can be extended to other architectures like CNNs (Geifman et al., [2022](#bib.bib28)).

##### Kernels do not do feature learning

There is a rich literature on the sample complexity for kernels which provides a full theory of generalisation for kernels and hence infinitely wide neural networks (Cohen et al., [2021](#bib.bib17)). However, there are several known datasets/target functions where neural networks strongly outperform kernels in terms of their sample complexity (Abbe et al., [2024](#bib.bib2); Refinetti et al., [2021](#bib.bib57); Ghorbani et al., [2020](#bib.bib30); Donhauser et al., [2021](#bib.bib23)). I.e., neural networks learn the target function with a much lower number of datapoints. This is generally attributed to feature learning, the ability of the neural network to adapt its hidden representation (Bordelon et al., [2024](#bib.bib9)). This part of deep learning cannot be understood through a kernel perspective.

## Appendix B Notes on complexity measures

In this section, we will discuss DNF complexity K(f)K{(f)} in more detail, and the alternative complexity measures. We will begin with Lempel-Ziv complexity, used on the string representation by (Valle-Pérez et al., [2018](#bib.bib68)) in [Section˜B.1](#A2.SS1 "B.1 A primer on Lempel-Ziv complexity ‣ Appendix B Notes on complexity measures ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"). We then discuss the relation between K(f)K{(f)} and the weight norm of the DFCN in [Section˜B.2](#A2.SS2 "B.2 Relating DNF complexity to weight norm ‣ Appendix B Notes on complexity measures ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"). In [Sections˜B.3](#A2.SS3 "B.3 Important differences between 𝐾(𝑓) and 𝐾_{𝐿𝑍}(𝑓) ‣ Appendix B Notes on complexity measures ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") and [B.4](#A2.SS4 "B.4 Zipf’s law ‣ Appendix B Notes on complexity measures ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") we discuss properties of “good” complexity measures, and introduce alternatives (e.g. complexity equal to the number of clauses). In [Section˜B.6](#A2.SS6 "B.6 Why does parity generalise badly? ‣ Appendix B Notes on complexity measures ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") we explain why parity generalises so badly (specifically, why adding more data makes training accuracy decrease), and in [Section˜B.7](#A2.SS7 "B.7 Proofs from Mingard et al. (2019) ‣ Appendix B Notes on complexity measures ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") prove the theorems relating DFCNs and DNFs stated in the main text.

### B.1 A primer on Lempel-Ziv complexity

The Lempel–Ziv parsing (Lempel & Ziv, [1976](#bib.bib39)) algorithm proceeds by scanning a finite string xx over any alphabet from left to right, maintaining a dictionary of distinct substrings (or “words”) observed so far. Whenever the algorithm encounters a new substring that is not already in the dictionary, it records that substring as a new dictionary entry; upon completion of the parse, the dictionary size Nw(x)N\_{w}(x) provides a raw measure of complexity. Intuitively, strings composed of a small number of repeated subpatterns yield small dictionaries and thus low complexity, whereas strings exhibiting many novel substrings produce large dictionaries and high complexity.
We use the definition of LZ-complexity from (Dingle et al., [2018](#bib.bib19)) (see Supplementary Note 7)

|  |  |  |  |
| --- | --- | --- | --- |
|  | KLZ(x)=log2(n)2[Nw(x1…n)+Nw(xn…1)],\displaystyle K\_{LZ}{(x)}\;=\;\frac{\log\_{2}(n)}{2}\bigl{[}N\_{w}(x\_{1\ldots n})+N\_{w}(x\_{n\ldots 1})\bigr{]}, |  | (8) |

i.e. the average of the forward and reverse parses, which increases the number of distinct complexity values assignable to strings of a given length.

The Lempel–Ziv complexity, KLZ(x)K\_{LZ}{(x)}, satisfies a number of important asymptotic and finite-size scaling laws. For an ergodic source and in the limit n→∞n\to\infty, one has

|  |  |  |  |
| --- | --- | --- | --- |
|  | limn→∞Nw(x)log2nn=h(x),\displaystyle\lim\_{n\to\infty}\frac{N\_{w}(x)\,\log\_{2}n}{n}\;=\;h(x), |  | (9) |

where h(x)h(x) is the Shannon entropy rate of the source, and consequently KLZ(x)/n→h(x)K\_{LZ}{(x)}/n\to h(x) for almost all long strings (Dingle et al., [2018](#bib.bib19)). Complexity is bounded above by entropy, so strings of low entropy cannot exhibit high KLZ(x)K\_{LZ}{(x)}, while high-entropy strings may nonetheless have simple structure and thus low KLZ(x)K\_{LZ}{(x)}. Empirically, for short to moderate nn, KLZ(x)K\_{LZ}{(x)} often outperforms generic lossless compressors in approximating Kolmogorov complexity (Dingle et al., [2018](#bib.bib19)). As nn increases, the mean and median of the normalised complexity distribution approach unity, and the relative standard deviation σ/μ\sigma/\mu decreases, indicating that typical complexities concentrate sharply around their mean. Strings whose complexity lies well below the mean become exponentially rare in nn, although a small fraction of “maximally complex” strings arise anomalously via simple LZ-specific constructions. Additive and multiplicative constants in KLZ(x)K\_{LZ}{(x)} may be absorbed into fitting parameters when modelling such simplicity-bias phenomena.

### B.2 Relating DNF complexity to weight norm

Each nonzero entry in W(1)W^{(1)} corresponds bijectively to a literal in some conjunctive clause of the DNF (Proposition [2.7](#S2.Thmtheorem7 "Proposition 2.7 (DNF-DFCN bijection). ‣ 2.2 DFCN–DNF correspondence ‣ 2 Preliminaries and intuition ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")). Hence

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∥W(1)∥1=∑i,j𝟙[W(1)ij≠0]=L(Φf).\|W^{(1)}\|\_{1}\;=\;\sum\_{i,j}\mathbbm{1}[W^{(1)}\_{ij}\neq 0]\;=\;L(\Phi\_{f}). |  | (10) |

Minimising this quantity gives us the complexity measure used in the main text, K(f)K{(f)} ([Proposition˜2.9](#S2.Thmtheorem9 "Proposition 2.9. ‣ 2.2 DFCN–DNF correspondence ‣ 2 Preliminaries and intuition ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")).
However, strictly speaking, this is not the true norm of the DFCN, which must take into account the second layer: ∥θ∥1:=∥W(1)∥1+∥W(2)∥1\|\theta\|\_{1}:=\|W^{(1)}\|\_{1}+\|W^{(2)}\|\_{1}. We define a second type of DNF-related complexity

|  |  |  |  |
| --- | --- | --- | --- |
|  | Kθ(f)=minW(1),W(2)(∥W(1)∥1+∥W(2)∥1)\displaystyle K\_{\theta}{(f)}=\underset{W^{(1)},W^{(2)}}{\min}\left(\|W^{(1)}\|\_{1}+\|W^{(2)}\|\_{1}\right) |  | (11) |

Using this norm is equivalent to defining the DNF complexity as the number of literals plus the number of clauses in the minimum representation.

Relating K(f)K{(f)} to Kθ(f)K\_{\theta}{(f)} We can lower bound the number of clauses as a function of the number of literals by creating ⌈K(f)/n⌉\lceil K{(f)}/n\rceil unique clauses with at most nn elements. We can upper bound this by remembering that if clauses span kk columns in W(1)W^{(1)}, we can have no more than 2k−12^{k-1} clauses in the minimum representation. This means we can never have more than 2⌈k⌉−12^{\lceil k\rceil-1} clauses, where kk satisfies K(f)=k2k−1K{(f)}=k2^{k-1}. We can rearrange and take logs to obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | K(f)+⌈K(f)/n⌉≤Kθ(f)≤K(f)+2⌈1+log2K(f)⌉−1.\displaystyle K{(f)}+\lceil K{(f)}/n\rceil\leq K\_{\theta}{(f)}\leq K{(f)}+2^{\lceil 1+\log\_{2}K{(f)}\rceil-1}. |  | (12) |

The maximum of each complexity is parity, K(f)=n22nK{(f)}=\tfrac{n}{2}2^{n} and Kθ(f)=n+122nK\_{\theta}{(f)}=\tfrac{n+1}{2}2^{n}.

Problems with K(f)K{(f)} and Kθ(f)K\_{\theta}{(f)} The maximum complexity for these two measures is O(n2n)O(n2^{n}): ideally, the maximum function should not have complexity greater than 2n+O(1)2^{n}+O(1). Assuming complexity is related to compression – that is, simple functions are highly compressible – we should not “compress” a function to more bits than its string representation, which needs 2n2^{n} bits.
One complexity measure that does satisfy this requirement is double the total number of clauses

|  |  |  |  |
| --- | --- | --- | --- |
|  | KC(f)=2∥W(2)∥1,\displaystyle K\_{C}{(f)}=2\|W^{(2)}\|\_{1}, |  | (13) |

which has a maximum complexity of exactly 2n2^{n}. ⌈K(f)/n⌉\lceil K{(f)}/n\rceil provides the minimum number of clauses we can fit K(f)K{(f)} literals in, and we can again use the fact that if the maximum clause has length kk, we can have no more than 2k−12^{k-1} clauses in the minimum representation to upper bound,

|  |  |  |  |
| --- | --- | --- | --- |
|  | ⌈K(f)/n⌉≤KC(f)≤2⌈1+log2K(f)⌉−1.\displaystyle\lceil K{(f)}/n\rceil\leq K\_{C}{(f)}\leq 2^{\lceil 1+\log\_{2}K{(f)}\rceil-1}. |  | (14) |

Desirable properties We can make some of these observations precise using known results. Blais & Tan ([2015](#bib.bib8)) lists the most important. The Korshunov-Kuznetsov Theorem states that a random boolean function requires Θ(2n/lognloglogn)\Theta(2^{n}/\log{n}\log\log n) clauses, each with an expected n−Θ(logn+loglogn)n-\Theta(\log n+\log\log n) literals. From this, we have the following scaling laws.

| Complexity | constant | tt-entropy | kk-parity | Random | Parity |
| --- | --- | --- | --- | --- | --- |
| K(f)K{(f)} | O(1)O(1) | O(nt)O(nt) | k2k−1k2^{k-1} | Θ(n2nlognloglogn)\Theta\left(\tfrac{n2^{n}}{\log{n}\log\log{n}}\right) | n2n−1n2^{n-1} |
| Kθ(f)K\_{\theta}{(f)} | O(1)O(1) | O((n+1)t)O\left(\left(n+1\right)t\right) | (k+1)2k−1(k+1)2^{k-1} | Θ((n+1)2nlognloglogn)\Theta\left(\tfrac{(n+1)2^{n}}{\log{n}\log\log{n}}\right) | (n+1)2n−1(n+1)2^{n-1} |
| KC(f)K\_{C}{(f)} | O(1)O(1) | O(2t)O(2t) | 2k2^{k} | Θ(2nlognloglogn)\Theta\left(\tfrac{2^{n}}{\log{n}\log\log{n}}\right) | 2n2^{n} |

So from an “optimal compression” point of view, KC(f)K\_{C}{(f)} is the best measure, Kθ(f)K\_{\theta}{(f)} is the most sensible for the DFCN, and K(f)K{(f)} is often the easiest to work with. See [Figure˜5](#A2.F5 "In B.2 Relating DNF complexity to weight norm ‣ Appendix B Notes on complexity measures ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") for empirical results with small nn.

![Refer to caption](/html/2505.24060/assets/x4.png)


Figure 5: Scaling of KLZ(f),K(f),Kθ(f),KC(f)K\_{LZ}{(f)},K{(f)},K\_{\theta}{(f)},K\_{C}{(f)} for random Boolean functions and the parity function for n=2n=2 to n=9n=9 (the length of the string representation of ff is therefore 2n2^{n}).
We expect random functions to be incompressible, and thus have a complexity ≈2n\approx 2^{n} for good complexity measures. LZ complexity is known to satisfy this requirement up to O(1)O(1) terms (Lempel & Ziv, [1976](#bib.bib39)), and for KC(f)K\_{C}{(f)} the worst case is exactly 2n2^{n}. For K(f),Kθ(f)K{(f)},K\_{\theta}{(f)} however, the worst case (parity) is n22n\tfrac{n}{2}2^{n} and n+122n\tfrac{n+1}{2}2^{n}, respectively, and whilst the typical random functions appear to have complexities close to 2n2^{n} for small nn, theoretical results in [Section˜B.2](#A2.SS2 "B.2 Relating DNF complexity to weight norm ‣ Appendix B Notes on complexity measures ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") show that this would change as nn increases.

### B.3 Important differences between K(f)K{(f)} and KLZ(f)K\_{LZ}{(f)}

One important class of functions where the two measures differ significantly is functions with repeating patterns. Consider the string representation of ff.

1. 1.

   Consider the function f="1001"×2n−2f=\texttt{"1001"}\times 2^{n-2}. This function is 2-sparse, represented by the DNF (¬x1∧x2)(\neg x\_{1}\land x\_{2}) (1 clause, 2 literals). As nn increases, its DNF complexity remains fixed at 2. It’s Lempel-Ziv complexity KLZ(f)=C+log2nK\_{LZ}{(f)}=C+\log\_{2}n (constant term CC for encoding the repeating string and the logn\log n term from the repetitions)
2. 2.

   However, if we generate a function ff by repeating the string "01001" (truncating at the end) is not a kk-sparse function. As a result, its K(f)K{(f)} will not be constant.

See [Figure˜6](#A2.F6 "In B.3 Important differences between 𝐾(𝑓) and 𝐾_{𝐿𝑍}(𝑓) ‣ Appendix B Notes on complexity measures ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") for empirical data showing the discrepancy between these measures.

![Refer to caption](/html/2505.24060/assets/x5.png)


Figure 6: K(f)K{(f)} v.s. KLZ(f)K\_{LZ}{(f)} on repeating functions. Dashed lines show KLZ(f)K\_{LZ}{(f)} and solid lines show K(f)K{(f)}.
The red curves show the 2-sparse function f="1001"×2n−2f=\texttt{"1001"}\times 2^{n-2}. K(f)K{(f)} remains constant at 2, and KLZ(f)K\_{LZ}{(f)} grows slowly with nn. By contrast the blue curves show the function generated by the repeated pattern "01001" which yields a function that is not 2-sparse and therefore does not enjoy a small, constant K(f)K{(f)}, but does enjoy a KLZ(f)K\_{LZ}{(f)} that grows at a similar rate to the KLZ(f)K\_{LZ}{(f)} of the 2-sparse function.

Empirical work in (Valle-Pérez et al., [2018](#bib.bib68)) shows that K(f)K{(f)} and KLZ(f)K\_{LZ}{(f)} are correlated. However, they have very different maximum complexities. Given that a function can be represented in its string representation with 2n2^{n} bits – a really good complexity measure should never go above 2n+O(1)2^{n}+O(1). The maximum value of K(f)K{(f)} is n22n\tfrac{n}{2}2^{n}, or O(n2n)O(n2^{n}). In contrast, KLZ(f)K\_{LZ}{(f)} has a worst-case of O(2n)O(2^{n}).

[Figure˜5](#A2.F5 "In B.2 Relating DNF complexity to weight norm ‣ Appendix B Notes on complexity measures ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") shows how the two measures scale for random boolean functions (generated by assigning a 1 or 0 randomly for each input xx) and parity, as a function of nn. Random functions should have complexities close to 2n2^{n} (as they are not compressible beyond their string representation), which is the case for both KLZ(f)K\_{LZ}{(f)} and K(f)K{(f)}. Parity, on the other hand, scales very differently. Future work could determine the fraction of functions with complexity greater than 2n2^{n}.

### B.4 Zipf’s law

Valle-Pérez et al. ([2018](#bib.bib68)) showed that the prior of neural networks is well-described by Zipf’s law. For a dataset of size 2n2^{n}, the probability of randomly initialising to a function P(f)P(f) is a function of the rank of the function R(f)R(f) (where the rank of the most probable function is 1, the second most is 2 and so on) that satisfies

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f)=12nlog21R(f),P(f)=\frac{1}{2^{n}\log{2}}\frac{1}{R(f)}, |  | (15) |

where the first term is a normalisation term given that there are 22n2^{2^{n}} functions. Ridout et al. ([2024](#bib.bib59)) show that when P(f)P(f) satisfies Zipf’s law, a Bayesian learning agent on this prior will learn optimally (for a full discussion on what optimal means, see (Ridout et al., [2024](#bib.bib59))). We find it a useful reference point when considering the scaling laws in [Table˜2](#S3.T2 "In 3.2 Simplicity bias in 𝑃(𝑓) ‣ 3 Untrained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data").

If Zipf’s law is satisfied, the most frequent functions (in our work, the constant functions) should have P(const)∼2−nP(\textrm{const})\sim 2^{-n}. There are 2n2^{n} unique functions with t=1t=1 (a single True input), so these functions occupy a total space of size
12nlog2∑i=3i=2n+2i−1∼2−nn\tfrac{1}{2^{n}\log{2}}\sum\_{i=3}^{i=2^{n}+2}i^{-1}\sim 2^{-n}n, meaning the average 1-entropy function has probability P(f(e)1)∼n2−2nP(f^{(e)}\_{1})\sim n2^{-2n}. (Note that i=1,2i=1,2 correspond to the two constant functions, and we ignore the flipped entropy functions that have only a single zero, which would be of the same order but only include an extra factor of 1/2, not affecting the overall approximation). We will use these results in [Section˜C.1](#A3.SS1 "C.1 Finding the optimum width ‣ Appendix C Approximating 𝑃(𝑓) by sampling ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data").

### B.5 DFCNs vs FCNs

The DFCN is not a typical discrete approximation of a neural network, as the bias terms are all functions of weights, to make sure each neuron represents a clause. One consequence is that P(f)P(f) is strongly width-dependent, unlike standard FCNs on Boolean data (Valle-Pérez et al., [2018](#bib.bib68)). In the limit of infinite width (0pt→∞0pt\rightarrow\infty), at initialisation, P(f)P(f) will be entirely dominated by a constant function ([Lemma˜D.5](#A4.Thmtheorem5 "Lemma D.5 (Lower bound for 𝑃(𝑓^(𝑐))). ‣ D.2 Function class: constant ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")). This is because adding more clauses can only set more inputs to True – eventually, every input will be covered by at least one clause. See [Figure˜8](#A3.F8 "In Appendix C Approximating 𝑃(𝑓) by sampling ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") for an empirical demonstration.

### B.6 Why does parity generalise badly?

In [Section˜4.3](#S4.SS3 "4.3 The special case of parity (Figure˜4) ‣ 4 Trained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") we discuss details about training DFCNs on kk-parity functions. An interesting result is that for the most complex function (7-parity), the test accuracy gets worse the larger the training set. This can be understood by thinking about what a network will predict on unseen test data. Consider an example for n=4n=4 where the target function is 4-parity: "0110100110010110". Let’s assume in the following examples that we train to the minimum norm solution.

1. 1.

   Suppose that we train on the first four bits (m=4m=4). The minimum norm solution for this is 2-parity (parity on the first two bits). Then, ff is just the first 4 bits repeated, 4 x "0110" == "0110011001100110". On the remaining 12 bits, our accuracy is 33%.
2. 2.

   Now we train on the first 8 bits. The minimum norm solution is now 3-parity, 2 x "01101001" == "0110100101101001", which has 0% test accuracy.

This argument can be straightforwardly generalised to larger nn. Choosing training examples in this way, when trying to learn parity generalisation, will get worse the larger the training set.

### B.7 Proofs from Mingard et al. ([2019](#bib.bib44))

##### Proof of proposition [2.7](#S2.Thmtheorem7 "Proposition 2.7 (DNF-DFCN bijection). ‣ 2.2 DFCN–DNF correspondence ‣ 2 Preliminaries and intuition ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")

###### Proof.

Fix an input dimension nn.
Let f:{0,1}n→{0,1}f:\{0,1\}^{n}\to\{0,1\} be an arbitrary Boolean function and set

|  |  |  |  |
| --- | --- | --- | --- |
|  | t=|{𝐯∈{0,1}n:f(𝐯)=1}|.\displaystyle t\;=\;\bigl{|}\{\,\mathbf{v}\in\{0,1\}^{n}\;:\;f(\mathbf{v})=1\}\bigr{|}. |  | (16) |

We adopt the notation for a clause CC as laid out in [Definition˜2.3](#S2.Thmtheorem3 "Definition 2.3 (Literal, clause, DNF). ‣ 2.1 Boolean functions and their disjunctive normal form ‣ 2 Preliminaries and intuition ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data").

Define the set of all DNFs by ℱDNF\mathcal{F}\_{\operatorname{DNF}} and an equivalence relation RDNFR\_{\operatorname{DNF}} to be permutations of DNF clauses such that the set of equivalence classes is ℱDNF/RDNF\mathcal{F}\_{\operatorname{DNF}}/R\_{\operatorname{DNF}}. Similarly, define the set of all DFCNs of width 2n−12^{n-1} by ℱDFCN\mathcal{F}\_{\operatorname{DFCN}} and an equivalence relation ℱDFCN\mathcal{F}\_{\operatorname{DFCN}} to be the permutations of rows in W(1)W^{(1)} of a DFCN such that the set of equivalence classes is ℱDFCN/RDFCN\mathcal{F}\_{\operatorname{DFCN}}/R\_{\operatorname{DFCN}}. We now show that there exists a bijective map 𝒢:ℱDNF/RDNF→ℱDFCN/RDFCN\mathcal{G}:\mathcal{F}\_{\operatorname{DNF}}/R\_{\operatorname{DNF}}\rightarrow\mathcal{F}\_{\operatorname{DFCN}}/R\_{\operatorname{DFCN}}.

##### 𝒢\mathcal{G} is injective

We begin by assuming that t≤2n−1t\leq 2^{n-1}.
Define a depth-two DFCN with layer sizes ⟨n,t,1⟩\langle n,t,1\rangle and β=1\beta=1 (following [Definition˜2.6](#S2.Thmtheorem6 "Definition 2.6 (DFCN). ‣ 2.2 DFCN–DNF correspondence ‣ 2 Preliminaries and intuition ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")).
For i∈{1,…,t}i\in\{1,...,t\}, let 𝐯(i)\mathbf{v}^{(i)} be the ii-th input vector for which ff outputs True such that γi=∑jv(i)j\gamma\_{i}=\sum\_{j}v^{(i)}\_{j} is the number of positive literals in clause CiC\_{i}.
Set

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | W(1)ij\displaystyle W^{(1)}\_{ij} | ={+1if v(i)j=1,−1if v(i)j=0,\displaystyle=\begin{cases}+1&\text{if }v^{(i)}\_{j}=1,\\ -1&\text{if }v^{(i)}\_{j}=0,\end{cases} | b(1)i\displaystyle b^{(1)}\_{i} | =1−γi,\displaystyle=1-\gamma\_{i}, |  | (17) |
|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | W(2)i\displaystyle W^{(2)}\_{i} | =1,\displaystyle=1, | b(2)\displaystyle b^{(2)} | =0.\displaystyle=0. |  | (18) |

For any 𝐯∈{0,1}n\mathbf{v}\in\{0,1\}^{n},

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | zi(𝐯)\displaystyle z\_{i}(\mathbf{v}) | =∑iW(1)ijvj+b(1)i=γi−d(𝐯,𝐯(i))+(1−γi)= 1−d(𝐯,𝐯(i)),\displaystyle=\sum\_{i}W^{(1)}\_{ij}v\_{j}+b^{(1)}\_{i}\;=\;\gamma\_{i}-d(\mathbf{v},\mathbf{v}^{(i)})+(1-\gamma\_{i})\;=\;1-d(\mathbf{v},\mathbf{v}^{(i)}), |  | (19) |

where d(⋅,⋅)d(\cdot,\cdot) denotes the hamming distance.
zi(𝐯)=1z\_{i}(\mathbf{v})=1 iff every literal in CiC\_{i} is satisfied
and zi(𝐯)≤0z\_{i}(\mathbf{v})\leq 0 otherwise.
Since σ(zi)=ReLU(zi)=max(zi,0)\sigma(z\_{i})=\mathrm{ReLU}(z\_{i})=\max(z\_{i},0),

|  |  |  |  |
| --- | --- | --- | --- |
|  | σ(zi(𝐯))=Ci(𝐯)={1if 𝐯(i)=𝐯,0if 𝐯(i)≠𝐯.\displaystyle\sigma\!\bigl{(}z\_{i}(\mathbf{v})\bigr{)}=C\_{i}(\mathbf{v})=\begin{cases}1&\text{if }\mathbf{v}^{(i)}=\mathbf{v},\\ 0&\text{if }\mathbf{v}^{(i)}\neq\mathbf{v}.\end{cases} |  | (20) |

In other words, the output of the first layer is 1 if the clause CiC\_{i} is satisfied and 0 not. W(2)W^{(2)} then effectively acts as an OR operator, giving us

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | fθ(𝐯)\displaystyle f\_{\theta}(\mathbf{v}) | =𝟙[W(2)σ(W(1)𝐯+b(1))+b(2)>0]\displaystyle=\mathbbm{1}[W^{(2)}\,\sigma\!\bigl{(}W^{(1)}\mathbf{v}+b^{(1)}\bigr{)}+b^{(2)}>0] |  | (21) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =C1(𝐯)∨⋯∨Ct(𝐯)=Φf(𝐯)=f(𝐯).\displaystyle=C\_{1}(\mathbf{v})\,\lor\cdots\lor\,C\_{t}(\mathbf{v})\;=\;\Phi\_{f}(\mathbf{v})\;=\;f(\mathbf{v}). |  | (22) |

If t>2n−1t>2^{n-1}, we instead use a network with layer sizes ⟨n,2n−1−t,1⟩\langle n,2^{n-1}-t,1\rangle and let 𝐯(1)\mathbf{v}^{(1)} be the ii-th input vector for which ff outputs False (which must be ≤2n−1\leq 2^{n-1}). We then set β=−1\beta=-1, which negates Wi(2)W\_{i}^{(2)} and sets b(2)=1b^{(2)}=1, giving us the following parameters,

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | W(1)ij\displaystyle W^{(1)}\_{ij} | ={+1if v(i)j=1,−1if v(i)j=0,\displaystyle=\begin{cases}+1&\text{if }v^{(i)}\_{j}=1,\\ -1&\text{if }v^{(i)}\_{j}=0,\end{cases} | b(1)i\displaystyle b^{(1)}\_{i} | =1−γi,\displaystyle=1-\gamma\_{i}, |  | (23) |
|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | W(2)i\displaystyle W^{(2)}\_{i} | =−1,\displaystyle=-1, | b(2)\displaystyle b^{(2)} | =1.\displaystyle=1. |  | (24) |

ziz\_{i} still outputs 1 if the clause CiC\_{i} is satisfied and 0 if not, but W(2)𝐳<0W^{(2)}\mathbf{z}<0 if any of the False clauses are satisfied. Thus, the only way to obtain a positive output inside the indicator function in [Equation˜21](#A2.E21 "In 𝒢 is injective ‣ B.7 Proofs from Mingard et al. (2019) ‣ Appendix B Notes on complexity measures ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") is if all CiC\_{i} are not satisfied (since b(2)=1b^{(2)}=1 brings the value above 0 in this case). This then gives us

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | fθ(𝐯)\displaystyle f\_{\theta}(\mathbf{v}) | =𝟙[W(2)σ(W(1)𝐯+b(1))+b(2)>0]\displaystyle=\mathbbm{1}[W^{(2)}\,\sigma\!\bigl{(}W^{(1)}\mathbf{v}+b^{(1)}\bigr{)}+b^{(2)}>0] |  | (25) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =¬[C1(𝐯)∨⋯∨Ct(𝐯)]=Φf(𝐯)=f(𝐯).\displaystyle=\neg\left[C\_{1}(\mathbf{v})\,\lor\cdots\lor\,C\_{t}(\mathbf{v})\right]\;=\;\Phi\_{f}(\mathbf{v})\;=\;f(\mathbf{v}). |  | (26) |

We can pad the width of the network to be 2n−12^{n-1} by adding rows of zeros to W(1)W^{(1)} and setting the rest of the weights according to [Definition˜2.6](#S2.Thmtheorem6 "Definition 2.6 (DFCN). ‣ 2.2 DFCN–DNF correspondence ‣ 2 Preliminaries and intuition ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"). We have extra degrees of freedom in the permutations of these rows and thus invoke the equivalence relation RDFCNR\_{\operatorname{DFCN}}, proving injectivity of 𝒢\mathcal{G}.

##### 𝒢\mathcal{G} is surjective

Conversely, let a parameter tensor
θ=(W(1),b(1),W(2),b(2),β)\theta=(W^{(1)},b^{(1)},W^{(2)},b^{(2)},\beta)
satisfy the constraints of Table [1](#S2.T1 "Table 1 ‣ Definition 2.6 (DFCN). ‣ 2.2 DFCN–DNF correspondence ‣ 2 Preliminaries and intuition ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")
with hidden width 2n−12^{n-1}.
Keep only the indices ii for which W(2)i=βW^{(2)}\_{i}=\beta;
there are t≤2n−1t\leq 2^{n-1} of them.
For each such ii, define the clause

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Ci\displaystyle C\_{i} | =(⋀{j:W(1)ij=+1}xj)⋀(⋀{j:W(1)ij=−1}¬xj).\displaystyle=\Bigl{(}\!\bigwedge\_{\{j:W^{(1)}\_{ij}=+1\}}x\_{j}\Bigr{)}\;\bigwedge\;\Bigl{(}\!\bigwedge\_{\{j:W^{(1)}\_{ij}=-1\}}\neg x\_{j}\Bigr{)}. |  | (27) |

Following the same reasoning as before, we conclude that

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | fθ(𝐯)\displaystyle f\_{\theta}(\mathbf{v}) | =𝟙[W(2)σ(W(1)𝐯+b(1))+b(2)>0]={C1(𝐯)∨⋯∨Ct(𝐯)if β=1,¬[C1(𝐯)∨⋯∨Ct(𝐯)]if β=−1.\displaystyle=\mathbbm{1}[W^{(2)}\,\sigma\!\bigl{(}W^{(1)}\mathbf{v}+b^{(1)}\bigr{)}+b^{(2)}>0]=\begin{cases}C\_{1}(\mathbf{v})\,\lor\cdots\lor\,C\_{t}(\mathbf{v})&\text{if }\beta=1,\\ \neg\left[C\_{1}(\mathbf{v})\,\lor\cdots\lor\,C\_{t}(\mathbf{v})\right]&\text{if }\beta=-1.\end{cases} |  | (28) |

Thus, every image has a preimage, which further holds true for the equivalence relations imposed on the DNFs and DFCNs, proving that 𝒢\mathcal{G} is surjectivity.

Bijectivity of 𝒢\mathcal{G} follows from injectivity and surjectivity.
See Appendix G of (Mingard et al., [2019](#bib.bib44)) for further details.
∎

## Appendix C Approximating P(f)P(f) by sampling

We approximate P(f)P(f) by Monte Carlo sampling from the uniform prior on network parameters, defined in [Equation˜6](#S3.E6 "In Definition 3.1 (Prior probability). ‣ 3.1 A DFCN induced prior over Boolean functions ‣ 3 Untrained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"):

|  |  |  |  |
| --- | --- | --- | --- |
|  | W(1)ij∼U{−1,0,1},β∼U[−1,1],W(2)=β(unless the corresponding clause is zero),\displaystyle W^{(1)}\_{ij}\sim\mathrm{U}\{-1,0,1\},\quad\beta\sim\mathrm{U}[-1,1],\quad W^{(2)}=\beta\;\text{(unless the corresponding clause is zero)}, |  | (29) |

and estimate

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f)=Pr(fθ=f)=|{θ:fθ=f}|2⋅3n 2n−1.\displaystyle P(f)=\Pr\left(f\_{\theta}=f\right)=\frac{|\{\theta:f\_{\theta}=f\}|}{2\cdot 3^{\,n\,2^{\,n-1}}}. |  | (30) |

Because P(f)P(f) counts how many choices of θ\theta implement ff, it is exactly equivalent to the volume of parameter-space occupied by ff. One may also view P(f)P(f) as a Bayesian prior probability assigned to ff.

In our Monte Carlo sampling to approximate P(f)P(f), we draw 10810^{8} independent parameter samples. Any function ff with P(f)≲10−8P(f)\lesssim 10^{-8} is vanishingly unlikely to appear in our search. For example, when n=4n=4, one computes

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(parity)=(23)! 3−4⋅23≈ 2×10−11,\displaystyle P\left(\mathrm{parity}\right)=(2^{3})!\;3^{-4\cdot 2^{3}}\;\approx\;2\times 10^{-11}, |  | (31) |

which explains why parity is never discovered (it lies three orders of magnitude below our sampling threshold). In fact, out of the 224=65,5362^{2^{4}}=65,536 possible Boolean functions on 44 bits, we fail to encounter 631631 of them even after 10810^{8} draws.

Figure [7](#A3.F7 "Figure 7 ‣ Appendix C Approximating 𝑃(𝑓) by sampling ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") plots the estimated P(f)P(f) for n=3,4,5,7n=3,4,5,7.

* •

  Top three rows: P(f)P(f) vs. K(f)K{(f)}, Kθ(f)K\_{\theta}{(f)} and KC(f)K\_{C}{(f)}. The horizontal line at P=10−8P=10^{-8} marks our effective sampling cutoff; finite-size artefacts appear for large K(f)K{(f)}, especially when n=7n=7. For n=3,4n=3,4 there is a clear log-linear relationship between P(f)P(f) and all complexity measures, with the difference between the maximum and minimum P(f)P(f) at a fixed complexity small relative to the overall range of P(f)P(f). For n=5,7n=5,7, the total range of P(f)P(f) is many orders of magnitude more than we can sample, but the upper bound P(f)∼2−K(f)+O(1)P(f)\sim 2^{-K{(f)}+O(1)} as observed in (Valle-Pérez et al., [2018](#bib.bib68)) still describes the distribution well.
* •

  Penultimate row: P(f)P(f) vs. KLZ(f)K\_{LZ}{(f)}, the Lempel–Ziv string complexity, as studied in (Mingard et al., [2025](#bib.bib46), [2019](#bib.bib44); Valle-Pérez et al., [2018](#bib.bib68)). The relation between P(f)P(f) and KLZ(f)K\_{LZ}{(f)} for n=3,4n=3,4 is much weaker than for DNF complexity K(f)K{(f)}. This is what we might expect, given that DNF complexity is intuitively more appropriate in this case (as K(f)K{(f)} is intricately connected to the architecture in a way KLZ(f)K\_{LZ}{(f)} is not, see [Section˜B.3](#A2.SS3 "B.3 Important differences between 𝐾(𝑓) and 𝐾_{𝐿𝑍}(𝑓) ‣ Appendix B Notes on complexity measures ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")). For n=5,7n=5,7, we are unable to sample enough times to properly compare the distributions.
* •

  Bottom row: P(f)P(f) vs. rank R(f)R(f) (where the most probable function has R=1R=1). The dashed orange line shows Zipf’s law,
  P(f)=(2nln2)−1R(f)−1,\;P(f)=(2^{n}\ln 2)^{-1}\,R(f)^{-1},
  which (Ridout et al., [2024](#bib.bib59)) identifies as the optimal prior for Bayesian learning.

[Figure˜8](#A3.F8 "In Appendix C Approximating 𝑃(𝑓) by sampling ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") shows the effect of the width of the hidden layer on the prior. We show widths αw2n−1\alpha\_{w}2^{n-1} for 0pt=0.5,1,2,40pt=0.5,1,2,4, with the top two rows showing n=4n=4 and the bottom two rows showing n=5n=5. As the width increases, the probability that any input is True increases. We can use [Lemma˜D.4](#A4.Thmtheorem4 "Lemma D.4 (Probability of input 𝐯 being True). ‣ D.1 Utility lemmas ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") to show that the probability that any input is False scales as (1−(2n−1)/3n)αw2n−1(1-(2^{n}-1)/3^{n})^{\alpha\_{w}2^{n-1}}. This expression decreases asymptotically to 0 as 0pt0pt increases. The bottom row in [Figure˜8](#A3.F8 "In Appendix C Approximating 𝑃(𝑓) by sampling ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") shows that the prior is not well-described by Zipf’s law for 0pt>10pt>1, indicating 0pt=10pt=1 gives the optimal width for learning (Ridout et al., [2024](#bib.bib59)). This is an interesting coincidence which we explore in the remainder of this section.

![Refer to caption](/html/2505.24060/assets/fig/app/type1_all.png)


Figure 7: 
Approximation of the prior probability P(f)P(f) by sampling 10810^{8} functions from each prior for n=3,4,5, and 7n=3,4,5\textrm{, and }7.
Top row: P(f)P(f) versus DNF complexity K(f)K{(f)}. Finite-size effects at P(f)=10−8P(f)=10^{-8} (sampling limit) produce artefacts at higher K(f)K{(f)} in the n=7n=7 panel.
Second row: P(f)P(f) versus neural network norm Kθ(f)K\_{\theta}{(f)}.
Third row: P(f)P(f) versus clause complexity K(f)K{(f)}.
Fourth row: P(f)P(f) versus Lempel–Ziv complexity KLZ(f)K\_{LZ}{(f)}, as used in (Mingard et al., [2025](#bib.bib46), [2019](#bib.bib44); Valle-Pérez et al., [2018](#bib.bib68)). Final row: P(f)P(f) versus rank R(f)R(f), with dotted orange lines showing Zipf’s law P(f)=(2nln2)−1R−1P(f)=(2^{n}\ln 2)^{-1}R^{-1} (Ridout et al., [2024](#bib.bib59)).



![Refer to caption](/html/2505.24060/assets/fig/app/width_comparison_w_n4.png)

![Refer to caption](/html/2505.24060/assets/fig/app/width_comparison_w_n5.png)

Figure 8: Coverage of the prior probability P(f)P(f) for top two rows n=4n=4 and bottom two rows n=5n=5 across three network widths w∈{1,2,4}×2n−1w\in\{1,2,4\}\times 2^{\,n-1}, estimated by sampling 10810^{8} functions per prior.
The P(f)P(f) versus DNF complexity K(f)K{(f)}, illu plots illustrate how the constant function’s probability mass grows with width. The plots showing P(f)P(f) versus rank R(f)R(f) (most probable is R=1R=1), with the dotted line marking Zipf’s law P(f)∝R−1P(f)\propto R^{-1}; larger widths exhibit marked departures from this scaling.

### C.1 Finding the optimum width

As stated in the main text, we require a width of αw2n−1\alpha\_{w}2^{n-1} with 0pt≥10pt\geq 1 to guarantee full expressivity.
However, [Table˜2](#S3.T2 "In 3.2 Simplicity bias in 𝑃(𝑓) ‣ 3 Untrained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") tells us that with this scaling, eventually the function space will be entirely dominated by the constant functions unless 0pt∼(3/4)n0pt\sim(3/4)^{n}. Furthermore, it would be completely impractical to use a DFCN with width 2n−12^{n-1} – by n=50n=50 to be fully expressive, you would need more than 101410^{14} neurons.

So, how could we determine the optimum scaling? We assume that the presence of Zipf’s law indicates an optimal prior. We argue this case in [Section˜B.4](#A2.SS4 "B.4 Zipf’s law ‣ Appendix B Notes on complexity measures ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"), and will also make use of the derived results P(const)∼2−nP(\textrm{const})\sim 2^{-n} and P(f(e)1)∼n2−2nP(f^{(e)}\_{1})\sim n2^{-2n}. We use the result from [Equation˜82](#A4.E82 "In Lemma D.8 (Bounds on 𝑡-entropy with 𝑡=1). ‣ D.3 Function class: entropy ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") with p=2n−13np=\tfrac{2^{n}-1}{3^{n}},

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | P(f(e)1)\displaystyle P(f^{(e)}\_{1}) | ≲(1−p)αw2n−1\displaystyle\lesssim\left(1-p\right)^{\alpha\_{w}2^{n-1}} |  | (32) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ≈exp(−αw2n−1p)(p≪1)\displaystyle\approx\exp(-\alpha\_{w}2^{n-1}p)\quad(p\ll 1) |  | (33) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ⟹exp(−αw2n−1p)\displaystyle\Longrightarrow\quad\exp(-\alpha\_{w}2^{n-1}p) | =n 2−n\displaystyle=n\,2^{-n} |  | (34) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | −αw2n−1p\displaystyle-\alpha\_{w}2^{n-1}p | =ln(n 2−n)=−nln2+lnn\displaystyle=\ln(n\,2^{-n})=-\,n\ln 2+\ln n |  | (35) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | αw2n−1\displaystyle\alpha\_{w}2^{n-1} | =nln2−lnnp∼nln2(32)n\displaystyle=\frac{n\ln 2-\ln n}{p}\sim n\ln 2\;\left(\tfrac{3}{2}\right)^{n} |  | (36) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 0pt\displaystyle 0pt | ∼2nln2(34)n.\displaystyle\sim 2n\ln 2\left(\tfrac{3}{4}\right)^{n}. |  | (37) |

We can also determine the width scaling to make the lower bound on P(f(c))P(f^{(c)}) scale as 2−n2^{-n} by using the result in [Lemma˜D.4](#A4.Thmtheorem4 "Lemma D.4 (Probability of input 𝐯 being True). ‣ D.1 Utility lemmas ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") that the probability that an input 𝐯\mathbf{v} is covered with probability (1−p)αw2n−1\left(1-p\right)^{\alpha\_{w}2^{n-1}}, where p=2n−13np=\tfrac{2^{n}-1}{3^{n}}. Assuming independence and taking the Poisson approximation, we get

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | P(f(c))\displaystyle P(f^{(c)}) | ≈exp(−2ne−αw2n−1p)= 2−n=exp(−nln2)\displaystyle\approx\exp\left(-2^{n}e^{-\alpha\_{w}2^{n-1}p}\right)\;=\;2^{-n}\;=\;\exp\left(-n\ln 2\right) |  | (38) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ⟹2ne−αw2n−1p\displaystyle\implies\quad 2^{n}e^{-\alpha\_{w}2^{n-1}p} | =nln2\displaystyle=n\ln 2 |  | (39) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | e−αw2n−1p\displaystyle e^{-\alpha\_{w}2^{n-1}p} | =nln22n\displaystyle=\frac{n\ln 2}{2^{n}} |  | (40) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | −αw2n−1p\displaystyle-\alpha\_{w}2^{n-1}p | =ln(nln2)−nln2\displaystyle=\ln\left(n\ln 2\right)-n\ln 2 |  | (41) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | αw2n−1\displaystyle\alpha\_{w}2^{n-1} | =nln2−ln(nln2)p≈(nln2)(32)n\displaystyle=\frac{n\ln 2-\ln\left(n\ln 2\right)}{p}\approx\left(n\ln 2\right)\,\left(\tfrac{3}{2}\right)^{n} |  | (42) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 0pt\displaystyle 0pt | ∼2nln2(34)n.\displaystyle\sim 2n\ln 2\left(\tfrac{3}{4}\right)^{n}. |  | (43) |

Both methods arrive at the same scaling for 0pt∼n(34)n0pt\sim n\left(\tfrac{3}{4}\right)^{n}. At small nn, this scaling is very close to 1 – explaining why [Figure˜8](#A3.F8 "In Appendix C Approximating 𝑃(𝑓) by sampling ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") shows Zipf’s law for 0pt=10pt=1. To test the proposed optimal scaling more thoroughly, one would have to either make the scaling arguments above more precise or gather empirical evidence at larger nn.

## Appendix D Results relating P(f)P(f) to K(f)K(f)

In this appendix, we relate P(f)P(f) to K(f)K(f) for the various classes of functions discussed in the main text ([Section˜3.3](#S3.SS3 "3.3 Understanding 𝑃(𝑓) v.s. 𝐾(𝑓) ‣ 3 Untrained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")): Constant, kk-parity and tt-entropy functions. We also briefly look at kk-sparse functions, which are not studied in the main text. The width of the DFCN is αw2n−1\alpha\_{w}2^{n-1}, where 0pt≥10pt\geq 1 for the network to be fully expressive.

### D.1 Utility lemmas

###### Lemma D.1 (Lower bound on P(f)P(f)).

We can lower bound P(f)P(f) using

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f)≥12P(f∣β)\displaystyle P(f)\geq\frac{1}{2}P(f\mid\beta) |  | (44) |

as P(β=1)=P(β=−1)=12P(\beta=1)=P(\beta=-1)=\tfrac{1}{2}

###### Lemma D.2 (Lower bound on P(f∣β)P(f\mid\beta) using the minimum representation).

Given a value for β∈{−1,1}\beta\in\{-1,1\}, we have the following lower bound on the conditional probability of ff:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | P(f∣β)\displaystyle P(f\mid\beta) | ≥3−njp!(p3k)0pt2n−1−j,\displaystyle\geq 3^{-nj}p!\left(\frac{p}{3^{k}}\right)^{0pt2^{n-1}-j}, |  | (45) |

where jj is the number of clauses, each of at most length kk.

###### Proof.

After permuting rows, the minimum representation of a function ff can be encoded as follows,

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W(1)=\displaystyle W^{(1)}= | [A⏟j×kBCD]⏟n}0pt2n−1,\displaystyle\underbrace{\left[\begin{array}[]{c|c}\underbrace{A}\_{j\times k}&B\\ \hline\cr C&D\end{array}\right]}\_{\displaystyle n}\left.\vphantom{\begin{bmatrix}A&B\\ E&F\\ C&D\end{bmatrix}}\right\}0pt2^{n-1}, |  | (48) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | W(2)=\displaystyle W^{(2)}= | [1,…,1⏟j,0,…,0⏟0pt2n−1−j]T\displaystyle\begin{bmatrix}\underbrace{1,\dots,1}\_{j},\underbrace{0,\dots,0}\_{0pt2^{n-1}-j}\end{bmatrix}^{T} |  | (49) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | β∈\displaystyle\beta\in | {−1,1}\displaystyle\{-1,1\} |  | (50) |

where AA contains pp clauses each of at most length kk, B=0B=0, C=0C=0 and D=0D=0.
We can vary the clauses (rows) in AA only up to permutation.
How much freedom do we have to vary B,C,DB,C,D? We must set B=0B=0, otherwise kk would not be the maximum length of the minimal representation. If every clause in CC is a copy of one in AA and B=0B=0, we can let DD be anything without affecting the overall function. This gives us the following lower bound:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | P(f∣β)\displaystyle P(f\mid\beta) | ≥j!3−jk⏟A×3−j(n−k)⏟B×(j/3k)0pt2n−1−j⏟C×1⏟D\displaystyle\geq\underbrace{j!3^{-jk}}\_{A}\times\underbrace{3^{-j(n-k)}}\_{B}\times\underbrace{\left(j/3^{k}\right)^{0pt2^{n-1}-j}}\_{C}\times\underbrace{1}\_{D} |  | (51) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ≥3−njj!(j3k)0pt2n−1−j.\displaystyle\geq 3^{-nj}j!\left(\frac{j}{3^{k}}\right)^{0pt2^{n-1}-j}. |  | (52) |

∎

###### Lemma D.3.

Denote 𝒩\mathcal{N} the set of all possible clauses given a β∈{−1,1}\beta\in\{-1,1\}, with N=|𝒩|=3nN=|\mathcal{N}|=3^{n}. Let M=0pt2n−1M=0pt2^{n-1} be the number of clauses drawn i.i.d. uniformly from 𝒩\mathcal{N}. Consider a subset Q⊆𝒩Q\subseteq\mathcal{N} of size q=|Q|q=|Q| containing clauses that we must have at least one copy of, and a disjoint subset of clauses R⊆𝒩R\subseteq\mathcal{N} of size r=|R|r=|R| that we must not have. We can then lower bound P(f)P(f) as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f∣β)=P=Pr(all x∈Q appear at least once, and no y∈R appears),\displaystyle P(f\mid\beta)=P=\Pr\bigl{(}\text{all }x\in Q\text{ appear at least once, and no }y\in R\text{ appears}\bigr{)}, |  | (53) |

where

|  |  |  |  |
| --- | --- | --- | --- |
|  | P=∑i=0q(−1)i(qi)(1−r+iN)0pt2n−1,\displaystyle P=\sum\_{i=0}^{q}(-1)^{i}\binom{q}{i}\Bigl{(}1-\tfrac{r+i}{N}\Bigr{)}^{0pt2^{n-1}}, |  | (54) |

and the union-bound lower bound is given by

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f∣β)≥(N−rN)0pt2n−1[1−q(1−1N−r)0pt2n−1].\displaystyle P(f\mid\beta)\geq\Bigl{(}\tfrac{N-r}{N}\Bigr{)}^{0pt2^{n-1}}\Bigl{[}1-q\bigl{(}1-\tfrac{1}{N-r}\bigr{)}^{0pt2^{n-1}}\Bigr{]}. |  | (55) |

In the rest of this section, we will rely on the first term in [Equation˜55](#A4.E55 "In Lemma D.3. ‣ D.1 Utility lemmas ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"), but as the second term is often very loose, we will often bound it below using an alternative task-specific bound.

###### Proof.

Let

|  |  |  |  |
| --- | --- | --- | --- |
|  | A={every x∈Q appears},B={no draw lies in R}.\displaystyle A=\{\text{every }x\in Q\text{ appears}\},\quad B=\{\text{no draw lies in }R\}. |  | (56) |

Then

|  |  |  |  |
| --- | --- | --- | --- |
|  | P=Pr(A∩B)=Pr(A∣B)Pr(B).\displaystyle P=\Pr(A\cap B)=\Pr(A\mid B)\Pr(B). |  | (57) |

Since each of the MM draws must avoid RR,

|  |  |  |  |
| --- | --- | --- | --- |
|  | Pr(B)=(N−rN)M.\displaystyle\Pr(B)=\Bigl{(}\tfrac{N-r}{N}\Bigr{)}^{M}. |  | (58) |

Conditioned on BB, draws are uniform on the remaining N−rN-r symbols, and by the inclusion–exclusion principle

|  |  |  |  |
| --- | --- | --- | --- |
|  | Pr(A∣B)=∑i=0q(−1)i(qi)((N−r)−iN−r)M.\displaystyle\Pr(A\mid B)=\sum\_{i=0}^{q}(-1)^{i}\binom{q}{i}\Bigl{(}\tfrac{(N-r)-i}{N-r}\Bigr{)}^{M}. |  | (59) |

Combining these and noting
(N−rN)M(N−r−iN−r)M=(N−r−iN)M\bigl{(}\tfrac{N-r}{N}\bigr{)}^{M}\bigl{(}\tfrac{N-r-i}{N-r}\bigr{)}^{M}=(\tfrac{N-r-i}{N})^{M}
yields the exact sum. Truncating after i=1i=1 gives us the union-bound of

|  |  |  |  |
| --- | --- | --- | --- |
|  | Pr(A∣B)≥1−q(1−1N−r)M,\displaystyle\Pr(A\mid B)\geq 1-q\left(1-\frac{1}{N-r}\right)^{M}, |  | (60) |

from which we obtain the union-bound on PP by multiplying with Pr(B)\Pr(B).

∎

###### Lemma D.4 (Probability of input 𝐯\mathbf{v} being True).

Given a fixed input 𝐯∈{0,1}n\mathbf{v}\in\{0,1\}^{n}, the probability that a randomly sampled clause CC covers 𝐯\mathbf{v} (i.e., is True on 𝐯\mathbf{v}) is

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(C(𝐯)=1)=2n−13n.P(C(\mathbf{v})=1)=\frac{2^{n}-1}{3^{n}}. |  | (61) |

Given a DFCN of width αw2n−1\alpha\_{w}2^{n-1}, the probability that any particular input 𝐯\mathbf{v} is True is

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f(𝐯)=1∣β=1)=1−(1−2n−13n)αw2n−1P(f(\mathbf{v})=1\mid\beta=1)=1-\Bigl{(}1-\frac{2^{n}-1}{3^{n}}\Bigr{)}^{\alpha\_{w}2^{n-1}} |  | (62) |

Moreover, the leading-order term for large nn is

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f(𝐯)=1∣β=1)∼0pt2(43)n.\displaystyle P(f(\mathbf{v})=1\mid\beta=1)\sim\frac{0pt}{2}\Bigl{(}\tfrac{4}{3}\Bigr{)}^{n}. |  | (63) |

###### Proof.

A clause is True on input xx if for each variable in the input, the corresponding entry of a clause in DFCN representation is 11 if xi=1x\_{i}=1, is −1-1 if xi=0x\_{i}=0 or is 0 (but ignoring the all 0s case, which is considered False). This means we have 2n−12^{n}-1 clauses that satisfy this criterion. Divide by the total number of clauses, 3n3^{n} for [Equation˜61](#A4.E61 "In Lemma D.4 (Probability of input 𝐯 being True). ‣ D.1 Utility lemmas ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"). Since all clauses are drawn independently, the probability that all clauses give False on a random input 𝐯\mathbf{v} is

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f(𝐯)=0∣β=1)=(1−2n−13n)αw2n−1,\displaystyle P(f(\mathbf{v})=0\mid\beta=1)=\left(1-\frac{2^{n}-1}{3^{n}}\right)^{\alpha\_{w}2^{n-1}}, |  | (64) |

from which [Equation˜62](#A4.E62 "In Lemma D.4 (Probability of input 𝐯 being True). ‣ D.1 Utility lemmas ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") follows.
∎

### D.2 Function class: constant

Denote the class of constant functions as f(c)f^{(c)}, which includes all functions where the output either always gives True or always gives False, regardless of the input.

###### Lemma D.5 (Lower bound for P(f(c))P(f^{(c)})).

We can bound either constant function, P(f(c))P(f^{(c)}) with

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f(c))≥12∑k=02n(−1)k(2nk)(1−kp)αw2n−1,P(f^{(c)})\geq\frac{1}{2}\sum\_{k=0}^{2^{n}}(-1)^{k}\binom{2^{n}}{k}\bigl{(}1-kp\bigr{)}^{\alpha\_{w}2^{n-1}}, |  | (65) |

where p=(2n−13n)p=\left(\tfrac{2^{n}-1}{3^{n}}\right).
When truncating after k=1k=1, we obtain the lower bound

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f(c))≥12(1−2n(1−p)αw2n−1)\displaystyle P(f^{(c)})\geq\tfrac{1}{2}\left(1-2^{n}(1-p)^{\alpha\_{w}2^{n-1}}\right) |  | (66) |

Substituting
p=2n−13n∼(23)np=\frac{2^{n}-1}{3^{n}}\sim\Bigl{(}\tfrac{2}{3}\Bigr{)}^{n},
we get

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f(c))≳12(1−exp(nln2−0pt2(4/3)n))(n→∞).P(f^{(c)})\gtrsim\tfrac{1}{2}\left(1-\exp\Bigl{(}n\ln 2-\tfrac{0pt}{2}(4/3)^{n}\Bigr{)}\right)\quad(n\to\infty). |  | (67) |

###### Proof.

Let n∈ℕn\in\mathbb{N}, M=αw2n−1M=\alpha\_{w}2^{n-1}, and p=2n−13np=\tfrac{2^{n}-1}{3^{n}} ([Lemma˜D.4](#A4.Thmtheorem4 "Lemma D.4 (Probability of input 𝐯 being True). ‣ D.1 Utility lemmas ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")). We fix β=1\beta=1 and aim to bound the function that returns True for all inputs. We label the 2n2^{n} Boolean inputs by 𝐯∈{0,1}n\mathbf{v}\in\{0,1\}^{n}, and for each 𝐯\mathbf{v} let

|  |  |  |  |
| --- | --- | --- | --- |
|  | A𝐯={no clause covers 𝐯},Pr(A𝐯)=(1−p)M.\displaystyle A\_{\mathbf{v}}=\bigl{\{}\text{no clause covers }\mathbf{v}\bigr{\}},\qquad\Pr(A\_{\mathbf{v}})=(1-p)^{M}. |  | (68) |

Then by the principle of inclusion-exclusion, the probability that every input is covered by at least one clause (i.e. no A𝐯A\_{\mathbf{v}} occurs) is exactly

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f(c)∣β=1)=Pr(⋂𝐯∈SA𝐯c)=1−Pr(⋃𝐯∈SA𝐯)=∑k=02n(−1)k∑S⊆{0,1}n|S|=kPr(⋂𝐯∈SA𝐯).\displaystyle P(f^{(c)}\mid\beta=1)=\Pr\Bigl{(}\bigcap\_{\mathbf{v}\in S}A\_{\mathbf{v}}^{c}\Bigr{)}=1-\Pr\Bigl{(}\bigcup\_{\mathbf{v}\in S}A\_{\mathbf{v}}\Bigr{)}=\sum\_{k=0}^{2^{n}}(-1)^{k}\sum\_{\begin{subarray}{c}S\subseteq\{0,1\}^{n}\\ |S|=k\end{subarray}}\Pr\Bigl{(}\bigcap\_{\mathbf{v}\in S}A\_{\mathbf{v}}\Bigr{)}. |  | (69) |

Moreover, for any fixed SS of size kk, one has

|  |  |  |  |
| --- | --- | --- | --- |
|  | Pr(⋂𝐯∈SA𝐯)=(1−pS)M,\displaystyle\Pr\Bigl{(}\bigcap\_{\mathbf{v}\in S}A\_{\mathbf{v}}\Bigr{)}=\bigl{(}1-p\_{S}\bigr{)}^{M}, |  | (70) |

where

|  |  |  |  |
| --- | --- | --- | --- |
|  | pS\displaystyle p\_{S} | =Pr(a single random clause covers at least one 𝐯∈S)\displaystyle=\Pr\bigl{(}\text{a single random clause covers at least one }\mathbf{v}\in S\bigr{)} |  |

Furthermore, by the union bound on the single-clause covering probabilities,

|  |  |  |  |
| --- | --- | --- | --- |
|  | pS≤∑x∈Sp=kp,\displaystyle p\_{S}\leq\sum\_{x\in S}p=kp, |  | (71) |

so that

|  |  |  |  |
| --- | --- | --- | --- |
|  | Pr(⋂𝐯∈SA𝐯)=(1−pS)M≥(1−kp)M.\displaystyle\Pr\Bigl{(}\bigcap\_{\mathbf{v}\in S}A\_{\mathbf{v}}\Bigr{)}=\bigl{(}1-p\_{S}\bigr{)}^{M}\geq(1-kp)^{M}. |  | (72) |

Substituting into the inclusion–exclusion sum gives the valid lower bound

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f(c)∣β=1)≥∑k=02n(−1)k(2nk)(1−kp)M.\displaystyle P(f^{(c)}\mid\beta=1)\geq\sum\_{k=0}^{2^{n}}(-1)^{k}\binom{2^{n}}{k}\bigl{(}1-kp\bigr{)}^{M}. |  | (73) |

Truncating after k=1k=1 gives
P(f(c)∣β=1)≥1−2n(1−p)MP(f^{(c)}\mid\beta=1)\geq 1-2^{n}(1-p)^{M}, and approximating pp to be small, so that (1−p)M≈e−Mp(1-p)^{M}\approx e^{-Mp}, yields the final estimate for PP,

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f(c)∣β=1)≥1−2n(1−p)αw2n−1≈1−2ne−αw2n−1p.\displaystyle P(f^{(c)}\mid\beta=1)\geq 1-2^{n}(1-p)^{\alpha\_{w}2^{n-1}}\approx 1-2^{n}e^{-\alpha\_{w}2^{n-1}p}. |  | (74) |

Using [Lemma˜D.1](#A4.Thmtheorem1 "Lemma D.1 (Lower bound on 𝑃(𝑓)). ‣ D.1 Utility lemmas ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"), we thus have P(f(c))≥12P(f(c)∣β=1)P(f^{(c)})\geq\tfrac{1}{2}P(f^{(c)}\mid\beta=1).

∎

### D.3 Function class: entropy

Consider the class of functions with tt 1s and 2n−t2^{n}-t 0s. We call this function tt-entropy and denote it with f(e)tf^{(e)}\_{t}. If tt is small, the function is simple (consistent with the intuition that low-entropy functions are simple). However, the converse does not hold: some high-entropy functions require a very small number of clauses (e.g. 1-parity needs just one clause: x1x\_{1}).

###### Lemma D.6.

Given a boolean function on nn variables with tt 1s and 2n−t2^{n}-t 0s, we denote RR the set of forbidden clauses with r=|R|r=|R| and N=3nN=3^{n} the total number of possible clauses ([Lemma˜D.3](#A4.Thmtheorem3 "Lemma D.3. ‣ D.1 Utility lemmas ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")). The fraction of clauses that would flip the function value if appended to its DNF is bounded below by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | rN≥(2/3)n−⌊log2(2n−t)⌋\frac{r}{N}\geq(2/3)^{n-\lfloor\log\_{2}(2^{n}-t)\rfloor} |  | (75) |

###### Proof.

Let f:{0,1}n→{0,1}f:\{0,1\}^{n}\to\{0,1\} have exactly t~=2n−t\tilde{t}=2^{n}-t inputs 𝐯(i)∈{𝐯(1),…,𝐯(t~)}\mathbf{v}^{(i)}\in\{\mathbf{v}^{(1)},...,\mathbf{v}^{(\tilde{t})}\} for which f(𝐯(i))=0f(\mathbf{v}^{(i)})=0, with
1≤t~≤2n−11\leq\tilde{t}\leq 2^{n}-1. By filling the largest possible dd-dimensional subspace of the nn-dimensional hypercube, given by d≤⌊log2t~⌋d\leq{\lfloor\log\_{2}\tilde{t}\rfloor}, this corresponds to finding the maximal correlation between these inputs, which minimises the set of not allowed clauses RR.

Given a maximally filled dd-dimensional subspace, the remaining n−dn-d bits for these points must be the same. Since the subspace is fully filled, all possible combinations of inputs in this subspace are exhausted, meaning that unless all entries in the DFCN representation of a clause are zero (which always gives False), one of these subspace inputs must give True. Thus, we need the probability that all remaining n−dn-d entries in a DFCN clause either match the corresponding remaining input bit (1 if xi=1x\_{i}=1, −1-1 if xi=0x\_{i}=0) or are 0, giving

|  |  |  |  |
| --- | --- | --- | --- |
|  | P=(23)n−d−1.\displaystyle P=\left(\frac{2}{3}\right)^{n-d}-1. |  | (76) |

(The −1-1 comes from the all zeros clause.)
Substituting the bound on dd and taking nn to be large gives us

|  |  |  |  |
| --- | --- | --- | --- |
|  | P=rN≥(23)n−⌊log2t~⌋.\displaystyle P=\frac{r}{N}\geq\left(\frac{2}{3}\right)^{n-\lfloor\log\_{2}\tilde{t}\rfloor}. |  | (77) |

∎

###### Lemma D.7 (Upper bound for tt-entropy).

We can upper bound P(f(e)t)P(f^{(e)}\_{t}) with the following

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f(e)t∣β)≲{exp(−0pt2n−1(2/3)n−⌊log2(2n−t)⌋)if β=1,exp(−0pt2n−1(2/3)n−⌊log2t⌋)if β=−1.P(f^{(e)}\_{t}\mid\beta)\lesssim\begin{cases}\exp\left(-0pt2^{n-1}(2/3)^{n-\lfloor\log\_{2}(2^{n}-t)\rfloor}\right)&\text{if }\beta=1,\\ \exp\left(-0pt2^{n-1}(2/3)^{n-\lfloor\log\_{2}t\rfloor}\right)&\text{if }\beta=-1.\end{cases} |  | (78) |

###### Proof.

Let f(e)t:{0,1}n→{0,1}f^{(e)}\_{t}:\{0,1\}^{n}\to\{0,1\} have exactly tt 1s, with 1≤t≤2n−11\leq t\leq 2^{n}-1. For the case of t≤2n−1t\leq 2^{n-1}, we take β=1\beta=1 ([Section˜B.7](#A2.SS7.SSS0.Px1 "Proof of proposition 2.7 ‣ B.7 Proofs from Mingard et al. (2019) ‣ Appendix B Notes on complexity measures ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")), giving us t~=2n−t\tilde{t}=2^{n}-t inputs 𝐯(i)∈{𝐯(1),…,𝐯(t~)}\mathbf{v}^{(i)}\in\{\mathbf{v}^{(1)},...,\mathbf{v}^{(\tilde{t})}\} for which f(e)t(𝐯(i))=0f^{(e)}\_{t}(\mathbf{v}^{(i)})=0. (For t>2n−1t>2^{n-1} we take β=−1\beta=-1 and simply replace t~\tilde{t} with tt.)
Given the set RR, with r=|R|r=|R|, of all forbidden clauses which would flip a 0 output to a 1, and N=3nN=3^{n} total clause options, a valid network must sample all MM clauses outside of RR, giving us an upper bound,

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f(e)t∣β=1)≤(1−rN)M≲exp(−Mr/N).P\bigl{(}f^{(e)}\_{t}\mid\beta=1\bigr{)}\leq\Bigl{(}1-\tfrac{r}{N}\Bigr{)}^{M}\lesssim\exp\bigl{(}-Mr/N\bigr{)}. |  | (79) |

Using the result in [Lemma˜D.6](#A4.Thmtheorem6 "Lemma D.6. ‣ D.3 Function class: entropy ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"), we have r/N≥(2/3)n−⌊log2t~⌋r/N\geq(2/3)^{n-\lfloor\log\_{2}\tilde{t}\rfloor}.
Substituting M=0pt2n−1M=0pt2^{n-1} into
([79](#A4.E79 "Equation 79 ‣ D.3 Function class: entropy ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")) then gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f(e)t∣β=1)≤(1−(2/3)n−⌊log2(2n−t)⌋)0pt2n−1≈exp(−0pt2n−1(2/3)n−⌊log2(2n−t)⌋).P\bigl{(}f^{(e)}\_{t}\mid\beta=1\bigr{)}\leq\left(1-(2/3)^{n-\lfloor\log\_{2}(2^{n}-t)\rfloor}\right)^{0pt2^{n-1}}\approx\exp\left(-0pt2^{n-1}(2/3)^{n-\lfloor\log\_{2}(2^{n}-t)\rfloor}\right). |  | (80) |

∎

###### Lemma D.8 (Bounds on tt-entropy with t=1t=1).

For a function with a single True output 𝐯\mathbf{v} and all else False,

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f(e)1∣β=−1)≤(1−2n−13n)αw2n−1P(f^{(e)}\_{1}\mid\beta=-1)\leq\left(1-\frac{2^{n}-1}{3^{n}}\right)^{\alpha\_{w}2^{n-1}} |  | (81) |

We also have a lower bound

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f(e)1∣β=−1)≥3−n2(1−2n−13n)αw2n−1−n.P(f^{(e)}\_{1}\mid\beta=-1)\geq 3^{-n^{2}}\left(1-\frac{2^{n}-1}{3^{n}}\right)^{\alpha\_{w}2^{n-1}-n}. |  | (82) |

The leading order behaviour of this (for constant 0pt0pt) is

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f(e)1∣β=−1)≳exp(−0pt2(43)n)P(f^{(e)}\_{1}\mid\beta=-1)\gtrsim\exp\left(-\frac{0pt}{2}\left(\frac{4}{3}\right)^{n}\right) |  | (83) |

###### Proof.

We set β=−1\beta=-1, so that we are now solving for a function where only one input 𝐯(𝟏)∈{0,1}n\mathbf{v^{(1)}}\in\{0,1\}^{n} has an output of 0.
To prove the upper bound, we simply require no clause to be True on input 𝐯(𝟏)\mathbf{v^{(1)}}.

To prove the lower bound, we can use the fact that the minimal representation of this function is given by a weight W(1)W^{(1)} with only non-zero entries in the main diagonal,

|  |  |  |  |
| --- | --- | --- | --- |
|  | W(1)ii={+1if v(1)i=0,−1if v(1)i=1.\displaystyle W^{(1)}\_{ii}=\begin{cases}+1&\text{if }v^{(1)}\_{i}=0,\\ -1&\text{if }v^{(1)}\_{i}=1.\end{cases} |  | (84) |

Note this is the opposite of the typical construction where we assign +1+1 if the input bit is 1 and −1-1 if the input bit is 0. Provided none of the 2n−12^{n}-1 clauses that make f(e)1(𝐯)f^{(e)}\_{1}(\mathbf{v}) output True are drawn ([Lemma˜D.4](#A4.Thmtheorem4 "Lemma D.4 (Probability of input 𝐯 being True). ‣ D.1 Utility lemmas ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")),
we can lower bound P(f(e)1)P(f^{(e)}\_{1}) by setting the first nn rows of W(1)W^{(1)} as described above (which would happen with probability 3−n23^{-n^{2}}) and require the rest of the rows to exclude any of the 2n−12^{n}-1 forbidden clauses.
∎

#### D.3.1 P(f)P(f) for a random function with fixed tt

In [Section˜D.3](#A4.SS3 "D.3 Function class: entropy ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"), we upper bounded P(f(e)t)P(f^{(e)}\_{t}) by computing the minimum number of forbidden clauses at every entropy class.
We know there can be a huge range in P(f(e)t)P(f^{(e)}\_{t}), the best example being t=2n−1t=2^{n-1}. Both 1-parity and nn-parity have 2n−12^{n-1} 1s, yet P(f(p)n)∼(3/2)−0ptn2n−1P(f^{(p)}\_{n})\sim(3/2)^{-0ptn2^{n-1}} and P(f(p)1)∼(3/2)−0pt2n−1P(f^{(p)}\_{1})\sim(3/2)^{-0pt2^{n-1}}. The bounds must satisfy P(f(p)1)P(f^{(p)}\_{1}) and are therefore far too loose for f(p)nf^{(p)}\_{n}.

We did not, however, determine how P(f(e)t)P(f^{(e)}\_{t}) should scale for the typical function with tt 1s (i.e. a uniformly sampled function from the set of functions with exactly t 1s1s).
Consider a function ff, and define the number of False outputs t~=2n−t\tilde{t}=2^{n}-t. The probability of drawing a clause CC which is True on an input 𝐯\mathbf{v} for which f(𝐯)f(\mathbf{v}) outputs False is p=(2n−1)/3np=(2^{n}-1)/3^{n} ([Lemma˜D.4](#A4.Thmtheorem4 "Lemma D.4 (Probability of input 𝐯 being True). ‣ D.1 Utility lemmas ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")). Then with RR as the set of all forbidden clauses and r=|R|r=|R|, the probability of drawing a forbidden clauses C∈RC\in R is

|  |  |  |  |
| --- | --- | --- | --- |
|  | Pr(C∈R)=r(n,t)3n=1−(1−p)2n−t,\Pr(C\in R)=\frac{r(n,t)}{3^{n}}=1-(1-p)^{2^{n}-t}, |  | (85) |

assuming independence.
[Figure˜9](#A4.F9 "In D.3.1 𝑃(𝑓) for a random function with fixed 𝑡 ‣ D.3 Function class: entropy ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") shows that this assumption is only valid for t~=O(n)\tilde{t}=O(n). As t~\tilde{t} increases, [Equation˜85](#A4.E85 "In D.3.1 𝑃(𝑓) for a random function with fixed 𝑡 ‣ D.3 Function class: entropy ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") overestimates P(C∈R)P(C\in R).
The empirical data was generated by uniformly sampling 20 functions at fixed tt and calculating rr by exhaustive enumeration for those functions. This gives us an idea of r(n,t)r(n,t) for the “typical” function with tt 1s. We also plotted the kk-parity functions for 1≤k≤n1\leq k\leq n (which all have t=2n−1t=2^{n-1}).

![Refer to caption](/html/2505.24060/assets/x6.png)


Figure 9: Fraction of accepted clauses 1−r(n,t)3n1-\tfrac{r(n,t)}{3^{n}} versus the number of zeros, t~\tilde{t}, in an nn-variable Boolean function.
Error bars show 1 standard deviation.
The theoretical line uses [Equation˜85](#A4.E85 "In D.3.1 𝑃(𝑓) for a random function with fixed 𝑡 ‣ D.3 Function class: entropy ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"), and assumes independence. This is only a good assumption for low t~\tilde{t}. As expected, when t~\tilde{t} is small (the function is almost entirely True), almost all clauses are accepted without changing the function. The red dots indicate all kk-parity functions for 1≤k≤n1\leq k\leq n.

### D.4 Function class: parity

Let f(p)k:{0,1}n→{0,1}f^{(p)}\_{k}\colon\{0,1\}^{n}\to\{0,1\} be the kk-parity function on the first kk bits:

|  |  |  |  |
| --- | --- | --- | --- |
|  | f(p)k(𝐯)=∑i=1kvi(mod2),1≤k≤n.f^{(p)}\_{k}(\mathbf{v})=\sum\_{i=1}^{k}v\_{i}\pmod{2},\qquad 1\leq k\leq n. |  | (86) |

As one checks directly, any representation of f(p)kf^{(p)}\_{k} in our random network model must use exactly j=2k−1j=2^{k-1}, distinct clauses of length kk, and no shorter set of clauses can realise parity. With M=αw2n−1M=\alpha\_{w}2^{n-1} and N=3nN=3^{n} we can construct lower and upper bounds with the following two propositions.

###### Proposition D.9 (Lower bound for kk-parity).

Using [Lemma˜D.2](#A4.Thmtheorem2 "Lemma D.2 (Lower bound on 𝑃(𝑓∣𝛽) using the minimum representation). ‣ D.1 Utility lemmas ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"), with the minimum representation size j×kj\times k, set j=2k−1j=2^{k-1}. Then,

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f(p)k)≥3−n2k−1(2k−1)!(2k−13k)αw2n−1−2k−1.P(f^{(p)}\_{k})\geq 3^{-n2^{k-1}}(2^{k-1})!\Bigl{(}\tfrac{2^{k-1}}{3^{k}}\Bigr{)}^{\alpha\_{w}2^{n-1}-2^{k-1}}. |  | (87) |

Applying Stirling’s inequality j!≥2πj(j/e)jj!\geq\sqrt{2\pi j}(j/e)^{j}, in the large nn limit we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | P(f(p)k)\displaystyle P(f^{(p)}\_{k}) | ≥3−n2k−12π2k−1(2k−1e)2k−1(2k−13k)αw2n−1−2k−1\displaystyle\geq 3^{-n2^{k-1}}\sqrt{2\pi 2^{k-1}}\left(\frac{2^{k-1}}{e}\right)^{2^{k-1}}\left(\frac{2^{k-1}}{3^{k}}\right)^{\alpha\_{w}2^{n-1}-2^{k-1}} |  | (88) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ≥exp(−αw2n−1(kln(3/2)+ln2)+O(n2k−1)).\displaystyle\geq\exp{\left(-\alpha\_{w}2^{n-1}(k\ln(3/2)+\ln 2)+O(n2^{{k-1}})\right)}. |  | (89) |

###### Proposition D.10 (Upper bound for kk-parity).

Exactly 2k−12^{k-1} of the 3k3^{k} clauses of length ≤k\leq k
implement kk-parity, so at each of the αw2n−1\alpha\_{w}2^{n-1} draws the chance of
choosing an admissible clause is at most
2k−13k\tfrac{2^{k-1}}{3^{k}}. Therefore

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f(p)k)≤(2k−13k)αw2n−1.P(f^{(p)}\_{k})\leq\Bigl{(}\tfrac{2^{k-1}}{3^{k}}\Bigr{)}^{\alpha\_{w}2^{n-1}}. |  | (90) |

At large nn,

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f(p)k)≤exp(−αw2n−1(kln(3/2)+ln2))P(f^{(p)}\_{k})\leq\exp{\left(-\alpha\_{w}2^{n-1}(k\ln(3/2)+\ln 2)\right)} |  | (91) |

The bounds above match up to constants, so

|  |  |  |  |
| --- | --- | --- | --- |
|  | P(f(p)k)=e−Θ(0ptk2n−1).P\bigl{(}f^{(p)}\_{k}\bigr{)}=e^{-\Theta(0ptk2^{n-1})}. |  | (92) |

### D.5 Function class: sparse

A Boolean function f:{0,1}n→{0,1}f\colon\{0,1\}^{n}\to\{0,1\} is called kk-sparse if it depends on exactly kk of its nn input bits (say x1,…,xkx\_{1},\dots,x\_{k}) and is independent of the remaining n−kn-k bits. Equivalently, for every fixed (x1,…,xk)(x\_{1},\dots,x\_{k}), flipping any of the last n−kn-k coordinates does not change ff. In the ordered listing of the truth table ([Figure˜2](#S3.F2 "In 3.3 Understanding 𝑃(𝑓) v.s. 𝐾(𝑓) ‣ 3 Untrained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")), ff then repeats its 2k2^{k}-bit pattern 2n−k2^{n-k} times.

It is hard to come up with bounds for kk-sparse functions that are meaningful, as the complexity range for a given kk can be very large. The most complex kk-sparse in each class is kk-parity. At the other end of the spectrum, there are functions where the minimum representation ([Equation˜48](#A4.E48 "In D.1 Utility lemmas ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")) has A=IkA=I\_{k} (the identity matrix with dimension kk). We could lower bound this type of function by requiring IkI\_{k} to exist (with probability 3−nk3^{-nk}), and that the first kk elements of the rest of the clauses must not contain exclusively −1-1 and 0 (but we permit all 0s). The probability that this happens is (23)k−3−n\left(\tfrac{2}{3}\right)^{k}-3^{-n}. We can multiply this by the total number of clause options, N=3nN=3^{n}, to give us r=3n(23)k−1r=3^{n}\left(\tfrac{2}{3}\right)^{k}-1 and then use [Lemma˜D.3](#A4.Thmtheorem3 "Lemma D.3. ‣ D.1 Utility lemmas ‣ Appendix D Results relating 𝑃(𝑓) to 𝐾(𝑓) ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") to get,

|  |  |  |  |
| --- | --- | --- | --- |
|  | 3−nk(1−(23)k+3−n)αw2n−1−k≤P(f(s)k)≤(1−(23)k+3−n)αw2n−1,\displaystyle 3^{-nk}\left(1-\left(\tfrac{2}{3}\right)^{k}+3^{-n}\right)^{\alpha\_{w}2^{n-1}-k}\leq P(f^{(s)}\_{k})\leq\left(1-\left(\tfrac{2}{3}\right)^{k}+3^{-n}\right)^{\alpha\_{w}2^{n-1}}, |  | (93) |

which to leading order scales as exp(−αw2n−1(23)k)\exp{\left(-\alpha\_{w}2^{n-1}\left(\tfrac{2}{3}\right)^{k}\right)}, decaying slower than kk-parity for large kk. (The upper bound comes from rejecting all forbidden clauses.)

## Appendix E Trained networks

### E.1 Generating datasets

Each function is a Boolean map f:{0,1}n→{0,1}f:\{0,1\}^{n}\to\{0,1\}, stored as an nn-dimensional binary input vector and a scalar output. To ensure reproducibility, we set a fixed random seed at the start of generation. Given a training set size mm, we randomly shuffle all 2n2^{n} possible inputs and take the first mm as training examples; the remaining 2n−m2^{n}-m points form the test set.

We train DFCNs on the following three functions

1. 1.

   kk-parity: Choose a random subset S⊆{1,…,n}S\subseteq\{1,\dots,n\} of size kk, and define:
   f(x)=⨁i∈Sxi.f(x)=\bigoplus\_{i\in S}x\_{i}.
2. 2.

   tt-entropy: Select tt input points uniformly at random from the 2n2^{n} possibilities and assign f(x)=1f(x)=1 on those points (all others map to 0), yielding functions of fixed Hamming weight tt.
3. 3.

   kk-sparse: Generate a random binary string s∈{0,1}2ks\in\{0,1\}^{2^{k}}, then tile it 2n−k2^{n-k} times to form the full function string of length 2n2^{n}.

Note that we do not study kk-sparse functions in the main text. These are functions generated by repeated patterns of length 2k2^{k}.

In our experiments, we fix n=7n=7. The parameter grids are:

* •

  kk-parity: k∈{1,2,…,7}k\in\{1,2,\dots,7\}.
* •

  tt-entropy: t∈{0,4,8,16,32,35,64}t\in\{0,4,8,16,32,35,64\}.
* •

  kk-sparse tile lengths l∈{2,4,5,8,13,16,32}l\in\{2,4,5,8,13,16,32\}.

Note that the repeating patterns with l=5,13l=5,13 do not generate kk-sparse functions (unlike the other lengths ll given above). We include them as they have low LZ complexity but not low DNF complexity (see [Section˜B.3](#A2.SS3 "B.3 Important differences between 𝐾(𝑓) and 𝐾_{𝐿𝑍}(𝑓) ‣ Appendix B Notes on complexity measures ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")), and this example is useful in demonstrating why K(f)K{(f)} is a better measure of complexity than KLZ(f)K\_{LZ}{(f)}.

### E.2 Metropolis-Hastings algorithm

While we could in theory use an SGD-like algorithm ([Algorithm˜3](#alg3 "In E.4 SGD-like algorithm ‣ Appendix E Trained networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")), which aims to find the direction of steepest descent and move there, it is not Bayesian, and enumerating the entire neighbourhood rapidly increases exponentially in computational complexity as nn increases. In this section, we define a Metropolis-Hastings algorithm that is Bayesian, and does not suffer from these scaling problems.

Let θ=(W(1),W(2))\theta=(W^{(1)},W^{(2)}) denote the parameter vector of weights in a DFCN, with

|  |  |  |
| --- | --- | --- |
|  | W(1)∈{−1,0,1}n×αw2n−1,W(2)∈{0,1}αw2n−1.W^{(1)}\in\{-1,0,1\}^{n\times\alpha\_{w}2^{n-1}},\quad W^{(2)}\in\{0,1\}^{\alpha\_{w}2^{n-1}}. |  |

We write f(θ;S)f(\theta;S) for the network’s predictions on a dataset SS, and L(f(θ;S))L\bigl{(}f(\theta;S)\bigr{)} for its empirical loss (e.g. classification error) on SS. We also use the ℓ1\ell\_{1}-norm regulariser

|  |  |  |
| --- | --- | --- |
|  | ∥θ∥1=∥W(1)∥1+∥W(2)∥1.\|\theta\|\_{1}\;=\;\|W^{(1)}\|\_{1}+\|W^{(2)}\|\_{1}. |  |

##### Target (posterior) density.

Given inverse-temperature κ>0\kappa>0 and weight-decay factor λ≥0\lambda\geq 0, we seek to sample from

|  |  |  |
| --- | --- | --- |
|  | π(θ)∝exp[−κL(f(θ;S))−λ∥θ∥1].\pi(\theta)\;\propto\;\exp\Bigl{[}-\,\kappa L\bigl{(}f(\theta;S)\bigr{)}-\lambda\,\|\theta\|\_{1}\Bigr{]}. |  |

##### Proposal distribution.

For any current state θ\theta, its 1-hop neighbourhood is

|  |  |  |
| --- | --- | --- |
|  | 𝒩(θ)={θ′:d(θ,θ′)=1},\mathcal{N}(\theta)\;=\;\{\theta^{\prime}\colon d(\theta,\theta^{\prime})=1\}, |  |

where d(θ,θ′)d(\theta,\theta^{\prime}) is the Hamming distance between discrete parameters. We use the uniform proposal

|  |  |  |
| --- | --- | --- |
|  | g(θ→θ′)={1|𝒩(θ)|,if θ′∈𝒩(θ),0,otherwise.g(\theta\to\theta^{\prime})\;=\;\begin{cases}\displaystyle\frac{1}{|\mathcal{N}(\theta)|},&\text{if }\theta^{\prime}\in\mathcal{N}(\theta),\\ 0,&\text{otherwise.}\end{cases} |  |

[Algorithm˜1](#alg1 "In Proposal distribution. ‣ E.2 Metropolis-Hastings algorithm ‣ Appendix E Trained networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") writes down the process explicitly.

This algorithm trains well with κ=1000\kappa=1000 and λ=0\lambda=0 – increasing λ\lambda to 0.1 had limited effect except to destabilise early training. [Figure˜10](#A5.F10 "In Proposal distribution. ‣ E.2 Metropolis-Hastings algorithm ‣ Appendix E Trained networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") shows the outcomes of training a DFCN with [Algorithm˜1](#alg1 "In Proposal distribution. ‣ E.2 Metropolis-Hastings algorithm ‣ Appendix E Trained networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") on different function classes. For entropy and repeated functions, we see that adding weight decay greatly improves performance, especially when the size of the training set mm is smaller. As discussed in [Section˜4.3](#S4.SS3 "4.3 The special case of parity (Figure˜4) ‣ 4 Trained neural networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"), weight decay does not provide any significant advantages when trying to learn highly complex functions such as 7-parity.

Algorithm 1  Metropolis–Hastings optimisation for DFCNs

Initialise:W(1)∼{−1,0,1}n×αw2n−1W^{(1)}\sim\{-1,0,1\}^{n\times\alpha\_{w}2^{n-1}}, β∼{−1,1}\beta\sim\{-1,1\}, W(2)∼{0,β}αw2n−1W^{(2)}\sim\{0,\beta\}^{\alpha\_{w}2^{n-1}}b(1)←b(1)(W(1))b^{(1)}\leftarrow b^{(1)}(W^{(1)}), b(2)←b(2)(β)b^{(2)}\leftarrow b^{(2)}(\beta)Biases are functions of weights, see [Definition˜2.6](#S2.Thmtheorem6 "Definition 2.6 (DFCN). ‣ 2.2 DFCN–DNF correspondence ‣ 2 Preliminaries and intuition ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")Input:\StateTraining set SS, batch size b=|S|b=|S|, iterations NNfull batch if Bayesian
Hyperparams:κ>0,λ≥0\kappa>0,\ \lambda\geq 0Initialise:\Forθ(0)\theta^{(0)}uniformly in parameter space
t=1,…,Nt=1,\dots,NSample minibatch St⊂SS\_{t}\subset S, |St|=b|S\_{t}|=bPropose θ′∼g(θ(t−1)→⋅)\theta^{\prime}\sim g(\theta^{(t-1)}\to\cdot)Compute losses
Lold=L(f(θ(t−1);St)),Lnew=L(f(θ′;St))L\_{\rm old}=L\bigl{(}f(\theta^{(t-1)};S\_{t})\bigr{)},\ L\_{\rm new}=L\bigl{(}f(\theta^{\prime};S\_{t})\bigr{)}Compute acceptance probability

|  |  |  |
| --- | --- | --- |
|  | α=min{1,exp[κ(Lold−Lnew)+λ(∥θ(t−1)∥1−∥θ′∥1)]}\alpha\;=\;\min\Bigl{\{}1,\;\exp\bigl{[}\kappa\,(L\_{\rm old}-L\_{\rm new})\;+\;\lambda\,(\|\theta^{(t-1)}\|\_{1}-\|\theta^{\prime}\|\_{1})\bigr{]}\Bigr{\}} |  |

Draw u∼Uniform(0,1)u\sim\mathrm{Uniform}(0,1)u<αu<\alphaθ(t)←θ′\theta^{(t)}\leftarrow\theta^{\prime}θ(t)←θ(t−1)\theta^{(t)}\leftarrow\theta^{(t-1)}

\State

\State

\State

\Comment

\State

\Comment

\State

\State

\State

\State

\State

\State

\If

\State

\Else

\State

\EndIf

\EndFor

![Refer to caption](/html/2505.24060/assets/x7.png)


Figure 10: MCMC algorithm ([Algorithm˜1](#alg1 "In Proposal distribution. ‣ E.2 Metropolis-Hastings algorithm ‣ Appendix E Trained networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") with κ=1000\kappa=1000) trained on different targets from the n=7n=7 dataset. Each column shows a different function class – parity, entropy and repeat. See [Section˜E.1](#A5.SS1 "E.1 Generating datasets ‣ Appendix E Trained networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data") for full experimental details and a description of each function type.
As with the SGD-like algorithm shown in LABEL:fig:app:discrete\_trained\_plot, weight decay (bottom row, λ=0.01\lambda=0.01) outperforms no weight decay (top row λ=0\lambda=0), especially for small training set size mm.

### E.3 Min norm Oracle algorithm

We also define an Oracle algorithm, which computes the minimal complexity DNF compatible with the training set. This is obtained by exhaustive search and is only possible for small enough nn. Since this always returns the minimum norm DFCN solution, this acts as a good baseline to compare other algorithms to, as we can see how close our trained function is to having the optimal minimal complexity.

Algorithm 2  Min norm Oracle

Initialise:\Comment\Comment\StateTrue inputs P←P\leftarrow{ } Inputs for which f(𝐯)=1f(\mathbf{v})=1DNF Minimum DNF
Input:Training data SS, test data TT𝐬∈S\mathbf{s}\in Sf(𝐬)=1f(\mathbf{s})=1P←P∪{𝐬}P\leftarrow P\,\cup\,\{\mathbf{s}\}DNF ←\leftarrowSOPform(variables=[x1,…,xnx\_{1},\dots,x\_{n}], minterms=PP, dontcares=TT)

\State

\State

\State

\For

\If

\State

\EndIf

\EndFor

\State

The training algorithm is shown in [Algorithm˜2](#alg2 "In E.3 Min norm Oracle algorithm ‣ Appendix E Trained networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"). The SOPform function comes from the sympy logic module (Meurer et al., [2017](#bib.bib43)) and finds the minimal DNF expression for a given set of inputs that output True. It takes the following arguments:

* •

  variables: A list of symbols denoting the literals in the DNF.
* •

  minterms: All inputs for which the output of the expression should give True.
* •

  dontcares: All inputs for which we don’t care about the output (i.e. the test data).

See LABEL:fig:app:oracle for data.

### E.4 SGD-like algorithm

Despite not being Bayesian, we also trained with an SGD-like algorithm, which worked well for small nn in which the algorithm is tractable. In [Algorithm˜3](#alg3 "In E.4 SGD-like algorithm ‣ Appendix E Trained networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"), we begin by randomly initializing the first-layer weights W(1)∈{−1,0,1}n×αw2n−1W^{(1)}\in\{-1,0,1\}^{n\times\alpha\_{w}2^{n-1}} and second-layer weights W(2)∈{0,1}αw2n−1W^{(2)}\in\{0,1\}^{\alpha\_{w}2^{n-1}}, and computing the dependent biases via b(1)=b(1)(W(1))b^{(1)}=b^{(1)}(W^{(1)}) and b(2)=b(2)(β)b^{(2)}=b^{(2)}(\beta) (see [Definition˜2.6](#S2.Thmtheorem6 "Definition 2.6 (DFCN). ‣ 2.2 DFCN–DNF correspondence ‣ 2 Preliminaries and intuition ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data")). Over NN iterations, we draw a minibatch StS\_{t} of size bb from the training set, evaluate the current network accuracy on StS\_{t}, and enumerate all one-hop neighbours of (W(1),W(2))(W^{(1)},W^{(2)}). For each neighbour, we recompute its biases and measure its batch accuracy, then collect the subset 𝒩best\mathcal{N}\_{\mathrm{best}} of weights achieving the highest performance. With probability pp we choose the neighbour from 𝒩best\mathcal{N}\_{\mathrm{best}}, which minimises the ℓ1\ell\_{1} norm ∥W(1)∥1+∥W(2)∥1\|W^{(1)}\|\_{1}+\|W^{(2)}\|\_{1} (thus encouraging sparsity), and otherwise select uniformly at random from 𝒩best\mathcal{N}\_{\mathrm{best}}. The chosen weights replace (W(1),W(2))(W^{(1)},W^{(2)}), their biases are updated, and the procedure repeats. Upon completion, the algorithm returns a two-layer DFCN that is locally optimised for the training data.

LABEL:fig:app:discrete\_trained\_plot shows this algorithm trained on a DFCN with n=7n=7 over a wide range of functions. We see that the performance is very similar to [Algorithm˜1](#alg1 "In Proposal distribution. ‣ E.2 Metropolis-Hastings algorithm ‣ Appendix E Trained networks ‣ Characterising the Inductive Biases of Neural Networks on Boolean Data"). However, since this SGD-like algorithm does not scale well, we would instead opt to use the MCMC algorithm for large nn.

Algorithm 3  SGD-like optimisation for DFCNs

Initialise:W(1)∼{−1,0,1}n×αw2n−1W^{(1)}\sim\{-1,0,1\}^{n\times\alpha\_{w}2^{n-1}}, β∼{−1,1}\beta\sim\{-1,1\}, W(2)∼{0,β}αw2n−1W^{(2)}\sim\{0,\beta\}^{\alpha\_{w}2^{n-1}}b(1)←b(1)(W(1))b^{(1)}\leftarrow b^{(1)}(W^{(1)})

\State\State\State

Conversion to HTML had a Fatal error and exited abruptly. This document may be truncated or damaged.

[◄](/html/2505.24059)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2505.24060)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2505.24060)
[View original  
on arXiv](https://arxiv.org/abs/2505.24060)[►](/html/2505.24061)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Thu Jun 5 18:13:27 2025 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
