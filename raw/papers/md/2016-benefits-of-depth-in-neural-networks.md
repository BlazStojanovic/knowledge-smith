---
arxiv: '1602.04485'
authors:
- Matus Telgarsky
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: Benefits of depth in neural networks
url: http://arxiv.org/abs/1602.04485v2
year: 2016
---

# Benefits of depth in neural networks

###### Abstract

For any positive integer k𝑘k,
there exist neural networks
with Θ​(k3)Θsuperscript𝑘3\Theta(k^{3}) layers,
Θ​(1)Θ1\Theta(1) nodes per layer,
and Θ​(1)Θ1\Theta(1) distinct parameters
which can not be approximated by networks with 𝒪​(k)𝒪𝑘\mathcal{O}(k) layers
unless they are exponentially large — they must possess Ω​(2k)Ωsuperscript2𝑘\Omega(2^{k}) nodes.
This result is proved here for a class of nodes termed *semi-algebraic gates* which includes
the common choices of ReLU, maximum, indicator, and piecewise polynomial functions, therefore establishing
benefits of depth against not just standard networks with ReLU gates, but also convolutional networks
with ReLU and maximization gates,
sum-product networks,
and boosted decision trees
(in this last case with a stronger separation: Ω​(2k3)Ωsuperscript2superscript𝑘3\Omega(2^{k^{3}}) total tree nodes are required).

###### keywords:

Neural networks, representation, approximation, depth hierarchy.

## 1 Setting and main results

A neural network is a model of real-valued computation defined by a connected directed graph as follows.
Nodes await real numbers on their incoming edges,
thereafter computing a function of these reals and transmitting it along their outgoing edges.
Root nodes apply their computation to a vector provided as input to the network,
whereas internal nodes apply their computation to the output of other nodes.
Different nodes may compute different functions, two common choices being
the maximization gate v↦maxi⁡vimaps-to𝑣subscript𝑖subscript𝑣𝑖v\mapsto\max\_{i}v\_{i} (where v𝑣v is the vector of values on incoming edges),
and the *standard ReLU gate* v↦σr​(⟨a,v⟩+b)maps-to𝑣subscript𝜎r

𝑎𝑣
𝑏v\mapsto\sigma\_{\textsc{r}}(\left\langle a,v\right\rangle+b)
where σr​(z):=max⁡{0,z}assignsubscript𝜎r𝑧0𝑧\sigma\_{\textsc{r}}(z):=\max\{0,z\} is called the ReLU
(rectified linear unit), and the parameters a𝑎a and b𝑏b may vary from node to node.
Graphs in the present work are acyclic,
and there is exactly one node with no outgoing edges
whose computation is the output of the network.

Neural networks distinguish themselves from many other function classes used in machine learning by
possessing multiple *layers*, meaning the output is the result of composing together an arbitrary
number of (potentially complicated) nonlinear operations;
by contrast, the functions computed by boosted decision stumps and SVMs can be written
as neural networks with a constant number of layers.

The purpose of the present work is to show that standard types of networks always gain in representation
power with the addition of layers.
Concretely: it is shown that for every positive integer k𝑘k,
there exist neural networks
with Θ​(k3)Θsuperscript𝑘3\Theta(k^{3}) layers,
Θ​(1)Θ1\Theta(1) nodes per layer,
and Θ​(1)Θ1\Theta(1) distinct parameters
which can not be approximated by networks with 𝒪​(k)𝒪𝑘\mathcal{O}(k) layers
and o​(2k)𝑜superscript2𝑘o(2^{k}) nodes.

### 1.1 Main result

Before stating the main result, a few choices and pieces of notation deserve explanation.
First, the target many-layered function uses standard ReLU gates;
this is by no means necessary, and a more general statement can be found in [Theorem 3.19](#S3.Thmtheorem19 "Theorem 3.19. ‣ 3.4 Proof of Theorem 1.1 ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks").
Secondly, the notion of approximation is the L1superscript𝐿1L^{1} distance: given two functions f𝑓f and g𝑔g,
their pointwise disagreement |f​(x)−g​(x)|𝑓𝑥𝑔𝑥|f(x)-g(x)| is averaged over the cube [0,1]dsuperscript01𝑑[0,1]^{d}.
Here as well, the same proofs allow flexibility (cf. [Theorem 3.19](#S3.Thmtheorem19 "Theorem 3.19. ‣ 3.4 Proof of Theorem 1.1 ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks")).
Lastly, the shallower networks used for approximation use *semi-algebraic gates*,
which generalize the earlier maximization and standard ReLU gates,
and allow for analysis of not just standard networks with ReLU gates,
but convolutional networks with ReLU and maximization gates (Krizhevsky et al., [2012](#bib.bib12)),
sum-product networks (where nodes compute polynomials) (Poon and Domingos, [2011](#bib.bib16)),
and boosted decision trees;
the full definition of semi-algebraic gates appears in [Section 2](#S2 "2 Semi-algebraic gates and assorted network notation ‣ Benefits of depth in neural networks").

###### Theorem 1.1.

Let any integer k≥1𝑘1k\geq 1 and any dimension d≥1𝑑1d\geq 1 be given.
There exists f:ℝd→ℝ:𝑓→superscriptℝ𝑑ℝf:\mathbb{R}^{d}\to\mathbb{R} computed by a neural network with standard ReLU gates
in 2​k3+82superscript𝑘382k^{3}+8 layers, 3​k3+123superscript𝑘3123k^{3}+12 total nodes, and 4+d4𝑑4+d distinct parameters
so that

|  |  |  |
| --- | --- | --- |
|  | infg∈𝒞∫[0,1]d|f​(x)−g​(x)|​𝑑x≥164,subscriptinfimum𝑔𝒞subscriptsuperscript01𝑑𝑓𝑥𝑔𝑥differential-d𝑥164\inf\_{g\in\mathcal{C}}\int\_{[0,1]^{d}}|f(x)-g(x)|dx\geq\frac{1}{64}, |  |

where 𝒞𝒞\mathcal{C} is the union of the following two sets of functions.

* •

  Functions computed by networks of (t,α,β)𝑡𝛼𝛽(t,\alpha,\beta)-semi-algebraic gates
  in ≤kabsent𝑘\leq k layers and ≤2k/(t​α​β)absentsuperscript2𝑘𝑡𝛼𝛽\leq 2^{k}/(t\alpha\beta) nodes.
  (E.g., as with standard ReLU networks
  or with convolutional neural networks with standard ReLU and maximization gates; cf. [Section 2](#S2 "2 Semi-algebraic gates and assorted network notation ‣ Benefits of depth in neural networks").)
* •

  Functions
  computed by linear combinations of ≤tabsent𝑡\leq t decision trees
  each with ≤2k3/tabsentsuperscript2superscript𝑘3𝑡\leq 2^{k^{3}}/t nodes.
  (E.g., the function class used by boosted decision trees;
  cf. [Section 2](#S2 "2 Semi-algebraic gates and assorted network notation ‣ Benefits of depth in neural networks").)

Analogs to [Theorem 1.1](#S1.Thmtheorem1 "Theorem 1.1. ‣ 1.1 Main result ‣ 1 Setting and main results ‣ Benefits of depth in neural networks") for boolean circuits — which have boolean inputs
routed through {and,or,not}andornot\{\textup{and},\textup{or},\textup{not}\} gates — have been studied extensively
by the circuit complexity community, where they are called *depth hierarchy theorems*.
The seminal result, due to Håstad ([1986](#bib.bib8)), establishes the inapproximability
of the parity function by shallow circuits (unless their size is exponential).
Standard neural networks appear to have received less study;
closest to the present work is an investigation by
Eldan and Shamir ([2015](#bib.bib6)) analyzing the case k=2𝑘2k=2 when the dimension d𝑑d is large,
showing an exponential separation between 2- and 3-layer networks, a regime not handled by [Theorem 1.1](#S1.Thmtheorem1 "Theorem 1.1. ‣ 1.1 Main result ‣ 1 Setting and main results ‣ Benefits of depth in neural networks").
Further bibliographic notes and open problems may be found in [Section 5](#S5 "5 Bibliographic notes and open problems ‣ Benefits of depth in neural networks").

The proof of [Theorem 1.1](#S1.Thmtheorem1 "Theorem 1.1. ‣ 1.1 Main result ‣ 1 Setting and main results ‣ Benefits of depth in neural networks") (and of the more general [Theorem 3.19](#S3.Thmtheorem19 "Theorem 3.19. ‣ 3.4 Proof of Theorem 1.1 ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks")) occupies [Section 3](#S3 "3 Benefits of depth ‣ Benefits of depth in neural networks").
The key idea is that just a few function compositions (layers) suffice to construct a highly oscillatory function,
whereas function addition (adding nodes but keeping depth fixed) gives a function with few oscillations.
Thereafter, an elementary counting argument suffices to show that low-oscillation functions can not approximate
high-oscillation functions.

### 1.2 Companion results

[Theorem 1.1](#S1.Thmtheorem1 "Theorem 1.1. ‣ 1.1 Main result ‣ 1 Setting and main results ‣ Benefits of depth in neural networks") only provides the existence of *one*
network (for each k𝑘k) which can not be approximated by
a network with many fewer layers. It is natural to wonder if there are *many* such
special functions. The following bound indicates their population is in fact quite modest.

Specifically, the construction behind [Theorem 1.1](#S1.Thmtheorem1 "Theorem 1.1. ‣ 1.1 Main result ‣ 1 Setting and main results ‣ Benefits of depth in neural networks"), as elaborated in [Theorem 3.19](#S3.Thmtheorem19 "Theorem 3.19. ‣ 3.4 Proof of Theorem 1.1 ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks"),
can be seen as exhibiting 𝒪​(2k3)𝒪superscript2superscript𝑘3\mathcal{O}(2^{k^{3}}) points, and a fixed labeling of these points, upon which a shallow network hardly improves
upon random guessing. The forthcoming [Theorem 1.2](#S1.Thmtheorem2 "Theorem 1.2. ‣ 1.2 Companion results ‣ 1 Setting and main results ‣ Benefits of depth in neural networks") similarly shows that even on the more simpler task of fitting 𝒪​(k9)𝒪superscript𝑘9\mathcal{O}(k^{9}) points,
the earlier class of networks is useless on most random labellings.

In order to state the result, a few more definitions are in order.
Firstly, for this result, the notion of neural network is more restrictive.
Let a *neural net graph 𝔊𝔊\mathfrak{G}* denote not only the graph structure (nodes and edges),
but also an assignment of gate functions to nodes, of edges to the inputs of gates,
and an assignment of free parameters w∈ℝp𝑤superscriptℝ𝑝w\in\mathbb{R}^{p} to the parameters of the gates.
Let 𝒩​(𝔊)𝒩𝔊\mathcal{N}(\mathfrak{G}) denote the class of functions obtained by varying the free parameters;
this definition is fairly standard, and is discussed in more detail in [Section 2](#S2 "2 Semi-algebraic gates and assorted network notation ‣ Benefits of depth in neural networks").
As a final piece of notation, given a function f:ℝd→ℝ:𝑓→superscriptℝ𝑑ℝf:\mathbb{R}^{d}\to\mathbb{R}, let f~:ℝd→{0,1}:~𝑓→superscriptℝ𝑑01\tilde{f}:\mathbb{R}^{d}\to\{0,1\} denote
the corresponding classifier f~​(x):=𝟏​[f​(x)≥1/2]assign~𝑓𝑥1delimited-[]𝑓𝑥12\tilde{f}(x):=\mathbf{1}[f(x)\geq 1/2].

###### Theorem 1.2.

Let any neural net graph 𝔊𝔊\mathfrak{G} be given with ≤pabsent𝑝\leq p parameters in ≤labsent𝑙\leq l layers and ≤mabsent𝑚\leq m total
(t,α,β)𝑡𝛼𝛽(t,\alpha,\beta)-semi-algebraic nodes.
Then for any δ>0𝛿0\delta>0
and any
n≥8​p​l2​ln⁡(8​e​m​t​α​β​p​(l+1))+4​ln⁡(1/δ)𝑛8𝑝superscript𝑙28𝑒𝑚𝑡𝛼𝛽𝑝𝑙141𝛿n\geq 8pl^{2}\ln(8emt\alpha\beta p(l+1))+4\ln(1/\delta) points (xi)i=1nsuperscriptsubscriptsubscript𝑥𝑖𝑖1𝑛(x\_{i})\_{i=1}^{n},
with probability ≥1−δabsent1𝛿\geq 1-\delta over uniform random labels (yi)i=1nsuperscriptsubscriptsubscript𝑦𝑖𝑖1𝑛(y\_{i})\_{i=1}^{n},

|  |  |  |
| --- | --- | --- |
|  | inff∈𝒩​(𝔊)1n​∑i=1n𝟏​[f~​(xi)≠yi]≥14.subscriptinfimum𝑓𝒩𝔊1𝑛superscriptsubscript𝑖1𝑛1delimited-[]~𝑓subscript𝑥𝑖subscript𝑦𝑖14\inf\_{f\in\mathcal{N}(\mathfrak{G})}\frac{1}{n}\sum\_{i=1}^{n}\mathbf{1}[\tilde{f}(x\_{i})\neq y\_{i}]\geq\frac{1}{4}. |  |

This proof is a direct corollary of the VC dimension of semi-algebraic networks,
which in turn can be proved by a small modification of the VC dimension proof for
piecewise polynomial networks (Anthony and Bartlett, [1999](#bib.bib1), Theorem 8.8).
Moreover, the core methodology for VC
dimension bounds of neural networks is due to [Warren](#bib.bib23),
whose goal was an analog of [Theorem 1.2](#S1.Thmtheorem2 "Theorem 1.2. ‣ 1.2 Companion results ‣ 1 Setting and main results ‣ Benefits of depth in neural networks") for polynomials (Warren, [1968](#bib.bib23), Theorem 7).

###### Lemma 1.3 (Simplification of [Lemma 4.2](#S4.Thmtheorem2 "Lemma 4.2. ‣ 4 Limitations of depth ‣ Benefits of depth in neural networks")).

Let any neural net graph 𝔊𝔊\mathfrak{G} be given with ≤pabsent𝑝\leq p parameters in ≤labsent𝑙\leq l layers and ≤mabsent𝑚\leq m total nodes,
each of which is (t,α,β)𝑡𝛼𝛽(t,\alpha,\beta)-semi-algebraic.
Then

|  |  |  |
| --- | --- | --- |
|  | VC​(𝒩​(𝔊))≤6​p​(l+1)​(ln⁡(2​p​(l+1))+ln⁡(8​e​m​t​α)+l​ln⁡(β)).VC𝒩𝔊6𝑝𝑙12𝑝𝑙18𝑒𝑚𝑡𝛼𝑙𝛽\textup{VC}(\mathcal{N}(\mathfrak{G}))\leq 6p(l+1)\big{(}\ln(2p(l+1))+\ln(8emt\alpha)+l\ln(\beta)\big{)}. |  |

The proof of [Theorem 1.2](#S1.Thmtheorem2 "Theorem 1.2. ‣ 1.2 Companion results ‣ 1 Setting and main results ‣ Benefits of depth in neural networks") and [Lemma 1.3](#S1.Thmtheorem3 "Lemma 1.3 (Simplification of Lemma 4.2). ‣ 1.2 Companion results ‣ 1 Setting and main results ‣ Benefits of depth in neural networks") may be found in [Section 4](#S4 "4 Limitations of depth ‣ Benefits of depth in neural networks").
The argument for the VC dimension is very close to the argument for [Theorem 1.1](#S1.Thmtheorem1 "Theorem 1.1. ‣ 1.1 Main result ‣ 1 Setting and main results ‣ Benefits of depth in neural networks")
that a network with few layers has few oscillations; see [Section 4](#S4 "4 Limitations of depth ‣ Benefits of depth in neural networks") for further discussion
of this relationship.

## 2 Semi-algebraic gates and assorted network notation

The definition of a semi-algebraic gate is unfortunately complicated;
it is designed to capture a few standard nodes in a single abstraction
without degrading the bounds.
Note that the name *semi-algebraic set* is standard (Bochnak et al., [1998](#bib.bib3), Definition 2.1.4),
and refers to a set defined by unions and intersections of polynomial inequalities
(and thus the name is somewhat abused here).

###### Definition 2.1.

A function f:ℝk→ℝ:𝑓→superscriptℝ𝑘ℝf:\mathbb{R}^{k}\to\mathbb{R} is *(t,α,β)𝑡𝛼𝛽(t,\alpha,\beta)-sa ((t,α,β)𝑡𝛼𝛽(t,\alpha,\beta)-semi-algebraic)*
if there exist t𝑡t polynomials (qi)i=1tsuperscriptsubscriptsubscript𝑞𝑖𝑖1𝑡(q\_{i})\_{i=1}^{t} of degree ≤αabsent𝛼\leq\alpha,
and m𝑚m triples (Uj,Lj,pj)j=1msuperscriptsubscriptsubscript𝑈𝑗subscript𝐿𝑗subscript𝑝𝑗𝑗1𝑚(U\_{j},L\_{j},p\_{j})\_{j=1}^{m} where Ujsubscript𝑈𝑗U\_{j} and Ljsubscript𝐿𝑗L\_{j} are subsets
of [t]delimited-[]𝑡[t] (where [t]:={1,…,t}assigndelimited-[]𝑡1…𝑡[t]:=\{1,\ldots,t\}) and pjsubscript𝑝𝑗p\_{j} is a polynomial of degree ≤βabsent𝛽\leq\beta, such that

|  |  |  |
| --- | --- | --- |
|  | f​(v)=∑j=1mpj​(v)​(∏i∈Lj𝟏​[qi​(v)<0])​(∏i∈Uj𝟏​[qi​(v)≥0]).𝑓𝑣superscriptsubscript𝑗1𝑚subscript𝑝𝑗𝑣subscriptproduct𝑖subscript𝐿𝑗1delimited-[]subscript𝑞𝑖𝑣0subscriptproduct𝑖subscript𝑈𝑗1delimited-[]subscript𝑞𝑖𝑣0f(v)=\sum\_{j=1}^{m}p\_{j}(v)\left(\prod\_{i\in L\_{j}}\mathbf{1}[q\_{i}(v)<0]\right)\left(\prod\_{i\in U\_{j}}\mathbf{1}[q\_{i}(v)\geq 0]\right). |  |

A notable trait of the definition is that the number of terms m𝑚m does not need to enter
the name as it does not affect any of the complexity estimates herein (e.g., [Theorem 1.1](#S1.Thmtheorem1 "Theorem 1.1. ‣ 1.1 Main result ‣ 1 Setting and main results ‣ Benefits of depth in neural networks") or [Theorem 1.2](#S1.Thmtheorem2 "Theorem 1.2. ‣ 1.2 Companion results ‣ 1 Setting and main results ‣ Benefits of depth in neural networks")).

Distinguished special cases of semi-algebraic gates are as follows in [Lemma 2.3](#S2.Thmtheorem3 "Lemma 2.3 (Example semi-algebraic gates). ‣ 2 Semi-algebraic gates and assorted network notation ‣ Benefits of depth in neural networks").
The standard piecewise polynomial gates generalize the ReLU and have received a fair bit of attention
in the theoretical community (Anthony and Bartlett, [1999](#bib.bib1), Chapter 8);
here a function σ:ℝ→ℝ:𝜎→ℝℝ\sigma:\mathbb{R}\to\mathbb{R} is *(t,α)𝑡𝛼(t,\alpha)-poly*
if ℝℝ\mathbb{R} can be partitioned into ≤tabsent𝑡\leq t intervals
so that σ𝜎\sigma is a polynomial of degree ≤αabsent𝛼\leq\alpha within each piece.
The maximization and minimization gates have become popular due to their use in convolutional
networks (Krizhevsky et al., [2012](#bib.bib12)), which will be discussed more in [Section 2.1](#S2.SS1 "2.1 Notation for neural networks ‣ 2 Semi-algebraic gates and assorted network notation ‣ Benefits of depth in neural networks").
Lastly, decision trees and boosted decision trees are practically successful classes usually viewed as competitors
to neural networks (Caruana and Niculescu-Mizil, [2006](#bib.bib4)), and have the following structure.

###### Definition 2.2.

A *k𝑘k-dt (decision tree with k𝑘k nodes)* is defined recursively as follows.
If k=1𝑘1k=1, it is a constant function.
If k>1𝑘1k>1, it first evaluates x↦𝟏​[⟨a,x⟩−b≥0]maps-to𝑥1delimited-[]

𝑎𝑥
𝑏0x\mapsto\mathbf{1}[\left\langle a,x\right\rangle-b\geq 0],
and thereafter conditionally evaluates either
a left l𝑙l-dt or a right r𝑟r-dt where l+r<k𝑙𝑟𝑘l+r<k.
A *(t,k)𝑡𝑘(t,k)-bdt* (boosted decision tree)
evaluates x↦∑i=1tci​gi​(x)maps-to𝑥superscriptsubscript𝑖1𝑡subscript𝑐𝑖subscript𝑔𝑖𝑥x\mapsto\sum\_{i=1}^{t}c\_{i}g\_{i}(x) where each ci∈ℝsubscript𝑐𝑖ℝc\_{i}\in\mathbb{R} and each gisubscript𝑔𝑖g\_{i} is a k𝑘k-dt.

###### Lemma 2.3 (Example semi-algebraic gates).

1. 1.

   If σ:ℝ→ℝ:𝜎→ℝℝ\sigma:\mathbb{R}\to\mathbb{R} is (t,β)𝑡𝛽(t,\beta)-poly and q:ℝd→ℝ:𝑞→superscriptℝ𝑑ℝq:\mathbb{R}^{d}\to\mathbb{R} is a polynomial of degree α𝛼\alpha,
   then the standard piecewise polynomial gate σ∘q𝜎𝑞\sigma\circ q is (t,α,α​β)𝑡𝛼𝛼𝛽(t,\alpha,\alpha\beta)-sa.
   In particular, the standard ReLU gate v↦σr​(⟨a,v⟩+b)maps-to𝑣subscript𝜎r
   𝑎𝑣𝑏v\mapsto\sigma\_{\textsc{r}}(\left\langle a,v\right\rangle+b) is (1,1,1)111(1,1,1)-sa.
2. 2.

   Given polynomials (pi)i=1rsuperscriptsubscriptsubscript𝑝𝑖𝑖1𝑟(p\_{i})\_{i=1}^{r} of degree ≤αabsent𝛼\leq\alpha,
   the standard (r,α)𝑟𝛼(r,\alpha)-min and -max gates ϕmin​(v):=mini∈[r]⁡pi​(v)assignsubscriptitalic-ϕ𝑣subscript𝑖delimited-[]𝑟subscript𝑝𝑖𝑣\phi\_{\min}(v):=\min\_{i\in[r]}p\_{i}(v)
   and ϕmax​(v):=maxi∈[r]⁡qi​(v)assignsubscriptitalic-ϕ𝑣subscript𝑖delimited-[]𝑟subscript𝑞𝑖𝑣\phi\_{\max}(v):=\max\_{i\in[r]}q\_{i}(v)
   are (r​(r−1),α,α)𝑟𝑟1𝛼𝛼(r(r-1),\alpha,\alpha)-sa.
3. 3.

   Every k𝑘k-dt is (k,1,0)𝑘10(k,1,0)-sa,
   and every (t,k)𝑡𝑘(t,k)-bdt is (t​k,1,0)𝑡𝑘10(tk,1,0).

The proof of [Lemma 2.3](#S2.Thmtheorem3 "Lemma 2.3 (Example semi-algebraic gates). ‣ 2 Semi-algebraic gates and assorted network notation ‣ Benefits of depth in neural networks") is mostly a matter of unwrapping definitions, and is deferred to
[Appendix A](#A1 "Appendix A Deferred proofs ‣ Benefits of depth in neural networks").
Perhaps the only interesting encoding is for the maximization gate (and similarly the minimization gate),
which uses maxi⁡vi=∑ivi​(∏j<i𝟏​[vi>vj])​(∏j>i𝟏​[vi≥vj])subscript𝑖subscript𝑣𝑖subscript𝑖subscript𝑣𝑖subscriptproduct𝑗𝑖1delimited-[]subscript𝑣𝑖subscript𝑣𝑗subscriptproduct𝑗𝑖1delimited-[]subscript𝑣𝑖subscript𝑣𝑗\max\_{i}v\_{i}=\sum\_{i}v\_{i}(\prod\_{j<i}\mathbf{1}[v\_{i}>v\_{j}])(\prod\_{j>i}\mathbf{1}[v\_{i}\geq v\_{j}]).

### 2.1 Notation for neural networks

A semi-algebraic gate is simply a function from some domain to ℝℝ\mathbb{R},
but its role in a neural network is more complicated as the domain of the function must be
partitioned into arguments of three types: the input x∈ℝd𝑥superscriptℝ𝑑x\in\mathbb{R}^{d} to the network, the parameter vector w∈ℝp𝑤superscriptℝ𝑝w\in\mathbb{R}^{p},
and a vector of real numbers coming from parent nodes.

As a convention, the input x∈ℝd𝑥superscriptℝ𝑑x\in\mathbb{R}^{d} is only accessed by the root nodes (otherwise “layer” has no meaning).
For convenience, let layer 0 denote the input itself: d𝑑d nodes where node i𝑖i is the map x↦ximaps-to𝑥subscript𝑥𝑖x\mapsto x\_{i}.
The parameter vector w∈ℝp𝑤superscriptℝ𝑝w\in\mathbb{R}^{p} will be made available to all nodes in layers above 0, though they might only
use a subset of it. Specifically,
an internal node computes a function f:ℝp×ℝd→ℝ:𝑓→superscriptℝ𝑝superscriptℝ𝑑ℝf:\mathbb{R}^{p}\times\mathbb{R}^{d}\to\mathbb{R} using parents (f1,…,fk)subscript𝑓1…subscript𝑓𝑘(f\_{1},\ldots,f\_{k})
and a semi-algebraic gate ϕ:ℝp×ℝk→ℝ:italic-ϕ→superscriptℝ𝑝superscriptℝ𝑘ℝ\phi:\mathbb{R}^{p}\times\mathbb{R}^{k}\to\mathbb{R}, meaning
f​(w,x):=ϕ​(w1,…,wp,f1​(w,x),…,fk​(w,x))assign𝑓𝑤𝑥italic-ϕsubscript𝑤1…subscript𝑤𝑝subscript𝑓1𝑤𝑥…subscript𝑓𝑘𝑤𝑥f(w,x):=\phi(w\_{1},\ldots,w\_{p},f\_{1}(w,x),\ldots,f\_{k}(w,x)).
Another common practice is to have nodes apply a univariate *activation function* to
an affine mapping of their parents (as with piecewise polynomial gates in [Lemma 2.3](#S2.Thmtheorem3 "Lemma 2.3 (Example semi-algebraic gates). ‣ 2 Semi-algebraic gates and assorted network notation ‣ Benefits of depth in neural networks")),
where the weights in the affine combination are the parameters to the network, and additionally
correspond to edges in the graph.
It is permitted for the same parameter to appear multiple times in a network, which explains how
the number of parameters in [Theorem 1.1](#S1.Thmtheorem1 "Theorem 1.1. ‣ 1.1 Main result ‣ 1 Setting and main results ‣ Benefits of depth in neural networks") can be less than the number of edges and nodes.
The entire network computes some function F𝔊:ℝp×ℝd→ℝ:subscript𝐹𝔊→superscriptℝ𝑝superscriptℝ𝑑ℝF\_{\mathfrak{G}}:\mathbb{R}^{p}\times\mathbb{R}^{d}\to\mathbb{R}, which is equivalent to the
function computed by the single node with no outgoing edges.

As stated previously, 𝔊𝔊\mathfrak{G} will denote not just the graph (nodes and edges) underlying a network,
but also an assignment of gates to nodes, and how parameters and parent outputs are plugged into
the gates (i.e., in the preceding paragraph, how to write f𝑓f via ϕitalic-ϕ\phi).
𝒩​(𝔊)𝒩𝔊\mathcal{N}(\mathfrak{G}) is the set of functions obtained by varying w∈ℝp𝑤superscriptℝ𝑝w\in\mathbb{R}^{p},
and thus 𝒩​(𝔊):={F𝔊​(w,⋅):w∈ℝp}assign𝒩𝔊conditional-setsubscript𝐹𝔊𝑤⋅𝑤superscriptℝ𝑝\mathcal{N}(\mathfrak{G}):=\{F\_{\mathfrak{G}}(w,\cdot):w\in\mathbb{R}^{p}\} where F𝔊subscript𝐹𝔊F\_{\mathfrak{G}} is the function defined as above,
corresponding to computation performed by 𝔊𝔊\mathfrak{G}.
The results related to VC dimension, meaning [Theorem 1.2](#S1.Thmtheorem2 "Theorem 1.2. ‣ 1.2 Companion results ‣ 1 Setting and main results ‣ Benefits of depth in neural networks") and [Lemma 1.3](#S1.Thmtheorem3 "Lemma 1.3 (Simplification of Lemma 4.2). ‣ 1.2 Companion results ‣ 1 Setting and main results ‣ Benefits of depth in neural networks"),
will use the class 𝒩​(𝔊)𝒩𝔊\mathcal{N}(\mathfrak{G}).

Some of the results, for instance [Theorem 1.1](#S1.Thmtheorem1 "Theorem 1.1. ‣ 1.1 Main result ‣ 1 Setting and main results ‣ Benefits of depth in neural networks") and its generalization [Theorem 3.19](#S3.Thmtheorem19 "Theorem 3.19. ‣ 3.4 Proof of Theorem 1.1 ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks"),
will let not only the parameters but also network graph 𝔊𝔊\mathfrak{G} vary.
Let 𝒩d​((mi,ti,αi,βi)i=1l)subscript𝒩𝑑superscriptsubscriptsubscript𝑚𝑖subscript𝑡𝑖subscript𝛼𝑖subscript𝛽𝑖𝑖1𝑙\mathcal{N}\_{d}((m\_{i},t\_{i},\alpha\_{i},\beta\_{i})\_{i=1}^{l}) denote a network where layer i𝑖i has ≤miabsentsubscript𝑚𝑖\leq m\_{i} nodes
where each is (ti,αi,βi)subscript𝑡𝑖subscript𝛼𝑖subscript𝛽𝑖(t\_{i},\alpha\_{i},\beta\_{i})-sa and the input has dimension d𝑑d.
As a simplification, let 𝒩d​(m,l,t,α,β)subscript𝒩𝑑𝑚𝑙𝑡𝛼𝛽\mathcal{N}\_{d}(m,l,t,\alpha,\beta) denote networks of (t,α,β)𝑡𝛼𝛽(t,\alpha,\beta)-sa gates in ≤labsent𝑙\leq l layers (not including layer 0)
each with ≤mabsent𝑚\leq m nodes.
There are various empirical prescriptions on how to vary the number of nodes per layer;
for instance, convolutional networks typically have an increase between layer 0 and layer 1,
followed by exponential decrease for a few layers, and finally a few layers with the same number of nodes
(Fukushima, [1980](#bib.bib7); LeCun et al., [1998](#bib.bib13); Krizhevsky et al., [2012](#bib.bib12)).

## 3 Benefits of depth

The purpose of this section is to prove [Theorem 1.1](#S1.Thmtheorem1 "Theorem 1.1. ‣ 1.1 Main result ‣ 1 Setting and main results ‣ Benefits of depth in neural networks") and its generalization [Theorem 3.19](#S3.Thmtheorem19 "Theorem 3.19. ‣ 3.4 Proof of Theorem 1.1 ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks")
in the following three steps.

1. 1.

   Functions with few oscillations poorly approximate functions with many oscillations.
2. 2.

   Functions computed by networks with few layers must have few oscillations.
3. 3.

   Functions computed by networks with many layers can have many oscillations.

### 3.1 Approximation via oscillation counting

!(/html/1602.04485/assets/x1.png)

Figure 1: f𝑓f crosses more than g𝑔g.

The idea behind this first step is depicted at right.
Given functions f:ℝ→ℝ:𝑓→ℝℝf:\mathbb{R}\to\mathbb{R} and g:ℝ→ℝ:𝑔→ℝℝg:\mathbb{R}\to\mathbb{R} (the multivariate case will come soon),
let ℐfsubscriptℐ𝑓\mathcal{I}\_{f} and ℐgsubscriptℐ𝑔\mathcal{I}\_{g} denote partitions of ℝℝ\mathbb{R} into intervals so
that the classifiers f~​(x)=𝟏​[f​(x)≥1/2]~𝑓𝑥1delimited-[]𝑓𝑥12\tilde{f}(x)=\mathbf{1}[f(x)\geq 1/2] and g~~𝑔\tilde{g} are constant
within each interval.
To formally count oscillations, define
*the crossing number *Cr*​(f)*Cr*𝑓\textup{Cr}(f) of f𝑓f* as Cr​(f)=|ℐf|Cr𝑓subscriptℐ𝑓\textup{Cr}(f)=|\mathcal{I}\_{f}| (thus Cr​(σr)=2Crsubscript𝜎r2\textup{Cr}(\sigma\_{\textsc{r}})=2).
If Cr​(f)Cr𝑓\textup{Cr}(f) is much larger than Cr​(g)Cr𝑔\textup{Cr}(g),
then most piecewise constant regions of g~~𝑔\tilde{g} will exhibit many oscillations of f𝑓f,
and thus g𝑔g poorly approximates f𝑓f.

###### Lemma 3.1.

Let f:ℝ→ℝ:𝑓→ℝℝf:\mathbb{R}\to\mathbb{R} and g:ℝ→ℝ:𝑔→ℝℝg:\mathbb{R}\to\mathbb{R} be given,
and take ℐfsubscriptℐ𝑓\mathcal{I}\_{f} to denote the partition of ℝℝ\mathbb{R} given by the pieces of f~~𝑓\tilde{f}
(meaning |ℐf|=Cr​(f)subscriptℐ𝑓Cr𝑓|\mathcal{I}\_{f}|=\textup{Cr}(f)).
Then

|  |  |  |
| --- | --- | --- |
|  | 1Cr​(f)​∑U∈ℐf𝟏​[∀x∈U​\centerdot​f~​(x)≠g~​(x)]≥12​(1−2​(Cr​(g)Cr​(f))).1Cr𝑓subscript𝑈subscriptℐ𝑓1delimited-[]for-all𝑥𝑈\centerdot~𝑓𝑥~𝑔𝑥1212Cr𝑔Cr𝑓\frac{1}{\textup{Cr}(f)}\sum\_{U\in\mathcal{I}\_{f}}\mathbf{1}[\forall x\in U\centerdot\tilde{f}(x)\neq\tilde{g}(x)]\geq\frac{1}{2}\left(1-2\left(\frac{\textup{Cr}(g)}{\textup{Cr}(f)}\right)\right). |  |

The arguably strange form of the left hand side of the bound in
[Lemma 3.1](#S3.Thmtheorem1 "Lemma 3.1. ‣ 3.1 Approximation via oscillation counting ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks") is to accommodate different notions of distance.
For the L1superscript𝐿1L^{1} distance with the Lebesgue measure as in [Theorem 1.1](#S1.Thmtheorem1 "Theorem 1.1. ‣ 1.1 Main result ‣ 1 Setting and main results ‣ Benefits of depth in neural networks"),
it does not suffice for f𝑓f to cross 1/2: it must be *regular*,
meaning
it must cross by an appreciable distance, and the crossings must be evenly spaced.
(It is worth highlighting that the ReLU easily gives rise to a regular f𝑓f.)
However, to merely show that f𝑓f and g𝑔g give very different classifiers f~~𝑓\tilde{f} and g~~𝑔\tilde{g}
over an arbitrary measure (as in part of [Theorem 3.19](#S3.Thmtheorem19 "Theorem 3.19. ‣ 3.4 Proof of Theorem 1.1 ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks")), no additional regularity is needed.

###### Proof 3.2.

(of [Lemma 3.1](#S3.Thmtheorem1 "Lemma 3.1. ‣ 3.1 Approximation via oscillation counting ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks"))
Let ℐfsubscriptℐf\mathcal{I}\_{f} and ℐgsubscriptℐg\mathcal{I}\_{g} respectively denote the sets of intervals
corresponding to f~~f\tilde{f} and g~~g\tilde{g},
and set sf:=Cr​(f)=|ℐf|assignsubscriptsfCrfsubscriptℐfs\_{f}:=\textup{Cr}(f)=|\mathcal{I}\_{f}| and sg:=Cr​(g)=|ℐg|assignsubscriptsgCrgsubscriptℐgs\_{g}:=\textup{Cr}(g)=|\mathcal{I}\_{g}|.

For every J∈ℐg𝐽subscriptℐ𝑔J\in\mathcal{I}\_{g}, set XJ:={U∈ℐf:U⊆J}assignsubscript𝑋𝐽conditional-set𝑈subscriptℐ𝑓𝑈𝐽X\_{J}:=\{U\in\mathcal{I}\_{f}:U\subseteq J\}.
Fixing any J∈ℐg𝐽subscriptℐ𝑔J\in\mathcal{I}\_{g},
since g~~𝑔\tilde{g} is constant on J𝐽J whereas f~~𝑓\tilde{f} alternates,
the number of elements in XJsubscript𝑋𝐽X\_{J} where g~~𝑔\tilde{g} disagrees everywhere with f~~𝑓\tilde{f}
is |XJ|/2subscript𝑋𝐽2|X\_{J}|/2 when |XJ|subscript𝑋𝐽|X\_{J}| is even
and at least (|XJ|−1)/2subscript𝑋𝐽12(|X\_{J}|-1)/2 when |XJ|subscript𝑋𝐽|X\_{J}| is odd,
thus at least (|XJ|−1)/2subscript𝑋𝐽12(|X\_{J}|-1)/2 in general.
As such,

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 1sf​∑U∈ℐf𝟏​[∀x∈U​\centerdot​f~​(x)≠g~​(x)]1subscript𝑠𝑓subscript𝑈subscriptℐ𝑓1delimited-[]for-all𝑥𝑈\centerdot~𝑓𝑥~𝑔𝑥\displaystyle\frac{1}{s\_{f}}\sum\_{U\in\mathcal{I}\_{f}}\mathbf{1}[\forall x\in U\centerdot\tilde{f}(x)\neq\tilde{g}(x)] | ≥1sf​∑J∈ℐg∑U∈XJ𝟏​[∀x∈U​\centerdot​f~​(x)≠g~​(x)]≥1sf​∑J∈ℐg|XJ|−12.absent1subscript𝑠𝑓subscript𝐽subscriptℐ𝑔subscript𝑈subscript𝑋𝐽1delimited-[]for-all𝑥𝑈\centerdot~𝑓𝑥~𝑔𝑥1subscript𝑠𝑓subscript𝐽subscriptℐ𝑔subscript𝑋𝐽12\displaystyle\geq\frac{1}{s\_{f}}\sum\_{J\in\mathcal{I}\_{g}}\sum\_{U\in X\_{J}}\mathbf{1}[\forall x\in U\centerdot\tilde{f}(x)\neq\tilde{g}(x)]\geq\frac{1}{s\_{f}}\sum\_{J\in\mathcal{I}\_{g}}\frac{|X\_{J}|-1}{2}. |  | (3.1) |

To control this expression, note that every XJsubscript𝑋𝐽X\_{J} is disjoint, however X:=∪J∈ℐjXjassign𝑋subscript𝐽subscriptℐ𝑗subscript𝑋𝑗X:=\cup\_{J\in\mathcal{I}\_{j}}X\_{j}
can be smaller than ℐfsubscriptℐ𝑓\mathcal{I}\_{f}: in particular, it misses intervals U∈ℐf𝑈subscriptℐ𝑓U\in\mathcal{I}\_{f}
whose interior intersects with the boundary of an interval in ℐgsubscriptℐ𝑔\mathcal{I}\_{g}.
Since there are at most sg−1subscript𝑠𝑔1s\_{g}-1 such boundaries,

|  |  |  |
| --- | --- | --- |
|  | sf=|ℐf|≤sg−1+|X|≤sg+∑J∈ℐg|XJ|,subscript𝑠𝑓subscriptℐ𝑓subscript𝑠𝑔1𝑋subscript𝑠𝑔subscript𝐽subscriptℐ𝑔subscript𝑋𝐽s\_{f}=|\mathcal{I}\_{f}|\leq s\_{g}-1+|X|\leq s\_{g}+\sum\_{J\in\mathcal{I}\_{g}}|X\_{J}|, |  |

which rearranges to gives ∑J∈ℐg|XJ|≥sf−sgsubscript𝐽subscriptℐ𝑔subscript𝑋𝐽subscript𝑠𝑓subscript𝑠𝑔\sum\_{J\in\mathcal{I}\_{g}}|X\_{J}|\geq s\_{f}-s\_{g}.
Combining this with [eq. 3.1](#S3.E1 "In Proof 3.2. ‣ 3.1 Approximation via oscillation counting ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks"),

|  |  |  |
| --- | --- | --- |
|  | 1sf​∑U∈ℐf𝟏​[∀x∈U​\centerdot​f~​(x)≠g~​(x)]≥12​sf​(sf−sg−sg)=12​(1−2​sgsf).1subscript𝑠𝑓subscript𝑈subscriptℐ𝑓1delimited-[]for-all𝑥𝑈\centerdot~𝑓𝑥~𝑔𝑥12subscript𝑠𝑓subscript𝑠𝑓subscript𝑠𝑔subscript𝑠𝑔1212subscript𝑠𝑔subscript𝑠𝑓\frac{1}{s\_{f}}\sum\_{U\in\mathcal{I}\_{f}}\mathbf{1}[\forall x\in U\centerdot\tilde{f}(x)\neq\tilde{g}(x)]\geq\frac{1}{2s\_{f}}\left(s\_{f}-s\_{g}-s\_{g}\right)=\frac{1}{2}\left(1-\frac{2s\_{g}}{s\_{f}}\right). |  |

### 3.2 Few layers, few oscillations

As in the preceding section, oscillations of a function f𝑓f will be counted via the crossing number Cr​(f)Cr𝑓\textup{Cr}(f).
Since Cr​(⋅)Cr⋅\textup{Cr}(\cdot) only handles univariate functions, the multivariate case is handled by first choosing an affine
map h:ℝ→ℝd:ℎ→ℝsuperscriptℝ𝑑h:\mathbb{R}\to\mathbb{R}^{d} (meaning h​(z)=a​z+bℎ𝑧𝑎𝑧𝑏h(z)=az+b) and considering Cr​(f∘h)Cr𝑓ℎ\textup{Cr}(f\circ h).

Before giving the central upper bounds and sketching their proofs, notice by analogy to polynomials how
compositions and additions vary in their impact upon oscillations. By adding together two polynomials,
the resulting polynomial has at most twice as many terms and does not exceed the maximum degree of either polynomial.
On the other hand, composing polynomials, the result has the product of the degrees and can have more than the product
of the terms. As both of these can impact the number of
roots or crossings (e.g., by the Bezout Theorem or Descartes’ Rule of Signs),
composition wins the race to higher oscillations.

###### Lemma 3.3.

Let h:ℝ→ℝd:ℎ→ℝsuperscriptℝ𝑑h:\mathbb{R}\to\mathbb{R}^{d} be affine.

1. 1.

   Suppose f∈𝒩d​((mi,ti,αi,βi)i=1l)𝑓subscript𝒩𝑑superscriptsubscriptsubscript𝑚𝑖subscript𝑡𝑖subscript𝛼𝑖subscript𝛽𝑖𝑖1𝑙f\in\mathcal{N}\_{d}((m\_{i},t\_{i},\alpha\_{i},\beta\_{i})\_{i=1}^{l})
   with mini⁡min⁡{αi,βi}≥1subscript𝑖subscript𝛼𝑖subscript𝛽𝑖1\min\_{i}\min\{\alpha\_{i},\beta\_{i}\}\geq 1.
   Setting α:=maxi⁡αi,β:=maxi⁡βiformulae-sequenceassign𝛼subscript𝑖subscript𝛼𝑖assign𝛽subscript𝑖subscript𝛽𝑖\alpha:=\max\_{i}\alpha\_{i},\beta:=\max\_{i}\beta\_{i}, t:=maxi⁡tiassign𝑡subscript𝑖subscript𝑡𝑖t:=\max\_{i}t\_{i},
   m:=∑imiassign𝑚subscript𝑖subscript𝑚𝑖m:=\sum\_{i}m\_{i}, then
   Cr​(f∘h)≤2​(2​t​m​α/l)l​βl2Cr𝑓ℎ2superscript2𝑡𝑚𝛼𝑙𝑙superscript𝛽superscript𝑙2\textup{Cr}(f\circ h)\leq 2(2tm\alpha/l)^{l}\beta^{l^{2}}.
2. 2.

   Let k𝑘k-dt f:ℝd→ℝ:𝑓→superscriptℝ𝑑ℝf:\mathbb{R}^{d}\to\mathbb{R} and (t,k)𝑡𝑘(t,k)-bdt g:ℝd→ℝ:𝑔→superscriptℝ𝑑ℝg:\mathbb{R}^{d}\to\mathbb{R} be given. Then Cr​(f∘h)≤kCr𝑓ℎ𝑘\textup{Cr}(f\circ h)\leq k and Cr​(g∘h)≤2​t​kCr𝑔ℎ2𝑡𝑘\textup{Cr}(g\circ h)\leq 2tk.

[Lemma 3.3](#S3.Thmtheorem3 "Lemma 3.3. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks") shows the key tradeoff: the number of layers is in the exponent,
while the number of nodes is in the base.

Rather than directly controlling Cr​(f∘h)Cr𝑓ℎ\textup{Cr}(f\circ h), the proofs will first show f∘h𝑓ℎf\circ h is (t,α)𝑡𝛼(t,\alpha)-poly,
which immediately bounds Cr​(f∘h)Cr𝑓ℎ\textup{Cr}(f\circ h) as follows.

###### Lemma 3.4.

If f:ℝ→ℝ:𝑓→ℝℝf:\mathbb{R}\to\mathbb{R} is (t,α)𝑡𝛼(t,\alpha)-poly,
then Cr​(f)≤t​(1+α)Cr𝑓𝑡1𝛼\textup{Cr}(f)\leq t(1+\alpha).

###### Proof 3.5.

The polynomial in each piece has at most α𝛼\alpha roots,
which thus divides each piece into ≤1+αabsent1𝛼\leq 1+\alpha further pieces
within which f~~𝑓\tilde{f} is constant.

A second technical lemma is needed to reason about combinations
of partitions defined by (t,α,β)𝑡𝛼𝛽(t,\alpha,\beta)-sa and (t,α)𝑡𝛼(t,\alpha)-poly functions.

###### Lemma 3.6.

Let k𝑘k partitions (Ai)i=1ksuperscriptsubscriptsubscript𝐴𝑖𝑖1𝑘(A\_{i})\_{i=1}^{k} of ℝℝ\mathbb{R} each into at most t𝑡t intervals be given,
and set A:=∪iAiassign𝐴subscript𝑖subscript𝐴𝑖A:=\cup\_{i}A\_{i}.
Then there exists a partition B𝐵B of ℝℝ\mathbb{R} of size at most k​t𝑘𝑡kt
so that every interval expressible as a union of intersections of elements of A𝐴A
is a union of elements of B𝐵B.

!(/html/1602.04485/assets/x2.png)

Figure 2: Three partitions.

The proof is somewhat painful owing to the fact that there is no convention on the structure
of the intervals in the partitions, namely which ends are closed and which are open,
and is thus deferred to [Appendix A](#A1 "Appendix A Deferred proofs ‣ Benefits of depth in neural networks").
The principle of the proof is elementary, and is depicted at right:
given a collection of partitions, an intersection of constituent intervals must share endpoints
with intervals in in the intersection, thus the total number of intervals bounds the total number
of possible intersections.
Arguably, this failure to increase complexity in the face of arbitrary intersections
is why semi-algebraic gates do not care about the number of terms in their definition.

Recall that (t,α,β)𝑡𝛼𝛽(t,\alpha,\beta)-sa means there is a set of t𝑡t polynomials of degree at most α𝛼\alpha
which form the regions defining the function by intersecting simpler regions
x↦𝟏​[q​(x)≥0]maps-to𝑥1delimited-[]𝑞𝑥0x\mapsto\mathbf{1}[q(x)\geq 0] and x↦𝟏​[q​(x)<0]maps-to𝑥1delimited-[]𝑞𝑥0x\mapsto\mathbf{1}[q(x)<0].
As such, in order to analyze semi-algebraic gates composed with piecewise polynomial gates,
consider first the behavior of these predicate polynomials.

###### Lemma 3.7.

Suppose f:ℝk→ℝ:𝑓→superscriptℝ𝑘ℝf:\mathbb{R}^{k}\to\mathbb{R} is polynomial with degree ≤αabsent𝛼\leq\alpha
and (gi)i=1ksuperscriptsubscriptsubscript𝑔𝑖𝑖1𝑘(g\_{i})\_{i=1}^{k} are each (t,γ)𝑡𝛾(t,\gamma)-poly.
Then h​(x):=f​(g1​(x),…,gk​(x))assignℎ𝑥𝑓subscript𝑔1𝑥…subscript𝑔𝑘𝑥h(x):=f(g\_{1}(x),\ldots,g\_{k}(x)) is (t​k,α​γ)𝑡𝑘𝛼𝛾(tk,\alpha\gamma)-poly,
and the partition defining hℎh is a refinement of the partitions for each gisubscript𝑔𝑖g\_{i}
(in particular, each gisubscript𝑔𝑖g\_{i} is a fixed polynomial (of degree ≤γabsent𝛾\leq\gamma)
within the ≤t​kabsent𝑡𝑘\leq tk pieces defining hℎh).

###### Proof 3.8.

By [Lemma 3.6](#S3.Thmtheorem6 "Lemma 3.6. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks"),
there exists a partition of ℝℝ\mathbb{R} into ≤t​kabsent𝑡𝑘\leq tk intervals
which refines the partitions defining each gisubscript𝑔𝑖g\_{i}.
Since f𝑓f is a polynomial with degree ≤αabsent𝛼\leq\alpha,
then within each of these intervals, its composition with (g1,…,gk)subscript𝑔1…subscript𝑔𝑘(g\_{1},\ldots,g\_{k})
gives a polynomial of degree ≤α​γabsent𝛼𝛾\leq\alpha\gamma.

This gives the following complexity bound for composing (s,α,β)𝑠𝛼𝛽(s,\alpha,\beta)-sa and (t,γ)𝑡𝛾(t,\gamma)-poly gates.

###### Lemma 3.9.

Suppose f:ℝk→ℝ:𝑓→superscriptℝ𝑘ℝf:\mathbb{R}^{k}\to\mathbb{R} is (s,α,β)𝑠𝛼𝛽(s,\alpha,\beta)-sa
and (g1,…,gk)subscript𝑔1…subscript𝑔𝑘(g\_{1},\ldots,g\_{k}) are (t,γ)𝑡𝛾(t,\gamma)-poly.
Then h​(x):=f​(g1​(x),…,gk​(x))assignℎ𝑥𝑓subscript𝑔1𝑥…subscript𝑔𝑘𝑥h(x):=f(g\_{1}(x),\ldots,g\_{k}(x)) is (s​t​k​(1+α​γ),β​γ)𝑠𝑡𝑘1𝛼𝛾𝛽𝛾(stk(1+\alpha\gamma),\beta\gamma)-poly.

###### Proof 3.10.

By definition, f𝑓f is polynomial in regions defined by intersections of the predicates
Ui​(x)=𝟏​[qi​(x)≥0]subscript𝑈𝑖𝑥1delimited-[]subscript𝑞𝑖𝑥0U\_{i}(x)=\mathbf{1}[q\_{i}(x)\geq 0] and Li​(x)=𝟏​[qi​(x)<0]subscript𝐿𝑖𝑥1delimited-[]subscript𝑞𝑖𝑥0L\_{i}(x)=\mathbf{1}[q\_{i}(x)<0].
By [Lemma 3.7](#S3.Thmtheorem7 "Lemma 3.7. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks"), qi​(g1,…,gk)subscript𝑞𝑖subscript𝑔1…subscript𝑔𝑘q\_{i}(g\_{1},\ldots,g\_{k}) is (t​k,α​γ)𝑡𝑘𝛼𝛾(tk,\alpha\gamma)-poly,
thus Uisubscript𝑈𝑖U\_{i} and Lisubscript𝐿𝑖L\_{i} together define a partition of ℝℝ\mathbb{R} which has Cr​(x↦qi​(g1​(x),…,gk​(x)))Crmaps-to𝑥subscript𝑞𝑖subscript𝑔1𝑥…subscript𝑔𝑘𝑥\textup{Cr}(x\mapsto q\_{i}(g\_{1}(x),\ldots,g\_{k}(x)))
pieces, which by [Lemma 3.4](#S3.Thmtheorem4 "Lemma 3.4. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks") has cardinality at most t​k​(1+α​γ)𝑡𝑘1𝛼𝛾tk(1+\alpha\gamma)
and refines the partitions for each gisubscript𝑔𝑖g\_{i}.
By [Lemma 3.6](#S3.Thmtheorem6 "Lemma 3.6. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks"), these partitions across all predicate polynomials (qi)i=1ssuperscriptsubscriptsubscript𝑞𝑖𝑖1𝑠(q\_{i})\_{i=1}^{s}
can be refined into a single partition of size
≤s​t​k​(1+α​γ)absent𝑠𝑡𝑘1𝛼𝛾\leq stk(1+\alpha\gamma),
and which thus also refines the partitions defined by (g1,…,gk)subscript𝑔1…subscript𝑔𝑘(g\_{1},\ldots,g\_{k}).
Thanks to these refinements, hℎh over any element U𝑈U of this final partition
is a fixed polynomial pU​(g1,…,gk)subscript𝑝𝑈subscript𝑔1…subscript𝑔𝑘p\_{U}(g\_{1},\ldots,g\_{k}) of degree ≤β​γabsent𝛽𝛾\leq\beta\gamma,
meaning hℎh is (s​t​k​(1+α​γ),β​γ)𝑠𝑡𝑘1𝛼𝛾𝛽𝛾(stk(1+\alpha\gamma),\beta\gamma)-poly.

The proof of [Lemma 3.3](#S3.Thmtheorem3 "Lemma 3.3. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks") now follows by [Lemma 3.9](#S3.Thmtheorem9 "Lemma 3.9. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks").
In particular, for semi-algebraic networks, the proof is an induction over layers,
establishing node j𝑗j is (tj,αj)subscript𝑡𝑗subscript𝛼𝑗(t\_{j},\alpha\_{j})-poly (for appropriate (tj,αj)subscript𝑡𝑗subscript𝛼𝑗(t\_{j},\alpha\_{j})).

### 3.3 Many layers, many oscillations

The idea behind this construction is as follows. Consider any continuous function f:[0,1]→[0,1]:𝑓→0101f:[0,1]\to[0,1] which is a generalization of
a triangle wave with a single peak: f​(0)=f​(1)=0𝑓0𝑓10f(0)=f(1)=0,
and there is some a∈(0,1)𝑎01a\in(0,1) with f​(a)=1𝑓𝑎1f(a)=1, and additionally f𝑓f strictly increases along [0,a]0𝑎[0,a] and strictly decreases along
[a,1]𝑎1[a,1].

Now consider the effect of the composition f∘f=f2𝑓𝑓superscript𝑓2f\circ f=f^{2}. Along [0,a]0𝑎[0,a], this is a stretched copy of f𝑓f, since f​(f​(a))=f​(1)=0=f​(0)=f​(f​(0))𝑓𝑓𝑎𝑓10𝑓0𝑓𝑓0f(f(a))=f(1)=0=f(0)=f(f(0))
and moreover f𝑓f is a bijection between [0,a]0𝑎[0,a] and [0,1]01[0,1] (when restricted to [0,a]0𝑎[0,a]). The same reasoning applies to f2superscript𝑓2f^{2} along [a,1]𝑎1[a,1],
meaning f2superscript𝑓2f^{2} is a function with two peaks. Iterating this argument implies fksuperscript𝑓𝑘f^{k} is a function with 2k−1superscript2𝑘12^{k-1} peaks; the following
definition and lemmas formalize this reasoning.

###### Definition 3.11.

f𝑓f is *(t,[a,b])𝑡𝑎𝑏(t,[a,b])-triangle* when it is continuous along [a,b]𝑎𝑏[a,b], and [a,b]𝑎𝑏[a,b] may
be divided into 2​t2𝑡2t intervals [ai,ai+1]subscript𝑎𝑖subscript𝑎𝑖1[a\_{i},a\_{i+1}] with a1=asubscript𝑎1𝑎a\_{1}=a and a2​t+1=bsubscript𝑎2𝑡1𝑏a\_{2t+1}=b,
f​(ai)=f​(ai+2)𝑓subscript𝑎𝑖𝑓subscript𝑎𝑖2f(a\_{i})=f(a\_{i+2}) whenever 1≤i≤2​t−11𝑖2𝑡11\leq i\leq 2t-1,
f​(a1)=0𝑓subscript𝑎10f(a\_{1})=0,
f​(a2)=1𝑓subscript𝑎21f(a\_{2})=1,
f𝑓f is strictly increasing along odd-numbered intervals (those starting from aisubscript𝑎𝑖a\_{i} with i𝑖i odd),
and strictly decreasing along even-numbered intervals.

###### Lemma 3.12.

If f𝑓f is (s,[0,1])𝑠01(s,[0,1])-triangle and g𝑔g is (t,[0,1])𝑡01(t,[0,1])-triangle,
then f∘g𝑓𝑔f\circ g is (2​s​t,[0,1])2𝑠𝑡01(2st,[0,1])-triangle.

###### Proof 3.13.

Since g​([0,1])=[0,1]𝑔0101g([0,1])=[0,1] and f𝑓f and g𝑔g are continuous along [0,1]01[0,1],
then f∘g𝑓𝑔f\circ g is continuous along [0,1]01[0,1].
In the remaining analysis,
let (a1,…,a2​s+1)subscript𝑎1…subscript𝑎2𝑠1(a\_{1},\ldots,a\_{2s+1}) and (c1,…,c2​t+1)subscript𝑐1…subscript𝑐2𝑡1(c\_{1},\ldots,c\_{2t+1})
respectively denote the interval boundaries for f𝑓f and g𝑔g.

Now consider any interval [cj,cj+1]subscript𝑐𝑗subscript𝑐𝑗1[c\_{j},c\_{j+1}] where j𝑗j is odd,
meaning the restriction gj:[cj,cj+1]→[0,1]:subscript𝑔𝑗→subscript𝑐𝑗subscript𝑐𝑗101g\_{j}:[c\_{j},c\_{j+1}]\to[0,1] of g𝑔g to [cj,cj+1]subscript𝑐𝑗subscript𝑐𝑗1[c\_{j},c\_{j+1}] is strictly increasing.
It will be shown that f∘gj𝑓subscript𝑔𝑗f\circ g\_{j} is (s,[cj,cj+1])𝑠subscript𝑐𝑗subscript𝑐𝑗1(s,[c\_{j},c\_{j+1}])-triangle,
and an analogous proof holds for the strictly decreasing restriction
gj+1:[cj+1,cj+2]→[0,1]:subscript𝑔𝑗1→subscript𝑐𝑗1subscript𝑐𝑗201g\_{j+1}:[c\_{j+1},c\_{j+2}]\to[0,1],
whereby it follows that f∘g𝑓𝑔f\circ g is (2​s​t,[0,1])2𝑠𝑡01(2st,[0,1]) by considering all choices of j𝑗j.

To this end, note for any i∈{1,…,2​s+1}𝑖1…2𝑠1i\in\{1,\ldots,2s+1\} that gj−1​(ai)superscriptsubscript𝑔𝑗1subscript𝑎𝑖g\_{j}^{-1}(a\_{i}) exists and is unique,
thus set ai′:=gj−1​(ai)assignsubscriptsuperscript𝑎′𝑖superscriptsubscript𝑔𝑗1subscript𝑎𝑖a^{\prime}\_{i}:=g\_{j}^{-1}(a\_{i}).
By this choice,
for odd i𝑖i it holds that f​(gj​(ai′))=f​(gj​(gj−1​(ai)))=f​(ai)=f​(a1)=0𝑓subscript𝑔𝑗subscriptsuperscript𝑎′𝑖𝑓subscript𝑔𝑗superscriptsubscript𝑔𝑗1subscript𝑎𝑖𝑓subscript𝑎𝑖𝑓subscript𝑎10f(g\_{j}(a^{\prime}\_{i}))=f(g\_{j}(g\_{j}^{-1}(a\_{i})))=f(a\_{i})=f(a\_{1})=0
and f∘gj𝑓subscript𝑔𝑗f\circ g\_{j} is strictly increasing
along [ai′,ai+1′]subscriptsuperscript𝑎′𝑖subscriptsuperscript𝑎′𝑖1[a^{\prime}\_{i},a^{\prime}\_{i+1}] (since gjsubscript𝑔𝑗g\_{j} is strictly increasing everywhere
and f𝑓f is strictly increasing along [gj​(ai′),gj​(ai+1′)]=[ai,ai+1]subscript𝑔𝑗subscriptsuperscript𝑎′𝑖subscript𝑔𝑗subscriptsuperscript𝑎′𝑖1subscript𝑎𝑖subscript𝑎𝑖1[g\_{j}(a^{\prime}\_{i}),g\_{j}(a^{\prime}\_{i+1})]=[a\_{i},a\_{i+1}]),
and similarly even i𝑖i has f​(gj​(ai′))=f​(a2)=1𝑓subscript𝑔𝑗subscriptsuperscript𝑎′𝑖𝑓subscript𝑎21f(g\_{j}(a^{\prime}\_{i}))=f(a\_{2})=1
and f∘gj𝑓subscript𝑔𝑗f\circ g\_{j} is strictly decreasing along [ai′,ai+1′]subscriptsuperscript𝑎′𝑖subscriptsuperscript𝑎′𝑖1[a^{\prime}\_{i},a^{\prime}\_{i+1}].

###### Corollary 3.14.

If f∈𝒩1​(m,l,t,α,β)𝑓subscript𝒩1𝑚𝑙𝑡𝛼𝛽f\in\mathcal{N}\_{1}(m,l,t,\alpha,\beta) is (t,[0,1])𝑡01(t,[0,1])-triangle with p𝑝p distinct parameters,
then fk∈𝒩1​(m,k​l,t,α,β)superscript𝑓𝑘subscript𝒩1𝑚𝑘𝑙𝑡𝛼𝛽f^{k}\in\mathcal{N}\_{1}(m,kl,t,\alpha,\beta) is (2k−1​tk,[0,1])superscript2𝑘1superscript𝑡𝑘01(2^{k-1}t^{k},[0,1])-triangle with p𝑝p distinct parameters
and Cr​(fk)=(2​t)k+1Crsuperscript𝑓𝑘superscript2𝑡𝑘1\textup{Cr}(f^{k})=(2t)^{k}+1.

###### Proof 3.15.

It suffices to perform k−1𝑘1k-1 applications of [Lemma 3.12](#S3.Thmtheorem12 "Lemma 3.12. ‣ 3.3 Many layers, many oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks").

Next, note the following examples of triangle functions.

###### Lemma 3.16.

The following functions are (1,[0,1])101(1,[0,1])-triangle.

1. 1.

   f​(z):=σr​(2​σr​(z)−4​σr​(z−1/2))∈𝒩1​(2,1,1,1,1)assign𝑓𝑧subscript𝜎r2subscript𝜎r𝑧4subscript𝜎r𝑧12subscript𝒩121111f(z):=\sigma\_{\textsc{r}}(2\sigma\_{\textsc{r}}(z)-4\sigma\_{\textsc{r}}(z-1/2))\in\mathcal{N}\_{1}(2,1,1,1,1).
2. 2.

   g​(z):=min⁡{σr​(2​z),σr​(2−2​z)}∈𝒩1​(2,1,2,1,1)assign𝑔𝑧subscript𝜎r2𝑧subscript𝜎r22𝑧subscript𝒩121211g(z):=\min\{\sigma\_{\textsc{r}}(2z),\sigma\_{\textsc{r}}(2-2z)\}\in\mathcal{N}\_{1}(2,1,2,1,1).
3. 3.

   h​(z):=4​z​(1−z)∈𝒩1​(1,1,0,2,0)assignℎ𝑧4𝑧1𝑧subscript𝒩111020h(z):=4z(1-z)\in\mathcal{N}\_{1}(1,1,0,2,0).
   Cf. Schmitt ([2000](#bib.bib18)).

Lastly, consider the first example f​(z)=σr​(2​σr​(z)−4​(σr​(z−1/2)))=min⁡{σr​(2​z),σr​(2−2​z)}𝑓𝑧subscript𝜎r2subscript𝜎r𝑧4subscript𝜎r𝑧12subscript𝜎r2𝑧subscript𝜎r22𝑧f(z)=\sigma\_{\textsc{r}}(2\sigma\_{\textsc{r}}(z)-4(\sigma\_{\textsc{r}}(z-1/2)))=\min\{\sigma\_{\textsc{r}}(2z),\sigma\_{\textsc{r}}(2-2z)\},
whose graph linearly interpolates (in ℝ2superscriptℝ2\mathbb{R}^{2}) between (0,0)00(0,0), (1/2,1)121(1/2,1), and (1,0)10(1,0).
Consequently, f∘f𝑓𝑓f\circ f along [0,1/2]012[0,1/2] linear interpolates between (0,0)00(0,0), (1/4,1)141(1/4,1), and (1/2,1)121(1/2,1), and
f∘f𝑓𝑓f\circ f is analogous on [1/2,1]121[1/2,1], meaning it has produced two copies of f𝑓f and then shrunken them horizontally
by a factor of 2. This process repeats, meaning fksuperscript𝑓𝑘f^{k} has 2k−1superscript2𝑘12^{k-1} copies of f𝑓f, and grants the regularity
needed to use the Lebesgue measure in [Theorem 1.1](#S1.Thmtheorem1 "Theorem 1.1. ‣ 1.1 Main result ‣ 1 Setting and main results ‣ Benefits of depth in neural networks").

###### Lemma 3.17.

Set f​(z):=σr​(2​σr​(z)−4​σr​(z−1/2))∈𝒩1​(2,1,1,1,1)assign𝑓𝑧subscript𝜎r2subscript𝜎r𝑧4subscript𝜎r𝑧12subscript𝒩121111f(z):=\sigma\_{\textsc{r}}(2\sigma\_{\textsc{r}}(z)-4\sigma\_{\textsc{r}}(z-1/2))\in\mathcal{N}\_{1}(2,1,1,1,1) (cf. [Lemma 3.16](#S3.Thmtheorem16 "Lemma 3.16. ‣ 3.3 Many layers, many oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks")).
Let real z∈[0,1]𝑧01z\in[0,1] and positive integer k𝑘k be given,
and choose the unique nonnegative integer ik∈{0,…,2k−1}subscript𝑖𝑘0…superscript2𝑘1i\_{k}\in\{0,\ldots,2^{k-1}\} and real zk∈[0,1)subscript𝑧𝑘01z\_{k}\in[0,1)
so that z=(ik+zk)​21−k𝑧subscript𝑖𝑘subscript𝑧𝑘superscript21𝑘z=(i\_{k}+z\_{k})2^{1-k}.
Then

|  |  |  |
| --- | --- | --- |
|  | fk​(z)={2​zkwhen 0≤zk≤1/2,2​(1−zk)when 1/2<zk<1.superscript𝑓𝑘𝑧cases2subscript𝑧𝑘when 0≤zk≤1/221subscript𝑧𝑘when 1/2<zk<1f^{k}(z)=\begin{cases}2z\_{k}&\textup{when $0\leq z\_{k}\leq 1/2$},\\ 2(1-z\_{k})&\textup{when $1/2<z\_{k}<1$}.\end{cases} |  |

### 3.4 Proof of [Theorem 1.1](#S1.Thmtheorem1 "Theorem 1.1. ‣ 1.1 Main result ‣ 1 Setting and main results ‣ Benefits of depth in neural networks")

The proof of [Theorem 1.1](#S1.Thmtheorem1 "Theorem 1.1. ‣ 1.1 Main result ‣ 1 Setting and main results ‣ Benefits of depth in neural networks") now follows:
[Lemma 3.17](#S3.Thmtheorem17 "Lemma 3.17. ‣ 3.3 Many layers, many oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks") shows that a many-layered ReLU network can give rise to
a highly oscillatory and regular function fksuperscript𝑓𝑘f^{k},
[Lemma 3.3](#S3.Thmtheorem3 "Lemma 3.3. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks") shows that few-layered networks and (boosted) decision trees
give rise to functions with few oscillations,
and lastly [Lemma 3.1](#S3.Thmtheorem1 "Lemma 3.1. ‣ 3.1 Approximation via oscillation counting ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks") shows how to combine these into an inapproximability result.

In this last piece, the proof averages over the possible offsets y∈ℝd−1𝑦superscriptℝ𝑑1y\in\mathbb{R}^{d-1} and considers univariate
problems after composing networks with the affine map hy​(z):=(z,y)assignsubscriptℎ𝑦𝑧𝑧𝑦h\_{y}(z):=(z,y). In this way, the result
carries some resemblance to the random projection technique used in depth hierarchy theorems
for boolean functions (Håstad, [1986](#bib.bib8); Rossman et al., [2015](#bib.bib17)),
as well as earlier techniques on complexities of multivariate sets (Vitushkin, [1955](#bib.bib21), [1959](#bib.bib22)),
albeit in an extremely primitive form (considering variations along only one dimension).

###### Proof 3.18.

(of [Theorem 1.1](#S1.Thmtheorem1 "Theorem 1.1. ‣ 1.1 Main result ‣ 1 Setting and main results ‣ Benefits of depth in neural networks"))
Set h​(z):=σr​(2​σr​(z)−4​σr​(z−1/2))assignhzsubscriptσr2subscriptσrz4subscriptσrz12h(z):=\sigma\_{\textsc{r}}(2\sigma\_{\textsc{r}}(z)-4\sigma\_{\textsc{r}}(z-1/2)) (cf. [Lemma 3.16](#S3.Thmtheorem16 "Lemma 3.16. ‣ 3.3 Many layers, many oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks")),
and define f0​(z):=hk3+4​(z)assignsubscriptf0zsuperscripthsuperscriptk34zf\_{0}(z):=h^{k^{3}+4}(z) and f:ℝd→ℝ:f→superscriptℝdℝf:\mathbb{R}^{d}\to\mathbb{R} as f​(x)=f0​(x1)fxsubscriptf0subscriptx1f(x)=f\_{0}(x\_{1}).
Let ℐfsubscriptℐf\mathcal{I}\_{f} denote the pieces of f~0subscript~f0\tilde{f}\_{0},
meaning |ℐf|=Cr​(f0)subscriptℐfCrsubscriptf0|\mathcal{I}\_{f}|=\textup{Cr}(f\_{0}),
and [Corollary 3.14](#S3.Thmtheorem14 "Corollary 3.14. ‣ 3.3 Many layers, many oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks") grants Cr​(f0)=2k3+4+1Crsubscriptf0superscript2superscriptk341\textup{Cr}(f\_{0})=2^{k^{3}+4}+1.
Moreover, by [Lemma 3.17](#S3.Thmtheorem17 "Lemma 3.17. ‣ 3.3 Many layers, many oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks"),
for any U∈ℐfUsubscriptℐfU\in\mathcal{I}\_{f}, f0−1/2subscriptf012f\_{0}-1/2 is a triangle with height 1/2
and base either 2−k−1superscript2k12^{-k-1} (when 0∈U0U0\in U or 1∈U1U1\in U)
or 2−ksuperscript2k2^{-k}, whereby
∫U|f0​(x)−1/2|​dx≥2−k−1/4≥|ℐf|/16subscriptUsubscriptf0x12differential-dxsuperscript2k14subscriptℐf16\int\_{U}|f\_{0}(x)-1/2|dx\geq 2^{-k-1}/4\geq|\mathcal{I}\_{f}|/16
(which has thus made use of the special regularity of hhh).

Now for any y∈ℝd−1𝑦superscriptℝ𝑑1y\in\mathbb{R}^{d-1} define the map py:ℝ→ℝd:subscript𝑝𝑦→ℝsuperscriptℝ𝑑p\_{y}:\mathbb{R}\to\mathbb{R}^{d} as py​(z):=(z,y)assignsubscript𝑝𝑦𝑧𝑧𝑦p\_{y}(z):=(z,y).
If g𝑔g is a semi-algebraic network with ≤kabsent𝑘\leq k layers and m≤2k/(t​α​β)𝑚superscript2𝑘𝑡𝛼𝛽m\leq 2^{k}/(t\alpha\beta) total nodes,
then [Lemma 3.3](#S3.Thmtheorem3 "Lemma 3.3. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks") grants
Cr​(g∘py)≤2​(2​t​m​α/k)k​βk2≤4​(t​m​α​β)k2≤2k3+2Cr𝑔subscript𝑝𝑦2superscript2𝑡𝑚𝛼𝑘𝑘superscript𝛽superscript𝑘24superscript𝑡𝑚𝛼𝛽superscript𝑘2superscript2superscript𝑘32\textup{Cr}(g\circ p\_{y})\leq 2(2tm\alpha/k)^{k}\beta^{k^{2}}\leq 4(tm\alpha\beta)^{k^{2}}\leq 2^{k^{3}+2}.
Otherwise, g𝑔g is (t,2k3/t)𝑡superscript2superscript𝑘3𝑡(t,2^{k^{3}}/t)-bdt,
whereby [Lemma 3.3](#S3.Thmtheorem3 "Lemma 3.3. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks") gives Cr​(g∘py)≤2​t​2k3/t≤2k3+2Cr𝑔subscript𝑝𝑦2𝑡superscript2superscript𝑘3𝑡superscript2superscript𝑘32\textup{Cr}(g\circ p\_{y})\leq 2t2^{k^{3}}/t\leq 2^{k^{3}+2} once again.

By [Lemma 3.1](#S3.Thmtheorem1 "Lemma 3.1. ‣ 3.1 Approximation via oscillation counting ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks"), for any y∈ℝd−1𝑦superscriptℝ𝑑1y\in\mathbb{R}^{d-1}, Cr​(f∘py)=Cr​(f0)Cr𝑓subscript𝑝𝑦Crsubscript𝑓0\textup{Cr}(f\circ p\_{y})=\textup{Cr}(f\_{0}), and

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∫[0,1]|f​(py​(z))−g​(py​(z))|​𝑑zsubscript01𝑓subscript𝑝𝑦𝑧𝑔subscript𝑝𝑦𝑧differential-d𝑧\displaystyle\int\_{[0,1]}|f(p\_{y}(z))-g(p\_{y}(z))|dz | =∑U∈ℐf∫U|(f∘py)​(z)−(g∘py)​(z)|​𝑑zabsentsubscript𝑈subscriptℐ𝑓subscript𝑈𝑓subscript𝑝𝑦𝑧𝑔subscript𝑝𝑦𝑧differential-d𝑧\displaystyle=\sum\_{U\in\mathcal{I}\_{f}}\int\_{U}|(f\circ p\_{y})(z)-(g\circ p\_{y})(z)|dz |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≥∑U∈ℐf∫U|(f∘py)​(z)−1/2|​𝟏​[∀z∈U​\centerdot​(f∘py)~​(z)≠(g∘py)~​(z)]​𝑑zabsentsubscript𝑈subscriptℐ𝑓subscript𝑈𝑓subscript𝑝𝑦𝑧121delimited-[]for-all𝑧𝑈\centerdot~𝑓subscript𝑝𝑦𝑧~𝑔subscript𝑝𝑦𝑧differential-d𝑧\displaystyle\geq\sum\_{U\in\mathcal{I}\_{f}}\int\_{U}|(f\circ p\_{y})(z)-1/2|\mathbf{1}[\forall z\in U\centerdot\widetilde{(f\circ p\_{y})}(z)\neq\widetilde{(g\circ p\_{y})}(z)]dz |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≥116​|ℐf|​∑U∈ℐf𝟏​[∀z∈U​\centerdot​(f∘py)~​(z)≠(g∘py)~​(z)]​d​zabsent116subscriptℐ𝑓subscript𝑈subscriptℐ𝑓1delimited-[]for-all𝑧𝑈\centerdot~𝑓subscript𝑝𝑦𝑧~𝑔subscript𝑝𝑦𝑧𝑑𝑧\displaystyle\geq\frac{1}{16|\mathcal{I}\_{f}|}\sum\_{U\in\mathcal{I}\_{f}}\mathbf{1}[\forall z\in U\centerdot\widetilde{(f\circ p\_{y})}(z)\neq\widetilde{(g\circ p\_{y})}(z)]dz |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≥132​(1−2​Cr​(g∘py)Cr​(f∘py))≥132​(1−2​(2k3+2)2k3+4)≥164.absent13212Cr𝑔subscript𝑝𝑦Cr𝑓subscript𝑝𝑦13212superscript2superscript𝑘32superscript2superscript𝑘34164\displaystyle\geq\frac{1}{32}\left(1-\frac{2\textup{Cr}(g\circ p\_{y})}{\textup{Cr}(f\circ p\_{y})}\right)\geq\frac{1}{32}\left(1-\frac{2(2^{k^{3}+2})}{2^{k^{3}+4}}\right)\geq\frac{1}{64}. |  |

To finish,

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∫[0,1]d|f​(x)−g​(x)|​𝑑xsubscriptsuperscript01𝑑𝑓𝑥𝑔𝑥differential-d𝑥\displaystyle\int\_{[0,1]^{d}}|f(x)-g(x)|dx | =∫[0,1]d−1∫[0,1]|(f∘py)​(z)−(g∘py)​(z)|​𝑑z​𝑑y≥164.absentsubscriptsuperscript01𝑑1subscript01𝑓subscript𝑝𝑦𝑧𝑔subscript𝑝𝑦𝑧differential-d𝑧differential-d𝑦164\displaystyle=\int\_{[0,1]^{d-1}}\int\_{[0,1]}|(f\circ p\_{y})(z)-(g\circ p\_{y})(z)|dzdy\geq\frac{1}{64}. |  |

Using nearly the same proof, but giving up on continuous uniform measure, it is possible
to handle other distances and more flexible target functions.

###### Theorem 3.19.

Let integer k≥1𝑘1k\geq 1
and function f:ℝ→ℝ:𝑓→ℝℝf:\mathbb{R}\to\mathbb{R} be given where f𝑓f is (1,[0,1])101(1,[0,1])-triangle,
and define h:ℝd→ℝ:ℎ→superscriptℝ𝑑ℝh:\mathbb{R}^{d}\to\mathbb{R} as h​(x):=fk​(x1)assignℎ𝑥superscript𝑓𝑘subscript𝑥1h(x):=f^{k}(x\_{1}).
For every y∈ℝd−1𝑦superscriptℝ𝑑1y\in\mathbb{R}^{d-1}, define the affine function py​(z):=(z,y)assignsubscript𝑝𝑦𝑧𝑧𝑦p\_{y}(z):=(z,y).
Then there exist Borel probability measures μ𝜇\mu and ν𝜈\nu over [0,1]dsuperscript01𝑑[0,1]^{d}
where ν𝜈\nu is discrete uniform on 2k+1superscript2𝑘12^{k}+1 points and μ𝜇\mu is continuous and positive on exactly [0,1]dsuperscript01𝑑[0,1]^{d}
so that
every g:ℝd→ℝ:𝑔→superscriptℝ𝑑ℝg:\mathbb{R}^{d}\to\mathbb{R} with
Cr​(g∘py)≤2k−2Cr𝑔subscript𝑝𝑦superscript2𝑘2\textup{Cr}(g\circ p\_{y})\leq 2^{k-2} for every y∈ℝd−1𝑦superscriptℝ𝑑1y\in\mathbb{R}^{d-1}
satisfies

|  |  |  |
| --- | --- | --- |
|  | ∫|h−g|​𝑑μ≥132,∫|h~−g~|​𝑑μ≥18,∫|h−g|​𝑑ν≥18,∫|h~−g~|​𝑑ν≥14.formulae-sequenceℎ𝑔differential-d𝜇132formulae-sequence~ℎ~𝑔differential-d𝜇18formulae-sequenceℎ𝑔differential-d𝜈18~ℎ~𝑔differential-d𝜈14\displaystyle\int|h-g|d\mu\geq\frac{1}{32},\qquad\int|\tilde{h}-\tilde{g}|d\mu\geq\frac{1}{8},\qquad\int|h-g|d\nu\geq\frac{1}{8},\qquad\int|\tilde{h}-\tilde{g}|d\nu\geq\frac{1}{4}. |  |

## 4 Limitations of depth

[Theorem 3.19](#S3.Thmtheorem19 "Theorem 3.19. ‣ 3.4 Proof of Theorem 1.1 ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks") can be taken to say: there exists a labeling of Θ​(2k3)Θsuperscript2superscript𝑘3\Theta(2^{k^{3}}) points
which is realizable by a network of depth and size Θ​(k3)Θsuperscript𝑘3\Theta(k^{3}),
but can not be approximated by networks with depth k𝑘k and size o​(2k)𝑜superscript2𝑘o(2^{k}).
On the other hand, this section will sketch the proof of [Theorem 1.2](#S1.Thmtheorem2 "Theorem 1.2. ‣ 1.2 Companion results ‣ 1 Setting and main results ‣ Benefits of depth in neural networks"),
which implies that these Θ​(k3)Θsuperscript𝑘3\Theta(k^{3}) depth networks realize relatively few different
labellings.
The proof is a quick consequence of the VC dimension of semi-algebraic
networks (cf. [Lemma 1.3](#S1.Thmtheorem3 "Lemma 1.3 (Simplification of Lemma 4.2). ‣ 1.2 Companion results ‣ 1 Setting and main results ‣ Benefits of depth in neural networks")) and the following fact,
where Sh​(⋅)Sh⋅\textup{Sh}(\cdot) is used to denote the *growth function*
(Anthony and Bartlett, [1999](#bib.bib1), Chapter 3).

###### Lemma 4.1.

Let any function class ℱℱ\mathcal{F}
and any distinct points (xi)i=1nsuperscriptsubscriptsubscript𝑥𝑖𝑖1𝑛(x\_{i})\_{i=1}^{n} be given.
Then with probability at least 1−δ1𝛿1-\delta over a uniform random draw of labels
(yi)i=1nsuperscriptsubscriptsubscript𝑦𝑖𝑖1𝑛(y\_{i})\_{i=1}^{n} (with yi∈{−1,+1}subscript𝑦𝑖11y\_{i}\in\{-1,+1\}),

|  |  |  |
| --- | --- | --- |
|  | inff∈ℱ1n​∑i=1n𝟏​[f~​(xi)≠yi]≥12​(1−ln⁡(Sh​(ℱ;n))+ln⁡(1/δ)2​n).subscriptinfimum𝑓ℱ1𝑛superscriptsubscript𝑖1𝑛1delimited-[]~𝑓subscript𝑥𝑖subscript𝑦𝑖121Sh  ℱ𝑛1𝛿2𝑛\inf\_{f\in\mathcal{F}}\frac{1}{n}\sum\_{i=1}^{n}\mathbf{1}[\tilde{f}(x\_{i})\neq y\_{i}]\geq\frac{1}{2}\left(1-\sqrt{\frac{\ln(\textup{Sh}(\mathcal{F};n))+\ln(1/\delta)}{2n}}\right). |  |

The proof of the preceding result is similar to proofs of the Gilbert-Varshamov packing bound via Hoeffding’s inequality
(Duchi, [2016](#bib.bib5), Lemma 13.5). Note that a
similar result was used by [Warren](#bib.bib23) to prove rates of approximation of continuous functions by polynomials,
but without invoking Hoeffding’s inequality (Warren, [1968](#bib.bib23), Theorem 7).

The remaining task is to control the VC dimension of semi-algebraic networks. To this end, note
the following generalization of [Lemma 1.3](#S1.Thmtheorem3 "Lemma 1.3 (Simplification of Lemma 4.2). ‣ 1.2 Companion results ‣ 1 Setting and main results ‣ Benefits of depth in neural networks"), which further provides that semi-algebraic networks
compute functions which are polynomial when restricted to certain polynomial regions.

###### Lemma 4.2.

Let neural network graph 𝔊𝔊\mathfrak{G} be given with ≤pabsent𝑝\leq p parameters, ≤labsent𝑙\leq l layers, and ≤mabsent𝑚\leq m total nodes,
and suppose every gate is (t,α,β)𝑡𝛼𝛽(t,\alpha,\beta)-sa. Then

|  |  |  |
| --- | --- | --- |
|  | VC​(𝒩​(𝔊))≤6​p​(l+1)​(ln⁡(2​p​(l+1))+ln⁡(8​e​m​t​α)+l​ln⁡(β)).VC𝒩𝔊6𝑝𝑙12𝑝𝑙18𝑒𝑚𝑡𝛼𝑙𝛽\textup{VC}(\mathcal{N}(\mathfrak{G}))\leq 6p(l+1)\big{(}\ln(2p(l+1))+\ln(8emt\alpha)+l\ln(\beta)\big{)}. |  |

Additionally, given any n≥p𝑛𝑝n\geq p data points, there exists a partition 𝒮𝒮\mathcal{S} of
ℝpsuperscriptℝ𝑝\mathbb{R}^{p} where each S∈𝒮𝑆𝒮S\in\mathcal{S} is an intersection of predicates 𝟏​[q⋄0]1delimited-[]⋄𝑞0\mathbf{1}[q\diamond 0] with ⋄∈{<,≥}\diamond\in\{<,\geq\} and
q𝑞q has degree ≤α​βl−1absent𝛼superscript𝛽𝑙1\leq\alpha\beta^{l-1}, such that F𝔊​(xi,⋅)subscript𝐹𝔊subscript𝑥𝑖⋅F\_{\mathfrak{G}}(x\_{i},\cdot) restricted to each S∈𝒮𝑆𝒮S\in\mathcal{S} is a fixed polynomial of degree ≤βlabsentsuperscript𝛽𝑙\leq\beta^{l}
for every example xisubscript𝑥𝑖x\_{i},
with
|𝒮|≤(8​e​n​m​t​α​βl)p​l𝒮superscript8𝑒𝑛𝑚𝑡𝛼superscript𝛽𝑙𝑝𝑙|\mathcal{S}|\leq\left(8enmt\alpha\beta^{l}\right)^{pl}
and
Sh​(𝒩​(𝔊);n)≤(8​e​n​m​t​α​βl)p​(l+1)Sh

𝒩𝔊𝑛superscript8𝑒𝑛𝑚𝑡𝛼superscript𝛽𝑙𝑝𝑙1\textup{Sh}(\mathcal{N}(\mathfrak{G});n)\leq\left(8enmt\alpha\beta^{l}\right)^{p(l+1)}

The proof follows the same basic structure of the VC bound for networks with piecewise polynomial
activation functions (Anthony and Bartlett, [1999](#bib.bib1), Theorem 8.8). The slightly modified proof here is also very
similar to the proof of [Lemma 3.3](#S3.Thmtheorem3 "Lemma 3.3. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks"), performing an induction up through the layers of the network,
arguing that each node computes a polynomial after restricting attention to some range of parameters.
The proof of [Lemma 4.2](#S4.Thmtheorem2 "Lemma 4.2. ‣ 4 Limitations of depth ‣ Benefits of depth in neural networks") manages to be multivariate (unlike [Lemma 3.3](#S3.Thmtheorem3 "Lemma 3.3. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks")), though this requires
arguments due to Warren ([1968](#bib.bib23)) which are significantly more complicated than those of [Lemma 3.3](#S3.Thmtheorem3 "Lemma 3.3. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks")
(without leading to a strengthening of [Theorem 1.1](#S1.Thmtheorem1 "Theorem 1.1. ‣ 1.1 Main result ‣ 1 Setting and main results ‣ Benefits of depth in neural networks")).

One minor departure from the VC dimension proof of piecewise polynomial networks (cf. (Anthony and Bartlett, [1999](#bib.bib1), Theorem 8.8))
is the following [lemma](#S4.Thmtheorem3 "Lemma 4.3. ‣ 4 Limitations of depth ‣ Benefits of depth in neural networks"), which is used to track the number of regions
with the more complicated semi-algebraic networks.
Despite this generalization, the VC dimension bound is basically the same as for piecewise polynomial networks.

###### Lemma 4.3.

Let a set of polynomials 𝒬𝒬\mathcal{Q} be given where each 𝒬∋q:ℝp→ℝ:𝑞𝒬→superscriptℝ𝑝ℝ\mathcal{Q}\ni q:\mathbb{R}^{p}\to\mathbb{R}
has degree ≤αabsent𝛼\leq\alpha.
Define an initial family 𝒮0subscript𝒮0\mathcal{S}\_{0} of subsets of ℝpsuperscriptℝ𝑝\mathbb{R}^{p} as
𝒮0:={{a∈ℝp:q(a)⋄0}:q∈𝒬,⋄∈{<,≥}}.\mathcal{S}\_{0}:=\big{\{}\{a\in\mathbb{R}^{p}:q(a)\diamond 0\}\ :\ q\in\mathcal{Q},\diamond\in\{<,\geq\}\big{\}}.
Then the collection 𝒮𝒮\mathcal{S} of all nonempty intersections of elements of 𝒮0subscript𝒮0\mathcal{S}\_{0} satisfies
|𝒮|≤2​(4​e​|𝒬|​αp)p.𝒮2superscript4𝑒𝒬𝛼𝑝𝑝|\mathcal{S}|\leq 2\left(\frac{4e|\mathcal{Q}|\alpha}{p}\right)^{p}.

## 5 Bibliographic notes and open problems

Arguably the first approximation theorem of a big class by a smaller one is the Weierstrass Approximation Theorem,
which states that polynomials uniformly approximate continuous functions over compact sets (Weierstrass, [1885](#bib.bib24)).
Refining this,
Kolmogorov ([1936](#bib.bib10)) gave a bound on how well subspaces of functions can approximate continuous functions,
and Vitushkin ([1955](#bib.bib21), [1959](#bib.bib22)) showed a similar bound for approximation by polynomials in
terms of
the polynomial degrees, dimension, and modulus of continuity of the target function.
Warren ([1968](#bib.bib23)) then gave an alternate proof and generalization of this result, in the process effectively
proving the VC dimension of polynomials
(developing tools still used to prove the VC dimension of neural networks (Anthony and Bartlett, [1999](#bib.bib1), Chapters 7-8)),
and producing an analog to [Theorem 1.2](#S1.Thmtheorem2 "Theorem 1.2. ‣ 1.2 Companion results ‣ 1 Setting and main results ‣ Benefits of depth in neural networks") for polynomials.

The preceding results, however, focused on separating large classes (e.g., continuous functions of bounded modulus) from small classes (polynomials
of bounded degree).
Aiming to refine this, depth hierarchy theorems in circuit complexity separated circuits of a certain depth from circuits of a slightly
smaller depth.
As mentioned in [Section 1](#S1 "1 Setting and main results ‣ Benefits of depth in neural networks"), the seminal result here is due to Håstad ([1986](#bib.bib8)).
For architectures closer to neural networks, *sum-product networks* (summation and product nodes) have been analyzed by
Bengio and Delalleau ([2011](#bib.bib2)) and more recently Martens and Medabalimi ([2015](#bib.bib14)),
and networks of linear threshold functions in 2 and 3 layers by Kane and Williams ([2015](#bib.bib9));
note that both polynomial gates (as in sum-product networks)
and linear threshold gates are semi-algebraic gates.
Most closely to the present
work (excluding (Telgarsky, [2015](#bib.bib20)), which is a vastly simplified account),
Eldan and Shamir ([2015](#bib.bib6)) analyze 2- and 3-layer networks with general activation functions composed with affine mappings,
showing separations which are exponential in the input dimension.
Due to this result and also recent advances in circuit complexity (Rossman et al., [2015](#bib.bib17)),
it is natural to suppose [Theorem 1.1](#S1.Thmtheorem1 "Theorem 1.1. ‣ 1.1 Main result ‣ 1 Setting and main results ‣ Benefits of depth in neural networks") can be strengthened to separating k𝑘k and k+1𝑘1k+1 layer networks when dimension d𝑑d is large;
however, none of the earlier works give a tight sense of the behavior as d↓1↓𝑑1d\downarrow 1.

The triangle wave target functions considered here (e.g., cf. [Lemma 3.16](#S3.Thmtheorem16 "Lemma 3.16. ‣ 3.3 Many layers, many oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks")) have appeared in various forms throughout the literature.
General properties of piecewise affine highly oscillating functions were investigated by
Szymanski and McCane ([2014](#bib.bib19)) and Montúfar et al. ([2014](#bib.bib15)).
Also, Schmitt ([2000](#bib.bib18)) investigated the map z↦4​z​(1−z)maps-to𝑧4𝑧1𝑧z\mapsto 4z(1-z) (as in [Lemma 3.16](#S3.Thmtheorem16 "Lemma 3.16. ‣ 3.3 Many layers, many oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks")) to show that sigmoidal networks
can not approximate high degree polynomials via an analysis similar to the one here,
however looseness in the VC bounds for sigmoidal networks prevented exponential separations and depth hierarchies.

A tantalizing direction for future work is to characterize not just one difficult function (e.g., triangle functions as in [Lemma 3.16](#S3.Thmtheorem16 "Lemma 3.16. ‣ 3.3 Many layers, many oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks")),
but many, or even all functions which are not well-approximated by smaller depths. Arguably, this direction could have
value in machine learning, as discovery of such underlying structure could lead to algorithms to recover it.
As a trivial example of the sort of structure which could arise,
considering the following [proposition](#S5.Thmtheorem1 "Proposition 5.1. ‣ 5 Bibliographic notes and open problems ‣ Benefits of depth in neural networks"),
stating that any symmetric signal may be repeated by pre-composing it with the ReLU triangle function.

###### Proposition 5.1.

Set f​(z):=σr​(2​σr​(z)−4​σr​(z−1/2))assign𝑓𝑧subscript𝜎r2subscript𝜎r𝑧4subscript𝜎r𝑧12f(z):=\sigma\_{\textsc{r}}(2\sigma\_{\textsc{r}}(z)-4\sigma\_{\textsc{r}}(z-1/2)) (cf. [Lemma 3.16](#S3.Thmtheorem16 "Lemma 3.16. ‣ 3.3 Many layers, many oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks")),
and let any g:[0,1]→[0,1]:𝑔→0101g:[0,1]\to[0,1] be given with g​(z)=g​(1−z)𝑔𝑧𝑔1𝑧g(z)=g(1-z).
Then h:=g∘fkassignℎ𝑔superscript𝑓𝑘h:=g\circ f^{k} satisfies h​(x)=h​(x+i​2k)=g​(x​2k)ℎ𝑥ℎ𝑥𝑖superscript2𝑘𝑔𝑥superscript2𝑘h(x)=h(x+i2^{k})=g(x2^{k}) for every real x∈[0,2−k]𝑥0superscript2𝑘x\in[0,2^{-k}] and integer i∈{0,…,2−k−1}𝑖0…superscript2𝑘1i\in\{0,\ldots,2^{-k}-1\};
in other words, hℎh is 2ksuperscript2𝑘2^{k} repetitions of g𝑔g with graph scaled horizontally and uniformly to fit within [0,1]2superscript012[0,1]^{2}.

\acks

The author is indebted to Joshua Zahl for help navigating semi-algebraic geometry and for a simplification of
the multivariate case in [Theorem 1.1](#S1.Thmtheorem1 "Theorem 1.1. ‣ 1.1 Main result ‣ 1 Setting and main results ‣ Benefits of depth in neural networks"),
and to
Rastislav Telgársky
for an introduction to this general topic via Kolmogorov’s Superposition Theorem (Kolmogorov, [1957](#bib.bib11)).
The author further thanks
Jacob Abernethy,
Peter Bartlett,
Sébastien Bubeck,
and Alex Kulesza
for valuable discussions.

## References

* Anthony and Bartlett (1999)

  Martin Anthony and Peter L. Bartlett.
  *Neural Network Learning: Theoretical Foundations*.
  Cambridge University Press, 1999.
* Bengio and Delalleau (2011)

  Yoshua Bengio and Olivier Delalleau.
  Shallow vs. deep sum-product networks.
  In *NIPS*, 2011.
* Bochnak et al. (1998)

  Jacek Bochnak, Michal Coste, and Marie-Françoise Roy.
  *Real Algebraic Geometry*.
  Springer, 1998.
* Caruana and Niculescu-Mizil (2006)

  Rich Caruana and Alexandru Niculescu-Mizil.
  An empirical comparison of supervised learning algorithms.
  pages 161–168, 2006.
* Duchi (2016)

  John Duchi.
  Statistics 311/electrical engineering 377: Information theory and
  statistics.
  Stanford University, 2016.
* Eldan and Shamir (2015)

  Ronen Eldan and Ohad Shamir.
  The power of depth for feedforward neural networks.
  2015.
  arXiv:1512.03965 [cs.LG].
* Fukushima (1980)

  Kunihiko Fukushima.
  Neocognitron: A self-organizing neural network model for a mechanism
  of pattern recognition unaffected by shift in position.
  *Biological Cybernetics*, 36:193–202, 1980.
* Håstad (1986)

  Johan Håstad.
  *Computational Limitations of Small Depth Circuits*.
  PhD thesis, Massachusetts Institute of Technology, 1986.
* Kane and Williams (2015)

  Daniel Kane and Ryan Williams.
  Super-linear gate and super-quadratic wire lower bounds for depth-two
  and depth-three threshold circuits.
  2015.
  arXiv:1511.07860v1 [cs.CC].
* Kolmogorov (1936)

  Andrei Kolmogorov.
  Über die beste annäherung von funktionen einer gegebenen
  funktionenklasse.
  *Annals of Mathematics*, 37(1):107–110,
  1936.
* Kolmogorov (1957)

  Andrey Nikolaevich Kolmogorov.
  On the representation of continuous functions of several variables by
  superpositions of continuous functions of one variable and addition.
  114:953–956, 1957.
* Krizhevsky et al. (2012)

  Alex Krizhevsky, Ilya Sutskever, and Geoffery Hinton.
  Imagenet classification with deep convolutional neural networks.
  In *NIPS*, 2012.
* LeCun et al. (1998)

  Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner.
  Gradient-based learning applied to document recognition.
  *Proceedings of the IEEE*, 86(11):2278–2324, 1998.
* Martens and Medabalimi (2015)

  James Martens and Venkatesh Medabalimi.
  On the expressive efficiency of sum product networks.
  2015.
  arXiv:1411.7717v3 [cs.LG].
* Montúfar et al. (2014)

  Guido Montúfar, Razvan Pascanu, Kyunghyun Cho, and Yoshua Bengio.
  On the number of linear regions of deep neural networks.
  In *NIPS*, 2014.
* Poon and Domingos (2011)

  Hoifung Poon and Pedro M. Domingos.
  Sum-product networks: A new deep architecture.
  In *UAI 2011*, pages 337–346, 2011.
* Rossman et al. (2015)

  Benjamin Rossman, Rocco A. Servedio, and Li-Yang Tan.
  An average-case depth hierarchy theorem for boolean circuits.
  In *FOCS*, 2015.
* Schmitt (2000)

  Michael Schmitt.
  Lower bounds on the complexity of approximating continuous functions
  by sigmoidal neural networks.
  In *NIPS*, 2000.
* Szymanski and McCane (2014)

  Lech Szymanski and Brendan McCane.
  Deep networks are effective encoders of periodicity.
  *IEEE Transactions on Neural Networks and Learning Systems*,
  25(10):1816–1827, 2014.
* Telgarsky (2015)

  Matus Telgarsky.
  Representation benefits of deep feedforward networks.
  2015.
  arXiv:1509.08101v2 [cs.LG].
* Vitushkin (1955)

  Anatoli Vitushkin.
  On multidimensional variations.
  *GITTL*, 1955.
  In Russian.
* Vitushkin (1959)

  Anatoli Vitushkin.
  Estimation of the complexity of the tabulation problem.
  *Fizmatgiz.*, 1959.
  In Russian.
* Warren (1968)

  Hugh E. Warren.
  Lower bounds for approximation by nonlinear manifolds.
  *Transactions of the American Mathematical Society*,
  133(1):167–178, 1968.
* Weierstrass (1885)

  Karl Weierstrass.
  Über die analytische darstellbarkeit sogenannter
  willkürlicher functionen einer reellen veränderlichen.
  *Sitzungsberichte der Akademie zu Berlin*, pages 633–639,
  789–805, 1885.

## Appendix A Deferred proofs

This appendix collects various proofs omitted from the main text.

### A.1 Deferred proofs from [Section 2](#S2 "2 Semi-algebraic gates and assorted network notation ‣ Benefits of depth in neural networks")

The following mechanical proof shows that standard piecewise polynomial gates,
maximization/minimization gates, and decision trees are all semi-algebraic gates.

###### Proof A.1.

(of [Lemma 2.3](#S2.Thmtheorem3 "Lemma 2.3 (Example semi-algebraic gates). ‣ 2 Semi-algebraic gates and assorted network notation ‣ Benefits of depth in neural networks"))

1. 1.

   To start, since σ:ℝ→ℝ:𝜎→ℝℝ\sigma:\mathbb{R}\to\mathbb{R} is piecewise polynomial,
   σ∘q𝜎𝑞\sigma\circ q can be written

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | σ​(q​(z))𝜎𝑞𝑧\displaystyle\sigma(q(z)) | :=p1(q(z))𝟏[q(z)⋄1b1]+∑i=2t−1pi(q(z))𝟏[−q(z)∗i−1−bi−1]𝟏[q(z)⋄ibi]\displaystyle:=p\_{1}(q(z))\mathbf{1}[q(z)\diamond\_{1}b\_{1}]+\sum\_{i=2}^{t-1}p\_{i}(q(z))\mathbf{1}[-q(z)\ast\_{i-1}-b\_{i-1}]\mathbf{1}[q(z)\diamond\_{i}b\_{i}] |  |
   |  |  |  |  |
   | --- | --- | --- | --- |
   |  |  | +pt(q(z))𝟏[−q(z)∗t−bt]\displaystyle\qquad+p\_{t}(q(z))\mathbf{1}[-q(z)\ast\_{t}-b\_{t}] |  |

   where for each i∈[t]𝑖delimited-[]𝑡i\in[t],
   pisubscript𝑝𝑖p\_{i} has degree ≤βabsent𝛽\leq\beta,
   ⋄i∈{<,≤}\diamond\_{i}\in\{<,\leq\},
   ∗i=``<"\ast\_{i}=``<" when ⋄i=``≤"\diamond\_{i}=``\leq" and otherwise ∗i=``≤"\ast\_{i}=``\leq",
   and bi∈ℝsubscript𝑏𝑖ℝb\_{i}\in\mathbb{R}.
   As such, setting qi​(z):=q​(z)−b1assignsubscript𝑞𝑖𝑧𝑞𝑧subscript𝑏1q\_{i}(z):=q(z)-b\_{1} whenever ⋄i=``<"\diamond\_{i}=``<" and
   qi​(z):=bi−q​(z)assignsubscript𝑞𝑖𝑧subscript𝑏𝑖𝑞𝑧q\_{i}(z):=b\_{i}-q(z) otherwise,
   it follows that σ∘q𝜎𝑞\sigma\circ q is (t,α,α​β)𝑡𝛼𝛼𝛽(t,\alpha,\alpha\beta)-sa.
2. 2.

   Since mini∈[r]⁡xi=−maxi∈[r]−xisubscript𝑖delimited-[]𝑟subscript𝑥𝑖subscript𝑖delimited-[]𝑟subscript𝑥𝑖\min\_{i\in[r]}x\_{i}=-\max\_{i\in[r]}-x\_{i}, it suffices to handle the maximum case,
   which has the form

   |  |  |  |
   | --- | --- | --- |
   |  | ϕmax​(v)=∑i=1dpi​(v)​(∏j<i𝟏​[pi​(v)>pj​(v)])​(∏j>i𝟏​[pi​(v)≥pj​(v)]).subscriptitalic-ϕ𝑣superscriptsubscript𝑖1𝑑subscript𝑝𝑖𝑣subscriptproduct𝑗𝑖1delimited-[]subscript𝑝𝑖𝑣subscript𝑝𝑗𝑣subscriptproduct𝑗𝑖1delimited-[]subscript𝑝𝑖𝑣subscript𝑝𝑗𝑣\displaystyle\phi\_{\max}(v)=\sum\_{i=1}^{d}p\_{i}(v)\left(\prod\_{j<i}\mathbf{1}[p\_{i}(v)>p\_{j}(v)]\right)\left(\prod\_{j>i}\mathbf{1}[p\_{i}(v)\geq p\_{j}(v)]\right). |  |

   Constructing polynomials qi,j=pj−pisubscript𝑞
   𝑖𝑗subscript𝑝𝑗subscript𝑝𝑖q\_{i,j}=p\_{j}-p\_{i} when j<i𝑗𝑖j<i and qi,j=pi−pjsubscript𝑞
   𝑖𝑗subscript𝑝𝑖subscript𝑝𝑗q\_{i,j}=p\_{i}-p\_{j} when j>i𝑗𝑖j>i,
   it follows that ϕmaxsubscriptitalic-ϕ\phi\_{\max} is (r​(r−1),α,α)𝑟𝑟1𝛼𝛼(r(r-1),\alpha,\alpha)-sa.
3. 3.

   First consider a k𝑘k-dt f𝑓f, wherein the proof follows by induction on tree size.
   In the base case k=1𝑘1k=1, f𝑓f is constant.
   Otherwise, there exist functions flsubscript𝑓𝑙f\_{l} and frsubscript𝑓𝑟f\_{r} which are respectively l𝑙l- and r𝑟r-dt
   with l+r<k𝑙𝑟𝑘l+r<k, and additionally an affine function qfsubscript𝑞𝑓q\_{f} so that

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | f​(x)𝑓𝑥\displaystyle f(x) | =fl​(x)​𝟏​[qf​(x)<0]+fr​(x)​𝟏​[qf​(x)≥0]absentsubscript𝑓𝑙𝑥1delimited-[]subscript𝑞𝑓𝑥0subscript𝑓𝑟𝑥1delimited-[]subscript𝑞𝑓𝑥0\displaystyle=f\_{l}(x)\mathbf{1}[q\_{f}(x)<0]+f\_{r}(x)\mathbf{1}[q\_{f}(x)\geq 0] |  |
   |  |  |  |  |
   | --- | --- | --- | --- |
   |  |  | =∑j=1mlpj(l)​(v)​𝟏​[qf​(x)<0]​(∏i∈Lj(l)𝟏​[qi(l)​(v)<0])​(∏i∈Uj(l)𝟏​[qi(l)​(v)≥0])absentsuperscriptsubscript𝑗1subscript𝑚𝑙superscriptsubscript𝑝𝑗𝑙𝑣1delimited-[]subscript𝑞𝑓𝑥0subscriptproduct𝑖superscriptsubscript𝐿𝑗𝑙1delimited-[]superscriptsubscript𝑞𝑖𝑙𝑣0subscriptproduct𝑖superscriptsubscript𝑈𝑗𝑙1delimited-[]superscriptsubscript𝑞𝑖𝑙𝑣0\displaystyle=\sum\_{j=1}^{m\_{l}}p\_{j}^{(l)}(v)\mathbf{1}[q\_{f}(x)<0]\left(\prod\_{i\in L\_{j}^{(l)}}\mathbf{1}[q\_{i}^{(l)}(v)<0]\right)\left(\prod\_{i\in U\_{j}^{(l)}}\mathbf{1}[q\_{i}^{(l)}(v)\geq 0]\right) |  |
   |  |  |  |  |
   | --- | --- | --- | --- |
   |  |  | +∑j=1mrpj(r)​(v)​𝟏​[qf​(x)≥0]​(∏i∈Lj(r)𝟏​[qi(r)​(v)<0])​(∏i∈Uj(r)𝟏​[qi(r)​(v)≥0]).superscriptsubscript𝑗1subscript𝑚𝑟superscriptsubscript𝑝𝑗𝑟𝑣1delimited-[]subscript𝑞𝑓𝑥0subscriptproduct𝑖superscriptsubscript𝐿𝑗𝑟1delimited-[]superscriptsubscript𝑞𝑖𝑟𝑣0subscriptproduct𝑖superscriptsubscript𝑈𝑗𝑟1delimited-[]superscriptsubscript𝑞𝑖𝑟𝑣0\displaystyle\qquad+\sum\_{j=1}^{m\_{r}}p\_{j}^{(r)}(v)\mathbf{1}[q\_{f}(x)\geq 0]\left(\prod\_{i\in L\_{j}^{(r)}}\mathbf{1}[q\_{i}^{(r)}(v)<0]\right)\left(\prod\_{i\in U\_{j}^{(r)}}\mathbf{1}[q\_{i}^{(r)}(v)\geq 0]\right). |  |

   where the last step expanded the semi-algebraic forms of flsubscript𝑓𝑙f\_{l} and frsubscript𝑓𝑟f\_{r}.
   As such, by combining the sets of predicate polynomials for flsubscript𝑓𝑙f\_{l} and frsubscript𝑓𝑟f\_{r} together with {qf}subscript𝑞𝑓\{q\_{f}\}
   (where the former two have cardinalities ≤labsent𝑙\leq l and ≤rabsent𝑟\leq r by the inductive hypothesis),
   and unioning together the triples for flsubscript𝑓𝑙f\_{l} and frsubscript𝑓𝑟f\_{r} but extending the triples to
   include 𝟏​[qf<0]1delimited-[]subscript𝑞𝑓0\mathbf{1}[q\_{f}<0] for triples in flsubscript𝑓𝑙f\_{l} and 𝟏​[qf≥0]1delimited-[]subscript𝑞𝑓0\mathbf{1}[q\_{f}\geq 0] for triples in frsubscript𝑓𝑟f\_{r},
   it follows by construction that f𝑓f is (k,1,0)𝑘10(k,1,0)-semi-algebraic.

   Now consider a (t,k)𝑡𝑘(t,k)-bdt g𝑔g. By the preceding expansion, each individual tree fisubscript𝑓𝑖f\_{i}
   is (k,1,0)𝑘10(k,1,0)-sa, thus their sum is (t​k,1,0)𝑡𝑘10(tk,1,0)
   by unioning together the sets of polynomials, triples, and adding together the expansions.

### A.2 Deferred proofs from [Section 3](#S3 "3 Benefits of depth ‣ Benefits of depth in neural networks")

The first proof shows that a collection of partitions may be refined into a single partition
whose size is at most the total number of intervals across all partitions.
As discussed in the text, while the proof has a simple idea (one need only consider boundaries
of intervals across all partitions), it is somewhat painful since there is not consistent rule for
whether specific endpoints endpoints of intervals are open or closed.

###### Proof A.2.

(of [Lemma 3.6](#S3.Thmtheorem6 "Lemma 3.6. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks"))
If k=1k1k=1, then the result follows with B=A=A1BAsubscriptA1B=A=A\_{1} (since all intersections are empty),
thus suppose k≥2k2k\geq 2.
Let {a1,…,aq}subscripta1…subscriptaq\{a\_{1},\ldots,a\_{q}\} denote the set of distinct boundaries of intervals of AAA,
and iteratively construct the partition BBB as follows,
where the construction will maintain that BjsubscriptBjB\_{j} is a partition whose boundary points
are {a1,…​aj}subscripta1…subscriptaj\{a\_{1},\ldots a\_{j}\}.
For the base case, set B0:={ℝ}assignsubscriptB0ℝB\_{0}:=\{\mathbb{R}\}.
Thereafter, for every i∈[q]idelimited-[]qi\in[q], consider boundary point aisubscriptaia\_{i};
since the boundary points are distinct, there must exist a single interval U∈Bi−1UsubscriptBi1U\in B\_{i-1}
with ai∈UsubscriptaiUa\_{i}\in U. BisubscriptBiB\_{i} will be formed from Bi−1subscriptBi1B\_{i-1} by refining UUU in one of the following
two ways.

* •

  Consider the case that each partition Alsubscript𝐴𝑙A\_{l} which contains the boundary point aisubscript𝑎𝑖a\_{i}
  has exactly two intervals meeting at aisubscript𝑎𝑖a\_{i} and moreover the closedness properties are the same,
  meaning either aisubscript𝑎𝑖a\_{i} is contained in the interval which ends at aisubscript𝑎𝑖a\_{i}, or it is
  contained in the interval which starts at aisubscript𝑎𝑖a\_{i}.
  In this case, partition U𝑈U into two intervals so that the treatment
  of the boundary is the same as those Alsubscript𝐴𝑙A\_{l}’s with a boundary at aisubscript𝑎𝑖a\_{i}.
* •

  Otherwise, it is either the case that some Alsubscript𝐴𝑙A\_{l} have aisubscript𝑎𝑖a\_{i} contained in the interval ending at
  aisubscript𝑎𝑖a\_{i} whereas others have it contained in the interval starting at aisubscript𝑎𝑖a\_{i},
  or simply some Alsubscript𝐴𝑙A\_{l} have three intervals meeting at aisubscript𝑎𝑖a\_{i}: namely, the singleton interval
  [al,al]subscript𝑎𝑙subscript𝑎𝑙[a\_{l},a\_{l}] as well as two intervals not containing alsubscript𝑎𝑙a\_{l}.
  In this case, partition U𝑈U into three intervals:
  one ending at aisubscript𝑎𝑖a\_{i} (but not containing it),
  the singleton interval [ai,ai]subscript𝑎𝑖subscript𝑎𝑖[a\_{i},a\_{i}],
  and an interval starting at aisubscript𝑎𝑖a\_{i} (but not containing it).

(These cases may also be described in a unified way: consider all intervals of A𝐴A which have
aisubscript𝑎𝑖a\_{i} as an endpoint, extend such intervals of positive length to have infinite length
while keeping endpoint aisubscript𝑎𝑖a\_{i} and the side it falls on,
and then refine U𝑈U by intersecting it with all of these intervals,
which as above results in either 2 or 3 intervals.)

Note that the construction never introduces more intervals at a boundary point than
exist in A𝐴A, thus |B|≤|A|=k​t𝐵𝐴𝑘𝑡|B|\leq|A|=kt.

It remains to be shown that a union of intersections of elements of A𝐴A is a union of elements of B𝐵B.
Note that it suffices to show that intersections of elements of A𝐴A are unions of elements of B𝐵B,
since thereafter these encodings can be used to express unions of intersections of A𝐴A as unions of B𝐵B.
As such, consider any intersection U𝑈U of elements of A𝐴A; there is nothing to show if U𝑈U is empty,
thus suppose it is nonempty. In this case, it must also be an interval (e.g., since intersections of convex
sets are convex), and its endpoints must coincide with endpoints of A𝐴A.
Moreover, if the left endpoint of U𝑈U is open, then U𝑈U must be formed from an intersection which
includes an interval with the same open left endpoint, thus there exists such an interval in A𝐴A,
and by the above construction of B𝐵B, there also exists an interval with such an open left endpoint in B𝐵B;
the same argument similarly handles the case of closed left endpoints,
as well as open and closed right endpoints,
namely giving elements in B𝐵B which match these traits.
Let arsubscript𝑎𝑟a\_{r} and assubscript𝑎𝑠a\_{s} denote these endpoints.
By the above construction of B𝐵B, intervals with endpoints {aj,aj+1}subscript𝑎𝑗subscript𝑎𝑗1\{a\_{j},a\_{j+1}\}
for j∈{r,…,s−1}𝑗𝑟…𝑠1j\in\{r,\ldots,s-1\} will be included in B𝐵B,
and since B𝐵B is a partition, the union of these elements will
be exactly U𝑈U. Since U𝑈U was an arbitrary intersection of elements of A𝐴A,
the proof is complete.

Next, the tools of [Section 3.2](#S3.SS2 "3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks") (culminating in the composition rule for semi-algebraic gates ([Lemma 3.9](#S3.Thmtheorem9 "Lemma 3.9. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks")))
are used to show crossing number bounds on semi-algebraic networks
and boosted decision trees.

###### Proof A.3.

(of [Lemma 3.3](#S3.Thmtheorem3 "Lemma 3.3. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks"))

1. 1.

   This proof first shows
   f∘h𝑓ℎf\circ h is
   (2i​ti​αi​∏j≤i−1tj​αj​βji−j+1​kj,∏j≤iβj)superscript2𝑖subscript𝑡𝑖subscript𝛼𝑖subscriptproduct𝑗𝑖1subscript𝑡𝑗subscript𝛼𝑗superscriptsubscript𝛽𝑗𝑖𝑗1subscript𝑘𝑗subscriptproduct𝑗𝑖subscript𝛽𝑗(2^{i}t\_{i}\alpha\_{i}\prod\_{j\leq i-1}t\_{j}\alpha\_{j}\beta\_{j}^{i-j+1}k\_{j},\prod\_{j\leq i}\beta\_{j})-poly,
   and then relaxes this expression and applies [Lemma 3.4](#S3.Thmtheorem4 "Lemma 3.4. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks") to obtain the desired bound.

   First consider the case d=1𝑑1d=1 and hℎh is the identity map, thus f∘h=f𝑓ℎ𝑓f\circ h=f.
   For convenience, set

   |  |  |  |
   | --- | --- | --- |
   |  | Ai:=∏j≤iαj,Bi:=∏j≤iβj,Ci:=∏j≤iβji−j+1=∏j≤iBj,Mi:=∏j≤imj,Ti:=∏j≤itj.formulae-sequenceformulae-sequenceassignsubscript𝐴𝑖subscriptproduct𝑗𝑖subscript𝛼𝑗formulae-sequenceassignsubscript𝐵𝑖subscriptproduct𝑗𝑖subscript𝛽𝑗assignsubscript𝐶𝑖subscriptproduct𝑗𝑖superscriptsubscript𝛽𝑗𝑖𝑗1subscriptproduct𝑗𝑖subscript𝐵𝑗formulae-sequenceassignsubscript𝑀𝑖subscriptproduct𝑗𝑖subscript𝑚𝑗assignsubscript𝑇𝑖subscriptproduct𝑗𝑖subscript𝑡𝑗A\_{i}:=\prod\_{j\leq i}\alpha\_{j},\quad B\_{i}:=\prod\_{j\leq i}\beta\_{j},\quad C\_{i}:=\prod\_{j\leq i}\beta\_{j}^{i-j+1}=\prod\_{j\leq i}B\_{j},\quad M\_{i}:=\prod\_{j\leq i}m\_{j},\quad T\_{i}:=\prod\_{j\leq i}t\_{j}. |  |

   The proof proceeds by induction on the layers of f𝑓f, showing that
   each node in layer i𝑖i is
   (2i​Ti​Ai​Ci−1​Mi−1,Bi)superscript2𝑖subscript𝑇𝑖subscript𝐴𝑖subscript𝐶𝑖1subscript𝑀𝑖1subscript𝐵𝑖(2^{i}T\_{i}A\_{i}C\_{i-1}M\_{i-1},B\_{i})-poly.

   For convenience, first consider layer i=0𝑖0i=0 of the inputs themselves:
   here, node i𝑖i outputs the ithsuperscript𝑖thi^{\textup{th}} coordinate of the input,
   and is thus affine and (1,1)11(1,1)-poly.
   Next consider layer i>0𝑖0i>0, where the inductive hypothesis grants
   that each node in layer i−1𝑖1i-1 is
   (2i−1​Ti−1​Ai−1​Ci−2​Mi−2,Bi−1)superscript2𝑖1subscript𝑇𝑖1subscript𝐴𝑖1subscript𝐶𝑖2subscript𝑀𝑖2subscript𝐵𝑖1(2^{i-1}T\_{i-1}A\_{i-1}C\_{i-2}M\_{i-2},B\_{i-1})-poly.
   Consequently, since any node in layer i𝑖i is (ti,αi,βi)subscript𝑡𝑖subscript𝛼𝑖subscript𝛽𝑖(t\_{i},\alpha\_{i},\beta\_{i})-sa,
   [Lemma 3.9](#S3.Thmtheorem9 "Lemma 3.9. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks") grants it is also
   (2i−1​ti​Ti−1​Ai−1​Ci−2​Mi−2​mi−1​(1+αi​Bi−1),βi​Bi−1)superscript2𝑖1subscript𝑡𝑖subscript𝑇𝑖1subscript𝐴𝑖1subscript𝐶𝑖2subscript𝑀𝑖2subscript𝑚𝑖11subscript𝛼𝑖subscript𝐵𝑖1subscript𝛽𝑖subscript𝐵𝑖1(2^{i-1}t\_{i}T\_{i-1}A\_{i-1}C\_{i-2}M\_{i-2}m\_{i-1}(1+\alpha\_{i}B\_{i-1}),\beta\_{i}B\_{i-1})-poly
   as desired (since 1+αi​Bi−1≤2​αi​Bi−11subscript𝛼𝑖subscript𝐵𝑖12subscript𝛼𝑖subscript𝐵𝑖11+\alpha\_{i}B\_{i-1}\leq 2\alpha\_{i}B\_{i-1}).

   Next, consider the general case d≥1𝑑1d\geq 1 and h:ℝ→ℝd:ℎ→ℝsuperscriptℝ𝑑h:\mathbb{R}\to\mathbb{R}^{d} is an affine map.
   Since every coordinate of hℎh is affine (and thus (1,1)11(1,1)-poly),
   composing hℎh with every polynomial in the semi-algebraic gates of layer 1
   gives a function g∈𝒩1​((mi,ti,αi,βi)i=1l)𝑔subscript𝒩1superscriptsubscriptsubscript𝑚𝑖subscript𝑡𝑖subscript𝛼𝑖subscript𝛽𝑖𝑖1𝑙g\in\mathcal{N}\_{1}((m\_{i},t\_{i},\alpha\_{i},\beta\_{i})\_{i=1}^{l}) which is equal to f∘h𝑓ℎf\circ h everywhere
   and whose gates are of the same semi-algebraic complexity.
   As such, the result follows by applying the preceding analysis to g𝑔g.

   Lastly, the simplified terms give
   f∘h𝑓ℎf\circ h is ((2​t​α)l​βl​(l−1)/2​∏j≤l−1mj,βl​(l+1)/2)superscript2𝑡𝛼𝑙superscript𝛽𝑙𝑙12subscriptproduct𝑗𝑙1subscript𝑚𝑗superscript𝛽𝑙𝑙12((2t\alpha)^{l}\beta^{l(l-1)/2}\prod\_{j\leq l-1}m\_{j},\beta^{l(l+1)/2})-poly.
   Since ln⁡(⋅)⋅\ln(\cdot) is strictly increasing and concave and ml=1subscript𝑚𝑙1m\_{l}=1,

   |  |  |  |
   | --- | --- | --- |
   |  | ln⁡(∏j≤l−1mj)=ln⁡(∏j≤lmj)=∑j≤lln⁡(mj)≤l​ln⁡(m/l)=ln⁡((m/l)l).subscriptproduct𝑗𝑙1subscript𝑚𝑗subscriptproduct𝑗𝑙subscript𝑚𝑗subscript𝑗𝑙subscript𝑚𝑗𝑙𝑚𝑙superscript𝑚𝑙𝑙\ln\left(\prod\_{j\leq l-1}m\_{j}\right)=\ln\left(\prod\_{j\leq l}m\_{j}\right)=\sum\_{j\leq l}\ln(m\_{j})\leq l\ln(m/l)=\ln((m/l)^{l}). |  |

   It follows that f∘h𝑓ℎf\circ h is
   ((2​t​m​α/l)l​βl​(l−1)/2,βl​(l+1)/2)superscript2𝑡𝑚𝛼𝑙𝑙superscript𝛽𝑙𝑙12superscript𝛽𝑙𝑙12((2tm\alpha/l)^{l}\beta^{l(l-1)/2},\beta^{l(l+1)/2})-poly,
   whereby the crossing number bound follows by [Lemma 3.4](#S3.Thmtheorem4 "Lemma 3.4. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks").
2. 2.

   Given any k𝑘k-dt f𝑓f, the affine function evaluated at each predicate may be composed with hℎh to yield
   another affine function, thus f∘h:ℝ→ℝ:𝑓ℎ→ℝℝf\circ h:\mathbb{R}\to\mathbb{R} is still a k𝑘k-dt,
   and thus (k,1,0)𝑘10(k,1,0)-sa by [Lemma 2.3](#S2.Thmtheorem3 "Lemma 2.3 (Example semi-algebraic gates). ‣ 2 Semi-algebraic gates and assorted network notation ‣ Benefits of depth in neural networks").
   As such, by [Lemma 3.9](#S3.Thmtheorem9 "Lemma 3.9. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks") (with g1​(z)=zsubscript𝑔1𝑧𝑧g\_{1}(z)=z as the identity map),
   f∘h𝑓ℎf\circ h is (k,0)𝑘0(k,0)-poly. (Invoking [Lemma 3.9](#S3.Thmtheorem9 "Lemma 3.9. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks") without massaging in hℎh introduces a factor d𝑑d.)
   Similarly, for a (t,k)𝑡𝑘(t,k)-bdt g𝑔g, g∘h:ℝ→ℝ:𝑔ℎ→ℝℝg\circ h:\mathbb{R}\to\mathbb{R} is another (t,k)𝑡𝑘(t,k)-bdt after pushing hℎh into the
   predicates of the constituent trees, thus [Lemma 2.3](#S2.Thmtheorem3 "Lemma 2.3 (Example semi-algebraic gates). ‣ 2 Semi-algebraic gates and assorted network notation ‣ Benefits of depth in neural networks") grants g∘h𝑔ℎg\circ h is (t​k,1,0)𝑡𝑘10(tk,1,0)-sa,
   and [Lemma 3.9](#S3.Thmtheorem9 "Lemma 3.9. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks") grants it is (t​k​(1+1),0)𝑡𝑘110(tk(1+1),0)-poly.
   The desired crossing number bounds follow by applying [Lemma 3.4](#S3.Thmtheorem4 "Lemma 3.4. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks").

Next, elementary computations verify that the three functions listed in [Lemma 3.16](#S3.Thmtheorem16 "Lemma 3.16. ‣ 3.3 Many layers, many oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks") are indeed (1,[0,1])101(1,[0,1])-triangle.

###### Proof A.4.

(of [Lemma 3.16](#S3.Thmtheorem16 "Lemma 3.16. ‣ 3.3 Many layers, many oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks"))

1. 1-2.

   By inspection, f​(0)=f​(1)=0𝑓0𝑓10f(0)=f(1)=0 and f​(1/2)=1𝑓121f(1/2)=1.
   Moreover, for x∈[0,1/2]𝑥012x\in[0,1/2], f​(x)=2​x𝑓𝑥2𝑥f(x)=2x meaning f𝑓f is increasing,
   and x∈[1/2,1]𝑥121x\in[1/2,1] means f​(x)=2​(1−x)𝑓𝑥21𝑥f(x)=2(1-x), meaning f𝑓f is decreasing.
   Lastly, the properties of g𝑔g follow since f=g𝑓𝑔f=g.
2. 3.

   By inspection, h​(0)=h​(1)=0ℎ0ℎ10h(0)=h(1)=0 and h​(1/2)=1ℎ121h(1/2)=1.
   Moreover hℎh is a quadratic, thus can cross 0 at most twice, and moreover 1/2121/2 is the unique
   critical point (since g′superscript𝑔′g^{\prime} has degree 1), thus g𝑔g is increasing on [0,1/2]012[0,1/2]
   and decreasing on [1/2,1]121[1/2,1].

In the case of the ReLU (1,[0,1])101(1,[0,1])-triangle function f𝑓f given in [Lemma 3.16](#S3.Thmtheorem16 "Lemma 3.16. ‣ 3.3 Many layers, many oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks"),
the exact form of fksuperscript𝑓𝑘f^{k} may be established as follows. (Recall that this refined form
allows for the use of Lebesgue measure in [Theorem 1.1](#S1.Thmtheorem1 "Theorem 1.1. ‣ 1.1 Main result ‣ 1 Setting and main results ‣ Benefits of depth in neural networks"),
and also the repetition statement in [Proposition 5.1](#S5.Thmtheorem1 "Proposition 5.1. ‣ 5 Bibliographic notes and open problems ‣ Benefits of depth in neural networks").)

###### Proof A.5.

(of [Lemma 3.17](#S3.Thmtheorem17 "Lemma 3.17. ‣ 3.3 Many layers, many oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks"))
The proof proceeds by induction on the number of compositions lll.
For the base case l=1l1l=1,

|  |  |  |
| --- | --- | --- |
|  | f1​(z)=f​(z)={2​zwhen z∈[0,1/2],2​(1−z)when z∈(1/2,1],0otherwise.superscript𝑓1𝑧𝑓𝑧cases2𝑧when z∈[0,1/2]21𝑧when z∈(1/2,1]0otherwisef^{1}(z)=f(z)=\begin{cases}2z&\textup{when $z\in[0,1/2]$},\\ 2(1-z)&\textup{when $z\in(1/2,1]$},\\ 0&\textup{otherwise}.\end{cases} |  |

For the inductive step,
first note for any z∈[0,1/2]𝑧012z\in[0,1/2],
by symmetry of flsuperscript𝑓𝑙f^{l} around 1/2 (i.e., fl​(z)=fl​(1−z)superscript𝑓𝑙𝑧superscript𝑓𝑙1𝑧f^{l}(z)=f^{l}(1-z) by the inductive hypothesis),
and by the above explicit form of f1superscript𝑓1f^{1},

|  |  |  |
| --- | --- | --- |
|  | fl+1​(z)=fl​(f​(z))=fl​(2​z)=fl​(1−2​z)=fl​(f​(1/2−z))=fl​(f​(z+1/2))=fl+1​(z+1/2),superscript𝑓𝑙1𝑧superscript𝑓𝑙𝑓𝑧superscript𝑓𝑙2𝑧superscript𝑓𝑙12𝑧superscript𝑓𝑙𝑓12𝑧superscript𝑓𝑙𝑓𝑧12superscript𝑓𝑙1𝑧12f^{l+1}(z)=f^{l}(f(z))=f^{l}(2z)=f^{l}(1-2z)=f^{l}(f(1/2-z))=f^{l}(f(z+1/2))=f^{l+1}(z+1/2), |  |

meaning the case z∈(1/2,1]𝑧121z\in(1/2,1] is implied by the case z∈[0,1/2]𝑧012z\in[0,1/2].
Since the unique nonnegative integer il+1subscript𝑖𝑙1i\_{l+1} and real zl+1∈[0,1)subscript𝑧𝑙101z\_{l+1}\in[0,1) satisfy
2​z=2​(il+1+zl+1)​2−l−1=(il+1+zl+1)​2−l2𝑧2subscript𝑖𝑙1subscript𝑧𝑙1superscript2𝑙1subscript𝑖𝑙1subscript𝑧𝑙1superscript2𝑙2z=2(i\_{l+1}+z\_{l+1})2^{-l-1}=(i\_{l+1}+z\_{l+1})2^{-l},
the inductive hypothesis grants

|  |  |  |
| --- | --- | --- |
|  | (fl∘f)​(z)=fl​(2​z)={2​zl+1when 0≤zl+1≤1/2,2​(1−zl+1)when 1/2<zl+1<1,superscript𝑓𝑙𝑓𝑧superscript𝑓𝑙2𝑧cases2subscript𝑧𝑙1when 0≤zl+1≤1/221subscript𝑧𝑙1when 1/2<zl+1<1(f^{l}\circ f)(z)=f^{l}(2z)=\begin{cases}2z\_{l+1}&\textup{when $0\leq z\_{l+1}\leq 1/2$},\\ 2(1-z\_{l+1})&\textup{when $1/2<z\_{l+1}<1$},\end{cases} |  |

which completes the proof.

The proof of the slightly more general form of [Theorem 1.1](#S1.Thmtheorem1 "Theorem 1.1. ‣ 1.1 Main result ‣ 1 Setting and main results ‣ Benefits of depth in neural networks")
is as follows;
it does not quite imply [Theorem 1.1](#S1.Thmtheorem1 "Theorem 1.1. ‣ 1.1 Main result ‣ 1 Setting and main results ‣ Benefits of depth in neural networks"), since the constructed measure is not the Lebesgue measure even for
the ReLU-based (1,[0,1])101(1,[0,1])-triangle function from [Lemma 3.16](#S3.Thmtheorem16 "Lemma 3.16. ‣ 3.3 Many layers, many oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks").

###### Proof A.6.

(of [Theorem 3.19](#S3.Thmtheorem19 "Theorem 3.19. ‣ 3.4 Proof of Theorem 1.1 ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks"))
First note some general properties of fksuperscriptfkf^{k}.
By [Corollary 3.14](#S3.Thmtheorem14 "Corollary 3.14. ‣ 3.3 Many layers, many oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks"),
fksuperscriptfkf^{k} is (2k−1,[0,1])superscript2k101(2^{k-1},[0,1])-triangle,
which means there exist s:=2k+1assignssuperscript2k1s:=2^{k}+1 points (zi)i=1ssuperscriptsubscriptsubscriptzii1s(z\_{i})\_{i=1}^{s}
so that fk​(zi)=𝟏​[i is odd]superscriptfksubscriptzi1delimited-[]i is oddf^{k}(z\_{i})=\mathbf{1}[\textup{$i$ is odd}],
and moreover fksuperscriptfkf^{k} is continuous and equal to 1/2121/2 at exactly
2ksuperscript2k2^{k} points (by the strict increasing/decreasing part of the triangle wave definition),
which is a finite set of points and thus has Lebesgue measure zero.
Taking py:ℝ→ℝd:subscriptpy→ℝsuperscriptℝdp\_{y}:\mathbb{R}\to\mathbb{R}^{d} to be the map py​(z)=(z,y)subscriptpyzzyp\_{y}(z)=(z,y) where y∈ℝd−1ysuperscriptℝd1y\in\mathbb{R}^{d-1},
then (h∘py)​(z)=h​((z,y))=fk​(z)hsubscriptpyzhzysuperscriptfkz(h\circ p\_{y})(z)=h((z,y))=f^{k}(z),
thus letting ℐℐ\mathcal{I} denote the 2ksuperscript2k2^{k} pieces within which fk~~superscriptfk\widetilde{f^{k}} is constant,
it follows that h∘py~~hsubscriptpy\widetilde{h\circ p\_{y}} is constant within the same set of pieces
and thus Cr​(h∘py)=sCrhsubscriptpys\textup{Cr}(h\circ p\_{y})=s.

Now consider the discrete case, where ν𝜈\nu denotes the uniform
measure over the s𝑠s points (xi)i=1ssuperscriptsubscriptsubscript𝑥𝑖𝑖1𝑠(x\_{i})\_{i=1}^{s} defined as xi:=p0​(zi)∈ℝdassignsubscript𝑥𝑖subscript𝑝0subscript𝑧𝑖superscriptℝ𝑑x\_{i}:=p\_{0}(z\_{i})\in\mathbb{R}^{d}.
Further consider the two types of distance.

* •

  Since zi<zi+1subscript𝑧𝑖subscript𝑧𝑖1z\_{i}<z\_{i+1} and fk~​(zi)≠fk~​(zi+1)~superscript𝑓𝑘subscript𝑧𝑖~superscript𝑓𝑘subscript𝑧𝑖1\widetilde{f^{k}}(z\_{i})\neq\widetilde{f^{k}}(z\_{i+1}),
  then taking (Ui)i=1ssuperscriptsubscriptsubscript𝑈𝑖𝑖1𝑠(U\_{i})\_{i=1}^{s} to denote the intervals of ℐℐ\mathcal{I} sorted by their left endpoint,
  zi∈Uisubscript𝑧𝑖subscript𝑈𝑖z\_{i}\in U\_{i} for i∈[s]𝑖delimited-[]𝑠i\in[s].
  By [Lemma 3.1](#S3.Thmtheorem1 "Lemma 3.1. ‣ 3.1 Approximation via oscillation counting ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks"),

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | ∫|h~−g~|​𝑑ν~ℎ~𝑔differential-d𝜈\displaystyle\int|\tilde{h}-\tilde{g}|d\nu | =1s​∑i=1s|h~​(xi)−g~​(xi)|=1s​∑i=1s|fk~​(zi)−g∘p0~​(zi)|absent1𝑠superscriptsubscript𝑖1𝑠~ℎsubscript𝑥𝑖~𝑔subscript𝑥𝑖1𝑠superscriptsubscript𝑖1𝑠~superscript𝑓𝑘subscript𝑧𝑖~𝑔subscript𝑝0subscript𝑧𝑖\displaystyle=\frac{1}{s}\sum\_{i=1}^{s}|\tilde{h}(x\_{i})-\tilde{g}(x\_{i})|=\frac{1}{s}\sum\_{i=1}^{s}|\widetilde{f^{k}}(z\_{i})-\widetilde{g\circ p\_{0}}(z\_{i})| |  |
  |  |  |  |  |
  | --- | --- | --- | --- |
  |  |  | ≥1s​∑i=1s𝟏​[∀z∈Ui​\centerdot​fk~​(z)≠g∘p0~​(z)]absent1𝑠superscriptsubscript𝑖1𝑠1delimited-[]for-all𝑧subscript𝑈𝑖\centerdot~superscript𝑓𝑘𝑧~𝑔subscript𝑝0𝑧\displaystyle\geq\frac{1}{s}\sum\_{i=1}^{s}\mathbf{1}[\forall z\in U\_{i}\centerdot\widetilde{f^{k}}(z)\neq\widetilde{g\circ p\_{0}}(z)] |  |
  |  |  |  |  |
  | --- | --- | --- | --- |
  |  |  | ≥12​(1−2​(2k−2s))≥14.absent1212superscript2𝑘2𝑠14\displaystyle\geq\frac{1}{2}\left(1-2\left(\frac{2^{k-2}}{s}\right)\right)\geq\frac{1}{4}. |  |
* •

  Since fk​(zi)∈{0,1}superscript𝑓𝑘subscript𝑧𝑖01f^{k}(z\_{i})\in\{0,1\}, then fk~​(zi)≠g~​(xi)~superscript𝑓𝑘subscript𝑧𝑖~𝑔subscript𝑥𝑖\widetilde{f^{k}}(z\_{i})\neq\widetilde{g}(x\_{i}) implies
  |fk​(zi)−g​(xi)|≥1/2superscript𝑓𝑘subscript𝑧𝑖𝑔subscript𝑥𝑖12|f^{k}(z\_{i})-g(x\_{i})|\geq 1/2,
  thus ∫[0,1]d|h−g|​𝑑ν≥∫[0,1]d|h~−g~|​𝑑ν/2≥1/8subscriptsuperscript01𝑑ℎ𝑔differential-d𝜈subscriptsuperscript01𝑑~ℎ~𝑔differential-d𝜈218\int\_{[0,1]^{d}}|h-g|d\nu\geq\int\_{[0,1]^{d}}|\tilde{h}-\tilde{g}|d\nu/2\geq 1/8.

Construct the continuous measure μ𝜇\mu as follows,
starting with the construction of a univariate measure μ0subscript𝜇0\mu\_{0}.
Since fksuperscript𝑓𝑘f^{k} is continuous, there exists a δ∈(0,mini∈[s−1]⁡|zi−zi+1|/2)𝛿0subscript𝑖delimited-[]𝑠1subscript𝑧𝑖subscript𝑧𝑖12\delta\in(0,\min\_{i\in[s-1]}|z\_{i}-z\_{i+1}|/2)
so that |fk​(z)−fk​(zi)|≤1/4superscript𝑓𝑘𝑧superscript𝑓𝑘subscript𝑧𝑖14|f^{k}(z)-f^{k}(z\_{i})|\leq 1/4 for any i∈[s]𝑖delimited-[]𝑠i\in[s] and z𝑧z with |z−zi|≤δ𝑧subscript𝑧𝑖𝛿|z-z\_{i}|\leq\delta.
As such, let μ0subscript𝜇0\mu\_{0} denote the probability measure which places half of its mass
uniformly on these s𝑠s balls of radius δ𝛿\delta (which must be disjoint since fksuperscript𝑓𝑘f^{k} alternates between
0 and 1 along (zi)i=1ssuperscriptsubscriptsubscript𝑧𝑖𝑖1𝑠(z\_{i})\_{i=1}^{s}),
and half of its mass uniformly on the remaining subset of [0,1]01[0,1].
Finally, extend this to a probability measure μ𝜇\mu on [0,1]dsuperscript01𝑑[0,1]^{d} uniformly, meaning
μ𝜇\mu is the product of μ0subscript𝜇0\mu\_{0} and the measure μ1subscript𝜇1\mu\_{1} which is uniform over [0,1]d−1superscript01𝑑1[0,1]^{d-1}.
Now consider the two types of distances.

* •

  By [Lemma 3.1](#S3.Thmtheorem1 "Lemma 3.1. ‣ 3.1 Approximation via oscillation counting ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks"),

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | ∫|h~−g~|​𝑑μ​(x)~ℎ~𝑔differential-d𝜇𝑥\displaystyle\int|\tilde{h}-\tilde{g}|d\mu(x) | =∬|fk~​(py​(z))−g~​(py​(z))|​𝑑μ0​(z)​𝑑μ1​(y)absentdouble-integral~superscript𝑓𝑘subscript𝑝𝑦𝑧~𝑔subscript𝑝𝑦𝑧differential-dsubscript𝜇0𝑧differential-dsubscript𝜇1𝑦\displaystyle=\iint|\widetilde{f^{k}}(p\_{y}(z))-\tilde{g}(p\_{y}(z))|d\mu\_{0}(z)d\mu\_{1}(y) |  |
  |  |  |  |  |
  | --- | --- | --- | --- |
  |  |  | =∫∑U∈ℐ∫𝟏[z∈U∧fk~(z))≠g~(py(z))]dμ0(z)dμ1(y)\displaystyle=\int\sum\_{U\in\mathcal{I}}\int\mathbf{1}[z\in U\land\widetilde{f^{k}}(z))\neq\tilde{g}(p\_{y}(z))]d\mu\_{0}(z)d\mu\_{1}(y) |  |
  |  |  |  |  |
  | --- | --- | --- | --- |
  |  |  | ≥∫12​s∑U∈ℐ𝟏[∀z∈U\centerdotfk~(z))≠g∘py~(z)]dμ1(y)\displaystyle\geq\int\frac{1}{2s}\sum\_{U\in\mathcal{I}}\mathbf{1}[\forall z\in U\centerdot\widetilde{f^{k}}(z))\neq\widetilde{g\circ p\_{y}}(z)]d\mu\_{1}(y) |  |
  |  |  |  |  |
  | --- | --- | --- | --- |
  |  |  | ≥14​(1−2​(2k−2s))≥18.absent1412superscript2𝑘2𝑠18\displaystyle\geq\frac{1}{4}\left(1-2\left(\frac{2^{k-2}}{s}\right)\right)\geq\frac{1}{8}. |  |
* •

  For any y∈ℝd−1𝑦superscriptℝ𝑑1y\in\mathbb{R}^{d-1} and Ui∈ℐsubscript𝑈𝑖ℐU\_{i}\in\mathcal{I} (with corresponding zi∈Uisubscript𝑧𝑖subscript𝑈𝑖z\_{i}\in U\_{i}),
  if fk~​(z)≠g∘py~​(z)~superscript𝑓𝑘𝑧~𝑔subscript𝑝𝑦𝑧\widetilde{f^{k}}(z)\neq\widetilde{g\circ p\_{y}}(z) for every z∈Ui𝑧subscript𝑈𝑖z\in U\_{i},
  then

  |  |  |  |
  | --- | --- | --- |
  |  | ∫Ui|fk​(z)−g​(py​(z))|​𝑑μ0​(z)≥∫|z−zi|≤δ|fk​(z)−1/2|​𝑑μ0​(z)≥14​μ0​({z∈Ui:|z−zi|≤δ})≥18​s.subscriptsubscript𝑈𝑖superscript𝑓𝑘𝑧𝑔subscript𝑝𝑦𝑧differential-dsubscript𝜇0𝑧subscript𝑧subscript𝑧𝑖𝛿superscript𝑓𝑘𝑧12differential-dsubscript𝜇0𝑧14subscript𝜇0conditional-set𝑧subscript𝑈𝑖𝑧subscript𝑧𝑖𝛿18𝑠\int\_{U\_{i}}|{f^{k}}(z)-g(p\_{y}(z))|d\mu\_{0}(z)\geq\int\_{|z-z\_{i}|\leq\delta}|{f^{k}}(z)-1/2|d\mu\_{0}(z)\geq\frac{1}{4}\mu\_{0}(\{z\in U\_{i}:|z-z\_{i}|\leq\delta\})\geq\frac{1}{8s}. |  |

  By [Lemma 3.1](#S3.Thmtheorem1 "Lemma 3.1. ‣ 3.1 Approximation via oscillation counting ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks"),

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | ∫|h−g|​𝑑μ​(x)ℎ𝑔differential-d𝜇𝑥\displaystyle\int|h-g|d\mu(x) | =∬|h​(py​(z))−g​(py​(z))|​𝑑μ0​(z)​𝑑μ1​(y)absentdouble-integralℎsubscript𝑝𝑦𝑧𝑔subscript𝑝𝑦𝑧differential-dsubscript𝜇0𝑧differential-dsubscript𝜇1𝑦\displaystyle=\iint|h(p\_{y}(z))-g(p\_{y}(z))|d\mu\_{0}(z)d\mu\_{1}(y) |  |
  |  |  |  |  |
  | --- | --- | --- | --- |
  |  |  | ≥∫∑U∈ℐ𝟏​[∀z∈U​\centerdot​fk~​(z)≠g~​(py​(z))]​∫U|fk​(z)−g​(py​(z))|​𝑑μ0​(z)​𝑑μ1​(y)absentsubscript𝑈ℐ1delimited-[]for-all𝑧𝑈\centerdot~superscript𝑓𝑘𝑧~𝑔subscript𝑝𝑦𝑧subscript𝑈superscript𝑓𝑘𝑧𝑔subscript𝑝𝑦𝑧differential-dsubscript𝜇0𝑧differential-dsubscript𝜇1𝑦\displaystyle\geq\int\sum\_{U\in\mathcal{I}}\mathbf{1}[\forall z\in U\centerdot\widetilde{f^{k}}(z)\neq\tilde{g}(p\_{y}(z))]\int\_{U}|{f^{k}}(z)-g(p\_{y}(z))|d\mu\_{0}(z)d\mu\_{1}(y) |  |
  |  |  |  |  |
  | --- | --- | --- | --- |
  |  |  | ≥∫18​s​∑U∈ℐ𝟏​[∀z∈U​\centerdot​fk~​(z)≠g∘py~​(z)]​d​μ1​(y)absent18𝑠subscript𝑈ℐ1delimited-[]for-all𝑧𝑈\centerdot~superscript𝑓𝑘𝑧~𝑔subscript𝑝𝑦𝑧𝑑subscript𝜇1𝑦\displaystyle\geq\int\frac{1}{8s}\sum\_{U\in\mathcal{I}}\mathbf{1}[\forall z\in U\centerdot\widetilde{f^{k}}(z)\neq\widetilde{g\circ p\_{y}}(z)]d\mu\_{1}(y) |  |
  |  |  |  |  |
  | --- | --- | --- | --- |
  |  |  | ≥116​(1−2​(2k−2s))≥132.absent11612superscript2𝑘2𝑠132\displaystyle\geq\frac{1}{16}\left(1-2\left(\frac{2^{k-2}}{s}\right)\right)\geq\frac{1}{32}. |  |

As a closing curiosity,
[Theorem 3.19](#S3.Thmtheorem19 "Theorem 3.19. ‣ 3.4 Proof of Theorem 1.1 ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks") implies the following statement regarding polynomials.

###### Corollary A.7.

For any integer k≥1𝑘1k\geq 1, there exists a polynomial h:ℝd→ℝ:ℎ→superscriptℝ𝑑ℝh:\mathbb{R}^{d}\to\mathbb{R} with degree 2ksuperscript2𝑘2^{k}
and a corresponding continuous measure μ𝜇\mu which is positive everywhere over [0,1]dsuperscript01𝑑[0,1]^{d}
so that every polynomial g:ℝd→ℝ:𝑔→superscriptℝ𝑑ℝg:\mathbb{R}^{d}\to\mathbb{R} of degree ≤2k−3absentsuperscript2𝑘3\leq 2^{k-3} satisfies
∫|h−g|​𝑑μ≥1/32.ℎ𝑔differential-d𝜇132\int|h-g|d\mu\geq 1/32.

###### Proof A.8.

Set f​(z)=4​z​(1−z)𝑓𝑧4𝑧1𝑧f(z)=4z(1-z), which by [Lemma 3.16](#S3.Thmtheorem16 "Lemma 3.16. ‣ 3.3 Many layers, many oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks") is (1,[0,1])101(1,[0,1])-triangle,
thus fksuperscript𝑓𝑘f^{k} is (2k−1,[0,1])superscript2𝑘101(2^{k-1},[0,1])-triangle with Cr​(fk)=2k+1Crsuperscript𝑓𝑘superscript2𝑘1\textup{Cr}(f^{k})=2^{k}+1 by [Corollary 3.14](#S3.Thmtheorem14 "Corollary 3.14. ‣ 3.3 Many layers, many oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks"),
and fksuperscript𝑓𝑘f^{k} has degree 2ksuperscript2𝑘2^{k} directly; thus set h​(x)=fk​(x1)ℎ𝑥superscript𝑓𝑘subscript𝑥1h(x)=f^{k}(x\_{1}).
Next, for any polynomial
g:ℝd→ℝ:𝑔→superscriptℝ𝑑ℝg:\mathbb{R}^{d}\to\mathbb{R} of degree ≤2k−3absentsuperscript2𝑘3\leq 2^{k-3},
g∘py:ℝ→ℝ:𝑔subscript𝑝𝑦→ℝℝg\circ p\_{y}:\mathbb{R}\to\mathbb{R} is still a polynomial of degree ≤2k−3absentsuperscript2𝑘3\leq 2^{k-3}
for every y∈ℝd−1𝑦superscriptℝ𝑑1y\in\mathbb{R}^{d-1} (where py​(z)=(z,y)subscript𝑝𝑦𝑧𝑧𝑦p\_{y}(z)=(z,y) as in [Theorem 3.19](#S3.Thmtheorem19 "Theorem 3.19. ‣ 3.4 Proof of Theorem 1.1 ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks")),
and so [Lemma 3.4](#S3.Thmtheorem4 "Lemma 3.4. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks") grants
Cr​(g∘py)≤1+2k−3≤2k−2Cr𝑔subscript𝑝𝑦1superscript2𝑘3superscript2𝑘2\textup{Cr}(g\circ p\_{y})\leq 1+2^{k-3}\leq 2^{k-2}.
The result follows by [Theorem 3.19](#S3.Thmtheorem19 "Theorem 3.19. ‣ 3.4 Proof of Theorem 1.1 ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks").

### A.3 Deferred proofs from [Section 4](#S4 "4 Limitations of depth ‣ Benefits of depth in neural networks")

First, the proof of a certain VC lower bound which mimics the Gilbert-Varshamov bound;
the proof is little more than a consequence of Hoeffding’s inequality.

###### Proof A.9.

(of [Lemma 4.1](#S4.Thmtheorem1 "Lemma 4.1. ‣ 4 Limitations of depth ‣ Benefits of depth in neural networks"))
For convenience, set m:=Sh​(ℱ;n)assignmSh

ℱnm:=\textup{Sh}(\mathcal{F};n),
and let (a1,…,am)subscripta1…subscriptam(a\_{1},\ldots,a\_{m}) denote these dichotomies (meaning aj∈{0,1}nsubscriptajsuperscript01na\_{j}\in\{0,1\}^{n}),
and with foresight set
ϵ:=ln⁡(m/δ)/(2​n)assignϵmδ2n\epsilon:=\sqrt{\ln(m/\delta)/(2n)}.
Let (Yi)i=1nsuperscriptsubscriptsubscriptYii1n(Y\_{i})\_{i=1}^{n} denote fair Bernoulli random labellings for each point,
and note by symmetry of the fair coin that for any fixed dichotomy ajsubscriptaja\_{j},

|  |  |  |
| --- | --- | --- |
|  | Pr​[1n​∑i=1n|(aj)i−Yi|<1/2−ϵ]=Pr​[1n​∑i=1nYi<1/2−ϵ].Prdelimited-[]1𝑛superscriptsubscript𝑖1𝑛subscriptsubscript𝑎𝑗𝑖subscript𝑌𝑖12italic-ϵPrdelimited-[]1𝑛superscriptsubscript𝑖1𝑛subscript𝑌𝑖12italic-ϵ\displaystyle\textup{Pr}\left[\frac{1}{n}\sum\_{i=1}^{n}|(a\_{j})\_{i}-Y\_{i}|<1/2-\epsilon\right]=\textup{Pr}\left[\frac{1}{n}\sum\_{i=1}^{n}Y\_{i}<1/2-\epsilon\right]. |  |

Consequently, by a union bound over all dichotomies and lastly by Hoeffding’s inequality,

|  |  |  |  |
| --- | --- | --- | --- |
|  | Pr​[∃f∈ℱ​\centerdot​1n​∑i=1n|f~​(xi)−Yi|<1/2−ϵ]Prdelimited-[]𝑓ℱ\centerdot1𝑛superscriptsubscript𝑖1𝑛~𝑓subscript𝑥𝑖subscript𝑌𝑖12italic-ϵ\displaystyle\textup{Pr}\left[\exists f\in\mathcal{F}\centerdot\frac{1}{n}\sum\_{i=1}^{n}|\tilde{f}(x\_{i})-Y\_{i}|<1/2-\epsilon\right] | ≤∑j=1mPr​[1n​∑i=1n|(vj)i−Yi|<1/2−ϵ]absentsuperscriptsubscript𝑗1𝑚Prdelimited-[]1𝑛superscriptsubscript𝑖1𝑛subscriptsubscript𝑣𝑗𝑖subscript𝑌𝑖12italic-ϵ\displaystyle\leq\sum\_{j=1}^{m}\textup{Pr}\left[\frac{1}{n}\sum\_{i=1}^{n}|(v\_{j})\_{i}-Y\_{i}|<1/2-\epsilon\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =m​Pr​[1n​∑i=1nYi<1/2−ϵ]absent𝑚Prdelimited-[]1𝑛superscriptsubscript𝑖1𝑛subscript𝑌𝑖12italic-ϵ\displaystyle=m\textup{Pr}\left[\frac{1}{n}\sum\_{i=1}^{n}Y\_{i}<1/2-\epsilon\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤m​exp⁡(−2​n​ϵ2)≤δ,absent𝑚2𝑛superscriptitalic-ϵ2𝛿\displaystyle\leq m\exp(-2n\epsilon^{2})\leq\delta, |  |

where the last step used the choice of ϵitalic-ϵ\epsilon.

The remaining deferred proofs do not exactly follow the order of [Section 4](#S4 "4 Limitations of depth ‣ Benefits of depth in neural networks"),
but instead the order of dependencies in the proofs. In particular, to control
the VC dimension, first it is useful to prove [Lemma 4.3](#S4.Thmtheorem3 "Lemma 4.3. ‣ 4 Limitations of depth ‣ Benefits of depth in neural networks"),
which is used to control the growth of numbers of regions as semi-algebraic gates are combined.

###### Proof A.10.

(of [Lemma 4.3](#S4.Thmtheorem3 "Lemma 4.3. ‣ 4 Limitations of depth ‣ Benefits of depth in neural networks"))
Fix some ordering (q1,q2,…,q|𝒬|)subscriptq1subscriptq2…subscriptq𝒬(q\_{1},q\_{2},\ldots,q\_{|\mathcal{Q}|}) of the elements of 𝒬𝒬\mathcal{Q},
and for each i∈[|𝒬|]idelimited-[]𝒬i\in[|\mathcal{Q}|] define two functions li​(a):=𝟏​[qi​(a)<0]assignsubscriptlia1delimited-[]subscriptqia0l\_{i}(a):=\mathbf{1}[q\_{i}(a)<0] and ui​(a):=𝟏​[qi​(a)≥0]assignsubscriptuia1delimited-[]subscriptqia0u\_{i}(a):=\mathbf{1}[q\_{i}(a)\geq 0],
as well as two sets Li:={a∈ℝp:li​(a)=1}assignsubscriptLiconditional-setasuperscriptℝpsubscriptlia1L\_{i}:=\{a\in\mathbb{R}^{p}:l\_{i}(a)=1\} and Ui:={a∈ℝp:ui​(a)=1}assignsubscriptUiconditional-setasuperscriptℝpsubscriptuia1U\_{i}:=\{a\in\mathbb{R}^{p}:u\_{i}(a)=1\}.
Note that

|  |  |  |
| --- | --- | --- |
|  | 𝒮:={(∩i∈ALi)∩(∩i∈B):A⊆[|𝒬|],B⊆[|𝒬|]}∖{∅}.assign𝒮conditional-setsubscript𝑖𝐴subscript𝐿𝑖subscript𝑖𝐵formulae-sequence𝐴delimited-[]𝒬𝐵delimited-[]𝒬\mathcal{S}:=\Big{\{}(\cap\_{i\in A}L\_{i})\cap(\cap\_{i\in B}):A\subseteq[|\mathcal{Q}|],B\subseteq[|\mathcal{Q}|]\ \Big{\}}\setminus\{\emptyset\}. |  |

Additionally consider the set of sign patterns

|  |  |  |
| --- | --- | --- |
|  | V:={(l1​(a),ui​(a),…,l|𝒬|​(a),u|𝒬|​(a)):a∈ℝp}.assign𝑉conditional-setsubscript𝑙1𝑎subscript𝑢𝑖𝑎…subscript𝑙𝒬𝑎subscript𝑢𝒬𝑎𝑎superscriptℝ𝑝V:=\left\{\left(l\_{1}(a),u\_{i}(a),\ldots,l\_{|\mathcal{Q}|}(a),u\_{|\mathcal{Q}|}(a)\right):a\in\mathbb{R}^{p}\right\}. |  |

Distinct elements of 𝒮𝒮\mathcal{S} correspond to distinct sign patterns in V𝑉V:
namely, for any C∈𝒮𝐶𝒮C\in\mathcal{S}, using the ordering of 𝒬𝒬\mathcal{Q} to encode A𝐴A and B𝐵B
as binary vectors of length |𝒬|𝒬|\mathcal{Q}|, the corresponding interleaved binary vector of length 2​|𝒬|2𝒬2|\mathcal{Q}|
is distinct for distinct choices of (A,B)𝐴𝐵(A,B).
(For each i𝑖i that appears in neither A𝐴A nor B𝐵B, there two possible encodings in V𝑉V:
having both coordinates corresponding to i𝑖i set to 1, and having them set to 0.
On the other hand, a more succinct encoding based just on (li)i=1|𝒬|superscriptsubscriptsubscript𝑙𝑖𝑖1𝒬(l\_{i})\_{i=1}^{|\mathcal{Q}|} fails
to capture those sets arising from intersections of proper subsets of 𝒬𝒬\mathcal{Q}.)
As such,
making use of growth function bounds for sets of polynomials (Anthony and Bartlett, [1999](#bib.bib1), Theorem 8.3),

|  |  |  |
| --- | --- | --- |
|  | |𝒮|≤|V|≤2​(4​e​α​|𝒬|p)p.𝒮𝑉2superscript4𝑒𝛼𝒬𝑝𝑝|\mathcal{S}|\leq|V|\leq 2\left(\frac{4e\alpha|\mathcal{Q}|}{p}\right)^{p}. |  |

Thanks to [Lemma 4.3](#S4.Thmtheorem3 "Lemma 4.3. ‣ 4 Limitations of depth ‣ Benefits of depth in neural networks"), the proof of the VC dimension bound [Lemma 4.2](#S4.Thmtheorem2 "Lemma 4.2. ‣ 4 Limitations of depth ‣ Benefits of depth in neural networks")
follows by induction over layers, effectively keeping track of a piecewise (regionwise?) polynomial
function as with the proof of [Lemma 3.3](#S3.Thmtheorem3 "Lemma 3.3. ‣ 3.2 Few layers, few oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks") (but now in the multivariate case).

###### Proof A.11.

(of [Lemma 4.2](#S4.Thmtheorem2 "Lemma 4.2. ‣ 4 Limitations of depth ‣ Benefits of depth in neural networks"))
First note that this proof follows the scheme of a VC dimension proof for networks with piecewise
polynomial activation functions (Anthony and Bartlett, [1999](#bib.bib1), Theorem 8.8),
but with [Lemma 4.3](#S4.Thmtheorem3 "Lemma 4.3. ‣ 4 Limitations of depth ‣ Benefits of depth in neural networks") allowing for the more complicated semi-algebraic gates,
and some additional bookkeeping for the (semi-algebraic) shapes of the regions of the partition 𝒮𝒮\mathcal{S}.

Let examples (xj)j=1nsuperscriptsubscriptsubscript𝑥𝑗𝑗1𝑛(x\_{j})\_{j=1}^{n} be given with n≥p𝑛𝑝n\geq p,
let misubscript𝑚𝑖m\_{i} denote the number of nodes in layer i𝑖i (whereby m1+⋯+ml=msubscript𝑚1⋯subscript𝑚𝑙𝑚m\_{1}+\cdots+m\_{l}=m),
and let f:=F𝔊:ℝp×ℝd→ℝ:assign𝑓subscript𝐹𝔊→superscriptℝ𝑝superscriptℝ𝑑ℝf:=F\_{\mathfrak{G}}:\mathbb{R}^{p}\times\mathbb{R}^{d}\to\mathbb{R} denote the function evaluating the neural network (as in [Section 2.1](#S2.SS1 "2.1 Notation for neural networks ‣ 2 Semi-algebraic gates and assorted network notation ‣ Benefits of depth in neural networks")),
where the two arguments are the parameters w∈ℝp𝑤superscriptℝ𝑝w\in\mathbb{R}^{p} and the input example x∈ℝd𝑥superscriptℝ𝑑x\in\mathbb{R}^{d}.
The goal is to upper bound the number of dichotomies

|  |  |  |
| --- | --- | --- |
|  | K:=Sh​(𝒩​(𝔊);n)=|{(sgn​(f​(w,x1)),…,sgn​(f​(w,xn))):w∈ℝp}|.assign𝐾Sh  𝒩𝔊𝑛conditional-setsgn𝑓𝑤subscript𝑥1…sgn𝑓𝑤subscript𝑥𝑛𝑤superscriptℝ𝑝K:=\textup{Sh}(\mathcal{N}(\mathfrak{G});n)=\left|\left\{(\textup{sgn}(f(w,x\_{1})),\ldots,\textup{sgn}(f(w,x\_{n}))):w\in\mathbb{R}^{p}\right\}\right|. |  |

The proof will proceed by producing a sequence of partitions (𝒮i)0=1lsuperscriptsubscriptsubscript𝒮𝑖01𝑙(\mathcal{S}\_{i})\_{0=1}^{l} of ℝpsuperscriptℝ𝑝\mathbb{R}^{p} and
two corresponding sequences of sets of polynomials (𝒫i)i=0lsuperscriptsubscriptsubscript𝒫𝑖𝑖0𝑙(\mathcal{P}\_{i})\_{i=0}^{l} and (𝒬i)i=0lsuperscriptsubscriptsubscript𝒬𝑖𝑖0𝑙(\mathcal{Q}\_{i})\_{i=0}^{l}
so that for each i𝑖i,
𝒫isubscript𝒫𝑖\mathcal{P}\_{i} has polynomials of degree at most βisuperscript𝛽𝑖\beta^{i},
𝒬isubscript𝒬𝑖\mathcal{Q}\_{i} has polynomials of degree at most α​βi−1𝛼superscript𝛽𝑖1\alpha\beta^{i-1},
and over any parameters S∈𝒮i𝑆subscript𝒮𝑖S\in\mathcal{S}\_{i},
there is an assignment of elements of 𝒫isubscript𝒫𝑖\mathcal{P}\_{i} to nodes of layer i𝑖i
so that for each example xjsubscript𝑥𝑗x\_{j}, every node in layer i𝑖i evaluates the corresponding fixed
polynomial in 𝒫isubscript𝒫𝑖\mathcal{P}\_{i};
lastly, the elements of 𝒮isubscript𝒮𝑖\mathcal{S}\_{i} are intersections of sets of the form
{w∈ℝp:q​(w)⋄0}conditional-set𝑤superscriptℝ𝑝⋄𝑞𝑤0\{w\in\mathbb{R}^{p}:q(w)\diamond 0\} where q∈𝒬i𝑞subscript𝒬𝑖q\in\mathcal{Q}\_{i} and ⋄∈{<,≥}\diamond\in\{<,\geq\},
and the partition 𝒮i+1subscript𝒮𝑖1\mathcal{S}\_{i+1} refines 𝒮isubscript𝒮𝑖\mathcal{S}\_{i} for each i𝑖i (meaning for each U∈𝒮i+1𝑈subscript𝒮𝑖1U\in\mathcal{S}\_{i+1} there exists S⊇U𝑈𝑆S\supseteq U with S∈𝒮i𝑆subscript𝒮𝑖S\in\mathcal{S}\_{i}).
Setting the final partition 𝒮:=𝒮lassign𝒮subscript𝒮𝑙\mathcal{S}:=\mathcal{S}\_{l},
this in turn will give an upper bound on K𝐾K,
since the final output within each element of 𝒮𝒮\mathcal{S} is a fixed polynomial of degree at most βlsuperscript𝛽𝑙\beta^{l},
whereby the VC dimension of polynomials (Anthony and Bartlett, [1999](#bib.bib1), Theorem 8.3) grants

|  |  |  |  |
| --- | --- | --- | --- |
|  | K≤∑S∈𝒮|{(sgn​(f​(w,x1)),…,sgn​(f​(w,xn))):w∈S}|≤2​|𝒮|​(2​e​n​βlp)p.𝐾subscript𝑆𝒮conditional-setsgn𝑓𝑤subscript𝑥1…sgn𝑓𝑤subscript𝑥𝑛𝑤𝑆2𝒮superscript2𝑒𝑛superscript𝛽𝑙𝑝𝑝\displaystyle K\leq\sum\_{S\in\mathcal{S}}\left|\left\{(\textup{sgn}(f(w,x\_{1})),\ldots,\textup{sgn}(f(w,x\_{n}))):w\in S\right\}\right|\leq 2|\mathcal{S}|\left(\frac{2en\beta^{l}}{p}\right)^{p}. |  | (A.1) |

To start, consider layer 0 of the input coordinates themselves,
a collection of d𝑑d affine maps.
Consequently, it suffices to set 𝒮0:={ℝp}assignsubscript𝒮0superscriptℝ𝑝\mathcal{S}\_{0}:=\{\mathbb{R}^{p}\},
𝒬0:=∅assignsubscript𝒬0\mathcal{Q}\_{0}:=\emptyset, and 𝒫0subscript𝒫0\mathcal{P}\_{0} to be the n​d𝑛𝑑nd possible coordinate maps corresponding to
all d𝑑d coordinates of all n𝑛n examples.

For the inductive step, consider some layer i+1𝑖1i+1.
Restricted to any S∈𝒮i𝑆subscript𝒮𝑖S\in\mathcal{S}\_{i}, the nodes of the previous layer i𝑖i compute fixed polynomials
of degree βisuperscript𝛽𝑖\beta^{i}.
Each node in layer i+1𝑖1i+1 is (t,α,β)𝑡𝛼𝛽(t,\alpha,\beta)-sa,
meaning there are t𝑡t predicates, defined by polynomials of degree ≤αabsent𝛼\leq\alpha, which define regions
wherein this node is a fixed polynomial.
Let QSsubscript𝑄𝑆Q\_{S} denote this set of predicates,
where |QS|≤t​n​mi+1subscript𝑄𝑆𝑡𝑛subscript𝑚𝑖1|Q\_{S}|\leq tnm\_{i+1} by considering the n𝑛n possible input examples and the t𝑡t possible predicates
encountered in each of the mi+1subscript𝑚𝑖1m\_{i+1} nodes in layer i+1𝑖1i+1,
and set
Qi+1:=Qi​⋃(∪S∈𝒮iQS).assignsubscript𝑄𝑖1subscript𝑄𝑖subscript𝑆subscript𝒮𝑖subscript𝑄𝑆Q\_{i+1}:=Q\_{i}\bigcup\left(\cup\_{S\in\mathcal{S}\_{i}}Q\_{S}\right).
By the definition of semi-algebraic gate, each node in layer i+1𝑖1i+1 computes a fixed polynomial when
restricted to a region defined by an intersection of predicates which moreover are defined by Qi+1subscript𝑄𝑖1Q\_{i+1}.
As such, defining 𝒮i+1subscript𝒮𝑖1\mathcal{S}\_{i+1} as the refinement of 𝒮i+1subscript𝒮𝑖1\mathcal{S}\_{i+1} which partitions each S∈𝒮i𝑆subscript𝒮𝑖S\in\mathcal{S}\_{i} according
to the intersections of predicates encountered in each node,
then [Lemma 4.3](#S4.Thmtheorem3 "Lemma 4.3. ‣ 4 Limitations of depth ‣ Benefits of depth in neural networks") on each QSsubscript𝑄𝑆Q\_{S} grants

|  |  |  |  |
| --- | --- | --- | --- |
|  | |𝒮i+1|≤∑S∈𝒮i|{all nonempty intersections of QS}|≤2​|𝒮i|​(4​e​n​mi+1​t​α​βip)p,subscript𝒮𝑖1subscript𝑆subscript𝒮𝑖all nonempty intersections of QS2subscript𝒮𝑖superscript4𝑒𝑛subscript𝑚𝑖1𝑡𝛼superscript𝛽𝑖𝑝𝑝|\mathcal{S}\_{i+1}|\leq\sum\_{S\in\mathcal{S}\_{i}}|\{\textup{all nonempty intersections of $Q\_{S}$}\}|\leq 2|\mathcal{S}\_{i}|\left(\frac{4enm\_{i+1}t\alpha\beta^{i}}{p}\right)^{p}, |  | (A.2) |

which completes the inductive construction.

The upper bound on K𝐾K may now be estimated.
First, |𝒮|𝒮|\mathcal{S}| may be upper bounded by applying [eq. A.2](#A1.E2 "In Proof A.11. ‣ A.3 Deferred proofs from Section 4 ‣ Appendix A Deferred proofs ‣ Benefits of depth in neural networks") recursively:

|  |  |  |
| --- | --- | --- |
|  | |𝒮|≤|𝒮0|​∏i=1l(8​e​n​mi​t​α​βi−1p)p≤(8​e​n​m​t​α​βl−1)p​l.𝒮subscript𝒮0superscriptsubscriptproduct𝑖1𝑙superscript8𝑒𝑛subscript𝑚𝑖𝑡𝛼superscript𝛽𝑖1𝑝𝑝superscript8𝑒𝑛𝑚𝑡𝛼superscript𝛽𝑙1𝑝𝑙\displaystyle|\mathcal{S}|\leq|\mathcal{S}\_{0}|\prod\_{i=1}^{l}\left(\frac{8enm\_{i}t\alpha\beta^{i-1}}{p}\right)^{p}\leq\left(8enmt\alpha\beta^{l-1}\right)^{pl}. |  |

Continuing from [Equation A.1](#A1.E1 "In Proof A.11. ‣ A.3 Deferred proofs from Section 4 ‣ Appendix A Deferred proofs ‣ Benefits of depth in neural networks"),

|  |  |  |  |
| --- | --- | --- | --- |
|  | K𝐾\displaystyle K | ≤2​|𝒮|​(2​e​m​βlp)p≤(8​e​n​m​t​α​βl)p​(l+1).absent2𝒮superscript2𝑒𝑚superscript𝛽𝑙𝑝𝑝superscript8𝑒𝑛𝑚𝑡𝛼superscript𝛽𝑙𝑝𝑙1\displaystyle\leq 2|\mathcal{S}|\left(\frac{2em\beta^{l}}{p}\right)^{p}\leq\left(8enmt\alpha\beta^{l}\right)^{p(l+1)}. |  |

  

To compute VC​(𝒩​(𝔊))VC𝒩𝔊\textup{VC}(\mathcal{N}(\mathfrak{G})), it suffices to find N𝑁N such that Sh​(𝒩​(𝔊);N)<2NSh

𝒩𝔊𝑁superscript2𝑁\textup{Sh}(\mathcal{N}(\mathfrak{G});N)<2^{N},
which in turn is implied by p​(l+1)​ln⁡(N)+p​(l+1)​ln⁡(8​e​m​t​α​βl)<N​ln⁡(2)𝑝𝑙1𝑁𝑝𝑙18𝑒𝑚𝑡𝛼superscript𝛽𝑙𝑁2p(l+1)\ln(N)+p(l+1)\ln(8emt\alpha\beta^{l})<N\ln(2).
Since ln(N)=ln(N/(2p(l+1))+ln(2p(l+1))≤N/(2p(l+1))−1+ln(2p(l+1))\ln(N)=\ln(N/(2p(l+1))+\ln(2p(l+1))\leq N/(2p(l+1))-1+\ln(2p(l+1))
and ln⁡(2)−1/2>1/621216\ln(2)-1/2>1/6,
it suffices to show

|  |  |  |
| --- | --- | --- |
|  | 6​p​(l+1)​(ln⁡(2​p​(l+1))+ln⁡(8​e​m​t​α​βl))≤N.6𝑝𝑙12𝑝𝑙18𝑒𝑚𝑡𝛼superscript𝛽𝑙𝑁6p(l+1)\left(\ln(2p(l+1))+\ln(8emt\alpha\beta^{l})\right)\leq N. |  |

As such, the left hand side of this expression is an upper bound on VC​(𝒩​(𝔊))VC𝒩𝔊\textup{VC}(\mathcal{N}(\mathfrak{G})).

The proofs of [Lemma 1.3](#S1.Thmtheorem3 "Lemma 1.3 (Simplification of Lemma 4.2). ‣ 1.2 Companion results ‣ 1 Setting and main results ‣ Benefits of depth in neural networks") and [Theorem 1.2](#S1.Thmtheorem2 "Theorem 1.2. ‣ 1.2 Companion results ‣ 1 Setting and main results ‣ Benefits of depth in neural networks") from [Section 1](#S1 "1 Setting and main results ‣ Benefits of depth in neural networks")
are now direct from [Lemma 4.2](#S4.Thmtheorem2 "Lemma 4.2. ‣ 4 Limitations of depth ‣ Benefits of depth in neural networks") and [Lemma 4.1](#S4.Thmtheorem1 "Lemma 4.1. ‣ 4 Limitations of depth ‣ Benefits of depth in neural networks").

###### Proof A.12.

(of [Lemma 1.3](#S1.Thmtheorem3 "Lemma 1.3 (Simplification of Lemma 4.2). ‣ 1.2 Companion results ‣ 1 Setting and main results ‣ Benefits of depth in neural networks"))
This statement is the same as [Lemma 4.2](#S4.Thmtheorem2 "Lemma 4.2. ‣ 4 Limitations of depth ‣ Benefits of depth in neural networks") with some details removed.

###### Proof A.13.

(of [Theorem 1.2](#S1.Thmtheorem2 "Theorem 1.2. ‣ 1.2 Companion results ‣ 1 Setting and main results ‣ Benefits of depth in neural networks"))
By the bound on Sh​(𝒩​(𝔊);n)Sh

𝒩𝔊n\textup{Sh}(\mathcal{N}(\mathfrak{G});n) from [Lemma 4.2](#S4.Thmtheorem2 "Lemma 4.2. ‣ 4 Limitations of depth ‣ Benefits of depth in neural networks"),

|  |  |  |  |
| --- | --- | --- | --- |
|  | n=n2+n2𝑛𝑛2𝑛2\displaystyle n=\frac{n}{2}+\frac{n}{2} | ≥2​ln⁡(1/δ)+4​p​l2​ln⁡(8​e​m​t​α​β​p​(l+1))+n2absent21𝛿4𝑝superscript𝑙28𝑒𝑚𝑡𝛼𝛽𝑝𝑙1𝑛2\displaystyle\geq 2\ln(1/\delta)+4pl^{2}\ln(8emt\alpha\beta p(l+1))+\frac{n}{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≥2ln(1/δ)+2p(l+1)ln(8emtαβl)+2p(l+1)(ln(p(l+1)))+n2​p​(l+1)−1)\displaystyle\geq 2\ln(1/\delta)+2p(l+1)\ln(8emt\alpha\beta^{l})+2p(l+1)\left(\ln(p(l+1)))+\frac{n}{2p(l+1)}-1\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≥2​ln⁡(1/δ)+2​p​(l+1)​ln⁡(8​e​m​t​α​βl)+2​p​(l+1)​ln⁡(n)absent21𝛿2𝑝𝑙18𝑒𝑚𝑡𝛼superscript𝛽𝑙2𝑝𝑙1𝑛\displaystyle\geq 2\ln(1/\delta)+2p(l+1)\ln(8emt\alpha\beta^{l})+2p(l+1)\ln(n) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≥2​ln⁡(1/δ)+2​ln⁡(Sh​(𝒩​(𝔊);n)).absent21𝛿2Sh  𝒩𝔊𝑛\displaystyle\geq 2\ln(1/\delta)+2\ln(\textup{Sh}(\mathcal{N}(\mathfrak{G});n)). |  |

The result follows by plugging this into [Lemma 4.1](#S4.Thmtheorem1 "Lemma 4.1. ‣ 4 Limitations of depth ‣ Benefits of depth in neural networks").

### A.4 Deferred proofs from [Section 5](#S5 "5 Bibliographic notes and open problems ‣ Benefits of depth in neural networks")

###### Proof A.14.

(of [Proposition 5.1](#S5.Thmtheorem1 "Proposition 5.1. ‣ 5 Bibliographic notes and open problems ‣ Benefits of depth in neural networks"))
Immediate from [Lemma 3.17](#S3.Thmtheorem17 "Lemma 3.17. ‣ 3.3 Many layers, many oscillations ‣ 3 Benefits of depth ‣ Benefits of depth in neural networks").
